#!/usr/bin/env python3
"""
Real-time VAD speech-to-speech pipeline entrypoint (PC).

Always-listening English mic -> Whisper (CUDA) -> a text stage -> Piper
(CPU) -> serial speaker playback. The text stage is chosen by `--mode` /
`mode` in the config and is one of:

  translate  (default)  MarianMT (CUDA, CPU fallback allowed), en -> hi
  reason                 Gemma Jarvis assistant (CUDA, 4-bit), multilingual
                          see docs/reasoningModel/01-gemma-reasoning-mode.md
                          barge_in default OFF (mute mic while speaking —
                          safer on USB mics / open speakers; see doc §19).

Design: docs/05-vad-realtime-integration.md
Build order (do these in sequence when first bringing this up):
  1. --stage capture    dump VAD-segmented utterances to spill/, no models
  2. --stage stt        add live transcripts
  3. --stage mt         add the text-stage output (translation or reply)
  4. --stage full       (default) complete loop with playback

Requires:
  pip: sounddevice webrtcvad piper-tts onnxruntime soundfile librosa
       transformers torch pyyaml
  Reason mode also needs: bitsandbytes (reason/requirements.txt)
  System: PortAudio (for sounddevice), aplay/paplay/ffplay (for playback)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline._bootstrap import ensure_venv_on_path  # noqa: E402

ensure_venv_on_path()

import argparse  # noqa: E402
import logging  # noqa: E402
import queue  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipeline.realtime.capture import AudioCapture  # noqa: E402
from pipeline.realtime.messages import STOP, Abort  # noqa: E402
from pipeline.realtime.workers import (  # noqa: E402
    MODEL_LOAD_LOCK,
    mt_worker,
    playback_worker,
    stt_worker,
    tts_worker,
)

log = logging.getLogger("peaktranslation.realtime")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: Path, rel: str | None) -> Path | None:
    if rel is None:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time PeakTranslation pipeline (PC)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "pipeline" / "config" / "realtime.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=["capture", "stt", "mt", "full"],
        default="full",
        help="Bring-up stage: how far down the pipeline to run (see module docstring).",
    )
    parser.add_argument(
        "--mode",
        choices=["translate", "reason"],
        default=None,
        help="Text stage between STT and TTS. Overrides `mode` in the config.",
    )
    parser.add_argument(
        "--reason-profile",
        default=None,
        help="Override reason.profile (see reason/configs/profiles.yaml).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Input device index or name substring (e.g. 'USB'). Overrides "
        "capture.device from the config. See --list-devices.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print all audio devices PortAudio can see, with index and "
        "channel counts, then exit. Use this to find your USB mic's index.",
    )
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    load_dotenv(ROOT / ".env")
    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.get("runtime", {}).get("log_level", "INFO")),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mode = args.mode or cfg.get("mode", "translate")
    if mode not in ("translate", "reason"):
        print(f"Unknown mode: {mode!r} (use 'translate' or 'reason')", file=sys.stderr)
        return 1
    # Barge-in only makes sense (and is only wired up) in reason mode — see
    # docs/reasoningModel/01-gemma-reasoning-mode.md §19. Requires capture to
    # stay unmuted while replying, which trades away the echo protection
    # `mute_capture_while_replying` gives on open speakers.
    # Default OFF in config — enable for interrupt-during-playback (reason mode).
    barge_in = mode == "reason" and bool(cfg.get("reason", {}).get("barge_in", False))
    reset_memory_on_barge_in = (
        barge_in and bool(cfg.get("reason", {}).get("reset_memory_on_barge_in", True))
    )
    mute_while_replying = mode == "reason" and (
        not barge_in and bool(cfg.get("reason", {}).get("mute_capture_while_replying", True))
    )

    stt_model_dir = resolve(ROOT, cfg["stt"]["model_dir"])
    # Modes are mutually exclusive (docs/reasoningModel/01-gemma-reasoning-mode.md
    # §5): reason mode never loads Marian, translate mode never loads Gemma.
    reason_voices: dict[str, Path] = {}
    if mode == "translate":
        mt_model_dir = resolve(ROOT, cfg["translate"]["model_dir"])
        mt_fallback_dir = resolve(ROOT, cfg["translate"].get("fallback_model_dir"))
        voice_onnx = resolve(ROOT, cfg["tts"]["voice"])
    else:
        mt_model_dir = None
        mt_fallback_dir = None
        r_cfg_early = cfg["reason"]
        voice_onnx = resolve(ROOT, r_cfg_early["voice"])
        for lang_key, rel in (r_cfg_early.get("voices") or {}).items():
            p = resolve(ROOT, rel)
            if p is not None:
                reason_voices[lang_key] = p
        reason_voices.setdefault("default", voice_onnx)
        reason_voices.setdefault("en", voice_onnx)
    spill_dir = resolve(ROOT, cfg["runtime"]["spill_dir"])

    if args.stage in ("stt", "mt", "full") and not stt_model_dir.exists():
        print(f"Missing STT model: {stt_model_dir}", file=sys.stderr)
        return 1
    if mode == "translate" and args.stage in ("mt", "full") and not mt_model_dir.exists():
        print(f"Missing MT model: {mt_model_dir}", file=sys.stderr)
        return 1
    if args.stage == "full":
        missing = [str(p) for p in ([voice_onnx] + list(reason_voices.values())) if not p.exists()]
        missing_u = list(dict.fromkeys(missing))
        if missing_u:
            print(
                "Missing Piper voice(s):\n  "
                + "\n  ".join(missing_u)
                + "\nRun: bash tts/scripts/download_voices.sh",
                file=sys.stderr,
            )
            return 1
    if mode == "reason" and args.stage in ("mt", "full"):
        # reason.model_id is a Hub id resolved at load time, not a local
        # path — it can't be preflight-checked here. Log it and let the
        # loader raise the real HF error (e.g. a 401 if the Gemma licence
        # hasn't been accepted) instead of a generic missing-file message.
        r_cfg = cfg["reason"]
        profile_id = args.reason_profile or r_cfg["profile"]
        log.info(
            "Reason mode: profile=%s barge_in=%s mute_while_replying=%s",
            profile_id,
            barge_in,
            mute_while_replying,
        )

    q_cfg = cfg["queues"]
    audio_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["audio_queue"]["maxsize"])
    transcript_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["transcript_queue"]["maxsize"])
    translation_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["translation_queue"]["maxsize"])
    wav_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["wav_queue"]["maxsize"])

    gpu_lock = threading.Lock()
    stop_event = threading.Event()
    # Set by the reasoning worker while the assistant is speaking, cleared by
    # playback once sound actually stops (or by reasoning_worker itself on a
    # barge-in/failure). Translate mode never touches it, so capture behaves
    # exactly as before. See docs/reasoningModel/01-gemma-reasoning-mode.md §14.
    speaking = threading.Event()
    speaking_started_at = [0.0]
    barge_in_grace_s = (
        float(cfg.get("reason", {}).get("barge_in_grace_ms", 900)) / 1000.0
        if barge_in
        else 0.0
    )
    # Set the instant a barge-in is detected; cleared by reasoning_worker at
    # the start of the next utterance. Only meaningful when barge_in is on —
    # see doc §19.
    interrupt_event = threading.Event()
    # Shared reason-mode memory (history + Jarvis sticky mode) for barge-in reset.
    reason_memory: dict = {}

    drop_count = {"n": 0}

    def on_drop(utt_id: str) -> None:
        drop_count["n"] += 1
        log.warning("[%s] Q0 audio_queue full — dropped oldest (total drops: %d)", utt_id, drop_count["n"])

    def reset_reason_memory(where: str) -> None:
        hist = reason_memory.get("history")
        sess = reason_memory.get("session")
        epoch = reason_memory.get("epoch_ref")
        if hist is None or sess is None:
            return
        hist.clear()
        sess.clear_mode()
        if epoch is not None:
            epoch[0] = time.monotonic()
        log.info("Reason memory reset (%s).", where)

    def on_speech_start() -> None:
        # User spoke while assistant audio is playing (or generating).
        if speaking.is_set():
            # Ignore speaker bleed right after output starts — otherwise
            # "Understood" from the speakers triggers instant memory wipe.
            if (
                barge_in_grace_s > 0
                and time.monotonic() - speaking_started_at[0] < barge_in_grace_s
            ):
                return
            interrupt_event.set()
            if reset_memory_on_barge_in:
                reset_reason_memory("barge-in")

    cap_cfg = cfg["capture"]
    device = args.device if args.device is not None else cap_cfg.get("device")
    # sounddevice accepts a bare int index or a numeric string ("3") or a
    # name substring ("USB"); coerce numeric-looking strings to int so both
    # `--device 3` and `--device USB` work.
    if isinstance(device, str) and device.strip().lstrip("-").isdigit():
        device = int(device)
    capture = AudioCapture(
        audio_queue=audio_queue,
        sample_rate=cap_cfg["sample_rate"],
        frame_ms=cap_cfg["frame_ms"],
        device=device,
        vad_kwargs=cap_cfg["vad"],
        on_drop=on_drop,
        # Muting and barge-in are mutually exclusive (doc §14 vs §19).
        speaking=speaking if mute_while_replying else None,
        on_speech_start=on_speech_start if barge_in else None,
        min_peak_abs=int(cap_cfg.get("min_peak_abs") or 0),
    )

    if args.stage in ("stt", "mt", "full"):
        # transformers' lazy module loader is not thread-safe: if the stt
        # and mt worker threads both trigger their first `from transformers
        # import ...` at nearly the same moment, one of them intermittently
        # dies with "ImportError: cannot import name X from transformers"
        # even though the class exists. Force the import to happen once,
        # single-threaded, right here, before any worker thread starts.
        import sys as _sys

        _stt_src = ROOT / "stt" / "src"
        if str(_stt_src) not in _sys.path:
            _sys.path.insert(0, str(_stt_src))
        from transformers import (  # noqa: F401
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        if mode == "reason":
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: F401

    threads: list[threading.Thread] = []

    if args.stage == "capture":
        # Bring-up mode: drain Q0 to spill/, no models loaded.
        import soundfile as sf

        spill_dir.mkdir(parents=True, exist_ok=True)

        def dump_loop() -> None:
            while not stop_event.is_set():
                try:
                    item = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is STOP:
                    break
                out = spill_dir / f"{item.utt_id}.wav"
                import numpy as np

                pcm = np.frombuffer(item.pcm, dtype="<i2")
                sf.write(str(out), pcm, item.sample_rate, subtype="PCM_16")
                log.info("[%s] utterance %.2fs -> %s", item.utt_id, item.duration_s, out)

        t = threading.Thread(target=dump_loop, name="dump", daemon=True)
        t.start()
        threads.append(t)

    else:
        t_stt = threading.Thread(
            target=stt_worker,
            name="stt",
            daemon=True,
            kwargs=dict(
                model_dir=stt_model_dir,
                audio_queue=audio_queue,
                transcript_queue=transcript_queue,
                gpu_lock=gpu_lock,
                stop_event=stop_event,
                language=cfg["stt"]["language"],
                beam_size=cfg["stt"]["beam_size"],
                drop_noise_transcripts=bool(cfg["stt"].get("drop_noise_transcripts", True)),
            ),
        )
        t_stt.start()
        threads.append(t_stt)

        if args.stage == "stt":

            def print_loop() -> None:
                while not stop_event.is_set():
                    try:
                        item = transcript_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if item is STOP:
                        break
                    log.info("[%s] TRANSCRIPT: %r", item.utt_id, item.text)

            t = threading.Thread(target=print_loop, name="print", daemon=True)
            t.start()
            threads.append(t)

        else:
            if mode == "reason":
                from pipeline.realtime.reasoning import reasoning_worker
                from reason.src.runtime import ConversationHistory, build_engine, load_system_prompt
                from reason.src.runtime.registry import resolve_profile
                from reason.src.runtime.session import JarvisSession
                from reason.src.runtime.streaming import SentenceAssembler

                r_cfg = cfg["reason"]
                profile_id = args.reason_profile or r_cfg["profile"]
                profile = resolve_profile(profile_id)
                engine = build_engine(profile_id, load_lock=MODEL_LOAD_LOCK)
                reason_history = ConversationHistory(max_turns=profile.get("history_turns", 4))
                jarvis_session = JarvisSession()
                history_epoch_ref = [time.monotonic()]
                reason_memory["history"] = reason_history
                reason_memory["session"] = jarvis_session
                reason_memory["epoch_ref"] = history_epoch_ref
                t_text = threading.Thread(
                    target=reasoning_worker,
                    name="reason",
                    daemon=True,
                    kwargs=dict(
                        engine=engine,
                        history=reason_history,
                        session=jarvis_session,
                        history_epoch_ref=history_epoch_ref,
                        assembler_factory=SentenceAssembler,
                        system_prompt=load_system_prompt(profile_id),
                        transcript_queue=transcript_queue,
                        reply_queue=translation_queue,
                        gpu_lock=gpu_lock,
                        stop_event=stop_event,
                        out_lang=profile.get("out_lang", "en"),
                        speaking=speaking,
                        interrupt_event=interrupt_event if barge_in else None,
                        history_ttl_s=float(profile.get("history_ttl_s", 30)),
                        reset_memory_on_barge_in=reset_memory_on_barge_in,
                    ),
                )
            else:
                t_text = threading.Thread(
                    target=mt_worker,
                    name="mt",
                    daemon=True,
                    kwargs=dict(
                        model_dir=mt_model_dir,
                        fallback_model_dir=mt_fallback_dir,
                        tgt_lang=cfg["translate"]["tgt_lang"],
                        transcript_queue=transcript_queue,
                        translation_queue=translation_queue,
                        gpu_lock=gpu_lock,
                        stop_event=stop_event,
                        max_new_tokens=cfg["translate"]["max_new_tokens"],
                        num_beams=cfg["translate"]["num_beams"],
                        allow_cpu_fallback=cfg["translate"]["allow_cpu_fallback"],
                    ),
                )
            t_text.start()
            threads.append(t_text)

            if args.stage == "mt":

                def print_loop() -> None:
                    while not stop_event.is_set():
                        try:
                            item = translation_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        if item is STOP:
                            break
                        if isinstance(item, Abort):
                            log.info("[%s] Reply aborted by barge-in.", item.utt_id)
                            continue
                        if not item.text.strip():
                            continue  # reason mode's end-of-utterance marker
                        # Translation has `tgt_lang`; Reply has `out_lang`.
                        lang = getattr(item, "tgt_lang", None) or getattr(item, "out_lang", "?")
                        log.info("[%s] TEXT (%s): %r", item.utt_id, lang, item.text)

                t = threading.Thread(target=print_loop, name="print", daemon=True)
                t.start()
                threads.append(t)

            else:  # full
                t_tts = threading.Thread(
                    target=tts_worker,
                    name="tts",
                    daemon=True,
                    kwargs=dict(
                        voice_onnx=voice_onnx,
                        spill_dir=spill_dir,
                        translation_queue=translation_queue,
                        wav_queue=wav_queue,
                        stop_event=stop_event,
                        sample_rate=cfg["tts"]["sample_rate"],
                        interrupt_event=interrupt_event if barge_in else None,
                        voices=reason_voices if mode == "reason" else None,
                    ),
                )
                t_tts.start()
                threads.append(t_tts)

                t_play = threading.Thread(
                    target=playback_worker,
                    name="playback",
                    daemon=True,
                    kwargs=dict(
                        wav_queue=wav_queue,
                        stop_event=stop_event,
                        keep_wavs=cfg["playback"]["keep_wavs"],
                        log_latency=cfg["runtime"]["log_latency"],
                        speaking=speaking,
                        interrupt_event=interrupt_event if barge_in else None,
                        speaking_started_at=speaking_started_at if barge_in else None,
                    ),
                )
                t_play.start()
                threads.append(t_play)

    log.info("Starting mic capture (stage=%s). Press Ctrl+C to stop.", args.stage)
    capture.start()

    shutdown = threading.Event()

    def handle_sigint(signum, frame) -> None:  # noqa: ANN001
        log.info("Shutdown requested...")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not shutdown.is_set():
            time.sleep(0.2)
    finally:
        log.info("Stopping capture...")
        capture.stop()  # pushes STOP into audio_queue, cascades down the pipeline
        stop_event.set()
        for t in threads:
            t.join(timeout=10.0)
        log.info("Shutdown complete. Total Q0 drops: %d", drop_count["n"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
