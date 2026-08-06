#!/usr/bin/env python3
"""
Real-time VAD speech-to-speech pipeline entrypoint (PC).

Always-listening English mic -> Whisper (CUDA) -> MarianMT (CUDA, CPU
fallback allowed) -> Piper (CPU) -> serial speaker playback.

Design: docs/05-vad-realtime-integration.md
Build order (do these in sequence when first bringing this up):
  1. --stage capture    dump VAD-segmented utterances to spill/, no models
  2. --stage stt        add live transcripts
  3. --stage mt         add translations
  4. --stage full       (default) complete loop with playback

Requires:
  pip: sounddevice webrtcvad piper-tts onnxruntime soundfile librosa
       transformers torch pyyaml
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
from pipeline.realtime.messages import STOP  # noqa: E402
from pipeline.realtime.workers import (  # noqa: E402
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

    stt_model_dir = resolve(ROOT, cfg["stt"]["model_dir"])
    mt_model_dir = resolve(ROOT, cfg["translate"]["model_dir"])
    mt_fallback_dir = resolve(ROOT, cfg["translate"].get("fallback_model_dir"))
    voice_onnx = resolve(ROOT, cfg["tts"]["voice"])
    spill_dir = resolve(ROOT, cfg["runtime"]["spill_dir"])

    if args.stage in ("stt", "mt", "full") and not stt_model_dir.exists():
        print(f"Missing STT model: {stt_model_dir}", file=sys.stderr)
        return 1
    if args.stage in ("mt", "full") and not mt_model_dir.exists():
        print(f"Missing MT model: {mt_model_dir}", file=sys.stderr)
        return 1
    if args.stage == "full" and not voice_onnx.exists():
        print(
            f"Missing Piper voice: {voice_onnx}\nRun: bash tts/scripts/download_voices.sh",
            file=sys.stderr,
        )
        return 1

    q_cfg = cfg["queues"]
    audio_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["audio_queue"]["maxsize"])
    transcript_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["transcript_queue"]["maxsize"])
    translation_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["translation_queue"]["maxsize"])
    wav_queue: "queue.Queue" = queue.Queue(maxsize=q_cfg["wav_queue"]["maxsize"])

    gpu_lock = threading.Lock()
    stop_event = threading.Event()

    drop_count = {"n": 0}

    def on_drop(utt_id: str) -> None:
        drop_count["n"] += 1
        log.warning("[%s] Q0 audio_queue full — dropped oldest (total drops: %d)", utt_id, drop_count["n"])

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
            t_mt = threading.Thread(
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
            t_mt.start()
            threads.append(t_mt)

            if args.stage == "mt":

                def print_loop() -> None:
                    while not stop_event.is_set():
                        try:
                            item = translation_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        if item is STOP:
                            break
                        log.info("[%s] TRANSLATION (%s): %r", item.utt_id, item.tgt_lang, item.text)

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
