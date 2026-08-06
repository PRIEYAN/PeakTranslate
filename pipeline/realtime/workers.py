"""Pipeline stage workers: STT, MT, TTS, playback.

Each worker owns one model instance (loaded once, resident for the process
lifetime) and one loop: pop its input queue, process, push to the next
queue. Coupling is only through queues + `gpu_lock`
(docs/05-vad-realtime-integration.md §6).

Two robustness rules apply to every worker below (learned the hard way):

1. Startup and every per-item step are wrapped so failures are logged with
   a full traceback (`log.exception`) instead of silently killing the
   daemon thread. A worker that dies silently otherwise looks identical to
   one that is just slow, and stalls every stage downstream of it forever.
2. Queue reads use a short timeout instead of blocking indefinitely, and
   also check `stop_event`. This means shutdown always completes even if
   an upstream worker crashed before it could forward the `STOP` sentinel.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from .messages import STOP, Sentence, Translation, Utterance, WavJob

log = logging.getLogger("peaktranslation.realtime")

_GET_TIMEOUT_S = 0.5

# transformers' `from_pretrained` monkeypatches process-global state while it
# loads (`PreTrainedModel.tie_weights` / `init_weights`), so two workers
# loading at the same time can swallow each other's tie call and leave tied
# weights unmaterialized on the meta device — which crashes on `.to("cuda")`
# with "Cannot copy out of meta tensor; no data!", in whichever worker lost
# the race that run. Loading is a one-off startup cost, so serialize it; only
# inference needs to stay concurrent (that's what `gpu_lock` is for).
# See docs/06-debugging-meta-tensor-load-race.md.
MODEL_LOAD_LOCK = threading.Lock()


def _assert_materialized(model, what: str) -> None:
    unmaterialized = [n for n, t in model.named_parameters() if t.is_meta]
    if unmaterialized:
        raise RuntimeError(
            f"{what} load left parameters on the meta device: {unmaterialized}. "
            "See docs/06-debugging-meta-tensor-load-race.md."
        )


def stt_worker(
    *,
    model_dir: Path,
    audio_queue: "queue.Queue",
    transcript_queue: "queue.Queue",
    gpu_lock: threading.Lock,
    stop_event: threading.Event,
    language: str = "en",
    beam_size: int = 1,
) -> None:
    try:
        import sys

        stt_src = Path(__file__).resolve().parents[2] / "stt" / "src"
        if str(stt_src) not in sys.path:
            sys.path.insert(0, str(stt_src))
        from runtime.messages import AudioChunk  # noqa: E402
        from runtime.whisper_pytorch import WhisperPytorchSTT  # noqa: E402

        log.info("Loading STT (CUDA, required) from %s ...", model_dir)
        engine = WhisperPytorchSTT(
            model_dir,
            device="cuda",
            require_cuda=True,
            allow_cpu_fallback=False,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            load_lock=MODEL_LOAD_LOCK,
        )
        engine.warmup()
        log.info("STT ready.")
    except Exception:
        log.exception("STT worker failed to start — pipeline cannot continue.")
        stop_event.set()
        transcript_queue.put(STOP)
        return

    while not stop_event.is_set():
        try:
            item = audio_queue.get(timeout=_GET_TIMEOUT_S)
        except queue.Empty:
            continue
        if item is STOP:
            transcript_queue.put(STOP)
            break
        utt: Utterance = item

        try:
            with gpu_lock:
                t0 = time.perf_counter()
                tr = engine.transcribe(
                    AudioChunk(pcm=utt.pcm, sample_rate=utt.sample_rate, session_id=utt.utt_id),
                    language=language,
                )
                t_stt = (time.perf_counter() - t0) * 1000
        except Exception:
            log.exception("[%s] STT failed on this utterance; skipping.", utt.utt_id)
            continue

        if not tr.text.strip():
            log.info("[%s] STT (%.0f ms): <non-speech, dropped>", utt.utt_id, t_stt)
            continue

        log.info("[%s] STT (%.0f ms): %r", utt.utt_id, t_stt, tr.text)
        transcript_queue.put(
            Sentence(
                utt_id=utt.utt_id,
                text=tr.text.strip(),
                src_lang=language,
                t_captured=utt.t_captured,
                t_stt_done=time.perf_counter(),
            )
        )

    log.info("STT worker stopped.")


def mt_worker(
    *,
    model_dir: Path,
    fallback_model_dir: Optional[Path],
    tgt_lang: str,
    transcript_queue: "queue.Queue",
    translation_queue: "queue.Queue",
    gpu_lock: threading.Lock,
    stop_event: threading.Event,
    max_new_tokens: int = 96,
    num_beams: int = 2,
    allow_cpu_fallback: bool = True,
) -> None:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        log.info("Loading MT (CUDA if available) from %s ...", model_dir)
        use_cuda = torch.cuda.is_available()
        with MODEL_LOAD_LOCK:
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
            _assert_materialized(model, "MT")
            if use_cuda:
                model = model.half().to("cuda")
        model.eval()
        log.info("MT ready. (cuda=%s)", use_cuda)
    except Exception:
        log.exception("MT worker failed to start — pipeline cannot continue.")
        stop_event.set()
        translation_queue.put(STOP)
        return

    fallback_tok = None
    fallback_model = None

    def load_cpu_fallback() -> None:
        nonlocal fallback_tok, fallback_model
        if fallback_model is not None or fallback_model_dir is None or not fallback_model_dir.exists():
            return
        log.warning("Loading CPU fallback MT model: %s", fallback_model_dir)
        with MODEL_LOAD_LOCK:
            fallback_tok = AutoTokenizer.from_pretrained(str(fallback_model_dir))
            fallback_model = AutoModelForSeq2SeqLM.from_pretrained(str(fallback_model_dir))
            _assert_materialized(fallback_model, "MT CPU fallback")
        fallback_model.eval()

    def translate_gpu(text: str) -> str:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=96)
        if use_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    def translate_cpu_fallback(text: str) -> str:
        load_cpu_fallback()
        if fallback_model is None:
            raise RuntimeError("No CPU fallback MT model available")
        inputs = fallback_tok(text, return_tensors="pt", truncation=True, max_length=96)
        with torch.inference_mode():
            out = fallback_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
        return fallback_tok.decode(out[0], skip_special_tokens=True).strip()

    while not stop_event.is_set():
        try:
            item = transcript_queue.get(timeout=_GET_TIMEOUT_S)
        except queue.Empty:
            continue
        if item is STOP:
            translation_queue.put(STOP)
            break
        sent: Sentence = item

        t0 = time.perf_counter()
        try:
            try:
                with gpu_lock:
                    text = translate_gpu(sent.text)
            except torch.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
                torch.cuda.empty_cache()
                if not allow_cpu_fallback:
                    log.error("[%s] MT CUDA OOM and CPU fallback disabled; dropping.", sent.utt_id)
                    continue
                log.warning("[%s] MT CUDA OOM; retrying on CPU fallback.", sent.utt_id)
                text = translate_cpu_fallback(sent.text)
        except Exception:
            log.exception("[%s] MT failed on this sentence; skipping.", sent.utt_id)
            continue
        t_mt = (time.perf_counter() - t0) * 1000

        log.info("[%s] MT (%.0f ms): %r", sent.utt_id, t_mt, text)
        translation_queue.put(
            Translation(
                utt_id=sent.utt_id,
                text=text,
                tgt_lang=tgt_lang,
                t_captured=sent.t_captured,
                t_mt_done=time.perf_counter(),
            )
        )

    log.info("MT worker stopped.")


def _piper_synth(text: str, voice_onnx: Path, out_wav: Path) -> None:
    import subprocess
    import wave

    import numpy as np

    try:
        from piper import PiperVoice

        voice = PiperVoice.load(str(voice_onnx))
        if hasattr(voice, "synthesize_wav"):
            with wave.open(str(out_wav), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            return
        with wave.open(str(out_wav), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            sr = getattr(getattr(voice, "config", None), "sample_rate", 22050)
            wav_file.setframerate(sr)
            for chunk in voice.synthesize(text):
                audio = getattr(chunk, "audio_int16_bytes", None) or getattr(chunk, "audio_bytes", None)
                if audio is None and hasattr(chunk, "audio_int16"):
                    audio = np.asarray(chunk.audio_int16, dtype=np.int16).tobytes()
                if audio:
                    wav_file.writeframes(audio)
        return
    except Exception as e:  # noqa: BLE001
        log.warning("piper python API failed (%s); trying CLI...", e)

    cmd = ["piper", "--model", str(voice_onnx), "--output_file", str(out_wav)]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def tts_worker(
    *,
    voice_onnx: Path,
    spill_dir: Path,
    translation_queue: "queue.Queue",
    wav_queue: "queue.Queue",
    stop_event: threading.Event,
    sample_rate: int = 22050,
) -> None:
    try:
        spill_dir.mkdir(parents=True, exist_ok=True)
        log.info("TTS ready (voice=%s).", voice_onnx)
    except Exception:
        log.exception("TTS worker failed to start — pipeline cannot continue.")
        stop_event.set()
        wav_queue.put(STOP)
        return

    while not stop_event.is_set():
        try:
            item = translation_queue.get(timeout=_GET_TIMEOUT_S)
        except queue.Empty:
            continue
        if item is STOP:
            wav_queue.put(STOP)
            break
        tr: Translation = item

        out_wav = spill_dir / f"{tr.utt_id}.wav"
        t0 = time.perf_counter()
        try:
            _piper_synth(tr.text, voice_onnx, out_wav)
        except Exception:
            log.exception("[%s] TTS failed; dropping.", tr.utt_id)
            continue
        t_tts = (time.perf_counter() - t0) * 1000

        log.info("[%s] TTS (%.0f ms): %s", tr.utt_id, t_tts, out_wav)
        wav_queue.put(
            WavJob(
                utt_id=tr.utt_id,
                wav_path=out_wav,
                sample_rate=sample_rate,
                t_captured=tr.t_captured,
                t_tts_done=time.perf_counter(),
            )
        )

    log.info("TTS worker stopped.")


def playback_worker(
    *,
    wav_queue: "queue.Queue",
    stop_event: threading.Event,
    keep_wavs: bool = False,
    log_latency: bool = True,
) -> None:
    from .audio_out import play_wav_blocking

    log.info("Playback ready.")
    while not stop_event.is_set():
        try:
            item = wav_queue.get(timeout=_GET_TIMEOUT_S)
        except queue.Empty:
            continue
        if item is STOP:
            break
        job: WavJob = item

        if log_latency:
            total_ms = (time.perf_counter() - job.t_captured) * 1000
            log.info("[%s] VAD-close -> audio-out: %.0f ms", job.utt_id, total_ms)

        try:
            play_wav_blocking(job.wav_path)
        except Exception:
            log.exception("[%s] Playback failed.", job.utt_id)

        if not keep_wavs:
            try:
                job.wav_path.unlink(missing_ok=True)
            except OSError:
                pass

    log.info("Playback worker stopped.")
