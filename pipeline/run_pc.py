#!/usr/bin/env python3
"""
PC end-to-end PeakTranslation pipeline (Jetson draft deferred).

Modes:
  --wav path.wav     STT (CUDA) → MT en→hi (CUDA) → Piper Hindi → play/save
  --text "Hello"     skip STT; MT → Piper → play/save

Requires:
  - stt/models/export/en-hi-base-v1-fp16 (or finetuned)
  - translate/models/export/en-hi-v1-fp16
  - tts/models/export/hi_official_v1/voice.onnx
  - pip: piper-tts onnxruntime soundfile librosa transformers torch
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stt" / "src"))


def play_wav(path: Path) -> None:
    # Try aplay, then paplay, then ffplay
    for cmd in (
        ["aplay", str(path)],
        ["paplay", str(path)],
        ["ffplay", "-nodisp", "-autoexit", str(path)],
    ):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"(no audio player found — wav saved at {path})")


def load_stt(model_dir: Path):
    from runtime.whisper_pytorch import WhisperPytorchSTT
    from runtime.messages import AudioChunk

    engine = WhisperPytorchSTT(
        model_dir,
        device="cuda",
        require_cuda=True,
        allow_cpu_fallback=False,
        language="en",
        task="transcribe",
        beam_size=1,
    )
    return engine, AudioChunk


def load_mt(model_dir: Path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    if torch.cuda.is_available():
        model = model.half().to("cuda")
    model.eval()
    return tok, model


def mt_translate(tok, model, text: str) -> str:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=96)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, num_beams=2)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def piper_synth(text: str, voice_onnx: Path, out_wav: Path) -> None:
    try:
        import wave

        from piper import PiperVoice

        voice = PiperVoice.load(str(voice_onnx))
        # piper-tts API variants
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
    except Exception as e:
        print(f"piper python API failed ({e}); trying CLI…")

    cmd = [
        "piper",
        "--model",
        str(voice_onnx),
        "--output_file",
        str(out_wav),
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="PC PeakTranslation pipeline")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", type=Path, help="English speech WAV → translate to Hindi audio")
    g.add_argument("--text", type=str, help="English text → translate to Hindi audio (skip STT)")
    parser.add_argument(
        "--stt-model",
        type=Path,
        default=ROOT / "stt" / "models" / "export" / "en-hi-base-v1-fp16",
    )
    parser.add_argument(
        "--mt-model",
        type=Path,
        default=ROOT / "translate" / "models" / "export" / "en-hi-v1-fp16",
    )
    parser.add_argument(
        "--voice",
        type=Path,
        default=ROOT / "tts" / "models" / "export" / "hi_official_v1" / "voice.onnx",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output wav path")
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not args.mt_model.exists():
        print(f"Missing MT model: {args.mt_model}", file=sys.stderr)
        return 1
    if not args.voice.exists():
        print(
            f"Missing Piper voice: {args.voice}\n"
            f"Run: bash tts/scripts/download_voices.sh",
            file=sys.stderr,
        )
        return 1

    english_text = ""
    t0 = time.perf_counter()

    if args.wav is not None:
        if not torch.cuda.is_available():
            print("STT requires CUDA on PC.", file=sys.stderr)
            return 1
        if not args.stt_model.exists():
            # fall back to finetuned
            alt = ROOT / "stt" / "models" / "finetuned" / "en-hi-base-v1"
            if alt.exists():
                args.stt_model = alt
            else:
                print(f"Missing STT model: {args.stt_model}", file=sys.stderr)
                return 1

        audio, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        pcm = (np.clip(audio, -1, 1) * 32767.0).astype(np.int16).tobytes()

        print("Loading STT (CUDA)…")
        stt, AudioChunk = load_stt(args.stt_model)
        stt.warmup()
        t_stt0 = time.perf_counter()
        tr = stt.transcribe(AudioChunk(pcm=pcm, sample_rate=sr), language="en")
        t_stt = (time.perf_counter() - t_stt0) * 1000
        english_text = tr.text.strip()
        print(f"STT ({t_stt:.0f} ms): {english_text!r}")
        if not english_text:
            print("Empty transcript — nothing to translate.")
            return 0
    else:
        english_text = args.text.strip()
        print(f"TEXT: {english_text!r}")

    print("Loading MT (CUDA if available)…")
    tok, mt = load_mt(args.mt_model)
    t_mt0 = time.perf_counter()
    hindi = mt_translate(tok, mt, english_text)
    t_mt = (time.perf_counter() - t_mt0) * 1000
    print(f"MT  ({t_mt:.0f} ms): {hindi!r}")

    out = args.out or Path(tempfile.gettempdir()) / "peaktranslation_out.wav"
    print("Synthesizing with Piper…")
    t_tts0 = time.perf_counter()
    piper_synth(hindi, args.voice, out)
    t_tts = (time.perf_counter() - t_tts0) * 1000
    print(f"TTS ({t_tts:.0f} ms): {out}")

    if not args.no_play:
        play_wav(out)

    total = (time.perf_counter() - t0) * 1000
    print(f"TOTAL {total:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
