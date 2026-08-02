#!/usr/bin/env python3
"""Smoke-test STT on a WAV (CUDA only)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv

STT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STT_ROOT.parent
sys.path.insert(0, str(STT_ROOT / "src"))

from runtime import build_engine  # noqa: E402
from runtime.messages import AudioChunk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--profile", default="en_hi_base_v1")
    parser.add_argument("--language", default=None, help="en | hi | omit for profile default")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not torch.cuda.is_available():
        print("FAIL: CUDA required for STT smoke test", file=sys.stderr)
        return 1

    audio, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    pcm = (np.clip(audio, -1, 1) * 32767.0).astype(np.int16).tobytes()

    engine = build_engine(args.profile, model_path=args.model)
    engine.warmup()
    t0 = time.perf_counter()
    result = engine.transcribe(
        AudioChunk(pcm=pcm, sample_rate=sr),
        language=args.language,
    )
    dt = (time.perf_counter() - t0) * 1000
    print(f"text: {result.text!r}")
    print(f"is_speech: {result.is_speech}")
    print(f"lang: {result.lang}")
    print(f"latency_ms: {dt:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
