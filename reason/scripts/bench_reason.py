#!/usr/bin/env python3
"""VRAM footprint + tokens/sec + first-token latency for a profile.

Run this before wiring anything into the pipeline — bring-up step 3 in
docs/reasoningModel/01-gemma-reasoning-mode.md §17 / §5. If the reported
VRAM plus ~0.5 GB for Whisper + CUDA context doesn't leave headroom, pick a
smaller model in configs/profiles.yaml before continuing.
"""
from __future__ import annotations

import sys
from pathlib import Path

REASON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REASON_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from pipeline._bootstrap import ensure_venv_on_path  # noqa: E402

# `venv/bin/python` here doesn't self-detect as a venv, so third-party
# imports (dotenv, torch, transformers, ...) 404 unless this runs first —
# same fix pipeline/run_realtime.py applies.
ensure_venv_on_path()

import argparse  # noqa: E402
import time  # noqa: E402

from dotenv import load_dotenv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="Profile id in configs/profiles.yaml.")
    parser.add_argument("--prompt", default="Tell me one interesting fact about the ocean.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    import torch

    from reason.src.runtime import Prompt, build_engine

    if not torch.cuda.is_available():
        print("CUDA not available — bench_reason.py requires a GPU.", file=sys.stderr)
        return 1

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    engine = build_engine(args.profile)
    load_s = time.perf_counter() - t_load0

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Load time:        {load_s:.1f} s")
    print(f"VRAM allocated:   {allocated:.2f} GB")
    print(f"VRAM reserved:    {reserved:.2f} GB")

    engine.warmup()

    prompt = Prompt(user_text=args.prompt)
    t0 = time.perf_counter()
    n_chars, n_deltas, first_token_ms = 0, 0, None
    for delta in engine.stream_reply(prompt):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t0) * 1000
        n_chars += len(delta)
        n_deltas += 1
    total_s = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9
    approx_tokens = max(1, n_chars // 4)  # rough chars-per-token estimate
    print(f"\nPrompt: {args.prompt!r}")
    print(f"First token:      {first_token_ms:.0f} ms")
    print(f"Total generation: {total_s:.2f} s ({n_deltas} deltas, ~{approx_tokens} tokens)")
    print(f"Approx tokens/s:  {approx_tokens / total_s:.1f}")
    print(f"Peak VRAM:        {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
