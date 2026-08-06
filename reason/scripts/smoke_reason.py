#!/usr/bin/env python3
"""One prompt in, reply out. No pipeline, no audio, no queues.

Confirms: model access/licence, chat template, and reply shape/length —
bring-up step 4 in docs/reasoningModel/01-gemma-reasoning-mode.md §17.
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
    parser.add_argument("--text", required=True, help="User prompt to send.")
    parser.add_argument("--profile", default=None, help="Profile id in configs/profiles.yaml.")
    parser.add_argument("--system", default=None, help="Override the profile's system prompt.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    from reason.src.runtime import Prompt, build_engine, load_system_prompt

    print(f"Loading profile {args.profile or '(default)'} ...")
    engine = build_engine(args.profile)
    engine.warmup()
    print("Ready.\n")

    system = args.system if args.system is not None else load_system_prompt(args.profile)
    prompt = Prompt(user_text=args.text, system=system)

    t0 = time.perf_counter()
    chunks: list[str] = []
    first_token_ms: float | None = None
    for delta in engine.stream_reply(prompt):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t0) * 1000
        chunks.append(delta)
        print(delta, end="", flush=True)
    total_ms = (time.perf_counter() - t0) * 1000
    print()

    reply = "".join(chunks).strip()
    n_tokens = len(reply.split())
    print(f"\n--- {n_tokens} words, first token {first_token_ms:.0f} ms, total {total_ms:.0f} ms ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
