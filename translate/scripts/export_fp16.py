#!/usr/bin/env python3
"""Export fine-tuned Marian to FP16 for Jetson GPU path."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

TR_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=TR_ROOT / "models" / "finetuned" / "en-hi-v1",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=TR_ROOT / "models" / "export" / "en-hi-v1-fp16",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"Missing {args.src}", file=sys.stderr)
        return 1

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading {args.src}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.src))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.src))
    model = model.half()
    args.dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.dst))
    tokenizer.save_pretrained(str(args.dst))
    for name in ("MODEL_CARD.md", "metrics.json"):
        src = args.src / name
        if src.exists():
            shutil.copy2(src, args.dst / name)
    total = sum(p.stat().st_size for p in args.dst.rglob("*") if p.is_file())
    print(f"Exported FP16 → {args.dst} ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
