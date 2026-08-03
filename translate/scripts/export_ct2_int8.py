#!/usr/bin/env python3
"""Optional: export CT2 int8 for Jetson CPU fallback."""
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
        default=TR_ROOT / "models" / "export" / "en-hi-v1-ct2-int8",
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"Missing {args.src}", file=sys.stderr)
        return 1

    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        print(
            "ctranslate2 not installed — skipping CT2 export.\n"
            "  pip install ctranslate2\n"
            "FP16 GPU export is still enough for primary path.",
            file=sys.stderr,
        )
        return 0

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.exists():
        shutil.rmtree(args.dst)
    print(f"Converting {args.src} → {args.dst} (int8)")
    TransformersConverter(str(args.src)).convert(
        str(args.dst),
        quantization="int8",
        force=True,
    )
    # copy tokenizer / spm files if converter did not
    for name in args.src.iterdir():
        if name.suffix in {".spm", ".json", ".model", ".txt"} or "tokenizer" in name.name:
            dest = args.dst / name.name
            if not dest.exists():
                shutil.copy2(name, dest)
    print("CT2 int8 export done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
