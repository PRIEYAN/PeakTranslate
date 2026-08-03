#!/usr/bin/env python3
"""
Prepare English→Hindi parallel TSV from Hugging Face.

Primary source: opus100 en-hi (public, parquet-friendly).
Fallback: cfilt/iitb-english-hindi if opus100 fails.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

TR_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TR_ROOT.parent


def write_tsv(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for src, tgt in pairs:
            src = src.replace("\t", " ").replace("\n", " ").strip()
            tgt = tgt.replace("\t", " ").replace("\n", " ").strip()
            if src and tgt:
                f.write(f"{src}\t{tgt}\n")


def from_opus100(token: str | None, max_n: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    print("Loading opus100 en-hi…")
    try:
        ds = load_dataset("opus100", "en-hi", split="train", token=token)
    except TypeError:
        ds = load_dataset("opus100", "en-hi", split="train")

    pairs: list[tuple[str, str]] = []
    for ex in ds:
        if max_n and len(pairs) >= max_n:
            break
        tr = ex.get("translation") or {}
        src = (tr.get("en") or "").strip()
        tgt = (tr.get("hi") or "").strip()
        if not src or not tgt:
            continue
        # light length filter
        if len(src.split()) > 80 or len(tgt.split()) > 80:
            continue
        ratio = len(src) / max(len(tgt), 1)
        if ratio < 0.3 or ratio > 3.0:
            continue
        pairs.append((src, tgt))
        if len(pairs) % 5000 == 0:
            print(f"  collected {len(pairs)}")
    return pairs


def from_iitb(token: str | None, max_n: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    print("Loading cfilt/iitb-english-hindi…")
    try:
        ds = load_dataset("cfilt/iitb-english-hindi", split="train", token=token)
    except TypeError:
        ds = load_dataset("cfilt/iitb-english-hindi", split="train")

    pairs: list[tuple[str, str]] = []
    for ex in ds:
        if max_n and len(pairs) >= max_n:
            break
        tr = ex.get("translation") or {}
        src = (tr.get("en") or "").strip()
        tgt = (tr.get("hi") or "").strip()
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=60000)
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--test-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", choices=["opus100", "iitb", "auto"], default="auto")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)

    random.seed(args.seed)
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []

    sources = ["opus100", "iitb"] if args.source == "auto" else [args.source]
    for src_name in sources:
        try:
            if src_name == "opus100":
                pairs = from_opus100(token, args.max_pairs)
            else:
                pairs = from_iitb(token, args.max_pairs)
            if pairs:
                print(f"Using source={src_name} n={len(pairs)}")
                break
        except Exception as e:
            errors.append(f"{src_name}: {e}")
            print(f"Failed {src_name}: {e}", file=sys.stderr)

    if not pairs:
        print("No parallel data loaded.", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    random.shuffle(pairs)
    n = len(pairs)
    n_test = max(1, int(n * args.test_ratio))
    n_val = max(1, int(n * args.val_ratio))
    test = pairs[:n_test]
    val = pairs[n_test : n_test + n_val]
    train = pairs[n_test + n_val :]

    out = TR_ROOT / "data" / "processed" / "en-hi"
    write_tsv(out / "train.tsv", train)
    write_tsv(out / "val.tsv", val)
    write_tsv(out / "test.tsv", test)
    print(f"Wrote {out}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
