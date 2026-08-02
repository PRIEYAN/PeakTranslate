#!/usr/bin/env python3
"""
Merge non-speech WAVs (empty transcript) into en-hi manifests.

Fine-tune intent: screaming / laughing / moaning / crying → model outputs ""
instead of hallucinated words.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

STT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nonspeech-dir",
        type=Path,
        default=STT_ROOT / "data" / "raw" / "nonspeech",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of nonspeech clips into train (rest split val/test)",
    )
    args = parser.parse_args()

    wavs = sorted(
        p
        for p in args.nonspeech_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
    )
    if not wavs:
        print(f"No audio files under {args.nonspeech_dir}")
        print("Add scream/laugh/moan WAVs there, then re-run.")
        return 1

    random.seed(args.seed)
    random.shuffle(wavs)
    ns_rows = []
    for p in wavs:
        try:
            rel = str(p.resolve().relative_to(STT_ROOT))
        except ValueError:
            rel = str(p.resolve())
        ns_rows.append(
            {
                "audio": rel,
                "text": "",
                "language": "und",
                "kind": "nonspeech",
            }
        )

    n = len(ns_rows)
    n_train = max(1, int(n * args.train_fraction))
    n_val = max(0, (n - n_train) // 2)
    train_ns = ns_rows[:n_train]
    val_ns = ns_rows[n_train : n_train + n_val]
    test_ns = ns_rows[n_train + n_val :]

    base = STT_ROOT / "data" / "processed" / "en-hi"
    for split, extra in (
        ("train", train_ns),
        ("val", val_ns),
        ("test", test_ns),
    ):
        path = base / split / "manifest.jsonl"
        rows = load_jsonl(path)
        # drop previous nonspeech rows then append fresh
        rows = [r for r in rows if r.get("kind") != "nonspeech"]
        rows.extend(extra)
        random.shuffle(rows)
        write_jsonl(path, rows)
        print(f"{split}: {len(rows)} rows (added {len(extra)} nonspeech)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
