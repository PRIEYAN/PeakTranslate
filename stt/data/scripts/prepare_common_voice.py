#!/usr/bin/env python3
"""
Build en+hi speech manifests for Whisper fine-tuning.

NOTE: Mozilla Common Voice left Hugging Face (Oct 2025 → Mozilla Data Collective).
This script uses Google FLEURS instead (works with modern `datasets`, no trust_remote_code):

  - English: google/fleurs  config en_us
  - Hindi:   google/fleurs  config hi_in

Same CLI as before so your command still works.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

STT_ROOT = Path(__file__).resolve().parents[2]   # → stt/
REPO_ROOT = Path(__file__).resolve().parents[3]  # → PeakTranslation/

# FLEURS language configs → our Whisper language codes
FLEURS_LANGS = {
    "en": "en_us",
    "hi": "hi_in",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare en/hi speech data from Google FLEURS (Common Voice unavailable on HF)."
    )
    parser.add_argument(
        "--max-en",
        type=int,
        default=2000,
        help="Max English clips (0 = all of train+validation+test combined cap per split loop)",
    )
    parser.add_argument("--max-hi", type=int, default=2000, help="Max Hindi clips")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated FLEURS splits to pull from (default: train,validation)",
    )
    args = parser.parse_args()

    env_path = REPO_ROOT / ".env"
    load_dotenv(env_path)
    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            f"HF_TOKEN missing.\n"
            f"  Looked for: {env_path}\n"
            f"  Exists: {env_path.exists()}\n"
            f"  Fix: put HF_TOKEN=hf_... in that file (repo root .env).",
            file=sys.stderr,
        )
        return 1

    from datasets import Audio, load_dataset
    from huggingface_hub import login
    import librosa
    import numpy as np
    import soundfile as sf

    login(token=token, add_to_git_credential=False)
    random.seed(args.seed)

    audio_root = STT_ROOT / "data" / "raw" / "fleurs"
    audio_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    def load_audio_array(audio_obj) -> tuple[np.ndarray, int]:
        """Decode without torchcodec (datasets 4+ default)."""
        import io

        if audio_obj is None:
            raise ValueError("audio is None")

        # Already-decoded dict from older backends
        if isinstance(audio_obj, dict) and audio_obj.get("array") is not None:
            arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate") or 16000)
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            return arr, 16000

        # decode=False shape: {"path": ..., "bytes": ...}
        if isinstance(audio_obj, dict):
            raw = audio_obj.get("bytes")
            path = audio_obj.get("path")
            if raw:
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            elif path:
                arr, sr = librosa.load(path, sr=16000, mono=True)
                return np.asarray(arr, dtype=np.float32), 16000
            else:
                raise ValueError(f"Unrecognized audio dict keys: {list(audio_obj)}")
            if getattr(arr, "ndim", 1) > 1:
                arr = arr.mean(axis=1)
            arr = np.asarray(arr, dtype=np.float32)
            if int(sr) != 16000:
                arr = librosa.resample(arr, orig_sr=int(sr), target_sr=16000)
            return arr, 16000

        raise TypeError(f"Unsupported audio type: {type(audio_obj)}")

    def ingest(lang: str, max_n: int) -> None:
        fleurs_name = FLEURS_LANGS[lang]
        print(f"Loading google/fleurs config={fleurs_name} splits={splits}…")
        lang_dir = audio_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for split in splits:
            if max_n and n >= max_n:
                break
            try:
                ds = load_dataset(
                    "google/fleurs",
                    fleurs_name,
                    split=split,
                    token=token,
                )
            except TypeError:
                ds = load_dataset("google/fleurs", fleurs_name, split=split)

            # Avoid torchcodec: do not auto-decode; we decode via soundfile/librosa
            try:
                ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
            except TypeError:
                ds = ds.cast_column("audio", Audio(decode=False))

            for i, ex in enumerate(ds):
                if max_n and n >= max_n:
                    break
                text = (ex.get("transcription") or ex.get("raw_transcription") or "").strip()
                if not text:
                    continue
                try:
                    arr, sr = load_audio_array(ex["audio"])
                except Exception as e:
                    print(f"  skip {lang}/{split}/{i}: {e}", file=sys.stderr)
                    continue
                wav_path = lang_dir / f"{lang}_{split}_{i:06d}.wav"
                if not wav_path.exists():
                    sf.write(str(wav_path), arr, sr)
                rows.append(
                    {
                        "audio": str(wav_path.relative_to(STT_ROOT)),
                        "text": text,
                        "language": lang,
                        "kind": "speech",
                        "source": f"fleurs:{fleurs_name}:{split}",
                    }
                )
                n += 1
                if n % 100 == 0:
                    print(f"  {lang}: {n} clips")
        print(f"  {lang}: done ({n})")
        if n == 0:
            print(
                f"WARNING: 0 clips for {lang}. Check HF login / network / fleurs access.",
                file=sys.stderr,
            )

    ingest("en", args.max_en)
    ingest("hi", args.max_hi)

    if not rows:
        print("No rows collected — aborting.", file=sys.stderr)
        return 1

    random.shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * args.test_ratio))
    n_val = max(1, int(n * args.val_ratio))
    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]

    out_base = STT_ROOT / "data" / "processed" / "en-hi"
    write_jsonl(out_base / "train" / "manifest.jsonl", train)
    write_jsonl(out_base / "val" / "manifest.jsonl", val)
    write_jsonl(out_base / "test" / "manifest.jsonl", test)

    ns_dir = STT_ROOT / "data" / "raw" / "nonspeech"
    ns_dir.mkdir(parents=True, exist_ok=True)
    (ns_dir / "README.md").write_text(
        """# Non-speech rejection clips (fine-tune target)

Put WAV files here of: screaming, laughing, moaning, crying, coughing, humming —
**without** linguistic speech. Then run:

```bash
python stt/data/scripts/merge_nonspeech.py
```

Each clip is labeled `text=""` so Whisper learns to emit an **empty transcript**.
""",
        encoding="utf-8",
    )

    print(f"Wrote manifests under {out_base}")
    print(f"Add non-speech WAVs to {ns_dir} then run merge_nonspeech.py")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(
        "Note: FLEURS has ~1–2k utterances/lang; --max-en/--max-hi above that just take all."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
