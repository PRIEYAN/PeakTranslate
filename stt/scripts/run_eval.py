#!/usr/bin/env python3
"""Evaluate WER + non-speech empty-rate on test manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

STT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STT_ROOT.parent
sys.path.insert(0, str(STT_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=STT_ROOT / "models" / "finetuned" / "en-hi-base-v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=STT_ROOT / "data" / "processed" / "en-hi" / "test" / "manifest.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not torch.cuda.is_available():
        print("CUDA required for eval (STT policy).", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    import importlib

    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    hf_evaluate = importlib.import_module("evaluate")
    processor = WhisperProcessor.from_pretrained(str(args.model))
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model)).to("cuda")
    model.eval()
    wer_metric = hf_evaluate.load("wer")

    rows = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_samples:
        rows = rows[: args.max_samples]

    preds, refs = [], []
    ns_total = ns_empty = 0

    for i, row in enumerate(rows):
        path = Path(row["audio"])
        if not path.is_absolute():
            path = STT_ROOT / path
        speech, _ = librosa.load(str(path), sr=16000, mono=True)
        lang = row.get("language") or "en"
        if lang == "und":
            lang = "en"
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
        # fp16 exports need the fp32 mel features cast to the weight dtype
        feats = inputs.input_features.to("cuda", dtype=model.dtype)
        forced = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
        with torch.inference_mode():
            ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=225)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        ref = (row.get("text") or "").strip()
        preds.append(hyp)
        refs.append(ref)
        if row.get("kind") == "nonspeech":
            ns_total += 1
            if hyp == "":
                ns_empty += 1
        if (i + 1) % 50 == 0:
            print(f"… {i + 1}/{len(rows)}")

    # WER on speech-only rows
    speech_idx = [i for i, r in enumerate(rows) if r.get("kind") != "nonspeech"]
    if speech_idx:
        wer = wer_metric.compute(
            predictions=[preds[i] for i in speech_idx],
            references=[refs[i] for i in speech_idx],
        )
    else:
        wer = None

    ns_rate = (ns_empty / ns_total) if ns_total else None
    metrics = {
        "wer_speech": wer,
        "nonspeech_total": ns_total,
        "nonspeech_empty_correct": ns_empty,
        "nonspeech_reject_rate": ns_rate,
        "n": len(rows),
    }
    out = args.model / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
