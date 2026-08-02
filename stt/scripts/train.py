#!/usr/bin/env python3
"""Fine-tune Whisper for en/hi speech + empty labels for non-speech rejection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

STT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STT_ROOT.parent
sys.path.insert(0, str(STT_ROOT / "src"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class ManifestDataset(torch.utils.data.Dataset):
    rows: list[dict]
    processor: Any
    stt_root: Path
    max_label_length: int = 225

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        import librosa

        row = self.rows[idx]
        audio_path = Path(row["audio"])
        if not audio_path.is_absolute():
            audio_path = self.stt_root / audio_path
        speech, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        text = row.get("text") or ""
        language = row.get("language") or None
        if language in (None, "und"):
            # non-speech / unknown: still need decoder prompt; use English task tokens
            language = "en"

        inputs = self.processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.squeeze(0)

        # Labels: empty string → mostly ignore via -100 after tokenization of empty?
        # Whisper: tokenize transcript as labels
        self.processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels = labels[: self.max_label_length]
        return {
            "input_features": input_features,
            "labels": labels,
            "language": language,
            "kind": row.get("kind", "speech"),
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # If bos token is prepended, whisper training often strips it — handled by processor
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=STT_ROOT / "configs" / "train_en_hi_base.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model path (default: local upstream or Hub id from config)",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA GPU required for fine-tuning on PC. "
            "nvidia-smi failed / torch.cuda.is_available() is False.",
            file=sys.stderr,
        )
        return 1

    cfg = load_yaml(args.config)
    upstream_local = STT_ROOT / "models" / "upstream" / "openai-whisper-base"
    model_id = args.model or (
        str(upstream_local) if (upstream_local / "config.json").exists() else cfg["model_name_or_path"]
    )

    train_path = STT_ROOT / cfg["train_manifest"]
    val_path = STT_ROOT / cfg["val_manifest"]
    if not train_path.exists():
        print(f"Missing {train_path}. Run prepare_common_voice.py first.", file=sys.stderr)
        return 1

    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    # Import HF `evaluate` explicitly (avoid shadowing by a local evaluate.py)
    import importlib

    hf_evaluate = importlib.import_module("evaluate")
    if not hasattr(hf_evaluate, "load"):
        raise RuntimeError(
            "Wrong 'evaluate' module imported (no .load). "
            "Use stt/scripts/run_eval.py for metrics; ensure HuggingFace evaluate is installed: "
            "pip install evaluate jiwer"
        )

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False

    if cfg.get("freeze_encoder"):
        for p in model.model.encoder.parameters():
            p.requires_grad = False

    train_ds = ManifestDataset(load_manifest(train_path), processor, STT_ROOT)
    val_ds = ManifestDataset(load_manifest(val_path), processor, STT_ROOT) if val_path.exists() else None
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    wer_metric = hf_evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        # Also track empty-rate on references that are empty (nonspeech)
        return {"wer": wer}

    out_dir = STT_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        fp16=cfg.get("fp16", True),
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 500),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        predict_with_generate=cfg.get("predict_with_generate", True),
        generation_max_length=cfg.get("generation_max_length", 225),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "wer"),
        greater_is_better=cfg.get("greater_is_better", False),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics if val_ds is not None else None,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    card = out_dir / "MODEL_CARD.md"
    card.write_text(
        f"""# en-hi-base-v1

- Base: `{model_id}`
- Languages: English, Hindi
- Fine-tune goal: speech WER + **empty output** on non-speech vocalizations
  (scream / laugh / moan / cry) via `kind=nonspeech` rows with `text=""`.
- Device policy (Jetson): CUDA only
""",
        encoding="utf-8",
    )
    print(f"Saved fine-tuned model → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
