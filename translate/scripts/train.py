#!/usr/bin/env python3
"""Fine-tune MarianMT opus-mt-en-hi on local GPU."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

TR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TR_ROOT.parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_tsv(path: Path, max_rows: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            src, tgt = line.split("\t", 1)
            src, tgt = src.strip(), tgt.strip()
            if src and tgt:
                rows.append({"translation": {"en": src, "hi": tgt}})
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TR_ROOT / "configs" / "train_en_hi.yaml",
    )
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for Marian fine-tune on PC.", file=sys.stderr)
        return 1

    cfg = load_yaml(args.config)
    upstream_local = TR_ROOT / "models" / "upstream" / "opus-mt-en-hi"
    model_id = args.model or (
        str(upstream_local) if (upstream_local / "config.json").exists() else cfg["model_name_or_path"]
    )

    train_path = TR_ROOT / cfg["train_file"]
    val_path = TR_ROOT / cfg["val_file"]
    if not train_path.exists():
        print(f"Missing {train_path}. Run prepare_en_hi.py first.", file=sys.stderr)
        return 1

    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    hf_evaluate = importlib.import_module("evaluate")
    if not hasattr(hf_evaluate, "load"):
        raise RuntimeError("HuggingFace evaluate package not found / shadowed")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_rows = read_tsv(train_path, int(cfg.get("max_train_samples") or 0))
    val_rows = read_tsv(val_path, int(cfg.get("max_val_samples") or 0)) if val_path.exists() else []
    print(f"train rows={len(train_rows)} val rows={len(val_rows)} model={model_id}")

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if val_rows else None

    max_src = int(cfg["max_source_length"])
    max_tgt = int(cfg["max_target_length"])

    def preprocess(batch: dict) -> dict:
        inputs = [ex["en"] for ex in batch["translation"]]
        targets = [ex["hi"] for ex in batch["translation"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_src,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_tgt,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="tokenize train",
    )
    val_tok = (
        val_ds.map(
            preprocess,
            batched=True,
            remove_columns=val_ds.column_names,
            desc="tokenize val",
        )
        if val_ds is not None
        else None
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    use_generate = bool(cfg.get("predict_with_generate", False))
    metric = hf_evaluate.load("sacrebleu") if use_generate else None

    def postprocess(preds: list[str], labels: list[str]) -> tuple[list[str], list[list[str]]]:
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        if metric is None:
            return {}
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    out_dir = TR_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.06),
        num_train_epochs=cfg["num_train_epochs"],
        fp16=cfg.get("fp16", True),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg.get("eval_strategy", "steps") if val_tok is not None else "no",
        eval_steps=cfg.get("eval_steps", 500),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        predict_with_generate=use_generate,
        generation_max_length=cfg.get("generation_max_length", 128),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True) if val_tok is not None else False,
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        report_to=[],
        seed=cfg.get("seed", 42),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics if use_generate and val_tok is not None else None,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {}
    if val_tok is not None:
        metrics = trainer.evaluate()
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (out_dir / "MODEL_CARD.md").write_text(
        f"""# en-hi-v1

- Base: `{model_id}`
- Direction: English → Hindi
- Train rows: {len(train_rows)}
- Device policy: CUDA primary; CT2 CPU fallback optional on Jetson
- Metrics: {json.dumps(metrics)}
""",
        encoding="utf-8",
    )
    print(f"Saved fine-tuned model → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
