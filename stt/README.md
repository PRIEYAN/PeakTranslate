# STT — Whisper: Download, Fine-Tune, Quantize, Jetson Runtime

Speech-to-text stage for PeakTranslation. Target device: **NVIDIA Jetson Nano Developer Kit** (4 GB, Maxwell `sm_53`, JetPack 4.6.x / CUDA 10.2). Train and quantize on a **PC/cloud GPU**; deploy artifacts to the Nano.

---

## 1. Goals

| Goal | Approach |
|------|----------|
| Good accuracy on domain speech | Fine-tune `tiny` / `base` (or `.en`) on your mic data |
| Small enough for Nano | Prefer **tiny → base**; avoid `small+` on classic Nano |
| Fast enough for realtime queue | FP16 where supported; INT8 only when runtime exists |
| **CUDA only on Jetson** | `require_cuda: true` — **no CPU fallback**; fail if no GPU |
| Easy language switch / retrain | Locale registry + one artifact tree per lang profile |

---

## 2. Runtime availability (research summary)

| Runtime | Classic Nano Dev Kit | Notes |
|---------|----------------------|--------|
| **PyTorch CUDA (Jetson wheel)** | **Yes — MVP default** | Match JetPack (CUDA 10.2). PyTorch 2.x needs CUDA 11+ → **not** for original Nano. Use ~1.10.x Jetson builds. |
| **openai-whisper / HF Whisper** | **Yes (with Py constraints)** | JP4 Python 3.6 vs Whisper needing 3.8+: use Q-engineering-style image, container, or forked Whisper. |
| **faster-whisper + CTranslate2 GPU** | **Not drop-in** | PyPI aarch64 CT2 = **CPU-only**. GPU needs source build; current CT2 CUDA path wants **CUDA ≥ 11**. Treat as **Orin / later**, not Nano MVP. |
| **CTranslate2 INT8 on CPU** | Possible but slow | Only if GPU path fails; realtime STT on Nano CPU is weak. |
| **whisper_trt / TensorRT** | **Mostly Orin+** | NVIDIA-AI-IOT `whisper_trt` / TensorRT-LLM Whisper demos target newer Jetsons. Classic Nano TensorRT is older and painful for Whisper. |
| **ONNX Runtime GPU** | **Not drop-in on Nano** | Jetson ORT-GPU wheels are mainly JP6/Orin. |

### Nano production choice

```text
Train PC:  HF Whisper (PyTorch) → fine-tune → eval WER
Export:    keep PyTorch checkpoint (Nano MVP)
Optional:  CT2 float16/int8 on training PC for future Orin deploy
Deploy:    Jetson PyTorch CUDA + whisper tiny/base.en
```

---

## 3. Production-grade folder structure

```text
stt/
├── README.md                          # this file
├── configs/
│   ├── languages.yaml                 # STT language profiles (switch here)
│   ├── train_tiny_en.yaml
│   ├── train_base_en.yaml
│   └── train_tiny_multilingual.yaml
├── data/
│   ├── raw/                           # original recordings (gitignored)
│   ├── processed/
│   │   ├── en/
│   │   │   ├── train/manifest.jsonl   # {"audio":"...","text":"..."}
│   │   │   ├── val/manifest.jsonl
│   │   │   └── test/manifest.jsonl
│   │   └── ta/                        # add locale folders as needed
│   └── scripts/
│       ├── prepare_dataset.py
│       ├── augment.py
│       └── validate_manifest.py
├── models/
│   ├── upstream/                      # downloaded base weights (gitignored)
│   │   └── openai-whisper-tiny.en/
│   ├── finetuned/                     # full precision after train
│   │   └── en-domain-v1/
│   │       ├── pytorch_model.bin / model.safetensors
│   │       ├── config.json
│   │       ├── tokenizer* / preprocessor*
│   │       ├── MODEL_CARD.md
│   │       └── metrics.json           # WER before/after
│   └── export/                        # size-reduced deploy artifacts
│       ├── en-domain-v1-fp16/         # torchscript or half weights
│       └── en-domain-v1-ct2-int8/     # optional; Orin path
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_fp16.py
│   ├── export_ctranslate2.py          # optional
│   ├── runtime/
│   │   ├── interface.py               # STTEngine protocol
│   │   ├── whisper_pytorch.py         # Nano MVP adapter
│   │   ├── whisper_ct2.py             # optional adapter
│   │   └── registry.py                # load profile from languages.yaml
│   └── jetson/
│       ├── smoke_stt.py
│       └── tegrastats_notes.md
├── scripts/
│   ├── download_upstream.sh
│   ├── train.sh
│   ├── quantize_export.sh
│   └── package_for_nano.sh            # rsync export → device
└── artifacts/                         # CI/release tarballs (gitignored)
    └── stt-en-domain-v1-nano.tar.gz
```

**Do not commit** `models/upstream`, `models/finetuned`, `data/raw`, or large exports — only manifests, configs, cards, and code.

---

## 4. Language / profile switch architecture

STT “language” is a **profile**, not hardcoded if/else.

```yaml
# stt/configs/languages.yaml
default_profile: en_domain_v1

profiles:
  en_domain_v1:
    lang_code: en
    multilingual: false
    upstream: openai/whisper-tiny.en   # or local models/upstream/...
    artifact: models/export/en-domain-v1-fp16
    runtime: whisper_pytorch           # CUDA-capable adapter only
    device: cuda                       # REQUIRED
    require_cuda: true
    allow_cpu_fallback: false          # HARD: never CPU
    sample_rate: 16000
    decode:
      beam_size: 1                     # lower = faster on Nano
      language: en
      task: transcribe

  en_base_v1:
    lang_code: en
    upstream: openai/whisper-base.en
    artifact: models/export/en-base-v1-fp16
    runtime: whisper_pytorch
    device: cuda
    require_cuda: true
    allow_cpu_fallback: false

  # Example: add Tamil-capable multilingual later
  ta_multilingual_v1:
    lang_code: ta
    multilingual: true
    upstream: openai/whisper-tiny
    artifact: models/export/ta-multi-v1-fp16
    runtime: whisper_pytorch
    device: cuda
    require_cuda: true
    allow_cpu_fallback: false
    decode:
      language: ta
      task: transcribe
```

**Device policy:** If `torch.cuda.is_available()` is false at startup, **abort** — do not load Whisper on CPU.

### How to switch or train a new language

1. Add `data/processed/<lang>/…` manifests.
2. Add a profile under `profiles:` pointing at upstream + future artifact path.
3. Run train → export → point `artifact:` at the new folder.
4. Pipeline config sets `stt.profile: ta_multilingual_v1` — **no adapter rewrite**.

```text
languages.yaml ──► registry.load(profile) ──► STTEngine adapter ──► Q1 transcripts
```

---

## 5. Download upstream model locally

On the **training machine**:

```bash
# Option A: Hugging Face CLI
huggingface-cli download openai/whisper-tiny.en \
  --local-dir stt/models/upstream/openai-whisper-tiny.en

# Option B: openai-whisper cache
python -c "import whisper; whisper.load_model('tiny.en', download_root='stt/models/upstream')"
```

Verify:

```bash
du -sh stt/models/upstream/*
ls stt/models/upstream/openai-whisper-tiny.en
```

---

## 6. Fine-tune (PC/cloud only)

### Data

- 16 kHz mono WAV/FLAC + transcript.
- Domain match: same mic class, noise, accents as Jetson deployment.
- Start with tens of hours if possible; even a few hours of targeted data helps adaptation.

### Training outline (HF Transformers)

```bash
python stt/src/train.py \
  --config stt/configs/train_tiny_en.yaml \
  --profile en_domain_v1 \
  --output_dir stt/models/finetuned/en-domain-v1
```

Typical knobs for edge-oriented quality:

- Freeze encoder early epochs **or** use LoRA/PEFT to keep adapters small.
- SpecAugment / noise / gain augmentation.
- Early stop on val WER.

### Acceptance before quantize

| Metric | Gate |
|--------|------|
| Val WER vs upstream | Must not regress badly on domain set (define your max ΔWER) |
| Smoke on 20 Jetson-like clips | Manual listen OK |
| Checkpoint size | Documented in `MODEL_CARD.md` |

Write `metrics.json` and `MODEL_CARD.md` next to the finetuned folder.

---

## 7. Reduce weights / quantize (accuracy-aware)

Order from **safest → most aggressive**:

| Step | Method | Accuracy impact | Nano usefulness |
|------|--------|-----------------|-----------------|
| 1 | Choose **tiny** / **tiny.en** over base | Some WER↑ | Highest win |
| 2 | **FP16** weights / half inference | Usually small | Good if Torch FP16 works on your build |
| 3 | Smaller decode (`beam_size=1`, no temperature fallbacks) | Small | Latency win |
| 4 | **CTranslate2 float16** | Small | Great on Orin; Nano GPU CT2 hard |
| 5 | **CTranslate2 int8** | Mild WER↑ | Size/speed; validate on domain set |
| 6 | TensorRT / whisper_trt | Varies | Prefer Orin |

### Export FP16 (Nano MVP)

```bash
python stt/src/export_fp16.py \
  --src stt/models/finetuned/en-domain-v1 \
  --dst stt/models/export/en-domain-v1-fp16
```

### Optional CT2 INT8 (training PC → future device)

```bash
ct2-transformers-converter \
  --model stt/models/finetuned/en-domain-v1 \
  --output_dir stt/models/export/en-domain-v1-ct2-int8 \
  --quantization int8 \
  --copy_files tokenizer.json preprocessor_config.json
```

**Always** re-run `evaluate.py` on the same test set after quantization. Keep the FP16 export if INT8 ΔWER is unacceptable.

### What “not much accuracy loss” means here

- Prefer **architecture shrink (tiny)** + **FP16** before INT8.
- INT8 only if domain WER stays within your product budget (e.g. ≤ +1–2 absolute WER — set your own gate).
- Never quantize without a frozen test set.

---

## 8. Build / package runtime for Jetson Nano

### On Nano

1. Install JetPack-matched **PyTorch CUDA**.
2. Install STT deps compatible with that Python/Torch.
3. Copy export tree:

```bash
./stt/scripts/package_for_nano.sh en-domain-v1-fp16 nano:/opt/peaktranslation/models/stt/
```

4. Point device config:

```yaml
stt:
  profile: en_domain_v1
  artifact: /opt/peaktranslation/models/stt/en-domain-v1-fp16
  runtime: whisper_pytorch
  device: cuda
```

5. Smoke:

```bash
python stt/src/jetson/smoke_stt.py --wav /opt/peaktranslation/samples/hello.wav
tegrastats  # watch RAM during infer
```

### Adapter contract (for queue pipeline)

```python
class STTEngine(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> Transcript: ...
```

`registry.py` selects `whisper_pytorch` vs `whisper_ct2` from `languages.yaml` so Orin can swap runtime later without touching the orchestrator.

---

## 9. Suggested model sizes for Nano

| Model | Rough fit | Recommendation |
|-------|-----------|----------------|
| `tiny.en` / `tiny` | Best | Default for continuous listening |
| `base.en` / `base` | Tight but workable | Use if WER needs it and MT is tiny |
| `small+` | Often too heavy with Marian + Piper | Avoid on 4 GB continuous pipeline |

---

## 10. Checklist

- [ ] Upstream downloaded under `models/upstream`
- [ ] Domain manifests for each language profile
- [ ] Fine-tune + `MODEL_CARD.md` + `metrics.json`
- [ ] FP16 (or CT2) export under `models/export`
- [ ] Post-quantize WER within gate
- [ ] Profile entry in `languages.yaml`
- [ ] Nano smoke test under `tegrastats`
- [ ] Wired to continuous pipeline Q1 (see `docs/04-realtime-queue-pipeline.md`)

---

## Related

- [translate/README.md](../translate/README.md) — MarianMT
- [tts/README.md](../tts/README.md) — Piper
- [docs/04-realtime-queue-pipeline.md](../docs/04-realtime-queue-pipeline.md)
