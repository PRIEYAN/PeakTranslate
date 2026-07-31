# Translation — MarianMT: Download, Fine-Tune, Quantize, Jetson Runtime

Text-to-text translation stage for PeakTranslation. Target: **Jetson Nano Dev Kit** (4 GB). Train/quantize on **PC/cloud**; deploy one **language-pair artifact** at a time. Switching pairs must be **config + folder**, not code forks.

---

## 1. Goals

| Goal | Approach |
|------|----------|
| Good domain translation | Fine-tune Helsinki-NLP / OPUS Marian on your bitext |
| Fit Nano with Whisper + Piper | Small bilingual Marian; **one pair loaded** |
| **GPU primary** | `device: cuda` under GPU lock with Whisper |
| **CPU fallback allowed** | CT2 int8 if OOM (`allow_cpu_fallback: true`) |
| Shrink weights, keep quality | FP16 → CT2 float16 → CT2 int8 with BLEU/chrF gates |
| Easy pair switch / retrain | Pair registry + identical folder layout per pair |

---

## 2. Runtime availability (research summary)

| Runtime | Classic Nano Dev Kit | Notes |
|---------|----------------------|--------|
| **HF MarianMT + Jetson PyTorch CUDA** | **Yes — MVP default** | Same Torch constraints as STT (JP4 / CUDA 10.2 wheels; not Torch 2.x). |
| **CTranslate2 Translator (GPU)** | **Not drop-in** | aarch64 PyPI CT2 = CPU-only; GPU needs source build + newer CUDA story. |
| **CTranslate2 Translator (CPU int8)** | **Yes as fallback** | Marian int8 on ARM CPU can be surprisingly usable for short sentences; frees GPU for Whisper. Trade latency. |
| **Marian C++ / official Marian** | Possible | Heavier ops path; usually unnecessary if HF/CT2 works. |
| **ONNX Runtime (Optimum export)** | Partial | Export+quantize on PC possible; ORT-GPU on Nano not drop-in. CPU ORT possible but CT2 CPU often better for Marian. |

### Nano production choice (two valid strategies)

**A — GPU Marian (default / preferred)**  
Whisper and Marian **time-share CUDA** via GPU lock; both PyTorch. Matches the product rule: **STT + MT on GPU**.

**B — CPU Marian fallback**  
Only when CUDA OOM or contention: CT2 int8 on CPU. Whisper **stays on CUDA** (never moves to CPU).

```text
Train PC:  opus-mt / Marian HF → fine-tune → eval BLEU/chrF
Export:    PyTorch FP16 pair folder (primary) + ct2 int8 (fallback artifact)
Deploy:    Strategy A first; enable Strategy B via allow_cpu_fallback
```

---

## 3. Production-grade folder structure

```text
translate/
├── README.md
├── configs/
│   ├── pairs.yaml                     # language-pair registry (switch here)
│   ├── train_en_ta.yaml
│   ├── train_en_hi.yaml
│   └── decode.yaml                    # beam, max_length, etc.
├── data/
│   ├── raw/                           # OPUS dumps, in-house bitext (gitignored)
│   ├── processed/
│   │   ├── en-ta/
│   │   │   ├── train.tsv              # src\ttgt
│   │   │   ├── val.tsv
│   │   │   └── test.tsv
│   │   └── en-hi/
│   └── scripts/
│       ├── download_opus.py
│       ├── clean_bitext.py            # lang-id, length ratio, dedupe
│       └── split_corpus.py
├── models/
│   ├── upstream/
│   │   ├── opus-mt-en-hi/
│   │   └── opus-mt-en-mul/            # or best available en→ta base
│   ├── finetuned/
│   │   ├── en-ta-v1/
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin / model.safetensors
│   │   │   ├── tokenizer* / source.spm / target.spm
│   │   │   ├── MODEL_CARD.md
│   │   │   └── metrics.json           # BLEU, chrF, latency
│   │   └── en-hi-v1/
│   └── export/
│       ├── en-ta-v1-fp16/
│       ├── en-ta-v1-ct2-int8/         # recommended shrink path
│       ├── en-hi-v1-fp16/
│       └── en-hi-v1-ct2-int8/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_fp16.py
│   ├── export_ctranslate2.py
│   ├── runtime/
│   │   ├── interface.py               # MTEngine protocol
│   │   ├── marian_pytorch.py          # Nano GPU MVP
│   │   ├── marian_ct2.py              # CPU int8 / future GPU CT2
│   │   └── registry.py                # pairs.yaml → engine + paths
│   └── jetson/
│       ├── smoke_mt.py
│       └── bench_pairs.py
├── scripts/
│   ├── download_upstream.sh
│   ├── train_pair.sh
│   ├── quantize_export.sh
│   └── package_for_nano.sh
└── artifacts/
    └── mt-en-ta-v1-nano.tar.gz
```

Each pair is a **self-contained directory** under `finetuned/` and `export/`. Adding `en-fr` means new data + train + export + one YAML entry.

---

## 4. Language-pair switch & train architecture

```yaml
# translate/configs/pairs.yaml
default_pair: en-ta

pairs:
  en-ta:
    src: en
    tgt: ta
    upstream: Helsinki-NLP/opus-mt-en-mul   # replace with best real base you choose
    artifact: models/export/en-ta-v1-fp16
    runtime: marian_pytorch                 # primary: GPU
    device: cuda
    allow_cpu_fallback: true
    fallback:
      runtime: marian_ct2
      device: cpu
      artifact: models/export/en-ta-v1-ct2-int8
    decode:
      beam_size: 2
      max_new_tokens: 128

  en-hi:
    src: en
    tgt: hi
    upstream: Helsinki-NLP/opus-mt-en-hi
    artifact: models/export/en-hi-v1-fp16
    runtime: marian_pytorch
    device: cuda
    allow_cpu_fallback: true
    fallback:
      runtime: marian_ct2
      device: cpu
      artifact: models/export/en-hi-v1-ct2-int8
    decode:
      beam_size: 2
      max_new_tokens: 128
```

### Switch pair at runtime

```text
pairs.yaml ──► registry.set_pair("en-hi")
                 ├─ unload previous weights
                 ├─ load CUDA artifact (primary)
                 └─ keep CPU fallback artifact ready if allow_cpu_fallback
```

Orchestrator only knows:

```python
class MTEngine(Protocol):
    def set_pair(self, pair_id: str) -> None: ...
    def translate(self, text: str) -> Translation: ...
```

### Train a new pair

1. Create `data/processed/<src>-<tgt>/{train,val,test}.tsv`.
2. Pick closest `upstream` Marian/OPUS model.
3. Add `configs/train_<pair>.yaml` + entry in `pairs.yaml` with `device: cuda`.
4. `./scripts/train_pair.sh en-fr` → fine-tune → export FP16 (GPU) **and** CT2 int8 (CPU fallback).
5. Add Piper voice for `tgt` in TTS registry (see `tts/`).

No changes to STT or queue workers required. STT stays CUDA-only even if MT falls back to CPU.

---

## 5. Download upstream model locally

```bash
# Example: English→Hindi
huggingface-cli download Helsinki-NLP/opus-mt-en-hi \
  --local-dir translate/models/upstream/opus-mt-en-hi

# Example: multilingual English→many (if using for en-ta bootstrap)
huggingface-cli download Helsinki-NLP/opus-mt-en-mul \
  --local-dir translate/models/upstream/opus-mt-en-mul
```

Confirm tokenizer + `pytorch_model.bin` / safetensors exist. Document the exact Hub revision in `MODEL_CARD.md`.

---

## 6. Fine-tune (PC/cloud only)

### Data hygiene (critical for quality)

- Parallel `src \t tgt` only.
- Lang-ID filter, length-ratio filter, dedupe, remove HTML/noise.
- Hold out a **domain** test set (product phrases, names, numbers).

### Training

```bash
./translate/scripts/train_pair.sh en-ta
# wraps:
python translate/src/train.py \
  --config translate/configs/train_en_ta.yaml \
  --pair en-ta \
  --output_dir translate/models/finetuned/en-ta-v1
```

Use HF `Seq2SeqTrainer` / Marian fine-tune recipe. Track **BLEU + chrF** (and human spot-checks for morphologically rich targets).

### Acceptance before quantize

| Gate | Rule |
|------|------|
| Domain BLEU/chrF | ≥ baseline upstream on your test.tsv |
| Hallucinations | Spot-check empty/odd STT inputs |
| Size | Document params + disk MB |

---

## 7. Reduce weights / quantize

| Step | Method | Typical quality | Deploy |
|------|--------|-----------------|--------|
| 1 | Small bilingual base (not LLM) | — | Always |
| 2 | Fine-tune then **FP16** save | ≈ FP32 | Torch CUDA |
| 3 | **CT2 float16** | ≈ FP16 | CT2 CPU/GPU |
| 4 | **CT2 int8** | Small BLEU drop usual | **Best size for Nano** |
| 5 | Lower `beam_size`, cap `max_new_tokens` | Small | Latency |

### Export CT2 INT8 (recommended shrink)

On training PC (CT2 installed for **conversion** — x86 GPU/CPU is fine):

```bash
ct2-transformers-converter \
  --model translate/models/finetuned/en-ta-v1 \
  --output_dir translate/models/export/en-ta-v1-ct2-int8 \
  --quantization int8 \
  --copy_files source.spm target.spm tokenizer_config.json vocab.json \
               special_tokens_map.json tokenizer.json
```

Re-evaluate:

```bash
python translate/src/evaluate.py \
  --artifact translate/models/export/en-ta-v1-ct2-int8 \
  --runtime marian_ct2 \
  --test translate/data/processed/en-ta/test.tsv
```

Keep FP16 export if int8 fails your BLEU/chrF gate (e.g. drop &gt; 1 BLEU — set your threshold).

### Disk size intuition

OPUS Marian bilingual checkpoints often land ~300 MB FP32; CT2 int8 commonly ~half or better. Exact numbers depend on the pair — measure with `du -sh`.

---

## 8. Build / package runtime for Jetson Nano

### Strategy A — PyTorch CUDA (primary)

```bash
rsync -avP translate/models/export/en-ta-v1-fp16/ \
  nano:/opt/peaktranslation/models/mt/en-ta-v1-fp16/
```

Nano: Jetson Torch + `transformers`. Use GPU lock with Whisper (STT always holds CUDA priority conceptually — STT never leaves GPU).

### Strategy B — CT2 CPU INT8 (fallback only)

Use when CUDA OOM or MT latency under lock is unacceptable. Whisper **remains on CUDA**.

1. Install **CPU** CTranslate2 aarch64 wheel on Nano.
2. Copy `en-ta-v1-ct2-int8/` to device.
3. Keep `allow_cpu_fallback: true` and `fallback:` block in `pairs.yaml` (primary still `device: cuda`).

```bash
python translate/src/jetson/smoke_mt.py --pair en-ta --text "How are you?"
```

### Adapter + queue

Marian worker pops **Q1** transcripts, pushes **Q2** translations (`docs/04-realtime-queue-pipeline.md`). Pair id comes from shared pipeline config.

---

## 9. Memory policy with Whisper on Nano

| Mode | Whisper | Marian | Note |
|------|---------|--------|------|
| **A (default)** | **CUDA only** | **CUDA** | GPU lock; tiny Whisper + small Marian |
| B (fallback) | **CUDA only** | CT2 CPU int8 | If OOM / contention — STT still never CPU |
| Forbidden | CPU | any | STT must not run on CPU |
| Never | — | Load all pairs | Load **one** pair; swap on demand |

---

## 10. Checklist

- [ ] Upstream Marian downloaded per pair
- [ ] Clean bitext + metrics on held-out domain set
- [ ] Finetuned folder + MODEL_CARD
- [ ] CT2 int8 (or FP16) export + post-quant metrics
- [ ] Entry in `pairs.yaml` with runtime/device
- [ ] Nano smoke + latency log
- [ ] TTS voice exists for `tgt` language

---

## Related

- [stt/README.md](../stt/README.md) — Whisper
- [tts/README.md](../tts/README.md) — Piper
- [docs/04-realtime-queue-pipeline.md](../docs/04-realtime-queue-pipeline.md)
