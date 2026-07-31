# Translation Deep Research — MarianMT / OPUS-MT on Jetson Nano

Research for PeakTranslation text translation: download, fine-tune, quantize, and run **MarianMT (Helsinki-NLP OPUS-MT family)** on the **Jetson Nano Developer Kit**, with a pair-registry architecture for easy language switching.

Companion: [`../../translate/README.md`](../../translate/README.md) · [`../../translate/configs/pairs.yaml`](../../translate/configs/pairs.yaml)

---

## 0. Production-grade loosely coupled file structure

Translation is a **standalone package**. STT/TTS never import Marian internals. Adding `en-fr` = new bitext + train + files under `models/` + one `pairs.yaml` row.

**Every checkpoint and quantized export is stored under `models/`.**

```text
translate/                                # standalone MT package (loosely coupled)
├── README.md
├── configs/
│   ├── pairs.yaml                        # pair registry — switch / add pairs HERE
│   ├── train_en_ta.yaml
│   ├── train_en_hi.yaml
│   └── decode.yaml
├── data/
│   ├── raw/                              # OPUS / in-house bitext dumps (gitignored)
│   ├── processed/
│   │   ├── en-ta/{train,val,test}.tsv    # src\ttgt
│   │   └── en-hi/{train,val,test}.tsv
│   └── scripts/
│       ├── download_opus.py
│       ├── clean_bitext.py
│       └── split_corpus.py
├── models/                               # ★ ALL model files stored here
│   ├── upstream/                         # Hub OPUS/Marian bases (gitignored)
│   │   ├── opus-mt-en-hi/
│   │   │   ├── pytorch_model.bin | model.safetensors
│   │   │   ├── config.json
│   │   │   ├── tokenizer* / source.spm / target.spm
│   │   │   └── vocab.json
│   │   └── opus-mt-en-mul/
│   ├── finetuned/                        # after domain fine-tune (gitignored)
│   │   ├── en-ta-v1/
│   │   │   ├── model.safetensors
│   │   │   ├── config.json
│   │   │   ├── tokenizer* / *.spm
│   │   │   ├── MODEL_CARD.md
│   │   │   └── metrics.json              # BLEU / chrF
│   │   └── en-hi-v1/
│   ├── export/                           # Nano deploy artifacts (gitignored)
│   │   ├── en-ta-v1-ct2-int8/            # preferred Nano coexistence w/ Whisper
│   │   │   ├── model.bin                 # CTranslate2
│   │   │   ├── config.json
│   │   │   └── tokenizer / spm files
│   │   ├── en-ta-v1-fp16/                # Torch CUDA alternative
│   │   ├── en-hi-v1-ct2-int8/
│   │   └── en-hi-v1-fp16/
│   └── .gitkeep
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_fp16.py
│   ├── export_ctranslate2.py
│   ├── runtime/                          # adapters — orchestrator uses MTEngine only
│   │   ├── interface.py                  # MTEngine protocol
│   │   ├── marian_pytorch.py
│   │   ├── marian_ct2.py
│   │   └── registry.py                   # pairs.yaml → load models/export/<pair>
│   └── jetson/
│       ├── smoke_mt.py
│       └── bench_pairs.py
├── scripts/
│   ├── download_upstream.sh              # → models/upstream/
│   ├── train_pair.sh                     # → models/finetuned/
│   ├── quantize_export.sh                # → models/export/
│   └── package_for_nano.sh               # rsync one pair export → device
└── artifacts/
    └── mt-en-ta-v1-nano.tar.gz
```

### Loose coupling rules

| Rule | Practice |
|------|----------|
| One pair = one folder | `models/finetuned/<pair>-vN` and `models/export/<pair>-vN-…` |
| Switch via registry | `registry.set_pair("en-hi")` unloads old weights, loads new `models/export/…` |
| No cross-imports | STT/TTS never open Marian files; only consume Q1→Q2 text |
| Runtime choice in YAML | `marian_ct2` (CPU int8) vs `marian_pytorch` (CUDA) |
| Git | Ignore `models/upstream`, `models/finetuned`, `models/export` binaries |

### What goes in each `models/` subfolder

| Path | Contents |
|------|----------|
| `models/upstream/` | Downloaded Helsinki-NLP / OPUS bases |
| `models/finetuned/` | Trained pair checkpoints + metrics |
| `models/export/` | CT2 int8 / FP16 trees the Jetson loads |

On device (one active pair):

```text
/opt/peaktranslation/models/mt/
└── en-ta-v1-ct2-int8/          # from translate/models/export/en-ta-v1-ct2-int8
```

---

## 1. Why MarianMT (not an LLM) on Nano

| Criterion | Marian / OPUS-MT | LLM translator |
|-----------|------------------|----------------|
| Disk / RAM | Tens–low hundreds of MB (esp. CT2 int8) | GBs |
| Latency on short sentences | Good | Often too slow on Nano |
| Pair specialization | One bilingual (or compact multi) checkpoint | General but heavy |
| Jetson coexistence with Whisper | Feasible | Usually not with Whisper+Piper |

OPUS-MT models are Marian Transformer encoder–decoders trained on OPUS bitext — the practical edge MT choice.

---

## 2. Model selection research

### 2.1 Hub naming

Typical Hugging Face ids:

```text
Helsinki-NLP/opus-mt-<src>-<tgt>
Helsinki-NLP/opus-mt-en-hi
Helsinki-NLP/opus-mt-en-mul   # English → many (useful bootstrap)
Helsinki-NLP/opus-mt-mul-en
```

Always verify:

- Exact language codes (ISO-ish Marian codes can differ)
- Whether the pair is **direct** bilingual vs routed through multilingual
- License / MODEL_CARD on Hub

### 2.2 Size / memory (published CT2 benchmarks)

CTranslate2 README benchmarks on an **OPUS-MT** model (illustrative, x86 numbers — use for **relative** shrink, then measure on Nano):

| Engine | Tokens/s (CPU example) | Max memory (example) | BLEU (example) |
|--------|------------------------|----------------------|----------------|
| HF Transformers + PyTorch | ~147 | ~2332 MB | ~27.9 |
| Marian C++ | ~344 | very high in bench | ~27.9 |
| **CT2** | ~525 | ~721 MB | ~27.9 |
| **CT2 int8** | ~696 | ~**516 MB** | ~27.7 |

GPU CT2 benches show further speedups with float16/int8 on **modern** NVIDIA GPUs — **not** equivalent to Nano Maxwell.

**Takeaway:** Converting OPUS-MT → **CTranslate2 int8** is the strongest **size + speed** lever with **small BLEU change** in published benches. Re-validate on **your** pair and domain set.

---

## 3. Runtime availability on Jetson Nano Dev Kit

### 3.1 Hugging Face MarianMT + Jetson PyTorch CUDA

| Item | Status |
|------|--------|
| Works? | **Yes** if JetPack-matched Torch CUDA 10.2 works |
| Pros | Same stack as Whisper; easy fine-tune→deploy |
| Cons | Heavier RAM than CT2; **GPU contention** with Whisper |
| Verdict | Viable; use GPU **lock** with STT |

### 3.2 CTranslate2 on Nano

| Mode | Available? | Notes |
|------|------------|--------|
| **CPU int8** (PyPI aarch64) | **Yes** | aarch64 wheels ship; ARM uses **Ruy** backend for int8 |
| **GPU** | **Not drop-in** | Need source build; CUDA ≥ 11 bias; Nano CC **5.3** gets poor GPU quant fallout (see below) |

#### CT2 GPU compute capability trap (Nano = 5.3)

From [CT2 quantization docs](https://opennmt.net/CTranslate2/quantization.html):

- GPU INT8 optimized for CC **≥ 7.0** (or 6.1)
- For GPU CC **≤ 6.0**, int8/float16 types **fall back toward float32**

So even a heroic CT2-CUDA build on Nano would **not** unlock Orin-like int8 GPU MT.

### 3.3 Recommended Nano strategy (research conclusion)

```text
┌─────────────────────────────────────────────────────────┐
│  DEVICE POLICY                                          │
│  Whisper  → Jetson PyTorch CUDA ONLY (no CPU fallback)  │
│  Marian   → PyTorch CUDA primary + CT2 CPU fallback OK  │
│  Piper    → ONNX Runtime CPU                            │
└─────────────────────────────────────────────────────────┘
```

| Strategy | Whisper | Marian | When |
|----------|---------|--------|------|
| **A (default)** | CUDA only | CUDA + GPU lock | Normal operation |
| **B (fallback)** | CUDA only | CT2 CPU int8 | OOM / GPU contention |
| Forbidden | CPU | — | Never for STT |

---

## 4. Download locally

```bash
# Direct pair example
huggingface-cli download Helsinki-NLP/opus-mt-en-hi \
  --local-dir translate/models/upstream/opus-mt-en-hi

# Multilingual English→* bootstrap (e.g. toward Tamil)
huggingface-cli download Helsinki-NLP/opus-mt-en-mul \
  --local-dir translate/models/upstream/opus-mt-en-mul
```

Pin revision; store in `MODEL_CARD.md`.

---

## 5. Fine-tuning research

### 5.1 Data

| Need | Detail |
|------|--------|
| Format | Parallel `src \t tgt` |
| Sources | OPUS subsets, in-domain bitext, carefully filtered crawled data |
| Cleaning | Lang-ID, length ratio, dedupe, strip noise/HTML |
| Eval | Domain **test.tsv** + BLEU **and** chrF (esp. morphologically rich targets) |

### 5.2 Train location

PC/cloud GPU only. Recipe: HF `MarianMTModel` + `MarianTokenizer` + `Seq2SeqTrainer` (or Marian/fairseq if you already use them).

### 5.3 Accuracy without huge models

- Prefer **bilingual** fine-tune over hoping a giant multilingual LLM will fit Nano
- Domain fine-tune of a small OPUS model often beats a larger generic model you can’t deploy
- Guard against **overfitting** tiny in-house sets (early stop on val chrF)

### 5.4 Gates before quantize

| Metric | Gate |
|--------|------|
| Domain BLEU/chrF | ≥ upstream baseline |
| Manual fluency | Native spot-check |
| Odd inputs | Empty / garbage STT strings shouldn’t explode |

---

## 6. Quantization / weight reduction (deep)

### 6.1 Conversion on PC (x86 is fine)

```bash
ct2-transformers-converter \
  --model translate/models/finetuned/en-ta-v1 \
  --output_dir translate/models/export/en-ta-v1-ct2-int8 \
  --quantization int8 \
  --copy_files source.spm target.spm vocab.json \
               tokenizer_config.json special_tokens_map.json tokenizer.json
```

Other useful `quantization` values: `float16`, `int8_float32`, `int8_float16` (see CT2 docs). Example disk shrinkage for a **base Transformer** in CT2 docs:

| Quantization | Example size |
|--------------|--------------|
| float32 | 364 MB |
| float16 | 182 MB |
| int8_float32 | 100 MB |
| int8_float16 | 95 MB |

Your Marian pair will differ — always `du -sh` after convert.

### 6.2 Runtime compute_type

```python
import ctranslate2
translator = ctranslate2.Translator(
    "en-ta-v1-ct2-int8",
    device="cpu",
    compute_type="int8",  # or "auto"
)
```

On **AArch64 CPU**, int8 is a supported optimized path (Ruy).

### 6.3 Quality policy (“not much accuracy loss”)

1. Measure BLEU/chrF on **frozen** domain test **before** and **after** int8.
2. Accept if within product Δ (e.g. ≤ 1 BLEU — choose explicitly).
3. If fail: ship `float16` CT2 or Torch FP16 instead of int8.
4. Never skip human review for your top-N product phrases.

Decode-side shrink (no weight change): lower `beam_size` (1–2), cap `max_new_tokens`.

---

## 7. Language-pair switch architecture (train & swap)

Pairs are data:

```text
pairs.yaml
  en-ta → artifact path + runtime + src/tgt
  en-hi → …
```

```text
registry.set_pair("en-hi")
  → unload previous Translator/model
  → load export/en-hi-v1-ct2-int8
  → TTS picks voice for tgt=hi
```

### Adding a language pair (production flow)

1. Collect/clean `data/processed/<src>-<tgt>/{train,val,test}.tsv`
2. Choose closest upstream OPUS/Marian
3. Fine-tune → `models/finetuned/<pair>-vN`
4. CT2 int8 export → `models/export/<pair>-vN-ct2-int8`
5. Append `pairs.yaml`; package to Nano
6. Ensure Piper voice exists for `tgt` (`tts/configs/voices.yaml`)

No changes to STT worker or queue topology.

---

## 8. Continuous pipeline role

```text
Q1 transcript_queue  ──pop──►  Marian worker  ──push──►  Q2 translation_queue
```

- Worker always looping (not call-and-exit).
- If strategy A (CUDA): acquire **same gpu_lock** as Whisper.
- If strategy B (CT2 CPU): no GPU lock; true overlap with Whisper CUDA.

Backpressure: bounded Q1/Q2; block or pause VAD when full.

---

## 9. Jetson package & smoke

```bash
# Copy one pair only
rsync -avP translate/models/export/en-ta-v1-ct2-int8/ \
  nano:/opt/peaktranslation/models/mt/en-ta-v1-ct2-int8/

# On Nano
python3 -m pip install ctranslate2   # CPU aarch64 wheel
python translate/src/jetson/smoke_mt.py --pair en-ta --text "How are you?"
```

Measure: `mt_ms`, RSS, BLEU on a tiny on-device phrase set.

---

## 10. Decision summary

| Question | Answer for Nano Dev Kit |
|----------|-------------------------|
| Best shrink path? | Fine-tune small OPUS → FP16 for GPU + CT2 int8 for CPU fallback |
| Best runtime with live Whisper? | **Marian on CUDA** (lock); CPU only if fallback needed |
| STT interaction | Whisper **never** leaves CUDA |
| CT2 GPU int8? | Not a Nano win (CC 5.3); CT2 CPU is the MT fallback |
| Easy new language? | New folder + `pairs.yaml` entry + TTS voice |

---

## 11. Sources

- Helsinki-NLP OPUS-MT / Marian on Hub
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) README benchmarks & OPUS-MT guide
- [CT2 quantization](https://opennmt.net/CTranslate2/quantization.html) (CPU ARM int8; GPU CC tables)
- PeakTranslation queue design — [`../04-realtime-queue-pipeline.md`](../04-realtime-queue-pipeline.md)
- Stage guide — [`../../translate/README.md`](../../translate/README.md)

---

## 12. Bottom line

Treat translation as **one bilingual artifact per pair**, primary on **GPU**. On classic Nano, run Marian on **CUDA with a GPU lock shared with Whisper**; keep **CT2 INT8 CPU** as optional fallback only. Switch languages by registry, not by rewriting the pipeline.
