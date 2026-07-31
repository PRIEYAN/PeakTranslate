# STT Deep Research — Whisper on Jetson Nano Dev Kit

Research notes for PeakTranslation speech-to-text: what fits the **original NVIDIA Jetson Nano Developer Kit** (4 GB unified memory, Maxwell GPU **compute capability 5.3**, JetPack **4.6.x / CUDA 10.2**), how to download / fine-tune / shrink weights, and which runtimes are actually available.

Companion implementation guide: [`../../stt/README.md`](../../stt/README.md) · registry: [`../../stt/configs/languages.yaml`](../../stt/configs/languages.yaml)

---

## 0. Production-grade loosely coupled file structure

STT is its **own package**. The orchestrator only talks to `STTEngine` + config. Swapping Whisper → another ASR (or PyTorch → CT2) means a new adapter under `src/runtime/`, not edits to MT/TTS.

**Model weights live under `models/`** (gitignored binaries). Config points at paths; code never hardcodes Hub IDs at inference time.

```text
stt/                                      # standalone STT package (loosely coupled)
├── README.md
├── configs/
│   ├── languages.yaml                    # profile registry — switch / add languages HERE
│   ├── train_tiny_en.yaml
│   ├── train_base_en.yaml
│   └── decode_defaults.yaml
├── data/
│   ├── raw/                              # recordings (gitignored)
│   ├── processed/
│   │   ├── en/{train,val,test}/manifest.jsonl
│   │   └── <lang>/{train,val,test}/manifest.jsonl
│   └── scripts/
│       ├── prepare_dataset.py
│       ├── augment.py
│       └── validate_manifest.py
├── models/                               # ★ ALL model files stored here
│   ├── upstream/                         # downloaded base checkpoints (gitignored)
│   │   ├── openai-whisper-tiny.en/
│   │   │   ├── model.safetensors | pytorch_model.bin
│   │   │   ├── config.json
│   │   │   ├── tokenizer.json
│   │   │   └── preprocessor_config.json
│   │   └── openai-whisper-base.en/
│   ├── finetuned/                        # full-precision after training (gitignored)
│   │   └── en-domain-v1/
│   │       ├── model.safetensors
│   │       ├── config.json
│   │       ├── tokenizer* / preprocessor*
│   │       ├── MODEL_CARD.md
│   │       └── metrics.json              # WER before/after
│   ├── export/                           # Nano / edge deploy artifacts (gitignored)
│   │   ├── en-domain-v1-fp16/            # production default for Nano
│   │   └── en-domain-v1-ct2-int8/        # optional (Orin / CPU CT2)
│   └── .gitkeep
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_fp16.py
│   ├── export_ctranslate2.py
│   ├── runtime/                          # adapters only — orchestrator depends on interface
│   │   ├── interface.py                  # STTEngine protocol
│   │   ├── whisper_pytorch.py
│   │   ├── whisper_ct2.py
│   │   └── registry.py                   # languages.yaml → load models/<artifact>
│   └── jetson/
│       ├── smoke_stt.py
│       └── bench_latency.py
├── scripts/
│   ├── download_upstream.sh              # → models/upstream/
│   ├── train.sh                          # → models/finetuned/
│   ├── quantize_export.sh                # → models/export/
│   └── package_for_nano.sh               # rsync models/export/<id> → device
└── artifacts/                            # release tarballs of export/ (gitignored)
    └── stt-en-domain-v1-nano.tar.gz
```

### Loose coupling rules

| Rule | Practice |
|------|----------|
| Interfaces only | Pipeline imports `STTEngine`, never `whisper` directly |
| Config = language switch | New lang/profile = folder under `models/` + entry in `languages.yaml` |
| Models isolated | All weights under `stt/models/`; MT/TTS have their own `models/` trees |
| Runtime swappable | `runtime: whisper_pytorch \| whisper_ct2` in YAML |
| Device policy | `configs/languages.yaml`: `device: cuda`, `require_cuda: true`, `allow_cpu_fallback: false` |
| Git | Commit code + YAML + `MODEL_CARD.md`; **ignore** weight binaries under `models/` |

### What goes in each `models/` subfolder

| Path | Contents |
|------|----------|
| `models/upstream/` | Untouched Hub downloads |
| `models/finetuned/` | Train outputs (FP32/FP16 HF layout) |
| `models/export/` | Quantized / trimmed artifacts the Nano loads |

On device, mirror only what you need:

```text
/opt/peaktranslation/models/stt/
└── en-domain-v1-fp16/          # copy of stt/models/export/en-domain-v1-fp16
```

---

## 1. Hardware constraints (why most “Jetson Whisper” blogs lie to you)

| Spec | Nano Dev Kit | Implication |
|------|--------------|-------------|
| RAM | 4 GB **shared** CPU+GPU | Whisper + Marian + Piper + OS must coexist |
| GPU arch | Maxwell `sm_53` (CC **5.3**) | No Tensor Core INT8/FP16 paths modern stacks assume |
| JetPack ceiling | **4.6.3** (no JP5/JP6) | CUDA **10.2**, Python **3.6** on stock image |
| CUDA upgrade | Cannot put CUDA 11 on Nano | Blocks PyTorch 2.x and many CT2 GPU builds |

Most public demos labeled “Jetson” are **Orin Nano / Xavier** (JP5–JP6, CC 7.2–8.7, 8 GB+). Treat those numbers as **upper bounds**, not Nano numbers.

---

## 2. Whisper model ladder (official sizes)

From OpenAI Whisper model card (VRAM is A100-class guidance; Nano is tighter):

| Model | Params | English-only | Multilingual | Listed VRAM | Relative speed |
|-------|--------|--------------|--------------|-------------|----------------|
| tiny | 39M | `tiny.en` | `tiny` | ~1 GB | ~10× |
| base | 74M | `base.en` | `base` | ~1 GB | ~7× |
| small | 244M | `small.en` | `small` | ~2 GB | ~4× |
| medium | 769M | `medium.en` | `medium` | ~5 GB | ~2× |
| large / turbo | 1.5B / 809M | — | yes | ~6–10 GB | 1× / faster large |

### What actually works on Nano 4 GB (field reports)

| Choice | Fit | Notes |
|--------|-----|--------|
| **`tiny.en` / `tiny`** | Best for continuous pipeline | Lowest memory; start here when MT+TTS also resident |
| **`base.en` / `base`** | Workable | [whisper-edge](https://github.com/maxbbraun/whisper-edge): `base` is largest that fits **without modification**; ~10× realtime alone, ~1× while recording |
| `small+` | Avoid for full PeakTranslation | Competes fatally with Marian + Piper + queues |

English-only `.en` checkpoints usually beat multilingual twins on English WER for tiny/base — prefer them if source speech is English.

---

## 3. Runtime matrix (deep availability research)

### 3.1 PyTorch + openai-whisper / HF Whisper — **Nano MVP**

| Item | Status |
|------|--------|
| Availability | **Yes**, with Jetson-built Torch for CUDA 10.2 |
| Wheel source | NVIDIA Jetson PyTorch archives / community (e.g. Q-engineering) for JP4 |
| PyTorch 2.x | **No** on Nano (needs CUDA ≥ 11) |
| Python friction | Stock JP4 = 3.6; upstream Whisper wants ≥ 3.8 → use maintained Nano image, container, or Python-3.6-compatible fork ([whisper-edge](https://github.com/maxbbraun/whisper-edge)) |
| Accuracy | Full fine-tuned FP32/FP16 quality (within Torch limits) |
| Verdict | **Default production runtime for classic Nano** |

### 3.2 faster-whisper + CTranslate2 GPU — **Not Nano-ready by default**

| Item | Status |
|------|--------|
| PyPI aarch64 CT2 | **CPU-only** |
| GPU CT2 | Must **build from source**; community docs target Orin / CUDA 12 |
| Upstream CT2 CUDA requirement | Docs lean **CUDA ≥ 11** for GPU backend |
| INT8 on GPU vs Nano CC | CT2 GPU INT8 optimized for CC **≥ 7.0** (or 6.1). For **CC ≤ 6.0** (Nano = **5.3**), CT2 **falls back to float32** even if weights were stored int8 — **little GPU quant win on Nano** |
| jetson-containers `faster-whisper` | Requires **L4T ≥ 34** (Xavier/Orin era), not classic Nano R32 |
| Verdict | Convert on PC for **future Orin**; do not block Nano MVP on CT2-GPU |

### 3.3 whisper_trt / TensorRT — **Orin-class**

[NVIDIA-AI-IOT/whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) reports on **Orin Nano**: `base.en` ~3× faster, ~60% memory vs PyTorch. Relies on torch2trt / modern TensorRT. Classic Nano TensorRT stack is older and not the supported target.

| Verdict | Optional later if you migrate hardware; not the Nano Dev Kit plan |

### 3.4 ONNX Runtime GPU Whisper

ORT-GPU Jetson wheels are published mainly for **JP6 / Orin**. Building ORT+CUDA for JP4 Nano is high-effort, low reward vs Torch Whisper tiny/base.

---

## 4. Download locally (training PC)

```bash
# Hugging Face (recommended for fine-tune)
huggingface-cli download openai/whisper-tiny.en \
  --local-dir stt/models/upstream/openai-whisper-tiny.en

huggingface-cli download openai/whisper-base.en \
  --local-dir stt/models/upstream/openai-whisper-base.en

# Official whisper cache
python -c "import whisper; whisper.load_model('tiny.en', download_root='stt/models/upstream')"
```

Record Hub **revision hash** in `MODEL_CARD.md`. Never assume “latest” is reproducible.

Disk (approx download):

| Model | ~Download |
|-------|-----------|
| tiny.en | ~75 MB |
| base.en | ~145 MB |
| small.en | ~480 MB |

---

## 5. Fine-tuning research (quality without oversized models)

### Why fine-tune instead of jumping to `small`

On Nano, **domain adaptation of tiny/base** usually beats shipping `small` that OOMs or kills realtime when Marian/Piper share RAM.

### Data that matters on device

- Same mic class / gain / room noise as Jetson
- 16 kHz mono
- Accent + vocabulary of your product
- Augment: noise, gain, mild speed perturbation

### Methods that keep deploy size small

| Method | Deploy impact |
|--------|----------------|
| Full fine-tune tiny.en | Same tiny footprint |
| LoRA / PEFT adapters | Tiny base + small adapter; merge before export if runtime can’t load LoRA |
| Freeze encoder, tune decoder | Faster train; good for accent/vocab |

Train **only on PC/cloud GPU**. Nano is inference-only.

### Acceptance gates (keep accuracy)

| Gate | Suggestion |
|------|------------|
| Domain WER | Beat upstream on **your** test set |
| Quant / export ΔWER | Cap absolute rise (e.g. ≤ 1–2 points — set product gate) |
| Nano latency | Median utterance within budget with VAD on |

---

## 6. Weight reduction / quantization (what helps on Nano)

Ordered by **accuracy risk ↑** and **Nano usefulness**:

| Rank | Technique | Accuracy | Nano effect |
|------|-----------|----------|-------------|
| 1 | Use `tiny.en` not `base`/`small` | Some WER↑ | **Largest real win** |
| 2 | FP16 weights / autocast if Torch build supports | Usually small | Memory/latency if stable on your wheel |
| 3 | Decode: `beam_size=1`, fixed language, no fallbacks | Small | Latency |
| 4 | Shorter VAD max utterance | Indirect | Less encoder work |
| 5 | CT2 float16/int8 | Mild for int8 | Great on **Orin CPU/GPU**; on Nano GPU CT2/INT8 **limited by CC 5.3** |
| 6 | TensorRT | Varies | Orin path |

### Critical CT2 GPU note for Nano (CC 5.3)

From [CTranslate2 quantization docs](https://opennmt.net/CTranslate2/quantization.html): on GPU, compute capability **≤ 6.0** maps int8/float16 compute types back toward **float32**. Even a successful CT2-GPU build on Nano would **not** get modern INT8 Tensor-Core speedups.

**Implication:** shrinking for Nano = **smaller architecture + careful Torch deploy**, not “int8 magic.”

### Optional export (PC) for later devices

```bash
ct2-transformers-converter \
  --model stt/models/finetuned/en-domain-v1 \
  --output_dir stt/models/export/en-domain-v1-ct2-int8 \
  --quantization int8
```

Keep FP16/PyTorch export as Nano artifact; keep CT2 for Orin/CPU experiments.

---

## 7. Continuous listening design (STT-specific)

Whisper is **not** a true streaming ASR by default. On Nano:

1. Capture thread always on.
2. VAD (WebRTC / Silero CPU / energy) cuts utterances.
3. Whisper worker transcribes segment → pushes **sentence** to Q1.
4. GPU lock shared with Marian if both on CUDA.

Chunk guidance from whisper-edge-style demos: multi-second chunks (e.g. ~5–10 s) trade latency vs stability. For “sentence” UX, prefer **silence endpointing** (~300–500 ms) with `max_utterance_ms` cap.

Partials: possible with sliding windows but costly on Nano — MVP = **final sentences only** into Q1.

---

## 8. Recommended Nano STT stack (decision)

```text
Download:  openai/whisper-tiny.en (or base.en if WER demands)
Fine-tune: HF/PEFT on PC with domain audio
Export:    PyTorch (FP16 if validated)
Runtime:   Jetson PyTorch CUDA + language profile in languages.yaml
Queue:     push finals → transcript_queue (Q1)
Avoid:     CT2-GPU, whisper_trt, small+, concurrent huge models
```

### Memory budget sketch (full product)

| Piece | Rough target |
|-------|----------------|
| OS + CUDA | 1.0–1.5 GB |
| Whisper tiny/base | 0.5–1.0 GB |
| Marian (GPU primary) | 0.3–0.8 GB |
| Marian CPU fallback (optional) | 0.3–0.6 GB if activated |
| Piper CPU | 0.1–0.3 GB |
| Queues / spill | keep text in RAM; WAV on disk |

If OOM: drop Whisper to `tiny.en`, enable Marian **CPU fallback**, shorten max utterance — **never** move STT to CPU.

---

## 9. Language switch / retrain architecture

Profiles live in `stt/configs/languages.yaml`:

```text
profile_id → upstream + artifact + runtime + decode.lang
```

| Task | Action |
|------|--------|
| Better English domain | New finetuned artifact; bump profile version |
| Non-English source | Multilingual tiny/base profile; set `decode.language` |
| Swap runtime later (Orin CT2) | Change `runtime:` only; same interface |

Orchestrator never imports Whisper directly — only `STTEngine` from registry.

---

## 10. Smoke / measure checklist on device

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
tegrastats
# transcribe fixed wav; log seconds and RSS
```

Log: `audio_s | stt_ms | peak_ram_mb | wer_sample`.

---

## 11. Sources (research trail)

- OpenAI Whisper model table — [github.com/openai/whisper](https://github.com/openai/whisper)
- Jetson Nano Whisper port — [maxbbraun/whisper-edge](https://github.com/maxbbraun/whisper-edge)
- CT2 quantization / CC fallbacks — [opennmt.net/CTranslate2/quantization.html](https://opennmt.net/CTranslate2/quantization.html)
- Orin TensorRT Whisper — [NVIDIA-AI-IOT/whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt)
- Jetson PyTorch / CUDA 10.2 limits — Q-engineering Jetson Nano PyTorch guides
- PeakTranslation pipeline — [`../04-realtime-queue-pipeline.md`](../04-realtime-queue-pipeline.md)

---

## 12. Bottom line

For **Jetson Nano Dev Kit**, production STT is **fine-tuned Whisper tiny/base.en on Jetson PyTorch CUDA only** (`require_cuda: true`, no CPU fallback), continuously fed by VAD into a transcript queue. Treat CTranslate2 GPU INT8 and TensorRT Whisper as **next-gen Jetson (Orin)** optimizations, not prerequisites for Nano.
