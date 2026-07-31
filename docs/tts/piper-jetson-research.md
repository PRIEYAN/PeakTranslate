# TTS Deep Research — Piper on Jetson Nano Dev Kit

Research for PeakTranslation speech synthesis: download / train / shrink Piper voices and run them efficiently on the **Jetson Nano Developer Kit**, with a voice registry keyed by MT target language.

Companion: [`../../tts/README.md`](../../tts/README.md) · [`../../tts/configs/voices.yaml`](../../tts/configs/voices.yaml)

---

## 0. Production-grade loosely coupled file structure

TTS is a **standalone package**. The pipeline only calls `TTSEngine`. MT exposes `tgt` lang; TTS registry maps that to a voice under **`models/`**. No Piper paths hardcoded in STT or Marian code.

**All voice ONNX files and training checkpoints are stored under `models/`.**

```text
tts/                                      # standalone TTS package (loosely coupled)
├── README.md
├── configs/
│   ├── voices.yaml                       # lang → voice registry — switch HERE
│   ├── train_ta_custom.yaml
│   └── synthesis.yaml
├── data/
│   ├── raw/                              # studio / mic recordings (gitignored)
│   ├── processed/
│   │   ├── ta_custom_v1/                 # LJSpeech-like
│   │   │   ├── wavs/
│   │   │   └── metadata.csv
│   │   └── hi_custom_v1/
│   └── scripts/
│       ├── normalize_audio.py
│       ├── build_ljspeech.py
│       └── validate_dataset.py
├── models/                               # ★ ALL model / voice files stored here
│   ├── upstream/                         # official Piper downloads (gitignored)
│   │   ├── en_US-lessac-medium.onnx
│   │   ├── en_US-lessac-medium.onnx.json
│   │   ├── ta_IN-<voice>-medium.onnx
│   │   └── ta_IN-<voice>-medium.onnx.json
│   ├── checkpoints/                      # PyTorch Lightning ckpts while training (gitignored)
│   │   └── ta_custom_v1/
│   │       └── epoch=N.ckpt
│   ├── finetuned/                        # raw export before optimize (gitignored)
│   │   └── ta_custom_v1/
│   │       ├── ta_custom_v1.onnx
│   │       ├── ta_custom_v1.onnx.json
│   │       └── MODEL_CARD.md
│   ├── export/                           # production voices for Nano (gitignored)
│   │   ├── ta_custom_v1/
│   │   │   ├── voice.onnx                # onnxsim / optional quant
│   │   │   ├── voice.onnx.json
│   │   │   ├── MODEL_CARD.md
│   │   │   └── metrics.json              # RTF, listening notes
│   │   ├── hi_official_v1/
│   │   │   ├── voice.onnx
│   │   │   └── voice.onnx.json
│   │   └── en_lessac_medium/
│   │       ├── voice.onnx
│   │       └── voice.onnx.json
│   └── .gitkeep
├── src/
│   ├── train_prep.py
│   ├── export_onnx.py
│   ├── optimize_onnx.py
│   ├── evaluate_rtf.py
│   ├── runtime/                          # adapters — orchestrator uses TTSEngine only
│   │   ├── interface.py                  # TTSEngine protocol
│   │   ├── piper_cpu.py
│   │   └── registry.py                   # voices.yaml → load models/export/<voice>
│   └── jetson/
│       ├── smoke_tts.py
│       └── bench_rtf.py
├── scripts/
│   ├── download_voices.sh                # → models/upstream/
│   ├── train_voice.sh                    # → models/checkpoints/ + finetuned/
│   ├── export_and_optimize.sh            # → models/export/
│   └── package_for_nano.sh               # rsync models/export/<voice> → device
└── artifacts/
    └── tts-ta-custom-v1-nano.tar.gz
```

### Loose coupling rules

| Rule | Practice |
|------|----------|
| Voice = folder | Each production voice is `models/export/<voice_id>/{voice.onnx,voice.onnx.json}` |
| Lang switch via YAML | `lang_to_voice[tgt]` → artifact path; MT only provides `tgt` code |
| No cross-imports | STT/MT never open ONNX files |
| One runtime adapter | `piper_cpu` today; swap backend later behind `TTSEngine` |
| Git | Ignore ONNX/ckpt binaries under `models/`; commit YAML + cards |

### What goes in each `models/` subfolder

| Path | Contents |
|------|----------|
| `models/upstream/` | Downloaded official `.onnx` + `.onnx.json` |
| `models/checkpoints/` | Training `.ckpt` (PC only) |
| `models/finetuned/` | First ONNX export from training |
| `models/export/` | Optimized voices the Jetson loads |

On device:

```text
/opt/peaktranslation/models/tts/
├── ta_custom_v1/voice.onnx
├── ta_custom_v1/voice.onnx.json
└── en_lessac_medium/...
```

---

## 1. Why Piper for Nano

| Need | Piper |
|------|--------|
| Offline, no cloud | Yes |
| ARM-friendly | Designed for Raspberry Pi–class CPUs; **arm64** binaries published |
| Format | VITS → **ONNX**; inference via **ONNX Runtime** |
| Footprint | ~10–100 MB per voice by quality tier |
| Speed | Typically **RTF ≪ 1** on modest CPUs (often far faster than realtime) |
| Jetson fit | Leave **GPU for Whisper**; TTS on **CPU** matches PeakTranslation architecture |

Alternatives (XTTS, Bark, large Tacotron stacks) are generally too heavy for continuous Nano pipelines.

---

## 2. Voice quality tiers (size vs quality research)

Community / Piper docs consistent picture:

| Tier | Sample rate | Approx ONNX size | Role on Nano |
|------|-------------|------------------|--------------|
| `x_low` | 16 kHz | ~10 MB | Tightest RAM; telephony-ish |
| `low` | 16 kHz | ~20 MB | Short assistant phrases |
| **`medium`** | 22.05 kHz | ~30–60 MB | **Default sweet spot** |
| `high` | 22.05 kHz | ~100 MB+ | Better prosody; more CPU/RAM |

Published CPU RTF anecdotes for Piper are often **orders of magnitude faster than realtime** on desktop CPUs; on Jetson Nano expect **still usually RTF &lt; 1** for medium, but **measure on-device** under concurrent Whisper/Marian load.

**Nano recommendation:** ship **medium** if RTF/queue stable; fall back to **low** if playback queue (Q3) backs up or RAM is tight.

---

## 3. Runtime availability (deep)

### 3.1 Piper + ONNX Runtime **CPU** — **Nano default**

| Item | Status |
|------|--------|
| Piper `arm64` release tarball | **Yes** ([piper releases](https://github.com/rhasspy/piper/releases) `piper_arm64.tar.gz`) |
| `pip install onnxruntime` aarch64 | **Yes** (CPU) |
| `piper-tts` Python | **Yes** (same models) |
| GPU required? | **No** |
| Verdict | **Production path for classic Nano** |

Verify:

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expect: ['CPUExecutionProvider'] (and maybe others non-CUDA)
```

### 3.2 Piper `--cuda` / `onnxruntime-gpu` — **Not Nano drop-in**

| Item | Status |
|------|--------|
| Upstream docs | Install `onnxruntime-gpu` + `--cuda` |
| Jetson reality | CUDA ORT wheels are **Jetson/JP-specific**; public JP6/Orin wheels ≠ JP4 Nano |
| Classic Nano | Building ORT-GPU for CUDA 10.2 / CC 5.3 is high effort |
| Need? | **Low** — CPU Piper is already fast enough for most queue designs |
| Verdict | Skip for Nano MVP; revisit on Orin if CPU is saturated |

### 3.3 Training runtime (PC only)

| Piece | Where |
|-------|--------|
| `piper_train` / PyTorch Lightning | **PC GPU** |
| Checkpoints | `rhasspy/piper-checkpoints` on Hub |
| Export | `python -m piper_train.export_onnx` → `.onnx` + `.onnx.json` |
| Nano | Inference only |

---

## 4. Download voices locally

Official voices: Hugging Face `rhasspy/piper-voices` (see [VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md)).

```bash
# Example pattern (replace with your lang/voice/quality URLs)
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0
DIR=tts/models/upstream

wget -O $DIR/en_US-lessac-medium.onnx \
  $BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O $DIR/en_US-lessac-medium.onnx.json \
  $BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Always download **both** `.onnx` and `.onnx.json`. Read each voice’s **MODEL_CARD** / license before production use.

Promote to production layout:

```text
tts/models/export/<voice_id>/
  voice.onnx
  voice.onnx.json
  MODEL_CARD.md
  metrics.json
```

---

## 5. Fine-tune / custom voice research

### 5.1 When to fine-tune

| Situation | Action |
|-----------|--------|
| Official voice exists for `tgt` lang | Prefer download medium/low |
| Missing language / brand voice / accent | Fine-tune from Piper checkpoint |
| Tiny dataset | Fine-tune ≫ train from scratch |

### 5.2 Data

- LJSpeech-like: `wavs/` + `metadata.csv`
- Clean, consistent mic; match quality tier sample rate
- Single speaker preferred for MVP

### 5.3 Train outline (PC)

Per Piper [TRAINING.md](https://github.com/rhasspy/piper/blob/master/TRAINING.md):

1. Preprocess dataset → `config.json`, `dataset.jsonl`, cached audio
2. `python3 -m piper_train ... --resume_from_checkpoint <same quality ckpt>`
3. Export ONNX; copy `config.json` → `model.onnx.json`

Quality flag must match checkpoint tier (`low` / `medium` / `high`).

### 5.4 Acceptance

| Gate | Check |
|------|--------|
| Listening | Native review of 20+ domain sentences |
| RTF on Nano | synthesis_s / audio_s **&lt; 1** under load |
| Alignment | No frequent cutoffs / garbage phonemes |

---

## 6. Shrink / optimize weights (keep quality)

Piper voices are already compact. Extra steps:

| Step | Tool | Quality risk | Nano value |
|------|------|--------------|------------|
| 1 | Prefer **low/medium** tier | Audible | High |
| 2 | **onnx-simplifier** (`onnxsim`) after export | Negligible | Recommended |
| 3 | Keep FP32 ONNX (default Piper) | Baseline | Default |
| 4 | ORT dynamic INT8 quant | Possible harshness | Only if measured need + listen gate |
| 5 | Tune `length_scale` / silence | Prosody only | Latency / pacing |

```bash
python3 -m piper_train.export_onnx /path/model.ckpt /path/voice.unopt.onnx
onnxsim /path/voice.unopt.onnx tts/models/export/ta_v1/voice.onnx
cp /path/training/config.json tts/models/export/ta_v1/voice.onnx.json
```

**Do not** chase GPU ORT quant for Nano if CPU medium already meets RTF.

---

## 7. Language / voice switch architecture

MT `tgt` language selects voice — no hardcoded Piper paths in the orchestrator.

```text
pairs.yaml tgt=ta ──► voices.yaml lang_to_voice[ta] ──► export/ta_*/voice.onnx
```

```yaml
# conceptual
lang_to_voice:
  ta: ta_custom_v1
  hi: hi_official_v1
voices:
  ta_custom_v1:
    artifact: models/export/ta_custom_v1
    runtime: piper_cpu
```

### Add a target language

1. MT pair artifact ready (`translate/`)
2. Download or train Piper voice for that lang
3. Add `voices:` + `lang_to_voice:` entries
4. Restart or hot-reload registry

Worker API:

```python
set_voice(voice_id)
synthesize_to_file(text, wav_path)  # push path to Q3
```

Keep the ONNX session **loaded** for the active voice; switching languages may reload once.

---

## 8. Continuous pipeline role (serial playback)

```text
Q2 translations ──pop──► Piper (warm, CPU) ──push WAV path──► Q3
Q3 ──pop──► playback blocks until WAV done ──► next WAV only then
```

Design rules from product constraints:

- Piper worker **always active** (loop on Q2)
- Write WAV under spill dir (swap/disk), queue **paths** not giant PCM blobs
- **One** speaker job at a time
- If Q3 fills, Piper blocks → backpressure to MT → STT

Raw pipe pattern (optional, lower disk IO):

```bash
echo "text" | piper -m voice.onnx --output-raw | \
  aplay -r 22050 -f S16_LE -t raw -
```

For queue decoupling and “only next after finished,” **file or bounded buffer + blocking play** is clearer than unbounded pipes.

---

## 9. Jetson install sketch

```bash
# Binary route
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz

# Or Python
python3 -m pip install onnxruntime piper-tts

# Voices
rsync -avP tts/models/export/ nano:/opt/peaktranslation/models/tts/

# Smoke
echo "Hello" | ./piper -m /opt/.../en_lessac_medium/voice.onnx -f /tmp/t.wav
aplay /tmp/t.wav
```

Benchmark under full pipeline:

```text
tts_ms | audio_dur_ms | rtf | q3_depth
```

---

## 10. Interaction with STT/MT memory

| Stage | Device | Note |
|-------|--------|------|
| Whisper | GPU | Dominates RAM |
| Marian | Prefer CPU CT2 int8 | Avoid fighting Whisper |
| Piper | CPU | Small; medium voice OK |

If RAM still tight: drop Piper to **low**, ensure only one voice loaded, delete WAV after play.

---

## 11. Decision summary

| Question | Answer |
|----------|--------|
| Runtime on Nano Dev Kit? | **Piper + ONNX Runtime CPU** |
| GPU ORT? | Skip on classic Nano |
| Best quality/size? | **medium**, else **low** |
| Shrink path? | Tier choice + `onnxsim`; INT8 only with listen gate |
| Language switch? | `voices.yaml` keyed by MT `tgt` |
| Train where? | PC GPU → export ONNX → copy to Nano |

---

## 12. Sources

- [rhasspy/piper](https://github.com/rhasspy/piper) README, TRAINING.md, VOICES.md
- Piper releases (`piper_arm64.tar.gz`)
- Hugging Face `rhasspy/piper-voices` / `piper-checkpoints`
- Jetson Piper CPU usage reports (Orin blogs for GPU ORT — contrast with Nano)
- PeakTranslation queue design — [`../04-realtime-queue-pipeline.md`](../04-realtime-queue-pipeline.md)
- Stage guide — [`../../tts/README.md`](../../tts/README.md)

---

## 13. Bottom line

On the **Jetson Nano Dev Kit**, Piper is the right TTS: **CPU ONNX**, medium/low voices, registry-switched by target language, continuously draining Q2 into Q3 with **strictly serial playback**. Leave **CUDA to Whisper (required)** and **Marian (preferred, CPU fallback OK)**.
