# TTS — Piper: Download, Fine-Tune, Quantize, Jetson Runtime

Text-to-speech stage for PeakTranslation. Runs on **CPU** on the Jetson Nano (by design). Train/export voices on a **PC/GPU**; deploy `.onnx` + `.onnx.json` per language/voice. Switching output language = switch **voice id** from the active MT target lang.

---

## 1. Goals

| Goal | Approach |
|------|----------|
| Natural speech for target langs | Official Piper voices or fine-tuned custom ONNX |
| Low RAM / always-on worker | Prefer **low/medium** quality voices |
| Shrink further if needed | onnx-simplifier; optional ORT dynamic quant (validate MOS) |
| Easy language switch | Voice registry keyed by `lang_code` / locale |

---

## 2. Runtime availability (research summary)

| Runtime | Classic Nano Dev Kit | Notes |
|---------|----------------------|--------|
| **Piper binary + ONNX Runtime CPU** | **Yes — MVP default** | Proven pattern on Jetson family. Install `onnxruntime` (CPU) aarch64. |
| **piper-tts Python package** | Yes | Same ONNX models underneath. |
| **ONNX Runtime GPU (`--cuda`)** | **Not drop-in on Nano** | Jetson CUDA ORT wheels target newer JP (e.g. Orin/JP6). Do **not** require GPU for Nano MVP. |
| **PyTorch VITS (pre-export)** | Train only | Training/fine-tune on PC; **never** run training on Nano. |
| **Other TTS (Coqui, etc.)** | Optional later | Keep behind same `TTSEngine` interface. |

### Nano production choice

```text
PC:   download voice OR fine-tune Piper checkpoint → export_onnx → onnxsim
Nano: piper + onnxruntime CPU → write WAV → push path to Q3
```

---

## 3. Production-grade folder structure

```text
tts/
├── README.md
├── configs/
│   ├── voices.yaml                    # language → voice registry (switch here)
│   ├── train_ta_custom.yaml
│   └── synthesis.yaml                 # length_scale, noise_scale, sentence_silence
├── data/
│   ├── raw/                           # studio/mic recordings (gitignored)
│   ├── processed/
│   │   ├── ta_custom_v1/              # LJSpeech-like layout for Piper train
│   │   │   ├── wavs/
│   │   │   ├── metadata.csv           # id|text  (Piper/LJSpeech convention)
│   │   │   └── ...
│   │   └── hi_custom_v1/
│   └── scripts/
│       ├── normalize_audio.py         # peak/lufs, mono, target sr
│       ├── build_ljspeech.py
│       └── validate_dataset.py
├── models/
│   ├── upstream/                      # downloaded official voices (gitignored)
│   │   ├── en_US-lessac-medium.onnx
│   │   ├── en_US-lessac-medium.onnx.json
│   │   ├── ta_IN-....onnx
│   │   └── ta_IN-....onnx.json
│   ├── checkpoints/                   # PyTorch Lightning ckpts while training
│   │   └── ta_custom_v1/
│   ├── finetuned/                     # exported but not yet optimized
│   │   └── ta_custom_v1/
│   │       ├── ta_custom_v1.onnx
│   │       ├── ta_custom_v1.onnx.json
│   │       └── MODEL_CARD.md
│   └── export/                        # production voices for Nano
│       ├── ta_custom_v1/
│       │   ├── voice.onnx             # simplified / optionally quantized
│       │   ├── voice.onnx.json
│       │   ├── MODEL_CARD.md
│       │   └── metrics.json           # RTF, duration err, listening notes
│       ├── hi_official_v1/
│       └── en_lessac_medium/
├── src/
│   ├── train_prep.py                  # wrap piper_train preprocessing
│   ├── export_onnx.py                 # thin wrapper around piper_train.export_onnx
│   ├── optimize_onnx.py               # onnxsim (+ optional quant)
│   ├── evaluate_rtf.py
│   ├── runtime/
│   │   ├── interface.py               # TTSEngine protocol
│   │   ├── piper_cpu.py               # Nano adapter
│   │   └── registry.py                # voices.yaml → model path
│   └── jetson/
│       ├── smoke_tts.py
│       └── bench_rtf.py
├── scripts/
│   ├── download_voices.sh
│   ├── train_voice.sh
│   ├── export_and_optimize.sh
│   └── package_for_nano.sh
└── artifacts/
    └── tts-ta-custom-v1-nano.tar.gz
```

**Rule:** every production voice is a directory with `voice.onnx` + `voice.onnx.json` + card. Registry only stores paths and lang codes.

---

## 4. Language / voice switch architecture

MT target language selects TTS voice — no hardcoding in Piper worker.

```yaml
# tts/configs/voices.yaml
default_voice: ta_custom_v1

# Map ISO lang (from MT tgt) → voice id
lang_to_voice:
  ta: ta_custom_v1
  hi: hi_official_v1
  en: en_lessac_medium

voices:
  ta_custom_v1:
    lang: ta
    locale: ta_IN
    quality: medium          # low | medium | high
    artifact: models/export/ta_custom_v1
    runtime: piper_cpu
    sample_rate: 22050
    synthesis:
      length_scale: 1.0
      noise_scale: 0.667
      noise_w: 0.8
      sentence_silence: 0.2

  hi_official_v1:
    lang: hi
    locale: hi_IN
    quality: medium
    artifact: models/export/hi_official_v1
    runtime: piper_cpu

  en_lessac_medium:
    lang: en
    locale: en_US
    quality: medium
    artifact: models/export/en_lessac_medium
    runtime: piper_cpu
```

### Switch language

```text
MT pair tgt=hi ──► voices.lang_to_voice["hi"] ──► load hi_official_v1 ONNX
```

### Train / add a language

1. Get or record data for target lang → `data/processed/<voice_id>/`.
2. Fine-tune from a Piper checkpoint (same quality tier) on PC GPU.
3. Export ONNX → optimize → `models/export/<voice_id>/`.
4. Add `voices:` + `lang_to_voice:` entries.
5. Ensure `translate/configs/pairs.yaml` `tgt` matches the lang key.

```python
class TTSEngine(Protocol):
    def set_voice(self, voice_id: str) -> None: ...
    def synthesize_to_file(self, text: str, out_wav: str) -> WavMeta: ...
```

---

## 5. Download official voices locally

Voices live on Hugging Face (`rhasspy/piper-voices`). Example:

```bash
# scripts/download_voices.sh sketch
VOICE_DIR=tts/models/upstream
BASE=https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0

# Pick URLs from https://github.com/rhasspy/piper/blob/master/VOICES.md
wget -O $VOICE_DIR/en_US-lessac-medium.onnx \
  "$BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
wget -O $VOICE_DIR/en_US-lessac-medium.onnx.json \
  "$BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

Promote into export layout:

```bash
mkdir -p tts/models/export/en_lessac_medium
cp tts/models/upstream/en_US-lessac-medium.onnx \
   tts/models/export/en_lessac_medium/voice.onnx
cp tts/models/upstream/en_US-lessac-medium.onnx.json \
   tts/models/export/en_lessac_medium/voice.onnx.json
```

Prefer **low** or **medium** on Nano for RTF & RAM; use **high** only if RTF stays &lt; 1 and queue does not back up.

---

## 6. Fine-tune / train custom voice (PC GPU)

Follow Piper’s training guide (`TRAINING.md`): dataset prep → `piper_train` → `export_onnx`.

### Dataset

- Clean single-speaker (or documented multi-speaker) audio.
- Match target sample rate for quality tier (low 16 kHz / medium-high 22.05 kHz).
- LJSpeech-style `metadata.csv` + `wavs/`.

### Fine-tune from checkpoint

Use checkpoints from `rhasspy/piper-checkpoints` with the **same quality** tier. Fine-tuning beats training from scratch for small datasets.

```bash
./tts/scripts/train_voice.sh ta_custom_v1
```

### Export

```bash
python3 -m piper_train.export_onnx \
  /path/to/model.ckpt \
  tts/models/finetuned/ta_custom_v1/ta_custom_v1.onnx

cp /path/to/training_dir/config.json \
  tts/models/finetuned/ta_custom_v1/ta_custom_v1.onnx.json
```

---

## 7. Reduce weights / optimize (keep quality)

Piper voices are already fairly small ONNX graphs. Extra shrink options:

| Step | Method | Quality risk | Use on Nano |
|------|--------|--------------|-------------|
| 1 | Choose **low/medium** not high | Audible but often OK | Preferred |
| 2 | **onnx-simplifier** (`onnxsim`) | Negligible | Always after export |
| 3 | Keep FP32 ONNX (Piper default) | Baseline | Default |
| 4 | ORT **dynamic INT8** quant | Possible artifacts | Only if RTF/RAM force it + listen test |
| 5 | Shorter sentences from MT (`max_new_tokens`) | Indirect | Helps latency |

### Optimize script

```bash
python tts/src/optimize_onnx.py \
  --src tts/models/finetuned/ta_custom_v1/ta_custom_v1.onnx \
  --dst tts/models/export/ta_custom_v1/voice.onnx

cp tts/models/finetuned/ta_custom_v1/ta_custom_v1.onnx.json \
   tts/models/export/ta_custom_v1/voice.onnx.json
```

### Optional INT8 (validate carefully)

```bash
# Conceptual — use onnxruntime quantization tools on PC
python tts/src/optimize_onnx.py --quantize-dynamic --arm64 ...
```

Gate: native listener preference / informal MOS on 20 sentences; RTF on Nano CPU. If harsh/robotic, ship FP32 simplified ONNX.

---

## 8. Build / package runtime for Jetson Nano

### On Nano

```bash
python3 -m pip install onnxruntime piper-tts   # or install piper binary
# Do NOT assume onnxruntime-gpu works on classic Nano
```

Verify providers:

```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Expect CPUExecutionProvider
```

Copy voices:

```bash
./tts/scripts/package_for_nano.sh ta_custom_v1 nano:/opt/peaktranslation/models/tts/
```

Device config:

```yaml
tts:
  runtime: piper_cpu
  voices_config: /opt/peaktranslation/tts/voices.yaml
  # lang_to_voice resolved from MT tgt
```

Smoke:

```bash
python tts/src/jetson/smoke_tts.py --voice ta_custom_v1 --text "<sample>" --out /tmp/t.wav
aplay /tmp/t.wav
python tts/src/jetson/bench_rtf.py --voice ta_custom_v1
```

**RTF target:** synthesis time / audio duration &lt; 1.0 (ideally ≤ 0.5) so Q3 does not starve playback.

### Continuous pipeline role

```text
pop Q2 translation → Piper synthesize_to_file(spill/utt_id.wav) → push Q3 path
playback worker: play one WAV to completion, then next
```

Keep Piper process/worker **warm** (model loaded once); do not reload ONNX per sentence.

---

## 9. Coordination with STT / MT language switching

| Change | STT | MT | TTS |
|--------|-----|----|-----|
| New source accent only | New STT profile | — | — |
| New target language | — | New pair artifact + `pairs.yaml` | New voice + `lang_to_voice` |
| Same langs, better voice | — | — | Replace `models/export/...` |

Shared pipeline config should reference:

```yaml
translation:
  default_pair: en-ta
tts:
  # voice resolved from pair.tgt via voices.yaml
```

---

## 10. Checklist

- [ ] Official or custom voice under `models/export/<id>/`
- [ ] `voices.yaml` maps every MT `tgt` you ship
- [ ] onnxsim (and optional quant) with listening gate
- [ ] Nano RTF &lt; 1 with CPU ORT
- [ ] Worker keeps model loaded; writes WAV paths for Q3
- [ ] Playback remains serial (one WAV at a time)

---

## Related

- [stt/README.md](../stt/README.md) — Whisper
- [translate/README.md](../translate/README.md) — MarianMT
- [docs/04-realtime-queue-pipeline.md](../docs/04-realtime-queue-pipeline.md)
