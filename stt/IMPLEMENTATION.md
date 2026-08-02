# STT implementation runbook (PC first → Jetson later)

## Your setup (locked in)

| Item | Choice |
|------|--------|
| Target device later | Jetson Nano Dev Kit 4GB |
| Build/train now | **This PC** |
| Model | `openai/whisper-base` (multilingual — needed for **Hindi**) |
| Languages | English + Hindi |
| Fine-tune goal | Real speech → text; scream/laugh/moan/cry → **empty** (no emotion words) |
| Jetson STT device | **CUDA only** (no CPU fallback) |
| Mic | On Nano later; PC uses WAV / manifests now |

## Security

- HF token lives in repo-root `.env` (gitignored).
- **Revoke/rotate** the token you pasted in chat on https://huggingface.co/settings/tokens and put the new one in `.env`.
- Never commit `.env` or `models/**` weights.

## PC prerequisites

1. **NVIDIA GPU + drivers** (`nvidia-smi` must work). This machine currently reported no driver — fix that before train.
2. **Python 3.10–3.12** recommended (3.14 often breaks ML wheels).
3. Accept dataset access if prompted (FLEURS is public; Common Voice is no longer on HF).

```bash
cd /home/prieyan/weeb/PeakTranslation
python3.11 -m venv .venv   # or 3.10/3.12
source .venv/bin/activate
pip install -U pip
pip install -r stt/requirements.txt
```

## Steps

### 1) Download Whisper base

```bash
python stt/scripts/download_upstream.py
# → stt/models/upstream/openai-whisper-base/
```

### 2) Build speech dataset (en + hi)

Common Voice is **no longer on Hugging Face** (moved to Mozilla Data Collective).
The prep script now uses **Google FLEURS** (`en_us` + `hi_in`).

```bash
# Same command as before
python stt/data/scripts/prepare_common_voice.py --max-en 2000 --max-hi 2000
# → stt/data/processed/en-hi/{train,val,test}/manifest.jsonl
# → audio under stt/data/raw/fleurs/{en,hi}/
```

FLEURS has roughly ~1–2k clips per language; larger `--max-*` just takes everything available.

### 3) Add non-speech rejection clips (your fine-tune intent)

Put WAVs of screaming / laughing / moaning / crying (no words) in:

`stt/data/raw/nonspeech/`

Then:

```bash
python stt/data/scripts/merge_nonspeech.py
```

Each gets `text=""` so the model learns **silence/empty** instead of inventing words.

### 4) Fine-tune

```bash
python stt/scripts/train.py --config stt/configs/train_en_hi_base.yaml
# → stt/models/finetuned/en-hi-base-v1/
```

### 5) Evaluate

```bash
python stt/scripts/run_eval.py
# prints speech WER + nonspeech_reject_rate
```

### 6) Export FP16 for Jetson package

```bash
python stt/scripts/export_fp16.py
# → stt/models/export/en-hi-base-v1-fp16/
```

### 7) Smoke on a WAV

```bash
python stt/scripts/smoke_stt.py --wav /path/to/sample.wav --language en
python stt/scripts/smoke_stt.py --wav /path/to/hindi.wav --language hi
```

## What goes to GitHub later (not weights)

Push: code, configs, docs, manifests **paths/scripts** — not `models/` binaries, not `.env`.

On Jetson after clone: copy `stt/models/export/en-hi-base-v1-fp16/` over Ethernet.

## What you still owe for a strong reject behavior

Common Voice alone teaches en/hi speech. **Non-speech empty labels need your clips** (or we can later script synthetic noise — worse for scream/laugh). Aim for at least a few hundred varied non-speech clips if you can.

