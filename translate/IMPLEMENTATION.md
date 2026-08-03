# Translation implementation runbook (PC → Jetson later)

## Current run: English → Hindi

| Item | Value |
|------|--------|
| Base | `Helsinki-NLP/opus-mt-en-hi` |
| Fine-tune on | Local PC GPU |
| Data | opus100 / IITB en-hi bitext |
| Primary export | `models/export/en-hi-v1-fp16` (CUDA on Jetson) |
| Fallback export | `models/export/en-hi-v1-ct2-int8` (CPU, optional) |

## Commands

```bash
cd /home/prieyan/weeb/PeakTranslation
export PYTHONPATH="$PWD/venv/lib/python3.11/site-packages"
PY=/home/prieyan/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11

$PY -m pip install -r translate/requirements.txt
$PY translate/scripts/download_upstream.py
$PY translate/data/scripts/prepare_en_hi.py --max-pairs 60000
$PY translate/scripts/train.py
$PY translate/scripts/export_fp16.py
$PY translate/scripts/export_ct2_int8.py   # optional
```
