# PC pipeline (Jetson deferred)

## What you run next

### A) Download Piper voices (TTS)

```bash
cd /home/prieyan/weeb/PeakTranslation
bash tts/scripts/download_voices.sh
```

### B) Install Piper Python package (once)

```bash
export PYTHONPATH="$PWD/venv/lib/python3.11/site-packages"
PY=/home/prieyan/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
$PY -m pip install --target "$PWD/venv/lib/python3.11/site-packages" piper-tts onnxruntime
```

### C) Smoke TTS only

```bash
$PY - <<'PY'
from pathlib import Path
from piper import PiperVoice
voice = PiperVoice.load("tts/models/export/hi_official_v1/voice.onnx")
with open("/tmp/hi_test.wav", "wb") as f:
    voice.synthesize("नमस्ते, यह एक परीक्षण है।", f)
print("wrote /tmp/hi_test.wav")
PY
aplay /tmp/hi_test.wav   # or: paplay /tmp/hi_test.wav
```

### D) Full PC pipeline

Text only (skip STT — good first test):

```bash
export PYTHONPATH="$PWD/venv/lib/python3.11/site-packages"
PY=/home/prieyan/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11

$PY pipeline/run_pc.py --text "Hello, how are you?"
```

WAV → STT → MT → TTS:

```bash
$PY pipeline/run_pc.py --wav /path/to/english_speech.wav
```

Save without playing:

```bash
$PY pipeline/run_pc.py --text "Good morning" --out /tmp/out.wav --no-play
```

## Already done
- STT export: `stt/models/export/en-hi-base-v1-fp16`
- MT export: `translate/models/export/en-hi-v1-fp16` (+ CT2 int8)
