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

`pipeline/_bootstrap.py` fixes up `sys.path` at import time, so once
dependencies are installed (step B / step E) you just run the entrypoint
with `venv/bin/python` directly — no `export PYTHONPATH=...` and no reaching
for the raw uv interpreter path every time.

Text only (skip STT — good first test):

```bash
venv/bin/python pipeline/run_pc.py --text "Hello, how are you?"
```

WAV → STT → MT → TTS:

```bash
venv/bin/python pipeline/run_pc.py --wav /path/to/english_speech.wav
```

Save without playing:

```bash
venv/bin/python pipeline/run_pc.py --text "Good morning" --out /tmp/out.wav --no-play
```

## Already done
- STT export: `stt/models/export/en-hi-base-v1-fp16`
- MT export: `translate/models/export/en-hi-v1-fp16` (+ CT2 int8)

### E) Real-time VAD speech-to-speech (always-listening)

Design: [docs/05-vad-realtime-integration.md](../docs/05-vad-realtime-integration.md).
Config: `pipeline/config/realtime.yaml`.

Install the extra deps once (PortAudio must be installed system-wide, e.g.
`sudo pacman -S portaudio`):

```bash
export PYTHONPATH="$PWD/venv/lib/python3.11/site-packages"
PY=/home/prieyan/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
$PY -m pip install --target "$PWD/venv/lib/python3.11/site-packages" -r pipeline/requirements.txt
```

Bring it up in stages (each is independently testable — see build order in the doc).
Same as above, `venv/bin/python` works directly — no PYTHONPATH export needed:

```bash
# 1) capture + VAD only — dumps utterances to pipeline/spill/, no models loaded
venv/bin/python pipeline/run_realtime.py --stage capture

# 2) + STT — prints live transcripts, aborts if CUDA is unavailable
venv/bin/python pipeline/run_realtime.py --stage stt

# 3) + MT — prints en -> hi translations
venv/bin/python pipeline/run_realtime.py --stage mt

# 4) full loop — STT -> MT -> Piper -> serial playback (this is the always-listening speech-to-speech mode)
venv/bin/python pipeline/run_realtime.py --stage full
```

Ctrl+C shuts down cleanly (stops the mic, drains queues, releases the audio device).

#### Picking your USB mic

PortAudio's "default input device" is not always your USB mic (could be a
laptop's built-in mic, a monitor/loopback device, etc.) — if the `mic level:
peak=` log lines stay near 0 while you're talking, that's what's happening.

```bash
# list every input-capable device with its index
venv/bin/python pipeline/run_realtime.py --list-devices

# then target it directly by index or by a substring of its name
venv/bin/python pipeline/run_realtime.py --device 3
venv/bin/python pipeline/run_realtime.py --device USB
```

To make it permanent, set `capture.device` in `pipeline/config/realtime.yaml`
to the same index or name instead of passing `--device` every time.
