#!/usr/bin/env bash
# Download English + Hindi Piper voices for PC pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$out" "$url"
  else
    echo "Need curl or wget installed." >&2
    exit 1
  fi
}

mkdir -p \
  "$ROOT/tts/models/upstream" \
  "$ROOT/tts/models/export/en_lessac_medium" \
  "$ROOT/tts/models/export/hi_official_v1"

echo "Downloading English lessac medium…"
download \
  "$BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
  "$ROOT/tts/models/upstream/en_US-lessac-medium.onnx"
download \
  "$BASE/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
  "$ROOT/tts/models/upstream/en_US-lessac-medium.onnx.json"

echo "Downloading Hindi priyamvada medium…"
download \
  "$BASE/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx" \
  "$ROOT/tts/models/upstream/hi_IN-priyamvada-medium.onnx"
download \
  "$BASE/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json" \
  "$ROOT/tts/models/upstream/hi_IN-priyamvada-medium.onnx.json"

cp "$ROOT/tts/models/upstream/en_US-lessac-medium.onnx" \
   "$ROOT/tts/models/export/en_lessac_medium/voice.onnx"
cp "$ROOT/tts/models/upstream/en_US-lessac-medium.onnx.json" \
   "$ROOT/tts/models/export/en_lessac_medium/voice.onnx.json"

cp "$ROOT/tts/models/upstream/hi_IN-priyamvada-medium.onnx" \
   "$ROOT/tts/models/export/hi_official_v1/voice.onnx"
cp "$ROOT/tts/models/upstream/hi_IN-priyamvada-medium.onnx.json" \
   "$ROOT/tts/models/export/hi_official_v1/voice.onnx.json"

echo "Done."
echo "  EN: $ROOT/tts/models/export/en_lessac_medium/voice.onnx"
echo "  HI: $ROOT/tts/models/export/hi_official_v1/voice.onnx"
