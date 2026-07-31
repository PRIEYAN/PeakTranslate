# PeakTranslation — Jetson Nano Deployment

How to run the loosely coupled STT → MT → TTS pipeline on an **NVIDIA Jetson Nano Dev Kit**, with Whisper + MarianMT on **GPU** and Piper on **CPU**.

---

## Hardware & software baseline

| Item | Guidance |
|------|----------|
| Board | Jetson Nano 4 GB (2 GB is much harder — prefer 4 GB) |
| Storage | 64 GB+ SD or preferably SSD via USB |
| Power | Official 5V/4A barrel supply recommended under GPU load |
| JetPack | Use a JetPack version with CUDA + cuDNN matching your PyTorch/CTranslate2 wheels |
| Cooler | Active cooling; throttle kills real-time latency |

Check device:

```bash
jetson_release  # or: cat /etc/nv_tegra_release
tegrastats
```

---

## Resource strategy

Nano has **shared** memory for CPU and GPU (~4 GB total).

### Recommended runtime policy

1. **Piper always on CPU** + **ONNX Runtime CPU** (GPU ORT wheels are not drop-in on classic Nano).
2. **Continuous queue pipeline** (Q1 transcript → Q2 translation → Q3 WAV) with workers always looping — see [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md).
3. **GPU lock** between Whisper and Marian infer calls (logical parallelism via queues; physical CUDA time-share on 4 GB).
4. **Default STT/MT:** Jetson **PyTorch CUDA** Whisper + Marian. Do **not** assume PyPI `faster-whisper` / CTranslate2 gives GPU on Nano (aarch64 wheels are CPU-only; CT2 CUDA often needs CUDA ≥ 11).
5. Use **tiny/base** Whisper; one Marian pair loaded at a time.
6. Disable unused desktop services if running headless.

### Swap (for queues / overflow, not model weights)

Configure swap + a spill directory so **queue backlog** (especially WAV files on disk) does not OOM the board. Do **not** rely on swap for hot Whisper/Marian weights — paging kills realtime. Details in doc 04.

---

## Host setup (outline)

Exact package versions depend on JetPack. Treat this as a checklist:

1. Flash JetPack; confirm CUDA works (`nvcc --version`, device query).
2. Create a Python venv (or conda if you standardize on it).
3. Install audio deps: PortAudio, ALSA utils, working mic/speaker.
4. Install inference stacks (Nano Dev Kit):
   - STT: OpenAI Whisper / HF Whisper on **Jetson PyTorch CUDA** (matched to JetPack 4.6 / CUDA 10.2).
   - MT: `transformers` + same Jetson `torch` CUDA build.
   - TTS: `piper` + **CPU** `onnxruntime` (not PyPI `onnxruntime-gpu` expectations from x86/Orin docs).
5. Copy `models/` onto the device (rsync/scp); do not git-commit large weights.
6. Install your app package + point `pipeline.yaml` at absolute model paths.

### PyTorch on Jetson

Use NVIDIA’s Jetson-compatible PyTorch wheels for your JetPack version. Generic `pip install torch` from PyPI often **does not** include CUDA for aarch64 Jetson.

### Audio devices

```bash
arecord -l
aplay -l
# test
arecord -d 3 -f S16_LE -r 16000 /tmp/test.wav && aplay /tmp/test.wav
```

Normalize capture to **16 kHz mono** for Whisper.

---

## Process architecture on device

```text
┌──────────────────────────────────────────────┐
│                 Orchestrator                 │
│  queues: audio → transcript → mt → speech    │
└──────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
   STT worker     MT worker      TTS worker
   (CUDA)         (CUDA)         (CPU)
```

### Implementation options

| Option | Pros | Cons |
|--------|------|------|
| Single process + `asyncio` + thread/process for Piper | Simple | GIL / careful GPU calling |
| Multiprocessing: GPU process + CPU TTS process | Isolates crashes; clearer memory | IPC overhead |
| One long-lived service + CLI/client | Better for demos/products | More ops surface |

Start with **one process**, two GPU call sites serialized by a lock, Piper in a thread pool. Split processes only if you need isolation or later scale-out.

### GPU lock + queues (not narrow series)

Do **not** chain listen→STT→MT→TTS→play in one blocking function. Use continuous workers and Q1/Q2/Q3 as in doc 04.

```python
gpu_lock = threading.Lock()

def whisper_worker():
    while True:
        audio = audio_q.get()
        with gpu_lock:
            text = stt.transcribe(audio)
        if text.strip():
            transcript_q.put(text)          # Q1

def marian_worker():
    while True:
        sent = transcript_q.get()           # pop Q1
        with gpu_lock:
            out = mt.translate(sent, src, tgt)
        translation_q.put(out)              # Q2

def piper_worker():
    while True:
        tr = translation_q.get()            # pop Q2
        path = tts.synthesize_to_wav(tr)
        wav_q.put(path)                     # Q3

def playback_worker():
    while True:
        path = wav_q.get()                  # pop Q3
        play_blocking(path)                 # next WAV only after this finishes
```

---

## Configuration on Nano

Keep environment-specific paths in config, not code:

```yaml
# config/pipeline.nano.yaml
audio:
  sample_rate: 16000
  channels: 1
  input_device: null    # default mic
  output_device: null

vad:
  enabled: true
  min_speech_ms: 250
  max_utterance_ms: 8000
  silence_end_ms: 400

pipeline:
  stt:
    backend: whisper
    model_path: /opt/peaktranslation/models/whisper/tiny-ct2
    device: cuda
    compute_type: float16
  translation:
    backend: marianmt
    device: cuda
    default_pair: en-ta
    pairs:
      en-ta:
        model_path: /opt/peaktranslation/models/marian/en-ta-ct2
        src: en
        tgt: ta
  tts:
    backend: piper
    device: cpu
    voices:
      ta: /opt/peaktranslation/models/piper/ta_IN-model.onnx

runtime:
  serialize_gpu: true
  log_latencies: true
```

Ship `pipeline.yaml` for laptop smoke tests and `pipeline.nano.yaml` for the device.

---

## Staging models onto the device

```bash
rsync -avP ./models/ nano:/opt/peaktranslation/models/
rsync -avP ./config/ nano:/opt/peaktranslation/config/
```

Verify:

```bash
du -sh /opt/peaktranslation/models/*
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Bring-up order on hardware

Same order as model work — validate each stage **on the Nano**, not only on the PC.

### 1. STT smoke test

```bash
python scripts/bench_stt.py --config config/pipeline.nano.yaml --wav samples/hello.wav
```

Record: latency, VRAM delta (`tegrastats`), transcript quality.

### 2. MT smoke test

```bash
python scripts/bench_mt.py --pair en-ta --text "Hello, how are you?"
```

### 3. TTS smoke test

```bash
python scripts/bench_tts.py --voice ta --text "<tamil sample>" --out /tmp/out.wav
aplay /tmp/out.wav
```

### 4. Full pipeline

```bash
python -m peaktranslation.run --config config/pipeline.nano.yaml
```

---

## Latency & quality tuning

| Symptom | Likely lever |
|---------|----------------|
| OOM / kill | Smaller Whisper; CT2 int8; unload MT while STT runs; shorter max utterance |
| STT slow | `tiny`, lower beam size, VAD to cut silence, CT2 |
| MT slow | shorter inputs, CT2, smaller Marian, max new tokens cap |
| TTS slow / choppy | smaller Piper voice quality tier; reduce sample rate if acceptable |
| Bad translations | domain fine-tune; fix STT errors first (garbage-in) |
| Thermal throttle | better cooling; reduce concurrent load; undervolt not needed — fix airflow |

Log per-stage timings:

```text
vad_ms | stt_ms | mt_ms | tts_ms | e2e_ms
```

---

## Switching language pairs in the field

Because pairs are registry-driven:

1. Copy new Marian export to `models/marian/<pair>/`.
2. Add voice under `models/piper/` if needed.
3. Edit `pipeline.nano.yaml` `translation.pairs` (+ `tts.voices`).
4. Restart service (or call a hot-reload if you implement one).

No rebuild of STT/TTS adapters required for a normal new pair.

---

## Service / product packaging (optional)

- systemd unit: start on boot, restart on crash.
- Health endpoint or log heartbeats.
- Capture anonymized latency metrics only (avoid storing raw audio unless required).

Example systemd sketch:

```ini
[Unit]
Description=PeakTranslation
After=network.target sound.target

[Service]
WorkingDirectory=/opt/peaktranslation
ExecStart=/opt/peaktranslation/.venv/bin/python -m peaktranslation.run --config config/pipeline.nano.yaml
Restart=on-failure
Environment=NVIDIA_VISIBLE_DEVICES=all

[Install]
WantedBy=multi-user.target
```

---

## Security & ops notes

- Models and configs under `/opt/...` with least-privilege user.
- Mic access only for the service user.
- Keep training data and raw voice corpora off the device unless needed.

---

## Definition of done (Nano MVP)

- [ ] Mic → Whisper (GPU) → MarianMT (GPU) → Piper (CPU) → speaker works live
- [ ] GPU serialization prevents OOM under normal utterance lengths
- [ ] New MT pair addable via config + model files only
- [ ] Stage latency logs available
- [ ] Survives 30+ min continuous demo without thermal collapse (with cooling)

---

## Related docs

- [01 — Architecture](./01-architecture.md)
- [02 — Models & fine-tuning](./02-models-and-finetuning.md)
- [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md)
