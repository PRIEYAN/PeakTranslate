# PeakTranslation — System Architecture

Real-time speech translation on **NVIDIA Jetson Nano**: speech in → translated speech out.

| Stage | Model | Device |
|-------|--------|--------|
| STT | Whisper | GPU |
| Translation | MarianMT | GPU |
| TTS | Piper | CPU |

---

## Goals

1. **Low latency** end-to-end on Jetson Nano (4 GB).
2. **Loosely coupled** stages — swap STT, MT, or TTS without rewriting the pipeline.
3. **Easy language-pair switching** — new MarianMT pair = config + model path, not code changes.
4. **Incremental delivery** — fine-tune and integrate one stage at a time.

---

## High-level pipeline

Live runtime is a **continuous multi-queue** design (not a one-shot series). See [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md).

```
Capture/VAD (always on) → Whisper GPU → Q1 sentences
                              ↕ gpu_lock (time-share)
                         MarianMT GPU ←── pop Q1 ──→ Q2 translations
                         Piper CPU    ←── pop Q2 ──→ Q3 WAV paths
                         Speaker      ←── pop Q3 only when previous WAV finished
```

Each arrow is a **queue message** (typed payload), not a direct function call into another model’s internals.

---

## Design principles

### 1. Interface contracts, not concrete models

Every stage implements a small protocol. The orchestrator only depends on those interfaces.

```text
STTEngine.translate_audio(audio) -> Transcript
MTEngine.translate(text, src_lang, tgt_lang) -> Translation
TTSEngine.synthesize(text, voice_id) -> AudioBuffer
```

Concrete classes (`WhisperSTT`, `MarianMTEngine`, `PiperTTS`) live behind those interfaces. Swapping Whisper → another STT means a new adapter class + config entry.

### 2. Config-driven language pairs

Language pairs are **data**, not hardcoded branches:

```yaml
# config/pipeline.yaml
pipeline:
  stt:
    backend: whisper
    model_path: /models/whisper/tiny.en  # or multilingual base/tiny
    device: cuda
    compute_type: float16   # or int8 if using faster-whisper

  translation:
    backend: marianmt
    device: cuda
    # Pair registry — add a new pair here, no code change
    pairs:
      en-ta:
        model_path: /models/marian/en-ta
        src: en
        tgt: ta
      en-hi:
        model_path: /models/marian/en-hi
        src: en
        tgt: hi
    default_pair: en-ta

  tts:
    backend: piper
    device: cpu
    voices:
      ta: /models/piper/ta_IN-model.onnx
      hi: /models/piper/hi_IN-model.onnx
      en: /models/piper/en_US-lessac-medium.onnx
```

**To add a new translation pair:** download/fine-tune MarianMT → drop weights under `/models/marian/<pair>` → add one YAML entry → optionally add a Piper voice for the target language.

### 3. Message bus / queue between stages

Prefer async queues over synchronous call chains so stages can:

- run at different speeds (GPU MT vs CPU TTS),
- be replaced or scaled independently,
- be tested in isolation with recorded messages.

Suggested internal messages:

| Message | Fields |
|---------|--------|
| `AudioChunk` | `pcm`, `sample_rate`, `timestamp`, `session_id` |
| `Transcript` | `text`, `lang`, `confidence`, `is_final`, `session_id` |
| `Translation` | `text`, `src_lang`, `tgt_lang`, `session_id` |
| `SpeechAudio` | `pcm`, `sample_rate`, `session_id` |

Use in-process `asyncio.Queue` / `multiprocessing.Queue` first. Later you can put Redis / ZeroMQ / gRPC between the same interfaces if you move a stage off-device.

### 4. Plugin / adapter layout

```text
peaktranslation/
  core/
    messages.py          # dataclasses / pydantic models
    interfaces.py        # STTEngine, MTEngine, TTSEngine protocols
    orchestrator.py      # wires queues + lifecycle
    config.py            # load YAML, resolve pair → model
  adapters/
    stt/
      whisper_gpu.py
    mt/
      marian_gpu.py
    tts/
      piper_cpu.py
  services/
    vad.py               # voice activity detection
    audio_io.py          # mic capture / playback
  config/
    pipeline.yaml
  models/                # gitignored; large weights on disk
```

The orchestrator never imports `transformers` or `whisper` directly — only adapters do.

---

## Runtime sequence (one utterance)

1. **VAD** detects speech start/end (or fixed chunk window for streaming).
2. **STT** receives PCM → emits partial and/or final `Transcript`.
3. **MT** loads (or already holds) the model for `src→tgt` from the pair registry → emits `Translation`.
4. **TTS** picks voice for `tgt_lang` → emits `SpeechAudio` → playback.

GPU time-share: Whisper and MarianMT both use CUDA under a **GPU lock** while queues keep the pipeline logically parallel. Piper (CPU) and serial playback run independently. Details: [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md).

---

## Scaling & swap scenarios

| Change | What you touch |
|--------|----------------|
| New lang pair (e.g. `en-fr`) | Marian checkpoint + YAML `pairs` + Piper voice |
| Better STT | New file under `adapters/stt/` + `stt.backend` |
| Cloud MT fallback | New `adapters/mt/http_client.py` implementing same interface |
| Batch offline mode | Same interfaces; orchestrator reads files instead of mic |
| Multi-user later | One session_id per stream; shared model process, per-session queues |

---

## Memory budget (Jetson Nano 4 GB)

Rough shared budget (tune per model size):

| Component | Typical target |
|-----------|----------------|
| OS + CUDA + system | ~1.0–1.5 GB |
| Whisper tiny / base (GPU) | ~0.5–1.0 GB |
| MarianMT (GPU) | ~0.3–0.8 GB |
| Piper (CPU) + audio buffers | ~0.2–0.4 GB |
| Headroom | keep ≥300–500 MB free |

**Rules of thumb**

- Prefer **Whisper tiny** or **base**; avoid large/medium on Nano unless quantized and carefully profiled.
- On **classic Nano Dev Kit**, prefer **PyTorch Whisper** (`tiny` / `base.en`). CTranslate2 GPU is **not** a drop-in on Nano (see doc 04).
- Load **one Marian pair at a time**; unload or swap when the pair changes.
- Do not keep Whisper + Marian + large TTS all maximally resident without measuring `tegrastats`.

---

## Non-goals (v1)

- Multi-speaker diarization
- Simultaneous bidirectional conversation (can add later as two pipelines)
- On-device training loop (fine-tune on a PC/cloud GPU; deploy weights to Nano)

---

## Suggested build order

1. Interfaces + dummy adapters + mic → file pipeline (prove contracts).
2. Whisper STT on GPU alone; measure latency/WER.
3. MarianMT on GPU alone; measure BLEU / latency for your pair.
4. Piper on CPU alone; measure RTF (real-time factor).
5. Wire continuous queues (Q1/Q2/Q3) + VAD + GPU lock (see `04-realtime-queue-pipeline.md`).
6. Fine-tune STT → MT → TTS in that order (see `02-models-and-finetuning.md`).
7. Harden for Jetson (see `03-jetson-deployment.md`).

---

## Related docs

- [02 — Models & fine-tuning](./02-models-and-finetuning.md)
- [03 — Jetson Nano deployment](./03-jetson-deployment.md)
- [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md)
- Deep research: [stt/whisper](./stt/whisper-jetson-research.md) · [translate/marian](./translate/marianmt-jetson-research.md) · [tts/piper](./tts/piper-jetson-research.md)
- Per-stage production guides: [stt/](../stt/README.md) · [translate/](../translate/README.md) · [tts/](../tts/README.md)
