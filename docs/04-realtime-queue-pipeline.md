# PeakTranslation — Continuous Real-Time Queue Pipeline (Jetson Nano)

This document defines the **live** pipeline: always-listening STT, parallel CUDA translation, continuous Piper TTS, and **serial speaker playback**. It is **not** a narrow one-utterance-then-wait chain.

It also corrects earlier docs on what runtimes actually work on the **NVIDIA Jetson Nano Developer Kit** (original Nano, Maxwell `sm_53`, typically JetPack 4.6.x / CUDA 10.2 / 4 GB shared RAM).

---

## 1. Runtime availability on Jetson Nano Dev Kit (re-check)

| Runtime | Drop-in on Nano Dev Kit? | Reality |
|---------|--------------------------|---------|
| **PyTorch + CUDA** | Yes, with Jetson wheels | Use NVIDIA / community wheels built for **JetPack 4.6 + CUDA 10.2**. Stock `pip install torch` from PyPI is wrong. PyTorch **2.x needs CUDA 11+** → **not** for original Nano. Stay on **1.10.x-class** Jetson builds. |
| **OpenAI Whisper (PyTorch)** | Feasible | Proven path on Nano (`tiny` / `base.en`). Python/JetPack friction: JP4 ships Python 3.6; upstream Whisper wants 3.8+. Use a maintained Nano image / container / forked Whisper that matches your PyTorch CUDA build. |
| **CTranslate2 (GPU)** | **Not drop-in** | PyPI **aarch64 wheels are CPU-only**. GPU needs **build from source** with CUDA. Upstream CT2 GPU path expects **CUDA ≥ 11** in current docs → **poor fit for Nano CUDA 10.2**. `jetson-containers` / faster-whisper images mostly target **L4T R34+ (Xavier/Orin)**, not classic Nano. Treat CT2-GPU as **optional / later / Orin**, not the Nano MVP default. |
| **faster-whisper** | Only if CT2-GPU works | Same dependency as above. Do **not** assume `pip install faster-whisper` gives CUDA on Nano. |
| **ONNX Runtime (CPU)** | Yes | Piper’s normal path. `pip install onnxruntime` (CPU) works on aarch64. |
| **ONNX Runtime (GPU / CUDA EP)** | **Not drop-in on Nano** | Jetson CUDA wheels are published mainly for **newer JetPack (e.g. JP6 / Orin)** via Jetson AI Lab. Building ORT-GPU for JP4 Nano is painful. **Keep Piper on CPU** as planned — correct choice. |
| **Piper TTS** | Yes (CPU) | ONNX voices + Piper binary/Python on CPU. Do not require CUDA EP for MVP. |
| **MarianMT (Hugging Face + PyTorch)** | Yes, if Torch CUDA works | Run with the same Jetson PyTorch. Prefer **small bilingual** checkpoints; load **one pair** at a time. CT2 Marian is the same “build CT2” caveat as Whisper. |
| **TensorRT** | Possible but heavy | Custom engine work per model/JetPack. Out of scope for MVP. |

### Recommended Nano MVP stack (honest)

| Stage | Runtime on Nano Dev Kit |
|-------|-------------------------|
| STT | **Whisper `tiny` / `base.en` via Jetson PyTorch CUDA** |
| MT | **MarianMT via same PyTorch CUDA** |
| TTS | **Piper + ONNX Runtime CPU** |

Use Linux **swap** as overflow for **queue payloads and OS pressure**, not as the place where model weights live during inference.

### If you upgrade hardware later

On **Orin Nano / JP5–JP6**, revisiting **CTranslate2-from-source**, **faster-whisper**, and **onnxruntime-gpu** becomes realistic. Keep adapters swappable so the queue pipeline below does not change.

---

## 2. Design goal: parallel pipeline, not narrow series

### Wrong (narrow series)

```text
listen → STT → wait → MT → wait → TTS → wait → play → only then listen again
```

That idles the mic and serializes the whole product around one utterance.

### Right (continuous + queues)

```text
[Mic/VAD always on] ──► Whisper (CUDA) ──push──► Q1 (sentences)
                              ▲                      │
                              │                      │ pop (parallel)
                              │                      ▼
                         still listening        MarianMT (CUDA)
                                                     │
                                                     │ push
                                                     ▼
                                               Q2 (translations)
                                                     │
                                                     │ pop
                                                     ▼
                                               Piper (CPU, always ready)
                                                     │
                                                     │ push
                                                     ▼
                                               Q3 (WAV / PCM jobs)
                                                     │
                                                     │ pop ONLY when speaker free
                                                     ▼
                                               Speaker (one WAV at a time)
```

Workers run **concurrently**. Coupling is only through **queues**.

---

## 3. Three queues (swap-backed buffer strategy)

All stage handoffs are **queue data structures**. Items are small records; large audio for TTS output may be a **path on disk** so RAM stays free.

| Queue | Name | Producers | Consumers | Payload |
|-------|------|-----------|-----------|---------|
| **Q1** | `transcript_queue` | Whisper / STT worker | MarianMT worker | Final **sentence** text (+ `utt_id`, `ts`, `src_lang`) |
| **Q2** | `translation_queue` | MarianMT worker | Piper worker | Translated sentence (+ `utt_id`, `tgt_lang`) |
| **Q3** | `wav_queue` | Piper worker | Playback worker | WAV path or PCM handle (+ `utt_id`, duration) |

### Where they live in memory / swap

Jetson Nano **4 GB is unified**. You will use:

1. **In-process queues** (`multiprocessing.Queue` or `queue.Queue`) for hot items (preferred for text).
2. **Optional disk spill** under e.g. `/var/peaktranslation/spill/` on a partition that is allowed to use **swap-backed pressure** (SSD/USB strongly preferred over slow SD).
3. System **swap file/zram** so the OS can page when Q2/Q3 grow — this prevents OOM kills, but **paged model weights = death for latency**. So:

**Policy**

- Queue **text** in RAM (tiny).
- Queue **WAV as files on disk**; Q3 only holds paths + metadata.
- Cap queue depths (`maxsize`). On full: block producer or drop oldest **policy** (choose explicitly; for translation, usually **block** STT push or pause VAD accept).
- Configure swap (e.g. 4–8 GB on USB SSD) as **safety net for buffers**, not for keeping Whisper+Marian weights hot.

Example item shapes:

```python
# Q1
{"utt_id": "20260731-00142", "text": "How are you today?", "src_lang": "en", "t_end": 1712.44}

# Q2
{"utt_id": "20260731-00142", "text": "<translated>", "src_lang": "en", "tgt_lang": "ta"}

# Q3
{"utt_id": "20260731-00142", "wav_path": "/var/peaktranslation/spill/20260731-00142.wav", "sample_rate": 22050}
```

---

## 4. Workers (always-on roles)

### 4.1 Capture + VAD (or realtime socket)

- Continuously read mic PCM (16 kHz mono) **or** accept a realtime audio socket (WebSocket/TCP) with the same PCM contract.
- Run **VAD** (Silero on CPU, WebRTC VAD, or energy VAD) to cut utterances / sentences.
- On speech end (silence threshold) or max duration → emit one `AudioUtterance` into an internal `audio_queue` (optional 4th queue) for Whisper.
- **Never stop the capture loop** while MT/TTS/playback run.

Sentence boundary options:

- Silence-based end-of-utterance (simplest).
- Whisper partials + punctuation / endpointing (harder, better for “continuous sentences”).

MVP: silence VAD → one final transcript per segment → push Q1.

### 4.2 Whisper worker (CUDA, continuous)

```text
loop:
  audio = audio_queue.get()          # block
  text  = whisper.transcribe(audio)  # GPU
  if text.strip():
      transcript_queue.put(sentence) # Q1 → RAM/spill metadata
```

- Stays alive for the process lifetime.
- While this worker runs, **MT/TTS/playback keep running on their own loops**.
- On Nano, two CUDA models at once are tight: use a **GPU lock** so Whisper and Marian **time-share** the GPU without corrupting contexts, while still being **logically parallel** via queues (STT can have queued audio; MT can have queued sentences).

```text
Logical parallelism  = queues + independent workers
Physical GPU        = mutex / lock between Whisper.infer and Marian.infer
```

That is still **not** a narrow series: capture continues, Q1/Q2/Q3 absorb bursts, Piper and playback proceed on CPU while GPU flips between STT and MT.

### 4.3 MarianMT worker (CUDA, parallel consumer of Q1)

```text
loop:
  sent = transcript_queue.get()      # pop Q1
  with gpu_lock:
      out = marian.translate(sent.text, pair)
  translation_queue.put(out)         # push Q2
```

- Pair comes from config registry (`en-ta`, etc.).
- Runs whenever Q1 is non-empty, interleaved with Whisper under `gpu_lock`.

### 4.4 Piper worker (CPU, continuously active)

```text
loop:
  tr = translation_queue.get()       # pop Q2
  wav_path = piper.synthesize(tr.text, voice[tr.tgt_lang])
  wav_queue.put({..., wav_path})     # push Q3
```

- Model stays loaded in the worker process.
- CPU-bound → true overlap with GPU STT/MT.

### 4.5 Playback worker (serial drain of Q3)

```text
loop:
  job = wav_queue.get()              # pop Q3
  play_blocking(job.wav_path)        # wait until finished
  delete_or_retain(job.wav_path)
  # ONLY THEN take the next WAV
```

**Hard rule:** one active speaker job. No overlapping playback. Next item starts only after `aplay` / PortAudio callback completes.

---

## 5. Sequence for overlapping speech

Example timeline:

```text
t0  User speaks sentence A
t1  VAD closes A → Whisper starts A
t2  User speaks sentence B (capture still running)
t3  Whisper finishes A → push Q1(A); starts B when audio ready
t4  Marian pops A → push Q2(A)          ⎫ GPU lock alternates
t5  Whisper finishes B → push Q1(B)     ⎭ with Marian as needed
t6  Piper pops A → push Q3(A.wav)
t7  Speaker plays A.wav to completion
t8  Piper may already have B.wav in Q3
t9  Speaker plays B.wav only after A done
```

Backpressure:

- If Q3 grows (slow speaker), Piper blocks on `wav_queue.put`.
- If Q2 grows, Marian blocks.
- If Q1 grows, Whisper blocks on put → eventually `audio_queue` fills → VAD should **pause accepting** or drop with a logged policy.

---

## 6. Process / thread layout on Nano

Recommended MVP:

```text
Process: peaktranslation
  Thread/async: capture + VAD
  Thread:       whisper_worker   (+ gpu_lock)
  Thread:       marian_worker    (+ gpu_lock)
  Thread:       piper_worker     (CPU)
  Thread:       playback_worker  (blocking audio out)
  Queues:       audio_q, transcript_q (Q1), translation_q (Q2), wav_q (Q3)
```

Alternative (stronger isolation):

- Process A: capture + Whisper (CUDA)
- Process B: Marian (CUDA) — still need care: one GPU, two processes → use file/socket queues and **single GPU owner** or CUDA MPS (limited on Nano)
- Process C: Piper + playback (CPU)

For Nano MVP, **one process + GPU lock** is simpler and safer.

---

## 7. Realtime input sockets (optional)

Same queue pipeline; only the capture source changes.

```text
WebSocket / TCP client ──PCM frames──► capture adapter ──► VAD ──► audio_q
Local mic ALSA/PortAudio ─────────────► capture adapter ──► VAD ──► audio_q
```

Contract:

- 16-bit PCM, mono, 16 kHz (or resample at edge).
- Framing: fixed 20–30 ms packets + VAD, or client sends “end of utterance” markers.

Do not put Whisper inside the socket handler; only enqueue audio.

---

## 8. Config sketch

```yaml
runtime:
  device: jetson_nano
  swap_spill_dir: /var/peaktranslation/spill
  gpu_lock: true          # required on Nano when Whisper + Marian share CUDA

queues:
  transcript_queue: { maxsize: 32 }   # Q1
  translation_queue: { maxsize: 32 }  # Q2
  wav_queue: { maxsize: 8 }           # Q3 — keep small; WAVs are files

capture:
  source: mic             # or: websocket
  sample_rate: 16000
  vad:
    backend: webrtc       # or silero
    silence_ms: 400
    max_utterance_ms: 8000

stt:
  backend: whisper_pytorch
  model: base.en          # or tiny / tiny.en
  device: cuda

translation:
  backend: marian_pytorch
  device: cuda
  default_pair: en-ta
  pairs:
    en-ta:
      model_path: /opt/peaktranslation/models/marian/en-ta
      src: en
      tgt: ta

tts:
  backend: piper
  device: cpu
  onnx_providers: [CPUExecutionProvider]
  voices:
    ta: /opt/peaktranslation/models/piper/ta_IN-model.onnx

playback:
  blocking: true
  device: default
```

---

## 9. Swap setup (queue overflow, not model home)

```bash
# Example: 8G swapfile on fast USB/SSD (adjust device/path)
sudo fallocate -l 8G /mnt/ssd/swapfile
sudo chmod 600 /mnt/ssd/swapfile
sudo mkswap /mnt/ssd/swapfile
sudo swapon /mnt/ssd/swapfile
```

Also create spill dir:

```bash
sudo mkdir -p /var/peaktranslation/spill
sudo chown $USER:$USER /var/peaktranslation/spill
```

Monitor:

```bash
free -h
tegrastats
ls /var/peaktranslation/spill | wc -l
```

If `si`/`so` in `vmstat` stay high during inference, models are thrashing — shrink Whisper, unload unused Marian pair, reduce Q3 retention.

---

## 10. Corrections to earlier PeakTranslation docs

| Earlier advice | Correction for Nano Dev Kit |
|----------------|-----------------------------|
| Prefer faster-whisper / CTranslate2 by default | **Not default** on classic Nano; CT2 GPU wheels missing; CUDA 11 expectation conflicts with Nano CUDA 10.2. Prefer **PyTorch Whisper**. |
| onnxruntime-gpu for edge TTS | **Not required**; Piper should use **CPU ORT**. GPU ORT is an Orin/JP6 story. |
| Serialize GPU “instead of parallel” | Keep **queue parallelism**; serialize only the **CUDA infer calls** with a lock. |
| Narrow mic→STT→MT→TTS function chain | Replaced by this **continuous multi-queue** design. |

Update mental model in [01-architecture.md](./01-architecture.md) and [03-jetson-deployment.md](./03-jetson-deployment.md) to point here for the live runtime.

---

## 11. Implementation checklist

- [ ] Capture loop never stops while other stages run
- [ ] VAD (or socket markers) emit sentence-level audio units
- [ ] Whisper pushes **sentences** to Q1 continuously
- [ ] Marian pops Q1 / pushes Q2 under GPU lock shared with Whisper
- [ ] Piper pops Q2 / writes WAV / pushes path to Q3
- [ ] Playback pops Q3 **one-at-a-time**, blocking until done
- [ ] Queue maxsizes + spill dir + swap configured
- [ ] Runtimes match Nano: PyTorch CUDA Whisper + Marian, Piper CPU ONNX
- [ ] Language pair still config-only (loose coupling preserved)

---

## Related docs

- [01 — Architecture](./01-architecture.md)
- [02 — Models & fine-tuning](./02-models-and-finetuning.md)
- [03 — Jetson Nano deployment](./03-jetson-deployment.md)
