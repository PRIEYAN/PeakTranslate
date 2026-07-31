# PeakTranslation — Models & Fine-Tuning

Fine-tune and prepare each stage **one after another**. Train on a workstation / cloud GPU; export artifacts that the Jetson Nano can load. Do not fine-tune heavy models on the Nano itself.

| Order | Stage | Train where | Run where |
|-------|--------|-------------|-----------|
| 1 | STT — Whisper | PC / cloud GPU | Jetson GPU |
| 2 | MT — MarianMT | PC / cloud GPU | Jetson GPU |
| 3 | TTS — Piper | PC (optional) | Jetson CPU |

---

## Shared conventions

- Keep a single `models/` tree mirrored on the Nano:

```text
models/
  whisper/
    tiny/                  # or faster-whisper CTranslate2 export
  marian/
    en-ta/
    en-hi/
  piper/
    ta_IN-xxx.onnx
    ta_IN-xxx.onnx.json
```

- Version every artifact: `en-ta-v1`, dataset hash, training date in a small `MODEL_CARD.md` next to the weights.
- Evaluation before integration: each stage must beat a frozen baseline on a held-out set **before** wiring into the live pipeline.

---

## 1. STT — Whisper (GPU)

### Role

`AudioChunk` → `Transcript` (text + language + optional confidence).

### Recommended variants for Jetson Nano

| Variant | Notes |
|---------|--------|
| `tiny` / `tiny.en` | Best fit for Nano latency/memory |
| `base` / `base.en` | Largest practical size on 4 GB Nano in many demos |
| `faster-whisper` + CTranslate2 | **Not default on Nano Dev Kit** — aarch64 PyPI CT2 is CPU-only; CUDA build expects newer CUDA than Nano’s 10.2. Revisit on Orin/JP5+ |

Avoid `small`+ on Nano unless heavily quantized and profiled. Deploy STT with **Jetson PyTorch CUDA** first.

### Data for fine-tuning

You need **paired audio + transcript** in your domain (accents, noise, vocabulary).

| Source | Use |
|--------|-----|
| Common Voice / FLEURS | Bootstrap |
| In-house recordings | Domain match (mic, room, speakers) |
| Augmentation | Noise, gain, speed perturbation |

Formats: WAV/FLAC, 16 kHz mono preferred for Whisper.

Minimum useful fine-tune set (rough): hundreds of minutes for light adaptation; more for strong domain shift.

### Fine-tuning approach

1. Start from OpenAI Whisper or Hugging Face `openai/whisper-tiny` (or `tiny.en` if English-only STT).
2. Fine-tune with Hugging Face Transformers / Whisper fine-tune scripts, or use **LoRA** / PEFT to keep adapters small.
3. Evaluate WER / CER on a held-out set that matches Jetson mic conditions.
4. Export for Nano:
   - **Option A (Nano Dev Kit default):** PyTorch / HF Whisper weights loaded with JetPack-matched Torch CUDA.
   - **Option B (Orin / after CT2-GPU works):** convert to **CTranslate2** for `faster-whisper` (`float16` or `int8`).

Example CTranslate2 conversion (training PC — only if the Nano/Orin runtime can load CT2 CUDA):

```bash
ct2-transformers-converter \
  --model ./checkpoints/whisper-tiny-finetuned \
  --output_dir ./models/whisper/tiny-ct2 \
  --quantization float16
```

See [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md) for runtime availability.

### Adapter contract expectations

```python
class WhisperSTT:  # implements STTEngine
    def transcribe(self, audio: AudioChunk) -> Transcript:
        ...
```

Config keys: `model_path`, `device=cuda`, `language` (fixed or auto), `beam_size`, `vad_filter`.

### Acceptance criteria before moving to MT

- [ ] WER acceptable on domain test set
- [ ] Median latency for ~3–5 s utterance within budget (measure on Nano)
- [ ] Stable under your mic gain / noise
- [ ] Artifact path documented in `pipeline.yaml`

---

## 2. Translation — MarianMT (GPU)

### Role

`Transcript` → `Translation` for a configured `src→tgt` pair.

### Why MarianMT

- Strong for **bilingual** pairs (Helsinki-NLP `opus-mt-*` family).
- Smaller than many LLM translators — better for Jetson.
- One checkpoint per direction (or use a multilingual Marian if you standardize on that later).

### Choosing base models

Examples (replace with your real pair):

| Pair | Typical HF id pattern |
|------|------------------------|
| English → Tamil | `Helsinki-NLP/opus-mt-en-mul` or dedicated `en-ta` if available |
| English → Hindi | `Helsinki-NLP/opus-mt-en-hi` |
| Reverse | separate `opus-mt-xx-en` checkpoint |

Confirm the exact Helsinki-NLP / community model for your languages. If no good bilingual model exists, start from the closest multilingual Marian and fine-tune on your parallel corpus.

### Data for fine-tuning

You need a **parallel corpus**: `src_text \t tgt_text`.

| Source | Use |
|--------|-----|
| OPUS (MultiUN, OpenSubtitles, CCAligned, etc.) | Bootstrap |
| Domain bitext (manual / professional) | Product vocabulary |
| Back-translation | Expand scarce target data |

Clean aggressively: dedupe, length ratio filters, lang-id filters, remove misaligned pairs.

### Fine-tuning approach

1. Load base MarianMT (`MarianMTModel` + `MarianTokenizer`).
2. Fine-tune with Hugging Face `Seq2SeqTrainer` (or fairseq/Marian tools if you prefer).
3. Track **BLEU / chrF** on a domain dev set; also spot-check fluency with native speakers.
4. Save full model + tokenizer under `models/marian/<pair>/`.
5. Optional later (Orin / working CT2-GPU): convert to **CTranslate2** Marian for faster inference. On classic Nano, ship **PyTorch Marian** weights.

```bash
ct2-transformers-converter \
  --model ./checkpoints/marian-en-ta \
  --output_dir ./models/marian/en-ta-ct2 \
  --quantization float16
```

### Pair registry (loose coupling)

Do **not** hardcode languages in Python. Register pairs in config:

```yaml
translation:
  backend: marianmt
  pairs:
    en-ta:
      model_path: /models/marian/en-ta-ct2
      src: en
      tgt: ta
    en-hi:
      model_path: /models/marian/en-hi-ct2
      src: en
      tgt: hi
  default_pair: en-ta
```

Runtime swap:

```python
engine.set_pair("en-hi")  # loads/unloads weights as needed
```

**Adding a new pair later:** fine-tune → export → add YAML entry → (optional) add TTS voice for `tgt`. No orchestrator changes.

### Adapter contract expectations

```python
class MarianMTEngine:  # implements MTEngine
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Translation:
        ...
```

Implementation should resolve `(src, tgt)` → `model_path` via the registry, not via `if lang == ...` trees.

### GPU coexistence with Whisper

On Nano, load Marian **after** STT finishes an utterance (or keep both but smaller quantized models only). Measure with `tegrastats`. Prefer **one active GPU model** at a time if you OOM.

### Acceptance criteria before moving to TTS

- [ ] BLEU/chrF vs baseline improved (or domain adequacy approved)
- [ ] Latency OK for typical transcript length
- [ ] Pair switch works via config only
- [ ] Tokenizer special tokens / prefix codes handled correctly for your base model

---

## 3. TTS — Piper (CPU)

### Role

`Translation` → `SpeechAudio` on **CPU**, freeing GPU for Whisper/Marian.

### Why Piper

- Fast ONNX voices, designed for edge/CPU.
- Simple: `text + voice model (.onnx)` → WAV/PCM.
- Easy to add languages by dropping new voice files.

### Voices

1. Check [Piper voices](https://github.com/rhasspy/piper/blob/master/VOICES.md) for your target language.
2. Download matching `.onnx` + `.onnx.json`.
3. Map `tgt_lang` → voice path in config:

```yaml
tts:
  backend: piper
  device: cpu
  voices:
    ta: /models/piper/ta_IN-....onnx
    hi: /models/piper/hi_IN-....onnx
```

### Fine-tuning / custom voice (optional)

Only if stock voices are wrong for your product (accent, brand voice):

1. Record / collect clean single-speaker audio + transcripts for the **target language**.
2. Follow Piper / VITS training docs (train on PC/GPU).
3. Export ONNX voice; deploy beside other Piper voices.
4. Keep the same `TTSEngine` interface — only the voice path changes.

If a Piper voice does not exist for your language, options: train one, or add a second TTS adapter later (e.g. another ONNX TTS) behind the same interface — do not bake Piper APIs into the orchestrator.

### Adapter contract expectations

```python
class PiperTTS:  # implements TTSEngine
    def synthesize(self, text: str, voice_id: str) -> SpeechAudio:
        ...
```

### Acceptance criteria

- [ ] RTF < 1.0 on Nano CPU for typical sentence length (faster than real time preferred)
- [ ] Intelligibility OK at your playback sample rate
- [ ] Voice selection follows `tgt_lang` from the active MT pair

---

## End-to-end evaluation (after all three)

Use a fixed set of utterances:

1. Record source speech → STT text → reference transcript (WER).
2. MT output vs reference translation (BLEU/chrF + human score).
3. Listen to Piper output (MOS-style or informal native review).
4. Measure **pipeline latency**: speech-end → first audio out.

Log each stage’s latency separately so you know which model to shrink next.

---

## Practical fine-tune schedule

| Week focus | Deliverable |
|------------|-------------|
| STT only | Fine-tuned Whisper + Nano benchmark notebook/script |
| MT only | Fine-tuned Marian pair(s) + pair registry working |
| TTS only | Piper voices wired; optional custom voice |
| Integration | Full pipeline on Nano with VAD + queues |
| Hardening | Quantization, model swap, thermal/power tests |

---

## Related docs

- [01 — Architecture](./01-architecture.md)
- [03 — Jetson Nano deployment](./03-jetson-deployment.md)
- [04 — Real-time queue pipeline](./04-realtime-queue-pipeline.md)
