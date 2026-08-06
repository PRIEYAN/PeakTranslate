# Reason — Gemma: local instruction-tuned reasoning/chat stage

Second text stage for PeakTranslation, alongside `stt/`, `translate/`, and
`tts/`. Where `translate/` turns a transcript into another language,
`reason/` turns a transcript into an answer: speak English → get spoken
English back. Selected at runtime via `pipeline/config/realtime.yaml`'s
`mode: reason`; `translate` stays the default and is untouched.

Full design, rationale, and the VRAM/latency constraints that shaped this:
[docs/reasoningModel/01-gemma-reasoning-mode.md](../docs/reasoningModel/01-gemma-reasoning-mode.md).
Read that before changing anything here — most decisions in this folder
exist because of a specific measured constraint (4 GB VRAM, gated models,
streaming latency) documented there.

---

## 1. Goals

| Goal | Approach |
|---|---|
| Fit on a 4 GB GPU alongside Whisper | 4-bit NF4 quantization is mandatory, not optional |
| Feel responsive despite slow generation | Stream tokens, synthesize sentence-by-sentence |
| Swap models without touching code | Profile registry (`configs/profiles.yaml`) |
| Stay usable standalone (no pipeline) | `src/runtime/` has zero dependency on `pipeline/` |
| Add a new backend later (e.g. GGUF) | One adapter + one profile entry, per the `ReasoningEngine` port |

---

## 2. Folder structure

```text
reason/
├── configs/
│   ├── profiles.yaml           # profile registry (switch model here)
│   └── prompts/
│       └── voice_assistant.md  # system prompt as an asset, not a literal
├── data/eval/smoke_prompts.jsonl
├── models/upstream/             # optional local mirror of Hub repos
├── src/runtime/
│   ├── __init__.py              # build_engine(): the only composition point
│   ├── interface.py             # ReasoningEngine Protocol      (port)
│   ├── messages.py              # Prompt, Turn                  (DTOs)
│   ├── registry.py              # profiles.yaml -> profile dict
│   ├── history.py                # ConversationHistory           (pure)
│   ├── streaming.py             # SentenceAssembler              (pure)
│   ├── prompting.py             # ChatPromptFormatter            (tokenizer)
│   ├── loading.py               # quantization + meta-tensor guard
│   └── gemma_pytorch.py         # adapter: transformers + bitsandbytes
└── scripts/
    ├── download_upstream.py
    ├── smoke_reason.py           # one prompt in, reply out, no audio
    └── bench_reason.py           # tokens/sec, VRAM, first-token latency
```

Layering (see doc §3 for the full dependency-rule table): `interface.py`,
`messages.py`, `history.py`, `streaming.py` import nothing but stdlib.
`gemma_pytorch.py`, `prompting.py`, `loading.py`, `registry.py` are where
`torch`/`transformers`/`yaml` live. `pipeline/realtime/reasoning.py` (outside
this folder) imports only `reason.src.runtime`'s public API — never `torch`.

---

## 3. Before you touch the model: VRAM

Measured on this machine — `nvidia-smi` reports **4096 MiB** total. Gemma 2B
in bf16 is ~5 GB and does not fit. 4-bit NF4 brings it to ~1.6 GB. Doc §5 has
the full budget table. Two rules that follow from this:

- Reason mode never loads Marian. Modes are mutually exclusive.
- `device_map` is pinned to `{"": 0}`, never `"auto"` — an over-budget model
  must raise, not silently offload layers to CPU and look "slow".

## 4. Model access

All Gemma repos are gated on Hugging Face. Accept the licence on the model
page with your account, then confirm the token in `.env` (`HF_TOKEN`) works:

```bash
venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

## 5. Switch models / add a profile

Add an entry to `configs/profiles.yaml` with a `model_id`, then either set
`reason.profile` in `pipeline/config/realtime.yaml` or pass
`--reason-profile <id>` to `run_realtime.py`. No code changes for a same-
shape model (any `-it` causal LM that fits the `gemma_pytorch` runtime).

## 6. Standalone smoke test (no pipeline, no audio)

```bash
venv/bin/python reason/scripts/smoke_reason.py --text "What is the capital of France?"
venv/bin/python reason/scripts/bench_reason.py   # VRAM + tokens/sec
```

## 7. Run it in the pipeline

```bash
venv/bin/python pipeline/run_realtime.py --mode reason --stage mt     # printed replies, no audio
venv/bin/python pipeline/run_realtime.py --mode reason --stage full   # spoken replies
```

See `pipeline/README.md` and doc §17 for the full bring-up order — do the
steps in order, headphones before speakers (self-feedback risk, doc §14).

## 8. Install

```bash
venv/bin/pip install -r reason/requirements.txt
venv/bin/python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

---

## Related

- [docs/reasoningModel/01-gemma-reasoning-mode.md](../docs/reasoningModel/01-gemma-reasoning-mode.md) — full spec
- [docs/06-debugging-meta-tensor-load-race.md](../docs/06-debugging-meta-tensor-load-race.md) — why loading is locked and guarded
- [translate/README.md](../translate/README.md) — the sibling stage this layout mirrors
- [stt/src/runtime/](../stt/src/runtime/) — the implemented precedent for this shape
