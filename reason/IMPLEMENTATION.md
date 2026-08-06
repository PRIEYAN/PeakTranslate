# Reason implementation runbook

## Your setup (locked in)

| Item | Choice |
|---|---|
| GPU | RTX 3050 Laptop, **4096 MiB** — 4-bit quantization is mandatory, not optional |
| Model | `google/gemma-2-2b-it` (gated, instruction-tuned) via `configs/profiles.yaml` |
| Quantization | NF4 (`bitsandbytes`), `device_map={"": 0}` — never `"auto"` |
| Relationship to `translate/` | Sibling stage, mutually exclusive at runtime (`mode: translate \| reason`) |
| Runs alongside | Whisper STT (always loaded); never alongside Marian MT |

Full rationale for every choice above: [docs/reasoningModel/01-gemma-reasoning-mode.md](../docs/reasoningModel/01-gemma-reasoning-mode.md).

## Security

- `HF_TOKEN` lives in repo-root `.env` (gitignored). Same token STT/translate already use.
- Gemma repos are gated — accept the licence on the model page with the account that owns the token.
- Never commit `.env` or `models/**` weights (`.gitignore` already excludes `**/models/upstream/**` etc.; only `.gitkeep` is tracked).

## Prerequisites

```bash
cd /home/prieyan/weeb/PeakTranslation
venv/bin/pip install -r reason/requirements.txt
venv/bin/python -c "import bitsandbytes; print(bitsandbytes.__version__)"
venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

If the last command fails or the licence hasn't been accepted, generation
will fail at load with an HTTP 401 — see doc §18.

## Bring-up order (doc §17 — do these in sequence)

1. **Access** — licence + token, above.
2. **Domain layer, no GPU** — unit-test `SentenceAssembler` / `ConversationHistory`:
   ```bash
   venv/bin/python -m pytest reason/tests -q
   ```
3. **Fit** — confirm the 4-bit load fits with headroom:
   ```bash
   venv/bin/python reason/scripts/bench_reason.py
   ```
4. **Engine alone, no pipeline, no audio**:
   ```bash
   venv/bin/python reason/scripts/smoke_reason.py --text "What is the capital of France?"
   ```
5. **Worker with `FakeEngine`** (still no GPU):
   ```bash
   venv/bin/python -m pytest pipeline/tests -q
   ```
6. **Real mic, printed replies, no TTS**:
   ```bash
   venv/bin/python pipeline/run_realtime.py --mode reason --stage mt
   ```
7. **Full loop, headphones** (feedback-loop risk before you've verified muting):
   ```bash
   venv/bin/python pipeline/run_realtime.py --mode reason --stage full
   ```
8. **Speakers**, with `reason.mute_capture_while_replying: true` in
   `pipeline/config/realtime.yaml` (default). If the assistant answers
   itself, the `speaking` mute isn't wired — see doc §14.
9. **Regression-check translate mode** — must be byte-for-byte unaffected:
   ```bash
   venv/bin/python pipeline/run_realtime.py --mode translate --stage mt
   ```

## Known constraints (see doc §18 for the full failure-mode table)

- No barge-in: you cannot interrupt the assistant mid-reply while capture is muted.
- 2B-class models are conversational, not chain-of-thought reasoners — don't expect multi-step problem solving.
- First-token latency is ~3–8s on this GPU; streaming makes first-*sentence* latency the number that matters.
