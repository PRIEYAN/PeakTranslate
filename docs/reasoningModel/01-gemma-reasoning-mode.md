# Reasoning mode — Gemma in the VAD real-time pipeline

Implementation spec for adding a **second text stage** to `pipeline/run_realtime.py`: instead of translating what you said, a local Gemma model answers it. Speak English → get spoken English back.

**The translation path stays exactly as it is.** This is a mode switch, not a replacement. `mt_worker`, the Marian export, and the Hindi Piper voice are all untouched; reason mode is a sibling stage selected by config.

- Pipeline design reference: [05 — VAD real-time integration](../05-vad-realtime-integration.md)
- Model loading rules you must follow: [06 — meta tensor load race](../06-debugging-meta-tensor-load-race.md)
- Layout precedent: [translate/README.md §3](../../translate/README.md), and the implemented [stt/src/runtime/](../../stt/src/runtime/)
- Nothing here is implemented yet. This is the build plan.

---

## 1. Where it slots in

The VAD front end, STT, TTS, and playback are unchanged. Only the Q1 → Q2 stage swaps:

```text
Mic (16 kHz mono)
   │  20 ms frames, never blocks
   ▼
VAD endpointer ──► Q0 audio_queue ──► Whisper worker (CUDA, required)
                                            │
                                            ▼
                                     Q1 transcript_queue
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │  --mode translate                             │  --mode reason
                    ▼                                               ▼
             Marian worker (en→hi)                          Gemma worker (en→en)
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                                     Q2 translation_queue
                                            │
                                            ▼
                                     Piper worker (CPU)
                                    hi voice │ en voice
                                            ▼
                                     Q3 wav_queue
                                            │
                                            ▼
                                  Playback (one at a time)
```

Exactly one of the two text workers is started. They never run together — that matters for VRAM (§5) and it means `gpu_lock` semantics don't change.

---

## 2. Module layout

Reason mode is a **top-level stage module**, peer to `stt/`, `translate/`, and `tts/` — not a file dropped into `pipeline/`. The tree mirrors [translate/README.md §3](../../translate/README.md) so anyone who knows one stage knows all of them:

```text
reason/
├── README.md
├── IMPLEMENTATION.md
├── requirements.txt                     # bitsandbytes; torch/transformers shared
├── configs/
│   ├── profiles.yaml                    # profile registry (switch model here)
│   └── prompts/
│       └── voice_assistant.md           # system prompt as an asset, not a literal
├── data/
│   └── eval/
│       └── smoke_prompts.jsonl          # {"prompt": ..., "must_contain": [...]}
├── models/
│   └── upstream/
│       └── gemma-2-2b-it/               # optional local mirror of the Hub repo
├── src/
│   └── runtime/
│       ├── __init__.py                  # build_engine(): the only composition point
│       ├── interface.py                 # ReasoningEngine Protocol  (port)
│       ├── messages.py                  # Prompt, Turn              (DTOs)
│       ├── registry.py                  # profiles.yaml → profile dict
│       ├── history.py                   # ConversationHistory       (pure)
│       ├── streaming.py                 # SentenceAssembler         (pure)
│       ├── prompting.py                 # ChatPromptFormatter       (tokenizer)
│       ├── loading.py                   # quantization + meta-tensor guard
│       └── gemma_pytorch.py             # adapter: transformers + bitsandbytes
└── scripts/
    ├── download_upstream.py
    ├── smoke_reason.py                  # one prompt in, reply out, no audio
    └── bench_reason.py                  # tokens/sec, VRAM, first-token latency
```

Plus exactly one new file in the pipeline, and no other new files anywhere:

```text
pipeline/realtime/reasoning.py           # reasoning_worker: queue plumbing only
```

Matching the existing convention matters in two concrete ways. `stt/src/runtime/` already has this exact shape — `interface.py`, `messages.py`, `registry.py`, one adapter, and `__init__.py` as the factory — so `reason/src/runtime/` is a copy of a layout that already works here. And `translate/src/runtime/` is currently an empty directory: the MT engine lives inline in `pipeline/realtime/workers.py` instead. Reason mode should not add a second inline stage; if anything, doing this properly gives translate a template to migrate onto later.

---

## 3. Architecture: layers and the dependency rule

Four layers. **Dependencies only ever point inward**, toward the domain:

```text
┌──────────────────────────────────────────────────────────────────┐
│ Composition root      pipeline/run_realtime.py                   │
│   picks concrete implementations, owns locks, wires threads       │
└───────────────────────────────┬──────────────────────────────────┘
                                │ injects
┌───────────────────────────────▼──────────────────────────────────┐
│ Application            pipeline/realtime/reasoning.py            │
│   queues, retries, backpressure, `speaking` event.               │
│   Imports NO torch, NO transformers.                             │
└───────────────────────────────┬──────────────────────────────────┘
                                │ calls through the port
┌───────────────────────────────▼──────────────────────────────────┐
│ Domain    reason/src/runtime/{interface,messages,history,        │
│                               streaming}.py                      │
│   Protocol + DTOs + pure logic. Zero third-party imports.        │
└───────────────────────────────▲──────────────────────────────────┘
                                │ implements
┌───────────────────────────────┴──────────────────────────────────┐
│ Infrastructure  reason/src/runtime/{gemma_pytorch,prompting,     │
│                                     loading,registry}.py         │
│   transformers, bitsandbytes, torch, yaml live here and only here│
└──────────────────────────────────────────────────────────────────┘
```

The rule you can check mechanically:

| Layer | May import | Must never import |
|---|---|---|
| `interface.py`, `messages.py`, `history.py`, `streaming.py` | stdlib, each other | torch, transformers, yaml, pipeline |
| `gemma_pytorch.py`, `prompting.py`, `loading.py`, `registry.py` | the domain layer, torch, transformers, yaml | pipeline |
| `pipeline/realtime/reasoning.py` | `reason.src.runtime` public API, stdlib | torch, transformers |
| `pipeline/run_realtime.py` | everything | — |

That third row is the payoff. The worker holds the pipeline's trickiest logic — backpressure, the `speaking` mute, shutdown sentinels — and if it cannot import `torch` then that logic is testable with a fake engine, no GPU and no 1.6 GB download.

A one-line test enforces it:

```python
def test_worker_has_no_ml_deps():
    src = Path("pipeline/realtime/reasoning.py").read_text()
    assert "import torch" not in src and "transformers" not in src
```

---

## 4. SOLID, file by file

Being honest about the motivation: the natural way to write this stage is one big `reasoning_worker` function that loads the model, builds the prompt, tracks history, splits sentences, and pumps queues. That function does five jobs, cannot be tested without a GPU, and has to be edited for every future change. The split below exists to avoid that, and each file earns its place.

### Single responsibility — one reason to change per file

| File | Sole job | Changes when |
|---|---|---|
| `interface.py` | declare the port | the stage contract changes |
| `messages.py` | carry data across boundaries | a field is added |
| `registry.py` | `profiles.yaml` → dict | config schema changes |
| `history.py` | keep last N turns | memory policy changes |
| `streaming.py` | deltas → sentences | chunking policy changes |
| `prompting.py` | messages → model-specific prompt string | a model needs different formatting |
| `loading.py` | quantization config + load-time guards | VRAM strategy changes |
| `gemma_pytorch.py` | drive HF `generate()` | the backend changes |
| `reasoning.py` | queue plumbing | the pipeline graph changes |

`streaming.py` is the clearest win. Sentence chunking is the fiddliest logic in the whole feature (abbreviations, decimals, quotes after terminators) and as a pure class it gets unit tests with plain strings.

### Open/closed — extend by adding a file

Adding a llama.cpp or Ollama backend means: one new adapter, one `profiles.yaml` entry, one line in `build_engine`. The worker, the domain, and `run_realtime.py` are not touched. This is the same escape hatch `translate/` designs for with `marian_pytorch` vs `marian_ct2` — and you will want it, because if 4-bit Gemma is too slow (§7) a GGUF backend is the fallback.

### Liskov — substitutable means the contract is precise

Any `ReasoningEngine` must:

- yield **text deltas** of any size, in order, and nothing after the last one;
- yield **zero** deltas for an empty reply rather than raising;
- raise only on genuine failure, so the worker's `except` means "this utterance failed", never "the reply was empty";
- be safe to call again after a failure.

Note what is deliberately *not* in the contract: sentence boundaries. If engines chunked their own sentences, every new backend would reimplement that logic and drift. Deltas in, `SentenceAssembler` in the worker.

### Interface segregation — keep the port narrow

```python
class ReasoningEngine(Protocol):
    def stream_reply(self, prompt: Prompt, *, cancel: Optional[threading.Event] = None) -> Iterator[str]: ...
    def warmup(self) -> None: ...
```

Two methods, matching the shape of the existing `STTEngine` (`transcribe` + `warmup`). History, prompt formatting, and token counting are all *not* here — a `FakeEngine` for tests is then five lines. Resist adding `set_temperature`, `get_vram_usage`, or `reset_history`; every one of those forces work onto every future backend.

`cancel` (added for barge-in, §19) is the one exception, and it earns its place by staying optional and inert by default: engines that ignore it are still contract-compliant (the worker stops consuming deltas either way, per Liskov above), it just costs a little extra GPU time on cancellation. It's the cooperative half of "stop generating, ASAP" — the only two-way communication this port allows in the direction of the engine.

### Dependency inversion — depend on the port, inject the rest

`reasoning_worker` receives a built `ReasoningEngine`; it never names `GemmaPytorchReasoner`. Same for the load lock: `build_engine` takes `load_lock` as a parameter rather than importing the pipeline's, exactly as `WhisperPytorchSTT` already does — which is what keeps `reason/` usable standalone from `scripts/smoke_reason.py`.

Worth knowing that the existing STT wiring gets this wrong and you should not copy it:

```69:70:pipeline/realtime/workers.py
        from runtime.messages import AudioChunk  # noqa: E402
        from runtime.whisper_pytorch import WhisperPytorchSTT  # noqa: E402
```

`stt_worker` imports the concrete adapter even though `stt/src/runtime/__init__.py` exports a perfectly good `build_engine` factory. Reason mode goes through `build_engine`.

---

## 5. Read this before writing any code: the VRAM budget

This is the constraint that decides the whole design. Measured on this machine:

```console
$ nvidia-smi --query-gpu=name,memory.total --format=csv
NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB
```

**4 GB total.** Gemma 2B in fp16/bf16 is roughly 5 GB of weights alone. It does not fit. Neither does it *nearly* fit — you are ~25% over budget before the CUDA context, the Whisper model, or a single KV cache entry.

So 4-bit quantization is mandatory, not an optimization:

| What | VRAM |
|---|---|
| CUDA context + torch overhead | ~0.3 GB |
| Whisper base fp16 (already loaded, resident) | ~0.15 GB |
| Gemma 2B **bf16** | **~5.0 GB — does not fit** |
| Gemma 2B **4-bit NF4** | ~1.6 GB |
| KV cache (short replies, Gemma 2B uses multi-query attention) | tens of MB |
| Headroom left with 4-bit | ~1.9 GB |

Two consequences to internalize:

**Do not load Marian in reason mode.** Modes are mutually exclusive, so `run_realtime.py` must skip the MT load entirely rather than load both and pick one. That is also why §1 insists only one text worker starts.

**Do not use `device_map="auto"`.** The snippet in the Hugging Face model card uses it, and on a 4 GB card with an oversized model it will "succeed" by silently offloading layers to CPU RAM (or disk). You get no error, just a pipeline that takes 30+ seconds per reply and looks broken. Pin it to the GPU so an over-budget model fails loudly:

```python
device_map={"": 0}   # not "auto" — fail loudly instead of offloading to CPU
```

Verify the real footprint once, via `scripts/bench_reason.py`, before wiring anything:

```bash
venv/bin/python - <<'PY'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                         quantization_config=qc,
                                         device_map={"": 0})
print("weights on GPU:", torch.cuda.memory_allocated() / 1e9, "GB")
print("reserved:", torch.cuda.memory_reserved() / 1e9, "GB")
PY
```

If that number plus ~0.5 GB of Whisper and context does not leave comfortable room, drop to a smaller model (§6) before continuing.

---

## 6. Which model to actually use

Your snippet names `google/gemma-2b`. Two corrections worth making before you build on it.

**`gemma-2b` is the base model, not instruction-tuned.** It has no chat template and was never trained to answer questions — prompt it with "what's the capital of France?" and it will happily continue the *document*, producing more questions or unrelated text. For a spoken assistant you want the `-it` variant:

| Repo | Verdict |
|---|---|
| `google/gemma-2b` | Base. No chat template. Will not behave like an assistant. |
| `google/gemma-2b-it` | Instruction-tuned. Minimum viable choice. |
| `google/gemma-2-2b-it` | Newer generation, same size class. **Recommended.** |

**Also: none of these are "reasoning models" in the chain-of-thought sense.** They don't produce a thinking trace, and at 2B they are weak at multi-step problems. If you want genuine step-by-step reasoning you would need a distilled reasoning model, and the useful ones do not fit in 4 GB at acceptable speed. Calling this "reasoning mode" is fine as a feature name; just don't expect it to solve puzzles. It is good at short conversational answers, which is what a speech loop wants anyway.

**All Gemma repos are gated.** Unauthenticated access returns HTTP 401:

```console
$ curl -s https://huggingface.co/google/gemma-2b/raw/main/config.json
Access to model google/gemma-2b is restricted. You must have access to it and be authenticated to access it.
```

You must accept the licence on the model page with your HF account. The token plumbing already exists — `.env` defines `HF_TOKEN` and `HF_HOME`, and `run_realtime.py` already calls `load_dotenv(ROOT / ".env")`, so `transformers` picks the token up automatically. Nothing to add.

---

## 7. Latency: the thing that will actually feel broken

Translation replies are short — a sentence in, a sentence out, a few hundred milliseconds of GPU. A chat reply is 60–150 tokens. On a 4-bit 2B model on a 3050 Laptop, expect roughly 15–30 tokens/sec, so **3–8 seconds of generation** before Piper has even started. Then TTS and playback on top.

If you generate the whole reply and *then* synthesize, the felt latency is the sum of everything. Don't. Two rules:

**Cap the reply length.** `max_new_tokens: 96` plus a system instruction to answer in one or two sentences. A spoken assistant that monologues is unusable regardless of speed.

**Stream, and synthesize sentence by sentence.** The engine yields deltas, `SentenceAssembler` turns them into sentences, and each finished sentence goes into Q2 immediately. Piper (CPU) synthesizes sentence 1 while Gemma (GPU) is still writing sentence 2, so playback starts after the *first* sentence instead of the last. First-audio latency drops from "whole reply" to "first sentence", which is the number the user perceives.

This is the one place where reason mode differs in shape from translate mode, and it has knock-on effects on the message type (§9) and TTS filenames (§12).

---

## 8. Config: two files, two jobs

**`reason/configs/profiles.yaml`** — everything about the model. Mirrors `translate/configs/pairs.yaml` and is read by `registry.py`:

```yaml
default_profile: gemma-2-2b-it-4bit

profiles:
  gemma-2-2b-it-4bit:
    model_id: google/gemma-2-2b-it     # gated; needs HF_TOKEN in .env
    runtime: gemma_pytorch
    device: cuda
    quantization: nf4                  # nf4 | none — mandatory on 4 GB (§5)
    compute_dtype: bfloat16
    out_lang: hi                       # Gemma replies in Hindi (no Marian)
    prompt: configs/prompts/voice_assistant.md
    decode:
      max_new_tokens: 96
      temperature: 0.7
      top_p: 0.9
    history_turns: 4                   # 0 disables multi-turn context

  gemma-2b-it-4bit:                    # the older model, same shape
    model_id: google/gemma-2b-it
    runtime: gemma_pytorch
    device: cuda
    quantization: nf4
    compute_dtype: bfloat16
    out_lang: hi
    prompt: configs/prompts/voice_assistant.md
    decode: { max_new_tokens: 96, temperature: 0.7, top_p: 0.9 }
    history_turns: 4
```

**`pipeline/config/realtime.yaml`** — everything about the pipeline. `translate` is unchanged; the new block only says which profile and which voice:

```yaml
mode: translate                        # translate | reason

translate:                             # unchanged — still the default path
  pair: en-hi
  tgt_lang: hi
  model_dir: translate/models/export/en-hi-v1-fp16
  fallback_model_dir: translate/models/export/en-hi-v1-ct2-int8
  allow_cpu_fallback: true
  max_new_tokens: 96
  num_beams: 2

reason:
  profile: gemma-2-2b-it-4bit          # key into reason/configs/profiles.yaml
  voice: tts/models/export/hi_official_v1/voice.onnx   # must match out_lang
  barge_in: true
  mute_capture_while_replying: true
```

Two deliberate choices. Model knobs live in `reason/configs/` so the module stays self-contained and `smoke_reason.py` needs no pipeline config — the same separation `stt/configs/languages.yaml` already has. And `reason.voice` must match `out_lang`: Hindi replies need the Hindi Piper voice (`hi_official_v1`). English-only assistants would switch both the prompt and `en_lessac_medium`.

The system prompt is a file, not a YAML string, because it is prose that will be edited often:

```markdown
<!-- reason/configs/prompts/voice_assistant.md -->
You are a spoken voice assistant, like ChatGPT Voice.
The user speaks English. Always reply in natural Hindi (Devanagari).
Keep replies to at most two short spoken sentences — no lists/markdown/emoji.
```

CLI override in `run_realtime.py`, next to the existing `--stage`:

```python
parser.add_argument("--mode", choices=["translate", "reason"], default=None,
                    help="Text stage between STT and TTS. Overrides `mode` in the config.")
parser.add_argument("--reason-profile", default=None,
                    help="Override reason.profile (see reason/configs/profiles.yaml).")
```

`--stage mt` keeps its meaning of "run the text stage and print, no TTS" in both modes.

---

## 9. Domain layer

Zero third-party imports in any of these four files. That is what makes them testable.

**`reason/src/runtime/messages.py`** — DTOs, mirroring `stt/src/runtime/messages.py`:

```python
"""Shared message types for the reasoning stage."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    role: str          # "user" | "assistant"
    content: str


@dataclass
class Prompt:
    user_text: str
    system: str = ""
    history: tuple[Turn, ...] = ()
    meta: dict = field(default_factory=dict)
```

**`reason/src/runtime/interface.py`** — the port, mirroring `stt/src/runtime/interface.py`:

```python
"""Reasoning engine protocol — the pipeline depends only on this."""
from __future__ import annotations

import threading
from typing import Iterator, Optional, Protocol, runtime_checkable

from .messages import Prompt


@runtime_checkable
class ReasoningEngine(Protocol):
    def stream_reply(
        self, prompt: Prompt, *, cancel: Optional[threading.Event] = None
    ) -> Iterator[str]:
        """Yield reply text deltas in order. Empty reply yields nothing.

        Raises only on genuine failure — never to signal an empty reply.

        `cancel`, if given, may become set *during* iteration (barge-in,
        §19). Engines should stop producing deltas as soon as practical
        once it's set and return normally — this is not a failure.
        """
        ...

    def warmup(self) -> None:
        ...
```

**`reason/src/runtime/streaming.py`** — the pure chunker:

```python
"""Token deltas -> speakable sentences. No model, no I/O, no torch."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…]['\")\]]*(?:\s|$)")
# Terminators that are not sentence ends: initials, abbreviations, decimals.
_NOT_AN_END = re.compile(
    r"(?:\b[A-Z]\.|\b(?:Mr|Mrs|Ms|Dr|Prof|vs|etc|e\.g|i\.e)\.|\d\.\d*)$"
)


class SentenceAssembler:
    """Accumulates deltas and emits complete sentences as they finish.

    `max_chars` bounds the wait: a model that never emits a terminator must
    still produce speakable audio rather than buffering the whole reply.
    """

    def __init__(self, *, max_chars: int = 180) -> None:
        self._buf = ""
        self._max_chars = max_chars

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            emitted = False
            pos = 0
            while (m := _SENTENCE_END.search(self._buf, pos)) is not None:
                head = self._buf[: m.end()]
                if _NOT_AN_END.search(head.rstrip()):
                    # False end ("Dr.", "3.5"). Keep scanning from just past it;
                    # breaking here would stall the buffer until max_chars.
                    pos = m.end()
                    continue
                self._buf = self._buf[m.end() :]
                if head.strip():
                    out.append(head.strip())
                emitted = True
                break
            if not emitted:
                break
        # Loop, not a single cut: one push can overshoot by several chunks.
        while len(self._buf) >= self._max_chars:
            cut = self._buf.rfind(" ", 0, self._max_chars)
            if cut <= 0:
                cut = self._max_chars
            head, self._buf = self._buf[:cut], self._buf[cut:]
            if head.strip():
                out.append(head.strip())
        return out

    def flush(self) -> list[str]:
        tail, self._buf = self._buf.strip(), ""
        return [tail] if tail else []
```

Verified against these cases, which are the unit tests to write in step 2 of §17:

| Input | Output |
|---|---|
| `"Hello"`, `" there"`, `". "`, `"How can"`, `" I help"`, `"?"` | `['Hello there.', 'How can I help?']` |
| `"Dr. Smith is here. "`, `"Next one."` | `['Dr. Smith is here.', 'Next one.']` |
| `"It costs 3.50 dollars. Ok."` | `['It costs 3.50 dollars.', 'Ok.']` |
| `'He said "hi." Then left.'` | `['He said "hi."', 'Then left.']` |
| `"One. Two. Three."` (one delta) | `['One.', 'Two.', 'Three.']` |
| `"Ask Dr."` (never resolves) | `['Ask Dr.']` at flush |
| 300 chars, no terminator | repeated ≤`max_chars` chunks, not one blob |
| no deltas at all | `[]` |

**`reason/src/runtime/history.py`** — bounded conversation memory:

```python
"""Bounded multi-turn memory. Pure; the engine never owns history."""
from __future__ import annotations

from collections import deque

from .messages import Turn


class ConversationHistory:
    def __init__(self, *, max_turns: int = 4) -> None:
        self._turns: deque[Turn] = deque(maxlen=max(0, max_turns) * 2)

    def add_exchange(self, user_text: str, reply: str) -> None:
        if self._turns.maxlen == 0:
            return
        self._turns.append(Turn("user", user_text))
        self._turns.append(Turn("assistant", reply))

    def snapshot(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def clear(self) -> None:
        self._turns.clear()
```

History lives outside the engine on purpose. Keeping it in the engine would make replies depend on hidden state, so identical input could produce different output and the engine would stop being unit-testable.

---

## 10. Infrastructure layer

**`reason/src/runtime/loading.py`** — the load-time policy from [doc 06](../06-debugging-meta-tensor-load-race.md), isolated so every backend gets it for free:

```python
"""Load-time policy: quantization config + post-load guards."""
from __future__ import annotations

from typing import Any, Optional


def build_quantization_config(kind: str, compute_dtype: str) -> Optional[Any]:
    """`nf4` -> BitsAndBytesConfig; `none` -> None. Anything else is an error."""
    import torch
    from transformers import BitsAndBytesConfig

    if kind in (None, "none"):
        return None
    if kind != "nf4":
        raise ValueError(f"Unknown quantization {kind!r} (use 'nf4' or 'none')")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
    )


def assert_materialized(model, what: str) -> None:
    """Fail at the load, naming the parameter, not later inside torch."""
    unmaterialized = [n for n, t in model.named_parameters() if t.is_meta]
    if unmaterialized:
        raise RuntimeError(
            f"{what} load left parameters on the meta device: {unmaterialized}. "
            "See docs/06-debugging-meta-tensor-load-race.md."
        )
```

`assert_materialized` is duplicated from `pipeline/realtime/workers.py` rather than imported. That is intentional: `reason/` must not depend on `pipeline/`, or the module stops being standalone and the dependency rule in §3 breaks. Six duplicated lines is the cheaper trade.

**`reason/src/runtime/prompting.py`** — the only place that knows about chat templates:

```python
"""Prompt -> model-specific prompt string, via the tokenizer's chat template."""
from __future__ import annotations

from .messages import Prompt


class ChatPromptFormatter:
    def __init__(self, tokenizer) -> None:
        if tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer has no chat template — this is a base model, not "
                "instruction-tuned. Use an -it variant (see §6)."
            )
        self._tok = tokenizer

    def render(self, prompt: Prompt) -> str:
        msgs = [{"role": t.role, "content": t.content} for t in prompt.history]
        # Gemma has no system role; fold the instruction into the first user turn.
        head = f"{prompt.system}\n\n{prompt.user_text}" if prompt.system else prompt.user_text
        msgs.append({"role": "user", "content": head})
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
```

Failing loudly on a missing chat template is what turns "the assistant rambles" (§6) from a puzzling behaviour into a startup error naming the cause.

**`reason/src/runtime/gemma_pytorch.py`** — the adapter:

```python
"""Gemma adapter: HF transformers + bitsandbytes, CUDA only."""
from __future__ import annotations

import threading
from typing import Iterator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .loading import assert_materialized, build_quantization_config
from .messages import Prompt
from .prompting import ChatPromptFormatter

# See docs/06: from_pretrained monkeypatches process-global state while loading.
# Callers sharing a process with other model loads must inject their own lock.
_DEFAULT_LOAD_LOCK = threading.Lock()


class _EventStoppingCriteria(StoppingCriteria):
    """Makes generate() return within one decode step of `event` being set,
    instead of running to max_new_tokens regardless (§19)."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._event.is_set()


class GemmaPytorchReasoner:
    def __init__(
        self,
        model_id: str,
        *,
        quantization: str = "nf4",
        compute_dtype: str = "bfloat16",
        max_new_tokens: int = 96,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_lock: Optional[threading.Lock] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Reasoning mode requires CUDA.")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        quant = build_quantization_config(quantization, compute_dtype)
        with load_lock if load_lock is not None else _DEFAULT_LOAD_LOCK:
            tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant,
                dtype=getattr(torch, compute_dtype),
                # Pinned, not "auto": an over-budget model must raise rather
                # than silently offload half its layers to CPU (§5).
                device_map={"": 0},
            )
            assert_materialized(self.model, "reasoning")
        self.model.eval()
        self.tokenizer = tok
        self.formatter = ChatPromptFormatter(tok)

    def warmup(self) -> None:
        for _ in self.stream_reply(Prompt(user_text="hello")):
            pass

    def stream_reply(
        self, prompt: Prompt, *, cancel: Optional[threading.Event] = None
    ) -> Iterator[str]:
        text = self.formatter.render(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if cancel is not None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_EventStoppingCriteria(cancel)])
        gen = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        gen.start()
        try:
            for delta in streamer:
                if cancel is not None and cancel.is_set():
                    break
                yield delta
        finally:
            gen.join(timeout=2.0)
```

Two loading rules that differ from `mt_worker` and will bite if ignored:

- **Never call `.to()` or `.half()` on a 4-bit model.** bitsandbytes raises if you move quantized params. `device_map` performs placement and is the only placement step. `mt_worker` does `model.half().to("cuda")` because Marian is unquantized; do not copy that here.
- **`TextIteratorStreamer` requires `generate` on another thread.** The consumer iterates while the producer generates. The `finally: gen.join()` matters — without it, an abandoned generator leaks a thread that still holds the GPU. With `cancel` set, `generate()` returns almost immediately (§19) so this join is prompt either way; without it, the 2s timeout is just a safety net.

**`reason/src/runtime/registry.py`** — a near-copy of `stt/src/runtime/registry.py`: `load_profiles()`, `resolve_profile(profile_id)`, and `resolve_prompt_path(profile)` reading the prompt markdown relative to `REASON_ROOT`. Same shape, same error messages, no surprises.

**`reason/src/runtime/__init__.py`** — the factory, mirroring `build_engine` in `stt/src/runtime/__init__.py`. This is the **only** place a concrete class is named:

```python
"""Factory: profile -> ReasoningEngine."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from .history import ConversationHistory
from .interface import ReasoningEngine
from .messages import Prompt, Turn
from .registry import resolve_profile, resolve_prompt_path

__all__ = ["build_engine", "ReasoningEngine", "Prompt", "Turn", "ConversationHistory"]


def build_engine(
    profile_id: Optional[str] = None,
    *,
    config_path: Path | str | None = None,
    load_lock: Optional[threading.Lock] = None,
) -> ReasoningEngine:
    profile = resolve_profile(profile_id, config_path)
    if profile.get("device", "cuda") != "cuda":
        raise ValueError("Reasoning profile device must be cuda")

    runtime = profile.get("runtime", "gemma_pytorch")
    if runtime != "gemma_pytorch":
        raise NotImplementedError(f"Runtime {runtime!r} not implemented yet")

    from .gemma_pytorch import GemmaPytorchReasoner  # imported per-runtime

    decode = profile.get("decode") or {}
    return GemmaPytorchReasoner(
        profile["model_id"],
        quantization=profile.get("quantization", "nf4"),
        compute_dtype=profile.get("compute_dtype", "bfloat16"),
        max_new_tokens=int(decode.get("max_new_tokens", 96)),
        temperature=float(decode.get("temperature", 0.7)),
        top_p=float(decode.get("top_p", 0.9)),
        load_lock=load_lock,
    )


def load_system_prompt(profile_id: Optional[str] = None) -> str:
    profile = resolve_profile(profile_id)
    path = resolve_prompt_path(profile)
    return path.read_text(encoding="utf-8").strip() if path else ""
```

The `runtime != ...` check plus the per-runtime import is the open/closed seam: a GGUF backend adds one branch here and touches nothing else. It also keeps `import reason.src.runtime` free of `torch`, so `registry.py` can be tested without a GPU.

---

## 11. Application layer: the worker

Queue plumbing only. No `torch`, no `transformers`, no prompt strings, no regex — all of that is behind the port.

```python
"""Reasoning worker: Q1 transcript_queue -> Q2 translation_queue.

The `--mode reason` counterpart to `mt_worker`. Exactly one of the two runs.
Depends only on the ReasoningEngine port; see
docs/reasoningModel/01-gemma-reasoning-mode.md §3.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .messages import STOP, Abort, Reply, Sentence

log = logging.getLogger("peaktranslation.realtime")

_GET_TIMEOUT_S = 0.5


def reasoning_worker(
    *,
    engine,                      # ReasoningEngine — injected, never constructed here
    history,                     # ConversationHistory
    assembler_factory,           # () -> SentenceAssembler, one per utterance
    system_prompt: str,
    transcript_queue: "queue.Queue",
    reply_queue: "queue.Queue",
    gpu_lock: threading.Lock,
    stop_event: threading.Event,
    out_lang: str = "en",
    speaking: Optional[threading.Event] = None,
    interrupt_event: Optional[threading.Event] = None,   # barge-in, §19
) -> None:
    from reason.src.runtime import Prompt

    try:
        engine.warmup()
        log.info("Reasoning worker ready.")
    except Exception:
        log.exception("Reasoning worker failed to start — pipeline cannot continue.")
        stop_event.set()
        reply_queue.put(STOP)
        return

    while not stop_event.is_set():
        try:
            item = transcript_queue.get(timeout=_GET_TIMEOUT_S)
        except queue.Empty:
            continue
        if item is STOP:
            reply_queue.put(STOP)
            break
        sent: Sentence = item
        if interrupt_event is not None:
            interrupt_event.clear()   # clean slate — a prior barge-in must not leak in

        t0 = time.perf_counter()
        assembler = assembler_factory()
        seq, full, aborted = 0, [], False

        def emit(text: str, is_last: bool) -> None:
            nonlocal seq
            reply_queue.put(Reply(
                utt_id=sent.utt_id, text=text, seq=seq, is_last=is_last,
                out_lang=out_lang, t_captured=sent.t_captured,
                t_reply_done=time.perf_counter(),
            ))
            if seq == 0:
                log.info("[%s] first reply chunk (%.0f ms): %r",
                         sent.utt_id, (time.perf_counter() - t0) * 1000, text)
            seq += 1

        try:
            if speaking is not None:
                speaking.set()          # mute (§14) or barge-in signal (§19)
            prompt = Prompt(
                user_text=sent.text, system=system_prompt, history=history.snapshot()
            )
            # gpu_lock held for the whole generation: this is the only GPU
            # consumer besides STT. Without barge-in, capture is muted so
            # this is safe regardless of duration (§14). With barge-in,
            # `cancel` keeps this window short even when interrupted (§19).
            with gpu_lock:
                for delta in engine.stream_reply(prompt, cancel=interrupt_event):
                    if interrupt_event is not None and interrupt_event.is_set():
                        aborted = True   # worker owns this even if engine ignores cancel
                        break
                    full.append(delta)
                    for s in assembler.push(delta):
                        emit(s, is_last=False)

            if aborted:
                # Void, not truncated: no flush, no is_last, no history update.
                reply_queue.put(Abort(utt_id=sent.utt_id))
                if speaking is not None:
                    speaking.clear()
                continue

            for s in assembler.flush():
                emit(s, is_last=False)
            emit("", is_last=True)      # end-of-utterance marker; TTS skips empties
            reply = "".join(full).strip()
            history.add_exchange(sent.text, reply)
            log.info("[%s] reply (%.0f ms, %d chunks): %r",
                     sent.utt_id, (time.perf_counter() - t0) * 1000, seq, reply)
        except Exception:
            # `speaking` is normally cleared by playback, but on failure no audio
            # will ever play — clear it here or the mic stays muted forever.
            log.exception("[%s] reasoning failed on this sentence; skipping.", sent.utt_id)
            if speaking is not None:
                speaking.clear()
            continue

    log.info("Reasoning worker stopped.")
```

This follows the two robustness rules from the `workers.py` docstring — startup and per-item failures logged with a full traceback rather than silently killing a daemon thread, and queue reads with a timeout that also check `stop_event` so shutdown completes even if an upstream worker died before forwarding `STOP`.

`assembler_factory` rather than a shared assembler: state must not leak between utterances, or a truncated sentence from utterance A gets spoken as part of B.

Because the only ML dependency is the injected `engine`, the whole worker tests against a fake:

```python
class FakeEngine:
    def warmup(self): pass
    def stream_reply(self, prompt, *, cancel=None):
        yield from ["Hello there. ", "How can I help", "?"]
```

That covers chunk ordering, the `is_last` marker, history updates, and `speaking` handling with no GPU and no downloads. Barge-in adds one more fake that sets `cancel` itself partway through, to test the abort path the same way (§19).

---

## 12. TTS and playback changes

Two small edits to `workers.py`, needed only because reason mode emits multiple chunks per utterance.

**The WAV filename collides.** Today:

```297:297:pipeline/realtime/workers.py
        out_wav = spill_dir / f"{tr.utt_id}.wav"
```

With three chunks sharing one `utt_id`, chunk 2 overwrites chunk 1 while playback may still be reading it — and `playback_worker` unlinks the file afterwards, so chunk 3 can vanish before it plays. Include the sequence number, defaulting to 0 so the translate path is byte-for-byte unaffected:

```python
seq = getattr(tr, "seq", 0)
out_wav = spill_dir / f"{tr.utt_id}-{seq}.wav"
```

**Skip empty chunks**, right after popping from Q2 — this is what makes the `is_last` marker harmless:

```python
if not tr.text.strip():
    continue
```

`tts_worker` needs no other change for the translate-mode-compatible v1: it already takes `voice_onnx` as a parameter, and it only reads `.utt_id`, `.text`, and `.t_captured` off the Q2 item, so `Reply` substitutes for `Translation` structurally.

Playback needed no changes at all for v1 — it already serializes strictly one WAV at a time, which is exactly what multi-chunk replies need to come out in order. Its `VAD-close -> audio-out` log now fires per chunk; the `seq == 0` line is the one that measures felt latency. §19 revisits both once barge-in needs to kill audio mid-play.

---

## 13. Message type

`Reply` goes in `pipeline/realtime/messages.py` next to `Translation` — it is a **pipeline** message, not a reason-module one, because it crosses Q2 between two pipeline workers:

```python
@dataclass
class Reply:
    utt_id: str
    text: str
    seq: int          # 0-based chunk index within this utterance
    is_last: bool
    out_lang: str
    t_captured: float
    t_reply_done: float
```

`seq` and `is_last` exist because streaming emits several Q2 items per utterance, which the translate path never does. `Translation` deliberately has no such fields, so don't reuse it.

Barge-in (§19) adds a second message, `Abort`, for the opposite case — "stop, don't play anything more for this utterance":

```python
@dataclass
class Abort:
    utt_id: str
```

It travels the same path as `Reply` (Q2 then Q3) so ordering with any chunks queued just before it is preserved for free.

---

## 14. The feedback loop — the failure that will surprise you

> **v2 note:** this section describes the `barge_in: false` fallback (mute, no interrupt). The default as of §19 is `barge_in: true`, which takes the opposite approach — capture stays live on purpose, so the assistant *can* be interrupted. Read this section for the tradeoff either way; §19 covers what changes.

The mic is always listening. When the speaker plays a reply, the mic hears it, VAD segments it, Whisper transcribes it, and Gemma answers *itself*. Translate mode gets away with this because replies are short and often land mid-cutoff. A 15-second spoken reply reliably triggers an infinite self-conversation.

Fix it with a shared `speaking` event that the VAD loop checks. Frames are already pulled one at a time:

```188:193:pipeline/realtime/capture.py
            while len(buf) >= self.frame_bytes:
                frame, buf = buf[: self.frame_bytes], buf[self.frame_bytes :]
                self._log_level(frame)
                utt_pcm = self._endpointer.push_frame(frame)
                if utt_pcm is not None:
                    self._emit(utt_pcm)
```

Take an optional `speaking: threading.Event` in `AudioCapture.__init__` and drop frames while it is set. Discard rather than buffer — buffered frames would replay the assistant's own voice the moment you unmute:

```python
if self._speaking is not None and self._speaking.is_set():
    continue    # discard; do not feed VAD, do not buffer
```

The reasoning worker sets it when generation starts; **`playback_worker` clears it** after the chunk with `is_last` finishes playing, because only playback knows when sound has actually stopped. Clear it a beat late — `time.sleep(0.2)` before clearing — so the tail of the speaker output doesn't leak into the first frame.

Cost (with `barge_in: false`): no barge-in. You cannot interrupt the assistant mid-reply. That was the deliberate v1 tradeoff, on the reasoning that real barge-in needs echo cancellation — but §19 implements it anyway, without echo cancellation, accepting a different cost (false interrupts on open speakers) instead.

Bonus of the mute approach: because capture is muted for the whole reply, nothing enters Q0 while `gpu_lock` is held for 3–8 seconds. That is why §11 can hold the lock across the entire generation without the drop-oldest policy in `capture.py:_emit` discarding real speech. §19's barge-in path gives this up too — capture stays live, so real speech during a reply does enter Q0 (that's the whole point) — but keeps the window short via cancellation instead.

---

## 15. Composition root

`run_realtime.py` is the only file that names concrete implementations, owns the locks, and wires threads. The translate branch is the existing code verbatim:

```python
mode = args.mode or cfg.get("mode", "translate")

if mode == "reason":
    import sys as _sys
    _sys.path.insert(0, str(ROOT))                    # `reason` package importable
    from reason.src.runtime import (
        ConversationHistory, build_engine, load_system_prompt,
    )
    from reason.src.runtime.streaming import SentenceAssembler
    from pipeline.realtime.reasoning import reasoning_worker
    from pipeline.realtime.workers import MODEL_LOAD_LOCK

    r_cfg = cfg["reason"]
    profile_id = args.reason_profile or r_cfg["profile"]
    voice_onnx = resolve(ROOT, r_cfg["voice"])        # English voice, not Hindi

    # Injected here, nowhere else: the SAME lock STT uses, because a lock only
    # helps if every racing thread takes the same one (docs/06).
    engine = build_engine(profile_id, load_lock=MODEL_LOAD_LOCK)

    t_text = threading.Thread(
        target=reasoning_worker, name="reason", daemon=True,
        kwargs=dict(
            engine=engine,
            history=ConversationHistory(max_turns=r_cfg.get("history_turns", 4)),
            assembler_factory=SentenceAssembler,
            system_prompt=load_system_prompt(profile_id),
            transcript_queue=transcript_queue,
            reply_queue=translation_queue,
            gpu_lock=gpu_lock,
            stop_event=stop_event,
            speaking=speaking,
        ),
    )
else:
    t_text = threading.Thread(target=mt_worker, name="mt", daemon=True, kwargs=dict(...))  # unchanged
```

Building the engine on the main thread rather than inside the worker means a bad profile, a missing licence, or an OOM fails before the mic opens, with a clean traceback instead of a dead daemon thread.

Three other spots need updating:

- The **model-existence preflight** (`run_realtime.py` lines 112–123) checks `mt_model_dir` and the Hindi voice. In reason mode it must check the English voice and *not* require the Marian export. It cannot check `model_id`, which is a Hub ID resolved at load time — log it and let the loader raise the real HF error.
- The **thread-safe import prewarm** (lines 156–173) force-imports transformers classes on the main thread because the lazy module loader is not thread-safe. Add `AutoModelForCausalLM` and `BitsAndBytesConfig` for reason mode.
- Create the shared event next to `gpu_lock`: `speaking = threading.Event()`, and pass it to `AudioCapture`.

---

## 16. Dependencies

`accelerate` is already installed (1.14.0). `bitsandbytes` is **not**. Put it in `reason/requirements.txt`, matching how each stage module owns its own deps:

```text
# reason/requirements.txt — torch/transformers come from stt/ and translate/
bitsandbytes>=0.43.0
```

Reference it from `pipeline/requirements.txt` with a comment, then install and confirm the CUDA build works — bitsandbytes is the single most common source of setup pain here:

```bash
venv/bin/pip install -r reason/requirements.txt
venv/bin/python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

---

## 17. Bring-up order

Each step is independently verifiable, so a failure tells you which layer broke.

1. **Access.** Accept the Gemma licence, then `venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"`.
2. **Domain layer, no GPU.** Unit-test `SentenceAssembler` and `ConversationHistory` with plain strings. Zero downloads. Do this first — it is the fiddliest logic and the cheapest to verify.
3. **Fit.** `scripts/bench_reason.py`: 4-bit load, print VRAM, tokens/sec. If it doesn't fit with ~0.5 GB spare, pick a smaller model now (§6).
4. **Engine alone.** `scripts/smoke_reason.py --text "what is the capital of France?"` — no pipeline, no audio. Confirms the profile, chat template, and reply length.
5. **Worker with `FakeEngine`.** Queue ordering, `is_last`, history, `speaking`. Still no GPU.
6. **`--mode reason --stage mt`.** Real mic, real STT, replies printed not spoken. No TTS, no feedback loop.
7. **`--mode reason --stage full`, headphones, `barge_in: false`.** Headphones so the feedback loop can't fire either way while you verify chunked WAV naming and in-order playback.
8. **Speakers, `barge_in: false`, `mute_capture_while_replying: true`.** The only step that tests §14's mute path. If the assistant answers itself, `speaking` isn't wired.
9. **Barge-in, headphones, `barge_in: true`.** Start a reply, talk over it. Confirm: audio cuts off within roughly one poll tick (§19), the log shows `reply interrupted by barge-in`, and your new utterance gets a normal reply afterward. Still no echo risk on headphones, so this isolates the cancellation/kill machinery from the false-positive risk in step 10.
10. **Barge-in, speakers.** Same test, no headphones. Watch for the assistant interrupting *itself* the moment it starts talking — that's mic bleed, not a bug in the interrupt logic. If it happens, lower `capture.vad.aggressiveness` or set `reason.barge_in: false` (§19).
11. **Regression-check translate mode.** `--mode translate` must behave exactly as before. That is the whole point of the mode split.

---

## 18. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `401 ... access to model is restricted` | licence not accepted, or `HF_TOKEN` unset | §6 |
| `CUDA out of memory` on load | Whisper loaded before Gemma on a 4 GB card, or bf16 instead of 4-bit, or Marian still loaded | Load Gemma before STT (`run_realtime.py`); skip `caching_allocator_warmup` (§10); stay on `nf4` (§5) |
| Loads fine but ~30 s per reply | `device_map="auto"` offloaded layers to CPU | pin `device_map={"": 0}` |
| `Tokenizer has no chat template` | base model, not `-it` | §6 — use an `-it` variant |
| Replies ramble or restate the question | base model, or system prompt not applied | §6, §10 |
| `Cannot copy out of meta tensor` | load raced with the STT load | inject `MODEL_LOAD_LOCK` (§15) |
| `.to()` raises on the quantized model | tried to move 4-bit params | don't; `device_map` places it |
| Assistant answers itself, forever | `speaking` event not wired | §14 |
| Only the last sentence is heard | chunk WAVs collide on `utt_id` | §12 — add `seq` to the filename |
| Sentences split at "Dr." or "3.5" | abbreviation guard missing | §9 — `_NOT_AN_END` |
| Mic never unmutes after an error | `speaking` not cleared on the failure path | §11 |
| `Q0 audio_queue full — dropped oldest` during replies | `gpu_lock` held while capture live | §14 — mute capture |
| Hindi voice reading English replies (or vice versa) | `reason.voice` language doesn't match profile `out_lang` / system prompt | §8 — keep voice and prompt in sync |
| Replies print but Piper never speaks | ran `--stage mt` (text-only bring-up) | use `--stage full` |
| A leaked thread per reply, GPU stays busy | streamer abandoned without `gen.join()` | §10 — `finally` |
| Assistant interrupts itself the instant it starts talking | mic bleed from open speakers, no AEC — `on_speech_start` fired on the assistant's own voice | §19 — headphones, lower `vad.aggressiveness`, or `barge_in: false` |
| Barge-in doesn't cut off playback, waits for the file to finish | `play_wav_blocking` still using `subprocess.run` (blocking) instead of `Popen` + poll | §19 |
| Barge-in interrupts, but the *next* reply is also immediately aborted | `interrupt_event` not cleared at the top of `reasoning_worker`'s loop | §19 — `reasoning.py` |
| STT for the interrupting utterance stalls for the rest of the old reply | `cancel` not wired into `model.generate()`, so `gpu_lock` stays held to completion | §19 — `_EventStoppingCriteria` |
| `TypeError: stream_reply() got an unexpected keyword argument 'cancel'` | a `ReasoningEngine`/`FakeEngine` not updated for the interface change | §4, §9 — add `cancel=None` to its signature |

---

## 19. Barge-in — interrupting the assistant mid-reply

§14 muted the mic for the whole reply specifically to dodge the feedback loop. This section reverses that: capture stays live, and speaking over a reply cuts it off — audio killed mid-file, generation cancelled, mic already listening for what you actually said. Default: `reason.barge_in: true`.

### The one thing this cannot do

Without acoustic echo cancellation, the mic *will* pick up some of the assistant's own voice through open speakers. There is no way around this in software alone without an AEC stage (a much larger project, same conclusion §14 reached and deferred). What changes here is which side of the tradeoff you're on:

- `barge_in: true` (default): capture stays live. Real barge-in works. On open speakers, the assistant may occasionally interrupt itself. Headphones eliminate the risk entirely; on speakers, lowering `capture.vad.aggressiveness` reduces false triggers at the cost of also being slower to notice real speech.
- `barge_in: false`: back to §14's mute. No false interrupts, no real ones either.

Be upfront with anyone using this on speakers: it's a real limitation, not a bug waiting to be fixed by more code.

### Design: three independent mechanisms, one event

"Interrupt" is not one operation — it's three, each needing a different trigger because each stage is blocked in a different way when the barge-in happens:

1. **Detection** has to be fast. Waiting for VAD to fully endpoint the interrupting utterance (silence_ms, ~500 ms after the user stops talking) makes "interrupt" feel laggy. Instead, `VadEndpointer` gains an `on_speech_start` hook fired on the *first* 20 ms frame classified as speech — before endpointing, before STT, before anything. `AudioCapture` forwards it; `run_realtime.py` wires a closure that only treats it as a barge-in if `speaking.is_set()` (the assistant is actually replying right now — otherwise it's just normal speech starting).

   ```95:101:pipeline/realtime/capture.py
           if not self._speaking:
               self._pre_roll.append(frame)
               if is_speech:
                   self._speaking = True
                   self._speech_frames = list(self._pre_roll)
                   self._silence_run = 0
                   if self._on_speech_start is not None:
                       self._on_speech_start()
   ```

2. **Cancelling generation** needs cooperation from `model.generate()`, or it keeps computing tokens nobody will hear for the rest of `max_new_tokens` — wasting GPU time and, worse, keeping `gpu_lock` held so STT for the interrupting utterance stalls behind it. `stream_reply` gains an optional `cancel: threading.Event`, threaded into `model.generate(stopping_criteria=...)` via a tiny `StoppingCriteria` (§9, §10). This is the one place the interface changes; it stays Liskov-safe because it's optional and inert for engines that ignore it (§4).

3. **Killing audio already playing** can't be done through a queue message — `playback_worker` is blocked inside a single `subprocess.run()` call for the file currently playing, and won't see anything else until that call returns. `audio_out.play_wav_blocking` switches to `Popen` + a poll loop (`_POLL_S = 0.03`) that checks the same event and `terminate()`s the process early.

One `threading.Event` (`interrupt_event`, created in `run_realtime.py` next to `speaking`) drives all three: `on_speech_start`'s closure sets it, `reasoning_worker` passes it as `cancel` *and* polls it independently (defense in depth — some future engine might not honour `cancel`), and both `tts_worker` and `playback_worker` check it to skip stale chunks. `reasoning_worker` clears it at the top of every utterance, before anything for that utterance is produced — otherwise a barge-in that just fired would immediately abort the very reply it triggered.

### What "abort" means for the interrupted reply

Not truncated — void. No `assembler.flush()` of the trailing partial sentence, no `is_last` marker, no `history.add_exchange()`. Half a sentence in the model's conversational memory would make its *next* reply nonsensical ("as I was saying, ..." about something that was never said). Instead, `reasoning_worker` pushes one `Abort(utt_id)` (§13) into Q2 so `tts_worker` and `playback_worker` know a reply that was in flight is now void, and `speaking.clear()`s immediately — it doesn't wait for playback's normal `is_last`-triggered clear (§14), because that clear is never coming for an aborted reply.

Ordering matters here and is free: `Abort` rides the same FIFO queues as `Reply`. Anything queued *before* the `Abort` was legitimately committed and still plays; nothing is queued *after* it, because `reasoning_worker` stops producing the instant it detects the interrupt. `tts_worker` and `playback_worker` additionally skip any item while `interrupt_event` is still set — belt-and-suspenders for the handful of chunks that might already be sitting in Q2/Q3 ahead of the `Abort` catching up.

### Why STT doesn't need its own GPU lock

An earlier version of this design considered giving `stt_worker` a separate `gpu_lock` from the text stage, reasoning that otherwise STT for the interrupting utterance would stall behind `gpu_lock` for however long the old generation had left (§11 already holds it for the whole reply). That turned out to be unnecessary: because `cancel` makes `model.generate()` return within about one decode step of the interrupt firing, the lock is held only briefly past the interrupt regardless — not for the multi-second tail of the original reply. Splitting the lock would have meant `stt_worker` and the text stage running truly concurrently on a 4 GB GPU, for a latency win cancellation already delivers more simply. Translate mode's locking is completely untouched by any of this.

### Config

```yaml
reason:
  barge_in: true                      # default; false restores §14's mute-only behaviour
  mute_capture_while_replying: true   # only consulted when barge_in is false
```

`barge_in: true` overrides `mute_capture_while_replying` in the composition root — the two are mutually exclusive ways of handling the same feedback loop, and capture cannot be simultaneously muted (for echo safety) and live (for barge-in).
