# 06 — Debugging "Cannot copy out of meta tensor" at worker startup

A field guide for one specific startup crash in `pipeline/run_realtime.py`:

```text
2026-08-06 21:41:35,221 ERROR STT worker failed to start — pipeline cannot continue.
Traceback (most recent call last):
  File ".../pipeline/realtime/workers.py", line 54, in stt_worker
    engine = WhisperPytorchSTT(
  File ".../stt/src/runtime/whisper_pytorch.py", line 47, in __init__
    self.model.to(self.device)
  ...
NotImplementedError: Cannot copy out of meta tensor; no data! Please use
torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module
from meta to a different device.
```

Same crash also appears in `mt_worker` at `model.half().to("cuda")`.

Environment where this was diagnosed: `transformers` 5.14.1, `torch` 2.13.0+cu130, Python 3.11.

---

## 1. What the error actually means

A **meta tensor** is a tensor that has a shape and dtype but no allocated storage — a placeholder. `transformers` v5 always builds the model skeleton on the meta device first (`with torch.device("meta"):` in `modeling_utils.py`), then materializes each parameter by streaming it in from the checkpoint file.

`.to("cuda")` copies bytes from source storage to GPU storage. A meta tensor has no source bytes, so the copy is impossible and torch raises `NotImplementedError`.

So the message is not really about `.to()`. It means: **at least one parameter never got materialized during loading, and you only found out later when you tried to move the model.** The load "succeeded" and returned a model object; it is just quietly incomplete.

**Do not follow the error message's advice.** `to_empty()` will move the module and silence the crash by giving those parameters fresh uninitialized memory — i.e. random garbage weights. The pipeline would then start up happily and emit nonsense transcriptions, which is far worse than a crash.

---

## 2. The tell in the log: the LOAD REPORT

Look immediately above the traceback:

```text
[transformers] WhisperForConditionalGeneration LOAD REPORT from: .../en-hi-base-v1-fp16
Key             | Status  |
----------------+---------+-
proj_out.weight | MISSING |
```

`transformers` only prints this table when keys are missing or unexpected, so **its presence is the signal**. The MISSING key is the parameter left on meta, and it names the culprit precisely.

The same run shows the MT model with three:

```text
[transformers] MarianMTModel LOAD REPORT from: .../en-hi-v1-fp16
lm_head.weight                    | MISSING |
model.encoder.embed_tokens.weight | MISSING |
model.decoder.embed_tokens.weight | MISSING |
```

These are all **tied weights**. Whisper declares `_tied_weights_keys = {"proj_out.weight": "model.decoder.embed_tokens.weight"}`; Marian ties `lm_head` and both `embed_tokens` to `model.shared.weight`. Tied weights are deliberately *not* stored in the checkpoint — they are supposed to be re-pointed at their source tensor by `model.tie_weights()` at the end of loading. Every MISSING key here is a tied one, which says the crash is about tying, not about a corrupt or truncated `model.safetensors`.

---

## 3. How to find it: narrow it down in three steps

Work in this order. Each step rules out a whole class of cause.

### Step 3.1 — Load the model alone, in one thread

The single most useful experiment. If the model loads clean in isolation, the checkpoint is fine and the bug is in *how* the pipeline loads it.

```bash
venv/bin/python - <<'PY'
from transformers import WhisperForConditionalGeneration
p = "stt/models/export/en-hi-base-v1-fp16"
m = WhisperForConditionalGeneration.from_pretrained(p)
print("META:", [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_meta])
print("tied ok:", m.proj_out.weight is m.model.decoder.embed_tokens.weight)
PY
```

Observed result:

```text
META: []
tied ok: True
```

No LOAD REPORT printed at all, no meta tensors, tying worked. **The checkpoint is healthy.** If instead you *do* see meta tensors here, stop and go to §6 — you have a genuinely broken export.

### Step 3.2 — Check for meta tensors at the load site, not at `.to()`

The traceback blames `.to()`, which is the symptom's location, not the cause's. Assert right after `from_pretrained` so the failure names the actual bad parameter:

```python
model = WhisperForConditionalGeneration.from_pretrained(path)
bad = [n for n, t in model.named_parameters() if t.is_meta]
assert not bad, f"unmaterialized params after load: {bad}"
```

### Step 3.3 — Reproduce the concurrency

`run_realtime.py` starts `stt_worker` and `mt_worker` as separate threads, and each calls `from_pretrained` in its own thread. Reproduce exactly that:

```bash
venv/bin/python - <<'PY'
import threading
from transformers import WhisperForConditionalGeneration, AutoModelForSeq2SeqLM

STT = "stt/models/export/en-hi-base-v1-fp16"
MT  = "translate/models/export/en-hi-v1-fp16"
res = {}

def load(name, fn):
    m = fn()
    res[name] = [n for n, t in m.named_parameters() if t.is_meta]

for trial in range(3):
    res.clear()
    ts = [threading.Thread(target=load, args=("stt", lambda: WhisperForConditionalGeneration.from_pretrained(STT))),
          threading.Thread(target=load, args=("mt",  lambda: AutoModelForSeq2SeqLM.from_pretrained(MT)))]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"trial {trial}:", res)
PY
```

Observed result — meta tensors on every trial, and *which* model gets corrupted changes run to run:

```text
trial 0: {'stt': ['proj_out.weight'], 'mt': []}
trial 1: {'mt': ['model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight'], 'stt': ['proj_out.weight']}
trial 2: {'stt': ['proj_out.weight'], 'mt': ['model.encoder.embed_tokens.weight', ...]}
```

That non-determinism is the confirmation. It also explains the confusing production logs: the 21:30 run had *both* workers fail, the 21:41 run had STT fail while MT logged `MT ready. (cuda=True)`. Nothing was fixed in between — the race just landed differently.

---

## 4. Root cause: `from_pretrained` is not thread-safe

`transformers` suppresses weight tying during loading, because tying would drop entries from the state dict before they can be checked. It does this by **monkeypatching the class attribute** — see `venv/lib/python3.11/site-packages/transformers/initialization.py`:

```python
@contextmanager
def no_tie_weights():
    from .modeling_utils import PreTrainedModel

    def empty_func(*args, **kwargs):
        pass

    try:
        original_tie_weights = PreTrainedModel.tie_weights
        PreTrainedModel.tie_weights = empty_func
        yield
    finally:
        PreTrainedModel.tie_weights = original_tie_weights
```

`PreTrainedModel.tie_weights` is process-global state. `no_init_weights()` in the same file is worse: it patches `PreTrainedModel.init_weights` *and* the torch init functions on every imported module.

With two threads loading at once, this interleaves:

1. Thread A (MT) enters `no_tie_weights()` → `PreTrainedModel.tie_weights` is now a no-op **for every thread**.
2. Thread B (STT) finishes streaming weights and calls `model.tie_weights()` → gets A's no-op. `proj_out.weight` is never pointed at `model.decoder.embed_tokens.weight`, so it stays the meta placeholder created during skeleton init.
3. Thread A exits its context and restores the real `tie_weights`, so A itself loads fine. B is silently broken.
4. Swap the timing and the victim swaps too — hence the run-to-run variation.

There is a second, independent hazard in the `finally` block: if both threads are inside the context manager, whichever exits second restores `original_tie_weights` captured from an already-patched state, permanently leaving the no-op installed.

### Why `low_cpu_mem_usage=False` does not help

The obvious-looking fix is to opt out of the meta-device fast path. It does nothing on `transformers` 5.x — the argument was removed and is silently discarded (`modeling_utils.py` line 4240):

```python
# Not used anymore -- remove them from the kwargs
for name in ["mirror", "_fast_init", "low_cpu_mem_usage", "from_tf", "from_flax", "offload_state_dict"]:
    _ = kwargs.pop(name, None)
```

`from_pretrained` accepts it and ignores it, so it silences nothing and misleads the next reader. Three places in this repo used to pass it with a comment claiming it prevented the meta-tensor crash — `stt/src/runtime/whisper_pytorch.py`, `pipeline/realtime/workers.py`, and `pipeline/run_pc.py`. All three have been reverted to a plain `from_pretrained`.

This is also why the fix "appeared to work" for MT on the 21:41 run: MT won the race that time. Nothing was actually fixed.

---

## 5. The fix (applied): serialize model loading

Loading is a once-at-startup cost, so serializing it is free. Concurrency only matters for inference, which stays parallel under `gpu_lock`.

One process-wide lock is held across the whole load — construction *and* the `.to(device)` / `.half()` move — and every load is then checked for leftover meta tensors.

`pipeline/realtime/workers.py` owns the lock and the check:

```python
MODEL_LOAD_LOCK = threading.Lock()


def _assert_materialized(model, what: str) -> None:
    unmaterialized = [n for n, t in model.named_parameters() if t.is_meta]
    if unmaterialized:
        raise RuntimeError(
            f"{what} load left parameters on the meta device: {unmaterialized}. "
            "See docs/06-debugging-meta-tensor-load-race.md."
        )
```

`mt_worker` and the MT CPU-fallback loader wrap their loads in it. Note that `torch.cuda.is_available()` is queried *before* taking the lock, so the critical section stays as short as possible:

```python
use_cuda = torch.cuda.is_available()
with MODEL_LOAD_LOCK:
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    _assert_materialized(model, "MT")
    if use_cuda:
        model = model.half().to("cuda")
model.eval()
```

`WhisperPytorchSTT.__init__` does its own `from_pretrained`, so it takes an optional `load_lock` parameter that falls back to a module-level `_DEFAULT_LOAD_LOCK`. That keeps `stt/src/runtime/` usable standalone (`run_pc.py`, eval scripts) while letting the pipeline pass the lock that MT also uses — the fallback lock alone would *not* protect against MT, since a lock only helps if every racing thread takes the same one. `stt_worker` passes `load_lock=MODEL_LOAD_LOCK`.

The guard means a regression now fails at the load, naming the parameter, instead of confusingly at `.to()`:

```text
RuntimeError: STT load left parameters on the meta device: ['proj_out.weight'].
See docs/06-debugging-meta-tensor-load-race.md.
```

Verification — the §3.3 concurrent load, using the real `MODEL_LOAD_LOCK` and `_assert_materialized` from `workers.py`, 5 trials:

```text
trial 0: {'mt': 'ok', 'stt': 'ok'}
trial 1: {'mt': 'ok', 'stt': 'ok'}
trial 2: {'mt': 'ok', 'stt': 'ok'}
trial 3: {'mt': 'ok', 'stt': 'ok'}
trial 4: {'mt': 'ok', 'stt': 'ok'}
```

Clean every time, versus broken every time without the lock.

`pipeline/run_pc.py` is one-shot and single-threaded, so it needs no lock — only the dead `low_cpu_mem_usage=False` was removed there.

### Related precedent

`run_realtime.py` already force-imports the transformers classes on the main thread before starting workers, because the lazy module loader is also not thread-safe (`ImportError: cannot import name X from transformers`). That prewarm fixes the *import*; it does not touch the *load*. Both are the same underlying theme, and the rule for this codebase is: **anything `transformers` does exactly once at startup should be assumed thread-hostile and serialized on the main thread or behind a lock.**

---

## 6. If the model is genuinely broken instead

If §3.1 shows meta tensors even for a single-threaded load, the race is not your problem. Check, in order:

- **Does `config.json` say `"tie_word_embeddings": true`?** If a checkpoint omits the tied weight but the config disables tying, nothing will ever fill it in. The STT config correctly has `"tie_word_embeddings": true`.
- **Does the export actually contain the key?** Inspect the safetensors header:

```bash
venv/bin/python -c "
from safetensors import safe_open
with safe_open('stt/models/export/en-hi-base-v1-fp16/model.safetensors','pt') as f:
    ks=list(f.keys()); print(len(ks)); print([k for k in ks if 'embed_tokens' in k or 'proj_out' in k])"
```

- **Is a MISSING key not in `_tied_weights_keys`?** Then it is a real architecture/checkpoint mismatch — a genuinely absent weight, randomly initialized. Re-export or re-train; do not paper over it.

---

## 7. Checklist

| Symptom | Meaning | Action |
|---|---|---|
| `Cannot copy out of meta tensor` on `.to(device)` | some parameter never materialized during load | find the LOAD REPORT above it |
| LOAD REPORT lists only tied keys | tying was skipped | serialize loading (§5) |
| Which worker fails changes between runs | thread race, not a bad checkpoint | serialize loading (§5) |
| Single-threaded load is also dirty | bad export or config | §6 |
| Tempted by `to_empty()` | would run with random weights | never; fix the load |
| `low_cpu_mem_usage=False` "fixed it" | no-op on transformers 5.x; you won the race | remove it, apply §5 |

Adding a new model to the pipeline? Load it under `MODEL_LOAD_LOCK` and run it through `_assert_materialized`, both in `pipeline/realtime/workers.py`.
