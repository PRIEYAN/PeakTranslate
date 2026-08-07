"""Gemma adapter: HF transformers + bitsandbytes, CUDA only.

See docs/reasoningModel/01-gemma-reasoning-mode.md §10 for the loading
rules this file follows (never .to()/.half() a 4-bit model; device_map is
pinned, not "auto"; generate() runs on its own thread for the streamer).
"""
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

from .loading import (
    assert_materialized,
    build_quantization_config,
    skip_caching_allocator_warmup,
)
from .messages import Prompt
from .prompting import ChatPromptFormatter

# See docs/06: from_pretrained monkeypatches process-global state while loading.
# Callers sharing a process with other model loads (the pipeline) must inject
# their own shared lock via `load_lock`; this default only protects against
# other GemmaPytorchReasoner instances in the same process.
_DEFAULT_LOAD_LOCK = threading.Lock()


class _EventStoppingCriteria(StoppingCriteria):
    """Makes `model.generate()` return within one decode step of `event`
    being set, instead of running to `max_new_tokens` regardless. This is
    what makes barge-in cancellation actually free the GPU promptly rather
    than merely abandoning a streamer nobody's reading from anymore. See
    docs/reasoningModel/01-gemma-reasoning-mode.md §19.
    """

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
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
            # Free any stale CUDA cache from a previous failed load / other
            # process leftovers before we touch the GPU.
            torch.cuda.empty_cache()
            tok = AutoTokenizer.from_pretrained(model_id)
            # See loading.skip_caching_allocator_warmup — required on ≤4 GB
            # when Whisper is (or was) resident; the stock warmup OOMs.
            with skip_caching_allocator_warmup():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant,
                    dtype=getattr(torch, compute_dtype),
                    # Pinned, not "auto": an over-budget model must raise rather
                    # than silently offload half its layers to CPU (see §5).
                    device_map={"": 0},
                )
            assert_materialized(self.model, "reasoning")
            torch.cuda.empty_cache()
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
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_EventStoppingCriteria(cancel)]
            )
        gen = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        gen.start()
        try:
            for delta in streamer:
                if cancel is not None and cancel.is_set():
                    # Also bail our own iteration immediately rather than
                    # waiting for one more delta the streamer may already
                    # have buffered — every token counts for felt latency.
                    break
                yield delta
        finally:
            # Without this, an abandoned generator leaks a thread that still
            # holds the GPU (see §10). Prompt with cancel set (generate()
            # returns within ~one decode step); a plain timeout otherwise.
            gen.join(timeout=2.0)
