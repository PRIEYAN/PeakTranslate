"""Reasoning engine protocol — the pipeline depends only on this.

Mirrors stt/src/runtime/interface.py's STTEngine: a narrow port, matched by
size to what the worker actually needs (docs/reasoningModel/01-gemma-
reasoning-mode.md §4, interface segregation).
"""
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

        `cancel`, if given, may become set *during* iteration (barge-in —
        docs/reasoningModel/01-gemma-reasoning-mode.md §19). Engines should
        stop producing deltas as soon as practical once it's set and return
        normally — cancellation is not a failure, so this must not raise.
        Engines that can't support early stopping may ignore it; the worker
        still stops consuming deltas either way, it just frees the GPU
        sooner if the engine cooperates.
        """
        ...

    def warmup(self) -> None:
        ...
