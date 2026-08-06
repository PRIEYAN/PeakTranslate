"""Bounded multi-turn memory. Pure; the engine never owns history.

History lives outside the engine on purpose: keeping it in the engine would
make replies depend on hidden state, so identical input could produce
different output and the engine would stop being unit-testable.
"""
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
