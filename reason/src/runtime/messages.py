"""Shared message types for the reasoning stage.

Mirrors stt/src/runtime/messages.py: plain DTOs, no behaviour, no
third-party imports. See docs/reasoningModel/01-gemma-reasoning-mode.md §9.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Prompt:
    user_text: str
    system: str = ""
    history: tuple[Turn, ...] = ()
    meta: dict = field(default_factory=dict)
