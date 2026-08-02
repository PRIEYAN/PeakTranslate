"""Shared message / result types for STT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioChunk:
    pcm: bytes
    sample_rate: int
    timestamp: float = 0.0
    session_id: str = ""


@dataclass
class Transcript:
    text: str
    lang: Optional[str] = None
    confidence: Optional[float] = None
    is_final: bool = True
    session_id: str = ""
    is_speech: bool = True  # False when model returns empty (non-speech reject)
    meta: dict = field(default_factory=dict)
