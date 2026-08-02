"""STT engine protocol — orchestrator depends only on this."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .messages import AudioChunk, Transcript


@runtime_checkable
class STTEngine(Protocol):
    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: Optional[str] = None,
    ) -> Transcript:
        """Transcribe PCM. Empty text means non-speech / rejected vocalization."""
        ...

    def warmup(self) -> None:
        ...
