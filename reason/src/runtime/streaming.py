"""Token deltas -> speakable sentences. No model, no I/O, no torch.

See docs/reasoningModel/01-gemma-reasoning-mode.md §9 for the verified
input/output table this class is tested against.
"""
from __future__ import annotations

import re

# Latin terminators plus Devanagari danda / double-danda (Hindi replies from
# Gemma — reason mode's default voice assistant speaks Hindi).
_SENTENCE_END = re.compile(r"[.!?…।॥]['\")\]]*(?:\s|$)")
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
