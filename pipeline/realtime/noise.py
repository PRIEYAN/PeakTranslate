"""Drop Whisper hallucinations / filler so Gemma doesn't answer noise.

USB mics + room hiss commonly produce short false transcripts ("you", "mm",
"Bye", "Thank you."). Feeding those to a voice assistant creates a spam loop.
Pure string rules — no model, no I/O.
"""
from __future__ import annotations

import re
import unicodedata

# Whole-utterance junk Whisper emits on silence / clicks / speaker bleed.
_BLOCKLIST = frozenset(
    {
        "a",
        "ah",
        "ahh",
        "and",
        "bye",
        "goodbye",
        "hey",
        "hi",
        "hello",
        "hmm",
        "hm",
        "huh",
        "mm",
        "mmm",
        "mhm",
        "nah",
        "no",
        "oh",
        "ok",
        "okay",
        "so",
        "the",
        "to",
        "uh",
        "um",
        "umm",
        "yes",
        "yeah",
        "yep",
        "yo",
        "you",
        "your",
        "thank",
        "thanks",
        "thank you",
        "thank you.",
        "thanks for watching",
        "thank you for watching",
        "subscribe",
        "please subscribe",
        "bye bye",
        "okay.",
        "ok.",
        "hmm.",
        "you.",
        "yeah.",
    }
)

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).strip().lower()
    t = _NON_WORD.sub(" ", t)
    return _WS.sub(" ", t).strip()


def is_noise_transcript(text: str, *, min_chars: int = 8, min_words: int = 3) -> bool:
    """True when `text` should not be sent to the text stage.

    Rules (any match → drop):
    - empty / whitespace only
    - too short after stripping punctuation
    - entire utterance is a known Whisper filler / hallucination
    - fewer than `min_words` and every word is in the blocklist
    """
    raw = (text or "").strip()
    if not raw:
        return True
    norm = _normalize(raw)
    if not norm:
        return True
    if norm in _BLOCKLIST:
        return True
    words = norm.split()
    if len(norm) < min_chars and len(words) < min_words:
        return True
    if len(words) < min_words and all(w in _BLOCKLIST for w in words):
        return True
    # Common YouTube-outro hallucination, any casing/punctuation.
    if "thank you for watching" in norm or "thanks for watching" in norm:
        return True
    return False


def pcm_peak_abs(pcm: bytes) -> int:
    """Max absolute int16 sample — cheap loudness gate before STT."""
    if not pcm:
        return 0
    # int16 little-endian, 2 bytes/sample
    n = len(pcm) // 2
    if n == 0:
        return 0
    peak = 0
    for i in range(0, n * 2, 2):
        sample = int.from_bytes(pcm[i : i + 2], "little", signed=True)
        a = -sample if sample < 0 else sample
        if a > peak:
            peak = a
    return peak


_EMOJI_OR_DINGBAT = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # misc symbols & pictographs … symbols extended
    "\U00002700-\U000027BF"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"            # ZWJ
    "]+",
    flags=re.UNICODE,
)


def strip_unspeakable(text: str) -> str:
    """Remove emoji / dingbats Piper cannot speak."""
    return _EMOJI_OR_DINGBAT.sub("", text or "").strip()


def pick_speech_lang(text: str) -> str:
    """Pick Piper voice key from reply script.

    hi = Devanagari (Hindi Piper). Everything else uses the English Piper —
    there is no Tamil Piper voice in this repo, so the Jarvis prompt asks
    Gemma to Latin-transliterate Tamil/other languages for speech.
    """
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:  # Devanagari
            return "hi"
    return "en"
