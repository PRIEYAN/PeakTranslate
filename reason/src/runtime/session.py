"""Jarvis keyword → sticky session mode (separate from chat history).

Wake word ``jarvis`` changes how the assistant behaves:

- ``jarvis translate everything into tamil`` → store a sticky translate mode;
  later utterances (even without the wake word) are translated into Tamil.
- ``jarvis what is a blockchain`` → new Jarvis command: clear chat history,
  clear any sticky mode, answer the question as a normal assistant.

Chat history (ConversationHistory) is turn memory. Sticky mode lives here so
a 30s history TTL cannot wipe a standing order the user just set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_WAKE = re.compile(
    r"^\s*jarvis\b[\s,.:\-]*",
    re.IGNORECASE,
)

_LANG_TAIL = r"(?:the\s+)?([a-zA-Z]+)"

# Standing orders that set sticky translate/speak-in mode.
_MODE_TRANSLATE = re.compile(
    r"(?:translate|convert|switch)\s+"
    r"(?:everything(?:\s+that)?(?:\s+(?:i|we))?\s+say|all(?:\s+that)?(?:\s+(?:i|we))?\s+say|"
    r"everything|all|all\s+(?:speech|audio|text)|anything|any\s+language)?"
    r".*?\b(?:into|to|in)\s+" + _LANG_TAIL,
    re.IGNORECASE,
)
_MODE_ALWAYS = re.compile(
    r"(?:always\s+)?(?:speak|reply|answer|talk|respond)(?:\s+only)?\s+"
    r"(?:in|into|to)\s+" + _LANG_TAIL,
    re.IGNORECASE,
)
_MODE_SHORT = re.compile(
    r"^(?:translate|convert)\s+(?:into|to|in)\s+" + _LANG_TAIL + r"\s*$",
    re.IGNORECASE,
)

_LANG_ALIASES = {
    "tamil": "tamil",
    "tamizh": "tamil",
    "ta": "tamil",
    "telugu": "telugu",
    "te": "telugu",
    "hindi": "hindi",
    "hin": "hindi",
    "english": "english",
    "en": "english",
    "eng": "english",
}

_DYNAMIC_SYSTEM_EXTRA = """\
MODE: {mode}
TARGET_LANG: {lang}

You are in a voice pipeline. Output ONLY what should be spoken.

Rules:
1. Reply entirely in TARGET_LANG. No English unless TARGET_LANG is English.
2. Translate meaning, do not answer as a chatbot, do not confirm, greet, explain, or add extras.
3. At most two short spoken sentences.
4. Script choice (automatic):
   - If TARGET_LANG has a native writing system the speech engine can read (e.g. Hindi Devanagari), use that script.
   - Otherwise (Tamil, Telugu, Spanish, Japanese, French, German, Chinese, etc. on English TTS): write clear Latin-letter transliteration of spoken TARGET_LANG — no native script, no romaji/kana mix unless Latin is natural for that language (Spanish/French stay in their normal Latin spelling).
5. Preserve names and numbers. Do not invent content.

MODE:
- mode_set → translate the user's command phrase into TARGET_LANG only
- sticky_translate → translate the user's utterance into TARGET_LANG only"""


@dataclass(frozen=True)
class StickyMode:
    """Standing order that applies until the next Jarvis command."""

    kind: str  # "translate"
    target_lang: str  # normalized language name (any supported utterance)
    raw: str  # original user wording


@dataclass(frozen=True)
class JarvisTurn:
    """One user utterance after Jarvis routing."""

    user_text: str
    system_extra: str
    reset_history: bool
    mode: Optional[StickyMode]  # current sticky mode after this turn
    log_note: str = ""


def _normalize_lang(token: str) -> Optional[str]:
    t = token.strip().lower()
    if not t or not re.fullmatch(r"[a-z]+", t):
        return None
    return _LANG_ALIASES.get(t, t)


def _system_extra(mode: str, lang: str) -> str:
    return _DYNAMIC_SYSTEM_EXTRA.format(mode=mode, lang=lang)


def _extract_mode(command: str) -> Optional[StickyMode]:
    for pattern in (_MODE_TRANSLATE, _MODE_ALWAYS, _MODE_SHORT):
        m = pattern.search(command)
        if not m:
            continue
        lang = _normalize_lang(m.group(1))
        if lang is None:
            continue
        return StickyMode(kind="translate", target_lang=lang, raw=command.strip())
    return None


class JarvisSession:
    """Owns sticky mode. Chat history stays in ConversationHistory."""

    def __init__(self) -> None:
        self._mode: Optional[StickyMode] = None

    @property
    def mode(self) -> Optional[StickyMode]:
        return self._mode

    def clear_mode(self) -> None:
        self._mode = None

    def route(self, text: str) -> JarvisTurn:
        raw = (text or "").strip()
        wake = _WAKE.match(raw)
        if wake:
            command = raw[wake.end() :].strip()
            if not command:
                return JarvisTurn(
                    user_text="I'm listening.",
                    system_extra="The user only said your name. Ask briefly how you can help.",
                    reset_history=True,
                    mode=None,
                    log_note="jarvis wake (no command) — history cleared, mode cleared",
                )

            mode = _extract_mode(command)
            if mode is not None:
                self._mode = mode
                return JarvisTurn(
                    # Reply = Tamil (etc.) translation of this exact phrase, e.g.
                    # "translate everything to tamil" → "tamil-la ellaa parimaanam sei"
                    user_text=command,
                    system_extra=_system_extra("mode_set", mode.target_lang),
                    reset_history=True,
                    mode=mode,
                    log_note=f"jarvis mode set: translate→{mode.target_lang}",
                )

            # New Jarvis question/command: wipe sticky mode + history.
            self._mode = None
            return JarvisTurn(
                user_text=command,
                system_extra="",
                reset_history=True,
                mode=None,
                log_note="jarvis command — history cleared, mode cleared",
            )

        # No wake word: apply sticky mode if any.
        if self._mode is not None and self._mode.kind == "translate":
            return JarvisTurn(
                user_text=raw,
                system_extra=_system_extra("sticky_translate", self._mode.target_lang),
                reset_history=False,
                mode=self._mode,
                log_note=f"sticky translate→{self._mode.target_lang}",
            )

        return JarvisTurn(
            user_text=raw,
            system_extra="",
            reset_history=False,
            mode=None,
            log_note="default assistant",
        )
