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
    r"^\s*(?:(?:hey|ok|okay|hi)\s+)?jarvis\b[\s,.:\-]*",
    re.IGNORECASE,
)

# Standing orders that set sticky translate/speak-in mode.
_MODE_TRANSLATE = re.compile(
    r"(?:translate|convert|switch)\s+"
    r"(?:everything|all|all\s+(?:speech|audio|text)|anything|any\s+language)?"
    r".*?\b(?:into|to|in)\s+([a-zA-Z]+)",
    re.IGNORECASE,
)
_MODE_ALWAYS = re.compile(
    r"(?:always\s+)?(?:speak|reply|answer|talk|respond)(?:\s+only)?\s+"
    r"(?:in|into|to)\s+([a-zA-Z]+)",
    re.IGNORECASE,
)
_MODE_SHORT = re.compile(
    r"^(?:translate|convert)\s+(?:into|to|in)\s+([a-zA-Z]+)\s*$",
    re.IGNORECASE,
)

_LANG_ALIASES = {
    "tamil": "tamil",
    "tamizh": "tamil",
    "ta": "tamil",
    "hindi": "hindi",
    "hin": "hindi",
    "hi": "hindi",
    "english": "english",
    "en": "english",
    "eng": "english",
}


@dataclass(frozen=True)
class StickyMode:
    """Standing order that applies until the next Jarvis command."""

    kind: str  # "translate"
    target_lang: str  # normalized: tamil | hindi | english
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
    return _LANG_ALIASES.get(token.strip().lower())


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


def _translate_system_extra(mode: StickyMode) -> str:
    lang = mode.target_lang
    script_rule = {
        "hindi": "Use Devanagari script.",
        "tamil": (
            "Write spoken Tamil in clear Latin-letter transliteration "
            "(not Tamil script) so English TTS can read it."
        ),
        "english": "Use plain English.",
    }.get(lang, f"Reply in {lang}.")
    return (
        f"STANDING ORDER (active until the user says Jarvis again): "
        f"translate EVERYTHING the user says into {lang}. "
        f"Do not answer as a chatbot — output only the {lang} translation "
        f"of their words, in at most two short spoken sentences. {script_rule}"
    )


def _mode_confirm_system_extra(mode: StickyMode) -> str:
    lang = mode.target_lang
    script_rule = {
        "hindi": "Confirm in Hindi (Devanagari).",
        "tamil": "Confirm in Latin-letter Tamil transliteration.",
        "english": "Confirm in English.",
    }.get(lang, f"Confirm in {lang}.")
    return (
        f"The user just set a standing order: translate everything they say "
        f"into {lang} from now on. Briefly confirm you will do that. {script_rule} "
        f"One short sentence."
    )


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
                    user_text=f"Standing order: {command}",
                    system_extra=_mode_confirm_system_extra(mode),
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
                system_extra=_translate_system_extra(self._mode),
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
