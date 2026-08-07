"""Jarvis wake-word routing + sticky translate mode."""
from reason.src.runtime.session import JarvisSession


def test_mode_set_translate_everything_into_tamil():
    s = JarvisSession()
    turn = s.route("jarvis translate everything into tamil")
    assert turn.reset_history is True
    assert s.mode is not None
    assert s.mode.target_lang == "tamil"
    assert "STANDING ORDER" in turn.system_extra or "standing order" in turn.system_extra.lower()
    assert "tamil" in turn.system_extra.lower()


def test_hey_jarvis_prefix_and_always_speak_hindi():
    s = JarvisSession()
    turn = s.route("hey jarvis always speak in hindi")
    assert turn.reset_history is True
    assert s.mode is not None
    assert s.mode.target_lang == "hindi"


def test_sticky_mode_applies_without_wake_word():
    s = JarvisSession()
    s.route("jarvis translate everything to tamil")
    turn = s.route("good morning friends")
    assert turn.reset_history is False
    assert turn.user_text == "good morning friends"
    assert "translate EVERYTHING" in turn.system_extra
    assert "tamil" in turn.system_extra.lower()
    assert s.mode is not None


def test_new_jarvis_question_clears_mode_and_requests_history_reset():
    s = JarvisSession()
    s.route("jarvis translate everything into tamil")
    assert s.mode is not None
    turn = s.route("jarvis what is a blockchain")
    assert turn.reset_history is True
    assert s.mode is None
    assert turn.user_text == "what is a blockchain"
    assert turn.system_extra == ""


def test_translate_to_english_short_form():
    s = JarvisSession()
    turn = s.route("Jarvis, translate to English")
    assert s.mode is not None
    assert s.mode.target_lang == "english"
    assert turn.reset_history is True


def test_unknown_language_is_not_a_mode_set():
    s = JarvisSession()
    turn = s.route("jarvis translate everything into klingon")
    # Not a known TTS/lang alias → treat as a normal Jarvis command (mode cleared).
    assert s.mode is None
    assert turn.reset_history is True
    assert "klingon" in turn.user_text.lower()


def test_default_utterance_without_mode():
    s = JarvisSession()
    turn = s.route("what time is it")
    assert turn.reset_history is False
    assert turn.system_extra == ""
    assert turn.user_text == "what time is it"
