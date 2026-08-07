"""Jarvis wake-word routing + sticky translate mode."""
from reason.src.runtime.session import JarvisSession


def test_mode_set_reply_is_translation_of_command_phrase():
    s = JarvisSession()
    turn = s.route("jarvis translate everything to tamil")
    assert turn.reset_history is True
    assert s.mode is not None
    assert s.mode.target_lang == "tamil"
    assert turn.user_text == "translate everything to tamil"
    assert "MODE: mode_set" in turn.system_extra
    assert "TARGET_LANG: tamil" in turn.system_extra


def test_wake_word_must_be_at_sentence_start():
    s = JarvisSession()
    turn = s.route("hey jarvis always speak in hindi")
    assert turn.reset_history is False
    assert s.mode is None
    assert turn.user_text == "hey jarvis always speak in hindi"
    assert turn.log_note == "default assistant"

    turn2 = s.route("please jarvis translate to tamil")
    assert s.mode is None
    assert turn2.log_note == "default assistant"


def test_telugu_everything_that_i_say_into_the_telugu():
    s = JarvisSession()
    turn = s.route("Jarvis translate everything that i say into the telugu")
    assert turn.reset_history is True
    assert s.mode is not None
    assert s.mode.target_lang == "telugu"
    assert turn.log_note == "jarvis mode set: translate→telugu"
    assert turn.user_text == "translate everything that i say into the telugu"
    assert "MODE: mode_set" in turn.system_extra
    assert "TARGET_LANG: telugu" in turn.system_extra


def test_sticky_mode_applies_without_wake_word():
    s = JarvisSession()
    s.route("jarvis translate everything to tamil")
    turn = s.route("good morning friends")
    assert turn.reset_history is False
    assert turn.user_text == "good morning friends"
    assert "MODE: sticky_translate" in turn.system_extra
    assert "TARGET_LANG: tamil" in turn.system_extra
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
    assert turn.user_text == "translate to English"


def test_arbitrary_language_sets_mode():
    s = JarvisSession()
    turn = s.route("jarvis translate everything into spanish")
    assert s.mode is not None
    assert s.mode.target_lang == "spanish"
    assert turn.reset_history is True
    assert "TARGET_LANG: spanish" in turn.system_extra

    turn2 = s.route("jarvis translate everything into japanese")
    assert s.mode.target_lang == "japanese"
    assert "TARGET_LANG: japanese" in turn2.system_extra


def test_default_utterance_without_mode():
    s = JarvisSession()
    turn = s.route("what time is it")
    assert turn.reset_history is False
    assert turn.system_extra == ""
    assert turn.user_text == "what time is it"
