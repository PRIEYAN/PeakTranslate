from reason.src.runtime.history import ConversationHistory
from reason.src.runtime.messages import Turn


def test_empty_history_snapshot():
    h = ConversationHistory(max_turns=4)
    assert h.snapshot() == ()


def test_add_exchange_appends_user_then_assistant():
    h = ConversationHistory(max_turns=4)
    h.add_exchange("hi", "hello")
    assert h.snapshot() == (Turn("user", "hi"), Turn("assistant", "hello"))


def test_bounded_to_max_turns():
    h = ConversationHistory(max_turns=1)
    h.add_exchange("a", "1")
    h.add_exchange("b", "2")
    assert h.snapshot() == (Turn("user", "b"), Turn("assistant", "2"))


def test_zero_max_turns_disables_history():
    h = ConversationHistory(max_turns=0)
    h.add_exchange("a", "1")
    assert h.snapshot() == ()


def test_clear():
    h = ConversationHistory(max_turns=4)
    h.add_exchange("a", "1")
    h.clear()
    assert h.snapshot() == ()
