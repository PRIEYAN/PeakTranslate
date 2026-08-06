"""No torch/transformers here: a plain object stands in for the tokenizer."""
import pytest

from reason.src.runtime.messages import Prompt, Turn
from reason.src.runtime.prompting import ChatPromptFormatter


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return " | ".join(f"{m['role']}:{m['content']}" for m in msgs)


class NoTemplateTokenizer:
    chat_template = None


def test_raises_on_missing_chat_template():
    with pytest.raises(ValueError, match="no chat template"):
        ChatPromptFormatter(NoTemplateTokenizer())


def test_render_folds_system_into_first_user_turn():
    fmt = ChatPromptFormatter(FakeTokenizer())
    out = fmt.render(Prompt(user_text="hi", system="Be brief."))
    assert out == "user:Be brief.\n\nhi"


def test_render_includes_history_in_order():
    fmt = ChatPromptFormatter(FakeTokenizer())
    history = (Turn("user", "earlier"), Turn("assistant", "reply"))
    out = fmt.render(Prompt(user_text="now", history=history))
    assert out == "user:earlier | assistant:reply | user:now"


def test_render_without_system_has_no_extra_prefix():
    fmt = ChatPromptFormatter(FakeTokenizer())
    out = fmt.render(Prompt(user_text="hi"))
    assert out == "user:hi"
