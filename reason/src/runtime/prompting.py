"""Prompt -> model-specific prompt string, via the tokenizer's chat template.

The only place that knows about chat templates. Failing loudly on a missing
one turns "the assistant rambles" from a puzzling behaviour into a startup
error naming the cause (docs/reasoningModel/01-gemma-reasoning-mode.md §6).
"""
from __future__ import annotations

from .messages import Prompt


class ChatPromptFormatter:
    def __init__(self, tokenizer) -> None:
        if tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer has no chat template — this is a base model, not "
                "instruction-tuned. Use an -it variant (see "
                "docs/reasoningModel/01-gemma-reasoning-mode.md §6)."
            )
        self._tok = tokenizer

    def render(self, prompt: Prompt) -> str:
        msgs = [{"role": t.role, "content": t.content} for t in prompt.history]
        # Gemma has no system role; fold the instruction into the first user turn.
        head = f"{prompt.system}\n\n{prompt.user_text}" if prompt.system else prompt.user_text
        msgs.append({"role": "user", "content": head})
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
