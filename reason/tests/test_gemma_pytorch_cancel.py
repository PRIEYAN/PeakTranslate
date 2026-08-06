"""`_EventStoppingCriteria` — the piece that lets barge-in (docs/reasoningModel/
01-gemma-reasoning-mode.md §19) cancel `model.generate()` within one decode
step instead of waiting for max_new_tokens. Pure logic around a
threading.Event; needs torch for the base class's tensor args but no CUDA.
"""
from __future__ import annotations

import threading

import torch

from reason.src.runtime.gemma_pytorch import _EventStoppingCriteria


def test_returns_false_while_event_is_unset():
    criteria = _EventStoppingCriteria(threading.Event())
    assert criteria(torch.zeros(1, 1), torch.zeros(1, 1)) is False


def test_returns_true_once_event_is_set():
    event = threading.Event()
    criteria = _EventStoppingCriteria(event)
    assert criteria(torch.zeros(1, 1), torch.zeros(1, 1)) is False
    event.set()
    assert criteria(torch.zeros(1, 1), torch.zeros(1, 1)) is True


def test_tracks_the_live_state_of_the_event_not_a_snapshot():
    event = threading.Event()
    criteria = _EventStoppingCriteria(event)
    event.set()
    event.clear()
    assert criteria(torch.zeros(1, 1), torch.zeros(1, 1)) is False
