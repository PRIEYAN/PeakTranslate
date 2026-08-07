"""reasoning_worker against a FakeEngine — no GPU, no model download.

This is the payoff of keeping pipeline/realtime/reasoning.py free of torch
and transformers imports (docs/reasoningModel/01-gemma-reasoning-mode.md §3,
§11): the pipeline's trickiest logic (backpressure, the `speaking` mute,
shutdown sentinels) is fully testable here.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

from pipeline.realtime.messages import STOP, Abort, Sentence
from pipeline.realtime.reasoning import reasoning_worker
from reason.src.runtime.history import ConversationHistory
from reason.src.runtime.streaming import SentenceAssembler


class FakeEngine:
    def __init__(self, deltas):
        self._deltas = deltas
        self.warmed_up = False

    def warmup(self):
        self.warmed_up = True

    def stream_reply(self, prompt, *, cancel=None):
        yield from self._deltas


class InterruptingEngine:
    """Sets `cancel` itself partway through — simulates a barge-in landing
    mid-generation, the way GemmaPytorchReasoner's own StoppingCriteria
    would once run_realtime.py's on_speech_start callback fires.
    """

    def __init__(self, deltas, *, set_cancel_after: int):
        self._deltas = deltas
        self._set_cancel_after = set_cancel_after

    def warmup(self):
        pass

    def stream_reply(self, prompt, *, cancel=None):
        for i, delta in enumerate(self._deltas):
            if i == self._set_cancel_after and cancel is not None:
                cancel.set()
            yield delta


def _run_worker(
    engine,
    *,
    speaking=None,
    history=None,
    interrupt_event=None,
    sentence=None,
    history_ttl_s=0.0,
):
    transcript_queue: "queue.Queue" = queue.Queue()
    reply_queue: "queue.Queue" = queue.Queue()
    history = history or ConversationHistory(max_turns=4)

    transcript_queue.put(
        sentence or Sentence(utt_id="u1", text="hello", src_lang="en", t_captured=0.0, t_stt_done=0.0)
    )
    transcript_queue.put(STOP)

    reasoning_worker(
        engine=engine,
        history=history,
        assembler_factory=SentenceAssembler,
        system_prompt="Be brief.",
        transcript_queue=transcript_queue,
        reply_queue=reply_queue,
        gpu_lock=threading.Lock(),
        stop_event=threading.Event(),
        speaking=speaking,
        interrupt_event=interrupt_event,
        history_ttl_s=history_ttl_s,
    )

    out = []
    while True:
        item = reply_queue.get_nowait()
        if item is STOP:
            break
        out.append(item)
    return out, history


def test_engine_warms_up_before_serving():
    engine = FakeEngine(["Hello there. "])
    _run_worker(engine)
    assert engine.warmed_up


def test_chunk_ordering_and_is_last_marker():
    engine = FakeEngine(["Hello there. ", "How can", " I help", "?"])
    out, _ = _run_worker(engine)
    assert [r.text for r in out] == ["Hello there.", "How can I help?", ""]
    assert [r.is_last for r in out] == [False, False, True]
    assert [r.seq for r in out] == [0, 1, 2]
    assert all(r.utt_id == "u1" for r in out)


def test_history_updated_with_the_full_joined_reply():
    engine = FakeEngine(["Hello there. ", "How can", " I help", "?"])
    _, history = _run_worker(engine, history=ConversationHistory(max_turns=4))
    snap = history.snapshot()
    assert (snap[0].role, snap[0].content) == ("user", "hello")
    assert (snap[1].role, snap[1].content) == ("assistant", "Hello there. How can I help?")


def test_history_ttl_clears_stale_memory_before_reply(monkeypatch):
    """Stale turns from noise/hallucination must not poison the next prompt."""
    history = ConversationHistory(max_turns=4)
    history.add_exchange("you", "नमस्ते")
    assert history.snapshot()

    clock = {"t": 0.0}
    monkeypatch.setattr(
        "pipeline.realtime.reasoning.time.monotonic", lambda: clock["t"]
    )

    seen_history = {}

    class RecordingEngine:
        def warmup(self):
            pass

        def stream_reply(self, prompt, *, cancel=None):
            seen_history["turns"] = prompt.history
            yield "Fresh."

    transcript_queue: "queue.Queue" = queue.Queue()
    reply_queue: "queue.Queue" = queue.Queue()
    transcript_queue.put(
        Sentence(utt_id="u1", text="explain blockchain", src_lang="en", t_captured=0.0, t_stt_done=0.0)
    )
    transcript_queue.put(STOP)

    real_get = transcript_queue.get

    def get_and_age(*args, **kwargs):
        item = real_get(*args, **kwargs)
        clock["t"] = 31.0  # past history_ttl_s=30 before maybe_refresh_history
        return item

    monkeypatch.setattr(transcript_queue, "get", get_and_age)

    reasoning_worker(
        engine=RecordingEngine(),
        history=history,
        assembler_factory=SentenceAssembler,
        system_prompt="Be brief.",
        transcript_queue=transcript_queue,
        reply_queue=reply_queue,
        gpu_lock=threading.Lock(),
        stop_event=threading.Event(),
        history_ttl_s=30.0,
    )

    assert seen_history["turns"] == ()
    snap = history.snapshot()
    assert [t.content for t in snap] == ["explain blockchain", "Fresh."]


def test_speaking_set_during_reply_and_not_cleared_by_the_worker():
    # Only playback clears `speaking` (doc §14) — the worker just sets it.
    engine = FakeEngine(["Hi."])
    speaking = threading.Event()
    _run_worker(engine, speaking=speaking)
    assert speaking.is_set()


def test_speaking_cleared_on_failure_so_mic_does_not_stay_muted():
    class FailingEngine:
        def warmup(self):
            pass

        def stream_reply(self, prompt, *, cancel=None):
            raise RuntimeError("boom")
            yield  # pragma: no cover - never reached, keeps this a generator

    speaking = threading.Event()
    _run_worker(FailingEngine(), speaking=speaking)
    assert not speaking.is_set()


def test_empty_reply_yields_only_the_end_of_utterance_marker():
    out, _ = _run_worker(FakeEngine([]))
    assert [r.text for r in out] == [""]
    assert out[0].is_last is True


def test_startup_failure_stops_the_pipeline_without_serving():
    class BrokenEngine:
        def warmup(self):
            raise RuntimeError("no GPU")

        def stream_reply(self, prompt, *, cancel=None):
            raise AssertionError("must not be called")

    reply_queue: "queue.Queue" = queue.Queue()
    stop_event = threading.Event()
    reasoning_worker(
        engine=BrokenEngine(),
        history=ConversationHistory(max_turns=4),
        assembler_factory=SentenceAssembler,
        system_prompt="",
        transcript_queue=queue.Queue(),
        reply_queue=reply_queue,
        gpu_lock=threading.Lock(),
        stop_event=stop_event,
    )
    assert stop_event.is_set()
    assert reply_queue.get_nowait() is STOP


def test_barge_in_stops_emitting_and_sends_abort():
    interrupt_event = threading.Event()
    # cancel fires before the very first delta is even processed.
    engine = InterruptingEngine(["One. ", "Two. ", "Three. ", "Four."], set_cancel_after=0)
    out, _ = _run_worker(engine, interrupt_event=interrupt_event)
    assert [type(r) for r in out] == [Abort]
    assert out[0].utt_id == "u1"


def test_barge_in_partial_sentences_before_cancel_still_reach_q2():
    interrupt_event = threading.Event()
    # Cancel fires after the 3rd delta — by then "One. Two." has already
    # completed a sentence and should have been emitted before the abort.
    engine = InterruptingEngine(["One. ", "Two. ", "Three, ", "not done."], set_cancel_after=3)
    out, _ = _run_worker(engine, interrupt_event=interrupt_event)
    replies = [r for r in out if not isinstance(r, Abort)]
    aborts = [r for r in out if isinstance(r, Abort)]
    assert [r.text for r in replies] == ["One.", "Two."]
    assert len(aborts) == 1


def test_barge_in_does_not_update_history():
    interrupt_event = threading.Event()
    engine = InterruptingEngine(["One. ", "Two. ", "Three."], set_cancel_after=1)
    _, history = _run_worker(engine, interrupt_event=interrupt_event, history=ConversationHistory(max_turns=4))
    assert history.snapshot() == ()


def test_barge_in_clears_speaking_immediately():
    interrupt_event = threading.Event()
    speaking = threading.Event()
    engine = InterruptingEngine(["One. ", "Two."], set_cancel_after=0)
    _run_worker(engine, interrupt_event=interrupt_event, speaking=speaking)
    assert not speaking.is_set()


def test_interrupt_event_is_cleared_at_the_start_of_each_utterance():
    # A leftover set from a previous (already-handled) barge-in must not
    # immediately abort the *next* utterance's reply.
    interrupt_event = threading.Event()
    interrupt_event.set()
    engine = FakeEngine(["Hello there."])
    out, _ = _run_worker(engine, interrupt_event=interrupt_event)
    assert [r.text for r in out] == ["Hello there.", ""]


def test_engine_receives_the_interrupt_event_as_cancel():
    seen = {}

    class RecordingEngine:
        def warmup(self):
            pass

        def stream_reply(self, prompt, *, cancel=None):
            seen["cancel"] = cancel
            yield "hi"

    interrupt_event = threading.Event()
    _run_worker(RecordingEngine(), interrupt_event=interrupt_event)
    assert seen["cancel"] is interrupt_event


def test_worker_module_has_no_ml_deps():
    src = Path("pipeline/realtime/reasoning.py").read_text()
    assert "import torch" not in src
    assert "transformers" not in src
