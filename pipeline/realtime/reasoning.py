"""Reasoning worker: Q1 transcript_queue -> Q2 translation_queue.

The `--mode reason` counterpart to `mt_worker` in workers.py. Exactly one of
the two runs (see pipeline/run_realtime.py). This module depends only on
the `ReasoningEngine` port injected by the caller — it imports no ML
framework at all, which is what makes it testable against a fake engine
with no GPU and no model download. See
docs/reasoningModel/01-gemma-reasoning-mode.md §3 and §11.

Follows the same two robustness rules as every worker in workers.py:
startup and per-item failures are logged with a full traceback instead of
silently killing the daemon thread, and queue reads use a short timeout and
check `stop_event` so shutdown completes even if an upstream worker died
before forwarding `STOP`.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

from .messages import STOP, Abort, Reply, Sentence
from .noise import pick_speech_lang, strip_unspeakable

if TYPE_CHECKING:  # type-only; never imported at runtime, no torch pulled in
    from reason.src.runtime import ConversationHistory
    from reason.src.runtime.interface import ReasoningEngine
    from reason.src.runtime.session import JarvisSession
    from reason.src.runtime.streaming import SentenceAssembler

log = logging.getLogger("peaktranslation.realtime")

_GET_TIMEOUT_S = 0.5


def reasoning_worker(
    *,
    engine: "ReasoningEngine",  # injected, never constructed here (DIP)
    history: "ConversationHistory",
    assembler_factory: Callable[[], "SentenceAssembler"],  # one per utterance
    system_prompt: str,
    transcript_queue: "queue.Queue",
    reply_queue: "queue.Queue",
    gpu_lock: threading.Lock,
    stop_event: threading.Event,
    out_lang: str = "en",
    speaking: Optional[threading.Event] = None,
    interrupt_event: Optional[threading.Event] = None,
    history_ttl_s: float = 30.0,
    session: Optional["JarvisSession"] = None,
) -> None:
    from reason.src.runtime import Prompt
    from reason.src.runtime.session import JarvisSession as _JarvisSession

    jarvis = session if session is not None else _JarvisSession()

    try:
        engine.warmup()
        log.info("Reasoning worker ready.")
    except Exception:
        log.exception("Reasoning worker failed to start — pipeline cannot continue.")
        stop_event.set()
        reply_queue.put(STOP)
        return

    # Wall-clock chat-history refresh only — sticky Jarvis mode is NOT cleared
    # here (only a new "jarvis …" command clears/replaces it).
    history_epoch = time.monotonic()

    def maybe_refresh_history() -> None:
        nonlocal history_epoch
        if history_ttl_s <= 0:
            return
        now = time.monotonic()
        if now - history_epoch < history_ttl_s:
            return
        if history.snapshot():
            history.clear()
            log.info(
                "Reasoning chat history refreshed (ttl=%.0fs); sticky Jarvis mode kept=%s.",
                history_ttl_s,
                jarvis.mode.target_lang if jarvis.mode else "none",
            )
        history_epoch = now

    while not stop_event.is_set():
        try:
            item = transcript_queue.get(timeout=_GET_TIMEOUT_S)
        except queue.Empty:
            maybe_refresh_history()
            continue
        if item is STOP:
            reply_queue.put(STOP)
            break
        sent: Sentence = item
        # Start every utterance with a clean slate: a barge-in mid-reply
        # sets this, and it must not leak into the reply that interrupted
        # it (doc §19).
        if interrupt_event is not None:
            interrupt_event.clear()

        maybe_refresh_history()

        turn = jarvis.route(sent.text)
        if turn.reset_history:
            history.clear()
            history_epoch = time.monotonic()
        if turn.log_note:
            log.info("[%s] Jarvis: %s", sent.utt_id, turn.log_note)

        system = system_prompt
        if turn.system_extra:
            system = f"{system_prompt}\n\n{turn.system_extra}".strip()

        t0 = time.perf_counter()
        assembler = assembler_factory()
        seq = 0
        full: list[str] = []
        aborted = False

        def emit(text: str, is_last: bool) -> None:
            nonlocal seq
            # Strip emoji Gemma sometimes sneaks in despite the prompt —
            # Piper can't speak them and they waste synthesis time.
            clean = strip_unspeakable(text)
            lang = pick_speech_lang(clean) if clean else out_lang
            reply_queue.put(
                Reply(
                    utt_id=sent.utt_id,
                    text=clean,
                    seq=seq,
                    is_last=is_last,
                    out_lang=lang,
                    t_captured=sent.t_captured,
                    t_reply_done=time.perf_counter(),
                )
            )
            if seq == 0:
                log.info(
                    "[%s] first reply chunk (%.0f ms): %r",
                    sent.utt_id,
                    (time.perf_counter() - t0) * 1000,
                    text,
                )
            seq += 1

        try:
            if speaking is not None:
                speaking.set()  # see doc §14 (mute) / §19 (barge-in signal)
            prompt = Prompt(
                user_text=turn.user_text,
                system=system,
                history=history.snapshot(),
            )
            # gpu_lock held for the whole generation: reasoning is the only
            # GPU consumer besides STT. Without barge-in, capture is muted
            # so this is safe regardless of duration (§14). With barge-in,
            # `cancel` below keeps this window short even when interrupted
            # (§19) — STT for the interrupting utterance only waits for
            # one decode step, not the full reply.
            with gpu_lock:
                for delta in engine.stream_reply(prompt, cancel=interrupt_event):
                    if interrupt_event is not None and interrupt_event.is_set():
                        # Stop consuming even if the engine doesn't honour
                        # `cancel` — the worker owns this decision either
                        # way (interface.py's contract).
                        aborted = True
                        break
                    full.append(delta)
                    for s in assembler.push(delta):
                        emit(s, is_last=False)

            if aborted:
                log.info(
                    "[%s] reply interrupted by barge-in after %d chunks (%.0f ms).",
                    sent.utt_id, seq, (time.perf_counter() - t0) * 1000,
                )
                # No flush, no is_last, no history update: the reply is
                # void, not merely truncated. `Abort` tells tts/playback to
                # drop anything already in flight for this utt_id — see
                # messages.py and doc §19.
                reply_queue.put(Abort(utt_id=sent.utt_id))
                if speaking is not None:
                    speaking.clear()
                if interrupt_event is not None:
                    # Consume it now rather than leaving it set until the next
                    # utterance's top-of-loop clear — that gap could otherwise
                    # be seconds while waiting on transcript_queue, during
                    # which a stale "set" could kill unrelated audio that's
                    # mid-play for a *different* reason entirely.
                    interrupt_event.clear()
                continue

            for s in assembler.flush():
                emit(s, is_last=False)
            emit("", is_last=True)  # end-of-utterance marker; TTS skips empties
            reply = "".join(full).strip()
            history.add_exchange(turn.user_text, reply)
            log.info(
                "[%s] reply (%.0f ms, %d chunks): %r",
                sent.utt_id,
                (time.perf_counter() - t0) * 1000,
                seq,
                reply,
            )
        except Exception:
            # `speaking` is normally cleared by playback once audio finishes,
            # but on failure no audio will ever play — clear it here or the
            # mic stays muted forever.
            log.exception("[%s] reasoning failed on this sentence; skipping.", sent.utt_id)
            if speaking is not None:
                speaking.clear()
            continue

    log.info("Reasoning worker stopped.")
