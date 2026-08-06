"""Reason mode's feedback-loop guard (docs/reasoningModel/01-gemma-reasoning-
mode.md §14): frames must be dropped before they ever reach the VAD
endpointer while `speaking` is set, and flow through normally once it's
clear. Uses the energy VAD backend so this needs neither webrtcvad tuning
nor real audio hardware.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from pipeline.realtime.capture import AudioCapture

_VAD_KWARGS = {
    "backend": "energy",
    "silence_ms": 40,
    "min_utterance_ms": 20,
    "max_utterance_ms": 2000,
    "pre_roll_ms": 0,
}


def _tone_frame(n_bytes: int, amplitude: int = 20000) -> bytes:
    n_samples = n_bytes // 2
    t = np.arange(n_samples)
    wave = (amplitude * np.sin(2 * np.pi * 440 * t / 16000)).astype("<i2")
    return wave.tobytes()


def _silence_frame(n_bytes: int) -> bytes:
    return b"\x00" * n_bytes


def _run_vad_loop(cap: AudioCapture, frames: list[bytes]) -> None:
    t = threading.Thread(target=cap._vad_loop, daemon=True)
    t.start()
    for f in frames:
        cap._raw_frames.put(f)
    time.sleep(0.3)
    cap._stop.set()
    t.join(timeout=1.0)


def test_frames_are_dropped_while_speaking_is_set():
    audio_queue: "queue.Queue" = queue.Queue(maxsize=8)
    speaking = threading.Event()
    speaking.set()
    cap = AudioCapture(
        audio_queue=audio_queue, sample_rate=16000, frame_ms=20,
        vad_kwargs=_VAD_KWARGS, speaking=speaking,
    )
    frames = [_tone_frame(cap.frame_bytes) for _ in range(30)] + [
        _silence_frame(cap.frame_bytes) for _ in range(5)
    ]
    _run_vad_loop(cap, frames)
    assert audio_queue.empty()


def test_frames_flow_normally_once_speaking_is_clear():
    audio_queue: "queue.Queue" = queue.Queue(maxsize=8)
    cap = AudioCapture(
        audio_queue=audio_queue, sample_rate=16000, frame_ms=20,
        vad_kwargs=_VAD_KWARGS, speaking=None,
    )
    frames = [_tone_frame(cap.frame_bytes) for _ in range(30)] + [
        _silence_frame(cap.frame_bytes) for _ in range(5)
    ]
    _run_vad_loop(cap, frames)
    assert not audio_queue.empty()


def test_speaking_toggled_mid_stream_only_drops_while_set():
    audio_queue: "queue.Queue" = queue.Queue(maxsize=8)
    speaking = threading.Event()
    speaking.set()  # muted for the first batch
    cap = AudioCapture(
        audio_queue=audio_queue, sample_rate=16000, frame_ms=20,
        vad_kwargs=_VAD_KWARGS, speaking=speaking,
    )
    t = threading.Thread(target=cap._vad_loop, daemon=True)
    t.start()
    for _ in range(30):
        cap._raw_frames.put(_tone_frame(cap.frame_bytes))
    time.sleep(0.2)
    assert audio_queue.empty()  # nothing got through while muted

    speaking.clear()  # playback finished
    for _ in range(30):
        cap._raw_frames.put(_tone_frame(cap.frame_bytes))
    for _ in range(5):
        cap._raw_frames.put(_silence_frame(cap.frame_bytes))
    time.sleep(0.3)
    cap._stop.set()
    t.join(timeout=1.0)
    assert not audio_queue.empty()


def test_on_speech_start_fires_once_on_the_first_speech_frame():
    # Barge-in (doc §19) needs this to fire the instant speech begins, not
    # once the full utterance is endpointed ~silence_ms later.
    audio_queue: "queue.Queue" = queue.Queue(maxsize=8)
    calls = []
    cap = AudioCapture(
        audio_queue=audio_queue, sample_rate=16000, frame_ms=20,
        vad_kwargs=_VAD_KWARGS, speaking=None,
        on_speech_start=lambda: calls.append(1),
    )
    frames = [_tone_frame(cap.frame_bytes) for _ in range(30)] + [
        _silence_frame(cap.frame_bytes) for _ in range(5)
    ]
    _run_vad_loop(cap, frames)
    assert len(calls) == 1


def test_on_speech_start_does_not_fire_during_silence():
    audio_queue: "queue.Queue" = queue.Queue(maxsize=8)
    calls = []
    cap = AudioCapture(
        audio_queue=audio_queue, sample_rate=16000, frame_ms=20,
        vad_kwargs=_VAD_KWARGS, speaking=None,
        on_speech_start=lambda: calls.append(1),
    )
    frames = [_silence_frame(cap.frame_bytes) for _ in range(30)]
    _run_vad_loop(cap, frames)
    assert calls == []


def test_on_speech_start_does_not_fire_while_frames_are_dropped_for_muting():
    # `speaking`-based dropping happens before frames ever reach the VAD
    # endpointer, so on_speech_start can't fire either. This is why
    # run_realtime.py treats the two as mutually exclusive (doc §19): to
    # get barge-in detection, `speaking` must be None so frames reach VAD.
    audio_queue: "queue.Queue" = queue.Queue(maxsize=8)
    speaking = threading.Event()
    speaking.set()
    calls = []
    cap = AudioCapture(
        audio_queue=audio_queue, sample_rate=16000, frame_ms=20,
        vad_kwargs=_VAD_KWARGS, speaking=speaking,
        on_speech_start=lambda: calls.append(1),
    )
    frames = [_tone_frame(cap.frame_bytes) for _ in range(30)]
    _run_vad_loop(cap, frames)
    assert calls == []
    assert audio_queue.empty()
