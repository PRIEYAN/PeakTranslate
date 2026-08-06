"""Interruptible playback for barge-in (docs/reasoningModel/01-gemma-
reasoning-mode.md §19): `play_wav_blocking` must return within roughly one
`_POLL_S` tick of `interrupt` being set, not wait for the player to exit on
its own. Uses `sleep N` in place of a real audio player (swapped in via
`_PLAYERS`) so this needs no audio hardware and no real WAV file — only
that `play_wav_blocking` builds `[*base_cmd, str(path)]` and runs it.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pipeline.realtime.audio_out as audio_out


def test_interrupt_kills_playback_well_before_natural_completion(monkeypatch):
    monkeypatch.setattr(audio_out, "_PLAYERS", (["sleep"],))
    interrupt = threading.Event()

    def trigger_soon():
        time.sleep(0.15)
        interrupt.set()

    threading.Thread(target=trigger_soon, daemon=True).start()

    t0 = time.perf_counter()
    audio_out.play_wav_blocking(Path("5"), interrupt=interrupt)  # `sleep 5`
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0  # nowhere near the 5s the process would take uninterrupted


def test_playback_runs_to_completion_when_never_interrupted(monkeypatch):
    monkeypatch.setattr(audio_out, "_PLAYERS", (["sleep"],))

    t0 = time.perf_counter()
    audio_out.play_wav_blocking(Path("0.2"), interrupt=threading.Event())
    elapsed = time.perf_counter() - t0

    assert elapsed >= 0.2


def test_no_interrupt_argument_behaves_as_before(monkeypatch):
    monkeypatch.setattr(audio_out, "_PLAYERS", (["sleep"],))
    audio_out.play_wav_blocking(Path("0.05"))  # must not raise with interrupt=None default


def test_falls_through_to_next_player_when_first_is_missing(monkeypatch):
    monkeypatch.setattr(
        audio_out, "_PLAYERS", (["definitely-not-a-real-binary"], ["sleep"])
    )
    audio_out.play_wav_blocking(Path("0.01"))  # must not raise; second player is used


def test_warns_and_returns_when_no_player_is_found(monkeypatch, caplog):
    monkeypatch.setattr(audio_out, "_PLAYERS", (["definitely-not-a-real-binary"],))
    audio_out.play_wav_blocking(Path("/tmp/whatever.wav"))
    assert "No audio player found" in caplog.text
