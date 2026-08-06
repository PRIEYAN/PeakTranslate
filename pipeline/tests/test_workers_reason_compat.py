"""Changes to workers.py needed only because reason mode streams several Q2
items per utterance (docs/reasoningModel/01-gemma-reasoning-mode.md §12):
seq-suffixed WAV filenames, skipping empty chunks, and playback clearing
`speaking` only after the last chunk. Translate-mode inputs (`Translation`,
default `seq=0`/`is_last=True`) must be unaffected by all three.
"""
from __future__ import annotations

import queue
import threading
import time

import pipeline.realtime.workers as workers
from pipeline.realtime.messages import STOP, Abort, Reply, Translation, WavJob


def _drain(q: "queue.Queue") -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


def test_tts_worker_skips_empty_chunks(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(workers, "_piper_synth", lambda text, voice, out: calls.append(text))

    translation_queue: "queue.Queue" = queue.Queue()
    wav_queue: "queue.Queue" = queue.Queue()
    translation_queue.put(
        Reply(utt_id="u1", text="", seq=2, is_last=True, out_lang="en", t_captured=0.0, t_reply_done=0.0)
    )
    translation_queue.put(STOP)

    workers.tts_worker(
        voice_onnx=tmp_path / "voice.onnx",
        spill_dir=tmp_path,
        translation_queue=translation_queue,
        wav_queue=wav_queue,
        stop_event=threading.Event(),
    )

    assert calls == []
    jobs = _drain(wav_queue)
    assert jobs == [STOP]


def test_tts_worker_wav_filename_includes_seq_for_reply_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "_piper_synth", lambda text, voice, out: out.touch())

    translation_queue: "queue.Queue" = queue.Queue()
    wav_queue: "queue.Queue" = queue.Queue()
    translation_queue.put(
        Reply(utt_id="u1", text="Hello.", seq=1, is_last=False, out_lang="en", t_captured=0.0, t_reply_done=0.0)
    )
    translation_queue.put(STOP)

    workers.tts_worker(
        voice_onnx=tmp_path / "voice.onnx",
        spill_dir=tmp_path,
        translation_queue=translation_queue,
        wav_queue=wav_queue,
        stop_event=threading.Event(),
    )

    jobs = [j for j in _drain(wav_queue) if j is not STOP]
    assert len(jobs) == 1
    assert jobs[0].wav_path.name == "u1-1.wav"
    assert jobs[0].is_last is False


def test_tts_worker_translation_filename_unaffected(monkeypatch, tmp_path):
    monkeypatch.setattr(workers, "_piper_synth", lambda text, voice, out: out.touch())

    translation_queue: "queue.Queue" = queue.Queue()
    wav_queue: "queue.Queue" = queue.Queue()
    translation_queue.put(Translation(utt_id="u2", text="Namaste.", tgt_lang="hi", t_captured=0.0, t_mt_done=0.0))
    translation_queue.put(STOP)

    workers.tts_worker(
        voice_onnx=tmp_path / "voice.onnx",
        spill_dir=tmp_path,
        translation_queue=translation_queue,
        wav_queue=wav_queue,
        stop_event=threading.Event(),
    )

    jobs = [j for j in _drain(wav_queue) if j is not STOP]
    assert len(jobs) == 1
    assert jobs[0].wav_path.name == "u2-0.wav"
    assert jobs[0].is_last is True


def test_playback_clears_speaking_only_after_the_last_chunk(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.realtime.audio_out.play_wav_blocking", lambda path, interrupt=None: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # skip the real 0.2s delay

    speaking = threading.Event()
    speaking.set()
    wav_queue: "queue.Queue" = queue.Queue()
    for seq, is_last in [(0, False), (1, False), (2, True)]:
        p = tmp_path / f"u1-{seq}.wav"
        p.touch()
        wav_queue.put(WavJob(utt_id="u1", wav_path=p, sample_rate=22050, t_captured=0.0, t_tts_done=0.0, is_last=is_last))
    wav_queue.put(STOP)

    workers.playback_worker(
        wav_queue=wav_queue, stop_event=threading.Event(), keep_wavs=False, speaking=speaking,
    )

    assert not speaking.is_set()


def test_playback_clears_speaking_even_if_play_raises(monkeypatch, tmp_path):
    def boom(path, interrupt=None):
        raise RuntimeError("no audio device")

    monkeypatch.setattr("pipeline.realtime.audio_out.play_wav_blocking", boom)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    speaking = threading.Event()
    speaking.set()
    wav_queue: "queue.Queue" = queue.Queue()
    p = tmp_path / "u1-0.wav"
    p.touch()
    wav_queue.put(WavJob(utt_id="u1", wav_path=p, sample_rate=22050, t_captured=0.0, t_tts_done=0.0, is_last=True))
    wav_queue.put(STOP)

    workers.playback_worker(wav_queue=wav_queue, stop_event=threading.Event(), speaking=speaking)

    assert not speaking.is_set()


def test_playback_translate_mode_unaffected_by_speaking_param(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.realtime.audio_out.play_wav_blocking", lambda path, interrupt=None: None)

    wav_queue: "queue.Queue" = queue.Queue()
    p = tmp_path / "u1-0.wav"
    p.touch()
    wav_queue.put(WavJob(utt_id="u1", wav_path=p, sample_rate=22050, t_captured=0.0, t_tts_done=0.0))
    wav_queue.put(STOP)

    # speaking=None (the default): must not raise, must not touch anything.
    workers.playback_worker(wav_queue=wav_queue, stop_event=threading.Event())


def test_tts_worker_forwards_abort_without_synthesizing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(workers, "_piper_synth", lambda text, voice, out: calls.append(text))

    translation_queue: "queue.Queue" = queue.Queue()
    wav_queue: "queue.Queue" = queue.Queue()
    translation_queue.put(Abort(utt_id="u1"))
    translation_queue.put(STOP)

    workers.tts_worker(
        voice_onnx=tmp_path / "voice.onnx",
        spill_dir=tmp_path,
        translation_queue=translation_queue,
        wav_queue=wav_queue,
        stop_event=threading.Event(),
    )

    assert calls == []
    jobs = _drain(wav_queue)
    assert jobs[0] == Abort(utt_id="u1")
    assert jobs[1] is STOP


def test_tts_worker_has_no_interrupt_event_gate(monkeypatch, tmp_path):
    # Regression test for the lockout bug: tts_worker must NOT skip a normal
    # chunk just because some *unrelated* speech onset happened to set
    # interrupt_event in the meantime — Abort (FIFO-ordered, per-utterance)
    # is the only authoritative "this is stale" signal. A blanket
    # interrupt_event check here previously meant that once the mic picked
    # up any bleed during a reply, `speaking` never got cleared (playback
    # never saw its last chunk), which re-armed the flag on every future
    # utterance forever — TTS stopped synthesizing anything, permanently.
    # See docs/reasoningModel/01-gemma-reasoning-mode.md §19.
    calls = []
    monkeypatch.setattr(workers, "_piper_synth", lambda text, voice, out: calls.append(text))

    translation_queue: "queue.Queue" = queue.Queue()
    wav_queue: "queue.Queue" = queue.Queue()
    translation_queue.put(
        Reply(utt_id="u1", text="Hello.", seq=0, is_last=False, out_lang="en", t_captured=0.0, t_reply_done=0.0)
    )
    translation_queue.put(STOP)

    workers.tts_worker(
        voice_onnx=tmp_path / "voice.onnx",
        spill_dir=tmp_path,
        translation_queue=translation_queue,
        wav_queue=wav_queue,
        stop_event=threading.Event(),
    )

    assert calls == ["Hello."]
    jobs = [j for j in _drain(wav_queue) if j is not STOP]
    assert len(jobs) == 1


def test_playback_skips_only_the_abort_marker_not_real_chunks(monkeypatch, tmp_path):
    played = []
    monkeypatch.setattr(
        "pipeline.realtime.audio_out.play_wav_blocking",
        lambda path, interrupt=None: played.append(path),
    )

    # Same regression as above, for playback: a chunk that legitimately
    # follows an Abort for a *different* utterance (or one queued while
    # interrupt_event happened to be set for an unrelated reason) must
    # still play. Only the Abort marker itself is skipped.
    interrupt_event = threading.Event()
    interrupt_event.set()
    wav_queue: "queue.Queue" = queue.Queue()
    wav_queue.put(Abort(utt_id="u1"))
    p = tmp_path / "u2-0.wav"
    p.touch()
    wav_queue.put(WavJob(utt_id="u2", wav_path=p, sample_rate=22050, t_captured=0.0, t_tts_done=0.0, is_last=True))
    wav_queue.put(STOP)

    workers.playback_worker(
        wav_queue=wav_queue, stop_event=threading.Event(), interrupt_event=interrupt_event,
    )

    assert played == [p]


def test_speaking_still_clears_even_with_a_persistently_set_interrupt_event(monkeypatch, tmp_path):
    # The exact lockout scenario observed with real mic bleed: interrupt_event
    # stays set continuously (constant false-positive speech onsets), but a
    # normal reply's chunks must still reach playback and its last chunk must
    # still clear `speaking` — otherwise every future onset re-arms the
    # barge-in path and TTS silently stops firing forever.
    monkeypatch.setattr(workers, "_piper_synth", lambda text, voice, out: out.touch())
    monkeypatch.setattr("pipeline.realtime.audio_out.play_wav_blocking", lambda path, interrupt=None: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    interrupt_event = threading.Event()
    interrupt_event.set()  # never clears for the duration of this test
    speaking = threading.Event()
    speaking.set()

    translation_queue: "queue.Queue" = queue.Queue()
    wav_queue: "queue.Queue" = queue.Queue()
    for seq, is_last in [(0, False), (1, True)]:
        translation_queue.put(
            Reply(utt_id="u1", text=f"chunk {seq}.", seq=seq, is_last=is_last, out_lang="en", t_captured=0.0, t_reply_done=0.0)
        )
    translation_queue.put(STOP)

    workers.tts_worker(
        voice_onnx=tmp_path / "voice.onnx",
        spill_dir=tmp_path,
        translation_queue=translation_queue,
        wav_queue=wav_queue,
        stop_event=threading.Event(),
    )
    jobs = [j for j in _drain(wav_queue) if j is not STOP]
    assert len(jobs) == 2  # both chunks reached Q3 despite interrupt_event being set throughout

    wav_queue2: "queue.Queue" = queue.Queue()
    for j in jobs:
        wav_queue2.put(j)
    wav_queue2.put(STOP)
    workers.playback_worker(
        wav_queue=wav_queue2, stop_event=threading.Event(), speaking=speaking, interrupt_event=interrupt_event,
    )

    assert not speaking.is_set()


def test_playback_passes_interrupt_event_into_play_wav_blocking(monkeypatch, tmp_path):
    seen = {}

    def fake_play(path, interrupt=None):
        seen["interrupt"] = interrupt

    monkeypatch.setattr("pipeline.realtime.audio_out.play_wav_blocking", fake_play)

    interrupt_event = threading.Event()
    wav_queue: "queue.Queue" = queue.Queue()
    p = tmp_path / "u1-0.wav"
    p.touch()
    wav_queue.put(WavJob(utt_id="u1", wav_path=p, sample_rate=22050, t_captured=0.0, t_tts_done=0.0))
    wav_queue.put(STOP)

    workers.playback_worker(
        wav_queue=wav_queue, stop_event=threading.Event(), interrupt_event=interrupt_event,
    )

    assert seen["interrupt"] is interrupt_event
