"""Queue payload types shared by all real-time pipeline workers.

Every message carries `utt_id` so one utterance can be traced end-to-end
(capture -> STT -> MT -> TTS -> playback) in logs and latency reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Sentinel pushed through every queue, in pipeline order, to trigger a clean
# shutdown of the worker that owns that queue.
STOP = None


@dataclass
class Utterance:
    utt_id: str
    pcm: bytes  # int16 mono PCM, includes pre-roll
    sample_rate: int
    duration_s: float
    t_captured: float  # time.perf_counter() at VAD close


@dataclass
class Sentence:
    utt_id: str
    text: str
    src_lang: str
    t_captured: float
    t_stt_done: float


@dataclass
class Translation:
    utt_id: str
    text: str
    tgt_lang: str
    t_captured: float
    t_mt_done: float


@dataclass
class Reply:
    """Reason mode's Q2 message. Not reused as `Translation`: streaming a
    reply (docs/reasoningModel/01-gemma-reasoning-mode.md §7) emits several
    of these per utterance, which the translate path never does — hence
    `seq`/`is_last`, which `Translation` deliberately does not have.
    """

    utt_id: str
    text: str
    seq: int  # 0-based chunk index within this utterance
    is_last: bool
    out_lang: str
    t_captured: float
    t_reply_done: float


@dataclass
class Abort:
    """Barge-in signal: 'stop whatever you're doing for this utterance.'

    Pushed into Q2 (and forwarded into Q3) the moment reasoning_worker
    detects the user talking again mid-reply. Ordering through the same
    FIFO queue means any chunks queued *before* the Abort still play (they
    were already committed); nothing queued *after* it exists, since
    reasoning_worker stops producing the instant it aborts. Killing audio
    that's already mid-playback needs a separate signal (`interrupt_event`,
    checked directly by playback_worker) because a queued message can't be
    seen while a worker is blocked inside a single item's processing. See
    docs/reasoningModel/01-gemma-reasoning-mode.md §19.
    """

    utt_id: str


@dataclass
class WavJob:
    utt_id: str
    wav_path: Path
    sample_rate: int
    t_captured: float
    t_tts_done: float
    # True for every Translation-derived job (one chunk = the whole
    # utterance). Reason mode's Reply carries the real value: playback only
    # clears the `speaking` mute once the LAST chunk has finished, since only
    # playback knows when sound has actually stopped (doc §14).
    is_last: bool = True
