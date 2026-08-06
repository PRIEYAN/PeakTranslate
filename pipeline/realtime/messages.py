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
class WavJob:
    utt_id: str
    wav_path: Path
    sample_rate: int
    t_captured: float
    t_tts_done: float
