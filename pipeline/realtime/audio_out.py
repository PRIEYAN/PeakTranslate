"""Blocking WAV playback for the playback worker.

Hard rule (docs/05-vad-realtime-integration.md §6.5): exactly one active
playback at a time. This module only ever plays one file synchronously;
serialization is guaranteed by the playback worker owning a single thread.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger("peaktranslation.realtime")

_PLAYERS = (
    ["aplay", "-q"],
    ["paplay"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
)


def play_wav_blocking(path: Path, *, device: str | None = None) -> None:
    for base_cmd in _PLAYERS:
        cmd = list(base_cmd)
        if device and cmd[0] == "aplay":
            cmd += ["-D", device]
        cmd += [str(path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    log.warning("No audio player found (aplay/paplay/ffplay) — leaving wav at %s", path)
