"""Blocking WAV playback for the playback worker.

Hard rule (docs/05-vad-realtime-integration.md §6.5): exactly one active
playback at a time. This module only ever plays one file synchronously;
serialization is guaranteed by the playback worker owning a single thread.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("peaktranslation.realtime")

_PLAYERS = (
    ["aplay", "-q"],
    ["paplay"],
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
)

# How often to check `interrupt` while a player subprocess runs. Bounds
# barge-in latency for audio that's already playing — see
# docs/reasoningModel/01-gemma-reasoning-mode.md §19.
_POLL_S = 0.03


def play_wav_blocking(
    path: Path, *, device: str | None = None, interrupt: Optional[threading.Event] = None
) -> None:
    """Play `path` to completion, or until `interrupt` is set.

    `subprocess.run()` (the old implementation) blocks until the player
    exits with no way to cut it off from another thread. Using `Popen` and
    polling instead lets a barge-in kill mid-playback audio within one
    `_POLL_S` tick rather than waiting for the file to finish.
    """
    for base_cmd in _PLAYERS:
        cmd = list(base_cmd)
        if device and cmd[0] == "aplay":
            cmd += ["-D", device]
        cmd += [str(path)]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue

        while True:
            try:
                returncode = proc.wait(timeout=_POLL_S)
                break
            except subprocess.TimeoutExpired:
                if interrupt is not None and interrupt.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    log.info("Playback interrupted (barge-in): %s", path)
                    return

        if returncode == 0:
            return
        # Non-zero exit (e.g. player installed but errored on this file):
        # fall through to the next candidate player, same as before.

    log.warning("No audio player found (aplay/paplay/ffplay) — leaving wav at %s", path)
