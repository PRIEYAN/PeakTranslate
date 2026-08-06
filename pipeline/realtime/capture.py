"""Mic capture + VAD endpointing.

Design: docs/05-vad-realtime-integration.md §5.

The sounddevice callback does the absolute minimum (append bytes to a
thread-safe queue) so ALSA never sees a slow callback and drops frames.
A separate thread pops raw frames, runs VAD, and emits complete
`Utterance`s into Q0 (`audio_queue`) with pre-roll prepended.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
import uuid
from typing import Optional

from .messages import STOP, Utterance

log = logging.getLogger("peaktranslation.realtime")


class VadEndpointer:
    """IDLE / SPEAKING state machine over fixed-size PCM frames.

    Frame size is fixed by `frame_ms` (webrtcvad only accepts 10/20/30 ms
    frames at 8/16/32/48 kHz).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        silence_ms: int = 500,
        min_utterance_ms: int = 300,
        max_utterance_ms: int = 10000,
        pre_roll_ms: int = 200,
        backend: str = "webrtc",
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # int16 mono
        self.backend = backend
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_utterance_ms // frame_ms)
        self.max_speech_frames = max(1, max_utterance_ms // frame_ms)
        pre_roll_frames = max(0, pre_roll_ms // frame_ms)
        self._pre_roll: collections.deque[bytes] = collections.deque(maxlen=pre_roll_frames)

        if backend == "webrtc":
            import webrtcvad

            self._vad = webrtcvad.Vad(aggressiveness)
        elif backend == "energy":
            self._vad = None
        else:
            raise ValueError(f"Unknown VAD backend: {backend!r} (use 'webrtc' or 'energy')")

        self._speaking = False
        self._speech_frames: list[bytes] = []
        self._silence_run = 0

    def _is_speech(self, frame: bytes) -> bool:
        if self.backend == "webrtc":
            return self._vad.is_speech(frame, self.sample_rate)
        # Energy fallback: RMS threshold, only for debugging without webrtcvad.
        import numpy as np

        samples = np.frombuffer(frame, dtype="<i2").astype("float32")
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
        return rms > 500.0

    def push_frame(self, frame: bytes) -> Optional[bytes]:
        """Feed one fixed-size PCM frame. Returns completed utterance PCM, if any."""
        if len(frame) != self.frame_bytes:
            return None

        is_speech = self._is_speech(frame)

        if not self._speaking:
            self._pre_roll.append(frame)
            if is_speech:
                self._speaking = True
                self._speech_frames = list(self._pre_roll)
                self._silence_run = 0
            return None

        self._speech_frames.append(frame)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        force_cut = len(self._speech_frames) >= self.max_speech_frames
        silence_cut = self._silence_run >= self.silence_frames
        if force_cut or silence_cut:
            frames = self._speech_frames
            self._speaking = False
            self._speech_frames = []
            self._silence_run = 0
            self._pre_roll.clear()
            if len(frames) >= self.min_speech_frames:
                return b"".join(frames)
        return None


class AudioCapture:
    """Owns the mic stream and the VAD thread. Emits Utterances into Q0.

    Many mics/PortAudio devices reject 16 kHz as an input rate (raises
    ``PortAudioError: Invalid sample rate``). To stay robust across
    hardware, the stream is opened at the device's own default rate and
    resampled down to ``sample_rate`` (16 kHz, what Whisper/VAD expect)
    before framing.
    """

    def __init__(
        self,
        *,
        audio_queue: "queue.Queue",
        sample_rate: int = 16000,
        frame_ms: int = 20,
        device: Optional[str] = None,
        vad_kwargs: Optional[dict] = None,
        on_drop=None,
    ) -> None:
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.device = device
        self.on_drop = on_drop or (lambda utt_id: None)

        self._endpointer = VadEndpointer(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            **(vad_kwargs or {}),
        )
        self._raw_frames: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._vad_thread: Optional[threading.Thread] = None
        self._capture_rate = sample_rate
        self._resample_up = 1
        self._resample_down = 1
        self._level_window: list = []
        self._last_level_log = 0.0

    def _sd_callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        # Runs on the PortAudio thread: append only, never block or infer here.
        self._raw_frames.put(bytes(indata))

    def _log_level(self, frame: bytes) -> None:
        import numpy as np

        peak = int(np.abs(np.frombuffer(frame, dtype="<i2")).max()) if frame else 0
        self._level_window.append(peak)
        now = time.time()
        if now - self._last_level_log >= 1.0:
            window_peak = max(self._level_window) if self._level_window else 0
            self._level_window.clear()
            self._last_level_log = now
            bar = "#" * min(40, window_peak // 800)
            log.info("mic level: peak=%5d %s%s", window_peak, bar, " (near silence — check input device/gain)" if window_peak < 300 else "")

    def _resample_to_target(self, chunk: bytes) -> bytes:
        if self._capture_rate == self.sample_rate:
            return chunk
        import numpy as np
        from scipy.signal import resample_poly

        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
        resampled = resample_poly(samples, self._resample_up, self._resample_down)
        return np.clip(resampled, -32768, 32767).astype("<i2").tobytes()

    def _vad_loop(self) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._raw_frames.get(timeout=0.1)
            except queue.Empty:
                continue
            buf += self._resample_to_target(chunk)
            while len(buf) >= self.frame_bytes:
                frame, buf = buf[: self.frame_bytes], buf[self.frame_bytes :]
                self._log_level(frame)
                utt_pcm = self._endpointer.push_frame(frame)
                if utt_pcm is not None:
                    self._emit(utt_pcm)

    def _emit(self, pcm: bytes) -> None:
        utt_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        utt = Utterance(
            utt_id=utt_id,
            pcm=pcm,
            sample_rate=self.sample_rate,
            duration_s=len(pcm) / 2 / self.sample_rate,
            t_captured=time.perf_counter(),
        )
        try:
            self.audio_queue.put_nowait(utt)
        except queue.Full:
            # Policy: drop oldest rather than block the capture path (see
            # docs/05-vad-realtime-integration.md §7). Never stall the mic.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(utt)
            except queue.Full:
                pass
            self.on_drop(utt_id)

    def start(self) -> None:
        import sounddevice as sd
        from fractions import Fraction

        self._stop.clear()

        # PortAudio's own "default device" sentinel (index -1) is unreliable
        # under some PipeWire/PulseAudio setups (raises "Error querying
        # device -1" even though a perfectly good input device exists).
        # Resolve a concrete device index up front instead of trusting it.
        self.device = self._resolve_device_index(sd)
        log.info(
            "Using input device #%s: %s",
            self.device,
            sd.query_devices(self.device)["name"],
        )

        self._capture_rate = self._pick_capture_rate(sd)
        frac = Fraction(self.sample_rate, self._capture_rate).limit_denominator(1000)
        self._resample_up, self._resample_down = frac.numerator, frac.denominator
        if self._capture_rate != self.sample_rate:
            log.info(
                "Mic does not support %d Hz directly; capturing at %d Hz and resampling (%d/%d).",
                self.sample_rate,
                self._capture_rate,
                self._resample_up,
                self._resample_down,
            )

        self._vad_thread = threading.Thread(target=self._vad_loop, name="vad", daemon=True)
        self._vad_thread.start()

        # blocksize is in frames-at-capture-rate; keep callbacks short regardless of rate.
        blocksize = int(self._capture_rate * self.frame_ms / 1000)
        self._stream = sd.RawInputStream(
            samplerate=self._capture_rate,
            blocksize=blocksize,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self._sd_callback,
        )
        self._stream.start()

    def _resolve_device_index(self, sd) -> int:  # noqa: ANN001
        """Resolve a concrete input device index, bypassing PortAudio's
        sometimes-broken -1 "default device" sentinel."""
        if self.device is not None:
            return self.device

        devices = sd.query_devices()

        try:
            default_idx = sd.default.device[0]
        except Exception:
            default_idx = None
        if default_idx is not None and default_idx >= 0:
            try:
                sd.query_devices(default_idx)
                return default_idx
            except Exception:
                pass

        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                log.warning(
                    "PortAudio default input device is unavailable; using #%d (%s) instead.",
                    i,
                    d["name"],
                )
                return i

        raise RuntimeError(
            "No input-capable audio device found via PortAudio. "
            "Run `python -c \"import sounddevice as sd; print(sd.query_devices())\"` to inspect."
        )

    def _pick_capture_rate(self, sd) -> int:  # noqa: ANN001
        """Try the target rate first; fall back to the device's own default rate."""
        try:
            sd.check_input_settings(device=self.device, samplerate=self.sample_rate, channels=1, dtype="int16")
            return self.sample_rate
        except Exception:
            pass
        try:
            info = sd.query_devices(self.device, "input")
            native = int(round(info["default_samplerate"]))
            sd.check_input_settings(device=self.device, samplerate=native, channels=1, dtype="int16")
            return native
        except Exception as e:  # noqa: BLE001
            log.warning("Could not confirm a working input samplerate (%s); trying %d Hz anyway.", e, self.sample_rate)
            return self.sample_rate

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._vad_thread is not None:
            self._vad_thread.join(timeout=2.0)
        # Cascade shutdown into the pipeline. Shutdown must never hang on a
        # full queue, so evict an item rather than give up on delivering STOP.
        for _ in range(2):
            try:
                self.audio_queue.put(STOP, timeout=1.0)
                return
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
        log.error("Could not deliver shutdown sentinel to audio_queue.")
