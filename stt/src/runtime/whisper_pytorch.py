"""Whisper PyTorch adapter — CUDA only (no CPU fallback)."""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .messages import AudioChunk, Transcript

# transformers' `from_pretrained` suppresses weight tying by monkeypatching
# `PreTrainedModel.tie_weights`, which is process-global state. Two threads
# loading models at once can therefore swallow each other's tie call, leaving
# tied weights (here `proj_out.weight`) as the unmaterialized meta-device
# placeholders the skeleton was built from — which then blow up on
# `.to(device)`. See docs/06-debugging-meta-tensor-load-race.md.
# Callers that load other transformers models concurrently must pass their own
# shared lock as `load_lock`; this default only guards WhisperPytorchSTT
# instances against each other.
_DEFAULT_LOAD_LOCK = threading.Lock()


class WhisperPytorchSTT:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        require_cuda: bool = True,
        allow_cpu_fallback: bool = False,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 1,
        load_lock: Optional[threading.Lock] = None,
    ) -> None:
        if allow_cpu_fallback:
            raise ValueError("STT policy forbids CPU fallback (allow_cpu_fallback must be false)")
        if require_cuda and (device != "cuda" or not torch.cuda.is_available()):
            raise RuntimeError(
                "STT requires CUDA. torch.cuda.is_available() is False or device!=cuda. "
                "Refusing CPU fallback."
            )
        self.device = torch.device("cuda")
        self.language = language
        self.task = task
        self.beam_size = beam_size
        path = str(model_path)
        with load_lock if load_lock is not None else _DEFAULT_LOAD_LOCK:
            self.processor = WhisperProcessor.from_pretrained(path)
            self.model = WhisperForConditionalGeneration.from_pretrained(path)
            unmaterialized = [n for n, t in self.model.named_parameters() if t.is_meta]
            if unmaterialized:
                raise RuntimeError(
                    f"Whisper load left parameters on the meta device: {unmaterialized}. "
                    "See docs/06-debugging-meta-tensor-load-race.md."
                )
            self.model.to(self.device)
        self.model.eval()
        # This export is fp16 (config.json "dtype": "float16") and transformers
        # honours that on load, but WhisperFeatureExtractor always emits fp32
        # mel features. Feeding those straight in fails at the first conv with
        # "Input type (float) and bias type (c10::Half) should be the same",
        # so remember the weight dtype and cast inputs to match it.
        self.dtype = self.model.dtype

    def warmup(self) -> None:
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)
        self._transcribe_array(silence, sr, language=self.language)

    def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: Optional[str] = None,
    ) -> Transcript:
        samples = self._pcm_to_float32(audio.pcm, audio.sample_rate)
        if audio.sample_rate != 16000:
            samples = self._resample(samples, audio.sample_rate, 16000)
            sr = 16000
        else:
            sr = audio.sample_rate
        lang = language if language is not None else self.language
        text = self._transcribe_array(samples, sr, language=lang).strip()
        return Transcript(
            text=text,
            lang=lang,
            is_final=True,
            session_id=audio.session_id,
            is_speech=bool(text),
        )

    def _transcribe_array(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        language: Optional[str],
    ) -> str:
        inputs = self.processor(
            samples,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        gen_kwargs = {
            "max_new_tokens": 225,
            "num_beams": self.beam_size,
        }
        forced = self.processor.get_decoder_prompt_ids(
            language=language or "en",
            task=self.task,
        )
        # When language is None, skip forced language if processor allows — use en as safe default for prompt structure
        if language is None:
            # multilingual free-decode: still need task token; use auto via generate without forced lang when possible
            try:
                forced = self.processor.get_decoder_prompt_ids(language=None, task=self.task)
            except Exception:
                forced = self.processor.get_decoder_prompt_ids(language="en", task=self.task)
        gen_kwargs["forced_decoder_ids"] = forced

        with torch.inference_mode():
            predicted = self.model.generate(input_features, **gen_kwargs)
        text = self.processor.batch_decode(predicted, skip_special_tokens=True)[0]
        return text

    @staticmethod
    def _pcm_to_float32(pcm: bytes, sample_rate: int) -> np.ndarray:
        # Assume s16le mono
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    @staticmethod
    def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        import librosa

        return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
