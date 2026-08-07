"""Factory: profile -> ReasoningEngine.

This is the only place a concrete backend class is named. Adding a new
runtime (e.g. a GGUF/llama.cpp backend) means one new adapter module, one
`profiles.yaml` entry, and one branch here — the pipeline worker,
`pipeline/run_realtime.py`, and every domain file are untouched.

Nothing in this module imports torch or transformers at import time: those
only load inside `build_engine`, via the per-runtime adapter import. That
keeps `import reason.src.runtime` cheap and keeps registry/profile errors
fast even without a GPU.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from .history import ConversationHistory
from .interface import ReasoningEngine
from .messages import Prompt, Turn
from .registry import resolve_profile, resolve_prompt_path
from .session import JarvisSession, JarvisTurn, StickyMode

__all__ = [
    "build_engine",
    "load_system_prompt",
    "ReasoningEngine",
    "Prompt",
    "Turn",
    "ConversationHistory",
    "JarvisSession",
    "JarvisTurn",
    "StickyMode",
]


def build_engine(
    profile_id: Optional[str] = None,
    *,
    config_path: Path | str | None = None,
    load_lock: Optional[threading.Lock] = None,
) -> ReasoningEngine:
    profile = resolve_profile(profile_id, config_path)
    if profile.get("device", "cuda") != "cuda":
        raise ValueError("Reasoning profile device must be cuda")

    runtime = profile.get("runtime", "gemma_pytorch")
    if runtime != "gemma_pytorch":
        raise NotImplementedError(f"Runtime {runtime!r} not implemented yet")

    from .gemma_pytorch import GemmaPytorchReasoner  # imported per-runtime

    decode = profile.get("decode") or {}
    return GemmaPytorchReasoner(
        profile["model_id"],
        quantization=profile.get("quantization", "nf4"),
        compute_dtype=profile.get("compute_dtype", "bfloat16"),
        max_new_tokens=int(decode.get("max_new_tokens", 96)),
        temperature=float(decode.get("temperature", 0.7)),
        top_p=float(decode.get("top_p", 0.9)),
        load_lock=load_lock,
    )


def load_system_prompt(profile_id: Optional[str] = None, *, config_path: Path | str | None = None) -> str:
    profile = resolve_profile(profile_id, config_path)
    path = resolve_prompt_path(profile)
    return path.read_text(encoding="utf-8").strip() if path else ""
