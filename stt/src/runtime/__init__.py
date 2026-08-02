"""Factory: profile → STTEngine (CUDA-only Whisper)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .registry import resolve_artifact_path, resolve_profile
from .whisper_pytorch import WhisperPytorchSTT


def build_engine(
    profile_id: Optional[str] = None,
    *,
    config_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> WhisperPytorchSTT:
    profile = resolve_profile(profile_id, config_path)
    if profile.get("allow_cpu_fallback", False):
        raise ValueError("Profile has allow_cpu_fallback=true — forbidden for STT")
    if profile.get("device", "cuda") != "cuda":
        raise ValueError("STT profile device must be cuda")

    runtime = profile.get("runtime", "whisper_pytorch")
    if runtime != "whisper_pytorch":
        raise NotImplementedError(f"Runtime '{runtime}' not implemented yet")

    artifact = Path(model_path) if model_path else resolve_artifact_path(profile)
    if not artifact.exists():
        # Fall back to finetuned, then upstream local, then Hub id
        finetuned = profile.get("finetuned")
        upstream = profile.get("upstream")
        candidates = []
        if finetuned:
            candidates.append(Path(finetuned) if Path(finetuned).is_absolute() else resolve_artifact_path({**profile, "artifact": finetuned}))
        # try relative from stt root via registry helper
        from .registry import STT_ROOT

        for rel in (finetuned, f"models/upstream/{Path(str(upstream)).name}" if upstream else None):
            if not rel:
                continue
            p = Path(rel) if Path(rel).is_absolute() else STT_ROOT / rel
            candidates.append(p)
        chosen: Any = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            if upstream:
                chosen = upstream  # Hub id / path string for from_pretrained
            else:
                raise FileNotFoundError(f"No model found for profile {profile['id']} at {artifact}")
        artifact = chosen

    decode = profile.get("decode") or {}
    return WhisperPytorchSTT(
        artifact,
        device=profile.get("device", "cuda"),
        require_cuda=bool(profile.get("require_cuda", True)),
        allow_cpu_fallback=bool(profile.get("allow_cpu_fallback", False)),
        language=decode.get("language"),
        task=decode.get("task", "transcribe"),
        beam_size=int(decode.get("beam_size", 1)),
    )
