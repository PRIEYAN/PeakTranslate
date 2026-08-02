"""Load languages.yaml profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

STT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = STT_ROOT / "configs" / "languages.yaml"


def load_profiles(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "profiles" not in data:
        raise ValueError(f"Invalid STT config: {cfg_path}")
    return data


def resolve_profile(profile_id: str | None = None, path: Path | str | None = None) -> dict[str, Any]:
    data = load_profiles(path)
    pid = profile_id or data.get("default_profile")
    if pid not in data["profiles"]:
        raise KeyError(f"Unknown STT profile '{pid}'. Known: {list(data['profiles'])}")
    profile = dict(data["profiles"][pid])
    profile["id"] = pid
    return profile


def resolve_artifact_path(profile: dict[str, Any], stt_root: Path | None = None) -> Path:
    root = stt_root or STT_ROOT
    rel = profile.get("artifact") or profile.get("finetuned")
    if not rel:
        raise KeyError("Profile missing artifact/finetuned path")
    p = Path(rel)
    return p if p.is_absolute() else (root / p)
