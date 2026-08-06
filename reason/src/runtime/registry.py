"""Load profiles.yaml profiles. Near-copy of stt/src/runtime/registry.py —
same shape, same error messages, no surprises.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

REASON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REASON_ROOT / "configs" / "profiles.yaml"


def load_profiles(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "profiles" not in data:
        raise ValueError(f"Invalid reason config: {cfg_path}")
    return data


def resolve_profile(profile_id: str | None = None, path: Path | str | None = None) -> dict[str, Any]:
    data = load_profiles(path)
    pid = profile_id or data.get("default_profile")
    if pid not in data["profiles"]:
        raise KeyError(f"Unknown reason profile '{pid}'. Known: {list(data['profiles'])}")
    profile = dict(data["profiles"][pid])
    profile["id"] = pid
    return profile


def resolve_prompt_path(profile: dict[str, Any], reason_root: Path | None = None) -> Optional[Path]:
    rel = profile.get("prompt")
    if not rel:
        return None
    root = reason_root or REASON_ROOT
    p = Path(rel)
    return p if p.is_absolute() else (root / p)
