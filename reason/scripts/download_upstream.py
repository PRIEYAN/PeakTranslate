#!/usr/bin/env python3
"""Download a Gemma profile's model_id into reason/models/upstream/.

Optional: the pipeline works fine downloading straight from the Hub cache
(HF_HOME in .env) on first run. This script is only useful for pinning a
local snapshot, e.g. before an offline demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

REASON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REASON_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from pipeline._bootstrap import ensure_venv_on_path  # noqa: E402

# `venv/bin/python` here doesn't self-detect as a venv, so third-party
# imports (dotenv, huggingface_hub, ...) 404 unless this runs first — same
# fix pipeline/run_realtime.py applies.
ensure_venv_on_path()

import argparse  # noqa: E402
import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="Profile id in configs/profiles.yaml (default: default_profile)")
    args = parser.parse_args()

    from reason.src.runtime.registry import resolve_profile

    profile = resolve_profile(args.profile)
    model_id = profile["model_id"]

    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing in .env — set it before downloading.", file=sys.stderr)
        return 1

    from huggingface_hub import login, snapshot_download

    login(token=token, add_to_git_credential=False)
    out = REASON_ROOT / "models" / "upstream" / model_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {out}")
    print("Note: this model is gated — accept its licence on the Hub model page first.")
    snapshot_download(repo_id=model_id, local_dir=str(out), token=token)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
