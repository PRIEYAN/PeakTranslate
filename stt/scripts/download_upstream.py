#!/usr/bin/env python3
"""Download openai/whisper-base into stt/models/upstream/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

STT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STT_ROOT.parent


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing in .env — set it before downloading.", file=sys.stderr)
        return 1

    from huggingface_hub import login, snapshot_download

    login(token=token, add_to_git_credential=False)
    out = STT_ROOT / "models" / "upstream" / "openai-whisper-base"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading openai/whisper-base → {out}")
    snapshot_download(
        repo_id="openai/whisper-base",
        local_dir=str(out),
        token=token,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
