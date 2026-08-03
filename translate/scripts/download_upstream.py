#!/usr/bin/env python3
"""Download Helsinki-NLP/opus-mt-en-hi into translate/models/upstream/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

TR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TR_ROOT.parent


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing in repo .env", file=sys.stderr)
        return 1

    from huggingface_hub import login, snapshot_download

    login(token=token, add_to_git_credential=False)
    out = TR_ROOT / "models" / "upstream" / "opus-mt-en-hi"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Helsinki-NLP/opus-mt-en-hi → {out}")
    snapshot_download(
        repo_id="Helsinki-NLP/opus-mt-en-hi",
        local_dir=str(out),
        token=token,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
