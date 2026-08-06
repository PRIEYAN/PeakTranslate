"""Loads the real configs/profiles.yaml — catches config typos without a GPU."""
from reason.src.runtime.registry import resolve_profile, resolve_prompt_path


def test_default_profile_resolves():
    profile = resolve_profile()
    assert profile["runtime"] == "gemma_pytorch"
    assert profile["device"] == "cuda"
    assert "model_id" in profile


def test_named_profile_resolves():
    profile = resolve_profile("gemma-2b-it-4bit")
    assert profile["id"] == "gemma-2b-it-4bit"
    assert profile["model_id"] == "google/gemma-2b-it"


def test_unknown_profile_raises_with_known_list():
    try:
        resolve_profile("does-not-exist")
    except KeyError as e:
        assert "does-not-exist" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_prompt_path_resolves_and_file_exists():
    profile = resolve_profile()
    path = resolve_prompt_path(profile)
    assert path is not None
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()
