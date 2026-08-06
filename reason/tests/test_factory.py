"""build_engine's validation happens before any torch/transformers import,
so the error paths are testable without a GPU. The happy path (constructing
a GemmaPytorchReasoner) is exercised by scripts/smoke_reason.py and
scripts/bench_reason.py instead, which do need CUDA.
"""
import pytest
import yaml

from reason.src.runtime import build_engine


def _write_config(tmp_path, profiles: dict):
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump({"default_profile": "p", "profiles": profiles}))
    return path


def test_non_cuda_device_rejected(tmp_path):
    cfg = _write_config(tmp_path, {"p": {"model_id": "x", "device": "cpu"}})
    with pytest.raises(ValueError, match="cuda"):
        build_engine("p", config_path=cfg)


def test_unknown_runtime_rejected(tmp_path):
    cfg = _write_config(
        tmp_path, {"p": {"model_id": "x", "device": "cuda", "runtime": "ggml"}}
    )
    with pytest.raises(NotImplementedError, match="ggml"):
        build_engine("p", config_path=cfg)


def test_unknown_profile_rejected(tmp_path):
    cfg = _write_config(tmp_path, {"p": {"model_id": "x", "device": "cuda"}})
    with pytest.raises(KeyError):
        build_engine("does-not-exist", config_path=cfg)
