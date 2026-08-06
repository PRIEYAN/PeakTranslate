"""Infrastructure layer — torch is allowed here, but none of this needs CUDA."""
import pytest
import torch

from reason.src.runtime.loading import assert_materialized, build_quantization_config


class FakeModel:
    def __init__(self, *, meta: bool) -> None:
        device = "meta" if meta else "cpu"
        self._params = [("weight", torch.empty(2, 2, device=device))]

    def named_parameters(self):
        return self._params


def test_none_quantization_returns_none():
    assert build_quantization_config(None, "bfloat16") is None
    assert build_quantization_config("none", "bfloat16") is None


def test_nf4_returns_a_config():
    cfg = build_quantization_config("nf4", "bfloat16")
    assert cfg is not None
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown quantization"):
        build_quantization_config("int8", "bfloat16")


def test_assert_materialized_passes_for_real_tensors():
    assert_materialized(FakeModel(meta=False), "test")  # must not raise


def test_assert_materialized_raises_and_names_the_parameter():
    with pytest.raises(RuntimeError, match="weight"):
        assert_materialized(FakeModel(meta=True), "test")
