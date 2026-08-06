"""Load-time policy: quantization config + post-load guards.

`assert_materialized` is deliberately duplicated from
pipeline/realtime/workers.py rather than imported: reason/ must not depend
on pipeline/, or it stops being usable standalone (scripts/smoke_reason.py,
scripts/bench_reason.py). See docs/06-debugging-meta-tensor-load-race.md.
"""
from __future__ import annotations

from typing import Any, Optional


def build_quantization_config(kind: Optional[str], compute_dtype: str) -> Optional[Any]:
    """`nf4` -> BitsAndBytesConfig; `none`/`None` -> None. Anything else errors."""
    import torch
    from transformers import BitsAndBytesConfig

    if kind in (None, "none"):
        return None
    if kind != "nf4":
        raise ValueError(f"Unknown quantization {kind!r} (use 'nf4' or 'none')")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
    )


def assert_materialized(model, what: str) -> None:
    """Fail at the load, naming the parameter, not later inside torch."""
    unmaterialized = [n for n, t in model.named_parameters() if t.is_meta]
    if unmaterialized:
        raise RuntimeError(
            f"{what} load left parameters on the meta device: {unmaterialized}. "
            "See docs/06-debugging-meta-tensor-load-race.md."
        )
