from __future__ import annotations

from phycausal_stgrn.utils.diagnostics import set_global_determinism

__all__ = ["seed_all"]

def seed_all(seed: int) -> None:
    """Set global RNG seeds (python/numpy/torch) for reproducibility."""
    set_global_determinism(seed)
