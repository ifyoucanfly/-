from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


logger = logging.getLogger(__name__)


@dataclass
class ODEStats:
    nfe: int = 0
    nan_encountered: bool = False
    max_abs_state: float = 0.0


def set_global_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def check_tensor_finite(x: torch.Tensor, name: str = "tensor", raise_on_fail: bool = True) -> bool:
    ok = torch.isfinite(x).all().item()
    if not ok:
        msg = f"Non-finite detected in tensor '{name}'."
        logger.error(msg)
        if raise_on_fail:
            raise FloatingPointError(msg)
    return bool(ok)


def count_nfe(odefunc: torch.nn.Module) -> int:
    return int(getattr(odefunc, "nfe", 0))


def mechanism_drift_heatmap(gate: torch.Tensor) -> torch.Tensor:
    """
    Returns a per-spot scalar drift intensity map from gate (N x d or N x 1).
    """
    if gate.ndim == 2:
        return gate.abs().mean(dim=1)
    return gate.abs().reshape(-1)


def total_variation_on_graph(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Graph TV penalty: sum_{(i,j)} |x_i - x_j|. x: (N,1) or (N,d)
    edge_index: (2, E)
    """
    i, j = edge_index[0], edge_index[1]
    return (x[i] - x[j]).abs().mean()


def safe_clip_grad_norm_(params, max_norm: float) -> float:
    """
    Clip gradients; returns the total norm before clipping.
    """
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))
