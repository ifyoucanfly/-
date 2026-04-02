from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import torch

from phycausal_stgrn.data.data_loader import SpatialDataset


@dataclass
class BaselinePred:
    x_pred_t1: torch.Tensor  # (N, G)
    grn_scores: Optional[torch.Tensor] = None  # (E,) optional edge scores


class BaselineRunner(Protocol):
    def run(self, dataset: SpatialDataset, **kwargs) -> BaselinePred: ...


def run_celcomen(dataset: SpatialDataset, **kwargs) -> BaselinePred:
    """Placeholder Celcomen baseline interface.

    Implement:
      - call your Celcomen model to predict X(t1) from X(t0), coords, etc.
      - optionally output GRN edge scores (E,)
    """
    # TODO: replace with actual Celcomen inference
    x_pred = dataset.snapshots[0].X.clone()
    return BaselinePred(x_pred_t1=x_pred, grn_scores=None)
