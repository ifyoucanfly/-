from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from phycausal_stgrn.data.data_loader import SpatialDataset
from phycausal_stgrn.baselines.celcomen_runner import BaselinePred


def run_spagrn(dataset: SpatialDataset, **kwargs) -> BaselinePred:
    """Placeholder SpaGRN baseline interface."""
    # TODO: replace with actual SpaGRN inference
    x_pred = dataset.snapshots[0].X.clone()
    return BaselinePred(x_pred_t1=x_pred, grn_scores=None)
