from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "cpu"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))


def load_config(path: str) -> Config:
    """Load YAML config and return a lightweight validated wrapper.

    We intentionally keep this minimal (no heavy config frameworks) to ensure
    CPU/GPU agnostic reproducibility and easy inspection.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping, got: {type(raw)}")
    # minimal required keys
    for k in ["data", "model", "solver", "train", "ot", "ood"]:
        if k not in raw:
            raise KeyError(f"Missing config section: {k}")
    return Config(raw=raw)
