from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(out_dir: str, level: int = logging.INFO, name: str = "phycausal_stgrn") -> None:
    os.makedirs(out_dir, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, "run.log"))]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    logging.getLogger(name).setLevel(level)
