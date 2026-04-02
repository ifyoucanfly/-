from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from typing import Dict
import os

def run(cfg: Dict) -> None:
    # Keep exact behavior of existing robustness_test.py (logic preservation)
    os.system("python robustness_test.py --config configs/default.yaml")
