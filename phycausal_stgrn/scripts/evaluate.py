from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import argparse
import logging
import os
from typing import Dict, Optional

import torch
import yaml

from phycausal_stgrn.baselines.celcomen_runner import run_celcomen
from phycausal_stgrn.baselines.spagrn_runner import run_spagrn
from phycausal_stgrn.data.data_loader import load_dataset_from_config
from phycausal_stgrn.utils.metrics import pearson_corr, wasserstein2_cost

logger = logging.getLogger(__name__)


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, "evaluate.log"))],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--baseline", type=str, choices=["celcomen", "spagrn"], required=True)
    ap.add_argument("--out_dir", type=str, default="runs/baselines")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    setup_logging(args.out_dir)

    data_cfg = dict(cfg["data"])
    data_cfg["seed"] = int(cfg["seed"])
    dataset = load_dataset_from_config(data_cfg)
    snap1 = dataset.snapshots[1]

    if args.baseline == "celcomen":
        pred = run_celcomen(dataset)
    else:
        pred = run_spagrn(dataset)

    pear = pearson_corr(pred.x_pred_t1, snap1.X)
    w2 = wasserstein2_cost(pred.x_pred_t1, snap1.X, p=int(cfg["ot"].get("w1_p", 2)))

    logger.info("baseline=%s pearson=%.4f wasserstein_cost=%.4f", args.baseline, pear, w2)
    print(f"{args.baseline} | pearson={pear:.4f} | W={w2:.4f}")


if __name__ == "__main__":
    main()
