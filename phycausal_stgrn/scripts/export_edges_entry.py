from __future__ import annotations

import logging
import os
from typing import Dict

import torch

from phycausal_stgrn.data.data_loader import load_dataset_from_config
from phycausal_stgrn.scripts.train_entry import build_model, export_grn_edges
from phycausal_stgrn.utils.diagnostics import set_global_determinism

logger = logging.getLogger(__name__)


def run(cfg: Dict, ckpt_path: str, out_dir: str | None = None) -> Dict[str, str]:
    set_global_determinism(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cpu"))

    data_cfg = dict(cfg["data"])
    data_cfg["seed"] = int(cfg["seed"])
    dataset = load_dataset_from_config(data_cfg)
    model = build_model(cfg, dataset).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    save_dir = out_dir or str(cfg.get("train", {}).get("save_dir", "runs"))
    os.makedirs(save_dir, exist_ok=True)
    return export_grn_edges(cfg, dataset, model, save_dir, device)
