from __future__ import annotations

# Ensure the local repo copy is importable regardless of current working directory.
import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import logging
import os as _os
import subprocess
from typing import Dict

import torch

from phycausal_stgrn.config import load_config
from phycausal_stgrn.logging_utils import setup_logging
from phycausal_stgrn.seed import seed_all

from phycausal_stgrn import __version__, __patch_version__, resolve_patch_version

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="phycausal-stgrn", description="PhyCausal-STGRN v2.2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model")
    p_train.add_argument("--config", type=str, default="configs/default.yaml")
    p_train.add_argument("--out_dir", type=str, default="runs")
    p_train.add_argument("--ckpt_name", type=str, default="model.pt")

    p_export = sub.add_parser("export-edges", help="Export GRN edges from an existing checkpoint")
    p_export.add_argument("--config", type=str, default="configs/default.yaml")
    p_export.add_argument("--ckpt", type=str, required=True)
    p_export.add_argument("--out_dir", type=str, default=None)

    p_rob = sub.add_parser("robustness", help="Run robustness + ablation")
    p_rob.add_argument("--config", type=str, default="configs/default.yaml")

    p_fig = sub.add_parser("figures", help="Generate paper figures & summaries")
    p_fig.add_argument("--config", type=str, default="configs/default.yaml")
    p_fig.add_argument("--ckpt", type=str, default="runs/model.pt")
    p_fig.add_argument("--out_dir", type=str, default="runs/paper_figures")

    return ap


def main(argv=None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    cfg = load_config(args.config).raw
    seed_all(int(cfg["seed"]))

    # route
    if args.cmd == "train":
        setup_logging(args.out_dir)
        logger.info("Running phycausal_stgrn %s (%s)", __version__, resolve_patch_version(getattr(args, "config", None)))
        import phycausal_stgrn.scripts.train_entry as train_entry
        train_entry.run(cfg, out_dir=args.out_dir, ckpt_name=args.ckpt_name)
        return

    if args.cmd == "export-edges":
        setup_logging(args.out_dir or str(cfg.get("train", {}).get("save_dir", "runs")))
        import phycausal_stgrn.scripts.export_edges_entry as export_entry
        export_entry.run(cfg, ckpt_path=args.ckpt, out_dir=args.out_dir)
        return

    if args.cmd == "robustness":
        import phycausal_stgrn.scripts.robustness_entry as rob_entry
        rob_entry.run(cfg)
        return

    if args.cmd == "figures":
        import phycausal_stgrn.scripts.paper_figures as pf
        # reuse existing script main via argparse-like call
        cmd = [
            "python",
            "-m",
            "phycausal_stgrn.scripts.paper_figures",
            "--config",
            str(args.config),
            "--ckpt",
            str(args.ckpt),
            "--out_dir",
            str(args.out_dir),
        ]
        subprocess.run(cmd, check=True, cwd=os.path.abspath(".."))
        return


if __name__ == "__main__":
    main()
