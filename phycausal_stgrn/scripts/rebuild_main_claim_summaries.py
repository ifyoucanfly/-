from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_save_dir(cfg: Dict, explicit_save_dir: Optional[str]) -> Path:
    if explicit_save_dir:
        p = Path(explicit_save_dir)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    save_dir = str(cfg.get("train", {}).get("save_dir", "") or "").strip()
    if not save_dir:
        raise ValueError("Could not resolve save_dir from config; pass --save_dir explicitly.")
    p = Path(save_dir)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _resolve_ckpt(save_dir: Path, explicit_ckpt: Optional[str]) -> Path:
    if explicit_ckpt:
        p = Path(explicit_ckpt)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    manifest = save_dir / "training_manifest.json"
    if manifest.exists():
        try:
            man = json.loads(manifest.read_text(encoding="utf-8"))
            for key in ["preferred_eval_ckpt_realized", "preferred_eval_ckpt"]:
                pref = str(man.get(key, "") or "").strip()
                if pref:
                    cand = save_dir / pref
                    if cand.exists():
                        return cand.resolve()
        except Exception:
            pass
    for name in ["model_main_claim_best.pt", "model_stage1_best.pt", "model_stage1_last.pt", "model_best_gate.pt", "model_best.pt", "model.pt", "checkpoint.pt"]:
        p = save_dir / name
        if p.exists():
            return p.resolve()
    cands = sorted(save_dir.glob("model_epoch_*.pt"))
    if cands:
        return cands[-1].resolve()
    raise FileNotFoundError(f"No checkpoint found under save_dir: {save_dir}")


def _split_summary(paper_summary: Dict) -> tuple[Dict, Dict, Dict]:
    task_primary = dict(paper_summary.get("task_primary", {}) or {})
    task_secondary = dict(paper_summary.get("task_secondary", {}) or {})
    component_switches = dict(paper_summary.get("component_switches", {}) or {})

    counterfactual_summary = {
        "patch_version": paper_summary.get("patch_version"),
        "component_switches": component_switches,
        "delta_p_cross": task_primary.get("delta_p_cross"),
        "p_cross_baseline": task_primary.get("p_cross_baseline"),
        "p_cross_do_z": task_primary.get("p_cross_do_z"),
        "p_cross_envelope_do_z": task_primary.get("p_cross_envelope_do_z"),
        "ood_flag_do_z": task_primary.get("ood_flag_do_z"),
        "roi_source": task_primary.get("roi_source"),
        "roi_x0": task_primary.get("roi_x0"),
        "roi_band": task_primary.get("roi_band"),
    }

    gate_summary = {
        "patch_version": paper_summary.get("patch_version"),
        "component_switches": component_switches,
        "gate_mean_interface": task_primary.get("gate_mean_interface"),
        "gate_mean_non_interface": task_primary.get("gate_mean_non_interface"),
        "gate_interface_enrichment": task_primary.get("gate_interface_enrichment"),
        "gate_activity_mean": task_primary.get("gate_activity_mean"),
        "gate_active_fraction": task_primary.get("gate_active_fraction"),
        "gate_sparsity": task_primary.get("gate_sparsity"),
        "roi_interface_fraction": task_primary.get("roi_interface_fraction"),
        "gate_num_cells": task_primary.get("gate_num_cells"),
        "roi_source": task_primary.get("roi_source"),
        "roi_x0": task_primary.get("roi_x0"),
        "roi_band": task_primary.get("roi_band"),
    }

    compact_paper = {
        "patch_version": paper_summary.get("patch_version"),
        "component_switches": component_switches,
        "task_primary": task_primary,
        "task_secondary": task_secondary,
        "ckpt_used": paper_summary.get("ckpt_used"),
        "ckpt_load_info": paper_summary.get("ckpt_load_info", {}),
        "plot_errors": paper_summary.get("plot_errors", []),
    }
    return compact_paper, counterfactual_summary, gate_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild main-claim summaries and copy them back into save_dir.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--save_dir", default=None, help="Optional override for train.save_dir.")
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint override.")
    ap.add_argument("--out_dir", default=None, help="Optional output directory for paper_figures.py.")
    ap.add_argument("--edge_method", default="jacobian", help="Edge method passed to paper_figures.py.")
    ap.add_argument("--top_k", type=int, default=2000, help="Top-k edges for auxiliary scoring.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    save_dir = _resolve_save_dir(cfg, args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _resolve_ckpt(save_dir, args.ckpt)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (save_dir / "paper_figures_rebuild").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_fig = PROJECT_ROOT / "phycausal_stgrn" / "scripts" / "paper_figures.py"
    if not paper_fig.exists():
        raise FileNotFoundError(f"paper_figures.py not found: {paper_fig}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        sys.executable,
        str(paper_fig),
        "--config", str(cfg_path),
        "--ckpt", str(ckpt_path),
        "--out_dir", str(out_dir),
        "--edge_method", str(args.edge_method),
        "--top_k", str(int(args.top_k)),
    ]
    print("[rebuild_main_claim_summaries] Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
    if rc != 0:
        raise RuntimeError(f"paper_figures.py failed with return code {rc}")

    paper_summary_path = out_dir / "paper_summary.json"
    paper_summary = _load_json(paper_summary_path)
    if paper_summary is None:
        err = _load_json(out_dir / "paper_error.json")
        raise RuntimeError(f"paper_summary.json missing or unreadable. paper_error={err}")

    compact_paper, counterfactual_summary, gate_summary = _split_summary(paper_summary)

    # Write into out_dir
    _write_json(out_dir / "paper_summary.json", compact_paper)
    _write_json(out_dir / "counterfactual_summary.json", counterfactual_summary)
    _write_json(out_dir / "gate_summary.json", gate_summary)

    # Copy back into save_dir for notebook/module3 discovery
    for name in ["paper_summary.json", "counterfactual_summary.json", "gate_summary.json"]:
        shutil.copy2(out_dir / name, save_dir / name)

    manifest = {
        "config_path": str(cfg_path),
        "save_dir": str(save_dir),
        "ckpt_used": str(ckpt_path),
        "out_dir": str(out_dir),
        "paper_summary_json": str(out_dir / "paper_summary.json"),
        "counterfactual_summary_json": str(out_dir / "counterfactual_summary.json"),
        "gate_summary_json": str(out_dir / "gate_summary.json"),
        "copied_back_to_save_dir": True,
    }
    _write_json(out_dir / "rebuild_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
