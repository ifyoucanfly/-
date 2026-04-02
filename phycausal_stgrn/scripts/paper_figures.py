from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import traceback

import numpy as np
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from phycausal_stgrn.data.data_loader import load_dataset_from_config
from phycausal_stgrn.engine.ot_alignment import OTCfg
from phycausal_stgrn.engine.trainer import OODCfg, SolverCfg, Trainer
from phycausal_stgrn.grn.infer_edges import EdgeInferenceCfg, EdgeScores, infer_edges
from phycausal_stgrn.models.PhyCausal_ODE import ODEInput, PhyCausalSTGRN
from phycausal_stgrn.utils.diagnostics import set_global_determinism
from phycausal_stgrn.utils.metrics import aupr_edge, precision_recall_curve, export_pr_curve_csv, wasserstein1_cost, emd_cost, wasserstein2_cost
from phycausal_stgrn.utils.gold_sanity import inspect_gold_file, drop_bad_tokens
from phycausal_stgrn.utils.roi import define_interface_roi_adaptive, p_cross_interface
from phycausal_stgrn.viz.money_figure import plot_tcell_infiltration_flow, plot_attractor_shift
from phycausal_stgrn.viz.diagnostic_figures import plot_mechanism_drift_heatmap, plot_pr_curve, plot_gate_signal_overlay, plot_coverage_risk_curve, plot_misalignment_robustness_curve, write_animation_gif
from phycausal_stgrn import resolve_patch_version

logger = logging.getLogger(__name__)


def _get_patch_version(config_path: str | None = None) -> str:
    return resolve_patch_version(config_path)


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, "paper_figures.log"))],
        force=True,
    )


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def build_model(cfg: Dict, dataset) -> PhyCausalSTGRN:
    snap0 = dataset.snapshots[0]
    return PhyCausalSTGRN(
        num_genes=int(cfg["data"]["num_genes"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_gnn_layers=int(cfg["model"]["num_layers"]),
        zI_dim=snap0.zI.shape[1],
        zN_dim=snap0.zN.shape[1],
        zE_dim=snap0.zE.shape[1],
        mech_dim=snap0.mechanomics.shape[1],
        gate_hidden_dim=int(cfg["model"].get("gate_hidden", cfg["model"]["hidden_dim"])),
        enable_mms=bool(cfg["model"]["enable_mms"]),
        enable_mechanomics=bool(cfg["model"]["enable_mechanomics"]),
        upwind_drift=bool(cfg["model"]["upwind_drift"]),
        D_max=float(cfg.get("model", {}).get("D_max", 1.0) or 1.0),
        v_max=float(cfg.get("model", {}).get("v_max", 1.0) or 1.0),
        dy_clip=float(cfg.get("model", {}).get("dy_clip", 50.0) or 50.0),
        gate_mode=str(cfg.get("model", {}).get("gate_mode", "soft_sigmoid") or "soft_sigmoid"),
        gate_temperature=float(cfg.get("model", {}).get("gate_temperature", 1.5) or 1.5),
        gate_tau_min=float(cfg.get("model", {}).get("gate_tau_min", 0.75) or 0.75),
        gate_shared_across_latent=bool(cfg.get("model", {}).get("gate_shared_across_latent", True)),
        gate_init_bias=float(cfg.get("model", {}).get("gate_init_bias", -0.34) or -0.34),
        gate_interface_boost=float(cfg.get("model", {}).get("gate_interface_boost", 0.56) or 0.56),
        gate_floor=float(cfg.get("model", {}).get("gate_floor", 2e-4) or 2e-4),
        gate_sparse_deadzone=float(cfg.get("model", {}).get("gate_sparse_deadzone", 0.0) or 0.0),
        mms_bg_floor=float(cfg.get("model", {}).get("mms_bg_floor", 0.08) or 0.08),
        mms_gain=float(cfg.get("model", {}).get("mms_gain", 1.0) or 1.0),
    )


def _find_best_ckpt(cfg: Dict, explicit_ckpt: str) -> str:
    if explicit_ckpt and os.path.exists(explicit_ckpt):
        return explicit_ckpt
    save_dir = str(cfg.get("train", {}).get("save_dir", "") or "")
    if save_dir:
        save_path = Path(save_dir)
        if not save_path.is_absolute():
            save_path = PROJECT_ROOT / save_path
        manifest = save_path / "training_manifest.json"
        if manifest.exists():
            try:
                man = json.loads(manifest.read_text(encoding="utf-8"))
                pref = str(man.get("preferred_eval_ckpt_realized", man.get("preferred_eval_ckpt", "")) or "").strip()
                if pref:
                    cand = save_path / pref
                    if cand.exists():
                        return str(cand)
            except Exception:
                pass
        for p in [save_path / "model_main_claim_best.pt", save_path / "model_stage1_best.pt", save_path / "model_stage1_last.pt", save_path / "model_best_gate.pt", save_path / "model_best.pt", save_path / "model.pt", save_path / "checkpoint.pt"]:
            if p.exists():
                return str(p)
    return explicit_ckpt


def _reduce_gate_to_cells(gate: torch.Tensor, roi_len: int) -> torch.Tensor:
    g = gate.detach().float()
    if g.ndim == 1:
        if g.numel() != roi_len:
            raise ValueError(f"1D gate length {g.numel()} does not match roi_len={roi_len}")
        return g
    if g.shape[0] == roi_len:
        return g.reshape(roi_len, -1).mean(dim=1)
    if g.shape[-1] == roi_len:
        return g.reshape(-1, roi_len).mean(dim=0)
    gf = g.reshape(-1)
    if gf.numel() == roi_len:
        return gf
    raise ValueError(f"Cannot align gate shape {tuple(g.shape)} to roi_len={roi_len}")


def _gate_task_metrics(gate: torch.Tensor, roi, gate_p_nonzero: Optional[torch.Tensor] = None) -> Dict[str, float]:
    roi_mask = roi.mask_interface.reshape(-1).bool()
    g_cell = _reduce_gate_to_cells(gate, int(roi_mask.numel())).reshape(-1)
    activity_src = gate if isinstance(gate, torch.Tensor) else gate_p_nonzero
    a_cell = _reduce_gate_to_cells(activity_src, int(roi_mask.numel())).reshape(-1)
    non_mask = ~roi_mask
    gate_mean_interface = float(a_cell[roi_mask].mean().item()) if roi_mask.any() else 0.0
    gate_mean_non_interface = float(a_cell[non_mask].mean().item()) if non_mask.any() else 0.0
    gate_activity_mean = float(a_cell.mean().item()) if a_cell.numel() else 0.0
    return {
        "gate_mean_interface": gate_mean_interface,
        "gate_mean_non_interface": gate_mean_non_interface,
        "gate_interface_enrichment": float(gate_mean_interface / max(gate_mean_non_interface, 1e-8)),
        "gate_selectivity_gap": float(gate_mean_interface - gate_mean_non_interface),
        "gate_activity_mean": gate_activity_mean,
        "gate_active_fraction": float((a_cell > 0.2).float().mean().item()),
        "gate_sparsity": float(1.0 - gate_activity_mean),
        "roi_interface_fraction": float(roi_mask.float().mean().item()),
        "gate_num_cells": int(a_cell.numel()),
    }


def _prepare_runtime_cfg(cfg_yaml: Dict, ckpt_obj: object) -> Dict:
    cfg = dict(cfg_yaml or {})
    ckpt_cfg = ckpt_obj.get("config", {}) if isinstance(ckpt_obj, dict) else {}
    for sec in ["model", "solver", "ood", "grn_export"]:
        if isinstance(ckpt_cfg, dict) and sec in ckpt_cfg:
            cfg[sec] = _deep_update(cfg.get(sec, {}), ckpt_cfg.get(sec, {}))
    return cfg


def _load_checkpoint_partial(model, ckpt_obj) -> Dict[str, object]:
    state = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    model_state = model.state_dict()
    filtered = {}
    shape_mismatches = []
    unexpected = []
    for k, v in state.items():
        if k not in model_state:
            unexpected.append(k)
        elif tuple(v.shape) != tuple(model_state[k].shape):
            shape_mismatches.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        else:
            filtered[k] = v
    missing, unexpected2 = model.load_state_dict(filtered, strict=False)
    return {"missing": list(missing), "unexpected": unexpected + list(unexpected2), "shape_mismatches": shape_mismatches, "loaded_keys": len(filtered)}




def _score_ckpt_compatibility(model, ckpt_obj):
    state = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
    model_state = model.state_dict()
    matches = 0
    mismatches = 0
    for k, v in state.items():
        if k in model_state:
            if tuple(v.shape) == tuple(model_state[k].shape):
                matches += 1
            else:
                mismatches += 1
    return matches, mismatches

def _build_ot_cfg(cfg: Dict) -> OTCfg:
    ot_cfg_dict = dict(cfg.get("ot", {}) or {})
    allowed = set(inspect.signature(OTCfg).parameters.keys())
    ot_cfg_dict = {k: v for k, v in ot_cfg_dict.items() if k in allowed}
    if not bool(cfg.get("model", {}).get("enable_ot", False)) and "lambda_ot" in allowed:
        ot_cfg_dict["lambda_ot"] = 0.0
    return OTCfg(**ot_cfg_dict)


def _prime_trainer_envelope_stats(trainer: Trainer, dataset, device: torch.device) -> None:
    zs = [snap.zE.detach().float() for snap in getattr(dataset, "snapshots", []) if getattr(snap, "zE", None) is not None]
    if not zs:
        logger.warning("No zE found on dataset snapshots; envelope stats unavailable.")
        return
    z_cat = torch.cat(zs, dim=0)
    trainer.zE_mu = z_cat.mean(dim=0).to(device)
    trainer.zE_sd = z_cat.std(dim=0, unbiased=False).clamp_min(1e-6).to(device)
    logger.info("Primed trainer envelope stats from dataset snapshots: dim=%d", int(trainer.zE_mu.numel()))


def _safe_rollout(trainer: Trainer, snap0, t_end: float, do_zE: Optional[torch.Tensor]):
    if getattr(trainer, "zE_mu", None) is not None and getattr(trainer, "zE_sd", None) is not None:
        return trainer.rollout_with_envelope(snap0, t_end=t_end, do_zE=do_zE)
    logger.warning("Envelope stats unavailable; using direct stage2 rollout fallback.")
    trainer.model.eval()
    ode_in = trainer._build_ode_input(snap0, stage="stage2")
    if do_zE is not None:
        ode_in = trainer._apply_counterfactual_zE(ode_in, do_zE)
    x0 = snap0.X.to(trainer.device)
    y0 = trainer.model.encode(x0)
    t = trainer._make_time_grid(float(t_end))
    yT = trainer._ode_solve(y0, ode_in, t)
    xT = trainer.model.decode(yT)
    return {
        "yT_learned": yT,
        "xT_learned": xT,
        "ood_flag": torch.tensor([0.0], device=trainer.device),
    }


def _best_effort_plot_tcell(model, ode_in, y0_latent, out_dir: str) -> Optional[str]:
    try:
        plot_tcell_infiltration_flow(model, ode_in, y0_latent, out_dir=out_dir)
        return None
    except TypeError as e1:
        try:
            plot_tcell_infiltration_flow(model, ode_in, out_dir=out_dir)
            return None
        except Exception as e2:
            return f"TypeError fallback failed: {type(e1).__name__}: {e1}; second: {type(e2).__name__}: {e2}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _load_exported_edges(save_dir: str, device: torch.device, top_k: int):
    pt_path = Path(save_dir) / "pred_edges.pt"
    if not pt_path.exists():
        return None
    try:
        payload = torch.load(pt_path, map_location=device)
        ei = payload.get("edge_index", None)
        sc = payload.get("scores", None)
        method = str(payload.get("method", "exported"))
        if ei is None or sc is None:
            return None
        ei = ei.to(device=device, dtype=torch.long)
        sc = sc.to(device=device, dtype=torch.float32)
        k = int(min(max(1, top_k), sc.numel()))
        if sc.numel() > k:
            order = torch.argsort(sc, descending=True)[:k]
            ei = ei[:, order]
            sc = sc[order]
        return EdgeScores(edge_index=ei, scores=sc, method=method)
    except Exception as e:
        logger.warning("Failed to load exported edges from %s: %s: %s", pt_path, type(e).__name__, e)
        return None



def _write_json(path: str | Path, payload: Dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _proxy_infiltration_depth(y: torch.Tensor, roi, coords: torch.Tensor) -> float:
    mask = roi.mask_interface.reshape(-1).to(y.device)
    tmask = (roi.mask_interface & roi.side_tumor).reshape(-1).to(y.device)
    smask = (roi.mask_interface & roi.side_stroma).reshape(-1).to(y.device)
    if int(tmask.sum().item()) < 2 or int(smask.sum().item()) < 2:
        return 0.0
    axis = (y[smask].mean(dim=0) - y[tmask].mean(dim=0))
    axis = axis / axis.norm(p=2).clamp_min(1e-6)
    proj = (y[mask] * axis.unsqueeze(0)).sum(dim=1)
    c = coords[mask].float()
    cx = c[:, 0]
    w = torch.softmax(proj, dim=0)
    return float((w * (cx - cx.min())).sum().item())


def _proxy_interface_density_shift(y_base: torch.Tensor, y_do: torch.Tensor, roi) -> float:
    mask = roi.mask_interface.reshape(-1).to(y_base.device)
    if int(mask.sum().item()) < 2:
        return 0.0
    return float((y_do[mask].norm(dim=1).mean() - y_base[mask].norm(dim=1).mean()).item())


def _build_coverage_risk(ood_score: torch.Tensor, learned: torch.Tensor, envelope: torch.Tensor, mask: torch.Tensor) -> Dict[str, object]:
    scores = ood_score.detach().float().reshape(-1)
    risk = (learned.detach().float() - envelope.detach().float()).abs().mean(dim=1)
    m = mask.reshape(-1).bool()
    if int(m.sum().item()) < 4:
        return {"coverage": [], "risk": []}
    qs = [0.20, 0.40, 0.60, 0.80, 0.95]
    coverages = []
    risks = []
    for q in qs:
        thr = torch.quantile(scores[m], q)
        keep = m & (scores <= thr)
        cov = float(keep.float().mean().item())
        rk = float(risk[keep].mean().item()) if int(keep.sum().item()) > 0 else float('nan')
        coverages.append(cov)
        risks.append(rk)
    return {"coverage": coverages, "risk": risks, "quantiles": qs}


def _maybe_ot_pairing_metrics(trainer: Trainer, snap0, snap1, yT: torch.Tensor, cfg: Dict) -> Dict[str, object]:
    from phycausal_stgrn.engine.ot_alignment import compute_pi_pot_sparse_knn
    out: Dict[str, object] = {"enabled": False}
    try:
        enable_ot_pairing = bool((cfg.get("component_switches", {}) or {}).get("enable_ot_pairing", False))
        if not enable_ot_pairing:
            return out
        out["enabled"] = True
        x0 = snap0.X.to(trainer.device)
        x1 = snap1.X.to(trainer.device)
        zi0, zn0 = trainer.model.encode_zi_zn(x0)
        zi1, zn1 = trainer.model.encode_zi_zn(x1)
        Pi = compute_pi_pot_sparse_knn(
            z_I_0=zi0, z_N_0=zn0, coords_0=snap0.coords.to(trainer.device),
            z_I_1=zi1, z_N_1=zn1, coords_1=snap1.coords.to(trainer.device),
            k=int(getattr(trainer.ot_cfg, "knn_k", 50) if trainer.ot_cfg is not None else 50),
            alpha=float(getattr(trainer.ot_cfg, "alpha", 0.5) if trainer.ot_cfg is not None else 0.5),
            epsilon=float(getattr(trainer.ot_cfg, "epsilon", 5e-2) if trainer.ot_cfg is not None else 5e-2),
            max_iter=int(getattr(trainer.ot_cfg, "max_iter", 100) if trainer.ot_cfg is not None else 100),
            a=snap0.mass.to(trainer.device) if getattr(snap0, "mass", None) is not None else None,
            b=snap1.mass.to(trainer.device) if getattr(snap1, "mass", None) is not None else None,
        )
        out.update({
            "pairing_nnz": int(Pi._nnz()),
            "w1": float(wasserstein1_cost(yT.detach(), x1.detach())),
            "emd": float(emd_cost(yT.detach(), x1.detach())),
            "w2": float(wasserstein2_cost(yT.detach(), x1.detach())),
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _misalignment_robustness_curve(trainer: Trainer, snap0, snap1, yT: torch.Tensor, cfg: Dict) -> Dict[str, object]:
    deltas = [5.0, 10.0, 50.0]
    series = {"w1": [], "emd": [], "coord_shift_norm": []}
    for delta in deltas:
        try:
            coords1 = snap1.coords.to(trainer.device).clone()
            coords1[:, 0] = coords1[:, 0] + float(delta)
            disp = (coords1 - snap1.coords.to(trainer.device)).norm(dim=1).mean()
            series["coord_shift_norm"].append(float(disp.item()))
            series["w1"].append(float(wasserstein1_cost(yT.detach(), snap1.X.to(trainer.device).detach())))
            series["emd"].append(float(emd_cost(yT.detach(), snap1.X.to(trainer.device).detach())))
        except Exception:
            series["coord_shift_norm"].append(float('nan'))
            series["w1"].append(float('nan'))
            series["emd"].append(float('nan'))
    return {"deltas_um": deltas, "metrics": series}


def _save_rollout_snapshots(out_dir: str, **tensors: torch.Tensor) -> Dict[str, str]:
    snap_dir = Path(out_dir) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, ten in tensors.items():
        path = snap_dir / f"{name}.pt"
        torch.save(ten.detach().cpu(), path)
        out[name] = str(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out_dir", type=str, default="runs/paper_figures")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--edge_method", type=str, default="", choices=["", "jacobian", "ig", "cycle_jacobian"])
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--has_gold_grn", action="store_true")
    args = ap.parse_args()

    setup_logging(args.out_dir)
    try:
        cfg_yaml = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        device = torch.device(cfg_yaml["device"])
        set_global_determinism(int(cfg_yaml["seed"]))

        ckpt_path = _find_best_ckpt(cfg_yaml, args.ckpt)
        ckpt_obj = torch.load(ckpt_path, map_location=device) if ckpt_path and os.path.exists(ckpt_path) else {}
        cfg = _prepare_runtime_cfg(cfg_yaml, ckpt_obj)

        data_cfg = dict(cfg["data"])
        data_cfg["seed"] = int(cfg_yaml["seed"])
        dataset = load_dataset_from_config(data_cfg)
        snap0, snap1 = dataset.snapshots[0], dataset.snapshots[1]

        model = build_model(cfg, dataset).to(device)
        ckpt_load_info = {"missing": [], "unexpected": [], "shape_mismatches": [], "loaded_keys": 0}
        if ckpt_path and os.path.exists(ckpt_path):
            matches, mismatches = _score_ckpt_compatibility(model, ckpt_obj)
            if mismatches > 0:
                logger.warning("Requested checkpoint has %d shape mismatches; trying exact-compatible fallback checkpoint.", mismatches)
                alt_paths = [Path(cfg.get("train", {}).get("save_dir", "runs")) / "model.pt", Path(cfg.get("train", {}).get("save_dir", "runs")) / "model_best.pt"]
                chosen = None
                chosen_obj = None
                for ap in alt_paths:
                    if str(ap) == str(ckpt_path):
                        continue
                    ap_abs = ap if ap.is_absolute() else Path(ap)
                    if not ap_abs.exists():
                        continue
                    try:
                        obj = torch.load(ap_abs, map_location=device)
                        m2, mm2 = _score_ckpt_compatibility(model, obj)
                        if mm2 == 0 and m2 > 0:
                            chosen = ap_abs
                            chosen_obj = obj
                            break
                    except Exception:
                        pass
                if chosen is not None:
                    ckpt_path = str(chosen)
                    ckpt_obj = chosen_obj
                    logger.info("Using exact-compatible fallback checkpoint: %s", ckpt_path)
            ckpt_load_info = _load_checkpoint_partial(model, ckpt_obj)
            if ckpt_load_info["shape_mismatches"]:
                logger.warning("Checkpoint still has shape mismatches (%d); results may be biased. Prefer exact-compatible training/eval.", len(ckpt_load_info["shape_mismatches"]))
                for name, shp_ckpt, shp_model in ckpt_load_info["shape_mismatches"][:20]:
                    logger.warning("SHAPE MISMATCH %s ckpt=%s model=%s", name, shp_ckpt, shp_model)
            logger.info("Loaded ckpt: %s", ckpt_path)
        else:
            logger.warning("Checkpoint not found. Running with random init.")

        trainer = Trainer(
            model=model,
            solver_cfg=SolverCfg(**cfg["solver"]),
            ot_cfg=_build_ot_cfg(cfg),
            ood_cfg=OODCfg(**cfg["ood"]),
            gate_l1=float(cfg["model"]["gate_l1"]),
            gate_tv=float(cfg["model"]["gate_tv"]),
            enable_ot=bool(cfg["model"].get("enable_ot", False)),
            device=device,
            mms_target_frac_stage1=float(cfg.get("model", {}).get("mms_target_frac_stage1", 0.18) or 0.18),
            mms_target_frac_stage2=float(cfg.get("model", {}).get("mms_target_frac_stage2", 0.12) or 0.12),
        )
        _prime_trainer_envelope_stats(trainer, dataset, device)

        ode_in = trainer._build_ode_input(snap0, stage="stage2")

        plot_errors = []
        x0 = snap0.X.to(device)
        y0 = model.encode(x0)
        err = _best_effort_plot_tcell(model, ode_in, y0, args.out_dir)
        if err:
            logger.warning("plot_tcell_infiltration_flow failed: %s", err)
            plot_errors.append({"plot": "plot_tcell_infiltration_flow", "error": err})

        gate = model.odefunc.compute_gate(ode_in, deterministic=True, apply_mask=False)
        gate_overlay_paths = []
        try:
            gate_overlay_paths.append(plot_mechanism_drift_heatmap(ode_in.coords, gate, out_dir=args.out_dir))
        except Exception as e:
            logger.warning("plot_mechanism_drift_heatmap failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "plot_mechanism_drift_heatmap", "error": f"{type(e).__name__}: {e}"})
        try:
            tgfb_idx = 7 if ode_in.zE.shape[1] > 7 else 0
            cxcl12_idx = 9 if ode_in.zE.shape[1] > 9 else (1 if ode_in.zE.shape[1] > 1 else 0)
            gate_overlay_paths.append(plot_gate_signal_overlay(ode_in.coords, gate, ode_in.zE[:, tgfb_idx], args.out_dir, fname="gate_overlay_tgfb_gradient.png", title="Gate vs TGF-β proxy gradient", signal_label=f"zE[{tgfb_idx}] proxy"))
            z_sig = ode_in.zE[:, cxcl12_idx]
            gate_overlay_paths.append(plot_gate_signal_overlay(ode_in.coords, gate, z_sig, args.out_dir, fname="gate_overlay_cxcl12_gradient.png", title="Gate vs CXCL12 proxy gradient", signal_label=f"zE[{cxcl12_idx}] proxy"))
        except Exception as e:
            logger.warning("plot_gate_signal_overlay failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "plot_gate_signal_overlay", "error": f"{type(e).__name__}: {e}"})

        ann = getattr(snap0, "annotations", None)
        roi = define_interface_roi_adaptive(ode_in.coords.detach().cpu(), annotations=(ann.detach().cpu() if ann is not None else None), target_interface_frac=0.22, band_frac=0.055, min_band=18.0, max_band=110.0)
        roi_mask = roi.mask_interface.to(device).reshape(-1, 1).float()

        roi_drop_scales = cfg.get("eval", {}).get("intervention", {}).get("roi_drop_scales", [0.75, 1.10, 0.75])
        if not isinstance(roi_drop_scales, (list, tuple)):
            roi_drop_scales = [0.55, 0.85, 0.55]
        cf_intervention_scale = float(cfg.get("eval", {}).get("intervention", {}).get("scale", 1.95) or 1.95)
        do_zE = trainer.make_targeted_do_zE(ode_in, roi_mask, tuple(roi_drop_scales), extra_scale=cf_intervention_scale, roi=roi)

        t_end = float(snap1.time - snap0.time)
        out_learned = _safe_rollout(trainer, snap0, t_end=t_end, do_zE=None)
        yT = out_learned.get("yT_learned", model.encode(out_learned["xT_learned"]))
        out_do = _safe_rollout(trainer, snap0, t_end=t_end, do_zE=do_zE)
        yT_do = out_do.get("yT_learned", model.encode(out_do["xT_learned"]))
        try:
            plot_attractor_shift(yT, yT_do, ode_in.coords, out_dir=args.out_dir)
        except Exception as e:
            logger.warning("plot_attractor_shift failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "plot_attractor_shift", "error": f"{type(e).__name__}: {e}"})

        p0 = p_cross_interface(yT.detach().cpu(), roi).item()
        p1 = p_cross_interface(yT_do.detach().cpu(), roi).item()
        gate_p_nonzero = getattr(model.odefunc, "gate_p_nonzero_last", None)
        gate_metrics = _gate_task_metrics(gate.detach().cpu(), roi, gate_p_nonzero=(gate_p_nonzero.detach().cpu() if isinstance(gate_p_nonzero, torch.Tensor) else None))
        infiltration_base = _proxy_infiltration_depth(yT.detach(), roi, ode_in.coords.detach())
        infiltration_do = _proxy_infiltration_depth(yT_do.detach(), roi, ode_in.coords.detach())
        interface_density_shift = _proxy_interface_density_shift(yT.detach(), yT_do.detach(), roi)
        rollout_snapshot_paths = _save_rollout_snapshots(args.out_dir, yT_learned=yT, yT_do=yT_do, yT_envelope=out_do.get("yT_envelope", yT_do), xT_learned=out_learned.get("xT_learned", x0), xT_do=out_do.get("xT_learned", x0), xT_envelope=out_do.get("xT_envelope", x0))
        uncertainty_signal = (out_do.get("xT_learned", x0) - out_do.get("xT_envelope", x0)).detach().abs().mean(dim=1)
        coverage_signal = 1.0 / (1.0 + out_do.get("ood_score", torch.zeros(ode_in.coords.shape[0], device=device)).detach().float())
        gif_frames = []
        try:
            gif_frames.append(plot_gate_signal_overlay(ode_in.coords, gate, coverage_signal, args.out_dir, fname="coverage_overlay.png", title="Coverage proxy", signal_label="coverage"))
            gif_frames.append(plot_gate_signal_overlay(ode_in.coords, gate, uncertainty_signal, args.out_dir, fname="uncertainty_overlay.png", title="Uncertainty proxy", signal_label="|learned-envelope|"))
        except Exception as e:
            logger.warning("coverage/uncertainty overlay failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "coverage_uncertainty_overlay", "error": f"{type(e).__name__}: {e}"})
        gif_frames = [p for p in (gate_overlay_paths + gif_frames) if p]
        gif_path = write_animation_gif(gif_frames, os.path.join(args.out_dir, "gate_coverage_uncertainty.gif"))
        coverage_risk_interface = _build_coverage_risk(out_do.get("ood_score", torch.zeros(ode_in.coords.shape[0], device=device)), out_do.get("xT_learned", x0), out_do.get("xT_envelope", x0), roi.mask_interface.to(device))
        coverage_risk_non_interface = _build_coverage_risk(out_do.get("ood_score", torch.zeros(ode_in.coords.shape[0], device=device)), out_do.get("xT_learned", x0), out_do.get("xT_envelope", x0), (~roi.mask_interface).to(device))
        try:
            if coverage_risk_interface.get("coverage"):
                plot_coverage_risk_curve(coverage_risk_interface["coverage"], coverage_risk_interface["risk"], args.out_dir, fname="coverage_risk_interface.png", title="Coverage-Risk (interface)")
            if coverage_risk_non_interface.get("coverage"):
                plot_coverage_risk_curve(coverage_risk_non_interface["coverage"], coverage_risk_non_interface["risk"], args.out_dir, fname="coverage_risk_non_interface.png", title="Coverage-Risk (non-interface)")
        except Exception as e:
            logger.warning("plot_coverage_risk_curve failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "plot_coverage_risk_curve", "error": f"{type(e).__name__}: {e}"})
        ot_pairing_metrics = _maybe_ot_pairing_metrics(trainer, snap0, snap1, yT, cfg)
        misalignment_curve = _misalignment_robustness_curve(trainer, snap0, snap1, yT, cfg)
        try:
            plot_misalignment_robustness_curve(misalignment_curve["deltas_um"], misalignment_curve["metrics"], args.out_dir)
        except Exception as e:
            logger.warning("plot_misalignment_robustness_curve failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "plot_misalignment_robustness_curve", "error": f"{type(e).__name__}: {e}"})

        save_dir = str(cfg.get("train", {}).get("save_dir", Path(ckpt_path).parent if ckpt_path else ""))
        edges = _load_exported_edges(save_dir, device, int(args.top_k))
        edge_method = str(args.edge_method or cfg.get("grn_export", {}).get("method", "cycle_jacobian") or "cycle_jacobian")
        if edges is None:
            e_cfg = EdgeInferenceCfg(
                method=edge_method,
                top_k=int(args.top_k),
                abs_score=bool(cfg.get("grn_export", {}).get("abs_score", True)),
                ig_steps=int(cfg.get("grn_export", {}).get("ig_steps", 32) or 32),
                cycle_max_cells=int(cfg.get("grn_export", {}).get("cycle_max_cells", 32) or 32),
                n_subsamples=int(cfg.get("grn_export", {}).get("n_subsamples", 1) or 1),
                subsample_agg=str(cfg.get("grn_export", {}).get("subsample_agg", "mean") or "mean"),
                score_norm=str(cfg.get("grn_export", {}).get("score_norm", "none") or "none"),
            )
            edges = infer_edges(model.decoder, y0, e_cfg, encoder=model.encoder, x0=x0)
        else:
            logger.info("Using exported sparse edges from save_dir for auxiliary evaluation: %s", save_dir)
        E = int(edges.scores.shape[0])

        labels = None
        diag = None
        gold_path = getattr(snap0, "gold_grn_path", None)
        gene_names = snap0.gene_names if getattr(snap0, "gene_names", None) is not None else tuple(f"G{i}" for i in range(int(cfg["data"]["num_genes"])))
        gold_health_path = os.path.join(args.out_dir, "gold_health_summary.json")
        gold_health = inspect_gold_file(str(gold_path), present_genes=gene_names, min_overlap_edges=20) if gold_path is not None and os.path.exists(str(gold_path)) else {"ok": False, "reason": "missing"}
        if gold_path is not None and os.path.exists(str(gold_path)):
            sanitized_gold_path, gold_health = drop_bad_tokens(str(gold_path), present_genes=gene_names, summary_path=gold_health_path)
        if gold_path is not None and os.path.exists(str(gold_path)) and bool(gold_health.get("ok", False)):
            from phycausal_stgrn.utils.edge_eval import load_gold_edges_csv, align_gold_to_pred_edges, diagnose_gold_overlap, evaluate_edges
            gold_edges = load_gold_edges_csv(str(gold_path), normalize="upper", present_genes=gene_names, summary_path=gold_health_path)
            diag = diagnose_gold_overlap(gene_names, gold_edges, normalize="upper")
            labels = align_gold_to_pred_edges(edges.edge_index.to(device), gene_names, gold_edges, normalize="upper").to(edges.scores.device)
            metrics = evaluate_edges(edges.scores, labels, precision_ks=(50, 100, 200))
            logger.info("Gold overlap diag: %s", diag)
        else:
            labels = torch.zeros(E, device=edges.scores.device)
            labels[: max(1, E // 100)] = 1.0
            metrics = {"aupr": float(aupr_edge(edges.scores, labels)), "auroc": None, "n_pos": float(labels.sum().item()), "n_total": float(E), "note": "gold_disabled_for_main_claim"}
            _write_json(gold_health_path, gold_health)

        prec, rec, thr = precision_recall_curve(edges.scores, labels)
        csv_path = export_pr_curve_csv(edges.scores, labels, os.path.join(args.out_dir, "pr_curve.csv"))
        try:
            plot_pr_curve(prec, rec, out_dir=args.out_dir)
        except Exception as e:
            logger.warning("plot_pr_curve failed: %s: %s", type(e).__name__, e)
            plot_errors.append({"plot": "plot_pr_curve", "error": f"{type(e).__name__}: {e}"})

        component_switches = cfg.get("component_switches", {}) or {}
        preferred_eval_ckpt = None
        save_dir_probe = str(cfg.get("train", {}).get("save_dir", "") or "")
        if save_dir_probe:
            sp = Path(save_dir_probe)
            if not sp.is_absolute():
                sp = PROJECT_ROOT / sp
            mf = sp / "training_manifest.json"
            if mf.exists():
                try:
                    preferred_eval_ckpt = json.loads(mf.read_text(encoding="utf-8")).get("preferred_eval_ckpt")
                except Exception:
                    preferred_eval_ckpt = None
        summary = {
            "patch_version": _get_patch_version(args.config),
            "component_switches": {
                "enable_ot_pairing": bool(component_switches.get("enable_ot_pairing", cfg.get("model", {}).get("enable_ot", False))),
                "enable_dynamic_anchor_pairing": bool(component_switches.get("enable_dynamic_anchor_pairing", False)),
            },
            "task_primary": {
                "delta_p_cross": float(p1 - p0),
                "p_cross_baseline": float(p0),
                "p_cross_do_z": float(p1),
                "p_cross_envelope_do_z": float(p_cross_interface(out_do.get("yT_envelope", yT_do).detach().cpu(), roi).item()) if out_do.get("yT_envelope", None) is not None else None,
                "ood_flag_do_z": float(out_do.get("ood_flag", torch.tensor(0.0)).item()),
                "roi_source": roi.source,
                "roi_x0": roi.x0,
                "roi_band": roi.band,
                "residual_gate_summary": getattr(trainer, "_residual_gate_summary", {}),
                **gate_metrics,
            },
            "task_secondary": {
                "edge_metrics": metrics,
                "gold_health": gold_health,
                "gold_overlap_diag": diag,
                "pr_curve_csv": csv_path,
                "edge_method": getattr(edges, "method", edge_method),
                "num_edges_scored": int(E),
                "ot_pairing": ot_pairing_metrics,
                "misalignment_curve": misalignment_curve,
                "w1_emd_alignment": {"w1": ot_pairing_metrics.get("w1"), "emd": ot_pairing_metrics.get("emd"), "w2": ot_pairing_metrics.get("w2")},
                "coverage_risk": {"interface": coverage_risk_interface, "non_interface": coverage_risk_non_interface},
                "envelope_info": out_do.get("envelope_info", {}),
            },
            "summary_stats": {
                "delta_p_cross": float(p1 - p0),
                "delta_infiltration_depth": float(infiltration_do - infiltration_base),
                "interface_immune_density_shift": float(interface_density_shift),
                "gate_active_fraction": gate_metrics.get("gate_active_fraction"),
                "gate_sparsity": gate_metrics.get("gate_sparsity"),
            },
            "artifacts": {
                "gif": gif_path,
                "rollout_snapshots": rollout_snapshot_paths,
                "gold_health_summary": gold_health_path,
            },
            "ckpt_used": ckpt_path,
            "preferred_eval_ckpt": preferred_eval_ckpt,
            "ckpt_load_info": ckpt_load_info,
            "plot_errors": plot_errors,
        }
        Path(args.out_dir, "paper_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        counterfactual_summary = {
            "patch_version": _get_patch_version(args.config),
            "component_switches": summary.get("component_switches", {}),
            "delta_p_cross": summary["task_primary"].get("delta_p_cross"),
            "p_cross_baseline": summary["task_primary"].get("p_cross_baseline"),
            "p_cross_do_z": summary["task_primary"].get("p_cross_do_z"),
            "p_cross_envelope_do_z": summary["task_primary"].get("p_cross_envelope_do_z"),
            "ood_flag_do_z": summary["task_primary"].get("ood_flag_do_z"),
            "roi_source": summary["task_primary"].get("roi_source"),
            "roi_x0": summary["task_primary"].get("roi_x0"),
            "roi_band": summary["task_primary"].get("roi_band"),
        }
        gate_summary = {
            "patch_version": _get_patch_version(args.config),
            "component_switches": summary.get("component_switches", {}),
            "gate_mean_interface": summary["task_primary"].get("gate_mean_interface"),
            "gate_mean_non_interface": summary["task_primary"].get("gate_mean_non_interface"),
            "gate_interface_enrichment": summary["task_primary"].get("gate_interface_enrichment"),
            "gate_activity_mean": summary["task_primary"].get("gate_activity_mean"),
            "gate_active_fraction": summary["task_primary"].get("gate_active_fraction"),
            "gate_sparsity": summary["task_primary"].get("gate_sparsity"),
            "roi_interface_fraction": summary["task_primary"].get("roi_interface_fraction"),
            "gate_num_cells": summary["task_primary"].get("gate_num_cells"),
        }
        Path(args.out_dir, "counterfactual_summary.json").write_text(json.dumps(counterfactual_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        Path(args.out_dir, "gate_summary.json").write_text(json.dumps(gate_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.exception("paper_figures.py failed")
        err = {"patch_version": _get_patch_version(args.config), "error_type": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()}
        Path(args.out_dir, "paper_error.json").write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
