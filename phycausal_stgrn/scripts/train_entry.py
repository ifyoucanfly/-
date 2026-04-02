from __future__ import annotations

import logging
import math
import os as _os
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Type

import numpy as np
import torch

from phycausal_stgrn.data.data_loader import load_dataset_from_config
from phycausal_stgrn.engine.ot_alignment import OTCfg
from phycausal_stgrn.engine.trainer import OODCfg, SolverCfg, TrainCfg, Trainer
from phycausal_stgrn.models.PhyCausal_ODE import PhyCausalSTGRN
from phycausal_stgrn.utils.diagnostics import set_global_determinism

logger = logging.getLogger(__name__)


def _filter_kwargs_for_dataclass(cls: Type, d: Dict[str, Any] | None, *, ctx: str = "") -> Dict[str, Any]:
    """Filter config kwargs with backward-compatible alias handling.

    In addition to dropping unknown keys, this function maps a few legacy names
    used by older YAMLs / notebooks to the current dataclass field names so that
    training does not fail on config drift.
    """
    if d is None:
        d = {}
    else:
        d = dict(d)
    if not is_dataclass(cls):
        return dict(d)

    # ---- Backward compatibility shims ----
    if cls.__name__ == "OODCfg":
        # Legacy key used in older configs.
        if "zscore_thresh" in d and "z_score_threshold" not in d:
            d["z_score_threshold"] = d.pop("zscore_thresh")
        # Legacy fallback mode enum -> current boolean flags.
        if "fallback_mode" in d:
            mode = str(d.pop("fallback_mode") or "").strip().lower()
            if "fallback_disable_mod" not in d:
                d["fallback_disable_mod"] = mode in {"disable_mod", "disable_both", "both", "mod", "all", "true"}
            if "fallback_disable_drift" not in d:
                d["fallback_disable_drift"] = mode in {"disable_drift", "disable_both", "both", "drift", "all", "true"}
        # Safe defaults so OODCfg is always constructible.
        d.setdefault("z_score_threshold", 3.0)
        d.setdefault("fallback_disable_mod", True)
        d.setdefault("fallback_disable_drift", True)

    if cls.__name__ == "OTCfg":
        if "reg_eps" in d and "epsilon" not in d:
            d["epsilon"] = d.pop("reg_eps")
        if "sinkhorn_iter" in d and "max_iter" not in d:
            d["max_iter"] = d.pop("sinkhorn_iter")
        # legacy enable flag is handled at model level; ignore here if present
        d.pop("enabled", None)

    if cls.__name__ == "TrainCfg":
        # Some older notebooks / YAMLs omitted these keys and relied on code defaults.
        d.setdefault("weight_decay", 1.0e-6)
        d.setdefault("grad_clip", 1.0)
        # Legacy / relocated knobs that are consumed from cfg["model"] or elsewhere.
        d.pop("batch_size", None)
        d.pop("gate_l1", None)
        d.pop("gate_tv", None)
        d.pop("mms_target_frac_stage1", None)
        d.pop("mms_target_frac_stage2", None)

    allowed = {f.name for f in fields(cls)}
    out = {k: v for k, v in d.items() if k in allowed}
    dropped = sorted([k for k in d.keys() if k not in allowed])
    if dropped:
        prefix = f"[Config] {ctx}: " if ctx else "[Config] "
        print(prefix + f"dropped unexpected keys for {cls.__name__}: {dropped}")
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



def export_grn_edges(cfg: Dict, dataset, model: PhyCausalSTGRN, save_dir: str, device: torch.device) -> Dict[str, str]:
    """Export sparse GRN edges for the current trained model.

    Two stability fixes are applied here:
    1) export uses a pooled cell set across snapshots instead of only snapshot-0;
    2) raw Jacobian scores are preserved, and a monotonic calibrated score is exported
       for downstream CSV / dense-adjacency consumers that were previously seeing a
       near-flat 1e-6 score range.
    """
    grn_cfg = dict(cfg.get("grn_export", {}) or {})
    enabled = bool(grn_cfg.get("enabled", True))
    if not enabled:
        logger.info("GRN export disabled in config.")
        return {}

    from phycausal_stgrn.grn.infer_edges import EdgeInferenceCfg, infer_edges

    method = str(grn_cfg.get("method", "cycle_jacobian"))
    top_k = int(grn_cfg.get("top_k", 2000))
    abs_score = bool(grn_cfg.get("abs_score", True))
    ig_steps = int(grn_cfg.get("ig_steps", 32))
    cycle_max_cells = int(grn_cfg.get("cycle_max_cells", 32))
    n_subsamples = int(grn_cfg.get("n_subsamples", 1))
    subsample_agg = str(grn_cfg.get("subsample_agg", "mean"))
    export_cells_per_snapshot = int(grn_cfg.get("cells_per_snapshot", max(128, cycle_max_cells)) or max(128, cycle_max_cells))
    score_norm = str(grn_cfg.get("score_norm", "none") or "none")

    snap0 = dataset.snapshots[0] if hasattr(dataset, "snapshots") and len(dataset.snapshots) > 0 else None
    if snap0 is None:
        raise RuntimeError("dataset has no snapshots; cannot export GRN edges.")

    gene_names = None
    for attr in ("gene_names", "genes", "var_names"):
        if hasattr(snap0, attr):
            g = getattr(snap0, attr)
            if g is not None:
                try:
                    gene_names = tuple([str(x) for x in list(g)])
                    break
                except Exception:
                    pass
    if gene_names is None and hasattr(dataset, "gene_names"):
        try:
            gene_names = tuple([str(x) for x in list(getattr(dataset, "gene_names"))])
        except Exception:
            gene_names = None
    if gene_names is None:
        G = int(cfg["data"]["num_genes"])
        gene_names = tuple([f"G{i}" for i in range(G)])

    def _to_tensor_x(x_like) -> torch.Tensor:
        try:
            import scipy.sparse as sp  # type: ignore
        except Exception:
            sp = None
        if sp is not None and sp.issparse(x_like):
            x_like = x_like.toarray()
        if isinstance(x_like, np.ndarray):
            return torch.from_numpy(np.asarray(x_like, dtype=np.float32))
        if torch.is_tensor(x_like):
            return x_like.detach().cpu().float()
        return torch.tensor(x_like, dtype=torch.float32)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(cfg.get("seed", 0) or 0))
    x_blocks = []
    snapshot_ids = []
    for s_idx, snap in enumerate(getattr(dataset, "snapshots", ())):
        Xs = getattr(snap, "X", None)
        if Xs is None:
            continue
        Xt = _to_tensor_x(Xs)
        if Xt.ndim != 2 or Xt.shape[1] != int(len(gene_names)):
            continue
        keep = int(min(export_cells_per_snapshot, Xt.shape[0])) if export_cells_per_snapshot > 0 else int(Xt.shape[0])
        if keep <= 0:
            continue
        if keep < Xt.shape[0]:
            idx = torch.randperm(int(Xt.shape[0]), generator=rng)[:keep]
            Xt = Xt[idx]
        x_blocks.append(Xt)
        snapshot_ids.extend([int(s_idx)] * int(Xt.shape[0]))

    if not x_blocks:
        raise RuntimeError("No snapshot expression matrix available for GRN export.")
    x0 = torch.cat(x_blocks, dim=0).float().to(device)

    model.eval()
    with torch.no_grad():
        y0 = model.encode(x0)

    edge_cfg = EdgeInferenceCfg(
        method=method,
        top_k=top_k,
        abs_score=abs_score,
        ig_steps=ig_steps,
        cycle_max_cells=cycle_max_cells,
        n_subsamples=n_subsamples,
        subsample_agg=subsample_agg,
        score_norm=score_norm,
    )
    edges = infer_edges(model.decoder, y0, cfg=edge_cfg, encoder=model.encoder, x0=x0)

    def _dedup_reciprocal_edges(ei: torch.Tensor, sc: torch.Tensor):
        ei = ei.long()
        sc = sc.float()
        keep = {}
        for k in range(ei.shape[1]):
            s = int(ei[0, k]); t = int(ei[1, k])
            if s == t:
                continue
            a, b = (s, t) if s < t else (t, s)
            key = (a, b)
            val = float(sc[k])
            if key not in keep or val > keep[key][0]:
                keep[key] = (val, s, t)
        scores = torch.tensor([v[0] for v in keep.values()], dtype=sc.dtype, device=sc.device)
        src = torch.tensor([v[1] for v in keep.values()], dtype=torch.long, device=ei.device)
        dst = torch.tensor([v[2] for v in keep.values()], dtype=torch.long, device=ei.device)
        order = torch.argsort(scores, descending=True)
        src = src[order]
        dst = dst[order]
        scores = scores[order]
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, scores

    if bool(grn_cfg.get("dedup_reciprocal", True)) and str(edges.method).endswith("_proxy"):
        ei2, sc2 = _dedup_reciprocal_edges(edges.edge_index, edges.scores)
        edges.edge_index = ei2
        edges.scores = sc2

    def _calibrate_scores(sc: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        raw = sc.detach().cpu().float().clamp_min(0.0)
        nz = raw[raw > 0]
        if int(nz.numel()) == 0:
            return raw, {"raw_median_nonzero": 0.0, "raw_q95_nonzero": 0.0}
        median = float(nz.median().item())
        if int(nz.numel()) >= 16:
            q95 = float(torch.quantile(nz, 0.95).item())
        else:
            q95 = float(nz.max().item())
        scale = max(median, 1e-12)
        cal = torch.log1p(raw / scale)
        denom = max(math.log1p(max(q95, scale) / scale), 1.0)
        cal = (cal / denom).float()
        return cal, {"raw_median_nonzero": median, "raw_q95_nonzero": q95}

    def _edges_to_dense(ei: torch.Tensor, sc: torch.Tensor, G: int) -> np.ndarray:
        adj = np.zeros((int(G), int(G)), dtype=np.float32)
        ei_np = ei.detach().cpu().long().numpy()
        sc_np = sc.detach().cpu().float().numpy()
        for k in range(ei_np.shape[1]):
            s = int(ei_np[0, k])
            t = int(ei_np[1, k])
            if s == t or s < 0 or t < 0 or s >= int(G) or t >= int(G):
                continue
            if float(sc_np[k]) > float(adj[s, t]):
                adj[s, t] = float(sc_np[k])
        np.fill_diagonal(adj, 0.0)
        return adj

    raw_scores = edges.scores.detach().cpu().float()
    cal_scores, cal_meta = _calibrate_scores(raw_scores)
    edges.scores = cal_scores.to(edges.edge_index.device)

    genes_path = _os.path.join(save_dir, f"genes_{len(gene_names)}.txt")
    with open(genes_path, "w", encoding="utf-8") as f:
        for g in gene_names:
            f.write(str(g) + "\n")

    ei = edges.edge_index.detach().cpu().long()
    raw_sc = raw_scores.cpu().float()
    cal_sc = cal_scores.cpu().float()

    pt_path = _os.path.join(save_dir, "pred_edges.pt")
    pt_raw_path = _os.path.join(save_dir, "pred_edges_raw.pt")
    payload_common = {
        "edge_index": ei,
        "method": edges.method,
        "gene_names": gene_names,
        "num_genes": int(len(gene_names)),
        "snapshot_ids_used": snapshot_ids,
        "score_calibration": cal_meta,
        "score_norm": score_norm,
    }
    torch.save({**payload_common, "scores": cal_sc, "scores_raw": raw_sc}, pt_path)
    torch.save({**payload_common, "scores": raw_sc, "score_view": "raw"}, pt_raw_path)

    csv_path = _os.path.join(save_dir, "pred_edges.csv")
    csv_raw_path = _os.path.join(save_dir, "pred_edges_raw.csv")
    with open(csv_path, "w", encoding="utf-8") as f, open(csv_raw_path, "w", encoding="utf-8") as fraw:
        f.write("src_gene,dst_gene,score,score_raw\n")
        fraw.write("src_gene,dst_gene,score\n")
        for k in range(ei.shape[1]):
            src = gene_names[int(ei[0, k])]
            dst = gene_names[int(ei[1, k])]
            f.write(f"{src},{dst},{float(cal_sc[k]):.9g},{float(raw_sc[k]):.9g}\n")
            fraw.write(f"{src},{dst},{float(raw_sc[k]):.9g}\n")

    adj_raw = _edges_to_dense(ei, raw_sc, len(gene_names))
    adj_cal = _edges_to_dense(ei, cal_sc, len(gene_names))
    pred_adj_raw_path = _os.path.join(save_dir, "pred_adj_raw.npy")
    pred_adj_path = _os.path.join(save_dir, "pred_adj.npy")
    np.save(pred_adj_raw_path, adj_raw.astype(np.float32))
    np.save(pred_adj_path, adj_cal.astype(np.float32))

    meta_path = _os.path.join(save_dir, "pred_edges_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        import json
        json.dump(
            {
                "method": edges.method,
                "num_genes": int(len(gene_names)),
                "num_edges": int(ei.shape[1]),
                "snapshots_used": sorted(list(set(snapshot_ids))),
                "cells_per_snapshot": int(export_cells_per_snapshot),
                "score_calibration": cal_meta,
                "score_norm": score_norm,
                "top_score_raw": float(raw_sc.max().item()) if raw_sc.numel() else 0.0,
                "top_score_calibrated": float(cal_sc.max().item()) if cal_sc.numel() else 0.0,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Exported GRN edges: %s | %s | %s | %s | %s | %s | %s",
        pt_path,
        pt_raw_path,
        csv_path,
        csv_raw_path,
        genes_path,
        pred_adj_raw_path,
        pred_adj_path,
    )
    return {
        "pt_path": pt_path,
        "pt_raw_path": pt_raw_path,
        "csv_path": csv_path,
        "csv_raw_path": csv_raw_path,
        "genes_path": genes_path,
        "pred_adj_raw_path": pred_adj_raw_path,
        "pred_adj_path": pred_adj_path,
        "meta_path": meta_path,
    }

def run(cfg: Dict, out_dir: str = "runs", ckpt_name: str = "model.pt") -> str:
    """Train pipeline entry. Returns checkpoint path."""
    set_global_determinism(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cpu"))

    data_cfg = dict(cfg["data"])
    data_cfg["seed"] = int(cfg["seed"])
    dataset = load_dataset_from_config(data_cfg)

    model = build_model(cfg, dataset).to(device)

    trainer = Trainer(
        model=model,
        solver_cfg=SolverCfg(**_filter_kwargs_for_dataclass(SolverCfg, cfg.get("solver", {}), ctx="solver")),
        ot_cfg=OTCfg(**_filter_kwargs_for_dataclass(OTCfg, cfg.get("ot", {}), ctx="ot")),
        ood_cfg=OODCfg(**_filter_kwargs_for_dataclass(OODCfg, cfg.get("ood", {}), ctx="ood")),
        gate_l1=float(cfg["model"]["gate_l1"]),
        gate_tv=float(cfg["model"]["gate_tv"]),
        enable_ot=bool(cfg["model"]["enable_ot"]),
        device=device,
        irm_lambda=float(cfg.get("model", {}).get("irm_lambda", 0.0) or 0.0),
        irm_split=str(cfg.get("model", {}).get("irm_split", "x_median")),
        irm_warmup_epochs=int(cfg.get("model", {}).get("irm_warmup_epochs", 0) or 0),
        irm_balance=bool(cfg.get("model", {}).get("irm_balance", True)),
        balance_recon=bool(cfg.get("model", {}).get("balance_recon", False)),
        mms_target_frac_stage1=float(cfg.get("model", {}).get("mms_target_frac_stage1", 0.18) or 0.18),
        mms_target_frac_stage2=float(cfg.get("model", {}).get("mms_target_frac_stage2", 0.12) or 0.12),
    )

    save_dir = str(cfg.get("train", {}).get("save_dir", out_dir) or out_dir)
    _os.makedirs(save_dir, exist_ok=True)

    train_dict = dict(cfg.get("train", {}) or {})
    train_dict["save_dir"] = save_dir
    train_cfg = TrainCfg(**_filter_kwargs_for_dataclass(TrainCfg, train_dict, ctx="train"))

    # Persist the effective runtime config used to build/train the model so evaluation scripts
    # can reconstruct the exact architecture later (critical for Gate width consistency).
    import copy, json
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("model", {})["gate_hidden"] = int(cfg["model"].get("gate_hidden", cfg["model"]["hidden_dim"]))

    trainer.fit(dataset, cfg=train_cfg)

    ckpt_path = _os.path.join(save_dir, ckpt_name)
    torch.save({"state_dict": model.state_dict(), "config": runtime_cfg}, ckpt_path)
    logger.info("Saved checkpoint: %s", ckpt_path)

    # Prefer exporting edges from the best checkpoint if training produced one.
    export_model = model
    best_ckpt = _os.path.join(save_dir, "model_best.pt")
    if _os.path.exists(best_ckpt):
        try:
            payload = torch.load(best_ckpt, map_location=device)
            state = payload.get("state_dict", payload)
            export_model = build_model(cfg, dataset).to(device)
            export_model.load_state_dict(state, strict=False)
            export_model.eval()
            logger.info("Using best checkpoint for GRN export: %s", best_ckpt)
        except Exception as e:
            logger.exception("Failed to load best checkpoint for export; falling back to final model: %s", e)
            export_model = model

    try:
        export_grn_edges(cfg, dataset, export_model, save_dir, device)
    except Exception as e:
        logger.exception("GRN export failed: %s", e)
        err_path = _os.path.join(save_dir, "pred_edges_error.txt")
        try:
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(str(e) + "\n")
        except Exception:
            pass

    return ckpt_path
