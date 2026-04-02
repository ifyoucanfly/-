from __future__ import annotations

# PATH_IMMUNE_BOOTSTRAP
# NOTE: Not touching any algorithmic logic; only hardening import/path context.
import sys, os
# If launched from notebooks/, add repo root to PYTHONPATH; otherwise keep cwd on sys.path
if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
else:
    sys.path.insert(0, os.path.abspath(os.getcwd()))


import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml

from phycausal_stgrn.data.data_loader import load_dataset_from_config, SpatialSnapshot, knn_graph
from phycausal_stgrn.engine.ot_alignment import OTCfg
from phycausal_stgrn.engine.trainer import OODCfg, SolverCfg, TrainCfg, Trainer
from phycausal_stgrn.models.PhyCausal_ODE import PhyCausalSTGRN, ODEInput
from phycausal_stgrn.utils.diagnostics import set_global_determinism
from phycausal_stgrn.utils.metrics import pearson_corr, wasserstein2_cost
from phycausal_stgrn.viz.money_figure import plot_tcell_infiltration_flow

logger = logging.getLogger(__name__)


def _setup_logging(save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "robustness.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def _tps_warp_2d(
    coords: np.ndarray,
    ctrl_pts: np.ndarray,
    ctrl_disp: np.ndarray,
    reg: float = 1e-3,
) -> np.ndarray:
    """Thin-Plate Spline (TPS) warp in 2D.

    coords: (N,2) points to warp
    ctrl_pts: (K,2) control points
    ctrl_disp: (K,2) displacement vectors at control points
    reg: small regularization for numerical stability
    """
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert ctrl_pts.ndim == 2 and ctrl_pts.shape[1] == 2
    assert ctrl_disp.shape == ctrl_pts.shape

    N = coords.shape[0]
    K = ctrl_pts.shape[0]

    # radial basis U(r) = r^2 log r (with r=0 -> 0)
    def U(r2: np.ndarray) -> np.ndarray:
        # r2 = r^2
        r = np.sqrt(np.maximum(r2, 1e-12))
        return (r2) * np.log(r + 1e-12)

    # Build K matrix (K,K)
    diff = ctrl_pts[:, None, :] - ctrl_pts[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    Kmat = U(r2)
    # Regularize diagonal
    Kmat = Kmat + reg * np.eye(K, dtype=np.float32)

    # Build P matrix (K,3): [1, x, y]
    P = np.concatenate([np.ones((K, 1), dtype=np.float32), ctrl_pts.astype(np.float32)], axis=1)

    # Assemble linear system:
    # [K  P] [w] = [d]
    # [P^T 0] [a]   [0]
    A = np.zeros((K + 3, K + 3), dtype=np.float32)
    A[:K, :K] = Kmat
    A[:K, K:] = P
    A[K:, :K] = P.T

    Y = np.zeros((K + 3, 2), dtype=np.float32)
    Y[:K, :] = ctrl_disp.astype(np.float32)

    params = np.linalg.solve(A, Y)
    w = params[:K, :]   # (K,2)
    a = params[K:, :]   # (3,2)

    # Warp all coords
    diff_q = coords[:, None, :] - ctrl_pts[None, :, :]  # (N,K,2)
    r2_q = np.sum(diff_q * diff_q, axis=2)              # (N,K)
    U_q = U(r2_q).astype(np.float32)                    # (N,K)

    P_q = np.concatenate([np.ones((N, 1), dtype=np.float32), coords.astype(np.float32)], axis=1)  # (N,3)
    disp = P_q @ a + U_q @ w  # (N,2)
    return coords + disp


def _inject_coord_noise(
    coords: torch.Tensor,
    delta_um: float,
    seed: int = 0,
    mode: str = "translation",
    tps_ctrl: int = 16,
    tps_reg: float = 1e-3,
) -> torch.Tensor:
    """Coordinate perturbation for robustness curves.

    mode:
      - "translation": isotropic random translation direction per point
      - "tps": thin-plate spline elastic warp (nonlinear local distortions)
    """
    rng = np.random.default_rng(seed)
    c = coords.detach().cpu().numpy().astype(np.float32)

    if mode == "translation":
        noise = rng.normal(size=c.shape).astype(np.float32)
        noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-8)
        pert = c + noise * float(delta_um)
        return torch.from_numpy(pert).to(coords.device)

    if mode == "tps":
        n = c.shape[0]
        K = int(min(max(4, tps_ctrl), n))
        # choose control points uniformly at random
        idx = rng.choice(n, size=K, replace=False)
        ctrl_pts = c[idx]
        # random displacements at control points with magnitude ~ delta_um
        disp = rng.normal(size=ctrl_pts.shape).astype(np.float32)
        disp = disp / (np.linalg.norm(disp, axis=1, keepdims=True) + 1e-8)
        disp = disp * float(delta_um)
        warped = _tps_warp_2d(c, ctrl_pts=ctrl_pts, ctrl_disp=disp, reg=float(tps_reg))
        return torch.from_numpy(warped.astype(np.float32)).to(coords.device)

    raise ValueError(f"Unknown noise mode: {mode}")


def _compute_boundary_interior_masks(
    model: PhyCausalSTGRN,
    snap: SpatialSnapshot,
    device: torch.device,
    boundary_q: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute boundary vs interior masks based on gate spatial gradients.

    We define a per-node gate score s_i = mean(|Gate_i|).
    Boundary nodes have high local variation (graph TV) of s across neighbors.
    """
    model.eval()
    ode_in = ODEInput(
        edge_index=snap.edge_index.to(device),
        coords=snap.coords.to(device),
        zI=snap.zI.to(device),
        zN=snap.zN.to(device),
        zE=snap.zE.to(device),
        mechanomics=snap.mechanomics.to(device),
        mass=None,
    )
    with torch.no_grad():
        gate = model.odefunc.compute_gate(ode_in)  # (N,d)
        score = gate.abs().mean(dim=1)             # (N,)

    edge = snap.edge_index.to(device)
    i, j = edge[0], edge[1]
    diff = (score[i] - score[j]).abs()

    tv = torch.zeros_like(score)
    deg = torch.zeros_like(score)
    tv.index_add_(0, i, diff)
    deg.index_add_(0, i, torch.ones_like(diff))
    tv = tv / deg.clamp_min(1.0)

    # boundary: top (1-boundary_q) fraction
    thr = torch.quantile(tv, boundary_q)
    boundary = tv >= thr
    interior = ~boundary
    return boundary.detach().cpu(), interior.detach().cpu()


def _metrics_on_mask(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: Optional[torch.Tensor],
    w2_p: int,
) -> Tuple[float, float]:
    if mask is None:
        pear = pearson_corr(x_pred.cpu(), x_true.cpu())
        w2 = wasserstein2_cost(x_pred.cpu(), x_true.cpu(), p=w2_p)
        return float(pear), float(w2)

    m = mask.to(x_pred.device)
    xp = x_pred[m]
    xt = x_true[m]
    pear = pearson_corr(xp.cpu(), xt.cpu())
    w2 = wasserstein2_cost(xp.cpu(), xt.cpu(), p=w2_p)
    return float(pear), float(w2)


def run_ablation_suite(cfg: Dict) -> None:
    device = torch.device(cfg["device"])
    set_global_determinism(int(cfg["seed"]))

    data_cfg = dict(cfg["data"])
    data_cfg["seed"] = int(cfg["seed"])
    dataset = load_dataset_from_config(data_cfg)
    snap0, snap1 = dataset.snapshots[0], dataset.snapshots[1]

    solver_cfg = SolverCfg(**cfg["solver"])
    ot_cfg = OTCfg(**cfg["ot"])
    ood_cfg = OODCfg(**cfg["ood"])
    train_cfg = TrainCfg(**cfg["train"])

    base_kwargs = dict(
        num_genes=int(cfg["data"]["num_genes"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_gnn_layers=int(cfg["model"]["num_layers"]),
        zI_dim=snap0.zI.shape[1],
        zN_dim=snap0.zN.shape[1],
        zE_dim=snap0.zE.shape[1],
        mech_dim=snap0.mechanomics.shape[1],
    )

    variants = [
        ("BaseODE", dict(enable_mms=False, enable_mechanomics=False, upwind_drift=False), False),
        ("MMS-ODE", dict(enable_mms=True, enable_mechanomics=False, upwind_drift=False), False),
        ("OT-ODE", dict(enable_mms=True, enable_mechanomics=True, upwind_drift=bool(cfg["model"]["upwind_drift"])), True),
    ]

    results: List[Tuple[str, str, float, float]] = []
    w2_p = int(cfg["ot"].get("w1_p", 2))

    for name, model_flags, enable_ot in variants:
        save_dir = os.path.join(train_cfg.save_dir, f"ablation_{name}")
        _setup_logging(save_dir)
        logger.info("=== Running variant: %s ===", name)

        model = PhyCausalSTGRN(**base_kwargs, **model_flags).to(device)
        trainer = Trainer(
            model=model,
            solver_cfg=solver_cfg,
            ot_cfg=ot_cfg,
            ood_cfg=ood_cfg,
            gate_l1=float(cfg["model"]["gate_l1"]),
            gate_tv=float(cfg["model"]["gate_tv"]),
            enable_ot=enable_ot,
            device=device,
            irm_lambda=float(cfg["model"].get("irm_lambda", 0.0)),
            irm_split=str(cfg["model"].get("irm_split", "x_median")),
            irm_warmup_epochs=int(cfg["model"].get("irm_warmup_epochs", 0)),
            irm_balance=bool(cfg["model"].get("irm_balance", True)),
        )
        trainer.fit(dataset, TrainCfg(**{**cfg["train"], "save_dir": save_dir}))

        # compute boundary/interior masks on snap0 (based on spatial gate gradients)
        boundary_mask, interior_mask = _compute_boundary_interior_masks(model, snap0, device=device, boundary_q=0.8)

        # evaluate on clean coords (overall + boundary + interior)
        out = trainer.rollout_with_envelope(snap0, t_end=float(snap1.time - snap0.time), do_zE=None)
        pear, w2 = _metrics_on_mask(out["xT_learned"], snap1.X, None, w2_p=w2_p)
        pear_b, w2_b = _metrics_on_mask(out["xT_learned"], snap1.X, boundary_mask, w2_p=w2_p)
        pear_i, w2_i = _metrics_on_mask(out["xT_learned"], snap1.X, interior_mask, w2_p=w2_p)

        results.append((name, "clean/all", pear, w2))
        results.append((name, "clean/boundary", pear_b, w2_b))
        results.append((name, "clean/interior", pear_i, w2_i))

        # robustness: inject coordinate noise (translation + TPS elastic)
        for delta in [5.0, 10.0, 50.0]:
            for mode in ["translation", "tps"]:
                coords_n = _inject_coord_noise(
                    snap0.coords,
                    delta_um=delta,
                    seed=int(cfg["seed"]) + int(delta) + (7 if mode == "tps" else 0),
                    mode=mode,
                    tps_ctrl=int(cfg["data"].get("tps_ctrl", 16)),
                    tps_reg=float(cfg["data"].get("tps_reg", 1e-3)),
                )
                edge_n = knn_graph(coords_n, k=int(cfg["data"].get("k_nn", 12)))
                snap0n = SpatialSnapshot(
                    X=snap0.X,
                    coords=coords_n,
                    zI=snap0.zI,
                    zN=snap0.zN,
                    zE=snap0.zE,
                    mechanomics=snap0.mechanomics,
                    edge_index=edge_n,
                    time=snap0.time,
                    annotations=getattr(snap0, 'annotations', None),
                )
                outn = trainer.rollout_with_envelope(snap0n, t_end=float(snap1.time - snap0.time), do_zE=None)
                pear_n, w2_n = _metrics_on_mask(outn["xT_learned"], snap1.X, None, w2_p=w2_p)
                pear_nb, w2_nb = _metrics_on_mask(outn["xT_learned"], snap1.X, boundary_mask, w2_p=w2_p)
                pear_ni, w2_ni = _metrics_on_mask(outn["xT_learned"], snap1.X, interior_mask, w2_p=w2_p)

                tag = f"{mode}/delta={delta}um"
                results.append((name, f"{tag}/all", pear_n, w2_n))
                results.append((name, f"{tag}/boundary", pear_nb, w2_nb))
                results.append((name, f"{tag}/interior", pear_ni, w2_ni))

        # money figure + Jacobian bifurcation diagnostics (interface proxy band)
        ode_in = ODEInput(
            edge_index=snap0.edge_index.to(device),
            coords=snap0.coords.to(device),
            zI=snap0.zI.to(device),
            zN=snap0.zN.to(device),
            zE=snap0.zE.to(device),
            mechanomics=snap0.mechanomics.to(device),
            mass=None,
        )
        with torch.no_grad():
            y0 = model.encode(snap0.X.to(device))
        plot_tcell_infiltration_flow(
            model,
            ode_in,
            y0_latent=y0,
            out_dir=os.path.join(save_dir, "figures"),
            title="T-cell infiltration flow field + interface Jacobian",
        )

    summary_path = os.path.join(train_cfg.save_dir, "robustness_summary.json")
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"variant": n, "condition": c, "pearson": p, "wasserstein_cost": w} for (n, c, p, w) in results],
            f,
            indent=2,
        )
    print(f"\nSaved: {summary_path}")

    print("\nAblation + robustness summary (Pearson / Wasserstein cost):")
    for name, cond, pear, w2 in results:
        print(f"{name:8s} | {cond:22s} | pear={pear:.4f} | W={w2:.4f}")

    print("\nBaseline hooks:")
    print("- Implement `baselines/celcomen_runner.py` and `baselines/spagrn_runner.py` to output predictions.")
    print("- Then compute Pearson/Wasserstein/AUPR using `utils/metrics.py` (TODO placeholder).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_ablation_suite(cfg)


if __name__ == "__main__":
    main()
