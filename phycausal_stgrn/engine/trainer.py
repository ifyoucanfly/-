from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:  # pragma: no cover - optional at import time
    odeint = None
    odeint_adjoint = None

from phycausal_stgrn.data.data_loader import SpatialDataset, SpatialSnapshot
from phycausal_stgrn.engine.ot_alignment import OTCfg, compute_pi_pot_sparse_knn, ot_alignment_loss
from phycausal_stgrn.models.PhyCausal_ODE import ODEInput, PhyCausalSTGRN
from phycausal_stgrn.utils.metrics import pearson_corr, wasserstein2_cost
from phycausal_stgrn.utils.roi import define_interface_roi_adaptive, p_cross_interface

from phycausal_stgrn.utils.diagnostics import (
    ODEStats,
    check_tensor_finite,
    count_nfe,
    mechanism_drift_heatmap,
    safe_clip_grad_norm_,
    total_variation_on_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainCfg:
    epochs: int
    lr: float
    weight_decay: float
    stage0_epochs: int
    stage1_epochs: int
    stage2_epochs: int
    log_every: int
    grad_clip: float
    save_dir: str
    mms_lr_mult: float = 2.0
    stage0_gate_open_target: float = 0.20
    stage0_gate_open_weight: float = 1.2
    stage0_gate_interface_margin: float = 0.06
    stage0_gate_interface_weight: float = 1.0
    stage0_gate_bce_weight: float = 1.6
    stage0_gate_entropy_weight: float = 0.05
    stage0_gate_l1_scale: float = 0.0
    stage0_gate_tv_scale: float = 0.0
    stage1_gate_l1_scale: float = 0.06
    stage1_gate_tv_scale: float = 0.04
    stage2_gate_l1_scale: float = 0.035
    stage2_gate_tv_scale: float = 0.025
    stage1_gate_open_target: float = 0.06
    stage1_gate_open_weight: float = 0.8
    stage2_gate_open_target: float = 0.18
    stage2_gate_open_weight: float = 1.8
    stage1_gate_interface_margin: float = 0.02
    stage1_gate_interface_weight: float = 0.6
    stage2_gate_interface_margin: float = 0.03
    stage2_gate_interface_weight: float = 0.45
    stage2_cf_roi_weight: float = 0.08
    stage2_cf_roi_margin: float = 0.0015
    stage2_cf_drop_scales: Tuple[float, float, float] = (0.15, 0.25, 0.15)
    stage2_lr_scale: float = 0.65
    stage1_base_lr_scale: float = 1.0
    stage2_base_lr_scale: float = 0.75
    stage1_mms_lr_scale: float = 1.0
    stage2_mms_lr_scale: float = 0.75
    stage1_gate_local_weight: float = 0.6
    stage2_gate_local_weight: float = 0.12
    stage1_gate_bg_allowance: float = 0.003
    stage2_gate_bg_allowance: float = 0.018
    stage1_gate_hard_bg_weight: float = 0.8
    stage2_gate_hard_bg_weight: float = 1.1
    stage1_gate_rank_weight: float = 0.5
    stage2_gate_rank_weight: float = 0.7
    gate_bg_top_quantile: float = 0.82
    physics_weight: float = 0.0
    velocity_divergence_weight: float = 0.0
    cf_intervention_scale: float = 1.0
    scheduler_t0: int = 0
    scheduler_t_mult: int = 1
    gate_lr_mult: float = 2.0
    stage1_gate_bce_weight: float = 1.2
    stage2_gate_bce_weight: float = 0.35
    stage1_gate_entropy_weight: float = 0.04
    stage2_gate_entropy_weight: float = 0.06
    stage1_gate_burnin_epochs: int = 20
    stage2_gate_burnin_epochs: int = 55
    stage0_gate_temperature: float = 1.9
    stage1_gate_temperature: float = 1.6
    stage2_gate_temperature: float = 1.8
    stage1_gate_interface_burnin_epochs: int = 45
    stage2_gate_interface_burnin_epochs: int = 45
    stage1_gate_local_burnin_epochs: int = 60
    stage2_gate_local_burnin_epochs: int = 55
    stage1_gate_reg_ramp_epochs: int = 35
    stage2_gate_reg_ramp_epochs: int = 55
    stage1_gate_survival_weight: float = 1.0
    stage2_gate_survival_weight: float = 2.2
    gate_survival_floor_ratio: float = 0.86
    irm_stage2_only: bool = True
    stage2_gate_freeze_epochs: int = 24
    stage2_base_freeze_epochs: int = 16
    stage2_mms_freeze_epochs: int = 8
    stage2_gate_param_anchor_weight: float = 0.025
    stage2_gate_param_anchor_ramp_epochs: int = 36
    checkpoint_gate_activity_floor: float = 0.18
    checkpoint_gate_penalty_weight: float = 24.0
    checkpoint_gate_stage1_bonus: float = 0.15
    checkpoint_stage2_penalty: float = 0.0
    stage1_best_start_epoch: int = 0
    checkpoint_main_claim_start_epoch: int = 0
    checkpoint_gate_activity_ceiling: float = 0.88
    checkpoint_gate_bg_ceiling: float = 0.46
    checkpoint_gate_gap_target: float = 0.12
    checkpoint_gate_roi_weight: float = 0.60
    checkpoint_gate_gap_weight: float = 8.00
    checkpoint_gate_non_interface_weight: float = 2.60
    checkpoint_gate_overopen_weight: float = 6.00
    checkpoint_cf_delta_weight: float = 5.50
    checkpoint_cf_bg_ratio_weight: float = 2.20
    checkpoint_cf_gap_ratio_weight: float = 1.60
    patch_version: str = ""
    stage1_roi_fraction_start: float = 0.10
    stage1_roi_fraction_end: float = 0.14
    stage2_roi_fraction_start: float = 0.14
    stage2_roi_fraction_end: float = 0.15
    gate_residual_threshold_quantile: float = 0.84
    gate_residual_threshold_min: float = 0.18
    gate_residual_threshold_max: float = 0.92
    gate_lambda_high_residual: float = 0.55
    gate_lambda_low_residual: float = 1.65
    cf_gradient_modulation: float = 1.35
    cf_contrastive_weight: float = 0.18
    cf_target_delta: float = 0.05
    cf_negative_delta_weight: float = 3.0
    cf_mid_delta_weight: float = 1.4
    cf_roi_shift_floor: float = 0.16
    cf_bg_ratio_cap: float = 0.55
    cf_gap_preserve_ratio: float = 0.78
    envelope_lower_diffusion_scales: Tuple[float, float, float] = (0.15, 0.25, 0.50)
    ood_nan_rollback: bool = True

    # Optional training quality controls (CPU-friendly)
    checkpoint_every: int = 10   # save intermediate checkpoints every N epochs (0 disables)
    early_stop_patience: int = 0  # stop if no improvement for N checkpoints (0 disables)
    early_stop_min_delta: float = 0.0
    # Do not start early stopping until this epoch (inclusive). If 0, defaults to stage0_epochs.
    early_stop_start_epoch: int = 0


@dataclass
class SolverCfg:
    method: str
    rtol: float
    atol: float
    max_steps: int
    use_adjoint: bool
    adjoint_rtol: float
    adjoint_atol: float
    # Number of observation points for the ODE solver.
    # Increasing this improves stability/accuracy for fixed-step solvers and serves as
    # a practical "time truncation" grid for long rollouts.
    num_time_points: int = 2
    # How to construct the time grid: "linspace" or "power" (denser near t=0)
    time_grid: str = "linspace"

    # For fixed-step solvers (rk4/euler/midpoint), torchdiffeq ignores max_num_steps and uses step_size.
    # Keep a safe default; can be overridden in YAML.
    step_size: float = 0.1

@dataclass
class OODCfg:
    z_score_threshold: float
    fallback_disable_mod: bool
    fallback_disable_drift: bool


def _standardize_z(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = z.mean(dim=0, keepdim=True)
    sd = z.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, sd


def _ood_score(z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return ((z - mu) / sd).abs().max(dim=1).values


class Trainer:
    def __init__(
        self,
        model: PhyCausalSTGRN,
        solver_cfg: SolverCfg,
        ot_cfg: OTCfg,
        ood_cfg: OODCfg,
        gate_l1: float,
        gate_tv: float,
        enable_ot: bool,
        device: torch.device,
        irm_lambda: float = 0.0,
        irm_split: str = "x_median",
        irm_warmup_epochs: int = 0,
        irm_balance: bool = True,
        balance_recon: bool = False,
        mms_target_frac_stage1: float = 0.30,
        mms_target_frac_stage2: float = 0.20,
    ):
        if odeint is None:
            raise ImportError("torchdiffeq is required for training/evaluation rollouts. Install torchdiffeq before constructing Trainer.")
        self.model = model.to(device)
        self.solver_cfg = solver_cfg
        self.ot_cfg = ot_cfg
        self.ood_cfg = ood_cfg
        self.gate_l1 = gate_l1
        self.gate_tv = gate_tv
        self.enable_ot = enable_ot
        self.irm_lambda = float(irm_lambda)
        self.irm_split = str(irm_split)
        self.irm_warmup_epochs = int(irm_warmup_epochs)
        self.irm_balance = bool(irm_balance)
        self.balance_recon = bool(balance_recon)
        self.mms_target_frac_stage1 = float(max(0.05, min(0.50, mms_target_frac_stage1)))
        self.mms_target_frac_stage2 = float(max(0.05, min(0.50, mms_target_frac_stage2)))
        self._gate_bg_top_quantile = 0.82

        self.device = device

        self._base_lr = 3e-4
        self._weight_decay = 1e-6
        self._mms_lr_mult = 2.0
        self.opt = AdamW(self.model.parameters(), lr=self._base_lr, weight_decay=self._weight_decay)
        self.scheduler = None

        # training atlas stats for OOD check (zE)
        self.zE_mu: Optional[torch.Tensor] = None
        self.zE_sd: Optional[torch.Tensor] = None
        self._active_stage: str = "stage0"
        self._active_stage_epoch: int = 0
        self._roi_target_frac: float = float(self.mms_target_frac_stage1)
        self._residual_gate_summary: Dict[str, float] = {}
        self._cf_intervention_scale: float = 1.0

    def configure_optim(self, lr: float, weight_decay: float, mms_lr_mult: float = 2.0) -> None:
        # Be tolerant to YAML / CLI configs where numeric values may arrive as strings
        # (for example `1e-6` is parsed as a string by some YAML loaders).
        lr = float(lr)
        weight_decay = float(weight_decay)
        mms_lr_mult = float(max(1.0, mms_lr_mult))
        self._base_lr = lr
        self._weight_decay = weight_decay
        self._mms_lr_mult = mms_lr_mult

        gate_params = list(self.model.odefunc.gate.parameters())
        gate_ids = {id(p) for p in gate_params}
        mms_params = []
        for module in (self.model.odefunc.mod_gnns, self.model.odefunc.mod_mlp):
            mms_params.extend(list(module.parameters()))
        mms_ids = {id(p) for p in mms_params} | gate_ids
        base_params = [p for p in self.model.parameters() if id(p) not in mms_ids]

        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": lr, "name": "base"})
        if mms_params:
            param_groups.append({"params": mms_params, "lr": lr * mms_lr_mult, "name": "mms"})
        if gate_params:
            param_groups.append({"params": gate_params, "lr": lr * mms_lr_mult * 1.75, "name": "gate"})
        self.opt = AdamW(param_groups, weight_decay=weight_decay)

    def _current_stage(self, epoch: int, cfg: TrainCfg) -> str:
        s0 = int(max(0, getattr(cfg, "stage0_epochs", 0) or 0))
        s1 = int(max(0, getattr(cfg, "stage1_epochs", 0) or 0))
        if epoch < s0:
            return "stage0"
        if epoch < (s0 + s1):
            return "stage1"
        return "stage2"

    def _apply_stage_learning_rates(self, stage: str, cfg: TrainCfg) -> None:
        if stage == "stage1":
            base_scale = float(getattr(cfg, "stage1_base_lr_scale", 1.0) or 1.0)
            mms_scale = float(getattr(cfg, "stage1_mms_lr_scale", 1.0) or 1.0)
        elif stage == "stage2":
            base_scale = float(getattr(cfg, "stage2_base_lr_scale", getattr(cfg, "stage2_lr_scale", 1.0)) or 1.0)
            mms_scale = float(getattr(cfg, "stage2_mms_lr_scale", 1.0) or 1.0)
        else:
            base_scale = 1.0
            mms_scale = 1.0
        for pg in self.opt.param_groups:
            name = str(pg.get("name", "base"))
            local = float(self._base_lr)
            if name == "mms":
                local *= float(getattr(self, "_mms_lr_mult", 1.0) or 1.0)
                local *= mms_scale if stage != "stage0" else 1.0
            elif name == "gate":
                local *= float(getattr(self, "_mms_lr_mult", 1.0) or 1.0)
                local *= 1.75
                local *= mms_scale if stage != "stage0" else 1.0
            else:
                local *= base_scale
            pg["lr"] = float(max(local, 1e-6))

    def _stage_gate_scales(self, stage: str, cfg: TrainCfg) -> tuple[float, float]:
        if stage == "stage1":
            return (
                float(getattr(cfg, "stage1_gate_l1_scale", 1.0) or 0.0),
                float(getattr(cfg, "stage1_gate_tv_scale", 1.0) or 0.0),
            )
        if stage == "stage2":
            return (
                float(getattr(cfg, "stage2_gate_l1_scale", 1.0) or 0.0),
                float(getattr(cfg, "stage2_gate_tv_scale", 1.0) or 0.0),
            )
        return (0.0, 0.0)

    def _ode_solve(self, y0: torch.Tensor, ode_in: ODEInput, t: torch.Tensor) -> torch.Tensor:
        """
        Solve ODE and return y(t_end) (same shape as y0).

        torchdiffeq expects `func(t, y) -> dy/dt`. Our vector field additionally
        depends on graph/env context (ODEInput). We therefore bind `ode_in`
        via a lightweight wrapper module, without changing any core algorithm
        or the underlying `PhyCausalODEFunc.forward(t, y, ode_in)` signature.
        """
        base_func = self.model.odefunc
        use_adjoint = bool(getattr(self.solver_cfg, "use_adjoint", True))

        class _BoundODEFunc(torch.nn.Module):
            def __init__(self, func: torch.nn.Module, bound_in: ODEInput):
                super().__init__()
                self.func = func
                self.bound_in = bound_in

            def forward(self, t_: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
                # Core vector field is unchanged; we only bind context.
                return self.func(t_, y_, self.bound_in)

        func = _BoundODEFunc(base_func, ode_in)

        # Forward integration budget from config
        fwd_max_steps = int(getattr(self.solver_cfg, "max_steps", 2000))

        # Backward (adjoint) budget: prefer runtime bump if present, else a larger default.
        runtime_bwd = int(getattr(self, "_runtime_adjoint_steps", 0) or 0)
        bwd_max_steps = max(runtime_bwd, max(fwd_max_steps * 10, fwd_max_steps + 1))

        # Separate adjoint tolerances if provided; otherwise fall back to forward ones.
        adj_rtol = float(getattr(self.solver_cfg, "adjoint_rtol", self.solver_cfg.rtol))
        adj_atol = float(getattr(self.solver_cfg, "adjoint_atol", self.solver_cfg.atol))

        def _run_solver(local_fwd_steps: int, local_bwd_steps: int) -> torch.Tensor:
            method = str(self.solver_cfg.method).lower()

            # torchdiffeq option compatibility:
            # - Adaptive solvers (dopri5/dopri8/bdf/adams/tsit5) accept `max_num_steps`
            # - Fixed-step solvers (rk4/euler/midpoint) accept `step_size` (and ignore max_num_steps)
            adaptive = method in {"dopri5", "dopri8", "bdf", "adams", "tsit5"}
            fixed = method in {"rk4", "euler", "midpoint"}

            local_fwd_opts = {}
            local_bwd_opts = {}

            if adaptive:
                local_fwd_opts["max_num_steps"] = int(local_fwd_steps)
                local_bwd_opts["max_num_steps"] = int(local_bwd_steps)
            elif fixed:
                # In fixed-step mode, cap step count indirectly by choosing a conservative step_size.
                # (The caller controls the t grid; t has only two points by default.)
                step_size = float(getattr(self.solver_cfg, "step_size", 0.05))
                local_fwd_opts["step_size"] = step_size
                local_bwd_opts["step_size"] = step_size
            else:
                # Unknown solver: be conservative and pass no options.
                pass

            if use_adjoint:
                return odeint_adjoint(
                    func,
                    y0,
                    t,
                    method=self.solver_cfg.method,
                    rtol=self.solver_cfg.rtol,
                    atol=self.solver_cfg.atol,
                    options=local_fwd_opts,
                    adjoint_rtol=adj_rtol,
                    adjoint_atol=adj_atol,
                    adjoint_options=local_bwd_opts,
                )
            return odeint(
                func,
                y0,
                t,
                method=self.solver_cfg.method,
                rtol=self.solver_cfg.rtol,
                atol=self.solver_cfg.atol,
                options=local_fwd_opts,
            )

        try:
            y_traj = _run_solver(fwd_max_steps, bwd_max_steps)
        except AssertionError as e:
            # torchdiffeq may hit the step limit in stiff/hard regions.
            # Core model/ODE is unchanged; we only relax integrator budgets.
            if "max_num_steps exceeded" in str(e):
                bumped_bwd = int(min(max(bwd_max_steps * 5, bwd_max_steps + 1), 500000))
                bumped_fwd = int(min(max(fwd_max_steps, bumped_bwd // 10), 500000))
                logger.warning(
                    "ODE integration hit step cap (fwd=%d, adj=%d); retrying with (fwd=%d, adj=%d).",
                    fwd_max_steps,
                    bwd_max_steps,
                    bumped_fwd,
                    bumped_bwd,
                )
                y_traj = _run_solver(bumped_fwd, bumped_bwd)
            else:
                logger.exception("ODE integration failed (assert): %s", e)
                raise
        except FloatingPointError as e:
            # Most common on CPU runs: dy/dt becomes non-finite due to large RK4 step_size or
            # out-of-range mechanomics drift/diffusion at initialization.
            # We do NOT change the core ODE; we only retry the integrator with a smaller step_size.
            if fixed:
                base_step = float(getattr(self.solver_cfg, "step_size", 0.05))
                # try a couple of smaller steps
                for factor in (0.5, 0.25):
                    try_step = max(1e-3, base_step * factor)
                    logger.warning("Non-finite during ODE solve; retrying with step_size=%.4g", try_step)
                    def _run_solver_retry(local_fwd_steps, local_bwd_steps):
                        local_fwd_opts2 = dict(local_fwd_opts)
                        local_bwd_opts2 = dict(local_bwd_opts)
                        local_fwd_opts2["step_size"] = try_step
                        local_bwd_opts2["step_size"] = try_step
                        if use_adjoint:
                            return odeint_adjoint(
                                func, y0, t, method=self.solver_cfg.method,
                                rtol=self.solver_cfg.rtol, atol=self.solver_cfg.atol,
                                options=local_fwd_opts2,
                                adjoint_rtol=adj_rtol, adjoint_atol=adj_atol,
                                adjoint_options=local_bwd_opts2,
                            )
                        return odeint(
                            func, y0, t, method=self.solver_cfg.method,
                            rtol=self.solver_cfg.rtol, atol=self.solver_cfg.atol,
                            options=local_fwd_opts2,
                        )
                    try:
                        y_traj = _run_solver_retry(fwd_max_steps, bwd_max_steps)
                        break
                    except FloatingPointError:
                        y_traj = None
                if y_traj is None:
                    logger.exception("ODE integration failed (non-finite) after retries: %s", e)
                    raise
            else:
                logger.exception("ODE integration failed (non-finite): %s", e)
                raise
        except Exception as e:
            logger.exception("ODE integration failed: %s", e)
            raise

        yT = y_traj[-1]
        check_tensor_finite(yT, "ode.solve.yT")
        return yT


    def _make_time_grid(self, t_end: float) -> torch.Tensor:
        """Construct a stable time grid for odeint/odeint_adjoint.

        Why:
          - Using only [0, t_end] can be numerically fragile for fixed-step solvers on real ST data.
          - A denser grid acts as additional truncation points for the integrator (not changing the core ODE),
            typically improving training stability and rollout fidelity.

        The grid is still very small (default 2) and CPU-friendly.
        """
        t_end = float(t_end)
        n = int(max(2, getattr(self.solver_cfg, "num_time_points", 2)))
        mode = str(getattr(self.solver_cfg, "time_grid", "linspace")).lower()
        if n == 2:
            return torch.tensor([0.0, t_end], device=self.device, dtype=torch.float32)
        if mode == "power":
            # denser near 0 for stiff transients
            u = torch.linspace(0.0, 1.0, n, device=self.device, dtype=torch.float32)
            u = u ** 2
            return u * t_end
        # default: uniform
        return torch.linspace(0.0, t_end, n, device=self.device, dtype=torch.float32)

    def _compute_mms_mask(self, snap: SpatialSnapshot, *, target_frac: float = 0.18) -> Optional[torch.Tensor]:
        """Build a narrow interface-focused MMS mask for Stage-1 training.

        The current model tends to learn a nearly uniform Gate. This helper shrinks the region where
        MMS is allowed to open, using annotations when available and falling back to mechanomics or
        zN residual signal. The mask is intentionally conservative to encourage localized mechanism drift.
        """
        coords = snap.coords.to(self.device).float()
        n = int(coords.shape[0])
        if n <= 4:
            return None

        def _shrink(mask: torch.Tensor, frac: float) -> torch.Tensor:
            frac = float(max(0.04, min(0.28, frac)))
            if int(mask.sum()) <= max(4, int(frac * n)):
                return mask
            x = coords[:, 0]
            y = coords[:, 1]
            x0 = x[mask].median() if int(mask.sum()) > 0 else x.median()
            y0 = y[mask].median() if int(mask.sum()) > 0 else y.median()
            dx = (x - x0).abs() / (x[mask].std().clamp_min(1.0) if int(mask.sum()) > 2 else x.std().clamp_min(1.0))
            dy = (y - y0).abs() / (y[mask].std().clamp_min(1.0) if int(mask.sum()) > 2 else y.std().clamp_min(1.0))
            d = dx + 0.60 * dy
            k = max(4, int(frac * n))
            keep_idx = torch.argsort(d)[:k]
            keep = torch.zeros(n, dtype=torch.bool, device=self.device)
            keep[keep_idx] = True
            return mask & keep if int((mask & keep).sum()) >= 4 else keep

        ann = getattr(snap, 'annotations', None)
        if isinstance(ann, torch.Tensor):
            lab = ann.to(self.device)
            if lab.ndim == 2 and lab.shape[1] >= 1:
                lab = lab[:, 0]
            lab = lab.reshape(-1).float()
            valid = torch.isfinite(lab)
            valid01 = valid & ((lab == 0) | (lab == 1))
            if int(valid01.sum()) >= 8:
                tumor = lab == 1
                d = torch.cdist(coords, coords)
                k = int(min(8, max(2, n - 1)))
                nn = torch.topk(d, k=k + 1, largest=False).indices[:, 1:]
                nbr_valid = valid01[nn]
                nbr_tumor = tumor[nn] & nbr_valid
                denom = nbr_valid.float().sum(dim=1).clamp_min(1.0)
                p = nbr_tumor.float().sum(dim=1) / denom
                hetero = (p > 0.05) & (p < 0.95) & valid01
                if int(hetero.sum()) >= 4:
                    nbr_hetero = hetero[nn].any(dim=1) & valid01
                    mask = hetero | nbr_hetero
                    return _shrink(mask, min(0.18, max(float(target_frac), 0.08))).float().unsqueeze(1)

        mech = snap.mechanomics.to(self.device) if getattr(snap, 'mechanomics', None) is not None else None
        if isinstance(mech, torch.Tensor) and mech.ndim == 2 and mech.shape[1] >= 1:
            chan = 1 if mech.shape[1] >= 2 else 0
            s = mech[:, chan].float()
            thr = torch.quantile(s, 1.0 - float(target_frac)) if chan == 1 else torch.quantile(s, float(target_frac))
            mask = (s >= thr) if chan == 1 else (s <= thr)
            if int(mask.sum()) >= 4:
                return mask.float().unsqueeze(1)

        zN = snap.zN.to(self.device) if getattr(snap, 'zN', None) is not None else None
        if isinstance(zN, torch.Tensor) and zN.ndim == 2 and zN.shape[1] >= 1:
            s = zN[:, -1].float()
            thr = torch.quantile(s, 1.0 - float(target_frac))
            mask = s >= thr
            if int(mask.sum()) >= 4:
                return _shrink(mask, min(0.16, max(float(target_frac), 0.08))).float().unsqueeze(1)
        return None

    def _build_ode_input(self, snap: SpatialSnapshot, *, stage: Optional[str] = None) -> ODEInput:
        mms_mask = None
        target_frac = self.mms_target_frac_stage1 if stage == "stage1" else self.mms_target_frac_stage2
        if stage == "stage1":
            mms_mask = self._compute_mms_mask(snap, target_frac=target_frac)
        elif stage == "stage2":
            mms_mask = self._compute_mms_mask(snap, target_frac=min(0.12, max(0.05, target_frac)))
        return ODEInput(
            edge_index=snap.edge_index.to(self.device),
            coords=snap.coords.to(self.device),
            zI=snap.zI.to(self.device),
            zN=snap.zN.to(self.device),
            zE=snap.zE.to(self.device),
            mechanomics=snap.mechanomics.to(self.device),
            mass=None,
            mms_mask=mms_mask,
        )

    def _loss_recon(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        # Robust reconstruction loss (Huber / Smooth L1) to suppress extreme technical outliers
        # while preserving informative gradients for small residuals.
        # delta=1.0 is a stable default; can be exposed via config if needed.
        loss = torch.nn.functional.huber_loss(x_pred, x_true, delta=1.0, reduction="mean")
        return loss


    def _loss_recon_balanced(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        env0: torch.Tensor,
        env1: torch.Tensor,
    ) -> torch.Tensor:
        """Environment-balanced recon loss (Tumor vs Stroma, etc.).

        Uses 0.5*(mean loss in env0 + mean loss in env1). Cells with ignore label (-1)
        (mask neither env0 nor env1) are excluded to reduce noise.
        Falls back to global recon if masks are unusable.
        """
        try:
            n = int(min(x_pred.shape[0], x_true.shape[0], env0.numel(), env1.numel()))
            if n <= 0:
                return self._loss_recon(x_pred, x_true)
            if x_pred.shape[0] != n or x_true.shape[0] != n or env0.numel() != n or env1.numel() != n:
                x_pred = x_pred[:n]
                x_true = x_true[:n]
                env0 = env0[:n]
                env1 = env1[:n]
            m0 = env0.to(dtype=torch.bool)
            m1 = env1.to(dtype=torch.bool)
            if int(m0.sum()) < 5 or int(m1.sum()) < 5:
                return self._loss_recon(x_pred, x_true)
            l0 = torch.nn.functional.huber_loss(x_pred[m0], x_true[m0], delta=1.0, reduction="mean")
            l1 = torch.nn.functional.huber_loss(x_pred[m1], x_true[m1], delta=1.0, reduction="mean")
            return 0.5 * l0 + 0.5 * l1
        except Exception:
            return self._loss_recon(x_pred, x_true)

    def _loss_gate_regularizers(
        self,
        gate: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        l1_scale: float = 1.0,
        tv_scale: float = 1.0,
        node_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Continuous Relaxation of L0 (Hard-Concrete) + graph TV for localization.
        # During Stage-1 we penalize background nodes more than interface-rich nodes,
        # so the mechanism branch can open locally before global fine-tuning.
        p_nonzero = getattr(self.model.odefunc, "gate_p_nonzero_last", None)
        if isinstance(p_nonzero, torch.Tensor) and p_nonzero.shape == gate.shape:
            node_activity = p_nonzero.float().mean(dim=1)
        else:
            node_activity = gate.float().mean(dim=1)

        if isinstance(node_weights, torch.Tensor) and node_weights.numel() == node_activity.numel():
            w = node_weights.reshape(-1).to(node_activity.device, dtype=node_activity.dtype).clamp_min(0.0)
            denom = w.sum().clamp_min(1e-6)
            l0 = (node_activity * w).sum() / denom
        else:
            l0 = node_activity.mean()

        tv = total_variation_on_graph(gate, edge_index)
        act = node_activity.mean().detach()
        relax = float(torch.clamp(act / 0.12, min=0.15, max=1.0).item())
        return (self.gate_l1 * float(l1_scale) * relax * l0) + (self.gate_tv * float(tv_scale) * relax * tv)

    def _current_roi_target_frac(self) -> float:
        stage = str(getattr(self, "_active_stage", "stage0") or "stage0")
        stage_epoch = int(getattr(self, "_active_stage_epoch", 0) or 0)
        cfg = getattr(self, "_stage_cfg", None)
        if cfg is None:
            return float(self._roi_target_frac)
        if stage == "stage1":
            s = float(getattr(cfg, "stage1_roi_fraction_start", max(0.08, self.mms_target_frac_stage1)) or max(0.08, self.mms_target_frac_stage1))
            e = float(getattr(cfg, "stage1_roi_fraction_end", min(0.15, max(s, self.mms_target_frac_stage1 + 0.02))) or min(0.15, max(s, self.mms_target_frac_stage1 + 0.02)))
            burn = max(1, int(getattr(cfg, "stage1_gate_reg_ramp_epochs", 30) or 30))
            prog = min(1.0, max(0.0, float(stage_epoch) / float(burn)))
            return float(s + (e - s) * prog)
        if stage == "stage2":
            s = float(getattr(cfg, "stage2_roi_fraction_start", max(0.10, self.mms_target_frac_stage2)) or max(0.10, self.mms_target_frac_stage2))
            e = float(getattr(cfg, "stage2_roi_fraction_end", max(s, 0.15)) or max(s, 0.15))
            burn = max(1, int(getattr(cfg, "stage2_gate_reg_ramp_epochs", 40) or 40))
            prog = min(1.0, max(0.0, float(stage_epoch) / float(burn)))
            return float(s + (e - s) * prog)
        return float(max(0.08, min(0.18, self.mms_target_frac_stage1)))

    def _compute_residual_focus(self, ode_in: ODEInput) -> Optional[Dict[str, torch.Tensor]]:
        zN = getattr(ode_in, "zN", None)
        if zN is None or not isinstance(zN, torch.Tensor) or zN.ndim != 2 or zN.shape[0] < 4:
            return None
        signal = zN[:, -1].detach().float()
        signal = signal - signal.median()
        mad = (signal - signal.median()).abs().median().clamp_min(1e-4)
        residual = (signal.abs() / mad).clamp(min=0.0)
        q = float(getattr(getattr(self, "_stage_cfg", None), "gate_residual_threshold_quantile", 0.84) or 0.84)
        q = min(0.98, max(0.50, q))
        thr = float(torch.quantile(residual, q).item())
        thr = max(float(getattr(getattr(self, "_stage_cfg", None), "gate_residual_threshold_min", 0.18) or 0.18), thr)
        thr = min(float(getattr(getattr(self, "_stage_cfg", None), "gate_residual_threshold_max", 0.92) or 0.92), thr)
        high = (residual >= thr).float()
        self._residual_gate_summary = {
            "threshold": float(thr),
            "high_fraction": float(high.mean().item()),
            "residual_mean": float(residual.mean().item()),
        }
        return {
            "residual": residual.to(ode_in.zN.device, dtype=ode_in.zN.dtype),
            "high": high.to(ode_in.zN.device, dtype=ode_in.zN.dtype),
            "threshold": ode_in.zN.new_tensor(thr),
        }

    def _gate_background_weights(self, ode_in: ODEInput) -> Optional[torch.Tensor]:
        zN = getattr(ode_in, "zN", None)
        if zN is None or not isinstance(zN, torch.Tensor) or zN.ndim != 2 or zN.shape[1] < 1:
            return None
        signal = zN[:, -1].detach().float()
        signal = signal - signal.min()
        signal = signal / signal.max().clamp_min(1e-6)
        residual_info = self._compute_residual_focus(ode_in)
        hi_lambda = float(getattr(getattr(self, "_stage_cfg", None), "gate_lambda_high_residual", 0.55) or 0.55)
        lo_lambda = float(getattr(getattr(self, "_stage_cfg", None), "gate_lambda_low_residual", 1.65) or 1.65)
        weights = (lo_lambda - 0.75 * signal).clamp(min=0.10, max=max(lo_lambda, 1.0))
        if residual_info is not None:
            high = residual_info["high"].reshape(-1).to(weights.device, dtype=weights.dtype)
            residual = residual_info["residual"].reshape(-1).to(weights.device, dtype=weights.dtype)
            residual = residual / residual.max().clamp_min(1e-6)
            high_weights = (hi_lambda * (1.0 - 0.35 * residual)).clamp(min=0.05, max=max(0.9, hi_lambda))
            low_weights = (lo_lambda + 0.35 * (1.0 - residual)).clamp(min=0.20, max=max(1.8, lo_lambda + 0.4))
            weights = torch.where(high > 0.5, high_weights, low_weights)
        return weights.clamp(min=0.05, max=2.25)

    
    

    def _loss_gate_activity_floor(
        self,
        gate: torch.Tensor,
        ode_in: ODEInput,
        *,
        target: float = 0.04,
        weight: float = 0.3,
    ) -> torch.Tensor:
        if weight <= 0.0 or target <= 0.0:
            return gate.new_zeros(())
        p_nonzero = getattr(self.model.odefunc, "gate_p_nonzero_last", None)
        if isinstance(p_nonzero, torch.Tensor) and p_nonzero.shape == gate.shape:
            node_activity = p_nonzero.float().mean(dim=1)
        else:
            node_activity = gate.float().mean(dim=1)

        mms_mask = getattr(ode_in, "mms_mask", None)
        if isinstance(mms_mask, torch.Tensor):
            focus = mms_mask.reshape(-1).to(node_activity.device, dtype=node_activity.dtype).clamp(0.0, 1.0)
        else:
            focus = None

        zN = getattr(ode_in, "zN", None)
        if isinstance(zN, torch.Tensor) and zN.ndim == 2 and zN.shape[1] >= 1:
            signal = zN[:, -1].detach().float()
            signal = signal - signal.min()
            signal = signal / signal.max().clamp_min(1e-6)
            signal = signal.to(node_activity.device, dtype=node_activity.dtype)
        else:
            signal = torch.ones_like(node_activity)

        if focus is None or float(focus.sum().item()) < 4.0:
            focus = (0.15 + 0.85 * signal).clamp(0.0, 1.0)
        else:
            focus = (0.35 + 0.65 * signal) * focus

        act_focus = (node_activity * focus).sum() / focus.sum().clamp_min(1e-6)
        deficit = torch.relu(gate.new_tensor(float(target)) - act_focus)

        if isinstance(mms_mask, torch.Tensor):
            bg = (1.0 - mms_mask.reshape(-1).to(node_activity.device, dtype=node_activity.dtype)).clamp(0.0, 1.0)
            bg_mean = (node_activity * bg).sum() / bg.sum().clamp_min(1e-6)
            bg_pen = torch.relu(bg_mean - gate.new_tensor(float(target) * 0.75))
        else:
            bg_pen = gate.new_zeros(())

        return gate.new_tensor(float(weight)) * (deficit ** 2 + 0.75 * deficit + 0.20 * (bg_pen ** 2))

    def _compute_interface_roi(self, snap: SpatialSnapshot) -> Optional[object]:
        ann = getattr(snap, "annotations", None)
        coords = getattr(snap, "coords", None)
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[0] < 8:
            return None
        try:
            target_frac = float(self._current_roi_target_frac())
            self._roi_target_frac = target_frac
            roi = define_interface_roi_adaptive(
                coords.detach(),
                annotations=(ann.detach() if isinstance(ann, torch.Tensor) else None),
                target_interface_frac=target_frac,
                band_frac=0.045,
                min_band=22.0,
                max_band=84.0,
            )
            if int(roi.mask_interface.reshape(-1).sum().item()) < 4:
                return None
            return roi
        except Exception:
            return None

    
    def _interface_target(self, snap: SpatialSnapshot) -> Optional[torch.Tensor]:
        roi = self._compute_interface_roi(snap)
        if roi is None:
            return None
        target = roi.mask_interface.reshape(-1).to(self.device, dtype=torch.float32)
        ann = getattr(snap, "annotations", None)
        if isinstance(ann, torch.Tensor):
            annv = ann.to(self.device)
            if annv.ndim == 2 and annv.shape[1] >= 1:
                annv = annv[:, 0]
            annv = annv.reshape(-1).float()
            valid = torch.isfinite(annv) & ((annv == 0) | (annv == 1))
            if int(valid.sum()) >= 8:
                coords = snap.coords.to(self.device).float()
                d = torch.cdist(coords, coords)
                k = int(min(10, max(2, coords.shape[0] - 1)))
                nn = torch.topk(d, k=k + 1, largest=False).indices[:, 1:]
                tumor = annv == 1
                p = (tumor[nn] & valid[nn]).float().sum(dim=1) / valid[nn].float().sum(dim=1).clamp_min(1.0)
                hetero = ((p > 0.22) & (p < 0.78) & valid).float()
                core = (target > 0.5).float() * hetero
                mixed = torch.where(core > 0.5, torch.full_like(target, 1.0), 0.55 * target + 0.15 * hetero)
                target = torch.where(valid, mixed.clamp(0.0, 1.0), target)
        return target.clamp(0.0, 1.0)

    def _gate_focus_mask(
        self,
        snap: SpatialSnapshot,
        ode_in: Optional[ODEInput] = None,
    ) -> Optional[torch.Tensor]:
        """Preferred supervision region for gate-related losses.

        Use the runtime MMS mask when available; otherwise fall back to the
        geometric/interface ROI target. This avoids recursion bugs and keeps
        the focus region aligned with the actual MMS-enabled area.
        """
        mm = getattr(ode_in, "mms_mask", None) if ode_in is not None else None
        if isinstance(mm, torch.Tensor) and mm.numel() == snap.coords.shape[0]:
            target = mm.reshape(-1).to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
            if int((target > 0.5).sum().item()) >= 4:
                return target
        target = self._interface_target(snap)
        if target is None:
            return None
        return target.reshape(-1).to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

    def _loss_gate_survival(
        self,
        gate: torch.Tensor,
        target_activity: float,
        *,
        weight: float = 0.0,
    ) -> torch.Tensor:
        if weight <= 0.0 or target_activity <= 0.0:
            return gate.new_zeros(())
        p_nonzero = getattr(self.model.odefunc, "gate_p_nonzero_last", None)
        if isinstance(p_nonzero, torch.Tensor) and p_nonzero.shape == gate.shape:
            node_activity = p_nonzero.float().mean(dim=1)
        else:
            node_activity = gate.float().mean(dim=1)
        act = node_activity.mean()
        floor = gate.new_tensor(float(target_activity))
        ceil = gate.new_tensor(float(min(0.24, max(0.12, 1.35 * float(target_activity)))))
        low_pen = 1.35 * torch.relu(floor - act).pow(2) + 0.25 * torch.relu(0.60 * floor - act)
        high_pen = 1.10 * torch.relu(act - ceil).pow(2) + 0.20 * torch.relu(act - 1.20 * ceil)
        return gate.new_tensor(float(weight)) * (low_pen + high_pen)


    def _capture_stage2_gate_reference(self) -> None:
        if hasattr(self, "_stage2_gate_param_ref"):
            return
        ref = {}
        for name, p in self.model.odefunc.gate.named_parameters():
            ref[name] = p.detach().clone()
        self._stage2_gate_param_ref = ref

    def _loss_gate_param_anchor(self, *, weight: float = 0.0) -> torch.Tensor:
        if weight <= 0.0 or not hasattr(self, "_stage2_gate_param_ref"):
            return torch.tensor(0.0, device=self.device)
        ref = getattr(self, "_stage2_gate_param_ref", None)
        if not isinstance(ref, dict) or len(ref) == 0:
            return torch.tensor(0.0, device=self.device)
        acc = None
        n = 0
        for name, p in self.model.odefunc.gate.named_parameters():
            rp = ref.get(name)
            if rp is None or rp.shape != p.shape:
                continue
            term = (p - rp.to(p.device, dtype=p.dtype)).pow(2).mean()
            acc = term if acc is None else (acc + term)
            n += 1
        if acc is None or n == 0:
            return torch.tensor(0.0, device=self.device)
        return acc * (float(weight) / float(max(1, n)))

    def _history_last(self, history: dict, key: str, default: float = 0.0) -> float:
        vals = history.get(key, None)
        if not vals:
            return float(default)
        try:
            return float(vals[-1])
        except Exception:
            return float(default)

    def _gate_checkpoint_penalty(self, cfg: TrainCfg, history: dict, stage: str) -> float:
        gate_act = self._history_last(history, "gate_activity_mean", 0.0)
        gate_roi = self._history_last(history, "gate_roi_mean", gate_act)
        gate_iface = self._history_last(history, "gate_interface_mean", gate_roi)
        gate_non = self._history_last(history, "gate_non_interface_mean", gate_act)
        gate_gap = gate_iface - gate_non
        floor = float(getattr(cfg, "checkpoint_gate_activity_floor", 0.18) or 0.18)
        ceil = float(getattr(cfg, "checkpoint_gate_activity_ceiling", 0.88) or 0.88)
        bg_ceil = float(getattr(cfg, "checkpoint_gate_bg_ceiling", 0.46) or 0.46)
        gap_target = float(getattr(cfg, "checkpoint_gate_gap_target", 0.12) or 0.12)
        w = float(getattr(cfg, "checkpoint_gate_penalty_weight", 24.0) or 24.0)
        roi_w = float(getattr(cfg, "checkpoint_gate_roi_weight", 0.60) or 0.60)
        gap_w = float(getattr(cfg, "checkpoint_gate_gap_weight", 8.0) or 8.0)
        non_w = float(getattr(cfg, "checkpoint_gate_non_interface_weight", 2.6) or 2.6)
        over_w = float(getattr(cfg, "checkpoint_gate_overopen_weight", 6.0) or 6.0)
        stage_bonus = float(getattr(cfg, "checkpoint_gate_stage1_bonus", 0.0) or 0.0) if stage == "stage1" else 0.0

        penalty = 0.0
        penalty += w * max(0.0, floor - gate_act)
        penalty += 0.5 * w * max(0.0, floor - gate_roi)
        penalty += roi_w * max(0.0, floor - gate_iface)
        penalty += gap_w * max(0.0, gap_target - gate_gap)
        penalty += non_w * max(0.0, gate_non - bg_ceil)
        penalty += over_w * max(0.0, gate_act - ceil)
        if gate_act <= 1e-5:
            penalty += 2.5 * w
        penalty -= stage_bonus * max(0.0, gate_gap)
        return float(penalty)

    def _cf_checkpoint_penalty(self, cfg: TrainCfg, history: dict) -> float:
        delta = self._history_last(history, "cf_delta_p_cross", 0.0)
        roi_shift = self._history_last(history, "cf_roi_shift", 0.0)
        bg_shift = self._history_last(history, "cf_bg_shift", 0.0)
        gap_ratio = self._history_last(history, "cf_gap_ratio", 1.0)
        target_delta = float(getattr(cfg, "cf_target_delta", 0.05) or 0.05)
        delta_w = float(getattr(cfg, "checkpoint_cf_delta_weight", 5.5) or 5.5)
        bg_w = float(getattr(cfg, "checkpoint_cf_bg_ratio_weight", 2.2) or 2.2)
        gap_w = float(getattr(cfg, "checkpoint_cf_gap_ratio_weight", 1.6) or 1.6)
        ratio_cap = float(getattr(cfg, "cf_bg_ratio_cap", 0.55) or 0.55)
        safe_roi = max(roi_shift, 1e-6)
        bg_ratio = bg_shift / safe_roi
        penalty = 0.0
        penalty += delta_w * max(0.0, target_delta - delta)
        penalty += 1.4 * delta_w * max(0.0, -delta)
        penalty += bg_w * max(0.0, bg_ratio - ratio_cap)
        penalty += gap_w * max(0.0, float(getattr(cfg, "cf_gap_preserve_ratio", 0.78) or 0.78) - gap_ratio)
        return float(penalty)

    def _main_claim_ckpt_score(self, cfg: TrainCfg, history: dict, stage: str, cur_loss: float) -> float:
        score = float(cur_loss)
        if stage == "stage0":
            return score
        score += self._gate_checkpoint_penalty(cfg, history, stage)
        if stage == "stage2":
            score += self._cf_checkpoint_penalty(cfg, history)
            score += float(getattr(cfg, "checkpoint_stage2_penalty", 0.0) or 0.0)
        return float(score)

    def _checkpoint_objective(self, cfg: TrainCfg, history: dict, stage: str, cur_loss: float) -> float:
        return self._main_claim_ckpt_score(cfg, history, stage, cur_loss)


    def _save_training_manifest(self, cfg: TrainCfg, manifest: Dict[str, object]) -> None:
        try:
            os.makedirs(cfg.save_dir, exist_ok=True)
            with open(os.path.join(cfg.save_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Failed to write training_manifest.json")

    def _stage1_gate_ckpt_score(self, history: dict, cur_loss: float) -> float:
        class _CfgProxy:
            checkpoint_gate_activity_floor = 0.08
            checkpoint_gate_activity_ceiling = 0.32
            checkpoint_gate_bg_ceiling = 0.18
            checkpoint_gate_gap_target = 0.08
            checkpoint_gate_penalty_weight = 10.0
            checkpoint_gate_roi_weight = 0.30
            checkpoint_gate_gap_weight = 7.40
            checkpoint_gate_non_interface_weight = 2.80
            checkpoint_gate_overopen_weight = 4.80
            checkpoint_gate_stage1_bonus = 0.10
        return float(cur_loss + self._gate_checkpoint_penalty(_CfgProxy(), history, "stage1"))

    def _loss_gate_hard_background(
        self,
        gate: torch.Tensor,
        snap: SpatialSnapshot,
        ode_in: Optional[ODEInput] = None,
        *,
        weight: float = 0.0,
        top_quantile: float = 0.76,
    ) -> torch.Tensor:
        if weight <= 0.0:
            return gate.new_zeros(())
        focus = self._gate_focus_mask(snap, ode_in)
        if focus is None:
            return gate.new_zeros(())
        node_gate = getattr(self.model.odefunc, "gate_node_last", None)
        pred = node_gate.reshape(-1).float().clamp(1e-5, 1.0 - 1e-5) if isinstance(node_gate, torch.Tensor) else gate.float().mean(dim=1).clamp(1e-5, 1.0 - 1e-5)
        bg_mask = ~(focus.reshape(-1).to(pred.device) > 0.5)
        if int(bg_mask.sum()) < 8:
            return gate.new_zeros(())
        bg_vals = pred[bg_mask]
        thr = torch.quantile(bg_vals.detach(), float(max(0.55, min(0.98, top_quantile))))
        hard_bg = bg_vals[bg_vals >= thr]
        if hard_bg.numel() == 0:
            hard_bg = bg_vals
        loss = torch.relu(hard_bg.mean() - gate.new_tensor(0.060))
        return gate.new_tensor(float(weight)) * (1.75 * loss.pow(2) + 0.80 * loss)

    def _loss_gate_rank_contrast(
        self,
        gate: torch.Tensor,
        snap: SpatialSnapshot,
        ode_in: Optional[ODEInput] = None,
        *,
        weight: float = 0.0,
    ) -> torch.Tensor:
        if weight <= 0.0:
            return gate.new_zeros(())
        focus = self._gate_focus_mask(snap, ode_in)
        if focus is None:
            return gate.new_zeros(())
        node_gate = getattr(self.model.odefunc, "gate_node_last", None)
        pred = node_gate.reshape(-1).float().clamp(1e-5, 1.0 - 1e-5) if isinstance(node_gate, torch.Tensor) else gate.float().mean(dim=1).clamp(1e-5, 1.0 - 1e-5)
        roi_mask = focus.reshape(-1).to(pred.device) > 0.5
        bg_mask = ~roi_mask
        if int(roi_mask.sum()) < 8 or int(bg_mask.sum()) < 8:
            return gate.new_zeros(())
        roi_vals = pred[roi_mask]
        bg_vals = pred[bg_mask]
        roi_top = torch.quantile(roi_vals, 0.72)
        roi_mid = torch.quantile(roi_vals, 0.50)
        bg_top = torch.quantile(bg_vals, float(getattr(self, "_gate_bg_top_quantile", 0.80)))
        loss = torch.relu((bg_top + gate.new_tensor(0.26)) - roi_top) + 0.45 * torch.relu((bg_top + gate.new_tensor(0.18)) - roi_mid)
        return gate.new_tensor(float(weight)) * (1.8 * loss.pow(2) + 0.75 * loss)

    def _loss_gate_interface_bce(
        self,
        gate: torch.Tensor,
        snap: SpatialSnapshot,
        ode_in: Optional[ODEInput] = None,
        *,
        weight: float = 0.0,
        entropy_weight: float = 0.0,
    ) -> torch.Tensor:
        if weight <= 0.0:
            return gate.new_zeros(())
        target = self._gate_focus_mask(snap, ode_in)
        if target is None or target.numel() != gate.shape[0]:
            return gate.new_zeros(())
        node_gate = getattr(self.model.odefunc, "gate_node_last", None)
        pred = node_gate.reshape(-1).clamp(1e-5, 1.0 - 1e-5) if isinstance(node_gate, torch.Tensor) else gate.float().mean(dim=1).clamp(1e-5, 1.0 - 1e-5)
        smooth_target = torch.where(target > 0.5, torch.full_like(target, 0.97), torch.full_like(target, 0.01))
        pos_frac = target.mean().clamp(1e-3, 1.0 - 1e-3)
        pos_w = ((1.0 - pos_frac) / pos_frac).clamp(1.5, 16.0)
        neg_hard = (target < 0.5) & (pred > 0.12)
        weights = torch.where(target > 0.5, pos_w, torch.ones_like(target))
        weights = torch.where(neg_hard, 6.0 * weights, weights)
        p_t = torch.where(target > 0.5, pred, 1.0 - pred)
        focal = (1.0 - p_t).pow(3.2)
        bce_elem = F.binary_cross_entropy(pred, smooth_target, reduction="none")
        bce = (bce_elem * weights * focal).mean()
        # Focal-Tversky style auxiliary term for small/imbalanced interface regions.
        y = (target > 0.5).float()
        tp = (pred * y).sum()
        fp = (pred * (1.0 - y)).sum()
        fn = ((1.0 - pred) * y).sum()
        tversky = (tp + 1e-4) / (tp + 0.82 * fp + 0.18 * fn + 1e-4)
        ft = (1.0 - tversky).pow(1.55)
        if entropy_weight > 0.0:
            ent = -(pred * torch.log(pred) + (1.0 - pred) * torch.log(1.0 - pred)).mean()
        else:
            ent = gate.new_zeros(())
        return gate.new_tensor(float(weight)) * (0.68 * bce + 0.32 * ft) + gate.new_tensor(float(entropy_weight)) * ent

    def make_targeted_do_zE(
        self,
        ode_in: ODEInput,
        roi_mask: torch.Tensor,
        drop_scales: Tuple[float, float, float],
        *,
        extra_scale: float = 1.0,
        roi: Optional[object] = None,
    ) -> torch.Tensor:
        do_zE = ode_in.zE.clone()
        roi_mask = roi_mask.reshape(-1, 1).to(self.device, dtype=do_zE.dtype)
        if do_zE.ndim != 2 or do_zE.shape[0] != roi_mask.shape[0]:
            return do_zE
        roi_bool = roi_mask.reshape(-1) > 0.5
        bg_bool = ~roi_bool
        tumor_side = (getattr(roi, 'side_tumor', None).reshape(-1).to(self.device).bool() if roi is not None and isinstance(getattr(roi, 'side_tumor', None), torch.Tensor) else roi_bool)
        stroma_side = (getattr(roi, 'side_stroma', None).reshape(-1).to(self.device).bool() if roi is not None and isinstance(getattr(roi, 'side_stroma', None), torch.Tensor) else roi_bool)
        roi_t = roi_bool & tumor_side
        roi_s = roi_bool & stroma_side
        if int(roi_bool.sum()) < 4 or int(bg_bool.sum()) < 4:
            order = list(range(min(3, int(do_zE.shape[1]))))
            roi_mean = do_zE.mean(dim=0)
            bg_mean = do_zE.mean(dim=0)
            t_mean = roi_mean
            s_mean = bg_mean
        else:
            roi_mean = do_zE[roi_bool].mean(dim=0)
            bg_mean = do_zE[bg_bool].mean(dim=0)
            t_mean = do_zE[roi_t].mean(dim=0) if int(roi_t.sum()) >= 2 else roi_mean
            s_mean = do_zE[roi_s].mean(dim=0) if int(roi_s.sum()) >= 2 else bg_mean
            contrast = (t_mean - s_mean).abs() + 0.35 * (roi_mean - bg_mean).abs()
            order = torch.argsort(contrast, descending=True).tolist()
        ds = list(drop_scales) + [drop_scales[-1] if len(drop_scales) > 0 else 0.0] * max(0, int(do_zE.shape[1]) - len(drop_scales))
        node_gate = getattr(self.model.odefunc, "gate_node_last", None)
        if isinstance(node_gate, torch.Tensor) and node_gate.numel() == do_zE.shape[0]:
            g = node_gate.reshape(-1, 1).to(do_zE.device, dtype=do_zE.dtype).clamp(0.0, 1.0)
        else:
            g = roi_mask
        tumor_focus = (roi_t.float().reshape(-1, 1).to(do_zE.device, dtype=do_zE.dtype)) * (0.15 + 0.85 * g)
        stroma_focus = (roi_s.float().reshape(-1, 1).to(do_zE.device, dtype=do_zE.dtype)) * (0.15 + 0.85 * g)
        for rank, j in enumerate(order[: min(len(order), len(ds), int(do_zE.shape[1]))]):
            scale = min(2.6, float(ds[min(rank, len(drop_scales)-1)]) * float(extra_scale))
            delta_ts = (t_mean[j] - s_mean[j]).clamp(min=-1.25, max=1.25)
            delta_rb = (roi_mean[j] - bg_mean[j]).clamp(min=-1.25, max=1.25)
            if max(float(delta_ts.abs().item()), float(delta_rb.abs().item())) < 0.04:
                continue
            cur = do_zE[:, j:j+1]
            midpoint = 0.5 * (t_mean[j] + s_mean[j])
            # Side-aware collapse of the interface contrast: pull tumor-side and stroma-side ROI cells
            # toward the cross-interface midpoint, with a mild extra drift toward the background mean.
            t_target = (0.55 * midpoint + 0.45 * bg_mean[j]).view(1, 1).to(do_zE.device, dtype=do_zE.dtype)
            s_target = (0.55 * midpoint + 0.45 * bg_mean[j]).view(1, 1).to(do_zE.device, dtype=do_zE.dtype)
            moved = cur
            moved = moved + tumor_focus * (scale * (t_target - moved))
            moved = moved + stroma_focus * (scale * (s_target - moved))
            # Keep counterfactual local and plausible; avoid forcing background to move with the ROI.
            if cur.dtype.is_floating_point:
                lo = cur.detach().amin(dim=0, keepdim=True) - 0.55 * cur.detach().std(dim=0, keepdim=True).clamp_min(1e-3)
                hi = cur.detach().amax(dim=0, keepdim=True) + 0.55 * cur.detach().std(dim=0, keepdim=True).clamp_min(1e-3)
                moved = torch.maximum(torch.minimum(moved, hi), lo)
            do_zE[:, j:j+1] = moved
        return do_zE

    def _loss_gate_interface_enrichment(
        self,
        gate: torch.Tensor,
        snap: SpatialSnapshot,
        ode_in: Optional[ODEInput] = None,
        *,
        margin: float = 0.02,
        weight: float = 0.5,
    ) -> torch.Tensor:
        if weight <= 0.0:
            return gate.new_zeros(())
        node_gate = getattr(self.model.odefunc, "gate_node_last", None)
        node_activity = node_gate.reshape(-1).float() if isinstance(node_gate, torch.Tensor) else gate.float().mean(dim=1)
        focus = self._gate_focus_mask(snap, ode_in)
        if focus is None:
            return gate.new_zeros(())
        roi_mask = focus.reshape(-1).to(node_activity.device) > 0.5
        if int(roi_mask.sum()) < 8 or int((~roi_mask).sum()) < 8:
            return gate.new_zeros(())
        roi_vals = node_activity[roi_mask]
        bg_vals = node_activity[~roi_mask]
        roi_mean = roi_vals.mean()
        bg_mean = bg_vals.mean()
        roi_top = torch.quantile(roi_vals, 0.70)
        bg_top = torch.quantile(bg_vals, float(getattr(self, "_gate_bg_top_quantile", 0.82)))
        target_margin = gate.new_tensor(float(margin))
        loss_margin = torch.relu((bg_mean + 1.35 * target_margin) - roi_mean)
        loss_rank = torch.relu((bg_top + 1.8 * target_margin) - roi_top)
        loss_bg = torch.relu(bg_mean - gate.new_tensor(0.12))
        loss_budget = torch.relu(node_activity.mean() - gate.new_tensor(0.24))
        select_gap = torch.relu(gate.new_tensor(0.10) - (roi_mean - bg_mean))
        return gate.new_tensor(float(weight)) * (2.1 * loss_margin.pow(2) + 2.6 * loss_rank.pow(2) + 1.2 * loss_bg.pow(2) + 0.35 * loss_budget.pow(2) + 2.4 * select_gap.pow(2))

    def _loss_gate_local_expert(
        self,
        x_with_gate: torch.Tensor,
        x_core: torch.Tensor,
        x_true: torch.Tensor,
        snap: SpatialSnapshot,
        ode_in: Optional[ODEInput] = None,
        *,
        weight: float = 0.0,
        bg_allowance: float = 0.002,
    ) -> torch.Tensor:
        """Constrain Gate to act as a focus-local residual expert, not a global 2nd backbone."""
        if weight <= 0.0:
            return x_with_gate.new_zeros(())
        focus = self._gate_focus_mask(snap, ode_in)
        if focus is None:
            return x_with_gate.new_zeros(())
        roi_mask = focus.reshape(-1).to(x_with_gate.device) > 0.5
        if int(roi_mask.sum()) < 4 or int((~roi_mask).sum()) < 4:
            return x_with_gate.new_zeros(())
        per_with = torch.nn.functional.huber_loss(x_with_gate, x_true, delta=1.0, reduction="none").mean(dim=1)
        per_core = torch.nn.functional.huber_loss(x_core, x_true, delta=1.0, reduction="none").mean(dim=1)
        improve = per_core - per_with
        roi_improve = improve[roi_mask].mean()
        bg_improve = improve[~roi_mask].mean()
        margin = x_with_gate.new_tensor(float(bg_allowance) * 0.5 + 0.001)
        loss_roi = torch.relu(margin - roi_improve)
        loss_bg = torch.relu(bg_improve - x_with_gate.new_tensor(float(bg_allowance)))
        loss_contrast = torch.relu((bg_improve + margin) - roi_improve)
        return x_with_gate.new_tensor(float(weight)) * (loss_roi.pow(2) + 1.5 * loss_bg.pow(2) + loss_contrast.pow(2))

    def _counterfactual_roi_objective(
        self,
        snap: SpatialSnapshot,
        y0: torch.Tensor,
        ode_in: ODEInput,
        t_end: float,
        *,
        weight: float = 0.0,
        margin: float = 0.002,
        drop_scales: Tuple[float, float, float] = (0.45, 0.75, 0.45),
    ) -> torch.Tensor:
        if weight <= 0.0 or t_end <= 0.0:
            return y0.new_zeros(())
        roi = self._compute_interface_roi(snap)
        if roi is None:
            return y0.new_zeros(())
        roi_mask = roi.mask_interface.reshape(-1, 1).to(self.device, dtype=ode_in.zE.dtype)
        if float(roi_mask.sum().item()) < 4.0:
            return y0.new_zeros(())
        residual_info = self._compute_residual_focus(ode_in)
        boost = float(getattr(self, "_cf_intervention_scale", 1.0) or 1.0) * (1.15 + max(0.0, float(weight)))
        if residual_info is not None:
            boost = boost * (1.0 + 0.35 * float(residual_info["high"].mean().item()))
        do_zE = self.make_targeted_do_zE(ode_in, roi_mask, drop_scales, extra_scale=boost, roi=roi)
        ode_in_cf = self._apply_counterfactual_zE(ode_in, do_zE)
        t = self._make_time_grid(float(t_end))
        with torch.no_grad():
            y_base = self._ode_solve(y0.detach(), ode_in, t).detach()
        y_cf = self._ode_solve(y0, ode_in_cf, t)
        p0 = p_cross_interface(y_base, roi)
        p1 = p_cross_interface(y_cf, roi)
        gate_last = getattr(self.model.odefunc, 'gate_last', None)
        if isinstance(gate_last, torch.Tensor):
            act = gate_last.detach().float().mean(dim=1)
            roi_mask_flat = roi.mask_interface.reshape(-1).to(act.device).bool()
            if int(roi_mask_flat.sum()) > 3 and int((~roi_mask_flat).sum()) > 3:
                enrich = torch.relu(act[roi_mask_flat].mean() - act[~roi_mask_flat].mean())
            else:
                enrich = y0.new_zeros(())
        else:
            enrich = y0.new_tensor(1.0)
        eff_w = y0.new_tensor(float(weight)) * (0.55 + 1.35 * enrich.clamp(0.0, 1.0))
        delta = p1 - p0
        target_delta = y0.new_tensor(max(float(getattr(getattr(self, "_stage_cfg", None), "cf_target_delta", 0.05) or 0.05), float(margin) * 6.0 + 0.018))
        roi_idx = roi.mask_interface.reshape(-1).to(y_cf.device)
        roi_shift = (y_cf[roi_idx] - y_base[roi_idx]).pow(2).mean().sqrt()
        bg_mask = (~roi.mask_interface.reshape(-1)).to(y_cf.device)
        if int(bg_mask.sum().item()) > 3:
            bg_shift = (y_cf[bg_mask] - y_base[bg_mask]).pow(2).mean().sqrt()
        else:
            bg_shift = y0.new_tensor(0.0)
        tumor_roi = (roi.mask_interface & roi.side_tumor).reshape(-1).to(y_cf.device)
        stroma_roi = (roi.mask_interface & roi.side_stroma).reshape(-1).to(y_cf.device)
        gap0 = y0.new_tensor(0.0)
        gap1 = y0.new_tensor(0.0)
        contrastive = y0.new_zeros(())
        if int(tumor_roi.sum().item()) > 2 and int(stroma_roi.sum().item()) > 2:
            t_base = y_base[tumor_roi]
            s_base = y_base[stroma_roi]
            t_cf = y_cf[tumor_roi]
            s_cf = y_cf[stroma_roi]
            gap0 = (t_base.mean(dim=0) - s_base.mean(dim=0)).norm(p=2)
            gap1 = (t_cf.mean(dim=0) - s_cf.mean(dim=0)).norm(p=2)
            intra = 0.5 * ((t_cf - t_cf.mean(dim=0, keepdim=True)).pow(2).mean().sqrt() + (s_cf - s_cf.mean(dim=0, keepdim=True)).pow(2).mean().sqrt())
            contrastive = torch.relu(gap0 * 0.90 - gap1) + 0.25 * intra
        mod = float(getattr(getattr(self, "_stage_cfg", None), "cf_gradient_modulation", 1.35) or 1.35)
        contrastive_w = float(getattr(getattr(self, "_stage_cfg", None), "cf_contrastive_weight", 0.18) or 0.18)
        neg_w = float(getattr(getattr(self, "_stage_cfg", None), "cf_negative_delta_weight", 3.0) or 3.0)
        mid_w = float(getattr(getattr(self, "_stage_cfg", None), "cf_mid_delta_weight", 1.4) or 1.4)
        roi_shift_floor = float(getattr(getattr(self, "_stage_cfg", None), "cf_roi_shift_floor", 0.16) or 0.16)
        bg_ratio_cap = float(getattr(getattr(self, "_stage_cfg", None), "cf_bg_ratio_cap", 0.55) or 0.55)
        gap_keep = float(getattr(getattr(self, "_stage_cfg", None), "cf_gap_preserve_ratio", 0.78) or 0.78)
        focus_scale = y0.new_tensor(mod) if residual_info is None else (1.0 + float(mod) * residual_info["high"].mean())
        neg_delta = torch.relu(y0.new_tensor(0.0) - delta)
        mid_delta = torch.relu(0.55 * target_delta - delta)
        safe_gap0 = float(max(gap0.item(), 1e-6)) if isinstance(gap0, torch.Tensor) else max(float(gap0), 1e-6)
        self._last_cf_summary = {
            "delta_p_cross": float(delta.detach().item()),
            "p_cross_baseline": float(p0.detach().item()),
            "p_cross_do_z": float(p1.detach().item()),
            "roi_shift": float(roi_shift.detach().item()),
            "bg_shift": float(bg_shift.detach().item()),
            "bg_ratio": float((bg_shift.detach() / roi_shift.detach().clamp_min(1e-6)).item()),
            "gap0": float(gap0.detach().item()),
            "gap1": float(gap1.detach().item()),
            "gap_ratio": float(gap1.detach().item() / safe_gap0),
        }
        return eff_w * (
            1.10 * torch.relu((p0 + y0.new_tensor(float(margin))) - p1)
            + 2.65 * focus_scale * torch.relu(target_delta - delta)
            + y0.new_tensor(neg_w) * neg_delta.pow(2)
            + y0.new_tensor(mid_w) * mid_delta.pow(2)
            + 0.75 * torch.relu(y0.new_tensor(float(roi_shift_floor)) - roi_shift)
            + 1.10 * torch.relu(bg_shift - y0.new_tensor(float(bg_ratio_cap)) * roi_shift)
            + 1.40 * torch.relu((y0.new_tensor(float(gap_keep)) * gap0) - gap1)
            + y0.new_tensor(contrastive_w) * contrastive
        )

    def _loss_physics_regularizer(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        ode_in: ODEInput,
        *,
        physics_weight: float = 0.0,
        divergence_weight: float = 0.0,
    ) -> torch.Tensor:
        if physics_weight <= 0.0 and divergence_weight <= 0.0:
            return y0.new_zeros(())
        edge_index = ode_in.edge_index
        i, j = edge_index[0], edge_index[1]
        vel = y1 - y0
        if vel.shape[0] < 4 or i.numel() < 4:
            return y0.new_zeros(())
        edge_delta = vel[j] - vel[i]
        smooth = edge_delta.pow(2).mean()
        mag = torch.linalg.norm(vel, dim=1).mean()
        mag_pen = torch.relu(mag - y0.new_tensor(5.0))
        coord_delta = ode_in.coords[j] - ode_in.coords[i]
        coord_norm = coord_delta.norm(dim=1, keepdim=True).clamp_min(1e-4)
        directional = (edge_delta.abs() / coord_norm).mean()
        return y0.new_tensor(float(physics_weight)) * (smooth + 0.10 * mag_pen) + y0.new_tensor(float(divergence_weight)) * directional

    def _split_env_masks(
        self,
        coords: torch.Tensor,
        zE: Optional[torch.Tensor] = None,
        mechanomics: Optional[torch.Tensor] = None,
        annotations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Biology-aware environment partitioning for IRM/ICM constraints.

        Priority order:
          0) Expert/pathology annotations when available (Tumor/Stroma, Interface/Interior).
          1) zE proxy (unsupervised environment score; e.g., malignancy/CAF).
          2) mechanomics proxy (morphometrics; e.g., boundary distance).
          3) x-median fallback (synthetic / unannotated settings).

        irm_split modes:
          - "bio_gt_auto" (recommended): annotations -> zE -> mechanomics -> x_median fallback
          - "annotations": force annotation-based split (fallback to x_median if unusable)
          - "bio_auto": zE -> mechanomics -> x_median fallback (legacy of previous patch)
          - "zE_dim0": force zE[:,0] median split (with balance check, fallback to x_median)
          - "mech_dim0": force mechanomics[:,0] quantile split (with balance check, fallback to x_median)
          - "x_median": geometric split
          - "x_quantile": alias of x_median
        """
        if coords.ndim != 2 or coords.shape[1] < 1:
            raise ValueError(f"coords must be (N,2+) but got {tuple(coords.shape)}")

        def _balanced(env0: torch.Tensor, env1: torch.Tensor, min_frac: float = 0.10) -> bool:
            n = env0.numel()
            if n == 0:
                return False
            return (env0.sum().item() >= min_frac * n) and (env1.sum().item() >= min_frac * n)

        def _x_median_split() -> Tuple[torch.Tensor, torch.Tensor]:
            x = coords[:, 0]
            thr = x.median()
            env0 = x <= thr
            env1 = x > thr
            if env0.sum() == 0 or env1.sum() == 0:
                n = x.shape[0]
                env0 = torch.zeros(n, dtype=torch.bool, device=coords.device)
                env0[: n // 2] = True
                env1 = ~env0
            return env0, env1

        def _split_by_annotations() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            if annotations is None:
                return None
            lab = None
            if annotations.ndim == 1:
                lab = annotations
            elif annotations.ndim == 2 and annotations.shape[1] >= 1:
                lab = annotations[:, 0]
            else:
                return None
            if lab.shape[0] != coords.shape[0]:
                return None

            # If user provided hypothesis-driven mapping: {1,0,-1}, use it directly.
            # -1 are ignored for IRM environment split.
            uniq = torch.unique(lab)
            if (uniq.numel() <= 3) and (torch.isin(uniq, torch.tensor([-1, 0, 1], device=lab.device)).all()):
                env1 = lab == 1
                env0 = lab == 0
                if _balanced(env0, env1, min_frac=0.02):
                    return env0, env1
                return None

            # Otherwise, do NOT median-split categorical codes (biologically meaningless).
            # Fall back to zE/mechanomics/x_median unless the user explicitly wants category-based splits.
            return None
            if annotations.ndim == 1:
                lab = annotations
            elif annotations.ndim == 2 and annotations.shape[1] >= 1:
                lab = annotations[:, 0]
            else:
                return None
            if lab.shape[0] != coords.shape[0]:
                return None

            # if boolean or binary categories
            uniq = torch.unique(lab)
            if lab.dtype == torch.bool or uniq.numel() == 2:
                thr = lab.float().median()
                env1 = lab.float() > thr
                env0 = ~env1
            else:
                thr = lab.median()
                env1 = lab > thr
                env0 = ~env1

            if _balanced(env0, env1, min_frac=0.05):
                return env0, env1
            return None

        def _split_by_zE_dim0() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            if zE is None:
                return None
            if zE.ndim != 2 or zE.shape[0] != coords.shape[0] or zE.shape[1] < 1:
                return None
            s = zE[:, 0]
            thr = s.median()
            env1 = s > thr
            env0 = ~env1
            if _balanced(env0, env1):
                return env0, env1
            return None

        def _split_by_mech_dim0(interface_q: float = 0.30) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            if mechanomics is None:
                return None
            if mechanomics.ndim != 2 or mechanomics.shape[0] != coords.shape[0] or mechanomics.shape[1] < 1:
                return None
            s = mechanomics[:, 0]
            thr = torch.quantile(s, interface_q)
            interface = s <= thr
            interior = ~interface
            if _balanced(interface, interior):
                return interface, interior
            return None

        mode = (self.irm_split or "bio_gt_auto").strip()

        if mode in ("bio_gt_auto", "tumor_stroma", "interface_interior"):
            out = _split_by_annotations()
            if out is not None:
                return out
            out = _split_by_zE_dim0()
            if out is not None:
                return out
            out = _split_by_mech_dim0(interface_q=0.30)
            if out is not None:
                return out
            return _x_median_split()

        if mode == "annotations":
            out = _split_by_annotations()
            return out if out is not None else _x_median_split()

        if mode == "bio_auto":
            out = _split_by_zE_dim0()
            if out is not None:
                return out
            out = _split_by_mech_dim0(interface_q=0.30)
            if out is not None:
                return out
            return _x_median_split()

        if mode == "zE_dim0":
            out = _split_by_zE_dim0()
            return out if out is not None else _x_median_split()

        if mode == "mech_dim0":
            out = _split_by_mech_dim0(interface_q=0.30)
            return out if out is not None else _x_median_split()

        if mode == "x_median":
            return _x_median_split()

        if mode == "x_quantile":
            return _x_median_split()

        raise ValueError(f"Unknown irm_split={self.irm_split}")

    def _irmv1_penalty(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        env0: torch.Tensor,
        env1: torch.Tensor,
    ) -> torch.Tensor:
        """IRMv1 penalty (Arjovsky et al.) with optional class-balance reweighting.

        Standard IRMv1 uses a scalar probe parameter w:
            penalty = sum_e || d/dw L_e(w * x_pred, x_true) ||^2

        Spatial transcriptomics commonly exhibits severe env imbalance (e.g., tumor bulk >> stroma).
        Even with per-env mean losses, gradient magnitudes can be dominated by the majority env due to
        variance/heterogeneity. We therefore optionally apply **balanced weighting**:
            penalty = 0.5 * ||g_env0||^2 + 0.5 * ||g_env1||^2
        ensuring rare microenvironments exert equal causal regularization force.

        Note: env masks should already exclude ignore_index (-1) cells.
        """
        # scalar w to probe invariance
        w = torch.tensor(1.0, device=x_pred.device, requires_grad=True)

        def env_loss(mask: torch.Tensor) -> torch.Tensor:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=x_pred.device)
            xp = x_pred[mask]
            xt = x_true[mask]
            # first-line defense: mean (not sum) prevents pure sample-count domination
                        # sample-level robust loss: compute per-sample Huber across genes then mean
            per = torch.nn.functional.huber_loss(w * xp, xt, delta=1.0, reduction="none").mean(dim=1)
            return per.mean()

        l0 = env_loss(env0)
        l1 = env_loss(env1)

        g0 = torch.autograd.grad(l0, [w], create_graph=True)[0]
        g1 = torch.autograd.grad(l1, [w], create_graph=True)[0]

        p0 = g0.pow(2)
        p1 = g1.pow(2)

        if self.irm_balance:
            penalty = 0.5 * p0 + 0.5 * p1
        else:
            penalty = p0 + p1
        return penalty

    def fit(self, dataset: SpatialDataset, cfg: TrainCfg) -> Dict[str, float]:
        self._gate_bg_top_quantile = float(max(0.55, min(0.98, getattr(cfg, "gate_bg_top_quantile", 0.82) or 0.82)))
        """Train with strictly Markovian pairwise time evolution (t -> t+1).

        - No hub-and-spoke alignment (forbidden).
        - OT coupling Π is updated in an EM-like fashion:
          E-step (every update_freq epochs): compute Π using denoised latent proxies (Z_I,Z_N) with no-grad.
          M-step: treat Π as detached constant and optimize ODE parameters.
        """
        os.makedirs(cfg.save_dir, exist_ok=True)
        self.configure_optim(lr=cfg.lr, weight_decay=cfg.weight_decay, mms_lr_mult=float(getattr(cfg, "mms_lr_mult", 2.0) or 2.0))
        self._cf_intervention_scale = float(getattr(cfg, "cf_intervention_scale", 1.0) or 1.0)
        self.scheduler = None
        if int(getattr(cfg, "scheduler_t0", 0) or 0) > 0:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.opt,
                T_0=int(getattr(cfg, "scheduler_t0", 50) or 50),
                T_mult=int(getattr(cfg, "scheduler_t_mult", 1) or 1),
            )

        if len(dataset.snapshots) < 2:
            raise ValueError("Need at least 2 snapshots/slices for temporal training.")

        # Pairwise consecutive (strictly Markovian)
        pairs = [(dataset.snapshots[i], dataset.snapshots[i + 1]) for i in range(len(dataset.snapshots) - 1)]

        # Build atlas stats from the earliest slice
        ode_in0 = self._build_ode_input(pairs[0][0], stage="stage0")
        with torch.no_grad():
            self.zE_mu, self.zE_sd = _standardize_z(ode_in0.zE)

        history = {"final_loss": float("nan")}

        for epoch in tqdm(range(cfg.epochs), desc="train"):
            self._last_cf_summary = {}
            self.model.train()

            # Stage schedule: Stage-0 Gate=0, Stage-1 localized gate learning, Stage-2 sparse fine-tune.
            # If we detect NaN/Inf instability, we temporarily disable MMS to recover (training sentry).
            stage = self._current_stage(epoch, cfg)
            stage_offset = int(getattr(cfg, "stage0_epochs", 0) or 0)
            if stage == "stage2":
                stage_offset += int(getattr(cfg, "stage1_epochs", 0) or 0)
            stage_epoch = epoch if stage == "stage0" else max(0, epoch - stage_offset)
            self._active_stage = str(stage)
            self._active_stage_epoch = int(stage_epoch)
            self._stage_cfg = cfg
            self._roi_target_frac = float(self._current_roi_target_frac())
            self._apply_stage_learning_rates(stage, cfg)
            if stage == "stage2":
                self._capture_stage2_gate_reference()
                gate_freeze_epochs = int(getattr(cfg, "stage2_gate_freeze_epochs", 0) or 0)
                base_freeze_epochs = int(getattr(cfg, "stage2_base_freeze_epochs", 0) or 0)
                mms_freeze_epochs = int(getattr(cfg, "stage2_mms_freeze_epochs", 0) or 0)
                for pg in self.opt.param_groups:
                    name = str(pg.get("name", ""))
                    if name == "gate" and stage_epoch < gate_freeze_epochs:
                        pg["lr"] = 0.0
                    elif name == "base" and stage_epoch < base_freeze_epochs:
                        pg["lr"] = 0.0
                    elif name == "mms" and stage_epoch < mms_freeze_epochs:
                        pg["lr"] = 0.0
            if hasattr(self.model.odefunc, "gate") and hasattr(self.model.odefunc.gate, "set_temperature"):
                if stage == "stage0":
                    self.model.odefunc.gate.set_temperature(float(getattr(cfg, "stage0_gate_temperature", 1.9) or 1.9))
                elif stage == "stage1":
                    self.model.odefunc.gate.set_temperature(float(getattr(cfg, "stage1_gate_temperature", 1.6) or 1.6))
                elif stage == "stage2":
                    self.model.odefunc.gate.set_temperature(float(getattr(cfg, "stage2_gate_temperature", 1.1) or 1.1))
            disable_until = int(getattr(self, "_nan_disable_mms_until", -1))
            if stage == "stage0" or (disable_until >= 0 and epoch < disable_until):
                self.model.odefunc.enable_mms = False
            else:
                self.model.odefunc.enable_mms = True
            gate_l1_scale, gate_tv_scale = self._stage_gate_scales(stage, cfg)

            # -------------------------
            # E-step: update Π on latent space every update_freq epochs
            # -------------------------
            update_freq = int(getattr(self.ot_cfg, "update_freq", 5)) if self.ot_cfg is not None else 5
            if self.enable_ot and self.ot_cfg is not None:
                need_update = (epoch - int(getattr(self.model, "last_pi_update_epoch", -10**9))) >= update_freq
                if need_update or len(getattr(self.model, "cached_pi", [])) != len(pairs):
                    self.model.eval()
                    cached_pi = []
                    with torch.no_grad():
                        for (snap0, snap1) in pairs:
                            x0 = snap0.X.to(self.device)
                            x1 = snap1.X.to(self.device)

                            zi0, zn0 = self.model.encode_zi_zn(x0)
                            zi1, zn1 = self.model.encode_zi_zn(x1)

                            Pi = compute_pi_pot_sparse_knn(
                                z_I_0=zi0,
                                z_N_0=zn0,
                                coords_0=snap0.coords.to(self.device),
                                z_I_1=zi1,
                                z_N_1=zn1,
                                coords_1=snap1.coords.to(self.device),
                                k=int(getattr(self.ot_cfg, "knn_k", 50)),
                                alpha=float(getattr(self.ot_cfg, "alpha", 0.5)),
                                epsilon=float(self.ot_cfg.epsilon),
                                max_iter=int(self.ot_cfg.max_iter),
                                a=snap0.mass.to(self.device) if snap0.mass is not None else None,
                                b=snap1.mass.to(self.device) if snap1.mass is not None else None,
                            )
                            cached_pi.append(Pi)

                    self.model.cached_pi = cached_pi
                    self.model.last_pi_update_epoch = epoch
                    self.model.train()
                    logger.info(
                        "E-step: updated Π for %d consecutive pairs at epoch %d (update_freq=%d).",
                        len(pairs),
                        epoch,
                        update_freq,
                    )
            else:
                self.model.cached_pi = []
                self.model.last_pi_update_epoch = -10**9

            # -------------------------
            # M-step: optimize θ with Π detached
            # -------------------------
            # Backward through adjoint ODE can occasionally hit torchdiffeq's step cap.
            # We keep the *core algorithm* unchanged and only increase the adjoint step budget,
            # retrying a single time for the current optimizer step.
            for _attempt in range(2):
                self.opt.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=self.device, dtype=torch.float32)

                for pair_idx, (snap0, snap1) in enumerate(pairs):
                    ode_in0 = self._build_ode_input(snap0, stage=stage)
                    t = self._make_time_grid(float(snap1.time - snap0.time))

                    x0 = snap0.X.to(self.device)
                    x1 = snap1.X.to(self.device)

                    # Optional: environment-balanced reconstruction (Tumor vs Stroma) to avoid majority env domination.
                    env0_recon = env1_recon = None
                    if getattr(self, "balance_recon", False):
                        ann0 = getattr(snap0, "annotations", None)
                        if isinstance(ann0, torch.Tensor):
                            ann0 = ann0.to(self.device)
                        try:
                            env0_recon, env1_recon = self._split_env_masks(
                                ode_in0.coords,
                                zE=ode_in0.zE,
                                mechanomics=ode_in0.mechanomics,
                                annotations=ann0,
                            )
                        except Exception:
                            env0_recon = env1_recon = None

                    y0 = self.model.encode(x0)
                    y1_hat = self._ode_solve(y0, ode_in0, t)  # (N, d_latent)
                    x1_hat = self.model.decode(y1_hat)

                    # primary reconstruction/prediction loss
                    # NOTE: In pseudo-time mode, consecutive snapshots can have slightly different cell counts.
                    # For better model stability/performance, if counts differ we compute the reconstruction loss
                    # on the OT-transported prediction (Π^T @ x1_hat) instead of naive row-wise alignment.
                    if x1_hat.shape[0] != x1.shape[0]:
                        if self.enable_ot and hasattr(self.model, "cached_pi") and pair_idx < len(self.model.cached_pi):
                            Pi_x = self.model.cached_pi[pair_idx].detach()
                            # Align to Π dimensions to avoid matmul shape errors.
                            if Pi_x.shape[0] != x1_hat.shape[0]:
                                x1_hat = x1_hat[: int(Pi_x.shape[0])]
                            if Pi_x.shape[1] != x1.shape[0]:
                                x1 = x1[: int(Pi_x.shape[1])]
                            logger.warning(
                                "Recon N mismatch (pred=%d, true=%d). Using OT-transported recon loss (patched_v12).",
                                int(x1_hat.shape[0]),
                                int(x1.shape[0]),
                            )
                            loss = ot_alignment_loss(y_src_pred=x1_hat, y_tgt_obs=x1, coupling=Pi_x)
                        else:
                            m = int(min(x1_hat.shape[0], x1.shape[0]))
                            logger.warning(
                                "Recon N mismatch (pred=%d, true=%d). Truncating to %d for recon loss (patched_v12).",
                                int(x1_hat.shape[0]),
                                int(x1.shape[0]),
                                m,
                            )
                            loss = self._loss_recon_balanced(x1_hat[:m], x1[:m], env0_recon[:m], env1_recon[:m]) if (env0_recon is not None and env1_recon is not None) else self._loss_recon(x1_hat[:m], x1[:m])
                    else:
                        loss = self._loss_recon_balanced(x1_hat, x1, env0_recon, env1_recon) if (env0_recon is not None and env1_recon is not None) else self._loss_recon(x1_hat, x1)

                    # IRM/ICM-style causality constraint (factual-only): enforce invariant *core* predictor across biological environments
                    # Key design to avoid suppressing legitimate biological variance:
                    #   - Apply IRM penalty on the conservative component f_core (disable MMS modulation temporarily).
                    #   - Environment-specific differences are absorbed by MMS (f_mod) via Gate, consistent with MMS-ODE story.
                    if self.irm_lambda > 0.0 and epoch >= self.irm_warmup_epochs and ((not bool(getattr(cfg, "irm_stage2_only", True))) or stage == "stage2"):
                        # Build environments using annotations (priority 0), then zE/mechanomics fallbacks
                        ann = getattr(snap0, "annotations", None)
                        if isinstance(ann, torch.Tensor):
                            ann = ann.to(self.device)
                        env0, env1 = self._split_env_masks(
                            ode_in0.coords,
                            zE=ode_in0.zE,
                            mechanomics=ode_in0.mechanomics,
                            annotations=ann,
                        )

                        # Compute core-only prediction by disabling MMS modulation
                        old_mms = bool(self.model.odefunc.enable_mms)
                        self.model.odefunc.enable_mms = False
                        with torch.no_grad():
                            y1_hat_core = self._ode_solve(y0.detach(), ode_in0, t)
                            x1_hat_core = self.model.decode(y1_hat_core).detach()
                        self.model.odefunc.enable_mms = old_mms

                        m_irm = int(min(x1_hat_core.shape[0], x1.shape[0], env0.shape[0], env1.shape[0]))
                        if (
                            x1_hat_core.shape[0] != m_irm
                            or x1.shape[0] != m_irm
                            or env0.shape[0] != m_irm
                            or env1.shape[0] != m_irm
                        ):
                            logger.warning(
                                "IRM N mismatch (pred=%d, true=%d, env0=%d, env1=%d). Truncating to %d for IRM penalty.",
                                int(x1_hat_core.shape[0]),
                                int(x1.shape[0]),
                                int(env0.shape[0]),
                                int(env1.shape[0]),
                                m_irm,
                            )

                        irm_pen = self._irmv1_penalty(
                            x_pred=x1_hat_core[:m_irm],
                            x_true=x1[:m_irm],
                            env0=env0[:m_irm],
                            env1=env1[:m_irm],
                        )
                        loss = loss + self.irm_lambda * irm_pen

                    phys_w = float(getattr(cfg, "physics_weight", 0.0) or 0.0)
                    div_w = float(getattr(cfg, "velocity_divergence_weight", 0.0) or 0.0)
                    if phys_w > 0.0 or div_w > 0.0:
                        loss = loss + self._loss_physics_regularizer(y0, y1_hat, ode_in0, physics_weight=phys_w, divergence_weight=div_w)

                    # OT alignment (optional) uses detached Π from E-step
                    if self.enable_ot and self.ot_cfg is not None and hasattr(self.model, "cached_pi") and pair_idx < len(self.model.cached_pi):
                        Pi = self.model.cached_pi[pair_idx].detach()
                        loss = loss + float(getattr(self.ot_cfg, "loss_weight", 0.5)) * ot_alignment_loss(
                            y_src_pred=y1_hat,
                            y_tgt_obs=self.model.encode(x1),
                            coupling=Pi,
                        )

                    # Stage-0 gate warmup: supervise the gate directly before MMS is enabled.
                    if stage == "stage0" and hasattr(self.model.odefunc, "gate"):
                        gate0 = self.model.odefunc.gate(ode_in0.zN, sample=self.model.training)
                        self.model.odefunc.gate_raw_last = gate0
                        self.model.odefunc.gate_last = gate0
                        self.model.odefunc.gate_p_nonzero_last = self.model.odefunc.gate.last_p_nonzero
                        self.model.odefunc.gate_node_last = self.model.odefunc.gate.last_gate_node
                        gate0_open = self._loss_gate_activity_floor(
                            gate0, ode_in0,
                            target=float(getattr(cfg, "stage0_gate_open_target", 0.20) or 0.20),
                            weight=float(getattr(cfg, "stage0_gate_open_weight", 1.2) or 1.2),
                        )
                        gate0_iface = self._loss_gate_interface_enrichment(
                            gate0, snap0,
                            margin=float(getattr(cfg, "stage0_gate_interface_margin", 0.06) or 0.06),
                            weight=float(getattr(cfg, "stage0_gate_interface_weight", 1.0) or 1.0),
                        )
                        gate0_bce = self._loss_gate_interface_bce(
                            gate0, snap0, ode_in0,
                            weight=float(getattr(cfg, "stage0_gate_bce_weight", 1.6) or 1.6),
                            entropy_weight=float(getattr(cfg, "stage0_gate_entropy_weight", 0.05) or 0.05),
                        )
                        gate0_hbg = self._loss_gate_hard_background(gate0, snap0, ode_in0, weight=float(getattr(cfg, "stage1_gate_hard_bg_weight", 0.0) or 0.0), top_quantile=self._gate_bg_top_quantile)
                        gate0_rank = self._loss_gate_rank_contrast(gate0, snap0, ode_in0, weight=float(getattr(cfg, "stage1_gate_rank_weight", 0.0) or 0.0))
                        loss = loss + gate0_open + gate0_iface + gate0_bce + gate0_hbg + gate0_rank
                        history.setdefault("gate_stage0_open_loss", []).append(float(gate0_open.detach().item()))
                        history.setdefault("gate_stage0_bce_loss", []).append(float(gate0_bce.detach().item()))

                    # Gate regularizers (Stage-1 focuses ROI via residual; core logic lives in model/diagnostics)
                    if self.model.odefunc.enable_mms:
                        gate = self.model.odefunc.gate_last
                        if gate is not None and (gate_l1_scale > 0.0 or gate_tv_scale > 0.0):
                            burn = int(getattr(cfg, f"{stage}_gate_burnin_epochs", 0) or 0) if stage in {"stage1", "stage2"} else 0
                            ramp = int(getattr(cfg, f"{stage}_gate_reg_ramp_epochs", 1) or 1) if stage in {"stage1", "stage2"} else 1
                            if stage in {"stage1", "stage2"} and stage_epoch < burn:
                                reg_scale = 0.0
                            else:
                                reg_prog = max(0.0, float(stage_epoch - burn + 1)) / float(max(1, ramp))
                                reg_scale = float(min(1.0, reg_prog))
                            gate_weights = self._gate_background_weights(ode_in0)
                            if isinstance(gate_weights, torch.Tensor) and gate_weights.numel() == gate.shape[0]:
                                focus = self._gate_focus_mask(snap0, ode_in0)
                                if isinstance(focus, torch.Tensor) and focus.numel() == gate.shape[0]:
                                    focus = focus.reshape(-1).to(gate_weights.device, dtype=gate_weights.dtype).clamp(0.0, 1.0)
                                    if stage == "stage2":
                                        gate_weights = (0.12 + 0.88 * (1.0 - 0.85 * focus) * gate_weights).clamp(min=0.05, max=1.25)
                                    else:
                                        gate_weights = torch.minimum(gate_weights, (0.20 + 0.80 * (1.0 - 0.92 * focus)).clamp(min=0.03, max=1.0))
                            loss = loss + self._loss_gate_regularizers(
                                gate,
                                ode_in0.edge_index,
                                l1_scale=gate_l1_scale * reg_scale,
                                tv_scale=gate_tv_scale * reg_scale,
                                node_weights=gate_weights,
                            )
                        if gate is not None and stage in {"stage1", "stage2"}:
                            # Ensure a core-only prediction is always available for local-Gate losses,
                            # even when IRM is disabled or still warming up.
                            if "x1_hat_core" not in locals():
                                old_mms_local = bool(self.model.odefunc.enable_mms)
                                self.model.odefunc.enable_mms = False
                                with torch.no_grad():
                                    y1_hat_core = self._ode_solve(y0.detach(), ode_in0, t)
                                    x1_hat_core = self.model.decode(y1_hat_core).detach()
                                self.model.odefunc.enable_mms = old_mms_local
                            open_target = float(getattr(cfg, f"{stage}_gate_open_target", 0.0) or 0.0)
                            open_weight = float(getattr(cfg, f"{stage}_gate_open_weight", 0.0) or 0.0)
                            gate_open_loss = self._loss_gate_activity_floor(
                                gate, ode_in0, target=open_target, weight=open_weight
                            )
                            iface_margin = float(getattr(cfg, f"{stage}_gate_interface_margin", 0.0) or 0.0)
                            iface_weight = float(getattr(cfg, f"{stage}_gate_interface_weight", 0.0) or 0.0)
                            iface_burn = int(getattr(cfg, f"{stage}_gate_interface_burnin_epochs", 0) or 0)
                            if stage_epoch < iface_burn:
                                iface_weight_eff = 0.0
                            else:
                                iface_ramp = int(getattr(cfg, f"{stage}_gate_interface_ramp_epochs", 1) or 1)
                                iface_prog = max(0.0, float(stage_epoch - iface_burn + 1)) / float(max(1, iface_ramp))
                                iface_weight_eff = iface_weight * float(min(1.0, iface_prog))
                            gate_iface_loss = self._loss_gate_interface_enrichment(
                                gate, snap0, ode_in0, margin=iface_margin, weight=iface_weight_eff
                            )
                            gate_local_weight = float(getattr(cfg, f"{stage}_gate_local_weight", 0.0) or 0.0)
                            local_burn = int(getattr(cfg, f"{stage}_gate_local_burnin_epochs", 0) or 0)
                            if stage_epoch < local_burn:
                                gate_local_weight_eff = 0.0
                            else:
                                local_ramp = int(getattr(cfg, f"{stage}_gate_local_ramp_epochs", 1) or 1)
                                local_prog = max(0.0, float(stage_epoch - local_burn + 1)) / float(max(1, local_ramp))
                                gate_local_weight_eff = gate_local_weight * float(min(1.0, local_prog))
                            gate_bg_allowance = float(getattr(cfg, f"{stage}_gate_bg_allowance", 0.002) or 0.0)
                            gate_local_loss = self._loss_gate_local_expert(
                                x1_hat, x1_hat_core, x1, snap0, ode_in0,
                                weight=gate_local_weight_eff, bg_allowance=gate_bg_allowance,
                            )
                            gate_bce_weight = float(getattr(cfg, f"{stage}_gate_bce_weight", 0.0) or 0.0)
                            gate_ent_weight = float(getattr(cfg, f"{stage}_gate_entropy_weight", 0.0) or 0.0)
                            gate_bce_loss = self._loss_gate_interface_bce(gate, snap0, ode_in0, weight=gate_bce_weight, entropy_weight=gate_ent_weight)
                            gate_hard_bg_weight = float(getattr(cfg, f"{stage}_gate_hard_bg_weight", 0.0) or 0.0)
                            gate_rank_weight = float(getattr(cfg, f"{stage}_gate_rank_weight", 0.0) or 0.0)
                            gate_hard_bg_loss = self._loss_gate_hard_background(gate, snap0, ode_in0, weight=gate_hard_bg_weight, top_quantile=self._gate_bg_top_quantile)
                            gate_rank_loss = self._loss_gate_rank_contrast(gate, snap0, ode_in0, weight=gate_rank_weight)
                            try:
                                p_nonzero = getattr(self.model.odefunc, "gate_p_nonzero_last", None)
                                act_src = p_nonzero if isinstance(p_nonzero, torch.Tensor) and p_nonzero.shape == gate.shape else gate
                                gate_act_scalar = float(act_src.detach().float().mean(dim=1).mean().item())
                            except Exception:
                                gate_act_scalar = 0.0
                            ema_key = f"_{stage}_gate_activity_ema"
                            prev_ema = float(getattr(self, ema_key, gate_act_scalar) or gate_act_scalar)
                            new_ema = 0.90 * prev_ema + 0.10 * gate_act_scalar
                            setattr(self, ema_key, new_ema)
                            surv_w = float(getattr(cfg, f"{stage}_gate_survival_weight", 0.0) or 0.0)
                            surv_ratio = float(getattr(cfg, "gate_survival_floor_ratio", 0.65) or 0.65)
                            surv_target = new_ema * surv_ratio if stage_epoch >= max(iface_burn, 10) else 0.0
                            gate_survival_loss = self._loss_gate_survival(gate, surv_target, weight=surv_w)
                            gate_anchor_loss = gate.new_zeros(())
                            if stage == "stage2":
                                anchor_w = float(getattr(cfg, "stage2_gate_param_anchor_weight", 0.0) or 0.0)
                                anchor_ramp = int(getattr(cfg, "stage2_gate_param_anchor_ramp_epochs", 1) or 1)
                                freeze_epochs = int(getattr(cfg, "stage2_gate_freeze_epochs", 0) or 0)
                                if stage_epoch >= freeze_epochs and anchor_w > 0.0:
                                    prog = max(0.0, float(stage_epoch - freeze_epochs + 1)) / float(max(1, anchor_ramp))
                                    gate_anchor_loss = self._loss_gate_param_anchor(weight=anchor_w * float(min(1.0, prog)))
                            loss = loss + gate_open_loss + gate_iface_loss + gate_local_loss + gate_bce_loss + gate_hard_bg_loss + gate_rank_loss + gate_survival_loss + gate_anchor_loss
                            try:
                                p_nonzero = getattr(self.model.odefunc, "gate_p_nonzero_last", None)
                                gate_raw = getattr(self.model.odefunc, "gate_raw_last", None)
                                act_src = p_nonzero if isinstance(p_nonzero, torch.Tensor) and p_nonzero.shape == gate.shape else (gate_raw if isinstance(gate_raw, torch.Tensor) and gate_raw.shape == gate.shape else gate)
                                node_act = act_src.detach().float().mean(dim=1)
                                history.setdefault("gate_activity_mean", []).append(float(node_act.mean().item()))
                                mms_mask = getattr(ode_in0, "mms_mask", None)
                                if isinstance(mms_mask, torch.Tensor):
                                    mm = mms_mask.reshape(-1).to(node_act.device).bool()
                                    if int(mm.sum()) > 0:
                                        history.setdefault("gate_roi_mean", []).append(float(node_act[mm].mean().item()))
                                    if int((~mm).sum()) > 0:
                                        history.setdefault("gate_bg_mean", []).append(float(node_act[~mm].mean().item()))
                                roi_obj = self._compute_interface_roi(snap0)
                                if roi_obj is not None:
                                    rm = roi_obj.mask_interface.reshape(-1).to(node_act.device).bool()
                                    if int(rm.sum()) > 0 and int((~rm).sum()) > 0:
                                        history.setdefault("gate_interface_mean", []).append(float(node_act[rm].mean().item()))
                                        history.setdefault("gate_non_interface_mean", []).append(float(node_act[~rm].mean().item()))
                                history.setdefault("gate_open_loss", []).append(float(gate_open_loss.detach().item()))
                                history.setdefault("gate_iface_loss", []).append(float(gate_iface_loss.detach().item()))
                                history.setdefault("gate_local_loss", []).append(float(gate_local_loss.detach().item()))
                                history.setdefault("gate_hard_bg_loss", []).append(float(gate_hard_bg_loss.detach().item()))
                                history.setdefault("gate_rank_loss", []).append(float(gate_rank_loss.detach().item()))
                                history.setdefault("gate_survival_loss", []).append(float(gate_survival_loss.detach().item()))
                                history.setdefault("gate_anchor_loss", []).append(float(gate_anchor_loss.detach().item()))
                            except Exception:
                                pass
                        if stage == "stage2":
                            cf_w = float(getattr(cfg, "stage2_cf_roi_weight", 0.0) or 0.0)
                            cf_m = float(getattr(cfg, "stage2_cf_roi_margin", 0.0) or 0.0)
                            cf_sc = tuple(getattr(cfg, "stage2_cf_drop_scales", (0.15, 0.25, 0.15)) or (0.15, 0.25, 0.15))
                            # Use the same factual pair time delta used to build `t` above.
                            # Previous patch referenced undefined names `t0`/`t1`, causing a stage2 crash.
                            t_end_cf = float(max(1e-6, float(snap1.time - snap0.time)))
                            cf_loss = self._counterfactual_roi_objective(
                                snap0, y0, ode_in0, t_end_cf,
                                weight=cf_w, margin=cf_m, drop_scales=cf_sc,
                            )
                            loss = loss + cf_loss
                            history.setdefault("cf_roi_loss", []).append(float(cf_loss.detach().item()))
                            cf_diag = dict(getattr(self, "_last_cf_summary", {}) or {})
                            if cf_diag:
                                history.setdefault("cf_delta_p_cross", []).append(float(cf_diag.get("delta_p_cross", 0.0)))
                                history.setdefault("cf_roi_shift", []).append(float(cf_diag.get("roi_shift", 0.0)))
                                history.setdefault("cf_bg_shift", []).append(float(cf_diag.get("bg_shift", 0.0)))
                                history.setdefault("cf_bg_ratio", []).append(float(cf_diag.get("bg_ratio", 0.0)))
                                history.setdefault("cf_gap_ratio", []).append(float(cf_diag.get("gap_ratio", 1.0)))

                    total_loss = total_loss + loss

                total_loss = total_loss / float(max(1, len(pairs)))
                # -------------------------
                # Training sentry: detect NaN/Inf early and recover (doc: isfinite + auto-lr + disable Gate)
                # -------------------------
                if not check_tensor_finite(total_loss, "total_loss"):
                    logger.warning("Non-finite loss at epoch %d (loss=%s). Skipping optimizer step, reducing LR, disabling MMS temporarily.", int(epoch), str(total_loss.detach().cpu()))
                    # Reduce LR
                    self._base_lr = float(max(getattr(self, "_base_lr", 1e-6) * 0.5, 1e-6))
                    self._apply_stage_learning_rates(stage, cfg)
                    # Temporarily disable MMS to recover
                    self._nan_disable_mms_until = int(max(getattr(self, "_nan_disable_mms_until", -1), epoch + 50))
                    # Clear grads and skip this epoch's step
                    self.opt.zero_grad(set_to_none=True)
                    history["final_loss"] = float("nan")
                    break

                try:
                    total_loss.backward()
                    break
                except AssertionError as e:
                    if "max_num_steps exceeded" in str(e) and _attempt == 0:
                        base = int(getattr(self, "_runtime_adjoint_steps", 0) or 0)
                        bump = max(base, int(getattr(self.solver_cfg, "max_steps", 2000)) * 50)
                        self._runtime_adjoint_steps = int(min(max(bump, 1), 500000))
                        logger.warning(
                            "Backward adjoint hit step cap; retrying once with adjoint max_num_steps=%d.",
                            int(self._runtime_adjoint_steps),
                        )
                        continue
                    raise

            # Clear runtime bump so later epochs use default budget unless needed again
            self._runtime_adjoint_steps = 0

            # Gradient sanity check (avoid propagating NaN/Inf)
            bad_grad = False
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                logger.warning("Non-finite gradient at epoch %d. Skipping step, reducing LR, disabling MMS temporarily.", int(epoch))
                self._base_lr = float(max(getattr(self, "_base_lr", 1e-6) * 0.5, 1e-6))
                self._apply_stage_learning_rates(stage, cfg)
                self._nan_disable_mms_until = int(max(getattr(self, "_nan_disable_mms_until", -1), epoch + 50))
                self.opt.zero_grad(set_to_none=True)
                continue

            safe_clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()

            history["final_loss"] = float(total_loss.detach().cpu())
            if (epoch + 1) % cfg.log_every == 0:
                gate_act = history.get("gate_activity_mean", [])
                gate_roi = history.get("gate_roi_mean", [])
                gate_bg = history.get("gate_bg_mean", [])
                gate_grad_norm = 0.0
                for n, p in self.model.named_parameters():
                    if "gate" in n and p.grad is not None:
                        gate_grad_norm += float(p.grad.norm().detach().item())
                logger.info(
                    "Epoch %d | stage=%s | loss=%.6f | pairs=%d | MMS=%s | IRM=%.2e (%s) | gate=%.4f roi=%.4f bg=%.4f grad=%.3e",
                    epoch,
                    stage,
                    history["final_loss"],
                    len(pairs),
                    str(self.model.odefunc.enable_mms),
                    self.irm_lambda,
                    self.irm_split,
                    float(gate_act[-1]) if gate_act else 0.0,
                    float(gate_roi[-1]) if gate_roi else 0.0,
                    float(gate_bg[-1]) if gate_bg else 0.0,
                    float(gate_grad_norm),
                )


            # -------------------------
            # Checkpointing + early stopping (training-loss based)
            # -------------------------
            if not hasattr(self, "_best_loss"):
                self._best_loss = float("inf")
                self._best_score = float("inf")
                self._best_gate_score = float("inf")
                self._best_stage1_gate_score = float("inf")
                self._best_main_claim_score = float("inf")
                self._best_stage1_epoch = None
                self._best_gate_epoch = None
                self._best_main_claim_epoch = None
                self._no_improve = 0

            cur = float(history["final_loss"])
            ckpt_score = self._checkpoint_objective(cfg, history, stage, cur)
            gate_act_now = float(history.get("gate_activity_mean", [0.0])[-1] if history.get("gate_activity_mean") else 0.0)
            gate_roi_now = float(history.get("gate_roi_mean", [gate_act_now])[-1] if history.get("gate_roi_mean") else gate_act_now)

            # Early stopping / best-ckpt selection should start once stage-1 has meaningfully begun.
            start_epoch = int(getattr(cfg, "early_stop_start_epoch", 0) or 0)
            if start_epoch <= 0:
                start_epoch = int(getattr(cfg, "stage0_epochs", 0) or 0) + 10
            stage1_best_start = int(getattr(cfg, "stage1_best_start_epoch", 0) or 0)
            if stage1_best_start <= 0:
                stage1_best_start = int(getattr(cfg, "stage0_epochs", 0) or 0) + 10

            improved = False
            if epoch >= start_epoch:
                improved = (self._best_score - ckpt_score) > float(getattr(cfg, "early_stop_min_delta", 0.0))
                if improved:
                    self._best_score = ckpt_score
                    self._best_loss = min(float(self._best_loss), cur)
                    self._no_improve = 0
                    best_path = os.path.join(cfg.save_dir, "model_best.pt")
                    torch.save({"state_dict": self.model.state_dict(), "epoch": int(epoch), "loss": cur, "score": float(ckpt_score), "stage": str(stage), "config": getattr(cfg, '__dict__', {})}, best_path)
                else:
                    self._no_improve += 1

            if epoch >= stage1_best_start and stage != "stage0":
                gate_ckpt_score = ckpt_score - 0.55 * gate_act_now - 0.20 * gate_roi_now
                if (self._best_gate_score - gate_ckpt_score) > float(getattr(cfg, "early_stop_min_delta", 0.0)):
                    self._best_gate_score = gate_ckpt_score
                    self._best_gate_epoch = int(epoch)
                    best_gate_path = os.path.join(cfg.save_dir, "model_best_gate.pt")
                    torch.save({"state_dict": self.model.state_dict(), "epoch": int(epoch), "loss": cur, "score": float(gate_ckpt_score), "stage": str(stage), "config": getattr(cfg, '__dict__', {})}, best_gate_path)

            main_claim_start = int(getattr(cfg, "checkpoint_main_claim_start_epoch", 0) or 0)
            if main_claim_start <= 0:
                main_claim_start = stage1_best_start
            if epoch >= main_claim_start and stage != "stage0":
                main_claim_score = self._main_claim_ckpt_score(cfg, history, stage, cur)
                if (self._best_main_claim_score - main_claim_score) > float(getattr(cfg, "early_stop_min_delta", 0.0)):
                    self._best_main_claim_score = main_claim_score
                    self._best_main_claim_epoch = int(epoch)
                    main_claim_path = os.path.join(cfg.save_dir, "model_main_claim_best.pt")
                    torch.save({"state_dict": self.model.state_dict(), "epoch": int(epoch), "loss": cur, "score": float(main_claim_score), "stage": str(stage), "config": getattr(cfg, '__dict__', {}), "cf_summary": dict(getattr(self, "_last_cf_summary", {}) or {})}, main_claim_path)

            if stage == "stage1" and epoch >= stage1_best_start:
                stage1_score = self._stage1_gate_ckpt_score(history, cur)
                if (self._best_stage1_gate_score - stage1_score) > float(getattr(cfg, "early_stop_min_delta", 0.0)):
                    self._best_stage1_gate_score = stage1_score
                    self._best_stage1_epoch = int(epoch)
                    stage1_path = os.path.join(cfg.save_dir, "model_stage1_best.pt")
                    torch.save({"state_dict": self.model.state_dict(), "epoch": int(epoch), "loss": cur, "score": float(stage1_score), "stage": str(stage), "config": getattr(cfg, '__dict__', {})}, stage1_path)

            if stage == "stage1" and (epoch + 1) == int(getattr(cfg, "stage0_epochs", 0) or 0) + int(getattr(cfg, "stage1_epochs", 0) or 0):
                stage1_last_path = os.path.join(cfg.save_dir, "model_stage1_last.pt")
                torch.save({"state_dict": self.model.state_dict(), "epoch": int(epoch), "loss": cur, "score": float(ckpt_score), "stage": str(stage), "config": getattr(cfg, '__dict__', {})}, stage1_last_path)

            self._save_stage_snapshot(cfg, stage, epoch, ckpt_score)
            self._save_training_manifest(cfg, {
                "epoch": int(epoch),
                "stage": str(stage),
                "current_loss": cur,
                "current_gate_activity": gate_act_now,
                "current_gate_roi": gate_roi_now,
                "best_score": float(self._best_score),
                "best_gate_score": float(self._best_gate_score),
                "best_gate_epoch": self._best_gate_epoch,
                "best_stage1_gate_score": float(self._best_stage1_gate_score),
                "best_main_claim_score": float(self._best_main_claim_score),
                "best_stage1_epoch": self._best_stage1_epoch,
                "best_main_claim_epoch": self._best_main_claim_epoch,
                "cf_summary": dict(getattr(self, "_last_cf_summary", {}) or {}),
                "preferred_eval_ckpt": ("model_stage1_best.pt" if self._best_stage1_epoch is not None else ("model_best_gate.pt" if self._best_gate_epoch is not None else "model_best.pt")),
                "preferred_eval_ckpt_realized": ("model_main_claim_best.pt" if self._best_main_claim_epoch is not None else ("model_stage1_best.pt" if self._best_stage1_epoch is not None else ("model_best_gate.pt" if self._best_gate_epoch is not None else "model_best.pt"))),
            })

            ck_every = int(getattr(cfg, "checkpoint_every", 0) or 0)
            if ck_every > 0 and ((epoch + 1) % ck_every == 0):
                ck_path = os.path.join(cfg.save_dir, f"model_epoch_{epoch+1:04d}.pt")
                torch.save({"state_dict": self.model.state_dict(), "epoch": int(epoch), "loss": cur, "score": float(ckpt_score), "stage": str(stage)}, ck_path)

            pat = int(getattr(cfg, "early_stop_patience", 0) or 0)
            if epoch >= start_epoch and pat > 0 and self._no_improve >= pat:
                logger.info(
                    "Early stopping at epoch %d (best_loss=%.6f, cur_loss=%.6f, no_improve=%d, patience=%d, start_epoch=%d).",
                    epoch,
                    float(self._best_score),
                    cur,
                    int(self._no_improve),
                    pat,
                    int(start_epoch),
                )
                break

        ckpt_path = os.path.join(cfg.save_dir, "model.pt")
        torch.save({"state_dict": self.model.state_dict(), "stage": str(stage), "score": float(ckpt_score)}, ckpt_path)
        logger.info("Saved checkpoint: %s", ckpt_path)
        return history

    def _save_stage_snapshot(self, cfg: TrainCfg, stage: str, epoch: int, score: float) -> None:
        try:
            path = os.path.join(cfg.save_dir, f"model_{stage}_snapshot.pt")
            torch.save({
                "state_dict": self.model.state_dict(),
                "epoch": int(epoch),
                "stage": str(stage),
                "score": float(score),
                "residual_gate_summary": dict(getattr(self, "_residual_gate_summary", {}) or {}),
                "roi_target_frac": float(getattr(self, "_roi_target_frac", 0.0) or 0.0),
            }, path)
        except Exception as e:
            logger.warning("Failed to save stage snapshot %s: %s", stage, e)

    def _apply_counterfactual_zE(self, ode_in: ODEInput, do_zE: torch.Tensor) -> ODEInput:
        do_zE = do_zE.to(self.device, dtype=ode_in.zE.dtype)
        mech = ode_in.mechanomics
        if isinstance(mech, torch.Tensor):
            mech_cf = mech.clone()
            if mech_cf.ndim == 2 and do_zE.ndim == 2:
                k = int(min(mech_cf.shape[1], do_zE.shape[1]))
                if k > 0:
                    mech_cf[:, :k] = do_zE[:, :k].to(mech_cf.dtype)
            else:
                mech_cf = mech
        else:
            mech_cf = mech
        return ODEInput(
            edge_index=ode_in.edge_index,
            coords=ode_in.coords,
            zI=ode_in.zI,
            zN=ode_in.zN,
            zE=do_zE,
            mechanomics=mech_cf,
            mass=ode_in.mass,
            mms_mask=ode_in.mms_mask,
        )

    def rollout_with_envelope(self, snap: SpatialSnapshot, t_end: float, do_zE: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Rollout with OOD envelope, lower-bound diffusion sensitivity, and NaN/Inf rollback."""
        self.model.eval()
        ode_in = self._build_ode_input(snap, stage="stage2")

        if do_zE is not None:
            ode_in = self._apply_counterfactual_zE(ode_in, do_zE)

        assert self.zE_mu is not None and self.zE_sd is not None, "Trainer must be fit() before envelope rollout."
        score = _ood_score(ode_in.zE, self.zE_mu, self.zE_sd)
        ood_flag = (score > self.ood_cfg.z_score_threshold).any()

        x0 = snap.X.to(self.device)
        y0 = self.model.encode(x0)
        t = self._make_time_grid(float(t_end))

        yT = self._ode_solve(y0, ode_in, t)
        xT = self.model.decode(yT)
        learned_isfinite = torch.isfinite(yT).all() and torch.isfinite(xT).all()

        orig_mms = self.model.odefunc.enable_mms
        orig_upwind = self.model.odefunc.upwind_drift
        orig_mech = self.model.odefunc.enable_mechanomics
        envelope_info: Dict[str, object] = {"lower_bound_diffusion": []}

        env_input = ode_in
        if bool(ood_flag):
            if self.ood_cfg.fallback_disable_mod:
                self.model.odefunc.enable_mms = False
            if self.ood_cfg.fallback_disable_drift:
                self.model.odefunc.upwind_drift = True
            zE_env = self.zE_mu.unsqueeze(0) + 0.25 * (ode_in.zE - self.zE_mu.unsqueeze(0))
            env_input = self._apply_counterfactual_zE(ode_in, zE_env)

        if (not bool(learned_isfinite)) and bool(getattr(getattr(self, "_stage_cfg", None), "ood_nan_rollback", True)):
            logger.warning("Non-finite learned rollout detected; rolling back to conservative envelope path.")
            self.model.odefunc.enable_mms = False
            self.model.odefunc.upwind_drift = True
            env_input = self._apply_counterfactual_zE(ode_in, self.zE_mu.unsqueeze(0).expand_as(ode_in.zE))
            envelope_info["nan_rollback_triggered"] = True
        else:
            envelope_info["nan_rollback_triggered"] = False

        yT_env = self._ode_solve(y0, env_input, t)
        xT_env = self.model.decode(yT_env)

        lower_scales = tuple(getattr(getattr(self, "_stage_cfg", None), "envelope_lower_diffusion_scales", (0.15, 0.25, 0.50)) or (0.15, 0.25, 0.50))
        sensitivity = []
        for scale in lower_scales:
            zE_lb = self.zE_mu.unsqueeze(0) + float(scale) * (env_input.zE - self.zE_mu.unsqueeze(0))
            ode_in_lb = self._apply_counterfactual_zE(env_input, zE_lb)
            yT_lb = self._ode_solve(y0, ode_in_lb, t)
            xT_lb = self.model.decode(yT_lb)
            sensitivity.append({
                "scale": float(scale),
                "finite": bool(torch.isfinite(xT_lb).all().item() if isinstance(torch.isfinite(xT_lb).all(), torch.Tensor) else torch.isfinite(xT_lb).all()),
                "mean_abs_shift": float((xT_lb - x0).abs().mean().item()),
            })
        envelope_info["lower_bound_diffusion"] = sensitivity

        self.model.odefunc.enable_mms = orig_mms
        self.model.odefunc.upwind_drift = orig_upwind
        self.model.odefunc.enable_mechanomics = orig_mech

        return {
            "ood_flag": torch.tensor([float(ood_flag)], device=self.device),
            "ood_score": score.detach(),
            "yT_learned": torch.nan_to_num(yT.detach(), nan=0.0, posinf=0.0, neginf=0.0),
            "xT_learned": torch.nan_to_num(xT.detach(), nan=0.0, posinf=0.0, neginf=0.0),
            "yT_envelope": torch.nan_to_num(yT_env.detach(), nan=0.0, posinf=0.0, neginf=0.0),
            "xT_envelope": torch.nan_to_num(xT_env.detach(), nan=0.0, posinf=0.0, neginf=0.0),
            "envelope_info": envelope_info,
        }
