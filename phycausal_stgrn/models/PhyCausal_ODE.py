from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phycausal_stgrn.models.layers import GNNBlock, MLP, MonotonicNet
from phycausal_stgrn.utils.diagnostics import check_tensor_finite


logger = logging.getLogger(__name__)


@dataclass
class ODEInput:
    edge_index: torch.Tensor            # (2, E)
    coords: torch.Tensor                # (N, 2)
    zI: torch.Tensor                    # (N, dI) invariant/slow env features (optional)
    zN: torch.Tensor                    # (N, dN) niche features (gate driver)
    zE: torch.Tensor                    # (N, dE) environment features for drift (includes mechanomics surrogates)
    mechanomics: torch.Tensor           # (N, dM) mechanomics features (subset of zE)
    mass: Optional[torch.Tensor] = None # (N,) optional weights
    mms_mask: Optional[torch.Tensor] = None # (N,1) optional stage-wise gate mask


class HardConcreteGateNet(nn.Module):
    """Mechanism-shift gate with a stable soft mode and optional hard-concrete mode.

    Design goals for real ST data:
      - avoid early collapse to all-zero gates,
      - keep an MMS-compatible sparse interpretation,
      - let interface-local supervision shape the gate before sparsity is enforced.

    In default ``mode="soft_sigmoid"`` the network predicts a *node-level* gate and
    expands it across latent dimensions. This is intentionally easier to optimize than
    a full latent-dimension-specific Bernoulli mask and empirically better matches the
    paper story: a local mechanism either opens or stays off at a spatial location.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        out_dim: int,
        temperature: float = 1.5,
        gamma: float = -0.1,
        zeta: float = 1.1,
        eps: float = 1e-7,
        dropout: float = 0.05,
        init_bias: float = -0.34,
        interface_boost: float = 0.56,
        gate_floor: float = 2e-4,
        sparse_deadzone: float = 0.0,
        mode: str = "soft_sigmoid",
        shared_across_latent: bool = True,
        tau_min: float = 0.75,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.temperature = float(max(0.1, temperature))
        self.tau_min = float(max(0.05, tau_min))
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.eps = float(eps)
        self.mode = str(mode or "soft_sigmoid").lower()
        self.shared_across_latent = bool(shared_across_latent)
        self.interface_boost = float(interface_boost)
        self.gate_floor = float(gate_floor)
        self.sparse_deadzone = float(max(0.0, min(0.45, sparse_deadzone)))

        out_logits = 1 if self.shared_across_latent else int(out_dim)
        self.pre = nn.Sequential(
            nn.Linear(int(z_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(int(hidden_dim), int(out_logits))
        self.skip = nn.Linear(int(z_dim), int(out_logits), bias=False)
        self.interface_head = nn.Linear(1, int(out_logits), bias=False)

        nn.init.xavier_uniform_(self.pre[0].weight)
        nn.init.zeros_(self.pre[0].bias)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.xavier_uniform_(self.skip.weight, gain=0.06)
        nn.init.constant_(self.head.bias, float(init_bias))
        nn.init.constant_(self.interface_head.weight, float(interface_boost))

        self.last_log_alpha: Optional[torch.Tensor] = None
        self.last_p_nonzero: Optional[torch.Tensor] = None
        self.last_gate: Optional[torch.Tensor] = None
        self.last_gate_raw: Optional[torch.Tensor] = None
        self.last_gate_node: Optional[torch.Tensor] = None
        self.last_logits_node: Optional[torch.Tensor] = None

    def set_temperature(self, value: float) -> None:
        self.temperature = float(max(self.tau_min, value))

    def expected_l0(self, log_alpha: torch.Tensor) -> torch.Tensor:
        log_ratio = math.log(-self.gamma / self.zeta)
        return torch.sigmoid(log_alpha - self.temperature * log_ratio)

    def _sample_hard_concrete(self, log_alpha: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(log_alpha).clamp(self.eps, 1.0 - self.eps)
        s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + log_alpha) / self.temperature)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0.0, 1.0)

    def _soft_gate(self, logits: torch.Tensor) -> torch.Tensor:
        tau = float(max(self.tau_min, self.temperature))
        return torch.sigmoid(logits / tau)

    def forward(self, z: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = self.pre(z)
        logits = self.head(h) + 0.05 * self.skip(z)
        if z.ndim == 2 and z.shape[1] >= 1:
            interface_signal = z[:, -1:].float()
            interface_signal = (interface_signal - interface_signal.mean(dim=0, keepdim=True)) / interface_signal.std(dim=0, keepdim=True).clamp_min(1e-4)
            interface_signal = torch.tanh(0.76 * interface_signal)
            logits = logits + self.interface_head(interface_signal)
            logits = logits.clamp(min=-3.4, max=3.4)
            floor = float(self.gate_floor) * (0.05 + 0.95 * torch.sigmoid(2.2 * interface_signal))
        else:
            interface_signal = None
            floor = logits.new_full(logits.shape, float(self.gate_floor))

        if self.mode in {"hard", "hard_concrete", "l0"}:
            p_nonzero = self.expected_l0(logits)
            gate_node = self._sample_hard_concrete(logits) if sample else self._soft_gate(logits)
        else:
            gate_node = self._soft_gate(logits)
            p_nonzero = gate_node

        if self.sparse_deadzone > 0.0:
            dz = float(self.sparse_deadzone)
            gate_node = torch.relu(gate_node - dz) / max(1e-6, 1.0 - dz)
            p_nonzero = torch.relu(p_nonzero - dz) / max(1e-6, 1.0 - dz)

        gate_node = torch.maximum(gate_node, floor)
        p_nonzero = torch.maximum(p_nonzero, floor)
        if self.shared_across_latent:
            gate = gate_node.expand(-1, self.out_dim)
            p_full = p_nonzero.expand(-1, self.out_dim)
            log_full = logits.expand(-1, self.out_dim)
        else:
            gate = gate_node
            p_full = p_nonzero
            log_full = logits

        self.last_log_alpha = log_full
        self.last_p_nonzero = p_full
        self.last_gate = gate
        self.last_gate_raw = gate
        self.last_gate_node = gate_node
        self.last_logits_node = logits
        return gate

    def l0_penalty(self, reduction: str = "mean") -> torch.Tensor:
        if self.last_p_nonzero is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        p = self.last_p_nonzero
        if reduction == "mean":
            return p.mean()
        if reduction == "sum":
            return p.sum()
        raise ValueError(f"Unknown reduction={reduction}")


class PhyCausalODEFunc(nn.Module):
    """
    Vector field for torchdiffeq.
    Implements:
      dy/dt = f_core(y, zI) + Gate(zN) ⊙ f_mod(y, zN)
            + diffusion(y; D(mech)) + drift(y; v(zE))
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_gnn_layers: int,
        zI_dim: int,
        zN_dim: int,
        zE_dim: int,
        mech_dim: int,
        gate_hidden_dim: Optional[int] = None,
        enable_mms: bool = True,
        enable_mechanomics: bool = True,
        upwind_drift: bool = True,
        D_max: float = 1.0,
        v_max: float = 1.0,
        dy_clip: float = 50.0,
        gate_mode: str = "soft_sigmoid",
        gate_temperature: float = 1.8,
        gate_tau_min: float = 0.9,
        gate_shared_across_latent: bool = True,
        gate_init_bias: float = -0.34,
        gate_interface_boost: float = 0.56,
        gate_floor: float = 2e-4,
        gate_sparse_deadzone: float = 0.0,
        mms_bg_floor: float = 0.08,
        mms_gain: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.enable_mms = enable_mms
        self.enable_mechanomics = enable_mechanomics
        self.upwind_drift = upwind_drift
        # Physical-stability clamps (do not change model structure; prevent NaN/Inf on CPU)
        self.D_max = float(D_max)
        self.v_max = float(v_max)
        self.dy_clip = float(dy_clip)

        self.nfe = 0  # incremented each forward call (for diagnostics)

        # Cache latest gate for regularization/diagnostics
        self.gate_last: Optional[torch.Tensor] = None
        self.gate_raw_last: Optional[torch.Tensor] = None
        self.gate_p_nonzero_last: Optional[torch.Tensor] = None
        self.gate_node_last: Optional[torch.Tensor] = None
        self.mms_bg_floor = float(max(0.0, min(0.2, mms_bg_floor)))
        self.mms_gain = float(max(0.1, mms_gain))

        # Core (environment-invariant) reaction skeleton
        self.core_gnns = nn.ModuleList([GNNBlock(latent_dim) for _ in range(num_gnn_layers)])
        self.core_mlp = MLP(latent_dim + zI_dim, hidden_dim, latent_dim, num_layers=2)

        # Modulation branch (mechanism shift) gated by niche features
        self.mod_gnns = nn.ModuleList([GNNBlock(latent_dim) for _ in range(max(1, num_gnn_layers))])
        self.mod_mlp = MLP(latent_dim + zN_dim, hidden_dim, latent_dim, num_layers=2)
        gate_hidden_dim = int(gate_hidden_dim or hidden_dim)
        self.gate = HardConcreteGateNet(zN_dim, gate_hidden_dim, latent_dim, mode=gate_mode, temperature=gate_temperature, tau_min=gate_tau_min, shared_across_latent=gate_shared_across_latent, init_bias=gate_init_bias, interface_boost=gate_interface_boost, gate_floor=gate_floor, sparse_deadzone=gate_sparse_deadzone)

        # Mechanomics diffusion & drift parameterizers
        self.D_net = MLP(mech_dim, hidden_dim, 1, num_layers=2)   # scalar diffusion factor per node
        self.v_net = MLP(zE_dim, hidden_dim, 2, num_layers=2)     # 2D drift vector per node

    def _graph_laplacian_diffusion(self, y: torch.Tensor, edge_index: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        Diffusion on graph: sum_j w_ij (y_j - y_i).
        Use D as per-node positive coefficient.
        """
        i, j = edge_index[0], edge_index[1]
        dy = y[j] - y[i]
        # symmetric weight: average of D_i and D_j
        w = 0.5 * (D[i] + D[j])  # (E,1)
        out = torch.zeros_like(y)
        out.index_add_(0, i, w * dy)
        return out

    def _drift_advection(self, y: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Drift/advection on graph. For each directed edge i<-j, keep downstream messages only if upwind enabled:
        alpha_ij = ReLU( v_i · (s_j - s_i) )
        term: sum_j alpha_ij * (y_j - y_i)
        """
        i, j = edge_index[0], edge_index[1]
        dij = coords[j] - coords[i]  # (E,2)
        vi = v[i]                    # (E,2)
        alpha = (vi * dij).sum(dim=1, keepdim=True)  # (E,1)
        if self.upwind_drift:
            alpha = F.relu(alpha)
        out = torch.zeros_like(y)
        out.index_add_(0, i, alpha * (y[j] - y[i]))
        # normalize by degree-ish to reduce scale explosions
        deg = torch.zeros(y.shape[0], device=y.device).index_add_(0, i, torch.ones_like(i, dtype=torch.float))
        out = out / deg.clamp_min(1.0).unsqueeze(1)
        return out

    def forward(self, t: torch.Tensor, y: torch.Tensor, ode_in: ODEInput) -> torch.Tensor:
        self.nfe += 1

        edge_index = ode_in.edge_index
        coords = ode_in.coords
        zI, zN, zE, mech = ode_in.zI, ode_in.zN, ode_in.zE, ode_in.mechanomics
        mms_mask = getattr(ode_in, "mms_mask", None)

        # Core dynamics
        h = y
        for gnn in self.core_gnns:
            h = gnn(h, edge_index)
        core = self.core_mlp(torch.cat([h, zI], dim=1))

        # Modulation dynamics (MMS)
        if self.enable_mms:
            hm = y
            for gnn in self.mod_gnns:
                hm = gnn(hm, edge_index)
            mod_raw = self.mod_mlp(torch.cat([hm, zN], dim=1))
            gate_raw = self.gate(zN, sample=self.training)  # (N, d) continuous node gate
            gate = gate_raw
            if mms_mask is not None:
                mask = mms_mask.to(device=gate.device, dtype=gate.dtype)
                if mask.ndim == 1:
                    mask = mask.unsqueeze(1)
                bg_floor = self.mms_bg_floor
                mask = mask * (1.0 - bg_floor) + bg_floor
                gate = gate * mask
            # Cache for Trainer regularizers (raw gate for supervision, masked gate for dynamics)
            self.gate_raw_last = gate_raw
            self.gate_last = gate
            self.gate_p_nonzero_last = self.gate.last_p_nonzero
            self.gate_node_last = self.gate.last_gate_node
            mod = self.mms_gain * gate * mod_raw
        else:
            gate = torch.zeros_like(y)
            self.gate_last = gate
            self.gate_p_nonzero_last = torch.zeros_like(gate)
            self.gate_node_last = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
            mod = torch.zeros_like(y)

        # Mechanomics diffusion & drift
        if self.enable_mechanomics:
            D = F.softplus(self.D_net(mech))  # (N,1)
            # Clamp diffusion to a feasible physical interval to avoid exploding Laplacian terms
            if self.D_max is not None:
                D = D.clamp(min=0.0, max=self.D_max)
            diff = self._graph_laplacian_diffusion(y, edge_index, D)
            v = self.v_net(zE)  # (N,2)
            # Clamp drift magnitude to avoid large advection steps at init
            if self.v_max is not None:
                v = torch.tanh(v) * self.v_max
            drift = self._drift_advection(y, coords, edge_index, v)
        else:
            D = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
            v = torch.zeros((y.shape[0], 2), device=y.device, dtype=y.dtype)
            diff = torch.zeros_like(y)
            drift = torch.zeros_like(y)

        dy = core + mod + diff + drift
        # Stabilize dy/dt on real ST data: replace NaN/Inf and clip extreme magnitudes.
        if not torch.isfinite(dy).all():
            dy = torch.nan_to_num(dy, nan=0.0, posinf=self.dy_clip, neginf=-self.dy_clip)
        dy = dy.clamp(min=-self.dy_clip, max=self.dy_clip)

        # Safety: abort early if still non-finite to avoid silent corruption
        check_tensor_finite(dy, "dy/dt", raise_on_fail=True)
        return dy

    @torch.no_grad()
    def compute_gate(self, ode_in: ODEInput, deterministic: bool = True, apply_mask: bool = True) -> torch.Tensor:
        if not self.enable_mms:
            return torch.zeros((ode_in.zN.shape[0], self.latent_dim), device=ode_in.zN.device)
        gate_raw = self.gate(ode_in.zN, sample=(self.training and not deterministic))
        gate = gate_raw
        mms_mask = getattr(ode_in, "mms_mask", None)
        if apply_mask and mms_mask is not None:
            mask = mms_mask.to(device=gate.device, dtype=gate.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(1)
            bg_floor = self.mms_bg_floor
            mask = mask * (1.0 - bg_floor) + bg_floor
            gate = gate * mask
        self.gate_raw_last = gate_raw
        self.gate_last = gate
        self.gate_p_nonzero_last = self.gate.last_p_nonzero
        self.gate_node_last = self.gate.last_gate_node
        return gate

class PhyCausalSTGRN(nn.Module):
    """End-to-end model:
      - Encoder: ST expression -> latent y0
      - ODE solve: y(t) rollout with torchdiffeq
      - Decoder: latent -> reconstructed expression
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        hidden_dim: int,
        num_gnn_layers: int,
        zI_dim: int,
        zN_dim: int,
        zE_dim: int,
        mech_dim: int,
        gate_hidden_dim: Optional[int] = None,
        enable_mms: bool = True,
        enable_mechanomics: bool = True,
        upwind_drift: bool = True,
        D_max: float = 1.0,
        v_max: float = 1.0,
        dy_clip: float = 50.0,
        gate_mode: str = "soft_sigmoid",
        gate_temperature: float = 1.8,
        gate_tau_min: float = 0.9,
        gate_shared_across_latent: bool = True,
        gate_init_bias: float = -0.34,
        gate_interface_boost: float = 0.56,
        gate_floor: float = 2e-4,
        gate_sparse_deadzone: float = 0.0,
        mms_bg_floor: float = 0.08,
        mms_gain: float = 1.0,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        self.latent_dim = int(latent_dim)

        self.encoder = MLP(self.num_genes, hidden_dim, self.latent_dim, num_layers=3)
        self.decoder = MLP(self.latent_dim, hidden_dim, self.num_genes, num_layers=3)

        # --- EM minimal cache (invalidation by epoch) ---
        self.cached_pi = []  # list[torch.Tensor], consecutive Π_{t->t+1}
        self.last_pi_update_epoch: int = -10**9

        # --- OT latent heads (do not alter core ODE mechanisms) ---
        self.zi_head = nn.Linear(self.latent_dim, int(zI_dim))
        self.zn_head = nn.Linear(self.latent_dim, int(zN_dim))

        self.odefunc = PhyCausalODEFunc(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            zI_dim=zI_dim,
            zN_dim=zN_dim,
            zE_dim=zE_dim,
            mech_dim=mech_dim,
            gate_hidden_dim=gate_hidden_dim,
            enable_mms=enable_mms,
            enable_mechanomics=enable_mechanomics,
            upwind_drift=upwind_drift,
            gate_mode=gate_mode,
            gate_temperature=gate_temperature,
            gate_tau_min=gate_tau_min,
            gate_shared_across_latent=gate_shared_across_latent,
            gate_init_bias=gate_init_bias,
            gate_interface_boost=gate_interface_boost,
            gate_floor=gate_floor,
            gate_sparse_deadzone=gate_sparse_deadzone,
            mms_bg_floor=mms_bg_floor,
            mms_gain=mms_gain,
        )

    def encode_zi_zn(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode x into denoised latent proxies (Z_I, Z_N) for OT E-step."""
        h = self.encoder(x)
        zi = self.zi_head(h)
        zn = self.zn_head(h)
        check_tensor_finite(zi, "zi_head")
        check_tensor_finite(zn, "zn_head")
        return zi, zn

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(y)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x0))
