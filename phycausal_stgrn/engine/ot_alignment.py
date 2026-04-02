from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Patch tag for sanity-checking which OT alignment implementation is active.
PATCH_TAG = "patched_v12_ot_alignment_shape_safe"

try:
    import ot  # POT
except Exception as e:  # pragma: no cover
    ot = None
    logger.warning("POT (ot) is not available: %s", e)

try:
    from torch_geometric.nn import knn
except Exception as e:  # pragma: no cover
    knn = None
    logger.warning("PyG knn is not available: %s", e)


@dataclass
class OTCfg:
    # Entropic regularization strength for Sinkhorn (POT or sparse-Sinkhorn)
    epsilon: float = 5e-2
    max_iter: int = 100
    # EM update frequency (epochs)
    update_freq: int = 5
    # kNN support size for sparse OT
    knn_k: int = 50
    # product-manifold mixing weight
    alpha: float = 0.5
    # weight for OT latent alignment loss in the training objective
    loss_weight: float = 0.2


def _acosh(x: Tensor) -> Tensor:
    # acosh(x) = log(x + sqrt(x^2 - 1))
    x = torch.clamp(x, min=1.0 + 1e-6)
    return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=1e-12)))


def _poincare_distance(u: Tensor, v: Tensor, eps: float = 1e-5) -> Tensor:
    """Stable Poincaré-ball distance for each row pair in (u, v).

    u, v: (E, d) in open unit ball. We clamp norms to avoid NaN/Inf.
    Returns: (E,)
    """
    # project inside ball
    u2 = torch.sum(u * u, dim=-1, keepdim=True)
    v2 = torch.sum(v * v, dim=-1, keepdim=True)
    # If norms are too close to 1, scale down (projection)
    maxnorm = 1.0 - eps
    u = u * (maxnorm / torch.sqrt(torch.clamp(u2, min=eps))) * (u2.sqrt() > maxnorm).to(u.dtype) + u * (u2.sqrt() <= maxnorm).to(u.dtype)
    v = v * (maxnorm / torch.sqrt(torch.clamp(v2, min=eps))) * (v2.sqrt() > maxnorm).to(v.dtype) + v * (v2.sqrt() <= maxnorm).to(v.dtype)
    u2 = torch.sum(u * u, dim=-1, keepdim=True)
    v2 = torch.sum(v * v, dim=-1, keepdim=True)

    diff2 = torch.sum((u - v) ** 2, dim=-1, keepdim=True)
    denom = torch.clamp((1.0 - u2) * (1.0 - v2), min=eps)
    x = 1.0 + 2.0 * diff2 / denom
    return _acosh(x.squeeze(-1))


@torch.no_grad()
def compute_pi_pot_sparse_knn(
    z_I_0: Tensor,
    z_N_0: Tensor,
    coords_0: Tensor,
    z_I_1: Tensor,
    z_N_1: Tensor,
    coords_1: Tensor,
    k: int = 50,
    alpha: float = 0.5,
    epsilon: float = 5e-2,
    max_iter: int = 100,
    a: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
) -> Tensor:
    """Sparse kNN-constrained Sinkhorn coupling Π (finite migration speed prior).

    Core idea:
      - Support set restricted to kNN in *physical space* (coords), preventing long-range jumps.
      - Cost computed on product manifold:
          C = alpha * d_H(zI) + (1-alpha) * d_E(zN)
      - Sinkhorn scaling performed on sparse edges only, using scatter-add (O(E)).

    Returns:
      Pi_sparse: torch.sparse_coo_tensor of shape (N0, N1), with nnz ~= N0*k.

    Notes on memory:
      - No torch.cdist (no NxN dense cost).
      - Stores only E = N0*k edges and their costs.
    """
    if knn is None:
        raise ImportError("torch_geometric is required for sparse kNN OT (torch_geometric.nn.knn).")

    device = z_I_0.device
    dtype = z_I_0.dtype

    # NOTE: In some pipelines, coords and X may be filtered by different steps.
    # To prevent hard crashes during the E-step, we defensively align lengths.
    N0c = int(coords_0.shape[0])
    N1c = int(coords_1.shape[0])
    N0z = int(min(z_I_0.shape[0], z_N_0.shape[0]))
    N1z = int(min(z_I_1.shape[0], z_N_1.shape[0]))
    N0 = int(min(N0c, N0z))
    N1 = int(min(N1c, N1z))

    if N0 != N0c or N0 != N0z:
        logger.warning(
            "OT alignment: length mismatch at t0 (coords=%d, z=%d). Truncating to %d.",
            N0c,
            N0z,
            N0,
        )
        coords_0 = coords_0[:N0]
        z_I_0 = z_I_0[:N0]
        z_N_0 = z_N_0[:N0]
        if a is not None:
            a = a[:N0]

    if N1 != N1c or N1 != N1z:
        logger.warning(
            "OT alignment: length mismatch at t1 (coords=%d, z=%d). Truncating to %d.",
            N1c,
            N1z,
            N1,
        )
        coords_1 = coords_1[:N1]
        z_I_1 = z_I_1[:N1]
        z_N_1 = z_N_1[:N1]
        if b is not None:
            b = b[:N1]

    if N0 <= 0 or N1 <= 0:
        raise ValueError(f"OT alignment: empty slice after alignment (N0={N0}, N1={N1}).")
    k = int(min(k, N1))

    # Build locality-constrained support.
    # PyG/torch_cluster's `knn(x, y, k)` semantics have varied across versions.
    # We therefore *auto-detect* which row corresponds to which set by comparing
    # index ranges against (N0, N1). This prevents the common out-of-range crash
    # when rows are interpreted in the wrong order.
    edge = knn(x=coords_1, y=coords_0, k=k)  # returns edge_index [2, E]
    e0 = edge[0].to(device)
    e1 = edge[1].to(device)

    max0 = int(e0.max().item()) if e0.numel() > 0 else -1
    max1 = int(e1.max().item()) if e1.numel() > 0 else -1

    # Preferred mapping: dst in [0, N1-1], src in [0, N0-1].
    if max0 < N1 and max1 < N0:
        dst, src = e0, e1
    elif max0 < N0 and max1 < N1:
        # Swapped mapping (observed in some environments)
        src, dst = e0, e1
    else:
        # Ambiguous/invalid: assume (dst=e0, src=e1) but filter invalid edges.
        dst, src = e0, e1
        logger.warning(
            "OT alignment: ambiguous kNN index ranges (max0=%d, max1=%d, N0=%d, N1=%d). "
            "Falling back to filtering invalid edges.",
            max0,
            max1,
            N0,
            N1,
        )

    # Final safety filter: remove any invalid indices.
    if src.numel() > 0:
        valid = (src >= 0) & (src < N0) & (dst >= 0) & (dst < N1)
        if not bool(valid.all()):
            n_bad = int((~valid).sum().item())
            logger.warning(
                "OT alignment: filtered %d/%d invalid kNN edges (src/dst out of range).",
                n_bad,
                int(valid.numel()),
            )
            src = src[valid]
            dst = dst[valid]

    if src.numel() == 0:
        # Extremely defensive fallback: if kNN produced no valid edges, return a simple
        # one-to-one coupling to keep training running.
        m = int(min(N0, N1))
        src = torch.arange(m, device=device, dtype=torch.long)
        dst = torch.arange(m, device=device, dtype=torch.long)
    E = int(src.numel())

    # Gather latent pairs on edges
    zi0 = z_I_0[src]
    zi1 = z_I_1[dst]
    zn0 = z_N_0[src]
    zn1 = z_N_1[dst]

    # Product manifold cost (sparse)
    d_hyp = _poincare_distance(zi0, zi1)  # (E,)
    d_euc = torch.norm(zn0 - zn1, dim=-1)  # (E,)
    alpha = float(alpha)
    cost = alpha * d_hyp + (1.0 - alpha) * d_euc

    # Kernel on edges: K = exp(-C/eps), clamp to avoid underflow -> zero mass on all edges
    eps = float(epsilon)
    scaled = torch.clamp(cost / max(eps, 1e-8), max=50.0)
    K = torch.exp(-scaled).to(dtype=dtype)  # (E,)

    # Marginals
    if a is None:
        a = torch.full((N0,), 1.0 / max(N0, 1), device=device, dtype=dtype)
    else:
        a = a.to(device=device, dtype=dtype)
        a = a / (a.sum() + 1e-12)

    if b is None:
        b = torch.full((N1,), 1.0 / max(N1, 1), device=device, dtype=dtype)
    else:
        b = b.to(device=device, dtype=dtype)
        b = b / (b.sum() + 1e-12)

    # Sinkhorn scaling vectors (dense but size N only)
    u = torch.ones((N0,), device=device, dtype=dtype)
    v = torch.ones((N1,), device=device, dtype=dtype)

    # Sparse Sinkhorn-Knopp iterations using scatter-add
    # row_sum_i = sum_j K_ij * v_j  for edges (i->j)
    for _ in range(int(max_iter)):
        Kv = K * v[dst]
        row_sum = torch.zeros((N0,), device=device, dtype=dtype)
        row_sum.scatter_add_(0, src, Kv)
        u = a / (row_sum + 1e-12)

        KTu = K * u[src]
        col_sum = torch.zeros((N1,), device=device, dtype=dtype)
        col_sum.scatter_add_(0, dst, KTu)
        v = b / (col_sum + 1e-12)

    # Π_ij = u_i * K_ij * v_j on sparse edges
    values = (u[src] * K * v[dst]).to(dtype=dtype)

    # Return sparse coupling tensor (COO)
    indices = torch.stack([src, dst], dim=0)
    Pi = torch.sparse_coo_tensor(indices, values, size=(N0, N1), device=device, dtype=dtype)
    return Pi.coalesce()


def compute_ot_coupling_dense_pot(
    x_src: Tensor,
    x_tgt: Tensor,
    a: Optional[Tensor] = None,
    b: Optional[Tensor] = None,
    epsilon: float = 5e-2,
    max_iter: int = 200,
) -> Tensor:
    """(Fallback) Dense POT-Sinkhorn coupling for small N (NOT for N~20k)."""
    if ot is None:
        raise ImportError("POT library is required. Install pot>=0.9.3")

    xs = x_src.detach().cpu().float().numpy()
    xt = x_tgt.detach().cpu().float().numpy()
    C = ot.dist(xs, xt, metric="euclidean") ** 2

    N, M = xs.shape[0], xt.shape[0]
    if a is None:
        a_np = torch.ones(N, dtype=torch.float64).numpy() / max(N, 1)
    else:
        a_np = (a.detach().cpu().double() / (a.sum() + 1e-12)).numpy()

    if b is None:
        b_np = torch.ones(M, dtype=torch.float64).numpy() / max(M, 1)
    else:
        b_np = (b.detach().cpu().double() / (b.sum() + 1e-12)).numpy()

    Pi = ot.sinkhorn(a_np, b_np, C, reg=float(epsilon), numItermax=int(max_iter))
    return torch.from_numpy(Pi).to(x_src.device, dtype=x_src.dtype)


def ot_alignment_loss(
    y_src_pred: Tensor,
    y_tgt_obs: Tensor,
    coupling: Tensor,
) -> Tensor:
    """OT soft-alignment loss supporting sparse Π (preferred) or dense Π.

    Transport predicted source latent to target support:
        y_hat_on_tgt = Π^T @ y_src_pred
    Loss:
        || y_hat_on_tgt - y_tgt_obs ||^2
    """
    if coupling.is_sparse:
        transported = torch.sparse.mm(coupling.transpose(0, 1), y_src_pred)
    else:
        transported = coupling.transpose(0, 1) @ y_src_pred
    if transported.shape[0] != y_tgt_obs.shape[0]:
        m = int(min(transported.shape[0], y_tgt_obs.shape[0]))
        logger.warning(
            "OT alignment loss: transported N=%d != target N=%d; truncating to %d. (%s)",
            int(transported.shape[0]),
            int(y_tgt_obs.shape[0]),
            m,
            PATCH_TAG,
        )
        transported = transported[:m]
        y_tgt_obs = y_tgt_obs[:m]
    return torch.mean((transported - y_tgt_obs) ** 2)
