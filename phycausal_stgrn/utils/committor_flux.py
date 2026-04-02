from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SparseTransition:
    edge_index: torch.Tensor  # (2, E) directed edges i->j
    prob: torch.Tensor        # (E,) transition probability P_ij aligned to edge_index
    N: int                    # number of nodes


@dataclass
class CommittorResult:
    q: torch.Tensor                # (N,) committor: Prob(reach B before A)
    flux_edge_index: torch.Tensor  # (2, E)
    flux: torch.Tensor             # (E,) net reactive flux on directed edges
    rate: torch.Tensor             # scalar, total reactive flux rate (A->B proxy)


@torch.no_grad()
def build_sparse_transition(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    v: torch.Tensor,
    D: torch.Tensor,
    beta: float = 1.0,
    upwind: bool = True,
    self_loop_eps: float = 1e-3,
    eps: float = 1e-12,
) -> SparseTransition:
    """Build sparse Markov transition on kNN graph without forming dense NxN.

    Edge weights:
      w_ij ∝ exp(beta * (v_i·(x_j-x_i))_+) * exp(-||x_j-x_i||^2 / (4 D_i + eps))
    then row-normalize per i.

    Returns:
      SparseTransition with prob aligned to edge_index (i->j).
    """
    i, j = edge_index[0], edge_index[1]
    N = int(coords.shape[0])

    dij = coords[j] - coords[i]
    adv = (v[i] * dij).sum(dim=1)
    if upwind:
        adv = F.relu(adv)

    dist2 = (dij ** 2).sum(dim=1)
    Di = D[i].reshape(-1).clamp_min(1e-6)

    w = torch.exp(beta * adv) * torch.exp(-dist2 / (4.0 * Di))

    # Row-normalize: p_ij = w_ij / (sum_j w_ij + self_loop_eps)
    row_sum = torch.zeros((N,), device=coords.device, dtype=coords.dtype)
    row_sum.index_add_(0, i, w)
    denom = row_sum + self_loop_eps
    prob = w / denom[i]

    # Add implicit self-loop probability mass via self_loop_eps / denom, handled in apply_P()
    return SparseTransition(edge_index=edge_index, prob=prob, N=N)


@torch.no_grad()
def apply_P(
    trans: SparseTransition,
    x: torch.Tensor,
    self_loop_eps: float = 1e-3,
) -> torch.Tensor:
    """Compute y = P x for sparse transition with implicit self-loops.

    P_ij exists for edges (i->j) as trans.prob.
    Self-loop probability is self_loop_eps / denom_i, where denom_i = sum_w_i + self_loop_eps.
    We don't store denom_i explicitly; approximate self-loop as 1 - sum_j p_ij.
    """
    edge_index, p = trans.edge_index, trans.prob
    i, j = edge_index[0], edge_index[1]
    N = trans.N
    y = torch.zeros((N,), device=x.device, dtype=x.dtype)
    # y_i += sum_j p_ij x_j
    y.index_add_(0, i, p * x[j])

    # self-loop: y_i += (1 - sum_j p_ij) * x_i
    row_p = torch.zeros((N,), device=x.device, dtype=x.dtype)
    row_p.index_add_(0, i, p)
    self_p = (1.0 - row_p).clamp_min(0.0)
    y = y + self_p * x
    return y


@torch.no_grad()
def solve_committor_dirichlet_sparse(
    trans: SparseTransition,
    A: torch.Tensor,
    B: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 20_000,
) -> torch.Tensor:
    """Solve committor q on sparse Markov chain with Dirichlet BCs.

    q=0 on A, q=1 on B, and q = P q on interior.

    Implementation: fixed-point iteration on full vector with boundary clamping.
    Converges for an ergodic chain with absorbing boundaries (practically robust for kNN with self-loops).
    """
    N = trans.N
    A = A.bool()
    B = B.bool()
    I = ~(A | B)

    q = torch.zeros((N,), device=trans.prob.device, dtype=trans.prob.dtype)
    q[B] = 1.0

    for _ in range(max_iter):
        q_new = apply_P(trans, q)
        q_new[A] = 0.0
        q_new[B] = 1.0
        err = torch.max(torch.abs(q_new[I] - q[I])) if I.any() else torch.tensor(0.0, device=q.device, dtype=q.dtype)
        q = q_new
        if float(err.item()) < tol:
            break

    return q.clamp(0.0, 1.0)


@torch.no_grad()
def estimate_stationary_pi_sparse(
    trans: SparseTransition,
    n_iter: int = 2000,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Power iteration to estimate stationary distribution pi for sparse P.

    For large graphs this is still cheap: O(E) per iteration.
    """
    N = trans.N
    pi = torch.ones((N,), device=trans.prob.device, dtype=trans.prob.dtype) / float(N)
    for _ in range(n_iter):
        pi_new = apply_P(trans, pi)
        pi_new = pi_new / (pi_new.sum() + 1e-12)
        err = torch.max(torch.abs(pi_new - pi))
        pi = pi_new
        if float(err.item()) < tol:
            break
    return pi


@torch.no_grad()
def reactive_flux_sparse(
    trans: SparseTransition,
    q: torch.Tensor,
    edge_index: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute net reactive flux on edge list for sparse transition.

    f_ij = pi_i P_ij q_i (1-q_j)
    F_ij = max(0, f_ij - f_ji)

    We approximate f_ji on the same (i->j) list by looking up reverse edge probability if present.
    This is exact when the graph has both directions for each neighbor pair (common with kNN if symmetrized).
    """
    N = trans.N
    if pi is None:
        pi = torch.ones((N,), device=q.device, dtype=q.dtype) / float(N)

    i, j = edge_index[0], edge_index[1]
    Pij = trans.prob  # aligned to edge_index
    fij = pi[i] * Pij * q[i] * (1.0 - q[j])

    # Build reverse map: (j,i) -> index
    # For minimal demo, use hashing on CPU tensors; for huge E this is heavy, but still ok for kNN.
    ei = edge_index.detach().cpu()
    key = (ei[0].numpy().astype("int64") << 32) + ei[1].numpy().astype("int64")
    rev_key = (ei[1].numpy().astype("int64") << 32) + ei[0].numpy().astype("int64")
    pos = {int(k): idx for idx, k in enumerate(key.tolist())}
    rev_idx = torch.tensor([pos.get(int(k), -1) for k in rev_key.tolist()], dtype=torch.long, device=q.device)

    Pji = torch.zeros_like(Pij)
    mask = rev_idx >= 0
    Pji[mask] = Pij[rev_idx[mask]]

    fji = pi[j] * Pji * q[j] * (1.0 - q[i])
    Fnet = torch.clamp(fij - fji, min=0.0)
    return Fnet, Fnet.sum()


@torch.no_grad()
def committor_and_flux(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
    v: torch.Tensor,
    D: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float = 1.0,
    upwind: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20_000,
    estimate_pi: bool = False,
) -> CommittorResult:
    """High-level API for committor + reactive flux (sparse, scalable).

    - Works for large N because it never builds dense NxN matrices.
    - For real runs: keep estimate_pi=False unless you need stationary-weighted flux; uniform pi often suffices for comparisons.
    """
    trans = build_sparse_transition(edge_index=edge_index, coords=coords, v=v, D=D, beta=beta, upwind=upwind)
    q = solve_committor_dirichlet_sparse(trans=trans, A=A, B=B, tol=tol, max_iter=max_iter)
    pi = estimate_stationary_pi_sparse(trans) if estimate_pi else None
    F, total = reactive_flux_sparse(trans=trans, q=q, edge_index=edge_index, pi=pi)
    return CommittorResult(q=q, flux_edge_index=edge_index, flux=F, rate=total)
