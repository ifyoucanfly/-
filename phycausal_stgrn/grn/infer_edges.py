from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EdgeInferenceCfg:
    method: Literal["jacobian", "ig", "cycle_jacobian"] = "jacobian"
    top_k: int = 2000
    abs_score: bool = True
    ig_steps: int = 32
    cycle_max_cells: int = 32
    # For cycle_jacobian: compute multiple subsamples of cells and fuse their score matrices.
    # This reduces ranking noise when max_cells is small or data is heterogeneous.
    n_subsamples: int = 1
    subsample_agg: Literal["mean", "median"] = "mean"
    score_norm: Literal["none", "rowcol"] = "none"

@dataclass
class EdgeScores:
    edge_index: torch.Tensor   # (2, E)
    scores: torch.Tensor       # (E,)
    method: str


def _flatten_score(S: torch.Tensor, abs_score: bool = True) -> torch.Tensor:
    # S: (G, G)
    if abs_score:
        S = S.abs()
    return S




def _normalize_score_matrix(S: torch.Tensor, mode: str = "none", eps: float = 1e-8) -> torch.Tensor:
    """Normalize a directed score matrix before ranking edges.

    rowcol: divide each entry by sqrt(out_strength[dst] * in_strength[src]) to reduce
    hub-dominated rankings that can overwhelm sparse GRN recovery.
    """
    mode = str(mode or "none").lower()
    if mode == "none":
        return S
    if mode != "rowcol":
        raise ValueError(f"Unknown score normalization mode: {mode}")
    row_strength = S.sum(dim=1, keepdim=True).clamp_min(eps)
    col_strength = S.sum(dim=0, keepdim=True).clamp_min(eps)
    denom = torch.sqrt(row_strength * col_strength)
    return S / denom.clamp_min(eps)


def infer_edges_from_decoder_jacobian(
    decoder: nn.Module,
    y0: torch.Tensor,
    gene_names: Optional[Tuple[str, ...]] = None,
    abs_score: bool = True,
    top_k: int = 2000,
    score_norm: str = "none",
) -> EdgeScores:
    """Infer GRN edges by the Jacobian of decoder output wrt latent state.

    This is a pragmatic approach for a *paper-grade* repository:
    - If your latent dims == genes (latent_dim == G), decoder can be identity-ish, then Jacobian aligns with gene->gene effect.
    - Otherwise treat it as a proxy: edges in latent gene-program space.

    Strategy:
    - compute J = d x_hat / d y at y0 (averaged over nodes)
    - map to gene-gene by projecting: S = J^T J (influence similarity)
    - pick top-K off-diagonal entries as edges

    Args:
        decoder: maps y -> x_hat (genes)
        y0: (N, d) latent at baseline
    """
    device = y0.device
    N, d = y0.shape

    # Use a small sample of nodes for Jacobian estimation (stable & fast)
    B = min(128, int(N))
    idx = torch.randperm(N, device=device)[:B]
    y = y0[idx].detach().requires_grad_(True)

    # IMPORTANT: do NOT run this under torch.no_grad(). We need autograd.
    with torch.enable_grad():
        x = decoder(y)  # (B, G)
        G = int(x.shape[1])

        # If latent_dim == num_genes, we can produce a DIRECTED gene->gene Jacobian.
        # This avoids the symmetric proxy (J^T J) that tends to look like a correlation network.
        max_full = 512  # safety for very large gene sets
        if d == G and G <= max_full:
            J = torch.zeros((G, G), device=device, dtype=y.dtype)
            # Row i: d x_hat_i / d y (averaged over sampled nodes)
            # Use retain_graph=True since we reuse the same forward graph.
            for i in range(G):
                s = x[:, i].sum()
                gi = torch.autograd.grad(s, y, retain_graph=True, create_graph=False)[0]  # (B, G)
                J[i] = gi.mean(dim=0)

            S = J
            if abs_score:
                S = S.abs()
            S.fill_diagonal_(0.0)
            S = _normalize_score_matrix(S, mode=score_norm)
            flat = S.flatten()
            k = min(int(top_k), int(flat.numel()))
            vals, inds = torch.topk(flat, k=k, largest=True, sorted=True)
            src = inds // S.shape[1]
            dst = inds % S.shape[1]
            edge_index = torch.stack([src, dst], dim=0).to(torch.long)
            return EdgeScores(edge_index=edge_index, scores=vals.detach(), method="jacobian_directed")

        # Otherwise: latent-program proxy via Hutchinson estimate of S = E[J^T J] (symmetric in latent space)
        num_probe = 16
        S = torch.zeros((d, d), device=device, dtype=y.dtype)
        for _ in range(num_probe):
            rvec = torch.randn((x.shape[0], G), device=device, dtype=y.dtype)
            s = (x * rvec).sum()
            grads = torch.autograd.grad(s, y, retain_graph=True, create_graph=False)[0]  # (B,d)
            g = grads.mean(dim=0, keepdim=True)  # (1,d)
            S = S + (g.t() @ g)
        S = S / float(num_probe)
        S = _flatten_score(S, abs_score=abs_score)
        S.fill_diagonal_(0.0)
        flat = S.flatten()
        k = min(int(top_k), int(flat.numel()))
        vals, inds = torch.topk(flat, k=k, largest=True, sorted=True)
        src = inds // S.shape[1]
        dst = inds % S.shape[1]
        edge_index = torch.stack([src, dst], dim=0).to(torch.long)
        return EdgeScores(edge_index=edge_index, scores=vals.detach(), method="jacobian_proxy")


def _integrated_gradients(
    model: nn.Module,
    y_baseline: torch.Tensor,
    y_input: torch.Tensor,
    target_gene: int,
    steps: int = 32,
) -> torch.Tensor:
    """Integrated gradients of x_hat[target_gene] wrt y input."""
    device = y_input.device
    with torch.enable_grad():
        alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1)  # (S,1,1)
        y = y_baseline.unsqueeze(0) + alphas * (y_input.unsqueeze(0) - y_baseline.unsqueeze(0))
        y = y.requires_grad_(True)

        # Most decoders expect 2D inputs; flatten time/batch dims.
        S, B, d = y.shape
        y_flat = y.reshape(S * B, d)
        x_flat = model(y_flat)  # (S*B, G)
        x = x_flat.reshape(S, B, -1)

        s = x[..., target_gene].sum()
        grad = torch.autograd.grad(s, y, create_graph=False, retain_graph=False)[0]  # (S,B,d)
        avg_grad = grad.mean(dim=0)  # (B,d)
        ig = (y_input - y_baseline) * avg_grad.mean(dim=0)  # (d,)
        return ig.detach()


def infer_edges_from_ig(
    decoder: nn.Module,
    y0: torch.Tensor,
    abs_score: bool = True,
    top_k: int = 2000,
    steps: int = 32,
    score_norm: str = "none",
) -> EdgeScores:
    """Infer edges using Integrated Gradients (IG) on the decoder.

    For each 'target gene' g, compute IG w.r.t. latent dims.
    Then assemble a latent->gene influence matrix A (G x d) and convert to dxd similarity S=A^T A.
    """
    device = y0.device
    N, d = y0.shape
    idx = torch.randperm(N, device=device)[: min(64, N)]
    y_input = y0[idx].detach()
    y_base = y_input.mean(dim=0, keepdim=True).repeat(y_input.shape[0], 1).detach()

    with torch.enable_grad():
        # Determine gene count
        G = int(decoder(y_input[:1]).shape[1])
        A = torch.zeros((G, d), device=device, dtype=y0.dtype)
        # sample subset of genes for speed if large
        gene_subset = list(range(G))
        max_genes = 64
        if G > max_genes:
            gene_subset = gene_subset[:max_genes]

        for g in gene_subset:
            ig = _integrated_gradients(decoder, y_base, y_input, target_gene=g, steps=steps)  # (d,)
            A[g] = ig

        S = (A.t() @ A)
        if abs_score:
            S = S.abs()
        S.fill_diagonal_(0.0)
        S = _normalize_score_matrix(S, mode=score_norm)

        flat = S.flatten()
        k = min(int(top_k), int(flat.numel()))
        vals, inds = torch.topk(flat, k=k, largest=True, sorted=True)
        src = inds // S.shape[1]
        dst = inds % S.shape[1]
        edge_index = torch.stack([src, dst], dim=0).to(torch.long)
        return EdgeScores(edge_index=edge_index, scores=vals.detach(), method="ig_proxy")




def _cycle_jacobian_matrix(
    encoder: nn.Module,
    decoder: nn.Module,
    x0: torch.Tensor,
    *,
    max_cells: int,
    aggregate_abs: bool = True,
) -> torch.Tensor:
    """Return directed influence matrix M (G,G) for one random cell subsample.

    Important implementation detail:
    - We aggregate *per-cell* encoder/decoder Jacobian factors and average only once.
    - The previous implementation averaged inside the gradient and again afterwards,
      which shrank scores by roughly O(B) and caused visible score collapse.
    """
    device = x0.device
    x0 = x0.detach()
    if x0.ndim != 2:
        raise ValueError(f"x0 must be (N,G), got {tuple(x0.shape)}")
    N, G = x0.shape
    B = int(min(max_cells, N))
    idx = torch.randperm(N, device=device)[:B]
    xb = x0[idx].detach().requires_grad_(True)

    with torch.enable_grad():
        yb = encoder(xb)  # (B, d)
        if yb.ndim != 2:
            raise ValueError(f"encoder(x) must be 2D, got {tuple(yb.shape)}")
        d = int(yb.shape[1])

        def _dec(inp: torch.Tensor) -> torch.Tensor:
            return decoder(inp)

        y_in = yb.detach().requires_grad_(True)
        M = torch.zeros((G, G), device=device, dtype=xb.dtype)

        for i in range(d):
            s = yb[:, i].sum()
            g = torch.autograd.grad(s, xb, retain_graph=True, create_graph=False)[0]  # (B,G)

            v = torch.zeros_like(y_in)
            v[:, i] = 1.0
            _, jvp_out = torch.autograd.functional.jvp(_dec, (y_in,), (v,), create_graph=False, strict=False)
            if aggregate_abs:
                M = M + torch.einsum("bg,bh->gh", jvp_out.abs(), g.abs()) / float(B)
            else:
                M = M + torch.einsum("bg,bh->gh", jvp_out, g) / float(B)

        return M


def infer_edges_from_cycle_jacobian(
    encoder: nn.Module,
    decoder: nn.Module,
    x0: torch.Tensor,
    *,
    abs_score: bool = True,
    top_k: int = 2000,
    max_cells: int = 32,
    n_subsamples: int = 1,
    subsample_agg: Literal["mean", "median"] = "mean",
    score_norm: str = "none",
) -> EdgeScores:
    """Infer **directed** gene->gene edges via local linearization of encoder+decoder.

    We approximate the end-to-end influence:
        x -> y = encoder(x) -> x_hat = decoder(y)
    by the Jacobian product:
        M = (d x_hat / d y) @ (d y / d x)
      where M has shape (G, G) and entry M[dst, src] estimates src->dst effect.

    To stabilize rankings, we can repeat the estimate over multiple random cell subsamples
    (n_subsamples) and fuse the resulting |M| matrices by mean/median.

    Returns:
      EdgeScores with edge_index = [src, dst] and directed method tag.
    """
    device = x0.device
    x0 = x0.detach()
    if x0.ndim != 2:
        raise ValueError(f"x0 must be (N,G), got {tuple(x0.shape)}")
    N, G = x0.shape

    n_sub = int(max(1, n_subsamples))
    Ms = []
    for _ in range(n_sub):
        M = _cycle_jacobian_matrix(encoder, decoder, x0, max_cells=int(max_cells), aggregate_abs=bool(abs_score))
        if abs_score:
            M = M.abs()
        M.fill_diagonal_(0.0)
        Ms.append(M)

    if len(Ms) == 1:
        M_fused = Ms[0]
    else:
        stack = torch.stack(Ms, dim=0)  # (S,G,G)
        if str(subsample_agg).lower() == "median":
            M_fused = stack.median(dim=0).values
        else:
            M_fused = stack.mean(dim=0)

    M_fused = _normalize_score_matrix(M_fused, mode=score_norm)
    flat = M_fused.flatten()
    k = min(int(top_k), int(flat.numel()))
    vals, inds = torch.topk(flat, k=k, largest=True, sorted=True)
    dst = inds // G
    src = inds % G
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)
    return EdgeScores(edge_index=edge_index, scores=vals.detach(), method="cycle_jacobian")



def infer_edges(
    decoder: nn.Module,
    y0: torch.Tensor,
    cfg: EdgeInferenceCfg,
    *,
    encoder: Optional[nn.Module] = None,
    x0: Optional[torch.Tensor] = None,
) -> EdgeScores:
    if cfg.method == "jacobian":
        return infer_edges_from_decoder_jacobian(decoder, y0, abs_score=cfg.abs_score, top_k=cfg.top_k, score_norm=getattr(cfg, "score_norm", "none"))
    if cfg.method == "ig":
        return infer_edges_from_ig(decoder, y0, abs_score=cfg.abs_score, top_k=cfg.top_k, steps=cfg.ig_steps, score_norm=getattr(cfg, "score_norm", "none"))
    if cfg.method == "cycle_jacobian":
        if encoder is None or x0 is None:
            raise ValueError("cycle_jacobian requires encoder and x0")
        return infer_edges_from_cycle_jacobian(
            encoder,
            decoder,
            x0,
            abs_score=cfg.abs_score,
            top_k=cfg.top_k,
            max_cells=int(getattr(cfg, "cycle_max_cells", 32)),
            n_subsamples=int(getattr(cfg, "n_subsamples", 1) or 1),
            subsample_agg=str(getattr(cfg, "subsample_agg", "mean") or "mean"),
            score_norm=str(getattr(cfg, "score_norm", "none") or "none"),
        )
    raise ValueError(f"Unknown edge inference method: {cfg.method}")
