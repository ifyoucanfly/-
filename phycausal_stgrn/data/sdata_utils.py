from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CAF_MARKERS: Tuple[str, ...] = (
    "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRB", "ACTA2", "TAGLN",
)

def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k = k / (k.sum() + 1e-12)
    return k

def gaussian_smooth_grid(values: np.ndarray, grid_shape: Tuple[int, int], sigma: float = 1.2, radius: int = 3) -> np.ndarray:
    assert values.shape[0] == grid_shape[0] * grid_shape[1]
    grid = values.reshape(grid_shape).astype(np.float64)
    k = _gaussian_kernel1d(sigma, radius)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=0, arr=grid)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis=1, arr=tmp)
    return out.reshape(-1).astype(np.float32)

def build_regular_grid_coords(coords_um: np.ndarray, grid_um: float = 20.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    x = coords_um[:, 0]
    y = coords_um[:, 1]
    gx = np.floor((x - x.min()) / grid_um).astype(np.int32)
    gy = np.floor((y - y.min()) / grid_um).astype(np.int32)
    H = int(gy.max() + 1)
    W = int(gx.max() + 1)
    return np.stack([gx, gy], axis=1), (H, W)

def caf_density_from_markers(X: np.ndarray, gene_names: Sequence[str], caf_markers: Sequence[str] = DEFAULT_CAF_MARKERS, log1p: bool = True) -> np.ndarray:
    name_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    idx = [name_to_idx[m.upper()] for m in caf_markers if m.upper() in name_to_idx]
    if len(idx) == 0:
        raise ValueError("No CAF markers found in gene_names. Provide correct gene symbols or marker list.")
    caf = X[:, idx].mean(axis=1)
    if log1p:
        caf = np.log1p(caf)
    return caf.astype(np.float32)

def normalize_minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = float(x.min()); hi = float(x.max())
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)

def aggregate_to_grid(
    X: np.ndarray,
    coords_um: np.ndarray,
    gene_names: Sequence[str],
    grid_um: float = 20.0,
    agg: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate cell/bin-level measurements to regular grid bins.

    This implementation aggregates **only visited bins** to avoid allocating a dense
    (H*W, G) matrix, which can explode memory for large coordinate ranges (e.g., Visium HD).

    Args:
      X: (N, G) expression (dense or sparse-like with .toarray()).
      coords_um: (N, 2) coordinates in microns.
      gene_names: kept for signature compatibility (not used).
      grid_um: grid size (microns).
      agg: "mean" or "sum".

    Returns:
      Xb: (Nb, G) aggregated expression (float32).
      coords_b: (Nb, 2) bin-center coordinates in microns (float32).
    """
    if X is None or coords_um is None:
        raise ValueError("X/coords_um cannot be None")
    if X.shape[0] != coords_um.shape[0]:
        raise ValueError(f"X and coords_um must have same N. Got {X.shape[0]} vs {coords_um.shape[0]}")
    if X.shape[0] == 0:
        return X.astype(np.float32, copy=False), coords_um.astype(np.float32, copy=False)

    grid_xy, (H, W) = build_regular_grid_coords(coords_um, grid_um=float(grid_um))
    # Flat bin id (int64) for each observation
    flat = (grid_xy[:, 1].astype(np.int64) * int(W) + grid_xy[:, 0].astype(np.int64)).astype(np.int64)

    uniq, inv = np.unique(flat, return_inverse=True)
    Nb = int(uniq.size)
    G = int(X.shape[1])

    # Dense output over visited bins only (float32)
    Xb = np.zeros((Nb, G), dtype=np.float32)

    # Ensure float32 input
    Xf = np.asarray(X, dtype=np.float32, order="C")
    # Sum within bins (vectorized)
    np.add.at(Xb, inv, Xf)

    if agg == "mean":
        cnt = np.bincount(inv, minlength=Nb).astype(np.float32)
        Xb = Xb / (cnt[:, None] + 1e-12)
    elif agg == "sum":
        pass
    else:
        raise ValueError(f"Unknown agg: {agg}")

    # Bin centers for visited bins only
    gx = (uniq % int(W)).astype(np.float32)
    gy = (uniq // int(W)).astype(np.float32)
    centers = np.stack([gx, gy], axis=1) * float(grid_um)
    centers[:, 0] += float(coords_um[:, 0].min())
    centers[:, 1] += float(coords_um[:, 1].min())

    coords_b = centers.astype(np.float32)
    return Xb.astype(np.float32, copy=False), coords_b

def intersect_genes_across_slices(
    slices: Sequence[Dict[str, object]],
) -> Tuple[Tuple[str, ...], Sequence[np.ndarray]]:
    """Intersect gene sets across slices and return reordered X matrices.

    Each slice dict must contain:
      - "X": np.ndarray (N,G)
      - "gene_names": tuple/list of gene symbols

    Returns:
      genes_common: tuple[str]
      X_list: list[np.ndarray] each with columns aligned to genes_common
    """
    gene_lists = [tuple(map(str, s["gene_names"])) for s in slices]
    common = set(gene_lists[0])
    for gl in gene_lists[1:]:
        common &= set(gl)
    genes_common = tuple(sorted(common))
    if len(genes_common) == 0:
        raise ValueError("No common genes across slices.")
    X_out = []
    for s, gl in zip(slices, gene_lists):
        idx = {g: i for i, g in enumerate(gl)}
        cols = [idx[g] for g in genes_common]
        X_out.append(np.asarray(s["X"])[:, cols].astype(np.float32))
    return genes_common, X_out


def normalize_log1p_cpm(X: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """CPM normalize then log1p. Deterministic and widely accepted for ST."""
    lib = X.sum(axis=1, keepdims=True).astype(np.float64)
    Xn = (X.astype(np.float64) / (lib + 1e-12)) * float(scale)
    return np.log1p(Xn).astype(np.float32)
