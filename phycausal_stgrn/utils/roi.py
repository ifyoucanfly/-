from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class InterfaceROI:
    mask_interface: torch.Tensor  # (N,) bool
    side_tumor: torch.Tensor      # (N,) bool
    side_stroma: torch.Tensor     # (N,) bool
    x0: Optional[float] = None
    band: Optional[float] = None
    source: str = "x_band"


def define_interface_roi(coords: torch.Tensor, x0: float = 500.0, band: float = 40.0) -> InterfaceROI:
    """Define interface ROI by x-coordinate band around x0."""
    x = coords[:, 0]
    mask = (x >= (x0 - band)) & (x <= (x0 + band))
    tumor = x < x0
    stroma = x >= x0
    return InterfaceROI(mask_interface=mask, side_tumor=tumor, side_stroma=stroma, x0=float(x0), band=float(band), source="x_band")


def define_interface_roi_adaptive(
    coords: torch.Tensor,
    annotations: Optional[torch.Tensor] = None,
    *,
    band_frac: float = 0.06,
    min_band: float = 28.0,
    max_band: float = 120.0,
    target_interface_frac: float = 0.18,
) -> InterfaceROI:
    """Adaptive interface ROI.

    Priority:
      1) Tumor/Stroma-style annotations if available (interface = heterogeneous local neighborhood)
      2) fallback to x-median band on coordinates

    The goal is not pathology-perfect segmentation, but a task-aligned ROI for
    interface-localized mechanism drift / crossing summaries.
    """
    coords = coords.float()
    x = coords[:, 0]
    n = int(coords.shape[0])
    if n == 0:
        return define_interface_roi(coords, x0=0.0, band=float(min_band))

    # Fallback x-band parameters
    x0 = float(torch.median(x).item())
    span = float((x.max() - x.min()).item()) if n > 1 else float(min_band)
    band = float(max(min_band, min(max_band, span * band_frac)))

    if annotations is None:
        return define_interface_roi(coords, x0=x0, band=band)

    ann = annotations
    if ann.ndim == 2 and ann.shape[1] >= 1:
        ann = ann[:, 0]
    ann = ann.reshape(-1).float()
    valid = torch.isfinite(ann)
    uniq = torch.unique(ann[valid]) if valid.any() else torch.tensor([], device=ann.device)
    uniq = uniq[torch.isfinite(uniq)]
    if int(uniq.numel()) < 2:
        return define_interface_roi(coords, x0=x0, band=band)

    # Binary split using median of observed labels
    thr = torch.median(uniq)
    tumor = ann > thr
    stroma = ~tumor

    # Approximate interface by kNN label heterogeneity, then shrink to a compact balanced rim.
    d = torch.cdist(coords, coords)
    k = int(min(12, max(2, n - 1)))
    nn = torch.topk(d, k=k + 1, largest=False).indices[:, 1:]
    nbr_lab = tumor[nn]
    p = nbr_lab.float().mean(dim=1)
    hetero = (p > 0.0) & (p < 1.0)
    hetero_core = (p > 0.18) & (p < 0.82)
    if int(hetero.sum().item()) == 0:
        return define_interface_roi(coords, x0=x0, band=band)

    # Compact 2D boundary neighborhood with preference for strong local mixing and side balance.
    seed = hetero_core if int(hetero_core.sum()) >= 4 else hetero
    x0 = float(torch.median(x[seed]).item()) if int(seed.sum()) > 0 else x0
    y = coords[:, 1]
    y0 = float(torch.median(y[seed]).item()) if int(seed.sum()) > 0 else float(torch.median(y).item())
    dx = (x - x0).abs()
    dy = (y - y0).abs()
    dx = dx / dx[seed].median().clamp_min(1.0)
    dy = dy / dy[seed].median().clamp_min(1.0)
    local_mix = 1.0 - ((p - 0.5).abs() / (p - 0.5).abs().max().clamp_min(1e-6))
    # Reward local mixing and penalize spread away from the compact interface core.
    dist = dx + 0.82 * dy - 0.90 * local_mix
    base = hetero_core if int(hetero_core.sum()) >= 4 else hetero
    base_dist = dist[base]
    q = float(min(max(float(target_interface_frac) / max(float(base.float().mean().item()), 1e-3), 0.05), 0.22))
    thr = torch.quantile(base_dist, q) if int(base_dist.numel()) > 0 else torch.quantile(dist, min(float(target_interface_frac), 0.25))
    narrow = dist <= thr
    hetero_narrow = base & narrow
    if int(hetero_narrow.sum()) < 4:
        target_k = max(4, int(min(max(float(target_interface_frac), 0.08), 0.18) * n))
        order = torch.argsort(dist)
        narrow = torch.zeros(n, dtype=torch.bool, device=coords.device)
        narrow[order[:target_k]] = True
        hetero_narrow = base & narrow
    if int(hetero_narrow.sum()) < 4:
        hetero_narrow = narrow
    # Keep both sides represented so P_cross summaries remain meaningful.
    if int((hetero_narrow & tumor).sum()) < 2 or int((hetero_narrow & stroma).sum()) < 2:
        balanced = hetero & narrow
        if int((balanced & tumor).sum()) >= 2 and int((balanced & stroma).sum()) >= 2:
            hetero_narrow = balanced

    return InterfaceROI(
        mask_interface=hetero_narrow,
        side_tumor=tumor,
        side_stroma=stroma,
        x0=x0,
        band=min(float(max_band), max(float(min_band), float(torch.quantile(dist.clamp_min(0.0), min(float(target_interface_frac), 0.30)).item()))),
        source="annotation_knn_heterogeneity_narrow",
    )


def p_cross_interface(
    y: torch.Tensor,
    roi: InterfaceROI,
    direction: str = "tumor_to_stroma",
) -> torch.Tensor:
    """Compute a calibrated P_cross proxy from latent states within the interface ROI.

    Compared with the earlier ``sigmoid(-||mu_t-mu_s||)`` proxy, this version normalizes
    the between-side separation by the within-side spread so that counterfactual changes
    around the tumor-stroma barrier remain observable on real CRC data.
    """
    mask = roi.mask_interface
    yt = y[mask & roi.side_tumor]
    ys = y[mask & roi.side_stroma]
    if yt.numel() == 0 or ys.numel() == 0:
        return torch.tensor(0.0, device=y.device, dtype=y.dtype)
    dt = yt.mean(dim=0)
    ds = ys.mean(dim=0)
    axis = dt - ds
    axis_norm = axis.norm(p=2).clamp_min(1e-4)
    proj_t = ((yt - ds.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=1) / axis_norm
    proj_s = ((ys - ds.unsqueeze(0)) * axis.unsqueeze(0)).sum(dim=1) / axis_norm
    between = (proj_t.mean() - proj_s.mean()).abs()
    wt = proj_t.std(unbiased=False) if yt.shape[0] > 1 else torch.tensor(0.0, device=y.device, dtype=y.dtype)
    ws = proj_s.std(unbiased=False) if ys.shape[0] > 1 else torch.tensor(0.0, device=y.device, dtype=y.dtype)
    within = 0.5 * (wt + ws)
    skew_t = (proj_t > 0).float().mean()
    skew_s = (proj_s > 0).float().mean()
    directional_mix = 0.5 * (skew_t + (1.0 - skew_s))
    midpoint = 0.5 * (dt + ds)
    mid_t = (yt - midpoint.unsqueeze(0)).norm(dim=1).mean()
    mid_s = (ys - midpoint.unsqueeze(0)).norm(dim=1).mean()
    midpoint_closeness = torch.sigmoid(1.55 * (1.0 - 0.85 * (mid_t + mid_s) / (between + within + y.new_tensor(1e-4))))
    mix = torch.sigmoid(2.10 * ((within - 0.72 * between) / (within + 0.72 * between + y.new_tensor(1e-4))))
    mix = 0.52 * mix + 0.24 * directional_mix + 0.24 * midpoint_closeness
    if direction == "stroma_to_tumor":
        mix = 1.0 - mix
    return mix


@torch.no_grad()
def p_cross_committor(q: torch.Tensor, roi: InterfaceROI, direction: str = "tumor_to_stroma") -> torch.Tensor:
    """Committor-based crossing probability summary (TPT standard)."""
    mask = roi.mask_interface
    if direction == "tumor_to_stroma":
        vals = q[mask & roi.side_tumor]
        return vals.mean() if vals.numel() else torch.tensor(0.0, device=q.device, dtype=q.dtype)
    if direction == "stroma_to_tumor":
        vals = (1.0 - q)[mask & roi.side_stroma]
        return vals.mean() if vals.numel() else torch.tensor(0.0, device=q.device, dtype=q.dtype)
    raise ValueError(f"Unknown direction: {direction}")
