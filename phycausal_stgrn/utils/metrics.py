from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import ot  # POT
except Exception:  # pragma: no cover
    ot = None

logger = logging.getLogger(__name__)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.flatten()
    y = y.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.std(unbiased=False) * vy.std(unbiased=False)).clamp_min(eps)
    return float((vx * vy).mean() / denom)


def _normalize_marginals(a: Optional[torch.Tensor], b: Optional[torch.Tensor], n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    if a is None:
        a_np = np.ones(n, dtype=np.float64) / max(n, 1)
    else:
        a_np = a.detach().cpu().numpy().astype(np.float64)
        a_np = a_np / (a_np.sum() + 1e-12)
    if b is None:
        b_np = np.ones(m, dtype=np.float64) / max(m, 1)
    else:
        b_np = b.detach().cpu().numpy().astype(np.float64)
        b_np = b_np / (b_np.sum() + 1e-12)
    return a_np, b_np


def _compute_cost_matrix(x_src: torch.Tensor, x_tgt: torch.Tensor, p: int = 2) -> np.ndarray:
    if ot is None:
        raise ImportError("POT required to compute Wasserstein/EMD cost. Install pot>=0.9.3")
    xs = x_src.detach().cpu().numpy().astype(np.float64)
    xt = x_tgt.detach().cpu().numpy().astype(np.float64)
    return ot.dist(xs, xt, metric="euclidean") ** p


def wasserstein2_cost(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    p: int = 2,
) -> float:
    C = _compute_cost_matrix(x_src, x_tgt, p=p)
    N, M = C.shape
    a_np, b_np = _normalize_marginals(a, b, N, M)
    if N * M <= 3_000_000:
        return float(ot.emd2(a_np, b_np, C))
    Pi = ot.sinkhorn(a_np, b_np, C, reg=5e-2, numItermax=500)
    return float((Pi * C).sum())


def wasserstein1_cost(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> float:
    C = _compute_cost_matrix(x_src, x_tgt, p=1)
    N, M = C.shape
    a_np, b_np = _normalize_marginals(a, b, N, M)
    if N * M <= 3_000_000:
        Pi = ot.emd(a_np, b_np, C)
    else:
        Pi = ot.sinkhorn(a_np, b_np, C, reg=5e-2, numItermax=500)
    return float((Pi * C).sum())


def emd_cost(
    x_src: torch.Tensor,
    x_tgt: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> float:
    return wasserstein1_cost(x_src, x_tgt, a=a, b=b)


def aupr_edge(scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12) -> float:
    s = scores.detach().cpu().float().numpy()
    y = labels.detach().cpu().float().numpy()
    idx = np.argsort(-s)
    y = y[idx]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, eps)
    rec = tp / np.maximum(tp[-1], eps)
    aupr = np.trapz(prec, rec)
    return float(aupr)


def precision_recall_curve(scores: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = scores.detach().cpu().float().numpy()
    y = labels.detach().cpu().float().numpy()
    idx = np.argsort(-s)
    s = s[idx]
    y = y[idx]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, eps)
    rec = tp / np.maximum(tp[-1], eps)
    thr = s
    return prec, rec, thr


def export_pr_curve_csv(scores: torch.Tensor, labels: torch.Tensor, out_path: str) -> str:
    import csv
    prec, rec, thr = precision_recall_curve(scores, labels)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision", "recall"])
        for t, p, r in zip(thr.tolist(), prec.tolist(), rec.tolist()):
            w.writerow([t, p, r])
    return out_path


def bootstrap_ci(metric_fn, scores: torch.Tensor, labels: torch.Tensor, n_boot: int = 200, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    s = scores.detach().cpu()
    y = labels.detach().cpu()
    n = s.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = torch.from_numpy(rng.integers(0, n, size=n)).long()
        vals.append(metric_fn(s[idx], y[idx]))
    vals = np.array(vals, dtype=np.float64)
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    mid = float(vals.mean())
    return mid, lo, hi
