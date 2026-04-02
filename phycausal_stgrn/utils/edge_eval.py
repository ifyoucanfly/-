from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from phycausal_stgrn.utils.metrics import aupr_edge
from phycausal_stgrn.utils.gold_sanity import drop_bad_tokens

logger = logging.getLogger(__name__)


def normalize_gene(g: str, mode: str = "upper") -> str:
    """Normalize gene symbols for robust alignment.

    mode:
      - "upper": strip and uppercase
      - "lower": strip and lowercase
      - "none": no normalization
    """
    s = str(g).strip()
    if mode == "upper":
        return s.upper()
    if mode == "lower":
        return s.lower()
    return s


@dataclass
class GoldEdgeSet:
    """Gold standard directed edges as normalized (src,dst) pairs."""
    edges_pos: set[Tuple[str, str]]
    n_rows: int


def load_gold_edges_csv(
    path: str,
    *,
    src_col: str = "src_gene",
    dst_col: str = "dst_gene",
    label_col: str = "label",
    normalize: str = "upper",
    threshold: float = 0.5,
    present_genes: Optional[Sequence[str]] = None,
    mapping_path: Optional[str] = None,
    summary_path: Optional[str] = None,
) -> GoldEdgeSet:
    edges_pos: set[Tuple[str, str]] = set()
    sanitized_path, summary = drop_bad_tokens(
        path,
        present_genes=present_genes,
        mapping_path=mapping_path,
        summary_path=summary_path,
        label_threshold=threshold,
    )
    n = int(summary.get("num_rows", 0) or 0)
    load_path = sanitized_path if sanitized_path and os.path.exists(sanitized_path) else path
    with open(load_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                y = float(row.get(label_col, "1"))
            except Exception:
                y = 1.0
            if y <= threshold:
                continue
            s = normalize_gene(row.get(src_col, ""), mode=normalize)
            t = normalize_gene(row.get(dst_col, ""), mode=normalize)
            if s and t:
                edges_pos.add((s, t))
    logger.info("Loaded sanitized gold edges from %s | summary=%s", load_path, summary)
    return GoldEdgeSet(edges_pos=edges_pos, n_rows=n)


def align_gold_to_pred_edges(
    pred_edge_index: torch.Tensor,
    pred_gene_names: Sequence[str],
    gold_edges: GoldEdgeSet,
    *,
    normalize: str = "upper",
) -> torch.Tensor:
    """Return labels (E,) for predicted edges given gold edge set."""
    if pred_edge_index.ndim != 2 or pred_edge_index.shape[0] != 2:
        raise ValueError(f"pred_edge_index must be (2,E), got {tuple(pred_edge_index.shape)}")

    # Normalize gene names once
    gnames = [normalize_gene(g, mode=normalize) for g in pred_gene_names]

    E = pred_edge_index.shape[1]
    labels = torch.zeros((E,), dtype=torch.float32, device=pred_edge_index.device)
    for e in range(E):
        s = gnames[int(pred_edge_index[0, e])]
        t = gnames[int(pred_edge_index[1, e])]
        labels[e] = 1.0 if (s, t) in gold_edges.edges_pos else 0.0
    return labels


def precision_at_k(y_true: torch.Tensor, y_score: torch.Tensor, k: int) -> float:
    k = int(max(1, min(int(k), int(y_true.numel()))))
    idx = torch.argsort(y_score, descending=True)[:k]
    return float(y_true[idx].sum().item() / float(k))


def roc_auc_safe(y_true: torch.Tensor, y_score: torch.Tensor) -> Optional[float]:
    """Compute AUROC without sklearn. Returns None if undefined."""
    y = y_true.detach().cpu().numpy().astype(np.int32)
    s = y_score.detach().cpu().numpy().astype(np.float64)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    # Rank-based AUROC (equivalent to Mann–Whitney U)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # Handle ties: average ranks within tied groups
    # (stable and fast enough for E~1e3-1e4)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    sum_pos = ranks[y == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def evaluate_edges(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    precision_ks: Tuple[int, ...] = (50, 100, 200),
) -> Dict[str, Optional[float]]:
    """Compute edge metrics on a *sparse candidate edge list* (recommended).

    Designed for GRN settings where the true graph is extremely sparse; evaluating on the full
    dense adjacency can produce degenerate metrics dominated by negatives.
    """
    scores = scores.detach().float()
    labels = labels.detach().float()

    n_pos = int((labels > 0.5).sum().item())
    n_total = int(labels.numel())

    out: Dict[str, Optional[float]] = {
        "n_pos": float(n_pos),
        "n_total": float(n_total),
    }

    if n_pos == 0 or n_pos == n_total:
        out.update(
            {
                "aupr": None,
                "auroc": None,
                "early_precision": None,
            }
        )
        for k in precision_ks:
            out[f"precision@{k}"] = None
        return out

    out["aupr"] = float(aupr_edge(scores, labels))
    out["auroc"] = roc_auc_safe(labels, scores)

    # Early precision: k = n_pos
    out["early_precision"] = precision_at_k(labels, scores, k=n_pos)

    for k in precision_ks:
        out[f"precision@{k}"] = precision_at_k(labels, scores, k=k)

    # Score separation (diagnostic)
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    out["mean_pos_score"] = float(pos.mean().item()) if pos.numel() else None
    out["mean_neg_score"] = float(neg.mean().item()) if neg.numel() else None
    return out


def diagnose_gold_overlap(
    pred_gene_names: Sequence[str],
    gold_edges: GoldEdgeSet,
    *,
    normalize: str = "upper",
    max_report: int = 10,
) -> Dict[str, object]:
    gset = set(normalize_gene(g, mode=normalize) for g in pred_gene_names)
    gold_genes = set()
    for s, t in gold_edges.edges_pos:
        gold_genes.add(s)
        gold_genes.add(t)

    common = gset & gold_genes
    # Count how many gold edges are even evaluable in this gene set
    evaluable = [(s, t) for (s, t) in gold_edges.edges_pos if (s in gset and t in gset)]
    rep = {
        "pred_genes": int(len(gset)),
        "gold_genes": int(len(gold_genes)),
        "common_genes": int(len(common)),
        "gold_pos_edges_total": int(len(gold_edges.edges_pos)),
        "gold_pos_edges_evaluable": int(len(evaluable)),
        "example_evaluable_edges": evaluable[:max_report],
    }
    return rep
