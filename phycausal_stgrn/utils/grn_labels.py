from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Tuple

import torch



def _norm(g: str, mode: str = "upper") -> str:
    s = str(g).strip()
    if mode == "upper":
        return s.upper()
    if mode == "lower":
        return s.lower()
    return s

@dataclass
class GoldGRN:
    gene_names: Tuple[str, ...]
    edge_index: torch.Tensor  # (2, E_gold)
    labels: torch.Tensor      # (E_gold,) 0/1


def load_gold_grn_csv(path: str) -> GoldGRN:
    edges = []
    genes = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["src_gene"]; t = row["dst_gene"]
            y = float(row.get("label", "1"))
            edges.append((s, t, y))
            genes.add(s); genes.add(t)

    gene_names = tuple(sorted(genes))
    gid = {g: i for i, g in enumerate(gene_names)}
    src = torch.tensor([gid[s] for s, _, _ in edges], dtype=torch.long)
    dst = torch.tensor([gid[t] for _, t, _ in edges], dtype=torch.long)
    lab = torch.tensor([y for *_, y in edges], dtype=torch.float32)
    return GoldGRN(gene_names=gene_names, edge_index=torch.stack([src, dst], dim=0), labels=lab)


def align_gold_to_pred(
    pred_edge_index: torch.Tensor,
    pred_gene_names: Tuple[str, ...],
    gold: GoldGRN,
    *,
    normalize: str = "upper",
) -> torch.Tensor:
    gold_pos = set()
    for k in range(gold.edge_index.shape[1]):
        if float(gold.labels[k].item()) > 0.5:
            s = _norm(gold.gene_names[int(gold.edge_index[0, k])], mode=normalize)
            t = _norm(gold.gene_names[int(gold.edge_index[1, k])], mode=normalize)
            gold_pos.add((s, t))

    labels = torch.zeros((pred_edge_index.shape[1],), dtype=torch.float32)
    for e in range(pred_edge_index.shape[1]):
        s = _norm(pred_gene_names[int(pred_edge_index[0, e])], mode=normalize)
        t = _norm(pred_gene_names[int(pred_edge_index[1, e])], mode=normalize)
        labels[e] = 1.0 if (s, t) in gold_pos else 0.0
    return labels
