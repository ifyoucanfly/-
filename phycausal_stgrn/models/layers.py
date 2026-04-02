from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for k in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MonotonicNet(nn.Module):
    """
    Simple monotonic network wrt inputs via non-negative weights (softplus parameterization).
    Useful for monotone time or gate constraints.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.w1_raw = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2_raw = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.05)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = F.softplus(self.w1_raw)
        w2 = F.softplus(self.w2_raw)
        h = F.silu(x @ w1 + self.b1)
        return h @ w2 + self.b2


class GraphMessagePassing(nn.Module):
    """
    Lightweight message passing (no external deps).
    Aggregation: mean of transformed neighbor features.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Mean aggregation message passing.

        Args:
            x: Node features of shape (N, d).
            edge_index: Graph edges of shape (2, E), where edge_index[0] are target nodes i
                        and edge_index[1] are source/neighbor nodes j (messages flow j -> i).

        Returns:
            Aggregated node features of shape (N, out_dim).
        """
        # x: (N, d), edge_index: (2, E)
        if edge_index.numel() == 0:
            # No edges: return zeros to keep residual blocks stable.
            return torch.zeros((x.shape[0], self.lin.out_features), device=x.device, dtype=x.dtype)

        i = edge_index[0].to(dtype=torch.long, device=x.device)
        j = edge_index[1].to(dtype=torch.long, device=x.device)

        # Messages on edges: (E, out_dim)
        msg = self.lin(x[j])

        N = x.shape[0]
        out_dim = msg.shape[1]

        # Aggregate onto nodes: (N, out_dim)
        out = torch.zeros((N, out_dim), device=x.device, dtype=msg.dtype)
        out.index_add_(0, i, msg)

        # Degree for mean aggregation: (N, 1)
        deg = torch.zeros((N,), device=x.device, dtype=msg.dtype)
        deg.index_add_(0, i, torch.ones((i.numel(),), device=x.device, dtype=msg.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)

        out = out / deg
        return out


class GNNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mp = GraphMessagePassing(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.mp(x, edge_index)
        x = x + h
        x = x + self.ff(x)
        return x
