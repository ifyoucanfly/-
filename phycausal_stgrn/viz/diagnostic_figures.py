from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from phycausal_stgrn.utils.diagnostics import mechanism_drift_heatmap

logger = logging.getLogger(__name__)


def plot_mechanism_drift_heatmap(
    coords: torch.Tensor,
    gate: torch.Tensor,
    out_dir: str,
    fname: str = "mechanism_drift_heatmap.png",
    title: str = "Mechanism drift localization (Gate heatmap)",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    m = mechanism_drift_heatmap(gate).detach().cpu().numpy()
    c = coords.detach().cpu().numpy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    sc = ax.scatter(c[:, 0], c[:, 1], c=m, s=6, alpha=0.9)
    fig.colorbar(sc, ax=ax, label="Gate intensity")
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    out_dir: str,
    fname: str = "pr_curve.png",
    title: str = "GRN edge prediction PR curve",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_gate_signal_overlay(
    coords: torch.Tensor,
    gate: torch.Tensor,
    signal: torch.Tensor,
    out_dir: str,
    *,
    fname: str,
    title: str,
    signal_label: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    c = coords.detach().cpu().numpy()
    g = gate.detach().float().reshape(coords.shape[0], -1).mean(dim=1).cpu().numpy()
    s = signal.detach().float().reshape(-1).cpu().numpy()
    if s.shape[0] != c.shape[0]:
        m = min(s.shape[0], c.shape[0])
        s = s[:m]
        g = g[:m]
        c = c[:m]
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(111)
    sc = ax.scatter(c[:, 0], c[:, 1], c=s, s=8, alpha=0.85)
    q = np.quantile(g, 0.82) if g.size else 0.0
    hi = g >= q
    if hi.any():
        ax.scatter(c[hi, 0], c[hi, 1], s=20, facecolors="none", edgecolors="k", linewidths=0.4, alpha=0.55)
    fig.colorbar(sc, ax=ax, label=signal_label)
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal", adjustable="box")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_coverage_risk_curve(
    coverages: Sequence[float],
    risks: Sequence[float],
    out_dir: str,
    *,
    fname: str,
    title: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(111)
    ax.plot(list(coverages), list(risks), marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_misalignment_robustness_curve(
    deltas_um: Sequence[float],
    metric_series: dict[str, Sequence[float]],
    out_dir: str,
    *,
    fname: str = "misalignment_robustness_curve.png",
    title: str = "Misalignment Robustness Curve",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111)
    xs = list(deltas_um)
    for name, vals in metric_series.items():
        ax.plot(xs, list(vals), marker="o", label=name)
    ax.set_xlabel("Injected shift δ (μm)")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend(loc="best")
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def write_animation_gif(frame_paths: Sequence[str], out_path: str, duration_ms: int = 800) -> Optional[str]:
    frames: List[Image.Image] = []
    for fp in frame_paths:
        if fp and os.path.exists(fp):
            try:
                frames.append(Image.open(fp).convert("RGB"))
            except Exception:
                continue
    if not frames:
        return None
    os.makedirs(str(Path(out_path).parent), exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    logger.info("Saved GIF: %s", out_path)
    return out_path
