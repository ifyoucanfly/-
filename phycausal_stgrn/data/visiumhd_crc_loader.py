from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence, Any

import numpy as np

from phycausal_stgrn.data.sdata_utils import (
    DEFAULT_CAF_MARKERS,
    build_regular_grid_coords,
    caf_density_from_markers,
    gaussian_smooth_grid,
    normalize_minmax,
    aggregate_to_grid,
    normalize_log1p_cpm,  # kept for backward compatibility
)

logger = logging.getLogger(__name__)


DEFAULT_TGFB_BARRIER_MARKERS: Tuple[str, ...] = (
    "TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "COL1A1", "COL1A2", "COL3A1", "FN1", "POSTN",
)
DEFAULT_CXCL12_AXIS_MARKERS: Tuple[str, ...] = (
    "CXCL12", "CXCR4", "CXCR7", "ACKR3", "VCAN",
)
DEFAULT_IMMUNE_INFILTRATION_MARKERS: Tuple[str, ...] = (
    "PTPRC", "CD3D", "CD3E", "TRBC1", "TRBC2", "NKG7", "CCL5", "CXCL9", "CXCL10", "CXCL11",
)
DEFAULT_BARRIER_FORCE_KEEP: Tuple[str, ...] = tuple(sorted(set(DEFAULT_CAF_MARKERS + DEFAULT_TGFB_BARRIER_MARKERS + DEFAULT_CXCL12_AXIS_MARKERS + DEFAULT_IMMUNE_INFILTRATION_MARKERS + (
    "EPCAM", "KRT8", "KRT18", "KRT19", "VIM", "MMP2", "MMP9", "ITGA4", "ITGB1", "TAGLN", "ACTA2", "PDGFRB"
))))
from phycausal_stgrn.utils.gold_sanity import inspect_gold_file, sanitize_gold_pairs


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Compatible getter for dict / dataclass-like objects.

    Also supports nested configs where data-related keys live under cfg["data"].
    """
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg.get(key, default)
        # common pattern: cfg["data"][key]
        d = cfg.get("data", None)
        if isinstance(d, dict) and key in d:
            return d.get(key, default)
        return default
    return getattr(cfg, key, default)


def _apply_env_map(raw_labels, env_map: Optional[Dict[str, object]]) -> Optional[np.ndarray]:
    """Map raw string/categorical labels to {1,0,-1} using env_map.

    env_map schema:
      env1_labels: [..]  -> mapped to 1
      env2_labels: [..]  -> mapped to 0
      ignore_labels: [..] (optional) -> mapped to -1 explicitly
    """
    if env_map is None:
        return None
    env1 = set([str(x).strip().lower() for x in env_map.get("env1_labels", [])])
    env2 = set([str(x).strip().lower() for x in env_map.get("env2_labels", [])])
    ign = (
        set([str(x).strip().lower() for x in env_map.get("ignore_labels", [])])
        if isinstance(env_map.get("ignore_labels", None), (list, tuple))
        else set()
    )

    lab = np.asarray(raw_labels).astype(str)
    mapped = np.full((lab.shape[0],), -1, dtype=np.int64)

    for i, s in enumerate(lab):
        k = str(s).strip().lower()
        if k in env1:
            mapped[i] = 1
        elif k in env2:
            mapped[i] = 0
        elif k in ign:
            mapped[i] = -1
        else:
            mapped[i] = -1
    return mapped.reshape(-1, 1)




def _normalize_annotation_semantics(raw_labels, cfg: Any) -> Optional[np.ndarray]:
    env_map = _cfg_get(cfg, "env_map", None)
    if env_map is not None:
        return _apply_env_map(raw_labels, env_map)
    ann_map = _cfg_get(cfg, "annotation_map", None)
    pos_labels = _cfg_get(cfg, "annotation_positive_labels", None)
    ignore_labels = _cfg_get(cfg, "ignore_labels", None)
    lab = np.asarray(raw_labels).astype(str)
    key = np.asarray([str(x).strip().lower() for x in lab], dtype=object)
    if isinstance(ann_map, dict) and len(ann_map) > 0:
        lut = {str(k).strip().lower(): int(v) for k, v in ann_map.items()}
        mapped = np.asarray([lut.get(k, -1) for k in key], dtype=np.int64)
        mapped = np.where(np.isin(mapped, [-1, 0, 1]), mapped, -1)
        return mapped.reshape(-1, 1)
    if isinstance(pos_labels, (list, tuple, set)) and len(pos_labels) > 0:
        pos = {str(x).strip().lower() for x in pos_labels}
        ign = {str(x).strip().lower() for x in (ignore_labels or [])} if isinstance(ignore_labels, (list, tuple, set)) else set()
        mapped = np.full((lab.shape[0],), 0, dtype=np.int64)
        mapped[np.asarray([k in pos for k in key], dtype=bool)] = 1
        if ign:
            mapped[np.asarray([k in ign for k in key], dtype=bool)] = -1
        return mapped.reshape(-1, 1)
    return None

def _annotation_interface_score(coords: np.ndarray, annotations: Optional[np.ndarray], k: int = 12) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    n = int(coords.shape[0])
    if annotations is None or n <= 1:
        return np.zeros((n,), dtype=np.float32)
    ann = np.asarray(annotations).reshape(-1).astype(np.int64, copy=False)
    kk = int(max(1, min(int(k), n - 1)))
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(coords)
        try:
            _, idx = tree.query(coords, k=kk + 1, workers=-1)
        except TypeError:
            _, idx = tree.query(coords, k=kk + 1)
        nbr = np.asarray(idx[:, 1:], dtype=np.int64)
    except Exception:
        import torch
        d = torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords))
        nbr = torch.topk(d, k=kk + 1, largest=False).indices[:, 1:].cpu().numpy().astype(np.int64)
    nbr_ann = ann[nbr]
    valid = (ann[:, None] >= 0) & (nbr_ann >= 0)
    diff = (nbr_ann != ann[:, None]) & valid
    denom = np.maximum(valid.sum(axis=1), 1)
    return (diff.sum(axis=1) / denom).astype(np.float32)

def _knn_mean_feature(coords: np.ndarray, values: np.ndarray, k: int = 12) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(coords.shape[0])
    if n <= 1:
        return v.astype(np.float32)
    kk = int(max(1, min(int(k), n - 1)))
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(coords)
        try:
            _, idx = tree.query(coords, k=kk + 1, workers=-1)
        except TypeError:
            _, idx = tree.query(coords, k=kk + 1)
        nbr = np.asarray(idx[:, 1:], dtype=np.int64)
    except Exception:
        import torch
        d = torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords))
        nbr = torch.topk(d, k=kk + 1, largest=False).indices[:, 1:].cpu().numpy().astype(np.int64)
    return v[nbr].mean(axis=1).astype(np.float32)

def _local_density_proxy(coords: np.ndarray, k: int = 12) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    n = int(coords.shape[0])
    if n <= 2:
        return np.zeros((n,), dtype=np.float32)
    kk = int(max(2, min(int(k), n - 1)))
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(coords)
        try:
            d, _ = tree.query(coords, k=kk + 1, workers=-1)
        except TypeError:
            d, _ = tree.query(coords, k=kk + 1)
        rad = np.asarray(d[:, -1], dtype=np.float32)
    except Exception:
        import torch
        d = torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords))
        rad = torch.topk(d, k=kk + 1, largest=False).values[:, -1].cpu().numpy().astype(np.float32)
    dens = 1.0 / np.maximum(rad, 1e-3)
    dens = dens - dens.min()
    return (dens / max(float(dens.max()), 1e-6)).astype(np.float32)


def _safe_minmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(np.nanmin(x)); hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + 1e-12)).astype(np.float32)

def _marker_proxy_from_markers(
    X: np.ndarray,
    gene_names: Sequence[str],
    markers: Sequence[str],
) -> np.ndarray:
    name_to_idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
    idx = [name_to_idx[m.upper()] for m in markers if m.upper() in name_to_idx]
    if len(idx) == 0:
        return np.zeros((X.shape[0],), dtype=np.float32)
    vals = np.asarray(X[:, idx], dtype=np.float32)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    return vals.mean(axis=1).astype(np.float32)

def _resolved_marker_tuple(raw, default_markers: Tuple[str, ...]) -> Tuple[str, ...]:
    if raw is None:
        return tuple(default_markers)
    if isinstance(raw, str):
        return tuple([g.strip() for g in raw.split(',') if g.strip()])
    try:
        return tuple(str(g) for g in raw if str(g).strip())
    except Exception:
        return tuple(default_markers)

def _default_barrier_force_keep(cfg) -> Tuple[str, ...]:
    extra = _cfg_get(cfg, "barrier_force_keep_genes", None)
    if extra is None:
        return DEFAULT_BARRIER_FORCE_KEEP
    if isinstance(extra, str):
        extra = [g.strip() for g in extra.split(',') if g.strip()]
    try:
        merged = tuple(sorted(set(DEFAULT_BARRIER_FORCE_KEEP) | {str(g) for g in extra if str(g).strip()}))
        return merged
    except Exception:
        return DEFAULT_BARRIER_FORCE_KEEP

def _build_crc_environment_features(coords_out: np.ndarray, caf_norm: np.ndarray, mapped: Optional[np.ndarray], X: Optional[np.ndarray] = None, gene_names: Optional[Sequence[str]] = None, cfg: Optional[object] = None) -> np.ndarray:
    caf = np.asarray(caf_norm, dtype=np.float32).reshape(-1)
    interface = _annotation_interface_score(coords_out, mapped, k=12).astype(np.float32) if mapped is not None else np.zeros_like(caf, dtype=np.float32)
    local_caf = _knn_mean_feature(coords_out, caf, k=12)
    caf_grad = _safe_minmax_np(np.abs(caf - local_caf).astype(np.float32))
    density = _local_density_proxy(coords_out, k=12)
    if interface.size:
        boundary_proximity = _safe_minmax_np(interface * (1.0 + caf_grad))
    else:
        boundary_proximity = np.zeros_like(caf, dtype=np.float32)
    if mapped is not None:
        ann = np.asarray(mapped).reshape(-1).astype(np.int64, copy=False)
        tumor_flag = (ann == 1).astype(np.float32)
        tumor_nbr = _knn_mean_feature(coords_out, tumor_flag, k=12)
    else:
        tumor_nbr = np.zeros_like(caf, dtype=np.float32)

    tgfb_markers = _resolved_marker_tuple(_cfg_get(cfg, "tgfb_barrier_markers", None) if cfg is not None else None, DEFAULT_TGFB_BARRIER_MARKERS)
    cxcl12_markers = _resolved_marker_tuple(_cfg_get(cfg, "cxcl12_axis_markers", None) if cfg is not None else None, DEFAULT_CXCL12_AXIS_MARKERS)
    immune_markers = _resolved_marker_tuple(_cfg_get(cfg, "immune_infiltration_markers", None) if cfg is not None else None, DEFAULT_IMMUNE_INFILTRATION_MARKERS)
    if X is not None and gene_names is not None:
        tgfb_raw = _marker_proxy_from_markers(X, gene_names, tgfb_markers)
        cxcl12_raw = _marker_proxy_from_markers(X, gene_names, cxcl12_markers)
        immune_raw = _marker_proxy_from_markers(X, gene_names, immune_markers)
    else:
        tgfb_raw = np.zeros_like(caf, dtype=np.float32)
        cxcl12_raw = np.zeros_like(caf, dtype=np.float32)
        immune_raw = np.zeros_like(caf, dtype=np.float32)

    tgfb = _safe_minmax_np(_knn_mean_feature(coords_out, tgfb_raw.astype(np.float32), k=12))
    cxcl12 = _safe_minmax_np(_knn_mean_feature(coords_out, cxcl12_raw.astype(np.float32), k=12))
    immune = _safe_minmax_np(_knn_mean_feature(coords_out, immune_raw.astype(np.float32), k=12))
    tgfb_grad = _safe_minmax_np(np.abs(tgfb - _knn_mean_feature(coords_out, tgfb, k=12)))
    cxcl12_grad = _safe_minmax_np(np.abs(cxcl12 - _knn_mean_feature(coords_out, cxcl12, k=12)))
    immune_grad = _safe_minmax_np(np.abs(immune - _knn_mean_feature(coords_out, immune, k=12)))
    barrier_program = _safe_minmax_np(0.42 * caf + 0.24 * boundary_proximity + 0.20 * tgfb + 0.14 * cxcl12)
    barrier_vs_immune = _safe_minmax_np(np.clip(barrier_program - 0.55 * immune, a_min=0.0, a_max=None))

    return np.stack([
        caf,
        interface,
        caf_grad,
        density,
        boundary_proximity,
        tumor_nbr,
        tgfb,
        tgfb_grad,
        cxcl12,
        cxcl12_grad,
        immune,
        immune_grad,
        barrier_program,
        barrier_vs_immune,
    ], axis=1).astype(np.float32)
@dataclass
class VisiumHDCRCConfig:
    """Visium HD CRC CSV loader implementing: global labeling, local filtering."""
    annotation_csv: str

    # Optional: gold GRN edgelist to bias gene pool selection (reduces n_pos=0 in evaluation)
    gold_grn_path: Optional[str] = None

    # final HVG subset size for ODE training
    num_genes: int = 128

    # Force-keep genes from the gold GRN (top-degree) inside the final num_genes set.
    # This is critical for making gold evaluation non-degenerate (avoids n_pos≈0/1).
    # Set to 0 to disable.
    gold_force_keep: int = 0

    # Optional: ensure that at least this fraction (0~1) of the final num_genes
    # come from gold gene universe (top-degree pool). If set >0, we will add
    # gold genes (ranked by variance within this dataset) into the forced set.
    restrict_gene_pool_to_gold_ratio: float = 0.0

    # for global labeling (before subsetting)
    cluster_hvg: int = 2000
    leiden_resolution: float = 0.6
    leiden_seed: int = 0
    unknown_margin: float = 0.15

    # Optional coarse-graining (N control)
    grid_um: float = 20.0
    bin_to_grid: bool = True
    agg: str = "mean"

    # normalization behavior: "log1p_cpm" | "none" | "auto"
    #   - auto: if X has negatives => skip log1p/CPM; else do log1p_cpm
    normalize: str = "auto"

    # env_map: Tumor->1, Stroma->0, Unknown->-1 (recommended)
    env_map: Optional[Dict[str, object]] = None
    annotation_positive_labels: Optional[Tuple[str, ...]] = None
    annotation_map: Optional[Dict[str, int]] = None
    ignore_labels: Optional[Tuple[str, ...]] = None

    # zE smoothing proxy
    smooth_sigma: float = 1.2
    smooth_radius: int = 3
    caf_markers: Tuple[str, ...] = DEFAULT_CAF_MARKERS

    # Optional hard cap for N
    max_cells: int = 0
    roi_mode: str = "none"      # "none" | "center" | "random"
    roi_center_frac: float = 0.4
    roi_seed: int = 0

    # For huge CSVs: only load a bounded gene pool for labeling+HVG, then shrink to num_genes.
    # This prevents allocating (N, 18k) matrices on CPU.
    max_gene_pool: int = 3000

    # HVG flavors to try (first one that succeeds is used). If X contains negatives
    # (already transformed), Scanpy HVG is skipped and we fall back to variance.
    hvg_flavors: Optional[Tuple[str, ...]] = None

    # When roi_mode=center and the ROI is dense, stop reading once we collected
    # roi_collect_multiplier * max_cells rows to reduce I/O.
    roi_collect_multiplier: int = 3

    # Optional marker overrides
    tumor_markers: Optional[Tuple[str, ...]] = None
    stroma_markers: Optional[Tuple[str, ...]] = None


def _infer_coords(df) -> np.ndarray:
    cand = [
        ("x", "y"),
        ("center_x", "center_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
    ]
    for a, b in cand:
        if a in df.columns and b in df.columns:
            return df[[a, b]].values.astype(np.float32)
    raise ValueError(f"Cannot infer coords from columns. Need one of: {cand}. Found: {list(df.columns)[:40]}...")


def _infer_expression(df):
    meta_like = {
        "barcode", "spot_id", "cell_id", "fov", "slide", "sample", "tissue",
        "annotation", "pathologist_annotation", "label", "class", "cluster",
        "x", "y", "center_x", "center_y", "pxl_col_in_fullres", "pxl_row_in_fullres", "array_col", "array_row",
        "in_tissue", "row", "col", "time",
    }
    num = df.select_dtypes(include=["number"])
    keep_cols = [c for c in num.columns if c.lower() not in meta_like]
    if len(keep_cols) == 0:
        return None, None
    X = num[keep_cols].values.astype(np.float32)
    gene_names = tuple(map(str, keep_cols))
    return X, gene_names


def _find_coord_cols(columns: Sequence[str]) -> Tuple[str, str]:
    """Find coordinate column names from header."""
    cand = [
        ("x", "y"),
        ("center_x", "center_y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
    ]
    cols = set(map(str, columns))
    for a, b in cand:
        if a in cols and b in cols:
            return a, b
    raise ValueError(f"Cannot find coords in columns. Need one of {cand}.")


def _find_annotation_col(columns: Sequence[str]) -> Optional[str]:
    cand = ["annotation", "pathologist_annotation", "label", "class", "cluster"]
    cols_lower = {str(c).lower(): str(c) for c in columns}
    for c in cand:
        if c in cols_lower:
            return cols_lower[c]
    return None


def _load_gold_gene_pool(gold_grn_path: Optional[str], max_keep: int = 2500) -> Tuple[str, ...]:
    """Load a bounded set of genes from a gold GRN edgelist.

    Supports common column pairs:
      - TF/Target
      - source/target
      - regulator/target
    Picks top genes by degree to stay within max_keep.
    """
    if not gold_grn_path:
        return tuple()
    import os
    if isinstance(gold_grn_path, str) and (not os.path.exists(gold_grn_path)):
        # Helpful warning: config points to a path that does not exist.
        base = os.path.basename(gold_grn_path)
        candidates = [
            base,
            os.path.join(os.getcwd(), base),
            os.path.join(os.path.dirname(__file__), "assets", base),
        ]
        logger.warning(
            "gold_grn_path does not exist: %s. Candidates to check: %s",
            str(gold_grn_path),
            ", ".join([c for c in candidates if c]),
        )
        return tuple()
    try:
        import pandas as pd

        res = sanitize_gold_pairs(gold_grn_path)
        if not res.rows:
            return tuple()
        from collections import Counter
        ct = Counter()
        for s, t in res.rows:
            ct.update([s, t])
        top = [g for g, _ in ct.most_common(int(max_keep))]
        return tuple(top)
    except Exception as e:
        logger.warning("Failed to load gold_grn_path for gene pooling (%s).", str(e))
        return tuple()


def _select_gene_pool_from_header(
    header_cols: Sequence[str],
    cfg: VisiumHDCRCConfig,
    tumor_markers: Sequence[str],
    stroma_markers: Sequence[str],
    caf_markers: Sequence[str],
) -> Tuple[str, ...]:
    """Select a bounded gene pool to load from a wide CSV.

    Priority:
      1) Tumor/Stroma/CAF markers (for labeling + zE)
      2) Gold GRN genes (bias toward evaluable positives)
      3) Fallback: first genes in header
    """
    meta_like = {
        "barcode", "spot_id", "cell_id", "fov", "slide", "sample", "tissue",
        "annotation", "pathologist_annotation", "label", "class", "cluster",
        "x", "y", "center_x", "center_y", "pxl_col_in_fullres", "pxl_row_in_fullres", "array_col", "array_row",
        "in_tissue", "row", "col", "time",
    }

    # candidate gene columns from header (unknown dtypes at header stage)
    gene_cols = [c for c in header_cols if str(c).lower() not in meta_like]
    if not gene_cols:
        return tuple()

    # Build case-insensitive map to original column names
    up_map = {}
    for c in gene_cols:
        uc = str(c).upper()
        if uc not in up_map:
            up_map[uc] = str(c)

    pool: list[str] = []

    def _add(genes: Sequence[str]):
        for g in genes:
            ug = str(g).upper()
            if ug in up_map:
                col = up_map[ug]
                if col not in pool:
                    pool.append(col)

    # 1) markers
    _add(tuple(tumor_markers) + tuple(stroma_markers) + tuple(caf_markers))

    # 2) gold genes (bounded)
    gold_pool = _load_gold_gene_pool(_cfg_get(cfg, "gold_grn_path", None), max_keep=int(_cfg_get(cfg, "max_gene_pool", 3000) or 3000))
    _add(gold_pool)

    # 3) fallback to fill up to max_gene_pool
    max_pool = int(_cfg_get(cfg, "max_gene_pool", 3000) or 3000)
    if len(pool) < min(max_pool, len(gene_cols)):
        for c in gene_cols:
            if c not in pool:
                pool.append(str(c))
            if len(pool) >= max_pool:
                break

    return tuple(pool)


def _read_csv_roi_subset(
    path: str,
    xcol: str,
    ycol: str,
    ann_col: Optional[str],
    gene_cols: Sequence[str],
    max_cells: int,
    roi_mode: str,
    roi_center_frac: float,
    roi_seed: int,
    collect_multiplier: int = 3,
    chunksize: int = 200_000,
) -> "object":
    """Read a bounded subset of a wide CSV without materializing the full matrix.

    Two-pass strategy (center ROI):
      pass1: stream coords to get bbox
      pass2: stream selected columns and collect ROI rows
    """
    import pandas as pd

    roi_mode = str(roi_mode).lower()
    rng = np.random.default_rng(int(roi_seed))

    # If max_cells<=0, interpret as "no cap" (may be very large / slow on huge CSVs).
    # We keep the semantics but warn loudly.
    if int(max_cells) <= 0:
        logger.warning("max_cells<=0 (no cap). This may be slow or OOM on huge CSVs. Consider setting max_cells.")
        max_cells = int(1e9)

    # Pass 1: bbox (only needed for center ROI)
    if roi_mode in ("center", "multicenter"):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for chunk in pd.read_csv(path, usecols=[xcol, ycol], chunksize=chunksize):
            x = chunk[xcol].to_numpy(dtype=np.float32, copy=False)
            y = chunk[ycol].to_numpy(dtype=np.float32, copy=False)
            if x.size:
                xmin = min(float(np.min(x)), xmin)
                xmax = max(float(np.max(x)), xmax)
                ymin = min(float(np.min(y)), ymin)
                ymax = max(float(np.max(y)), ymax)
        if not np.isfinite([xmin, xmax, ymin, ymax]).all():
            raise ValueError("Failed to infer bbox from coords.")

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        frac = float(np.clip(float(roi_center_frac), 1e-3, 1.0))
        rx = 0.5 * (xmax - xmin) * frac
        ry = 0.5 * (ymax - ymin) * frac
        rx = max(rx, 1e-6)
        ry = max(ry, 1e-6)

        def in_roi(xx, yy):
            return (np.abs(xx - cx) <= rx) & (np.abs(yy - cy) <= ry)
    else:
        in_roi = None  # type: ignore

    # Pass 2: stream selected columns
    usecols = [xcol, ycol] + ([ann_col] if ann_col else []) + list(gene_cols)
    # remove duplicates while preserving order
    seen = set()
    usecols = [c for c in usecols if not (c in seen or seen.add(c))]

    frames = []
    collected = 0
    target_collect = int(max_cells) * max(1, int(collect_multiplier))
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        xx = chunk[xcol].to_numpy(dtype=np.float32, copy=False)
        yy = chunk[ycol].to_numpy(dtype=np.float32, copy=False)
        if roi_mode in ("center", "multicenter") and in_roi is not None:
            m = in_roi(xx, yy)
            if not np.any(m):
                continue
            chunk = chunk.loc[m]
        elif roi_mode == "random":
            # quick approximate sampling: keep a fraction then cap
            if chunk.shape[0] > 0:
                keep_prob = min(1.0, float(max_cells) / float(max(1, chunk.shape[0])))
                m = rng.random(chunk.shape[0]) < keep_prob
                chunk = chunk.loc[m]

        if chunk.shape[0] == 0:
            continue
        frames.append(chunk)
        collected += int(chunk.shape[0])
        if collected >= target_collect:
            break

    if not frames:
        raise ValueError("ROI selection resulted in 0 rows. Consider increasing roi_center_frac or using roi_mode=random.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    if df.shape[0] > int(max_cells) and int(max_cells) < int(1e9):
        idx = rng.choice(df.shape[0], size=int(max_cells), replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    # Ensure annotation column exists
    if ann_col is None:
        df["annotation"] = "Unknown"
    return df


def _case_insensitive_pick(var_names: Sequence[str], genes: Sequence[str]) -> list[str]:
    up_map = {}
    for g in var_names:
        ug = str(g).upper()
        if ug not in up_map:
            up_map[ug] = str(g)
    picked = []
    for g in genes:
        ug = str(g).upper()
        if ug in up_map:
            picked.append(up_map[ug])
    return picked


def _sanitize_X_inplace(adata) -> Dict[str, float]:
    """Make X finite; return quick stats. Does NOT change scale unless needed."""
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        if X.data.size:
            X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            X.eliminate_zeros()
            data_min = float(X.data.min()) if X.data.size else 0.0
            data_max = float(X.data.max()) if X.data.size else 0.0
        else:
            data_min, data_max = 0.0, 0.0
        # implicit zeros mean global min could be 0 even if data_min>0
        global_min = min(0.0, data_min)
        global_max = max(0.0, data_max)
        adata.X = X
    else:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        adata.X = X
        global_min = float(X.min()) if X.size else 0.0
        global_max = float(X.max()) if X.size else 0.0

    return {"min": global_min, "max": global_max}


def _safe_hvg(adata, n_top: int, flavors: Sequence[str] = ("cell_ranger", "seurat_v3", "seurat")) -> np.ndarray:
    """
    Safe HVG selection:
      - if n_top >= n_vars -> mark all as HVG
      - try scanpy highly_variable_genes
      - fallback: variance ranking (works with any real-valued matrix)
    Returns hv_mask (bool array length n_vars).
    """
    import scipy.sparse as sp

    n_top = int(n_top)
    if n_top <= 0:
        hv = np.ones(adata.n_vars, dtype=bool)
        adata.var["highly_variable"] = hv
        return hv

    if adata.n_vars <= n_top:
        hv = np.ones(adata.n_vars, dtype=bool)
        adata.var["highly_variable"] = hv
        return hv

    # If the data has negatives, it is likely already transformed (log/scale/center).
    # Many Scanpy HVG flavors assume non-negative counts. In that case, skip Scanpy HVG.
    try:
        xstats = _sanitize_X_inplace(adata)
        has_neg = float(xstats.get("min", 0.0)) < 0.0
    except Exception:
        has_neg = False

    if not has_neg:
        try:
            import scanpy as sc

            for fl in list(flavors):
                try:
                    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=str(fl), subset=False)
                    hv = adata.var["highly_variable"].to_numpy().astype(bool)
                    if int(hv.sum()) == 0:
                        continue
                    return hv
                except Exception:
                    continue
        except Exception as e1:
            logger.warning("scanpy.highly_variable_genes failed (%s). Fallback to variance ranking.", str(e1))

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mean2 - mean * mean
    else:
        X = np.asarray(X)
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)

    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    top_idx = np.argsort(var)[::-1][:n_top]
    hv = np.zeros(adata.n_vars, dtype=bool)
    hv[top_idx] = True
    adata.var["highly_variable"] = hv
    return hv






def _top_boundary_genes(adata, annotations: np.ndarray, top_k: int = 128) -> list[str]:
    ann = np.asarray(annotations).reshape(-1)
    if ann.size != int(adata.n_obs):
        return []
    tumor = ann == 1
    stroma = ann == 0
    if int(tumor.sum()) < 8 or int(stroma.sum()) < 8:
        return []
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        mt = np.asarray(X[tumor].mean(axis=0)).reshape(-1)
        ms = np.asarray(X[stroma].mean(axis=0)).reshape(-1)
    else:
        X = np.asarray(X)
        mt = X[tumor].mean(axis=0)
        ms = X[stroma].mean(axis=0)
    score = np.abs(np.asarray(mt - ms, dtype=np.float32))
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(score)[::-1][: int(max(0, top_k))]
    return [str(adata.var_names[i]) for i in order]

def _load_gold_edges_upper(
    gold_grn_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load gold edges (src,dst) as uppercase numpy arrays.

    Supports common schemas:
      - src_gene/dst_gene[/label]
      - src/dst[/label]
      - tf/target[/label]
      - source/target[/label]
      - regulator/target[/label]
    If a label column exists, keeps only label != 0.
    """
    if not gold_grn_path:
        return np.asarray([], dtype=object), np.asarray([], dtype=object)

    import os
    if isinstance(gold_grn_path, str) and (not os.path.exists(gold_grn_path)):
        base = os.path.basename(gold_grn_path)
        candidates = [
            base,
            os.path.join(os.getcwd(), base),
            os.path.join(os.path.dirname(__file__), "assets", base),
        ]
        logger.warning(
            "gold_grn_path does not exist: %s. Candidates to check: %s",
            str(gold_grn_path),
            ", ".join([c for c in candidates if c]),
        )
        return np.asarray([], dtype=object), np.asarray([], dtype=object)

    res = sanitize_gold_pairs(gold_grn_path)
    if not res.rows:
        return np.asarray([], dtype=object), np.asarray([], dtype=object)
    src = np.asarray([s for s, _ in res.rows], dtype=object)
    dst = np.asarray([t for _, t in res.rows], dtype=object)
    return src, dst


def _choose_gold_force_genes(
    gold_grn_path: Optional[str],
    present_genes: Sequence[str],
    *,
    hard_keep: int = 0,
    min_pos_edges: int = 200,
    max_force_keep: int = 512,
) -> Tuple[Tuple[str, ...], int]:
    """Choose a set of 'gold-important' genes to force-keep in the final gene subset.

    Strategy:
      - restrict gold edges to genes present in this dataset
      - rank genes by (in+out) degree in gold
      - if hard_keep>0: keep top hard_keep (capped by max_force_keep)
      - else (auto): increase K until the induced gold edge count >= min_pos_edges
    Returns: (genes_upper, induced_edge_count)
    """
    if not gold_grn_path:
        return tuple(), 0

    pres = [str(g) for g in present_genes]
    pres_up_set = set([g.upper() for g in pres])

    src, dst = _load_gold_edges_upper(gold_grn_path)
    if src.size == 0:
        return tuple(), 0

    # restrict to present genes
    m = np.fromiter(((s in pres_up_set) and (t in pres_up_set) for s, t in zip(src, dst)), dtype=bool, count=src.size)
    if m.sum() == 0:
        return tuple(), 0
    src = src[m]; dst = dst[m]

    # degree
    from collections import Counter
    ct = Counter()
    ct.update(src.tolist())
    ct.update(dst.tolist())
    nodes_sorted = [g for g, _ in ct.most_common()]

    max_force_keep = int(max(0, max_force_keep))
    if max_force_keep <= 0:
        return tuple(), 0
    max_force_keep = min(max_force_keep, len(nodes_sorted))

    hard_keep = int(max(0, hard_keep))
    if hard_keep > 0:
        k = min(hard_keep, max_force_keep)
        chosen = tuple(nodes_sorted[:k])
        chosen_set = set(chosen)
        induced = int(np.sum([(s in chosen_set) and (t in chosen_set) for s, t in zip(src, dst)]))
        return chosen, induced

    # auto
    min_pos_edges = int(max(1, min_pos_edges))
    # grow K in geometric steps, but never exceed max_force_keep
    steps = []
    k = 32
    while k < max_force_keep:
        steps.append(k)
        k *= 2
    steps.append(max_force_keep)
    steps = sorted(set([min(max_force_keep, x) for x in steps if x > 0]))

    best = tuple(nodes_sorted[:steps[-1]])
    best_induced = 0
    for k in steps:
        chosen = tuple(nodes_sorted[:k])
        chosen_set = set(chosen)
        induced = int(np.sum([(s in chosen_set) and (t in chosen_set) for s, t in zip(src, dst)]))
        best, best_induced = chosen, induced
        if induced >= min_pos_edges:
            return chosen, induced
    return best, best_induced


def _safe_score_genes(adata, genes: Sequence[str], score_name: str) -> bool:
    import scanpy as sc
    gl = _case_insensitive_pick(adata.var_names, genes)
    if len(gl) < 3:
        adata.obs[score_name] = 0.0
        logger.warning("Marker overlap too small for %s: n=%d (skip scoring).", score_name, len(gl))
        return False
    sc.tl.score_genes(adata, gene_list=gl, score_name=score_name, use_raw=False)
    return True


def _run_leiden(adata, resolution: float, seed: int, key_added: str = "leiden") -> None:
    import scanpy as sc
    try:
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            random_state=int(seed),
            key_added=key_added,
            flavor="leidenalg",
        )
        return
    except Exception as e:
        logger.warning("Leiden failed (%s). Falling back to MiniBatchKMeans on PCA.", str(e))

    from sklearn.cluster import MiniBatchKMeans
    Xp = np.asarray(adata.obsm.get("X_pca", None))
    if Xp is None or Xp.ndim != 2:
        raise RuntimeError("Fallback clustering requires adata.obsm['X_pca'].")
    n_clusters = int(np.clip(round(0.6 * np.sqrt(adata.n_obs)), 2, 30))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=int(seed), batch_size=2048, n_init="auto")
    lab = km.fit_predict(Xp).astype(int)
    adata.obs[key_added] = np.asarray(lab).astype(str)


def _assign_pseudo_pathology_labels(
    adata,
    cluster_key: str,
    tumor_markers: Sequence[str],
    stroma_markers: Sequence[str],
    out_key: str = "annotation",
    unknown_margin: float = 0.15,
) -> Dict[str, str]:
    ok_t = _safe_score_genes(adata, tumor_markers, "_tumor_score")
    ok_s = _safe_score_genes(adata, stroma_markers, "_stroma_score")

    if cluster_key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{cluster_key}'] for pseudo-labeling.")

    df = adata.obs[[cluster_key, "_tumor_score", "_stroma_score"]].copy()
    df["_delta"] = df["_tumor_score"].astype(float) - df["_stroma_score"].astype(float)

    g = (
        df.groupby(cluster_key)
        .agg(
            tumor_score=("_tumor_score", "mean"),
            stroma_score=("_stroma_score", "mean"),
            delta=("_delta", "mean"),
            n=(cluster_key, "size"),
        )
        .sort_values("delta", ascending=False)
    )

    if (not ok_t) and (not ok_s):
        logger.warning("Both Tumor/Stroma marker sets missing. Label all clusters as Unknown.")
        cluster2ann = {str(cl): "Unknown" for cl in g.index.astype(str)}
        adata.obs[out_key] = adata.obs[cluster_key].astype(str).map(cluster2ann).astype("category")
        return cluster2ann

    delta = g["delta"].values
    scale = float(np.std(delta)) if len(delta) > 1 else 1.0
    thr = float(unknown_margin) * max(scale, 1e-6)

    cluster2ann: Dict[str, str] = {}
    for cl, row in g.iterrows():
        if float(row["delta"]) > thr:
            cluster2ann[str(cl)] = "Tumor"
        elif float(row["delta"]) < -thr:
            cluster2ann[str(cl)] = "Stroma"
        else:
            cluster2ann[str(cl)] = "Unknown"

    adata.obs[out_key] = adata.obs[cluster_key].astype(str).map(cluster2ann).astype("category")
    return cluster2ann


def _resolve_annotation_csv_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    parent = p.parent if str(p.parent) not in {"", "."} else Path(".")
    stem = p.stem
    name = p.name
    if "hvg512" in name:
        cand = parent / name.replace("hvg512", "hvg1024")
        if cand.exists():
            logger.warning("annotation_csv not found: %s; falling back to available sibling file: %s", path, cand)
            return str(cand)
    gsm_tokens = [tok for tok in stem.replace('.csv','').split('_') if tok.startswith('GSM')]
    candidates = sorted(parent.glob("*.csv.gz")) + sorted(parent.glob("*.csv"))
    if gsm_tokens:
        gsm = gsm_tokens[0]
        matched = [c for c in candidates if gsm in c.name]
        if len(matched) == 1:
            logger.warning("annotation_csv not found: %s; falling back to GSM-matched file: %s", path, matched[0])
            return str(matched[0])
    raise FileNotFoundError(f"annotation_csv not found: {path}")


def load_visiumhd_crc(cfg: VisiumHDCRCConfig) -> Dict[str, object]:
    """
    Strategy 2: global labeling, local filtering (robust version).

    Key robustness additions vs previous:
      - Auto-detect transformed expression: if X has negatives, skip log1p/CPM to avoid NaNs.
      - Sanitize NaN/Inf in X before any Scanpy ops.
      - Safe HVG selection: if scanpy HVG fails (e.g., pandas.cut bins error), fallback to variance ranking.
      - If n_vars <= requested HVG, skip HVG and use all genes (prevents pd.cut monotonic bins failure).
    """
    import pandas as pd
    import scanpy as sc
    import anndata as ad
    import scipy.sparse as sp

    # -------------------------------------------------
    # Wide CSV safety: stream-load only a bounded subset
    # -------------------------------------------------
    # This prevents allocating gigantic (N, G_full) matrices on CPU.
    annotation_csv = _resolve_annotation_csv_path(str(cfg.annotation_csv))
    cfg.annotation_csv = annotation_csv
    header_cols = pd.read_csv(annotation_csv, nrows=0).columns.tolist()
    xcol, ycol = _find_coord_cols(header_cols)
    ann_col = _find_annotation_col(header_cols)

    tumor_markers = tuple(_cfg_get(cfg, "tumor_markers", None) or (
        "EPCAM", "TACSTD2", "KRT8", "KRT18", "KRT19", "CEACAM5", "CEACAM6", "MKI67", "TOP2A", "MUC1", "MUC13", "VIL1",
        "GUCA2A", "CLCA1", "TFF3",
    ))
    stroma_markers = tuple(_cfg_get(cfg, "stroma_markers", None) or (
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "SPARC", "TAGLN", "ACTA2", "VIM", "PDGFRB", "RGS5", "COL5A1",
        "DDR2", "DPT", "LMOD1", "ACTG2", "DES", "CCDC80", "PODN", "PRELP", "MYLK",
    ))
    caf_markers = tuple(_cfg_get(cfg, "caf_markers", DEFAULT_CAF_MARKERS))

    gene_pool = _select_gene_pool_from_header(header_cols, cfg, tumor_markers, stroma_markers, caf_markers)

    df = _read_csv_roi_subset(
        path=cfg.annotation_csv,
        xcol=xcol,
        ycol=ycol,
        ann_col=ann_col,
        gene_cols=gene_pool,
        max_cells=int(_cfg_get(cfg, "max_cells", 0) or 0),
        roi_mode=str(_cfg_get(cfg, "roi_mode", "none")),
        roi_center_frac=float(_cfg_get(cfg, "roi_center_frac", 0.4) or 0.4),
        roi_seed=int(_cfg_get(cfg, "roi_seed", 0) or 0),
        collect_multiplier=int(_cfg_get(cfg, "roi_collect_multiplier", 3) or 3),
    )

    # normalize annotation column name for downstream
    if ann_col is not None and ann_col != "annotation" and ann_col in df.columns:
        df = df.rename(columns={ann_col: "annotation"})

    coords = _infer_coords(df)
    X, gene_names = _infer_expression(df)
    if X is None:
        raise ValueError(
            "annotation_csv does not contain numeric gene expression columns after subsetting. "
            "Check gene_pool selection or CSV header."
        )

    # Optional grid aggregation (N control) BEFORE Scanpy (now safe because N is capped)
    if bool(_cfg_get(cfg, "bin_to_grid", True)):
        X, coords = aggregate_to_grid(
            X,
            coords,
            gene_names,
            grid_um=float(_cfg_get(cfg, "grid_um", 20.0)),
            agg=str(_cfg_get(cfg, "agg", "mean")),
        )
        logger.info(
            "Applied grid aggregation (grid_um=%.2f, agg=%s). New N=%d",
            float(_cfg_get(cfg, "grid_um", 20.0)),
            str(_cfg_get(cfg, "agg", "mean")),
            int(X.shape[0]),
        )

    # Build AnnData on current gene set (might already be 128 if your CSV was prefiltered)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=np.arange(X.shape[0])),
        var=pd.DataFrame(index=pd.Index(list(gene_names), name="gene")),
    )
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)

    # Sanitize + detect transformation
    stats = _sanitize_X_inplace(adata)
    logger.info("Input X stats: min=%.4g max=%.4g (n_obs=%d n_vars=%d)", stats["min"], stats["max"], int(adata.n_obs), int(adata.n_vars))

    # Auto normalize decision:
    norm_mode = str(_cfg_get(cfg, "normalize", "auto")).lower()
    # If data contains negatives, it is very likely already transformed (log/scale/center).
    if norm_mode == "auto":
        norm_mode = "none" if stats["min"] < 0.0 else "log1p_cpm"

    if norm_mode == "log1p_cpm":
        # Filter zero-count spots to avoid downstream NaNs/degenerate stats
        try:
            sc.pp.filter_cells(adata, min_counts=1)
        except Exception:
            pass
        # normalize + log1p
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # safety: remove any newly created non-finite
        _sanitize_X_inplace(adata)
    elif norm_mode in ("none", "raw"):
        # Do NOT log1p if already transformed (negatives). Just keep as-is.
        # But still remove non-finite.
        _sanitize_X_inplace(adata)
    else:
        raise ValueError(f"Unknown normalize mode: {norm_mode}")

    logger.info("Post-normalization: N=%d, genes(full)=%d", int(adata.n_obs), int(adata.n_vars))
    num_genes = int(_cfg_get(cfg, "num_genes", 128))

    # Optional hard cap / ROI BEFORE expensive clustering (robust)
    max_cells = int(_cfg_get(cfg, "max_cells", 0) or 0)
    if max_cells > 0 and int(adata.n_obs) > max_cells:
        rng = np.random.default_rng(int(_cfg_get(cfg, "roi_seed", 0) or 0))
        mode = str(_cfg_get(cfg, "roi_mode", "none")).lower()
        N = int(adata.n_obs)
        cxy = np.asarray(adata.obsm["spatial"], dtype=np.float32)

        if mode == "random":
            idx = rng.choice(N, size=max_cells, replace=False)
        else:
            frac = float(_cfg_get(cfg, "roi_center_frac", 0.4) or 0.4)
            frac = float(np.clip(frac, 1e-3, 1.0))
            cx, cy = float(np.median(cxy[:, 0])), float(np.median(cxy[:, 1]))
            rx = float((cxy[:, 0].max() - cxy[:, 0].min()) * 0.5 * frac)
            ry = float((cxy[:, 1].max() - cxy[:, 1].min()) * 0.5 * frac)
            rx = max(rx, 1e-6); ry = max(ry, 1e-6)
            in_roi = (np.abs(cxy[:, 0] - cx) <= rx) & (np.abs(cxy[:, 1] - cy) <= ry)
            idx = np.flatnonzero(in_roi)
            if idx.size < max_cells:
                rest = np.setdiff1d(np.arange(N), idx, assume_unique=False)
                need = max_cells - idx.size
                add = rng.choice(rest, size=need, replace=False) if rest.size >= need else rest
                idx = np.concatenate([idx, add])
            elif idx.size > max_cells:
                idx = rng.choice(idx, size=max_cells, replace=False)

        adata = adata[np.asarray(idx, dtype=int)].copy()
        logger.warning("Applied max_cells=%d with roi_mode=%s BEFORE labeling (kept %d/%d).", max_cells, mode, int(adata.n_obs), N)

    # -------------------------
    # Global labeling: HVG(cluster_hvg) -> PCA -> Leiden -> marker score -> annotation
    # -------------------------
    # tumor_markers/stroma_markers were already resolved above for streaming gene pool selection.

    n_hvg_cluster = max(8, int(_cfg_get(cfg, "cluster_hvg", 2000) or 2000))
    flavors = _cfg_get(cfg, "hvg_flavors", None)
    if flavors is None:
        flavors = ("cell_ranger", "seurat_v3", "seurat")
    hv_cluster = _safe_hvg(adata, n_top=min(n_hvg_cluster, adata.n_vars), flavors=tuple(flavors))

    # 执行降维与聚类（删除 scale 防止矩阵变密集爆内存，强制 PCA 仅使用 HVG）
    seed = int(_cfg_get(cfg, "leiden_seed", 0) or 0)
    sc.tl.pca(
        adata,
        n_comps=min(50, adata.n_vars),
        use_highly_variable=True,
        svd_solver="arpack",
        random_state=seed,
    )
    sc.pp.neighbors(
        adata,
        n_neighbors=15,
        n_pcs=min(30, adata.obsm["X_pca"].shape[1]),
        random_state=seed,
    )
    _run_leiden(
        adata,
        resolution=float(_cfg_get(cfg, "leiden_resolution", 0.6) or 0.6),
        seed=seed,
        key_added="leiden",
    )
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)
    _assign_pseudo_pathology_labels(
        adata,
        cluster_key="leiden",
        tumor_markers=tumor_markers,
        stroma_markers=stroma_markers,
        out_key="annotation",
        unknown_margin=float(_cfg_get(cfg, "unknown_margin", 0.15) or 0.15),
    )

    # -------------------------
    # Local filtering: AFTER annotation exists, subset to cfg.num_genes HVGs for downstream training
    # -------------------------
    logger.info("Labeling done; selecting final gene set (num_genes=%d).", int(num_genes))
    if adata.n_vars > num_genes:
        # Variance ranking (robust for any real-valued matrix)
        Xv = adata.X
        if sp.issparse(Xv):
            Xv = Xv.tocsr()
            mean = np.asarray(Xv.mean(axis=0)).ravel()
            mean2 = np.asarray(Xv.multiply(Xv).mean(axis=0)).ravel()
            var = mean2 - mean * mean
        else:
            Xv = np.asarray(Xv)
            var = np.var(Xv, axis=0)
        var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        mean_expr = np.asarray(adata.X.mean(axis=0)).ravel() if sp.issparse(adata.X) else np.asarray(adata.X).mean(axis=0)
        mean_expr = np.nan_to_num(mean_expr, nan=0.0, posinf=0.0, neginf=0.0)
        ann_for_rank = adata.obs["annotation"].astype(str).to_numpy() if "annotation" in adata.obs.columns else None
        if ann_for_rank is not None and np.isin(ann_for_rank, ["Tumor", "Stroma"]).sum() >= 16:
            mask_t = ann_for_rank == "Tumor"
            mask_s = ann_for_rank == "Stroma"
            Xt = adata.X[mask_t]
            Xs = adata.X[mask_s]
            if sp.issparse(Xt):
                mt = np.asarray(Xt.mean(axis=0)).ravel()
                ms = np.asarray(Xs.mean(axis=0)).ravel()
            else:
                mt = np.asarray(Xt).mean(axis=0)
                ms = np.asarray(Xs).mean(axis=0)
            boundary_score = np.abs(mt - ms)
        else:
            boundary_score = np.zeros_like(var)
        def _robust_z(a):
            a = np.asarray(a, dtype=float)
            med = np.median(a)
            mad = np.median(np.abs(a - med))
            return (a - med) / max(1e-6, 1.4826 * mad)
        score = 0.50 * _robust_z(var) + 0.40 * _robust_z(boundary_score) + 0.10 * _robust_z(mean_expr)
        ranked = list(np.asarray(adata.var_names)[np.argsort(score)[::-1]])

        # Force-keep markers + (optionally) top-degree gold genes
        forced: set[str] = set()
        forced.update(_case_insensitive_pick(adata.var_names, tumor_markers))
        forced.update(_case_insensitive_pick(adata.var_names, stroma_markers))
        forced.update(_case_insensitive_pick(adata.var_names, caf_markers))
        forced.update(_case_insensitive_pick(adata.var_names, _default_barrier_force_keep(cfg)))
        boundary_top_k = int(_cfg_get(cfg, "boundary_gene_top_k", 128) or 128)
        ann_for_boundary = adata.obs["annotation"].astype(str).to_numpy()
        boundary_genes = _top_boundary_genes(adata, ann_for_boundary, top_k=boundary_top_k)
        forced.update(_case_insensitive_pick(adata.var_names, boundary_genes))

        gold_force_keep_raw = _cfg_get(cfg, "gold_force_keep", 0)
        try:
            gold_force_keep = int(gold_force_keep_raw or 0)
        except Exception:
            gold_force_keep = 0

        gold_grn_path = _cfg_get(cfg, "gold_grn_path", None)
        gold_min_pos_edges = int(_cfg_get(cfg, "gold_min_pos_edges", 200) or 200)
        gold_max_force_keep = int(_cfg_get(cfg, "gold_max_force_keep", min(int(num_genes), 1024)) or min(int(num_genes), 1024))
        gold_ok = False

        if gold_grn_path:
            try:
                gold_diag = inspect_gold_file(str(gold_grn_path), present_genes=adata.var_names, min_overlap_edges=gold_min_pos_edges)
                gold_ok = bool(gold_diag.get("ok", False))
                if not gold_ok:
                    logger.warning("Gold GRN disabled for gene selection: %s | diag=%s", str(gold_grn_path), gold_diag)
                else:
                    chosen_up, induced = _choose_gold_force_genes(
                        str(gold_grn_path),
                        present_genes=adata.var_names,
                        hard_keep=gold_force_keep,
                        min_pos_edges=gold_min_pos_edges,
                        max_force_keep=gold_max_force_keep,
                    )
                    if len(chosen_up) > 0:
                        forced_gold = _case_insensitive_pick(adata.var_names, chosen_up)
                        forced.update(forced_gold)
                        logger.info(
                            "Gold force-keep: added %d genes (hard_keep_raw=%s, max_force_keep=%d, induced_edges=%d).",
                            int(len(forced_gold)),
                            str(gold_force_keep_raw),
                            int(gold_max_force_keep),
                            int(induced),
                        )
            except Exception as e:
                gold_ok = False
                logger.warning("Gold GRN inspection failed and will be ignored: %s | err=%s", str(gold_grn_path), str(e))

        # Optional: increase gold overlap by forcing a minimum fraction of gold genes.
        try:
            gold_ratio = float(_cfg_get(cfg, "restrict_gene_pool_to_gold_ratio", 0.0) or 0.0)
        except Exception:
            gold_ratio = 0.0
        if gold_ratio > 0.0 and gold_grn_path and gold_ok:
            # Use the same bounded gold pool loader (degree-ranked) and keep high-variance gold genes.
            gold_pool_up = _load_gold_gene_pool(str(gold_grn_path), max_keep=int(_cfg_get(cfg, "max_gene_pool", 3000) or 3000))
            gold_present = _case_insensitive_pick(adata.var_names, gold_pool_up)
            want = int(round(min(1.0, max(0.0, gold_ratio)) * float(num_genes)))
            if want > 0 and len(gold_present) > 0:
                # Add the top-variance gold genes (within this dataset) to forced until reaching 'want'.
                # ranked is variance-sorted descending; keep that order.
                gold_ranked = [g for g in ranked if g in set(gold_present)]
                add = [g for g in gold_ranked if g not in forced][: max(0, want - len([g for g in forced if g in set(gold_present)]))]
                if len(add) > 0:
                    forced.update(add)
                    logger.info("Gold ratio force: added %d genes to reach ~%.0f%% gold coverage in final gene set.", int(len(add)), float(gold_ratio) * 100.0)

        # Deterministic ordering: keep forced genes in ranked order
        selected = [g for g in ranked if g in forced]
        if len(selected) > num_genes:
            logger.warning("Forced gene set size %d exceeds num_genes=%d; truncating by variance rank.", len(selected), int(num_genes))
            selected = selected[:num_genes]

        remaining = [g for g in ranked if g not in forced]
        need = int(max(0, num_genes - len(selected)))
        selected = selected + remaining[:need]

        adata = adata[:, selected].copy()
        logger.info("Selected %d genes after annotation (forced=%d, remaining=%d).", int(adata.n_vars), int(len([g for g in selected if g in forced])), int(need))

    annotations = adata.obs["annotation"].astype(str).to_numpy()
    mapped = _normalize_annotation_semantics(annotations, cfg)
    if mapped is None:
        cats = pd.Categorical(annotations)
        mapped = cats.codes.astype(np.int64).reshape(-1, 1)

    # Export numpy
    X_out = adata.X
    if sp.issparse(X_out):
        X_out = X_out.toarray()
    X_out = np.asarray(X_out, dtype=np.float32)

    coords_out = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    gene_names_out = tuple(map(str, adata.var_names.tolist()))

    # zE: CAF proxy + interface-aware mechanomic surrogates
    caf_raw = caf_density_from_markers(X_out, gene_names_out, caf_markers=_cfg_get(cfg, "caf_markers", DEFAULT_CAF_MARKERS), log1p=False)

    # Grid smoothing can still explode if the selected ROI spans a huge bbox.
    # Use a safety switch: if grid is too large, fallback to kNN Gaussian smoothing on points.
    grid_xy, grid_shape = build_regular_grid_coords(coords_um=coords_out, grid_um=float(_cfg_get(cfg, "grid_um", 20.0)))
    H, W = grid_shape
    grid_cells = int(H) * int(W)
    flat = (grid_xy[:, 1].astype(np.int64) * int(W) + grid_xy[:, 0].astype(np.int64)).astype(np.int64)

    max_grid_cells = int(_cfg_get(cfg, "max_grid_cells", 1_000_000) or 1_000_000)
    if grid_cells <= max_grid_cells:
        grid_sum = np.zeros((grid_cells,), dtype=np.float64)
        grid_cnt = np.zeros((grid_cells,), dtype=np.float64)
        np.add.at(grid_sum, flat, caf_raw.astype(np.float64))
        np.add.at(grid_cnt, flat, 1.0)
        caf_grid = (grid_sum / (grid_cnt + 1e-12)).astype(np.float32)

        caf_smooth = gaussian_smooth_grid(
            caf_grid,
            grid_shape=grid_shape,
            sigma=float(_cfg_get(cfg, "smooth_sigma", 1.2)),
            radius=int(_cfg_get(cfg, "smooth_radius", 3)),
        )
        caf_norm = normalize_minmax(caf_smooth[flat])
    else:
        logger.warning(
            "Grid too large for smoothing (H*W=%d > %d). Falling back to kNN smoothing on points.",
            grid_cells,
            max_grid_cells,
        )
        caf_s = caf_raw.astype(np.float32)
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(np.asarray(coords_out, dtype=np.float32))
            k = int(np.clip(int(_cfg_get(cfg, "smooth_knn_k", 30) or 30), 5, 200))
            d, idx = tree.query(np.asarray(coords_out, dtype=np.float32), k=k)
            sigma = float(_cfg_get(cfg, "smooth_sigma", 1.2) or 1.2) * float(_cfg_get(cfg, "grid_um", 20.0) or 20.0)
            w = np.exp(-(d.astype(np.float32) ** 2) / (2.0 * (sigma ** 2) + 1e-12)).astype(np.float32)
            num = (w * caf_s[idx].astype(np.float32)).sum(axis=1)
            den = w.sum(axis=1) + 1e-12
            caf_norm = normalize_minmax((num / den).astype(np.float32))
        except Exception as e:
            logger.warning("kNN smoothing failed (%s). Using raw CAF proxy.", str(e))
            caf_norm = normalize_minmax(caf_s)

    zE = _build_crc_environment_features(coords_out, caf_norm, mapped.reshape(-1) if mapped is not None else None, X=X_out, gene_names=gene_names_out, cfg=cfg)

    out: Dict[str, object] = {
        "X": X_out,
        "coords": coords_out,
        "gene_names": gene_names_out,
        "zE": zE,
        "annotations": mapped.astype(np.int64),
    }
    return out
