from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _auto_find_file(patterns: Tuple[str, ...], base_dir: str = "data") -> Optional[str]:
    b = Path(base_dir)
    for pat in patterns:
        hits = sorted([str(p) for p in b.glob(pat)])
        if hits:
            return hits[0]
    return None


@dataclass
class VisiumHD10xConfig:
    """10x Genomics Visium HD loader."""

    h5_path: str = "auto"
    tissue_positions_parquet: str = "auto"
    annotations_csv: str = "auto"

    barcode_col: str = "barcode"
    x_col: str = "x"
    y_col: str = "y"
    annotation_col: str = "annotation"

    strip_dash_suffix: bool = True

    grid_um: float = 20.0
    smooth_sigma: float = 1.5
    smooth_radius: int = 3
    stroma_labels: Tuple[str, ...] = ("Stroma", "Fibroblast", "Fibroblasts", "CAF", "CAFs")


def _normalize_barcode_series(s: pd.Series, strip_dash_suffix: bool) -> pd.Series:
    s = s.astype(str)
    if strip_dash_suffix:
        s = s.str.replace(r"-\d+$", "", regex=True)
    return s


def _infer_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _gaussian_smooth_grid(values: np.ndarray, H: int, W: int, sigma: float, radius: int) -> np.ndarray:
    v = values.reshape(H, W).astype(np.float32)
    rad = int(max(1, radius))
    sig = float(max(1e-3, sigma))

    xs = np.arange(-rad, rad + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs)
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sig * sig))
    ker = ker / (np.sum(ker) + 1e-12)

    pad = rad
    vp = np.pad(v, pad_width=pad, mode="reflect")
    out = np.zeros_like(vp, dtype=np.float32)

    for i in range(pad, pad + H):
        for j in range(pad, pad + W):
            patch = vp[i - pad : i + pad + 1, j - pad : j + pad + 1]
            out[i, j] = float(np.sum(patch * ker))

    return out[pad : pad + H, pad : pad + W].reshape(-1)


def _read_annotations(path: str, barcode_col: str, annotation_col: str, strip_dash_suffix: bool) -> pd.DataFrame:
    try:
        ann = pd.read_csv(path)
        if ann.shape[1] == 1:
            raise ValueError("single-column")
    except Exception:
        ann = pd.read_csv(path, sep="\t", header=None, names=[barcode_col, annotation_col])

    bc = _infer_col(ann, (barcode_col, "Barcode", "barcodes", "spot_id")) or barcode_col
    ac = _infer_col(ann, (annotation_col, "Annotation", "region", "Region", "cell_type")) or annotation_col

    ann = ann.copy()
    ann[bc] = _normalize_barcode_series(ann[bc], strip_dash_suffix)
    ann = ann.drop_duplicates(subset=[bc]).set_index(bc)
    ann = ann.rename(columns={ac: annotation_col})
    return ann[[annotation_col]]


def load_visiumhd_10x(cfg: VisiumHD10xConfig) -> Dict[str, object]:
    try:
        import scanpy as sc
    except Exception as e:
        raise ImportError("scanpy is required for 10x Visium HD loader.") from e

    h5_path = cfg.h5_path
    pos_path = cfg.tissue_positions_parquet
    ann_path = cfg.annotations_csv

    if str(h5_path).lower() in ("auto", "autodetect", ""):
        h5_path = _auto_find_file(("*filtered_feature_bc_matrix.h5", "*feature_bc_matrix.h5", "*.h5"))
    if str(pos_path).lower() in ("auto", "autodetect", ""):
        pos_path = _auto_find_file(("*tissue_positions.parquet", "*positions*.parquet*", "*.parquet*"))
    if str(ann_path).lower() in ("auto", "autodetect", ""):
        ann_path = _auto_find_file(("*annotations*.csv", "*annotation*.csv", "*squares_annotation.csv", "*.csv"))

    if pos_path is None:
        raise FileNotFoundError("Cannot auto-detect tissue_positions parquet under data/.")
    if ann_path is None:
        raise FileNotFoundError("Cannot auto-detect annotations CSV/TSV under data/.")
    if h5_path is None:
        raise FileNotFoundError("Cannot auto-detect expression .h5 under data/. Place *_filtered_feature_bc_matrix.h5 and retry.")

    logger.info("Reading 10x h5: %s", h5_path)
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    pos = pd.read_parquet(pos_path)
    bc_col = _infer_col(pos, (cfg.barcode_col, "Barcode", "barcodes", "spot_id")) or cfg.barcode_col
    x_col = _infer_col(pos, (cfg.x_col, "X", "pxl_col_in_fullres", "center_x")) or cfg.x_col
    y_col = _infer_col(pos, (cfg.y_col, "Y", "pxl_row_in_fullres", "center_y")) or cfg.y_col

    pos = pos.copy()
    pos[bc_col] = _normalize_barcode_series(pos[bc_col], cfg.strip_dash_suffix)
    pos = pos.drop_duplicates(subset=[bc_col]).set_index(bc_col)

    ann = _read_annotations(ann_path, cfg.barcode_col, cfg.annotation_col, cfg.strip_dash_suffix)

    obs_index = pd.Index(
        _normalize_barcode_series(pd.Series(adata.obs_names), cfg.strip_dash_suffix).values,
        name="barcode",
    )
    adata.obs = adata.obs.copy()
    adata.obs["barcode_raw"] = adata.obs_names.astype(str)
    adata.obs.index = obs_index

    adata.obs = adata.obs.join(pos[[x_col, y_col]], how="left")
    adata.obs = adata.obs.join(ann[[cfg.annotation_col]], how="left")

    coords = adata.obs[[x_col, y_col]].to_numpy(dtype=np.float32)
    keep = np.isfinite(coords).all(axis=1)
    if int(keep.sum()) < int(adata.n_obs):
        adata = adata[keep].copy()
        coords = coords[keep]

    labels = adata.obs[cfg.annotation_col].astype("string").fillna("")
    is_stroma = np.zeros((adata.n_obs,), dtype=bool)
    for lab in cfg.stroma_labels:
        is_stroma |= (labels.str.lower() == str(lab).lower()).to_numpy()

    xmin, ymin = float(coords[:, 0].min()), float(coords[:, 1].min())
    gx = np.floor((coords[:, 0] - xmin) / float(cfg.grid_um)).astype(np.int32)
    gy = np.floor((coords[:, 1] - ymin) / float(cfg.grid_um)).astype(np.int32)
    W = int(gx.max() + 1)
    H = int(gy.max() + 1)
    flat = gy * W + gx

    occ = np.zeros((H * W,), dtype=np.float32)
    cnt = np.zeros((H * W,), dtype=np.float32)
    np.add.at(occ, flat[is_stroma], 1.0)
    np.add.at(cnt, flat, 1.0)
    density = occ / (cnt + 1e-6)

    smooth = _gaussian_smooth_grid(density, H=H, W=W, sigma=float(cfg.smooth_sigma), radius=int(cfg.smooth_radius))
    caf_proxy = smooth[flat].astype(np.float32)
    mn, mx = float(np.min(caf_proxy)), float(np.max(caf_proxy))
    caf_proxy = (caf_proxy - mn) / (mx - mn + 1e-12)

    zE = np.stack([caf_proxy, np.zeros_like(caf_proxy)], axis=1).astype(np.float32)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    return {"X": X, "coords": coords.astype(np.float32), "gene_names": tuple(map(str, list(adata.var_names))), "zE": zE}
