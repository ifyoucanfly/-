from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from phycausal_stgrn.data.sdata_utils import (
    normalize_log1p_cpm,
    DEFAULT_CAF_MARKERS,
    build_regular_grid_coords,
    caf_density_from_markers,
    gaussian_smooth_grid,
    normalize_minmax,
    aggregate_to_grid,
)

logger = logging.getLogger(__name__)

@dataclass
class CosMxConfig:
    h5ad_path: Optional[str] = None
    expr_csv: Optional[str] = None
    coords_csv: Optional[str] = None
    grid_um: float = 20.0
    cell_to_grid: bool = True
    agg: str = "mean"
    normalize: str = "log1p_cpm"  # or "none"
    smooth_sigma: float = 1.2
    smooth_radius: int = 3
    caf_markers: Tuple[str, ...] = DEFAULT_CAF_MARKERS

def _read_csv_matrix(expr_csv: str) -> Tuple[np.ndarray, Tuple[str, ...]]:
    import pandas as pd
    df = pd.read_csv(expr_csv, index_col=0)
    return df.values.astype(np.float32), tuple(map(str, df.columns))

def _read_csv_coords(coords_csv: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(coords_csv)
    if not {"x","y"}.issubset(df.columns):
        raise ValueError("coords_csv must have columns x,y (microns).")
    return df[["x","y"]].values.astype(np.float32)

def load_cosmx(cfg: CosMxConfig) -> Dict[str, object]:
    if cfg.h5ad_path:
        try:
            import anndata as ad
        except Exception as e:
            raise ImportError("Install anndata/scanpy to read .h5ad, or provide expr_csv+coords_csv.") from e
        adata = ad.read_h5ad(cfg.h5ad_path)
        X = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)
        X = X.astype(np.float32)
        gene_names = tuple(map(str, list(adata.var_names)))
        if "x" in adata.obs.columns and "y" in adata.obs.columns:
            coords = adata.obs[["x","y"]].values.astype(np.float32)
        elif "center_x" in adata.obs.columns and "center_y" in adata.obs.columns:
            coords = adata.obs[["center_x","center_y"]].values.astype(np.float32)
        else:
            raise ValueError("AnnData.obs must contain x/y or center_x/center_y columns.")
    else:
        if not (cfg.expr_csv and cfg.coords_csv):
            raise ValueError("Provide either h5ad_path or expr_csv+coords_csv.")
        X, gene_names = _read_csv_matrix(cfg.expr_csv)
        coords = _read_csv_coords(cfg.coords_csv)

    caf_raw = caf_density_from_markers(X, gene_names, caf_markers=cfg.caf_markers, log1p=True)
    grid_xy, grid_shape = build_regular_grid_coords(coords_um=coords, grid_um=float(cfg.grid_um))
    H, W = grid_shape
    flat = grid_xy[:,1]*W + grid_xy[:,0]

    grid_sum = np.zeros((H*W,), dtype=np.float64)
    grid_cnt = np.zeros((H*W,), dtype=np.float64)
    np.add.at(grid_sum, flat, caf_raw.astype(np.float64))
    np.add.at(grid_cnt, flat, 1.0)
    caf_grid = (grid_sum/(grid_cnt+1e-12)).astype(np.float32)

    caf_smooth = gaussian_smooth_grid(caf_grid, grid_shape=grid_shape, sigma=float(cfg.smooth_sigma), radius=int(cfg.smooth_radius))
    caf_norm = normalize_minmax(caf_smooth[flat])

    ecm_proxy = np.zeros_like(caf_norm, dtype=np.float32)  # explicit placeholder hook
    zE = np.stack([caf_norm, ecm_proxy], axis=1).astype(np.float32)
    return {"X": X, "coords": coords, "gene_names": gene_names, "zE": zE}
