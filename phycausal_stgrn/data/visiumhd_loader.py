from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from phycausal_stgrn.data.sdata_utils import (
    DEFAULT_CAF_MARKERS,
    aggregate_to_grid,
    build_regular_grid_coords,
    caf_density_from_markers,
    gaussian_smooth_grid,
    normalize_minmax,
    normalize_log1p_cpm,
)

logger = logging.getLogger(__name__)

def _apply_env_map(raw_labels, env_map: Optional[Dict[str, object]]) -> Optional[np.ndarray]:
    """Map raw string/categorical labels to {1,0,-1} using env_map.

    env_map schema:
      env1_labels: [..]  -> mapped to 1
      env2_labels: [..]  -> mapped to 0
      ignore_labels: [..] (optional) -> mapped to -1 explicitly

    Unseen labels are mapped to -1 (ignore in IRM), but still participate in
    reconstruction / message passing because the rest of training uses all cells.
    """
    if env_map is None:
        return None
    env1 = set([str(x).strip().lower() for x in env_map.get("env1_labels", [])])
    env2 = set([str(x).strip().lower() for x in env_map.get("env2_labels", [])])
    ign = set([str(x).strip().lower() for x in env_map.get("ignore_labels", [])]) if isinstance(env_map.get("ignore_labels", None), (list, tuple)) else set()

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


@dataclass
class VisiumHDConfig:
    h5ad_path: Optional[str] = None
    spaceranger_dir: Optional[str] = None
    grid_um: float = 20.0

    # binning + normalization
    bin_to_grid: bool = True
    agg: str = "mean"
    normalize: str = "log1p_cpm"  # or "none"

    # CAF smoothing
    smooth_sigma: float = 1.2
    smooth_radius: int = 3
    caf_markers: Tuple[str, ...] = DEFAULT_CAF_MARKERS    # Optional hypothesis-driven environment mapping (labels -> {1,0,-1})
    env_map: Optional[Dict[str, object]] = None




def load_visiumhd(cfg: VisiumHDConfig) -> Dict[str, object]:
    """Load Visium/VisiumHD into numpy arrays for the pipeline."""

    if cfg.h5ad_path:
        try:
            import anndata as ad
        except Exception as e:
            raise ImportError("Install anndata/scanpy to read .h5ad.") from e

        adata = ad.read_h5ad(cfg.h5ad_path)
        X = adata.X
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        gene_names = tuple(map(str, list(adata.var_names)))

        if "spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        elif {"x", "y"}.issubset(adata.obs.columns):
            coords = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
        elif {"array_row", "array_col"}.issubset(adata.obs.columns):
            coords = adata.obs[["array_col", "array_row"]].to_numpy(dtype=np.float32)
        else:
            raise ValueError("AnnData must contain obsm['spatial'] or obs columns x/y or array_row/array_col.")

        if cfg.normalize == "log1p_cpm":
            X = normalize_log1p_cpm(X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32))
        elif cfg.normalize != "none":
            raise ValueError(f"Unknown normalize option: {cfg.normalize}")

        if cfg.bin_to_grid:
            X, coords = aggregate_to_grid(np.asarray(X, dtype=np.float32), coords, gene_names, grid_um=float(cfg.grid_um), agg=str(cfg.agg))
        else:
            X = np.asarray(X, dtype=np.float32)

    elif cfg.spaceranger_dir:
        try:
            import scanpy as sc
        except Exception as e:
            raise ImportError("Install scanpy to read SpaceRanger outputs.") from e

        adata = sc.read_visium(cfg.spaceranger_dir, load_images=False)
        X = adata.X
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        gene_names = tuple(map(str, list(adata.var_names)))

        if "spatial" not in adata.obsm:
            raise KeyError("Visium AnnData missing obsm['spatial']")
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

        if cfg.normalize == "log1p_cpm":
            X = normalize_log1p_cpm(X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32))
        elif cfg.normalize != "none":
            raise ValueError(f"Unknown normalize option: {cfg.normalize}")

        if cfg.bin_to_grid:
            X, coords = aggregate_to_grid(np.asarray(X, dtype=np.float32), coords, gene_names, grid_um=float(cfg.grid_um), agg=str(cfg.agg))
        else:
            X = np.asarray(X, dtype=np.float32)

    else:
        raise ValueError("Provide either h5ad_path or spaceranger_dir.")

    caf_raw = caf_density_from_markers(np.asarray(X, dtype=np.float32), gene_names, caf_markers=cfg.caf_markers, log1p=True)

    grid_xy, grid_shape = build_regular_grid_coords(coords_um=coords, grid_um=float(cfg.grid_um))
    H, W = grid_shape
    flat = grid_xy[:, 1] * W + grid_xy[:, 0]

    grid_sum = np.zeros((H * W,), dtype=np.float64)
    grid_cnt = np.zeros((H * W,), dtype=np.float64)
    np.add.at(grid_sum, flat, caf_raw.astype(np.float64))
    np.add.at(grid_cnt, flat, 1.0)
    caf_grid = (grid_sum / (grid_cnt + 1e-12)).astype(np.float32)

    caf_smooth = gaussian_smooth_grid(caf_grid, grid_shape=grid_shape, sigma=float(cfg.smooth_sigma), radius=int(cfg.smooth_radius))
    caf_norm = normalize_minmax(caf_smooth[flat])

    ecm_proxy = np.zeros_like(caf_norm, dtype=np.float32)
    zE = np.stack([caf_norm, ecm_proxy], axis=1).astype(np.float32)

    return {"X": np.asarray(X, dtype=np.float32), "coords": coords.astype(np.float32), "gene_names": gene_names, "zE": zE}
