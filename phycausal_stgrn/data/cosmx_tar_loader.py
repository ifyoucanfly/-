from __future__ import annotations

import logging
import os
import tarfile
from dataclasses import dataclass
from typing import Dict, Optional

from phycausal_stgrn.data.cosmx_loader import CosMxConfig, load_cosmx

logger = logging.getLogger(__name__)


@dataclass
class CosMxTarConfig:
    tar_gz_path: str
    extract_dir: str = "data/_extracted/cosmx"
    expr_csv_name: str = "expr.csv"
    coords_csv_name: str = "coords.csv"
    h5ad_name: Optional[str] = None
    grid_um: float = 20.0
    smooth_sigma: float = 1.2
    smooth_radius: int = 3


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    for m in tar.getmembers():
        out = os.path.abspath(os.path.join(path, m.name))
        if not out.startswith(os.path.abspath(path) + os.sep):
            raise RuntimeError("Unsafe tar path detected.")
    tar.extractall(path)


def load_cosmx_from_tar(cfg: CosMxTarConfig) -> Dict[str, object]:
    os.makedirs(cfg.extract_dir, exist_ok=True)
    marker = os.path.join(cfg.extract_dir, ".extracted.ok")
    if not os.path.exists(marker):
        logger.info("Extracting CosMx tar.gz to %s", cfg.extract_dir)
        with tarfile.open(cfg.tar_gz_path, "r:gz") as t:
            _safe_extract(t, cfg.extract_dir)
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")

    h5ad_path = None
    if cfg.h5ad_name:
        cand = os.path.join(cfg.extract_dir, cfg.h5ad_name)
        if os.path.exists(cand):
            h5ad_path = cand

    expr_csv = os.path.join(cfg.extract_dir, cfg.expr_csv_name)
    coords_csv = os.path.join(cfg.extract_dir, cfg.coords_csv_name)

    if h5ad_path:
        return load_cosmx(CosMxConfig(
            h5ad_path=h5ad_path,
            grid_um=float(cfg.grid_um),
            smooth_sigma=float(cfg.smooth_sigma),
            smooth_radius=int(cfg.smooth_radius),
        ))

    if not (os.path.exists(expr_csv) and os.path.exists(coords_csv)):
        raise FileNotFoundError(
            "After extraction, cannot find expected expr/coords CSVs. "
            f"Expected: {expr_csv} and {coords_csv}. "
            "Rename extracted files or set expr_csv_name/coords_csv_name in config. "
            "expr.csv should be cells x genes (header=genes). coords.csv must contain x,y columns (microns)."
        )

    return load_cosmx(CosMxConfig(
        expr_csv=expr_csv,
        coords_csv=coords_csv,
        grid_um=float(cfg.grid_um),
        smooth_sigma=float(cfg.smooth_sigma),
        smooth_radius=int(cfg.smooth_radius),
    ))
