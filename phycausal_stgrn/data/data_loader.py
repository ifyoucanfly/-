from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch

from phycausal_stgrn.data.cosmx_loader import CosMxConfig, load_cosmx
from phycausal_stgrn.data.cosmx_tar_loader import CosMxTarConfig, load_cosmx_from_tar
from phycausal_stgrn.data.sdata_utils import intersect_genes_across_slices
from phycausal_stgrn.data.visiumhd_10x_loader import VisiumHD10xConfig, load_visiumhd_10x
from phycausal_stgrn.data.visiumhd_crc_loader import VisiumHDCRCConfig, load_visiumhd_crc
from phycausal_stgrn.data.visiumhd_loader import VisiumHDConfig, load_visiumhd

logger = logging.getLogger(__name__)


@dataclass
class SpatialSnapshot:
    X: torch.Tensor  # (N, G) expression
    coords: torch.Tensor  # (N, 2) in micrometers
    zI: torch.Tensor  # (N, dI)
    zN: torch.Tensor  # (N, dN)
    zE: torch.Tensor  # (N, dE)
    mechanomics: torch.Tensor  # (N, dM)
    edge_index: torch.Tensor  # (2, E)
    time: float
    gene_names: Optional[Tuple[str, ...]] = None
    gold_grn_path: Optional[str] = None
    mass: Optional[torch.Tensor] = None
    annotations: Optional[torch.Tensor] = None  # (N,1) domain labels (e.g., Tumor/Stroma)


@dataclass
class SpatialDataset:
    snapshots: Tuple[SpatialSnapshot, ...]
    name: str


def _pseudotime_from_snapshot_arrays(
    coords: np.ndarray,
    zE: Optional[np.ndarray] = None,
    annotations: Optional[np.ndarray] = None,
    *,
    mode: str = "coord_pca",
) -> np.ndarray:
    """Compute a 1D pseudo-time coordinate for splitting a single spatial snapshot.

    This is used to construct multiple *ordered* snapshots from a single tissue section.
    It is a practical workaround for real spatial datasets that lack true temporal slices.

    Supported modes:
      - "coord_pca" (default): first principal axis of spatial coordinates
      - "x" / "y": raw coordinate axis
      - "zE0": first environment proxy component
      - "annotation": uses annotations as numeric (if provided)
    """
    mode = str(mode or "coord_pca").lower()
    c = np.asarray(coords, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords must be (N,2), got {c.shape}")

    if mode == "x":
        return c[:, 0]
    if mode == "y":
        return c[:, 1]

    if mode == "ze0" and zE is not None:
        z = np.asarray(zE, dtype=np.float32)
        if z.ndim == 2 and z.shape[1] >= 1:
            return z[:, 0]

    if mode == "annotation" and annotations is not None:
        a = np.asarray(annotations)
        a = a.reshape(-1)
        # if categorical encoded as strings, hash to stable ints
        if a.dtype.kind in ("U", "S", "O"):
            uniq = {v: i for i, v in enumerate(sorted(set(map(str, a))))}
            return np.asarray([uniq[str(v)] for v in a], dtype=np.float32)
        return a.astype(np.float32)

    # Default: coord_pca
    cc = c - c.mean(axis=0, keepdims=True)
    # robust 2D PCA via SVD (cheap)
    _, _, vt = np.linalg.svd(cc, full_matrices=False)
    axis = vt[0].astype(np.float32)
    return (cc @ axis).astype(np.float32)


def _split_indices_by_pseudotime(pt: np.ndarray, num_timepoints: int) -> list[np.ndarray]:
    """Split indices into num_timepoints contiguous chunks along pseudo-time."""
    pt = np.asarray(pt).reshape(-1)
    n = int(pt.shape[0])
    k = int(max(2, min(int(num_timepoints), n)))
    order = np.argsort(pt, kind="mergesort")
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(order, k)]


def _standardize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _pad_or_truncate_features(x: np.ndarray, out_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n = int(x.shape[0])
    d = int(max(0, out_dim))
    if d == 0:
        return np.zeros((n, 0), dtype=np.float32)
    if x.shape[1] >= d:
        return x[:, :d].astype(np.float32, copy=False)
    pad = np.zeros((n, d - x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=1).astype(np.float32, copy=False)


def _annotation_interface_score(coords: np.ndarray, annotations: Optional[np.ndarray], k: int = 12) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    n = int(coords.shape[0])
    if annotations is None or n <= 1:
        return np.zeros((n,), dtype=np.float32)

    ann = np.asarray(annotations).reshape(-1)
    if ann.dtype.kind in ("U", "S", "O"):
        uniq = {v: i for i, v in enumerate(sorted(set(map(str, ann))))}
        ann_num = np.asarray([uniq[str(v)] for v in ann], dtype=np.int64)
    else:
        ann_num = ann.astype(np.int64, copy=False)

    kk = int(max(1, min(int(k), n - 1)))
    if kk <= 0:
        return np.zeros((n,), dtype=np.float32)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(coords.astype(np.float32, copy=False))
        try:
            _, idx = tree.query(coords.astype(np.float32, copy=False), k=kk + 1, workers=-1)
        except TypeError:
            _, idx = tree.query(coords.astype(np.float32, copy=False), k=kk + 1)
        nbr = np.asarray(idx[:, 1:], dtype=np.int64)
    except Exception:
        d = torch.cdist(torch.from_numpy(coords), torch.from_numpy(coords))
        nbr = torch.topk(d, k=kk + 1, largest=False).indices[:, 1:].cpu().numpy().astype(np.int64)

    mismatch = (ann_num[nbr] != ann_num[:, None]).astype(np.float32)
    return mismatch.mean(axis=1).astype(np.float32)


def _derive_snapshot_condition_features(
    *,
    coords: np.ndarray,
    zE: np.ndarray,
    annotations: Optional[np.ndarray],
    zI_dim: int,
    zN_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32)
    zE = np.asarray(zE, dtype=np.float32)
    n = int(coords.shape[0])

    coords_std = _standardize_feature(coords)
    radius = np.linalg.norm(coords - coords.mean(axis=0, keepdims=True), axis=1, keepdims=True).astype(np.float32)
    radius_std = _standardize_feature(radius)
    xy_prod = _standardize_feature((coords_std[:, :1] * coords_std[:, 1:2]).astype(np.float32))

    if annotations is not None:
        ann = np.asarray(annotations).reshape(-1)
        if ann.dtype.kind in ("U", "S", "O"):
            uniq = {v: i for i, v in enumerate(sorted(set(map(str, ann))))}
            ann_num = np.asarray([uniq[str(v)] for v in ann], dtype=np.float32)
        else:
            ann_num = ann.astype(np.float32, copy=False)
        ann_std = _standardize_feature(ann_num)
    else:
        ann_std = np.zeros((n, 1), dtype=np.float32)

    zE_std = _standardize_feature(zE) if zE.size else np.zeros((n, 0), dtype=np.float32)
    interface = _annotation_interface_score(coords, annotations, k=12).reshape(-1, 1).astype(np.float32)

    zI_base = np.concatenate([coords_std, radius_std, xy_prod], axis=1).astype(np.float32)
    # Keep interface score as the LAST column so trainer can apply stage-1 background weighting.
    zN_base = np.concatenate([zE_std, ann_std, coords_std, interface], axis=1).astype(np.float32)

    zI = _pad_or_truncate_features(zI_base, int(zI_dim))
    if int(zN_dim) > 0:
        if zN_base.shape[1] >= int(zN_dim):
            keep_head = max(0, int(zN_dim) - 1)
            if keep_head == 0:
                zN = interface.astype(np.float32)
            else:
                head = zN_base[:, :keep_head]
                zN = np.concatenate([head, interface], axis=1).astype(np.float32)
        else:
            zN = _pad_or_truncate_features(zN_base, int(zN_dim))
            zN[:, -1] = interface[:, 0]
    else:
        zN = np.zeros((n, 0), dtype=np.float32)
    return zI.astype(np.float32), zN.astype(np.float32)


def _make_pseudotime_snapshots(
    *,
    base: Dict[str, Any],
    X: np.ndarray,
    num_timepoints: int,
    delta_t: float,
    k_nn: int,
    zI_dim: int,
    zN_dim: int,
    gold_grn_path: Optional[str],
    mode: str,
    seed: int,
) -> Tuple[SpatialSnapshot, ...]:
    """Create ordered snapshots by splitting a single spatial dataset."""
    rng = np.random.default_rng(int(seed))
    coords = np.asarray(base["coords"], dtype=np.float32)
    zE = np.asarray(base["zE"], dtype=np.float32)
    annotations = base.get("annotations", None)
    mass = base.get("mass", None)

    pt = _pseudotime_from_snapshot_arrays(coords, zE=zE, annotations=annotations, mode=mode)
    chunks = _split_indices_by_pseudotime(pt, int(num_timepoints))
    snaps: list[SpatialSnapshot] = []
    for i, idx in enumerate(chunks):
        if idx.size == 0:
            continue
        # Optional shuffle within each slice to avoid any unintended ordering artifacts
        rng.shuffle(idx)
        s_bin = dict(base)
        s_bin["coords"] = coords[idx]
        s_bin["zE"] = zE[idx]
        if annotations is not None:
            s_bin["annotations"] = np.asarray(annotations)[idx]
        if mass is not None:
            s_bin["mass"] = np.asarray(mass)[idx]

        snaps.append(
            _snapshot_from_loaded(
                s=s_bin,
                X=np.asarray(X, dtype=np.float32)[idx],
                time=float(i) * float(delta_t),
                k_nn=int(k_nn),
                zI_dim=int(zI_dim),
                zN_dim=int(zN_dim),
                gold_grn_path=gold_grn_path,
                mass=s_bin.get("mass"),
            )
        )
    if len(snaps) < 2:
        raise ValueError("Pseudo-time splitting produced <2 snapshots; increase max_cells or reduce num_timepoints.")
    return tuple(snaps)


def knn_graph(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Build directed kNN graph (2, E) with a memory-safe backend.

    NOTE:
      The original implementation used `torch.cdist(coords, coords)` which allocates O(N^2) memory.
      For N~5e5 (Visium HD), that becomes ~TB and will crash with:
        DefaultCPUAllocator: not enough memory ... allocate 1e12 bytes

    This implementation prefers `scipy.spatial.cKDTree` (exact kNN, O(N log N) time, O(N) memory),
    and falls back to the original torch.cdist for small N if SciPy is unavailable.
    """
    with torch.no_grad():
        coords_cpu = coords.detach().cpu().float()
        n = int(coords_cpu.shape[0])
        if n == 0:
            return torch.empty((2, 0), dtype=torch.long, device=coords.device)

        # Prefer SciPy KDTree (exact kNN, memory-safe)
        try:
            import numpy as _np
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(_np.asarray(coords_cpu.numpy(), dtype=_np.float32))
            kk = int(k) + 1  # include self
            try:
                _, idx = tree.query(coords_cpu.numpy(), k=kk, workers=-1)
            except TypeError:
                # older SciPy without workers
                _, idx = tree.query(coords_cpu.numpy(), k=kk)

            knn = _np.asarray(idx[:, 1:], dtype=_np.int64)  # exclude self
            src = _np.repeat(_np.arange(n, dtype=_np.int64), int(k))
            dst = knn.reshape(-1)

            edge = _np.stack([src, dst], axis=0)
            edge_index = torch.from_numpy(edge).to(device=coords.device, dtype=torch.long)
            return edge_index

        except Exception:
            # Fallback (may OOM for huge N): only safe for small graphs
            d = torch.cdist(coords_cpu, coords_cpu)
            knn = torch.topk(d, k=int(k) + 1, largest=False).indices[:, 1:]
            src = torch.arange(n, device=coords_cpu.device).unsqueeze(1).repeat(1, int(k)).reshape(-1)
            dst = knn.reshape(-1)
            edge_index = torch.stack([src, dst], dim=0).to(device=coords.device, dtype=torch.long)
            return edge_index



def _simulate_crc_interface(
    num_spots: int,
    num_genes: int,
    delta_t: float,
    noise_std: float,
    k_nn: int,
    seed: int,
) -> SpatialDataset:
    """Simulated CRC-like tumor-stroma interface (toy)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1000, size=(num_spots, 2)).astype(np.float32)
    x = coords[:, 0]

    caf = 1 / (1 + np.exp(-(x - 520) / 30))
    ecm = np.clip(caf + 0.1 * rng.normal(size=num_spots), 0, 1)
    immune_base = 1 / (1 + np.exp((x - 480) / 30))
    exclusion_band = np.exp(-((x - 500) ** 2) / (2 * (35**2)))
    immune = np.clip(immune_base * (1 - 0.7 * exclusion_band), 0, 1)

    coords_t0 = torch.from_numpy(coords)
    edge_index = knn_graph(coords_t0, k=k_nn)

    zI = torch.zeros((num_spots, 1), dtype=torch.float32)
    zN = torch.from_numpy(np.stack([caf, ecm, immune], axis=1).astype(np.float32))
    zE = torch.from_numpy(np.stack([caf, ecm, immune], axis=1).astype(np.float32))
    mechanomics = torch.from_numpy(np.stack([ecm, caf], axis=1).astype(np.float32))

    W = rng.normal(size=(3, num_genes)).astype(np.float32)
    state = np.stack([immune, caf, ecm], axis=1).astype(np.float32) @ W
    X0 = state + noise_std * rng.normal(size=(num_spots, num_genes)).astype(np.float32)
    X0 = np.maximum(X0, 0)

    drift = (caf - immune)[:, None] * (0.05 * W[0:1, :])
    X1 = X0 + delta_t * (0.02 * (state) - 0.01 * X0 + drift) + noise_std * rng.normal(size=X0.shape).astype(np.float32)
    X1 = np.maximum(X1, 0)

    snap0 = SpatialSnapshot(
        X=torch.from_numpy(X0),
        coords=coords_t0,
        zI=zI,
        zN=zN,
        zE=zE,
        mechanomics=mechanomics,
        edge_index=edge_index,
        time=0.0,
        gene_names=tuple([f"G{i}" for i in range(num_genes)]),
        gold_grn_path="data/assets/sim_gold_grn.csv",
    )
    snap1 = SpatialSnapshot(
        X=torch.from_numpy(X1),
        coords=coords_t0.clone(),
        zI=zI,
        zN=zN,
        zE=zE,
        mechanomics=mechanomics,
        edge_index=edge_index,
        time=float(delta_t),
        gene_names=tuple([f"G{i}" for i in range(num_genes)]),
        gold_grn_path="data/assets/sim_gold_grn.csv",
    )
    return SpatialDataset(snapshots=(snap0, snap1), name="sim_crc")


def maybe_download_dataset(name: str, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    logger.info(
        "Dataset '%s': using local files/simulator. Implement download in maybe_download_dataset() if needed.",
        name,
    )


def _load_slices_from_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    slices = cfg.get("slices")
    if slices is None:
        return []
    if not isinstance(slices, list):
        raise ValueError("cfg['slices'] must be a list of slice specs.")
    out: List[Dict[str, Any]] = []
    for s in slices:
        if not isinstance(s, dict):
            raise ValueError("Each item in cfg['slices'] must be a dict.")
        out.append(s)
    return out


def _snapshot_from_loaded(
    s: Dict[str, Any],
    X: np.ndarray,
    time: float,
    k_nn: int,
    zI_dim: int,
    zN_dim: int,
    gold_grn_path: Optional[str],
    mass: Optional[np.ndarray] = None,
) -> SpatialSnapshot:
    coords_np = np.asarray(s["coords"], dtype=np.float32)
    zE_np = np.asarray(s["zE"], dtype=np.float32)
    ann_np = np.asarray(s["annotations"], dtype=np.float32) if ("annotations" in s and s["annotations"] is not None) else None

    coords = torch.from_numpy(coords_np)
    zE = torch.from_numpy(zE_np)
    mechanomics = zE.clone()
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))

    if ("zI" in s and s["zI"] is not None) or ("zN" in s and s["zN"] is not None):
        zI_np = _pad_or_truncate_features(np.asarray(s.get("zI", np.zeros((X_t.shape[0], 0), dtype=np.float32)), dtype=np.float32), int(zI_dim))
        zN_np = _pad_or_truncate_features(np.asarray(s.get("zN", np.zeros((X_t.shape[0], 0), dtype=np.float32)), dtype=np.float32), int(zN_dim))
        if int(zN_dim) > 0 and zN_np.shape[1] > 0 and ann_np is not None:
            zN_np[:, -1] = _annotation_interface_score(coords_np, ann_np, k=8)
    else:
        zI_np, zN_np = _derive_snapshot_condition_features(
            coords=coords_np,
            zE=zE_np,
            annotations=ann_np,
            zI_dim=int(zI_dim),
            zN_dim=int(zN_dim),
        )

    zI = torch.from_numpy(zI_np.astype(np.float32, copy=False))
    zN = torch.from_numpy(zN_np.astype(np.float32, copy=False))
    edge_index = knn_graph(coords, k=int(k_nn))

    mass_t: Optional[torch.Tensor] = None
    if mass is not None:
        mass_t = torch.from_numpy(np.asarray(mass, dtype=np.float32))

    return SpatialSnapshot(
        X=X_t,
        coords=coords,
        zI=zI,
        zN=zN,
        zE=zE,
        mechanomics=mechanomics,
        edge_index=edge_index,
        time=float(time),
        gene_names=tuple(map(str, s["gene_names"])),
        gold_grn_path=gold_grn_path,
        mass=mass_t,
        annotations=(torch.from_numpy(ann_np.astype(np.float32, copy=False)) if ann_np is not None else None),
    )


def load_dataset_from_config(cfg: Dict[str, Any]) -> SpatialDataset:
    """Unified dataset loader."""
    name = str(cfg["name"])

    # Multi-slice mode
    slices = _load_slices_from_cfg(cfg)
    if slices:
        loaded: List[Dict[str, Any]] = []
        for sl in slices:
            if ("spaceranger_dir" in sl) or ("h5ad_path" in sl and sl.get("h5ad_path")):
                loaded.append(
                    load_visiumhd(
                        VisiumHDConfig(
                            h5ad_path=sl.get("h5ad_path"),
                            spaceranger_dir=sl.get("spaceranger_dir"),
                            grid_um=float(sl.get("grid_um", cfg.get("grid_um", 20.0))),
                            smooth_sigma=float(sl.get("smooth_sigma", cfg.get("smooth_sigma", 1.2))),
                            smooth_radius=int(sl.get("smooth_radius", cfg.get("smooth_radius", 3))),
                            env_map=sl.get("env_map", cfg.get("env_map")),
                        )
                    )
                )
            elif ("expr_csv" in sl) or ("coords_csv" in sl):
                loaded.append(
                    load_cosmx(
                        CosMxConfig(
                            h5ad_path=sl.get("h5ad_path"),
                            expr_csv=sl.get("expr_csv"),
                            coords_csv=sl.get("coords_csv"),
                            grid_um=float(sl.get("grid_um", cfg.get("grid_um", 20.0))),
                            smooth_sigma=float(sl.get("smooth_sigma", cfg.get("smooth_sigma", 1.2))),
                            smooth_radius=int(sl.get("smooth_radius", cfg.get("smooth_radius", 3))),
                            env_map=sl.get("env_map", cfg.get("env_map")),
                        )
                    )
                )
            else:
                raise ValueError(f"Unrecognized slice spec keys: {list(sl.keys())}")

        genes_common, X_list = intersect_genes_across_slices(loaded)
        snaps: List[SpatialSnapshot] = []
        for sl, s, X in zip(slices, loaded, X_list):
            time = float(sl.get("time", 0.0))
            s = dict(s)
            s["gene_names"] = genes_common
            snaps.append(
                _snapshot_from_loaded(
                    s=s,
                    X=X,
                    time=time,
                    k_nn=int(cfg.get("k_nn", 12)),
                    zI_dim=int(cfg.get("zI_dim", 4)),
                    zN_dim=int(cfg.get("zN_dim", 4)),
                    gold_grn_path=cfg.get("gold_grn_path"),
                    mass=s.get("mass"),
                )
            )
        snaps_sorted = tuple(sorted(snaps, key=lambda ss: float(ss.time)))
        return SpatialDataset(snapshots=snaps_sorted, name=name)

    # Single dataset mode
    if name in ("cosmx_nsclc", "cosmx"):
        s = load_cosmx(
            CosMxConfig(
                h5ad_path=cfg.get("h5ad_path"),
                expr_csv=cfg.get("expr_csv"),
                coords_csv=cfg.get("coords_csv"),
                grid_um=float(cfg.get("grid_um", 20.0)),
                smooth_sigma=float(cfg.get("smooth_sigma", 1.2)),
                smooth_radius=int(cfg.get("smooth_radius", 3)),
                env_map=cfg.get("env_map"),
            )
        )
        snap0 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=0.0,
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        snap1 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=float(cfg.get("delta_t", 1.0)),
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        return SpatialDataset(snapshots=(snap0, snap1), name=name)

    if name in ("visium_hd", "visiumhd"):
        s = load_visiumhd(
            VisiumHDConfig(
                h5ad_path=cfg.get("h5ad_path"),
                spaceranger_dir=cfg.get("spaceranger_dir"),
                grid_um=float(cfg.get("grid_um", 20.0)),
                smooth_sigma=float(cfg.get("smooth_sigma", 1.2)),
                smooth_radius=int(cfg.get("smooth_radius", 3)),
                env_map=cfg.get("env_map"),
            )
        )
        snap0 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=0.0,
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        snap1 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=float(cfg.get("delta_t", 1.0)),
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        return SpatialDataset(snapshots=(snap0, snap1), name=name)

    if name in ("visium_hd_crc", "visium_crc"):
        # IMPORTANT:
        # Previously we only forwarded a small subset of config keys to VisiumHDCRCConfig,
        # causing user-provided settings (e.g., max_cells/roi_mode/num_genes/bin_to_grid)
        # to be silently ignored. This led to huge allocations on large CSV inputs.
        from dataclasses import fields

        allowed = {f.name for f in fields(VisiumHDCRCConfig)}
        kwargs = {k: v for k, v in cfg.items() if k in allowed}
        # Required key
        kwargs["annotation_csv"] = cfg["annotation_csv"]
        s = load_visiumhd_crc(VisiumHDCRCConfig(**kwargs))

        # Real spatial datasets typically lack true time slices. Instead of duplicating the same
        # snapshot twice (which creates a degenerate learning signal), we split the tissue into
        # ordered pseudo-time bins.
        num_tp = int(cfg.get("num_timepoints", 2) or 2)
        mode = str(cfg.get("pseudotime_mode", "coord_pca"))
        snaps = _make_pseudotime_snapshots(
            base=s,
            X=np.asarray(s["X"], dtype=np.float32),
            num_timepoints=num_tp,
            delta_t=float(cfg.get("delta_t", 1.0)),
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mode=mode,
            seed=int(cfg.get("seed", 0) or 0),
        )
        return SpatialDataset(snapshots=snaps, name=name)

    if name in ("visium_hd_10x", "visiumhd_10x", "visium_10x"):
        s = load_visiumhd_10x(VisiumHD10xConfig(**cfg))
        snap0 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=0.0,
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        snap1 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=float(cfg.get("delta_t", 1.0)),
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        return SpatialDataset(snapshots=(snap0, snap1), name=name)

    if name in ("cosmx_tar", "cosmx_nsclc_tar"):
        s = load_cosmx_from_tar(
            CosMxTarConfig(
                tar_gz_path=cfg["tar_gz_path"],
                extract_dir=str(cfg.get("extract_dir", "data/_extracted/cosmx")),
                expr_csv_name=str(cfg.get("expr_csv_name", "expr.csv")),
                coords_csv_name=str(cfg.get("coords_csv_name", "coords.csv")),
                h5ad_name=cfg.get("h5ad_name"),
                grid_um=float(cfg.get("grid_um", 20.0)),
                smooth_sigma=float(cfg.get("smooth_sigma", 1.2)),
                smooth_radius=int(cfg.get("smooth_radius", 3)),
                env_map=cfg.get("env_map"),
            )
        )
        snap0 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=0.0,
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        snap1 = _snapshot_from_loaded(
            s=s,
            X=np.asarray(s["X"], dtype=np.float32),
            time=float(cfg.get("delta_t", 1.0)),
            k_nn=int(cfg.get("k_nn", 12)),
            zI_dim=int(cfg.get("zI_dim", 4)),
            zN_dim=int(cfg.get("zN_dim", 4)),
            gold_grn_path=cfg.get("gold_grn_path"),
            mass=s.get("mass"),
        )
        return SpatialDataset(snapshots=(snap0, snap1), name=name)

    if name == "sim_crc":
        return _simulate_crc_interface(
            num_spots=int(cfg["num_spots"]),
            num_genes=int(cfg["num_genes"]),
            delta_t=float(cfg["delta_t"]),
            noise_std=float(cfg.get("noise_std", 0.05)),
            k_nn=int(cfg.get("k_nn", 12)),
            seed=int(cfg.get("seed", 7)),
        )

    raise ValueError(f"Unknown dataset name: {name}")
