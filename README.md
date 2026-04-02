# PhyCausal-STGRN v2.2 (NAR-ready)

Physics-enhanced causal Neural ODE for spatial transcriptomics GRN dynamics with:
- **MMS (Minimal Mechanism Shift)**: explicit `f_core + Gate ⊙ f_mod` separation (Gate sparsity + TV, staged training).
- **OT soft alignment (POT)**: cross-slice/time probabilistic coupling Π (mass-conserving, distribution-level supervision).
- **Mechanomics drift**: drift/advection term driven by environment `z^E` (upwind stabilization option).
- **OOD envelope**: counterfactual do(z) outside training distribution triggers conservative fallback (diffusion-only lower bound).
- **Committor + reactive flux (TPT)**: reviewer-grade P_cross definition; **sparse implementation** (no NxN dense).

## Project layout (reference-style)
- `phycausal_stgrn/`: installable package (modules, CLI, config loader)
- `configs/`: YAML configs
- `notebooks/`: CPU/GPU independent minimal demo + full paper storyline
- `runs/`: outputs (logs, checkpoints, figures)
- `data/assets/`: demo gold GRN labels for AUPR end-to-end tests

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart (CPU ok)
```bash
# Train
phycausal-stgrn train --config configs/default.yaml --out_dir runs

# Robustness + ablation
phycausal-stgrn robustness --config configs/default.yaml

# Paper figures (money figure + committor/flux + PR/AUPR/CI)
phycausal-stgrn figures --config configs/default.yaml --ckpt runs/model.pt --out_dir runs/paper_figures
```

## Reproducibility & diagnostics (defensive engineering)
- Global seeding via `seed_all(seed)` (python/numpy/torch).
- ODE integration safeguards: NaN/isfinite checks, NFE logging, optional drift upwind stabilization.
- OT safeguards: POT coupling computation isolated with explicit dtype/device handling.
- OOD envelope: always produces learned rollout + conservative diffusion-only fallback for out-of-support interventions.

## License
MIT (see LICENSE)

## S-level dataset loader (CosMx NSCLC / Visium HD)

We provide a **transparent** environment proxy field `zE` (no black-box latent):

- **CAF proxy**: mean expression of explicit CAF markers (COL1A1/COL1A2/COL3A1/DCN/LUM/FAP/PDGFRB/ACTA2/TAGLN), log1p
- **Spatial smoothing**: aggregate to a regular grid, apply separable Gaussian smoothing
- **Normalization**: min-max to [0,1] for interpretable intervention bounds & OOD checks

Code:
- `phycausal_stgrn/data/cosmx_loader.py`
- `phycausal_stgrn/data/visiumhd_loader.py`
- `phycausal_stgrn/data/sdata_utils.py`

Usage (edit paths in configs):
```bash
phycausal-stgrn train --config configs/cosmx_nsclc.yaml --out_dir runs_cosmx
phycausal-stgrn train --config configs/visium_hd.yaml --out_dir runs_visiumhd
```

### OOD definition for do(zE)
Since `zE[:,0]` is normalized CAF density in **[0,1]**, any counterfactual input outside [0,1]
is unambiguously OOD and must trigger the **diffusion-only conservative envelope**.

## Multi-slice/timepoint mode (OT supervision across slices)

For CosMx / VisiumHD you can provide a list of slices/timepoints:

```yaml
data:
  name: cosmx_nsclc
  slices:
    - {h5ad_path: "PATH/slice0.h5ad", time: 0.0}
    - {h5ad_path: "PATH/slice1.h5ad", time: 1.0}
    - {h5ad_path: "PATH/slice2.h5ad", time: 2.0}
```

Training will iterate over **consecutive pairs** (t0→t1, t1→t2, …) and apply:
- Neural ODE prediction loss
- **OT coupling Π**-based distribution alignment loss (POT) on each pair

## Aggregation (transparent, reviewer-friendly)

- CosMx: `cell_to_grid=True` aggregates cell-level points into regular grid bins (default).
- VisiumHD: `bin_to_grid=True` aggregates bins to coarser grid for stability and speed.

## OT caching & pairing strategies (engineering-only, does not alter OT logic)

In multi-slice runs, OT coupling computation can be a bottleneck. We provide on-disk caching of Π:
- `ot.cache_dir: runs/ot_cache`

Pairing modes:
- `ot.pairing: consecutive` (default): (t0→t1), (t1→t2), ...
- `ot.pairing: hub`: align all slices to a fixed hub slice (e.g., baseline)
  - `ot.hub_index: 0`

Gene alignment across slices:
- For multi-slice inputs, we automatically take **gene intersection** and reorder columns consistently.
  This prevents silent gene-order bugs and improves reproducibility.

## Quick-start with your local datasets (Visium HD CRC 8μm/16μm, CosMx Lung9)

Put your files under `data/`:

- `data/8um_squares_annotation.csv`
- `data/16um_squares_annotation.csv`
- `data/Lung9_Rep1+SMI+Flat+data.tar.gz`

Run:

```bash
phycausal-stgrn train --config configs/visium_hd_crc_8um.yaml --out_dir runs
phycausal-stgrn train --config configs/visium_hd_crc_16um.yaml --out_dir runs
phycausal-stgrn train --config configs/cosmx_nsclc_lung9_tar.yaml --out_dir runs
```

Notes:
- The Visium *annotation* CSV must include gene expression columns to train. If it only contains labels,
  provide a `.h5ad` / SpaceRanger expression dataset and use `configs/visium_hd.yaml` instead.
- For CosMx tar.gz, after extraction the loader expects `expr.csv` (cells x genes) and `coords.csv` with columns `x,y` (microns).
  If extracted filenames differ, set `expr_csv_name/coords_csv_name` in the config.

## Visium HD 10x format (new release)

For recent 10x Visium HD releases, use:

- `*_filtered_feature_bc_matrix.h5`
- `*_tissue_positions.parquet`
- `*_annotations.csv`

Edit `configs/visium_hd_10x_crc.yaml` to point to your local files and run:

```bash
phycausal-stgrn train --config configs/visium_hd_10x_crc.yaml --out_dir runs
```

The loader will:
- read expression with `scanpy.read_10x_h5` (sparse CSR internally)
- join spatial coords by barcode and expose `coords` used to build the PyG graph
- join pathology annotations by barcode (handles optional `-1` suffix)
- build mechanomics surrogate field zE as smoothed stroma/fibroblast density on a regular grid

### Auto-detect local Visium HD files under `data/`

If your Visium HD files are under `data/`, run:

```bash
phycausal-stgrn train --config configs/visium_hd_10x_crc_autodetect.yaml --out_dir runs
```

The loader auto-detects:
- `*filtered_feature_bc_matrix.h5` (expression)
- `*tissue_positions.parquet` or `.parquet.gz` (coords; `.gz` is decompressed once)
- `*annotations*.csv` / `*squares_annotation.csv` (pathology labels)

If no expression `.h5` is found, it runs a **metadata-only** sanity pipeline with a dummy gene
(to validate barcode alignment + zE field construction). For real training, add the `.h5` file.
