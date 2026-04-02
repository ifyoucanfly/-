from __future__ import annotations

from setuptools import setup, find_packages

# ------------------------------------------------------------------------------
# Reproducibility notes:
# - Spatial transcriptomics pipelines are extremely sensitive to dependency drift.
# - PyTorch Geometric wheels are platform/CUDA-specific; we list core PyG packages
#   but users may still need to follow the official install matrix for their CUDA.
# ------------------------------------------------------------------------------

install_requires = [
    # Core scientific stack
    "numpy>=1.23,<3.0",
    "scipy>=1.10,<2.0",
    "pandas>=1.5,<3.0",
    "pyyaml>=6.0,<7.0",
    "tqdm>=4.65,<5.0",
    "scikit-learn>=1.2,<2.0",
    "anndata>=0.9,<0.11",
    "h5py>=3.8,<4.0",

    # Deep learning / Neural ODE
    "torch>=2.1,<2.6",
    "torchdiffeq>=0.2.3,<0.3.0",

    # Optimal transport
    "POT>=0.9.0,<0.10.0",

    # ST ecosystem (I/O + graph + neighborhood)
    "scanpy>=1.9.6,<2.0.0",
    "squidpy>=1.4.1,<2.0.0",

    # CLI / UX
    "click>=8.1,<9.0",
    "rich>=13.0,<14.0",
]

# PyTorch Geometric (platform/CUDA dependent).
# We keep them in install_requires for a "one-command" demo on CPU-only setups
# that provide wheels, but users may need the official installation command:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
install_requires += [
    "torch-geometric>=2.4.0,<2.6.0",
    "torch-scatter>=2.1.2,<3.0.0",
    "torch-sparse>=0.6.18,<1.0.0",
    "torch-cluster>=1.6.3,<2.0.0",
    "torch-spline-conv>=1.2.2,<2.0.0",
]

setup(
    name="phycausal-stgrn",
    version="0.2.2",
    description="Physics-enhanced causal Neural ODE for spatial transcriptomics GRN dynamics with OT alignment.",
    author="Heath Sherlock",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=install_requires,
    entry_points={"console_scripts": ["phycausal-stgrn=phycausal_stgrn.cli:main"]},
)
