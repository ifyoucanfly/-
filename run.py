from __future__ import annotations

# PATH_IMMUNE_BOOTSTRAP
# NOTE: Not touching any algorithmic logic; only hardening import/path context.
import sys, os
# If launched from notebooks/, add repo root to PYTHONPATH; otherwise keep cwd on sys.path
if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
else:
    sys.path.insert(0, os.path.abspath(os.getcwd()))


import os
import sys
from pathlib import Path

# ------------------------------------------------------------------------------
# Phase-1 hardening:
# - Anchor CWD to project root to avoid path context drift (configs/, data/, runs/)
# - Ensure PROJECT_ROOT is on PYTHONPATH for module discovery
# - Warn on Python >= 3.12 (PyG / torch extensions frequently break)
# ------------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR  # run.py lives at repo root by design

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.version_info >= (3, 12):
    print(
        "⚠️ [WARNING] Python >= 3.12 detected. "
        "Strongly recommend Python 3.10/3.11 for stable PyG & Neural ODE installs.",
        file=sys.stderr,
    )

from phycausal_stgrn.cli import main  # noqa: E402


def _ensure_subprocess_env() -> None:
    """Export PYTHONPATH for any subprocess invoked by this process."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    os.environ.update(env)


if __name__ == "__main__":
    _ensure_subprocess_env()
    main()
