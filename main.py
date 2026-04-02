from __future__ import annotations

# PATH_IMMUNE_BOOTSTRAP
# NOTE: Not touching any algorithmic logic; only hardening import/path context.
import sys, os
# If launched from notebooks/, add repo root to PYTHONPATH; otherwise keep cwd on sys.path
if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
else:
    sys.path.insert(0, os.path.abspath(os.getcwd()))


from phycausal_stgrn.cli import main

if __name__ == "__main__":
    # backward compatible: `python main.py --config ...` still works via subcommand default
    import sys
    # If user calls legacy main.py with --config, map to `train`
    argv = sys.argv[1:]
    if argv and argv[0].startswith("--"):
        argv = ["train"] + argv
    main(argv)
