"""Copy pulled per-run artifacts into ``analysis/results/<arm_name>/``.

Reads the rr pull directory tree (``.rr/pulled/<exp>/result/.../runs/arm_<name>``)
and mirrors each arm directory into the analysis results area used by
``se_analysis``.  Pure file movement; never touches a checkpoint or the model.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import se_common as C
import se_analysis as A


def find_arm_dirs(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in root.rglob("arm_*"):
        if not d.is_dir():
            continue
        name = d.name[len("arm_"):]
        if (d / "metrics.json").is_file():
            # keep the shallowest (skip git.untracked mirrors)
            if name not in out or len(d.parts) < len(out[name].parts):
                out[name] = d
    return out


def ingest(pulled_root: Path | None = None, *, verbose: bool = True) -> dict[str, str]:
    root = pulled_root or (C.REPO / ".rr" / "pulled")
    A.RESULTS.mkdir(parents=True, exist_ok=True)
    found = find_arm_dirs(Path(root))
    moved: dict[str, str] = {}
    for name, src in sorted(found.items()):
        dst = A.RESULTS / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        moved[name] = C.track_rel(dst)
        if verbose:
            print(f"[ingest] {name} <- {src}")
    if verbose:
        print(f"[ingest] {len(moved)} arms into {C.track_rel(A.RESULTS)}")
    return moved


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    ingest(root)
