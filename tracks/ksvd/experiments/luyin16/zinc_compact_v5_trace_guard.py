"""Compare a compact-v5 run's validation trace against a canonical run.

Bit-identity gate for ``quantile_mode=none``: the seed-0 validation trace of
the v4-none guard must equal the canonical compact-v4-hinge run trace
epoch-by-epoch (same best MAE, same selected epoch, same epochs-run count).

Usage:
    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v5_trace_guard \
        <guard_run_id> <canonical_run_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _locate_run(run_id: str) -> Path:
    for manifest in (REPO_ROOT / "tracks/ksvd/runs").rglob(
        f"{run_id}/manifest.json"
    ):
        return manifest.parent
    raise FileNotFoundError(f"run {run_id} not found under runs/")


def _valid_trace(run_dir: Path) -> dict:
    artifact = run_dir / "artifacts/legacy_full_result.json"
    if not artifact.exists():
        raise FileNotFoundError(f"artifact missing for {run_dir}")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    valid = payload["evaluation"]["valid"]
    return {
        "trace": [(row["epoch"], row["mae"]) for row in valid["trace"]],
        "best_mae": valid["best_mae"],
        "selected_epoch": valid["selected_epoch"],
        "epochs_run": valid["epochs_run"],
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: zinc_compact_v5_trace_guard <guard_run_id> <canonical_run_id>")
        return 2
    guard_id, canonical_id = argv
    guard = _valid_trace(_locate_run(guard_id))
    canonical = _valid_trace(_locate_run(canonical_id))
    same_epochs = [a == b for a, b in zip(guard["trace"], canonical["trace"])]
    first_divergent = next(
        (i for i, same in enumerate(same_epochs) if not same), None
    )
    print(f"guard run:            {guard_id}")
    print(f"canonical run:        {canonical_id}")
    print(f"trace epochs:         {len(guard['trace'])} vs {len(canonical['trace'])}")
    print(f"best_mae:             {guard['best_mae']!r} vs {canonical['best_mae']!r}")
    print(f"selected_epoch:       {guard['selected_epoch']} vs {canonical['selected_epoch']}")
    print(f"epochs_run:           {guard['epochs_run']} vs {canonical['epochs_run']}")
    print(f"first divergent epoch: {first_divergent}")
    all_equal = (
        guard["trace"] == canonical["trace"]
        and guard["best_mae"] == canonical["best_mae"]
        and guard["selected_epoch"] == canonical["selected_epoch"]
        and guard["epochs_run"] == canonical["epochs_run"]
    )
    print("BIT-IDENTICAL" if all_equal else "DIVERGED")
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
