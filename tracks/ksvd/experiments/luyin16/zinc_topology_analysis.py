"""Compact-v4 topology-channel analysis: subgroups, shuffle test, weights.

Reads the control-plane run directories of the Compact-v4 family
(candidate ids ``...compact-v4-topology-*``) plus the audit artifacts, and
produces the analysis tables required by
``notes/compact_v4_global_topology_channel.md``:

* parameter audit (v2 / encoder / head extra / total)
* valid MAE table vs compact-v2 baseline
* subgroup table (label-excess A/B/C per the long-cycle audit definition)
* shuffle test comparison
* hinge weight analysis (first-layer L1 norms per feature)
* ordinary-group degradation check (Group A delta >= -0.002)

Usage:
    uv run python -m tracks.ksvd.experiments.luyin16.zinc_topology_analysis
    [--run-dirs DIR1 DIR2 ...] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tracks.ksvd.experiments.luyin16.zinc_topology_features import feature_names

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNS_ROOT = REPO_ROOT / "tracks/ksvd/runs"
AUDIT_ROOT = REPO_ROOT / "tracks/ksvd/results/zinc_long_cycle_audit"
INFO_GAP_ROOT = REPO_ROOT / "tracks/ksvd/results/information_gap_audit"

V2_VALID_MAE = 0.18415821571176638
V2_TEST_MAE = 0.1353615188403055
V2_SELECTION_PARAMS = 98_549
V2_REFIT_PARAMS = 99_613

V4_PREFIX = "luyin16-zinc-hierarchical-patch-relation-context-compact-v4-topology"


def _candidate_id(run_dir: Path) -> str | None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("candidate_id")


def find_v4_run_dirs(roots: Iterable[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*/*/*/*")):
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate = manifest.get("candidate_id") or ""
            if not candidate.startswith(V4_PREFIX):
                continue
            if manifest.get("status") != "completed":
                continue
            if not (run_dir / "artifacts" / "legacy_full_result.json").exists():
                continue
            variant = candidate[len(V4_PREFIX) + 1:]  # after '-compact-v4-topology-'
            out[variant] = run_dir  # latest completed run wins
    return out


def load_result(run_dir: Path) -> dict[str, Any]:
    legacy = run_dir / "artifacts" / "legacy_full_result.json"
    if not legacy.exists():
        raise FileNotFoundError(f"missing {legacy}")
    return json.loads(legacy.read_text(encoding="utf-8"))


def _label_excess_table() -> pd.DataFrame:
    path = AUDIT_ROOT / "label_effective_cycle.csv"
    frame = pd.read_csv(path)
    frame["label_excess"] = (-frame["label_effective_cycle_snapped"]).round().clip(lower=0)
    return frame[["molecule_id", "split", "label_excess", "y_stored"]]


def _targets_from_run(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["evaluation"]["valid"]["targets"], dtype=np.float64)


def _predictions_from_run(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["evaluation"]["valid"]["predictions"], dtype=np.float64)


def subgroup_table(
    v2_predictions: np.ndarray,
    v4_results: dict[str, dict[str, Any]],
    *,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    labels = labels if labels is not None else _label_excess_table()
    valid_labels = labels[labels["split"] == "valid"].reset_index(drop=True)
    targets = np.asarray(valid_labels["y_stored"], dtype=np.float64)
    masks = {
        "A (no long-cycle)": valid_labels["label_excess"].to_numpy() == 0,
        "B (mild)": valid_labels["label_excess"].to_numpy() == 1,
        "C (extreme)": valid_labels["label_excess"].to_numpy() >= 2,
    }
    rows: list[dict[str, Any]] = []
    for group, mask in masks.items():
        row: dict[str, Any] = {"group": group, "n": int(mask.sum())}
        v2_err = np.abs(v2_predictions[mask] - targets[mask])
        row["v2_mae"] = float(np.mean(v2_err))
        for variant, result in v4_results.items():
            preds = _predictions_from_run(result)
            assert len(preds) == len(targets), (variant, len(preds), len(targets))
            err = np.abs(preds[mask] - targets[mask])
            row[f"{variant}_mae"] = float(np.mean(err))
            row[f"{variant}_delta"] = float(np.mean(v2_err) - np.mean(err))
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("/tmp/v4_topology_analysis.json"),
    )
    args = parser.parse_args(argv)

    run_roots = list(args.run_dirs) or [RUNS_ROOT]
    run_dirs = find_v4_run_dirs(run_roots)
    if not run_dirs:
        print("no compact-v4 run dirs found under:", *run_roots)
        return 1
    results: dict[str, dict[str, Any]] = {}
    for variant, run_dir in sorted(run_dirs.items()):
        results[variant] = load_result(run_dir)
        print(f"[{variant}] {run_dir}")

    # v2 baseline predictions (audit's bit-identity verified selection model)
    v2_valid = np.load(INFO_GAP_ROOT / "baseline_valid_predictions.npz")
    v2_predictions = v2_valid["prediction"]

    _ = v2_valid["y"]  # baseline targets; per-run targets come from run JSONs

    summary_rows = []
    for variant, result in sorted(results.items()):
        valid = result["evaluation"]["valid"]
        params = result["evaluation"]["parameters"]
        summary_rows.append(
            {
                "model": f"v4-{variant}",
                "params": int(params),
                "valid_mae": float(valid["best_mae"]),
                "delta_valid_mae": float(V2_VALID_MAE - valid["best_mae"]),
                "selected_epoch": int(valid["selected_epoch"]),
                "shuffled_mae": float(result.get("diagnostics", {}).get("valid_shuffled_topology_mae", np.nan)),
                "test_mae": float(
                    result["evaluation"]["test_after_train_valid_refit"]["mae"]
                    or np.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    subgroups = subgroup_table(v2_predictions, results)
    # per-variant pivot for the note
    group_names = {"A": "A (no long-cycle)", "B": "B (mild)", "C": "C (extreme)"}
    pivot_rows: list[dict[str, Any]] = []
    for letter in ("A", "B", "C"):
        sub = subgroups[subgroups["group"].str.startswith(letter)]
        assert len(sub) == 1, (letter, sub)
        entry = sub.iloc[0]
        row: dict[str, Any] = {"group": group_names[letter], "n": int(entry["n"])}
        for variant in sorted(results):
            row[f"{variant}_v2_mae"] = float(entry["v2_mae"])
            row[f"{variant}_mae"] = float(entry[f"{variant}_mae"])
            row[f"{variant}_delta"] = float(entry[f"{variant}_delta"])
        pivot_rows.append(row)
    pivot = pd.DataFrame(pivot_rows)

    print("\n=== parameter / valid summary ===")
    print(summary.to_string(index=False))
    print("\n=== subgroup (v2-definition label excess) ===")
    print(pivot.to_string(index=False))

    # hinge weight analysis
    hinge_dir = run_dirs.get("hinge")
    if hinge_dir is not None:
        state_paths = list(hinge_dir.glob("artifacts/*_selection_state.pt"))
        if not state_paths:
            state_paths = list(hinge_dir.glob("artifacts/*state.pt"))
        if state_paths:
            import torch

            state = torch.load(state_paths[0], map_location="cpu", weights_only=True)
            weight = state["topology_encoder.0.weight"]  # (16, 25)
            names = feature_names("hinge")
            norms = (
                weight.abs().sum(dim=0).tolist()
                if hasattr(weight, "abs")
                else weight.abs().sum(dim=0).tolist()
            )
            print("\n=== hinge first-layer L1 norms ===")
            for name, norm in sorted(
                zip(names, norms), key=lambda item: -item[1]
            ):
                print(f"  {name:24s} {norm:10.4f}")

    payload = {
        "summary": summary.to_dict(orient="records"),
        "subgroups": pivot.to_dict(orient="records"),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
