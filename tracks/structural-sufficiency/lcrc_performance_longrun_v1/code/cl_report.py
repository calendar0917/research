"""Build the Sprint-04 result tables, Test ledger, checkpoint manifest and GPU ledger.

Reads only committed/run-local artifacts under ``analysis/runs``.  Historical BASE
references are read from the frozen Sprint-01/02/03 tracks; they are never
re-trained here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cl_common as C

RUNS = C.ANALYSIS / "runs"

HIST_REF: dict[tuple[str, str], str] = {
    ("Lipophilicity", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/runs/Lipophilicity_s0",
    ("Lipophilicity", "lipo-repeat:123"): "tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/analysis/runs/lipo_r123",
    ("Lipophilicity", "lipo-repeat:124"): "tracks/structural-sufficiency/lcrc_taskaware_representation_pilot_v1/analysis/runs/lipo_r124",
    ("ESOL", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/runs/ESOL_s0",
    ("BACE", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/runs/BACE_s0",
    ("FreeSolv", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/runs/FreeSolv_s0",
    ("BBBP", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/runs/BBBP_s0",
    ("Tox21", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/runs/Tox21_s0",
    ("ogbg-molhiv", "published"): "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/runs/ogbg-molhiv_s0",
}

PARITY = {
    "Lipophilicity": "p0_base_lipo_parity",
    "ESOL": "p0_base_esol_parity",
    "BACE": "p0_base_bace_parity",
}

PRIMARY = {"regression": "rmse", "binary": "roc_auc", "classification": "roc_auc", "multitask": "macro_roc_auc"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_meta(run_dir: Path) -> dict[str, Any]:
    train = _read(run_dir / "train.json")
    test = _read(run_dir / "test_eval.json") if (run_dir / "test_eval.json").exists() else None
    key = train.get("run_key") or run_dir.name
    task = train["task"]
    primary = PRIMARY[task]
    valid = train["metrics"]["valid"][primary]
    test_value = train["metrics"]["test"][primary] if "test" in train["metrics"] else None
    stage = key.split("_")[0].upper()
    if key.startswith("p2b_"):
        stage = "P2B"
    elif key.startswith("p4b_"):
        stage = "P4B"
    elif key.startswith("p5b_"):
        stage = "P5B"
    return {
        "key": key,
        "stage": stage,
        "dataset": train["dataset"],
        "split": train["split"],
        "task": task,
        "arm": train["arm"],
        "seed": int(train["seed"]),
        "scale": train.get("scale", "base"),
        "protocol": train.get("protocol", "default"),
        "params": int(train["model"]["parameters"]),
        "best_epoch": int(train["checkpoint"]["best_epoch"]),
        "valid": float(valid),
        "test": None if test_value is None else float(test_value),
        "primary": primary,
        "wall_seconds": float(train.get("timing", {}).get("process_wall_seconds") or 0.0),
        "train_wall_seconds": float(train.get("timing", {}).get("train_wall_seconds") or 0.0),
        "commit": train.get("git_commit", ""),
        "test_read_utc": (test or {}).get("test_read_utc"),
        "test_rows_sha256": (test or {}).get("test_rows_sha256"),
        "n_test": (test or {}).get("n_test"),
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "checkpoint_sha256": C.sha256_file(run_dir / "checkpoint.pt"),
        "test_eval": str(run_dir / "test_eval.json"),
        "test_predictions": str(run_dir / "test_predictions.npz"),
        "run_dir": str(run_dir),
    }


def historical_ref(dataset: str, split: str) -> dict[str, Any] | None:
    rel = HIST_REF.get((dataset, split))
    if rel is None:
        return None
    base = C.REPO / rel
    train = _read(base / "train.json")
    test = _read(base / "test_eval.json")
    task = train["task"]
    primary = PRIMARY[task]
    payload = test.get("metrics", test)
    return {
        "key": f"hist:{base.name}",
        "task": task,
        "dataset": dataset,
        "split": split,
        "valid": float(payload["valid"][primary]),
        "test": float(payload["test"][primary]),
        "best_epoch": int(payload.get("best_epoch", train["checkpoint"]["best_epoch"])),
        "path": rel,
    }


def parity_ref(dataset: str) -> dict[str, Any] | None:
    key = PARITY.get(dataset)
    if key is None or not (RUNS / key).exists():
        return None
    meta = run_meta(RUNS / key)
    return {"key": key, "valid": meta["valid"], "test": meta["test"], "best_epoch": meta["best_epoch"], "path": meta["run_dir"]}


def improved(task: str, delta: float) -> bool:
    if task == "regression":
        return delta < 0
    return delta > 0


def add_row(stage: str, meta: dict[str, Any], ref: dict[str, Any], ref_label: str) -> dict[str, Any]:
    dv = meta["valid"] - ref["valid"]
    dt = meta["test"] - ref["test"]
    return {
        "stage": stage,
        "key": meta["key"],
        "dataset": meta["dataset"],
        "split": meta["split"],
        "task": meta["task"],
        "arm": meta["arm"],
        "seed": meta["seed"],
        "scale": meta["scale"],
        "protocol": meta["protocol"],
        "params": meta["params"],
        "best_epoch": meta["best_epoch"],
        "valid": meta["valid"],
        "test": meta["test"],
        "ref": ref_label,
        "ref_key": ref["key"],
        "ref_valid": ref["valid"],
        "ref_test": ref["test"],
        "d_valid": dv,
        "d_test": dt,
        "valid_better": improved(meta["task"], dv),
        "test_better": improved(meta["task"], dt),
        "both_better": improved(meta["task"], dv) and improved(meta["task"], dt),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    metas = {p.name: run_meta(p) for p in sorted(RUNS.iterdir()) if (p / "train.json").exists()}
    # stage tables
    p1: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P1":
            continue
        hist = historical_ref(meta["dataset"], meta["split"])
        par = parity_ref(meta["dataset"])
        if hist:
            p1.append(add_row("P1", meta, hist, "historical"))
        if par:
            p1.append(add_row("P1", meta, par, "same-round-parity"))
    write_csv(C.ANALYSIS / "p1_table.csv", p1)

    p2: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P2":
            continue
        par = parity_ref(meta["dataset"])
        hist = historical_ref(meta["dataset"], meta["split"])
        if par:
            p2.append(add_row("P2", meta, par, "same-round-parity"))
        elif hist:
            p2.append(add_row("P2", meta, hist, "historical"))
        # best parent at the same split
        parents = [m for m in metas.values() if m["stage"] == "P1" and m["dataset"] == meta["dataset"] and m["split"] == meta["split"]]
        if parents:
            best = min(parents, key=lambda m: m["test"]) if meta["task"] == "regression" else max(parents, key=lambda m: m["test"])
            p2.append(add_row("P2", meta, {"key": best["key"], "valid": best["valid"], "test": best["test"]}, "best-single"))
    write_csv(C.ANALYSIS / "p2_table.csv", p2)

    p3: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P3" or meta["arm"] == "base":
            continue
        refkey = f"p3_base_scalem_{'lipo' if meta['dataset'] == 'Lipophilicity' else 'esol'}"
        if refkey in metas:
            ref = metas[refkey]
            p3.append(add_row("P3", meta, ref, "base-scalem"))
    write_csv(C.ANALYSIS / "p3_table.csv", p3)

    p4: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P4" or meta["arm"] == "base":
            continue
        if meta["dataset"] == "ogbg-molhiv" and "p4_base_hiv" in metas:
            ref = metas["p4_base_hiv"]
            p4.append(add_row("P4", meta, ref, "matched-short-base"))
        else:
            hist = historical_ref(meta["dataset"], meta["split"])
            if hist:
                p4.append(add_row("P4", meta, hist, "historical"))
    write_csv(C.ANALYSIS / "p4_table.csv", p4)

    p5: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P5" or meta["arm"] == "base":
            continue
        basekey = key.replace("_rho_", "_base_")
        if basekey in metas:
            p5.append(add_row("P5", meta, metas[basekey], "matched-base-same-seed"))
    write_csv(C.ANALYSIS / "p5_table.csv", p5)

    p2b: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P2B":
            continue
        par = parity_ref(meta["dataset"])
        if par:
            p2b.append(add_row("P2B", meta, par, "same-round-parity"))
        parents = [m for m in metas.values() if m["stage"] == "P1" and m["dataset"] == meta["dataset"] and m["split"] == meta["split"]]
        if parents:
            best = min(parents, key=lambda m: m["test"]) if meta["task"] == "regression" else max(parents, key=lambda m: m["test"])
            p2b.append(add_row("P2B", meta, {"key": best["key"], "valid": best["valid"], "test": best["test"]}, "best-single"))
    write_csv(C.ANALYSIS / "p2b_table.csv", p2b)

    p4b: list[dict[str, Any]] = []
    for key, meta in metas.items():
        if meta["stage"] != "P4B":
            continue
        if meta["dataset"] == "ogbg-molhiv" and "p4_base_hiv" in metas:
            p4b.append(add_row("P4B", meta, metas["p4_base_hiv"], "matched-short-base"))
        else:
            hist = historical_ref(meta["dataset"], meta["split"])
            if hist:
                p4b.append(add_row("P4B", meta, hist, "historical"))
    write_csv(C.ANALYSIS / "p4b_table.csv", p4b)

    p5b: list[dict[str, Any]] = []
    p5b_base = {"p5b_combo_lipo_s1": "p5_base_lipo_s1", "p5b_combo_lipo_s2": "p5_base_lipo_s2"}
    for key, meta in metas.items():
        if meta["stage"] != "P5B":
            continue
        basekey = p5b_base.get(key)
        if basekey and basekey in metas:
            p5b.append(add_row("P5B", meta, metas[basekey], "matched-base-same-seed"))
    write_csv(C.ANALYSIS / "p5b_table.csv", p5b)

    # ledgers
    ledger = []
    for meta in sorted(metas.values(), key=lambda m: (m["key"])):
        if meta["test"] is None:
            continue
        ledger.append(
            {
                "key": meta["key"],
                "stage": meta["stage"],
                "dataset": meta["dataset"],
                "split": meta["split"],
                "arm": meta["arm"],
                "seed": meta["seed"],
                "scale": meta["scale"],
                "test_read_utc": meta["test_read_utc"],
                "n_test": meta["n_test"],
                "test_rows_sha256": meta["test_rows_sha256"],
                "test_metric": meta["primary"],
                "test_value": meta["test"],
                "test_used_for_development": True,
                "checkpoint_selected_by": "Validation only",
                "commit": meta["commit"],
            }
        )
    C.write_json(C.ANALYSIS / "test_access_ledger.json", {"schema": "lcrc-test-access-ledger/v1", "entries": ledger})

    manifest = []
    for meta in sorted(metas.values(), key=lambda m: m["key"]):
        manifest.append(
            {
                "key": meta["key"],
                "path": meta["checkpoint"],
                "sha256": meta["checkpoint_sha256"],
                "arm": meta["arm"],
                "dataset": meta["dataset"],
                "split": meta["split"],
                "seed": meta["seed"],
                "scale": meta["scale"],
                "best_epoch": meta["best_epoch"],
                "valid": meta["valid"],
                "test": meta["test"],
                "primary": meta["primary"],
            }
        )
    C.write_json(C.ANALYSIS / "checkpoint_manifest.json", {"schema": "lcrc-checkpoint-manifest/v1", "checkpoints": manifest})

    gpu_seconds = sum(m["wall_seconds"] for m in metas.values() if m["wall_seconds"] > 0)
    per_stage: dict[str, float] = {}
    for meta in metas.values():
        per_stage[meta["stage"]] = per_stage.get(meta["stage"], 0.0) + meta["wall_seconds"]
    C.write_json(
        C.ANALYSIS / "gpu_ledger.json",
        {
            "schema": "lcrc-gpu-ledger/v1",
            "note": "process_wall_seconds from each one-GPU A100 run; includes data loading and the one-shot Test read",
            "n_runs": len(metas),
            "total_gpu_hours": round(gpu_seconds / 3600.0, 4),
            "budget_gpu_hours": 15.0,
            "per_stage_gpu_hours": {k: round(v / 3600.0, 4) for k, v in sorted(per_stage.items())},
            "per_run": [{"key": m["key"], "seconds": round(m["wall_seconds"], 3), "gpu_hours": round(m["wall_seconds"] / 3600.0, 6)} for m in sorted(metas.values(), key=lambda x: x["key"])],
        },
    )

    # compact index for the report
    C.write_json(
        C.ANALYSIS / "run_index.json",
        {"schema": "lcrc-run-index/v1", "runs": [{k: v for k, v in m.items() if k not in ("run_dir",)} for m in sorted(metas.values(), key=lambda x: x["key"])]},
    )
    C.write_json(
        C.ANALYSIS / "stage_summary.json",
        {
            "P1": {"n": sum(1 for m in metas.values() if m["stage"] == "P1"), "both_better_vs_historical": sum(1 for r in p1 if r["ref"] == "historical" and r["both_better"]), "both_better_vs_parity": sum(1 for r in p1 if r["ref"] == "same-round-parity" and r["both_better"])},
            "P2": {"n": sum(1 for m in metas.values() if m["stage"] == "P2"), "both_better_vs_parity": sum(1 for r in p2 if r["ref"] == "same-round-parity" and r["both_better"])},
            "P3": {"n": sum(1 for m in metas.values() if m["stage"] == "P3")},
            "P4": {"n": sum(1 for m in metas.values() if m["stage"] == "P4"), "both_better": sum(1 for r in p4 if r["both_better"])},
            "P5": {"n": sum(1 for m in metas.values() if m["stage"] == "P5"), "both_better": sum(1 for r in p5 if r["both_better"])},
            "P2B": {"n": sum(1 for m in metas.values() if m["stage"] == "P2B"), "both_better_vs_parity": sum(1 for r in p2b if r["ref"] == "same-round-parity" and r["both_better"])},
            "P4B": {"n": sum(1 for m in metas.values() if m["stage"] == "P4B"), "both_better": sum(1 for r in p4b if r["both_better"])},
            "P5B": {"n": sum(1 for m in metas.values() if m["stage"] == "P5B"), "both_better": sum(1 for r in p5b if r["both_better"])},
        },
    )
    print(f"wrote tables for {len(metas)} runs; total GPU hours {gpu_seconds / 3600.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
