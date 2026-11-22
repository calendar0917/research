"""Frozen final evaluation on official Valid.

Two modes:

``--write-lock``
    List every finished full-Train run (checkpoint + prediction sha256), the
    locked configs/epochs/seeds and the evaluator code sha256, and write
    ``analysis/final_eval_lock.json``.  No Valid label is touched here.

``--score``
    Refuse to run without an existing final-eval lock, then open exactly the
    ``row_id``/``label`` members of the frozen Valid file once, verify the row
    set against the official Valid split, score every listed model, and write
    ``analysis/final_valid.json`` + update ``results/main_table.csv``.

Official Valid results never feed back into training or selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import psc_common as C
import psc_data as D
import psc_analysis as A

FINAL_LOCK = A.ANALYSIS / "final_eval_lock.json"
FINAL_OUT = A.ANALYSIS / "final_valid.json"


def _pred_path(rid: str) -> Path:
    p = A.RESULTS / rid / "predictions_final_eval.npz"
    if p.is_file():
        return p
    return A.CPU_OUT / rid / "predictions.npz"


def write_lock() -> dict[str, Any]:
    selection = C.read_json(A.ANALYSIS / "selection_lock.json")
    entries = []
    models = [selection["champion"], selection["ACRA_B_star"], "GINE_REF"]
    if selection.get("M_star"):
        models.append(selection["M_star"])
    for model in dict.fromkeys(models):
        for seed in (0, 1, 2):
            split = "FULL"
            rid = A.run_id(model, split, seed) if model in A.ARM_SPEC else f"CPU_{model}_FULL"
            base = A.RESULTS / rid if model in A.ARM_SPEC else A.CPU_OUT / rid
            man_path = base / "manifest.json"
            pred = _pred_path(rid)
            ckpt = base / "state_final.pt"
            if not man_path.is_file() or not pred.is_file():
                entries.append({"model": model, "seed": seed, "run_id": rid,
                                "status": "SKIPPED_BUDGET_OR_MISSING"})
                continue
            man = json.loads(man_path.read_text(encoding="utf-8"))
            entries.append({
                "model": model, "seed": seed, "run_id": rid, "split": "FULL",
                "status": "READY",
                "selection_lock_sha256": C.sha256_file(A.ANALYSIS / "selection_lock.json"),
                "checkpoint": str(ckpt.relative_to(C.REPO_ROOT)) if ckpt.is_file() else None,
                "checkpoint_sha256": C.sha256_file(ckpt) if ckpt.is_file() else None,
                "predictions": str(pred.relative_to(C.REPO_ROOT)),
                "predictions_sha256": C.sha256_file(pred),
                "epochs": int(man.get("final", {}).get("epoch", 0)),
                "E_m_expected": selection["E_m"].get(model),
                "recipe": man.get("recipe"),
                "fit_rows_sha256": man.get("split", {}).get("fit_rows_sha256"),
                "split_sha256": man.get("split", {}).get("sha256"),
                "n_fit": man.get("split", {}).get("n_fit"),
                "commit": man.get("provenance", {}).get("git_commit"),
                "selection": "fixed epoch; final checkpoint; no Valid-based selection",
                "wall_seconds": man.get("wall_seconds"),
                "gpu_hours": man.get("gpu_hours"),
            })
    for model in A.CPU_MODELS:
        rid = f"CPU_{model}_FULL"
        base = A.CPU_OUT / rid
        man_path = base / "manifest.json"
        pred = _pred_path(rid)
        if not man_path.is_file() or not pred.is_file():
            entries.append({"model": model, "seed": 0, "run_id": rid,
                            "status": "SKIPPED_BUDGET_OR_MISSING"})
            continue
        man = json.loads(man_path.read_text(encoding="utf-8"))
        entries.append({
            "model": model, "seed": 0, "run_id": rid, "split": "FULL", "status": "READY",
            "checkpoint": None, "checkpoint_sha256": None,
            "predictions": str(pred.relative_to(C.REPO_ROOT)),
            "predictions_sha256": C.sha256_file(pred),
            "hyperparameters": selection.get("cpu_lock", {}).get(model),
            "commit": man.get("provenance", {}).get("git_commit"),
            "selection": "hyper-parameters locked from D0",
        })
    payload = {
        "generated_utc": C.now_utc(), "git_commit": C.git_commit(),
        "selection_lock": str((A.ANALYSIS / "selection_lock.json").relative_to(C.REPO_ROOT)),
        "selection_lock_sha256": C.sha256_file(A.ANALYSIS / "selection_lock.json"),
        "evaluator_code_sha256": C.code_sha(["psc_final_eval.py", "psc_common.py"]),
        "valid_label_source": str(Path(C.VALID_LABEL_SOURCE).relative_to(C.REPO_ROOT)),
        "valid_label_source_sha256": C.sha256_file(C.VALID_LABEL_SOURCE),
        "valid_label_members": ["row_id", "label"],
        "models": entries,
        "skipped": [e for e in entries if e["status"] != "READY"],
        "note": "written after every planned final checkpoint exists and before any "
                "official Valid label is opened; nothing here can be changed afterwards",
    }
    C.write_json(FINAL_LOCK, payload)
    print(f"wrote {FINAL_LOCK} with {sum(1 for e in entries if e['status']=='READY')} ready, "
          f"{len(payload['skipped'])} skipped")
    return payload


def score() -> dict[str, Any]:
    if not FINAL_LOCK.is_file():
        raise SystemExit("final_eval_lock.json missing: refusing to open Valid labels")
    lock = C.read_json(FINAL_LOCK)
    if lock.get("evaluator_code_sha256") != C.code_sha(["psc_final_eval.py", "psc_common.py"]):
        raise SystemExit("evaluator code changed after the freeze; refusing to score")
    bundle = D.load_bundle()
    valid = C.load_valid_labels(bundle.raw)
    official = np.sort(bundle.official_valid)
    if not np.array_equal(np.sort(valid["row_id"]), official):
        raise ValueError("Valid label rows differ from the official Valid split")
    y = valid["label"]
    rows = valid["row_id"]
    out: dict[str, Any] = {
        "generated_utc": C.now_utc(), "git_commit": C.git_commit(),
        "label_access": {"source": valid["source"], "sha256": valid["source_sha256"],
                         "members_read": valid["members_read"],
                         "row_id_sha256": valid["row_id_sha256"],
                         "label_sha256": valid["label_sha256"],
                         "n": int(rows.size),
                         "accessed_after_final_eval_lock": True,
                         "any_feedback_to_training_or_selection": False},
        "final_eval_lock_sha256": C.sha256_file(FINAL_LOCK),
        "models": {},
    }
    preds: dict[str, list[np.ndarray]] = {}
    for entry in lock["models"]:
        if entry["status"] != "READY":
            out["models"][f"{entry['run_id']}"] = {"status": entry["status"]}
            continue
        path = C.REPO_ROOT / entry["predictions"]
        with np.load(path, allow_pickle=False) as z:
            rid = np.asarray(z["row_id"], np.int64)
            pred = np.asarray(z["pred"], np.float64)
        if not np.array_equal(rid, rows):
            if not np.array_equal(np.sort(rid), rows) or rid.size != rows.size:
                raise ValueError(f"{entry['run_id']}: Valid row set mismatch")
            order = np.argsort(rid)
            pred = pred[order]
        key = entry["model"]
        preds.setdefault(key, []).append(pred)
        out["models"].setdefault(key, {"seeds": []})
        out["models"][key]["seeds"].append({
            "seed": entry["seed"], "run_id": entry["run_id"],
            "rmse": C.rmse(pred, y), "mae": C.mae(pred, y),
            "bias": float(np.mean(pred - y)),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "predictions_sha256": entry["predictions_sha256"],
            "epochs": entry.get("epochs")})
    for model, lst in out["models"].items():
        seeds = lst.get("seeds")
        if seeds:
            rms = np.asarray([s["rmse"] for s in seeds])
            maes = np.asarray([s["mae"] for s in seeds])
            lst["rmse_mean"] = float(rms.mean())
            lst["rmse_sd"] = float(rms.std(ddof=1)) if rms.size > 1 else 0.0
            lst["mae_mean"] = float(maes.mean())
            lst["n_seeds"] = int(rms.size)
    # molecule bootstrap on Valid with one shared resampling index
    cfg = C.load_config()
    reps, seed0 = int(cfg["bootstrap"]["reps"]), int(cfg["bootstrap"]["seed"])
    rng = np.random.default_rng(seed0)
    n = rows.size
    draws = {m: [] for m in preds}
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        for m, lst in preds.items():
            rms = [float(np.sqrt(np.mean((p[idx] - y[idx]) ** 2))) for p in lst]
            draws[m].append(float(np.mean(rms)))
    ref = lock["models"][0]["model"]
    acra = next((e["model"] for e in lock["models"]
                 if e["model"].startswith("ACRA")), None)
    for m in preds:
        if acra is None or m == acra:
            continue
        d = np.asarray(draws[acra]) - np.asarray(draws[m])
        out["models"][m]["bootstrap_vs_ACRA_B"] = {
            "point": out["models"][acra]["rmse_mean"] - out["models"][m]["rmse_mean"],
            "ci95": C.percentile_ci(d),
            "definition": "ACRA_B* - candidate; >0 candidate better"}
    C.write_json(FINAL_OUT, out)
    # update the main table in place (FULL rows get their frozen Valid numbers)
    table = A.RESULTS / "main_table.csv"
    if table.is_file():
        import csv

        with table.open() as fh:
            table_rows = list(csv.DictReader(fh))
        fields = list(table_rows[0].keys()) if table_rows else []
        for name in ("rmse", "mae", "bias"):
            if name not in fields:
                fields.append(name)
        by_run = {s["run_id"]: s for m in out["models"] for s in
                  (out["models"][m].get("seeds") or [])}
        for row in table_rows:
            if row["run_id"] in by_run:
                s = by_run[row["run_id"]]
                row["status"] = "OK"
                row["rmse"] = s["rmse"]
                row["mae"] = s["mae"]
                row["bias"] = s["bias"]
                row["checkpoint"] = "final epoch (frozen Valid review)"
        C.write_csv(table, table_rows, columns=fields)
    print(json.dumps({m: {"rmse_mean": v.get("rmse_mean"), "mae_mean": v.get("mae_mean"),
                          "n_seeds": v.get("n_seeds")}
                      for m, v in out["models"].items()}, indent=1))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args(argv)
    if args.write_lock:
        write_lock()
    if args.score:
        score()
    if not args.write_lock and not args.score:
        raise SystemExit("choose --write-lock or --score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())