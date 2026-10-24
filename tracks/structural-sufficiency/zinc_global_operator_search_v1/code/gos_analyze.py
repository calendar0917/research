"""Curve / strata analysis for zinc-global-operator-search-v1.

Development-phase tool: reads train/valid only (curve.csv + valid_best.npz).
Official test is never read here; see gos_test.py for the frozen terminal read.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

RUNS = _EXP / "runs"
ANALYSIS = _EXP / "analysis"
ZPS = _SSUF / "zinc_performance_search_v1" / "analysis"
DCC = _SSUF / "zinc_dual_channel_composer_v1" / "runs"
TAIL_Y = -3.0
ROW172 = 172


def load_curve(path: Path) -> dict[int, dict]:
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[int(r["epoch"])] = {
                "valid": float(r["valid_mae"]),
                "train": float(r["online_train_mae"]),
                "lr": float(r["lr"]),
                "seconds": float(r["seconds"]),
            }
    return out


def discover() -> list[dict]:
    runs = []
    for d in sorted(RUNS.iterdir()):
        cp = d / "curve.csv"
        tj = d / "train.json"
        if not cp.exists() or not tj.exists():
            continue
        info = C.read_json(tj)
        curve = load_curve(cp)
        epochs = sorted(curve)
        best_ep = min(epochs, key=lambda e: curve[e]["valid"])
        w20 = float(np.mean([curve[e]["valid"] for e in epochs[-20:]]))
        runs.append({
            "name": d.name,
            "route": d.name.split("_s")[0],
            "seed": int(info.get("seed", -1)),
            "params": int(info.get("parameter_count", -1)),
            "curve": curve,
            "best_ep": int(best_ep),
            "best_valid": float(curve[best_ep]["valid"]),
            "w20": w20,
            "final": float(curve[epochs[-1]]["valid"]),
            "train_at_best": float(curve[best_ep]["train"]),
            "gpu_hours": float(info.get("gpu_hours", 0.0)),
            "epochs": len(epochs),
            "nextag": d,
        })
    return runs


def reference_curves() -> dict[str, dict[int, dict]]:
    refs = {}
    for s in (0, 1, 2, 3):
        p = DCC / f"L_s{s}" / "curve.csv"
        if p.exists():
            refs[f"DCC-L-s{s}"] = load_curve(p)
    p = ZPS / "curves_valid.csv"
    if p.exists():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r["run"].startswith("B-R0") or r["run"].startswith("A-R0-"):
                    refs.setdefault(r["run"], {})[int(r["epoch"])] = {
                        "valid": float(r["valid_mae"]), "train": float(r["train_mae"]),
                        "lr": float(r["lr"]), "seconds": float(r["seconds"])}
    return refs


def at_epoch(curve: dict[int, dict], ep: int) -> float | None:
    if not curve:
        return None
    e = min(curve, key=lambda x: abs(x - ep))
    return float(curve[e]["valid"])


def main() -> int:
    runs = discover()
    refs = reference_curves()
    labels = C.read_json(ZPS / "per_molecule_valid.csv") if False else None
    # strata from the frozen per-molecule table of the previous round
    meta = {}
    with open(ZPS / "per_molecule_valid.csv") as fh:
        for r in csv.DictReader(fh):
            meta[int(r["row_id"])] = (float(r["y_raw"]), float(r["r2_unseen_atom_frac"]))
    y = np.asarray([meta[i][0] for i in range(len(meta))], dtype=np.float64)
    unseen = np.asarray([meta[i][1] > 0 for i in range(len(meta))])
    tail = y <= TAIL_Y
    strata = {
        "all": np.ones(len(y), bool),
        "ordinary_935": ~tail,
        "tail_65": tail,
        "seen_749": ~unseen,
        "unseen_251": unseen,
        "unseen_ordinary_225": unseen & ~tail,
        "unseen_tail_26": unseen & tail,
    }

    table = []
    curves_out = []
    permol = {"row_id": np.arange(len(y)), "y_raw": y}
    for r in runs:
        npz = r["nextag"] / "predictions" / "valid_best.npz"
        pred = None
        if npz.exists():
            d = np.load(npz)
            pred = np.zeros(len(y))
            pred[np.asarray(d["row"])] = np.asarray(d["pred"])
            permol[f"pred_{r['name']}"] = pred
            permol[f"ae_{r['name']}"] = np.abs(pred - y)
        row = {
            "run": r["name"], "route": r["route"], "seed": r["seed"], "params": r["params"],
            "epochs_run": r["epochs"], "best_valid": r["best_valid"], "best_ep": r["best_ep"],
            "w20": r["w20"], "final": r["final"], "train_at_best": r["train_at_best"],
            "gpu_hours": r["gpu_hours"],
        }
        if pred is not None:
            ae = np.abs(pred - y)
            keep172 = np.arange(len(y)) != ROW172
            top10 = np.argsort(-ae)[:10]
            keep10 = np.ones(len(y), bool)
            keep10[top10] = False
            row.update({
                "mae": float(ae.mean()),
                "mae_drop_row172": float(ae[keep172].mean()),
                "mae_drop_top10": float(ae[keep10].mean()),
                "d_row172": float(ae[ROW172]),
                "d_top10": float(ae[top10].mean()),
            })
            for k, m in strata.items():
                row[f"mae_{k}"] = float(ae[m].mean())
        table.append(row)
        for e in sorted(r["curve"]):
            curves_out.append({"run": r["name"], "epoch": e,
                               "valid_mae": r["curve"][e]["valid"],
                               "train_mae": r["curve"][e]["train"],
                               "lr": r["curve"][e]["lr"],
                               "seconds": r["curve"][e]["seconds"]})

    # matched-progress comparison at fixed epochs
    marks = [100, 200, 300, 400, 500, 600]
    matched = {}
    for name, cur in [(r["name"], r["curve"]) for r in runs]:
        matched[name] = {m: at_epoch(cur, m) for m in marks}
    for name, cur in refs.items():
        matched[name] = {m: at_epoch(cur, m) for m in marks}

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "main_table.json", {"runs": table, "strata_def": {
        "tail": "y_raw <= -3.0 (65)", "unseen": "r2_unseen_atom_frac > 0 (251)",
        "source": "zinc_performance_search_v1/analysis/per_molecule_valid.csv"}})
    C.write_json(ANALYSIS / "matched_epochs.json", {"marks": marks, "valid_mae": matched})

    with open(ANALYSIS / "curves_valid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "epoch", "valid_mae", "train_mae", "lr", "seconds"])
        w.writeheader()
        w.writerows(curves_out)
    np.savez(ANALYSIS / "per_molecule_valid.npz", **permol)

    # markdown
    lines = ["| run | params | ep-run | best-valid | best ep | W20 | final | train@best | GPU·h |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(
            f"| {r['run']} | {r['params']} | {r['epochs_run']} | {r['best_valid']:.6f} | {r['best_ep']} | "
            f"{r['w20']:.6f} | {r['final']:.6f} | {r['train_at_best']:.6f} | {r['gpu_hours']:.3f} |")
    lines.append("")
    lines.append("| run | " + " | ".join(f"ep{m}" for m in marks) + " |")
    lines.append("|---|" + "---:|" * len(marks))
    for name in sorted(matched):
        vals = " | ".join("—" if matched[name][m] is None else f"{matched[name][m]:.6f}" for m in marks)
        lines.append(f"| {name} | {vals} |")
    if any("mae_all" in r for r in table):
        lines.append("")
        lines.append("| run | all | ordinary | tail | unseen | unseen_ord | drop row172 | drop top10 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in table:
            if "mae_all" not in r:
                continue
            lines.append(
                f"| {r['run']} | {r['mae_all']:.6f} | {r['mae_ordinary_935']:.6f} | {r['mae_tail_65']:.6f} | "
                f"{r['mae_unseen_251']:.6f} | {r['mae_unseen_ordinary_225']:.6f} | "
                f"{r['mae_drop_row172']:.6f} | {r['mae_drop_top10']:.6f} |")
    (ANALYSIS / "main_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
