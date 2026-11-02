"""Build the frozen tables: main runs, seed means, predefined contrasts,
scaffold bootstrap intervals, cross-repeat aggregation, size bins, GPU ledger.

Everything is derived from ``runs/<run>/`` artefacts; nothing is re-trained or
re-selected here.  Negative deltas mean "candidate error lower than reference".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import dissect_common as C
import dissect_data as D
import dissect_source as S

TABLES = C.ANALYSIS / "tables"
BOOTSTRAP_SEED = 20261102
BOOTSTRAP_REPS = 2000

A_CONTRASTS = [
    ("A2", "A0", "NO_G - FULL (remove g, keep additive head)"),
    ("A4", "A3", "POOL_ONLY - NO_ADD (remove g, pooled readout)"),
    ("A3", "A0", "NO_ADD - FULL (remove additive head, keep g)"),
    ("A4", "A2", "POOL_ONLY - NO_G (remove additive head, no g)"),
]
B_CONTRASTS = [
    ("B1", "A0", "NO_JOINT - FULL"),
    ("B3", "B2", "NO_JOINT_CONTENT - NO_CONTENT"),
    ("B2", "A0", "NO_CONTENT - FULL"),
    ("B3", "B1", "NO_JOINT_CONTENT - NO_JOINT"),
]
OTHER_CONTRASTS = [
    ("B0", "A0", "TYPED_S2_OFF - FULL"),
    ("B4", "A0", "MARGINAL_ALL - FULL"),
    ("B5", "A0", "MARGINAL_BT2 - FULL"),
    ("B6", "A0", "NO_CLOSURE_CHANNELS - FULL"),
] + [(v, "A0", f"{v} - FULL") for v in ("C0", "C1", "C2", "C3", "C4")]


# ---------------------------------------------------------------------------
# artefact loading
# ---------------------------------------------------------------------------


def load_run(repeat: int, variant: str, seed: int) -> dict[str, Any] | None:
    run_dir = C.run_dir(repeat, variant, seed)
    record_path = run_dir / "train.json"
    if not record_path.exists():
        return None
    rec = C.read_json(record_path)
    if rec.get("status") != "completed":
        return None
    out: dict[str, Any] = {
        "record": rec,
        "dir": run_dir,
        "repeat": repeat,
        "variant": variant,
        "seed": seed,
        "valid": None,
        "test": None,
    }
    vp = run_dir / "valid_predictions.npz"
    if vp.exists():
        z = np.load(vp)
        out["valid"] = {
            "rows": np.asarray(z["rows"], dtype=np.int64),
            "raw": np.asarray(z["raw"], dtype=np.float64),
            "y": np.asarray(z["y_true"], dtype=np.float64),
        }
    te = run_dir / "test_eval.json"
    if te.exists():
        payload = C.read_json(te)
        tp = run_dir / "test_predictions.npz"
        z = np.load(tp)
        out["test"] = {
            "metrics": payload["metrics"],
            "status": payload["status"],
            "rows": np.asarray(z["rows"], dtype=np.int64),
            "raw": np.asarray(z["raw"], dtype=np.float64),
        }
    return out


def available(repeat: int) -> dict[str, bool]:
    return {
        v: all(load_run(repeat, v, s) is not None for s in C.SEEDS) for v in C.VARIANT_ORDER
    }


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def main_table(repeat: int = C.MAIN_REPEAT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in C.VARIANT_ORDER:
        for seed in C.SEEDS:
            run = load_run(repeat, variant, seed)
            if run is None:
                continue
            rec, v = run["record"], run["valid"]
            m = rec["metrics"]
            timing = rec.get("timing", {})
            test = run["test"]
            rows.append(
                {
                    "repeat": repeat,
                    "stage": C.VARIANT_DEFS[variant]["stage"],
                    "variant": variant,
                    "variant_name": C.VARIANT_NAMES[variant],
                    "seed": seed,
                    "params_nominal": rec["model"]["parameters_nominal"],
                    "params_trainable": rec["model"]["parameters"],
                    "params_expected": rec["model"]["parameters_expected"],
                    "optimizer_steps": rec["recipe"]["optimizer_steps"],
                    "epochs_run": rec["recipe"]["epochs_run"],
                    "best_epoch": rec["checkpoint"]["best_epoch"],
                    "train_rmse": m["train"]["rmse"],
                    "train_mae": m["train"]["mae"],
                    "train_r2": m["train"]["r2"],
                    "valid_rmse": m["valid"]["rmse"],
                    "valid_mae": m["valid"]["mae"],
                    "valid_r2": m["valid"]["r2"],
                    "valid_bias": m["valid"]["bias"],
                    "test_rmse": test["metrics"]["rmse"] if test else None,
                    "test_mae": test["metrics"]["mae"] if test else None,
                    "test_r2": test["metrics"]["r2"] if test else None,
                    "test_status": test["status"] if test else "NOT_SCORED",
                    "train_wall_seconds": timing.get("train_wall_seconds"),
                    "mean_epoch_seconds": timing.get("mean_epoch_seconds"),
                    "inference_seconds_per_valid_split": timing.get("inference_seconds_per_valid_split"),
                    "gpu_hours": rec.get("gpu", {}).get("gpu_hours_charged", 0.0),
                    "checkpoint": rec["checkpoint"]["path"],
                    "checkpoint_sha256": rec["checkpoint"]["sha256"],
                    "valid_predictions": str((run["dir"] / "valid_predictions.npz").relative_to(C.REPO)),
                    "valid_predictions_sha256": rec["predictions"]["valid_sha256"],
                    "test_predictions": str((run["dir"] / "test_predictions.npz").relative_to(C.REPO))
                    if test
                    else None,
                    "grad_clip_events": rec["anomalies"]["grad_clip_events"],
                    "nonfinite_events": rec["anomalies"]["nonfinite_events"],
                    "shared_init_bitwise_equal_to_full": rec["model"]["shared_init_bitwise_equal_to_full"],
                    "valid_n": len(v["raw"]) if v else None,
                }
            )
    return rows


def seed_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault((row["repeat"], row["variant"]), []).append(row)
    out = []
    for (repeat, variant), group in by_variant.items():
        group = sorted(group, key=lambda r: r["seed"])
        item: dict[str, Any] = {
            "repeat": repeat,
            "variant": variant,
            "variant_name": group[0]["variant_name"],
            "stage": group[0]["stage"],
            "seeds": ",".join(str(r["seed"]) for r in group),
            "n_seeds": len(group),
            "params_trainable": group[0]["params_trainable"],
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in group])),
            "best_epoch_min": int(min(r["best_epoch"] for r in group)),
            "best_epoch_max": int(max(r["best_epoch"] for r in group)),
            "gpu_hours_total": float(sum(r["gpu_hours"] or 0.0 for r in group)),
        }
        for key in ("valid_rmse", "valid_mae", "valid_r2", "train_rmse", "test_rmse", "test_mae", "test_r2"):
            vals = [r[key] for r in group if r.get(key) is not None]
            if not vals:
                item[key + "_mean"] = None
                continue
            arr = np.asarray(vals, dtype=np.float64)
            item[key + "_mean"] = float(arr.mean())
            item[key + "_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            item[key + "_min"] = float(arr.min())
            item[key + "_max"] = float(arr.max())
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------


def _seed_diffs(rows: Sequence[Mapping[str, Any]], cand: str, ref: str, key: str) -> list[float]:
    by = {(r["variant"], r["seed"]): r for r in rows}
    diffs = []
    for seed in C.SEEDS:
        a, b = by.get((cand, seed)), by.get((ref, seed))
        if a is None or b is None or a.get(key) is None or b.get(key) is None:
            continue
        diffs.append(float(a[key]) - float(b[key]))
    return diffs


def contrast_table(rows: Sequence[Mapping[str, Any]], *, split: str = "valid") -> list[dict[str, Any]]:
    rmse_key, mae_key = f"{split}_rmse", f"{split}_mae"
    out = []
    for cand, ref, label in A_CONTRASTS + B_CONTRASTS + OTHER_CONTRASTS:
        d_r = _seed_diffs(rows, cand, ref, rmse_key)
        d_a = _seed_diffs(rows, cand, ref, mae_key)
        if not d_r:
            out.append({"contrast": f"{cand}-{ref}", "label": label, "candidate": cand, "reference": ref, "available": False})
            continue
        out.append(
            {
                "contrast": f"{cand}-{ref}",
                "label": label,
                "candidate": cand,
                "reference": ref,
                "available": True,
                "n_seeds": len(d_r),
                "d_rmse_mean": float(np.mean(d_r)),
                "d_rmse_sd": float(np.std(d_r, ddof=1)) if len(d_r) > 1 else 0.0,
                "d_rmse_min": float(np.min(d_r)),
                "d_rmse_max": float(np.max(d_r)),
                "d_rmse_per_seed": json.dumps([round(x, 6) for x in d_r]),
                "d_mae_mean": float(np.mean(d_a)) if d_a else None,
                "d_mae_sd": float(np.std(d_a, ddof=1)) if len(d_a) > 1 else 0.0,
                "d_mae_per_seed": json.dumps([round(x, 6) for x in d_a]),
                "positive_seeds_rmse": int(sum(1 for x in d_r if x > 0)),
                "negative_seeds_rmse": int(sum(1 for x in d_r if x < 0)),
            }
        )
    # 2x2 interaction: (NO_G - FULL) - (POOL_ONLY - NO_ADD)
    by = {(r["variant"], r["seed"]): r for r in rows}
    inter = []
    for seed in C.SEEDS:
        cells = [by.get((v, seed)) for v in ("A0", "A2", "A3", "A4")]
        if any(c is None for c in cells):
            continue
        inter.append(
            (cells[1][rmse_key] - cells[0][rmse_key]) - (cells[3][rmse_key] - cells[2][rmse_key])
        )
    if inter:
        out.append(
            {
                "contrast": "A_INTERACTION",
                "label": "(NO_G-FULL) - (POOL_ONLY-NO_ADD): g x additive-head interaction",
                "candidate": "A2,A4",
                "reference": "A0,A3",
                "available": True,
                "n_seeds": len(inter),
                "d_rmse_mean": float(np.mean(inter)),
                "d_rmse_sd": float(np.std(inter, ddof=1)) if len(inter) > 1 else 0.0,
                "d_rmse_min": float(np.min(inter)),
                "d_rmse_max": float(np.max(inter)),
                "d_rmse_per_seed": json.dumps([round(x, 6) for x in inter]),
            }
        )
    inter_b = []
    for seed in C.SEEDS:
        cells = [by.get((v, seed)) for v in ("A0", "B1", "B2", "B3")]
        if any(c is None for c in cells):
            continue
        inter_b.append(
            (cells[1][rmse_key] - cells[0][rmse_key]) - (cells[3][rmse_key] - cells[2][rmse_key])
        )
    if inter_b:
        out.append(
            {
                "contrast": "B_INTERACTION",
                "label": "(NO_JOINT-FULL) - (NO_JOINT_CONTENT-NO_CONTENT)",
                "candidate": "B1,B3",
                "reference": "A0,B2",
                "available": True,
                "n_seeds": len(inter_b),
                "d_rmse_mean": float(np.mean(inter_b)),
                "d_rmse_sd": float(np.std(inter_b, ddof=1)) if len(inter_b) > 1 else 0.0,
                "d_rmse_min": float(np.min(inter_b)),
                "d_rmse_max": float(np.max(inter_b)),
                "d_rmse_per_seed": json.dumps([round(x, 6) for x in inter_b]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# scaffold bootstrap
# ---------------------------------------------------------------------------


def valid_scaffolds(repeat: int) -> list[str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    src = S.load_lipop(repeat, labels="train_valid")
    rows = src["rows"]["valid"]
    smiles = [src["smiles"][r] for r in rows]
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            keys.append(smi)
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        keys.append(scaffold if scaffold else Chem.MolToSmiles(mol))
    return keys


def bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeat: int = C.MAIN_REPEAT,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    keys = valid_scaffolds(repeat)
    groups: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    group_list = [np.asarray(v, dtype=np.int64) for v in groups.values()]
    rng = np.random.default_rng(seed)

    usable = [r for r in rows if r.get("valid") is not None and r["repeat"] == repeat]
    if not usable:
        return {"reps": reps, "seed": seed, "n_molecules": 0, "n_scaffold_groups": len(group_list), "contrasts": {}}
    y = np.asarray(usable[0]["valid"]["y"], dtype=np.float64)
    for r in usable:
        if not np.array_equal(r["valid"]["rows"], usable[0]["valid"]["rows"]):
            raise SystemExit("Valid row order differs across runs; cannot pair")
    sq: dict[tuple[str, int], np.ndarray] = {}
    ae: dict[tuple[str, int], np.ndarray] = {}
    for r in usable:
        key = (r["variant"], r["seed"])
        err = r["valid"]["raw"] - y
        sq[key] = (err * err).astype(np.float32)
        ae[key] = np.abs(err).astype(np.float32)
    point = {k: {"rmse": float(np.sqrt(sq[k].mean())), "mae": float(ae[k].mean())} for k in sq}

    # cluster bootstrap: resample scaffold groups with replacement (variable
    # molecule count per draw, multiplicity preserved), one draw used for all
    # variants and seeds.
    draw_rmse = {k: np.empty(reps, dtype=np.float64) for k in sq}
    draw_mae = {k: np.empty(reps, dtype=np.float64) for k in sq}
    for b in range(reps):
        picked = rng.integers(0, len(group_list), size=len(group_list))
        idx_b = np.concatenate([group_list[g] for g in picked])
        for k in sq:
            draw_rmse[k][b] = math.sqrt(float(sq[k][idx_b].mean()))
            draw_mae[k][b] = float(ae[k][idx_b].mean())

    out: dict[str, Any] = {
        "reps": reps,
        "seed": seed,
        "n_molecules": len(keys),
        "n_scaffold_groups": len(group_list),
        "contrasts": {},
    }
    for cand, ref, label in A_CONTRASTS + B_CONTRASTS + OTHER_CONTRASTS:
        seeds = [s for s in C.SEEDS if (cand, s) in sq and (ref, s) in sq]
        if not seeds:
            continue
        d_rmse_point = [point[(cand, s)]["rmse"] - point[(ref, s)]["rmse"] for s in seeds]
        d_mae_point = [point[(cand, s)]["mae"] - point[(ref, s)]["mae"] for s in seeds]
        d_r = np.zeros(reps, dtype=np.float64)
        d_a = np.zeros(reps, dtype=np.float64)
        for s in seeds:
            d_r += draw_rmse[(cand, s)] - draw_rmse[(ref, s)]
            d_a += draw_mae[(cand, s)] - draw_mae[(ref, s)]
        d_r /= len(seeds)
        d_a /= len(seeds)
        lo, hi = np.percentile(d_r, [2.5, 97.5])
        lo_a, hi_a = np.percentile(d_a, [2.5, 97.5])
        out["contrasts"][f"{cand}-{ref}"] = {
            "label": label,
            "seeds": seeds,
            "d_rmse_mean_point": float(np.mean(d_rmse_point)),
            "d_rmse_mean_ci95": [float(lo), float(hi)],
            "d_rmse_seed_sd": float(np.std(d_rmse_point, ddof=1)) if len(seeds) > 1 else 0.0,
            "d_mae_mean_point": float(np.mean(d_mae_point)),
            "d_mae_mean_ci95": [float(lo_a), float(hi_a)],
            "ci_crosses_zero": bool(lo <= 0.0 <= hi),
        }
    # 2x2 interaction intervals (same draws)
    for tag, (full, no_g, no_add, pool) in (
        ("A_INTERACTION", ("A0", "A2", "A3", "A4")),
        ("B_INTERACTION", ("A0", "B1", "B2", "B3")),
    ):
        seeds = [s for s in C.SEEDS if all((v, s) in sq for v in (full, no_g, no_add, pool))]
        if not seeds:
            continue
        point_inter = [
            (point[(no_g, s)]["rmse"] - point[(full, s)]["rmse"])
            - (point[(pool, s)]["rmse"] - point[(no_add, s)]["rmse"])
            for s in seeds
        ]
        d = np.zeros(reps, dtype=np.float64)
        for s in seeds:
            d += (draw_rmse[(no_g, s)] - draw_rmse[(full, s)]) - (draw_rmse[(pool, s)] - draw_rmse[(no_add, s)])
        d /= len(seeds)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["contrasts"][tag] = {
            "label": f"({no_g}-{full}) - ({pool}-{no_add})",
            "seeds": seeds,
            "d_rmse_mean_point": float(np.mean(point_inter)),
            "d_rmse_mean_ci95": [float(lo), float(hi)],
            "d_rmse_seed_sd": float(np.std(point_inter, ddof=1)) if len(seeds) > 1 else 0.0,
            "d_mae_mean_point": None,
            "d_mae_mean_ci95": [None, None],
            "ci_crosses_zero": bool(lo <= 0.0 <= hi),
        }
    return out


# ---------------------------------------------------------------------------
# size bins, cross-repeat, ledger
# ---------------------------------------------------------------------------


def size_bins(rows: Sequence[Mapping[str, Any]], repeat: int = C.MAIN_REPEAT) -> list[dict[str, Any]]:
    view = D.build_view(repeat)
    train_counts = np.diff(np.asarray(view.train_cache["atom_ptr"], dtype=np.float64))
    cuts = np.percentile(train_counts, [25, 50, 75])
    edges = [-math.inf, *cuts.tolist(), math.inf]
    valid_counts = np.diff(np.asarray(view.valid_cache["atom_ptr"], dtype=np.float64))
    bin_id = np.digitize(valid_counts, cuts)
    ref = load_run(repeat, "A0", 0)
    if ref is None or ref["valid"] is None:
        return []
    order = ref["valid"]["rows"]
    assert np.array_equal(order, np.asarray(view.rows["valid"], dtype=np.int64))
    out = []
    for b in range(len(edges) - 1):
        mask = bin_id == b
        item: dict[str, Any] = {
            "repeat": repeat,
            "bin": b,
            "atom_count_low": None if b == 0 else float(edges[b]),
            "atom_count_high": None if b == len(edges) - 2 else float(edges[b + 1]),
            "n_valid": int(mask.sum()),
            "train_edge_values": [float(x) for x in cuts],
        }
        for variant in C.VARIANT_ORDER:
            vals_r, vals_a, deltas = [], [], []
            for seed in C.SEEDS:
                run = load_run(repeat, variant, seed)
                if run is None or run["valid"] is None:
                    continue
                pred, y = run["valid"]["raw"], run["valid"]["y"]
                vals_r.append(float(np.sqrt(((pred[mask] - y[mask]) ** 2).mean())))
                vals_a.append(float(np.abs(pred[mask] - y[mask]).mean()))
                ref_run = load_run(repeat, "A0", seed)
                deltas.append(
                    float(np.sqrt(((pred[mask] - y[mask]) ** 2).mean()))
                    - float(np.sqrt(((ref_run["valid"]["raw"][mask] - y[mask]) ** 2).mean()))
                )
            if vals_r:
                item[f"{variant}_rmse"] = float(np.mean(vals_r))
                item[f"{variant}_mae"] = float(np.mean(vals_a))
                item[f"{variant}_d_rmse"] = float(np.mean(deltas))
        out.append(item)
    return out


def cross_repeat(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    variants = [v for v in ("A0", "B5") if any(r["variant"] == v for r in rows)]
    compact = C.ANALYSIS / "compact_selection.json"
    if compact.exists():
        chosen = C.read_json(compact).get("compact")
        if chosen and chosen not in variants and any(r["variant"] == chosen for r in rows):
            variants.append(chosen)
    out = []
    for variant in variants:
        per_repeat = {}
        for repeat in C.REPEATS:
            sel = [r for r in rows if r["variant"] == variant and r["repeat"] == repeat]
            if not sel:
                continue
            per_repeat[repeat] = {
                "n_seeds": len(sel),
                "valid_rmse_mean": float(np.mean([r["valid_rmse"] for r in sel])),
                "valid_rmse_sd": float(np.std([r["valid_rmse"] for r in sel], ddof=1)) if len(sel) > 1 else 0.0,
                "valid_mae_mean": float(np.mean([r["valid_mae"] for r in sel])),
                "test_rmse_mean": float(np.mean([r["test_rmse"] for r in sel])) if all(r["test_rmse"] is not None for r in sel) else None,
                "test_mae_mean": float(np.mean([r["test_mae"] for r in sel])) if all(r["test_mae"] is not None for r in sel) else None,
            }
        seed_across = []
        for seed in C.SEEDS:
            sel = [r for r in rows if r["variant"] == variant and r["seed"] == seed]
            if sel:
                seed_across.append({"seed": seed, "valid_rmse_mean": float(np.mean([r["valid_rmse"] for r in sel]))})
        all_sel = [r for r in rows if r["variant"] == variant]
        out.append(
            {
                "variant": variant,
                "variant_name": C.VARIANT_NAMES[variant],
                "per_repeat": per_repeat,
                "seed_across_repeats": seed_across,
                "grand_valid_rmse_mean": float(np.mean([r["valid_rmse"] for r in all_sel])) if all_sel else None,
                "grand_test_rmse_mean": float(np.mean([r["test_rmse"] for r in all_sel]))
                if all_sel and all(r["test_rmse"] is not None for r in all_sel)
                else None,
            }
        )
    return out


def gpu_ledger(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_run = []
    for r in rows:
        per_run.append(
            {
                "run": C.run_name(r["repeat"], r["variant"], r["seed"]),
                "gpu_hours": float(r["gpu_hours"] or 0.0),
                "train_wall_seconds": r["train_wall_seconds"],
                "mean_epoch_seconds": r["mean_epoch_seconds"],
            }
        )
    total = float(sum(x["gpu_hours"] for x in per_run))
    return {
        "n_runs": len(per_run),
        "gpu_hours_total": total,
        "soft_budget_gpu_hours": C.GPU_SOFT_CAP_HOURS,
        "hard_budget_gpu_hours": C.GPU_HARD_CAP_HOURS,
        "within_soft": total <= C.GPU_SOFT_CAP_HOURS,
        "within_hard": total <= C.GPU_HARD_CAP_HOURS,
        "per_run": per_run,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=C.MAIN_REPEAT)
    ap.add_argument("--bootstrap-reps", type=int, default=C.BOOTSTRAP_REPLICATES)
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args(argv)

    TABLES.mkdir(parents=True, exist_ok=True)
    rows = main_table(C.MAIN_REPEAT)
    for repeat in C.REPEATS[1:]:
        rows.extend(main_table(repeat))
    _write_csv(TABLES / "main_runs.csv", rows)
    C.log(f"[tables] main_runs.csv: {len(rows)} rows")
    _write_csv(TABLES / "variant_means.csv", seed_means(rows))
    C.write_json(TABLES / "contrasts_valid.json", contrast_table(rows, split="valid"))
    C.write_json(
        TABLES / "contrasts_test_exposed.json",
        contrast_table([r for r in rows if r["test_rmse"] is not None], split="test"),
    )
    _write_csv(TABLES / "size_bins.csv", size_bins(rows))
    C.write_json(TABLES / "cross_repeat.json", cross_repeat(rows))
    C.write_json(TABLES / "gpu_ledger.json", gpu_ledger(rows))
    if not args.no_bootstrap:
        main_rows = [
            r
            for r in (load_run(C.MAIN_REPEAT, v, s) for v in C.VARIANT_ORDER for s in C.SEEDS)
            if r is not None and r["valid"] is not None
        ]
        boot = {
            "schema": "lcrc-lipo-dissection-bootstrap-v1",
            "grouping": "Bemis-Murcko scaffold; empty scaffold -> canonical SMILES",
            "interpretation": (
                "evaluation-sample interval under the fixed training seeds; resamples Valid molecules by "
                "scaffold group with the same draw for every variant and seed. NOT a training-randomness CI."
            ),
            "valid": bootstrap(main_rows, reps=args.bootstrap_reps),
        }
        C.write_json(C.ANALYSIS / "bootstrap.json", boot)
        C.log("[tables] bootstrap.json written")
    status = {
        "utc": C.utcnow(),
        "repeat": C.MAIN_REPEAT,
        "runs_present": len(rows),
        "variants_complete": available(C.MAIN_REPEAT),
        "variants_scored_on_test": sorted({r["variant"] for r in rows if r["repeat"] == C.MAIN_REPEAT and r["test_rmse"] is not None}),
    }
    C.write_json(C.ANALYSIS / "tables" / "status.json", status)
    C.log(json.dumps({k: status[k] for k in ("runs_present",)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
