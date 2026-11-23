"""Post-run tables and scaffold / pair-component bootstrap.

Reads only local, pulled artefacts.  Whole-molecule bootstrap resamples frozen
Bemis-Murcko scaffold clusters (5,000 draws, rng 1703, shared draw within a
split); response bootstrap resamples official-Train pair-graph components and
keeps repeated draws.  Multi-seed metrics are computed per seed inside each
replicate and then averaged, never from an ensemble prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as C
import sr_data as D
import sr_ckernel as K

N_BOOT = 5000
SEED = 1703


def run_split(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("split_name") or manifest.get("split")
    if isinstance(value, Mapping):
        value = value.get("name")
    return str(value)


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_predictions(run_dir: Path, tag: str = "dev") -> dict[int, float] | None:
    npz = run_dir / f"predictions_best_{tag}.npz"
    if npz.is_file():
        with np.load(npz) as data:
            return {int(r): float(p) for r, p in
                    zip(data["row_id"].tolist(), data["pred"].tolist())}
    csv_path = run_dir / f"predictions_best_{tag}.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return {int(row["row_id"]): float(row["pred_raw"])
                    for row in csv.DictReader(handle)}
    return None


def discover(results: Path) -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(results.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith(("SMOKE", "HALF", ".")):
            continue
        manifest = read_manifest(run_dir)
        if not manifest:
            continue
        pred = read_predictions(run_dir)
        if pred is None:
            continue
        runs.append({"dir": run_dir, "manifest": manifest, "pred": pred})
    return runs


def scaffold_draws(split: str, rows: np.ndarray, n_boot: int,
                   seed: int) -> list[np.ndarray]:
    scaf = D.scaffolds(split)
    groups: dict[str, list[int]] = defaultdict(list)
    for pos, row in enumerate(rows.tolist()):
        groups[scaf[int(row)]].append(pos)
    keys = sorted(groups)
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(n_boot):
        take = rng.integers(0, len(keys), size=len(keys))
        pos = np.concatenate([np.asarray(groups[keys[j]], np.int64)
                              for j in take.tolist()])
        draws.append(pos)
    return draws


def whole_molecule(runs: list[dict[str, Any]], n_boot: int = N_BOOT,
                   seed: int = SEED) -> dict[str, Any]:
    labels = {s: D.load_bundle(s).labels for s in ("D0", "D1")}
    rows = {s: D.load_bundle(s).eval_rows for s in ("D0", "D1")}
    draws = {s: scaffold_draws(s, rows[s], n_boot, seed) for s in ("D0", "D1")}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        m = run["manifest"]
        arm = "R2_REF" if str(m["arm"]) == "SENTENCE_BASE" else str(m["arm"])
        groups[(arm, run_split(m))].append(run)
    table = []
    boot = {}
    for (arm, split), items in sorted(groups.items()):
        pts = rows[split]
        y = labels[split][pts]
        items = [r for r in items if all(int(x) in r["pred"] for x in pts.tolist())]
        if not items:
            continue
        for run in items:
            pred = np.asarray([run["pred"][int(r)] for r in pts.tolist()])
            m = run["manifest"]
            prov_path = run["dir"] / "rr_provenance.json"
            prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
            meta_rmse = float(m["dev_rmse"])
            recomputed = C.rmse(pred, y)
            table.append({
                "run_id": m["run_id"], "arm": arm, "split": split,
                "seed": int(m["seed"]), "family": m.get("family", ""),
                "dev_rmse": recomputed, "dev_mae": C.mae(pred, y),
                "manifest_dev_rmse": meta_rmse,
                "abs_diff_manifest": abs(recomputed - meta_rmse),
                "fit_rmse_at_best": float(m.get("train_rmse_at_best", np.nan)),
                "best_epoch": int(m.get("best_epoch", -1)),
                "optimizer_steps": int(m.get("optimizer_steps")
                                       or (m.get("steps_per_epoch", 0)
                                           * m.get("epochs_run", 0))),
                "wall_seconds": float(m.get("wall_seconds", np.nan)),
                "gpu_hours": float(m.get("wall_seconds", 0.0)) / 3600.0,
                "params": int((m.get("params") or {}).get("total", 0)),
                "pred_std": float(np.std(pred)),
                "n_dev": int(pts.size),
                "git_commit": prov.get("commit") or m.get("git_commit"),
                "checkpoint": m.get("checkpoint"),
            })
        seeds = sorted({int(r["manifest"]["seed"]) for r in items})
        preds = {int(r["manifest"]["seed"]): np.asarray(
            [r["pred"][int(x)] for x in pts.tolist()]) for r in items}
        per_model = np.zeros((len(seeds), n_boot))
        for si, s in enumerate(seeds):
            for b, pos in enumerate(draws[split]):
                per_model[si, b] = C.rmse(preds[s][pos], y[pos])
        values = per_model.mean(axis=0)
        lo, hi = np.percentile(values, [2.5, 97.5])
        boot[f"{arm}|{split}"] = {
            "seeds": seeds, "mean_rmse": float(values.mean()),
            "ci95": [float(lo), float(hi)],
            "per_seed_mean": {str(s): float(per_model[i].mean())
                              for i, s in enumerate(seeds)},
        }
    return {"table": table, "bootstrap": boot,
            "n_boot": int(n_boot), "rng_seed": int(seed),
            "cluster": "Bemis-Murcko scaffold (shared draw within split)"}


def contrast_values(runs: list[dict[str, Any]], split: str,
                    draws: Sequence[np.ndarray]) -> dict[tuple[str, int], np.ndarray]:
    labels = D.load_bundle(split).labels
    rows = D.load_bundle(split).eval_rows
    y = labels[rows]
    values: dict[tuple[str, int], np.ndarray] = {}
    for run in runs:
        m = run["manifest"]
        if run_split(m) != split:
            continue
        if not all(int(r) in run["pred"] for r in rows.tolist()):
            continue
        pred = np.asarray([run["pred"][int(r)] for r in rows.tolist()])
        values[(str(m["arm"]), int(m["seed"]))] = np.asarray(
            [C.rmse(pred[pos], y[pos]) for pos in draws])
    return values


def contrasts(runs: list[dict[str, Any]], split: str,
              draws: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    values = contrast_values(runs, split, draws)
    point = {k: float(np.mean(v)) for k, v in values.items()}
    defs = [
        ("C1", "E_RAW", "E_MOB"),
        ("C2", "F_RAW", "F_MOB"),
        ("C3", "E_MOB", "F_MOB"),
        ("C4", "G_EGO2_5", "G_FULL5"),
        ("SUP_F_RAW_vs_E_RAW", "E_RAW", "F_RAW"),
        ("SUP_F_RAW_vs_R2", "R2_REF", "F_RAW"),
        ("SUP_F_MOB_vs_R2", "R2_REF", "F_MOB"),
        ("SUP_E_MOB_vs_R2", "R2_REF", "E_MOB"),
        ("SUP_EGO2_vs_R2", "R2_REF", "G_EGO2_5"),
        ("SUP_FULL_vs_R2", "R2_REF", "G_FULL5"),
    ]
    out = []
    for name, ref, cand in defs:
        seeds = sorted({s for (a, s) in values if a == ref}
                       & {s for (a, s) in values if a == cand})
        if not seeds:
            out.append({"split": split, "contrast": name, "reference": ref,
                        "candidate": cand, "status": "MISSING"})
            continue
        per_seed_gain = np.stack([values[(ref, s)] - values[(cand, s)]
                                  for s in seeds])
        gain = per_seed_gain.mean(axis=0)
        lo, hi = np.percentile(gain, [2.5, 97.5])
        out.append({
            "split": split, "contrast": name, "reference": ref,
            "candidate": cand, "metric": "RMSE",
            "gain": float(gain.mean()), "ci95_low": float(lo),
            "ci95_high": float(hi), "n_seeds": len(seeds),
            "seeds": "|".join(str(s) for s in seeds),
            "reference_error": float(np.mean([point[(ref, s)] for s in seeds])),
            "candidate_error": float(np.mean([point[(cand, s)] for s in seeds])),
            "status": "OK",
        })
    # interaction C2 - C1, only on the seeds shared by all four arms
    arms4 = ("E_RAW", "E_MOB", "F_RAW", "F_MOB")
    seeds4 = sorted(set.intersection(*[{s for (a, s) in values if a == arm}
                                       for arm in arms4]))
    if seeds4:
        gain = np.stack([(values[("F_RAW", s)] - values[("F_MOB", s)])
                         - (values[("E_RAW", s)] - values[("E_MOB", s)])
                         for s in seeds4]).mean(axis=0)
        lo, hi = np.percentile(gain, [2.5, 97.5])
        out.append({"split": split, "contrast": "C2_minus_C1",
                    "reference": "E_RAW-E_MOB", "candidate": "F_RAW-F_MOB",
                    "metric": "RMSE interaction",
                    "gain": float(gain.mean()), "ci95_low": float(lo),
                    "ci95_high": float(hi), "n_seeds": len(seeds4),
                    "seeds": "|".join(str(s) for s in seeds4), "status": "OK"})
    return out


# ---------------------------------------------------------------------------
# route C: neural Delta f on the frozen real-edit pairs
# ---------------------------------------------------------------------------
def _block_values(err: np.ndarray, comp: np.ndarray, block_ids: Sequence[int]
                  ) -> np.ndarray:
    return np.asarray([float(np.mean(err[comp == c])) for c in block_ids])


def component_bootstrap(errs: Mapping[str, np.ndarray], comp: np.ndarray,
                        n_boot: int = N_BOOT, seed: int = SEED,
                        ref: str = "K_TRANSFORM") -> dict[str, Any]:
    all_components = np.unique(comp).tolist()
    local = {int(c): j for j, c in enumerate(all_components)}
    block_ids = sorted({int(c) for c in comp.tolist()})
    block_pos = {local[c]: j for j, c in enumerate(block_ids)}
    blocks = {m: _block_values(errs[m], comp, block_ids) for m in errs}
    rng = np.random.default_rng(int(seed))
    n_global = len(all_components)
    vals = {m: np.zeros(n_boot) for m in errs}
    draws = []
    for b in range(n_boot):
        take = rng.integers(0, n_global, size=n_global)
        pos = np.asarray([block_pos[int(local[int(all_components[j])])]
                          for j in take.tolist()
                          if int(local[int(all_components[j])]) in block_pos],
                         np.int64)
        draws.append(pos)
        for m in errs:
            vals[m][b] = blocks[m][pos].mean() if pos.size else np.nan
    summary = {}
    for m in errs:
        lo, hi = np.nanpercentile(vals[m], [2.5, 97.5])
        summary[m] = {"component_equal_mae": {
            "mean": float(np.nanmean(vals[m])),
            "ci95": [float(lo), float(hi)]}}
        if ref in errs and ref != m:
            g = vals[m] - vals[ref]
            glo, ghi = np.nanpercentile(g, [2.5, 97.5])
            summary[m][f"gain_vs_{ref}"] = {
                "mean": float(np.nanmean(g)), "ci95": [float(glo), float(ghi)]}
    return {"n_boot": int(n_boot), "seed": int(seed),
            "n_components": len(block_ids),
            "fixed_component_equal_mae": {
                m: float(np.mean(_block_values(errs[m], comp, block_ids)))
                for m in errs},
            "summary": summary}


def route_c(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"deltas": [], "bootstrap": {}}
    for split in ("D0", "D1"):
        pred_path = C.ANALYSIS_DIR / f"response_dev_predictions_{split}.npz"
        if not pred_path.is_file():
            continue
        with np.load(pred_path) as data:
            idx = data["pair_id"]
            row_a = data["row_a"]
            row_b = data["row_b"]
            delta = data["delta"]
            comp = data["component"]
            kernel_pred = {k[5:]: data[k] for k in data.files if k.startswith("pred_")}
        labels = D.load_bundle(split).labels
        errs: dict[str, np.ndarray] = {}
        for name, p in kernel_pred.items():
            err = np.abs(p - delta)
            errs[name] = err
            out["deltas"].append({
                "split": split, "arm": name, "seed": "kernel",
                "pair_metric": "component_equal_mae",
                "component_equal_mae": K.component_equal_mae(err, comp),
                "pair_equal_mae": float(np.mean(err)),
                "rmse": float(np.sqrt(np.mean((p - delta) ** 2))),
                "n_pairs": int(idx.size), "status": "OK",
            })
        if split == "D0":
            for run in runs:
                m = run["manifest"]
                if run_split(m) != split:
                    continue
                pred = run["pred"]
                if not all(int(r) in pred for r in row_a.tolist()) or \
                        not all(int(r) in pred for r in row_b.tolist()):
                    continue
                d = np.asarray([pred[int(b)] - pred[int(a)]
                                for a, b in zip(row_a.tolist(), row_b.tolist())])
                err = np.abs(d - delta)
                arm_name = "R2_REF" if str(m["arm"]) == "SENTENCE_BASE" else str(m["arm"])
                name = f"{arm_name}_S{m['seed']}"
                errs[name] = err
                out["deltas"].append({
                    "split": split, "arm": name, "seed": int(m["seed"]),
                    "pair_metric": "component_equal_mae",
                    "component_equal_mae": K.component_equal_mae(err, comp),
                    "pair_equal_mae": float(np.mean(err)),
                    "rmse": float(np.sqrt(np.mean((d - delta) ** 2))),
                    "n_pairs": int(idx.size), "status": "OK",
                })
        if errs:
            out["bootstrap"][split] = component_bootstrap(errs, comp)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    C.write_csv(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(C.RESULTS_DIR))
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR))
    parser.add_argument("--boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = discover(Path(args.results))
    whole = whole_molecule(runs, n_boot=args.boot)
    write_csv(out / "main_table.csv", whole["table"])
    rows_d0 = D.load_bundle("D0").eval_rows
    rows_d1 = D.load_bundle("D1").eval_rows
    con = []
    con += contrasts(runs, "D0", scaffold_draws("D0", rows_d0, args.boot, SEED))
    con += contrasts(runs, "D1", scaffold_draws("D1", rows_d1, args.boot, SEED))
    # C5, computed from the pair-component bootstrap (not the scaffold one)
    resp_boot = out / "response_bootstrap.json"
    if resp_boot.is_file():
        rb = json.loads(resp_boot.read_text(encoding="utf-8"))
        for split, block in rb.items():
            x = block.get("Dev-Dev", {}).get("summary", {})
            if "K_TRANSFORM" in x and "K_CONDITIONAL_R2" in x:
                g = x["K_CONDITIONAL_R2"].get("gain_vs_K_TRANSFORM")
                if g:
                    con.append({
                        "split": split, "contrast": "C5", "metric": "component-equal MAE",
                        "reference": "K_TRANSFORM",
                        "candidate": "K_CONDITIONAL_R2",
                        "gain": -float(g["mean"]),
                        "ci95_low": -float(g["ci95"][1]),
                        "ci95_high": -float(g["ci95"][0]),
                        "n_seeds": 0, "seeds": "kernel",
                        "reference_error": float(x["K_TRANSFORM"]["component_equal_mae"]["mean"]),
                        "candidate_error": float(x["K_CONDITIONAL_R2"]["component_equal_mae"]["mean"]),
                        "status": "OK", "source": "pair_component",
                    })
    write_csv(out / "contrasts.csv", con)
    C.write_json(out / "bootstrap_whole.json", whole["bootstrap"])
    support_path = out / "response_support.json"
    if support_path.is_file():
        support = json.loads(support_path.read_text(encoding="utf-8"))
        rows = []
        for split, scopes in support.items():
            for scope, values in scopes.items():
                row = {"split": split, "scope": scope}
                row.update(values)
                rows.append(row)
        C.write_csv(out / "response_support.csv", rows)
    rc = route_c(runs)
    write_csv(out / f"response_neural_deltas.csv", rc["deltas"])
    C.write_json(out / "response_neural_bootstrap.json", rc["bootstrap"])
    print(json.dumps({"n_runs": len(runs), "n_contrasts": len(con),
                      "main_rows": len(whole["table"]),
                      "neural_delta_rows": len(rc["deltas"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())