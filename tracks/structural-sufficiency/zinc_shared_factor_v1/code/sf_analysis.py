"""Collect, contrast and gate the runs of ``zinc_shared_factor_v1``.

Reads only artifacts written by :mod:`sf_runner` plus the frozen grouping
reference of ``zinc_restricted_adaptation_v1`` (radius-2 seen/unseen flags) and
the ``zinc_branching_fragment_v1`` pattern caches (for atom/bond type coverage).
No training, no official-test data, no threshold is invented here: the gates
below are the ones pre-registered in ``DESIGN.md`` §7.

Outputs (``analysis/``):

* ``summary.json``     per arm/seed metrics, contrasts, groups, diagnostics
* ``main_table.md``    the small table the report quotes
* ``preds_valid.csv``  per-molecule valid predictions of every finished arm
* ``gates.json``       stage A/B/C verdicts with the numbers behind them
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
EXP = _HERE.parent
_TRACKS = EXP.parent
_BF = _TRACKS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bf_data as BFD  # noqa: E402

RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
REF = _TRACKS / "zinc_restricted_adaptation_v1" / "analysis" / "reference_preds_valid.csv"
ARMS = {"D": "SF-Dense", "T": "SF-SharedState", "N": "SF-IndependentCopies"}
SEEDS = (0, 1)
TAIL_EPS = 1e-9
WINDOW = 20
TAIL = 100
MARGIN = 0.003          # stage A margin, DESIGN.md §7
ABS_GATE = 0.075        # stage A absolute valid MAE, DESIGN.md §7
BOOT = 10_000
BOOT_SEED = 20261022


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_curve(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, list] = {}
    for rec in rows:
        for k, v in rec.items():
            out.setdefault(k, []).append(v)
    res = {}
    for k, v in out.items():
        try:
            res[k] = np.asarray([float(x) for x in v], dtype=np.float64)
        except ValueError:
            res[k] = np.asarray(v)
    return res


def load_preds(arm: str, seed: int, tag: str = "best"):
    npz = RUNS / f"SF_{arm}_s{seed}" / "predictions" / f"valid_{tag}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as z:
        return {"row": np.asarray(z["row"], dtype=np.int64),
                "pred": np.asarray(z["pred"], dtype=np.float64),
                "y": np.asarray(z["y"], dtype=np.float64)}


# ---------------------------------------------------------------------------
# per-run metrics
# ---------------------------------------------------------------------------


def curve_metrics(curve: dict[str, np.ndarray]) -> dict:
    valid = curve["valid_mae"]
    ep = curve["epoch"].astype(int)
    n = len(valid)
    best_i = int(np.argmin(valid))
    med, med_end = None, None
    if n >= WINDOW:
        meds = [float(np.median(valid[i:i + WINDOW])) for i in range(0, n - WINDOW + 1)]
        j = int(np.argmin(meds))
        med, med_end = meds[j], int(ep[j + WINDOW - 1])
    return {
        "epochs": n,
        "best_valid": float(valid[best_i]),
        "best_epoch": int(ep[best_i]),
        "final_valid": float(valid[-1]),
        "best20_window_median": med,
        "best20_window_end_epoch": med_end,
        "tail100_median": (float(np.median(valid[-TAIL:])) if n >= TAIL else None),
        "seconds_per_epoch": float(np.mean(curve["seconds"])),
        "wall_seconds_curve": float(np.sum(curve["seconds"])),
        "w_s_norm_last": float(curve["w_s_norm"][-1]),
        "online_train_mae_last": float(curve["online_train_mae"][-1]),
    }


def run_metrics(arm: str, seed: int) -> dict | None:
    run = RUNS / f"SF_{arm}_s{seed}"
    curve_path = run / "curve.csv"
    if not curve_path.exists():
        return None
    train = read_json(run / "train.json")
    scored = train.get("scored", {})
    out = {
        "arm": arm,
        "arm_name": ARMS[arm],
        "seed": int(seed),
        "params": train.get("parameter_count"),
        "init": train.get("init", {}),
        "gpu_hours": train.get("gpu_hours"),
        "peak_gpu_allocated_mib": train.get("peak_gpu_allocated_mib"),
        "curve": curve_metrics(read_curve(curve_path)),
        "scored": scored,
        "frozen": (ANALYSIS / f"SF_{arm}_s{seed}" / "freeze.json").exists(),
        "status": read_json(ANALYSIS / f"SF_{arm}_s{seed}" / "status.json").get("status"),
    }
    # lift the curve metrics the gates compare to the top level
    out.update({k: out["curve"][k] for k in
                ("best20_window_median", "best20_window_end_epoch", "tail100_median",
                 "seconds_per_epoch")})
    if scored:
        # best/final numbers are the reloaded ones (same checkpoints the preds come from)
        out["best_valid"] = float(scored["best"]["valid_mae"])
        out["best_epoch"] = int(scored["best"]["epoch"])
        out["final_valid"] = float(scored["final"]["valid_mae"])
        out["eval_train_at_best"] = float(scored["best"]["train_mae_eval_mode"])
        out["best_sha256"] = scored["best"]["sha256"]
    return out


# ---------------------------------------------------------------------------
# per-molecule contrasts
# ---------------------------------------------------------------------------


def bootstrap_delta(ae_a: np.ndarray, ae_b: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired bootstrap of ``MAE(a) - MAE(b)`` over molecules (not seeds)."""
    d = ae_a - ae_b
    idx = rng.integers(0, d.size, size=(BOOT, d.size))
    means = d[idx].mean(axis=1)
    return {
        "delta_mae": float(d.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n_molecules": int(d.size),
        "frac_improved": float(np.mean(d < 0.0)),
        "median_delta_ae": float(np.median(d)),
    }


def contrast(pred_a, pred_b, rng) -> dict:
    if pred_a is None or pred_b is None:
        return {"available": False}
    if not np.array_equal(pred_a["row"], pred_b["row"]):
        raise RuntimeError("per-molecule rows are not aligned")
    ae_a = np.abs(pred_a["pred"] - pred_a["y"])
    ae_b = np.abs(pred_b["pred"] - pred_b["y"])
    out = {"available": True, **bootstrap_delta(ae_a, ae_b, rng),
           "mae_a": float(ae_a.mean()), "mae_b": float(ae_b.mean())}
    # concentration of the difference: is it one molecule or many?
    d = ae_a - ae_b
    total = float(d.sum())
    abs_total = float(np.abs(d).sum())
    out["concentration"] = {
        "sum_delta_ae": total,
        "sum_abs_delta_ae": abs_total,
        "max_single_contribution": float(d.max()),
        "max_single_share_of_abs": (float(np.abs(d).max() / abs_total) if abs_total else None),
        "top1pct_share_of_abs": (float(np.sort(np.abs(d))[-max(1, d.size // 100):].sum()
                                       / abs_total) if abs_total else None),
        "share_from_best_10_molecules": (float(np.sort(np.abs(d))[-10:].sum() / abs_total)
                                         if abs_total else None),
        "frac_molecules_better": float(np.mean(d < 0)),
        "frac_molecules_worse": float(np.mean(d > 0)),
        "frac_molecules_tied": float(np.mean(d == 0)),
    }
    return out


# ---------------------------------------------------------------------------
# groups: radius-2 seen/unseen x ordinary/tail, and type coverage
# ---------------------------------------------------------------------------


def load_groups() -> dict[str, np.ndarray]:
    if not REF.exists():
        return {}
    with open(REF, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "row": np.asarray([int(r["row"]) for r in rows], dtype=np.int64),
        "r2_unseen_atom_frac": np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows]),
        "signature_unseen": np.asarray([int(r["signature_unseen"]) for r in rows]),
        "ordinary": np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows]),
    }


def type_coverage() -> dict:
    """Are the atom/bond *types* of valid molecules present in train at all?"""
    tr = BFD.load_cache("train")
    va = BFD.load_cache("valid")
    # atom category 0 is a real type (and also the padding value), bond 0 is padding only
    seen_atoms = set(int(x) for x in np.unique(tr["pat_atom"].numpy()))
    seen_bonds = set(int(x) for x in np.unique(tr["pat_pair"].numpy()) if int(x) > 0)
    pat_atom = va["pat_atom"].numpy()
    pat_pair = va["pat_pair"].numpy()
    pat_len = va["pat_len"].numpy()
    ent_ptr = va["ent_ptr"].numpy()
    ent_pat = va["ent_pat"].numpy()
    n_mol = len(ent_ptr) - 1
    atom_seen = np.zeros(n_mol, dtype=bool)
    bond_seen = np.zeros(n_mol, dtype=bool)
    for m in range(n_mol):
        pats = ent_pat[ent_ptr[m]:ent_ptr[m + 1]]
        a_ok, b_ok = True, True
        for p in pats:
            for t in range(int(pat_len[p])):
                if int(pat_atom[p, t]) not in seen_atoms:
                    a_ok = False
            for k in range(pat_pair.shape[1]):
                b = int(pat_pair[p, k])
                if b > 0 and b not in seen_bonds:
                    b_ok = False
        atom_seen[m], bond_seen[m] = a_ok, b_ok
    return {
        "train_atom_types": sorted(seen_atoms),
        "train_bond_types": sorted(seen_bonds),
        "valid_all_atom_types_seen_frac": float(atom_seen.mean()),
        "valid_all_bond_types_seen_frac": float(bond_seen.mean()),
        "valid_all_types_seen_frac": float((atom_seen & bond_seen).mean()),
        "atom_seen": atom_seen,
        "bond_seen": bond_seen,
    }


def group_metrics(preds: dict[str, dict], groups: dict[str, np.ndarray],
                  cover: dict) -> dict:
    if not groups:
        return {}
    base_rows = next(iter(preds.values()))["row"] if preds else None
    if base_rows is None:
        return {}
    order = np.argsort(groups["row"])
    g_row = groups["row"][order]
    pos = np.searchsorted(g_row, base_rows)
    if not np.array_equal(g_row[pos], base_rows):
        raise RuntimeError("group reference rows do not match the prediction rows")
    unseen = groups["r2_unseen_atom_frac"][order][pos] > 0.0
    ordinary = groups["ordinary"][order][pos] > 0
    cells = {
        "seen_ordinary": (~unseen) & ordinary,
        "seen_tail": (~unseen) & (~ordinary),
        "unseen_ordinary": unseen & ordinary,
        "unseen_tail": unseen & (~ordinary),
    }
    coverage = cover.get("atom_seen") if cover else None
    out: dict[str, dict] = {}
    n_total = base_rows.size
    for name, mask in cells.items():
        cell: dict = {"n": int(mask.sum())}
        for key, pr in preds.items():
            ae = np.abs(pr["pred"][mask] - pr["y"][mask])
            cell[key] = {"mae": float(ae.mean()) if ae.size else None,
                         "weighted_contribution": (float(ae.sum()) / n_total
                                                   if ae.size else None)}
        if coverage is not None and mask.any():
            cell["all_atom_types_seen_frac"] = float(coverage[mask].mean())
        out[name] = cell
    out["n_total"] = int(n_total)
    out["unseen_frac"] = float(unseen.mean())
    return out


def extreme_diagnostics(preds: dict[str, dict], keys: tuple[str, str], top: int = 10) -> dict:
    a, b = keys
    if a not in preds or b not in preds:
        return {}
    pa, pb = preds[a], preds[b]
    ae_a = np.abs(pa["pred"] - pa["y"])
    ae_b = np.abs(pb["pred"] - pb["y"])
    idx = np.argsort(-ae_a)[:top]
    rows = [{
        "row": int(pa["row"][i]),
        "y_raw": float(pa["y"][i]),
        "pred_a": float(pa["pred"][i]),
        "pred_b": float(pb["pred"][i]),
        "ae_a": float(ae_a[i]),
        "ae_b": float(ae_b[i]),
        "delta_ae": float(ae_a[i] - ae_b[i]),
    } for i in idx]
    d = ae_a - ae_b
    order = np.argsort(-np.abs(d))
    return {
        "pair": [a, b],
        "worst_of_a": rows,
        "largest_abs_delta": [{
            "row": int(pa["row"][i]), "y_raw": float(pa["y"][i]),
            "ae_a": float(ae_a[i]), "ae_b": float(ae_b[i]),
            "delta_ae": float(d[i]),
            "y_ne_minus3": bool(pa["y"][i] <= -3.0),
        } for i in order[:top]],
        "n_y_ne_minus3": int((pa["y"] <= -3.0).sum()),
        "mae_excluding_y_ne_minus3_a": float(ae_a[pa["y"] > -3.0].mean()),
        "mae_excluding_y_ne_minus3_b": float(ae_b[pb["y"] > -3.0].mean()),
    }


# ---------------------------------------------------------------------------
# trained atomic state diagnostics (auxiliary evidence)
# ---------------------------------------------------------------------------


def state_diagnostics(arm: str, seed: int) -> dict:
    if arm == "D":
        return {}
    run = RUNS / f"SF_{arm}_s{seed}" / "best.pt"
    if not run.exists():
        return {}
    import torch
    import sf_models as SF  # noqa: F401  (registers the module layout)

    ck = torch.load(run, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    st = SF.AtomicState()
    st.load_state_dict({k.split("branch.state.")[1]: v for k, v in sd.items()
                        if k.startswith("branch.state.")})
    st = st.double()
    with torch.no_grad():
        q_log, log_k = st.q_log(), st.log_k()
    q = q_log.exp().numpy()                                           # [28,C,R]
    kk = log_k.exp().numpy()                                          # [3,C,R,R]
    ent = -(q * np.log(q + 1e-30)).sum(-1)
    off = ~np.eye(SF.R_STATE, dtype=bool)
    return {
        "q_max_prob_mean": float(q.max(-1).mean()),
        "q_entropy_mean": float(ent.mean()),
        "q_entropy_max": float(ent.max()),
        "q_argmax_state_hist": np.bincount(q.reshape(-1, SF.R_STATE).argmax(-1),
                                          minlength=SF.R_STATE).tolist(),
        "k_min": float(kk.min()), "k_max": float(kk.max()),
        "k_offdiag_std": float(kk[:, :, off].std()),
        "k_offdiag_mean": float(kk[:, :, off].mean()),
        "pot_abs_max": float(sd["branch.state.pot"].abs().max()),
        "q_min_prob": float(q.min()),
        "w_s_norm": float(np.linalg.norm(sd["w_s.weight"].double().numpy())),
    }


# ---------------------------------------------------------------------------
# gates (pre-registered, DESIGN.md §7)
# ---------------------------------------------------------------------------


def _cmp(a: dict | None, b: dict | None, key: str) -> dict:
    """Compare ``a`` (candidate) against ``b`` (baseline) on one metric."""
    if not a or not b or a.get(key) is None or b.get(key) is None:
        return {"available": False, "metric": key}
    return {"available": True, "metric": key,
            "candidate": float(a[key]), "baseline": float(b[key]),
            "delta": float(a[key] - b[key]), "supports_candidate": bool(a[key] < b[key])}


def gates(runs: dict[str, dict]) -> dict:
    out: dict = {"thresholds": {"margin": MARGIN, "abs_gate": ABS_GATE,
                                "window": WINDOW, "tail": TAIL}, "stages": {}}
    for seed in SEEDS:
        T = runs.get(f"T_s{seed}")
        D = runs.get(f"D_s{seed}")
        if not T or not D:
            out["stages"][f"A_s{seed}"] = {"available": False}
            continue
        checks = {
            "margin": {"T": T["best_valid"], "D": D["best_valid"],
                       "required": "T <= D - 0.003",
                       "pass": bool(T["best_valid"] <= D["best_valid"] - MARGIN)},
            "absolute": {"T": T["best_valid"], "required": "T <= 0.075",
                         "pass": bool(T["best_valid"] <= ABS_GATE)},
            "window_median": _cmp(T, D, "best20_window_median") | {
                "required": "T < D (20-epoch sliding median)"},
            "tail100_median": _cmp(T, D, "tail100_median") | {
                "required": "T < D (tail-100 median)"},
        }
        window_ok = checks["window_median"].get("supports_candidate", False)
        tail_ok = checks["tail100_median"].get("supports_candidate", False)
        checks["window_median"]["pass"] = bool(window_ok)
        checks["tail100_median"]["pass"] = bool(tail_ok)
        stage_b_ok = (T["best_valid"] < D["best_valid"] and window_ok)
        out["stages"][f"A_s{seed}"] = {
            "available": True,
            "checks": checks,
            "pass": bool(checks["margin"]["pass"] and checks["absolute"]["pass"]
                         and window_ok and tail_ok),
            "would_satisfy_B_direction": bool(stage_b_ok),
            "B_executed": False,
            "B_reason": ("stage A failed, so stage B/C were not launched (DESIGN.md §7)"
                         if not (checks["margin"]["pass"] and checks["absolute"]["pass"]
                                 and window_ok and tail_ok) else "stage A passed; B not run yet"),
        }
    n_s1 = runs.get("N_s0") and runs.get("N_s1")
    if n_s1:
        out["stages"]["C"] = {
            "available": True,
            "T_vs_N_s0": _cmp(runs.get("T_s0"), runs.get("N_s0"), "best_valid"),
            "T_vs_N_s1": _cmp(runs.get("T_s1"), runs.get("N_s1"), "best_valid"),
        }
    else:
        out["stages"]["C"] = {"available": False,
                              "reason": "N runs not executed (stage B not repeated)"}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="analyse zinc_shared_factor_v1 runs")
    ap.add_argument("--arms", default="D,T,N")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    runs: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    preds_final: dict[str, dict] = {}
    for arm in arms:
        for seed in seeds:
            m = run_metrics(arm, seed)
            if m is None:
                continue
            key = f"{arm}_s{seed}"
            m["state"] = state_diagnostics(arm, seed)
            runs[key] = m
            p = load_preds(arm, seed, "best")
            if p is not None:
                preds[key] = p
                preds_final[key] = load_preds(arm, seed, "final") or p

    rng = np.random.default_rng(BOOT_SEED)
    contrasts: dict[str, dict] = {}
    for seed in seeds:
        contrasts[f"T_minus_D_s{seed}"] = contrast(preds.get(f"T_s{seed}"),
                                                   preds.get(f"D_s{seed}"), rng)
        contrasts[f"T_minus_N_s{seed}"] = contrast(preds.get(f"T_s{seed}"),
                                                   preds.get(f"N_s{seed}"), rng)
        contrasts[f"N_minus_D_s{seed}"] = contrast(preds.get(f"N_s{seed}"),
                                                   preds.get(f"D_s{seed}"), rng)

    cover = type_coverage() if preds else {}
    groups = group_metrics(preds, load_groups(), cover) if preds else {}
    extremes = {f"T_vs_D_s{seed}": extreme_diagnostics(preds, (f"D_s{seed}", f"T_s{seed}"))
                for seed in seeds if f"D_s{seed}" in preds and f"T_s{seed}" in preds}

    # merged per-molecule predictions
    if preds:
        base = next(iter(preds.values()))
        cols = {"row": base["row"], "y_raw": base["y"]}
        for key, p in preds.items():
            if not np.array_equal(p["row"], base["row"]):
                raise RuntimeError("prediction rows differ across arms")
            cols[f"pred_{key}"] = p["pred"]
        for key, p in preds_final.items():
            if key not in preds or p is base:
                continue
            cols[f"pred_{key}_final"] = p["pred"]
        if groups:
            order = np.argsort(load_groups()["row"])
            g = load_groups()
            for name in ("r2_unseen_atom_frac", "signature_unseen", "ordinary"):
                cols[name] = g[name][order][np.searchsorted(g["row"][order], base["row"])]
        path = ANALYSIS / "preds_valid.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            keys = list(cols)
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(base["row"].size):
                w.writerow([f"{cols[k][i]:.9f}" if k != "row" and np.issubdtype(
                    np.asarray(cols[k]).dtype, np.floating) else int(cols[k][i])
                    if np.issubdtype(np.asarray(cols[k]).dtype, np.integer) else cols[k][i]
                    for k in keys])

    summary = {
        "protocol": "zinc-shared-factor-v1",
        "runs": runs,
        "contrasts_best_checkpoint": contrasts,
        "groups_radius2_x_ordinary": groups,
        "type_coverage": {k: v for k, v in (cover or {}).items()
                          if not isinstance(v, np.ndarray)},
        "extreme_diagnostics": extremes,
        "notes": [
            "main metric is full-split valid MAE over all 1000 molecules",
            "bootstrap intervals describe molecule sampling only, not seed variance",
            "groups come from the frozen zinc_restricted_adaptation_v1 reference",
            "no official-test predictions exist in this experiment",
        ],
    }
    write_json(ANALYSIS / "summary.json", summary)
    gate_doc = gates(runs)
    write_json(ANALYSIS / "gates.json", gate_doc)

    lines = ["# zinc_shared_factor_v1", "",
             "| arm | seed | params | best valid | best ep | final valid | best20 median"
             " | tail100 median | s/epoch | GPU-h |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, m in runs.items():
        c = m["curve"]
        lines.append(
            f"| {m['arm_name']} | {m['seed']} | {m['params']} | {m['best_valid']:.6f} | "
            f"{m['best_epoch']} | {m['final_valid']:.6f} | "
            f"{(c['best20_window_median'] or float('nan')):.6f} | "
            f"{(c['tail100_median'] or float('nan')):.6f} | "
            f"{c['seconds_per_epoch']:.2f} | {(m['gpu_hours'] or 0.0):.3f} |")
    lines += ["", "## contrasts (best checkpoint, per molecule)", "",
              "| pair | dMAE | 95% CI (molecules) | frac improved | median dAE |",
              "| --- | --- | --- | --- | --- |"]
    for key, ct in contrasts.items():
        if not ct.get("available"):
            lines.append(f"| {key} | n/a | n/a | n/a | n/a |")
            continue
        lo, hi = ct["ci95"]
        lines.append(f"| {key} | {ct['delta_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
                     f"{ct['frac_improved']:.3f} | {ct['median_delta_ae']:+.6f} |")
    if groups:
        lines += ["", "## groups (valid, radius-2 x ordinary)", "",
                  "| cell | n | " + " | ".join(preds) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in preds) + " |"]
        for cell in ("seen_ordinary", "seen_tail", "unseen_ordinary", "unseen_tail"):
            g = groups[cell]
            lines.append(f"| {cell} | {g['n']} | " + " | ".join(
                f"{(g[k]['mae'] or float('nan')):.4f}" for k in preds) + " |")
    lines += ["", "## stage gates", "", "```json",
              json.dumps(gate_doc["stages"], indent=1), "```", ""]
    (ANALYSIS / "main_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n[analysis] {len(runs)} runs -> {ANALYSIS/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
