"""Analysis and frozen decision rules for LCRC-HC3-v1.

Reads only train/valid caches and the frozen historical references.  Official
test is never opened.  Writes machine-readable JSON plus markdown tables used by
REPORT.md / DECISION.md.

Usage:
    python hc_analysis.py --stage A
    python hc_analysis.py --stage B
    python hc_analysis.py --stage C
    python hc_analysis.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import hc_common as C

SLICES = ("full", "ordinary", "tail", "any_unseen", "unseen_ordinary",
          "seen_ordinary", "minus_fixed_worst10", "minus_row172")
GROUPS_MAIN = ("full", "ordinary", "unseen_ordinary", "seen_ordinary", "tail")
N_BOOT = 5000
BOOT_SEED = 20261111
PROTOCOL = C.read_json(C.CONFIGS / "frozen_protocol.json")


# ---------------------------------------------------------------------------
# data / support
# ---------------------------------------------------------------------------


def _signatures(graphs):
    per_atom = []
    for atoms, bonds in graphs:
        nbrs = [[] for _ in range(len(atoms))]
        for u, v, t in bonds:
            nbrs[u].append((t, v))
            nbrs[v].append((t, u))
        one = [f"r1({int(atoms[i])};"
               + ",".join(sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
               for i in range(len(atoms))]
        two = [f"r2({one[i]};"
               + ",".join(sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
               for i in range(len(atoms))]
        per_atom.append(two)
    return per_atom


def support_masks(train, fit_rows, query_cache, query_rows):
    """Radius-2 unseen with Fit8k and with full-train10k vocabularies."""
    from hc_erratum import _signatures  # exact same frozen construction
    train_graphs = _molecules(train)
    all_query = _molecules(query_cache)
    query_rows = np.asarray(query_rows, dtype=np.int64)
    query_graphs = [all_query[int(r)] for r in query_rows]
    ts = _signatures(train_graphs)
    qs = _signatures(query_graphs)
    fit_set = set(int(x) for x in np.asarray(fit_rows))
    full_freq, fit_freq = {}, {}
    for m, sigs in enumerate(ts):
        for s in sigs:
            full_freq[s] = full_freq.get(s, 0) + 1
            if m in fit_set:
                fit_freq[s] = fit_freq.get(s, 0) + 1
    old_u, new_u = [], []
    for sigs in qs:
        old_u.append(any(fit_freq.get(s, 0) == 0 for s in sigs) if sigs else False)
        new_u.append(any(full_freq.get(s, 0) == 0 for s in sigs) if sigs else False)
    y = np.asarray(query_cache["labels"], dtype=np.float64)[np.asarray(query_rows)]
    ordinary = y > C.PENALTY_Y
    out = {"ordinary": ordinary, "tail": ~ordinary,
           "full": np.ones(len(y), dtype=bool),
           "vocab": {"fit8k": int(len(fit_freq)), "train10k": int(len(full_freq))}}
    for tag, unseen in (("fit8k", np.asarray(old_u)), ("train10k", np.asarray(new_u))):
        out[tag] = {"any_unseen": unseen,
                    "unseen_ordinary": unseen & ordinary,
                    "seen_ordinary": (~unseen) & ordinary}
    return out


def _molecules(cache):
    from hc_erratum import XM
    return XM.molecules(cache)


def _slices(sup, vocab):
    unseen = sup[vocab]["any_unseen"]
    out = {
        "full": sup["full"],
        "ordinary": sup["ordinary"],
        "tail": sup["tail"],
        "any_unseen": unseen,
        "unseen_ordinary": unseen & sup["ordinary"],
        "seen_ordinary": (~unseen) & sup["ordinary"],
    }
    return out


# ---------------------------------------------------------------------------
# run loading / metrics
# ---------------------------------------------------------------------------


def run_dir(name: str) -> Path:
    return C.RUNS / name


def load_train(name: str) -> dict:
    return C.read_json(run_dir(name) / "train.json")


def load_pred(name: str, checkpoint: str = "best") -> tuple[np.ndarray, np.ndarray]:
    path = run_dir(name) / f"predictions_{checkpoint}.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    rows = np.asarray(d["rows"], dtype=np.int64)
    pred = np.asarray(d["pred"], dtype=np.float64)
    order = np.argsort(rows)
    return rows[order], pred[order]


def load_lcrc_baseline(seed: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(C.LCRC / "runs" / f"lcrcA_s{seed}" / "predictions" / "valid_best.npz")
    rows = np.asarray(d["row"], dtype=np.int64)
    pred = np.asarray(d["pred"], dtype=np.float64)
    order = np.argsort(rows)
    return rows[order], pred[order]


def load_dccl() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(C.DCCL_PERMOL)
    rows = np.asarray(d["row_id"], dtype=np.int64)
    preds = np.stack([np.asarray(d[f"pred_DCCL_s{s}"], dtype=np.float64) for s in range(4)], 0)
    order = np.argsort(rows)
    return rows[order], preds[:, order]


def mae(pred, y, mask) -> float:
    diff = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return float(diff[np.asarray(mask, bool)].mean())


def ae(pred, y) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))


def paired_bootstrap(pred_a, pred_b, y, mask, *, n=N_BOOT, seed=BOOT_SEED) -> dict:
    mask = np.asarray(mask, bool)
    diff = (np.abs(np.asarray(pred_a)[mask] - np.asarray(y)[mask])
            - np.abs(np.asarray(pred_b)[mask] - np.asarray(y)[mask]))
    if diff.size == 0:
        return {"n": 0, "delta": None, "ci95": [None, None], "win_rate": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(1)
    return {"n": int(diff.size), "delta": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "win_rate": float((diff < 0).mean())}


def curve_stats(train: dict) -> dict:
    vals = [float(e["monitor_mae"]) for e in train["curve"]]
    w20, w20_ep = None, None
    for end in range(20, len(vals) + 1):
        med = float(np.median(vals[end - 20:end]))
        if w20 is None or med < w20:
            w20, w20_ep = med, end
    tail = vals[-min(100, len(vals)):]
    return {
        "best": float(train["best"]["monitor_mae"]),
        "best_epoch": int(train["best"]["epoch"]),
        "best_step": int(train["best"]["step"]),
        "W20": w20,
        "W20_epoch": w20_ep,
        "T100": float(np.median(tail)) if tail else None,
        "final": float(vals[-1]),
        "epochs_run": int(len(vals)),
        "gpu_hours": float(train.get("gpu_hours") or 0.0),
        "parameters": int(train["parameters"]),
    }


# ---------------------------------------------------------------------------
# fixed extreme rows (frozen before the runs)
# ---------------------------------------------------------------------------


def fixed_monitor_worst10() -> dict:
    base = C.read_json(C.SUFF / "zinc_compositional_research_tree_v1/analysis/frozen_baseline.json")
    positions = [int(x) for x in base["fixed_worst10_monitor"]["positions"]]
    return {"positions": positions, "source": base["fixed_worst10_monitor"]["selection"]}


def fixed_valid_worst10() -> dict:
    """Top-10 |error| of the historical LCRC-A seed-0 valid checkpoint."""
    rows, pred = load_lcrc_baseline(0)
    va = C.load_cache("valid")
    y = np.asarray(va["labels"], dtype=np.float64)[rows]
    err = np.abs(pred - y)
    pos = np.argsort(-err)[:10]
    return {"positions": [int(x) for x in pos], "rows": [int(rows[x]) for x in pos],
            "source": "historical lcrcA_s0 best-valid checkpoint, fixed before Stage C"}


def penalty_mask() -> np.ndarray:
    vals = {}
    with open(C.CYCLE_AUDIT_VALID, newline="") as fh:
        for r in csv.DictReader(fh):
            idx = int(r["subset_index"])
            vals[idx] = float(r["label_excess"]) > 0
    return np.asarray([vals[i] for i in range(len(vals))], dtype=bool)


def penalty_mask_train() -> np.ndarray:
    path = C.CYCLE_AUDIT_VALID.parent / "train_cycle_audit_label.csv"
    vals = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            idx = int(r["subset_index"])
            vals[idx] = float(r["label_excess"]) > 0
    return np.asarray([vals[i] for i in range(len(vals))], dtype=bool)


# ---------------------------------------------------------------------------
# Stage A / B
# ---------------------------------------------------------------------------


def analyse_fit8k(stage: str, seed: int) -> dict:
    names = {"B": f"B_s{seed}_fit8k", "H": f"H_s{seed}_fit8k", "F": f"F_s{seed}_fit8k"}
    train = {k: load_train(v) for k, v in names.items()}
    preds = {k: load_pred(v) for k, v in names.items()}
    rows = preds["B"][0]

    train_cache = C.load_cache("train")
    fit_rows = C.split_rows()["rows"]["fit8k"]
    sup = support_masks(train_cache, fit_rows, train_cache, rows)
    base_slices = _slices(sup, "fit8k")
    fixed = fixed_monitor_worst10()
    pos = fixed["positions"]
    masks = dict(base_slices)
    for k, p in (("minus_fixed_worst10", pos), ("minus_row172", 172)):
        m = np.ones(len(rows), dtype=bool)
        m[p] = False
        masks[k] = m
    y = np.asarray(train_cache["labels"], dtype=np.float64)[rows]

    table = {}
    for arm in ("B", "H", "F"):
        table[arm] = {s: mae(preds[arm][1], y, masks[s]) for s in SLICES}
        table[arm]["curve"] = curve_stats(train[arm])

    deltas = {}
    for name, a, b in (("H_minus_B", "H", "B"), ("H_minus_F", "H", "F"),
                       ("F_minus_B", "F", "B")):
        deltas[name] = {}
        for i, s in enumerate(SLICES):
            deltas[name][s] = paired_bootstrap(preds[a][1], preds[b][1], y, masks[s],
                                               seed=BOOT_SEED + i + 17 * seed)

    # descriptive penalty x tail decomposition on Monitor (train cycle audit)
    pen_all = penalty_mask_train()
    pen = pen_all[rows]
    decomp = []
    for pen_name, pm in (("penalty", pen), ("no_penalty", ~pen)):
        for tail_name, tm in (("tail", y <= C.PENALTY_Y), ("ordinary", y > C.PENALTY_Y)):
            m = pm & tm
            if not m.any():
                continue
            row = {"penalty": pen_name, "tail": tail_name, "n": int(m.sum())}
            for arm in ("B", "H", "F"):
                row[arm] = mae(preds[arm][1], y, m)
            row["H_minus_B"] = paired_bootstrap(preds["H"][1], preds["B"][1], y, m,
                                                seed=BOOT_SEED + 200 + len(decomp) + 17 * seed)
            decomp.append(row)

    # paired per-molecule deltas (negative favors H)
    ae_b = ae(preds["B"][1], y)
    ae_h = ae(preds["H"][1], y)
    ae_f = ae(preds["F"][1], y)
    d_hb, d_hf = ae_h - ae_b, ae_h - ae_f
    pairwise = {}
    for label, m in masks.items():
        pairwise[label] = {}
        for name, d in (("H_minus_B", d_hb), ("H_minus_F", d_hf),
                        ("F_minus_B", ae_f - ae_b)):
            x = d[m]
            pairwise[label][name] = {
                "n": int(x.size), "mean": float(x.mean()) if x.size else None,
                "median": float(np.median(x)) if x.size else None,
                "H_wins": int((x < 0).sum()), "ties": int((x == 0).sum()),
                "H_losses": int((x > 0).sum()),
            }
    order = np.argsort(-np.abs(d_hf))
    top_hf = [{"row": int(rows[i]), "y": float(y[i]),
               "pred_H": float(preds["H"][1][i]), "pred_F": float(preds["F"][1][i]),
               "pred_B": float(preds["B"][1][i]),
               "d_HF": float(d_hf[i]), "d_HB": float(d_hb[i]),
               "ordinary": bool(masks["ordinary"][i]),
               "unseen_ordinary": bool(masks["unseen_ordinary"][i]),
               "penalty": bool(pen[i])} for i in order[:10]]

    # descriptive one-row sensitivity (row 374 is not a pre-registered gate)
    i374 = int(np.where(rows == 374)[0][0])
    keep = np.ones(len(rows), dtype=bool)
    keep[i374] = False
    sens = {
        "row": 374, "position": int(i374),
        "d_HB": float(d_hb[i374]), "d_HF": float(d_hf[i374]),
        "full_H_minus_B_excluding": float(d_hb[keep].mean()),
        "full_H_minus_F_excluding": float(d_hf[keep].mean()),
        "share_of_HB_delta_from_row": float(d_hb[i374] / d_hb.sum()),
        "share_of_HF_delta_from_row": float(d_hf[i374] / d_hf.sum()),
        "post_hoc_not_a_gate": True,
    }
    sens["by_slice_excluding"] = {}
    for label, m in masks.items():
        k = m & keep
        sens["by_slice_excluding"][label] = {
            "n": int(k.sum()),
            "H_minus_B": float(d_hb[k].mean()) if k.any() else None,
            "H_minus_F": float(d_hf[k].mean()) if k.any() else None,
        }

    # fixed-worst10 share of the net AE reduction H vs B
    net = float((ae_b - ae_h).sum())
    w10 = float((ae_b[pos] - ae_h[pos]).sum())
    worst10_share = float(w10 / net) if net > 0 else None

    curves = {arm: table[arm]["curve"] for arm in ("B", "H", "F")}
    smooth = {}
    for a, b in (("H", "B"), ("H", "F")):
        smooth[f"{a}_minus_{b}"] = {
            k: (curves[a][k] - curves[b][k] if curves[a][k] is not None
                and curves[b][k] is not None else None)
            for k in ("W20", "T100", "final", "best")
        }

    rules = PROTOCOL["stage_a_gates"] if stage == "A" else PROTOCOL["stage_b_requirements"]
    if stage == "A":
        checks = {
            "full_H_minus_B": deltas["H_minus_B"]["full"]["delta"] <= rules["full_H_minus_B_max"],
            "full_H_minus_F": deltas["H_minus_F"]["full"]["delta"] <= rules["full_H_minus_F_max"],
            "ordinary_H_minus_B": deltas["H_minus_B"]["ordinary"]["delta"]
            <= rules["ordinary_H_minus_B_max"],
            "unseen_ordinary_H_minus_B": deltas["H_minus_B"]["unseen_ordinary"]["delta"]
            <= rules["unseen_ordinary_H_minus_B_max"],
            "minus_worst10_H_minus_B": deltas["H_minus_B"]["minus_fixed_worst10"]["delta"]
            <= rules["minus_fixed_worst10_H_minus_B_max"],
            "worst10_share": (worst10_share is not None
                              and worst10_share <= rules["worst10_net_AE_share_max"]),
            "W20_H_minus_B": smooth["H_minus_B"]["W20"] is not None
            and smooth["H_minus_B"]["W20"] <= rules["W20_H_minus_B_max"],
            "W20_H_minus_F": smooth["H_minus_F"]["W20"] is not None
            and smooth["H_minus_F"]["W20"] <= rules["W20_H_minus_F_max"],
            "T100_H_minus_B": smooth["H_minus_B"]["T100"] is not None
            and smooth["H_minus_B"]["T100"] < rules["T100_H_minus_B_max"],
            "T100_H_minus_F": smooth["H_minus_F"]["T100"] is not None
            and smooth["H_minus_F"]["T100"] < rules["T100_H_minus_F_max"],
            "final_H_minus_B": smooth["H_minus_B"]["final"] <= rules["final_H_minus_B_max"],
            "final_H_minus_F": smooth["H_minus_F"]["final"] <= rules["final_H_minus_F_max"],
        }
        verdict = "PASS" if all(checks.values()) else "STOP"
    else:
        checks = {
            "full_H_minus_B": deltas["H_minus_B"]["full"]["delta"]
            < rules["full_H_minus_B_max"],
            "full_H_minus_F": deltas["H_minus_F"]["full"]["delta"]
            < rules["full_H_minus_F_max"],
            "W20_H_minus_B": smooth["H_minus_B"]["W20"] is not None
            and smooth["H_minus_B"]["W20"] < rules["W20_H_minus_B_max"],
            "W20_H_minus_F": smooth["H_minus_F"]["W20"] is not None
            and smooth["H_minus_F"]["W20"] < rules["W20_H_minus_F_max"],
            "ordinary_no_reversal": deltas["H_minus_B"]["ordinary"]["delta"]
            <= rules["ordinary_H_minus_B_max"],
            "unseen_ordinary_no_reversal": deltas["H_minus_B"]["unseen_ordinary"]["delta"]
            <= rules["unseen_ordinary_H_minus_B_max"],
        }
        verdict = "PASS" if all(checks.values()) else "STOP"

    out = {
        "round": C.ROUND_ID, "stage": stage, "seed": int(seed),
        "fit": "fit8k", "monitor": "monitor",
        "runs": names, "rows": rows.tolist(),
        "support": {"vocab_fit8k": sup["vocab"]["fit8k"],
                    "counts": {k: int(v.sum()) for k, v in base_slices.items()}},
        "fixed_worst10": fixed,
        "table": table, "deltas": deltas, "smoothing": smooth,
        "penalty_tail": decomp, "pairwise": pairwise, "top_HF_rows": top_hf,
        "row374_sensitivity": sens,
        "worst10_net_AE_reduction": {"net": net, "worst10": w10,
                                     "share": worst10_share},
        "checks": checks, "verdict": verdict,
        "rules": rules,
        "provenance": {arm: {"train_json": str(run_dir(names[arm]) / "train.json"),
                             "sha256": C.sha256_file(run_dir(names[arm]) / "train.json"),
                             "checkpoint_sha256": train[arm]["checkpoint_sha256"]}
                       for arm in names},
        "gpu_hours": float(sum(train[a]["gpu_hours"] for a in names)),
    }
    C.write_json(C.ANALYSIS / f"stage{stage}_s{seed}.json", out)
    np.savez(C.ANALYSIS / f"per_molecule_stage{stage}_s{seed}.npz",
             rows=rows, y=y,
             pred_B=preds["B"][1], pred_H=preds["H"][1], pred_F=preds["F"][1],
             penalty=pen, **{k: v for k, v in masks.items()})
    import pandas as pd
    pd.DataFrame({
        "row": rows, "y": y, "ordinary": masks["ordinary"],
        "tail": masks["tail"], "penalty": pen,
        "unseen_ordinary": masks["unseen_ordinary"],
        "pred_B": preds["B"][1], "pred_H": preds["H"][1], "pred_F": preds["F"][1],
        "AE_B": ae_b, "AE_H": ae_h, "AE_F": ae_f,
        "d_HB": d_hb, "d_HF": d_hf,
    }).to_csv(C.ANALYSIS / f"per_molecule_stage{stage}_s{seed}.csv", index=False)
    _write_ab_markdown(out)
    return out


def _write_ab_markdown(out: dict) -> None:
    stage, seed = out["stage"], out["seed"]
    lines = [f"# Stage {stage} — Fit8k / Monitor, seed {seed}", "",
             f"Verdict: **{out['verdict']}**", "",
             "## Support groups (Fit8k vocabulary)", "",
             "| group | n |", "|---|---:|"]
    for k, v in out["support"]["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Best-checkpoint Monitor MAE", "",
              "| slice | B | H | F | H−B [95% CI] | H−F [95% CI] |",
              "|---|---:|---:|---:|---|---|"]
    for s in SLICES:
        b, h, f = out["table"]["B"][s], out["table"]["H"][s], out["table"]["F"][s]
        hb, hf = out["deltas"]["H_minus_B"][s], out["deltas"]["H_minus_F"][s]
        lines.append(
            f"| {s} | {b:.6f} | {h:.6f} | {f:.6f} | "
            f"{hb['delta']:+.6f} [{hb['ci95'][0]:+.6f}, {hb['ci95'][1]:+.6f}] | "
            f"{hf['delta']:+.6f} [{hf['ci95'][0]:+.6f}, {hf['ci95'][1]:+.6f}] |")
    lines += ["", "## Curves", "",
              "| arm | best | best ep | W20 | T100 | final | GPU·h |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("B", "H", "F"):
        c = out["table"][arm]["curve"]
        lines.append(f"| {arm} | {c['best']:.6f} | {c['best_epoch']} | {c['W20']:.6f} | "
                     f"{c['T100']:.6f} | {c['final']:.6f} | {c['gpu_hours']:.3f} |")
    lines += ["", "Penalty vs tail on Monitor (descriptive, never pooled):", "",
              "| group | n | B | H | F | H−B |", "|---|---:|---:|---:|---:|---|"]
    for r in out["penalty_tail"]:
        lines.append(f"| {r['penalty']}/{r['tail']} | {r['n']} | {r['B']:.6f} | "
                     f"{r['H']:.6f} | {r['F']:.6f} | {r['H_minus_B']['delta']:+.6f} |")
    lines += ["", "Paired per-molecule wins (H better / ties / H worse):", "",
              "| slice | H vs B | H vs F |", "|---|---|---|"]
    for s in SLICES:
        hb = out["pairwise"][s]["H_minus_B"]
        hf = out["pairwise"][s]["H_minus_F"]
        lines.append(f"| {s} | {hb['H_wins']}/{hb['ties']}/{hb['H_losses']} | "
                     f"{hf['H_wins']}/{hf['ties']}/{hf['H_losses']} |")
    lines += ["", "Largest |H−F| rows (negative favors H):", "",
              "| row | y | pred H | pred F | d_HF | d_HB | group |", "|---:|---:|---:|---:|---:|---:|---|"]
    for r in out["top_HF_rows"]:
        g = "penalty" if r["penalty"] else ("unseen_ord" if r["unseen_ordinary"]
                                            else ("tail" if not r["ordinary"] else "seen_ord"))
        lines.append(f"| {r['row']} | {r['y']:.4f} | {r['pred_H']:.4f} | "
                     f"{r['pred_F']:.4f} | {r['d_HF']:+.4f} | {r['d_HB']:+.4f} | {g} |")
    sens = out["row374_sensitivity"]
    lines += ["", "One-row sensitivity (molecule 374, Monitor position 39; post-hoc, **not** a gate): "
              f"full H−B without it {sens['full_H_minus_B_excluding']:+.6f}; "
              f"full H−F without it {sens['full_H_minus_F_excluding']:+.6f}.", ""]
    lines += ["Smoothing deltas:", ""]
    for k, v in out["smoothing"].items():
        lines.append(f"* {k}: " + ", ".join(
            f"{kk}={vv:+.6f}" for kk, vv in v.items() if vv is not None))
    lines += ["", "## Gates", ""]
    for k, v in out["checks"].items():
        lines.append(f"* {'PASS' if v else 'FAIL'} — {k}")
    w = out["worst10_net_AE_reduction"]
    share = "n/a" if w["share"] is None else f"{w['share']:.3f}"
    lines += ["", f"Fixed worst-10 share of net AE reduction: {share}"]
    (C.ANALYSIS / f"stage{stage}_s{seed}.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Stage C
# ---------------------------------------------------------------------------


def analyse_stage_c(seed: int) -> dict:
    names = {"H": f"H_s{seed}_train10k", "F": f"F_s{seed}_train10k"}
    train = {k: load_train(v) for k, v in names.items()}
    hpred = load_pred(names["H"])
    fpred = load_pred(names["F"])
    rows = hpred[0]
    assert np.array_equal(rows, fpred[0])
    b_rows, b_pred = load_lcrc_baseline(seed)
    assert np.array_equal(rows, b_rows)
    d_rows, d_pred = load_dccl()
    assert np.array_equal(rows, d_rows)
    dccl_s0 = d_pred[0]
    dccl_mean = d_pred.mean(0)

    va = C.load_cache("valid")
    tr = C.load_cache("train")
    sup = support_masks(tr, np.arange(int(tr["n_molecules"])), va, rows)
    base_slices = _slices(sup, "train10k")
    fixed = fixed_valid_worst10()
    masks = dict(base_slices)
    for k, p in (("minus_fixed_worst10", fixed["positions"]), ("minus_row172", 172)):
        m = np.ones(len(rows), dtype=bool)
        m[p] = False
        masks[k] = m
    y = np.asarray(va["labels"], dtype=np.float64)[rows]
    penalty = penalty_mask()
    assert len(penalty) == len(rows)

    arms = {"H": hpred[1], "F": fpred[1], "B": b_pred,
            "DCCL_s0": dccl_s0, "DCCL_mean": dccl_mean}
    table = {a: {s: mae(p, y, masks[s]) for s in SLICES} for a, p in arms.items()}
    table["H"]["curve"] = curve_stats(train["H"])
    table["F"]["curve"] = curve_stats(train["F"])
    ref = train_ref_lcrc(seed)
    table["B"]["curve"] = {
        "best": float(ref["best"]["valid_mae"]),
        "best_epoch": int(ref["best"]["epoch"]),
        "W20": float(ref["W20"]),
        "T100": float(ref["T100"]),
        "final": float(ref["final_valid_mae"]),
        "gpu_hours": float(ref["gpu_hours_estimate"]),
        "parameters": int(ref["model_info"]["parameters"]),
    }

    deltas = {}
    for name, a, b in (("H_minus_F", "H", "F"), ("H_minus_B", "H", "B"),
                       ("H_minus_DCCL_s0", "H", "DCCL_s0"),
                       ("H_minus_DCCL_mean", "H", "DCCL_mean"),
                       ("F_minus_B", "F", "B")):
        deltas[name] = {s: paired_bootstrap(arms[a], arms[b], y, masks[s],
                                            seed=BOOT_SEED + i + 17 * seed)
                        for i, s in enumerate(SLICES)}

    # penalty x tail decomposition
    decomp = []
    for pen_name, pen in (("penalty", penalty), ("no_penalty", ~penalty)):
        for tail_name, tail in (("tail", y <= C.PENALTY_Y), ("ordinary", y > C.PENALTY_Y)):
            m = pen & tail
            if not m.any():
                continue
            row = {"penalty": pen_name, "tail": tail_name, "n": int(m.sum())}
            for a, p in arms.items():
                row[a] = mae(p, y, m)
                row[f"{a}_ae"] = float(ae(p, y)[m].sum())
            row["H_minus_B"] = paired_bootstrap(arms["H"], arms["B"], y, m,
                                                seed=BOOT_SEED + len(decomp) + 100 * seed)
            decomp.append(row)

    claim = _claim_stage_c(table, deltas, masks)
    out = {
        "round": C.ROUND_ID, "stage": "C", "seed": int(seed), "fit": "train10k",
        "monitor": "valid", "runs": names,
        "rows": rows.tolist(),
        "support": {"vocab": "train10k", "size": sup["vocab"]["train10k"],
                    "counts": {k: int(v.sum()) for k, v in base_slices.items()}},
        "fixed_worst10": fixed,
        "table": table, "deltas": deltas, "penalty_tail": decomp,
        "claim": claim,
        "provenance": {
            "H": {"train_json": str(run_dir(names["H"]) / "train.json"),
                  "sha256": C.sha256_file(run_dir(names["H"]) / "train.json"),
                  "checkpoint_sha256": train["H"]["checkpoint_sha256"]},
            "F": {"train_json": str(run_dir(names["F"]) / "train.json"),
                  "sha256": C.sha256_file(run_dir(names["F"]) / "train.json"),
                  "checkpoint_sha256": train["F"]["checkpoint_sha256"]},
            "B": {"source": "historical lcrcA_s%d" % seed,
                  "checkpoint": str(C.LCRC / "runs" / f"lcrcA_s{seed}" / "best.pt"),
                  "checkpoint_sha256": C.sha256_file(
                      C.LCRC / "runs" / f"lcrcA_s{seed}" / "best.pt"),
                  "train_json": str(C.LCRC / "runs" / f"lcrcA_s{seed}" / "train.json")},
            "DCCL": {"source": str(C.DCCL_PERMOL),
                     "sha256": C.sha256_file(C.DCCL_PERMOL)},
        },
        "gpu_hours": float(train["H"]["gpu_hours"] + train["F"]["gpu_hours"]),
    }
    C.write_json(C.ANALYSIS / f"stageC_s{seed}.json", out)
    np.savez(C.ANALYSIS / f"per_molecule_stageC_s{seed}.npz",
             rows=rows, y=y,
             pred_H=arms["H"], pred_F=arms["F"], pred_B=arms["B"],
             pred_DCCL_s0=arms["DCCL_s0"], pred_DCCL_mean=arms["DCCL_mean"],
             ordinary=base_slices["ordinary"], tail=base_slices["tail"],
             any_unseen=base_slices["any_unseen"],
             unseen_ordinary=base_slices["unseen_ordinary"],
             seen_ordinary=base_slices["seen_ordinary"], penalty=penalty)
    _write_c_markdown(out)
    return out


def train_ref_lcrc(seed: int) -> dict:
    return C.read_json(C.LCRC / "runs" / f"lcrcA_s{seed}" / "train.json")


def _claim_stage_c(table, deltas, masks) -> dict:
    def ci_upper_neg(d):
        return d["ci95"][1] is not None and d["ci95"][1] < 0

    beats_b_full = deltas["H_minus_B"]["full"]["delta"] < 0 and \
        ci_upper_neg(deltas["H_minus_B"]["full"])
    beats_b_ord = deltas["H_minus_B"]["ordinary"]["delta"] < 0 and \
        ci_upper_neg(deltas["H_minus_B"]["ordinary"])
    beats_d_full = deltas["H_minus_DCCL_mean"]["full"]["delta"] < 0 and \
        ci_upper_neg(deltas["H_minus_DCCL_mean"]["full"])
    beats_d_ord = deltas["H_minus_DCCL_mean"]["ordinary"]["delta"] < 0 and \
        ci_upper_neg(deltas["H_minus_DCCL_mean"]["ordinary"])
    unseen_ok = deltas["H_minus_B"]["unseen_ordinary"]["delta"] <= 0.001
    beats_f = deltas["H_minus_F"]["full"]["delta"] < 0
    if beats_b_full and beats_b_ord and beats_d_full and beats_d_ord and unseen_ok:
        verdict = "BREAKTHROUGH"
    elif beats_b_full and beats_b_ord:
        verdict = "IMPROVEMENT_OVER_LCRC_ONLY"
    elif beats_f:
        verdict = "HIERARCHICAL_SIGNAL_ONLY"
    else:
        verdict = "NO_ADVANTAGE"
    return {
        "verdict": verdict,
        "beats_B_full_ci": beats_b_full,
        "beats_B_ordinary_ci": beats_b_ord,
        "beats_DCCL_mean_full_ci": beats_d_full,
        "beats_DCCL_mean_ordinary_ci": beats_d_ord,
        "unseen_ordinary_not_worse": unseen_ok,
        "beats_F_full": beats_f,
    }


def _write_c_markdown(out: dict) -> None:
    seed = out["seed"]
    lines = [f"# Stage C — official train10k / valid, seed {seed}", "",
             f"Claim: **{out['claim']['verdict']}**", "",
             "## Support (train10k vocabulary)", "",
             "| group | n |", "|---|---:|"]
    for k, v in out["support"]["counts"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Best-checkpoint valid MAE", "",
              "| slice | H | F | B (LCRC-A) | DCC-L s0 | DCC-L mean | H−F | H−B | H−DCCL mean |",
              "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for s in SLICES:
        t = out["table"]
        lines.append(
            f"| {s} | {t['H'][s]:.6f} | {t['F'][s]:.6f} | {t['B'][s]:.6f} | "
            f"{t['DCCL_s0'][s]:.6f} | {t['DCCL_mean'][s]:.6f} | "
            f"{out['deltas']['H_minus_F'][s]['delta']:+.6f} | "
            f"{out['deltas']['H_minus_B'][s]['delta']:+.6f} | "
            f"{out['deltas']['H_minus_DCCL_mean'][s]['delta']:+.6f} |")
    lines += ["", "## Curves", "",
              "| arm | best | W20 | T100 | final | GPU·h |", "|---|---:|---:|---:|---:|---:|"]
    for a in ("H", "F", "B"):
        c = out["table"][a]["curve"]
        lines.append(f"| {a} | {c['best']:.6f} | {c['W20']:.6f} | {c['T100']:.6f} | "
                     f"{c['final']:.6f} | {c['gpu_hours']:.3f} |")
    lines += ["", "## Penalty vs tail (never pooled)", "",
              "| group | n | H | F | B | H−B |", "|---|---:|---:|---:|---:|---|"]
    for r in out["penalty_tail"]:
        lines.append(f"| {r['penalty']}/{r['tail']} | {r['n']} | {r['H']:.6f} | "
                     f"{r['F']:.6f} | {r['B']:.6f} | {r['H_minus_B']['delta']:+.6f} |")
    (C.ANALYSIS / f"stageC_s{seed}.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def summary(stages: list[str]) -> dict:
    out = {"round": C.ROUND_ID, "stages": {}, "created": C.now()}
    for stage in ("A", "B", "C"):
        files = sorted(C.ANALYSIS.glob(f"stage{stage}_s*.json"))
        out["stages"][stage] = [C.read_json(f) for f in files]
    C.write_json(C.ANALYSIS / "decision_inputs.json", out)
    rows = []
    for stage, entries in out["stages"].items():
        for e in entries:
            for arm in ("B", "H", "F"):
                if arm not in e["table"]:
                    continue
                r = {"stage": stage, "seed": e["seed"], "arm": arm,
                     "best": e["table"][arm]["curve"]["best"],
                     "W20": e["table"][arm]["curve"]["W20"],
                     "T100": e["table"][arm]["curve"]["T100"],
                     "final": e["table"][arm]["curve"]["final"],
                     "gpu_hours": e["table"][arm]["curve"]["gpu_hours"]}
                for s in GROUPS_MAIN:
                    r[s] = e["table"][arm][s]
                rows.append(r)
    import pandas as pd
    pd.DataFrame(rows).to_csv(C.ANALYSIS / "main_table.csv", index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("A", "B", "C", "all"), default="all")
    args = p.parse_args()
    stages = []
    if args.stage in ("A", "all"):
        analyse_fit8k("A", 0)
        stages.append("A")
    if args.stage in ("B", "all"):
        analyse_fit8k("B", 1)
        stages.append("B")
    if args.stage in ("C", "all"):
        for seed in (0, 1):
            if (C.RUNS / f"H_s{seed}_train10k" / "train.json").exists():
                analyse_stage_c(seed)
                stages.append(f"C{seed}")
    summary(stages)
    print(json.dumps({"analysed": stages,
                      "files": sorted(str(p.name) for p in C.ANALYSIS.glob("stage*.json"))},
                     indent=1))


if __name__ == "__main__":
    main()
