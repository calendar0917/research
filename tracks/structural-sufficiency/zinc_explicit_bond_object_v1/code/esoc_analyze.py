"""ESOC-v1 frozen analysis (valid only; official test is never touched).

Reads the two formal runs plus historical reference predictions, and writes the
frozen deliverables:

  analysis/main_table.json        per-run metrics (best / W20 / T100 / final / eval-mode train)
  analysis/per_molecule_valid.csv per-molecule predictions and errors
  analysis/paired_contrasts.json  REAL - NULL paired bootstrap (fixed resamples)
  analysis/groups.json            frozen subgroup table
  analysis/structure_sensitivity.json  fixed-weight structure-sensitivity diagnostic
  analysis/ledger.json            GPU budget ledger

Every group mask, metric definition and resample seed is fixed here, before the
formal numbers are known, and no group is ever used for loss / routing /
selection.  ``valid`` is an exposed development split; intervals are descriptive
fixed-checkpoint intervals, not cross-seed confirmation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SUFF = _EXP.parent
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_models as M  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUNS = _EXP / "runs"
AUDIT_CSV = _SUFF / "zinc_generalization_audit_v1" / "analysis" / "per_molecule.csv"
WINDOW = 20
RESAMPLES = 10000
BOOT_SEED = 20261024
# historical absolute references (valid split, best checkpoint)
DCCL_VALID = 0.075990
DCCL_TEST = 0.067319
DCCL_PARAMS = 494193


# --------------------------------------------------------------------------- io
def load_run(arm: str, seed: int) -> dict:
    d = RUNS / f"{arm}_s{seed}"
    info = C.read_json(d / "train.json")
    out = {"arm": arm, "seed": seed, "dir": str(d), "info": info}
    for tag in ("best", "final"):
        z = np.load(d / "predictions" / f"valid_{tag}.npz", allow_pickle=False)
        out[f"pred_{tag}"] = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
        out[f"row_{tag}"] = np.asarray(z["row"], dtype=np.int64).reshape(-1)
        out[f"y_{tag}"] = np.asarray(z["y"], dtype=np.float64).reshape(-1)
    return out


def read_curve(path: Path) -> list[dict]:
    """train.json deliberately omits the full curve; it lives in curve.csv."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({"epoch": int(float(r["epoch"])),
                    "step": int(float(r["step"])),
                    "lr": float(r["lr"]),
                    "train_mae_online": float(r["train_mae_online"]),
                    "valid_mae": float(r["valid_mae"])})
    return out


def curve_metrics(curve: list[dict]) -> dict:
    valid = np.asarray([e["valid_mae"] for e in curve], dtype=np.float64)
    epochs = [int(e["epoch"]) for e in curve]
    n = len(valid)
    w20, w20_start = float("inf"), None
    if n >= WINDOW:
        for s in range(0, n - WINDOW + 1):
            med = float(np.median(valid[s:s + WINDOW]))
            if med < w20:
                w20, w20_start = med, epochs[s]
    t100 = float(np.median(valid[-100:])) if n >= 100 else float(np.median(valid))
    return {
        "W20": w20 if np.isfinite(w20) else float(np.median(valid)),
        "W20_window_start_epoch": w20_start,
        "T100": t100,
        "valid_min": float(valid.min()),
        "valid_min_epoch": int(epochs[int(np.argmin(valid))]),
        "valid_last": float(valid[-1]),
        "n_epochs": int(n),
    }


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    ae = np.abs(e)
    order = np.sort(ae)[::-1]
    total = float(ae.sum())
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(ae.max()),
        "mean_signed_error": float(e.mean()),
        "positive_error_fraction": float(np.mean(e > 0)),
        "top1_abs_error_share": float(order[:1].sum() / total),
        "top10_abs_error_share": float(order[:10].sum() / total),
        "top50_abs_error_share": float(order[:50].sum() / total),
        "top100_abs_error_share": float(order[:100].sum() / total),
    }


def bootstrap_delta(delta: np.ndarray, idx: np.ndarray) -> dict:
    means = delta[idx].mean(axis=1)
    return {
        "mean_delta_mae": float(delta.mean()),
        "median_delta_mae": float(np.median(delta)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_two_sided_permutation_free": float(
            2.0 * min((means <= 0).mean(), (means >= 0).mean())),
        "fraction_rows_improved": float(np.mean(delta < 0)),
    }


# ------------------------------------------------------------------ subgroups
def load_audit_rows() -> list[dict]:
    with open(AUDIT_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def frozen_masks(rows: list[dict], y: np.ndarray) -> dict:
    """All masks fixed before the ESOC results are seen."""
    n = len(rows)
    assert n == len(y), (n, len(y))
    unseen = np.asarray([float(r["r2_unseen_atom_frac"]) > 0 for r in rows])
    ordinary = y > -3.0
    rings4 = np.asarray([int(r["n_rings"]) >= 4 for r in rows])
    cyc = np.asarray([int(r["cycle_rank"]) for r in rows])
    row172 = np.zeros(n, dtype=bool)
    row172[172] = True
    # DCC-L worst-10 monitor: the 10 hardest valid rows for the historical
    # reference model, computed from its published per-molecule errors.
    ref_ae = np.asarray([float(r["abs_error_P3B_s0"]) for r in rows])
    worst10 = np.zeros(n, dtype=bool)
    worst10[np.argsort(ref_ae)[-10:]] = True
    worst100 = np.zeros(n, dtype=bool)
    worst100[np.argsort(ref_ae)[-100:]] = True
    return {
        "signature_unseen_251": unseen,
        "seen_ordinary": (~unseen) & ordinary,
        "unseen_ordinary": unseen & ordinary,
        "seen_tail": (~unseen) & (~ordinary),
        "unseen_tail": unseen & (~ordinary),
        "signature_covered_749": ~unseen,
        "ordinary_y_gt_minus3": ordinary,
        "tail_y_le_minus3": ~ordinary,
        "rings_ge4": rings4,
        "rings_ge4_and_signature_unseen": rings4 & unseen,
        "cycle_rank_ge3": cyc >= 3,
        "row_172": row172,
        "dcc_worst10": worst10,
        "dcc_worst100": worst100,
    }


def group_row(mask: np.ndarray, y: np.ndarray, pred: dict[str, np.ndarray]) -> dict:
    ae = {k: np.abs(v[mask] - y[mask]) for k, v in pred.items()}
    out = {"n": int(mask.sum()), "mae": {k: float(v.mean()) for k, v in ae.items()}}
    # Report the contrasts in the substantive direction, not in whatever order
    # ``sorted`` happens to produce.
    for a, b in (("real_s0", "null_s0"), ("real_s0", "hardperm_s0"),
                 ("hardperm_s0", "null_s0")):
        if a in ae and b in ae:
            d = ae[a] - ae[b]
            out[f"delta_{a}_minus_{b}"] = float(d.mean())
            out[f"fraction_rows_{a}_better_than_{b}"] = float(np.mean(d < 0))
    if "real_s0" in ae and "null_s0" in ae:
        out["fraction_rows_real_better"] = float(np.mean(ae["real_s0"] - ae["null_s0"] < 0))
    names = sorted(pred)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = ae[a] - ae[b]
            out[f"delta_{a}_minus_{b}"] = float(d.mean())
    return out


# ------------------------------------------------- structure-sensitivity probe
def structure_sensitivity(runs: dict, scales: dict, n_rows: int = 256,
                          n_perm: int = 3) -> dict:
    """Fixed-weight diagnostic: intervene on the *pairing* only.

    For every selected valid molecule the frozen model is evaluated three ways,
    with weights frozen (eval mode, no grad, dropout off):

      base    real pair-edge correspondence, as trained;
      perm    marginal-preserving permutation of the typed pair labels inside each
              object (``D.pairing_permutation``): every object keeps its exact
              multiset of pair labels, so the object-level class-frequency vector
              and the role/atom multiset are unchanged, and only the label-to-pair
              correspondence moves;
      destroy labels replaced by the object-level frequency vector (the NULL
              construction), i.e. correspondence removed entirely.

    Read-outs: mean |delta pred| for perm and for destroy, the fraction of rows
    whose prediction moves by more than 1e-6, and the correlation between the
    per-molecule perm sensitivity and the per-molecule REAL-vs-NULL advantage.
    A model that uses correspondence must react to ``perm``; a model that does
    not (the NULL arm) must be *exactly* invariant to it.
    """
    import torch

    valid_data = D.SplitTensors(D.load_cache("valid"), "cpu")
    rows = np.arange(min(n_rows, valid_data.n))
    b = valid_data.batch(rows)
    out = {"rows": int(len(rows)), "n_perm": int(n_perm),
           "role": ("descriptive fixed-weight diagnostic; weights frozen at the best "
                    "checkpoint, never used for selection")}

    def predict(net, batch):
        net.eval()
        with torch.no_grad():
            return net(batch).detach().double().numpy()

    et = b["loc_et"].clone()
    lm = b["loc_mask"]
    B, E, L, _ = et.shape
    perm_batches = []
    rng = np.random.default_rng(20261024)
    for _ in range(n_perm):
        et2 = et.clone().numpy()
        for i in range(B):
            for j in range(E):
                if not bool(lm[i, j].any()):
                    continue
                et2[i, j] = D.pairing_permutation(et2[i, j], lm[i, j].numpy(), rng)
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["loc_et"] = torch.as_tensor(np.ascontiguousarray(et2), dtype=torch.long)
        perm_batches.append(b2)

    # marginal check: the permutation must preserve each object's class histogram
    hist_ok = True
    for b2 in perm_batches:
        for i in range(B):
            for j in range(E):
                m = lm[i, j]
                if not bool(m.any()):
                    continue
                tri = np.triu_indices(int(m.sum()), k=1)
                idx = np.flatnonzero(m.numpy())[: int(m.sum())]
                a = np.bincount(np.asarray([et[i, j, idx[p], idx[q]].item()
                                            for p, q in zip(*tri)]), minlength=4)
                c = np.bincount(np.asarray([b2["loc_et"][i, j, idx[p], idx[q]].item()
                                            for p, q in zip(*tri)]), minlength=4)
                hist_ok = hist_ok and bool(np.array_equal(a, c))
    out["permutation_preserves_class_histogram"] = bool(hist_ok)

    for k, run in runs.items():
        net = M.build_model(run["seed"], run["arm"], scales)
        net.load_state_dict(torch.load(run["dir"] + "/best.pt", map_location="cpu",
                                       weights_only=False)["state"])
        base = predict(net, b)
        d_perm = [np.abs(predict(net, b2) - base) for b2 in perm_batches]
        # correspondence destroyed: the NULL forward path ignores the pairing
        b_destroy = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in b.items()}
        net_null = M.build_model(run["seed"], "null", scales)
        net_null.load_state_dict(net.state_dict())
        destroy = predict(net_null, b)
        out[k] = {
            "arm": run["arm"],
            "base_mean_pred": float(base.mean()),
            "perm_mean_abs_delta": float(np.mean([d.mean() for d in d_perm])),
            "perm_mean_abs_delta_per_perm": [float(d.mean()) for d in d_perm],
            "perm_max_abs_delta": float(max(d.max() for d in d_perm)),
            "perm_fraction_rows_moved_gt_1e-6": float(
                np.mean(np.stack([d > 1e-6 for d in d_perm]).any(axis=0))),
            "destroy_mean_abs_delta": float(np.abs(destroy - base).mean()),
            "destroy_max_abs_delta": float(np.abs(destroy - base).max()),
            "per_row_perm_abs_delta": [float(x) for x in np.mean(d_perm, axis=0)],
            "per_row_base_pred": [float(x) for x in base],
        }
    # sensitivity vs REAL-vs-NULL advantage, per molecule (best checkpoints)
    if "real_s0" in out and "null_s0" in out:
        y = runs["real_s0"]["y_best"][rows]
        adv = (np.abs(runs["null_s0"]["pred_best"][rows] - y)
               - np.abs(runs["real_s0"]["pred_best"][rows] - y))   # >0 => REAL better
        sens = np.asarray(out["real_s0"]["per_row_perm_abs_delta"])
        if sens.std() > 0 and adv.std() > 0:
            out["corr_perm_sensitivity_vs_real_advantage"] = float(np.corrcoef(sens, adv)[0, 1])
        else:
            out["corr_perm_sensitivity_vs_real_advantage"] = None
        out["real_advantage_definition"] = ("AE(NULL) - AE(REAL) per valid row; >0 means REAL "
                                            "is better on that molecule")
        out["real_advantage_mean_over_diagnostic_rows"] = float(adv.mean())
    for k in list(out):
        if isinstance(out[k], dict) and "per_row_perm_abs_delta" in out[k]:
            out[k]["per_row_perm_abs_delta"] = out[k]["per_row_perm_abs_delta"]
    return out


# --------------------------------------------------------------------- driver
def main() -> int:
    arms = [a for a in ("real", "null", "hardperm")
            if (RUNS / f"{a}_s0" / "train.json").exists()]
    runs = {f"{a}_s0": load_run(a, 0) for a in arms}
    if len(runs) < 2:
        raise SystemExit(f"need at least two seed-0 arms, found {sorted(runs)}")
    names = sorted(runs)
    y = runs[names[0]]["y_best"]
    for k in names:
        if not np.array_equal(runs[k]["row_best"], np.arange(len(y))):
            raise SystemExit(f"{k}: valid rows are not the fixed 0..999 order")
        if not np.allclose(runs[k]["y_best"], y):
            raise SystemExit(f"{k}: labels differ from the reference order")
        for tag in ("best", "final"):
            recomputed = float(np.abs(runs[k][f"pred_{tag}"] - y).mean())
            stored = float(runs[k]["info"]["checkpoints"][tag]["valid_mae"])
            if abs(recomputed - stored) > 1e-5:
                raise SystemExit(f"{k}/{tag}: recomputed {recomputed} != stored {stored}")

    # ---- per-run main table
    rows_main = []
    for k in names:
        info = runs[k]["info"]
        cm = curve_metrics(read_curve(Path(runs[k]["dir"]) / "curve.csv"))
        audit = info["param_audit"]
        rows_main.append({
            "run": k,
            "arm": runs[k]["arm"],
            "seed": runs[k]["seed"],
            "parameters": int(audit["param_total"]),
            "parameters_within_cap": bool(audit["param_total"] <= 500000),
            "best_valid": float(info["checkpoints"]["best"]["valid_mae"]),
            "best_epoch": int(info["checkpoints"]["best"]["epoch"]),
            "final_valid": float(info["checkpoints"]["final"]["valid_mae"]),
            "final_epoch": int(info["checkpoints"]["final"]["epoch"]),
            "W20": cm["W20"],
            "W20_window_start_epoch": cm["W20_window_start_epoch"],
            "T100": cm["T100"],
            "valid_min": cm["valid_min"],
            "valid_min_epoch": cm["valid_min_epoch"],
            "valid_last": cm["valid_last"],
            "epochs_completed": int(info["curve_note"]["epochs_completed"]),
            "epochs_declared": int(info["curve_note"]["epochs_declared"]),
            "aborted": info["curve_note"]["aborted"],
            "train_mae_eval_mode_at_best": float(info["checkpoints"]["best"]["train_mae_eval_mode"]),
            "train_mae_eval_mode_at_final": float(info["checkpoints"]["final"]["train_mae_eval_mode"]),
            "wall_hours": float(info["wall_seconds"]) / 3600.0,
            "epoch_seconds_median": info.get("epoch_seconds_median"),
            "compile": bool(info["recipe"].get("compile")),
            "compile_mode": info["recipe"].get("compile_mode"),
            "state_hash_final": info.get("state_hash_final"),
            "distance_to_dccl_valid": float(info["checkpoints"]["best"]["valid_mae"]) - DCCL_VALID,
            "dccl_valid_reference": DCCL_VALID,
        })
    C.write_json(ANALYSIS / "main_table.json", {
        "rows": rows_main,
        "valid_role": "exposed development split; official test never loaded",
        "dccl_reference": {"valid": DCCL_VALID, "test": DCCL_TEST, "parameters": DCCL_PARAMS,
                           "note": "historical absolute reference only; not a controlled contrast"},
        "metric_definitions": {
            "W20": "min over contiguous 20-epoch windows of the window median valid MAE",
            "T100": "median valid MAE over the last 100 epochs",
            "best_valid": "strict-< minimum of the per-epoch valid MAE",
        },
    })

    # ---- per-molecule table
    pred = {k: runs[k]["pred_best"] for k in names}
    ae = {k: np.abs(pred[k] - y) for k in names}
    audit_rows = load_audit_rows()
    fields = ["row_id", "y_raw", "n_atoms", "n_bonds", "n_rings", "cycle_rank",
              "max_ring_size", "r2_unseen_atom_frac"]
    fields += [f"pred_{k}" for k in names] + [f"signed_error_{k}" for k in names]
    fields += [f"abs_error_{k}" for k in names] + ["abs_error_delta_real_minus_null"]
    fields += ["dcc_P3B_abs_error", "dcc_worst100"]
    with open(ANALYSIS / "per_molecule_valid.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(audit_rows):
            out = {"row_id": i, "y_raw": r["y_raw"], "n_atoms": r["n_atoms"],
                   "n_bonds": r["n_bonds"], "n_rings": r["n_rings"],
                   "cycle_rank": r["cycle_rank"], "max_ring_size": r["max_ring_size"],
                   "r2_unseen_atom_frac": r["r2_unseen_atom_frac"]}
            for k in names:
                out[f"pred_{k}"] = f"{pred[k][i]:.9g}"
                out[f"signed_error_{k}"] = f"{pred[k][i] - y[i]:.9g}"
                out[f"abs_error_{k}"] = f"{ae[k][i]:.9g}"
            out["abs_error_delta_real_minus_null"] = f"{ae['real_s0'][i] - ae['null_s0'][i]:.9g}"
            out["dcc_P3B_abs_error"] = r["abs_error_P3B_s0"]
            out["dcc_worst100"] = r.get("P3B_worst100", "")
            w.writerow(out)

    # ---- paired contrasts
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(y), size=(RESAMPLES, len(y)))
    contrasts = {}
    pairs = [("real_s0", "null_s0"), ("real_s0", "hardperm_s0"), ("hardperm_s0", "null_s0")]
    for a, b in pairs:
        if a not in ae or b not in ae:
            continue
        delta = ae[a] - ae[b]
        contrasts[f"{a}_minus_{b}"] = bootstrap_delta(delta, idx)
        contrasts[f"{a}_minus_{b}"]["delta_definition"] = (
            "abs_error(first) - abs_error(second); negative means the first model is better")
    C.write_json(ANALYSIS / "paired_contrasts.json", {
        "resamples": RESAMPLES, "rng_seed": BOOT_SEED,
        "same_row_resamples_for_all_contrasts": True,
        "interval_role": ("descriptive fixed-checkpoint interval on the exposed official-valid 1000; "
                          "not cross-seed confirmation"),
        "contrasts": contrasts,
    })

    # ---- frozen groups
    masks = frozen_masks(audit_rows, y)
    groups = {name: group_row(m, y, pred) for name, m in masks.items()}
    dod = {}
    for a, b in (("real_s0", "null_s0"), ("real_s0", "hardperm_s0"),
                 ("hardperm_s0", "null_s0")):
        ku, kc = "signature_unseen_251", "signature_covered_749"
        key = f"delta_{a}_minus_{b}"
        if key in groups[ku] and key in groups[kc]:
            dod[f"{a}_minus_{b}"] = groups[ku][key] - groups[kc][key]
    dod["definition"] = ("mean paired delta inside unseen minus mean paired delta inside "
                         "covered; negative means the first model's advantage is larger on "
                         "unseen signatures")
    groups["unseen_minus_covered_delta_of_deltas"] = dod
    C.write_json(ANALYSIS / "groups.json", {
        "groups": groups,
        "masks_frozen_before_results": True,
        "mask_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "note": ("Fixed DCC-L worst-100 is a hard-case monitor; shares are descriptive only and "
                 "never used for selection."),
    })

    # ---- structure sensitivity (frozen weights, pairing-only intervention)
    scales = C.read_json(ANALYSIS / "scales.json")
    C.write_json(ANALYSIS / "structure_sensitivity.json",
                 structure_sensitivity(runs, scales))

    print(json.dumps({"runs": names,
                      "best_valid": {k: runs[k]["info"]["checkpoints"]["best"]["valid_mae"] for k in names},
                      "W20": {r["run"]: r["W20"] for r in rows_main},
                      "T100": {r["run"]: r["T100"] for r in rows_main}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
