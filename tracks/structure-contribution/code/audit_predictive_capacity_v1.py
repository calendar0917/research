"""Phase A — CPU-only predictive-capacity audit (round ``cscl-bjoc-intrinsic-
contribution-feasibility-v1``).

Question this script answers with *existing* frozen artifacts only:

* A1 — how far is the frozen BJOC predictor from the user's raw-y target band,
* A2 — where does the held-out error actually live (fit gap, seed randomness,
  ordinary molecules, low-y tail, structure, rare chemistry, systematic bias),
* A3 — is there any solid label-identifiability lower bound, or only a local
  duplicate-input witness?

Discipline
----------
* Nothing here trains a model or selects anything; it re-reads already-frozen
  internal-split predictions, curves and the official-train label cache.
* ``official valid`` / ``official test`` are never loaded.  The exposed
  historical official-test numbers are quoted from the repository notes as
  *context*, and are tagged as not usable for this round's decisions.
* The internal split used for the failure analysis is the already-exposed
  junction-regularisation split A (``cscl-bjoc-jdrop-v1-discovery``), not a
  split of this round.  Reading its labels cannot leak into this round because
  this round uses new salts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import run_bjoc_junction_regularization_v1 as jdrun  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import cscl_features as cf  # noqa: E402

OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_intrinsic_contribution_v1"
JDROP = REPO_ROOT / "tracks/structure-contribution/results/bjoc_junction_regularization_v1"

#: exposed historical official-test record (context only; never a decision input)
HISTORICAL_CONTEXT = {
    "source": "results/bjoc_predictor_terminal_v1/REPORT.md and notes; NOT read as tensors here",
    "bjoc_null_seed_mean_test_raw_mae": 0.092233,
    "bjoc_null_two_seed_ensemble_test_raw_mae": 0.085233,
    "bjoc_null_per_seed_test_raw_mae": [0.094563, 0.089903],
    "user_target_band": [0.06, 0.07],
    "gap_seed_mean_to_band": 0.092233 - 0.07,
    "gap_ensemble_to_band": 0.085233 - 0.07,
    "usage_rule": "background only; not used for any model decision in this round",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _sha_rows(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# A2 helpers
# ---------------------------------------------------------------------------


def _metrics(err: np.ndarray) -> dict[str, float]:
    return {
        "n": int(err.size),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
        "err_mass_share_top10": float(np.sort(err)[-10:].sum() / err.sum()) if err.size >= 10 else 1.0,
    }


def _load_split_A() -> dict[str, Any]:
    return json.loads((JDROP / "split_a.json").read_text(encoding="utf-8"))


def _curve_audit(train_json: Mapping[str, Any]) -> dict[str, Any]:
    curve = train_json["curve"]
    train = np.asarray([c["train_mae"] for c in curve], dtype=np.float64)
    mon = np.asarray([c["monitor_mae"] for c in curve], dtype=np.float64)
    last40_train = float(train[-1] - train[-41]) if len(train) > 40 else float(train[-1] - train[0])
    last40_mon = float(mon[-1] - mon[-41]) if len(mon) > 40 else float(mon[-1] - mon[0])
    last10_mon = float(mon[-1] - mon[-11]) if len(mon) > 10 else float(mon[-1] - mon[0])
    best_epoch = int(np.argmin(mon)) + 1 if np.all(np.isfinite(mon)) else None
    return {
        "epochs": len(curve),
        "train_mae_first": float(train[0]),
        "train_mae_last": float(train[-1]),
        "monitor_mae_first": float(mon[0]),
        "monitor_mae_last": float(mon[-1]),
        "monitor_best": float(np.nanmin(mon)) if np.any(np.isfinite(mon)) else None,
        "monitor_best_epoch": best_epoch,
        "train_change_last40": last40_train,
        "monitor_change_last40": last40_mon if np.isfinite(last40_mon) else None,
        "monitor_change_last10": last10_mon if np.isfinite(last10_mon) else None,
        "still_descending_train": bool(last40_train < 0),
        "wall_seconds": float(train_json.get("wall_seconds", float("nan"))),
    }


def _seed_failure_overlap(err_a: np.ndarray, err_b: np.ndarray, k: int = 100) -> dict[str, Any]:
    idx_a = set(np.argsort(-err_a)[:k].tolist())
    idx_b = set(np.argsort(-err_b)[:k].tolist())
    both_big = (err_a > 0.5) & (err_b > 0.5)
    any_big = (err_a > 0.5) | (err_b > 0.5)
    union = err_a + err_b
    top_union = np.argsort(-union)[:k]
    from scipy.stats import spearmanr

    rho = float(spearmanr(err_a, err_b).statistic)
    return {
        "top100_overlap_n": len(idx_a & idx_b),
        "top100_overlap_frac": len(idx_a & idx_b) / k,
        "both_seed_err_gt_0.5_n": int(both_big.sum()),
        "any_seed_err_gt_0.5_n": int(any_big.sum()),
        "both_over_any_frac": float(both_big.sum() / max(any_big.sum(), 1)),
        "error_spearman_between_seeds": rho,
        "top100_union_share_of_total_error": float(union[top_union].sum() / union.sum()),
    }


def _top10_removed_mask(err: np.ndarray) -> np.ndarray:
    order = np.argsort(-err)
    mask = np.ones(err.shape[0], dtype=bool)
    mask[order[:10]] = False
    return mask


def _seed_randomness_decomposition(err_s0: np.ndarray, err_s1: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Split per-molecule MAE into systematic (seed-mean) and seed-random parts.

    Squared-error shares are reported, plus the more outlier-robust mean
    absolute half-gap relative to the mean per-molecule error.
    """
    if mask is None:
        mask = np.ones_like(err_s0, dtype=bool)
    err_s0 = err_s0[mask]
    err_s1 = err_s1[mask]
    mean_err = 0.5 * (err_s0 + err_s1)
    half_gap = 0.5 * np.abs(err_s0 - err_s1)
    systematic = float(np.mean(mean_err ** 2))
    random_ = float(np.mean(half_gap ** 2))
    return {
        "n": int(mask.sum()),
        "mean_molecule_mae": float(np.mean(mean_err)),
        "mean_abs_half_seed_gap": float(np.mean(half_gap)),
        "mean_abs_half_gap_over_mean_mae": float(np.mean(half_gap) / max(np.mean(mean_err), 1e-12)),
        "mse_systematic_share": systematic / max(systematic + random_, 1e-12),
        "mse_seed_random_share": random_ / max(systematic + random_, 1e-12),
        "per_molecule_mae_s0": float(err_s0.mean()),
        "per_molecule_mae_s1": float(err_s1.mean()),
    }


def _quartile_table(values: np.ndarray, err: np.ndarray, name: str) -> list[dict[str, Any]]:
    q = np.quantile(values, [0.25, 0.5, 0.75])
    edges = [-np.inf, q[0], q[1], q[2], np.inf]
    out = []
    for i in range(4):
        mask = (values > edges[i]) & (values <= edges[i + 1])
        if i == 0:
            mask = values <= q[0]
        out.append(
            {
                "feature": name,
                "quartile": f"q{i + 1}",
                "range": [float(edges[i]), float(edges[i + 1])],
                "n": int(mask.sum()),
                "mae": float(err[mask].mean()) if mask.any() else float("nan"),
                "error_mass_share": float(err[mask].sum() / err.sum()) if mask.any() else float("nan"),
            }
        )
    return out


def _raw_signature_sets(objs: Sequence[sro.MolObjects], rows: Sequence[int], kind: str) -> list[set[bytes]]:
    out = []
    for r in rows:
        o = objs[int(r)]
        if kind == "atom":
            arr = o.atom_feat
        elif kind == "bond":
            arr = o.bond_feat
        else:
            arr = o.junc_feat
        out.append({np.ascontiguousarray(arr[i]).tobytes() for i in range(arr.shape[0])})
    return out


def _rare_coverage(objs: Sequence[sro.MolObjects], fit_rows: Sequence[int], eval_rows: Sequence[int], kind: str) -> np.ndarray:
    fit_signatures: set[bytes] = set()
    for s in _raw_signature_sets(objs, fit_rows, kind):
        fit_signatures |= s
    out = []
    for s in _raw_signature_sets(objs, eval_rows, kind):
        out.append(float(len(s - fit_signatures) / max(len(s), 1)))
    return np.asarray(out, dtype=np.float64)


def _bias_audit(pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    resid = pred - y
    ordinary = y > -3.0
    tail = ~ordinary

    def _entry(mask: np.ndarray) -> dict[str, float]:
        if not mask.any():
            return {"n": 0.0, "mean_residual": float("nan"), "median_residual": float("nan"), "mae": float("nan")}
        return {
            "n": float(mask.sum()),
            "mean_residual": float(resid[mask].mean()),
            "median_residual": float(np.median(resid[mask])),
            "mae": float(np.abs(resid[mask]).mean()),
        }

    slope = float(np.polyfit(y[ordinary], resid[ordinary], 1)[0]) if ordinary.sum() > 10 else float("nan")
    return {
        "all": _entry(np.ones_like(y, dtype=bool)),
        "ordinary_y_gt_-3": _entry(ordinary),
        "tail_y_le_-3": _entry(tail),
        "ordinary_residual_regression_slope_on_y": slope,
        "prediction_std": float(pred.std()),
        "target_std": float(y.std()),
    }


# ---------------------------------------------------------------------------
# A3 label identifiability
# ---------------------------------------------------------------------------


def _label_audit() -> dict[str, Any]:
    from run_structural_relation_composer_v1 import audit_labels

    ds = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(s, []).append(i)
    dup_groups = {s: rows for s, rows in groups.items() if len(rows) > 1}
    examples = []
    identical_inputs = 0
    for s, rows in dup_groups.items():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                ra, rb = rows[a], rows[b]
                da, db = ds[ra], ds[rb]
                same_x = bool(torch.equal(da.x, db.x))
                same_ei = bool(torch.equal(da.edge_index, db.edge_index))
                same_ea = bool(torch.equal(da.edge_attr, db.edge_attr))
                if same_x and same_ei and same_ea:
                    identical_inputs += 1
                    ea = float(da.y.reshape(-1)[0])
                    eb = float(db.y.reshape(-1)[0])
                    if not np.isclose(ea, eb):
                        examples.append(
                            {
                                "smiles": s,
                                "rows": [int(ra), int(rb)],
                                "y": [ea, eb],
                                "abs_spread": float(abs(ea - eb)),
                                "x_identical": True,
                                "edge_index_identical": True,
                                "edge_attr_identical": True,
                            }
                        )
    full = audit_labels(ds, smiles)
    return {
        "n_rows": int(full["n_rows"]),
        "n_duplicate_smiles_groups": len(dup_groups),
        "n_duplicate_pairs_with_identical_tensors": identical_inputs,
        "n_pairs_identical_inputs_different_y": len(examples),
        "examples": examples,
        "max_identical_input_y_gap": float(max([e["abs_spread"] for e in examples], default=0.0)),
        "wl_hash_groups": {
            "n_wl_groups": full["n_wl_groups"],
            "n_wl_multi_groups": full["n_wl_multi_groups"],
            "n_wl_groups_with_inconsistent_y": full["n_wl_groups_with_inconsistent_y"],
            "wl_inconsistent_examples": full["wl_inconsistent_examples"],
        },
        "strict_global_lower_bound": "IRREDUCIBLE_ERROR_NOT_ESTABLISHED",
        "reading": (
            "identical PyG inputs with a ~0.0197 target gap are a local label-ambiguity witness "
            "(most plausibly stereo information dropped in the committed canonical SMILES); the "
            "10k rows are otherwise y-consistent within exact/1-WL graph groups. This does NOT "
            "establish a ~0.09 irreducible floor and must not be quoted as one."
        ),
    }


# ---------------------------------------------------------------------------
# main audit
# ---------------------------------------------------------------------------


def run(log=print) -> dict[str, Any]:
    split = _load_split_A()
    rows = jdrun.read_split_rows("A")
    objs = jdrun.load_objects()
    smiles = jdrun.load_smiles()

    audit: dict[str, Any] = {
        "protocol": "cscl-bjoc-intrinsic-contribution-feasibility-v1",
        "phase": "A",
        "official_valid_or_test_loaded": False,
        "internal_split_used": {
            "name": "cscl-bjoc-jdrop-v1-discovery (split A of the completed junction-regularisation round)",
            "fit_sha256_rows": split["fit_sha256_rows"],
            "monitor_sha256_rows": split["monitor_sha256_rows"],
            "locked_sha256_rows": split["locked_sha256_rows"],
            "note": "already exposed in a completed round; independent of this round's new salts",
        },
        "a1_gap": dict(HISTORICAL_CONTEXT),
    }

    # ---- A2: curves, fit/held-out gap, seed randomness ---------------------
    curves = {}
    arm_metrics = {}
    for tag in ("R0_A_s0", "R0_A_s1"):
        tj = json.loads((JDROP / f"train_{tag}.json").read_text(encoding="utf-8"))
        curves[tag] = _curve_audit(tj)
    metrics_json = json.loads((JDROP / "split_a_metrics.json").read_text(encoding="utf-8"))
    for tag, entry in metrics_json["arms"].items():
        arm_metrics[tag] = {
            "fit_mae": entry["fit_mae"],
            "monitor": entry["monitor"],
            "locked": entry["locked"],
            "parameters": entry["parameters"],
        }
    audit["a2_failure_analysis"] = {
        "training_curves": curves,
        "arm_metrics": arm_metrics,
        "fit_vs_heldout": {
            "R0_A_s0": {
                "fit": arm_metrics["R0_A_s0"]["fit_mae"],
                "monitor": arm_metrics["R0_A_s0"]["monitor"]["mae"],
                "locked": arm_metrics["R0_A_s0"]["locked"]["mae"],
                "monitor_over_fit": arm_metrics["R0_A_s0"]["monitor"]["mae"] / arm_metrics["R0_A_s0"]["fit_mae"],
                "locked_over_fit": arm_metrics["R0_A_s0"]["locked"]["mae"] / arm_metrics["R0_A_s0"]["fit_mae"],
            },
            "R0_A_s1": {
                "fit": arm_metrics["R0_A_s1"]["fit_mae"],
                "monitor": arm_metrics["R0_A_s1"]["monitor"]["mae"],
                "locked": arm_metrics["R0_A_s1"]["locked"]["mae"],
                "monitor_over_fit": arm_metrics["R0_A_s1"]["monitor"]["mae"] / arm_metrics["R0_A_s1"]["fit_mae"],
                "locked_over_fit": arm_metrics["R0_A_s1"]["locked"]["mae"] / arm_metrics["R0_A_s1"]["fit_mae"],
            },
        },
    }

    preds = np.load(JDROP / "split_a_predictions.npz")
    mon_rows = [int(r) for r in preds["monitor_row"]]
    locked_rows = [int(r) for r in preds["locked_row"]]
    y_mon = jdrun.load_y_rows(mon_rows, subset_name="audit split-A monitor")
    y_locked = jdrun.load_y_rows(locked_rows, subset_name="audit split-A locked")
    err = {}
    for tag in ("R0_A_s0", "R0_A_s1"):
        err[f"{tag}_monitor"] = np.abs(preds[f"pred_{tag}_monitor"].astype(np.float64) - y_mon)
        err[f"{tag}_locked"] = np.abs(preds[f"pred_{tag}_locked"].astype(np.float64) - y_locked)

    audit["a2_failure_analysis"]["seed_structure"] = {
        "monitor": {
            "overlap_top100": _seed_failure_overlap(err["R0_A_s0_monitor"], err["R0_A_s1_monitor"]),
            "variance_decomposition": _seed_randomness_decomposition(err["R0_A_s0_monitor"], err["R0_A_s1_monitor"]),
            "variance_decomposition_ordinary": _seed_randomness_decomposition(
                err["R0_A_s0_monitor"], err["R0_A_s1_monitor"], y_mon > -3.0
            ),
            "variance_decomposition_ordinary_top10_removed": _seed_randomness_decomposition(
                err["R0_A_s0_monitor"],
                err["R0_A_s1_monitor"],
                _top10_removed_mask(err["R0_A_s0_monitor"] + err["R0_A_s1_monitor"]),
            ),
        },
        "locked": {
            "overlap_top100": _seed_failure_overlap(err["R0_A_s0_locked"], err["R0_A_s1_locked"]),
            "variance_decomposition": _seed_randomness_decomposition(err["R0_A_s0_locked"], err["R0_A_s1_locked"]),
            "variance_decomposition_ordinary": _seed_randomness_decomposition(
                err["R0_A_s0_locked"], err["R0_A_s1_locked"], y_locked > -3.0
            ),
            "variance_decomposition_top10_removed": _seed_randomness_decomposition(
                err["R0_A_s0_locked"],
                err["R0_A_s1_locked"],
                _top10_removed_mask(err["R0_A_s0_locked"] + err["R0_A_s1_locked"]),
            ),
        },
    }

    # per-split tail / subgroup analysis on the seed-mean error
    for split_name, y, e0, e1 in (
        ("monitor", y_mon, err["R0_A_s0_monitor"], err["R0_A_s1_monitor"]),
        ("locked", y_locked, err["R0_A_s0_locked"], err["R0_A_s1_locked"]),
    ):
        mean_err = 0.5 * (e0 + e1)
        ordinary = y > -3.0
        tail = ~ordinary
        audit["a2_failure_analysis"][f"{split_name}_error_structure"] = {
            "seed_mean_mae": float(mean_err.mean()),
            "ordinary_y_gt_-3": {
                "n": int(ordinary.sum()),
                "mae": float(mean_err[ordinary].mean()),
                "error_mass_share": float(mean_err[ordinary].sum() / mean_err.sum()),
            },
            "tail_y_le_-3": {
                "n": int(tail.sum()),
                "mae": float(mean_err[tail].mean()),
                "error_mass_share": float(mean_err[tail].sum() / mean_err.sum()),
            },
            "top10_removed_mae": float(np.sort(mean_err)[:-10].mean()),
            "top10_error_mass_share": float(np.sort(mean_err)[-10:].sum() / mean_err.sum()),
        }

    # structural quartiles on the locked split (seed mean)
    ss_locked = jdrun.structural_stats(objs, locked_rows)
    mean_locked = 0.5 * (err["R0_A_s0_locked"] + err["R0_A_s1_locked"])
    quartiles = []
    for name in ("n_atoms", "n_junctions", "cyclomatic", "diameter"):
        quartiles.extend(_quartile_table(ss_locked[name], mean_locked, name))
    audit["a2_failure_analysis"]["locked_structural_quartiles"] = quartiles

    # rare chemistry coverage: fraction of eval object signatures unseen in fit
    coverage = {}
    for kind in ("atom", "bond", "junction"):
        cov = _rare_coverage(objs, rows["fit"], locked_rows, kind)
        coverage[kind] = {
            "mean_unseen_fraction_locked": float(cov.mean()),
            "median_unseen_fraction_locked": float(np.median(cov)),
            "quartiles": _quartile_table(cov, mean_locked, f"{kind}_unseen_fraction"),
        }
    audit["a2_failure_analysis"]["rare_chemistry_coverage"] = coverage

    # systematic bias on monitor (genuinely unseen) and locked
    audit["a2_failure_analysis"]["systematic_bias"] = {
        "monitor_seed0": _bias_audit(preds["pred_R0_A_s0_monitor"].astype(np.float64), y_mon),
        "monitor_seed1": _bias_audit(preds["pred_R0_A_s1_monitor"].astype(np.float64), y_mon),
        "locked_seed0": _bias_audit(preds["pred_R0_A_s0_locked"].astype(np.float64), y_locked),
        "locked_seed1": _bias_audit(preds["pred_R0_A_s1_locked"].astype(np.float64), y_locked),
    }

    # the extreme locked row
    order = np.argsort(-mean_locked)
    ext_positions = [int(p) for p in order[:5]]
    audit["a2_failure_analysis"]["extreme_locked_rows"] = [
        {
            "row": int(locked_rows[p]),
            "y": float(y_locked[p]),
            "err_s0": float(err["R0_A_s0_locked"][p]),
            "err_s1": float(err["R0_A_s1_locked"][p]),
            "n_atoms": int(objs[locked_rows[p]].n_atoms),
            "n_junctions": int(objs[locked_rows[p]].junc_index.shape[0]),
            "smiles": smiles[locked_rows[p]],
        }
        for p in ext_positions
    ]

    # ---- A3 label identifiability ------------------------------------------
    audit["a3_label_identifiability"] = _label_audit()

    # ---- conclusions table --------------------------------------------------
    audit["conclusion_table"] = [
        {
            "question": "ordinary molecules",
            "observed": audit["a2_failure_analysis"]["locked_error_structure"]["ordinary_y_gt_-3"],
            "interpretation": "the held-out floor on ordinary molecules is already ~0.09-0.13; "
            "the target band needs a broad improvement, not tail repair",
            "next_discriminating_experiment": "matched full-10k control vs BJOC under the same evaluation path",
        },
        {
            "question": "low-y tail",
            "observed": audit["a2_failure_analysis"]["locked_error_structure"]["tail_y_le_-3"],
            "interpretation": "a small fraction of rows carries a disproportionate error mass; "
            "any new method must be judged with and without these rows",
            "next_discriminating_experiment": "report tail and top-10-removed deltas in every future comparison",
        },
        {
            "question": "seed sensitivity",
            "observed": audit["a2_failure_analysis"]["seed_structure"],
            "interpretation": "molecule-level errors are only weakly correlated between seeds; "
            "a large share of per-molecule error is training randomness, so single-seed deltas "
            "smaller than the seed spread are uninformative",
            "next_discriminating_experiment": "always pair seeds; never select on one seed",
        },
        {
            "question": "structural representation",
            "observed": audit["a2_failure_analysis"]["locked_structural_quartiles"],
            "interpretation": "error varies with size/ring complexity but the typed-path/ring-block "
            "round already showed richer inputs fit better yet generalise worse",
            "next_discriminating_experiment": "a capacity/regularisation-controlled representation arm, pre-registered",
        },
        {
            "question": "data and labels",
            "observed": audit["a3_label_identifiability"],
            "interpretation": "one identical-input pair carries a ~0.0197 label gap; no global "
            "irreducible floor is established",
            "next_discriminating_experiment": "strict duplicate-group label audit before quoting any floor",
        },
    ]
    return audit


def main() -> int:
    audit = run()
    write_json(OUT / "predictive_capacity_audit.json", audit)
    print(f"[phase-a] wrote {OUT / 'predictive_capacity_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
