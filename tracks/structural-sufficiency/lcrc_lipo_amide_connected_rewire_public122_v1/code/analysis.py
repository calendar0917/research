"""Local analysis: selective-response contrasts ``J``/``D``/``B``, bootstrap, tables.

All quantities are in raw logD.  For an original molecule ``m``, target instance
``j`` and the same donor ``d``:

``p0 = f(O)``, ``pT = f(T)``, ``pK = f(K)`` - the frozen model's prediction for
the original graph and for the two rewired graphs (same donor).

``J_m  = mean_j mean_d (|pT-p0| - |pK-p0|)``   extra *prediction movement* when the
                                              amide pattern is destroyed instead of kept
``D_m  = mean_j mean_d (|pT-y| - |pK-y|)``     extra deviation from the original label
``B_m  = mean_j mean_d (|pT-y| - |p0-y|)``     deviation of the rewired graph vs the original graph

``J`` measures response only.  ``D`` and ``B`` are computational fidelity contrasts
against the **original molecule's** experimental value; the rewired graph is not a
measured compound and has no experimental label.

Weights: molecules equal; instances equal inside a molecule; donors equal inside
an instance.  Losses/movements are formed per view and then averaged (predictions
are never averaged before the loss).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import prepare as P

FULL = "FULL"
G_FIXED = "G_FIXED"

MAIN_POP = "main"
SUPP_ALL = "struct_chem_all"
SUPP_MATCHED = "matched_supp"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def verify_protocol_inputs(analysis_dir: Path) -> dict[str, Any]:
    manifest = C.read_json(analysis_dir / "inputs_manifest.json")
    recorded = (manifest.get("artifacts") or {}).get("files") or {}
    out: dict[str, Any] = {"recorded": recorded, "observed": {}, "pass": True}
    for name, want in recorded.items():
        path = analysis_dir / name
        got = C.sha256_file(path) if path.exists() else "missing"
        out["observed"][name] = got
        if got != want:
            out["pass"] = False
    freeze = C.read_json(analysis_dir / "freeze.json")
    out["freeze_recorded"] = freeze.get("files") or {}
    for name, want in (freeze.get("files") or {}).items():
        path = analysis_dir / name
        got = C.sha256_file(path) if path.exists() else "missing"
        if got != want:
            out["pass"] = False
            out["observed"][f"freeze::{name}"] = got
    return out


def inference_gate_summary(checks: Mapping[str, Any]) -> dict[str, Any]:
    baseline = checks.get("baseline", {}) or {}
    state = checks.get("state_identity", {}) or {}
    views = checks.get("views", {}) or {}
    return {
        "pass": bool(checks.get("pass")),
        "freeze_pass": bool((checks.get("freeze") or {}).get("pass")),
        "view_gates": checks.get("view_gates") or {s: bool(b.get("pass")) for s, b in views.items()},
        "view_gates_detail": {
            s: {
                "n_views": b.get("n_views"),
                "n_cut_views_checked": b.get("n_cut_views_checked"),
                "structural_failures": b.get("structural_failures"),
                "af51_failures": b.get("af51_failures"),
                "af51_max_abs_atom_feat_diff": b.get("af51_max_abs_atom_feat_diff"),
                "atom_scalar_max_abs_diff": b.get("atom_scalar_max_abs_diff"),
                "input_count_failures": b.get("input_count_failures"),
                "n_components_not_one": b.get("n_components_not_one"),
                "n_views_out_of_train_range": b.get("n_views_out_of_train_range"),
                "o_view_max_abs_diff_full_vs_gfixed": b.get("o_view_max_abs_diff_full_vs_gfixed"),
            }
            for s, b in views.items()
        },
        "baseline_pass": bool(baseline.get("pass")),
        "baseline_tolerance": baseline.get("tolerance"),
        "baseline_per_split": {
            s: {
                "n": d.get("n"),
                "rmse_raw": d.get("rmse_raw"),
                "rmse_diff": d.get("rmse_diff"),
                "mae_raw": d.get("mae_raw"),
                "mae_diff": d.get("mae_diff"),
                "max_abs_pred_diff_vs_previous_round": d.get("max_abs_pred_diff_vs_previous_round"),
            }
            for s, d in (baseline.get("per_split") or {}).items()
        },
        "state_hash_before": state.get("state_hash_before"),
        "state_hash_after": state.get("state_hash_after"),
        "norm_fingerprint_unchanged": state.get("norm_fingerprint_before")
        == state.get("norm_fingerprint_after"),
        "optimizer_created": state.get("optimizer_created"),
        "backward_calls": state.get("backward_calls"),
        "model_training_steps": state.get("model_training_steps"),
    }


def _to_int(row: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    out = dict(row)
    for k in keys:
        if k in out and str(out[k]).strip() != "":
            out[k] = int(float(out[k]))
    return out


def load_catalogs(
    analysis_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = [
        _to_int(
            r,
            (
                "row_id", "c", "o", "n", "acyl_c", "matched", "donor_u", "donor_v", "donor_bond_id",
                "donor_degree_u", "donor_degree_v", "donor_atom_h_u", "donor_atom_h_v",
                "donor_ring_u", "donor_ring_v", "donor_distance_to_c",
                "legal_T_crossings", "legal_K_crossings", "structural_ok", "chem_ok",
                "range_supported", "out_of_range_dims", "pop_struct_chem", "pop_main", "pop_matched_supp",
            ),
        )
        for r in P.read_csv(analysis_dir / "donor_catalog.csv")
    ]
    instances = [
        _to_int(
            r,
            (
                "row_id", "c", "o", "n", "acyl_c", "degree_n", "degree_acyl", "ring_n", "ring_acyl",
                "matched", "n_side", "acyl_side", "side_diff", "component_size", "bond_cn",
                "bond_c_acyl", "bond_c_o", "n_donors", "n_structural_ok_pairs", "n_chem_ok_pairs",
                "n_range_supported_pairs", "in_matched_subset",
            ),
        )
        for r in P.read_csv(analysis_dir / "motif_catalog.csv")
    ]
    views = P.read_csv(analysis_dir / "view_manifest.csv")
    return pairs, instances, views


def load_predictions(analysis_dir: Path) -> dict[tuple[str, str], float]:
    rows = P.read_csv(analysis_dir / "per_view_predictions.csv")
    return {(str(r["view_key"]), str(r["scope"])): float(r["pred_raw"]) for r in rows}


# ---------------------------------------------------------------------------
# per-pair panel
# ---------------------------------------------------------------------------


def population_pairs(pairs: Sequence[Mapping[str, Any]], split: str, population: str) -> list[dict[str, Any]]:
    if population == MAIN_POP:
        key = "pop_main"
    elif population == SUPP_ALL:
        key = "pop_struct_chem"
    elif population == SUPP_MATCHED:
        key = "pop_matched_supp"
    else:  # pragma: no cover
        raise KeyError(population)
    return [dict(p) for p in pairs if p["split"] == split and int(p[key]) == 1]


def pair_panel(
    *,
    pairs: Sequence[Mapping[str, Any]],
    preds: Mapping[tuple[str, str], float],
    y: Mapping[int, float],
    scope: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pr in pairs:
        rid = int(pr["row_id"])
        p0 = float(preds[(f"{rid}:O", FULL)])
        pT = float(preds[(str(pr["t_view_key"]), scope)])
        pK = float(preds[(str(pr["k_view_key"]), scope)])
        yy = float(y[rid])
        out.append(
            {
                "pair_id": pr["pair_id"],
                "row_id": rid,
                "split": pr["split"],
                "instance_id": pr["instance_id"],
                "scaffold_key": pr["scaffold_key"],
                "matched": int(pr["matched"]),
                "c": int(pr["c"]),
                "o": int(pr["o"]),
                "n": int(pr["n"]),
                "acyl_c": int(pr["acyl_c"]),
                "donor_u": int(pr["donor_u"]),
                "donor_v": int(pr["donor_v"]),
                "donor_distance_to_c": int(pr["donor_distance_to_c"]),
                "t_remove": pr["t_remove"],
                "t_add": pr["t_add"],
                "k_remove": pr["k_remove"],
                "k_add": pr["k_add"],
                "range_supported": int(pr["range_supported"]),
                "pred_O": p0,
                "pred_T": pT,
                "pred_K": pK,
                "y": yy,
                "loss_O": abs(p0 - yy),
                "loss_T": abs(pT - yy),
                "loss_K": abs(pK - yy),
                "move_T": abs(pT - p0),
                "move_K": abs(pK - p0),
                "move_signed_T": pT - p0,
                "move_signed_K": pK - p0,
                "J": abs(pT - p0) - abs(pK - p0),
                "D": abs(pT - yy) - abs(pK - yy),
                "B": abs(pT - yy) - abs(p0 - yy),
            }
        )
    return out


def molecule_panel(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Molecule-equal, instance-equal, donor-equal aggregation of the pair panel."""
    by_mol: dict[int, dict[str, list[float]]] = {}
    by_mol_inst: dict[int, dict[str, dict[str, list[float]]]] = {}
    meta: dict[int, dict[str, Any]] = {}
    keys = ("J", "D", "B", "loss_O", "loss_T", "loss_K", "move_T", "move_K", "move_signed_T", "move_signed_K")
    for r in pair_rows:
        rid = int(r["row_id"])
        iid = str(r["instance_id"])
        for k in keys:
            by_mol_inst.setdefault(rid, {}).setdefault(iid, {}).setdefault(k, []).append(float(r[k]))
        meta[rid] = {
            "split": r["split"],
            "scaffold_key": r["scaffold_key"],
            "y": float(r["y"]),
            "pred_O": float(r["pred_O"]),
        }
    rows: list[int] = []
    out: dict[str, list[float]] = {k: [] for k in keys}
    n_inst: list[int] = []
    n_don: list[int] = []
    for rid in sorted(by_mol_inst):
        inst_map = by_mol_inst[rid]
        per_inst: dict[str, dict[str, float]] = {}
        donors = 0
        for iid, blocks in inst_map.items():
            per_inst[iid] = {k: float(np.mean(v)) for k, v in blocks.items()}
            donors += len(blocks["J"])
        rows.append(rid)
        n_inst.append(len(per_inst))
        n_don.append(donors)
        for k in out:
            out[k].append(float(np.mean([per_inst[i][k] for i in per_inst])))
    panel = {
        "rows": np.asarray(rows, dtype=np.int64),
        "n_instances": np.asarray(n_inst, dtype=np.int64),
        "n_donors": np.asarray(n_don, dtype=np.int64),
        "scaffold": {rid: meta[rid]["scaffold_key"] for rid in rows},
        "y": np.asarray([meta[rid]["y"] for rid in rows], dtype=np.float64),
        "pred_O": np.asarray([meta[rid]["pred_O"] for rid in rows], dtype=np.float64),
    }
    for k, v in out.items():
        panel[k] = np.asarray(v, dtype=np.float64)
    return panel


# ---------------------------------------------------------------------------
# point estimates, bootstrap, influence
# ---------------------------------------------------------------------------


def _mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def point_estimates(panel: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "J": _mean(panel["J"]),
        "D": _mean(panel["D"]),
        "B": _mean(panel["B"]),
        "loss_O": _mean(panel["loss_O"]),
        "loss_T": _mean(panel["loss_T"]),
        "loss_K": _mean(panel["loss_K"]),
        "move_T": _mean(panel["move_T"]),
        "move_K": _mean(panel["move_K"]),
        "move_signed_T": _mean(panel["move_signed_T"]),
        "move_signed_K": _mean(panel["move_signed_K"]),
        "frac_molecules_J_positive": float(np.mean(panel["J"] > 0)) if panel["rows"].size else float("nan"),
        "frac_molecules_D_positive": float(np.mean(panel["D"] > 0)) if panel["rows"].size else float("nan"),
        "frac_molecules_B_positive": float(np.mean(panel["B"] > 0)) if panel["rows"].size else float("nan"),
        "median_D": float(np.median(panel["D"])) if panel["rows"].size else float("nan"),
        "median_J": float(np.median(panel["J"])) if panel["rows"].size else float("nan"),
        "n_molecules": int(panel["rows"].size),
        "n_instances": int(panel["n_instances"].sum()),
        "n_donors": int(panel["n_donors"].sum()),
    }


def cluster_weights(
    rows: np.ndarray,
    scaffold: Mapping[int, str],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    keys = [scaffold[int(r)] for r in rows.tolist()]
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    cluster_idx = np.asarray([uniq[k] for k in keys], dtype=np.int64)
    k_total = len(uniq)
    draws = rng.integers(0, k_total, size=(int(replicates), k_total))
    counts = np.zeros((int(replicates), k_total), dtype=np.int64)
    for b in range(int(replicates)):
        counts[b] = np.bincount(draws[b], minlength=k_total)
    sizes = np.bincount(cluster_idx, minlength=k_total)
    return {
        "weights": counts[:, cluster_idx].astype(np.float64),
        "draw_counts": counts,
        "n_clusters": int(k_total),
        "cluster_index": cluster_idx,
        "cluster_sizes": sizes,
        "max_cluster_share": float(sizes.max() / sizes.sum()) if sizes.size else 0.0,
        "max_cluster_key": sorted(uniq)[int(np.argmax(sizes))] if sizes.size else "",
    }


def weighted_means(weights: np.ndarray, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("J", "D", "B", "loss_O", "loss_T", "loss_K", "move_T", "move_K"):
        v = np.asarray(values[key], dtype=np.float64)
        ok = np.isfinite(v)
        w = weights * ok.reshape(1, -1)
        num = w @ np.nan_to_num(v, nan=0.0)
        den = w.sum(axis=1)
        out[key] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


def percentile_ci(reps: np.ndarray, level: float = 95.0) -> list[float]:
    reps = np.asarray(reps, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return [float("nan"), float("nan")]
    lo = (100.0 - level) / 2.0
    return [float(np.percentile(reps, lo)), float(np.percentile(reps, 100.0 - lo))]


def influence_checks(panel: Mapping[str, Any], ranking: str = "D") -> dict[str, Any]:
    rows = np.asarray(panel["rows"], dtype=np.int64)
    ref = np.asarray(panel[ranking], dtype=np.float64)
    n = rows.size
    order = sorted(range(n), key=lambda i: (-abs(float(ref[i])), int(rows[i])))
    worst = order[0]
    k = int(math.ceil(C.INFLUENCE_FRACTION * n))
    drop_rows = sorted(int(rows[i]) for i in order[:k])
    keep_single = np.ones(n, dtype=bool)
    keep_single[worst] = False
    keep_block = ~np.isin(rows, drop_rows)
    return {
        "ranking_metric": ranking,
        "n_molecules": int(n),
        "n_dropped_block": k,
        "max_abs_row": int(rows[worst]),
        "max_abs_value": float(ref[worst]),
        "dropped_block_rows": drop_rows,
        "without_max_row": {m: _mean(panel[m][keep_single]) for m in ("J", "D", "B")},
        "without_top_block": {m: _mean(panel[m][keep_block]) for m in ("J", "D", "B")},
    }


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def cases(
    *,
    main_panel_full: Mapping[str, Any],
    main_panels_by_scope: Mapping[str, Mapping[str, Any]],
    pair_rows_by_scope: Mapping[str, list[dict[str, Any]]],
    instances: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = np.asarray(main_panel_full["rows"], dtype=np.int64)
    D = np.asarray(main_panel_full["D"], dtype=np.float64)
    med = float(np.median(D))
    order = np.argsort(D, kind="stable")
    picks = [
        ("median", int(np.argmin(np.abs(D - med)))),
        ("max_D", int(np.argmax(D))),
        ("min_D", int(np.argmin(D))),
    ]
    inst_by_row: dict[int, list[Mapping[str, Any]]] = {}
    for ins in instances:
        inst_by_row.setdefault(int(ins["row_id"]), []).append(ins)
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for tag, i in picks:
        rid = int(rows[i])
        if rid in seen:
            continue
        seen.add(rid)
        entry: dict[str, Any] = {
            "tag": tag,
            "row_id": rid,
            "y": float(main_panel_full["y"][i]),
            "pred_O": float(main_panel_full["pred_O"][i]),
            "D_rank": int(np.where(order == i)[0][0]) + 1,
            "n_molecules": int(rows.size),
            "ranking_scope": FULL,
            "D": float(main_panel_full["D"][i]),
            "J": float(main_panel_full["J"][i]),
            "B": float(main_panel_full["B"][i]),
            "scaffold_key": main_panel_full["scaffold"][rid],
            "scopes": {},
            "instances": [],
        }
        for scope, panel in main_panels_by_scope.items():
            entry["scopes"][scope] = {
                "D": float(panel["D"][i]),
                "J": float(panel["J"][i]),
                "B": float(panel["B"][i]),
                "loss_O": float(panel["loss_O"][i]),
                "loss_T": float(panel["loss_T"][i]),
                "loss_K": float(panel["loss_K"][i]),
                "move_T": float(panel["move_T"][i]),
                "move_K": float(panel["move_K"][i]),
            }
        for ins in sorted(inst_by_row.get(rid, []), key=lambda r: int(r["c"])):
            inst_entry = {
                "instance_id": ins["instance_id"],
                "c": int(ins["c"]),
                "o": int(ins["o"]),
                "n": int(ins["n"]),
                "acyl_c": int(ins["acyl_c"]),
                "matched": int(ins["matched"]),
                "n_donors_in_main": int(ins.get("n_range_supported_pairs", 0) or 0),
                "donors": [],
            }
            for scope, prows in pair_rows_by_scope.items():
                for pr in prows:
                    if pr["instance_id"] != ins["instance_id"]:
                        continue
                    inst_entry["donors"].append(
                        {
                            "scope": scope,
                            "pair_id": pr["pair_id"],
                            "donor_u": pr["donor_u"],
                            "donor_v": pr["donor_v"],
                            "t_remove": pr["t_remove"],
                            "t_add": pr["t_add"],
                            "k_remove": pr["k_remove"],
                            "k_add": pr["k_add"],
                            "pred_O": pr["pred_O"],
                            "pred_T": pr["pred_T"],
                            "pred_K": pr["pred_K"],
                            "J": pr["J"],
                            "D": pr["D"],
                            "B": pr["B"],
                        }
                    )
            entry["instances"].append(inst_entry)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> int:
    analysis_dir = C.ANALYSIS
    pairs, instances, view_rows = load_catalogs(analysis_dir)
    preds = load_predictions(analysis_dir)
    prev = C.load_previous_inputs()
    y = prev["y"]

    runtime = C.read_json(analysis_dir / "runtime.json") if (analysis_dir / "runtime.json").exists() else {}
    infer_checks = (
        C.read_json(analysis_dir / "infer_checks.json") if (analysis_dir / "infer_checks.json").exists() else {}
    )
    manifest = C.read_json(analysis_dir / "inputs_manifest.json")

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    results: list[dict[str, Any]] = []
    uncertainty: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "protocol_inputs": verify_protocol_inputs(analysis_dir),
        "inference_gates": inference_gate_summary(infer_checks),
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "bootstrap_replicates": C.BOOTSTRAP_REPLICATES,
        "cluster_definition": "Bemis-Murcko scaffold of the original molecule (canonical, non-isomeric; "
        "empty scaffold -> molecule's own non-isomeric canonical SMILES), reused verbatim from the "
        "previous round's molecule catalog",
        "populations": {},
    }
    panels: dict[tuple[str, str, str], dict[str, Any]] = {}
    per_pair_out: list[dict[str, Any]] = []
    for split in ("valid", "test"):
        for population in (MAIN_POP, SUPP_ALL, SUPP_MATCHED):
            plist = population_pairs(pairs, split, population)
            if not plist:
                continue
            scope_panels: dict[str, dict[str, Any]] = {}
            for scope in C.SCOPES:
                prows = pair_panel(pairs=plist, preds=preds, y=y, scope=scope)
                per_pair_out.extend(
                    [
                        {
                            "population": population,
                            "scope": scope,
                            **{k: v for k, v in r.items() if k not in ("range_supported",)},
                        }
                        for r in prows
                    ]
                )
                scope_panels[scope] = molecule_panel(prows)
            rows = scope_panels[FULL]["rows"]
            cw = cluster_weights(rows, scope_panels[FULL]["scaffold"], rng=rng, replicates=C.BOOTSTRAP_REPLICATES)
            W = cw["weights"]
            uncertainties: dict[str, Any] = {}
            for scope, panel in scope_panels.items():
                panels[(population, split, scope)] = panel
                pe = point_estimates(panel)
                reps = weighted_means(W, panel)
                for key in ("J", "D", "B"):
                    lo, hi = percentile_ci(reps[key])
                    uncertainties[f"{scope}:{key}"] = {
                        "point": pe[key],
                        "ci95": [lo, hi],
                        "n_replicates_finite": int(np.isfinite(reps[key]).sum()),
                    }
                for key in ("loss_O", "loss_T", "loss_K", "move_T", "move_K"):
                    lo, hi = percentile_ci(reps[key])
                    uncertainties[f"{scope}:{key}"] = {"point": pe[key], "ci95": [lo, hi]}
                for metric, value in pe.items():
                    results.append(
                        {
                            "population": population,
                            "split": split,
                            "scope": scope,
                            "metric": metric,
                            "value": value,
                            "n_molecules": pe["n_molecules"],
                            "n_instances": pe["n_instances"],
                            "n_donors": pe["n_donors"],
                        }
                    )
                for key in ("J", "D", "B"):
                    lo, hi = percentile_ci(reps[key])
                    for suffix, val in (("ci95_low", lo), ("ci95_high", hi)):
                        results.append(
                            {
                                "population": population,
                                "split": split,
                                "scope": scope,
                                "metric": f"{key}_{suffix}",
                                "value": val,
                                "n_molecules": pe["n_molecules"],
                                "n_instances": pe["n_instances"],
                                "n_donors": pe["n_donors"],
                            }
                        )
            uncertainty["populations"][f"{population}/{split}"] = {
                "n_molecules": int(rows.size),
                "n_instances": int(scope_panels[FULL]["n_instances"].sum()),
                "n_donors": int(scope_panels[FULL]["n_donors"].sum()),
                "n_clusters": int(cw["n_clusters"]),
                "max_cluster_share": float(cw["max_cluster_share"]),
                "max_cluster_key": cw["max_cluster_key"],
                "scopes": uncertainties,
            }

    # ---- influence checks -------------------------------------------------
    for scope in C.SCOPES:
        key = (MAIN_POP, "test", scope)
        if key in panels:
            uncertainty[f"influence_main_test_{scope}"] = influence_checks(panels[key], ranking="D")
    for scope in C.SCOPES:
        key = (SUPP_MATCHED, "test", scope)
        if key in panels:
            uncertainty[f"influence_matched_supp_test_{scope}"] = influence_checks(panels[key], ranking="D")

    # ---- coverage / validity summary -------------------------------------
    uncertainty["coverage"] = {
        "structural_counts": manifest.get("structural_counts"),
        "matched_subset_counts": manifest.get("matched_subset_counts"),
        "population_support": manifest.get("population_support"),
        "selection": manifest.get("selection"),
        "n_components_census": (manifest.get("data") or {}).get("n_components_census"),
        "out_of_range_rows": len(P.read_csv(analysis_dir / "view_out_of_range.csv"))
        if (analysis_dir / "view_out_of_range.csv").exists()
        else None,
    }

    # ---- cases -----------------------------------------------------------
    main_pr = {
        scope: pair_panel(
            pairs=population_pairs(pairs, "test", MAIN_POP), preds=preds, y=y, scope=scope
        )
        for scope in C.SCOPES
    }
    main_panels = {scope: molecule_panel(main_pr[scope]) for scope in C.SCOPES}
    case_rows = cases(
        main_panel_full=main_panels[FULL],
        main_panels_by_scope=main_panels,
        pair_rows_by_scope=main_pr,
        instances=instances,
    )
    C.write_json(
        analysis_dir / "cases.json",
        {
            "main_population": "main/test",
            "ranking_scope": FULL,
            "note": "the rewired graphs are not measured compounds; y is the original molecule's label and "
            "is used only as a computational reference, never as the new graph's experimental label",
            "cases": case_rows,
        },
    )

    # ---- per-molecule summary -------------------------------------------
    mol_rows: list[dict[str, Any]] = []
    for split in ("valid", "test"):
        for population in (MAIN_POP, SUPP_MATCHED):
            if (population, split, FULL) not in panels:
                continue
            pfull = panels[(population, split, FULL)]
            pgf = panels[(population, split, G_FIXED)]
            for i, rid in enumerate(np.asarray(pfull["rows"]).tolist()):
                mol_rows.append(
                    {
                        "population": population,
                        "split": split,
                        "row_id": int(rid),
                        "scaffold_key": pfull["scaffold"][int(rid)],
                        "y": float(pfull["y"][i]),
                        "pred_O": float(pfull["pred_O"][i]),
                        "n_instances": int(pfull["n_instances"][i]),
                        "n_donors": int(pfull["n_donors"][i]),
                        "FULL_J": float(pfull["J"][i]),
                        "FULL_D": float(pfull["D"][i]),
                        "FULL_B": float(pfull["B"][i]),
                        "FULL_loss_O": float(pfull["loss_O"][i]),
                        "FULL_loss_T": float(pfull["loss_T"][i]),
                        "FULL_loss_K": float(pfull["loss_K"][i]),
                        "FULL_move_T": float(pfull["move_T"][i]),
                        "FULL_move_K": float(pfull["move_K"][i]),
                        "G_FIXED_J": float(pgf["J"][i]),
                        "G_FIXED_D": float(pgf["D"][i]),
                        "G_FIXED_B": float(pgf["B"][i]),
                        "G_FIXED_loss_O": float(pgf["loss_O"][i]),
                        "G_FIXED_loss_T": float(pgf["loss_T"][i]),
                        "G_FIXED_loss_K": float(pgf["loss_K"][i]),
                        "G_FIXED_move_T": float(pgf["move_T"][i]),
                        "G_FIXED_move_K": float(pgf["move_K"][i]),
                    }
                )
    P.write_csv(
        analysis_dir / "molecule_summary.csv",
        mol_rows,
        [
            "population", "split", "row_id", "scaffold_key", "y", "pred_O", "n_instances", "n_donors",
            "FULL_J", "FULL_D", "FULL_B", "FULL_loss_O", "FULL_loss_T", "FULL_loss_K",
            "FULL_move_T", "FULL_move_K",
            "G_FIXED_J", "G_FIXED_D", "G_FIXED_B", "G_FIXED_loss_O", "G_FIXED_loss_T", "G_FIXED_loss_K",
            "G_FIXED_move_T", "G_FIXED_move_K",
        ],
    )
    pair_columns = [
        "population", "scope", "pair_id", "row_id", "split", "instance_id", "scaffold_key", "matched",
        "c", "o", "n", "acyl_c", "donor_u", "donor_v", "donor_distance_to_c",
        "t_remove", "t_add", "k_remove", "k_add",
        "pred_O", "pred_T", "pred_K", "y", "loss_O", "loss_T", "loss_K",
        "move_T", "move_K", "move_signed_T", "move_signed_K", "J", "D", "B",
    ]
    P.write_csv(analysis_dir / "per_pair_predictions.csv", per_pair_out, pair_columns)

    columns = ["population", "split", "scope", "metric", "value", "n_molecules", "n_instances", "n_donors"]
    P.write_csv(analysis_dir / "results.csv", results, columns)
    C.write_json(analysis_dir / "uncertainty.json", uncertainty)

    # ---- tables ----------------------------------------------------------
    lines = [
        "# 主表 — 主人群 (RANGE_SUPPORTED 成对改线, 105/121 分子)",
        "",
        "J = mean(|pT-p0| - |pK-p0|); D = mean(|pT-y|-|pK-y|); B = mean(|pT-y|-|p0-y|); 单位原始 logD。",
        "y 是 **原分子** 的实验标签, 只作计算参照; 改线图不是被测化合物。",
        "CI 为 scaffold cluster bootstrap 95% percentile (1000 次, seed 20261114)。",
        "",
        "| split | scope | n_mol | n_inst | n_donor | n_cluster | MAE_O | MAE_T | MAE_K | move_T | move_K | J [95% CI] | D [95% CI] | B [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split in ("valid", "test"):
        for scope in (FULL, G_FIXED):
            key = (MAIN_POP, split, scope)
            if key not in panels:
                continue
            pe = point_estimates(panels[key])
            u = uncertainty["populations"][f"{MAIN_POP}/{split}"]
            us = u["scopes"]

            def ci(name: str) -> str:
                d = us[f"{scope}:{name}"]
                return f"{_fmt(d['point'])} [{_fmt(d['ci95'][0])}, {_fmt(d['ci95'][1])}]"

            lines.append(
                f"| {split} | {scope} | {pe['n_molecules']} | {pe['n_instances']} | {pe['n_donors']} | "
                f"{u['n_clusters']} | {_fmt(pe['loss_O'])} | {_fmt(pe['loss_T'])} | {_fmt(pe['loss_K'])} | "
                f"{_fmt(pe['move_T'])} | {_fmt(pe['move_K'])} | {ci('J')} | {ci('D')} | {ci('B')} |"
            )
    (analysis_dir / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sup = [
        "# 补充表 — 全部结构/化学合法 donor 人群 与 前轮 MATCHED 子集",
        "",
        "| population | split | scope | n_mol | n_inst | n_donor | n_cluster | max_cluster | MAE_O | MAE_T | MAE_K | J | D | B |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for population in (SUPP_ALL, SUPP_MATCHED):
        for split in ("valid", "test"):
            for scope in (FULL, G_FIXED):
                key = (population, split, scope)
                if key not in panels:
                    continue
                pe = point_estimates(panels[key])
                u = uncertainty["populations"][f"{population}/{split}"]
                sup.append(
                    f"| {population} | {split} | {scope} | {pe['n_molecules']} | {pe['n_instances']} | "
                    f"{pe['n_donors']} | {u['n_clusters']} | {_fmt(u['max_cluster_share'], 3)} | "
                    f"{_fmt(pe['loss_O'])} | {_fmt(pe['loss_T'])} | {_fmt(pe['loss_K'])} | "
                    f"{_fmt(pe['J'])} | {_fmt(pe['D'])} | {_fmt(pe['B'])} |"
                )
    (analysis_dir / "supplementary_table.md").write_text("\n".join(sup) + "\n", encoding="utf-8")

    C.log(f"analysis done: {len(results)} result rows, {len(mol_rows)} molecule rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
