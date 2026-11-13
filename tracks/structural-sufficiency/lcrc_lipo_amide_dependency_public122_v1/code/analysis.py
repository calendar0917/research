"""Local analysis: masked-prediction losses, paired contrasts, bootstrap, tables.

Definitions (protocol section 6), all in raw logD:

``L0_m``  ``|f(O_m) - y_m|``                      original molecule, original inputs
``LT_m``  ``mean_j |f(T_mj) - y_m|``              c-n cut, mean over qualified instances
``LK_m``  ``mean_j |f(K_mj) - y_m|``              c-acyl_c cut
``LR_m``  ``mean_j mean_cand |f(R_mjc) - y_m|``   remote q-r cut, minimal-cost ties averaged
``B``     ``mean_m (LT_m - L0_m)``                masking cost of the target link
``D``     ``mean_m (LT_m - LK_m)``                target-minus-matched-control masking cost
``B_R``   ``mean_m (LR_m - L0_m)``                same for the remote control
``D_R``   ``mean_m (LT_m - LR_m)``                target minus remote, same molecules

Molecules are equally weighted and instances are equally weighted inside a
molecule.  Losses are computed per view and then averaged; predictions are
never averaged before the loss.  ``RMSE`` uses the same weights on squared
errors.

Uncertainty: Bemis-Murcko scaffold clusters, paired cluster bootstrap with a
fixed seed; the same resample is used for every view/scope of one population.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import prepare as P
import views as V

FULL = "FULL"
G_FIXED = "G_FIXED"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def inference_gate_summary(checks: Mapping[str, Any]) -> dict[str, Any]:
    """Compact copy of the frozen-inference gates recorded by infer.py."""
    baseline = checks.get("baseline", {}) or {}
    state = checks.get("state_identity", {}) or {}
    perm = checks.get("permutation", {}) or {}
    per_split = {
        split: {
            "n": d.get("n"),
            "rmse_raw": d.get("rmse_raw"),
            "rmse_diff": d.get("rmse_diff"),
            "mae_raw": d.get("mae_raw"),
            "mae_diff": d.get("mae_diff"),
            "max_abs_pred_diff_vs_reference": d.get("max_abs_pred_diff_vs_reference"),
        }
        for split, d in (baseline.get("per_split", {}) or {}).items()
    }
    return {
        "pass": bool(checks.get("pass")),
        "baseline_pass": bool(baseline.get("pass")),
        "baseline_tolerance": baseline.get("tolerance"),
        "baseline_per_split": per_split,
        "permutation_pass": bool(perm.get("pass")),
        "permutation_max_abs_pred_diff": max(
            (float(m.get("max_abs_pred_diff", 0.0)) for m in perm.get("molecules", []) or []),
            default=None,
        ),
        "state_hash_before": state.get("state_hash_before"),
        "state_hash_after": state.get("state_hash_after"),
        "norm_fingerprint_unchanged": state.get("norm_fingerprint_before")
        == state.get("norm_fingerprint_after"),
        "optimizer_created": state.get("optimizer_created"),
        "backward_calls": state.get("backward_calls"),
        "model_training_steps": state.get("model_training_steps"),
    }


def verify_protocol_inputs(analysis_dir: Path) -> dict[str, Any]:
    """The committed protocol inputs must be the bytes the inference consumed."""
    manifest = C.read_json(analysis_dir / "inputs_manifest.json")
    recorded = manifest.get("artifacts", {}).get("files", {})
    out: dict[str, Any] = {"recorded": recorded, "observed": {}, "pass": True}
    for name, want in recorded.items():
        path = analysis_dir / name
        got = C.sha256_file(path) if path.exists() else "missing"
        out["observed"][name] = got
        if got != want:
            out["pass"] = False
    return out


def load_predictions(analysis_dir: Path) -> dict[tuple[str, str], float]:
    rows = P.read_csv(analysis_dir / "per_view_predictions.csv")
    return {(str(r["view_id"]), str(r["scope"])): float(r["pred_raw"]) for r in rows}


def load_catalog(
    analysis_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inst_raw = P.read_csv(analysis_dir / "motif_catalog.csv")
    inst: list[dict[str, Any]] = []
    for r in inst_raw:
        d: dict[str, Any] = dict(r)
        for k in (
            "row_id",
            "c",
            "o",
            "n",
            "acyl_c",
            "degree_n",
            "degree_acyl",
            "ring_n",
            "ring_acyl",
            "matched",
            "n_side",
            "acyl_side",
            "component_size",
            "side_diff",
            "balanced",
            "small_side_target",
            "bond_cn",
            "bond_c_acyl",
            "bond_c_o",
            "r_n_candidates",
            "r_n_candidates_any_component",
            "r_n_ties",
        ):
            d[k] = int(r[k])
        d["views_r"] = [x for x in str(r.get("view_r", "")).split(";") if x]
        inst.append(d)
    mol_raw = P.read_csv(analysis_dir / "molecule_catalog.csv")
    mol: list[dict[str, Any]] = []
    for r in mol_raw:
        d: dict[str, Any] = dict(r)
        for k in (
            "row_id",
            "n_atoms",
            "n_bonds",
            "n_components",
            "n_instances",
            "n_matched",
            "n_balanced",
            "n_r_instances",
            "pop_all",
            "pop_matched",
            "pop_balanced",
            "pop_r",
        ):
            d[k] = int(r[k])
        mol.append(d)
    views = P.read_csv(analysis_dir / "view_manifest.csv")
    return inst, mol, views


def load_targets(analysis_dir: Path) -> tuple[dict[int, float], dict[int, str]]:
    rows = P.read_csv(analysis_dir / "reference_o_predictions.csv")
    y = {int(r["row_id"]): float(r["y"]) for r in rows}
    split = {int(r["row_id"]): str(r["split"]) for r in rows}
    return y, split


# ---------------------------------------------------------------------------
# per-molecule panel
# ---------------------------------------------------------------------------


def population_instances(
    inst: Sequence[Mapping[str, Any]], population: str
) -> dict[int, list[Mapping[str, Any]]]:
    if population == "ALL":
        keep = lambda r: True  # noqa: E731
    elif population == "MATCHED":
        keep = lambda r: bool(r["matched"])  # noqa: E731
    elif population == "BALANCED":
        keep = lambda r: bool(r["balanced"])  # noqa: E731
    elif population == "R":
        keep = lambda r: bool(r["views_r"])  # noqa: E731
    else:
        raise KeyError(population)
    out: dict[int, list[Mapping[str, Any]]] = {}
    for r in inst:
        if keep(r):
            out.setdefault(int(r["row_id"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r["c"]))
    return out


def panel(
    population: str,
    split: str,
    scope: str,
    *,
    inst: Sequence[Mapping[str, Any]],
    y: Mapping[int, float],
    preds: Mapping[tuple[str, str], float],
    split_ids: set[int],
) -> dict[str, Any]:
    """Per-molecule losses / contrasts for one population, split and scope."""
    pop = population_instances(inst, population)
    split_rows = sorted(rid for rid in pop if rid in y and int(rid) in split_ids)

    out_rows: list[int] = []
    l0, lt, lk, lr = [], [], [], []
    sq0, sqt, sqk, sqr = [], [], [], []
    mv_abs_t, mv_sig_t, mv_abs_k, mv_sig_k = [], [], [], []
    mv_abs_r, mv_sig_r = [], []
    n_inst, n_r_cands = [], []
    l0_rsub, lt_rsub = [], []
    for rid in split_rows:
        insts = pop[rid]
        yy = float(y[rid])
        o = float(preds[(f"{rid}:O", FULL)])
        err_t, err_k, err_r = [], [], []
        mabs_t, msig_t, mabs_k, msig_k, mabs_r, msig_r = [], [], [], [], [], []
        n_c = 0
        for r in insts:
            pt = float(preds[(str(r["view_t"]), scope)])
            pk = float(preds[(str(r["view_k"]), scope)])
            err_t.append(abs(pt - yy))
            err_k.append(abs(pk - yy))
            mabs_t.append(abs(pt - o))
            msig_t.append(pt - o)
            mabs_k.append(abs(pk - o))
            msig_k.append(pk - o)
            if r["views_r"]:
                e_r = [abs(float(preds[(vid, scope)]) - yy) for vid in r["views_r"]]
                mo_r = [float(preds[(vid, scope)]) - o for vid in r["views_r"]]
                err_r.append(float(np.mean(e_r)))
                mabs_r.append(float(np.mean(np.abs(mo_r))))
                msig_r.append(float(np.mean(mo_r)))
                n_c += len(r["views_r"])
        out_rows.append(rid)
        n_inst.append(len(insts))
        n_r_cands.append(n_c)
        l_t, l_k = float(np.mean(err_t)), float(np.mean(err_k))
        l0.append(abs(o - yy))
        lt.append(l_t)
        lk.append(l_k)
        sq0.append((o - yy) ** 2)
        sqt.append(float(np.mean([e * e for e in err_t])))
        sqk.append(float(np.mean([e * e for e in err_k])))
        mv_abs_t.append(float(np.mean(mabs_t)))
        mv_sig_t.append(float(np.mean(msig_t)))
        mv_abs_k.append(float(np.mean(mabs_k)))
        mv_sig_k.append(float(np.mean(msig_k)))
        if err_r:
            l0_rsub.append(abs(o - yy))
            lt_rsub.append(
                float(
                    np.mean(
                        [abs(float(preds[(str(r["view_t"]), scope)]) - yy) for r in insts if r["views_r"]]
                    )
                )
            )
            lr.append(float(np.mean(err_r)))
            # RMSE analogue: per instance, mean squared error over the tied
            # candidates (the same loss-layer averaging), then over instances.
            sq_r_inst = []
            for r in insts:
                if not r["views_r"]:
                    continue
                sq_r_inst.append(
                    float(np.mean([(float(preds[(vid, scope)]) - yy) ** 2 for vid in r["views_r"]]))
                )
            sqr.append(float(np.mean(sq_r_inst)))
            mv_abs_r.append(float(np.mean(mabs_r)))
            mv_sig_r.append(float(np.mean(msig_r)))
        else:
            l0_rsub.append(np.nan)
            lt_rsub.append(np.nan)
            lr.append(np.nan)
            sqr.append(np.nan)
            mv_abs_r.append(np.nan)
            mv_sig_r.append(np.nan)
    d = {
        "population": population,
        "split": split,
        "scope": scope,
        "rows": np.asarray(out_rows, dtype=np.int64),
        "n_instances": np.asarray(n_inst, dtype=np.int64),
        "n_r_candidates": np.asarray(n_r_cands, dtype=np.int64),
        "L0": np.asarray(l0, dtype=np.float64),
        "LT": np.asarray(lt, dtype=np.float64),
        "LK": np.asarray(lk, dtype=np.float64),
        "LR": np.asarray(lr, dtype=np.float64),
        "sq0": np.asarray(sq0, dtype=np.float64),
        "sqT": np.asarray(sqt, dtype=np.float64),
        "sqK": np.asarray(sqk, dtype=np.float64),
        "sqR": np.asarray(sqr, dtype=np.float64),
        "move_abs_T": np.asarray(mv_abs_t, dtype=np.float64),
        "move_signed_T": np.asarray(mv_sig_t, dtype=np.float64),
        "move_abs_K": np.asarray(mv_abs_k, dtype=np.float64),
        "move_signed_K": np.asarray(mv_sig_k, dtype=np.float64),
        "move_abs_R": np.asarray(mv_abs_r, dtype=np.float64),
        "move_signed_R": np.asarray(mv_sig_r, dtype=np.float64),
        "L0_Rsub": np.asarray(l0_rsub, dtype=np.float64),
        "LT_Rsub": np.asarray(lt_rsub, dtype=np.float64),
    }
    d["B"] = d["LT"] - d["L0"]
    d["D"] = d["LT"] - d["LK"]
    d["B_R"] = d["LR"] - d["L0"]
    d["D_R"] = d["LT"] - d["LR"]
    return d


# ---------------------------------------------------------------------------
# point estimates
# ---------------------------------------------------------------------------


def _mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(a.mean())


def _rmse(sq: np.ndarray) -> float:
    sq = np.asarray(sq, dtype=np.float64)
    sq = sq[np.isfinite(sq)]
    if sq.size == 0:
        return float("nan")
    return float(np.sqrt(sq.mean()))


def point_estimates(d: Mapping[str, Any]) -> dict[str, float]:
    return {
        "MAE_O": _mean(d["L0"]),
        "MAE_T": _mean(d["LT"]),
        "MAE_K": _mean(d["LK"]),
        "MAE_R": _mean(d["LR"]),
        "RMSE_O": _rmse(d["sq0"]),
        "RMSE_T": _rmse(d["sqT"]),
        "RMSE_K": _rmse(d["sqK"]),
        "RMSE_R": _rmse(d["sqR"]),
        "B": _mean(d["B"]),
        "D": _mean(d["D"]),
        "B_R": _mean(d["B_R"]),
        "D_R": _mean(d["D_R"]),
        "move_abs_T": _mean(d["move_abs_T"]),
        "move_signed_T": _mean(d["move_signed_T"]),
        "move_abs_K": _mean(d["move_abs_K"]),
        "move_signed_K": _mean(d["move_signed_K"]),
        "move_abs_R": _mean(d["move_abs_R"]),
        "move_signed_R": _mean(d["move_signed_R"]),
        "MAE_O_Rsub": _mean(d["L0_Rsub"]),
        "MAE_T_Rsub": _mean(d["LT_Rsub"]),
        "n_molecules": int(d["rows"].size),
        "n_instances": int(d["n_instances"].sum()),
        "n_r_candidates": int(d["n_r_candidates"].sum()),
        "n_molecules_with_R": int(np.isfinite(d["LR"]).sum()),
        "n_instances_with_R": int(sum(1 for x in d["n_r_candidates"] if int(x) > 0)),
    }


# ---------------------------------------------------------------------------
# paired cluster bootstrap
# ---------------------------------------------------------------------------


def cluster_weights(
    rows: np.ndarray,
    scaffold: Mapping[int, str],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    """One (replicates x molecules) multiplicity matrix, cluster-grouped."""
    keys = [scaffold[int(r)] for r in rows.tolist()]
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    cluster_idx = np.asarray([uniq[k] for k in keys], dtype=np.int64)
    k_total = len(uniq)
    draws = rng.integers(0, k_total, size=(int(replicates), k_total))
    counts = np.zeros((int(replicates), k_total), dtype=np.int64)
    for b in range(int(replicates)):
        counts[b] = np.bincount(draws[b], minlength=k_total)
    weights = counts[:, cluster_idx].astype(np.float64)
    sizes = np.bincount(cluster_idx, minlength=k_total)
    return {
        "weights": weights,
        "cluster_draw_counts": counts,
        "n_clusters": int(k_total),
        "cluster_index": cluster_idx,
        "cluster_sizes": sizes,
        "max_cluster_share": float(sizes.max() / sizes.sum()) if sizes.size else 0.0,
        "max_cluster_key": sorted(uniq)[int(np.argmax(sizes))],
    }


def weighted_stats(weights: np.ndarray, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Weighted means of the per-molecule contrasts for every replicate."""
    out: dict[str, np.ndarray] = {}
    for key in ("B", "D", "B_R", "D_R", "L0", "LT", "LK", "LR"):
        v = np.asarray(values[key], dtype=np.float64)
        ok = np.isfinite(v)
        w = weights * ok.reshape(1, -1)
        num = w @ np.nan_to_num(v, nan=0.0)
        den = w.sum(axis=1)
        out[key] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    for key, sq_key in (("RMSE_O", "sq0"), ("RMSE_T", "sqT"), ("RMSE_K", "sqK"), ("RMSE_R", "sqR")):
        v = np.asarray(values[sq_key], dtype=np.float64)
        ok = np.isfinite(v)
        w = weights * ok.reshape(1, -1)
        num = w @ np.nan_to_num(v, nan=0.0)
        den = w.sum(axis=1)
        out[key] = np.sqrt(np.where(den > 0, num / np.maximum(den, 1e-12), np.nan))
    return out


def percentile_ci(reps: np.ndarray, level: float = 95.0) -> list[float]:
    reps = np.asarray(reps, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return [float("nan"), float("nan")]
    lo = (100.0 - level) / 2.0
    hi = 100.0 - lo
    return [float(np.percentile(reps, lo)), float(np.percentile(reps, hi))]


# ---------------------------------------------------------------------------
# influence checks
# ---------------------------------------------------------------------------


def influence_checks(d: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed leave-out rules on the main D contrast (protocol section 6)."""
    rows = np.asarray(d["rows"], dtype=np.int64)
    D = np.asarray(d["D"], dtype=np.float64)
    B = np.asarray(d["B"], dtype=np.float64)
    n = rows.size
    order = sorted(range(n), key=lambda i: (-abs(float(D[i])), int(rows[i])))
    worst = order[0]
    k = int(math.ceil(C.INFLUENCE_FRACTION * n))
    drop_rows = sorted(int(rows[i]) for i in order[:k])
    keep_single = np.ones(n, dtype=bool)
    keep_single[worst] = False
    keep_block = ~np.isin(rows, drop_rows)
    return {
        "n_molecules": int(n),
        "n_dropped_block": k,
        "max_abs_D_row": int(rows[worst]),
        "max_abs_D_value": float(D[worst]),
        "max_abs_D_signed": float(D[worst]),
        "B_without_max_row": _mean(B[keep_single]),
        "D_without_max_row": _mean(D[keep_single]),
        "B_without_top_block": _mean(B[keep_block]),
        "D_without_top_block": _mean(D[keep_block]),
        "dropped_block_rows": drop_rows,
    }


def _instance_payload(
    r: Mapping[str, Any],
    rid: int,
    preds: Mapping[tuple[str, str], float],
    y: Mapping[int, float],
) -> dict[str, Any]:
    scoped: dict[str, Any] = {}
    # the O view is scope independent (FULL identical to G_FIXED, checked in infer.py)
    po = float(preds[(f"{rid}:O", C.PRIMARY_SCOPE)])
    for scope in C.SCOPES:
        pt = float(preds[(str(r["view_t"]), scope)])
        pk = float(preds[(str(r["view_k"]), scope)])
        scoped[scope] = {
            "pred_O": po,
            "pred_T": pt,
            "pred_K": pk,
            "err_O": abs(po - float(y[rid])),
            "err_T": abs(pt - float(y[rid])),
            "err_K": abs(pk - float(y[rid])),
        }
    return {
        "instance_id": str(r["instance_id"]),
        "c": int(r["c"]),
        "o": int(r["o"]),
        "n": int(r["n"]),
        "acyl_c": int(r["acyl_c"]),
        "matched": int(r["matched"]),
        "balanced": int(r["balanced"]),
        "n_views_r": len(r["views_r"]),
        "view_t": str(r["view_t"]),
        "view_k": str(r["view_k"]),
        "scopes": scoped,
    }


def _molecule_entry(
    i: int,
    *,
    panels_by_scope: Mapping[str, Mapping[str, Any]],
    pop_insts: Mapping[int, Sequence[Mapping[str, Any]]],
    inst_by_row: Mapping[int, Sequence[Mapping[str, Any]]],
    preds: Mapping[tuple[str, str], float],
    y: Mapping[int, float],
) -> dict[str, Any]:
    rows = np.asarray(panels_by_scope[C.PRIMARY_SCOPE]["rows"], dtype=np.int64)
    rid = int(rows[i])
    scoped: dict[str, Any] = {}
    for scope, d in panels_by_scope.items():
        block = {
            "L0": float(d["L0"][i]),
            "LT": float(d["LT"][i]),
            "LK": float(d["LK"][i]),
            "B": float(d["B"][i]),
            "D": float(d["D"][i]),
        }
        for k in ("LR", "B_R", "D_R"):
            val = float(d[k][i])
            block[k] = val if np.isfinite(val) else None
        scoped[scope] = block
    entry = {
        "row_id": rid,
        "y": float(y[rid]),
        "n_population_instances": len(pop_insts.get(rid, [])),
        "scopes": scoped,
        "instances_in_population": [str(r["instance_id"]) for r in pop_insts.get(rid, [])],
        "instances": [_instance_payload(r, rid, preds, y) for r in inst_by_row.get(rid, [])],
    }
    # convenience mirrors of the primary scope (kept so tables/figures degrade gracefully)
    entry.update({k: scoped[C.PRIMARY_SCOPE][k] for k in ("L0", "LT", "LK", "B", "D")})
    return entry


def max_impact_case(
    *,
    panels_by_scope: Mapping[str, Mapping[str, Any]],
    pop_insts: Mapping[int, Sequence[Mapping[str, Any]]],
    inst_by_row: Mapping[int, Sequence[Mapping[str, Any]]],
    preds: Mapping[tuple[str, str], float],
    y: Mapping[int, float],
    ranking_scope: str,
) -> dict[str, Any]:
    """Molecule with the largest |D| in ``ranking_scope`` (ties -> smallest row)."""
    d = panels_by_scope[ranking_scope]
    rows = np.asarray(d["rows"], dtype=np.int64)
    D = np.asarray(d["D"], dtype=np.float64)
    order = sorted(range(rows.size), key=lambda i: (-abs(float(D[i])), int(rows[i])))
    i = order[0]
    out = _molecule_entry(
        i, panels_by_scope=panels_by_scope, pop_insts=pop_insts, inst_by_row=inst_by_row, preds=preds, y=y
    )
    out["ranking_scope"] = ranking_scope
    return out


def cases(
    panels_by_scope: Mapping[str, Mapping[str, Any]],
    *,
    inst: Sequence[Mapping[str, Any]],
    population: str,
    preds: Mapping[tuple[str, str], float],
    y: Mapping[int, float],
    scaffold: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Median / max / min signed-D molecules of the main population (primary scope)."""
    d = panels_by_scope[C.PRIMARY_SCOPE]
    rows = np.asarray(d["rows"], dtype=np.int64)
    D = np.asarray(d["D"], dtype=np.float64)
    order = np.argsort(D, kind="stable")
    med = float(np.median(D))
    cand = [
        ("median", int(np.argmin(np.abs(D - med)))),
        ("max_D", int(np.argmax(D))),
        ("min_D", int(np.argmin(D))),
    ]
    pop_insts = population_instances(inst, population)
    inst_by_row: dict[int, list[Mapping[str, Any]]] = {}
    for r in inst:
        inst_by_row.setdefault(int(r["row_id"]), []).append(r)
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for tag, i in cand:
        rid = int(rows[i])
        if rid in seen:
            continue
        seen.add(rid)
        entry = _molecule_entry(
            i, panels_by_scope=panels_by_scope, pop_insts=pop_insts, inst_by_row=inst_by_row, preds=preds, y=y
        )
        entry["tag"] = tag
        entry["D_rank"] = int(np.where(order == i)[0][0]) + 1
        entry["n_molecules"] = int(rows.size)
        entry["ranking_scope"] = C.PRIMARY_SCOPE
        entry["scaffold_key"] = scaffold.get(rid, "")
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# frozen normalisation range check
# ---------------------------------------------------------------------------


def normalised_range_check(
    *,
    norm: Mapping[str, np.ndarray],
    spec,
    raw_rows: Sequence[Mapping[str, Any]],
    view_rows: Sequence[Mapping[str, Any]],
    remote_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """T/K normalised global/scalar values outside the full Train min/max."""
    import torch

    train = torch.load(C.DD.cache_path(C.DATA_REPEAT, "train"), map_location="cpu", weights_only=False)
    gm = np.asarray(norm["global_mean"], dtype=np.float64)
    gs = np.asarray(norm["global_scale"], dtype=np.float64)
    sm = np.asarray(norm["scalar_mean"], dtype=np.float64)
    ss = np.asarray(norm["scalar_scale"], dtype=np.float64)
    g_train = (np.asarray(train["global"], dtype=np.float64) - gm) / gs
    s_train = (np.asarray(train["atom_scalar"], dtype=np.float64) - sm) / ss
    g_lo, g_hi = g_train.min(axis=0), g_train.max(axis=0)
    s_lo, s_hi = s_train.min(axis=0), s_train.max(axis=0)

    out: dict[str, Any] = {
        "train_cache": {
            "path": C.track_rel(C.DD.cache_path(C.DATA_REPEAT, "train")),
            "n_molecules": int(train["n_molecules"]),
            "content_fingerprint": train["meta"].get("content_fingerprint"),
            "n_atoms": int(np.asarray(train["atom_ptr"])[-1]),
        },
        "per_split": {},
    }
    for split in ("valid", "test"):
        entries = []
        for r in view_rows:
            if r["split"] != split:
                continue
            rid = int(r["row_id"])
            base = V.row_from_artifact(raw_rows[rid])
            base["n_components"] = V.n_components_of_row(base)
            if int(r["cut_a"]) < 0:
                entries.append((str(r["view_id"]), base, "O"))
            else:
                cut, _ = V.cut_row(base, int(r["cut_a"]), int(r["cut_b"]))
                entries.append((str(r["view_id"]), cut, "cut"))
        cache = V.build_cache([(vid, row) for vid, row, _k in entries], spec, split=f"range_{split}")
        fp = cache["meta"].get("content_fingerprint")
        g = (np.asarray(cache["global"], dtype=np.float64) - gm) / gs
        kinds = np.asarray([k for _vid, _row, k in entries])
        stats: dict[str, Any] = {}
        for kind in ("cut", "O"):
            mask = kinds == kind
            if not mask.any():
                continue
            gg = g[mask]
            below = gg < g_lo.reshape(1, -1)
            above = gg > g_hi.reshape(1, -1)
            outside = below | above
            excess = np.where(below, g_lo.reshape(1, -1) - gg, 0.0) + np.where(
                above, gg - g_hi.reshape(1, -1), 0.0
            )
            # atom scalars: per-atom values of the same views
            ai = np.asarray(cache["atom_ptr"])
            atom_mask = np.zeros(int(ai[-1]), dtype=bool)
            for gi in np.flatnonzero(mask):
                atom_mask[int(ai[gi]) : int(ai[gi + 1])] = True
            sc = (np.asarray(cache["atom_scalar"], dtype=np.float64)[atom_mask] - sm) / ss
            s_below = sc < s_lo.reshape(1, -1)
            s_above = sc > s_hi.reshape(1, -1)
            s_out = s_below | s_above
            s_excess = np.where(s_below, s_lo.reshape(1, -1) - sc, 0.0) + np.where(
                s_above, sc - s_hi.reshape(1, -1), 0.0
            )
            stats[kind] = {
                "n_views": int(mask.sum()),
                "n_atoms": int(atom_mask.sum()),
                "global": {
                    "n_values": int(gg.size),
                    "n_outside": int(outside.sum()),
                    "fraction_outside": float(outside.sum() / gg.size),
                    "n_views_with_any_outside": int(outside.any(axis=1).sum()),
                    "fraction_views_with_any_outside": float(outside.any(axis=1).mean()),
                    "max_excess_normalised": float(excess.max()) if excess.size else 0.0,
                    "per_dim_fraction_outside": [float(x) for x in outside.mean(axis=0)],
                    "per_dim_max_excess": [float(x) for x in excess.max(axis=0)],
                },
                "atom_scalar": {
                    "n_values": int(sc.size),
                    "n_outside": int(s_out.sum()),
                    "fraction_outside": float(s_out.sum() / sc.size),
                    "n_atoms_with_any_outside": int(s_out.any(axis=1).sum()),
                    "fraction_atoms_with_any_outside": float(s_out.any(axis=1).mean()),
                    "max_excess_normalised": float(s_excess.max()) if s_excess.size else 0.0,
                    "per_dim_fraction_outside": [float(x) for x in s_out.mean(axis=0)],
                    "per_dim_max_excess": [float(x) for x in s_excess.max(axis=0)],
                },
            }
        # how far each global dimension moves when the bond is cut (same molecules)
        rids = np.asarray([int(r["row_id"]) for _vid, _row, _k in entries])
        o_index = {int(rid): j for j, (rid, k) in enumerate(zip(rids, kinds)) if k == "O"}
        delta_rows = []
        for j in np.flatnonzero(kinds == "cut"):
            rid = int(rids[j])
            if rid in o_index:
                delta_rows.append(g[j] - g[o_index[rid]])
        split_block: dict[str, Any] = {
            "view_cache_content_fingerprint": fp,
            "view_cache_fingerprint_matches_remote": bool(remote_fingerprints.get(split) in (None, fp)),
            "stats": stats,
        }
        if delta_rows:
            delta = np.vstack(delta_rows)
            field_names = [f.name for f in spec.global_feature_fields]
            mean_abs = np.abs(delta).mean(axis=0)
            rank = np.argsort(-mean_abs)[: min(8, mean_abs.size)]
            split_block["top_moving_global_dims"] = {
                "n_pairs": int(delta.shape[0]),
                "unit": "Train standard deviations of the frozen normalisation",
                "dims": [
                    {
                        "index": int(i),
                        "name": field_names[int(i)] if int(i) < len(field_names) else f"dim{i}",
                        "mean_abs_change_sd": float(mean_abs[int(i)]),
                        "mean_signed_change_sd": float(delta[:, int(i)].mean()),
                    }
                    for i in rank
                ],
            }
        out["per_split"][split] = split_block
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
    inst, mol, view_rows = load_catalog(analysis_dir)
    preds = load_predictions(analysis_dir)
    y, split_of = load_targets(analysis_dir)
    scaffold = {int(m["row_id"]): str(m["scaffold_key"]) for m in mol}

    runtime = C.read_json(analysis_dir / "runtime.json") if (analysis_dir / "runtime.json").exists() else {}
    infer_checks = (
        C.read_json(analysis_dir / "infer_checks.json")
        if (analysis_dir / "infer_checks.json").exists()
        else {}
    )
    remote_fps = runtime.get("view_cache_fingerprints") or {}

    ck = C.load_checkpoint()
    norm = C.frozen_norm(ck)
    spec = C.DD.frozen_spec()

    # ---- point estimates + bootstrap -------------------------------------
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    results: list[dict[str, Any]] = []
    uncertainty: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "protocol_inputs": verify_protocol_inputs(analysis_dir),
        "inference_gates": inference_gate_summary(infer_checks),
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "bootstrap_replicates": C.BOOTSTRAP_REPLICATES,
        "cluster_definition": "Bemis-Murcko scaffold (isomeric-flag-free canonical SMILES); "
        "empty scaffold -> molecule's own non-isomeric canonical SMILES",
        "populations": {},
    }
    panels: dict[tuple[str, str, str], dict[str, Any]] = {}
    split_ids = {
        s: {
            int(r["row_id"])
            for r in P.read_csv(analysis_dir / "reference_o_predictions.csv")
            if r["split"] == s
        }
        for s in ("valid", "test")
    }
    mae_src = {"MAE_O": "L0", "MAE_T": "LT", "MAE_K": "LK", "MAE_R": "LR"}
    for split in ("valid", "test"):
        for population in ("MATCHED", "ALL", "BALANCED", "R"):
            base = panel(population, split, FULL, inst=inst, y=y, preds=preds, split_ids=split_ids[split])
            fixed = panel(population, split, G_FIXED, inst=inst, y=y, preds=preds, split_ids=split_ids[split])
            rows = base["rows"]
            if rows.size == 0:
                continue
            cw = cluster_weights(rows, scaffold, rng=rng, replicates=C.BOOTSTRAP_REPLICATES)
            W = cw["weights"]
            uncertainties: dict[str, Any] = {}
            for scope, d in ((FULL, base), (G_FIXED, fixed)):
                panels[(population, split, scope)] = d
                pe = point_estimates(d)
                reps = weighted_stats(W, d)
                for key in ("B", "D", "B_R", "D_R"):
                    lo, hi = percentile_ci(reps[key])
                    uncertainties[f"{scope}:{key}"] = {
                        "point": pe[key],
                        "ci95": [lo, hi],
                        "n_replicates_finite": int(np.isfinite(reps[key]).sum()),
                    }
                for key in ("MAE_O", "MAE_T", "MAE_K", "MAE_R", "RMSE_O", "RMSE_T", "RMSE_K", "RMSE_R"):
                    src = mae_src.get(key, key)
                    lo, hi = percentile_ci(reps[src])
                    uncertainties[f"{scope}:{key}"] = {"point": pe[key], "ci95": [lo, hi]}
                for metric, value in pe.items():
                    results.append(
                        {
                            "population": population,
                            "split": split,
                            "scope": scope,
                            "view": metric.split("_")[-1] if metric[:3] in ("MAE", "RMS") else "",
                            "metric": metric,
                            "value": value,
                            "n_molecules": pe["n_molecules"],
                            "n_instances": pe["n_instances"],
                        }
                    )
                # ci columns in results for the four main contrasts
                for key in ("B", "D", "B_R", "D_R"):
                    lo, hi = percentile_ci(reps[key])
                    results.append(
                        {
                            "population": population,
                            "split": split,
                            "scope": scope,
                            "view": "",
                            "metric": f"{key}_ci95_low",
                            "value": lo,
                            "n_molecules": pe["n_molecules"],
                            "n_instances": pe["n_instances"],
                        }
                    )
                    results.append(
                        {
                            "population": population,
                            "split": split,
                            "scope": scope,
                            "view": "",
                            "metric": f"{key}_ci95_high",
                            "value": hi,
                            "n_molecules": pe["n_molecules"],
                            "n_instances": pe["n_instances"],
                        }
                    )
            uncertainty["populations"][f"{population}/{split}"] = {
                "n_molecules": int(rows.size),
                "n_instances": int(base["n_instances"].sum()),
                "n_clusters": int(cw["n_clusters"]),
                "max_cluster_share": float(cw["max_cluster_share"]),
                "max_cluster_share_rule": "largest cluster as a fraction of the population's molecules",
                "scopes": uncertainties,
            }

    # ---- influence checks (fixed rule, applied to both frozen scopes) ----
    main_panels = {scope: panels[("MATCHED", "test", scope)] for scope in C.SCOPES}
    pop_insts = population_instances(inst, "MATCHED")
    inst_by_row: dict[int, list[Mapping[str, Any]]] = {}
    for r in inst:
        inst_by_row.setdefault(int(r["row_id"]), []).append(r)
    for scope in C.SCOPES:
        influence = influence_checks(main_panels[scope])
        influence["max_impact_case"] = max_impact_case(
            panels_by_scope=main_panels,
            pop_insts=pop_insts,
            inst_by_row=inst_by_row,
            preds=preds,
            y=y,
            ranking_scope=scope,
        )
        influence["ranking_scope"] = scope
        uncertainty[f"influence_main_MATCHED_test_{scope}"] = influence

    # ---- case selection -------------------------------------------------
    case_rows = cases(
        main_panels,
        inst=inst,
        population="MATCHED",
        preds=preds,
        y=y,
        scaffold=scaffold,
    )
    C.write_json(
        analysis_dir / "cases.json",
        {
            "main_population": "MATCHED/test",
            "ranking_scope": C.PRIMARY_SCOPE,
            "note": "all instances of the molecule are listed; instances_in_population are the ones entering the loss",
            "cases": case_rows,
        },
    )

    # ---- frozen normalisation range check --------------------------------
    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    range_check = normalised_range_check(
        norm=norm, spec=spec, raw_rows=raw_rows, view_rows=view_rows, remote_fingerprints=remote_fps
    )
    uncertainty["normalised_range_check"] = range_check

    # ---- results.csv -----------------------------------------------------
    columns = ["population", "split", "scope", "view", "metric", "value", "n_molecules", "n_instances"]
    P.write_csv(analysis_dir / "results.csv", results, columns)
    C.write_json(analysis_dir / "uncertainty.json", uncertainty)

    # ---- tables ----------------------------------------------------------
    main_lines = [
        "# 主表 - MATCHED 人群 (FULL 为主估计, G_FIXED 为定位诊断)",
        "",
        "loss 为原始 logD 尺度上的绝对误差; B = mean(LT - L0), D = mean(LT - LK)。",
        "CI 为 scaffold cluster bootstrap 95% percentile 区间 (1000 次, seed 20261114)。",
        "",
        "| split | scope | view | MAE | RMSE | B [95% CI] | D [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in ("valid", "test"):
        for scope in (FULL, G_FIXED):
            key = ("MATCHED", split, scope)
            if key not in panels:
                continue
            pe = point_estimates(panels[key])
            u = uncertainty["populations"][f"MATCHED/{split}"]["scopes"]
            b = u[f"{scope}:B"]
            d = u[f"{scope}:D"]
            main_lines.append(
                f"| {split} | {scope} | O | {_fmt(pe['MAE_O'])} | {_fmt(pe['RMSE_O'])} | "
                f"{_fmt(b['point'])} [{_fmt(b['ci95'][0])}, {_fmt(b['ci95'][1])}] | "
                f"{_fmt(d['point'])} [{_fmt(d['ci95'][0])}, {_fmt(d['ci95'][1])}] |"
            )
            main_lines.append(f"| {split} | {scope} | T | {_fmt(pe['MAE_T'])} | {_fmt(pe['RMSE_T'])} | | |")
            main_lines.append(f"| {split} | {scope} | K | {_fmt(pe['MAE_K'])} | {_fmt(pe['RMSE_K'])} | | |")
    (analysis_dir / "main_table.md").write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    sup_lines = [
        "# 补充表 - 全部人群与支持数",
        "",
        "| population | split | scope | n_mol | n_inst | n_clusters | max_cluster | MAE_O | MAE_T | MAE_K | n_mol_R | n_inst_R | MAE_T_Rsub | MAE_R | B | D | B_R | D_R |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for population in ("MATCHED", "ALL", "BALANCED", "R"):
        for split in ("valid", "test"):
            for scope in (FULL, G_FIXED):
                key = (population, split, scope)
                if key not in panels:
                    continue
                pe = point_estimates(panels[key])
                up = uncertainty["populations"][f"{population}/{split}"]
                sup_lines.append(
                    f"| {population} | {split} | {scope} | {pe['n_molecules']} | {pe['n_instances']} | "
                    f"{up['n_clusters']} | {_fmt(up['max_cluster_share'], 3)} | "
                    f"{_fmt(pe['MAE_O'])} | {_fmt(pe['MAE_T'])} | {_fmt(pe['MAE_K'])} | "
                    f"{pe['n_molecules_with_R']} | {pe['n_instances_with_R']} | "
                    f"{_fmt(pe['MAE_T_Rsub'])} | {_fmt(pe['MAE_R'])} | "
                    f"{_fmt(pe['B'])} | {_fmt(pe['D'])} | {_fmt(pe['B_R'])} | {_fmt(pe['D_R'])} |"
                )
    (analysis_dir / "supplementary_table.md").write_text("\n".join(sup_lines) + "\n", encoding="utf-8")

    C.log(f"analysis done: {len(results)} result rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
