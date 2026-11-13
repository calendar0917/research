"""Part A: three zero-GPU clarifications of the frozen multi-motif round.

No model, no optimiser, no new seed, no new inference: every input is an already
committed artifact of the executed multi-motif round (protocol ``0273cad6``,
report ``a5febe96``).

* **A1 coverage & claim matrix** — per family/split: pre-calibration and
  post-calibration eligible molecules / scaffold clusters / instances / T pairs,
  the C link count, the C unique-view count, the calibration retention ratio and
  the largest scaffold-cluster share.  The 539 C links are never described as
  independent molecules.
* **A2 same-population three tables** — ``ALL_K0`` (all pre-calibration
  T/K0), ``RETAINED_K0`` (only focal T that have a C, still compared to the
  original K0) and ``RETAINED_C`` (the identical focal T compared to the matched
  C).  Instance/control/donor weights are kept; nothing becomes view-equal.
* **A3 conditional scalar / S2 recovery** — ``R3 = value(123)-value(12)`` and
  ``R2 = value(123)-value(13)`` in the F (raw prediction), H (response) and L
  (reference deviation) spaces, with paired means, 95% intervals, per-view
  absolute recovery and the arithmetic identity
  ``R3 = gamma3+gamma13+gamma23+gamma123`` (residual <= 1e-5).

The clarifications never change an old protocol or stored number; they are
appended to the old report as a sourced note.  Results are kept whatever their
sign.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import round_common as RC

SUBSETS = ("", "1", "2", "3", "12", "13", "23", "123")
TERMS = ("g1", "g2", "g3", "g12", "g13", "g23", "g123")
MOBIUS_COEF = RC.AN.MOBIUS_COEF
SAME_POPULATIONS = ("ALL_K0", "RETAINED_K0", "RETAINED_C")


def _f(x: Any) -> float:
    try:
        s = str(x).strip()
        return float(s) if s not in ("", "nan", "None") else float("nan")
    except Exception:
        return float("nan")


def _i(x: Any, d: int = 0) -> int:
    v = _f(x)
    return d if not np.isfinite(v) else int(v)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_source_inputs() -> dict[str, Any]:
    analysis = RC.SOURCE_ANALYSIS
    pairs = RC.read_csv(analysis / "pair_catalog.csv")
    caliper = RC.read_csv(analysis / "caliper_catalog.csv")
    graph_manifest = RC.read_csv(analysis / "graph_manifest.csv")
    internal = RC.read_csv(analysis / "per_graph_internal.csv")
    prev = RC.SC.load_previous_inputs()
    raw = RC.SC.load_raw_rows()["rows"]["Lipophilicity"]
    scaffold = {rid: RC.G.bemis_murcko_key(str(raw[rid]["smiles"])) for rid in range(RC.SC.N_MOLECULES)}
    return {
        "pairs": pairs,
        "caliper": caliper,
        "graph_manifest": graph_manifest,
        "internal": internal,
        "scaffold": scaffold,
        "y": prev["y"],
    }


def mult_for_split(split: str, scaffold: Mapping[int, str]) -> dict[str, Any]:
    rng = np.random.default_rng(RC.BOOTSTRAP_SEED)
    rows = [int(x) for x in RC.SC.load_source("none")["rows"][split]]
    return RC.AN.cluster_multiplicity(rows, scaffold, rng=rng, replicates=RC.BOOTSTRAP_REPLICATES)


# ---------------------------------------------------------------------------
# A1 coverage & claim matrix
# ---------------------------------------------------------------------------


def coverage_matrix(pairs, caliper, scaffold) -> dict[str, Any]:
    c_links: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in caliper:
        c_links.setdefault((str(r["family"]), str(r["split"])), []).append(r)
    out: dict[str, Any] = {}
    for fam in RC.SC.FAMILIES:
        for split in ("valid", "test"):
            main = [p for p in pairs if str(p["family"]) == fam and str(p["split"]) == split and _i(p["pop_main"]) == 1]
            ret = [p for p in main if _i(p.get("n_c_candidates")) > 0]
            links = c_links.get((fam, split), [])
            c_views = {str(r["c_graph_id"]) for r in links}

            def census(sel: Sequence[Mapping[str, str]]) -> dict[str, Any]:
                mols = {int(p["row_id"]) for p in sel}
                insts = {(int(p["row_id"]), int(p["center"]), int(p["target"])) for p in sel}
                controls = {(int(p["row_id"]), int(p["center"]), int(p["target"]), int(p["control"])) for p in sel}
                clusters = {scaffold[int(p["row_id"])] for p in sel}
                by_cluster: dict[str, int] = {}
                for p in sel:
                    by_cluster[scaffold[int(p["row_id"])]] = by_cluster.get(scaffold[int(p["row_id"])], 0) + 1
                max_share = max(by_cluster.values()) / len(sel) if sel else 0.0
                return {
                    "molecules": len(mols), "clusters": len(clusters), "instances": len(insts),
                    "controls": len(controls), "t_pairs": len(sel), "max_cluster_share": float(max_share),
                }

            pre = census(main)
            post = census(ret)
            out[f"{fam}/{split}"] = {
                "pre_calibration": pre,
                "post_calibration": post,
                "c_links": len(links),
                "c_unique_views": len(c_views),
                "retention_ratio_pairs": (len(ret) / len(main)) if main else None,
                "retention_ratio_molecules": (post["molecules"] / pre["molecules"]) if pre["molecules"] else None,
                "note": "539 C links across all families are structure links, not independent molecules; "
                        "counts above are the per-family/split breakdown",
            }
    total_links = len(caliper)
    out["_total"] = {"c_links_all_families": total_links,
                     "c_unique_views_all_families": len({str(r["c_graph_id"]) for r in caliper}),
                     "note": "the C link total is not a molecule count"}
    return out


# ---------------------------------------------------------------------------
# A2 same-population three tables
# ---------------------------------------------------------------------------


def three_tables_quantities(pair_rows, all_pairs, caliper, mult, scaffold) -> dict[str, Any]:
    """J/D/B and the 7 mechanism terms for ALL_K0 / RETAINED_K0 / RETAINED_C."""
    c_by_pair: dict[str, list[str]] = {}
    for r in caliper:
        c_by_pair.setdefault(str(r["pair_id"]), []).append(str(r["c_graph_id"]))
    for k in c_by_pair:
        c_by_pair[k] = sorted(set(c_by_pair[k]))
    out: dict[str, Any] = {}
    for fam in RC.SC.FAMILIES:
        for split in ("valid", "test"):
            main = [r for r in pair_rows if str(r["family"]) == fam and str(r["split"]) == split and _i(r["pop_main"]) == 1]
            ret = [r for r in main if _i(r.get("n_c_candidates")) > 0]
            frames = {"ALL_K0": (main, "K"), "RETAINED_K0": (ret, "K"), "RETAINED_C": (ret, "C")}
            for pop, (sel, mode) in frames.items():
                key = f"{fam}/{split}/{pop}"
                if not sel:
                    out[key] = {"n_pairs": 0, "n_molecules": 0, "NA": "empty population"}
                    continue
                if mode == "K":
                    jk = f"J_K_{RC.AN.G_FIXED}"; dk = f"D_K_{RC.AN.G_FIXED}"; b = f"B_{RC.AN.G_FIXED}"
                    hj = "HJ"; dj = "DJ"
                    metric_keys = [jk, dk, b, f"pT_{RC.AN.G_FIXED}", f"pK_{RC.AN.G_FIXED}", f"aT_{RC.AN.G_FIXED}", f"aK_{RC.AN.G_FIXED}"] \
                        + [f"HJ_{t}" for t in TERMS] + [f"DJ_{t}" for t in TERMS]
                else:
                    jk = f"J_C_{RC.AN.G_FIXED}"; dk = f"D_C_{RC.AN.G_FIXED}"; b = f"B_{RC.AN.G_FIXED}"
                    hj = "HJ_C"; dj = "DJ_C"
                    metric_keys = [jk, dk, b, f"aT_{RC.AN.G_FIXED}", f"aC_{RC.AN.G_FIXED}"] \
                        + [f"HJ_C_{t}" for t in TERMS] + [f"DJ_C_{t}" for t in TERMS]
                mm = RC.AN.molecule_means(sel, tuple(metric_keys))
                block: dict[str, Any] = {
                    "n_pairs": len(sel), "n_molecules": len(mm),
                    "population_kind": "all main-queue pairs, compare original K0" if mode == "K" and pop == "ALL_K0"
                    else ("same focal T that have a C, compare original K0" if mode == "K"
                          else "identical focal T, compare the matched C"),
                }
                for q in (jk, dk, b, f"aT_{RC.AN.G_FIXED}"):
                    if any(q in d for d in mm.values()):
                        vals = {rid: d.get(q, np.nan) for rid, d in mm.items()}
                        block[q] = {"point": RC.AN.point_mean(vals),
                                    "ci95": RC.AN.pct_ci(RC.AN.bootstrap_mean(mult, vals))}
                mech = {}
                for space, prefix in (("H", hj), ("L", dj)):
                    terms = {}
                    total = 0.0
                    for t in TERMS:
                        vals = {rid: d.get(f"{prefix}_{t}", np.nan) for rid, d in mm.items()}
                        p = RC.AN.point_mean(vals)
                        terms[t] = {"point": p, "ci95": RC.AN.pct_ci(RC.AN.bootstrap_mean(mult, vals))}
                        total += p if np.isfinite(p) else 0.0
                    mech[space] = {"terms": terms, "sum_of_terms": total}
                block["mechanism"] = mech
                ref_key = jk
                summ = mech["H"]["sum_of_terms"]
                block["mechanism_sum_residual"] = abs(summ - RC.AN.point_mean({rid: d.get(ref_key, np.nan) for rid, d in mm.items()}))
                out[key] = block
    return out


# ---------------------------------------------------------------------------
# A3 conditional scalar / S2 recovery
# ---------------------------------------------------------------------------


def _recovery_values(internal_map, gid: str, p0: float, y: float) -> dict[str, float] | None:
    vals = {}
    for s in SUBSETS:
        key = (gid, s or "empty")
        if key not in internal_map:
            return None
        vals[s] = float(internal_map[key])
    # F-space: pV(S) - p0
    F = {s: vals[s] - p0 for s in SUBSETS}
    H = {s: abs(vals[s] - p0) for s in SUBSETS}
    L = {s: abs(vals[s] - y) - abs(p0 - y) for s in SUBSETS}
    out = {}
    for space, table in (("F", F), ("H", H), ("L", L)):
        r3 = table["123"] - table["12"]
        r2 = table["123"] - table["13"]
        g = {}
        for name, coefs in MOBIUS_COEF.items():
            g[name] = float(sum(c * table[s] for s, c in coefs.items()))
        out[f"R3_{space}"] = r3
        out[f"R2_{space}"] = r2
        out[f"R3_sum_{space}"] = g["g3"] + g["g13"] + g["g23"] + g["g123"]
        out[f"R2_sum_{space}"] = g["g2"] + g["g12"] + g["g23"] + g["g123"]
    return out


def recovery_analysis(analysis_dir: Path, pairs, caliper, internal, scaffold, y) -> dict[str, Any]:
    internal_map = {(str(r["graph_id"]), str(r["subset"])): float(r["pred_raw"]) for r in internal}
    # p0 per row from the frozen trained per-graph predictions (state trained, scope FULL kind O)
    pred_rows = RC.read_csv(analysis_dir / "per_graph_predictions.csv")
    p0 = {_i(r["row_id"]): float(r["pred_raw"]) for r in pred_rows if str(r["kind"]) == "O" and str(r["scope"]) == "FULL"}
    c_by_pair: dict[str, list[str]] = {}
    for r in caliper:
        c_by_pair.setdefault(str(r["pair_id"]), []).append(str(r["c_graph_id"]))
    for k in c_by_pair:
        c_by_pair[k] = sorted(set(c_by_pair[k]))

    rows: list[dict[str, Any]] = []
    residual_max = 0.0
    for p in pairs:
        if _i(p["pop_main"]) != 1:
            continue
        pid = str(p["pair_id"]); rid = _i(p["row_id"])
        t_gid, k_gid = str(p["t_graph_id"]), str(p["k_graph_id"])
        yy = float(y[rid]); p0v = p0[rid]
        rt = _recovery_values(internal_map, t_gid, p0v, yy)
        rk = _recovery_values(internal_map, k_gid, p0v, yy)
        if rt is None or rk is None:
            continue
        for space in ("F", "H", "L"):
            residual_max = max(residual_max, abs(rt[f"R3_{space}"] - rt[f"R3_sum_{space}"]))
            residual_max = max(residual_max, abs(rt[f"R2_{space}"] - rt[f"R2_sum_{space}"]))
            residual_max = max(residual_max, abs(rk[f"R3_{space}"] - rk[f"R3_sum_{space}"]))
            residual_max = max(residual_max, abs(rk[f"R2_{space}"] - rk[f"R2_sum_{space}"]))
        c_ids = c_by_pair.get(pid, [])
        c_recs = [_recovery_values(internal_map, g, p0v, yy) for g in c_ids]
        c_recs = [c for c in c_recs if c is not None]
        has_c = len(c_recs) > 0
        row: dict[str, Any] = {
            "family": str(p["family"]), "split": str(p["split"]), "pair_id": pid, "row_id": rid,
            "center": _i(p["center"]), "target": _i(p["target"]), "control": _i(p["control"]),
            "has_c": int(has_c),
        }
        for space in ("F", "H", "L"):
            row[f"R3_T_{space}"] = rt[f"R3_{space}"]
            row[f"R2_T_{space}"] = rt[f"R2_{space}"]
            row[f"R3_K_{space}"] = rk[f"R3_{space}"]
            row[f"R2_K_{space}"] = rk[f"R2_{space}"]
            row[f"R3_TminusK_{space}"] = rt[f"R3_{space}"] - rk[f"R3_{space}"]
            row[f"R2_TminusK_{space}"] = rt[f"R2_{space}"] - rk[f"R2_{space}"]
            if has_c:
                cR3 = float(np.mean([c[f"R3_{space}"] for c in c_recs]))
                cR2 = float(np.mean([c[f"R2_{space}"] for c in c_recs]))
                row[f"R3_C_{space}"] = cR3
                row[f"R2_C_{space}"] = cR2
                row[f"R3_TminusC_{space}"] = rt[f"R3_{space}"] - cR3
                row[f"R2_TminusC_{space}"] = rt[f"R2_{space}"] - cR2
        rows.append(row)

    # aggregate per family/split/population
    out: dict[str, Any] = {}
    mult_cache: dict[str, Any] = {}
    frames_removed = None  # kept for readability: the population selection is inline below
    for fam in RC.SC.FAMILIES:
        for split in ("valid", "test"):
            if split not in mult_cache:
                mult_cache[split] = mult_for_split(split, scaffold)
            mult = mult_cache[split]
            for pop, pred in (("ALL_K0", lambda r: True), ("RETAINED_K0", lambda r: r["has_c"] == 1)):
                sel = [r for r in rows if r["family"] == fam and r["split"] == split and pred(r)]
                if not sel:
                    out[f"{fam}/{split}/{pop}"] = {"n_pairs": 0}
                    continue
                block: dict[str, Any] = {"n_pairs": len(sel), "comparison": "T minus K0"}
                block.update(_aggregate_recovery(sel, ("R3_T", "R2_T", "R3_TminusK", "R2_TminusK"), mult))
                out[f"{fam}/{split}/{pop}"] = block
            selc = [r for r in rows if r["family"] == fam and r["split"] == split and r["has_c"] == 1]
            if selc:
                block = {"n_pairs": len(selc), "comparison": "T minus mean C"}
                block.update(_aggregate_recovery(selc, ("R3_T", "R2_T", "R3_TminusC", "R2_TminusC"), mult))
                out[f"{fam}/{split}/RETAINED_C"] = block
            else:
                out[f"{fam}/{split}/RETAINED_C"] = {"n_pairs": 0}

    return {
        "definition": {
            "R3": "value(123)-value(12) = gamma3+gamma13+gamma23+gamma123 (conditional scalar recovery)",
            "R2": "value(123)-value(13) = gamma2+gamma12+gamma23+gamma123 (conditional S2 recovery)",
            "spaces": {"F": "pV(S) - p0 (raw prediction)", "H": "|pV(S)-p0| (response, J endpoint)",
                        "L": "|pV(S)-y0| - |p0-y0| (reference deviation, D endpoint)"},
            "populations": {"ALL_K0": "all main-queue pairs vs original K0",
                            "RETAINED_K0": "focal T with a C vs original K0",
                            "RETAINED_C": "identical focal T vs matched C"},
        },
        "arithmetic_residual_max": float(residual_max),
        "arithmetic_residual_tolerance": 1e-5,
        "arithmetic_pass": bool(residual_max <= 1e-5),
        "per_pair": rows,
        "aggregate": out,
    }


def _aggregate_recovery(sel, quantities, mult) -> dict[str, Any]:
    """Molecule-equal aggregation of recovery quantities, with per-view absolute means."""
    # build synthetic molecule keys preserving the (row, instance, control) hierarchy:
    # a molecule may carry several pairs; use the frozen molecule_means by row_id.
    keys: list[str] = []
    for q in quantities:
        for space in ("F", "H", "L"):
            keys.append(f"{q}_{space}")
    transformed = []
    for r in sel:
        base = {"row_id": r["row_id"], "center": r["center"], "target": r["target"], "control": r["control"]}
        d = dict(base)
        for k in keys:
            d[k] = r.get(k)
        transformed.append(d)
    mm = RC.AN.molecule_means(transformed, tuple(keys))
    out: dict[str, Any] = {}
    for q in quantities:
        for space in ("F", "H", "L"):
            k = f"{q}_{space}"
            vals = {rid: d.get(k, np.nan) for rid, d in mm.items()}
            reps = RC.AN.bootstrap_mean(mult, vals)
            # per-view absolute recovery (no cancellation)
            abs_vals = np.asarray([abs(float(r[k])) for r in sel if np.isfinite(_f(r.get(k)))], dtype=np.float64)
            out[k] = {
                "point": RC.AN.point_mean(vals),
                "ci95": RC.AN.pct_ci(reps),
                "abs_per_view_mean": float(abs_vals.mean()) if abs_vals.size else float("nan"),
                "abs_per_view_median": float(np.median(abs_vals)) if abs_vals.size else float("nan"),
                "n_views": int(abs_vals.size),
            }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=str(RC.SOURCE_ANALYSIS))
    args = ap.parse_args(argv)
    analysis_dir = Path(args.analysis_dir)
    RC.log("Part A: loading frozen multi-motif artifacts ...")
    data = load_source_inputs()
    pairs = data["pairs"]
    caliper = data["caliper"]
    internal = data["internal"]
    scaffold = data["scaffold"]
    y = data["y"]

    # frozen per-pair quantities (J/D/B + 7 mechanism terms)
    pred_t = {(str(r["graph_id"]), str(r["scope"])): float(r["pred_raw"]) for r in RC.read_csv(analysis_dir / "per_graph_predictions.csv")}
    pred_init = {("__init__" + str(r["graph_id"]), str(r["scope"])): float(r["pred_raw"]) for r in RC.read_csv(analysis_dir / "per_graph_predictions_init.csv")}
    pred = {**pred_t, **pred_init}
    internal_map = {(str(r["graph_id"]), str(r["subset"])): float(r["pred_raw"]) for r in internal}
    pair_rows = RC.AN.build_pair_quantities(pairs=pairs, caliper=caliper, pred=pred, y=y, internal=internal_map)
    RC.log(f"Part A: {len(pair_rows)} frozen per-pair quantity rows")

    coverage = coverage_matrix(pairs, caliper, scaffold)
    RC.log("Part A: A1 coverage matrix done")

    mult_cache = {"valid": mult_for_split("valid", scaffold), "test": mult_for_split("test", scaffold)}
    tables = three_tables_quantities(pair_rows, pairs, caliper, mult_cache["test"], scaffold)
    # valid uses its own multiplicity
    tables_valid = three_tables_quantities(pair_rows, pairs, caliper, mult_cache["valid"], scaffold)
    for k, v in tables_valid.items():
        if "/valid/" in k:
            tables[k] = v
    RC.log("Part A: A2 three tables done")

    recovery = recovery_analysis(analysis_dir, pairs, caliper, internal, scaffold, y)
    RC.log(f"Part A: A3 recovery done (arithmetic residual {recovery['arithmetic_residual_max']:.2e})")

    payload = {
        "round_id": RC.ROUND_ID,
        "generated_utc": RC.utcnow(),
        "git_commit": RC.git_commit(),
        "source_round": {
            "track": RC.track_rel(RC.SOURCE_TRACK),
            "protocol_revision": RC.SOURCE_PROTOCOL_REVISION,
            "report_revision": RC.SOURCE_REPORT_REVISION,
            "inputs": {
                "pair_catalog.csv": RC.sha256_file(analysis_dir / "pair_catalog.csv"),
                "caliper_catalog.csv": RC.sha256_file(analysis_dir / "caliper_catalog.csv"),
                "per_graph_predictions.csv": RC.sha256_file(analysis_dir / "per_graph_predictions.csv"),
                "per_graph_predictions_init.csv": RC.sha256_file(analysis_dir / "per_graph_predictions_init.csv"),
                "per_graph_internal.csv": RC.sha256_file(analysis_dir / "per_graph_internal.csv"),
                "uncertainty.json": RC.sha256_file(analysis_dir / "uncertainty.json"),
                "mechanism.json": RC.sha256_file(analysis_dir / "mechanism.json"),
            },
        },
        "A1_coverage_and_claim_matrix": coverage,
        "A2_same_population_three_tables": tables,
        "A3_conditional_recovery": {k: v for k, v in recovery.items() if k != "per_pair"},
    }
    RC.write_json(RC.NEW_ANALYSIS / "partA_clarifications.json", payload)
    # A3 per-pair table
    RC.write_csv(RC.NEW_ANALYSIS / "partA_recovery_pairs.csv",
                 recovery["per_pair"], list(recovery["per_pair"][0].keys()) if recovery["per_pair"] else [])
    RC.log("Part A complete: analysis/partA_clarifications.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
