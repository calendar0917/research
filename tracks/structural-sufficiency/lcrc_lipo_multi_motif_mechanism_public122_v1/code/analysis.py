"""Stage C: statistics, internal-path (Mobius) decomposition, natural pairs, tables.

Everything is in raw logD.  For one molecule ``m``, target instance ``j``,
control ``k`` and donor ``d`` (the frozen pair catalogue fixes the weights):

``p0 = f(O)``, ``pT = f(T)``, ``pK = f(K0)``, ``pC`` = the prediction of one
dose-calipered control view of the same molecule.

``aT = |pT-p0|``, ``aK = |pK-p0|``, ``aC = mean_c |pC-p0|``
``loss_X = |pX - y|`` with ``y`` the **original molecule's** label.

``J_K = aT-aK``       extra movement when the target is destroyed instead of kept
``J_C = aT-aC``       extra movement relative to equal-statistical-dose rewires
``D_K = lossT-lossK``, ``D_C = lossT-lossC``, ``B = lossT-loss0``

``rC = (aT-aC)/(aT+aC)`` (0 when the denominator is <= 1e-12), ``rK`` likewise,
and the parameter-dependence quantity is ``E = rC_TRAINED - rC_INIT`` on the
*identical* views / C set / weights / queue.

Weights: molecules equal; instances equal inside a molecule; controls equal
inside an instance; donors equal inside a control; C views equal inside a focal
T.  Never averaged predictions before the loss.

Uncertainty: one cluster bootstrap per split (Bemis-Murcko scaffold clusters of
the whole split, 5000 replicates, seed 20261114) whose multiplicity is **shared**
by every family / scope / parameter state / mechanism term of that split.

The internal-path terms are the Mobius coefficients of the three-factor
``{G1,G2,G3}`` intervention (see ``infer.py``), summed over the seven non-empty
subsets; their sum reproduces the full-input difference to 1e-5 by construction
and their *differences* (T minus K0, T minus mean C) reproduce ``J`` / ``D``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import common as C
import prepare as P

FULL = "FULL"
G_FIXED = "G_FIXED"
PRIMARY_SCOPE = G_FIXED  # main selectivity / internal-path analysis uses the frozen global42
POP_MAIN = "main"
POP_ALL = "struct_chem_all"
POP_CAL = "calibration"

SUBSET_ORDER = ("", "1", "2", "3", "12", "13", "23", "123")
TERMS = ("g1", "g2", "g3", "g12", "g13", "g23", "g123")
TERM_OF_SUBSET = {"1": "g1", "2": "g2", "3": "g3", "12": "g12", "13": "g13", "23": "g23", "123": "g123"}
MOBIUS_COEF = {
    "g1": {"1": 1.0},
    "g2": {"2": 1.0},
    "g3": {"3": 1.0},
    "g12": {"12": 1.0, "1": -1.0, "2": -1.0},
    "g13": {"13": 1.0, "1": -1.0, "3": -1.0},
    "g23": {"23": 1.0, "2": -1.0, "3": -1.0},
    "g123": {"123": 1.0, "12": -1.0, "13": -1.0, "23": -1.0, "1": 1.0, "2": 1.0, "3": 1.0},
}

N_PRIMARY_ITEMS = 12  # 6 families x (rC_TRAINED, E) on Test
ALPHA = 0.05


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    return P.write_csv(path, rows, columns)


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        s = str(x).strip()
        return float(s) if s not in ("", "nan", "None") else default
    except Exception:
        return default


def _i(x: Any, default: int = 0) -> int:
    v = _f(x, float("nan"))
    return default if not np.isfinite(v) else int(v)


def load_inputs(analysis_dir: Path) -> dict[str, Any]:
    pairs = read_csv(analysis_dir / "pair_catalog.csv")
    caliper = read_csv(analysis_dir / "caliper_catalog.csv")
    preds_trained = read_csv(analysis_dir / "per_graph_predictions.csv")
    preds_init = read_csv(analysis_dir / "per_graph_predictions_init.csv")
    internal = read_csv(analysis_dir / "per_graph_internal.csv")
    manifest = C.read_json(analysis_dir / "inputs_manifest.json")
    previous = C.load_previous_inputs()
    return {
        "pairs": pairs,
        "caliper": caliper,
        "pred_trained": preds_trained,
        "pred_init": preds_init,
        "internal": internal,
        "manifest": manifest,
        "previous": previous,
    }


def prediction_map(rows: Iterable[Mapping[str, Any]], scope: str | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        if scope is not None and str(r["scope"]) != scope:
            continue
        out[str(r["graph_id"])] = float(r["pred_raw"])
    return out


def internal_map(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    return {(str(r["graph_id"]), str(r["subset"])): float(r["pred_raw"]) for r in rows}


# ---------------------------------------------------------------------------
# per-pair quantities
# ---------------------------------------------------------------------------


def build_pair_quantities(
    *,
    pairs: Sequence[Mapping[str, str]],
    caliper: Sequence[Mapping[str, str]],
    pred: Mapping[tuple[str, str], float],
    y: Mapping[int, float],
    internal: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """One row per frozen pair with all per-view-derived quantities and Mobius terms."""
    c_by_pair: dict[str, list[str]] = {}
    for r in caliper:
        c_by_pair.setdefault(str(r["pair_id"]), []).append(str(r["c_graph_id"]))
    for k in c_by_pair:
        c_by_pair[k] = sorted(set(c_by_pair[k]))
    out: list[dict[str, Any]] = []
    for pr in pairs:
        if int(pr["pop_struct_chem"] or 0) != 1:
            # structurally / chemically rejected: kept in the catalogue only
            continue
        pid = str(pr["pair_id"])
        rid = int(pr["row_id"])
        yy = float(y[rid])
        p0 = pred[(str(pr["o_graph_id"]), FULL)]
        row: dict[str, Any] = {
            "pair_id": pid,
            "family": str(pr["family"]),
            "row_id": rid,
            "split": str(pr["split"]),
            "instance_id": str(pr["instance_id"]),
            "center": int(pr["center"]),
            "target": int(pr["target"]),
            "control": int(pr["control"]),
            "scaffold_key": str(pr["scaffold_key"]),
            "scaffold_seen_in_train": int(pr["scaffold_seen_in_train"]),
            "pop_main": int(pr["pop_main"]),
            "pop_struct_chem": int(pr["pop_struct_chem"]),
            "n_c_candidates": len(c_by_pair.get(pid, [])),
            "y": yy,
            "p0": p0,
            "loss0": abs(p0 - yy),
        }
        for scope in (FULL, G_FIXED):
            pT = pred[(str(pr["t_graph_id"]), scope)]
            pK = pred[(str(pr["k_graph_id"]), scope)]
            row[f"pT_{scope}"] = pT
            row[f"pK_{scope}"] = pK
            row[f"aT_{scope}"] = abs(pT - p0)
            row[f"aK_{scope}"] = abs(pK - p0)
            row[f"lossT_{scope}"] = abs(pT - yy)
            row[f"lossK_{scope}"] = abs(pK - yy)
            row[f"J_K_{scope}"] = row[f"aT_{scope}"] - row[f"aK_{scope}"]
            row[f"D_K_{scope}"] = row[f"lossT_{scope}"] - row[f"lossK_{scope}"]
            row[f"B_{scope}"] = row[f"lossT_{scope}"] - row["loss0"]
        # INIT state (G_FIXED only: the same views and the same global42 handling)
        p0_init = pred[("__init__" + str(pr["o_graph_id"]), G_FIXED)] if ("__init__" + str(pr["o_graph_id"]), G_FIXED) in pred else p0
        pT_init = pred[("__init__" + str(pr["t_graph_id"]), G_FIXED)]
        pK_init = pred[("__init__" + str(pr["k_graph_id"]), G_FIXED)]
        row["p0_init"] = p0_init
        row["aT_init"] = abs(pT_init - p0_init)
        row["aK_init"] = abs(pK_init - p0_init)
        # C views
        c_ids = c_by_pair.get(pid, [])
        if c_ids:
            for scope in (FULL, G_FIXED):
                aC = float(np.mean([abs(pred[(g, scope)] - p0) for g in c_ids]))
                lossC = float(np.mean([abs(pred[(g, scope)] - yy) for g in c_ids]))
                row[f"aC_{scope}"] = aC
                row[f"lossC_{scope}"] = lossC
                row[f"J_C_{scope}"] = row[f"aT_{scope}"] - aC
                row[f"D_C_{scope}"] = row[f"lossT_{scope}"] - lossC
            row["pC_mean_G_FIXED"] = float(np.mean([pred[(g, G_FIXED)] for g in c_ids]))
            row["pC_mean_FULL"] = float(np.mean([pred[(g, FULL)] for g in c_ids]))
            aC_init = float(np.mean([abs(pred[("__init__" + g, G_FIXED)] - p0_init) for g in c_ids]))
            row["aC_init"] = aC_init
            row["J_C_init"] = row["aT_init"] - aC_init
            row["rC"] = _ratio(row[f"aT_{G_FIXED}"], aC, init=False)
            row["rC_init"] = _ratio(row["aT_init"], aC_init, init=True)
            row["rK"] = _ratio(row[f"aT_{G_FIXED}"], row[f"aK_{G_FIXED}"], init=False)
            row["rK_init"] = _ratio(row["aT_init"], row["aK_init"], init=True)
        # mechanism: Mobius over the 8 frozen internal cells (G_FIXED / global42 = O)
        terms = mobius_terms(internal, str(pr["t_graph_id"]), p0=p0, y=yy)
        k_terms = mobius_terms(internal, str(pr["k_graph_id"]), p0=p0, y=yy)
        row["mobius_available"] = int(terms is not None and k_terms is not None)
        if terms is not None and k_terms is not None:
            for name, value in terms["H"].items():
                row[f"HT_{name}"] = value
                row[f"HK_{name}"] = k_terms["H"][name]
                row[f"HJ_{name}"] = value - k_terms["H"][name]
            for name, value in terms["L"].items():
                row[f"LT_{name}"] = value
                row[f"LK_{name}"] = k_terms["L"][name]
                row[f"DJ_{name}"] = value - k_terms["L"][name]
            row["H_123_minus_13_T"] = terms["H"]["g123"] + terms["H"]["g12"] + terms["H"]["g1"] + terms["H"]["g2"] + terms["H"]["g3"] + terms["H"]["g13"] + terms["H"]["g23"]
            row["H_s2_recovery_123_vs_13_T"] = terms["raw"]["H"]["123"] - terms["raw"]["H"]["13"]
            row["H_s2_recovery_123_vs_23_T"] = terms["raw"]["H"]["123"] - terms["raw"]["H"]["23"]
        if c_ids:
            cH = [mobius_terms(internal, g, p0=p0, y=yy) for g in c_ids]
            cH = [t for t in cH if t is not None]
            if cH and terms is not None:
                for name in TERMS:
                    mt = float(np.mean([t["H"][name] for t in cH]))
                    row[f"HC_{name}"] = mt
                    row[f"HJ_C_{name}"] = terms["H"][name] - mt
                    ml = float(np.mean([t["L"][name] for t in cH]))
                    row[f"DJ_C_{name}"] = terms["L"][name] - ml
        out.append(row)
    return out


def _ratio(aT: float, aX: float, *, init: bool) -> float:
    den = float(aT) + float(aX)
    if not np.isfinite(den) or den <= C.R_DENOM_EPS:
        return 0.0
    return float((float(aT) - float(aX)) / den)


def mobius_terms(
    internal: Mapping[tuple[str, str], float], gid: str, *, p0: float, y: float
) -> dict[str, dict[str, float]] | None:
    """Mobius coefficients (7 terms) of F / H / L for one internal view."""
    raw_H: dict[str, float] = {}
    raw_L: dict[str, float] = {}
    raw_F: dict[str, float] = {}
    for subset in SUBSET_ORDER:
        key = (str(gid), subset or "empty")
        if key not in internal:
            return None
        p = float(internal[key])
        raw_F[subset] = p - p0
        raw_H[subset] = abs(p - p0)
        raw_L[subset] = abs(p - y) - abs(p0 - y)
    out: dict[str, dict[str, float]] = {"F": {}, "H": {}, "L": {}, "raw": {}}
    for name, coefs in MOBIUS_COEF.items():
        for space, table in (("F", raw_F), ("H", raw_H), ("L", raw_L)):
            out[space][name] = float(sum(c * table[s] for s, c in coefs.items()))
    out["raw"] = {"F": raw_F, "H": raw_H, "L": raw_L}
    return out


# ---------------------------------------------------------------------------
# weights, aggregation, bootstrap
# ---------------------------------------------------------------------------


def molecule_means(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[int, dict[str, float]]:
    """Molecule-equal, instance-equal, control-equal, donor-equal aggregation."""
    per: dict[int, dict[str, dict[str, list[float]]]] = {}
    for r in rows:
        rid = int(r["row_id"])
        iid = (int(r["center"]), int(r["target"]), int(r["control"]))
        for k in keys:
            v = r.get(k)
            if v is None:
                continue
            fv = _f(v)
            if not np.isfinite(fv):
                continue
            per.setdefault(rid, {}).setdefault(iid, {}).setdefault(k, []).append(fv)
    out: dict[int, dict[str, float]] = {}
    for rid, inst_map in per.items():
        per_inst: dict[tuple[int, int, int], dict[str, float]] = {}
        for iid, blocks in inst_map.items():
            per_inst[iid] = {k: float(np.mean(v)) for k, v in blocks.items()}
        out[rid] = {}
        for k in keys:
            vals = [per_inst[i][k] for i in per_inst if k in per_inst[i]]
            if vals:
                out[rid][k] = float(np.mean(vals))
    return out


def cluster_multiplicity(
    rows_all: Sequence[int], scaffold: Mapping[int, str], *, rng: np.random.Generator, replicates: int
) -> dict[str, Any]:
    """One shared scaffold-cluster multiplicity for the whole split."""
    rows = np.asarray([int(r) for r in rows_all], dtype=np.int64)
    keys = [scaffold[int(r)] for r in rows.tolist()]
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    cluster_idx = np.asarray([uniq[k] for k in keys], dtype=np.int64)
    n_clusters = len(uniq)
    draws = rng.integers(0, n_clusters, size=(int(replicates), n_clusters))
    counts = np.zeros((int(replicates), n_clusters), dtype=np.int64)
    for b in range(int(replicates)):
        counts[b] = np.bincount(draws[b], minlength=n_clusters)
    weights = counts[:, cluster_idx].astype(np.float64)
    sizes = np.bincount(cluster_idx, minlength=n_clusters)
    return {
        "rows": rows,
        "weights": weights,
        "n_clusters": int(n_clusters),
        "cluster_index": cluster_idx,
        "cluster_sizes": sizes,
        "max_cluster_share": float(sizes.max() / sizes.sum()) if sizes.size else 0.0,
        "max_cluster_key": sorted(uniq)[int(np.argmax(sizes))] if sizes.size else "",
        "row_column": {int(r): j for j, r in enumerate(rows.tolist())},
    }


def bootstrap_mean(
    mult: Mapping[str, Any], values: Mapping[int, float]
) -> np.ndarray:
    """Weighted mean over the molecules that carry a finite value; NaN if mass 0."""
    rows = np.asarray(mult["rows"], dtype=np.int64)
    W = np.asarray(mult["weights"], dtype=np.float64)
    v = np.asarray([float(values.get(int(r), np.nan)) for r in rows.tolist()], dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(W.shape[0], np.nan)
    w = W[:, ok]
    num = w @ v[ok]
    den = w.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def point_mean(values: Mapping[int, float]) -> float:
    v = np.asarray([x for x in values.values() if np.isfinite(x)], dtype=np.float64)
    return float(v.mean()) if v.size else float("nan")


def pct_ci(reps: np.ndarray, level: float = 95.0) -> list[float]:
    reps = np.asarray(reps, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return [float("nan"), float("nan")]
    lo = (100.0 - level) / 2.0
    return [float(np.percentile(reps, lo)), float(np.percentile(reps, 100.0 - lo))]


def quantile_ci(reps: np.ndarray, q_lo: float, q_hi: float) -> list[float]:
    reps = np.asarray(reps, dtype=np.float64)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(reps, 100.0 * q_lo)), float(np.percentile(reps, 100.0 * q_hi))]


# ---------------------------------------------------------------------------
# families / populations
# ---------------------------------------------------------------------------


def family_rows(
    pair_rows: Sequence[Mapping[str, Any]], family: str, split: str, population: str
) -> list[Mapping[str, Any]]:
    if population == POP_CAL:
        sel = [r for r in pair_rows if int(r["pop_main"]) == 1 and int(r["n_c_candidates"]) > 0]
    elif population == POP_MAIN:
        sel = [r for r in pair_rows if int(r["pop_main"]) == 1]
    elif population == POP_ALL:
        sel = [r for r in pair_rows if int(r["pop_struct_chem"]) == 1]
    else:
        raise KeyError(population)
    return [r for r in sel if str(r["family"]) == family and str(r["split"]) == split]


QUANT_KEYS = tuple(
    [f"{q}_{s}" for q in ("aT", "aK", "lossT", "lossK", "J_K", "D_K", "B") for s in (FULL, G_FIXED)]
    + ["aT_init", "aK_init", "rC", "rC_init", "rK", "rK_init", "J_C_init"]
)


def calibration_keys() -> tuple[str, ...]:
    return tuple(
        [f"{q}_{s}" for q in ("aT", "aC", "J_C", "D_C", "lossC") for s in (FULL, G_FIXED)]
        + ["aC_init", "J_C_init"]
    )


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901
    analysis_dir = C.ANALYSIS
    data = load_inputs(analysis_dir)
    manifest = data["manifest"]
    previous = data["previous"]
    y = previous["y"]
    import graphlib as G

    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    scaffold = {rid: G.bemis_murcko_key(str(raw_rows[rid]["smiles"])) for rid in range(C.N_MOLECULES)}
    recorded = {int(r["row_id"]): str(r["scaffold_key"]) for r in read_csv(analysis_dir / "family_catalog.csv")}
    bad = [rid for rid, key in recorded.items() if scaffold.get(rid) != key]
    if bad:
        raise SystemExit(f"SCAFFOLD_KEY_MISMATCH: {bad[:10]}")

    pred_t = {
        (str(r["graph_id"]), str(r["scope"])): float(r["pred_raw"]) for r in data["pred_trained"]
    }
    pred_init = {
        ("__init__" + str(r["graph_id"]), str(r["scope"])): float(r["pred_raw"]) for r in data["pred_init"]
    }
    pred = {**pred_t, **pred_init}
    internal = internal_map(data["internal"])

    pair_rows = build_pair_quantities(
        pairs=data["pairs"], caliper=data["caliper"], pred=pred, y=y, internal=internal
    )
    C.log(f"per-pair quantities: {len(pair_rows)} pairs")

    # ---- shared multiplicities per split ------------------------------
    src = C.load_source("none")
    mult: dict[str, Any] = {}
    for split in ("valid", "test"):
        rng = np.random.default_rng(C.BOOTSTRAP_SEED)
        mult[split] = cluster_multiplicity(
            [int(r) for r in src["rows"][split]], scaffold, rng=rng, replicates=C.BOOTSTRAP_REPLICATES
        )
        C.log(
            f"{split}: {mult[split]['n_clusters']} scaffold clusters, "
            f"max share {mult[split]['max_cluster_share']:.3f}"
        )

    all_keys = tuple(sorted({k for r in pair_rows for k in r if isinstance(r.get(k), float)}))

    results: list[dict[str, Any]] = []
    uncertainty: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "bootstrap_replicates": C.BOOTSTRAP_REPLICATES,
        "cluster_definition": "Bemis-Murcko scaffold of the original molecule (shared multiplicity per split)",
        "primary_scope": PRIMARY_SCOPE,
        "bonferroni": {"n_items": N_PRIMARY_ITEMS, "alpha": ALPHA},
        "splits": {},
        "families": {},
    }

    # per-split per-scope per-population per-family point + bootstrap
    reps_store: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    mol_store: dict[tuple[str, str, str, str], dict[int, dict[str, float]]] = {}
    for split in ("valid", "test"):
        for population in (POP_MAIN, POP_ALL, POP_CAL):
            for family in C.FAMILIES:
                rows = family_rows(pair_rows, family, split, population)
                if not rows:
                    continue
                keys = tuple(sorted({k for r in rows for k in all_keys}))
                mm = molecule_means(rows, keys)
                mol_store[(population, split, family)] = mm
                n_mol = len(mm)
                for key in keys:
                    values = {rid: d.get(key, float("nan")) for rid, d in mm.items()}
                    point = point_mean(values)
                    reps = bootstrap_mean(mult[split], values)
                    reps_store[(population, split, family, key)] = reps
                    lo, hi = pct_ci(reps)
                    n_finite = int(np.isfinite(reps).sum())
                    results.append(
                        {
                            "population": population,
                            "split": split,
                            "family": family,
                            "scope": key.rsplit("_", 1)[-1] if key.endswith(("FULL", "G_FIXED")) else "",
                            "metric": key,
                            "value": point,
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "n_molecules": n_mol,
                            "n_pairs": len(rows),
                            "n_replicates_finite": n_finite,
                        }
                    )

    # ---- primary Test items: rC_TRAINED > 0 and E > 0 -------------------
    q_lo = ALPHA / (2.0 * N_PRIMARY_ITEMS)
    q_hi = 1.0 - q_lo
    primary: dict[str, Any] = {"bonferroni_quantiles": [q_lo, q_hi], "items": {}}
    family_support: dict[str, Any] = {}
    for family in C.FAMILIES:
        for split in ("valid", "test"):
            for population in (POP_CAL, POP_MAIN):
                rows = family_rows(pair_rows, family, split, population)
                family_support.setdefault(family, {})[f"{population}/{split}"] = {
                    "molecules": len({int(r["row_id"]) for r in rows}),
                    "instances": len({(int(r["row_id"]), int(r["center"]), int(r["target"])) for r in rows}),
                    "controls": len({(int(r["row_id"]), int(r["center"]), int(r["target"]), int(r["control"])) for r in rows}),
                    "pairs": len(rows),
                }
        key_t = (POP_CAL, "test", family)
        if key_t not in mol_store:
            primary["items"][family] = {
                "supported": False,
                "reason": "no Test molecule with a dose-calipered C control (competition not excluded)",
            }
            continue
        mm = mol_store[key_t]
        rC = {rid: d.get("rC", float("nan")) for rid, d in mm.items()}
        rC_init = {rid: d.get("rC_init", float("nan")) for rid, d in mm.items()}
        E = {rid: (d.get("rC", np.nan) - d.get("rC_init", np.nan)) for rid, d in mm.items()}
        rC_p, rC_lo, rC_hi = point_mean(rC), *pct_ci(reps_store[(POP_CAL, "test", family, "rC")])
        E_point = point_mean(E)
        reps_E = reps_store[(POP_CAL, "test", family, "rC")] - reps_store[(POP_CAL, "test", family, "rC_init")]
        E_lo, E_hi = pct_ci(reps_E)
        rC_bonf = quantile_ci(reps_store[(POP_CAL, "test", family, "rC")], q_lo, q_hi)
        E_bonf = quantile_ci(reps_E, q_lo, q_hi)
        primary["items"][family] = {
            "supported": True,
            "split": "test",
            "population": POP_CAL,
            "scope": G_FIXED,
            "n_molecules": len(mm),
            "n_pairs": len(family_rows(pair_rows, family, "test", POP_CAL)),
            "rC_trained": {"point": rC_p, "ci95": [rC_lo, rC_hi], "bonferroni": rC_bonf},
            "E": {"point": E_point, "ci95": [E_lo, E_hi], "bonferroni": E_bonf},
            "rC_trained_positive": bool(np.isfinite(rC_lo) and rC_lo > 0),
            "E_positive": bool(np.isfinite(E_lo) and E_lo > 0),
            "rC_bonferroni_excludes_zero": bool(
                np.isfinite(rC_bonf[0]) and np.isfinite(rC_bonf[1]) and (rC_bonf[0] > 0 or rC_bonf[1] < 0)
            ),
            "E_bonferroni_excludes_zero": bool(
                np.isfinite(E_bonf[0]) and np.isfinite(E_bonf[1]) and (E_bonf[0] > 0 or E_bonf[1] < 0)
            ),
        }
    uncertainty["primary_test"] = primary
    uncertainty["family_support"] = family_support

    # ---- Valid as a pre-registered direction check ---------------------
    valid_check: dict[str, Any] = {}
    for family in C.FAMILIES:
        key_v = (POP_CAL, "valid", family)
        if key_v not in mol_store:
            valid_check[family] = {"supported": False}
            continue
        reps = reps_store[(POP_CAL, "valid", family, "rC")]
        reps_init = reps_store[(POP_CAL, "valid", family, "rC_init")]
        valid_check[family] = {
            "supported": True,
            "rC_trained": point_mean({rid: d.get("rC", np.nan) for rid, d in mol_store[key_v].items()}),
            "rC_init": point_mean({rid: d.get("rC_init", np.nan) for rid, d in mol_store[key_v].items()}),
            "rC_ci95": pct_ci(reps),
            "E_point": point_mean(
                {rid: d.get("rC", np.nan) - d.get("rC_init", np.nan) for rid, d in mol_store[key_v].items()}
            ),
            "E_ci95": pct_ci(reps - reps_init),
        }
    uncertainty["valid_direction_check"] = valid_check

    # ---- influence checks ---------------------------------------------
    influence = influence_checks(pair_rows, mol_store, mult, reps_store)
    uncertainty["influence"] = influence

    # ---- independent re-verification of the dose caliper ----------------
    caliper_check = verify_calipers(analysis_dir, data["caliper"], data["pairs"], src)
    uncertainty["caliper_verification"] = caliper_check
    if not caliper_check["pass"]:
        raise SystemExit(f"CALIPER_VERIFICATION_FAILED: {caliper_check['failures'][:3]}")

    # ---- impact of the C pool composition -----------------------------
    uncertainty["caliper_composition"] = caliper_composition(data["caliper"], data["pairs"])
    uncertainty["denominator_guard"] = denominator_guard(pair_rows)

    # ---- fixed Mobius endpoint identities ------------------------------
    analysis_checks = mobius_identity_checks(pair_rows)

    # ---- secondary transparency: focal pairs whose C set is not their own K0
    uncertainty["foreign_c_only"] = foreign_c_only_estimates(
        pair_rows, data["pairs"], data["caliper"], mult
    )

    # ---- mechanism tables ---------------------------------------------
    mechanism = mechanism_tables(pair_rows, mol_store, mult)
    analysis_checks["mechanism_terms_sum_matches_total"] = mechanism["sum_check"]

    # ---- natural pairs (MMP) ------------------------------------------
    mmp = natural_pairs(analysis_dir, scaffold, previous, read_csv)

    # ---- scaffold stratification --------------------------------------
    strat = scaffold_strata(pair_rows, mult)

    # ---- write outputs -------------------------------------------------
    write_csv(
        analysis_dir / "per_pair_quantities.csv",
        pair_rows,
        ["pair_id", "family", "row_id", "split", "instance_id", "center", "target", "control",
         "scaffold_key", "scaffold_seen_in_train", "pop_main", "pop_struct_chem", "n_c_candidates",
         "y", "p0", "loss0",
         "pT_FULL", "pK_FULL", "aT_FULL", "aK_FULL", "lossT_FULL", "lossK_FULL", "J_K_FULL", "D_K_FULL", "B_FULL",
         "pT_G_FIXED", "pK_G_FIXED", "aT_G_FIXED", "aK_G_FIXED", "lossT_G_FIXED", "lossK_G_FIXED",
         "J_K_G_FIXED", "D_K_G_FIXED", "B_G_FIXED",
         "p0_init", "aT_init", "aK_init",
         "aC_FULL", "lossC_FULL", "J_C_FULL", "D_C_FULL", "aC_G_FIXED", "lossC_G_FIXED", "J_C_G_FIXED", "D_C_G_FIXED",
         "aC_init", "J_C_init", "pC_mean_G_FIXED", "pC_mean_FULL", "rC", "rC_init", "rK", "rK_init",
         "mobius_available"],
    )
    write_csv(
        analysis_dir / "molecule_summary.csv",
        molecule_rows(mol_store),
        ["population", "split", "family", "row_id", "scaffold_key", "scaffold_seen_in_train",
         "n_instances", "n_pairs"] + list(sorted({k for mm in mol_store.values() for d in mm.values() for k in d})),
    )
    write_csv(
        analysis_dir / "results.csv",
        results,
        ["population", "split", "family", "scope", "metric", "value", "ci95_low", "ci95_high",
         "n_molecules", "n_pairs", "n_replicates_finite"],
    )
    C.write_json(analysis_dir / "analysis_checks.json", analysis_checks)
    C.write_json(analysis_dir / "uncertainty.json", uncertainty)
    C.write_json(analysis_dir / "mechanism.json", mechanism)
    write_csv(
        analysis_dir / "mechanism_terms.csv",
        mechanism["rows"],
        ["split", "population", "family", "space", "comparison", "term", "point", "ci95_low", "ci95_high", "n_molecules"],
    )
    write_csv(
        analysis_dir / "dose_table.csv",
        dose_table(read_csv(analysis_dir / "pair_catalog.csv")),
        ["family", "split", "kind", "n", "d1n", "d1b", "d2j", "d2c", "d2k", "d3", "dr"],
    )
    (analysis_dir / "mmp_summary.json").write_text(C.stable_json(mmp["summary"]) + "\n", encoding="utf-8")
    write_csv(analysis_dir / "natural_pairs_multi.csv", mmp["rows"], mmp["columns"])
    C.write_json(analysis_dir / "scaffold_strata.json", strat)
    cases = select_cases(pair_rows, data["pairs"], data["caliper"], mol_store)
    C.write_json(analysis_dir / "cases.json", cases)
    write_tables(analysis_dir, results, uncertainty, mechanism, mmp, strat, mol_store, reps_store)
    C.log("analysis done")
    return 0


# ---------------------------------------------------------------------------
# case selection (median + counterexample per non-empty family, Test)
# ---------------------------------------------------------------------------


def select_cases(pair_rows, catalogue, caliper, mol_store) -> dict[str, Any]:
    """One median and one counterexample molecule per non-empty family (Test main).

    Ranking quantity is the calibrated response ``J_C`` (G_FIXED) on the
    calibration queue; ``J_K`` is used only for families with no C candidate.
    """
    c_by_pair: dict[str, list[str]] = {}
    for r in caliper:
        c_by_pair.setdefault(str(r["pair_id"]), []).append(str(r["c_graph_id"]))
    pair_by_id = {str(p["pair_id"]): p for p in catalogue}
    cases: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for family in C.FAMILIES:
        rowskey = (POP_CAL, "test", family)
        used = "J_C"
        if rowskey not in mol_store:
            rowskey = (POP_MAIN, "test", family)
            used = "J_K"
        if rowskey not in mol_store:
            summary[family] = {"non_empty": False}
            continue
        mm = mol_store[rowskey]
        metric = "J_C_G_FIXED" if used == "J_C" else "J_K_G_FIXED"
        vals = {rid: d.get(metric, np.nan) for rid, d in mm.items()}
        vals = {rid: v for rid, v in vals.items() if np.isfinite(v)}
        if not vals:
            summary[family] = {"non_empty": False}
            continue
        med = float(np.median(list(vals.values())))
        med_row = sorted(vals, key=lambda r: (abs(vals[r] - med), r))[0]
        sign = 1.0 if med >= 0 else -1.0
        opp = sorted(vals, key=lambda r: (sign * vals[r] if np.isfinite(vals[r]) else np.inf, r))[0]
        summary[family] = {
            "non_empty": True,
            "ranking_metric": metric,
            "population": rowskey[0],
            "n_molecules": len(vals),
            "median_value": med,
            "median_row": int(med_row),
            "counterexample_row": int(opp),
            "counterexample_value": float(vals[opp]),
        }
        for tag, rid in (("median", med_row), ("counterexample", opp)):
            prs = [r for r in pair_rows if int(r["row_id"]) == int(rid) and str(r["family"]) == family
                   and str(r["split"]) == "test" and int(r["pop_main"]) == 1]
            if not prs:
                continue
            # prefer a pair that has C candidates so the case figure can show C
            cand = [r for r in prs if int(r["n_c_candidates"]) > 0] or prs
            pr = sorted(cand, key=lambda r: (abs(r.get(metric, 0.0) or 0.0), str(r["pair_id"])))[len(cand) // 2]
            pid = str(pr["pair_id"])
            cat = pair_by_id[pid]
            cases.append(
                {
                    "tag": f"{family}_{tag}",
                    "family": family,
                    "case_kind": tag,
                    "ranking_metric": metric,
                    "row_id": int(pr["row_id"]),
                    "split": "test",
                    "instance_id": str(pr["instance_id"]),
                    "center": int(pr["center"]),
                    "target": int(pr["target"]),
                    "control": int(pr["control"]),
                    "scaffold_key": str(pr["scaffold_key"]),
                    "scaffold_seen_in_train": int(pr["scaffold_seen_in_train"]),
                    "y": float(pr["y"]),
                    "preds": {
                        "O": float(pr["p0"]),
                        "T_G_FIXED": float(pr[f"pT_{G_FIXED}"]),
                        "K0_G_FIXED": float(pr[f"pK_{G_FIXED}"]),
                        "T_FULL": float(pr[f"pT_{FULL}"]),
                        "K0_FULL": float(pr[f"pK_{FULL}"]),
                        "C_mean_G_FIXED": (_f(pr.get("pC_mean_G_FIXED")) if c_by_pair.get(pid) else None),
                    },
                    "aT_G_FIXED": float(pr[f"aT_{G_FIXED}"]),
                    "aK_G_FIXED": float(pr[f"aK_{G_FIXED}"]),
                    "aC_G_FIXED": (_f(pr.get(f"aC_{G_FIXED}")) if c_by_pair.get(pid) else None),
                    "J_C_G_FIXED": (_f(pr.get(f"J_C_{G_FIXED}")) if c_by_pair.get(pid) else None),
                    "J_K_G_FIXED": _f(pr.get(f"J_K_{G_FIXED}")),
                    "D_K_G_FIXED": _f(pr.get(f"D_K_{G_FIXED}")),
                    "B_G_FIXED": _f(pr.get(f"B_{G_FIXED}")),
                    "t_remove": str(cat["t_remove"]),
                    "t_add": str(cat["t_add"]),
                    "k_remove": str(cat["k_remove"]),
                    "k_add": str(cat["k_add"]),
                    "n_c_candidates": len(c_by_pair.get(pid, [])),
                }
            )
    return {
        "main_population": "Test, main rewiring queue (RANGE_SUPPORTED)",
        "note": "ranking uses G_FIXED; J_C when a dose-calipered C exists, else J_K; "
                "counterexample = strongest opposite-direction molecule",
        "families": summary,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# influence
# ---------------------------------------------------------------------------


def influence_checks(pair_rows, mol_store, mult, reps_store) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in C.FAMILIES:
        for queue, metric in ((POP_CAL, "J_C_G_FIXED"), (POP_MAIN, "J_K_G_FIXED")):
            key = (queue, "test", family)
            if key not in mol_store:
                continue
            mm = mol_store[key]
            rows = np.asarray(sorted(mm), dtype=np.int64)
            ref = np.asarray([mm[int(r)].get(metric, np.nan) for r in rows], dtype=np.float64)
            finite = np.isfinite(ref)
            if not finite.any():
                continue
            order = sorted(
                range(len(rows)), key=lambda i: (-abs(float(ref[i])) if np.isfinite(ref[i]) else 0.0, int(rows[i]))
            )
            worst = order[0]
            k_block = max(1, int(math.ceil(C.INFLUENCE_FRACTION * len(rows))))
            dropped = sorted(int(rows[i]) for i in order[:k_block])
            variants = {
                "all": set(int(r) for r in rows),
                "drop_max": set(int(r) for r in rows) - {int(rows[worst])},
                "drop_block": set(int(r) for r in rows) - set(dropped),
            }
            block: dict[str, Any] = {
                "queue": queue,
                "ranking_metric": metric,
                "n_molecules": int(len(rows)),
                "n_dropped_block": int(k_block),
                "max_abs_row": int(rows[worst]),
                "max_abs_value": float(ref[worst]),
                "dropped_block_rows": dropped,
                "variants": {},
            }
            for name, keep in variants.items():
                vals = {
                    q: {rid: d.get(q, np.nan) for rid, d in mm.items() if rid in keep}
                    for q in ("rC", "rC_init", "J_C_G_FIXED", "J_K_G_FIXED", "J_C_FULL", "J_K_FULL")
                }
                entry = {q: point_mean(v) for q, v in vals.items()}
                entry["E"] = point_mean(
                    {rid: (vals["rC"][rid] - vals["rC_init"][rid]) for rid in vals["rC"]}
                )
                block["variants"][name] = entry
            out[f"{family}/{queue}"] = block
    return out


def denominator_guard(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often the rC / rK denominators are (near) zero, per state and family."""
    out: dict[str, Any] = {}
    for family in C.FAMILIES:
        rows = [r for r in pair_rows if str(r["family"]) == family and int(r["pop_main"]) == 1]
        if not rows:
            continue
        block = {"n_pairs": len(rows)}
        for name, (a, b) in (
            ("rC_trained", (f"aT_{G_FIXED}", f"aC_{G_FIXED}")),
            ("rC_init", ("aT_init", "aC_init")),
            ("rK_trained", (f"aT_{G_FIXED}", f"aK_{G_FIXED}")),
            ("rK_init", ("aT_init", "aK_init")),
        ):
            pairs_ab = [(_f(r.get(a)), _f(r.get(b))) for r in rows]
            pairs_ab = [(x, y) for x, y in pairs_ab if np.isfinite(x) and np.isfinite(y)]
            dens = np.asarray([x + y for x, y in pairs_ab], dtype=np.float64)
            if not dens.size:
                continue
            block[name] = {
                "n": int(dens.size),
                "frac_denominator_le_1e-12": float(np.mean(dens <= C.R_DENOM_EPS)),
                "median_denominator": float(np.median(dens)),
            }
        # is the INIT state constant on the real views?
        a_init = np.asarray([_f(r.get("aT_init")) for r in rows], dtype=np.float64)
        a_trained = np.asarray([_f(r.get(f"aT_{G_FIXED}")) for r in rows], dtype=np.float64)
        block["init_frac_aT_zero"] = float(np.mean(np.abs(a_init) <= C.R_DENOM_EPS))
        block["trained_frac_aT_zero"] = float(np.mean(np.abs(a_trained) <= C.R_DENOM_EPS))
        block["median_aT_trained"] = float(np.nanmedian(a_trained))
        block["median_aT_init"] = float(np.nanmedian(a_init))
        out[family] = block
    return out


# ---------------------------------------------------------------------------
# mechanism tables
# ---------------------------------------------------------------------------


def mobius_identity_checks(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The seven Mobius terms must sum back to the full-input difference."""
    out: dict[str, Any] = {
        "tolerance": C.MOBIUS_TOL,
        "note": "sum of the seven terms == full local-input change (H: |pT-p0|, L: loss change)",
    }
    checks: dict[str, float] = {}
    for name, total, compare in (
        ("H_T_vs_O", ["HT_" + t for t in TERMS], f"aT_{G_FIXED}"),
        ("H_K_vs_O", ["HK_" + t for t in TERMS], f"aK_{G_FIXED}"),
        ("L_T_vs_O", ["LT_" + t for t in TERMS], f"B_{G_FIXED}"),
        ("L_K_vs_O", ["LK_" + t for t in TERMS], "lossK_minus_loss0"),
        ("H_T_vs_K0", ["HJ_" + t for t in TERMS], f"J_K_{G_FIXED}"),
        ("L_T_vs_K0", ["DJ_" + t for t in TERMS], f"D_K_{G_FIXED}"),
        ("H_T_vs_C", ["HJ_C_" + t for t in TERMS], f"J_C_{G_FIXED}"),
        ("L_T_vs_C", ["DJ_C_" + t for t in TERMS], f"D_C_{G_FIXED}"),
    ):
        worst = 0.0
        n = 0
        for r in pair_rows:
            if not int(r.get("mobius_available", 0) or 0):
                continue
            vals = [_f(r.get(t)) for t in total]
            if compare == "lossK_minus_loss0":
                ref = _f(r.get(f"lossK_{G_FIXED}")) - _f(r.get("loss0"))
            else:
                ref = _f(r.get(compare))
            if not np.isfinite(ref) or not all(np.isfinite(v) for v in vals):
                continue
            worst = max(worst, abs(float(sum(vals)) - ref))
            n += 1
        checks[name] = {"max_abs_residual": worst, "n_pairs": n}
    out["checks"] = checks
    out["pass"] = all(c["max_abs_residual"] <= C.MOBIUS_TOL for c in checks.values())
    return out


def foreign_c_only_estimates(pair_rows, catalogue, caliper, mult) -> dict[str, Any]:
    """Descriptive robustness: the calibration queue restricted to focal pairs whose
    C set is not exactly their own K0 view (main analysis still uses every C)."""
    k_view = {str(p["pair_id"]): str(p["k_graph_id"]) for p in catalogue}
    own_k0: set[str] = set()
    with_any: set[str] = set()
    for r in caliper:
        pid = str(r["pair_id"])
        with_any.add(pid)
        if str(r["c_graph_id"]) == k_view.get(pid):
            own_k0.add(pid)
    only_own = own_k0  # focal pairs whose *only* C candidates are their own K0
    by_pair_c: dict[str, set[str]] = {}
    for r in caliper:
        by_pair_c.setdefault(str(r["pair_id"]), set()).add(str(r["c_graph_id"]))
    only_own = {pid for pid, cs in by_pair_c.items() if cs <= {k_view.get(pid)}}
    out: dict[str, Any] = {}
    for family in C.FAMILIES:
        for split in ("test",):
            rows = family_rows(pair_rows, family, split, POP_CAL)
            keep = [r for r in rows if str(r["pair_id"]) not in only_own]
            if not keep:
                out[f"{family}/{split}"] = {"supported": False, "reason": "all focal C sets are own K0"}
                continue
            mm = molecule_means(keep, ("rC", "rC_init", "J_C_G_FIXED"))
            rC = {rid: d.get("rC", np.nan) for rid, d in mm.items()}
            rC_init = {rid: d.get("rC_init", np.nan) for rid, d in mm.items()}
            reps = bootstrap_mean(mult[split], rC)
            reps_i = bootstrap_mean(mult[split], rC_init)
            out[f"{family}/{split}"] = {
                "supported": True,
                "n_molecules": len(mm),
                "n_pairs": len(keep),
                "rC_trained": point_mean(rC),
                "rC_ci95": pct_ci(reps),
                "E_point": point_mean({rid: rC[rid] - rC_init[rid] for rid in rC}),
                "E_ci95": pct_ci(reps - reps_i),
                "J_C": point_mean({rid: d.get("J_C_G_FIXED", np.nan) for rid, d in mm.items()}),
            }
    return out



def mechanism_tables(pair_rows, mol_store, mult) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    worst_sum = 0.0
    for split in ("valid", "test"):
        for population, cmp_H, cmp_L in (
            (POP_CAL, "HJ_C", "DJ_C"),
            (POP_MAIN, "HJ", "DJ"),
        ):
            for family in C.FAMILIES:
                sel = family_rows(pair_rows, family, split, population)
                if not sel:
                    continue
                keys = [f"{cmp_H}_{t}" for t in TERMS] + [f"{cmp_L}_{t}" for t in TERMS]
                if not any(k in sel[0] for k in keys):
                    continue
                mm = molecule_means(sel, keys)
                block: dict[str, Any] = {}
                for space, prefix in (("H", cmp_H), ("L", cmp_L)):
                    terms_point: dict[str, float] = {}
                    total_point = 0.0
                    terms_ci: dict[str, list[float]] = {}
                    for term in TERMS:
                        values = {rid: d.get(f"{prefix}_{term}", np.nan) for rid, d in mm.items()}
                        p = point_mean(values)
                        reps = bootstrap_mean(mult[split], values)
                        terms_point[term] = p
                        terms_ci[term] = pct_ci(reps)
                        total_point += p if np.isfinite(p) else 0.0
                        rows.append(
                            {
                                "split": split,
                                "population": population,
                                "family": family,
                                "space": space,
                                "comparison": "T_minus_C" if population == POP_CAL else "T_minus_K0",
                                "term": term,
                                "point": p,
                                "ci95_low": terms_ci[term][0],
                                "ci95_high": terms_ci[term][1],
                                "n_molecules": len(mm),
                            }
                        )
                    block[space] = {"terms": terms_point, "ci95": terms_ci, "sum_of_terms": total_point}
                # conditional S2 recovery (computational, not a retraining claim)
                rec = {}
                for name in ("H_s2_recovery_123_vs_13_T", "H_s2_recovery_123_vs_23_T"):
                    values = {rid: d.get(name, np.nan) for rid, d in mm.items()}
                    rec[name] = {"point": point_mean(values), "ci95": pct_ci(bootstrap_mean(mult[split], values))}
                block["s2_conditional_recovery"] = rec
                ref_key = "J_C_G_FIXED" if population == POP_CAL else "J_K_G_FIXED"
                ref = point_mean({rid: d.get(ref_key, np.nan) for rid, d in mm.items()})
                worst_sum = max(worst_sum, abs(block["H"]["sum_of_terms"] - ref))
                summary[f"{split}/{population}/{family}"] = block
    return {
        "rows": rows,
        "summary": summary,
        "sum_check": {"max_abs_sum_of_H_terms_minus_reference": float(worst_sum), "tolerance": C.MOBIUS_TOL,
                      "pass": bool(worst_sum <= C.MOBIUS_TOL)},
    }


# ---------------------------------------------------------------------------
# dose table
# ---------------------------------------------------------------------------


def dose_table(catalogue: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Median (and mean) structural dose per family / split / kind (structure only)."""
    out: list[dict[str, Any]] = []
    keys = ("d1n", "d1b", "d2j", "d2c", "d2k", "d3", "dr")
    for family in C.FAMILIES:
        for split in ("valid", "test"):
            for kind, prefix in (("T", "dose_"), ("K0", "doseK_")):
                rows = [
                    r for r in catalogue
                    if str(r["family"]) == family
                    and str(r["split"]) == split
                    and int(float(r["pop_main"] or 0)) == 1
                ]
                if not rows:
                    continue
                entry: dict[str, Any] = {"family": family, "split": split, "kind": kind, "n": len(rows)}
                for k in keys:
                    arr = np.asarray([_f(r.get(prefix + k)) for r in rows], dtype=np.float64)
                    arr = arr[np.isfinite(arr)]
                    entry[k] = float(np.median(arr)) if arr.size else None
                entry["mean"] = {k: float(np.mean([_f(r.get(prefix + k)) for r in rows])) for k in keys}
                out.append(entry)
    return out


# ---------------------------------------------------------------------------
# independent caliper re-verification (rebuilds every T / C graph from scratch)
# ---------------------------------------------------------------------------


def _parse_view_key(key: str) -> tuple[int, str, str, list[tuple[int, int]], list[tuple[int, int]]]:
    rid, kind, family, rem, add = str(key).split(":")
    def bonds(code: str) -> list[tuple[int, int]]:
        code = code[1:] if code.startswith(("R", "A")) else code
        if not code:
            return []
        return [tuple(int(x) for x in chunk.split("-")) for chunk in code.split(".")]
    return int(rid), str(kind), str(family), bonds(rem), bonds(add)


def verify_calipers(
    analysis_dir: Path,
    caliper: Sequence[Mapping[str, str]],
    pairs: Sequence[Mapping[str, str]],
    src: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every T / C dose from rebuilt graphs and re-test the caliper.

    Reads only frozen catalogue rows plus the frozen raw rows and the frozen
    Train-only scalar normalisation; never touches a prediction or a label.
    """
    import feats as F
    import infer as INF
    import doses as D
    import rewire as R

    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    ck = C.load_checkpoint()
    norm = C.frozen_norm(ck)
    graph_rows = read_csv(analysis_dir / "graph_manifest.csv")
    pair_by_id = {str(p["pair_id"]): p for p in pairs}
    extra: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in caliper:
        gid = str(r["c_graph_id"])
        if gid in seen:
            continue
        seen.add(gid)
        rid, kind, family, rem, add = _parse_view_key(str(r["c_view_keys"]))
        extra.append(
            {
                "graph_id": gid,
                "row_id": str(rid),
                "split": str(r["split"]),
                "kind": kind,
                "families": family,
                "remove": ".".join(f"{a}-{b}" for a, b in rem),
                "add": ".".join(f"{a}-{b}" for a, b in add),
                "scopes": "G_FIXED",
            }
        )
    failures: list[dict[str, Any]] = []
    out: dict[str, Any] = {"pass": True, "failures": failures, "per_split": {}}
    for split in ("valid", "test"):
        rows = [r for r in graph_rows if str(r["split"]) == split] + [r for r in extra if str(r["split"]) == split]
        spec = C.DD.frozen_spec()
        entries, kept = INF.build_split_cache(analysis_dir, rows, split, raw_rows, spec)
        cache = F.build_cache(entries, spec, split=f"caliper_verify_{split}")
        idx_by = {str(k): i for i, k in enumerate(cache["view_ids"])}
        tables = {gid: D.molecule_tables(cache, i) for gid, i in idx_by.items()}
        n_checked = 0
        max_abs_diff = 0.0
        n_d1n_bad = n_d1b_bad = n_d2k_bad = n_ratio_bad = 0
        n_bond_delta_bad = 0
        for r in caliper:
            if str(r["split"]) != split:
                continue
            pr = pair_by_id.get(str(r["pair_id"]))
            if pr is None:
                failures.append({"reason": "pair_missing", "pair_id": r["pair_id"]})
                continue
            t_id, c_id = str(pr["t_graph_id"]), str(r["c_graph_id"])
            if t_id not in tables or c_id not in tables:
                failures.append({"reason": "graph_missing", "t": t_id, "c": c_id})
                continue
            n_checked += 1
            # bond-count delta of each view vs the original molecule
            o_id = str(pr["o_graph_id"])
            for gid in (t_id, c_id):
                if abs(len(tables[gid]["nbr_rows"]) - len(tables[o_id]["nbr_rows"])) != 0:
                    n_bond_delta_bad += 1
            dt = D.doses(tables[o_id], tables[t_id], scalar_mean=norm["scalar_mean"], scalar_scale=norm["scalar_scale"])
            dc = D.doses(tables[o_id], tables[c_id], scalar_mean=norm["scalar_mean"], scalar_scale=norm["scalar_scale"])
            for k in ("d1n", "d1b", "d2j", "d2c", "d2k", "d3", "dr"):
                max_abs_diff = max(max_abs_diff, abs(float(dc[k]) - _f(r.get(k), float("nan"))))
            if int(dc["d1n"]) != int(dt["d1n"]):
                n_d1n_bad += 1
            if int(dc["d1b"]) != int(dt["d1b"]):
                n_d1b_bad += 1
            if int(dc["d2k"]) != int(dt["d2k"]):
                n_d2k_bad += 1
            ok, _detail = D.caliper_ok(dt, dc, C.C_CALIPER_RATIO, C.C_EPS)
            if not ok:
                n_ratio_bad += 1
                failures.append({"reason": "caliper_recheck_failed", "pair_id": r["pair_id"], "c": c_id})
        blk = {
            "n_candidates_checked": int(n_checked),
            "n_d1n_unequal": int(n_d1n_bad),
            "n_d1b_unequal": int(n_d1b_bad),
            "n_d2k_unequal": int(n_d2k_bad),
            "n_ratio_recheck_failed": int(n_ratio_bad),
            "n_bond_count_delta_bad": int(n_bond_delta_bad),
            "max_abs_diff_vs_catalogue": float(max_abs_diff),
        }
        blk["pass"] = (
            n_d1n_bad == 0 and n_d1b_bad == 0 and n_d2k_bad == 0 and n_ratio_bad == 0
            and n_bond_delta_bad == 0 and max_abs_diff <= 1e-6
        )
        out["per_split"][split] = blk
        out["pass"] = bool(out["pass"] and blk["pass"])
    return out


# ---------------------------------------------------------------------------
# caliper composition
# ---------------------------------------------------------------------------


def caliper_composition(caliper: Sequence[Mapping[str, str]], pairs: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    pair_family = {str(p["pair_id"]): str(p["family"]) for p in pairs}
    pair_kview = {str(p["pair_id"]): str(p["k_graph_id"]) for p in pairs}
    pair_tview = {str(p["pair_id"]): str(p["t_graph_id"]) for p in pairs}
    out: dict[str, Any] = {}
    for family in C.FAMILIES:
        rows = [r for r in caliper if str(r["family"]) == family]
        if not rows:
            out[family] = {"n_candidates": 0}
            continue
        kinds: dict[str, int] = {}
        c_families: dict[str, int] = {}
        own_k0 = 0
        focal = {str(r["pair_id"]) for r in rows}
        for r in rows:
            kinds[str(r["c_kind"])] = kinds.get(str(r["c_kind"]), 0) + 1
            for f in str(r["c_family"]).split(","):
                c_families[f] = c_families.get(f, 0) + 1
            if str(r["c_graph_id"]) == pair_kview.get(str(r["pair_id"])):
                own_k0 += 1
        n_per_focal = [len({str(r["c_graph_id"]) for r in rows if str(r["pair_id"]) == pid}) for pid in sorted(focal)]
        out[family] = {
            "n_focal_pairs_with_c": len(focal),
            "n_candidates": len(rows),
            "c_kinds": kinds,
            "c_families": c_families,
            "c_is_own_k0": own_k0,
            "candidates_per_focal": {
                "min": int(min(n_per_focal)) if n_per_focal else 0,
                "median": float(np.median(n_per_focal)) if n_per_focal else None,
                "max": int(max(n_per_focal)) if n_per_focal else 0,
            },
            "c_is_own_T": sum(1 for r in rows if str(r["c_graph_id"]) == pair_tview.get(str(r["pair_id"]))),
            "n_focal_own_k0_only": len(
                {
                    str(r["pair_id"]) for r in rows
                    if str(r["pair_id"]) not in _focal_with_non_own_k0(rows, pair_kview)
                }
            ),
            "n_focal_with_foreign_c": len(_focal_with_non_own_k0(rows, pair_kview)),
        }
    # coverage: share of the family population that has any C
    for family in C.FAMILIES:
        for split in ("valid", "test"):
            tot = [p for p in pairs if str(p["family"]) == family and str(p["split"]) == split and int(p["pop_main"]) == 1]
            cov = [p for p in tot if int(p["n_c_candidates"] or 0) > 0]
            if not tot:
                continue
            out[family][f"coverage/{split}"] = {
                "pairs": len(tot),
                "pairs_with_c": len(cov),
                "fraction": float(len(cov) / len(tot)),
                "molecules": len({p["row_id"] for p in tot}),
                "molecules_with_c": len({p["row_id"] for p in cov}),
            }
    return out


# ---------------------------------------------------------------------------
# scaffold strata
# ---------------------------------------------------------------------------


def _focal_with_non_own_k0(rows: Sequence[Mapping[str, str]], pair_kview: Mapping[str, str]) -> set[str]:
    """Focal pairs that have at least one C candidate that is not their own K0 view."""
    return {
        str(r["pair_id"]) for r in rows if str(r["c_graph_id"]) != pair_kview.get(str(r["pair_id"]))
    }


def scaffold_strata(pair_rows, mult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in C.FAMILIES:
        for split in ("valid", "test"):
            rows = [r for r in pair_rows if str(r["family"]) == family and str(r["split"]) == split and int(r["pop_main"]) == 1]
            if not rows:
                continue
            mm = molecule_means(rows, ("J_K_G_FIXED", "J_C_G_FIXED", "rC", "rC_init"))
            seen = {int(r["row_id"]): int(r["scaffold_seen_in_train"]) for r in rows}
            block = {}
            for label, flag in (("unseen_in_train", 0), ("seen_in_train", 1)):
                keep = {rid: d for rid, d in mm.items() if seen.get(rid, 0) == flag}
                if not keep:
                    block[label] = {"n_molecules": 0}
                    continue
                block[label] = {
                    "n_molecules": len(keep),
                    "J_K": point_mean({k: v.get("J_K_G_FIXED", np.nan) for k, v in keep.items()}),
                    "J_C": point_mean({k: v.get("J_C_G_FIXED", np.nan) for k, v in keep.items()}),
                    "rC": point_mean({k: v.get("rC", np.nan) for k, v in keep.items()}),
                    "rC_init": point_mean({k: v.get("rC_init", np.nan) for k, v in keep.items()}),
                }
            out[f"{family}/{split}"] = block
    return out


# ---------------------------------------------------------------------------
# natural pairs (reused 823 MMP pairs, no new mining)
# ---------------------------------------------------------------------------


def natural_pairs(analysis_dir: Path, scaffold, previous, reader) -> dict[str, Any]:
    import graphlib as G
    import families as FAM

    src_csv = C.PAIRS_SOURCE
    if not src_csv.exists():
        return {"summary": {"error": "MISSING_BASELINE: pairs.csv not found"}, "rows": [], "columns": []}
    pairs = reader(src_csv)
    raw = C.load_raw_rows()["rows"]["Lipophilicity"]
    counts: dict[int, dict[str, int]] = {}
    for r in pairs:
        for key in ("row_a", "row_b"):
            rid = int(r[key])
            if rid in counts:
                continue
            g = G.summarise_raw(raw[rid], rid)
            counts[rid] = {fam: len(FAM.family_keys(g, fam)) for fam in C.FAMILIES}
    y = previous["y"]
    split = previous["split"]
    ref_pred_path = C.PREV_REFERENCE_PREDICTIONS
    ref_rows = reader(ref_pred_path)
    fpred = {int(r["row_id"]): float(r["pred_raw"]) for r in ref_rows}
    # class_id baseline from the frozen edit-supervision pair table (Train-Train only)
    class_of = {str(r["pair_id"]): str(r.get("class_id", "")) for r in pairs}
    dy_by_class: dict[str, list[float]] = {}
    train_pairs = [
        r for r in pairs if str(r.get("split_combo", "")).lower() in ("train-train",)
    ]
    for r in train_pairs:
        cid = class_of.get(str(r["pair_id"]), "")
        dy_by_class.setdefault(cid, []).append(float(y[int(r["row_a"])]) - float(y[int(r["row_b"])]))

    rows: list[dict[str, Any]] = []
    for family in C.FAMILIES:
        n_family = C.FAMILIES.index(family)
        for r in pairs:
            ra, rb = int(r["row_b"]), int(r["row_a"])  # arbitrary; direction set below
            ca, cb = counts.get(int(r["row_a"])), counts.get(int(r["row_b"]))
            if ca is None or cb is None:
                continue
            na, nb = ca[family], cb[family]
            if abs(na - nb) != 1:
                continue
            hi_row, lo_row = (int(r["row_a"]), int(r["row_b"])) if na > nb else (int(r["row_b"]), int(r["row_a"]))
            others_unchanged = all(
                ca[f] == cb[f] for f in C.FAMILIES if f != family
            )
            dy = float(y[hi_row]) - float(y[lo_row])
            df = float(fpred[hi_row]) - float(fpred[lo_row])
            sa, sb = split[hi_row], split[lo_row]
            if sa == "train" and sb == "train":
                group = "P_train"
            elif "test" in (sa, sb) and "train" in (sa, sb):
                group = "E_T_train_endpoint"
            elif "test" in (sa, sb) and "train" not in (sa, sb):
                group = "E_HH_nonTest_train_pair"
            elif sa == "valid" and sb == "valid":
                group = "V_valid"
            else:
                group = "other"
            cid = class_of.get(str(r["pair_id"]), "")
            base = float(np.mean(dy_by_class.get(cid, [0.0]))) if cid in dy_by_class else 0.0
            rows.append(
                {
                    "family": family,
                    "pair_id": str(r["pair_id"]),
                    "row_more": hi_row,
                    "row_less": lo_row,
                    "split_more": sa,
                    "split_less": sb,
                    "split_combo": f"{sa}-{sb}",
                    "group": group,
                    "clean_other_families_unchanged": int(others_unchanged),
                    "n_target_more": na,
                    "n_target_less": nb,
                    "dy": dy,
                    "df": df,
                    "abs_residual": abs(df - dy),
                    "abs_dy": abs(dy),
                    "train_class_id": cid,
                    "train_class_mean_dy": base if cid in dy_by_class else 0.0,
                    "train_class_available": int(cid in dy_by_class),
                    "scaffold_more": scaffold.get(hi_row, ""),
                    "scaffold_less": scaffold.get(lo_row, ""),
                    "eval_component": int(r.get("eval_component", 0) or 0),
                }
            )
    summary: dict[str, Any] = {
        "source": C.track_rel(src_csv),
        "source_sha256": C.sha256_file(src_csv),
        "n_source_pairs": len(pairs),
        "selection": "|Delta family-count| == 1 on the FULL family count; direction = more minus fewer; "
        "clean subset = the other five family counts both unchanged",
        "groups": {},
    }
    for family in C.FAMILIES:
        for clean in (0, 1):
            for group in ("P_train", "E_T_train_endpoint", "E_HH_nonTest_train_pair", "V_valid", "other"):
                sel = [x for x in rows if x["family"] == family and x["group"] == group
                       and x["clean_other_families_unchanged"] == clean]
                if not sel:
                    continue
                comps: dict[int, list[float]] = {}
                comps_base: dict[int, list[float]] = {}
                comps_class: dict[int, list[float]] = {}
                for x in sel:
                    comps.setdefault(int(x["eval_component"]), []).append(float(x["abs_residual"]))
                    comps_base.setdefault(int(x["eval_component"]), []).append(float(x["abs_dy"]))
                    if int(x["train_class_available"]):
                        comps_class.setdefault(int(x["eval_component"]), []).append(
                            abs(float(x["train_class_mean_dy"]) - float(x["dy"]))
                        )
                key = f"{family}/{group}/{'clean' if clean else 'all'}"
                summary["groups"][key] = {
                    "n_pairs": len(sel),
                    "n_components": len(comps),
                    "pair_equal_mae": float(np.mean([abs(x["df"] - x["dy"]) for x in sel])),
                    "component_equal_mae": float(np.mean([np.mean(v) for v in comps.values()])),
                    "zero_change_component_mae": float(np.mean([np.mean(v) for v in comps_base.values()])),
                    "train_class_component_mae": (
                        float(np.mean([np.mean(v) for v in comps_class.values()])) if comps_class else None
                    ),
                    "n_components_with_train_class": len(comps_class),
                    "train_class_coverage": (
                        float(len(comps_class) / len(comps)) if comps else None
                    ),
                    "n_pairs_df_closer_than_zero": int(sum(1 for x in sel if abs(x["df"] - x["dy"]) < abs(x["dy"]))),
                    "sign_agreement": float(
                        np.mean([np.sign(x["df"]) == np.sign(x["dy"]) for x in sel])
                    ),
                    "support": "LOW_SUPPORT" if len(comps) < 10 else "SUPPORTED",
                }
    columns = list(rows[0].keys()) if rows else []
    return {"summary": summary, "rows": rows, "columns": columns}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def molecule_rows(mol_store) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (population, split, family), mm in mol_store.items():
        for rid, d in sorted(mm.items()):
            out.append({"population": population, "split": split, "family": family, "row_id": rid, **d})
    return out


def _fmt(x: Any, nd: int = 4) -> str:
    v = _f(x)
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_tables(analysis_dir, results, uncertainty, mechanism, mmp, strat, mol_store, reps_store) -> None:
    # ---- applicability x effect matrix ---------------------------------
    lines = [
        "# 六格适用范围与效应矩阵",
        "",
        "主统计口径 = G_FIXED（local 输入来自新图、global42 固定为 O），校准队列 = 存在剂量 caliper 通过的 C 的 T。",
        "CI = scaffold cluster bootstrap 95%（每 split 一套共享 multiplicity，5000 次，seed 20261114）。",
        "`rC` = (aT-aC)/(aT+aC)，`E` = rC_TRAINED - rC_INIT。Bonferroni 同时区间见 `uncertainty.json`。",
        "",
        "| family | split | 人群 | n_mol | n_pair | rC [95% CI] | rC_INIT | E [95% CI] | J_C | J_K | D_C | D_K | B |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family in C.FAMILIES:
        for split in ("valid", "test"):
            for population in (POP_CAL, POP_MAIN):
                key = (population, split, family)
                if key not in mol_store:
                    lines.append(f"| {family} | {split} | {population} | 0 | 0 | — | — | — | — | — | — | — | — |")
                    continue
                mm = mol_store[key]
                support = uncertainty["family_support"][family][f"{population}/{split}"]

                def ci(name):
                    reps = reps_store.get((population, split, family, name))
                    if reps is None:
                        return "n/a"
                    lo, hi = pct_ci(reps)
                    return f"[{_fmt(lo)}, {_fmt(hi)}]"

                rC = point_mean({rid: d.get("rC", np.nan) for rid, d in mm.items()})
                rC_init = point_mean({rid: d.get("rC_init", np.nan) for rid, d in mm.items()})
                E = point_mean({rid: d.get("rC", np.nan) - d.get("rC_init", np.nan) for rid, d in mm.items()})
                reps_rC = reps_store.get((population, split, family, "rC"))
                reps_init = reps_store.get((population, split, family, "rC_init"))
                if reps_rC is not None and reps_init is not None:
                    e_lo, e_hi = pct_ci(reps_rC - reps_init)
                    E_ci = f"[{_fmt(e_lo)}, {_fmt(e_hi)}]"
                else:
                    E_ci = "n/a"
                jc = point_mean({rid: d.get("J_C_G_FIXED", np.nan) for rid, d in mm.items()})
                jk = point_mean({rid: d.get("J_K_G_FIXED", np.nan) for rid, d in mm.items()})
                dc = point_mean({rid: d.get("D_C_G_FIXED", np.nan) for rid, d in mm.items()})
                dk = point_mean({rid: d.get("D_K_G_FIXED", np.nan) for rid, d in mm.items()})
                bb = point_mean({rid: d.get("B_G_FIXED", np.nan) for rid, d in mm.items()})
                lines.append(
                    f"| {family} | {split} | {population} | {support['molecules']} | {support['pairs']} | "
                    f"{_fmt(rC)} {ci('rC')} | {_fmt(rC_init)} | {_fmt(E)} {E_ci} | "
                    f"{_fmt(jc)} | {_fmt(jk)} | {_fmt(dc)} | {_fmt(dk)} | {_fmt(bb)} |"
                )
    (analysis_dir / "effect_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- mechanism matrix ----------------------------------------------
    lines = [
        "# 机制矩阵（G_FIXED, 三因素 Mobius 七项）",
        "",
        "项 = S1(G1 一阶聚合) / S2(G2 路径聚合) / scalar(G3) 各自单独项与二阶、三阶交互项。",
        "对比 `T_minus_K0`（全队列）与 `T_minus_C`（校准队列）分别给出 H（响应）与 L（参照偏差）。",
        "七项之和恒等于全输入变化（端点恒等式 ≤1e-5）。",
        "",
        "| split | 人群 | family | space | S1 | S2 | scalar | S1xS2 | S1xS | S2xS | S1xS2xS | 合计 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, block in sorted(mechanism["summary"].items()):
        split, population, family = key.split("/")
        for space in ("H", "L"):
            terms = block[space]["terms"]
            lines.append(
                f"| {split} | {population} | {family} | {space} | "
                + " | ".join(_fmt(terms[t]) for t in TERMS)
                + f" | {_fmt(block[space]['sum_of_terms'])} |"
            )
    (analysis_dir / "mechanism_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- primary answers ------------------------------------------------
    lines = ["# 主问题回答（Test，校准队列，G_FIXED）", ""]
    for family, item in uncertainty["primary_test"]["items"].items():
        if not item.get("supported"):
            lines.append(f"- **{family}**: 无 C 校准样本 → 该竞争解释未排除（{item.get('reason', '')}）")
            continue
        lines.append(
            f"- **{family}** (n_mol={item['n_molecules']}, n_pair={item['n_pairs']}): "
            f"rC={_fmt(item['rC_trained']['point'])} {item['rC_trained']['ci95']}, "
            f"E={_fmt(item['E']['point'])} {item['E']['ci95']}, "
            f"rC>0:{item['rC_trained_positive']}, E>0:{item['E_positive']}, "
            f"Bonferroni rC {item['rC_trained']['bonferroni']}, E {item['E']['bonferroni']}"
        )
    (analysis_dir / "primary_answers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- natural pairs table -------------------------------------------
    lines = [
        "# 真实分子对（沿用 823 MMP 对，不新挖掘）",
        "",
        "`Δf` = 冻结模型对**原分子**的预测差；`Δy` = 实验差；`|Δf-Δy|` 与零变化基线 `|Δy|` 按 eval_component 聚合。",
        "",
        "| family | group | subset | pairs | components | MAE | zero-change MAE | Train-class MAE (coverage) | support |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, v in sorted(mmp["summary"].get("groups", {}).items()):
        family, group, subset = key.split("/")
        lines.append(
            f"| {family} | {group} | {subset} | {v['n_pairs']} | {v['n_components']} | "
            f"{_fmt(v['component_equal_mae'])} | {_fmt(v['zero_change_component_mae'])} | "
            f"{_fmt(v['train_class_component_mae'])} ({_fmt(v['train_class_coverage'], 2)}) | {v['support']} |"
        )
    (analysis_dir / "mmp_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- scaffold strata ------------------------------------------------
    lines = [
        "# 原 scaffold 是否在 Train 出现（仅结构定义分层）",
        "",
        "| family | split | stratum | n_mol | J_K | J_C | rC | rC_INIT |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, block in sorted(strat.items()):
        family, split = key.split("/")
        for label in ("seen_in_train", "unseen_in_train"):
            v = block.get(label, {})
            lines.append(
                f"| {family} | {split} | {label} | {v.get('n_molecules', 0)} | {_fmt(v.get('J_K'))} | "
                f"{_fmt(v.get('J_C'))} | {_fmt(v.get('rC'))} | {_fmt(v.get('rC_init'))} |"
            )
    (analysis_dir / "scaffold_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
