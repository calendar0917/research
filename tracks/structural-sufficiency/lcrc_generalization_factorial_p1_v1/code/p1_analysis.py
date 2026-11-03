"""Metric, interaction, tail-risk and subgroup analysis for LCRC-P1.

Every function takes already-aligned :class:`p1_sources.Unit` objects and a
per-molecule feature frame from :func:`p1_sources.molecule_frame`.  Nothing here
reads a Test label for selection: metrics are descriptive, computed after every
checkpoint was frozen by Validation RMSE.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from p1_common import (
    ARMS,
    ARM_SHORT,
    BOOTSTRAP_SEED,
    N_BOOT,
    N_QUANTILE_BINS,
    Q2_ARMS,
    SEEDS,
    SPLIT_ORDER,
)
import p1_stats as S
from p1_sources import FrozenSplit, Unit
from p1_stats import ClusterIndex

PARTS = ("valid", "test")
PRIMARY_PART = "test"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def errors_for(unit: Unit, part: str) -> tuple[np.ndarray, np.ndarray]:
    return unit.ae(part), unit.se(part)


def sub_index(index: ClusterIndex, mask: Sequence[bool]) -> ClusterIndex:
    """Cluster index restricted to a subgroup (clusters re-derived on the subset)."""
    mask = np.asarray(mask, dtype=bool)
    names = index.cluster_names[index.ids[mask]]
    return ClusterIndex.from_values([str(x) for x in names])


def cluster_index_for(split: FrozenSplit, part: str) -> ClusterIndex:
    scaffolds = [split.molecules[int(i)].scaffold for i in split.rows[part]]
    return ClusterIndex.from_values(scaffolds)


def combo_base_statistics(
    positions: np.ndarray,
    ae_base: np.ndarray,
    se_base: np.ndarray,
    ae_combo: np.ndarray,
    se_combo: np.ndarray,
) -> dict[str, float]:
    b_ae = ae_base[positions]
    b_se = se_base[positions]
    c_ae = ae_combo[positions]
    c_se = se_combo[positions]
    return {
        "delta_rmse": S.rmse_from_se(c_se) - S.rmse_from_se(b_se),
        "delta_mae": S.mae_from_ae(c_ae) - S.mae_from_ae(b_ae),
        "delta_mse": float(np.mean(c_se) - np.mean(b_se)),
        "delta_p90_ae": S.p90_abs(c_ae) - S.p90_abs(b_ae),
        "delta_cvar90_ae": S.cvar90_ae(c_ae) - S.cvar90_ae(b_ae),
        "frac_improved": float(np.mean(c_se < b_se)),
    }


def combo_base_point(
    ae_base: np.ndarray, se_base: np.ndarray, ae_combo: np.ndarray, se_combo: np.ndarray
) -> dict[str, float]:
    return {
        "rmse_base": S.rmse_from_se(se_base),
        "rmse_combo": S.rmse_from_se(se_combo),
        "mae_base": S.mae_from_ae(ae_base),
        "mae_combo": S.mae_from_ae(ae_combo),
        "p90_base": S.p90_abs(ae_base),
        "p90_combo": S.p90_abs(ae_combo),
        "cvar90_base": S.cvar90_ae(ae_base),
        "cvar90_combo": S.cvar90_ae(ae_combo),
        "sse_base": S.sse_from_se(se_base),
        "sse_combo": S.sse_from_se(se_combo),
        "delta_rmse": S.rmse_from_se(se_combo) - S.rmse_from_se(se_base),
        "delta_mae": S.mae_from_ae(ae_combo) - S.mae_from_ae(ae_base),
        "delta_mse": float(np.mean(se_combo) - np.mean(se_base)),
        "delta_p90_ae": S.p90_abs(ae_combo) - S.p90_abs(ae_base),
        "delta_cvar90_ae": S.cvar90_ae(ae_combo) - S.cvar90_ae(ae_base),
        "frac_improved": float(np.mean(se_combo < se_base)),
    }


# ---------------------------------------------------------------------------
# 1. metrics table
# ---------------------------------------------------------------------------


def metrics_table(units: Mapping[tuple[str, str, int], Unit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, arm, seed), unit in sorted(units.items(), key=lambda kv: (SPLIT_ORDER.index(kv[0][0]), kv[0][2], ARMS.index(kv[0][1]))):
        for part in PARTS:
            entry = {
                "split": split,
                "seed": int(seed),
                "arm": arm,
                "arm_label": ARM_SHORT[arm],
                "unit_key": unit.key,
                "origin": unit.origin,
                "part": part,
                "n": int(len(unit.y[part])),
            }
            entry.update({k: float(v) for k, v in unit.metrics[part].items()})
            rows.append(entry)
    frame = pd.DataFrame(rows)
    order = ["split", "seed", "arm", "arm_label", "unit_key", "origin", "part", "n", "rmse", "mae", "p90_ae", "cvar90_ae", "sse", "bias"]
    return frame[order]


# ---------------------------------------------------------------------------
# 2. per-molecule table
# ---------------------------------------------------------------------------


def per_molecule_table(
    units: Mapping[tuple[str, str, int], Unit],
    frames: Mapping[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for split in SPLIT_ORDER:
        feature_frame, _ = frames[split]
        for part in PARTS:
            feats = feature_frame[feature_frame["part"] == part].reset_index(drop=True)
            for seed in SEEDS[split]:
                base = units.get((split, "base", seed))
                if base is None:
                    continue
                base_ae = base.ae(part)
                base_se = base.se(part)
                k_tail = S.tail_count(len(base_ae))
                tail_idx = set(S.top_abs_indices(base_ae, k_tail).tolist())
                for arm in ARMS if split == "random122" else Q2_ARMS:
                    unit = units.get((split, arm, seed))
                    if unit is None:
                        continue
                    ae = unit.ae(part)
                    se = unit.se(part)
                    frame = pd.DataFrame(
                        {
                            "split": split,
                            "part": part,
                            "seed": int(seed),
                            "arm": arm,
                            "arm_label": ARM_SHORT[arm],
                            "unit_key": unit.key,
                            "origin": unit.origin,
                            "row_id": feats["row_id"].to_numpy(dtype=np.int64),
                            "y_true": unit.y[part],
                            "pred": unit.pred[part],
                            "abs_err": ae,
                            "sq_err": se,
                            "base_abs_err": base_ae,
                            "base_sq_err": base_se,
                            "sq_err_diff_vs_base": se - base_se,
                            "improved_vs_base": se < base_se,
                            "base_tail_posthoc": [i in tail_idx for i in range(len(ae))],
                            "base_tail_label": "LABEL_SELECTED_POSTHOC_TAIL",
                        }
                    )
                    for column in (
                        "smiles",
                        "scaffold",
                        "scaffold_seen",
                        "heavy_atoms",
                        "tpsa",
                        "max_train_tanimoto",
                        "unseen_atom_frac",
                        "rare_atom_frac",
                        "median_log1p_df",
                        "p10_log1p_df",
                        "heavy_atoms_bin",
                        "tpsa_bin",
                        "max_train_tanimoto_bin",
                    ):
                        frame[column] = feats[column].to_numpy()
                    out.append(frame)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. paired uncertainty (COMBO - BASE) with scaffold-cluster bootstrap
# ---------------------------------------------------------------------------


def paired_uncertainty(
    units: Mapping[tuple[str, str, int], Unit],
    splits: Mapping[str, FrozenSplit],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "definition": (
            "COMBO - BASE on the same frozen rows, same seed. Negative favours COMBO. "
            "Percentile intervals from a Bemis-Murcko scaffold-cluster bootstrap."
        ),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "unit": "Bemis-Murcko scaffold cluster"},
        "splits": {},
    }
    for split in SPLIT_ORDER:
        payload["splits"][split] = {"parts": {}, "seed_mean": {}}
        per_seed_draws: dict[str, dict[int, dict[str, np.ndarray]]] = {p: {} for p in PARTS}
        for part in PARTS:
            index = cluster_index_for(splits[split], part)
            payload["splits"][split]["parts"].setdefault(part, {})["cluster_profile"] = index.profile()
            payload["splits"][split]["parts"][part]["per_seed"] = {}
            for s in SEEDS[split]:
                base = units.get((split, "base", s))
                combo = units.get((split, "combo_chem_contrib", s))
                if base is None or combo is None:
                    payload["splits"][split]["parts"][part]["per_seed"][str(s)] = {"available": False}
                    continue
                ae_b, se_b = errors_for(base, part)
                ae_c, se_c = errors_for(combo, part)
                point = combo_base_point(ae_b, se_b, ae_c, se_c)
                draws = S.cluster_bootstrap(
                    index,
                    lambda pos, a=ae_b, b=se_b, c=ae_c, d=se_c: combo_base_statistics(pos, a, b, c, d),
                    n_boot=n_boot,
                    seed=seed,
                )
                per_seed_draws[part][s] = draws
                payload["splits"][split]["parts"][part]["per_seed"][str(s)] = {
                    "available": True,
                    "unit_keys": {"base": base.key, "combo": combo.key},
                    "origins": {"base": base.origin, "combo": combo.origin},
                    "n": int(len(ae_b)),
                    "point": point,
                    "bootstrap": S.interval_summary(point, draws),
                }
            # seed-mean: the same resample is used for every seed, so the seed
            # spread is preserved and seeds are never treated as molecule samples
            available = sorted(per_seed_draws[part])
            if available:
                point = {k: float(np.mean([payload["splits"][split]["parts"][part]["per_seed"][str(s)]["point"][k] for s in available])) for k in (
                    "rmse_base", "rmse_combo", "mae_base", "mae_combo", "p90_base", "p90_combo",
                    "cvar90_base", "cvar90_combo", "sse_base", "sse_combo",
                    "delta_rmse", "delta_mae", "delta_mse", "delta_p90_ae", "delta_cvar90_ae", "frac_improved",
                )}
                keys = list(next(iter(per_seed_draws[part].values())).keys())
                draws = {
                    key: np.mean([per_seed_draws[part][s][key] for s in available], axis=0)
                    for key in keys
                }
                payload["splits"][split]["parts"][part]["seed_mean"] = {
                    "n_seeds": len(available),
                    "seeds": available,
                    "point": point,
                    "bootstrap": S.interval_summary(point, draws),
                }
    return payload


# ---------------------------------------------------------------------------
# 4. 2x2 factorial interaction (random122 only)
# ---------------------------------------------------------------------------


def factorial_interaction(
    units: Mapping[tuple[str, str, int], Unit],
    splits: Mapping[str, FrozenSplit],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """I = MSE(COMBO) - MSE(CHEM8) - MSE(MULTICONTRIB8) + MSE(BASE) on random122."""
    split = "random122"
    arms = ("base", "chem8", "multicontrib", "combo_chem_contrib")
    payload: dict[str, Any] = {
        "definition": "I = MSE(COMBO) - MSE(CHEM8) - MSE(MULTICONTRIB8) + MSE(BASE); negative = favourable MSE interaction",
        "split": split,
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "unit": "Bemis-Murcko scaffold cluster"},
        "seeds": list(SEEDS[split]),
        "parts": {},
    }

    def interaction_point(se: Mapping[str, np.ndarray]) -> dict[str, float]:
        mse = {arm: float(np.mean(se[arm])) for arm in arms}
        i = mse["combo_chem_contrib"] - mse["chem8"] - mse["multicontrib"] + mse["base"]
        return {
            "mse_base": mse["base"],
            "mse_chem8": mse["chem8"],
            "mse_multicontrib": mse["multicontrib"],
            "mse_combo": mse["combo_chem_contrib"],
            "main_effect_chem8": mse["chem8"] - mse["base"],
            "main_effect_multicontrib": mse["multicontrib"] - mse["base"],
            "additive_prediction_mse": mse["base"] + (mse["chem8"] - mse["base"]) + (mse["multicontrib"] - mse["base"]),
            "interaction_mse": i,
        }

    for part in PARTS:
        index = cluster_index_for(splits[split], part)
        se_by_seed: dict[int, dict[str, np.ndarray]] = {}
        ae_by_seed: dict[int, dict[str, np.ndarray]] = {}
        for s in SEEDS[split]:
            entry: dict[str, Any] = {"available": True, "units": {}}
            misses = [arm for arm in arms if (split, arm, s) not in units]
            if misses:
                payload["parts"].setdefault(part, {}).setdefault("per_seed", {})[str(s)] = {
                    "available": False,
                    "missing_arms": misses,
                }
                continue
            se_by_seed[s] = {}
            ae_by_seed[s] = {}
            for arm in arms:
                unit = units[(split, arm, s)]
                ae_by_seed[s][arm] = unit.ae(part)
                se_by_seed[s][arm] = unit.se(part)
                entry["units"][arm] = {"key": unit.key, "origin": unit.origin, "rmse": float(unit.metrics[part]["rmse"])}
            point = interaction_point(se_by_seed[s])
            draws = S.cluster_bootstrap(
                index,
                lambda pos, se=se_by_seed[s]: interaction_point({arm: se[arm][pos] for arm in arms}),
                n_boot=n_boot,
                seed=seed,
            )
            entry["point"] = point
            entry["bootstrap"] = S.interval_summary(point, draws)
            payload["parts"].setdefault(part, {}).setdefault("per_seed", {})[str(s)] = entry

        if se_by_seed:
            seeds_here = sorted(se_by_seed)
            point = {
                key: float(np.mean([interaction_point(se_by_seed[s])[key] for s in seeds_here]))
                for key in interaction_point(se_by_seed[seeds_here[0]])
            }

            def compute(pos: np.ndarray) -> dict[str, float]:
                per_seed = [
                    interaction_point({arm: se_by_seed[s][arm][pos] for arm in arms}) for s in seeds_here
                ]
                return {key: float(np.mean([entry[key] for entry in per_seed])) for key in point}

            draws = S.cluster_bootstrap(index, compute, n_boot=n_boot, seed=seed)
            payload["parts"][part]["seed_mean"] = {
                "n_seeds": len(seeds_here),
                "seeds": seeds_here,
                "point": point,
                "bootstrap": S.interval_summary(point, draws),
            }
        payload["parts"][part]["cluster_profile"] = index.profile()

    # direction consistency across seeds
    consistency: dict[str, Any] = {}
    for part in PARTS:
        per_seed = payload["parts"].get(part, {}).get("per_seed", {})
        values = [v["point"]["interaction_mse"] for v in per_seed.values() if v.get("available")]
        consistency[part] = {
            "n_seeds": len(values),
            "interaction_values": values,
            "all_same_sign": bool(len(values) > 0 and (all(v < 0 for v in values) or all(v > 0 for v in values))),
            "n_negative": int(sum(1 for v in values if v < 0)),
            "n_positive": int(sum(1 for v in values if v > 0)),
        }
    payload["direction_consistency"] = consistency
    return payload


# ---------------------------------------------------------------------------
# 5. tail risk
# ---------------------------------------------------------------------------


def tail_risk(
    units: Mapping[tuple[str, str, int], Unit],
    splits: Mapping[str, FrozenSplit],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cvar90_definition": "mean absolute error of this model's own worst ceil(0.10*n) molecules",
        "posthoc_tail_definition": "top ceil(0.10*n) molecules by same-seed BASE absolute error; label LABEL_SELECTED_POSTHOC_TAIL",
        "warning": (
            "The BASE-defined tail is selected using BASE error, so an improvement inside it is a "
            "post-hoc quantity. It must not be read as a pre-identifiable tail-repair capability and "
            "it is never a substitute for CVaR90-AE or P90 absolute error."
        ),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "unit": "Bemis-Murcko scaffold cluster"},
        "splits": {},
    }
    for split in SPLIT_ORDER:
        payload["splits"][split] = {}
        for part in PARTS:
            index = cluster_index_for(splits[split], part)
            scaffolds = [splits[split].molecules[int(i)].scaffold for i in splits[split].rows[part]]
            block: dict[str, Any] = {"cvar90_by_seed_arm": {}, "posthoc_base_tail": {}, "per_seed": {}}
            for s in SEEDS[split]:
                for arm in ARMS if split == "random122" else Q2_ARMS:
                    unit = units.get((split, arm, s))
                    if unit is None:
                        continue
                    block["cvar90_by_seed_arm"][f"s{s}_{arm}"] = {
                        "cvar90_ae": S.cvar90_ae(unit.ae(part)),
                        "p90_ae": S.p90_abs(unit.ae(part)),
                        "tail_size": S.tail_count(len(unit.y[part])),
                    }
                base = units.get((split, "base", s))
                combo = units.get((split, "combo_chem_contrib", s))
                entry: dict[str, Any] = {"available": bool(base is not None and combo is not None)}
                if base is not None and combo is not None:
                    ae_b, se_b = errors_for(base, part)
                    ae_c, se_c = errors_for(combo, part)
                    k = S.tail_count(len(ae_b))
                    order = S.top_abs_indices(ae_b, k)
                    mask = np.zeros(len(ae_b), dtype=bool)
                    mask[order] = True
                    tail_index = sub_index(index, mask)
                    point = {
                        "tail_n": int(k),
                        "tail_fraction": float(k / len(ae_b)),
                        "rmse_base": S.rmse_from_se(se_b[mask]),
                        "rmse_combo": S.rmse_from_se(se_c[mask]),
                        "mae_base": S.mae_from_ae(ae_b[mask]),
                        "mae_combo": S.mae_from_ae(ae_c[mask]),
                        "sse_base": S.sse_from_se(se_b[mask]),
                        "sse_combo": S.sse_from_se(se_c[mask]),
                        "delta_rmse": S.rmse_from_se(se_c[mask]) - S.rmse_from_se(se_b[mask]),
                        "delta_mae": S.mae_from_ae(ae_c[mask]) - S.mae_from_ae(ae_b[mask]),
                        "delta_sse": S.sse_from_se(se_c[mask]) - S.sse_from_se(se_b[mask]),
                        "frac_improved": float(np.mean(se_c[mask] < se_b[mask])),
                        "mean_abs_improvement": float(np.mean(ae_b[mask] - ae_c[mask])),
                    }
                    draws = S.cluster_bootstrap(
                        tail_index,
                        lambda pos, a=ae_b[mask], b=se_b[mask], c=ae_c[mask], d=se_c[mask]: {
                            "delta_rmse": S.rmse_from_se(d[pos]) - S.rmse_from_se(b[pos]),
                            "delta_mae": S.mae_from_ae(c[pos]) - S.mae_from_ae(a[pos]),
                            "frac_improved": float(np.mean(d[pos] < b[pos])),
                        },
                        n_boot=n_boot,
                        seed=seed,
                    )
                    entry.update(
                        {
                            "label": "LABEL_SELECTED_POSTHOC_TAIL",
                            "selection_rule": "same-seed BASE absolute error, top 10%",
                            "point": point,
                            "bootstrap": S.interval_summary(point, draws),
                            "tail_cluster_profile": tail_index.profile(),
                            "tail_row_ids": [int(splits[split].rows[part][i]) for i in order.tolist()],
                            "tail_scaffolds_distinct": int(len({scaffolds[i] for i in order.tolist()})),
                        }
                    )
                block["posthoc_base_tail"][f"s{s}"] = entry
            payload["splits"][split][part] = block
    return payload


# ---------------------------------------------------------------------------
# 6. subgroup analysis (scaffold seen/unseen + Train-fitted quartiles)
# ---------------------------------------------------------------------------


def _group_stats(
    index: ClusterIndex,
    mask: np.ndarray,
    ae_b: np.ndarray,
    se_b: np.ndarray,
    ae_c: np.ndarray,
    se_c: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "available": False}
    point = {
        "n": n,
        "rmse_base": S.rmse_from_se(se_b[mask]),
        "rmse_combo": S.rmse_from_se(se_c[mask]),
        "mae_base": S.mae_from_ae(ae_b[mask]),
        "mae_combo": S.mae_from_ae(ae_c[mask]),
        "delta_rmse": S.rmse_from_se(se_c[mask]) - S.rmse_from_se(se_b[mask]),
        "delta_mae": S.mae_from_ae(ae_c[mask]) - S.mae_from_ae(ae_b[mask]),
        "frac_improved": float(np.mean(se_c[mask] < se_b[mask])),
    }
    if n < 2:
        return {"n": n, "available": True, "point": point, "bootstrap": None, "low_support": True}
    tail_index = sub_index(index, mask)
    draws = S.cluster_bootstrap(
        tail_index,
        lambda pos, a=ae_b[mask], b=se_b[mask], c=ae_c[mask], d=se_c[mask]: {
            "delta_rmse": S.rmse_from_se(d[pos]) - S.rmse_from_se(b[pos]),
            "delta_mae": S.mae_from_ae(c[pos]) - S.mae_from_ae(b[pos]),
            "frac_improved": float(np.mean(d[pos] < b[pos])),
        },
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "n": n,
        "available": True,
        "low_support": bool(n < 30),
        "point": point,
        "bootstrap": S.interval_summary(point, draws),
        "cluster_profile": tail_index.profile(),
    }


def scaffold_subgroups(
    units: Mapping[tuple[str, str, int], Unit],
    frames: Mapping[str, tuple[pd.DataFrame, dict[str, Any]]],
    splits: Mapping[str, FrozenSplit],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "grouping_rule": (
            "scaffold_seen = Bemis-Murcko scaffold (canonical SMILES fallback for acyclic molecules) "
            "present in the Train split of the same split; quantile boundaries are computed from the "
            "same split's Train molecules only and applied unchanged to Validation and Test"
        ),
        "quartile_edges": {},
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "unit": "Bemis-Murcko scaffold cluster within the subgroup"},
        "splits": {},
    }
    for split in SPLIT_ORDER:
        feature_frame, meta = frames[split]
        payload["quartile_edges"][split] = {
            "edges": meta["quantile_edges"],
            "train_bin_counts": meta["train_bin_counts"],
            "max_train_tanimoto_train_side": meta["max_train_tanimoto_train_side"],
            "max_train_tanimoto_query_side": meta["max_train_tanimoto_query_side"],
        }
        payload["splits"][split] = {}
        for part in PARTS:
            feats = feature_frame[feature_frame["part"] == part].reset_index(drop=True)
            index = cluster_index_for(splits[split], part)
            seeds_here = [
                s
                for s in SEEDS[split]
                if (split, "base", s) in units and (split, "combo_chem_contrib", s) in units
            ]
            block: dict[str, Any] = {"per_seed": {}, "seed_mean": {}}
            groupings = {
                "scaffold_seen": [bool(x) for x in feats["scaffold_seen"]],
                "scaffold_unseen": [not bool(x) for x in feats["scaffold_seen"]],
            }
            for column in ("heavy_atoms", "tpsa", "max_train_tanimoto"):
                labels = feats[f"{column}_bin"].tolist()
                for b in range(1, N_QUANTILE_BINS + 1):
                    groupings[f"{column}_Q{b}"] = [lab == f"{column}_Q{b}" for lab in labels]
            for s in seeds_here:
                base = units[(split, "base", s)]
                combo = units[(split, "combo_chem_contrib", s)]
                ae_b, se_b = errors_for(base, part)
                ae_c, se_c = errors_for(combo, part)
                entry = {}
                for name, mask in groupings.items():
                    entry[name] = _group_stats(
                        index, np.asarray(mask, dtype=bool), ae_b, se_b, ae_c, se_c, n_boot=n_boot, seed=seed
                    )
                block["per_seed"][str(s)] = entry
            if seeds_here:
                entry = {}
                for name, mask in groupings.items():
                    first = block["per_seed"][str(seeds_here[0])][name]
                    n = int(first.get("n", 0))
                    aggregated: dict[str, Any] = {
                        "n": n,
                        "available": bool(first.get("available", False)),
                        "low_support": bool(n < 30),
                        "n_seeds": len(seeds_here),
                        "seed_mean_point": {
                            key: float(
                                np.mean([block["per_seed"][str(s)][name]["point"][key] for s in seeds_here])
                            )
                            for key in (
                                "rmse_base",
                                "rmse_combo",
                                "mae_base",
                                "mae_combo",
                                "delta_rmse",
                                "delta_mae",
                                "frac_improved",
                            )
                        },
                        "seed_mean_bootstrap": _seed_mean_group_bootstrap(
                            index,
                            np.asarray(mask, dtype=bool),
                            units,
                            split,
                            part,
                            seeds_here,
                            n_boot=n_boot,
                            seed=seed,
                        ),
                    }
                    entry[name] = aggregated
                block["seed_mean"] = entry
            payload["splits"][split][part] = block
    return payload


def _seed_mean_group_bootstrap(
    index: ClusterIndex,
    mask: np.ndarray,
    units: Mapping[tuple[str, str, int], Unit],
    split: str,
    part: str,
    seeds_here: Sequence[int],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if int(mask.sum()) < 2:
        return {}
    tail_index = sub_index(index, mask)
    per_seed = {
        s: (
            units[(split, "base", s)].ae(part)[mask],
            units[(split, "base", s)].se(part)[mask],
            units[(split, "combo_chem_contrib", s)].ae(part)[mask],
            units[(split, "combo_chem_contrib", s)].se(part)[mask],
        )
        for s in seeds_here
    }

    def compute(pos: np.ndarray) -> dict[str, float]:
        per = []
        for s in seeds_here:
            ae_b, se_b, ae_c, se_c = per_seed[s]
            per.append(
                {
                    "delta_rmse": S.rmse_from_se(se_c[pos]) - S.rmse_from_se(se_b[pos]),
                    "delta_mae": S.mae_from_ae(ae_c[pos]) - S.mae_from_ae(ae_b[pos]),
                    "frac_improved": float(np.mean(se_c[pos] < se_b[pos])),
                }
            )
        return {key: float(np.mean([p[key] for p in per])) for key in per[0]}

    point = compute(np.arange(int(mask.sum())))
    draws = S.cluster_bootstrap(tail_index, compute, n_boot=n_boot, seed=seed)
    return S.interval_summary(point, draws)


# ---------------------------------------------------------------------------
# 7. concentration diagnostics (top-k SSE contributions)
# ---------------------------------------------------------------------------


def concentration_diagnostics(
    units: Mapping[tuple[str, str, int], Unit], splits: Mapping[str, FrozenSplit]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "definition": (
            "per-molecule squared-error difference (COMBO - BASE) sorted by row id; positive = COMBO "
            "worse. % of net reduction is reported only as a concentration diagnostic."
        ),
        "splits": {},
    }
    for split in SPLIT_ORDER:
        payload["splits"][split] = {}
        for part in PARTS:
            block = {}
            for s in SEEDS[split]:
                base = units.get((split, "base", s))
                combo = units.get((split, "combo_chem_contrib", s))
                if base is None or combo is None:
                    continue
                diff = combo.se(part) - base.se(part)
                order = np.argsort(-diff, kind="stable")
                top5 = order[:5]
                top10 = order[:10]
                net = float(-np.sum(diff))
                block[str(s)] = {
                    "n": int(len(diff)),
                    "net_sse_reduction": net,
                    "gross_positive_sse": float(np.sum(diff[diff > 0])),
                    "gross_negative_sse": float(np.sum(diff[diff < 0])),
                    "n_improved": int(np.sum(diff < 0)),
                    "n_worsened": int(np.sum(diff > 0)),
                    "n_exact_tie": int(np.sum(diff == 0)),
                    "fraction_improved": float(np.mean(diff < 0)),
                    "top5_row_ids": [int(splits[split].rows[part][i]) for i in top5.tolist()],
                    "top5_sse_contribution": [float(diff[i]) for i in top5.tolist()],
                    "top5_share_of_net_reduction": float(np.sum(diff[top5]) / net) if net != 0 else None,
                    "top10_row_ids": [int(splits[split].rows[part][i]) for i in top10.tolist()],
                    "top10_sse_contribution": [float(diff[i]) for i in top10.tolist()],
                    "top10_share_of_net_reduction": float(np.sum(diff[top10]) / net) if net != 0 else None,
                    "top5_negative_row_ids": [int(splits[split].rows[part][i]) for i in np.argsort(diff, kind="stable")[:5].tolist()],
                    "top5_negative_contribution": [float(diff[i]) for i in np.argsort(diff, kind="stable")[:5].tolist()],
                }
            payload["splits"][split][part] = block
    return payload


# ---------------------------------------------------------------------------
# 8. validation/test direction consistency
# ---------------------------------------------------------------------------


def direction_consistency(paired: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        block = paired["splits"][split]["parts"]
        entry: dict[str, Any] = {}
        for s in SEEDS[split]:
            v = block["valid"]["per_seed"].get(str(s), {})
            t = block["test"]["per_seed"].get(str(s), {})
            if not (v.get("available") and t.get("available")):
                entry[str(s)] = {"available": False}
                continue
            entry[str(s)] = {
                "available": True,
                "delta_rmse_valid": v["point"]["delta_rmse"],
                "delta_rmse_test": t["point"]["delta_rmse"],
                "same_sign_rmse": bool(np.sign(v["point"]["delta_rmse"]) == np.sign(t["point"]["delta_rmse"])),
                "delta_mae_valid": v["point"]["delta_mae"],
                "delta_mae_test": t["point"]["delta_mae"],
                "same_sign_mae": bool(np.sign(v["point"]["delta_mae"]) == np.sign(t["point"]["delta_mae"])),
                "delta_cvar90_valid": v["point"]["delta_cvar90_ae"],
                "delta_cvar90_test": t["point"]["delta_cvar90_ae"],
                "same_sign_cvar90": bool(
                    np.sign(v["point"]["delta_cvar90_ae"]) == np.sign(t["point"]["delta_cvar90_ae"])
                ),
            }
        out[split] = entry
    return out


__all__ = [
    "PARTS",
    "metrics_table",
    "per_molecule_table",
    "paired_uncertainty",
    "factorial_interaction",
    "tail_risk",
    "scaffold_subgroups",
    "concentration_diagnostics",
    "direction_consistency",
    "cluster_index_for",
    "sub_index",
]
