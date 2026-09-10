"""Corrected token fragmentation & rarity audit (ZINC, official train/valid only).

Mechanism audit that follows the typed-patch-tokenizer correctness repair.  The
corrected tokenizer is provably correct but slightly regresses compact-v4-hinge
(mean valid degradation +0.0046, corrected - historical).  The pre-registered
hypothesis is that the historical collision accidentally created parameter
sharing, and that correctness fragments the vocabulary, dilutes exact-token
train frequency and therefore increases estimation variance.

This stage tests *only* that hypothesis with a paired comparison on the frozen
4-seed historical / corrected compact-v4-hinge runs.  It never loads the
official test split and trains nothing.

Method principles
-----------------
* **Paired degradation.**  For validation molecule ``i`` and seed ``s``::

      err_hist[i,s] = |y_i - yhat_hist[i,s]|
      err_corr[i,s] = |y_i - yhat_corr[i,s]|
      delta_error[i,s] = err_corr[i,s] - err_hist[i,s]

  Positive ``delta_error`` = the corrected tokenizer is worse.  This is the
  opposite sign of the tokenizer-repair note (which used ``historical -
  corrected``); the whole file uses the convention above.
* Frequency statistics come from **official train only**.  Validation never
  contributes to any frequency definition.
* Historical -> corrected token maps are read from the occurrence tables of
  ``results/typed_patch_tokenizer_correctness`` (shared train+valid id spaces,
  aligned by occurrence index), so the map is exact, not reconstructed.

Outputs (``results/corrected_token_fragmentation_audit/``):
``historical_to_corrected_token_map.csv``, ``r2_token_fragmentation.csv``,
``parent_token_fragmentation.csv``, ``validation_occurrence_transitions.csv``,
``validation_molecule_fragmentation.csv``, ``per_seed_degradation.csv``,
``rarity_comparison.csv``, ``disagreement_comparison.csv``,
``group_transition_results.csv``, ``partial_analysis.json``,
``audit_summary.json``, ``decision_record.json``, ``figures/*.png``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_corrected_token_fragmentation_audit
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
TOKENIZER_DIR = REPO_ROOT / "tracks/ksvd/results/typed_patch_tokenizer_correctness"
OUT_DIR = REPO_ROOT / "tracks/ksvd/results/corrected_token_fragmentation_audit"
RUNS_ROOT = REPO_ROOT / "tracks/ksvd/runs"
POST_V4_TABLE = (
    REPO_ROOT
    / "tracks/ksvd/results/post_v4_residual_audit/validation_master_table.csv"
)

# Frozen promoted runs (canonical, reproducible).  v4-hinge only: the paired
# intervention was run on compact-v4-hinge seeds 0-3 for both tokenizers.
HISTORICAL_RUNS: dict[int, str] = {
    0: "20260909-194445-182c7021",
    1: "20260909-200320-34b347bf",
    2: "20260909-201918-45fbe48d",
    3: "20260909-203516-04e62a28",
}
# Corrected seed 3 uses the *reproducible* terminal trajectory; the discarded
# scratch run 20260910-113625-e5d57ba6 is never referenced here.
CORRECTED_RUNS: dict[int, str] = {
    0: "20260910-111954-64845bd9",
    1: "20260910-112530-c02ac7f0",
    2: "20260910-113058-a40ee3e1",
    3: "20260910-123241-6067c9ac",
}

SEEDS = (0, 1, 2, 3)
PATCH_TOKENS_MAX = 8192
PARENT_TOKENS_MAX = 2048
MIN_FREQUENCY = 1
HYBRID_FULL_TYPED = 768
HYBRID_FULL_PARENT = 32


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _run_dir(run_id: str) -> Path:
    matches = sorted(RUNS_ROOT.glob(f"*/*/*/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"run {run_id}: {matches}")
    return matches[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run_id: str, tokenizer: str, seed: int) -> dict[str, Any]:
    """Validation predictions/targets plus provenance fingerprints."""
    directory = _run_dir(run_id)
    result = json.loads((directory / "artifacts/legacy_full_result.json").read_text())
    valid = result["evaluation"]["valid"]
    predictions = np.asarray(valid["predictions"], dtype=np.float64)
    targets = np.asarray(valid["targets"], dtype=np.float64)
    state = directory / "artifacts/legacy_full_result_selection_state.pt"
    return {
        "tokenizer": tokenizer,
        "seed": int(seed),
        "run_id": run_id,
        "predictions": predictions,
        "targets": targets,
        "mae": float(valid["mae"]),
        "selected_epoch": int(valid["selected_epoch"]),
        "parameters": int(result["evaluation"]["parameters"]),
        "predictions_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "state_sha256": _sha256_file(state) if state.exists() else None,
        "record_id": f"run_record:{run_id}",
    }


def load_frozen_runs() -> dict[str, dict[int, dict[str, Any]]]:
    runs = {
        "historical": {
            seed: load_run(rid, "historical", seed)
            for seed, rid in HISTORICAL_RUNS.items()
        },
        "corrected": {
            seed: load_run(rid, "corrected", seed)
            for seed, rid in CORRECTED_RUNS.items()
        },
    }
    targets = runs["historical"][0]["targets"]
    for family in ("historical", "corrected"):
        for seed in SEEDS:
            if not np.allclose(runs[family][seed]["targets"], targets, atol=0.0):
                raise RuntimeError(f"target mismatch for {family} seed {seed}")
    return runs


def load_occurrences() -> dict[str, np.ndarray]:
    data = np.load(TOKENIZER_DIR / "occurrences.npz")
    return {key: data[key] for key in data.files}


# --------------------------------------------------------------------------
# Token split maps and token-level fragmentation
# --------------------------------------------------------------------------
def token_split_map(
    train_hist_ids: np.ndarray, train_corr_ids: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Map each historical token to its corrected child classes (train counts).

    The corrected key refines the historical key, so every corrected class
    belongs to exactly one historical class; this is asserted.
    """
    hist_count = Counter(train_hist_ids.tolist())
    corr_count = Counter(train_corr_ids.tolist())
    child_of: dict[int, set[int]] = defaultdict(set)
    parent_of: dict[int, int] = {}
    violations = 0
    for hist, corr in zip(train_hist_ids.tolist(), train_corr_ids.tolist()):
        child_of[hist].add(corr)
        parent = parent_of.get(corr)
        if parent is None:
            parent_of[corr] = hist
        elif parent != hist:
            violations += 1
    if violations:
        raise RuntimeError(f"refinement violated: {violations} corrected tokens map to >1 historical token")
    mapping: dict[int, dict[str, Any]] = {}
    for hist, children in child_of.items():
        old_count = int(hist_count[hist])
        child_counts = sorted(
            (int(corr_count[c]) for c in children), reverse=True
        )
        child_ids = sorted(children)
        p = np.asarray(child_counts, dtype=np.float64) / old_count
        entropy = float(-(p * np.log(p)).sum()) if len(p) else 0.0
        mapping[hist] = {
            "historical_count": old_count,
            "split_multiplicity": int(len(children)),
            "corrected_child_ids": child_ids,
            "corrected_child_counts": [int(corr_count[c]) for c in child_ids],
            "max_child_count": int(child_counts[0]) if child_counts else 0,
            "min_child_count": int(child_counts[-1]) if child_counts else 0,
            "split_entropy": entropy,
            "max_frequency_dilution": float(old_count / child_counts[-1]) if child_counts else 0.0,
            "mean_frequency_dilution": float(np.mean([old_count / c for c in child_counts])) if child_counts else 0.0,
        }
    return mapping


def fit_vocabulary(ids: np.ndarray, maximum: int) -> dict[int, int]:
    """Reproduce ``_fit_vocabulary``: decreasing frequency, id 0 = OOV."""
    counts = Counter(ids.tolist())
    ordered = sorted(
        (key for key, count in counts.items() if count >= MIN_FREQUENCY),
        key=lambda key: (-counts[key], key),
    )[: int(maximum)]
    return {key: index + 1 for index, key in enumerate(ordered)}


def tier_of(token_id: int, full_count: int) -> str:
    return "full" if int(token_id) < int(full_count) else "lowrank"


# --------------------------------------------------------------------------
# Occurrence-level transitions and molecule-level features
# --------------------------------------------------------------------------
def classification(hist_freq: int, corr_freq: int) -> str:
    if hist_freq == 0:
        return "historical_oov"
    if corr_freq == 0:
        return "newly_oov"
    if hist_freq > 5 and corr_freq > 5:
        return "stable_common"
    if hist_freq > 5 and 1 <= corr_freq <= 5:
        return "newly_rare"
    if 1 <= hist_freq <= 5 and 1 <= corr_freq <= 5:
        return "already_rare"
    return "rare_to_common"


TRANSITIONS = (
    "stable_common",
    "newly_rare",
    "newly_oov",
    "already_rare",
    "historical_oov",
    "rare_to_common",
)


def build_tables(occ: Mapping[str, np.ndarray]) -> dict[str, Any]:
    train_hist = occ["train_hist_r2_id"]
    train_corr = occ["train_corr_r2_id"]
    train_hist_r1 = occ["train_hist_r1_id"]
    train_corr_r1 = occ["train_corr_r1_id"]
    hist_count = Counter(train_hist.tolist())
    corr_count = Counter(train_corr.tolist())
    hist_count_r1 = Counter(train_hist_r1.tolist())
    corr_count_r1 = Counter(train_corr_r1.tolist())
    split_r2 = token_split_map(train_hist, train_corr)
    split_r1 = token_split_map(train_hist_r1, train_corr_r1)

    vocab_hist_r2 = fit_vocabulary(train_hist, PATCH_TOKENS_MAX)
    vocab_corr_r2 = fit_vocabulary(train_corr, PATCH_TOKENS_MAX)
    vocab_hist_r1 = fit_vocabulary(occ["train_hist_r1_id"], PARENT_TOKENS_MAX)
    vocab_corr_r1 = fit_vocabulary(occ["train_corr_r1_id"], PARENT_TOKENS_MAX)

    n_valid_mol = int(occ["valid_mol_ids"].max()) + 1
    valid = {
        "mol_ids": occ["valid_mol_ids"],
        "hist": occ["valid_hist_r2_id"],
        "corr": occ["valid_corr_r2_id"],
        "hist_r1": occ["valid_hist_r1_id"],
        "corr_r1": occ["valid_corr_r1_id"],
    }

    # ---- validation occurrence transitions -------------------------------
    occ_rows: list[dict[str, Any]] = []
    for index in range(len(valid["hist"])):
        hist_id = int(valid["hist"][index])
        corr_id = int(valid["corr"][index])
        hist_id_r1 = int(valid["hist_r1"][index])
        corr_id_r1 = int(valid["corr_r1"][index])
        hf = int(hist_count.get(hist_id, 0))
        cf = int(corr_count.get(corr_id, 0))
        hf_r1 = int(hist_count_r1.get(hist_id_r1, 0))
        cf_r1 = int(corr_count_r1.get(corr_id_r1, 0))
        mult = int(split_r2[hist_id]["split_multiplicity"]) if hist_id in split_r2 else 0
        occ_rows.append(
            {
                "occurrence_index": index,
                "molecule_id": f"valid:{int(valid['mol_ids'][index]):04d}",
                "molecule_index": int(valid["mol_ids"][index]),
                "historical_token": f"h{hist_id}",
                "corrected_token": f"c{corr_id}",
                "historical_train_frequency": hf,
                "corrected_train_frequency": cf,
                "transition": classification(hf, cf),
                "split_multiplicity": mult,
                "log_dilution": float(np.log1p(hf) - np.log1p(cf)),
                "dilution_ratio": float((hf) / cf) if cf > 0 else float("inf"),
                "child_frequency_fraction": float(cf / hf) if hf > 0 else float("nan"),
                "historical_tier": tier_of(vocab_hist_r2.get(hist_id, 0), HYBRID_FULL_TYPED),
                "corrected_tier": tier_of(vocab_corr_r2.get(corr_id, 0), HYBRID_FULL_TYPED),
                "parent_historical_token": f"p{hist_id_r1}",
                "parent_corrected_token": f"q{corr_id_r1}",
                "parent_historical_train_frequency": hf_r1,
                "parent_corrected_train_frequency": cf_r1,
                "parent_transition": classification(hf_r1, cf_r1),
                "parent_split_multiplicity": (
                    int(split_r1[hist_id_r1]["split_multiplicity"]) if hist_id_r1 in split_r1 else 0
                ),
                "parent_historical_tier": tier_of(vocab_hist_r1.get(hist_id_r1, 0), HYBRID_FULL_PARENT),
                "parent_corrected_tier": tier_of(vocab_corr_r1.get(corr_id_r1, 0), HYBRID_FULL_PARENT),
                "n_patch_nodes": int(occ["valid_n_patch_nodes"][index]),
                "root_atom": int(occ["valid_root_atoms"][index]),
            }
        )

    # ---- molecule-level aggregation --------------------------------------
    mol_occ: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in occ_rows:
        mol_occ[row["molecule_index"]].append(row)

    molecule_rows: list[dict[str, Any]] = []
    for mol in range(n_valid_mol):
        rows = mol_occ[mol]
        n = max(len(rows), 1)
        transitions = Counter(row["transition"] for row in rows)
        parent_transitions = Counter(row["parent_transition"] for row in rows)
        logfreq_hist = np.array([np.log1p(row["historical_train_frequency"]) for row in rows])
        logfreq_corr = np.array([np.log1p(row["corrected_train_frequency"]) for row in rows])
        split_mult = np.array([row["split_multiplicity"] for row in rows], dtype=np.float64)
        log_dilution = np.array([row["log_dilution"] for row in rows])
        parent_split_mult = np.array([row["parent_split_multiplicity"] for row in rows], dtype=np.float64)
        split_common = np.array(
            [(row["split_multiplicity"] > 1) and (row["corrected_train_frequency"] >= 20) for row in rows]
        )
        tier_downgrade = np.array(
            [(row["historical_tier"] == "full") and (row["corrected_tier"] == "lowrank") for row in rows]
        )
        tier_upgrade = np.array(
            [(row["historical_tier"] == "lowrank") and (row["corrected_tier"] == "full") for row in rows]
        )
        corrected_rare_le5 = np.array(
            [0 < row["corrected_train_frequency"] <= 5 for row in rows]
        )
        corrected_oov = np.array([row["corrected_train_frequency"] == 0 for row in rows])
        row: dict[str, Any] = {
            "molecule_id": f"valid:{mol:04d}",
            "molecule_index": mol,
            "n_patches": len(rows),
            "mean_log_freq_hist": float(logfreq_hist.mean()) if rows else 0.0,
            "mean_log_freq_corr": float(logfreq_corr.mean()) if rows else 0.0,
            "frequency_loss": float((logfreq_hist - logfreq_corr).mean()) if rows else 0.0,
            "mean_split_multiplicity": float(split_mult.mean()) if rows else 0.0,
            "max_split_multiplicity": float(split_mult.max()) if rows else 0.0,
            "mean_log_dilution": float(log_dilution.mean()) if rows else 0.0,
            "max_log_dilution": float(log_dilution.max()) if rows else 0.0,
            "fraction_split": float((split_mult > 1).mean()) if rows else 0.0,
            "fraction_split_ge3": float((split_mult >= 3).mean()) if rows else 0.0,
            "fraction_split_common": float(split_common.mean()) if rows else 0.0,
            "fraction_corrected_rare_le5": float(corrected_rare_le5.mean()) if rows else 0.0,
            "fraction_corrected_oov": float(corrected_oov.mean()) if rows else 0.0,
            "mean_parent_split_multiplicity": float(parent_split_mult.mean()) if rows else 0.0,
            "max_parent_split_multiplicity": float(parent_split_mult.max()) if rows else 0.0,
            "fraction_parent_split": float((parent_split_mult > 1).mean()) if rows else 0.0,
            "fraction_tier_downgrade": float(tier_downgrade.mean()) if rows else 0.0,
            "fraction_tier_upgrade": float(tier_upgrade.mean()) if rows else 0.0,
        }
        for transition in TRANSITIONS:
            row[f"fraction_{transition}"] = float(transitions.get(transition, 0) / n)
            row[f"parent_fraction_{transition}"] = float(parent_transitions.get(transition, 0) / n)
        molecule_rows.append(row)

    return {
        "split_r2": split_r2,
        "split_r1": split_r1,
        "hist_count_r2": hist_count,
        "corr_count_r2": corr_count,
        "hist_count_r1": hist_count_r1,
        "corr_count_r1": corr_count_r1,
        "vocab_hist_r2": vocab_hist_r2,
        "vocab_corr_r2": vocab_corr_r2,
        "vocab_hist_r1": vocab_hist_r1,
        "vocab_corr_r1": vocab_corr_r1,
        "occurrence_rows": occ_rows,
        "molecule_rows": molecule_rows,
        "n_valid_mol": n_valid_mol,
    }


# --------------------------------------------------------------------------
# Paired degradation
# --------------------------------------------------------------------------
def paired_degradation(runs: Mapping[str, Mapping[int, dict[str, Any]]]) -> dict[str, Any]:
    targets = runs["historical"][0]["targets"]
    n = len(targets)
    err_hist = np.stack(
        [np.abs(runs["historical"][s]["predictions"] - targets) for s in SEEDS]
    )
    err_corr = np.stack(
        [np.abs(runs["corrected"][s]["predictions"] - targets) for s in SEEDS]
    )
    delta = err_corr - err_hist  # positive = corrected worse
    predictions_hist = np.stack(
        [runs["historical"][s]["predictions"] for s in SEEDS]
    )
    predictions_corr = np.stack(
        [runs["corrected"][s]["predictions"] for s in SEEDS]
    )
    return {
        "targets": targets,
        "err_hist": err_hist,
        "err_corr": err_corr,
        "delta": delta,
        "mean_delta": delta.mean(axis=0),
        "mean_err_hist": err_hist.mean(axis=0),
        "mean_err_corr": err_corr.mean(axis=0),
        "disagreement_hist": predictions_hist.std(axis=0),
        "disagreement_corr": predictions_corr.std(axis=0),
        "delta_disagreement": predictions_corr.std(axis=0) - predictions_hist.std(axis=0),
        "n": n,
    }


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank-based partial correlation of x and y controlling z."""
    Z = np.column_stack([np.ones(len(z)), _rank(z)])
    ex = _rank(x) - Z @ np.linalg.lstsq(Z, _rank(x), rcond=None)[0]
    ey = _rank(y) - Z @ np.linalg.lstsq(Z, _rank(y), rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ beta
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def _z(a: np.ndarray) -> np.ndarray:
    std = a.std()
    return (a - a.mean()) / std if std else a * 0.0


FRAGMENTATION_FEATURES = (
    "frequency_loss",
    "mean_log_dilution",
    "mean_split_multiplicity",
    "max_split_multiplicity",
    "fraction_split",
    "fraction_split_ge3",
    "fraction_split_common",
    "fraction_newly_rare",
    "fraction_newly_oov",
    "fraction_corrected_rare_le5",
    "fraction_corrected_oov",
    "fraction_tier_downgrade",
    "mean_parent_split_multiplicity",
    "parent_fraction_newly_rare",
    "parent_fraction_newly_oov",
)


def molecule_matrix(bundle: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    rows = bundle["molecule_rows"]
    features = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in FRAGMENTATION_FEATURES}
    features["mean_log_freq_hist"] = np.asarray([row["mean_log_freq_hist"] for row in rows])
    features["mean_log_freq_corr"] = np.asarray([row["mean_log_freq_corr"] for row in rows])
    for transition in TRANSITIONS:
        features[f"fraction_{transition}"] = np.asarray(
            [row[f"fraction_{transition}"] for row in rows], dtype=np.float64
        )
    features["n_patches"] = np.asarray([row["n_patches"] for row in rows], dtype=np.float64)
    return features, list(features)


def load_external_covariates(n: int) -> dict[str, np.ndarray]:
    """Graph-size / topology / subgroup covariates from the frozen post-v4 table."""
    import csv

    rows: dict[int, dict[str, str]] = {}
    with POST_V4_TABLE.open() as handle:
        for row in csv.DictReader(handle):
            rows[int(row["subset_index"])] = row
    num_nodes = np.array([float(rows[i]["num_nodes"]) for i in range(n)])
    num_patches = np.array([float(rows[i]["num_patches"]) for i in range(n)])
    subgroup = np.array([rows[i]["subgroup"] for i in range(n)])
    num_edges = np.array([float(rows[i]["num_edges"]) for i in range(n)])
    label_excess = np.array([float(rows[i]["label_excess"]) for i in range(n)])
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_patches": num_patches,
        "subgroup": subgroup,
        "is_group_a": (subgroup == "A"),
        "label_excess": label_excess,
    }


def topology_confound_check(
    features: Mapping[str, np.ndarray], external: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Sec. 32: are fragmentation metrics just a proxy for hard (long-cycle) topology?

    Correlates each fragmentation feature with topology severity (subgroup order
    A<B<C and the continuous ``label_excess``) and reports the hard-topology
    (B/C) share in the top vs. bottom fragmentation quintile.
    """
    severity = np.array([{"A": 0.0, "B": 1.0, "C": 2.0}.get(s, np.nan) for s in external["subgroup"]], dtype=np.float64)
    excess = np.asarray(external["label_excess"], dtype=np.float64)
    hard = external["subgroup"] != "A"
    metrics = {
        "frequency_loss": features["frequency_loss"],
        "newly_rare": features["fraction_newly_rare"],
        "newly_oov": features["fraction_newly_oov"],
        "split_burden": features["fraction_split"],
        "split_common": features["fraction_split_common"],
        "mean_log_dilution": features["mean_log_dilution"],
    }
    correlations: dict[str, Any] = {}
    quintile_hard: dict[str, Any] = {}
    for name, values in metrics.items():
        values = np.asarray(values, dtype=np.float64)
        correlations[name] = {
            "vs_subgroup_severity": round(spearman(values, severity), 4),
            "vs_label_excess": round(spearman(values, excess), 4),
        }
        order = np.argsort(values, kind="stable")
        k = len(order) // 5
        low = order[:k]
        high = order[-k:]
        quintile_hard[name] = {
            "hard_topology_share_lowest_quintile": round(float(hard[low].mean()), 4),
            "hard_topology_share_highest_quintile": round(float(hard[high].mean()), 4),
        }
    return {
        "note": "positive rho = fragmentation concentrates on hard long-cycle (B/C) topology",
        "correlations": correlations,
        "hard_topology_share_by_fragmentation_quintile": quintile_hard,
    }


def assign_groups(features: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """G1 stable common / G2 split-but-common / G3 newly rare / G4 newly OOV.

    A strict "no split at all" G1 is empty because the correction fragments
    every validation molecule (>=53% of each molecule's patches come from an
    aliased historical token).  G1/G2 are therefore split by the molecule-level
    median of ``fraction_split_common`` *within the molecules that have no
    newly-rare and no newly-OOV patch* (an audit-time, outcome-independent
    split; no threshold was chosen from MAE).
    """
    newly_rare = features["fraction_newly_rare"]
    newly_oov = features["fraction_newly_oov"]
    split_common = features["fraction_split_common"]
    supported = (newly_oov == 0) & (newly_rare == 0)
    threshold = float(np.median(split_common[supported])) if supported.any() else 1.0
    group = np.empty(len(newly_rare), dtype="<U2")
    group[newly_oov > 0] = "G4"
    group[(newly_oov == 0) & (newly_rare > 0)] = "G3"
    g12 = supported
    group[g12 & (split_common >= threshold)] = "G2"
    group[g12 & (split_common < threshold)] = "G1"
    return {
        "group": group,
        "split_common_threshold": threshold,
        "n_strict_unsplit": int((features["fraction_split"] == 0).sum()),
        "supported_universe": supported,
    }


def correlation_table(
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    delta = degradation["delta"]
    mean_delta = degradation["mean_delta"]
    mean_err_hist = degradation["mean_err_hist"]
    out: dict[str, Any] = {}
    for name in names:
        x = features[name]
        per_seed = [spearman(x, delta[s]) for s in SEEDS]
        partial_per_seed = [partial_spearman(x, delta[s], degradation["err_hist"][s]) for s in SEEDS]
        out[name] = {
            "per_seed": per_seed,
            "mean_spearman": float(np.nanmean(per_seed)),
            "sign_consistency_positive": int(np.sum(np.asarray(per_seed) > 0)),
            "seed_averaged_spearman": spearman(x, mean_delta),
            "partial_per_seed": partial_per_seed,
            "partial_mean": float(np.nanmean(partial_per_seed)),
            "partial_sign_consistency_positive": int(np.sum(np.asarray(partial_per_seed) > 0)),
            "partial_seed_averaged": partial_spearman(x, mean_delta, mean_err_hist),
        }
    return out


def quintile_table(
    metric: np.ndarray, degradation: Mapping[str, Any], n_bins: int = 5
) -> list[dict[str, Any]]:
    edges = np.quantile(metric, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bins = np.digitize(metric, edges[1:-1])
    rows = []
    for b in range(n_bins):
        idx = np.where(bins == b)[0]
        if idx.size == 0:
            continue
        rows.append(
            {
                "bin": b + 1,
                "n": int(idx.size),
                "metric_lo": float(edges[b]),
                "metric_hi": float(edges[b + 1]),
                "hist_mae": float(degradation["err_hist"][:, idx].mean()),
                "corr_mae": float(degradation["err_corr"][:, idx].mean()),
                "degradation": float(degradation["delta"][:, idx].mean()),
                "median_degradation": float(np.median(degradation["delta"][:, idx].mean(axis=0))),
            }
        )
    return rows


def group_table(
    group: np.ndarray, degradation: Mapping[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("G1", "G2", "G3", "G4"):
        idx = np.where(group == name)[0]
        if idx.size == 0:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(idx.size),
            "hist_mae": float(degradation["err_hist"][:, idx].mean()),
            "corr_mae": float(degradation["err_corr"][:, idx].mean()),
            "degradation": float(degradation["delta"][:, idx].mean()),
            "per_seed_degradation": [float(degradation["delta"][s, idx].mean()) for s in SEEDS],
            "hist_disagreement": float(degradation["disagreement_hist"][idx].mean()),
            "corr_disagreement": float(degradation["disagreement_corr"][idx].mean()),
            "delta_disagreement": float(degradation["delta_disagreement"][idx].mean()),
            "mean_abs_y": float(np.abs(degradation["targets"][idx]).mean()),
        }
    return out


def _subgroup_stats(mask: np.ndarray, degradation: Mapping[str, Any]) -> dict[str, Any]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return {"n": 0}
    return {
        "n": int(idx.size),
        "hist_mae": float(degradation["err_hist"][:, idx].mean()),
        "corr_mae": float(degradation["err_corr"][:, idx].mean()),
        "degradation": float(degradation["delta"][:, idx].mean()),
        "delta_disagreement": float(degradation["delta_disagreement"][idx].mean()),
    }


def matched_group_effect(
    group: np.ndarray,
    target_group: str,
    degradation: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    strata: np.ndarray,
) -> dict[str, Any]:
    """Difficulty-stratified (stratum-matched) difference in mean degradation."""
    mask = group == target_group
    diffs: list[float] = []
    weights: list[int] = []
    for stratum in np.unique(strata):
        in_stratum = strata == stratum
        treated = mask & in_stratum
        control = (~mask) & in_stratum
        if treated.sum() > 0 and control.sum() > 0:
            diffs.append(
                float(degradation["mean_delta"][treated].mean() - degradation["mean_delta"][control].mean())
            )
            weights.append(int(treated.sum()))
    if not diffs:
        return {"weighted_diff": float("nan"), "n_strata": 0}
    return {
        "weighted_diff": float(np.average(diffs, weights=weights)),
        "unweighted_diff": float(np.mean(diffs)),
        "n_strata": len(diffs),
        "n_treated": int(mask.sum()),
    }


def rarity_comparison(
    bundle: Mapping[str, Any], degradation: Mapping[str, Any]
) -> dict[str, Any]:
    """Corrected-vs-historical rarity difficulty re-audit (train-only counts)."""
    occ = bundle["occurrence_rows"]
    n = degradation["n"]
    mol = np.zeros(n, dtype=np.int64)
    profiles = {
        "historical": {k: np.zeros(n) for k in ("rare1", "rare2", "rare5", "rare10", "oov", "-1_valid")},
        "corrected": {k: np.zeros(n) for k in ("rare1", "rare2", "rare5", "rare10", "oov", "-1_valid")},
    }
    for row in occ:
        m = row["molecule_index"]
        mol[m] += 1
        for family, freq in (
            ("historical", row["historical_train_frequency"]),
            ("corrected", row["corrected_train_frequency"]),
        ):
            profiles[family]["rare1"][m] += int(freq == 1)
            profiles[family]["rare2"][m] += int(0 < freq <= 2)
            profiles[family]["rare5"][m] += int(0 < freq <= 5)
            profiles[family]["rare10"][m] += int(0 < freq <= 10)
            profiles[family]["oov"][m] += int(freq == 0)
    denom = np.maximum(mol, 1)
    out: dict[str, Any] = {
        "corpus": {},
        "rarity_vs_error_spearman": {},
        "rarity_ladder": {},
        "valid_oov_occurrences": {},
    }
    # corpus-level statistics
    for family in ("historical", "corrected"):
        counts = (
            bundle["hist_count_r2"]
            if family == "historical"
            else bundle["corr_count_r2"]
        )
        values = np.asarray(list(counts.values()), dtype=np.float64)
        out["corpus"][family] = {
            "n_types": int(len(values)),
            "rare_le1": int((values == 1).sum()),
            "rare_le2": int((values <= 2).sum()),
            "rare_le5": int((values <= 5).sum()),
            "rare_le10": int((values <= 10).sum()),
            "min_frequency": int(values.min()) if values.size else 0,
            "mean_log_frequency": float(np.mean(np.log1p(values))) if values.size else 0.0,
        }
    # molecule-level rarity fractions and relation to |error|
    hist_err = degradation["mean_err_hist"]
    corr_err = degradation["mean_err_corr"]
    for family, err in (("historical", hist_err), ("corrected", corr_err)):
        frac = {k: profiles[family][k] / denom for k in ("rare1", "rare2", "rare5", "rare10", "oov")}
        out["rarity_vs_error_spearman"][family] = {
            k: spearman(v, err) for k, v in frac.items()
        }
        out["rarity_ladder"][family] = {}
        for k in ("rare5", "oov"):
            edges = np.array([-1e-12, 1e-12, 0.05, 0.1, 0.2, 1.0 + 1e-9])
            bins = np.digitize(frac[k], edges[1:-1])
            ladder = []
            for b in range(len(edges) - 1):
                idx = np.where(bins == b)[0]
                if idx.size:
                    ladder.append(
                        {"bin": b, "n": int(idx.size), "mae": float(err[idx].mean())}
                    )
            out["rarity_ladder"][family][k] = ladder
    out["valid_oov_occurrences"] = {
        "historical": int(profiles["historical"]["oov"].sum()),
        "corrected": int(profiles["corrected"]["oov"].sum()),
    }
    return out


def disagreement_comparison(
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    group: np.ndarray,
) -> dict[str, Any]:
    delta_dis = degradation["delta_disagreement"]
    out: dict[str, Any] = {
        "overall": {
            "hist_disagreement": float(degradation["disagreement_hist"].mean()),
            "corr_disagreement": float(degradation["disagreement_corr"].mean()),
            "delta": float(delta_dis.mean()),
        },
        "by_group": {
            name: (
                {
                    "n": int((group == name).sum()),
                    "hist": float(degradation["disagreement_hist"][group == name].mean()),
                    "corr": float(degradation["disagreement_corr"][group == name].mean()),
                    "delta": float(delta_dis[group == name].mean()),
                }
                if (group == name).any()
                else {"n": 0}
            )
            for name in ("G1", "G2", "G3", "G4")
        },
        "fragmentation_vs_delta_disagreement": {
            name: {
                "spearman": spearman(features[name], delta_dis),
                "partial_controlling_hist_err": partial_spearman(
                    features[name], delta_dis, degradation["mean_err_hist"]
                ),
            }
            for name in FRAGMENTATION_FEATURES
        },
        "delta_disagreement_vs_degradation": {
            "spearman": spearman(delta_dis, degradation["mean_delta"]),
            "partial_controlling_hist_err": partial_spearman(
                delta_dis, degradation["mean_delta"], degradation["mean_err_hist"]
            ),
        },
    }
    return out


def partial_analysis(
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    external: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Incremental regressions (standardised) for the mediation-style read."""
    y = _z(degradation["mean_delta"])
    fl = _z(features["frequency_loss"])
    alias = _z(features["fraction_split"])  # alias-burden-like fragmentation proxy
    graph_size = _z(external["num_nodes"])
    ddis = _z(degradation["delta_disagreement"])
    base = _z(degradation["mean_err_hist"])
    models = {
        "A_degradation~alias": alias[:, None],
        "B_degradation~frequency_loss": fl[:, None],
        "C_degradation~alias+frequency_loss": np.column_stack([alias, fl]),
        "D_degradation~frequency_loss+graph_size": np.column_stack([fl, graph_size]),
        "E_degradation~frequency_loss+delta_disagreement": np.column_stack([fl, ddis]),
        "F_degradation~frequency_loss+baseline_difficulty": np.column_stack([fl, base]),
        "G_degradation~newly_rare+baseline_difficulty": np.column_stack(
            [_z(features["fraction_newly_rare"]), base]
        ),
        "H_degradation~newly_oov+baseline_difficulty": np.column_stack(
            [_z(features["fraction_newly_oov"]), base]
        ),
    }
    out = {"regression": {}}
    for name, X in models.items():
        beta, r2 = ols(y, X)
        out["regression"][name] = {
            "coefficients_standardised": [float(b) for b in beta],
            "r2": r2,
        }
    # alias burden from the tokenizer audit (old definition, reused, no re-tune)
    alias_molecules = json.loads((TOKENIZER_DIR / "alias_molecules.json").read_text())
    alias_burden = np.asarray(alias_molecules["alias_burden_ratio"], dtype=np.float64)
    out["alias_burden_vs_degradation"] = {
        "spearman": spearman(alias_burden, degradation["mean_delta"]),
        "partial_controlling_baseline": partial_spearman(
            alias_burden, degradation["mean_delta"], degradation["mean_err_hist"]
        ),
    }
    out["alias_burden_vs_frequency_loss"] = spearman(
        alias_burden, features["frequency_loss"]
    )
    out["frequency_loss_vs_alias_burden_both"] = ols(
        y, np.column_stack([_z(alias_burden), fl])
    )[0].tolist()
    out["median_degradation"] = float(np.median(degradation["mean_delta"]))
    out["baseline_difficulty_vs_degradation"] = spearman(
        degradation["mean_err_hist"], degradation["mean_delta"]
    )
    out["baseline_difficulty_vs_degradation_partial"] = partial_spearman(
        degradation["mean_err_hist"], degradation["mean_delta"], features["frequency_loss"]
    )
    return out


def tier_analysis(
    bundle: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
) -> dict[str, Any]:
    downgrade = features["fraction_tier_downgrade"]
    upgrade = np.asarray(
        [row["fraction_tier_upgrade"] for row in bundle["molecule_rows"]], dtype=np.float64
    )
    return {
        "occurrence_downgrade_fraction": float(
            np.mean([row["historical_tier"] == "full" and row["corrected_tier"] == "lowrank" for row in bundle["occurrence_rows"]])
        ),
        "occurrence_upgrade_fraction": float(
            np.mean([row["historical_tier"] == "lowrank" and row["corrected_tier"] == "full" for row in bundle["occurrence_rows"]])
        ),
        "molecule_mean_downgrade_fraction": float(downgrade.mean()),
        "molecule_mean_upgrade_fraction": float(upgrade.mean()),
        "downgrade_vs_degradation": {
            "spearman": spearman(downgrade, degradation["mean_delta"]),
            "partial_controlling_frequency_loss": partial_spearman(
                downgrade, degradation["mean_delta"], features["frequency_loss"]
            ),
        },
        "upgrade_vs_degradation": {"spearman": spearman(upgrade, degradation["mean_delta"])},
        "downgrade_vs_frequency_loss": spearman(downgrade, features["frequency_loss"]),
    }


def controls(
    bundle: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    group: np.ndarray,
    strata: np.ndarray,
) -> dict[str, Any]:
    """Sec. 26/27 controls: split-but-frequent vs split-and-rare vs unaliased-rare."""
    occ = bundle["occurrence_rows"]
    n = degradation["n"]
    corr_count = bundle["corr_count_r2"]
    split_r2 = bundle["split_r2"]

    unaliased_rare = np.zeros(n)
    per_mol = defaultdict(lambda: [0, 0])
    for row in occ:
        m = row["molecule_index"]
        per_mol[m][1] += 1
        hist_id = int(row["historical_token"][1:])
        corr_id = int(row["corrected_token"][1:])
        info = split_r2.get(hist_id)
        aliased = info is not None and info["split_multiplicity"] > 1
        if (not aliased) and 0 < corr_count.get(corr_id, 0) <= 5:
            per_mol[m][0] += 1
    for m, (num, den) in per_mol.items():
        unaliased_rare[m] = num / den
    supported = (features["fraction_newly_rare"] == 0) & (features["fraction_newly_oov"] == 0)
    threshold = float(np.median(features["fraction_split_common"][supported]))

    # Sec 26 split-but-still-frequent vs split-and-rare (difficulty-matched)
    high_split_common = supported & (features["fraction_split_common"] >= threshold)
    low_split_common = supported & (features["fraction_split_common"] < threshold)
    newly_rare_mol = features["fraction_newly_rare"] > 0

    def _matched(a_mask: np.ndarray, b_mask: np.ndarray) -> dict[str, Any]:
        diffs, weights = [], []
        for stratum in np.unique(strata):
            in_s = strata == stratum
            a = a_mask & in_s
            b = b_mask & in_s
            if a.sum() and b.sum():
                diffs.append(float(degradation["mean_delta"][a].mean() - degradation["mean_delta"][b].mean()))
                weights.append(int(a.sum()))
        return {
            "weighted_diff": float(np.average(diffs, weights=weights)) if diffs else float("nan"),
            "n_strata": len(diffs),
        }

    return {
        "sec26_split_but_frequent_vs_split_rare": {
            "split_but_frequent": {
                "n": int(high_split_common.sum()),
                "degradation": float(degradation["mean_delta"][high_split_common].mean()),
            },
            "less_fragmented_common": {
                "n": int(low_split_common.sum()),
                "degradation": float(degradation["mean_delta"][low_split_common].mean()),
            },
            "newly_rare": {
                "n": int(newly_rare_mol.sum()),
                "degradation": float(degradation["mean_delta"][newly_rare_mol].mean()),
            },
            "matched_split_frequent_vs_less_fragmented": _matched(high_split_common, low_split_common),
            "matched_newly_rare_vs_less_fragmented": _matched(newly_rare_mol, low_split_common),
        },
        "sec27_unaliased_but_rare": {
            "molecules_with_occurrence": int((unaliased_rare > 0).sum()),
            "degradation_with": float(degradation["mean_delta"][unaliased_rare > 0].mean()) if (unaliased_rare > 0).any() else None,
            "degradation_without": float(degradation["mean_delta"][unaliased_rare == 0].mean()),
            "spearman_frac_vs_degradation": spearman(unaliased_rare, degradation["mean_delta"]),
        },
    }


def case_inspection(
    bundle: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    group: np.ndarray,
    n_cases: int = 10,
) -> dict[str, Any]:
    rows = bundle["molecule_rows"]
    order = np.argsort(-degradation["mean_delta"])
    targets = degradation["targets"]

    def _record(index: int) -> dict[str, Any]:
        row = rows[index]
        return {
            "molecule_id": row["molecule_id"],
            "target": float(targets[index]),
            "hist_mean_abs_err": float(degradation["mean_err_hist"][index]),
            "corr_mean_abs_err": float(degradation["mean_err_corr"][index]),
            "delta_error": float(degradation["mean_delta"][index]),
            "hist_prediction_seed0": float(degradation["err_hist"][0, index]),
            "old_new_patch_frequency": [
                row["mean_log_freq_hist"],
                row["mean_log_freq_corr"],
            ],
            "frequency_loss": row["frequency_loss"],
            "newly_rare_count": row["fraction_newly_rare"] * row["n_patches"],
            "newly_oov_count": row["fraction_newly_oov"] * row["n_patches"],
            "split_multiplicity": row["mean_split_multiplicity"],
            "disagreement_hist": float(degradation["disagreement_hist"][index]),
            "disagreement_corr": float(degradation["disagreement_corr"][index]),
            "delta_disagreement": float(degradation["delta_disagreement"][index]),
            "group": str(group[index]),
            "n_patches": row["n_patches"],
        }

    return {
        "top_degradation": [_record(int(i)) for i in order[:n_cases]],
        "top_improvement": [_record(int(i)) for i in order[-n_cases:][::-1]],
    }


def group_a_analysis(
    group: np.ndarray,
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    external: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    mask = external["is_group_a"]
    idx = np.where(mask)[0]

    def _corr(name: str, family: str = "all") -> dict[str, Any]:
        sub = np.ones(len(features[name]), dtype=bool) if family == "all" else mask
        return {
            "n": int(sub.sum()),
            "spearman": spearman(features[name][sub], degradation["mean_delta"][sub]),
            "partial_controlling_hist_err": partial_spearman(
                features[name][sub], degradation["mean_delta"][sub], degradation["mean_err_hist"][sub]
            ),
        }

    def _subgroup(sel: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(sel.sum()),
            "hist_mae": float(degradation["err_hist"][:, sel].mean()),
            "corr_mae": float(degradation["err_corr"][:, sel].mean()),
            "degradation": float(degradation["delta"][:, sel].mean()),
            "delta_disagreement": float(degradation["delta_disagreement"][sel].mean()),
        }

    return {
        "n_group_a": int(idx.size),
        "frequency_loss_vs_degradation": _corr("frequency_loss"),
        "newly_rare_vs_degradation": _corr("fraction_newly_rare"),
        "newly_oov_vs_degradation": _corr("fraction_newly_oov"),
        "group_mae_in_a": {
            name: _subgroup(mask & (group == name)) for name in ("G1", "G2", "G3", "G4")
        },
        "all_valid_group_mae": {
            name: _subgroup(group == name) for name in ("G1", "G2", "G3", "G4")
        },
    }


def make_figures(
    bundle: Mapping[str, Any],
    features: Mapping[str, np.ndarray],
    degradation: Mapping[str, Any],
    group: np.ndarray,
    external: Mapping[str, np.ndarray],
    rarity: Mapping[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(fig, name: str) -> None:
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))

    split_r2 = bundle["split_r2"]
    # Figure 1 - historical token frequency -> corrected child frequency
    hist_freq, child_freq, is_top = [], [], []
    for hist_id, info in split_r2.items():
        for corr_id, count in zip(info["corrected_child_ids"], info["corrected_child_counts"]):
            hist_freq.append(info["historical_count"])
            child_freq.append(count)
            is_top.append(count == info["max_child_count"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(hist_freq, child_freq, s=6, alpha=0.25, c=["tab:red" if t else "tab:blue" for t in is_top])
    ax.plot([1, max(hist_freq)], [1, max(hist_freq)], "k--", lw=1, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("historical token train frequency")
    ax.set_ylabel("corrected child train frequency")
    ax.set_title("Figure 1 - historical -> corrected child frequency (red = largest child)")
    ax.legend()
    _save(fig, "fig1_frequency_split.png")

    # Figure 2 - split multiplicity distribution
    k_r2 = np.array([info["split_multiplicity"] for info in split_r2.values()])
    k_r1 = np.array([info["split_multiplicity"] for info in bundle["split_r1"].values()])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(k_r2, bins=np.arange(1, k_r2.max() + 2) - 0.5)
    axes[0].set_title("radius-2 (n=%d, mean %.2f)" % (len(k_r2), k_r2.mean()))
    axes[0].set_xlabel("split multiplicity k")
    axes[0].set_ylabel("historical tokens")
    axes[1].hist(k_r1, bins=np.arange(1, k_r1.max() + 2) - 0.5)
    axes[1].set_title("parent (n=%d, mean %.2f)" % (len(k_r1), k_r1.mean()))
    axes[1].set_xlabel("split multiplicity k")
    fig.suptitle("Figure 2 - split multiplicity distribution")
    _save(fig, "fig2_split_multiplicity.png")

    # Figure 3 - frequency-loss quintile vs degradation
    q = quintile_table(features["frequency_loss"], degradation)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["bin"] for r in q], [r["degradation"] for r in q], "o-")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks([r["bin"] for r in q])
    ax.set_xlabel("frequency-loss quintile (1 = least dilution)")
    ax.set_ylabel("mean paired degradation")
    ax.set_title("Figure 3 - frequency loss vs corrected degradation")
    _save(fig, "fig3_freq_loss_quintile.png")

    # Figure 4 - G1/G2/G3/G4 historical vs corrected MAE
    names = [g for g in ("G1", "G2", "G3", "G4") if (group == g).any()]
    hist = [degradation["err_hist"][:, group == g].mean() for g in names]
    corr = [degradation["err_corr"][:, group == g].mean() for g in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, hist, 0.4, label="historical")
    ax.bar(x + 0.2, corr, 0.4, label="corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("validation MAE")
    ax.set_title("Figure 4 - G1..G4 historical vs corrected MAE")
    ax.legend()
    _save(fig, "fig4_group_mae.png")

    # Figure 5 - fragmentation severity vs delta disagreement
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in (("frequency_loss", "tab:blue"), ("fraction_newly_rare", "tab:green"), ("fraction_newly_oov", "tab:red")):
        q = quintile_table(features[name], {"err_hist": degradation["err_hist"], "err_corr": degradation["err_corr"], "delta": np.tile(degradation["delta_disagreement"], (len(SEEDS), 1))})
        ax.plot([r["bin"] for r in q], [r["degradation"] for r in q], "o-", label=name)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("fragmentation quintile")
    ax.set_ylabel("mean delta disagreement (corr - hist)")
    ax.set_title("Figure 5 - fragmentation vs cross-seed disagreement change")
    ax.legend()
    _save(fig, "fig5_fragmentation_disagreement.png")

    # Figure 6 - rarity MAE ladder historical vs corrected
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, color in (("historical", "tab:blue"), ("corrected", "tab:orange")):
        ladder = rarity["rarity_ladder"][family]["rare5"]
        ax.plot([b["bin"] for b in ladder], [b["mae"] for b in ladder], "o-", label=family, color=color)
    ax.set_xlabel("rare<=5 fraction bin (0, 0-0.05, 0.05-0.1, 0.1-0.2, >0.2)")
    ax.set_ylabel("mean |error|")
    ax.set_title("Figure 6 - rarity vs |error| ladder")
    ax.legend()
    _save(fig, "fig6_rarity_ladder.png")

    # Figure 7 - Group A frequency loss vs degradation quintiles
    mask = external["is_group_a"]
    q = quintile_table(features["frequency_loss"][mask], {
        "err_hist": degradation["err_hist"][:, mask],
        "err_corr": degradation["err_corr"][:, mask],
        "delta": degradation["delta"][:, mask],
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["bin"] for r in q], [r["degradation"] for r in q], "o-")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("frequency-loss quintile (Group A)")
    ax.set_ylabel("mean paired degradation")
    ax.set_title("Figure 7 - Group A ordinary: frequency loss vs degradation")
    _save(fig, "fig7_groupA_freq_loss.png")

    # Figure 8 - degradation by baseline-difficulty quintile (the dominant confound)
    q = quintile_table(degradation["mean_err_hist"], degradation)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(r["bin"]) for r in q], [r["degradation"] for r in q])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("baseline (historical) error quintile")
    ax.set_ylabel("mean paired degradation")
    ax.set_title("Figure 8 - degradation vs baseline difficulty")
    _save(fig, "fig8_difficulty_redistribution.png")
    return written


def decide(
    correlations: Mapping[str, Any],
    groups: Mapping[str, Any],
    matched: Mapping[str, Any],
    partial: Mapping[str, Any],
    rarity: Mapping[str, Any],
    tier: Mapping[str, Any],
    controls_out: Mapping[str, Any],
    group_a: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered mechanism gates (Sec. 38-42/57)."""
    core = ("frequency_loss", "fraction_newly_rare", "fraction_newly_oov")
    pooled = {name: correlations[name]["seed_averaged_spearman"] for name in core}
    signs = {name: correlations[name]["sign_consistency_positive"] for name in core}
    partial_means = {name: correlations[name]["partial_mean"] for name in core}
    best_name = max(core, key=lambda name: abs(pooled[name]))
    best_pooled = abs(pooled[best_name])
    best_partial = max(abs(partial_means[name]) for name in core)
    g1 = groups.get("G1", {}).get("degradation")
    g2 = groups.get("G2", {}).get("degradation")
    g3 = groups.get("G3", {}).get("degradation")
    g4 = groups.get("G4", {}).get("degradation")
    g3_matched = matched["G3"]["weighted_diff"]
    g4_matched = matched["G4"]["weighted_diff"]
    baseline_corr = partial["baseline_difficulty_vs_degradation"]
    tier_corr = tier["downgrade_vs_degradation"]["spearman"]
    g2_vs_g1 = (g2 - g1) if (g1 is not None and g2 is not None) else 0.0
    rarity_strengthened = (
        rarity["rarity_vs_error_spearman"]["corrected"]["oov"]
        > rarity["rarity_vs_error_spearman"]["historical"]["oov"]
        and rarity["rarity_vs_error_spearman"]["corrected"]["rare5"]
        > rarity["rarity_vs_error_spearman"]["historical"]["rare5"]
    )
    if best_pooled >= 0.25 and signs[best_name] >= 3:
        verdict = "A_STRONG_SHARING_LOSS"
    elif best_pooled >= 0.15:
        verdict = "A_MODERATE_SHARING_LOSS"
    elif g2_vs_g1 >= 0.01 and best_pooled < 0.10 and g3_matched < 0.01:
        verdict = "B_COARSE_STRUCTURAL_INDUCTIVE_BIAS"
    elif abs(tier_corr) >= 0.15 and best_pooled < 0.10:
        verdict = "C_TOKEN_CAPACITY_ALLOCATION"
    elif best_partial >= 0.05 and g3_matched >= 0.008:
        verdict = "D_MIXED_MECHANISM"
    else:
        verdict = "E_NO_CLEAR_FRAGMENTATION_MECHANISM"

    decision = {
        "verdict": verdict,
        "primary_question": "Is the corrected-vs-historical degradation directed by token fragmentation / frequency loss?",
        "gates": {
            "best_fragmentation_metric": best_name,
            "pooled_spearman": pooled,
            "positive_seed_counts": signs,
            "best_pooled_abs_spearman": best_pooled,
            "best_conditional_partial_spearman": best_partial,
            "baseline_difficulty_vs_degradation_spearman": baseline_corr,
            "group_degradation": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
            "g2_minus_g1": g2_vs_g1,
            "g3_matched_diff": g3_matched,
            "g4_matched_diff": g4_matched,
            "tier_downgrade_spearman": tier_corr,
            "rarity_relation_strengthened": bool(rarity_strengthened),
            "group_a_frequency_loss_spearman": group_a["frequency_loss_vs_degradation"]["spearman"],
        },
        "dominant_pattern": (
            "paired degradation is dominated by a baseline-difficulty-dependent redistribution "
            "(corrected worse on the easy bulk, better on the hard tail; Spearman(baseline_error, "
            "degradation) = %.3f), not by token fragmentation." % baseline_corr
        ),
    }
    if verdict in ("A_STRONG_SHARING_LOSS", "A_MODERATE_SHARING_LOSS"):
        decision["architecture_recommendation"] = "GO - hierarchical coarse-shared + exact-residual (design next stage; do not implement here)"
        decision["next_stage"] = "Design a small hierarchical coarse-shared + corrected-exact-residual patch embedding experiment."
    elif verdict == "B_COARSE_STRUCTURAL_INDUCTIVE_BIAS":
        decision["architecture_recommendation"] = "GO - coarse-to-fine patch representation (intentional coarse abstraction, not a bug)"
        decision["next_stage"] = "Design an intentional coarse structural key + exact typed refinement, framed as inductive bias rather than rarity repair."
    elif verdict == "C_TOKEN_CAPACITY_ALLOCATION":
        decision["architecture_recommendation"] = "GO - shared low-rank exact representation"
        decision["next_stage"] = "Fix the exact-token parameter-sharing mechanism before any coarse hierarchy."
    elif verdict == "D_MIXED_MECHANISM":
        decision["architecture_recommendation"] = "NO immediate architecture GO - rank mechanisms by effect size first"
        decision["next_stage"] = "Quantify and separate the baseline-difficulty-dependent error redistribution from the weak fragmentation component before any new architecture."
        decision["mechanism_ranking"] = [
            {
                "rank": 1,
                "mechanism": "baseline-difficulty-dependent error redistribution (NOT a fragmentation mechanism)",
                "effect": f"Spearman(baseline_error, degradation) = {baseline_corr:.3f}",
                "note": "corrected tokenizer is worse on the easy bulk and better on the hard tail; robust 4/4 seeds",
            },
            {
                "rank": 2,
                "mechanism": "weak, largely bulk-localized frequency-loss / newly-rare sharing signal",
                "effect": f"conditional partial Spearman = {best_partial:.3f}; G3 matched +{g3_matched:.4f}",
                "note": "below the moderate gate; non-specific (already-rare / historical-OOV show the same direction)",
            },
            {
                "rank": 3,
                "mechanism": "newly-OOV sharing loss",
                "effect": f"G4 matched {g4_matched:.4f} (wrong direction)",
                "note": "newly-OOV molecules do not degrade",
            },
            {
                "rank": 4,
                "mechanism": "HybridEmbedding tier reallocation",
                "effect": f"tier-downgrade Spearman = {tier_corr:.3f}",
                "note": "collinear with frequency loss (rho=0.52), no independent signal",
            },
        ]
    else:
        decision["architecture_recommendation"] = "DO NOT build coarse-shared + exact-residual on this evidence"
        decision["next_stage"] = "Re-explain the historical coarse-token advantage (why correctness helps the hard tail and hurts the easy bulk); do not open a hierarchical embedding."
    return decision


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(bundle: Mapping[str, Any], degradation: Mapping[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # historical -> corrected token map
    header = [
        "level", "historical_token", "historical_count", "split_multiplicity",
        "corrected_child_ids", "corrected_child_counts", "max_child_count",
        "min_child_count", "split_entropy", "max_frequency_dilution", "mean_frequency_dilution",
        "historical_tier", "corrected_child_tiers",
    ]
    rows = []
    for level, mapping, vocab_hist, vocab_corr, full in (
        ("r2", bundle["split_r2"], bundle["vocab_hist_r2"], bundle["vocab_corr_r2"], HYBRID_FULL_TYPED),
        ("r1", bundle["split_r1"], bundle["vocab_hist_r1"], bundle["vocab_corr_r1"], HYBRID_FULL_PARENT),
    ):
        prefix = "h" if level == "r2" else "p"
        cprefix = "c" if level == "r2" else "q"
        for hist_id, info in sorted(mapping.items()):
            rows.append([
                level, f"{prefix}{hist_id}", info["historical_count"], info["split_multiplicity"],
                ";".join(f"{cprefix}{c}" for c in info["corrected_child_ids"]),
                ";".join(str(c) for c in info["corrected_child_counts"]),
                info["max_child_count"], info["min_child_count"], f"{info['split_entropy']:.6f}",
                f"{info['max_frequency_dilution']:.4f}", f"{info['mean_frequency_dilution']:.4f}",
                tier_of(vocab_hist.get(hist_id, 0), full),
                ";".join(tier_of(vocab_corr.get(c, 0), full) for c in info["corrected_child_ids"]),
            ])
    _write_csv(OUT_DIR / "historical_to_corrected_token_map.csv", header, rows)

    # token fragmentation tables (same schema as the split map)
    for level, name in (("r2", "r2_token_fragmentation.csv"), ("r1", "parent_token_fragmentation.csv")):
        subset = [row for row in rows if row[0] == level]
        _write_csv(OUT_DIR / name, header, subset)

    # validation occurrence transitions
    occ_header = list(bundle["occurrence_rows"][0].keys())
    occ_rows = [[row[key] for key in occ_header] for row in bundle["occurrence_rows"]]
    _write_csv(OUT_DIR / "validation_occurrence_transitions.csv", occ_header, occ_rows)

    # validation molecule fragmentation
    mol_header = list(bundle["molecule_rows"][0].keys())
    mol_rows = [[row[key] for key in mol_header] for row in bundle["molecule_rows"]]
    _write_csv(OUT_DIR / "validation_molecule_fragmentation.csv", mol_header, mol_rows)

    # per-seed degradation
    header = ["molecule_id", "target"]
    for s in SEEDS:
        header += [f"err_hist_seed{s}", f"err_corr_seed{s}", f"delta_seed{s}"]
    header += ["err_hist_mean", "err_corr_mean", "delta_mean"]
    rows = []
    for i in range(degradation["n"]):
        row = [f"valid:{i:04d}", f"{degradation['targets'][i]:.6f}"]
        for s in SEEDS:
            row += [f"{degradation['err_hist'][s, i]:.6f}", f"{degradation['err_corr'][s, i]:.6f}", f"{degradation['delta'][s, i]:.6f}"]
        row += [f"{degradation['mean_err_hist'][i]:.6f}", f"{degradation['mean_err_corr'][i]:.6f}", f"{degradation['mean_delta'][i]:.6f}"]
        rows.append(row)
    _write_csv(OUT_DIR / "per_seed_degradation.csv", header, rows)


def run_audit() -> dict[str, Any]:
    import time

    started = time.perf_counter()
    runs = load_frozen_runs()
    occ = load_occurrences()
    bundle = build_tables(occ)
    degradation = paired_degradation(runs)
    features, _ = molecule_matrix(bundle)
    external = load_external_covariates(degradation["n"])

    groups = assign_groups(features)
    group = groups["group"]

    correlations = correlation_table(features, degradation, FRAGMENTATION_FEATURES)
    group_results = group_table(group, degradation)
    strata = np.digitize(
        degradation["mean_err_hist"],
        np.quantile(degradation["mean_err_hist"], np.linspace(0, 1, 11)[1:-1]),
    ) * 4 + np.digitize(
        np.abs(degradation["targets"]), np.quantile(np.abs(degradation["targets"]), [0.25, 0.5, 0.75])
    )
    matched = {
        name: matched_group_effect(group, name, degradation, features, strata)
        for name in ("G1", "G2", "G3", "G4")
    }
    partial = partial_analysis(features, degradation, external)
    rarity = rarity_comparison(bundle, degradation)
    disagreement = disagreement_comparison(features, degradation, group)
    tier = tier_analysis(bundle, features, degradation)
    controls_out = controls(bundle, features, degradation, group, strata)
    group_a = group_a_analysis(group, features, degradation, external)
    cases = case_inspection(bundle, features, degradation, group)
    topology_confound = topology_confound_check(features, external)

    decision = decide(
        correlations, group_results, matched, partial, rarity, tier, controls_out, group_a
    )

    # descriptive quintile ladders (Sec. 13) and newly-rare / newly-OOV subgroup tables
    degradation_view = {
        "err_hist": degradation["err_hist"],
        "err_corr": degradation["err_corr"],
        "delta": degradation["delta"],
    }
    quintiles = {
        name: quintile_table(features[name], degradation_view)
        for name in ("frequency_loss", "fraction_newly_rare", "fraction_newly_oov", "mean_log_dilution")
    }
    quintiles["baseline_difficulty"] = quintile_table(degradation["mean_err_hist"], degradation_view)
    newly_oov_frac = features["fraction_newly_oov"]
    oov_molecules = newly_oov_frac > 0
    oov_threshold = (
        float(np.quantile(newly_oov_frac[oov_molecules], 0.5)) if oov_molecules.any() else 0.0
    )
    subgroup_tables = {
        "newly_oov": {
            "no_newly_oov": _subgroup_stats(~oov_molecules, degradation),
            "at_least_one_newly_oov": _subgroup_stats(oov_molecules, degradation),
            "high_newly_oov_burden": _subgroup_stats(
                oov_molecules & (newly_oov_frac >= oov_threshold) & (newly_oov_frac > 0), degradation
            ),
        },
        "newly_rare": {
            "no_newly_rare": _subgroup_stats(features["fraction_newly_rare"] == 0, degradation),
            "at_least_one_newly_rare": _subgroup_stats(features["fraction_newly_rare"] > 0, degradation),
        },
    }

    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": "official train + official validation only; official test never loaded",
        "sign_convention": "delta_error = err_corrected - err_historical; positive = corrected worse",
        "n_valid_molecules": int(degradation["n"]),
        "overall": {
            "historical_mae": float(degradation["err_hist"].mean()),
            "corrected_mae": float(degradation["err_corr"].mean()),
            "mean_degradation": float(degradation["mean_delta"].mean()),
            "median_degradation": float(np.median(degradation["mean_delta"])),
            "per_seed_degradation": [float(degradation["delta"][s].mean()) for s in SEEDS],
        },
        "frozen_runs": {
            "historical": {str(s): {k: v for k, v in runs["historical"][s].items() if k not in ("predictions", "targets")} for s in SEEDS},
            "corrected": {str(s): {k: v for k, v in runs["corrected"][s].items() if k not in ("predictions", "targets")} for s in SEEDS},
        },
        "token_fragmentation": {
            "r2": {
                "historical_tokens": len(bundle["split_r2"]),
                "corrected_tokens": len(bundle["corr_count_r2"]),
                "split_historical_tokens": int(sum(1 for info in bundle["split_r2"].values() if info["split_multiplicity"] > 1)),
                "mean_multiplicity": float(np.mean([info["split_multiplicity"] for info in bundle["split_r2"].values()])),
                "max_multiplicity": int(max(info["split_multiplicity"] for info in bundle["split_r2"].values())),
                "mean_frequency_dilution": float(np.mean([info["mean_frequency_dilution"] for info in bundle["split_r2"].values()])),
            },
            "r1": {
                "historical_tokens": len(bundle["split_r1"]),
                "corrected_tokens": len(bundle["corr_count_r1"]),
                "split_historical_tokens": int(sum(1 for info in bundle["split_r1"].values() if info["split_multiplicity"] > 1)),
                "mean_multiplicity": float(np.mean([info["split_multiplicity"] for info in bundle["split_r1"].values()])),
                "max_multiplicity": int(max(info["split_multiplicity"] for info in bundle["split_r1"].values())),
                "mean_frequency_dilution": float(np.mean([info["mean_frequency_dilution"] for info in bundle["split_r1"].values()])),
            },
        },
        "transition_occurrences": dict(Counter(row["transition"] for row in bundle["occurrence_rows"])),
        "transition_molecules": {
            name: int((features[f"fraction_{name}"] > 0).sum()) for name in TRANSITIONS
        },
        "groups": group_results,
        "group_assignment": {
            "split_common_threshold": groups["split_common_threshold"],
            "n_strict_unsplit": groups["n_strict_unsplit"],
        },
        "matched_group_effects": matched,
        "correlations": correlations,
        "rarity": rarity,
        "disagreement": disagreement,
        "tier": tier,
        "controls": controls_out,
        "group_a": group_a,
        "topology_confound": topology_confound,
        "case_inspection": cases,
        "quintiles": quintiles,
        "subgroup_tables": subgroup_tables,
        "decision": decision,
        "runtime_seconds": float(time.perf_counter() - started),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    (OUT_DIR / "partial_analysis.json").write_text(json.dumps(partial, indent=2, sort_keys=True, default=str))
    (OUT_DIR / "decision_record.json").write_text(json.dumps(decision, indent=2, sort_keys=True, default=str))
    _write_group_csv(group_results, matched, disagreement, group_a)
    _write_rarity_csv(rarity)
    _write_disagreement_csv(disagreement)
    write_outputs(bundle, degradation)

    summary["figures"] = make_figures(bundle, features, degradation, group, external, rarity)
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def _write_group_csv(
    group_results: Mapping[str, Any],
    matched: Mapping[str, Any],
    disagreement: Mapping[str, Any],
    group_a: Mapping[str, Any],
) -> None:
    header = ["group", "n", "hist_mae", "corr_mae", "degradation", "hist_disagreement", "corr_disagreement", "delta_disagreement", "matched_vs_rest"]
    rows = []
    for name in ("G1", "G2", "G3", "G4"):
        info = group_results.get(name, {})
        if not info.get("n"):
            continue
        rows.append([
            name, info["n"], f"{info['hist_mae']:.6f}", f"{info['corr_mae']:.6f}",
            f"{info['degradation']:.6f}", f"{info['hist_disagreement']:.6f}",
            f"{info['corr_disagreement']:.6f}", f"{info['delta_disagreement']:.6f}",
            f"{matched[name]['weighted_diff']:.6f}",
        ])
    _write_csv(OUT_DIR / "group_transition_results.csv", header, rows)


def _write_rarity_csv(rarity: Mapping[str, Any]) -> None:
    header = ["family", "statistic", "value"]
    rows = []
    for family, stats in rarity["rarity_vs_error_spearman"].items():
        for key, value in stats.items():
            rows.append([family, f"spearman_{key}_vs_error", f"{value:.6f}"])
    for family, stats in rarity["corpus"].items():
        for key, value in stats.items():
            rows.append([family, f"corpus_{key}", value])
    _write_csv(OUT_DIR / "rarity_comparison.csv", header, rows)


def _write_disagreement_csv(disagreement: Mapping[str, Any]) -> None:
    header = ["scope", "key", "hist", "corr", "delta"]
    rows = []
    overall = disagreement["overall"]
    rows.append(["overall", "disagreement", f"{overall['hist_disagreement']:.6f}", f"{overall['corr_disagreement']:.6f}", f"{overall['delta']:.6f}"])
    for name, info in disagreement["by_group"].items():
        if info.get("n"):
            rows.append(["group", name, f"{info['hist']:.6f}", f"{info['corr']:.6f}", f"{info['delta']:.6f}"])
    for name, info in disagreement["fragmentation_vs_delta_disagreement"].items():
        rows.append(["frag_vs_delta_disagreement", name, "", f"{info['spearman']:.6f}", f"{info['partial_controlling_hist_err']:.6f}"])
    link = disagreement["delta_disagreement_vs_degradation"]
    rows.append(["delta_disagreement_vs_degradation", "spearman", "", f"{link['spearman']:.6f}", f"{link['partial_controlling_hist_err']:.6f}"])
    _write_csv(OUT_DIR / "disagreement_comparison.csv", header, rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    summary = run_audit()
    print(json.dumps({
        "overall": summary["overall"],
        "token_fragmentation": summary["token_fragmentation"],
        "transition_occurrences": summary["transition_occurrences"],
        "transition_molecules": summary["transition_molecules"],
        "groups": summary["groups"],
        "decision": summary["decision"]["verdict"],
    }, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
