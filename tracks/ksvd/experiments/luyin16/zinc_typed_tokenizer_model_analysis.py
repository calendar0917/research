"""Model-side analysis of the corrected typed patch tokenizer (validation only).

Reads the validation-selected checkpoints' saved validation predictions and
targets for the historical and corrected tokenizer runs, plus the tokenizer
audit artifacts, and produces Tables D-G and Figures 3-6.  Never loads test.

Run: ``uv run python -m tracks.ksvd.experiments.luyin16.zinc_typed_tokenizer_model_analysis``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = REPO_ROOT / "tracks/ksvd/results/typed_patch_tokenizer_correctness"
OUT_DIR = AUDIT_DIR / "model_analysis"
RUNS_ROOT = REPO_ROOT / "tracks/ksvd/runs"

# This stage's runs (validation-selected checkpoints; scratch mode = valid-only).
RUNS: dict[str, dict[int, str]] = {
    "v2_historical": {
        0: "20260909-193141-6962889e",
        1: "20260909-195530-666da952",
        2: "20260909-201121-e3e57f05",
        3: "20260909-202722-a1dd7a00",
    },
    "v2_corrected": {
        0: "20260910-111031-2b8cce42",
        1: "20260910-114152-3295454c",
        2: "20260910-114713-132cd654",
        3: "20260910-115243-a96622ae",
    },
    "v4_historical": {
        0: "20260909-194445-182c7021",
        1: "20260909-200320-34b347bf",
        2: "20260909-201918-45fbe48d",
        3: "20260909-203516-04e62a28",
    },
    "v4_corrected": {
        0: "20260910-111954-64845bd9",
        1: "20260910-112530-c02ac7f0",
        2: "20260910-113058-a40ee3e1",
        # seed 3: the first (20260910-113625-e5d57ba6) diverged at epoch 55-60
        # (a nondeterministic late-training trajectory).  The reproducible
        # value is the terminal run / scratch rerun below (0.174880).
        3: "20260910-123241-6067c9ac",
    },
}


def _run_dir(run_id: str) -> Path:
    matches = sorted(RUNS_ROOT.glob(f"*/*/*/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"run {run_id}: {matches}")
    return matches[0]


def _load_run(run_id: str) -> dict[str, Any]:
    result = json.loads((_run_dir(run_id) / "artifacts/legacy_full_result.json").read_text())
    valid = result["evaluation"]["valid"]
    return {
        "predictions": np.asarray(valid["predictions"], dtype=np.float64),
        "targets": np.asarray(valid["targets"], dtype=np.float64),
        "mae": float(valid["mae"]),
        "selected_epoch": int(valid["selected_epoch"]),
        "parameters": int(result["evaluation"]["parameters"]),
        "embedding": result["evaluation"].get("parameter_audit", {})
        .get("embedding", {})
        if isinstance(result["evaluation"].get("parameter_audit"), dict)
        else {},
        "typed_vocab": result["vocabulary_audit"]["valid"]["typed_vocabulary"]["fit"][
            "vocabulary_size"
        ],
    }


def _subgroup_mae(pred: np.ndarray, target: np.ndarray, indices) -> float:
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[idx] - target[idx])))


def _subgroup_mae_over_seeds(preds: np.ndarray, target: np.ndarray, indices) -> float:
    """Mean over seeds of the per-seed subgroup MAE (matches the overall metric)."""
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return float("nan")
    return float(np.mean([np.mean(np.abs(preds[s][idx] - target[idx])) for s in range(preds.shape[0])]))


def _subgroup_disagreement(predictions: np.ndarray, indices) -> float:
    """Mean over molecules of the cross-seed prediction std."""
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return float("nan")
    return float(np.mean(np.std(predictions[:, idx], axis=0)))


def _paper_stats(historical: Mapping[int, float], corrected: Mapping[int, float]) -> dict[str, Any]:
    seeds = sorted(set(historical) & set(corrected))
    deltas = np.asarray([historical[s] - corrected[s] for s in seeds], dtype=np.float64)
    return {
        "seeds": seeds,
        "historical_mean": float(np.mean([historical[s] for s in seeds])),
        "corrected_mean": float(np.mean([corrected[s] for s in seeds])),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "median_delta": float(np.median(deltas)),
        "positive_seeds": int(np.sum(deltas > 0)),
        "n_seeds": len(seeds),
        "per_seed": {str(s): float(deltas[i]) for i, s in enumerate(seeds)},
    }


def _rarity_profiles() -> dict[str, Any]:
    """Per-validation-molecule rare/OOV patch fractions (train-only statistics).

    The saved npz uses separate integer-id spaces per split, so the keys are
    aligned through ``keys.json`` (id -> hex key) before counting.
    """
    data = np.load(AUDIT_DIR / "occurrences.npz")
    keys = json.loads((AUDIT_DIR / "keys.json").read_text())
    valid_mol = data["valid_mol_ids"]
    n_valid = int(valid_mol.max()) + 1
    profiles: dict[str, Any] = {}
    for label, tag in (("historical", "hist_r2"), ("corrected", "corr_r2")):
        if f"{tag}" in keys:
            # shared train+valid id space (current format)
            key_map = keys[tag]
            train_keys = [key_map[str(int(i))] for i in data[f"train_{tag}_id"]]
            valid_keys = [key_map[str(int(i))] for i in data[f"valid_{tag}_id"]]
        else:
            # legacy per-split id space: align through the hex keys
            train_map = keys[f"train_{tag}"]
            valid_map = keys[f"valid_{tag}"]
            train_keys = [train_map[str(int(i))] for i in data[f"train_{tag}_id"]]
            valid_keys = [valid_map[str(int(i))] for i in data[f"valid_{tag}_id"]]
        train_key_set = set(train_keys)
        from collections import Counter

        counts = Counter(train_keys)
        rare_frac = np.zeros(n_valid, dtype=np.float64)
        oov_frac = np.zeros(n_valid, dtype=np.float64)
        for mol in range(n_valid):
            local = np.flatnonzero(valid_mol == mol)
            if local.size == 0:
                continue
            rare_frac[mol] = float(
                np.mean([0 < counts.get(valid_keys[i], 0) <= 5 for i in local])
            )
            oov_frac[mol] = float(
                np.mean([valid_keys[i] not in train_key_set for i in local])
            )
        profiles[label] = {"rare_frac": rare_frac, "oov_frac": oov_frac}
    return profiles


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def run_analysis() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alias = json.loads((AUDIT_DIR / "alias_molecules.json").read_text())
    audit = json.loads((AUDIT_DIR / "audit.json").read_text())

    loaded = {name: {seed: _load_run(rid) for seed, rid in mapping.items()} for name, mapping in RUNS.items()}

    # Table D -- seed0
    table_d = {}
    for family in ("v2", "v4"):
        hist = loaded[f"{family}_historical"][0]
        corr = loaded[f"{family}_corrected"][0]
        table_d[family] = {
            "historical": {"params": hist["parameters"], "valid_mae": hist["mae"], "epoch": hist["selected_epoch"]},
            "corrected": {"params": corr["parameters"], "valid_mae": corr["mae"], "epoch": corr["selected_epoch"]},
            "delta_valid_mae": hist["mae"] - corr["mae"],
            "delta_params": corr["parameters"] - hist["parameters"],
            "delta_params_pct": 100.0 * (corr["parameters"] - hist["parameters"]) / hist["parameters"],
        }

    # Table E -- multi-seed v4
    hist_v4 = {s: loaded["v4_historical"][s]["mae"] for s in range(4)}
    corr_v4 = {s: loaded["v4_corrected"][s]["mae"] for s in range(4)}
    hist_v2 = {s: loaded["v2_historical"][s]["mae"] for s in range(4)}
    corr_v2 = {s: loaded["v2_corrected"][s]["mae"] for s in range(4)}
    table_e = {
        "v4": _paper_stats(hist_v4, corr_v4),
        "v2": _paper_stats(hist_v2, corr_v2),
    }

    # Subgroups: pre-registered (degenerate) + burden quartiles (audit-time).
    burden = np.asarray(alias["alias_burden_ratio"], dtype=np.float64)
    order = np.argsort(burden)
    quartiles = np.array_split(order, 4)
    quartile_groups = {f"q{i+1}": q.tolist() for i, q in enumerate(quartiles)}
    pre_registered = {
        "unaffected": alias["unaffected"],
        "alias_affected": alias["alias_affected"],
        "root_collision": alias["root_collision"],
        "nonroot_collision": alias["nonroot_collision"],
        "high_alias_burden": alias["high_alias_burden"],
    }

    hist_preds = np.stack([loaded["v4_historical"][s]["predictions"] for s in range(4)])
    corr_preds = np.stack([loaded["v4_corrected"][s]["predictions"] for s in range(4)])
    targets = loaded["v4_historical"][0]["targets"]

    def subgroup_table(groups: Mapping[str, list[int]]) -> dict[str, Any]:
        out = {}
        for name, indices in groups.items():
            idx = np.asarray(indices, dtype=int)
            hist_mae = _subgroup_mae_over_seeds(hist_preds, targets, idx)
            corr_mae = _subgroup_mae_over_seeds(corr_preds, targets, idx)
            out[name] = {
                "n": int(idx.size),
                "historical_mae": hist_mae,
                "corrected_mae": corr_mae,
                "delta": hist_mae - corr_mae,
            }
        return out

    table_f = {
        "pre_registered_v4": subgroup_table(pre_registered),
        "alias_burden_quartiles_v4": subgroup_table(quartile_groups),
    }
    # per-observation (single-seed mean) subgroup deltas, averaged over seed pairs
    seed_deltas = {}
    for name, indices in quartile_groups.items():
        idx = np.asarray(indices, dtype=int)
        per_seed = []
        for s in range(4):
            h = _subgroup_mae(hist_preds[s], targets, idx)
            c = _subgroup_mae(corr_preds[s], targets, idx)
            per_seed.append(h - c)
        seed_deltas[name] = [float(x) for x in per_seed]
    table_f["alias_burden_quartiles_v4_per_seed_delta"] = seed_deltas

    # Table G -- disagreement
    table_g = {}
    for name, indices in {**pre_registered, **quartile_groups}.items():
        table_g[name] = {
            "n": len(indices),
            "historical_disagreement": _subgroup_disagreement(hist_preds, indices),
            "corrected_disagreement": _subgroup_disagreement(corr_preds, indices),
        }
        table_g[name]["delta"] = (
            table_g[name]["historical_disagreement"] - table_g[name]["corrected_disagreement"]
        )

    # rarity vs |residual|
    profiles = _rarity_profiles()
    resid_hist = np.mean([np.abs(hist_preds[s] - targets) for s in range(4)], axis=0)
    resid_corr = np.mean([np.abs(corr_preds[s] - targets) for s in range(4)], axis=0)
    rarity_corr = {
        "historical_spearman_rare_frac": _spearman(profiles["historical"]["rare_frac"], resid_hist),
        "historical_spearman_oov_frac": _spearman(profiles["historical"]["oov_frac"], resid_hist),
        "corrected_spearman_rare_frac": _spearman(profiles["corrected"]["rare_frac"], resid_corr),
        "corrected_spearman_oov_frac": _spearman(profiles["corrected"]["oov_frac"], resid_corr),
    }

    report = {
        "table_d_seed0": table_d,
        "table_e_multiseed": table_e,
        "table_f_subgroups": table_f,
        "table_g_disagreement": table_g,
        "rarity_residual": rarity_corr,
        "vocabulary": audit["vocabulary"],
        "historical_bucket_audit": audit["historical_bucket_audit"],
        "v4_seed0_predictions": loaded["v4_historical"][0]["predictions"].tolist(),
        "v4_seed0_targets": targets.tolist(),
    }
    (OUT_DIR / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _figures(report, profiles, hist_preds, corr_preds, targets, burden)
    return report


def _figures(report, profiles, hist_preds, corr_preds, targets, burden) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 3: alias burden vs historical validation MAE (seed-averaged)
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_err = np.mean([np.abs(hist_preds[s] - targets) for s in range(hist_preds.shape[0])], axis=0)
    order = np.argsort(burden)
    ax.scatter(burden[order], hist_err[order], s=6, alpha=0.4)
    ax.set_xlabel("per-molecule historical alias burden ratio")
    ax.set_ylabel("historical |residual| (seed-mean)")
    ax.set_title("Figure 3 - alias burden vs historical validation error")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_alias_burden_vs_error.png", dpi=130)
    plt.close(fig)

    # Figure 4: historical vs corrected MAE by alias burden quartile
    groups = report["table_f_subgroups"]["alias_burden_quartiles_v4"]
    names = list(groups)
    hist = [groups[n]["historical_mae"] for n in names]
    corr = [groups[n]["corrected_mae"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, hist, 0.4, label="historical")
    ax.bar(x + 0.2, corr, 0.4, label="corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("validation MAE")
    ax.set_title("Figure 4 - MAE by alias-burden quartile (v4, seed-mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_subgroup_mae.png", dpi=130)
    plt.close(fig)

    # Figure 5: disagreement by alias burden quartile
    dis = report["table_g_disagreement"]
    hist_d = [dis[n]["historical_disagreement"] for n in names]
    corr_d = [dis[n]["corrected_disagreement"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, hist_d, 0.4, label="historical")
    ax.bar(x + 0.2, corr_d, 0.4, label="corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mean cross-seed prediction std")
    ax.set_title("Figure 5 - cross-seed disagreement by alias-burden quartile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_disagreement.png", dpi=130)
    plt.close(fig)

    # Figure 6: rarity vs |residual|, historical vs corrected
    hist_rare = profiles["historical"]["rare_frac"]
    corr_rare = profiles["corrected"]["rare_frac"]
    hist_err = np.mean([np.abs(hist_preds[s] - targets) for s in range(hist_preds.shape[0])], axis=0)
    corr_err = np.mean([np.abs(corr_preds[s] - targets) for s in range(corr_preds.shape[0])], axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 6)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for label, rare, err in (
        ("historical", hist_rare, hist_err),
        ("corrected", corr_rare, corr_err),
    ):
        idx = np.digitize(rare, bins) - 1
        means = [err[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(centers))]
        ax.plot(centers, means, marker="o", label=label)
    ax.set_xlabel("fraction of rare (<=5) radius-2 patches")
    ax.set_ylabel("mean |residual|")
    ax.set_title("Figure 6 - rarity vs |residual| (v4, seed-mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_rarity_residual.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import json as _json

    report = run_analysis()
    print(_json.dumps({
        "table_d_seed0": report["table_d_seed0"],
        "table_e_multiseed": report["table_e_multiseed"],
        "table_g_disagreement": report["table_g_disagreement"],
        "rarity_residual": report["rarity_residual"],
    }, indent=2))
