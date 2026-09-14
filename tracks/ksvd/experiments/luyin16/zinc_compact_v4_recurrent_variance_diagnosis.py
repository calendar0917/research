"""Training-variance diagnosis for the canonical compact-v4 T=2 recurrent Q16.

Purpose
-------
The frozen T=2 recurrent Q16 model (82,115 params) shows a large 2-seed
ensemble gain (+0.0097 historically) while extending the horizon and decaying
the LR do not improve single-model validation MAE.  Before touching the
architecture this module answers whether the single-model estimator itself is
*training-variance limited* and whether that variance is worth regularising.

Phases
------
* A  freeze the currently bit-reproducible canonical code path / config.
* B  fixed equal-weight top-5 checkpoint soup (the repo's locked Top-5 rule,
     reused verbatim: five lowest-validation checkpoints, ties -> earliest
     epoch, arithmetic weight average) for each canonical seed.
* C  train the canonical seed2 with the *identical* protocol and evaluate
     single models, pair ensembles and the 3-seed ensemble.
* D  variance interpretation (disagreement / residual correlation / scaling).
* E  minimal single-model regularisation: only ``weight_decay 1e-5 -> 1e-4``,
     canonical seed0 first, seed1 only if the seed0 improvement is >= +0.002.

Hard constraints
----------------
* knowledge distillation / teacher-student: never.
* Q24/Q32, new architecture, T=3, pair-to-pair, LR/scheduler/dropout/batch
  sweeps, ensemble-weight search: never.
* official **test** is never loaded.
* ensemble / soup are diagnostics only, not the final model proposal.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_variance_diagnosis <stage>

Stages: ``freeze train soup variance wd_decision report``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_recurrent_pair_centre as rec
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen layout / constants
# ---------------------------------------------------------------------------

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/compact_v4_recurrent_variance_diagnosis"
RUNS_DIR = RESULTS_DIR / "runs"
CURVE_DIR = RESULTS_DIR / "curves"
STATE_DIR = RESULTS_DIR / "states"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
SOUP_DIR = RESULTS_DIR / "soup_states"

PROTOCOL_VERSION = "compact_v4_recurrent_variance_diagnosis_v1"

EXPECTED_TOTAL = 82115
EXPECTED_HEAD = 4135

CANONICAL_SEEDS = (0, 1, 2)
CANONICAL_TAG = "canonical"
WD_TAG = "wd1e-4"
CANONICAL_WEIGHT_DECAY = 1.0e-5
WD_WEIGHT_DECAY = 1.0e-4

# Top-5 rule frozen by the repo's top5_checkpoint_aggregation_stabilization
# audit.  The canonical training loop early-stops on the official-valid MAE, so
# "lowest selection MAE" is the per-epoch official-valid MAE.
SOUP_K = 5
WD_TRIGGER = 0.002  # seed0 improvement over canonical required to buy seed1

# Source files whose hash defines the frozen canonical implementation.
CANONICAL_SOURCES = (
    "tracks/ksvd/experiments/luyin16/zinc_compact_v4_recurrent_pair_centre.py",
    "tracks/ksvd/experiments/luyin16/zinc_compact_v4_recurrent_pair_centre_capacity.py",
    "tracks/ksvd/experiments/luyin16/zinc_compact_v4_smallhead_e2e.py",
    "tracks/ksvd/experiments/luyin16/zinc_patch_path_pooling.py",
    "tracks/ksvd/experiments/luyin16/zinc_compact_v4_training_sufficiency.py",
)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()
    except OSError:
        return "unknown"


def _run_path(tag: str, seed: int) -> Path:
    return RUNS_DIR / f"{tag}_seed{seed}.json"


def _load_run(tag: str, seed: int) -> dict[str, Any]:
    return _read_json(_run_path(tag, seed))


# ---------------------------------------------------------------------------
# Phase A -- freeze canonical reference
# ---------------------------------------------------------------------------


def freeze() -> dict[str, Any]:
    sources = {
        rel: _sha256_file(REPO_ROOT / rel) for rel in CANONICAL_SOURCES
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_status_short": _git("status", "--short").splitlines(),
        "canonical_source_sha256": sources,
        "training_config": dict(shead.OPTIMIZED_PROTOCOL),
        "seed_handling": (
            "build_recurrent(seed) -> shead.build_baseline(seed) then "
            "shead._seed_everything(seed); head re-initialised under "
            "torch.manual_seed(head_seed=0); train shuffle seed = "
            "seed + train_shuffle_seed_offset; eval order not shuffled."
        ),
        "threads": {
            "module_main_set_num_threads": 4,
            "torch_threads": int(torch.get_num_threads()),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "architecture": {
            "name": "compact-v4-smallhead T=2 weight-tied recurrent pair-centre",
            "recurrence_rounds": rec.RECURRENCE_ROUNDS,
            "q_dim": rec.Q_DIM,
            "R_dim": rec.R_DIM,
            "total_params": EXPECTED_TOTAL,
            "head_params": EXPECTED_HEAD,
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "canonical_reference.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase C (and E) -- canonical training
# ---------------------------------------------------------------------------


def run_training(
    tag: str,
    seed: int,
    *,
    weight_decay: float | None = None,
    snapshots: bool = False,
    data_seed: int | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train_data, valid_data, _ = rec.load_encoded()
    override: dict[str, Any] = dict(protocol_override or {})
    if weight_decay is not None:
        override["weight_decay"] = float(weight_decay)
    snap_dir = SNAPSHOT_DIR / f"{tag}_seed{seed}" if snapshots else None
    summary = rec.train(
        rec.build_recurrent,
        train_data,
        valid_data,
        seed=int(seed),
        tag=str(tag),
        protocol_override=override or None,
        snapshot_dir=snap_dir,
        data_seed=data_seed,
    )
    # ``rec.train`` redirects the selection state into the recurrent results
    # dir; copy it next to this diagnosis' own outputs so the stages are
    # self-contained.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(str(summary["state_path"]))
    dst = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    dst.write_bytes(src.read_bytes())
    summary["diagnosis_state_path"] = str(dst)
    _write_json(_run_path(tag, seed), summary)
    return summary


# ---------------------------------------------------------------------------
# prediction / evaluation helpers
# ---------------------------------------------------------------------------


def _valid_data() -> list[Any]:
    _train, valid, _audit = rec.load_encoded()
    return list(valid)


def _eval_loader(valid_data: Sequence[Any]):
    return zpp._make_loader(
        list(valid_data), int(shead.OPTIMIZED_PROTOCOL["batch_size"]), False, 0
    )


def _predict_state(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor], loader
) -> tuple[np.ndarray, np.ndarray]:
    model.load_state_dict({k: v for k, v in state_dict.items()}, strict=True)
    model.eval()
    targets: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            targets.append(batch.y.view(-1).cpu().numpy())
            preds.append(model(batch).view(-1).cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float64),
        np.concatenate(preds).astype(np.float64),
    )


def _mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


def _load_selection_state(tag: str, seed: int) -> dict[str, torch.Tensor]:
    path = STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def selection_predictions(
    tag: str, seed: int, loader, model: torch.nn.Module
) -> dict[str, Any]:
    state = _load_selection_state(tag, seed)
    targets, preds = _predict_state(model, state, loader)
    summary = _load_run(tag, seed)
    return {
        "tag": tag,
        "seed": int(seed),
        "best_valid_mae_reported": float(summary["best_valid_mae"]),
        "best_valid_mae_recomputed": _mae(targets, preds),
        "best_epoch": int(summary["best_epoch"]),
        "targets": targets,
        "predictions": preds,
    }


# ---------------------------------------------------------------------------
# Phase B -- fixed top-5 checkpoint soup
# ---------------------------------------------------------------------------


def _top5_epochs(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = list(summary.get("snapshot_manifest") or [])
    if len(manifest) < SOUP_K:
        raise RuntimeError(
            f"need >= {SOUP_K} snapshots for a strict top-5 soup, "
            f"found {len(manifest)}"
        )
    ranked = sorted(
        manifest, key=lambda row: (float(row["valid_mae"]), int(row["epoch"]))
    )
    return ranked[:SOUP_K]


def build_soup_state(epochs: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    states = [
        torch.load(Path(str(row["path"])), map_location="cpu", weights_only=True)
        for row in epochs
    ]
    keys = list(states[0].keys())
    soup: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([state[key].float() for state in states], dim=0)
        soup[key] = stacked.mean(dim=0).to(states[0][key].dtype)
    return soup


def soup(tag: str, seed: int) -> dict[str, Any]:
    summary = _load_run(tag, seed)
    top = _top5_epochs(summary)
    soup_state = build_soup_state(top)
    SOUP_DIR.mkdir(parents=True, exist_ok=True)
    soup_path = SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    torch.save(soup_state, soup_path)

    valid_data = _valid_data()
    loader = _eval_loader(valid_data)
    model = rec.build_recurrent(seed)

    best_state = _load_selection_state(tag, seed)
    targets, best_preds = _predict_state(model, best_state, loader)
    _targets2, soup_preds = _predict_state(model, soup_state, loader)
    best_mae = _mae(targets, best_preds)
    soup_mae = _mae(targets, soup_preds)

    payload = {
        "tag": tag,
        "seed": int(seed),
        "protocol_version": PROTOCOL_VERSION,
        "soup_rule": {
            "K": SOUP_K,
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "arithmetic mean",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "top5_epochs": [int(row["epoch"]) for row in top],
        "top5_valid_mae": [float(row["valid_mae"]) for row in top],
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_valid_mae": best_mae,
        "top5_soup_valid_mae": soup_mae,
        "soup_improvement_over_best": best_mae - soup_mae,
        "soup_state_path": str(soup_path),
        "soup_sha256": _sha256_file(soup_path),
        "parameters": int(sum(p.numel() for p in soup_state.values())),
        "official_test_loaded": False,
        "best_predictions": best_preds.tolist(),
        "soup_predictions": soup_preds.tolist(),
    }
    _write_json(RESULTS_DIR / f"soup_{tag}_seed{seed}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase D -- variance / ensemble scaling
# ---------------------------------------------------------------------------


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def variance() -> dict[str, Any]:
    valid_data = _valid_data()
    loader = _eval_loader(valid_data)
    model = rec.build_recurrent(0)

    singles: dict[int, dict[str, Any]] = {}
    for seed in CANONICAL_SEEDS:
        singles[seed] = selection_predictions(CANONICAL_TAG, seed, loader, model)

    targets = singles[CANONICAL_SEEDS[0]]["targets"]
    for seed in CANONICAL_SEEDS:
        assert np.array_equal(
            targets, singles[seed]["targets"]
        ), "validation order mismatch across seeds"

    per_seed_mae = {
        int(seed): _mae(singles[seed]["targets"], singles[seed]["predictions"])
        for seed in CANONICAL_SEEDS
    }
    single_values = np.array([per_seed_mae[int(s)] for s in CANONICAL_SEEDS])

    pair_results: dict[str, Any] = {}
    for i, a in enumerate(CANONICAL_SEEDS):
        for b in CANONICAL_SEEDS[i + 1 :]:
            pred = 0.5 * (singles[a]["predictions"] + singles[b]["predictions"])
            pair_results[f"{a}{b}"] = {
                "seeds": [int(a), int(b)],
                "valid_mae": _mae(targets, pred),
            }
    pair_maes = np.array([v["valid_mae"] for v in pair_results.values()])

    triple_pred = np.mean(
        np.stack([singles[s]["predictions"] for s in CANONICAL_SEEDS], axis=0), axis=0
    )
    triple_mae = _mae(targets, triple_pred)

    # disagreement / correlation
    disagreements = []
    residual_corrs = []
    prediction_corrs = []
    for i, a in enumerate(CANONICAL_SEEDS):
        for b in CANONICAL_SEEDS[i + 1 :]:
            disagreements.append(
                float(
                    np.mean(
                        np.abs(singles[a]["predictions"] - singles[b]["predictions"])
                    )
                )
            )
            ra = singles[a]["targets"] - singles[a]["predictions"]
            rb = singles[b]["targets"] - singles[b]["predictions"]
            residual_corrs.append(_corr(ra, rb))
            prediction_corrs.append(
                _corr(singles[a]["predictions"], singles[b]["predictions"])
            )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seeds": [int(s) for s in CANONICAL_SEEDS],
        "single_model_valid_mae": {
            str(s): per_seed_mae[int(s)] for s in CANONICAL_SEEDS
        },
        "single_model_mean": float(single_values.mean()),
        "single_model_std": float(single_values.std(ddof=1)),
        "single_model_range": [float(single_values.min()), float(single_values.max())],
        "pair_ensembles": pair_results,
        "pair_ensemble_mean": float(pair_maes.mean()),
        "pair_ensemble_range": [
            float(pair_maes.min()),
            float(pair_maes.max()),
        ],
        "pair_ensemble_spread": float(pair_maes.max() - pair_maes.min()),
        "three_seed_ensemble_valid_mae": triple_mae,
        "ensemble_gain_3seed_over_mean_single": float(
            single_values.mean() - triple_mae
        ),
        "incremental_3seed_over_mean_pair": float(pair_maes.mean() - triple_mae),
        "mean_abs_prediction_disagreement": float(np.mean(disagreements)),
        "pairwise_prediction_correlation": prediction_corrs,
        "pairwise_residual_correlation": residual_corrs,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "variance_diagnosis.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase B (cross-seed soup diagnostic) & Phase E decision
# ---------------------------------------------------------------------------


def soup_ceiling() -> dict[str, Any]:
    """0.5 * (seed0_soup_pred + seed1_soup_pred): variance-ceiling diagnostic."""
    valid_data = _valid_data()
    loader = _eval_loader(valid_data)
    model = rec.build_recurrent(0)
    targets, _ = _predict_state(model, _load_selection_state(CANONICAL_TAG, 0), loader)
    preds = []
    for seed in (0, 1):
        state = torch.load(
            SOUP_DIR / f"{CANONICAL_TAG}_seed{seed}_top5_soup.pt",
            map_location="cpu",
            weights_only=True,
        )
        _, p = _predict_state(model, state, loader)
        preds.append(p)
    ceiling_pred = 0.5 * (preds[0] + preds[1])
    payload = {
        "soup_ensemble_valid_mae": _mae(targets, ceiling_pred),
        "role": "variance ceiling diagnostic only; not a final model",
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "soup_ensemble_ceiling.json", payload)
    return payload


def wd_decision() -> dict[str, Any]:
    canonical = _load_run(CANONICAL_TAG, 0)
    if not _run_path(WD_TAG, 0).exists():
        raise RuntimeError("run wd1e-4 seed0 first")
    wd0 = _load_run(WD_TAG, 0)
    improvement = float(canonical["best_valid_mae"]) - float(wd0["best_valid_mae"])
    seed1 = None
    if _run_path(WD_TAG, 1).exists():
        seed1 = _load_run(WD_TAG, 1)
    payload = {
        "canonical_seed0_valid_mae": float(canonical["best_valid_mae"]),
        "canonical_seed0_best_epoch": int(canonical["best_epoch"]),
        "canonical_seed0_train_at_best": float(canonical["train_loss_at_best"]),
        "wd1e-4_seed0_valid_mae": float(wd0["best_valid_mae"]),
        "wd1e-4_seed0_best_epoch": int(wd0["best_epoch"]),
        "wd1e-4_seed0_train_at_best": float(wd0["train_loss_at_best"]),
        "seed0_improvement": improvement,
        "wd_trigger": WD_TRIGGER,
        "buy_seed1": bool(improvement >= WD_TRIGGER),
        "seed1_result": (
            None
            if seed1 is None
            else {
                "canonical_seed1_valid_mae": float(
                    _load_run(CANONICAL_TAG, 1)["best_valid_mae"]
                ),
                "wd1e-4_seed1_valid_mae": float(seed1["best_valid_mae"]),
                "improvement": float(
                    _load_run(CANONICAL_TAG, 1)["best_valid_mae"]
                    - seed1["best_valid_mae"]
                ),
            }
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "wd_decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "canonical_reference": (
            _read_json(RESULTS_DIR / "canonical_reference.json")
            if (RESULTS_DIR / "canonical_reference.json").exists()
            else None
        ),
        "soups": {},
        "variance": (
            _read_json(RESULTS_DIR / "variance_diagnosis.json")
            if (RESULTS_DIR / "variance_diagnosis.json").exists()
            else None
        ),
        "soup_ceiling": (
            _read_json(RESULTS_DIR / "soup_ensemble_ceiling.json")
            if (RESULTS_DIR / "soup_ensemble_ceiling.json").exists()
            else None
        ),
        "wd_decision": (
            _read_json(RESULTS_DIR / "wd_decision.json")
            if (RESULTS_DIR / "wd_decision.json").exists()
            else None
        ),
        "official_test_loaded": False,
    }
    for seed in (0, 1):
        path = RESULTS_DIR / f"soup_{CANONICAL_TAG}_seed{seed}.json"
        if path.exists():
            payload["soups"][str(seed)] = _read_json(path)
    _write_json(RESULTS_DIR / "final_report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["freeze", "train", "soup", "soup_ceiling", "variance", "wd_decision", "report"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=CANONICAL_TAG)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--data-seed", type=int, default=None)
    args = parser.parse_args(argv)

    _configure_threads()
    if args.stage == "freeze":
        print(json.dumps(freeze(), indent=2, default=str), flush=True)
    if args.stage == "train":
        print(
            json.dumps(
                run_training(
                    args.tag,
                    args.seed,
                    weight_decay=args.weight_decay,
                    snapshots=bool(args.snapshots),
                    data_seed=args.data_seed,
                ),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "soup":
        print(json.dumps(soup(args.tag, args.seed), indent=2, default=str), flush=True)
    if args.stage == "soup_ceiling":
        print(json.dumps(soup_ceiling(), indent=2, default=str), flush=True)
    if args.stage == "variance":
        print(json.dumps(variance(), indent=2, default=str), flush=True)
    if args.stage == "wd_decision":
        print(json.dumps(wd_decision(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


def _configure_threads() -> None:
    torch.set_num_threads(int(shead.TORCH_THREADS))


if __name__ == "__main__":
    raise SystemExit(main())
