"""Seed-variance source decomposition for the canonical compact-v4 T=2 recurrent Q16.

Question
--------
The canonical single-model valid MAE is strongly seed dependent
(seed0 0.140609 / seed1 0.133440 / seed2 0.142722; 3-seed ensemble 0.125962).
Does that variance come mainly from **parameter initialization** or from
**mini-batch / data-order stochasticity**?

Design
------
* Phase A: fixed Top-5 checkpoint soup for canonical seed2 (same locked rule as
  seed0/seed1), then a cross-seed soup-gain summary.
* Phase B: a 2x2 initialization x shuffle factorial.  The canonical training
  loop is minimally refactored so that the initialization seed and the
  data-order seed can be set independently (default stays coupled):
  ``A = init0+shuffle0`` (= canonical seed0), ``B = init0+shuffle1``,
  ``C = init1+shuffle0``, ``D = init1+shuffle1`` (= canonical seed1).
  The post-build global torch RNG (shared dropout/forward stream) is identical
  across init seeds, so the two factors are cleanly separated.
* Phase C: initialization / shuffle / interaction effect decomposition.
* Phase D: pre-registered interpretation gate.

Constraints
-----------
No official test, no distillation, no new architecture / T=3 / Q24 / Q32, no
WD / LR / scheduler / dropout sweep, no EMA/SWA, no loss sweep, no new soup
rule, no ensemble-weight search.  All new training runs strictly sequential.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_compact_v4_recurrent_seed_variance_source <stage>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)

RESULTS_DIR = vd.RESULTS_DIR.parent / "compact_v4_recurrent_seed_variance_source"
PROTOCOL_VERSION = "compact_v4_recurrent_seed_variance_source_v1"

# Minimum effect size (in valid MAE) for the interpretation gate.
EFFECT_GATE = 0.002
INTERACTION_GATE = 0.002
SANITY_EPOCHS = 61

# The canonical 4-thread training loop is NOT process-to-process reproducible:
# two independent sequential runs of the same (init, shuffle) cell are
# bit-identical for ~26 epochs and then diverge by ~5e-10 (a nondeterministic
# ``index_add``-family path), amplifying to ~0.003 valid MAE.  The 2x2 factorial
# is therefore run with ``torch.use_deterministic_algorithms(True)`` so each cell
# is a well-defined reproducible point.  That flag is concurrency-safe.
DETERMINISTIC = True

# cell -> (init_seed, shuffle_seed, deterministic tag, state run seed)
CELLS: dict[str, dict[str, Any]] = {
    "A": {"init_seed": 0, "shuffle_seed": 0, "tag": "detA", "run_seed": 0},
    "B": {"init_seed": 0, "shuffle_seed": 1, "tag": "detB", "run_seed": 0},
    "C": {"init_seed": 1, "shuffle_seed": 0, "tag": "detC", "run_seed": 1},
    "D": {"init_seed": 1, "shuffle_seed": 1, "tag": "detD", "run_seed": 1},
}

# canonical (non-deterministic, 4-thread) reference for A and D
CANONICAL_REFERENCE: dict[str, tuple[str, int]] = {
    "A": ("canonical", 0),
    "D": ("canonical", 1),
}

# canonical-mode cross cells, run sequentially (single noisy samples; the
# canonical 4-thread loop is not process-to-process reproducible)
CANONICAL_CELLS: dict[str, dict[str, Any]] = {
    "B": {"init_seed": 0, "shuffle_seed": 1, "tag": "canonB", "run_seed": 0},
    "C": {"init_seed": 1, "shuffle_seed": 0, "tag": "canonC", "run_seed": 1},
}

# full canonical 2x2: cell -> (tag, state run seed)
_CANONICAL_STATE: dict[str, tuple[str, int]] = {
    "A": ("canonical", 0),
    "B": ("canonB", 0),
    "C": ("canonC", 1),
    "D": ("canonical", 1),
}
CANONICAL_INIT_SHUFFLE: dict[str, tuple[int, int]] = {
    "A": (0, 0),
    "B": (0, 1),
    "C": (1, 0),
    "D": (1, 1),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_run(cell: str) -> dict[str, Any]:
    spec = CELLS[cell]
    return _read_json(vd._run_path(str(spec["tag"]), int(spec["run_seed"])))


def _cell_state_path(cell: str) -> Path:
    spec = CELLS[cell]
    return vd.STATE_DIR / f"{spec['tag']}_seed{spec['run_seed']}_selection_state.pt"


# ---------------------------------------------------------------------------
# Phase A -- seed2 soup + cross-seed summary
# ---------------------------------------------------------------------------


def soup_seed2() -> dict[str, Any]:
    return vd.soup("canonical", 2)


def soup_summary() -> dict[str, Any]:
    gains = {}
    details = {}
    for seed in (0, 1, 2):
        path = vd.RESULTS_DIR / f"soup_canonical_seed{seed}.json"
        payload = _read_json(path)
        gains[int(seed)] = float(payload["soup_improvement_over_best"])
        details[int(seed)] = {
            "best_checkpoint_valid_mae": float(payload["best_checkpoint_valid_mae"]),
            "top5_soup_valid_mae": float(payload["top5_soup_valid_mae"]),
            "top5_epochs": list(payload["top5_epochs"]),
            "soup_improvement": float(payload["soup_improvement_over_best"]),
        }
    values = np.array(list(gains.values()))
    out = {
        "protocol_version": PROTOCOL_VERSION,
        "rule": {
            "K": vd.SOUP_K,
            "ranking": "lowest official-valid MAE",
            "tie_rule": "earliest epoch",
            "weight_aggregation": "arithmetic mean",
            "no_k_search": True,
            "no_weight_search": True,
        },
        "per_seed": details,
        "gains": gains,
        "mean_gain": float(values.mean()),
        "std_gain": float(values.std()),
        "min_gain": float(values.min()),
        "max_gain": float(values.max()),
        "all_positive": bool((values > 0).all()),
        "reproducible_stabilizer": bool((values > 0).all() and float(values.min()) > 0.0),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "soup_summary.json", out)
    return out


# ---------------------------------------------------------------------------
# Phase B -- factorial cells
# ---------------------------------------------------------------------------


def run_canonical_cell(cell: str) -> dict[str, Any]:
    """Run a cross cell in the canonical non-deterministic 4-thread mode.

    Must be launched strictly sequentially: the canonical loop is not
    process-to-process reproducible and concurrent load perturbs it.
    """
    if cell not in CANONICAL_CELLS:
        raise ValueError(f"unknown canonical cell {cell!r}")
    spec = CANONICAL_CELLS[cell]
    __import__("torch").use_deterministic_algorithms(False)
    summary = vd.run_training(
        str(spec["tag"]),
        int(spec["init_seed"]),
        data_seed=int(spec["shuffle_seed"]),
        snapshots=False,
    )
    _write_json(RESULTS_DIR / f"cell_canonical_{cell}.json", summary)
    return summary


def run_cell(cell: str) -> dict[str, Any]:
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}")
    spec = CELLS[cell]
    summary = vd.run_training(
        str(spec["tag"]),
        int(spec["init_seed"]),
        data_seed=int(spec["shuffle_seed"]),
        snapshots=False,
    )
    _write_json(RESULTS_DIR / f"cell_{cell}.json", summary)
    return summary


# ---------------------------------------------------------------------------
# deterministic sanity (short duplicate)
# ---------------------------------------------------------------------------


def sanity_rep(
    rep: int,
    init_seed: int = 0,
    shuffle_seed: int = 1,
    epochs: int = SANITY_EPOCHS,
) -> dict[str, Any]:
    """One short repetition of a fixed (init, shuffle) cell, in its own process."""
    summary = vd.run_training(
        f"sanity_{init_seed}_{shuffle_seed}_rep{rep}",
        int(init_seed),
        data_seed=int(shuffle_seed),
        protocol_override={"max_epochs": int(epochs), "patience": int(epochs)},
    )
    return {
        "rep": int(rep),
        "init_seed": int(init_seed),
        "shuffle_seed": int(shuffle_seed),
        "epochs": int(epochs),
        "best_valid_mae": float(summary["best_valid_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "curve_path": str(summary["curve_path"]),
        "official_test_loaded": False,
    }


def sanity_compare(init_seed: int = 0, shuffle_seed: int = 1) -> dict[str, Any]:
    reps = []
    for rep in (1, 2):
        path = RESULTS_DIR / f"cell_sanity_{init_seed}_{shuffle_seed}_rep{rep}.json"
        if path.exists():
            reps.append(_read_json(path))
    if len(reps) != 2:
        raise RuntimeError("run both sanity_rep processes first")
    curve0 = [
        (int(r["epoch"]), float(r["train_mae"]), float(r["valid_mae"]))
        for r in _read_curve(Path(reps[0]["curve_path"]))
    ]
    curve1 = [
        (int(r["epoch"]), float(r["train_mae"]), float(r["valid_mae"]))
        for r in _read_curve(Path(reps[1]["curve_path"]))
    ]
    n = min(len(curve0), len(curve1))
    max_train = max(abs(curve0[i][1] - curve1[i][1]) for i in range(n))
    max_valid = max(abs(curve0[i][2] - curve1[i][2]) for i in range(n))
    payload = {
        "init_seed": int(init_seed),
        "shuffle_seed": int(shuffle_seed),
        "epochs": int(n),
        "best_valid_rep1": float(reps[0]["best_valid_mae"]),
        "best_valid_rep2": float(reps[1]["best_valid_mae"]),
        "bit_identical_best_valid": bool(
            reps[0]["best_valid_mae"] == reps[1]["best_valid_mae"]
        ),
        "max_train_mae_diff": float(max_train),
        "max_valid_mae_diff": float(max_valid),
        "bit_identical": bool(max_train == 0.0 and max_valid == 0.0),
        "separate_processes": True,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "determinism_sanity.json", payload)
    return payload


def _read_curve(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Phase C -- decomposition
# ---------------------------------------------------------------------------


def _factorial_effects(mae: Mapping[str, float]) -> dict[str, Any]:
    """Balanced 2x2 effect decomposition (cells A=(0,0) B=(0,1) C=(1,0) D=(1,1))."""
    init_effect = [abs(mae["A"] - mae["C"]), abs(mae["B"] - mae["D"])]
    shuffle_effect = [abs(mae["A"] - mae["B"]), abs(mae["C"] - mae["D"])]
    init0_mean = 0.5 * (mae["A"] + mae["B"])
    init1_mean = 0.5 * (mae["C"] + mae["D"])
    shuffle0_mean = 0.5 * (mae["A"] + mae["C"])
    shuffle1_mean = 0.5 * (mae["B"] + mae["D"])
    interaction_signed = (mae["D"] - mae["C"]) - (mae["B"] - mae["A"])
    return {
        "cells_mae": {cell: float(mae[cell]) for cell in ("A", "B", "C", "D")},
        "init_effect_shuffle0": float(init_effect[0]),
        "init_effect_shuffle1": float(init_effect[1]),
        "mean_init_effect": float(np.mean(init_effect)),
        "shuffle_effect_init0": float(shuffle_effect[0]),
        "shuffle_effect_init1": float(shuffle_effect[1]),
        "mean_shuffle_effect": float(np.mean(shuffle_effect)),
        "signed_effects": {
            "init_at_shuffle0": float(mae["A"] - mae["C"]),
            "init_at_shuffle1": float(mae["B"] - mae["D"]),
            "shuffle_at_init0": float(mae["A"] - mae["B"]),
            "shuffle_at_init1": float(mae["C"] - mae["D"]),
        },
        "two_way": {
            "init0_mean": float(init0_mean),
            "init1_mean": float(init1_mean),
            "shuffle0_mean": float(shuffle0_mean),
            "shuffle1_mean": float(shuffle1_mean),
            "init_main_effect": float(abs(init1_mean - init0_mean)),
            "shuffle_main_effect": float(abs(shuffle1_mean - shuffle0_mean)),
            "interaction_signed": float(interaction_signed),
            "interaction_abs": float(abs(interaction_signed)),
        },
    }


def _canonical_run(cell: str) -> dict[str, Any]:
    tag, seed = _CANONICAL_STATE[cell]
    return _read_json(vd._run_path(tag, seed))


def _canonical_state_path(cell: str) -> Path:
    tag, seed = _CANONICAL_STATE[cell]
    return vd.STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"


def _predictions(cell: str, model, loader) -> tuple[np.ndarray, np.ndarray]:
    state = __import__("torch").load(_cell_state_path(cell), map_location="cpu", weights_only=True)
    return vd._predict_state(model, state, loader)


def decompose() -> dict[str, Any]:
    runs = {cell: _cell_run(cell) for cell in CELLS}
    mae = {cell: float(runs[cell]["best_valid_mae"]) for cell in CELLS}
    epochs = {cell: int(runs[cell]["best_epoch"]) for cell in CELLS}
    train = {cell: float(runs[cell]["train_loss_at_best"]) for cell in CELLS}
    gap = {cell: mae[cell] - train[cell] for cell in CELLS}

    det = _factorial_effects(mae)
    signed = det["signed_effects"]
    init_effect = [det["init_effect_shuffle0"], det["init_effect_shuffle1"]]
    shuffle_effect = [det["shuffle_effect_init0"], det["shuffle_effect_init1"]]
    mean_init = det["mean_init_effect"]
    mean_shuffle = det["mean_shuffle_effect"]
    init0_mean = det["two_way"]["init0_mean"]
    init1_mean = det["two_way"]["init1_mean"]
    shuffle0_mean = det["two_way"]["shuffle0_mean"]
    shuffle1_mean = det["two_way"]["shuffle1_mean"]
    init_main = det["two_way"]["init_main_effect"]
    shuffle_main = det["two_way"]["shuffle_main_effect"]
    interaction_signed = det["two_way"]["interaction_signed"]
    interaction_abs = det["two_way"]["interaction_abs"]

    # prediction disagreement / residual correlation on the four cells
    valid_data = vd._valid_data()
    loader = vd._eval_loader(valid_data)
    model = rec.build_recurrent(0)
    preds: dict[str, np.ndarray] = {}
    targets: np.ndarray | None = None
    for cell in CELLS:
        t, p = _predictions(cell, model, loader)
        preds[cell] = p
        targets = t if targets is None else targets
    assert targets is not None
    pair_kinds = {
        "init_AC": ("A", "C"),
        "init_BD": ("B", "D"),
        "shuffle_AB": ("A", "B"),
        "shuffle_CD": ("C", "D"),
        "diag_AD": ("A", "D"),
        "diag_BC": ("B", "C"),
    }
    disagreement = {}
    residual_corr = {}
    pred_corr = {}
    for name, (x, y) in pair_kinds.items():
        disagreement[name] = float(np.mean(np.abs(preds[x] - preds[y])))
        residual_corr[name] = vd._corr(targets - preds[x], targets - preds[y])
        pred_corr[name] = vd._corr(preds[x], preds[y])

    canonical_ref = {}
    for cell, (tag, run_seed) in CANONICAL_REFERENCE.items():
        run = _read_json(vd._run_path(tag, run_seed))
        canonical_ref[cell] = {
            "tag": tag,
            "run_seed": int(run_seed),
            "valid_mae": float(run["best_valid_mae"]),
            "best_epoch": int(run["best_epoch"]),
            "train_at_best": float(run["train_loss_at_best"]),
        }

    canon_runs = {cell: _canonical_run(cell) for cell in ("A", "B", "C", "D")}
    canon_mae = {
        cell: float(canon_runs[cell]["best_valid_mae"])
        for cell in ("A", "B", "C", "D")
    }
    canon = _factorial_effects(canon_mae)
    canon_cells = {
        cell: {
            "tag": _CANONICAL_STATE[cell][0],
            "run_seed": _CANONICAL_STATE[cell][1],
            "init_seed": CANONICAL_INIT_SHUFFLE[cell][0],
            "shuffle_seed": CANONICAL_INIT_SHUFFLE[cell][1],
            "valid_mae": canon_mae[cell],
            "best_epoch": int(canon_runs[cell]["best_epoch"]),
            "train_at_best": float(canon_runs[cell]["train_loss_at_best"]),
        }
        for cell in ("A", "B", "C", "D")
    }
    torch = __import__("torch")
    canon_preds: dict[str, np.ndarray] = {}
    for cell in ("A", "B", "C", "D"):
        state = torch.load(
            _canonical_state_path(cell), map_location="cpu", weights_only=True
        )
        _, canon_preds[cell] = vd._predict_state(model, state, loader)
    canon_disagreement = {}
    canon_residual_corr = {}
    canon_pred_corr = {}
    for name, (x, y) in pair_kinds.items():
        canon_disagreement[name] = float(
            np.mean(np.abs(canon_preds[x] - canon_preds[y]))
        )
        canon_residual_corr[name] = vd._corr(
            targets - canon_preds[x], targets - canon_preds[y]
        )
        canon_pred_corr[name] = vd._corr(canon_preds[x], canon_preds[y])

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "deterministic_algorithms": bool(DETERMINISTIC),
        "canonical_reference": canonical_ref,
        "canonical_factorial": {
            "cells": canon_cells,
            **canon,
            "prediction_disagreement": canon_disagreement,
            "residual_correlation": canon_residual_corr,
            "prediction_correlation": canon_pred_corr,
            "single_noisy_sample": True,
        },
        "cells": {
            cell: {
                "init_seed": CELLS[cell]["init_seed"],
                "shuffle_seed": CELLS[cell]["shuffle_seed"],
                "valid_mae": mae[cell],
                "best_epoch": epochs[cell],
                "train_at_best": train[cell],
                "train_valid_gap": gap[cell],
            }
            for cell in CELLS
        },
        "init_effect_shuffle0": init_effect[0],
        "init_effect_shuffle1": init_effect[1],
        "mean_init_effect": mean_init,
        "shuffle_effect_init0": shuffle_effect[0],
        "shuffle_effect_init1": shuffle_effect[1],
        "mean_shuffle_effect": mean_shuffle,
        "signed_effects": signed,
        "two_way": {
            "init0_mean": init0_mean,
            "init1_mean": init1_mean,
            "shuffle0_mean": shuffle0_mean,
            "shuffle1_mean": shuffle1_mean,
            "init_main_effect": init_main,
            "shuffle_main_effect": shuffle_main,
            "interaction_signed": interaction_signed,
            "interaction_abs": interaction_abs,
        },
        "prediction_disagreement": disagreement,
        "residual_correlation": residual_corr,
        "prediction_correlation": pred_corr,
        "effect_gate": EFFECT_GATE,
        "interaction_gate": INTERACTION_GATE,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decomposition.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase D -- interpretation gate
# ---------------------------------------------------------------------------


def _case_from(
    mean_init: float, mean_shuffle: float, interaction: float
) -> tuple[str, str]:
    both = mean_init >= EFFECT_GATE and mean_shuffle >= EFFECT_GATE
    init_dom = mean_init >= EFFECT_GATE and (mean_init - mean_shuffle) >= EFFECT_GATE
    shuffle_dom = (
        mean_shuffle >= EFFECT_GATE and (mean_shuffle - mean_init) >= EFFECT_GATE
    )
    strong_interaction = interaction >= INTERACTION_GATE and not (
        init_dom or shuffle_dom
    )
    if strong_interaction:
        return "interaction", "strong init x shuffle interaction"
    if both:
        return "both", "both sources materially contribute"
    if init_dom:
        return "init", "initialization-dominated variance"
    if shuffle_dom:
        return "shuffle", "shuffle/data-order-dominated variance"
    return "neither", "neither individually explains the variance"


def decision() -> dict[str, Any]:
    dec = _read_json(RESULTS_DIR / "decomposition.json")
    canon = dec["canonical_factorial"]

    det_case, det_label = _case_from(
        float(dec["mean_init_effect"]),
        float(dec["mean_shuffle_effect"]),
        float(dec["two_way"]["interaction_abs"]),
    )
    canon_case, canon_label = _case_from(
        float(canon["mean_init_effect"]),
        float(canon["mean_shuffle_effect"]),
        float(canon["two_way"]["interaction_abs"]),
    )

    if canon_case == det_case:
        case, label = canon_case, canon_label
    elif canon_case == "interaction" and det_case == "both":
        case = "interaction"
        label = (
            "strong init x shuffle interaction in the canonical 4-thread regime; "
            "both contribute additively in the reproducible deterministic regime"
        )
    else:
        case = canon_case
        label = f"canonical: {canon_label}; deterministic: {det_label}"

    next_priority = {
        "init": "recurrent update stability / residual scaling + gating / normalization / initialization scheme",
        "shuffle": "gradient-noise reduction objective (e.g. SmoothL1/Huber) as a single-model training target",
        "both": "EMA / SWA / architecture-level recurrent stabilization",
        "interaction": "trajectory stabilization rather than further random-seed decomposition",
        "neither": "trajectory stabilization rather than further random-seed decomposition",
    }[case]

    payload = {
        "case": case,
        "label": label,
        "canonical_case": canon_case,
        "canonical_label": canon_label,
        "deterministic_case": det_case,
        "deterministic_label": det_label,
        "mean_init_effect": float(dec["mean_init_effect"]),
        "mean_shuffle_effect": float(dec["mean_shuffle_effect"]),
        "interaction_abs": float(dec["two_way"]["interaction_abs"]),
        "canonical_mean_init_effect": float(canon["mean_init_effect"]),
        "canonical_mean_shuffle_effect": float(canon["mean_shuffle_effect"]),
        "canonical_interaction_abs": float(canon["two_way"]["interaction_abs"]),
        "effect_gate": EFFECT_GATE,
        "interaction_gate": INTERACTION_GATE,
        "next_priority_direction": next_priority,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "soup_summary": _read_json(RESULTS_DIR / "soup_summary.json")
        if (RESULTS_DIR / "soup_summary.json").exists()
        else None,
        "determinism_sanity": _read_json(RESULTS_DIR / "determinism_sanity.json")
        if (RESULTS_DIR / "determinism_sanity.json").exists()
        else None,
        "decomposition": _read_json(RESULTS_DIR / "decomposition.json")
        if (RESULTS_DIR / "decomposition.json").exists()
        else None,
        "decision": _read_json(RESULTS_DIR / "decision.json")
        if (RESULTS_DIR / "decision.json").exists()
        else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "final_report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "soup_seed2",
            "soup_summary",
            "cell",
            "canon_cell",
            "sanity_rep",
            "sanity_compare",
            "decompose",
            "decision",
            "report",
        ],
    )
    parser.add_argument("--cell", choices=["A", "B", "C", "D"], default="B")
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=1)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=SANITY_EPOCHS)
    args = parser.parse_args(argv)

    __import__("torch").set_num_threads(4)
    __import__("torch").use_deterministic_algorithms(bool(DETERMINISTIC))
    if args.stage == "soup_seed2":
        print(json.dumps(soup_seed2(), indent=2, default=str), flush=True)
    if args.stage == "soup_summary":
        print(json.dumps(soup_summary(), indent=2, default=str), flush=True)
    if args.stage == "cell":
        print(json.dumps(run_cell(args.cell), indent=2, default=str), flush=True)
    if args.stage == "canon_cell":
        print(
            json.dumps(run_canonical_cell(args.cell), indent=2, default=str),
            flush=True,
        )
    if args.stage == "sanity_rep":
        payload = sanity_rep(
            args.rep, args.init_seed, args.shuffle_seed, args.epochs
        )
        _write_json(
            RESULTS_DIR
            / f"cell_sanity_{args.init_seed}_{args.shuffle_seed}_rep{args.rep}.json",
            payload,
        )
        print(json.dumps(payload, indent=2, default=str), flush=True)
    if args.stage == "sanity_compare":
        print(
            json.dumps(
                sanity_compare(args.init_seed, args.shuffle_seed),
                indent=2,
                default=str,
            ),
            flush=True,
        )
    if args.stage == "decompose":
        print(json.dumps(decompose(), indent=2, default=str), flush=True)
    if args.stage == "decision":
        print(json.dumps(decision(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
