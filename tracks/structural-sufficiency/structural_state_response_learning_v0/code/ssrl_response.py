"""SSRL v0 response evaluation on the frozen official-Valid directory.

Inputs
------
* ``data/valid_eval_directory.npz`` -- the frozen SFEC v1 / SED v0 / SSTA v0
  Valid observations and rectangle directory (built once, read-only here).
* ``runs/PCQM_SSRL_{ABS,PAIR,RECT}/predictions_epochs.npz`` -- independent
  whole-molecule predictions for every evaluated epoch.

Outputs
-------
* ``analysis/response_metrics.json`` / ``.csv``
* ``analysis/response_comparisons.csv``
* ``analysis/model_comparison.csv``
* ``analysis/response_curves.csv``

The comparison set is fixed before looking at the held-out response scores:

    ABS vs epoch 0, PAIR vs ABS, RECT vs ABS, RECT vs PAIR, RECT vs epoch 0

at each arm's best full-Valid-MAE checkpoint (epoch 0 included) and, separately,
at the fixed final epoch.  No subset is selected by its score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C


ARMS = ("ABS", "PAIR", "RECT")


def _load_eval_dir() -> dict[str, np.ndarray]:
    path = C.DATA_DIR / "valid_eval_directory.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _rect(d: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k[5:]: np.asarray(v) for k, v in d.items() if k.startswith("rect_")}


def _load_run(arm: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    run_dir = C.RUNS_DIR / f"PCQM_SSRL_{arm}"
    run = C.read_json(run_dir / "run.json")
    with np.load(run_dir / "predictions_epochs.npz") as z:
        pred = {k: z[k] for k in z.files}
    return run, pred


def _align_valid(d: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray]) -> None:
    if not np.array_equal(np.asarray(pred["valid_gids"], np.int64),
                          np.asarray(d["valid_gids"], np.int64)):
        raise RuntimeError("saved Valid predictions are not in the frozen Valid row order")
    if not np.array_equal(np.asarray(pred["valid_target_eV"], np.float64),
                          np.asarray(d["valid_y_eV"], np.float64)):
        raise RuntimeError("saved Valid targets do not match the frozen labels")


def _delta_from_molecules(d, pred_eV: np.ndarray) -> np.ndarray:
    cell_pred = np.asarray(pred_eV, np.float64)[np.asarray(d["cell_cache_row"], np.int64)]
    pi = np.asarray(d["pair_pos_i"], np.int64)
    pj = np.asarray(d["pair_pos_j"], np.int64)
    sign = np.asarray(d["pair_sign"], np.float64)
    return sign * (cell_pred[pj] - cell_pred[pi])


def _macro_index(keys: np.ndarray) -> tuple[np.ndarray, int]:
    u, inv = np.unique(np.asarray(keys, np.int64), return_inverse=True)
    return inv.astype(np.int64), int(u.size)


def _block(d_true: np.ndarray, pred: Mapping[str, np.ndarray],
           pair_index: np.ndarray, n_pairs: int, seed: int) -> dict[str, Any]:
    import ssa_interaction as IA

    out: dict[str, Any] = {"models": {}}
    for name, dp in pred.items():
        obs = IA.dd_metrics(d_true, dp)
        mac = IA.macro_metrics(d_true, dp, pair_index, n_pairs, seed=seed)
        out["models"][name] = {
            "observation": obs,
            "pair_type_macro": {k: mac[k] for k in
                                ("n_pairs", "mae_eV", "rmse_eV", "dir_acc", "dir_n", "ci_mae")},
        }
    return out


def _comparisons(d_true: np.ndarray, pred: Mapping[str, np.ndarray],
                 pair_index: np.ndarray, n_pairs: int,
                 comparisons: list[tuple[str, str]], seed: int) -> dict[str, Any]:
    import ssa_interaction as IA

    out: dict[str, Any] = {}
    n = int(d_true.size)
    rng = np.random.default_rng(seed + 100)
    boot = rng.integers(0, n, size=(C.BOOTSTRAP_REPS, n))
    for j, (a, b) in enumerate(comparisons):
        ae_a = np.abs(pred[a] - d_true)
        ae_b = np.abs(pred[b] - d_true)
        mac = IA.macro_contrast(ae_a, ae_b, pair_index, n_pairs, seed=seed + j)
        # observation-equal paired difference with a shared unit bootstrap
        diff = ae_a - ae_b
        means = diff[boot].mean(axis=1)
        out[f"{a}_minus_{b}"] = {
            "observation_equal": {
                "point": float(diff.mean()),
                "ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
                "n": int(n),
                "bootstrap_unit": "observation (rectangle / pair observation)",
            },
            "pair_type_macro": mac,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "response_metrics.json"))
    args = parser.parse_args(argv)

    d = _load_eval_dir()
    rect = _rect(d)
    comparisons = [
        ("ABS", "epoch0"), ("PAIR", "ABS"), ("RECT", "ABS"),
        ("RECT", "PAIR"), ("RECT", "epoch0"),
    ]
    # also report the same contrasts at the fixed final epoch
    comparisons_final = [
        ("ABS_final", "epoch0"), ("PAIR_final", "ABS_final"),
        ("RECT_final", "ABS_final"), ("RECT_final", "PAIR_final"),
        ("RECT_final", "epoch0"),
    ]

    runs: dict[str, dict[str, Any]] = {}
    preds: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        run, pred = _load_run(arm)
        _align_valid(d, pred)
        runs[arm] = run
        preds[arm] = pred
    # all three arms share the original checkpoint at epoch 0.  Re-running the
    # same weights on different GPUs changes the last few float32 bits, so the
    # identity check is numerical (<=1e-5 eV), not byte equality.
    e0 = np.asarray(preds["ABS"]["valid_pred_eV"][0], np.float64)
    for arm in ARMS:
        d0 = np.abs(np.asarray(preds[arm]["valid_pred_eV"][0], np.float64) - e0)
        if float(d0.max()) > 1e-5:
            raise RuntimeError(f"{arm} epoch-0 predictions differ from ABS by {d0.max():.3e} eV")

    def epoch_pred(arm: str, epoch: int) -> np.ndarray:
        epochs = np.asarray(preds[arm]["epochs"], np.int64).tolist()
        if int(epoch) not in epochs:
            raise KeyError((arm, epoch))
        return np.asarray(preds[arm]["valid_pred_eV"][epochs.index(int(epoch))], np.float64)

    # ---------------------------------------------------------------- B / C
    pair_mask_all = np.asarray(d["subset_all_evaluable"], bool)
    pair_subsets = {
        "all_matched": pair_mask_all,
        "unseen_exact_context": np.asarray(d["subset_unseen_exact_context"], bool),
        "unseen_context_q_supported": np.asarray(d["subset_unseen_context_q_supported"], bool),
        "clean_subset": np.asarray(d["subset_clean_subset"], bool),
    }
    # C uses the frozen SSTA directory and its fixed subsets.
    import ssa_interaction as IA

    mol_cols = {
        "m1": np.asarray(d["cell_mol"], np.int64)[np.asarray(rect["pos1_i"], np.int64)],
        "m2": np.asarray(d["cell_mol"], np.int64)[np.asarray(rect["pos1_j"], np.int64)],
        "m3": np.asarray(d["cell_mol"], np.int64)[np.asarray(rect["pos2_i"], np.int64)],
        "m4": np.asarray(d["cell_mol"], np.int64)[np.asarray(rect["pos2_j"], np.int64)],
    }
    disjoint = IA.molecule_disjoint(rect, mol_cols)
    same_idx = np.asarray(d["same_edit_rect_index"], np.int64)
    strict_idx = np.asarray(d["strict_rect_index"], np.int64)
    rect_subsets: dict[str, np.ndarray] = {
        "all": np.ones(len(rect["pair_key"]), bool),
        "both_contexts_unseen": np.asarray(rect["unseen_both"], bool),
        "molecule_disjoint": disjoint,
        "same_edit": np.isin(np.arange(len(rect["pair_key"])), same_idx),
        "strict": np.isin(np.arange(len(rect["pair_key"])), strict_idx),
    }
    rect_pair_types = np.asarray(d["rect_pair_types"], np.int64)
    rect_pair_index_all = np.searchsorted(rect_pair_types, np.asarray(rect["pair_key"], np.int64))
    n_rect_pairs = int(rect_pair_types.size)

    # model names at best and final checkpoints
    best_epoch = {arm: int(runs[arm]["best_epoch"]) for arm in ARMS}
    final_epoch = {arm: int(runs[arm]["final_epoch"]) for arm in ARMS}
    best_models = {"epoch0": e0}
    final_models = {"epoch0": e0}
    for arm in ARMS:
        best_models[arm] = epoch_pred(arm, best_epoch[arm])
        final_models[f"{arm}_final"] = epoch_pred(arm, final_epoch[arm])

    def full_valid(pred: np.ndarray) -> dict[str, Any]:
        y = np.asarray(d["valid_y_eV"], np.float64)
        mae = float(np.mean(np.abs(pred - y)))
        return {
            "n": int(y.size),
            "mae_eV": mae,
            "rmse_eV": float(np.sqrt(np.mean((pred - y) ** 2))),
            "pred_mean_eV": float(pred.mean()),
            "pred_std_eV": float(pred.std()),
        }

    full = {
        "epoch0": full_valid(e0),
        **{arm: full_valid(best_models[arm]) for arm in ARMS},
        **{f"{arm}_final": full_valid(final_models[f"{arm}_final"]) for arm in ARMS},
    }
    # cross-check against the training curve's official evaluator values
    for arm in ARMS:
        curve = {int(r["epoch"]): r for r in runs[arm]["curve"]}
        for tag, ep in (("best", best_epoch[arm]), ("final", final_epoch[arm])):
            key = arm if tag == "best" else f"{arm}_final"
            if abs(full[key]["mae_eV"] - float(curve[ep]["valid_mae_eV"])) > 1e-6:
                raise RuntimeError(f"{arm} {tag} full-Valid MAE does not match its run curve")

    replacement: dict[str, Any] = {}
    for sub_name, mask in pair_subsets.items():
        m = np.asarray(mask, bool)
        if int(m.sum()) == 0:
            replacement[sub_name] = {"n_obs": 0}
            continue
        d_true = np.asarray(d["pair_delta"], np.float64)[m]
        pair_index, n_pairs = _macro_index(np.asarray(d["pair_key"], np.int64)[m])
        model_pred = {"epoch0": _delta_from_molecules(d, best_models["epoch0"])[m]}
        for arm in ARMS:
            model_pred[arm] = _delta_from_molecules(d, best_models[arm])[m]
        rec = {
            "n_obs": int(m.sum()),
            "n_pair_types": int(n_pairs),
            "n_contexts": int(np.unique(np.asarray(d["pair_ctx"], np.int64)[m]).size),
            "metrics": _block(d_true, model_pred, pair_index, n_pairs, seed=11),
            "comparisons": _comparisons(d_true, model_pred, pair_index, n_pairs,
                                         comparisons, seed=21),
        }
        # fixed final-epoch sensitivity
        final_pred = {"epoch0": model_pred["epoch0"]}
        for arm in ARMS:
            final_pred[f"{arm}_final"] = _delta_from_molecules(
                d, final_models[f"{arm}_final"])[m]
        rec["final_epoch"] = {
            "metrics": _block(d_true, final_pred, pair_index, n_pairs, seed=31),
            "comparisons": _comparisons(d_true, final_pred, pair_index, n_pairs,
                                         comparisons_final, seed=41),
        }
        replacement[sub_name] = rec

    interaction: dict[str, Any] = {}
    for sub_name, mask in rect_subsets.items():
        m = np.asarray(mask, bool)
        if int(m.sum()) == 0:
            interaction[sub_name] = {"n_rect": 0}
            continue
        d_true = (np.asarray(rect["delta_true_2"], np.float64)[m]
                  - np.asarray(rect["delta_true_1"], np.float64)[m])
        pair_index = rect_pair_index_all[m]
        n_pairs = n_rect_pairs
        model_pred = {"epoch0": (_delta_from_molecules(d, best_models["epoch0"])[
            np.asarray(rect["row2"], np.int64)[m]]
            - _delta_from_molecules(d, best_models["epoch0"])[
                np.asarray(rect["row1"], np.int64)[m]])}
        for arm in ARMS:
            dp = _delta_from_molecules(d, best_models[arm])
            model_pred[arm] = dp[np.asarray(rect["row2"], np.int64)[m]] - dp[
                np.asarray(rect["row1"], np.int64)[m]]
        rec = {
            "n_rect": int(m.sum()),
            "n_pair_types": int(np.unique(pair_index).size),
            "n_contexts": int(np.unique(np.r_[np.asarray(rect["ctx1"], np.int64)[m],
                                              np.asarray(rect["ctx2"], np.int64)[m]]).size),
            "n_unique_molecules": int(np.unique(np.r_[
                mol_cols["m1"][m], mol_cols["m2"][m],
                mol_cols["m3"][m], mol_cols["m4"][m]]).size),
            "metrics": _block(d_true, model_pred, pair_index, n_pairs, seed=51),
            "comparisons": _comparisons(d_true, model_pred, pair_index, n_pairs,
                                         comparisons, seed=61),
        }
        final_pred = {"epoch0": model_pred["epoch0"]}
        for arm in ARMS:
            dp = _delta_from_molecules(d, final_models[f"{arm}_final"])
            final_pred[f"{arm}_final"] = dp[np.asarray(rect["row2"], np.int64)[m]] - dp[
                np.asarray(rect["row1"], np.int64)[m]]
        rec["final_epoch"] = {
            "metrics": _block(d_true, final_pred, pair_index, n_pairs, seed=71),
            "comparisons": _comparisons(d_true, final_pred, pair_index, n_pairs,
                                         comparisons_final, seed=81),
        }
        interaction[sub_name] = rec

    # ------------------------------------------------------------ per epoch
    per_epoch: list[dict[str, Any]] = []
    for arm in ARMS:
        epochs = np.asarray(preds[arm]["epochs"], np.int64)
        for j, ep in enumerate(epochs.tolist()):
            pv = np.asarray(preds[arm]["valid_pred_eV"][j], np.float64)
            full_rec = full_valid(pv)
            d_all = np.asarray(d["pair_delta"], np.float64)
            dp_all = _delta_from_molecules(d, pv)
            # compact row with the primary three summaries
            m = pair_mask_all
            pi, npi = _macro_index(np.asarray(d["pair_key"], np.int64)[m])
            b_all = _block(d_all[m], {"model": dp_all[m]}, pi, npi, seed=91)
            m2 = pair_subsets["unseen_exact_context"]
            pi2, npi2 = _macro_index(np.asarray(d["pair_key"], np.int64)[m2])
            b_un = _block(d_all[m2], {"model": dp_all[m2]}, pi2, npi2, seed=92)
            dt_c = (np.asarray(rect["delta_true_2"], np.float64)
                    - np.asarray(rect["delta_true_1"], np.float64))
            dp_c = dp_all[np.asarray(rect["row2"], np.int64)] - dp_all[
                np.asarray(rect["row1"], np.int64)]
            c_all = _block(dt_c, {"model": dp_c}, rect_pair_index_all, n_rect_pairs, seed=93)
            c_un = _block(dt_c[rect_subsets["both_contexts_unseen"]],
                           {"model": dp_c[rect_subsets["both_contexts_unseen"]]},
                           rect_pair_index_all[rect_subsets["both_contexts_unseen"]],
                           n_rect_pairs, seed=94)
            c_dis = _block(dt_c[disjoint], {"model": dp_c[disjoint]},
                            rect_pair_index_all[disjoint], n_rect_pairs, seed=95)
            per_epoch.append({
                "arm": arm,
                "epoch": int(ep),
                "is_best": bool(int(ep) == best_epoch[arm]),
                "is_final": bool(int(ep) == final_epoch[arm]),
                "full_valid_mae_eV": float(full_rec["mae_eV"]),
                "B_all_observation_mae_eV": float(b_all["models"]["model"]["observation"]["mae_eV"]),
                "B_all_pair_type_macro_mae_eV": float(b_all["models"]["model"]["pair_type_macro"]["mae_eV"]),
                "B_unseen_observation_mae_eV": float(b_un["models"]["model"]["observation"]["mae_eV"]),
                "B_unseen_pair_type_macro_mae_eV": float(b_un["models"]["model"]["pair_type_macro"]["mae_eV"]),
                "C_all_observation_mae_eV": float(c_all["models"]["model"]["observation"]["mae_eV"]),
                "C_all_pair_type_macro_mae_eV": float(c_all["models"]["model"]["pair_type_macro"]["mae_eV"]),
                "C_unseen_pair_type_macro_mae_eV": float(c_un["models"]["model"]["pair_type_macro"]["mae_eV"]),
                "C_disjoint_pair_type_macro_mae_eV": float(c_dis["models"]["model"]["pair_type_macro"]["mae_eV"]),
                "train_subset_mae_eV": float(runs[arm]["curve"][j]["train_subset_mae_eV"]),
            })

    # ------------------------------------------------------------- summary
    def obs_mae(block: Mapping[str, Any]) -> float:
        return float(block["models"]["model"]["observation"]["mae_eV"])

    def mac_mae(block: Mapping[str, Any]) -> float:
        return float(block["models"]["model"]["pair_type_macro"]["mae_eV"])

    model_rows = []
    for label, pv in (("epoch0", e0),
                      *[(arm, best_models[arm]) for arm in ARMS],
                      *[(f"{arm}_final", final_models[f"{arm}_final"]) for arm in ARMS]):
        dp_all = _delta_from_molecules(d, pv)
        m = pair_mask_all
        pi, npi = _macro_index(np.asarray(d["pair_key"], np.int64)[m])
        b_all = _block(np.asarray(d["pair_delta"], np.float64)[m],
                       {"model": dp_all[m]}, pi, npi, seed=101)
        m2 = pair_subsets["unseen_exact_context"]
        pi2, npi2 = _macro_index(np.asarray(d["pair_key"], np.int64)[m2])
        b_un = _block(np.asarray(d["pair_delta"], np.float64)[m2],
                      {"model": dp_all[m2]}, pi2, npi2, seed=102)
        dt_c = (np.asarray(rect["delta_true_2"], np.float64)
                - np.asarray(rect["delta_true_1"], np.float64))
        dp_c = dp_all[np.asarray(rect["row2"], np.int64)] - dp_all[
            np.asarray(rect["row1"], np.int64)]
        c_all = _block(dt_c, {"model": dp_c}, rect_pair_index_all, n_rect_pairs, seed=103)
        c_un = _block(dt_c[rect_subsets["both_contexts_unseen"]],
                       {"model": dp_c[rect_subsets["both_contexts_unseen"]]},
                       rect_pair_index_all[rect_subsets["both_contexts_unseen"]],
                       n_rect_pairs, seed=104)
        c_dis = _block(dt_c[disjoint], {"model": dp_c[disjoint]},
                        rect_pair_index_all[disjoint], n_rect_pairs, seed=105)
        c_same = _block(dt_c[same_idx], {"model": dp_c[same_idx]},
                         rect_pair_index_all[same_idx], n_rect_pairs, seed=106)
        model_rows.append({
            "model": label,
            "full_valid_mae_eV": full_valid(pv)["mae_eV"],
            "B_all_observation_mae_eV": obs_mae(b_all),
            "B_all_pair_type_macro_mae_eV": mac_mae(b_all),
            "B_unseen_observation_mae_eV": obs_mae(b_un),
            "B_unseen_pair_type_macro_mae_eV": mac_mae(b_un),
            "C_all_observation_mae_eV": obs_mae(c_all),
            "C_all_pair_type_macro_mae_eV": mac_mae(c_all),
            "C_unseen_pair_type_macro_mae_eV": mac_mae(c_un),
            "C_disjoint_pair_type_macro_mae_eV": mac_mae(c_dis),
            "C_same_edit_pair_type_macro_mae_eV": mac_mae(c_same),
        })

    # flatten comparison rows for the report
    comparison_rows: list[dict[str, Any]] = []
    for group_name, group in (("replacement_delta", replacement),
                              ("interaction_D", interaction)):
        for sub_name, rec in group.items():
            for phase in ("best", "final_epoch"):
                if phase not in rec:
                    continue
                comp = rec[phase]["comparisons"]
                for key, c in comp.items():
                    comparison_rows.append({
                        "group": group_name, "subset": sub_name, "phase": phase,
                        "comparison": key,
                        "observation_delta": c["observation_equal"]["point"],
                        "observation_ci_lo": c["observation_equal"]["ci"][0],
                        "observation_ci_hi": c["observation_equal"]["ci"][1],
                        "observation_n": c["observation_equal"]["n"],
                        "macro_delta": c["pair_type_macro"]["point"],
                        "macro_ci_lo": c["pair_type_macro"]["ci"][0],
                        "macro_ci_hi": c["pair_type_macro"]["ci"][1],
                        "macro_n_pairs": c["pair_type_macro"]["n_pairs"],
                    })

    payload = {
        "schema": "ssrl-v0-response-metrics",
        "round": "structural-state-response-learning-v0",
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "data": {
            "eval_directory": C.track_rel(C.DATA_DIR / "valid_eval_directory.npz"),
            "eval_directory_sha256": C.sha256_file(C.DATA_DIR / "valid_eval_directory.npz"),
            "official_valid_rows": int(np.asarray(d["valid_gids"]).size),
            "replacement_observations": int(pair_mask_all.sum()),
            "replacement_pair_types": int(_macro_index(
                np.asarray(d["pair_key"], np.int64)[pair_mask_all])[1]),
            "interaction_rectangles": int(len(rect["pair_key"])),
            "interaction_pair_types": int(n_rect_pairs),
            "same_edit_rectangles": int(same_idx.size),
            "strict_rectangles": int(strict_idx.size),
            "subsets": {
                "pair_observation_counts": {k: int(np.asarray(v).sum())
                                            for k, v in pair_subsets.items()},
                "rectangle_counts": {k: int(np.asarray(v).sum())
                                      for k, v in rect_subsets.items()},
            },
            "test_labels_read": False,
        },
        "checkpoint_selection": {
            arm: {
                "best_epoch": int(best_epoch[arm]),
                "best_valid_mae_eV": float(runs[arm]["best_valid_mae_eV"]),
                "final_epoch": int(final_epoch[arm]),
                "final_valid_mae_eV": float(runs[arm]["final_valid_mae_eV"]),
                "epoch0_valid_mae_eV": float(runs[arm]["epoch0_valid_mae_eV"]),
                "source_checkpoint_sha256": runs[arm]["source_checkpoint_sha256"],
            }
            for arm in ARMS
        },
        "full_valid": full,
        "replacement_delta": replacement,
        "interaction_D": interaction,
        "model_comparison": model_rows,
        "per_epoch": per_epoch,
        "comparison_set": [f"{a}_minus_{b}" for a, b in comparisons],
        "comparison_set_final": [f"{a}_minus_{b}" for a, b in comparisons_final],
        "bootstrap": {
            "reps": int(C.BOOTSTRAP_REPS),
            "unit": "pair type for macro; observation for the observation-equal interval",
            "direction_threshold_eV": float(C.DIRECTION_THRESHOLD_EV),
        },
        "test_evaluated": False,
    }
    C.write_json(Path(args.out), payload)
    C.write_csv(C.ANALYSIS_DIR / "model_comparison.csv", model_rows)
    C.write_csv(C.ANALYSIS_DIR / "response_curves.csv", per_epoch)
    C.write_csv(C.ANALYSIS_DIR / "response_comparisons.csv", comparison_rows,
                columns=["group", "subset", "phase", "comparison",
                         "observation_delta", "observation_ci_lo", "observation_ci_hi",
                         "observation_n", "macro_delta", "macro_ci_lo", "macro_ci_hi",
                         "macro_n_pairs"])
    print(f"wrote {args.out}; {len(model_rows)} model rows, "
          f"{len(per_epoch)} epoch rows, {len(comparison_rows)} comparisons", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
