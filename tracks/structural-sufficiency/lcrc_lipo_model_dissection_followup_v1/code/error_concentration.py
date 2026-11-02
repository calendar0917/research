"""Per-sample Valid error concentration for the already trained checkpoints.

Objects are fixed to repeat 122, variants A0/FULL, A3/NO_ADD, A5/ADD_ONLY,
C1/NO_BOND_MEAN and B5/MARGINAL_BT2, seeds 0/1/2.  A0 is the within-seed
reference.  Predictions come only from the existing ``valid_predictions.npz``
files and are aligned by published ``row_id`` (never by position or SMILES).
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np

import followup_common as F

HARD_TOP = (1, 5, 10)


def _sort_by_full_error(rows: np.ndarray, e_full: np.ndarray) -> np.ndarray:
    """|error| descending, row_id ascending for ties."""
    order = np.lexsort((rows, -np.abs(e_full)))
    return np.asarray(order, dtype=np.int64)


def _set_stats(
    mask: np.ndarray,
    rows: np.ndarray,
    e_full: np.ndarray,
    e_cand: np.ndarray,
    dsse: np.ndarray,
    dae: np.ndarray,
    gross_gain: float,
    gross_loss: float,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    outside = ~mask
    sse_full = float(np.sum(e_full * e_full))
    sse_cand = float(np.sum(e_cand * e_cand))
    set_sse_full = float(np.sum(e_full[mask] ** 2))
    set_sse_cand = float(np.sum(e_cand[mask] ** 2))
    inside = {
        "n": int(mask.sum()),
        "sse_full": set_sse_full,
        "share_of_full_sse": (set_sse_full / sse_full) if sse_full > 0 else None,
        "sse_candidate": set_sse_cand,
        "delta_sse": float(np.sum(dsse[mask])),
        "delta_ae": float(np.sum(dae[mask])),
        "gross_gain": float(np.sum(np.maximum(-dsse[mask], 0.0))),
        "gross_loss": float(np.sum(np.maximum(dsse[mask], 0.0))),
        "share_of_total_gross_gain": (
            float(np.sum(np.maximum(-dsse[mask], 0.0)) / gross_gain) if gross_gain > 0 else None
        ),
        "share_of_total_gross_loss": (
            float(np.sum(np.maximum(dsse[mask], 0.0)) / gross_loss) if gross_loss > 0 else None
        ),
        "ae_gross_gain": float(np.sum(np.maximum(-dae[mask], 0.0))),
        "ae_gross_loss": float(np.sum(np.maximum(dae[mask], 0.0))),
        "n_improved": int(np.sum(dae[mask] < 0)),
        "n_worsened": int(np.sum(dae[mask] > 0)),
        "n_equal": int(np.sum(dae[mask] == 0)),
    }
    outside_mask = outside
    n_out = int(outside_mask.sum())
    out: dict[str, Any] = {
        "n": n_out,
        "delta_sse": float(np.sum(dsse[outside_mask])),
        "delta_ae": float(np.sum(dae[outside_mask])),
        "gross_gain": float(np.sum(np.maximum(-dsse[outside_mask], 0.0))),
        "gross_loss": float(np.sum(np.maximum(dsse[outside_mask], 0.0))),
        "share_of_total_gross_gain": (
            float(np.sum(np.maximum(-dsse[outside_mask], 0.0)) / gross_gain) if gross_gain > 0 else None
        ),
        "share_of_total_gross_loss": (
            float(np.sum(np.maximum(dsse[outside_mask], 0.0)) / gross_loss) if gross_loss > 0 else None
        ),
        "ae_gross_gain": float(np.sum(np.maximum(-dae[outside_mask], 0.0))),
        "ae_gross_loss": float(np.sum(np.maximum(dae[outside_mask], 0.0))),
        "n_improved": int(np.sum(dae[outside_mask] < 0)),
        "n_worsened": int(np.sum(dae[outside_mask] > 0)),
        "n_equal": int(np.sum(dae[outside_mask] == 0)),
    }
    if n_out:
        full_rmse = float(np.sqrt(np.mean(e_full[outside_mask] ** 2)))
        cand_rmse = float(np.sqrt(np.mean(e_cand[outside_mask] ** 2)))
        full_mae = float(np.mean(np.abs(e_full[outside_mask])))
        cand_mae = float(np.mean(np.abs(e_cand[outside_mask])))
        out.update(
            {
                "full_rmse": full_rmse,
                "candidate_rmse": cand_rmse,
                "delta_rmse": cand_rmse - full_rmse,
                "full_mae": full_mae,
                "candidate_mae": cand_mae,
                "delta_mae": cand_mae - full_mae,
            }
        )
    else:
        out.update(
            {
                "full_rmse": None,
                "candidate_rmse": None,
                "delta_rmse": None,
                "full_mae": None,
                "candidate_mae": None,
                "delta_mae": None,
            }
        )
    return {"inside": inside, "outside": out}


def _subset_metrics(mask: np.ndarray, e_full: np.ndarray, e_cand: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            "n": 0,
            "full_rmse": None,
            "candidate_rmse": None,
            "delta_rmse": None,
            "full_mae": None,
            "candidate_mae": None,
            "delta_mae": None,
        }
    full_rmse = float(np.sqrt(np.mean(e_full[mask] ** 2)))
    cand_rmse = float(np.sqrt(np.mean(e_cand[mask] ** 2)))
    full_mae = float(np.mean(np.abs(e_full[mask])))
    cand_mae = float(np.mean(np.abs(e_cand[mask])))
    return {
        "n": int(mask.sum()),
        "full_rmse": full_rmse,
        "candidate_rmse": cand_rmse,
        "delta_rmse": cand_rmse - full_rmse,
        "full_mae": full_mae,
        "candidate_mae": cand_mae,
        "delta_mae": cand_mae - full_mae,
    }


def analyse(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    F.ERRORS.mkdir(parents=True, exist_ok=True)
    main_rows = list(F.load_main_rows() if rows is None else rows)
    main_index = {(int(r["repeat"]), str(r["variant"]), int(r["seed"])): r for r in main_rows}
    ref = F.valid_reference()
    valid_rows = np.asarray(ref["rows"], dtype=np.int64)
    y = np.asarray(ref["y"], dtype=np.float64)
    smiles = list(ref["smiles"])

    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    alignment: list[dict[str, Any]] = []
    for variant in F.ERROR_VARIANTS:
        for seed in F.SEEDS:
            raw = F.load_valid_predictions(F.MAIN_REPEAT, variant, seed)
            aligned = F.align_predictions(raw, ref)
            alignment.append({"variant": variant, "seed": seed, "ok": aligned["ok"], **{k: aligned[k] for k in ("reason", "y_max_abs_diff") if k in aligned}})
            if not aligned["ok"]:
                raise SystemExit(f"prediction alignment failed for {variant} s{seed}: {aligned}")
            payloads[(variant, seed)] = aligned

    # The reference predictions must be identical across seeds for the same
    # published row and target.  They are different trained models, so only the
    # rows/target identity is asserted here.
    for variant in F.ERROR_VARIANTS:
        for seed in F.SEEDS:
            if not np.array_equal(payloads[(variant, seed)]["rows"], valid_rows):
                raise SystemExit("row order mismatch after alignment")

    sample_rows: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    hard_sets: dict[str, Any] = {}
    top_abs_dsse_rows: list[dict[str, Any]] = []
    seed0_top10_rows: list[dict[str, Any]] = []

    # Fixed per-seed FULL top sets are defined *before* looking at candidates.
    per_seed_sets: dict[int, dict[str, np.ndarray]] = {}
    per_seed_orders: dict[int, np.ndarray] = {}
    for seed in F.SEEDS:
        e_full = payloads[("A0", seed)]["raw"] - y
        order = _sort_by_full_error(valid_rows, e_full)
        per_seed_orders[seed] = order
        per_seed_sets[seed] = {
            f"full_s{seed}_top{k}": order[:k] for k in HARD_TOP
        }
        top1 = order[0]
        hard_sets[f"full_s{seed}_top1"] = {
            "definition": f"FULL seed{seed}: |error| descending, row_id ascending tie-break",
            "n": 1,
            "row_ids": [int(valid_rows[top1])],
            "abs_errors": [float(abs(e_full[top1]))],
            "rank0_row_id": int(valid_rows[top1]),
        }
        hard_sets[f"full_s{seed}_top5"] = {
            "definition": f"FULL seed{seed}: |error| descending, row_id ascending tie-break",
            "n": 5,
            "row_ids": [int(valid_rows[i]) for i in order[:5]],
            "abs_errors": [float(abs(e_full[i])) for i in order[:5]],
        }
        hard_sets[f"full_s{seed}_top10"] = {
            "definition": f"FULL seed{seed}: |error| descending, row_id ascending tie-break",
            "n": 10,
            "row_ids": [int(valid_rows[i]) for i in order[:10]],
            "abs_errors": [float(abs(e_full[i])) for i in order[:10]],
        }

    # Cross-seed common sets are fixed by FULL seed0's row ids.
    seed0_order = per_seed_orders[0]
    cross_sets: dict[str, np.ndarray] = {}
    for k in HARD_TOP:
        row_ids = [int(valid_rows[i]) for i in seed0_order[:k]]
        mask = np.isin(valid_rows, np.asarray(row_ids, dtype=np.int64))
        cross_sets[f"full_s0_common_top{k}"] = np.where(mask)[0]
        hard_sets[f"full_s0_common_top{k}"] = {
            "definition": "FULL seed0 ranking; same row_ids reused in every seed",
            "n": k,
            "row_ids": row_ids,
            "abs_errors_seed0": [float(abs(payloads[("A0", 0)]["raw"][i] - y[i])) for i in seed0_order[:k]],
        }

    row2087_pos = int(np.where(valid_rows == 2087)[0][0])
    row2087_mask = valid_rows == 2087
    row2087_info = {
        "row_id": 2087,
        "position_zero_based": row2087_pos,
        "smiles": smiles[row2087_pos],
        "y": float(y[row2087_pos]),
        "full_pred_seed0": float(payloads[("A0", 0)]["raw"][row2087_pos]),
        "full_abs_error_seed0": float(abs(payloads[("A0", 0)]["raw"][row2087_pos] - y[row2087_pos])),
        "full_sse_seed0": float((payloads[("A0", 0)]["raw"][row2087_pos] - y[row2087_pos]) ** 2),
        "full_sse_total_seed0": float(np.sum((payloads[("A0", 0)]["raw"] - y) ** 2)),
    }
    row2087_info["share_of_full_sse_seed0"] = (
        row2087_info["full_sse_seed0"] / row2087_info["full_sse_total_seed0"]
        if row2087_info["full_sse_total_seed0"] > 0
        else None
    )
    row2087_info["seed0_full_abs_rank_zero_based"] = int(np.where(seed0_order == row2087_pos)[0][0])
    row2087_info["seed0_top10_member"] = bool(np.where(seed0_order == row2087_pos)[0][0] < 10)
    # The original attribution table evaluated this molecule in a different
    # batching context; record its value too instead of silently replacing it.
    attribution = F.read_json(F.ORIG_ANALYSIS / "attribution" / "path_loo.json")
    attr_example = next(x for x in attribution["examples"] if int(x["row_id"]) == 2087)
    row2087_info["original_attribution_context"] = {
        "prediction": float(attr_example["prediction"]),
        "y_true": float(attr_example["y_true"]),
        "abs_error": float(attr_example["abs_error"]),
        "valid_position": int(attr_example["valid_position"]),
        "smiles": str(attr_example["smiles"]),
        "sse": float((attr_example["prediction"] - attr_example["y_true"]) ** 2),
    }
    row2087_info["attribution_minus_saved_prediction"] = (
        row2087_info["original_attribution_context"]["prediction"] - row2087_info["full_pred_seed0"]
    )
    row2087_info["attribution_context_note"] = (
        "The saved Valid prediction file gives -7.021600776; the original single-molecule attribution "
        "JSON gives -7.021601929. The 1.15e-6 difference is a batching/precision difference, not a row "
        "or label mismatch. B uses the saved Valid predictions; the attribution value is recorded for audit."
    )
    hard_sets["row2087"] = {
        "definition": "fixed a-priori single-row diagnostic; not an outlier label",
        **row2087_info,
    }

    # Main per-candidate analyses.
    for variant in F.ERROR_VARIANTS:
        for seed in F.SEEDS:
            full = payloads[("A0", seed)]["raw"]
            cand = payloads[(variant, seed)]["raw"]
            e_full = full - y
            e_cand = cand - y
            dsse = e_cand * e_cand - e_full * e_full
            dae = np.abs(e_cand) - np.abs(e_full)
            gross_gain = float(np.sum(np.maximum(-dsse, 0.0)))
            gross_loss = float(np.sum(np.maximum(dsse, 0.0)))
            net = float(np.sum(dsse))
            sse_full = float(np.sum(e_full * e_full))
            sse_cand = float(np.sum(e_cand * e_cand))
            m_full = F.metrics(full, y)
            m_cand = F.metrics(cand, y)
            main = main_index[(F.MAIN_REPEAT, variant, seed)]
            main_abs_rmse = abs(float(main["valid_rmse"]) - m_cand["rmse"])
            main_abs_mae = abs(float(main["valid_mae"]) - m_cand["mae"])
            item = {
                "variant": variant,
                "seed": seed,
                "full_metrics": m_full,
                "candidate_metrics": m_cand,
                "delta_rmse": m_cand["rmse"] - m_full["rmse"],
                "delta_mae": m_cand["mae"] - m_full["mae"],
                "sse_full": sse_full,
                "sse_candidate": sse_cand,
                "delta_sse": net,
                "gross_gain": gross_gain,
                "gross_loss": gross_loss,
                "net_equals_loss_minus_gain": bool(abs(net - (gross_loss - gross_gain)) <= 1e-9),
                "gross_gain_fraction_of_full_sse": gross_gain / sse_full if sse_full > 0 else None,
                "gross_loss_fraction_of_full_sse": gross_loss / sse_full if sse_full > 0 else None,
                "ae_gross_gain": float(np.sum(np.maximum(-dae, 0.0))),
                "ae_gross_loss": float(np.sum(np.maximum(dae, 0.0))),
                "ae_net": float(np.sum(dae)),
                "n_ae_improved": int(np.sum(dae < 0)),
                "n_ae_worsened": int(np.sum(dae > 0)),
                "n_ae_equal": int(np.sum(dae == 0)),
                "median_delta_ae": float(np.median(dae)),
                "mean_delta_ae": float(np.mean(dae)),
                "max_abs_delta_sse": float(np.max(np.abs(dsse))),
                "main_runs_valid_rmse": float(main["valid_rmse"]),
                "main_runs_valid_mae": float(main["valid_mae"]),
                "main_abs_rmse_difference": main_abs_rmse,
                "main_abs_mae_difference": main_abs_mae,
                "main_metrics_match_tolerance": bool(main_abs_rmse <= 1e-6 and main_abs_mae <= 1e-6),
                "sets": {},
            }
            # Fixed set diagnostics, including complement metrics.
            for set_name, positions in {
                **per_seed_sets[seed],
                **cross_sets,
                "row2087": np.asarray([row2087_pos], dtype=np.int64),
            }.items():
                mask = np.zeros(valid_rows.size, dtype=bool)
                mask[positions] = True
                stats = _set_stats(mask, valid_rows, e_full, e_cand, dsse, dae, gross_gain, gross_loss)
                stats["set_row_ids"] = [int(valid_rows[i]) for i in positions]
                item["sets"][set_name] = stats
            item["sets"]["all_valid"] = {
                "inside": {
                    "n": int(valid_rows.size),
                    "sse_full": sse_full,
                    "share_of_full_sse": 1.0,
                    "sse_candidate": sse_cand,
                    "delta_sse": net,
                    "delta_ae": float(np.sum(dae)),
                    "gross_gain": gross_gain,
                    "gross_loss": gross_loss,
                    "share_of_total_gross_gain": 1.0 if gross_gain > 0 else None,
                    "share_of_total_gross_loss": 1.0 if gross_loss > 0 else None,
                    "n_improved": int(np.sum(dae < 0)),
                    "n_worsened": int(np.sum(dae > 0)),
                    "n_equal": int(np.sum(dae == 0)),
                },
                "outside": {
                    "n": 0,
                    "delta_sse": None,
                    "delta_ae": None,
                    "gross_gain": None,
                    "gross_loss": None,
                    "share_of_total_gross_gain": None,
                    "share_of_total_gross_loss": None,
                    "n_improved": None,
                    "n_worsened": None,
                    "n_equal": None,
                },
            }
            decomposition.append(item)

            for pos in range(valid_rows.size):
                sample_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "position": pos,
                        "row_id": int(valid_rows[pos]),
                        "smiles": smiles[pos],
                        "y": float(y[pos]),
                        "pred_full": float(full[pos]),
                        "pred_candidate": float(cand[pos]),
                        "e_full": float(e_full[pos]),
                        "e_candidate": float(e_cand[pos]),
                        "abs_e_full": float(abs(e_full[pos])),
                        "abs_e_candidate": float(abs(e_cand[pos])),
                        "delta_se": float(dsse[pos]),
                        "delta_ae": float(dae[pos]),
                        "full_abs_rank": int(np.where(per_seed_orders[seed] == pos)[0][0]),
                        "in_full_s0_common_top10": bool(valid_rows[pos] in hard_sets["full_s0_common_top10"]["row_ids"]),
                    }
                )

            # Contribution-ranking display only; explicitly not a new subset.
            order_abs = np.lexsort((valid_rows, -np.abs(dsse)))[:5]
            for rank, pos in enumerate(order_abs, start=1):
                top_abs_dsse_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "rank_abs_delta_se": rank,
                        "row_id": int(valid_rows[pos]),
                        "smiles": smiles[pos],
                        "y": float(y[pos]),
                        "e_full": float(e_full[pos]),
                        "e_candidate": float(e_cand[pos]),
                        "delta_se": float(dsse[pos]),
                        "delta_ae": float(dae[pos]),
                        "display_only_not_an_evaluation_subset": True,
                    }
                )

            if variant != "A0" and seed == 0:
                # Filled once below for the seed0 FULL top10 display.
                pass

    # Seed0 FULL top10 detailed comparison across candidates.
    top10_positions = per_seed_sets[0]["full_s0_top10"]
    for rank, pos in enumerate(top10_positions, start=1):
        row: dict[str, Any] = {
            "full_rank_seed0": rank,
            "row_id": int(valid_rows[pos]),
            "position": int(pos),
            "smiles": smiles[pos],
            "y": float(y[pos]),
            "pred_full_s0": float(payloads[("A0", 0)]["raw"][pos]),
            "e_full_s0": float(payloads[("A0", 0)]["raw"][pos] - y[pos]),
            "abs_e_full_s0": float(abs(payloads[("A0", 0)]["raw"][pos] - y[pos])),
        }
        for variant in F.ERROR_VARIANTS:
            pred = payloads[(variant, 0)]["raw"][pos]
            e = pred - y[pos]
            e_full = payloads[("A0", 0)]["raw"][pos] - y[pos]
            row[f"pred_{variant}_s0"] = float(pred)
            row[f"e_{variant}_s0"] = float(e)
            row[f"delta_se_{variant}_s0"] = float(e * e - e_full * e_full)
            row[f"delta_ae_{variant}_s0"] = float(abs(e) - abs(e_full))
        seed0_top10_rows.append(row)

    F.write_csv(F.ERRORS / "paired_samples.csv", sample_rows)
    F.write_csv(F.ERRORS / "top5_abs_delta_se_display.csv", top_abs_dsse_rows)
    F.write_csv(F.ERRORS / "seed0_full_top10_rows.csv", seed0_top10_rows)
    F.write_json(F.ERRORS / "error_decomposition.json", {"items": decomposition})
    F.write_json(F.ERRORS / "hard_sets.json", hard_sets)
    F.write_json(
        F.ERRORS / "alignment.json",
        {
            "valid_rows": int(valid_rows.size),
            "row_id_sha256": F.sha256_array(valid_rows),
            "y_sha256": F.sha256_array(y),
            "rows_equal_published_split": True,
            "per_prediction": alignment,
            "all_ok": all(x["ok"] for x in alignment),
        },
    )

    # Seed-level descriptive summaries (not pooled across seeds).
    summary: dict[str, Any] = {"variants": {}, "notes": []}
    for variant in F.ERROR_VARIANTS:
        items = [x for x in decomposition if x["variant"] == variant]
        summary["variants"][variant] = {
            "per_seed": {
                str(x["seed"]): {
                    "delta_rmse": x["delta_rmse"],
                    "delta_mae": x["delta_mae"],
                    "delta_sse": x["delta_sse"],
                    "gross_gain": x["gross_gain"],
                    "gross_loss": x["gross_loss"],
                    "n_ae_improved": x["n_ae_improved"],
                    "n_ae_worsened": x["n_ae_worsened"],
                    "n_ae_equal": x["n_ae_equal"],
                    "median_delta_ae": x["median_delta_ae"],
                }
                for x in items
            },
            "mean_delta_rmse": float(np.mean([x["delta_rmse"] for x in items])),
            "mean_delta_mae": float(np.mean([x["delta_mae"] for x in items])),
            "mean_delta_sse": float(np.mean([x["delta_sse"] for x in items])),
            "mean_gross_gain": float(np.mean([x["gross_gain"] for x in items])),
            "mean_gross_loss": float(np.mean([x["gross_loss"] for x in items])),
            "mean_n_ae_improved": float(np.mean([x["n_ae_improved"] for x in items])),
            "mean_n_ae_worsened": float(np.mean([x["n_ae_worsened"] for x in items])),
            "mean_median_delta_ae": float(np.mean([x["median_delta_ae"] for x in items])),
            "all_seeds_delta_rmse_sign": (
                "all_negative" if all(x["delta_rmse"] < 0 for x in items) else "all_positive" if all(x["delta_rmse"] > 0 for x in items) else "mixed"
            ),
            "all_seeds_delta_mae_sign": (
                "all_negative" if all(x["delta_mae"] < 0 for x in items) else "all_positive" if all(x["delta_mae"] > 0 for x in items) else "mixed"
            ),
        }
    summary["row2087"] = row2087_info
    summary["notes"] = [
        "Delta is candidate minus FULL; positive means the candidate is worse.",
        "Sets are fixed from FULL |error| only; candidate outcomes are not used to define them.",
        "gross_gain=sum max(-delta_SE,0), gross_loss=sum max(delta_SE,0); delta_SSE=gross_loss-gross_gain.",
        "SSE/SE contributions are not percentages of the non-linear RMSE difference.",
        "The top5 |delta_SE| table is a contribution-ranking display, not an evaluation subset.",
        "Excluding a set is diagnostic only; it is not a new primary metric.",
    ]
    F.write_json(F.ERRORS / "summary.json", summary)
    return {"summary": summary, "decomposition": decomposition, "hard_sets": hard_sets, "row2087": row2087_info}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    result = analyse()
    s = result["summary"]
    print("row2087:", s["row2087"])
    for variant, item in s["variants"].items():
        print(
            variant,
            "dRMSE", round(item["mean_delta_rmse"], 6),
            "dMAE", round(item["mean_delta_mae"], 6),
            "AE improved/worse", round(item["mean_n_ae_improved"], 1), round(item["mean_n_ae_worsened"], 1),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
