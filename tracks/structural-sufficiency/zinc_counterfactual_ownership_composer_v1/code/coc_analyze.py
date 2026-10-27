"""Monitor-stage scoring and pre-registered COC-v1 gates.

This is the only script that reads Monitor labels for the decision.  It does not
read Locked labels and never constructs official valid/test.  It freezes the
Monitor predictions and all hashes before any Locked access is possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import coc_common as CC  # noqa: E402
import coc_model as CM  # noqa: E402
import ad_common as C  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coc_run(seed: int) -> Path:
    return CC.RUNS / f"COC_fit8k_s{int(seed)}"


def _state_from_ckpt(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck


def _ae(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))


def _group_mae(ae: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(ae[m].mean()) if m.any() else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(CC.ANALYSIS / "monitor_metrics.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    cfg = CC.load_cfg()
    split = CC.split_rows()
    fit_rows = split["fit"]
    monitor_rows = split["monitor"]
    cache = CC.load_train_cache()
    support, monitor = CC.prepare_monitor_groups(cache, fit_rows, monitor_rows)
    y = np.asarray(monitor["y"], dtype=np.float64)

    rec: dict = {"protocol_id": cfg["protocol_id"], "seed_runs": {}, "curves": {}}
    preds: dict[str, np.ndarray] = {}
    aes: dict[str, np.ndarray] = {}

    for seed in (0, 1):
        run = _coc_run(seed)
        if not (run / "train.json").exists():
            raise FileNotFoundError(f"missing COC run: {run}")
        coc_info = _load_json(run / "train.json")
        l_info = CC.load_frozen_train_json(seed)
        if coc_info["initial_state_sha256"] != cfg["frozen_l_reference"]["initial_state_sha256"][str(seed)]:
            raise RuntimeError(f"COC seed {seed} init hash mismatch")
        if coc_info["split_hashes"] != l_info["split_hashes"]:
            raise RuntimeError(f"COC seed {seed} split hash mismatch vs L")
        if int(coc_info["total_updates_completed"]) != 47400:
            raise RuntimeError(f"COC seed {seed} incomplete: {coc_info['total_updates_completed']}")

        norm = CC.norm_from_train_json(coc_info)
        seed_rec = {
            "coc_train_json_sha256": CC.sha256_file(run / "train.json"),
            "coc_parameters": int(coc_info["parameters"]),
            "coc_initial_state_sha256": coc_info["initial_state_sha256"],
            "coc_first_batch_rows_sha256": coc_info["first_batch_rows_sha256"],
            "coc_gradient_audit": coc_info["gradient_audit"],
            "l_train_json_sha256": CC.sha256_file(
                CC.ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{seed}" / "train.json"),
            "l_split_hashes": l_info["split_hashes"],
            "checkpoints": {},
        }
        for tag in ("best_monitor", "final"):
            coc_ck_path = run / f"{tag}.pt"
            coc_ck = _state_from_ckpt(coc_ck_path)
            coc = CM.load_model_from_state(norm, coc_ck["state_dict"], device)
            coc_pred, _, coc_mae = CC.eval_rows(
                coc, cache, monitor_rows, device=device, batch_size=256)
            l_model, l_ck, l_ck_path = CC.load_frozen_l_model(seed, tag, device)
            l_pred, _, l_mae = CC.eval_rows(
                l_model, cache, monitor_rows, device=device, batch_size=256)
            saved = torch.load(CC.ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" /
                               f"fit8k_s{seed}" / f"predictions_{tag}.pt",
                               map_location="cpu", weights_only=False)
            if not np.array_equal(np.asarray(saved["monitor_rows"]), monitor_rows):
                raise RuntimeError(f"L seed {seed} {tag}: monitor row order mismatch")
            l_saved = np.asarray(saved["monitor_pred"], dtype=np.float64)
            l_saved_diff = float(np.max(np.abs(l_pred.astype(np.float64) - l_saved)))
            l_stored_mae = float(l_info["eval_reads"][tag]["monitor_mae"])
            if l_saved_diff > 1e-5:
                raise RuntimeError(f"L seed {seed} {tag}: reproduction diff {l_saved_diff}")
            if abs(l_mae - l_stored_mae) > 1e-6:
                raise RuntimeError(f"L seed {seed} {tag}: stored monitor MAE mismatch {l_mae} vs {l_stored_mae}")
            ae_coc = _ae(coc_pred, y)
            ae_l = _ae(l_pred, y)
            preds[f"coc_s{seed}_{tag}"] = coc_pred.astype(np.float64)
            preds[f"l_s{seed}_{tag}"] = l_pred.astype(np.float64)
            aes[f"coc_s{seed}_{tag}"] = ae_coc
            aes[f"l_s{seed}_{tag}"] = ae_l
            seed_rec["checkpoints"][tag] = {
                "coc_checkpoint_sha256": CC.sha256_file(coc_ck_path),
                "coc_epoch": int(coc_ck.get("epoch", -1)),
                "coc_step": int(coc_ck.get("step", -1)),
                "coc_monitor_mae": float(coc_mae),
                "coc_selection_checkpoint_mae": float(coc_ck.get("monitor_mae", np.nan)),
                "l_checkpoint_sha256": CC.sha256_file(l_ck_path),
                "l_epoch": int(l_ck.get("epoch", -1)),
                "l_monitor_mae": float(l_mae),
                "l_stored_monitor_mae": float(l_stored_mae),
                "l_saved_prediction_max_diff": float(l_saved_diff),
                "delta_monitor_mae": float(coc_mae - l_mae),
                "coc_prediction_sha256": CC.sha256_array(coc_pred),
                "l_prediction_sha256": CC.sha256_array(l_pred),
                "all_finite": bool(np.isfinite(coc_pred).all() and np.isfinite(l_pred).all()),
            }
        # Fit8k train MAE for the stability gate (labels are Fit8k labels).
        coc_best = CM.load_model_from_state(norm, _state_from_ckpt(run / "best_monitor.pt")["state_dict"], device)
        coc_final = CM.load_model_from_state(norm, _state_from_ckpt(run / "final.pt")["state_dict"], device)
        _, _, coc_fit_best = CC.eval_rows(coc_best, cache, fit_rows, device=device, batch_size=256)
        _, _, coc_fit_final = CC.eval_rows(coc_final, cache, fit_rows, device=device, batch_size=256)
        l_best, _, _ = CC.load_frozen_l_model(seed, "best_monitor", device)
        l_final, _, _ = CC.load_frozen_l_model(seed, "final", device)
        _, _, l_fit_best = CC.eval_rows(l_best, cache, fit_rows, device=device, batch_size=256)
        _, _, l_fit_final = CC.eval_rows(l_final, cache, fit_rows, device=device, batch_size=256)
        seed_rec["fit8k_train_mae"] = {
            "coc_best_monitor": float(coc_fit_best),
            "coc_final": float(coc_fit_final),
            "l_best_monitor": float(l_fit_best),
            "l_final": float(l_fit_final),
            "coc_over_l_best_ratio": float(coc_fit_best / l_fit_best) if l_fit_best > 0 else float("inf"),
            "coc_over_l_final_ratio": float(coc_fit_final / l_fit_final) if l_fit_final > 0 else float("inf"),
        }
        coc_curve = CC.w20_t100(coc_info["curve"])
        l_curve = CC.w20_t100(l_info["curve"])
        seed_rec["curve"] = {"coc": coc_curve, "l": l_curve}
        seed_rec["curve_semantics"] = {
            "W20": "minimum over 20-consecutive-epoch windows of the window median Monitor MAE",
            "T100": "median Monitor MAE over the final 100 epochs",
        }
        # Group table and deltas.
        groups = {"overall": np.ones(len(y), dtype=bool), **monitor["masks"]}
        gt = {}
        for name, mask in groups.items():
            ae_c = aes[f"coc_s{seed}_best_monitor"]
            ae_l = aes[f"l_s{seed}_best_monitor"]
            gt[name] = {
                "n": int(np.asarray(mask, dtype=bool).sum()),
                "coc_mae": _group_mae(ae_c, mask),
                "l_mae": _group_mae(ae_l, mask),
                "delta": _group_mae(ae_c, mask) - _group_mae(ae_l, mask),
            }
        seed_rec["groups_best_monitor"] = gt
        seed_rec["natural_overall"] = {
            "coc": CC.S.natural_overall(preds[f"coc_s{seed}_best_monitor"], monitor),
            "l": CC.S.natural_overall(preds[f"l_s{seed}_best_monitor"], monitor),
        }
        seed_rec["natural_overall"]["delta"] = (
            seed_rec["natural_overall"]["coc"] - seed_rec["natural_overall"]["l"])
        rec["seed_runs"][str(seed)] = seed_rec

    # Paired deltas, bootstrap and gate computation.
    deltas = {}
    for seed in (0, 1):
        d_best = aes[f"coc_s{seed}_best_monitor"] - aes[f"l_s{seed}_best_monitor"]
        d_final = aes[f"coc_s{seed}_final"] - aes[f"l_s{seed}_final"]
        deltas[str(seed)] = {"best_monitor": d_best, "final": d_final}
    d_mean = 0.5 * (deltas["0"]["best_monitor"] + deltas["1"]["best_monitor"])
    boot = {
        "seed0_best_monitor": CC.bootstrap_mean(deltas["0"]["best_monitor"], n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
        "seed1_best_monitor": CC.bootstrap_mean(deltas["1"]["best_monitor"], n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
        "seed_mean_best_monitor": CC.bootstrap_mean(d_mean, n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
        "cross_seed_difference_s1_minus_s0": CC.bootstrap_mean(
            deltas["1"]["best_monitor"] - deltas["0"]["best_monitor"],
            n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
    }

    def group_delta(seed: int, mask: np.ndarray) -> float:
        ae_c = aes[f"coc_s{seed}_best_monitor"]
        ae_l = aes[f"l_s{seed}_best_monitor"]
        return _group_mae(ae_c, mask) - _group_mae(ae_l, mask)

    primary_mask = monitor["masks"]["primary"]
    nopen_mask = monitor["masks"]["nopen"]
    ordinary_mask = monitor["masks"]["ordinary_seen"]
    overall_delta = {str(s): float(deltas[str(s)]["best_monitor"].mean()) for s in (0, 1)}
    primary_delta = {str(s): group_delta(s, primary_mask) for s in (0, 1)}
    nopen_delta = {str(s): group_delta(s, nopen_mask) for s in (0, 1)}
    ordinary_delta = {str(s): group_delta(s, ordinary_mask) for s in (0, 1)}
    w20_delta = {str(s): float(rec["seed_runs"][str(s)]["curve"]["coc"]["w20"]
                              - rec["seed_runs"][str(s)]["curve"]["l"]["w20"]) for s in (0, 1)}
    t100_delta = {str(s): float(rec["seed_runs"][str(s)]["curve"]["coc"]["t100"]
                               - rec["seed_runs"][str(s)]["curve"]["l"]["t100"]) for s in (0, 1)}
    final_delta = {str(s): float(deltas[str(s)]["final"].mean()) for s in (0, 1)}
    fit_ratio_best = {str(s): float(rec["seed_runs"][str(s)]["fit8k_train_mae"]["coc_over_l_best_ratio"]) for s in (0, 1)}
    fit_ratio_final = {str(s): float(rec["seed_runs"][str(s)]["fit8k_train_mae"]["coc_over_l_final_ratio"]) for s in (0, 1)}
    monitor_tail_ratio = {
        str(s): float(rec["seed_runs"][str(s)]["curve"]["coc"]["t100"]
                      / max(rec["seed_runs"][str(s)]["curve"]["coc"]["best"], 1e-12))
        for s in (0, 1)
    }

    def mean2(d):
        return float(0.5 * (d["0"] + d["1"]))

    gates = {
        "G1_both_seed_overall_negative": {
            "pass": bool(overall_delta["0"] < 0 and overall_delta["1"] < 0),
            "values": overall_delta,
        },
        "G2_seed_mean_overall_at_most_minus_0.005": {
            "pass": bool(mean2(overall_delta) <= cfg["monitor_gates"]["seed_mean_overall_delta_max"]),
            "seed_mean": mean2(overall_delta),
            "threshold": cfg["monitor_gates"]["seed_mean_overall_delta_max"],
        },
        "G3_nopen_any_unseen": {
            "pass": bool(primary_delta["0"] < 0 and primary_delta["1"] < 0
                         and mean2(primary_delta) <= cfg["monitor_gates"]["nopen_any_unseen_seed_mean_delta_max"]),
            "values": primary_delta,
            "seed_mean": mean2(primary_delta),
            "threshold": cfg["monitor_gates"]["nopen_any_unseen_seed_mean_delta_max"],
            "n": int(primary_mask.sum()),
        },
        "G4_nopen_overall_improves": {
            "pass": bool(nopen_delta["0"] < 0 and nopen_delta["1"] < 0
                         and mean2(nopen_delta) <= cfg["monitor_gates"]["nopen_overall_seed_mean_delta_max"]),
            "values": nopen_delta,
            "seed_mean": mean2(nopen_delta),
            "threshold": cfg["monitor_gates"]["nopen_overall_seed_mean_delta_max"],
            "n": int(nopen_mask.sum()),
        },
        "G5_w20_t100_support": {
            "pass": bool(
                mean2(w20_delta) <= cfg["monitor_gates"]["w20_seed_mean_delta_max"]
                and mean2(t100_delta) <= cfg["monitor_gates"]["t100_seed_mean_delta_max"]
                and w20_delta["0"] <= cfg["monitor_gates"]["w20_each_seed_delta_max"]
                and w20_delta["1"] <= cfg["monitor_gates"]["w20_each_seed_delta_max"]
                and t100_delta["0"] <= cfg["monitor_gates"]["t100_each_seed_delta_max"]
                and t100_delta["1"] <= cfg["monitor_gates"]["t100_each_seed_delta_max"]
            ),
            "w20_delta": w20_delta, "t100_delta": t100_delta,
            "w20_seed_mean": mean2(w20_delta), "t100_seed_mean": mean2(t100_delta),
        },
        "G6_training_stability": {
            "pass": bool(
                all(rec["seed_runs"][str(s)]["checkpoints"][tag]["all_finite"]
                    for s in (0, 1) for tag in ("best_monitor", "final"))
                and max(fit_ratio_best.values()) <= cfg["monitor_gates"]["train_fit_ratio_cap_vs_L"]
                and max(fit_ratio_final.values()) <= cfg["monitor_gates"]["train_fit_ratio_cap_vs_L"]
                and max(monitor_tail_ratio.values()) <= cfg["monitor_gates"]["monitor_tail100_vs_best_cap"]
            ),
            "fit_ratio_best_vs_L": fit_ratio_best,
            "fit_ratio_final_vs_L": fit_ratio_final,
            "monitor_tail100_over_best": monitor_tail_ratio,
            "all_finite": all(rec["seed_runs"][str(s)]["checkpoints"][tag]["all_finite"]
                              for s in (0, 1) for tag in ("best_monitor", "final")),
        },
    }
    monitor_pass = bool(all(g["pass"] for g in gates.values()))
    summary = {
        "overall_best_monitor_delta": overall_delta,
        "overall_best_monitor_delta_seed_mean": mean2(overall_delta),
        "overall_final_delta": final_delta,
        "overall_final_delta_seed_mean": mean2(final_delta),
        "nopen_any_unseen_delta": primary_delta,
        "nopen_any_unseen_delta_seed_mean": mean2(primary_delta),
        "nopen_any_unseen_n": int(primary_mask.sum()),
        "nopen_delta": nopen_delta,
        "nopen_delta_seed_mean": mean2(nopen_delta),
        "ordinary_seen_delta": ordinary_delta,
        "ordinary_seen_delta_seed_mean": mean2(ordinary_delta),
        "w20_delta": w20_delta, "t100_delta": t100_delta,
        "w20_delta_seed_mean": mean2(w20_delta),
        "t100_delta_seed_mean": mean2(t100_delta),
    }
    rec.update({
        "summary": summary,
        "bootstrap": boot,
        "gates": gates,
        "monitor_pass": monitor_pass,
        "verdict": "MONITOR_PASS" if monitor_pass else "STOP_NO_MATERIAL_GAIN",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
        "monitor_rows_sha256": CC.sha256_array(monitor_rows),
        "monitor_groups": {k: int(v.sum()) for k, v in
                           ({"overall": np.ones(len(y), dtype=bool), **monitor["masks"]}).items()},
        "support_meta": support["meta"],
    })

    # CSV group table (seed-level and seed-mean deltas).
    gcsv = CC.ANALYSIS / "monitor_group_table.csv"
    CC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(gcsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n", "seed0_coc", "seed0_l", "seed0_delta",
                    "seed1_coc", "seed1_l", "seed1_delta", "seed_mean_delta"])
        for name in ("overall", "natural_overall", *monitor["masks"].keys()):
            if name == "natural_overall":
                c0 = rec["seed_runs"]["0"]["natural_overall"]["coc"]
                l0 = rec["seed_runs"]["0"]["natural_overall"]["l"]
                c1 = rec["seed_runs"]["1"]["natural_overall"]["coc"]
                l1 = rec["seed_runs"]["1"]["natural_overall"]["l"]
                n = len(y)
            else:
                mask = np.ones(len(y), dtype=bool) if name == "overall" else monitor["masks"][name]
                c0 = _group_mae(aes["coc_s0_best_monitor"], mask)
                l0 = _group_mae(aes["l_s0_best_monitor"], mask)
                c1 = _group_mae(aes["coc_s1_best_monitor"], mask)
                l1 = _group_mae(aes["l_s1_best_monitor"], mask)
                n = int(mask.sum())
            w.writerow([name, n, c0, l0, c0 - l0, c1, l1, c1 - l1,
                        0.5 * ((c0 - l0) + (c1 - l1))])

    npz_path = CC.ANALYSIS / "monitor_predictions.npz"
    arrays = {"rows": monitor_rows, "y": y}
    arrays.update({f"pred_{k}": v for k, v in preds.items()})
    arrays.update({f"ae_{k}": v for k, v in aes.items()})
    for name, mask in monitor["masks"].items():
        arrays[f"mask_{name}"] = np.asarray(mask, dtype=bool)
    arrays["mask_overall"] = np.ones(len(y), dtype=bool)
    np.savez_compressed(npz_path, **arrays)

    CC.write_json(Path(args.out), CC.jsonable(rec))
    decision = {
        "protocol_id": cfg["protocol_id"],
        "monitor_pass": bool(monitor_pass),
        "verdict": rec["verdict"],
        "summary": summary,
        "gates": gates,
        "bootstrap": boot,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
        "monitor_rows_sha256": rec["monitor_rows_sha256"],
        "monitor_metrics_sha256": CC.sha256_file(Path(args.out)),
        "monitor_predictions_sha256": CC.sha256_file(npz_path),
        "monitor_group_table_sha256": CC.sha256_file(gcsv),
    }
    CC.write_json(CC.ANALYSIS / "monitor_decision.json", CC.jsonable(decision))

    # Freeze only after the decision is written.  Locked is still untouched.
    freeze = {
        "protocol_id": cfg["protocol_id"],
        "monitor_pass": bool(monitor_pass),
        "verdict": rec["verdict"],
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_sha256": CC.sha256_file(CC.CFG_PATH),
        "protocol_sha256": CC.sha256_file(CC.ROUND / "PROTOCOL.md"),
        "code_sha256": {p.name: CC.sha256_file(p) for p in sorted(_HERE.glob("*.py"))},
        "run_sha256": {
            f"seed{s}": {
                "train_json": CC.sha256_file(_coc_run(s) / "train.json"),
                "best_monitor": CC.sha256_file(_coc_run(s) / "best_monitor.pt"),
                "final": CC.sha256_file(_coc_run(s) / "final.pt"),
                "predictions_monitor": CC.sha256_file(_coc_run(s) / "predictions_monitor.pt"),
            } for s in (0, 1)
        },
        "l_reference_sha256": {
            f"seed{s}_{tag}": CC.sha256_file(
                CC.ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{s}" / f"{tag}.pt")
            for s in (0, 1) for tag in ("best_monitor", "final")
        },
        "monitor_metrics_sha256": CC.sha256_file(Path(args.out)),
        "monitor_decision_sha256": CC.sha256_file(CC.ANALYSIS / "monitor_decision.json"),
        "monitor_predictions_sha256": CC.sha256_file(npz_path),
        "monitor_group_table_sha256": CC.sha256_file(gcsv),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": False,
        "note": "This freeze is written before any Locked access. coc_locked.py refuses to run unless monitor_pass is true and these hashes still match.",
    }
    CC.write_json(CC.ANALYSIS / "monitor_freeze.json", CC.jsonable(freeze))
    C.log(json.dumps({
        "monitor_pass": monitor_pass,
        "verdict": rec["verdict"],
        "summary": summary,
        "gates": {k: v["pass"] for k, v in gates.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
