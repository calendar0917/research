"""One-shot Locked1k read for COC-v1 after a Monitor PASS and hash freeze.

The script refuses to run unless:
  * ``analysis/monitor_freeze.json`` exists and says ``monitor_pass=true``;
  * the current config, protocol, code, COC checkpoints, L checkpoints, Monitor
    metrics, Monitor predictions and Monitor group table all match their frozen
    hashes;
  * no previous Locked read sentinel exists.

It then reads Locked labels once, scores the pre-registered Locked gate, and
writes a new sentinel.  Official valid/test are never touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import coc_common as CC  # noqa: E402
import coc_model as CM  # noqa: E402
import ad_common as C  # noqa: E402

SENTINEL = CC.ANALYSIS / "locked_read.json"
_RUN_FILES = {
    "train_json": "train.json",
    "best_monitor": "best_monitor.pt",
    "final": "final.pt",
    "predictions_monitor": "predictions_monitor.pt",
}


def _ae(pred, y):
    return np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))


def _gmae(ae, mask):
    m = np.asarray(mask, dtype=bool)
    return float(ae[m].mean()) if m.any() else float("nan")


def _verify_freeze(freeze: dict, device) -> dict:
    checks = {
        "config_sha256": CC.sha256_file(CC.CFG_PATH) == freeze["config_sha256"],
        "protocol_sha256": CC.sha256_file(CC.ROUND / "PROTOCOL.md") == freeze["protocol_sha256"],
        "code_sha256": all(
            CC.sha256_file(_HERE / name) == digest
            for name, digest in freeze["code_sha256"].items()
        ),
        "run_sha256": all(
            CC.sha256_file(CC.RUNS / f"COC_fit8k_s{seed}" / _RUN_FILES[name]) == digest
            for seed in (0, 1)
            for name, digest in freeze["run_sha256"][f"seed{seed}"].items()
        ),
        "l_reference_sha256": all(
            CC.sha256_file(CC.ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" /
                           f"fit8k_s{s}" / f"{tag}.pt") == digest
            for s in (0, 1) for tag in ("best_monitor", "final")
            for digest in [freeze["l_reference_sha256"][f"seed{s}_{tag}"]]
        ),
        "monitor_metrics_sha256": CC.sha256_file(CC.ANALYSIS / "monitor_metrics.json") == freeze["monitor_metrics_sha256"],
        "monitor_decision_sha256": CC.sha256_file(CC.ANALYSIS / "monitor_decision.json") == freeze["monitor_decision_sha256"],
        "monitor_predictions_sha256": CC.sha256_file(CC.ANALYSIS / "monitor_predictions.npz") == freeze["monitor_predictions_sha256"],
        "monitor_group_table_sha256": CC.sha256_file(CC.ANALYSIS / "monitor_group_table.csv") == freeze["monitor_group_table_sha256"],
    }
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-reread", action="store_true")
    p.add_argument("--reason", default="")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    cfg = CC.load_cfg()
    if SENTINEL.exists() and not args.allow_reread:
        print(f"refusing second Locked read: {SENTINEL} exists")
        return 2
    if args.allow_reread and not args.reason:
        print("--allow-reread requires --reason")
        return 2

    freeze_path = CC.ANALYSIS / "monitor_freeze.json"
    decision_path = CC.ANALYSIS / "monitor_decision.json"
    if not freeze_path.exists() or not decision_path.exists():
        print("missing monitor freeze/decision; Locked is not authorized")
        return 2
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not freeze.get("monitor_pass") or not decision.get("monitor_pass"):
        print("Monitor did not pass; Locked is not authorized")
        return 2
    hash_checks = _verify_freeze(freeze, device)
    if not all(hash_checks.values()):
        print(json.dumps({"freeze_hash_checks": hash_checks}, indent=2))
        print("freeze hash mismatch; refusing Locked")
        return 2

    split = CC.split_rows()
    fit_rows = split["fit"]
    monitor_rows = split["monitor"]
    locked_rows = split["locked"]
    cache = CC.load_train_cache()
    monitor_metrics = json.loads((CC.ANALYSIS / "monitor_metrics.json").read_text(encoding="utf-8"))
    # Rebuild the Fit8k support table, never adding Locked structure to it.
    _, per_atom = CC.S.build_signatures(cache)
    support = CC.S.build_support(cache, fit_rows, per_atom)
    if support["meta"]["df_sha256"] != monitor_metrics["support_meta"]["df_sha256"]:
        raise RuntimeError("support table hash changed between Monitor and Locked")

    # Write an authorization sentinel before the only Locked label read.
    sentinel = {
        "protocol_id": cfg["protocol_id"],
        "status": "authorized",
        "utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "monitor_pass": True,
        "monitor_decision_sha256": CC.sha256_file(decision_path),
        "monitor_freeze_sha256": CC.sha256_file(freeze_path),
        "locked_rows_sha256": CC.sha256_array(locked_rows),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    CC.write_json(SENTINEL, sentinel)

    # ---- the one and only Locked label read -------------------------------
    locked = CC.prepare_locked_groups(cache, support, locked_rows)
    tiers = Counter(locked["tier"].tolist())
    if dict(tiers) != CC.S.LOCKED_TIER_COUNTS:
        raise RuntimeError(f"Locked tier counts changed: {dict(tiers)}")
    if int(locked["counts"]["primary"]) != CC.S.PHASE1_PRIMARY_N:
        raise RuntimeError(f"Locked primary count changed: {locked['counts']['primary']}")
    y = np.asarray(locked["y"], dtype=np.float64)

    preds: dict[str, np.ndarray] = {}
    aes: dict[str, np.ndarray] = {}
    seed_rec: dict = {}
    for seed in (0, 1):
        run = CC.RUNS / f"COC_fit8k_s{seed}"
        coc_info = json.loads((run / "train.json").read_text(encoding="utf-8"))
        norm = CC.norm_from_train_json(coc_info)
        rec = {"checkpoints": {}, "fit8k_norm_sha256": CC.sha256_array(
            np.concatenate([np.asarray(norm["global_mean"], dtype=np.float64),
                            np.asarray(norm["global_scale"], dtype=np.float64)]))}
        for tag in ("best_monitor", "final"):
            coc_ck = torch.load(run / f"{tag}.pt", map_location="cpu", weights_only=False)
            coc = CM.load_model_from_state(norm, coc_ck["state_dict"], device)
            coc_pred, _, coc_mae = CC.eval_rows(coc, cache, locked_rows, device=device, batch_size=256)
            l_model, l_ck, l_ck_path = CC.load_frozen_l_model(seed, tag, device)
            l_pred, _, l_mae = CC.eval_rows(l_model, cache, locked_rows, device=device, batch_size=256)
            saved = torch.load(CC.ROUND.parent / "zinc_dccl_bottleneck_v1" / "runs" /
                               f"fit8k_s{seed}" / f"predictions_{tag}.pt",
                               map_location="cpu", weights_only=False)
            if not np.array_equal(np.asarray(saved["locked_rows"]), locked_rows):
                raise RuntimeError(f"L seed {seed} {tag}: Locked row order mismatch")
            l_saved = np.asarray(saved["locked_pred"], dtype=np.float64)
            l_saved_diff = float(np.max(np.abs(l_pred.astype(np.float64) - l_saved)))
            if l_saved_diff > 1e-5:
                raise RuntimeError(f"L seed {seed} {tag}: Locked reproduction diff {l_saved_diff}")
            ae_coc = _ae(coc_pred, y)
            ae_l = _ae(l_pred, y)
            preds[f"coc_s{seed}_{tag}"] = coc_pred
            preds[f"l_s{seed}_{tag}"] = l_pred
            aes[f"coc_s{seed}_{tag}"] = ae_coc
            aes[f"l_s{seed}_{tag}"] = ae_l
            rec["checkpoints"][tag] = {
                "coc_checkpoint_sha256": CC.sha256_file(run / f"{tag}.pt"),
                "coc_epoch": int(coc_ck.get("epoch", -1)),
                "coc_step": int(coc_ck.get("step", -1)),
                "coc_locked_mae": float(coc_mae),
                "l_checkpoint_sha256": CC.sha256_file(l_ck_path),
                "l_locked_mae": float(l_mae),
                "l_saved_prediction_max_diff": float(l_saved_diff),
                "delta_locked_mae": float(coc_mae - l_mae),
                "coc_prediction_sha256": CC.sha256_array(coc_pred),
                "l_prediction_sha256": CC.sha256_array(l_pred),
                "all_finite": bool(np.isfinite(coc_pred).all() and np.isfinite(l_pred).all()),
            }
        seed_rec[str(seed)] = rec

    # Groups and paired deltas on the pre-registered Locked rows.
    group_masks = {"overall": np.ones(len(y), dtype=bool), **locked["masks"]}
    gt = {}
    for name, mask in group_masks.items():
        rec = {}
        for seed in (0, 1):
            c = _gmae(aes[f"coc_s{seed}_best_monitor"], mask)
            l = _gmae(aes[f"l_s{seed}_best_monitor"], mask)
            rec[str(seed)] = {"n": int(np.asarray(mask, dtype=bool).sum()),
                              "coc_mae": c, "l_mae": l, "delta": c - l}
        gt[name] = rec
    d = {
        str(seed): {
            "best_monitor": aes[f"coc_s{seed}_best_monitor"] - aes[f"l_s{seed}_best_monitor"],
            "final": aes[f"coc_s{seed}_final"] - aes[f"l_s{seed}_final"],
        } for seed in (0, 1)
    }
    d_best = {str(s): float(d[str(s)]["best_monitor"].mean()) for s in (0, 1)}
    d_final = {str(s): float(d[str(s)]["final"].mean()) for s in (0, 1)}
    d_mean = 0.5 * (d["0"]["best_monitor"] + d["1"]["best_monitor"])
    boot = {
        "seed0_best_monitor": CC.bootstrap_mean(d["0"]["best_monitor"], n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
        "seed1_best_monitor": CC.bootstrap_mean(d["1"]["best_monitor"], n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
        "seed_mean_best_monitor": CC.bootstrap_mean(d_mean, n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
        "cross_seed_difference_s1_minus_s0": CC.bootstrap_mean(
            d["1"]["best_monitor"] - d["0"]["best_monitor"],
            n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"]),
    }

    def gdelta(seed, mask):
        return _gmae(aes[f"coc_s{seed}_best_monitor"], mask) - _gmae(aes[f"l_s{seed}_best_monitor"], mask)

    primary = locked["masks"]["primary"]
    ordinary = locked["masks"]["ordinary_seen"]
    penalty = locked["masks"]["penalty_all"]
    primary_delta = {str(s): gdelta(s, primary) for s in (0, 1)}
    ordinary_delta = {str(s): gdelta(s, ordinary) for s in (0, 1)}

    # Fixed drop rules.  The worst-10 is defined by the frozen L best-monitor
    # absolute error, never by COC or by the delta.
    l_best_ae = 0.5 * (aes["l_s0_best_monitor"] + aes["l_s1_best_monitor"])
    worst10 = np.argsort(-l_best_ae)[:10]
    keep10 = np.ones(len(y), dtype=bool)
    keep10[worst10] = False
    drop10_delta = {str(s): float(d[str(s)]["best_monitor"][keep10].mean()) for s in (0, 1)}
    drop10_boot = CC.bootstrap_mean(
        0.5 * (d["0"]["best_monitor"][keep10] + d["1"]["best_monitor"][keep10]),
        n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"])
    penalty_idx = np.where(penalty)[0]
    # The single most influential penalty row is chosen by seed-mean |delta AE|.
    abs_mean_delta = np.abs(0.5 * (d["0"]["best_monitor"] + d["1"]["best_monitor"]))
    top1_pen = int(penalty_idx[np.argmax(abs_mean_delta[penalty_idx])]) if len(penalty_idx) else None
    keep_top1 = np.ones(len(y), dtype=bool)
    if top1_pen is not None:
        keep_top1[top1_pen] = False
    keep_nopen = ~penalty
    drop_all_pen = {str(s): float(d[str(s)]["best_monitor"][keep_nopen].mean()) for s in (0, 1)}
    drop_top1_pen = {str(s): float(d[str(s)]["best_monitor"][keep_top1].mean()) for s in (0, 1)}
    drop_all_boot = CC.bootstrap_mean(
        0.5 * (d["0"]["best_monitor"][keep_nopen] + d["1"]["best_monitor"][keep_nopen]),
        n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"])

    # Prediction-change fraction and post-training difference representation.
    eps = float(cfg["locked_gates"]["prediction_change_eps"])
    pred_change = {
        str(seed): np.abs(preds[f"coc_s{seed}_best_monitor"]
                          - preds[f"l_s{seed}_best_monitor"]) > eps
        for seed in (0, 1)
    }
    pred_change_frac = {str(seed): float(np.mean(pred_change[str(seed)]))
                        for seed in (0, 1)}
    pred_change_magnitude = {
        str(seed): {
            "mean_abs_delta_prediction": float(np.mean(np.abs(
                preds[f"coc_s{seed}_best_monitor"] - preds[f"l_s{seed}_best_monitor"]))),
            "median_abs_delta_prediction": float(np.median(np.abs(
                preds[f"coc_s{seed}_best_monitor"] - preds[f"l_s{seed}_best_monitor"]))),
            "fraction_changed": pred_change_frac[str(seed)],
            "eps": eps,
        } for seed in (0, 1)
    }
    rep_diag = {}
    rep_gate_by_seed = {}
    for seed in (0, 1):
        info = json.loads((CC.RUNS / f"COC_fit8k_s{seed}" / "train.json").read_text(encoding="utf-8"))
        ck = torch.load(CC.RUNS / f"COC_fit8k_s{seed}" / "best_monitor.pt",
                        map_location="cpu", weights_only=False)
        coc_best = CM.load_model_from_state(CC.norm_from_train_json(info), ck["state_dict"], device)
        cap = CC.representation_capture(coc_best, cache, locked_rows, device=device)
        rep_diag[str(seed)] = CC.difference_diagnostics(cap)
        rep_gate_by_seed[str(seed)] = CC.representation_gate(
            rep_diag[str(seed)], cfg, post_training=True)
    rep_gate = {
        "pass": bool(all(v["pass"] for v in rep_gate_by_seed.values())),
        "seeds": rep_gate_by_seed,
    }

    # W20/T100 are curve semantics from the Monitor stage; final is the Locked
    # read of the final checkpoint.  Keep them separate in the report.
    monitor_metrics = json.loads((CC.ANALYSIS / "monitor_metrics.json").read_text(encoding="utf-8"))
    w20_delta = monitor_metrics["summary"]["w20_delta"]
    t100_delta = monitor_metrics["summary"]["t100_delta"]

    def mean2(x):
        return float(0.5 * (x["0"] + x["1"]))

    gates = {
        "L1_overall": {
            "pass": bool(d_best["0"] < 0 and d_best["1"] < 0
                         and mean2(d_best) <= cfg["locked_gates"]["overall_seed_mean_delta_max"]),
            "delta": d_best, "seed_mean": mean2(d_best),
            "threshold": cfg["locked_gates"]["overall_seed_mean_delta_max"],
        },
        "L2_nopen_any_unseen": {
            "pass": bool(primary_delta["0"] < 0 and primary_delta["1"] < 0
                         and mean2(primary_delta) <= cfg["locked_gates"]["nopen_any_unseen_seed_mean_delta_max"]),
            "delta": primary_delta, "seed_mean": mean2(primary_delta),
            "threshold": cfg["locked_gates"]["nopen_any_unseen_seed_mean_delta_max"],
            "n": int(primary.sum()),
        },
        "L3_ordinary_not_worse": {
            "pass": bool(mean2(ordinary_delta) <= cfg["locked_gates"]["ordinary_seed_mean_delta_max"]
                         and ordinary_delta["0"] <= cfg["locked_gates"]["ordinary_each_seed_delta_max"]
                         and ordinary_delta["1"] <= cfg["locked_gates"]["ordinary_each_seed_delta_max"]),
            "delta": ordinary_delta, "seed_mean": mean2(ordinary_delta),
            "threshold_mean": cfg["locked_gates"]["ordinary_seed_mean_delta_max"],
            "threshold_each": cfg["locked_gates"]["ordinary_each_seed_delta_max"],
            "n": int(ordinary.sum()),
        },
        "L4_drop_worst10": {
            "pass": bool(drop10_delta["0"] < 0 and drop10_delta["1"] < 0
                         and mean2(drop10_delta) <= cfg["locked_gates"]["drop_worst10_seed_mean_delta_max"]),
            "delta": drop10_delta, "seed_mean": mean2(drop10_delta),
            "threshold": cfg["locked_gates"]["drop_worst10_seed_mean_delta_max"],
            "removed_rows": worst10.tolist(),
            "removed_l_best_ae": l_best_ae[worst10].tolist(),
            "bootstrap": drop10_boot,
        },
        "L5_not_one_penalty_row": {
            "pass": bool(
                drop_all_pen["0"] < 0 and drop_all_pen["1"] < 0
                and mean2(drop_all_pen) <= cfg["locked_gates"]["drop_all_penalty_seed_mean_delta_max"]
                and mean2(drop_top1_pen) <= cfg["locked_gates"]["drop_top1_penalty_seed_mean_delta_max"]
            ),
            "drop_all_penalty_delta": drop_all_pen,
            "drop_all_penalty_seed_mean": mean2(drop_all_pen),
            "drop_top1_penalty_delta": drop_top1_pen,
            "drop_top1_penalty_seed_mean": mean2(drop_top1_pen),
            "top1_penalty_row": top1_pen,
            "n_penalty": int(penalty.sum()),
            "bootstrap_drop_all_penalty": drop_all_boot,
        },
        "L6_mechanism_support": {
            "pass": bool(
                mean2(t100_delta) <= cfg["locked_gates"]["t100_seed_mean_delta_max"]
                and t100_delta["0"] <= cfg["locked_gates"]["t100_each_seed_delta_max"]
                and t100_delta["1"] <= cfg["locked_gates"]["t100_each_seed_delta_max"]
                and mean2(d_final) <= cfg["locked_gates"]["final_seed_mean_delta_max"]
                and d_final["0"] <= cfg["locked_gates"]["final_each_seed_delta_max"]
                and d_final["1"] <= cfg["locked_gates"]["final_each_seed_delta_max"]
                and min(pred_change_frac.values()) >= cfg["locked_gates"]["prediction_change_fraction_min"]
                and rep_gate["pass"]
            ),
            "t100_delta": t100_delta,
            "t100_seed_mean": mean2(t100_delta),
            "final_delta": d_final,
            "final_seed_mean": mean2(d_final),
            "prediction_change": pred_change_magnitude,
            "prediction_change_fraction": pred_change_frac,
            "representation_gate": rep_gate,
            "representation_diagnostics": rep_diag,
        },
    }
    locked_pass = bool(all(g["pass"] for g in gates.values()))
    summary = {
        "overall_best_monitor_delta": d_best,
        "overall_best_monitor_delta_seed_mean": mean2(d_best),
        "overall_final_delta": d_final,
        "overall_final_delta_seed_mean": mean2(d_final),
        "nopen_any_unseen_delta": primary_delta,
        "nopen_any_unseen_delta_seed_mean": mean2(primary_delta),
        "nopen_any_unseen_n": int(primary.sum()),
        "ordinary_seen_delta": ordinary_delta,
        "ordinary_seen_delta_seed_mean": mean2(ordinary_delta),
        "drop_worst10_delta": drop10_delta,
        "drop_worst10_delta_seed_mean": mean2(drop10_delta),
        "drop_all_penalty_delta": drop_all_pen,
        "drop_all_penalty_delta_seed_mean": mean2(drop_all_pen),
        "drop_top1_penalty_delta": drop_top1_pen,
        "drop_top1_penalty_delta_seed_mean": mean2(drop_top1_pen),
        "w20_delta_monitor_curve": w20_delta,
        "t100_delta_monitor_curve": t100_delta,
        "prediction_change": pred_change_magnitude,
        "prediction_change_fraction": pred_change_frac,
        "representation_diagnostics": rep_diag,
    }
    out = {
        "protocol_id": cfg["protocol_id"],
        "locked_pass": locked_pass,
        "verdict": "LOCKED_PASS_FORM_FULL10K_PROPOSAL" if locked_pass else "STOP_LOCKED_NOT_CONFIRMED",
        "summary": summary,
        "gates": gates,
        "bootstrap": boot,
        "group_table": gt,
        "seed_runs": seed_rec,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": True,
        "locked_rows_sha256": CC.sha256_array(locked_rows),
        "monitor_decision_sha256": CC.sha256_file(decision_path),
        "monitor_freeze_sha256": CC.sha256_file(freeze_path),
        "support_meta": support["meta"],
    }
    CC.write_json(CC.ANALYSIS / "locked_metrics.json", CC.jsonable(out))

    # CSV group table.
    gcsv = CC.ANALYSIS / "locked_group_table.csv"
    with open(gcsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n", "seed0_coc", "seed0_l", "seed0_delta",
                    "seed1_coc", "seed1_l", "seed1_delta", "seed_mean_delta"])
        for name in ("overall", *locked["masks"].keys()):
            r = gt[name]
            n = r["0"]["n"]
            c0, l0 = r["0"]["coc_mae"], r["0"]["l_mae"]
            c1, l1 = r["1"]["coc_mae"], r["1"]["l_mae"]
            w.writerow([name, n, c0, l0, c0 - l0, c1, l1, c1 - l1,
                        0.5 * ((c0 - l0) + (c1 - l1))])

    npz = { "rows": locked_rows, "y": y }
    npz.update({f"pred_{k}": v for k, v in preds.items()})
    npz.update({f"ae_{k}": v for k, v in aes.items()})
    for name, mask in locked["masks"].items():
        npz[f"mask_{name}"] = np.asarray(mask, dtype=bool)
    npz["mask_overall"] = np.ones(len(y), dtype=bool)
    np.savez_compressed(CC.ANALYSIS / "locked_predictions.npz", **npz)

    decision_out = {
        "protocol_id": cfg["protocol_id"],
        "locked_pass": locked_pass,
        "verdict": out["verdict"],
        "summary": summary,
        "gates": gates,
        "bootstrap": boot,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_labels_read": True,
        "locked_metrics_sha256": CC.sha256_file(CC.ANALYSIS / "locked_metrics.json"),
        "locked_predictions_sha256": CC.sha256_file(CC.ANALYSIS / "locked_predictions.npz"),
        "locked_group_table_sha256": CC.sha256_file(gcsv),
    }
    CC.write_json(CC.ANALYSIS / "locked_decision.json", CC.jsonable(decision_out))
    sentinel.update({
        "status": "completed",
        "locked_pass": locked_pass,
        "verdict": out["verdict"],
        "completed_utc": C.utcnow(),
        "locked_metrics_sha256": CC.sha256_file(CC.ANALYSIS / "locked_metrics.json"),
        "locked_decision_sha256": CC.sha256_file(CC.ANALYSIS / "locked_decision.json"),
        "locked_predictions_sha256": CC.sha256_file(CC.ANALYSIS / "locked_predictions.npz"),
        "locked_group_table_sha256": CC.sha256_file(gcsv),
    })
    CC.write_json(SENTINEL, sentinel)
    C.log(json.dumps({
        "locked_pass": locked_pass,
        "verdict": out["verdict"],
        "summary": summary,
        "gates": {k: v["pass"] for k, v in gates.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
