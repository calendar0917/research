"""One frozen read of the Locked split for all Phase 1 arms.

This is the only Phase 1 code that touches Locked labels.  It is run once after
all training is complete.  It does not load official valid or official test.
The best-monitor checkpoint is the pre-registered primary readout; final is a
robustness readout.

A sentinel (``analysis/locked_read.json``) makes a second read refuse unless
``--allow-reread --reason`` is given explicitly; any reread is recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import scb_common as S
import scb_model as SM
import ad_common as C

SENTINEL = S.ANALYSIS / "locked_read.json"


def norm_from_train_json(info: dict) -> dict:
    norm = dict(info["norm"])
    for k in ("global_mean", "global_scale", "root_mean", "root_scale",
              "block_mean", "block_scale"):
        if k in norm and isinstance(norm[k], list):
            norm[k] = np.asarray(norm[k], dtype=np.float32)
    return norm


def alpha_locked_for_arm(cache, support, locked_rows, arm: str) -> np.ndarray | None:
    if arm == "L":
        return None
    alpha = S.alpha_for_rows(cache, support, locked_rows)
    if arm == "P":
        salt = int(S.load_cfg()["permutation"]["salt"])
        alpha = S.permute_alpha_within_molecules(cache, alpha, locked_rows, salt)
    return alpha


def ck_monitor(ck, train_json, tag):
    if "monitor_mae" in ck:
        return float(ck["monitor_mae"])
    if tag == "final" and train_json.get("curve"):
        return float(train_json["curve"][-1]["monitor_mae"])
    return None


def score_prediction(pred, group) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    y = group["y"]
    out = {
        "locked_mae": float(np.mean(np.abs(pred - y))),
        "natural_overall": S.natural_overall(pred, group),
        "groups": {},
    }
    for name, mask in group["masks"].items():
        out["groups"][name] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(pred - y)[mask])) if mask.any() else None,
        }
    return out


def score_scb_run(arm: str, seed: int, cache, support, locked, fit_rows, device) -> dict:
    run = S.RUNS / f"{arm}_fit8k_s{seed}"
    train_json = S.read_json(run / "train.json")
    norm = norm_from_train_json(train_json)
    model, _ = SM.build_scb(seed, norm)
    model.to(device)
    rec = {"arm": arm, "seed": int(seed), "fit": "fit8k",
           "train_json_sha256": S.sha256_file(run / "train.json"),
           "support_sha256_rebuilt": support["meta"]["df_sha256"],
           "alpha_fit_sha256_expected": train_json["alpha_sha256"]}
    alpha_locked = alpha_locked_for_arm(cache, support, locked["rows"], arm)
    alpha_locked_sha = S.sha256_array(alpha_locked) if alpha_locked is not None else None
    if arm == "L":
        alpha_fit = None
    else:
        alpha_fit = S.alpha_for_rows(cache, support, fit_rows, leave_one_out_for=fit_rows)
        if arm == "P":
            alpha_fit = S.permute_alpha_within_molecules(
                cache, alpha_fit, fit_rows, int(S.load_cfg()["permutation"]["salt"]))
    alpha_fit_sha = S.sha256_array(alpha_fit) if alpha_fit is not None else None
    if alpha_fit_sha != train_json.get("alpha_sha256"):
        raise RuntimeError(f"{arm} s{seed}: rebuilt fit alpha hash != train.json")
    reads = {}
    for tag, ck_name in (("best_monitor", "best_monitor.pt"), ("final", "final.pt")):
        ck = torch.load(run / ck_name, map_location="cpu", weights_only=False)
        if ck.get("alpha_sha256") != train_json.get("alpha_sha256"):
            raise RuntimeError(f"{arm} s{seed} {tag}: alpha hash mismatch")
        C.load_state(model, ck["state_dict"], device)
        model.eval()
        pred, _ = S.eval_with_alpha(model, cache, locked["rows"], alpha_locked, device)
        sc = score_prediction(pred, locked)
        _, fit_mae = S.eval_with_alpha(model, cache, fit_rows, alpha_fit, device)
        reads[tag] = {
            "checkpoint": ck_name,
            "checkpoint_sha256": S.sha256_file(run / ck_name),
            "epoch": int(ck["epoch"]),
            "step": int(ck["step"]),
            "monitor_mae": ck_monitor(ck, train_json, tag),
            "alpha_fit_sha256": ck.get("alpha_sha256"),
            "alpha_fit_sha256_rebuilt": alpha_fit_sha,
            "alpha_locked_sha256": alpha_locked_sha,
            "train_mae_at_best": float(fit_mae),
            "curve": S.w20_t100(train_json["curve"]),
            **sc,
        }
        # Persist the per-row predictions for paired bootstraps.
        C.atomic_torch_save(
            {"locked_pred": pred, "locked_rows": locked["rows"], "y": locked["y"],
             "arm": arm, "seed": int(seed), "fit": "fit8k", "checkpoint": tag,
             "alpha_locked_sha256": alpha_locked_sha},
            run / f"predictions_locked_{tag}.pt",
        )
    rec["reads"] = reads
    rec["parameters"] = int(train_json["parameters"])
    rec["official_valid_loaded"] = False
    rec["official_test_loaded"] = False
    rec["locked_read"] = True
    return rec


def score_l_reference(seed: int, cache, support, locked, fit_rows, device) -> dict:
    run = S.DCB_RUNS / f"fit8k_s{seed}"
    train_json = S.read_json(run / "train.json")
    norm = norm_from_train_json(train_json)
    model, _ = SM.build_scb(seed, norm)
    model.to(device)
    rec = {"arm": "L", "seed": int(seed), "fit": "fit8k",
           "train_json_sha256": S.sha256_file(run / "train.json"),
           "support_sha256_rebuilt": support["meta"]["df_sha256"],
           "alpha_fit_sha256_expected": None}
    reads = {}
    for tag, ck_name, pred_name in (
        ("best_monitor", "best_monitor.pt", "predictions_best_monitor.pt"),
        ("final", "final.pt", "predictions_final.pt"),
    ):
        ck = torch.load(run / ck_name, map_location="cpu", weights_only=False)
        saved = torch.load(run / pred_name, map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(saved["locked_rows"]), locked["rows"]):
            raise RuntimeError("L locked row order mismatch")
        C.load_state(model, ck["state_dict"], device)
        model.eval()
        pred, _ = S.eval_with_alpha(model, cache, locked["rows"], None, device)
        max_diff = float(np.max(np.abs(pred - np.asarray(saved["locked_pred"]))))
        if max_diff > 1e-5:
            raise RuntimeError(f"L reproduction mismatch {max_diff}")
        sc = score_prediction(pred, locked)
        _, fit_mae = S.eval_with_alpha(model, cache, fit_rows, None, device)
        reads[tag] = {
            "checkpoint": ck_name,
            "checkpoint_sha256": S.sha256_file(run / ck_name),
            "predictions_sha256": S.sha256_file(run / pred_name),
            "epoch": int(ck["epoch"]),
            "step": int(ck["step"]),
            "monitor_mae": ck_monitor(ck, train_json, tag),
            "alpha_fit_sha256": None,
            "alpha_locked_sha256": None,
            "train_mae_at_best": float(fit_mae),
            "curve": S.w20_t100(train_json["curve"]),
            "max_abs_diff_vs_saved_prediction": max_diff,
            **sc,
        }
        l_run = S.RUNS / f"L_fit8k_s{seed}"
        l_run.mkdir(parents=True, exist_ok=True)
        C.atomic_torch_save(
            {"locked_pred": pred, "locked_rows": locked["rows"], "y": locked["y"],
             "arm": "L", "seed": int(seed), "fit": "fit8k", "checkpoint": tag,
             "alpha_locked_sha256": None},
            l_run / f"predictions_locked_{tag}.pt",
        )
    rec["reads"] = reads
    rec["parameters"] = int(train_json["parameters"])
    rec["official_valid_loaded"] = False
    rec["official_test_loaded"] = False
    rec["locked_read"] = True
    return rec


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-reread", action="store_true")
    p.add_argument("--reason", default="")
    args = p.parse_args(argv)
    if SENTINEL.exists() and not args.allow_reread:
        print(f"refusing second Locked read: {SENTINEL} exists; use --allow-reread --reason")
        return 2
    if args.allow_reread and not args.reason:
        print("--allow-reread requires --reason")
        return 2
    device = torch.device(args.device)
    cache = S.load_train_cache()
    rows = S.split_rows()
    _, per_atom = S.build_signatures(cache)
    support = S.build_support(cache, rows["fit"], per_atom)
    locked = S.group_payload(cache, support, rows["locked"], "train")
    fit_rows = rows["fit"]

    scores = {}
    for arm in ("L", "S", "P"):
        for seed in (0, 1):
            print(f"[score] {arm} s{seed}", flush=True)
            rec = (score_l_reference(seed, cache, support, locked, fit_rows, device)
                   if arm == "L" else
                   score_scb_run(arm, seed, cache, support, locked, fit_rows, device))
            scores[f"{arm}_s{seed}"] = rec
            S.write_json(S.ANALYSIS / f"score_{arm}_s{seed}.json", rec)

    payload = {
        "protocol": S.load_cfg()["protocol_id"],
        "utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "locked_rows_sha256": S.sha256_array(rows["locked"]),
        "locked_labels_read": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "reread": bool(args.allow_reread),
        "reread_reason": args.reason or None,
        "scores": scores,
    }
    S.write_json(S.ANALYSIS / "phase1_scores.json", payload)
    S.write_json(SENTINEL, {
        "utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "locked_rows_sha256": payload["locked_rows_sha256"],
        "n_scores": int(len(scores)),
        "score_keys": sorted(scores),
        "official_test_loaded": False,
    })
    print(json.dumps({"scores": sorted(scores), "sentinel": str(SENTINEL)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
