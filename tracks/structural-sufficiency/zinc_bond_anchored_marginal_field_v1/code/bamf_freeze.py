"""Freeze BAMF predictions, then read Locked1k exactly once.

The ``freeze`` command never touches Locked.  The ``score`` command refuses to
run unless the freeze record exists and no earlier Locked record exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
for _p in (
    str(_HERE),
    str(_SUFF / "zinc_arch_discovery_v1" / "code"),
    str(_SUFF / "zinc_dual_channel_composer_v1" / "code"),
    str(_SUFF / "zinc_dccl_bottleneck_v1" / "code"),
    str(_SUFF / "zinc_support_conditioned_backoff_v1" / "code"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bamf_models as BM  # noqa: E402
import bamf_runner as R  # noqa: E402
import dcb_train as DT  # noqa: E402
import dcc_models as DCC  # noqa: E402
import dcc_runner as DR  # noqa: E402
import scb_common as SCB  # noqa: E402


def _sha(path: Path) -> str:
    return C.sha256_file(path)


def _load_ckpt(arm: str, seed: int, checkpoint: str):
    path = R.run_dir(arm, seed) / f"{checkpoint}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return path, ck


def _eval_monitor(arm: str, seed: int, checkpoint: str, device, plan=None) -> dict:
    split = DT.split_rows()
    cache = D.load_cache("train")
    norm = DT.fit_only_norm(cache, split["fit8k"])
    if plan is None and arm == "G":
        plan = R.build_plan(cache)
    state: dict = {}
    collate = R.make_collate(arm, plan, device, state)
    model, _ = BM.build_model("BAMF", norm, seed, device, arm=arm,
                              plan=None if plan is None else __import__("copy").deepcopy(plan).to(device))
    path, ck = _load_ckpt(arm, seed, checkpoint)
    C.load_state(model, ck["state_dict"], device)
    pred, mae = T1.evaluate(model, cache, split["monitor"], candidate="BAMF", device=device,
                            batch_size=256, collate_fn=collate)
    return {"mae": float(mae), "pred": pred, "rows": split["monitor"],
            "checkpoint_path": str(path.relative_to(C.REPO)), "checkpoint_sha256": _sha(path)}


def freeze(device, *, freezefile: Path) -> dict:
    if freezefile.exists():
        raise FileExistsError(f"{freezefile} already exists; refusing to overwrite a freeze")
    split = DT.split_rows()
    cache = D.load_cache("train")
    plan = R.build_plan(cache)
    records = {}
    for arm in ("G", "I"):
        for seed in (0, 1):
            for checkpoint in ("best_monitor", "final"):
                out = _eval_monitor(arm, seed, checkpoint, device, plan=plan)
                pred_path = R.run_dir(arm, seed) / f"frozen_monitor_{checkpoint}.npz"
                np.savez_compressed(
                    pred_path,
                    pred=out["pred"], rows=out["rows"], arm=arm, seed=seed,
                    checkpoint=checkpoint, mae=out["mae"],
                )
                records[f"{arm}_s{seed}_{checkpoint}"] = {
                    "arm": arm, "seed": seed, "checkpoint": checkpoint,
                    "monitor_mae": out["mae"],
                    "checkpoint_path": out["checkpoint_path"],
                    "checkpoint_sha256": out["checkpoint_sha256"],
                    "prediction_path": str(pred_path.relative_to(C.REPO)),
                    "prediction_sha256": _sha(pred_path),
                }
    # L baselines: matched historical DCC-L, frozen checkpoints only.
    norm = DT.fit_only_norm(cache, split["fit8k"])
    for seed in (0, 1):
        cp = _SUFF / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{seed}" / "best_monitor.pt"
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        model = DCC.DualChannel(norm)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        pred, mae = T1.evaluate(model, cache, split["monitor"], candidate="P3N",
                                device=device, batch_size=256, collate_fn=DR.collate_p3n)
        records[f"L_s{seed}_best_monitor"] = {
            "arm": "L", "seed": seed, "checkpoint": "best_monitor",
            "monitor_mae": float(mae), "checkpoint_path": str(cp.relative_to(C.REPO)),
            "checkpoint_sha256": _sha(cp),
            "frozen_train_json_monitor_mae": float(ck["monitor_mae"]),
        }
    train_info = {a: {s: json.loads((R.run_dir(a, s) / "train.json").read_text())
                      for s in (0, 1)} for a in ("G", "I")}
    payload = {
        "protocol_id": R.ROUND_ID,
        "freeze_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": C.git_head() if hasattr(C, "git_head") else None,
        "split_hashes": split["hashes"],
        "plan_sha256": R.plan_hash(plan),
        "plan_max_width": int(plan.max_width),
        "records": records,
        "train_initial_state_sha256": {
            f"{a}_s{s}": train_info[a][s]["model_info"]["initial_state_sha256"]
            for a in ("G", "I") for s in (0, 1)
        },
        "train_gpu_hours": {
            f"{a}_s{s}": float(train_info[a][s]["gpu_hours"])
            for a in ("G", "I") for s in (0, 1)
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_loaded": False,
    }
    freezefile.parent.mkdir(parents=True, exist_ok=True)
    freezefile.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _load_model(arm: str, seed: int, checkpoint: str, cache, split, device, plan):
    norm = DT.fit_only_norm(cache, split["fit8k"])
    state: dict = {}
    collate = R.make_collate(arm, plan, device, state)
    model, _ = BM.build_model("BAMF", norm, seed, device, arm=arm,
                              plan=None if plan is None else __import__("copy").deepcopy(plan).to(device))
    path, ck = _load_ckpt(arm, seed, checkpoint)
    C.load_state(model, ck["state_dict"], device)
    return model, collate, path, ck


def _eval_rows(model, collate, cache, rows, device, arm):
    pred, mae = T1.evaluate(model, cache, rows, candidate="BAMF", device=device,
                            batch_size=256, collate_fn=collate)
    return pred, float(mae)


def _bootstrap_ci(x, clusters, n_boot=10000, seed=20261028):
    x = np.asarray(x, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    sums = np.bincount(inv, weights=x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(int(n_boot), len(uniq)))
    vals = sums[idx].sum(1) / counts[idx].sum(1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(vals, 0.025)),
                                              float(np.quantile(vals, 0.975))],
            "n_molecules": int(len(x)), "n_clusters": int(len(uniq))}


def score(device, *, freezefile: Path, outfile: Path) -> dict:
    if not freezefile.exists():
        raise FileNotFoundError("freeze record missing; run freeze first")
    if outfile.exists():
        raise FileExistsError(f"{outfile} already exists; Locked is read exactly once")
    freeze = json.loads(freezefile.read_text())
    split = DT.split_rows()
    locked = split["locked"]
    cache = D.load_cache("train")
    plan = R.build_plan(cache)
    rows = np.asarray(locked, dtype=np.int64)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]

    # Group definitions are label-free except for the penalty flag, which is
    # used only for post-hoc attribution, not training.
    sig_sets, per_atom = SCB.build_signatures(cache)
    support = SCB.build_support(cache, split["fit8k"], per_atom)
    groups = SCB.group_payload(cache, support, rows, "train")

    preds = {}
    records = {}
    for arm in ("G", "I"):
        for seed in (0, 1):
            for checkpoint in ("best_monitor", "final"):
                model, collate, path, ck = _load_model(arm, seed, checkpoint, cache, split, device, plan)
                pred, mae = _eval_rows(model, collate, cache, rows, device, arm)
                key = f"{arm}_s{seed}_{checkpoint}"
                preds[key] = pred
                records[key] = {
                    "mae": mae,
                    "checkpoint_sha256": _sha(path),
                    "locked_pred_sha256": hashlib.sha256(pred.astype(np.float32).tobytes()).hexdigest(),
                }
                np.savez_compressed(
                    R.run_dir(arm, seed) / f"locked_{checkpoint}.npz",
                    pred=pred, rows=rows, y=y, arm=arm, seed=seed, checkpoint=checkpoint,
                )

    # L baselines
    norm = DT.fit_only_norm(cache, split["fit8k"])
    for seed in (0, 1):
        cp = _SUFF / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{seed}" / "best_monitor.pt"
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        model = DCC.DualChannel(norm)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        pred, mae = T1.evaluate(model, cache, rows, candidate="P3N", device=device,
                                batch_size=256, collate_fn=DR.collate_p3n)
        preds[f"L_s{seed}_best_monitor"] = pred
        records[f"L_s{seed}_best_monitor"] = {
            "mae": float(mae), "checkpoint_sha256": _sha(cp),
            "locked_pred_sha256": hashlib.sha256(pred.astype(np.float32).tobytes()).hexdigest(),
        }

    # Paired differences on Locked, plus subgroup attribution.
    def diff(a, b):
        return preds[a] - preds[b]

    def mae(p):
        return float(np.abs(np.asarray(p) - y).mean())

    def subgroup_stats(a, b, mask):
        d = diff(a, b)[mask]
        return {
            "n": int(mask.sum()),
            "mae_a": mae(preds[a][mask]) if mask.any() else None,
            "mae_b": mae(preds[b][mask]) if mask.any() else None,
            "mean_diff_a_minus_b": float(d.mean()) if mask.any() else None,
            "median_diff": float(np.median(d)) if mask.any() else None,
        }

    comparisons = {}
    for seed in (0, 1):
        comparisons[f"G_minus_I_s{seed}"] = {
            "all": subgroup_stats(f"G_s{seed}_best_monitor", f"I_s{seed}_best_monitor",
                                  np.ones(len(rows), bool)),
            "any_unseen_nopen": subgroup_stats(
                f"G_s{seed}_best_monitor", f"I_s{seed}_best_monitor",
                groups["masks"]["primary"]),
        }
        comparisons[f"G_minus_L_s{seed}"] = {
            "all": subgroup_stats(f"G_s{seed}_best_monitor", f"L_s{seed}_best_monitor",
                                  np.ones(len(rows), bool)),
            "any_unseen_nopen": subgroup_stats(
                f"G_s{seed}_best_monitor", f"L_s{seed}_best_monitor",
                groups["masks"]["primary"]),
            "penalty": subgroup_stats(f"G_s{seed}_best_monitor", f"L_s{seed}_best_monitor",
                                      groups["masks"]["penalty_all"]),
            "nopen": subgroup_stats(f"G_s{seed}_best_monitor", f"L_s{seed}_best_monitor",
                                    groups["masks"]["nopen"]),
        }

    # Robustness: drop the top-10 rows by |G-L| per seed, recompute the mean diff.
    def drop_top(a, b, k=10):
        d = diff(a, b)
        idx = np.argsort(-np.abs(d))[:k]
        keep = np.ones(len(d), bool)
        keep[idx] = False
        return {
            "kept_mean_diff": float(d[keep].mean()),
            "dropped_rows": rows[idx].tolist(),
            "dropped_abs_diffs": np.abs(d[idx]).tolist(),
        }

    drop = {f"s{seed}": drop_top(f"G_s{seed}_best_monitor", f"L_s{seed}_best_monitor")
            for seed in (0, 1)}

    # Bootstrap CI for G-L and G-I across molecules (Locked rows are already
    # disjoint molecules; no clusters needed beyond molecule identity).
    ci = {}
    for seed in (0, 1):
        for name, a, b in (
            ("G_minus_I", f"G_s{seed}_best_monitor", f"I_s{seed}_best_monitor"),
            ("G_minus_L", f"G_s{seed}_best_monitor", f"L_s{seed}_best_monitor"),
        ):
            ci[f"{name}_s{seed}"] = _bootstrap_ci(diff(a, b), np.arange(len(rows)))

    payload = {
        "protocol_id": R.ROUND_ID,
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "freeze": str(freezefile),
        "freeze_sha256": _sha(freezefile),
        "locked_rows": rows.tolist(),
        "locked_rows_sha256": split["hashes"]["locked"],
        "group_counts": groups["counts"],
        "group_tier_counts": {k: int((groups["tier"] == k).sum()) for k in ("unseen", "low4", "full")},
        "records": records,
        "comparisons": comparisons,
        "drop_top10_G_minus_L": drop,
        "molecule_bootstrap_ci": ci,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(payload, indent=2) + "\n")
    # Record the single Locked read, with prediction hashes.
    (R.ANALYSIS / "LOCKED_READ.json").write_text(json.dumps({
        "scored_utc": payload["scored_utc"],
        "freeze_sha256": payload["freeze_sha256"],
        "outfile": str(outfile),
        "outfile_sha256": _sha(outfile),
        "records": records,
    }, indent=2) + "\n")
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "score"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--freeze-file", default=str(R.ANALYSIS / "FREEZE.json"))
    p.add_argument("--out", default=str(R.ANALYSIS / "locked_scores.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "freeze":
        out = freeze(device, freezefile=Path(args.freeze_file))
        print(json.dumps({k: out[k] for k in ("freeze_utc", "plan_sha256", "split_hashes",
                                               "train_gpu_hours", "official_valid_loaded",
                                               "official_test_loaded", "locked_loaded")}, indent=2))
    else:
        out = score(device, freezefile=Path(args.freeze_file), outfile=Path(args.out))
        print(json.dumps({k: out[k] for k in ("group_counts", "group_tier_counts",
                                               "comparisons", "drop_top10_G_minus_L",
                                               "molecule_bootstrap_ci")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
