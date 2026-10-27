"""Monitor-only analysis and pre-registered gates for BAMF-v1.

Never reads Locked1k, official valid, or official test.  Locked scoring lives in
``bamf_freeze.py score`` and is allowed only after FREEZE.json exists.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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


def _mae(pred, y):
    return float(np.abs(np.asarray(pred, dtype=np.float64) - y).mean())


def _boot_mae_diff(ae_a, ae_b, n_boot=10000, seed=20261028):
    d = np.asarray(ae_a, dtype=np.float64) - np.asarray(ae_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(int(n_boot), len(d)))
    vals = d[idx].mean(1)
    return {"mean_mae_diff": float(d.mean()), "ci95": [float(np.quantile(vals, 0.025)),
                                                       float(np.quantile(vals, 0.975))],
            "n": int(len(d))}


def _load_pred(path: Path, key: str = "monitor_pred"):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and key in obj:
        return np.asarray(obj[key], dtype=np.float64)
    if isinstance(obj, dict) and "pred" in obj:
        return np.asarray(obj["pred"], dtype=np.float64)
    raise KeyError(f"{path}: no {key}/pred")


def _groups(cache, split):
    sig_sets, per_atom = SCB.build_signatures(cache)
    support = SCB.build_support(cache, split["fit8k"], per_atom)
    groups = SCB.group_payload(cache, support, split["monitor"], "train")
    return support, groups


def _stats(pred_a, pred_b, y, mask):
    ae_a = np.abs(pred_a - y)
    ae_b = np.abs(pred_b - y)
    out = {
        "n": int(mask.sum()),
        "mae_a": _mae(pred_a[mask], y[mask]) if mask.any() else None,
        "mae_b": _mae(pred_b[mask], y[mask]) if mask.any() else None,
        "mae_diff_a_minus_b": float((ae_a[mask] - ae_b[mask]).mean()) if mask.any() else None,
        "pred_diff_mean": float((pred_a[mask] - pred_b[mask]).mean()) if mask.any() else None,
    }
    if mask.any():
        boot = _boot_mae_diff(ae_a[mask], ae_b[mask])
        out["bootstrap"] = boot
        out["mean_mae_diff"] = boot["mean_mae_diff"]
        out["ci95"] = boot["ci95"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(_ROUND / "analysis" / "monitor_gates.json"))
    args = p.parse_args(argv)
    device = torch.device(args.device)
    split = DT.split_rows()
    cache = D.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)[split["monitor"]]
    support, groups = _groups(cache, split)
    masks = groups["masks"]

    info = {}
    preds = {}
    for arm in ("G", "I"):
        for seed in (0, 1):
            run = R.run_dir(arm, seed)
            tj = run / "train.json"
            if not tj.exists():
                continue
            d = json.loads(tj.read_text())
            pred = _load_pred(run / "predictions_best_monitor.pt")
            preds[f"{arm}_s{seed}"] = pred
            curve = d["curve"]
            vals = np.asarray([float(e["monitor_mae"]) for e in curve])
            w20 = float(np.min([np.median(vals[i:i + 20]) for i in range(len(vals) - 19)])) if len(vals) >= 20 else float(np.median(vals))
            t100 = float(np.median(vals[-min(100, len(vals)):]))
            info[f"{arm}_s{seed}"] = {
                "monitor_mae": _mae(pred, y),
                "train_json_best_monitor": d["best_monitor"],
                "w20": w20,
                "t100": t100,
                "final_monitor": float(vals[-1]),
                "epochs_run": int(d["epochs_run"]),
                "parameters": int(d["parameters"]),
                "gpu_hours": float(d["gpu_hours"]),
                "initial_state_sha256": d["model_info"]["initial_state_sha256"],
                "plan": d.get("plan"),
            }
    # Historical L baselines: regenerate Monitor predictions from the checkpoint.
    # Do NOT open the historical predictions_*.pt file: it also contains unused
    # locked_pred/official_valid_pred arrays.  The checkpoint contains only
    # Fit8k-trained weights.
    norm = DT.fit_only_norm(cache, split["fit8k"])
    for seed in (0, 1):
        cp = _SUFF / "zinc_dccl_bottleneck_v1" / "runs" / f"fit8k_s{seed}" / "best_monitor.pt"
        if not cp.exists():
            continue
        model = DCC.DualChannel(norm).to(device)
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        C.load_state(model, ck["state_dict"], device)
        pred, mae = T1.evaluate(model, cache, split["monitor"], candidate="DCC-L",
                                device=device, batch_size=int(R.RECIPE["valid_batch_size"]),
                                collate_fn=DR.collate_p3n)
        preds[f"L_s{seed}"] = np.asarray(pred, dtype=np.float64)
        info[f"L_s{seed}"] = {"monitor_mae": float(mae),
                              "source": str(cp.relative_to(C.REPO)),
                              "regenerated_from_checkpoint": True}

    out = {"protocol_id": R.ROUND_ID, "rows": "monitor",
           "group_counts": groups["counts"],
           "group_tier_counts": {k: int((groups["tier"] == k).sum()) for k in ("unseen", "low4", "full")},
           "models": info, "comparisons": {}}

    for seed in (0, 1):
        g = f"G_s{seed}"
        i = f"I_s{seed}"
        l = f"L_s{seed}"
        if g not in preds or i not in preds:
            continue
        comp = {
            "G_minus_I": {
                "all": _stats(preds[g], preds[i], y, np.ones(len(y), bool)),
                "any_unseen_nopen": _stats(preds[g], preds[i], y, masks["primary"]),
            },
            "G_minus_L": {
                "all": _stats(preds[g], preds[l], y, np.ones(len(y), bool)),
                "any_unseen_nopen": _stats(preds[g], preds[l], y, masks["primary"]),
                "penalty": _stats(preds[g], preds[l], y, masks["penalty_all"]),
                "nopen": _stats(preds[g], preds[l], y, masks["nopen"]),
            },
        }
        # top-10 influential rows for G-L, by absolute row AE difference
        ae_g = np.abs(preds[g] - y)
        ae_l = np.abs(preds[l] - y)
        d = ae_g - ae_l
        idx = np.argsort(-np.abs(d))[:10]
        keep = np.ones(len(d), bool)
        keep[idx] = False
        comp["G_minus_L"]["drop_top10"] = {
            "kept_mean_mae_diff": float(d[keep].mean()),
            "dropped_row_indices": idx.tolist(),
            "dropped_abs_ae_diffs": np.abs(d[idx]).tolist(),
        }
        out["comparisons"][f"s{seed}"] = comp

    # Pre-registered seed-0 gate.
    gate0 = {"status": "NOT_RUN", "checks": {}}
    if "s0" in out["comparisons"]:
        c = out["comparisons"]["s0"]
        gi = c["G_minus_I"]["all"]["mae_diff_a_minus_b"]
        gl = c["G_minus_L"]["all"]["mae_diff_a_minus_b"]
        un = c["G_minus_L"]["any_unseen_nopen"]["mae_diff_a_minus_b"]
        checks = {
            "G_minus_I_le_minus_0.004": bool(gi <= -0.004),
            "G_minus_L_le_plus_0.002": bool(gl <= 0.002),
            "any_unseen_nopen_vs_L_improvement_ge_0.005_suggested": bool(-un >= 0.005),
        }
        gate0 = {"status": "PASS" if checks["G_minus_I_le_minus_0.004"] and checks["G_minus_L_le_plus_0.002"] else "FAIL",
                 "values": {"G_minus_I": gi, "G_minus_L": gl,
                            "any_unseen_nopen_vs_L_improvement": -un},
                 "checks": checks}
    out["seed0_gate"] = gate0

    # Final Monitor-side gates (Locked is separate and later).
    final = {"status": "NOT_READY", "checks": {}}
    if all(f"s{s}" in out["comparisons"] for s in (0, 1)):
        gi = [out["comparisons"][f"s{s}"]["G_minus_I"]["all"]["mae_diff_a_minus_b"] for s in (0, 1)]
        gl = [out["comparisons"][f"s{s}"]["G_minus_L"]["all"]["mae_diff_a_minus_b"] for s in (0, 1)]
        un = [-out["comparisons"][f"s{s}"]["G_minus_L"]["any_unseen_nopen"]["mae_diff_a_minus_b"] for s in (0, 1)]
        drops = [out["comparisons"][f"s{s}"]["G_minus_L"]["drop_top10"]["kept_mean_mae_diff"] for s in (0, 1)]
        pen = [out["comparisons"][f"s{s}"]["G_minus_L"]["penalty"]["mae_diff_a_minus_b"] for s in (0, 1)]
        nopen = [out["comparisons"][f"s{s}"]["G_minus_L"]["nopen"]["mae_diff_a_minus_b"] for s in (0, 1)]
        checks = {
            "G_minus_I_both_seeds_improve": bool(all(v < 0 for v in gi)),
            "G_minus_I_mean_le_minus_0.005": bool(np.mean(gi) <= -0.005),
            "G_minus_L_both_seeds_improve": bool(all(v < 0 for v in gl)),
            "G_minus_L_mean_le_minus_0.006": bool(np.mean(gl) <= -0.006),
            "any_unseen_nopen_both_ge_0.010": bool(all(v >= 0.010 for v in un)),
            "drop_top10_both_ge_0.002_improvement": bool(all(v <= -0.002 for v in drops)),
            "penalty_not_the_only_source": bool(np.mean(nopen) <= 0 and np.mean(pen) <= 0),
        }
        final = {"status": "PASS" if all(checks.values()) else "FAIL",
                 "values": {"G_minus_I": gi, "G_minus_L": gl,
                            "any_unseen_nopen_improvement": un,
                            "drop_top10_kept_mae_diff": drops,
                            "penalty_mae_diff": pen, "nopen_mae_diff": nopen,
                            "G_minus_I_mean": float(np.mean(gi)),
                            "G_minus_L_mean": float(np.mean(gl))},
                 "checks": checks}
    out["final_monitor_gate"] = final
    out["official_valid_loaded"] = False
    out["official_test_loaded"] = False
    out["locked_loaded"] = False
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("models", "comparisons", "seed0_gate",
                                           "final_monitor_gate", "locked_loaded")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
