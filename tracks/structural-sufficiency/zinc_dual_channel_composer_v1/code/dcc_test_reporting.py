"""Authorized reporting-only evaluation on the official ZINC test split.

This script is deliberately *not* a selector.  It verifies the frozen checkpoint
hashes, evaluates the three best-valid checkpoints once, and writes descriptive
test metrics.  No model, epoch, recipe, subgroup route, or decision is changed
from test feedback.

The test split is spent and is not a pristine blind test.  This is a re-read
authorized by the user after the valid-only decision was frozen; see TEST_PREREG.md.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_REPO = _HERE.parents[3]
for _p in (
    str(_HERE),
    str(_SUFF / "zinc_arch_discovery_v1" / "code"),
    str(_SUFF / "zinc_recipe_trunk_v2" / "code"),
    str(_SUFF / "zinc_generalization_audit_v1" / "code"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import audit_common as AC  # noqa: E402
import audit_coverage as ACov  # noqa: E402
import dcc_models as M  # noqa: E402
import dcc_runner as R  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RUNS = EXP / "runs"
P3B_RUN = _SUFF / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0"
TEST_CACHE = _SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_test.pt"
PREREG = EXP / "TEST_PREREG.md"
MODELS = ("P3B_s0", "L_s0", "C_s0")

SPECS = {
    "P3B_s0": {
        "path": P3B_RUN / "best.pt",
        "sha256": "4c379c93214c003a6c58fd4add4a64ecb4fe23603233738fab18f3885abdc34f",
        "candidate": "P3N",
        "family": "P3B",
        "organized_local": None,
        "batch_size": 256,
        "valid_best": 0.07622375113883754,
        "epoch": 377,
        "run": P3B_RUN,
    },
    "L_s0": {
        "path": RUNS / "L_s0" / "best.pt",
        "sha256": "fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5",
        "candidate": "DCC_L",
        "family": "DCC",
        "organized_local": True,
        "batch_size": 256,
        "valid_best": 0.0785371521859779,
        "epoch": 489,
        "run": RUNS / "L_s0",
    },
    "C_s0": {
        "path": RUNS / "C_s0" / "best.pt",
        "sha256": "0a828c494f75138653ceb5677199f7b17e77943b5251bfd4b901406fe6739067",
        "candidate": "DCC_C",
        "family": "DCC",
        "organized_local": False,
        "batch_size": 256,
        "valid_best": 0.0823674315995886,
        "epoch": 482,
        "run": RUNS / "C_s0",
    },
}

TEST_CACHE_SHA256 = "981e2b9a2588b300024947edac4a5053d054199051ca19e016d350ef3ece0dc1"
EXPECTED_LABELS_SHA256 = "66fd23ecdb1b59c76cee597efc81272e0d2744eeac80744f4159d436e5388768"


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    ae = np.abs(e)
    order = np.sort(ae)[::-1]
    total = float(ae.sum())
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(ae.max()),
        "mean_signed_error": float(e.mean()),
        "positive_error_fraction": float(np.mean(e > 0)),
        "top1_abs_error_share": float(order[:1].sum() / total),
        "top10_abs_error_share": float(order[:10].sum() / total),
        "top50_abs_error_share": float(order[:50].sum() / total),
        "top100_abs_error_share": float(order[:100].sum() / total),
    }


def bootstrap_delta(delta: np.ndarray, idx: np.ndarray) -> dict:
    means = delta[idx].mean(axis=1)
    return {
        "mean_delta_mae": float(delta.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def group_row(mask: np.ndarray, y: np.ndarray, pred: dict) -> dict:
    ae = {k: np.abs(pred[k] - y) for k in MODELS}
    out = {"n": int(mask.sum())}
    for k in MODELS:
        out[f"mae_{k}"] = float(ae[k][mask].mean())
    out["delta_L_minus_P3B"] = float((ae["L_s0"] - ae["P3B_s0"])[mask].mean())
    out["delta_C_minus_P3B"] = float((ae["C_s0"] - ae["P3B_s0"])[mask].mean())
    out["delta_L_minus_C"] = float((ae["L_s0"] - ae["C_s0"])[mask].mean())
    return out


def build_and_eval(name: str, spec: dict, test: dict, rows: np.ndarray, norm: dict) -> tuple[np.ndarray, dict]:
    path = Path(spec["path"])
    actual = C.sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(f"{name}: checkpoint hash mismatch {actual} != {spec['sha256']}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("epoch", -1)) != int(spec["epoch"]):
        raise RuntimeError(f"{name}: checkpoint epoch {ck.get('epoch')} != frozen {spec['epoch']}")
    if abs(float(ck["valid_mae"]) - float(spec["valid_best"])) > 1e-8:
        raise RuntimeError(f"{name}: checkpoint valid MAE {ck['valid_mae']} != frozen {spec['valid_best']}")

    if spec["family"] == "P3B":
        model, _ = A2.build_model(spec["candidate"], norm, 0, "cpu")
    else:
        model, _ = M.build_model(
            spec["candidate"], norm, 0, "cpu", organized_local=bool(spec["organized_local"])
        )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    pred, mae = T1.evaluate(
        model,
        test,
        rows,
        candidate=spec["candidate"],
        device="cpu",
        batch_size=int(spec["batch_size"]),
        collate_fn=R.collate_p3n,
    )
    info = {
        "checkpoint": str(path.relative_to(_REPO)),
        "checkpoint_sha256": actual,
        "checkpoint_epoch": int(ck["epoch"]),
        "checkpoint_valid_mae": float(ck["valid_mae"]),
        "parameters": int(C.parameter_count(model)),
        "test_mae_recomputed": float(mae),
        "prediction_sha256": C.sha256_array(pred),
        "candidate": spec["candidate"],
        "organized_local": spec["organized_local"],
    }
    return pred, info


def main() -> int:
    if not PREREG.exists():
        raise FileNotFoundError(PREREG)
    if not TEST_CACHE.exists():
        raise FileNotFoundError(TEST_CACHE)
    cache_sha = C.sha256_file(TEST_CACHE)
    if cache_sha != TEST_CACHE_SHA256:
        raise RuntimeError(f"test cache hash mismatch: {cache_sha}")

    test = D.load_cache("test")
    rows = np.arange(int(test["meta"]["n_molecules"]), dtype=np.int64)
    y = np.asarray(test["labels"], dtype=np.float64)
    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 test rows, got {len(rows)}")
    labels_sha = C.sha256_array(y)
    if labels_sha != EXPECTED_LABELS_SHA256:
        raise RuntimeError(f"test label hash mismatch: {labels_sha}")

    norm = D.load_norm()
    pred = {}
    model_info = {}
    for name in MODELS:
        pred[name], model_info[name] = build_and_eval(name, SPECS[name], test, rows, norm)
        print(f"{name}: test_mae={model_info[name]['test_mae_recomputed']:.9f} "
              f"valid={model_info[name]['checkpoint_valid_mae']:.9f} "
              f"ep={model_info[name]['checkpoint_epoch']} params={model_info[name]['parameters']}")

    # Per-molecule descriptors: label-free graph descriptors + fixed signature coverage.
    graph_rows = []
    for i in rows:
        st = AC.graph_stats(test, int(i))
        graph_rows.append(st)
    train = D.load_cache("train")
    train_sigs, _ = ACov.radius_signatures(train)
    _, test_sigs = ACov.radius_signatures(test)
    sig2 = ACov.signature_coverage(train_sigs, test_sigs, test)
    r2_unseen = np.asarray(sig2["unseen_atom_fraction"], dtype=np.float64)
    if len(r2_unseen) != len(rows):
        raise RuntimeError("signature coverage length mismatch")

    unseen = r2_unseen > 0.0
    ordinary = y > -3.0
    rings4 = np.asarray([int(st["n_rings"]) >= 4 for st in graph_rows])
    ae = {k: np.abs(pred[k] - y) for k in MODELS}
    worst100 = set(np.argsort(ae["P3B_s0"])[-100:].tolist())
    worst100_mask = np.zeros(len(rows), dtype=bool)
    worst100_mask[list(worst100)] = True

    groups = {
        "signature_unseen": unseen,
        "signature_covered": ~unseen,
        "ordinary_y_gt_minus3": ordinary,
        "tail_y_le_minus3": ~ordinary,
        "rings_ge4": rings4,
        "rings_ge4_and_signature_unseen": rings4 & unseen,
        "P3B_test_worst100": worst100_mask,
    }
    subgroup = {name: group_row(mask, y, pred) for name, mask in groups.items()}
    subgroup["unseen_minus_covered_delta_of_deltas"] = {
        "L_minus_P3B": subgroup["signature_unseen"]["delta_L_minus_P3B"] - subgroup["signature_covered"]["delta_L_minus_P3B"],
        "C_minus_P3B": subgroup["signature_unseen"]["delta_C_minus_P3B"] - subgroup["signature_covered"]["delta_C_minus_P3B"],
        "L_minus_C": subgroup["signature_unseen"]["delta_L_minus_C"] - subgroup["signature_covered"]["delta_L_minus_C"],
        "definition": "mean paired delta inside unseen minus mean paired delta inside covered; negative means the first model's advantage is larger in the unseen group",
    }

    rng = np.random.default_rng(20261021)
    boot_idx = rng.integers(0, len(rows), size=(10000, len(rows)))
    contrasts = {}
    for a, b in (("L_s0", "P3B_s0"), ("C_s0", "P3B_s0"), ("L_s0", "C_s0")):
        contrasts[f"{a}_minus_{b}"] = bootstrap_delta(ae[a] - ae[b], boot_idx)

    # Outputs.
    pred_dir = ANALYSIS / "test_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_hashes = {}
    for name in MODELS:
        p = pred_dir / f"{name}_test.npz"
        np.savez_compressed(p, row=rows, pred=pred[name].astype(np.float32), y=y.astype(np.float32))
        pred_hashes[name] = C.sha256_file(p)

    per_path = ANALYSIS / "test_per_molecule.csv"
    fields = [
        "row_id", "y_raw", "n_atoms", "n_bonds", "n_components", "cycle_rank", "n_rings",
        "max_ring_size", "r2_unseen_atom_frac",
        "pred_P3B_s0", "pred_L_s0", "pred_C_s0",
        "signed_error_P3B_s0", "signed_error_L_s0", "signed_error_C_s0",
        "abs_error_P3B_s0", "abs_error_L_s0", "abs_error_C_s0",
        "signature_unseen", "ordinary_y_gt_minus3", "tail_y_le_minus3", "rings_ge4",
        "P3B_test_worst100",
    ]
    with open(per_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, st in enumerate(graph_rows):
            out = {
                "row_id": i,
                "y_raw": f"{y[i]:.9g}",
                "n_atoms": st["n_atoms"],
                "n_bonds": st["n_bonds"],
                "n_components": st["n_components"],
                "cycle_rank": st["cycle_rank"],
                "n_rings": st["n_rings"],
                "max_ring_size": st["max_ring_size"],
                "r2_unseen_atom_frac": f"{r2_unseen[i]:.9g}",
                "signature_unseen": int(unseen[i]),
                "ordinary_y_gt_minus3": int(ordinary[i]),
                "tail_y_le_minus3": int(~ordinary[i]),
                "rings_ge4": int(rings4[i]),
                "P3B_test_worst100": int(worst100_mask[i]),
            }
            for k in MODELS:
                out[f"pred_{k}"] = f"{pred[k][i]:.9g}"
                out[f"signed_error_{k}"] = f"{pred[k][i] - y[i]:.9g}"
                out[f"abs_error_{k}"] = f"{ae[k][i]:.9g}"
            w.writerow(out)

    main_metrics = {k: metrics(pred[k], y) for k in MODELS}
    table_rows = []
    for k in MODELS:
        table_rows.append({
            "model": k,
            "parameters": model_info[k]["parameters"],
            "best_valid": model_info[k]["checkpoint_valid_mae"],
            "best_epoch": model_info[k]["checkpoint_epoch"],
            "test_mae": main_metrics[k]["mae"],
            "test_rmse": main_metrics[k]["rmse"],
            "test_median_ae": main_metrics[k]["median_ae"],
            "test_p90_ae": main_metrics[k]["p90_ae"],
            "test_p95_ae": main_metrics[k]["p95_ae"],
            "test_max_ae": main_metrics[k]["max_ae"],
            "test_mean_signed_error": main_metrics[k]["mean_signed_error"],
            "test_top100_abs_error_share": main_metrics[k]["top100_abs_error_share"],
            "test_minus_valid_mae": main_metrics[k]["mae"] - model_info[k]["checkpoint_valid_mae"],
        })

    payload = {
        "protocol_id": "zinc-dual-channel-composer-v1-test-reporting",
        "status": "REPORTING_ONLY",
        "authorization": "User explicitly authorized on 2026-10-21 after valid-only decision was frozen: '我授权了，做吧'.",
        "prereg": {
            "path": str(PREREG.relative_to(_REPO)),
            "sha256": C.sha256_file(PREREG),
        },
        "prior_valid_decision": "C_OWNERSHIP_ONLY_NO_PERFORMANCE",
        "prior_valid_commit": "e76cc7d",
        "test_is_spent_not_pristine": True,
        "official_test_loaded": True,
        "test_read_count": 1,
        "selection_rule": "No selection: three checkpoints were frozen at their valid-best epoch before this read; test cannot change any model, epoch, recipe, or decision.",
        "test_used_for_selection": False,
        "data": {
            "split": "official PyG ZINC subset=True test",
            "n": int(len(rows)),
            "raw_y_mae": True,
            "cache_path": str(TEST_CACHE.relative_to(_REPO)),
            "cache_sha256": cache_sha,
            "labels_sha256": labels_sha,
        },
        "models": model_info,
        "metrics": main_metrics,
        "main_table": table_rows,
        "paired_contrasts": {
            "resamples": 10000,
            "rng_seed": 20261021,
            "same_row_resamples_for_all_contrasts": True,
            "delta_definition": "abs_error(first) - abs_error(second); negative means the first model is better",
            "interval_role": "descriptive fixed-checkpoint interval on the spent official-test 1000; not cross-seed confirmation",
            "contrasts": contrasts,
        },
        "subgroups": {
            "groups": subgroup,
            "note": "Fixed semantic definitions; P3B_test_worst100 is defined by a fixed rule from P3B test AE and is descriptive only, not a gate or selector.",
        },
        "artifact_hashes": {
            "test_predictions": pred_hashes,
            "test_per_molecule_csv": C.sha256_file(per_path),
            "test_prereg": C.sha256_file(PREREG),
        },
        "checkpoint_hashes": {k: model_info[k]["checkpoint_sha256"] for k in MODELS},
        "notes": [
            "The official test split is already spent; this is a reporting-only re-read, not a pristine blind test.",
            "No test number was used to select or change a model, epoch, recipe, subgroup, or follow-up run.",
            "All three checkpoints are the same valid-best checkpoints analyzed in REPORT.md.",
        ],
    }
    C.write_json(ANALYSIS / "test_reporting.json", payload)

    # Compact markdown table.
    lines = [
        "# Official ZINC test — reporting-only",
        "",
        "| model | params | best valid | best ep | test MAE | test RMSE | median AE | p90 AE | p95 AE | max AE | mean signed | top-100 share | test−valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['model']} | {r['parameters']:,} | {r['best_valid']:.9f} | {r['best_epoch']} | "
            f"{r['test_mae']:.9f} | {r['test_rmse']:.9f} | {r['test_median_ae']:.9f} | "
            f"{r['test_p90_ae']:.9f} | {r['test_p95_ae']:.9f} | {r['test_max_ae']:.9f} | "
            f"{r['test_mean_signed_error']:.9f} | {r['test_top100_abs_error_share']:.6f} | "
            f"{r['test_minus_valid_mae']:+.9f} |"
        )
    (ANALYSIS / "test_main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", ANALYSIS / "test_reporting.json")
    print("wrote", ANALYSIS / "test_main_table.md")
    print("wrote", per_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
