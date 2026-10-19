"""Phase 1e Stage B2 — one strictly isolated strong baseline on the SSRC split.

The recipe is the frozen, two-seed-replicated BJOC-Null configuration from
``tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1``
(masked atom/bond SSL initialisation + R3 360-epoch cosine + last-5-epoch
parameter-mean soup, 413,505 parameters).  It is re-run here once, seed 0,
against the **SSRC** Fit 8,000 / Monitor 1,000 rows so that the resulting number
is comparable inside one protocol.

Isolation contract (enforced in code, not just documented)
----------------------------------------------------------
* objects, standardiser, masked-SSL encoder and supervised batches use the
  SSRC Fit rows only;
* the internal Reserved rows and the official valid/test splits are never
  loaded;
* Monitor predictions are written and hashed **before** the Monitor labels are
  read; the label read happens once, in ``score``;
* single seed, single configuration, fixed epoch count, no early stopping and
  no model selection.

Commands: ``prep`` -> ``ssl`` -> ``train`` -> ``predict`` -> ``score``.
``run`` executes the whole chain (used for the GPU job).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CODE_SC = REPO / "tracks/structure-contribution/code"
for _p in (HERE, REPO / "tracks/structural-sufficiency/code/phase1b", CODE_SC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase1e_common as C  # noqa: E402

import cscl_features as cf  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402

OUT = C.RESULTS_1E / ("baseline_smoke" if os.environ.get("PHASE1E_SMOKE") == "1" else "baseline")
SPLIT_TAG = "ssrc-phase1e"
SEED = 0
ARM = "P3E"

ENCODED_TRAIN = REPO / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
LABELS_FIT_MONITOR = C.RESULTS_1B / "zinc_labels_fit_monitor.json"

OBJ_FIT = OUT / "objects_fit.pt"
OBJ_MONITOR = OUT / "objects_monitor.pt"
PREP = OUT / "prep_fit.json"
SSL_PT = OUT / "pretrain_s0.pt"
PREFIX = OUT / "train_P3E_s0"
PRED = OUT / "monitor_pred_s0.npz"
FREEZE = OUT / "monitor_prediction_freeze.json"
SCORE = OUT / "strong_baseline.json"


def _rows() -> tuple[list[int], list[int], list[int]]:
    split = C.read_json(C.SPLIT_PATH)
    fit = [int(x) for x in split["rows"]["fit"]]
    monitor = [int(x) for x in split["rows"]["monitor"]]
    reserved = [int(x) for x in split["rows"]["reserved"]]
    return fit, monitor, reserved


def _guard_isolation(fit: Sequence[int], monitor: Sequence[int], reserved: Sequence[int]) -> dict[str, Any]:
    f, m, r = set(fit), set(monitor), set(reserved)
    if f & m or f & r or m & r:
        raise SystemExit("SSRC split sides overlap; refusing to train")
    if len(f) != 8000 or len(m) != 1000 or len(r) != 1000:
        raise SystemExit("unexpected SSRC split sizes")
    return {
        "n_fit": len(f),
        "n_monitor": len(m),
        "n_reserved": len(r),
        "fit_monitor_overlap": len(f & m),
        "fit_reserved_overlap": len(f & r),
        "supervised_rows_are_fit_only": True,
        "official_valid_test_loaded": False,
        "slot_reserved_labels_read": 0,
    }


def cmd_prep(args: argparse.Namespace) -> int:
    fit, monitor, reserved = _rows()
    guard = _guard_isolation(fit, monitor, reserved)
    ds = cf.load_official_train()
    t0 = time.perf_counter()
    fit_objs = bp.build_objects_for_rows(fit, ds)
    mon_objs = bp.build_objects_for_rows(monitor, ds)
    build_s = time.perf_counter() - t0
    if [int(o.mol_index) for o in fit_objs] != fit:
        raise SystemExit("fit object order mismatch")
    if [int(o.mol_index) for o in mon_objs] != monitor:
        raise SystemExit("monitor object order mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(fit_objs, OBJ_FIT)
    torch.save(mon_objs, OBJ_MONITOR)
    prep = bp.fit_standardiser(fit_objs)
    prep["protocol"] = "ssrc-phase1e-zinc-v1"
    prep["split_tag"] = SPLIT_TAG
    prep["fit_only"] = True
    prep["isolation"] = guard
    C.write_json(PREP, prep)
    manifest = {
        "stage": "prep",
        "build_seconds": build_s,
        "fit_rows_sha256": bp.rows_sha256(fit),
        "monitor_rows_sha256": bp.rows_sha256(monitor),
        "objects_fit_sha256": bp.file_sha256(OBJ_FIT),
        "objects_monitor_sha256": bp.file_sha256(OBJ_MONITOR),
        "prep_sha256": bp.file_sha256(PREP),
        "isolation": guard,
    }
    C.write_json(OUT / "prep_manifest.json", manifest)
    print(f"[prep] objects {build_s:.1f}s; isolation={guard['supervised_rows_are_fit_only']}")
    return 0


def cmd_ssl(args: argparse.Namespace) -> int:
    fit, monitor, reserved = _rows()
    _guard_isolation(fit, monitor, reserved)
    if SSL_PT.exists():
        raise SystemExit("SSL checkpoint exists; retraining is not allowed")
    import torch

    fit_objs = f10k.load_objects(OBJ_FIT)
    if [int(o.mol_index) for o in fit_objs] != fit:
        raise SystemExit("fit object cache does not match the SSRC fit rows")
    ds = cf.load_official_train()
    payload = bp.pretrain_masked(
        seed=SEED,
        fit_objs=fit_objs,
        fit_rows=fit,
        raw_dataset=ds,
        device=torch.device(args.device),
        mode="atom_bond",
        smoke_updates=args.smoke_updates,
    )
    torch.save(payload["encoder_state"], SSL_PT)
    record = {k: v for k, v in payload.items() if k != "encoder_state"}
    record["encoder_path"] = str(SSL_PT.relative_to(REPO))
    record["encoder_file_sha256"] = bp.file_sha256(SSL_PT)
    record["split_tag"] = SPLIT_TAG
    record["fit_rows_only"] = True
    C.write_json(OUT / "pretrain_s0.json", record)
    print(f"[ssl] wall {payload['wall_seconds']:.0f}s acc_atom {payload['final_acc_atom']:.4f}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    fit, monitor, reserved = _rows()
    _guard_isolation(fit, monitor, reserved)
    if PREFIX.with_suffix(".json").exists() or PREFIX.with_name(PREFIX.name + "_soup.pt").exists():
        raise SystemExit("supervised arm exists; retraining is not allowed")
    import torch

    fit_objs = f10k.load_objects(OBJ_FIT)
    norm = C.read_json(PREP)
    if not norm.get("fit_only"):
        raise SystemExit("standardiser is not marked fit-only")
    y_fit = bp.load_y_rows(ENCODED_TRAIN, fit)
    ssl_state = torch.load(SSL_PT, map_location="cpu", weights_only=False)
    ssl_recorded = C.read_json(OUT / "pretrain_s0.json")["encoder_final_sha256"]
    payload = bp.train_supervised(
        arm=ARM,
        seed=SEED,
        recipe_key="R3",
        init="ssl",
        train_objs=fit_objs,
        rows=fit,
        y_fit=y_fit,
        norm=norm,
        device=torch.device(args.device),
        ssl_state=ssl_state,
        model_kind="base",
        smoke_epochs=args.smoke_epochs,
    )
    if payload["encoder_used_sha256"] != ssl_recorded:
        raise SystemExit("encoder used does not match the recorded pretraining state")
    soup = PREFIX.with_name(PREFIX.name + "_soup.pt")
    state = {k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}
    torch.save(state, soup)
    record = {k: v for k, v in payload.items() if k not in ("soup_state", "encoder_state")}
    record["soup_path"] = str(soup.relative_to(REPO))
    record["soup_sha256"] = bp.file_sha256(soup)
    record["split_tag"] = SPLIT_TAG
    record["n_train_rows"] = len(fit)
    record["labels_used"] = "SSRC Fit rows raw y only"
    C.write_json(PREFIX.with_suffix(".json"), record)
    print(
        f"[train] params {record.get('parameter_audit', {}).get('total', 'n/a')} "
        f"soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s"
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    fit, monitor, reserved = _rows()
    _guard_isolation(fit, monitor, reserved)
    if FREEZE.exists():
        raise SystemExit("predictions already frozen; refusing to rewrite")
    import torch

    norm = C.read_json(PREP)
    model = bp.build_bjoc_null(SEED, norm, torch.device(args.device))
    soup = PREFIX.with_name(PREFIX.name + "_soup.pt")
    model.load_state_dict(torch.load(soup, map_location=args.device, weights_only=False))
    mon_objs = f10k.load_objects(OBJ_MONITOR)
    pred = bp.predict_model(model, mon_objs, torch.device(args.device))
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite Monitor prediction")
    np.savez_compressed(PRED, row=np.arange(len(mon_objs), dtype=np.int64), pred=pred.astype(np.float32))
    freeze = {
        "stage": "predict_freeze",
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "split_tag": SPLIT_TAG,
        "arm": ARM,
        "seed": SEED,
        "soup_sha256": bp.file_sha256(soup),
        "prediction_sha256": bp.file_sha256(PRED),
        "prediction_array_sha256": bp.array_sha256(pred),
        "monitor_rows_sha256": bp.rows_sha256(monitor),
        "monitor_labels_read_before_freeze": False,
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
    }
    C.write_json(FREEZE, freeze)
    print(f"[predict] frozen {len(pred)} Monitor predictions sha={freeze['prediction_sha256'][:16]}…")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    fit, monitor, reserved = _rows()
    _guard_isolation(fit, monitor, reserved)
    if not FREEZE.exists():
        raise SystemExit("predictions are not frozen; run predict first")
    if SCORE.exists():
        raise SystemExit("already scored; the Monitor label read is one-shot")
    freeze = C.read_json(FREEZE)
    if bp.file_sha256(PRED) != freeze["prediction_sha256"]:
        raise SystemExit("prediction file changed after the freeze")

    # --- the single Monitor label read -------------------------------------
    vault = C.read_json(LABELS_FIT_MONITOR)
    labels = {int(k): float(v) for k, v in vault["labels"].items()}
    y = np.asarray([labels[int(r)] for r in monitor], dtype=np.float64)
    with np.load(PRED) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)

    mon = C.load_monitor()
    if np.abs(mon.y - y).max() > 1e-6:
        raise SystemExit("scored labels disagree with the frozen run labels")

    ae = np.abs(pred - y)
    mae = float(ae.mean())

    ref = C.read_json(C.RESULTS_1E / "reference_audit.json")
    internal = ref["protocol_tables"]["internal_ssrc_monitor"]["rows"]
    best_key = min(internal, key=lambda k: internal[k])
    best_mae = float(internal[best_key])
    best_pred = mon.pred[best_key]
    deltas = ae - np.abs(best_pred - y)
    boot = C.paired_bootstrap_mean(deltas, n_boot=5000, seed=20261022)
    vs_gref = C.paired_bootstrap_mean(
        ae - np.abs(mon.pred["G_ref_seed0"] - y), n_boot=5000, seed=20261022
    )

    payload = {
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "stage": "B2_strong_baseline",
        "status": "STRICTLY_ISOLATED_SINGLE_SEED_BASELINE",
        "recipe": "BJOC-Null (masked atom/bond SSL, Fit-only) + R3 360-epoch cosine + last-5 soup",
        "recipe_source": "tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1",
        "arm": ARM,
        "seed": SEED,
        "params": 413505,
        "monitor_rows": int(len(y)),
        "monitor_mae": mae,
        "monitor_rmse": float(np.sqrt((ae**2).mean())),
        "monitor_median_ae": float(np.median(ae)),
        "monitor_p90_ae": float(np.percentile(ae, 90)),
        "best_single_r1_or_reference": {"key": best_key, "monitor_mae": best_mae},
        "bootstrap_gap_vs_best_r1_ci95": boot["ci95"],
        "gap_vs_best_r1": boot["mean"],
        "gap_vs_G_ref": vs_gref["mean"],
        "gap_vs_G_ref_ci95": vs_gref["ci95"],
        "prediction_sha256": freeze["prediction_sha256"],
        "soup_sha256": freeze["soup_sha256"],
        "isolation": _guard_isolation(fit, monitor, reserved),
        "monitor_labels_read_count": 1,
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
        "early_stopping_on_monitor": False,
        "model_selection": "none (single fixed recipe, fixed 360 epochs)",
        "comparable_with": [
            "SSRC internal Fit-8000/Monitor-1000 numbers only",
            "not the exposed official-test BJOC-P3 number",
            "not the literature official-test numbers",
        ],
    }
    C.write_json(SCORE, payload)
    print(
        f"[score] strong baseline Monitor MAE {mae:.6f} "
        f"vs best internal {best_key} {best_mae:.6f} "
        f"(delta {boot['mean']:+.6f}, CI {boot['ci95'][0]:+.6f}..{boot['ci95'][1]:+.6f})"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    for fn in (cmd_prep, cmd_ssl, cmd_train, cmd_predict, cmd_score):
        rc = fn(args)
        if rc:
            return rc
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage", choices=["prep", "ssl", "train", "predict", "score", "run"]
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-updates", type=int, default=None)
    ap.add_argument("--smoke-epochs", type=int, default=None)
    args = ap.parse_args(argv)
    return {
        "prep": cmd_prep,
        "ssl": cmd_ssl,
        "train": cmd_train,
        "predict": cmd_predict,
        "score": cmd_score,
        "run": cmd_run,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
