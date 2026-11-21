"""CLI for the frozen word-response-operator round.

``prepare``  structure cache + support statistics + pre-training freeze
``checks``   engineering / algebra checks (Section 7)
``timing``   2 warm + 3 timed epochs, freezes the shared horizon
``train``    one allowlisted arm, seed 0 only
``refs``     ADD_WORD / FP_KRR (deterministic) and GINE_RAW_REF
``analyze``  main table, decisions, manifest (no H labels)
``holdout``  one-shot H evaluation, run only after all training is frozen
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import wo_common as C

OPERATOR_ARMS = ("FULL_WORD_OP", "LOCAL_WORD_OP", "ATOMIC_OP_CAP")
TRAINABLE_ARMS = OPERATOR_ARMS + ("GINE_RAW_REF", "PORT_DIAG_WORD_OP")


def _check_arm(arm: str, seed: int, dataset: str) -> None:
    cfg = C.load_config()
    allow = cfg["allowlist"]
    if dataset != "ogbg-mollipo" or dataset not in allow["datasets"]:
        raise SystemExit(f"dataset {dataset!r} is outside the allowlist")
    if int(seed) not in [int(x) for x in allow["seeds"]]:
        raise SystemExit(f"seed {seed} is outside the allowlist")
    if arm not in allow["arms"]:
        raise SystemExit(f"arm {arm!r} is outside the allowlist")
    if arm == "PORT_DIAG_WORD_OP" and not (C.analysis_dir() / "green_unlock.json").is_file():
        raise SystemExit("PORT_DIAG_WORD_OP is locked until a saved GREEN decision exists")


def cmd_prepare(args) -> None:
    import wo_struct as S
    import wo_baselines as B
    import wo_train as T

    t0 = time.perf_counter()
    record = S.prepare_all()
    data = T.load_all()
    support = B.support_stats(data)
    freeze = {
        "protocol_id": C.load_config()["protocol_id"],
        "prepared_at_unix": time.time(),
        "prepare_seconds": time.perf_counter() - t0,
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "structure_record_sha256": C.sha256_json(record),
        "split_freeze_sha256": C.sha256_file(C.analysis_dir() / "split_freeze.json"),
        "fit_row_ids": data["split"]["fit_row_ids"],
        "valid_row_ids": data["split"]["valid_row_ids"],
        "h_row_ids": data["split"]["h_row_ids"],
        "h_enabled": bool(data["split"]["h_enabled"]),
        "support": support,
        "scaler": record["scaler"],
        "z_dim": int(data["fit"].z_std.shape[1]),
        "dims": record["dims"],
        "n_templates": int(record["n_templates"]),
        "template_sha256": record["template_sha256"],
        "old_freeze_sha256": record["old_freeze_sha256"],
        "official_cache_sha256": record["official_cache_sha256"],
        "s_type": record["s_type"],
        "model_config": C.load_config()["operator"],
        "recipe": C.load_config()["recipe"],
    }
    C.write_json(C.analysis_dir() / "pre_train_freeze.json", C.jsonable(freeze))
    print(json.dumps({"ok": True, "prepare_seconds": freeze["prepare_seconds"],
                      "z_dim": freeze["z_dim"], "h_enabled": freeze["h_enabled"]},
                     indent=2))


def cmd_checks(args) -> None:
    import wo_probes as P
    import wo_train as T

    data = T.load_all()
    res = {
        "structure_audit": P.structure_audit(data),
        "replay": P.replay_check(data),
        "equivalence_fp64": P.equivalence(data, "float64"),
        "equivalence_fp32": P.equivalence(data, "float32"),
        "batch_split": P.batch_split_invariance(data),
        "shared_init": P.shared_init(data),
        "width": P.width_selection(data),
        "merge": P.merge_check(data),
        "health": P.health(data, steps=int(args.steps), device=args.device),
    }
    idx = C.analysis_dir() / "checks.json"
    old = C.read_json(idx) if idx.is_file() else {}
    old.update(C.jsonable(res))
    C.write_json(idx, old)
    if data["split"]["h_enabled"]:
        print(json.dumps({"note": "H is enabled; labels were not read by any check"}))
    summary = {
        "structure_audit": res["structure_audit"]["ok"],
        "replay": all(v["pass"] for v in res["replay"].values()
                      if isinstance(v, Mapping) and "pass" in v),
        "equivalence_fp64": res["equivalence_fp64"]["pass"],
        "equivalence_fp32": res["equivalence_fp32"]["pass"],
        "batch_split": res["batch_split"]["pass"],
        "shared_init": res["shared_init"]["pass"],
        "width": res["width"]["pass"],
        "merge": res["merge"]["pass"],
        "health": bool(res["health"]["loss_decreased"] and res["health"]["no_nan"]
                       and all(v > 0 for k, v in res["health"]["grad_abs_sum"].items()
                               if k in ("theta", "psi", "alpha", "pair"))),
    }
    summary["all_pass"] = all(summary.values())
    C.write_json(C.analysis_dir() / "checks_summary.json", summary)
    print(json.dumps(summary, indent=2))


def cmd_timing(args) -> None:
    import torch

    import wo_train as T

    cfg = C.load_config()
    recipe = cfg["recipe"]
    data = T.load_all()
    device = args.device
    warm = int(args.warm)
    timed = int(args.timed)
    rows = []
    for arm in OPERATOR_ARMS:
        res = T.train_arm(data, arm, warm + timed, device, score_valid=False,
                          dry_steps=None)
        per_epoch = [row["seconds"] for row in res["curves"]]
        deltas = np.diff([0.0] + per_epoch)
        t_epoch = float(np.median(deltas[warm:])) if timed else float(np.median(deltas))
        rows.append({"arm": arm, "epoch_seconds": [float(x) for x in deltas],
                     "median_timed_epoch_seconds": t_epoch,
                     "n_parameters": res["n_parameters"]})
    worst = max(rows, key=lambda r: r["median_timed_epoch_seconds"])
    safety = float(recipe["timing_safety"])
    budget_min = float(recipe["epoch_budget_minutes"])
    projected_160 = worst["median_timed_epoch_seconds"] * 160 * safety / 60.0
    epochs = int(recipe["epochs_default"])
    if projected_160 > budget_min:
        epochs = int(recipe["epochs_short"])
    out = {
        "warm_epochs": warm,
        "timed_epochs": timed,
        "rows": rows,
        "slowest_arm": worst["arm"],
        "projected_160_minutes": projected_160,
        "epoch_budget_minutes": budget_min,
        "frozen_epochs": epochs,
        "device": device,
        "torch": torch.__version__,
        "seconds_at": time.time(),
        "pass": bool(projected_160 <= budget_min or epochs == int(recipe["epochs_short"])),
    }
    C.write_json(C.analysis_dir() / "timing.json", C.jsonable(out))
    print(json.dumps({k: out[k] for k in ("slowest_arm", "projected_160_minutes",
                                          "frozen_epochs", "pass")}, indent=2))


def _frozen_epochs() -> int:
    cfg = C.load_config()
    timing = C.analysis_dir() / "timing.json"
    if not timing.is_file():
        raise SystemExit("timing.json is missing: run the timing command first")
    return int(C.read_json(timing)["frozen_epochs"])


def cmd_train(args) -> None:
    import torch

    import wo_train as T

    dataset, arm, seed = args.dataset, args.arm, int(args.seed)
    _check_arm(arm, seed, dataset)
    cfg = C.load_config()
    if arm == "GINE_RAW_REF":
        raise SystemExit("use the refs command for GINE_RAW_REF")
    epochs = int(args.epochs or _frozen_epochs())
    if epochs not in (int(cfg["recipe"]["epochs_default"]), int(cfg["recipe"]["epochs_short"])):
        raise SystemExit(f"epochs {epochs} is not one of the frozen horizons")
    data = T.load_all()
    hidden = int(args.hidden or cfg["operator"]["hidden"])
    if arm == "ATOMIC_OP_CAP":
        frozen_width = C.read_json(C.analysis_dir() / "width_selection.json")["selected_width"]
        if args.hidden is None:
            hidden = int(frozen_width)
        elif int(args.hidden) != int(frozen_width):
            raise SystemExit(f"ATOMIC width is frozen at {frozen_width}")
    deadline = _deadline() if args.respect_deadline else None
    t0 = time.perf_counter()
    result = T.train_arm(data, arm, epochs, args.device, hidden=hidden, deadline=deadline)
    run_id = f"lipo_{arm}_s{seed}"
    path = C.results_root() / run_id
    result["run_id"] = run_id
    result["dataset"] = dataset
    result["commit"] = C.git_commit()
    result["reference_commit"] = cfg["reference_commit"]
    result["config_sha256"] = C.sha256_file(C.CONFIG_PATH)
    result["split_freeze_sha256"] = C.sha256_file(C.analysis_dir() / "split_freeze.json")
    result["pre_train_freeze_sha256"] = C.sha256_file(C.analysis_dir() / "pre_train_freeze.json")
    result["regime"] = C.environment_snapshot()
    result["wall_seconds"] = time.perf_counter() - t0
    result["gpu_hours"] = result["wall_seconds"] / 3600.0 * (1 if "cuda" in args.device else 0)
    T.save_run(result, path)
    manifest = C.analysis_dir() / "run_manifest.json"
    rows = C.read_json(manifest) if manifest.is_file() else []
    rows = [r for r in rows if r.get("run_id") != run_id]
    rows.append({k: C.jsonable(v) for k, v in {
        "run_id": run_id, "arm": arm, "seed": seed, "epochs": result["epochs"],
        "hidden": result["hidden"], "n_parameters": result["n_parameters"],
        "init_state_hash": result["init_state_hash"], "seconds": result["seconds"],
        "wall_seconds": result["wall_seconds"], "gpu_hours": result["gpu_hours"],
        "cholesky_jitter": result["cholesky_jitter"],
        "best_epoch": result.get("best", {}).get("epoch"),
        "best_valid_rmse": result.get("best", {}).get("valid_rmse_raw"),
        "final_valid_rmse": result.get("final", {}).get("valid_rmse_raw"),
        "truncated_by_deadline": result["truncated_by_deadline"],
        "commit": result["commit"], "regime": result["regime"].get("torch"),
        "device": args.device,
    }.items()})
    C.write_json(manifest, rows)
    print(json.dumps({"run_id": run_id, "params": result["n_parameters"],
                      "best_valid_rmse": result.get("best", {}).get("valid_rmse_raw"),
                      "epochs": result["epochs"], "seconds": result["seconds"]}, indent=2))


def cmd_refs(args) -> None:
    cfg = C.load_config()
    data = None

    import wo_train as T

    data = T.load_all()
    if args.ref in ("all", "add"):
        import wo_baselines as B
        out = B.add_word(data)
        print(json.dumps({"ADD_WORD": out["valid_rmse_raw"]}))
    if args.ref in ("all", "krr"):
        import wo_baselines as B
        out = B.fp_krr(data)
        print(json.dumps({"FP_KRR": out.get("valid_rmse_raw"),
                          "unavailable": out.get("unavailable", False)}))
    if args.ref in ("all", "gine"):
        import wo_baselines as B
        _check_arm("GINE_RAW_REF", 0, "ogbg-mollipo")
        deadline = _deadline() if args.respect_deadline else None
        out = B.gine_raw(data, args.device, epochs=int(args.epochs or 240),
                         deadline=deadline)
        payload = {k: v for k, v in out.items() if k not in ("state_best", "state_final")}
        payload["commit"] = C.git_commit()
        payload["reference_commit"] = cfg["reference_commit"]
        payload["regime"] = C.environment_snapshot()
        C.write_json(C.run_dir("lipo", "GINE_RAW_REF") / "run.json", C.jsonable(payload))
        manifest = C.analysis_dir() / "run_manifest.json"
        rows = C.read_json(manifest) if manifest.is_file() else []
        rows = [r for r in rows if r.get("run_id") != "lipo_GINE_RAW_REF_s0"]
        rows.append({"run_id": "lipo_GINE_RAW_REF_s0", "arm": "GINE_RAW_REF", "seed": 0,
                     "epochs": int(out.get("epochs", 240)), "hidden": cfg["gine_raw"]["hidden"],
                     "n_parameters": int(out.get("n_parameters", 0)),
                     "seconds": float(out.get("seconds", 0.0)),
                     "wall_seconds": float(out.get("seconds", 0.0)),
                     "best_valid_rmse": out.get("best", {}).get("valid_rmse_raw"),
                     "final_valid_rmse": out.get("final", {}).get("valid_rmse_raw"),
                     "best_epoch": out.get("best", {}).get("epoch"),
                     "commit": C.git_commit(), "device": args.device,
                     "recipe": {k: v for k, v in out.get("recipe", {}).items()} if
                     isinstance(out.get("recipe"), Mapping) else None})
        C.write_json(manifest, rows)
        print(json.dumps({"GINE_RAW": out.get("best", {}).get("valid_rmse_raw")}))


def _deadline() -> float | None:
    cfg = C.load_config()
    stop = cfg["budget"].get("stop_training_unix")
    if stop:
        return float(stop)
    return None


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _run_path(arm: str) -> Path:
    for name in (f"lipo_{arm}_s0", f"lipo_{arm}"):
        path = C.results_root() / name
        if path.is_dir():
            return path
    return C.results_root() / f"lipo_{arm}"


def _load_run(arm: str) -> dict[str, Any] | None:
    path = _run_path(arm) / "run.json"
    return C.read_json(path) if path.is_file() else None


def main_table() -> dict[str, Any]:
    cfg = C.load_config()
    std = float(C.read_json(C.analysis_dir() / "structure_record.json")["scaler"]["std"])
    delta = float(cfg["decisions"]["delta_std_factor"]) * std
    band = float(cfg["decisions"]["band_std_factor"]) * std
    rows: dict[str, Any] = {}
    for arm in ("FULL_WORD_OP", "LOCAL_WORD_OP", "ATOMIC_OP_CAP", "ADD_WORD", "FP_KRR"):
        run = _load_run(arm)
        if run is None:
            rows[arm] = {"available": False}
            continue
        best = run.get("best") or {}
        rows[arm] = {
            "available": True,
            "valid_rmse": best.get("valid_rmse_raw") if "valid_rmse_raw" in best
            else run.get("valid_rmse_raw"),
            "valid_mae": best.get("valid_mae_raw") if "valid_mae_raw" in best
            else run.get("valid_mae_raw"),
            "fit_rmse": run.get("fit_rmse_raw"),
            "fit_mae": run.get("fit_mae_raw"),
            "n_parameters": run.get("n_parameters"),
            "seconds": run.get("seconds"),
            "epochs": run.get("epochs"),
            "best_epoch": best.get("epoch"),
        }
    gine = _load_run("GINE_RAW_REF")
    if gine is not None:
        g = gine.get("best") or {}
        rows["GINE_RAW_REF"] = {"available": True,
                                "valid_rmse": g.get("valid_rmse_raw", gine.get("valid_rmse_raw")),
                                "valid_mae": g.get("valid_mae_raw", gine.get("valid_mae_raw")),
                                "fit_rmse": gine.get("fit_rmse_raw"),
                                "n_parameters": gine.get("n_parameters"),
                                "seconds": gine.get("seconds"),
                                "epochs": gine.get("epochs")}
    out = {"delta": delta, "band": band, "fit_label_std": std, "rows": rows}
    r_f = rows["FULL_WORD_OP"].get("valid_rmse")
    r_l = rows["LOCAL_WORD_OP"].get("valid_rmse")
    r_a = rows["ATOMIC_OP_CAP"].get("valid_rmse")
    r_add = rows["ADD_WORD"].get("valid_rmse")
    strong = [x for x in (rows.get("FP_KRR", {}).get("valid_rmse"),
                          rows.get("GINE_RAW_REF", {}).get("valid_rmse")) if x is not None]
    if None in (r_f, r_l, r_a, r_add):
        decision = "INCOMPLETE"
        detail = "a required operator arm or the scalar reference is missing"
    else:
        beats_controls = (r_l - r_f >= delta) and (r_a - r_f >= delta) and (r_add - r_f >= delta)
        competitive = bool(strong) and (r_f <= min(strong) + band)
        if beats_controls and competitive:
            decision = "GREEN"
            detail = "FULL beats LOCAL/ATOMIC/ADD by delta and is competitive with the strong reference"
        elif beats_controls:
            decision = "YELLOW"
            detail = "FULL beats LOCAL/ATOMIC/ADD by delta but is behind the strong reference by more than band"
        elif competitive:
            decision = "BASELINE_SIGNAL_ONLY"
            detail = "FULL is competitive but does not beat both LOCAL and ATOMIC by delta"
        else:
            decision = "RED"
            detail = "no operator advantage and no competitive score"
    out["decision"] = decision
    out["decision_detail"] = detail
    out["comparisons"] = {
        "full_minus_local": None if None in (r_f, r_l) else r_l - r_f,
        "full_minus_atomic": None if None in (r_f, r_a) else r_a - r_f,
        "full_minus_add": None if None in (r_f, r_add) else r_add - r_f,
        "full_minus_best_strong": None if None in (strong + [r_f]) else r_f - min(strong),
    }
    return out


def cmd_analyze(args) -> None:
    out = main_table()
    out["analyzed_at_unix"] = time.time()
    out["commit"] = C.git_commit()
    C.write_json(C.analysis_dir() / "summary.json", C.jsonable(out))
    if out["decision"] == "GREEN":
        C.write_json(C.analysis_dir() / "green_unlock.json",
                     {"decision": "GREEN", "at_unix": time.time(),
                      "source": "analysis/summary.json"})
    print(json.dumps({k: out[k] for k in ("decision", "comparisons")}, indent=2))


def cmd_candidates(args) -> None:
    """Freeze the candidate/weight list before the H labels are opened."""
    summary = C.read_json(C.analysis_dir() / "summary.json")
    rows = []
    for arm in ("FULL_WORD_OP", "LOCAL_WORD_OP", "ATOMIC_OP_CAP", "GINE_RAW_REF",
                "ADD_WORD", "FP_KRR"):
        path = _run_path(arm) / "best.pt"
        preds = _run_path(arm) / "predictions_best.npz"
        rows.append({"arm": arm,
                     "weight": str(path.relative_to(C.ROUND_DIR)) if path.is_file() else None,
                     "weight_sha256": C.sha256_file(path) if path.is_file() else None,
                     "predictions": str(preds.relative_to(C.ROUND_DIR)) if preds.is_file() else None,
                     "predictions_sha256": C.sha256_file(preds) if preds.is_file() else None,
                     "valid_rmse": summary["rows"].get(arm, {}).get("valid_rmse")})
    out = {"protocol_id": C.load_config()["protocol_id"], "decision": summary["decision"],
           "frozen_at_unix": time.time(), "commit": C.git_commit(), "arms": rows}
    C.write_json(C.analysis_dir() / "candidates_frozen.json", C.jsonable(out))
    print(json.dumps({"n_arms": len(rows), "decision": out["decision"]}, indent=2))


def _predict(arm: str, split_name: str, data) -> tuple[np.ndarray, str]:
    """Best-checkpoint predictions for one arm on one split."""
    import torch

    import wo_train as T

    path = _run_path(arm) / "best.pt"
    if not path.is_file():
        raise SystemExit(f"missing checkpoint {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(payload["state"]["node.0.weight"].shape[0])
    model = T.build_model(C.load_config(), data, arm, hidden=hidden)
    model.load_state_dict(payload["state"])
    model.eval()
    ev = T.evaluate(model, data[split_name], "cpu")
    return ev["pred_raw"], ev["row_ids"]


def cmd_holdout(args) -> None:
    if not (C.analysis_dir() / "candidates_frozen.json").is_file():
        raise SystemExit("freeze the candidate list (candidates) before opening the H labels")
    cfg = C.load_config()
    import wo_train as T

    data = T.load_all()
    h = data["h"]
    if h.labels.size == 0:
        raise SystemExit("H labels are not available in this cache")
    rows: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for arm in ("FULL_WORD_OP", "LOCAL_WORD_OP", "ATOMIC_OP_CAP", "GINE_RAW_REF"):
        try:
            pred, row_ids = _predict(arm, "h", data)
        except SystemExit as exc:
            rows[arm] = {"available": False, "reason": str(exc)}
            continue
        if not np.array_equal(row_ids.astype(np.int64), h.row_ids.astype(np.int64)):
            raise SystemExit(f"row id order mismatch for {arm}")
        preds[arm] = pred
        rows[arm] = {"available": True,
                     "rmse": C.rmse(pred, h.labels), "mae": C.mae(pred, h.labels),
                     "n": int(len(pred))}
    try:
        import wo_baselines as B
        pred = B.gine_predict(h, "cpu")
        preds["GINE_RAW_REF"] = pred
        rows["GINE_RAW_REF"] = {"available": True, "rmse": C.rmse(pred, h.labels),
                                "mae": C.mae(pred, h.labels), "n": int(len(pred))}
        np.savez_compressed(_run_path("GINE_RAW_REF") / "predictions_h.npz",
                            row_ids=h.row_ids, pred_raw=pred, target_raw=h.labels)
    except Exception as exc:
        rows["GINE_RAW_REF"] = {"available": False, "reason": str(exc)}
    for arm in ("ADD_WORD", "FP_KRR"):
        p = _run_path(arm) / "predictions_best.npz"
        state = _run_path(arm) / "run.json"
        if not p.is_file():
            rows[arm] = {"available": False}
            continue
        z = np.load(p)
        best = C.read_json(state)
        rows[arm] = {"available": True, "rmse": None, "mae": None,
                     "note": "H predictions are produced by the holdout-h refs step"}
    pair = _pair_eval(data, preds)
    table = _table_eval(data, preds)
    boot = _bootstrap(data, preds)
    out = {"n_h_molecules": int(h.n_molecules), "rows": rows, "pair": pair,
           "table": table, "bootstrap": boot, "at_unix": time.time()}
    C.write_json(C.analysis_dir() / "holdout.json", C.jsonable(out))
    print(json.dumps({"rows": {k: v.get("rmse") for k, v in rows.items()},
                      "pair": pair.get("n_pairs")}, indent=2))


def _refs_holdout(args) -> None:
    """ADD_WORD / FP_KRR predictions on H, using the already-fit coefficients."""
    import wo_train as T

    data = T.load_all()
    out = {}
    add = C.read_json(C.run_dir("lipo", "ADD_WORD") / "run.json")
    beta = np.asarray(add["beta"], dtype=np.float64)
    h = data["h"]
    pred = (add["beta0"] + h.z_std.astype(np.float64) @ beta) * h.target_std + h.target_mean
    out["ADD_WORD"] = {"rmse": C.rmse(pred, h.labels), "mae": C.mae(pred, h.labels)}
    np.savez_compressed(C.run_dir("lipo", "ADD_WORD") / "predictions_h.npz",
                        row_ids=h.row_ids, pred_raw=pred, target_raw=h.labels)
    krr = C.read_json(C.run_dir("lipo", "FP_KRR") / "run.json")
    if not krr.get("unavailable"):
        fp = C.run_dir("lipo", "FP_KRR") / "predictions_h.npz"
        if fp.is_file():
            z = np.load(fp)
            out["FP_KRR"] = {"rmse": C.rmse(z["pred_raw"], z["target_raw"]),
                             "mae": C.mae(z["pred_raw"], z["target_raw"])}
    idx = C.analysis_dir() / "holdout.json"
    payload = C.read_json(idx) if idx.is_file() else {}
    payload.setdefault("rows", {}).update(out)
    C.write_json(idx, C.jsonable(payload))
    print(json.dumps(out, indent=2))


def _frozen_pairs(data):
    """The frozen evaluable H pairs (Section 2.1), as row-index arrays."""
    import pandas as pd

    pairs = data["split"]["evaluable_pairs"]
    h = data["h"]
    pos = {int(r): i for i, r in enumerate(h.row_ids.tolist())}
    a = np.asarray([int(p["row_a"]) for p in pairs], dtype=np.int64)
    b = np.asarray([int(p["row_b"]) for p in pairs], dtype=np.int64)
    keep = np.asarray([bool(int(x) in pos and int(y) in pos) for x, y in zip(a, b)])
    a, b = a[keep], b[keep]
    transform = np.asarray([str(p["transform_id"]) for p, k in zip(pairs, keep) if k])
    core = np.asarray([str(p["core_id"]) for p, k in zip(pairs, keep) if k])
    ia = np.asarray([pos[int(x)] for x in a.tolist()], dtype=np.int64)
    ib = np.asarray([pos[int(x)] for x in b.tolist()], dtype=np.int64)
    return {"a": ia, "b": ib, "transform": transform, "core": core,
            "n_frozen": int(len(pairs))}


def _pair_eval(data, preds) -> dict[str, Any]:
    fp = _frozen_pairs(data)
    h = data["h"]
    y_a = h.labels[fp["a"]]
    y_b = h.labels[fp["b"]]
    delta = y_b - y_a
    out: dict[str, Any] = {"n_pairs": int(len(fp["a"])), "n_transforms":
                           int(len(set(fp["transform"].tolist()))), "rows": {}}
    for arm, pred in preds.items():
        d = pred[fp["b"]] - pred[fp["a"]]
        out["rows"][arm] = {
            "mae": C.mae(d, delta), "rmse": C.rmse(d, delta),
            "direction_accuracy": float(np.mean(np.sign(d) == np.sign(delta))),
            "pearson": float(np.corrcoef(d, delta)[0, 1]) if len(d) > 2 else None,
            "mean_pred_delta": float(np.mean(d)), "mean_true_delta": float(np.mean(delta)),
        }
    return out


def _table_eval(data, preds) -> dict[str, Any]:
    fp = _frozen_pairs(data)
    h = data["h"]
    dy = h.labels[fp["b"]] - h.labels[fp["a"]]
    transforms = sorted(set(fp["transform"].tolist()))
    rows = {}
    for arm, pred in preds.items():
        dp = pred[fp["b"]] - pred[fp["a"]]
        per_true, per_pred = [], []
        for t in transforms:
            sel = fp["transform"] == t
            per_true.append(float(np.mean(dy[sel])))
            per_pred.append(float(np.mean(dp[sel])))
        per_true = np.asarray(per_true)
        per_pred = np.asarray(per_pred)
        rows[arm] = {"table_mae": float(np.mean(np.abs(per_pred - per_true))),
                     "table_rmse": float(np.sqrt(np.mean((per_pred - per_true) ** 2))),
                     "n_transforms": len(transforms)}
    return {"n_pairs": int(len(dy)), "n_transforms": len(transforms), "rows": rows}


def _bootstrap(data, preds, n_boot: int = 500) -> dict[str, Any]:
    rng = np.random.default_rng(20261121)
    fp = _frozen_pairs(data)
    h = data["h"]
    dy = h.labels[fp["b"]] - h.labels[fp["a"]]
    comps = fp["core"]
    uniq = sorted(set(comps.tolist()))
    groups = {c: np.where(comps == c)[0] for c in uniq}
    out: dict[str, Any] = {"n_boot": n_boot, "n_components": len(uniq), "rows": {}}
    for arm, pred in preds.items():
        dp = pred[fp["b"]] - pred[fp["a"]]
        stats = []
        for _ in range(n_boot):
            pick = rng.choice(len(uniq), size=len(uniq), replace=True)
            idx = np.concatenate([groups[uniq[int(i)]] for i in pick])
            stats.append(np.mean(np.abs(dp[idx] - dy[idx])))
        stats = np.asarray(stats)
        out["rows"][arm] = {"point_mae": float(np.mean(np.abs(dp - dy))),
                            "lo": float(np.quantile(stats, 0.025)),
                            "hi": float(np.quantile(stats, 0.975))}
    return out


def cmd_manifest(args) -> None:
    cfg = C.load_config()
    rows = C.read_json(C.analysis_dir() / "run_manifest.json") \
        if (C.analysis_dir() / "run_manifest.json").is_file() else []
    out = {"protocol_id": cfg["protocol_id"], "commit": C.git_commit(),
           "reference_commit": cfg["reference_commit"],
           "regime": C.environment_snapshot(),
           "budget": cfg["budget"], "runs": rows,
           "total_gpu_hours": float(sum(r.get("gpu_hours") or 0.0 for r in rows)),
           "total_wall_seconds": float(sum(r.get("wall_seconds") or 0.0 for r in rows))}
    C.write_json(C.analysis_dir() / "run_manifest.json", C.jsonable(out["runs"]))
    C.write_json(C.analysis_dir() / "run_manifest_frozen.json", C.jsonable(out))
    print(json.dumps({k: out[k] for k in ("total_gpu_hours", "total_wall_seconds")}, indent=2))


# ---------------------------------------------------------------------------
# structural cases (Section 10)
# ---------------------------------------------------------------------------


def cmd_cases(args) -> None:
    import wo_probes as P
    import wo_train as T

    data = T.load_all()
    out = P.cases(data, n=4)
    C.write_json(C.analysis_dir() / "cases.json", C.jsonable(out))
    print(json.dumps({"n": len(out)}, indent=2))


def cmd_unlock_h(args) -> None:
    """One-shot H label unlock; refuses unless the candidate list is frozen."""
    frozen = C.analysis_dir() / "candidates_frozen.json"
    if not frozen.is_file():
        raise SystemExit("candidates_frozen.json is missing: no H label access")
    import wo_train as T

    data = T.load_all()
    if (C.data_dir() / "wo_h_labels.npz").is_file():
        raise SystemExit("H labels are already unlocked")
    payload = C.dataset_cache("ogbg-mollipo")
    rows = np.asarray(data["split"]["h_row_ids"], dtype=np.int64)
    labels = np.asarray(payload["labels"], dtype=np.float32)[rows]
    C.save_npz_atomic(C.data_dir() / "wo_h_labels.npz", {"labels": labels,
                                                         "row_ids": rows})
    C.write_json(C.analysis_dir() / "h_unlock.json",
                 {"at_unix": time.time(), "commit": C.git_commit(),
                  "n_rows": int(len(rows)), "candidates_frozen_sha256":
                  C.sha256_file(frozen), "labels_sha256": C.sha256_bytes(labels.tobytes())})
    print(json.dumps({"unlocked": int(len(rows))}, indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("checks")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_checks)

    p = sub.add_parser("timing")
    p.add_argument("--warm", type=int, default=2)
    p.add_argument("--timed", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_timing)

    p = sub.add_parser("train")
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="ogbg-mollipo")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--respect-deadline", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("refs")
    p.add_argument("--ref", default="all", choices=("all", "add", "krr", "gine"))
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--device", default="cuda")
    p.add_argument("--respect-deadline", action="store_true")
    p.set_defaults(func=cmd_refs)

    p = sub.add_parser("analyze")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("candidates")
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("unlock-h")
    p.set_defaults(func=cmd_unlock_h)

    p = sub.add_parser("holdout")
    p.set_defaults(func=cmd_holdout)

    p = sub.add_parser("holdout-refs")
    p.set_defaults(func=_refs_holdout)

    p = sub.add_parser("manifest")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("cases")
    p.set_defaults(func=cmd_cases)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
