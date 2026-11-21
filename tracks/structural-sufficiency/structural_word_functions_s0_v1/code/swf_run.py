"""Command entry points for the structural-word-function round."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import swf_common as C


def _args_dict(args) -> dict:
    """Arguments without the bound subcommand handler (not JSON serialisable)."""
    return {k: v for k, v in vars(args).items() if k != "func"}


def _ledger(name: str) -> C.Ledger:
    return C.Ledger(C.analysis_dir() / "resource_ledger.jsonl")


def _write_run(run_dir: Path, payload: Mapping[str, Any]) -> None:
    C.write_json(run_dir / "run.json", payload)


def _manifest_update(entry: Mapping[str, Any]) -> None:
    path = C.analysis_dir() / "run_manifest.json"
    payload = C.read_json(path) if path.is_file() else {"runs": []}
    payload["runs"] = [r for r in payload["runs"] if r.get("run_id") != entry.get("run_id")]
    payload["runs"].append(dict(entry))
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    C.write_json(path, payload)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def cmd_prepare(args) -> int:
    import swf_cache as K
    import swf_data as D

    cfg = C.load_config()
    ledger = _ledger("prepare")
    t0 = time.perf_counter()
    stats = K.prepare(verbose=True)
    seconds = time.perf_counter() - t0
    ledger.record("prepare", seconds)

    raw = D.load_raw_cache(cfg)
    sizes = {
        "base_bytes": stats["files"]["base"]["bytes"],
        "sent_bytes": stats["files"]["sent"]["bytes"],
        "base_sha256": stats["files"]["base"]["sha256"],
        "sent_sha256": stats["files"]["sent"]["sha256"],
    }
    total_frames = stats["n_frames"]
    payload = {
        "dataset": cfg["dataset"],
        "raw_cache_sha256": raw.get("cache_sha256"),
        "molecules": stats["n_molecules"],
        "split_sizes": stats["split_sizes"],
        "words": stats["n_words"],
        "frames": stats["n_frames"],
        "frames_per_word": total_frames / max(1, stats["n_words"]),
        "sentences": stats["n_sentences"],
        "covers": stats["n_covers"],
        "frame_pair_rows": stats["n_rows"],
        "rows_per_cover": stats["n_rows"] / max(1, stats["n_covers"]),
        "unique_word_frame_inputs": stats["n_unique_win"],
        "word_frame_dedup_rate": 1.0 - stats["n_unique_win"] / max(1, total_frames),
        "unique_relation_inputs": stats["n_unique_rel"],
        "relation_dedup_rate": 1.0 - stats["n_unique_rel"] / max(1, stats["n_rows"]),
        "template_train_df_molecules": stats["template_train_df_molecules"],
        "template_active": stats["template_active"],
        "mu_T": stats["mu_T"],
        "nu_k": stats["nu_k"],
        "sentence_size_mean_per_molecule": stats["sentence_size_mean_per_molecule"],
        "target_scaler": stats["target_scaler"],
        "files": sizes,
        "build_seconds": seconds,
        "enumeration_seconds": stats["build_seconds"],
    }
    C.write_json(C.analysis_dir() / "cache_stats.json", payload)
    _manifest_update({
        "run_id": "prepare",
        "kind": "cache",
        "wall_seconds": seconds,
        "commit": C.git_commit(),
        "cache_sha256": {"base": sizes["base_sha256"], "sent": sizes["sent_sha256"]},
    })
    print(json.dumps(C.jsonable(payload), indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def cmd_checks(args) -> int:
    import swf_checks as CH

    cfg = C.load_config()
    ledger = _ledger("checks")
    t0 = time.perf_counter()
    device = args.device or ("cuda" if _cuda() else "cpu")
    prefix = None
    if args.prefix_ckpt:
        prefix = Path(args.prefix_ckpt)
    elif (C.results_root() / "lipo_WORD_PREFIX_s0" / "prefix_best.pt").is_file():
        prefix = C.results_root() / "lipo_WORD_PREFIX_s0" / "prefix_best.pt"
    out = CH.run_checks(cfg, device=device, prefix_ckpt=prefix)
    out["device"] = device
    out["commit"] = C.git_commit()
    out["prefix_ckpt"] = str(prefix) if prefix else None
    seconds = time.perf_counter() - t0
    ledger.record("checks", seconds)
    C.write_json(C.analysis_dir() / "checks.json", out)
    summary = {
        "identity_ok": out["identity"]["split_sizes_official"] and out["identity"]["row_ids_are_arange"],
        "binding_algebra_pass": out["binding_algebra"]["pass"],
        "relabel_fp64_pass": out.get("relabel_invariance", {}).get("fp64_pass"),
        "relabel_fp32_gpu_pass": out.get("relabel_invariance", {}).get("fp32_gpu_pass"),
        "cover_swap_scatter_pass": out.get("cover_swap_and_scatter", {}).get("pass"),
        "branch_pass": out.get("branch", {}).get("pass"),
        "seconds": seconds,
    }
    C.write_json(C.analysis_dir() / "checks_summary.json", summary)
    print(json.dumps(C.jsonable(summary), indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# training commands
# ---------------------------------------------------------------------------


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _run_id(arm: str, seed: int, explicit: str | None) -> str:
    return explicit or f"lipo_{arm}_s{seed}"


def cmd_train_prefix(args) -> int:
    import swf_cache as K
    import swf_train as T

    cfg = C.load_config()
    cfg_recipe = cfg["recipe"]
    run_id = _run_id("WORD_PREFIX", int(cfg_recipe["seed"]), args.run_id)
    run_dir = C.run_dir(run_id)
    ledger = _ledger(run_id)
    device = args.device or ("cuda" if _cuda() else "cpu")
    cache = K.Cache(need_sentences=False)
    torch_info = C.enable_tf32(bool(cfg_recipe["tf32"]))
    t0 = time.perf_counter()
    result = T.train_prefix(
        cfg=cfg, cache=cache, device=device, out_dir=run_dir, horizon=int(args.epochs),
        run_id=run_id, ledger=ledger,
        deadline_epoch=args.deadline_epoch,
    )
    wall = time.perf_counter() - t0
    ledger.record("train", wall, run_id=run_id, arm="WORD_PREFIX")
    payload = {
        **result,
        "kind": "train",
        "device": device,
        "environment": C.environment_snapshot(),
        "config": cfg,
        "args": _args_dict(args),
        "commit": C.git_commit(),
        "cache_stats": cache.sample_sizes(),
        "tf32": torch_info,
        "wall_seconds_total": wall,
        "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
    }
    _write_run(run_dir, payload)
    _manifest_update({
        "run_id": run_id, "kind": "train", "arm": "WORD_PREFIX", "device": device,
        "wall_seconds": wall, "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
        "commit": C.git_commit(), "best_epoch": result["best_epoch"],
        "best_valid_rmse": result["best_valid_rmse"],
    })
    print(json.dumps(C.jsonable({
        "run_id": run_id, "best_epoch": result["best_epoch"],
        "best": result["best"], "steady_epoch_seconds_median": result["steady_epoch_seconds_median"],
        "wall_seconds": wall,
    }), indent=2))
    return 0


def _ensure_branch_init(cfg, cache, device: str) -> tuple[Path, dict[str, Any]]:
    import swf_model as M
    import swf_train as T

    path = C.results_root() / "_branch_init" / "branch_init.pt"
    meta_path = C.results_root() / "_branch_init" / "branch_init.json"
    prefix_ckpt = C.results_root() / "lipo_WORD_PREFIX_s0" / "prefix_best.pt"
    if not prefix_ckpt.is_file():
        raise FileNotFoundError(f"prefix checkpoint missing: {prefix_ckpt}")
    if path.is_file() and meta_path.is_file():
        return path, C.read_json(meta_path)
    out_dir = C.results_root() / "_branch_init"
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    prefix_state = torch.load(prefix_ckpt, map_location="cpu")
    info = T.build_branch_init(cfg=cfg, cache=cache, device=device, prefix_state=prefix_state, out_dir=out_dir)
    info["prefix_ckpt"] = str(prefix_ckpt)
    info["prefix_sha256"] = C.sha256_file(prefix_ckpt)
    C.write_json(meta_path, info)
    return path, info


def cmd_train_branch(args) -> int:
    import swf_cache as K
    import swf_train as T

    cfg = C.load_config()
    arm = str(args.arm)
    run_id = _run_id(arm, int(cfg["recipe"]["seed"]), args.run_id)
    run_dir = C.run_dir(run_id)
    ledger = _ledger(run_id)
    device = args.device or ("cuda" if _cuda() else "cpu")
    cache = K.Cache(need_sentences=True)
    init_ckpt, init_info = _ensure_branch_init(cfg, cache, device)
    prefix_run = C.read_json(C.results_root() / "lipo_WORD_PREFIX_s0" / "run.json")
    t0 = time.perf_counter()
    result = T.train_branch(
        cfg=cfg, cache=cache, device=device, out_dir=run_dir, arm=arm,
        epochs=int(args.epochs), run_id=run_id, ledger=ledger,
        prefix_record=prefix_run, parent_prefix_epoch=int(prefix_run["best_epoch"]),
        init_ckpt=init_ckpt, deadline_epoch=args.deadline_epoch,
    )
    wall = time.perf_counter() - t0
    ledger.record("train", wall, run_id=run_id, arm=arm)
    payload = {
        **result,
        "kind": "train",
        "device": device,
        "environment": C.environment_snapshot(),
        "config": cfg,
        "args": _args_dict(args),
        "commit": C.git_commit(),
        "cache_stats": cache.sample_sizes(),
        "branch_init": {k: init_info.get(k) for k in ("sha256", "state_hash", "prefix_ckpt", "prefix_sha256")},
        "wall_seconds_total": wall,
        "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
    }
    _write_run(run_dir, payload)
    _manifest_update({
        "run_id": run_id, "kind": "train", "arm": arm, "device": device,
        "wall_seconds": wall, "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
        "commit": C.git_commit(), "best_epoch": result["best_epoch"],
        "best_valid_rmse": result["best_valid_rmse"],
        "parent_prefix_epoch": result["parent_prefix_epoch"],
    })
    print(json.dumps(C.jsonable({
        "run_id": run_id, "arm": arm, "best_epoch": result["best_epoch"],
        "best": result["best"], "steady_epoch_seconds_median": result["steady_epoch_seconds_median"],
        "wall_seconds": wall,
    }), indent=2))
    return 0


def cmd_train_ref(args) -> int:
    import swf_train as T

    cfg = C.load_config()
    run_id = args.run_id or "lipo_GINE_RAW_REF"
    run_dir = C.run_dir(run_id)
    ledger = _ledger(run_id)
    device = args.device or ("cuda" if _cuda() else "cpu")
    t0 = time.perf_counter()
    result = T.train_gine_ref(
        cfg=cfg, device=device, out_dir=run_dir, run_id=run_id,
        deadline_epoch=args.deadline_epoch,
    )
    wall = time.perf_counter() - t0
    ledger.record("train", wall, run_id=run_id, arm="GINE_RAW_REF")
    payload = {
        **result,
        "kind": "train",
        "device": device,
        "environment": C.environment_snapshot(),
        "args": _args_dict(args),
        "commit": C.git_commit(),
        "wall_seconds_total": wall,
        "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
    }
    _write_run(run_dir, payload)
    _manifest_update({
        "run_id": run_id, "kind": "train", "arm": "GINE_RAW_REF", "device": device,
        "wall_seconds": wall, "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
        "commit": C.git_commit(), "best_epoch": result["best_epoch"],
        "best_valid_rmse": result["best_valid_rmse"],
    })
    print(json.dumps(C.jsonable({
        "run_id": run_id, "best_epoch": result["best_epoch"], "best": result["best"],
        "steady_epoch_seconds_median": result["steady_epoch_seconds_median"],
        "wall_seconds": wall, "n_parameters": result["n_parameters"],
    }), indent=2))
    return 0


def cmd_timing(args) -> int:
    """Measure the steady-state epoch cost of one arm (formal-cost probe)."""
    import shutil

    import swf_cache as K
    import swf_train as T

    cfg = C.load_config()
    arm = str(args.arm)
    run_id = f"timing_{arm}"
    run_dir = C.run_dir(run_id)
    ledger = _ledger(run_id)
    device = args.device or ("cuda" if _cuda() else "cpu")
    cache = K.Cache(need_sentences=(arm != "WORD_PREFIX"))
    t0 = time.perf_counter()
    if arm == "WORD_PREFIX":
        result = T.train_prefix(
            cfg=cfg, cache=cache, device=device, out_dir=run_dir,
            horizon=int(args.epochs), run_id=run_id, ledger=ledger,
        )
    else:
        init_ckpt, _info = _ensure_branch_init(cfg, cache, device)
        prefix_run = C.read_json(C.results_root() / "lipo_WORD_PREFIX_s0" / "run.json")
        result = T.train_branch(
            cfg=cfg, cache=cache, device=device, out_dir=run_dir, arm=arm,
            epochs=int(args.epochs), run_id=run_id, ledger=ledger,
            prefix_record=prefix_run, parent_prefix_epoch=int(prefix_run["best_epoch"]),
            init_ckpt=init_ckpt,
        )
    wall = time.perf_counter() - t0
    ledger.record("timing", wall, run_id=run_id, arm=arm)
    times = result["epoch_seconds"]
    steady = sorted(times[2:5]) if len(times) >= 5 else sorted(times)
    median = float(np.median(steady))
    timing = cfg["recipe"]["timing"]
    safety = float(timing["safety"])
    long_horizon = int(cfg["recipe"]["prefix"]["max_epochs"]) if arm == "WORD_PREFIX" else int(cfg["recipe"]["branch"]["epochs"])
    budget_minutes = float(timing["prefix_budget_minutes"] if arm == "WORD_PREFIX" else timing["branch_budget_minutes"])
    predicted_minutes = median * long_horizon * safety / 60.0
    short_horizon = int(timing["prefix_short_epochs"] if arm == "WORD_PREFIX" else timing["branch_short_epochs"])
    predicted_short_minutes = median * short_horizon * safety / 60.0
    payload = {
        "arm": arm,
        "device": device,
        "probe_epochs": int(args.epochs),
        "epoch_seconds": times,
        "steady_epoch_seconds_median": median,
        "safety": safety,
        "long_horizon": long_horizon,
        "predicted_minutes_long": predicted_minutes,
        "budget_minutes": budget_minutes,
        "exceeds_budget": bool(predicted_minutes > budget_minutes),
        "short_horizon": short_horizon,
        "predicted_minutes_short": predicted_short_minutes,
        "recommended_horizon": int(short_horizon if predicted_minutes > budget_minutes else long_horizon),
        "wall_seconds": wall,
        "max_vram_mb": result.get("max_vram_mb"),
        "commit": C.git_commit(),
        "environment": C.environment_snapshot(),
    }
    C.write_json(C.analysis_dir() / f"timing_{arm}.json", payload)
    _manifest_update({
        "run_id": run_id, "kind": "timing", "arm": arm, "device": device,
        "wall_seconds": wall, "gpu_hours": wall / 3600.0 if device.startswith("cuda") else 0.0,
        "commit": C.git_commit(),
    })
    if args.cleanup:
        shutil.rmtree(run_dir, ignore_errors=True)
    print(json.dumps(C.jsonable(payload), indent=2))
    return 0


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def cmd_baselines(args) -> int:
    import swf_baselines as B
    import swf_cache as K

    cfg = C.load_config()
    ledger = _ledger("baselines")
    only = args.only
    out: dict[str, Any] = {}
    if only in (None, "count"):
        cache = K.Cache(need_sentences=False)
        t0 = time.perf_counter()
        out["count_ridge"] = B.count_ridge(cfg=cfg, cache=cache)
        ledger.record("count_ridge", time.perf_counter() - t0)
    if only in (None, "fp"):
        cache = K.Cache(need_sentences=False)
        try:
            t0 = time.perf_counter()
            out["fp_krr"] = B.fp_krr(cfg=cfg, cache=cache)
            ledger.record("fp_krr", time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            out["fp_krr"] = {"method": "FP_KRR", "available": False, "error": repr(exc)}
            print(f"[FP_KRR] unavailable: {exc!r}", flush=True)
    out["commit"] = C.git_commit()
    out["environment"] = C.environment_snapshot()
    C.write_json(C.analysis_dir() / "baselines.json", out)
    _manifest_update({
        "run_id": "baselines",
        "kind": "cpu_baseline",
        "wall_seconds": sum(v for v in (ledger.summary()["wall_seconds_by_stage"].values())),
        "commit": C.git_commit(),
    })
    print(json.dumps(C.jsonable({
        k: (v.get("selected") if isinstance(v, dict) else None) for k, v in out.items()
    }), indent=2))
    return 0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def cmd_summarize(args) -> int:
    import swf_analyze as A

    out = A.summarize(write=True)
    print(json.dumps(C.jsonable(out.get("status", {})), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swf_run")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("checks")
    p.add_argument("--device", default=None)
    p.add_argument("--prefix-ckpt", default=None)
    p.set_defaults(func=cmd_checks)

    p = sub.add_parser("train-prefix")
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--run-id", default=None)
    p.add_argument("--deadline-epoch", type=float, default=None)
    p.set_defaults(func=cmd_train_prefix)

    p = sub.add_parser("train-ref")
    p.add_argument("--device", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--deadline-epoch", type=float, default=None)
    p.set_defaults(func=cmd_train_ref)

    p = sub.add_parser("train-branch")
    p.add_argument("--arm", required=True, choices=["FULL", "BAG_CONT", "EARLY_POOL", "FLAT_CONTEXT"])
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--run-id", default=None)
    p.add_argument("--deadline-epoch", type=float, default=None)
    p.set_defaults(func=cmd_train_branch)

    p = sub.add_parser("timing")
    p.add_argument("--arm", default="FULL", choices=["FULL", "BAG_CONT", "EARLY_POOL", "FLAT_CONTEXT", "WORD_PREFIX"])
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--cleanup", action="store_true")
    p.set_defaults(func=cmd_timing)

    p = sub.add_parser("baselines")
    p.add_argument("--only", default=None, choices=["count", "fp"])
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("summarize")
    p.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
