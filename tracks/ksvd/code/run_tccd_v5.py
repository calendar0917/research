#!/usr/bin/env python
"""TCCD-v5 occurrence-preserving pair nonlinearity runner.

Formal remote execution is GPU1-only and official ZINC test is never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v5 as V
from tracks.ksvd.code.run_tccd_v0 import internal_split

OUT_DIR = V.RESULTS_DIR


def _peak_reset(device: str) -> None:
    torch = V._torch()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> float | None:
    torch = V._torch()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return None


def _gpu_info(device: str, physical_gpu: int) -> dict[str, Any]:
    torch = V._torch()
    return {
        "device": str(device),
        "physical_gpu_requested": int(physical_gpu),
        "logical_gpu_name": (
            torch.cuda.get_device_name(0)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "official_test_loaded": False,
    }


def _records(args, log=print):
    records, meta = V.V2.load_train_records()
    if args.limit:
        log(f"[debug] truncating records to {args.limit}; formal runs must omit --limit")
        records = records[: int(args.limit)]
    return list(records), meta


def _save_state(path: Path, state: Any) -> None:
    if state is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    V._torch().save(state, path)


def _load_state(path: Path, device: str):
    return V._torch().load(path, map_location=device, weights_only=True)


def _arm_summary(res: V.StageAResult) -> dict[str, Any]:
    return {
        "best_valid_mae": float(res.best_valid),
        "best_epoch": int(res.best_epoch),
        "soup_valid_mae": None if res.soup_valid is None else float(res.soup_valid),
        "soup_members": list(res.soup_members),
        "init_checksum": res.init_checksum,
        "wall_s": float(res.wall_s),
    }


def _stage_a_model(placement: str, state: Any, device: str):
    model = V.StageAModelFactory.build(placement, seed=0).to(device)
    if state is not None:
        model.load_state_dict(state)
    return model


def gate0(args) -> int:
    checks = V.gate0_checks()
    grads = V.gradient_checks()
    checks.update({f"grad_{k}": v for k, v in grads.items()})
    checks["all_pass"] = bool(checks.get("all_pass", False) and grads.get("stage_a_ok", False))
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "pair_descriptor_width": V.pair_descriptor_width(),
        "pair_mlp_parameters": V.pair_mlp_parameter_count(),
        "phi": "Linear(d_p,64)->ReLU->Linear(64,16)",
        "official_test_loaded": False,
        "checks": checks,
    }
    V.write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if checks.get("all_pass", False) else 1


def stage_a(args) -> int:
    seed0 = None
    if int(args.seed) == 1:
        seed0_path = OUT_DIR / "stageA_seed0.json"
        if not seed0_path.exists():
            raise SystemExit("seed-1 Stage A requires stageA_seed0.json")
        seed0 = json.loads(seed0_path.read_text(encoding="utf-8"))
        if seed0.get("verdict") != "AMBIGUOUS_NEEDS_SEED1":
            raise SystemExit("seed-1 Stage A is authorized only after seed-0 ambiguity")

    records, meta = _records(args)
    train_idx, dev_idx = internal_split(len(records))
    _peak_reset(args.device)
    started = time.time()
    cache = V.build_or_load_pair_cache(records, list(range(len(records))), "all", args.device, force=args.force)
    stats = cache.pair_statistics()

    base_res = V.train_frozen_base_reader(
        cache, train_idx, dev_idx, args.device, seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch=args.batch, log=print,
    )
    post_res = V.train_stage_a(
        cache, records, train_idx, dev_idx, args.device, "post", seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch=args.batch, log=print,
    )
    pre_res = V.train_stage_a(
        cache, records, train_idx, dev_idx, args.device, "pre", seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch=args.batch, log=print,
    )

    pre_soup_state = pre_res.state_soup
    pre_model = _stage_a_model("pre", pre_soup_state, args.device)
    pre_shuffle_soup = V._evaluate_stage_a(pre_model, cache, records, list(dev_idx), args.device, shuffle=True)
    pre_model_best = _stage_a_model("pre", pre_res.state_best, args.device)
    pre_shuffle_best = V._evaluate_stage_a(pre_model_best, cache, records, list(dev_idx), args.device, shuffle=True)

    results = {
        "BASE": _arm_summary(base_res),
        "POST": _arm_summary(post_res),
        "PRE": _arm_summary(pre_res),
        "PRE_SHUFFLE": {
            "soup_valid_mae": float(pre_shuffle_soup),
            "best_valid_mae": float(pre_shuffle_best),
        },
    }
    results["BASE"]["feature_dim"] = V.frozen_base_dim()
    results["POST"]["feature_dim"] = V.base_representation_dim()
    results["PRE"]["feature_dim"] = V.base_representation_dim()

    delta_place = float(post_res.soup_valid - pre_res.soup_valid)
    delta_add = float(base_res.soup_valid - pre_res.soup_valid)
    delta_shuffle = float(pre_shuffle_soup - pre_res.soup_valid)
    verdict = V.stage_a_decision(delta_place, delta_add, delta_shuffle, args.seed, seed0=seed0)

    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "A_frozen_screen",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "split_seed": T.SPLIT_SEED,
        "n_train": len(train_idx),
        "n_dev": len(dev_idx),
        "records_meta": meta,
        "v2_checkpoint": str(V.V2_BEST_CHECKPOINT.relative_to(V.REPO_ROOT)),
        "v2_checkpoint_sha256": V.V2_BEST_CHECKPOINT_SHA256,
        "pair_statistics": stats,
        "phi_hidden": V.PHI_HIDDEN,
        "phi_out": V.PHI_OUT,
        "pair_mlp_parameters": V.pair_mlp_parameter_count(),
        "primary_metric": "top5_soup",
        "results": results,
        "delta_place": delta_place,
        "delta_add": delta_add,
        "delta_shuffle": delta_shuffle,
        "delta_place_best": float(post_res.best_valid - pre_res.best_valid),
        "delta_add_best": float(base_res.best_valid - pre_res.best_valid),
        "delta_shuffle_best": float(pre_shuffle_best - pre_res.best_valid),
        "thresholds": {
            "strong_pass": V.STAGE_A_PASS,
            "fail": V.STAGE_A_FAIL,
            "paired_mean_pass": V.STAGE_A_MEAN_PASS,
        },
        "verdict": verdict,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / f"stageA_seed{args.seed}.json", payload)
    _save_state(OUT_DIR / f"stageA_post_seed{args.seed}_soup.pt", post_res.state_soup)
    _save_state(OUT_DIR / f"stageA_pre_seed{args.seed}_soup.pt", pre_res.state_soup)
    _save_state(OUT_DIR / f"stageA_post_seed{args.seed}_best.pt", post_res.state_best)
    _save_state(OUT_DIR / f"stageA_pre_seed{args.seed}_best.pt", pre_res.state_best)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if verdict in {"PASS", "AMBIGUOUS_NEEDS_SEED1"} else 1


def stage_b(args) -> int:
    stage_a_path = OUT_DIR / f"stageA_seed{args.seed}.json"
    if not stage_a_path.exists():
        raise SystemExit("run Stage A before Stage B")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    if stage_a.get("verdict") != "PASS":
        raise SystemExit("Stage A did not pass; Stage B is blocked")

    seed0 = None
    if int(args.seed) == 1:
        seed0_path = OUT_DIR / "stageB_seed0.json"
        if not seed0_path.exists():
            raise SystemExit("seed-1 Stage B requires stageB_seed0.json")
        seed0 = json.loads(seed0_path.read_text(encoding="utf-8"))
        if seed0.get("verdict") != "AMBIGUOUS_NEEDS_SEED1":
            raise SystemExit("seed-1 Stage B is authorized only after seed-0 ambiguity")

    records, meta = _records(args)
    train_idx, dev_idx = internal_split(len(records))
    _peak_reset(args.device)
    started = time.time()

    post_res = V.train_e2e(
        "post", records, records, train_idx, dev_idx, args.device, seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch=args.batch, log=print,
    )
    post_wall = post_res.wall_s
    pre_res = V.train_e2e(
        "pre", records, records, train_idx, dev_idx, args.device, seed=args.seed,
        max_epochs=args.max_epochs, patience=args.patience, batch=args.batch, log=print,
    )

    pre_model = V.build_e2e_model("pre", seed=args.seed).to(args.device)
    pre_model.load_state_dict(pre_res.state_soup)
    real_soup = V.evaluate_mae(pre_model, records, list(dev_idx), args.device, batch=64)
    shuffle_soup = V.evaluate_mae(pre_model, records, list(dev_idx), args.device, batch=64, shuffle=True)
    pre_model.load_state_dict(pre_res.state_best)
    real_best = V.evaluate_mae(pre_model, records, list(dev_idx), args.device, batch=64)
    shuffle_best = V.evaluate_mae(pre_model, records, list(dev_idx), args.device, batch=64, shuffle=True)

    delta_place = float(post_res.soup_valid - pre_res.soup_valid)
    delta_base = float(V.TCCD_V2_SOUP - pre_res.soup_valid)
    delta_shuffle = float(shuffle_soup - real_soup)
    verdict = V.stage_b_decision(delta_place, delta_base, args.seed, seed0=seed0)

    vocab = V.vocabulary_diagnostics(pre_model, records, list(dev_idx), args.device)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "B_end_to_end",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "split_seed": T.SPLIT_SEED,
        "n_train": len(train_idx),
        "n_dev": len(dev_idx),
        "records_meta": meta,
        "base_reference": {
            "commit": V.TCCD_V2_COMMIT,
            "best_valid_mae": V.TCCD_V2_BEST,
            "soup_valid_mae": V.TCCD_V2_SOUP,
            "checkpoint_reused": True,
        },
        "pair_mlp_parameters": V.pair_mlp_parameter_count(),
        "feature_dim": V.model_feature_dim(),
        "POST_E2E": _arm_summary(post_res),
        "PRE_E2E": _arm_summary(pre_res),
        "pre_shuffle": {
            "real_soup": float(real_soup),
            "shuffle_soup": float(shuffle_soup),
            "real_best": float(real_best),
            "shuffle_best": float(shuffle_best),
        },
        "delta_place_e2e": delta_place,
        "delta_base_e2e": delta_base,
        "delta_shuffle_e2e": delta_shuffle,
        "thresholds": {
            "strong_pass": V.STAGE_B_STRONG_PASS,
            "fail": V.STAGE_B_FAIL,
            "paired_mean_pass": V.STAGE_B_MEAN_PASS,
            "shuffle_pass": V.STAGE_B_SHUFFLE_PASS,
        },
        "vocabulary": vocab,
        "verdict": verdict,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "post_wall_s": post_wall,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / f"stageB_seed{args.seed}.json", payload)
    _save_state(OUT_DIR / f"stageB_post_seed{args.seed}_soup.pt", post_res.state_soup)
    _save_state(OUT_DIR / f"stageB_pre_seed{args.seed}_soup.pt", pre_res.state_soup)
    _save_state(OUT_DIR / f"stageB_post_seed{args.seed}_best.pt", post_res.state_best)
    _save_state(OUT_DIR / f"stageB_pre_seed{args.seed}_best.pt", pre_res.state_best)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if verdict in {"PASS", "AMBIGUOUS_NEEDS_SEED1"} else 1


def official(args) -> int:
    stage_b_path = OUT_DIR / f"stageB_seed{args.seed}.json"
    if not stage_b_path.exists():
        raise SystemExit("run Stage B before official-valid authorization")
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    if stage_b.get("verdict") != "PASS":
        raise SystemExit("Stage B did not pass; official-valid is blocked")
    final_soup = float(stage_b["PRE_E2E"]["soup_valid_mae"])
    if not V.official_authorized(final_soup):
        raise SystemExit("official-valid route not authorized by preregistered thresholds")

    train_records, valid_records, meta = V.official_records(args.data_root)
    _peak_reset(args.device)
    started = time.time()
    res = V.train_e2e(
        "pre",
        train_records,
        valid_records,
        list(range(len(train_records))),
        list(range(len(valid_records))),
        args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=print,
    )
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "official_valid_full_run",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "placement": "pre",
        "n_official_train": len(train_records),
        "n_official_valid": len(valid_records),
        "records_meta": meta,
        "best_valid_mae": float(res.best_valid),
        "best_epoch": int(res.best_epoch),
        "soup_valid_mae": None if res.soup_valid is None else float(res.soup_valid),
        "soup_members": list(res.soup_members),
        "canonical_gpu1_baseline": V.CANONICAL_GPU1_BASELINE,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / f"official_seed{args.seed}.json", payload)
    _save_state(OUT_DIR / f"official_pre_seed{args.seed}_best.pt", res.state_best)
    _save_state(OUT_DIR / f"official_pre_seed{args.seed}_soup.pt", res.state_soup)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gate0", "stageA", "stageB", "official"])
    p.add_argument("--data-root", type=Path, default=V.REPO_ROOT / "data/ZINC")
    p.add_argument("--device", default="cpu")
    p.add_argument("--physical-gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "stageA":
        return stage_a(args)
    if args.stage == "stageB":
        return stage_b(args)
    if args.stage == "official":
        return official(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
