#!/usr/bin/env python
"""TCCD-v4 higher-order assembly audit runner.

Formal remote execution is GPU1-only and official ZINC test is never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v4 as V
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


def gate0(args) -> int:
    checks = V.gate0_checks()
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "fixed_permutation_seed": V.PERM_SEED,
        "existing_relations_unchanged": True,
        "official_test_loaded": False,
        "checks": checks,
    }
    V.write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if checks.get("all_pass", False) else 1


def _stage_a_arm(name: str, cache: dict[str, Any], train_idx: np.ndarray, dev_idx: np.ndarray, device: str, seed: int):
    if name == "BASE":
        xtr, xdv = cache["train"]["base"], cache["dev"]["base"]
    elif name == "REAL2":
        xtr = np.concatenate([cache["train"]["base"], cache["train"]["real2"]], axis=1)
        xdv = np.concatenate([cache["dev"]["base"], cache["dev"]["real2"]], axis=1)
    elif name == "MIS2":
        xtr = np.concatenate([cache["train"]["base"], cache["train"]["mis2"]], axis=1)
        xdv = np.concatenate([cache["dev"]["base"], cache["dev"]["mis2"]], axis=1)
    else:
        raise ValueError(name)
    ytr = cache["train"]["y"]
    ydv = cache["dev"]["y"]
    return V.train_frozen_reader(xtr, ytr, xdv, ydv, device, seed=seed, log=print)


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
    train_cache = V.build_or_load_frozen_cache(records, train_idx, "internal_train", args.device, force=args.force)
    dev_cache = V.build_or_load_frozen_cache(records, dev_idx, "internal_dev", args.device, force=args.force)
    cache = {"train": train_cache, "dev": dev_cache}
    results: dict[str, Any] = {}
    for arm in ("BASE", "REAL2", "MIS2"):
        res = _stage_a_arm(arm, cache, train_idx, dev_idx, args.device, args.seed)
        results[arm] = {
            "best_valid_mae": res.best_valid,
            "best_epoch": res.best_epoch,
            "soup_valid_mae": res.soup_valid,
            "soup_members": res.soup_members,
            "feature_dim": int(cache["train"]["base"].shape[1] if arm == "BASE" else cache["train"]["base"].shape[1] + cache["train"]["real2"].shape[1]),
            "wall_s": res.wall_s,
        }
    delta_topo = float(results["MIS2"]["best_valid_mae"] - results["REAL2"]["best_valid_mae"])
    delta_add = float(results["BASE"]["best_valid_mae"] - results["REAL2"]["best_valid_mae"])
    verdict = V.stage_a_decision(delta_topo, delta_add, args.seed, seed0=seed0)
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
        "fixed_permutation_seed": V.PERM_SEED,
        "results": results,
        "delta_topo": delta_topo,
        "delta_add": delta_add,
        "thresholds": {
            "strong_pass": V.STAGE_A_TOPO_PASS,
            "fail": V.STAGE_A_FAIL,
            "paired_mean_pass": V.STAGE_A_MEAN_PASS,
        },
        "verdict": verdict,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / f"stageA_seed{args.seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if verdict in {"PASS", "AMBIGUOUS_NEEDS_SEED1"} else 1


def _train_e2e(args, max_order: int, records_train, records_dev, train_idx, dev_idx, seed: int):
    model = V.build_model(max_order=max_order, seed=seed).to(args.device)
    _peak_reset(args.device)
    started = time.time()
    result = V.train_model(
        model,
        records_train,
        records_dev,
        list(train_idx),
        list(dev_idx),
        args.device,
        seed=seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=print,
    )
    return model, result, time.time() - started, _peak_mb(args.device)


def stage_b(args) -> int:
    stage_a_path = OUT_DIR / f"stageA_seed{args.seed}.json"
    if not stage_a_path.exists():
        raise SystemExit("run Stage A before Stage B")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    if stage_a.get("verdict") != "PASS":
        raise SystemExit("Stage A did not pass; Stage B and S^3 are blocked")
    records, meta = _records(args)
    train_idx, dev_idx = internal_split(len(records))
    model, res, wall_s, peak_mb = _train_e2e(args, 2, records, records, train_idx, dev_idx, args.seed)
    best_path = OUT_DIR / f"twohop_seed{args.seed}_best.pt"
    soup_path = OUT_DIR / f"twohop_seed{args.seed}_soup.pt"
    _save_state(best_path, res.state_best)
    _save_state(soup_path, res.state_soup)
    valid_idx = list(dev_idx)
    real_best = V.evaluate_mae(model, records, valid_idx, args.device, batch=64)
    mismatch_best = V.evaluate_mae(model, records, valid_idx, args.device, batch=64, mismatch=True)
    real_soup = mismatch_soup = None
    if res.state_soup is not None:
        model.load_state_dict(res.state_soup)
        real_soup = V.evaluate_mae(model, records, valid_idx, args.device, batch=64)
        mismatch_soup = V.evaluate_mae(model, records, valid_idx, args.device, batch=64, mismatch=True)
        if res.state_best is not None:
            model.load_state_dict(res.state_best)
    delta_e2e_soup = float(V.TCCD_V2_SOUP - float(res.soup_valid))
    delta_intervention_best = float(mismatch_best - real_best)
    delta_intervention_soup = None if mismatch_soup is None else float(mismatch_soup - real_soup)
    seed0 = None
    if int(args.seed) == 1:
        seed0_path = OUT_DIR / "stageB_seed0.json"
        if not seed0_path.exists():
            raise SystemExit("seed-1 Stage B requires stageB_seed0.json")
        seed0 = json.loads(seed0_path.read_text(encoding="utf-8"))
        if seed0.get("verdict") != "AMBIGUOUS_NEEDS_SEED1":
            raise SystemExit("seed-1 Stage B is authorized only after seed-0 ambiguity")
    verdict = V.stage_b_decision(delta_e2e_soup, args.seed, seed0=seed0)
    vocab = V.vocabulary_diagnostics(model, records, valid_idx, args.device)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "B_end_to_end_twohop",
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
        "twohop": {
            "best_valid_mae": res.best_valid,
            "best_epoch": res.best_epoch,
            "soup_valid_mae": res.soup_valid,
            "soup_members": res.soup_members,
            "state_best": str(best_path.relative_to(V.REPO_ROOT)),
            "state_soup": str(soup_path.relative_to(V.REPO_ROOT)),
            "regularization": res.regularization,
            "feature_dim": V.model_feature_dim(2),
        },
        "delta_e2e_soup": delta_e2e_soup,
        "delta_intervention_best": delta_intervention_best,
        "delta_intervention_soup": delta_intervention_soup,
        "mismatch_eval": {
            "real_best": real_best,
            "mismatch_best": mismatch_best,
            "real_soup": real_soup,
            "mismatch_soup": mismatch_soup,
            "fixed_permutation_seed": V.PERM_SEED,
        },
        "vocabulary": vocab,
        "thresholds": {
            "strong_pass": V.STAGE_B_STRONG_PASS,
            "fail": V.STAGE_B_FAIL,
            "paired_mean_pass": V.STAGE_B_MEAN_PASS,
        },
        "verdict": verdict,
        "official_test_loaded": False,
        "wall_s": wall_s,
        "peak_mem_mb": peak_mb,
    }
    V.write_json(OUT_DIR / f"stageB_seed{args.seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if verdict in {"PASS", "AMBIGUOUS_NEEDS_SEED1"} else 1


def stage_c(args) -> int:
    stage_b_path = OUT_DIR / f"stageB_seed{args.seed}.json"
    if not stage_b_path.exists():
        raise SystemExit("run Stage B before Stage C")
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    if stage_b.get("verdict") != "PASS":
        raise SystemExit("Stage B did not pass; Stage C is blocked")
    records, meta = _records(args)
    train_idx, dev_idx = internal_split(len(records))
    model, res, wall_s, peak_mb = _train_e2e(args, 3, records, records, train_idx, dev_idx, args.seed)
    best_path = OUT_DIR / f"threehop_seed{args.seed}_best.pt"
    soup_path = OUT_DIR / f"threehop_seed{args.seed}_soup.pt"
    _save_state(best_path, res.state_best)
    _save_state(soup_path, res.state_soup)
    increment = float(stage_b["twohop"]["soup_valid_mae"] - res.soup_valid)
    verdict = "PASS" if increment >= V.STAGE_C_PASS else "SATURATED"
    vocab = V.vocabulary_diagnostics(model, records, list(dev_idx), args.device)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "C_conditional_threehop",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "records_meta": meta,
        "twohop_soup_reference": stage_b["twohop"]["soup_valid_mae"],
        "threehop": {
            "best_valid_mae": res.best_valid,
            "best_epoch": res.best_epoch,
            "soup_valid_mae": res.soup_valid,
            "soup_members": res.soup_members,
            "state_best": str(best_path.relative_to(V.REPO_ROOT)),
            "state_soup": str(soup_path.relative_to(V.REPO_ROOT)),
            "feature_dim": V.model_feature_dim(3),
            "regularization": res.regularization,
        },
        "increment_over_twohop_soup": increment,
        "threshold": V.STAGE_C_PASS,
        "verdict": verdict,
        "vocabulary": vocab,
        "official_test_loaded": False,
        "wall_s": wall_s,
        "peak_mem_mb": peak_mb,
    }
    V.write_json(OUT_DIR / f"stageC_seed{args.seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def vocabulary(args) -> int:
    stage_b_path = OUT_DIR / f"stageB_seed{args.seed}.json"
    if not stage_b_path.exists():
        raise SystemExit("run Stage B before vocabulary diagnostics")
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    state_rel = stage_b["twohop"]["state_best"]
    records, _ = _records(args)
    _, dev_idx = internal_split(len(records))
    model = V.build_model(2, args.seed).to(args.device)
    model.load_state_dict(_load_state(V.REPO_ROOT / state_rel, args.device))
    vocab = V.vocabulary_diagnostics(model, records, list(dev_idx), args.device)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "vocabulary",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "vocabulary": vocab,
        "official_test_loaded": False,
    }
    V.write_json(OUT_DIR / f"vocabulary_seed{args.seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def official(args) -> int:
    stage_b_path = OUT_DIR / f"stageB_seed{args.seed}.json"
    if not stage_b_path.exists():
        raise SystemExit("run Stage B before official-valid authorization")
    stage_b = json.loads(stage_b_path.read_text(encoding="utf-8"))
    final_soup = float(stage_b["twohop"]["soup_valid_mae"])
    if (OUT_DIR / f"stageC_seed{args.seed}.json").exists():
        stage_c = json.loads((OUT_DIR / f"stageC_seed{args.seed}.json").read_text(encoding="utf-8"))
        if stage_c.get("verdict") == "PASS":
            final_soup = float(stage_c["threehop"]["soup_valid_mae"])
    if not V.official_authorized(final_soup):
        raise SystemExit("official-valid route not authorized by preregistered thresholds")
    train_records, valid_records, meta = V.official_records(args.data_root)
    max_order = 3 if (OUT_DIR / f"stageC_seed{args.seed}.json").exists() and json.loads((OUT_DIR / f"stageC_seed{args.seed}.json").read_text()).get("verdict") == "PASS" else 2
    model, res, wall_s, peak_mb = _train_e2e(
        args,
        max_order,
        train_records,
        valid_records,
        list(range(len(train_records))),
        list(range(len(valid_records))),
        args.seed,
    )
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "official_valid_full_run",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "architecture_order": max_order,
        "n_official_train": len(train_records),
        "n_official_valid": len(valid_records),
        "records_meta": meta,
        "best_valid_mae": res.best_valid,
        "best_epoch": res.best_epoch,
        "soup_valid_mae": res.soup_valid,
        "soup_members": res.soup_members,
        "regularization": res.regularization,
        "canonical_gpu1_baseline": V.CANONICAL_GPU1_BASELINE,
        "official_test_loaded": False,
        "wall_s": wall_s,
        "peak_mem_mb": peak_mb,
    }
    V.write_json(OUT_DIR / f"official_seed{args.seed}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gate0", "stageA", "stageB", "stageC", "vocabulary", "official"])
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
    if args.stage == "stageC":
        return stage_c(args)
    if args.stage == "vocabulary":
        return vocabulary(args)
    if args.stage == "official":
        return official(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
