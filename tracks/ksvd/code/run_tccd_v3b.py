#!/usr/bin/env python
"""TCCD-v3b correct pre-pair local-state bridge runner.

Formal execution is GPU1-only through the repository remote wrapper. The
frozen B-full source model is used only during the explicit cache stage; all
formal bridge training stages require a valid train/valid cache and never
load official ZINC test data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v3b as V

OUT_DIR = V.RESULTS_DIR


def _peak_reset(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> float | None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return None


def _device_provenance(device: str, physical_gpu: int) -> dict[str, Any]:
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


def _save_state(path: Path, state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _load_state(path: Path, device: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location=device, weights_only=True)


def gate0(args: argparse.Namespace) -> int:
    checks = V.gate0_checks(args.device, require_cache=True)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": T.git("rev-parse", "HEAD"),
        "device": str(args.device),
        "gpu": _device_provenance(args.device, args.physical_gpu),
        "source_checkpoint": str(V.STRONG_CHECKPOINT.relative_to(V.REPO_ROOT)),
        "source_checkpoint_sha256": V._sha256_file(V.STRONG_CHECKPOINT),
        "strong_source_commit": V.STRONG_SOURCE_COMMIT,
        "selected_tensor": V.SELECTED_TENSOR,
        "selected_layer": V.SELECTED_LAYER,
        "raw_width": V.SELECTED_WIDTH,
        "checks": checks,
        "official_test_loaded": False,
    }
    V._write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if checks.get("all_pass", False) else 1


def cache(args: argparse.Namespace) -> int:
    started = time.time()
    train_records, valid_records, meta = V.build_bridge_records(
        args.data_root,
        args.device,
        force_embeddings=bool(args.force),
        require_existing_cache=False,
    )
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "cache",
        "commit": T.git("rev-parse", "HEAD"),
        "gpu": _device_provenance(args.device, args.physical_gpu),
        "selected_tensor": V.SELECTED_TENSOR,
        "selected_layer": V.SELECTED_LAYER,
        "raw_width": V.SELECTED_WIDTH,
        "n_train": len(train_records),
        "n_valid": len(valid_records),
        "n_train_patches": int(sum(int(r["n"]) for r in train_records)),
        "n_valid_patches": int(sum(int(r["n"]) for r in valid_records)),
        "metadata": meta,
        "wall_s": float(time.time() - started),
        "official_test_loaded": False,
    }
    V._write_json(OUT_DIR / "cache.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _load_records_for_training(args: argparse.Namespace):
    train_records, valid_records, meta = V.build_bridge_records(
        args.data_root,
        args.device,
        require_existing_cache=True,
    )
    if len(train_records) != 10_000 or len(valid_records) != 1_000:
        raise RuntimeError(
            f"expected official train/valid 10000/1000, got {len(train_records)}/{len(valid_records)}"
        )
    return train_records, valid_records, meta


def _train_arm(args: argparse.Namespace, arm: str) -> int:
    gate_path = OUT_DIR / "gate0.json"
    cache_path = OUT_DIR / "cache.json"
    if not gate_path.exists():
        raise SystemExit("run gate0 before formal bridge training")
    if not cache_path.exists():
        raise SystemExit("run cache before formal bridge training")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if not gate.get("checks", {}).get("all_pass", False):
        raise SystemExit("Gate 0 failed; formal bridge training blocked")
    if cache.get("official_test_loaded") is not False:
        raise SystemExit("cache provenance is invalid")

    train_records, valid_records, meta = _load_records_for_training(args)
    train_idx = list(range(len(train_records)))
    valid_idx = list(range(len(valid_records)))
    model = V.build_model(arm, seed=args.seed).to(args.device)
    _peak_reset(args.device)
    started = time.time()
    result = V.train_bridge(
        model,
        train_records,
        valid_records,
        train_idx,
        valid_idx,
        args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=print,
    )
    state_best_path = OUT_DIR / f"{arm}_seed{args.seed}_best.pt"
    state_soup_path = OUT_DIR / f"{arm}_seed{args.seed}_soup.pt"
    _save_state(state_best_path, result.state_best)
    if result.state_soup is not None:
        _save_state(state_soup_path, result.state_soup)

    out: dict[str, Any] = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": f"train_{arm}",
        "arm": arm,
        "commit": T.git("rev-parse", "HEAD"),
        "seed": int(args.seed),
        "gpu": _device_provenance(args.device, args.physical_gpu),
        "input_representation": V.SELECTED_TENSOR,
        "selected_layer": V.SELECTED_LAYER,
        "raw_width": V.SELECTED_WIDTH,
        "adapter": "Linear(64,64)",
        "trainable_strong_encoder": False,
        "strong_checkpoint": str(V.STRONG_CHECKPOINT.relative_to(V.REPO_ROOT)),
        "strong_checkpoint_sha256": V._sha256_file(V.STRONG_CHECKPOINT),
        "strong_source_commit": V.STRONG_SOURCE_COMMIT,
        "data_protocol": meta,
        "n_train": len(train_records),
        "n_valid": len(valid_records),
        "best_valid_mae": result.best_valid,
        "best_epoch": result.best_epoch,
        "soup_valid_mae": result.soup_valid,
        "soup_members": result.soup_members,
        "regularization": result.regularization,
        "wall_s": float(time.time() - started),
        "peak_mem_mb": _peak_mb(args.device),
        "state_best": str(state_best_path.relative_to(V.REPO_ROOT)),
        "state_soup": str(state_soup_path.relative_to(V.REPO_ROOT)) if result.state_soup is not None else None,
        "official_test_loaded": False,
    }
    V._write_json(OUT_DIR / f"{arm}_seed{args.seed}.json", out)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


def diagnostics(args: argparse.Namespace) -> int:
    proto_path = OUT_DIR / f"prototype_seed{args.seed}.json"
    if not proto_path.exists():
        raise SystemExit("run prototype training before diagnostics")
    train_records, valid_records, meta = _load_records_for_training(args)
    model = V.build_model("prototype", seed=args.seed).to(args.device)
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    best_state = _load_state(V.REPO_ROOT / proto["state_best"], args.device)
    model.load_state_dict(best_state)
    valid_idx = list(range(len(valid_records)))
    real_best = V.evaluate_mae(model, valid_records, valid_idx, args.device, batch=64)
    shuffle_best = V.evaluate_mae(
        model,
        valid_records,
        valid_idx,
        args.device,
        batch=64,
        shuffle=True,
        seed=args.seed,
    )
    soup_mae = None
    if proto.get("state_soup"):
        soup_state = _load_state(V.REPO_ROOT / proto["state_soup"], args.device)
        model.load_state_dict(soup_state)
        soup_mae = V.evaluate_mae(model, valid_records, valid_idx, args.device, batch=64)
        model.load_state_dict(best_state)
    vocab = V.vocabulary_diagnostics(model, valid_records, valid_idx, args.device)
    dense_path = OUT_DIR / f"dense_seed{args.seed}.json"
    dense = json.loads(dense_path.read_text(encoding="utf-8")) if dense_path.exists() else {}
    strong_proto = float(real_best)
    strong_dense = float(dense.get("best_valid_mae", np.nan))
    delta_local = float(V.TCCD_V2_PROTO_BEST - strong_proto)
    delta_proto = float(strong_proto - strong_dense) if np.isfinite(strong_dense) else None
    delta_comp = float(shuffle_best - real_best)
    decision = V.classify_decision(strong_proto)
    out = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "diagnostics",
        "commit": T.git("rev-parse", "HEAD"),
        "seed": int(args.seed),
        "gpu": _device_provenance(args.device, args.physical_gpu),
        "selected_tensor": V.SELECTED_TENSOR,
        "selected_layer": V.SELECTED_LAYER,
        "raw_width": V.SELECTED_WIDTH,
        "strong_proto_best_valid_mae": strong_proto,
        "strong_proto_soup_valid_mae": soup_mae,
        "strong_dense_best_valid_mae": strong_dense,
        "strong_dense_soup_valid_mae": dense.get("soup_valid_mae"),
        "shuffle_best_valid_mae": shuffle_best,
        "delta_local": delta_local,
        "decision": decision,
        "delta_proto_strong": delta_proto,
        "prototype_interpretation": V.interpretation_proto_gap(delta_proto) if delta_proto is not None else None,
        "delta_comp_strong": delta_comp,
        "composition_interpretation": V.interpretation_composition(delta_comp),
        "vocabulary": vocab,
        "data_protocol": meta,
        "official_test_loaded": False,
    }
    V._write_json(OUT_DIR / "diagnostics.json", out)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["gate0", "cache", "dense", "prototype", "diagnostics"])
    parser.add_argument("--data-root", type=Path, default=V.REPO_ROOT / "data/ZINC")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=T.BATCH)
    parser.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=T.PATIENCE)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "cache":
        return cache(args)
    if args.stage == "dense":
        return _train_arm(args, "dense")
    if args.stage == "prototype":
        return _train_arm(args, "prototype")
    if args.stage == "diagnostics":
        return diagnostics(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
