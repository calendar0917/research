#!/usr/bin/env python
"""CENTER-COMP-v0 runner: center-preserving composition diagnostic.

Stages:

* ``gate0``  : data-free and real-cache correctness checks.  Must pass first.
* ``stageA`` : train one CENTER-COMP seed-0 model and evaluate the
               evaluation-only CENTER-BIND-SHUFFLE intervention.

Formal execution is remote A100 GPU1 only; official ZINC valid/test are never
loaded.  Existing TCCD-v2/v5/v6 artifacts are read, never re-run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v5 as V5
from tracks.ksvd.code import tccd_v6 as V6
from tracks.ksvd.code import center_comp_v0 as V
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


def _commit() -> str:
    return T.git("rev-parse", "HEAD")


def _save_state(path: Path, state: Any) -> None:
    if state is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    V._torch().save(state, path)


def gate0(args) -> int:
    cache = V6.load_label_free_cache()
    records, _meta = V5.V2.load_train_records()
    data_free = V.gate0_data_free_checks(args.device)
    real = V.gate0_real_checks(cache, records, args.device)
    all_pass = bool(data_free.get("data_free_all_pass") and real.get("real_all_pass"))
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "parameter_accounting": V.parameter_accounting(),
        "preregistered_thresholds": {"material": V.MATERIAL, "bind_fail": V.BIND_FAIL},
        "data_free": data_free,
        "real_cache": real,
        "official_test_loaded": False,
        "all_pass": all_pass,
    }
    V.write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if all_pass else 1


def stage_a(args) -> int:
    if int(args.seed) != 0:
        raise SystemExit("CENTER-COMP-v0 is a single-seed round; seed 0 only")
    _peak_reset(args.device)
    started = time.time()

    refs = V.frozen_reference_values()
    cache = V6.load_label_free_cache()
    y = V6.load_cache_labels()
    records, _meta = V5.V2.load_train_records()

    # provenance gates
    if V._sha256_file(V.V2_BEST_CHECKPOINT) != V.V2_BEST_CHECKPOINT_SHA256:
        raise SystemExit("frozen TCCD-v2 checkpoint SHA mismatch; STOP")
    if cache.meta.get("v2_checkpoint_sha256") != V.V2_BEST_CHECKPOINT_SHA256:
        raise SystemExit("reused pair cache was not built from the frozen checkpoint; STOP")
    if cache.meta.get("official_test_loaded") is not False:
        raise SystemExit("pair cache does not certify official_test_loaded == false; STOP")

    # real-cache Gate 0 (C re-derivation + descriptor equivalence) before training
    real = V.gate0_real_checks(cache, records, args.device)
    if not real.get("real_all_pass"):
        raise SystemExit(f"real-cache Gate 0 failed: {real}")

    train_idx, dev_idx = internal_split(cache.n_graphs)
    train_idx = list(map(int, train_idx))
    dev_idx = list(map(int, dev_idx))

    res = V.train_center_comp(
        cache,
        y,
        train_idx,
        dev_idx,
        args.device,
        seed=0,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=print,
    )

    # real / shuffle evaluations on both soup and best
    def _load_and_eval(state):
        model = V.CenterCompFactory.build(seed=0).to(args.device)
        model.load_state_dict(state)
        real_pred = V.predict(model, cache, y, dev_idx, args.device, shuffle=False)
        shuf_pred = V.predict(model, cache, y, dev_idx, args.device, shuffle=True)
        return real_pred, shuf_pred

    soup_real, soup_shuf = _load_and_eval(res.state_soup)
    best_real, best_shuf = _load_and_eval(res.state_best)
    rows = cache.rows_for(dev_idx)
    ydev = np.asarray(y[rows], dtype=np.float64)

    mae_center_soup = float(np.abs(soup_real - ydev).mean())
    mae_shuffle_soup = float(np.abs(soup_shuf - ydev).mean())
    mae_center_best = float(np.abs(best_real - ydev).mean())
    mae_shuffle_best = float(np.abs(best_shuf - ydev).mean())

    g_center = float(refs["mae_prev"] - mae_center_soup)
    g_bind = float(mae_shuffle_soup - mae_center_soup)
    decision = V.center_comp_decision(
        g_center=g_center,
        g_bind=g_bind,
        mae_prev=refs["mae_prev"],
        mae_center=mae_center_soup,
        mae_shuffle=mae_shuffle_soup,
    )

    mechanism = {
        "soup": V.prediction_change_report(soup_real, soup_shuf),
        "best": V.prediction_change_report(best_real, best_shuf),
        "material_change_threshold": V.MATERIAL_CHANGE,
    }
    strata = V.size_stratification(cache, dev_idx, y, soup_real, soup_shuf)

    stats = {
        "n_graphs": int(cache.n_graphs),
        "total_pairs": int(sum(cache.pair_count_of_row(r) for r in range(cache.n_graphs))),
        "total_occurrences": int(cache.C.shape[0]),
        "relation_count": int(V.N_REL),
        "pair_descriptor_width": int(V.PAIR_WIDTH),
        "phi": "Linear(197,64)->ReLU->Linear(64,16)",
        "rho": "Linear(80,64)->ReLU->Linear(64,16)",
        "center_operator": "q_i=sum_{j!=i} phi(p_ij); u_i=rho([c_i,q_i]); h_center=sum_i u_i",
        "shuffle": "q_i -> q_{pi(i)} with c_i fixed, deterministic per-graph perm seed 20260922",
    }

    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "A_center_preserving_composition",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": 0,
        "split_seed": T.SPLIT_SEED,
        "n_train": len(train_idx),
        "n_dev": len(dev_idx),
        "v2_checkpoint": str(V.V2_BEST_CHECKPOINT.relative_to(V.REPO_ROOT)),
        "v2_checkpoint_sha256": V.V2_BEST_CHECKPOINT_SHA256,
        "pair_cache": cache.meta,
        "statistics": stats,
        "parameter_accounting": V.parameter_accounting(),
        "init_checksum": res.init_checksum,
        "primary_metric": "top5_soup",
        "results": {
            "CENTER_COMP": {
                "best_valid_mae": mae_center_best,
                "soup_valid_mae": mae_center_soup,
                "best_epoch": int(res.best_epoch),
                "soup_members": list(res.soup_members),
                "init_checksum": res.init_checksum,
                "wall_s": float(res.wall_s),
                "source": "new this round",
            },
            "CENTER_BIND_SHUFFLE": {
                "best_valid_mae": mae_shuffle_best,
                "soup_valid_mae": mae_shuffle_soup,
                "source": "evaluation-only intervention on CENTER_COMP seed 0",
            },
        },
        "frozen_references": refs,
        "g_center": g_center,
        "g_bind": g_bind,
        "gap_to_strong": float(mae_center_soup - V.CANONICAL_GPU1_BASELINE),
        "decision": decision,
        "mechanism": mechanism,
        "size_stratification": strata,
        "real_cache_gate0": real,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / "stageA_seed0.json", payload)
    V.write_json(OUT_DIR / "stageA_decision.json", {"protocol": V.PROTOCOL_VERSION, **decision})
    _save_state(OUT_DIR / "center_comp_seed0_soup.pt", res.state_soup)
    _save_state(OUT_DIR / "center_comp_seed0_best.pt", res.state_best)
    np.save(OUT_DIR / "dev_soup_real.npy", soup_real)
    np.save(OUT_DIR / "dev_soup_shuffle.npy", soup_shuf)
    np.save(OUT_DIR / "dev_best_real.npy", best_real)
    np.save(OUT_DIR / "dev_best_shuffle.npy", best_shuf)
    np.save(OUT_DIR / "dev_indices.npy", np.asarray(dev_idx, dtype=np.int64))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gate0", "stageA"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--physical-gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "stageA":
        return stage_a(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
