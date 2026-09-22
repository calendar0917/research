#!/usr/bin/env python
"""TCCD-v6 POST gain decomposition runner.

Stages:

* ``audit``   : label-free A/D reconstructibility audit on the frozen TCCD-v5 cache.
* ``gate0``   : data-free + real-cache correctness checks.  Must pass before Stage A.
* ``stageA``  : frozen-screen decomposition arms on seed 0 (FULL reused from v5).
* ``stageA-recon-seed1`` : single authorized paired seed for the RECON nonlinearity
                            ambiguity interval only.

Formal remote execution is GPU1-only; official ZINC test is never loaded.
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
from tracks.ksvd.code import tccd_v6 as V
from tracks.ksvd.code.run_tccd_v0 import internal_split

OUT_DIR = V.RESULTS_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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


def _load_state(path: Path, device: str):
    return V._torch().load(path, map_location=device, weights_only=True)


def _v5_reference() -> dict[str, Any]:
    if not V.V5_STAGE_A_JSON.exists():
        raise SystemExit(f"missing frozen TCCD-v5 Stage-A result: {V.V5_STAGE_A_JSON}")
    payload = json.loads(V.V5_STAGE_A_JSON.read_text(encoding="utf-8"))
    if payload.get("official_test_loaded") is not False:
        raise SystemExit("TCCD-v5 result does not certify official_test_loaded == false")
    if payload.get("verdict") != "FAIL":
        raise SystemExit("TCCD-v5 result is not the frozen STOP result expected by TCCD-v6")
    return payload


def _provenance() -> dict[str, Any]:
    v5 = _v5_reference()
    if not V.V2_BEST_CHECKPOINT.exists():
        raise SystemExit(f"missing frozen TCCD-v2 checkpoint: {V.V2_BEST_CHECKPOINT}")
    v2_sha = V._sha256_file(V.V2_BEST_CHECKPOINT)
    if v2_sha != V.V2_BEST_CHECKPOINT_SHA256:
        raise SystemExit("TCCD-v2 checkpoint SHA mismatch")
    sha: dict[str, Any] = {
        "v2_best_checkpoint": str(V.V2_BEST_CHECKPOINT.relative_to(V.REPO_ROOT)),
        "v2_best_checkpoint_sha256": v2_sha,
        "v5_stage_a_json": str(V.V5_STAGE_A_JSON.relative_to(V.REPO_ROOT)),
        "v5_stage_a_sha256": V._sha256_file(V.V5_STAGE_A_JSON),
    }
    for name, path in (("v5_post_soup_sha256", V.V5_POST_SOUP), ("v5_post_best_sha256", V.V5_POST_BEST)):
        sha[name] = V._sha256_file(path) if path.exists() else None
    cached = v5["results"]
    return {
        "v2_commit": V5.TCCD_V2_COMMIT,
        **sha,
        "recorded": {
            "BASE_best": float(cached["BASE"]["best_valid_mae"]),
            "BASE_soup": float(cached["BASE"]["soup_valid_mae"]),
            "FULL_best": float(cached["POST"]["best_valid_mae"]),
            "FULL_soup": float(cached["POST"]["soup_valid_mae"]),
        },
        "official_test_loaded": False,
    }


def _arm_summary(res: V.ArmResult) -> dict[str, Any]:
    return {
        "best_valid_mae": float(res.best_valid),
        "best_epoch": int(res.best_epoch),
        "soup_valid_mae": None if res.soup_valid is None else float(res.soup_valid),
        "soup_members": list(res.soup_members),
        "init_checksum": res.init_checksum,
        "wall_s": float(res.wall_s),
    }


def _arm_paths(arm: str, seed: int) -> dict[str, Path]:
    tag = f"arm_{arm}_seed{seed}"
    return {
        "summary": OUT_DIR / f"{tag}.json",
        "soup": OUT_DIR / f"{tag}_soup.pt",
        "best": OUT_DIR / f"{tag}_best.pt",
    }


def _reuse_arm_summary(arm: str, seed: int, commit: str) -> dict[str, Any] | None:
    paths = _arm_paths(arm, seed)
    if not all(p.exists() for p in paths.values()):
        return None
    try:
        payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        payload.get("protocol") == V.PROTOCOL_VERSION
        and payload.get("commit") == commit
        and int(payload.get("seed", -1)) == int(seed)
        and payload.get("arm") == arm
    ):
        return payload
    return None


def _write_arm_summary(arm: str, seed: int, commit: str, summary: dict[str, Any]) -> None:
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "stageA_arm",
        "arm": arm,
        "seed": int(seed),
        "commit": commit,
        "summary": summary,
        "official_test_loaded": False,
    }
    V.write_json(_arm_paths(arm, seed)["summary"], payload)


def _train_or_reuse(
    arm: str,
    cache,
    y,
    train_idx,
    dev_idx,
    device: str,
    *,
    seed: int,
    commit: str,
    args,
    log=print,
) -> dict[str, Any]:
    paths = _arm_paths(arm, seed)
    existing = _reuse_arm_summary(arm, seed, commit)
    if existing is not None and not getattr(args, "force", False):
        summary = dict(existing["summary"])
        summary["reused"] = True
        log(f"[stageA:{arm}] reusing completed arm {paths['summary'].name}")
        return summary
    res = V.train_arm(
        cache,
        y,
        train_idx,
        dev_idx,
        device,
        arm,
        seed=seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=log,
    )
    summary = _arm_summary(res)
    summary["reused"] = False
    _save_state(paths["soup"], res.state_soup)
    _save_state(paths["best"], res.state_best)
    _write_arm_summary(arm, seed, commit, summary)
    return summary


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def audit(args) -> int:
    cache = V.load_label_free_cache()
    result = V.label_free_audit(cache)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "label_free_reconstructibility_audit",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "cache_meta": cache.meta,
        "descriptor_block_dims": dict(V.BLOCK_DIMS),
        "result": result,
        "official_test_loaded": False,
    }
    V.write_json(OUT_DIR / "label_free_audit.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if result["all_pass"] and result["frozen_mask_matches"] else 1


def gate0(args) -> int:
    cache = V.load_label_free_cache()
    data_free = V.gate0_data_free_checks(args.device)
    real = V.gate0_real_cache_checks(cache, args.device)
    all_pass = bool(data_free.get("data_free_all_pass") and real.get("real_all_pass"))
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "descriptor_block_dims": dict(V.BLOCK_DIMS),
        "descriptor_block_slices": V.descriptor_block_slices(),
        "parameter_accounting": V.parameter_accounting(),
        "data_free": data_free,
        "real_cache": real,
        "official_test_loaded": False,
        "all_pass": all_pass,
    }
    V.write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if all_pass else 1


def _full_reuse_verification(cache, y, dev_idx, device: str, recorded: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"official_test_loaded": False}
    torch = V._torch()
    if V.V5_POST_SOUP.exists():
        state = torch.load(V.V5_POST_SOUP, map_location=device, weights_only=True)
        mae = V.evaluate_saved_state("full_nl", state, cache, y, dev_idx, device)
        out["soup_recomputed_mae"] = float(mae)
        out["soup_recorded_mae"] = float(recorded["FULL_soup"])
        out["soup_delta"] = float(mae - recorded["FULL_soup"])
        out["soup_pass"] = bool(abs(out["soup_delta"]) <= 1e-5)
    else:
        out["soup_pass"] = False
    if V.V5_POST_BEST.exists():
        state = torch.load(V.V5_POST_BEST, map_location=device, weights_only=True)
        mae = V.evaluate_saved_state("full_nl", state, cache, y, dev_idx, device)
        out["best_recomputed_mae"] = float(mae)
        out["best_recorded_mae"] = float(recorded["FULL_best"])
        out["best_delta"] = float(mae - recorded["FULL_best"])
        out["best_pass"] = bool(abs(out["best_delta"]) <= 1e-5)
    else:
        out["best_pass"] = False
    out["pass"] = bool(out["soup_pass"] and out["best_pass"])
    return out


def stage_a(args) -> int:
    commit = _commit()
    seed = int(args.seed)
    if seed != 0:
        raise SystemExit("stageA is seed 0 only; the paired seed is stageA-recon-seed1")
    _peak_reset(args.device)
    started = time.time()

    cache = V.load_label_free_cache()
    audit_result = V.label_free_audit(cache)
    if not audit_result["all_pass"] or not audit_result["frozen_mask_matches"]:
        raise SystemExit("label-free audit failed or froze a different mask; STOP")
    y = V.load_cache_labels()
    train_idx, dev_idx = internal_split(cache.n_graphs)
    provenance = _provenance()
    recorded = provenance["recorded"]
    full_check = _full_reuse_verification(cache, y, dev_idx, args.device, recorded)
    if not full_check.get("pass"):
        raise SystemExit("FULL-NL reuse verification failed; Gate 0.1 is not satisfied; STOP")

    log = print
    log(f"[stageA] cache={cache.n_graphs} graphs; train={len(train_idx)} dev={len(dev_idx)}")
    log(f"[stageA] audit recon={audit_result['recon_mask_nonzero']} novel={audit_result['novel_mask_nonzero']}")
    log(f"[stageA] FULL reuse: soup {full_check.get('soup_recomputed_mae')} vs {recorded['FULL_soup']}")

    arms: dict[str, dict[str, Any]] = {}
    for arm in ("recon_nl", "recon_linfact", "novel_nl"):
        log(f"[stageA] training arm {arm} (seed {seed})")
        arms[arm] = _train_or_reuse(
            arm, cache, y, train_idx, dev_idx, args.device, seed=seed, commit=commit, args=args, log=log
        )

    recon_nl = arms["recon_nl"]["soup_valid_mae"]
    recon_linfact = arms["recon_linfact"]["soup_valid_mae"]
    novel_nl = arms["novel_nl"]["soup_valid_mae"]

    seed1_gap = None
    seed1_path = OUT_DIR / "stageA_recon_seed1.json"
    if seed1_path.exists():
        seed1_payload = json.loads(seed1_path.read_text(encoding="utf-8"))
        if int(seed1_payload.get("seed", -1)) == 1:
            seed1_gap = float(seed1_payload["gap_seed1"])

    decision = V.decomposition_decision(
        recon_nl_soup=float(recon_nl),
        recon_linfact_soup=float(recon_linfact),
        novel_nl_soup=float(novel_nl),
        recon_linfact_gap_seed1=seed1_gap,
    )

    arm_payloads = {
        "BASE": {
            "best_valid_mae": float(recorded["BASE_best"]),
            "soup_valid_mae": float(recorded["BASE_soup"]),
            "source": "frozen TCCD-v5 Stage-A result (reused)",
        },
        "FULL_NL": {
            "best_valid_mae": float(full_check.get("best_recomputed_mae", recorded["FULL_best"])),
            "soup_valid_mae": float(full_check.get("soup_recomputed_mae", recorded["FULL_soup"])),
            "source": "frozen TCCD-v5 POST checkpoint (reused)",
            "init_checksum": V.FULL_INIT_CHECKSUM,
        },
        "RECON_NL": {**arms["recon_nl"]},
        "RECON_LINFACT": {**arms["recon_linfact"]},
        "NOVEL_NL": {**arms["novel_nl"]},
    }

    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "A_frozen_decomposition",
        "commit": commit,
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": seed,
        "split_seed": T.SPLIT_SEED,
        "n_train": int(len(train_idx)),
        "n_dev": int(len(dev_idx)),
        "provenance": provenance,
        "full_reuse_verification": full_check,
        "descriptor": dict(V.BLOCK_DIMS),
        "reconstructibility": audit_result,
        "recon_mask_nonzero": int(V.RECON_MASK.sum()),
        "novel_mask_nonzero": int(V.NOVEL_MASK.sum()),
        "recon_coordinates": np.nonzero(V.RECON_MASK)[0].tolist(),
        "novel_coordinates": np.nonzero(V.NOVEL_MASK)[0].tolist(),
        "parameter_accounting": V.parameter_accounting(),
        "primary_metric": "top5_soup",
        "results": arm_payloads,
        "decision": decision,
        "case": decision["case"],
        "case_status": "NEEDS_SEED1_RECON" if decision["needs_seed1_recon"] else "FINAL",
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / "stageA_seed0.json", payload)
    V.write_json(OUT_DIR / "stageA_decision.json", {"protocol": V.PROTOCOL_VERSION, "case": decision["case"], **decision})
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def stage_a_recon_seed1(args) -> int:
    commit = _commit()
    seed = 1
    seed0_path = OUT_DIR / "stageA_seed0.json"
    if not seed0_path.exists():
        raise SystemExit("stageA_recon_seed1 requires stageA_seed0.json")
    seed0 = json.loads(seed0_path.read_text(encoding="utf-8"))
    if seed0.get("case_status") != "NEEDS_SEED1_RECON":
        raise SystemExit("paired RECON seed is authorized only in the registered ambiguity interval")
    if int(args.seed) != 1:
        raise SystemExit("stageA-recon-seed1 is seed 1 only")

    _peak_reset(args.device)
    started = time.time()
    cache = V.load_label_free_cache()
    y = V.load_cache_labels()
    train_idx, dev_idx = internal_split(cache.n_graphs)
    log = print
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("recon_nl", "recon_linfact"):
        log(f"[stageA-seed1] training arm {arm} (seed {seed})")
        arms[arm] = _train_or_reuse(
            arm, cache, y, train_idx, dev_idx, args.device, seed=seed, commit=commit, args=args, log=log
        )
    gap0 = float(seed0["results"]["RECON_LINFACT"]["soup_valid_mae"]) - float(seed0["results"]["RECON_NL"]["soup_valid_mae"])
    gap1 = float(arms["recon_linfact"]["soup_valid_mae"]) - float(arms["recon_nl"]["soup_valid_mae"])
    mean_gap = 0.5 * (gap0 + gap1)
    case = "R2" if mean_gap >= V.NONLIN_GAP_SEED1_MEAN else "R1"

    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "A_recon_nonlinearity_paired_seed",
        "commit": commit,
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": seed,
        "split_seed": T.SPLIT_SEED,
        "n_train": int(len(train_idx)),
        "n_dev": int(len(dev_idx)),
        "results_seed1": {
            "RECON_NL": {**arms["recon_nl"]},
            "RECON_LINFACT": {**arms["recon_linfact"]},
        },
        "gap_seed0": gap0,
        "gap_seed1": gap1,
        "mean_gap": mean_gap,
        "threshold_mean_gap": V.NONLIN_GAP_SEED1_MEAN,
        "case": case,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
    }
    V.write_json(OUT_DIR / "stageA_recon_seed1.json", payload)
    V.write_json(OUT_DIR / "stageA_decision.json", {"protocol": V.PROTOCOL_VERSION, "case": case, "mean_gap": mean_gap})
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["audit", "gate0", "stageA", "stageA-recon-seed1"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--physical-gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    p.add_argument("--force", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        return audit(args)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "stageA":
        return stage_a(args)
    if args.stage == "stageA-recon-seed1":
        return stage_a_recon_seed1(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
