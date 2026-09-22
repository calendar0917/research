#!/usr/bin/env python
"""TCCD-v7 runner: explicit normalized prototype / relation moment closure.

Stages:

* ``gate0``    : data-free + real-cache correctness checks (Gate 0 A-G).
* ``stageA``   : frozen-representation closure.  RAW (re-screen for protocol
                 identity), NORM, NORM+MOM linear readers on the fixed internal
                 split, then the preregistered Stage-A decision.
* ``stageB``   : one final end-to-end NORM+MOM TCCD run, seed 0, from scratch
                 (only authorized when Stage A PASSes).
* ``official`` : one official-train -> official-valid run (only authorized for
                 Stage-B Case P or G).  Official test is never loaded.

Formal execution is GPU1-only; GPU0 is forbidden for this round.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V2
from tracks.ksvd.code import tccd_v6 as V6
from tracks.ksvd.code import tccd_v7 as V
from tracks.ksvd.code.run_tccd_v0 import internal_split, load_or_build_records

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


def _raw_reference() -> dict[str, Any]:
    if not V.V5_STAGE_A_JSON.exists():
        raise SystemExit(f"missing frozen TCCD-v5 Stage-A result: {V.V5_STAGE_A_JSON}")
    sha = V._sha256_file(V.V5_STAGE_A_JSON)
    if sha != V.V5_STAGE_A_SHA256:
        raise SystemExit(f"TCCD-v5 Stage-A JSON SHA mismatch: {sha}")
    payload = json.loads(V.V5_STAGE_A_JSON.read_text(encoding="utf-8"))
    if payload.get("official_test_loaded") is not False:
        raise SystemExit("TCCD-v5 result does not certify official_test_loaded == false")
    base = payload["results"]["BASE"]
    if int(base["feature_dim"]) != V.RAW_DIM:
        raise SystemExit("TCCD-v5 BASE feature_dim is not the frozen RAW representation")
    return {
        "path": str(V.V5_STAGE_A_JSON.relative_to(V.REPO_ROOT)),
        "sha256": sha,
        "feature_dim": int(base["feature_dim"]),
        "best_valid_mae": float(base["best_valid_mae"]),
        "soup_valid_mae": float(base["soup_valid_mae"]),
        "best_epoch": int(base["best_epoch"]),
        "soup_members": list(base["soup_members"]),
        "protocol": payload.get("protocol"),
    }


def _v2_reference() -> dict[str, Any]:
    if not V.V2_BEST_CHECKPOINT.exists():
        raise SystemExit(f"missing frozen TCCD-v2 checkpoint: {V.V2_BEST_CHECKPOINT}")
    sha = V._sha256_file(V.V2_BEST_CHECKPOINT)
    if sha != V.V2_BEST_CHECKPOINT_SHA256:
        raise SystemExit("TCCD-v2 checkpoint SHA mismatch")
    gate_a = V2.RESULTS_DIR / "gateA_seed0.json"
    out = {
        "checkpoint": str(V.V2_BEST_CHECKPOINT.relative_to(V.REPO_ROOT)),
        "checkpoint_sha256": sha,
        "internal_best_mae": V.V2_INTERNAL_BEST_MAE,
        "internal_soup_mae": V.V2_INTERNAL_SOUP_MAE,
        "official_valid_best_mae": V.V2_OFFICIAL_VALID_BEST,
        "official_valid_soup_mae": V.V2_OFFICIAL_VALID_SOUP,
        "canonical_gpu1_baseline": V.CANONICAL_GPU1_BASELINE,
        "gate_a_json_present": bool(gate_a.exists()),
        "official_test_loaded": False,
    }
    if gate_a.exists():
        payload = json.loads(gate_a.read_text(encoding="utf-8"))
        out["gate_a_protocol"] = payload.get("protocol")
        out["gate_a_n_train"] = payload.get("n_train")
        out["gate_a_n_dev"] = payload.get("n_dev")
        out["gate_a_prototype_rel_best"] = float(payload["prototype_rel"]["best_valid"])
        out["gate_a_prototype_rel_soup"] = float(payload["prototype_rel"]["soup_valid"])
        out["gate_a_split_seed"] = payload.get("split_seed")
    return out


def _summary(res: V.ReaderResult) -> dict[str, Any]:
    return {
        "best_valid_mae": float(res.best_valid),
        "best_epoch": int(res.best_epoch),
        "soup_valid_mae": None if res.soup_valid is None else float(res.soup_valid),
        "soup_members": list(res.soup_members),
        "feature_dim": int(res.feature_dim),
        "init_checksum": res.init_checksum,
        "wall_s": float(res.wall_s),
    }


def _arm_paths(arm: str, seed: int) -> dict[str, Path]:
    tag = f"stageA_{arm}_seed{seed}"
    return {
        "summary": OUT_DIR / f"{tag}.json",
        "best": OUT_DIR / f"{tag}_best.pt",
        "soup": OUT_DIR / f"{tag}_soup.pt",
    }


def _reuse_arm(arm: str, seed: int, commit: str) -> dict[str, Any] | None:
    paths = _arm_paths(arm, seed)
    if not paths["summary"].exists():
        return None
    try:
        payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        payload.get("protocol") == V.PROTOCOL_VERSION
        and payload.get("arm") == arm
        and int(payload.get("seed", -1)) == int(seed)
        and payload.get("commit") == commit
    ):
        return payload
    return None


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def gate0(args) -> int:
    cache = V6.load_label_free_cache()
    data_free = V.gate0_data_free_checks(args.device)
    real = V.gate0_real_cache_checks(cache, args.device)
    gpu = _gpu_info(args.device, args.physical_gpu)
    checks = {
        "A_permutation_invariance": bool(
            data_free["permutation_invariant"] and data_free["permutation_components_invariant"]
        ),
        "B_batching_invariance": bool(data_free["batching_invariant"] and data_free["padding_invariant"]),
        "C_algebra": bool(
            data_free["mu_row_sum_ok"] and data_free["variance_identity_ok"] and data_free["variance_clamp_path_ok"]
        ),
        "D_normalized_mass": bool(data_free["normalized_mass_ok"] and real["real_normalized_mass_ok"]),
        "E_zero_mass_relation": bool(
            data_free["zero_mass_relation_exact_zero"]
            and data_free["zero_mass_representation_finite"]
            and real["real_zero_mass_exact"]
        ),
        "F_no_target_leakage": bool(
            data_free["target_free"] and data_free["feature_builder_signature_target_free"]
        ),
        "G_official_test_blocked": bool(
            gpu["official_test_loaded"] is False and cache.meta.get("official_test_loaded") is False
        ),
        "H_reference_implementation": bool(data_free["reference_implementation_ok"]),
        "I_cache_vs_tensor_equivalence": bool(
            real["cache_vs_tensor_norm_ok"] and real["cache_vs_tensor_norm_mom_ok"]
        ),
        "J_relation_mass_source_agreement": bool(real["relation_mass_source_agreement"]),
        "K_raw_is_frozen_base": bool(real["raw_matches_base_exact"] and real["raw_dim"] == V.RAW_DIM),
        "L_representation_finite": bool(
            np.isfinite(data_free["reference_implementation"]["norm_mom"]["max_abs_diff"])
            and data_free["zero_mass_representation_finite"]
        ),
    }
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": _commit(),
        "gpu": gpu,
        "cache_meta": {
            "protocol": cache.meta.get("protocol"),
            "v2_checkpoint_sha256": cache.meta.get("v2_checkpoint_sha256"),
            "official_test_loaded": cache.meta.get("official_test_loaded"),
            "n_graphs": int(cache.n_graphs),
        },
        "dims": {"raw": V.RAW_DIM, "norm": V.NORM_DIM, "norm_mom": V.NORM_MOM_DIM},
        "feature_layout_norm": V.feature_layout(False),
        "feature_layout_norm_mom": V.feature_layout(True),
        "stage_b_parameter_accounting": V.parameter_accounting(True),
        "data_free": data_free,
        "real_cache": real,
        "checks": checks,
        "all_pass": bool(
            data_free.get("data_free_all_pass") and real.get("real_all_pass") and all(checks.values())
        ),
        "official_test_loaded": False,
    }
    V.write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if payload["all_pass"] else 1


def _train_arm(arm: str, args, cache, y, tr_idx, dev_idx, log=print) -> dict[str, Any]:
    commit = _commit()
    existing = _reuse_arm(arm, int(args.seed), commit)
    if existing is not None and not args.force:
        log(f"[stageA:{arm}] reusing completed arm {_arm_paths(arm, args.seed)['summary'].name}")
        summary = dict(existing["summary"])
        summary["reused"] = True
        return summary
    diag: dict[str, Any]
    if arm == "raw" or arm == "raw_rescreen":
        X = V.raw_features(cache)
        diag = {"kind": "raw", "feature_dim": int(X.shape[1])}
    else:
        with_moments = bool(V.ARM_WITH_MOMENTS[arm])
        X, diag = V.stage_a_features(cache, with_moments=with_moments)
    log(f"[stageA:{arm}] features {X.shape} (dim {V.ARM_DIM[arm]}); training linear reader")
    res = V.train_linear_reader(
        X,
        y,
        tr_idx,
        dev_idx,
        args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=log,
    )
    summary = _summary(res)
    summary["reused"] = False
    paths = _arm_paths(arm, args.seed)
    _save_state(paths["best"], res.state_best)
    _save_state(paths["soup"], res.state_soup)
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "stageA_arm",
        "arm": arm,
        "with_moments": None if arm.startswith("raw") else bool(V.ARM_WITH_MOMENTS[arm]),
        "seed": int(args.seed),
        "commit": commit,
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "feature_diagnostics": diag,
        "summary": summary,
        "official_test_loaded": False,
    }
    V.write_json(paths["summary"], payload)
    log(
        f"[stageA:{arm}] best {summary['best_valid_mae']:.6f}@{summary['best_epoch']} "
        f"soup {summary['soup_valid_mae']:.6f}"
    )
    return summary


def stage_a(args) -> int:
    _peak_reset(args.device)
    started = time.time()
    cache = V6.load_label_free_cache()
    y = V6.load_cache_labels()
    tr_idx, dev_idx = internal_split(int(cache.n_graphs))
    raw_ref = _raw_reference()
    v2_ref = _v2_reference()
    log = print
    log(f"[stageA] graphs={cache.n_graphs} train={len(tr_idx)} dev={len(dev_idx)}")
    log(f"[stageA] frozen RAW reference soup={raw_ref['soup_valid_mae']} best={raw_ref['best_valid_mae']}")

    arms: dict[str, dict[str, Any]] = {}
    for arm in ("raw_rescreen", "norm", "norm_mom"):
        arms[arm] = _train_arm(arm, args, cache, y, tr_idx, dev_idx, log=log)

    decision = V.stage_a_decision(
        raw_soup=float(raw_ref["soup_valid_mae"]),
        raw_rescreen_soup=float(arms["raw_rescreen"]["soup_valid_mae"]),
        norm_soup=float(arms["norm"]["soup_valid_mae"]),
        nm_soup=float(arms["norm_mom"]["soup_valid_mae"]),
        raw_best=float(raw_ref["best_valid_mae"]),
        raw_rescreen_best=float(arms["raw_rescreen"]["best_valid_mae"]),
        norm_best=float(arms["norm"]["best_valid_mae"]),
        nm_best=float(arms["norm_mom"]["best_valid_mae"]),
    )
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "A_frozen_screen",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "split_seed": T.SPLIT_SEED,
        "n_train": int(len(tr_idx)),
        "n_dev": int(len(dev_idx)),
        "primary_metric": "top5_soup",
        "reader": "Linear(h_G, 1)",
        "raw_reference": raw_ref,
        "v2_reference": v2_ref,
        "arms": arms,
        "decision": decision,
        "wall_s": time.time() - started,
        "peak_mem_mb": _peak_mb(args.device),
        "official_test_loaded": False,
    }
    V.write_json(OUT_DIR / f"stageA_seed{args.seed}.json", payload)
    V.write_json(OUT_DIR / f"stageA_decision_seed{args.seed}.json", decision)
    log(
        "[stageA] verdict={verdict} raw={raw_soup:.6f} raw_rescreen={raw_rescreen_soup:.6f} "
        "norm={norm_soup:.6f} norm_mom={nm_soup:.6f} delta_norm={delta_norm:.6f} "
        "delta_mom={delta_mom:.6f} drift={protocol_drift:.6f}".format(**decision)
    )
    return 0 if decision["pass"] else 1


def _load_stage_a_decision(seed: int) -> dict[str, Any]:
    path = OUT_DIR / f"stageA_decision_seed{seed}.json"
    if not path.exists():
        raise SystemExit("stage A decision missing; run the stageA stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_b(args) -> int:
    decision_a = _load_stage_a_decision(int(args.seed))
    if not decision_a.get("stage_b_authorized"):
        raise SystemExit(
            f"Stage A did not authorize Stage B (verdict={decision_a.get('verdict')}); STOP per preregistration"
        )
    out_path = OUT_DIR / f"stageB_seed{args.seed}.json"
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing.get("protocol") == V.PROTOCOL_VERSION and int(existing.get("seed", -1)) == int(args.seed):
            print(f"[stageB] reusing completed result {out_path.name}")
            return 0 if not existing["stage_b_decision"]["stop"] else 1

    log = print
    _peak_reset(args.device)
    started = time.time()
    records, records_meta = V2.load_train_records()
    tr_idx, dev_idx = internal_split(len(records))
    model = V.build_nm_model(args.device, with_moments=True, seed=args.seed)
    checksum = V._state_checksum(model)
    log(
        f"[stageB] end-to-end NORM+MOM: graphs={len(records)} train={len(tr_idx)} dev={len(dev_idx)} "
        f"seed={args.seed} params={sum(p.numel() for p in model.parameters())}"
    )
    res = V.train_stage_b(
        model,
        records,
        records,
        tr_idx,
        dev_idx,
        args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=log,
    )
    peak = _peak_mb(args.device)
    soup_path = OUT_DIR / f"stageB_seed{args.seed}_soup.pt"
    best_path = OUT_DIR / f"stageB_seed{args.seed}_best.pt"
    _save_state(soup_path, res.state_soup)
    _save_state(best_path, res.state_best)

    # vocabulary health on the internal dev split (best checkpoint)
    if res.state_best is not None:
        model.load_state_dict(res.state_best)
    C_dev, mol_ids = V2.collect_assignments(model, records, list(dev_idx), args.device, batch=64)
    usage = V2.usage_metrics(C_dev, mol_ids)
    semantic = V2.semantic_coherence(C_dev, records, list(dev_idx))
    learned_tau = float(model.temperature().detach())
    vocab = {
        "active_prototypes": int(usage["active_prototypes"]),
        "dead_prototypes": int(usage["dead_prototypes"]),
        "effective_prototype_count": float(usage["effective_prototype_count"]),
        "top1_usage_mass": float(usage["top1_usage_mass"]),
        "top8_usage_mass": float(usage["top8_usage_mass"]),
        "mean_local_assignment_entropy": float(usage["mean_local_assignment_entropy"]),
        "global_usage_entropy": float(usage["global_usage_entropy"]),
        "global_usage_entropy_normalized": float(usage["global_usage_entropy_normalized"]),
        "learned_temperature": learned_tau,
        "semantic_coherence_summary": semantic.get("_summary"),
        "n_dev_patches": int(C_dev.shape[0]),
        "dead_threshold": V2.DEAD_THRESHOLD,
        "top8_collapse_threshold": 0.80,
    }
    vocab["vocabulary_pass"] = bool(
        usage["active_prototypes"] >= 48 and float(usage["top8_usage_mass"]) <= 0.80
    )
    decision_b = V.stage_b_decision(float(res.soup_valid))
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "B_end_to_end",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "split_seed": T.SPLIT_SEED,
        "n_train": int(len(tr_idx)),
        "n_dev": int(len(dev_idx)),
        "architecture": "x_v -> z_v -> c_v -> h_NM -> Linear -> y",
        "representation": "NORM+MOM (explicit normalized prototype mean/variance + normalized relation contractions)",
        "primary_metric": "top5_soup",
        "trainable": ["encoder_W", "prototypes_P", "temperature_logit", "linear_reader"],
        "regularization": res.regularization,
        "init_checksum": checksum,
        "parameter_accounting": V.parameter_accounting(True),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_valid_mae": float(res.best_valid),
        "best_epoch": int(res.best_epoch),
        "soup_valid_mae": None if res.soup_valid is None else float(res.soup_valid),
        "soup_members": list(res.soup_members),
        "checkpoint_best": str(best_path.relative_to(V.REPO_ROOT)),
        "checkpoint_soup": str(soup_path.relative_to(V.REPO_ROOT)),
        "matched_reference": {
            "v2_internal_best": V.V2_INTERNAL_BEST_MAE,
            "v2_internal_soup": V.V2_INTERNAL_SOUP_MAE,
        },
        "delta_vs_v2_best": float(V.V2_INTERNAL_BEST_MAE) - float(res.best_valid),
        "delta_vs_v2_soup": float(V.V2_INTERNAL_SOUP_MAE) - float(res.soup_valid),
        "vocabulary": vocab,
        "records_meta": records_meta,
        "stage_a_decision": decision_a,
        "stage_b_decision": decision_b,
        "train_history": res.train_history,
        "wall_s": float(res.wall_s),
        "wall_s_total": float(time.time() - started),
        "peak_mem_mb": peak,
        "official_test_loaded": False,
    }
    V.write_json(out_path, payload)
    V.write_json(OUT_DIR / f"stageB_decision_seed{args.seed}.json", decision_b)
    log(
        f"[stageB] best {res.best_valid:.6f} soup {res.soup_valid:.6f} tau {learned_tau:.5f} "
        f"active {vocab['active_prototypes']} top8 {vocab['top8_usage_mass']:.4f} -> case {decision_b['case']}"
    )
    return 0 if not decision_b["stop"] else 1


def _load_stage_b_decision(seed: int) -> dict[str, Any]:
    path = OUT_DIR / f"stageB_decision_seed{seed}.json"
    if not path.exists():
        raise SystemExit("stage B decision missing; run the stageB stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def official(args) -> int:
    decision_b = _load_stage_b_decision(int(args.seed))
    if not decision_b.get("official_valid_authorized"):
        raise SystemExit(f"Stage B case {decision_b.get('case')} does not authorize the official-valid run; STOP")
    out_path = OUT_DIR / f"official_seed{args.seed}.json"
    if out_path.exists() and not args.force:
        print(f"[official] reusing completed result {out_path.name}")
        return 0

    log = print
    _peak_reset(args.device)
    started = time.time()
    train_records, train_meta = V2.load_train_records()
    layout = V2.frozen_layout()
    train_mols, _ = T.load_mols(args.data_root, "train")
    atom_index, bond_index = T.category_catalog(train_mols)
    if len(atom_index) != layout.n_atom or len(bond_index) != layout.n_bond:
        raise SystemExit("frozen layout catalog mismatch on full train")
    valid_mols, valid_y = T.load_mols(args.data_root, "valid")
    valid_records, valid_cached = load_or_build_records(
        "valid", valid_mols, layout, atom_index, bond_index, valid_y, len(valid_mols), log=log
    )
    log(
        f"[official] full train={len(train_records)} official valid={len(valid_records)} "
        f"cached_valid={valid_cached}"
    )
    model = V.build_nm_model(args.device, with_moments=True, seed=args.seed)
    res = V.train_stage_b(
        model,
        train_records,
        valid_records,
        list(range(len(train_records))),
        list(range(len(valid_records))),
        args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=log,
    )
    peak = _peak_mb(args.device)
    soup_path = OUT_DIR / f"official_seed{args.seed}_soup.pt"
    best_path = OUT_DIR / f"official_seed{args.seed}_best.pt"
    _save_state(soup_path, res.state_soup)
    _save_state(best_path, res.state_best)
    if res.state_best is not None:
        model.load_state_dict(res.state_best)
    C_valid, mol_ids = V2.collect_assignments(model, valid_records, list(range(len(valid_records))), args.device)
    usage = V2.usage_metrics(C_valid, mol_ids)
    delta_abs = float(res.soup_valid) - V.CANONICAL_GPU1_BASELINE
    band = V.official_band(float(res.soup_valid))
    payload = {
        "protocol": V.PROTOCOL_VERSION,
        "stage": "official_valid",
        "commit": _commit(),
        "gpu": _gpu_info(args.device, args.physical_gpu),
        "seed": int(args.seed),
        "n_official_train": int(len(train_records)),
        "n_official_valid": int(len(valid_records)),
        "valid_records_cached": bool(valid_cached),
        "split": "official train 10000 -> official valid 1000; official test never loaded",
        "architecture": "x_v -> z_v -> c_v -> h_NM -> Linear -> y",
        "primary_metric": "top5_soup",
        "best_valid_mae": float(res.best_valid),
        "best_epoch": int(res.best_epoch),
        "soup_valid_mae": None if res.soup_valid is None else float(res.soup_valid),
        "soup_members": list(res.soup_members),
        "checkpoint_best": str(best_path.relative_to(V.REPO_ROOT)),
        "checkpoint_soup": str(soup_path.relative_to(V.REPO_ROOT)),
        "learned_temperature": float(model.temperature().detach()),
        "regularization": res.regularization,
        "canonical_gpu1_baseline": V.CANONICAL_GPU1_BASELINE,
        "delta_abs": delta_abs,
        "band": band,
        "v2_official_valid_best": V.V2_OFFICIAL_VALID_BEST,
        "v2_official_valid_soup": V.V2_OFFICIAL_VALID_SOUP,
        "delta_vs_v2_official_best": float(V.V2_OFFICIAL_VALID_BEST) - float(res.best_valid),
        "delta_vs_v2_official_soup": float(V.V2_OFFICIAL_VALID_SOUP) - float(res.soup_valid),
        "usage_on_official_valid": usage,
        "stage_b_decision": decision_b,
        "train_meta": train_meta,
        "train_history": res.train_history,
        "wall_s": float(res.wall_s),
        "wall_s_total": float(time.time() - started),
        "peak_mem_mb": peak,
        "official_test_loaded": False,
    }
    V.write_json(out_path, payload)
    log(
        f"[official] best {res.best_valid:.6f} soup {res.soup_valid:.6f} "
        f"baseline {V.CANONICAL_GPU1_BASELINE:.6f} delta_abs {delta_abs:.6f} -> {band}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gate0", "stageA", "stageB", "official"])
    p.add_argument("--data-root", type=Path, default=T.REPO_ROOT / "data/ZINC")
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
