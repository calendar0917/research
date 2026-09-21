#!/usr/bin/env python
"""TCCD-v1 runner — frozen gates from ``notes/tccd_v1_preregistration.md``.

TCCD-v1 **reuses the frozen TCCD-v0 dictionary D0** and the cached canonical
patch records.  It never refits K-SVD.

Stages
------
``smoke``  cheap GPU1 optimization/gradient smoke (not a scientific gate)
``gateA``  frozen composition: BAG / REL / REL-SHUFFLE  (§6)
``gateB``  task coupling: FROZEN-D vs TASK-D            (§7)
``gateC``  dictionary uniqueness: TASK-D vs DENSE       (§9)
``gateD``  absolute performance on full official data   (§10)

Only GPU1 is used (``bash scripts/run_remote.sh 1 …``).
Official test is never loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v1 as V
from tracks.ksvd.code.run_tccd_v0 import (
    atom_semantics,
    continuity_audit,
    internal_split,
    load_or_build_records,
)

OUT_DIR = V.RESULTS_DIR


def _load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _records_for(args, log=print):
    """Full cached official-train records; ``--limit`` is a DEBUG-only subset."""
    records, meta = V.load_train_records()
    if args.limit:
        log(f"[debug] truncating records to {args.limit} (DEBUG ONLY; not a formal run)")
        records = records[: int(args.limit)]
    return records, meta


def _peak_reset(device: str):
    torch = T._torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return torch
    return None


def _peak_mb(device: str) -> float | None:
    torch = T._torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return None


# ===========================================================================
# smoke (not a scientific gate)
# ===========================================================================
def smoke(args, log=print) -> int:
    torch = T._torch()
    device = args.device
    D, dobj = V.load_frozen_dictionary()
    layout = V.frozen_layout()
    if D.shape != (layout.feature_dim, T.K_DICT):
        raise SystemExit(f"frozen D0 shape {D.shape} != layout {(layout.feature_dim, T.K_DICT)}")
    records, meta = V.load_train_records()
    tr_idx, dev_idx = internal_split(len(records))
    n_smoke = int(min(args.limit or 128, len(tr_idx)))
    sub = [int(i) for i in tr_idx[:n_smoke]]
    log(f"[smoke] {n_smoke} graphs; layout={layout.descriptor()}; device={device}")

    model = T.TCCDModel.build(layout.feature_dim, T.K_DICT, V.N_REL,
                              np.asarray(D, dtype=np.float32), dense=False, frozen_D=False)
    model = model.to(device)
    b = T.make_batch(records, sub[:32], device)

    checks: dict[str, Any] = {}

    # 1. exact sparsity under tied IHT
    model.zero_grad(set_to_none=True)
    C = model.encode(b["X_all"], s=T.SPARSITY)
    nnz = (C.abs() > 0).sum(dim=1)
    checks["sparsity_exact"] = bool(int(nnz.max().item()) == T.SPARSITY and int(nnz.min().item()) == T.SPARSITY)

    # 2. reconstruction gradient reaches D
    model.zero_grad(set_to_none=True)
    rec = ((b["X_all"] - C @ model.D.t()) ** 2).sum(dim=1) / ((b["X_all"] ** 2).sum(dim=1) + T.EPS)
    rec.mean().backward()
    checks["grad_rec_to_D"] = bool(model.D.grad is not None and float(model.D.grad.abs().sum()) > 0)
    grad_rec = float(model.D.grad.abs().sum()) if model.D.grad is not None else 0.0

    # 3. task gradient reaches D (through C^T R C -> C -> D)
    model.zero_grad(set_to_none=True)
    pred = model(b)
    task = (pred - b["y"]).abs().mean()
    task.backward()
    checks["grad_task_to_D"] = bool(model.D.grad is not None and float(model.D.grad.abs().sum()) > 0)
    grad_task = float(model.D.grad.abs().sum()) if model.D.grad is not None else 0.0

    # 4. batching invariance of predictions
    model.eval()
    with torch.no_grad():
        pred_batch = model(b)
        singles = []
        for gi in sub[:8]:
            bb = T.make_batch(records, [gi], device)
            singles.append(model(bb).reshape(-1))
        pred_single = torch.cat(singles)
    checks["batching_invariance"] = bool(float((pred_batch[:8] - pred_single).abs().max()) <= 1e-5)

    # 5. D does not collapse and train loss decreases over a few steps
    before = model.D.detach().clone()
    me = min(int(args.max_epochs), 30)  # smoke is not a formal training run
    res = V.train_model_fast(model, records, records, sub, sub, device, seed=args.seed,
                             max_epochs=me, patience=max(me, 1),
                             batch=int(args.batch), calibrate_rec=True, log=log)
    after = model.D.detach()
    checks["D_updated"] = bool(float((after - before).abs().max()) > 0)
    col = after.norm(dim=0)
    checks["no_column_collapse"] = bool(float(col.min()) > 1e-3)
    checks["no_nan"] = bool(torch.isfinite(after).all().item())
    hist = res.train_history
    checks["train_loss_decreases"] = bool(len(hist) >= 2 and hist[-1]["train_loss"] < hist[0]["train_loss"])

    checks["smoke_pass"] = bool(all(v for k, v in checks.items() if k != "smoke_pass"))
    out = {
        "protocol": V.PROTOCOL_VERSION,
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "n_graphs": n_smoke,
        "layout": layout.descriptor(),
        "dictionary_fingerprint": V.dictionary_fingerprint(D),
        "ksvd_refit_performed": False,
        "grad_rec_to_D": grad_rec,
        "grad_task_to_D": grad_task,
        "train_history": hist,
        "checks": checks,
        "official_test_loaded": False,
    }
    T.write_json(OUT_DIR / f"smoke_seed{args.seed}.json", out)
    log(f"[smoke] {'PASS' if checks['smoke_pass'] else 'FAIL'}: {checks}")
    return 0 if checks["smoke_pass"] else 1


# ===========================================================================
# Gate A — frozen composition
# ===========================================================================
def _reader_json(r: V.ReaderResult) -> dict[str, Any]:
    return {
        "best_valid": r.best_valid,
        "alpha": r.alpha,
        "grid_valid": r.grid_valid,
        "wall_s": r.wall_s,
    }


def gate_a(args, log=print) -> int:
    device = args.device
    D, _ = V.load_frozen_dictionary()
    layout = V.frozen_layout()
    records, meta = _records_for(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    y = np.asarray([float(r["y"]) for r in records], dtype=np.float32)

    log(f"[gateA] graphs={len(records)} internal_train={len(tr_idx)} internal_dev={len(dev_idx)} seed={args.seed}")
    codes = V.load_or_build_codes(records, D, log=log)
    bag, rel = V.load_or_build_common(codes, records, log=log)
    shuf = V.load_or_build_shuffle(codes, records, args.seed, log=log)

    tr = np.asarray(tr_idx, dtype=np.int64)
    dev = np.asarray(dev_idx, dtype=np.int64)
    arms: dict[str, Any] = {}
    _peak_reset(device)
    for name, X in [("BAG", bag), ("REL", rel), ("REL-SHUFFLE", shuf)]:
        log(f"[gateA] fitting shared linear reader for arm={name} (dim={X.shape[1]})")
        res = V.ridge_reader(X[tr], y[tr], X[dev], y[dev], log=log)
        arms[name] = _reader_json(res)
    peak = _peak_mb(device)

    delta_comp = arms["REL-SHUFFLE"]["best_valid"] - arms["REL"]["best_valid"]
    seed0 = None
    if int(args.seed) != V.PRIMARY_SEED:
        prev = _load_json(OUT_DIR / f"gateA_seed{V.PRIMARY_SEED}.json")
        if prev is None:
            raise SystemExit("seed-1 Gate A requires the seed-0 result json")
        seed0 = float(prev["delta_comp"])
    verdict = V.gate_a_verdict(delta_comp, args.seed, seed0)
    mean_delta = delta_comp if seed0 is None else 0.5 * (seed0 + delta_comp)

    out: dict[str, Any] = {
        "protocol": V.PROTOCOL_VERSION,
        "gate": "A",
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "seed": args.seed,
        "split_seed": T.SPLIT_SEED,
        "n_train": int(len(tr_idx)),
        "n_dev": int(len(dev_idx)),
        "n_patches": int(sum(int(r["n"]) for r in records)),
        "reused_tccd_v0_dictionary": True,
        "ksvd_refit_performed": False,
        "frozen_dictionary_fingerprint": V.dictionary_fingerprint(D),
        "features": {"bag_dim": int(bag.shape[1]), "rel_dim": int(rel.shape[1]),
                     "shuffle_dim": int(shuf.shape[1])},
        "arms": arms,
        "delta_comp": delta_comp,
        "delta_comp_mean": mean_delta,
        "mae_bag_minus_rel": arms["BAG"]["best_valid"] - arms["REL"]["best_valid"],
        "thresholds": {"pass": V.GATE_A_PASS, "fail": V.GATE_A_FAIL,
                       "ambiguous_mean_pass": V.GATE_A_AMBIG_MEAN_PASS},
        "verdict": verdict,
        "peak_mem_mb": peak,
        "official_test_loaded": False,
    }
    T.write_json(OUT_DIR / f"gateA_seed{args.seed}.json", out)
    log(f"[gateA] REL={arms['REL']['best_valid']:.6f} REL-SHUFFLE={arms['REL-SHUFFLE']['best_valid']:.6f} "
        f"BAG={arms['BAG']['best_valid']:.6f} delta_comp={delta_comp:.5f} -> {verdict}")

    if verdict == "AMBIGUOUS_NEEDS_SEED1":
        return 3
    decision = {**out, "seeds": [V.PRIMARY_SEED] if seed0 is None else [V.PRIMARY_SEED, int(args.seed)],
                "final_verdict": verdict}
    T.write_json(OUT_DIR / "gateA_decision.json", decision)
    return V.verdict_to_exit(verdict)


# ===========================================================================
# Gate B — task coupling
# ===========================================================================
def _train_arm_timed(arm, args, layout, records, tr_idx, dev_idx, device, D, log):
    """Train Gate B/C arms with the vectorized math-equivalent path."""
    torch = T._torch()
    dense = arm == "DENSE"
    frozen = arm == "FROZEN-D"
    model = T.TCCDModel.build(
        layout.feature_dim, T.K_DICT, V.N_REL, D,
        dense=dense, frozen_D=frozen, readout_dim=None, seed=args.seed,
    )
    if frozen:
        model.D.requires_grad_(False)
    model = model.to(device)
    _peak_reset(device)
    t0 = time.time()
    res = V.train_model_fast(
        model, records, records, list(tr_idx), list(dev_idx), device,
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
        batch=args.batch, calibrate_rec=(arm == "TASK-D"), log=log,
    )
    return res, {"wall_s": time.time() - t0, "peak_mem_mb": _peak_mb(device)}


def gate_b(args, log=print) -> int:
    device = args.device
    D, _ = V.load_frozen_dictionary()
    layout = V.frozen_layout()
    records, meta = _records_for(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    log(f"[gateB] seed={args.seed} FROZEN-D vs TASK-D (K={T.K_DICT}, s={T.SPARSITY})")

    out: dict[str, Any] = {
        "protocol": V.PROTOCOL_VERSION, "gate": "B", "commit": T.git("rev-parse", "HEAD"),
        "device": device, "seed": args.seed, "split_seed": T.SPLIT_SEED,
        "n_train": int(len(tr_idx)), "n_dev": int(len(dev_idx)),
        "reused_tccd_v0_dictionary": True, "ksvd_refit_performed": False,
        "frozen_dictionary_fingerprint": V.dictionary_fingerprint(D),
        "arms": {}, "official_test_loaded": False,
    }
    for arm in ["FROZEN-D", "TASK-D"]:
        log(f"[gateB] arm={arm}")
        res, timing = _train_arm_timed(arm, args, layout, records, tr_idx, dev_idx, device, D, log)
        out["arms"][arm] = {
            "best_valid": res.best_valid, "best_epoch": res.best_epoch,
            "soup_valid": res.soup_valid, "soup_members": res.soup_members,
            "lam_rec": res.lam_rec, **timing,
        }
    delta = out["arms"]["FROZEN-D"]["best_valid"] - out["arms"]["TASK-D"]["best_valid"]
    seed0 = None
    if int(args.seed) != V.PRIMARY_SEED:
        prev = _load_json(OUT_DIR / f"gateB_seed{V.PRIMARY_SEED}.json")
        if prev is None:
            raise SystemExit("seed-1 Gate B requires the seed-0 result json")
        seed0 = float(prev["delta_task"])
    verdict = V.gate_b_verdict(delta, args.seed, seed0)
    out["delta_task"] = delta
    out["delta_task_mean"] = delta if seed0 is None else 0.5 * (seed0 + delta)
    out["thresholds"] = {"pass": V.GATE_B_PASS, "ambiguous_mean_pass": V.GATE_B_AMBIG_MEAN_PASS}
    out["verdict"] = verdict
    T.write_json(OUT_DIR / f"gateB_seed{args.seed}.json", out)
    log(f"[gateB] FROZEN-D={out['arms']['FROZEN-D']['best_valid']:.6f} "
        f"TASK-D={out['arms']['TASK-D']['best_valid']:.6f} delta_task={delta:.5f} -> {verdict}")

    if verdict == "AMBIGUOUS_NEEDS_SEED1":
        return 3
    decision = {**out, "seeds": [V.PRIMARY_SEED] if seed0 is None else [V.PRIMARY_SEED, int(args.seed)],
                "final_verdict": verdict}
    T.write_json(OUT_DIR / "gateB_decision.json", decision)
    return V.verdict_to_exit(verdict)


# ===========================================================================
# Gate B diagnostics — report-only continuity / reuse / semantics
# ===========================================================================
def _stack_codes_fast(model, records, indices, device, batch=64):
    torch = T._torch()
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = list(indices[start : start + batch])
            b = V.make_padded_batch(records, chunk, device)
            _, C_flat, _ = V._fast_forward(model, b)
            C_pad = C_flat.reshape(b["X_pad"].shape[0], b["X_pad"].shape[1], -1)
            chunks.append(C_pad[b["valid"]].detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _reuse_stats(records, dev_idx, C_dev):
    support = np.abs(C_dev) > 1e-8
    atom_support = support.sum(axis=0)
    atom_mass = np.abs(C_dev).sum(axis=0)
    mol_of_patch = np.concatenate([
        np.full(int(records[i]["n"]), k, dtype=np.int64)
        for k, i in enumerate(dev_idx)
    ])
    mol_presence = np.zeros((support.shape[1],), dtype=np.int64)
    for k in range(support.shape[1]):
        mol_presence[k] = len(np.unique(mol_of_patch[support[:, k]]))
    order = np.argsort(atom_mass)[::-1]
    total_mass = float(atom_mass.sum())
    return {
        "dead_atoms": int(np.sum(atom_support < 5)),
        "reused_atoms": int(np.sum((atom_support >= 20) & (mol_presence >= 5))),
        "top1_mass_share": float(atom_mass[order[0]] / max(total_mass, 1e-12)),
        "top8_mass_share": float(atom_mass[order[:8]].sum() / max(total_mass, 1e-12)),
        "mean_coeff_entropy": float(T.coefficient_entropy(C_dev).mean()),
        "atom_support": atom_support.astype(int).tolist(),
        "atom_mol_presence": mol_presence.astype(int).tolist(),
        "top_atoms": order[:8].astype(int).tolist(),
    }


def gate_b_diagnostics(args, log=print) -> int:
    decision = _load_json(OUT_DIR / "gateB_decision.json")
    if decision is None or decision.get("final_verdict") != "PASS":
        raise SystemExit("Gate B diagnostics require a Gate B PASS decision.")
    D0, _ = V.load_frozen_dictionary()
    layout = V.frozen_layout()
    records, _ = _records_for(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    device = args.device
    log("[diagB] loading official-train molecules for the frozen continuity audit")
    mols, _ = T.load_mols(args.data_root, "train")

    # Re-run exactly the single Gate-B TASK-D arm to retain its soup state.
    model = T.TCCDModel.build(
        layout.feature_dim, T.K_DICT, V.N_REL, D0,
        dense=False, frozen_D=False, readout_dim=None, seed=args.seed,
    ).to(device)
    res = V.train_model_fast(
        model, records, records, list(tr_idx), list(dev_idx), device,
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
        batch=args.batch, calibrate_rec=True, log=log,
    )
    if res.state_soup is None:
        raise SystemExit("TASK-D soup state missing; cannot run post-Gate-B diagnostics")
    model.load_state_dict(res.state_soup)
    D_task = model.D.detach().cpu().numpy().astype(np.float64)

    X_dev = np.concatenate([np.asarray(records[i]["X"], dtype=np.float32) for i in dev_idx], axis=0)
    C_frozen = T.omp_codes(D0, X_dev, s=T.SPARSITY)
    C_task = _stack_codes_fast(model, records, list(dev_idx), device, batch=64)
    log("[diagB] continuity audit: frozen D0")
    cont_frozen = continuity_audit(mols, records, dev_idx, D0, C_frozen, log=log)
    log("[diagB] continuity audit: TASK-D soup")
    cont_task = continuity_audit(mols, records, dev_idx, D_task, C_task, log=log)
    reuse_frozen = _reuse_stats(records, list(dev_idx), C_frozen)
    reuse_task = _reuse_stats(records, list(dev_idx), C_task)
    sem_frozen = atom_semantics(records, dev_idx, C_frozen, reuse_frozen["top_atoms"])
    sem_task = atom_semantics(records, dev_idx, C_task, reuse_task["top_atoms"])

    cache_path = V.CACHE_DIR / "taskD_seed0_soup.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        import pickle
        pickle.dump({"state_soup": res.state_soup, "D_task": D_task}, fh, protocol=4)
    out = {
        "protocol": V.PROTOCOL_VERSION,
        "gate": "B_diagnostics",
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "seed": args.seed,
        "execution_regime": "local_cpu",
        "reused_tccd_v0_dictionary": True,
        "ksvd_refit_performed": False,
        "task_checkpoint": "TASK-D top-5 soup state (diagnostic only)",
        "task_soup_valid": res.soup_valid,
        "continuity": {"frozen": cont_frozen, "task": cont_task},
        "reuse": {"frozen": reuse_frozen, "task": reuse_task},
        "atom_semantics": {"frozen": sem_frozen, "task": sem_task},
        "official_test_loaded": False,
    }
    T.write_json(OUT_DIR / "gateB_diagnostics_seed0.json", out)
    log(f"[diagB] frozen_code_auc={cont_frozen['code_auc']:.6f} "
        f"task_code_auc={cont_task['code_auc']:.6f}; report-only complete")
    return 0


# ===========================================================================
# Gate C — dictionary uniqueness (DENSE control)
# ===========================================================================
def gate_c(args, log=print) -> int:
    device = args.device
    D, _ = V.load_frozen_dictionary()
    layout = V.frozen_layout()
    records, meta = _records_for(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    decision = _load_json(OUT_DIR / "gateB_decision.json")
    if decision is None or decision.get("final_verdict") != "PASS":
        raise SystemExit("Gate C requires a Gate B PASS decision; refusing to run out of order.")
    seeds = [int(s) for s in decision.get("seeds", [V.PRIMARY_SEED])]
    log(f"[gateC] DENSE control at seeds={seeds}")

    task_seed = {int(json.loads((OUT_DIR / f"gateB_seed{s}.json").read_text())["seed"]):
                 json.loads((OUT_DIR / f"gateB_seed{s}.json").read_text())["arms"]["TASK-D"]["best_valid"]
                 for s in seeds}
    dense_vals = {}
    for s in seeds:
        args.seed = s
        log(f"[gateC] DENSE seed={s}")
        res, timing = _train_arm_timed("DENSE", args, layout, records, tr_idx, dev_idx, device, D, log)
        dense_vals[s] = {"best_valid": res.best_valid, "best_epoch": res.best_epoch,
                         "soup_valid": res.soup_valid, "soup_members": res.soup_members, **timing}
    mae_task = float(np.mean([task_seed[s] for s in seeds]))
    mae_dense = float(np.mean([dense_vals[s]["best_valid"] for s in seeds]))
    ok = mae_task <= mae_dense + V.GATE_C_MARGIN
    out = {
        "protocol": V.PROTOCOL_VERSION, "gate": "C", "commit": T.git("rev-parse", "HEAD"),
        "device": device, "seeds": seeds, "split_seed": T.SPLIT_SEED,
        "reused_tccd_v0_dictionary": True, "ksvd_refit_performed": False,
        "task_d": task_seed, "dense": dense_vals,
        "mae_task_d": mae_task, "mae_dense": mae_dense, "margin": V.GATE_C_MARGIN,
        "delta_dense_minus_task": mae_dense - mae_task,
        "verdict": "PASS" if ok else "FAIL", "official_test_loaded": False,
    }
    T.write_json(OUT_DIR / "gateC.json", out)
    log(f"[gateC] TASK-D={mae_task:.6f} DENSE={mae_dense:.6f} delta={mae_dense - mae_task:.5f} -> {out['verdict']}")
    return 0 if ok else 1


# ===========================================================================
# Gate D — absolute performance (full official train / valid)
# ===========================================================================
def gate_d(args, log=print) -> int:
    device = args.device
    c = _load_json(OUT_DIR / "gateC.json")
    if c is None or c.get("verdict") != "PASS":
        raise SystemExit("Gate D requires a Gate C PASS decision; refusing to run out of order.")
    D, _ = V.load_frozen_dictionary()
    layout = V.frozen_layout()
    records, meta = V.load_train_records()

    # official valid records, built once with the FROZEN train catalog/layout
    train_mols, _ = T.load_mols(args.data_root, "train")
    atom_index, bond_index = T.category_catalog(train_mols)
    if len(atom_index) != layout.n_atom or len(bond_index) != layout.n_bond:
        raise SystemExit("frozen layout catalog mismatch on full train")
    valid_mols, valid_y = T.load_mols(args.data_root, "valid")
    valid_records, _ = load_or_build_records(
        "valid", valid_mols, layout, atom_index, bond_index, valid_y, len(valid_mols), log=log
    )
    log(f"[gateD] full official train={len(records)} official valid={len(valid_records)}")

    model = T.TCCDModel.build(layout.feature_dim, T.K_DICT, V.N_REL,
                              np.asarray(D, dtype=np.float32), dense=False, frozen_D=False)
    model = model.to(device)
    _peak_reset(device)
    t0 = time.time()
    res = V.train_model_fast(
        model, records, valid_records, list(range(len(records))), list(range(len(valid_records))),
        device, seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
        batch=args.batch, calibrate_rec=True, log=log,
    )
    delta_abs = res.best_valid - V.CANONICAL_GPU1_BASELINE
    band = V.gate_d_band(delta_abs)
    out = {
        "protocol": V.PROTOCOL_VERSION, "gate": "D", "commit": T.git("rev-parse", "HEAD"),
        "device": device, "seed": args.seed,
        "n_official_train": len(records), "n_official_valid": len(valid_records),
        "reused_tccd_v0_dictionary": True, "ksvd_refit_performed": False,
        "lambda_rec": res.lam_rec,
        "best_valid": res.best_valid, "best_epoch": res.best_epoch,
        "soup_valid": res.soup_valid, "soup_members": res.soup_members,
        "canonical_gpu1_baseline": V.CANONICAL_GPU1_BASELINE,
        "delta_abs": delta_abs, "band": band,
        "wall_s": time.time() - t0, "peak_mem_mb": _peak_mb(device),
        "official_test_loaded": False,
    }
    T.write_json(OUT_DIR / "gateD.json", out)
    log(f"[gateD] TCCD={res.best_valid:.6f} baseline={V.CANONICAL_GPU1_BASELINE:.6f} "
        f"delta_abs={delta_abs:.5f} -> {band}")
    return 0 if band != "NOT_VIABLE" else 1


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["smoke", "gateA", "gateB", "diagB", "gateC", "gateD"])
    p.add_argument("--data-root", type=Path, default=T.REPO_ROOT / "data/ZINC")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=V.PRIMARY_SEED)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    p.add_argument("--limit", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        return smoke(args)
    if args.stage == "gateA":
        return gate_a(args)
    if args.stage == "gateB":
        return gate_b(args)
    if args.stage == "diagB":
        return gate_b_diagnostics(args)
    if args.stage == "gateC":
        return gate_c(args)
    if args.stage == "gateD":
        return gate_d(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
