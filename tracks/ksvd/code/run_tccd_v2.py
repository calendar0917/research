#!/usr/bin/env python
"""TCCD-v2 runner: task-learned local prototype vocabulary.

Formal remote execution is GPU1-only and official test is never loaded.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v1 as V1
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code.run_tccd_v0 import continuity_audit, internal_split

OUT_DIR = V.RESULTS_DIR


def _peak_reset(device: str):
    torch = T._torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str):
    torch = T._torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return None


def _records(args, log=print):
    records, meta = V.load_train_records()
    if args.limit:
        log(f"[debug] truncating records to {args.limit}; formal runs must omit --limit")
        records = records[: int(args.limit)]
    return records, meta


def _make_model(mode: str, args, layout, device: str):
    init = V.load_frozen_encoder_init()
    model = V.PrototypeModelFactory.build(
        layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init,
        mode=mode, proto_seed=V.PROTO_INIT_SEED, seed=args.seed,
    ).to(device)
    return model


def _save_model_state(name: str, state: Any):
    path = OUT_DIR / f"{name}.pt"
    V.save_state(path, state)
    return path


def _train_dense(args, records, tr_idx, dev_idx, device, log):
    torch = T._torch()
    layout = V.frozen_layout()
    D0 = V.load_frozen_encoder_init()
    model = T.TCCDModel.build(
        layout.feature_dim, V.D_LOCAL, V.N_REL, D0, dense=True, frozen_D=False,
        readout_dim=None, seed=args.seed,
    ).to(device)
    _peak_reset(device)
    t0 = time.time()
    res = V1.train_model_fast(
        model, records, records, list(tr_idx), list(dev_idx), device,
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
        batch=args.batch, calibrate_rec=False, log=log,
    )
    res_wall = time.time() - t0
    return model, res, {"wall_s": res_wall, "peak_mem_mb": _peak_mb(device)}


def gate0(args, log=print) -> int:
    torch = T._torch()
    device = args.device
    records, _ = _records(args, log)
    tr_idx, _ = internal_split(len(records))
    idx = [int(i) for i in tr_idx[: min(args.n_graphs, len(tr_idx))]]
    layout = V.frozen_layout()
    model = _make_model("rel", args, layout, device)
    model.eval()
    b = V.make_batch(records, idx, device)
    checks: dict[str, Any] = {}

    with torch.no_grad():
        pred, C_flat, h = model.forward_padded(b)
    valid = b["valid"].reshape(-1)
    C = C_flat[valid]
    row_sum = C.sum(dim=1)
    checks["assignment_nonnegative"] = bool(float(C.min()) >= -1e-7)
    checks["assignment_rows_sum_one"] = bool(float((row_sum - 1.0).abs().max()) <= 1e-6)
    checks["no_nan"] = bool(torch.isfinite(C).all() and torch.isfinite(pred).all())
    checks["prototype_norm_min"] = float(model.P.detach().norm(dim=0).min())
    checks["prototype_norm_valid"] = checks["prototype_norm_min"] > 1e-3
    checks["temperature_initial"] = float(model.temperature().detach())
    checks["temperature_valid"] = V.TEMP_MIN < checks["temperature_initial"] < 1.0
    checks["no_bypass_flag"] = bool(model.uses_latent_bypass is False)
    checks["graph_path"] = model.graph_feature_kind
    checks["graph_feature_dim"] = int(h.shape[1])

    # Composition sensitivity: same assignment multiset, shuffled rows, fixed R.
    with torch.no_grad():
        C3 = C_flat.reshape(b["X_pad"].shape[0], b["X_pad"].shape[1], V.K_PROTO)
        shuf = V.shuffle_assignments(C3, records, idx, args.seed, device)
        h_shuf = V.compose_padded(shuf, b["R_pad"], b["valid"], b["iu0"], b["iu1"])
    checks["composition_sensitive"] = bool(float((h - h_shuf).abs().max()) > 1e-6)
    checks["composition_delta"] = float((h - h_shuf).norm() / (h.norm() + 1e-12))

    # Task gradients reach encoder, prototypes and temperature.
    model.train()
    model.zero_grad(set_to_none=True)
    pred, _, _ = model.forward_padded(b)
    loss = (pred - b["y"]).abs().mean()
    loss.backward()
    checks["grad_encoder"] = float(model.W.grad.abs().sum())
    checks["grad_prototypes"] = float(model.P.grad.abs().sum())
    checks["grad_temperature"] = float(model.temp_logit.grad.abs().sum())
    checks["task_grad_encoder"] = checks["grad_encoder"] > 0
    checks["task_grad_prototypes"] = checks["grad_prototypes"] > 0
    checks["task_grad_temperature"] = checks["grad_temperature"] > 0

    # Permutation invariance on a cached graph by jointly permuting rows and R.
    gi = idx[0]
    rec = records[gi]
    n = int(rec["n"])
    rng = np.random.default_rng(4242)
    perm = rng.permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    with torch.no_grad():
        single = V.make_batch(records, [gi], device)
        p0, c0, h0 = model.forward_padded(single)
        Xp = single["X_pad"][:, :n][..., :].index_select(1, torch.as_tensor(perm, device=device))
        Rp = single["R_pad"][:, :, :n, :n].index_select(2, torch.as_tensor(perm, device=device)).index_select(3, torch.as_tensor(perm, device=device))
        bp = dict(single)
        bp["X_pad"] = torch.nn.functional.pad(Xp, (0, 0, 0, single["X_pad"].shape[1] - n))
        bp["R_pad"] = torch.zeros_like(single["R_pad"])
        bp["R_pad"][:, :, :n, :n] = Rp
        p1, c1, h1 = model.forward_padded(bp)
    checks["permutation_prediction"] = float((p0 - p1).abs().max())
    checks["permutation_composition"] = float((h0 - h1).abs().max())
    checks["permutation_invariant"] = checks["permutation_prediction"] <= 1e-5 and checks["permutation_composition"] <= 1e-5

    bool_checks = {k: v for k, v in checks.items() if isinstance(v, (bool, np.bool_))}
    passed = bool(all(bool_checks.values()))
    out = {
        "protocol": V.PROTOCOL_VERSION,
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else None,
        "seed": args.seed,
        "n_graphs": len(idx),
        "layout": layout.descriptor(),
        "encoder_init": "reused_tccd_v1_dense_D0_714x64",
        "prototype_init_seed": V.PROTO_INIT_SEED,
        "prototype_fingerprint": V.prototype_fingerprint(model.P.detach().cpu().numpy()),
        "ksvd_refit_performed": False,
        "omp_or_iht_used": False,
        "reconstruction_loss_used": False,
        "official_test_loaded": False,
        "checks": checks,
        "gate0_pass": passed,
    }
    T.write_json(OUT_DIR / f"gate0_seed{args.seed}.json", out)
    log(f"[gate0] {'PASS' if passed else 'FAIL'}: {checks}")
    return 0 if passed else 1


def _train_arm(mode: str, args, records, tr_idx, dev_idx, device, log):
    layout = V.frozen_layout()
    model = _make_model(mode, args, layout, device)
    _peak_reset(device)
    res = V.train_model(
        model, records, records, list(tr_idx), list(dev_idx), device,
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
        batch=args.batch, log=log,
    )
    res.peak_mem_mb = _peak_mb(device)
    return model, res


def gate_a(args, log=print) -> int:
    device = args.device
    records, meta = _records(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    layout = V.frozen_layout()
    y = np.asarray([float(r["y"]) for r in records], dtype=np.float32)
    log(f"[gateA] graphs={len(records)} train={len(tr_idx)} dev={len(dev_idx)} seed={args.seed}")

    dense_model, dense_res, dense_timing = _train_dense(args, records, tr_idx, dev_idx, device, log)
    dense = {
        "best_valid": dense_res.best_valid,
        "best_epoch": dense_res.best_epoch,
        "soup_valid": dense_res.soup_valid,
        "soup_members": dense_res.soup_members,
        "wall_s": dense_timing["wall_s"],
        "peak_mem_mb": dense_timing["peak_mem_mb"],
        "source": "GPU1 rerun because prior TCCD-v1 DENSE result was local CPU; same split/encoder/relations/reader/optimizer",
        "historical_tccd_v1_best_valid": V.DENSE_REFERENCE_BEST,
        "historical_tccd_v1_soup_valid": V.DENSE_REFERENCE_SOUP,
    }

    model_rel, rel = _train_arm("rel", args, records, tr_idx, dev_idx, device, log)
    rel_best_path = _save_model_state(f"prototype_rel_seed{args.seed}_best", rel.state_best)
    rel_soup_path = _save_model_state(f"prototype_rel_seed{args.seed}_soup", rel.state_soup) if rel.state_soup is not None else None
    rel_shuffle = V.evaluate_mae(model_rel, records, dev_idx, device, batch=64, shuffle=True, seed=args.seed)

    model_bag, bag = _train_arm("bag", args, records, tr_idx, dev_idx, device, log)
    bag_best_path = _save_model_state(f"prototype_bag_seed{args.seed}_best", bag.state_best)
    delta_comp = float(rel_shuffle - rel.best_valid)
    delta_proto = float(rel.best_valid - dense["best_valid"])
    verdict = V.gate_a_verdict(delta_comp, delta_proto, args.seed)
    out = {
        "protocol": V.PROTOCOL_VERSION,
        "gate": "A",
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "gpu": T.provenance(device).get("gpu"),
        "seed": args.seed,
        "split_seed": T.SPLIT_SEED,
        "n_train": len(tr_idx),
        "n_dev": len(dev_idx),
        "n_patches": int(sum(int(r["n"]) for r in records)),
        "layout": layout.descriptor(),
        "encoder": "714-D raw local patch -> task-learned linear 64-D latent; shared across arms",
        "K": V.K_PROTO,
        "d": V.D_LOCAL,
        "temperature_parameterization": "0.05+0.95*sigmoid(a)",
        "regularization": "0.05 initial task scale each; calibrated once and frozen",
        "relations": "TCCD-v1 overlap + native bonds + global relative position",
        "ksvd_refit_performed": False,
        "omp_or_iht_used": False,
        "reconstruction_loss_used": False,
        "official_test_loaded": False,
        "dense_rel": dense,
        "prototype_bag": {
            "best_valid": bag.best_valid, "best_epoch": bag.best_epoch, "soup_valid": bag.soup_valid,
            "soup_members": bag.soup_members, "wall_s": bag.wall_s, "peak_mem_mb": bag.peak_mem_mb,
            "regularization": bag.regularization, "checkpoint": str(bag_best_path),
        },
        "prototype_rel": {
            "best_valid": rel.best_valid, "best_epoch": rel.best_epoch, "soup_valid": rel.soup_valid,
            "soup_members": rel.soup_members, "wall_s": rel.wall_s, "peak_mem_mb": rel.peak_mem_mb,
            "regularization": rel.regularization, "checkpoint_best": str(rel_best_path),
            "checkpoint_soup": str(rel_soup_path) if rel_soup_path else None,
        },
        "prototype_rel_shuffle": {"best_valid": rel_shuffle, "evaluation_only": True, "trained_checkpoint": str(rel_best_path)},
        "delta_comp": delta_comp,
        "delta_proto": delta_proto,
        "thresholds": {
            "composition_min": V.GATE_COMP_MIN, "prototype_pass": V.GATE_PROTO_PASS,
            "prototype_fail": V.GATE_PROTO_FAIL, "prototype_ambiguous_mean_pass": V.GATE_PROTO_AMBIG_MEAN_PASS,
        },
        "verdict": verdict,
    }
    T.write_json(OUT_DIR / f"gateA_seed{args.seed}.json", out)
    log(f"[gateA] Dense={dense['best_valid']:.6f} BAG={bag.best_valid:.6f} REL={rel.best_valid:.6f} SHUFFLE={rel_shuffle:.6f} delta_comp={delta_comp:.6f} delta_proto={delta_proto:.6f} -> {verdict}")
    return 0 if verdict == "PASS" else (3 if verdict == "AMBIGUOUS_NEEDS_SEED1" else 1)


def vocabulary(args, log=print) -> int:
    device = args.device
    gate = json.loads((OUT_DIR / f"gateA_seed{args.seed}.json").read_text())
    if gate.get("verdict") != "PASS":
        raise SystemExit("vocabulary stage requires Gate A PASS")
    records, _ = _records(args, log)
    _, dev_idx = internal_split(len(records))
    layout = V.frozen_layout()
    model = _make_model("rel", args, layout, device)
    state_path = Path(gate["prototype_rel"]["checkpoint_best"])
    model.load_state_dict(V.load_state(state_path, device))
    C_dev, mol_ids = V.collect_assignments(model, records, list(dev_idx), device, batch=64)
    usage = V.usage_metrics(C_dev, mol_ids)
    sem = V.semantic_coherence(C_dev, records, list(dev_idx))
    # Existing continuity diagnostic is report-only. It needs molecules but no labels.
    continuity = None
    try:
        mols, _ = T.load_mols(args.data_root, "train")
        P = model.P.detach().cpu().numpy().astype(np.float64)
        continuity = continuity_audit(mols, records, dev_idx, P, C_dev, log=log)
    except Exception as exc:
        continuity = {"skipped": True, "reason": repr(exc)}
    tau = float(model.temperature().detach())
    passed = usage["active_prototypes"] >= 48 and usage["top8_usage_mass"] <= 0.80
    out = {
        "protocol": V.PROTOCOL_VERSION, "gate": "B_vocabulary", "commit": T.git("rev-parse", "HEAD"),
        "device": device, "gpu": T.provenance(device).get("gpu"), "seed": args.seed,
        "official_test_loaded": False, "ksvd_refit_performed": False,
        "n_dev_graphs": len(dev_idx), "n_dev_patches": int(C_dev.shape[0]),
        "dead_threshold": V.DEAD_THRESHOLD, "top8_collapse_threshold": 0.80,
        "learned_temperature": tau, "usage": usage, "semantic_coherence": sem,
        "structural_coherence_diagnostics": continuity,
        "vocabulary_pass": bool(passed),
        "stopping_reason": None if passed else "prototype usage collapse",
    }
    T.write_json(OUT_DIR / f"vocabulary_seed{args.seed}.json", out)
    log(f"[vocab] active={usage['active_prototypes']} eff={usage['effective_prototype_count']:.2f} top8={usage['top8_usage_mass']:.4f} tau={tau:.5f} -> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gate0", "gateA", "vocabulary"])
    p.add_argument("--data-root", type=Path, default=T.REPO_ROOT / "data/ZINC")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-graphs", type=int, default=64)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "gateA":
        return gate_a(args)
    if args.stage == "vocabulary":
        return vocabulary(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
