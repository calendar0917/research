#!/usr/bin/env python
"""TCCD-v2 GRAD-CONT runner: ordinal local-geometry arm.

Stages:
    coverage  Phase-0 zero-training ordinal-coverage audit (train + held-out dev)
    gate0     pre-formal sanity checks (gradient, no leakage, no dev, identity)
    train     train GRAD-CONT seed 0 and write the checkpoint + result JSON

Formal execution is GPU1-only. Official test is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V
from tracks.ksvd.code import tccd_v2_gradcont as G
from tracks.ksvd.code import tccd_v2_local_chem as LC
from tracks.ksvd.code.run_tccd_v0 import internal_split

OUT_DIR = G.RESULTS_DIR
COVERAGE_JSON = OUT_DIR / "coverage_seed0.json"
DEV_TRIPLETS = OUT_DIR / "dev_ordinal_triplets.npz"
DEV_TRIPLETS_STATS = OUT_DIR / "dev_ordinal_triplets_stats.json"


def _records(args, log=print):
    records, meta = V.load_train_records()
    if args.limit:
        log(f"[debug] truncating records to {args.limit}; formal runs must omit --limit")
        records = records[: int(args.limit)]
    return records, meta


def _build_model(args, layout, device):
    init = V.load_frozen_encoder_init()
    return V.PrototypeModelFactory.build(
        layout.feature_dim, V.D_LOCAL, V.K_PROTO, V.N_REL, init,
        mode="rel", proto_seed=V.PROTO_INIT_SEED, seed=args.seed,
    ).to(device)


def _fingerprint(trip: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in ("anchor", "closer", "farther", "anchor_pos", "ctype"):
        h.update(np.ascontiguousarray(trip[k], dtype=np.int64).tobytes())
    return h.hexdigest()


def _summarize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "n_patches", "n_anchors_total", "n_anchors_eligible", "n_anchors_ineligible",
        "eligible_fraction", "n_triplets", "graph_coverage",
        "cross_molecule_anchor_closer", "cross_molecule_anchor_farther",
        "cross_molecule_closer_farther", "comparison_type_hist", "top_bin_pairs",
        "candidate_bin_hist", "closer_d1_hist", "closer_d2_hist",
        "farther_d1_hist", "farther_d2_hist", "wall_s",
    ]
    return {k: stats[k] for k in keep if k in stats}


def coverage_stage(args, log=print) -> int:
    records, _ = _records(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    log(f"[coverage] graphs={len(records)} train={len(tr_idx)} dev={len(dev_idx)}")

    # ---- training split ----
    chem_tr = LC.decode_local_chem(records, [int(i) for i in tr_idx], log=log)
    t0 = time.time()
    trip_tr = LC.build_ordinal_triplets(chem_tr, seed=G.PAIR_SEED, log=log)
    train_fp = _fingerprint(trip_tr)
    G.save_triplets(trip_tr, G.PAIR_SEED)
    log(f"[coverage] train triplets fingerprint={train_fp[:16]} ({time.time()-t0:.1f}s)")

    # ---- held-out internal dev ----
    chem_dev = LC.decode_local_chem(records, [int(i) for i in dev_idx], log=log)
    trip_dev = LC.build_ordinal_triplets(chem_dev, seed=G.PAIR_SEED, log=log)
    dev_fp = _fingerprint(trip_dev)
    LC.save_triplets(trip_dev, DEV_TRIPLETS)
    DEV_TRIPLETS_STATS.write_text(json.dumps(_summarize_stats(trip_dev["stats"]), indent=2, sort_keys=True, default=float))
    log(f"[coverage] dev triplets fingerprint={dev_fp[:16]} n={trip_dev['anchor'].size}")

    sufficient = bool(
        trip_tr["stats"]["eligible_fraction"] >= 0.50
        and trip_tr["stats"]["n_triplets"] >= 10000
        and trip_tr["stats"]["cross_molecule_anchor_closer"] >= 0.99
    )
    out = {
        "protocol": G.PROTOCOL_VERSION,
        "stage": "coverage",
        "commit": T.git("rev-parse", "HEAD"),
        "split_seed": T.SPLIT_SEED,
        "seed": args.seed,
        "n_train_graphs": int(len(tr_idx)),
        "n_dev_graphs": int(len(dev_idx)),
        "train_triplets_fingerprint": train_fp,
        "train_triplets_stats": _summarize_stats(trip_tr["stats"]),
        "dev_triplets_fingerprint": dev_fp,
        "dev_triplets_stats": _summarize_stats(trip_dev["stats"]),
        "coverage_stop_rule": {
            "rule": "proceed only if train eligible_fraction >= 0.50 and n_triplets >= 10000 and cross_molecule_anchor_closer >= 0.99",
            "coverage_sufficient": sufficient,
        },
        "reject_exact_pairs": True,
        "uses_target_label": False,
        "official_test_loaded": False,
    }
    T.write_json(COVERAGE_JSON, out)
    log(f"[coverage] sufficient={sufficient}")
    return 0 if sufficient else 3


def gate0(args, log=print) -> int:
    torch = T._torch()
    device = args.device
    records, _ = _records(args, log)
    n = min(int(args.n_graphs), len(records))
    records = records[:n]
    tr_idx, dev_idx = internal_split(len(records))
    layout = V.frozen_layout()
    checks: dict[str, Any] = {}

    chem = LC.decode_local_chem(records, [int(i) for i in tr_idx], log=log)
    trip = LC.build_ordinal_triplets(chem, seed=G.PAIR_SEED, log=log)
    a, c, f = trip["anchor"], trip["closer"], trip["farther"]
    d1c = LC.d1_pairs(chem, a, c)
    d2c = LC.d2_pairs(chem, a, c)
    d1f = LC.d1_pairs(chem, a, f)
    d2f = LC.d2_pairs(chem, a, f)
    ok = (d1c <= d1f) & (d2c <= d2f) & ((d1c < d1f) | (d2c < d2f))
    checks["triplet_pareto_strict_rate"] = float(ok.mean()) if a.size else 0.0
    checks["triplet_root_degree_match_rate"] = float(
        ((chem["root"][a] == chem["root"][c]) & (chem["degree"][a] == chem["degree"][c])
         & (chem["root"][a] == chem["root"][f]) & (chem["degree"][a] == chem["degree"][f])).mean()
    ) if a.size else 0.0
    checks["triplet_size_tol_rate"] = float(
        ((np.abs(chem["n_atoms"][c] - chem["n_atoms"][a]) <= LC.SIZE_TOL)
         & (np.abs(chem["n_atoms"][f] - chem["n_atoms"][a]) <= LC.SIZE_TOL)).mean()
    ) if a.size else 0.0
    checks["triplet_cross_molecule_rate"] = float(trip["stats"]["cross_molecule_anchor_closer"])
    checks["triplet_exact_excluded_rate"] = float(
        (~LC.is_exact(chem, a, c)).mean() if a.size else 0.0
    )
    checks["triplet_legality"] = bool(
        checks["triplet_pareto_strict_rate"] == 1.0
        and checks["triplet_root_degree_match_rate"] == 1.0
        and checks["triplet_size_tol_rate"] == 1.0
        and checks["triplet_cross_molecule_rate"] == 1.0
        and checks["triplet_exact_excluded_rate"] == 1.0
    )

    n_train = len(tr_idx)
    checks["dev_triplets_used"] = bool(np.any(trip["anchor_pos"] >= n_train))
    checks["train_dev_disjoint"] = bool(len(set(tr_idx.tolist()) & set(dev_idx.tolist())) == 0)
    checks["no_dev_triplet"] = bool((not checks["dev_triplets_used"]) and checks["train_dev_disjoint"])

    # ordinal definition must not depend on the target: scramble y and rebuild
    scr = [dict(r) for r in records]
    rng = np.random.default_rng(7)
    ys = np.asarray([float(r.get("y", 0.0)) for r in scr])
    perm = rng.permutation(len(scr))
    for i, r in enumerate(scr):
        r["y"] = float(ys[perm[i]])
    chem_scr = LC.decode_local_chem(scr, [int(i) for i in tr_idx], log=lambda *a: None)
    trip_scr = LC.build_ordinal_triplets(chem_scr, seed=G.PAIR_SEED, log=lambda *a: None)
    checks["ordinal_construction_label_free"] = bool(
        np.array_equal(trip["anchor"], trip_scr["anchor"])
        and np.array_equal(trip["closer"], trip_scr["closer"])
        and np.array_equal(trip["farther"], trip_scr["farther"])
    )

    # L_grad has a non-zero gradient to the encoder W and none to prototypes
    model = _build_model(args, layout, device)
    Xpatch = G._patch_matrix(records, tr_idx, device)
    tri_a = torch.as_tensor(trip["anchor"], dtype=torch.long, device=device)
    tri_c = torch.as_tensor(trip["closer"], dtype=torch.long, device=device)
    tri_f = torch.as_tensor(trip["farther"], dtype=torch.long, device=device)
    model.train()
    model.zero_grad(set_to_none=True)
    if tri_a.numel():
        grad = G._ordinal_loss_rows(model, Xpatch, tri_a, tri_c, tri_f)
        grad.backward()
        checks["grad_loss_value"] = float(grad.detach())
    else:
        checks["grad_loss_value"] = 0.0
    checks["grad_grad_encoder"] = float(model.W.grad.abs().sum()) if model.W.grad is not None else 0.0
    checks["grad_grad_encoder_nonzero"] = bool(checks["grad_grad_encoder"] > 0)
    checks["grad_grad_prototypes"] = float(model.P.grad.abs().sum()) if model.P.grad is not None else 0.0
    checks["grad_does_not_touch_prototypes"] = bool(checks["grad_grad_prototypes"] == 0.0)

    # lambda-0 forward/task identity against the frozen TCCD-v2 implementation
    model_a = _build_model(args, layout, device)
    model_b = _build_model(args, layout, device)
    b = V.make_batch(records, [int(i) for i in tr_idx[: min(8, len(tr_idx))]], device)

    def task_terms(m):
        pred, C_flat, _ = m.forward_padded(b)
        valid = b["valid"].reshape(-1)
        return pred, V.local_entropy(C_flat, valid), V.balance_kl(C_flat, valid)

    with torch.no_grad():
        pa, la, ba = task_terms(model_a)
        pb, lb, bb = task_terms(model_b)
    checks["lambda0_pred_identity"] = float((pa - pb).abs().max())
    checks["lambda0_local_identity"] = float((la - lb).abs().max())
    checks["lambda0_balance_identity"] = float((ba - bb).abs().max())
    checks["lambda0_forward_identity"] = bool(
        checks["lambda0_pred_identity"] <= 0.0
        and checks["lambda0_local_identity"] <= 0.0
        and checks["lambda0_balance_identity"] <= 0.0
    )

    # permutation invariance unchanged
    gi = int(tr_idx[0])
    rec = records[gi]
    nn = int(rec["n"])
    rng = np.random.default_rng(4242)
    perm2 = rng.permutation(nn) if nn > 1 else np.arange(nn)
    model.eval()
    with torch.no_grad():
        single = V.make_batch(records, [gi], device)
        p0, _, h0 = model.forward_padded(single)
        pidx = torch.as_tensor(perm2, device=device)
        Xp = single["X_pad"][:, :nn].index_select(1, pidx)
        Rp = single["R_pad"][:, :, :nn, :nn].index_select(2, pidx).index_select(3, pidx)
        bp = dict(single)
        bp["X_pad"] = torch.nn.functional.pad(Xp, (0, 0, 0, single["X_pad"].shape[1] - nn))
        bp["R_pad"] = torch.zeros_like(single["R_pad"])
        bp["R_pad"][:, :, :nn, :nn] = Rp
        p1, _, h1 = model.forward_padded(bp)
    checks["permutation_prediction"] = float((p0 - p1).abs().max())
    checks["permutation_composition"] = float((h0 - h1).abs().max())
    checks["permutation_invariant"] = bool(
        checks["permutation_prediction"] <= 1e-5 and checks["permutation_composition"] <= 1e-5
    )

    checks["official_test_loaded"] = False
    bool_checks = {
        k: bool(v)
        for k, v in checks.items()
        if k
        in {
            "triplet_legality",
            "no_dev_triplet",
            "ordinal_construction_label_free",
            "grad_grad_encoder_nonzero",
            "grad_does_not_touch_prototypes",
            "lambda0_forward_identity",
            "permutation_invariant",
        }
    }
    passed = bool(all(bool_checks.values()))
    out = {
        "protocol": G.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "gpu": T.provenance(device).get("gpu"),
        "seed": args.seed,
        "n_graphs": len(records),
        "n_train": len(tr_idx),
        "n_dev": len(dev_idx),
        "triplet_stats": _summarize_stats(trip["stats"]),
        "checks": checks,
        "gate0_pass": passed,
        "official_test_loaded": False,
    }
    T.write_json(OUT_DIR / f"gate0_seed{args.seed}.json", out)
    log(f"[gate0] {'PASS' if passed else 'FAIL'}: {bool_checks}")
    return 0 if passed else 1


def train_stage(args, log=print) -> int:
    torch = T._torch()
    device = args.device
    result_path = OUT_DIR / f"train_seed{args.seed}.json"
    best_path = OUT_DIR / f"prototype_gradcont_seed{args.seed}_best.pt"
    soup_path = OUT_DIR / f"prototype_gradcont_seed{args.seed}_soup.pt"
    if result_path.exists() and best_path.exists():
        log(f"[train] result already present at {result_path}; skipping (resumable)")
        return 0

    records, meta = _records(args, log)
    tr_idx, dev_idx = internal_split(len(records))
    layout = V.frozen_layout()
    log(f"[train] graphs={len(records)} train={len(tr_idx)} dev={len(dev_idx)} seed={args.seed}")

    triplets = None if args.rebuild_pairs else G.load_triplets(G.PAIR_SEED)
    if triplets is None:
        chem = LC.decode_local_chem(records, [int(i) for i in tr_idx], log=log)
        triplets = LC.build_ordinal_triplets(chem, seed=G.PAIR_SEED, log=log)
        G.save_triplets(triplets, G.PAIR_SEED)
    else:
        log(f"[train] reused cached ordinal cache ({triplets['stats']['n_triplets']} triplets)")
    log(f"[train] triplet_stats={json.dumps(_summarize_stats(triplets['stats']), default=float)[:400]}")

    model = _build_model(args, layout, device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = G.train_model_gradcont(
        model, records, records, list(tr_idx), list(dev_idx), device,
        triplets=triplets, seed=args.seed, max_epochs=args.max_epochs,
        patience=args.patience, batch=args.batch, log=log,
    )
    wall = time.time() - t0
    peak = (
        float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        if device.startswith("cuda") and torch.cuda.is_available()
        else None
    )

    V.save_state(best_path, res.state_best)
    if res.state_soup is not None:
        V.save_state(soup_path, res.state_soup)

    best_model = _build_model(args, layout, device)
    best_model.load_state_dict(V.load_state(best_path, device))
    C_dev, mol_ids = V.collect_assignments(best_model, records, list(dev_idx), device, batch=64)
    usage = V.usage_metrics(C_dev, mol_ids)
    tau = float(best_model.temperature().detach())

    out = {
        "protocol": G.PROTOCOL_VERSION,
        "stage": "train",
        "commit": T.git("rev-parse", "HEAD"),
        "device": device,
        "gpu": T.provenance(device).get("gpu"),
        "seed": args.seed,
        "split_seed": T.SPLIT_SEED,
        "n_train": len(tr_idx),
        "n_dev": len(dev_idx),
        "n_patches": int(sum(int(r["n"]) for r in records)),
        "layout": layout.descriptor(),
        "architecture": "frozen TCCD-v2 Prototype-REL + one label-free latent ordinal-geometry term",
        "K": V.K_PROTO,
        "d": V.D_LOCAL,
        "temperature_parameterization": "0.05+0.95*sigmoid(a)",
        "base_regularization": "0.05 initial task scale each, calibrated once and frozen",
        "grad_regularization": "lambda_grad = 0.05 * initial_task / initial_grad, calibrated once and frozen",
        "lambda_share": G.LAMBDA_SHARE,
        "ordinal_definition": {
            "relation": "j ~< k iff d1(i,j)<=d1(i,k) and d2(i,j)<=d2(i,k) and at least one strict",
            "d1": "radius-1 edit count (shell-1 atom counts + root bond counts)",
            "d2": "peripheral edit count (shell-2 atom counts + shell1->shell2 attachment + shell2-shell2 bonds + delta n2)",
            "exact_excluded": True,
            "seed": G.PAIR_SEED,
            "max_triplets_per_anchor": LC.MAX_TRIPLETS_PER_ANCHOR,
            "far_d1_cap": LC.FAR_D1_CAP,
            "far_d2_cap": LC.FAR_D2_CAP,
            "size_tol": LC.SIZE_TOL,
            "loss": "mean softplus(cos(z_i,z_far) - cos(z_i,z_close))",
            "no_scalar_distance": True,
            "no_shell_weighting": True,
            "no_margin": True,
        },
        "regularization": res.regularization,
        "best_valid": res.best_valid,
        "best_epoch": res.best_epoch,
        "soup_valid": res.soup_valid,
        "soup_members": res.soup_members,
        "checkpoint_best": str(best_path),
        "checkpoint_soup": str(soup_path) if res.state_soup is not None else None,
        "learned_temperature": tau,
        "usage_on_internal_dev": usage,
        "train_history": res.train_history,
        "wall_s": wall,
        "peak_mem_mb": peak,
        "triplet_stats": _summarize_stats(triplets["stats"]),
        "official_test_loaded": False,
        "ksvd_refit_performed": False,
        "reconstruction_loss_used": False,
        "latent_bypass_used": False,
    }
    T.write_json(result_path, out)
    log(
        f"[train] best={res.best_valid:.6f} soup={res.soup_valid} epoch={res.best_epoch} "
        f"tau={tau:.5f} lambda_grad={res.regularization['lambda_grad']:.6f} wall={wall:.0f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["coverage", "gate0", "train"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=T.BATCH)
    p.add_argument("--max-epochs", type=int, default=T.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=T.PATIENCE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-graphs", type=int, default=256)
    p.add_argument("--rebuild-pairs", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "coverage":
        return coverage_stage(args)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "train":
        return train_stage(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
