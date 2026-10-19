"""SSRC Phase 2a runner — P3E + {NodeFFN-2, LocalMP-2} on the internal ZINC split.

Protocol: ``notes/phase2a_protocol.md`` (``ssrc-phase2a-zinc-v1``).

Stages
------
``replay``   verify the frozen Phase 1e artefacts and reproduce the frozen P3E
             Monitor predictions from the frozen soup (no training).
``gates``    the ten pre-GPU CPU correctness gates (no labels, no Monitor read).
``flops``    forward GEMM FLOPs for P3E / F / M on one fixed Fit batch.
``train``    one arm, one seed: frozen R3 recipe, shared Fit-only SSL encoder.
``predict``  freeze Monitor predictions + label-free usage probes (no labels).
``score``    the single Monitor label read; subgroup metrics + decision.

Every stage refuses to overwrite an existing artefact: the freeze-before-score
discipline is enforced in code, not only documented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (HERE, REPO / "tracks/structure-contribution/code", REPO / "tracks/structural-sufficiency/code/phase1b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase2a_common as C  # noqa: E402
import phase2a_models as M  # noqa: E402

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

OUT = C.RESULTS_2A
ARM_KINDS = {"F": M.KIND_NODE_FFN, "M": M.KIND_LOCAL_MP}
ALLOWED_BATCH_KEYS = {
    "n_graphs",
    "atom_feat",
    "atom_batch",
    "bond_feat",
    "bond_batch",
    "bond_index",
    "junc_feat",
    "junc_batch",
    "junc_index",
    "junc_bond",
    "global_feat",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _paths(arm: str, seed: int) -> dict[str, Path]:
    tag = f"{arm}_s{seed}"
    return {
        "train": OUT / f"train_{tag}.json",
        "soup": OUT / f"train_{tag}_soup.pt",
        "pred": OUT / f"monitor_pred_{tag}.npz",
        "freeze": OUT / f"monitor_freeze_{tag}.json",
    }


def _objects(rows: Sequence[int], *, which: str, allow_rebuild: bool = True):
    manifest = C.read_json(C.P3E_PREP_MANIFEST) if C.P3E_PREP_MANIFEST.exists() else {}
    key = "objects_fit_sha256" if which == "fit" else "objects_monitor_sha256"
    expected = manifest.get(key)
    path = C.BASELINE_DIR / ("objects_fit.pt" if which == "fit" else "objects_monitor.pt")
    return C.load_objects(path, rows, expected_sha=expected, allow_rebuild=allow_rebuild)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> int:
    out = OUT / "replay.json"
    if out.exists():
        raise SystemExit("replay record exists; refusing to rewrite")
    split = C.load_split()
    checks: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "replay",
        "split_sha256": split.sha256,
        "monitor_rows_sha256": C.rows_sha256(split.monitor),
        "ssl_state": C.sha256_file(C.SSL_STATE),
        "soup_sha256": C.sha256_file(C.P3E_SOUP),
        "p3e_prediction_file_sha256": C.sha256_file(C.P3E_PRED),
    }
    if checks["ssl_state"] != C.P3E_SSL_FILE_SHA:
        raise SystemExit("SSL state hash mismatch")
    if checks["soup_sha256"] != C.P3E_SOUP_SHA:
        raise SystemExit("P3E soup hash mismatch")
    if checks["p3e_prediction_file_sha256"] != C.P3E_PRED_SHA:
        raise SystemExit("P3E prediction hash mismatch")

    norm = C.read_json(C.BASELINE_DIR / "prep_fit.json")
    if not norm.get("fit_only"):
        raise SystemExit("standardiser is not marked Fit-only")
    base, model, info = M.build_local_model(M.KIND_NONE, seed=0, norm=norm, device=args.device)
    if not info["trunk_matches_base_init"]:
        raise SystemExit("extended trunk does not match the frozen P3E scratch init")
    p3e_run = C.read_json(C.P3E_RUN)
    if info["base_scratch_init_full_sha256"] != p3e_run["init_full_sha256"]:
        raise SystemExit("scratch init hash disagrees with the frozen P3E run")
    if info["base_scratch_init_group_state_sha256"] != p3e_run["init_group_state_sha256"]:
        raise SystemExit("composer/reader init hash disagrees with the frozen P3E run")
    if info["base_scratch_init_encoder_sha256"] != p3e_run["init_encoder_sha256"]:
        raise SystemExit("encoder init hash disagrees with the frozen P3E run")

    ssl_state, ssl_meta = C.load_ssl_state()
    enc = C.verify_encoder_state(model, ssl_state)
    soup = torch.load(C.P3E_SOUP, map_location=args.device, weights_only=False)
    model.load_state_dict(soup, strict=True)

    mon_objs, mon_src = _objects(split.monitor, which="monitor")
    pred = bp.predict_model(model, mon_objs, args.device)
    frozen = C.load_p3e_predictions()
    max_abs = float(np.abs(pred - frozen.pred).max())
    # The frozen prediction was produced on the same GPU model but scatter-add
    # order is not bit-reproducible; a float32 replay is expected to agree to
    # well below 1e-5, not bitwise.
    if max_abs > 1e-5:
        raise SystemExit(f"P3E replay disagrees with the frozen predictions: {max_abs}")

    labels = C.load_vault()
    y = C.row_labels(split.monitor, labels)
    mae = float(np.abs(pred - y).mean())
    mae_tol = 1e-6
    if abs(mae - C.P3E_MAE) > mae_tol:
        raise SystemExit(f"replayed P3E MAE {mae!r} != frozen {C.P3E_MAE!r} (tol {mae_tol})")

    payload = {
        **checks,
        "encoder": enc,
        "ssl_meta": ssl_meta,
        "trunk_init_audit": info,
        "monitor_objects": mon_src,
        "replayed_prediction_array_sha256": C.sha256_array(pred),
        "frozen_prediction_array_sha256": C.sha256_array(frozen.pred),
        "max_abs_prediction_diff": max_abs,
        "replay_tolerance": 1e-5,
        "replay_tolerance_note": (
            "GPU scatter-add order is not bit-reproducible; the frozen array is the "
            "reference and agreement is float32-level"
        ),
        "replayed_monitor_mae": mae,
        "frozen_monitor_mae": C.P3E_MAE,
        "mae_abs_diff": abs(mae - C.P3E_MAE),
        "mae_tolerance": 1e-6,
        "monitor_rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "p3e_reusable": True,
        "retrained_p3e": False,
        "monitor_labels_read_for_replay_only": True,
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
        "device": str(args.device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.device(args.device).type == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    C.write_json(out, payload)
    log(f"[replay] P3E reproduced: max|d|={max_abs:.3e} MAE={mae:.9f} device={args.device}")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _synthetic_molecules() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(name, x, edge_index, edge_attr) for a ring, a chain and a fused bicycle."""

    def pack(n: int, bonds: list[tuple[int, int, int]], types: list[int]):
        x = np.asarray(types, dtype=np.int64)
        ei = []
        ea = []
        for u, v, bt in bonds:
            ei.append([u, v])
            ei.append([v, u])
            ea.append(bt)
            ea.append(bt)
        return x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64)

    benzene = [(i, (i + 1) % 6, 3) for i in range(6)]
    chain = [(i, i + 1, 0) for i in range(7)]
    # naphthalene-like fused bicyclic, all aromatic
    fused = [
        (0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3), (4, 5, 3), (5, 0, 3),
        (1, 6, 3), (6, 7, 3), (7, 8, 3), (8, 9, 3), (9, 4, 3), (4, 1, 3),
    ]
    out = [
        ("ring6", *pack(6, benzene, [1, 1, 1, 1, 1, 1])),
        ("chain7", *pack(8, chain, [0, 0, 0, 0, 0, 0, 0, 0])),
        ("fused", *pack(10, fused, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])),
    ]
    return [
        (name, sro.build_mol_objects(i, x, ei, ea)) for i, (name, x, ei, ea) in enumerate(out)
    ]


def _fresh_model(kind: str, seed: int = 0, norm: Mapping[str, Any] | None = None):
    norm = norm or {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }
    return M.build_local_model(kind, seed=seed, norm=norm, device="cpu")


def _batch(objs: Sequence[sro.MolObjects]) -> dict[str, torch.Tensor]:
    b = sro.batch_objects(list(objs))
    bad = set(b) - ALLOWED_BATCH_KEYS
    if bad:
        raise SystemExit(f"unexpected batch keys: {sorted(bad)}")
    return b


def _reindex_objects(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Apply an atom relabelling to a molecule and rebuild its objects."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    n = obj.n_atoms
    # recover a raw edge list with types from the canonical bond objects
    pairs = [(int(u), int(v), int(t)) for (u, v), t in zip(obj.bond_index, obj.bond_type)]
    ei = []
    ea = []
    for u, v, t in pairs:
        ei += [[int(inv[u]), int(inv[v])], [int(inv[v]), int(inv[u])]]
        ea += [t, t]
    # atom types: the first ATOM_CATEGORIES block of atom_feat is a one-hot
    atom_type = obj.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1)
    x = atom_type[perm]
    return sro.build_mol_objects(
        obj.mol_index,
        np.asarray(x, dtype=np.int64),
        np.asarray(ei, dtype=np.int64).T,
        np.asarray(ea, dtype=np.int64),
    )


def cmd_gates(args: argparse.Namespace) -> int:
    out = OUT / "cpu_gates_phase2a.json"
    if out.exists():
        raise SystemExit("gate record exists; refusing to rewrite")
    torch.manual_seed(0)
    gates: dict[str, Any] = {}
    mols = _synthetic_molecules()
    names = [n for n, _ in mols]
    objs = [o for _, o in mols]

    models = {k: _fresh_model(k)[1] for k in (M.KIND_NONE, M.KIND_NODE_FFN, M.KIND_LOCAL_MP)}
    for m in models.values():
        m.eval()

    # --- G1: row / endpoint mapping ---------------------------------------
    checks = []
    for name, obj in mols:
        ok = True
        for (u, v) in obj.bond_index:
            ok &= int(u) < int(v) < obj.n_atoms
        # every bond appears exactly once, and each junction references bonds
        ok &= len({tuple(map(int, p)) for p in obj.bond_index}) == obj.bond_index.shape[0]
        for j in range(obj.junc_index.shape[0]):
            ok &= bool(all(0 <= int(x) < obj.n_atoms for x in obj.junc_index[j]))
            ok &= bool(all(0 <= int(x) < obj.bond_index.shape[0] for x in obj.junc_bond[j]))
        checks.append({"molecule": name, "ok": bool(ok)})
    # Arm M's directed edge list must be exactly forward + reverse of bond_index
    batch = _batch(objs)
    m_model = models[M.KIND_LOCAL_MP]
    src_idx, dst_idx = m_model.local.directed_edges(batch)
    bi = batch["bond_index"]
    forward_ok = torch.equal(src_idx[: bi.shape[0]], bi[:, 0]) and torch.equal(dst_idx[: bi.shape[0]], bi[:, 1])
    reverse_ok = torch.equal(src_idx[bi.shape[0] :], bi[:, 1]) and torch.equal(dst_idx[bi.shape[0] :], bi[:, 0])
    gates["G1_mapping"] = {
        "ok": all(c["ok"] for c in checks) and forward_ok and reverse_ok,
        "molecules": checks,
        "directed_edges_are_forward_plus_reverse": bool(forward_ok and reverse_ok),
    }

    # --- G2: relabelling / direction / batch-order invariance --------------
    inv_rows = {}
    for name, obj in mols:
        rng = np.random.default_rng(12345 + int(obj.mol_index))
        perm = rng.permutation(obj.n_atoms)
        relabelled = _reindex_objects(obj, perm)
        b0 = _batch([obj])
        b1 = _batch([relabelled])
        row = {}
        for key, model in models.items():
            with torch.no_grad():
                p0 = float(model(b0)[0])
                p1 = float(model(b1)[0])
            row[key] = abs(p0 - p1)
        inv_rows[name] = row
    # edge direction swap: reverse the stored columns of a doubled edge list
    swap = 0.0
    for name, obj in mols:
        pairs = [(int(u), int(v), int(t)) for (u, v), t in zip(obj.bond_index, obj.bond_type)]
        ei = []
        ea = []
        for u, v, t in pairs:
            ei += [[v, u], [u, v]]  # doubled, both orders
            ea += [t, t]
        atom_type = obj.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1)
        swapped = sro.build_mol_objects(obj.mol_index, atom_type, np.asarray(ei).T, np.asarray(ea))
        b0 = _batch([obj])
        b1 = _batch([swapped])
        with torch.no_grad():
            for model in models.values():
                swap = max(swap, abs(float(model(b0)[0]) - float(model(b1)[0])))
    # batch reorder
    order = [2, 0, 1]
    b_all = _batch(objs)
    b_perm = _batch([objs[i] for i in order])
    reorder = {}
    for key, model in models.items():
        with torch.no_grad():
            p_all = model(b_all).detach()
            p_perm = model(b_perm).detach()
        reorder[key] = float((p_all[order] - p_perm).abs().max())
    gates["G2_invariance"] = {
        "ok": max(max(r.values()) for r in inv_rows.values()) <= 1e-5
        and swap <= 1e-5
        and max(reorder.values()) <= 1e-5,
        "relabelling_max_abs": inv_rows,
        "edge_direction_swap_max_abs": swap,
        "batch_reorder_max_abs": reorder,
    }

    # --- G3/G4: adjacency and bond-attribute sensitivity of the branch -----
    base_batch = _batch(objs)
    with torch.no_grad():
        a0 = models[M.KIND_NONE].encode_atoms(base_batch)
        a_f = models[M.KIND_NODE_FFN].encode_atoms(base_batch)
        a_m = models[M.KIND_LOCAL_MP].encode_atoms(base_batch)
    m_model = models[M.KIND_LOCAL_MP]
    f_model = models[M.KIND_NODE_FFN]
    b_index = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_batch.items()}
    b_index["bond_index"] = torch.roll(b_index["bond_index"], shifts=1, dims=0)
    b_feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_batch.items()}
    b_feat["bond_feat"] = torch.roll(b_feat["bond_feat"], shifts=3, dims=0)
    perm = torch.randperm(a0.shape[0], generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        # Arm F must be bitwise blind to every adjacency/bond-attribute mutation
        f_changes = {
            "bond_index_roll": float((f_model.encode_atoms(b_index) - a_f).abs().max()),
            "bond_feat_roll": float((f_model.encode_atoms(b_feat) - a_f).abs().max()),
        }
        # Arm M: isolate the message aggregate itself, then the full block output
        msg0 = m_model.local.messages(a0, base_batch)
        msg_probe = {
            "endpoint_roll": float((m_model.local.messages(a0, b_index) - msg0).abs().max()),
            "bond_feature_roll": float((m_model.local.messages(a0, b_feat) - msg0).abs().max()),
            "neighbour_state_permutation": float(
                (m_model.local.messages(a0[perm], base_batch) - msg0).abs().max()
            ),
        }
        msg_probe_mean = {
            "endpoint_roll": float((m_model.local.messages(a0, b_index) - msg0).abs().mean()),
            "bond_feature_roll": float((m_model.local.messages(a0, b_feat) - msg0).abs().mean()),
            "neighbour_state_permutation": float(
                (m_model.local.messages(a0[perm], base_batch) - msg0).abs().mean()
            ),
        }
        m_changes = {
            "bond_index_roll": float((m_model.encode_atoms(b_index) - a_m).abs().max()),
            "bond_feat_roll": float((m_model.encode_atoms(b_feat) - a_m).abs().max()),
            "neighbour_state_permutation": float(
                (m_model.local(a0[perm], base_batch) - a_m).abs().max()
            ),
        }
    # neighbour-state gradient: d(local_mp output)/d(neighbour atom state) != 0
    a_in = a0.clone().requires_grad_(True)
    out_m = m_model.local(a_in, base_batch)
    out_m.sum().backward()
    nb_grad = a_in.grad.detach()
    msg_scale = float(msg0.abs().max())
    gates["G3_G4_branch_sensitivity"] = {
        "ok": max(f_changes.values()) <= 1e-7
        and all(v > 1e-4 for v in msg_probe.values())
        and all(v > 1e-4 for v in m_changes.values())
        and float(nb_grad.abs().max()) > 0.0,
        "node_ffn_max_abs_change_under_adjacency_mutation": f_changes,
        "local_mp_message_max_abs_change": msg_probe,
        "local_mp_message_mean_abs_change": msg_probe_mean,
        "local_mp_output_max_abs_change": m_changes,
        "message_scale_max_abs": msg_scale,
        "neighbour_input_grad_max_abs": float(nb_grad.abs().max()),
        "note": (
            "endpoint roll keeps raw features and the edge multiset but breaks the "
            "bond->endpoint correspondence; neighbour_state_permutation keeps the graph "
            "and changes only which atom state sits at each node"
        ),
    }

    # --- G5: the branch is not a copy of the P3E atom state -----------------
    shifts = {
        M.KIND_NODE_FFN: M.local_branch_shift(a0, a_f),
        M.KIND_LOCAL_MP: M.local_branch_shift(a0, a_m),
    }
    gates["G5_not_a_copy"] = {
        "ok": all(v["relative_shift"] > 1e-3 and v["cosine"] < 0.999 for v in shifts.values()),
        "shifts": shifts,
    }

    # --- G6: every new parameter receives gradient; trunk too --------------
    b = base_batch
    y = torch.zeros(b["n_graphs"])
    grads = {}
    for key, model in ((M.KIND_NODE_FFN, models[M.KIND_NODE_FFN]), (M.KIND_LOCAL_MP, models[M.KIND_LOCAL_MP])):
        model.zero_grad(set_to_none=True)
        loss = (model(b).view(-1) ** 2).mean()
        loss.backward()
        new = {n: float(p.grad.abs().sum()) for n, p in model.named_parameters() if n.startswith("local.")}
        old = {n: float(p.grad.abs().sum()) for n, p in model.named_parameters() if not n.startswith("local.")}
        grads[key] = {
            "new_params": len(new),
            "new_min_abs_grad_sum": min(new.values()),
            "new_zero_params": [n for n, v in new.items() if v == 0.0],
            "trunk_min_abs_grad_sum": min(old.values()),
            "all_new_nonzero": all(v > 0.0 for v in new.values()),
        }
    gates["G6_gradients"] = {
        "ok": all(g["all_new_nonzero"] and g["trunk_min_abs_grad_sum"] > 0 for g in grads.values()),
        "per_arm": grads,
    }

    # --- G7: shared init parity across arms --------------------------------
    info = {k: models[k].config for k in models}
    inits = {}
    for key, model in models.items():
        inits[key] = {
            "trunk_param_sha256": M.trunk_param_sha256(model),
            "local_block_sha256": M.local_block_state_sha256(model.local),
            "params": M.parameter_audit(model)["actual_parameters"],
        }
    base_hash = inits[M.KIND_NONE]["trunk_param_sha256"]
    gates["G7_init_parity"] = {
        "ok": all(v["trunk_param_sha256"] == base_hash for v in inits.values())
        and inits[M.KIND_NODE_FFN]["local_block_sha256"] == inits[M.KIND_LOCAL_MP]["local_block_sha256"]
        and inits[M.KIND_NODE_FFN]["params"] == 463489
        and inits[M.KIND_LOCAL_MP]["params"] == 464257,
        "per_kind": inits,
        "parameter_delta": inits[M.KIND_LOCAL_MP]["params"] - inits[M.KIND_NODE_FFN]["params"],
        "parameter_delta_frac_of_local": (inits[M.KIND_LOCAL_MP]["params"] - inits[M.KIND_NODE_FFN]["params"])
        / (inits[M.KIND_NODE_FFN]["params"] - C.P3E_PARAMS),
    }

    # --- G8: no label / target / split leakage in the forward path ---------
    src_text = (HERE / "phase2a_models.py").read_text(encoding="utf-8")
    banned = ["zinc_labels", "reserved", "official_test", "cycle_component", '"y"', "monitor_pred"]
    hits = [t for t in banned if t in src_text]
    gates["G8_no_label_input"] = {
        "ok": not hits,
        "banned_tokens_found": hits,
        "batch_keys": sorted(base_batch),
        "forward_reads_only_batch": True,
    }

    # --- G9: identity insertion reproduces the published forward -----------
    p3e = models[M.KIND_NONE]
    b = base_batch
    with torch.no_grad():
        p_ref = p3e(b).detach()
        feats = p3e.forward_features(b)
        p_again = p3e.reader(feats["reader_in"]).view(-1)
    # inserting a local module must keep the reader layout; a zero-output local
    # module would be a gate (forbidden), so we instead assert the published
    # reader input width and block order are untouched by construction.
    reader_in_local = models[M.KIND_NODE_FFN].forward_features(b)["reader_in"]
    gates["G9_reader_layout"] = {
        "ok": float((p_ref - p_again).abs().max()) == 0.0
        and reader_in_local.shape == feats["reader_in"].shape
        and reader_in_local.shape[1] == 96 + 2 * 96 + sro.GLOBAL_DIM,
        "base_forward_recompute_max_abs": float((p_ref - p_again).abs().max()),
        "reader_in_dim": int(reader_in_local.shape[1]),
        "expected_reader_in_dim": 96 + 2 * 96 + sro.GLOBAL_DIM,
    }

    # --- G10: forward/backward on ring, chain, fused, batch ----------------
    fwd = {}
    for key, model in models.items():
        b = base_batch
        model.zero_grad(set_to_none=True)
        pred = model(b)
        loss = (pred**2).mean()
        loss.backward()
        fwd[key] = {
            "pred_finite": bool(torch.isfinite(pred).all()),
            "grad_finite": bool(
                all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            ),
            "pred": [float(v) for v in pred],
        }
    gates["G10_execution"] = {
        "ok": all(v["pred_finite"] and v["grad_finite"] for v in fwd.values()),
        "molecules": names,
        "per_kind": fwd,
    }

    ok = all(bool(v.get("ok")) for v in gates.values())
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "cpu_gates",
        "status": "PASS" if ok else "FAIL",
        "n_gates": len(gates),
        "n_passed": sum(1 for v in gates.values() if v.get("ok")),
        "gates": gates,
        "synthetic_molecules": names,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    C.write_json(out, payload)
    log(f"[gates] {payload['n_passed']}/{payload['n_gates']} -> {payload['status']}")
    if not ok:
        for name, v in gates.items():
            if not v.get("ok"):
                log(f"[gates] FAILED {name}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def cmd_flops(args: argparse.Namespace) -> int:
    out = OUT / "flops_phase2a.json"
    if out.exists():
        raise SystemExit("flops record exists; refusing to rewrite")
    split = C.load_split()
    fit_objs, src_info = _objects(split.fit[:128], which="fit")
    batch = sro.batch_objects(fit_objs)
    rows = [int(r) for r in split.fit[:128]]
    norm = {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }
    out_rows = {}
    for key, kind in (("P3E", M.KIND_NONE), ("F", M.KIND_NODE_FFN), ("M", M.KIND_LOCAL_MP)):
        model = _fresh_model(kind, norm=norm)[1]
        flops = M.forward_gemm_flops(model, batch)
        out_rows[key] = {
            "params": M.parameter_audit(model)["actual_parameters"],
            "forward_gemm_flops": flops["total"],
            "by_module": flops["by_module"],
        }
    for key in ("F", "M"):
        out_rows[key]["flops_ratio_vs_p3e"] = out_rows[key]["forward_gemm_flops"] / out_rows["P3E"]["forward_gemm_flops"]
        out_rows[key]["param_ratio_vs_p3e"] = out_rows[key]["params"] / out_rows["P3E"]["params"]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "flops",
        "batch_rows": rows,
        "batch_rows_sha256": C.rows_sha256(rows),
        "n_atoms": int(batch["atom_feat"].shape[0]),
        "n_bonds": int(batch["bond_feat"].shape[0]),
        "n_junctions": int(batch["junc_feat"].shape[0]),
        "objects": src_info,
        "counter": "torch.utils.flop_counter.FlopCounterMode (matmul/conv FLOPs; scatter-add not counted)",
        "models": out_rows,
    }
    C.write_json(out, payload)
    log(
        "[flops] "
        + " ".join(f"{k}={v['forward_gemm_flops'] / 1e9:.3f}G" for k, v in out_rows.items())
        + f" ratioF={out_rows['F']['flops_ratio_vs_p3e']:.3f} ratioM={out_rows['M']['flops_ratio_vs_p3e']:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _grad_group_norms(model: torch.nn.Module) -> dict[str, float]:
    groups: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        group = parts[0]
        if group == "local":
            group = "local." + (parts[1] if parts[1] == "bond_proj" else ".".join(parts[1:3]))
        groups[group] = groups.get(group, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(groups.items())}


def train_arm(
    *,
    arm: str,
    kind: str,
    seed: int,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: torch.device,
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]],
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    """The frozen ``bp.train_supervised`` loop with a parameterised model."""
    recipe = bp.RECIPES["R3"]
    base, model, build_info = M.build_local_model(kind, seed=seed, norm=norm, device="cpu")
    n_params = M.parameter_audit(model)["actual_parameters"]
    expected_params = base.config.inventory()["total"] + M.local_parameters(kind)
    if n_params != expected_params:
        raise SystemExit(f"parameter audit failed: {n_params} != {expected_params}")

    enc_init = bp.encoder_group_hash(model)
    trunk_init = M.trunk_param_sha256(model)
    group_init = M.group_state_sha256(model)
    local_init = M.local_state_sha256(model.local)
    local_block_init = M.local_block_state_sha256(model.local)
    C.verify_encoder_state(model, ssl_state)
    encoder_used = bp.encoder_group_hash(model)
    model = model.to(device)

    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"]))

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    first_step_grads: dict[str, float] | None = None
    final_step_grads: dict[str, float] | None = None
    step_seconds: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            t_step = time.perf_counter()
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if first_step_grads is None:
                first_step_grads = _grad_group_norms(model)
            if epoch == epochs:
                final_step_grads = _grad_group_norms(model)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
            lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            scheduler.step()
            step_seconds.append(time.perf_counter() - t_step)
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = bp.predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not np.isfinite(fit_pred).all():
        raise RuntimeError("non-finite fit prediction after soup")

    return {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "kind": kind,
        "seed": int(seed),
        "recipe_key": "R3",
        "recipe": dict(recipe),
        "init": "ssl",
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "warmup": "none",
            "lr0": float(recipe["lr"]),
            "lr_min": float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "build": build_info,
        "init_full_sha256": trunk_init,
        "init_group_state_sha256": group_init,
        "init_encoder_sha256": enc_init,
        "encoder_used_sha256": encoder_used,
        "local_init_sha256": local_init,
        "local_block_init_sha256": local_block_init,
        "parameter_audit": M.parameter_audit(model),
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "matches_frozen_p3e": order_hash.hexdigest() == C.P3E_BATCH_ORDER_SHA,
            "seed": int(seed) + bp.TRAIN_SHUFFLE_OFFSET,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "grad_group_norms_first_step": first_step_grads,
        "grad_group_norms_final_step": final_step_grads,
        "step_seconds": {
            "mean": float(np.mean(step_seconds)),
            "median": float(np.median(step_seconds)),
            "n": len(step_seconds),
        },
        "lr_curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "SSRC Fit rows raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "smoke": bool(smoke_epochs is not None),
        "monitor_read_at_train": False,
    }


def cmd_train(args: argparse.Namespace) -> int:
    arm = args.arm
    seed = int(args.seed)
    if arm not in ARM_KINDS:
        raise SystemExit(f"unknown arm {arm!r}")
    if args.smoke_epochs is None and seed != 0 and not args.authorize_seed1:
        raise SystemExit("seed 1 requires --authorize-seed1 after the frozen purchase rule fires")
    paths = _paths(arm, seed)
    if args.smoke_epochs is not None:
        paths["train"] = OUT / f"smoke_train_{arm}_s{seed}.json"
        paths["soup"] = OUT / f"smoke_train_{arm}_s{seed}_soup.pt"
    if paths["train"].exists() or paths["soup"].exists():
        raise SystemExit("arm already trained; retraining is not allowed")
    split = C.load_split()
    norm = C.read_json(C.BASELINE_DIR / "prep_fit.json")
    if not norm.get("fit_only"):
        raise SystemExit("standardiser is not marked fit-only")
    fit_objs, obj_src = _objects(split.fit, which="fit")
    labels = C.load_vault()
    y_fit = C.row_labels(split.fit, labels)
    ssl_state, ssl_meta = C.load_ssl_state()
    payload = train_arm(
        arm=arm,
        kind=ARM_KINDS[arm],
        seed=seed,
        train_objs=fit_objs,
        rows=split.fit,
        y_fit=y_fit,
        norm=norm,
        device=torch.device(args.device),
        ssl_state=ssl_state,
        smoke_epochs=args.smoke_epochs,
    )
    soup = payload.pop("soup_state")
    torch.save(soup, paths["soup"])
    record = dict(payload)
    record["soup_path"] = str(paths["soup"].relative_to(REPO))
    record["soup_sha256"] = C.sha256_file(paths["soup"])
    record["split_sha256"] = split.sha256
    record["fit_rows_sha256"] = C.rows_sha256(split.fit)
    record["monitor_rows_sha256"] = C.rows_sha256(split.monitor)
    record["fit_objects"] = obj_src
    record["ssl_state"] = ssl_meta
    record["local_params"] = M.local_parameters(ARM_KINDS[arm])
    C.write_json(paths["train"], record)
    log(
        f"[train {arm} s{seed}] params {record['parameter_audit']['actual_parameters']} "
        f"soup fit {record['soup_fit_mae']:.6f} wall {record['wall_seconds']:.0f}s "
        f"batch_order_ok={record['batch_order']['matches_frozen_p3e']}"
    )
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    arm = args.arm
    seed = int(args.seed)
    if arm not in ARM_KINDS:
        raise SystemExit(f"unknown arm {arm!r}")
    paths = _paths(arm, seed)
    if args.smoke_epochs is not None:
        raise SystemExit("smoke runs do not produce frozen predictions")
    if paths["freeze"].exists():
        raise SystemExit("predictions already frozen; refusing to rewrite")
    if not paths["train"].exists() or not paths["soup"].exists():
        raise SystemExit("arm is not trained")
    record = C.read_json(paths["train"])
    if record["smoke"]:
        raise SystemExit("smoke run cannot be used for a frozen prediction")
    if C.sha256_file(paths["soup"]) != record["soup_sha256"]:
        raise SystemExit("soup changed after training")

    split = C.load_split()
    norm = C.read_json(C.BASELINE_DIR / "prep_fit.json")
    _, model, _ = M.build_local_model(ARM_KINDS[arm], seed=seed, norm=norm, device=args.device)
    model.load_state_dict(torch.load(paths["soup"], map_location=args.device, weights_only=False), strict=True)
    mon_objs, mon_src = _objects(split.monitor, which="monitor")
    t0 = time.perf_counter()
    pred = bp.predict_model(model, mon_objs, args.device)
    infer_s = time.perf_counter() - t0
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite Monitor prediction")

    # label-free usage probes on the trained model (never touches labels)
    model.eval()
    probe_objs = mon_objs[:256]
    batch = bp._mv(sro.batch_objects(probe_objs), torch.device(args.device))
    with torch.no_grad():
        a = model.atom_encoder(batch["atom_feat"])
        a_local = model.local(a, batch) if model.local is not None else a
        shift = M.local_branch_shift(a, a_local)
        b_roll = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        b_roll["bond_index"] = torch.roll(b_roll["bond_index"], shifts=1, dims=0)
        a_roll = model.local(a, b_roll) if model.local is not None else a
        branch_adjacency_change = float((a_roll - a_local).abs().max())
        p_true = model(batch)
        b_perm = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        b_perm["bond_feat"] = torch.roll(b_perm["bond_feat"], shifts=1, dims=0)
        p_perm = model(b_perm)

    np.savez_compressed(
        paths["pred"],
        row=np.arange(len(mon_objs), dtype=np.int64),
        pred=pred.astype(np.float32),
    )
    freeze = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "predict_freeze",
        "arm": arm,
        "kind": ARM_KINDS[arm],
        "seed": seed,
        "soup_sha256": record["soup_sha256"],
        "prediction_sha256": C.sha256_file(paths["pred"]),
        "prediction_array_sha256": C.sha256_array(pred),
        "monitor_rows_sha256": C.rows_sha256(split.monitor),
        "monitor_objects": mon_src,
        "monitor_labels_read_before_freeze": False,
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
        "inference_seconds_1000": infer_s,
        "label_free_usage_probe": {
            "n_molecules": len(probe_objs),
            "local_shift": shift,
            "local_adjacency_roll_max_abs_change": branch_adjacency_change,
            "pred_bond_feature_roll_mean_abs_change": float((p_true - p_perm).abs().mean()),
            "pred_bond_feature_roll_max_abs_change": float((p_true - p_perm).abs().max()),
        },
    }
    C.write_json(paths["freeze"], freeze)
    log(
        f"[predict {arm} s{seed}] frozen {len(pred)} rows sha={freeze['prediction_sha256'][:16]} "
        f"shift={shift['relative_shift']:.4f} adj_change={branch_adjacency_change:.4f}"
    )
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _arm_record(arm: str, seed: int) -> dict[str, Any]:
    paths = _paths(arm, seed)
    if not paths["freeze"].exists() or not paths["train"].exists():
        raise SystemExit(f"arm {arm} seed {seed} is not frozen")
    freeze = C.read_json(paths["freeze"])
    train = C.read_json(paths["train"])
    with np.load(paths["pred"]) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    if C.sha256_file(paths["pred"]) != freeze["prediction_sha256"]:
        raise SystemExit(f"prediction file changed after the freeze: {arm}")
    if C.sha256_array(pred) != freeze["prediction_array_sha256"]:
        raise SystemExit(f"prediction array changed after the freeze: {arm}")
    return {"freeze": freeze, "train": train, "pred": pred}


def cmd_score(args: argparse.Namespace) -> int:
    out = OUT / "analysis_phase2a.json"
    decision_path = OUT / "decision_phase2a.json"
    if out.exists() or decision_path.exists():
        raise SystemExit("already scored; the Monitor label read is one-shot")
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = ["F", "M"]
    records = {(a, s): _arm_record(a, s) for a in arms for s in seeds}
    split = C.load_split()

    # --- the single Monitor label read --------------------------------------
    labels = C.load_vault()
    y = C.row_labels(split.monitor, labels)
    groups_info = C.label_groups(split.monitor)
    masks = groups_info["masks"]
    cycle_component = groups_info["cycle_component"]
    y_nocycle = y - cycle_component

    p3e = C.load_p3e_predictions()
    preds: dict[str, np.ndarray] = {"P3E": p3e.pred}
    for (a, s), rec in records.items():
        preds[f"{a}_s{s}"] = rec["pred"]

    metrics = {k: C.arm_metrics(v, y, y_nocycle=y_nocycle, cycle_component=cycle_component, groups=masks) for k, v in preds.items()}
    contrasts = {
        "M_vs_F": C.contrast(preds[f"M_s{seeds[0]}"], preds[f"F_s{seeds[0]}"], y, masks,
                             definition="mean |e_M| - |e_F|; negative means M is better"),
        "M_vs_P3E": C.contrast(preds[f"M_s{seeds[0]}"], preds["P3E"], y, masks,
                               definition="mean |e_M| - |e_P3E|; negative means M is better"),
        "F_vs_P3E": C.contrast(preds[f"F_s{seeds[0]}"], preds["P3E"], y, masks,
                               definition="mean |e_F| - |e_P3E|; negative means F is better"),
    }
    for a in arms:
        for s in seeds:
            if s == seeds[0]:
                continue
            contrasts[f"{a}_s{s}_vs_{a}_s{seeds[0]}"] = C.contrast(
                preds[f"{a}_s{s}"], preds[f"{a}_s{seeds[0]}"], y, masks,
                definition=f"mean |e_{a}_s{s}| - |e_{a}_s{seeds[0]}|; seed check",
            )

    flops = C.read_json(OUT / "flops_phase2a.json")["models"]
    flops_f = flops["F"]["flops_ratio_vs_p3e"]
    flops_m = flops["M"]["flops_ratio_vs_p3e"]

    optimization_ok = True
    opt_details: dict[str, Any] = {}
    for (a, s), rec in records.items():
        tr = rec["train"]
        last5 = [e["train_mae"] for e in tr["lr_curve"][-5:]]
        finite = all(math.isfinite(e["train_mae"]) for e in tr["lr_curve"])
        new_grads = {
            k: v
            for k, v in (tr["grad_group_norms_final_step"] or {}).items()
            if k.startswith("local.")
        }
        ok = (
            finite
            and max(last5) < 0.01
            and tr["soup_fit_mae"] < 0.05
            and all(v > 0 for v in new_grads.values())
            and math.isfinite(tr["anomalies"]["grad_norm_max"])
        )
        optimization_ok &= bool(ok)
        opt_details[f"{a}_s{s}"] = {
            "ok": bool(ok),
            "last5_train_mae": last5,
            "soup_fit_mae": tr["soup_fit_mae"],
            "grad_norm_max": tr["anomalies"]["grad_norm_max"],
            "grad_clip_events": tr["anomalies"]["grad_clip_events"],
            "final_new_param_grad_norms": new_grads,
            "description": "finite losses, last-5 train MAE < 0.01, soup fit MAE < 0.05, new-param gradients nonzero",
        }

    complement_gain = (
        metrics["P3E"]["groups"]["label_complement"]["mae"]
        - metrics[f"M_s{seeds[0]}"]["groups"]["label_complement"]["mae"]
    )
    decision = C.decide(
        mae_f=metrics[f"F_s{seeds[0]}"]["mae"],
        mae_m=metrics[f"M_s{seeds[0]}"]["mae"],
        rmse_m=metrics[f"M_s{seeds[0]}"]["rmse"],
        complement_gain=complement_gain,
        flops_ratio_f=flops_f,
        flops_ratio_m=flops_m,
        optimization_ok=optimization_ok,
    )
    decision["optimization_details"] = opt_details

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "score",
        "split_sha256": split.sha256,
        "monitor_rows_sha256": C.rows_sha256(split.monitor),
        "monitor_labels_read_count": 1,
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
        "early_stopping_on_monitor": False,
        "model_selection": "none (single fixed recipe, fixed 360 epochs, seed 0 screening)",
        "groups": {
            "definitions": {
                "label_complement": "official train target carries no GVAE cycle penalty",
                "label_penalised": "non-zero GVAE cycle penalty in the stored target",
                "rdkit_large": "RDKit SSSR max ring > 6",
                "penalty_only_rdkit_missed": "label penalised but RDKit SSSP max ring <= 6",
                "permutation_sensitive": "label cycle level changes under atom permutation (diagnostic only)",
            },
            "counts": groups_info["counts"],
            "sigma_cycle": groups_info["sigma_cycle"],
            "mu_cycle": groups_info["mu_cycle"],
            "caveat": groups_info["caveat"],
        },
        "metrics": metrics,
        "contrasts": contrasts,
        "efficiency": {
            "params": {k: v["params"] for k, v in flops.items()},
            "forward_gemm_flops": {k: v["forward_gemm_flops"] for k, v in flops.items()},
            "flops_ratio_vs_p3e": {"F": flops_f, "M": flops_m},
            "wall_seconds": {f"{a}_s{s}": records[(a, s)]["train"]["wall_seconds"] for a in arms for s in seeds},
            "step_seconds": {f"{a}_s{s}": records[(a, s)]["train"]["step_seconds"] for a in arms for s in seeds},
            "inference_seconds_1000": {
                f"{a}_s{s}": records[(a, s)]["freeze"]["inference_seconds_1000"] for a in arms for s in seeds
            },
            "label_free_usage_probe": {
                f"{a}_s{s}": records[(a, s)]["freeze"]["label_free_usage_probe"] for a in arms for s in seeds
            },
        },
        "p3e_reference": {
            "monitor_mae": C.P3E_MAE,
            "monitor_rmse": C.P3E_RMSE,
            "prediction_sha256": p3e.sha256,
            "reused_frozen_soup": True,
        },
        "seeds": seeds,
    }
    C.write_json(out, payload)
    C.write_json(decision_path, decision)
    log(
        f"[score] MAE F={metrics[f'F_s{seeds[0]}']['mae']:.6f} M={metrics[f'M_s{seeds[0]}']['mae']:.6f} "
        f"P3E={metrics['P3E']['mae']:.6f} -> {decision['status']}"
    )
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Replay the frozen P3E artefacts, run the CPU gates, count FLOPs."""
    for fn in (cmd_replay, cmd_gates, cmd_flops):
        rc = fn(args)
        if rc:
            return rc
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Two-epoch timing runs for both arms; never produces frozen predictions."""
    epochs = int(args.smoke_epochs or 2)
    for arm in ("F", "M"):
        ns = argparse.Namespace(**vars(args))
        ns.arm = arm
        ns.smoke_epochs = epochs
        rc = cmd_train(ns)
        if rc:
            return rc
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        choices=["prep", "replay", "gates", "flops", "smoke", "train", "predict", "score"],
    )
    ap.add_argument("--arm", default="F", choices=sorted(ARM_KINDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0", help="comma-separated seeds for score")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-epochs", type=int, default=None)
    ap.add_argument("--authorize-seed1", action="store_true")
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    return {
        "prep": cmd_prep,
        "smoke": cmd_smoke,
        "replay": cmd_replay,
        "gates": cmd_gates,
        "flops": cmd_flops,
        "train": cmd_train,
        "predict": cmd_predict,
        "score": cmd_score,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
