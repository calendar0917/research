"""Runner for zinc_native_relation_v1.

Commands
--------
prep     build the compact train/valid relation cache + null mapping
checks   CPU implementation gates (native replay, bypass, invariances, null)
timing   full-step / full-valid / save timing on GPU and budget projection
train    one formal arm/seed through the unchanged recipe-B trainer
score    per-run valid metrics, fixed groups, per-molecule table
analyze  paired contrasts, persistence metrics, frozen decision
probe    frozen-checkpoint relation bypass / real-null forward diagnostic
test     terminal official-test read (only after the frozen trigger fires)

Official test is never loaded by prep, checks, timing, train, score or analyze.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
if str(_V1) not in sys.path:
    sys.path.insert(0, str(_V1))

import ad_common as AC
import ad_data as AD
import ad_train as AT

import nr_common as C
import nr_data as D
import nr_models as M


# ---------------------------------------------------------------------------
# paths / small helpers
# ---------------------------------------------------------------------------


def rel_cache_path(split: str) -> Path:
    return C.CACHE_DIR / f"nr_{split}.pt"


def load_rel(split: str) -> list[dict[str, Any]]:
    path = rel_cache_path(split)
    if not path.exists():
        raise SystemExit(f"missing relation cache {path}; run prep first")
    return D.load_cache(path)


def load_ys() -> tuple[np.ndarray, np.ndarray]:
    train = AD.load_cache("train")
    valid = AD.load_cache("valid")
    return (
        np.asarray(train["labels"], dtype=np.float64),
        np.asarray(valid["labels"], dtype=np.float64),
    )


def p3_collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
    """Map this round's arm names onto the unchanged P3 collate branch."""
    return AT.collate(cache, rows, "P3N", valid=valid)


def eval_pred(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: np.ndarray,
    device,
    batch_size: int,
    collate_fn,
    *,
    relation_mode: str | None = None,
    bypass: bool = False,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), AC.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            batch = AT._to_device(collate_fn(cache, chunk, "P3N", valid=True), device)
            if isinstance(model, M.NativeRelationModel):
                pred = model(batch, relation_mode=relation_mode, bypass=bypass)
            else:
                pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    model.train()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _state_sha(path: Path) -> str:
    return C.sha256_state_dict(M.state_from_checkpoint(path))


def order_sequence_hash(seed: int, *, epochs: int = 600, n_rows: int = 10000) -> dict[str, Any]:
    """Replay the trainer's shuffle stream and hash it (data order provenance)."""
    gen = torch.Generator().manual_seed(int(seed) + 101)
    h = __import__("hashlib").sha256()
    per_epoch = []
    for _ in range(int(epochs)):
        perm = torch.randperm(int(n_rows), generator=gen).numpy().astype(np.int64)
        h.update(perm.tobytes())
        per_epoch.append(__import__("hashlib").sha256(perm.tobytes()).hexdigest())
    return {
        "generator_seed": int(seed) + 101,
        "epochs": int(epochs),
        "rows": int(n_rows),
        "stream_sha256": h.hexdigest(),
        "first_epoch_sha256": per_epoch[0] if per_epoch else "",
        "last_epoch_sha256": per_epoch[-1] if per_epoch else "",
    }


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    t0 = time.time()
    cfg = C.load_config()
    ssl_path = C.REPO / cfg["ssl"]["path"]
    if C.sha256_file(ssl_path) != cfg["ssl"]["sha256"]:
        raise SystemExit("SSL checkpoint hash mismatch")
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_entries = D.build_split("train")
    train_sha = D.save_cache(train_entries, rel_cache_path("train"))
    valid_entries = D.build_split("valid")
    valid_sha = D.save_cache(valid_entries, rel_cache_path("valid"))
    train_null = D.aggregate_null_stats(train_entries)
    valid_null = D.aggregate_null_stats(valid_entries)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "ssl": {"path": cfg["ssl"]["path"], "sha256": cfg["ssl"]["sha256"]},
        "data": {
            "source": "frozen P3 ad_data cache + official PyG ZINC subset labels for train/valid only",
            "p3_cache": {
                "train": {"path": str(AD.cache_path("train").relative_to(C.REPO)),
                          "sha256": C.sha256_file(AD.cache_path("train"))},
                "valid": {"path": str(AD.cache_path("valid").relative_to(C.REPO)),
                          "sha256": C.sha256_file(AD.cache_path("valid"))},
            },
            "relation_token_layout": "junctions first, then original bonds; one bond token per P3 bond",
            "pair_rule": "clipped anchor distance <= 3 OR shared biconnected ring block; both directions + self loops",
            "relation_formula": "1 + 8*min(d,3) + 4*same_ring_block + min(A^2,3); self = 0",
            "null_rule": "within molecule and JJ/JB/BB role stratum, fixed permutation of non-self relation labels; self stays 0",
            "null_seed": int(D.NULL_SEED),
            "null_uses_labels_or_predictions": False,
            "test_split_loaded": False,
            "labels_read_during_prep": False,
        },
        "splits": {
            "train": {
                "n": len(train_entries),
                "cache": str(rel_cache_path("train").relative_to(C.REPO)),
                "sha256": train_sha,
                "n_atoms": int(sum(e["n_atoms"] for e in train_entries)),
                "n_bonds": int(sum(e["n_bonds"] for e in train_entries)),
                "n_junctions": int(sum(e["n_junc"] for e in train_entries)),
                "n_tokens": int(sum(e["n_tokens"] for e in train_entries)),
                "n_non_self_pairs": int(sum(len(e["r_u"]) for e in train_entries)),
                "null_stats": train_null,
            },
            "valid": {
                "n": len(valid_entries),
                "cache": str(rel_cache_path("valid").relative_to(C.REPO)),
                "sha256": valid_sha,
                "n_atoms": int(sum(e["n_atoms"] for e in valid_entries)),
                "n_bonds": int(sum(e["n_bonds"] for e in valid_entries)),
                "n_junctions": int(sum(e["n_junc"] for e in valid_entries)),
                "n_tokens": int(sum(e["n_tokens"] for e in valid_entries)),
                "n_non_self_pairs": int(sum(len(e["r_u"]) for e in valid_entries)),
                "null_stats": valid_null,
            },
        },
        "relation_fingerprints": {
            "train": D.relation_fingerprint(train_entries),
            "valid": D.relation_fingerprint(valid_entries),
        },
        "prep_seconds": time.time() - t0,
    }
    replay = C.ANALYSIS_DIR / "hr_relation_replay.json"
    if replay.exists():
        payload["hr_relation_replay"] = {
            "path": str(replay.relative_to(C.REPO)),
            "sha256": C.sha256_file(replay),
        }
    out = Path(args.manifest)
    if not out.is_absolute():
        out = C.REPO / out
    C.write_json(out, payload)
    print(json.dumps({k: payload[k] for k in ("prep_seconds", "splits", "relation_fingerprints")}, indent=2)[:4000])
    return 0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _cpu_batch(cache, entries, rows, device="cpu"):
    return AT._to_device(D.make_collate(entries)(cache, rows, "P3N", valid=True), device)


def _model_pred(model, batch, *, mode=None, bypass=False):
    if isinstance(model, M.NativeRelationModel):
        return model(batch, relation_mode=mode, bypass=bypass)
    return model(batch)


def _maxdiff(a, b) -> float:
    return float(torch.max(torch.abs(a - b)).item()) if a.numel() else 0.0


def _permuted_atom_batch(batch: Mapping[str, Any], perm: np.ndarray):
    """Atom-renumber a collated single-molecule batch; token order stays fixed."""
    import copy

    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out["atom_feat"] = batch["atom_feat"][torch.as_tensor(perm, dtype=torch.long)]
    bi = batch["bond_index"].clone()
    bi[:, 0] = torch.as_tensor(inv[bi[:, 0].numpy()], dtype=torch.long)
    bi[:, 1] = torch.as_tensor(inv[bi[:, 1].numpy()], dtype=torch.long)
    out["bond_index"] = bi
    ji = batch["junc_index"].clone()
    ji[:, 0] = torch.as_tensor(inv[ji[:, 0].numpy()], dtype=torch.long)
    ji[:, 1] = torch.as_tensor(inv[ji[:, 1].numpy()], dtype=torch.long)
    ji[:, 2] = torch.as_tensor(inv[ji[:, 2].numpy()], dtype=torch.long)
    out["junc_index"] = ji
    return out


def _synthetic_cache() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Three tiny graphs: one atom; two isolated atoms; a 2-atom bond + isolated atom."""
    atom_feat = np.zeros((5, AD.ATOM_FEAT_DIM), dtype=np.float32)
    atom_feat[0, 0] = 1.0
    atom_feat[1, 1] = 1.0
    atom_feat[2, 2] = 1.0
    atom_feat[3, 3] = 1.0
    atom_feat[4, 4] = 1.0
    bond_feat = np.zeros((1, AD.BOND_FEAT_DIM), dtype=np.float32)
    junc_feat = np.zeros((0, AD.JUNC_FEAT_DIM), dtype=np.float32)
    cache = {
        "n_molecules": 3,
        "atom_ptr": np.asarray([0, 1, 3, 5], dtype=np.int32),
        "bond_ptr": np.asarray([0, 0, 0, 1], dtype=np.int32),
        "junc_ptr": np.asarray([0, 0, 0, 0], dtype=np.int32),
        "atom_feat": atom_feat,
        "atom_type": np.zeros(5, dtype=np.int64),
        "bond_feat": bond_feat,
        "bond_type": np.ones(1, dtype=np.int64),
        "bond_index": np.asarray([[0, 1]], dtype=np.int16),
        "junc_feat": junc_feat,
        "junc_index": np.zeros((0, 3), dtype=np.int16),
        "junc_bond": np.zeros((0, 2), dtype=np.int16),
        "global_feat": np.zeros((3, AD.GLOBAL_DIM), dtype=np.float32),
        "dist_ptr": np.asarray([0, 1, 5, 9], dtype=np.int32),
        "dist_flat": np.asarray([0, 0, 99, 99, 0, 0, 1, 1, 0], dtype=np.int64),
    }
    entries = [
        {"n_atoms": 1, "n_bonds": 0, "n_junc": 0, "n_tokens": 0,
         "r_u": np.zeros(0, np.int32), "r_v": np.zeros(0, np.int32),
         "r_rel": np.zeros(0, np.int8), "r_rel_null": np.zeros(0, np.int8)},
        {"n_atoms": 2, "n_bonds": 0, "n_junc": 0, "n_tokens": 0,
         "r_u": np.zeros(0, np.int32), "r_v": np.zeros(0, np.int32),
         "r_rel": np.zeros(0, np.int8), "r_rel_null": np.zeros(0, np.int8)},
        {"n_atoms": 2, "n_bonds": 1, "n_junc": 0, "n_tokens": 1,
         "r_u": np.zeros(0, np.int32), "r_v": np.zeros(0, np.int32),
         "r_rel": np.zeros(0, np.int8), "r_rel_null": np.zeros(0, np.int8)},
    ]
    return cache, entries


def cmd_checks(args: argparse.Namespace) -> int:
    t0 = time.time()
    cfg = C.load_config()
    norm = AD.load_norm()
    cache_train = AD.load_cache("train")
    cache_valid = AD.load_cache("valid")
    entries_train = load_rel("train")
    entries_valid = load_rel("valid")
    rows = np.arange(64, dtype=np.int64)
    batch = _cpu_batch(cache_valid, entries_valid, rows)
    results: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "git_commit": C.git_commit()}

    p_model, p_info = M.build_arm("P", norm, 0, "cpu")
    r_model, r_info = M.build_arm("R", norm, 0, "cpu")
    n_model, n_info = M.build_arm("N", norm, 0, "cpu")
    results["parameters"] = {
        "P": p_info["parameters"], "R": r_info["parameters"], "N": n_info["parameters"],
        "R_new": r_info["parameters"] - p_info["parameters"],
        "N_new": n_info["parameters"] - p_info["parameters"],
        "cap": 500000,
        "breakdown_R": r_info["parameter_breakdown"],
    }

    # 1. decoder weights: our P builder is the mature P3N builder at seed 0
    mature, mature_info = M.build_p3_mature(norm, 0, "cpu")
    results["mature_p3_state_match"] = {
        "max_abs_diff": _maxdiff(
            torch.cat([v.flatten() for v in mature.state_dict().values()]),
            torch.cat([v.flatten() for v in p_model.state_dict().values()]),
        ),
        "mature_parameters": mature_info["parameters"],
        "our_parameters": p_info["parameters"],
    }

    # 2. common state + relation state
    results["init_state"] = {
        "P_common": p_info["common_state_sha256"],
        "R_common": r_info["common_state_sha256"],
        "N_common": n_info["common_state_sha256"],
        "R_relation": r_info["relation_state_sha256"],
        "N_relation": n_info["relation_state_sha256"],
        "common_equal": bool(
            p_info["common_state_sha256"] == r_info["common_state_sha256"] == n_info["common_state_sha256"]
        ),
        "relation_equal": bool(r_info["relation_state_sha256"] == n_info["relation_state_sha256"]),
        "ssl_equal": bool(r_info["ssl"]["sha256"] == n_info["ssl"]["sha256"] == p_info["ssl"]["sha256"]),
    }

    # 3. native encoder replay, layer by layer
    with torch.no_grad():
        a_p, b_p, w_p = p_model.encoders(batch)
        a_r, b_r, w_r = r_model.encoders(batch)
        a_n, b_n, w_n = n_model.encoders(batch)
    results["native_encoder_replay"] = {
        "a_max_abs_diff_R": _maxdiff(a_r, a_p),
        "b_max_abs_diff_R": _maxdiff(b_r, b_p),
        "w_max_abs_diff_R": _maxdiff(w_r, w_p),
        "a_max_abs_diff_N": _maxdiff(a_n, a_p),
        "b_max_abs_diff_N": _maxdiff(b_n, b_p),
        "w_max_abs_diff_N": _maxdiff(w_n, w_p),
        "a_sha256": C.sha256_array(a_p.numpy()),
        "b_sha256": C.sha256_array(b_p.numpy()),
        "w_sha256": C.sha256_array(w_p.numpy()),
    }

    # 4. full bypass replay: R/N with h1 = h0 must equal P exactly
    with torch.no_grad():
        pred_p = p_model(batch)
        pred_r_bypass = r_model(batch, bypass=True)
        pred_n_bypass = n_model(batch, bypass=True)
        pred_r = r_model(batch, relation_mode="real")
        pred_n = n_model(batch, relation_mode="null")
    results["bypass_replay"] = {
        "R_vs_P_max_abs_diff": _maxdiff(pred_r_bypass, pred_p),
        "N_vs_P_max_abs_diff": _maxdiff(pred_n_bypass, pred_p),
        "P_pred_sha256": C.sha256_array(pred_p.numpy()),
    }
    results["relation_changes_prediction"] = {
        "R_real_vs_bypass_max_abs_diff": _maxdiff(pred_r, pred_p),
        "N_null_vs_bypass_max_abs_diff": _maxdiff(pred_n, pred_p),
        "R_real_vs_N_null_max_abs_diff": _maxdiff(pred_r, pred_n),
    }

    # 5. atom renumbering, bond direction, batch-vs-single, synthetic empties
    one = 17
    batch_one = _cpu_batch(cache_valid, entries_valid, np.asarray([one], dtype=np.int64))
    n_atoms = int(cache_valid["atom_ptr"][one + 1] - cache_valid["atom_ptr"][one])
    rng = np.random.default_rng(4242)
    perm = rng.permutation(n_atoms).astype(np.int64)
    batch_perm = _permuted_atom_batch(batch_one, perm)
    inv = [0.0, 0.0, 0.0]
    for model, name in ((p_model, "P"), (r_model, "R"), (n_model, "N")):
        with torch.no_grad():
            base = _model_pred(model, batch_one)
            ren = _model_pred(model, batch_perm)
            inv[{"P": 0, "R": 1, "N": 2}[name]] = _maxdiff(ren, base)
    results["renumbering"] = {
        "molecule": int(one), "n_atoms": n_atoms,
        "P_max_abs_diff": inv[0], "R_max_abs_diff": inv[1], "N_max_abs_diff": inv[2],
    }

    batch_dir = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_one.items()}
    bi = batch_dir["bond_index"].clone()
    bi = bi[:, [1, 0]]
    batch_dir["bond_index"] = bi
    dir_diff = {}
    for model, name in ((p_model, "P"), (r_model, "R"), (n_model, "N")):
        with torch.no_grad():
            dir_diff[name] = _maxdiff(_model_pred(model, batch_dir), _model_pred(model, batch_one))
    results["bond_direction"] = dir_diff

    small_rows = np.arange(8, dtype=np.int64)
    batch_small = _cpu_batch(cache_valid, entries_valid, small_rows)
    with torch.no_grad():
        pred_all = _model_pred(r_model, batch_small).numpy()
        singles = []
        for i in small_rows:
            b = _cpu_batch(cache_valid, entries_valid, np.asarray([i], dtype=np.int64))
            singles.append(float(_model_pred(r_model, b).numpy()[0]))
    results["batch_vs_single_R_max_abs_diff"] = float(np.max(np.abs(pred_all - np.asarray(singles))))

    syn_cache, syn_entries = _synthetic_cache()
    syn_batch = AT._to_device(D.make_collate(syn_entries)(syn_cache, np.arange(3), "P3N", valid=True), "cpu")
    with torch.no_grad():
        syn = {}
        for model, name in ((p_model, "P"), (r_model, "R"), (n_model, "N")):
            pred = _model_pred(model, syn_batch)
            syn[name] = {
                "pred": pred.tolist(),
                "finite": bool(torch.isfinite(pred).all().item()),
                "n_tokens": int(syn_batch["n_tokens_total"]),
            }
    results["synthetic_edges"] = syn

    # 6. relation isolation / self loops / null transport
    b = _cpu_batch(cache_valid, entries_valid, np.arange(32, dtype=np.int64))
    src = b["r_src"].numpy(); dst = b["r_dst"].numpy()
    n_tokens = int(b["n_tokens_total"])
    # token layout is all junctions first, then all bonds
    nj = np.diff(np.asarray(cache_valid["junc_ptr"]))[np.arange(32)]
    nb = np.diff(np.asarray(cache_valid["bond_ptr"]))[np.arange(32)]
    token_mol = np.concatenate([
        np.repeat(np.arange(32), nj),
        np.repeat(np.arange(32), nb),
    ])
    same_mol = int((token_mol[src] == token_mol[dst]).all())
    self_mask = src == dst
    self_counts = np.bincount(src[self_mask], minlength=n_tokens)
    self_rel_ok = bool((b["r_rel"][self_mask] == 0).all() and (b["r_rel_null"][self_mask] == 0).all())
    # segment softmax sum for R
    with torch.no_grad():
        _, b_enc, w_enc = r_model.encoders(b)
        h0 = torch.cat([w_enc, b_enc], dim=0)
        u = r_model.relation.norm(h0 + r_model.relation.role(torch.cat([
            torch.zeros(int(nj.sum()), dtype=torch.long),
            torch.ones(int(nb.sum()), dtype=torch.long),
        ])))
        q = r_model.relation.q(u); k = r_model.relation.k(u)
        score = (q[b["r_src"]] * (k[b["r_dst"]] + r_model.relation.rel_emb(b["r_rel"]))).sum(-1) / math.sqrt(r_model.relation.rank)
        score = score + r_model.relation.rel_bias[b["r_rel"]]
        seg_max = torch.full((n_tokens,), -1.0e9)
        seg_max.scatter_reduce_(0, b["r_src"], score.detach(), reduce="amax", include_self=True)
        alpha = torch.exp(score - seg_max[b["r_src"]])
        denom = torch.zeros(n_tokens)
        denom.index_add_(0, b["r_src"], alpha)
        alpha = alpha / denom[b["r_src"]].clamp_min(1e-12)
        sums = torch.zeros(n_tokens).index_add_(0, b["r_src"], alpha)
    pair_set_equal = bool(
        b["r_src"].numel() == b["r_dst"].numel() == b["r_rel"].numel() == b["r_rel_null"].numel()
        and int((b["r_rel"] != b["r_rel_null"]).sum().item()) > 0
    )
    results["relation_structure"] = {
        "n_tokens": n_tokens,
        "all_edges_within_one_molecule": same_mol == 1,
        "self_loop_once_per_token": bool((self_counts == 1).all()),
        "self_loop_rel_zero": self_rel_ok,
        "softmax_sum_max_abs_dev": float(torch.max(torch.abs(sums - 1.0)).item()),
        "real_null_pair_set_same": pair_set_equal,
        "changed_labels": int((b["r_rel"] != b["r_rel_null"]).sum().item()),
        "total_directed_non_self": int(((src != dst).sum())),
    }

    # 7. null transport under atom renumbering (labels and N prediction unchanged)
    # rebuild relations from the renumbered graph for this molecule
    bond_index = cache_valid["bond_index"][
        int(cache_valid["bond_ptr"][one]) : int(cache_valid["bond_ptr"][one + 1])
    ].astype(np.int64)
    junc_index = cache_valid["junc_index"][
        int(cache_valid["junc_ptr"][one]) : int(cache_valid["junc_ptr"][one + 1])
    ].astype(np.int64)
    ptr = int(cache_valid["dist_ptr"][one]), int(cache_valid["dist_ptr"][one + 1])
    dist = np.asarray(cache_valid["dist_flat"][ptr[0] : ptr[1]]).reshape(n_atoms, n_atoms)
    bi_perm = np.empty_like(bond_index)
    bi_perm[:, 0] = np.asarray([np.where(perm == u)[0][0] for u in bond_index[:, 0]])
    bi_perm[:, 1] = np.asarray([np.where(perm == v)[0][0] for v in bond_index[:, 1]])
    ji_perm = np.asarray([[int(np.where(perm == x)[0][0]) for x in row] for row in junc_index])
    u0, v0, r0 = D.relation_labels(dist, n_atoms, bond_index, junc_index)
    dist_perm = dist[np.ix_(perm, perm)]
    u1, v1, r1 = D.relation_labels(dist_perm, n_atoms, bi_perm, ji_perm)
    n0, _ = D.null_relation_labels(u0, v0, r0, len(junc_index), one)
    n1, _ = D.null_relation_labels(u1, v1, r1, len(junc_index), one)
    n_pred_diff = _maxdiff(_model_pred(n_model, batch_perm), _model_pred(n_model, batch_one))
    results["null_transport_renumbering"] = {
        "pair_u_equal": bool(np.array_equal(u0, u1)),
        "pair_v_equal": bool(np.array_equal(v0, v1)),
        "real_equal": bool(np.array_equal(r0, r1)),
        "null_equal": bool(np.array_equal(n0, n1)),
        "n_pred_max_abs_diff": n_pred_diff,
        "n_pred_equal": bool(n_pred_diff <= 1e-5),
    }

    # 8. gradient coverage / relation branch active
    train_batch = _cpu_batch(cache_train, entries_train, np.arange(8, dtype=np.int64))
    y = torch.as_tensor(np.asarray(cache_train["labels"])[:8], dtype=torch.float32)
    grad_stats: dict[str, Any] = {}
    for model, name in ((r_model, "R"), (n_model, "N")):
        model.train()
        model.zero_grad(set_to_none=True)
        pred = model(train_batch)
        loss = (pred - y).abs().mean()
        loss.backward()
        groups = {
            "relation": list(model.relation.parameters()),
            "common": [p for k, p in model.named_parameters() if not k.startswith("relation.")],
        }
        grad_stats[name] = {}
        for gname, params in groups.items():
            grads = [p.grad for p in params if p.grad is not None]
            grad_stats[name][gname] = {
                "n_tensors": len(params),
                "n_with_grad": len(grads),
                "all_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                "grad_norm": float(math.sqrt(sum(float((g * g).sum()) for g in grads))),
                "nonzero_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
            }
        grad_stats[name]["loss"] = float(loss.detach())
        model.zero_grad(set_to_none=True)
    results["gradients"] = grad_stats
    # relation and null must produce different relation gradients
    r_model.zero_grad(set_to_none=True)
    (r_model(train_batch) - y).abs().mean().backward()
    g_r = torch.cat([p.grad.flatten() for p in r_model.relation.parameters() if p.grad is not None])
    r_model.zero_grad(set_to_none=True)
    (r_model(train_batch, relation_mode="null") - y).abs().mean().backward()
    g_n = torch.cat([p.grad.flatten() for p in r_model.relation.parameters() if p.grad is not None])
    results["relation_gradient_real_vs_null_max_abs_diff"] = float((g_r - g_n).abs().max().item())

    # 9. optional full replay against the old HR relation arrays
    hr_dir = C.REPO / "tracks/structural-sufficiency/zinc_hierarchical_relation_v1/cache"
    if (hr_dir / "hr_train.pt").exists():
        replay_train = D.replay_against_hr(hr_dir / "hr_train.pt", "train", limit=args.replay_limit)
        replay_valid = D.replay_against_hr(hr_dir / "hr_val.pt", "valid", limit=args.replay_limit)
        results["hr_relation_replay"] = {"train": replay_train, "valid": replay_valid}
        C.write_json(C.ANALYSIS_DIR / "hr_relation_replay.json", {
            "source": str(hr_dir.relative_to(C.REPO)),
            "train": replay_train,
            "valid": replay_valid,
            "note": "relation labels/pairs are rebuilt from the frozen P3 objects; the old HR cache is an audit target only",
        })

    results["pass"] = {
        "P_parameters_413505": p_info["parameters"] == 413505,
        "R_N_under_cap": r_info["parameters"] <= 500000 and n_info["parameters"] <= 500000,
        "mature_p3_state_match": results["mature_p3_state_match"]["max_abs_diff"] <= 1e-6,
        "common_init_equal": results["init_state"]["common_equal"],
        "relation_init_equal": results["init_state"]["relation_equal"],
        "ssl_equal": results["init_state"]["ssl_equal"],
        "native_replay_R": max(results["native_encoder_replay"][f"{k}_max_abs_diff_R"] for k in "abw") <= 1e-6,
        "native_replay_N": max(results["native_encoder_replay"][f"{k}_max_abs_diff_N"] for k in "abw") <= 1e-6,
        "bypass_replay": max(results["bypass_replay"]["R_vs_P_max_abs_diff"],
                              results["bypass_replay"]["N_vs_P_max_abs_diff"]) <= 1e-6,
        "renumbering": max(results["renumbering"]["R_max_abs_diff"],
                           results["renumbering"]["N_max_abs_diff"],
                           results["renumbering"]["P_max_abs_diff"]) <= 1e-5,
        "bond_direction": max(results["bond_direction"].values()) <= 1e-6,
        "batch_vs_single": results["batch_vs_single_R_max_abs_diff"] <= 1e-5,
        "synthetic_finite": all(v["finite"] for v in syn.values()),
        "relation_within_molecule": results["relation_structure"]["all_edges_within_one_molecule"],
        "self_loops": results["relation_structure"]["self_loop_once_per_token"]
        and results["relation_structure"]["self_loop_rel_zero"],
        "softmax_segments": results["relation_structure"]["softmax_sum_max_abs_dev"] <= 1e-5,
        "null_changes_labels": results["relation_structure"]["changed_labels"] > 0,
        "null_transport": all(v for k, v in results["null_transport_renumbering"].items()),
        "relation_gradients": all(
            grad_stats[a]["relation"]["all_finite"] and grad_stats[a]["relation"]["nonzero_tensors"] > 0
            for a in ("R", "N")
        ),
        "common_gradients": all(grad_stats[a]["common"]["all_finite"] for a in ("R", "N")),
        "real_null_gradients_differ": results["relation_gradient_real_vs_null_max_abs_diff"] > 0,
    }
    if "hr_relation_replay" in results:
        results["pass"]["hr_replay"] = bool(
            results["hr_relation_replay"]["train"]["all_match"]
            and results["hr_relation_replay"]["valid"]["all_match"]
        )
    results["all_pass"] = bool(all(results["pass"].values()))
    results["seconds"] = time.time() - t0
    C.write_json(args.out, results)
    print(json.dumps({"all_pass": results["all_pass"], "failed": [k for k, v in results["pass"].items() if not v],
                      "parameters": results["parameters"], "seconds": results["seconds"]}, indent=2))
    return 0 if results["all_pass"] else 1


# ---------------------------------------------------------------------------
# timing / budget freeze
# ---------------------------------------------------------------------------


def cmd_timing(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    norm = AD.load_norm()
    cache_train = AD.load_cache("train")
    cache_valid = AD.load_cache("valid")
    entries_train = load_rel("train")
    entries_valid = load_rel("valid")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    y_train, y_valid = load_ys()
    batch_size = int(cfg["recipe"]["batch_size"])
    valid_batch = int(cfg["recipe"]["valid_batch_size"])
    steps = int(args.steps)
    out: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID, "device": str(device),
        "steps_per_epoch": 79, "batch_size": batch_size, "valid_batch_size": valid_batch,
        "steps_measured": steps, "budget_gpu_hours": float(cfg["budget"]["gpu_hours_total"]),
    }
    for arm in C.ARMS:
        model, info = M.build_arm(arm, norm, 0, device)
        collate = p3_collate if arm == "P" else D.make_collate(entries_train)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        gen = torch.Generator().manual_seed(101)
        model.train()
        # warm-up (not measured)
        for _ in range(3):
            rows = torch.randperm(len(entries_train), generator=gen)[:batch_size].numpy()
            batch = AT._to_device(collate(cache_train, rows, "P3N", valid=False), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            loss = (model(batch) - target).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            rows = torch.randperm(len(entries_train), generator=gen)[:batch_size].numpy()
            batch = AT._to_device(collate(cache_train, rows, "P3N", valid=False), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            loss = (model(batch) - target).abs().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_sec = (time.perf_counter() - t0) / steps
        # one real epoch is 79 steps
        epoch_steps = 79 * step_sec
        collate_v = p3_collate if arm == "P" else D.make_collate(entries_valid)
        t1 = time.perf_counter()
        pred = eval_pred(model, cache_valid, np.arange(C.N_VALID), device, valid_batch, collate_v)
        valid_sec = time.perf_counter() - t1
        t2 = time.perf_counter()
        probe_path = Path(args.save_probe)
        if not probe_path.is_absolute():
            probe_path = C.TRACK / probe_path
        torch.save({"state_dict": AC.cpu_state(model)}, probe_path)
        save_sec = time.perf_counter() - t2
        collate_t = p3_collate if arm == "P" else D.make_collate(entries_train)
        t3 = time.perf_counter()
        pred_train = eval_pred(model, cache_train, np.arange(C.N_TRAIN), device, 256, collate_t)
        train_eval_sec = time.perf_counter() - t3
        train_eval_mae = C.mae(pred_train, y_train)
        valid_mae = C.mae(pred, y_valid)
        # end-of-run: 2 full-train evals + 1 full-valid re-eval + 2 checkpoint saves
        projected = (
            600.0 * epoch_steps
            + 600.0 * valid_sec
            + 2.0 * save_sec
            + 2.0 * train_eval_sec
            + 1.0 * valid_sec
        )
        out[arm] = {
            "parameters": info["parameters"],
            "seconds_per_step": step_sec,
            "epoch_steps_79_seconds": epoch_steps,
            "full_valid_seconds": valid_sec,
            "save_seconds": save_sec,
            "full_train_eval_seconds": train_eval_sec,
            "valid_mae_after_warmup": valid_mae,
            "train_eval_mae_after_warmup": train_eval_mae,
            "projected_run_seconds": projected,
            "projected_run_gpu_hours": projected / 3600.0,
            "projected_run_with_15pct_margin_gpu_hours": projected * 1.15 / 3600.0,
            "peak_cuda_memory_gb": (torch.cuda.max_memory_allocated(device) / 1e9)
            if device.type == "cuda" else 0.0,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    arm_hours = {a: out[a]["projected_run_with_15pct_margin_gpu_hours"] for a in C.ARMS}
    out["projection"] = {
        "per_arm_margin_gpu_hours": arm_hours,
        "six_runs_margin_gpu_hours": 2.0 * sum(arm_hours.values()),
        "four_runs_margin_gpu_hours": 2.0 * (arm_hours["P"] + arm_hours["R"]),
        "fits_5h_with_6_runs": bool(2.0 * sum(arm_hours.values()) < cfg["budget"]["gpu_hours_total"]),
        "smoke_and_analysis_reserve_gpu_hours": 0.2,
    }
    C.write_json(args.out, out)
    print(json.dumps(out["projection"], indent=2))
    return 0


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    arm, seed = str(args.arm), int(args.seed)
    if arm not in C.ARMS or seed not in C.SEEDS:
        raise SystemExit("arm must be P/R/N and seed 0/1")
    device = torch.device(args.device)
    norm = AD.load_norm()
    cache_train = AD.load_cache("train")
    cache_valid = AD.load_cache("valid")
    entries_train = load_rel("train")
    entries_valid = load_rel("valid")
    y_train, y_valid = load_ys()
    out = C.RUNS_DIR / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    builder = lambda candidate, norm_, seed_, device_: M.build_arm(arm, norm_, seed_, device_)
    collate = p3_collate if arm == "P" else D.make_collate(entries_train)
    info = AT.train(
        arm, seed, device, recipe_variant="B", model_builder=builder,
        out_dir=out, collate_fn=collate, force=args.force,
    )
    print(json.dumps({"best_valid": info["best_valid"], "final": info["final"],
                      "timing": info["timing"]}, indent=2))

    # extra predictions / provenance (does not modify the trainer's train.json)
    final_model, final_info = M.build_arm(arm, norm, seed, device)
    M.load_checkpoint(final_model, out / "final.pt", device)
    final_pred = eval_pred(final_model, cache_valid, np.arange(C.N_VALID), device,
                           int(cfg["recipe"]["valid_batch_size"]),
                           p3_collate if arm == "P" else D.make_collate(entries_valid))
    (out / "predictions").mkdir(exist_ok=True)
    np.savez(out / "predictions/valid_final.npz", row=np.arange(C.N_VALID, dtype=np.int64),
             pred=final_pred.astype(np.float32))
    best_z = np.load(out / "predictions/valid_best.npz", allow_pickle=False)
    final_mae = C.mae(final_pred, y_valid)
    best_mae = C.mae(best_z["pred"].astype(np.float64), y_valid)
    if abs(best_mae - float(info["best_valid"]["valid_mae"])) > 5e-6:
        raise SystemExit(f"best-valid replay mismatch: {best_mae} vs {info['best_valid']['valid_mae']}")
    if abs(final_mae - float(info["final"]["valid_mae"])) > 5e-6:
        raise SystemExit(f"final-valid replay mismatch: {final_mae} vs {info['final']['valid_mae']}")
    win = C.window_metrics(info["curve"])
    init_model, init_info = M.build_arm(arm, norm, seed, "cpu")
    extra = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm, "seed": seed,
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "ssl": final_info["ssl"],
        "parameters": final_info["parameters"],
        "parameter_breakdown": final_info["parameter_breakdown"],
        "init_common_state_sha256": init_info["common_state_sha256"],
        "init_relation_state_sha256": init_info["relation_state_sha256"],
        "relation_mode": getattr(init_model, "relation_mode", None),
        "best_checkpoint_state_sha256": _state_sha(out / "best.pt"),
        "final_checkpoint_state_sha256": _state_sha(out / "final.pt"),
        "best_valid_replay_mae": best_mae,
        "final_valid_replay_mae": final_mae,
        "predictions": {
            "valid_best_sha256": C.sha256_file(out / "predictions/valid_best.npz"),
            "valid_final_sha256": C.sha256_file(out / "predictions/valid_final.npz"),
        },
        "labels": {
            "train_sha256": C.sha256_array(y_train),
            "valid_sha256": C.sha256_array(y_valid),
        },
        "data_order": order_sequence_hash(seed, epochs=int(cfg["recipe"]["epochs"]), n_rows=len(y_train)),
        "windows": {k: v for k, v in win.items() if k not in ("epochs", "valid_curve")},
        "relation_cache": {
            "train_sha256": C.sha256_file(rel_cache_path("train")),
            "valid_sha256": C.sha256_file(rel_cache_path("valid")),
        },
        "null_fingerprint": {
            "train": D.relation_fingerprint(entries_train),
            "valid": D.relation_fingerprint(entries_valid),
            "seed": int(D.NULL_SEED),
        },
        "official_test_loaded": False,
        "curve_sha256": C.sha256_file(out / "curve.csv"),
    }
    C.write_json(out / "run_extra.json", extra)
    return 0


# ---------------------------------------------------------------------------
# score / analyze
# ---------------------------------------------------------------------------


def _load_pred(run: Path, which: str) -> np.ndarray:
    z = np.load(run / f"predictions/valid_{which}.npz", allow_pickle=False)
    pred = z["pred"].astype(np.float64)
    rows = z["row"].astype(np.int64)
    if len(pred) != C.N_VALID or not np.array_equal(rows, np.arange(C.N_VALID)):
        raise SystemExit(f"{run}: predictions are not the full 1000-row valid split")
    return pred


def cmd_score(args: argparse.Namespace) -> int:
    run = Path(args.run).expanduser().resolve()
    info = C.read_json(run / "train.json")
    extra = C.read_json(run / "run_extra.json")
    groups = C.load_valid_groups()
    y = groups["y"]
    assert np.allclose(y, np.asarray(AD.load_cache("valid")["labels"], dtype=np.float64), rtol=0.0, atol=0.0)
    arm, seed = str(info["candidate"]), int(info["seed"])
    out: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID, "arm": arm, "seed": seed,
        "git_commit": info["git_commit"], "parameters": info["parameter_count"],
        "recipe": info["recipe"], "best_valid": info["best_valid"], "final": info["final"],
        "windows": extra["windows"], "run_extra": extra,
        "runs": str(run.relative_to(C.REPO)) if run.is_relative_to(C.REPO) else str(run),
    }
    for which in ("best", "final"):
        pred = _load_pred(run, which)
        ae = np.abs(pred - y)
        out[which + "_metrics"] = C.metric_block(pred, y)
        out[which + "_groups"] = C.group_table(ae, groups["groups"])
        out[which + "_tail_calibration"] = {
            "tail": C.tail_calibration(pred, y, groups["tail"]),
            "tail_drop_row172": C.tail_calibration(pred, y, groups["tail"], drop_row172=True),
        }
        out[which + "_p3b_comparison"] = {
            "gain": float(groups["p3b_ae"].mean() - ae.mean()),
            "drop_top10": C.topk_removed_gain(ae, groups["p3b_ae"], fixed_rows=groups["top10_p3b"]),
            "bootstrap": C.paired_bootstrap(ae, groups["p3b_ae"]),
            "row172_ae": float(ae[172]),
            "mae_drop_row172": float(np.delete(ae, 172).mean()),
        }
        pred_path = C.ANALYSIS_DIR / "per_molecule" / f"{arm}_s{seed}_{which}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row_id", "y", "pred", "ae", "p3b_pred", "p3b_ae", "delta_ae_vs_p3b",
                        "radius2_unseen", "ordinary", "tail", "ring_rich"])
            for i in range(C.N_VALID):
                w.writerow([i, y[i], pred[i], ae[i], groups["p3b_pred"][i], groups["p3b_ae"][i],
                            ae[i] - groups["p3b_ae"][i], int(groups["radius2_unseen"][i]),
                            int(groups["ordinary"][i]), int(groups["tail"][i]),
                            int(groups["n_rings"][i] >= 4)])
        out[which + "_per_molecule_sha256"] = C.sha256_file(pred_path)
    C.write_json(C.ANALYSIS_DIR / "scores" / f"{arm}_s{seed}.json", out)
    print(json.dumps({"arm": arm, "seed": seed, "best": out["best_metrics"]["mae"],
                      "final": out["final_metrics"]["mae"], "windows": out["windows"]}, indent=2))
    return 0


def _gain(ae_a: np.ndarray, ae_b: np.ndarray) -> float:
    """Positive = first (a) better than second (b)."""
    return float(np.asarray(ae_b).mean() - np.asarray(ae_a).mean())


def _contrast(ae_a, ae_b, groups, name_a, name_b) -> dict[str, Any]:
    boot = C.paired_bootstrap(ae_a, ae_b)
    d = C.delta_stats(ae_a, ae_b, groups["groups"])
    return {
        "first": name_a, "second": name_b, "positive_means_first_better": True,
        "gain": _gain(ae_a, ae_b),
        "bootstrap": boot,
        "delta": d,
        "drop_row172": {
            "mae_first": float(np.delete(ae_a, 172).mean()),
            "mae_second": float(np.delete(ae_b, 172).mean()),
            "gain": float(np.delete(ae_b, 172).mean() - np.delete(ae_a, 172).mean()),
        },
        "row172_contribution_to_gain": float((ae_b[172] - ae_a[172]) / len(ae_a)),
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = C.load_config()
    groups = C.load_valid_groups()
    y = groups["y"]
    scores = {}
    for p in sorted((C.ANALYSIS_DIR / "scores").glob("*_s*.json")):
        rec = C.read_json(p)
        scores[(rec["arm"], int(rec["seed"]))] = rec
    missing = [f"{a}_s{s}" for s in C.SEEDS for a in C.ARMS if (a, s) not in scores]
    summary: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "git_commit": C.git_commit(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "completed": sorted(f"{a}_s{s}" for (a, s) in scores),
        "missing": missing,
        "runs": {},
        "seeds": {},
        "decisions": {},
        "official_test_loaded": False,
    }
    for (arm, seed), rec in scores.items():
        ae_best = np.abs(_load_pred(C.REPO / rec["runs"], "best") - y)
        ae_final = np.abs(_load_pred(C.REPO / rec["runs"], "final") - y)
        summary["runs"][f"{arm}_s{seed}"] = {
            "best_valid": float(rec["best_valid"]["valid_mae"]),
            "best_epoch": int(rec["best_valid"]["epoch"]),
            "final_valid": float(rec["final"]["valid_mae"]),
            "eval_train_at_best": rec["best_valid"]["train_mae_eval_mode"],
            "W20": float(rec["windows"]["W20"]),
            "T100": float(rec["windows"]["T100"]),
            "parameters": int(rec["parameters"]),
            "init_common_state_sha256": rec["run_extra"]["init_common_state_sha256"],
            "init_relation_state_sha256": rec["run_extra"]["init_relation_state_sha256"],
            "valid_best_sha256": rec["run_extra"]["predictions"]["valid_best_sha256"],
            "valid_final_sha256": rec["run_extra"]["predictions"]["valid_final_sha256"],
            "order_stream_sha256": rec["run_extra"]["data_order"]["stream_sha256"],
            "old_groups_best": rec["best_groups"],
            "old_groups_final": rec["final_groups"],
        }
        summary["runs"][f"{arm}_s{seed}"]["_ae_best"] = ae_best
        summary["runs"][f"{arm}_s{seed}"]["_ae_final"] = ae_final

    for seed in C.SEEDS:
        if not all((a, seed) in scores for a in C.ARMS):
            continue
        sp = {}
        for arm in C.ARMS:
            rec = scores[(arm, seed)]
            ae = summary["runs"][f"{arm}_s{seed}"]["_ae_best"]
            sp[arm] = {
                "best_valid": float(rec["best_valid"]["valid_mae"]),
                "best_epoch": int(rec["best_valid"]["epoch"]),
                "final_valid": float(rec["final"]["valid_mae"]),
                "W20": float(rec["windows"]["W20"]),
                "T100": float(rec["windows"]["T100"]),
                "ae_best": ae,
            }
        contrasts = {}
        for a, b in (("R", "P"), ("N", "P"), ("R", "N")):
            contrasts[f"{a}_minus_{b}"] = _contrast(sp[a]["ae_best"], sp[b]["ae_best"], groups, a, b)
        sustained = {}
        for a, b in (("R", "P"), ("N", "P"), ("R", "N")):
            sustained[f"{a}_minus_{b}"] = {
                "W20_gain": float(sp[b]["W20"] - sp[a]["W20"]),
                "T100_gain": float(sp[b]["T100"] - sp[a]["T100"]),
            }
        summary["seeds"][str(seed)] = {
            "run_metrics": {a: {k: sp[a][k] for k in ("best_valid", "best_epoch", "final_valid", "W20", "T100")}
                            for a in C.ARMS},
            "contrasts_best": contrasts,
            "sustained": sustained,
        }
        # final-checkpoint contrasts are secondary and not used for the verdict
        final_contrasts = {}
        for a, b in (("R", "P"), ("N", "P"), ("R", "N")):
            final_contrasts[f"{a}_minus_{b}"] = _contrast(
                summary["runs"][f"{a}_s{seed}"]["_ae_final"],
                summary["runs"][f"{b}_s{seed}"]["_ae_final"], groups, a, b)
        summary["seeds"][str(seed)]["contrasts_final"] = final_contrasts

    # pre-registered decision rules (config, frozen before the first formal run)
    dec = cfg["decision"]
    both = all(str(s) in summary["seeds"] for s in C.SEEDS)
    summary["decisions"]["both_seeds_complete"] = bool(both)
    summary["decisions"]["completed_seed_count"] = int(
        sum(1 for s in C.SEEDS if all((a, s) in scores for a in C.ARMS))
    )
    if both:
        def _seed_gain(a, b, key="best_valid"):
            return [float(summary["seeds"][str(s)]["run_metrics"][b][key]
                          - summary["seeds"][str(s)]["run_metrics"][a][key]) for s in C.SEEDS]

        def _seed_group_gain(a, b, group):
            out = []
            for s in C.SEEDS:
                aa = summary["runs"][f"{a}_s{s}"]["_ae_best"]
                bb = summary["runs"][f"{b}_s{s}"]["_ae_best"]
                mask = groups["groups"][group]
                out.append(float(bb[mask].mean() - aa[mask].mean()))
            return out

        rp = _seed_gain("R", "P")
        rn = _seed_gain("R", "N")
        np_ = _seed_gain("N", "P")
        rp_u = _seed_group_gain("R", "P", "unseen_ordinary")
        rn_u = _seed_group_gain("R", "N", "unseen_ordinary")
        w20_rp = [summary["seeds"][str(s)]["sustained"]["R_minus_P"]["W20_gain"] for s in C.SEEDS]
        t100_rp = [summary["seeds"][str(s)]["sustained"]["R_minus_P"]["T100_gain"] for s in C.SEEDS]
        w20_rn = [summary["seeds"][str(s)]["sustained"]["R_minus_N"]["W20_gain"] for s in C.SEEDS]
        t100_rn = [summary["seeds"][str(s)]["sustained"]["R_minus_N"]["T100_gain"] for s in C.SEEDS]
        min_gain = float(dec["h1_min_mean_gain"])
        min_group = float(dec["h1_unseen_group_min_gain"])
        no_reverse = float(dec["sustained_reverse_max"])
        h1_overall = bool(all(g > 0 for g in rp) and float(np.mean(rp)) >= min_gain
                          and float(np.mean(w20_rp)) >= -no_reverse
                          and float(np.mean(t100_rp)) >= -no_reverse)
        h1_unseen = bool(all(g > 0 for g in rp_u) and float(np.mean(rp_u)) >= min_group
                         and float(np.mean(rp)) >= -no_reverse)
        h1_small_consistent = bool(all(g > 0 for g in rp) and not h1_overall)
        h2 = bool(all(g > 0 for g in rn) and float(np.mean(rn)) >= min_gain
                  and float(np.mean(w20_rn)) >= -no_reverse
                  and float(np.mean(t100_rn)) >= -no_reverse)
        h2_small_consistent = bool(all(g > 0 for g in rn) and not h2)
        # row172 / top10 dominance of the R-P gain
        dom = {}
        for s in C.SEEDS:
            aa = summary["runs"][f"R_s{s}"]["_ae_best"]; bb = summary["runs"][f"P_s{s}"]["_ae_best"]
            cont = summary["seeds"][str(s)]["contrasts_best"]["R_minus_P"]
            dom[s] = {
                "gain": float(np.mean(bb) - np.mean(aa)),
                "gain_drop_row172": float(np.delete(bb, 172).mean() - np.delete(aa, 172).mean()),
                "row172_ae_R": float(aa[172]),
                "row172_ae_P": float(bb[172]),
                "tail_gain": float(bb[groups["tail"]].mean() - aa[groups["tail"]].mean()),
                "tail_drop_row172_gain": float(
                    bb[groups["tail"] & (np.arange(C.N_VALID) != 172)].mean()
                    - aa[groups["tail"] & (np.arange(C.N_VALID) != 172)].mean()),
                "ordinary_gain": float(bb[groups["ordinary"]].mean() - aa[groups["ordinary"]].mean()),
                "unseen_ordinary_gain": float(
                    bb[groups["groups"]["unseen_ordinary"]].mean()
                    - aa[groups["groups"]["unseen_ordinary"]].mean()),
                "better_fraction": float((aa < bb).mean()),
            }
        summary["decisions"].update({
            "H1_overall_supported": h1_overall,
            "H1_unseen_ordinary_supported": h1_unseen,
            "H1_supported": bool(h1_overall or h1_unseen),
            "H1_small_consistent_only": h1_small_consistent,
            "H2_supported": h2,
            "H2_small_consistent_only": h2_small_consistent,
            "R_minus_P_seed_gains": rp,
            "R_minus_P_mean_gain": float(np.mean(rp)),
            "N_minus_P_seed_gains": np_,
            "R_minus_N_seed_gains": rn,
            "R_minus_N_mean_gain": float(np.mean(rn)),
            "R_minus_P_unseen_ordinary_seed_gains": rp_u,
            "R_minus_N_unseen_ordinary_seed_gains": rn_u,
            "R_minus_P_W20_seed_gains": w20_rp,
            "R_minus_P_T100_seed_gains": t100_rp,
            "R_minus_N_W20_seed_gains": w20_rn,
            "R_minus_N_T100_seed_gains": t100_rn,
            "row172_dominance": dom,
            "test_trigger": bool(h1_overall and both and all(g > 0 for g in rp)),
            "verdict": _verdict(h1_overall, h1_unseen, h2, h1_small_consistent, rp, rn, np_),
        })
    else:
        summary["decisions"]["test_trigger"] = False
        summary["decisions"]["verdict"] = "INCOMPLETE_SEEDS"

    # strip large arrays before JSON
    for rec in summary["runs"].values():
        rec.pop("_ae_best", None)
        rec.pop("_ae_final", None)
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)
    _write_tables(summary, groups)
    print(json.dumps({k: summary["decisions"].get(k) for k in
                      ("verdict", "H1_overall_supported", "H1_unseen_ordinary_supported",
                       "H2_supported", "R_minus_P_mean_gain", "R_minus_N_mean_gain", "test_trigger")},
                     indent=2))
    return 0


def _verdict(h1_overall, h1_unseen, h2, h1_small, rp, rn, np_) -> str:
    if h1_overall and h2:
        return "RELATION_CORRESPONDENCE_WITH_REPLICATED_OVERALL_GAIN"
    if h1_overall:
        return "REPLICATED_OVERALL_GAIN_NOT_R_N_SPECIFIC"
    if h1_unseen and h2:
        return "UNSEEN_GROUP_GAIN_WITH_R_N_SUPPORT_SMALL_OVERALL_COST"
    if h1_unseen:
        return "UNSEEN_GROUP_GAIN_ONLY"
    if h1_small:
        return "SMALL_CONSISTENT_GAIN_BELOW_PRE_REGISTERED_BAND"
    if all(g < 0 for g in rp) and all(g < 0 for g in rn) and all(g < 0 for g in np_):
        return "CURRENT_CONFIG_NO_SUPPORT_FOR_CONTINUATION"
    return "UNSTABLE_OR_MIXED"


def _write_tables(summary: Mapping[str, Any], groups: Mapping[str, Any]) -> None:
    rows = ["# Main valid table", "",
            "| run | best valid | epoch | final valid | W20 | T100 | train@best | params |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("P_s0", "R_s0", "N_s0", "P_s1", "R_s1", "N_s1"):
        if name not in summary["runs"]:
            continue
        r = summary["runs"][name]
        rows.append(f"| {name} | {r['best_valid']:.6f} | {r['best_epoch']} | {r['final_valid']:.6f} | "
                    f"{r['W20']:.6f} | {r['T100']:.6f} | {r['eval_train_at_best']:.6f} | {r['parameters']} |")
    rows += ["", "Positive contrast = first arm better.", "",
             "| seed | contrast | gain | 95% CI | median ΔAE | molecules better |",
             "|---:|---|---:|---|---:|---:|"]
    for s in (0, 1):
        if str(s) not in summary["seeds"]:
            continue
        for key, c in summary["seeds"][str(s)]["contrasts_best"].items():
            b = c["bootstrap"]
            rows.append(f"| {s} | {key} | {c['gain']:+.6f} | "
                        f"[{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | "
                        f"{b['median_delta']:+.6f} | {b['better_fraction']:.3f} |")
    rows += ["", "## W20 / T100 persistence (positive = first better)", "",
             "| seed | contrast | W20 gain | T100 gain |", "|---:|---|---:|---:|"]
    for s in (0, 1):
        if str(s) not in summary["seeds"]:
            continue
        for key, c in summary["seeds"][str(s)]["sustained"].items():
            rows.append(f"| {s} | {key} | {c['W20_gain']:+.6f} | {c['T100_gain']:+.6f} |")
    (C.ANALYSIS_DIR / "main_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    g = ["# Fixed old groups (best-valid checkpoint)", "",
         "| run | all | seen_ord | unseen_ord | seen_tail | unseen_tail | ordinary | tail | tail−172 | ring-rich | ring-rich-unseen |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("P_s0", "R_s0", "N_s0", "P_s1", "R_s1", "N_s1"):
        if name not in summary["runs"]:
            continue
        t = summary["runs"][name]["old_groups_best"]
        def q(k, key="mae"):
            return f"{t[k][key]:.5f}" if k in t else "—"
        g.append(f"| {name} | {q('all')} | {q('seen_ordinary')} | {q('unseen_ordinary')} | "
                 f"{q('seen_tail')} | {q('unseen_tail')} | {q('ordinary_935')} | {q('tail_65')} | "
                 f"{q('tail_drop_row172')} | {q('ring_rich_213')} | {q('ring_rich_unseen_47')} |")
    g += ["", "Row 172 is retained in every headline number; the tail−172 column is a diagnostic only."]
    (C.ANALYSIS_DIR / "groups.md").write_text("\n".join(g) + "\n", encoding="utf-8")

    d = ["# Frozen decision", ""]
    if "both_seeds_complete" not in summary["decisions"]:
        d.append("No decision recorded.")
    else:
        for k in ("both_seeds_complete", "H1_overall_supported", "H1_unseen_ordinary_supported",
                  "H1_supported", "H1_small_consistent_only", "H2_supported",
                  "H2_small_consistent_only", "R_minus_P_mean_gain", "R_minus_N_mean_gain",
                  "R_minus_P_unseen_ordinary_seed_gains", "R_minus_N_unseen_ordinary_seed_gains",
                  "R_minus_P_W20_seed_gains", "R_minus_P_T100_seed_gains",
                  "R_minus_N_W20_seed_gains", "R_minus_N_T100_seed_gains",
                  "test_trigger", "verdict"):
            d.append(f"* **{k}**: {summary['decisions'].get(k)}")
        d += ["", "## Row-172 / tail dominance of R−P", ""]
        for s, v in summary["decisions"]["row172_dominance"].items():
            d.append(f"* seed {s}: gain {v['gain']:+.6f}, drop-row172 {v['gain_drop_row172']:+.6f}, "
                     f"row172 AE R/P {v['row172_ae_R']:.4f}/{v['row172_ae_P']:.4f}, "
                     f"tail gain {v['tail_gain']:+.6f}, ordinary gain {v['ordinary_gain']:+.6f}, "
                     f"unseen+ordinary gain {v['unseen_ordinary_gain']:+.6f}, better fraction {v['better_fraction']:.3f}")
    (C.ANALYSIS_DIR / "decision.md").write_text("\n".join(d) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    arm, seed, which = str(args.arm), int(args.seed), str(args.which)
    if arm not in ("R", "N"):
        raise SystemExit("probe is defined for R/N")
    device = torch.device(args.device)
    norm = AD.load_norm()
    cache_train = AD.load_cache("train")
    cache_valid = AD.load_cache("valid")
    entries_train = load_rel("train")
    entries_valid = load_rel("valid")
    y_train, y_valid = load_ys()
    run = C.RUNS_DIR / f"{arm}_s{seed}"
    model, info = M.build_arm(arm, norm, seed, device)
    M.load_checkpoint(model, run / f"{which}.pt", device)
    collate_v = D.make_collate(entries_valid)
    modes = ("real", "null", "bypass")
    preds = {m: eval_pred(model, cache_valid, np.arange(C.N_VALID), device, 256, collate_v,
                          relation_mode=m, bypass=(m == "bypass")) for m in modes}
    groups = C.load_valid_groups()
    y = groups["y"]
    out: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID, "arm": arm, "seed": seed, "which": which,
        "checkpoint_sha256": _state_sha(run / f"{which}.pt"),
        "modes": {}, "amplitude": {}, "gradient": {},
    }
    for m, pred in preds.items():
        ae = np.abs(pred - y)
        out["modes"][m] = {
            "valid_mae": float(ae.mean()),
            "metrics": C.metric_block(pred, y),
            "groups": C.group_table(ae, groups["groups"]),
            "pred_sha256": C.sha256_array(pred.astype(np.float32)),
        }
    base = preds["real"]
    for m in ("null", "bypass"):
        d = np.abs(preds[m] - base)
        out["modes"][f"{m}_vs_real"] = {
            "mean_abs_pred_change": float(d.mean()),
            "max_abs_pred_change": float(d.max()),
            "pred_correlation": float(np.corrcoef(preds[m], base)[0, 1]),
            "mae_change": float(out["modes"][m]["valid_mae"] - out["modes"]["real"]["valid_mae"]),
        }
    # fixed real train probe for branch/main amplitude
    rows = np.arange(128, dtype=np.int64)
    batch = AT._to_device(D.make_collate(entries_train)(cache_train, rows, "P3N", valid=False), device)
    model.eval()
    with torch.no_grad():
        _, stats = model(batch, relation_mode="real", return_stats=True)
    out["amplitude"] = {k: float(v.mean()) for k, v in stats["relation"].items()}
    out["amplitude"]["n_tokens"] = int(batch["n_tokens_total"])
    out["amplitude"]["n_directed_edges"] = int(batch["r_src"].numel())
    # gradient norms on one fixed train batch
    model.train()
    target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
    model.zero_grad(set_to_none=True)
    loss = (model(batch, relation_mode="real") - target).abs().mean()
    loss.backward()
    rel = [p for n, p in model.named_parameters() if n.startswith("relation.")]
    com = [p for n, p in model.named_parameters() if not n.startswith("relation.")]
    def _gnorm(ps):
        gs = [p.grad for p in ps if p.grad is not None]
        return float(math.sqrt(sum(float((g * g).sum()) for g in gs))), len(gs), bool(all(torch.isfinite(g).all().item() for g in gs))
    gr, nr, fr = _gnorm(rel)
    gc, nc, fc = _gnorm(com)
    out["gradient"] = {
        "loss": float(loss.detach()),
        "relation_grad_norm": gr, "relation_tensors": nr, "relation_finite": fr,
        "common_grad_norm": gc, "common_tensors": nc, "common_finite": fc,
        "relation_share": float(gr / (gr + gc + 1e-12)),
        "per_relation_tensor": {n: float(p.grad.norm()) for n, p in model.named_parameters()
                                if n.startswith("relation.") and p.grad is not None},
    }
    model.zero_grad(set_to_none=True)
    out["official_test_loaded"] = False
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = C.REPO / out_dir
    C.write_json(out_dir / f"{arm}_s{seed}_{which}.json", out)
    print(json.dumps({"arm": arm, "seed": seed, "which": which,
                      "valid_mae": {m: out["modes"][m]["valid_mae"] for m in modes},
                      "mae_change_null": out["modes"]["null_vs_real"]["mae_change"],
                      "mae_change_bypass": out["modes"]["bypass_vs_real"]["mae_change"],
                      "amplitude": out["amplitude"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# terminal test (guarded)
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> int:
    summary_path = C.ANALYSIS_DIR / "summary.json"
    if not summary_path.exists():
        raise SystemExit("run analyze first; the frozen trigger is not on record")
    summary = C.read_json(summary_path)
    trigger = bool(summary.get("decisions", {}).get("test_trigger", False))
    if not trigger and not args.force:
        raise SystemExit("frozen valid trigger did not fire; official test stays unread")
    if int(summary.get("decisions", {}).get("completed_seed_count", 2)) < 2 and not args.force:
        raise SystemExit("second supervised seed is incomplete; test trigger requires two seeds")
    cfg = C.load_config()
    device = torch.device(args.device)
    # terminal: build/load the official test object cache (labels are read here)
    if not AD.cache_path("test").exists():
        AD.prep(("test",), force=False)
    cache_test = AD.load_cache("test")
    entries_test = D.build_split("test")
    y_test = np.asarray(cache_test["labels"], dtype=np.float64)
    test_ref = C.REPO / "tracks/structural-sufficiency/zinc_restricted_adaptation_v1/analysis/reference_preds_test.csv"
    with open(test_ref, newline="", encoding="utf-8") as fh:
        ref_rows = list(csv.DictReader(fh))
    ref_rows.sort(key=lambda r: int(r["row"]))
    if not np.allclose(np.asarray([float(r["y_raw"]) for r in ref_rows]), y_test, rtol=0.0, atol=0.0):
        raise SystemExit("test label table disagrees with the official PyG test labels")
    unseen = np.asarray([int(r["signature_unseen"]) for r in ref_rows], dtype=bool)
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in ref_rows], dtype=bool)
    tail = ~ordinary
    test_groups = {
        "all": np.ones(len(y_test), dtype=bool),
        "seen_ordinary": ~unseen & ordinary, "unseen_ordinary": unseen & ordinary,
        "seen_tail": ~unseen & tail, "unseen_tail": unseen & tail,
    }
    norm = AD.load_norm()
    collate_v = p3_collate
    collate_r = D.make_collate(entries_test)
    out: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID, "split": "official PyG ZINC test", "n": len(y_test),
        "read_once": True, "first_blind_test": False,
        "main_table": "best-valid checkpoints; final is a pre-declared sensitivity table",
        "test_label_sha256": C.sha256_array(y_test),
        "columns": {},
    }
    pred_dir = C.ANALYSIS_DIR / "test_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("P", "R"):
        for seed in C.SEEDS:
            run = C.RUNS_DIR / f"{arm}_s{seed}"
            for which in ("best", "final"):
                model, _ = M.build_arm(arm, norm, seed, device)
                M.load_checkpoint(model, run / f"{which}.pt", device)
                pred = eval_pred(model, cache_test, np.arange(len(y_test)), device, 256,
                                 collate_v if arm == "P" else collate_r,
                                 relation_mode="real" if arm == "R" else None)
                key = f"{arm}_s{seed}_{which}"
                np.savez(pred_dir / f"{key}_test.npz", pred=pred.astype(np.float32),
                         y=y_test.astype(np.float32), row=np.arange(len(y_test), dtype=np.int64))
                ae = np.abs(pred - y_test)
                out["columns"][key] = {
                    "test_mae": float(ae.mean()),
                    "metrics": C.metric_block(pred, y_test),
                    "groups": C.group_table(ae, test_groups),
                    "pred_sha256": C.sha256_array(pred.astype(np.float32)),
                }
    # per-selector R-P, best and final
    for which in ("best", "final"):
        for seed in C.SEEDS:
            a = out["columns"][f"R_s{seed}_{which}"]["test_mae"]
            b = out["columns"][f"P_s{seed}_{which}"]["test_mae"]
            out["columns"][f"R_minus_P_s{seed}_{which}"] = {"test_mae_delta": float(a - b)}
        r = [out["columns"][f"R_s{s}_{which}"]["test_mae"] for s in C.SEEDS]
        p = [out["columns"][f"P_s{s}_{which}"]["test_mae"] for s in C.SEEDS]
        out["columns"][f"R_mean_{which}"] = {"test_mae": float(np.mean(r))}
        out["columns"][f"P_mean_{which}"] = {"test_mae": float(np.mean(p))}
        out["columns"][f"R_minus_P_mean_{which}"] = {"test_mae_delta": float(np.mean(r) - np.mean(p))}
    out["reached_0.060"] = {
        k: bool(v.get("test_mae", 1e9) < 0.060) for k, v in out["columns"].items() if "test_mae" in v
    }
    C.write_json(C.ANALYSIS_DIR / "test_terminal.json", out)
    lines = ["# Terminal test read (post-selection, descriptive)", "",
             f"* test label sha256: `{out['test_label_sha256']}`",
             "* main table = best-valid checkpoints; final = pre-declared sensitivity table",
             "* the official test is historically exposed; this is not a first blind test",
             "", "| column | test MAE | <0.060 |", "|---|---:|---|"]
    for k, v in out["columns"].items():
        if "test_mae" in v:
            lines.append(f"| {k} | {v['test_mae']:.6f} | {v['test_mae'] < 0.060} |")
        elif "test_mae_delta" in v:
            lines.append(f"| {k} (Δ) | {v['test_mae_delta']:+.6f} | — |")
    (C.ANALYSIS_DIR / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out["columns"].items() if "test_mae" in v}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prep")
    q.add_argument("--manifest", default="analysis/prep.json")
    q.set_defaults(func=cmd_prep)

    q = sub.add_parser("checks")
    q.add_argument("--out", default=str(C.ANALYSIS_DIR / "cpu_checks.json"))
    q.add_argument("--replay-limit", type=int, default=None)
    q.set_defaults(func=cmd_checks)

    q = sub.add_parser("timing")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--steps", type=int, default=30)
    q.add_argument("--out", default=str(C.ANALYSIS_DIR / "timing.json"))
    q.add_argument("--save-probe", default="cache/timing_probe.pt")
    q.set_defaults(func=cmd_timing)

    q = sub.add_parser("train")
    q.add_argument("--arm", choices=C.ARMS, required=True)
    q.add_argument("--seed", type=int, choices=C.SEEDS, required=True)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_train)

    q = sub.add_parser("score")
    q.add_argument("--run", required=True)
    q.set_defaults(func=cmd_score)

    q = sub.add_parser("analyze")
    q.set_defaults(func=cmd_analyze)

    q = sub.add_parser("probe")
    q.add_argument("--arm", choices=("R", "N"), required=True)
    q.add_argument("--seed", type=int, choices=C.SEEDS, required=True)
    q.add_argument("--which", choices=("best", "final"), default="best")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--out-dir", default="cache/probe")
    q.set_defaults(func=cmd_probe)

    q = sub.add_parser("test")
    q.add_argument("--device", default="cpu")
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_test)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
