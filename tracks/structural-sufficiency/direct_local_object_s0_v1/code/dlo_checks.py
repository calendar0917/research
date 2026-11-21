"""Non-trivial pre-training checks (the only checks this round runs).

1. data identity / split / padding / coverage / old core counts;
2. LUT canonicalisation against brute force;
3. re-labelling invariance of the complete raw frame multisets and of the
   whole network (CPU FP64; optional GPU FP32);
4. batch-order consistency (single molecule vs shuffled multi-molecule batch;
   scatter vs explicit per-object loop);
5. one real Train batch: finite non-zero gradients in every module and an
   optimizer step that actually moves the frame encoder.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Mapping

import numpy as np
import torch

import dlo_common as C
import dlo_frames as F
import dlo_model as M
from dlo_cache import ArmCache, load_payload
from dlo_train import predict


def _tensor_pack(pack: Mapping[str, Any], device: str, dtype=torch.float32) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pack.items():
        if key in {"n_molecules", "n_objects", "n_frames"}:
            out[key] = int(value)
        elif key in {"frame_weight", "frame_node_mask", "frame_edge_mask", "frame_adj", "obj_bucket"}:
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device)
        else:
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.long, device=device)
    return out


def _mol_arrays(payload: Mapping[str, np.ndarray], row: int, atom_inv: np.ndarray, bond_inv: np.ndarray):
    node_ptr = payload["node_ptr"]
    bond_ptr = payload["bond_ptr"]
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    n = a1 - a0
    atom_row = atom_inv[a0:a1].astype(np.int64)
    bond_row = np.full((n, n), -1, np.int32)
    adj: list[list[int]] = [[] for _ in range(n)]
    for e in range(b0, b1):
        u, v = int(payload["bond_u"][e]), int(payload["bond_v"][e])
        bond_row[u, v] = bond_row[v, u] = int(bond_inv[e])
        adj[u].append(v)
        adj[v].append(u)
    return {
        "n": n,
        "atom_row": atom_row,
        "bond_row": bond_row,
        "adj": adj,
        "n_bonds": b1 - b0,
    }


def _objects_from_cache(cache: ArmCache, row: int) -> list[tuple[int, tuple[int, ...]]]:
    s, e = int(cache.mol_obj_ptr[row]), int(cache.mol_obj_ptr[row + 1])
    out = []
    for i in range(s, e):
        k = int(cache.obj_size[i])
        out.append((int(cache.obj_root[i]), tuple(int(x) for x in cache.obj_atoms[i, :k])))
    return out


def _relabel(objects, perm: np.ndarray):
    mapped = [(int(perm[root]), tuple(sorted(int(perm[a]) for a in atoms))) for root, atoms in objects]
    mapped.sort(key=lambda x: (len(x[1]), x[1], x[0]))
    return mapped


def _frame_multiset(frames: Mapping[str, np.ndarray]) -> list[tuple[int, bytes, bytes, float]]:
    counts = frames["obj_frame_count"]
    weights = np.repeat(1.0 / counts.astype(np.float64), counts)
    out = []
    for i in range(frames["frame_k"].size):
        out.append((
            int(frames["frame_k"][i]),
            frames["frame_atom_attr"][i].tobytes(),
            frames["frame_bond_attr"][i].astype(np.int32).tobytes(),
            float(weights[i]),
        ))
    out.sort()
    return out


def _forward_single_pack(model: M.DLOFrameModel, frames: Mapping[str, np.ndarray],
                         sizes: np.ndarray, buckets: np.ndarray, n_atoms: int, n_bonds: int,
                         dtype=torch.float64, device: str = "cpu") -> np.ndarray:
    counts = frames["obj_frame_count"]
    n_obj = len(counts)
    f_ptr = np.zeros(n_obj + 1, np.int64)
    np.cumsum(counts, out=f_ptr[1:])
    n_frames = int(f_ptr[-1])
    ref_to_input = np.arange(n_frames, dtype=np.int64)
    frame_to_obj = np.repeat(np.arange(n_obj, dtype=np.int64), counts)
    weight = np.repeat(1.0 / counts.astype(np.float64), counts).astype(np.float32)
    pack = {
        "n_molecules": 1,
        "n_objects": n_obj,
        "n_frames": n_frames,
        "obj_to_mol": np.zeros(n_obj, np.int64),
        "obj_bucket": buckets.astype(np.uint8),
        "frame_to_obj": frame_to_obj,
        "frame_inverse": ref_to_input,
        "frame_weight": weight,
        "frame_k": frames["frame_k"],
        "frame_atom_attr": frames["frame_atom_attr"],
        "frame_bond_attr": frames["frame_bond_attr"],
        "frame_node_mask": (np.arange(6)[None, :] < frames["frame_k"][:, None]).astype(np.uint8),
        "frame_edge_mask": (frames["frame_bond_attr"] >= 0).astype(np.uint8),
        "frame_adj": (frames["frame_bond_attr"] >= 0).astype(np.uint8),
        "n_bucket": np.asarray([[int((buckets == 0).sum()), int((buckets == 1).sum())]], np.int32),
        "n_atoms": np.asarray([n_atoms], np.int64),
        "n_bonds": np.asarray([n_bonds], np.int64),
    }
    b = _tensor_pack(pack, device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        return model.forward_packed(b).detach().cpu().numpy()


def relabel_invariance(cfg, payload, stats, luts, device: str, gpu: bool) -> dict[str, Any]:
    atom_inv_map, atom_inv = np.unique(payload["atom_feat"], axis=0, return_inverse=True)
    bond_inv_map, bond_inv = np.unique(payload["bond_feat"], axis=0, return_inverse=True)
    atom_inv = atom_inv.reshape(-1).astype(np.int64)
    bond_inv = bond_inv.reshape(-1).astype(np.int64)
    if not np.array_equal(atom_inv_map, payload["atom_code"]):
        raise AssertionError("atom code table mismatch")
    if not np.array_equal(bond_inv_map, payload["bond_code"]):
        raise AssertionError("bond code table mismatch")
    rng = np.random.default_rng(7)
    cases: list[tuple[str, str, int, dict, list[tuple[int, tuple[int, ...]]], np.ndarray]] = []
    for arm in C.ARMS:
        cache = ArmCache(payload, stats, arm)
        for row in (0, 137, 2048):
            mol = _mol_arrays(payload, row, atom_inv, bond_inv)
            objs = _objects_from_cache(cache, row)
            perm = rng.permutation(mol["n"])
            cases.append(("real", arm, row, mol, objs, perm))
        # synthetic graph: 6-ring with a 2-atom branch
        n = 8
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (2, 6), (6, 7)]
        adj: list[list[int]] = [[] for _ in range(n)]
        bond_row = np.full((n, n), -1, np.int32)
        for e, (u, v) in enumerate(edges):
            adj[u].append(v)
            adj[v].append(u)
            bond_row[u, v] = bond_row[v, u] = int(e % len(payload["bond_code"]))
        atom_row = np.asarray([int((3 * i + 1) % payload["atom_code"].shape[0]) for i in range(n)], np.int64)
        sizes = C.load_config()["objects"]["arms"][arm]["core_sizes"]
        objs = F.all_connected_subsets(adj, sizes)
        objs.sort(key=lambda x: (len(x[1]), x[1], x[0]))
        perm = rng.permutation(n)
        cases.append(("synthetic", arm, -1, {"n": n, "atom_row": atom_row, "bond_row": bond_row,
                                             "adj": adj, "n_bonds": len(edges)}, objs, perm))

    records: list[dict[str, Any]] = []
    for kind, arm, row, mol, objs, perm in cases:
        rel = _relabel(objs, perm)
        if kind == "real":
            atom_row2 = np.zeros_like(mol["atom_row"])
            atom_row2[perm] = mol["atom_row"]
            bond_row2 = np.full_like(mol["bond_row"], -1)
            for i in range(mol["n"]):
                for j in range(mol["n"]):
                    if mol["bond_row"][i, j] >= 0:
                        bond_row2[perm[i], perm[j]] = mol["bond_row"][i, j]
        else:
            atom_row2 = np.zeros_like(mol["atom_row"])
            atom_row2[perm] = mol["atom_row"]
            bond_row2 = np.full_like(mol["bond_row"], -1)
            for i in range(mol["n"]):
                for j in range(mol["n"]):
                    if mol["bond_row"][i, j] >= 0:
                        bond_row2[perm[i], perm[j]] = mol["bond_row"][i, j]
        f1 = F.frames_for_objects(mol["atom_row"], mol["bond_row"], objs, luts, check=True)
        f2 = F.frames_for_objects(atom_row2, bond_row2, rel, luts, check=True)
        m1 = _frame_multiset(f1)
        m2 = _frame_multiset(f2)
        if m1 != m2:
            raise AssertionError(f"frame multiset differs after relabelling ({kind}, {arm}, {row})")
        sizes1 = np.asarray([len(x[1]) for x in objs])
        sizes2 = np.asarray([len(x[1]) for x in rel])
        arm_cfg = cfg["objects"]["arms"][arm]
        bsplit = int(arm_cfg["bucket_split"])
        buckets1 = (sizes1 > bsplit).astype(np.uint8)
        buckets2 = (sizes2 > bsplit).astype(np.uint8)
        model = M.build_model(cfg, payload, seed=0)
        M.apply_train_support(model, stats)
        M.bind_constants(model, stats, arm)
        model.double()
        y1 = _forward_single_pack(model, f1, sizes1, buckets1, mol["n"], mol["n_bonds"], torch.float64, "cpu")
        y2 = _forward_single_pack(model, f2, sizes2, buckets2, mol["n"], mol["n_bonds"], torch.float64, "cpu")
        diff64 = float(np.max(np.abs(y1 - y2)))
        rec: dict[str, Any] = {
            "kind": kind, "arm": arm, "row": int(row),
            "n_objects": int(sizes1.size), "n_frames": int(f1["frame_k"].size),
            "frame_multiset_equal": True,
            "cpu_fp64_max_abs_diff": diff64,
        }
        if gpu and torch.cuda.is_available():
            model32 = M.build_model(cfg, payload, seed=0).to("cuda")
            M.apply_train_support(model32, stats)
            M.bind_constants(model32, stats, arm)
            g1 = _forward_single_pack(model32, f1, sizes1, buckets1, mol["n"], mol["n_bonds"], torch.float32, "cuda")
            g2 = _forward_single_pack(model32, f2, sizes2, buckets2, mol["n"], mol["n_bonds"], torch.float32, "cuda")
            rec["gpu_fp32_max_abs_diff"] = float(np.max(np.abs(g1 - g2)))
        records.append(rec)
        if diff64 > 1e-10:
            raise AssertionError(f"relabelling FP64 diff {diff64} > 1e-10")
        if "gpu_fp32_max_abs_diff" in rec and rec["gpu_fp32_max_abs_diff"] > 1e-4:
            raise AssertionError(f"relabelling FP32 diff {rec['gpu_fp32_max_abs_diff']} > 1e-4")
    return {"records": records, "max_cpu_fp64": max(r["cpu_fp64_max_abs_diff"] for r in records),
            "max_gpu_fp32": max((r.get("gpu_fp32_max_abs_diff", 0.0) for r in records), default=None)}


def batch_and_gradient(cfg, payload, stats, cache: ArmCache, device: str) -> dict[str, Any]:
    C.set_deterministic(0)
    model, support = M.build_model(cfg, payload, seed=0), None
    model = model.to(device)
    support = M.apply_train_support(model, stats)
    M.bind_constants(model, stats, cache.arm)

    # --- batch-order consistency
    rows = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], np.int64)
    rows = rows[rows < cache.n_molecules]
    alone = np.stack([predict(cache, model, np.asarray([r]), device, 1)[0] for r in rows])
    shuffled_rows = rows[np.random.default_rng(11).permutation(rows.size)]
    together = predict(cache, model, shuffled_rows, device, int(cfg["recipe"]["eval_batch"]))
    # map predictions back by row identity
    pos = {int(r): i for i, r in enumerate(shuffled_rows)}
    together_aligned = np.asarray([together[pos[int(r)]] for r in rows])
    batch_diff = float(np.max(np.abs(alone - together_aligned)))

    # --- scatter vs explicit object loop on one batch
    pack = cache.pack(rows)
    b = _tensor_pack(pack, device)
    model.eval()
    with torch.no_grad():
        H = model.encode_frames(b["frame_atom_attr"], b["frame_bond_attr"], b["frame_node_mask"],
                                b["frame_edge_mask"], b["frame_adj"], b["frame_k"])
        hf = H[b["frame_inverse"]] * b["frame_weight"].unsqueeze(-1)
        h_scatter = hf.new_zeros((int(pack["n_objects"]), hf.shape[1])).index_add_(0, b["frame_to_obj"], hf)
        ptr = np.zeros(int(pack["n_objects"]) + 1, np.int64)
        np.cumsum(np.bincount(np.asarray(pack["frame_to_obj"]), minlength=int(pack["n_objects"])), out=ptr[1:])
        h_loop = []
        for o in range(int(pack["n_objects"])):
            sl = slice(int(ptr[o]), int(ptr[o + 1]))
            rows_h = b["frame_inverse"][sl]
            h_loop.append((H[rows_h] * b["frame_weight"][sl].unsqueeze(-1)).sum(0))
        h_loop = torch.stack(h_loop) if h_loop else torch.zeros_like(h_scatter)
    scatter_diff = float((h_scatter - h_loop).abs().max().cpu())

    # --- gradient check on one real Train batch (the actual batch size)
    train_rows = cache.split_rows["train"]
    batch = int(cfg["recipe"]["batch"])
    chunk = train_rows[:batch]
    pack = cache.pack(chunk)
    b = _tensor_pack(pack, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["recipe"]["peak_lr"]),
                                  weight_decay=float(cfg["recipe"]["weight_decay"]))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    y = model.forward_packed(b)
    target = torch.as_tensor(cache.target_std(chunk), dtype=torch.float32, device=device)
    loss = torch.nn.functional.mse_loss(y, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    groups: dict[str, dict[str, float]] = {}
    finite = True
    nonzero = True
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        module = name.split(".")[0]
        norm = float(g.norm().cpu())
        entry = groups.setdefault(module, {"grad_norm_sq": 0.0, "grad_numel": 0, "grad_rms": 0.0})
        entry["grad_norm_sq"] += float((g * g).sum().cpu())
        entry["grad_numel"] += int(g.numel())
        finite = finite and bool(torch.isfinite(g).all().cpu())
        if module in {"atom_mlp", "bond_mlp", "frame_encoder", "graph_head"}:
            nonzero = nonzero and norm > 0.0
    for entry in groups.values():
        entry["grad_rms"] = float(np.sqrt(entry.pop("grad_norm_sq") / max(1, entry["grad_numel"])))
    optimizer.step()
    moved = float((model.frame_encoder[2].weight.detach() - before["frame_encoder.2.weight"]).abs().max().cpu())
    encoder_grad = float(model.frame_encoder[2].weight.grad.norm().cpu())

    return {
        "batch_order_max_abs_diff": batch_diff,
        "scatter_vs_loop_max_abs_diff": scatter_diff,
        "gradient": {
            "loss": float(loss.detach().cpu()),
            "finite": finite,
            "nonzero_required_modules": nonzero,
            "groups": groups,
            "encoder_second_layer_grad_norm": encoder_grad,
            "frame_encoder_update_max_abs": moved,
            "optimizer_updates_frame_encoder": bool(moved > 0.0),
        },
        "support_zeroed_atom_rows": int(sum(len(x["zeroed"]) for x in support["atom"])),
        "support_zeroed_bond_rows": int(sum(len(x["zeroed"]) for x in support["bond"])),
        "parameters": model.parameter_ledger(),
    }


def data_checks(cfg, payload, stats) -> dict[str, Any]:
    cache_s56 = ArmCache(payload, stats, C.ARM_S56)
    cache_w34 = ArmCache(payload, stats, C.ARM_W34)
    checks: dict[str, Any] = {}
    checks["n_molecules"] = int(cache_s56.n_molecules)
    checks["row_ids_are_arange"] = bool(np.array_equal(cache_s56.row_ids, np.arange(cache_s56.n_molecules)))
    checks["split_counts"] = {name: int(len(cache_s56.split_rows[name])) for name in C.SPLITS}
    checks["split_counts_expected"] = bool(all(len(cache_s56.split_rows[n]) == c
                                               for n, c in zip(C.SPLITS, (3360, 420, 420))))
    checks["labels_finite"] = bool(np.isfinite(cache_s56.labels).all())
    checks["scaler"] = cache_s56.scaler
    expected_scaler = C.target_scaler(payload["labels"], cache_s56.split_rows["train"])
    checks["scaler_matches_train_only"] = bool(
        abs(cache_s56.scaler["mean"] - expected_scaler["mean"]) < 1e-12
        and abs(cache_s56.scaler["std"] - expected_scaler["std"]) < 1e-12
    )
    checks["old_sentence_core_rows"] = int(stats["old_sentence_core_rows"])
    checks["old_w34_core_rows"] = int(stats["old_w34_core_rows"])
    checks["old_sentence_count_ok"] = bool(stats["old_sentence_core_rows"] == 3316287)
    checks["old_w34_count_ok"] = bool(stats["old_w34_core_rows"] == 1593359)
    checks["core_rows_equal_old_cache"] = bool(
        stats["arms"][C.ARM_S56]["core_rows"] == stats["old_sentence_core_rows"]
        and stats["arms"][C.ARM_W34]["core_rows"] == stats["old_w34_core_rows"]
        and stats["arms"][C.ARM_S56]["core_duplicates"] == 0
        and stats["arms"][C.ARM_W34]["core_duplicates"] == 0
    )
    checks["arms"] = {}
    for cache, arm in ((cache_s56, C.ARM_S56), (cache_w34, C.ARM_W34)):
        arm_stats = stats["arms"][arm]
        sizes = cache.obj_size.astype(np.int64)
        checks["arms"][arm] = {
            "n_objects": int(sizes.size),
            "n_core": int(cache.obj_core.sum()),
            "n_pad": int((~cache.obj_core).sum()),
            "core_size_values": sorted(set(int(x) for x in np.unique(sizes[cache.obj_core]))),
            "pad_size_values": sorted(set(int(x) for x in np.unique(sizes[~cache.obj_core]))),
            "pad_max_allowed": int(cfg["objects"]["arms"][arm]["pad_component_max"]),
            "root_in_set": bool(np.all(
                (((cache.obj_atoms == cache.obj_root[:, None])
                  & (np.arange(6)[None, :] < sizes[:, None])).any(axis=1))
            )),
            "padding_zero_beyond_size": bool(np.all(
                cache.obj_atoms[np.arange(6)[None, :] >= sizes[:, None]] == 0
            )),
            "bucket_consistent": bool(np.array_equal(
                cache.obj_bucket, (sizes > int(cfg["objects"]["arms"][arm]["bucket_split"])).astype(np.uint8)
            )),
            "frame_weight_sums_to_one_max_abs": float(np.max(np.abs(
                np.add.reduceat(cache.frame_weight.astype(np.float64), cache.obj_frame_ptr[:-1]) - 1.0
            ))) if sizes.size else 0.0,
            "frame_weight_sums_to_one": bool(np.max(np.abs(
                np.add.reduceat(cache.frame_weight.astype(np.float64), cache.obj_frame_ptr[:-1]) - 1.0
            )) <= 1e-6) if sizes.size else True,
            "all_atoms_covered": bool(arm_stats["atoms_uncovered"] == 0),
            "atoms_uncovered": int(arm_stats["atoms_uncovered"]),
            "frame_table_masks_ok": bool(
                np.array_equal(cache.frame_node_mask.astype(bool),
                               np.arange(6)[None, :] < cache.frame_k[:, None])
                and np.array_equal(cache.frame_edge_mask, cache.frame_adj)
                and np.array_equal(cache.frame_edge_mask.astype(bool),
                                   cache.frame_bond_attr != 65535)
            ),
            "frame_input_id_in_range": bool(cache.frame_input_id.max(initial=0) < cache.frame_k.size
                                           if cache.frame_k.size else True),
        }
        if arm == C.ARM_S56:
            checks["arms"][arm]["large_components_without_s56"] = int(arm_stats["large_components_without_s56"] or 0)
    checks["frame_input_width"] = 6 * 32 + 15 * 16 + 6 + 15 + 15 + 6
    checks["graph_input_width"] = 4 * 128 + 4
    return checks


def lut_checks() -> dict[str, Any]:
    t0 = time.perf_counter()
    luts, info = F.build_luts(6)
    records = []
    for k in range(1, 6):
        canon, winners = F.brute_force_lut(k)
        got = luts[k]["canon"]
        ok = np.array_equal(canon, got)
        wok = True
        for m in range(canon.size):
            exp = set(winners[m])
            got_w = set(tuple(x) for x in luts[k]["winner_perm"][luts[k]["ptr"][m]:luts[k]["ptr"][m + 1]].tolist())
            if exp != got_w:
                wok = False
                break
        records.append({"k": k, "canon_equal": bool(ok), "winners_equal": bool(wok)})
    # k=6: independent brute force on a fixed sample of 500 masks (all 120 perms)
    pairs6 = F._pairs(6)
    pi6 = {p: i for i, p in enumerate(pairs6)}
    perms6 = [(0,) + p for p in __import__("itertools").permutations(range(1, 6))]
    sample = sorted(set([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 4095, 8191, 12345,
                         16384, 20000, 24576, 30000, 32767]
                        + np.random.default_rng(0).integers(0, 32768, size=500).tolist()))
    ok6 = True
    w6 = True
    for m in sample:
        arr = np.asarray([m], np.uint32)
        vals = [int(F._permuted_bits(arr, p, pairs6, pi6)[0]) for p in perms6]
        best = min(vals)
        ok6 = ok6 and int(luts[6]["canon"][m]) == best
        exp = {p for p, v in zip(perms6, vals) if v == best}
        got_w = set(tuple(x) for x in luts[6]["winner_perm"][luts[6]["ptr"][m]:luts[6]["ptr"][m + 1]].tolist())
        w6 = w6 and exp == got_w
    return {
        "records": records,
        "k6_sampled_masks": len(sample),
        "k6_canon_equal": bool(ok6),
        "k6_sampled_winners_equal": bool(w6),
        "k6_sample_masks": sample[:20],
        "lut_build_seconds": float(time.perf_counter() - t0),
        "configured_lut_info": info,
        "all_pass": bool(ok6 and w6 and all(r["canon_equal"] and r["winners_equal"] for r in records)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=str(C.data_dir() / "dlo_objects_s0.npz"))
    ap.add_argument("--stats", default=str(C.data_dir() / "dlo_cache_stats.json"))
    ap.add_argument("--out", default=str(C.analysis_dir() / "checks.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-relabel", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    stats = C.read_json(args.stats)
    payload = load_payload(args.cache)
    t0 = time.perf_counter()
    checks: dict[str, Any] = {
        "cache_path": args.cache,
        "cache_sha256": stats.get("cache_sha256"),
        "data": data_checks(cfg, payload, stats),
        "lut": lut_checks(),
    }
    luts, _ = F.build_luts(6)
    checks["relabel"] = relabel_invariance(cfg, payload, stats, luts, args.device, args.gpu_relabel)
    cache = ArmCache(payload, stats, C.ARM_S56)
    checks["batch_and_gradient"] = batch_and_gradient(cfg, payload, stats, cache,
                                                      "cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    checks["elapsed_seconds"] = float(time.perf_counter() - t0)
    checks["all_pass"] = bool(
        checks["data"]["split_counts_expected"]
        and checks["data"]["scaler_matches_train_only"]
        and checks["data"]["old_sentence_count_ok"]
        and checks["data"]["old_w34_count_ok"]
        and checks["data"]["core_rows_equal_old_cache"]
        and checks["lut"]["all_pass"]
        and checks["relabel"]["max_cpu_fp64"] <= 1e-10
        and checks["batch_and_gradient"]["batch_order_max_abs_diff"] <= 1e-4
        and checks["batch_and_gradient"]["scatter_vs_loop_max_abs_diff"] <= 1e-4
        and checks["batch_and_gradient"]["gradient"]["finite"]
        and checks["batch_and_gradient"]["gradient"]["nonzero_required_modules"]
        and checks["batch_and_gradient"]["gradient"]["optimizer_updates_frame_encoder"]
        and all(v["all_atoms_covered"] and v["frame_table_masks_ok"] and v["bucket_consistent"]
                and v["root_in_set"] and v["padding_zero_beyond_size"] for v in checks["data"]["arms"].values())
    )
    C.write_json(args.out, checks)
    print(json.dumps({"all_pass": checks["all_pass"], "elapsed_seconds": checks["elapsed_seconds"]}, indent=2))
    if not checks["all_pass"]:
        raise SystemExit("checks failed")


if __name__ == "__main__":
    main()
