"""Build the two object corpora and one frame table per arm.

Objects are complete induced local subgraphs:
  DIRECT_S56 : every unique 5/6-atom ``(root, atom_set)`` from the old sentence
               cache, plus every complete connected component of size 1..4 with
               each of its atoms as root;
  MATCHED_W34: every unique 3/4-atom word from the old word cache, plus complete
               connected components of size 1..2.

Frames are the full attribute-labelled induced subgraph, one per root-fixed
permutation that realises the canonical unlabelled skeleton.  They are deduped
by the *complete* raw input (size + 6 atom attribute rows + 15 bond rows) and
weighted by ``1 / n_frames(object)``; objects themselves are never deduped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import dlo_common as C
import dlo_frames as F


def _load_npz_subset(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        missing = [k for k in keys if k not in handle.files]
        if missing:
            raise KeyError(f"{path} lacks {missing}")
        return {k: np.ascontiguousarray(handle[k]) for k in keys}


def _split_id(raw: dict[str, np.ndarray]) -> np.ndarray:
    split = np.full(raw["row_ids"].size, -1, np.int8)
    split[np.asarray(raw["train_idx"], np.int64)] = 0
    split[np.asarray(raw["valid_idx"], np.int64)] = 1
    split[np.asarray(raw["test_idx"], np.int64)] = 2
    if np.any(split < 0) or not np.array_equal(np.asarray(raw["row_ids"]), np.arange(raw["row_ids"].size)):
        raise ValueError("raw split is not a partition of row_id = arange(n)")
    return split


def _dedup_objects(rows: list[tuple[int, tuple[int, ...]]], what: str) -> tuple[list[tuple[int, tuple[int, ...]]], int]:
    seen: set[tuple[int, tuple[int, ...]]] = set()
    out: list[tuple[int, tuple[int, ...]]] = []
    for root, atoms in rows:
        key = (int(root), tuple(sorted(int(a) for a in atoms)))
        if key[0] not in key[1]:
            raise ValueError(f"{what}: root not in atom set")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out, len(rows) - len(out)


def _component_objects(comps: list[list[int]], max_size: int) -> list[tuple[int, tuple[int, ...]]]:
    out: list[tuple[int, tuple[int, ...]]] = []
    for comp in comps:
        if len(comp) <= max_size:
            block = tuple(int(x) for x in comp)
            for root in comp:
                out.append((int(root), block))
    return out


def _arm_objects(row: int, comps: list[list[int]],
                 sent: dict[str, np.ndarray] | None,
                 word: dict[str, np.ndarray] | None,
                 cfg_arm: dict[str, Any]) -> tuple[list[tuple[int, tuple[int, ...]]], dict[str, int], np.ndarray]:
    core_sizes = set(int(x) for x in cfg_arm["core_sizes"])
    pad_max = int(cfg_arm["pad_component_max"])
    info = {"core_rows": 0, "core_unique": 0, "core_duplicates": 0, "pad": 0}
    rows: list[tuple[int, tuple[int, ...]]] = []
    if sent is not None:
        s0, s1 = int(sent["ptr"][row]), int(sent["ptr"][row + 1])
        info["core_rows"] = s1 - s0
        for s in range(s0, s1):
            k = int(sent["n"][s])
            if k not in core_sizes:
                raise ValueError("unexpected sentence size")
            atoms = tuple(int(x) for x in sent["atoms"][s, :k])
            rows.append((int(sent["root"][s]), atoms))
    elif word is not None:
        w0, w1 = int(word["ptr"][row]), int(word["ptr"][row + 1])
        n = word["n"][w0:w1]
        sel = np.nonzero((n == 3) | (n == 4))[0]
        info["core_rows"] = int(sel.size)
        for j in sel:
            w = w0 + int(j)
            k = int(word["n"][w])
            atoms = tuple(int(x) for x in word["atoms"][w, :k])
            rows.append((int(word["root"][w]), atoms))
    else:
        raise ValueError("one core source is required")
    unique, dup = _dedup_objects(rows, "core")
    info["core_unique"] = len(unique)
    info["core_duplicates"] = int(dup)
    pads = _component_objects(comps, pad_max)
    info["pad"] = len(pads)
    merged = unique + pads
    merged.sort(key=lambda x: (len(x[1]), x[1], x[0]))
    core_set = set(unique)
    flags = np.asarray([key in core_set for key in merged], dtype=bool)
    if int(flags.sum()) != len(unique):
        raise AssertionError("core flag construction lost objects")
    return merged, info, flags


def _frame_keys(k: np.ndarray, atom_attr: np.ndarray, bond_attr: np.ndarray) -> np.ndarray:
    """Exact structured key of the complete frame input."""
    n = k.size
    key = np.empty(n, dtype=[("k", "u1"), ("a", "u2", (C.MAX_ATOMS,)), ("b", "u2", (len(C.EDGE_PAIRS),))])
    key["k"] = k
    key["a"] = atom_attr
    bond = bond_attr.astype(np.int64)
    if np.any(bond < -1) or np.any(bond > 65534):
        raise ValueError("bond attribute row out of range")
    key["b"] = np.where(bond < 0, 65535, bond).astype(np.uint16)
    return key


def build_arm(name: str, cfg_arm: dict[str, Any], raw: dict[str, np.ndarray],
              atom_code: np.ndarray, bond_code: np.ndarray,
              atom_inv: np.ndarray, bond_inv: np.ndarray,
              sent: dict[str, np.ndarray] | None, word: dict[str, np.ndarray] | None,
              luts: dict[int, dict[str, np.ndarray]], max_molecules: int | None,
              verbose: bool = True) -> dict[str, Any]:
    node_ptr = raw["node_ptr"]
    bond_ptr = raw["bond_ptr"]
    bu = raw["bond_u"]
    bv = raw["bond_v"]
    n_mol = raw["row_ids"].size if max_molecules is None else min(int(max_molecules), raw["row_ids"].size)
    split_id = raw["split_id"]
    c_arm = cfg_arm
    bucket_split = int(c_arm["bucket_split"])

    mol_obj_ptr = np.zeros(n_mol + 1, np.int64)
    obj_root_parts: list[np.ndarray] = []
    obj_atoms_parts: list[np.ndarray] = []
    obj_size_parts: list[np.ndarray] = []
    obj_bucket_parts: list[np.ndarray] = []
    obj_core_parts: list[np.ndarray] = []
    obj_mol_parts: list[np.ndarray] = []
    frame_count_parts: list[np.ndarray] = []
    frame_atom_parts: list[np.ndarray] = []
    frame_bond_parts: list[np.ndarray] = []
    frame_k_parts: list[np.ndarray] = []
    mol_n_bucket = np.zeros((n_mol, 2), np.int32)
    coverage = np.zeros(n_mol, np.int64)
    uncovered = np.zeros(n_mol, np.int64)
    core_rows = unique_rows = dup_rows = pad_rows = 0
    comp_hist = np.zeros(16, np.int64)
    big_comp_no_s56 = 0
    t0 = time.perf_counter()
    for row in range(n_mol):
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        adj: list[list[int]] = [[] for _ in range(n)]
        bond_row = np.full((n, n), -1, np.int32)
        for e in range(b0, b1):
            u = int(bu[e])
            v = int(bv[e])
            if not (0 <= u < n and 0 <= v < n) or u == v:
                raise ValueError("bond endpoint outside the molecule or self loop")
            adj[u].append(v)
            adj[v].append(u)
            bond_row[u, v] = bond_row[v, u] = int(bond_inv[e])
        atom_row = atom_inv[a0:a1]
        comps = F.connected_components(adj)
        for comp in comps:
            comp_hist[min(len(comp), comp_hist.size - 1)] += 1
        if name == C.ARM_S56:
            big = [c for c in comps if len(c) >= 5]
            if big and sent is not None:
                for comp in big:
                    cset = set(comp)
                    s0, s1 = int(sent["ptr"][row]), int(sent["ptr"][row + 1])
                    ok = False
                    for s in range(s0, s1):
                        k = int(sent["n"][s])
                        at = sent["atoms"][s, :k]
                        if int(sent["root"][s]) in cset and all(int(x) in cset for x in at):
                            ok = True
                            break
                    if not ok:
                        big_comp_no_s56 += 1
        objects, info, core_flags = _arm_objects(row, comps, sent, word, c_arm)
        core_rows += info["core_rows"]
        unique_rows += info["core_unique"]
        dup_rows += info["core_duplicates"]
        pad_rows += info["pad"]
        n_obj = len(objects)
        mol_obj_ptr[row + 1] = mol_obj_ptr[row] + n_obj
        roots = np.fromiter((r for r, _ in objects), np.int64, count=n_obj)
        sizes = np.fromiter((len(s) for _, s in objects), np.int64, count=n_obj)
        atoms = np.zeros((n_obj, C.MAX_ATOMS), np.uint8)
        for i, (_, s) in enumerate(objects):
            atoms[i, :len(s)] = s
        buckets = np.where(sizes <= bucket_split, 0, 1).astype(np.uint8)
        mol_n_bucket[row, 0] = int((buckets == 0).sum())
        mol_n_bucket[row, 1] = int((buckets == 1).sum())
        covered = np.zeros(n, bool)
        for _, s in objects:
            covered[list(s)] = True
        coverage[row] = int(covered.sum())
        uncovered[row] = int(n - coverage[row])
        frame = F.frames_for_objects(atom_row, bond_row, objects, luts, check=False)
        counts = frame["obj_frame_count"]
        if int(counts.sum()) != frame["frame_k"].size or np.any(counts < 1):
            raise AssertionError("frame expansion is inconsistent")
        obj_root_parts.append(roots.astype(np.uint8))
        obj_atoms_parts.append(atoms)
        obj_size_parts.append(sizes.astype(np.uint8))
        obj_bucket_parts.append(buckets)
        obj_core_parts.append(core_flags)
        obj_mol_parts.append(np.full(n_obj, row, np.int32))
        frame_count_parts.append(counts.astype(np.int64))
        frame_atom_parts.append(frame["frame_atom_attr"])
        frame_bond_parts.append(frame["frame_bond_attr"])
        frame_k_parts.append(frame["frame_k"].astype(np.uint8))
        if verbose and row and row % 1000 == 0:
            print(f"[{name}] {row}/{n_mol} molecules  {time.perf_counter()-t0:.1f}s", flush=True)

    obj_root = np.concatenate(obj_root_parts) if obj_root_parts else np.zeros(0, np.uint8)
    obj_atoms = np.concatenate(obj_atoms_parts) if obj_atoms_parts else np.zeros((0, C.MAX_ATOMS), np.uint8)
    obj_size = np.concatenate(obj_size_parts) if obj_size_parts else np.zeros(0, np.uint8)
    obj_bucket = np.concatenate(obj_bucket_parts) if obj_bucket_parts else np.zeros(0, np.uint8)
    obj_core = np.concatenate(obj_core_parts) if obj_core_parts else np.zeros(0, bool)
    obj_mol = np.concatenate(obj_mol_parts) if obj_mol_parts else np.zeros(0, np.int32)
    obj_frame_count = np.concatenate(frame_count_parts) if frame_count_parts else np.zeros(0, np.int64)
    frame_atom_attr = np.concatenate(frame_atom_parts) if frame_atom_parts else np.zeros((0, C.MAX_ATOMS), np.uint16)
    frame_bond_attr = np.concatenate(frame_bond_parts) if frame_bond_parts else np.zeros((0, len(C.EDGE_PAIRS)), np.int32)
    frame_k = np.concatenate(frame_k_parts) if frame_k_parts else np.zeros(0, np.uint8)
    obj_frame_ptr = np.zeros(obj_size.size + 1, np.int64)
    np.cumsum(obj_frame_count, out=obj_frame_ptr[1:])
    if int(obj_frame_ptr[-1]) != frame_k.size:
        raise AssertionError("object/frame pointer mismatch")

    t_frames = time.perf_counter()
    keys = _frame_keys(frame_k, frame_atom_attr, frame_bond_attr)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    t_unique = time.perf_counter()
    n_frames = int(frame_k.size)
    n_unique = int(unique_keys.size)
    frame_input_id = np.asarray(inverse, np.uint32)
    frame_weight = np.empty(n_frames, np.float32)
    per_object = 1.0 / obj_frame_count.astype(np.float64)
    frame_weight = np.repeat(per_object, obj_frame_count).astype(np.float32)
    wsum = np.zeros(obj_size.size, np.float64)
    np.add.at(wsum, np.repeat(np.arange(obj_size.size), obj_frame_count), per_object.repeat(obj_frame_count))
    if obj_size.size and not np.allclose(wsum, 1.0, atol=1e-12):
        raise AssertionError("object frame weights do not sum to one")
    uniq_atom = unique_keys["a"].astype(np.uint16)
    uniq_bond = unique_keys["b"].astype(np.uint16)
    uniq_k = unique_keys["k"].astype(np.uint8)
    node_mask = (np.arange(C.MAX_ATOMS)[None, :] < uniq_k[:, None]).astype(np.uint8)
    edge_mask = (uniq_bond != 65535).astype(np.uint8)
    adjacency = edge_mask.copy()
    # slots outside the true sise must never claim a bond
    pair_index = np.asarray(C.EDGE_PAIRS)
    inside = (pair_index[None, :, 0] < uniq_k[:, None]) & (pair_index[None, :, 1] < uniq_k[:, None])
    edge_mask = (edge_mask & inside).astype(np.uint8)
    adjacency = edge_mask.copy()
    if np.any(uniq_bond[edge_mask.astype(bool)] == 65535):
        raise AssertionError("true edge marked missing")
    stats = {
        "arm": name,
        "n_molecules": int(n_mol),
        "n_objects": int(obj_size.size),
        "n_core": int(obj_core.sum()),
        "n_pad": int((~obj_core).sum()),
        "core_rows": int(core_rows),
        "core_unique": int(unique_rows),
        "core_duplicates": int(dup_rows),
        "pad_objects": int(pad_rows),
        "n_frames": n_frames,
        "n_unique_frames": n_unique,
        "frame_dedup_ratio": float(1.0 - n_unique / max(1, n_frames)),
        "frames_per_object": float(n_frames / max(1, obj_size.size)),
        "unique_frames_per_object": float(n_unique / max(1, obj_size.size)),
        "object_size_counts": {str(k): int((obj_size == k).sum()) for k in range(1, C.MAX_ATOMS + 1)},
        "bucket_counts": [int((obj_bucket == 0).sum()), int((obj_bucket == 1).sum())],
        "mol_n_bucket_train_mean": [float(mol_n_bucket[split_id[:n_mol] == 0, b].mean()) for b in range(2)],
        "mol_n_bucket_train_max": [int(mol_n_bucket[split_id[:n_mol] == 0, b].max()) for b in range(2)],
        "mol_n_bucket_train_min": [int(mol_n_bucket[split_id[:n_mol] == 0, b].min()) for b in range(2)],
        "objects_per_split": {
            C.SPLIT_NAME[s]: int((split_id[:n_mol][obj_mol] == s).sum()) for s in range(3)
        }
        if obj_mol.size else {C.SPLIT_NAME[s]: 0 for s in range(3)},
        "molecules_with_objects": {
            C.SPLIT_NAME[s]: int((np.diff(mol_obj_ptr)[split_id[:n_mol] == s] > 0).sum()) for s in range(3)
        },
        "atoms_covered": int(coverage.sum()),
        "atoms_total": int((raw["node_ptr"][1:n_mol + 1] - raw["node_ptr"][:n_mol]).sum()),
        "atoms_uncovered": int(uncovered.sum()),
        "molecules_with_uncovered_atoms": int((uncovered > 0).sum()),
        "component_size_hist": {str(i): int(comp_hist[i]) for i in range(comp_hist.size)},
        "large_components_without_s56": int(big_comp_no_s56) if name == C.ARM_S56 else None,
        "timing_seconds": {
            "build_objects_frames": float(t_frames - t0),
            "frame_dedup": float(t_unique - t_frames),
            "total": float(time.perf_counter() - t0),
        },
        "obj_frame_weight_check": float(np.abs(wsum - 1.0).max()) if obj_size.size else 0.0,
    }
    arrays = {
        "mol_obj_ptr": mol_obj_ptr,
        "obj_root": obj_root,
        "obj_atoms": obj_atoms,
        "obj_size": obj_size,
        "obj_bucket": obj_bucket,
        "obj_core": obj_core,
        "obj_mol": obj_mol,
        "mol_n_bucket": mol_n_bucket,
        "obj_frame_ptr": obj_frame_ptr,
        "frame_input_id": frame_input_id,
        "frame_weight": frame_weight,
        "frame_k": uniq_k,
        "frame_atom_attr": uniq_atom,
        "frame_bond_attr": uniq_bond,
        "frame_node_mask": node_mask,
        "frame_edge_mask": edge_mask,
        "frame_adj": adjacency,
    }
    if verbose:
        print(
            f"[{name}] objects={obj_size.size} core={int(obj_core.sum())} pad={int((~obj_core).sum())} "
            f"frames={n_frames} unique={n_unique} dedup={stats['frame_dedup_ratio']:.4f} "
            f"uncovered_atoms={int(uncovered.sum())} large_comp_no_s56={stats['large_components_without_s56']} "
            f"{time.perf_counter()-t0:.1f}s",
            flush=True,
        )
    return {"arrays": arrays, "stats": stats, "mol_n_bucket": mol_n_bucket}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(C.data_dir() / "dlo_objects_s0.npz"))
    ap.add_argument("--stats", default=str(C.data_dir() / "dlo_cache_stats.json"))
    ap.add_argument("--max-molecules", type=int, default=None)
    args = ap.parse_args()
    cfg = C.load_config()
    raw_path = C.REPO_ROOT / cfg["raw_cache"]
    digest = C.sha256_file(raw_path)
    if digest != cfg["raw_cache_sha256"]:
        raise ValueError(f"raw cache sha256 mismatch: {digest}")
    t_all = time.perf_counter()
    with np.load(raw_path, allow_pickle=False) as handle:
        raw_keys = ["node_ptr", "bond_ptr", "atom_feat", "bond_feat", "bond_u", "bond_v",
                    "row_ids", "labels", "train_idx", "valid_idx", "test_idx",
                    "graph_n_atoms", "graph_n_bonds", "graph_n_components"]
        raw = {k: np.ascontiguousarray(handle[k]) for k in raw_keys}
    raw["split_id"] = _split_id(raw)
    atom_code, atom_inv = np.unique(raw["atom_feat"], axis=0, return_inverse=True)
    bond_code, bond_inv = np.unique(raw["bond_feat"], axis=0, return_inverse=True)
    atom_inv = np.ascontiguousarray(atom_inv.reshape(-1), dtype=np.int64)
    bond_inv = np.ascontiguousarray(bond_inv.reshape(-1), dtype=np.int64)

    old_base_path = C.REPO_ROOT / cfg["old_base_cache"]
    old_sent_path = C.REPO_ROOT / cfg["old_sent_cache"]
    for p, expected in ((old_base_path, cfg["old_base_cache_sha256"]), (old_sent_path, cfg["old_sent_cache_sha256"])):
        got = C.sha256_file(p)
        if got != expected:
            raise ValueError(f"old cache sha256 mismatch for {p}: {got}")
    base = _load_npz_subset(old_base_path, ["mol_word_ptr", "word_root", "word_n", "word_atoms", "split_id"])
    sent = _load_npz_subset(old_sent_path, ["mol_sent_ptr", "sent_root", "sent_n", "sent_atoms"])
    sent = {
        "ptr": sent["mol_sent_ptr"],
        "root": sent["sent_root"],
        "n": sent["sent_n"],
        "atoms": sent["sent_atoms"],
    }
    word = {
        "ptr": base["mol_word_ptr"],
        "root": base["word_root"],
        "n": base["word_n"],
        "atoms": base["word_atoms"],
    }
    if not np.array_equal(base["split_id"].astype(np.int8), raw["split_id"]):
        raise ValueError("old cache split ids disagree with raw cache")
    n_mol = raw["row_ids"].size if args.max_molecules is None else min(int(args.max_molecules), raw["row_ids"].size)
    old_sent_total = int(sent["ptr"][n_mol])
    w34_total = int(((word["n"][: int(word["ptr"][n_mol])] == 3) | (word["n"][: int(word["ptr"][n_mol])] == 4)).sum())
    if args.max_molecules is None:
        if old_sent_total != 3316287:
            raise AssertionError(f"old sentence count changed: {old_sent_total}")
        if w34_total != 1593359:
            raise AssertionError(f"old 3/4-word count changed: {w34_total}")

    luts, lut_info = F.build_luts(C.MAX_ATOMS)
    print(f"LUT built in {lut_info['total_seconds']:.3f}s", flush=True)
    payload: dict[str, np.ndarray] = {
        "row_ids": raw["row_ids"],
        "split_id": raw["split_id"],
        "labels": raw["labels"],
        "n_atoms": raw["graph_n_atoms"],
        "n_bonds": raw["graph_n_bonds"],
        "n_components": raw["graph_n_components"],
        "node_ptr": raw["node_ptr"],
        "bond_ptr": raw["bond_ptr"],
        "atom_feat": raw["atom_feat"],
        "bond_feat": raw["bond_feat"],
        "bond_u": raw["bond_u"],
        "bond_v": raw["bond_v"],
        "atom_code": atom_code,
        "bond_code": bond_code,
        "raw_cache_sha256": np.frombuffer(digest.encode(), np.uint8),
        "old_base_cache_sha256": np.frombuffer(cfg["old_base_cache_sha256"].encode(), np.uint8),
        "old_sent_cache_sha256": np.frombuffer(cfg["old_sent_cache_sha256"].encode(), np.uint8),
    }
    stats: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "raw_cache": str(raw_path),
        "raw_cache_sha256": digest,
        "old_base_cache_sha256": cfg["old_base_cache_sha256"],
        "old_sent_cache_sha256": cfg["old_sent_cache_sha256"],
        "n_molecules": int(n_mol),
        "n_atom_attr_rows": int(atom_code.shape[0]),
        "n_bond_attr_rows": int(bond_code.shape[0]),
        "old_sentence_core_rows": old_sent_total,
        "old_w34_core_rows": w34_total,
        "luts": lut_info,
        "split_counts": {C.SPLIT_NAME[s]: int((raw["split_id"][:n_mol] == s).sum()) for s in range(3)},
        "arms": {},
    }
    for name, cfg_arm, sent_arg, word_arg in (
        (C.ARM_S56, cfg["objects"]["arms"][C.ARM_S56], sent, None),
        (C.ARM_W34, cfg["objects"]["arms"][C.ARM_W34], None, word),
    ):
        arm = build_arm(name, cfg_arm, raw, atom_code, bond_code, atom_inv, bond_inv,
                        sent_arg, word_arg, luts, args.max_molecules)
        pref = C.ARM_PREFIX[name]
        for key, value in arm["arrays"].items():
            payload[f"{pref}_{key}"] = value
        stats["arms"][name] = arm["stats"]

    # Train-only support, scaler, and mu_b (max(1, Train mean)).
    train_rows = np.nonzero(raw["split_id"][:n_mol] == 0)[0]
    node_ptr = raw["node_ptr"]
    bond_ptr = raw["bond_ptr"]
    train_atom_rows = np.concatenate(
        [np.arange(int(node_ptr[r]), int(node_ptr[r + 1])) for r in train_rows]
    ) if train_rows.size else np.zeros(0, np.int64)
    train_bond_rows = np.concatenate(
        [np.arange(int(bond_ptr[r]), int(bond_ptr[r + 1])) for r in train_rows]
    ) if train_rows.size else np.zeros(0, np.int64)
    atom_support = []
    for j in range(C.ATOM_COLUMNS):
        vals = np.unique(raw["atom_feat"][train_atom_rows, j])
        atom_support.append([int(x) for x in vals])
    bond_support = []
    for j in range(C.BOND_COLUMNS):
        vals = np.unique(raw["bond_feat"][train_bond_rows, j])
        bond_support.append([int(x) for x in vals])
    scaler = C.target_scaler(raw["labels"][:n_mol], train_rows)
    stats["train_support"] = {"atom": atom_support, "bond": bond_support}
    stats["target_scaler"] = scaler
    for name in C.ARMS:
        arr = payload[f"{C.ARM_PREFIX[name]}_mol_n_bucket"]
        mean = arr[train_rows].mean(axis=0)
        stats["arms"][name]["mu_bucket"] = [float(max(1.0, x)) for x in mean]
        stats["arms"][name]["bucket_normalizer"] = [float(max(1.0, x)) for x in mean]
    stats["prepare_seconds"] = float(time.perf_counter() - t_all)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    stats["cache_path"] = str(out_path)
    stats["cache_sha256"] = C.sha256_file(out_path)
    stats["cache_bytes"] = int(out_path.stat().st_size)
    C.write_json(args.stats, stats)
    print(f"wrote {out_path} ({stats['cache_bytes']/2**20:.1f} MiB) sha256={stats['cache_sha256']}", flush=True)
    print(f"wrote {args.stats}", flush=True)


if __name__ == "__main__":
    main()
