"""Stage 0: multi-granularity rooted structure library, REAL/NULL relations, basis.

Writes ``data/hsrf_stage0.npz`` plus ``data/stage0_meta.json`` and the coverage
report.  Everything scientific is Train-only; Valid/Test structures are only
encoded against the frozen Train vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

import hrf_common as C

SIZES = (1, 2, 3, 4)
GRANS = ("TOPO", "ELEMENT", "COARSE", "FULL")
SPLITS = ("train", "valid", "test")


def _log(message: str) -> None:
    print(f"[stage0 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sig_int32(sig: np.ndarray) -> np.ndarray:
    if sig.size == 0:
        return np.zeros((0, 10), np.int32)
    return np.ascontiguousarray(sig.astype(np.int32))


def _unique_rows_int32(sig: np.ndarray) -> np.ndarray:
    if sig.shape[0] == 0:
        return np.zeros((0, 10), np.int32)
    return np.unique(sig, axis=0)


def _row_keys(rows: np.ndarray) -> list[bytes]:
    return [np.ascontiguousarray(r, np.int32).tobytes() for r in rows]


def enumerate_signatures(payload: Mapping[str, Any], atom_dims: Sequence[int],
                         bond_dims: Sequence[int], out_path: Path) -> dict[str, Any]:
    """Pass 1: enumerate every rooted support and cache four signatures."""
    n_mol = int(np.asarray(payload["row_ids"]).size)
    ptr = {size: np.zeros(n_mol + 1, np.int64) for size in SIZES}
    roots = {size: [] for size in SIZES}
    flat: dict[tuple[str, int], list[np.ndarray]] = {(g, s): [] for g in GRANS for s in SIZES}
    named = np.zeros((n_mol, 4), np.int64)
    t0 = time.time()
    for row in range(n_mol):
        st = C.build_struct(payload, row)
        named[row] = [st.n_atoms, st.n_bonds,
                      int(payload["graph_n_components"][row]),
                      int(payload["graph_cycle_rank"][row])]
        for size in SIZES:
            if size == 1:
                n_inst = st.n_atoms
                if n_inst:
                    roots[size].append(np.arange(n_inst, dtype=np.int32))
            else:
                n_inst = int(st.atoms[size].shape[0])
                if n_inst:
                    roots[size].append(np.asarray(st.atoms[size][:, 0], np.int32))
            ptr[size][row + 1] = ptr[size][row] + n_inst
            for gran in GRANS:
                sig = C.signatures_for(st, size, gran, atom_dims, bond_dims)
                flat[(gran, size)].append(_sig_int32(sig))
        if (row + 1) % 500 == 0:
            _log(f"enumerated {row + 1}/{n_mol} molecules  ({time.time() - t0:.1f}s)")
    payload_npz: dict[str, np.ndarray] = {"named_counts": named}
    for size in SIZES:
        payload_npz[f"ptr_{size}"] = ptr[size]
        payload_npz[f"root_{size}"] = (np.concatenate(roots[size])
                                       if roots[size] else np.zeros(0, np.int32))
    n_instances: dict[str, int] = {}
    for gran in GRANS:
        for size in SIZES:
            arr = (np.concatenate(flat[(gran, size)], axis=0) if flat[(gran, size)]
                   else np.zeros((0, 10), np.int32))
            payload_npz[f"sig_{gran}_{size}"] = arr
            n_instances[f"{gran}_{size}"] = int(arr.shape[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload_npz)
    meta = {
        "n_molecules": n_mol,
        "instances": n_instances,
        "ptr": {str(s): int(ptr[s][-1]) for s in SIZES},
        "enumeration_seconds": float(time.time() - t0),
        "file": str(out_path),
        "sha256": C.sha256_file(out_path),
        "signature_dtype": "int32 (10 fields, root-marked canonical certificate)",
    }
    _log(f"signature cache written: {out_path} ({meta['sha256'][:12]}…)")
    return meta


def _instance_arrays(cache: Mapping[str, np.ndarray], gran: str, size: int) -> np.ndarray:
    return np.asarray(cache[f"sig_{gran}_{size}"], np.int32)


def build_vocabulary(cache: Mapping[str, np.ndarray], rows_train: np.ndarray,
                     n_mol: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Train-only document frequency, constant-column drop, cap, type ids."""
    nm = cfg["structure"]["min_train_molecules"]
    cap = int(cfg["structure"]["vocab_cap"])
    train_mask = np.zeros(n_mol, bool)
    train_mask[np.asarray(rows_train, np.int64)] = True
    vocab: dict[str, Any] = {}
    global_offset = 0
    for size in SIZES:
        ptr = np.asarray(cache[f"ptr_{size}"], np.int64)
        mol_of_inst = np.repeat(np.arange(n_mol), np.diff(ptr))
        for gran in GRANS:
            sig = _instance_arrays(cache, gran, size)
            keep_inst = train_mask[mol_of_inst]
            sig_train = sig[keep_inst]
            mol_train = mol_of_inst[keep_inst]
            # document frequency: one presence per (molecule, type)
            df: dict[bytes, int] = {}
            counts_by_key: dict[bytes, dict[int, int]] = defaultdict(dict)
            for m in np.unique(mol_train).tolist():
                sel = mol_train == m
                uniq, cnt = np.unique(sig_train[sel], axis=0, return_counts=True)
                for r, c in zip(uniq, cnt):
                    key = np.ascontiguousarray(r, np.int32).tobytes()
                    df[key] = df.get(key, 0) + 1
                    counts_by_key[key][int(m)] = int(c)
            candidates = [k for k, v in df.items() if v >= int(nm)]
            # constant count columns: DF == n_train and every molecule carries the
            # same count (a DF < n_train column necessarily contains zeros)
            nonconstant = []
            constant_keys = []
            for key in candidates:
                if df[key] < int(rows_train.size):
                    nonconstant.append(key)
                    continue
                vals = np.asarray(list(counts_by_key[key].values()), np.int64)
                if vals.size == int(rows_train.size) and np.all(vals == vals[0]):
                    constant_keys.append(key)
                else:
                    nonconstant.append(key)
            nonconstant.sort(key=lambda k: (-df[k], k))
            if gran == "TOPO":
                kept = nonconstant
            else:
                kept = nonconstant[:cap]
            types = []
            mapping: dict[bytes, int] = {}
            for rank, key in enumerate(kept):
                sig_arr = np.frombuffer(key, np.int32)
                local = int(len(types))
                mapping[key] = local
                types.append({
                    "local_id": local,
                    "global_id": int(global_offset + local),
                    "signature": [int(x) for x in sig_arr],
                    "key": key.hex(),
                    "train_df": int(df[key]),
                    "train_count_total": int(sum(counts_by_key[key].values())),
                    "rank": int(rank),
                })
            vocab[f"{gran}_{size}"] = {
                "gran": gran, "size": int(size),
                "df_all": df,
                "n_candidates": int(len(candidates)),
                "n_constant_dropped": int(len(constant_keys)),
                "n_kept": int(len(kept)),
                "capped": bool(gran != "TOPO" and len(nonconstant) > cap),
                "types": types,
                "mapping": mapping,
                "global_offset": int(global_offset),
                "df_quantiles_all": _quantiles(list(df.values())),
                "df_quantiles_kept": _quantiles([df[k] for k in kept]),
            }
            global_offset += len(types)
            _log(f"vocab {gran}_{size}: candidates={len(candidates)} "
                 f"constant={len(constant_keys)} kept={len(kept)}")
    return {"groups": vocab, "n_base_structure_types": int(global_offset)}


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not len(values):
        return {"n": 0}
    arr = np.asarray(values, np.float64)
    qs = np.quantile(arr, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return {"n": int(arr.size), "min": float(qs[0]), "p10": float(qs[1]),
            "p25": float(qs[2]), "p50": float(qs[3]), "p75": float(qs[4]),
            "p90": float(qs[5]), "p99": float(qs[6]), "max": float(qs[7]),
            "mean": float(arr.mean())}


def build_base_counts(payload: Mapping[str, Any], cache: Mapping[str, np.ndarray],
                      vocab: Mapping[str, Any], rows_by_split: Mapping[str, np.ndarray],
                      n_mol: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Dense X_BASE structure counts, per-split coverage and support classes."""
    groups = vocab["groups"]
    p = int(vocab["n_base_structure_types"])
    min_df = int(cfg["structure"]["min_train_molecules"])
    counts = np.zeros((n_mol, p), np.int32)
    per_mol_types = np.zeros(n_mol, np.int32)
    per_mol_roots = {size: np.zeros((n_mol, len(GRANS)), np.int32) for size in SIZES}
    per_mol_total = {size: np.zeros((n_mol, len(GRANS)), np.int64) for size in SIZES}
    per_mol_kept = {size: np.zeros((n_mol, len(GRANS)), np.int64) for size in SIZES}
    status_by_group: dict[tuple[str, int], np.ndarray] = {}
    support_total = {split: 0 for split in SPLITS}
    any_kept_instances = {split: 0 for split in SPLITS}
    full_kept_instances = {split: 0 for split in SPLITS}
    all_unseen_instances = {split: 0 for split in SPLITS}
    status_counts = {split: np.zeros(4, np.int64) for split in SPLITS}
    full_uncovered = {split: [0, 0] for split in SPLITS}
    per_group = {}

    for size in SIZES:
        ptr = np.asarray(cache[f"ptr_{size}"], np.int64)
        root = np.asarray(cache[f"root_{size}"], np.int64)
        mol_of_inst = np.repeat(np.arange(n_mol), np.diff(ptr))
        inst_split = np.empty(mol_of_inst.size, dtype="<U5")
        for split in SPLITS:
            inst_split[np.isin(mol_of_inst, np.asarray(rows_by_split[split], np.int64))] = split
        for split in SPLITS:
            support_total[split] += int(np.count_nonzero(inst_split == split))
        for gi, gran in enumerate(GRANS):
            sig = _instance_arrays(cache, gran, size)
            group = groups[f"{gran}_{size}"]
            mapping = group["mapping"]
            df_all = group["df_all"]
            offset = int(group["global_offset"])
            status = np.full(sig.shape[0], 3, np.uint8)  # 0 retained 1 low 2 capped 3 unseen
            ids = np.full(sig.shape[0], -1, np.int64)
            for i, row in enumerate(sig):
                key = np.ascontiguousarray(row, np.int32).tobytes()
                lid = mapping.get(key)
                if lid is not None:
                    status[i] = 0
                    ids[i] = offset + lid
                else:
                    dv = df_all.get(key)
                    if dv is None:
                        status[i] = 3
                    elif dv < int(min_df):
                        status[i] = 1
                    else:
                        status[i] = 2
            status_by_group[(gran, size)] = status
            kept = status == 0
            sel = np.nonzero(kept)[0]
            if sel.size:
                pairs, cnt = np.unique(np.stack([mol_of_inst[sel], ids[sel]], axis=1),
                                       axis=0, return_counts=True)
                counts[pairs[:, 0], pairs[:, 1]] += cnt.astype(np.int32)
                per_mol_types += np.bincount(pairs[:, 0], minlength=n_mol).astype(np.int32)
            per_mol_kept[size][:, gi] = np.bincount(mol_of_inst[kept], minlength=n_mol)
            per_mol_total[size][:, gi] = np.bincount(mol_of_inst, minlength=n_mol)
            rep_pairs = np.unique(np.stack([mol_of_inst[kept], root[kept]], axis=1), axis=0)
            per_mol_roots[size][:, gi] = np.bincount(rep_pairs[:, 0], minlength=n_mol)
            for split in SPLITS:
                mask = inst_split == split
                status_counts[split] += np.bincount(status[mask], minlength=4).astype(np.int64)
            if gran == "FULL":
                other = np.zeros(sig.shape[0], bool)
                for ogran in ("COARSE", "ELEMENT", "TOPO"):
                    other |= status_by_group[(ogran, size)] == 0
                for split in SPLITS:
                    mask = (inst_split == split) & (~kept)
                    full_uncovered[split][1] += int(mask.sum())
                    full_uncovered[split][0] += int((mask & other).sum())
            rec = {
                "size": int(size), "gran": gran,
                "n_types_observed_train": int(len(df_all)),
                "n_types_qualified_df8": int(group["n_candidates"]),
                "n_types_in_vocab": int(group["n_kept"]),
                "n_constant_dropped": int(group["n_constant_dropped"]),
                "capped": bool(group["capped"]),
                "df_quantiles_all": group["df_quantiles_all"],
                "df_quantiles_kept": group["df_quantiles_kept"],
                "per_split": {},
            }
            for split in SPLITS:
                mask = inst_split == split
                total = int(mask.sum())
                st = np.bincount(status[mask], minlength=4)
                rec["per_split"][split] = {
                    "instances_total": total,
                    "instances_retained": int(st[0]),
                    "retained_instance_fraction": float(st[0] / max(1, total)),
                    "low_support_instances": int(st[1]),
                    "low_support_instance_fraction": float(st[1] / max(1, total)),
                    "capped_instances": int(st[2]),
                    "capped_instance_fraction": float(st[2] / max(1, total)),
                    "unseen_instances": int(st[3]),
                    "unseen_instance_fraction": float(st[3] / max(1, total)),
                    "roots_total": int(np.asarray(payload["graph_n_atoms"], np.int64)[
                        np.asarray(rows_by_split[split], np.int64)].sum()),
                    "roots_with_retained": int(per_mol_roots[size][
                        np.asarray(rows_by_split[split], np.int64), gi].sum()),
                }
            per_group[f"{gran}_{size}"] = rec
        # overall (across granularities) view for this size
        any_kept = np.zeros(mol_of_inst.size, bool)
        unseen_all = np.ones(mol_of_inst.size, bool)
        for gi, gran in enumerate(GRANS):
            any_kept |= status_by_group[(gran, size)] == 0
            unseen_all &= status_by_group[(gran, size)] == 3
        full_kept = status_by_group[("FULL", size)] == 0
        for split in SPLITS:
            mask = inst_split == split
            support_total[split] += 0
            any_kept_instances[split] += int((mask & any_kept).sum())
            full_kept_instances[split] += int((mask & full_kept).sum())
            all_unseen_instances[split] += int((mask & unseen_all).sum())

    # overall root coverage: a root is represented if any granularity/size is retained
    root_pairs = []
    any_size_kept = {}
    for size in SIZES:
        ptr = np.asarray(cache[f"ptr_{size}"], np.int64)
        root = np.asarray(cache[f"root_{size}"], np.int64)
        mol_of_inst = np.repeat(np.arange(n_mol), np.diff(ptr))
        any_kept = np.zeros(mol_of_inst.size, bool)
        for gi, gran in enumerate(GRANS):
            any_kept |= status_by_group[(gran, size)] == 0
        any_size_kept[size] = any_kept
        if any_kept.any():
            root_pairs.append(np.unique(np.stack([mol_of_inst[any_kept], root[any_kept]],
                                                 axis=1), axis=0))
    per_mol_root_any = np.zeros(n_mol, np.int64)
    if root_pairs:
        all_pairs = np.unique(np.concatenate(root_pairs, axis=0), axis=0)
        per_mol_root_any = np.bincount(all_pairs[:, 0], minlength=n_mol)

    coverage_rows = []
    for split in SPLITS:
        rows = np.asarray(rows_by_split[split], np.int64)
        eff = per_mol_types[rows]
        support = int(support_total[split])
        roots_total = int(np.asarray(payload["graph_n_atoms"], np.int64)[rows].sum())
        coverage_rows.append({
            "split": split,
            "molecules": int(rows.size),
            "support_instances": support,
            "any_granularity_retained_instances": int(any_kept_instances[split]),
            "any_granularity_retained_fraction": float(any_kept_instances[split] /
                                                       max(1, support)),
            "full_retained_instances": int(full_kept_instances[split]),
            "full_retained_fraction": float(full_kept_instances[split] / max(1, support)),
            "all_granularities_unseen_instances": int(all_unseen_instances[split]),
            "all_granularities_unseen_fraction": float(all_unseen_instances[split] /
                                                       max(1, support)),
            "roots_total": roots_total,
            "roots_with_any_retained_type": int(per_mol_root_any[rows].sum()),
            "root_coverage": float(per_mol_root_any[rows].sum() / max(1, roots_total)),
            "molecules_with_any_retained_type": int(np.count_nonzero(eff)),
            "molecule_coverage": float(np.count_nonzero(eff) / max(1, rows.size)),
            "effective_features_per_molecule_p10": float(np.quantile(eff, 0.10)),
            "effective_features_per_molecule_p50": float(np.quantile(eff, 0.50)),
            "effective_features_per_molecule_p90": float(np.quantile(eff, 0.90)),
            "effective_features_per_molecule_max": int(eff.max()) if eff.size else 0,
            "full_uncovered_but_coarser_covered": int(full_uncovered[split][0]),
            "full_uncovered_total": int(full_uncovered[split][1]),
            "full_uncovered_but_coarser_covered_fraction": float(
                full_uncovered[split][0] / max(1, full_uncovered[split][1])),
        })
    return {"counts": counts, "per_mol_types": per_mol_types,
            "per_mol_roots": per_mol_roots,
            "per_mol_total": per_mol_total, "per_mol_kept": per_mol_kept,
            "coverage_rows": coverage_rows, "per_group": per_group,
            "status_counts": {k: [int(x) for x in v] for k, v in status_counts.items()}}


def select_anchors(vocab: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    top = int(cfg["relations"]["anchor_top_per_size"])
    anchors: list[dict[str, Any]] = []
    for size in cfg["relations"]["anchor_sizes"]:
        group = vocab["groups"][f"{cfg['relations']['anchor_gran']}_{size}"]
        for rec in group["types"][:top]:
            anchors.append({
                "anchor_id": int(len(anchors)),
                "gran": cfg["relations"]["anchor_gran"],
                "size": int(size),
                "local_type_id": int(rec["local_id"]),
                "global_type_id": int(rec["global_id"]),
                "signature": [int(x) for x in rec["signature"]],
                "key": rec["key"],
                "train_df": int(rec["train_df"]),
                "rank_in_size": int(rec["rank"]),
            })
    return anchors


def build_relations(payload: Mapping[str, Any], cache: Mapping[str, np.ndarray],
                    vocab: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]],
                    rows_by_split: Mapping[str, np.ndarray], n_mol: int,
                    cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Two passes over molecules: Train DF selection, then retained counts/hits."""
    ptr_map = {s: np.asarray(cache[f"ptr_{s}"], np.int64) for s in SIZES}
    # lookup tables per anchor size
    lookups: dict[int, dict[bytes, int]] = {}
    local_to_anchor: dict[int, dict[int, int]] = {}
    for size in (2, 3):
        group = vocab["groups"][f"COARSE_{size}"]
        mapping = group["mapping"]
        lut = {}
        l2a = {}
        for a in anchors:
            if int(a["size"]) != size:
                continue
            key = bytes.fromhex(a["key"])
            lut[key] = int(a["anchor_id"])
            l2a[int(a["local_type_id"])] = int(a["anchor_id"])
        lookups[size] = lut
        local_to_anchor[size] = l2a

    def presence_for(row: int, n_atoms: int):
        present = np.zeros((n_atoms, len(anchors)), np.uint8)
        inst = np.full((n_atoms, len(anchors)), -1, np.int64)
        for size in (2, 3):
            ptr = ptr_map[size]
            lo, hi = int(ptr[row]), int(ptr[row + 1])
            sig = _instance_arrays(cache, "COARSE", size)[lo:hi]
            root = np.asarray(cache[f"root_{size}"], np.int64)[lo:hi]
            lut = lookups[size]
            l2a = local_to_anchor[size]
            # local type ids are not recomputed here; map signatures directly
            for i, s in enumerate(sig):
                key = np.ascontiguousarray(s, np.int32).tobytes()
                # anchor lookup uses the full signature -> anchor id
                aid = lut.get(key)
                if aid is None:
                    continue
                # the anchor record's local id is not needed beyond this point
                present[int(root[i]), aid] = 1
                if inst[int(root[i]), aid] < 0:
                    inst[int(root[i]), aid] = lo + i
        return present, inst

    # pass A: Train relation DF
    train_rows = np.asarray(rows_by_split["train"], np.int64)
    df: dict[bytes, int] = {}
    for row in train_rows.tolist():
        n_atoms = int(ptr_map[1][row + 1] - ptr_map[1][row])
        present, _ = presence_for(row, n_atoms)
        rel = _relation_counts_for_molecule(payload, row, present)
        for key in rel:
            df[key] = df.get(key, 0) + 1
    candidates = [(k, v) for k, v in df.items() if v >= int(cfg["relations"]["min_train_molecules"])]
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    cap = int(cfg["relations"]["relation_cap"])
    kept = candidates[:cap]
    rel_meta = []
    rel_lookup: dict[bytes, int] = {}
    for rank, (key, dv) in enumerate(kept):
        t, s, b = _decode_rel_key(key)
        rel_lookup[key] = rank
        rel_meta.append({
            "relation_id": int(rank), "t": int(t), "s": int(s), "bond_type": int(b),
            "train_df": int(dv), "rank": int(rank),
            "canonical_key": key.hex(),
            "t_anchor": dict(anchors[int(t)]),
            "s_anchor": dict(anchors[int(s)]),
        })
    _log(f"relations: {len(df)} observed keys, {len(candidates)} qualified, kept {len(kept)}")

    # pass B: retained counts, marginals, hits
    n_rel = len(rel_meta)
    n_bond = int(cfg["relations"]["max_bond_types"])
    rel_counts = np.zeros((n_mol, max(1, n_rel)), np.int32)
    anchor_marginal = np.zeros((n_mol, len(anchors), n_bond), np.int32)
    bond_slots = np.zeros((n_mol, n_bond), np.int32)
    hit_rows: dict[str, list[int]] = {
        "rel_id": [], "mol": [], "bond": [], "u": [], "v": [],
        "inst_t": [], "inst_s": [], "anchor_t": [], "anchor_s": [],
    }
    for row in range(n_mol):
        n_atoms = int(ptr_map[1][row + 1] - ptr_map[1][row])
        present, inst = presence_for(row, n_atoms)
        bu = np.asarray(payload["bond_u"][int(payload["bond_ptr"][row]):
                                          int(payload["bond_ptr"][row + 1])], np.int64)
        bv = np.asarray(payload["bond_v"][int(payload["bond_ptr"][row]):
                                          int(payload["bond_ptr"][row + 1])], np.int64)
        bt = np.asarray(payload["bond_feat"][int(payload["bond_ptr"][row]):
                                             int(payload["bond_ptr"][row + 1]), 0], np.int64)
        rel_row: dict[int, int] = defaultdict(int)
        row_marg = np.zeros((len(anchors), n_bond), np.int32)
        row_slots = np.zeros(n_bond, np.int32)
        for bid in range(bu.size):
            b = int(bt[bid])
            row_slots[b] += 2
            for (u, v) in ((int(bu[bid]), int(bv[bid])),
                           (int(bv[bid]), int(bu[bid]))):
                src = np.nonzero(present[u])[0]
                dst = np.nonzero(present[v])[0]
                if src.size:
                    row_marg[src, b] += 1
                for t in src.tolist():
                    for s in dst.tolist():
                        if t > s:
                            continue
                        t2, s2 = t, s
                        key = _encode_rel_key(t2, s2, b)
                        rid = rel_lookup.get(key)
                        if rid is None:
                            continue
                        rel_row[rid] += 1
                        hit_rows["rel_id"].append(rid)
                        hit_rows["mol"].append(row)
                        hit_rows["bond"].append(bid)
                        hit_rows["u"].append(u)
                        hit_rows["v"].append(v)
                        hit_rows["inst_t"].append(int(inst[u, t]))
                        hit_rows["inst_s"].append(int(inst[v, s]))
                        hit_rows["anchor_t"].append(int(t))
                        hit_rows["anchor_s"].append(int(s))
        for rid, val in rel_row.items():
            rel_counts[row, rid] = val
        anchor_marginal[row] = row_marg
        bond_slots[row] = row_slots
        if (row + 1) % 500 == 0:
            _log(f"relations pass B {row + 1}/{n_mol}")
    # marginals for kept relation anchors only
    used_anchor = sorted({int(m["t"]) for m in rel_meta} | {int(m["s"]) for m in rel_meta})
    used_bond = sorted({int(m["bond_type"]) for m in rel_meta})
    return {"rel_counts": rel_counts, "rel_meta": rel_meta,
            "anchor_marginal": anchor_marginal, "bond_slots": bond_slots,
            "used_anchor": used_anchor, "used_bond": used_bond,
            "hit_rows": hit_rows,
            "n_observed_keys": int(len(df)), "n_candidates": int(len(candidates)),
            "n_kept": int(n_rel), "df_quantiles_all": _quantiles(list(df.values())),
            "df_quantiles_kept": _quantiles([m["train_df"] for m in rel_meta])}


def _encode_rel_key(t: int, s: int, b: int) -> bytes:
    return np.asarray([t, s, b], np.int32).tobytes()


def _decode_rel_key(key: bytes) -> tuple[int, int, int]:
    arr = np.frombuffer(key, np.int32)
    return int(arr[0]), int(arr[1]), int(arr[2])


def _relation_counts_for_molecule(payload: Mapping[str, Any], row: int,
                                  present: np.ndarray) -> dict[bytes, int]:
    bu = np.asarray(payload["bond_u"][int(payload["bond_ptr"][row]):
                                      int(payload["bond_ptr"][row + 1])], np.int64)
    bv = np.asarray(payload["bond_v"][int(payload["bond_ptr"][row]):
                                      int(payload["bond_ptr"][row + 1])], np.int64)
    bt = np.asarray(payload["bond_feat"][int(payload["bond_ptr"][row]):
                                         int(payload["bond_ptr"][row + 1]), 0], np.int64)
    out: dict[bytes, int] = defaultdict(int)
    for bid in range(bu.size):
        b = int(bt[bid])
        for (u, v) in ((int(bu[bid]), int(bv[bid])), (int(bv[bid]), int(bu[bid]))):
            for t in np.nonzero(present[u])[0].tolist():
                for s in np.nonzero(present[v])[0].tolist():
                    if t > s:
                        continue
                    out[_encode_rel_key(t, s, b)] += 1
    return out


def build_basis(values: np.ndarray, rows_train: np.ndarray,
                cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed hinge basis with Train-only knots, scales and constant checks."""
    values = np.asarray(values, np.float64)
    n, p = values.shape
    train = np.asarray(rows_train, np.int64)
    floor = float(cfg["basis"]["scale_floor"])
    min_side = int(cfg["basis"]["min_side_samples"])
    basis_cols: list[np.ndarray] = []
    raw_cols: list[int] = []
    knots: list[int] = []
    scales: list[float] = []
    kept_raw: list[int] = []
    raw_info: list[dict[str, Any]] = []
    for j in range(p):
        x = values[:, j]
        xt = x[train]
        if np.all(xt == xt[0]):
            raw_info.append({"raw_col": j, "kept": False, "reason": "constant_train"})
            continue
        pos = xt[xt > 0]
        cand = {1}
        if pos.size:
            cand.add(int(round(float(np.median(pos)))))
            cand.add(int(round(float(np.quantile(pos, 0.9)))))
        cand = sorted(k for k in cand if k >= 1)
        kept_knots = []
        for k in cand:
            h = np.maximum(x - k, 0.0)
            below = int(np.count_nonzero(xt <= k))
            above = int(np.count_nonzero(xt > k))
            if (below >= min_side and above >= min_side
                    and np.any(h[train] != h[train][0])):
                kept_knots.append(int(k))
        # h0 always
        cols = [x] + [np.maximum(x - k, 0.0) for k in kept_knots]
        idxs = []
        for c, k in zip(cols, [0] + kept_knots):
            scale = max(float(np.sqrt(np.mean(c[train] ** 2))), floor)
            basis_cols.append(c.astype(np.float64))
            raw_cols.append(j)
            knots.append(int(k))
            scales.append(scale)
            idxs.append(len(scales) - 1)
        kept_raw.append(j)
        raw_info.append({"raw_col": j, "kept": True,
                         "positive_train": int(pos.size),
                         "positive_median": float(np.median(pos)) if pos.size else 0.0,
                         "positive_p90": float(np.quantile(pos, 0.9)) if pos.size else 0.0,
                         "knots": [0] + kept_knots,
                         "basis_indices": idxs})
    if basis_cols:
        basis = np.stack(basis_cols, axis=1)
    else:
        basis = np.zeros((n, 0), np.float64)
    scales_arr = np.asarray(scales, np.float64)
    return {"basis": basis.astype(np.float64), "raw_col": np.asarray(raw_cols, np.int32),
            "knot": np.asarray(knots, np.int32), "scale": scales_arr,
            "raw_info": raw_info, "kept_raw": np.asarray(kept_raw, np.int32),
            "lin_index": np.asarray([i for i, k in enumerate(knots) if k == 0], np.int64),
            "basis_name": [f"c{j}::k{k}" for j, k in zip(raw_cols, knots)]}


def null_expectations(rel_counts: np.ndarray, rel_meta: Sequence[Mapping[str, Any]],
                      anchor_marginal: np.ndarray, bond_slots: np.ndarray,
                      knots_by_rel: Sequence[Sequence[int]]) -> np.ndarray:
    """Exact E[h(K)] under K ~ Hypergeometric(M_b, B_sb, A_tb), one col per REAL basis."""
    n, r = rel_counts.shape
    out = np.zeros((n, sum(len(k) for k in knots_by_rel)), np.float64)
    col = 0
    for j, meta in enumerate(rel_meta):
        t = int(meta["t"]); s = int(meta["s"]); b = int(meta["bond_type"])
        A = anchor_marginal[:, t, b].astype(np.int64)
        B = anchor_marginal[:, s, b].astype(np.int64)
        M = bond_slots[:, b].astype(np.int64)
        ok = M > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ek = np.where(ok, (A * B) / np.maximum(M, 1), 0.0)
        knots = knots_by_rel[j]
        out[:, col] = ek
        col += 1
        max_support = np.minimum(A, B)
        for k in knots:
            if k <= 0:
                continue
            total = np.zeros(n, np.float64)
            upper = int(max_support.max()) if max_support.size else 0
            for jj in range(int(k) + 1, upper + 1):
                mask = max_support >= jj
                if not mask.any():
                    break
                total[mask] += stats.hypergeom.sf(jj - 1, M[mask], B[mask], A[mask])
            out[:, col] = np.where(ok, total, 0.0)
            col += 1
    return out


def hierarchy_map(cache: Mapping[str, np.ndarray], vocab: Mapping[str, Any],
                  n_mol: int) -> dict[str, Any]:
    """FULL -> COARSE -> ELEMENT -> TOPO signature map for retained types."""
    out: dict[str, Any] = {}
    for size in SIZES:
        ptr = np.asarray(cache[f"ptr_{size}"], np.int64)
        mol_of_inst = np.repeat(np.arange(n_mol), np.diff(ptr))
        sigs = {g: _instance_arrays(cache, g, size) for g in GRANS}
        for src_gran in ("FULL", "COARSE", "ELEMENT"):
            src_group = vocab["groups"][f"{src_gran}_{size}"]
            src_map = src_group["mapping"]
            # first instance of every retained source type
            first: dict[bytes, int] = {}
            for i, s in enumerate(sigs[src_gran]):
                key = np.ascontiguousarray(s, np.int32).tobytes()
                if key in src_map and key not in first:
                    first[key] = i
            entries = []
            for key, i in sorted(first.items(), key=lambda kv: src_map[kv[0]]):
                rec = {"source_key": key.hex(), "source_type": int(src_map[key])}
                for dst_gran in GRANS:
                    if dst_gran == src_gran:
                        continue
                    if GRANS.index(dst_gran) > GRANS.index(src_gran):
                        continue
                    dkey = np.ascontiguousarray(sigs[dst_gran][i], np.int32).tobytes()
                    dst_group = vocab["groups"][f"{dst_gran}_{size}"]
                    rec[f"{dst_gran}_key"] = dkey.hex()
                    rec[f"{dst_gran}_retained"] = bool(dkey in dst_group["mapping"])
                entries.append(rec)
            out[f"{src_gran}_{size}"] = entries
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(C.DATA_DIR / "hsrf_stage0.npz"))
    ap.add_argument("--meta-out", type=str, default=str(C.DATA_DIR / "stage0_meta.json"))
    ap.add_argument("--coverage-out", type=str, default=str(C.ANALYSIS_DIR / "coverage.json"))
    ap.add_argument("--sig-cache", type=str, default=str(C.DATA_DIR / "signatures.npz"))
    ap.add_argument("--reuse-signatures", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    C.assert_source_pins()
    payload = C.load_payload()
    atom_dims, bond_dims = C.atom_bond_dims()
    n_mol = int(np.asarray(payload["row_ids"]).size)
    rows_by_split = {s: np.asarray(payload["split_rows"][s], np.int64) for s in SPLITS}
    t_start = time.time()

    sig_path = Path(args.sig_cache)
    if args.reuse_signatures and sig_path.is_file():
        cache = dict(np.load(sig_path, allow_pickle=False))
        enum_meta = {"reused": True, "file": str(sig_path),
                     "sha256": C.sha256_file(sig_path)}
        _log(f"reused signature cache {sig_path}")
    else:
        enum_meta = enumerate_signatures(payload, atom_dims, bond_dims, sig_path)
        cache = dict(np.load(sig_path, allow_pickle=False))

    vocab = build_vocabulary(cache, rows_by_split["train"], n_mol, cfg)
    base = build_base_counts(payload, cache, vocab, rows_by_split, n_mol, cfg)
    anchors = select_anchors(vocab, cfg)
    rel = build_relations(payload, cache, vocab, anchors, rows_by_split, n_mol, cfg)

    # named + marginal columns appended to X_BASE
    named = np.asarray(cache["named_counts"], np.int64)
    used_bond = [int(b) for b in rel["used_bond"]]
    used_anchor = [int(a) for a in rel["used_anchor"]]
    marg_cols = []
    marg_names = []
    for b in used_bond:
        marg_cols.append(rel["bond_slots"][:, b].astype(np.int64))
        marg_names.append({"kind": "bond_slot_count", "bond_type": b, "anchor": None})
    for a in used_anchor:
        for b in used_bond:
            # only include marginals actually touched by a retained relation
            touched = any((int(m["t"]) == a or int(m["s"]) == a) and int(m["bond_type"]) == b
                          for m in rel["rel_meta"])
            if not touched:
                continue
            marg_cols.append(rel["anchor_marginal"][:, a, b].astype(np.int64))
            marg_names.append({"kind": "anchor_marginal", "bond_type": b, "anchor": a})
    named_names = ["n_atoms", "n_bonds", "n_components", "cycle_rank"]
    extra = np.stack(marg_cols, axis=1) if marg_cols else np.zeros((n_mol, 0), np.int64)
    base_raw = np.concatenate([base["counts"].astype(np.int64), named, extra], axis=1)
    _log(f"X_BASE raw: {base_raw.shape} (structure {base['counts'].shape[1]}, "
         f"named 4, marginals {extra.shape[1]})")

    # raw column catalogue
    base_meta = []
    gid = 0
    for size in SIZES:
        for gran in GRANS:
            group = vocab["groups"][f"{gran}_{size}"]
            for rec in group["types"]:
                base_meta.append({
                    "raw_col": int(gid), "kind": "rooted_structure",
                    "size": int(size), "gran": gran,
                    "type_local": int(rec["local_id"]),
                    "type_global": int(rec["global_id"]),
                    "signature": [int(x) for x in rec["signature"]],
                    "train_df": int(rec["train_df"]),
                    "key": rec["key"],
                    "name": f"{gran}_{size}_t{rec['local_id']}",
                })
                gid += 1
    for i, name in enumerate(named_names):
        base_meta.append({"raw_col": int(gid), "kind": "named_graph_count",
                          "name": name})
        gid += 1
    for rec in marg_names:
        base_meta.append({"raw_col": int(gid), "kind": rec["kind"],
                          "bond_type": rec["bond_type"], "anchor": rec["anchor"],
                          "name": f"{rec['kind']}_b{rec['bond_type']}"
                                  + (f"_a{rec['anchor']}" if rec["anchor"] is not None else "")})
        gid += 1
    assert gid == base_raw.shape[1]

    base_basis = build_basis(base_raw, rows_by_split["train"], cfg)
    rel_counts64 = rel["rel_counts"].astype(np.int64)
    rel_basis = build_basis(rel_counts64, rows_by_split["train"], cfg)

    knots_by_rel = []
    for info in rel_basis["raw_info"]:
        if info.get("kept", False):
            knots_by_rel.append([int(k) for k in info["knots"]])
    null_basis = null_expectations(rel["rel_counts"], rel["rel_meta"],
                                   rel["anchor_marginal"], rel["bond_slots"],
                                   knots_by_rel)
    null_scale = np.maximum(np.sqrt(np.mean(null_basis[rows_by_split["train"]] ** 2, axis=0)),
                            float(cfg["basis"]["scale_floor"]))
    _log(f"basis: base {base_basis['basis'].shape}, relation {rel_basis['basis'].shape}, "
         f"null {null_basis.shape}")

    # relation diagnostics
    rel_diag = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        real = rel["rel_counts"][rows].astype(np.float64)
        ek = null_basis[rows][:, np.asarray(rel_basis["lin_index"])] if rel_basis["lin_index"].size else real
        nz_real = np.count_nonzero(real, axis=1)
        nz_diff = np.count_nonzero(np.abs(real - ek) > 1e-12, axis=1)
        rel_diag[split] = {
            "molecules": int(rows.size),
            "molecules_with_any_relation": int(np.count_nonzero(nz_real)),
            "fraction_with_any_relation": float(np.count_nonzero(nz_real) / rows.size),
            "molecules_real_ne_null_linear": int(np.count_nonzero(nz_diff)),
            "fraction_real_ne_null_linear": float(np.count_nonzero(nz_diff) / rows.size),
            "median_nonzero_relations": float(np.median(nz_real)),
            "mean_nonzero_relations": float(np.mean(nz_real)),
            "relation_count_total": int(real.sum()),
            "relation_columns_nonzero": int(np.count_nonzero(real.sum(axis=0))),
        }

    out = {
        "named_counts": named.astype(np.int32),
        "base_counts": base_raw.astype(np.float64),
        "base_basis": base_basis["basis"],
        "base_basis_raw_col": base_basis["raw_col"],
        "base_basis_knot": base_basis["knot"],
        "base_scale": base_basis["scale"],
        "base_lin_index": base_basis["lin_index"],
        "rel_counts": rel["rel_counts"].astype(np.int32),
        "rel_basis": rel_basis["basis"],
        "rel_basis_raw_col": rel_basis["raw_col"],
        "rel_basis_knot": rel_basis["knot"],
        "rel_scale": rel_basis["scale"],
        "rel_lin_index": rel_basis["lin_index"],
        "null_basis": null_basis.astype(np.float64),
        "null_scale_self": null_scale,
        "anchor_marginal": rel["anchor_marginal"].astype(np.int32),
        "bond_slots": rel["bond_slots"].astype(np.int32),
        "rows_train": rows_by_split["train"],
        "rows_valid": rows_by_split["valid"],
        "rows_test": rows_by_split["test"],
        "labels": np.asarray(payload["labels"], np.float64),
        "row_ids": np.asarray(payload["row_ids"], np.int64),
        "hit_rel_id": np.asarray(rel["hit_rows"]["rel_id"], np.int32),
        "hit_mol": np.asarray(rel["hit_rows"]["mol"], np.int32),
        "hit_bond": np.asarray(rel["hit_rows"]["bond"], np.int32),
        "hit_u": np.asarray(rel["hit_rows"]["u"], np.int32),
        "hit_v": np.asarray(rel["hit_rows"]["v"], np.int32),
        "hit_inst_t": np.asarray(rel["hit_rows"]["inst_t"], np.int32),
        "hit_inst_s": np.asarray(rel["hit_rows"]["inst_s"], np.int32),
        "hit_anchor_t": np.asarray(rel["hit_rows"]["anchor_t"], np.int32),
        "hit_anchor_s": np.asarray(rel["hit_rows"]["anchor_s"], np.int32),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    _log(f"stage0 written: {out_path} ({C.sha256_file(out_path)[:12]}…)")

    catalog = {
        "base_raw_columns": base_meta,
        "base_basis": {
            "n": int(base_basis["basis"].shape[1]),
            "columns": [
                {"col": i, "raw_col": int(base_basis["raw_col"][i]),
                 "knot": int(base_basis["knot"][i]),
                 "scale": float(base_basis["scale"][i]),
                 "name": base_basis["basis_name"][i]}
                for i in range(base_basis["basis"].shape[1])
            ],
            "raw_info": base_basis["raw_info"],
        },
        "relation_columns": rel["rel_meta"],
        "relation_basis": {
            "n": int(rel_basis["basis"].shape[1]),
            "columns": [
                {"col": i, "raw_col": int(rel_basis["raw_col"][i]),
                 "knot": int(rel_basis["knot"][i]),
                 "scale": float(rel_basis["scale"][i]),
                 "name": rel_basis["basis_name"][i]}
                for i in range(rel_basis["basis"].shape[1])
            ],
            "raw_info": rel_basis["raw_info"],
            "null_scale_self": [float(x) for x in null_scale],
        },
        "anchors": [dict(a) for a in anchors],
        "vocabulary_summary": {
            k: {"size": v["size"], "gran": v["gran"], "n_candidates": v["n_candidates"],
                "n_constant_dropped": v["n_constant_dropped"], "n_kept": v["n_kept"],
                "capped": v["capped"], "df_quantiles_kept": v["df_quantiles_kept"]}
            for k, v in vocab["groups"].items()
        },
        "hierarchy_map": hierarchy_map(cache, vocab, n_mol),
        "data": {
            "cache_sha256": C.CACHE_SHA,
            "label_sha256": payload["label_sha256"],
            "split_row_sha256": {s: C.array_sha256(rows_by_split[s]) for s in SPLITS},
            "split_sizes": {s: int(rows_by_split[s].size) for s in SPLITS},
        },
        "enumeration": enum_meta,
    }
    C.write_json(C.DATA_DIR / "feature_catalog.json", catalog)

    coverage = {
        "round": cfg["round_id"],
        "source_pins": C.assert_source_pins(),
        "cache_sha256": C.CACHE_SHA,
        "label_sha256": payload["label_sha256"],
        "split_row_sha256": {s: C.array_sha256(rows_by_split[s]) for s in SPLITS},
        "enumeration": enum_meta,
        "n_base_structure_types": int(vocab["n_base_structure_types"]),
        "n_base_raw_columns": int(base_raw.shape[1]),
        "n_base_named_columns": 4,
        "n_base_marginal_columns": int(extra.shape[1]),
        "n_base_basis_columns": int(base_basis["basis"].shape[1]),
        "n_relation_types": int(len(rel["rel_meta"])),
        "n_relation_observed_keys": int(rel["n_observed_keys"]),
        "n_relation_candidates": int(rel["n_candidates"]),
        "n_relation_basis_columns": int(rel_basis["basis"].shape[1]),
        "relation_df_quantiles_all": rel["df_quantiles_all"],
        "relation_df_quantiles_kept": rel["df_quantiles_kept"],
        "relation_per_split": rel_diag,
        "per_group": base["per_group"],
        "coverage_rows": base["coverage_rows"],
        "anchor_count": int(len(anchors)),
        "anchor_used_count": int(len(used_anchor)),
    }
    C.write_json(Path(args.coverage_out), coverage)
    meta = {
        "round": cfg["round_id"],
        "commit": C.git_commit(),
        "created_at": C.now_utc(),
        "stage0_file": str(out_path),
        "stage0_sha256": C.sha256_file(out_path),
        "signature_cache": str(sig_path),
        "signature_cache_sha256": C.sha256_file(sig_path),
        "enumeration_seconds": enum_meta.get("enumeration_seconds"),
        "total_seconds": float(time.time() - t_start),
        "n_base_raw_columns": int(base_raw.shape[1]),
        "n_base_basis_columns": int(base_basis["basis"].shape[1]),
        "n_relation_types": int(len(rel["rel_meta"])),
        "n_relation_basis_columns": int(rel_basis["basis"].shape[1]),
        "n_hits": int(out["hit_rel_id"].size),
        "source_pins": C.assert_source_pins(),
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
    }
    C.write_json(Path(args.meta_out), meta)
    _log(f"done in {time.time() - t_start:.1f}s; hits={out['hit_rel_id'].size}")


if __name__ == "__main__":
    main()