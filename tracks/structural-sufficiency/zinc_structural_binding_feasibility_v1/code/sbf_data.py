"""Label-free binding cache and marginal-preserving permutation controls.

The native LCRC two-hop record for a root ``v`` and path ``v -> w -> u`` is

    mid_code = type(w) * 9 + (bt1 - 1) * 3 + (bt2 - 1)      (189 classes)
    end_type = type(u)                                      (21 classes)

The frozen LCRC ``path_end63`` also contains ``bt2``; ``bt2`` is already part of
``mid_code``.  A permutation of the raw 63-way end code would therefore break a
bit that is duplicated in both halves.  The primary control permutes only the
terminal atom type; ``bt2`` stays with its middle half.  This is the native
mid--end correspondence, not the duplicated bond-type bit.

Only structural fields are read.  Atom indices, molecule ids and path array
order are used for bookkeeping, never as model inputs.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

import sbf_common as C

L = C.L
LD = C.LD
AD = C.AD


# ---------------------------------------------------------------------------
# raw train split (labels poisoned before any field is touched)
# ---------------------------------------------------------------------------

def load_raw_train() -> list[dict[str, Any]]:
    """Official PyG ZINC subset=True train, raw x/edge_index/edge_attr only.

    ``d.y`` is replaced by ``_Poison`` before the structural fields are read;
    any accidental label access raises.
    """
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(L.C.ZINC_ROOT), subset=True, split="train")
    out: list[dict[str, Any]] = []
    for i in range(len(ds)):
        d = ds[i]
        d.y = C._Poison()
        out.append({
            "mol_index": int(i),
            "x": np.ascontiguousarray(d.x.reshape(-1).long().numpy().astype(np.int64)),
            "edge_index": np.ascontiguousarray(d.edge_index.numpy().astype(np.int64)),
            "edge_attr": np.ascontiguousarray(d.edge_attr.reshape(-1).long().numpy().astype(np.int64)),
        })
    return out


def graph_fingerprint(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> str:
    """Canonical typed Weisfeiler-Lehman fingerprint (5 rounds)."""
    g = nx.Graph()
    g.add_nodes_from((int(i), {"t": int(t)}) for i, t in enumerate(np.asarray(x).reshape(-1)))
    seen: set[tuple[int, int]] = set()
    ei = np.asarray(edge_index)
    ea = np.asarray(edge_attr).reshape(-1)
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        g.add_edge(a, b, bt=int(ea[k]))
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="t", edge_attr="bt", iterations=5, digest_size=16)


# ---------------------------------------------------------------------------
# frozen LCRC encoding + binding records
# ---------------------------------------------------------------------------

def _adjacency(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray):
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n_atoms)]
    for e, ((u, v), bt) in enumerate(zip(np.asarray(bond_index), np.asarray(bond_type))):
        u, v, bt = int(u), int(v), int(bt)
        adj[u].append((v, bt, e))
        adj[v].append((u, bt, e))
    for i in range(n_atoms):
        adj[i].sort()
    return adj


def build_binding_molecule(lc: Mapping[str, Any]) -> dict[str, Any]:
    """One molecule: frozen LCRC arrays -> per-root typed mid/end records.

    The path enumeration order is exactly LCRC's; every record is checked
    against ``path_cat``, ``path_end63``, ``path_end`` and ``path_closure``.
    """
    n = int(lc["n_atoms"])
    m = int(lc["n_bonds"])
    atom_type = np.asarray(lc["atom_type"], dtype=np.int64)
    atom_scalar = np.asarray(lc["atom_scalar"], dtype=np.float32)
    bond_index = np.asarray(lc["bond_index"], dtype=np.int64)
    bond_type = np.asarray(lc["bond_type"], dtype=np.int64)
    path_ptr = np.asarray(lc["path_ptr"], dtype=np.int64)
    path_cat = np.asarray(lc["path_cat"], dtype=np.int64)
    path_end = np.asarray(lc["path_end"], dtype=np.int64)
    path_end63 = np.asarray(lc["path_end63"], dtype=np.int64)
    path_closure = np.asarray(lc["path_closure"], dtype=np.int64)

    adj = _adjacency(n, bond_index, bond_type)
    degree = np.asarray([len(adj[i]) for i in range(n)], dtype=np.int64)

    # neighbour type counts per (atom, bond type), including every neighbour
    counts = [[np.zeros(C.N_ATOM_TYPES, dtype=np.int16) for _ in range(C.N_BOND_TYPES + 1)]
              for _ in range(n)]
    for (u, v), bt in zip(bond_index, bond_type):
        u, v, bt = int(u), int(v), int(bt)
        counts[u][bt][int(atom_type[v])] += 1
        counts[v][bt][int(atom_type[u])] += 1

    roots: list[dict[str, Any]] = []
    for v in range(n):
        s, e = int(path_ptr[v]), int(path_ptr[v + 1])
        if e == s:
            continue
        recs: list[dict[str, Any]] = []
        for w, bt1, _e1 in adj[v]:
            tv = int(atom_type[v])
            cw = counts[w]
            for u, bt2, _e2 in adj[w]:
                if u == v:
                    continue
                # allowed terminal *types* for this (v, w, bt2), excluding v
                allowed = 0
                cvec = cw[bt2]
                for t in range(C.N_ATOM_TYPES):
                    cnt = int(cvec[t])
                    if bt2 == bt1 and t == tv:
                        cnt -= 1
                    if cnt > 0:
                        allowed |= 1 << t
                mid = int(atom_type[w]) * 9 + (bt1 - 1) * 3 + (bt2 - 1)
                end_type = int(atom_type[u])
                if not ((allowed >> end_type) & 1):
                    raise RuntimeError("real path violates local type compatibility")
                recs.append({
                    "mid": mid,
                    "end": end_type,
                    "mid_type": int(atom_type[w]),
                    "mid_deg": int(degree[w]),
                    "mid_ring": int(atom_scalar[w, 4] > 0.5),
                    "closure": int(path_closure[s + len(recs)]),
                    "allowed": allowed,
                    "bt2": int(bt2),
                    "mid_atom": int(w),
                })
        if len(recs) != e - s:
            raise RuntimeError("path count mismatch")
        # exact frozen-record checks, in LCRC enumeration order
        for k, rec in enumerate(recs):
            if rec["mid"] != int(path_cat[s + k]):
                raise RuntimeError("mid code mismatch")
            if rec["end"] != int(path_end63[s + k]) // 3:
                raise RuntimeError("end type mismatch")
            if rec["end"] != int(atom_type[path_end[s + k]]):
                raise RuntimeError("end atom mismatch")
            if rec["closure"] != int(path_closure[s + k]):
                raise RuntimeError("closure mismatch")
        roots.append({
            "atom": v,
            "type": int(atom_type[v]),
            "deg": int(degree[v]),
            "ring": int(atom_scalar[v, 4] > 0.5),
            "mring": int(round(float(atom_scalar[v, 5]) * 8.0)),
            "paths": recs,
        })
    return {"n_atoms": n, "n_bonds": m, "roots": roots}


def build_binding_cache() -> dict[str, Any]:
    """Build the full train10k binding cache and the structural LCRC arrays."""
    t0 = time.perf_counter()
    raws = load_raw_train()
    C.log(f"[sbf data] raw train {len(raws)} molecules")
    import structural_relation_objects_v1 as sro

    mols: list[dict[str, Any]] = []
    lc_mols: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for i, raw in enumerate(raws):
        obj = sro.build_mol_objects(
            i, raw["x"], raw["edge_index"], raw["edge_attr"]
        )
        lc = LD.build_molecule_from_object(obj)
        b = build_binding_molecule(lc)
        mols.append(b)
        lc_mols.append(lc)
        fingerprints.append(graph_fingerprint(raw["x"], raw["edge_index"], raw["edge_attr"]))
        if (i + 1) % 2000 == 0:
            C.log(f"[sbf data] {i + 1}/{len(raws)} molecules {time.perf_counter() - t0:.1f}s")

    # frozen LCRC arrays, concatenated exactly as in lc_data._concat_molecules
    lc_all = LD._concat_molecules(lc_mols)

    root_mol: list[int] = []
    root_atom: list[int] = []
    root_type: list[int] = []
    root_deg: list[int] = []
    root_ring: list[int] = []
    root_mring: list[int] = []
    path_mid: list[int] = []
    path_end: list[int] = []
    path_mid_type: list[int] = []
    path_mid_deg: list[int] = []
    path_mid_ring: list[int] = []
    path_closure: list[int] = []
    path_allowed: list[int] = []
    path_bt2: list[int] = []
    path_mid_atom: list[int] = []
    path_root: list[int] = []
    root_ptr: list[int] = [0]
    for mi, mol in enumerate(mols):
        for r in mol["roots"]:
            ri = len(root_type)
            root_mol.append(mi)
            root_atom.append(int(r["atom"]))
            root_type.append(int(r["type"]))
            root_deg.append(int(r["deg"]))
            root_ring.append(int(r["ring"]))
            root_mring.append(int(r["mring"]))
            for p in r["paths"]:
                path_mid.append(int(p["mid"]))
                path_end.append(int(p["end"]))
                path_mid_type.append(int(p["mid_type"]))
                path_mid_deg.append(int(p["mid_deg"]))
                path_mid_ring.append(int(p["mid_ring"]))
                path_closure.append(int(p["closure"]))
                path_allowed.append(int(p["allowed"]))
                path_bt2.append(int(p["bt2"]))
                path_mid_atom.append(int(p["mid_atom"]))
                path_root.append(ri)
            root_ptr.append(len(path_mid))

    cache = {
        "root_ptr": np.asarray(root_ptr, dtype=np.int64),
        "root_mol": np.asarray(root_mol, dtype=np.int32),
        "root_atom": np.asarray(root_atom, dtype=np.int32),
        "root_type": np.asarray(root_type, dtype=np.int8),
        "root_deg": np.asarray(root_deg, dtype=np.int8),
        "root_ring": np.asarray(root_ring, dtype=np.int8),
        "root_mring": np.asarray(root_mring, dtype=np.int8),
        "path_mid": np.asarray(path_mid, dtype=np.uint8),
        "path_end": np.asarray(path_end, dtype=np.uint8),
        "path_mid_type": np.asarray(path_mid_type, dtype=np.uint8),
        "path_mid_deg": np.asarray(path_mid_deg, dtype=np.uint8),
        "path_mid_ring": np.asarray(path_mid_ring, dtype=np.uint8),
        "path_closure": np.asarray(path_closure, dtype=np.uint8),
        "path_allowed": np.asarray(path_allowed, dtype=np.uint32),
        "path_bt2": np.asarray(path_bt2, dtype=np.uint8),
        "path_mid_atom": np.asarray(path_mid_atom, dtype=np.int32),
        "path_root": np.asarray(path_root, dtype=np.int32),
        "mol_fingerprint": np.asarray(fingerprints, dtype=object),
        "n_molecules": len(mols),
        "lc_arrays": lc_all,
        "meta": {
            "round": C.ROUND_ID,
            "feature_source": "raw PyG train x/edge_index/edge_attr -> frozen P3 object -> frozen LCRC lc_data.build_molecule_from_object",
            "labels_accessed": False,
            "official_test_loaded": False,
            "feature_version": L.FEATURE_VERSION,
            "n_molecules": len(mols),
            "n_atoms": int(lc_all["atom_type"].shape[0]),
            "n_bonds": int(lc_all["bond_type"].shape[0]),
            "n_paths": int(len(path_mid)),
            "n_roots": int(len(root_type)),
            "max_paths_observed": int(max(np.diff(np.asarray(root_ptr)))),
            "created": C.now(),
            "seconds": float(time.perf_counter() - t0),
        },
    }
    cache["meta"]["content_sha256"] = binding_content_sha(cache)
    return cache


def binding_content_sha(cache: Mapping[str, Any]) -> str:
    keys = [
        "root_ptr", "root_mol", "root_type", "root_deg", "root_ring", "root_mring",
        "path_mid", "path_end", "path_mid_type", "path_mid_deg", "path_mid_ring",
        "path_closure", "path_allowed", "path_bt2", "path_root",
    ]
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode())
        h.update(C.sha256_array(np.asarray(cache[k])).encode())
    h.update(str(cache["n_molecules"]).encode())
    return h.hexdigest()


def save_cache(cache: Mapping[str, Any], path=None) -> str:
    path = path or (C.CACHE / "binding_train10k.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), path)
    return C.sha256_file(path)


def load_cache(path=None) -> dict:
    path = path or (C.CACHE / "binding_train10k.pt")
    if not path.exists():
        raise FileNotFoundError(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache["meta"].get("round") != C.ROUND_ID:
        raise RuntimeError("unexpected cache round")
    if cache["meta"].get("content_sha256") != binding_content_sha(cache):
        raise RuntimeError("binding cache content hash mismatch")
    return cache


# ---------------------------------------------------------------------------
# root records / fakes
# ---------------------------------------------------------------------------

def root_slice(cache: Mapping[str, Any], ri: int) -> dict[str, np.ndarray]:
    s, e = int(cache["root_ptr"][ri]), int(cache["root_ptr"][ri + 1])
    return {
        "mid": np.asarray(cache["path_mid"][s:e], dtype=np.int64),
        "end": np.asarray(cache["path_end"][s:e], dtype=np.int64),
        "mid_type": np.asarray(cache["path_mid_type"][s:e], dtype=np.int64),
        "mid_deg": np.asarray(cache["path_mid_deg"][s:e], dtype=np.int64),
        "mid_ring": np.asarray(cache["path_mid_ring"][s:e], dtype=np.int64),
        "closure": np.asarray(cache["path_closure"][s:e], dtype=np.int64),
        "allowed": np.asarray(cache["path_allowed"][s:e], dtype=np.int64),
        "bt2": np.asarray(cache["path_bt2"][s:e], dtype=np.int64),
        "root_type": int(cache["root_type"][ri]),
        "root_deg": int(cache["root_deg"][ri]),
        "root_ring": int(cache["root_ring"][ri]),
        "root_mring": int(cache["root_mring"][ri]),
        "n": int(e - s),
    }


def canonical_signature(r: Mapping[str, Any]) -> tuple:
    rows = sorted(
        (int(r["mid"][i]), int(r["end"][i]), int(r["closure"][i]),
         int(r["mid_type"][i]), int(r["mid_deg"][i]), int(r["mid_ring"][i]),
         int(r["allowed"][i]), int(r["bt2"][i]))
        for i in range(r["n"])
    )
    return (int(r["root_type"]), int(r["root_deg"]), int(r["root_ring"]),
            int(r["root_mring"]), tuple(rows))


def _rng_for(sig: tuple, salt: str) -> np.random.Generator:
    h = hashlib.sha256((repr(sig) + "|" + salt).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def make_standard_fake(r: Mapping[str, Any]) -> np.ndarray | None:
    """Uniform marginal-preserving end-type permutation, or None if degenerate."""
    n = int(r["n"])
    if n < 2:
        return None
    mids = np.asarray(r["mid"], dtype=np.int64)
    ends = np.asarray(r["end"], dtype=np.int64)
    if len(set(mids.tolist())) < 2 or len(set(ends.tolist())) < 2:
        return None
    real_key = tuple(sorted(zip(mids.tolist(), ends.tolist())))
    rng = _rng_for(canonical_signature(r), C.SALT_STANDARD)
    for _ in range(200):
        perm = rng.permutation(n)
        new = ends[perm]
        if tuple(sorted(zip(mids.tolist(), new.tolist()))) != real_key:
            return new
    return None


def make_hard_fake(r: Mapping[str, Any]) -> np.ndarray | None:
    """Compatibility-preserving 2-swap with the same end-type marginal.

    A swap i<->j is accepted iff mid_i != mid_j, end_i != end_j, and each new
    end type is an allowed terminal type for its middle half.  The resulting
    record has exactly the same per-path local type compatibility as the real
    one, so the simple compatibility rule is constant and cannot decide.
    """
    n = int(r["n"])
    if n < 2:
        return None
    mids = np.asarray(r["mid"], dtype=np.int64)
    ends = np.asarray(r["end"], dtype=np.int64)
    allowed = np.asarray(r["allowed"], dtype=np.int64)
    rows = sorted(
        range(n),
        key=lambda i: (int(mids[i]), int(ends[i]), int(r["closure"][i]),
                       int(r["mid_type"][i]), int(r["mid_deg"][i]),
                       int(r["mid_ring"][i]), int(allowed[i])),
    )
    valid: list[tuple[int, int]] = []
    for a in range(n):
        i = rows[a]
        for b in range(a + 1, n):
            j = rows[b]
            if int(mids[i]) == int(mids[j]) or int(ends[i]) == int(ends[j]):
                continue
            if not ((int(allowed[i]) >> int(ends[j])) & 1):
                continue
            if not ((int(allowed[j]) >> int(ends[i])) & 1):
                continue
            valid.append((i, j))
    if not valid:
        return None
    # uniform over the available compatibility-preserving 2-swaps; the seed is
    # a canonical function of the root's path multiset, not of row order.
    rng = _rng_for(canonical_signature(r), C.SALT_HARD)
    i, j = valid[int(rng.integers(0, len(valid)))]
    new = ends.copy()
    new[i], new[j] = new[j], new[i]
    return new


def fake_stats(r: Mapping[str, Any], fake: np.ndarray) -> dict[str, Any]:
    mids = np.asarray(r["mid"], dtype=np.int64)
    ends = np.asarray(r["end"], dtype=np.int64)
    allowed = np.asarray(r["allowed"], dtype=np.int64)
    new = np.asarray(fake, dtype=np.int64)
    real_key = tuple(sorted(zip(mids.tolist(), ends.tolist())))
    fake_key = tuple(sorted(zip(mids.tolist(), new.tolist())))
    changed = int(np.sum(ends != new))
    compat_real = float(np.mean([(int(allowed[i]) >> int(ends[i])) & 1 for i in range(len(ends))]))
    compat_fake = float(np.mean([(int(allowed[i]) >> int(new[i])) & 1 for i in range(len(new))]))
    return {
        "changed_paths": changed,
        "joint_changed": bool(fake_key != real_key),
        "compat_real": compat_real,
        "compat_fake": compat_fake,
        "fully_compatible_fake": bool(compat_fake >= 1.0 - 1e-12),
    }


def context_key(r: Mapping[str, Any], i: int) -> tuple:
    return (
        int(r["mid"][i]), int(r["end"][i]), int(r["root_type"]), int(r["root_ring"]),
        int(r["root_mring"]), int(r["mid_type"][i]), int(r["mid_deg"][i]),
        int(r["mid_ring"][i]), int(r["closure"][i]),
    )


def root_keys(r: Mapping[str, Any]) -> set[tuple]:
    return {context_key(r, i) for i in range(int(r["n"]))}


# ---------------------------------------------------------------------------
# molecule split
# ---------------------------------------------------------------------------

def assign_folds(cache: Mapping[str, Any]) -> dict[str, Any]:
    fps = list(cache["mol_fingerprint"])
    fold = np.empty(len(fps), dtype=np.int8)
    for i, fp in enumerate(fps):
        h = hashlib.sha256((str(fp) + "|" + C.SALT_SPLIT).encode()).digest()
        fold[i] = int.from_bytes(h[:4], "little") % C.N_FOLDS
    # duplicate structure never crosses folds
    by_fp: dict[str, set[int]] = defaultdict(set)
    for fp, f in zip(fps, fold):
        by_fp[str(fp)].add(int(f))
    duplicates_crossing = sum(1 for v in by_fp.values() if len(v) > 1)
    return {
        "fold": fold,
        "fold_sizes": [int(np.sum(fold == f)) for f in range(C.N_FOLDS)],
        "unique_fingerprints": int(len(by_fp)),
        "duplicate_fingerprints": int(sum(1 for v in Counter(fps).values() if v > 1)),
        "duplicate_groups_crossing_folds": int(duplicates_crossing),
    }


def split_indices(fold: np.ndarray, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = np.where(fold == test_fold)[0]
    dev = np.where(fold == (test_fold + 1) % C.N_FOLDS)[0]
    train = np.where(~np.isin(fold, [test_fold, (test_fold + 1) % C.N_FOLDS]))[0]
    return train, dev, test


__all__ = [
    "load_raw_train", "graph_fingerprint", "build_binding_molecule", "build_binding_cache",
    "binding_content_sha", "save_cache", "load_cache", "root_slice", "canonical_signature",
    "make_standard_fake", "make_hard_fake", "fake_stats", "context_key", "root_keys",
    "assign_folds", "split_indices",
]
