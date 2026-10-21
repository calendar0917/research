"""Rooted induced-path enumeration and batching for the induced-pattern pilot.

Objects
-------
For every molecule and every root atom ``v`` we enumerate **all simple paths**
starting at ``v`` with 2, 3, 4 or 5 edges (3-6 atoms).  A path is an ordered
sequence of distinct atom ids; the root is fixed at position 0.  Reversing a
path therefore changes the root and is a different object.

For every path we store the raw atom category at each position and the full
typed **induced** adjacency on the visited nodes: the consecutive path edges
plus every non-consecutive edge that exists in the raw graph.  A missing edge
and a padded pair are kept distinct by an explicit pair mask.

Patterns
--------
A *pattern* is the content tuple

    (length, atom categories at positions 0..5, pair types for 15 pairs)

where pair type is ``0 = no edge`` or the raw bond type ``1..3``.  Identical
content traced from several distinct node sequences is one pattern with a true
per-root multiplicity; the multiplicity is counted, never discarded.  Pattern
ids are a reversible content address (byte-sorted), not learned ids and not a
frequency vocabulary.

The ``Q`` variant uses exactly the same enumeration objects and multiplicities
but replaces every non-consecutive pair type by ``0`` (the typed path sequence
only).

No labels, predictions or test data are used anywhere in this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

EXP = _HERE.parent
CACHE = EXP / "cache"

LENGTHS = (2, 3, 4, 5)
LENGTH_INDEX = {l: i for i, l in enumerate(LENGTHS)}
MAX_NODES = 6
N_PAIRS = 15
PAIR_INDEX = tuple((i, j) for i in range(MAX_NODES) for j in range(i + 1, MAX_NODES))
PAIR_J = np.asarray([j for _, j in PAIR_INDEX], dtype=np.int64)
KEY_LEN = MAX_NODES + N_PAIRS

_PATH_CACHE: dict[str, dict] = {}


def cache_path(split: str) -> Path:
    return CACHE / f"path_{split}.pt"


def atom_categories(obj) -> np.ndarray:
    feat = np.asarray(obj.atom_feat, dtype=np.float32)
    if feat.shape[1] < D.N_ATOM_TYPES:
        raise ValueError("atom_feat has fewer than 21 columns")
    onehot = feat[:, : D.N_ATOM_TYPES]
    cats = np.argmax(onehot, axis=1).astype(np.uint8)
    if not np.all(onehot[np.arange(len(cats)), cats] == 1.0):
        raise RuntimeError("atom_feat first 21 columns are not a one-hot atom category")
    return cats


def _bond_map(obj) -> dict[tuple[int, int], int]:
    bi = np.asarray(obj.bond_index, dtype=np.int64)
    bt = np.asarray(obj.bond_type, dtype=np.int64)
    out: dict[tuple[int, int], int] = {}
    for (a, b), t in zip(bi, bt):
        a, b, t = int(a), int(b), int(t)
        if a == b or t not in (1, 2, 3):
            raise RuntimeError(f"invalid raw bond ({a},{b}) type {t}")
        out[(a, b)] = t
        out[(b, a)] = t
    return out


def enumerate_rooted_paths(obj) -> dict[int, dict[bytes, dict[int, int]]]:
    """Return ``{length: {pattern_key: {root: multiplicity}}}`` for one molecule.

    ``pattern_key`` is exactly ``atom_categories[6] + pair_types[15]`` bytes in
    the canonical pair order.  The root is position 0.  Every distinct simple
    path is counted once at its starting root.
    """
    n = int(obj.n_atoms)
    cats = atom_categories(obj)
    bmap = _bond_map(obj)
    adj: list[list[int]] = [[] for _ in range(n)]
    for (a, b) in bmap:
        if a < b:
            adj[a].append(b)
            adj[b].append(a)
    for a in range(n):
        adj[a].sort()

    out = {l: {} for l in LENGTHS}
    for root in range(n):
        stack: list[tuple[int, tuple[int, ...]]] = [(root, (root,))]
        while stack:
            u, path = stack.pop()
            length = len(path) - 1
            if length in out:
                atom_part = np.zeros(MAX_NODES, dtype=np.uint8)
                atom_part[: length + 1] = cats[list(path)]
                pair_part = np.zeros(N_PAIRS, dtype=np.uint8)
                for k, (i, j) in enumerate(PAIR_INDEX):
                    if j <= length:
                        pair_part[k] = bmap.get((path[i], path[j]), 0)
                key = atom_part.tobytes() + pair_part.tobytes()
                per_root = out[length].setdefault(key, {})
                per_root[root] = per_root.get(root, 0) + 1
            if length < max(LENGTHS):
                for w in adj[u]:
                    if w not in path:
                        stack.append((w, path + (w,)))
    return out


def q_key_from_key(key: bytes) -> bytes:
    """Drop every non-consecutive pair type from a full pattern key."""
    if len(key) != KEY_LEN:
        raise ValueError(f"bad pattern key length {len(key)}")
    atom = key[:MAX_NODES]
    pair = np.frombuffer(key[MAX_NODES:], dtype=np.uint8).copy()
    for k, (i, j) in enumerate(PAIR_INDEX):
        if j != i + 1:
            pair[k] = 0
    return atom + pair.tobytes()


def _finalise_table(key_list: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort content keys and return (atom [P,6], pair [P,15], length [P])."""
    order = sorted(range(len(key_list)), key=lambda i: key_list[i])
    remap = np.empty((len(key_list),), dtype=np.int32)
    remap[np.asarray(order, dtype=np.int64)] = np.arange(len(key_list), dtype=np.int32)
    atom = np.zeros((len(key_list), MAX_NODES), dtype=np.uint8)
    pair = np.zeros((len(key_list), N_PAIRS), dtype=np.uint8)
    for new_id, old_id in enumerate(order):
        raw = np.frombuffer(key_list[old_id], dtype=np.uint8)
        atom[new_id] = raw[:MAX_NODES]
        pair[new_id] = raw[MAX_NODES:]
    return atom, pair, remap


def build_split(split: str, *, verbose: bool = True) -> dict:
    objs, source = D.load_p3_objects(split)
    return build_from_objects(objs, split, source, verbose=verbose)


def build_from_objects(objs, split: str, source: dict, *, verbose: bool = True) -> dict:
    t0 = __import__("time").perf_counter()
    n_mol = len(objs)
    keys_j: list[dict[bytes, int]] = [dict() for _ in LENGTHS]
    keys_q: list[dict[bytes, int]] = [dict() for _ in LENGTHS]
    list_j: list[list[bytes]] = [[] for _ in LENGTHS]
    list_q: list[list[bytes]] = [[] for _ in LENGTHS]
    ent_mol: list[int] = []
    ent_len: list[int] = []
    ent_j_raw: list[int] = []
    ent_q_raw: list[int] = []
    ent_root: list[int] = []
    ent_count: list[int] = []
    mol_entries = np.zeros((n_mol,), dtype=np.int64)
    occ_by_len = np.zeros((len(LENGTHS),), dtype=np.int64)

    for m, obj in enumerate(objs):
        per_len = enumerate_rooted_paths(obj)
        start_entries = len(ent_mol)
        for length in LENGTHS:
            li = LENGTH_INDEX[length]
            for key, per_root in per_len[length].items():
                qkey = q_key_from_key(key)
                jid = keys_j[li].get(key)
                if jid is None:
                    jid = len(keys_j[li])
                    keys_j[li][key] = jid
                    list_j[li].append(key)
                qid = keys_q[li].get(qkey)
                if qid is None:
                    qid = len(keys_q[li])
                    keys_q[li][qkey] = qid
                    list_q[li].append(qkey)
                for root, count in per_root.items():
                    ent_mol.append(m)
                    ent_len.append(li)
                    ent_j_raw.append(jid)
                    ent_q_raw.append(qid)
                    ent_root.append(int(root))
                    ent_count.append(int(count))
                    occ_by_len[li] += int(count)
        mol_entries[m] = len(ent_mol) - start_entries
        if verbose and (m + 1) % 2000 == 0:
            C.log(f"[ip-prep {split}] {m + 1}/{n_mol} molecules {__import__('time').perf_counter() - t0:.1f}s")

    # Build content-sorted pattern tables and per-length raw->sorted id maps.
    pat_atom_j = np.zeros((0, MAX_NODES), dtype=np.uint8)
    pat_pair_j = np.zeros((0, N_PAIRS), dtype=np.uint8)
    pat_len_j = np.zeros((0,), dtype=np.uint8)
    pat_atom_q = np.zeros((0, MAX_NODES), dtype=np.uint8)
    pat_pair_q = np.zeros((0, N_PAIRS), dtype=np.uint8)
    pat_len_q = np.zeros((0,), dtype=np.uint8)
    remap_j: list[np.ndarray] = []
    remap_q: list[np.ndarray] = []
    for li, length in enumerate(LENGTHS):
        aj, pj, rj = _finalise_table(list_j[li])
        aq, pq, rq = _finalise_table(list_q[li])
        pat_atom_j = np.concatenate([pat_atom_j, aj], axis=0)
        pat_pair_j = np.concatenate([pat_pair_j, pj], axis=0)
        pat_len_j = np.concatenate([pat_len_j, np.full((len(aj),), length, dtype=np.uint8)])
        pat_atom_q = np.concatenate([pat_atom_q, aq], axis=0)
        pat_pair_q = np.concatenate([pat_pair_q, pq], axis=0)
        pat_len_q = np.concatenate([pat_len_q, np.full((len(aq),), length, dtype=np.uint8)])
        remap_j.append(rj)
        remap_q.append(rq)

    ent_len_arr = np.asarray(ent_len, dtype=np.uint8)
    ent_j_raw_arr = np.asarray(ent_j_raw, dtype=np.int32)
    ent_q_raw_arr = np.asarray(ent_q_raw, dtype=np.int32)
    ent_pat_j = np.zeros_like(ent_j_raw_arr)
    ent_pat_q = np.zeros_like(ent_q_raw_arr)
    offsets = np.zeros((len(LENGTHS) + 1,), dtype=np.int64)
    for li in range(len(LENGTHS)):
        offsets[li + 1] = offsets[li] + (len(remap_j[li]))
        mask = ent_len_arr == li
        if mask.any():
            ent_pat_j[mask] = remap_j[li][ent_j_raw_arr[mask]] + offsets[li]
            ent_pat_q[mask] = remap_q[li][ent_q_raw_arr[mask]] + np.array(
                [0] + [len(remap_q[k]) for k in range(li)], dtype=np.int64
            )[li]

    ent_ptr = np.zeros((n_mol + 1,), dtype=np.int64)
    ent_ptr[1:] = np.cumsum(mol_entries)

    data = {
        "split": split,
        "lengths": LENGTHS,
        "pat_atom_j": torch.from_numpy(pat_atom_j),
        "pat_pair_j": torch.from_numpy(pat_pair_j),
        "pat_len_j": torch.from_numpy(pat_len_j),
        "pat_atom_q": torch.from_numpy(pat_atom_q),
        "pat_pair_q": torch.from_numpy(pat_pair_q),
        "pat_len_q": torch.from_numpy(pat_len_q),
        "ent_ptr": torch.from_numpy(ent_ptr),
        "ent_len": torch.from_numpy(ent_len_arr.astype(np.int64)),
        "ent_pat_j": torch.from_numpy(ent_pat_j.astype(np.int64)),
        "ent_pat_q": torch.from_numpy(ent_pat_q.astype(np.int64)),
        "ent_root": torch.from_numpy(np.asarray(ent_root, dtype=np.int64)),
        "ent_count": torch.from_numpy(np.asarray(ent_count, dtype=np.int64)),
        "meta": {
            "object_source": source,
            "n_molecules": int(n_mol),
            "lengths": list(LENGTHS),
            "n_entries": int(len(ent_mol)),
            "n_unique_patterns_j": int(len(pat_len_j)),
            "n_unique_patterns_q": int(len(pat_len_q)),
            "occurrences_by_length": {str(l): int(occ_by_len[i]) for i, l in enumerate(LENGTHS)},
            "definition": "all distinct oriented simple paths from each root; root at position 0; full typed induced adjacency",
            "q_definition": "same paths/multiplicities; non-consecutive pair types set to 0 (no edge)",
            "labels_used": False,
            "test_used": False,
            "created": C.utcnow(),
            "prep_seconds": float(__import__("time").perf_counter() - t0),
        },
    }
    return data


def save_split(split: str) -> Path:
    data = build_split(split)
    path = cache_path(split)
    C.atomic_torch_save(data, path)
    C.log(f"[ip-prep] wrote {path} ({path.stat().st_size / 1e6:.1f} MB) {data['meta']}")
    return path


def load_path_cache(split: str) -> dict:
    if split not in _PATH_CACHE:
        path = cache_path(split)
        if not path.exists():
            raise FileNotFoundError(f"missing path cache {path}; run `python ip_data.py prep`")
        _PATH_CACHE[split] = torch.load(path, map_location="cpu", weights_only=False)
    return _PATH_CACHE[split]


def _pattern_masks(length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = np.asarray(length, dtype=np.int64)
    node_mask = np.arange(MAX_NODES)[None, :] <= length[:, None]
    pair_mask = PAIR_J[None, :] <= length[:, None]
    return node_mask, pair_mask


def collate_paths(cache, rows, candidate: str, *, valid: bool = False, walk_cols=None):
    """Base P3/T batch plus the batched path-pattern occurrence structure."""
    if candidate not in ("J", "Q"):
        raise ValueError(f"collate_paths expects J or Q, got {candidate!r}")
    import ad_train as T1

    base = T1.collate(cache, rows, "T", valid=valid)
    pc = load_path_cache(str(cache["split"]))
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    ent_ptr = np.asarray(pc["ent_ptr"])
    counts = ent_ptr[rows + 1] - ent_ptr[rows]
    if int(counts.sum()):
        ent_idx = np.concatenate(
            [np.arange(int(s), int(e), dtype=np.int64) for s, e in zip(ent_ptr[rows], ent_ptr[rows + 1])]
        )
    else:
        ent_idx = np.zeros((0,), dtype=np.int64)
    ent_mol = np.repeat(np.arange(B, dtype=np.int64), counts)

    atom_ptr = np.asarray(cache["atom_ptr"])
    atom_counts = np.diff(atom_ptr)[rows].astype(np.int64)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)

    ent_len = np.asarray(pc["ent_len"])[ent_idx].astype(np.int64)
    ent_root = np.asarray(pc["ent_root"])[ent_idx].astype(np.int64)
    ent_count = np.asarray(pc["ent_count"])[ent_idx].astype(np.float32)
    key_pat = "ent_pat_j" if candidate == "J" else "ent_pat_q"
    ent_pat = np.asarray(pc[key_pat])[ent_idx].astype(np.int64)

    if ent_pat.size:
        uniq, inv = np.unique(ent_pat, return_inverse=True)
    else:
        uniq = np.zeros((0,), dtype=np.int64)
        inv = np.zeros((0,), dtype=np.int64)

    suffix = "j" if candidate == "J" else "q"
    pat_atom = np.asarray(pc[f"pat_atom_{suffix}"])[uniq]
    pat_pair = np.asarray(pc[f"pat_pair_{suffix}"])[uniq]
    pat_len = np.asarray(pc[f"pat_len_{suffix}"])[uniq]
    node_mask, pair_mask = _pattern_masks(pat_len)

    occ_root = atom_off[ent_mol] + ent_root if ent_mol.size else np.zeros((0,), dtype=np.int64)
    base.update(
        {
            "pat_atom": torch.from_numpy(pat_atom.astype(np.int64)),
            "pat_pair": torch.from_numpy(pat_pair.astype(np.int64)),
            "pat_len": torch.from_numpy(pat_len.astype(np.int64)),
            "pat_node_mask": torch.from_numpy(node_mask),
            "pat_pair_mask": torch.from_numpy(pair_mask),
            "occ_pat": torch.from_numpy(inv.astype(np.int64)),
            "occ_len": torch.from_numpy(ent_len.astype(np.int64)),
            "occ_root": torch.from_numpy(occ_root.astype(np.int64)),
            "occ_count": torch.from_numpy(np.ascontiguousarray(ent_count, dtype=np.float32)),
            "n_atoms_total": int(atom_counts.sum()),
            "path_variant": candidate,
        }
    )
    return base


def audit_split(split: str) -> dict:
    """Label-free integrity audit of a built path cache."""
    pc = load_path_cache(split)
    ent_ptr = np.asarray(pc["ent_ptr"])
    ent_len = np.asarray(pc["ent_len"])
    ent_root = np.asarray(pc["ent_root"])
    ent_count = np.asarray(pc["ent_count"])
    out = {"split": split, "n_molecules": int(len(ent_ptr) - 1), "meta": pc["meta"]}
    for suffix in ("j", "q"):
        pat_atom = np.asarray(pc[f"pat_atom_{suffix}"])
        pat_pair = np.asarray(pc[f"pat_pair_{suffix}"])
        pat_len = np.asarray(pc[f"pat_len_{suffix}"])
        node_mask, pair_mask = _pattern_masks(pat_len)
        bad = 0
        for p in range(len(pat_len)):
            l = int(pat_len[p])
            if not node_mask[p, : l + 1].all() or node_mask[p, l + 1 :].any():
                bad += 1
                continue
            if int(pat_atom[p, l + 1 :].max(initial=0)) != 0:
                bad += 1
            if not pair_mask[p, :].any():
                bad += 1
            # symmetry cannot be checked on a packed upper triangle, but every
            # valid pair must be either 0 (no edge) or a raw bond type 1..3.
            if not np.isin(pat_pair[p][pair_mask[p]], (0, 1, 2, 3)).all():
                bad += 1
            if l == int(np.nonzero(node_mask[p])[0].max()):
                pass
        out[f"pattern_bad_{suffix}"] = int(bad)
    # multiplicity sum equals occurrence count
    out["n_entries"] = int(len(ent_count))
    out["occ_by_length"] = {
        str(l): int(ent_count[ent_len == i].sum()) for i, l in enumerate(LENGTHS)
    }
    out["entry_multiplicity_max"] = int(ent_count.max(initial=0))
    out["root_max"] = int(ent_root.max(initial=0))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "audit"])
    p.add_argument("--split", action="append", choices=list(LENGTHS) + ["train", "valid"], default=None,
                   help="split to build/audit (repeatable); default train+valid")
    args = p.parse_args(argv)
    splits = args.split or ["train", "valid"]
    if args.command == "prep":
        CACHE.mkdir(parents=True, exist_ok=True)
        for split in splits:
            if split in ("train", "valid"):
                save_split(split)
    else:
        for split in splits:
            if split in ("train", "valid"):
                C.log(audit_split(split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
