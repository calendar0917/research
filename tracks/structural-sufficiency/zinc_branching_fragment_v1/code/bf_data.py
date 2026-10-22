"""Rooted branching-fragment enumeration and batching for P3-BF.

Objects (label-free, exhaustive, no sampling)
---------------------------------------------
For every molecule and every root atom ``v`` three families are enumerated
from the raw atom categories and the real physical bonds only:

``S3``  root ``v`` plus three distinct neighbours (4 nodes, one root + 3);
``S4``  root ``v`` plus four distinct neighbours (5 nodes, one root + 4);
``F4``  neighbour ``u`` of ``v`` plus two distinct ``w1, w2`` from
        ``N(u) \\ {v}``; the node set is ``{v, u, w1, w2}`` (4 nodes).

Every object keeps the complete typed induced adjacency on the selected nodes:
the template edges and every additional real edge.  A pair type is
``0 = no real edge`` and ``1..3`` is the raw bond category (the ZINC subset
bond codes actually present; verified in the P3 mask-token audit).

Canonical content
-----------------
The root is fixed at position 0.  The remaining nodes are ordered by taking
the lexicographic minimum, over all permutations of those nodes, of

    atom categories (5 slots) + pair types (10 slots, lexicographic pair order)

so the content key contains no atom index, split id, traversal order or
temporary id.  Invalid slots are zero and are separated from real "no edge"
pairs by the node/pair masks in the model.

Multiplicity
------------
One ``(root, family, actual node set)`` counts once.  F4 generation orders /
centres that yield the same node set are deduplicated.  Two different node
sets with the same canonical content remain two multiplicity units; the
weighted root statistics keep the true count.

Families are independent: one node set may satisfy several families and is
counted in each.  A root with no object of a family has the exact empty-set
semantics (no fabricated token).

Caches store only static content / index / multiplicity arrays.  No labels,
predictions or test data are used anywhere in this module.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
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

FAMILIES = ("S3", "S4", "F4")
FAMILY_INDEX = {f: i for i, f in enumerate(FAMILIES)}
ATOM_CATEGORIES = 28  # same category space as the P3 raw108 atom one-hot
MAX_NODES = 5
N_PAIRS = 10
PAIR_INDEX = tuple((i, j) for i in range(MAX_NODES) for j in range(i + 1, MAX_NODES))
PAIR_J = np.asarray([j for _, j in PAIR_INDEX], dtype=np.int64)
KEY_LEN = MAX_NODES + N_PAIRS  # 15 bytes per content key

_CACHE: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# raw graph -> content
# ---------------------------------------------------------------------------


def cache_path(split: str) -> Path:
    return CACHE / f"bf_{split}.pt"


def atom_categories(obj) -> np.ndarray:
    """Raw atom category from the P3 28-slot atom one-hot block."""
    feat = np.asarray(obj.atom_feat, dtype=np.float32)
    if feat.shape[1] < ATOM_CATEGORIES:
        raise ValueError("atom_feat has fewer than 28 columns")
    onehot = feat[:, :ATOM_CATEGORIES]
    cats = np.argmax(onehot, axis=1).astype(np.uint8)
    if not np.all(onehot[np.arange(len(cats)), cats] == 1.0):
        raise RuntimeError("atom_feat first 28 columns are not a one-hot atom category")
    return cats


def bond_map(obj) -> dict[tuple[int, int], int]:
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


def adjacency(n_atoms: int, bmap: dict[tuple[int, int], int]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for (a, b) in bmap:
        if a < b:
            adj[a].append(b)
            adj[b].append(a)
    for a in range(n_atoms):
        adj[a].sort()
    return adj


def canonical_key(cats: np.ndarray, bmap: dict[tuple[int, int], int], root: int,
                  others) -> bytes:
    """Lexicographic-min typed content key with the root fixed at slot 0."""
    best = None
    for perm in itertools.permutations(others):
        order = (int(root),) + perm
        atom = [0] * MAX_NODES
        for k, node in enumerate(order):
            atom[k] = int(cats[node])
        pairs = [0] * N_PAIRS
        for k, (i, j) in enumerate(PAIR_INDEX):
            if j < len(order):
                pairs[k] = bmap.get((order[i], order[j]), 0)
        key = bytes(atom) + bytes(pairs)
        if best is None or key < best:
            best = key
    return best


def enumerate_rooted_branching(obj) -> list[dict[int, dict[bytes, int]]]:
    """Return ``[family][root] -> {content_key: multiplicity}`` for one molecule."""
    n = int(obj.n_atoms)
    cats = atom_categories(obj)
    bmap = bond_map(obj)
    adj = adjacency(n, bmap)
    out: list[dict[int, dict[bytes, int]]] = [{} for _ in FAMILIES]
    for root in range(n):
        nbrs = adj[root]
        sets: tuple[set, set, set] = (set(), set(), set())
        if len(nbrs) >= 3:
            for combo in itertools.combinations(nbrs, 3):  # sorted neighbours
                sets[0].add(combo)
        if len(nbrs) >= 4:
            for combo in itertools.combinations(nbrs, 4):
                sets[1].add(combo)
        for u in nbrs:
            rest = [w for w in adj[u] if w != root]
            for combo in itertools.combinations(rest, 2):
                sets[2].add(tuple(sorted((u,) + combo)))
        for fi, node_sets in enumerate(sets):
            per_root: dict[bytes, int] = {}
            for others in node_sets:
                key = canonical_key(cats, bmap, root, others)
                per_root[key] = per_root.get(key, 0) + 1
            out[fi][root] = per_root
    return out


# ---------------------------------------------------------------------------
# cache construction
# ---------------------------------------------------------------------------


def _decode_keys(keys: list[bytes]):
    atom = np.zeros((len(keys), MAX_NODES), dtype=np.uint8)
    pair = np.zeros((len(keys), N_PAIRS), dtype=np.uint8)
    for i, key in enumerate(keys):
        raw = np.frombuffer(key, dtype=np.uint8)
        atom[i] = raw[:MAX_NODES]
        pair[i] = raw[MAX_NODES:]
    return atom, pair


def build_from_objects(objs, split: str, source: dict, *, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    n_mol = len(objs)
    keys: list[dict[bytes, int]] = [{} for _ in FAMILIES]
    key_lists: list[list[bytes]] = [[] for _ in FAMILIES]
    ent_pat_raw: list[int] = []
    ent_root: list[int] = []
    ent_family: list[int] = []
    ent_count: list[int] = []
    mol_entries = np.zeros((n_mol,), dtype=np.int64)
    occ_by_family = np.zeros((len(FAMILIES),), dtype=np.int64)
    nonempty_roots = np.zeros((len(FAMILIES),), dtype=np.int64)
    max_per_root = np.zeros((len(FAMILIES),), dtype=np.int64)
    max_per_graph = np.zeros((len(FAMILIES),), dtype=np.int64)
    atoms_per_mol = np.zeros((n_mol,), dtype=np.int64)

    for m, obj in enumerate(objs):
        n_atoms = int(obj.n_atoms)
        atoms_per_mol[m] = n_atoms
        per = enumerate_rooted_branching(obj)
        start = len(ent_pat_raw)
        per_mol = [0, 0, 0]
        for root in range(n_atoms):
            for fi in range(len(FAMILIES)):
                content = per[fi][root]
                if not content:
                    continue
                nonempty_roots[fi] += 1
                total = int(sum(content.values()))
                max_per_root[fi] = max(max_per_root[fi], total)
                per_mol[fi] += total
                for key in sorted(content):
                    count = int(content[key])
                    pid = keys[fi].get(key)
                    if pid is None:
                        pid = len(keys[fi])
                        keys[fi][key] = pid
                        key_lists[fi].append(key)
                    ent_pat_raw.append(pid)
                    ent_root.append(root)
                    ent_family.append(fi)
                    ent_count.append(count)
        for fi in range(len(FAMILIES)):
            max_per_graph[fi] = max(max_per_graph[fi], per_mol[fi])
            occ_by_family[fi] += per_mol[fi]
        mol_entries[m] = len(ent_pat_raw) - start
        if verbose and (m + 1) % 2000 == 0:
            C.log(f"[bf-prep {split}] {m + 1}/{n_mol} molecules "
                  f"{time.perf_counter() - t0:.1f}s")

    pat_atom = np.zeros((0, MAX_NODES), dtype=np.uint8)
    pat_pair = np.zeros((0, N_PAIRS), dtype=np.uint8)
    pat_len = np.zeros((0,), dtype=np.uint8)
    pat_family = np.zeros((0,), dtype=np.uint8)
    remaps: list[np.ndarray] = []
    for fi in range(len(FAMILIES)):
        keys_fi = key_lists[fi]
        order = sorted(range(len(keys_fi)), key=lambda i: keys_fi[i])
        remap = np.empty((len(keys_fi),), dtype=np.int64)
        remap[np.asarray(order, dtype=np.int64)] = np.arange(len(keys_fi), dtype=np.int64)
        atom, pair = _decode_keys([keys_fi[i] for i in order])
        length = np.full((len(order),), MAX_NODES - 1 if FAMILIES[fi] != "S4" else MAX_NODES,
                         dtype=np.uint8)
        pat_atom = np.concatenate([pat_atom, atom], axis=0)
        pat_pair = np.concatenate([pat_pair, pair], axis=0)
        pat_len = np.concatenate([pat_len, length])
        pat_family = np.concatenate([pat_family, np.full((len(order),), fi, dtype=np.uint8)])
        remaps.append(remap)

    ent_pat = np.zeros((len(ent_pat_raw),), dtype=np.int64)
    offsets = np.zeros((len(FAMILIES) + 1,), dtype=np.int64)
    for fi in range(len(FAMILIES)):
        offsets[fi + 1] = offsets[fi] + len(remaps[fi])
    for fi in range(len(FAMILIES)):
        mask = np.asarray(ent_family, dtype=np.int64) == fi
        if mask.any():
            ent_pat[mask] = remaps[fi][np.asarray(ent_pat_raw, dtype=np.int64)[mask]] + offsets[fi]

    ent_ptr = np.zeros((n_mol + 1,), dtype=np.int64)
    ent_ptr[1:] = np.cumsum(mol_entries)

    n_roots_total = int(atoms_per_mol.sum())
    data = {
        "split": split,
        "families": FAMILIES,
        "max_nodes": MAX_NODES,
        "n_pairs": N_PAIRS,
        "pat_atom": torch.from_numpy(pat_atom),
        "pat_pair": torch.from_numpy(pat_pair),
        "pat_len": torch.from_numpy(pat_len.astype(np.int64)),
        "pat_family": torch.from_numpy(pat_family.astype(np.int64)),
        "ent_ptr": torch.from_numpy(ent_ptr),
        "ent_pat": torch.from_numpy(ent_pat.astype(np.int64)),
        "ent_root": torch.from_numpy(np.asarray(ent_root, dtype=np.int64)),
        "ent_family": torch.from_numpy(np.asarray(ent_family, dtype=np.int64)),
        "ent_count": torch.from_numpy(np.asarray(ent_count, dtype=np.int64)),
        "atoms_per_mol": torch.from_numpy(atoms_per_mol),
        "meta": {
            "split": split,
            "families": list(FAMILIES),
            "object_source": source,
            "n_molecules": int(n_mol),
            "n_atoms": n_roots_total,
            "n_entries": int(len(ent_pat)),
            "n_unique_patterns": int(len(pat_len)),
            "unique_patterns_by_family": {
                f: int(len(remaps[fi])) for fi, f in enumerate(FAMILIES)
            },
            "occurrences_by_family": {
                f: int(occ_by_family[fi]) for fi, f in enumerate(FAMILIES)
            },
            "nonempty_roots_by_family": {
                f: int(nonempty_roots[fi]) for fi, f in enumerate(FAMILIES)
            },
            "empty_root_fraction_by_family": {
                f: float(1.0 - nonempty_roots[fi] / max(n_roots_total, 1))
                for fi, f in enumerate(FAMILIES)
            },
            "max_objects_per_root_by_family": {
                f: int(max_per_root[fi]) for fi, f in enumerate(FAMILIES)
            },
            "max_objects_per_graph_by_family": {
                f: int(max_per_graph[fi]) for fi, f in enumerate(FAMILIES)
            },
            "max_entries_per_molecule": int(mol_entries.max(initial=0)),
            "definition": (
                "exhaustive rooted branching fragments S3 (root+3 neighbours), "
                "S4 (root+4 neighbours), F4 (root-u-{w1,w2}); complete typed "
                "induced adjacency; root at canonical position 0"
            ),
            "multiplicity_definition": (
                "distinct actual node sets with equal content key are counted; "
                "one (root, family, node set) exactly once"
            ),
            "canonicalisation": "lexicographic-min over permutations of the non-root nodes",
            "labels_used": False,
            "test_used": split == "test",
            "created": C.utcnow(),
            "prep_seconds": float(time.perf_counter() - t0),
        },
    }
    return data


def build_split(split: str, *, verbose: bool = True) -> dict:
    objs, source = D.load_p3_objects(split)
    return build_from_objects(objs, split, source, verbose=verbose)


def save_split(split: str) -> Path:
    data = build_split(split)
    path = cache_path(split)
    C.atomic_torch_save(data, path)
    C.log(f"[bf-prep] wrote {path} ({path.stat().st_size / 1e6:.1f} MB) {data['meta']}")
    return path


def load_cache(split: str) -> dict:
    if split not in _CACHE:
        path = cache_path(split)
        if not path.exists():
            raise FileNotFoundError(f"missing fragment cache {path}; run `bf_data.py prep`")
        _CACHE[split] = torch.load(path, map_location="cpu", weights_only=False)
    return _CACHE[split]


def register_cache(split: str, data: dict) -> None:
    """Register an in-memory cache (used by the CPU checks on synthetic graphs)."""
    _CACHE[split] = data


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def pattern_masks(length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = np.asarray(length, dtype=np.int64)
    node_mask = np.arange(MAX_NODES)[None, :] < length[:, None]
    pair_mask = PAIR_J[None, :] < length[:, None]
    return node_mask, pair_mask


def collate_bf(cache, rows, candidate: str = "P3BF", *, valid: bool = False, walk_cols=None):
    """Base P3/T batch plus the batched fragment occurrence structure."""
    import ad_train as T1

    base = T1.collate(cache, rows, "T", valid=valid)
    pc = load_cache(str(cache["split"]))
    rows = np.asarray(rows, dtype=np.int64)
    ent_ptr = np.asarray(pc["ent_ptr"])
    counts = ent_ptr[rows + 1] - ent_ptr[rows]
    if int(counts.sum()):
        ent_idx = np.concatenate(
            [np.arange(int(s), int(e), dtype=np.int64) for s, e in zip(ent_ptr[rows], ent_ptr[rows + 1])]
        )
    else:
        ent_idx = np.zeros((0,), dtype=np.int64)
    ent_mol = np.repeat(np.arange(len(rows), dtype=np.int64), counts)

    atom_ptr = np.asarray(cache["atom_ptr"])
    atom_counts = np.diff(atom_ptr)[rows].astype(np.int64)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)

    pat_ids = np.asarray(pc["ent_pat"])[ent_idx].astype(np.int64)
    if pat_ids.size:
        uniq, inv = np.unique(pat_ids, return_inverse=True)
    else:
        uniq = np.zeros((0,), dtype=np.int64)
        inv = np.zeros((0,), dtype=np.int64)
    occ_root = atom_off[ent_mol] + np.asarray(pc["ent_root"])[ent_idx] if ent_mol.size else \
        np.zeros((0,), dtype=np.int64)

    base.update({
        "pat_atom": torch.from_numpy(np.asarray(pc["pat_atom"])[uniq].astype(np.int64)),
        "pat_pair": torch.from_numpy(np.asarray(pc["pat_pair"])[uniq].astype(np.int64)),
        "pat_len": torch.from_numpy(np.asarray(pc["pat_len"])[uniq].astype(np.int64)),
        "pat_family": torch.from_numpy(np.asarray(pc["pat_family"])[uniq].astype(np.int64)),
        "occ_pat": torch.from_numpy(inv.astype(np.int64)),
        "occ_root": torch.from_numpy(np.ascontiguousarray(occ_root, dtype=np.int64)),
        "occ_family": torch.from_numpy(np.asarray(pc["ent_family"])[ent_idx].astype(np.int64)),
        "occ_count": torch.from_numpy(
            np.ascontiguousarray(np.asarray(pc["ent_count"])[ent_idx], dtype=np.float32)
        ),
        "n_atoms_total": int(atom_counts.sum()),
    })
    return base


def audit_cache(split: str) -> dict:
    """Label-free integrity audit of a built fragment cache."""
    pc = load_cache(split)
    pat_atom = np.asarray(pc["pat_atom"])
    pat_pair = np.asarray(pc["pat_pair"])
    pat_len = np.asarray(pc["pat_len"])
    pat_family = np.asarray(pc["pat_family"])
    node_mask, pair_mask = pattern_masks(pat_len)
    bad = 0
    for p in range(len(pat_len)):
        length = int(pat_len[p])
        if not node_mask[p, : length].all() or node_mask[p, length:].any():
            bad += 1
            continue
        if int(pat_atom[p, length:].max(initial=0)) != 0:
            bad += 1
        if not pair_mask[p].any():
            bad += 1
        if not np.isin(pat_pair[p][pair_mask[p]], (0, 1, 2, 3)).all():
            bad += 1
        # the induced matrix must be symmetric in the 5-node layout
        mat = np.zeros((MAX_NODES, MAX_NODES), dtype=np.int64)
        for k, (i, j) in enumerate(PAIR_INDEX):
            mat[i, j] = mat[j, i] = int(pat_pair[p, k])
        if not np.array_equal(mat, mat.T):
            bad += 1
        if int(pat_family[p]) not in range(len(FAMILIES)):
            bad += 1
    counts = np.asarray(pc["ent_count"])
    fam = np.asarray(pc["ent_family"])
    root = np.asarray(pc["ent_root"])
    atoms = np.asarray(pc["atoms_per_mol"])
    mol = np.repeat(np.arange(len(atoms)), np.diff(np.asarray(pc["ent_ptr"])))
    roots_in_range = bool(np.all(root < atoms[mol])) if len(root) else True
    return {
        "split": split,
        "pattern_bad": int(bad),
        "patterns": int(len(pat_len)),
        "entries": int(len(counts)),
        "entries_positive": bool(np.all(counts >= 1)) if len(counts) else True,
        "roots_in_range": roots_in_range,
        "families_ok": bool(np.all((fam >= 0) & (fam < len(FAMILIES)))) if len(fam) else True,
        "meta": pc["meta"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "audit"])
    p.add_argument("--split", action="append", default=None)
    args = p.parse_args(argv)
    splits = args.split or ["train", "valid"]
    if args.command == "prep":
        CACHE.mkdir(parents=True, exist_ok=True)
        for split in splits:
            save_split(split)
    else:
        for split in splits:
            C.log(audit_cache(split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
