"""Fixed structural descriptors of the radius-1 / radius-2 local environments.

The descriptors are computed from the raw attributed graph only (no labels).
The vocabulary and standardisation statistics are built from Train-sub pairs
only; Dev structures are encoded with the frozen vocabulary.

Two fixed descriptor blocks are produced for each ball:

* ``compact``: atom/bond category fractions plus size/ring/root scalars;
* ``fingerprint``: a rooted Weisfeiler-Lehman count fingerprint
  (fixed hash dimension and iteration count, no learning).

The pair feature vector concatenates the radius-1 and radius-2 blocks of the
environment side of the matched pair.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import audit_common as A

# read-only parent code for the raw graph loader
if str(A.TRC_DIR / "code") not in sys.path:
    sys.path.insert(0, str(A.TRC_DIR / "code"))

import trc_common as PC  # noqa: E402
import trc_data as PD  # noqa: E402


@dataclass
class FeatureSet:
    F: np.ndarray                 # (n_pairs, n_raw_features)
    compact_dim: int
    fingerprint_dim: int
    n_used: int
    mu: np.ndarray
    sd: np.ndarray
    keep_columns: np.ndarray
    checks: dict[str, Any]


def _ball(mol: PD.Molecule, port: int, banned: tuple[int, int],
          radius: int) -> tuple[list[int], list[tuple[int, int, int]], dict[int, int]]:
    adj = mol.adjacency_lists()
    dist = {int(port): 0}
    frontier = [int(port)]
    for _ in range(int(radius)):
        nxt: list[int] = []
        for u in frontier:
            for w in adj[u]:
                if (u == banned[0] and w == banned[1]) or (u == banned[1] and w == banned[0]):
                    continue
                if w not in dist:
                    dist[w] = dist[u] + 1
                    nxt.append(w)
        frontier = nxt
    atoms = sorted(dist)
    pos = {a: i for i, a in enumerate(atoms)}
    pairs: list[tuple[int, int, int]] = []
    for e, (u, v) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        if (u == banned[0] and v == banned[1]) or (u == banned[1] and v == banned[0]):
            continue
        if u in pos and v in pos:
            pairs.append((pos[u], pos[v], e))
    return atoms, pairs, dist


def _compact(mol: PD.Molecule, port: int, banned: tuple[int, int], radius: int,
             atom_vocab: Sequence[Sequence[int]],
             bond_vocab: Sequence[Sequence[int]]) -> np.ndarray:
    atoms, pairs, _dist = _ball(mol, port, banned, radius)
    n = len(atoms)
    nb = len(pairs)
    degree = defaultdict(int)
    for i, j, _e in pairs:
        degree[i] += 1
        degree[j] += 1
    root = atoms.index(int(port))
    root_deg = degree[root]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _e in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    n_comp = len({find(i) for i in range(n)})
    cycle_rank = nb - n + n_comp

    neighbours: list[int] = []
    for i, j, _e in pairs:
        if i == root:
            neighbours.append(j)
        if j == root:
            neighbours.append(i)
    root_in_ring = 0
    if len(neighbours) >= 2:
        allowed = set(range(n)) - {root}
        seen = {neighbours[0]}
        stack = [neighbours[0]]
        while stack:
            x = stack.pop()
            for i, j, _e in pairs:
                if i == x and j in allowed and j not in seen:
                    seen.add(j)
                    stack.append(j)
                if j == x and i in allowed and i not in seen:
                    seen.add(i)
                    stack.append(i)
        root_in_ring = int(neighbours[1] in seen)

    pieces: list[np.ndarray] = []
    for c, values in enumerate(atom_vocab):
        index = {int(v): i for i, v in enumerate(values)}
        counts = np.zeros(len(values) + 1, np.float32)
        for a in atoms:
            counts[index.get(int(mol.atom_feat[a, c]), len(values))] += 1.0
        pieces.append(counts / max(1, n))
    for c, values in enumerate(bond_vocab):
        index = {int(v): i for i, v in enumerate(values)}
        counts = np.zeros(len(values) + 1, np.float32)
        for i, j, e in pairs:
            counts[index.get(int(mol.bond_feat[e, c]), len(values))] += 1.0
        pieces.append(counts / max(1, nb))
    scalars = np.asarray([
        np.log1p(n), np.log1p(nb), cycle_rank, root_deg, root_in_ring,
        int(cycle_rank > 0), nb / max(1, n),
    ], np.float32)
    pieces.append(scalars)
    return np.concatenate(pieces).astype(np.float32)


def _fingerprint(mol: PD.Molecule, port: int, banned: tuple[int, int], radius: int,
                 dim: int = A.FINGERPRINT_BITS, iters: int = A.FINGERPRINT_ITERS) -> np.ndarray:
    atoms, pairs, dist = _ball(mol, port, banned, radius)
    adj: list[list[tuple[int, int, int, int]]] = [[] for _ in atoms]
    for i, j, e in pairs:
        bt, bs, bc = (int(mol.bond_feat[e, k]) for k in range(3))
        adj[i].append((j, bt, bs, bc))
        adj[j].append((i, bt, bs, bc))
    labels = [tuple(int(x) for x in mol.atom_feat[a]) + (int(dist[a]),) for a in atoms]
    vec = np.zeros(int(dim), np.float32)

    def add(label: Any) -> None:
        h = int.from_bytes(hashlib.blake2b(repr(label).encode(), digest_size=8).digest(), "little")
        vec[h % int(dim)] += 1.0

    for label in labels:
        add(("sra-init", label))
    for it in range(int(iters)):
        new = []
        for i in range(len(atoms)):
            neigh = tuple(sorted((bt, bs, bc, labels[j]) for j, bt, bs, bc in adj[i]))
            new.append(tuple(labels[i]) + (it, neigh))
        labels = new
        for label in labels:
            add(("sra-wl", label))
    return vec


class EnvironmentFeatures:
    """Compute and memoise the environment descriptors of all matched pairs."""

    def __init__(self, inp: A.Inputs, pairs: A.PairTable) -> None:
        self.inp = inp
        self.pairs = pairs
        self.raw = PD.load_raw_cache(PC.load_config(), "lipo")
        self._mol_cache: dict[int, PD.Molecule] = {}
        self._compact_cache: dict[int, np.ndarray] = {}
        self._fp_cache: dict[int, np.ndarray] = {}
        self._pmap = A._parse_lookup(inp.trc)
        self._build_vocab()

    def _mol(self, row: int) -> PD.Molecule:
        got = self._mol_cache.get(int(row))
        if got is None:
            got = PD.raw_molecule(self.raw, int(row))
            self._mol_cache[int(row)] = got
        return got

    def _build_vocab(self) -> None:
        rows = np.nonzero(self.inp.train_sub)[0]
        aptr = np.asarray(self.raw["node_ptr"], np.int64)
        bptr = np.asarray(self.raw["bond_ptr"], np.int64)
        ai = np.concatenate([np.arange(aptr[r], aptr[r + 1]) for r in rows])
        bi = np.concatenate([np.arange(bptr[r], bptr[r + 1]) for r in rows])
        af = np.asarray(self.raw["atom_feat"])[ai]
        bf = np.asarray(self.raw["bond_feat"])[bi]
        self.atom_vocab = [sorted(set(af[:, c].tolist())) for c in range(af.shape[1])]
        self.bond_vocab = [sorted(set(bf[:, c].tolist())) for c in range(bf.shape[1])]

    def _representative(self, j: int) -> int:
        """A parse of the environment side of pair ``j`` (a-side is sufficient)."""
        p = self._pmap.get((int(self.pairs.a[j]), int(self.pairs.Sa[j]),
                            int(self.pairs.C[j]), int(self.pairs.bond_row[j])), [])
        if not p:
            raise KeyError(f"no parse for pair {j}")
        return int(p[0])

    def _ball_parts(self, j: int) -> tuple[PD.Molecule, int, tuple[int, int]]:
        p = self._representative(j)
        mol = self._mol(int(self.pairs.a[j]))
        port = int(self.inp.trc["parse_port_C"][p])
        banned = (int(self.inp.trc["parse_cut_u"][p]), int(self.inp.trc["parse_cut_v"][p]))
        return mol, port, banned

    def _compact_of(self, key: int, mol: PD.Molecule, port: int,
                    banned: tuple[int, int], radius: int) -> np.ndarray:
        got = self._compact_cache.get(int(key))
        if got is None:
            got = _compact(mol, port, banned, radius, self.atom_vocab, self.bond_vocab)
            self._compact_cache[int(key)] = got
        return got

    def _fp_of(self, key: int, mol: PD.Molecule, port: int,
               banned: tuple[int, int], radius: int) -> np.ndarray:
        got = self._fp_cache.get(int(key))
        if got is None:
            got = _fingerprint(mol, port, banned, radius)
            self._fp_cache[int(key)] = got
        return got

    def matrix(self) -> FeatureSet:
        pairs = self.pairs
        n = pairs.a.size
        compact_parts: list[np.ndarray] = []
        fp_parts: list[np.ndarray] = []
        seen_compact: dict[int, np.ndarray] = {}
        seen_fp: dict[int, np.ndarray] = {}
        mismatches = 0
        for j in range(n):
            mol, port, banned = self._ball_parts(j)
            c1 = self._compact_of(int(pairs.r1e[j]), mol, port, banned, 1)
            c2 = self._compact_of(int(pairs.r2e[j]), mol, port, banned, 2)
            f1 = self._fp_of(int(pairs.r1e[j]), mol, port, banned, 1)
            f2 = self._fp_of(int(pairs.r2e[j]), mol, port, banned, 2)
            if int(pairs.r1e[j]) in seen_compact and not np.array_equal(seen_compact[int(pairs.r1e[j])], c1):
                mismatches += 1
            if int(pairs.r2e[j]) in seen_compact and not np.array_equal(seen_compact[int(pairs.r2e[j])], c2):
                mismatches += 1
            if int(pairs.r1e[j]) in seen_fp and not np.array_equal(seen_fp[int(pairs.r1e[j])], f1):
                mismatches += 1
            if int(pairs.r2e[j]) in seen_fp and not np.array_equal(seen_fp[int(pairs.r2e[j])], f2):
                mismatches += 1
            seen_compact[int(pairs.r1e[j])] = c1
            seen_compact[int(pairs.r2e[j])] = c2
            seen_fp[int(pairs.r1e[j])] = f1
            seen_fp[int(pairs.r2e[j])] = f2
            compact_parts.append(np.concatenate([c1, c2]))
            fp_parts.append(np.concatenate([f1, f2]))
        compact = np.stack(compact_parts).astype(np.float32)
        fingerprint = np.stack(fp_parts).astype(np.float32)
        F = np.concatenate([compact, fingerprint], axis=1)
        if not np.isfinite(F).all():
            raise ValueError("non-finite context features")

        train = pairs.scope == 0
        mu = F[train].mean(0)
        sd = F[train].std(0)
        keep_columns = sd > 1e-8
        used_mu = mu[keep_columns]
        used_sd = sd[keep_columns]
        used_sd[used_sd == 0] = 1.0
        checks = {
            "n_pairs": int(n),
            "compact_dim_per_ball": int(compact.shape[1] // 2),
            "fingerprint_dim_per_ball": int(fingerprint.shape[1] // 2),
            "n_raw_features": int(F.shape[1]),
            "n_used_features": int(keep_columns.sum()),
            "n_zero_variance_columns": int((~keep_columns).sum()),
            "context_feature_mismatch": int(mismatches),
            "n_context_keys_compact": int(len(self._compact_cache)),
            "n_context_keys_fingerprint": int(len(self._fp_cache)),
            "atom_vocab_sizes": [len(v) for v in self.atom_vocab],
            "bond_vocab_sizes": [len(v) for v in self.bond_vocab],
            "features_finite": True,
        }
        if mismatches:
            raise ValueError(f"context features are not a function of context identity ({mismatches})")
        return FeatureSet(F=F, compact_dim=int(compact.shape[1]), fingerprint_dim=int(fingerprint.shape[1]),
                          n_used=int(keep_columns.sum()), mu=used_mu, sd=used_sd,
                          keep_columns=keep_columns, checks=checks)