"""Frozen word-response structure: orbits, per-word factors and packed split tensors.

Everything here is *structure*: the old Train-only templates and merge history
are replayed, the Fit word document frequency is counted, and every molecule is
packed with

* its full 9/3-column attributed physical graph (unchanged, all salts kept),
* the word partition (word type, canonical site order, internal bonds),
* word-internal non-edge pairs (template shortest path >= 2) with their template
  orbit and distance bucket,
* boundary atoms (at least one crossing physical bond), boundary slots,
* crossing physical bonds (never de-duplicated),
* the raw count vector of the shared linear constant term.

No lipophilicity label enters the build; the H split's labels are never stored.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import wo_common as C

MAX_ATOMS = C.MAX_WORD_ATOMS
NONEDGE_FEAT_DIM = 36 + 36 + 6 + 3  # x_a+x_b, x_a*x_b, dist onehot 0..5, 3 word scalars


# ---------------------------------------------------------------------------
# templates and automorphism orbits
# ---------------------------------------------------------------------------


def load_word_table(freeze: Mapping[str, Any]):
    templates = C.load_templates(freeze)
    table = C.W.TemplateTable()
    for row in templates:
        if int(row["tid"]) != len(table.templates):
            raise ValueError("template tids are not the freeze order")
        table._register(str(row["canon"]), np.asarray(row["labels"], dtype=np.int64),
                        np.asarray(row["bu"], dtype=np.int64),
                        np.asarray(row["bv"], dtype=np.int64),
                        np.asarray(row["btype"], dtype=np.int64))
    history = [int(tid) for tid in freeze["merge"]["history"]]
    if sorted(set(history)) != sorted(int(t) for t in table.multi):
        raise ValueError("merge history does not cover exactly the multi-atom word types")
    return table, history


def _adjacency(tmpl) -> np.ndarray:
    n = tmpl.n
    mat = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(tmpl.bu.tolist(), tmpl.bv.tolist()):
        mat[a, b] = mat[b, a] = 1
    return mat


def _distances(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full((n, n), -1, dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        queue = [s]
        while queue:
            node = queue.pop(0)
            for nxt in np.nonzero(adj[node])[0].tolist():
                if dist[s, nxt] < 0:
                    dist[s, nxt] = dist[s, node] + 1
                    queue.append(nxt)
    return dist


def automorphisms(tmpl) -> list[np.ndarray]:
    n = tmpl.n
    mat = _adjacency(tmpl)
    labels = tmpl.labels
    out: list[np.ndarray] = []
    for perm in itertools.permutations(range(n)):
        p = np.asarray(perm, dtype=np.int64)
        if not np.array_equal(labels[p], labels):
            continue
        if not np.array_equal(mat[np.ix_(p, p)], mat):
            continue
        out.append(p)
    return out


def orbit_tables(table) -> dict[str, np.ndarray]:
    """Exact single-root / double-root orbits of every template (via its automorphisms)."""
    n_t = len(table.templates)
    site_orbit = np.zeros((n_t, MAX_ATOMS), dtype=np.int32)
    pair_orbit = np.zeros((n_t, MAX_ATOMS, MAX_ATOMS), dtype=np.int32)
    tmpl_dist = np.zeros((n_t, MAX_ATOMS, MAX_ATOMS), dtype=np.int8)
    tmpl_n = np.zeros(n_t, dtype=np.int32)
    next_site = 0
    next_pair = 0
    for tid, tmpl in enumerate(table.templates):
        n = tmpl.n
        tmpl_n[tid] = n
        dist = _distances(_adjacency(tmpl))
        tmpl_dist[tid, :n, :n] = np.clip(dist, 0, 5)
        autos = automorphisms(tmpl)
        seen_site: dict[int, int] = {}
        for a in range(n):
            members = sorted({int(p[a]) for p in autos})
            root = members[0]
            if root not in seen_site:
                seen_site[root] = next_site
                next_site += 1
            site_orbit[tid, a] = seen_site[root]
        seen_pair: dict[tuple[int, int], int] = {}
        for a in range(n):
            for b in range(a + 1, n):
                members = sorted({tuple(sorted((int(p[a]), int(p[b])))) for p in autos})
                root = members[0]
                if root not in seen_pair:
                    seen_pair[root] = next_pair
                    next_pair += 1
                pair_orbit[tid, a, b] = seen_pair[root]
                pair_orbit[tid, b, a] = seen_pair[root]
    return {
        "site_orbit": site_orbit,
        "pair_orbit": pair_orbit,
        "tmpl_dist": tmpl_dist,
        "tmpl_n": tmpl_n,
        "n_site_orbits": np.asarray(next_site, dtype=np.int32),
        "n_pair_orbits": np.asarray(next_pair, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# replay of the frozen partition
# ---------------------------------------------------------------------------


def graph_structures(molecules, graphs: Sequence[int], table, history: Sequence[int]) -> list[dict]:
    states = C.W.replay_history(molecules, graphs, table, history)
    out = []
    for g in graphs:
        mol = molecules[g]
        state = states[g]
        atom_piece = state.atom_piece()
        words = []
        for idx, atoms in enumerate(state.pieces):
            tid = int(state.tmpl[idx])
            tmpl = table.templates[tid]
            pos = {a: i for i, a in enumerate(atoms)}
            internal = []
            for a in atoms:
                for nbr, bk in mol.adj[a]:
                    if atom_piece[nbr] == idx and pos[a] < pos[nbr]:
                        internal.append((pos[a], pos[nbr], int(mol.bond_feat[bk, 0])))
            internal.sort()
            words.append({
                "atoms": [int(a) for a in atoms],
                "tid": tid,
                "internal": internal,
                "m": len(internal),
                "rank": len(internal) - len(atoms) + 1,
                "n": int(tmpl.n),
            })
        out.append({"mol": mol, "atom_piece": atom_piece, "words": words})
    return out


def build_split(structs: Sequence[dict], row_ids: np.ndarray, labels: np.ndarray,
                orbit: Mapping[str, np.ndarray], s_type: np.ndarray,
                dims: Mapping[str, Sequence[int]], n_templates: int) -> dict[str, np.ndarray]:
    """Pack one split; atom index is ``graph_ptr_atom[m] + original atom id``."""
    n_mol = len(structs)
    graph_ptr_atom = np.concatenate([[0], np.cumsum([r["mol"].n_atoms for r in structs])]).astype(np.int64)
    graph_ptr_bond = np.concatenate([[0], np.cumsum([len(r["mol"].bond_u) for r in structs])]).astype(np.int64)
    graph_ptr_word = np.concatenate([[0], np.cumsum([len(r["words"]) for r in structs])]).astype(np.int64)
    n_atoms = int(graph_ptr_atom[-1])
    n_bonds = int(graph_ptr_bond[-1])
    n_words = int(graph_ptr_word[-1])

    site_orbit_t = orbit["site_orbit"]
    pair_orbit_t = orbit["pair_orbit"]
    dist_t = orbit["tmpl_dist"]

    atom_feat = np.zeros((n_atoms, 9), dtype=np.int16)
    atom_mol = np.zeros(n_atoms, dtype=np.int32)
    atom_word = np.zeros(n_atoms, dtype=np.int32)
    atom_site = np.zeros(n_atoms, dtype=np.int8)
    atom_bnd_local = np.full(n_atoms, -1, dtype=np.int8)
    atom_bnd_slot = np.full(n_atoms, -1, dtype=np.int32)
    word_mol = np.zeros(n_words, dtype=np.int32)
    word_tmpl = np.zeros(n_words, dtype=np.int16)
    word_n_atoms = np.zeros(n_words, dtype=np.int8)
    word_m = np.zeros(n_words, dtype=np.int8)
    word_rank = np.zeros(n_words, dtype=np.int8)
    word_s = np.zeros(n_words, dtype=np.float32)
    word_site_orbit = np.zeros(n_atoms, dtype=np.int32)
    word_atoms = np.zeros(n_atoms, dtype=np.int32)
    word_n_boundary = np.zeros(n_words, dtype=np.int8)
    word_boundary_start = np.zeros(n_words, dtype=np.int32)
    word_site_bnd_local = np.full((n_words, MAX_ATOMS), -1, dtype=np.int8)

    bond_mol = np.zeros(n_bonds, dtype=np.int32)
    bond_u = np.zeros(n_bonds, dtype=np.int32)
    bond_v = np.zeros(n_bonds, dtype=np.int32)
    bond_feat = np.zeros((n_bonds, 3), dtype=np.int8)
    bond_word_u = np.zeros(n_bonds, dtype=np.int32)
    bond_word_v = np.zeros(n_bonds, dtype=np.int32)
    bond_cross = np.zeros(n_bonds, dtype=np.int8)

    pair_chunks = []
    cross_chunks = []
    boundary_cursor = 0
    n_boundary_graph = np.zeros(n_mol, dtype=np.int32)
    for m, rec in enumerate(structs):
        mol = rec["mol"]
        words = rec["words"]
        atom_piece = rec["atom_piece"]
        a0 = int(graph_ptr_atom[m])
        b0 = int(graph_ptr_bond[m])
        w0 = int(graph_ptr_word[m])
        n = mol.n_atoms
        atom_feat[a0:a0 + n] = mol.atom_feat
        atom_mol[a0:a0 + n] = m
        for idx, word in enumerate(words):
            w = w0 + idx
            atoms = word["atoms"]
            word_mol[w] = m
            word_tmpl[w] = word["tid"]
            word_n_atoms[w] = word["n"]
            word_m[w] = word["m"]
            word_rank[w] = word["rank"]
            word_s[w] = float(s_type[word["tid"]])
            for site, a in enumerate(atoms):
                atom_word[a0 + a] = w
                atom_site[a0 + a] = site
                word_site_orbit[a0 + a] = int(site_orbit_t[word["tid"], site])
        # physical bonds (each exactly once, raw 3-column attributes)
        for k in range(len(mol.bond_u)):
            a = int(mol.bond_u[k])
            b = int(mol.bond_v[k])
            e = b0 + k
            bond_mol[e] = m
            bond_u[e] = a0 + a
            bond_v[e] = a0 + b
            bond_feat[e] = mol.bond_feat[k]
            bond_word_u[e] = atom_word[a0 + a]
            bond_word_v[e] = atom_word[a0 + b]
            bond_cross[e] = int(bond_word_u[e] != bond_word_v[e])
        # boundary sites per word (any crossing physical bond endpoint)
        for idx in range(len(words)):
            w = w0 + idx
            sites = []
            for site, a in enumerate(words[idx]["atoms"]):
                crosses = any(atom_piece[nbr] != idx for nbr, _ in mol.adj[a])
                if crosses:
                    sites.append(site)
            word_n_boundary[w] = len(sites)
            word_boundary_start[w] = boundary_cursor
            for local, site in enumerate(sites):
                a = words[idx]["atoms"][site]
                atom_bnd_local[a0 + a] = local
                atom_bnd_slot[a0 + a] = boundary_cursor + local
                word_site_bnd_local[w, site] = local
            boundary_cursor += len(sites)
            n_boundary_graph[m] += len(sites)
        # word-internal non-edge pairs (template shortest path >= 2)
        for idx, word in enumerate(words):
            w = w0 + idx
            tid = word["tid"]
            nn = word["n"]
            for x in range(nn):
                for y in range(x + 1, nn):
                    d = int(dist_t[tid, x, y])
                    if d < 2:
                        continue
                    pair_chunks.append((m, a0 + word["atoms"][x], a0 + word["atoms"][y], w,
                                       min(d, 5), int(pair_orbit_t[tid, x, y])))
        # crossing physical bonds, never de-duplicated
        for k in range(len(mol.bond_u)):
            e = b0 + k
            if not bond_cross[e]:
                continue
            cross_chunks.append((m, int(bond_u[e]), int(bond_v[e]), int(bond_word_u[e]),
                                 int(bond_word_v[e]), int(atom_bnd_local[bond_u[e]]),
                                 int(atom_bnd_local[bond_v[e]]), int(atom_bnd_slot[bond_u[e]]),
                                 int(atom_bnd_slot[bond_v[e]])))

    # word atoms in word order (site order)
    flat = np.zeros((n_words, MAX_ATOMS), dtype=np.int32)
    for a in range(n_atoms):
        flat[int(atom_word[a]), int(atom_site[a])] = a
    word_ptr_atom = np.concatenate([[0], np.cumsum(word_n_atoms.astype(np.int64))]).astype(np.int64)
    word_atoms = np.concatenate([flat[w, :int(word_n_atoms[w])] for w in range(n_words)]).astype(np.int32) \
        if n_words else np.zeros(0, np.int32)

    word_ii_sites = np.zeros((n_words, MAX_ATOMS), dtype=np.int8)
    word_ii_mask = np.zeros((n_words, MAX_ATOMS), dtype=bool)
    word_bnd_sites = np.zeros((n_words, MAX_ATOMS), dtype=np.int8)
    word_bnd_mask = np.zeros((n_words, MAX_ATOMS), dtype=bool)
    for w in range(n_words):
        nn = int(word_n_atoms[w])
        internal, boundary = [], []
        base = int(word_ptr_atom[w])
        for site in range(nn):
            a = int(word_atoms[base + site])
            if atom_bnd_local[a] < 0:
                internal.append(site)
            else:
                boundary.append(site)
        for k, site in enumerate(internal):
            word_ii_sites[w, k] = site
            word_ii_mask[w, k] = True
        for k, site in enumerate(boundary):
            word_bnd_sites[w, k] = site
            word_bnd_mask[w, k] = True

    pair_fields = np.asarray(pair_chunks, dtype=np.int64) if pair_chunks else np.zeros((0, 6), np.int64)
    cross_fields = np.asarray(cross_chunks, dtype=np.int64) if cross_chunks else np.zeros((0, 9), np.int64)
    pair_mol = pair_fields[:, 0].astype(np.int32)
    cross_mol = cross_fields[:, 0].astype(np.int32)
    graph_ptr_pair = np.concatenate([[0], np.cumsum([int(np.sum(pair_mol == m)) for m in range(n_mol)])]).astype(np.int64)
    graph_ptr_cross = np.concatenate([[0], np.cumsum([int(np.sum(cross_mol == m)) for m in range(n_mol)])]).astype(np.int64)
    graph_ptr_boundary = np.concatenate([[0], np.cumsum(n_boundary_graph)]).astype(np.int64)

    out: dict[str, np.ndarray] = {
        "row_ids": np.asarray(row_ids, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.float32),
        "graph_ptr_atom": graph_ptr_atom,
        "graph_ptr_bond": graph_ptr_bond,
        "graph_ptr_word": graph_ptr_word,
        "graph_ptr_pair": graph_ptr_pair,
        "graph_ptr_cross": graph_ptr_cross,
        "graph_ptr_boundary": graph_ptr_boundary,
        "graph_n_atoms": np.asarray([r["mol"].n_atoms for r in structs], dtype=np.int32),
        "graph_n_bonds": np.asarray([len(r["mol"].bond_u) for r in structs], dtype=np.int32),
        "graph_n_words": np.asarray([len(r["words"]) for r in structs], dtype=np.int32),
        "graph_n_boundary": n_boundary_graph,
        "atom_feat": atom_feat,
        "atom_mol": atom_mol,
        "atom_word": atom_word,
        "atom_site": atom_site,
        "atom_boundary_local": atom_bnd_local,
        "atom_boundary_slot": atom_bnd_slot,
        "bond_mol": bond_mol,
        "bond_u": bond_u,
        "bond_v": bond_v,
        "bond_feat": bond_feat,
        "bond_word_u": bond_word_u,
        "bond_word_v": bond_word_v,
        "bond_cross": bond_cross,
        "word_mol": word_mol,
        "word_tmpl": word_tmpl,
        "word_n_atoms": word_n_atoms,
        "word_m": word_m,
        "word_rank": word_rank,
        "word_s": word_s,
        "word_site_orbit": word_site_orbit,
        "word_n_boundary": word_n_boundary,
        "word_boundary_start": word_boundary_start,
        "word_site_bnd_local": word_site_bnd_local,
        "word_ptr_atom": word_ptr_atom,
        "word_atoms": word_atoms,
        "word_ii_sites": word_ii_sites,
        "word_ii_mask": word_ii_mask,
        "word_bnd_sites": word_bnd_sites,
        "word_bnd_mask": word_bnd_mask,
        "pair_mol": pair_mol,
        "pair_u": pair_fields[:, 1].astype(np.int32),
        "pair_v": pair_fields[:, 2].astype(np.int32),
        "pair_word": pair_fields[:, 3].astype(np.int32),
        "pair_dist": pair_fields[:, 4].astype(np.int8),
        "pair_orbit": pair_fields[:, 5].astype(np.int32),
        "cross_mol": cross_mol,
        "cross_u": cross_fields[:, 1].astype(np.int32),
        "cross_v": cross_fields[:, 2].astype(np.int32),
        "cross_word_u": cross_fields[:, 3].astype(np.int32),
        "cross_word_v": cross_fields[:, 4].astype(np.int32),
        "cross_local_u": cross_fields[:, 5].astype(np.int8),
        "cross_local_v": cross_fields[:, 6].astype(np.int8),
        "cross_slot_u": cross_fields[:, 7].astype(np.int32),
        "cross_slot_v": cross_fields[:, 8].astype(np.int32),
    }
    out["graph_n_components"] = np.zeros(n_mol, np.int32)  # filled by caller from the raw cache
    out["graph_cycle_rank"] = np.zeros(n_mol, np.int32)
    return out


# ---------------------------------------------------------------------------
# count vector
# ---------------------------------------------------------------------------


def count_vector(struct: Mapping[str, np.ndarray], dims: Mapping[str, Sequence[int]],
                 n_templates: int) -> np.ndarray:
    """Raw per-molecule category counts (9 atom cols, 3 bond cols, word types, 4 scalars)."""
    n_mol = len(struct["row_ids"])
    atom_dims = [int(d) for d in dims["atom"]]
    bond_dims = [int(d) for d in dims["bond"]]
    atom_dim = int(sum(atom_dims))
    bond_dim = int(sum(bond_dims))
    total = atom_dim + bond_dim + n_templates + 4
    z = np.zeros((n_mol, total), dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(atom_dims)])
    atom_feat = struct["atom_feat"].astype(np.int64)
    atom_mol = struct["atom_mol"].astype(np.int64)
    for col in range(9):
        flat = atom_mol * total + offsets[col] + atom_feat[:, col]
        np.add.at(z.reshape(-1), flat, 1.0)
    base = atom_dim
    offsets_b = np.concatenate([[0], np.cumsum(bond_dims)])
    bond_feat = struct["bond_feat"].astype(np.int64)
    bond_mol = struct["bond_mol"].astype(np.int64)
    for col in range(3):
        flat = bond_mol * total + base + offsets_b[col] + bond_feat[:, col]
        np.add.at(z.reshape(-1), flat, 1.0)
    base += bond_dim
    np.add.at(z.reshape(-1), struct["word_mol"].astype(np.int64) * total + base
              + struct["word_tmpl"].astype(np.int64), 1.0)
    base += n_templates
    mol = np.arange(n_mol, dtype=np.int64)
    z[mol, base + 0] = struct["graph_n_atoms"]
    z[mol, base + 1] = struct["graph_n_bonds"]
    z[mol, base + 2] = struct["graph_n_components"]
    z[mol, base + 3] = struct["graph_cycle_rank"]
    return z


# ---------------------------------------------------------------------------
# padded gather tables (batch construction)
# ---------------------------------------------------------------------------


def pad_table(values: np.ndarray, mol: np.ndarray, n_mol: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((n_mol, width), dtype=np.int32)
    mask = np.zeros((n_mol, width), dtype=bool)
    counts = np.bincount(mol, minlength=n_mol) if len(mol) else np.zeros(n_mol, np.int64)
    if len(mol) and int(counts.max()) > width:
        raise ValueError("pad width too small")
    order = np.argsort(mol, kind="stable")
    cursor = np.zeros(n_mol, dtype=np.int64)
    for idx in order.tolist():
        m = int(mol[idx])
        out[m, cursor[m]] = int(idx)
        mask[m, cursor[m]] = True
        cursor[m] += 1
    return out, mask


def pad_tables(struct: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_mol = len(struct["row_ids"])
    out: dict[str, np.ndarray] = {}
    for name, key, ptr in (("atom", "atom_mol", "graph_ptr_atom"),
                           ("bond", "bond_mol", "graph_ptr_bond"),
                           ("word", "word_mol", "graph_ptr_word"),
                           ("pair", "pair_mol", "graph_ptr_pair"),
                           ("cross", "cross_mol", "graph_ptr_cross")):
        values = np.arange(len(struct[key]), dtype=np.int32)
        width = int(np.diff(struct[ptr]).max()) if n_mol else 1
        table, mask = pad_table(values, struct[key], n_mol, max(1, width))
        out[f"{name}_pad"] = table
        out[f"{name}_pad_mask"] = mask
    word_atoms_pad = np.zeros((len(struct["word_mol"]), MAX_ATOMS), dtype=np.int32)
    word_atoms_mask = np.zeros((len(struct["word_mol"]), MAX_ATOMS), dtype=bool)
    ptr = struct["word_ptr_atom"]
    for w in range(len(struct["word_mol"])):
        n = int(struct["word_n_atoms"][w])
        word_atoms_pad[w, :n] = struct["word_atoms"][int(ptr[w]):int(ptr[w]) + n]
        word_atoms_mask[w, :n] = True
    out["word_atoms_pad"] = word_atoms_pad
    out["word_atoms_mask"] = word_atoms_mask
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def df_by_type(struct: Mapping[str, np.ndarray], n_templates: int) -> np.ndarray:
    pairs = np.unique(np.stack([struct["word_mol"].astype(np.int64),
                                struct["word_tmpl"].astype(np.int64)], axis=1), axis=0)
    df = np.zeros(n_templates, dtype=np.int64)
    for _, tid in pairs.tolist():
        df[int(tid)] += 1
    return df


def prepare_all() -> dict[str, Any]:
    cfg = C.load_config()
    split = C.read_json(C.analysis_dir() / "split_freeze.json")
    freeze = C.old_freeze()
    payload = C.dataset_cache(cfg["dataset"])
    molecules = C.W.load_molecules(payload)
    table, history = load_word_table(freeze)
    n_templates = len(table.templates)      # the frozen 148-type vocabulary
    orbit = orbit_tables(table)
    dims = freeze["official_dims"]

    sides = {"fit": split["fit_row_ids"], "valid": split["valid_row_ids"], "h": split["h_row_ids"]}
    structs = {name: graph_structures(molecules, rows, table, history) for name, rows in sides.items()}
    # replay may intern candidate unions that were never applied; every *applied*
    # word keeps one of the frozen tids, which is asserted here.
    for name, recs in structs.items():
        for rec in recs:
            for word in rec["words"]:
                if int(word["tid"]) >= n_templates:
                    raise ValueError("replay produced a word outside the frozen vocabulary")
    n_interned_replay = int(len(table.templates) - n_templates)
    # fit document frequency (distinct Fit molecules per word type)
    df_pairs = set()
    for m, rec in enumerate(structs["fit"]):
        for word in rec["words"]:
            df_pairs.add((m, word["tid"]))
    df = np.zeros(n_templates, dtype=np.int64)
    for _, tid in sorted(df_pairs):
        df[int(tid)] += 1
    smoothing = float(cfg["operator"]["df_smoothing"])
    s_type = np.sqrt(df / (df + smoothing)).astype(np.float32)

    labels_raw = np.asarray(payload["labels"], dtype=np.float32)
    graph_written = {}
    for name, rows in sides.items():
        rows = np.asarray(rows, dtype=np.int64)
        labels = np.zeros(len(rows), dtype=np.float32)
        if name != "h":
            labels = labels_raw[rows]
        packed = build_split(structs[name], rows, labels, orbit, s_type, dims, n_templates)
        packed["graph_n_components"] = np.asarray(payload["graph_n_components"], dtype=np.int32)[rows]
        packed["graph_cycle_rank"] = np.asarray(payload["graph_cycle_rank"], dtype=np.int32)[rows]
        packed["z_count"] = count_vector(packed, dims, n_templates)
        graph_written[name] = packed

    record = {
        "protocol_id": cfg["protocol_id"],
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "split_sha256": split["split_sha256"],
        "old_freeze_sha256": C.sha256_file(C.OLD_FREEZE),
        "official_cache_sha256": payload["cache_sha256"],
        "n_templates": int(n_templates),
        "n_interned_during_replay": int(n_interned_replay),
        "n_multi_atom_words": int(sum(1 for t in table.templates[:n_templates] if t.n > 1)),
        "n_site_orbits": int(orbit["n_site_orbits"]),
        "n_pair_orbits": int(orbit["n_pair_orbits"]),
        "df_fit": {str(t): int(v) for t, v in enumerate(df.tolist())},
        "s_type": [float(x) for x in s_type.tolist()],
        "dims": dims,
        "splits": {},
        "labels_in_h": False,
        "n_nonedge_feature_dim": NONEDGE_FEAT_DIM,
    }
    data = C.data_dir()
    hashes = {}
    for name, packed in graph_written.items():
        path = data / f"wo_struct_{name}.npz"
        payload_arrays = {k: np.asarray(v) for k, v in packed.items()}
        payload_arrays.update({f"pad/{k}": np.asarray(v) for k, v in pad_tables(packed).items()})
        C.save_npz_atomic(path, payload_arrays)
        digest = C.content_sha256(payload_arrays)
        hashes[name] = digest
        record["splits"][name] = {
            "path": str(path.relative_to(C.ROUND_DIR)),
            "sha256": digest,
            "n_molecules": int(len(packed["row_ids"])),
            "n_atoms": int(len(packed["atom_mol"])),
            "n_bonds": int(len(packed["bond_mol"])),
            "n_words": int(len(packed["word_mol"])),
            "n_pairs": int(len(packed["pair_mol"])),
            "n_cross": int(len(packed["cross_mol"])),
            "n_boundary": int(packed["graph_ptr_boundary"][-1]),
            "max_atoms": int(packed["graph_n_atoms"].max()),
            "max_boundary": int(np.diff(packed["graph_ptr_boundary"]).max()),
            "max_words": int(packed["graph_n_words"].max()),
            "max_pairs": int(np.diff(packed["graph_ptr_pair"]).max()),
            "max_cross": int(np.diff(packed["graph_ptr_cross"]).max()),
        }
    # template table for orbit lookups
    site_orbit = orbit["site_orbit"]
    pair_orbit = orbit["pair_orbit"]
    site_type = np.zeros(int(orbit["n_site_orbits"]), dtype=np.int32)
    pair_type = np.zeros(int(orbit["n_pair_orbits"]), dtype=np.int32)
    for tid in range(n_templates):
        tmpl = table.templates[tid]
        for a in range(tmpl.n):
            site_type[int(site_orbit[tid, a])] = tid
        for a in range(tmpl.n):
            for b in range(a + 1, tmpl.n):
                pair_type[int(pair_orbit[tid, a, b])] = tid
    tmpl_payload = {f"orbit/{k}": np.asarray(v) for k, v in orbit.items()}
    tmpl_payload["orbit/site_type"] = site_type
    tmpl_payload["orbit/pair_type"] = pair_type
    tmpl_payload["tmpl/n"] = np.asarray([t.n for t in table.templates[:n_templates]],
                                        dtype=np.int32)
    tmpl_payload["tmpl/df_fit"] = df.astype(np.int32)
    tmpl_payload["tmpl/s_type"] = s_type.astype(np.float32)
    tmpl_payload["tmpl/history"] = np.asarray(history, dtype=np.int32)
    tmpl_payload["tmpl/is_multi"] = np.asarray([t.n > 1 for t in table.templates[:n_templates]],
                                               dtype=bool)
    path = data / "wo_templates.npz"
    C.save_npz_atomic(path, tmpl_payload)
    record["template_path"] = str(path.relative_to(C.ROUND_DIR))
    record["template_sha256"] = C.content_sha256(tmpl_payload)

    # Fit-only target scaler and n0
    fit_labels = labels_raw[np.asarray(sides["fit"], dtype=np.int64)]
    scaler = {"mean": float(fit_labels.mean()), "std": float(fit_labels.std(ddof=0))}
    n0 = float(np.mean([r["mol"].n_atoms for r in structs["fit"]]))
    record["scaler"] = scaler
    record["n0"] = n0
    record["z_dim"] = int(graph_written["fit"]["z_count"].shape[1])
    record["content_sha256"] = C.sha256_json(hashes)
    C.write_json(C.analysis_dir() / "structure_record.json", record)
    return record


def cmd_prepare(args) -> None:
    record = prepare_all()
    summary = {
        "n_templates": record["n_templates"],
        "n_site_orbits": record["n_site_orbits"],
        "n_pair_orbits": record["n_pair_orbits"],
        "z_dim": record["z_dim"],
        "scaler": record["scaler"],
        "n0": record["n0"],
        "splits": {k: {kk: vv for kk, vv in v.items() if kk in
                       ("n_molecules", "n_atoms", "n_words", "n_pairs", "n_cross", "n_boundary",
                        "max_atoms", "max_boundary")} for k, v in record["splits"].items()},
    }
    print(C.jsonable(summary))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").set_defaults(func=cmd_prepare)
    ns = parser.parse_args()
    ns.func(ns)
