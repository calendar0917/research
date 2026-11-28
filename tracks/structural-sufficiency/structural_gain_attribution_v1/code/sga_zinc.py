"""Phase C: ZINC data, fixed representation ``Z`` and the exact structure block ``Z-CF``.

Data: PyG ``ZINC(subset=True)`` official split (10 000 / 1 000 / 1 000), loaded
directly from ``data/ZINC/subset/processed/*.pt`` (a plain tensor bundle, no PyG
``Data`` object is ever batched, so no attribute whose name contains ``index`` can
be given a cumulative node offset).

Representations (PROTOCOL.md §8):

``Z``   n_atoms, n_bonds, raw node-class counts, raw edge-class counts, and all-atom
        rooted **typed** WL counts for rounds 0/1/2/3 with Train document-frequency
        top-2048 per round.  Count columns are divided by ``sqrt(n_atoms)``.
``Z-CF`` exact enumeration of the predecessor round (all connected induced 2/3/4-node
        subgraphs, all chordless 5/6-cycles), ``COARSE`` = topology only,
        ``FULL`` = topology + the dataset's own node/edge classes, Train-only
        document-frequency gates and per-family caps, no scaffold gate (ZINC has no
        reliable scaffold identity).
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

import sga_common as K

K.add_prev_code_path()

import msds_enum as ENUM  # noqa: E402  (predecessor round: exact enumeration + canonical keys)

FAMILIES = ("size-2", "size-3", "size-4", "cycle-5", "cycle-6")
FAMILY_SIZE = {"size-2": 2, "size-3": 3, "size-4": 4, "cycle-5": 5, "cycle-6": 6}
COARSE_MIN_DF = 20
COARSE_MAX_PER_FAMILY = 256
FULL_MIN_DF = 5
FULL_MAX_PER_FAMILY = 1024
WL_TOPK = 2048
WL_ROUNDS = 3


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_split(name: str) -> dict[str, Any]:
    import torch

    path = K.ZINC_DIR / "subset" / "processed" / f"{'val' if name == 'valid' else name}.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    data, slices = d[0], d[1]
    x = data["x"].reshape(-1).to(torch.int64).numpy()
    ea = data["edge_attr"].reshape(-1).to(torch.int64).numpy()
    ei = data["edge_index"].to(torch.int64).numpy()
    y = data["y"].reshape(-1).to(torch.float64).numpy()
    x_off = slices["x"].to(torch.int64).numpy()
    e_off = slices["edge_index"].to(torch.int64).numpy()
    n = int(x_off.size - 1)
    graphs = []
    edge_class_values = set()
    node_class_values = set()
    for g in range(n):
        lo, hi = int(x_off[g]), int(x_off[g + 1])
        elo, ehi = int(e_off[g]), int(e_off[g + 1])
        src = ei[0, elo:ehi]
        dst = ei[1, elo:ehi]
        u = np.minimum(src, dst)
        v = np.maximum(src, dst)
        key = u.astype(np.int64) * 1000 + v
        uniq, idx = np.unique(key, return_index=True)
        bu = (uniq // 1000).astype(np.int32)
        bv = (uniq % 1000).astype(np.int32)
        bf = ea[elo:ehi][idx].astype(np.int16)
        order = np.lexsort((bv, bu))
        graphs.append({
            "x": x[lo:hi].astype(np.int16),
            "y": float(y[g]) if y.size > g else float("nan"),
            "bu": bu[order], "bv": bv[order], "bf": bf[order],
        })
        node_class_values.update(int(c) for c in np.unique(x[lo:hi]))
        edge_class_values.update(int(c) for c in bf)
    return {
        "name": name, "graphs": graphs, "y": np.asarray([g["y"] for g in graphs], dtype=np.float64),
        "node_classes": sorted(node_class_values), "edge_classes": sorted(edge_class_values),
        "path": str(path),
        "file_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
        "n_graphs": n,
        "n_raw_edge_records": int(ei.shape[1]),
        "physical_bonds": int(sum(g["bu"].size for g in graphs)),
    }


def load_all() -> dict[str, Any]:
    out = {name: load_split(name) for name in ("train", "valid", "test")}
    node_classes = sorted(set().union(*[set(v["node_classes"]) for v in out.values()]))
    edge_classes = sorted(set().union(*[set(v["edge_classes"]) for v in out.values()]))
    out["node_classes"] = node_classes
    out["edge_classes"] = edge_classes
    return out


# ---------------------------------------------------------------------------
# Z: size + raw class counts + typed rooted WL counts
# ---------------------------------------------------------------------------


def _digest(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=16, person=b"zinc-typedwl-1").digest()


def _bytes(values: Sequence[int]) -> bytes:
    a = np.asarray(values, dtype=np.int16).reshape(-1)
    return struct.pack("<H", len(a)) + a.tobytes()


def typed_rooted_signatures(g: dict[str, Any], max_round: int = WL_ROUNDS) -> list[list[bytes]]:
    """Rooted typed WL refinement over the dataset's own node/edge classes."""
    n = int(g["x"].size)
    nbrs: list[list[tuple[int, bytes]]] = [[] for _ in range(n)]
    for u, v, c in zip(g["bu"].tolist(), g["bv"].tolist(), g["bf"].tolist()):
        eb = _bytes([c])
        nbrs[u].append((v, eb))
        nbrs[v].append((u, eb))
    labels = [_digest(b"N" + _bytes([int(c)])) for c in g["x"].tolist()]
    out = [list(labels)]
    for r in range(1, max_round + 1):
        nxt = []
        for u in range(n):
            messages = sorted(eb + labels[v] for v, eb in nbrs[u])
            payload = b"R" + bytes([r]) + labels[u] + struct.pack("<H", len(messages)) + b"".join(messages)
            nxt.append(_digest(payload))
        labels = nxt
        out.append(list(labels))
    return out


def build_Z(data: dict[str, Any], train_idx: np.ndarray) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Return the ``Z`` design (count columns already divided by sqrt(n_atoms))."""
    node_classes = data["node_classes"]
    edge_classes = data["edge_classes"]
    node_pos = {c: i for i, c in enumerate(node_classes)}
    edge_pos = {c: i for i, c in enumerate(edge_classes)}
    graphs: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        graphs.extend(data[split]["graphs"])
    n = len(graphs)
    train_set = set(int(x) for x in train_idx.tolist())
    # ---- WL vocabulary from Train only
    id_of: dict[bytes, int] = {}
    train_df: list[int] = []
    round_of: list[int] = []
    per_mol: list[dict[bytes, int]] = []
    for i, g in enumerate(graphs):
        sigs = typed_rooted_signatures(g)
        counts: dict[bytes, int] = {}
        for r, labels in enumerate(sigs):
            prefix = bytes([r])
            for lab in labels:
                k = prefix + lab
                counts[k] = counts.get(k, 0) + 1
        per_mol.append(counts)
        for k in counts:
            if k not in id_of:
                id_of[k] = len(id_of)
                train_df.append(0)
                round_of.append(k[0])
        if i in train_set:
            for k in counts:
                train_df[id_of[k]] += 1
    df = np.asarray(train_df, dtype=np.int64)
    keys = list(id_of.keys())
    selected_ids: list[int] = []
    for r in range(WL_ROUNDS + 1):
        cand = [j for j in range(df.size) if round_of[j] == r]
        cand.sort(key=lambda j: (-int(df[j]), keys[j]))
        selected_ids.extend(cand[:WL_TOPK])
    selected = np.asarray(sorted(selected_ids), dtype=np.int64)
    col_of: dict[int, int] = {}
    for c, j in enumerate(selected.tolist()):
        col_of[j] = c
    n_wl = selected.size
    width = 2 + len(node_classes) + len(edge_classes) + n_wl
    data_rows: list[int] = []
    data_cols: list[int] = []
    data_vals: list[float] = []
    for i, g in enumerate(graphs):
        inv = 1.0 / np.sqrt(max(int(g["x"].size), 1))
        na = int(g["x"].size)
        nb = int(g["bu"].size)
        data_rows.append(i); data_cols.append(0); data_vals.append(float(na))
        data_rows.append(i); data_cols.append(1); data_vals.append(float(nb))
        for c in g["x"].tolist():
            data_rows.append(i); data_cols.append(2 + node_pos[int(c)]); data_vals.append(float(inv))
        for c in g["bf"].tolist():
            data_rows.append(i)
            data_cols.append(2 + len(node_classes) + edge_pos[int(c)])
            data_vals.append(float(inv))
        base = 2 + len(node_classes) + len(edge_classes)
        for k, cnt in per_mol[i].items():
            c = col_of.get(id_of[k])
            if c is not None:
                data_rows.append(i); data_cols.append(base + c); data_vals.append(float(cnt) * inv)
    mat = sp.csr_matrix(
        (np.asarray(data_vals, dtype=np.float32),
         (np.asarray(data_rows, dtype=np.int64), np.asarray(data_cols, dtype=np.int64))),
        shape=(n, width),
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    meta = {
        "columns": {
            "n_atoms": 1, "n_bonds": 1,
            "node_class_counts": len(node_classes), "edge_class_counts": len(edge_classes),
            "typed_wl_counts": int(n_wl),
        },
        "node_classes": node_classes,
        "edge_classes": edge_classes,
        "n_signatures_all": int(df.size),
        "n_signatures_by_round_all": {str(r): int(sum(1 for x in round_of if x == r))
                                      for r in range(WL_ROUNDS + 1)},
        "n_selected_by_round": {str(r): int(sum(1 for j in selected_ids if round_of[j] == r))
                                for r in range(WL_ROUNDS + 1)},
        "wl_topk_per_round": WL_TOPK,
        "scaling": "counts / sqrt(n_atoms); n_atoms and n_bonds kept raw",
        "width": int(width),
        "nnz": int(mat.nnz),
    }
    return mat, meta


# ---------------------------------------------------------------------------
# Z-CF: exact typed subgraph enumeration (predecessor scheme, ZINC attributes)
# ---------------------------------------------------------------------------


def _type_digest(key: tuple) -> int:
    """64-bit digest of a canonical key: dictionary key + deterministic tie-break."""
    return int.from_bytes(
        hashlib.blake2b(repr(key).encode(), digest_size=8, person=b"zinc-type-1").digest(), "big"
    )


def _graph_types(g: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-graph unique (family, FULL digest, COARSE digest) rows with their counts.

    Returning 64-bit digests instead of the nested canonical keys keeps the worker
    payload small (the exact keys are reconstructed only inside the worker, where
    they are needed).  Type identity is still the complete canonical invariant; the
    digest is its 64-bit projection, used as a dictionary key and as the
    deterministic tie-break of the Train-frequency ranking.
    """
    n = int(g["x"].size)
    atom_feat = g["x"].reshape(-1, 1).astype(np.int16)
    bond_attr = {(int(u), int(v)): (int(c),) for u, v, c in
                 zip(g["bu"].tolist(), g["bv"].tolist(), g["bf"].tolist())}
    adj = ENUM.adjacency(n, g["bu"], g["bv"])
    fams: list[int] = []
    fulls: list[int] = []
    coarses: list[int] = []
    counts: dict[tuple[int, int, int], int] = {}
    for fam_idx, family in enumerate(FAMILIES):
        k = FAMILY_SIZE[family]
        if family.startswith("size"):
            subs = ENUM.enumerate_family_connected(adj, n, k)
        else:
            subs = ENUM.enumerate_induced_cycles(adj, n, k)
        for nodes in subs:
            nodes = tuple(sorted(nodes))
            internal = {}
            for a in nodes:
                for b in nodes:
                    if (a, b) in bond_attr:
                        internal[(a, b)] = bond_attr[(a, b)]
            fh = _type_digest(ENUM.canonical_key(nodes, atom_feat, internal, (0,), (0,)))
            ch = _type_digest(ENUM.canonical_key(nodes, atom_feat, internal, (), ()))
            key = (fam_idx, fh, ch)
            counts[key] = counts.get(key, 0) + 1
    keys = sorted(counts)
    fams = np.asarray([k[0] for k in keys], dtype=np.uint8)
    fulls = np.asarray([k[1] for k in keys], dtype=np.uint64)
    coarses = np.asarray([k[2] for k in keys], dtype=np.uint64)
    cnts = np.asarray([counts[k] for k in keys], dtype=np.int32)
    return fams, fulls, coarses, cnts


def _worker(args: tuple[int, dict[str, Any]]):
    i, g = args
    fams, fulls, coarses, cnts = _graph_types(g)
    return i, fams, fulls, coarses, cnts


def build_ZCF(
    data: dict[str, Any], train_idx: np.ndarray, n_atoms_all: np.ndarray, workers: int = 0
) -> tuple[sp.csr_matrix, sp.csr_matrix, dict[str, Any]]:
    """Exact enumeration -> Train-only vocabulary -> COARSE and FULL count blocks."""
    import os

    graphs: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        graphs.extend(data[split]["graphs"])
    n = len(graphs)
    if workers <= 0:
        workers = min(16, os.cpu_count() or 1)
    print(f"[zinc-cf] enumerating {n} graphs with {workers} workers", flush=True)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_worker, list(enumerate(graphs)), chunksize=16))
    else:
        results = [_worker((i, g)) for i, g in enumerate(graphs)]
    results.sort(key=lambda x: x[0])

    full_id: dict[int, int] = {}
    coarse_id: dict[int, int] = {}
    full_family: list[int] = []
    coarse_family: list[int] = []
    full_parent: list[int] = []
    conflicts = 0
    rows_f: list[np.ndarray] = []
    cols_f: list[np.ndarray] = []
    vals_f: list[np.ndarray] = []
    rows_c: list[np.ndarray] = []
    cols_c: list[np.ndarray] = []
    vals_c: list[np.ndarray] = []
    total_instances = 0
    full_keys_digest: list[int] = []
    coarse_keys_digest: list[int] = []
    for i, fams, fulls, coarses, cnts in results:
        ti = np.empty(fulls.size, dtype=np.int64)
        ci = np.empty(coarses.size, dtype=np.int64)
        for j in range(fulls.size):
            fd = int(fulls[j])
            cd = int(coarses[j])
            t = full_id.get(fd)
            if t is None:
                t = len(full_id)
                full_id[fd] = t
                full_family.append(int(fams[j]))
                full_keys_digest.append(fd)
            c = coarse_id.get(cd)
            if c is None:
                c = len(coarse_id)
                coarse_id[cd] = c
                coarse_family.append(int(fams[j]))
                coarse_keys_digest.append(cd)
            if len(full_parent) <= t:
                full_parent.extend([-1] * (t + 1 - len(full_parent)))
            if full_parent[t] == -1:
                full_parent[t] = c
            elif full_parent[t] != c:
                conflicts += 1
            ti[j] = t
            ci[j] = c
        rows_f.append(np.full(fulls.size, i, dtype=np.int64))
        cols_f.append(ti)
        vals_f.append(cnts.astype(np.int64))
        rows_c.append(np.full(coarses.size, i, dtype=np.int64))
        cols_c.append(ci)
        vals_c.append(cnts.astype(np.int64))
        total_instances += int(cnts.sum())
        if (i + 1) % 2000 == 0:
            print(f"[zinc-cf]   {i + 1}/{n} graphs, {len(full_id)} FULL / {len(coarse_id)} COARSE types",
                  flush=True)

    full_family_a = np.asarray(full_family, dtype=np.int64)
    coarse_family_a = np.asarray(coarse_family, dtype=np.int64)
    full_parent_a = np.asarray(full_parent[: len(full_id)], dtype=np.int64)
    full_digest = np.asarray(full_keys_digest, dtype=np.uint64)
    coarse_digest = np.asarray(coarse_keys_digest, dtype=np.uint64)
    full_raw = sp.csr_matrix((np.concatenate(vals_f), (np.concatenate(rows_f), np.concatenate(cols_f))),
                             shape=(n, len(full_id)))
    coarse_raw = sp.csr_matrix((np.concatenate(vals_c), (np.concatenate(rows_c), np.concatenate(cols_c))),
                               shape=(n, len(coarse_id)))
    train = np.asarray(train_idx, dtype=np.int64)
    c_df = np.asarray(coarse_raw[train].getnnz(axis=0), dtype=np.int64)
    f_df = np.asarray(full_raw[train].getnnz(axis=0), dtype=np.int64)
    node_count = np.asarray([g["x"].size for g in graphs], dtype=np.int64)

    c_cols: list[int] = []
    for fam in range(len(FAMILIES)):
        cand = [j for j in range(c_df.size) if coarse_family_a[j] == fam and c_df[j] >= COARSE_MIN_DF]
        cand.sort(key=lambda j: (-int(c_df[j]), int(coarse_digest[j])))
        c_cols.extend(cand[:COARSE_MAX_PER_FAMILY])
    c_col_of = {t: i for i, t in enumerate(c_cols)}
    f_cols: list[int] = []
    for fam in range(len(FAMILIES)):
        cand = [j for j in range(f_df.size)
                if full_family_a[j] == fam and f_df[j] >= FULL_MIN_DF and int(full_parent_a[j]) in c_col_of]
        cand.sort(key=lambda j: (-int(f_df[j]), int(full_digest[j])))
        f_cols.extend(cand[:FULL_MAX_PER_FAMILY])
    if conflicts:
        raise SystemExit(f"FULL->COARSE projection conflicts: {conflicts}")

    def _scaled(raw: sp.csr_matrix, cols: list[int]) -> sp.csr_matrix:
        sub = raw[:, np.asarray(cols, dtype=np.int64)].astype(np.float64)
        inv = 1.0 / np.sqrt(np.maximum(node_count.astype(np.float64), 1.0))
        out = sp.csr_matrix(sub.multiply(inv[:, None])).astype(np.float32)
        out.eliminate_zeros()
        return out

    Cm = _scaled(coarse_raw, c_cols)
    Fm = _scaled(full_raw, f_cols)
    meta = {
        "families": list(FAMILIES),
        "n_instances": int(total_instances),
        "n_instances_by_family": {
            f: int(np.asarray(coarse_raw[:, np.flatnonzero(coarse_family_a == i)].sum()).item())
            for i, f in enumerate(FAMILIES)
        },
        "n_full_types_discovered": int(len(full_id)),
        "n_coarse_types_discovered": int(len(coarse_id)),
        "n_full_types_by_family_discovered": {f: int((full_family_a == i).sum()) for i, f in enumerate(FAMILIES)},
        "n_coarse_types_by_family_discovered": {f: int((coarse_family_a == i).sum()) for i, f in enumerate(FAMILIES)},
        "filters": {
            "coarse": {"min_document_frequency_train": COARSE_MIN_DF,
                       "max_types_per_family": COARSE_MAX_PER_FAMILY},
            "full": {"min_document_frequency_train": FULL_MIN_DF,
                     "max_types_per_family": FULL_MAX_PER_FAMILY,
                     "requires_coarse_parent_retained": True},
            "scaffold_gate": "dropped: ZINC has no reliable scaffold identity (document-frequency gate only)",
            "ranking": "document_frequency desc, 64-bit digest of the canonical key asc",
        },
        "type_identity": "canonical key (complete invariant of the attributed induced subgraph); the "
                         "64-bit digest is only a dictionary key / tie-break",
        "n_selected_coarse": int(len(c_cols)),
        "n_selected_full": int(len(f_cols)),
        "n_selected_coarse_by_family": {f: int((coarse_family_a[c_cols] == i).sum()) for i, f in enumerate(FAMILIES)},
        "n_selected_full_by_family": {f: int((full_family_a[f_cols] == i).sum()) for i, f in enumerate(FAMILIES)},
        "coarse_shape": list(Cm.shape), "full_shape": list(Fm.shape),
        "coarse_nnz": int(Cm.nnz), "full_nnz": int(Fm.nnz),
        "scaling": "count / sqrt(n_atoms)",
        "full_to_coarse": {
            "rule": "deterministic projection: the instance node set + internal bonds are "
                    "re-canonicalised without node/edge attributes",
            "conflicts": int(conflicts),
            "types_without_retained_parent": int(sum(1 for t in f_cols if int(full_parent_a[t]) not in c_col_of)),
        },
    }
    return Cm, Fm, meta
