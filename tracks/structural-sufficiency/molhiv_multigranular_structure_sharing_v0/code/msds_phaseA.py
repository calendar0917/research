"""Phase A driver: exact multigranular structure discovery over all 41,127 molecules.

Outputs (all under ``data/``, git-ignored because they are large label-free caches):

``phaseA_instances.npz``
    per-molecule sparse instance counts at both granularities, plus the type
    tables (canonical keys, family, FULL->COARSE parent) and example instances.

``phaseA_run.json``
    timings, worker count, hashes, and the enumeration/consistency checks that
    ran during discovery.

The vocabulary is *not* selected here: :mod:`msds_vocab` freezes it from Train
statistics only, in a separate step, and Valid/Test are never consulted.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import msds_common as C
import msds_data as D
import msds_enum as E

_MOLECULES: D.Molecules | None = None


def _init_worker() -> None:
    global _MOLECULES
    _MOLECULES = D.load_molecules()


def _adjacency_map(molecules: D.Molecules, g: int) -> dict[int, dict[int, tuple]]:
    bu, bv, bf = molecules.bonds(g)
    out: dict[int, dict[int, tuple]] = {}
    for u, v, f in zip(bu.tolist(), bv.tolist(), bf.tolist()):
        t = tuple(int(x) for x in f)
        out.setdefault(u, {})[v] = t
        out.setdefault(v, {})[u] = t
    return out


def _process_chunk(graph_ids: Sequence[int]) -> dict[str, Any]:
    """Canonicalise every Phase A instance of the given molecules.

    Returns local (chunk-scoped) type ids so no global dictionary is shared
    between workers; the parent process merges them.
    """
    mol = _MOLECULES
    assert mol is not None
    full_index: dict[bytes, int] = {}
    coarse_index: dict[bytes, int] = {}
    full_keys: list[bytes] = []
    coarse_keys: list[bytes] = []
    full_family: list[int] = []
    coarse_family: list[int] = []
    full_n_edges: list[int] = []
    coarse_n_edges: list[int] = []
    out_mol: list[int] = []
    out_full: list[int] = []
    out_coarse: list[int] = []
    examples: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    n_instances_family = np.zeros((len(graph_ids), len(C.FAMILIES)), dtype=np.int32)

    def _id(
        index: dict[bytes, int],
        keys: list[bytes],
        families: list[int],
        edges: list[int],
        key: tuple,
        fam: int,
        n_internal_bonds: int,
    ) -> int:
        raw = repr(key).encode()
        got = index.get(raw)
        if got is None:
            got = len(keys)
            index[raw] = got
            keys.append(raw)
            families.append(fam)
            edges.append(n_internal_bonds)
        return got

    for row, gid in enumerate(graph_ids):
        gid = int(gid)
        n = int(mol.n_atoms[gid])
        bu, bv, _bf = mol.bonds(gid)
        instances = E.instances_for_molecule(n, bu, bv)
        adj = _adjacency_map(mol, gid)
        atom_feat = mol.atoms(gid)
        for fam, nodes in instances:
            n_instances_family[row, fam] += 1
            bond_attr = {
                (a, b): adj[a][b]
                for a in nodes
                for b in nodes
                if b > a and b in adj.get(a, {})
            }
            kf = E.canonical_key(nodes, atom_feat, bond_attr, tuple(range(9)), (0, 1, 2))
            kc = E.canonical_key(
                nodes, atom_feat, bond_attr, C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS
            )
            n_bonds = len(bond_attr)
            fid = _id(full_index, full_keys, full_family, full_n_edges, kf, fam, n_bonds)
            cid = _id(coarse_index, coarse_keys, coarse_family, coarse_n_edges, kc, fam, n_bonds)
            out_mol.append(gid)
            out_full.append(fid)
            out_coarse.append(cid)
            bucket = examples.setdefault(fid, [])
            if len(bucket) < 6:
                bucket.append((gid, tuple(int(x) for x in nodes)))

    return {
        "graph_ids": np.asarray(graph_ids, dtype=np.int64),
        "mol": np.asarray(out_mol, dtype=np.int32),
        "full": np.asarray(out_full, dtype=np.int32),
        "coarse": np.asarray(out_coarse, dtype=np.int32),
        "full_keys": full_keys,
        "coarse_keys": coarse_keys,
        "full_family": np.asarray(full_family, dtype=np.int8),
        "coarse_family": np.asarray(coarse_family, dtype=np.int8),
        "full_n_edges": np.asarray(full_n_edges, dtype=np.int8),
        "coarse_n_edges": np.asarray(coarse_n_edges, dtype=np.int8),
        "examples": examples,
        "n_instances_family": n_instances_family,
    }


def _csr(rows: np.ndarray, cols: np.ndarray, n_rows: int, n_cols: int) -> dict[str, np.ndarray]:
    from scipy.sparse import coo_matrix

    mat = coo_matrix(
        (np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n_rows, n_cols)
    ).tocsr()
    mat.sort_indices()
    return {"indptr": mat.indptr.astype(np.int64), "indices": mat.indices.astype(np.int32),
            "data": mat.data.astype(np.int32)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A: exact multigranular structure discovery")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N molecules")
    ap.add_argument("--out", default=str(C.CACHE / "phaseA_instances.npz"))
    args = ap.parse_args()

    C.ensure_dirs()
    t_start_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.perf_counter()
    molecules = D.load_molecules()
    n_mol = molecules.n_molecules if args.limit <= 0 else min(args.limit, molecules.n_molecules)
    ids = list(range(n_mol))
    chunks = [ids[i: i + args.chunk] for i in range(0, len(ids), args.chunk)]
    print(f"[phaseA] {n_mol} molecules in {len(chunks)} chunks, {args.workers} workers", flush=True)

    results: list[dict[str, Any]] = []
    t_canon = 0.0
    t0 = time.perf_counter()
    if args.workers <= 1:
        _init_worker()
        for i, chunk in enumerate(chunks):
            results.append(_process_chunk(chunk))
            if (i + 1) % 10 == 0:
                print(f"[phaseA] chunk {i + 1}/{len(chunks)} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
            for i, res in enumerate(pool.imap(_process_chunk, chunks, chunksize=1)):
                results.append(res)
                if (i + 1) % 10 == 0:
                    print(f"[phaseA] chunk {i + 1}/{len(chunks)} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    t_canon = time.perf_counter() - t0
    print(f"[phaseA] canonicalisation done in {t_canon:.1f}s", flush=True)

    # ---------------------------------------------------------------- merge ids
    full_map: dict[bytes, int] = {}
    coarse_map: dict[bytes, int] = {}
    full_keys: list[bytes] = []
    coarse_keys: list[bytes] = []
    full_family = np.zeros(0, dtype=np.int8)
    coarse_family = np.zeros(0, dtype=np.int8)
    fam_full: list[int] = []
    fam_coarse: list[int] = []
    edges_full: list[int] = []
    edges_coarse: list[int] = []
    examples: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    mol_all: list[np.ndarray] = []
    full_all: list[np.ndarray] = []
    coarse_all: list[np.ndarray] = []
    n_instances_family = np.zeros((n_mol, len(C.FAMILIES)), dtype=np.int32)

    for res in results:
        n_instances_family[res["graph_ids"], :] = res["n_instances_family"]
        remap_full = np.empty(len(res["full_keys"]), dtype=np.int32)
        for i, key in enumerate(res["full_keys"]):
            gid = full_map.get(key)
            if gid is None:
                gid = len(full_keys)
                full_map[key] = gid
                full_keys.append(key)
                fam_full.append(int(res["full_family"][i]))
                edges_full.append(int(res["full_n_edges"][i]))
            remap_full[i] = gid
        remap_coarse = np.empty(len(res["coarse_keys"]), dtype=np.int32)
        for i, key in enumerate(res["coarse_keys"]):
            gid = coarse_map.get(key)
            if gid is None:
                gid = len(coarse_keys)
                coarse_map[key] = gid
                coarse_keys.append(key)
                fam_coarse.append(int(res["coarse_family"][i]))
                edges_coarse.append(int(res["coarse_n_edges"][i]))
            remap_coarse[i] = gid
        for fid, bucket in res["examples"].items():
            gid = int(remap_full[fid])
            slot = examples.setdefault(gid, [])
            for item in bucket:
                if len(slot) < 6:
                    slot.append(item)
        mol_all.append(res["mol"])
        full_all.append(remap_full[res["full"]])
        coarse_all.append(remap_coarse[res["coarse"]])
        del res

    mol_ids = np.concatenate(mol_all)
    full_ids = np.concatenate(full_all)
    coarse_ids = np.concatenate(coarse_all)
    n_full, n_coarse = len(full_keys), len(coarse_keys)
    print(f"[phaseA] merged: {mol_ids.size} instances, {n_full} FULL types, {n_coarse} COARSE types",
          flush=True)

    # ---------------------------------------------- FULL -> COARSE parent consistency
    parent = np.full(n_full, -1, dtype=np.int32)
    conflict = 0
    for f, c in zip(full_ids.tolist(), coarse_ids.tolist()):
        if parent[f] < 0:
            parent[f] = c
        elif parent[f] != c:
            conflict += 1
    missing_parent = int((parent < 0).sum())
    if conflict or missing_parent:
        raise SystemExit(f"FULL->COARSE projection inconsistent: conflicts={conflict} missing={missing_parent}")

    # per-molecule counts of every instance incl. types that later fail the filter
    full_csr = _csr(mol_ids.astype(np.int64), full_ids.astype(np.int64), n_mol, n_full)
    coarse_csr = _csr(mol_ids.astype(np.int64), coarse_ids.astype(np.int64), n_mol, n_coarse)

    # ---------------------------------------------- structural sanity: sizes / edges
    sizes = np.array([C.FAMILY_SIZE[C.FAMILIES[i]] for i in fam_full], dtype=np.int8)
    n_edges = np.asarray(edges_full, dtype=np.int8)
    cycle_mask = np.array([C.FAMILIES[i].startswith("cycle") for i in fam_full])
    ok_edges = bool(
        np.all(n_edges[~cycle_mask] >= sizes[~cycle_mask] - 1)
        and np.all(n_edges[cycle_mask] == sizes[cycle_mask])
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        n_molecules=np.int64(n_mol),
        full_indptr=full_csr["indptr"],
        full_indices=full_csr["indices"],
        full_data=full_csr["data"],
        coarse_indptr=coarse_csr["indptr"],
        coarse_indices=coarse_csr["indices"],
        coarse_data=coarse_csr["data"],
        full_keys=np.asarray(full_keys, dtype=object),
        coarse_keys=np.asarray(coarse_keys, dtype=object),
        full_family=np.asarray(fam_full, dtype=np.int8),
        coarse_family=np.asarray(fam_coarse, dtype=np.int8),
        full_n_edges=np.asarray(edges_full, dtype=np.int8),
        coarse_n_edges=np.asarray(edges_coarse, dtype=np.int8),
        full_parent=parent,
        n_instances_family=n_instances_family,
        examples=np.asarray([examples.get(i, []) for i in range(n_full)], dtype=object),
    )
    report = {
        "round_id": C.ROUND_ID,
        "phase": "A: exact multigranular structure discovery",
        "started_utc": t_start_utc,
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_molecules": n_mol,
        "n_instances": int(mol_ids.size),
        "n_instances_by_family": {
            fam: int(n_instances_family[:, i].sum()) for i, fam in enumerate(C.FAMILIES)
        },
        "instances_per_molecule": {
            "mean": float(n_instances_family.sum(axis=1).mean()),
            "median": float(np.median(n_instances_family.sum(axis=1))),
            "min": int(n_instances_family.sum(axis=1).min()),
            "max": int(n_instances_family.sum(axis=1).max()),
        },
        "n_full_types": n_full,
        "n_coarse_types": n_coarse,
        "n_full_types_by_family": {
            fam: int((np.asarray(fam_full) == i).sum()) for i, fam in enumerate(C.FAMILIES)
        },
        "n_coarse_types_by_family": {
            fam: int((np.asarray(fam_coarse) == i).sum()) for i, fam in enumerate(C.FAMILIES)
        },
        "full_to_coarse": {
            "conflicts": int(conflict),
            "types_without_parent": missing_parent,
            "distinct_parents": int(np.unique(parent).size),
        },
        "instance_edge_count_check": {
            "families_approx_tree_or_cycle": bool(ok_edges),
            "note": "size-k instances have >= k-1 internal bonds; cycle-k instances have exactly k",
        },
        "workers": args.workers,
        "chunk_size": args.chunk,
        "seconds_canonicalisation": float(t_canon),
        "seconds_total": float(time.perf_counter() - t_start),
        "output": C.rel(out),
        "output_sha256": C.sha256_file(out),
        "output_size_bytes": int(out.stat().st_size),
        "git": C.git_state(),
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "phaseA_run.json", report)
    print(f"[phaseA] wrote {C.rel(out)} ({out.stat().st_size / 1e6:.1f} MB) and phaseA_run.json", flush=True)
    print(f"[phaseA] total {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
