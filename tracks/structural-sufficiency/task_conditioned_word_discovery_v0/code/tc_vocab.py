"""Exact candidate enumeration, Fit-only registry freeze and feature matrices.

Everything here is a pure structure function.  The registry is defined on the
current split's Fit rows only; Dev structures are only *encoded* against that
frozen registry (an unseen Dev signature cannot create a type).

Candidate identities are the exact root-marked attributed-graph canonical
certificates of ``rice_structure.canonical_signature``.  Two granularities are
used for size 3/4:

* FULL   : all 9 atom fields / all 3 bond fields
* COARSE : atomic number, formal charge, aromaticity / bond type

Size 1/2 FULL rooted types are the low-order X0 reference, not vocabulary words.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

import tc_common as C

SIZES = (1, 2, 3, 4)
WORD_SIZES = (3, 4)
GRANS = ("FULL", "COARSE")


def _family(gran: str, size: int) -> str:
    return f"{gran}_{int(size)}"


def _empty_sig() -> np.ndarray:
    return np.zeros((0, 10), np.int64)


def _size1_signature(st: C.RS.MoleculeStruct, atom_dims: Sequence[int]) -> np.ndarray:
    label = C.RS._mixed_radix(st.atom_feat, atom_dims)  # noqa: SLF001 exact reuse
    sig = np.full((st.n_atoms, 10), -1, np.int64)
    sig[:, 0] = label
    return sig


def _signatures_for(st: C.RS.MoleculeStruct, size: int, gran: str,
                    atom_dims: Sequence[int], bond_dims: Sequence[int]) -> np.ndarray:
    if size == 1:
        if gran != "FULL":
            raise ValueError("size-1 COARSE is not used")
        return _size1_signature(st, atom_dims)
    return C.canonical_signature(st, size, gran, atom_dims, bond_dims)


def _unique_rows(sig: np.ndarray) -> np.ndarray:
    if sig.shape[0] == 0:
        return _empty_sig()
    return np.unique(sig, axis=0)


def enumerate_families(rows: Sequence[int], structs: Mapping[int, C.RS.MoleculeStruct],
                       atom_dims: Sequence[int], bond_dims: Sequence[int]
                       ) -> dict[str, tuple[dict[bytes, int], list[dict[str, Any]]]]:
    """For every family, the Fit signature -> global type id map and metadata.

    Type ids are assigned family by family in the frozen order
    ``FULL_1, FULL_2, FULL_3, COARSE_3, FULL_4, COARSE_4``; within a family they
    are sorted by canonical key bytes.
    """
    families: dict[str, tuple[dict[bytes, int], list[dict[str, Any]]]] = {}
    next_id = 0
    for gran, size in (("FULL", 1), ("FULL", 2), ("FULL", 3), ("COARSE", 3),
                       ("FULL", 4), ("COARSE", 4)):
        parts = [_signatures_for(structs[int(r)], size, gran, atom_dims, bond_dims)
                 for r in rows]
        cat = np.concatenate(parts, axis=0) if parts else _empty_sig()
        uniq = _unique_rows(cat)
        order = np.argsort([C.signature_bytes(row) for row in uniq]) if uniq.size else []
        uniq = uniq[np.asarray(order, np.int64)] if uniq.size else uniq
        mapping: dict[bytes, int] = {}
        meta: list[dict[str, Any]] = []
        for row in uniq:
            sig = [int(x) for x in row]
            mapping[C.signature_bytes(sig)] = next_id
            meta.append({
                "type_id": int(next_id),
                "size": int(size),
                "gran": str(gran),
                "signature": sig,
                "key": C.signature_key(gran, size, sig),
                "low_order": bool(size <= 2),
            })
            next_id += 1
        families[_family(gran, size)] = (mapping, meta)
    return families


def build_registry(rows_fit: Sequence[int], rows_eval: Sequence[int],
                   structs: Mapping[int, C.RS.MoleculeStruct],
                   atom_dims: Sequence[int], bond_dims: Sequence[int]
                   ) -> tuple[dict[str, Any], sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build Fit-defined type ids and per-atom counts on ``rows_fit + rows_eval``.

    Returns ``(registry, atom_counts, rows_all, atom_ptr)`` where ``atom_counts``
    is ``[n_atoms_all, n_types]`` CSR with integer occurrence counts.
    """
    rows_fit = np.asarray(rows_fit, np.int64)
    rows_eval = np.asarray(rows_eval, np.int64)
    rows_all = np.union1d(rows_fit, rows_eval).astype(np.int64)
    families = enumerate_families(rows_fit.tolist(), structs, atom_dims, bond_dims)
    n_types = sum(len(meta) for _, meta in families.values())

    atom_ptr = np.zeros(rows_all.size + 1, np.int64)
    atom_ptr[1:] = np.cumsum([int(structs[int(r)].n_atoms) for r in rows_all])
    row_to_local = {int(r): i for i, r in enumerate(rows_all.tolist())}
    n_atoms_all = int(atom_ptr[-1])

    atom_idx_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    for gran, size in (("FULL", 1), ("FULL", 2), ("FULL", 3), ("COARSE", 3),
                       ("FULL", 4), ("COARSE", 4)):
        mapping, _ = families[_family(gran, size)]
        for row in rows_all.tolist():
            st = structs[int(row)]
            sig = _signatures_for(st, size, gran, atom_dims, bond_dims)
            if sig.shape[0] == 0:
                continue
            ids = np.fromiter((mapping.get(C.signature_bytes(s), -1)
                               for s in sig), dtype=np.int64, count=sig.shape[0])
            keep = ids >= 0
            if not keep.any():
                continue
            ids = ids[keep]
            if size == 1:
                roots = np.arange(st.n_atoms, dtype=np.int64)[keep]
            else:
                roots = np.asarray(st.atoms[int(size)][:, 0], np.int64)[keep]
            base = int(atom_ptr[row_to_local[int(row)]])
            atom_idx_parts.append(roots + base)
            type_parts.append(ids)

    if atom_idx_parts:
        atom_idx = np.concatenate(atom_idx_parts).astype(np.int64)
        type_idx = np.concatenate(type_parts).astype(np.int64)
        pairs = np.stack([atom_idx, type_idx], axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        atom_counts = sparse.csr_matrix(
            (counts.astype(np.int32), (uniq[:, 0], uniq[:, 1])),
            shape=(n_atoms_all, n_types))
    else:  # pragma: no cover - only for empty inputs
        atom_counts = sparse.csr_matrix((n_atoms_all, n_types), dtype=np.int32)

    registry = {
        "n_types": int(n_types),
        "n_types_by_family": {k: int(len(v[1])) for k, v in families.items()},
        "families": {k: v[1] for k, v in families.items()},
        "rows_fit_sha256": C.array_sha256(rows_fit),
        "rows_eval_sha256": C.array_sha256(rows_eval),
        "rows_all_sha256": C.array_sha256(rows_all),
        "n_atoms_all": n_atoms_all,
    }
    return registry, atom_counts, rows_all, atom_ptr


def molecule_matrix(atom_counts: sparse.csr_matrix, rows_all: Sequence[int],
                    atom_ptr: Sequence[int], rows: Sequence[int]) -> sparse.csr_matrix:
    """Map per-atom rows to per-molecule rows for an arbitrary subset."""
    rows = np.asarray(rows, np.int64)
    local = {int(r): i for i, r in enumerate(np.asarray(rows_all, np.int64).tolist())}
    mol_rows: list[int] = []
    atom_rows: list[int] = []
    for i, row in enumerate(rows.tolist()):
        j = local[int(row)]
        lo, hi = int(atom_ptr[j]), int(atom_ptr[j + 1])
        atom_rows.extend(range(lo, hi))
        mol_rows.extend([i] * (hi - lo))
    selector = sparse.csr_matrix(
        (np.ones(len(atom_rows), np.float32), (np.asarray(mol_rows, np.int64),
                                               np.asarray(atom_rows, np.int64))),
        shape=(rows.size, atom_counts.shape[0]))
    return (selector @ atom_counts).tocsr()


def family_signatures(rows: Sequence[int], structs: Mapping[int, C.RS.MoleculeStruct],
                      atom_dims: Sequence[int], bond_dims: Sequence[int]
                      ) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for gran, size in (("FULL", 1), ("FULL", 2), ("FULL", 3), ("COARSE", 3),
                       ("FULL", 4), ("COARSE", 4)):
        parts = [_signatures_for(structs[int(r)], size, gran, atom_dims, bond_dims)
                 for r in rows]
        out[_family(gran, size)] = np.concatenate(parts, axis=0) if parts else _empty_sig()
    return out


def _column_digest(indices: np.ndarray, data: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=16, person=b"tcv0col")
    h.update(np.asarray(indices, np.int64).tobytes())
    h.update(np.asarray(data, np.int32).tobytes())
    return h.digest()


def dedupe_columns(atom_counts_fit: sparse.csc_matrix, type_ids: Sequence[int],
                   df: np.ndarray, keys: Sequence[str], sizes: Sequence[int],
                   grans: Sequence[str]) -> tuple[list[int], list[dict[str, Any]]]:
    """Group qualified candidates with identical Fit root-level count columns.

    Preference inside a group: smaller Fit document frequency, then COARSE, then
    canonical key.  Every member is recorded; only the representative is used by
    the selection/control pools.
    """
    groups: dict[bytes, list[int]] = defaultdict(list)
    for tid in type_ids:
        col = atom_counts_fit[:, int(tid)]
        groups[_column_digest(col.indices, col.data)].append(int(tid))
    representatives: list[int] = []
    records: list[dict[str, Any]] = []
    for digest, members in groups.items():
        # verify exact equality (hash is only an index)
        verified: list[list[int]] = []
        for tid in members:
            placed = False
            col = atom_counts_fit[:, tid]
            for bucket in verified:
                ref = atom_counts_fit[:, bucket[0]]
                if (np.array_equal(col.indices, ref.indices)
                        and np.array_equal(col.data, ref.data)):
                    bucket.append(tid)
                    placed = True
                    break
            if not placed:
                verified.append([tid])
        for bucket in verified:
            bucket = sorted(bucket)
            rep = min(bucket, key=lambda t: (int(df[t]), 0 if grans[t] == "COARSE" else 1,
                                             keys[t]))
            representatives.append(int(rep))
            records.append({
                "group_id": f"{digest.hex()}",
                "representative": int(rep),
                "members": [int(x) for x in bucket],
                "member_keys": [keys[int(x)] for x in bucket],
                "n_members": int(len(bucket)),
            })
    representatives = sorted(representatives)
    return representatives, records


def freeze_split(split: str, out_dir: Any = None) -> dict[str, Any]:
    """Build and persist the frozen candidate registry/features for one split."""
    from pathlib import Path

    cfg = C.load_config()
    pins = C.assert_source_pins()
    bundle = C.split_bundle(split)
    rows_fit = np.asarray(bundle.fit_rows, np.int64)
    rows_eval = np.asarray(bundle.eval_rows, np.int64)
    atom_dims, bond_dims = C.atom_bond_dims()
    scaffold_by_row = C.scaffold_map(split)
    scaffold_fit = {int(r): scaffold_by_row[str(int(r))] for r in rows_fit.tolist()}

    structs = {int(r): C.build_struct(bundle, int(r))
               for r in np.union1d(rows_fit, rows_eval).tolist()}
    registry, atom_counts, rows_all, atom_ptr = build_registry(
        rows_fit, rows_eval, structs, atom_dims, bond_dims)
    types: list[dict[str, Any]] = []
    for family, meta in registry["families"].items():
        types.extend(meta)
    types = sorted(types, key=lambda x: int(x["type_id"]))
    n_types = len(types)
    keys = [str(t["key"]) for t in types]
    sizes = np.asarray([int(t["size"]) for t in types], np.int64)
    grans = [str(t["gran"]) for t in types]

    # per-atom counts restrict to Fit atoms for the frozen statistics
    fit_atom_mask = np.zeros(atom_counts.shape[0], bool)
    local_all = {int(r): i for i, r in enumerate(rows_all.tolist())}
    for row in rows_fit.tolist():
        j = local_all[int(row)]
        fit_atom_mask[int(atom_ptr[j]):int(atom_ptr[j + 1])] = True
    counts_fit_atoms = atom_counts[fit_atom_mask, :].tocsc()
    fit_atom_rows = np.nonzero(fit_atom_mask)[0]
    # molecule id of every Fit atom
    mol_of_atom = np.zeros(atom_counts.shape[0], np.int64)
    for j, row in enumerate(rows_all.tolist()):
        mol_of_atom[int(atom_ptr[j]):int(atom_ptr[j + 1])] = j
    fit_mol_of_atom = mol_of_atom[fit_atom_rows]
    fit_local_to_global = {int(r): i for i, r in enumerate(rows_fit.tolist())}
    fit_mol_of_atom = np.asarray([fit_local_to_global[int(rows_all[j])]
                                  for j in fit_mol_of_atom], np.int64)

    coo = counts_fit_atoms.tocoo()
    presence = sparse.csr_matrix(
        (np.ones(coo.nnz, np.float32), (fit_mol_of_atom[coo.row], coo.col)),
        shape=(rows_fit.size, n_types))
    presence.data[:] = 1.0
    df = np.asarray(presence.sum(axis=0)).ravel().astype(np.int64)

    scaf_keys = sorted(set(scaffold_fit.values()))
    scaf_idx = {k: i for i, k in enumerate(scaf_keys)}
    scaf_of_fit_row = np.asarray([scaf_idx[scaffold_fit[int(r)]] for r in rows_fit.tolist()],
                                 np.int64)
    scaf_presence = sparse.csr_matrix(
        (np.ones(coo.nnz, np.float32), (scaf_of_fit_row[fit_mol_of_atom[coo.row]], coo.col)),
        shape=(len(scaf_keys), n_types))
    scaf_presence.data[:] = 1.0
    scaffold_df = np.asarray(scaf_presence.sum(axis=0)).ravel().astype(np.int64)
    occurrence = np.asarray(counts_fit_atoms.sum(axis=0)).ravel().astype(np.int64)

    # x_t(G) = sum_v log1p(c(v,t)); raw counts are needed for size 1/2 X0.
    # Every atom is a root, so the molecule-level sum over roots is a
    # molecule selector times the per-atom log-count matrix.
    log_atom_counts = atom_counts.copy()
    log_atom_counts.data = np.log1p(log_atom_counts.data.astype(np.float64))
    raw_fit = molecule_matrix(atom_counts, rows_all, atom_ptr, rows_fit).tocsc()
    raw_eval = molecule_matrix(atom_counts, rows_all, atom_ptr, rows_eval).tocsc()
    x_fit = molecule_matrix(log_atom_counts, rows_all, atom_ptr, rows_fit).tocsc()
    x_eval = molecule_matrix(log_atom_counts, rows_all, atom_ptr, rows_eval).tocsc()

    low_ids = [int(t["type_id"]) for t in types if int(t["size"]) <= 2]
    x0_names = [keys[t] for t in low_ids] + ["n_atoms_raw", "n_bonds_raw"] + \
               [f"g8_{i}" for i in range(8)]
    x0_fit = np.zeros((rows_fit.size, len(x0_names)), np.float32)
    x0_eval = np.zeros((rows_eval.size, len(x0_names)), np.float32)
    if low_ids:
        x0_fit[:, :len(low_ids)] = raw_fit[:, low_ids].toarray()
        x0_eval[:, :len(low_ids)] = raw_eval[:, low_ids].toarray()
    x0_fit[:, len(low_ids)] = [structs[int(r)].n_atoms for r in rows_fit.tolist()]
    x0_fit[:, len(low_ids) + 1] = [structs[int(r)].n_bonds for r in rows_fit.tolist()]
    x0_eval[:, len(low_ids)] = [structs[int(r)].n_atoms for r in rows_eval.tolist()]
    x0_eval[:, len(low_ids) + 1] = [structs[int(r)].n_bonds for r in rows_eval.tolist()]
    gfit = np.asarray([structs[int(r)].graph_scalars for r in rows_fit.tolist()], np.float32)
    gev = np.asarray([structs[int(r)].graph_scalars for r in rows_eval.tolist()], np.float32)
    x0_fit[:, len(low_ids) + 2:] = gfit
    x0_eval[:, len(low_ids) + 2:] = gev

    qualified = [int(t["type_id"]) for t in types
                 if int(t["size"]) in WORD_SIZES
                 and int(df[int(t["type_id"])]) >= int(cfg["candidate"]["min_molecules"])
                 and int(scaffold_df[int(t["type_id"])]) >= int(cfg["candidate"]["min_scaffolds"])]
    representatives, equiv = dedupe_columns(
        counts_fit_atoms, qualified, df, keys, sizes, grans)
    equiv_of = {}
    for rec in equiv:
        for tid in rec["members"]:
            equiv_of[int(tid)] = rec
    rep_set = set(int(x) for x in representatives)
    cand_meta: list[dict[str, Any]] = []
    for tid in representatives:
        t = types[int(tid)]
        rec = equiv_of[int(tid)]
        cand_meta.append({
            **t,
            "df": int(df[int(tid)]),
            "scaffold_df": int(scaffold_df[int(tid)]),
            "occurrence_fit": int(occurrence[int(tid)]),
            "equiv_group": str(rec["group_id"]),
            "equiv_members": [int(x) for x in rec["members"]],
            "equiv_member_keys": list(rec["member_keys"]),
            "is_representative": True,
        })
    # map old type ids -> candidate column index
    col_of_type = {int(t["type_id"]): i for i, t in enumerate(cand_meta)}
    xc_fit = x_fit[:, representatives].tocsc()
    xc_eval = x_eval[:, representatives].tocsc()

    out = Path(out_dir) if out_dir is not None else C.ANALYSIS_DIR
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": cfg["protocol_id"],
        "version": cfg["version"],
        "split": split,
        "split_meta": C.split_meta(bundle),
        "scaffold_definition": "rdkit Bemis-Murcko, acyclic keyed by own SMILES",
        "n_fit": int(rows_fit.size), "n_eval": int(rows_eval.size),
        "n_scaffolds_fit": int(len(scaf_keys)),
        "n_types_total": int(n_types),
        "n_types_by_family": dict(registry["n_types_by_family"]),
        "n_candidates_qualified": int(len(qualified)),
        "n_candidates_dedup": int(len(representatives)),
        "n_equivalence_groups": int(len(equiv)),
        "n_types_absorbed_by_dedup": int(len(qualified) - len(representatives)),
        "candidate_thresholds": {
            "min_molecules": int(cfg["candidate"]["min_molecules"]),
            "min_scaffolds": int(cfg["candidate"]["min_scaffolds"]),
        },
        "n_type_keys": int(len(keys)),
        "type_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "candidates": cand_meta,
        "equivalence_groups": equiv,
        "x0_columns": x0_names,
        "n_x0_columns": int(len(x0_names)),
        "source_pins": pins,
        "rows_fit_sha256": C.array_sha256(rows_fit),
        "rows_eval_sha256": C.array_sha256(rows_eval),
        "rows_all_sha256": C.array_sha256(rows_all),
        "atom_counts_sha256": hashlib.sha256(
            atom_counts.data.tobytes() + atom_counts.indices.tobytes()
            + atom_counts.indptr.tobytes()).hexdigest(),
        "x_fit_sha256": hashlib.sha256(
            xc_fit.data.tobytes() + xc_fit.indices.tobytes()
            + xc_fit.indptr.tobytes()).hexdigest(),
        "x_eval_sha256": hashlib.sha256(
            xc_eval.data.tobytes() + xc_eval.indices.tobytes()
            + xc_eval.indptr.tobytes()).hexdigest(),
        "x0_fit_sha256": C.array_sha256(x0_fit),
        "x0_eval_sha256": C.array_sha256(x0_eval),
        "frozen_utc": C.now_utc(),
    }
    npz_path = out / f"candidate_columns_{split}.npz"
    np.savez_compressed(
        npz_path,
        x_fit_data=xc_fit.data, x_fit_indices=xc_fit.indices, x_fit_indptr=xc_fit.indptr,
        x_fit_shape=np.asarray(xc_fit.shape, np.int64),
        x_eval_data=xc_eval.data, x_eval_indices=xc_eval.indices,
        x_eval_indptr=xc_eval.indptr, x_eval_shape=np.asarray(xc_eval.shape, np.int64),
        x0_fit=x0_fit, x0_eval=x0_eval,
        rows_fit=rows_fit, rows_eval=rows_eval,
        col_of_type=np.asarray(sorted(col_of_type.items()), np.int64),
    )
    payload["candidate_columns_sha256"] = C.sha256_file(npz_path)
    C.write_json(out / f"candidates_{split}.json", payload)
    return payload


def load_candidates(split: str) -> tuple[dict[str, Any], sparse.csc_matrix,
                                         sparse.csc_matrix, np.ndarray, np.ndarray]:
    payload = C.read_json(C.ANALYSIS_DIR / f"candidates_{split}.json")
    with np.load(C.ANALYSIS_DIR / f"candidate_columns_{split}.npz",
                allow_pickle=False) as z:
        x_fit = sparse.csc_matrix(
            (z["x_fit_data"], z["x_fit_indices"], z["x_fit_indptr"]),
            shape=tuple(int(x) for x in z["x_fit_shape"]))
        x_eval = sparse.csc_matrix(
            (z["x_eval_data"], z["x_eval_indices"], z["x_eval_indptr"]),
            shape=tuple(int(x) for x in z["x_eval_shape"]))
        x0_fit = np.asarray(z["x0_fit"], np.float32)
        x0_eval = np.asarray(z["x0_eval"], np.float32)
    return payload, x_fit, x_eval, x0_fit, x0_eval