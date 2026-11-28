"""Phase A representations: MC, MU, WL, C, F, CF-PRES.

Every block is fitted on **Train only** (vocabulary / column order / scaling) and
is then applied unchanged to Valid and Test.  All structural count blocks use the
predecessor round's frozen molecule-size scaling ``count / sqrt(max(n_atoms,1))``.

Blocks
------
``C`` / ``F``      predecessor round's frozen COARSE (754) / FULL (3652) count
                   columns, rebuilt here from the frozen enumeration cache *and*
                   checked column-by-column against the frozen design matrix.
``PRES``           the same 4406 frozen types binarised (``count > 0``) before the
                   same ``1/sqrt(n_atoms)`` scaling: repeat multiplicity removed.
``MC``             folded Morgan **count** fingerprint: identical RDKit generator
                   settings as the historical binary ECFP4 (radius 2, 2048 bits,
                   default invariants / chirality / bond types), only the output
                   changed from bit-OR to per-bucket count.
``MU``             **unfolded** Morgan count fingerprint (RDKit's own 32-bit
                   environment identifiers), Train document-frequency top-K
                   (K = 4406).  Hash collisions can still occur in the underlying
                   RDKit identifier; this is *not* claimed to be a collision-free
                   certificate.
``WL``             attributed rooted 1-WL refinement over all 9 OGB atom features
                   and all 3 OGB bond features, rounds 0/1/2, Train
                   document-frequency top-K (K = 4406) over ``(round, signature)``.
                   WL equivalence is *not* exact subgraph isomorphism.
"""

from __future__ import annotations

import struct
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix

import sga_common as K

K.add_prev_code_path()

import msds_common as MC  # noqa: E402  (predecessor round)
import msds_enum as ENUM  # noqa: E402
import msds_vocab as VOCAB  # noqa: E402

K_EXTRA = 4406  #: shared extra-feature budget for MU / WL / CF-PRES (= 754 + 3652)


# ---------------------------------------------------------------------------
# C / F / PRES from the frozen enumeration cache
# ---------------------------------------------------------------------------


def load_frozen_structure() -> dict[str, Any]:
    """Frozen COARSE/FULL vocabulary + raw integer instance counts (all molecules)."""
    discovered = VOCAB.load_discovered(str(K.MSDS_DATA / "phaseA_instances.npz"))
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        vocab = {k: np.asarray(z[k]) for k in z.files}
    coarse_ids = vocab["coarse_ids"].astype(np.int64)
    full_ids = vocab["full_ids"].astype(np.int64)
    # raw instance counts (integers) before the 1/sqrt(n_atoms) scaling
    c_raw = csr_matrix(discovered.coarse[:, coarse_ids]).astype(np.int64)
    f_raw = csr_matrix(discovered.full[:, full_ids]).astype(np.int64)
    return {
        "discovered": discovered,
        "vocab": vocab,
        "coarse_ids": coarse_ids,
        "full_ids": full_ids,
        "c_raw": c_raw,
        "f_raw": f_raw,
        "n_coarse": int(coarse_ids.size),
        "n_full": int(full_ids.size),
    }


def scale_rows(mat: csr_matrix, n_atoms: np.ndarray) -> csr_matrix:
    """``count / sqrt(max(n_atoms, 1))`` with zeros eliminated."""
    inv = 1.0 / np.sqrt(np.maximum(n_atoms.astype(np.float64), 1.0))
    out = csr_matrix(mat.multiply(inv[:, None])).astype(np.float32)
    out.eliminate_zeros()
    return out


def binarise(mat: csr_matrix) -> csr_matrix:
    out = mat.copy().astype(np.float32)
    out.data[:] = 1.0
    return out


def check_against_frozen_design(c: csr_matrix, f: csr_matrix) -> dict[str, Any]:
    """Column-by-column identity of the rebuilt C/F against the frozen matrix."""
    import scipy.sparse as sp

    path = K.MSDS_DATA / "design_CF.npz"
    with np.load(path) as z:
        n_rows, n_c, n_f = int(z["n_rows"]), int(z["n_coarse"]), int(z["n_full"])
        c0 = sp.csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]), shape=(n_rows, n_c))
        f0 = sp.csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]), shape=(n_rows, n_f))
    dc = abs(c0 - c).max() if (c0 - c).nnz else 0.0
    df = abs(f0 - f).max() if (f0 - f).nnz else 0.0
    return {
        "frozen_design_path": K.rel(path),
        "frozen_design_sha256": K.sha256_file(path),
        "shape_match": [list(c.shape) == [n_rows, n_c], list(f.shape) == [n_rows, n_f]],
        "max_abs_diff_coarse": float(dc),
        "max_abs_diff_full": float(df),
        "identical": bool(dc == 0.0 and df == 0.0),
    }


# ---------------------------------------------------------------------------
# MC / MU: Morgan counts
# ---------------------------------------------------------------------------


def _rdkit_mol(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "strict"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "failed"
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        return None, "failed"
    return mol, "unsanitized_fallback"


def build_mc(binary_bits: np.ndarray, smiles: Sequence[str]) -> tuple[csr_matrix, dict[str, Any]]:
    """Folded Morgan count fingerprint (2048 buckets), raw counts, no scaling.

    The generator call is the *count* analogue of the frozen binary call
    ``GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)``: radius 2, 2048 bits,
    default invariants, default chirality, default bond types.  The frozen
    binary block is recomputed as ``count > 0`` and compared per molecule.
    """
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    n = len(smiles)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[int] = []
    support_mismatch: list[dict[str, Any]] = []
    methods: list[str] = []
    for i, smi in enumerate(smiles):
        mol, method = _rdkit_mol(smi)
        methods.append(method)
        if mol is None:
            support_mismatch.append({"graph_index": i, "reason": "rdkit_parse_failed"})
            continue
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, 2, nBits=2048)
        items = sorted(fp.GetNonzeroElements().items())
        if not items:
            continue
        idx = np.asarray([k for k, _ in items], dtype=np.int64)
        cnt = np.asarray([v for _, v in items], dtype=np.int64)
        rows.append(np.full(idx.size, i, dtype=np.int64))
        cols.append(idx)
        vals.append(cnt)
        support_count = np.zeros(2048, dtype=bool)
        support_count[idx] = True
        if not np.array_equal(support_count, binary_bits[i].astype(bool)):
            support_mismatch.append({
                "graph_index": i,
                "n_support_diff": int((support_count ^ binary_bits[i].astype(bool)).sum()),
                "reason": "support_differs",
            })
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals).astype(np.float32)
    mat = csr_matrix((v, (r, c)), shape=(n, 2048))
    meta = {
        "impl": "rdkit.Chem.rdMolDescriptors.GetHashedMorganFingerprint(mol, 2, nBits=2048)",
        "radius": 2,
        "n_bits": 2048,
        "folded": "count per 2048-bucket (NOT bit-OR)",
        "n_molecules": n,
        "nnz": int(mat.nnz),
        "total_count": float(mat.sum()),
        "n_parse_failed": methods.count("failed"),
        "n_unsanitized_fallback": methods.count("unsanitized_fallback"),
        "support_mismatch_vs_binary_block": support_mismatch,
        "n_support_mismatch": len(support_mismatch),
        "support_identity_holds": len(support_mismatch) == 0,
    }
    return mat, meta


def build_mu(
    smiles: Sequence[str], train_rows: np.ndarray, k: int = K_EXTRA
) -> tuple[csr_matrix, np.ndarray, dict[str, Any], csr_matrix]:
    """Unfolded Morgan counts, Train document-frequency top-K environment ids.

    Two passes: the first assigns global identifiers and accumulates the Train
    document frequency per identifier (a molecule counts once, no matter how
    often the environment repeats inside it); the second fills a CSR matrix.
    """
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    n = len(smiles)
    train_set = set(int(r) for r in train_rows.tolist())
    id_of: dict[int, int] = {}
    train_df: list[int] = []
    per_mol_nnz = np.zeros(n, dtype=np.int64)
    methods: list[str] = []
    f_rows: list[np.ndarray] = []
    f_cols: list[np.ndarray] = []
    f_vals: list[np.ndarray] = []
    for i, smi in enumerate(smiles):
        mol, method = _rdkit_mol(smi)
        methods.append(method)
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        env = fp.GetNonzeroElements()
        if env:
            e_all = np.fromiter(env.keys(), dtype=np.int64, count=len(env))
            v_all = np.fromiter(env.values(), dtype=np.float32, count=len(env))
            f_rows.append(np.full(e_all.size, i, dtype=np.int64))
            f_cols.append(e_all % 2048)
            f_vals.append(v_all)
        per_mol_nnz[i] = len(env)
        is_train = i in train_set
        for eid in env:
            gid = id_of.get(eid)
            if gid is None:
                gid = len(id_of)
                id_of[eid] = gid
                train_df.append(0)
            if is_train:
                train_df[gid] += 1
    df = np.asarray(train_df, dtype=np.int64)
    env_ids = np.fromiter(id_of.keys(), dtype=np.int64, count=len(id_of))
    order = np.lexsort((env_ids, -df))  # document frequency desc, then id asc
    selected = order[:k]
    keep_env = env_ids[selected]
    keep_df = df[selected]
    col_of_env = {int(e): j for j, e in enumerate(keep_env.tolist())}
    rows = np.zeros(int(per_mol_nnz.sum()), dtype=np.int64)
    cols = np.zeros(int(per_mol_nnz.sum()), dtype=np.int64)
    vals = np.zeros(int(per_mol_nnz.sum()), dtype=np.float32)
    pos = 0
    for i, smi in enumerate(smiles):
        if per_mol_nnz[i] == 0:
            continue
        mol, _ = _rdkit_mol(smi)
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        env = fp.GetNonzeroElements()
        e = np.fromiter(env.keys(), dtype=np.int64, count=len(env))
        v = np.fromiter(env.values(), dtype=np.float32, count=len(env))
        j = np.fromiter((col_of_env.get(int(x), -1) for x in e.tolist()), dtype=np.int64, count=e.size)
        keep = j >= 0
        m = int(keep.sum())
        rows[pos: pos + m] = i
        cols[pos: pos + m] = j[keep]
        vals[pos: pos + m] = v[keep]
        pos += m
    rows, cols, vals = rows[:pos], cols[:pos], vals[:pos]
    mat = csr_matrix((vals, (rows, cols)), shape=(n, int(selected.size)))
    fold_all = csr_matrix(
        (np.concatenate(f_vals) if f_vals else np.zeros(0, np.float32),
         (np.concatenate(f_rows) if f_rows else np.zeros(0, np.int64),
          np.concatenate(f_cols) if f_cols else np.zeros(0, np.int64))),
        shape=(n, 2048),
    )
    fold_all.sum_duplicates()
    meta = {
        "impl": "rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)",
        "radius": 2,
        "n_distinct_environments_all": int(df.size),
        "k_requested": int(k),
        "k_selected": int(selected.size),
        "selection": "Train document frequency desc, RDKit environment id asc on ties",
        "train_df_min_selected": int(keep_df.min()) if keep_df.size else None,
        "train_df_max_selected": int(keep_df.max()) if keep_df.size else None,
        "nnz": int(mat.nnz),
        "total_count": float(mat.sum()),
        "n_parse_failed": methods.count("failed"),
        "caveat": "RDKit unfolded Morgan ids are 32-bit hashes; residual collisions are possible",
    }
    return mat, keep_env, meta, fold_all


# ---------------------------------------------------------------------------
# WL: attributed rooted refinement over the OGB attributes
# ---------------------------------------------------------------------------


def _digest(payload: bytes) -> bytes:
    import hashlib

    return hashlib.blake2b(payload, digest_size=16, person=b"molhiv-motif-v1").digest()


def _feat_bytes(values: np.ndarray) -> bytes:
    a = np.asarray(values, dtype=np.int16).reshape(-1)
    return struct.pack("<H", len(a)) + a.tobytes()


def rooted_signatures(
    atom_feat: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_feat: np.ndarray,
    max_round: int,
) -> list[list[bytes]]:
    """Deterministic attributed rooted refinement (the KSVD probe's recursion).

    Round 0 is a digest of the atom's own 9 attributes; round ``r`` digests the
    node's round ``r-1`` label together with the sorted multiset of
    ``(bond attributes, neighbour round r-1 label)``.  Two atoms share a
    signature at round ``r`` iff they are indistinguishable by ``r`` rounds of
    attributed 1-WL refinement -- not iff their rooted subgraphs are isomorphic.
    """
    n = int(atom_feat.shape[0])
    nbrs: list[list[tuple[int, bytes]]] = [[] for _ in range(n)]
    for u, v, bf in zip(bond_u.tolist(), bond_v.tolist(), bond_feat):
        eb = _feat_bytes(bf)
        nbrs[u].append((v, eb))
        nbrs[v].append((u, eb))
    labels = [_digest(b"A" + _feat_bytes(atom_feat[u])) for u in range(n)]
    out = [list(labels)]
    for r in range(1, max_round + 1):
        nxt: list[bytes] = []
        for u in range(n):
            messages = sorted(eb + labels[v] for v, eb in nbrs[u])
            payload = b"R" + bytes([r]) + labels[u] + struct.pack("<H", len(messages)) + b"".join(messages)
            nxt.append(_digest(payload))
        labels = nxt
        out.append(list(labels))
    return out


def build_wl(
    molecules: Any, train_rows: np.ndarray, k: int = K_EXTRA, max_round: int = 2
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    """Attributed rooted-environment counts, Train document-frequency top-K."""
    n = molecules.n_molecules
    train_set = set(int(r) for r in train_rows.tolist())
    id_of: dict[bytes, int] = {}
    train_df: list[int] = []
    per_mol_nnz = np.zeros(n, dtype=np.int64)
    round_of: list[int] = []
    for i in range(n):
        lo, hi = int(molecules.atom_offsets[i]), int(molecules.atom_offsets[i + 1])
        blo, bhi = int(molecules.bond_offsets[i]), int(molecules.bond_offsets[i + 1])
        sigs = rooted_signatures(
            molecules.atom_feat[lo:hi],
            molecules.bond_u[blo:bhi],
            molecules.bond_v[blo:bhi],
            molecules.bond_feat[blo:bhi],
            max_round,
        )
        seen: dict[bytes, int] = {}
        for r, labels in enumerate(sigs):
            prefix = bytes([r])
            for lab in labels:
                key = prefix + lab
                gid = id_of.get(key)
                if gid is None:
                    gid = len(id_of)
                    id_of[key] = gid
                    train_df.append(0)
                    round_of.append(r)
                seen[key] = seen.get(key, 0) + 1
        per_mol_nnz[i] = len(seen)
        if i in train_set:
            for key in seen:
                train_df[id_of[key]] += 1
    df = np.asarray(train_df, dtype=np.int64)
    keys = list(id_of.keys())
    # deterministic tie-break on the raw signature bytes
    order = sorted(range(df.size), key=lambda j: (-int(df[j]), keys[j]))
    selected = np.asarray(order[:k], dtype=np.int64)
    keep_df = df[selected]
    remap = np.full(df.size, -1, dtype=np.int64)
    remap[selected] = np.arange(selected.size, dtype=np.int64)
    rows = np.zeros(int(per_mol_nnz.sum()), dtype=np.int64)
    cols = np.zeros(int(per_mol_nnz.sum()), dtype=np.int64)
    vals = np.zeros(int(per_mol_nnz.sum()), dtype=np.float32)
    pos = 0
    for i in range(n):
        lo, hi = int(molecules.atom_offsets[i]), int(molecules.atom_offsets[i + 1])
        blo, bhi = int(molecules.bond_offsets[i]), int(molecules.bond_offsets[i + 1])
        sigs = rooted_signatures(
            molecules.atom_feat[lo:hi],
            molecules.bond_u[blo:bhi],
            molecules.bond_v[blo:bhi],
            molecules.bond_feat[blo:bhi],
            max_round,
        )
        counts: dict[bytes, int] = {}
        for r, labels in enumerate(sigs):
            prefix = bytes([r])
            for lab in labels:
                key = prefix + lab
                counts[key] = counts.get(key, 0) + 1
        for key, cnt in counts.items():
            j = int(remap[id_of[key]])
            if j >= 0:
                rows[pos] = i
                cols[pos] = j
                vals[pos] = float(cnt)
                pos += 1
    mat = csr_matrix((vals[:pos], (rows[:pos], cols[:pos])), shape=(n, int(selected.size)))
    selected_round = np.asarray([round_of[int(j)] for j in selected], dtype=np.int64)
    meta = {
        "impl": "blake2b-16 attributed rooted 1-WL refinement over all 9 OGB atom + 3 OGB bond features",
        "rounds": [0, max_round],
        "n_distinct_signatures_all": int(df.size),
        "n_signatures_by_round_all": {
            str(r): int(sum(1 for x in round_of if x == r)) for r in range(max_round + 1)
        },
        "k_requested": int(k),
        "k_selected": int(selected.size),
        "selection": "Train document frequency desc, signature bytes asc on ties",
        "n_selected_by_round": {str(r): int((selected_round == r).sum()) for r in range(max_round + 1)},
        "train_df_min_selected": int(keep_df.min()) if keep_df.size else None,
        "train_df_max_selected": int(keep_df.max()) if keep_df.size else None,
        "nnz": int(mat.nnz),
        "total_count": float(mat.sum()),
        "caveat": "WL colour equality is an approximation of rooted-subgraph identity, not isomorphism",
    }
    return mat, selected, meta
