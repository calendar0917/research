"""Frozen sparse S2-pattern identity: keys, vocabulary, vectors, null, fold.

Nothing here is trained and nothing here reads a label.  All integer codes come
verbatim from the production ``lcrc_data`` / ``mb2_data`` caches; the only
composition is

    key = (atom_type[root atom of the S2 record],
           path_cat, path_end63, path_closure)

The binary presence vector over the 512 frozen keys, its frozen normalisation
``x = c / sqrt(max(1, |c|_0))``, the label-blind bipartite double-edge swap
(NULL) and the salted-hash 512 -> 64 fold (FOLD64) live here.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import si_common as C

import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec

KEY_FIELDS = ("root_atom_type", "path_cat", "path_end63", "path_closure")


# ---------------------------------------------------------------------------
# integer key codec
# ---------------------------------------------------------------------------


def encode_key(
    root: np.ndarray, path_cat: np.ndarray, path_end63: np.ndarray, closure: np.ndarray, spec: FeatureSpec
) -> np.ndarray:
    """Single int64 code for a 4-tuple; invertible inside the code ranges."""
    root = np.asarray(root, dtype=np.int64)
    path_cat = np.asarray(path_cat, dtype=np.int64)
    path_end63 = np.asarray(path_end63, dtype=np.int64)
    closure = np.asarray(closure, dtype=np.int64)
    out = root * int(spec.n_path_codes) + path_cat
    out = out * int(spec.n_nbr_codes) + path_end63
    out = out * 2 + closure
    return out


def decode_key(code: int, spec: FeatureSpec) -> tuple[int, int, int, int]:
    code = int(code)
    closure = code % 2
    code //= 2
    path_end63 = code % int(spec.n_nbr_codes)
    code //= int(spec.n_nbr_codes)
    path_cat = code % int(spec.n_path_codes)
    root = code // int(spec.n_path_codes)
    return (root, path_cat, path_end63, closure)


def key_tuple_message(key: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in key)


# ---------------------------------------------------------------------------
# per-path records of one split cache
# ---------------------------------------------------------------------------


def path_keys(cache: Mapping[str, Any], spec: FeatureSpec) -> dict[str, np.ndarray]:
    """Flatten every S2 record of a split cache to ``(molecule, root, key)``.

    ``root`` is the global atom index whose ``path_ptr`` range contains the
    record; ``root_atom_type`` is ``atom_type[root]``.  Record order is the
    production cache order and is irrelevant to every downstream use.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    n_atoms = int(atom_ptr[-1])
    if path_ptr.shape != (n_atoms + 1,):
        raise ValueError(f"path_ptr shape {path_ptr.shape} != {(n_atoms + 1,)}")
    path_counts = np.diff(path_ptr)
    if int(path_counts.sum()) != int(np.asarray(cache["path_cat"]).size):
        raise ValueError("path record count does not match path_ptr")
    root_atom = np.repeat(np.arange(n_atoms, dtype=np.int64), path_counts)
    mol_of_atom = np.repeat(np.arange(atom_ptr.size - 1, dtype=np.int64), np.diff(atom_ptr))
    mol_of_path = mol_of_atom[root_atom] if root_atom.size else np.zeros(0, dtype=np.int64)
    root_type = atom_type[root_atom] if root_atom.size else np.zeros(0, dtype=np.int64)
    key = encode_key(
        root_type,
        np.asarray(cache["path_cat"], dtype=np.int64),
        np.asarray(cache["path_end63"], dtype=np.int64),
        np.asarray(cache["path_closure"], dtype=np.int64),
        spec,
    )
    return {
        "mol_of_path": mol_of_path,
        "root_atom": root_atom,
        "root_atom_type": root_type,
        "path_cat": np.asarray(cache["path_cat"], dtype=np.int64),
        "path_end63": np.asarray(cache["path_end63"], dtype=np.int64),
        "path_closure": np.asarray(cache["path_closure"], dtype=np.int64),
        "key": key,
    }


def molecule_keys_loop(cache: Mapping[str, Any], spec: FeatureSpec, mol: int) -> list[tuple[int, int, int, int]]:
    """Independent per-molecule reference implementation (gate 3 cross-check)."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    path_cat = np.asarray(cache["path_cat"], dtype=np.int64)
    path_end63 = np.asarray(cache["path_end63"], dtype=np.int64)
    path_closure = np.asarray(cache["path_closure"], dtype=np.int64)
    out: list[tuple[int, int, int, int]] = []
    a0, a1 = int(atom_ptr[int(mol)]), int(atom_ptr[int(mol) + 1])
    for atom in range(a0, a1):
        p0, p1 = int(path_ptr[atom]), int(path_ptr[atom + 1])
        for p in range(p0, p1):
            out.append(
                (
                    int(atom_type[atom]),
                    int(path_cat[p]),
                    int(path_end63[p]),
                    int(path_closure[p]),
                )
            )
    return out


# ---------------------------------------------------------------------------
# document frequency and the frozen vocabulary
# ---------------------------------------------------------------------------


def document_frequency(records: Mapping[str, np.ndarray]) -> dict[int, int]:
    """Distinct-molecule counts per key (a molecule counts once per key)."""
    mol = np.asarray(records["mol_of_path"], dtype=np.int64)
    key = np.asarray(records["key"], dtype=np.int64)
    if mol.size == 0:
        return {}
    pairs = np.unique(np.stack([mol, key], axis=1), axis=0)
    keys, counts = np.unique(pairs[:, 1], return_counts=True)
    return {int(k): int(c) for k, c in zip(keys.tolist(), counts.tolist())}


def build_vocabulary(train_cache: Mapping[str, Any], spec: FeatureSpec, k: int = C.K) -> dict[str, Any]:
    """Top-``k`` keys by Train document frequency (label-free, deterministic).

    Ordering: ``(-document_frequency, key tuple ascending)``.  The function only
    receives Train structures, so Validation/Test cannot leak into the choice.
    """
    records = path_keys(train_cache, spec)
    freq = document_frequency(records)
    decoded = {code: decode_key(code, spec) for code in freq}
    ordered = sorted(freq.items(), key=lambda kv: (-int(kv[1]), decoded[kv[0]]))
    if len(ordered) < int(k):
        payload = {
            "schema": "lcrc-sparse-identity-vocabulary-v1",
            "status": "FEATURE_INFEASIBLE",
            "k_requested": int(k),
            "n_unique_keys": int(len(ordered)),
            "train_n_molecules": int(np.asarray(train_cache["n_molecules"])),
            "reason": f"only {len(ordered)} distinct legal S2 keys in Train, fewer than K={k}; K is not adjusted",
        }
        return payload
    selected = ordered[: int(k)]
    keys = [decoded[code] for code, _ in selected]
    freq_values = [int(freq) for _, freq in selected]
    payload = {
        "schema": "lcrc-sparse-identity-vocabulary-v1",
        "status": "OK",
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "k": int(k),
        "n_unique_keys": int(len(ordered)),
        "train_n_molecules": int(np.asarray(train_cache["n_molecules"])),
        "train_n_s2_paths": int(records["key"].size),
        "ordering_rule": "document_frequency_desc_then_key_tuple_ascending",
        "document_frequency_definition": "number of distinct Train molecules containing the key",
        "labels_used": False,
        "valid_or_test_used": False,
        "feature_spec_hash": spec.hash(),
        "key_fields": list(KEY_FIELDS),
        "keys": [[int(x) for x in key] for key in keys],
        "document_frequency": freq_values,
        "n_keys_with_df_1": int(sum(1 for f in freq.values() if int(f) == 1)),
        "vocabulary_sha256": vocabulary_hash(keys, freq_values),
    }
    return payload


def vocabulary_hash(keys: Sequence[Sequence[int]], freq: Sequence[int]) -> str:
    payload = {
        "keys": [[int(x) for x in k] for k in keys],
        "document_frequency": [int(x) for x in freq],
        "ordering_rule": "document_frequency_desc_then_key_tuple_ascending",
    }
    return C.sha256_text(C.stable_json(payload))


def vocab_key_array(vocab: Mapping[str, Any], spec: FeatureSpec) -> np.ndarray:
    keys = np.asarray(vocab["keys"], dtype=np.int64)
    if keys.ndim != 2 or keys.shape[1] != 4:
        raise ValueError(f"vocabulary keys shape {keys.shape}")
    return encode_key(keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], spec)


def load_vocab_records(cache: Mapping[str, Any], spec: FeatureSpec, vocab: Mapping[str, Any]) -> dict[str, Any]:
    """Vectorised lookup of the frozen keys in one split cache."""
    records = path_keys(cache, spec)
    vocab_int = vocab_key_array(vocab, spec)
    order = np.argsort(vocab_int, kind="stable")
    sorted_keys = vocab_int[order]
    pos = np.searchsorted(sorted_keys, records["key"])
    pos_clipped = np.clip(pos, 0, sorted_keys.size - 1)
    hit = sorted_keys[pos_clipped] == records["key"]
    col = np.where(hit, order[pos_clipped], -1)
    return {"records": records, "hit": hit, "col": col, "vocab_int": vocab_int}


# ---------------------------------------------------------------------------
# molecule binary / normalised vectors
# ---------------------------------------------------------------------------


def encode_split(
    cache: Mapping[str, Any], spec: FeatureSpec, vocab: Mapping[str, Any]
) -> dict[str, Any]:
    """Binary presence matrix and the frozen normalised vector for one split."""
    n_mol = int(np.asarray(cache["n_molecules"]))
    k = int(vocab["k"])
    lookup = load_vocab_records(cache, spec, vocab)
    records = lookup["records"]
    hit = lookup["hit"]
    binary = np.zeros((n_mol, k), dtype=np.uint8)
    if hit.any():
        binary[records["mol_of_path"][hit], lookup["col"][hit]] = 1
    active = binary.sum(axis=1).astype(np.int64)
    scale = (1.0 / np.sqrt(np.maximum(1, active).astype(np.float64))).astype(np.float32)
    x = (binary.astype(np.float32) * scale[:, None]).astype(np.float32)
    n_paths = int(records["key"].size)
    stats = {
        "n_molecules": n_mol,
        "n_s2_paths": n_paths,
        "n_paths_in_vocabulary": int(hit.sum()),
        "oov_path_ratio": float(1.0 - (int(hit.sum()) / max(n_paths, 1))),
        "n_distinct_keys_observed": int(np.unique(records["key"]).size),
        "molecule_coverage": float((active > 0).mean()),
        "n_molecules_with_no_pattern": int((active == 0).sum()),
        "active_count": {
            "min": int(active.min()) if active.size else 0,
            "max": int(active.max()) if active.size else 0,
            "mean": float(active.mean()) if active.size else 0.0,
            "sd": float(active.std(ddof=0)) if active.size else 0.0,
            "median": float(np.median(active)) if active.size else 0.0,
            "percentiles": {
                str(p): float(np.percentile(active, p)) for p in (5, 10, 25, 50, 75, 90, 95, 99)
            },
        },
        "active_count_histogram": {
            str(int(k_)): int(v) for k_, v in zip(*np.unique(active, return_counts=True))
        },
        "column_document_frequency": {
            "min": int(binary.sum(axis=0).min()) if k else 0,
            "max": int(binary.sum(axis=0).max()) if k else 0,
            "mean": float(binary.sum(axis=0).mean()) if k else 0.0,
        },
        "n_nonzero": int(binary.sum()),
        "row_sum_sha256": C.sha256_bytes(active.astype(np.int64).tobytes()),
        "row_ids_sha256": C.index_sha256(np.asarray(cache["rows"], dtype=np.int64).tolist())
        if "rows" in cache
        else None,
        "binary_sha256": C.array_hash_u8(binary),
        "vector_sha256": C.array_hash_f32(x),
        "vector_dtype": "float32",
        "normalisation": "c / sqrt(max(1, ||c||_0))",
        "vocabulary_sha256": vocab["vocabulary_sha256"],
        "feature_spec_hash": spec.hash(),
    }
    return {"binary": binary, "x": x, "stats": stats, "hit": hit, "col": lookup["col"], "records": records}


# ---------------------------------------------------------------------------
# NULL: label-blind bipartite double-edge swap
# ---------------------------------------------------------------------------


def double_edge_swap(
    binary: np.ndarray, seed: int, *, multiplier: int = C.NULL_SWAP_MULTIPLIER
) -> tuple[np.ndarray, dict[str, Any]]:
    """Randomise a binary matrix while conserving both marginals exactly.

    Each attempt samples two existing non-zero positions uniformly; the swap is
    accepted only when the two rows and the two columns differ and both cross
    cells are zero.  Exactly ``multiplier * nnz`` attempts are made (no early
    stop, no adaptation).  Labels are never involved: the routine only sees a
    binary matrix.
    """
    mat = np.asarray(binary, dtype=np.uint8)
    n_rows, n_cols = int(mat.shape[0]), int(mat.shape[1])
    rows: list[set[int]] = [set() for _ in range(n_rows)]
    cols: list[set[int]] = [set() for _ in range(n_cols)]
    edges: list[list[int]] = []
    for i, a in zip(*np.nonzero(mat)):
        i = int(i)
        a = int(a)
        rows[i].add(a)
        cols[a].add(i)
        edges.append([i, a])
    nnz = len(edges)
    if nnz == 0:
        out = {"n_rows": n_rows, "n_cols": n_cols, "n_nonzero": 0, "n_attempts": 0, "n_accepted": 0}
        return mat.copy(), out
    rng = np.random.default_rng(int(seed))
    attempts = int(multiplier) * int(nnz)
    accepted = 0
    for _ in range(attempts):
        p1, p2 = rng.integers(0, nnz, size=2).tolist()
        i, a = edges[p1]
        j, b = edges[p2]
        if i == j or a == b:
            continue
        if b in rows[i] or a in rows[j]:
            continue
        rows[i].discard(a)
        rows[i].add(b)
        rows[j].discard(b)
        rows[j].add(a)
        cols[a].discard(i)
        cols[a].add(j)
        cols[b].discard(j)
        cols[b].add(i)
        edges[p1] = [i, b]
        edges[p2] = [j, a]
        accepted += 1
    out_mat = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for i, a in edges:
        out_mat[i, a] = 1
    stats = compare_matrices(mat, out_mat)
    stats.update(
        {
            "seed": int(seed),
            "multiplier": int(multiplier),
            "n_molecules": n_rows,
            "n_patterns": n_cols,
            "n_nonzero": nnz,
            "n_attempts": attempts,
            "n_accepted": int(accepted),
            "acceptance_rate": float(accepted / max(attempts, 1)),
            "null_matrix_sha256": C.array_hash_u8(out_mat),
            "source_matrix_sha256": C.array_hash_u8(mat),
            "labels_used": False,
        }
    )
    return out_mat, stats


def compare_matrices(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    rs_a, rs_b = a.sum(axis=1), b.sum(axis=1)
    cs_a, cs_b = a.sum(axis=0), b.sum(axis=0)
    inter = int(np.minimum(a, b).sum())
    nnz_a = int(a.sum())
    nnz_b = int(b.sum())
    union = nnz_a + nnz_b - inter
    hamming = int((a != b).sum())
    return {
        "row_sums_equal": bool(np.array_equal(rs_a, rs_b)),
        "col_sums_equal": bool(np.array_equal(cs_a, cs_b)),
        "n_nonzero_equal": bool(nnz_a == nnz_b),
        "identical": bool(np.array_equal(a, b)),
        "n_nonzero_true": nnz_a,
        "n_nonzero_other": nnz_b,
        "n_shared_positions": inter,
        "jaccard_nonzero": float(inter / union) if union else 1.0,
        "changed_fraction_of_nonzero": float(1.0 - inter / max(nnz_a, 1)),
        "hamming_positions": hamming,
        "hamming_fraction_of_matrix": float(hamming / max(a.size, 1)),
    }


# ---------------------------------------------------------------------------
# FOLD64: salted-hash deterministic bucket assignment
# ---------------------------------------------------------------------------


def fold_buckets(vocab: Mapping[str, Any], *, n_buckets: int = C.N_BUCKETS, salt: str = C.FOLD_SALT) -> dict[str, Any]:
    keys = [[int(x) for x in k] for k in vocab["keys"]]
    k = len(keys)
    if k % int(n_buckets) != 0:
        raise ValueError(f"{k} keys do not divide into {n_buckets} equal buckets")
    digests = [
        hashlib.sha256(f"{salt}|{key_tuple_message(key)}".encode("utf-8")).hexdigest() for key in keys
    ]
    order = sorted(range(k), key=lambda i: (digests[i], tuple(keys[i])))
    bucket_of_col = np.zeros(k, dtype=np.int64)
    for rank, col in enumerate(order):
        bucket_of_col[int(col)] = rank % int(n_buckets)
    counts = np.bincount(bucket_of_col, minlength=int(n_buckets))
    if not np.all(counts == k // int(n_buckets)):
        raise RuntimeError(f"bucket sizes are not uniform: {counts.tolist()}")
    return {
        "salt": salt,
        "n_buckets": int(n_buckets),
        "per_bucket": int(k // int(n_buckets)),
        "bucket_of_col": bucket_of_col,
        "bucket_sizes": counts.tolist(),
        "hash_order": order,
        "digest_prefix": [digests[i][:16] for i in order],
        "salt_rule": "sha256(salt + '|' + ','.join(str(int(x)) for x in key)), sort ascending by digest, bucket = rank % 64",
        "assignment_sha256": C.sha256_bytes(
            np.asarray(bucket_of_col, dtype=np.int64).tobytes() + C.sha256_text(salt).encode("utf-8")
        ),
    }


def project_fold64(x: np.ndarray, bucket_of_col: np.ndarray, *, n_buckets: int = C.N_BUCKETS) -> np.ndarray:
    """``z_j = (1/sqrt 8) * sum_{k in bucket j} x_k`` (float64 accumulation)."""
    x = np.asarray(x, dtype=np.float32)
    bucket_of_col = np.asarray(bucket_of_col, dtype=np.int64)
    per_bucket = int(x.shape[1] // int(n_buckets))
    scale = 1.0 / np.sqrt(float(per_bucket))
    out = np.zeros((x.shape[0], int(n_buckets)), dtype=np.float32)
    for j in range(int(n_buckets)):
        cols = np.flatnonzero(bucket_of_col == j)
        if cols.size != per_bucket:
            raise RuntimeError(f"bucket {j} has {cols.size} columns")
        out[:, j] = (np.asarray(x[:, cols], dtype=np.float64).sum(axis=1) * scale).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# branch input matrices per arm
# ---------------------------------------------------------------------------


def branch_input(model: str, split: str, features: Mapping[str, Any], fold: Mapping[str, Any]) -> np.ndarray:
    """The exact matrix a wide arm's residual branch receives."""
    if model == C.MODEL_TRUE:
        return np.asarray(features[split]["x"], dtype=np.float32)
    if model == C.MODEL_FOLD:
        return project_fold64(features[split]["x"], fold["bucket_of_col"])
    raise KeyError(f"{model} has no branch input beyond the NULL randomisation")


# ---------------------------------------------------------------------------
# 18-ring witness pair in the 512-dim representation
# ---------------------------------------------------------------------------


def ring18_collision(vocab: Mapping[str, Any]) -> dict[str, Any]:
    """Build the historical 18-ring pair through the production pipeline."""
    import mb2_rdkit as RDK

    spec = frozen_spec()
    out: dict[str, Any] = {"pair": C.RING18_PAIR, "representation": "512-dim frozen S2-pattern presence"}
    records: dict[str, Any] = {}
    for name, info in C.RING18_PAIR.items():
        raw = RDK.smiles_to_raw_row(info["smiles"], 0)
        graph = RDK.raw_row_to_graph(raw, 0, spec)
        mol = PREP_DATA.build_molecule(graph, spec)
        cache = PREP_DATA.concat_molecules([mol], spec, split=f"ring18_{name}")
        enc = encode_split(cache, spec, vocab)
        records[name] = {
            "n_atoms": int(cache["atom_ptr"][-1]),
            "n_s2_paths": int(enc["stats"]["n_s2_paths"]),
            "n_active_patterns": int(enc["stats"]["n_nonzero"]),
            "active_count": int(enc["binary"].sum()),
            "binary": enc["binary"][0].tolist(),
            "vector_sha256": enc["stats"]["vector_sha256"],
            "patterns_in_vocab": int(np.unique(enc["records"]["key"][enc["hit"]]).size),
        }
    same_vector = records["A"]["vector_sha256"] == records["B"]["vector_sha256"]
    set_a = {int(x) for x in np.flatnonzero(np.asarray(records["A"]["binary"]))}
    set_b = {int(x) for x in np.flatnonzero(np.asarray(records["B"]["binary"]))}
    out.update(
        {
            "A": records["A"],
            "B": records["B"],
            "collision": bool(same_vector or set_a == set_b),
            "jaccard": float(len(set_a & set_b) / len(set_a | set_b)) if (set_a | set_b) else 1.0,
            "expected": "collision expected: the pair has an identical radius-2 labelled environment multiset",
            "is_gate": False,
            "note": "a collision is not a failure of this round; this round is not about long-range expressivity",
        }
    )
    return out


def frozen_spec() -> FeatureSpec:
    """The frozen ``lipo_v1`` FeatureSpec (hash re-verified by the caller)."""
    import mb2_data as MD

    return MD.frozen_spec()


# ---------------------------------------------------------------------------
# split-level feature bundle
# ---------------------------------------------------------------------------


def build_feature_bundle(
    caches: Mapping[str, Mapping[str, Any]], spec: FeatureSpec, vocab: Mapping[str, Any], splits: Iterable[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in splits:
        enc = encode_split(caches[split], spec, vocab)
        out[split] = {"binary": enc["binary"], "x": enc["x"], "stats": enc["stats"]}
    return out


__all__ = [
    "KEY_FIELDS",
    "encode_key",
    "decode_key",
    "key_tuple_message",
    "path_keys",
    "molecule_keys_loop",
    "document_frequency",
    "build_vocabulary",
    "vocabulary_hash",
    "vocab_key_array",
    "load_vocab_records",
    "encode_split",
    "double_edge_swap",
    "compare_matrices",
    "fold_buckets",
    "project_fold64",
    "branch_input",
    "ring18_collision",
    "build_feature_bundle",
]
