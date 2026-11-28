"""Phase A statistics + frozen vocabulary (Train-only) + Phase B design matrices.

The vocabulary is fitted on **Train only**.  Valid and Test are encoded after the
vocabulary is frozen (``analysis/vocabulary.json`` + its sha256) and can never add
a type or change a column order.

Design-matrix contract (PROTOCOL.md §5.1)
----------------------------------------
``C`` and ``F`` hold *instance counts per molecule*, scaled once and for all by
``1 / sqrt(max(n_atoms, 1))``.  For a COARSE parent ``p``, the retained FULL
children satisfy ``sum_f F_f <= C_p`` **per molecule**, which is verified here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

import msds_common as C
import msds_enum as E


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


@dataclass
class Discovered:
    n_molecules: int
    full: csr_matrix
    coarse: csr_matrix
    full_keys: list[bytes]
    coarse_keys: list[bytes]
    full_family: np.ndarray
    coarse_family: np.ndarray
    full_n_edges: np.ndarray
    coarse_n_edges: np.ndarray
    full_parent: np.ndarray
    n_instances_family: np.ndarray
    examples: list[list[tuple[int, tuple[int, ...]]]]

    def key_string(self, gid: int, granularity: str) -> str:
        raw = self.full_keys[gid] if granularity == "FULL" else self.coarse_keys[gid]
        return E.key_to_string(eval(raw.decode()))  # noqa: S307 - our own repr tuple

    def key_tuple(self, gid: int, granularity: str) -> tuple:
        raw = self.full_keys[gid] if granularity == "FULL" else self.coarse_keys[gid]
        return eval(raw.decode())  # noqa: S307


def load_discovered(path: str | None = None) -> Discovered:
    path = path or str(C.CACHE / "phaseA_instances.npz")
    with np.load(path, allow_pickle=True) as z:
        n_mol = int(z["n_molecules"])
        full = csr_matrix(
            (z["full_data"], z["full_indices"], z["full_indptr"]),
            shape=(n_mol, int(z["full_indices"].max()) + 1),
        )
        coarse = csr_matrix(
            (z["coarse_data"], z["coarse_indices"], z["coarse_indptr"]),
            shape=(n_mol, int(z["coarse_indices"].max()) + 1),
        )
        return Discovered(
            n_molecules=int(z["n_molecules"]),
            full=full,
            coarse=coarse,
            full_keys=[bytes(k) for k in z["full_keys"]],
            coarse_keys=[bytes(k) for k in z["coarse_keys"]],
            full_family=np.asarray(z["full_family"], dtype=np.int64),
            coarse_family=np.asarray(z["coarse_family"], dtype=np.int64),
            full_n_edges=np.asarray(z["full_n_edges"], dtype=np.int64),
            coarse_n_edges=np.asarray(z["coarse_n_edges"], dtype=np.int64),
            full_parent=np.asarray(z["full_parent"], dtype=np.int64),
            n_instances_family=np.asarray(z["n_instances_family"], dtype=np.int64),
            examples=[list(x) for x in z["examples"]],
        )


# ---------------------------------------------------------------------------
# train-only type statistics
# ---------------------------------------------------------------------------


def _scaffold_pairs(sub: csr_matrix, row_ids: np.ndarray, scaffold_ids: np.ndarray,
                    n_types: int, n_scaffolds: int) -> np.ndarray:
    """Distinct (type, scaffold) pairs over the given rows -> scaffold frequency per type.

    ``sub`` is ``mat[row_ids]``.  Inside one row a type appears at most once
    (CSR stores one entry per column), so the unique-pair count below is exactly
    the number of distinct Train scaffolds in which that type occurs.
    """
    if sub.nnz == 0:
        return np.zeros(n_types, dtype=np.int64)
    rows = np.repeat(np.arange(sub.shape[0], dtype=np.int64), np.diff(sub.indptr).astype(np.int64))
    cols = sub.indices.astype(np.int64)
    scaf = scaffold_ids[row_ids][rows]
    keys = np.unique(cols * n_scaffolds + scaf)
    return np.bincount(keys // n_scaffolds, minlength=n_types).astype(np.int64)


def type_stats(
    mat: csr_matrix,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    test_rows: np.ndarray,
    label: np.ndarray,
    scaffold_ids: np.ndarray,
    n_scaffolds: int,
) -> dict[str, np.ndarray]:
    n_types = mat.shape[1]
    sub = mat[train_rows]
    df = np.asarray(sub.getnnz(axis=0), dtype=np.int64)
    occurrence = np.asarray(sub.sum(axis=0)).reshape(-1).astype(np.int64)
    scaf = _scaffold_pairs(sub, train_rows, scaffold_ids, n_types, n_scaffolds)
    pos_rows = train_rows[label[train_rows] == 1]
    neg_rows = train_rows[label[train_rows] == 0]
    df_pos = np.asarray(mat[pos_rows].getnnz(axis=0), dtype=np.int64)
    df_neg = np.asarray(mat[neg_rows].getnnz(axis=0), dtype=np.int64)
    return {
        "occurrence_train": occurrence,
        "document_frequency_train": df,
        "scaffold_frequency_train": scaf,
        "positive_molecules_train": df_pos,
        "negative_molecules_train": df_neg,
    }


@dataclass
class FrozenVocab:
    coarse_ids: np.ndarray  # column order (global type ids)
    full_ids: np.ndarray
    coarse_family: np.ndarray
    full_family: np.ndarray
    full_stats: dict[str, np.ndarray]  # aligned with full_ids
    coarse_stats: dict[str, np.ndarray]
    support_weight: np.ndarray  # a_f aligned with full_ids
    support_weight_null: np.ndarray
    full_parent_in_vocab: np.ndarray  # index into coarse_ids, -1 if parent not retained
    full_parent_raw: np.ndarray  # global COARSE type id
    median_scaffold_freq: float
    meta: dict[str, Any]


def _select(
    ids: np.ndarray,
    stats: Mapping[str, np.ndarray],
    keys: Sequence[bytes],
    min_df: int,
    min_scaf: int,
    cap: int,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    df = stats["document_frequency_train"]
    scaf = stats["scaffold_frequency_train"]
    ok = (df >= min_df) & (scaf >= min_scaf)
    if allowed is not None:
        ok &= allowed
    cand = np.flatnonzero(ok)
    order = sorted(cand.tolist(), key=lambda i: (-int(df[i]), -int(scaf[i]), keys[i]))
    return np.asarray(order[:cap], dtype=np.int64)


def freeze_vocab(
    discovered: Discovered,
    splits: Mapping[str, np.ndarray],
    label: np.ndarray,
    scaffold_ids: np.ndarray,
    n_scaffolds: int,
) -> FrozenVocab:
    train = splits["train"]
    cstats = type_stats(discovered.coarse, train, splits["valid"], splits["test"], label, scaffold_ids, n_scaffolds)
    fstats = type_stats(discovered.full, train, splits["valid"], splits["test"], label, scaffold_ids, n_scaffolds)

    coarse_ids: list[int] = []
    full_ids: list[int] = []
    parent_in_vocab: list[int] = []
    for fam in range(len(C.FAMILIES)):
        fam_mask_c = discovered.coarse_family == fam
        allowed = fam_mask_c
        keep_c = _select(
            np.flatnonzero(fam_mask_c),
            cstats,
            discovered.coarse_keys,
            C.COARSE_MIN_DOC_FREQ,
            C.COARSE_MIN_SCAFFOLDS,
            C.COARSE_MAX_PER_FAMILY,
            allowed=allowed,
        )
        coarse_index = {int(t): i + len(coarse_ids) for i, t in enumerate(keep_c.tolist())}
        coarse_ids.extend(int(t) for t in keep_c.tolist())

        fam_mask_f = discovered.full_family == fam
        parent = discovered.full_parent
        parent_kept = np.zeros(discovered.full.shape[1], dtype=bool)
        for t in range(discovered.full.shape[1]):
            parent_kept[t] = int(parent[t]) in coarse_index
        keep_f = _select(
            np.flatnonzero(fam_mask_f),
            fstats,
            discovered.full_keys,
            C.FULL_MIN_DOC_FREQ,
            C.FULL_MIN_SCAFFOLDS,
            C.FULL_MAX_PER_FAMILY,
            allowed=fam_mask_f & parent_kept,
        )
        for t in keep_f.tolist():
            full_ids.append(int(t))
            parent_in_vocab.append(coarse_index[int(parent[t])])

    coarse_ids_arr = np.asarray(coarse_ids, dtype=np.int64)
    full_ids_arr = np.asarray(full_ids, dtype=np.int64)
    coarse_family = discovered.coarse_family[coarse_ids_arr]
    full_family = discovered.full_family[full_ids_arr]
    cstats_sel = {k: v[coarse_ids_arr] for k, v in cstats.items()}
    fstats_sel = {k: v[full_ids_arr] for k, v in fstats.items()}

    m_f = fstats_sel["scaffold_frequency_train"].astype(np.float64)
    m_med = float(np.median(m_f))
    a = np.clip(np.sqrt(m_med / np.maximum(m_f, 1.0)), 0.25, 4.0)

    rng = np.random.default_rng(C.SEED)
    a_null = a.copy()
    for fam in range(len(C.FAMILIES)):
        idx = np.flatnonzero(full_family == fam)
        if idx.size > 1:
            a_null[idx] = a[rng.permutation(idx)]

    meta = {
        "filters": {
            "coarse": {"min_document_frequency_train": C.COARSE_MIN_DOC_FREQ,
                        "min_scaffold_frequency_train": C.COARSE_MIN_SCAFFOLDS,
                        "max_types_per_family": C.COARSE_MAX_PER_FAMILY},
            "full": {"min_document_frequency_train": C.FULL_MIN_DOC_FREQ,
                      "min_scaffold_frequency_train": C.FULL_MIN_SCAFFOLDS,
                      "max_types_per_family": C.FULL_MAX_PER_FAMILY,
                      "requires_coarse_parent_retained": True},
            "ranking": "document_frequency desc, scaffold_frequency desc, canonical key asc",
            "family_order": list(C.FAMILIES),
        },
        "n_selected_coarse": int(coarse_ids_arr.size),
        "n_selected_full": int(full_ids_arr.size),
        "n_selected_coarse_by_family": {
            fam: int((coarse_family == i).sum()) for i, fam in enumerate(C.FAMILIES)
        },
        "n_selected_full_by_family": {
            fam: int((full_family == i).sum()) for i, fam in enumerate(C.FAMILIES)
        },
        "median_full_scaffold_frequency_train": m_med,
        "support_weight": {
            "definition": "a_f = clip(sqrt(m_med/m_f), 0.25, 4.0); m_f = #distinct Train scaffolds of FULL type f",
            "min": float(a.min()) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "max": float(a.max()) if a.size else None,
            "mean": float(a.mean()) if a.size else None,
            "n_at_lower_clip": int((a <= 0.2500001).sum()),
            "n_at_upper_clip": int((a >= 3.9999999).sum()),
        },
        "null_support_weight": {
            "definition": "a_f permuted within structural family, numpy default_rng(seed=0)",
            "n_changed": int((a_null != a).sum()),
            "fraction_changed": float((a_null != a).mean()),
        },
    }
    return FrozenVocab(
        coarse_ids=coarse_ids_arr,
        full_ids=full_ids_arr,
        coarse_family=coarse_family,
        full_family=full_family,
        full_stats=fstats_sel,
        coarse_stats=cstats_sel,
        support_weight=a,
        support_weight_null=a_null,
        full_parent_in_vocab=np.asarray(parent_in_vocab, dtype=np.int64),
        full_parent_raw=discovered.full_parent[full_ids_arr],
        median_scaffold_freq=m_med,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# design matrices
# ---------------------------------------------------------------------------


def scaled_blocks(
    discovered: Discovered,
    vocab: FrozenVocab,
    n_atoms: np.ndarray,
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    """C and F matrices with the frozen ``count / sqrt(n_atoms)`` scaling."""
    inv = (1.0 / np.sqrt(np.maximum(n_atoms.astype(np.float64), 1.0))).astype(np.float64)
    Cm = discovered.coarse[:, vocab.coarse_ids].astype(np.float64)
    Fm = discovered.full[:, vocab.full_ids].astype(np.float64)
    Cm = csr_matrix(Cm.multiply(inv[:, None]))
    Fm = csr_matrix(Fm.multiply(inv[:, None]))
    Cm.eliminate_zeros()
    Fm.eliminate_zeros()
    return Cm, Fm, {"scaling": "count / sqrt(max(n_atoms, 1))", "n_columns_coarse": int(Cm.shape[1]),
                    "n_columns_full": int(Fm.shape[1])}


def parent_remainder(Cm: csr_matrix, Fm: csr_matrix, vocab: FrozenVocab) -> csr_matrix:
    """R_p = C_p - sum_f F_f over the retained FULL children of p (per molecule)."""
    n, n_coarse = Cm.shape
    rows = []
    cols = []
    vals = []
    Fm_csc = Fm.tocsc()
    for p in range(n_coarse):
        children = np.flatnonzero(vocab.full_parent_in_vocab == p)
        block = np.asarray(Cm[:, p].todense()).reshape(-1)
        if children.size:
            block = block - np.asarray(Fm_csc[:, children].sum(axis=1)).reshape(-1)
        if np.any(block < -1e-9):
            raise ValueError(f"negative remainder for coarse type {p}: {block.min()}")
        nonzero = np.flatnonzero(block > 1e-12)
        if nonzero.size:
            rows.append(nonzero)
            cols.append(np.full(nonzero.size, p, dtype=np.int64))
            vals.append(block[nonzero])
    rows_np = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols_np = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals_np = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float64)
    return csr_matrix((vals_np, (rows_np, cols_np)), shape=(n, n_coarse))


def parent_shuffle(vocab: FrozenVocab) -> tuple[np.ndarray, dict[str, Any]]:
    """Shuffle FULL->COARSE parent assignments *within* structural family (seed 0).

    The multiset of parents per family is preserved, hence so is the number of
    children per parent and the remaining-count vector R_p.
    """
    rng = np.random.default_rng(C.SEED)
    shuffled = np.empty_like(vocab.full_parent_in_vocab)
    changed = 0
    for fam in range(len(C.FAMILIES)):
        idx = np.flatnonzero(vocab.full_family == fam)
        if idx.size == 0:
            continue
        parents = vocab.full_parent_in_vocab[idx]
        perm = rng.permutation(idx.size)
        shuffled[idx] = parents[perm]
        changed += int((shuffled[idx] != parents).sum())
    info = {
        "definition": "permute the selected FULL types' parent labels within family, numpy default_rng(seed=0)",
        "n_full_types": int(vocab.full_parent_in_vocab.size),
        "n_changed": int(changed),
        "fraction_changed": float(changed / max(vocab.full_parent_in_vocab.size, 1)),
        "parents_multiset_preserved": bool(
            sorted(vocab.full_parent_in_vocab.tolist()) == sorted(shuffled.tolist())
        ),
        "strength": "ADEQUATE" if changed / max(vocab.full_parent_in_vocab.size, 1) >= 0.5 else "INSUFFICIENT",
    }
    return shuffled, info


def null_coarse(
    R: csr_matrix,
    Fm: csr_matrix,
    shuffled_parent: np.ndarray,
    n_coarse: int,
) -> csr_matrix:
    """C_null_p = R_p + sum_{f: shuffled_parent(f)=p} F_f."""
    Fm_csc = Fm.tocsc()
    rows, cols, vals = [], [], []
    for p in range(n_coarse):
        children = np.flatnonzero(shuffled_parent == p)
        if children.size:
            block = np.asarray(Fm_csc[:, children].sum(axis=1)).reshape(-1)
        else:
            block = np.zeros(Fm.shape[0], dtype=np.float64)
        block = block + np.asarray(R[:, p].todense()).reshape(-1)
        nz = np.flatnonzero(block > 1e-12)
        if nz.size:
            rows.append(nz)
            cols.append(np.full(nz.size, p, dtype=np.int64))
            vals.append(block[nz])
    rows_np = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols_np = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals_np = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float64)
    return csr_matrix((vals_np, (rows_np, cols_np)), shape=(n_mol := Fm.shape[0], n_coarse))
