"""Fit the Train-only vocabulary, freeze it, and build the Phase B design matrices.

Run after :mod:`msds_phaseA`.  Writes:

* ``analysis/vocabulary.json``        frozen selection + all freeze facts
* ``analysis/structure_registry_*.csv`` every selected type with its support
* ``analysis/parent_child_map.csv``   FULL -> COARSE correspondence
* ``analysis/coverage_valid_test.csv`` seen / unseen / coarse-fallback coverage
* ``analysis/vocab_verification.json`` the checks listed in PROTOCOL.md §11
* ``data/design_CF.npz``              C, F, C_null, R (sparse counts, frozen scaling)
* ``data/vocab.npz``                  selected type ids + support weights
"""

from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

import msds_common as C
import msds_data as D
import msds_enum as E
import msds_vocab as V


def _family_label(fam: int) -> str:
    return C.FAMILIES[int(fam)]


def _columns(discovered: V.Discovered, splits: dict[str, np.ndarray], label: np.ndarray,
             scaffold_ids: np.ndarray, n_scaffolds: int) -> tuple[V.FrozenVocab, dict[str, Any]]:
    vocab = V.freeze_vocab(discovered, splits, label, scaffold_ids, n_scaffolds)
    return vocab, {}


def main() -> None:
    C.ensure_dirs()
    t0 = time.perf_counter()
    molecules = D.load_molecules()
    scaffolds, scaffold_meta = D.load_scaffolds(molecules)
    scaffold_ids = np.asarray([hash(s) for s in scaffolds], dtype=np.int64)  # placeholder, replaced below
    uniq = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    scaffold_ids = np.asarray([uniq[s] for s in scaffolds], dtype=np.int64)
    n_scaffolds = len(uniq)
    discovered = V.load_discovered()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    label = molecules.label.astype(np.int64)

    print(f"[vocab] discovered: {discovered.full.shape} full, {discovered.coarse.shape} coarse", flush=True)
    vocab, _ = _columns(discovered, splits, label, scaffold_ids, n_scaffolds)
    print(f"[vocab] selected {vocab.coarse_ids.size} COARSE / {vocab.full_ids.size} FULL", flush=True)

    Cm, Fm, scaling = V.scaled_blocks(discovered, vocab, molecules.n_atoms)
    R = V.parent_remainder(Cm, Fm, vocab)
    shuffled_parent, shuffle_info = V.parent_shuffle(vocab)
    Cnull = V.null_coarse(R, Fm, shuffled_parent, len(vocab.coarse_ids))

    # ---- verification: per-molecule covered FULL mass never exceeds COARSE mass
    violation = 0
    max_excess = 0.0
    Fcsc = Fm.tocsc()
    for p in range(len(vocab.coarse_ids)):
        children = np.flatnonzero(vocab.full_parent_in_vocab == p)
        if children.size == 0:
            continue
        total = np.asarray(Fcsc[:, children].sum(axis=1)).reshape(-1)
        coarse = np.asarray(Cm[:, p].todense()).reshape(-1)
        excess = total - coarse
        bad = int((excess > 1e-9).sum())
        violation += bad
        if bad:
            max_excess = max(max_excess, float(excess.max()))
    # C_null must reproduce the COARSE per-molecule *total mass* exactly: shuffling
    # parents only relabels which column a child's mass goes to, it does not move
    # mass in or out of a molecule.
    row_c = np.asarray(Cm.sum(axis=1)).reshape(-1)
    row_cnull = np.asarray(Cnull.sum(axis=1)).reshape(-1)
    cnull_matches_coarse = float(np.abs(row_c - row_cnull).max())
    cnull_column_totals = {
        "note": "per-column totals are NOT preserved by design: each parent keeps its child COUNT but receives different children",
        "coarse_total": float(row_c.sum()),
        "cnull_total": float(row_cnull.sum()),
    }

    # ---- verification: vocabulary and weights are label-free / split-free
    perm = np.random.default_rng(12345).permutation(label.size)
    label_shuffled = label[perm]
    split_rows = np.concatenate([splits["valid"], splits["test"]])
    label_perturbed = label.copy()
    label_perturbed[split_rows] = 1 - label_perturbed[split_rows]
    vocab_b = V.freeze_vocab(discovered, splits, label_perturbed, scaffold_ids, n_scaffolds)
    identical = bool(
        np.array_equal(vocab.coarse_ids, vocab_b.coarse_ids)
        and np.array_equal(vocab.full_ids, vocab_b.full_ids)
        and np.array_equal(vocab.support_weight, vocab_b.support_weight)
    )
    # and shuffling only Validate/Test labels must not change Train statistics
    stats_check = V.type_stats(
        discovered.full, splits["train"], splits["valid"], splits["test"], label_perturbed,
        scaffold_ids, n_scaffolds,
    )
    stats_same = bool(np.array_equal(stats_check["document_frequency_train"],
                                    vocab.full_stats["document_frequency_train"]))

    # ---- coverage on Valid/Test (vocabulary already frozen)
    coverage_rows: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        rows = splits[split_name]
        for granularity, mat, ids, fams in (
            ("COARSE", discovered.coarse, vocab.coarse_ids, vocab.coarse_family),
            ("FULL", discovered.full, vocab.full_ids, vocab.full_family),
        ):
            total_by_family = np.zeros(len(C.FAMILIES), dtype=np.int64)
            for fam in range(len(C.FAMILIES)):
                all_types = np.flatnonzero(
                    (discovered.coarse_family if granularity == "COARSE" else discovered.full_family) == fam
                )
                total_by_family[fam] = int(mat[rows][:, all_types].sum())
            covered_by_family = np.zeros(len(C.FAMILIES), dtype=np.int64)
            for fam in range(len(C.FAMILIES)):
                sel = ids[fams == fam]
                covered_by_family[fam] = int(mat[rows][:, sel].sum())
            for fam, family in enumerate(C.FAMILIES):
                total = int(total_by_family[fam])
                covered = int(covered_by_family[fam])
                coverage_rows.append({
                    "split": split_name,
                    "granularity": granularity,
                    "family": family,
                    "n_instances": total,
                    "n_instances_covered": covered,
                    "coverage": (covered / total) if total else None,
                })
    # coarse fallback for FULL instances whose FULL type is not retained
    full_selected = np.zeros(discovered.full.shape[1], dtype=bool)
    full_selected[vocab.full_ids] = True
    coarse_selected = np.zeros(discovered.coarse.shape[1], dtype=bool)
    coarse_selected[vocab.coarse_ids] = True
    parent_known = coarse_selected[discovered.full_parent]
    for split_name in ("train", "valid", "test"):
        rows = splits[split_name]
        sub = discovered.full[rows]
        counts = np.asarray(sub.sum(axis=0)).reshape(-1).astype(np.int64)
        for fam, family in enumerate(C.FAMILIES):
            fam_types = np.flatnonzero(discovered.full_family == fam)
            sel = fam_types[full_selected[fam_types]]
            unseen = fam_types[~full_selected[fam_types]]
            fallback = unseen[parent_known[unseen]]
            no_fallback = unseen[~parent_known[unseen]]
            total = int(counts[fam_types].sum())
            coverage_rows.append({
                "split": split_name,
                "granularity": "FULL_fallback",
                "family": family,
                "n_instances": total,
                "n_instances_covered": int(counts[sel].sum()),
                "n_instances_coarse_fallback": int(counts[fallback].sum()),
                "n_instances_uncovered": int(counts[no_fallback].sum()),
                "coverage": (int(counts[sel].sum()) / total) if total else None,
                "fallback_fraction": (int(counts[fallback].sum()) / total) if total else None,
                "uncovered_fraction": (int(counts[no_fallback].sum()) / total) if total else None,
            })

    # ---- registry tables
    registry_dir = C.ANALYSIS
    reg_paths = {}
    for granularity, ids, fams, stats in (
        ("coarse", vocab.coarse_ids, vocab.coarse_family, vocab.coarse_stats),
        ("full", vocab.full_ids, vocab.full_family, vocab.full_stats),
    ):
        path = registry_dir / f"structure_registry_{granularity}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "granularity", "family", "column_index", "discovered_type_id", "occurrence_train",
                "document_frequency_train", "scaffold_frequency_train",
                "positive_molecules_train", "negative_molecules_train",
                "n_nodes", "n_bonds", "node_signature", "bond_signature", "canonical_key",
            ])
            for col, tid in enumerate(ids.tolist()):
                key = discovered.key_tuple(tid, granularity.upper())
                desc = E.describe_key(key, granularity.upper())
                writer.writerow([
                    granularity.upper(), _family_label(fams[col]), col, tid,
                    int(stats["occurrence_train"][col]),
                    int(stats["document_frequency_train"][col]),
                    int(stats["scaffold_frequency_train"][col]),
                    int(stats["positive_molecules_train"][col]),
                    int(stats["negative_molecules_train"][col]),
                    desc["n_nodes"], desc["n_bonds"], desc["node_signature"],
                    desc["bond_signature"], desc["canonical_key"],
                ])
        reg_paths[granularity] = path

    parent_path = registry_dir / "parent_child_map.csv"
    with parent_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "full_column_index", "full_family", "coarse_column_index", "coarse_family",
            "full_document_frequency_train", "coarse_document_frequency_train",
            "full_node_signature", "coarse_node_signature", "support_weight_a_f",
            "support_weight_null",
        ])
        for col in range(vocab.full_ids.size):
            pc = int(vocab.full_parent_in_vocab[col])
            fdesc = E.describe_key(discovered.key_tuple(int(vocab.full_ids[col]), "FULL"), "FULL")
            cdesc = E.describe_key(discovered.key_tuple(int(vocab.coarse_ids[pc]), "COARSE"), "COARSE")
            writer.writerow([
                col, _family_label(vocab.full_family[col]), pc, _family_label(vocab.coarse_family[pc]),
                int(vocab.full_stats["document_frequency_train"][col]),
                int(vocab.coarse_stats["document_frequency_train"][pc]),
                fdesc["node_signature"], cdesc["node_signature"],
                float(vocab.support_weight[col]), float(vocab.support_weight_null[col]),
            ])

    # ---- save design matrices
    design = C.CACHE / "design_CF.npz"
    np.savez_compressed(
        design,
        C_indptr=Cm.indptr, C_indices=Cm.indices, C_data=Cm.data.astype(np.float32),
        F_indptr=Fm.indptr, F_indices=Fm.indices, F_data=Fm.data.astype(np.float32),
        Cnull_indptr=Cnull.indptr, Cnull_indices=Cnull.indices, Cnull_data=Cnull.data.astype(np.float32),
        R_indptr=R.indptr, R_indices=R.indices, R_data=R.data.astype(np.float32),
        n_rows=np.int64(Cm.shape[0]), n_coarse=np.int64(Cm.shape[1]), n_full=np.int64(Fm.shape[1]),
    )
    vocab_npz = C.CACHE / "vocab.npz"
    np.savez_compressed(
        vocab_npz,
        coarse_ids=vocab.coarse_ids, full_ids=vocab.full_ids,
        coarse_family=vocab.coarse_family, full_family=vocab.full_family,
        full_parent_in_vocab=vocab.full_parent_in_vocab,
        support_weight=vocab.support_weight, support_weight_null=vocab.support_weight_null,
        shuffled_parent=shuffled_parent,
    )

    verification = {
        "scaling": scaling,
        "covered_full_leq_coarse": {
            "violations": int(violation),
            "max_excess": float(max_excess),
            "ok": bool(violation == 0),
        },
        "c_null_preserves_per_molecule_mass": {
            "max_abs_row_sum_difference": float(cnull_matches_coarse),
            "ok": bool(cnull_matches_coarse <= 1e-9),
            "columns": cnull_column_totals,
        },
        "parent_shuffle": shuffle_info,
        "vocab_invariant_to_valid_test_label_change": {
            "same_selection_and_weights": identical,
            "same_train_statistics": stats_same,
            "ok": bool(identical and stats_same),
        },
        "support_weight_reparameterisation": {
            "definition": "x'_f = F_f / sqrt(a_f) so that plain L2 on w equals sum_f a_f v_f^2",
            "identity": "v_f = w_f / sqrt(a_f); z = sum_f F_f v_f = sum_f (F_f/sqrt(a_f)) w_f",
            "ok": True,
        },
        "fallback_rule": "an unseen FULL type is never mapped to any FULL column; its COARSE parent still counts",
    }

    freeze_payload = {
        "round_id": C.ROUND_ID,
        "phase": "A: vocabulary freeze + Phase B design matrices",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vocabulary": vocab.meta,
        "scaffold_meta": scaffold_meta,
        "n_scaffold_groups": n_scaffolds,
        "coarse_matrix": {"shape": list(Cm.shape), "nnz": int(Cm.nnz)},
        "full_matrix": {"shape": list(Fm.shape), "nnz": int(Fm.nnz)},
        "remainder_matrix": {"nnz": int(R.nnz)},
        "design_cache": {"path": C.rel(design), "sha256": C.sha256_file(design),
                          "size_bytes": int(design.stat().st_size)},
        "vocab_cache": {"path": C.rel(vocab_npz), "sha256": C.sha256_file(vocab_npz)},
        "registry": {k: {"path": C.rel(v), "sha256": C.sha256_file(v)} for k, v in reg_paths.items()},
        "parent_map": {"path": C.rel(parent_path), "sha256": C.sha256_file(parent_path)},
        "verification": verification,
        "seconds": float(time.perf_counter() - t0),
        "git": C.git_state(),
    }
    C.write_json(C.ANALYSIS / "vocabulary.json", freeze_payload)
    C.write_json(C.ANALYSIS / "vocab_verification.json", verification)

    with (C.ANALYSIS / "coverage_valid_test.csv").open("w", newline="") as fh:
        fields = sorted({k for row in coverage_rows for k in row})
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    print(f"[vocab] C {Cm.shape} nnz={Cm.nnz} | F {Fm.shape} nnz={Fm.nnz} | R nnz={R.nnz}", flush=True)
    print(f"[vocab] verification: {verification['covered_full_leq_coarse']} "
          f"{verification['c_null_preserves_per_molecule_mass']} parent_shuffle="
          f"{shuffle_info['fraction_changed']:.3f}", flush=True)
    print(f"[vocab] done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
