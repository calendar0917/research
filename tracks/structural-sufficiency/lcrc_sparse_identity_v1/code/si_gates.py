"""Correctness gates for the sparse-identity round (CPU only, label-free).

Produces

    analysis/feature_vocabulary.json
    analysis/feature_checks.json       gates 1-5, 9 (+ 18-ring witness)
    analysis/null_checks.json          gates 7-8
    analysis/initialization_checks.json gates 6, 10, 11, 12
    analysis/correctness_gates.json    aggregate 1-14

Gates 13 (checkpoint reload) and 14 (no Test access) are enforced per training run
and again at freeze time; this script records their static/precondition state and
``si_freeze.py`` completes them.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import si_common as C
import si_features as FEAT
import si_model as M
import si_train as TRAIN

import dissect_common as DC
import lcrc_data as PREP_DATA
import ptb_data as PD
import ptb_model as PM
import ptb_train_mol as PT

RING18_SMILES_KEYS = tuple(C.RING18_PAIR)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _key_multiset(keys: Sequence[Sequence[int]]) -> dict[tuple[int, ...], int]:
    out: dict[tuple[int, ...], int] = {}
    for key in keys:
        t = tuple(int(x) for x in key)
        out[t] = out.get(t, 0) + 1
    return out


def _mol_keys(mol: Mapping[str, Any], spec) -> list[tuple[int, int, int, int]]:
    path_ptr = np.asarray(mol["path_ptr"], dtype=np.int64)
    path_cat = np.asarray(mol["path_cat"], dtype=np.int64)
    path_end63 = np.asarray(mol["path_end63"], dtype=np.int64)
    path_closure = np.asarray(mol["path_closure"], dtype=np.int64)
    atom_type = np.asarray(mol["atom_type"], dtype=np.int64)
    out: list[tuple[int, int, int, int]] = []
    for atom in range(int(mol["n_atoms"])):
        for p in range(int(path_ptr[atom]), int(path_ptr[atom + 1])):
            out.append(
                (int(atom_type[atom]), int(path_cat[p]), int(path_end63[p]), int(path_closure[p]))
            )
    return out


def _single_molecule_vector(mol: Mapping[str, Any], spec, vocab) -> np.ndarray:
    cache = PREP_DATA.concat_molecules([dict(mol)], spec, split="one")
    return FEAT.encode_split(cache, spec, vocab)["x"][0]


def _smiles_molecule(smiles: str, spec):
    import mb2_rdkit as RDK

    raw = RDK.smiles_to_raw_row(smiles, 0)
    graph = RDK.raw_row_to_graph(raw, 0, spec)
    return PREP_DATA.build_molecule(graph, spec)


def _renumbered_molecule(smiles: str, spec, order: Sequence[int]):
    from rdkit import Chem

    import mb2_rdkit as RDK

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    Chem.SanitizeMol(mol)
    ren = Chem.RenumberAtoms(mol, [int(x) for x in order])
    raw = RDK._mol_to_raw_row(ren, Chem.MolToSmiles(ren), 0)
    graph = RDK.raw_row_to_graph(raw, 0, spec)
    return PREP_DATA.build_molecule(graph, spec)


# ---------------------------------------------------------------------------
# gate 1: split alignment
# ---------------------------------------------------------------------------


def gate_split_alignment(view) -> dict[str, Any]:
    import json

    manifest_path = (
        C.REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/split_manifest_ogbg-molhiv.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sizes = {k: len(view.rows[k]) for k in ("train", "valid", "test")}
    row_hashes = {k: C.index_sha256(view.rows[k]) for k in ("train", "valid", "test")}
    y_train = np.asarray(view.targets.y_train).reshape(-1)
    y_valid = np.asarray(view.targets.y_valid).reshape(-1)
    details = {
        "source": view.source_meta,
        "sizes": sizes,
        "expected_sizes": C.EXPECTED_SPLIT_SIZES,
        "row_sha256": row_hashes,
        "frozen_row_sha256": manifest["rows_sha256"],
        "split_file_sha256": view.source_meta["split_file_sha256"],
        "frozen_split_file_sha256": manifest["selected_split_sha256"],
        "train_cache_n_molecules": int(view.train_cache["n_molecules"]),
        "valid_cache_n_molecules": int(view.valid_cache["n_molecules"]),
        "train_cache_rows_match_view": bool(
            np.array_equal(np.asarray(view.train_cache["rows"]), np.asarray(view.rows["train"], dtype=np.int64))
        ),
        "valid_cache_rows_match_view": bool(
            np.array_equal(np.asarray(view.valid_cache["rows"]), np.asarray(view.rows["valid"], dtype=np.int64))
        ),
        "labels": {
            "train_n": int(y_train.size),
            "valid_n": int(y_valid.size),
            "train_finite": bool(np.isfinite(y_train).all()),
            "valid_finite": bool(np.isfinite(y_valid).all()),
            "train_classes": sorted(set(int(x) for x in y_train)),
            "valid_classes": sorted(set(int(x) for x in y_valid)),
            "train_positive_rate": float(y_train.mean()),
            "valid_positive_rate": float(y_valid.mean()),
        },
        "labels_aligned_with_cache": bool(
            y_train.size == int(view.train_cache["n_molecules"]) and y_valid.size == int(view.valid_cache["n_molecules"])
        ),
        "feature_spec_hash": view.spec.hash(),
        "frozen_feature_spec_hash": C.FROZEN_SPEC_HASH,
        "train_cache_content_fingerprint": view.train_cache["meta"].get("content_fingerprint"),
        "valid_cache_content_fingerprint": view.valid_cache["meta"].get("content_fingerprint"),
        "test_labels_read": False,
        "test_cache_built": False,
    }
    passed = bool(
        sizes == C.EXPECTED_SPLIT_SIZES
        and row_hashes == manifest["rows_sha256"]
        and details["split_file_sha256"] == manifest["selected_split_sha256"]
        and details["train_cache_rows_match_view"]
        and details["valid_cache_rows_match_view"]
        and details["labels_aligned_with_cache"]
        and view.spec.hash() == C.FROZEN_SPEC_HASH
    )
    return {"id": 1, "name": "official_split_alignment", "pass": passed, "details": details}


# ---------------------------------------------------------------------------
# gate 2: vocabulary from Train structures only
# ---------------------------------------------------------------------------


def gate_vocabulary(vocab, train_cache, valid_cache, spec, feat_train) -> dict[str, Any]:
    sig = inspect.signature(FEAT.build_vocabulary)
    rebuilt = FEAT.build_vocabulary(train_cache, spec, C.K)
    records = FEAT.path_keys(train_cache, spec)
    rng = np.random.default_rng(20261103)
    sample = sorted(int(x) for x in rng.choice(int(train_cache["n_molecules"]), size=200, replace=False))
    # loop-based document frequency restricted to the sampled molecules
    loop_counts: dict[tuple[int, int, int, int], int] = {}
    for g in sample:
        keys = FEAT.molecule_keys_loop(train_cache, spec, g)
        for key in set(keys):
            loop_counts[key] = loop_counts.get(key, 0) + 1
    sel = np.isin(records["mol_of_path"], np.asarray(sample, dtype=np.int64))
    vectorised = FEAT.document_frequency(
        {"mol_of_path": records["mol_of_path"][sel], "key": records["key"][sel]}
    )
    mismatches = []
    for key, count in loop_counts.items():
        code = int(FEAT.encode_key([key[0]], [key[1]], [key[2]], [key[3]], spec)[0])
        if int(vectorised.get(code, 0)) != int(count):
            mismatches.append({"key": list(key), "loop": int(count), "vectorised": int(vectorised.get(code, 0))})
    # a Valid-only vocabulary must not reproduce the frozen one (proves the source)
    valid_only = FEAT.build_vocabulary(valid_cache, spec, C.K)
    details = {
        "signature": str(sig),
        "function_inputs": list(sig.parameters),
        "labels_used": False,
        "valid_or_test_used": False,
        "train_n_molecules": int(train_cache["n_molecules"]),
        "n_unique_keys": int(vocab["n_unique_keys"]),
        "n_selected": int(len(vocab["keys"])),
        "k": int(vocab["k"]),
        "rebuilt_hash_matches": bool(rebuilt["vocabulary_sha256"] == vocab["vocabulary_sha256"]),
        "sample_molecules_for_recount": len(sample),
        "document_frequency_recount_mismatches": mismatches[:10],
        "n_document_frequency_recount_mismatches": int(len(mismatches)),
        "valid_only_vocabulary_hash": valid_only.get("vocabulary_sha256"),
        "valid_only_vocabulary_differs_from_train_vocabulary": bool(
            valid_only.get("vocabulary_sha256") != vocab["vocabulary_sha256"]
        ),
        "train_vector_stats": feat_train["stats"],
    }
    passed = bool(
        list(sig.parameters) == ["train_cache", "spec", "k"]
        and rebuilt["vocabulary_sha256"] == vocab["vocabulary_sha256"]
        and not mismatches
        and vocab.get("labels_used") is False
        and vocab.get("valid_or_test_used") is False
        and int(vocab["train_n_molecules"]) == int(train_cache["n_molecules"])
    )
    return {"id": 2, "name": "vocabulary_train_structures_only", "pass": passed, "details": details}


# ---------------------------------------------------------------------------
# gate 3: key construction matches production S2 arrays
# ---------------------------------------------------------------------------


def gate_key_construction(train_cache, spec) -> dict[str, Any]:
    records = FEAT.path_keys(train_cache, spec)
    rng = np.random.default_rng(20261104)
    sample = sorted(int(x) for x in rng.choice(int(train_cache["n_molecules"]), size=120, replace=False))
    mismatches = []
    n_checked = 0
    for g in sample:
        loop = sorted(FEAT.molecule_keys_loop(train_cache, spec, g))
        sel = records["mol_of_path"] == int(g)
        vec = sorted(
            tuple(int(x) for x in FEAT.decode_key(int(code), spec)) for code in records["key"][sel].tolist()
        )
        n_checked += len(loop)
        if loop != vec:
            mismatches.append({"molecule_index": int(g), "n_loop": len(loop), "n_vectorised": len(vec)})
    example = sample[0]
    atom_ptr = np.asarray(train_cache["atom_ptr"])
    a0, a1 = int(atom_ptr[example]), int(atom_ptr[example + 1])
    path_ptr = np.asarray(train_cache["path_ptr"])
    atom_type = np.asarray(train_cache["atom_type"])
    path_cat = np.asarray(train_cache["path_cat"])
    path_end63 = np.asarray(train_cache["path_end63"])
    path_closure = np.asarray(train_cache["path_closure"])
    raw_example = []
    for atom in range(a0, min(a0 + 2, a1)):
        for p in range(int(path_ptr[atom]), min(int(path_ptr[atom + 1]), int(path_ptr[atom]) + 3)):
            raw_example.append(
                {
                    "root_atom_index": int(atom),
                    "root_atom_type": int(atom_type[atom]),
                    "path_cat": int(path_cat[p]),
                    "path_end63": int(path_end63[p]),
                    "path_closure": int(path_closure[p]),
                    "key": [int(x) for x in FEAT.decode_key(
                        int(FEAT.encode_key([atom_type[atom]], [path_cat[p]], [path_end63[p]], [path_closure[p]], spec)[0]),
                        spec,
                    )],
                }
            )
    details = {
        "n_molecules_checked": len(sample),
        "n_path_records_checked": int(n_checked),
        "mismatches": mismatches,
        "raw_example": raw_example,
        "key_codec": "((root_atom_type * n_path_codes + path_cat) * n_nbr_codes + path_end63) * 2 + path_closure",
        "n_path_codes": int(spec.n_path_codes),
        "n_nbr_codes": int(spec.n_nbr_codes),
        "n_atom_types": int(spec.n_atom_types),
        "records_are_production_cache_order": True,
    }
    return {"id": 3, "name": "key_matches_production_s2", "pass": not mismatches, "details": details}


# ---------------------------------------------------------------------------
# gate 4: invariance to atom renumbering and path record order
# ---------------------------------------------------------------------------


def gate_invariance(train_cache, spec, vocab) -> dict[str, Any]:
    rng = np.random.default_rng(20261105)
    smiles = list(train_cache["smiles"])
    picks = sorted(int(x) for x in rng.choice(len(smiles), size=6, replace=False))
    rows = []
    ok_renumber = True
    ok_order = True
    for g in picks:
        smi = str(smiles[g])
        base = _smiles_molecule(smi, spec)
        base_keys = _key_multiset(_mol_keys(base, spec))
        vec0 = _single_molecule_vector(base, spec, vocab)
        order = list(rng.permutation(int(base["n_atoms"])))
        ren = _renumbered_molecule(smi, spec, order)
        ren_keys = _key_multiset(_mol_keys(ren, spec))
        vec1 = _single_molecule_vector(ren, spec, vocab)
        # reverse each root atom's S2 record block
        reversed_mol = dict(base)
        path_ptr = np.asarray(base["path_ptr"], dtype=np.int64)
        order_idx: list[int] = []
        for atom in range(int(base["n_atoms"])):
            order_idx.extend(range(int(path_ptr[atom]), int(path_ptr[atom + 1]))[::-1])
        for field in ("path_cat", "path_end63", "path_end", "path_closure"):
            reversed_mol[field] = np.asarray(base[field])[order_idx]
        vec2 = _single_molecule_vector(reversed_mol, spec, vocab)
        r_ok = bool(base_keys == ren_keys and np.array_equal(vec0, vec1))
        o_ok = bool(np.array_equal(vec0, vec2))
        ok_renumber = ok_renumber and r_ok
        ok_order = ok_order and o_ok
        rows.append(
            {
                "train_row": int(g),
                "smiles": smi,
                "n_atoms": int(base["n_atoms"]),
                "n_s2_paths": int(path_ptr[-1]),
                "n_active_patterns": int((vec0 > 0).sum()),
                "atom_renumbering_invariant": r_ok,
                "path_order_invariant": o_ok,
                "vector_sha256": C.array_hash_f32(vec0.reshape(1, -1)),
            }
        )
    details = {
        "n_molecules": len(rows),
        "per_molecule": rows,
        "atom_renumbering_invariant": ok_renumber,
        "path_order_invariant": ok_order,
        "method": "RDKit RenumberAtoms -> frozen raw-attribute extraction -> frozen graph builder; "
        "path-record block reversal inside the production molecule dict",
    }
    return {
        "id": 4,
        "name": "invariance_to_renumbering_and_order",
        "pass": bool(ok_renumber and ok_order),
        "details": details,
    }


# ---------------------------------------------------------------------------
# gate 5: batched vs single-molecule vectors
# ---------------------------------------------------------------------------


def gate_batching(train_cache, spec, vocab) -> dict[str, Any]:
    train_enc = FEAT.encode_split(train_cache, spec, vocab)
    rng = np.random.default_rng(20261106)
    picks = sorted(int(x) for x in rng.choice(int(train_cache["n_molecules"]), size=5, replace=False))
    rows = []
    ok = True
    for g in picks:
        subset = PT._subset_cache(train_cache, [int(g)], spec)
        one = FEAT.encode_split(subset, spec, vocab)
        same = bool(np.array_equal(one["x"][0], train_enc["x"][int(g)]))
        ok = ok and same
        rows.append({"train_row": int(g), "identical": same})
    # a batch of many rows must equal the row-wise slice of the full encoding
    batch_rows = sorted(int(x) for x in rng.choice(int(train_cache["n_molecules"]), size=64, replace=False))
    subset = PT._subset_cache(train_cache, batch_rows, spec)
    batch = FEAT.encode_split(subset, spec, vocab)
    batch_ok = bool(np.array_equal(batch["x"], train_enc["x"][np.asarray(batch_rows)]))
    details = {
        "single_molecule_checks": rows,
        "single_molecule_all_identical": ok,
        "batch_of_64_identical_to_full_slice": batch_ok,
        "n_batch_rows": len(batch_rows),
    }
    return {"id": 5, "name": "batch_equals_single_molecule", "pass": bool(ok and batch_ok), "details": details}


# ---------------------------------------------------------------------------
# gate 9: FOLD64 buckets
# ---------------------------------------------------------------------------


def gate_fold(vocab, fold) -> dict[str, Any]:
    bucket_of_col = np.asarray(fold["bucket_of_col"])
    counts = np.bincount(bucket_of_col, minlength=C.N_BUCKETS)
    again = FEAT.fold_buckets(vocab)
    details = {
        "n_buckets": int(C.N_BUCKETS),
        "per_bucket": int(C.PER_BUCKET),
        "bucket_sizes_unique": sorted(set(int(x) for x in counts.tolist())),
        "bucket_ids_present": int(np.unique(bucket_of_col).size),
        "all_keys_assigned_once": bool(bucket_of_col.size == C.K),
        "deterministic_recompute_matches": bool(
            np.array_equal(bucket_of_col, np.asarray(again["bucket_of_col"]))
        ),
        "assignment_sha256": fold["assignment_sha256"],
        "salt": fold["salt"],
        "salt_rule": fold["salt_rule"],
        "n_keys": int(len(vocab["keys"])),
    }
    passed = bool(
        len(set(counts.tolist())) == 1
        and int(counts[0]) == C.PER_BUCKET
        and int(counts.size) == C.N_BUCKETS
        and details["bucket_ids_present"] == C.N_BUCKETS
        and details["all_keys_assigned_once"]
        and details["deterministic_recompute_matches"]
    )
    return {"id": 9, "name": "fold64_buckets", "pass": passed, "details": details}


# ---------------------------------------------------------------------------
# gates 7-8: NULL matrix
# ---------------------------------------------------------------------------


def gate_null(feats: Mapping[str, Any]) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    ok = True
    for split in ("train", "valid"):
        mat = feats[split]["binary"]
        null_mat, stats = FEAT.double_edge_swap(mat, C.NULL_SEED[split])
        cmp = FEAT.compare_matrices(mat, null_mat)
        row_sums = np.asarray(mat, dtype=np.int64).sum(axis=1)
        col_sums = np.asarray(mat, dtype=np.int64).sum(axis=0)
        pass_split = bool(
            cmp["row_sums_equal"]
            and cmp["col_sums_equal"]
            and cmp["n_nonzero_equal"]
            and not cmp["identical"]
            and stats["n_accepted"] > 0
            and int(null_mat.shape[1]) == C.K
        )
        ok = ok and pass_split
        row_hist = {}
        for v, cnt in zip(*np.unique(row_sums, return_counts=True)):
            row_hist[str(int(v))] = int(cnt)
        per_split[split] = {
            "pass": pass_split,
            "seed": int(C.NULL_SEED[split]),
            "n_attempts": int(stats["n_attempts"]),
            "n_accepted": int(stats["n_accepted"]),
            "acceptance_rate": float(stats["acceptance_rate"]),
            "n_nonzero": int(cmp["n_nonzero_true"]),
            "n_nonzero_null": int(cmp["n_nonzero_other"]),
            "row_sums_equal": bool(cmp["row_sums_equal"]),
            "col_sums_equal": bool(cmp["col_sums_equal"]),
            "identical": bool(cmp["identical"]),
            "n_shared_positions": int(cmp["n_shared_positions"]),
            "jaccard_nonzero": float(cmp["jaccard_nonzero"]),
            "changed_fraction_of_nonzero": float(cmp["changed_fraction_of_nonzero"]),
            "hamming_positions": int(cmp["hamming_positions"]),
            "hamming_fraction_of_matrix": float(cmp["hamming_fraction_of_matrix"]),
            "row_sum_histogram": row_hist,
            "col_sum_min": int(col_sums.min()),
            "col_sum_max": int(col_sums.max()),
            "col_sum_mean": float(col_sums.mean()),
            "source_matrix_sha256": stats["source_matrix_sha256"],
            "null_matrix_sha256": stats["null_matrix_sha256"],
            "labels_used": False,
            "recompute_deterministic": bool(
                np.array_equal(null_mat, FEAT.double_edge_swap(mat, C.NULL_SEED[split])[0])
            ),
        }
        ok = ok and per_split[split]["recompute_deterministic"]
    details = {
        "per_split": per_split,
        "swap_rule": "bipartite double-edge swap; 8 * nnz attempts; accept only if rows and columns differ "
        "and both cross cells are zero",
        "test_split": "not built here; the Test randomisation is created inside the Test scoring pass only",
    }
    return {"id": 7, "name": "null_marginals_conserved", "pass": ok, "details": details}, {
        "id": 8,
        "name": "null_differs_from_true",
        "pass": all(v["pass"] and not v["identical"] for v in details["per_split"].values()),
        "details": {
            "per_split": {
                k: {
                    "identical": v["identical"],
                    "changed_fraction_of_nonzero": v["changed_fraction_of_nonzero"],
                    "jaccard_nonzero": v["jaccard_nonzero"],
                    "hamming_positions": v["hamming_positions"],
                    "n_accepted": v["n_accepted"],
                    "n_attempts": v["n_attempts"],
                }
                for k, v in details["per_split"].items()
            }
        },
    }


# ---------------------------------------------------------------------------
# gates 6, 10, 11, 12: parameters, zero-init equivalence, gradients, trunk identity
# ---------------------------------------------------------------------------


def gate_parameters(spec, norm) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_seed: dict[str, Any] = {}
    a0_expected = int(PM.parameter_count(PM.PTBLCRC(spec, norm, C.MODEL_A0, n_tasks=C.N_TASKS)))
    registry: dict[int, dict[str, Any]] = {}
    for seed in C.SEEDS:
        built = {}
        for arm in C.NEURAL:
            model, info = M.build_model(arm, spec, norm, int(seed), "cpu")
            registry[int(seed)] = registry.get(int(seed), {})
            registry[int(seed)][arm] = (model, info)
            built[arm] = int(sum(p.numel() for p in model.parameters()))
        per_seed[str(int(seed))] = {
            "parameters": built,
            "true_equals_null": bool(built[C.MODEL_TRUE] == built[C.MODEL_NULL]),
            "a0_plus_512": bool(built[C.MODEL_TRUE] == built[C.MODEL_A0] + C.K),
            "fold64_a0_plus_64": bool(built[C.MODEL_FOLD] == built[C.MODEL_A0] + C.N_BUCKETS),
            "a0_matches_frozen": bool(built[C.MODEL_A0] == a0_expected),
        }
    passed = all(
        v["true_equals_null"] and v["a0_plus_512"] and v["fold64_a0_plus_64"] and v["a0_matches_frozen"]
        for v in per_seed.values()
    )
    gate6 = {
        "id": 6,
        "name": "wide_parameter_counts",
        "pass": bool(passed),
        "details": {
            "per_seed": per_seed,
            "a0_parameter_count": a0_expected,
            "expected_a0_parameter_count": int(C.PC.EXPECTED_PARAMS_SINGLE[C.MODEL_A0]),
            "branch_parameters": {"WIDE_TRUE": C.K, "WIDE_NULL": C.K, "FOLD64": C.N_BUCKETS},
        },
    }
    return gate6, registry, {"a0_parameter_count": a0_expected}


def gate_zero_init(
    spec, norm, registry, valid_cache, x_valid, fold_valid, null_valid
) -> dict[str, Any]:
    device = torch.device("cpu")
    rows = list(range(32))
    per_seed: dict[str, Any] = {}
    ok = True
    for seed in C.SEEDS:
        a0_model = registry[int(seed)][C.MODEL_A0][0]
        a0_model.eval()
        batch_ref = TRAIN.batch_with_features(valid_cache, rows, None, device)
        with torch.no_grad():
            ref = a0_model(batch_ref).float()
        entry: dict[str, Any] = {}
        for arm, feats in (
            (C.MODEL_TRUE, x_valid),
            (C.MODEL_NULL, null_valid),
            (C.MODEL_FOLD, fold_valid),
        ):
            model = registry[int(seed)][arm][0]
            model.eval()
            batch = TRAIN.batch_with_features(valid_cache, rows, feats, device)
            with torch.no_grad():
                out = model(batch).float()
            diff = float((out - ref).abs().max())
            entry[arm] = {
                "max_abs_diff_vs_a0": diff,
                "tolerance": 1e-6,
                "pass": bool(diff <= 1e-6),
                "branch_weight_abs_sum": float(
                    model.branch.weight.detach().abs().sum() if hasattr(model, "branch") else 0.0
                ),
            }
            ok = ok and entry[arm]["pass"]
        per_seed[str(int(seed))] = entry
    details = {
        "device": "cpu",
        "dtype": "float32",
        "n_molecules": len(rows),
        "per_seed": per_seed,
        "tolerance": 1e-6,
        "note": "branches are zero-initialised, so the wide forward must equal the same-seed A0 forward",
    }
    return {"id": 10, "name": "zero_init_matches_a0", "pass": bool(ok), "details": details}


def _binary_from_x(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) > 0).astype(np.uint8)


def _normalise_binary(binary: np.ndarray) -> np.ndarray:
    active = np.asarray(binary, dtype=np.int64).sum(axis=1)
    scale = (1.0 / np.sqrt(np.maximum(1, active).astype(np.float64))).astype(np.float32)
    return (np.asarray(binary, dtype=np.float32) * scale[:, None]).astype(np.float32)


def null_branch_input(binary: np.ndarray, split: str) -> tuple[np.ndarray, dict[str, Any]]:
    """NULL branch input for one split (identical normalisation, swapped matrix)."""
    null_mat, stats = FEAT.double_edge_swap(binary, C.NULL_SEED[split])
    return _normalise_binary(null_mat), stats


def gate_branch_gradient(spec, norm, valid_cache, x_valid, y_valid) -> dict[str, Any]:
    device = torch.device("cpu")
    model, _info = M.build_model(C.MODEL_TRUE, spec, norm, 0, device)
    recipe = dict(C.PC.RECIPE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    rows = list(range(64))
    batch = TRAIN.batch_with_features(valid_cache, rows, x_valid, device)
    y = torch.as_tensor(np.asarray(y_valid, dtype=np.float64).reshape(-1)[rows], dtype=torch.float32)
    model.train()
    pred = model(batch)
    loss = F.binary_cross_entropy_with_logits(pred.reshape(-1), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = model.branch.weight.grad
    in_optimizer = any(p is model.branch.weight for group in optimizer.param_groups for p in group["params"])
    before = model.branch.weight.detach().clone()
    grad_ok = grad is not None and float(grad.abs().sum()) > 0.0
    optimizer.step()
    after = model.branch.weight.detach().clone()
    changed = bool(not torch.equal(before, after))
    details = {
        "branch_grad_abs_sum": float(grad.abs().sum()) if grad is not None else None,
        "branch_grad_max_abs": float(grad.abs().max()) if grad is not None else None,
        "branch_grad_nonzero_entries": int((grad != 0).sum()) if grad is not None else 0,
        "branch_in_optimizer": bool(in_optimizer),
        "weight_changed_after_one_step": changed,
        "weight_abs_change_l2": float((after - before).norm()),
        "optimizer": "torch.optim.AdamW",
        "n_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    passed = bool(grad_ok and in_optimizer and changed)
    return {"id": 11, "name": "branch_is_trained", "pass": passed, "details": details}


def _valid_labels(cache, rows):  # pragma: no cover - kept for symmetry with PTB helpers
    view = PD.molecule_view(C.DATASET)
    return np.asarray(view.targets.y_valid, dtype=np.float64).reshape(-1)[np.asarray(rows, dtype=np.int64)]


def gate_trunk_identity(spec, norm, view, registry) -> dict[str, Any]:
    import json

    details: dict[str, Any] = {}
    recipe = dict(C.PC.RECIPE)
    details["recipe_equals_frozen_dissection_recipe"] = bool(
        {k: v for k, v in recipe.items() if k != "epochs_by_dataset"} == dict(DC.RECIPE)
    )
    details["recipe_epochs_by_dataset"] = recipe.get("epochs_by_dataset")
    details["recipe"] = recipe
    details["epochs"] = int(C.EPOCHS)
    details["dataset_epoch_cap"] = int(C.PC.MOLECULE_DATASETS[C.DATASET]["epochs"])
    details["seeds"] = list(C.SEEDS)
    details["selection_rule"] = PT.Tracker("binary").rule()
    details["selection_rule_expected"] = (
        "highest_validation_roc_auc__tie_lower_validation_bce__then_earlier_epoch"
    )
    details["model_code_path"] = "ptb_model.build_model -> PTBLCRC(variant=A0)"
    details["feature_spec_hash"] = spec.hash()
    details["frozen_feature_spec_hash"] = C.FROZEN_SPEC_HASH
    init_matches = {}
    for seed in C.SEEDS:
        full = PM.full_state(spec, norm, int(seed), C.N_TASKS)
        full_hash = PM.state_fingerprint(full)
        info = registry[int(seed)][C.MODEL_A0][1]
        init_matches[str(int(seed))] = {
            "full_state_hash": full_hash,
            "build_model_init_hash": info["init_state_hash"],
            "match": bool(full_hash == info["init_state_hash"]),
        }
    details["a0_init_hash_reproduced"] = init_matches
    shared = {}
    for seed in C.SEEDS:
        a0_state = registry[int(seed)][C.MODEL_A0][0].state_dict()
        for arm in C.WIDE:
            wide_state = registry[int(seed)][arm][0].state_dict()
            base_prefix = "base."
            shared_ok = all(
                torch.equal(v.cpu(), wide_state[f"{base_prefix}{k}"].cpu()) for k, v in a0_state.items()
            )
            shared[f"s{seed}_{arm}"] = bool(shared_ok)
    details["wide_trunk_bitwise_equal_to_a0"] = shared
    manifest_path = (
        C.REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/analysis/split_manifest_ogbg-molhiv.json"
    )
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    details["cache_rows_sha256"] = {k: C.index_sha256(view.rows[k]) for k in ("train", "valid", "test")}
    details["frozen_cache_rows_sha256"] = frozen["rows_sha256"]
    details["cache_rows_match_frozen_manifest"] = bool(
        details["cache_rows_sha256"] == frozen["rows_sha256"]
    )
    details["train_norm_fingerprint"] = PREP_DATA.norm_fingerprint(view.train_norm)
    details["train_norm_source"] = view.train_norm["source"]
    details["train_norm_valid_or_test_used"] = view.train_norm["valid_or_test_used"]
    passed = bool(
        details["recipe_equals_frozen_dissection_recipe"]
        and details["selection_rule"] == details["selection_rule_expected"]
        and all(v["match"] for v in init_matches.values())
        and all(shared.values())
        and details["cache_rows_match_frozen_manifest"]
        and details["train_norm_valid_or_test_used"] is False
        and spec.hash() == C.FROZEN_SPEC_HASH
    )
    return {"id": 12, "name": "trunk_and_recipe_identical_to_history", "pass": passed, "details": details}


# ---------------------------------------------------------------------------
# gate 14: no Test access in the training path (static);
# ---------------------------------------------------------------------------


def gate_no_test_access() -> dict[str, Any]:
    import ast

    forbidden = ("load_test_cache", "molecule_test", "build_test_structure_cache")

    def symbols(path: Path) -> dict[str, set[str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        attrs: set[str] = set()
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                attrs.add(node.attr)
            elif isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
                imports.update(alias.name for alias in node.names)
        return {"names": names, "attrs": attrs, "imports": imports}

    hits: dict[str, list[str]] = {}
    for name in ("si_train.py", "si_model.py", "si_features.py", "si_sparse.py"):
        sym = symbols(C.CODE_DIR / name)
        used = sym["names"] | sym["attrs"]
        hits[name] = sorted(t for t in forbidden if t in used or any(t in m for m in sym["imports"]))
    ledger_path = C.ANALYSIS / "test_access_ledger.json"
    ledger_events = 0
    if ledger_path.exists():
        ledger_events = int(len(C.read_json(ledger_path).get("events", [])))
    manifest_path = C.ANALYSIS / "checkpoint_manifest.json"
    details = {
        "forbidden_symbols": list(forbidden),
        "static_hits_by_ast": hits,
        "scan_mode": "python AST (names / attributes / imports); docstrings and comments are not code",
        "test_access_ledger_events_before_freeze": ledger_events,
        "training_module_imports": [
            "si_common",
            "si_features",
            "si_model",
            "lcrc_data",
            "mb2_metrics",
            "ptb_data",
            "ptb_train_mol",
        ],
        "train_view_scope": "ptb_data.molecule_view -> Train + Validation only",
        "checkpoint_manifest_exists": manifest_path.exists(),
    }
    passed = bool(
        not any(hits.values()) and ledger_events == 0 and not manifest_path.exists()
    )
    return {"id": 14, "name": "no_test_access_during_training", "pass": passed, "details": details}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_gates() -> dict[str, Any]:
    torch.set_grad_enabled(True)
    view = PD.molecule_view(C.DATASET)
    spec = view.spec
    if spec.hash() != C.FROZEN_SPEC_HASH:
        raise RuntimeError(f"FeatureSpec hash {spec.hash()} != frozen {C.FROZEN_SPEC_HASH}")
    norm = view.train_norm

    C.log("gate set: building vocabulary and split vectors")
    vocab = FEAT.build_vocabulary(view.train_cache, spec, C.K)
    if vocab.get("status") != "OK":
        C.write_json(C.ANALYSIS / "feature_vocabulary.json", vocab)
        raise SystemExit("FEATURE_INFEASIBLE: fewer than K legal Train S2 keys; the round stops here")
    C.write_json(C.ANALYSIS / "feature_vocabulary.json", vocab)
    fold = FEAT.fold_buckets(vocab)
    feats = {}
    for split, cache in (("train", view.train_cache), ("valid", view.valid_cache)):
        feats[split] = FEAT.encode_split(cache, spec, vocab)
    x_valid = FEAT.branch_input(C.MODEL_TRUE, "valid", feats, fold)
    fold_valid = FEAT.branch_input(C.MODEL_FOLD, "valid", feats, fold)
    null_valid, _null_stats = null_branch_input(feats["valid"]["binary"], "valid")
    y_valid = np.asarray(view.targets.y_valid, dtype=np.float64).reshape(-1)

    C.log("gates 1-5, 9 (features) and the 18-ring witness")
    g1 = gate_split_alignment(view)
    g2 = gate_vocabulary(vocab, view.train_cache, view.valid_cache, spec, feats["train"])
    g3 = gate_key_construction(view.train_cache, spec)
    g4 = gate_invariance(view.train_cache, spec, vocab)
    g5 = gate_batching(view.train_cache, spec, vocab)
    g9 = gate_fold(vocab, fold)
    C.log("gates 7-8 (NULL)")
    g7, g8 = gate_null(feats)
    C.log("gates 6, 10, 11, 12 (parameters / initialisation / gradient / trunk)")
    g6, registry, _meta = gate_parameters(spec, norm)
    g10 = gate_zero_init(spec, norm, registry, view.valid_cache, x_valid, fold_valid, null_valid)
    g11 = gate_branch_gradient(spec, norm, view.valid_cache, x_valid, y_valid)
    g12 = gate_trunk_identity(spec, norm, view, registry)
    g14 = gate_no_test_access()

    ring = FEAT.ring18_collision(vocab)

    feature_checks = {
        "schema": "lcrc-sparse-identity-feature-checks-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "vocabulary": {
            "k": int(vocab["k"]),
            "n_unique_keys": int(vocab["n_unique_keys"]),
            "vocabulary_sha256": vocab["vocabulary_sha256"],
            "train_n_molecules": int(vocab["train_n_molecules"]),
            "n_keys_with_df_1": int(vocab["n_keys_with_df_1"]),
            "top10": [
                {"key": k, "document_frequency": int(f)}
                for k, f in zip(vocab["keys"][:10], vocab["document_frequency"][:10])
            ],
            "df_range": [int(min(vocab["document_frequency"])), int(max(vocab["document_frequency"]))],
            "selected_keys": [[int(x) for x in k] for k in vocab["keys"]],
            "selected_document_frequency": [int(x) for x in vocab["document_frequency"]],
        },
        "splits": {
            "train": feats["train"]["stats"],
            "valid": feats["valid"]["stats"],
        },
        "fold64": {k: v for k, v in fold.items() if k in ("salt", "n_buckets", "per_bucket", "bucket_sizes",
                                                          "assignment_sha256", "salt_rule")},
        "fold64_hash_order": fold["hash_order"],
        "fold64_digest_prefix": fold["digest_prefix"],
        "ring18_witness": ring,
        "gates": [g1, g2, g3, g4, g5, g9],
    }
    null_checks = {
        "schema": "lcrc-sparse-identity-null-checks-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "gates": [g7, g8],
        "vocabulary_sha256": vocab["vocabulary_sha256"],
    }
    init_checks = {
        "schema": "lcrc-sparse-identity-initialization-checks-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "gates": [g6, g10, g11, g12],
    }
    C.write_json(C.ANALYSIS / "feature_checks.json", feature_checks)
    C.write_json(C.ANALYSIS / "null_checks.json", null_checks)
    C.write_json(C.ANALYSIS / "initialization_checks.json", init_checks)

    gates = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]
    aggregate = {
        "schema": "lcrc-sparse-identity-correctness-gates-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "status": "PASS" if all(g["pass"] for g in gates) else "FAIL",
        "gates": [{"id": g["id"], "name": g["name"], "pass": bool(g["pass"])} for g in gates]
        + [
            {"id": 13, "name": "checkpoint_reload_reproduces_validation", "pass": None,
             "status": "enforced per training run; completed at freeze"},
            {"id": 14, "name": "no_test_access_during_training", "pass": bool(g14["pass"]),
             "status": "static precondition verified here; completed at freeze"},
        ],
        "gate_14_static": g14,
        "ring18_witness": {"collision": ring["collision"], "jaccard": ring["jaccard"], "is_gate": False},
        "evidence_files": {
            "feature_checks": "analysis/feature_checks.json",
            "null_checks": "analysis/null_checks.json",
            "initialization_checks": "analysis/initialization_checks.json",
        },
    }
    C.write_json(C.ANALYSIS / "correctness_gates.json", aggregate)
    for g in gates:
        C.log(f"gate {g['id']:>2} {g['name']:<42} {'PASS' if g['pass'] else 'FAIL'}")
    C.log(f"18-ring witness collision={ring['collision']} (not a gate)")
    if aggregate["status"] != "PASS":
        raise SystemExit("correctness gates FAILED; formal training must not start")
    return aggregate


def finalize_gates(extra: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    path = C.ANALYSIS / "correctness_gates.json"
    payload = C.read_json(path)
    by_id = {int(g["id"]): g for g in payload["gates"]}
    for gid, record_ in extra.items():
        by_id[int(gid)].update({"pass": bool(record_["pass"]), "status": "completed at freeze"})
        by_id[int(gid)]["details"] = record_["details"]
    payload["gates"] = [by_id[k] for k in sorted(by_id)]
    payload["status"] = "PASS" if all(bool(g["pass"]) for g in payload["gates"]) else "FAIL"
    payload["finalized_utc"] = C.utcnow()
    C.write_json(path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the sparse-identity correctness gates (CPU)")
    ap.add_argument("--quiet", action="store_true")
    ap.parse_args(argv)
    aggregate = run_gates()
    print(C.stable_json({"status": aggregate["status"]}))
    return 0 if aggregate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
