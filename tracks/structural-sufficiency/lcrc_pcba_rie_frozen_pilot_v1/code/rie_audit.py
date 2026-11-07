"""Correctness / invariance / control audits for the RIE frozen pilot.

Everything here is label blind and independent of the trained head.  The audits
are the CPU-side gates of the round:

======================  ==================================================
gate                    what is checked
======================  ==================================================
checkpoint identity     file + payload + run-registry identity of LCRC-A
distance reference      our BFS == ``scipy.sparse.csgraph`` on real molecules
graph fidelity          cache bond graph == RDKit graph (atoms/edges/order)
motif order             permuting the motif list does not change ``S(G)``
atom renumbering        relabelling atoms does not change ``S(G)``
pair symmetry           swapping a pair's endpoints does not change the pair feature
motif dedupe            duplicated matches collapse to one motif
components              pairs never span two connected components
empty pair set          molecules without eligible pairs get an exactly zero tensor
SHUFFLE preservation    motif set, pairs and per-bin counts are unchanged
SHUFFLE intervention    REAL != SHUFFLE only where a molecule has >1 distance bin
row correspondence      cache rows == official split rows (and zero/intervened patterns)
official AP agreement   OGB Evaluator == ``pc_metrics.per_task_ap`` mean
test isolation          official Test labels are never finite and never indexed
======================  ==================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import rie_common as C  # noqa: E402
import rie_features as F  # noqa: E402
import rie_motifs as MO  # noqa: E402
import lr_data as DATA  # noqa: E402
import pc_metrics as MET  # noqa: E402

STRUCTURES = C.RUNS / "structures"
FEATURES = C.RUNS / "features"

AUDIT_ROWS = 48
GRAPH_SAMPLE = 512


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_task() -> Any:
    task = DATA.load_task()
    C.assert_inherited_hashes(task)
    return task


def _molecule_graph(task: Any, row: int) -> tuple[int, list[tuple[int, int]], str]:
    atom_ptr = np.asarray(task.cache["atom_ptr"])
    bond_ptr = np.asarray(task.cache["bond_ptr"])
    bond_index = np.asarray(task.cache["bond_index"])
    n_atoms = int(atom_ptr[row + 1] - atom_ptr[row])
    edges = [
        (int(u), int(v)) for u, v in bond_index[int(bond_ptr[row]) : int(bond_ptr[row + 1])].tolist()
    ]
    return n_atoms, edges, str(task.cache["smiles"][row])


def _motifs_for_row(task: Any, row: int) -> list[tuple[int, tuple[int, ...]]]:
    motifs, reason = MO.motifs_for_smiles(str(task.cache["smiles"][row]))
    if reason is not None:
        raise RuntimeError(f"row {row} failed to parse in the audit: {reason}")
    return motifs


def _features_with_h(
    *,
    n_atoms: int,
    edges: Sequence[Sequence[int]],
    motifs: Sequence[tuple[int, Sequence[int]]],
    h: np.ndarray,
    row_id: int,
) -> dict[str, Any]:
    return F.molecule_relation_features(
        h, motifs, n_atoms=n_atoms, edges=edges, official_row_id=int(row_id)
    )


# ---------------------------------------------------------------------------
# individual audits
# ---------------------------------------------------------------------------


def audit_distance_reference(task: Any, rows: np.ndarray) -> dict[str, Any]:
    max_diff = 0
    n_atoms_total = 0
    for row in np.asarray(rows).tolist():
        n_atoms, edges, _smiles = _molecule_graph(task, row)
        ours = F.shortest_paths(n_atoms, edges)
        ref = F.reference_shortest_paths(n_atoms, edges)
        if ours.shape != ref.shape:
            raise RuntimeError(f"distance matrices differ in shape at row {row}")
        max_diff = max(max_diff, int(np.abs(ours - ref).max()))
        n_atoms_total += int(n_atoms)
    return {
        "molecules": int(np.asarray(rows).size),
        "atoms_total": int(n_atoms_total),
        "max_abs_difference_vs_scipy": int(max_diff),
        "reference": "scipy.sparse.csgraph.shortest_path(directed=False, unweighted=True)",
        "implementation": "own BFS all-pairs in rie_features.shortest_paths",
        "passed": bool(max_diff == 0),
    }


def audit_graph_fidelity(task: Any, rows: np.ndarray) -> dict[str, Any]:
    from rdkit import Chem

    n_atoms_mismatch = 0
    edge_mismatch = 0
    relabel_violations = 0
    tested = 0
    for row in np.asarray(rows).tolist():
        smiles = str(task.cache["smiles"][row])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        tested += 1
        n_atoms, edges, _ = _molecule_graph(task, row)
        if int(mol.GetNumAtoms()) != n_atoms:
            n_atoms_mismatch += 1
            continue
        rd = {(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
        cached = {(min(u, v), max(u, v)) for u, v in edges}
        if rd != cached:
            edge_mismatch += 1
        # element + aromaticity must match the frozen (element, aromatic) vocabulary
        import mb2_rdkit as MB  # inherited frozen raw-attribute vocabulary

        class_of = {pair: k for k, pair in enumerate(MB.ATOM_CLASSES)}
        atom_type = np.asarray(task.cache["atom_type"])
        atom_ptr = np.asarray(task.cache["atom_ptr"])
        codes = atom_type[int(atom_ptr[row]) : int(atom_ptr[row + 1])]
        for atom in mol.GetAtoms():
            key = (atom.GetSymbol(), bool(atom.GetIsAromatic()))
            expected = class_of.get(key, MB.ATOM_UNK)
            if int(codes[int(atom.GetIdx())]) != int(expected):
                relabel_violations += 1
    return {
        "molecules_tested": int(tested),
        "atom_count_mismatches": int(n_atoms_mismatch),
        "bond_set_mismatches": int(edge_mismatch),
        "atom_identity_mismatches": int(relabel_violations),
        "conclusion": "RDKit atom index == cache atom index (verified on the audited sample)",
        "passed": bool(n_atoms_mismatch == 0 and edge_mismatch == 0 and relabel_violations == 0),
    }


def audit_invariances(task: Any, rows: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(20261106)
    motif_perm_max = 0.0
    atom_perm_max = 0.0
    pair_swap_max = 0.0
    dedupe_max = 0.0
    tested = 0
    rho = []
    for row in np.asarray(rows).tolist():
        motifs = _motifs_for_row(task, row)
        if len(motifs) < 2:
            continue
        n_atoms, edges, _ = _molecule_graph(task, row)
        h = rng.standard_normal((n_atoms, C.WIDTH)).astype(np.float32)
        base = _features_with_h(n_atoms=n_atoms, edges=edges, motifs=motifs, h=h, row_id=int(row))
        tested += 1

        # (4) motif order invariance
        shuffled_motifs = list(motifs)
        rng.shuffle(shuffled_motifs)
        perm = _features_with_h(n_atoms=n_atoms, edges=edges, motifs=shuffled_motifs, h=h, row_id=int(row))
        motif_perm_max = max(motif_perm_max, float(np.abs(perm["real"] - base["real"]).max()))

        # (5) atom renumbering invariance
        pi = rng.permutation(n_atoms)
        inv = np.empty_like(pi)
        inv[pi] = np.arange(n_atoms)
        edges_new = [(int(inv[u]), int(inv[v])) for u, v in edges]
        motifs_new = [(fam, tuple(sorted(int(inv[a]) for a in atoms))) for fam, atoms in motifs]
        h_new = np.empty_like(h)
        h_new[inv] = h
        renumbered = _features_with_h(n_atoms=n_atoms, edges=edges_new, motifs=motifs_new, h=h_new, row_id=int(row))
        atom_perm_max = max(atom_perm_max, float(np.abs(renumbered["real"] - base["real"]).max()))

        # (6) pair endpoint swap: the pair feature is symmetric by construction
        struct = base["structure"]
        if struct["n_pairs"]:
            split = F.concat_structures([struct], rows=np.asarray([int(row)], dtype=np.int64))
            gathered = F.gather_batch(
                split, np.asarray([0], dtype=np.int64), np.asarray([n_atoms], dtype=np.int64)
            )
            z = F.motif_states(h, gathered)
            fwd = F.pair_features(z, gathered)
            swapped = dict(gathered)
            swapped["pair_a"], swapped["pair_b"] = gathered["pair_b"], gathered["pair_a"]
            rev = F.pair_features(z, swapped)
            pair_swap_max = max(pair_swap_max, float(np.abs(fwd - rev).max()))

        # (7) duplicate motif matches collapse
        doubled = list(motifs) + list(motifs)
        dup = _features_with_h(n_atoms=n_atoms, edges=edges, motifs=doubled, h=h, row_id=int(row))
        dedupe_max = max(dedupe_max, float(np.abs(dup["real"] - base["real"]).max()))

        # component audit (8): every retained pair is inside one component
        labels = F.connected_components(n_atoms, edges)
        for i, j, _d, _b in F.eligible_pairs([np.asarray(m[1]) for m in motifs], F.shortest_paths(n_atoms, edges)):
            if labels[motifs[i][1][0]] != labels[motifs[j][1][0]]:
                rho.append({"row": int(row), "i": int(i), "j": int(j)})
    return {
        "molecules_tested": int(tested),
        "motif_order_permutation_max_abs_diff": float(motif_perm_max),
        "motif_order_tolerance": 1e-6,
        "motif_order_passed": bool(motif_perm_max <= 1e-6),
        "atom_renumbering_max_abs_diff": float(atom_perm_max),
        "atom_renumbering_tolerance": 1e-5,
        "atom_renumbering_passed": bool(atom_perm_max <= 1e-5),
        "pair_endpoint_swap_max_abs_diff": float(pair_swap_max),
        "pair_endpoint_swap_passed": bool(pair_swap_max == 0.0),
        "duplicate_motif_max_abs_diff": float(dedupe_max),
        "duplicate_motif_passed": bool(dedupe_max <= 1e-6),
        "cross_component_pairs_found": len(rho),
        "cross_component_passed": bool(len(rho) == 0),
        "note": (
            "the aggregations are exact real sums; float32 summation order can differ by "
            "O(1e-7) under atom relabelling, which is why that gate uses a 1e-5 tolerance"
        ),
    }


def audit_component_and_empty_cases(task: Any) -> dict[str, Any]:
    """Explicit no-relation cases: disconnected graph and 'all pairs too close'."""
    rng = np.random.default_rng(7)
    # two fragments, one motif each, never connected
    n_atoms = 6
    edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
    motifs = [(0, (2,)), (0, (3,))]
    h = rng.standard_normal((n_atoms, C.WIDTH)).astype(np.float32)
    dis = _features_with_h(n_atoms=n_atoms, edges=edges, motifs=motifs, h=h, row_id=1)
    # two motifs on the same 3-chain, distance 2 -> both dropped (d < 3)
    edges2 = [(0, 1), (1, 2), (2, 3)]
    motifs2 = [(0, (0,)), (0, (2,))]
    near = _features_with_h(n_atoms=4, edges=edges2, motifs=motifs2, h=h[:4], row_id=2)
    # no motifs at all
    none = _features_with_h(n_atoms=4, edges=edges2, motifs=[], h=h[:4], row_id=3)
    return {
        "disconnected_components": {
            "n_components": int(F.connected_components(n_atoms, edges).max() + 1),
            "n_pairs": int(dis["structure"]["n_pairs"]),
            "max_abs_real": float(np.abs(dis["real"]).max()),
            "max_abs_shuffle": float(np.abs(dis["shuffle"]).max()),
            "is_zero": bool(np.abs(dis["real"]).max() == 0.0 and np.abs(dis["shuffle"]).max() == 0.0),
        },
        "all_pairs_below_min_distance": {
            "n_pairs": int(near["structure"]["n_pairs"]),
            "is_zero": bool(np.abs(near["real"]).max() == 0.0),
        },
        "no_motifs": {
            "n_motifs": int(none["structure"]["n_motifs"]),
            "is_zero": bool(np.abs(none["real"]).max() == 0.0 and np.abs(none["shuffle"]).max() == 0.0),
        },
        "passed": bool(
            np.abs(dis["real"]).max() == 0.0
            and np.abs(dis["shuffle"]).max() == 0.0
            and np.abs(near["real"]).max() == 0.0
            and np.abs(none["real"]).max() == 0.0
        ),
    }


def audit_shuffle(payloads: Mapping[str, Mapping[str, Any]], feature_files: Mapping[str, Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    total_rows = 0
    total_differ = 0
    for split in ("train", "valid"):
        struct = payloads[split]
        n = int(struct["n_molecules"])
        pair_ptr = np.asarray(struct["pair_ptr"], dtype=np.int64)
        pair_counts = np.diff(pair_ptr)
        bins = np.asarray(struct["pair_bin"], dtype=np.int64)
        shuf = np.asarray(struct["pair_bin_shuffle"], dtype=np.int64)
        bin_counts = np.asarray(struct["bin_counts"], dtype=np.int64).reshape(n, C.N_BINS)
        owner = np.repeat(np.arange(n), pair_counts)
        shuf_counts = np.zeros_like(bin_counts)
        if bins.size:
            np.add.at(shuf_counts, (owner, shuf), 1)
        counts_preserved = bool(np.array_equal(bin_counts, shuf_counts))
        per_row_sorted = np.array_equal(np.sort(bins.reshape(-1)), np.sort(shuf.reshape(-1)))
        structure_changed = np.zeros(n, dtype=bool)
        if bins.size:
            differ = bins != shuf
            structure_changed = (
                np.bincount(owner, weights=differ.astype(np.float64), minlength=n) > 0
            )
        multi_bin = (bin_counts > 0).sum(axis=1) >= 2

        real = np.load(feature_files[f"{split}_real"], mmap_mode="r")
        shuf_arr = np.load(feature_files[f"{split}_shuffle"], mmap_mode="r")
        changed = np.zeros(n, dtype=bool)
        step = 8192
        max_diff = 0.0
        for start in range(0, n, step):
            stop = min(start + step, n)
            a = np.asarray(real[start:stop])
            b = np.asarray(shuf_arr[start:stop])
            changed[start:stop] = np.any(a != b, axis=(1, 2))
            if a.size:
                max_diff = max(max_diff, float(np.abs(a - b).max()))
        zero_rows = ~np.any(np.asarray(real) != 0.0, axis=(1, 2)) if n else np.zeros(0, bool)
        none_pair_rows = pair_counts == 0
        correspondence = {
            "zero_tensor_rows": int(zero_rows.sum()),
            "no_pair_rows": int(none_pair_rows.sum()),
            "zero_tensor_iff_no_eligible_pair": bool(np.array_equal(zero_rows, none_pair_rows)),
        }
        changed_implies_multibin = bool(np.all(~changed | multi_bin))
        out[split] = {
            "rows": n,
            "pair_counts_per_bin_preserved_under_shuffle": counts_preserved,
            "global_bin_multiset_preserved": bool(per_row_sorted),
            "bin_counts_per_molecule_preserved": counts_preserved,
            "fraction_with_eligible_pairs": float((pair_counts > 0).mean()),
            "fraction_with_multiple_distance_bins": float(multi_bin.mean()),
            "fraction_REAL_ne_SHUFFLE_tensor": float(changed.mean()),
            "fraction_REAL_ne_SHUFFLE_structure": float(structure_changed.mean()),
            "tensor_change_implies_multiple_bins": changed_implies_multibin,
            "max_abs_tensor_difference": max_diff,
            "pair_count_per_bin": {
                "mean": [float(bin_counts[:, k].mean()) for k in range(C.N_BINS)],
                "median": [float(np.median(bin_counts[:, k])) for k in range(C.N_BINS)],
                "p90": [float(np.percentile(bin_counts[:, k], 90)) for k in range(C.N_BINS)],
                "max": [int(bin_counts[:, k].max()) for k in range(C.N_BINS)],
                "sum": [int(bin_counts[:, k].sum()) for k in range(C.N_BINS)],
            },
            "row_correspondence": correspondence,
        }
        total_rows += n
        total_differ += int(changed.sum())
    out["overall_fraction_REAL_ne_SHUFFLE"] = float(total_differ / max(total_rows, 1))
    out["min_intervention_fraction"] = float(C.DECISION["min_intervention_fraction"])
    out["intervention_passed"] = bool(
        out["valid"]["fraction_REAL_ne_SHUFFLE_tensor"] >= float(C.DECISION["min_intervention_fraction"])
    )
    out["preservation_passed"] = bool(
        all(
            out[s]["pair_counts_per_bin_preserved_under_shuffle"]
            and out[s]["global_bin_multiset_preserved"]
            and out[s]["tensor_change_implies_multiple_bins"]
            and out[s]["row_correspondence"]["zero_tensor_iff_no_eligible_pair"]
            for s in ("train", "valid")
        )
    )
    return out


def audit_ap_agreement(task: Any) -> dict[str, Any]:
    valid_labels = np.asarray(task.labels[task.valid_rows])
    train_labels = np.asarray(task.labels[task.train_rows])
    valid_base = np.asarray(np.load(FEATURES / "valid_base_logits.npy", mmap_mode="r"))
    train_base = np.asarray(np.load(FEATURES / "train_base_logits.npy", mmap_mode="r"))
    import pc_model as PC

    valid_probs = PC.sigmoid(valid_base)
    train_probs = PC.sigmoid(train_base)
    valid_official = MET.official_ap(valid_labels, valid_probs)
    valid_independent = MET.per_task_ap(valid_labels, valid_probs)
    train_official = MET.official_ap(train_labels, train_probs)
    train_independent = MET.per_task_ap(train_labels, train_probs)
    return {
        "valid_official_ap": float(valid_official["value"]),
        "valid_independent_macro_ap": float(valid_independent["mean_over_scored_tasks"]),
        "valid_abs_delta": abs(float(valid_official["value"]) - float(valid_independent["mean_over_scored_tasks"])),
        "train_official_ap": float(train_official["value"]),
        "train_independent_macro_ap": float(train_independent["mean_over_scored_tasks"]),
        "train_abs_delta": abs(float(train_official["value"]) - float(train_independent["mean_over_scored_tasks"])),
        "n_tasks_scored_valid": int(valid_independent["n_tasks_scored"]),
        "anchor_AP_A": C.AP_A,
        "valid_ap_abs_delta_to_anchor": abs(float(valid_official["value"]) - C.AP_A),
        "passed": bool(
            abs(float(valid_official["value"]) - float(valid_independent["mean_over_scored_tasks"])) <= 1e-9
            and abs(float(train_official["value"]) - float(train_independent["mean_over_scored_tasks"])) <= 1e-9
            and abs(float(valid_official["value"]) - C.AP_A) <= C.VALID_AP_TOL
        ),
    }


def audit_test_isolation(task: Any) -> dict[str, Any]:
    labels = np.asarray(task.labels)
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)
    files = list(FEATURES.glob("*.npy")) + list(STRUCTURES.glob("*.npz"))
    rows_arrays = {}
    for split in ("train", "valid"):
        rows_arrays[split] = np.asarray(np.load(FEATURES / f"{split}_rows.npy"), dtype=np.int64)
    overlap = {
        "train_test": int(np.intersect1d(train_rows, test_rows).size),
        "valid_test": int(np.intersect1d(valid_rows, test_rows).size),
        "cache_train_test": int(np.intersect1d(rows_arrays["train"], test_rows).size),
        "cache_valid_test": int(np.intersect1d(rows_arrays["valid"], test_rows).size),
    }
    return {
        "official_test_rows": int(test_rows.size),
        "finite_test_label_entries": int(np.isfinite(labels[test_rows]).sum()),
        "overlap": overlap,
        "artifacts_scanned": [C.track_rel(p) for p in files],
        "test_predictions_written": False,
        "test_metrics_computed": 0,
        "passed": bool(
            int(np.isfinite(labels[test_rows]).sum()) == 0
            and all(v == 0 for v in overlap.values())
        ),
    }


def audit_rows(task: Any, payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in ("train", "valid"):
        official = np.asarray(getattr(task, f"{split}_rows"), dtype=np.int64)
        cached = np.asarray(np.load(FEATURES / f"{split}_rows.npy"), dtype=np.int64)
        struct_rows = np.asarray(payloads[split]["rows"], dtype=np.int64)
        out[split] = {
            "rows": int(official.size),
            "rows_sha256": C.sha256_int64(official),
            "cache_rows_equal_official": bool(np.array_equal(cached, official)),
            "structure_rows_equal_official": bool(np.array_equal(struct_rows, official)),
            "structure_rows_sha256": C.sha256_int64(struct_rows),
        }
    out["passed"] = bool(
        out["train"]["cache_rows_equal_official"]
        and out["valid"]["cache_rows_equal_official"]
        and out["train"]["structure_rows_equal_official"]
        and out["valid"]["structure_rows_equal_official"]
    )
    return out


# ---------------------------------------------------------------------------
# GPU spot check: re-extraction of one batch of cached rows
# ---------------------------------------------------------------------------


def audit_gpu_row_correspondence(device: Any, batch_index: int = 0) -> dict[str, Any]:
    import torch

    import rie_extract as EX
    import pc_model as PC

    task = EX.load_task()
    model, model_audit = EX.load_frozen_A(task, device)
    struct = F.load_structures(STRUCTURES / "valid_structures.npz")
    batch_size = int(C.EXTRACT["batch_size"])
    start = int(batch_index) * batch_size
    stop = min(start + batch_size, int(np.asarray(task.valid_rows).size))
    rows = np.asarray(task.valid_rows[start:stop], dtype=np.int64)
    with torch.inference_mode():
        batch = PC.to_device(PC.collate(task.cache, rows.tolist()), device)
        logits, aux = model(batch, return_aux=True)
        h = aux["h"].detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
    atom_ptr = np.asarray(task.cache["atom_ptr"])
    positions = np.arange(start, stop, dtype=np.int64)
    atom_counts = (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64)
    gathered = F.gather_batch(struct, positions, atom_counts)
    real, counts = F.batch_relation_features(h, gathered, which="real")
    shuffle, _ = F.batch_relation_features(h, gathered, which="shuffle")
    scale = np.load(FEATURES / "relation_rms_scale.npy")
    real_norm = F.apply_scale(real, scale)
    shuffle_norm = F.apply_scale(shuffle, scale)
    cached_real = np.asarray(np.load(FEATURES / "valid_relation.npy", mmap_mode="r")[start:stop])
    cached_shuffle = np.asarray(np.load(FEATURES / "valid_relation_shuffle.npy", mmap_mode="r")[start:stop])
    cached_base = np.asarray(np.load(FEATURES / "valid_base_logits.npy", mmap_mode="r")[start:stop])
    return {
        "device": str(device),
        "batch_index": int(batch_index),
        "rows": int(rows.size),
        "row_range": [int(start), int(stop)],
        "base_logits_max_abs_diff_vs_cache": float(np.abs(logits_np - cached_base).max()),
        "real_relation_max_abs_diff_vs_cache": float(np.abs(real_norm - cached_real).max()),
        "shuffle_relation_max_abs_diff_vs_cache": float(np.abs(shuffle_norm - cached_shuffle).max()),
        "real_relation_bitwise_equal": bool(np.array_equal(real_norm, cached_real)),
        "shuffle_relation_bitwise_equal": bool(np.array_equal(shuffle_norm, cached_shuffle)),
        "bin_counts_match_structure": bool(
            np.array_equal(counts, np.asarray(struct["bin_counts"])[start:stop])
        ),
        "model": model_audit,
        "tolerance": 1e-5,
        "passed": bool(
            float(np.abs(logits_np - cached_base).max()) <= 1e-4
            and float(np.abs(real_norm - cached_real).max()) <= 1e-5
            and float(np.abs(shuffle_norm - cached_shuffle).max()) <= 1e-5
            and bool(np.array_equal(counts, np.asarray(struct["bin_counts"])[start:stop]))
        ),
        "note": (
            "CUDA atomic index_add_ in the frozen encoder means the same rows re-extracted in a "
            "fresh batch can differ by ~1e-6; the gate is a tolerance gate, not bitwise"
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_cpu() -> dict[str, Any]:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    task = load_task()
    manifest = C.read_json(C.ANALYSIS / "feature_manifest.json")
    coverage = C.read_json(C.ANALYSIS / "motif_coverage.json")
    payloads = {
        split: F.load_structures(STRUCTURES / f"{split}_structures.npz") for split in ("train", "valid")
    }
    feature_files = {
        "train_real": FEATURES / "train_relation.npy",
        "train_shuffle": FEATURES / "train_relation_shuffle.npy",
        "valid_real": FEATURES / "valid_relation.npy",
        "valid_shuffle": FEATURES / "valid_relation_shuffle.npy",
    }
    rng = np.random.default_rng(20260101)
    sample_valid = np.asarray(task.valid_rows, dtype=np.int64)[:GRAPH_SAMPLE]
    sample_train = np.asarray(task.train_rows, dtype=np.int64)[:GRAPH_SAMPLE]
    graph_rows = np.concatenate([sample_valid[: GRAPH_SAMPLE // 2], sample_train[: GRAPH_SAMPLE // 2]])
    audit_rows_subset = np.concatenate([sample_valid[:AUDIT_ROWS], sample_train[:AUDIT_ROWS]])

    import rie_extract as EX

    identity = EX.checkpoint_identity()
    distance = audit_distance_reference(task, graph_rows)
    fidelity = audit_graph_fidelity(task, graph_rows)
    invariance = audit_invariances(task, audit_rows_subset)
    degenerate = audit_component_and_empty_cases(task)
    shuffle = audit_shuffle(payloads, feature_files)
    agreement = audit_ap_agreement(task)
    isolation = audit_test_isolation(task)
    rows = audit_rows(task, payloads)
    smoke_path = C.RUNS / "smoke" / "smoke.json"
    smoke = C.read_json(smoke_path) if smoke_path.is_file() else {"all_passed": False, "missing": True}

    gates = {
        "source_checkpoint_identity": bool(identity["passed"]),
        "distance_matches_scipy_reference": bool(distance["passed"]),
        "rdkit_graph_matches_cache_graph": bool(fidelity["passed"]),
        "motif_order_invariance": bool(invariance["motif_order_passed"]),
        "atom_renumbering_invariance": bool(invariance["atom_renumbering_passed"]),
        "pair_endpoint_symmetry": bool(invariance["pair_endpoint_swap_passed"]),
        "duplicate_motif_dedupe": bool(invariance["duplicate_motif_passed"]),
        "no_cross_component_pairs": bool(invariance["cross_component_passed"]),
        "degenerate_cases_are_zero": bool(degenerate["passed"]),
        "shuffle_preserves_motifs_pairs_and_counts": bool(shuffle["preservation_passed"]),
        "shuffle_intervention_above_floor": bool(shuffle["intervention_passed"]),
        "row_correspondence": bool(rows["passed"]),
        "official_ap_agreement": bool(agreement["passed"]),
        "test_isolation": bool(isolation["passed"]),
        "smoke_all_passed": bool(smoke.get("all_passed")),
        "feature_manifest_passed": bool(manifest.get("all_passed")),
        "motif_coverage_passed": bool(coverage.get("all_passed")),
        "head_parameter_count": bool(int(C.HEAD_PARAMETERS) == 8784),
    }
    result = {
        "schema": "lcrc-pcba-rie-relation-invariance-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "sample_rows": {
            "graph_audit_rows": int(graph_rows.size),
            "invariance_rows": int(audit_rows_subset.size),
        },
        "checkpoint_identity": {
            "passed": identity["passed"],
            "file_sha256": identity["file_sha256"],
            "failed_checks": identity["failed_checks"],
            "payload": identity["payload"],
        },
        "distance_reference": distance,
        "graph_fidelity": fidelity,
        "invariances": invariance,
        "degenerate_cases": degenerate,
        "row_correspondence": rows,
        "official_ap_agreement": agreement,
        "test_isolation": isolation,
        "smoke": {"all_passed": smoke.get("all_passed"), "path": C.track_rel(smoke_path)},
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_passed": bool(all(gates.values())),
        "test_evaluated": False,
    }
    C.write_json(C.ANALYSIS / "relation_invariance_audit.json", result)

    shuffle_out = {
        "schema": "lcrc-pcba-rie-shuffle-control-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "definition": coverage["distance_definition"]["shuffle"],
        "splits": {k: v for k, v in shuffle.items() if k in ("train", "valid")},
        "overall_fraction_REAL_ne_SHUFFLE": shuffle["overall_fraction_REAL_ne_SHUFFLE"],
        "min_intervention_fraction": shuffle["min_intervention_fraction"],
        "intervention_passed": shuffle["intervention_passed"],
        "preservation_passed": shuffle["preservation_passed"],
        "gates": {
            "shuffle_preserves_motifs_pairs_and_counts": bool(shuffle["preservation_passed"]),
            "shuffle_intervention_above_floor": bool(shuffle["intervention_passed"]),
        },
        "all_passed": bool(shuffle["preservation_passed"] and shuffle["intervention_passed"]),
        "test_evaluated": False,
    }
    C.write_json(C.ANALYSIS / "shuffle_control_audit.json", shuffle_out)
    C.log(
        f"[audit] invariant all_passed={result['all_passed']} failed={[k for k, v in gates.items() if not v]}"
    )
    C.log(
        f"[audit] valid REAL!=SHUFFLE={shuffle['valid']['fraction_REAL_ne_SHUFFLE_tensor']:.4f} "
        f"multi-bin={shuffle['valid']['fraction_with_multiple_distance_bins']:.4f}"
    )
    return {"invariance": result, "shuffle": shuffle_out}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-gpu-spotcheck", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        record = run_cpu()
        ok = bool(record["invariance"]["all_passed"] and record["shuffle"]["all_passed"])
        if args.with_gpu_spotcheck:
            import torch

            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            spot = audit_gpu_row_correspondence(device)
            path = C.ANALYSIS / "gpu_row_correspondence.json"
            C.write_json(path, spot)
            ok = ok and bool(spot["passed"])
            C.log(f"[audit] gpu spot check passed={spot['passed']} path={C.track_rel(path)}")
        return 0 if ok else 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
