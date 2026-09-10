"""Typed patch tokenizer correctness audit (ZINC, official train/valid only).

Diagnostic stage: audits the historical certificate-derived patch tokenizer,
validates the corrected canonical key against an independent ground-truth
isomorphism oracle, and quantifies the vocabulary / rarity / parameter shift.
It never loads the official test split and never trains a model.

Outputs (under ``results/typed_patch_tokenizer_correctness/``):

* ``audit.json`` -- Table A/B/C numbers and gate results.
* ``occurrences.npz`` -- integer-id occurrence tables for train and valid.
* ``keys.json`` -- id -> hex key maps (historical + corrected, r2 + r1).
* ``alias_molecules.json`` -- per-validation-molecule pre-registered alias
  subgroup labels.
* ``figures/*.png``.

Run: ``uv run python -m tracks.ksvd.experiments.luyin16.zinc_typed_tokenizer_audit``
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _data_to_graph,
    _load_zinc,
)
from tracks.ksvd.experiments.luyin16.typed_patch_tokenizer import (
    TYPED_TOKENIZER_V1_HISTORICAL,
    TYPED_TOKENIZER_V2_CORRECTED,
    build_colored_incidence,
    corrected_canonical_key,
    historical_certificate,
    typed_isomorphic_vf2,
    typed_patch_key,
    typed_tokenizer_fingerprint,
)

DATA_ROOT = REPO_ROOT / "data/ZINC"
RESULT_DIR = REPO_ROOT / "tracks/ksvd/results/typed_patch_tokenizer_correctness"
ZINC_PATCH_RADIUS = 2
PARENT_RADIUS = 1
PATCH_TOKENS_MAX = 8192
PARENT_TOKENS_MAX = 2048
MIN_FREQUENCY = 1
# Hybrid embedding config, identical to compact-v2 / compact-v4.
TYPED_WIDTH, TYPED_RANK, TYPED_FULL = 16, 4, 768
PARENT_WIDTH, PARENT_FULL = 8, 32


# --------------------------------------------------------------------------
# Occurrence extraction
# --------------------------------------------------------------------------
def _patch_occurrences(
    dataset: Sequence[Any], split: str, *, max_graphs: int | None = None
) -> dict[str, Any]:
    """One row per (molecule, center); incidence graphs built once per patch."""
    rows: dict[str, list[Any]] = {
        "hist_r2": [], "corr_r2": [], "hist_r1": [], "corr_r1": [],
        "inc_r2": [], "inc_r1": [], "mol_ids": [], "root_atoms": [],
        "n_patch_nodes": [],
    }
    histogram: Counter[int] = Counter()
    limit = len(dataset) if max_graphs is None else min(int(max_graphs), len(dataset))
    for mol_index in range(limit):
        data = dataset[mol_index]
        graph, node_types, edge_types = _data_to_graph(data)
        for center in graph.nodes:
            inc2 = build_colored_incidence(
                graph, int(center), node_types, edge_types, ZINC_PATCH_RADIUS
            )
            inc1 = build_colored_incidence(
                graph, int(center), node_types, edge_types, PARENT_RADIUS
            )
            rows["inc_r2"].append(inc2)
            rows["inc_r1"].append(inc1)
            rows["hist_r2"].append(historical_certificate(inc2))
            rows["corr_r2"].append(corrected_canonical_key(inc2))
            rows["hist_r1"].append(historical_certificate(inc1))
            rows["corr_r1"].append(corrected_canonical_key(inc1))
            rows["mol_ids"].append(mol_index)
            rows["root_atoms"].append(int(node_types[int(center)]))
            rows["n_patch_nodes"].append(int(inc2.n_nodes))
            histogram[int(inc2.n_nodes)] += 1
    rows["mol_ids"] = np.asarray(rows["mol_ids"], dtype=np.int64)
    rows["root_atoms"] = np.asarray(rows["root_atoms"], dtype=np.int64)
    rows["n_patch_nodes"] = np.asarray(rows["n_patch_nodes"], dtype=np.int64)
    rows["split"] = split
    rows["n_molecules"] = limit
    rows["n_occurrences"] = len(rows["hist_r2"])
    rows["path_node_histogram"] = {int(k): int(v) for k, v in sorted(histogram.items())}
    return rows


# --------------------------------------------------------------------------
# Correctness gates
# --------------------------------------------------------------------------
def _key_of(inc) -> bytes:
    return corrected_canonical_key(inc)


def _soundness_gate(occurrences: Mapping[str, Any], tag: str) -> dict[str, Any]:
    """Every corrected-key bucket must be a single ground-truth iso class."""
    keys = occurrences[f"corr_{tag}"]
    incs = occurrences[f"inc_{tag}"]
    reps: dict[bytes, Any] = {}
    failures = 0
    checked = 0
    examples: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        rep = reps.get(key)
        if rep is None:
            reps[key] = incs[index]
            continue
        checked += 1
        if not typed_isomorphic_vf2(rep, incs[index]):
            failures += 1
            if len(examples) < 5:
                examples.append({"occurrence_index": int(index)})
    return {
        "n_corrected_keys": int(len(reps)),
        "checked_occurrences_beyond_representative": int(checked),
        "soundness_failures": int(failures),
        "examples": examples,
    }


def _determinism_gate(occurrences: Mapping[str, Any], tag: str, n: int = 300) -> dict[str, Any]:
    keys = occurrences[f"corr_{tag}"]
    keys_again = [_key_of(inc) for inc in occurrences[f"inc_{tag}"][:n]]
    failures = sum(1 for a, b in zip(keys[:n], keys_again) if a != b)
    return {"cases": int(min(n, len(keys))), "failures": int(failures)}


def _permutation_invariance_gate(dataset, tag: str = "r2") -> dict[str, Any]:
    """Apply 100 random vertex permutations to representative patches."""
    from tracks.ksvd.code.graph import from_edges

    rng = random.Random(20260910)
    radius = ZINC_PATCH_RADIUS if tag == "r2" else PARENT_RADIUS
    cases = 0
    failures = 0
    for mol_index in range(min(40, len(dataset))):
        data = dataset[mol_index]
        graph, node_types, edge_types = _data_to_graph(data)
        nodes = list(graph.nodes)
        if len(nodes) < 3:
            continue
        for _ in range(4):
            center = int(rng.choice(nodes))
            base = typed_patch_key(
                TYPED_TOKENIZER_V2_CORRECTED, graph, center, node_types, edge_types, radius
            )
            for _ in range(100):
                order = list(range(len(nodes)))
                rng.shuffle(order)
                old_to_new = {old: order[position] for position, old in enumerate(nodes)}
                new_edges = [
                    tuple(sorted((old_to_new[int(left)], old_to_new[int(right)])))
                    for left, right in graph.edges()
                ]
                new_graph = from_edges(len(nodes), new_edges)
                new_nodes = np.zeros_like(node_types)
                for old in nodes:
                    new_nodes[old_to_new[old]] = node_types[old]
                new_edges_map: dict[tuple[int, int], int] = {}
                for left, right in graph.edges():
                    new_edges_map[
                        tuple(sorted((old_to_new[int(left)], old_to_new[int(right)])))
                    ] = int(edge_types[graph.edge_key(int(left), int(right))])
                permuted = typed_patch_key(
                    TYPED_TOKENIZER_V2_CORRECTED,
                    new_graph,
                    old_to_new[center],
                    new_nodes,
                    new_edges_map,
                    radius,
                )
                cases += 1
                if permuted != base:
                    failures += 1
    return {"cases": int(cases), "failures": int(failures)}


def _adversarial_separation_gate() -> list[dict[str, Any]]:
    from tracks.ksvd.code.graph import from_edges

    def build(n, edges, node_types, edge_types):
        return (
            from_edges(n, [tuple(edge) for edge in edges]),
            np.asarray(node_types, dtype=np.int64),
            {tuple(sorted(edge)): int(value) for edge, value in edge_types.items()},
        )

    path = [(0, 1), (1, 2), (2, 3)]
    zero = {edge: 0 for edge in path}
    cases = [
        ("different_root_atom", build(4, path, [0, 1, 1, 1], zero), 0,
         build(4, path, [2, 1, 1, 1], zero), 0, 2),
        ("different_atom_type_same_root", build(4, path, [0, 1, 1, 1], zero), 0,
         build(4, path, [0, 1, 2, 1], zero), 0, 2),
        ("different_bond_type", build(4, path, [0, 1, 1, 1], zero), 0,
         build(4, path, [0, 1, 1, 1], {e: (1 if e == (1, 2) else 0) for e in path}), 0, 2),
        ("same_adjacency_different_root_position", build(4, path, [0, 0, 0, 0], zero), 0,
         build(4, path, [0, 0, 0, 0], zero), 1, 2),
        ("star_vs_path_root_degree", build(4, [(0, 1), (0, 2), (0, 3)], [0, 0, 0, 0], {(0,1):0,(0,2):0,(0,3):0}), 0,
         build(4, path, [0, 0, 0, 0], zero), 0, 2),
    ]
    results: list[dict[str, Any]] = []
    for name, (lg, ln, le), lc, (rg, rn, re), rc, radius in cases:
        lk = typed_patch_key(TYPED_TOKENIZER_V2_CORRECTED, lg, lc, ln, le, radius)
        rk = typed_patch_key(TYPED_TOKENIZER_V2_CORRECTED, rg, rc, rn, re, radius)
        li = build_colored_incidence(lg, lc, ln, le, radius)
        ri = build_colored_incidence(rg, rc, rn, re, radius)
        results.append(
            {
                "name": name,
                "keys_differ": bool(lk != rk),
                "oracle_says_iso": bool(typed_isomorphic_vf2(li, ri)),
                "historical_keys_differ": bool(
                    historical_certificate(li) != historical_certificate(ri)
                ),
            }
        )
    return results


# --------------------------------------------------------------------------
# Historical bucket audit
# --------------------------------------------------------------------------
def _historical_bucket_audit(occurrences: Mapping[str, Any], tag: str) -> dict[str, Any]:
    """Split every historical bucket by ground-truth iso class."""
    hist_keys = occurrences[f"hist_{tag}"]
    incs = occurrences[f"inc_{tag}"]
    root_atoms = occurrences["root_atoms"]
    mol_ids = occurrences["mol_ids"]
    buckets: dict[bytes, list[int]] = defaultdict(list)
    for index, key in enumerate(hist_keys):
        buckets[key].append(index)

    aliased = 0
    aliased_occurrences = 0
    affected_molecules: set[int] = set()
    multiplicities: list[int] = []
    root_only = 0
    non_root = 0
    root_diversity: Counter[int] = Counter()
    multiplicity_hist: Counter[int] = Counter()
    over_split = 0
    examples: list[dict[str, Any]] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        distinct_roots = sorted({int(root_atoms[index]) for index in members})
        classes: list[list[int]] = []
        class_reps: list[Any] = []
        for index in members:
            matched = False
            for class_index, rep in enumerate(class_reps):
                if typed_isomorphic_vf2(rep, incs[index]):
                    classes[class_index].append(index)
                    matched = True
                    break
            if not matched:
                class_reps.append(incs[index])
                classes.append([index])
        n_classes = len(classes)
        if n_classes <= 1:
            continue
        aliased += 1
        aliased_occurrences += len(members)
        multiplicities.append(n_classes)
        multiplicity_hist[n_classes] += 1
        root_diversity[len(distinct_roots)] += 1
        for index in members:
            affected_molecules.add(int(mol_ids[index]))
        if len(distinct_roots) > 1:
            root_only += 1
        else:
            non_root += 1
        corrected = {_key_of(incs[index]) for index in members}
        if len(corrected) != n_classes:
            over_split += 1
        if len(examples) < 15:
            examples.append(
                {
                    "historical_key_sha256": hashlib.sha256(key).hexdigest()[:16],
                    "occurrences": len(members),
                    "n_true_classes": int(n_classes),
                    "n_corrected_keys": int(len(corrected)),
                    "n_distinct_root_atoms": int(len(distinct_roots)),
                    "class_sizes": sorted((len(cls) for cls in classes), reverse=True),
                    "root_atoms": distinct_roots,
                }
            )
    return {
        "unique_historical_tokens": int(len(buckets)),
        "aliased_tokens": int(aliased),
        "aliased_occurrences": int(aliased_occurrences),
        "affected_molecules": int(len(affected_molecules)),
        "max_split_multiplicity": int(max(multiplicities) if multiplicities else 1),
        "mean_split_multiplicity": float(np.mean(multiplicities) if multiplicities else 1.0),
        "root_only_collisions": int(root_only),
        "non_root_collisions": int(non_root),
        "root_diversity_distribution": {str(k): int(v) for k, v in sorted(root_diversity.items())},
        "split_multiplicity_histogram": {str(k): int(v) for k, v in sorted(multiplicity_hist.items())},
        "completeness_over_split_violations": int(over_split),
        "examples": examples,
    }


# --------------------------------------------------------------------------
# Vocabulary / parameter shift
# --------------------------------------------------------------------------
def _fit_vocab(keys: Sequence[bytes], maximum: int, minimum_frequency: int) -> dict[bytes, int]:
    counts: Counter[bytes] = Counter(keys)
    ordered = sorted(
        (key for key, count in counts.items() if count >= int(minimum_frequency)),
        key=lambda key: (-counts[key], key),
    )[: int(maximum)]
    return {key: index + 1 for index, key in enumerate(ordered)}


def _hybrid_params(vocab_with_oov: int, width: int, rank: int, full: int) -> int:
    if full >= vocab_with_oov:
        return int(full * width)
    return int(full * width + (vocab_with_oov - full) * rank + rank * width)


def _vocab_report_one(train_keys, valid_keys, *, is_r2: bool) -> dict[str, Any]:
    maximum = PATCH_TOKENS_MAX if is_r2 else PARENT_TOKENS_MAX
    counts = Counter(train_keys)
    vocab = _fit_vocab(train_keys, maximum, MIN_FREQUENCY)
    vocab_with_oov = len(vocab) + 1
    valid_oov_tokens = {key for key in set(valid_keys) if key not in vocab}
    valid_oov_occurrences = sum(1 for key in valid_keys if key not in vocab)
    rare = {key for key, count in counts.items() if count <= 5 and key in vocab}
    frequent = {key for key, count in counts.items() if count >= 20 and key in vocab}
    width = TYPED_WIDTH if is_r2 else PARENT_WIDTH
    rank = TYPED_RANK if is_r2 else 4
    full = TYPED_FULL if is_r2 else PARENT_FULL
    freq_values = sorted(counts.values())
    return {
        "unique_train_tokens": int(len(counts)),
        "vocab_known": int(len(vocab)),
        "vocab_with_oov": int(vocab_with_oov),
        "rare_le1": int(sum(1 for c in counts.values() if c == 1 and counts and True)),
        "rare_le2": int(sum(1 for c in counts.values() if c <= 2)),
        "rare_le5": int(len(rare)),
        "rare_le10": int(sum(1 for c in counts.values() if c <= 10)),
        "frequent_ge20": int(len(frequent)),
        "valid_oov_types": int(len(valid_oov_tokens)),
        "valid_oov_occurrences": int(valid_oov_occurrences),
        "valid_occurrences": int(len(valid_keys)),
        "embedding_params": _hybrid_params(vocab_with_oov, width, rank, full),
        "mean_log_frequency": float(np.mean([np.log1p(c) for c in freq_values])) if freq_values else 0.0,
        "minimum_frequency": int(min(freq_values)) if freq_values else 0,
        "vocab_truncated": bool(len(vocab) >= maximum and len(counts) > maximum),
    }


def _vocab_report(train_occ, valid_occ) -> dict[str, Any]:
    return {
        "r2_historical": _vocab_report_one(train_occ["hist_r2"], valid_occ["hist_r2"], is_r2=True),
        "r2_corrected": _vocab_report_one(train_occ["corr_r2"], valid_occ["corr_r2"], is_r2=True),
        "r1_historical": _vocab_report_one(train_occ["hist_r1"], valid_occ["hist_r1"], is_r2=False),
        "r1_corrected": _vocab_report_one(train_occ["corr_r1"], valid_occ["corr_r1"], is_r2=False),
    }


# --------------------------------------------------------------------------
# Alias subgroup labelling (validation) -- pre-registered on historical audit
# --------------------------------------------------------------------------
def _validation_alias_labels(train_occ, valid_occ) -> dict[str, Any]:
    train_inc = train_occ["inc_r2"]
    train_buckets: dict[bytes, list[int]] = defaultdict(list)
    for index, key in enumerate(train_occ["hist_r2"]):
        train_buckets[key].append(index)
    aliased_tokens: set[bytes] = set()
    root_tokens: set[bytes] = set()
    nonroot_tokens: set[bytes] = set()
    for key, members in train_buckets.items():
        if len(members) < 2:
            continue
        roots = {int(train_occ["root_atoms"][index]) for index in members}
        classes: list[Any] = []
        for index in members:
            if not any(typed_isomorphic_vf2(rep, train_inc[index]) for rep in classes):
                classes.append(train_inc[index])
        if len(classes) > 1:
            aliased_tokens.add(key)
            (root_tokens if len(roots) > 1 else nonroot_tokens).add(key)

    n_valid = int(valid_occ["n_molecules"])
    mol_patch = np.zeros(n_valid, dtype=np.int64)
    mol_alias = np.zeros(n_valid, dtype=np.int64)
    mol_root = np.zeros(n_valid, dtype=np.int64)
    mol_nonroot = np.zeros(n_valid, dtype=np.int64)
    mol_oov = np.zeros(n_valid, dtype=np.int64)
    valid_vocab = _fit_vocab(train_occ["hist_r2"], PATCH_TOKENS_MAX, MIN_FREQUENCY)
    for index, key in enumerate(valid_occ["hist_r2"]):
        mol = int(valid_occ["mol_ids"][index])
        mol_patch[mol] += 1
        if key in aliased_tokens:
            mol_alias[mol] += 1
        if key in root_tokens:
            mol_root[mol] += 1
        if key in nonroot_tokens:
            mol_nonroot[mol] += 1
        if key not in valid_vocab:
            mol_oov[mol] += 1
    ratio = mol_alias / np.maximum(mol_patch, 1)
    positive = ratio[mol_alias > 0]
    threshold = float(np.quantile(positive, 0.80)) if positive.size else 1.0
    high = (mol_alias > 0) & (ratio >= threshold)
    return {
        "n_molecules": int(n_valid),
        "unaffected": [int(i) for i in np.flatnonzero(mol_alias == 0)],
        "alias_affected": [int(i) for i in np.flatnonzero(mol_alias > 0)],
        "root_collision": [int(i) for i in np.flatnonzero(mol_root > 0)],
        "nonroot_collision": [int(i) for i in np.flatnonzero(mol_nonroot > 0)],
        "high_alias_burden": [int(i) for i in np.flatnonzero(high)],
        "high_alias_threshold": threshold,
        "alias_burden_ratio": ratio.tolist(),
        "mol_patch_count": mol_patch.tolist(),
        "mol_alias_count": mol_alias.tolist(),
        "mol_root_collision_count": mol_root.tolist(),
        "mol_nonroot_collision_count": mol_nonroot.tolist(),
        "mol_oov_count": mol_oov.tolist(),
        "n_aliased_train_tokens": int(len(aliased_tokens)),
        "n_root_collision_train_tokens": int(len(root_tokens)),
        "n_nonroot_collision_train_tokens": int(len(nonroot_tokens)),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _figures(report, alias_labels, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    mult = report["historical_bucket_audit"]["r2"].get("split_multiplicity_histogram", {})
    if mult:
        fig, ax = plt.subplots(figsize=(6, 4))
        keys = sorted(int(k) for k in mult)
        ax.bar([str(k) for k in keys], [mult[str(k)] for k in keys])
        ax.set_xlabel("ground-truth iso classes within a historical r2 token")
        ax.set_ylabel("number of historical tokens")
        ax.set_title("Figure 1 - historical r2 token split multiplicity")
        fig.tight_layout()
        path = out_dir / "fig1_split_multiplicity.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, hist in (
        ("historical", report.get("frequency_hist_r2_historical")),
        ("corrected", report.get("frequency_hist_r2_corrected")),
    ):
        if not hist:
            continue
        xs = list(range(len(hist)))
        ys = [hist[k] for k in hist]
        ax.bar([x + (0.0 if label == "historical" else 0.4) for x in xs], ys, width=0.4, label=label)
    ax.set_xticks(range(len(report.get("frequency_hist_r2_historical", {}))))
    ax.set_xticklabels(list(report.get("frequency_hist_r2_historical", {})), rotation=30)
    ax.set_xlabel("train token frequency bin")
    ax.set_ylabel("number of tokens")
    ax.set_title("Figure 2 - r2 token frequency distribution")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "fig2_frequency_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path))
    return written


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _key_ids(keys: Sequence[bytes]) -> tuple[np.ndarray, dict[int, bytes]]:
    mapping: dict[bytes, int] = {}
    ids = np.empty(len(keys), dtype=np.int64)
    for index, key in enumerate(keys):
        token = mapping.get(key)
        if token is None:
            token = len(mapping)
            mapping[key] = token
        ids[index] = token
    return ids, {token: key for key, token in mapping.items()}


def _save_occurrences(train_occ, valid_occ) -> None:
    """Save occurrence ids in a *shared* train+valid id space per tag.

    A shared id space lets downstream code compare validation occurrence
    frequencies to the train vocabulary directly.
    """
    arrays: dict[str, np.ndarray] = {}
    keys: dict[str, dict[str, str]] = {}
    for prefix, occ in (("train", train_occ), ("valid", valid_occ)):
        arrays[f"{prefix}_mol_ids"] = occ["mol_ids"]
        arrays[f"{prefix}_root_atoms"] = occ["root_atoms"]
        arrays[f"{prefix}_n_patch_nodes"] = occ["n_patch_nodes"]
    for tag in ("hist_r2", "corr_r2", "hist_r1", "corr_r1"):
        combined = list(train_occ[tag]) + list(valid_occ[tag])
        _, mapping = _key_ids(combined)
        key_to_id = {key: token for token, key in mapping.items()}
        arrays[f"train_{tag}_id"] = np.asarray(
            [key_to_id[key] for key in train_occ[tag]], dtype=np.int64
        )
        arrays[f"valid_{tag}_id"] = np.asarray(
            [key_to_id[key] for key in valid_occ[tag]], dtype=np.int64
        )
        keys[tag] = {str(token): key.hex() for token, key in mapping.items()}
    np.savez_compressed(RESULT_DIR / "occurrences.npz", **arrays)
    (RESULT_DIR / "keys.json").write_text(json.dumps(keys), encoding="utf-8")


def _freq_hist(counts: Counter[bytes]) -> dict[str, int]:
    buckets: Counter[str] = Counter()
    for count in counts.values():
        if count <= 10:
            key = str(count)
        elif count <= 100:
            key = "11-100"
        elif count <= 1000:
            key = "101-1000"
        else:
            key = "1000+"
        buckets[key] += 1
    return dict(buckets)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_audit(max_train: int | None = None, max_valid: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    train_dataset = _load_zinc(DATA_ROOT, "train")
    valid_dataset = _load_zinc(DATA_ROOT, "val")
    print(f"loaded ZINC train={len(train_dataset)} valid={len(valid_dataset)}", flush=True)
    train_occ = _patch_occurrences(train_dataset, "train", max_graphs=max_train)
    valid_occ = _patch_occurrences(valid_dataset, "valid", max_graphs=max_valid)
    print(
        f"occurrences train={train_occ['n_occurrences']} valid={valid_occ['n_occurrences']}",
        flush=True,
    )

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(DATA_ROOT),
        "typed_tokenizer_version": TYPED_TOKENIZER_V2_CORRECTED,
        "historical_tokenizer_version": TYPED_TOKENIZER_V1_HISTORICAL,
        "typed_tokenizer_fingerprint": typed_tokenizer_fingerprint(
            TYPED_TOKENIZER_V2_CORRECTED, ZINC_PATCH_RADIUS
        ),
        "n_train_molecules": int(train_occ["n_molecules"]),
        "n_valid_molecules": int(valid_occ["n_molecules"]),
        "n_train_occurrences": int(train_occ["n_occurrences"]),
        "n_valid_occurrences": int(valid_occ["n_occurrences"]),
        "patch_node_histogram": train_occ["path_node_histogram"],
    }

    print("determinism ...", flush=True)
    determinism = {"r2": _determinism_gate(train_occ, "r2"), "r1": _determinism_gate(train_occ, "r1")}
    print("permutation invariance ...", flush=True)
    permutation = {
        "r2": _permutation_invariance_gate(train_dataset, "r2"),
        "r1": _permutation_invariance_gate(train_dataset, "r1"),
    }
    print("adversarial separation ...", flush=True)
    adversarial = _adversarial_separation_gate()
    print("soundness gate ...", flush=True)
    soundness = {
        "train_r2": _soundness_gate(train_occ, "r2"),
        "train_r1": _soundness_gate(train_occ, "r1"),
        "valid_r2": _soundness_gate(valid_occ, "r2"),
        "valid_r1": _soundness_gate(valid_occ, "r1"),
    }
    total_failures = (
        determinism["r2"]["failures"] + determinism["r1"]["failures"]
        + permutation["r2"]["failures"] + permutation["r1"]["failures"]
        + sum(0 if case["keys_differ"] else 1 for case in adversarial)
        + sum(stats["soundness_failures"] for stats in soundness.values())
    )
    report["correctness_gates"] = {
        "permutation_invariance": permutation,
        "adversarial_separation": adversarial,
        "determinism": determinism,
        "soundness": soundness,
        "total_failures": int(total_failures),
    }
    if total_failures:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "audit.FAILED.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise RuntimeError(f"HARD FAIL: {total_failures} correctness-gate failures")

    print("historical bucket audit r2 ...", flush=True)
    r2_audit = _historical_bucket_audit(train_occ, "r2")
    print("historical bucket audit r1 ...", flush=True)
    r1_audit = _historical_bucket_audit(train_occ, "r1")
    report["historical_bucket_audit"] = {"r2": r2_audit, "r1": r1_audit}

    print("vocabulary shift ...", flush=True)
    report["vocabulary"] = _vocab_report(train_occ, valid_occ)
    report["frequency_hist_r2_historical"] = _freq_hist(Counter(train_occ["hist_r2"]))
    report["frequency_hist_r2_corrected"] = _freq_hist(Counter(train_occ["corr_r2"]))

    print("validation alias labels ...", flush=True)
    alias_labels = _validation_alias_labels(train_occ, valid_occ)
    report["alias_subgroups"] = {
        "n_unaffected": len(alias_labels["unaffected"]),
        "n_alias_affected": len(alias_labels["alias_affected"]),
        "n_root_collision": len(alias_labels["root_collision"]),
        "n_nonroot_collision": len(alias_labels["nonroot_collision"]),
        "n_high_alias_burden": len(alias_labels["high_alias_burden"]),
        "high_alias_threshold": alias_labels["high_alias_threshold"],
        "n_aliased_train_tokens": alias_labels["n_aliased_train_tokens"],
        "n_root_collision_train_tokens": alias_labels["n_root_collision_train_tokens"],
        "n_nonroot_collision_train_tokens": alias_labels["n_nonroot_collision_train_tokens"],
    }
    report["runtime_seconds"] = float(time.perf_counter() - started)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    report["figures"] = _figures(report, alias_labels, RESULT_DIR / "figures")
    (RESULT_DIR / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (RESULT_DIR / "alias_molecules.json").write_text(
        json.dumps(alias_labels, indent=2), encoding="utf-8"
    )
    _save_occurrences(train_occ, valid_occ)
    print(f"wrote {RESULT_DIR}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    args = parser.parse_args(argv)
    report = run_audit(max_train=args.max_train, max_valid=args.max_valid)
    print(json.dumps(report["correctness_gates"], indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
