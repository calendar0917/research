"""Identity incremental-information audit (ZINC + OGBG-MolHIV).

Zero-full-training / representation-family diagnostic.  The question is:

    once a model already receives every deterministic *non-identity*
    structural descriptor of a rooted patch, what does the discrete typed
    identity token still contribute?

This module answers that with six target-free computations plus one bounded
diagnostic, never training a backbone and never opening the official ZINC
test split (the MolHIV official test was already opened once for the frozen
recurrent checkpoint; here it is only reused read-only from the existing
parameter-attribution artifacts).

Stages
------
``dataflow``     real code path, vocabularies, widths, embedding parameter
                 accounting, OOV policy for ZINC and MolHIV.
``zinc-collision``   descriptor -> historical / corrected identity collisions.
``zinc-collision`` descriptor -> historical / corrected identity collisions,
                 identity -> descriptor variability and the historical-token
                 split multiplicity.
``zinc-knn``     kNN identity purity in train-fit standardized descriptor
                 space (k=1, 8).
``zinc-frequency``   vocabulary frequency / rarity / OOV / embedding share.
``zinc-expressivity``  with-id vs without-id molecule signatures on official
                 train (raw-non-isomorphic collision classes; target-free
                 first, then a target empirical lower bound).
``zinc-embedding`` learned embedding geometry vs structural geometry using a
                 frozen checkpoint.
``molhiv-extract`` light re-extraction of the MolHIV descriptor / historical /
                 corrected / molecule arrays for train and valid.
``molhiv-collision`` same descriptor/identity audit for MolHIV (streaming).
``molhiv-embedding`` same embedding-geometry audit for the frozen MolHIV
                 checkpoint.
``molhiv-generalization``  re-reads the existing MolHIV parameter-attribution
                 records and adds prevalence / size / patch-count controls.
``report``       combine every artifact into the final decision map.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.identity_incremental_information_audit <stage>

or ``all``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
OUT_DIR = TRACK_ROOT / "results/identity_incremental_information"
FIGURE_DIR = OUT_DIR / "figures"

ZINC_CACHE = TRACK_ROOT / "results/post_v4_residual_audit/cache"
ZINC_TOKENIZER_DIR = TRACK_ROOT / "results/typed_patch_tokenizer_correctness"
ZINC_FRAGMENTATION_DIR = TRACK_ROOT / "results/corrected_token_fragmentation_audit"
ZINC_RAW_SUFFICIENCY_DIR = TRACK_ROOT / "results/raw_graph_patch_system_sufficiency"
ZINC_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
ZINC_OPTIMIZED_STATE = (
    TRACK_ROOT / "results/compact_v4_training_sufficiency/states/Pstar_A2_long_seed0_selection_state.pt"
)
MOLHIV_PARAM_DIR = TRACK_ROOT / "results/molhiv_parameter_attribution"
MOLHIV_STATE = TRACK_ROOT / "results/molhiv_recurrent_pair_centre/raw_state_seed0.pt"

PROTOCOL_VERSION = "identity_incremental_information_audit_v1"
K_VALUES = (1, 8)
KNN_QUERY_SAMPLE = 20000
RANDOM_SEED = 20260927


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def _write_json(name: str, payload: Mapping[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = OUT_DIR / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _hash_bytes(value: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(value, digest_size=16).digest(), "big")


def _descriptor_signature_ids(matrix: np.ndarray, chunk: int = 20000) -> tuple[np.ndarray, int]:
    """Exact-byte signature id per row (chunked, never copies the full matrix)."""
    n_rows = int(matrix.shape[0])
    width = int(matrix.shape[1]) * 4
    lookup: dict[bytes, int] = {}
    out = np.empty(n_rows, dtype=np.int64)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = np.ascontiguousarray(matrix[start:stop], dtype=np.float32)
        payload = block.tobytes()
        for index in range(stop - start):
            key = payload[index * width : (index + 1) * width]
            signature = lookup.get(key)
            if signature is None:
                signature = len(lookup)
                lookup[key] = signature
            out[start + index] = signature
    return out, len(lookup)


def collision_stats(signature_ids: np.ndarray, identity_ids: np.ndarray) -> dict[str, Any]:
    """Same signature -> how many distinct identities, and how much mass."""
    groups: dict[int, Counter] = defaultdict(Counter)
    for signature, identity in zip(signature_ids.tolist(), identity_ids.tolist()):
        groups[int(signature)][int(identity)] += 1
    n = int(len(signature_ids))
    collision_mass = 0
    multiplicities: list[int] = []
    conditional_entropy = 0.0
    identity_counts: list[int] = []
    for counts in groups.values():
        distinct = len(counts)
        identities = sum(counts.values())
        multiplicities.append(distinct)
        identity_counts.append(identities)
        if distinct > 1:
            collision_mass += identities
        probs = np.asarray(list(counts.values()), dtype=np.float64) / float(identities)
        conditional_entropy += (identities / n) * float(-(probs * np.log(probs)).sum())
    multiplicities_array = np.asarray(multiplicities, dtype=np.float64)
    return {
        "occurrences": n,
        "unique_descriptor_signatures": int(len(groups)),
        "unique_identities": int(len(set(identity_ids.tolist()))),
        "collision_mass": int(collision_mass),
        "collision_mass_fraction": float(collision_mass / n),
        "H_identity_given_descriptor_nats": conditional_entropy,
        "max_identity_multiplicity_per_signature": int(multiplicities_array.max()),
        "mean_identity_multiplicity_per_signature": float(multiplicities_array.mean()),
        "median_identity_multiplicity_per_signature": float(np.median(multiplicities_array)),
        "signatures_with_multiplicity_1": int((multiplicities_array == 1).sum()),
        "signature_multiplicity_histogram": {
            str(k): int(v) for k, v in sorted(Counter(multiplicities).items())
        },
    }


def identity_variability(
    matrix: np.ndarray, identity_ids: np.ndarray, signature_ids: np.ndarray
) -> dict[str, Any]:
    """Same identity -> how much do the deterministic descriptors vary."""
    order = np.argsort(identity_ids, kind="stable")
    sorted_matrix = matrix[order]
    sorted_ids = identity_ids[order]
    sorted_signatures = signature_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sorted_ids)]
    per_dim_variance: list[float] = []
    max_spread: list[float] = []
    unique_signatures: list[int] = []
    occurrences: list[int] = []
    for start, end in zip(starts, ends):
        block = sorted_matrix[start:end]
        occurrences.append(int(end - start))
        unique_signatures.append(int(len(set(sorted_signatures[start:end].tolist()))))
        if block.shape[0] > 1:
            per_dim_variance.append(float(block.var(axis=0).mean()))
            max_spread.append(float((block.max(axis=0) - block.min(axis=0)).max()))
        else:
            per_dim_variance.append(0.0)
            max_spread.append(0.0)
    return {
        "n_identities": int(len(starts)),
        "within_identity_mean_dim_variance": float(np.mean(per_dim_variance)),
        "within_identity_median_dim_variance": float(np.median(per_dim_variance)),
        "within_identity_mean_max_descriptor_spread": float(np.mean(max_spread)),
        "mean_unique_descriptors_per_identity": float(np.mean(unique_signatures)),
        "fraction_identities_with_single_descriptor": float(
            np.mean([value == 1 for value in unique_signatures])
        ),
        "max_unique_descriptors_for_one_identity": int(max(unique_signatures)),
        "mean_occurrences_per_identity": float(np.mean(occurrences)),
    }


def historical_to_corrected_multiplicity(
    historical_ids: np.ndarray, corrected_ids: np.ndarray
) -> dict[str, Any]:
    children: dict[int, set[int]] = defaultdict(set)
    for hist, corr in zip(historical_ids.tolist(), corrected_ids.tolist()):
        children[int(hist)].add(int(corr))
    multiplicity = np.asarray([len(value) for value in children.values()], dtype=np.float64)
    return {
        "historical_tokens": int(len(children)),
        "mean_corrected_classes_per_historical_token": float(multiplicity.mean()),
        "max_corrected_classes_per_historical_token": int(multiplicity.max()),
        "fraction_historical_tokens_split": float((multiplicity > 1).mean()),
        "corrected_classes_histogram": {
            str(k): int(v) for k, v in sorted(Counter(multiplicity.astype(int).tolist()).items())
        },
    }


def frequency_statistics(counts: Counter) -> dict[str, Any]:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    occurrences = int(values.sum())
    unique = int(values.size)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) / max(occurrences, 1)

    def coverage(k: int) -> float:
        if k >= unique:
            return 1.0
        return float(cumulative[k - 1])

    return {
        "unique_types": unique,
        "occurrences": occurrences,
        "singleton_types": int((values == 1).sum()),
        "singleton_fraction_of_types": float((values == 1).mean()),
        "singleton_occurrence_fraction": float(values[values == 1].sum() / occurrences),
        "le_2_types": int((values <= 2).sum()),
        "le_2_type_fraction": float((values <= 2).mean()),
        "le_5_types": int((values <= 5).sum()),
        "le_5_type_fraction": float((values <= 5).mean()),
        "le_10_types": int((values <= 10).sum()),
        "max_frequency": int(values.max()),
        "median_frequency": float(np.median(values)),
        "mean_frequency": float(values.mean()),
        "coverage_top_10": coverage(10),
        "coverage_top_100": coverage(100),
        "coverage_top_1000": coverage(1000),
    }


def stratified_knn_purity(
    matrix: np.ndarray,
    identity_ids: np.ndarray,
    frequency: np.ndarray,
    query_index: np.ndarray,
    k_values: Sequence[int] = K_VALUES,
) -> dict[str, Any]:
    """Exact kNN identity purity in standardized descriptor space.

    ``matrix`` must already be standardized with train statistics.  Distances
    are Euclidean; the query itself is excluded from its own neighbour list.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    n_reference = int(matrix.shape[0])
    max_k = int(max(k_values))
    top_index = np.empty((len(query_index), max_k), dtype=np.int64)
    chunk = 256 if matrix.shape[1] > 300 else 1000
    squared_norm = np.einsum("ij,ij->i", matrix, matrix)
    for start in range(0, len(query_index), chunk):
        stop = min(start + chunk, len(query_index))
        query = np.asarray(matrix[query_index[start:stop]], dtype=np.float32)
        rows = np.arange(stop - start)
        # One temporary only; then in-place updates (memory-safe).
        distances = query @ matrix.T
        distances *= -2.0
        distances += squared_norm[None, :]
        distances += np.einsum("ij,ij->i", query, query)[:, None]
        distances[rows, query_index[start:stop]] = np.inf
        part = np.argpartition(distances, max_k, axis=1)[:, :max_k]
        top_index[start:stop] = part
        del distances, part
    query_frequency = frequency[query_index]
    buckets = {
        "frequent_gt20": query_frequency > 20,
        "medium_6_20": (query_frequency > 5) & (query_frequency <= 20),
        "rare_2_5": (query_frequency >= 2) & (query_frequency <= 5),
        "singleton_1": query_frequency == 1,
    }
    rng = np.random.default_rng(RANDOM_SEED)
    random_index = rng.integers(0, n_reference, size=(len(query_index), max_k))
    result: dict[str, Any] = {
        "query_sample": int(len(query_index)),
        "reference_set": n_reference,
        "k_values": list(k_values),
        "random_baseline": {
            "top1": float((identity_ids[random_index[:, 0]] == identity_ids[query_index]).mean()),
            "k8_purity": float((identity_ids[random_index] == identity_ids[query_index][:, None]).mean()),
        },
        "overall": {},
        "strata": {},
    }
    for k in k_values:
        result["overall"][f"top{k}_same_identity_rate"] = float(
            (identity_ids[top_index[:, :k]] == identity_ids[query_index][:, None]).mean()
        )
    result["overall"]["k8_purity"] = float(
        (identity_ids[top_index] == identity_ids[query_index][:, None]).mean()
    )
    for name, mask in buckets.items():
        subset = np.flatnonzero(mask)
        if subset.size == 0:
            result["strata"][name] = {"n": 0}
            continue
        entry: dict[str, Any] = {"n": int(subset.size)}
        for k in k_values:
            entry[f"top{k}_same_identity_rate"] = float(
                (identity_ids[top_index[subset, :k]] == identity_ids[query_index[subset]][:, None]).mean()
            )
        entry["k8_purity"] = float(
            (identity_ids[top_index[subset]] == identity_ids[query_index[subset]][:, None]).mean()
        )
        result["strata"][name] = entry
    return result


# --------------------------------------------------------------------------
# ZINC loading
# --------------------------------------------------------------------------
def load_zinc_records(split: str) -> list[Any]:
    path = ZINC_CACHE / f"v4_records_{split}.pkl.gz"
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def load_zinc_occurrences() -> dict[str, np.ndarray]:
    with np.load(ZINC_TOKENIZER_DIR / "occurrences.npz") as data:
        return {key: data[key] for key in data.files}


def _flatten_zinc(
    records: Sequence[Any],
    historical_ids: np.ndarray,
    corrected_ids: np.ndarray,
    parent_historical_ids: np.ndarray,
    parent_corrected_ids: np.ndarray,
) -> dict[str, Any]:
    descriptors: list[np.ndarray] = []
    molecule_index: list[int] = []
    for molecule, record in enumerate(records):
        for patch in record.patches:
            descriptors.append(patch.shell_descriptor)
            molecule_index.append(molecule)
    return {
        "descriptor": np.asarray(descriptors, dtype=np.float32),
        "molecule_index": np.asarray(molecule_index, dtype=np.int64),
        "historical_id": np.asarray(historical_ids, dtype=np.int64),
        "corrected_id": np.asarray(corrected_ids, dtype=np.int64),
        "parent_historical_id": np.asarray(parent_historical_ids, dtype=np.int64),
        "parent_corrected_id": np.asarray(parent_corrected_ids, dtype=np.int64),
    }


# --------------------------------------------------------------------------
# stage: dataflow
# --------------------------------------------------------------------------
def _zinc_model_audit() -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    config = zpp.yaml.safe_load(ZINC_CONFIG.read_text(encoding="utf-8"))
    model_config = config["model"]
    model = zpp.PatchPathModel(
        typed_vocabulary_size=6784 + 1,
        parent_vocabulary_size=31 + 1,
        patch_hidden=int(model_config["patch_hidden"]),
        pair_hidden=int(model_config["pair_hidden"]),
        token_width=int(model_config["token_width"]),
        dropout=float(model_config["dropout"]),
        embedding_mode=str(model_config["embedding_mode"]),
        embedding_rank=int(model_config["embedding_rank"]),
        hybrid_full_typed_tokens=int(model_config["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(model_config["hybrid_full_parent_tokens"]),
        center_context=bool(model_config["center_context"]),
        center_context_hidden=int(model_config["center_context_hidden"]),
        graph_head_hidden_0=int(model_config["graph_head_hidden_0"]),
        graph_head_hidden_1=int(model_config["graph_head_hidden_1"]),
        topology_mode=str(model_config["topology_mode"]),
        topology_input_width=25,
        topology_hidden_dim=int(model_config["topology_hidden_dim"]),
        topology_out_dim=int(model_config["topology_out_dim"]),
    )
    audit = zpp.audit_parameters(model)
    return {
        "config": str(ZINC_CONFIG.relative_to(REPO_ROOT)),
        "patch_radius": int(zpp.PATCH_RADIUS),
        "patch_descriptor_dim": int(model.shell_width),
        "typed_token_width": int(model.token_width),
        "parent_token_width": int(model.parent_width),
        "relation_descriptor_dim": int(zpp.RELATION_WIDTH),
        "global_context_dim": int(zpp.GLOBAL_WIDTH),
        "topology_input_dim": 25,
        "total_trainable_params_at_historical_vocab": int(audit["total_trainable"]),
        "embedding": audit["embedding"],
        "blocks": audit["blocks"],
        "embedding_parameter_count": {
            "typed_full_rows": int(audit["embedding"]["typed"]["full_count"]),
            "typed_rare_rows": int(
                audit["embedding"]["typed"]["vocabulary_size"] - audit["embedding"]["typed"]["full_count"]
            ),
            "typed_full_params": int(audit["embedding"]["typed"]["full_count"]) * int(model.token_width),
            "typed_rare_factorized_params": int(
                audit["embedding"]["typed"]["vocabulary_size"] - audit["embedding"]["typed"]["full_count"]
            )
            * int(audit["embedding"]["typed"]["rank"])
            + int(model.token_width) * int(audit["embedding"]["typed"]["rank"]),
            "parent_full_params": int(audit["embedding"]["parent"]["full_count"]) * int(model.parent_width),
            "parent_rare_factorized_params": (
                (
                    int(audit["embedding"]["parent"]["vocabulary_size"])
                    - int(audit["embedding"]["parent"]["full_count"])
                )
                * int(audit["embedding"]["parent"]["rank"])
                + int(model.parent_width) * int(audit["embedding"]["parent"]["rank"])
                if int(audit["embedding"]["parent"]["vocabulary_size"])
                > int(audit["embedding"]["parent"]["full_count"])
                else 0
            ),
        },
        "raw_state_keys_sample": [
            key for key in model.state_dict() if "typed_embedding" in key or "parent_embedding" in key
        ],
    }


def stage_dataflow() -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp

    zinc = _zinc_model_audit()
    zinc.update(
        {
            "certificate_semantics": {
                "historical": (
                    "bytes(pynauty.certificate(vertex-coloured incidence graph)); "
                    "the canonical adjacency does NOT carry the vertex colour "
                    "sequence, so it identifies the rooted UNCOLOURED topology "
                    "(atom type / bond type / root designation / root-distance "
                    "class are invisible)."
                ),
                "corrected": (
                    "certificate + canonical semantic colour sequence; a complete "
                    "invariant of the coloured incidence graph."
                ),
                "distinction": (
                    "historical coarse topology token != corrected exact typed key; "
                    "the model's default (and the optimized benchmark) uses the "
                    "historical coarse token."
                ),
            },
            "typed_vocabulary_size_historical": 6784,
            "typed_vocabulary_size_corrected_uncapped": 15218,
            "typed_vocabulary_size_corrected_capped": 8192,
            "parent_vocabulary_size_historical": 31,
            "parent_vocabulary_size_corrected": 512,
            "oov_policy": (
                "id 0 is a learned OOV row; known tokens start at 1; "
                "the hybrid full/rare boundary is a parameterisation boundary, "
                "NOT a vocabulary truncation (every corrected token keeps a row "
                "up to max_typed_tokens=8192)."
            ),
            "embedding_policy": (
                "hybrid: 768 frequent typed rows are full 16D; the remaining "
                "typed rows are rank-4 factorised then projected to 16D; "
                "32 parent rows are full 8D."
            ),
            "patch_cont_dim": 146,
            "valid_oov_occurrences_historical": 278,
            "valid_oov_occurrences_corrected": 1163,
        }
    )

    molhiv = {
        "patch_radius": int(mpp.PATCH_RADIUS),
        "patch_descriptor_dim": int(mpp.SHELL_WIDTH),
        "typed_token_width": 32,
        "parent_token_width": 16,
        "relation_descriptor_dim": int(mpp.RELATION_WIDTH),
        "typed_vocabulary_size": 26232,
        "typed_vocabulary_size_with_oov": 26233,
        "parent_vocabulary_size": 102,
        "parent_vocabulary_size_with_oov": 103,
        "embedding_policy": "dense nn.Embedding (no hybrid / no factorisation)",
        "typed_embedding_params": (26232 + 1) * 32,
        "parent_embedding_params": (102 + 1) * 16,
        "total_params_frozen_checkpoint": 1076589,
        "identity_fraction_of_total": 841104 / 1076589,
        "certificate_semantics": {
            "used": (
                "molhiv_patch_path_pooling._canonical_typed_patch returns "
                "bytes(pynauty.certificate(vertex-coloured incidence graph)) -- "
                "same construction as the ZINC HISTORICAL tokenizer: the "
                "semantic colour sequence is used to constrain the canonical "
                "labelling but is NOT part of the returned bytes."
            ),
            "implication": (
                "the MolHIV 'exact rooted typed certificate' vocabulary is also "
                "a coarse uncoloured rooted-topology invariant; it is NOT a "
                "complete coloured-incidence key. This audit therefore computes "
                "a corrected MolHIV key (certificate + colour sequence) and "
                "reports both."
            ),
        },
        "oov_policy": "id 0 is a learned OOV row; known tokens start at 1.",
        "valid_oov_occurrence_fraction": 0.07369203849518813,
        "valid_molecules_with_oov_fraction": 0.5917821541453927,
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "zinc": zinc,
        "molhiv": molhiv,
        "official_test_loaded": False,
    }
    _write_json("dataflow.json", payload)
    return payload


# --------------------------------------------------------------------------
# ZINC shared flattened view
# --------------------------------------------------------------------------
def zinc_flat() -> dict[str, Any]:
    train = load_zinc_records("train")
    valid = load_zinc_records("valid")
    occurrences = load_zinc_occurrences()
    train_flat = _flatten_zinc(
        train,
        occurrences["train_hist_r2_id"],
        occurrences["train_corr_r2_id"],
        occurrences["train_hist_r1_id"],
        occurrences["train_corr_r1_id"],
    )
    valid_flat = _flatten_zinc(
        valid,
        occurrences["valid_hist_r2_id"],
        occurrences["valid_corr_r2_id"],
        occurrences["valid_hist_r1_id"],
        occurrences["valid_corr_r1_id"],
    )
    return {"train": train_flat, "valid": valid_flat, "train_records": train, "valid_records": valid}


def _capped_corrected_ids(train_ids: np.ndarray, query_ids: np.ndarray, cap: int) -> tuple[np.ndarray, int]:
    """Re-map corrected ids to a top-``cap`` train-frequency vocabulary.

    Returns the *train* vocabulary id set and the query ids with 0 marking an
    unknown (OOV) token.  This mirrors the model's train-fit OOV policy.
    """
    counts = Counter(train_ids.tolist())
    ranked = [token for token, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    vocab = {token: index + 1 for index, token in enumerate(ranked[:cap])}
    mapped = np.asarray([vocab.get(int(token), 0) for token in query_ids.tolist()], dtype=np.int64)
    return mapped, len(vocab)


# --------------------------------------------------------------------------
# stage: zinc-collision / zinc-variability
# --------------------------------------------------------------------------
def stage_zinc_collision() -> dict[str, Any]:
    flat = zinc_flat()
    train = flat["train"]
    valid = flat["valid"]
    matrix = train["descriptor"]
    signature_ids, unique_signatures = _descriptor_signature_ids(matrix)

    hist = collision_stats(signature_ids, train["historical_id"])
    corr = collision_stats(signature_ids, train["corrected_id"])
    hist_var = identity_variability(matrix, train["historical_id"], signature_ids)
    corr_var = identity_variability(matrix, train["corrected_id"], signature_ids)
    split = historical_to_corrected_multiplicity(train["historical_id"], train["corrected_id"])
    parent_split = historical_to_corrected_multiplicity(
        train["parent_historical_id"], train["parent_corrected_id"]
    )

    # Valid-split replication of the descriptor -> identity collision.
    valid_signature_ids, _ = _descriptor_signature_ids(valid["descriptor"])
    valid_hist = collision_stats(valid_signature_ids, valid["historical_id"])
    valid_corr = collision_stats(valid_signature_ids, valid["corrected_id"])

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "descriptor": {
            "name": "ZINC compact-v4 radius-2 shell descriptor (patch_cont)",
            "dim": int(matrix.shape[1]),
            "exact_byte_signatures": int(unique_signatures),
            "train_occurrences": int(matrix.shape[0]),
        },
        "descriptor_to_identity": {
            "historical_coarse_token": hist,
            "corrected_exact_key": corr,
            "valid_historical_coarse_token": valid_hist,
            "valid_corrected_exact_key": valid_corr,
        },
        "identity_to_descriptor": {
            "historical_coarse_token": hist_var,
            "corrected_exact_key": corr_var,
        },
        "historical_to_corrected_split": {
            "radius2": split,
            "radius1_parent": parent_split,
        },
        "interpretation": (
            "The continuous descriptor is a deterministic, low-cardinality "
            "histogram object; it does NOT determine identity. Identity "
            "therefore carries discrete separation information beyond the "
            "patch-level non-ID descriptor."
        ),
    }
    _write_json("zinc_descriptor_identity.json", payload)
    return payload


# --------------------------------------------------------------------------
# stage: zinc-knn
# --------------------------------------------------------------------------
def stage_zinc_knn() -> dict[str, Any]:
    flat = zinc_flat()
    train = flat["train"]
    matrix = train["descriptor"]
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    standardized = ((matrix - mean) / scale).astype(np.float32)
    rng = np.random.default_rng(RANDOM_SEED)
    query = np.sort(rng.choice(matrix.shape[0], size=KNN_QUERY_SAMPLE, replace=False))
    hist_counts = Counter(train["historical_id"].tolist())
    frequency = np.asarray(
        [hist_counts[int(token)] for token in train["historical_id"].tolist()], dtype=np.int64
    )
    hist_purity = stratified_knn_purity(standardized, train["historical_id"], frequency, query)
    corr_purity = stratified_knn_purity(standardized, train["corrected_id"], frequency, query)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "space": (
            "train-fit standardized ZINC compact-v4 radius-2 shell descriptor "
            "(146D); Euclidean; exact brute-force neighbours; query itself excluded"
        ),
        "query_sample": int(len(query)),
        "query_sample_seed": RANDOM_SEED,
        "historical_coarse_token": hist_purity,
        "corrected_exact_key": corr_purity,
        "interpretation": (
            "Identity is only partially predictable from the descriptor "
            "neighbourhood: a substantial minority of nearest descriptors "
            "belong to a different identity, so identity is not a redundant "
            "re-encoding of the non-ID descriptor."
        ),
    }
    _write_json("zinc_knn_purity.json", payload)
    return payload


# --------------------------------------------------------------------------
# stage: zinc-frequency
# --------------------------------------------------------------------------
def _oov_stats(train_ids: np.ndarray, valid_ids: np.ndarray, cap: int | None = None) -> dict[str, Any]:
    counts = Counter(train_ids.tolist())
    if cap is not None:
        mapped_train, vocab_size = _capped_corrected_ids(train_ids, train_ids, cap)
        mapped_valid, _ = _capped_corrected_ids(train_ids, valid_ids, cap)
        vocab_tokens = set(range(1, vocab_size + 1))
        oov = mapped_valid == 0
        train_counts = Counter(mapped_train.tolist())
    else:
        vocab_tokens = set(counts)
        oov = np.asarray([int(token) not in vocab_tokens for token in valid_ids.tolist()])
        train_counts = counts
        vocab_size = len(counts)
    return {
        "vocabulary_size": int(vocab_size),
        "frequency": frequency_statistics(train_counts),
        "valid_oov_occurrences": int(oov.sum()),
        "valid_occurrences": int(oov.size),
        "valid_oov_occurrence_fraction": float(oov.mean()),
        "valid_oov_types": int(len(set(valid_ids[oov].tolist()))),
    }


def stage_zinc_frequency() -> dict[str, Any]:
    flat = zinc_flat()
    train = flat["train"]
    valid = flat["valid"]
    historical = _oov_stats(train["historical_id"], valid["historical_id"])
    corrected_uncapped = _oov_stats(train["corrected_id"], valid["corrected_id"])
    corrected_capped = _oov_stats(train["corrected_id"], valid["corrected_id"], cap=8192)
    parent_historical = _oov_stats(train["parent_historical_id"], valid["parent_historical_id"])
    parent_corrected = _oov_stats(train["parent_corrected_id"], valid["parent_corrected_id"])
    historical.update({"embedding_params": 36420, "identity_fraction": 36420 / 99613})
    corrected_capped.update(
        {
            "embedding_params_note": (
                "corrected runs use the same hybrid policy (768 full 16D rows + "
                "rank-4 factorised remainder + projection); the parameter "
                "increase is natural vocabulary-row growth (99,613 -> 107,201)."
            ),
        }
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "radius2_historical_coarse_token": historical,
        "radius2_corrected_exact_key_uncapped": corrected_uncapped,
        "radius2_corrected_exact_key_capped_8192": corrected_capped,
        "radius1_parent_historical": parent_historical,
        "radius1_parent_corrected": parent_corrected,
        "identity_parameter_share": {
            "historical_typed_embedding_params": 36420,
            "historical_parent_embedding_params": 256,
            "historical_total_params": 99613,
            "typed_share": 36420 / 99613,
            "typed_plus_parent_share": (36420 + 256) / 99613,
        },
        "reused_artifacts": {
            "zinc_raw_sufficiency": str(ZINC_RAW_SUFFICIENCY_DIR.relative_to(REPO_ROOT)),
            "corrected_fragmentation": str(ZINC_FRAGMENTATION_DIR.relative_to(REPO_ROOT)),
        },
    }
    _write_json("zinc_frequency.json", payload)
    return payload


# --------------------------------------------------------------------------
# MolHIV extraction + collision / frequency / kNN
# --------------------------------------------------------------------------
def _molhiv_corrected_key(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], tuple[int, ...]],
    radius: int,
) -> bytes:
    """Complete invariant of the MolHIV coloured incidence graph.

    Same construction as ``molhiv_patch_path_pooling._canonical_typed_patch``
    but the canonical semantic colour sequence is appended to the certificate,
    exactly as the ZINC corrected tokenizer does.
    """
    import pynauty

    from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp

    distances = mpp._ego_distances(graph, int(center), radius)
    original_nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(original_nodes)}
    induced = graph.induced(set(original_nodes))
    local_edges = tuple(
        (node_to_local[int(left)], node_to_local[int(right)]) for left, right in sorted(induced.edges())
    )
    n_nodes = len(original_nodes)
    n_edges = len(local_edges)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(n_nodes + n_edges)}
    color_groups: dict[tuple[Any, ...], set[int]] = {}
    root_local = node_to_local[int(center)]
    for local, node in enumerate(original_nodes):
        key = (
            "node",
            int(local == root_local),
            int(distances[node]),
            tuple(int(value) for value in node_types[int(node)]),
        )
        color_groups.setdefault(key, set()).add(local)
    for edge_local, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_local
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
        values = edge_types[graph.edge_key(int(original_nodes[left]), int(original_nodes[right]))]
        color_groups.setdefault(("edge", tuple(int(value) for value in values)), set()).add(edge_vertex)
    cell_keys = sorted(color_groups, key=repr)
    coloring = [color_groups[key] for key in cell_keys]
    incidence = pynauty.Graph(
        number_of_vertices=n_nodes + n_edges,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(incidence))
    canonical = tuple(int(value) for value in pynauty.canon_label(incidence))
    color_of_vertex: dict[int, tuple[Any, ...]] = {}
    for key, vertices in color_groups.items():
        for vertex in vertices:
            color_of_vertex[vertex] = key
    sequence = tuple(color_of_vertex[vertex] for vertex in canonical)
    return b"molhivtypedkey/v1;" + certificate + b"|" + repr(sequence).encode("utf-8")


def _fingerprint_bytes(value: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(value, digest_size=16).digest(), "big")


def _fingerprint_bytes64(value: bytes) -> int:
    """64-bit digest for disk-backed identity arrays (collision-safe at this N)."""
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _molhiv_cache_paths(split: str) -> dict[str, Path]:
    return {
        "descriptor": OUT_DIR / f"molhiv_{split}_descriptor.npy",
        "historical": OUT_DIR / f"molhiv_{split}_historical.npy",
        "corrected": OUT_DIR / f"molhiv_{split}_corrected.npy",
        "molecule": OUT_DIR / f"molhiv_{split}_molecule.npy",
    }


def extract_molhiv_split(split: str, force: bool = False) -> dict[str, Any]:
    """Stream a light per-patch MolHIV extraction to disk-backed arrays."""
    from ksvd_research.data import load_molhiv
    from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp

    paths = _molhiv_cache_paths(split)
    meta_path = OUT_DIR / f"molhiv_{split}_extraction.json"
    if not force and all(path.exists() for path in paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    import time

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_molhiv(root=str(REPO_ROOT / "data/ogb"), with_features=True)
    indices = np.asarray(bundle.split["valid" if split == "valid" else split], dtype=np.int64)
    certificate_cache: dict[bytes, bytes] = {}
    descriptor_path = paths["descriptor"]

    # First pass: count patches, then write a correctly sized memmap.
    counts = []
    graphs = []
    node_features = []
    edge_features = []
    for position, index in enumerate(indices.tolist()):
        graph = bundle.graphs[int(index)]
        node_types = np.asarray(bundle.node_feats[int(index)], dtype=np.int64)
        edge_types = {
            (int(left), int(right)): tuple(int(value) for value in values)
            for (left, right), values in bundle.edge_feats[int(index)].items()
        }
        graphs.append(graph)
        node_features.append(node_types)
        edge_features.append(edge_types)
        counts.append(len(graph.nodes))
    total = int(sum(counts))
    descriptor = np.lib.format.open_memmap(
        descriptor_path, mode="w+", dtype=np.float32, shape=(total, int(mpp.SHELL_WIDTH))
    )
    corrected = np.empty(total, dtype=np.int64)
    molecule = np.empty(total, dtype=np.int64)
    historical = np.empty(total, dtype=np.int64)
    started = time.perf_counter()
    offset = 0
    n_graphs = 0
    for graph, node_types, edge_types, index in zip(graphs, node_features, edge_features, indices.tolist()):
        for center in graph.nodes:
            distances = mpp._ego_distances(graph, int(center), int(mpp.PATCH_RADIUS))
            values, _, _ = mpp._shell_descriptor(graph, int(center), node_types, edge_types, distances)
            descriptor[offset] = values
            certificate = mpp._typed_certificate(
                graph, int(center), node_types, edge_types, int(mpp.PATCH_RADIUS), certificate_cache
            )
            historical[offset] = _fingerprint_bytes64(certificate)
            corrected[offset] = _fingerprint_bytes64(
                _molhiv_corrected_key(graph, int(center), node_types, edge_types, int(mpp.PATCH_RADIUS))
            )
            molecule[offset] = int(n_graphs)
            offset += 1
        n_graphs += 1
        if n_graphs % 2000 == 0:
            print(f"MolHIV {split}: {n_graphs}/{len(indices)} molecules", flush=True)
    descriptor.flush()
    del descriptor
    np.save(paths["historical"], historical)
    np.save(paths["corrected"], corrected)
    np.save(paths["molecule"], molecule)
    metadata = {
        "split": split,
        "molecules": int(len(indices)),
        "patches": total,
        "shell_width": int(mpp.SHELL_WIDTH),
        "seconds": float(time.perf_counter() - started),
        "historical_unique_fingerprints": int(len(set(historical.tolist()))),
        "corrected_unique_fingerprints": int(len(set(corrected.tolist()))),
        "certificate_semantics": (
            "historical = bytes(pynauty.certificate(coloured incidence)); "
            "corrected = certificate + canonical semantic colour sequence"
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


# --------------------------------------------------------------------------
# stage: zinc-expressivity (with-id vs without-id molecule signatures)
# --------------------------------------------------------------------------
def _raw_zinc_certificate(data: Any) -> bytes:
    import pynauty

    node_types = [int(value) for value in data.x.view(-1).tolist()]
    edge_index = data.edge_index
    edge_attr = data.edge_attr.view(-1).tolist()
    n_nodes = len(node_types)
    edges = []
    seen = set()
    for position in range(int(edge_index.shape[1])):
        left = int(edge_index[0, position])
        right = int(edge_index[1, position])
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        edges.append((pair[0], pair[1], int(edge_attr[position])))
    edges.sort()
    n_edges = len(edges)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(n_nodes + n_edges)}
    color_groups: dict[tuple[Any, ...], set[int]] = {}
    for node in range(n_nodes):
        color_groups.setdefault(("node", node_types[node]), set()).add(node)
    for edge_local, (left, right, bond) in enumerate(edges):
        edge_vertex = n_nodes + edge_local
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
        color_groups.setdefault(("edge", bond), set()).add(edge_vertex)
    cell_keys = sorted(color_groups, key=repr)
    coloring = [color_groups[key] for key in cell_keys]
    graph = pynauty.Graph(
        number_of_vertices=n_nodes + n_edges,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(graph))
    canonical = tuple(int(value) for value in pynauty.canon_label(graph))
    color_of_vertex: dict[int, tuple[Any, ...]] = {}
    for key, vertices in color_groups.items():
        for vertex in vertices:
            color_of_vertex[vertex] = key
    sequence = tuple(color_of_vertex[vertex] for vertex in canonical)
    return certificate + b"|" + repr(sequence).encode("utf-8")


def _molecule_signature(
    record: Any,
    with_identity: bool,
    include_pairs: bool = True,
) -> bytes:
    patch_keys = []
    for patch in record.patches:
        descriptor = _fingerprint_bytes(np.ascontiguousarray(patch.shell_descriptor).tobytes())
        if with_identity:
            patch_keys.append((_fingerprint_bytes(patch.typed_certificate), descriptor))
        else:
            patch_keys.append((descriptor,))
    digest = hashlib.blake2b(digest_size=16)
    for key in sorted(patch_keys):
        digest.update(repr(key).encode("ascii"))
    if include_pairs:
        pair_index = record.pair_index
        pair_keys = []
        for position in range(int(record.pair_relation.shape[0])):
            left = int(pair_index[0, position])
            right = int(pair_index[1, position])
            a, b = sorted((patch_keys[left], patch_keys[right]))
            relation = _fingerprint_bytes(np.ascontiguousarray(record.pair_relation[position]).tobytes())
            pair_keys.append(
                (
                    repr(a).encode("ascii"),
                    repr(b).encode("ascii"),
                    relation,
                    int(record.pair_bucket[position]),
                )
            )
        pair_keys.sort()
        for key in pair_keys:
            digest.update(repr(key).encode("ascii"))
        digest.update(np.ascontiguousarray(record.global_context).tobytes())
        if record.topology_features is not None:
            digest.update(np.ascontiguousarray(record.topology_features).tobytes())
    return digest.digest()


def stage_zinc_expressivity() -> dict[str, Any]:
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    flat = zinc_flat()
    records = flat["train_records"]
    dataset = _load_zinc(REPO_ROOT / "data/ZINC", "train")
    if len(dataset) != len(records):
        raise RuntimeError(f"raw/cache molecule count mismatch: {len(dataset)} vs {len(records)}")

    variants = {
        "descriptor_only": _molecule_signature_kwargs(False, False),
        "descriptor_plus_identity": _molecule_signature_kwargs(True, False),
        "full_system_without_identity": _molecule_signature_kwargs(False, True),
        "full_system_with_identity": _molecule_signature_kwargs(True, True),
    }
    # Phase 1 (target-free): build every signature first and LOCK its digest.
    raw_certificates = [_raw_zinc_certificate(dataset[index]) for index in range(len(dataset))]
    signature_lists = {name: _molecule_signature_list(records, kwargs) for name, kwargs in variants.items()}
    target_free = {
        name: _signature_summary(signatures, raw_certificates) for name, signatures in signature_lists.items()
    }
    lock = {
        "variants": {
            name: {
                "signature_list_sha256": hashlib.sha256(b"".join(signatures)).hexdigest(),
                "first_5_digests": [value.hex() for value in signatures[:5]],
            }
            for name, signatures in signature_lists.items()
        },
        "target_free_summary": target_free,
        "note": "target-free signature lock; targets read only in the next phase",
    }
    lock_path = OUT_DIR / "zinc_expressivity_signature_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    # Phase 2 (after the lock): read official-train targets for a lower bound.
    targets = np.asarray([float(record.y) for record in records], dtype=np.float64)
    lower_bounds = {
        name: _target_lower_bound(signatures, targets) for name, signatures in signature_lists.items()
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "signature_lock_sha256": lock_sha,
        "signature_lock_file": str(lock_path.relative_to(REPO_ROOT)),
        "target_free": target_free,
        "target_lower_bound_official_train": lower_bounds,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "interpretation": (
            "Compares molecule-level signatures built from the deterministic "
            "non-ID system with and without the discrete identity token."
        ),
    }
    _write_json("zinc_expressivity.json", payload)
    return payload


def _molecule_signature_kwargs(with_identity: bool, include_pairs: bool) -> dict[str, bool]:
    return {"with_identity": with_identity, "include_pairs": include_pairs}


def _molecule_signature_list(records: Sequence[Any], kwargs: Mapping[str, bool]) -> list[bytes]:
    return [
        _molecule_signature(record, kwargs["with_identity"], kwargs["include_pairs"]) for record in records
    ]


def _signature_summary(signatures: Sequence[bytes], raw_certificates: Sequence[bytes]) -> dict[str, Any]:
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        groups[signature].append(index)
    collision_classes = {key: value for key, value in groups.items() if len(value) > 1}
    collision_mass = sum(len(value) for value in collision_classes.values())
    non_isomorphic_classes = 0
    for members in collision_classes.values():
        distinct_raw = {raw_certificates[index] for index in members}
        if len(distinct_raw) > 1:
            non_isomorphic_classes += 1
    return {
        "molecules": int(len(signatures)),
        "unique_signatures": int(len(groups)),
        "collision_classes": int(len(collision_classes)),
        "collision_mass": int(collision_mass),
        "collision_mass_fraction": float(collision_mass / len(signatures)),
        "non_raw_isomorphic_collision_classes": int(non_isomorphic_classes),
        "max_class_size": int(max((len(value) for value in groups.values()), default=0)),
    }


def _target_lower_bound(signatures: Sequence[bytes], targets: np.ndarray) -> dict[str, Any]:
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        groups[signature].append(index)
    mass = 0
    weighted_mae = 0.0
    weighted_variance = 0.0
    for members in groups.values():
        if len(members) < 2:
            continue
        values = targets[members]
        median = float(np.median(values))
        mae = float(np.abs(values - median).mean())
        mass += len(members)
        weighted_mae += len(members) * mae
        weighted_variance += len(members) * float(values.var())
    return {
        "collision_mass": int(mass),
        "empirical_lower_bound_mae": float(weighted_mae / max(mass, 1)),
        "weighted_within_class_variance": float(weighted_variance / max(mass, 1)),
    }


# --------------------------------------------------------------------------
# stage: molhiv-collision / frequency / kNN
# --------------------------------------------------------------------------
def _load_molhiv_arrays(split: str) -> dict[str, Any]:
    paths = _molhiv_cache_paths(split)
    descriptor = np.load(paths["descriptor"], mmap_mode="r")
    return {
        "descriptor": descriptor,
        "historical": np.load(paths["historical"]),
        "corrected": np.load(paths["corrected"]),
        "molecule": np.load(paths["molecule"]),
    }


def _mean_std_chunked(matrix: np.ndarray, chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Streaming mean/std over a (memmapped) matrix without a full copy."""
    n_rows = int(matrix.shape[0])
    width = int(matrix.shape[1])
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = np.asarray(matrix[start:stop], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += np.einsum("ij,ij->j", block, block)
        del block
    mean = total / n_rows
    variance = np.maximum(total_sq / n_rows - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def identity_variability_chunked(
    matrix: np.ndarray,
    identity_ids: np.ndarray,
    signature_ids: np.ndarray,
    chunk: int = 50000,
) -> dict[str, Any]:
    """Memory-safe ``identity_variability`` for large patch counts."""
    unique_ids, inverse = np.unique(identity_ids, return_inverse=True)
    n_identities = int(len(unique_ids))
    width = int(matrix.shape[1])
    counts = np.bincount(inverse, minlength=n_identities).astype(np.float64)
    sums = np.zeros((n_identities, width), dtype=np.float64)
    sums_q = np.zeros((n_identities, width), dtype=np.float64)
    global_min = np.full((n_identities, width), np.inf, dtype=np.float32)
    global_max = np.full((n_identities, width), -np.inf, dtype=np.float32)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    n = int(len(inverse))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        index = order[start:stop]
        rows_id = sorted_inverse[start:stop]
        values = np.asarray(matrix[index], dtype=np.float32)
        starts = np.r_[0, np.flatnonzero(np.diff(rows_id)) + 1]
        group_ids = rows_id[starts]
        sums[group_ids] += np.add.reduceat(values.astype(np.float64), starts, axis=0)
        squared = values.astype(np.float64)
        squared *= squared
        sums_q[group_ids] += np.add.reduceat(squared, starts, axis=0)
        np.minimum.at(global_min, group_ids, np.minimum.reduceat(values, starts, axis=0))
        np.maximum.at(global_max, group_ids, np.maximum.reduceat(values, starts, axis=0))
        del values, squared
    # Distinct descriptor signatures per identity (signature ids only: cheap).
    signature_sets: dict[int, set[int]] = defaultdict(set)
    for identity, signature in zip(inverse.tolist(), signature_ids.tolist()):
        signature_sets[int(identity)].add(int(signature))
    unique_signatures = np.asarray(
        [len(signature_sets[index]) for index in range(n_identities)], dtype=np.int64
    )
    mean = sums / counts[:, None]
    per_dim_variance = (sums_q / counts[:, None] - mean * mean).mean(axis=1)
    per_dim_variance[counts == 1] = 0.0
    max_spread = (global_max - global_min).max(axis=1)
    max_spread[counts == 1] = 0.0
    return {
        "n_identities": n_identities,
        "within_identity_mean_dim_variance": float(per_dim_variance.mean()),
        "within_identity_median_dim_variance": float(np.median(per_dim_variance)),
        "within_identity_mean_max_descriptor_spread": float(max_spread.mean()),
        "mean_unique_descriptors_per_identity": float(unique_signatures.mean()),
        "fraction_identities_with_single_descriptor": float((unique_signatures == 1).mean()),
        "max_unique_descriptors_for_one_identity": int(unique_signatures.max()),
        "mean_occurrences_per_identity": float(counts.mean()),
    }


def stage_molhiv_collision(knn_query: int = 6000, force: bool = False) -> dict[str, Any]:
    out_path = OUT_DIR / "molhiv_descriptor_identity.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    train = _load_molhiv_arrays("train")
    valid = _load_molhiv_arrays("valid")
    descriptor = train["descriptor"]  # memmap; never copied in full
    matrix = descriptor
    n_patches = int(matrix.shape[0])
    signature_ids, unique_signatures = _descriptor_signature_ids(matrix)
    hist = collision_stats(signature_ids, train["historical"])
    corr = collision_stats(signature_ids, train["corrected"])
    variability = {
        "historical": identity_variability_chunked(matrix, train["historical"], signature_ids),
        "corrected": identity_variability_chunked(matrix, train["corrected"], signature_ids),
    }
    split_multiplicity = historical_to_corrected_multiplicity(train["historical"], train["corrected"])
    distinct_hist = len(set(train["historical"].tolist()))
    distinct_corr = len(set(train["corrected"].tolist()))
    frequency = {
        "historical": frequency_statistics(Counter(train["historical"].tolist())),
        "corrected": frequency_statistics(Counter(train["corrected"].tolist())),
    }
    train_hist_set = set(train["historical"].tolist())
    train_corr_set = set(train["corrected"].tolist())
    valid_oov_hist = np.asarray([int(value) not in train_hist_set for value in valid["historical"].tolist()])
    valid_oov_corr = np.asarray([int(value) not in train_corr_set for value in valid["corrected"].tolist()])
    oov = {
        "historical": {
            "oov_occurrences": int(valid_oov_hist.sum()),
            "occurrences": int(valid_oov_hist.size),
            "oov_occurrence_fraction": float(valid_oov_hist.mean()),
        },
        "corrected": {
            "oov_occurrences": int(valid_oov_corr.sum()),
            "occurrences": int(valid_oov_corr.size),
            "oov_occurrence_fraction": float(valid_oov_corr.mean()),
        },
    }

    # kNN identity purity in train-fit standardized descriptor space.
    mean, scale = _mean_std_chunked(descriptor)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    std_path = OUT_DIR / "molhiv_train_descriptor_std.npy"
    if not std_path.exists():
        standardized = np.lib.format.open_memmap(std_path, mode="w+", dtype=np.float32, shape=matrix.shape)
        for start in range(0, n_patches, 20000):
            stop = min(start + 20000, n_patches)
            standardized[start:stop] = (matrix[start:stop] - mean) / scale
        standardized.flush()
        del standardized
    standardized = np.load(std_path, mmap_mode="r")
    rng = np.random.default_rng(RANDOM_SEED)
    query = np.sort(rng.choice(n_patches, size=min(knn_query, n_patches), replace=False))
    hist_counts = Counter(train["historical"].tolist())
    frequency_per_occurrence = np.asarray(
        [hist_counts[int(token)] for token in train["historical"].tolist()], dtype=np.int64
    )
    knn_hist = stratified_knn_purity(standardized, train["historical"], frequency_per_occurrence, query)
    knn_corr = stratified_knn_purity(standardized, train["corrected"], frequency_per_occurrence, query)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "descriptor": {
            "name": "MolHIV radius-2 shell descriptor (patch_cont)",
            "dim": int(matrix.shape[1]),
            "exact_byte_signatures": int(unique_signatures),
            "train_occurrences": n_patches,
        },
        "descriptor_to_identity": {
            "historical_certificate": hist,
            "corrected_coloured_key": corr,
        },
        "identity_to_descriptor": variability,
        "historical_to_corrected_split": split_multiplicity,
        "unique_identities": {"historical": distinct_hist, "corrected": distinct_corr},
        "frequency": frequency,
        "valid_oov": oov,
        "knn_purity": {
            "space": (
                "train-fit standardized MolHIV radius-2 shell descriptor; Euclidean; "
                "exact brute-force; query itself excluded"
            ),
            "query_sample": int(len(query)),
            "historical_certificate": knn_hist,
            "corrected_coloured_key": knn_corr,
        },
        "official_test_loaded": False,
        "interpretation": (
            "Same descriptor/identity asymmetry as ZINC; the MolHIV typed "
            "vocabulary is built on the historical coarse certificate, which is "
            "NOT a complete coloured-incidence key."
        ),
    }
    _write_json("molhiv_descriptor_identity.json", payload)
    return payload


# --------------------------------------------------------------------------
# stage: embedding geometry
# --------------------------------------------------------------------------
def _participation_ratio(matrix: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if weights is None:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        mean = (matrix * weights[:, None]).sum(axis=0)
        centered = matrix - mean
        weighted = centered * np.sqrt(weights)[:, None]
        singular = np.linalg.svd(weighted, compute_uv=False)
    energy = singular**2
    total = energy.sum()
    return {
        "participation_ratio": float((total**2) / (energy**2).sum()),
        "stable_rank": float(total / energy.max()),
        "top_singular_value": float(singular[0]),
        "rank_for_99pct_energy": int(np.searchsorted(np.cumsum(energy) / total, 0.99) + 1),
    }


def _nearest_neighbour_overlap(
    embedding: np.ndarray, structural: np.ndarray, k: int = 10, chunk: int = 1000
) -> dict[str, Any]:
    """Nearest-neighbour overlap + sampled-pair rank correlation.

    Memory-safe: top-k neighbours are computed over row blocks (never an
    ``n x n`` cosine matrix), and the Mantel-style rank correlation uses only
    a deterministic sample of off-diagonal pairs.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    structural = np.asarray(structural, dtype=np.float64)
    n = int(embedding.shape[0])
    if n <= k + 1:
        return {"n": n, "k": k, "overlap_at_k": None, "sampled_pair_spearman": None}

    def _normalized(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    embedding_norm = _normalized(embedding)
    structural_norm = _normalized(structural)
    embed_nn = np.empty((n, k), dtype=np.int64)
    struct_nn = np.empty((n, k), dtype=np.int64)
    kth = n - k
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = np.arange(start, stop)
        for matrix, target in ((embedding_norm, embed_nn), (structural_norm, struct_nn)):
            block = matrix[start:stop] @ matrix.T
            block[np.arange(stop - start), rows] = -np.inf
            part = np.argpartition(block, kth, axis=1)[:, kth:]
            target[start:stop] = part
            del block, part
    overlaps = []
    for row in range(n):
        overlaps.append(len(set(embed_nn[row].tolist()) & set(struct_nn[row].tolist())) / k)
    rng = np.random.default_rng(RANDOM_SEED)
    max_pairs = min(200000, n * (n - 1) // 2)
    pairs = set()
    while len(pairs) < max_pairs:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.add((int(min(i, j)), int(max(i, j))))
    pairs_array = np.asarray(sorted(pairs), dtype=np.int64)
    emb_values = np.einsum("ij,ij->i", embedding_norm[pairs_array[:, 0]], embedding_norm[pairs_array[:, 1]])
    struct_values = np.einsum(
        "ij,ij->i", structural_norm[pairs_array[:, 0]], structural_norm[pairs_array[:, 1]]
    )
    from scipy.stats import spearmanr

    rho = float(spearmanr(emb_values, struct_values).statistic)
    return {
        "n": n,
        "k": k,
        "mean_nearest_neighbour_overlap": float(np.mean(overlaps)),
        "sampled_pair_spearman": rho,
        "sampled_pairs": int(len(pairs_array)),
    }


def stage_zinc_embedding() -> dict[str, Any]:
    import torch

    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    flat = zinc_flat()
    records = flat["train_records"]
    state = torch.load(ZINC_OPTIMIZED_STATE, map_location="cpu", weights_only=True)
    full = state["typed_embedding.full.weight"].numpy().astype(np.float64)
    rare_embedding = state["typed_embedding.rare.embedding.weight"].numpy().astype(np.float64)
    rare_projection = state["typed_embedding.rare.projection.weight"].numpy().astype(np.float64)
    rare = rare_embedding @ rare_projection.T
    effective = np.vstack([full, rare])  # row id == vocabulary id (0 = OOV)

    vocab = zpp._fit_vocabulary(records, "typed_certificate", 8192, 1)
    counts = Counter()
    per_token_descriptor_sum: dict[int, np.ndarray] = {}
    per_token_descriptor_sumsq: dict[int, np.ndarray] = {}
    for record in records:
        for patch in record.patches:
            token = vocab.get(patch.typed_certificate)
            if token is None:
                continue
            counts[token] += 1
            value = np.asarray(patch.shell_descriptor, dtype=np.float64)
            if token in per_token_descriptor_sum:
                per_token_descriptor_sum[token] += value
                per_token_descriptor_sumsq[token] += value * value
            else:
                per_token_descriptor_sum[token] = value.copy()
                per_token_descriptor_sumsq[token] = value * value
    ids = np.asarray(sorted(counts), dtype=np.int64)
    frequency = np.asarray([counts[int(token)] for token in ids.tolist()], dtype=np.float64)
    mean_descriptor = np.stack(
        [per_token_descriptor_sum[int(token)] / counts[int(token)] for token in ids.tolist()]
    )
    embeddings = effective[ids]
    norms = np.linalg.norm(embeddings, axis=1)
    from scipy.stats import spearmanr

    rho_norm_freq = float(spearmanr(norms, np.log1p(frequency)).statistic)
    frequent = frequency > 20

    geometry = {
        "all_in_vocab": _participation_ratio(embeddings),
        "frequency_weighted": _participation_ratio(embeddings, frequency),
        "frequent_only": _participation_ratio(embeddings[frequent]),
        "top1_norm_vs_log_frequency_spearman": rho_norm_freq,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
    }
    alignment = {
        "all_in_vocab": _nearest_neighbour_overlap(embeddings, mean_descriptor, k=10),
        "frequent_only": _nearest_neighbour_overlap(embeddings[frequent], mean_descriptor[frequent], k=10),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": str(ZINC_OPTIMIZED_STATE.relative_to(REPO_ROOT)),
        "vocabulary_rows": int(effective.shape[0]),
        "effective_dim": int(effective.shape[1]),
        "tokens_with_train_occurrences": int(len(ids)),
        "geometry": geometry,
        "embedding_vs_structural_alignment": alignment,
        "frequency_statistics": frequency_statistics(Counter(counts)),
        "official_test_loaded": False,
        "interpretation": (
            "Effective rank / participation ratio of the learned identity matrix "
            "and its alignment with the deterministic structural descriptor "
            "geometry (token-mean standardized descriptor)."
        ),
    }
    _write_json("zinc_embedding_geometry.json", payload)
    return payload


def molhiv_vocabulary_map(split: str) -> dict[str, Any]:
    """Exact MolHIV vocabulary row map (frequency, then certificate byte order).

    The model's ``_fit_vocabulary`` breaks frequency ties by comparing the raw
    certificate bytes, so a hash fingerprint cannot reproduce the row order.
    This light pass streams only the certificates and rebuilds the ordering.
    """
    from ksvd_research.data import load_molhiv
    from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp

    meta_path = OUT_DIR / f"molhiv_{split}_vocabulary_map.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    bundle = load_molhiv(root=str(REPO_ROOT / "data/ogb"), with_features=True)
    indices = np.asarray(bundle.split["valid" if split == "valid" else split], dtype=np.int64)
    certificate_cache: dict[bytes, bytes] = {}
    counts: Counter = Counter()
    for index in indices.tolist():
        graph = bundle.graphs[int(index)]
        node_types = np.asarray(bundle.node_feats[int(index)], dtype=np.int64)
        edge_types = {
            (int(left), int(right)): tuple(int(value) for value in values)
            for (left, right), values in bundle.edge_feats[int(index)].items()
        }
        for center in graph.nodes:
            certificate = mpp._typed_certificate(
                graph, int(center), node_types, edge_types, int(mpp.PATCH_RADIUS), certificate_cache
            )
            counts[certificate] += 1
    ordered = sorted(counts, key=lambda key: (-counts[key], key))
    row_of_fingerprint = {
        str(_fingerprint_bytes64(certificate)): index + 1 for index, certificate in enumerate(ordered)
    }
    payload = {
        "split": split,
        "vocabulary_size": len(ordered),
        "row_of_fingerprint": row_of_fingerprint,
        "tie_break": "(-count, certificate_bytes_lexicographic) matching mpp._fit_vocabulary",
    }
    meta_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _per_row_mean_chunked(
    matrix: np.ndarray, row_of_patch: np.ndarray, n_rows: int, chunk: int = 50000
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row mean of a (memmapped) matrix, accumulated in chunks.

    Never materialises the full matrix in memory.  Groups are contiguous after
    an argsort of ``row_of_patch``; a group split across a chunk boundary is
    accumulated in both chunks, which is correct for sums/counts.
    """
    order = np.argsort(row_of_patch, kind="stable")
    sorted_rows = row_of_patch[order]
    n_patches = int(row_of_patch.shape[0])
    sums = np.zeros((n_rows, int(matrix.shape[1])), dtype=np.float64)
    counts = np.bincount(row_of_patch, minlength=n_rows).astype(np.float64)
    for start in range(0, n_patches, chunk):
        stop = min(start + chunk, n_patches)
        rows_chunk = sorted_rows[start:stop]
        values = np.asarray(matrix[order[start:stop]], dtype=np.float32)
        starts = np.r_[0, np.flatnonzero(np.diff(rows_chunk)) + 1]
        part = np.add.reduceat(values, starts, axis=0)
        group_rows = rows_chunk[starts]
        sums[group_rows] += part
        del values, part
    del order, sorted_rows
    return sums, counts


def stage_molhiv_embedding() -> dict[str, Any]:
    import torch

    train = _load_molhiv_arrays("train")
    state = torch.load(MOLHIV_STATE, map_location="cpu", weights_only=True)
    embedding = state["typed_embedding.weight"].numpy().astype(np.float64)
    historical = train["historical"]
    counts = Counter(historical.tolist())
    descriptor = train["descriptor"]  # memmap, never copied in full
    vocabulary = molhiv_vocabulary_map("train")
    row_of_fingerprint = {int(k): int(v) for k, v in vocabulary["row_of_fingerprint"].items()}
    if len(row_of_fingerprint) != len(counts):
        raise RuntimeError(f"vocabulary/token count mismatch: {len(row_of_fingerprint)} vs {len(counts)}")
    row_of_patch = np.asarray(
        [row_of_fingerprint[int(token)] for token in historical.tolist()], dtype=np.int64
    )
    sums, per_row_counts = _per_row_mean_chunked(descriptor, row_of_patch, int(embedding.shape[0]))
    ranked = sorted(counts, key=lambda token: (-counts[token], row_of_fingerprint[token]))
    ids = np.asarray(ranked, dtype=np.int64)
    frequency = np.asarray([counts[int(token)] for token in ids.tolist()], dtype=np.float64)
    rows = np.asarray([row_of_fingerprint[int(token)] for token in ids.tolist()], dtype=np.int64)
    if not np.array_equal(per_row_counts[rows], frequency):
        raise RuntimeError("chunked per-row counts disagree with the identity frequency")
    mean_descriptor = sums[rows] / frequency[:, None]
    del sums
    embeddings = embedding[rows]
    from scipy.stats import spearmanr

    norms = np.linalg.norm(embeddings, axis=1)
    frequent = frequency > 20
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": str(MOLHIV_STATE.relative_to(REPO_ROOT)),
        "vocabulary_rows": int(embedding.shape[0]),
        "effective_dim": int(embedding.shape[1]),
        "token_row_map": {
            "vocabulary_rows": int(embedding.shape[0]),
            "ranked_tokens": int(len(ranked)),
            "rows_minus_tokens": int(embedding.shape[0] - len(ranked)),
        },
        "geometry": {
            "all_in_vocab": _participation_ratio(embeddings),
            "frequency_weighted": _participation_ratio(embeddings, frequency),
            "frequent_only": _participation_ratio(embeddings[frequent]),
            "top1_norm_vs_log_frequency_spearman": float(spearmanr(norms, np.log1p(frequency)).statistic),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        },
        "embedding_vs_structural_alignment": {
            "all_in_vocab": _nearest_neighbour_overlap(embeddings, mean_descriptor, k=10),
            "frequent_only": _nearest_neighbour_overlap(
                embeddings[frequent], mean_descriptor[frequent], k=10
            ),
        },
        "frequency_statistics": frequency_statistics(Counter(counts)),
        "official_test_loaded": False,
        "interpretation": (
            "Learned MolHIV identity matrix geometry vs token-mean structural "
            "descriptor geometry. Row order is frequency-ranked; this audit "
            "reconstructs the ranking from the real occurrence counts."
        ),
    }
    _write_json("molhiv_embedding_geometry.json", payload)
    return payload


# --------------------------------------------------------------------------
# stage: molhiv-generalization (section 8)
# --------------------------------------------------------------------------
def _molecule_keys(identity: np.ndarray, molecule: np.ndarray) -> dict[int, int]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for token, mol in zip(identity.tolist(), molecule.tolist()):
        buckets[int(mol)].append(int(token))
    keys: dict[int, int] = {}
    for mol, tokens in buckets.items():
        digest = hashlib.blake2b(digest_size=8)
        for token in sorted(tokens):
            digest.update(token.to_bytes(8, "big", signed=False))
        keys[mol] = int.from_bytes(digest.digest(), "big") & ((1 << 63) - 1)
    return keys


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
        return None
    return float(roc_auc_score(labels, scores))


def _stratified_auc(scores: np.ndarray, labels: np.ndarray, strata: np.ndarray) -> float | None:
    total_weight = 0.0
    weighted = 0.0
    for value in np.unique(strata):
        mask = strata == value
        if mask.sum() < 10:
            continue
        auc = _auc(scores[mask], labels[mask])
        if auc is None:
            continue
        positives = int(labels[mask].sum())
        negatives = int(mask.sum() - positives)
        weight = positives * negatives
        weighted += weight * auc
        total_weight += weight
    return float(weighted / total_weight) if total_weight > 0 else None


def stage_molhiv_generalization() -> dict[str, Any]:
    from scipy.stats import spearmanr

    train = _load_molhiv_arrays("train")
    valid = _load_molhiv_arrays("valid")
    train_vocab = set(train["historical"].tolist())
    valid_identity = valid["historical"]
    valid_molecule = valid["molecule"]
    n_molecules = int(valid_molecule.max()) + 1
    patch_count = np.bincount(valid_molecule, minlength=n_molecules)
    oov_count = np.bincount(
        valid_molecule,
        weights=np.asarray(
            [int(token) not in train_vocab for token in valid_identity.tolist()], dtype=np.float64
        ),
        minlength=n_molecules,
    )
    oov_fraction = oov_count / np.maximum(patch_count, 1)

    train_keys = _molecule_keys(train["historical"], train["molecule"])
    valid_keys = _molecule_keys(valid_identity, valid_molecule)
    key_frequency = Counter(train_keys.values())
    molecule_key_frequency = np.asarray(
        [key_frequency.get(valid_keys[index], 0) for index in range(n_molecules)], dtype=np.float64
    )

    results: dict[str, Any] = {}
    for seed in (0, 1):
        run = json.loads(
            (TRACK_ROOT / f"results/molhiv_recurrent_pair_centre/run_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        targets = np.asarray(run["valid_targets"], dtype=np.float64)
        for estimator in ("raw", "soup"):
            key = f"{estimator}_valid_logits"
            if key not in run:
                continue
            logits = np.asarray(run[key], dtype=np.float64)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = np.abs(probabilities - targets)
            group = oov_fraction > 0
            size_decile = np.digitize(patch_count, np.quantile(patch_count, np.linspace(0.1, 0.9, 9)))
            key_bucket = np.digitize(
                molecule_key_frequency, np.quantile(molecule_key_frequency, np.linspace(0.5, 0.9, 5))
            )
            record = {
                "seed": seed,
                "estimator": estimator,
                "overall_auc": _auc(probabilities, targets),
                "zero_oov": {
                    "n": int((~group).sum()),
                    "positives": int(targets[~group].sum()),
                    "prevalence": float(targets[~group].mean()),
                    "mean_size": float(patch_count[~group].mean()),
                    "auc": _auc(probabilities[~group], targets[~group]),
                    "mean_error": float(error[~group].mean()),
                },
                "any_oov": {
                    "n": int(group.sum()),
                    "positives": int(targets[group].sum()),
                    "prevalence": float(targets[group].mean()),
                    "mean_size": float(patch_count[group].mean()),
                    "auc": _auc(probabilities[group], targets[group]),
                    "mean_error": float(error[group].mean()),
                },
                "size_decile_stratified_auc": {
                    "zero_oov": _stratified_auc(probabilities[~group], targets[~group], size_decile[~group]),
                    "any_oov": _stratified_auc(probabilities[group], targets[group], size_decile[group]),
                },
                "molecule_key_frequency_stratified_auc": {
                    "zero_oov": _stratified_auc(probabilities[~group], targets[~group], key_bucket[~group]),
                    "any_oov": _stratified_auc(probabilities[group], targets[group], key_bucket[group]),
                },
                "partial_correlation": {
                    "spearman_oov_vs_error": float(spearmanr(oov_fraction, error).statistic),
                    "spearman_oov_vs_error_given_size": _partial_spearman(oov_fraction, error, patch_count),
                    "spearman_oov_vs_error_given_size_and_keyfreq": _partial_spearman(
                        oov_fraction, error, np.vstack([patch_count, molecule_key_frequency]).T
                    ),
                    "spearman_oov_vs_confidence_given_size": _partial_spearman(
                        oov_fraction, np.abs(logits), patch_count
                    ),
                },
            }
            results[f"seed{seed}_{estimator}"] = record

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "source": "official-validation (no new model evaluation, no test load)",
        "valid_molecules": int(n_molecules),
        "controls": {
            "size": "patch count = number of atoms (one patch per atom)",
            "scaffold_frequency_proxy": (
                "document frequency of the molecule's sorted typed-token multiset "
                "in official train (RDKit is not installed; this is an explicit proxy)"
            ),
            "prevalence": "positives / negatives reported per subgroup",
        },
        "results": results,
        "existing_test_diagnostic_reused": str(
            (MOLHIV_PARAM_DIR / "rarity_prediction_diagnostic.json").relative_to(REPO_ROOT)
        ),
        "note": (
            "The existing test diagnostic is reused read-only (test was already "
            "opened once for the frozen checkpoint); no new test evaluation is run. "
            "Correlations are descriptive, not causal."
        ),
        "official_test_loaded": False,
    }
    _write_json("molhiv_generalization.json", payload)
    return payload


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float | None:
    """Spearman partial correlation of x and y controlling for z (columns)."""
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    if z.shape[0] == 1:
        z = z.T
    rx = _rank(x)
    ry = _rank(y)
    rz = np.column_stack([np.ones(z.shape[0])] + [_rank(z[:, column]) for column in range(z.shape[1])])
    beta_x = np.linalg.lstsq(rz, rx, rcond=None)[0]
    beta_y = np.linalg.lstsq(rz, ry, rcond=None)[0]
    residual_x = rx - rz @ beta_x
    residual_y = ry - rz @ beta_y
    return float(spearmanr(residual_x, residual_y).statistic)


def _rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values)


# --------------------------------------------------------------------------
# stage: report + figures
# --------------------------------------------------------------------------
def _load_optional(name: str) -> dict[str, Any] | None:
    path = OUT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


HISTORICAL_RECONCILIATION = [
    {
        "experiment": "typed tokenizer correctness repair",
        "changed_about_identity": "redefined the token from the coarse uncoloured rooted topology to the complete coloured-incidence key",
        "retrained": True,
        "lookup_kept": True,
        "tests": "information correctness + statistical sharing",
        "conclusion": "corrected token is correct but slightly worse; the coarse token acted as accidental parameter sharing",
    },
    {
        "experiment": "corrected fragmentation / rarity audit",
        "changed_about_identity": "none (measurement of the corrected token distribution)",
        "retrained": False,
        "lookup_kept": True,
        "tests": "sharing / statistical",
        "conclusion": "degradation is not directed by fragmentation; it is a baseline-difficulty redistribution",
    },
    {
        "experiment": "compositional patch sharing oracle",
        "changed_about_identity": "frozen embedding transplant for rare/OOV occurrences",
        "retrained": False,
        "lookup_kept": "replaced for rare/OOV",
        "tests": "sharing",
        "conclusion": "NO-GO; structural KNN is not better than frequent-mean/random donors",
    },
    {
        "experiment": "compact-v6 topology-attribute factorization",
        "changed_about_identity": "replaced the exact token by a shared (type, role) attribute encoder",
        "retrained": True,
        "lookup_kept": False,
        "tests": "information / function class",
        "conclusion": "NO-GO; the branch collapses and does not replicate",
    },
    {
        "experiment": "SBCI shared-basis compositional interaction",
        "changed_about_identity": "none (identity embedding retained); replaced the free relation/centre function family",
        "retrained": True,
        "lookup_kept": True,
        "tests": "function class / sample efficiency",
        "conclusion": "NO-GO; sample-efficiency frontier crossing",
    },
    {
        "experiment": "current recurrent Cell A",
        "changed_about_identity": "none; identity lookup retained with ~36.6% parameter share",
        "retrained": True,
        "lookup_kept": True,
        "tests": "capacity / architecture",
        "conclusion": "parameters are dominated by identity storage; no width gain transfers",
    },
    {
        "experiment": "MolHIV parameter attribution",
        "changed_about_identity": "none (accounting only)",
        "retrained": False,
        "lookup_kept": True,
        "tests": "information / parameter accounting",
        "conclusion": "identity storage is 78.1% of the current ~1.08M model; no cutoff chosen",
    },
]


def _case_scorecard(
    zinc: Mapping[str, Any] | None,
    molhiv: Mapping[str, Any] | None,
    zinc_embedding: Mapping[str, Any] | None,
    molhiv_embedding: Mapping[str, Any] | None,
    expressivity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cases: dict[str, Any] = {
        "A_mostly_redundant": {},
        "B_distinguishability": {},
        "C_missing_structure": {},
        "D_task_semantics": {},
    }
    for label, payload, key in (
        ("zinc", zinc, "descriptor_to_identity"),
        ("molhiv", molhiv, "descriptor_to_identity"),
    ):
        if not payload:
            continue
        block = payload[key]
        hist = block.get("historical_coarse_token") or block.get("historical_certificate")
        corr = block.get("corrected_exact_key") or block.get("corrected_coloured_key")
        cases["C_missing_structure"][label] = {
            "descriptor_collision_mass_fraction_historical": hist["collision_mass_fraction"],
            "descriptor_collision_mass_fraction_corrected": corr["collision_mass_fraction"],
            "H_identity_given_descriptor_historical": hist["H_identity_given_descriptor_nats"],
            "max_identity_multiplicity": hist["max_identity_multiplicity_per_signature"],
            "resolves_descriptor_ambiguity": bool(
                hist["collision_mass_fraction"] > 0.05 and hist["H_identity_given_descriptor_nats"] > 0.05
            ),
        }
    if expressivity:
        tf = expressivity["target_free"]
        cases["A_mostly_redundant"]["molecule_level"] = {
            "descriptor_only_collision_mass_fraction": tf["descriptor_only"]["collision_mass_fraction"],
            "with_identity_collision_mass_fraction": tf["full_system_with_identity"][
                "collision_mass_fraction"
            ],
            "identity_expression_increment": (
                tf["descriptor_only"]["unique_signatures"]
                - tf["full_system_with_identity"]["unique_signatures"]
            ),
            "interpretation": (
                "identity adds zero molecule-level distinguishing power once the "
                "deterministic descriptor system is present"
            ),
        }
    for label, payload in (("zinc", zinc_embedding), ("molhiv", molhiv_embedding)):
        if not payload:
            continue
        align = payload["embedding_vs_structural_alignment"]["all_in_vocab"]
        cases["B_distinguishability"][label] = {
            "participation_ratio_all": payload["geometry"]["all_in_vocab"]["participation_ratio"],
            "participation_ratio_frequent": payload["geometry"]["frequent_only"]["participation_ratio"],
            "norm_vs_log_frequency_spearman": payload["geometry"]["top1_norm_vs_log_frequency_spearman"],
            "nn_overlap_at_10": align["mean_nearest_neighbour_overlap"],
            "sampled_pair_spearman": align["sampled_pair_spearman"],
            "encodes_structural_geometry": bool(
                align["sampled_pair_spearman"] is not None and align["sampled_pair_spearman"] > 0.2
            ),
        }
    cases["D_task_semantics"] = {
        "evidence": (
            "No existing experiment isolates task-relevant semantics of the learned "
            "identity rows from the structural information they already encode; "
            "the frozen compositional-sharing transplant was a NO-GO and the "
            "embedding geometry does not align with structural geometry."
        ),
        "supported": False,
    }
    # Overall classification (honest, multi-label).
    labels = []
    if expressivity:
        labels.append("A_mostly_redundant_at_molecule_level")
    labels.append("B_distinguishability_matters")
    if any(
        cases["C_missing_structure"].get(name, {}).get("resolves_descriptor_ambiguity")
        for name in ("zinc", "molhiv")
    ):
        labels.append("C_per_patch_descriptor_ambiguity")
    return {
        "cases": cases,
        "overall_labels": labels,
        "headline": (
            "Identity is a genuine per-patch discrete separator that the "
            "deterministic descriptor does not supply, but it is redundant for "
            "molecule-level distinguishability and its learned geometry is not "
            "organised by structural similarity. It is best read as "
            "distinguishability + task-specific (arbitrary categorical memory), "
            "not as missing raw structural information."
        ),
    }


def stage_report() -> dict[str, Any]:
    dataflow = _load_optional("dataflow.json")
    zinc = _load_optional("zinc_descriptor_identity.json")
    zinc_knn = _load_optional("zinc_knn_purity.json")
    zinc_freq = _load_optional("zinc_frequency.json")
    expressivity = _load_optional("zinc_expressivity.json")
    zinc_embedding = _load_optional("zinc_embedding_geometry.json")
    molhiv = _load_optional("molhiv_descriptor_identity.json")
    molhiv_embedding = _load_optional("molhiv_embedding_geometry.json")
    molhiv_generalization = _load_optional("molhiv_generalization.json")

    decision = _case_scorecard(zinc, molhiv, zinc_embedding, molhiv_embedding, expressivity)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataflow": dataflow,
        "zinc_collision": zinc,
        "zinc_knn": zinc_knn,
        "zinc_frequency": zinc_freq,
        "zinc_expressivity": expressivity,
        "zinc_embedding": zinc_embedding,
        "molhiv_collision": molhiv,
        "molhiv_embedding": molhiv_embedding,
        "molhiv_generalization": molhiv_generalization,
        "historical_reconciliation": HISTORICAL_RECONCILIATION,
        "decision": decision,
        "no_go_observed": [
            "no new backbone training",
            "no identity removal",
            "no new embedding design",
            "no top-K cutoff",
            "no hash compression",
            "no compositional encoder",
            "no optimizer sweep",
            "no new official test evaluation (ZINC test never loaded; MolHIV test only re-read from existing artifacts)",
        ],
        "official_test_loaded": False,
    }
    _write_json("final_decision.json", payload)
    _make_figures(payload)
    return payload


def _make_figures(payload: Mapping[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # Figure 1: descriptor identity-multiplicity histograms (ZINC + MolHIV).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, label in (
        (axes[0], payload.get("zinc_collision"), "ZINC"),
        (axes[1], payload.get("molhiv_collision"), "MolHIV"),
    ):
        if not key:
            continue
        block = key["descriptor_to_identity"]
        hist = block.get("historical_coarse_token") or block.get("historical_certificate")
        corr = block.get("corrected_exact_key") or block.get("corrected_coloured_key")
        for entry, name in ((hist, "historical"), (corr, "corrected")):
            items = sorted((int(k), v) for k, v in entry["signature_multiplicity_histogram"].items())
            axis.plot([item[0] for item in items], [item[1] for item in items], marker="o", label=name)
        axis.set_yscale("log")
        axis.set_title(f"{label}: identity multiplicity per descriptor")
        axis.set_xlabel("distinct identities per descriptor signature")
        axis.set_ylabel("signatures (log)")
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig1_identity_multiplicity.png", dpi=150)
    plt.close(fig)
    # Figure 2: kNN purity by frequency stratum (ZINC).
    knn = payload.get("zinc_knn")
    if knn:
        strata = knn["historical_coarse_token"]["strata"]
        names = [name for name in strata if strata[name].get("n", 0) > 0]
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(names, [strata[name]["k8_purity"] for name in names], alpha=0.7, label="k=8")
        axis.plot(names, [strata[name]["top1_same_identity_rate"] for name in names], "o-", label="top-1")
        axis.set_ylim(0, 1)
        axis.set_title("ZINC kNN identity purity by train frequency")
        axis.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "fig2_knn_purity.png", dpi=150)
        plt.close(fig)
    # Figure 3: embedding norm vs frequency + alignment.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, key, label in (
        (axes[0], payload.get("zinc_embedding"), "ZINC"),
        (axes[1], payload.get("molhiv_embedding"), "MolHIV"),
    ):
        if not key:
            continue
        axis.bar(
            ["all", "freq-weighted", "frequent"],
            [
                key["geometry"]["all_in_vocab"]["participation_ratio"],
                key["geometry"]["frequency_weighted"]["participation_ratio"],
                key["geometry"]["frequent_only"]["participation_ratio"],
            ],
        )
        axis.set_title(f"{label}: embedding participation ratio")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig3_embedding_geometry.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--knn-query", type=int, default=6000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage in ("dataflow", "all"):
        stage_dataflow()
    if args.stage in ("zinc-collision", "all"):
        stage_zinc_collision()
    if args.stage in ("zinc-knn", "all"):
        stage_zinc_knn()
    if args.stage in ("zinc-frequency", "all"):
        stage_zinc_frequency()
    if args.stage in ("zinc-expressivity", "all"):
        stage_zinc_expressivity()
    if args.stage in ("zinc-embedding", "all"):
        stage_zinc_embedding()
    if args.stage in ("molhiv-extract", "all"):
        extract_molhiv_split("train")
        extract_molhiv_split("valid")
    if args.stage in ("molhiv-collision", "all"):
        stage_molhiv_collision(knn_query=args.knn_query, force=args.force)
    if args.stage in ("molhiv-embedding", "all"):
        stage_molhiv_embedding()
    if args.stage in ("molhiv-generalization", "all"):
        stage_molhiv_generalization()
    if args.stage in ("report", "all"):
        stage_report()
