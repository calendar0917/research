"""MolHIV current-model parameter attribution + vocabulary audit (zero training).

Question
--------
The current OGBG-MolHIV recurrent pair--centre port
(``molhiv_recurrent_pair_centre_v1``) is reported at ~1.08M parameters.  Before
any compact redesign, this module answers *exactly* where those parameters are,
how much is dataset-dependent identity storage, and what the real vocabulary
looks like under the actual pipeline.

This is an **audit only**: no architecture search, no training, no tokenizer
change, no vocabulary cutoff decision, and no embedding compression.  The
official MolHIV test split was already opened once in the frozen
``molhiv_recurrent_pair_centre_v1`` run; here it is used **read-only** for
vocabulary diagnostics.

Stages
------
``params``     named-parameter classification, largest tensors, fixed vs
               dataset-dependent parameters, feature-block attribution.
``vocabulary`` real-pipeline vocabulary statistics on train/valid/test,
               frequency distribution, OOV and rarity accounting.
``growth``     parameter growth law + pure top-K accounting (no model built).
``compare``    ZINC cell A / H96 identity-vs-fixed parameter structure for the
               cross-track comparison.
``diagnostic`` optional zero-training rarity/OOV x prediction subgroup
               diagnostic, only if the frozen raw checkpoints are present.
``report``     combine all written artifacts.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.molhiv_parameter_attribution <stage>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

REPO_ROOT = mpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/molhiv_parameter_attribution"

PROTOCOL_VERSION = "molhiv_parameter_attribution_v1"

SPLITS = ("train", "valid", "test")

# Architecture constants of the frozen MolHIV port.
TOKEN_WIDTH = int(rpc.TOKEN_WIDTH)
PARENT_WIDTH = max(int(rpc.TOKEN_WIDTH // 2), 1)
H_DIM = int(rpc.H_DIM)
Q_DIM = int(rpc.Q_DIM)

# ---- parameter classification -------------------------------------------------

# Every named parameter is assigned to exactly one category by first match.
CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("typed_embedding.", "identity_storage.certificate_patch_token"),
    ("parent_embedding.", "identity_storage.radius1_parent_certificate"),
    ("distance_gate.", "learned_embedding_tables.fixed.distance_bucket"),
    ("pair_projection.", "pair_projection"),
    ("relation_encoder.", "relation_related_modules.relation_encoder"),
    ("pair_encoder.", "pair_encoder"),
    ("center_update.", "center_update"),
    ("global_encoder.", "global_encoder"),
    ("patch_encoder.", "patch_encoder"),
    ("head.0.", "readout_projection"),
    ("head.1.", "prediction_head.head_layer_norm"),
    ("head.4.", "prediction_head.head_hidden"),
    ("head.6.", "prediction_head.head_output"),
)

# Categories the task asks for that do not exist as dedicated modules in this
# architecture.  Reported explicitly with an explanation.
ABSENT_CATEGORIES: dict[str, str] = {
    "atom_node_encoding_table": (
        "no dedicated atom embedding table: atom identity is a continuous "
        "one-hot shell histogram inside patch_cont (consumed by the patch_encoder "
        "first layer) and inside the global_context atom histogram (consumed by "
        "the global_encoder first layer)"
    ),
    "bond_edge_encoding_table": (
        "no dedicated bond embedding table: bond identity is a continuous "
        "one-hot shell histogram inside patch_cont and inside the global_context "
        "and pair-relation descriptors"
    ),
    "topology_encoder": (
        "the MolHIV patch-path port has no topology branch; the topology features "
        "and topology encoder exist only in the ZINC hinge variants"
    ),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _category(name: str) -> str:
    for prefix, category in CATEGORY_RULES:
        if name.startswith(prefix):
            return category
    return "other_uncategorized"


# ---------------------------------------------------------------------------
# data / vocabulary (real pipeline)
# ---------------------------------------------------------------------------


def load_records(split: str) -> list[Any]:
    records, _meta = rpc._load_records(split)
    return records


def fit_train_vocabularies() -> dict[str, Any]:
    """Exactly the frozen training-time vocabulary fit (train split only)."""
    return fit_vocabularies_from(load_records("train"))


def fit_vocabularies_from(records: Sequence[Any]) -> dict[str, Any]:
    typed = mpp._fit_vocabulary(records, "typed_certificate", 32768, 1)
    parent = mpp._fit_vocabulary(records, "parent_certificate", 4096, 1)
    return {
        "typed_vocabulary": typed,
        "parent_vocabulary": parent,
        "typed_vocabulary_size": len(typed),
        "parent_vocabulary_size": len(parent),
        "typed_vocabulary_size_with_oov": len(typed) + 1,
        "parent_vocabulary_size_with_oov": len(parent) + 1,
        "caps": {
            "max_typed_tokens": 32768,
            "max_parent_tokens": 4096,
            "minimum_frequency": 1,
        },
    }


def build_actual_model(seed: int = 0) -> tuple[torch.nn.Module, dict[str, Any]]:
    vocab = fit_train_vocabularies()
    model = rpc.build_model(
        int(vocab["typed_vocabulary_size_with_oov"]),
        int(vocab["parent_vocabulary_size_with_oov"]),
        int(seed),
    )
    return model, vocab


# ---------------------------------------------------------------------------
# params -- exact classification
# ---------------------------------------------------------------------------


def _classify(model: torch.nn.Module) -> dict[str, Any]:
    """Named-parameter classification that sums exactly to the model total."""
    total = int(sum(p.numel() for p in model.parameters()))
    named_total = int(sum(p.numel() for name, p in model.named_parameters()))

    categories: dict[str, dict[str, Any]] = {}
    other: list[str] = []
    for name, parameter in model.named_parameters():
        category = _category(name)
        if category == "other_uncategorized":
            other.append(name)
        bucket = categories.setdefault(category, {"parameter_count": 0, "tensors": []})
        bucket["parameter_count"] += int(parameter.numel())
        bucket["tensors"].append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
            }
        )

    for bucket in categories.values():
        bucket["tensors"].sort(key=lambda row: (-row["numel"], row["name"]))
        bucket["fraction_of_total"] = bucket["parameter_count"] / total

    top20 = sorted(
        (
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "fraction_of_total": int(parameter.numel()) / total,
            }
            for name, parameter in model.named_parameters()
        ),
        key=lambda row: (-row["numel"], row["name"]),
    )[:20]

    identity = int(
        model.typed_embedding.weight.numel() + model.parent_embedding.weight.numel()
    )
    return {
        "total_params": total,
        "named_parameters_total": named_total,
        "named_parameters_match_parameters": named_total == total,
        "category_breakdown": categories,
        "category_sum": int(sum(b["parameter_count"] for b in categories.values())),
        "category_sum_matches_total": int(
            sum(b["parameter_count"] for b in categories.values())
        )
        == total,
        "top20_tensors": top20,
        "identity_storage_params": identity,
        "identity_storage_fraction": identity / total,
        "fixed_size_params": total - identity,
        "fixed_size_fraction": (total - identity) / total,
        "other_uncategorized_tensors": other,
    }


def params() -> dict[str, Any]:
    model, vocab = build_actual_model()
    classification = _classify(model)
    total = int(classification["total_params"])
    identity = int(classification["identity_storage_params"])
    fixed = int(classification["fixed_size_params"])

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "architecture": {
            "token_width": TOKEN_WIDTH,
            "parent_width": PARENT_WIDTH,
            "h_dim": H_DIM,
            "q_dim": Q_DIM,
            "recurrence_rounds": int(rpc.RECURRENCE_ROUNDS),
            "center_context_hidden": int(rpc.CENTER_CONTEXT_HIDDEN),
            "shell_width": int(mpp.SHELL_WIDTH),
            "global_width": int(mpp.GLOBAL_WIDTH),
            "relation_width": int(mpp.RELATION_WIDTH),
            "distance_buckets": int(mpp.DISTANCE_BUCKETS),
        },
        "vocabulary_used": {
            "typed_vocabulary_size_with_oov": int(
                vocab["typed_vocabulary_size_with_oov"]
            ),
            "parent_vocabulary_size_with_oov": int(
                vocab["parent_vocabulary_size_with_oov"]
            ),
        },
        "total_params": total,
        "named_parameters_total": classification["named_parameters_total"],
        "named_parameters_match_parameters": classification[
            "named_parameters_match_parameters"
        ],
        "category_breakdown": classification["category_breakdown"],
        "category_sum": classification["category_sum"],
        "category_sum_matches_total": classification["category_sum_matches_total"],
        "top20_tensors": classification["top20_tensors"],
        "identity_storage_params": identity,
        "identity_storage_fraction": identity / total,
        "fixed_size_params": fixed,
        "fixed_size_fraction": fixed / total,
        "dataset_dependent_params": identity,
        "dataset_dependent_fraction": identity / total,
        "absent_categories": ABSENT_CATEGORIES,
        "other_uncategorized_tensors": classification["other_uncategorized_tensors"],
        "official_test_loaded": False,
    }
    if (
        not payload["named_parameters_match_parameters"]
        or not payload["category_sum_matches_total"]
    ):
        raise RuntimeError("parameter accounting does not sum to the model total")

    payload["feature_block_attribution"] = _feature_block_attribution(model)
    _write_json(RESULTS_DIR / "parameter_breakdown.json", payload)
    return payload


def _feature_block_attribution(model: torch.nn.Module) -> dict[str, Any]:
    """Attribute the first-layer input columns of the two encoders.

    Secondary view only: the primary module-level table already assigns whole
    encoder tensors; the two views must never be added together.
    """
    atom_width = int(mpp.ATOM_WIDTH)
    bond_width = int(mpp.BOND_WIDTH)
    radius = int(mpp.PATCH_RADIUS)
    atom_shell_cols = (radius + 1) * atom_width
    bond_shell_cols = len(mpp.SHELL_PAIRS) * bond_width
    shell = int(mpp.SHELL_WIDTH)

    # shell = [atom_shell | bond_shell | root_atom | incident_bonds | scalars]
    atom_cols = list(range(0, atom_shell_cols)) + list(
        range(atom_shell_cols + bond_shell_cols, atom_shell_cols + bond_shell_cols + atom_width)
    )
    bond_cols = list(range(atom_shell_cols, atom_shell_cols + bond_shell_cols)) + list(
        range(
            atom_shell_cols + bond_shell_cols + atom_width,
            atom_shell_cols + bond_shell_cols + atom_width + bond_width,
        )
    )
    scalar_cols = list(
        range(atom_shell_cols + bond_shell_cols + atom_width + bond_width, shell)
    )
    typed_cols = list(range(shell, shell + TOKEN_WIDTH))
    parent_cols = list(range(shell + TOKEN_WIDTH, shell + TOKEN_WIDTH + PARENT_WIDTH))

    patch_first = model.patch_encoder.layers[0].weight
    hidden = int(patch_first.shape[0])

    def _weight_numel(block_cols: Sequence[int]) -> int:
        return int(hidden * len(block_cols))

    patch_blocks = {
        "atom_node_shell_columns": _weight_numel(atom_cols),
        "bond_edge_shell_columns": _weight_numel(bond_cols),
        "shell_scalar_columns": _weight_numel(scalar_cols),
        "typed_token_columns": _weight_numel(typed_cols),
        "parent_token_columns": _weight_numel(parent_cols),
        "layer_bias": int(patch_first.shape[0]),
        "second_layer": int(
            sum(
                p.numel()
                for name, p in model.patch_encoder.named_parameters()
                if not name.startswith("layers.0.")
            )
        ),
    }

    global_first = model.global_encoder.layers[0].weight
    short = int(mpp.GLOBAL_SHORT_WIDTH)
    long_ = int(mpp.GLOBAL_LONG_WIDTH)
    global_blocks = {
        "global_short_columns": int(global_first.shape[0] * short),
        "global_long_columns": int(global_first.shape[0] * long_),
        "atom_histogram_columns": int(global_first.shape[0] * atom_width),
        "bond_histogram_columns": int(global_first.shape[0] * bond_width),
        "layer_bias": int(global_first.shape[0]),
        "second_layer": int(
            sum(
                p.numel()
                for name, p in model.global_encoder.named_parameters()
                if not name.startswith("layers.0.")
            )
        ),
    }

    return {
        "note": (
            "secondary view only; the primary module-level ownership assigns the "
            "whole patch_encoder/global_encoder tensors"
        ),
        "patch_encoder_first_layer_blocks": patch_blocks,
        "patch_encoder_first_layer_block_sum": int(sum(patch_blocks.values())),
        "patch_encoder_total": int(
            sum(p.numel() for p in model.patch_encoder.parameters())
        ),
        "global_encoder_first_layer_blocks": global_blocks,
        "global_encoder_first_layer_block_sum": int(sum(global_blocks.values())),
        "global_encoder_total": int(
            sum(p.numel() for p in model.global_encoder.parameters())
        ),
    }


# ---------------------------------------------------------------------------
# vocabulary -- real-pipeline audit
# ---------------------------------------------------------------------------


def _frequency_distribution(counts: Counter) -> dict[str, Any]:
    frequencies = np.asarray(sorted(counts.values()), dtype=np.int64)
    total = int(frequencies.sum())
    out: dict[str, Any] = {
        "unique_types": int(frequencies.size),
        "occurrences": total,
        "singleton_types": int(np.sum(frequencies == 1)),
        "rare_le_2_types": int(np.sum(frequencies <= 2)),
        "rare_le_5_types": int(np.sum(frequencies <= 5)),
        "rare_le_10_types": int(np.sum(frequencies <= 10)),
        "max_frequency": int(frequencies.max()) if frequencies.size else 0,
        "median_frequency": float(np.median(frequencies)) if frequencies.size else 0.0,
        "mean_frequency": float(frequencies.mean()) if frequencies.size else 0.0,
    }
    for key, value in (
        ("singleton", 1),
        ("le_2", 2),
        ("le_5", 5),
        ("le_10", 10),
    ):
        out[f"{key}_occurrence_fraction"] = (
            float(frequencies[frequencies <= value].sum() / total) if total else 0.0
        )
    ordered = np.sort(frequencies)[::-1]
    cumulative = np.cumsum(ordered) / total if total else np.zeros_like(ordered)
    for top in (50, 100, 250, 500, 1000, 5000):
        out[f"coverage_top_{top}"] = (
            float(cumulative[min(top, ordered.size) - 1]) if ordered.size else 0.0
        )
    return out


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _split_vocabulary_stats(
    records: Sequence[Any],
    vocabulary: Mapping[bytes, int],
    field: str,
    train_frequency: Mapping[bytes, int],
) -> dict[str, Any]:
    counts: Counter = Counter()
    per_molecule_oov: list[float] = []
    per_molecule_min_freq: list[float] = []
    per_molecule_mean_freq: list[float] = []
    known = 0
    n_patches = 0
    for record in records:
        values = [getattr(patch, field) for patch in record.patches]
        if not values:
            continue
        counts.update(values)
        oov = [value not in vocabulary for value in values]
        known += int(sum(not flag for flag in oov))
        n_patches += len(values)
        per_molecule_oov.append(float(np.mean(oov)))
        freqs = np.asarray(
            [
                0 if value not in vocabulary else int(train_frequency.get(value, 0))
                for value in values
            ],
            dtype=np.float64,
        )
        per_molecule_min_freq.append(float(freqs.min()))
        per_molecule_mean_freq.append(float(freqs.mean()))
    types = set(counts)
    oov_molecule_array = np.asarray(per_molecule_oov, dtype=np.float64)
    return {
        "field": field,
        "molecules": int(len(records)),
        "patch_occurrences": int(n_patches),
        "unique_types": int(len(types)),
        "vocabulary_size": int(len(vocabulary)),
        "known_occurrence_fraction": float(known / max(n_patches, 1)),
        "oov_occurrence_fraction": float(1.0 - known / max(n_patches, 1)),
        "known_type_fraction": float(
            sum(t in vocabulary for t in types) / max(len(types), 1)
        ),
        "all_patches_known_molecule_fraction": float(
            np.mean(oov_molecule_array == 0.0)
        )
        if oov_molecule_array.size
        else 0.0,
        "per_molecule_oov_fraction": _summarize(per_molecule_oov),
        "per_molecule_min_train_frequency": _summarize(per_molecule_min_freq),
        "per_molecule_mean_train_frequency": _summarize(per_molecule_mean_freq),
    }


def vocabulary() -> dict[str, Any]:
    train_records = load_records("train")
    typed, parent = (
        mpp._fit_vocabulary(train_records, "typed_certificate", 32768, 1),
        mpp._fit_vocabulary(train_records, "parent_certificate", 4096, 1),
    )
    typed_counts: Counter = Counter(
        patch.typed_certificate for record in train_records for patch in record.patches
    )
    parent_counts: Counter = Counter(
        patch.parent_certificate for record in train_records for patch in record.patches
    )

    splits: dict[str, Any] = {
        "train": {
            "typed_certificate": _split_vocabulary_stats(
                train_records, typed, "typed_certificate", typed_counts
            ),
            "parent_certificate": _split_vocabulary_stats(
                train_records, parent, "parent_certificate", parent_counts
            ),
        }
    }
    for split in ("valid", "test"):
        records = load_records(split)
        splits[split] = {
            "typed_certificate": _split_vocabulary_stats(
                records, typed, "typed_certificate", typed_counts
            ),
            "parent_certificate": _split_vocabulary_stats(
                records, parent, "parent_certificate", parent_counts
            ),
        }
        del records

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "pipeline": "molhiv_patch_path_pooling exact-rooted record cache (real pipeline)",
        "train_molecules": int(len(train_records)),
        "typed": {
            "train_vocabulary_size": int(len(typed)),
            "train_vocabulary_size_with_oov": int(len(typed) + 1),
            "embedding_dim": TOKEN_WIDTH,
            "identity_params": int((len(typed) + 1) * TOKEN_WIDTH),
            "train_frequency_distribution": _frequency_distribution(typed_counts),
            "train_unique_certificate_types": int(len(typed_counts)),
        },
        "parent": {
            "train_vocabulary_size": int(len(parent)),
            "train_vocabulary_size_with_oov": int(len(parent) + 1),
            "embedding_dim": PARENT_WIDTH,
            "identity_params": int((len(parent) + 1) * PARENT_WIDTH),
            "train_frequency_distribution": _frequency_distribution(parent_counts),
            "train_unique_certificate_types": int(len(parent_counts)),
        },
        "splits": splits,
        "distribution_shift_check": _distribution_shift_check(splits),
        "official_test_loaded": "read-only vocabulary diagnostics (already opened once)",
    }
    _write_json(RESULTS_DIR / "vocabulary_audit.json", payload)
    return payload


def _distribution_shift_check(splits: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("typed_certificate", "parent_certificate"):
        valid = splits["valid"][field]
        test = splits["test"][field]
        out[field] = {
            "valid_oov_occurrence_fraction": valid["oov_occurrence_fraction"],
            "test_oov_occurrence_fraction": test["oov_occurrence_fraction"],
            "oov_occurrence_delta_valid_minus_test": (
                valid["oov_occurrence_fraction"] - test["oov_occurrence_fraction"]
            ),
            "valid_molecules_with_oov_fraction": (
                1.0 - valid["all_patches_known_molecule_fraction"]
            ),
            "test_molecules_with_oov_fraction": (
                1.0 - test["all_patches_known_molecule_fraction"]
            ),
            "valid_per_molecule_oov_mean": valid["per_molecule_oov_fraction"]["mean"],
            "test_per_molecule_oov_mean": test["per_molecule_oov_fraction"]["mean"],
            "material_shift": bool(
                abs(
                    valid["oov_occurrence_fraction"]
                    - test["oov_occurrence_fraction"]
                )
                > 0.05
                or abs(
                    valid["per_molecule_oov_fraction"]["mean"]
                    - test["per_molecule_oov_fraction"]["mean"]
                )
                > 0.05
            ),
        }
    out["summary"] = (
        "valid and test are compared only descriptively; no cutoff is chosen here"
    )
    return out


# ---------------------------------------------------------------------------
# growth law + top-K accounting (pure accounting, no model built)
# ---------------------------------------------------------------------------


def _topk_accounting(
    fixed_params: int, total_params: int, typed_size: int, parent_size: int
) -> list[dict[str, Any]]:
    """Pure top-K accounting for the typed identity vocabulary.

    The parent vocabulary is held fixed; the typed vocabulary keeps the ``K``
    most frequent train identities plus one OOV/UNK row.
    """
    rows = []
    for k in (50, 100, 250, 500, 1000):
        capped = min(int(k), int(typed_size))
        typed_params = (capped + 1) * TOKEN_WIDTH
        parent_params = (int(parent_size) + 1) * PARENT_WIDTH
        theoretical_total = int(fixed_params) + typed_params + parent_params
        rows.append(
            {
                "top_k": int(k),
                "typed_vocabulary_kept": capped,
                "typed_oov_plus_unk": 1,
                "parent_vocabulary_kept": int(parent_size),
                "identity_params": int(typed_params + parent_params),
                "theoretical_total_params": int(theoretical_total),
                "identity_fraction_of_total": float(
                    (typed_params + parent_params) / theoretical_total
                ),
                "saved_vs_current": int(total_params - theoretical_total),
            }
        )
    return rows


def growth() -> dict[str, Any]:
    vocab = fit_train_vocabularies()
    typed_size = int(vocab["typed_vocabulary_size"])
    parent_size = int(vocab["parent_vocabulary_size"])
    model, _ = build_actual_model()
    total = int(sum(p.numel() for p in model.parameters()))
    identity = (typed_size + 1) * TOKEN_WIDTH + (parent_size + 1) * PARENT_WIDTH
    fixed = total - identity

    rows = _topk_accounting(fixed, total, typed_size, parent_size)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "growth_law": (
            "identity_params(V_typed, V_parent) = "
            "(V_typed + 1) * token_width + (V_parent + 1) * parent_width"
        ),
        "fixed_backbone_params": int(fixed),
        "current": {
            "typed_vocabulary_size": typed_size,
            "parent_vocabulary_size": parent_size,
            "token_width": TOKEN_WIDTH,
            "parent_width": PARENT_WIDTH,
            "typed_identity_params": int((typed_size + 1) * TOKEN_WIDTH),
            "parent_identity_params": int((parent_size + 1) * PARENT_WIDTH),
            "identity_params": int(identity),
            "total_params": int(total),
        },
        "top_k_accounting": rows,
        "note": (
            "pure accounting for later design discussion; these cutoffs are NOT "
            "recommended, NOT ranked, and were NOT chosen using valid/test "
            "performance"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "growth_law.json", payload)
    return payload


# ---------------------------------------------------------------------------
# compare -- ZINC identity vs fixed structure
# ---------------------------------------------------------------------------


def compare() -> dict[str, Any]:
    """Build the ZINC cell A / H96 attribute structure from the frozen code."""
    from tracks.ksvd.experiments.luyin16 import (
        zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
    )

    def _vocab_size(embedding: torch.nn.Module) -> int:
        if hasattr(embedding, "vocabulary_size"):
            return int(embedding.vocabulary_size)
        return int(embedding.num_embeddings)

    def _embedding_dim(embedding: torch.nn.Module) -> int:
        if hasattr(embedding, "output_width"):
            return int(embedding.output_width)
        return int(embedding.embedding_dim)

    rows = []
    for cell in ("A", "D"):
        model = cd.build_cell(cell, 0)
        total = int(sum(p.numel() for p in model.parameters()))
        named = {name: int(p.numel()) for name, p in model.named_parameters()}
        typed = sum(v for k, v in named.items() if k.startswith("typed_embedding."))
        parent = sum(v for k, v in named.items() if k.startswith("parent_embedding."))
        identity = typed + parent
        rows.append(
            {
                "cell": cell,
                "total_params": total,
                "identity_storage_params": identity,
                "identity_storage_fraction": identity / total,
                "typed_certificate_embedding_params": typed,
                "parent_certificate_embedding_params": parent,
                "fixed_backbone_params": total - identity,
                "fixed_fraction": (total - identity) / total,
                "vocabulary": {
                    "typed_vocabulary_size_with_oov": _vocab_size(
                        model.typed_embedding
                    ),
                    "parent_vocabulary_size_with_oov": _vocab_size(
                        model.parent_embedding
                    ),
                    "typed_embedding_dim": _embedding_dim(model.typed_embedding),
                    "parent_embedding_dim": _embedding_dim(model.parent_embedding),
                    "typed_lookup_kind": type(model.typed_embedding).__name__,
                    "parent_lookup_kind": type(model.parent_embedding).__name__,
                },
            }
        )

    breakdown_path = RESULTS_DIR / "parameter_breakdown.json"
    molhiv = _read_json(breakdown_path) if breakdown_path.exists() else None
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "zinc": rows,
        "molhiv": None
        if molhiv is None
        else {
            "total_params": molhiv["total_params"],
            "identity_storage_params": molhiv["identity_storage_params"],
            "identity_storage_fraction": molhiv["identity_storage_fraction"],
            "fixed_backbone_params": molhiv["fixed_size_params"],
            "fixed_fraction": molhiv["fixed_size_fraction"],
            "vocabulary": molhiv["vocabulary_used"],
        },
        "comparison_note": (
            "same model family (compact patch encoder + pair/centre machinery + "
            "typed certificate identity lookup); the parameter scale difference is "
            "driven by the number of train-derived certificate identities, not by a "
            "different backbone size"
        ),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "zinc_comparison.json", payload)
    return payload


# ---------------------------------------------------------------------------
# optional diagnostic -- rarity/OOV vs frozen predictions
# ---------------------------------------------------------------------------


def diagnostic() -> dict[str, Any]:
    """Zero-training rarity/OOV x prediction subgroup diagnostic.

    Runs only when the frozen raw checkpoints from the already-completed
    ``molhiv_recurrent_pair_centre_v1`` run are present.  Never trains and never
    selects an architecture.
    """
    from sklearn.metrics import roc_auc_score

    raw_paths = {seed: rpc.RESULTS_DIR / f"raw_state_seed{seed}.pt" for seed in (0, 1)}
    if not all(path.exists() for path in raw_paths.values()):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "ran": False,
            "reason": "frozen raw checkpoints not present locally",
            "official_test_loaded": False,
        }
        _write_json(RESULTS_DIR / "rarity_prediction_diagnostic.json", payload)
        return payload

    train_records = load_records("train")
    transforms = rpc.fit_train_transforms(train_records)
    typed = transforms["typed_vocabulary"]
    train_counts: Counter = Counter(
        patch.typed_certificate for record in train_records for patch in record.patches
    )
    del train_records

    model, _vocab = build_actual_model()
    model.eval()

    out: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "ran": True,
        "note": (
            "read-only diagnostic on the already-opened official test split; "
            "subgroup AUCs are descriptive, not causal, and not an architecture "
            "decision"
        ),
        "splits": {},
        "official_test_loaded": True,
    }
    for split in ("valid", "test"):
        records = load_records(split)
        graphs = rpc.encode_records(records, transforms)
        loader = rpc._valid_loader(graphs, 0)
        logits = {}
        for seed, path in raw_paths.items():
            state = torch.load(path, map_location="cpu", weights_only=True)
            _targets, prediction = rpc._predict(model, state, loader, torch.device("cpu"))
            logits[seed] = prediction
        mean_logits = np.mean(
            np.stack([logits[s] for s in sorted(logits)], axis=0), axis=0
        )
        labels = np.asarray([float(record.y) for record in records], dtype=np.float64)
        out["splits"][split] = _diagnostic_split(
            split, records, typed, train_counts, labels, mean_logits, roc_auc_score
        )
        del records, graphs, loader, logits
    _write_json(RESULTS_DIR / "rarity_prediction_diagnostic.json", out)
    return out


def _bucket_auc(
    mask: np.ndarray, labels: np.ndarray, logits: np.ndarray, roc_auc_score
) -> dict[str, Any]:
    n = int(mask.sum())
    positives = int(labels[mask].sum()) if n else 0
    negatives = n - positives
    payload: dict[str, Any] = {"n": n, "positives": positives, "negatives": negatives}
    if positives >= 10 and negatives >= 10:
        payload["auc"] = float(roc_auc_score(labels[mask], logits[mask]))
    else:
        payload["auc"] = None
        payload["auc_skipped_reason"] = "fewer than 10 positives or 10 negatives"
    return payload


def _diagnostic_split(
    split: str,
    records: Sequence[Any],
    typed: Mapping[bytes, int],
    train_counts: Mapping[bytes, int],
    labels: np.ndarray,
    logits: np.ndarray,
    roc_auc_score,
) -> dict[str, Any]:
    oov = []
    min_freq = []
    mean_freq = []
    for record in records:
        values = [patch.typed_certificate for patch in record.patches]
        if values:
            oov.append(float(np.mean([value not in typed for value in values])))
            freqs = np.asarray(
                [int(train_counts.get(value, 0)) for value in values], dtype=np.float64
            )
            min_freq.append(float(freqs.min()))
            mean_freq.append(float(freqs.mean()))
        else:
            oov.append(0.0)
            min_freq.append(0.0)
            mean_freq.append(0.0)
    oov = np.asarray(oov, dtype=np.float64)
    min_freq = np.asarray(min_freq, dtype=np.float64)
    mean_freq = np.asarray(mean_freq, dtype=np.float64)

    edges = np.quantile(mean_freq, [1 / 3, 2 / 3]) if mean_freq.size else [0.0, 0.0]
    terciles = {
        "low": mean_freq <= edges[0],
        "mid": (mean_freq > edges[0]) & (mean_freq <= edges[1]),
        "high": mean_freq > edges[1],
    }
    return {
        "split": split,
        "molecules": int(len(records)),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "oov_fraction_buckets": {
            "zero": _bucket_auc(oov == 0.0, labels, logits, roc_auc_score),
            "(0,0.1]": _bucket_auc(
                (oov > 0.0) & (oov <= 0.1), labels, logits, roc_auc_score
            ),
            "(0.1,0.3]": _bucket_auc(
                (oov > 0.1) & (oov <= 0.3), labels, logits, roc_auc_score
            ),
            ">0.3": _bucket_auc(oov > 0.3, labels, logits, roc_auc_score),
        },
        "min_train_frequency_buckets": {
            "0_oov_present": _bucket_auc(
                min_freq == 0.0, labels, logits, roc_auc_score
            ),
            "1": _bucket_auc(min_freq == 1.0, labels, logits, roc_auc_score),
            "(1,5]": _bucket_auc(
                (min_freq > 1.0) & (min_freq <= 5.0), labels, logits, roc_auc_score
            ),
            ">5": _bucket_auc(min_freq > 5.0, labels, logits, roc_auc_score),
        },
        "mean_train_frequency_terciles": {
            "edges": [float(e) for e in edges],
            "buckets": {
                name: _bucket_auc(mask, labels, logits, roc_auc_score)
                for name, mask in terciles.items()
            },
        },
        "overall_auc": float(roc_auc_score(labels, logits))
        if 0 < int(labels.sum()) < labels.size
        else None,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / name
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "parameter_breakdown": _maybe("parameter_breakdown.json"),
        "vocabulary_audit": _maybe("vocabulary_audit.json"),
        "growth_law": _maybe("growth_law.json"),
        "zinc_comparison": _maybe("zinc_comparison.json"),
        "rarity_prediction_diagnostic": _maybe("rarity_prediction_diagnostic.json"),
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["params", "vocabulary", "growth", "compare", "diagnostic", "report"],
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.stage == "params":
        print(json.dumps(params(), indent=2, default=str), flush=True)
    if args.stage == "vocabulary":
        print(json.dumps(vocabulary(), indent=2, default=str), flush=True)
    if args.stage == "growth":
        print(json.dumps(growth(), indent=2, default=str), flush=True)
    if args.stage == "compare":
        print(json.dumps(compare(), indent=2, default=str), flush=True)
    if args.stage == "diagnostic":
        print(json.dumps(diagnostic(), indent=2, default=str), flush=True)
    if args.stage == "report":
        print(json.dumps(report(), indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
