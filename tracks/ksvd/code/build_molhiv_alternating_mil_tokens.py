"""Refit matched dictionaries on cross-fitted MIL-selected MolHIV witnesses.

This is a low-cost one-step discriminative/alternating dictionary screen.  A
previously fitted, graph-level MIL scorer over a frozen KSVD cache ranks atom
patches without assigning the molecule label to every patch.  For every positive
inner-train molecule and a matched deterministic set of negative molecules, the
same fixed number of highest-scoring patches is retained.  KSVD, PCA, and random-
patch dictionaries are then fitted to the *identical* witness matrix with the
same numeric seed.  Only official-train graphs are encoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from ogb.utils.features import get_atom_feature_dims
from sklearn.model_selection import StratifiedKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.build_molhiv_mil_novelty_tokens import fit_dictionary, summarize_graph
from code.data_molhiv import load_molhiv
from code.ksvd import _omp
from code.molhiv_node_tokens import centered_ego_vector, graph_node_offsets


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def select_graph_balanced_atoms(
    dictionary: np.ndarray,
    witness_matrix: np.ndarray,
    witness_graph_indices: np.ndarray,
    graph_labels: np.ndarray,
    *,
    keep_atoms: int,
    sparsity: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select atoms by graph-level positive/negative usage effect.

    Sparse codes are first averaged within each selected graph, so every graph
    contributes one observation regardless of witness count.  Ranking uses
    absolute atom usage and therefore is invariant to dictionary-atom signs.
    Equal budgets are retained from the positive- and negative-associated ends.
    """
    if keep_atoms <= 0 or keep_atoms > dictionary.shape[1]:
        raise ValueError("invalid graph-balanced atom keep budget")
    if witness_matrix.shape[1] != witness_graph_indices.size:
        raise ValueError("witness graph indices do not match witness columns")
    codes = np.stack([
        _omp(dictionary, witness_matrix[:, j], sparsity)
        for j in range(witness_matrix.shape[1])
    ], axis=1)
    absolute = np.abs(codes)
    unique_graphs = np.unique(witness_graph_indices)
    graph_usage = np.stack([
        absolute[:, witness_graph_indices == graph_i].mean(axis=1)
        for graph_i in unique_graphs
    ], axis=0)
    labels = np.asarray(graph_labels[unique_graphs], dtype=np.int64)
    positive = graph_usage[labels == 1]
    negative = graph_usage[labels == 0]
    if positive.shape[0] == 0 or negative.shape[0] == 0:
        raise RuntimeError("atom selection requires both graph classes")
    difference = positive.mean(axis=0) - negative.mean(axis=0)
    standard_error = np.sqrt(
        positive.var(axis=0) / positive.shape[0]
        + negative.var(axis=0) / negative.shape[0]
        + 1e-8
    )
    effect = difference / standard_error
    positive_budget = keep_atoms // 2
    negative_budget = keep_atoms - positive_budget
    selected_positive = [int(i) for i in np.argsort(-effect)[:positive_budget]]
    selected_negative = [
        int(i) for i in np.argsort(effect)
        if int(i) not in selected_positive
    ][:negative_budget]
    selected = np.asarray(
        selected_positive + selected_negative, dtype=np.int64
    )
    return dictionary[:, selected], {
        "method": "graph_balanced_signed_usage_effect",
        "candidate_atom_count": int(dictionary.shape[1]),
        "keep_atom_count": int(keep_atoms),
        "coding_sparsity": int(sparsity),
        "graph_count": int(unique_graphs.size),
        "positive_graph_count": int(positive.shape[0]),
        "negative_graph_count": int(negative.shape[0]),
        "selected_atom_indices": selected.tolist(),
        "selected_positive_associated_atoms": selected_positive,
        "selected_negative_associated_atoms": selected_negative,
        "selected_effects": effect[selected].astype(float).tolist(),
        "all_effects": effect.astype(float).tolist(),
        "all_mean_usage_difference": difference.astype(float).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-token-cache", required=True)
    ap.add_argument("--initial-family", choices=("ksvd", "pca", "random_patch"), default="ksvd")
    ap.add_argument("--mil-result", required=True)
    ap.add_argument("--inner-split-cache", required=True)
    ap.add_argument("--inner-fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=8)
    ap.add_argument("--n-atoms", type=int, default=8)
    ap.add_argument("--sparsity", type=int, default=2)
    ap.add_argument(
        "--encoding-mode",
        choices=("joint_omp", "background_then_witness"),
        default="joint_omp",
        help=(
            "joint_omp uses the historical single pursuit over the concatenated "
            "bank; background_then_witness first codes common chemistry and "
            "then codes its residual with the witness bank"
        ),
    )
    ap.add_argument(
        "--background-encoding-sparsity", type=int, default=1,
        help="background pursuit budget for background_then_witness encoding",
    )
    ap.add_argument(
        "--witness-encoding-sparsity", type=int, default=1,
        help="witness-residual pursuit budget for background_then_witness encoding",
    )
    ap.add_argument(
        "--witness-fit-sparsity", type=int, default=0,
        help=(
            "OMP sparsity used while fitting the retained witness bank; "
            "0 preserves the historical behavior and reuses --sparsity. "
            "The final joint node encoding always uses --sparsity."
        ),
    )
    ap.add_argument(
        "--candidate-witness-atoms", type=int, default=0,
        help=(
            "if greater than --n-atoms, fit this overcomplete witness bank "
            "then retain --n-atoms by graph-balanced signed usage effect"
        ),
    )
    ap.add_argument("--candidate-sparsity", type=int, default=3)
    ap.add_argument("--ksvd-iter", type=int, default=4)
    ap.add_argument(
        "--witness-consistency-target",
        choices=("none", "oof_score", "oof_score_and_prominence"),
        default="none",
        help=(
            "augment each selected witness patch during dictionary fitting with "
            "cross-fitted MIL evidence.  The deployed dictionary retains only "
            "the chemistry rows; no held-out label or score enters encoding"
        ),
    )
    ap.add_argument(
        "--witness-consistency-weight", type=float, default=0.0,
        help=(
            "scale of the standardized OOF witness-evidence rows in the "
            "augmented LC/D-KSVD objective; zero preserves historical fitting"
        ),
    )
    ap.add_argument(
        "--witness-consistency-clip", type=float, default=3.0,
        help="absolute clipping applied after fit-witness target standardization",
    )
    ap.add_argument(
        "--ksvd-anchor-strength", type=float, default=0.0,
        help=(
            "proximal shrinkage toward each KSVD atom's initial sampled patch "
            "after every dictionary sweep; zero preserves historical KSVD"
        ),
    )
    ap.add_argument(
        "--ksvd-coherence-step", type=float, default=0.0,
        help=(
            "projected gradient step on off-diagonal witness-atom Gram "
            "penalty after each KSVD update; zero preserves historical KSVD"
        ),
    )
    ap.add_argument(
        "--ksvd-initialization",
        choices=("random_columns", "deterministic_svd", "maxmin_columns"),
        default="random_columns",
        help=(
            "initialization used only for KSVD witness dictionaries; "
            "deterministic_svd supports a zero-update PCA control and "
            "SVD-warm-started KSVD; maxmin_columns selects diverse real "
            "witness patches before standard KSVD updates"
        ),
    )
    ap.add_argument("--witnesses-per-graph", type=int, default=2)
    ap.add_argument("--negative-graph-ratio", type=float, default=1.0)
    ap.add_argument("--balance-seed", type=int, default=20260727)
    ap.add_argument("--dict-seed", type=int, default=1000003)
    ap.add_argument("--families", default="ksvd,pca,random_patch")
    ap.add_argument(
        "--shared-background-family",
        choices=("none", "ksvd", "pca", "random_patch"),
        default="none",
        help=(
            "optionally concatenate the same train-only background dictionary "
            "for every witness-dictionary family; this makes KSVD/PCA/random "
            "witness controls differ only in the witness bank"
        ),
    )
    ap.add_argument(
        "--witness-target",
        choices=("raw_patch", "background_residual"),
        default="raw_patch",
        help=(
            "fit witness atoms to the selected raw patches (historical path) "
            "or to their unit-normalized residual after background-only OMP"
        ),
    )
    ap.add_argument(
        "--retain-initial-dictionary", action="store_true",
        help=(
            "concatenate each family's train-only background dictionary with "
            "its OOF-MIL witness dictionary, then re-encode by joint OMP"
        ),
    )
    ap.add_argument(
        "--class-conditional-witness-dictionary", action="store_true",
        help=(
            "split the fixed witness atom budget equally between negative- and "
            "positive-bag witness pools; requires --retain-initial-dictionary"
        ),
    )
    ap.add_argument(
        "--emit-background-witness-gain-features", action="store_true",
        help=(
            "store a compact graph residual describing the reconstruction gain "
            "of the witness bank over background-only OMP; requires "
            "--retain-initial-dictionary"
        ),
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.inner_fold < 0 or args.witnesses_per_graph <= 0:
        raise ValueError("invalid inner fold or witness budget")
    if args.n_atoms <= 0 or not 1 <= args.sparsity <= args.n_atoms:
        raise ValueError("invalid atom count or final coding sparsity")
    if args.background_encoding_sparsity <= 0 or args.witness_encoding_sparsity <= 0:
        raise ValueError("hierarchical encoding sparsities must be positive")
    if args.encoding_mode == "background_then_witness":
        if not args.retain_initial_dictionary:
            raise ValueError(
                "background_then_witness requires --retain-initial-dictionary"
            )
        if (
            args.background_encoding_sparsity
            + args.witness_encoding_sparsity
            != args.sparsity
        ):
            raise ValueError(
                "hierarchical background+witness budgets must sum to --sparsity"
            )
    if args.ksvd_iter < 0:
        raise ValueError("--ksvd-iter must be non-negative")
    if (
        not np.isfinite(args.witness_consistency_weight)
        or args.witness_consistency_weight < 0.0
    ):
        raise ValueError("--witness-consistency-weight must be finite and non-negative")
    if (
        not np.isfinite(args.witness_consistency_clip)
        or args.witness_consistency_clip <= 0.0
    ):
        raise ValueError("--witness-consistency-clip must be finite and positive")
    consistency_enabled = (
        args.witness_consistency_target != "none"
        and args.witness_consistency_weight > 0.0
    )
    if (args.witness_consistency_target == "none") != (
        args.witness_consistency_weight == 0.0
    ):
        raise ValueError(
            "use target=none with weight=0, or a non-none target with positive weight"
        )
    if (
        not np.isfinite(args.ksvd_anchor_strength)
        or not 0.0 <= args.ksvd_anchor_strength < 1.0
    ):
        raise ValueError("--ksvd-anchor-strength must be finite and in [0,1)")
    if not np.isfinite(args.ksvd_coherence_step) or args.ksvd_coherence_step < 0.0:
        raise ValueError("--ksvd-coherence-step must be finite and non-negative")
    if args.witness_fit_sparsity < 0:
        raise ValueError("--witness-fit-sparsity must be non-negative")
    effective_witness_fit_sparsity = (
        args.witness_fit_sparsity if args.witness_fit_sparsity else args.sparsity
    )
    if not 1 <= effective_witness_fit_sparsity <= args.n_atoms:
        raise ValueError("invalid effective witness fit sparsity")
    if args.candidate_witness_atoms < 0:
        raise ValueError("--candidate-witness-atoms must be non-negative")
    if args.candidate_witness_atoms and args.candidate_witness_atoms <= args.n_atoms:
        raise ValueError("--candidate-witness-atoms must exceed --n-atoms")
    if args.candidate_witness_atoms and not (
        1 <= args.candidate_sparsity <= args.candidate_witness_atoms
    ):
        raise ValueError("invalid --candidate-sparsity")
    if args.negative_graph_ratio <= 0:
        raise ValueError("--negative-graph-ratio must be positive")
    if consistency_enabled and args.candidate_witness_atoms:
        raise ValueError(
            "witness consistency is intentionally screened without candidate pruning"
        )
    if consistency_enabled and args.class_conditional_witness_dictionary:
        raise ValueError(
            "witness consistency is intentionally screened without class-conditional banks"
        )
    if consistency_enabled and args.witness_target != "raw_patch":
        raise ValueError(
            "witness consistency currently requires the shared raw-patch target"
        )
    if args.class_conditional_witness_dictionary:
        if args.candidate_witness_atoms:
            raise ValueError(
                "candidate atom selection is incompatible with class-conditional banks"
            )
        if not args.retain_initial_dictionary:
            raise ValueError(
                "--class-conditional-witness-dictionary requires "
                "--retain-initial-dictionary"
            )
        if args.n_atoms < 2:
            raise ValueError("class-conditional witness banks require at least 2 atoms")
    if args.emit_background_witness_gain_features and not args.retain_initial_dictionary:
        raise ValueError(
            "--emit-background-witness-gain-features requires "
            "--retain-initial-dictionary"
        )

    started = time.time()
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    if len(set(families)) != len(families):
        raise ValueError("duplicate dictionary family")
    if args.shared_background_family != "none" and not args.retain_initial_dictionary:
        raise ValueError(
            "--shared-background-family requires --retain-initial-dictionary"
        )

    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    graphs = bundle.graphs
    labels = (np.asarray(bundle.y) > 0.5).astype(np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.inner_split_cache) as split:
        inner_train = np.asarray(
            split[f"fold_{args.inner_fold}_train_indices"], dtype=np.int64
        )
        inner_valid = np.asarray(
            split[f"fold_{args.inner_fold}_valid_indices"], dtype=np.int64
        )
    if not np.all(np.isin(inner_train, official_train)):
        raise RuntimeError("inner train is not contained in official train")

    with np.load(args.initial_token_cache) as initial:
        offsets = np.asarray(initial["offsets"], dtype=np.int64)
        original_indices = np.asarray(initial["original_indices"], dtype=np.int64)
        initial_codes = np.asarray(initial[f"tokens_{args.initial_family}"], dtype=np.float64)
        initial_dictionary = np.asarray(initial[f"dictionary_{args.initial_family}"], dtype=np.float64)
        required_background_families = set(families)
        if args.shared_background_family != "none":
            required_background_families.add(args.shared_background_family)
        initial_dictionaries = {
            family: np.asarray(initial[f"dictionary_{family}"], dtype=np.float64)
            for family in sorted(required_background_families)
        }
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    if not np.array_equal(original_indices, expected_original):
        raise RuntimeError("initial cache does not match dataset subset")
    if offsets.shape != (len(graphs) + 1,):
        raise RuntimeError("initial cache offsets mismatch")
    if np.any(initial_codes[int(offsets[official_valid[0]]):] != 0):
        # The caches in this protocol have official-valid followed by test, but
        # retain explicit split checks below for a clearer audit failure.
        for graph_i in np.concatenate([official_valid, official_test]).tolist():
            lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
            if np.any(initial_codes[lo:hi] != 0):
                raise RuntimeError("initial cache encoded official-valid/test")

    mil_result = json.loads(Path(args.mil_result).read_text())
    readout = mil_result["token_preprocessing"]["graph_code_readout"]
    if readout.get("kind") != "crossfit_graph_label_mil_positive_novelty":
        raise RuntimeError("--mil-result is not a cross-fitted MIL novelty run")
    if int(readout["crossfit_folds"]) != len(readout["crossfit_models"]):
        raise RuntimeError("MIL cross-fit metadata is incomplete")
    if Path(mil_result["token_cache"]).resolve() != Path(args.initial_token_cache).resolve():
        raise RuntimeError("MIL result and initial token cache do not match")
    if mil_result.get("family") != args.initial_family:
        raise RuntimeError("MIL result family and --initial-family do not match")

    d = initial_codes.shape[1]
    gram = initial_dictionary.T @ initial_dictionary
    absolute = np.abs(initial_codes)
    assignment = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1e-10)
    explained = np.einsum("ni,ij,nj->n", initial_codes, gram, initial_codes, optimize=True)
    reconstruction = np.sqrt(np.maximum(1.0 - explained, 0.0))
    energy = np.sqrt(np.square(initial_codes).sum(axis=1))
    instance_features = np.concatenate([
        assignment,
        ((reconstruction - float(readout["reconstruction_fit_mean"])) /
         max(float(readout["reconstruction_fit_scale"]), 1e-6))[:, None],
        ((energy - float(readout["energy_fit_mean"])) /
         max(float(readout["energy_fit_scale"]), 1e-6))[:, None],
    ], axis=1)

    # Reconstruct the exact cross-fitting partition used by the runner.  Each
    # inner-train molecule is ranked by a scorer that was not fitted on it.
    labels_fit = labels[inner_train]
    splitter = StratifiedKFold(
        n_splits=int(readout["crossfit_folds"]),
        shuffle=True,
        random_state=int(mil_result["seed"]) + 271828,
    )
    graph_model: dict[int, dict[str, Any]] = {}
    for model_info, (_, heldout_pos) in zip(
        readout["crossfit_models"], splitter.split(inner_train, labels_fit)
    ):
        for graph_i in inner_train[heldout_pos].tolist():
            graph_model[int(graph_i)] = model_info
    if len(graph_model) != len(inner_train):
        raise RuntimeError("failed to assign one OOF MIL model per inner-train graph")

    positives = inner_train[labels_fit == 1]
    negatives = inner_train[labels_fit == 0]
    n_negative = min(
        len(negatives), int(round(len(positives) * args.negative_graph_ratio))
    )
    balance_rng = np.random.default_rng(args.balance_seed + args.inner_fold)
    selected_negatives = balance_rng.choice(negatives, size=n_negative, replace=False)
    selected_graphs = np.concatenate([positives, selected_negatives])
    balance_rng.shuffle(selected_graphs)

    selected_nodes: list[tuple[int, int, float]] = []
    selected_prominence: list[float] = []
    score_by_class: dict[int, list[float]] = {0: [], 1: []}
    for graph_i in selected_graphs.tolist():
        model_info = graph_model[int(graph_i)]
        weights = np.asarray(model_info["atom_weights"], dtype=np.float64)
        novelty_weight = float(model_info["novelty_weight"])
        energy_weight = float(model_info["energy_weight"])
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        features = instance_features[lo:hi]
        scores = (
            features[:, :d] @ weights
            + novelty_weight * features[:, d]
            + energy_weight * features[:, d + 1]
        )
        take = min(args.witnesses_per_graph, hi - lo)
        chosen = np.argsort(-scores, kind="stable")[:take]
        score_center = float(np.median(scores))
        score_scale = float(np.std(scores))
        score_scale = max(score_scale, 1e-6)
        for local_i in chosen.tolist():
            score = float(scores[local_i])
            selected_nodes.append((int(graph_i), int(local_i), score))
            selected_prominence.append((score - score_center) / score_scale)
            score_by_class[int(labels[graph_i])].append(score)

    if bundle.node_feats is None or bundle.edge_feats is None:
        raise RuntimeError("MolHIV chemical features are required")
    atom_dims = get_atom_feature_dims()
    wl_topology_dim = (
        args.max_nodes
        + args.max_nodes * (args.max_nodes + 1) // 2
        + 64 * 3 + 3
    )
    labeled_base_dim = wl_topology_dim + 64 + 16 + 64 * 3
    explicit_ring_dim = 13
    center_start = labeled_base_dim + explicit_ring_dim
    center_end = center_start + sum(atom_dims)
    patch_view_indices: np.ndarray | None = None

    def patch(graph_i: int, node_i: int) -> np.ndarray:
        nonlocal patch_view_indices
        vec = centered_ego_vector(
            graphs[graph_i], node_i,
            bundle.node_feats[graph_i], bundle.edge_feats[graph_i],
            radius=args.radius, max_nodes=args.max_nodes,
        )
        y = np.asarray(vec, dtype=np.float64)
        y /= max(float(np.linalg.norm(y)), 1e-12)
        if patch_view_indices is None:
            patch_view_indices = np.concatenate([
                np.arange(labeled_base_dim, dtype=np.int64),
                np.arange(center_start, y.size, dtype=np.int64),
            ])
            if center_end > y.size:
                raise RuntimeError("unexpected centered patch layout")
        y = y[patch_view_indices]
        return y / max(float(np.linalg.norm(y)), 1e-12)

    witness_matrix = np.stack(
        [patch(graph_i, node_i) for graph_i, node_i, _ in selected_nodes], axis=1
    )
    print(
        f"shared OOF-MIL witness pool={witness_matrix.shape}; "
        f"graphs pos/neg={len(positives)}/{len(selected_negatives)}",
        flush=True,
    )

    witness_labels = np.asarray(
        [labels[graph_i] for graph_i, _, _ in selected_nodes], dtype=np.int64
    )

    # Continuous label-consistency target derived only from cross-fitted MIL
    # instance evidence.  Each score comes from a model that did not train on
    # that molecule.  Unlike historical patch-label augmentation, graph labels
    # are not replicated onto patches and are not part of this target matrix.
    witness_consistency: np.ndarray | None = None
    witness_consistency_info: dict[str, Any] = {
        "enabled": bool(consistency_enabled),
        "target": args.witness_consistency_target,
        "weight": float(args.witness_consistency_weight),
        "clip": float(args.witness_consistency_clip),
        "source": "cross_fitted_oof_mil_instance_evidence",
        "graph_labels_copied_to_patches": False,
    }
    if consistency_enabled:
        raw_score = np.asarray([x[2] for x in selected_nodes], dtype=np.float64)
        raw_columns = [raw_score]
        target_names = ["oof_instance_score"]
        if args.witness_consistency_target == "oof_score_and_prominence":
            raw_columns.append(np.asarray(selected_prominence, dtype=np.float64))
            target_names.append("within_graph_score_prominence")
        raw_target = np.stack(raw_columns, axis=0)
        target_mean = raw_target.mean(axis=1, keepdims=True)
        target_scale = raw_target.std(axis=1, keepdims=True)
        target_scale = np.maximum(target_scale, 1e-6)
        witness_consistency = (raw_target - target_mean) / target_scale
        witness_consistency = np.clip(
            witness_consistency,
            -args.witness_consistency_clip,
            args.witness_consistency_clip,
        )
        witness_consistency_info.update({
            "target_names": target_names,
            "target_dim": int(witness_consistency.shape[0]),
            "raw_mean": target_mean[:, 0].astype(float).tolist(),
            "raw_scale": target_scale[:, 0].astype(float).tolist(),
            "standardized_min": witness_consistency.min(axis=1).astype(float).tolist(),
            "standardized_max": witness_consistency.max(axis=1).astype(float).tolist(),
            "target_sha256": array_hash(witness_consistency.astype(np.float32)),
        })

    backgrounds = {
        family: (
            initial_dictionaries[args.shared_background_family]
            if args.shared_background_family != "none"
            else initial_dictionaries[family]
        )
        for family in families
    }
    witness_targets: dict[str, np.ndarray] = {}
    witness_target_info: dict[str, dict[str, Any]] = {}
    for family in families:
        if args.witness_target == "raw_patch":
            target = witness_matrix.copy()
            residual_norms = np.zeros(witness_matrix.shape[1], dtype=np.float64)
        else:
            background = backgrounds[family]
            codes = np.stack([
                _omp(background, witness_matrix[:, j], args.sparsity)
                for j in range(witness_matrix.shape[1])
            ], axis=1)
            residual = witness_matrix - background @ codes
            residual_norms = np.linalg.norm(residual, axis=0)
            if np.any(residual_norms <= 1e-10):
                raise RuntimeError(
                    f"{family} background exactly reconstructs a selected witness"
                )
            target = residual / residual_norms[None, :]
        witness_targets[family] = target
        witness_target_info[family] = {
            "kind": args.witness_target,
            "background_family": (
                args.shared_background_family
                if args.shared_background_family != "none" else family
            ),
            "background_atom_count": int(backgrounds[family].shape[1]),
            "residual_norm_mean": (
                float(residual_norms.mean())
                if args.witness_target == "background_residual" else None
            ),
            "residual_norm_std": (
                float(residual_norms.std())
                if args.witness_target == "background_residual" else None
            ),
            "residual_norm_min": (
                float(residual_norms.min())
                if args.witness_target == "background_residual" else None
            ),
            "residual_norm_max": (
                float(residual_norms.max())
                if args.witness_target == "background_residual" else None
            ),
            "target_sha256": array_hash(target.astype(np.float32)),
        }

    target_hashes = {
        row["target_sha256"] for row in witness_target_info.values()
    }
    shared_witness_target = len(target_hashes) == 1
    if args.shared_background_family != "none" and not shared_witness_target:
        raise RuntimeError("shared background produced non-identical witness targets")

    dictionaries: dict[str, np.ndarray] = {}
    dictionary_info: dict[str, Any] = {}
    for family in families:
        family_witness_matrix = witness_targets[family]
        if args.class_conditional_witness_dictionary:
            negative_atom_count = args.n_atoms // 2
            positive_atom_count = args.n_atoms - negative_atom_count
            negative_witnesses = family_witness_matrix[:, witness_labels == 0]
            positive_witnesses = family_witness_matrix[:, witness_labels == 1]
            if effective_witness_fit_sparsity > min(
                negative_atom_count, positive_atom_count
            ):
                raise ValueError(
                    "witness fit sparsity exceeds a class-conditional atom bank"
                )
            negative_dictionary, negative_info = fit_dictionary(
                negative_witnesses, family, negative_atom_count,
                effective_witness_fit_sparsity,
                args.ksvd_iter, args.dict_seed,
                ksvd_initialization=args.ksvd_initialization,
                ksvd_coherence_step=args.ksvd_coherence_step,
                ksvd_anchor_strength=args.ksvd_anchor_strength,
            )
            positive_dictionary, positive_info = fit_dictionary(
                positive_witnesses, family, positive_atom_count,
                effective_witness_fit_sparsity,
                args.ksvd_iter, args.dict_seed + 1000003,
                ksvd_initialization=args.ksvd_initialization,
                ksvd_coherence_step=args.ksvd_coherence_step,
                ksvd_anchor_strength=args.ksvd_anchor_strength,
            )
            dictionary = np.concatenate(
                [negative_dictionary, positive_dictionary], axis=1
            )
            info = {
                "method": "class_conditional_oof_mil_witness_dictionary",
                "negative_atom_count": int(negative_atom_count),
                "positive_atom_count": int(positive_atom_count),
                "negative_witness_count": int(negative_witnesses.shape[1]),
                "positive_witness_count": int(positive_witnesses.shape[1]),
                "negative_dictionary": negative_info,
                "positive_dictionary": positive_info,
                "negative_dictionary_seed": int(args.dict_seed),
                "positive_dictionary_seed": int(args.dict_seed + 1000003),
            }
        else:
            fit_atom_count = (
                args.candidate_witness_atoms
                if args.candidate_witness_atoms else args.n_atoms
            )
            fit_sparsity = (
                args.candidate_sparsity
                if args.candidate_witness_atoms
                else effective_witness_fit_sparsity
            )
            fit_matrix = family_witness_matrix
            chemistry_dim = int(family_witness_matrix.shape[0])
            if witness_consistency is not None:
                fit_matrix = np.vstack([
                    family_witness_matrix,
                    args.witness_consistency_weight * witness_consistency,
                ])
            dictionary, info = fit_dictionary(
                fit_matrix, family, fit_atom_count, fit_sparsity,
                args.ksvd_iter, args.dict_seed,
                ksvd_initialization=args.ksvd_initialization,
                ksvd_coherence_step=args.ksvd_coherence_step,
                ksvd_anchor_strength=args.ksvd_anchor_strength,
            )
            if witness_consistency is not None:
                augmented_dictionary = dictionary
                dictionary = augmented_dictionary[:chemistry_dim].copy()
                chemistry_norm = np.linalg.norm(dictionary, axis=0)
                if np.any(chemistry_norm <= 1e-10):
                    raise RuntimeError(
                        f"{family} consistency fit produced a zero chemistry atom"
                    )
                dictionary /= chemistry_norm[None, :]
                chemistry_codes = np.stack([
                    _omp(dictionary, family_witness_matrix[:, j], fit_sparsity)
                    for j in range(family_witness_matrix.shape[1])
                ], axis=1)
                chemistry_residual = family_witness_matrix - dictionary @ chemistry_codes
                target_head, _, _, _ = np.linalg.lstsq(
                    chemistry_codes.T, witness_consistency.T, rcond=None
                )
                target_prediction = chemistry_codes.T @ target_head
                target_error = witness_consistency.T - target_prediction
                target_variance = np.square(
                    witness_consistency.T - witness_consistency.T.mean(axis=0, keepdims=True)
                ).sum()
                target_r2 = 1.0 - float(np.square(target_error).sum()) / max(
                    float(target_variance), 1e-12
                )
                info = {
                    "method": "continuous_oof_witness_consistent_dictionary",
                    "family": family,
                    "augmented_fit": info,
                    "chemistry_dimension": chemistry_dim,
                    "consistency_dimension": int(witness_consistency.shape[0]),
                    "consistency_weight": float(args.witness_consistency_weight),
                    "chemistry_atom_norm_before_renormalization": chemistry_norm.astype(float).tolist(),
                    "chemistry_reconstruction_relative_error": float(
                        np.linalg.norm(chemistry_residual)
                        / max(np.linalg.norm(family_witness_matrix), 1e-12)
                    ),
                    "sparse_code_to_consistency_train_r2": float(target_r2),
                    "augmented_dictionary_sha256": array_hash(
                        augmented_dictionary.astype(np.float32)
                    ),
                    "deployed_chemistry_dictionary_sha256": array_hash(
                        dictionary.astype(np.float32)
                    ),
                }
            if args.candidate_witness_atoms:
                candidate_dictionary = dictionary
                dictionary, selection_info = select_graph_balanced_atoms(
                    candidate_dictionary,
                    family_witness_matrix,
                    np.asarray(
                        [x[0] for x in selected_nodes], dtype=np.int64
                    ),
                    labels,
                    keep_atoms=args.n_atoms,
                    sparsity=args.candidate_sparsity,
                )
                selection_info["candidate_dictionary_sha256"] = array_hash(
                    candidate_dictionary.astype(np.float32)
                )
                selection_info["selected_dictionary_sha256"] = array_hash(
                    dictionary.astype(np.float32)
                )
                info = {
                    "family": family,
                    "method": "overcomplete_dictionary_graph_balanced_atom_selection",
                    "candidate_dictionary": info,
                    "atom_selection": selection_info,
                }
        if args.retain_initial_dictionary:
            background = backgrounds[family]
            if background.shape[0] != dictionary.shape[0]:
                raise RuntimeError(
                    f"{family} background/witness feature dimensions differ: "
                    f"{background.shape[0]} vs {dictionary.shape[0]}"
                )
            dictionary = np.concatenate([background, dictionary], axis=1)
            info = {
                "method": (
                    "train_background_plus_class_conditional_oof_mil_witness_joint_omp"
                    if args.class_conditional_witness_dictionary
                    else (
                        "train_background_plus_oof_mil_selected_witness_joint_omp"
                        if args.candidate_witness_atoms
                        else (
                            "train_background_plus_oof_mil_residual_witness_joint_omp"
                            if args.witness_target == "background_residual"
                            else "train_background_plus_oof_mil_witness_joint_omp"
                        )
                    )
                ),
                "background_atom_count": int(background.shape[1]),
                "witness_atom_count": int(dictionary.shape[1] - background.shape[1]),
                "witness_target": witness_target_info[family],
                "witness_dictionary": info,
            }
        dictionaries[family] = dictionary
        dictionary_info[family] = info
        print(
            f"fitted {family} dictionary with {dictionary.shape[1]} atoms",
            flush=True,
        )

    atom_counts = {int(dictionary.shape[1]) for dictionary in dictionaries.values()}
    if len(atom_counts) != 1:
        raise RuntimeError(f"dictionary families have different atom counts: {atom_counts}")
    final_n_atoms = atom_counts.pop()
    total_nodes = int(offsets[-1])
    tokens = {
        family: np.zeros((total_nodes, final_n_atoms), dtype=np.float32)
        for family in families
    }
    graph_features: dict[str, np.ndarray] = {}
    background_grams: dict[str, np.ndarray] = {}
    joint_grams: dict[str, np.ndarray] = {}
    background_atom_counts = {
        family: int(backgrounds[family].shape[1]) for family in families
    }
    if args.emit_background_witness_gain_features:
        for family in families:
            background_count = background_atom_counts[family]
            witness_count = final_n_atoms - background_count
            if witness_count <= 0:
                raise RuntimeError(f"{family} has no witness atoms")
            graph_features[family] = np.zeros(
                (len(graphs), 32 + 4 * witness_count), dtype=np.float32
            )
            background_grams[family] = (
                backgrounds[family].T @ backgrounds[family]
            )
            joint_grams[family] = dictionaries[family].T @ dictionaries[family]

    for count, graph_i in enumerate(official_train.tolist(), 1):
        lo = int(offsets[graph_i])
        background_errors: dict[str, list[float]] = {family: [] for family in families}
        joint_errors: dict[str, list[float]] = {family: [] for family in families}
        witness_codes: dict[str, list[np.ndarray]] = {family: [] for family in families}
        for node_i in range(graphs[graph_i].n):
            y = patch(int(graph_i), int(node_i))
            for family in families:
                if args.encoding_mode == "joint_omp":
                    code = _omp(dictionaries[family], y, args.sparsity)
                else:
                    background = backgrounds[family]
                    background_code = _omp(
                        background, y, args.background_encoding_sparsity
                    )
                    residual = y - background @ background_code
                    background_count = background_atom_counts[family]
                    witness_dictionary = dictionaries[family][:, background_count:]
                    if witness_dictionary.shape[1] <= 0:
                        raise RuntimeError(f"{family} has no hierarchical witness bank")
                    witness_code = _omp(
                        witness_dictionary, residual,
                        args.witness_encoding_sparsity,
                    )
                    code = np.concatenate([background_code, witness_code])
                tokens[family][lo + node_i] = code.astype(np.float32)
                if args.emit_background_witness_gain_features:
                    background = backgrounds[family]
                    background_code = _omp(background, y, args.sparsity)
                    background_errors[family].append(float(max(
                        1.0 - background_code @ background_grams[family] @ background_code,
                        0.0,
                    )))
                    joint_errors[family].append(float(max(
                        1.0 - code @ joint_grams[family] @ code, 0.0
                    )))
                    witness_codes[family].append(
                        code[background_atom_counts[family]:].copy()
                    )
        if args.emit_background_witness_gain_features:
            for family in families:
                graph_features[family][graph_i] = summarize_graph(
                    np.asarray(background_errors[family], dtype=np.float64),
                    np.asarray(joint_errors[family], dtype=np.float64),
                    np.stack(witness_codes[family], axis=0),
                ).astype(np.float32)
        if count % 500 == 0 or count == len(official_train):
            print(f"encoded official-train graphs {count}/{len(official_train)}", flush=True)

    for graph_i in np.concatenate([official_valid, official_test]).tolist():
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        for family in families:
            if np.any(tokens[family][lo:hi] != 0):
                raise RuntimeError("official-valid/test tokens are nonzero")
            if args.emit_background_witness_gain_features and np.any(
                graph_features[family][graph_i] != 0
            ):
                raise RuntimeError(
                    "official-valid/test background-witness graph features are nonzero"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "offsets": offsets,
        "original_indices": original_indices,
        "train_indices": official_train,
        "valid_indices": official_valid,
        "test_indices": official_test,
        "inner_train_indices": inner_train,
        "inner_valid_indices": inner_valid,
        "selected_graph_indices": selected_graphs.astype(np.int64),
        "selected_witness_graph_indices": np.asarray(
            [x[0] for x in selected_nodes], dtype=np.int64
        ),
        "selected_witness_node_indices": np.asarray(
            [x[1] for x in selected_nodes], dtype=np.int64
        ),
        "selected_witness_scores": np.asarray(
            [x[2] for x in selected_nodes], dtype=np.float32
        ),
    }
    for family in families:
        archive[f"dictionary_{family}"] = dictionaries[family].astype(np.float32)
        archive[f"tokens_{family}"] = tokens[family]
        if args.emit_background_witness_gain_features:
            archive[f"graph_features_{family}"] = graph_features[family]
    np.savez_compressed(output, **archive)

    metadata = {
        "protocol_id": (
            "molhiv-oof-mil-background-plus-class-conditional-witness-dictionary-v1"
            if args.class_conditional_witness_dictionary
            else (
                "molhiv-oof-mil-background-plus-graph-balanced-selected-witness-dictionary-v1"
                if args.candidate_witness_atoms and args.retain_initial_dictionary
                else (
                    "molhiv-one-step-oof-mil-graph-balanced-selected-witness-dictionary-v1"
                    if args.candidate_witness_atoms
                    else (
                        (
                            "molhiv-oof-mil-background-plus-residual-witness-dictionary-v1"
                            if args.witness_target == "background_residual"
                            else (
                                "molhiv-oof-mil-background-plus-incoherent-witness-dictionary-v1"
                                if args.ksvd_coherence_step > 0.0
                                else (
                                    "molhiv-oof-mil-background-plus-maxmin-initialized-witness-dictionary-v1"
                                    if args.ksvd_initialization == "maxmin_columns"
                                    else "molhiv-oof-mil-background-plus-witness-dictionary-v1"
                                )
                            )
                        )
                        if args.retain_initial_dictionary
                        else "molhiv-one-step-oof-mil-selected-shared-witness-dictionary-v1"
                    )
                )
            )
        ),
        "test_policy": "official-valid/test patches were not vectorized or encoded",
        "config": vars(args),
        "families": families,
        "initial_family": args.initial_family,
        "initial_mil_crossfit_oof_auc": float(readout["crossfit_oof_auc"]),
        "inner_train_count": int(len(inner_train)),
        "inner_valid_count": int(len(inner_valid)),
        "selected_positive_graph_count": int(len(positives)),
        "selected_negative_graph_count": int(len(selected_negatives)),
        "witness_count": int(witness_matrix.shape[1]),
        "witness_feature_dim": int(witness_matrix.shape[0]),
        "final_dictionary_atom_count": int(final_n_atoms),
        "effective_witness_fit_sparsity": int(
            effective_witness_fit_sparsity
        ),
        "final_joint_encoding_sparsity": int(args.sparsity),
        "encoding_mode": args.encoding_mode,
        "background_encoding_sparsity": (
            int(args.background_encoding_sparsity)
            if args.encoding_mode == "background_then_witness" else None
        ),
        "witness_encoding_sparsity": (
            int(args.witness_encoding_sparsity)
            if args.encoding_mode == "background_then_witness" else None
        ),
        "retained_initial_dictionary": bool(args.retain_initial_dictionary),
        "class_conditional_witness_dictionary": bool(
            args.class_conditional_witness_dictionary
        ),
        "background_witness_gain_features": bool(
            args.emit_background_witness_gain_features
        ),
        "background_witness_gain_feature_dim": (
            int(next(iter(graph_features.values())).shape[1])
            if graph_features else 0
        ),
        "witness_score_mean_negative": float(np.mean(score_by_class[0])),
        "witness_score_mean_positive": float(np.mean(score_by_class[1])),
        "shared_witness_pool_across_families": True,
        "shared_witness_target_across_families": bool(shared_witness_target),
        "witness_target_info": witness_target_info,
        "witness_consistency_info": witness_consistency_info,
        "matched_numeric_dictionary_seed": int(args.dict_seed),
        "dictionary_info": dictionary_info,
        "explicit_ring_features": False,
        "official_valid_encoded": False,
        "official_test_encoded": False,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "fingerprints": {
            "official_train_sha256": array_hash(official_train),
            "inner_train_sha256": array_hash(inner_train),
            "inner_valid_sha256": array_hash(inner_valid),
            "selected_graphs_sha256": array_hash(selected_graphs),
            "selected_witness_graphs_sha256": array_hash(archive["selected_witness_graph_indices"]),
            "selected_witness_nodes_sha256": array_hash(archive["selected_witness_node_indices"]),
            "shared_witness_matrix_sha256": array_hash(witness_matrix.astype(np.float32)),
            "witness_target_sha256": {
                family: witness_target_info[family]["target_sha256"]
                for family in families
            },
            "witness_consistency_sha256": (
                witness_consistency_info.get("target_sha256")
                if consistency_enabled else None
            ),
        },
        "elapsed_sec": time.time() - started,
        "archive": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output} and {output.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
