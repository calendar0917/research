"""Strict paired GINE vs localized KSVD/PCA/random node-token experiment.

Each atom receives a deterministic radius-r ego patch code.  Token models begin
exactly as the GINE baseline because all token-to-node gates are zero initialized.
Epoch selection is performed inside official train; official valid is evaluated
once after exact-seed full-train retraining.  By default official-test token rows
must remain zero and test graphs are not evaluated.  The explicit post-freeze
``--evaluate-test`` mode requires a frozen configuration id and a fully encoded
frozen-test cache, then evaluates official test exactly once.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-cache", required=True)
    ap.add_argument(
        "--family",
        choices=(
            "none", "random_patch", "pca", "ksvd",
            "random_patch_multiscale", "pca_multiscale", "ksvd_multiscale",
            "ksvd_ring",
            "ksvd_ring_graph",
            "ksvd_ring_typed",
            "ksvd_cluster_balanced",
            "ksvd_mixture",
            "ksvd_consensus_atoms",
            "ksvd_consensus_codes",
            "ksvd_consensus_shrink",
            "ksvd_recon_weighted",
            "ksvd_task_aware",
        ),
        required=True,
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument(
        "--residual-gine", action="store_true",
        help="add parameter-free hidden-state residuals after every GINE layer",
    )
    ap.add_argument(
        "--jk-readout", choices=("last", "gated_sum"), default="last",
        help=(
            "graph readout from the final layer only, or final plus independently "
            "zero-initialized gated earlier-layer graph states"
        ),
    )
    ap.add_argument(
        "--graph-readout", choices=("mean", "gated_sum_mean"), default="mean",
        help=(
            "mean pooling, or a zero-initialized gated interpolation toward "
            "size-normalized sum pooling"
        ),
    )
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--token-lr-scale", type=float, default=1.0)
    ap.add_argument("--token-weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--token-warmup-epochs", type=int, default=0,
        help=(
            "keep the token optimizer group frozen for the first N epochs; "
            "useful for zero-init residual branches that should not perturb "
            "early backbone optimization"
        ),
    )
    ap.add_argument(
        "--parameter-ema-decay", type=float, default=0.0,
        help="EMA decay for all trainable parameters during training; 0 disables EMA",
    )
    ap.add_argument(
        "--token-dropout", type=float, default=0.0,
        help="drop projected node-token embeddings during training only",
    )
    ap.add_argument(
        "--token-gate-scale", type=float, default=1.0,
        help="multiply tanh token gates by this fixed scale",
    )
    ap.add_argument(
        "--ring-gate-scale", type=float, default=1.0,
        help="separate fixed scale for explicit ring injection/readout gates",
    )
    ap.add_argument(
        "--context-gate-scale", type=float, default=1.0,
        help="fixed cap/scale for graph-global KSVD context injection",
    )
    ap.add_argument(
        "--token-gate-l2", type=float, default=0.0,
        help="L2 penalty on effective token gates",
    )
    ap.add_argument(
        "--token-inject-layers", type=int, default=0,
        help="inject into the first N GINE layers; 0 means all layers",
    )
    ap.add_argument(
        "--token-normalization", choices=("zscore", "rms", "none"), default="zscore",
        help="train-only code scaling; rms preserves exact sparse zeros",
    )
    ap.add_argument(
        "--token-zscore-clip", type=float, default=0.0,
        help="clip normalized signed coefficients to +/- this value; 0 disables clipping",
    )
    ap.add_argument(
        "--token-prevalence-power", type=float, default=0.0,
        help=(
            "fit-only graph-frequency shrinkage exponent: multiply each atom by "
            "graph_prevalence**power and renormalize retained weights to mean one"
        ),
    )
    ap.add_argument(
        "--token-min-graph-frequency", type=float, default=0.0,
        help="mask dictionary atoms appearing in fewer than this fraction of fit graphs",
    )
    ap.add_argument(
        "--token-channels", choices=("signed", "signed_abs_support"), default="signed",
        help="optionally expose coefficient magnitude and exact OMP support",
    )
    ap.add_argument(
        "--token-fusion",
        choices=(
            "inject", "adaptive_inject", "inject_motif_readout",
            "ring_dual_inject", "ring_dual_readout",
            "bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject",
            "unrolled_ksvd_inject", "unrolled_ksvd_mix",
            "ksvd_context_inject", "ksvd_context_tied",
            "ksvd_augmented_edges",
            "ksvd_graph_residual_mean", "ksvd_graph_residual_max",
            "ksvd_code_residual_maxabs",
            "ksvd_code_residual_rich", "ksvd_code_residual_rich_recon",
            "ksvd_code_residual_mil_novelty",
            "ksvd_precomputed_graph_residual",
            "ksvd_atom_additive_readout",
            "ksvd_atom_specific_readout",
            "motif_transition_readout",
            "motif_geometry_transition_readout",
            "motif_transition_matrix_readout",
            "ksvd_jk_router",
            "ksvd_aux_support", "ksvd_aux_code",
            "motif_slot", "motif_slot_no_id", "motif_slot_shuffled_id",
            "motif_slot_graph", "motif_slot_graph_no_id",
            "motif_slot_graph_shuffled_id",
            "motif_slot_recurrent", "motif_slot_recurrent_no_id",
            "motif_slot_recurrent_shuffled_id",
            "motif_occurrence", "motif_occurrence_no_id",
            "motif_occurrence_shuffled_id",
            "motif_ego", "motif_ego_no_id", "motif_ego_shuffled_id",
            "motif_sparse_ego", "motif_sparse_ego_no_id",
            "motif_sparse_ego_shuffled_id",
            "ksvd_cell_recurrent", "ksvd_cell_recurrent_no_id",
            "ksvd_cell_recurrent_shuffled_id",
            "ksvd_witness_cell_readout", "ksvd_witness_cell_readout_no_id",
            "ksvd_witness_cell_readout_shuffled_id",
        ),
        default="inject",
        help=(
            "choose scalar, node-adaptive, ring, dictionary-slot, or low-rank "
            "bilinear KSVD-atom fusion"
        ),
    )
    ap.add_argument(
        "--graph-residual-hidden", type=int, default=0,
        help=(
            "optional hidden width for an additive nonlinear correction over "
            "the fixed KSVD graph descriptor; 0 preserves the historical "
            "linear residual exactly"
        ),
    )
    ap.add_argument(
        "--graph-residual-training",
        choices=("joint", "independent", "fixed_probe"),
        default="joint",
        help=(
            "joint trains the historical additive residual end to end; "
            "independent trains separate BCE objectives for the GINE and "
            "dictionary-only graph predictors; fixed_probe installs a convex "
            "balanced logistic probe and trains only GINE"
        ),
    )
    ap.add_argument(
        "--graph-residual-combination",
        choices=("logit_add", "probability_average"),
        default="logit_add",
        help=(
            "fixed inference rule for graph-residual models; logit_add "
            "preserves the historical implementation exactly"
        ),
    )
    ap.add_argument(
        "--graph-residual-probe-c", type=float, default=0.1,
        help="L2 inverse regularization for fixed balanced convex probe",
    )
    ap.add_argument(
        "--graph-residual-independent-loss",
        choices=("bce", "pairwise_logistic"),
        default="bce",
        help=(
            "objective for the dictionary-only predictor in independent mode; "
            "pairwise_logistic directly optimizes positive-negative ranking"
        ),
    )
    ap.add_argument(
        "--motif-slot-rank", type=int, default=16,
        help="bottleneck/atom-ID width for incidence-topology motif slots",
    )
    ap.add_argument(
        "--mil-temperature", type=float, default=0.25,
        help=(
            "soft maximum temperature for the fixed graph-label MIL witness "
            "scorer over sparse-code instances"
        ),
    )
    ap.add_argument(
        "--mil-fit-epochs", type=int, default=120,
        help="optimization epochs for each small fixed MIL witness scorer",
    )
    ap.add_argument(
        "--mil-fit-lr", type=float, default=0.05,
        help="learning rate for the fixed MIL witness scorer",
    )
    ap.add_argument(
        "--mil-weight-decay", type=float, default=0.01,
        help="L2 penalty on the fixed MIL instance direction",
    )
    ap.add_argument(
        "--mil-crossfit-folds", type=int, default=3,
        help=(
            "stratified cross-fitting folds for train-graph MIL features; "
            "held-out scaffold graphs always use the full fit-only scorer"
        ),
    )
    ap.add_argument(
        "--motif-geometry-assignment", choices=("abs", "signed"), default="abs",
        help=(
            "coefficient transport for fixed dictionary geometry: abs gives a "
            "support distribution; signed preserves the OMP reconstruction direction"
        ),
    )
    ap.add_argument(
        "--motif-geometry-atom-normalization", choices=("row", "none"), default="row",
        help=(
            "row reproduces the normalized atom embedding pilot; none keeps exact "
            "truncated-SVD reconstruction coordinates"
        ),
    )
    ap.add_argument(
        "--motif-slot-backend", choices=("dense", "sparse"), default="dense",
        help="dense is the frozen pilot implementation; sparse is a faster numerical variant",
    )
    ap.add_argument(
        "--motif-atom-id-source",
        choices=("learned", "dictionary_svd"), default="learned",
        help=(
            "represent motif type by a learned lookup, or by a fixed rank-r "
            "geometry embedding derived from the fitted dictionary columns"
        ),
    )
    ap.add_argument(
        "--motif-gate-mode", choices=("scalar", "zero_out", "atomwise"), default="scalar",
        help=(
            "scalar preserves the P0 outer zero gate; zero_out instead zero-"
            "initializes the final broadcast projection so the residual starts "
            "at zero but receives nonzero gradients immediately after warmup"
        ),
    )
    ap.add_argument(
        "--motif-gate-wake-init", type=float, default=0.0,
        help=(
            "for scalar motif gates, set the raw gate to this value exactly "
            "when token warmup ends; this preserves an exact GINE warmup while "
            "avoiding the zero-gate branch-gradient dead zone afterwards"
        ),
    )
    ap.add_argument(
        "--motif-slot-shuffle-seed", type=int, default=314159,
        help="deterministic graph-wise atom-ID permutation seed for shuffled controls",
    )
    ap.add_argument(
        "--motif-occurrence-radius", type=int, choices=(1, 2), default=1,
        help=(
            "connect centers assigned to the same dictionary atom when their "
            "molecular graph distance is at most this radius"
        ),
    )
    ap.add_argument(
        "--motif-ego-radius", type=int, choices=(1, 2), default=2,
        help=(
            "radius of each centered learned-motif occurrence; its top-1 "
            "dictionary atom supplies the occurrence type"
        ),
    )
    ap.add_argument(
        "--motif-ego-confidence",
        choices=("none", "dominance"), default="none",
        help=(
            "for top-1 motif egos, weight overlapping occurrences by the "
            "dominant OMP coefficient's share of total sparse-code mass"
        ),
    )
    ap.add_argument(
        "--motif-ego-min-atom-index", type=int, default=0,
        help=(
            "retain top-1 motif egos only when the dominant dictionary atom "
            "index is at least this value; 0 keeps every dictionary bank"
        ),
    )
    ap.add_argument(
        "--motif-ego-min-dominance", type=float, default=0.0,
        help=(
            "retain top-1 motif egos only when max(|code|)/sum(|code|) is at "
            "least this value; 0 preserves the historical implementation"
        ),
    )
    ap.add_argument(
        "--ksvd-aux-weight", type=float, default=0.05,
        help=(
            "weight of the train-only KSVD node-target auxiliary objective; "
            "used only by ksvd_aux_support / ksvd_aux_code"
        ),
    )
    ap.add_argument(
        "--ksvd-aux-epochs", type=int, default=0,
        help=(
            "apply the KSVD auxiliary objective only through this epoch; "
            "0 keeps it active for all training epochs"
        ),
    )
    ap.add_argument(
        "--token-interaction-rank", type=int, default=8,
        help="rank of the low-rank KSVD atom interaction branch",
    )
    ap.add_argument(
        "--token-interaction-support-mask", action="store_true",
        help=(
            "before bilinear fusion, restore exact OMP sparsity by zeroing "
            "z-scored coefficients whose raw KSVD coefficient was zero"
        ),
    )
    ap.add_argument("--unrolled-steps", type=int, default=2)
    ap.add_argument(
        "--unrolled-step-init", type=float, default=0.05,
        help="initial positive ISTA step size; it is bounded below 1/L",
    )
    ap.add_argument(
        "--unrolled-lambda-init", type=float, default=0.02,
        help="initial soft-threshold lambda (actual threshold is eta*lambda)",
    )
    ap.add_argument(
        "--unrolled-threshold-mode", choices=("scalar", "per_atom"), default="per_atom",
    )
    ap.add_argument(
        "--unrolled-recon-weight", type=float, default=0.0,
        help="inner-train-only normalized patch reconstruction auxiliary weight",
    )
    ap.add_argument(
        "--unrolled-sparse-weight", type=float, default=0.0,
        help="inner-train-only refined-code L1 auxiliary weight",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--inner-split-seed", type=int, default=1729)
    ap.add_argument("--inner-valid-fraction", type=float, default=0.15)
    ap.add_argument(
        "--inner-split-cache", default="",
        help="optional official-train-only fold archive produced by build_molhiv_scaffold_folds",
    )
    ap.add_argument(
        "--inner-fold", type=int, default=-1,
        help="fold id inside --inner-split-cache; legacy random split is used when unset",
    )
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--save-inner-predictions", action="store_true",
        help=(
            "store inner-validation labels/probabilities for each epoch; "
            "selection-only diagnostics such as fixed-weight ensembling only"
        ),
    )
    ap.add_argument(
        "--selection-only", action="store_true",
        help="stop after inner-train/inner-valid screening; never evaluate official valid",
    )
    ap.add_argument(
        "--capture-official-valid-predictions", action="store_true",
        help=(
            "store official-valid labels/probabilities without evaluating test; "
            "for a pre-frozen validation-only ensemble audit"
        ),
    )
    ap.add_argument(
        "--evaluate-test", action="store_true",
        help="post-freeze only: evaluate official test once and store predictions",
    )
    ap.add_argument(
        "--frozen-config-id", default=None,
        help="required with --evaluate-test; identifies the pre-frozen configuration",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    try:
        import torch

        if not getattr(torch.load, "_ksvd_patched", False):
            original_load = torch.load

            def patched_load(*a, **kw):  # type: ignore[no-untyped-def]
                kw.setdefault("weights_only", False)
                return original_load(*a, **kw)

            patched_load._ksvd_patched = True  # type: ignore[attr-defined]
            torch.load = patched_load  # type: ignore[assignment]

        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import (
            GINEConv,
            global_add_pool,
            global_max_pool,
            global_mean_pool,
        )
    except ImportError as exc:
        raise RuntimeError(f"missing GINE dependencies: {exc}") from exc

    from code.data_molhiv import check_env, load_molhiv

    if args.evaluate_test and args.selection_only:
        raise ValueError("--evaluate-test cannot be combined with --selection-only")
    if args.capture_official_valid_predictions and args.selection_only:
        raise ValueError("--capture-official-valid-predictions cannot be selection-only")
    if args.evaluate_test and not args.frozen_config_id:
        raise ValueError("--evaluate-test requires --frozen-config-id")
    if not 0.05 <= args.inner_valid_fraction <= 0.4:
        raise ValueError("--inner-valid-fraction must be in [0.05, 0.4]")
    if not 0.0 <= args.token_dropout < 1.0:
        raise ValueError("--token-dropout must be in [0, 1)")
    if not 0.0 <= args.parameter_ema_decay < 1.0:
        raise ValueError("--parameter-ema-decay must be in [0, 1)")
    if args.token_gate_scale < 0.0:
        raise ValueError("--token-gate-scale must be nonnegative")
    if args.motif_gate_wake_init < 0.0:
        raise ValueError("--motif-gate-wake-init must be nonnegative")
    if args.motif_gate_mode != "scalar" and args.motif_gate_wake_init != 0.0:
        raise ValueError("--motif-gate-wake-init is only valid with scalar gate mode")
    if args.ring_gate_scale < 0.0:
        raise ValueError("--ring-gate-scale must be nonnegative")
    if args.context_gate_scale < 0.0:
        raise ValueError("--context-gate-scale must be nonnegative")
    if args.token_gate_l2 < 0.0:
        raise ValueError("--token-gate-l2 must be nonnegative")
    if args.ksvd_aux_weight < 0.0:
        raise ValueError("--ksvd-aux-weight must be nonnegative")
    if not 0 <= args.ksvd_aux_epochs <= args.epochs:
        raise ValueError("--ksvd-aux-epochs must be between 0 and --epochs")
    if not 0 <= args.token_warmup_epochs <= args.epochs:
        raise ValueError("--token-warmup-epochs must be between 0 and --epochs")
    if args.token_interaction_rank <= 0:
        raise ValueError("--token-interaction-rank must be positive")
    if args.graph_residual_hidden < 0:
        raise ValueError("--graph-residual-hidden must be nonnegative")
    if args.motif_slot_rank <= 0:
        raise ValueError("--motif-slot-rank must be positive")
    if args.unrolled_steps <= 0:
        raise ValueError("--unrolled-steps must be positive")
    if args.unrolled_step_init <= 0.0:
        raise ValueError("--unrolled-step-init must be positive")
    if args.unrolled_lambda_init <= 0.0:
        raise ValueError("--unrolled-lambda-init must be positive")
    if args.unrolled_recon_weight < 0.0 or args.unrolled_sparse_weight < 0.0:
        raise ValueError("unrolled auxiliary weights must be nonnegative")
    if args.motif_ego_min_atom_index < 0:
        raise ValueError("--motif-ego-min-atom-index must be nonnegative")
    if not 0.0 <= args.motif_ego_min_dominance <= 1.0:
        raise ValueError("--motif-ego-min-dominance must be in [0, 1]")
    if (
        args.token_interaction_support_mask
        and args.token_fusion not in {"bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject"}
    ):
        raise ValueError(
            "--token-interaction-support-mask requires a bilinear interaction fusion"
        )
    if args.token_zscore_clip < 0.0:
        raise ValueError("--token-zscore-clip must be nonnegative")
    if args.token_zscore_clip > 0.0 and args.token_normalization != "zscore":
        raise ValueError("--token-zscore-clip requires --token-normalization zscore")
    if args.token_prevalence_power < 0.0:
        raise ValueError("--token-prevalence-power must be nonnegative")
    if args.mil_temperature <= 0.0:
        raise ValueError("--mil-temperature must be positive")
    if args.mil_fit_epochs <= 0 or args.mil_fit_lr <= 0.0:
        raise ValueError("MIL fit epochs and learning rate must be positive")
    if args.mil_weight_decay < 0.0:
        raise ValueError("--mil-weight-decay must be nonnegative")
    if args.mil_crossfit_folds < 2:
        raise ValueError("--mil-crossfit-folds must be at least 2")
    if not 0.0 <= args.token_min_graph_frequency <= 1.0:
        raise ValueError("--token-min-graph-frequency must be in [0, 1]")
    if not 0 <= args.token_inject_layers <= args.layers:
        raise ValueError("--token-inject-layers must be between 0 and --layers")
    if args.token_fusion in {"ring_dual_inject", "ring_dual_readout"} and args.token_channels != "signed":
        raise ValueError("ring dual fusion currently requires --token-channels signed")
    unrolled_fusions = {"unrolled_ksvd_inject", "unrolled_ksvd_mix"}
    motif_slot_fusions = {
        "motif_slot", "motif_slot_no_id", "motif_slot_shuffled_id",
        "motif_slot_graph", "motif_slot_graph_no_id",
        "motif_slot_graph_shuffled_id",
        "motif_slot_recurrent", "motif_slot_recurrent_no_id",
        "motif_slot_recurrent_shuffled_id",
    }
    motif_slot_graph_fusions = {
        "motif_slot_graph", "motif_slot_graph_no_id",
        "motif_slot_graph_shuffled_id",
    }
    motif_slot_recurrent_fusions = {
        "motif_slot_recurrent", "motif_slot_recurrent_no_id",
        "motif_slot_recurrent_shuffled_id",
    }
    motif_occurrence_fusions = {
        "motif_occurrence", "motif_occurrence_no_id",
        "motif_occurrence_shuffled_id",
    }
    motif_ego_fusions = {
        "motif_ego", "motif_ego_no_id", "motif_ego_shuffled_id",
        "motif_sparse_ego", "motif_sparse_ego_no_id",
        "motif_sparse_ego_shuffled_id",
    }
    learned_cell_fusions = {
        "ksvd_cell_recurrent", "ksvd_cell_recurrent_no_id",
        "ksvd_cell_recurrent_shuffled_id",
    }
    witness_cell_readout_fusions = {
        "ksvd_witness_cell_readout", "ksvd_witness_cell_readout_no_id",
        "ksvd_witness_cell_readout_shuffled_id",
    }
    motif_sparse_ego_fusions = {
        "motif_sparse_ego", "motif_sparse_ego_no_id",
        "motif_sparse_ego_shuffled_id",
    }
    motif_topology_fusions = (
        motif_slot_fusions | motif_occurrence_fusions | motif_ego_fusions
        | learned_cell_fusions | witness_cell_readout_fusions
    )
    augmented_edge_fusions = {"ksvd_augmented_edges"}
    graph_residual_fusions = {
        "ksvd_graph_residual_mean", "ksvd_graph_residual_max",
        "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
        "ksvd_code_residual_rich_recon", "ksvd_code_residual_mil_novelty",
        "ksvd_precomputed_graph_residual",
    }
    if (
        args.graph_residual_training != "joint"
        and args.token_fusion not in graph_residual_fusions
    ):
        raise ValueError(
            "--graph-residual-training independent requires a graph-residual fusion"
        )
    if (
        args.graph_residual_combination != "logit_add"
        and args.graph_residual_training not in {"independent", "fixed_probe"}
    ):
        raise ValueError(
            "--graph-residual-combination probability_average requires "
            "independent or fixed-probe training"
        )
    if (
        args.graph_residual_independent_loss != "bce"
        and args.graph_residual_training != "independent"
    ):
        raise ValueError(
            "non-BCE graph-residual loss requires independent training"
        )
    if args.motif_gate_mode == "atomwise" and args.token_fusion not in motif_slot_fusions:
        raise ValueError("--motif-gate-mode atomwise requires a motif-slot fusion")
    if args.token_fusion in motif_topology_fusions | augmented_edge_fusions:
        if args.family not in {"ksvd", "pca", "random_patch"}:
            raise ValueError("motif topology requires ksvd/pca/random_patch family")
        if args.token_channels != "signed":
            raise ValueError(
                "motif topology uses raw signed sparse codes and requires signed channels"
            )
    if args.token_fusion in unrolled_fusions:
        if args.family != "ksvd":
            raise ValueError("unrolled KSVD fusion requires --family ksvd")
        if args.token_channels != "signed":
            raise ValueError("unrolled KSVD fusion currently requires signed channels")
    if (
        args.token_fusion in {
            "ksvd_graph_residual_mean", "ksvd_graph_residual_max"
        }
        and args.family != "ksvd"
    ):
        raise ValueError("embedded KSVD graph residual fusion requires --family ksvd")
    if (
        args.token_fusion in {
            "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
            "ksvd_code_residual_rich_recon", "ksvd_code_residual_mil_novelty",
            "ksvd_precomputed_graph_residual",
        }
        and args.family not in {"ksvd", "pca", "random_patch"}
    ):
        raise ValueError(
            "sparse-code graph residual requires ksvd/pca/random_patch family"
        )
    if (
        args.token_fusion in {
            "ksvd_atom_additive_readout", "ksvd_atom_specific_readout"
        }
        and args.family not in {"ksvd", "pca", "random_patch"}
    ):
        raise ValueError(
            "atom-additive readout requires ksvd/pca/random_patch family"
        )
    if (
        args.token_fusion in {
            "motif_transition_readout", "motif_geometry_transition_readout",
            "motif_transition_matrix_readout"
        }
        and args.family not in {"ksvd", "pca", "random_patch"}
    ):
        raise ValueError(
            "motif transition readout requires ksvd/pca/random_patch family"
        )
    if args.token_fusion == "ksvd_jk_router":
        if args.family != "ksvd":
            raise ValueError("KSVD JK router requires --family ksvd")
        if args.jk_readout != "gated_sum":
            raise ValueError("KSVD JK router requires --jk-readout gated_sum")
    if args.token_fusion in {"ksvd_aux_support", "ksvd_aux_code"}:
        if args.family != "ksvd":
            raise ValueError("KSVD auxiliary supervision requires --family ksvd")
        if args.ksvd_aux_weight <= 0.0:
            raise ValueError("KSVD auxiliary supervision requires positive --ksvd-aux-weight")
    max_graphs = None if args.max_graphs <= 0 else args.max_graphs
    device = torch.device(args.device)
    repo = Path(__file__).resolve().parents[3]
    root = repo / "data" / "ogb"
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bundle = load_molhiv(
        root=root,
        max_graphs=max_graphs,
        seed=args.data_seed,
        with_features=False,
    )
    graphs, y_np = bundle.graphs, bundle.y
    tr = np.asarray(bundle.split["train"], dtype=np.int64)
    va = np.asarray(bundle.split["valid"], dtype=np.int64)
    te = np.asarray(bundle.split["test"], dtype=np.int64)
    inner_split_protocol = "stratified-shuffle"
    inner_split_cache_path = None
    if args.inner_split_cache:
        if args.inner_fold < 0:
            raise ValueError("--inner-split-cache requires --inner-fold >= 0")
        inner_split_cache_path = Path(args.inner_split_cache)
        with np.load(inner_split_cache_path, allow_pickle=False) as fold_cache:
            train_key = f"fold_{args.inner_fold}_train_indices"
            valid_key = f"fold_{args.inner_fold}_valid_indices"
            if train_key not in fold_cache.files or valid_key not in fold_cache.files:
                raise KeyError(
                    f"fold {args.inner_fold} not present in {inner_split_cache_path}"
                )
            inner_tr = np.asarray(fold_cache[train_key], dtype=np.int64)
            inner_va = np.asarray(fold_cache[valid_key], dtype=np.int64)
            if "official_train_indices" in fold_cache.files:
                cached_tr = np.asarray(
                    fold_cache["official_train_indices"], dtype=np.int64
                )
                if not np.array_equal(cached_tr, tr):
                    raise ValueError("inner split cache official train does not match dataset")
        tr_set = set(tr.tolist())
        if set(inner_tr.tolist()) & set(inner_va.tolist()):
            raise ValueError("external inner train/valid overlap")
        if set(inner_tr.tolist()) | set(inner_va.tolist()) != tr_set:
            raise ValueError("external inner fold does not partition official train")
        if not set(inner_tr.tolist()).issubset(tr_set) or not set(inner_va.tolist()).issubset(tr_set):
            raise ValueError("external inner fold contains non-train indices")
        inner_split_protocol = "official-train-bemis-murcko-scaffold-fold"
    else:
        if args.inner_fold >= 0:
            raise ValueError("--inner-fold requires --inner-split-cache")
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.inner_valid_fraction,
            random_state=args.inner_split_seed,
        )
        inner_train_pos, inner_valid_pos = next(splitter.split(tr, y_np[tr]))
        inner_tr = tr[inner_train_pos]
        inner_va = tr[inner_valid_pos]
    log(
        f"n={len(graphs)} official train/valid={len(tr)}/{len(va)}; "
        f"inner train/valid={len(inner_tr)}/{len(inner_va)} family={args.family}"
    )

    cache_path = Path(args.token_cache)
    with np.load(cache_path) as cache:
        offsets = np.asarray(cache["offsets"], dtype=np.int64)
        original_indices = np.asarray(cache["original_indices"], dtype=np.int64)
        expected_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
        if not np.array_equal(original_indices, expected_indices):
            raise ValueError("token cache original_indices do not match the loaded subset")
        if offsets.shape != (len(graphs) + 1,):
            raise ValueError(f"offset shape mismatch: {offsets.shape}")
        dual_base_token_dim = 0
        ring_token_dim = 0
        ring_memberships_raw = None
        unrolled_correlations_raw = None
        unrolled_dictionary_raw = None
        motif_dictionary_raw = None
        graph_code_dictionary_raw = None
        precomputed_graph_features_raw = None
        if args.family == "none":
            token_raw = None
            base_token_dim = 0
            token_dim = 0
        else:
            key = f"tokens_{args.family}"
            if key not in cache.files:
                raise KeyError(f"{key!r} not in token cache {cache.files}")
            token_raw = np.asarray(cache[key], dtype=np.float32)
            base_token_dim = int(token_raw.shape[1])
            token_dim = base_token_dim * (3 if args.token_channels == "signed_abs_support" else 1)
            if args.token_fusion == "ksvd_precomputed_graph_residual":
                graph_feature_key = f"graph_features_{args.family}"
                if graph_feature_key not in cache.files:
                    raise KeyError(
                        f"{graph_feature_key!r} required for precomputed graph residual"
                    )
                precomputed_graph_features_raw = np.asarray(
                    cache[graph_feature_key], dtype=np.float64
                )
                if precomputed_graph_features_raw.shape[0] != len(graphs):
                    raise ValueError("precomputed graph feature row count mismatch")
            if args.token_fusion in {
                "ksvd_code_residual_rich_recon",
                "ksvd_code_residual_mil_novelty",
            }:
                dictionary_key = f"dictionary_{args.family}"
                if dictionary_key not in cache.files:
                    raise KeyError(
                        f"{dictionary_key!r} required for reconstruction readout"
                    )
                graph_code_dictionary_raw = np.asarray(
                    cache[dictionary_key], dtype=np.float64
                )
                if graph_code_dictionary_raw.shape[1] != base_token_dim:
                    raise ValueError("graph-code dictionary/code atom count mismatch")
            if (
                (
                    args.token_fusion in motif_topology_fusions
                    and args.motif_atom_id_source == "dictionary_svd"
                )
                or args.token_fusion == "motif_geometry_transition_readout"
            ):
                dictionary_key = f"dictionary_{args.family}"
                if dictionary_key not in cache.files:
                    raise KeyError(
                        f"{dictionary_key!r} required for dictionary_svd motif IDs"
                    )
                motif_dictionary_raw = np.asarray(
                    cache[dictionary_key], dtype=np.float32
                )
                if motif_dictionary_raw.shape[1] != base_token_dim:
                    raise ValueError("motif dictionary/code atom count mismatch")
            dual_base_token_dim = base_token_dim
            if args.token_fusion in {"ring_dual_inject", "ring_dual_readout"}:
                slices_key = (
                    "graph_token_slices" if args.family == "ksvd_ring_graph" else "token_slices"
                )
                memberships_key = (
                    "graph_ring_memberships"
                    if args.family == "ksvd_ring_graph"
                    else "ring_memberships"
                )
                if slices_key not in cache.files:
                    raise KeyError("ring dual fusion requires token_slices in token cache")
                token_slices = np.asarray(cache[slices_key], dtype=np.int64)
                if (
                    token_slices.shape != (3,)
                    or int(token_slices[0]) != 0
                    or int(token_slices[-1]) != base_token_dim
                    or not 0 < int(token_slices[1]) < base_token_dim
                ):
                    raise ValueError(f"invalid ring token_slices={token_slices.tolist()}")
                dual_base_token_dim = int(token_slices[1])
                ring_token_dim = base_token_dim - dual_base_token_dim
                if memberships_key not in cache.files:
                    raise KeyError("ring dual fusion requires ring_memberships in token cache")
                ring_memberships_raw = np.asarray(cache[memberships_key], dtype=np.int16)
                if ring_memberships_raw.shape != (token_raw.shape[0],):
                    raise ValueError("ring_memberships row count mismatch")
            if token_raw.shape[0] != int(offsets[-1]):
                raise ValueError("token cache row count does not match graph node offsets")
            if args.token_fusion in unrolled_fusions:
                if "correlations_ksvd" not in cache.files or "dictionary_ksvd" not in cache.files:
                    raise KeyError(
                        "unrolled KSVD fusion requires correlations_ksvd and dictionary_ksvd"
                    )
                unrolled_correlations_raw = np.asarray(
                    cache["correlations_ksvd"], dtype=np.float32
                )
                unrolled_dictionary_raw = np.asarray(
                    cache["dictionary_ksvd"], dtype=np.float32
                )
                if unrolled_correlations_raw.shape != token_raw.shape:
                    raise ValueError("KSVD correlation shape must match raw code shape")
                if unrolled_dictionary_raw.shape[1] != base_token_dim:
                    raise ValueError("KSVD dictionary atom count must match code dimension")
                # Selection requires sufficient statistics for every official-train
                # graph. Official-valid may remain untouched in --selection-only.
                for i in tr:
                    lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
                    if not np.any(unrolled_correlations_raw[lo:hi] != 0):
                        raise ValueError(f"missing unrolled patch correlations for train graph {i}")
                if not args.selection_only:
                    for i in va:
                        lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
                        if not np.any(unrolled_correlations_raw[lo:hi] != 0):
                            raise ValueError(
                                "full-train evaluation requires post-freeze official-valid correlations"
                            )
            test_graphs_with_codes = 0
            for i in te:
                lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
                test_graphs_with_codes += int(np.any(token_raw[lo:hi] != 0))
            if args.evaluate_test:
                if test_graphs_with_codes != len(te):
                    raise ValueError(
                        "--evaluate-test requires frozen-cache codes for every official-test graph"
                    )
            elif test_graphs_with_codes != 0:
                raise ValueError("token cache contains encoded official-test nodes")
    if token_raw is not None and args.token_fusion in {
        "motif_slot_shuffled_id", "motif_slot_graph_shuffled_id",
        "motif_slot_recurrent_shuffled_id",
        "motif_occurrence_shuffled_id", "motif_ego_shuffled_id",
        "ksvd_cell_recurrent_shuffled_id",
        "ksvd_witness_cell_readout_shuffled_id",
        "motif_sparse_ego_shuffled_id",
    }:
        # Destroy cross-graph dictionary-atom identity while preserving each
        # graph's exact sparse support cardinalities, coefficients, and generic
        # atom<->slot transport topology.
        token_raw = token_raw.copy()
        for graph_i in range(len(graphs)):
            lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
            permutation = np.random.default_rng(
                args.motif_slot_shuffle_seed + 1000003 * graph_i
            ).permutation(base_token_dim)
            token_raw[lo:hi] = token_raw[lo:hi, permutation]
    log(f"validated token cache {cache_path}; token_dim={token_dim}")

    def build_motif_occurrence_ids(
        graph_list, graph_offsets, sparse_codes, radius: int
    ):
        """Split each (graph, dictionary atom) support into local components.

        Active center nodes are adjacent in the lifted incidence graph exactly
        when their molecular graph distance is at most ``radius``.  Component
        IDs are global only for convenient caching; each mini-batch compacts
        them before scatter aggregation.
        """
        occurrence_ids = np.full(sparse_codes.shape, -1, dtype=np.int64)
        next_occurrence = 0
        occurrence_sizes: list[int] = []
        occurrences_per_graph: list[int] = []

        for graph_i, graph in enumerate(graph_list):
            lo, hi = int(graph_offsets[graph_i]), int(graph_offsets[graph_i + 1])
            support = sparse_codes[lo:hi] != 0
            active = np.argwhere(support)
            n_active = int(active.shape[0])
            if n_active == 0:
                occurrences_per_graph.append(0)
                continue

            # Compact union-find over only OMP-active (node, atom) incidences.
            position = np.full(support.shape, -1, dtype=np.int32)
            position[active[:, 0], active[:, 1]] = np.arange(
                n_active, dtype=np.int32
            )
            parent = np.arange(n_active, dtype=np.int32)
            component_size = np.ones(n_active, dtype=np.int32)

            def find(x: int) -> int:
                while int(parent[x]) != x:
                    parent[x] = parent[int(parent[x])]
                    x = int(parent[x])
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                if int(component_size[ra]) < int(component_size[rb]):
                    ra, rb = rb, ra
                parent[rb] = ra
                component_size[ra] += component_size[rb]

            for u in range(graph.n):
                reachable = set(graph.neighbors(u))
                if radius == 2:
                    for middle in tuple(reachable):
                        reachable.update(graph.neighbors(middle))
                    reachable.discard(u)
                for v in reachable:
                    if v <= u:
                        continue
                    shared_atoms = np.flatnonzero(support[u] & support[v])
                    for atom_j in shared_atoms.tolist():
                        union(
                            int(position[u, atom_j]),
                            int(position[v, atom_j]),
                        )

            root_to_occurrence: dict[int, int] = {}
            local_sizes: dict[int, int] = {}
            for incidence_i, (node_i, atom_j) in enumerate(active.tolist()):
                root = find(incidence_i)
                occurrence = root_to_occurrence.get(root)
                if occurrence is None:
                    occurrence = next_occurrence
                    root_to_occurrence[root] = occurrence
                    local_sizes[root] = 0
                    next_occurrence += 1
                occurrence_ids[lo + node_i, atom_j] = occurrence
                local_sizes[root] += 1
            occurrence_sizes.extend(local_sizes.values())
            occurrences_per_graph.append(len(root_to_occurrence))

        active_mask = sparse_codes != 0
        if np.any(occurrence_ids[active_mask] < 0):
            raise RuntimeError("some active sparse-code incidences lack occurrence IDs")
        if np.any(occurrence_ids[~active_mask] >= 0):
            raise RuntimeError("inactive sparse-code entries received occurrence IDs")
        sizes = np.asarray(occurrence_sizes, dtype=np.int64)
        graph_counts = np.asarray(occurrences_per_graph, dtype=np.int64)
        stats = {
            "radius": int(radius),
            "n_occurrences": int(len(sizes)),
            "n_incidences": int(active_mask.sum()),
            "singleton_fraction": float(np.mean(sizes == 1)) if len(sizes) else 0.0,
            "multi_node_fraction": float(np.mean(sizes > 1)) if len(sizes) else 0.0,
            "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
            "median_size": float(np.median(sizes)) if len(sizes) else 0.0,
            "max_size": int(sizes.max()) if len(sizes) else 0,
            "mean_occurrences_per_graph": (
                float(graph_counts.mean()) if len(graph_counts) else 0.0
            ),
            "max_occurrences_per_graph": (
                int(graph_counts.max()) if len(graph_counts) else 0
            ),
        }
        return occurrence_ids, stats

    motif_occurrence_stats = None
    motif_occurrence_tensor = None
    if args.token_fusion in motif_occurrence_fusions:
        if token_raw is None:
            raise ValueError("motif occurrence transport requires raw sparse codes")
        occurrence_ids_raw, motif_occurrence_stats = build_motif_occurrence_ids(
            graphs, offsets, token_raw, args.motif_occurrence_radius
        )
        motif_occurrence_tensor = torch.tensor(
            occurrence_ids_raw, dtype=torch.long, device=device
        )
        log(f"motif occurrence audit: {motif_occurrence_stats}")

    ring_membership_tensor = (
        torch.tensor(ring_memberships_raw > 0, dtype=torch.float32, device=device)
        if ring_memberships_raw is not None
        else None
    )
    token_support_tensor = (
        torch.tensor(token_raw != 0, dtype=torch.float32, device=device)
        if token_raw is not None and args.token_interaction_support_mask
        else None
    )
    raw_token_tensor = (
        torch.tensor(token_raw, dtype=torch.float32, device=device)
        if (
            unrolled_correlations_raw is not None
            or args.token_fusion in {
                "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
                "ksvd_code_residual_rich_recon",
                "ksvd_code_residual_mil_novelty",
                "ksvd_atom_additive_readout",
                "ksvd_atom_specific_readout", "motif_transition_readout",
                "motif_geometry_transition_readout", "motif_transition_matrix_readout",
                "ksvd_jk_router", "ksvd_aux_support", "ksvd_aux_code",
            }
            or args.token_fusion in motif_topology_fusions
        )
        else None
    )
    unrolled_correlation_tensor = (
        torch.tensor(unrolled_correlations_raw, dtype=torch.float32, device=device)
        if unrolled_correlations_raw is not None
        else None
    )
    unrolled_dictionary_tensor = (
        torch.tensor(unrolled_dictionary_raw, dtype=torch.float32, device=device)
        if unrolled_dictionary_raw is not None
        else None
    )
    motif_atom_features_tensor = None
    if motif_dictionary_raw is not None:
        atom_matrix = motif_dictionary_raw.T.astype(np.float64, copy=False)
        u, singular_values, _ = np.linalg.svd(atom_matrix, full_matrices=False)
        rank = min(args.motif_slot_rank, u.shape[1])
        atom_features = u[:, :rank] * singular_values[:rank]
        if args.motif_geometry_atom_normalization == "row":
            atom_norm = np.linalg.norm(atom_features, axis=1, keepdims=True)
            atom_features = (
                atom_features / np.maximum(atom_norm, 1e-12) * np.sqrt(rank)
            )
        if rank < args.motif_slot_rank:
            atom_features = np.pad(
                atom_features, ((0, 0), (0, args.motif_slot_rank - rank))
            )
        motif_atom_features_tensor = torch.tensor(
            atom_features, dtype=torch.float32, device=device
        )
        log(
            "using fixed dictionary-SVD motif IDs; "
            f"shape={tuple(motif_atom_features_tensor.shape)}"
        )

    # Build model objects for train/valid only during exploration, and include
    # official test only for an explicitly frozen final evaluation.
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    evaluator = Evaluator(name="ogbg-molhiv")
    keep = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    needed_indices = np.concatenate([tr, va, te]) if args.evaluate_test else np.concatenate([tr, va])
    needed = set(needed_indices.tolist())
    data_by_idx: dict[int, Data] = {}
    for new_i in sorted(needed):
        data = pyg[int(keep[new_i])].clone()
        expected_nodes = int(offsets[new_i + 1] - offsets[new_i])
        if int(data.num_nodes) != expected_nodes:
            raise ValueError(f"node count mismatch for graph {new_i}")
        data.idx = torch.tensor([new_i], dtype=torch.long)
        data.y = data.y.view(-1).float()
        # PyG concatenates this node-aligned tensor without index increments.
        data.token_row = torch.arange(
            int(offsets[new_i]), int(offsets[new_i + 1]), dtype=torch.long
        )
        if args.token_fusion in augmented_edge_fusions:
            if token_raw is None:
                raise ValueError("KSVD augmented edges require sparse codes")
            lo, hi = int(offsets[new_i]), int(offsets[new_i + 1])
            local_codes = token_raw[lo:hi, :base_token_dim]
            top_atom = np.abs(local_codes).argmax(axis=1)
            original_pairs = set(map(tuple, data.edge_index.t().tolist()))
            motif_src: list[int] = []
            motif_dst: list[int] = []
            motif_atom: list[int] = []
            for atom_j in range(base_token_dim):
                members = np.flatnonzero(top_atom == atom_j).tolist()
                for src in members:
                    for dst in members:
                        if src == dst or (src, dst) in original_pairs:
                            continue
                        motif_src.append(src)
                        motif_dst.append(dst)
                        motif_atom.append(atom_j)
            n_original_edges = int(data.edge_index.shape[1])
            if motif_src:
                added_index = torch.tensor(
                    [motif_src, motif_dst], dtype=data.edge_index.dtype
                )
                data.edge_index = torch.cat([data.edge_index, added_index], dim=1)
                added_attr = torch.zeros(
                    (len(motif_src), data.edge_attr.shape[1]),
                    dtype=data.edge_attr.dtype,
                )
                data.edge_attr = torch.cat([data.edge_attr, added_attr], dim=0)
            data.motif_edge_mask = torch.cat([
                torch.zeros(n_original_edges, dtype=torch.bool),
                torch.ones(len(motif_src), dtype=torch.bool),
            ])
            data.motif_edge_atom = torch.cat([
                torch.zeros(n_original_edges, dtype=torch.long),
                torch.tensor(motif_atom, dtype=torch.long),
            ])
        if args.token_fusion in (
            motif_ego_fusions | learned_cell_fusions | witness_cell_readout_fusions
        ):
            graph = graphs[new_i]
            member_nodes: list[int] = []
            occurrence_centers: list[int] = []
            occurrence_distances: list[int] = []
            for center in range(graph.n):
                members = {center}
                frontier = {center}
                distance = {center: 0}
                for hop in range(1, args.motif_ego_radius + 1):
                    frontier = {
                        neighbor
                        for node in frontier
                        for neighbor in graph.neighbors(node)
                    } - members
                    for member in frontier:
                        distance[member] = hop
                    members.update(frontier)
                for member in sorted(members):
                    member_nodes.append(member)
                    occurrence_centers.append(center)
                    occurrence_distances.append(distance[member])
            data.motif_ego_edge_index = torch.tensor(
                [member_nodes, occurrence_centers], dtype=torch.long
            )
            data.motif_ego_distance = torch.tensor(
                occurrence_distances, dtype=torch.long
            )
        data_by_idx[new_i] = data

    class GINEStack(nn.Module):
        def __init__(self, hidden: int, layers: int):
            super().__init__()
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            # Construct readout-only parameters after the complete legacy stack,
            # preserving all old convolution/BN tensors for paired same-seed runs.
            if args.jk_readout == "gated_sum":
                self.jk_gates = nn.Parameter(torch.zeros(max(layers - 1, 0)))
            if args.graph_readout == "gated_sum_mean":
                self.sum_mean_gate = nn.Parameter(torch.zeros(()))

        def pool_graph(self, x, batch):
            mean_h = global_mean_pool(x, batch)
            if args.graph_readout == "mean":
                return mean_h
            sum_h = global_add_pool(x, batch)
            counts = global_add_pool(
                x.new_ones((x.shape[0], 1)), batch
            ).clamp_min(1.0)
            normalized_sum_h = sum_h / counts.sqrt()
            return mean_h + torch.tanh(self.sum_mean_gate) * (
                normalized_sum_h - mean_h
            )

        def forward(
            self,
            data,
            token_emb=None,
            token_gates=None,
            token_routers=None,
            interaction_emb=None,
            interaction_gates=None,
            refined_emb=None,
            refined_gates=None,
            context_emb=None,
            context_gates=None,
            dynamic_token_factor=None,
            dynamic_state_projs=None,
            dynamic_out_projs=None,
            ring_emb=None,
            ring_gates=None,
            dynamic_jk_gates=None,
            motif_slot_module=None,
            motif_occurrence_module=None,
            motif_ego_module=None,
            learned_cell_module=None,
            motif_codes=None,
            motif_occurrence_ids=None,
            motif_ego_edge_index=None,
            motif_ego_distance=None,
            motif_edge_encoder=None,
        ):
            x = self.atom_encoder(data.x)
            edge_attr = self.bond_encoder(data.edge_attr)
            learned_cell_state = None
            if motif_edge_encoder is not None:
                motif_mask = data.motif_edge_mask
                if bool(motif_mask.any()):
                    edge_attr = edge_attr.clone()
                    edge_attr[motif_mask] = motif_edge_encoder(
                        data.motif_edge_atom[motif_mask]
                    )
            layer_graph_states = []
            for layer, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                inject_here = (
                    token_emb is not None
                    and (args.token_inject_layers == 0 or layer < args.token_inject_layers)
                )
                if inject_here:
                    if token_routers is not None:
                        # A zero-initialized, node-specific residual router.  The
                        # global scalar gate keeps the current model as an exact
                        # nested special case, while the router can learn which
                        # atoms should consume a KSVD token at each layer.
                        local_gate = token_routers[layer](
                            torch.cat([x, token_emb], dim=-1)
                        )
                        effective_gate = torch.tanh(
                            token_gates[layer] + local_gate
                        )
                    else:
                        effective_gate = torch.tanh(token_gates[layer])
                    x = x + args.token_gate_scale * effective_gate * token_emb
                    if interaction_emb is not None:
                        x = x + args.token_gate_scale * torch.tanh(
                            interaction_gates[layer]
                        ) * interaction_emb
                    if refined_emb is not None:
                        x = x + args.token_gate_scale * torch.tanh(
                            refined_gates[layer]
                        ) * refined_emb
                    if context_emb is not None:
                        x = x + args.context_gate_scale * torch.tanh(
                            context_gates[layer]
                        ) * context_emb[data.batch]
                    if dynamic_token_factor is not None:
                        state_factor = torch.tanh(dynamic_state_projs[layer](x))
                        dynamic_emb = dynamic_out_projs[layer](
                            dynamic_token_factor * state_factor
                        )
                        dynamic_emb = F.dropout(
                            dynamic_emb,
                            p=args.token_dropout,
                            training=self.training,
                        )
                        x = x + args.token_gate_scale * torch.tanh(
                            interaction_gates[layer]
                        ) * dynamic_emb
                    if ring_emb is not None:
                        x = x + args.ring_gate_scale * torch.tanh(ring_gates[layer]) * ring_emb
                previous_x = x
                x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
                x = F.dropout(x, p=args.dropout, training=self.training)
                if args.residual_gine:
                    x = x + previous_x
                if learned_cell_module is not None:
                    if (
                        motif_codes is None
                        or motif_ego_edge_index is None
                        or motif_ego_distance is None
                    ):
                        raise ValueError(
                            "learned cell module requires sparse codes, ego "
                            "incidence, and incidence distance"
                        )
                    cell_message, learned_cell_state = learned_cell_module(
                        x,
                        motif_codes,
                        motif_ego_edge_index,
                        motif_ego_distance,
                        learned_cell_state,
                    )
                    motif_scale = (
                        1.0 if args.motif_gate_mode == "zero_out"
                        else torch.tanh(learned_cell_module.gate)
                    )
                    x = x + args.token_gate_scale * motif_scale * cell_message
                if (
                    motif_slot_module is not None
                    and (layer == 0 or args.token_fusion in motif_slot_recurrent_fusions)
                ):
                    if motif_codes is None:
                        raise ValueError("motif slot module requires raw sparse codes")
                    motif_scale = (
                        1.0 if args.motif_gate_mode in {"zero_out", "atomwise"}
                        else torch.tanh(motif_slot_module.gate)
                    )
                    x = x + args.token_gate_scale * motif_scale * motif_slot_module(
                        x, data.batch, motif_codes, data.edge_index
                    )
                if layer == 0 and motif_occurrence_module is not None:
                    if motif_codes is None or motif_occurrence_ids is None:
                        raise ValueError(
                            "motif occurrence module requires sparse codes and component IDs"
                        )
                    motif_scale = (
                        1.0 if args.motif_gate_mode == "zero_out"
                        else torch.tanh(motif_occurrence_module.gate)
                    )
                    x = x + args.token_gate_scale * motif_scale * (
                        motif_occurrence_module(
                            x, motif_codes, motif_occurrence_ids
                        )
                    )
                if layer == 0 and motif_ego_module is not None:
                    if motif_codes is None or motif_ego_edge_index is None:
                        raise ValueError(
                            "motif ego module requires sparse codes and ego incidence"
                        )
                    motif_scale = (
                        1.0 if args.motif_gate_mode == "zero_out"
                        else torch.tanh(motif_ego_module.gate)
                    )
                    x = x + args.token_gate_scale * motif_scale * motif_ego_module(
                        x, motif_codes, motif_ego_edge_index
                    )
                layer_graph_states.append(self.pool_graph(x, data.batch))
            graph_h = layer_graph_states[-1]
            if args.jk_readout == "gated_sum":
                for layer, (gate, earlier_h) in enumerate(
                    zip(self.jk_gates, layer_graph_states[:-1])
                ):
                    if dynamic_jk_gates is None:
                        effective_gate = torch.tanh(gate)
                    else:
                        effective_gate = torch.tanh(
                            gate + dynamic_jk_gates[:, layer]
                        ).unsqueeze(-1)
                    graph_h = graph_h + effective_gate * earlier_h
            return graph_h, x

    class MotifSlotTransport(nn.Module):
        """One atom->learned-slot->atom message pass over sparse-code incidence."""

        def __init__(
            self, n_slots: int, hidden: int, rank: int, use_atom_id: bool,
            atom_features=None, use_transition_graph: bool = False,
        ):
            super().__init__()
            self.n_slots = n_slots
            self.rank = rank
            self.use_atom_id = use_atom_id
            self.use_transition_graph = use_transition_graph
            if use_atom_id:
                if atom_features is None:
                    self.atom_id = nn.Embedding(n_slots, rank)
                    self.register_buffer("fixed_atom_id", None)
                else:
                    self.atom_id = None
                    self.register_buffer("fixed_atom_id", atom_features.clone())
            self.slot_update = nn.Sequential(
                nn.Linear(hidden + rank + 2, rank),
                nn.ReLU(),
                nn.Linear(rank, hidden),
            )
            if use_transition_graph:
                # Learn on the graph quotient induced by KSVD assignments:
                # molecular edges vote for directed dictionary-atom transitions.
                # Shared weights keep capacity low while each slot remains a
                # self-learned motif type rather than a hand-specified cell.
                self.transition_update = nn.Sequential(
                    nn.Linear(2 * hidden, rank),
                    nn.ReLU(),
                    nn.Linear(rank, hidden),
                )
            else:
                self.transition_update = None
            self.broadcast = nn.Sequential(
                nn.Linear(hidden, rank, bias=False),
                nn.ReLU(),
                nn.Linear(rank, hidden, bias=False),
            )
            # Exact nesting: same-seed initialization predicts exactly as GINE.
            self.gate = nn.Parameter(
                torch.zeros(()), requires_grad=(args.motif_gate_mode == "scalar")
            )
            if args.motif_gate_mode == "atomwise":
                self.atom_gates = nn.Parameter(torch.zeros(n_slots))
            else:
                self.register_parameter("atom_gates", None)
            if args.motif_gate_mode == "zero_out":
                nn.init.zeros_(self.broadcast[-1].weight)

        def mix_transition_graph(
            self, slot_base, node_weights, batch, edge_index
        ):
            if not self.use_transition_graph:
                return slot_base
            if edge_index is None:
                raise ValueError("motif slot graph requires molecular edge_index")
            n_graphs = slot_base.shape[0]
            src, dst = edge_index
            if src.numel() == 0:
                neighbor = torch.zeros_like(slot_base)
            else:
                # Q^T A Q: a soft quotient graph over learned dictionary atoms.
                # OGB molecular edges are batched without cross-graph edges.
                edge_outer = (
                    node_weights[src].unsqueeze(-1)
                    * node_weights[dst].unsqueeze(-2)
                )
                transition = slot_base.new_zeros(
                    (n_graphs, self.n_slots, self.n_slots)
                )
                transition.index_add_(0, batch[src], edge_outer)
                transition = transition / transition.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                neighbor = torch.bmm(transition, slot_base)
            return slot_base + self.transition_update(
                torch.cat([slot_base, neighbor], dim=-1)
            )

        def forward(self, node_h, batch, raw_codes, edge_index=None):
            weights = raw_codes.abs()
            node_weights = weights / weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            if args.motif_slot_backend == "dense":
                support = (raw_codes != 0).to(node_h.dtype)
                mass = global_add_pool(weights, batch)
                count = global_add_pool(support, batch)
                weighted_states = (
                    weights.unsqueeze(-1) * node_h.unsqueeze(1)
                ).reshape(node_h.shape[0], self.n_slots * node_h.shape[1])
                slot_sum = global_add_pool(weighted_states, batch).reshape(
                    -1, self.n_slots, node_h.shape[1]
                )
                pooled = slot_sum / mass.clamp_min(1e-6).unsqueeze(-1)
                signed_mass = global_add_pool(weights * raw_codes.sign(), batch)
                sign_mean = signed_mass / mass.clamp_min(1e-6)
                mean_abs = mass / count.clamp_min(1.0)
                if self.use_atom_id:
                    atom_table = (
                        self.atom_id.weight
                        if self.atom_id is not None else self.fixed_atom_id
                    )
                    ids = atom_table.unsqueeze(0).expand(pooled.shape[0], -1, -1)
                else:
                    ids = pooled.new_zeros((pooled.shape[0], self.n_slots, self.rank))
                slot_input = torch.cat(
                    [pooled, ids, sign_mean.unsqueeze(-1), mean_abs.unsqueeze(-1)],
                    dim=-1,
                )
                slot_base = pooled + self.slot_update(slot_input)
                slot_base = self.mix_transition_graph(
                    slot_base, node_weights, batch, edge_index
                )
                slot_h = self.broadcast(slot_base)
                if self.atom_gates is not None:
                    slot_h = slot_h * torch.tanh(self.atom_gates).view(1, -1, 1)
                message = (
                    node_weights.unsqueeze(-1) * slot_h[batch]
                ).sum(dim=1)
                active = (weights.sum(dim=-1, keepdim=True) > 0).to(message.dtype)
                return F.dropout(
                    message * active, p=args.token_dropout, training=self.training
                )
            # OMP has only T=3 nonzeros per node. Work directly on those
            # incidences instead of materializing an N x D x H dense tensor.
            incidence = torch.nonzero(raw_codes, as_tuple=False)
            if incidence.numel() == 0:
                return torch.zeros_like(node_h)
            node_index, slot_index = incidence.unbind(dim=1)
            coefficient = raw_codes[node_index, slot_index]
            weight = coefficient.abs()
            n_graphs = int(batch.max().item()) + 1
            flat_slot = batch[node_index] * self.n_slots + slot_index
            n_flat_slots = n_graphs * self.n_slots
            mass = node_h.new_zeros(n_flat_slots)
            mass.index_add_(0, flat_slot, weight)
            count = node_h.new_zeros(n_flat_slots)
            count.index_add_(0, flat_slot, torch.ones_like(weight))
            slot_sum = node_h.new_zeros((n_flat_slots, node_h.shape[1]))
            slot_sum.index_add_(0, flat_slot, weight.unsqueeze(-1) * node_h[node_index])
            pooled = slot_sum / mass.clamp_min(1e-6).unsqueeze(-1)
            signed_mass = node_h.new_zeros(n_flat_slots)
            signed_mass.index_add_(0, flat_slot, weight * coefficient.sign())
            sign_mean = signed_mass / mass.clamp_min(1e-6)
            mean_abs = mass / count.clamp_min(1.0)
            flat_slot_ids = torch.arange(n_flat_slots, device=node_h.device) % self.n_slots
            if self.use_atom_id:
                ids = (
                    self.atom_id(flat_slot_ids)
                    if self.atom_id is not None
                    else self.fixed_atom_id[flat_slot_ids]
                )
            else:
                ids = pooled.new_zeros((n_flat_slots, self.rank))
            slot_input = torch.cat(
                [pooled, ids, sign_mean.unsqueeze(-1), mean_abs.unsqueeze(-1)], dim=-1
            )
            slot_base = pooled + self.slot_update(slot_input)
            slot_base = slot_base.reshape(n_graphs, self.n_slots, -1)
            slot_base = self.mix_transition_graph(
                slot_base, node_weights, batch, edge_index
            )
            slot_h = self.broadcast(slot_base).reshape(n_flat_slots, -1)
            if self.atom_gates is not None:
                slot_h = slot_h * torch.tanh(
                    self.atom_gates[flat_slot_ids]
                ).unsqueeze(-1)
            node_mass = node_h.new_zeros(node_h.shape[0])
            node_mass.index_add_(0, node_index, weight)
            normalized_weight = weight / node_mass[node_index].clamp_min(1e-6)
            message = torch.zeros_like(node_h)
            message.index_add_(
                0, node_index, normalized_weight.unsqueeze(-1) * slot_h[flat_slot]
            )
            return F.dropout(message, p=args.token_dropout, training=self.training)

    class MotifOccurrenceTransport(nn.Module):
        """Atom->local learned-motif-occurrence->atom message transport."""

        def __init__(
            self, n_atoms: int, hidden: int, rank: int, use_atom_id: bool,
            atom_features=None,
        ):
            super().__init__()
            self.n_atoms = n_atoms
            self.rank = rank
            self.use_atom_id = use_atom_id
            if use_atom_id:
                if atom_features is None:
                    self.atom_id = nn.Embedding(n_atoms, rank)
                    self.register_buffer("fixed_atom_id", None)
                else:
                    self.atom_id = None
                    self.register_buffer("fixed_atom_id", atom_features.clone())
            # Parameterization is intentionally matched to P0 motif slots.
            self.occurrence_update = nn.Sequential(
                nn.Linear(hidden + rank + 2, rank),
                nn.ReLU(),
                nn.Linear(rank, hidden),
            )
            self.broadcast = nn.Sequential(
                nn.Linear(hidden, rank, bias=False),
                nn.ReLU(),
                nn.Linear(rank, hidden, bias=False),
            )
            self.gate = nn.Parameter(
                torch.zeros(()), requires_grad=(args.motif_gate_mode == "scalar")
            )
            if args.motif_gate_mode == "zero_out":
                nn.init.zeros_(self.broadcast[-1].weight)

        def forward(self, node_h, raw_codes, occurrence_ids):
            incidence = torch.nonzero(raw_codes, as_tuple=False)
            if incidence.numel() == 0:
                return torch.zeros_like(node_h)
            node_index, atom_index = incidence.unbind(dim=1)
            global_occurrence = occurrence_ids[node_index, atom_index]
            if bool((global_occurrence < 0).any()):
                raise RuntimeError("active motif incidence has no occurrence ID")
            _, compact_occurrence = torch.unique(
                global_occurrence, sorted=True, return_inverse=True
            )
            n_occurrences = int(compact_occurrence.max().item()) + 1

            coefficient = raw_codes[node_index, atom_index]
            weight = coefficient.abs()
            mass = node_h.new_zeros(n_occurrences)
            mass.index_add_(0, compact_occurrence, weight)
            count = node_h.new_zeros(n_occurrences)
            count.index_add_(0, compact_occurrence, torch.ones_like(weight))
            state_sum = node_h.new_zeros((n_occurrences, node_h.shape[1]))
            state_sum.index_add_(
                0, compact_occurrence, weight.unsqueeze(-1) * node_h[node_index]
            )
            pooled = state_sum / mass.clamp_min(1e-6).unsqueeze(-1)
            signed_mass = node_h.new_zeros(n_occurrences)
            signed_mass.index_add_(
                0, compact_occurrence, weight * coefficient.sign()
            )
            sign_mean = signed_mass / mass.clamp_min(1e-6)
            mean_abs = mass / count.clamp_min(1.0)

            # Every component is type-homogeneous by construction.  amin is a
            # deterministic assertion-friendly way to recover its atom type.
            occurrence_atom = torch.full(
                (n_occurrences,), self.n_atoms, dtype=torch.long,
                device=node_h.device,
            )
            occurrence_atom.scatter_reduce_(
                0, compact_occurrence, atom_index, reduce="amin", include_self=True
            )
            if bool((occurrence_atom >= self.n_atoms).any()):
                raise RuntimeError("empty motif occurrence after compaction")
            if self.use_atom_id:
                ids = (
                    self.atom_id(occurrence_atom)
                    if self.atom_id is not None
                    else self.fixed_atom_id[occurrence_atom]
                )
            else:
                ids = pooled.new_zeros((n_occurrences, self.rank))
            occurrence_input = torch.cat(
                [pooled, ids, sign_mean.unsqueeze(-1), mean_abs.unsqueeze(-1)],
                dim=-1,
            )
            occurrence_h = self.broadcast(
                pooled + self.occurrence_update(occurrence_input)
            )

            node_mass = node_h.new_zeros(node_h.shape[0])
            node_mass.index_add_(0, node_index, weight)
            normalized_weight = weight / node_mass[node_index].clamp_min(1e-6)
            message = torch.zeros_like(node_h)
            message.index_add_(
                0, node_index,
                normalized_weight.unsqueeze(-1)
                * occurrence_h[compact_occurrence],
            )
            return F.dropout(
                message, p=args.token_dropout, training=self.training
            )

    class MotifEgoTransport(nn.Module):
        """Lift each centered KSVD patch into a typed motif occurrence."""

        def __init__(
            self, n_atoms: int, hidden: int, rank: int, use_atom_id: bool,
            atom_features=None, all_sparse_atoms: bool = False,
        ):
            super().__init__()
            self.n_atoms = n_atoms
            self.rank = rank
            self.use_atom_id = use_atom_id
            self.all_sparse_atoms = all_sparse_atoms
            if use_atom_id:
                if atom_features is None:
                    self.atom_id = nn.Embedding(n_atoms, rank)
                    self.register_buffer("fixed_atom_id", None)
                else:
                    self.atom_id = None
                    self.register_buffer("fixed_atom_id", atom_features.clone())
            self.occurrence_update = nn.Sequential(
                nn.Linear(hidden + rank + 2, rank),
                nn.ReLU(),
                nn.Linear(rank, hidden),
            )
            self.broadcast = nn.Sequential(
                nn.Linear(hidden, rank, bias=False),
                nn.ReLU(),
                nn.Linear(rank, hidden, bias=False),
            )
            self.gate = nn.Parameter(
                torch.zeros(()), requires_grad=(args.motif_gate_mode == "scalar")
            )
            if args.motif_gate_mode == "zero_out":
                nn.init.zeros_(self.broadcast[-1].weight)

        def forward(self, node_h, raw_codes, ego_edge_index):
            member_index, center_index = ego_edge_index
            if center_index.numel() == 0:
                return torch.zeros_like(node_h)

            if self.all_sparse_atoms:
                # Every active OMP coefficient defines one typed occurrence on
                # its center's molecular ego.  This preserves all T sparse
                # assignments instead of discarding everything but top-1.
                incidence = torch.nonzero(raw_codes, as_tuple=False)
                if incidence.numel() == 0:
                    return torch.zeros_like(node_h)
                occurrence_center, occurrence_atom = incidence.unbind(dim=1)
                n_occurrences = int(incidence.shape[0])
                occurrence_coefficient = raw_codes[
                    occurrence_center, occurrence_atom
                ]
                occurrence_map = torch.full(
                    raw_codes.shape, -1, dtype=torch.long, device=node_h.device
                )
                occurrence_map[occurrence_center, occurrence_atom] = torch.arange(
                    n_occurrences, device=node_h.device
                )

                # Expand each center-ego membership to the center's active
                # dictionary atoms.  With OMP T=3 this is exactly 3x the
                # ordinary centered-ego incidence, still sparse and local.
                edge_position, edge_atom = torch.nonzero(
                    raw_codes[center_index] != 0, as_tuple=False
                ).unbind(dim=1)
                expanded_member = member_index[edge_position]
                expanded_center = center_index[edge_position]
                expanded_occurrence = occurrence_map[
                    expanded_center, edge_atom
                ]
                if bool((expanded_occurrence < 0).any()):
                    raise RuntimeError("sparse ego incidence lost an active atom")

                member_sum = node_h.new_zeros(
                    (n_occurrences, node_h.shape[1])
                )
                member_sum.index_add_(
                    0, expanded_occurrence, node_h[expanded_member]
                )
                member_count = node_h.new_zeros(n_occurrences)
                member_count.index_add_(
                    0, expanded_occurrence,
                    node_h.new_ones(expanded_occurrence.shape[0]),
                )
                pooled = member_sum / member_count.clamp_min(1.0).unsqueeze(-1)

                if self.use_atom_id:
                    ids = (
                        self.atom_id(occurrence_atom)
                        if self.atom_id is not None
                        else self.fixed_atom_id[occurrence_atom]
                    )
                else:
                    ids = pooled.new_zeros((n_occurrences, self.rank))
                occurrence_input = torch.cat(
                    [
                        pooled, ids,
                        occurrence_coefficient.sign().unsqueeze(-1),
                        occurrence_coefficient.abs().unsqueeze(-1),
                    ],
                    dim=-1,
                )
                occurrence_h = self.broadcast(
                    pooled + self.occurrence_update(occurrence_input)
                )

                # Coefficient magnitude controls how strongly an occurrence
                # contributes to each member atom; normalize competing cells at
                # the receiver to prevent degree/ego-size scale leakage.
                expanded_weight = occurrence_coefficient[
                    expanded_occurrence
                ].abs()
                message = torch.zeros_like(node_h)
                message.index_add_(
                    0, expanded_member,
                    expanded_weight.unsqueeze(-1)
                    * occurrence_h[expanded_occurrence],
                )
                receiving_mass = node_h.new_zeros(node_h.shape[0])
                receiving_mass.index_add_(
                    0, expanded_member, expanded_weight
                )
                message = message / receiving_mass.clamp_min(1e-6).unsqueeze(-1)
                return F.dropout(
                    message, p=args.token_dropout, training=self.training
                )

            # Top-1 centered ego implementation. Optional atom-index and
            # dominance filters can retain a sparse learned-cell subcomplex
            # (for example, only the OOF-witness half of a dual-bank cache).
            n_occurrences = node_h.shape[0]
            if int(center_index.max().item()) >= n_occurrences:
                raise RuntimeError("batched ego occurrence index exceeds node count")

            absolute_codes = raw_codes.abs()
            absolute_mass = absolute_codes.sum(dim=-1)
            top_atom = absolute_codes.argmax(dim=-1)
            top_coefficient = raw_codes.gather(
                1, top_atom.unsqueeze(-1)
            ).squeeze(-1)
            dominance = top_coefficient.abs() / absolute_mass.clamp_min(1e-6)
            selected_occurrence = (
                (absolute_mass > 0)
                & (top_atom >= args.motif_ego_min_atom_index)
                & (dominance >= args.motif_ego_min_dominance)
            )
            selected_edge = selected_occurrence[center_index]
            member_index = member_index[selected_edge]
            center_index = center_index[selected_edge]
            if center_index.numel() == 0:
                return torch.zeros_like(node_h)

            member_sum = node_h.new_zeros((n_occurrences, node_h.shape[1]))
            member_sum.index_add_(0, center_index, node_h[member_index])
            member_count = node_h.new_zeros(n_occurrences)
            member_count.index_add_(
                0, center_index, node_h.new_ones(center_index.shape[0])
            )
            pooled = member_sum / member_count.clamp_min(1.0).unsqueeze(-1)

            if args.motif_ego_confidence == "dominance":
                occurrence_confidence = dominance
            else:
                occurrence_confidence = node_h.new_ones(n_occurrences)
            if self.use_atom_id:
                ids = (
                    self.atom_id(top_atom)
                    if self.atom_id is not None
                    else self.fixed_atom_id[top_atom]
                )
            else:
                ids = pooled.new_zeros((n_occurrences, self.rank))
            occurrence_input = torch.cat(
                [
                    pooled, ids, top_coefficient.sign().unsqueeze(-1),
                    top_coefficient.abs().unsqueeze(-1),
                ],
                dim=-1,
            )
            occurrence_h = self.broadcast(
                pooled + self.occurrence_update(occurrence_input)
            )

            edge_confidence = occurrence_confidence[center_index]
            message = torch.zeros_like(node_h)
            message.index_add_(
                0, member_index,
                edge_confidence.unsqueeze(-1) * occurrence_h[center_index],
            )
            receiving_count = node_h.new_zeros(node_h.shape[0])
            receiving_count.index_add_(0, member_index, edge_confidence)
            message = message / receiving_count.clamp_min(1e-6).unsqueeze(-1)
            return F.dropout(
                message, p=args.token_dropout, training=self.training
            )

    class LearnedPatchCellTransport(nn.Module):
        """Recurrent node<->KSVD-cell transport over learned patch hyperedges.

        Every molecular atom is the center of one radius-r patch cell.  The
        cell's type is the normalized sparse assignment to dictionary atoms;
        its boundary is the center's graph-distance ego.  Alternating this
        transport after every molecular GINE layer lets overlapping learned
        cells communicate through shared boundary atoms, rather than adding a
        second ordinary edge-GNN branch.
        """

        def __init__(
            self, n_atoms: int, hidden: int, rank: int, use_atom_id: bool,
            atom_features=None,
        ):
            super().__init__()
            self.n_atoms = n_atoms
            self.rank = rank
            self.use_atom_id = use_atom_id
            if use_atom_id:
                if atom_features is None:
                    self.atom_id = nn.Embedding(n_atoms, rank)
                    self.register_buffer("fixed_atom_id", None)
                else:
                    self.atom_id = None
                    self.register_buffer("fixed_atom_id", atom_features.clone())
            input_dim = 2 * hidden + 2 * rank + 2
            self.cell_update = nn.Sequential(
                nn.Linear(input_dim, rank),
                nn.SiLU(),
                nn.Linear(rank, hidden),
            )
            self.cell_norm = nn.LayerNorm(hidden)
            self.broadcast = nn.Sequential(
                nn.Linear(hidden, rank, bias=False),
                nn.SiLU(),
                nn.Linear(rank, hidden, bias=False),
            )
            # Graph distance is structural incidence information, not a
            # molecular ring/scaffold feature.  The monotone initialization
            # keeps the center strongest while allowing training to adapt it.
            initial_distance = -0.7 * torch.arange(
                args.motif_ego_radius + 1, dtype=torch.float32
            )
            self.distance_logits = nn.Parameter(initial_distance)
            self.recurrence_logit = nn.Parameter(torch.zeros(()))
            self.gate = nn.Parameter(
                torch.zeros(()), requires_grad=(args.motif_gate_mode == "scalar")
            )
            if args.motif_gate_mode == "zero_out":
                nn.init.zeros_(self.broadcast[-1].weight)

        def forward(
            self, node_h, raw_codes, ego_edge_index, ego_distance,
            previous_cell_state,
        ):
            member_index, center_index = ego_edge_index
            n_cells = node_h.shape[0]
            if center_index.numel() == 0:
                return torch.zeros_like(node_h), torch.zeros_like(node_h)
            if int(center_index.max().item()) >= n_cells:
                raise RuntimeError("learned patch-cell center exceeds node count")

            distance_weight = torch.sigmoid(self.distance_logits)[ego_distance]
            cell_mass = node_h.new_zeros(n_cells)
            cell_mass.index_add_(0, center_index, distance_weight)
            cell_sum = node_h.new_zeros((n_cells, node_h.shape[1]))
            cell_sum.index_add_(
                0,
                center_index,
                distance_weight.unsqueeze(-1) * node_h[member_index],
            )
            pooled = cell_sum / cell_mass.clamp_min(1e-6).unsqueeze(-1)

            if previous_cell_state is None:
                previous_cell_state = torch.zeros_like(pooled)
            if self.use_atom_id:
                atom_table = (
                    self.atom_id.weight
                    if self.atom_id is not None else self.fixed_atom_id
                )
                absolute_codes = raw_codes.abs()
                code_mass = absolute_codes.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                abs_assignment = absolute_codes / code_mass
                signed_assignment = raw_codes / code_mass
                abs_id = abs_assignment @ atom_table
                signed_id = signed_assignment @ atom_table
                dominance = absolute_codes.max(dim=-1).values / code_mass.squeeze(-1)
                signed_balance = signed_assignment.sum(dim=-1)
            else:
                abs_id = pooled.new_zeros((n_cells, self.rank))
                signed_id = pooled.new_zeros((n_cells, self.rank))
                dominance = pooled.new_zeros(n_cells)
                signed_balance = pooled.new_zeros(n_cells)

            update_input = torch.cat(
                [
                    pooled,
                    previous_cell_state,
                    abs_id,
                    signed_id,
                    dominance.unsqueeze(-1),
                    signed_balance.unsqueeze(-1),
                ],
                dim=-1,
            )
            recurrence = torch.sigmoid(self.recurrence_logit)
            cell_state = self.cell_norm(
                pooled
                + recurrence * previous_cell_state
                + self.cell_update(update_input)
            )
            cell_message = self.broadcast(cell_state)

            # Incidence-normalized cell->node broadcast.  Overlapping cells
            # interact at the next alternation through their shared atoms.
            edge_confidence = distance_weight
            if self.use_atom_id:
                edge_confidence = edge_confidence * (0.5 + dominance[center_index])
            message = torch.zeros_like(node_h)
            message.index_add_(
                0,
                member_index,
                edge_confidence.unsqueeze(-1) * cell_message[center_index],
            )
            receiving_mass = node_h.new_zeros(n_cells)
            receiving_mass.index_add_(0, member_index, edge_confidence)
            message = message / receiving_mass.clamp_min(1e-6).unsqueeze(-1)
            message = F.dropout(
                message, p=args.token_dropout, training=self.training
            )
            return message, cell_state

    class WitnessPatchCellReadout(nn.Module):
        """Joint graph readout over KSVD-learned OOF-witness patch cells."""

        def __init__(
            self, n_atoms: int, hidden: int, rank: int, use_atom_id: bool,
            atom_features=None,
        ):
            super().__init__()
            self.n_atoms = n_atoms
            self.rank = rank
            self.use_atom_id = use_atom_id
            if use_atom_id:
                if atom_features is None:
                    self.atom_id = nn.Embedding(n_atoms, rank)
                    self.register_buffer("fixed_atom_id", None)
                else:
                    self.atom_id = None
                    self.register_buffer("fixed_atom_id", atom_features.clone())
            self.cell_update = nn.Sequential(
                nn.Linear(hidden + 2 * rank + 2, rank),
                nn.SiLU(),
                nn.Linear(rank, hidden),
            )
            self.cell_norm = nn.LayerNorm(hidden)
            self.readout = nn.Sequential(
                nn.Linear(2 * hidden + 2, rank),
                nn.SiLU(),
                nn.Linear(rank, hidden, bias=False),
            )
            self.distance_logits = nn.Parameter(
                -0.7 * torch.arange(
                    args.motif_ego_radius + 1, dtype=torch.float32
                )
            )
            self.gate = nn.Parameter(
                torch.zeros(()), requires_grad=(args.motif_gate_mode == "scalar")
            )
            if args.motif_gate_mode == "zero_out":
                nn.init.zeros_(self.readout[-1].weight)

        def forward(
            self, node_h, batch, raw_codes, ego_edge_index, ego_distance,
        ):
            member_index, center_index = ego_edge_index
            n_cells = node_h.shape[0]
            distance_weight = torch.sigmoid(self.distance_logits)[ego_distance]
            cell_mass = node_h.new_zeros(n_cells)
            cell_mass.index_add_(0, center_index, distance_weight)
            cell_sum = node_h.new_zeros((n_cells, node_h.shape[1]))
            cell_sum.index_add_(
                0,
                center_index,
                distance_weight.unsqueeze(-1) * node_h[member_index],
            )
            pooled = cell_sum / cell_mass.clamp_min(1e-6).unsqueeze(-1)

            absolute_codes = raw_codes.abs()
            code_mass = absolute_codes.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            # The cache is a concatenation of equal-size background and
            # OOF-witness banks.  This learned bank boundary is part of the
            # strict cache protocol, not a molecular hand feature.
            witness_start = self.n_atoms // 2
            witness_mass = (
                absolute_codes[:, witness_start:].sum(dim=-1)
                / code_mass.squeeze(-1)
            )
            if self.use_atom_id:
                atom_table = (
                    self.atom_id.weight
                    if self.atom_id is not None else self.fixed_atom_id
                )
                abs_id = (absolute_codes / code_mass) @ atom_table
                signed_assignment = raw_codes / code_mass
                signed_id = signed_assignment @ atom_table
                signed_balance = signed_assignment.sum(dim=-1)
            else:
                abs_id = pooled.new_zeros((n_cells, self.rank))
                signed_id = pooled.new_zeros((n_cells, self.rank))
                signed_balance = pooled.new_zeros(n_cells)
            cell_h = self.cell_norm(
                pooled
                + self.cell_update(
                    torch.cat(
                        [
                            pooled,
                            abs_id,
                            signed_id,
                            witness_mass.unsqueeze(-1),
                            signed_balance.unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                )
            )

            weighted_cell = witness_mass.unsqueeze(-1) * cell_h
            graph_mass = global_add_pool(
                witness_mass.unsqueeze(-1), batch
            ).clamp_min(1e-6)
            graph_mean = global_add_pool(weighted_cell, batch) / graph_mass
            graph_max = global_max_pool(weighted_cell, batch)
            n_cells_graph = global_add_pool(
                node_h.new_ones((n_cells, 1)), batch
            ).clamp_min(1.0)
            witness_fraction = graph_mass / n_cells_graph
            active_fraction = global_add_pool(
                (witness_mass > 0).to(node_h.dtype).unsqueeze(-1), batch
            ) / n_cells_graph
            graph_cell = self.readout(
                torch.cat(
                    [graph_mean, graph_max, witness_fraction, active_fraction],
                    dim=-1,
                )
            )
            return F.dropout(
                graph_cell, p=args.token_dropout, training=self.training
            )

    class TokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Construct the complete baseline first.  Thus same-seed `none` and
            # token families have exactly identical baseline tensors.
            self.gine = GINEStack(args.hidden, args.layers)
            self.head = nn.Linear(args.hidden, 1)
            self.use_tokens = args.family != "none"
            if self.use_tokens:
                self.token_proj = nn.Sequential(
                    nn.Linear(dual_base_token_dim, args.hidden),
                    nn.ReLU(),
                    nn.Linear(args.hidden, args.hidden),
                )
                self.token_gates = nn.Parameter(torch.zeros(args.layers))
                if (
                    args.token_fusion in {
                        "ksvd_code_residual_maxabs",
                        "ksvd_code_residual_rich", "ksvd_code_residual_rich_recon",
                        "ksvd_code_residual_mil_novelty",
                        "ksvd_precomputed_graph_residual",
                        "ksvd_atom_additive_readout",
                        "ksvd_atom_specific_readout", "motif_transition_readout",
                        "motif_geometry_transition_readout",
                        "motif_transition_matrix_readout", "ksvd_jk_router", "ksvd_aux_support", "ksvd_aux_code",
                    }
                    or args.token_fusion in motif_topology_fusions | augmented_edge_fusions
                ):
                    # This branch intentionally keeps the KSVD representation
                    # fixed and trains only a 32-to-1 graph residual.
                    self.token_proj.requires_grad_(False)
                    self.token_gates.requires_grad_(False)
                if args.token_fusion == "ksvd_jk_router":
                    self.jk_router = nn.Linear(
                        base_token_dim, max(args.layers - 1, 0), bias=False
                    )
                    nn.init.zeros_(self.jk_router.weight)
                if args.token_fusion in {"ksvd_aux_support", "ksvd_aux_code"}:
                    # KSVD is used as a train-only self-supervised teacher rather
                    # than injected into the predictive path. The downstream
                    # logits are therefore exactly the matched GINE/JK model at
                    # inference, while node states are regularized to retain the
                    # learned radius-2 sparse motif assignment.
                    self.ksvd_aux_head = nn.Linear(args.hidden, base_token_dim)
                    self.ksvd_aux_scale = 1.0
                if args.token_fusion == "ksvd_context_inject":
                    # Molecule-wide composition of self-learned KSVD atoms. This
                    # feeds global dictionary context back to every node without
                    # any explicit ring/scaffold annotations or extra MLP weights.
                    self.context_gates = nn.Parameter(torch.zeros(args.layers))
                # Keep this branch after the frozen baseline and ordinary token
                # projection so same-seed initialization of both remains exact.
                # Two learned views of the signed sparse code are multiplied in
                # a low-rank space, exposing atom co-activation without any
                # hand-crafted ring/scaffold labels.
                if args.token_fusion in {"bilinear_inject", "edge_bilinear_inject"}:
                    rank = args.token_interaction_rank
                    self.interaction_left = nn.Linear(
                        dual_base_token_dim, rank, bias=False
                    )
                    self.interaction_right = nn.Linear(
                        dual_base_token_dim, rank, bias=False
                    )
                    self.interaction_proj = nn.Linear(rank, args.hidden)
                    self.interaction_gates = nn.Parameter(torch.zeros(args.layers))
                elif args.token_fusion == "dynamic_bilinear_inject":
                    rank = args.token_interaction_rank
                    self.dynamic_token_proj = nn.Linear(
                        dual_base_token_dim, rank, bias=False
                    )
                    self.dynamic_state_projs = nn.ModuleList(
                        [nn.Linear(args.hidden, rank, bias=False) for _ in range(args.layers)]
                    )
                    self.dynamic_out_projs = nn.ModuleList(
                        [nn.Linear(rank, args.hidden) for _ in range(args.layers)]
                    )
                    self.interaction_gates = nn.Parameter(torch.zeros(args.layers))
                elif args.token_fusion in unrolled_fusions:
                    assert unrolled_dictionary_tensor is not None
                    D0 = unrolled_dictionary_tensor
                    D0 = D0 / D0.norm(dim=0, keepdim=True).clamp_min(1e-12)
                    gram = D0.transpose(0, 1) @ D0
                    self.register_buffer("unrolled_gram", gram)
                    lipschitz = float(torch.linalg.eigvalsh(gram).max().cpu())
                    self.unrolled_max_step = 0.99 / max(lipschitz, 1e-12)
                    if args.unrolled_step_init >= self.unrolled_max_step:
                        raise ValueError(
                            f"--unrolled-step-init must be below {self.unrolled_max_step:.6f} "
                            "for this dictionary"
                        )
                    step_ratio = args.unrolled_step_init / self.unrolled_max_step
                    step_logit = np.log(step_ratio / (1.0 - step_ratio))
                    self.unrolled_step_logits = nn.Parameter(
                        torch.full((args.unrolled_steps,), float(step_logit))
                    )
                    threshold_width = (
                        base_token_dim
                        if args.unrolled_threshold_mode == "per_atom"
                        else 1
                    )
                    lambda_raw = np.log(np.expm1(args.unrolled_lambda_init))
                    self.unrolled_lambda_raw = nn.Parameter(
                        torch.full(
                            (args.unrolled_steps, threshold_width), float(lambda_raw)
                        )
                    )
                    if args.token_fusion == "unrolled_ksvd_inject":
                        self.refined_gates = nn.Parameter(torch.zeros(args.layers))
                    else:
                        # A more constrained alternative: interpolate in sparse-code
                        # space before the existing token projector. At zero this is
                        # exactly the frozen OMP baseline with no extra layer gates.
                        self.unrolled_mix_raw = nn.Parameter(torch.zeros(()))
                if args.token_fusion == "adaptive_inject":
                    self.token_routers = nn.ModuleList(
                        [nn.Linear(2 * args.hidden, 1) for _ in range(args.layers)]
                    )
                    for router in self.token_routers:
                        nn.init.zeros_(router.weight)
                        nn.init.zeros_(router.bias)
                if args.token_fusion in {"ring_dual_inject", "ring_dual_readout"}:
                    self.ring_proj = nn.Sequential(
                        nn.Linear(ring_token_dim, args.hidden),
                        nn.ReLU(),
                        nn.Linear(args.hidden, args.hidden),
                    )
                    if args.token_fusion == "ring_dual_inject":
                        self.ring_gates = nn.Parameter(torch.zeros(args.layers))
                    else:
                        self.ring_readout_gate = nn.Parameter(torch.zeros(()))
                if args.token_fusion == "inject_motif_readout":
                    self.motif_proj = nn.Sequential(
                        nn.Linear(base_token_dim * args.hidden, args.hidden),
                        nn.ReLU(),
                        nn.Linear(args.hidden, args.hidden),
                    )
                    self.motif_gate = nn.Parameter(torch.zeros(()))
                if args.token_fusion == "ksvd_atom_additive_readout":
                    # One GINE pass, followed by an additive, atom-decomposable
                    # KSVD-conditioned readout. Sparse OMP assignments define
                    # which node states belong to each learned dictionary atom.
                    # Zero atom weights make the initial predictor exactly the
                    # matched GINE-JK backbone.
                    self.ksvd_atom_score = nn.Linear(args.hidden, 1, bias=False)
                    self.ksvd_atom_weights = nn.Parameter(
                        torch.zeros(base_token_dim)
                    )
                if args.token_fusion == "ksvd_atom_specific_readout":
                    # Stronger but still decomposable: every learned atom owns
                    # its task direction over the shared GINE node state.  This
                    # remains a single-GINE model and starts exactly at GINE
                    # because all atom-specific directions are zero initialized.
                    self.ksvd_atom_score_weights = nn.Parameter(
                        torch.zeros(base_token_dim, args.hidden)
                    )
                if args.token_fusion == "motif_transition_readout":
                    # Parameter-efficient graph-level quotient readout.  Sparse
                    # assignments induce soft motif identities q_v; molecular
                    # edges contribute low-rank pair channels
                    # (q_u L) * (q_v L).  The zero output direction nests the
                    # exact matched GINE predictor at initialization.
                    self.motif_transition_proj = nn.Linear(
                        base_token_dim, args.token_interaction_rank, bias=False
                    )
                    self.motif_transition_head = nn.Linear(
                        args.token_interaction_rank, 1, bias=False
                    )
                    nn.init.zeros_(self.motif_transition_head.weight)
                if args.token_fusion == "motif_geometry_transition_readout":
                    if motif_atom_features_tensor is None:
                        raise ValueError("geometry transition readout requires dictionary")
                    self.register_buffer(
                        "motif_transition_features", motif_atom_features_tensor.clone()
                    )
                    self.motif_transition_weights = nn.Parameter(
                        torch.zeros(args.motif_slot_rank)
                    )
                if args.token_fusion == "motif_transition_matrix_readout":
                    triu = torch.triu_indices(base_token_dim, base_token_dim)
                    self.register_buffer("motif_transition_triu", triu)
                    self.motif_transition_matrix_weights = nn.Parameter(
                        torch.zeros(triu.shape[1])
                    )
                if args.token_fusion in {
                    "ksvd_graph_residual_mean", "ksvd_graph_residual_max",
                    "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
                    "ksvd_code_residual_rich_recon",
                    "ksvd_code_residual_mil_novelty",
                    "ksvd_precomputed_graph_residual",
                }:
                    # A direct low-capacity graph branch over the self-learned
                    # localized dictionary embedding. Zero initialization makes
                    # this exactly the matched backbone at step zero.
                    if args.token_fusion == "ksvd_code_residual_maxabs":
                        graph_residual_dim = base_token_dim
                    elif args.token_fusion == "ksvd_code_residual_rich":
                        # Ten fixed, dictionary-agnostic sparse-code summaries
                        # per atom: mean/max/top3/std/usage/q75/q90/energy/
                        # signed-mean/winner frequency.  This is the strongest
                        # historical standalone KSVD readout, transferred here
                        # as a low-capacity zero-init residual rather than an MLP.
                        graph_residual_dim = 10 * base_token_dim
                    elif args.token_fusion == "ksvd_code_residual_rich_recon":
                        graph_residual_dim = 10 * base_token_dim + 8
                    elif args.token_fusion == "ksvd_code_residual_mil_novelty":
                        # Cross-fitted bag-level witness logit, soft maximum,
                        # hard maximum, top-3 mean, and positive-witness mass.
                        graph_residual_dim = 5
                    elif args.token_fusion == "ksvd_precomputed_graph_residual":
                        if precomputed_graph_features_raw is None:
                            raise RuntimeError("missing precomputed graph features")
                        graph_residual_dim = int(
                            precomputed_graph_features_raw.shape[1]
                        )
                    else:
                        graph_residual_dim = args.hidden
                    self.graph_residual_head = nn.Linear(graph_residual_dim, 1)
                    nn.init.zeros_(self.graph_residual_head.weight)
                    nn.init.zeros_(self.graph_residual_head.bias)
                    # Retain the successful linear KSVD residual and optionally
                    # add a small nonlinear interaction correction.  The final
                    # correction layer is zero initialized, so the full model
                    # still begins as the exact matched GINE predictor.
                    self.graph_residual_nonlinear = None
                    if args.graph_residual_hidden > 0:
                        self.graph_residual_nonlinear = nn.Sequential(
                            nn.Linear(
                                graph_residual_dim, args.graph_residual_hidden
                            ),
                            nn.SiLU(),
                            nn.Linear(args.graph_residual_hidden, 1),
                        )
                        nn.init.zeros_(self.graph_residual_nonlinear[-1].weight)
                        nn.init.zeros_(self.graph_residual_nonlinear[-1].bias)
                if args.token_fusion in augmented_edge_fusions:
                    self.motif_edge_encoder = nn.Embedding(
                        base_token_dim, args.hidden
                    )
                    nn.init.normal_(self.motif_edge_encoder.weight, std=0.02)
                if args.token_fusion in motif_slot_fusions:
                    self.motif_slot = MotifSlotTransport(
                        base_token_dim,
                        args.hidden,
                        args.motif_slot_rank,
                        use_atom_id=(
                            args.token_fusion not in {
                                "motif_slot_no_id", "motif_slot_graph_no_id",
                                "motif_slot_recurrent_no_id",
                            }
                        ),
                        atom_features=motif_atom_features_tensor,
                        use_transition_graph=(
                            args.token_fusion in motif_slot_graph_fusions
                        ),
                    )
                if args.token_fusion in motif_occurrence_fusions:
                    self.motif_occurrence = MotifOccurrenceTransport(
                        base_token_dim,
                        args.hidden,
                        args.motif_slot_rank,
                        use_atom_id=(
                            args.token_fusion != "motif_occurrence_no_id"
                        ),
                        atom_features=motif_atom_features_tensor,
                    )
                if args.token_fusion in motif_ego_fusions:
                    self.motif_ego = MotifEgoTransport(
                        base_token_dim,
                        args.hidden,
                        args.motif_slot_rank,
                        use_atom_id=(
                            args.token_fusion not in {
                                "motif_ego_no_id", "motif_sparse_ego_no_id",
                            }
                        ),
                        atom_features=motif_atom_features_tensor,
                        all_sparse_atoms=(
                            args.token_fusion in motif_sparse_ego_fusions
                        ),
                    )
                if args.token_fusion in learned_cell_fusions:
                    self.learned_cell = LearnedPatchCellTransport(
                        base_token_dim,
                        args.hidden,
                        args.motif_slot_rank,
                        use_atom_id=(
                            args.token_fusion != "ksvd_cell_recurrent_no_id"
                        ),
                        atom_features=motif_atom_features_tensor,
                    )
                if args.token_fusion in witness_cell_readout_fusions:
                    self.witness_cell_readout = WitnessPatchCellReadout(
                        base_token_dim,
                        args.hidden,
                        args.motif_slot_rank,
                        use_atom_id=(
                            args.token_fusion
                            != "ksvd_witness_cell_readout_no_id"
                        ),
                        atom_features=motif_atom_features_tensor,
                    )

        def unrolled_parameters(self):
            eta = self.unrolled_max_step * torch.sigmoid(self.unrolled_step_logits)
            lambdas = F.softplus(self.unrolled_lambda_raw)
            return eta, lambdas

        def refine_ksvd_codes(self, rows):
            assert raw_token_tensor is not None
            assert unrolled_correlation_tensor is not None
            z0 = raw_token_tensor[rows, :base_token_dim]
            correlations = unrolled_correlation_tensor[rows]
            z = z0
            eta, lambdas = self.unrolled_parameters()
            for step in range(args.unrolled_steps):
                step_size = eta[step]
                proposal = z + step_size * (
                    correlations - z @ self.unrolled_gram
                )
                threshold = step_size * lambdas[step]
                z = proposal.sign() * F.relu(proposal.abs() - threshold)
            return z0, z, correlations

        def forward(
            self, data, token_tensor, code_delta_scale=None,
            graph_code_features=None, return_aux=False,
            return_components=False,
        ):
            token_emb = None
            token_gates = None
            interaction_emb = None
            interaction_gates = None
            refined_emb = None
            refined_gates = None
            context_emb = None
            context_gates = None
            unrolled_aux = None
            dynamic_token_factor = None
            ring_emb = None
            ring_gates = None
            dynamic_jk_gates = None
            if self.use_tokens:
                node_tokens = token_tensor[data.token_row]
                if args.token_fusion not in {
                    "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
                    "ksvd_code_residual_rich_recon",
                    "ksvd_code_residual_mil_novelty",
                    "ksvd_precomputed_graph_residual",
                    "ksvd_atom_additive_readout",
                    "ksvd_atom_specific_readout", "motif_transition_readout",
                    "motif_geometry_transition_readout",
                    "motif_transition_matrix_readout", "ksvd_jk_router", "ksvd_aux_support", "ksvd_aux_code",
                    "motif_slot", "motif_slot_no_id", "motif_slot_shuffled_id",
                    "motif_occurrence", "motif_occurrence_no_id",
                    "motif_occurrence_shuffled_id",
                    "motif_ego", "motif_ego_no_id", "motif_ego_shuffled_id",
                }:
                    token_emb = self.token_proj(node_tokens[:, :dual_base_token_dim])
                    token_emb = F.dropout(
                        token_emb, p=args.token_dropout, training=self.training
                    )
                token_gates = self.token_gates
                if args.token_fusion in {"ksvd_context_inject", "ksvd_context_tied"}:
                    if args.token_fusion == "ksvd_context_inject":
                        context_emb = global_mean_pool(token_emb, data.batch)
                        context_gates = self.context_gates
                    else:
                        # Aggregate signed KSVD coefficients before the shared
                        # nonlinear projector, then reuse the local token gates.
                        # This adds zero parameters and ties local/global signs.
                        graph_codes = global_mean_pool(
                            node_tokens[:, :dual_base_token_dim], data.batch
                        )
                        context_emb = self.token_proj(graph_codes)
                        context_emb = F.dropout(
                            context_emb, p=args.token_dropout, training=self.training
                        )
                        context_gates = self.token_gates
                if args.token_fusion in {"bilinear_inject", "edge_bilinear_inject"}:
                    primary_tokens = node_tokens[:, :dual_base_token_dim]
                    if token_support_tensor is not None:
                        primary_tokens = primary_tokens * token_support_tensor[
                            data.token_row, :dual_base_token_dim
                        ]
                    left = torch.tanh(self.interaction_left(primary_tokens))
                    right = torch.tanh(self.interaction_right(primary_tokens))
                    if args.token_fusion == "edge_bilinear_inject":
                        # A KSVD-native analogue of a learned cell-boundary
                        # interaction: atom combinations are formed only across
                        # actual molecular edges, then mean-aggregated at the
                        # receiving atom. No ring/scaffold annotations enter.
                        src, dst = data.edge_index
                        rank_messages = left[src] * right[dst]
                        rank_aggregate = torch.zeros_like(left)
                        rank_aggregate.index_add_(0, dst, rank_messages)
                        degree = left.new_zeros((left.shape[0], 1))
                        degree.index_add_(
                            0, dst, left.new_ones((dst.shape[0], 1))
                        )
                        rank_aggregate = rank_aggregate / degree.clamp_min(1.0)
                        interaction_emb = self.interaction_proj(rank_aggregate)
                        interaction_emb = interaction_emb * (degree > 0).to(
                            interaction_emb.dtype
                        )
                    else:
                        interaction_emb = self.interaction_proj(left * right)
                    interaction_emb = F.dropout(
                        interaction_emb,
                        p=args.token_dropout,
                        training=self.training,
                    )
                    interaction_gates = self.interaction_gates
                elif args.token_fusion == "dynamic_bilinear_inject":
                    primary_tokens = node_tokens[:, :dual_base_token_dim]
                    if token_support_tensor is not None:
                        primary_tokens = primary_tokens * token_support_tensor[
                            data.token_row, :dual_base_token_dim
                        ]
                    dynamic_token_factor = torch.tanh(
                        self.dynamic_token_proj(primary_tokens)
                    )
                    interaction_gates = self.interaction_gates
                elif args.token_fusion in unrolled_fusions:
                    if code_delta_scale is None:
                        raise ValueError("unrolled KSVD fusion requires code_delta_scale")
                    z0, refined_z, correlations = self.refine_ksvd_codes(data.token_row)
                    refined_delta = (refined_z - z0) * code_delta_scale
                    refined_token_input = (
                        node_tokens[:, :dual_base_token_dim]
                        + refined_delta
                    )
                    if args.token_fusion == "unrolled_ksvd_inject":
                        refined_emb = (
                            self.token_proj(refined_token_input)
                            - self.token_proj(node_tokens[:, :dual_base_token_dim])
                        )
                        refined_emb = F.dropout(
                            refined_emb, p=args.token_dropout, training=self.training
                        )
                        refined_gates = self.refined_gates
                    else:
                        mix = torch.tanh(self.unrolled_mix_raw)
                        token_emb = self.token_proj(
                            node_tokens[:, :dual_base_token_dim] + mix * refined_delta
                        )
                        token_emb = F.dropout(
                            token_emb, p=args.token_dropout, training=self.training
                        )
                    if return_aux:
                        recon = (
                            1.0
                            - 2.0 * (refined_z * correlations).sum(dim=1)
                            + ((refined_z @ self.unrolled_gram) * refined_z).sum(dim=1)
                        ).mean()
                        sparse = refined_z.abs().mean()
                        unrolled_aux = (
                            args.unrolled_recon_weight * recon
                            + args.unrolled_sparse_weight * sparse
                        )
                if args.token_fusion in {"ring_dual_inject", "ring_dual_readout"}:
                    ring_emb = self.ring_proj(node_tokens[:, dual_base_token_dim:])
                    ring_emb = F.dropout(
                        ring_emb, p=args.token_dropout, training=self.training
                    )
                    if args.token_fusion == "ring_dual_inject":
                        ring_gates = self.ring_gates
                if args.token_fusion == "ksvd_jk_router":
                    if code_delta_scale is None:
                        raise ValueError("KSVD JK router requires fit-only scaling")
                    assert raw_token_tensor is not None
                    scaled_sparse_codes = (
                        raw_token_tensor[data.token_row, :base_token_dim]
                        * code_delta_scale
                    )
                    graph_code_presence = global_max_pool(
                        scaled_sparse_codes.abs(), data.batch
                    )
                    dynamic_jk_gates = self.jk_router(graph_code_presence)
            nonlocal_token_fusions = (
                graph_residual_fusions
                | motif_topology_fusions
                | augmented_edge_fusions
                | {
                    "ksvd_atom_additive_readout", "ksvd_atom_specific_readout",
                    "motif_transition_readout", "motif_geometry_transition_readout",
                    "motif_transition_matrix_readout", "ksvd_jk_router",
                    "ksvd_aux_support", "ksvd_aux_code",
                }
            )
            graph_h, node_h = self.gine(
                data,
                token_emb=(
                    None if args.token_fusion in nonlocal_token_fusions else token_emb
                ),
                token_gates=(
                    None if args.token_fusion in nonlocal_token_fusions else token_gates
                ),
                token_routers=(
                    self.token_routers
                    if args.token_fusion == "adaptive_inject" else None
                ),
                interaction_emb=interaction_emb,
                interaction_gates=interaction_gates,
                refined_emb=refined_emb,
                refined_gates=refined_gates,
                context_emb=context_emb,
                context_gates=context_gates,
                dynamic_token_factor=dynamic_token_factor,
                dynamic_state_projs=(
                    self.dynamic_state_projs
                    if args.token_fusion == "dynamic_bilinear_inject" else None
                ),
                dynamic_out_projs=(
                    self.dynamic_out_projs
                    if args.token_fusion == "dynamic_bilinear_inject" else None
                ),
                ring_emb=ring_emb if args.token_fusion == "ring_dual_inject" else None,
                ring_gates=ring_gates,
                dynamic_jk_gates=dynamic_jk_gates,
                motif_slot_module=(
                    self.motif_slot if args.token_fusion in motif_slot_fusions else None
                ),
                motif_occurrence_module=(
                    self.motif_occurrence
                    if args.token_fusion in motif_occurrence_fusions else None
                ),
                motif_ego_module=(
                    self.motif_ego if args.token_fusion in motif_ego_fusions else None
                ),
                learned_cell_module=(
                    self.learned_cell
                    if args.token_fusion in learned_cell_fusions else None
                ),
                motif_codes=(
                    raw_token_tensor[data.token_row, :base_token_dim]
                    if args.token_fusion in motif_topology_fusions else None
                ),
                motif_occurrence_ids=(
                    motif_occurrence_tensor[data.token_row, :base_token_dim]
                    if args.token_fusion in motif_occurrence_fusions else None
                ),
                motif_ego_edge_index=(
                    data.motif_ego_edge_index
                    if args.token_fusion in (
                        motif_ego_fusions
                        | learned_cell_fusions
                        | witness_cell_readout_fusions
                    )
                    else None
                ),
                motif_ego_distance=(
                    data.motif_ego_distance
                    if args.token_fusion in learned_cell_fusions else None
                ),
                motif_edge_encoder=(
                    self.motif_edge_encoder
                    if args.token_fusion in augmented_edge_fusions else None
                ),
            )
            if self.use_tokens and args.token_fusion == "ring_dual_readout":
                membership = ring_membership_tensor[data.token_row].unsqueeze(-1)
                ring_sum = global_add_pool(ring_emb * membership, data.batch)
                ring_den = global_add_pool(membership, data.batch).clamp_min(1.0)
                graph_h = graph_h + args.ring_gate_scale * torch.tanh(
                    self.ring_readout_gate
                ) * (ring_sum / ring_den)
            if self.use_tokens and args.token_fusion == "inject_motif_readout":
                signed = token_tensor[data.token_row, :base_token_dim]
                if args.token_channels == "signed_abs_support":
                    support = token_tensor[
                        data.token_row, 2 * base_token_dim : 3 * base_token_dim
                    ]
                    weights = support * (signed.abs() + 1e-3)
                else:
                    weights = signed.abs()
                weighted = (weights.unsqueeze(-1) * node_h.unsqueeze(1)).reshape(
                    node_h.shape[0], base_token_dim * args.hidden
                )
                slot_sum = global_add_pool(weighted, data.batch).reshape(
                    -1, base_token_dim, args.hidden
                )
                slot_den = global_add_pool(weights, data.batch).clamp_min(1e-6).unsqueeze(-1)
                slot_mean = slot_sum / slot_den
                motif_h = self.motif_proj(slot_mean.reshape(-1, base_token_dim * args.hidden))
                graph_h = graph_h + torch.tanh(self.motif_gate) * motif_h
            if self.use_tokens and args.token_fusion in witness_cell_readout_fusions:
                assert raw_token_tensor is not None
                cell_graph_h = self.witness_cell_readout(
                    node_h,
                    data.batch,
                    raw_token_tensor[data.token_row, :base_token_dim],
                    data.motif_ego_edge_index,
                    data.motif_ego_distance,
                )
                motif_scale = (
                    1.0 if args.motif_gate_mode == "zero_out"
                    else torch.tanh(self.witness_cell_readout.gate)
                )
                graph_h = graph_h + args.token_gate_scale * motif_scale * cell_graph_h
            logits = self.head(graph_h)
            base_logits = logits
            residual_logits = None
            if self.use_tokens and args.token_fusion in {
                "ksvd_atom_additive_readout", "ksvd_atom_specific_readout"
            }:
                assert raw_token_tensor is not None
                sparse_weights = raw_token_tensor[
                    data.token_row, :base_token_dim
                ].abs()
                weighted_states = (
                    sparse_weights.unsqueeze(-1) * node_h.unsqueeze(1)
                ).reshape(node_h.shape[0], base_token_dim * args.hidden)
                atom_state_sum = global_add_pool(
                    weighted_states, data.batch
                ).reshape(-1, base_token_dim, args.hidden)
                atom_mass = global_add_pool(
                    sparse_weights, data.batch
                )
                atom_states = atom_state_sum / atom_mass.clamp_min(1e-6).unsqueeze(-1)
                atom_active = (atom_mass > 0).to(atom_states.dtype)
                if args.token_fusion == "ksvd_atom_additive_readout":
                    atom_scores = self.ksvd_atom_score(atom_states).squeeze(-1)
                    atom_scores = atom_scores * torch.tanh(
                        self.ksvd_atom_weights
                    ).unsqueeze(0)
                else:
                    atom_scores = (
                        atom_states * self.ksvd_atom_score_weights.unsqueeze(0)
                    ).sum(dim=-1) / (args.hidden ** 0.5)
                atom_contributions = atom_active * atom_scores
                active_scale = atom_active.sum(dim=-1).clamp_min(1.0)
                logits = logits + (
                    atom_contributions.sum(dim=-1) / active_scale
                ).unsqueeze(-1)
            if (
                return_aux
                and self.use_tokens
                and args.token_fusion in {"ksvd_aux_support", "ksvd_aux_code"}
            ):
                assert raw_token_tensor is not None
                aux_logits = self.ksvd_aux_head(node_h)
                raw_codes = raw_token_tensor[data.token_row, :base_token_dim]
                if args.token_fusion == "ksvd_aux_support":
                    support = (raw_codes != 0).to(aux_logits.dtype)
                    positive = support.bool()
                    negative = ~positive
                    # OMP T=3 makes positives sparse. Equal positive/negative
                    # averaging prevents the trivial all-zero support predictor.
                    positive_loss = F.softplus(-aux_logits[positive]).mean()
                    negative_loss = F.softplus(aux_logits[negative]).mean()
                    unrolled_aux = (
                        self.ksvd_aux_scale
                        * args.ksvd_aux_weight
                        * 0.5
                        * (positive_loss + negative_loss)
                    )
                else:
                    if code_delta_scale is None:
                        raise ValueError("KSVD code auxiliary requires fit-only scaling")
                    target_codes = raw_codes * code_delta_scale
                    cosine = F.cosine_similarity(
                        aux_logits, target_codes, dim=-1, eps=1e-8
                    )
                    unrolled_aux = (
                        self.ksvd_aux_scale
                        * args.ksvd_aux_weight
                        * (1.0 - cosine).mean()
                    )
            if self.use_tokens and args.token_fusion == "motif_transition_readout":
                assert raw_token_tensor is not None
                raw_codes = raw_token_tensor[data.token_row, :base_token_dim]
                node_assignment = raw_codes.abs()
                node_assignment = node_assignment / node_assignment.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                node_factor = self.motif_transition_proj(node_assignment)
                src, dst = data.edge_index
                if src.numel() == 0:
                    graph_transition = node_factor.new_zeros(
                        (int(data.num_graphs), args.token_interaction_rank)
                    )
                else:
                    edge_transition = node_factor[src] * node_factor[dst]
                    graph_transition = global_mean_pool(
                        edge_transition, data.batch[src], size=int(data.num_graphs)
                    )
                logits = logits + self.motif_transition_head(graph_transition)
            if self.use_tokens and args.token_fusion == "motif_geometry_transition_readout":
                assert raw_token_tensor is not None
                raw_codes = raw_token_tensor[data.token_row, :base_token_dim]
                assignment_scale = raw_codes.abs().sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                if args.motif_geometry_assignment == "signed":
                    node_assignment = raw_codes / assignment_scale
                else:
                    node_assignment = raw_codes.abs() / assignment_scale
                node_factor = node_assignment @ self.motif_transition_features
                src, dst = data.edge_index
                if src.numel() == 0:
                    graph_transition = node_factor.new_zeros(
                        (int(data.num_graphs), args.motif_slot_rank)
                    )
                else:
                    edge_transition = node_factor[src] * node_factor[dst]
                    graph_transition = global_mean_pool(
                        edge_transition, data.batch[src], size=int(data.num_graphs)
                    )
                logits = logits + (
                    graph_transition * self.motif_transition_weights.unsqueeze(0)
                ).sum(dim=-1, keepdim=True)
            if self.use_tokens and args.token_fusion == "motif_transition_matrix_readout":
                assert raw_token_tensor is not None
                raw_codes = raw_token_tensor[data.token_row, :base_token_dim]
                node_assignment = raw_codes.abs()
                node_assignment = node_assignment / node_assignment.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                src, dst = data.edge_index
                if src.numel() == 0:
                    graph_transition = node_assignment.new_zeros(
                        (int(data.num_graphs), base_token_dim, base_token_dim)
                    )
                else:
                    edge_outer = (
                        node_assignment[src].unsqueeze(-1)
                        * node_assignment[dst].unsqueeze(-2)
                    ).reshape(src.shape[0], -1)
                    graph_transition = global_mean_pool(
                        edge_outer, data.batch[src], size=int(data.num_graphs)
                    ).reshape(-1, base_token_dim, base_token_dim)
                transition_features = graph_transition[
                    :, self.motif_transition_triu[0], self.motif_transition_triu[1]
                ]
                logits = logits + (
                    transition_features
                    * self.motif_transition_matrix_weights.unsqueeze(0)
                ).sum(dim=-1, keepdim=True)
            if self.use_tokens and args.token_fusion in graph_residual_fusions:
                if args.token_fusion == "ksvd_code_residual_maxabs":
                    if code_delta_scale is None:
                        raise ValueError("KSVD code residual requires fit-only scaling")
                    assert raw_token_tensor is not None
                    scaled_sparse_codes = (
                        raw_token_tensor[data.token_row, :base_token_dim]
                        * code_delta_scale
                    )
                    graph_token_h = global_max_pool(
                        scaled_sparse_codes.abs(), data.batch
                    )
                elif args.token_fusion in {
                    "ksvd_code_residual_rich",
                    "ksvd_code_residual_rich_recon",
                    "ksvd_code_residual_mil_novelty",
                    "ksvd_precomputed_graph_residual",
                }:
                    if graph_code_features is None:
                        raise ValueError(
                            "rich sparse-code residual requires fit-standardized "
                            "graph features"
                        )
                    graph_token_h = graph_code_features[data.idx.view(-1)]
                elif args.token_fusion == "ksvd_graph_residual_mean":
                    graph_token_h = global_mean_pool(token_emb, data.batch)
                else:
                    graph_token_h = global_max_pool(token_emb, data.batch)
                residual_logits = self.graph_residual_head(graph_token_h)
                if self.graph_residual_nonlinear is not None:
                    residual_logits = (
                        residual_logits
                        + self.graph_residual_nonlinear(graph_token_h)
                    )
                if args.graph_residual_combination == "probability_average":
                    combined_probability = 0.5 * (
                        base_logits.sigmoid() + residual_logits.sigmoid()
                    )
                    logits = torch.logit(
                        combined_probability.clamp(min=1e-6, max=1.0 - 1e-6)
                    )
                else:
                    # Keep this single addition identical to the historical path.
                    logits = logits + residual_logits
            if return_aux:
                if unrolled_aux is None:
                    unrolled_aux = logits.new_zeros(())
                if return_components:
                    return logits, unrolled_aux, base_logits, residual_logits
                return logits, unrolled_aux
            if return_components:
                return logits, base_logits, residual_logits
            return logits

    def state_hash(module, *, base_only: bool = False) -> str:
        h = hashlib.sha256()
        for name, tensor in module.state_dict().items():
            if base_only and not (name.startswith("gine.") or name.startswith("head.")):
                continue
            h.update(name.encode("utf-8"))
            arr = tensor.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode("ascii"))
            h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            h.update(arr.tobytes())
        return h.hexdigest()

    def array_hash(array: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(array, dtype=np.int64).tobytes()).hexdigest()

    def make_model_optimizer(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = TokenModel().to(device)
        if model.use_tokens:
            if args.token_fusion in {
                "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
                "ksvd_code_residual_rich_recon",
                "ksvd_code_residual_mil_novelty",
                "ksvd_precomputed_graph_residual",
            }:
                token_params = list(model.graph_residual_head.parameters())
                if model.graph_residual_nonlinear is not None:
                    token_params += list(
                        model.graph_residual_nonlinear.parameters()
                    )
            elif args.token_fusion == "ksvd_atom_additive_readout":
                token_params = (
                    list(model.ksvd_atom_score.parameters())
                    + [model.ksvd_atom_weights]
                )
            elif args.token_fusion == "ksvd_atom_specific_readout":
                token_params = [model.ksvd_atom_score_weights]
            elif args.token_fusion == "motif_transition_readout":
                token_params = (
                    list(model.motif_transition_proj.parameters())
                    + list(model.motif_transition_head.parameters())
                )
            elif args.token_fusion == "motif_geometry_transition_readout":
                token_params = [model.motif_transition_weights]
            elif args.token_fusion == "motif_transition_matrix_readout":
                token_params = [model.motif_transition_matrix_weights]
            elif args.token_fusion == "ksvd_jk_router":
                token_params = list(model.jk_router.parameters())
            elif args.token_fusion in {"ksvd_aux_support", "ksvd_aux_code"}:
                token_params = list(model.ksvd_aux_head.parameters())
            elif args.token_fusion in augmented_edge_fusions:
                token_params = list(model.motif_edge_encoder.parameters())
            elif args.token_fusion in motif_slot_fusions:
                token_params = list(model.motif_slot.parameters())
            elif args.token_fusion in motif_occurrence_fusions:
                token_params = list(model.motif_occurrence.parameters())
            elif args.token_fusion in motif_ego_fusions:
                token_params = list(model.motif_ego.parameters())
            elif args.token_fusion in learned_cell_fusions:
                token_params = list(model.learned_cell.parameters())
            elif args.token_fusion in witness_cell_readout_fusions:
                token_params = list(model.witness_cell_readout.parameters())
            else:
                token_params = list(model.token_proj.parameters()) + [model.token_gates]
            if args.token_fusion in {"bilinear_inject", "edge_bilinear_inject"}:
                token_params += (
                    list(model.interaction_left.parameters())
                    + list(model.interaction_right.parameters())
                    + list(model.interaction_proj.parameters())
                    + [model.interaction_gates]
                )
            elif args.token_fusion == "dynamic_bilinear_inject":
                token_params += (
                    list(model.dynamic_token_proj.parameters())
                    + list(model.dynamic_state_projs.parameters())
                    + list(model.dynamic_out_projs.parameters())
                    + [model.interaction_gates]
                )
            elif args.token_fusion in unrolled_fusions:
                token_params += [model.unrolled_step_logits, model.unrolled_lambda_raw]
                if args.token_fusion == "unrolled_ksvd_inject":
                    token_params += [model.refined_gates]
                else:
                    token_params += [model.unrolled_mix_raw]
            if args.token_fusion == "ksvd_context_inject":
                token_params += [model.context_gates]
            if args.token_fusion == "adaptive_inject":
                token_params += list(model.token_routers.parameters())
            if args.token_fusion == "ring_dual_inject":
                token_params += list(model.ring_proj.parameters()) + [model.ring_gates]
            elif args.token_fusion == "ring_dual_readout":
                token_params += list(model.ring_proj.parameters()) + [model.ring_readout_gate]
            if args.token_fusion == "inject_motif_readout":
                token_params += list(model.motif_proj.parameters()) + [model.motif_gate]
            if (
                args.token_fusion in graph_residual_fusions
                and args.token_fusion not in {
                    "ksvd_code_residual_maxabs", "ksvd_code_residual_rich",
                    "ksvd_code_residual_rich_recon",
                    "ksvd_code_residual_mil_novelty",
                    "ksvd_precomputed_graph_residual",
                }
            ):
                token_params += list(model.graph_residual_head.parameters())
                if model.graph_residual_nonlinear is not None:
                    token_params += list(
                        model.graph_residual_nonlinear.parameters()
                    )
            token_ids = {id(p) for p in token_params}
            base_params = [
                p for p in model.parameters()
                if p.requires_grad and id(p) not in token_ids
            ]
            optimizer = torch.optim.Adam(
                [
                    {"params": base_params, "lr": args.lr},
                    {
                        "params": token_params,
                        "lr": args.lr * args.token_lr_scale,
                        "weight_decay": args.token_weight_decay,
                    },
                ]
            )
            model.token_parameter_count = sum(p.numel() for p in token_params)
            model.base_parameter_count = sum(p.numel() for p in base_params)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.token_parameter_count = 0
            model.base_parameter_count = sum(p.numel() for p in model.parameters())
        model.trainable_parameter_count = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        return model, optimizer

    def make_parameter_ema(model):
        if args.parameter_ema_decay <= 0.0:
            return None
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def set_token_group_lr(model, optimizer, epoch: int) -> None:
        if not model.use_tokens:
            return
        if (
            args.motif_gate_mode == "scalar"
            and args.motif_gate_wake_init > 0.0
            and epoch == args.token_warmup_epochs + 1
        ):
            motif_module = None
            if args.token_fusion in motif_slot_fusions:
                motif_module = model.motif_slot
            elif args.token_fusion in motif_occurrence_fusions:
                motif_module = model.motif_occurrence
            elif args.token_fusion in motif_ego_fusions:
                motif_module = model.motif_ego
            elif args.token_fusion in learned_cell_fusions:
                motif_module = model.learned_cell
            elif args.token_fusion in witness_cell_readout_fusions:
                motif_module = model.witness_cell_readout
            if motif_module is not None:
                with torch.no_grad():
                    motif_module.gate.fill_(args.motif_gate_wake_init)
                log(
                    f"woke scalar motif gate at epoch {epoch}: "
                    f"raw={args.motif_gate_wake_init:.6g}, "
                    f"effective={args.token_gate_scale * np.tanh(args.motif_gate_wake_init):.6g}"
                )
        if args.token_fusion in {"ksvd_aux_support", "ksvd_aux_code"}:
            model.ksvd_aux_scale = float(
                args.ksvd_aux_epochs == 0 or epoch <= args.ksvd_aux_epochs
            )
        optimizer.param_groups[1]["lr"] = (
            0.0
            if epoch <= args.token_warmup_epochs
            else args.lr * args.token_lr_scale
        )

    @torch.no_grad()
    def update_parameter_ema(model, ema_state):
        if ema_state is None:
            return
        decay = args.parameter_ema_decay
        for name, param in model.named_parameters():
            if name in ema_state:
                ema_state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def install_parameter_ema(model, ema_state):
        if ema_state is None:
            return
        for name, param in model.named_parameters():
            if name in ema_state:
                param.copy_(ema_state[name])

    def make_loader(indices: np.ndarray, shuffle: bool, phase_offset: int):
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(args.seed + 9173 + phase_offset)
        return DataLoader(
            [data_by_idx[int(i)] for i in indices],
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def normalize_tokens(fit_indices: np.ndarray):
        if token_raw is None:
            return None, {
                "enabled": False,
                "fit_graph_count": int(len(fit_indices)),
                "fit_node_count": 0,
            }, None, None

        # All preprocessing statistics are estimated strictly from the phase's
        # fit graphs.  Graph-level document frequency prevents large molecules
        # from dominating the stability weight merely because they have more atoms.
        fit_mask = np.zeros(token_raw.shape[0], dtype=bool)
        graph_support_count = np.zeros(base_token_dim, dtype=np.int64)
        for i in fit_indices:
            lo, hi = int(offsets[int(i)]), int(offsets[int(i) + 1])
            fit_mask[lo:hi] = True
            graph_support_count += np.any(token_raw[lo:hi] != 0, axis=0)
        fit_values = token_raw[fit_mask]
        graph_frequency = graph_support_count.astype(np.float64) / max(len(fit_indices), 1)

        if args.token_normalization == "zscore":
            mean = fit_values.mean(axis=0, keepdims=True)
            scale = fit_values.std(axis=0, keepdims=True)
            scaled = (token_raw - mean) / np.maximum(scale, 1e-6)
        elif args.token_normalization == "rms":
            mean = np.zeros((1, base_token_dim), dtype=np.float32)
            scale = np.sqrt(np.mean(np.square(fit_values), axis=0, keepdims=True))
            scaled = token_raw / np.maximum(scale, 1e-6)
        else:
            mean = np.zeros((1, base_token_dim), dtype=np.float32)
            scale = np.ones((1, base_token_dim), dtype=np.float32)
            scaled = token_raw.copy()

        # For a refined raw code z, the normalized residual relative to OMP z0
        # is exactly (z-z0) * code_delta_scale; the z-score mean cancels.
        code_delta_scale = 1.0 / np.maximum(scale, 1e-6)

        preclip_fit = scaled[fit_mask]
        clipped_fraction = np.zeros(base_token_dim, dtype=np.float64)
        if args.token_zscore_clip > 0.0:
            clipped_fraction = np.mean(
                np.abs(preclip_fit) > args.token_zscore_clip, axis=0
            )
            scaled = np.clip(
                scaled, -args.token_zscore_clip, args.token_zscore_clip
            )

        atom_weight = np.ones(base_token_dim, dtype=np.float64)
        if args.token_prevalence_power > 0.0:
            atom_weight = np.power(graph_frequency, args.token_prevalence_power)
        retained = graph_frequency >= args.token_min_graph_frequency
        atom_weight[~retained] = 0.0
        positive = atom_weight > 0.0
        if not np.any(positive):
            raise ValueError(
                "token preprocessing masked every dictionary atom; lower "
                "--token-min-graph-frequency"
            )
        # Keep the average retained scale fixed so this knob changes relative
        # atom reliability rather than the global scale seen by the learned gate.
        atom_weight[positive] /= atom_weight[positive].mean()
        scaled = scaled * atom_weight.astype(np.float32)[None, :]
        code_delta_scale = code_delta_scale * atom_weight.astype(np.float32)[None, :]

        if args.token_channels == "signed_abs_support":
            support = (token_raw != 0).astype(np.float32)
            support *= atom_weight.astype(np.float32)[None, :]
            scaled = np.concatenate([scaled, np.abs(scaled), support], axis=1)

        stats = {
            "enabled": True,
            "fit_graph_count": int(len(fit_indices)),
            "fit_node_count": int(fit_mask.sum()),
            "normalization": args.token_normalization,
            "zscore_clip": args.token_zscore_clip,
            "prevalence_power": args.token_prevalence_power,
            "min_graph_frequency": args.token_min_graph_frequency,
            "graph_frequency": graph_frequency.tolist(),
            "graph_frequency_min": float(graph_frequency.min()),
            "graph_frequency_median": float(np.median(graph_frequency)),
            "graph_frequency_max": float(graph_frequency.max()),
            "atom_weight": atom_weight.tolist(),
            "atom_weight_min": float(atom_weight.min()),
            "atom_weight_median": float(np.median(atom_weight)),
            "atom_weight_max": float(atom_weight.max()),
            "masked_atom_count": int((~positive).sum()),
            "normalization_mean": mean.reshape(-1).astype(float).tolist(),
            "normalization_scale": scale.reshape(-1).astype(float).tolist(),
            "clipped_fit_fraction": clipped_fraction.tolist(),
            "clipped_fit_fraction_mean": float(clipped_fraction.mean()),
            "clipped_fit_fraction_max": float(clipped_fraction.max()),
        }
        graph_code_features = None
        if args.token_fusion in {
            "ksvd_code_residual_rich", "ksvd_code_residual_rich_recon"
        }:
            # Compute the fixed rich signature from raw OMP coefficients, then
            # standardize each graph feature using only this phase's fit graphs.
            # No held-out scaffold label or feature statistic enters the fit.
            dictionary_gram = None
            if args.token_fusion == "ksvd_code_residual_rich_recon":
                if graph_code_dictionary_raw is None:
                    raise RuntimeError("missing reconstruction dictionary")
                dictionary_gram = (
                    graph_code_dictionary_raw.T @ graph_code_dictionary_raw
                )
            rich_rows = []
            for graph_i in range(len(graphs)):
                lo = int(offsets[graph_i])
                hi = int(offsets[graph_i + 1])
                codes = token_raw[lo:hi, :base_token_dim].astype(
                    np.float64, copy=False
                )
                absolute = np.abs(codes)
                n_nodes = int(absolute.shape[0])
                if n_nodes == 0:
                    blocks = [
                        np.zeros(base_token_dim, dtype=np.float64)
                        for _ in range(10)
                    ]
                else:
                    quantiles = np.quantile(absolute, [0.75, 0.90], axis=0)
                    top_count = min(3, n_nodes)
                    top_mean = np.partition(
                        absolute, n_nodes - top_count, axis=0
                    )[n_nodes - top_count:].mean(axis=0)
                    winner = np.bincount(
                        np.argmax(absolute, axis=1), minlength=base_token_dim
                    ).astype(np.float64) / n_nodes
                    blocks = [
                        absolute.mean(axis=0),
                        absolute.max(axis=0),
                        top_mean,
                        absolute.std(axis=0),
                        (absolute > 1e-10).mean(axis=0),
                        quantiles[0],
                        quantiles[1],
                        np.square(codes).mean(axis=0),
                        codes.mean(axis=0),
                        winner,
                    ]
                if dictionary_gram is not None:
                    if n_nodes:
                        explained = np.einsum(
                            "ni,ij,nj->n", codes, dictionary_gram, codes,
                            optimize=True,
                        )
                        errors = np.sqrt(np.maximum(1.0 - explained, 0.0))
                        error_features = np.asarray(
                            [
                                errors.mean(), errors.std(),
                                *np.quantile(errors, [0.50, 0.75, 0.90]),
                                errors.max(), float(n_nodes), np.log1p(n_nodes),
                            ],
                            dtype=np.float64,
                        )
                    else:
                        error_features = np.zeros(8, dtype=np.float64)
                    blocks.append(error_features)
                rich_rows.append(np.concatenate(blocks))
            rich = np.stack(rich_rows, axis=0)
            rich_mean = rich[fit_indices].mean(axis=0, keepdims=True)
            rich_scale = rich[fit_indices].std(axis=0, keepdims=True)
            rich_scale = np.maximum(rich_scale, 1e-6)
            rich_scaled = (rich - rich_mean) / rich_scale
            graph_code_features = torch.tensor(
                rich_scaled, dtype=torch.float32, device=device
            )
            stats["graph_code_readout"] = {
                "kind": (
                    "rich_with_omp_reconstruction"
                    if dictionary_gram is not None
                    else "rich_no_reconstruction"
                ),
                "feature_dim": int(rich.shape[1]),
                "fit_graph_count": int(len(fit_indices)),
                "fit_feature_mean_sha256": array_hash(
                    rich_mean.astype(np.float32).reshape(-1)
                ),
                "fit_feature_scale_sha256": array_hash(
                    rich_scale.astype(np.float32).reshape(-1)
                ),
                "fit_scaled_abs_max": float(
                    np.abs(rich_scaled[fit_indices]).max()
                ),
            }
        elif args.token_fusion == "ksvd_code_residual_mil_novelty":
            # A molecule is treated as a bag of radius-r patch instances.  The
            # fixed scorer is trained with one loss per graph (never by copying
            # the graph label onto every patch).  Its instance direction combines
            # normalized sparse assignments with a nonnegative reconstruction-
            # novelty coefficient.  Cross-fitted rows are used for the fit
            # graphs, while a full fit-only scorer encodes held-out scaffolds.
            if graph_code_dictionary_raw is None:
                raise RuntimeError("MIL novelty readout requires a dictionary")
            dictionary_gram = (
                graph_code_dictionary_raw.T @ graph_code_dictionary_raw
            )
            codes64 = token_raw[:, :base_token_dim].astype(np.float64, copy=False)
            absolute64 = np.abs(codes64)
            assignment64 = absolute64 / np.maximum(
                absolute64.sum(axis=1, keepdims=True), 1e-10
            )
            explained64 = np.einsum(
                "ni,ij,nj->n", codes64, dictionary_gram, codes64,
                optimize=True,
            )
            reconstruction64 = np.sqrt(np.maximum(1.0 - explained64, 0.0))
            energy64 = np.sqrt(np.square(codes64).sum(axis=1))
            recon_mean = float(reconstruction64[fit_mask].mean())
            recon_scale = float(reconstruction64[fit_mask].std())
            energy_mean = float(energy64[fit_mask].mean())
            energy_scale = float(energy64[fit_mask].std())
            instance_features = np.concatenate(
                [
                    assignment64,
                    ((reconstruction64 - recon_mean) / max(recon_scale, 1e-6))[:, None],
                    ((energy64 - energy_mean) / max(energy_scale, 1e-6))[:, None],
                ],
                axis=1,
            ).astype(np.float32)
            instance_tensor = torch.tensor(instance_features, dtype=torch.float32)

            class MILNoveltyScorer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.atom_weights = nn.Parameter(torch.zeros(base_token_dim))
                    # softplus(-2.252...) ~= 0.1: start with a small explicitly
                    # positive novelty direction, not a privileged atom label.
                    self.novelty_raw = nn.Parameter(torch.tensor(-2.2521685))
                    self.energy_weight = nn.Parameter(torch.zeros(()))
                    self.bias = nn.Parameter(torch.zeros(()))

                def node_scores(self, features):
                    return (
                        features[:, :base_token_dim] @ self.atom_weights
                        + F.softplus(self.novelty_raw)
                        * features[:, base_token_dim]
                        + self.energy_weight * features[:, base_token_dim + 1]
                    )

            def pack_mil_graphs(graph_indices: np.ndarray):
                node_rows: list[np.ndarray] = []
                batch_rows: list[np.ndarray] = []
                for local_i, graph_i in enumerate(graph_indices.tolist()):
                    lo = int(offsets[graph_i])
                    hi = int(offsets[graph_i + 1])
                    if hi <= lo:
                        raise RuntimeError("MIL graph has no patch instances")
                    node_rows.append(np.arange(lo, hi, dtype=np.int64))
                    batch_rows.append(
                        np.full(hi - lo, local_i, dtype=np.int64)
                    )
                return (
                    torch.tensor(np.concatenate(node_rows), dtype=torch.long),
                    torch.tensor(np.concatenate(batch_rows), dtype=torch.long),
                    torch.tensor(
                        y_np[graph_indices].astype(np.float32),
                        dtype=torch.float32,
                    ),
                )

            def soft_top_pool(node_scores, batch, n_graphs: int):
                maximum = global_max_pool(
                    node_scores[:, None], batch, size=n_graphs
                ).reshape(-1)
                centered = (
                    node_scores - maximum[batch]
                ) / args.mil_temperature
                exp_sum = global_add_pool(
                    torch.exp(centered)[:, None], batch, size=n_graphs
                ).reshape(-1)
                count = global_add_pool(
                    torch.ones_like(node_scores)[:, None], batch, size=n_graphs
                ).reshape(-1)
                return maximum + args.mil_temperature * (
                    torch.log(exp_sum.clamp_min(1e-12))
                    - torch.log(count.clamp_min(1.0))
                )

            def fit_mil_scorer(graph_indices: np.ndarray):
                label_values = y_np[graph_indices]
                if len(np.unique(label_values)) != 2:
                    raise RuntimeError("MIL fit subset lacks a class")
                rows, batch, labels = pack_mil_graphs(graph_indices)
                scorer = MILNoveltyScorer()
                optimizer = torch.optim.Adam(
                    scorer.parameters(), lr=args.mil_fit_lr
                )
                loss_curve: list[float] = []
                for _ in range(args.mil_fit_epochs):
                    node_scores = scorer.node_scores(instance_tensor[rows])
                    logits = soft_top_pool(
                        node_scores, batch, len(graph_indices)
                    ) + scorer.bias
                    positive = labels > 0.5
                    negative = ~positive
                    balanced_bce = 0.5 * (
                        F.softplus(-logits[positive]).mean()
                        + F.softplus(logits[negative]).mean()
                    )
                    penalty = args.mil_weight_decay * (
                        scorer.atom_weights.square().mean()
                        + scorer.energy_weight.square()
                        + F.softplus(scorer.novelty_raw).square()
                    )
                    loss = balanced_bce + penalty
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_curve.append(float(loss.detach()))
                return scorer, loss_curve

            @torch.no_grad()
            def predict_mil_features(
                scorer: MILNoveltyScorer, graph_indices: np.ndarray
            ) -> np.ndarray:
                rows_out = np.zeros((len(graph_indices), 5), dtype=np.float64)
                for row_i, graph_i in enumerate(graph_indices.tolist()):
                    lo = int(offsets[graph_i])
                    hi = int(offsets[graph_i + 1])
                    node_scores = scorer.node_scores(
                        instance_tensor[lo:hi]
                    )
                    n_nodes = int(node_scores.numel())
                    maximum = node_scores.max()
                    soft_max = maximum + args.mil_temperature * (
                        torch.log(
                            torch.exp(
                                (node_scores - maximum) / args.mil_temperature
                            ).mean().clamp_min(1e-12)
                        )
                    )
                    top3 = node_scores.topk(min(3, n_nodes)).values.mean()
                    top10_count = max(1, int(np.ceil(0.10 * n_nodes)))
                    top10 = node_scores.topk(top10_count).values.mean()
                    witness_mass = torch.sigmoid(
                        node_scores + scorer.bias
                    ).mean()
                    rows_out[row_i] = [
                        float(soft_max + scorer.bias),
                        float(maximum),
                        float(top3),
                        float(top10),
                        float(witness_mass),
                    ]
                return rows_out

            full_scorer, full_loss_curve = fit_mil_scorer(fit_indices)
            all_graph_indices = np.arange(len(graphs), dtype=np.int64)
            mil_rows = predict_mil_features(full_scorer, all_graph_indices)

            labels_fit = y_np[fit_indices]
            max_crossfit = int(
                min(
                    args.mil_crossfit_folds,
                    np.sum(labels_fit == 0),
                    np.sum(labels_fit == 1),
                )
            )
            if max_crossfit < 2:
                raise RuntimeError("too few examples for MIL cross-fitting")
            splitter = StratifiedKFold(
                n_splits=max_crossfit, shuffle=True,
                random_state=args.seed + 271828,
            )
            crossfit_models: list[dict[str, Any]] = []
            for crossfit_i, (train_pos, heldout_pos) in enumerate(
                splitter.split(fit_indices, labels_fit)
            ):
                crossfit_train = fit_indices[train_pos]
                crossfit_heldout = fit_indices[heldout_pos]
                scorer, loss_curve = fit_mil_scorer(crossfit_train)
                mil_rows[crossfit_heldout] = predict_mil_features(
                    scorer, crossfit_heldout
                )
                crossfit_models.append({
                    "fold": int(crossfit_i),
                    "fit_graphs": int(len(crossfit_train)),
                    "heldout_graphs": int(len(crossfit_heldout)),
                    "final_loss": float(loss_curve[-1]),
                    "novelty_weight": float(
                        F.softplus(scorer.novelty_raw).detach()
                    ),
                    "energy_weight": float(scorer.energy_weight.detach()),
                    "atom_weights": scorer.atom_weights.detach().tolist(),
                })

            mil_mean = mil_rows[fit_indices].mean(axis=0, keepdims=True)
            mil_scale = np.maximum(
                mil_rows[fit_indices].std(axis=0, keepdims=True), 1e-6
            )
            mil_scaled = (mil_rows - mil_mean) / mil_scale
            graph_code_features = torch.tensor(
                mil_scaled, dtype=torch.float32, device=device
            )
            oof_auc = float(roc_auc_score(
                labels_fit, mil_rows[fit_indices, 0]
            ))
            full_novelty_weight = float(
                F.softplus(full_scorer.novelty_raw).detach()
            )
            stats["graph_code_readout"] = {
                "kind": "crossfit_graph_label_mil_positive_novelty",
                "feature_dim": 5,
                "instance_dim": int(instance_features.shape[1]),
                "fit_graph_count": int(len(fit_indices)),
                "fit_positive_graph_count": int(np.sum(labels_fit == 1)),
                "fit_negative_graph_count": int(np.sum(labels_fit == 0)),
                "temperature": float(args.mil_temperature),
                "fit_epochs": int(args.mil_fit_epochs),
                "fit_lr": float(args.mil_fit_lr),
                "weight_decay": float(args.mil_weight_decay),
                "crossfit_folds": int(max_crossfit),
                "crossfit_oof_auc": oof_auc,
                "full_fit_final_loss": float(full_loss_curve[-1]),
                "full_fit_novelty_weight": full_novelty_weight,
                "full_fit_energy_weight": float(
                    full_scorer.energy_weight.detach()
                ),
                "full_fit_bias": float(full_scorer.bias.detach()),
                "full_fit_atom_weights": (
                    full_scorer.atom_weights.detach().tolist()
                ),
                "crossfit_models": crossfit_models,
                "reconstruction_fit_mean": recon_mean,
                "reconstruction_fit_scale": recon_scale,
                "energy_fit_mean": energy_mean,
                "energy_fit_scale": energy_scale,
                "feature_names": [
                    "bag_soft_top_logit", "node_score_max",
                    "node_score_top3_mean", "node_score_top10pct_mean",
                    "positive_witness_mass",
                ],
                "fit_scaled_abs_max": float(
                    np.abs(mil_scaled[fit_indices]).max()
                ),
            }
        elif args.token_fusion == "ksvd_precomputed_graph_residual":
            if precomputed_graph_features_raw is None:
                raise RuntimeError("missing precomputed graph features")
            graph_mean = precomputed_graph_features_raw[fit_indices].mean(
                axis=0, keepdims=True
            )
            graph_scale = np.maximum(
                precomputed_graph_features_raw[fit_indices].std(
                    axis=0, keepdims=True
                ),
                1e-6,
            )
            graph_scaled = (
                precomputed_graph_features_raw - graph_mean
            ) / graph_scale
            graph_code_features = torch.tensor(
                graph_scaled, dtype=torch.float32, device=device
            )
            stats["graph_code_readout"] = {
                "kind": "precomputed_fit_standardized_graph_residual",
                "feature_dim": int(precomputed_graph_features_raw.shape[1]),
                "fit_graph_count": int(len(fit_indices)),
                "fit_scaled_abs_max": float(
                    np.abs(graph_scaled[fit_indices]).max()
                ),
            }
        return (
            torch.tensor(scaled, dtype=torch.float32, device=device),
            stats,
            torch.tensor(
                code_delta_scale.reshape(-1), dtype=torch.float32, device=device
            ),
            graph_code_features,
        )

    def train_epoch(
        model, optimizer, loader, token_tensor, code_delta_scale=None,
        graph_code_features=None, first_batches=None, ema_state=None
    ):
        model.train()
        total, n_seen = 0.0, 0
        for batch in loader:
            if first_batches is not None and len(first_batches) < 3:
                first_batches.append(batch.idx.view(-1).tolist())
            batch = batch.to(device)
            if (
                args.graph_residual_training in {"independent", "fixed_probe"}
                and model.use_tokens
                and args.token_fusion in graph_residual_fusions
            ):
                pred, auxiliary_loss, base_pred, residual_pred = model(
                    batch, token_tensor, code_delta_scale,
                    graph_code_features=graph_code_features,
                    return_aux=True, return_components=True,
                )
            else:
                pred, auxiliary_loss = model(
                    batch, token_tensor, code_delta_scale,
                    graph_code_features=graph_code_features, return_aux=True
                )
                base_pred = residual_pred = None
            pred = pred.view(-1)
            labels = batch.y.view(-1).float()
            mask = (labels == 0) | (labels == 1)
            if not bool(mask.any()):
                continue
            if base_pred is not None and residual_pred is not None:
                # The fixed ensemble is used only for evaluation.
                base_loss = F.binary_cross_entropy_with_logits(
                    base_pred.view(-1)[mask], labels[mask]
                )
                if args.graph_residual_training == "fixed_probe":
                    loss = base_loss + auxiliary_loss
                    residual_scores = None
                else:
                    residual_scores = residual_pred.view(-1)[mask]
                    residual_labels = labels[mask]
                    if args.graph_residual_independent_loss == "pairwise_logistic":
                        positive_scores = residual_scores[residual_labels == 1]
                        negative_scores = residual_scores[residual_labels == 0]
                        if positive_scores.numel() and negative_scores.numel():
                            residual_loss = F.softplus(
                                -(positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0))
                            ).mean()
                        else:
                            residual_loss = F.binary_cross_entropy_with_logits(
                                residual_scores, residual_labels
                            )
                    else:
                        residual_loss = F.binary_cross_entropy_with_logits(
                            residual_scores, residual_labels
                        )
                    loss = base_loss + residual_loss + auxiliary_loss
            else:
                loss = (
                    F.binary_cross_entropy_with_logits(pred[mask], labels[mask])
                    + auxiliary_loss
                )
            if model.use_tokens and args.token_gate_l2 > 0.0:
                effective_gates = args.token_gate_scale * torch.tanh(model.token_gates)
                loss = loss + args.token_gate_l2 * effective_gates.square().sum()
                if args.token_fusion in {"bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject"}:
                    effective_interaction_gates = (
                        args.token_gate_scale * torch.tanh(model.interaction_gates)
                    )
                    loss = loss + args.token_gate_l2 * (
                        effective_interaction_gates.square().sum()
                    )
                if args.token_fusion == "ksvd_context_inject":
                    effective_context_gates = (
                        args.context_gate_scale * torch.tanh(model.context_gates)
                    )
                    loss = loss + args.token_gate_l2 * (
                        effective_context_gates.square().sum()
                    )
                if args.token_fusion == "ring_dual_inject":
                    effective_ring_gates = args.ring_gate_scale * torch.tanh(model.ring_gates)
                    loss = loss + args.token_gate_l2 * effective_ring_gates.square().sum()
                elif args.token_fusion == "ring_dual_readout":
                    effective_ring_gate = args.ring_gate_scale * torch.tanh(model.ring_readout_gate)
                    loss = loss + args.token_gate_l2 * effective_ring_gate.square()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_parameter_ema(model, ema_state)
            n = int(mask.sum())
            total += float(loss.item()) * n
            n_seen += n
        return total / max(n_seen, 1)

    def evaluate(
        model, loader, token_tensor, code_delta_scale=None,
        graph_code_features=None, *,
        return_predictions: bool = False, return_component_aucs: bool = False,
        ema_state=None
    ):
        parameter_backup = None
        if ema_state is not None:
            parameter_backup = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if name in ema_state
            }
            install_parameter_ema(model, ema_state)
        model.eval()
        ys, preds = [], []
        base_preds, residual_preds = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if return_component_aucs:
                    pred, base_pred, residual_pred = model(
                        batch, token_tensor, code_delta_scale,
                        graph_code_features=graph_code_features,
                        return_components=True,
                    )
                    if residual_pred is None:
                        raise RuntimeError(
                            "component AUCs require a graph-residual fusion"
                        )
                    base_preds.append(base_pred.view(-1).sigmoid().cpu())
                    residual_preds.append(
                        residual_pred.view(-1).sigmoid().cpu()
                    )
                else:
                    pred = model(
                        batch, token_tensor, code_delta_scale,
                        graph_code_features=graph_code_features,
                    )
                pred = pred.view(-1)
                labels = batch.y.view(-1).float()
                mask = (labels == 0) | (labels == 1)
                ys.append(labels[mask].cpu())
                preds.append(pred[mask].sigmoid().cpu())
                if return_component_aucs:
                    base_preds[-1] = base_preds[-1][mask.cpu()]
                    residual_preds[-1] = residual_preds[-1][mask.cpu()]
        y_true = torch.cat(ys).numpy().reshape(-1, 1)
        y_pred = torch.cat(preds).numpy().reshape(-1, 1)
        auc = float(evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"])
        component_aucs = None
        if return_component_aucs:
            base_y_pred = torch.cat(base_preds).numpy().reshape(-1, 1)
            residual_y_pred = torch.cat(residual_preds).numpy().reshape(-1, 1)
            component_aucs = {
                "base_inner_valid_auc": float(
                    evaluator.eval(
                        {"y_true": y_true, "y_pred": base_y_pred}
                    )["rocauc"]
                ),
                "residual_inner_valid_auc": float(
                    evaluator.eval(
                        {"y_true": y_true, "y_pred": residual_y_pred}
                    )["rocauc"]
                ),
            }
        if parameter_backup is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in parameter_backup:
                        param.copy_(parameter_backup[name])
        if return_predictions:
            output = (
                auc, y_true.reshape(-1).tolist(), y_pred.reshape(-1).tolist()
            )
            if return_component_aucs:
                return (*output, component_aucs)
            return output
        if return_component_aucs:
            return auc, component_aucs
        return auc

    def install_fixed_graph_probe(model, graph_features, fit_indices):
        if args.graph_residual_training != "fixed_probe":
            return None
        if args.graph_residual_hidden != 0:
            raise ValueError("fixed_probe requires --graph-residual-hidden 0")
        if graph_features is None:
            raise ValueError("fixed_probe requires graph residual features")
        x_fit = graph_features[fit_indices].detach().cpu().numpy()
        y_fit = y_np[fit_indices]
        probe = LogisticRegression(
            penalty="l2", C=args.graph_residual_probe_c,
            class_weight="balanced", solver="lbfgs", max_iter=2000,
            tol=1e-8, random_state=0,
        )
        probe.fit(x_fit, y_fit)
        with torch.no_grad():
            model.graph_residual_head.weight.copy_(
                torch.tensor(probe.coef_, dtype=torch.float32, device=device)
            )
            model.graph_residual_head.bias.copy_(
                torch.tensor(probe.intercept_, dtype=torch.float32, device=device)
            )
        return {
            "classifier": "balanced_l2_logistic_regression",
            "c": args.graph_residual_probe_c,
            "iterations": int(probe.n_iter_[0]),
            "coefficient_norm": float(np.linalg.norm(probe.coef_)),
            "intercept": float(probe.intercept_[0]),
            "fit_graph_count": int(len(fit_indices)),
            "fit_positive_count": int(y_fit.sum()),
        }

    # Phase 1: train-only epoch selection.
    (
        selection_tokens,
        selection_token_preprocessing,
        selection_code_delta_scale,
        selection_graph_code_features,
    ) = normalize_tokens(inner_tr)
    selection_train_loader = make_loader(inner_tr, True, phase_offset=0)
    selection_valid_loader = make_loader(inner_va, False, phase_offset=0)
    model, optimizer = make_model_optimizer(args.seed)
    selection_fixed_probe = install_fixed_graph_probe(
        model, selection_graph_code_features, inner_tr
    )
    selection_ema = make_parameter_ema(model)
    selection_initial_base_hash = state_hash(model, base_only=True)
    selection_initial_token_proj_hash = (
        state_hash(model.token_proj) if model.use_tokens else None
    )
    selection_initial_graph_residual_hash = (
        state_hash(model.graph_residual_head)
        if model.use_tokens and args.token_fusion in graph_residual_fusions
        else None
    )
    selection_initial_graph_residual_nonlinear_hash = (
        state_hash(model.graph_residual_nonlinear)
        if (
            model.use_tokens
            and args.token_fusion in graph_residual_fusions
            and model.graph_residual_nonlinear is not None
        )
        else None
    )
    selection_initial_interaction_gates = (
        model.interaction_gates.detach().cpu().tolist()
        if model.use_tokens and args.token_fusion in {"bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject"}
        else None
    )
    selection_initial_refined_gates = (
        model.refined_gates.detach().cpu().tolist()
        if model.use_tokens and args.token_fusion == "unrolled_ksvd_inject"
        else None
    )
    selection_initial_unrolled_mix = (
        float(torch.tanh(model.unrolled_mix_raw).detach().cpu())
        if model.use_tokens and args.token_fusion == "unrolled_ksvd_mix"
        else None
    )
    selection_initial_context_gates = (
        (
            model.context_gates if args.token_fusion == "ksvd_context_inject"
            else model.token_gates
        ).detach().cpu().tolist()
        if model.use_tokens and args.token_fusion in {"ksvd_context_inject", "ksvd_context_tied"}
        else None
    )
    selection_initial_motif_slot_gate = (
        float(model.motif_slot.gate.detach().cpu())
        if model.use_tokens and args.token_fusion in motif_slot_fusions else None
    )
    selection_initial_motif_occurrence_gate = (
        float(model.motif_occurrence.gate.detach().cpu())
        if model.use_tokens and args.token_fusion in motif_occurrence_fusions else None
    )
    selection_initial_motif_ego_gate = (
        float(model.motif_ego.gate.detach().cpu())
        if model.use_tokens and args.token_fusion in motif_ego_fusions else None
    )
    selection_first_batches: list[list[int]] = []
    history: list[dict[str, Any]] = []
    best_epoch, best_inner_auc = 1, -1.0
    for epoch in range(1, args.epochs + 1):
        set_token_group_lr(model, optimizer, epoch)
        loss = train_epoch(
            model, optimizer, selection_train_loader, selection_tokens,
            selection_code_delta_scale,
            selection_graph_code_features,
            selection_first_batches if epoch == 1 else None, selection_ema,
        )
        component_eval = (
            args.graph_residual_training in {"independent", "fixed_probe"}
            and model.use_tokens
            and args.token_fusion in graph_residual_fusions
        )
        if args.save_inner_predictions:
            if component_eval:
                (
                    inner_auc, inner_y, inner_predictions, component_aucs
                ) = evaluate(
                    model, selection_valid_loader, selection_tokens,
                    selection_code_delta_scale, selection_graph_code_features,
                    return_predictions=True, return_component_aucs=True,
                    ema_state=selection_ema,
                )
            else:
                inner_auc, inner_y, inner_predictions = evaluate(
                    model, selection_valid_loader, selection_tokens,
                    selection_code_delta_scale, selection_graph_code_features,
                    return_predictions=True,
                    ema_state=selection_ema,
                )
                component_aucs = None
        else:
            if component_eval:
                inner_auc, component_aucs = evaluate(
                    model, selection_valid_loader, selection_tokens,
                    selection_code_delta_scale, selection_graph_code_features,
                    return_component_aucs=True, ema_state=selection_ema
                )
            else:
                inner_auc = evaluate(
                    model, selection_valid_loader, selection_tokens,
                    selection_code_delta_scale, selection_graph_code_features,
                    ema_state=selection_ema
                )
                component_aucs = None
            inner_y, inner_predictions = None, None
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss,
            "inner_valid_auc": inner_auc,
        }
        if args.save_inner_predictions:
            row["inner_valid_y"] = inner_y
            row["inner_valid_predictions"] = inner_predictions
        if component_aucs is not None:
            row.update(component_aucs)
        if args.jk_readout == "gated_sum":
            row["jk_gates"] = (
                torch.tanh(model.gine.jk_gates).detach().cpu().tolist()
            )
        if args.graph_readout == "gated_sum_mean":
            row["sum_mean_gate"] = float(
                torch.tanh(model.gine.sum_mean_gate).detach().cpu()
            )
        if model.use_tokens and args.token_fusion in graph_residual_fusions:
            row["graph_residual_weight_norm"] = float(
                model.graph_residual_head.weight.detach().norm().cpu()
            )
            row["graph_residual_bias"] = float(
                model.graph_residual_head.bias.detach().cpu().item()
            )
            if model.graph_residual_nonlinear is not None:
                row["graph_residual_hidden_weight_norm"] = float(
                    model.graph_residual_nonlinear[0].weight.detach().norm().cpu()
                )
                row["graph_residual_nonlinear_output_weight_norm"] = float(
                    model.graph_residual_nonlinear[-1].weight.detach().norm().cpu()
                )
                row["graph_residual_nonlinear_output_bias"] = float(
                    model.graph_residual_nonlinear[-1].bias.detach().cpu().item()
                )
        if model.use_tokens:
            row["token_gates"] = (args.token_gate_scale * torch.tanh(model.token_gates)).detach().cpu().tolist()
            if args.token_fusion in {"bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject"}:
                row["interaction_gates"] = (
                    args.token_gate_scale * torch.tanh(model.interaction_gates)
                ).detach().cpu().tolist()
            if args.token_fusion in unrolled_fusions:
                eta, lambdas = model.unrolled_parameters()
                if args.token_fusion == "unrolled_ksvd_inject":
                    row["refined_gates"] = (
                        args.token_gate_scale * torch.tanh(model.refined_gates)
                    ).detach().cpu().tolist()
                else:
                    row["unrolled_mix"] = float(
                        torch.tanh(model.unrolled_mix_raw).detach().cpu()
                    )
                row["unrolled_step_sizes"] = eta.detach().cpu().tolist()
                row["unrolled_lambda_mean"] = (
                    lambdas.mean(dim=1).detach().cpu().tolist()
                )
                row["unrolled_lambda_min"] = (
                    lambdas.amin(dim=1).detach().cpu().tolist()
                )
                row["unrolled_lambda_max"] = (
                    lambdas.amax(dim=1).detach().cpu().tolist()
                )
            if args.token_fusion == "adaptive_inject":
                row["token_router_weight_norms"] = [
                    float(router.weight.detach().norm().cpu())
                    for router in model.token_routers
                ]
            if args.token_fusion in {"ksvd_context_inject", "ksvd_context_tied"}:
                context_gate_parameters = (
                    model.context_gates
                    if args.token_fusion == "ksvd_context_inject"
                    else model.token_gates
                )
                row["context_gates"] = (
                    args.context_gate_scale * torch.tanh(context_gate_parameters)
                ).detach().cpu().tolist()
            if args.token_fusion == "ring_dual_inject":
                row["ring_gates"] = (
                    args.ring_gate_scale * torch.tanh(model.ring_gates)
                ).detach().cpu().tolist()
            elif args.token_fusion == "ring_dual_readout":
                row["ring_readout_gate"] = float(
                    (args.ring_gate_scale * torch.tanh(model.ring_readout_gate)).detach().cpu()
                )
            if args.token_fusion == "inject_motif_readout":
                row["motif_gate"] = float(torch.tanh(model.motif_gate).detach().cpu())
            if args.token_fusion in motif_slot_fusions:
                row["motif_slot_gate"] = float(
                    (args.token_gate_scale * torch.tanh(model.motif_slot.gate)).detach().cpu()
                )
                row["motif_slot_atom_gates"] = (
                    None if model.motif_slot.atom_gates is None else
                    (args.token_gate_scale * torch.tanh(model.motif_slot.atom_gates))
                    .detach().cpu().tolist()
                )
            if args.token_fusion in motif_occurrence_fusions:
                row["motif_occurrence_gate"] = float(
                    (
                        args.token_gate_scale
                        * torch.tanh(model.motif_occurrence.gate)
                    ).detach().cpu()
                )
            if args.token_fusion in motif_ego_fusions:
                row["motif_ego_gate"] = float(
                    (
                        args.token_gate_scale * torch.tanh(model.motif_ego.gate)
                    ).detach().cpu()
                )
            if args.token_fusion == "ksvd_atom_additive_readout":
                effective_atom_weights = torch.tanh(
                    model.ksvd_atom_weights
                ).detach().cpu()
                row["ksvd_atom_weights"] = effective_atom_weights.tolist()
                row["ksvd_atom_weight_norm"] = float(
                    effective_atom_weights.norm()
                )
                row["ksvd_atom_score_norm"] = float(
                    model.ksvd_atom_score.weight.detach().norm().cpu()
                )
            elif args.token_fusion == "ksvd_atom_specific_readout":
                score_weights = model.ksvd_atom_score_weights.detach().cpu()
                row["ksvd_atom_specific_score_norm"] = float(
                    score_weights.norm()
                )
                row["ksvd_atom_specific_per_atom_norms"] = (
                    score_weights.norm(dim=1).tolist()
                )
        history.append(row)
        if inner_auc > best_inner_auc:
            best_epoch, best_inner_auc = epoch, inner_auc
        if epoch == 1 or epoch % 5 == 0:
            gates = "" if not model.use_tokens else f" gates={row['token_gates']}"
            if args.jk_readout == "gated_sum":
                gates += f" jk_gates={row['jk_gates']}"
            if args.graph_readout == "gated_sum_mean":
                gates += f" sum_mean_gate={row['sum_mean_gate']:.4f}"
            if model.use_tokens and args.token_fusion in graph_residual_fusions:
                gates += (
                    f" graph_residual_norm={row['graph_residual_weight_norm']:.4f}"
                    f" bias={row['graph_residual_bias']:.4f}"
                )
                if component_aucs is not None:
                    gates += (
                        f" base_auc={row['base_inner_valid_auc']:.4f}"
                        f" residual_auc={row['residual_inner_valid_auc']:.4f}"
                    )
            if model.use_tokens and args.token_fusion in {"bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject"}:
                gates += f" interaction_gates={row['interaction_gates']}"
            if model.use_tokens and args.token_fusion in unrolled_fusions:
                if args.token_fusion == "unrolled_ksvd_inject":
                    gates += f" refined_gates={row['refined_gates']}"
                else:
                    gates += f" unrolled_mix={row['unrolled_mix']:.4f}"
                gates += (
                    f" eta={row['unrolled_step_sizes']}"
                    f" lambda_mean={row['unrolled_lambda_mean']}"
                )
            if model.use_tokens and args.token_fusion == "adaptive_inject":
                gates += f" router_norms={row['token_router_weight_norms']}"
            if model.use_tokens and args.token_fusion in {"ksvd_context_inject", "ksvd_context_tied"}:
                gates += f" context_gates={row['context_gates']}"
            if model.use_tokens and args.token_fusion == "ring_dual_inject":
                gates += f" ring_gates={row['ring_gates']}"
            elif model.use_tokens and args.token_fusion == "ring_dual_readout":
                gates += f" ring_readout_gate={row['ring_readout_gate']:.4f}"
            if model.use_tokens and args.token_fusion == "ksvd_atom_additive_readout":
                gates += (
                    f" atom_weight_norm={row['ksvd_atom_weight_norm']:.4f}"
                    f" atom_score_norm={row['ksvd_atom_score_norm']:.4f}"
                )
            elif model.use_tokens and args.token_fusion == "ksvd_atom_specific_readout":
                gates += (
                    f" atom_specific_norm="
                    f"{row['ksvd_atom_specific_score_norm']:.4f}"
                )
            log(
                f"select ep={epoch:02d} loss={loss:.4f} inner={inner_auc:.4f} "
                f"best={best_inner_auc:.4f}@{best_epoch}{gates}"
            )

    if args.selection_only:
        result = {
            "protocol_id": "molhiv-localized-node-token-inner-selection-only-v1",
            "family": args.family,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "inner_split_seed": args.inner_split_seed,
            "inner_valid_fraction": args.inner_valid_fraction,
            "max_selection_epochs": args.epochs,
            "save_inner_predictions": args.save_inner_predictions,
            "selected_epoch": best_epoch,
            "best_inner_valid_auc": best_inner_auc,
            "official_valid_auc": None,
            "official_valid_evaluations": 0,
            "capture_official_valid_predictions": args.capture_official_valid_predictions,
            "official_test_auc": None,
            "official_test_evaluations": 0,
            "test_policy": "official valid and official test were not evaluated",
            "n_used": len(graphs),
            "n_official_train": int(len(tr)),
            "n_official_valid": int(len(va)),
            "n_inner_train": int(len(inner_tr)),
            "n_inner_valid": int(len(inner_va)),
            "inner_split_protocol": inner_split_protocol,
            "inner_split_cache": (
                str(inner_split_cache_path) if inner_split_cache_path is not None else None
            ),
            "inner_fold": args.inner_fold if inner_split_cache_path is not None else None,
            "n_official_train_pos": int(y_np[tr].sum()),
            "n_official_valid_pos": int(y_np[va].sum()),
            "n_inner_train_pos": int(y_np[inner_tr].sum()),
            "n_inner_valid_pos": int(y_np[inner_va].sum()),
            "hidden": args.hidden,
            "layers": args.layers,
            "residual_gine": args.residual_gine,
            "jk_readout": args.jk_readout,
            "graph_readout": args.graph_readout,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "token_lr_scale": args.token_lr_scale,
            "token_weight_decay": args.token_weight_decay,
            "token_warmup_epochs": args.token_warmup_epochs,
            "parameter_ema_decay": args.parameter_ema_decay,
            "token_dropout": args.token_dropout,
            "token_gate_scale": args.token_gate_scale,
            "ring_gate_scale": args.ring_gate_scale,
            "context_gate_scale": args.context_gate_scale,
            "token_gate_l2": args.token_gate_l2,
            "ksvd_aux_weight": args.ksvd_aux_weight,
            "ksvd_aux_epochs": args.ksvd_aux_epochs,
            "token_inject_layers": args.token_inject_layers,
            "token_normalization": args.token_normalization,
            "token_zscore_clip": args.token_zscore_clip,
            "token_prevalence_power": args.token_prevalence_power,
            "token_min_graph_frequency": args.token_min_graph_frequency,
            "token_preprocessing": selection_token_preprocessing,
            "selection_fixed_probe": selection_fixed_probe,
            "token_channels": args.token_channels,
            "token_fusion": args.token_fusion,
            "graph_residual_hidden": args.graph_residual_hidden,
            "graph_residual_probe_c": args.graph_residual_probe_c,
            "graph_residual_training": args.graph_residual_training,
            "graph_residual_combination": args.graph_residual_combination,
            "graph_residual_independent_loss": (
                args.graph_residual_independent_loss
            ),
            "token_interaction_rank": args.token_interaction_rank,
            "motif_slot_rank": args.motif_slot_rank,
            "motif_geometry_assignment": args.motif_geometry_assignment,
            "motif_geometry_atom_normalization": args.motif_geometry_atom_normalization,
            "motif_atom_id_source": args.motif_atom_id_source,
            "motif_gate_mode": args.motif_gate_mode,
            "motif_gate_wake_init": args.motif_gate_wake_init,
            "motif_slot_backend": args.motif_slot_backend,
            "motif_slot_shuffle_seed": args.motif_slot_shuffle_seed,
            "motif_occurrence_radius": args.motif_occurrence_radius,
            "motif_occurrence_stats": motif_occurrence_stats,
            "motif_ego_radius": args.motif_ego_radius,
            "motif_ego_confidence": args.motif_ego_confidence,
            "motif_ego_min_atom_index": args.motif_ego_min_atom_index,
            "motif_ego_min_dominance": args.motif_ego_min_dominance,
            "token_interaction_support_mask": args.token_interaction_support_mask,
            "unrolled_steps": args.unrolled_steps,
            "unrolled_step_init": args.unrolled_step_init,
            "unrolled_lambda_init": args.unrolled_lambda_init,
            "unrolled_threshold_mode": args.unrolled_threshold_mode,
            "unrolled_recon_weight": args.unrolled_recon_weight,
            "unrolled_sparse_weight": args.unrolled_sparse_weight,
            "base_token_dim": base_token_dim,
            "primary_token_dim": dual_base_token_dim,
            "ring_token_dim": ring_token_dim,
            "token_cache": str(cache_path),
            "token_dim": token_dim,
            "selection_history": history,
            "trainable_parameter_count": model.trainable_parameter_count,
            "base_parameter_count": model.base_parameter_count,
            "token_parameter_count": model.token_parameter_count,
            "final_jk_gates": None,
            "final_sum_mean_gate": None,
            "final_train_losses": None,
            "final_token_gates": None,
            "final_interaction_gates": None,
            "final_refined_gates": None,
            "final_unrolled_mix": None,
            "final_unrolled_step_sizes": None,
            "final_unrolled_lambdas": None,
            "final_context_gates": None,
            "final_ring_gates": None,
            "final_ring_readout_gate": None,
            "final_motif_gate": None,
            "final_motif_slot_gate": (
                float((args.token_gate_scale * torch.tanh(model.motif_slot.gate)).detach().cpu())
                if model.use_tokens and args.token_fusion in motif_slot_fusions else None
            ),
            "final_motif_slot_atom_gates": (
                (args.token_gate_scale * torch.tanh(model.motif_slot.atom_gates))
                .detach().cpu().tolist()
                if (
                    model.use_tokens
                    and args.token_fusion in motif_slot_fusions
                    and model.motif_slot.atom_gates is not None
                ) else None
            ),
            "final_motif_occurrence_gate": (
                float(
                    (
                        args.token_gate_scale
                        * torch.tanh(model.motif_occurrence.gate)
                    ).detach().cpu()
                )
                if model.use_tokens and args.token_fusion in motif_occurrence_fusions
                else None
            ),
            "final_motif_ego_gate": (
                float(
                    (
                        args.token_gate_scale * torch.tanh(model.motif_ego.gate)
                    ).detach().cpu()
                )
                if model.use_tokens and args.token_fusion in motif_ego_fusions
                else None
            ),
            "audit_fingerprints": {
                "original_indices_sha256": array_hash(original_indices),
                "official_train_sha256": array_hash(tr),
                "official_valid_sha256": array_hash(va),
                "inner_train_sha256": array_hash(inner_tr),
                "inner_valid_sha256": array_hash(inner_va),
                "selection_initial_base_state_sha256": selection_initial_base_hash,
                "selection_initial_token_proj_state_sha256": selection_initial_token_proj_hash,
                "selection_initial_graph_residual_state_sha256": (
                    selection_initial_graph_residual_hash
                ),
                "selection_initial_graph_residual_nonlinear_state_sha256": (
                    selection_initial_graph_residual_nonlinear_hash
                ),
                "selection_initial_interaction_gates": selection_initial_interaction_gates,
                "selection_initial_refined_gates": selection_initial_refined_gates,
                "selection_initial_unrolled_mix": selection_initial_unrolled_mix,
                "selection_initial_context_gates": selection_initial_context_gates,
                "selection_initial_motif_slot_gate": selection_initial_motif_slot_gate,
                "selection_initial_motif_occurrence_gate": (
                    selection_initial_motif_occurrence_gate
                ),
                "selection_initial_motif_ego_gate": selection_initial_motif_ego_gate,
                "selection_first_three_batches": selection_first_batches,
                "selection_loader_seed": args.seed + 9173,
            },
            "elapsed_sec": time.time() - t0,
            "env": check_env(),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(
            f"selection-only best inner={best_inner_auc:.6f}@{best_epoch}; "
            f"official valid evaluations=0; wrote {output}"
        )
        return

    # Phase 2: exact reset, all official train, one official-valid evaluation.
    (
        final_tokens,
        final_token_preprocessing,
        final_code_delta_scale,
        final_graph_code_features,
    ) = normalize_tokens(tr)
    final_train_loader = make_loader(tr, True, phase_offset=100_000)
    official_valid_loader = make_loader(va, False, phase_offset=100_000)
    official_test_loader = (
        make_loader(te, False, phase_offset=100_000) if args.evaluate_test else None
    )
    model, optimizer = make_model_optimizer(args.seed)
    final_fixed_probe = install_fixed_graph_probe(
        model, final_graph_code_features, tr
    )
    final_ema = make_parameter_ema(model)
    final_initial_base_hash = state_hash(model, base_only=True)
    final_initial_token_proj_hash = (
        state_hash(model.token_proj) if model.use_tokens else None
    )
    final_first_batches: list[list[int]] = []
    final_losses = []
    for epoch in range(1, best_epoch + 1):
        set_token_group_lr(model, optimizer, epoch)
        loss = train_epoch(
            model, optimizer, final_train_loader, final_tokens,
            final_code_delta_scale,
            final_graph_code_features,
            final_first_batches if epoch == 1 else None, final_ema,
        )
        final_losses.append(loss)
        if epoch == 1 or epoch == best_epoch or epoch % 5 == 0:
            log(f"retrain ep={epoch:02d}/{best_epoch} loss={loss:.4f}")
    install_parameter_ema(model, final_ema)
    if args.evaluate_test:
        official_valid_auc, official_valid_y, official_valid_predictions = evaluate(
            model, official_valid_loader, final_tokens, final_code_delta_scale,
            final_graph_code_features,
            return_predictions=True
        )
        official_test_auc, official_test_y, official_test_predictions = evaluate(
            model, official_test_loader, final_tokens, final_code_delta_scale,
            final_graph_code_features,
            return_predictions=True
        )
    else:
        if args.capture_official_valid_predictions:
            (
                official_valid_auc, official_valid_y, official_valid_predictions
            ) = evaluate(
                model, official_valid_loader, final_tokens, final_code_delta_scale,
                final_graph_code_features,
                return_predictions=True,
            )
        else:
            official_valid_auc = evaluate(
                model, official_valid_loader, final_tokens,
                final_code_delta_scale, final_graph_code_features,
            )
            official_valid_y = official_valid_predictions = None
        official_test_auc = None
        official_test_y = official_test_predictions = None
    final_jk_gates = (
        torch.tanh(model.gine.jk_gates).detach().cpu().tolist()
        if args.jk_readout == "gated_sum" else None
    )
    final_sum_mean_gate = (
        float(torch.tanh(model.gine.sum_mean_gate).detach().cpu())
        if args.graph_readout == "gated_sum_mean" else None
    )
    final_gates = (
        (args.token_gate_scale * torch.tanh(model.token_gates)).detach().cpu().tolist()
        if model.use_tokens else None
    )
    final_token_router_weight_norms = (
        [float(router.weight.detach().norm().cpu()) for router in model.token_routers]
        if model.use_tokens and args.token_fusion == "adaptive_inject"
        else None
    )
    final_interaction_gates = (
        (args.token_gate_scale * torch.tanh(model.interaction_gates)).detach().cpu().tolist()
        if model.use_tokens and args.token_fusion in {"bilinear_inject", "edge_bilinear_inject", "dynamic_bilinear_inject"}
        else None
    )
    final_context_gates = (
        (args.context_gate_scale * torch.tanh(
            model.context_gates
            if args.token_fusion == "ksvd_context_inject"
            else model.token_gates
        ))
        .detach().cpu().tolist()
        if model.use_tokens and args.token_fusion in {"ksvd_context_inject", "ksvd_context_tied"}
        else None
    )
    if model.use_tokens and args.token_fusion in unrolled_fusions:
        final_eta, final_lambdas_tensor = model.unrolled_parameters()
        final_refined_gates = (
            (args.token_gate_scale * torch.tanh(model.refined_gates))
            .detach().cpu().tolist()
            if args.token_fusion == "unrolled_ksvd_inject"
            else None
        )
        final_unrolled_mix = (
            float(torch.tanh(model.unrolled_mix_raw).detach().cpu())
            if args.token_fusion == "unrolled_ksvd_mix"
            else None
        )
        final_unrolled_step_sizes = final_eta.detach().cpu().tolist()
        final_unrolled_lambdas = final_lambdas_tensor.detach().cpu().tolist()
    else:
        final_refined_gates = None
        final_unrolled_mix = None
        final_unrolled_step_sizes = None
        final_unrolled_lambdas = None
    final_ring_gates = (
        (args.ring_gate_scale * torch.tanh(model.ring_gates)).detach().cpu().tolist()
        if model.use_tokens and args.token_fusion == "ring_dual_inject"
        else None
    )
    final_ring_readout_gate = (
        float((args.ring_gate_scale * torch.tanh(model.ring_readout_gate)).detach().cpu())
        if model.use_tokens and args.token_fusion == "ring_dual_readout"
        else None
    )
    final_motif_gate = (
        float(torch.tanh(model.motif_gate).detach().cpu())
        if model.use_tokens and args.token_fusion == "inject_motif_readout"
        else None
    )
    final_ksvd_atom_weights = (
        torch.tanh(model.ksvd_atom_weights).detach().cpu().tolist()
        if model.use_tokens and args.token_fusion == "ksvd_atom_additive_readout"
        else None
    )
    final_ksvd_atom_score_norm = (
        float(model.ksvd_atom_score.weight.detach().norm().cpu())
        if model.use_tokens and args.token_fusion == "ksvd_atom_additive_readout"
        else None
    )
    log(
        f"official valid (single evaluation)={official_valid_auc:.6f}; "
        f"gates={final_gates}; router_norms={final_token_router_weight_norms}; "
        f"jk_gates={final_jk_gates}; sum_mean_gate={final_sum_mean_gate}; "
        f"interaction_gates={final_interaction_gates}; "
        f"refined_gates={final_refined_gates}; "
        f"unrolled_mix={final_unrolled_mix}; "
        f"context_gates={final_context_gates}; "
        f"ring_gates={final_ring_gates}; "
        f"ring_readout_gate={final_ring_readout_gate}; motif_gate={final_motif_gate}; "
        f"ksvd_atom_weights={final_ksvd_atom_weights}; "
        f"ksvd_atom_score_norm={final_ksvd_atom_score_norm}"
    )
    if args.evaluate_test:
        log(f"official test (single frozen evaluation)={official_test_auc:.6f}")

    result = {
        "protocol_id": (
            "molhiv-localized-node-token-frozen-test-v1"
            if args.evaluate_test else "molhiv-localized-node-token-inner-v1"
        ),
        "family": args.family,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "inner_split_seed": args.inner_split_seed,
        "inner_valid_fraction": args.inner_valid_fraction,
        "max_selection_epochs": args.epochs,
        "save_inner_predictions": args.save_inner_predictions,
        "selected_epoch": best_epoch,
        "best_inner_valid_auc": best_inner_auc,
        "official_valid_auc": official_valid_auc,
        "official_valid_evaluations": 1,
        "capture_official_valid_predictions": args.capture_official_valid_predictions,
        "official_test_auc": official_test_auc,
        "official_test_evaluations": int(args.evaluate_test),
        "frozen_config_id": args.frozen_config_id,
        "test_policy": (
            "configuration frozen before test encoding/evaluation; official test evaluated once"
            if args.evaluate_test
            else "official test nodes were not sparse-coded or indexed into model data objects; test was not evaluated"
        ),
        "official_valid_y": official_valid_y,
        "official_valid_predictions": official_valid_predictions,
        "official_test_y": official_test_y,
        "official_test_predictions": official_test_predictions,
        "n_used": len(graphs),
        "n_official_train": int(len(tr)),
        "n_official_valid": int(len(va)),
        "n_official_test": int(len(te)),
        "n_inner_train": int(len(inner_tr)),
        "n_inner_valid": int(len(inner_va)),
        "inner_split_protocol": inner_split_protocol,
        "inner_split_cache": (
            str(inner_split_cache_path) if inner_split_cache_path is not None else None
        ),
        "inner_fold": args.inner_fold if inner_split_cache_path is not None else None,
        "n_official_train_pos": int(y_np[tr].sum()),
        "n_official_valid_pos": int(y_np[va].sum()),
        "n_inner_train_pos": int(y_np[inner_tr].sum()),
        "n_inner_valid_pos": int(y_np[inner_va].sum()),
        "hidden": args.hidden,
        "layers": args.layers,
        "residual_gine": args.residual_gine,
        "jk_readout": args.jk_readout,
        "graph_readout": args.graph_readout,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "token_lr_scale": args.token_lr_scale,
        "token_weight_decay": args.token_weight_decay,
        "token_warmup_epochs": args.token_warmup_epochs,
        "parameter_ema_decay": args.parameter_ema_decay,
        "token_dropout": args.token_dropout,
        "token_gate_scale": args.token_gate_scale,
        "ring_gate_scale": args.ring_gate_scale,
        "context_gate_scale": args.context_gate_scale,
        "token_gate_l2": args.token_gate_l2,
        "ksvd_aux_weight": args.ksvd_aux_weight,
        "ksvd_aux_epochs": args.ksvd_aux_epochs,
        "token_inject_layers": args.token_inject_layers,
        "token_normalization": args.token_normalization,
        "token_zscore_clip": args.token_zscore_clip,
        "token_prevalence_power": args.token_prevalence_power,
        "token_min_graph_frequency": args.token_min_graph_frequency,
        "selection_token_preprocessing": selection_token_preprocessing,
        "final_token_preprocessing": final_token_preprocessing,
        "selection_fixed_probe": selection_fixed_probe,
        "final_fixed_probe": final_fixed_probe,
        "token_channels": args.token_channels,
        "token_fusion": args.token_fusion,
        "graph_residual_hidden": args.graph_residual_hidden,
        "graph_residual_probe_c": args.graph_residual_probe_c,
        "graph_residual_training": args.graph_residual_training,
        "graph_residual_combination": args.graph_residual_combination,
        "graph_residual_independent_loss": args.graph_residual_independent_loss,
        "token_interaction_rank": args.token_interaction_rank,
        "motif_slot_rank": args.motif_slot_rank,
        "motif_geometry_assignment": args.motif_geometry_assignment,
        "motif_geometry_atom_normalization": args.motif_geometry_atom_normalization,
        "motif_atom_id_source": args.motif_atom_id_source,
        "motif_gate_mode": args.motif_gate_mode,
        "motif_slot_backend": args.motif_slot_backend,
        "motif_slot_shuffle_seed": args.motif_slot_shuffle_seed,
        "motif_occurrence_radius": args.motif_occurrence_radius,
        "motif_occurrence_stats": motif_occurrence_stats,
        "motif_ego_radius": args.motif_ego_radius,
        "motif_ego_confidence": args.motif_ego_confidence,
        "motif_ego_min_atom_index": args.motif_ego_min_atom_index,
        "motif_ego_min_dominance": args.motif_ego_min_dominance,
        "token_interaction_support_mask": args.token_interaction_support_mask,
        "unrolled_steps": args.unrolled_steps,
        "unrolled_step_init": args.unrolled_step_init,
        "unrolled_lambda_init": args.unrolled_lambda_init,
        "unrolled_threshold_mode": args.unrolled_threshold_mode,
        "unrolled_recon_weight": args.unrolled_recon_weight,
        "unrolled_sparse_weight": args.unrolled_sparse_weight,
        "base_token_dim": base_token_dim,
        "primary_token_dim": dual_base_token_dim,
        "ring_token_dim": ring_token_dim,
        "token_cache": str(cache_path),
        "token_dim": token_dim,
        "selection_history": history,
        "trainable_parameter_count": model.trainable_parameter_count,
        "base_parameter_count": model.base_parameter_count,
        "token_parameter_count": model.token_parameter_count,
        "final_jk_gates": final_jk_gates,
        "final_sum_mean_gate": final_sum_mean_gate,
        "final_train_losses": final_losses,
        "final_token_gates": final_gates,
        "final_interaction_gates": final_interaction_gates,
        "final_refined_gates": final_refined_gates,
        "final_unrolled_mix": final_unrolled_mix,
        "final_unrolled_step_sizes": final_unrolled_step_sizes,
        "final_unrolled_lambdas": final_unrolled_lambdas,
        "final_context_gates": final_context_gates,
        "final_token_router_weight_norms": final_token_router_weight_norms,
        "final_ring_gates": final_ring_gates,
        "final_ring_readout_gate": final_ring_readout_gate,
        "final_motif_gate": final_motif_gate,
        "final_ksvd_atom_weights": final_ksvd_atom_weights,
        "final_ksvd_atom_score_norm": final_ksvd_atom_score_norm,
        "audit_fingerprints": {
            "original_indices_sha256": array_hash(original_indices),
            "official_train_sha256": array_hash(tr),
            "official_valid_sha256": array_hash(va),
            "inner_train_sha256": array_hash(inner_tr),
            "inner_valid_sha256": array_hash(inner_va),
            "selection_initial_base_state_sha256": selection_initial_base_hash,
            "final_initial_base_state_sha256": final_initial_base_hash,
            "selection_initial_token_proj_state_sha256": selection_initial_token_proj_hash,
            "final_initial_token_proj_state_sha256": final_initial_token_proj_hash,
            "selection_initial_interaction_gates": selection_initial_interaction_gates,
            "selection_initial_refined_gates": selection_initial_refined_gates,
            "selection_initial_unrolled_mix": selection_initial_unrolled_mix,
            "selection_initial_context_gates": selection_initial_context_gates,
            "selection_first_three_batches": selection_first_batches,
            "final_first_three_batches": final_first_batches,
            "selection_loader_seed": args.seed + 9173,
            "final_loader_seed": args.seed + 9173 + 100_000,
        },
        "elapsed_sec": time.time() - t0,
        "env": check_env(),
        "paired_invariants": {
            "fixed_data_subset_across_model_seeds": True,
            "same_base_initialization_for_same_seed_across_families": True,
            "same_minibatch_order_for_same_seed_across_families": True,
            "zero_initial_token_effect": True,
            "train_only_token_normalization": True,
            "fixed_dictionary_and_tokens_across_neural_seeds": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
