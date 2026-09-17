"""FSAR-R2-AR0 mechanism analysis (eval-only) + frozen-M0 residual diagnostic.

This module does **not** change the frozen AR0 model definition, the training
protocol, or any durable verdict.  It explains what the already-trained
node-level assignment residual ``b_s(G) = <W_B^(s), C~>`` actually learned, and
it runs one new very small diagnostic (``frozen_m0``) that re-trains only the
``65 x 28`` node-assignment weight on top of a completely frozen ``M0`` soup.

Stages
------
``h1_seed_agreement``  pairwise seed agreement of branch outputs / coefficients
``h2_rank``            corrected rank / spectrum analysis of ``C`` at several scalings
``h3_pc_explanation``  how much of the trained ``B`` output a train-fit PC basis explains
``h4_contributions``   interpretable structural-group / atom-category contributions
``h5_frozen_m0``       frozen-M0 + linear-``C`` residual experiment (3 seeds)
``h6_witnesses``       illustrative near-collision witnesses
``all``                run every stage and write ``mechanism_summary.json``

Everything reads only the AR0 train/valid cache and the pulled AR0 soup states.
The official ZINC test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as runner
from tracks.ksvd.experiments.luyin16.zinc_fsar import _set_deterministic

REPO_ROOT = runner.REPO_ROOT
RESULTS_DIR = runner.RESULTS_DIR
MECH_DIR = RESULTS_DIR / "mechanism"
RAW_ZINC_DIR = runner.ZINC_ROOT / "raw"

PROTOCOL_VERSION = "fsar_r2_ar0_mechanism_v1"
SEEDS = (0, 1, 2)

STRUCTURAL_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("root_basis", 0, 11),
    ("node_mean", 11, 22),
    ("node_std", 22, 33),
    ("edge_mean", 33, 48),
    ("edge_std", 48, 63),
    ("patch_size", 63, 65),
)


# ---------------------------------------------------------------------------
# schema mapping (verified from the raw ZINC dictionaries)
# ---------------------------------------------------------------------------


class _SchemaDictionary:
    """Stub for the benchmarking-gnns ``Dictionary`` pickle class."""


class _SchemaUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "Dictionary":
            return _SchemaDictionary
        return super().find_class(module, name)


def load_zinc_schema() -> dict[str, Any]:
    """Human-readable ZINC schema, read from the raw dataset dictionaries.

    ``idx2word`` is the reliable category-index -> element/bond-name mapping
    shipped with the dataset; nothing is guessed from memory.
    """
    result: dict[str, Any] = {}
    for key in ("atom_dict", "bond_dict"):
        path = RAW_ZINC_DIR / f"{key}.pickle"
        with path.open("rb") as handle:
            loaded = _SchemaUnpickler(handle).load()
        state = loaded.__dict__
        result[key] = {
            "idx2word": [str(word) for word in state["idx2word"]],
            "word2idx": {str(k): int(v) for k, v in state["word2idx"].items()},
        }
    return result


def atom_category_labels() -> list[str]:
    return load_zinc_schema()["atom_dict"]["idx2word"]


def bond_category_labels() -> list[str]:
    return load_zinc_schema()["bond_dict"]["idx2word"]


# ---------------------------------------------------------------------------
# shared per-molecule features
# ---------------------------------------------------------------------------


@dataclass
class NodeArrays:
    """Per-molecule raw / scaled node assignment statistics on one split."""

    C_raw: np.ndarray  # [N, 65, 28]
    C_tilde: np.ndarray  # [N, 65, 28]
    n_nodes: np.ndarray  # [N]
    n_edges: np.ndarray  # [N]
    y: np.ndarray  # [N]
    S_marg: np.ndarray  # [N, 132]


def _node_arrays(molecules: Sequence[r2.MoleculeFeatures], scalers) -> NodeArrays:
    d_c = np.asarray(scalers["C_rms"], dtype=np.float64)
    mask_c = np.asarray(scalers["C_mask"], dtype=np.float64)
    c_raw: list[np.ndarray] = []
    s_marg: list[np.ndarray] = []
    for molecule in molecules:
        phi = np.asarray(molecule.phi, dtype=np.float64)
        c_matrix, _p = r2.center_stats(phi, r2.one_hot_q(molecule.atom_idx))
        c_raw.append(c_matrix)
        s_marg.append(
            np.concatenate(
                [phi.mean(axis=0), phi.std(axis=0), [math.log1p(molecule.n_nodes), math.log1p(molecule.n_edges)]]
            )
        )
    c_raw_array = np.stack(c_raw, axis=0)
    return NodeArrays(
        C_raw=c_raw_array,
        C_tilde=c_raw_array / d_c[None, :, :] * mask_c[None, :, :],
        n_nodes=np.asarray([m.n_nodes for m in molecules], dtype=np.float64),
        n_edges=np.asarray([m.n_edges for m in molecules], dtype=np.float64),
        y=np.asarray([m.y for m in molecules], dtype=np.float64),
        S_marg=np.stack(s_marg, axis=0),
    )


def _batched_prediction(
    variant: str,
    seed: int,
    molecules: Sequence[r2.MoleculeFeatures],
    scalers,
    base_only: bool = False,
    state: str = "soup",
) -> dict[str, np.ndarray]:
    """Run one trained AR0 variant over a split and return base / branch / final."""
    model = r2.build_model(variant, scalers)
    tag = f"r2ar0_{variant.lower()}"
    if state == "soup":
        path = runner.SOUP_DIR / f"{tag}_seed{seed}_top5_soup.pt"
    else:
        path = runner.STATE_DIR / f"{tag}_seed{seed}_selection_state.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    loader = runner._loader(molecules, 128, False, 0)
    base_values: list[np.ndarray] = []
    branch_values: list[np.ndarray] = []
    final_values: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            base = model.base_prediction(batch).view(-1).numpy()
            base_values.append(base)
            targets.append(batch.y.view(-1).numpy())
            if not base_only and model.variant in r2.BINDING_MODELS:
                c_matrix, p_matrix = model.compute_statistics(batch)
                branch_values.append(model.assignment_term(c_matrix, p_matrix).view(-1).numpy())
                final_values.append(model(batch).view(-1).numpy())
    base_all = np.concatenate(base_values)
    out = {"base": base_all, "target": np.concatenate(targets)}
    if branch_values:
        out["branch"] = np.concatenate(branch_values)
        out["final"] = np.concatenate(final_values)
    return out


# ---------------------------------------------------------------------------
# spectrum helpers
# ---------------------------------------------------------------------------


def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
    """Effective rank = exp(entropy(singular values / sum)), rows centred."""
    if matrix.size == 0:
        return {"effective_rank": 0.0, "top_singular_fraction": 0.0}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float(singular.sum())
    if total <= 0.0:
        return {"effective_rank": 0.0, "top_singular_fraction": 0.0}
    probabilities = singular / total
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    return {
        "effective_rank": float(math.exp(entropy)),
        "top_singular_fraction": float(singular[0] / total),
    }


def _rank_block(matrix: np.ndarray, k_top: int = 10) -> dict[str, Any]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / energy.sum()
    block = {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "singular_top10": [float(value) for value in singular[:k_top]],
        "pc1_explained_variance": float(cumulative[0]),
        "pc1_5_cumulative_variance": float(cumulative[min(4, len(cumulative) - 1)]),
        **_effective_rank(matrix),
    }
    return block


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= 0.0 or b.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if a.std() <= 0.0 or b.std() <= 0.0:
        return 0.0
    return float(spearmanr(a, b).statistic)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _linear_r2(features: np.ndarray, target: np.ndarray) -> float:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    ss_res = float(((target - prediction) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _write(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    MECH_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return dict(payload)


# ---------------------------------------------------------------------------
# H1: do the three seeds learn the same assignment score?
# ---------------------------------------------------------------------------


def h1_seed_agreement() -> dict[str, Any]:
    _train, valid, scalers, _meta = runner.build_datasets()
    arrays = _node_arrays(valid, scalers)
    weights_scaled: dict[int, np.ndarray] = {}
    weights_raw: dict[int, np.ndarray] = {}
    branches: dict[int, np.ndarray] = {}
    finals: dict[int, np.ndarray] = {}
    bases: dict[int, np.ndarray] = {}
    targets: np.ndarray | None = None
    for seed in SEEDS:
        model = r2.build_model("MB", scalers)
        path = runner.SOUP_DIR / f"r2ar0_mb_seed{seed}_top5_soup.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        weight = model.W_B.detach().numpy().astype(np.float64)
        weights_scaled[seed] = weight
        weights_raw[seed] = weight / np.asarray(scalers["C_rms"], dtype=np.float64)
        prediction = _batched_prediction("MB", seed, valid, scalers)
        branches[seed] = prediction["branch"]
        finals[seed] = prediction["final"]
        bases[seed] = prediction["base"]
        targets = prediction["target"]

    mask_flat = np.asarray(scalers["C_mask"], dtype=np.float64).reshape(-1) > 0.0
    pairs = [(0, 1), (0, 2), (1, 2)]
    table: list[dict[str, Any]] = []
    for a, b in pairs:
        wa = weights_scaled[a].reshape(-1)[mask_flat]
        wb = weights_scaled[b].reshape(-1)[mask_flat]
        ra = weights_raw[a].reshape(-1)[mask_flat]
        rb = weights_raw[b].reshape(-1)[mask_flat]
        table.append(
            {
                "pair": f"{a}-{b}",
                "branch_pearson": _pearson(branches[a], branches[b]),
                "branch_spearman": _spearman(branches[a], branches[b]),
                "final_pearson": _pearson(finals[a], finals[b]),
                "W_scaled_cosine": _cosine(wa, wb),
                "W_raw_unit_cosine": _cosine(ra, rb),
                "W_scaled_norm": [float(np.linalg.norm(weights_scaled[a])), float(np.linalg.norm(weights_scaled[b]))],
                "W_raw_unit_norm": [float(np.linalg.norm(ra)), float(np.linalg.norm(rb))],
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "n_valid_molecules": int(arrays.C_raw.shape[0]),
        "definition": {
            "branch": "b_s(G) = <W_B^(s), C~(G)>",
            "W_scaled_cosine": "cosine over W_B restricted to non-zero-RMS coordinates",
            "W_raw_unit_cosine": "cosine over beta_raw = W_B / D_C restricted to the same coordinates",
        },
        "branch_output_std": {str(seed): float(branches[seed].std()) for seed in SEEDS},
        "pairwise": table,
        "official_test_loaded": False,
    }
    _write(MECH_DIR / "h1_seed_agreement.json", payload)
    print(json.dumps(payload["pairwise"], indent=2, sort_keys=True), flush=True)
    return payload


# ---------------------------------------------------------------------------
# H2: corrected rank / size diagnostics
# ---------------------------------------------------------------------------


def _scalings(arrays: NodeArrays) -> dict[str, np.ndarray]:
    n = arrays.n_nodes[:, None, None]
    m = arrays.n_edges[:, None, None]
    return {
        "raw_C": arrays.C_raw,
        "model_input_Ctilde": arrays.C_tilde,
        "C_over_n": arrays.C_raw / n,
        "C_over_sqrt_n": arrays.C_raw / np.sqrt(n),
        "C_over_m": arrays.C_raw / m,
        "C_over_sqrt_m": arrays.C_raw / np.sqrt(m),
    }


def h2_rank() -> dict[str, Any]:
    train, valid, scalers, _meta = runner.build_datasets()
    train_arrays = _node_arrays(train, scalers)
    valid_arrays = _node_arrays(valid, scalers)
    mb0 = _batched_prediction("MB", 0, valid, scalers)
    m0_0 = _batched_prediction("M0", 0, valid, scalers, base_only=True)

    blocks: dict[str, Any] = {}
    for name, train_matrix in _scalings(train_arrays).items():
        flat_train = train_matrix.reshape(train_matrix.shape[0], -1)
        flat_valid = _scalings(valid_arrays)[name].reshape(valid_arrays.C_raw.shape[0], -1)
        block = _rank_block(flat_train)
        # train-fit, validation-projected PC1 score
        mean = flat_train.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(flat_train - mean, full_matrices=False)
        pc1 = vt[0]
        z1 = (flat_valid - mean) @ pc1
        m0_residual = m0_0["target"] - m0_0["base"]
        block["pc1_score_vs_n_nodes_pearson"] = _pearson(z1, valid_arrays.n_nodes)
        block["pc1_score_vs_n_nodes_spearman"] = _spearman(z1, valid_arrays.n_nodes)
        block["pc1_score_vs_n_edges_pearson"] = _pearson(z1, valid_arrays.n_edges)
        block["pc1_score_vs_n_edges_spearman"] = _spearman(z1, valid_arrays.n_edges)
        block["pc1_score_vs_target_pearson"] = _pearson(z1, valid_arrays.y)
        block["pc1_score_vs_target_spearman"] = _spearman(z1, valid_arrays.y)
        block["pc1_score_vs_M0_residual_pearson"] = _pearson(z1, m0_residual)
        block["pc1_score_vs_M0_residual_spearman"] = _spearman(z1, m0_residual)
        block["pc1_score_vs_MB0_branch_pearson"] = _pearson(z1, mb0["branch"])
        block["pc1_score_vs_MB0_branch_spearman"] = _spearman(z1, mb0["branch"])
        blocks[name] = block

    # Reproduce the historical number exactly, to document the slicing bug.
    legacy_matrix = train_arrays.C_raw.reshape(train_arrays.C_raw.shape[0], -1)[:, :64]
    legacy = _effective_rank(legacy_matrix)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": {
            "effective_rank": "exp(entropy(singular_values / sum)) on the row-centred flattened matrix",
            "top_singular_fraction": "s_0 / sum(s)",
            "fit": "train split only; PC1 score projected onto validation",
            "pi": "flattened C is the row-major [65, 28] matrix, width 1820",
        },
        "train_blocks": blocks,
        "legacy_audit_slice_reproduced": {
            "slice": "flattened C[:, :64] (first 64 of 1820 coordinates) as used by runner.audit",
            **legacy,
        },
        "corrected_full_C": _rank_block(train_arrays.C_raw.reshape(train_arrays.C_raw.shape[0], -1)),
        "official_test_loaded": False,
    }
    _write(MECH_DIR / "h2_rank.json", payload)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("pc1_explained_variance", "effective_rank", "top_singular_fraction")} for k, v in blocks.items()}, indent=2, sort_keys=True), flush=True)
    print("legacy slice reproduced:", json.dumps(legacy), flush=True)
    return payload


# ---------------------------------------------------------------------------
# H3: how much of the trained B does the train-fit PC basis explain?
# ---------------------------------------------------------------------------


def h3_pc_explanation() -> dict[str, Any]:
    train, valid, scalers, _meta = runner.build_datasets()
    train_arrays = _node_arrays(train, scalers)
    valid_arrays = _node_arrays(valid, scalers)
    flat_train = train_arrays.C_tilde.reshape(train_arrays.C_tilde.shape[0], -1)
    flat_valid = valid_arrays.C_tilde.reshape(valid_arrays.C_tilde.shape[0], -1)
    mean = flat_train.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(flat_train - mean, full_matrices=False)
    z_train = (flat_train - mean) @ vt[:10].T
    z_valid = (flat_valid - mean) @ vt[:10].T

    ks = (1, 2, 5, 10)
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        train_prediction = _batched_prediction("MB", seed, train, scalers)
        valid_prediction = _batched_prediction("MB", seed, valid, scalers)
        b_train = train_prediction["branch"]
        b_valid = valid_prediction["branch"]
        row = {
            "pearson_z1_valid": _pearson(z_valid[:, 0], b_valid),
            "spearman_z1_valid": _spearman(z_valid[:, 0], b_valid),
            "cumulative_r2_valid_insample": {
                str(k): _linear_r2(z_valid[:, :k], b_valid) for k in ks
            },
            "cumulative_r2_valid_outofsample": {
                str(k): _linear_r2_train_to_valid(z_train[:, :k], b_train, z_valid[:, :k], b_valid)
                for k in ks
            },
            "branch_out_std_train": float(b_train.std()),
            "branch_out_std_valid": float(b_valid.std()),
        }
        per_seed[str(seed)] = row
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": {
            "basis": "PCA/SVD fitted on the flattened train C~ (model input), k = 1/2/5/10",
            "r2": "linear least-squares R^2 of b_s on z_1..z_k",
            "insample": "fit and evaluate on validation (descriptive)",
            "outofsample": "fit on train, evaluate on validation",
        },
        "pc_variance_fraction_train": [
            float(value) for value in _pc_variance(flat_train, 10)
        ],
        "per_seed": per_seed,
        "official_test_loaded": False,
    }
    _write(MECH_DIR / "h3_pc_explanation.json", payload)
    print(json.dumps(per_seed, indent=2, sort_keys=True), flush=True)
    return payload


def _pc_variance(flat: np.ndarray, k: int) -> np.ndarray:
    mean = flat.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(flat - mean, compute_uv=False)
    energy = singular**2
    return energy[:k] / energy.sum()


def _linear_r2_train_to_valid(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray
) -> float:
    design = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    prediction = np.concatenate([x_valid, np.ones((x_valid.shape[0], 1))], axis=1) @ coefficients
    ss_res = float(((y_valid - prediction) ** 2).sum())
    ss_tot = float(((y_valid - y_valid.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


# ---------------------------------------------------------------------------
# H4: interpretable contributions
# ---------------------------------------------------------------------------


def h4_contributions() -> dict[str, Any]:
    _train, valid, scalers, _meta = runner.build_datasets()
    arrays = _node_arrays(valid, scalers)
    labels = atom_category_labels()
    per_seed: dict[str, Any] = {}
    contrib_rms_all: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        model = r2.build_model("MB", scalers)
        path = runner.SOUP_DIR / f"r2ar0_mb_seed{seed}_top5_soup.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        weight = model.W_B.detach().numpy().astype(np.float64)
        d_c = np.asarray(scalers["C_rms"], dtype=np.float64)
        beta_raw = weight / d_c
        contribution = weight[None, :, :] * arrays.C_tilde  # [N, 65, 28]
        contrib_rms = np.sqrt((contribution**2).mean(axis=0))  # [65, 28]
        contrib_mean_abs = np.abs(contribution).mean(axis=0)
        contrib_rms_all[seed] = contrib_rms
        branch_prediction = _batched_prediction("MB", seed, valid, scalers)["branch"]
        contribution_sum = contribution.sum(axis=(1, 2))
        max_contribution_error = float(np.max(np.abs(contribution_sum - branch_prediction)))

        total_rms = float(contrib_rms.sum())
        group_share = {
            name: float(contrib_rms[start:end, :].sum() / total_rms)
            for name, start, end in STRUCTURAL_GROUPS
        }
        category_share = {
            labels[j] if j < len(labels) else f"cat{j}": float(contrib_rms[:, j].sum() / total_rms)
            for j in range(contrib_rms.shape[1])
        }
        top_coordinates = []
        flat_rms = contrib_rms.reshape(-1)
        order = np.argsort(flat_rms)[::-1][:20]
        for flat_index in order:
            row, column = divmod(int(flat_index), contrib_rms.shape[1])
            top_coordinates.append(
                {
                    "row": row,
                    "column": column,
                    "atom_category": labels[column] if column < len(labels) else f"cat{column}",
                    "structural_group": _group_of_row(row),
                    "W_scaled": float(weight[row, column]),
                    "beta_raw": float(beta_raw[row, column]),
                    "C_rms": float(d_c[row, column]),
                    "contribution_rms": float(contrib_rms[row, column]),
                    "contribution_mean_abs": float(contrib_mean_abs[row, column]),
                }
            )
        top_by_raw_weight = []
        for flat_index in np.argsort(np.abs(beta_raw).reshape(-1))[::-1][:20]:
            row, column = divmod(int(flat_index), contrib_rms.shape[1])
            top_by_raw_weight.append(
                {
                    "row": row,
                    "column": column,
                    "atom_category": labels[column] if column < len(labels) else f"cat{column}",
                    "structural_group": _group_of_row(row),
                    "beta_raw": float(beta_raw[row, column]),
                    "contribution_rms": float(contrib_rms[row, column]),
                }
            )
        per_seed[str(seed)] = {
            "W_scaled_norm": float(np.linalg.norm(weight)),
            "beta_raw_norm": float(np.linalg.norm(beta_raw)),
            "structural_group_contribution_share": group_share,
            "atom_category_contribution_share": category_share,
            "top_coordinates_by_contribution": top_coordinates,
            "top_coordinates_by_raw_weight": top_by_raw_weight,
            "max_abs_contribution_minus_recorded_branch": max_contribution_error,
        }

    cross_seed = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        cross_seed[f"{a}-{b}"] = {
            "contribution_rms_cosine": _cosine(contrib_rms_all[a].reshape(-1), contrib_rms_all[b].reshape(-1)),
            "contribution_rms_spearman": _spearman(contrib_rms_all[a].reshape(-1), contrib_rms_all[b].reshape(-1)),
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": {
            "contribution": "contrib_kj(G) = beta_raw_kj * C_raw_kj(G) = W_B_kj * C~_kj(G)",
            "structural_groups": {name: [start, end] for name, start, end in STRUCTURAL_GROUPS},
            "atom_categories": labels,
            "note": "a large raw-unit weight on a near-zero-RMS coordinate is not necessarily a large contribution",
        },
        "per_seed": per_seed,
        "cross_seed": cross_seed,
        "official_test_loaded": False,
    }
    _write(MECH_DIR / "h4_contributions.json", payload)
    print(json.dumps({k: v["structural_group_contribution_share"] for k, v in per_seed.items()}, indent=2, sort_keys=True), flush=True)
    return payload


def _group_of_row(row: int) -> str:
    for name, start, end in STRUCTURAL_GROUPS:
        if start <= row < end:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# H5: frozen-M0 + linear-C residual
# ---------------------------------------------------------------------------


def _train_frozen_linear(
    c_train: np.ndarray,
    base_train: np.ndarray,
    y_train: np.ndarray,
    c_valid: np.ndarray,
    base_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    protocol: Mapping[str, Any],
    *,
    phi_dim: int = r2.PHI_DIM,
    atom_categories: int = r2.ATOM_CATEGORIES,
) -> dict[str, Any]:
    """Train only a bias-free zero-init [phi_dim, atom_categories] weight on a frozen base."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    weight = torch.zeros(int(phi_dim), int(atom_categories), dtype=torch.float64)
    weight.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [weight],
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    c_train_t = torch.as_tensor(c_train, dtype=torch.float64)
    residual_train = torch.as_tensor(y_train - base_train, dtype=torch.float64)
    c_valid_t = torch.as_tensor(c_valid, dtype=torch.float64)
    base_valid_t = torch.as_tensor(base_valid, dtype=torch.float64)
    y_valid_t = torch.as_tensor(y_valid, dtype=torch.float64)
    n = int(c_train_t.shape[0])
    batch_size = int(protocol["batch_size"])
    generator = torch.Generator().manual_seed(int(seed) + int(protocol["train_shuffle_seed_offset"]))
    max_epochs = int(protocol["max_epochs"])
    patience = int(protocol["patience"])
    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top5: list[tuple[float, int, torch.Tensor]] = []
    curve: list[dict[str, float]] = []

    def _valid_mae(current: torch.Tensor) -> float:
        prediction = base_valid_t + (c_valid_t * current).sum(dim=(1, 2))
        return float((prediction - y_valid_t).abs().mean())

    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            prediction = (c_train_t[index] * weight).sum(dim=(1, 2))
            loss = F.l1_loss(prediction, residual_train[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight], float(protocol["gradient_clip_norm"]))
            optimizer.step()
        valid_mae = _valid_mae(weight)
        curve.append({"epoch": float(epoch), "valid_mae": float(valid_mae)})
        state = weight.detach().clone()
        top5.append((float(valid_mae), int(epoch), state))
        top5.sort(key=lambda item: (item[0], item[1]))
        top5 = top5[:5]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = state
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    soup_state = torch.stack([item[2] for item in top5], dim=0).mean(dim=0)
    return {
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curve)),
        "top5_epochs": [int(item[1]) for item in top5],
        "top5_soup_valid_mae": _valid_mae(soup_state),
        "W_norm": float(best_state.norm()),
        "W_soup_norm": float(soup_state.norm()),
        "W_soup": soup_state,
        "curve": curve,
    }


def h5_frozen_m0() -> dict[str, Any]:
    train, valid, scalers, _meta = runner.build_datasets()
    train_arrays = _node_arrays(train, scalers)
    valid_arrays = _node_arrays(valid, scalers)
    c_train = train_arrays.C_tilde
    c_valid = valid_arrays.C_tilde
    y_train = train_arrays.y
    y_valid = valid_arrays.y
    protocol = dict(runner.shead.OPTIMIZED_PROTOCOL)
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        m0_train = _batched_prediction("M0", seed, train, scalers, base_only=True)
        m0_valid = _batched_prediction("M0", seed, valid, scalers, base_only=True)
        base_train = m0_train["base"]
        base_valid = m0_valid["base"]
        m0_valid_mae = float(np.mean(np.abs(y_valid - base_valid)))
        mb_json = RESULTS_DIR / f"soup_r2ar0_mb_seed{seed}.json"
        mb_soup = float(runner._read_json(mb_json)["top5_soup_valid_mae"]) if mb_json.exists() else float("nan")
        started = time.perf_counter()
        result = _train_frozen_linear(
            c_train, base_train, y_train, c_valid, base_valid, y_valid, seed, protocol
        )
        wall = time.perf_counter() - started
        frozen_soup_mae = float(result["top5_soup_valid_mae"])
        gain_frozen = m0_valid_mae - frozen_soup_mae
        full_gain = m0_valid_mae - mb_soup
        per_seed[str(seed)] = {
            "M0_valid_mae": m0_valid_mae,
            "frozen_M0_plus_B_best_valid_mae": float(result["best_valid_mae"]),
            "frozen_M0_plus_B_soup_valid_mae": frozen_soup_mae,
            "original_MB_soup_valid_mae": mb_soup,
            "gain_frozen_soup": gain_frozen,
            "gain_MB": full_gain,
            "recovery_fraction": float(gain_frozen / full_gain) if full_gain > 0.0 else float("nan"),
            "best_epoch": int(result["best_epoch"]),
            "epochs_run": int(result["epochs_run"]),
            "top5_epochs": result["top5_epochs"],
            "W_norm": float(result["W_norm"]),
            "W_soup_norm": float(result["W_soup_norm"]),
            "wall_clock_s": float(wall),
        }
        # Persist the trained frozen weight for later contribution comparison.
        MECH_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"W_soup": result["W_soup"], "seed": int(seed)},
            MECH_DIR / f"h5_frozen_m0_seed{seed}_W.pt",
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": {
            "model": "yhat = M0_soup(x) + <W_frozen, C~>, W_frozen zero-init, no bias, no MLP",
            "trainable_params": int(r2.PHI_DIM * r2.ATOM_CATEGORIES),
            "protocol": "identical for every seed: Adam lr 1e-3, wd 1e-5, batch 128, L1, clip 5, max 240, patience 40, best official-valid checkpoint, fixed Top-5 soup",
            "base_features": "M0 soup frozen; train/valid base predictions precomputed once (base has no dropout / BN, so this is exact)",
        },
        "per_seed": per_seed,
        "official_test_loaded": False,
    }
    _write(MECH_DIR / "h5_frozen_m0.json", payload)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "top5_epochs"} for k, v in per_seed.items()}, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# H6: illustrative witnesses
# ---------------------------------------------------------------------------


def _count_keys(molecules: Sequence[r2.MoleculeFeatures]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Exact (atom-count, bond-count) keys reconstructed from the cached A."""
    keys: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for molecule in molecules:
        a = np.asarray(molecule.A, dtype=np.float64)
        atom_log = a[: r2.ATOM_CATEGORIES]
        bond_log = a[2 * r2.ATOM_CATEGORIES : 2 * r2.ATOM_CATEGORIES + r2.BOND_CATEGORIES]
        atom_counts = tuple(int(round(value)) for value in np.expm1(atom_log))
        bond_counts = tuple(int(round(value)) for value in np.expm1(bond_log))
        keys.append((atom_counts, bond_counts))
    return keys


def h6_witnesses(max_witnesses: int = 8) -> dict[str, Any]:
    _train, valid, scalers, _meta = runner.build_datasets()
    arrays = _node_arrays(valid, scalers)
    keys = _count_keys(valid)
    mb0 = _batched_prediction("MB", 0, valid, scalers)
    m0_0 = _batched_prediction("M0", 0, valid, scalers, base_only=True)
    branch = mb0["branch"]
    residual = m0_0["target"] - m0_0["base"]

    # exact A-key (atom counts + bond counts) collision groups
    groups: dict[tuple, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    candidates: list[dict[str, Any]] = []
    s_marg = arrays.S_marg
    s_scale = float(s_marg.std(axis=0).mean()) + 1e-12
    for key, members in groups.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                left, right = members[i], members[j]
                distance = float(np.linalg.norm(s_marg[left] - s_marg[right]) / s_scale)
                delta_branch = float(abs(branch[left] - branch[right]))
                candidates.append(
                    {
                        "left": int(left),
                        "right": int(right),
                        "A_key": {"atom_counts": list(key[0]), "bond_counts": list(key[1])},
                        "structural_marginal_distance": distance,
                        "abs_branch_difference": delta_branch,
                    }
                )
    candidates.sort(key=lambda row: (-row["abs_branch_difference"], row["structural_marginal_distance"]))
    witnesses: list[dict[str, Any]] = []
    for row in candidates[: max(5 * int(max_witnesses), 40)]:
        left, right = row["left"], row["right"]
        residual_delta = float(residual[left] - residual[right])
        branch_delta = float(branch[left] - branch[right])
        witnesses.append(
            {
                **row,
                "target": [float(mb0["target"][left]), float(mb0["target"][right])],
                "M0_residual": [float(residual[left]), float(residual[right])],
                "M0_residual_difference": residual_delta,
                "MB_branch": [float(branch[left]), float(branch[right])],
                "MB_branch_difference": branch_delta,
                "n_nodes": [int(arrays.n_nodes[left]), int(arrays.n_nodes[right])],
                "n_edges": [int(arrays.n_edges[left]), int(arrays.n_edges[right])],
                "sign_consistency": bool(np.sign(residual_delta) == np.sign(branch_delta)),
            }
        )
    witnesses = witnesses[: int(max_witnesses)]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "definition": {
            "A_key": "exact atom-count vector + exact bond-count vector (hence identical A)",
            "structural_marginal": "[mean_v phi_v(65), std_v phi_v(65), log1p n, log1p m] (132-D), scaled distance",
            "ranking": "largest |b(G1)-b(G2)| within exact-A-key collisions, then smallest structural-marginal distance; target/residual inspected only afterwards",
            "status": "illustrative witnesses, not a statistical proof",
        },
        "n_exact_A_key_collision_pairs": int(len(candidates)),
        "n_valid_molecules": int(len(valid)),
        "witnesses": witnesses,
        "all_witness_sign_consistency": bool(all(row["sign_consistency"] for row in witnesses)) if witnesses else False,
        "official_test_loaded": False,
    }
    _write(MECH_DIR / "h6_witnesses.json", payload)
    print(json.dumps(payload["witnesses"], indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def all_stages() -> dict[str, Any]:
    started = time.perf_counter()
    h1_seed_agreement()
    h2_rank()
    h3_pc_explanation()
    h4_contributions()
    h5_frozen_m0()
    h6_witnesses()
    summary = assemble_summary()
    summary["seconds"] = float(time.perf_counter() - started)
    _write(MECH_DIR / "mechanism_summary.json", summary)
    return summary


def assemble_summary() -> dict[str, Any]:
    """Assemble the durable mechanism summary from the per-stage JSON files."""
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": runner._git_commit(),
        "official_test_loaded": False,
    }
    for name in (
        "h1_seed_agreement",
        "h2_rank",
        "h3_pc_explanation",
        "h4_contributions",
        "h5_frozen_m0",
        "h6_witnesses",
    ):
        path = MECH_DIR / f"{name}.json"
        if path.exists():
            summary[name] = json.loads(path.read_text(encoding="utf-8"))
    _write(MECH_DIR / "mechanism_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "h1_seed_agreement",
            "h2_rank",
            "h3_pc_explanation",
            "h4_contributions",
            "h5_frozen_m0",
            "h6_witnesses",
            "summary",
            "all",
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.deterministic:
        _set_deterministic(True)
    if arguments.stage == "h1_seed_agreement":
        h1_seed_agreement()
    elif arguments.stage == "h2_rank":
        h2_rank()
    elif arguments.stage == "h3_pc_explanation":
        h3_pc_explanation()
    elif arguments.stage == "h4_contributions":
        h4_contributions()
    elif arguments.stage == "h5_frozen_m0":
        h5_frozen_m0()
    elif arguments.stage == "h6_witnesses":
        h6_witnesses()
    elif arguments.stage == "summary":
        assemble_summary()
    else:
        all_stages()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
