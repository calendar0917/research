"""Paired structural audit for MolHIV CIN--Beam8 prediction files.

The script is intentionally read-only with respect to model evaluation: it
loads prediction-bearing official-train internal-fold results, reconstructs
label-free molecule/Beam8 descriptors, and reports paired error and subgroup
metrics.  Official validation and test graphs are never selected or encoded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_beam8_incidence import build_molhiv_beam8_incidence
from code.molhiv_ring_cells import chordless_cycles


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _components(graph: Any) -> int:
    unseen = set(range(graph.n))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            for neighbour in graph.neighbors(node):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    return count


def _descriptor_row(graph: Any, item: Any, *, max_ring_size: int) -> dict[str, float]:
    bonds = tuple(sorted(graph.edges()))
    bond_lookup = {bond: index for index, bond in enumerate(bonds)}
    rings = tuple(chordless_cycles(graph, min_size=3, max_size=max_ring_size))
    ring_bonds: set[tuple[int, int]] = set()
    ring_boundary_incidence = 0
    for ring in rings:
        ring_boundary_incidence += len(ring)
        for offset, left in enumerate(ring):
            right = ring[(offset + 1) % len(ring)]
            ring_bonds.add((left, right) if left < right else (right, left))

    occurrence_counts = np.zeros(len(bonds), dtype=np.float64)
    ring_occurrences = 0
    patch_ring_counts: list[int] = []
    for nodes in item.slot_nodes:
        local_ring = 0
        node_set = set(nodes)
        for bond, bond_index in bond_lookup.items():
            if bond[0] in node_set and bond[1] in node_set:
                occurrence_counts[bond_index] += 1.0
                if bond in ring_bonds:
                    ring_occurrences += 1
                    local_ring += 1
        patch_ring_counts.append(local_ring)

    completion = np.asarray(item.patch_is_completion, dtype=np.bool_)
    patch_sizes = np.asarray([len(nodes) for nodes in item.slot_nodes], dtype=np.float64)
    patch_ring = np.asarray(patch_ring_counts, dtype=np.float64)
    ring_mask = np.asarray([bond in ring_bonds for bond in bonds], dtype=np.bool_)
    ring_mean = float(occurrence_counts[ring_mask].mean()) if ring_mask.any() else 0.0
    nonring_mean = (
        float(occurrence_counts[~ring_mask].mean()) if (~ring_mask).any() else 0.0
    )
    total_occurrences = float(occurrence_counts.sum())
    n_atoms = int(graph.n)
    n_bonds = len(bonds)
    n_patches = item.n_patches
    completion_count = int(completion.sum())
    cycle_rank = n_bonds - n_atoms + _components(graph)
    return {
        "n_atoms": float(n_atoms),
        "n_bonds": float(n_bonds),
        "mean_degree": _safe_div(2 * n_bonds, n_atoms),
        "cycle_rank": float(cycle_rank),
        "cycle_rank_density": _safe_div(cycle_rank, n_atoms),
        "ring_count": float(len(rings)),
        "ring_density": _safe_div(len(rings), n_atoms),
        "ring_boundary_incidence": float(ring_boundary_incidence),
        "ring_bond_fraction": _safe_div(len(ring_bonds), n_bonds),
        "patch_count": float(n_patches),
        "base_patch_count": float(n_patches - completion_count),
        "completion_count": float(completion_count),
        "completion_fraction": _safe_div(completion_count, n_patches),
        "patches_per_atom": _safe_div(n_patches, n_atoms),
        "patches_per_bond": _safe_div(n_patches, n_bonds),
        "mean_patch_size": float(patch_sizes.mean()) if len(patch_sizes) else 0.0,
        "full_patch_fraction": float(np.mean(patch_sizes == 8)) if len(patch_sizes) else 0.0,
        "bond_occurrences_per_bond": _safe_div(total_occurrences, n_bonds),
        "ring_occurrence_fraction": _safe_div(ring_occurrences, total_occurrences),
        "ring_bond_patch_multiplicity": ring_mean,
        "nonring_bond_patch_multiplicity": nonring_mean,
        "ring_vs_nonring_multiplicity": _safe_div(ring_mean, nonring_mean),
        "ring_patch_fraction": float(np.mean(patch_ring > 0)) if len(patch_ring) else 0.0,
        "mean_ring_bonds_per_patch": float(patch_ring.mean()) if len(patch_ring) else 0.0,
        "completion_ring_patch_fraction": (
            float(np.mean(patch_ring[completion] > 0)) if completion.any() else 0.0
        ),
        "base_ring_patch_fraction": (
            float(np.mean(patch_ring[~completion] > 0)) if (~completion).any() else 0.0
        ),
        "overlap_edges_per_patch": _safe_div(item.overlap_index.shape[1] / 2, n_patches),
        "base_overlap_edges_per_patch": _safe_div(
            item.base_overlap_index.shape[1] / 2, n_patches
        ),
        "chain_edges_per_patch": _safe_div(item.chain_index.shape[1] / 2, n_patches),
        "incidences_per_atom": _safe_div(item.incidence_index.shape[1], n_atoms),
    }


def _load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("official_valid_evaluations") != 0:
        raise ValueError(f"official-valid access recorded in {path}")
    if result.get("official_test_evaluations") != 0:
        raise ValueError(f"official-test access recorded in {path}")
    return result


def _predictions(result: Mapping[str, Any], variant: str, split: str) -> dict[int, tuple[float, float]]:
    metrics = result["results"][variant][split]
    indices = metrics.get("graph_indices")
    labels = metrics.get("labels")
    probabilities = metrics.get("probabilities")
    if indices is None or labels is None or probabilities is None:
        raise ValueError(f"{variant}/{split} does not contain saved predictions")
    if not (len(indices) == len(labels) == len(probabilities)):
        raise ValueError("prediction arrays have inconsistent lengths")
    return {
        int(index): (float(label), float(probability))
        for index, label, probability in zip(indices, labels, probabilities)
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int | None]:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(labels, probabilities)) if labels.sum() > 0 else None
        ),
        "log_loss": float(log_loss(labels, clipped, labels=[0.0, 1.0])),
        "brier": float(np.mean((probabilities - labels) ** 2)),
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in ("roc_auc", "average_precision", "log_loss", "brier"):
        output[key] = (
            None
            if left[key] is None or right[key] is None
            else float(left[key] - right[key])
        )
    return output


def _paired_audit(
    descriptors: Mapping[int, Mapping[str, float]],
    left: Mapping[int, tuple[float, float]],
    right: Mapping[int, tuple[float, float]],
) -> dict[str, Any]:
    indices = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("paired models have different graph sets")
    labels = np.asarray([left[index][0] for index in indices], dtype=np.float64)
    right_labels = np.asarray([right[index][0] for index in indices], dtype=np.float64)
    if not np.array_equal(labels, right_labels):
        raise ValueError("paired labels disagree")
    left_p = np.asarray([left[index][1] for index in indices], dtype=np.float64)
    right_p = np.asarray([right[index][1] for index in indices], dtype=np.float64)
    left_metrics = _metrics(labels, left_p)
    right_metrics = _metrics(labels, right_p)
    eps = 1e-7
    left_loss = -(labels * np.log(np.clip(left_p, eps, 1 - eps)) + (1 - labels) * np.log(np.clip(1 - left_p, eps, 1 - eps)))
    right_loss = -(labels * np.log(np.clip(right_p, eps, 1 - eps)) + (1 - labels) * np.log(np.clip(1 - right_p, eps, 1 - eps)))
    loss_gain = right_loss - left_loss

    positive = labels == 1
    negative = ~positive
    left_auc_contribution = np.zeros(len(labels), dtype=np.float64)
    right_auc_contribution = np.zeros(len(labels), dtype=np.float64)
    if positive.any() and negative.any():
        for probabilities, contribution in (
            (left_p, left_auc_contribution),
            (right_p, right_auc_contribution),
        ):
            comparisons = probabilities[positive, None] - probabilities[None, negative]
            pair_scores = (comparisons > 0).astype(np.float64) + 0.5 * (
                comparisons == 0
            )
            contribution[positive] = pair_scores.mean(axis=1)
            contribution[negative] = pair_scores.mean(axis=0)
    auc_contribution_gain = left_auc_contribution - right_auc_contribution

    descriptor_audit: dict[str, Any] = {}
    names = sorted(next(iter(descriptors.values())))
    for name in names:
        values = np.asarray([descriptors[index][name] for index in indices], dtype=np.float64)
        if np.std(values) <= 1e-12:
            correlation = None
        else:
            correlation = float(np.corrcoef(values, loss_gain)[0, 1])
        auc_correlations: dict[str, float | None] = {}
        for group_name, mask in (
            ("all", np.ones(len(labels), dtype=np.bool_)),
            ("positive", positive),
            ("negative", negative),
        ):
            if mask.sum() < 2 or np.std(values[mask]) <= 1e-12:
                auc_correlations[group_name] = None
            else:
                auc_correlations[group_name] = float(
                    np.corrcoef(values[mask], auc_contribution_gain[mask])[0, 1]
                )
        quantiles = np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])
        bins = []
        for bin_index in range(3):
            lower = quantiles[bin_index]
            upper = quantiles[bin_index + 1]
            mask = (values >= lower) & (values <= upper if bin_index == 2 else values < upper)
            if not mask.any():
                continue
            left_bin = _metrics(labels[mask], left_p[mask])
            right_bin = _metrics(labels[mask], right_p[mask])
            bins.append(
                {
                    "range": [float(lower), float(upper)],
                    "left": left_bin,
                    "right": right_bin,
                    "delta_left_minus_right": _metric_delta(left_bin, right_bin),
                    "mean_logloss_gain_left": float(loss_gain[mask].mean()),
                    "mean_auc_contribution_gain_left": float(
                        auc_contribution_gain[mask].mean()
                    ),
                }
            )
        descriptor_audit[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "correlation_with_logloss_gain": correlation,
            "correlation_with_auc_contribution_gain": auc_correlations,
            "tertiles": bins,
        }
    return {
        "left": left_metrics,
        "right": right_metrics,
        "delta_left_minus_right": _metric_delta(left_metrics, right_metrics),
        "fraction_lower_logloss_left": float(np.mean(loss_gain > 0)),
        "mean_logloss_gain_left": float(loss_gain.mean()),
        "mean_auc_contribution_gain_left": {
            "positive": float(auc_contribution_gain[positive].mean()),
            "negative": float(auc_contribution_gain[negative].mean()),
        },
        "descriptor_audit": descriptor_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bond-result", action="append", required=True)
    parser.add_argument("--cin-result", action="append", default=[])
    parser.add_argument("--max-graphs", type=int, default=8000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--coverage-checkpoint", default="edge100")
    parser.add_argument("--max-ring-size", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
    except ImportError as exc:
        raise RuntimeError("OGB is required for MolHIV feature dimensions") from exc

    paths = [Path(value) for value in args.bond_result + args.cin_result]
    loaded = {str(path): _load_result(path) for path in paths}
    all_indices: set[int] = set()
    for result in loaded.values():
        for variant in result["results"]:
            for split in ("fit", "heldout"):
                all_indices.update(_predictions(result, variant, split))

    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    official_train = set(np.asarray(bundle.split["train"], dtype=np.int64).tolist())
    forbidden = set(np.asarray(bundle.split["valid"], dtype=np.int64).tolist()) | set(
        np.asarray(bundle.split["test"], dtype=np.int64).tolist()
    )
    if not all_indices <= official_train or all_indices & forbidden:
        raise AssertionError("prediction files contain non-official-train graphs")
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("MolHIV categorical features are required")

    atom_dims = tuple(int(value) for value in get_atom_feature_dims())
    bond_dims = tuple(int(value) for value in get_bond_feature_dims())
    descriptors: dict[int, dict[str, float]] = {}
    for count, index in enumerate(sorted(all_indices), 1):
        item = build_molhiv_beam8_incidence(
            index,
            bundle.graphs[index],
            float(bundle.y[index]),
            bundle.node_feats[index],
            bundle.edge_feats[index],
            atom_feature_dims=atom_dims,
            bond_feature_dims=bond_dims,
            patch_size=args.patch_size,
            overlap=args.overlap,
            retained_beam=args.retained_beam,
            seed=20260815,
            coverage_checkpoint=args.coverage_checkpoint,
        )
        descriptors[index] = _descriptor_row(
            bundle.graphs[index], item, max_ring_size=args.max_ring_size
        )
        if count == 1 or count % 250 == 0 or count == len(all_indices):
            print(f"prepared descriptors {count}/{len(all_indices)}", flush=True)

    cin_by_key: dict[tuple[int, int], tuple[str, Mapping[str, Any]]] = {}
    for path in args.cin_result:
        result = loaded[str(Path(path))]
        key = (int(result["config"]["fold"]), int(result["config"]["seed"]))
        cin_by_key[key] = (str(path), result)

    audits: dict[str, Any] = {}
    for path in args.bond_result:
        result = loaded[str(Path(path))]
        fold = int(result["config"]["fold"])
        seed = int(result["config"]["seed"])
        key_name = f"fold{fold}_seed{seed}"
        comparisons: dict[str, Any] = {}
        for split in ("fit", "heldout"):
            aligned = _predictions(result, "cin_beam8_bond", split)
            comparisons[f"{split}_vs_shuffled"] = _paired_audit(
                descriptors,
                aligned,
                _predictions(result, "cin_beam8_bond_shuffled", split),
            )
            comparisons[f"{split}_vs_no_patch"] = _paired_audit(
                descriptors,
                aligned,
                _predictions(result, "cin_beam8_bond_no_patch", split),
            )
            if (fold, seed) in cin_by_key:
                _cin_path, cin_result = cin_by_key[(fold, seed)]
                comparisons[f"{split}_vs_cin"] = _paired_audit(
                    descriptors, aligned, _predictions(cin_result, "cin", split)
                )
        audits[key_name] = {
            "bond_result": str(path),
            "cin_result": cin_by_key.get((fold, seed), (None, None))[0],
            "comparisons": comparisons,
        }

    output = {
        "protocol_id": "molhiv-official-train-internal-cin-beam8-subgroup-v1",
        "scope": "paired diagnostic on official-train internal folds only",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "config": vars(args),
        "descriptor_names": sorted(next(iter(descriptors.values()))),
        "audits": audits,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "audits": list(audits)}, indent=2))


if __name__ == "__main__":
    main()
