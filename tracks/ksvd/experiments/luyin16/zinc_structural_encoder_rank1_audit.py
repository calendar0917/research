"""Zero-training rank-1 mechanism audit of the shared structural encoder.

Workstream B of the structural-encoder mechanism decomposition.  Uses the
**frozen** B-full (``patch_representation="shared_structural"``) seed0/seed1
checkpoints and official **train** + **valid** data only.  No gradients, no
retraining, no architecture selection, and the official ZINC test is never
loaded.

What is measured
----------------
B1  spectrum of the 16-D ``e_struct`` over official-train patch occurrences,
    occurrence-weighted and unique-structure (typed-certificate) weighted:
    centre, covariance, eigenvalues, effective rank / participation ratio,
    PC1 explained variance and PC1-PC4 cumulative variance.  Per seed.
B2  what PC1 is: Spearman correlations of the train-fit PC1 score with existing
    deterministic patch descriptors, plus a fixed train->valid ridge probe that
    predicts the PC1 score from those descriptors (valid R^2).  Per seed.
B3  frozen functional diagnostic: replace ``e_struct`` by
    ``mean + <e_struct - mean, v1> * v1`` (rank-1 reconstruction, train-fit
    basis) and evaluate official-valid MAE for the raw selection checkpoint and
    the fixed Top-5 soup of both seeds.
B4  secondary sanity: ``e_struct := train mean`` (no patch-specific modulation).

The rank-1 replacement only changes the structural-encoder output tensor; the
checkpoint, the rest of the model and the evaluation loader are untouched.

Stages: ``collect spectrum functional report``.

Run::

    uv run python -m tracks.ksvd.experiments.luyin16.zinc_structural_encoder_rank1_audit <stage>
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from scipy.stats import spearmanr

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_shared_structural_patch_encoder as sspe,
)

REPO_ROOT = zpp.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"

RESULTS_DIR = TRACK_ROOT / "results/structural_encoder_rank1_audit"
PROTOCOL_VERSION = "structural_encoder_rank1_audit_v1"

SEEDS = (0, 1)
RIDGE_ALPHA = 1.0

_write_json = sspe._write_json
_read_json = sspe._read_json
_git_commit = sspe._git_commit
_effective_rank = sspe._effective_rank
_n_params = sspe._n_params


def _mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - predictions)))


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


def molecule_descriptors(graph: Any) -> dict[str, np.ndarray]:
    """Deterministic (target-free) per-patch descriptors from a PatchGraph."""
    patch = np.asarray(graph.patch, dtype=np.int64)
    n = int(graph.n_patches)
    total_nodes = int(graph.atom.shape[0])
    atom = np.asarray(graph.atom, dtype=np.int64)
    dist = np.asarray(graph.dist, dtype=np.int64)
    root = np.asarray(graph.root, dtype=np.int64)
    edge_patch = np.asarray(graph.edge_patch, dtype=np.int64)
    bond = np.asarray(graph.bond, dtype=np.int64)
    src = np.asarray(graph.src, dtype=np.int64)

    n_nodes = np.bincount(patch, minlength=n).astype(np.float64)
    node_start = np.concatenate([[0], np.cumsum(n_nodes.astype(np.int64))[:-1]])
    glob_src = node_start[edge_patch] + src
    degree = np.bincount(glob_src, minlength=total_nodes).astype(np.float64)

    root_node = np.zeros(n, dtype=np.int64)
    root_index = np.nonzero(root > 0)[0]
    root_node[patch[root_index]] = root_index
    root_degree = degree[root_node]

    deg_sum = np.bincount(patch, weights=degree, minlength=n)
    deg_max = np.zeros(n, dtype=np.float64)
    np.maximum.at(deg_max, patch, degree)
    branching = np.bincount(
        patch, weights=(degree >= 3).astype(np.float64), minlength=n
    )
    n_bonds = np.bincount(edge_patch, minlength=n).astype(np.float64) / 2.0
    cycle = n_bonds - n_nodes + 1.0

    dist_max = np.zeros(n, dtype=np.float64)
    np.maximum.at(dist_max, patch, dist.astype(np.float64))
    boundary = np.bincount(
        patch, weights=(dist == dist_max[patch]).astype(np.float64), minlength=n
    )

    atom_counts = np.zeros((n, zpp.ATOM_CATEGORIES), dtype=np.float64)
    for category in range(zpp.ATOM_CATEGORIES):
        atom_counts[:, category] = np.bincount(
            patch, weights=(atom == category).astype(np.float64), minlength=n
        )
    bond_counts = np.zeros((n, zpp.BOND_CATEGORIES), dtype=np.float64)
    for category in range(zpp.BOND_CATEGORIES):
        bond_counts[:, category] = (
            np.bincount(
                edge_patch,
                weights=(bond == category).astype(np.float64),
                minlength=n,
            )
            / 2.0
        )
    dist_hist = np.zeros((n, 3), dtype=np.float64)
    for level in range(3):
        dist_hist[:, level] = np.bincount(
            patch, weights=(dist == level).astype(np.float64), minlength=n
        )

    return {
        "scalar": np.stack(
            [
                n_nodes,
                n_bonds,
                root_degree,
                deg_sum / np.maximum(n_nodes, 1.0),
                deg_max,
                branching,
                cycle,
                boundary,
            ],
            axis=1,
        ),
        "atom_counts": atom_counts,
        "bond_counts": bond_counts,
        "dist_hist": dist_hist,
    }


SCALAR_NAMES = (
    "n_atoms",
    "n_bonds",
    "root_degree",
    "mean_degree",
    "max_degree",
    "branching_count",
    "cycle_rank",
    "boundary_size",
)


def descriptor_matrix(
    graphs: Sequence[Any], patch_cont: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Concatenate per-patch descriptors; returns (matrix, names)."""
    parts = {"scalar": [], "atom_counts": [], "bond_counts": [], "dist_hist": []}
    for graph in graphs:
        row = molecule_descriptors(graph)
        for key in parts:
            parts[key].append(row[key])
    scalar = np.concatenate(parts["scalar"], axis=0)
    atom_counts = np.concatenate(parts["atom_counts"], axis=0)
    bond_counts = np.concatenate(parts["bond_counts"], axis=0)
    dist_hist = np.concatenate(parts["dist_hist"], axis=0)
    n_nodes = np.maximum(scalar[:, 0:1], 1.0)
    n_bonds = np.maximum(scalar[:, 1:2], 1.0)
    atom_composition = atom_counts / n_nodes
    bond_composition = bond_counts / n_bonds
    dist_composition = dist_hist / n_nodes

    names: list[str] = list(SCALAR_NAMES)
    blocks = [scalar]
    for index in range(atom_counts.shape[1]):
        names.append(f"atom_count_{index}")
    blocks.append(atom_counts)
    for index in range(atom_counts.shape[1]):
        names.append(f"atom_composition_{index}")
    blocks.append(atom_composition)
    for index in range(bond_counts.shape[1]):
        names.append(f"bond_count_{index}")
    blocks.append(bond_counts)
    for index in range(bond_counts.shape[1]):
        names.append(f"bond_composition_{index}")
    blocks.append(bond_composition)
    for index in range(dist_hist.shape[1]):
        names.append(f"dist_count_{index}")
    blocks.append(dist_hist)
    for index in range(dist_hist.shape[1]):
        names.append(f"dist_composition_{index}")
    blocks.append(dist_composition)
    for index in range(patch_cont.shape[1]):
        names.append(f"patch_cont_{index}")
    blocks.append(np.asarray(patch_cont, dtype=np.float64))
    return np.concatenate(blocks, axis=1), names


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _patch_keys(records: Sequence[Any]) -> list[str]:
    keys: list[str] = []
    for record in records:
        for patch in record.patches:
            keys.append(bytes(patch.typed_certificate).hex())
    return keys


def _run_encoder(model: nn.Module, data_list: Sequence[Data]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for batch in sspe._make_struct_loader(list(data_list), 128, False, 0):
        with torch.no_grad():
            blocks.append(model.structural_encoder(batch).cpu().numpy())
    return np.concatenate(blocks, axis=0)


def collect(force: bool = False) -> dict[str, Any]:
    """Collect per-seed train/valid ``e_struct``, descriptors and keys."""
    out = RESULTS_DIR / "collect"
    expected = [
        out / f"e_struct_train_seed{seed}.npy" for seed in SEEDS
    ] + [out / f"e_struct_valid_seed{seed}.npy" for seed in SEEDS]
    if not force and all(path.exists() for path in expected) and (
        out / "metadata.json"
    ).exists():
        meta = _read_json(out / "metadata.json")
        if meta.get("protocol_version") == PROTOCOL_VERSION:
            print("[collect] reuse existing artifacts", flush=True)
            return meta

    started = time.perf_counter()
    train_bundle, valid_bundle, _cache_meta = sspe.extract_records()
    train_data, valid_data, _audit = sspe.build_encoded_records()
    out.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        model = sspe.build_candidate(seed)
        model.load_state_dict(
            torch.load(
                sspe.STATE_DIR / f"sspe_seed{seed}_selection_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        np.save(
            out / f"e_struct_train_seed{seed}.npy",
            _run_encoder(model, train_data).astype(np.float32),
        )
        np.save(
            out / f"e_struct_valid_seed{seed}.npy",
            _run_encoder(model, valid_data).astype(np.float32),
        )

    patch_cont_train = np.concatenate(
        [np.asarray(data.patch_cont, dtype=np.float64) for data in train_data], axis=0
    )
    patch_cont_valid = np.concatenate(
        [np.asarray(data.patch_cont, dtype=np.float64) for data in valid_data], axis=0
    )
    x_train, names = descriptor_matrix(train_bundle["graphs"], patch_cont_train)
    x_valid, names_valid = descriptor_matrix(valid_bundle["graphs"], patch_cont_valid)
    if names != names_valid:
        raise RuntimeError("descriptor name mismatch between train and valid")
    np.save(out / "descriptors_train.npy", x_train.astype(np.float32))
    np.save(out / "descriptors_valid.npy", x_valid.astype(np.float32))
    with (out / "descriptor_names.json").open("w", encoding="utf-8") as handle:
        json.dump(names, handle)
    with (out / "keys_train.pkl").open("wb") as handle:
        pickle.dump(_patch_keys(train_bundle["records"]), handle)
    with (out / "keys_valid.pkl").open("wb") as handle:
        pickle.dump(_patch_keys(valid_bundle["records"]), handle)

    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "n_train_patches": int(x_train.shape[0]),
        "n_valid_patches": int(x_valid.shape[0]),
        "n_descriptors": int(x_train.shape[1]),
        "seconds": float(time.perf_counter() - started),
        "checkpoints": {
            str(seed): str(sspe.STATE_DIR / f"sspe_seed{seed}_selection_state.pt")
            for seed in SEEDS
        },
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(out / "metadata.json", meta)
    print(f"[collect] {meta}", flush=True)
    return meta


def _load_collect(seed: int) -> dict[str, Any]:
    out = RESULTS_DIR / "collect"
    return {
        "e_train": np.load(out / f"e_struct_train_seed{seed}.npy").astype(np.float64),
        "e_valid": np.load(out / f"e_struct_valid_seed{seed}.npy").astype(np.float64),
        "x_train": np.load(out / "descriptors_train.npy").astype(np.float64),
        "x_valid": np.load(out / "descriptors_valid.npy").astype(np.float64),
        "names": json.loads((out / "descriptor_names.json").read_text(encoding="utf-8")),
        "keys_train": pickle.loads((out / "keys_train.pkl").read_bytes()),
    }


# ---------------------------------------------------------------------------
# B1: spectrum
# ---------------------------------------------------------------------------


def _sign_fix(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return direction


def _spectrum(matrix: np.ndarray, label: str) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    centre = matrix.mean(axis=0)
    centered = matrix - centre
    covariance = centered.T @ centered / max(matrix.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    pc1 = _sign_fix(eigenvectors[:, 0])
    return {
        "label": label,
        "n": int(matrix.shape[0]),
        "centre": centre.tolist(),
        "covariance": covariance.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": ratio.tolist(),
        "pc1_direction": pc1.tolist(),
        "pc1_explained_variance": float(ratio[0]),
        "pc4_cumulative_variance": float(ratio[:4].sum()),
        "effective_rank": _effective_rank(matrix),
    }


def _unique_structure_matrix(matrix: np.ndarray, keys: Sequence[str]):
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    unique = np.stack(
        [matrix[indices].mean(axis=0) for indices in groups.values()], axis=0
    )
    return unique, len(groups)


def spectrum() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "unique_structure_key": "PatchRecord.typed_certificate (historical uncolored rooted topology)",
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        data = _load_collect(seed)
        e_train = data["e_train"]
        occurrence = _spectrum(e_train, "occurrence_weighted")
        unique_matrix, n_unique = _unique_structure_matrix(
            e_train, data["keys_train"]
        )
        unique = _spectrum(unique_matrix, "unique_structure_weighted")
        payload["per_seed"][str(seed)] = {
            "occurrence_weighted": occurrence,
            "unique_structure_weighted": unique,
            "n_train_patches": int(e_train.shape[0]),
            "n_unique_structures": int(n_unique),
        }
        np.save(
            RESULTS_DIR / f"pc1_mean_seed{seed}.npy",
            np.asarray(occurrence["centre"]),
        )
        np.save(
            RESULTS_DIR / f"pc1_direction_seed{seed}.npy",
            np.asarray(occurrence["pc1_direction"]),
        )
        np.save(
            RESULTS_DIR / f"pc1_mean_unique_seed{seed}.npy",
            np.asarray(unique["centre"]),
        )
        np.save(
            RESULTS_DIR / f"pc1_direction_unique_seed{seed}.npy",
            np.asarray(unique["pc1_direction"]),
        )
    _write_json(RESULTS_DIR / "spectrum.json", payload)
    return payload


# ---------------------------------------------------------------------------
# B2: PC1 correlations + linear probe
# ---------------------------------------------------------------------------


def _standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1.0e-12] = 1.0
    return (train - mean) / std, (other - mean) / std


def _ridge_r2(train_x, train_y, valid_x, valid_y, alpha=RIDGE_ALPHA):
    xtx = train_x.T @ train_x + alpha * np.eye(train_x.shape[1])
    xty = train_x.T @ train_y
    weights = np.linalg.solve(xtx, xty)
    intercept = float(train_y.mean() - train_x.mean(axis=0) @ weights)
    prediction = valid_x @ weights + intercept
    residual = float(((valid_y - prediction) ** 2).sum())
    total = float(((valid_y - valid_y.mean()) ** 2).sum())
    return {
        "valid_r2": float(1.0 - residual / total) if total > 0 else 0.0,
        "alpha": float(alpha),
        "n_features": int(train_x.shape[1]),
        "train_n": int(train_x.shape[0]),
        "valid_n": int(valid_x.shape[0]),
    }


def correlations() -> dict[str, Any]:
    spectrum_payload = _read_json(RESULTS_DIR / "spectrum.json")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "official_test_loaded": False,
    }
    for seed in SEEDS:
        data = _load_collect(seed)
        entry = spectrum_payload["per_seed"][str(seed)]["occurrence_weighted"]
        centre = np.asarray(entry["centre"])
        direction = np.asarray(entry["pc1_direction"])
        train_score = (data["e_train"] - centre) @ direction
        valid_score = (data["e_valid"] - centre) @ direction

        names = data["names"]
        rows = []
        for index, name in enumerate(names):
            column = data["x_train"][:, index]
            if float(np.std(column)) < 1.0e-12:
                rho = 0.0
            else:
                rho = float(spearmanr(column, train_score).statistic)
                if not np.isfinite(rho):
                    rho = 0.0
            rows.append({"name": name, "spearman_rho": rho, "abs_rho": abs(rho)})
        rows.sort(key=lambda row: row["abs_rho"], reverse=True)

        train_x, valid_x = _standardize(data["x_train"], data["x_valid"])
        probe = _ridge_r2(train_x, train_score, valid_x, valid_score)
        payload["per_seed"][str(seed)] = {
            "n_descriptors": len(names),
            "top_absolute_spearman": rows[:25],
            "spearman_all": rows,
            "linear_probe_pc1_from_descriptors": probe,
        }
    _write_json(RESULTS_DIR / "correlations.json", payload)
    return payload


# ---------------------------------------------------------------------------
# B3/B4: frozen functional diagnostic
# ---------------------------------------------------------------------------


class _StructuralOverride(nn.Module):
    """Replace the structural-encoder output by a train-fit diagnostic."""

    def __init__(
        self,
        base: nn.Module,
        centre: np.ndarray,
        direction: np.ndarray | None,
        mode: str,
    ):
        super().__init__()
        self.base = base
        self.mode = str(mode)
        self.register_buffer(
            "centre", torch.tensor(np.asarray(centre), dtype=torch.float32)
        )
        if direction is None:
            self.register_buffer("direction", torch.zeros_like(self.centre))
        else:
            self.register_buffer(
                "direction",
                torch.tensor(np.asarray(direction), dtype=torch.float32),
            )

    def forward(self, data: Any) -> torch.Tensor:
        full = self.base(data)
        if self.mode == "full":
            return full
        if self.mode == "mean":
            return self.centre.unsqueeze(0).expand_as(full).contiguous()
        if self.mode == "rank1":
            score = (full - self.centre) @ self.direction
            return (
                self.centre.unsqueeze(0)
                + score.unsqueeze(1) * self.direction.unsqueeze(0)
            )
        raise ValueError(f"unknown override mode={self.mode!r}")


def _evaluate_override(model, state, loader, centre, direction, mode) -> float:
    model.load_state_dict(state)
    base = model.structural_encoder
    model.structural_encoder = _StructuralOverride(base, centre, direction, mode)
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            targets.append(batch.y.view(-1).cpu().numpy())
            predictions.append(model(batch).view(-1).cpu().numpy())
    model.structural_encoder = base
    return _mae(
        np.concatenate(targets).astype(np.float64),
        np.concatenate(predictions).astype(np.float64),
    )


def _collect_e_struct(model, loader, centre, direction, mode) -> np.ndarray:
    model.eval()
    blocks: list[np.ndarray] = []
    base = model.structural_encoder
    model.structural_encoder = _StructuralOverride(base, centre, direction, mode)
    with torch.no_grad():
        for batch in loader:
            blocks.append(model.structural_encoder(batch).cpu().numpy())
    model.structural_encoder = base
    return np.concatenate(blocks, axis=0).astype(np.float64)


def functional() -> dict[str, Any]:
    _train, valid_data, _audit = sspe.build_encoded_records()
    loader = sspe._selection_loader(valid_data)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "per_seed": {},
        "official_test_loaded": False,
    }

    for seed in SEEDS:
        centre = np.load(RESULTS_DIR / f"pc1_mean_seed{seed}.npy")
        direction = np.load(RESULTS_DIR / f"pc1_direction_seed{seed}.npy")
        model = sspe.build_candidate(seed)
        model.load_state_dict(
            torch.load(
                sspe.STATE_DIR / f"sspe_seed{seed}_selection_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        e_valid = _collect_e_struct(model, loader, centre, direction, "full")
        score = (e_valid - centre) @ direction
        reconstruction = centre[None, :] + score[:, None] * direction[None, :]
        centered_energy = float((((e_valid - centre) ** 2).sum(axis=1)).mean())
        residual_energy = float((((e_valid - reconstruction) ** 2).sum(axis=1)).mean())
        reconstruction_error = (
            float(residual_energy / centered_energy) if centered_energy > 0 else 0.0
        )

        selection_state = torch.load(
            sspe.STATE_DIR / f"sspe_seed{seed}_selection_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        soup_state = torch.load(
            sspe.SOUP_DIR / f"sspe_seed{seed}_top5_soup.pt",
            map_location="cpu",
            weights_only=True,
        )

        rows: dict[str, Any] = {}
        for label, state in (("selection", selection_state), ("soup", soup_state)):
            results = {
                mode: _evaluate_override(model, state, loader, centre, direction, mode)
                for mode in ("full", "rank1", "mean")
            }
            rows[label] = {
                **results,
                "rank1_minus_full": results["rank1"] - results["full"],
                "mean_minus_full": results["mean"] - results["full"],
            }

        payload["per_seed"][str(seed)] = {
            "train_fit_centre": np.asarray(centre).tolist(),
            "train_fit_pc1_direction": np.asarray(direction).tolist(),
            "valid_full_selection": rows["selection"]["full"],
            "valid_rank1_selection": rows["selection"]["rank1"],
            "valid_mean_selection": rows["selection"]["mean"],
            "valid_full_soup": rows["soup"]["full"],
            "valid_rank1_soup": rows["soup"]["rank1"],
            "valid_mean_soup": rows["soup"]["mean"],
            "delta_rank1_selection": rows["selection"]["rank1_minus_full"],
            "delta_mean_selection": rows["selection"]["mean_minus_full"],
            "delta_rank1_soup": rows["soup"]["rank1_minus_full"],
            "delta_mean_soup": rows["soup"]["mean_minus_full"],
            "relative_reconstruction_error": reconstruction_error,
            "pc1_valid_explained_variance": float(1.0 - reconstruction_error),
            "details": rows,
        }
        del model

    # 2-seed means
    payload["summary"] = {
        "selection_full_mean": float(
            np.mean([payload["per_seed"][str(s)]["valid_full_selection"] for s in SEEDS])
        ),
        "selection_rank1_mean": float(
            np.mean(
                [payload["per_seed"][str(s)]["valid_rank1_selection"] for s in SEEDS]
            )
        ),
        "soup_full_mean": float(
            np.mean([payload["per_seed"][str(s)]["valid_full_soup"] for s in SEEDS])
        ),
        "soup_rank1_mean": float(
            np.mean([payload["per_seed"][str(s)]["valid_rank1_soup"] for s in SEEDS])
        ),
        "soup_mean_mean": float(
            np.mean([payload["per_seed"][str(s)]["valid_mean_soup"] for s in SEEDS])
        ),
    }
    _write_json(RESULTS_DIR / "functional.json", payload)
    return payload


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    def _maybe(name: str) -> Any:
        path = RESULTS_DIR / f"{name}.json"
        return _read_json(path) if path.exists() else None

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "spectrum": _maybe("spectrum"),
        "correlations": _maybe("correlations"),
        "functional": _maybe("functional"),
        "official_test_loaded": False,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "report.json", payload)
    return payload


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("collect", "spectrum", "correlations", "functional", "report", "all"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.stage in {"collect", "all"}:
        collect(force=bool(args.force))
    if args.stage in {"spectrum", "all"}:
        print(spectrum(), flush=True)
    if args.stage in {"correlations", "all"}:
        print(correlations(), flush=True)
    if args.stage in {"functional", "all"}:
        print(functional(), flush=True)
    if args.stage in {"report", "all"}:
        print(report(), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
