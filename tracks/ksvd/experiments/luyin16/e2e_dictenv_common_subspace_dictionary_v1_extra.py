"""Post-hoc descriptive extras for round ``e2e_dictenv_common_subspace_dictionary_v1``.

Nothing here is trained and nothing here is part of the frozen gate: the script
only re-reads artifacts produced by the frozen runner and adds three descriptive
views.

1. ``graph_reuse.csv`` / ``argmax_atom_hist.csv`` — per-graph activation-mass
   statistics (effective number of atoms, atoms carrying >=5% mass, argmax atom
   histogram) for the RAW seed-0 dictionary and the trained CSSD soup.
2. ``dictionary_geometry.csv`` — how much of every RAW atom pointed along the
   train-mean direction ``u1`` and how the trained CSSD atom relates to the RAW
   atom and to its residual projection.
3. ``extra_summary.json`` — the above rolled up, plus the common-coordinate
   correlations with the named structural descriptors.

Usage::

    CUDA_VISIBLE_DEVICES="" uv run python -m \\
        tracks.ksvd.experiments.luyin16.e2e_dictenv_common_subspace_dictionary_v1_extra
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_common_subspace_dictionary_v1 as runner
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_dictionary_coder_audit_v1 as dcarun
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

RESULTS = runner.RESULTS_DIR
STRUCTURE = RESULTS / "reusable_structure"
SPLITS = ("train", "valid")


def _graph_mass_stats(alpha: np.ndarray, ids: np.ndarray, n_graphs: int) -> dict[str, np.ndarray]:
    """Per-graph activation-mass statistics (one entry per graph).

    Mass uses ``|alpha|``: IHT coefficients are signed, so a plain sum over a
    graph can cancel to ~0 and is not a usable activation budget.
    """
    n_atoms = alpha.shape[1]
    mass = np.zeros((n_graphs, n_atoms), dtype=np.float64)
    np.add.at(mass, ids, np.abs(alpha))
    totals = mass.sum(axis=1)
    safe = np.maximum(totals, 1e-12)
    p = mass / safe[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    entropy = -(p * logp).sum(axis=1)
    effective = np.exp(entropy)
    big = (p >= 0.05).sum(axis=1)
    argmax = p.argmax(axis=1)
    return {
        "effective_atoms": effective,
        "atoms_ge_5pct": big.astype(np.float64),
        "max_share": p.max(axis=1),
        "total_mass": totals,
        "argmax": argmax,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> dict[str, Any]:
    runner._ensure_dirs()
    cssd.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(runner.THREADS)
    final = runner._read_json(RESULTS / "training/final.json")
    if not final.get("completed"):
        raise RuntimeError("training did not complete; extras are skipped")
    selected = final["selected_common_dim"]
    subspace = runner._load_subspace(
        runner._read_json(RESULTS / "common_subspace.json"), selected
    )
    U = subspace.components
    u1 = U[:, 0] / np.linalg.norm(U[:, 0])

    # --- trained CSSD soup ---------------------------------------------------
    state_path = RESULTS / f"training/checkpoints/CSSD-{selected.upper()}-seed0_soup_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    dictionary_init, _sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    model = cssd.build_cssd_model(dictionary_init, runner.CSSD_SEED, subspace)
    model.load_state_dict({key: value.float() for key, value in state.items()})
    D_cssd = dca.effective_dictionary(model.residual_dictionary().detach().double().numpy())
    Draw_entry = dcarun.load_dictionary_entry(("C6", 0))
    Draw = dca.effective_dictionary(Draw_entry["dictionary"])
    D_raw_perp = dca.effective_dictionary(Draw - U @ (U.T @ Draw))

    # --- geometry ------------------------------------------------------------
    raw_u1_cos = np.abs(Draw.T @ u1)
    geometry_rows = []
    for atom in range(cssd.K_ATOMS):
        geometry_rows.append(
            {
                "atom": atom,
                "raw_u1_abs_cos": float(raw_u1_cos[atom]),
                "raw_u1_energy_fraction": float(raw_u1_cos[atom] ** 2),
                "raw_norm": float(np.linalg.norm(Draw[:, atom])),
                "cssd_norm": float(np.linalg.norm(D_cssd[:, atom])),
                "cos_raw_cssd": float(Draw[:, atom] @ D_cssd[:, atom]),
                "cos_raw_perp_cssd": float(D_raw_perp[:, atom] @ D_cssd[:, atom]),
                "cos_raw_rawperp": float(Draw[:, atom] @ D_raw_perp[:, atom]),
                "cssd_u1_abs_cos": float(np.abs(D_cssd[:, atom] @ u1)),
            }
        )
    runner._write_csv(
        STRUCTURE / "dictionary_geometry.csv",
        geometry_rows,
        [
            "atom",
            "raw_u1_abs_cos",
            "raw_u1_energy_fraction",
            "raw_norm",
            "cssd_norm",
            "cos_raw_cssd",
            "cos_raw_perp_cssd",
            "cos_raw_rawperp",
            "cssd_u1_abs_cos",
        ],
    )

    # --- per-graph reuse ------------------------------------------------------
    train_phi = dcarun.load_phi("train")
    valid_phi = dcarun.load_phi("valid")
    alpha_cssd_train = cssd.residual_codes(model, train_phi)
    alpha_cssd_valid = cssd.residual_codes(model, valid_phi)
    raw_train = cssd.raw_codes(train_phi, Draw_entry["dictionary"])
    raw_valid = cssd.raw_codes(valid_phi, Draw_entry["dictionary"])
    codes_by_split = {
        "train": (raw_train, alpha_cssd_train),
        "valid": (raw_valid, alpha_cssd_valid),
    }
    graph_rows: list[dict[str, Any]] = []
    argmax_rows: list[dict[str, Any]] = []
    argmax_payload: dict[str, dict[int, int]] = {}
    for split in SPLITS:
        phi = train_phi if split == "train" else valid_phi
        ids = runner._graph_ids(split)
        n_graphs = int(ids.max()) + 1
        alpha_raw, alpha_cssd = codes_by_split[split]
        for label, alpha in (("RAW", alpha_raw), ("CSSD", alpha_cssd)):
            stats = _graph_mass_stats(alpha, ids, n_graphs)
            graph_rows.append(
                {
                    "split": split,
                    "representation": label,
                    "n_graphs": n_graphs,
                    "effective_atoms_mean": float(np.mean(stats["effective_atoms"])),
                    "effective_atoms_median": float(np.median(stats["effective_atoms"])),
                    "effective_atoms_p10": float(np.quantile(stats["effective_atoms"], 0.10)),
                    "effective_atoms_p90": float(np.quantile(stats["effective_atoms"], 0.90)),
                    "atoms_ge_5pct_mean": float(np.mean(stats["atoms_ge_5pct"])),
                    "atoms_ge_5pct_median": float(np.median(stats["atoms_ge_5pct"])),
                    "max_share_mean": float(np.mean(stats["max_share"])),
                    "max_share_median": float(np.median(stats["max_share"])),
                }
            )
            counts = np.bincount(stats["argmax"], minlength=cssd.K_ATOMS)
            argmax_payload[f"{split}/{label}"] = {
                int(atom): int(count) for atom, count in enumerate(counts) if count
            }
            for atom, count in enumerate(counts):
                argmax_rows.append(
                    {
                        "split": split,
                        "representation": label,
                        "atom": atom,
                        "n_graphs_top1": int(count),
                        "fraction_graphs_top1": float(count / n_graphs),
                    }
                )
    runner._write_csv(
        STRUCTURE / "graph_reuse.csv",
        graph_rows,
        [
            "split",
            "representation",
            "n_graphs",
            "effective_atoms_mean",
            "effective_atoms_median",
            "effective_atoms_p10",
            "effective_atoms_p90",
            "atoms_ge_5pct_mean",
            "atoms_ge_5pct_median",
            "max_share_mean",
            "max_share_median",
        ],
    )
    runner._write_csv(
        STRUCTURE / "argmax_atom_hist.csv",
        argmax_rows,
        ["split", "representation", "atom", "n_graphs_top1", "fraction_graphs_top1"],
    )

    # --- common-coordinate correlations (valid, descriptive) -----------------
    Z_valid = dca.structural_descriptors(valid_phi)
    c = np.asarray(valid_phi, dtype=np.float64) @ U
    from scipy.stats import spearmanr

    correlations: list[dict[str, Any]] = []
    for coordinate in range(U.shape[1]):
        for index, name in enumerate(dca.STRUCTURAL_DESCRIPTOR_NAMES):
            column = Z_valid[:, index]
            if column.std() == 0:
                rho = float("nan")
            else:
                rho = float(spearmanr(c[:, coordinate], column).statistic)
            correlations.append({"coordinate": f"c{coordinate + 1}", "feature": name, "spearman": rho})
    finite = sorted(
        (row for row in correlations if np.isfinite(row["spearman"])),
        key=lambda row: -abs(row["spearman"]),
    )

    # --- usage vs geometry ----------------------------------------------------
    rates = np.asarray(
        [
            float(row["activation_rate_valid"])
            for row in _read_csv(STRUCTURE / "atom_profiles.csv")
        ]
    )
    rho_u1_rate, _p = spearmanr(raw_u1_cos, rates)

    # --- reverse recoverability: is the CSSD code a re-parameterisation? -----
    reverse_rows: list[dict[str, Any]] = []
    metrics = dca.fit_linear_recoverability(
        alpha_cssd_train, raw_train, alpha_cssd_valid, raw_valid
    )
    metrics.update({"source_representation": "cssd_iht10", "target_representation": "raw_iht10"})
    reverse_rows.append(metrics)
    c1_train = np.asarray(train_phi, dtype=np.float64) @ U
    c1_valid = np.asarray(valid_phi, dtype=np.float64) @ U
    metrics = dca.fit_linear_recoverability(
        np.concatenate([c1_train, alpha_cssd_train], axis=1),
        raw_train,
        np.concatenate([c1_valid, alpha_cssd_valid], axis=1),
        raw_valid,
    )
    metrics.update({"source_representation": "c1_plus_cssd_iht10", "target_representation": "raw_iht10"})
    reverse_rows.append(metrics)
    runner._write_csv(
        STRUCTURE / "reverse_recoverability.csv",
        [
            {
                "source_representation": row["source_representation"],
                "target_representation": row["target_representation"],
                "valid_r2": row["valid_r2"],
                "per_dim_r2_median": row["per_dimension_r2"]["median"],
                "valid_mean_cosine": row["valid_mean_cosine"]["mean"],
                "linear_cka": row["linear_cka"],
            }
            for row in reverse_rows
        ],
        ["source_representation", "target_representation", "valid_r2", "per_dim_r2_median", "valid_mean_cosine", "linear_cka"],
    )

    # --- vocabulary diversity: top-5 |SMD| feature sets ----------------------
    def _top5_sets(alpha: np.ndarray) -> list[set[str]]:
        Z = dca.structural_descriptors(train_phi)
        active = np.abs(alpha) > 0.0
        mu = Z.mean(axis=0)
        sd = np.where(Z.std(axis=0) > 1e-12, Z.std(axis=0), 1.0)
        sets: list[set[str]] = []
        for atom in range(alpha.shape[1]):
            rows = active[:, atom]
            if rows.sum() < 2:
                sets.append(set())
                continue
            smd = (Z[rows].mean(axis=0) - mu) / sd
            order = np.argsort(-np.abs(smd))[:5]
            sets.append({dca.STRUCTURAL_DESCRIPTOR_NAMES[i] for i in order})
        return sets

    def _diversity(sets: list[set[str]]) -> dict[str, float]:
        nonempty = [s for s in sets if s]
        overlaps: list[float] = []
        for i in range(len(nonempty)):
            for j in range(i + 1, len(nonempty)):
                union = nonempty[i] | nonempty[j]
                overlaps.append(len(nonempty[i] & nonempty[j]) / len(union) if union else 0.0)
        covered = set().union(*nonempty) if nonempty else set()
        return {
            "mean_pairwise_jaccard_top5": float(np.mean(overlaps)),
            "n_distinct_features_covered": float(len(covered)),
            "n_atoms_with_top5": float(len(nonempty)),
        }

    diversity = {
        "CSSD": _diversity(_top5_sets(alpha_cssd_train)),
        "RAW": _diversity(_top5_sets(raw_train)),
    }
    payload = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "git_commit": runner._git_commit(),
        "official_test_loaded": False,
        "post_hoc_descriptive_only": True,
        "selected_common_dim": selected,
        "geometry_summary": {
            "raw_u1_abs_cos_mean": float(np.mean(raw_u1_cos)),
            "raw_u1_abs_cos_median": float(np.median(raw_u1_cos)),
            "n_atoms_abs_cos_gt_050": int((raw_u1_cos > 0.5).sum()),
            "n_atoms_abs_cos_gt_090": int((raw_u1_cos > 0.9).sum()),
            "mean_cos_raw_cssd": float(np.mean([row["cos_raw_cssd"] for row in geometry_rows])),
            "median_cos_raw_cssd": float(np.median([row["cos_raw_cssd"] for row in geometry_rows])),
            "mean_cos_raw_perp_cssd": float(
                np.mean([row["cos_raw_perp_cssd"] for row in geometry_rows])
            ),
            "mean_cos_raw_rawperp": float(np.mean([row["cos_raw_rawperp"] for row in geometry_rows])),
            "max_cssd_u1_abs_cos": float(np.max([row["cssd_u1_abs_cos"] for row in geometry_rows])),
            "spearman_raw_u1_cos_vs_cssd_valid_rate": float(rho_u1_rate),
        },
        "common_coordinate_top_correlations_valid": finite[:12],
        "graph_reuse": graph_rows,
        "argmax_hist": argmax_payload,
        "reverse_recoverability": [
            {
                "source_representation": row["source_representation"],
                "target_representation": row["target_representation"],
                "valid_r2": row["valid_r2"],
                "per_dim_r2_median": row["per_dimension_r2"]["median"],
                "valid_mean_cosine": row["valid_mean_cosine"]["mean"],
                "linear_cka": row["linear_cka"],
            }
            for row in reverse_rows
        ],
        "vocabulary_diversity": diversity,
        "atom_profiles": {
            "n_atoms_rate_gt_090": int((rates > 0.9).sum()),
            "n_atoms_rate_gt_050": int((rates > 0.5).sum()),
            "max_rate": float(rates.max()),
            "top_rate_atom": int(rates.argmax()),
        },
    }
    runner._write_json(STRUCTURE / "extra_summary.json", payload)
    print(json.dumps({
        "geometry_summary": payload["geometry_summary"],
        "top_correlations": payload["common_coordinate_top_correlations_valid"][:6],
        "graph_reuse": graph_rows,
    }, indent=1))
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
