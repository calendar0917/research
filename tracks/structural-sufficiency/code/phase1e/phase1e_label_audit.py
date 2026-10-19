"""Phase 1e Stage A1 — audit the ZINC target's cycle-penalty mechanism.

Answers, on the frozen 1,000-molecule Monitor and (as a reference distribution)
on the 10,000-molecule official train subset:

* which ring statistic the label actually uses (NetworkX ``cycle_basis`` in the
  generator's canonical atom order) versus the RDKit ``GetSymmSSSR`` statistic
  Phase 1d used for its large-ring split;
* how many of the 40 "large ring" molecules really carry a non-zero cycle
  penalty, and how many penalised molecules the RDKit split misses;
* how large the label reconstruction error is, per group;
* for how many penalised molecules the penalty changes under node relabelling.

CPU only.  No model is trained, no label is used as a model input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (HERE, REPO / "tracks/structural-sufficiency/code/phase1b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase1e_common as C  # noqa: E402

OUT = C.RESULTS_1E / "label_audit.json"
PERM_REPS = 30
PERM_SEED = 20261020


def _confusion(true_mask: np.ndarray, pred_mask: np.ndarray) -> dict[str, int]:
    t = np.asarray(true_mask, dtype=bool)
    p = np.asarray(pred_mask, dtype=bool)
    return {
        "labeled_and_rdkit_large": int((t & p).sum()),
        "labeled_not_rdkit_large": int((t & ~p).sum()),
        "rdkit_large_not_labeled": int((~t & p).sum()),
        "neither": int((~t & ~p).sum()),
    }


def permutation_span(
    atom_count: int,
    bonds: Sequence[tuple[int, int]],
    *,
    reps: int,
    rng: np.random.Generator,
) -> list[int]:
    """Cycle-basis excess over random node orderings (the order is not observable)."""
    import networkx as nx

    edges = [(int(u), int(v)) for u, v in bonds]
    n = int(atom_count)
    vals: set[int] = set()
    for _ in range(int(reps)):
        perm = rng.permutation(n)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((int(perm[u]), int(perm[v])) for u, v in edges)
        vals.add(C.cycle_excess(nx.cycle_basis(g)))
    return sorted(vals)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm-reps", type=int, default=PERM_REPS)
    args = ap.parse_args(argv)

    mon = C.load_monitor()
    feat = C.align_features(C.load_features(), mon.rows)
    smiles = C.load_canonical_smiles()

    graphs, y_all = C.load_zinc_pool()

    comp = C.reconstruct_label_components(smiles)
    lab = C.recover_label_cycle(y_all, comp["logP"], comp["SA"])

    n_atoms_graph = np.asarray([g[0] for g in graphs], dtype=np.int64)
    atom_count_mismatch = np.where(comp["n_atoms"] != n_atoms_graph)[0]

    # --- KDVD audit cross-check (optional; local-only CSV) -------------------
    ksvd = C.load_ksvd_label_table()
    ksvd_check: dict[str, Any] = {"available": ksvd is not None}
    if ksvd is not None:
        ksvd_excess = np.asarray(
            [(-float(ksvd[r]["label_effective_cycle_snapped"])) for r in range(10000)]
        )
        ksvd_stored = np.asarray(
            [float(ksvd[r]["cycle_score_stored_order"] or 0.0) for r in range(10000)]
        )
        ksvd_check.update(
            {
                "n_agree_with_recovered_excess": int((ksvd_excess == lab["excess"]).sum()),
                "n_rows": 10000,
                "n_penalty_rows_ksvd": int((ksvd_excess > 0).sum()),
                "n_penalty_rows_recovered": int((lab["excess"] > 0).sum()),
                "max_abs_excess_gap": int(np.abs(ksvd_excess - lab["excess"]).max()),
            }
        )
        stored_order = ksvd_stored
    else:
        stored_order = None

    # --- per-row structure-side ring statistics (all 10,000 rows) ------------
    large_ring_10k = np.zeros(10000, dtype=bool)
    stored_excess_10k = np.zeros(10000, dtype=np.int64)
    for r in range(10000):
        stored_excess_10k[r] = C.cycle_excess(C.pyg_cycle_basis(graphs[r]))
    canon_excess_10k = comp["canonical_excess"]
    sssr_excess_10k = np.zeros(10000, dtype=np.int64)
    # Phase 1d features are Monitor-only; recompute SSSR for the reference split.
    sssr_excess_10k, large_ring_10k = _rdkit_sssr_excess(smiles)

    penalised_10k = lab["excess"] > 0
    agreement = {
        "label_excess_vs_canonical_order": int((lab["excess"] == canon_excess_10k).sum()),
        "label_excess_vs_stored_pyg_order": int((lab["excess"] == stored_excess_10k).sum()),
        "label_excess_vs_rdkit_sssr_order_independent": int(
            (lab["excess"] == sssr_excess_10k).sum()
        ),
        "n_rows": 10000,
    }

    # --- permutation stability on the penalised-union-large set --------------
    rng = np.random.default_rng(PERM_SEED)
    union_10k = np.where(penalised_10k | large_ring_10k)[0]
    spans: dict[int, list[int]] = {}
    for r in union_10k:
        spans[int(r)] = permutation_span(
            graphs[r][0], graphs[r][1], reps=int(args.perm_reps), rng=rng
        )
    unstable_10k = [r for r, v in spans.items() if len(v) > 1]

    # --- Monitor slice -------------------------------------------------------
    pos = {int(r): i for i, r in enumerate(np.arange(10000))}
    mon_rows = mon.rows
    idx10k = np.asarray([pos[int(r)] for r in mon_rows], dtype=np.int64)
    mon_label_excess = lab["excess"][idx10k]
    mon_canon = canon_excess_10k[idx10k]
    mon_stored = stored_excess_10k[idx10k]
    mon_sssr = sssr_excess_10k[idx10k]
    mon_large = feat["large_cycles_gt6"] >= 1
    mon_pen = mon_label_excess > 0

    # Consistency of the Phase 1d `large_cycles_gt6` feature with the recomputed SSSR.
    sssr_feature_check = int((mon_large == (mon_sssr > 0)).sum())

    recon_abs = np.abs(lab["snap_error"]) / lab["sigma_cycle"]

    def group(name: str, mask: np.ndarray) -> dict[str, Any]:
        m = np.asarray(mask, dtype=bool)
        return {
            "name": name,
            "n": int(m.sum()),
            "mean_label_excess": float(mon_label_excess[m].mean()) if m.any() else 0.0,
            "mean_cycle_component": float(lab["cycle_component"][idx10k][m].mean())
            if m.any()
            else 0.0,
            "mean_abs_cycle_component": float(
                np.abs(lab["cycle_component"][idx10k][m]).mean()
            )
            if m.any()
            else 0.0,
            "mean_abs_y_without_cycle": float(
                np.abs(lab["y_without_cycle"][idx10k][m]).mean()
            )
            if m.any()
            else 0.0,
            "recon_abs_error_mean": float(recon_abs[idx10k][m].mean()) if m.any() else 0.0,
            "recon_abs_error_p95": float(np.percentile(recon_abs[idx10k][m], 95))
            if m.any()
            else 0.0,
            "recon_abs_error_max": float(recon_abs[idx10k][m].max()) if m.any() else 0.0,
        }

    mon_groups = [
        group("rdkit_large_ring_only", mon_large & ~mon_pen),
        group("label_penalty_only", mon_pen & ~mon_large),
        group("both", mon_large & mon_pen),
        group("neither", ~mon_large & ~mon_pen),
        group("label_penalty_union", mon_pen),
        group("complement_of_label_penalty", ~mon_pen),
    ]

    # --- per-molecule rows for the penalised Monitor molecules --------------
    per_mol: list[dict[str, Any]] = []
    for i, r in enumerate(mon_rows):
        if not mon_pen[i]:
            continue
        rep = int(idx10k[i])
        per_mol.append(
            {
                "row": int(r),
                "y": float(mon.y[i]),
                "y_without_cycle": float(lab["y_without_cycle"][rep]),
                "cycle_component": float(lab["cycle_component"][rep]),
                "label_excess": int(mon_label_excess[i]),
                "canonical_order_excess": int(mon_canon[i]),
                "stored_pyg_order_excess": int(mon_stored[i]),
                "rdkit_sssr_excess": int(mon_sssr[i]),
                "permutation_excess_values": spans.get(rep, []),
                "rdkit_large_ring": bool(mon_large[i]),
                "recon_abs_error": float(recon_abs[rep]),
                "snap_error": float(lab["snap_error"][rep]),
            }
        )

    unstable_monitor = [p for p in per_mol if len(p["permutation_excess_values"]) > 1]

    payload = {
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "stage": "A1_label_mechanism",
        "scope": {
            "monitor_rows": int(mon_rows.size),
            "official_train_rows": 10000,
            "reserved_labels_read": 0,
            "official_valid_test_read": 0,
        },
        "formula": {
            "source": "Kusner et al. 2017 GVAE generate_latent_features_and_targets.py",
            "source_sha256": C.sha256_file(
                REPO
                / "tracks/ksvd/results/zinc_long_cycle_audit/upstream"
                / "generate_latent_features_and_targets.py"
            ),
            "cycle_statistic": "nx.cycle_basis(nx.Graph(rdmops.GetAdjacencyMatrix(mol)))",
            "cycle_excess": "max(0, max_cycle_len - 6)",
            "atom_order": "RDKit canonical SMILES order of the label generator's release",
            "not_the_benchmark_definition": "PyG's docstring says 'number of cycles with >6 atoms'; that is NOT what the generator computes",
            "constants": C.COMMUNITY_CONSTANTS,
            "fitted_cycle_scale": {
                "sigma_cycle": float(lab["sigma_cycle"]),
                "mu_cycle": float(lab["mu_cycle"]),
            },
            "input_hashes": {
                "zinc_smi": C.sha256_file(C.ZINC_SMI),
                "fpscores": C.sha256_file(C.FPSCORES),
                "sascorer": C.sha256_file(C.SASCORER_SRC),
                "smiles_table": C.sha256_file(C.SMILES_TABLE),
                "split": C.sha256_file(C.SPLIT_PATH),
                "features": C.sha256_file(C.FEATURES_PATH),
            },
        },
        "reconstruction": {
            "ksvd_cross_check": ksvd_check,
            "excess_agreement_10k": agreement,
            "canonical_smiles_atom_count_mismatch_rows": [int(x) for x in atom_count_mismatch],
            "n_canonical_smiles_atom_count_mismatch": int(atom_count_mismatch.size),
            "phase1d_large_cycles_gt6_vs_recomputed_sssr_agree_rows": sssr_feature_check,
            "phase1d_large_cycles_gt6_vs_recomputed_sssr_n": int(mon_rows.size),
            "snap_error_p95_y_units": float(np.percentile(recon_abs, 95)),
            "snap_error_max_y_units": float(recon_abs.max()),
            "n_penalty_rows_10k": int(penalised_10k.sum()),
            "n_rdkit_large_rows_10k": int(large_ring_10k.sum()),
        },
        "monitor_confusion": {
            "label_penalty_x_rdkit_large_ring": _confusion(mon_pen, mon_large),
            "counts": {
                "n_label_penalty": int(mon_pen.sum()),
                "n_rdkit_large": int(mon_large.sum()),
                "n_union": int((mon_pen | mon_large).sum()),
            },
            "rdkit_large_group_purity": float((mon_large & mon_pen).sum() / max(mon_large.sum(), 1)),
            "rdkit_large_group_recall_of_penalty": float(
                (mon_large & mon_pen).sum() / max(mon_pen.sum(), 1)
            ),
            "missing_rows": [int(r) for r in mon_rows[mon_pen & ~mon_large]],
        },
        "train10k_confusion": {
            "label_penalty_x_rdkit_large_ring": _confusion(penalised_10k, large_ring_10k),
            "counts": {
                "n_label_penalty": int(penalised_10k.sum()),
                "n_rdkit_large": int(large_ring_10k.sum()),
            },
            "rdkit_large_group_purity": float(
                (penalised_10k & large_ring_10k).sum() / max(large_ring_10k.sum(), 1)
            ),
            "rdkit_large_group_recall_of_penalty": float(
                (penalised_10k & large_ring_10k).sum() / max(penalised_10k.sum(), 1)
            ),
        },
        "monitor_ring_statistic_agreement": {
            "label_vs_canonical_order_rows": int((mon_label_excess == mon_canon).sum()),
            "label_vs_stored_pyg_order_rows": int((mon_label_excess == mon_stored).sum()),
            "label_vs_rdkit_sssr_rows": int((mon_label_excess == mon_sssr).sum()),
            "n": int(mon_rows.size),
        },
        "permutation_stability": {
            "reps": int(args.perm_reps),
            "seed": PERM_SEED,
            "n_candidates_10k": int(union_10k.size),
            "n_unstable_10k": int(len(unstable_10k)),
            "unstable_rows_10k": [int(r) for r in sorted(unstable_10k)],
            "n_penalty_monitor": int(mon_pen.sum()),
            "n_unstable_penalty_monitor": int(len(unstable_monitor)),
            "unstable_penalty_monitor_rows": [int(p["row"]) for p in unstable_monitor],
            "interpretation": (
                "the cycle basis is not a permutation-invariant graph function; the "
                "label used one particular atom ordering, and a model sees only the "
                "unordered graph"
            ),
        },
        "monitor_groups": mon_groups,
        "monitor_penalised_molecules": per_mol,
    }
    C.write_json(OUT, payload)
    print(
        f"[A1] monitor penalty {int(mon_pen.sum())} / rdkit-large {int(mon_large.sum())} "
        f"| purity {(mon_large & mon_pen).sum()}/{mon_large.sum()} "
        f"| missed {int((mon_pen & ~mon_large).sum())} "
        f"| unstable {len(unstable_monitor)}"
    )
    print(f"[A1] wrote {OUT.relative_to(REPO)}")
    return 0


def _rdkit_sssr_excess(smiles: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem

    n = len(smiles)
    excess = np.zeros(n, dtype=np.int64)
    large = np.zeros(n, dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        Chem.GetSymmSSSR(mol)
        sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
        longest = max(sizes) if sizes else 0
        excess[i] = 0 if longest <= 6 else int(longest - 6)
        large[i] = longest > 6
    return excess, large


if __name__ == "__main__":
    raise SystemExit(main())
