"""cscl-strong-nongnn-predictor-v1 — Phase A CPU-only audit (no training).

Question (protocol: ``configs/strong_nongnn_predictor_v1.yaml``, frozen):
Before the single authorized GPU run, restate what limits the current Full-Y
prediction (from existing curves and per-molecule exports), audit the
historical strong-prediction / high-order-combination routes, and quantify the
proposed static triplet definition (centre atom + two distinct real-bond
neighbours, unordered endpoints) on the frozen internal split.

This script only READS committed historical artifacts and the frozen
internal-split data.  It never trains, never loads official valid/test, and
never reads target-decomposition / g / ell / s / c files.

Outputs: ``results/strong_nongnn_predictor_v1/phase_a_s0.json``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as drv2  # noqa: E402

OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/strong_nongnn_predictor_v1"
FULL_Y_TRAIN = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/train_Full-Y-fit7200-seed0.json"
FULL_Y_DEVS = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_dev_preds.npz"
O_BUCKET_TRAIN = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d/train_O_bucket_s0.json"
V3D_PHASE_A = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d/phase_a_s0.json"
V3D_ANALYSIS = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d/analysis_s0.json"


def _json(payload, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. existing fit / monitor / dev curves
# ---------------------------------------------------------------------------


def existing_curves() -> dict:
    full = json.loads(FULL_Y_TRAIN.read_text())
    curve = full["curve"]
    best = min(curve, key=lambda c: c["monitor_mae"])
    # last-40-epoch monitor tendency vs the earlier plateau
    tail = [c["monitor_mae"] for c in curve if c["epoch"] >= 201]
    head = [c["monitor_mae"] for c in curve if c["epoch"] <= 60]
    ob = json.loads(O_BUCKET_TRAIN.read_text())
    ob_curve = ob["curve"]
    ob_best = min(ob_curve, key=lambda c: c["monitor_mae"])
    return {
        "full_y": {
            "source": str(FULL_Y_TRAIN.relative_to(REPO_ROOT)),
            "primary_soup": full["primary_soup"],
            "secondary_best_monitor": full["secondary_best_monitor"],
            "tertiary_last": full["tertiary_last"],
            "soup_members": full["soup_members"],
            "best_monitor_epoch": best["epoch"],
            "best_monitor_mae": best["monitor_mae"],
            "tail_monitor_mae_201_240": {"mean": float(np.mean(tail)), "min": float(np.min(tail)), "max": float(np.max(tail))},
            "head_monitor_mae_1_60": {"mean": float(np.mean(head)), "min": float(np.min(head)), "max": float(np.max(head))},
            "train_curve_keys": [c["epoch"] for c in curve if c["epoch"] in (1, 5, 10, 20, 40, 60, 80, 120, 160, 200, 240)],
            "train_task_mae": {str(c["epoch"]): c["train_task_mae"] for c in curve if c["epoch"] in (1, 5, 10, 20, 40, 60, 80, 120, 160, 200, 240)},
            "monitor_mae": {str(c["epoch"]): c["monitor_mae"] for c in curve if c["epoch"] in (1, 5, 10, 20, 40, 60, 80, 120, 160, 200, 240)},
            "gap_fit_to_monitor_soup": full["primary_soup"]["monitor_mae"] - full["primary_soup"]["fit_mae"],
            "gap_fit_to_dev_soup": full["primary_soup"]["dev_mae"] - full["primary_soup"]["fit_mae"],
            "gap_monitor_to_dev_soup": full["primary_soup"]["dev_mae"] - full["primary_soup"]["monitor_mae"],
            "wall_seconds": full["wall_seconds"],
            "n_params": full["n_params"],
        },
        "o_bucket": {
            "source": str(O_BUCKET_TRAIN.relative_to(REPO_ROOT)),
            "primary_best_monitor": ob["primary_best_monitor"],
            "best_epoch": ob["best_epoch"],
            "soup_epochs": ob["soup_epochs"],
            "best_monitor_mae": ob_best["monitor_mae"],
            "gap_fit_to_monitor": ob["primary_best_monitor"]["monitor_mae"] - ob["primary_best_monitor"]["fit_mae"],
            "gap_fit_to_dev": ob["primary_best_monitor"]["dev_mae"] - ob["primary_best_monitor"]["fit_mae"],
            "gap_monitor_to_dev": ob["primary_best_monitor"]["dev_mae"] - ob["primary_best_monitor"]["monitor_mae"],
            "curve_has_train_mae": "train_task_mae" in ob_curve[0],
        },
    }


# ---------------------------------------------------------------------------
# 2. Full-Y per-molecule dev error structure (from the frozen export)
# ---------------------------------------------------------------------------


def full_y_dev_errors() -> dict:
    z = np.load(FULL_Y_DEVS)
    pred = z["pred"].astype(np.float64)
    y = z["y"].astype(np.float64)
    e = np.abs(pred - y)
    order = np.argsort(-e)
    total = float(e.sum())
    topk = {}
    for k in (1, 5, 10, 20, 50, 100, 200):
        topk[f"top{k}"] = float(e[order[:k]].sum() / total)
    bands = {
        "y_le_-3": y <= -3.0,
        "y_-3_-1": (y > -3.0) & (y <= -1.0),
        "y_-1_1": (y > -1.0) & (y <= 1.0),
        "y_gt_1": y > 1.0,
    }
    band_stats = {}
    for name, mask in bands.items():
        band_stats[name] = {
            "n": int(mask.sum()),
            "mae": float(e[mask].mean()) if mask.any() else None,
            "error_mass_share": float(e[mask].sum() / total),
        }
    return {
        "source": str(FULL_Y_DEVS.relative_to(REPO_ROOT)),
        "n": int(len(y)),
        "mae": float(e.mean()),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "max": float(e.max()),
        "corr_abs_err_y": float(np.corrcoef(e, y)[0, 1]),
        "topk_error_mass_share": topk,
        "y_band_error": band_stats,
        "y_le_-3_share_n": float((y <= -3.0).mean()),
        "y_le_-3_share_error_mass": band_stats["y_le_-3"]["error_mass_share"],
    }


# ---------------------------------------------------------------------------
# 3. subgroup / UNK evidence copied from the frozen v3d round (same molecules)
# ---------------------------------------------------------------------------


def subgroup_evidence() -> dict:
    v = json.loads(V3D_ANALYSIS.read_text())
    phase = json.loads(V3D_PHASE_A.read_text())
    subgroups = {}
    for key, row in v["subgroups"].items():
        if isinstance(row, dict) and "mae_F" in row:
            subgroups[key] = {"n": row["n"], "mae_F": row["mae_F"]}
    return {
        "source": str(V3D_ANALYSIS.relative_to(REPO_ROOT)),
        "full_y_dev_subgroups": subgroups,
        "unk": {
            "source": str(V3D_PHASE_A.relative_to(REPO_ROOT)),
            "n_with_unk": phase.get("unk_audit", {}).get("n_with_unk"),
            "n_without_unk": phase.get("unk_audit", {}).get("n_without_unk"),
            "full_y_with_unk": phase.get("unk_audit", {}).get("mae_with_unk", {}).get("F"),
            "full_y_without_unk": phase.get("unk_audit", {}).get("mae_without_unk", {}).get("F"),
        },
    }


# ---------------------------------------------------------------------------
# 4. historical audit (unchanged facts from tracks/ksvd; read-only summary)
# ---------------------------------------------------------------------------


def historical_audit() -> dict:
    return {
        "source": "tracks/ksvd/STATE.yaml (read-only; exact texts at the cited keys)",
        "effective_non_gnn_representations": [
            "CSSD-q1 + C6 static incidence environment + pair backend + reader (Sem108 / LatentBridge / Scale-v1): the direct ancestor of Full-Y; Sem108 soup valid 0.12370, latent-bridge 0.12106 (borderline, no matched control), scale-v1 Full m=3 0.11915 official-valid under the OLD 10k protocol.",
            "v3d bucket arms: Full-Y distance-bucket pair moments recover the O-F gap (O_bucket 0.11610 vs F 0.11641 internal dev); readout structure is not the information bottleneck.",
        ],
        "failed_high_order_or_pair_state_routes": [
            "PSD-v0 (pair-state H_ij + M_ij = sum_k phi(H_ik)psi(H_kj), L=3): dominated by a raw node baseline at matched budget (best monitor 0.5038 vs 0.4115; train 0.195 vs monitor 0.513 = overfits). Decision: stop; no dictionary.",
            "GSCN-v0 (task-driven sparse dictionary core): capacity-only failure (generic param-matched control 0.1982 <= sparse 0.2123); mechanism active but no task gain. Decision: reject dictionary core.",
            "BAT-v1 (bond-anchored triples (i,j;k), 64-D sym(F(p_ij,p_ik,p_jk)) appended to the frozen parent's 302-D reader input, re-init Reader, 80 epochs on official train 10k): M_soup 0.12300 vs unmatched parent replay 0.12370, 0.00300 above the frozen 0.120 gate -> no strong signal; triple operator computed from frozen pair tokens, not live E.",
            "BAT-joint-tune-v1 (hot start from BAT soup, jointly fine-tune pair path + F + reader online): M 0.123159 vs hot start 0.123003 (worse); train MAE fell ~18% while valid never improved.",
            "JointBond-v1 (key-level structure-semantics fusion): branch exactly inert at the endpoint (denormal underflow, G_branch_off = 0.0); only capacity/trajectory differences.",
            "RoleCorr / RNDB / hierarchical-relation / typed-cycle: mechanism load-bearing but no absolute-gain band; capacity-localization screens had no power.",
            "TCCD-v0..v7 (fixed-coordinate canonical patches / normalised moments): representation not a viable K-SVD domain or standalone insufficient; frozen standalone route CLOSED.",
        ],
        "what_the_new_round_changes": [
            "triplet set = centre j with two DISTINCT real-bond neighbours i,k (bond-angle wedges; unordered endpoints), not bond-anchor x arbitrary third atom and not O(n^3) atom triples",
            "triplet features are built from the LIVE learned per-atom environment E (through a shared 144->32 projection), not from frozen cached pair tokens",
            "the branch is trained END-TO-END jointly with the whole Full-Y (no frozen parent, no detached head, no distillation labels)",
            "the branch is appended as new reader input columns (zero-initialised) so the untouched Full-Y function is exactly contained at initialisation",
        ],
        "unresolved_confounds": [
            "no parameter-matched extra-training control (protocol forbids): improvement cannot be fully attributed to the triplet mechanism vs added capacity/optimisation trajectory",
            "single seed 0; internal dev 2000 reused since cscl-v0 (development bias)",
            "historical BAT-v1 already showed a triple-to-reader composition with frozen features did not move held-out MAE; the new mechanism must beat that precedent, not just the parent.",
        ],
    }


# ---------------------------------------------------------------------------
# 5. triplet definition audit on the frozen internal split (no training)
# ---------------------------------------------------------------------------


def triplet_definition_audit() -> dict:
    train = drv2.load_train_rows(log=lambda *a, **k: None)
    drv2.verify_row_alignment(train, log=lambda *a, **k: None)
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])

    n_mols = len(train)
    split_of = np.full(n_mols, -1, dtype=np.int64)
    for code, name in ((0, "fit_inner"), (1, "monitor"), (2, "dev")):
        split_of[np.asarray(idx[name], dtype=np.int64)] = code
    code_to_name = {0: "fit_inner", 1: "monitor", 2: "dev"}

    stats = {name: {"n_molecules": 0, "n_atoms": 0, "n_bonds": 0, "n_triplets": 0,
                    "bond_mismatch_env_vs_pair": 0, "duplicate_bond_rows": 0, "self_loop_rows": 0,
                    "max_degree": 0, "deg_hist": {}, "n_zero_triplet_mols": 0}
             for name in ("fit_inner", "monitor", "dev", "all")}
    extent = {}
    per_mol = {name: [] for name in ("fit_inner", "monitor", "dev", "all")}
    triplet_per_atom = []
    atom_counts = []
    triplet_counts = []
    deg_counts_global = np.zeros(6, dtype=np.int64)
    for i in range(n_mols):
        d = train[i]
        n = int(d.dict_phi.shape[0])
        if split_of[i] < 0:
            continue
        split_name = code_to_name[int(split_of[i])]
        pair_index = d.pair_index
        pair_bucket = d.pair_bucket
        bond_mask = pair_bucket == 0
        bu = pair_index[0][bond_mask]
        bv = pair_index[1][bond_mask]
        n_bonds = int(bu.numel())
        # unique env bonds
        lu = d.env_bond_u
        lv = d.env_bond_v
        if lu.numel() and lv.numel():
            lo = np.minimum(lu.numpy(), lv.numpy())
            hi = np.maximum(lu.numpy(), lv.numpy())
            env_keys = set(zip(lo.tolist(), hi.tolist()))
        else:
            env_keys = set()
        pair_keys = set(zip(bu.numpy().tolist(), bv.numpy().tolist()))
        mismatch = len(env_keys ^ pair_keys)
        dup = n_bonds - len(pair_keys)
        self_loops = int((bu == bv).sum())
        # degree / wedge count
        deg = np.bincount(np.concatenate([bu.numpy(), bv.numpy()]), minlength=n) if n_bonds else np.zeros(n, dtype=int)
        dmax = int(deg.max()) if n else 0
        n_triplets = int(sum(int(v) * (int(v) - 1) // 2 for v in deg))
        deg_hist = np.bincount(deg, minlength=6)[:6] if n else np.zeros(6, dtype=np.int64)
        deg_counts_global += deg_hist.astype(np.int64)
        # per-triplet degree slots sanity
        extent["max_degree_seen"] = max(extent.get("max_degree_seen", 0), dmax)
        for name in (split_name, "all"):
            s = stats[name]
            s["n_molecules"] += 1
            s["n_atoms"] += n
            s["n_bonds"] += n_bonds
            s["n_triplets"] += n_triplets
            s["bond_mismatch_env_vs_pair"] += mismatch
            s["duplicate_bond_rows"] += int(dup)
            s["self_loop_rows"] += self_loops
            s["max_degree"] = max(s["max_degree"], dmax)
            s["deg_hist"][str(dmax)] = s["deg_hist"].get(str(dmax), 0) + 1
            if n_triplets == 0:
                s["n_zero_triplet_mols"] += 1
        per_mol[split_name].append((n, n_bonds, n_triplets, dmax))
        per_mol["all"].append((n, n_bonds, n_triplets, dmax))
        atom_counts.append(n)
        triplet_counts.append(n_triplets)
        triplet_per_atom.append(n_triplets / max(n, 1))

    def _corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "bond_definition": "pair_bucket == 0 over the frozen full pair_index (source<target), i.e. graph distance 1",
        "triplet_definition": "for each atom j, every unordered pair {i,k} of distinct real-bond neighbours of j (wedge); emitted once per centre",
        "max_degree_seen": extent.get("max_degree_seen", 0),
        "bonds_equal_env_occurrence_dedup_all_10k": bool(all(s["bond_mismatch_env_vs_pair"] == 0 for s in stats.values())),
        "duplicate_bond_rows_all_10k": int(stats["all"]["duplicate_bond_rows"]),
        "self_loop_rows_all_10k": int(stats["all"]["self_loop_rows"]),
        "splits": {name: {k: v for k, v in s.items() if k != "deg_hist"} | {"deg_hist": s["deg_hist"]} for name, s in stats.items()},
        "dev": {
            "mean_triplets_per_molecule": float(np.mean([t for _, _, t, _ in per_mol["dev"]])),
            "mean_triplets_per_atom": float(np.mean(triplet_per_atom)),
            "zero_triplet_molecules": int(sum(1 for _, _, t, _ in per_mol["dev"] if t == 0)),
            "corr_triplets_atoms": _corr(atom_counts, triplet_counts),
            "corr_triplets_per_atom_atoms": _corr(atom_counts, triplet_per_atom),
            "triplet_count_quantiles": [float(np.percentile(triplet_counts, q)) for q in (0, 25, 50, 75, 90, 99, 100)],
            "atom_degree_histogram_0_to_5": [int(v) for v in deg_counts_global],
            "max_degree_hist": stats["dev"]["deg_hist"],
        },
        "growth_note": "triplet count = sum_j C(deg_j, 2); it grows quadratically with the centre degree and linearly with the number of atoms at fixed degree profile",
    }


def main() -> int:
    payload = {
        "protocol": "cscl-strong-nongnn-predictor-v1",
        "phase": "A (CPU-only audit, no training)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "existing_curves": existing_curves(),
        "full_y_dev_error_structure": full_y_dev_errors(),
        "subgroup_evidence": subgroup_evidence(),
        "historical_audit": historical_audit(),
        "triplet_definition_audit": triplet_definition_audit(),
    }
    _json(payload, "phase_a_s0.json")
    print(json.dumps({k: (v if k in ("protocol", "phase", "official_valid_loaded", "official_test_loaded") else "...see json...") for k, v in payload.items()}, indent=2))
    print(f"\nwrote {OUT_DIR / 'phase_a_s0.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
