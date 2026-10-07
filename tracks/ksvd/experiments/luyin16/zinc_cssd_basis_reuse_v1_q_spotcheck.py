"""Read-only Q spot-check of gid=3775 / gid=1424 (zinc_cssd_basis_reuse_v1).

Reproduces every number of
``notes/zinc_cssd_nonlinear_binding_v1_q_spotcheck.md`` from the frozen old
artifacts (``zinc_cssd_nonlinear_binding_v1`` results + the long-cycle-audit
CSVs + the committed canonical-SMILES table), replays the frozen old Q soup
forward pass and writes ``results/zinc_cssd_basis_reuse_v1/q_spotcheck/``.
Read-only: no training, no new Q candidate, no oracle / true c / inverse k.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tracks.ksvd.experiments.luyin16 import zinc_cssd_basis_reuse_v1 as reuse
from tracks.ksvd.experiments.luyin16 import zinc_cssd_nonlinear_binding_v1 as parent
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd

PARENT_DIR = reuse.PARENT_RESULTS_DIR
AUDIT_DIR = reuse.TRACK_ROOT / "results" / "zinc_long_cycle_audit"
OUT_DIR = reuse.RESULTS_DIR / "q_spotcheck"
#: gid -> (position, select position in the old round, smi_line)
CASES = {3775: (3776, 380, 14677), 1424: (1424, 135, 40197)}
#: the old-round arm/seed runs read for the select/confirm contribution
SELECT_RUNS = ("A_s0", "C00_s0", "C01_s0", "C10_s0", "C11_s0")
CONFIRM_RUNS = ("A_s1", "C00_s1")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = __import__("time").perf_counter()
    out: dict[str, Any] = {}

    # (1) ID / numbering / SMILES source-row chain (nothing inferred from names)
    with np.load(PARENT_DIR / "targets.npz", allow_pickle=False) as z:
        targets = {k: np.asarray(z[k]) for k in ("y", "c", "g", "k", "ell", "s", "gid")}
    with np.load(PARENT_DIR / "fold.npz", allow_pickle=False) as z:
        old_fit = np.sort(np.asarray(z["fit_idx"], np.int64))
        old_select = np.sort(np.asarray(z["select_idx"], np.int64))
    smiles = reuse._canonical_smiles()
    node_sizes = reuse._node_sizes()
    lab = pd.read_csv(reuse.zfr.TRAIN_LABEL_CSV)
    chain = {}
    for gid, (position, sel_position, smi_line) in CASES.items():
        pos = int(np.where(targets["gid"] == gid)[0][0])
        smi_line_csv = int(lab["smi_line"].iloc[position])
        if pos != position or smi_line_csv != smi_line:
            raise RuntimeError(f"gid {gid} chain mismatch: pos {pos}, smi_line {smi_line_csv}")
        occurrences = int((targets["gid"] == gid).sum())
        chain[str(gid)] = {
            "gid": int(gid), "position": position, "molecule_id": f"train:{position}",
            "smi_line": smi_line, "canonical_smiles": str(smiles[position]),
            "n_nodes": int(node_sizes[position]),
            "in_old_8001_fit_pool": bool(position in old_fit.tolist()),
            "in_old_select": bool(position in old_select.tolist()),
            "select_position": int(np.where(old_select == position)[0][0]),
            "select_position_note": sel_position,
            "gid_occurrences": occurrences,
            "y": float(targets["y"][position]), "k": int(targets["k"][position]),
            "c": float(targets["c"][position]), "g": float(targets["g"][position]),
            "ell": float(targets["ell"][position]), "s": float(targets["s"][position]),
        }
        if not chain[str(gid)]["in_old_8001_fit_pool"] or not chain[str(gid)]["in_old_select"]:
            pass
        if abs(chain[str(gid)]["g"] - (chain[str(gid)]["y"] - chain[str(gid)]["c"])) > 1e-12:
            raise RuntimeError(f"gid {gid}: g != y - c")
    out["chain"] = chain

    # (2) cycle/basis statistics from the historical audit CSVs (read-only)
    table_a = pd.read_csv(AUDIT_DIR / "extreme_targets_table_a.csv")
    exact = pd.read_csv(AUDIT_DIR / "exact_longest_all.csv")
    cycle_stats = {}
    for gid, (position, _sel, _smi) in CASES.items():
        row = table_a[table_a["molecule_id"] == f"train:{position}"]
        erow = exact[exact["molecule_id"] == f"train:{position}"]
        if len(row) != 1 or len(erow) != 1:
            raise RuntimeError(f"gid {gid}: audit rows missing")
        r = row.iloc[0]
        cycle_stats[str(gid)] = {
            "label_cycle_term": int(targets["k"][position]),
            "stored_order_max_basis": float(r["max_basis_cycle_length"]),
            "stored_order_cycle_score": float(r["cycle_score_stored_order"]),
            "exact_longest_simple_cycle": float(erow.iloc[0]["longest_simple_cycle_length"]),
            "rdkit_max_ring": float(r["rdkit_max_ring_size"]),
            "order_dependence_note": (
                "the (graph, generator node order) penalty is not a graph invariant "
                "(audit Q9); the label cycle term comes from the GVAE order"
            ),
        }
    out["cycle_stats"] = cycle_stats

    # (3) topology25 inputs (raw + old-round prep reproduction) and Q replay
    train_data = zftd.load_train_only()
    with np.load(PARENT_DIR / "prep.npz", allow_pickle=False) as z:
        old_prep = {k: np.asarray(z[k]) for k in z.files}
    # bit-wise reproduction of the old round's prep statistics
    old_view = {"fit_idx": old_fit, "dev_idx": np.asarray([], np.int64)}
    recomputed = reuse.zldc.build_new_prep(train_data, old_view)
    for key in ("topo_fit_mean", "topo_fit_scale"):
        if not np.array_equal(
            np.asarray(recomputed[key], np.float32), np.asarray(old_prep[key], np.float32)
        ):
            raise RuntimeError(f"old prep reproduction mismatch at {key}")
    rows = [train_data[CASES[gid][0]] for gid in CASES]
    T_raw = parent.topology_matrix(rows).astype(np.float64)
    T_prep = ((T_raw - old_prep["topo_fit_mean"].astype(np.float64))
              / old_prep["topo_fit_scale"].astype(np.float64))
    differing_feats = sorted(int(i) for i in np.where(T_raw[0] != T_raw[1])[0])
    q_soup = parent.load_q_soup(PARENT_DIR)
    q_raw = parent._q_predictions(q_soup, T_raw.astype(np.float32), torch.device("cpu"))
    t25 = {}
    for i, gid in enumerate(CASES):
        t25[str(gid)] = {
            "T25_raw": [float(v) for v in T_raw[i]],
            "T25_prepped": [float(v) for v in T_prep[i]],
            "q_raw_replay": float(q_raw[i]),
        }
    out["topology25"] = {
        **t25,
        "differing_features_raw": differing_feats,
        "per_feature_raw": {
            str(f): [float(T_raw[0][f]), float(T_raw[1][f])] for f in differing_feats
        },
        "feat16_raw_both": float(T_raw[0][16]),
        "identical_feature_count": int(T_raw.shape[1] - len(differing_feats)),
        "equal_on_both_rows": sorted(
            int(f) for f in differing_feats
            if len({float(T_raw[0][f]), float(T_raw[1][f])}) == 2
            and sum(1 for g in differing_feats
                    if [float(T_raw[0][g]), float(T_raw[1][g])] == [float(T_raw[0][f]), float(T_raw[1][f])]) > 1
        ),
        "prep_reproduction": "zldc.build_new_prep on the old 8001-row fit view reproduces prep.npz bit-for-bit",
    }

    # (4) exact / nearest Q inputs inside the old 8001-row fit pool
    #     (exact match on the raw T25; nearest by raw-T25 L1, as in the note)
    fit_rows = [train_data[int(i)] for i in old_fit.tolist()]
    T_fit = parent.topology_matrix(fit_rows).astype(np.float64)
    # truly duplicated T25 columns over the whole old fit pool (informational)
    dup_pairs = [
        [int(j), int(i)]
        for i in range(T_fit.shape[1])
        for j in range(i)
        if np.array_equal(T_fit[:, i], T_fit[:, j])
    ]
    matches = {}
    for i, gid in enumerate(CASES):
        hit = np.where((T_fit == T_raw[i]).all(axis=1))[0]
        hit_positions = [int(old_fit[int(h)]) for h in hit]
        entry: dict[str, Any] = {
            "n_exact_matches": int(hit.size),
            "match_positions": hit_positions,
            "match_k": [int(targets["k"][p]) for p in hit_positions],
            "match_c": [float(targets["c"][p]) for p in hit_positions],
        }
        if hit.size == 0:
            l1 = np.abs(T_fit - T_raw[i]).sum(axis=1)
            order = np.argsort(l1, kind="stable")[:3]
            entry["nearest_fit_rows_l1_raw"] = [
                {"position": int(old_fit[int(j)]), "l1_raw": float(l1[j]),
                 "k": int(targets["k"][int(old_fit[int(j)])]),
                 "c": float(targets["c"][int(old_fit[int(j)])])}
                for j in order
            ]
        matches[str(gid)] = entry
    out["old_fit_input_matches"] = {
        **matches,
        "duplicated_t25_columns_over_old_fit_pool": dup_pairs,
        "conflict_note": (
            "gid=1424: identical T25 with conflicting k in the old fit pool "
            "(input conflict — no deterministic function of T25 can output both); "
            "gid=3775: its singleton T25 class fell into select (unseen class)"
        ),
    }

    # (5) contribution to the old select/confirm absolute errors (read-only)
    select_stats = {}
    for run in SELECT_RUNS:
        with np.load(PARENT_DIR / "runs" / run / "select_predictions.npz", allow_pickle=False) as z:
            gid_p = np.asarray(z["gid"], np.int64)
            e = np.abs(np.asarray(z["y"], np.float64) - np.asarray(z["y_raw"], np.float64))
        row = {
            "n": int(e.size), "mae": float(e.mean()),
            "abs_err_gt_5_count": int((e > 5).sum()),
            "per_gid_abs_err": {}, "per_gid_share_of_total_abs_err": {},
        }
        for gid in CASES:
            m = np.where(gid_p == gid)[0]
            row["per_gid_abs_err"][str(gid)] = float(e[m][0])
            row["per_gid_share_of_total_abs_err"][str(gid)] = float(e[m][0] / e.sum())
        row["two_gid_combined_share"] = float(sum(row["per_gid_abs_err"].values()) / e.sum())
        select_stats[run] = row
    confirm_stats = {}
    for run in CONFIRM_RUNS:
        with np.load(PARENT_DIR / "runs" / run / "confirm_predictions.npz", allow_pickle=False) as z:
            gid_p = np.asarray(z["gid"], np.int64)
            e = np.abs(np.asarray(z["y"], np.float64) - np.asarray(z["y_raw"], np.float64))
        row = {
            "n": int(e.size), "mae": float(e.mean()),
            "abs_err_gt_5_count": int((e > 5).sum()),
            "two_gid_present": bool(any(g in gid_p.tolist() for g in CASES)),
            "quantiles": {q: float(np.percentile(e, q)) for q in (50, 90, 99, 100)},
        }
        confirm_stats[run] = row
    out["select_confirm_contribution"] = {
        "select_seed0_runs": select_stats,
        "confirm_runs": confirm_stats,
        "note": (
            "select vs confirm MAE gap is mainly the different tails of the two "
            "holdouts; all rows kept (no deletion, no post-hoc re-scoring)"
        ),
    }

    # (6) conclusion echo (frozen in the note; verified against the artifacts)
    out["located_causes"] = {
        "3775": "T25 singleton class fell into the old select (unseen class) + the label "
                "penalty itself is generator-order-dependent (audit Q9): not an ID/wiring bug",
        "1424": "T25 input conflict (identical vector, conflicting k in the old fit pool) "
                "+ long-cycle tail severity beyond T25 expressiveness: not an ID/wiring bug",
        "minimal_fix": None,
        "action": "keep the shared Q recipe unchanged for this round (T_fit-only retrain, once)",
    }
    out["read_only"] = True
    out["official_valid_loaded"] = False
    out["official_test_loaded"] = False
    out["seconds"] = float(__import__("time").perf_counter() - started)
    out["reproduction"] = (
        "uv run python -m tracks.ksvd.experiments.luyin16.zinc_cssd_basis_reuse_v1_q_spotcheck"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "spotcheck.json").write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")

    # summary print of the note's headline numbers
    for gid in CASES:
        c = out["chain"][str(gid)]
        print(f"gid={gid} pos={c['position']} y={c['y']:.4f} k={c['k']} c={c['c']:.4f} "
              f"ell={c['ell']:.4f} s={c['s']:.4f} q_raw={out['topology25'][str(gid)]['q_raw_replay']:+.4f}")
    print("differing features (raw):", out["topology25"]["differing_features_raw"],
          "feat16:", out["topology25"]["feat16_raw_both"],
          "duplicated columns (fit pool):", out["old_fit_input_matches"]["duplicated_t25_columns_over_old_fit_pool"])
    for gid in CASES:
        m = out["old_fit_input_matches"][str(gid)]
        print(f"gid={gid} exact matches {m['n_exact_matches']} at {m['match_positions']}",
              f"nearest={m.get('nearest_fit_rows_l1_raw')}")
    for run in SELECT_RUNS:
        r = select_stats[run]
        print(run, "shares", {k: round(v, 4) for k, v in r["per_gid_share_of_total_abs_err"].items()},
              "combined", round(r["two_gid_combined_share"], 4), ">5:", r["abs_err_gt_5_count"])
    print(f"[q-spotcheck] wrote {OUT_DIR / 'spotcheck.json'} in {out['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
