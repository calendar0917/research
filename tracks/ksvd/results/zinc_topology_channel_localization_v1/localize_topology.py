"""Frozen-Full global topology channel localization (read-only).

Three layers:
  A. topology25 input ambiguity for the 35 cycle-penalty graphs (train NN).
  B. encoder response / local Jacobian / reader gradient / composite g25.
  C. two frozen full-block replacements: train-mean input, matched donors.

No training, no optimizer step, no new features, no test access.  The only
fitted scalar is the model's existing train-median output bias (reused, never
refit per intervention).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
HANDOFF = REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"
CYCLE = REPO / "tracks/ksvd/results/zinc_long_cycle_audit"
TOPO_SLICE = slice(806, 814)
N_ATOM, N_BOND = 28, 4
TOL = 1e-5


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def reader_pred(R, rd):
    h1 = np.maximum(R @ rd["W1"].T + rd["b1"], 0.0)
    h2 = np.maximum(h1 @ rd["W2"].T + rd["b2"], 0.0)
    return (h2 @ rd["W3"].T + rd["b3"]).ravel()


def encode(t, rd):
    h1 = np.maximum(t @ rd["topology_W1"].T + rd["topology_b1"], 0.0)
    return (h1 @ rd["topology_W2"].T + rd["topology_b2"]), h1


def composition(node_ptr, atom_type, edge_ptr, edge_type, n):
    """Per-graph [n_nodes, atom frac(28), bond frac(4)] feature."""
    F = np.zeros((n, 1 + N_ATOM + N_BOND))
    for i in range(n):
        a = atom_type[node_ptr[i]:node_ptr[i + 1]]
        e = edge_type[edge_ptr[i]:edge_ptr[i + 1]]
        F[i, 0] = len(a)
        for k in range(N_ATOM):
            F[i, 1 + k] = np.sum(a == k) / max(len(a), 1)
        for k in range(N_BOND):
            F[i, 1 + N_ATOM + k] = np.sum(e == k) / max(len(e), 1)
    return F


def nearest(F_pool, F_q, k, ids_pool):
    """k nearest by euclidean on F; deterministic tie-break by pool order."""
    order = np.lexsort((ids_pool,))  # stable by id
    Fs, ids_s = F_pool[order], ids_pool[order]
    idx = []
    for q in F_q:
        d = np.linalg.norm(Fs - q, axis=1)
        idx.append(order[np.argsort(d, kind="stable")[:k]])
    return idx


def main():
    v = np.load(HANDOFF / "valid.npz", allow_pickle=True)
    t = np.load(HANDOFF / "train.npz", allow_pickle=True)
    rd = {k: np.load(HANDOFF / "reader.npz")[k].astype(np.float64)
          for k in np.load(HANDOFF / "reader.npz").files}
    R = v["R"].astype(np.float64); y = v["y"]; ids = v["ids"]
    T25 = v["topo_model_input"].astype(np.float64)
    T25tr = t["topo_model_input"].astype(np.float64)
    ytr = t["y"]; idtr = t["ids"]

    # ---- identity ----
    Tenc, H1 = encode(T25, rd)
    ident = {
        "encoder_replay_vs_R_topology_block_max_abs_diff":
            float(np.max(np.abs(Tenc - R[:, TOPO_SLICE]))),
        "reader_replay_valid_raw_mae": mae(reader_pred(R, rd), y),
        "published_raw_valid_mae": 0.1191540920053958,
        "p_base_max_abs_diff": float(np.max(np.abs(reader_pred(R, rd) - v["p_base"]))),
    }
    ident["encoder_identity_ok"] = ident["encoder_replay_vs_R_topology_block_max_abs_diff"] < TOL
    ident["reader_identity_ok"] = ident["reader_replay_valid_raw_mae"] - 0.1191540920053958 < 1e-5
    if not (ident["encoder_identity_ok"] and ident["reader_identity_ok"]):
        print("REPLAY_MISMATCH", ident)

    pred_native_raw = reader_pred(R, rd)
    b = float(np.median(ytr - reader_pred(t["R"].astype(np.float64), rd)))
    base_cal = pred_native_raw + b

    # ---- groups ----
    lab = pd.read_csv(CYCLE / "valid_cycle_audit_label.csv")
    assert np.array_equal(lab["subset_index"].to_numpy(), ids)
    assert np.max(np.abs(lab["target"].to_numpy() - y)) < TOL
    lc = lab["label_effective_cycle_snapped"].to_numpy()
    g172 = ids == 172
    g1 = (lc < 0) & ~g172
    g0 = (lc == 0) & ~g172
    cycle_mask = g1 | g172
    cyc_idx = np.where(cycle_mask)[0]

    def grp_rows(pred_cal):
        rows = {}
        abs_err = np.abs(pred_cal - y)
        for name, m in (("G0", g0), ("G1", g1), ("G172", g172)):
            rows[name] = {"n": int(m.sum()), "mae": mae(pred_cal[m], y[m]),
                          "contribution": float(abs_err[m].sum() / len(y)),
                          "signed_mean": float((pred_cal[m] - y[m]).mean())}
        rows["overall"] = {"mae": mae(pred_cal, y), "contribution": float(abs_err.mean())}
        return rows

    native_groups = grp_rows(base_cal)

    # ================= A. topology25 ambiguity =================
    Ftr = composition(t["node_ptr"], t["atom_type"], t["edge_ptr"], t["edge_type"], len(idtr))
    Fva = composition(v["node_ptr"], v["atom_type"], v["edge_ptr"], v["edge_type"], len(ids))
    ltr = np.zeros(len(idtr)); ltr_pen = None
    labt = pd.read_csv(CYCLE / "train_cycle_audit_label.csv")
    ltr_pen = labt["label_effective_cycle_snapped"].to_numpy()
    nn_idx = nearest(T25tr, T25[cyc_idx], 5, idtr)
    A = []
    for qi, gi in enumerate(cyc_idx):
        nb = nn_idx[qi]
        A.append({
            "valid_index": int(gi), "molecule_id": str(lab["molecule_id"].iloc[gi]),
            "label_penalty": float(lc[gi]),
            "nn_train_ids": idtr[nb].tolist(),
            "nn_label_penalty": ltr_pen[nb].tolist(),
            "nn_topology25_dist": np.linalg.norm(T25tr[nb] - T25[gi], axis=1).tolist(),
            "same_dist_penalty_penalty": [float(lc[gi])] * 5,
        })
    # ambiguity summary: exact/near-identical topology25 pairs
    A_sum = {
        "n_cycle_graphs": int(len(cyc_idx)),
        "nn_with_penalty_lt0": int(sum(any(p < 0 for p in a["nn_label_penalty"]) for a in A)),
        "nn_all_zero_penalty": int(sum(all(p == 0 for p in a["nn_label_penalty"]) for a in A)),
        "min_nn_dist_overall": float(min(min(a["nn_topology25_dist"]) for a in A)),
        "median_nn_dist": float(np.median([a["nn_topology25_dist"][0] for a in A])),
        "id172_nn": next(a for a in A if a["valid_index"] == 172),
    }

    # ================= B. encoding / reader local response =================
    # encoder unit activation over cycle graphs + all valid
    h1_all = np.maximum(T25 @ rd["topology_W1"].T + rd["topology_b1"], 0.0)
    dead_units = np.where(np.all(h1_all == 0, axis=0))[0].tolist()
    h1_cyc = H1[cyc_idx]
    B = {
        "encoder_dead_units_over_valid": dead_units,
        "n_dead_units": len(dead_units),
        "encoder_h1_std_over_valid": np.std(h1_all, axis=0).tolist(),
        "topology8_std_over_valid": np.std(Tenc, axis=0).tolist(),
        "topology8_std_over_cycle": np.std(Tenc[cyc_idx], axis=0).tolist(),
    }
    # Jacobians at cycle graphs
    W1t, b1t = rd["topology_W1"], rd["topology_b1"]
    W2t = rd["topology_W2"]
    W1r, b1r, W2r, b2r, W3r, b3r = (rd["W1"], rd["b1"], rd["W2"], rd["b2"], rd["W3"], rd["b3"])
    def g8_at(tv, Rv):
        h1 = np.maximum(Rv @ W1r.T + b1r, 0.0)
        h2 = np.maximum(h1 @ W2r.T + b2r, 0.0)
        m2 = (h2 > 0).astype(float)
        m1 = (h1 > 0).astype(float)
        # d pred/dR = ((W3 * m2) @ W2r) * m1 @ W1r  -> chain
        g2 = (W3r * m2)                 # [1,39]
        g1 = (g2 @ W2r) * m1            # [1,39]
        return (g1 @ W1r).ravel()       # [814]
    jac = []
    for gi in cyc_idx:
        act = (T25[gi] @ W1t.T + b1t) > 0
        JT = (W2t * act.astype(float)) @ W1t        # [8,25]
        g8 = g8_at(T25[gi], R[gi])[TOPO_SLICE]       # [8]
        g25 = JT.T @ g8                              # [25]
        jac.append({"valid_index": int(gi), "encoder_units_active": int(act.sum()),
                    "J_T_fro": float(np.linalg.norm(JT)), "g8_norm": float(np.linalg.norm(g8)),
                    "g25_norm": float(np.linalg.norm(g25)),
                    "g25_absmax": float(np.abs(g25).max())})
    B["cycle_jacobians"] = jac
    B["mean_encoder_units_active_cycle"] = float(np.mean([j["encoder_units_active"] for j in jac]))
    B["mean_JT_fro_cycle"] = float(np.mean([j["J_T_fro"] for j in jac]))
    B["mean_g8_norm_cycle"] = float(np.mean([j["g8_norm"] for j in jac]))
    B["mean_g25_norm_cycle"] = float(np.mean([j["g25_norm"] for j in jac]))
    # same for G0 comparison
    jac0 = []
    for gi in np.where(g0)[0]:
        act = (T25[gi] @ W1t.T + b1t) > 0
        JT = (W2t * act.astype(float)) @ W1t
        g8 = g8_at(T25[gi], R[gi])[TOPO_SLICE]
        jac0.append((int(act.sum()), float(np.linalg.norm(JT)), float(np.linalg.norm(g8)),
                     float(np.linalg.norm(JT.T @ g8))))
    jac0 = np.array(jac0, float)
    B["mean_encoder_units_active_G0"] = float(jac0[:, 0].mean())
    B["mean_JT_fro_G0"] = float(jac0[:, 1].mean())
    B["mean_g8_norm_G0"] = float(jac0[:, 2].mean())
    B["mean_g25_norm_G0"] = float(jac0[:, 3].mean())

    # ================= C. block replacements =================
    t_mean = T25tr.mean(0)
    Tmean_enc, _ = encode(np.tile(t_mean, (len(ids), 1)), rd)
    Rm = R.copy(); Rm[:, TOPO_SLICE] = Tmean_enc
    pred_mean = reader_pred(Rm, rd) + b
    C = {"mean_input_norm": float(np.linalg.norm(t_mean)),
         "mean_input_absmax": float(np.abs(t_mean).max()),
         "mean_replacement_all": grp_rows(pred_mean)}

    # donor matching: small G0 control set = nearest valid G0 for the 35 cycle graphs
    g0_idx = np.where(g0)[0]
    nn0 = nearest(Fva[g0_idx], Fva[cyc_idx], 2, ids[g0_idx])
    ctrl = sorted(set(int(x) for nb in nn0 for x in nb))
    FtrG0 = np.where(ltr_pen == 0)[0]
    FtrG1 = np.where(ltr_pen < 0)[0]
    donors = {}
    for pool_name, pool in (("G0", FtrG0), ("G1", FtrG1)):
        donors[pool_name] = nearest(Ftr[pool], Fva[cyc_idx], 3, idtr[pool])

    tgt = list(cyc_idx) + ctrl
    delta_rows = []
    for qi, gi in enumerate(tgt):
        native = base_cal[gi]
        for pool_name in ("G0", "G1"):
            for dpos, dj in enumerate(donors[pool_name][qi] if qi < len(cyc_idx) else nearest(
                    Ftr[FtrG0 if pool_name == "G0" else FtrG1], Fva[[gi]], 3,
                    idtr[FtrG0 if pool_name == "G0" else FtrG1])[0]):
                Rr = R.copy()
                Td, _ = encode(T25tr[dj][None, :], rd)
                Rr[:, TOPO_SLICE] = np.tile(Td, (len(ids), 1))
                pred = reader_pred(Rr, rd) + b
                delta_rows.append({"target_valid_index": int(gi), "target_role":
                                   "cycle" if qi < len(cyc_idx) else "control_G0",
                                   "donor_pool": pool_name, "donor_train_id": int(idtr[dj]),
                                   "donor_penalty": float(ltr_pen[dj]),
                                   "delta_pred": float(pred[gi] - native),
                                   "native_signed": float(native - y[gi]),
                                   "intervention_signed": float(pred[gi] - y[gi]),
                                   "native_abs": float(abs(native - y[gi])),
                                   "intervention_abs": float(abs(pred[gi] - y[gi])),
                                   "gain": float(abs(native - y[gi]) - abs(pred[gi] - y[gi]))})
    D = pd.DataFrame(delta_rows)
    donor_summary = {p: {
        "n_rows": int((D["donor_pool"] == p).sum()),
        "mean_delta_pred_cycle": float(D[(D.donor_pool == p) & (D.target_role == "cycle")]["delta_pred"].mean()),
        "mean_delta_pred_control": float(D[(D.donor_pool == p) & (D.target_role == "control_G0")]["delta_pred"].mean()),
        "mean_gain_cycle": float(D[(D.donor_pool == p) & (D.target_role == "cycle")]["gain"].mean()),
        "mean_gain_control": float(D[(D.donor_pool == p) & (D.target_role == "control_G0")]["gain"].mean()),
        "std_delta_pred_cycle": float(D[(D.donor_pool == p) & (D.target_role == "cycle")]["delta_pred"].std()),
    } for p in ("G0", "G1")}

    out = {"identity": ident, "bias_train_median": b, "native_groups": native_groups,
           "A_ambiguity": A_sum, "A_rows": A, "B_local_response": B,
           "C_mean": C, "C_donor": donor_summary,
           "n_control_G0": len(ctrl), "control_valid_indices": ctrl,
           "donor_pool_sizes": {"G0": int(len(FtrG0)), "G1": int(len(FtrG1))},
           "official_test_loaded": False}
    (OUT / "localization.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(A).to_csv(OUT / "nn_ambiguity.csv", index=False)
    D.to_csv(OUT / "donor_interventions.csv", index=False)
    pd.DataFrame(jac).to_csv(OUT / "cycle_jacobians.csv", index=False)
    print(json.dumps({k: out[k] for k in ("identity", "native_groups", "A_ambiguity",
                                          "C_mean", "C_donor", "n_control_G0",
                                          "donor_pool_sizes")}, indent=1, default=float))
    print("B:", json.dumps({k: B[k] for k in B if k != "cycle_jacobians"}, default=float))


if __name__ == "__main__":
    main()