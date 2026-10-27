"""Zero-GPU audit backing ``ERRATA.md`` for ``zinc_adaptive_arch_search_v1``.

A. SSL-seed protocol: show that the *DCC-L / P3* training path serves seeds >= 1
   from ``pretrain_encoder_s1.pt`` (so 4+ supervised seeds were always runnable)
   while the *BOSI* path used in AAS v1 indexes the encoder file by seed and
   therefore raises for seed >= 2.  Evidence is read from the recorded
   ``train.json`` files of the round that actually did it.
B. Variance decomposition of the "seed floor": encoder identity vs supervised
   seed vs numeric drift vs fixed-valid row leverage, using recorded runs only.
C. BOSI-B moment algebra: M[:, r] = (n_r / max(n_partners, 1)) * S with
   S = sum over ALL partners of chebyshev(x), verified (i) against a brute-force
   loop and (ii) as a rank-1-across-classes identity with information content
   (S, n) rather than 384 free numbers.
D. SQBC S1/S2 audit: enumerate what S2 changes beyond the pooling position.

Writes ``analysis/errata_audit.json``.  No GPU, no training, and official test is
never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_bosi_architecture_v0" / "code"),
           str(_SS / "zinc_bosi_joint_moment_v1" / "code"),
           str(_SS / "zinc_arch_discovery_v1" / "code"),
           str(_SS / "zinc_performance_search_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = _SS / "zinc_adaptive_arch_search_v1" / "analysis" / "errata_audit.json"


# ---------------------------------------------------------------------------
# A. SSL seed protocol
# ---------------------------------------------------------------------------
def audit_ssl_seeds() -> dict:
    records = json.loads((_SS / "zinc_performance_search_v1" / "analysis"
                          / "main_table.json").read_text())["records"]
    rows = []
    for r in records:
        ssl = (r.get("ssl") or {}).get("source") or ""
        rows.append({"arm": r["arm"], "recipe": r["recipe"], "seed": int(r["seed"]),
                     "best_valid": float(r["best_valid"]),
                     "ssl_file": Path(ssl).name})
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    # what the two code paths do with seed >= 2
    import bosi_common as BC
    code = {
        "bosi_common.ssl_path(2)": str(BC.ssl_path(2).name),
        "bosi_common.ssl_path(1)": str(BC.ssl_path(1).name),
        "bosi_common.ssl_path(0)": str(BC.ssl_path(0).name),
    }
    dcc_src = (_SS / "zinc_dual_channel_composer_v1" / "code" / "dcc_models.py").read_text()
    ad_src = (_SS / "zinc_arch_discovery_v1" / "code" / "ad_models.py").read_text()
    convention = {
        "dcc_models": "P3_SSL_S0 if int(seed) == 0 else P3_SSL_S1" in dcc_src,
        "ad_models": "P3_SSL_S0 if int(seed) == 0 else P3_SSL_S1" in ad_src,
    }
    s0_only = sorted({json.dumps(r) for r in rows if r["seed"] == 0})
    return {
        "recorded_runs": rows,
        "runs_with_seed_ge_2": [r for r in rows if r["seed"] >= 2],
        "s1_encoder_users": sorted({r["arm"] for r in rows
                                    if r["ssl_file"] == "pretrain_encoder_s1.pt"}),
        "bosi_ssl_path_for_seeds": code,
        "dcc_ad_convention_seed0_else_s1": convention,
        "verdict": ("seeds >= 2 WERE runnable and were actually run for both A (P3Null) "
                    "and B (DCC-L) by reusing pretrain_encoder_s1.pt; the AAS v1 claim that "
                    "'only two SSL encoders exist, so two supervised seeds is the hard "
                    "ceiling' is wrong. The BOSI path's seed-indexed ssl_path() is the only "
                    "reason seed 2 failed inside that round."),
    }


# ---------------------------------------------------------------------------
# B. variance decomposition
# ---------------------------------------------------------------------------
def audit_variance() -> dict:
    ps = json.loads((_SS / "zinc_performance_search_v1" / "analysis"
                     / "main_table.json").read_text())["records"]
    fam = {}
    for r in ps:
        if r["recipe"] != "R0":
            continue
        fam.setdefault(r["arm"], {})[int(r["seed"])] = float(r["best_valid"])
    out = {}
    for arm, d in fam.items():
        s0 = d[0]
        s1 = [v for k, v in d.items() if k >= 1]
        out[arm] = {
            "seed0_using_s0_encoder": s0,
            "seeds_ge1_using_s1_encoder": s1,
            "encoder_effect_seed0_minus_s1mean": float(s0 - np.mean(s1)),
            "within_encoder_supervised_spread": float(max(s1) - min(s1)),
            "n_within": len(s1),
        }
    # AAS v1 runs (own round): same-seed pairs, both from this round
    aas = {}
    for tag in ("B_s0", "B_s1", "A0_s0", "A0_s1", "A1_s0", "A1_s1", "S1_s0", "S2_s0"):
        p = _SS / "zinc_adaptive_arch_search_v1" / "runs" / tag / "train.json"
        if p.exists():
            rec = json.loads(p.read_text())
            best = rec.get("best") or rec.get("best_valid") or {}
            aas[tag] = {"best_valid": float(best["valid_mae"]),
                        "init": (rec.get("initial") or {}).get("initial_state_sha256", "")[:12]}
    # numeric drift: the only same-code duplicate we own (S1 seed 0, two runs)
    drift = {"pair": ["S1_s0 (first run, killed by the train.json bookkeeping bug)",
                      "S1_s0 (rerun)"],
             "values": [0.08688874, aas.get("S1_s0", {}).get("best_valid")],
             "delta": None}
    if drift["values"][1] is not None:
        drift["delta"] = float(drift["values"][1] - drift["values"][0])

    # fixed-valid row leverage (from the frozen predictions of this round)
    lev = {}
    y = None
    try:
        import ad_data as D
        y = np.asarray(D.load_cache("valid")["labels"], dtype=np.float64)
    except Exception:
        pass
    if y is not None:
        for tag in list(aas) + ["B_s0"]:
            p = _SS / "zinc_adaptive_arch_search_v1" / "runs" / tag / "valid_best.npz"
            if not p.exists():
                continue
            ae = np.abs(np.asarray(np.load(p)["pred"], dtype=np.float64) - y)
            lev[tag] = {"row172_ae": float(ae[172]),
                        "mean_ae": float(ae.mean()),
                        "mean_ae_without_row172": float(np.delete(ae, 172).mean()),
                        "top10_share": float(np.sort(ae)[-10:].sum() / ae.sum())}
    # second, cleaner drift measurement: pilot re-run of the frozen E anchor
    pilot = _SS / "zinc_support_weighted_objective_v1" / "runs" / "E_s0" / "train.json"
    anchor = _SS / "zinc_dual_channel_composer_v1" / "runs" / "L_s0" / "artifacts.json"
    drift2 = {"available": False}
    if pilot.exists():
        rec = json.loads(pilot.read_text())
        art = json.loads(anchor.read_text())
        fresh = float(rec["best_valid"]["valid_mae"])
        hist = float(art["best_valid"]["valid_mae"])
        drift2 = {
            "available": True,
            "fresh_E_s0": fresh, "historical_L_s0": hist,
            "delta": float(fresh - hist),
            "init_hashes": {"fresh": rec["init"]["initial_state_sha256"],
                            "historical": art["initial_state_sha256"]},
            "init_bit_identical": bool(rec["init"]["initial_state_sha256"]
                                       == art["initial_state_sha256"]),
            "epochs": [int(rec["best_valid"]["epoch"]), int(art["best_valid"]["epoch"])],
            "note": ("same seed, same SSL encoder, same initial state and recipe, "
                     "different execution: the residual is GPU-side non-determinism"),
        }
    return {
        "population_runs_B_R0_and_A_R0": out,
        "drift_rerun_of_frozen_E_anchor": drift2,
        "aas_v1_own_runs": aas,
        "numeric_drift_same_code_duplicate": drift,
        "fixed_valid_row_leverage": lev,
        "reading": {
            "encoder_identity": "for P3Null the s0-encoder seed-0 run is 0.0060 better than all three s1-encoder seeds; for DCC-L the s1-encoder seeds are 0.0034 better than the s0-encoder seed. The sign is body-dependent, so 'encoder effect' is an interaction, not an additive offset.",
            "supervised_seed_within_one_encoder": "range over three s1-encoder seeds: P3Null 0.00055, DCC-L 0.00307.",
            "numeric_drift": "one same-code duplicate pair (S1 seed 0) differs by the value below; that is the only direct estimate we have and it should be treated as an upper bound on pure numeric non-determinism, not a calibrated sd.",
            "fixed_valid_leverage": "valid is a fixed 1000-row set: row 172 alone moves the mean by up to 0.0086 and the fixed worst-10 rows by up to 0.0076 between arms, so a paired 1000-row bootstrap CI reflects row heterogeneity (a property of the evaluation), not training-seed noise.",
        },
    }


# ---------------------------------------------------------------------------
# C. BOSI-B moment algebra
# ---------------------------------------------------------------------------
def audit_moment_algebra() -> dict:
    import torch
    import bosi_data as BD
    import bosi_models as BM
    import ad_data as D

    norm = D.load_norm()
    model, info = BM.build_model(norm, arm="B", seed=0, device="cpu")
    model.eval()
    cache = D.load_cache("train")
    import bjm_common as BJM
    rel = BJM.load_rel("train")
    batch = BD.collate(cache, rel, np.arange(64))
    with torch.no_grad():
        a, b = model.encode_tokens(batch)
        x = model.half_edges(batch, a, b)
        M, n = model.relation_stats(batch, x)

    vals = BM.chebyshev(x)
    tgt, part, relc = batch["partner_tgt"], batch["partner_half"], batch["partner_rel"]
    H = int(x.shape[0])
    R, Dm = BM.N_REL, int(vals.shape[1])
    # brute force
    M_bf = torch.zeros(H, R * Dm, dtype=x.dtype)
    n_bf = torch.zeros(H, R, dtype=x.dtype)
    S_bf = torch.zeros(H, Dm, dtype=x.dtype)
    for h in range(H):
        mask = tgt == h
        for f_idx in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
            S_bf[h] += vals[part[f_idx]]
            M_bf[h, int(relc[f_idx]) * Dm:(int(relc[f_idx]) + 1) * Dm] += vals[part[f_idx]]
            n_bf[h, int(relc[f_idx])] += 1.0
    denom = (batch["half_deg"] - 1.0).clamp(min=1.0).view(-1, 1)
    M_ref = ((n_bf.unsqueeze(-1) / denom.unsqueeze(-1)) * S_bf.unsqueeze(1)).reshape(H, R * Dm)

    # rank-1 across classes: M[:, r] / n_r must be the same vector for every r
    with torch.no_grad():
        blocks = M.view(H, R, Dm)
        # pairwise identity check: on halves where classes r and s both occur,
        # M[:,r]/n_r must equal M[:,s]/n_s exactly (rank-1 across classes).
        pairs = {}
        worst = 0.0
        for r in range(R):
            for sp in range(r + 1, R):
                m2 = (n[:, r] > 0) & (n[:, sp] > 0)
                k = int(m2.sum())
                d = float("nan")
                if k > 0:
                    d = float((blocks[m2, r] / n[m2, r].unsqueeze(-1)
                               - blocks[m2, sp] / n[m2, sp].unsqueeze(-1)).abs().max())
                    worst = max(worst, d)
                pairs[f"class{r}_vs_class{sp}"] = {"n_halves": k, "max_abs_dev": d}
        ratio_spread = worst
        n_all_classes = max(v["n_halves"] for v in pairs.values())
        # single-class halves: n_1 == denom, so M_1 == S exactly
        single = (n > 0).sum(1) == 1
        k_single = int(single.sum())
        # recover S from (M, n) and compare with the true total
        r0 = 0
        m0 = n[:, 0] > 0
        S_rec = torch.zeros_like(S_bf)
        S_rec[m0] = blocks[m0, 0] * denom[m0] / n[m0, 0].unsqueeze(-1)
        rec_err = float((S_rec[m0] - S_bf[m0]).abs().max())
        # how many *distinct* numbers does the block carry?
        eff = {"block_dims": int(R * Dm), "distinct_dims": int(Dm),
               "class_counts": int(R)}
        out = {
            "N_REL": int(R), "chebyshev_dim": Dm, "M_shape": list(M.shape),
            "max_abs_diff_vectorized_vs_bruteforce": float((M - M_ref).abs().max()),
            "max_abs_diff_M_vs_bf_sums_expected_nonzero": float((M - M_bf).abs().max()),
            "class_ratio_max_dev": ratio_spread,
            "pairwise_identity": pairs,
            "n_halves_for_best_pair": n_all_classes,
            "n_single_class_halves": k_single,
            "recover_S_from_M_and_n_max_err": rec_err,
            "effective_numbers": eff,
            "n_partner_counts": [int((n[:, r] > 0).sum()) for r in range(R)],
            "verdict_formula": "M[v, r] = (n_r[v] / max(n_partners[v], 1)) * S[v], S[v] = sum over ALL partners of chebyshev(x_partner)",
            "pass": bool(float((M - M_ref).abs().max()) < 1e-5 and ratio_spread < 1e-4),
        }
    return out


# ---------------------------------------------------------------------------
# D. SQBC S1/S2 audit
# ---------------------------------------------------------------------------
def audit_sqbc() -> dict:
    src = (_SS / "zinc_adaptive_arch_search_v1" / "code" / "aas_sqbc.py").read_text()
    return {
        "s1_statistic": "s1 = P_u + P_v, s2 = |P_u - P_v|, s3 = P_u * P_v, where P[v,e] = sum over the centre's OTHER incident bonds of Phi(pair)",
        "s2_statistic": "s1 = mean_g(pair content), s2 = mean_g(pair content^2), s3 = s1 * s2, i.e. a per-graph constant broadcast to every bond",
        "changes_beyond_pooling_position": [
            "slot 2 changes KIND: S1 uses |P_u - P_v| (an endpoint contrast), S2 uses mean_g(w^2) (a dispersion statistic)",
            "per-bond locality is destroyed: in S2 all three 32-d slots are constant within a molecule, so pair content can no longer modulate individual bonds",
            "normalisation changes: S1 sums over (n_v - 1) partners and therefore encodes local multiplicity through magnitude; S2 averages over all pair rows in the graph (degree-free)",
            "count information changes: S1's sum magnitude is a degree signal; S2 discards the number of pair rows entirely",
            "effective conditioning changes: S2 reduces the composer's inputs to a global context vector plus bond features, i.e. it is closer to a P3Null-style global-context body than to a 'local vs global pooling' control",
        ],
        "therefore": ("S1 - S2 = -0.0107 supports only: 'for this pair content, replacing the "
                      "local per-bond statistic by a graph-level average costs 0.011'. It does "
                      "not test direction C (global/block structure statistics), which would "
                      "need graph-level statistics of junction/block OBJECTS with the local "
                      "channels kept and capacity matched."),
        "code_markers_found": {
            "s2_branch": "self.pooling = \"global\" if mode == \"S2\" else \"local\"" in src,
            "w1_mean": "w1 = a.new_zeros((Bn0, RANK)).index_add_(0, pg, w_ef)" in src,
            "w2_meansq": "w2 = a.new_zeros((Bn0, RANK)).index_add_(0, pg, w_ef * w_ef)" in src,
        },
    }


def main() -> int:
    out = {"written": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
           "round": "zinc_adaptive_arch_search_v1 errata",
           "official_test_loaded": False}
    out["A_ssl_seed_protocol"] = audit_ssl_seeds()
    out["B_variance_decomposition"] = audit_variance()
    out["C_bosi_moment_algebra"] = audit_moment_algebra()
    out["D_sqbc_s1_s2_audit"] = audit_sqbc()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {OUT}")
    print("C pass:", out["C_bosi_moment_algebra"]["pass"],
          "| max|Δ|:", out["C_bosi_moment_algebra"]["max_abs_diff_vectorized_vs_bruteforce"],
          "| ratio dev:", out["C_bosi_moment_algebra"]["class_ratio_max_dev"])
    print("A seeds>=2:", [r["seed"] for r in out["A_ssl_seed_protocol"]["runs_with_seed_ge_2"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
