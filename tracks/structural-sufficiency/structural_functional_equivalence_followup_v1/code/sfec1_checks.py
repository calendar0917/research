"""Engineering checks for SFEC v1 corrections and strict masking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C
import sfec1_data as D


def _make_design(slists: list[list[int]], ylists: list[list[float]],
                 qlist: list[int]) -> D.Design:
    off = [0]
    O_parts, y_parts, q_parts, pos_parts, k_parts = [], [], [], [], []
    pos = 0
    for S, y, q in zip(slists, ylists, qlist):
        S = np.asarray(S, dtype=np.int64)
        y = np.asarray(y, dtype=np.float64)
        O_parts.append(S)
        y_parts.append(y)
        q_parts.append(np.full(len(S), int(q), dtype=np.int64))
        pos_parts.append(np.arange(pos, pos + len(S), dtype=np.int64))
        pos += len(S)
        k_parts.append(len(S))
        off.append(pos)
    n = len(slists)
    return D.Design(
        ctx_idx=np.arange(n, dtype=np.int64),
        off=np.asarray(off, dtype=np.int64),
        O=np.concatenate(O_parts) if O_parts else np.zeros(0, np.int64),
        y=np.concatenate(y_parts) if y_parts else np.zeros(0, np.float64),
        q=np.concatenate(q_parts) if q_parts else np.zeros(0, np.int64),
        k=np.asarray(k_parts, dtype=np.int64),
        cell_pos=np.concatenate(pos_parts) if pos_parts else np.zeros(0, np.int64),
    )


def _make_contexts(slists: list[list[int]], ylists: list[list[float]]) -> D.Contexts:
    k = np.asarray([len(x) for x in slists], dtype=np.int32)
    off = np.r_[0, np.cumsum(k)].astype(np.int64)
    S = np.concatenate([np.asarray(x, dtype=np.int64) for x in slists])
    y = np.concatenate([np.asarray(x, dtype=np.float64) for x in ylists])
    n_cells = len(S)
    return D.Contexts(
        C=np.arange(len(slists), dtype=np.int32), off=off, k=k, S=S.astype(np.int32),
        y=y, n=np.ones(n_cells, dtype=np.int32), sd=np.zeros(n_cells),
        cell_key=np.arange(n_cells, dtype=np.int64),
        cell_C=np.repeat(np.arange(len(slists)), k).astype(np.int32),
        cell_S=S.astype(np.int32), cell_b=np.ones(n_cells, np.int8),
        cell_mol_first=np.arange(n_cells, dtype=np.int32),
        cell_mol_n=np.ones(n_cells, dtype=np.int32),
    )


def _control_additive() -> dict[str, Any]:
    b_true = np.array([0.4, -0.2, 0.7, 0.1, -0.5, 0.3])
    a_true = np.array([1.0, -0.3, 0.2, 0.8])
    S = [[0, 1, 2, 3], [0, 2, 4, 5], [1, 3, 4, 5], [0, 1, 4, 5]]
    y = [[a_true[c] + b_true[s] for s in s_list] for c, s_list in enumerate(S)]
    fit = D.fit_shared(_make_design(S, y, [0] * len(S)))
    err = 0.0
    for s_list in S:
        for i in s_list:
            for j in s_list:
                err = max(err, abs((fit["b"][j] - fit["b"][i]) - (b_true[j] - b_true[i])))
    return {"name": "A_additive_shared", "ok": bool(err <= 1e-8),
            "max_b_difference_error": float(err), "grad_max_abs": fit["grad_max_abs"]}


def _control_crossed() -> dict[str, Any]:
    b_true = np.array([0.3, -0.4, 0.2, 0.8, -0.1, 0.5, -0.6, 0.0])
    a_true = np.array([0.5, -0.2, 0.9, -0.7, 0.1, 0.3])
    q_true = np.array([0, 1, 0, 1, 0, 1])
    d_true = np.array([
        [0.0, 0.0, 0.1, -0.2, 0.05, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.15, 0.25, 0.0, 0.1, 0.0, 0.0],
    ])
    S = [[0, 1, 2, 3, 4], [0, 2, 3, 5, 6], [1, 2, 4, 6, 7],
         [0, 3, 4, 5, 7], [1, 2, 3, 5, 6], [0, 1, 4, 6, 7]]
    y = [[a_true[c] + b_true[s] + d_true[q_true[c], s] for s in s_list]
         for c, s_list in enumerate(S)]
    design = _make_design(S, y, q_true.tolist())
    base = D.fit_shared(design)
    local = D.fit_local_ctx(design, base["b"], n_q=2, lam=5.0, min_contexts=1)
    sse_s = sse_l = 0.0
    for c, s_list in enumerate(S):
        ss = np.asarray(s_list)
        yy = np.asarray(y[c])
        yt = yy - yy.mean()
        shared = base["b"][ss] - base["b"][ss].mean()
        q = int(q_true[c])
        loc = shared + local["d"][q, ss] - local["d"][q, ss].mean()
        sse_s += float(np.sum((yt - shared) ** 2))
        sse_l += float(np.sum((yt - loc) ** 2))
    return {"name": "B_crossed_local", "ok": bool(sse_l < sse_s - 1e-8),
            "shared_sse": sse_s, "local_sse": sse_l,
            "relative_improvement": float(1.0 - sse_l / max(sse_s, 1e-12))}


def _control_strict_mask() -> dict[str, Any]:
    b_true = np.array([0.5, -0.3, 0.8, 0.1, -0.2])
    a_true = np.array([1.0, 0.2, -0.4, 0.7, -0.1, 0.3, 0.5])
    S = [[0, 2, 3], [0, 2, 4], [0, 1, 3], [1, 2, 3],
         [1, 2, 4], [0, 3, 4], [2, 3, 4]]
    y = [[a_true[c] + b_true[s] for s in s_list] for c, s_list in enumerate(S)]
    design = _make_design(S, y, [0] * len(S))
    # delete B=2 from the two direct 0-2 contexts only
    keep = np.ones(len(design.O), dtype=bool)
    for ci in (0, 1):
        a, b = int(design.off[ci]), int(design.off[ci + 1])
        pos = a + int(np.flatnonzero(design.O[a:b] == 2)[0])
        keep[pos] = False
    fit = D.fit_shared(design, keep=keep)
    truth = b_true[2] - b_true[0]
    est = fit["b"][2] - fit["b"][0]
    return {"name": "C_strict_mask_indirect", "ok": bool(abs(est - truth) <= 1e-8
                                                          and fit["comp"][0] == fit["comp"][2]),
            "truth_delta": float(truth), "estimated_delta": float(est),
            "error": float(abs(est - truth))}


def _control_unidentifiable() -> dict[str, Any]:
    b_true = np.array([0.5, -0.3, 0.8, 0.1])
    S = [[0, 1], [0, 1], [2, 3], [2, 3]]
    y = [[b_true[s] for s in s_list] for s_list in S]
    fit = D.fit_shared(_make_design(S, y, [0] * len(S)))
    return {"name": "D_unidentifiable_components",
            "ok": bool(fit["comp"][0] != fit["comp"][2] and fit["comp"][0] >= 0 and fit["comp"][2] >= 0),
            "component_A": int(fit["comp"][0]), "component_B": int(fit["comp"][2])}


def _canonical_shuffle_check(b: D.Bundle) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(b.ctx_disc.k >= 2)
    pick = rng.choice(idx, size=min(500, len(idx)), replace=False)
    ok = True
    checked = 0
    for ci in pick.tolist():
        lo, hi = int(b.ctx_disc.off[ci]), int(b.ctx_disc.off[ci + 1])
        S = b.ctx_disc.S[lo:hi]
        y = b.ctx_disc.y[lo:hi]
        r = b.rank[S].astype(np.int64)
        # canonical reference from a one-context table
        ctx = _make_contexts([r.tolist()], [y.tolist()])
        ref = D.build_pairs(ctx, np.arange(int(r.max()) + 1), canonical=True)
        order = rng.permutation(len(r))
        ctx2 = _make_contexts([r[order].tolist()], [y[order].tolist()])
        got = D.build_pairs(ctx2, np.arange(int(r.max()) + 1), canonical=True)
        a1 = sorted(zip(ref["A"].tolist(), ref["B"].tolist(), ref["delta"].round(12).tolist()))
        a2 = sorted(zip(got["A"].tolist(), got["B"].tolist(), got["delta"].round(12).tolist()))
        ok = ok and (a1 == a2)
        checked += 1
    return {"ok": bool(ok), "contexts_checked": checked,
            "note": "cell order inside each context was randomly permuted; canonical pairs and deltas are unchanged"}


def _identity_row_check(b: D.Bundle, n: int = 200) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n_mol = len(b.M["mol_id"])
    pick = np.sort(rng.choice(n_mol, size=min(n, n_mol), replace=False))
    row_ids = b.M["row_id"][pick].astype(np.int64)
    keys = [str(x) for x in b.M["mol_key"][pick].tolist()]
    cache = C.PARENT_DIR.parents[0]  # tracks/structural-sufficiency
    cache_dir = cache / "lcrc_pcqm4mv2_transfer_pilot_v1" / "cache" / "pcqm_train1m"
    gids = np.load(cache_dir / "gids.npy").astype(np.int64)
    pos = {int(g): i for i, g in enumerate(gids.tolist())}
    wanted = {pos[int(r)]: (str(k), int(r)) for r, k in zip(row_ids.tolist(), keys) if int(r) in pos}
    found = {}
    with (cache_dir / "smiles.jsonl").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i in wanted:
                found[i] = line.strip()
                if len(found) == len(wanted):
                    break
    from rdkit import Chem
    ok = True
    n_checked = 0
    mismatches = []
    for i, (key, row) in wanted.items():
        raw = json.loads(found.get(i, '""'))
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            ok = False
            continue
        canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if canon != key:
            ok = False
            mismatches.append({"row_id": row, "raw_canonical": canon, "cache_key": key})
        n_checked += 1
    return {"ok": bool(ok and n_checked == len(wanted)),
            "n_checked": int(n_checked), "n_requested": int(len(wanted)),
            "mismatches": mismatches[:5],
            "note": "cache row maps to the same official CSV row id; RDKit canonical isomeric SMILES matches the cached molecule key"}


def run() -> dict[str, Any]:
    cfg = C.load_config()
    b = D.load_bundle(cfg)
    prep = {k: v for k, v in np.load(C.DATA_DIR / "prepared_pcqm.npz").items()}
    pd, pe = b.pairs_disc, b.pairs_eval
    checks: dict[str, Any] = {
        "environment": C.environment_snapshot(),
        "peak_rss_gb": C.peak_rss_gb(),
    }
    # canonical orientation
    canonical = {
        "all_A_lt_B_discovery": bool(np.all(pd["A"] < pd["B"])),
        "all_A_lt_B_evaluation": bool(np.all(pe["A"] < pe["B"])),
        "unique_ctx_A_B_discovery": int(len(set(zip(pd["ctx"].tolist(), pd["A"].tolist(), pd["B"].tolist())))),
        "rows_discovery": int(len(pd["delta"])),
        "unique_ctx_A_B_evaluation": int(len(set(zip(pe["ctx"].tolist(), pe["A"].tolist(), pe["B"].tolist())))),
        "rows_evaluation": int(len(pe["delta"])),
    }
    # recompute delta from context cells for a sample
    rng = np.random.default_rng(0)
    sample = rng.choice(len(pd["delta"]), size=min(2000, len(pd["delta"])), replace=False)
    err = 0.0
    for i in sample.tolist():
        a = int(pd["A"][i]); bb = int(pd["B"][i])
        pos_a = int(pd["pos_i"][i]); pos_b = int(pd["pos_j"][i])
        # canonical orientation means pos order is not guaranteed
        ya = b.ctx_disc.y[pos_a]; yb = b.ctx_disc.y[pos_b]
        ra = int(b.rank[b.ctx_disc.S[pos_a]]); rb = int(b.rank[b.ctx_disc.S[pos_b]])
        truth = (yb - ya) if rb > ra else (ya - yb)
        err = max(err, abs(truth - float(pd["delta"][i])))
    canonical["max_delta_recompute_error"] = float(err)
    canonical["shuffle_invariance"] = _canonical_shuffle_check(b)
    canonical["ok"] = bool(
        canonical["all_A_lt_B_discovery"] and canonical["all_A_lt_B_evaluation"]
        and canonical["unique_ctx_A_B_discovery"] == canonical["rows_discovery"]
        and canonical["unique_ctx_A_B_evaluation"] == canonical["rows_evaluation"]
        and canonical["max_delta_recompute_error"] <= 1e-12
        and canonical["shuffle_invariance"]["ok"]
    )
    checks["canonical_orientation"] = canonical
    # parent fit equality
    parent_p2 = C.read_json(C.PARENT_ANALYSIS / "p2_pcqm.json")
    parent_p3 = C.read_json(C.PARENT_ANALYSIS / "p3_pcqm.json")
    b_diff = float(np.max(np.abs(prep["shared_b"] - np.asarray(parent_p2["shared_fit"]["b"]))))
    pe_all = prep
    shared = prep["shared_b"][pe_all["eval_B"].astype(np.int64)] - prep["shared_b"][pe_all["eval_A"].astype(np.int64)]
    local_pred = pe_all["eval_preds_LOCAL"]
    null_pred = pe_all["eval_preds_LOCAL_NULL"]
    delta_all = pe_all["eval_delta"]
    fit_equal = {
        "shared_b_max_abs_diff_vs_parent": b_diff,
        "SHARED_mae_vs_parent_abs_diff": abs(float(np.mean(np.abs(shared - delta_all)))
                                             - parent_p3["metrics"]["all"]["SHARED"]["mae"]),
        "LOCAL_mae_vs_parent_abs_diff": abs(float(np.mean(np.abs(local_pred - delta_all)))
                                            - parent_p3["metrics"]["all"]["LOCAL"]["mae"]),
        "LOCAL_NULL_mae_vs_parent_abs_diff": abs(float(np.mean(np.abs(null_pred - delta_all)))
                                                  - parent_p3["metrics"]["all"]["LOCAL_NULL"]["mae"]),
    }
    fit_equal["ok"] = bool(b_diff <= 1e-10 and all(
        fit_equal[k] <= 1e-10 for k in (
            "SHARED_mae_vs_parent_abs_diff", "LOCAL_mae_vs_parent_abs_diff",
            "LOCAL_NULL_mae_vs_parent_abs_diff")))
    checks["parent_fit_equality"] = fit_equal
    # correction table and strict-mask artefacts
    corr = C.read_json(C.ANALYSIS_DIR / "correction_details.json")
    checks["correction"] = {
        "ok": bool(abs(corr["max_abs_p1_mae_difference"]) <= 1e-12
                   and abs(corr["max_abs_p1_rmse_difference"]) <= 1e-12
                   and corr["n_reversed_discovery_rows"] > 0
                   and corr["n_pairs_with_flipped_sign"] > 0),
        **{k: corr[k] for k in (
            "n_reversed_discovery_rows", "n_reversed_evaluation_rows",
            "fraction_reversed_discovery_rows", "n_pairs_with_flipped_sign",
            "max_abs_p1_mae_difference", "p1_v0_supported_n_obs",
            "p1_canonical_supported_n_obs")},
    }
    mask = C.read_json(C.ANALYSIS_DIR / "strict_mask_details.json")
    recs = [x for x in mask["pair_records"] if x.get("evaluable")]
    mask_check = {
        "n_selected_pairs": int(mask["n_selected_pairs"]),
        "n_evaluable_pairs": int(mask["n_evaluable_pairs"]),
        "all_direct_cooccurrence_removed": bool(all(x["direct_cooccurrence_after_mask"] == 0 for x in recs)),
        "all_removed_molecule_cells_absent": bool(all(x["removed_molecule_cells_left_in_fit"] == 0 for x in recs)),
        "all_connected_after_mask": bool(all(x["connected_after_mask"] for x in recs)),
        "all_pairs_have_5_eval_contexts": bool(all(x["n_evaluation_contexts"] >= 5 for x in recs)),
        "all_pairs_removed_positive_molecules": bool(all(x["n_molecules_removed"] > 0 for x in recs)),
    }
    mask_check["ok"] = bool(
        mask_check["all_direct_cooccurrence_removed"]
        and mask_check["all_removed_molecule_cells_absent"]
        and mask_check["all_connected_after_mask"]
        and mask_check["all_pairs_have_5_eval_contexts"]
        and mask_check["all_pairs_removed_positive_molecules"]
        and mask_check["n_evaluable_pairs"] >= 10
    )
    checks["strict_mask"] = mask_check
    # clean subset
    clean = D.molecule_clean_flags(b.cells, b.M)
    clean_mols = np.flatnonzero(clean["clean"])
    sample_mols = clean_mols[rng.choice(len(clean_mols), size=min(500, len(clean_mols)), replace=False)]
    keys = [str(b.M["mol_key"][i]) for i in sample_mols.tolist()]
    clean_check = {
        "n_clean_molecules": int(clean["clean"].sum()),
        "n_stereo_molecules": int(clean["stereo"].sum()),
        "n_duplicate_identity_molecules": int(clean["duplicate_identity"].sum()),
        "clean_keys_without_stereo_markers": bool(all(("@" not in k) and ("/" not in k) and ("\\" not in k) for k in keys)),
        "clean_molecules_have_at_most_one_row": bool(all(clean["n_distinct_rows"][i] <= 1 for i in sample_mols.tolist())),
        "n_clean_molecules_with_cuts": int(np.sum(clean["clean"] & (clean["n_distinct_rows"] >= 1))),
    }
    cell_clean = (b.ctx_eval.cell_mol_n == 1) & clean["clean"][np.maximum(b.ctx_eval.cell_mol_first, 0)]
    clean_check["clean_pair_rows"] = int(np.sum(pe_all["eval_clean"]))
    clean_check["clean_pair_rows_consistent"] = bool(np.all(
        pe_all["eval_clean"] == (cell_clean[pe_all["eval_pos_i"]] & cell_clean[pe_all["eval_pos_j"]])))
    clean_check["ok"] = bool(clean_check["clean_keys_without_stereo_markers"]
                             and clean_check["clean_molecules_have_at_most_one_row"]
                             and clean_check["clean_pair_rows_consistent"])
    checks["clean_subset"] = clean_check
    checks["identity_row"] = _identity_row_check(b)
    # synthetic controls
    controls = [_control_additive(), _control_crossed(), _control_strict_mask(),
                _control_unidentifiable()]
    checks["synthetic_controls"] = {"controls": controls,
                                    "all_ok": bool(all(x["ok"] for x in controls))}
    checks["resource"] = {
        "max_workers_allowed": int(cfg["resources"]["max_workers"]),
        "rss_ceiling_gb": int(cfg["resources"]["rss_ceiling_gb"]),
        "peak_rss_gb_this_process": C.peak_rss_gb(),
        "note": "each stage ran as a single CPU process with OMP_NUM_THREADS=1; no dense full-cell object table",
    }
    checks["all_ok"] = bool(
        checks["canonical_orientation"]["ok"]
        and checks["parent_fit_equality"]["ok"]
        and checks["correction"]["ok"]
        and checks["strict_mask"]["ok"]
        and checks["clean_subset"]["ok"]
        and checks["identity_row"]["ok"]
        and checks["synthetic_controls"]["all_ok"]
    )
    C.write_json(C.ANALYSIS_DIR / "checks.json", checks)
    return checks


def main() -> int:
    checks = run()
    print(json.dumps({k: checks[k] for k in (
        "all_ok", "canonical_orientation", "parent_fit_equality", "correction",
        "strict_mask", "clean_subset", "identity_row", "synthetic_controls")}, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
