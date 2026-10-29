"""Phase A2/A3 — CPU-only importance audit of the third-order path channel.

    uv run python his_phase_a.py            (device is forced to CPU)

Reads **only**:

* the frozen LCRC-A seed-0 ``best.pt`` checkpoint (never retrained here),
* official-train features + labels for a fixed set of 512 molecules,
* the frozen train/valid structure caches (valid *labels* are not touched),
* the frozen radii-2 support column and the frozen worst-10 list.

It never constructs, loads or scores the official test split.

Writes ``analysis/phase_a.json`` with the bootstrapping decision:

  A3 gate 1  ANOVA / gradient correctness            (from cpu_checks.json)
  A3 gate 2  ||H_3|| / ||H|| >= 0.10
  A3 gate 3  mean |delta prediction| >= 0.010 on 512 fixed train molecules
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import his_anova as A
import his_common as HIS

L, D, M, T = HIS.L, HIS.D, HIS.M, HIS.T

N_DIAG = 512
GATE_H3_RATIO = 0.10
GATE_PRED_SHIFT = 0.010
DEVICE = "cpu"


def per_molecule_codes(cache: dict) -> list[np.ndarray]:
    """Distinct ``path_cat`` codes present in each molecule (structure only)."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    path_cat = np.asarray(cache["path_cat"])
    out = []
    for r in range(len(atom_ptr) - 1):
        a0, a1 = int(atom_ptr[r]), int(atom_ptr[r + 1])
        segs = [path_cat[path_ptr[a]:path_ptr[a + 1]] for a in range(a0, a1)]
        out.append(np.unique(np.concatenate(segs)) if segs else np.zeros(0, dtype=np.uint8))
    return out


def per_molecule_joint(cache: dict) -> list[np.ndarray]:
    """Distinct ``(path_cat, path_end63)`` joint 2-hop contexts per molecule."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    cat = np.asarray(cache["path_cat"]).astype(np.int64)
    end = np.asarray(cache["path_end63"]).astype(np.int64)
    out = []
    for r in range(len(atom_ptr) - 1):
        a0, a1 = int(atom_ptr[r]), int(atom_ptr[r + 1])
        parts = [cat[path_ptr[a]:path_ptr[a + 1]] * 63 + end[path_ptr[a]:path_ptr[a + 1]]
                 for a in range(a0, a1)]
        out.append(np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64))
    return out


def calibrate_preview(model, train: dict) -> dict:
    """CPU preview of the frozen gradient-norm matched lambda (real first batch)."""
    gen = torch.Generator().manual_seed(101)          # frozen data-order seed (seed 0)
    order = torch.randperm(int(train["n_molecules"]), generator=gen).numpy()
    rows = order[:128]
    batch = D.collate(train, rows)
    target = torch.as_tensor(np.asarray(train["labels"])[rows], dtype=torch.float32)
    with L.C.isolated_rng():
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        g_l1 = A.affected_grad_vector(model).clone()
        model.zero_grad(set_to_none=True)
        pen, _ = A.his_penalty(model)
        pen.backward()
        g_reg = A.affected_grad_vector(model).clone()
        model.zero_grad(set_to_none=True)
    n_l1 = float(g_l1.norm())
    n_reg = float(g_reg.norm())
    lam = 0.10 * n_l1 / n_reg if n_reg > 0 else float("inf")
    return {
        "device": DEVICE,
        "rows": rows.tolist(),
        "rows_sha256": L.C.sha256_array(rows),
        "l1_loss": float(loss),
        "g_l1_affected_norm": n_l1,
        "g_reg_affected_norm": n_reg,
        "affected_elements": int(g_l1.numel()),
        "lambda": lam,
        "lambda_rule": "lambda = 0.10 * ||g_L1|| / ||g_reg|| on the affected parameters",
        "note": ("CPU preview only. The official calibration is recomputed on the "
                 "training GPU with the same frozen first batch and frozen initial "
                 "state, and is frozen in configs/his_v0.json before the 600-epoch runs."),
    }


def intervention(model, rows: np.ndarray, cache: dict) -> dict:
    """Replace ``path_emb`` by its <=2-order ANOVA projection and measure the shift."""
    base_pred, base_mae = T.evaluate(model, cache, rows, device=DEVICE)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]

    emb = model.local.path_emb.weight.detach()
    orig = emb.clone()
    proj = A.project_embedding(emb.detach()).reshape(orig.shape)
    with torch.no_grad():
        model.local.path_emb.weight.copy_(proj)
    proj_pred, proj_mae = T.evaluate(model, cache, rows, device=DEVICE)

    # reference: collapse every path code onto the global mean embedding
    with torch.no_grad():
        model.local.path_emb.weight.copy_(orig.mean(0, keepdim=True).expand_as(orig))
    mean_pred, mean_mae = T.evaluate(model, cache, rows, device=DEVICE)
    with torch.no_grad():
        model.local.path_emb.weight.copy_(orig)

    d = np.abs(proj_pred - base_pred)
    d_ref = np.abs(mean_pred - base_pred)
    return {
        "n_molecules": int(len(rows)),
        "rows": "official-train rows 0..511 (fixed, deterministic)",
        "labels_used": "official-train only",
        "base_mae_train512": base_mae,
        "projected_mae_train512": proj_mae,
        "mean_abs_prediction_shift": float(d.mean()),
        "median_abs_prediction_shift": float(np.median(d)),
        "p90_abs_prediction_shift": float(np.percentile(d, 90)),
        "max_abs_prediction_shift": float(d.max()),
        "fraction_molecules_shift_gt_0.010": float((d > GATE_PRED_SHIFT).mean()),
        "prediction_std": float(np.std(base_pred)),
        "reference_total_ablation": {
            "what": "path_emb rows replaced by the global mean row (all code variation removed)",
            "mean_abs_prediction_shift": float(d_ref.mean()),
            "mae_train512": mean_mae,
        },
        "relation_to_mae": float(d.mean()) / max(base_mae, 1e-12),
    }


def category_stats(train: dict) -> dict:
    codes = per_molecule_codes(train)
    n_mol = len(codes)
    df = np.zeros(189, dtype=np.int64)
    for s in codes:
        df[s] += 1
    occ = np.zeros(189, dtype=np.int64)
    np.add.at(occ, np.asarray(train["path_cat"]), 1)
    used = np.nonzero(df > 0)[0]
    total_occ = int(occ.sum())
    order = np.argsort(-occ)
    uniq = lambda v: [int(x) for x in np.unique(v)]  # noqa: E731
    return {
        "n_categories": 189,
        "n_categories_used_train": int((df > 0).sum()),
        "n_categories_unused_train": int((df == 0).sum()),
        "unused_fraction": float((df == 0).mean()),
        "df_quantiles_all": {f"p{q}": float(np.percentile(df, q)) for q in (0, 5, 25, 50, 75, 90, 100)},
        "df_quantiles_used": {f"p{q}": float(np.percentile(df[used], q)) for q in (0, 5, 25, 50, 75, 100)},
        "used_codes": [
            {"code": int(c), "a_middle_type": int(c // 9), "b_bond1": int((c % 9) // 3) + 1,
             "c_bond2": int(c % 3) + 1, "document_frequency": int(df[c]),
             "molecule_share": float(df[c] / n_mol), "occurrences": int(occ[c])}
            for c in used
        ],
        "low_frequency": {
            "used_code_df_lt_10": int(((df > 0) & (df < 10)).sum()),
            "used_code_df_lt_100": int(((df > 0) & (df < 100)).sum()),
            "used_code_df_lt_1000": int(((df > 0) & (df < 1000)).sum()),
        },
        "occurrence_concentration": {
            "total_tokens_train": total_occ,
            "tokens_per_molecule": total_occ / n_mol,
            "top10_occurrence_share": float(occ[order[:10]].sum() / total_occ),
            "distinct_codes_with_occurrences": int((occ > 0).sum()),
            "top10_codes": uniq(order[:10]),
        },
        "n_path_tokens": int(np.asarray(train["path_cat"]).shape[0]),
        "mean_paths_per_molecule": float(np.asarray(train["path_cat"]).shape[0] / n_mol),
    }


def coverage_relation(train: dict, valid: dict, masks: dict) -> dict:
    tr_codes = per_molecule_codes(train)
    va_codes = per_molecule_codes(valid)
    tr_joint = per_molecule_joint(train)
    va_joint = per_molecule_joint(valid)
    df = np.zeros(189, dtype=np.int64)
    for s in tr_codes:
        df[s] += 1
    dfj = np.zeros(189 * 63, dtype=np.int64)
    for s in tr_joint:
        dfj[s] += 1

    all_codes_seen = np.asarray([bool(len(s) and (df[s] > 0).all()) for s in va_codes])
    all_joint_seen = np.asarray([bool(len(s) and (dfj[s] > 0).all()) for s in va_joint])
    unseen = np.asarray(masks["r2_unseen"])
    min_df = np.asarray([int(df[s].min()) if len(s) else 0 for s in va_codes])
    return {
        "valid_molecules_all_path_codes_seen_in_train": int(all_codes_seen.sum()),
        "valid_molecules_all_joint_contexts_seen_in_train": int(all_joint_seen.sum()),
        "r2_unseen_rows_reused_from_frozen_anchor": int(unseen.sum()),
        "r2_unseen_ordinary_rows": int(np.asarray(masks["r2_unseen_ordinary"]).sum()),
        "among_r2_unseen_all_codes_seen": int(all_codes_seen[unseen].sum()),
        "among_r2_unseen_all_joint_seen": int(all_joint_seen[unseen].sum()),
        "among_r2_unseen_all_joint_seen_fraction": float(all_joint_seen[unseen].mean()),
        "among_r2_seen_all_joint_seen_fraction": float(all_joint_seen[~unseen].mean()),
        "median_min_code_df_r2_unseen": float(np.median(min_df[unseen])),
        "median_min_code_df_r2_seen": float(np.median(min_df[~unseen])),
        "n_joint_contexts_used_train": int((dfj > 0).sum()),
        "n_joint_contexts_total": int(189 * 63),
        "joint_context_coverage_note": (
            "'component seen' means the (middle type, bond1, bond2) token appears in train. "
            "'joint seen' additionally requires the (token, end type, second bond) context to "
            "appear. Neither implies that the label rule over that context is learnable."),
    }


def main() -> int:
    HIS.ANALYSIS.mkdir(parents=True, exist_ok=True)
    inputs = HIS.verify_frozen_inputs(verbose=False)
    train = D.load_cache("train")
    valid = D.load_cache("valid")
    norm = D.load_norm()
    masks = HIS.valid_masks()

    payload = torch.load(HIS.LCRC_A_S0_BEST, map_location="cpu", weights_only=False)
    model = M.load_model_from_state("A", norm, payload["state_dict"], DEVICE)
    model.eval()

    h = A.effective_response(model).detach()
    h3 = A.third_order(h)
    h_stats = {
        "h_fro": A.fro(h), "h3_fro": A.fro(h3),
        "h3_ratio": A.fro(h3) / A.fro(h),
        "h3_ratio_to_centered_h": A.fro(h3) / A.fro(h - h.mean(dim=(0, 1, 2), keepdim=True)),
        "h_rms": float(h.pow(2).mean().sqrt()),
        "h3_rms": float(h3.pow(2).mean().sqrt()),
        "h_le2_fro": A.fro(A.anova_le2(h)),
        "marginal_norms_of_h3": A.margin_norms(h3),
        "checkpoint": {"path": HIS.track_rel(HIS.LCRC_A_S0_BEST), "sha256": HIS.sha256_file(HIS.LCRC_A_S0_BEST),
                       "epoch": int(payload["epoch"]), "valid_mae": float(payload["valid_mae"])},
        "note": ("H_3 is the pure three-way residual of the 21x3x3 path-code read-out. "
                 "The ratio is over the raw H (including its grand-mean component), which is "
                 "the quantity named in the pre-registered gate."),
    }

    rows = np.arange(N_DIAG, dtype=np.int64)
    iv = intervention(model, rows, train)

    # a second, cheaper check that the intervention is not invisible: relative
    # shift of the *per-category* read-out, not just the prediction
    emb = model.local.path_emb.weight.detach()
    proj = A.project_embedding(emb).reshape(emb.shape)
    cat_shift = float((proj - emb).norm() / emb.norm())

    fresh, fresh_info = M.build_model("A", norm, 0, DEVICE)
    calib = calibrate_preview(fresh, train)

    gates = {
        "math_and_gradient_correct": bool(
            HIS.read_json(HIS.ANALYSIS / "cpu_checks.json")["all_pass"]),
        "h3_ratio": h_stats["h3_ratio"],
        "h3_ratio_threshold": GATE_H3_RATIO,
        "h3_ratio_pass": bool(h_stats["h3_ratio"] >= GATE_H3_RATIO),
        "mean_abs_prediction_shift": iv["mean_abs_prediction_shift"],
        "prediction_shift_threshold": GATE_PRED_SHIFT,
        "prediction_shift_pass": bool(iv["mean_abs_prediction_shift"] >= GATE_PRED_SHIFT),
    }
    proceed = (gates["math_and_gradient_correct"] and gates["h3_ratio_pass"]
               and gates["prediction_shift_pass"])
    decision = "PROCEED_TO_GPU_SMOKE" if proceed else "CPU_STOP"
    if not gates["math_and_gradient_correct"]:
        decision = "CPU_STOP_CORRECTNESS"
    elif not gates["h3_ratio_pass"]:
        decision = "NO_MATERIAL_THIRD_ORDER_CHANNEL_H3_RATIO"
    elif not gates["prediction_shift_pass"]:
        decision = "NO_MATERIAL_THIRD_ORDER_CHANNEL_PRED_SHIFT"

    out = {
        "round": HIS.ROUND_ID,
        "phase": "A2/A3",
        "created": HIS.now(),
        "git_commit": L.C.git_commit(),
        "device": DEVICE,
        "official_test_loaded": False,
        "valid_labels_read": False,
        "frozen_inputs": inputs,
        "third_order_energy_frozen_checkpoint": h_stats,
        "intervention_path_emb_le2_projection": {**iv, "embedding_table_relative_shift": cat_shift},
        "path_category_statistics_train": category_stats(train),
        "coverage_relation": coverage_relation(train, valid, masks),
        "lambda_cpu_preview": calib,
        "initial_state": {"parameters": fresh_info["parameters"],
                          "sha256": fresh_info["initial_state_sha256"]},
        "gates": gates,
        "decision": decision,
        "decision_note": (
            "The thresholds only decide whether one exploratory training run is worth its "
            "GPU cost; they are not evidence for or against the value of third-order "
            "interactions in general."),
    }
    HIS.write_json(HIS.ANALYSIS / "phase_a.json", out)

    print("== HIS-v0 Phase A ==")
    print(f"||H3||/||H||              = {h_stats['h3_ratio']:.6f}  (gate >= {GATE_H3_RATIO})")
    print(f"mean |dpred| on train512  = {iv['mean_abs_prediction_shift']:.6f}  "
          f"(gate >= {GATE_PRED_SHIFT}); total-ablation reference "
          f"{iv['reference_total_ablation']['mean_abs_prediction_shift']:.6f}")
    print(f"train512 MAE base/proj    = {iv['base_mae_train512']:.6f} / {iv['projected_mae_train512']:.6f}")
    print(f"used path codes           = {out['path_category_statistics_train']['n_categories_used_train']}/189")
    print(f"lambda CPU preview        = {calib['lambda']:.6e} "
          f"(||g_L1||={calib['g_l1_affected_norm']:.4e}, ||g_reg||={calib['g_reg_affected_norm']:.4e})")
    print(f"decision                  = {decision}")
    return 0 if proceed else 1


if __name__ == "__main__":
    raise SystemExit(main())
