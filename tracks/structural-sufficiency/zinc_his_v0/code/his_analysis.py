"""Phase C — one-shot valid-only analysis for HIS-v0 (no official test access).

    uv run python his_analysis.py [--replay runs/REPLAY_s0] [--his runs/HIS_s0]

Reads the two frozen training payloads and their per-molecule valid predictions,
recomputes every pre-registered group, the paired bootstrap and the verdict, and
writes

    analysis/main_table.{json,md}
    analysis/groups.{json,md}
    analysis/contrasts.{json,md}
    analysis/mechanism.json
    analysis/per_molecule_valid.csv
    analysis/summary.json

``delta = HIS - REPLAY`` on per-molecule absolute error; negative means HIS is
better.  The official test split is never constructed, loaded or scored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import his_anova as A
import his_common as HIS

L, D, M, T = HIS.L, HIS.D, HIS.M, HIS.T

BOOTSTRAP_SEED = 20261030
N_BOOT = 10000
GROUP_ORDER = [
    "all", "ordinary_y_gt_-3", "r2_unseen_ordinary", "seen_ordinary", "tail_y_le_-3",
    "all_minus_row172", "all_minus_worst10", "row172", "a_s0_worst10_fixed",
    "r2_unseen", "ring_rich_ge4",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mae(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    ae = np.abs(np.asarray(pred, np.float64) - np.asarray(y, np.float64))
    if mask is not None:
        ae = ae[mask]
    return float(ae.mean()) if len(ae) else float("nan")


def _bootstrap(delta: np.ndarray, *, seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT) -> dict:
    d = np.asarray(delta, np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "win_fraction_his_better": float((d < 0).mean()), "n": int(n)}


def _load_run(path: Path) -> dict:
    payload = HIS.read_json(path / "train.json")
    npz = np.load(path / "predictions" / "valid_best.npz")
    final_npz = np.load(path / "predictions" / "valid_final.npz")
    best_file = path / "best.pt"
    final_file = path / "final.pt"
    return {
        "dir": str(path),
        "payload": payload,
        "rows": np.asarray(npz["row"]),
        "pred_best": np.asarray(npz["pred"], dtype=np.float64),
        "pred_final": np.asarray(final_npz["pred"], dtype=np.float64),
        "best_sha256": HIS.sha256_file(best_file),
        "final_sha256": HIS.sha256_file(final_file),
        "best_path": best_file,
        "final_path": final_file,
    }


def _model_from(ckpt_path: Path, norm) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = M.load_model_from_state("A", norm, payload["state_dict"], "cpu")
    model.eval()
    return model


def _projection_shift(model, cache, rows) -> dict:
    base, mae_base = T.evaluate(model, cache, rows, device="cpu")
    emb = model.local.path_emb.weight.detach()
    orig = emb.clone()
    with torch.no_grad():
        model.local.path_emb.weight.copy_(A.project_embedding(emb.detach()).reshape(orig.shape))
    proj, mae_proj = T.evaluate(model, cache, rows, device="cpu")
    with torch.no_grad():
        model.local.path_emb.weight.copy_(orig)
    d = np.abs(proj - base)
    return {"mae_base": mae_base, "mae_projected": mae_proj,
            "mean_abs_prediction_shift": float(d.mean()),
            "median_abs_prediction_shift": float(np.median(d)),
            "fraction_shift_gt_0.010": float((d > 0.010).mean())}


def _joint_response_table(model, train_cache, rows) -> dict:
    """Average effective cat-indexed response of the *unconstrained* joint channel.

    ``path_joint(v) = joint_proj(path_u[cat] * path_v[end63])`` is read by the
    first-layer weight columns [384, 416).  It is not linear in ``cat`` because it
    multiplies by the end-type embedding, so we linearise it over the empirical
    end-type distribution of the given molecules:

        J[cat] = W_joint @ joint_proj( path_u[cat] * mean_{p: cat_p = cat} path_v[end_p] )

    ``||J_3|| / ||J||`` is therefore the pure three-way share of the cat-indexed
    structure this *unpenalised* channel exposes.  It is a proxy (the channel is
    genuinely bilinear), reported to test whether the penalty was routed around.
    """
    local = model.local
    sl_joint = slice(384, 416)          # end_emb ends at 384, path_joint is 32 wide
    w_joint = local.mlp.lins[0].weight.detach()[:, sl_joint]
    cat = np.asarray(train_cache["path_cat"]).astype(np.int64)
    end = np.asarray(train_cache["path_end63"]).astype(np.int64)
    atom_ptr = np.asarray(train_cache["atom_ptr"])
    path_ptr = np.asarray(train_cache["path_ptr"])
    keep = np.zeros(int(atom_ptr[-1]), dtype=bool)
    for r in rows:
        keep[atom_ptr[r]:atom_ptr[r + 1]] = True
    sel = np.zeros(len(cat), dtype=bool)
    for a in np.nonzero(keep)[0]:
        sel[path_ptr[a]:path_ptr[a + 1]] = True
    cat, end = cat[sel], end[sel]
    with torch.no_grad():
        u = local.path_u.weight.detach()
        v = local.path_v.weight.detach()
        jp = local.joint_proj
        table = torch.zeros(189, v.shape[1], dtype=torch.float32)
        used = 0
        for c in range(189):
            m = cat == c
            if not bool(m.any()):
                table[c] = u[c] * v.mean(0)
                continue
            vbar = v[torch.as_tensor(end[m], dtype=torch.long)].mean(0)
            table[c] = u[c] * vbar
            used += 1
        joint4 = (jp(table) @ w_joint.T).reshape(21, 3, 3, -1)      # [21, 3, 3, 192]
        j3 = A.third_order(joint4)
        out = {
            "joint_response_fro": A.fro(joint4),
            "joint_third_order_fro": A.fro(j3),
            "joint_third_order_ratio": A.fro(j3) / max(A.fro(joint4), 1e-30),
            "joint_third_order_ratio_centered": A.fro(j3) / max(
                A.fro(joint4 - joint4.mean(dim=(0, 1, 2), keepdim=True)), 1e-30),
            "categories_with_paths_in_rows": int(used),
            "path_u_fro": A.fro(u),
            "path_gate_fro": A.fro(local.path_gate.weight.detach()),
        }
        g = local.path_gate.weight.detach().reshape(21, 3, 3, 1)
        g3 = A.third_order(g)
        out["path_gate_third_order_ratio"] = A.fro(g3) / max(A.fro(g), 1e-30)
        out["path_u_third_order_ratio"] = A.fro(A.third_order(
            u.reshape(21, 3, 3, -1))) / max(A.fro(u), 1e-30)
    return out


def _channel_ablation(model, train_cache, rows) -> dict:
    """Prediction shift (512 fixed train molecules) under single-channel edits.

    Diagnostic only -- nothing here is added to the training objective.
    """
    local = model.local
    base, _ = T.evaluate(model, train_cache, rows, device="cpu")

    def shift(fn) -> float:
        with torch.no_grad():
            fn()
        pred, _ = T.evaluate(model, train_cache, rows, device="cpu")
        return float(np.abs(pred - base).mean())

    out: dict[str, float] = {}
    emb = local.path_emb.weight.detach().clone()
    u = local.path_u.weight.detach().clone()
    gate = local.path_gate.weight.detach().clone()

    def restore() -> None:
        with torch.no_grad():
            local.path_emb.weight.copy_(emb)
            local.path_u.weight.copy_(u)
            local.path_gate.weight.copy_(gate)

    out["project_path_emb_le2"] = shift(
        lambda: local.path_emb.weight.copy_(A.project_embedding(emb).reshape(emb.shape)))
    restore()
    out["project_path_u_le2"] = shift(
        lambda: local.path_u.weight.copy_(A.project_embedding(u).reshape(u.shape)))
    restore()
    out["ablate_path_emb_rowmean"] = shift(
        lambda: local.path_emb.weight.copy_(emb.mean(0, keepdim=True).expand_as(emb)))
    restore()
    out["ablate_path_u_rowmean"] = shift(
        lambda: local.path_u.weight.copy_(u.mean(0, keepdim=True).expand_as(u)))
    restore()
    out["ablate_path_u_zero"] = shift(lambda: local.path_u.weight.zero_())
    restore()
    out["ablate_path_gate_mean"] = shift(
        lambda: local.path_gate.weight.fill_(float(gate.mean())))
    restore()
    out["ablate_all_cat_indexed"] = shift(lambda: (
        local.path_emb.weight.copy_(emb.mean(0, keepdim=True).expand_as(emb)),
        local.path_u.weight.copy_(u.mean(0, keepdim=True).expand_as(u)),
        local.path_gate.weight.fill_(float(gate.mean())),
    ))
    restore()
    out["restored_prediction_max_abs_delta"] = float(
        np.abs(T.evaluate(model, train_cache, rows, device="cpu")[0] - base).max())
    return out


def _h_stats(model) -> dict:
    with torch.no_grad():
        _, stats = A.his_penalty(model)
    return {k: float(v) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HIS-v0 Phase C valid analysis")
    ap.add_argument("--replay", default=str(HIS.RUNS / "REPLAY_s0"))
    ap.add_argument("--his", default=str(HIS.RUNS / "HIS_s0"))
    ap.add_argument("--out", default=str(HIS.ANALYSIS))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    L.ensure_dir(out_dir)

    replay = _load_run(Path(args.replay))
    his = _load_run(Path(args.his))
    valid = D.load_cache("valid")
    norm = D.load_norm()
    masks = HIS.valid_masks()
    anchor = HIS.load_anchor()
    y = np.asarray(anchor["y"], dtype=np.float64)
    rows = np.arange(len(y))
    assert np.array_equal(replay["rows"], rows) and np.array_equal(his["rows"], rows)
    # valid labels must equal the frozen anchor labels (float32 cache vs 9-dp CSV)
    label_gap = float(np.abs(np.asarray(valid["labels"], np.float64) - y).max())
    assert label_gap < 1e-6, label_gap

    ae_r = np.abs(replay["pred_best"] - y)
    ae_h = np.abs(his["pred_best"] - y)
    delta = ae_h - ae_r
    ae_rf = np.abs(replay["pred_final"] - y)
    ae_hf = np.abs(his["pred_final"] - y)
    delta_final = ae_hf - ae_rf

    # ---------------- per-group table -----------------------------------
    groups: dict[str, dict] = {}
    for name in GROUP_ORDER:
        m = masks[name]
        groups[name] = {
            "n": int(m.sum()),
            "replay_mae": _mae(replay["pred_best"], y, m),
            "his_mae": _mae(his["pred_best"], y, m),
            "delta": _mae(his["pred_best"], y, m) - _mae(replay["pred_best"], y, m),
            "replay_final_mae": _mae(replay["pred_final"], y, m),
            "his_final_mae": _mae(his["pred_final"], y, m),
            "delta_final": _mae(his["pred_final"], y, m) - _mae(replay["pred_final"], y, m),
        }
    for name in GROUP_ORDER:
        g = groups[name]
        g["paired"] = _bootstrap(delta[masks[name]]) if g["n"] > 1 else None

    # ---------------- contrasts / robustness ----------------------------
    def contrast(mask: np.ndarray, label: str) -> dict:
        d = delta[mask]
        base = _bootstrap(d)
        base["label"] = label
        # drop the single biggest HIS win and the single biggest HIS loss
        if len(d) >= 3:
            i_win = int(np.argmin(d))
            i_loss = int(np.argmax(d))
            base["delta_drop_biggest_his_win"] = float(np.delete(d, i_win).mean())
            base["delta_drop_biggest_his_loss"] = float(np.delete(d, i_loss).mean())
            order = np.argsort(-np.abs(d))
            keep = np.ones(len(d), dtype=bool)
            keep[order[:10]] = False
            base["delta_drop_top10_abs"] = float(d[keep].mean())
            base["dropped_biggest_his_win_row"] = int(np.nonzero(mask)[0][i_win])
        return base

    contrasts = {
        "full": contrast(masks["all"], "all"),
        "ordinary": contrast(masks["ordinary_y_gt_-3"], "ordinary"),
        "r2_unseen_ordinary": contrast(masks["r2_unseen_ordinary"], "r2_unseen_ordinary"),
        "seen_ordinary": contrast(masks["seen_ordinary"], "seen_ordinary"),
        "tail": contrast(masks["tail_y_le_-3"], "tail"),
        "minus_row172": contrast(masks["all_minus_row172"], "all_minus_row172"),
        "minus_worst10": contrast(masks["all_minus_worst10"], "all_minus_worst10"),
    }

    # ---------------- windows / endpoints --------------------------------
    def endpoints(run: dict) -> dict:
        p = run["payload"]
        curve = p["curve"]
        return {
            "best_valid_mae": float(p["best"]["valid_mae"]),
            "best_epoch": int(p["best"]["epoch"]),
            "final_valid_mae": float(p["final_valid_mae"]),
            "W20": None if p["W20"] is None else float(p["W20"]),
            "T100": None if p["T100"] is None else float(p["T100"]),
            "best_train_mae_eval": float(p["best_train_mae_eval"]),
            "final_train_mae_eval": float(p["final_train_mae_eval"]),
            "updates": int(p["total_updates"]),
            "gpu_hours": float(p["gpu_hours_estimate"]),
            "lambda": float(p["lambda"]),
            "initial_state_sha256": p["model_info"]["initial_state_sha256"],
            "first_batch_rows_sha256": L.C.sha256_array(np.asarray(p["first_batch_rows"])),
            "lr_sha256": L.C.sha256_array(np.asarray([c["lr"] for c in curve], np.float64)),
            "h3_ratio_first": float(curve[0]["h3_ratio"]),
            "h3_ratio_last": float(curve[-1]["h3_ratio"]),
            "h3_ratio_best_epoch": float(curve[int(p["best"]["epoch"]) - 1]["h3_ratio"]),
            "penalty_last": float(curve[-1]["penalty"]),
            "curve": curve,
        }

    ep_r, ep_h = endpoints(replay), endpoints(his)

    def _diff(a, b):
        return None if a is None or b is None else float(a) - float(b)

    windows = {
        "best_valid": ep_h["best_valid_mae"] - ep_r["best_valid_mae"],
        "final_valid": ep_h["final_valid_mae"] - ep_r["final_valid_mae"],
        "W20": _diff(ep_h["W20"], ep_r["W20"]),
        "T100": _diff(ep_h["T100"], ep_r["T100"]),
        "best_train_mae_eval": ep_h["best_train_mae_eval"] - ep_r["best_train_mae_eval"],
        "final_train_mae_eval": ep_h["final_train_mae_eval"] - ep_r["final_train_mae_eval"],
    }

    # ---------------- mechanism -----------------------------------------
    train_cache = D.load_cache("train")
    diag_rows = np.arange(512, dtype=np.int64)
    mech = {
        "replay": {
            "best_h_stats": _h_stats(_model_from(replay["best_path"], norm)),
            "final_h_stats": _h_stats(_model_from(replay["final_path"], norm)),
            "best_projection_shift_train512": _projection_shift(
                _model_from(replay["best_path"], norm), train_cache, diag_rows),
        },
        "his": {
            "best_h_stats": _h_stats(_model_from(his["best_path"], norm)),
            "final_h_stats": _h_stats(_model_from(his["final_path"], norm)),
            "best_projection_shift_train512": _projection_shift(
                _model_from(his["best_path"], norm), train_cache, diag_rows),
        },
        "curve_h3_ratio": {
            "replay": [float(c["h3_ratio"]) for c in ep_r["curve"]],
            "his": [float(c["h3_ratio"]) for c in ep_h["curve"]],
        },
        "penalty_curve": {
            "replay": [float(c["penalty"]) for c in ep_r["curve"]],
            "his": [float(c["penalty"]) for c in ep_h["curve"]],
        },
        "lambda": {"replay": ep_r["lambda"], "his": ep_h["lambda"]},
        "lambda_check": his["payload"].get("lambda_check"),
        "penalty_first_step": his["payload"].get("penalty_first_step"),
        "penalty_last_step": his["payload"].get("penalty_last_step"),
        "penalty_min": his["payload"].get("penalty_min"),
        "penalty_max": his["payload"].get("penalty_max"),
        "gradient_audit_his": his["payload"].get("gradient_audit_first_batch"),
        "gradient_audit_replay": replay["payload"].get("gradient_audit_first_batch"),
    }
    for tag, ckpt in (("replay", replay), ("his", his)):
        mdl = _model_from(ckpt["best_path"], norm)
        mech[tag]["joint_channel"] = _joint_response_table(mdl, train_cache, diag_rows)
        mech[tag]["channel_ablation_train512"] = _channel_ablation(mdl, train_cache, diag_rows)

    mech["h3_ratio_reduction_best"] = (
        mech["replay"]["best_h_stats"]["h3_ratio"] - mech["his"]["best_h_stats"]["h3_ratio"])
    mech["projection_shift_ratio"] = (
        mech["his"]["best_projection_shift_train512"]["mean_abs_prediction_shift"]
        / max(mech["replay"]["best_projection_shift_train512"]["mean_abs_prediction_shift"], 1e-12))
    mech["path_emb_third_order_share_of_total_cat_dependence"] = {
        "replay": mech["replay"]["channel_ablation_train512"]["project_path_emb_le2"]
        / max(mech["replay"]["channel_ablation_train512"]["ablate_all_cat_indexed"], 1e-12),
        "his": mech["his"]["channel_ablation_train512"]["project_path_emb_le2"]
        / max(mech["his"]["channel_ablation_train512"]["ablate_all_cat_indexed"], 1e-12),
    }
    mech["regulariser_active"] = bool(
        mech["h3_ratio_reduction_best"] > 0.02
        and his["payload"].get("penalty_first_step") is not None
        and float(his["payload"]["penalty_first_step"]) > 0
        and np.isfinite(mech["his"]["best_h_stats"]["penalty"]))

    # ---------------- execution-fairness gates ---------------------------
    gates = {
        "same_initial_state": ep_r["initial_state_sha256"] == ep_h["initial_state_sha256"]
        and ep_r["initial_state_sha256"] == HIS.FROZEN["lcrc_a_s0_initial_state_sha256"],
        "same_first_batch_rows": ep_r["first_batch_rows_sha256"] == ep_h["first_batch_rows_sha256"],
        "same_lr_trajectory": ep_r["lr_sha256"] == ep_h["lr_sha256"],
        "same_updates": ep_r["updates"] == ep_h["updates"] == 47400,
        "same_parameters": (replay["payload"]["model_info"]["parameters"]
                            == his["payload"]["model_info"]["parameters"] == 237519),
        "same_ssl": replay["payload"]["ssl_sha256"] == his["payload"]["ssl_sha256"],
        "same_train_cache": (replay["payload"]["data"]["train_cache_content_sha256"]
                             == his["payload"]["data"]["train_cache_content_sha256"]),
        "same_valid_cache": (replay["payload"]["data"]["valid_cache_content_sha256"]
                             == his["payload"]["data"]["valid_cache_content_sha256"]),
        "same_norm": replay["payload"]["norm"]["fingerprint"] == his["payload"]["norm"]["fingerprint"],
        "no_test_access": (not replay["payload"]["data"]["official_test_loaded"]
                           and not his["payload"]["data"]["official_test_loaded"]),
        "regulariser_active": mech["regulariser_active"],
    }

    # ---------------- verdict -------------------------------------------
    g = {k: groups[k]["delta"] for k in groups}
    w = windows
    green = {
        "full_le_-0.005": g["all"] <= -0.005,
        "ordinary_le_-0.002": g["ordinary_y_gt_-3"] <= -0.002,
        "r2_unseen_ordinary_le_-0.008": g["r2_unseen_ordinary"] <= -0.008,
        "minus_row172_improves": g["all_minus_row172"] < 0,
        "minus_worst10_improves": g["all_minus_worst10"] < 0,
        "W20_not_worse_than_+0.002": w["W20"] is not None and w["W20"] <= 0.002,
        "T100_not_worse_than_+0.002": w["T100"] is not None and w["T100"] <= 0.002,
    }
    amber = {
        "r2_unseen_ordinary_le_-0.008": g["r2_unseen_ordinary"] <= -0.008,
        "ordinary_le_-0.002": g["ordinary_y_gt_-3"] <= -0.002,
        "full_not_worse_than_+0.002": g["all"] <= 0.002,
        "r2_unseen_not_single_row_driven": (
            groups["r2_unseen_ordinary"]["paired"] is not None
            and contrasts["r2_unseen_ordinary"].get("delta_drop_biggest_his_win", 0.0) < 0),
        "ordinary_not_single_row_driven": (
            contrasts["ordinary"].get("delta_drop_biggest_his_win", 0.0) < 0),
    }
    if not gates["regulariser_active"]:
        verdict = "RED"
        verdict_reason = "the regulariser did not measurably change the third-order response"
    elif not all(gates[k] for k in ("same_initial_state", "same_first_batch_rows",
                                    "same_lr_trajectory", "same_updates", "same_parameters",
                                    "same_ssl", "same_train_cache", "same_valid_cache",
                                    "same_norm", "no_test_access")):
        verdict = "RED"
        verdict_reason = "execution gate failed; the two arms are not comparable"
    elif all(green.values()):
        verdict = "GREEN"
        verdict_reason = "all GREEN conditions met"
    elif all(amber.values()):
        verdict = "AMBER"
        verdict_reason = "local unseen/ordinary signal without a full-MAE gain"
    else:
        verdict = "RED"
        verdict_reason = "neither GREEN nor AMBER conditions met"

    # ---------------- top contributors ----------------------------------
    contrib = []
    for i in np.argsort(-np.abs(delta))[:20]:
        contrib.append({
            "row": int(i), "y": float(y[i]), "ae_replay": float(ae_r[i]),
            "ae_his": float(ae_h[i]), "delta": float(delta[i]),
            "r2_unseen": bool(masks["r2_unseen"][i]),
            "family": ("tail" if y[i] <= -3 else "ordinary"),
        })

    # ---------------- write --------------------------------------------
    main_table = {
        "round": HIS.ROUND_ID,
        "created": HIS.now(),
        "git_commit": L.C.git_commit(),
        "primary_metric": "official valid 1000 full raw-y MAE, best-by-valid checkpoint",
        "protocol": {
            "replay": {"dir": replay["dir"], "best_sha256": replay["best_sha256"],
                       "final_sha256": replay["final_sha256"], "lambda": ep_r["lambda"],
                       "git_commit": replay["payload"]["git_commit"]},
            "his": {"dir": his["dir"], "best_sha256": his["best_sha256"],
                    "final_sha256": his["final_sha256"], "lambda": ep_h["lambda"],
                    "git_commit": his["payload"]["git_commit"]},
            "environment": his["payload"]["environment"],
        },
        "replay": {k: v for k, v in ep_r.items() if k != "curve"},
        "his": {k: v for k, v in ep_h.items() if k != "curve"},
        "windows": windows,
        "gates": gates,
        "groups": groups,
        "history_anchors": HIS.HIST,
        "valid_label_max_abs_gap_vs_anchor_csv": label_gap,
        "official_test_loaded": False,
    }
    groups_doc = {"round": HIS.ROUND_ID, "created": HIS.now(), "groups": groups,
                  "masks_note": {
                      "r2_unseen": "frozen anchor column r2_unseen_atom_frac > 0 (zinc_performance_search_v1)",
                      "a_s0_worst10_fixed": "frozen list from zinc_cjp_ssl_v1 config",
                      "row172": HIS.ROW172},
                  "official_test_loaded": False}
    contrasts_doc = {"round": HIS.ROUND_ID, "created": HIS.now(), "contrasts": contrasts,
                     "sign": "delta = HIS - REPLAY absolute error; negative = HIS better",
                     "bootstrap": {"n_boot": N_BOOT, "seed": BOOTSTRAP_SEED,
                                   "resampling": "per molecule, valid only"},
                     "top_contributors": contrib,
                     "delta_final_checkpoint": {
                         "full": float(delta_final.mean()),
                         "ordinary": float(delta_final[masks["ordinary_y_gt_-3"]].mean()),
                         "r2_unseen_ordinary": float(delta_final[masks["r2_unseen_ordinary"]].mean()),
                         "minus_row172": float(delta_final[masks["all_minus_row172"]].mean()),
                         "minus_worst10": float(delta_final[masks["all_minus_worst10"]].mean()),
                     },
                     "official_test_loaded": False}
    summary = {
        "round": HIS.ROUND_ID,
        "created": HIS.now(),
        "git_commit": L.C.git_commit(),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "green_conditions": green,
        "amber_conditions": amber,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "primary": {
            "replay_valid": ep_r["best_valid_mae"], "his_valid": ep_h["best_valid_mae"],
            "delta": groups["all"]["delta"],
            "replay_best_epoch": ep_r["best_epoch"], "his_best_epoch": ep_h["best_epoch"],
            "W20_delta": windows["W20"], "T100_delta": windows["T100"],
            "final_delta": windows["final_valid"],
        },
        "mechanism": {
            "lambda": ep_h["lambda"],
            "h3_ratio_replay_best": mech["replay"]["best_h_stats"]["h3_ratio"],
            "h3_ratio_his_best": mech["his"]["best_h_stats"]["h3_ratio"],
            "h3_ratio_reduction": mech["h3_ratio_reduction_best"],
            "projection_shift_replay": mech["replay"]["best_projection_shift_train512"],
            "projection_shift_his": mech["his"]["best_projection_shift_train512"],
            "regulariser_active": mech["regulariser_active"],
        },
        "official_test_loaded": False,
    }

    HIS.write_json(out_dir / "main_table.json", main_table)
    HIS.write_json(out_dir / "groups.json", groups_doc)
    HIS.write_json(out_dir / "contrasts.json", contrasts_doc)
    HIS.write_json(out_dir / "mechanism.json", mech)
    HIS.write_json(out_dir / "summary.json", summary)

    # per-molecule CSV
    with open(out_dir / "per_molecule_valid.csv", "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["row_id", "y_raw", "r2_unseen", "pred_replay", "pred_his",
                      "ae_replay", "ae_his", "delta", "ae_replay_final", "ae_his_final",
                      "delta_final"])
        for i in rows:
            wtr.writerow([int(i), f"{y[i]:.9f}", int(masks["r2_unseen"][i]),
                          f"{replay['pred_best'][i]:.9f}", f"{his['pred_best'][i]:.9f}",
                          f"{ae_r[i]:.9f}", f"{ae_h[i]:.9f}", f"{delta[i]:.9f}",
                          f"{ae_rf[i]:.9f}", f"{ae_hf[i]:.9f}", f"{delta_final[i]:.9f}"])

    def md_table() -> str:
        lines = ["| group | n | REPLAY | HIS | delta | CI95 | win |",
                 "|---|---:|---:|---:|---:|---|---:|"]
        for name in GROUP_ORDER:
            gg = groups[name]
            ci = gg["paired"]["ci95"] if gg["paired"] else [float("nan")] * 2
            win = gg["paired"]["win_fraction_his_better"] if gg["paired"] else float("nan")
            lines.append(f"| {name} | {gg['n']} | {gg['replay_mae']:.6f} | {gg['his_mae']:.6f} | "
                         f"{gg['delta']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {win:.3f} |")
        return "\n".join(lines)

    (out_dir / "groups.md").write_text(
        "# HIS-v0 groups (official valid, best-by-valid checkpoint)\n\n"
        "`delta = HIS - REPLAY`; negative = HIS better.\n\n" + md_table() + "\n")
    def _f(v):
        return "n/a" if v is None else f"{v:.6f}"

    (out_dir / "main_table.md").write_text(
        "# HIS-v0 main table\n\n"
        f"- REPLAY_s0 best valid **{ep_r['best_valid_mae']:.6f}** @ epoch {ep_r['best_epoch']}; "
        f"final {ep_r['final_valid_mae']:.6f}; W20 {_f(ep_r['W20'])}; T100 {_f(ep_r['T100'])}\n"
        f"- HIS_s0 best valid **{ep_h['best_valid_mae']:.6f}** @ epoch {ep_h['best_epoch']}; "
        f"final {ep_h['final_valid_mae']:.6f}; W20 {_f(ep_h['W20'])}; T100 {_f(ep_h['T100'])}\n"
        f"- lambda = {ep_h['lambda']:.6e}; updates = {ep_h['updates']}; "
        f"GPU·h = {ep_r['gpu_hours']:.3f} + {ep_h['gpu_hours']:.3f}\n\n" + md_table() + "\n")
    (out_dir / "contrasts.md").write_text(
        "# HIS-v0 paired contrasts\n\n`delta = HIS - REPLAY` (negative = HIS better), "
        f"bootstrap {N_BOOT} resamples (seed {BOOTSTRAP_SEED}) over valid molecules only.\n\n"
        "| contrast | n | mean delta | CI95 | win | drop biggest HIS win | drop top-10 abs |\n"
        "|---|---:|---:|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {k} | {v['n']} | {v['mean_delta']:+.6f} | "
            f"[{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}] | {v['win_fraction_his_better']:.3f} | "
            f"{v.get('delta_drop_biggest_his_win', float('nan')):+.6f} | "
            f"{v.get('delta_drop_top10_abs', float('nan')):+.6f} |"
            for k, v in contrasts.items()) + "\n")

    print(f"REPLAY best {ep_r['best_valid_mae']:.6f} | HIS best {ep_h['best_valid_mae']:.6f} | "
          f"delta {groups['all']['delta']:+.6f}")
    print(f"ordinary {g['ordinary_y_gt_-3']:+.6f} | r2-unseen ord {g['r2_unseen_ordinary']:+.6f} | "
          f"minus row172 {g['all_minus_row172']:+.6f} | minus worst10 {g['all_minus_worst10']:+.6f}")
    print("W20 {} | T100 {} | final {:+.6f}".format(
        "n/a" if windows["W20"] is None else f"{windows['W20']:+.6f}",
        "n/a" if windows["T100"] is None else f"{windows['T100']:+.6f}",
        windows["final_valid"]))
    print(f"h3 ratio REPLAY/HIS (best) = {mech['replay']['best_h_stats']['h3_ratio']:.4f} / "
          f"{mech['his']['best_h_stats']['h3_ratio']:.4f}")
    print(f"VERDICT = {verdict}  ({verdict_reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
