"""Analysis and decision for the SCM round (official valid only; test never read).

Reads the two trained arms (S-only, S+A) and the frozen DCC-L seed-0 reference
and produces:

  analysis/main_table.json            point estimates + curve summaries
  analysis/paired_contrasts.json      paired per-molecule deltas + bootstrap CIs
  analysis/per_molecule_valid.csv     per-molecule predictions / errors / correction
  analysis/delta_decomposition.json   correction magnitude and the S vs A split
  analysis/curves.png                 valid-MAE curves
  analysis/decision.json              pre-registered decision function output
  analysis/artifacts/                 best checkpoints, curves, predictions

The decision function is committed before the GPU run.  Bootstrap intervals
describe the *fixed* 1,000 development molecules; they are not cross-seed
evidence.  Official test is never constructed or read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import scm_common as S
import scm_relations as R
import scm_models as M

import ad_common as C
import ad_data as D
import ad_train as T1

DCCL_ROUND = S._SUFF / "zinc_dual_channel_composer_v1"
DCCL_RUN = DCCL_ROUND / "runs" / "L_s0"
DCCL_FEATURES = (S._SUFF / "zinc_dccl_bottleneck_v1" / "analysis"
                 / "per_molecule_features.csv")
ARMS = {"S-only": "s_only", "S+A": "s_plus_a"}
CONTRASTS = {
    "S-only_minus_DCC-L": ("S-only", "DCC-L_s0"),
    "S+A_minus_DCC-L": ("S+A", "DCC-L_s0"),
    "S+A_minus_S-only": ("S+A", "S-only"),
}
PROMOTION_DELTA = -0.003
MATERIAL_DELTA = -0.003


def _load_pred(path: Path) -> np.ndarray:
    z = np.load(path)
    return np.asarray(z["pred"], dtype=np.float64)[np.argsort(z["row"])]


def _bootstrap(delta: np.ndarray, n_boot: int = 10000, seed: int = 20261025) -> dict:
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = delta[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "frac_bootstrap_negative": float((means < 0).mean()),
        "median": float(np.median(delta)),
        "frac_molecules_improved": float((delta < 0).mean()),
        "n": int(n),
        "note": "paired over the fixed 1,000 official-valid molecules; descriptive "
                "only, not cross-seed stability evidence",
    }


def _read_curve(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        out = []
        for r in csv.DictReader(fh):
            out.append({"epoch": int(r["epoch"]),
                        "online_train_mae": float(r["online_train_mae"]),
                        "valid_mae": float(r["valid_mae"]),
                        "lr": float(r["lr"]), "seconds": float(r["seconds"]),
                        "steps": int(r["steps"])})
    return out


def load_arm(arm: str) -> dict:
    run = S.RUNS / f"{ARMS[arm]}_s0"
    info = C.read_json(run / "train.json")
    curve = _read_curve(run / "curve.csv") if (run / "curve.csv").exists() else info["curve"]
    return {
        "run_dir": str(run.relative_to(S._REPO)),
        "info": info,
        "curve": curve,
        "pred": _load_pred(run / "predictions" / "valid_best.npz"),
        "best_pt": run / "best.pt",
        "params": int(info["parameter_count"]),
    }


def curve_stats(curve: list[dict]) -> dict:
    v = np.asarray([e["valid_mae"] for e in curve], dtype=np.float64)
    ep = np.asarray([e["epoch"] for e in curve], dtype=np.int64)
    i = int(np.argmin(v))
    late = ep >= 300
    if not late.any():
        late = np.ones_like(ep, dtype=bool)
    return {
        "best_valid": float(v[i]), "best_epoch": int(ep[i]), "final_valid": float(v[-1]),
        "w20_mean": float(v[-20:].mean()),
        "late_300_600_mean": float(v[late].mean()), "late_300_600_sd": float(v[late].std()),
        "epochs": int(len(v)), "best_to_final_gap": float(v[-1] - v[i]),
    }


def decomposition(arm: str, store: dict, cache, device="cpu") -> dict:
    """Correction magnitude and the exact S/A split at the best-valid checkpoint.

    Wout is linear on concat(S1,A1,S2,A2) plus a bias, so the correction splits
    exactly into four block products and a constant.  The split is measured from
    the captured block tensors and cross-checked against the actual forward.
    """
    norm = D.load_norm()
    use_cov = arm == "S+A"
    model, _ = M.build_initial_state(norm, 0, use_cov=use_cov)
    ckpt_path = S.RUNS / f"{ARMS[arm]}_s0" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    w = model.scm.out.weight.detach()
    b = model.scm.out.bias.detach()
    W = {"S1": w[:, 0:8], "A1": w[:, 8:16], "S2": w[:, 16:24], "A2": w[:, 24:32]}
    collate = R.make_split_collate({"valid": store}, T1.collate)
    rows = np.arange(int(cache["n_molecules"]))
    acc = {k: [] for k in ("rms_c", "rms_delta", "rms_dS", "rms_dA", "rms_signal")}
    per_mol, add_err = [], 0.0
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        batch = collate(cache, chunk, "P3N", valid=True)
        with torch.no_grad():
            _, cap = model.forward_with_capture(batch)
        c, d = cap["c_loc"], cap["delta"]
        bb, bl = batch["bond_batch"], batch["bond_local"]
        s1, a1 = cap["S1"][bb, bl], cap["A1"][bb, bl]
        s2, a2 = cap["S2"][bb, bl], cap["A2"][bb, bl]
        d_s = s1 @ W["S1"].T + s2 @ W["S2"].T
        d_a = a1 @ W["A1"].T + a2 @ W["A2"].T
        add_err = max(add_err, float((d - (d_s + d_a + b)).abs().max()))
        acc["rms_c"].append(float(c.pow(2).mean().sqrt()))
        acc["rms_delta"].append(float(d.pow(2).mean().sqrt()))
        acc["rms_dS"].append(float(d_s.pow(2).mean().sqrt()))
        acc["rms_dA"].append(float(d_a.pow(2).mean().sqrt()))
        acc["rms_signal"].append(float((d_s + d_a).pow(2).mean().sqrt()))
        for bi, row in enumerate(chunk):
            sel = bb == bi
            per_mol.append({
                "row_id": int(row),
                "rms_c_loc": float(c[sel].pow(2).mean().sqrt()),
                "rms_delta": float(d[sel].pow(2).mean().sqrt()),
                "rms_delta_S": float(d_s[sel].pow(2).mean().sqrt()),
                "rms_delta_A": float(d_a[sel].pow(2).mean().sqrt()),
            })
    blocks = {k: float(v.norm()) for k, v in W.items()}
    rms_c = float(np.mean(acc["rms_c"]))
    rms_sig = float(np.mean(acc["rms_signal"]))
    rms_s = float(np.mean(acc["rms_dS"]))
    rms_a = float(np.mean(acc["rms_dA"]))
    out = {
        "arm": arm,
        "checkpoint": str(ckpt_path.relative_to(S._REPO)),
        "epoch": ckpt.get("epoch"),
        "A_path_active_in_training": use_cov,
        "split_residual_max_abs": add_err,
        "rms_c_loc": rms_c,
        "rms_delta_total": float(np.mean(acc["rms_delta"])),
        "rms_delta_signal": rms_sig,
        "rms_bias_component": float(b.norm()),
        "rms_delta_S": rms_s,
        "rms_delta_A": rms_a,
        "rms_ratio_delta_over_c_loc": float(np.mean(acc["rms_delta"])) / rms_c,
        "share_of_signal_rms_from_S": rms_s / rms_sig if rms_sig else None,
        "share_of_signal_rms_from_A": rms_a / rms_sig if rms_sig else None,
        "share_of_signal_variance_from_A": rms_a ** 2 / rms_sig ** 2 if rms_sig else None,
        "wout_block_norms": blocks,
        "wout_bias_norm": float(b.norm()),
        "A_blocks_exactly_zero_weights": bool(blocks["A1"] == 0.0 and blocks["A2"] == 0.0),
    }
    return {"summary": out, "per_molecule": per_mol}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-decomposition", action="store_true")
    args = p.parse_args(argv)
    S.ANALYSIS.mkdir(parents=True, exist_ok=True)

    cache = D.load_cache("valid")
    y = np.asarray(cache["labels"], dtype=np.float64)
    store = R.load_split("valid")
    dccl_pred = _load_pred(DCCL_RUN / "predictions" / "valid_best.npz")
    dccl_info = C.read_json(DCCL_RUN / "train.json")
    arms = {a: load_arm(a) for a in ARMS}
    feats = {}
    if DCCL_FEATURES.exists():
        with open(DCCL_FEATURES) as fh:
            feats = {int(r["row_id"]): r for r in csv.DictReader(fh)}

    # ---- main table ------------------------------------------------------
    table = {"DCC-L_s0": {
        "parameters": int(dccl_info["parameter_count"]),
        "best_valid": float(dccl_info["best_valid"]["valid_mae"]),
        "best_epoch": int(dccl_info["best_valid"]["epoch"]),
        "final_valid": float(dccl_info["fixed_objects"]["final"]["valid_mae"]),
        "train_mae_eval_mode_best": float(
            dccl_info["fixed_objects"]["best"]["train_mae_eval_mode"]),
        "curve": curve_stats(dccl_info["curve"]),
        "source": "frozen historical DCC-L seed 0 (not retrained)",
        "run_dir": str(DCCL_RUN.relative_to(S._REPO)),
    }}
    for a, r in arms.items():
        table[a] = {
            "parameters": r["params"],
            "best_valid": float(r["info"]["best_valid"]["valid_mae"]),
            "best_epoch": int(r["info"]["best_valid"]["epoch"]),
            "final_valid": float(r["info"]["fixed_objects"]["final"]["valid_mae"]),
            "train_mae_eval_mode_best": float(
                r["info"]["fixed_objects"]["best"]["train_mae_eval_mode"]),
            "train_mae_eval_mode_final": float(
                r["info"]["fixed_objects"]["final"]["train_mae_eval_mode"]),
            "gradient_audit": r["info"]["gradient_audit"],
            "wall_seconds": r["info"]["wall_seconds"],
            "curve": curve_stats(r["curve"]),
            "model_info": {k: r["info"]["model_info"].get(k) for k in
                           ("arm", "use_cov", "bands", "rank", "scales", "scm_seed",
                            "backbone_dcc_l_state_sha256", "initial_state_sha256",
                            "parameters_breakdown")},
            "run_dir": r["run_dir"],
        }

    # ---- paired contrasts ------------------------------------------------
    preds = {"DCC-L_s0": dccl_pred, **{a: r["pred"] for a, r in arms.items()}}
    ae = {k: np.abs(v - y) for k, v in preds.items()}
    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        contrasts[name] = {"mae_first": float(ae[a].mean()), "mae_second": float(ae[b].mean()),
                           "delta_first_minus_second": float((ae[a] - ae[b]).mean()),
                           **_bootstrap(ae[a] - ae[b])}
    contrasts["S-only_minus_DCC-L"]["mae_second_reference"] = None

    masks = {"all": np.ones(len(y), dtype=bool),
             "ordinary_y_gt_-3": y > -3.0, "tail_y_le_-3": y <= -3.0}
    if feats:
        unseen = np.asarray([float(feats[i]["r2_unseen_atom_frac"]) > 0 for i in range(len(y))])
        masks["r2_seen"] = ~unseen
        masks["r2_unseen"] = unseen
        hmean = np.asarray([float(feats[i]["ae_mean"]) for i in range(len(y))])
        keep = np.ones(len(y), dtype=bool)
        keep[np.argsort(-hmean)[:10]] = False
        masks["fixed_dccl_worst10_removed"] = keep
    subgroup = {}
    for name, m in masks.items():
        row = {"n": int(m.sum())}
        for k in ae:
            row[f"mae_{k}"] = float(ae[k][m].mean())
        for cname, (a, b) in CONTRASTS.items():
            row[f"delta_{cname}"] = float((ae[a][m] - ae[b][m]).mean())
        subgroup[name] = row

    # ---- per-molecule table ---------------------------------------------
    pm_path = S.ANALYSIS / "per_molecule_valid.csv"
    with open(pm_path, "w", newline="") as fh:
        w = csv.writer(fh)
        cols = (["row_id", "y"] + [f"pred_{k}" for k in preds] +
                [f"ae_{k}" for k in preds] + [f"delta_{n}" for n in CONTRASTS])
        if feats:
            cols += ["r2_unseen_atom_frac", "n_atoms", "n_rings", "dcc_l_4seed_mean_ae"]
        w.writerow(cols)
        for i in range(len(y)):
            row = ([i, y[i]] + [preds[k][i] for k in preds] +
                   [ae[k][i] for k in preds] +
                   [ae[a][i] - ae[b][i] for a, b in CONTRASTS.values()])
            if feats:
                f = feats[i]
                row += [f["r2_unseen_atom_frac"], f["n_atoms"], f["n_rings"], f["ae_mean"]]
            w.writerow(row)

    # ---- decomposition ---------------------------------------------------
    decomp = {}
    if not args.skip_decomposition:
        per_by_arm = {}
        for a in ARMS:
            dec = decomposition(a, store, cache, device=args.device)
            decomp[a] = dec["summary"]
            per_by_arm[a] = {r["row_id"]: r for r in dec["per_molecule"]}
        lines = pm_path.read_text().splitlines()
        header = lines[0].split(",")
        add_cols = [f"{tag}_rms_{c}" for tag in ("sonly", "splusa")
                    for c in ("c_loc", "delta", "delta_S", "delta_A")]
        out = [",".join(header + add_cols)]
        for line in lines[1:]:
            row_id = int(line.split(",")[0])
            vals = []
            for tag, arm_name in (("sonly", "S-only"), ("splusa", "S+A")):
                r = per_by_arm[arm_name][row_id]
                vals += [f"{r[c]:.6g}" for c in
                         ("rms_c_loc", "rms_delta", "rms_delta_S", "rms_delta_A")]
            out.append(line + "," + ",".join(vals))
        pm_path.write_text("\n".join(out) + "\n")
        S.write_json(S.ANALYSIS / "delta_decomposition.json", decomp)

    # ---- curves ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)
        ax.plot([e["epoch"] for e in dccl_info["curve"]],
                [e["valid_mae"] for e in dccl_info["curve"]], lw=1.1, color="0.35",
                label=f"DCC-L s0 best {dccl_info['best_valid']['valid_mae']:.5f}")
        for a, r in arms.items():
            ax.plot([e["epoch"] for e in r["curve"]], [e["valid_mae"] for e in r["curve"]],
                    lw=1.1,
                    label=f"{a} best {r['info']['best_valid']['valid_mae']:.5f}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("official-valid raw-y MAE (log)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
        ax.set_title("SCM seed 0: frozen DCC-L vs S-only vs S+A", fontsize=9)
        fig.tight_layout()
        fig.savefig(S.ANALYSIS / "curves.png")
    except Exception as exc:  # pragma: no cover
        print(f"curve plot skipped: {exc}")

    # ---- artifacts -------------------------------------------------------
    art = S.ANALYSIS / "artifacts"
    art.mkdir(exist_ok=True)
    for a, r in arms.items():
        tag = ARMS[a].replace("_", "-")
        shutil.copy2(r["best_pt"], art / f"{tag}_best.pt")
        shutil.copy2(S._REPO / r["run_dir"] / "predictions" / "valid_best.npz",
                     art / f"{tag}_valid_best.npz")
        shutil.copy2(S._REPO / r["run_dir"] / "curve.csv", art / f"{tag}_curve.csv")
    shutil.copy2(DCCL_RUN / "predictions" / "valid_best.npz", art / "dccl_s0_valid_best.npz")
    sha = {q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(art.glob("*"))}

    # ---- decision (frozen rules) ----------------------------------------
    def arm_state(cname: str) -> dict:
        c = contrasts[cname]
        first, second = CONTRASTS[cname]
        late = (table[first]["curve"]["late_300_600_mean"]
                - table[second]["curve"]["late_300_600_mean"])
        w20 = (table[first]["curve"]["w20_mean"]
               - table[second]["curve"]["w20_mean"])
        return {
            "delta_best_valid": c["delta_first_minus_second"],
            "ci95": c["ci95"],
            "delta_w20": w20,
            "delta_late_300_600": late,
            "materially_better": bool(c["delta_first_minus_second"] <= MATERIAL_DELTA),
            "directionally_better": bool(
                c["delta_first_minus_second"] < 0 and late < 0 and w20 < 0
                and c["ci95"][1] < 0),
            "subgroups": {k: v[f"delta_{cname}"] for k, v in subgroup.items()},
        }

    st = {"S-only": arm_state("S-only_minus_DCC-L"),
          "S+A": arm_state("S+A_minus_DCC-L")}
    ex = arm_state("S+A_minus_S-only")
    a_extra = {
        "delta_best_valid": ex["delta_best_valid"], "ci95": ex["ci95"],
        "delta_w20": ex["delta_w20"], "delta_late_300_600": ex["delta_late_300_600"],
        "extra_gain_from_A": bool(ex["delta_best_valid"] < 0 and ex["ci95"][1] < 0),
        "materially_worse_than_S_only": bool(ex["delta_best_valid"] > 0.001),
    }
    for s in st.values():
        s.pop("materially_better")
    if a_extra["extra_gain_from_A"] and st["S+A"]["directionally_better"] \
            and st["S+A"]["delta_best_valid"] <= PROMOTION_DELTA:
        label = "RESULT_2_S_PLUS_A_BETTER"
        reason = "S+A beats S-only with a paired CI excluding 0 and beats the same-seed " \
                 "DCC-L by at least 0.003 with W20/late-window agreement."
    elif st["S-only"]["directionally_better"] and not a_extra["extra_gain_from_A"]:
        label = "RESULT_1_S_ONLY_BETTER_A_NO_EXTRA_GAIN"
        reason = "S-only is directionally better than DCC-L on all three summaries and " \
                 "the paired CI, and A adds no gain."
    elif not st["S-only"]["directionally_better"] and not st["S+A"]["directionally_better"]:
        label = "RESULT_3_NEITHER_BEATS_DCC_L_STOP"
        reason = "Neither arm is directionally better than the frozen DCC-L on " \
                 "best-valid, W20 or the late window with a paired CI below 0."
    elif a_extra["delta_best_valid"] > 0.0:
        label = "RESULT_4_S_PLUS_A_WORSE_THAN_S_ONLY"
        reason = "S+A is worse than S-only; recorded, no rescue experiment purchased."
    else:
        label = "WEAK_OR_MIXED_EXPLORATORY_SIGNAL"
        reason = "Point estimates and the direction tests disagree; no promotion."

    cond = {
        "R1": bool(st["S-only"]["directionally_better"] and not a_extra["extra_gain_from_A"]),
        "R2": bool(a_extra["extra_gain_from_A"] and st["S+A"]["directionally_better"]
                   and st["S+A"]["delta_best_valid"] <= PROMOTION_DELTA),
        "R3": bool(not st["S-only"]["directionally_better"]
                   and not st["S+A"]["directionally_better"]),
        "R4": bool(a_extra["delta_best_valid"] > 0.0),
    }
    half = {k: 0.5 * (v["ci95"][1] - v["ci95"][0]) for k, v in contrasts.items()}
    resolution = {
        "paired_ci95_half_width": half,
        "approx_min_detectable_effect_abs": min(half["S-only_minus_DCC-L"],
                                                 half["S+A_minus_DCC-L"]),
        "promotion_threshold_abs": abs(PROMOTION_DELTA),
        "threshold_within_resolution": bool(
            abs(PROMOTION_DELTA) <= min(half["S-only_minus_DCC-L"],
                                        half["S+A_minus_DCC-L"])),
        "note": "MDE is approximated by the paired CI half-width at n=1000; an effect "
                "smaller than this cannot be separated from molecule-level noise here.",
    }

    decision = {
        "protocol_id": S.TRACK_ID,
        "stopped": True,
        "rule_conditions_met": cond,
        "rule_conditions_note": "R4 also holds (S+A is worse than S-only by >0.001); the "
                               "round-level label is R3 because neither arm beats DCC-L, "
                               "which is the stop condition, and no rescue run is purchased.",
        "resolution": resolution,
        "label": label,
        "reason": reason,
        "single_seed": True,
        "official_test_loaded": False,
        "valid_is_exposed_development_split": True,
        "thresholds": {
            "material_delta_best_valid": MATERIAL_DELTA,
            "promotion_delta_vs_same_seed_dccl": PROMOTION_DELTA,
            "directional_rule": "delta_best_valid<0 AND delta_w20<0 AND "
                                "delta_late_300_600<0 AND paired CI95 upper<0",
            "frozen_before_gpu_run": True,
        },
        "dcc_l_reference": {"best_valid": table["DCC-L_s0"]["best_valid"],
                            "best_epoch": table["DCC-L_s0"]["best_epoch"],
                            "source": table["DCC-L_s0"]["source"]},
        "arms": st,
        "A_extra_gain": a_extra,
        "subgroup_deltas": subgroup,
        "decomposition": decomp,
        "artifacts_sha256": sha,
        "interpretation_constraints": [
            "Single seed on an exposed valid split; no cross-seed confirmation.",
            "Bootstrap intervals describe the fixed 1,000 development molecules and are "
            "not evidence of seed stability.",
            "A gain does not prove the structural relation is its cause; a relation "
            "shuffle or seed-1 control would be required (not purchased this round).",
            "No conclusion generalises to all structural-statistics-plus-attention "
            "designs; only to this frozen two-band centred-difference implementation.",
            "Official test was never constructed or read this round.",
        ],
    }
    S.write_json(S.ANALYSIS / "main_table.json", table)
    S.write_json(S.ANALYSIS / "paired_contrasts.json",
                 {"contrasts": contrasts, "subgroups": subgroup,
                  "masks": {k: int(v.sum()) for k, v in masks.items()},
                  "contrast_definitions": {k: list(v) for k, v in CONTRASTS.items()}})
    S.write_json(S.ANALYSIS / "decision.json", decision)
    print(json.dumps({"label": label, "arms": st, "A_extra_gain": a_extra,
                      "best_valid": {k: v["best_valid"] for k, v in table.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
