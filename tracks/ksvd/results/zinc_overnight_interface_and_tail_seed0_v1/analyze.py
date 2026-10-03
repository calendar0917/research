"""Local analysis for zinc-overnight-interface-and-tail-seed0-v1.

Reads the pulled per-arm prediction artifacts (fit + internal dev only; no
official-valid/test) and produces the main tables, group contributions, paired
bootstrap CIs (shared indices), extreme-row sensitivity and the
interpretation witnesses.

Usage:
    uv run python analyze.py --stage 1
    uv run python analyze.py --stage 2
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TRACK = ROOT / "tracks/ksvd"
RESULTS = TRACK / "results/zinc_overnight_interface_and_tail_seed0_v1"
FACTORIAL = TRACK / "results/zinc_structure_semantic_factorial_seed0_v1"
ZERO_BINDING = TRACK / "results/zinc_zero_binding_baseline_seed0_v1"

BOOT_SEED = 20261003
N_BOOT = 1000
DELTA = 0.003
GROUP_KEYS = ("k0", "k-1", "kle-2", "kle-3")
GROUP_NAMES = {"k0": "k=0", "k-1": "k=-1", "kle-2": "k<=-2", "kle-3": "k<=-3"}


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def group_masks(k: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k0": k == 0,
        "k-1": k == -1,
        "kle-2": k <= -2,
        "kle-3": k <= -3,
    }


def load_dev():
    with np.load(TRACK / "results/zinc_full_cycle_target_decomposition_v1/target_decomposition.npz") as z:
        y = np.asarray(z["y"], np.float64)
        k = np.asarray(z["k"], np.int64)
    prep = np.load(TRACK / "results/zinc_joint_dictionary_decision_v1/prep/fold_objects.npz", allow_pickle=False)
    fit_idx = np.asarray(prep["fit_idx"], np.int64)
    dev_idx = np.asarray(prep["dev_idx"], np.int64)
    return y, k, fit_idx, dev_idx


def load_arm(name: str, out_dir: Path) -> dict[str, Any]:
    preds = dict(np.load(out_dir / f"{name}_predictions.npz", allow_pickle=False))
    meta = json.loads((out_dir / f"{name}.json").read_text())
    b = float(preds["b"])
    return {
        "name": name,
        "fit_raw": np.asarray(preds["fit_raw"], np.float64),
        "dev_raw": np.asarray(preds["dev_raw"], np.float64),
        "b": b,
        "meta": meta,
    }


def load_b_reference() -> dict[str, Any]:
    replay = np.load(ZERO_BINDING / "phaseA_sm_replay.npz")
    meta = json.loads((FACTORIAL / "S_M.json").read_text())
    b = float(meta["calibration"]["b"])
    return {
        "name": "B",
        "fit_raw": np.asarray(replay["fit_raw"], np.float64),
        "dev_raw": np.asarray(replay["dev_raw"], np.float64),
        "b": b,
        "meta": {"note": "compressed S_M reference; calibration b from factorial S_M.json"},
    }


def metrics(arm: Mapping[str, Any], y: np.ndarray, k: np.ndarray, fit_idx: np.ndarray, dev_idx: np.ndarray):
    y_fit, y_dev = y[fit_idx], y[dev_idx]
    k_dev = k[dev_idx]
    fit_raw = arm["fit_raw"]
    dev_raw = arm["dev_raw"]
    b = float(arm["b"])
    out: dict[str, Any] = {
        "fit_raw": float(np.mean(np.abs(fit_raw - y_fit))),
        "fit_cal": float(np.mean(np.abs(fit_raw + b - y_fit))),
        "dev_raw": float(np.mean(np.abs(dev_raw - y_dev))),
        "dev_cal": float(np.mean(np.abs(dev_raw + b - y_dev))),
        "b": b,
    }
    masks = group_masks(k_dev)
    out["dev_g0_raw"] = float(np.mean(np.abs(dev_raw[masks["k0"]] - y_dev[masks["k0"]])))
    out["dev_g0_cal"] = float(np.mean(np.abs(dev_raw[masks["k0"]] + b - y_dev[masks["k0"]])))
    out["cal_fit_dev_gap"] = out["dev_cal"] - out["fit_cal"]
    out["groups"] = {}
    for key, mask in masks.items():
        res = np.abs(dev_raw[mask] + b - y_dev[mask])
        out["groups"][GROUP_NAMES[key]] = {
            "n": int(mask.sum()),
            "raw_mae": float(np.mean(np.abs(dev_raw[mask] - y_dev[mask]))),
            "cal_mae": float(np.mean(res)),
            "contribution_cal": float(res.sum() / len(dev_idx)),
        }
    return out


def paired_gain(
    ref_res: np.ndarray,
    cand_res: np.ndarray,
    strata: Sequence[np.ndarray],
    *,
    seed: int = BOOT_SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """gain = mean|ref residual| - mean|cand residual|; shared resample indices."""
    rng = np.random.default_rng(int(seed))
    point = float(np.abs(ref_res).mean() - np.abs(cand_res).mean())
    rows = [np.where(m)[0] for m in strata]
    vals = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        idx = np.concatenate([rng.choice(r, size=r.size, replace=True) for r in rows])
        vals[b] = np.abs(ref_res[idx]).mean() - np.abs(cand_res[idx]).mean()
    return {
        "point": point,
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "n_boot": int(n_boot),
        "seed": int(seed),
        "shared_indices_across_contrasts": True,
    }


def bootstrap_contrasts(
    residuals_cal: Mapping[str, np.ndarray],
    residuals_raw: Mapping[str, np.ndarray],
    strata: Sequence[np.ndarray],
    contrasts: Sequence[tuple[str, str]],
    y_dev: np.ndarray,
    k_dev: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ref, cand in contrasts:
        for endpoint, res in (("cal", residuals_cal), ("raw", residuals_raw)):
            out[f"gain_{cand}_vs_{ref}_{endpoint}"] = paired_gain(res[ref], res[cand], strata)
    g0 = k_dev == 0
    for ref, cand in contrasts:
        out[f"gain_{cand}_vs_{ref}_g0_cal"] = paired_gain(
            residuals_cal[ref][g0], residuals_cal[cand][g0], [np.ones(int(g0.sum()))]
        )
    return out


def witnesses(residual_a: np.ndarray, residual_b: np.ndarray, strata: Sequence[np.ndarray]) -> dict[str, Any]:
    out = {}
    out["identical"] = paired_gain(residual_a, residual_a, strata)
    out["swap_sign_mirror"] = {
        "forward": paired_gain(residual_a, residual_b, strata)["point"],
        "backward": paired_gain(residual_b, residual_a, strata)["point"],
    }
    shifted = residual_a - 0.5
    out["constant_shift_0p5"] = paired_gain(residual_a, shifted, strata)
    return out


def extreme_rows(
    residuals: Mapping[str, np.ndarray],
    arms: Mapping[str, Mapping[str, Any]],
    y_dev: np.ndarray,
    k_dev: np.ndarray,
    dev_idx: np.ndarray,
    top: int = 5,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, res in residuals.items():
        order = np.argsort(-np.abs(res))[: int(top)]
        rows = []
        for i in order:
            rows.append(
                {
                    "dev_row": int(i),
                    "global_index": int(dev_idx[i]),
                    "k": int(k_dev[i]),
                    "y": float(y_dev[i]),
                    "pred_cal": float(y_dev[i] - res[i]),
                    "residual": float(res[i]),
                }
            )
        worst = int(np.argmax(np.abs(res)))
        without = np.delete(res, worst)
        out[name] = {
            "top": rows,
            "worst_row": int(worst),
            "mae_full": float(np.mean(np.abs(res))),
            "mae_without_worst_row": float(np.mean(np.abs(without))),
            "worst_row_absolute_share": float(abs(res[worst]) / np.abs(res).sum()),
        }
    return out


def contribution_identity_ok(metrics_by_arm: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for name, m in metrics_by_arm.items():
        s = sum(v["contribution_cal"] for v in m["groups"].values())
        out[name] = {
            "sum": float(s),
            "overall": float(m["dev_cal"]),
            "abs_diff": float(abs(s - m["dev_cal"])),
            "pass": bool(abs(s - m["dev_cal"]) <= 1e-12),
        }
    return out


def analyze_stage1(out_dir: Path) -> dict[str, Any]:
    y, k, fit_idx, dev_idx = load_dev()
    k_dev = k[dev_idx]
    arm_names = ["R_CS", "R_SM", "R_SJ", "R_DM", "R_DJ"]
    arms = {"B": load_b_reference()}
    for name in arm_names:
        arms[name] = load_arm(name, out_dir)
    metrics_by_arm = {name: metrics(a, y, k, fit_idx, dev_idx) for name, a in arms.items()}
    residuals_cal = {name: a["dev_raw"] + a["b"] - y[dev_idx] for name, a in arms.items()}
    residuals_raw = {name: a["dev_raw"] - y[dev_idx] for name, a in arms.items()}
    strata = [k_dev == 0, k_dev == -1, k_dev <= -2]
    contrasts = [
        ("B", "R_CS"),
        ("B", "R_SM"),
        ("B", "R_SJ"),
        ("B", "R_DM"),
        ("B", "R_DJ"),
        ("R_CS", "R_SM"),
        ("R_CS", "R_SJ"),
        ("R_CS", "R_DM"),
        ("R_CS", "R_DJ"),
        # J - M effect per coding (gain of J over M)
        ("R_SM", "R_SJ"),
        ("R_DM", "R_DJ"),
        # sparse - dense at the same block
        ("R_DM", "R_SM"),
        ("R_DJ", "R_SJ"),
    ]
    boot = bootstrap_contrasts(residuals_cal, residuals_raw, strata, contrasts, y[dev_idx], k_dev)
    # interaction of coding form with J/M (difference of gains)
    g_sparse = boot["gain_R_SJ_vs_R_SM_cal"]["point"]
    g_dense = boot["gain_R_DJ_vs_R_DM_cal"]["point"]
    boot["interaction_coding_x_joint_cal"] = {
        "definition": "gain(J vs M | sparse) - gain(J vs M | dense)",
        "point": float(g_sparse - g_dense),
        "j_minus_m_sparse": float(g_sparse),
        "j_minus_m_dense": float(g_dense),
    }
    # selection rule
    structural = ["R_SM", "R_SJ", "R_DM", "R_DJ"]
    better = [n for n in structural if metrics_by_arm[n]["dev_g0_cal"] < metrics_by_arm["B"]["dev_g0_cal"]]
    if better:
        selected = min(better, key=lambda n: metrics_by_arm[n]["dev_g0_cal"])
        rule = "best structural arm with dev G0 cal better than B"
    else:
        selected = "R_SJ"
        rule = "fallback: none better than B -> R_SJ"
    order = ["R_SJ", "R_SM", "R_DJ", "R_DM"]
    selected = sorted(better, key=lambda n: (metrics_by_arm[n]["dev_g0_cal"], order.index(n)))[0] if better else "R_SJ"
    spec = {"R_SJ": ("sparse", "joint"), "R_SM": ("sparse", "marginal"), "R_DJ": ("dense", "joint"), "R_DM": ("dense", "marginal")}
    decision = {
        "selected_T": selected,
        "selected_spec": {"code_mode": spec[selected][0], "block_mode": spec[selected][1]},
        "rule": rule,
        "dev_g0_cal": {n: metrics_by_arm[n]["dev_g0_cal"] for n in structural},
        "B_dev_g0_cal": metrics_by_arm["B"]["dev_g0_cal"],
        "better_than_B": better,
        "exploratory": True,
        "note": "selection uses the reused internal dev; CI does not correct for adaptive selection",
    }
    write_json(out_dir / "stage2_decision.json", decision)
    write_json(out_dir / "selection.json", decision)
    payload = {
        "stage": 1,
        "metrics": metrics_by_arm,
        "bootstrap": boot,
        "witnesses": witnesses(residuals_cal["B"], residuals_cal["R_SJ"], strata),
        "extreme_rows": extreme_rows(residuals_cal, arms, y[dev_idx], k_dev, dev_idx),
        "contribution_identity": contribution_identity_ok(metrics_by_arm),
        "selection": decision,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    return payload


def analyze_stage2(out_dir: Path) -> dict[str, Any]:
    y, k, fit_idx, dev_idx = load_dev()
    k_dev = k[dev_idx]
    arms = {"B": load_b_reference()}
    for name in ("A0", "C", "T"):
        arms[name] = load_arm(name, out_dir)
    metrics_by_arm = {name: metrics(a, y, k, fit_idx, dev_idx) for name, a in arms.items()}
    residuals_cal = {name: a["dev_raw"] + a["b"] - y[dev_idx] for name, a in arms.items()}
    residuals_raw = {name: a["dev_raw"] - y[dev_idx] for name, a in arms.items()}
    strata = [k_dev == 0, k_dev == -1, k_dev <= -2]
    contrasts = [("A0", "C"), ("A0", "T"), ("C", "T"), ("B", "A0")]
    boot = bootstrap_contrasts(residuals_cal, residuals_raw, strata, contrasts, y[dev_idx], k_dev)
    gate: dict[str, Any] = {}
    for arm in ("C", "T"):
        g0 = boot[f"gain_{arm}_vs_A0_g0_cal"]
        overall_cal = boot[f"gain_{arm}_vs_A0_cal"]
        overall_raw = boot[f"gain_{arm}_vs_A0_raw"]
        metrics_arm = metrics_by_arm[arm]
        metrics_a0 = metrics_by_arm["A0"]
        g0_worsening = metrics_arm["dev_g0_cal"] - metrics_a0["dev_g0_cal"]
        passed = bool(
            g0["point"] >= DELTA
            and g0["ci95"][0] > 0
            and overall_cal["point"] >= DELTA
            and overall_raw["point"] > 0
            and g0_worsening <= 0.001
        )
        gate[arm] = {
            "gain_g0_cal": g0,
            "gain_overall_cal": overall_cal,
            "gain_overall_raw": overall_raw,
            "g0_cal_worsening": float(g0_worsening),
            "passed": passed,
            "rule": {
                "g0_cal_gain_ge": DELTA,
                "g0_ci_lower_gt": 0.0,
                "overall_cal_gain_ge": DELTA,
                "overall_raw_gain_gt": 0.0,
                "g0_cal_worsening_le": 0.001,
            },
        }
    payload = {
        "stage": 2,
        "metrics": metrics_by_arm,
        "bootstrap": boot,
        "gate": gate,
        "extreme_rows": extreme_rows(residuals_cal, arms, y[dev_idx], k_dev, dev_idx),
        "contribution_identity": contribution_identity_ok(metrics_by_arm),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "stage2_gate.json", payload)
    return payload


def write_stage1_tables(payload: Mapping[str, Any], out_dir: Path) -> None:
    m = payload["metrics"]
    order = ["B", "R_CS", "R_SM", "R_SJ", "R_DM", "R_DJ"]
    lines = ["arm,fit_raw,fit_cal,dev_g0_raw,dev_g0_cal,dev_overall_raw,dev_overall_cal,b,cal_fit_dev_gap"]
    for name in order:
        v = m[name]
        lines.append(
            f"{name},{v['fit_raw']:.6f},{v['fit_cal']:.6f},{v['dev_g0_raw']:.6f},{v['dev_g0_cal']:.6f},"
            f"{v['dev_raw']:.6f},{v['dev_cal']:.6f},{v['b']:.8f},{v['cal_fit_dev_gap']:.6f}"
        )
    (out_dir / "stage1_main_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    glines = ["arm,group,n,raw_mae,cal_mae,contribution_cal"]
    for name in order:
        for gname, g in m[name]["groups"].items():
            glines.append(f"{name},{gname},{g['n']},{g['raw_mae']:.6f},{g['cal_mae']:.6f},{g['contribution_cal']:.6f}")
    (out_dir / "stage1_group_table.csv").write_text("\n".join(glines) + "\n", encoding="utf-8")
    blines = ["contrast,endpoint,point,ci_lo,ci_hi"]
    for key, v in payload["bootstrap"].items():
        if not isinstance(v, dict) or "point" not in v:
            continue
        ci = v.get("ci95", ["", ""])
        lo = f"{ci[0]:.6f}" if isinstance(ci[0], (int, float)) else ""
        hi = f"{ci[1]:.6f}" if isinstance(ci[1], (int, float)) else ""
        blines.append(f"{key},,{v['point']:.6f},{lo},{hi}")
    (out_dir / "stage1_gain_table.csv").write_text("\n".join(blines) + "\n", encoding="utf-8")


def mechanism(out_dir: Path) -> dict[str, Any]:
    """Full fit+dev mechanism health for the pulled arm soups (CPU, no y)."""
    from tracks.ksvd.experiments.luyin16 import zinc_overnight_interface_and_tail_seed0_v1 as ov
    from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as zsf
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_sem108_v1 as sem
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
    import torch

    torch.set_num_threads(8)
    blob, decomp, train_data, fit_data, dev_data, fit_idx, dev_idx, y = ov.load_data()
    structural = ov.load_dictionary_objects()
    stats = ov.load_interface_stats()
    del blob, decomp
    y_fit = np.asarray(y[fit_idx], np.float64)
    spec_by_arm = dict(ov.ARM_SPEC)
    out: dict[str, Any] = {}
    for arm, (code_mode, block_mode) in spec_by_arm.items():
        if not (out_dir / f"{arm}_raw_soup_state.pt").exists():
            continue
        model = ov.build_interface_model(structural, stats, code_mode=code_mode, block_mode=block_mode, device=torch.device("cpu"))
        state = torch.load(out_dir / f"{arm}_raw_soup_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()
        entry: dict[str, Any] = {"code_mode": code_mode, "block_mode": block_mode}
        arm_meta = json.loads((out_dir / f"{arm}.json").read_text())
        b_arm = float(arm_meta["calibration"]["b"])
        for split, data in (("fit", fit_data), ("dev", dev_data)):
            delta_sq = 0.0
            delta_abs = 0.0
            delta_elems = 0
            bridge_abs = bridge_nnz = bridge_elems = 0
            coord_nnz = coord_elems = 0
            dcache = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for indices in ov.zftd.epoch_batches(len(data), ov.BATCH_SIZE, dcache, False):
                    batch = ov.zftd.make_batch(data, indices, torch.zeros(len(indices)), torch.device("cpu"))
                    coord = model.code(batch.dict_phi)
                    interface = sem.SEM108Model.semantic_interface(model, coord, batch, cm.C6_MASK, None)
                    delta = model.adapter(model.adapter_input(interface, coord, batch))
                    delta_sq += float(delta.double().pow(2).sum())
                    delta_abs += float(delta.abs().sum())
                    delta_elems += int(delta.numel())
                    h = model.fusion(interface) + delta
                    _E, aux = model.local_dictionary_bridge(h, return_aux=True)
                    alpha = aux["alpha"]
                    bridge_abs += float(alpha.abs().sum())
                    bridge_nnz += int((alpha != 0).sum())
                    bridge_elems += int(alpha.numel())
                    coord_nnz += int((coord[:, model.common_dim :] != 0).sum())
                    coord_elems += int(coord[:, model.common_dim :].numel())
            entry[split] = {
                "delta_rms": float(math.sqrt(delta_sq / max(delta_elems, 1))),
                "delta_absmean": float(delta_abs / max(delta_elems, 1)),
                "bridge_alpha_absmean": float(bridge_abs / max(bridge_elems, 1)),
                "bridge_alpha_nonzero_fraction": float(bridge_nnz / max(bridge_elems, 1)),
                "structural_alpha_nonzero_fraction": float(coord_nnz / max(coord_elems, 1)),
            }
        # J -> M inference switch (trained weights held fixed)
        flipped = "marginal" if block_mode == "joint" else ("joint" if block_mode == "marginal" else "none")
        if flipped != "none":
            model.block_mode = flipped
            pred, _ = ov._predict(model, dev_data, torch.as_tensor(y[dev_idx], dtype=torch.float32), torch.device("cpu"))
            entry["dev_cal_mae_block_flipped_to_" + flipped] = float(np.mean(np.abs(pred + b_arm - y[dev_idx])))
            model.block_mode = block_mode
        out[arm] = entry
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2))
    parser.add_argument("--mechanism", action="store_true")
    args = parser.parse_args(argv)
    out_dir = RESULTS
    if args.mechanism:
        payload = mechanism(out_dir)
        write_json(out_dir / "mechanism_health.json", payload)
        print(json.dumps(payload, indent=1)[:3000])
        return 0
    assert args.stage is not None, "--stage is required unless --mechanism"
    if args.stage == 1:
        payload = analyze_stage1(out_dir)
        write_stage1_tables(payload, out_dir)
        write_json(out_dir / "stage1_analysis.json", payload)
        sel = payload["selection"]
        print(f"[stage1] selected T={sel['selected_T']} spec={sel['selected_spec']}")
        for name in ("B", "R_CS", "R_SM", "R_SJ", "R_DM", "R_DJ"):
            v = payload["metrics"][name]
            print(f"  {name:5s} dev G0 cal={v['dev_g0_cal']:.6f} overall cal={v['dev_cal']:.6f} raw={v['dev_raw']:.6f} b={v['b']:.6f}")
        for key in ("gain_R_SJ_vs_R_SM_cal", "gain_R_DJ_vs_R_DM_cal", "gain_R_SJ_vs_R_DJ_cal", "gain_R_SM_vs_R_DM_cal"):
            print(" ", key, json.dumps(payload["bootstrap"][key]))
        print(" ", "interaction", json.dumps(payload["bootstrap"]["interaction_coding_x_joint_cal"]))
    else:
        payload = analyze_stage2(out_dir)
        for name in ("B", "A0", "C", "T"):
            v = payload["metrics"][name]
            print(f"  {name:4s} dev G0 cal={v['dev_g0_cal']:.6f} overall cal={v['dev_cal']:.6f} raw={v['dev_raw']:.6f} b={v['b']:.6f}")
        print(json.dumps(payload["gate"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
