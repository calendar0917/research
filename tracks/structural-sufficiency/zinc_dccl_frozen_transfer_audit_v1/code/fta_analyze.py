"""Aggregate the frozen probe scores, the group tables and the pre-registered verdict.

Reads only artefacts written by ``fta_export.py`` / ``fta_probe.py`` /
``fta_score.py``.  No label access beyond the frozen Monitor table.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fta_common as K

TRUNK_SEEDS = (0, 1)
HEAD_INITS = (0, 1)
BLOCKS = ("atom_global", "local_global", "full")
CONDS = ("fit4k_s0", "fit4k_s1", "fit8k_s0", "fit8k_s1")


def bootstrap_paired(d: np.ndarray, n_boot: int, seed: int, ci: float = 95.0) -> dict:
    rng = np.random.default_rng(int(seed))
    n = len(d)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, (100.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 - (100.0 - ci) / 2.0))
    return {"mean": float(d.mean()), "ci95": [lo, hi], "n": int(n),
            "frac_positive": float((d > 0).mean())}


def group_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    y = df.y.to_numpy()
    ordinary = df.ordinary_nopenalty.to_numpy().astype(bool)
    # PROTOCOL §5: the two `any_unseen` *group* filters are
    # (any-unseen w.r.t. that fit set) AND (no artificial ring penalty).
    u4 = df.any_unseen_fit4k.to_numpy().astype(bool) & ordinary
    u8 = df.any_unseen_fit8k.to_numpy().astype(bool) & ordinary
    un4 = df.unseen_frac_fit4k.to_numpy() > 0
    un8 = df.unseen_frac_fit8k.to_numpy() > 0
    low4 = (~un4) & (df.supported_frac_fit4k.to_numpy() < 0.999)
    low8 = (~un8) & (df.supported_frac_fit8k.to_numpy() < 0.999)
    full4 = df.supported_frac_fit4k.to_numpy() >= 0.999
    full8 = df.supported_frac_fit8k.to_numpy() >= 0.999
    return {
        "overall": np.ones(len(df), dtype=bool),
        "ordinary_nopenalty": ordinary,
        "penalised_ring_gt6": ~ordinary,
        "any_unseen_fit4k": u4,
        "any_unseen_fit8k": u8,
        "low_support_fit4k": low4,
        "low_support_fit8k": low8,
        "full_support_fit4k": full4,
        "full_support_fit8k": full8,
        "newly_covered_4k_to_8k": un4 & ~un8,
        "still_unseen_both": un4 & un8,
        "extreme_tail": y <= -3.0,
        "non_tail": y > -3.0,
        "drop_fixed_top10": ~df.in_drop_fixed_top10.to_numpy().astype(bool),
    }


def main() -> int:
    t0 = time.perf_counter()
    cfg = K.load_config()
    metrics = json.loads((K.ANALYSIS / "probe_monitor_metrics.json").read_text())
    probe = json.loads((K.ANALYSIS / "probe_freeze.json").read_text())
    preds = np.load(K.ANALYSIS / "probe_monitor_predictions.npz")
    df = pd.read_csv(K.ANALYSIS / "per_molecule_monitor.csv")
    y = df.y.to_numpy()
    masks = group_masks(df)

    by_cell = {(c["cell"], c["checkpoint"]): c for c in metrics["cells"]}
    ae_cache: dict[str, np.ndarray] = {}

    def ae(cell: str, tag: str = "best") -> np.ndarray:
        key = (cell, tag)
        if key not in ae_cache:
            p = np.asarray(preds[f"{cell}__{tag}"], dtype=np.float64)
            ae_cache[key] = np.abs(p - y)
        return ae_cache[key]

    def cell_of(fit: str, seed: int, block: str, init: int) -> str:
        return f"{fit}_s{seed}__{block}__i{init}"

    # ------------------------------------------------------------------ deltas
    delta = {}      # (block, seed, init) -> Δ
    for block, seed, init in itertools.product(BLOCKS, TRUNK_SEEDS, HEAD_INITS):
        d = ae(cell_of("fit4k", seed, block, init)).mean() - ae(cell_of("fit8k", seed, block, init)).mean()
        delta[(block, seed, init)] = float(d)
    delta_seed = {block: {s: float(np.mean([delta[(block, s, i)] for i in HEAD_INITS]))
                          for s in TRUNK_SEEDS} for block in BLOCKS}
    delta_init = {block: {i: float(np.mean([delta[(block, s, i)] for s in TRUNK_SEEDS]))
                          for i in HEAD_INITS} for block in BLOCKS}
    delta_all = {block: float(np.mean([delta[(block, s, i)] for s in TRUNK_SEEDS
                                       for i in HEAD_INITS])) for block in BLOCKS}

    # final-checkpoint robustness (descriptive only)
    delta_final = {}
    for block, seed, init in itertools.product(BLOCKS, TRUNK_SEEDS, HEAD_INITS):
        d = (ae(cell_of("fit4k", seed, block, init), "final").mean()
             - ae(cell_of("fit8k", seed, block, init), "final").mean())
        delta_final[(block, seed, init)] = float(d)
    delta_all_final = {block: float(np.mean([delta_final[(block, s, i)] for s in TRUNK_SEEDS
                                             for i in HEAD_INITS])) for block in BLOCKS}

    # monitor MAE levels per condition (mean over the 4 head cells of a block)
    levels = {}
    for block in BLOCKS:
        levels[block] = {}
        for fit, seed in K.CONDITIONS:
            cid = f"{fit}_s{seed}"
            per = [float(by_cell[(cell_of(fit, seed, block, i), "best")]["monitor_mae"])
                   for i in HEAD_INITS]
            levels[block][cid] = {"per_init": per, "mean": float(np.mean(per))}

    # ------------------------------------------------------------------ groups
    group_table = {}
    for block in BLOCKS:
        group_table[block] = {}
        for gname, m in masks.items():
            if m.sum() == 0:
                continue
            row = {"n": int(m.sum())}
            for fit, seed in K.CONDITIONS:
                cid = f"{fit}_s{seed}"
                per_cell = []
                for init in HEAD_INITS:
                    per_cell.append(float(ae(cell_of(fit, seed, block, init))[m].mean()))
                row[cid] = {"per_init": per_cell, "mean": float(np.mean(per_cell))}
            row["delta_seed"] = {
                str(s): float(np.mean([row[f"fit4k_s{s}"]["per_init"][i]
                                       - row[f"fit8k_s{s}"]["per_init"][i]
                                       for i in range(len(HEAD_INITS))]))
                for s in TRUNK_SEEDS}
            row["delta_cell"] = {
                f"s{s}_i{i}": float(row[f"fit4k_s{s}"]["per_init"][i]
                                    - row[f"fit8k_s{s}"]["per_init"][i])
                for s in TRUNK_SEEDS for i in range(len(HEAD_INITS))}
            row["delta_mean"] = float(np.mean(list(row["delta_cell"].values())))
            base4 = float(np.mean([row[f"fit4k_s{s}"]["mean"] for s in TRUNK_SEEDS]))
            row["delta_relative"] = row["delta_mean"] / max(base4, 1e-9)
            group_table[block][gname] = row

    # paired bootstrap on the full block: per-molecule mean over the four cells
    boot = {}
    for gname, m in masks.items():
        if m.sum() == 0:
            continue
        d = np.zeros(int(m.sum()), dtype=np.float64)
        for seed in TRUNK_SEEDS:
            for init in HEAD_INITS:
                d += (ae(cell_of("fit4k", seed, "full", init))[m]
                      - ae(cell_of("fit8k", seed, "full", init))[m])
        d /= 4.0
        boot[gname] = bootstrap_paired(
            d, n_boot=int(cfg["evaluation"]["bootstrap"]["n_boot"]),
            seed=int(cfg["evaluation"]["bootstrap"]["seed"]),
            ci=float(cfg["evaluation"]["bootstrap"]["ci"]))

    # --------------------------------------------------------------- controls
    recovery = {}
    for fit, seed in K.CONDITIONS:
        cid = f"{fit}_s{seed}"
        cell = f"{cid}__full__i0__recovery"
        frozen = by_cell[(cell, "best")]
        raw = np.asarray(preds[f"raw_{cid}"], dtype=np.float64)
        pred_mon = np.asarray(preds[f"{cell}__best"], dtype=np.float64)
        recovery[cid] = {
            "probe_sel_l1_vs_trunk_pred": frozen["probe_sel_l1_frozen"],
            "monitor_mae_vs_trunk_pred": float(np.abs(pred_mon - raw).mean()),
            "monitor_mae_vs_y": frozen["monitor_mae"],
            "trunk_monitor_mae_vs_y": float(np.abs(raw - y).mean()),
        }
    null_cell = "fit8k_s0__full__i0__null"
    const_pred_mae = float(np.abs(y - np.median(y)).mean())
    label_null = {
        "monitor_mae_vs_y": by_cell[(null_cell, "best")]["monitor_mae"],
        "probe_sel_l1_frozen": by_cell[(null_cell, "best")]["probe_sel_l1_frozen"],
        "constant_median_predictor_monitor_mae": const_pred_mae,
        "y_monitor_mean_abs": float(np.abs(y).mean()),
        "permutation": probe["label_permutation"],
    }

    # ----------------------------------------------------------------- gates
    u1 = {cid: v["probe_sel_l1_vs_trunk_pred"] for cid, v in recovery.items()}
    full_cells = [cell_of(fit, s, "full", i) for fit, s in K.CONDITIONS for i in HEAD_INITS]
    train_best = {c["id"]: c["train_l1_at_best"] for c in probe["cells"]
                  if c["kind"] in ("main", "repro") and c["block"] == "full"}
    u3 = float(max(abs(delta[("full", s, 0)] - delta[("full", s, 1)]) for s in TRUNK_SEEDS))
    repro = probe["reproducibility"]
    gates = {
        "U1_recovery_max_probe_sel_l1": max(u1.values()),
        "U1_threshold": float(cfg["gates"]["U1_recovery_max_mae"]),
        "U1_pass": bool(max(u1.values()) <= float(cfg["gates"]["U1_recovery_max_mae"])),
        "U2_train_fit_max_l1": max(train_best.values()),
        "U2_threshold": float(cfg["gates"]["U2_train_fit_max_l1"]),
        "U2_pass": bool(max(train_best.values()) <= float(cfg["gates"]["U2_train_fit_max_l1"])),
        "U3_init_instability_max_abs": u3,
        "U3_threshold": float(cfg["gates"]["U3_init_instability_max_abs"]),
        "U3_pass": bool(u3 <= float(cfg["gates"]["U3_init_instability_max_abs"])),
        "U4_reproducibility": repro,
        "U4_pass": bool(repro is not None and repro["abs_diff"] <= float(cfg["gates"]["U4_repro_tol"])),
    }
    instrument_ok = all(gates[k] for k in ("U1_pass", "U2_pass", "U3_pass", "U4_pass"))

    raw_gain = float(metrics["raw_gain_fit4k_minus_fit8k_monitor"])
    t1 = bool(delta_seed["full"][0] >= float(cfg["gates"]["T1_delta_min"])
              and delta_seed["full"][1] >= float(cfg["gates"]["T1_delta_min"]))
    t2_detail = {g: {str(s): group_table["full"][g]["delta_seed"][str(s)] for s in TRUNK_SEEDS}
                 for g in ("any_unseen_fit4k", "any_unseen_fit8k")}
    t2 = bool(all(v > 0 for v in t2_detail["any_unseen_fit4k"].values())
              and all(v > 0 for v in t2_detail["any_unseen_fit8k"].values()))
    t3 = bool(all(group_table["full"]["non_tail"]["delta_seed"][str(s)]
                  >= float(cfg["gates"]["T3_non_tail_delta_min"]) for s in TRUNK_SEEDS))
    threshold = {
        "T1": t1, "T2": t2, "T3": t3,
        "T1_delta_seed": {str(s): delta_seed["full"][s] for s in TRUNK_SEEDS},
        "T2_group_delta_seed": t2_detail,
        "T3_non_tail_delta_seed": {str(s): group_table["full"]["non_tail"]["delta_seed"][str(s)]
                                   for s in TRUNK_SEEDS},
    }
    sign_flip = bool(np.sign(delta_seed["full"][0]) != np.sign(delta_seed["full"][1]))
    rocs = bool(raw_gain >= float(cfg["gates"]["raw_gain_min_for_rocs"])
                and (delta_all["full"] < float(cfg["gates"]["rocs_delta_max"]) or sign_flip))

    if not instrument_ok:
        verdict = "PROBE_UNINFORMATIVE"
        rule = "instrument gate failed (U1-U4)"
    elif t1 and t2 and t3:
        verdict = "TRUNK_TRANSFER_SIGNAL"
        rule = "T1 & T2 & T3 all pass"
    elif rocs:
        verdict = "READOUT_OR_COADAPTATION_SUSPECTED"
        rule = ("raw_gain >= 0.025 and (delta_all < 0.005 or trunk-seed sign flip); "
                "this is NOT a causal attribution to the reader")
    else:
        verdict = "UNRESOLVED"
        rule = "neither the transfer signal nor the readout/co-adaptation condition is met"

    # recovered fraction of the raw gain and probe-vs-raw level gaps (reporting only)
    raw_mae = {cid: float(metrics["raw"][cid]["monitor_mae"]) for cid in CONDS}
    recovered = {
        "all": float(delta_all["full"] / raw_gain) if raw_gain else float("nan"),
        **{f"s{s}": float(delta_seed["full"][s] / raw_gain) if raw_gain else float("nan")
           for s in TRUNK_SEEDS},
    }
    level_gap = {}
    for cid in CONDS:
        probe_mean = float(levels["full"][cid]["mean"])
        level_gap[cid] = {"probe_best_mean": probe_mean, "raw_final": raw_mae[cid],
                          "probe_minus_raw": probe_mean - raw_mae[cid]}
    block_levels = {}
    for block in BLOCKS:
        b4 = float(np.mean([levels[block][f"fit4k_s{s}"]["mean"] for s in TRUNK_SEEDS]))
        b8 = float(np.mean([levels[block][f"fit8k_s{s}"]["mean"] for s in TRUNK_SEEDS]))
        block_levels[block] = {"mae_4k_mean": b4, "mae_8k_mean": b8,
                               "delta_abs": delta_all[block],
                               "delta_relative": delta_all[block] / max(b4, 1e-9)}

    payload = {
        "round": K.ROUND_ID,
        "protocol_sha256": K.sha256_file(K.ROUND / "PROTOCOL.md"),
        "config_sha256": K.sha256_file(K.CONFIG_PATH),
        "primary_checkpoint": "best_on_probe_selection",
        "delta_definition": "MAE(probe on z_4k) - MAE(probe on z_8k); positive = 8k easier to read",
        "delta_cell": {f"{b}|s{s}|i{i}": delta[(b, s, i)] for b, s, i in delta},
        "delta_seed": delta_seed,
        "delta_init": delta_init,
        "delta_all": delta_all,
        "delta_all_final_checkpoint": delta_all_final,
        "delta_final_cell": {f"{b}|s{s}|i{i}": delta_final[(b, s, i)] for b, s, i in delta_final},
        "levels_best": levels,
        "raw": metrics["raw"],
        "raw_gain_monitor": raw_gain,
        "probe_recovers_fraction_of_raw_gain": recovered,
        "probe_vs_raw_monitor_levels": level_gap,
        "block_levels_and_relative": block_levels,
        "raw_gain_reference_locked": metrics["raw_gain_reference_locked_best_monitor"],
        "groups": group_table,
        "bootstrap_full_block": boot,
        "controls": {"readout_recovery": recovery, "label_permutation_null": label_null},
        "gates": gates,
        "thresholds": threshold,
        "instrument_ok": instrument_ok,
        "raw_gain_gate_pass": bool(raw_gain >= float(cfg["gates"]["raw_gain_min_for_rocs"])),
        "sign_flip_across_trunk_seeds": sign_flip,
        "rocs_condition": rocs,
        "verdict": verdict,
        "verdict_rule": rule,
        "decision_order": "U -> TTS -> ROCS -> UNRESOLVED",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_loaded": False,
        "wall_seconds": float(time.perf_counter() - t0),
    }
    K.write_json(K.ANALYSIS / "fta_decision.json", payload)
    K.ledger_append({"stage": "analyze", "wall_seconds": payload["wall_seconds"], "device": "cpu"})

    # ------------------------------------------------------------ markdown tables
    L = ["# DCC-L frozen-transfer audit — machine-generated tables", ""]
    L.append("Δ = MAE(probe on z_4k) − MAE(probe on z_8k); **positive = 8k representation easier to read**.")
    L.append("")
    L.append("## 1. Primary cells (full 513, best checkpoint, Monitor1k)")
    L.append("")
    L.append("| trunk seed | head init | MAE 4k | MAE 8k | Δ |")
    L.append("|---|---:|---:|---:|---:|")
    for s in TRUNK_SEEDS:
        for i in HEAD_INITS:
            m4 = by_cell[(cell_of("fit4k", s, "full", i), "best")]["monitor_mae"]
            m8 = by_cell[(cell_of("fit8k", s, "full", i), "best")]["monitor_mae"]
            L.append(f"| {s} | {i} | {m4:.6f} | {m8:.6f} | {delta[('full', s, i)]:+.6f} |")
    L.append("")
    L.append("| block | Δ s0 | Δ s1 | Δ init0 | Δ init1 | Δ all | Δ all (final ckpt) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in BLOCKS:
        L.append(f"| {b} | {delta_seed[b][0]:+.6f} | {delta_seed[b][1]:+.6f} | "
                 f"{delta_init[b][0]:+.6f} | {delta_init[b][1]:+.6f} | "
                 f"{delta_all[b]:+.6f} | {delta_all_final[b]:+.6f} |")
    L.append("")
    L.append("## 2. Group table (full 513, best checkpoint)")
    L.append("")
    L.append("| group | n | MAE 4k | MAE 8k | Δ mean | Δ s0 | Δ s1 | rel. Δ | bootstrap 95% |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for g, row in group_table["full"].items():
        b = boot.get(g, {"ci95": [float("nan")] * 2})
        L.append(f"| {g} | {row['n']} | {row['fit4k_s0']['mean']:.4f} | "
                 f"{row['fit8k_s0']['mean']:.4f} | {row['delta_mean']:+.6f} | "
                 f"{row['delta_seed']['0']:+.6f} | {row['delta_seed']['1']:+.6f} | "
                 f"{row['delta_relative']:+.1%} | [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] |")
    L.append("")
    L.append("## 3. Block probes (same reader shape, unused columns zeroed)")
    L.append("")
    L.append("| group | n | Δ atom_global | Δ local_global | Δ full |")
    L.append("|---|---:|---:|---:|---:|")
    for g in group_table["full"]:
        n = group_table["full"][g]["n"]
        vals = [group_table[b].get(g, {}).get("delta_mean", float("nan")) for b in BLOCKS]
        L.append(f"| {g} | {n} | {vals[0]:+.6f} | {vals[1]:+.6f} | {vals[2]:+.6f} |")
    L.append("")
    L.append("### 3a. All block probe cells (best checkpoint, Monitor1k)")
    L.append("")
    L.append("| block | trunk seed | head init | MAE 4k | MAE 8k | Δ |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for b in BLOCKS:
        for s_ in TRUNK_SEEDS:
            for i_ in HEAD_INITS:
                m4 = by_cell[(cell_of("fit4k", s_, b, i_), "best")]["monitor_mae"]
                m8 = by_cell[(cell_of("fit8k", s_, b, i_), "best")]["monitor_mae"]
                L.append(f"| {b} | {s_} | {i_} | {m4:.6f} | {m8:.6f} | "
                         f"{delta[(b, s_, i_)]:+.6f} |")
    L.append("")
    L.append("### 3b. Block levels (probe MAE on Monitor1k, best checkpoint)")
    L.append("")
    L.append("| block | MAE 4k (2 seeds × 2 inits) | MAE 8k | Δ abs | Δ rel |")
    L.append("|---|---:|---:|---:|---:|")
    for b in BLOCKS:
        v = block_levels[b]
        L.append(f"| {b} | {v['mae_4k_mean']:.4f} | {v['mae_8k_mean']:.4f} | "
                 f"{v['delta_abs']:+.6f} | {v['delta_relative']:+.2%} |")
    L.append("")
    L.append("### 3c. Probe vs raw trunk (full 513, Monitor1k)")
    L.append("")
    L.append("| condition | probe MAE (head-init mean) | raw trunk MAE (final ckpt) | probe − raw |")
    L.append("|---|---:|---:|---:|")
    for cid in CONDS:
        v = level_gap[cid]
        L.append(f"| {cid} | {v['probe_best_mean']:.5f} | {v['raw_final']:.5f} | "
                 f"{v['probe_minus_raw']:+.5f} |")
    L.append("")
    L.append(f"Raw 4k→8k gain (Monitor, final ckpt) = **{raw_gain:+.5f}**; "
             f"fixed-head Δ = **{delta_all['full']:+.5f}** "
             f"(recovered fraction {recovered['all']:.1%}; s0 {recovered['s0']:.1%}, "
             f"s1 {recovered['s1']:.1%}).")
    L.append("")
    L.append("## 4. Controls")
    L.append("")
    L.append("| control | value |")
    L.append("|---|---:|")
    for cid, v in recovery.items():
        L.append(f"| recovery {cid}: probe_sel L1 vs trunk pred | {v['probe_sel_l1_vs_trunk_pred']:.5f} |")
        L.append(f"| recovery {cid}: monitor MAE vs trunk pred | {v['monitor_mae_vs_trunk_pred']:.5f} |")
    L.append(f"| null (permuted labels): monitor MAE vs y | {label_null['monitor_mae_vs_y']:.5f} |")
    L.append(f"| constant median predictor: monitor MAE | {const_pred_mae:.5f} |")
    L.append("")
    L.append("## 5. Gates and verdict")
    L.append("")
    for k in ("U1_pass", "U2_pass", "U3_pass", "U4_pass", "T1", "T2", "T3"):
        L.append(f"- `{k}` = {gates.get(k, threshold.get(k))}")
    L.append(f"- raw_gain (Monitor, final ckpt) = {raw_gain:+.6f}")
    L.append(f"- Δ_all (full) = {delta_all['full']:+.6f}")
    L.append(f"- **verdict = `{verdict}`** ({rule})")
    (K.ANALYSIS / "report_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    K.log(f"verdict {verdict}; delta_all(full)={delta_all['full']:+.6f}; "
          f"raw_gain={raw_gain:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
