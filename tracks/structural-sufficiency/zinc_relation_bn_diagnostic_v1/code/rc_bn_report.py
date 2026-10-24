"""Merge the two per-arm diagnostic JSONs into the round's deliverables.

Reads ``analysis/diag_{D,S}.json`` + ``analysis/raw_{arm}_{tag}.npz`` (produced on
the GPUs, pulled back) and writes the small, reviewable artifacts:

* manifest.json   checkpoint inventory (file/state/param/BN hashes, epochs)
* checks.json     the calibration-process checks, per checkpoint
* summary.json     the four-state table (O/B/T/C) plus deltas
* bn_layers.json   per-BN entry counts and old-vs-new statistic summaries
* main_table.md    the minimal result table
* predictions_train.csv / predictions_valid.csv   per-molecule O/B/C predictions
* budget.json      wall clock and GPU hours
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ANALYSIS = EXP / "analysis"
ARMS = ("D", "S")
TAGS = ("best", "final")
REFERENCE = {  # from zinc_relation_composition_probe_v1/REPORT.md (eval mode)
    ("D", "best"): (0.02357758441468468, 0.08449332812236389),
    ("D", "final"): (0.033237688037042974, 0.0954388127065613),
    ("S", "best"): (0.053498606581462084, 0.08183847504806519),
    ("S", "final"): (0.10889866944144014, 0.09921123916649958),
}
ONLINE = {  # online train MAE at the checkpoint's epoch (round-internal reference)
    ("D", "best"): 0.0087352783, ("D", "final"): 0.0082290242,
    ("S", "best"): 0.0090593297, ("S", "final"): 0.0084401009,
}
B_R0_VALID = 0.07853714964160463


def main() -> int:
    diag = {a: json.loads((ANALYSIS / f"diag_{a}.json").read_text()) for a in ARMS}
    manifest, checks, summary, layers = {}, {}, {}, {}
    for arm in ARMS:
        for tag in TAGS:
            e = diag[arm]["checkpoints"][tag]
            key = f"{arm}_{tag}"
            manifest[key] = {
                "source_file": e["source_file"],
                "file_sha256": e["file_sha256"],
                "state_sha256": e["state_sha256"],
                "epoch": e["epoch"],
                "parameters": e["parameters"],
                "tensors": e["tensors"],
                "param_hash": e["param_hash"],
                "source_bn_buffer_hash": e["bn_buffer_hash"],
                "derived_bn_buffer_hash": e["derived_bn_buffer_hash"],
                "bn_modules": len(e["bn_modules"]),
                "arm": arm, "seed": 0, "selection": tag,
            }
            checks[key] = {
                "moment_streaming_selftest": diag[arm]["moment_selftest"],
                "bn_call_order_repeated": e["bn_call_order_repeated"],
                "bn_entry_counts": e["bn_count_check"],
                "calibration_self_consistency": e["verify"],
                "params_identical_after_calibration": e["params_identical"],
                "state_diff_only_running_stats": e["state_diff_only_running_stats"],
                "state_diff_keys": e["state_diff_keys"],
                "source_unchanged_by_diagnostics": e["O"]["unchanged_by_diagnostics"],
                "derived_bn_buffers_stable_through_eval": e["C"]["bn_hash_stable_after_valid"],
                "C_rebatch_invariance_max_abs_diff": e["C"]["rebatch_max_abs_diff"],
                "O_replay_train_mae": e["O"]["train_mae"],
                "O_replay_valid_mae": e["O"]["valid_mae"],
                "O_replay_reference_train": REFERENCE[(arm, tag)][0],
                "O_replay_reference_valid": REFERENCE[(arm, tag)][1],
                "O_replay_max_abs_diff_vs_reference": max(
                    abs(e["O"]["train_mae"] - REFERENCE[(arm, tag)][0]),
                    abs(e["O"]["valid_mae"] - REFERENCE[(arm, tag)][1])),
            }
            t = e.get("T", {})
            summary[key] = {
                "epoch": e["epoch"],
                "online_train_mae": ONLINE[(arm, tag)],
                "O_train": e["O"]["train_mae"], "B_train": e["B"]["train_mae"],
                "T_train_subset": t.get("subset_mae"), "T_ran": not t.get("skipped", False),
                "C_train": e["C"]["train_mae"],
                "O_valid": e["O"]["valid_mae"], "C_valid": e["C"]["valid_mae"],
                "C_minus_O_train": e["C"]["train_mae"] - e["O"]["train_mae"],
                "C_minus_O_valid": e["C"]["valid_mae"] - e["O"]["valid_mae"],
                "B_minus_O_train": e["B"]["train_mae"] - e["O"]["train_mae"],
                "O_subset": e["O"]["subset_mae"], "B_subset": e["B"]["subset_mae"],
                "C_subset": e["C"]["subset_mae"],
                "B_subset_second_permutation": e["B"]["batch_context_sensitivity"][
                    "subset_mae_second_permutation"],
                "B_batch_context_mean_abs_pred_diff": e["B"]["batch_context_sensitivity"][
                    "mean_abs_diff_matched_rows"],
                "B_vs_online_absdiff": abs(e["B"]["train_mae"] - ONLINE[(arm, tag)]),
                "C_valid_minus_B_R0": e["C"]["valid_mae"] - B_R0_VALID,
            }
            layers[key] = e["bn_layers"]

    # ---- merged per-molecule predictions
    for split in ("train", "valid"):
        arrs = {}
        for arm in ARMS:
            for tag in TAGS:
                z = np.load(ANALYSIS / f"raw_{arm}_{tag}.npz")
                arrs[(arm, tag, "O")] = z[f"{split}_o"]
                if split == "train":
                    arrs[(arm, tag, "B")] = z["train_b"]
                arrs[(arm, tag, "C")] = z[f"{split}_c"]
        y = None
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                               / "zinc_arch_discovery_v1" / "code"))
        import ad_data as D
        y = np.asarray(D.load_cache(split)["labels"], dtype=np.float64)
        n = len(y)
        cols = ["row", "y"]
        for k in sorted(arrs, key=lambda t: (t[0], TAGS.index(t[1]), t[2])):
            cols.append(f"{k[0]}_{k[1]}_{k[2]}")
        with open(ANALYSIS / f"predictions_{split}.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for i in range(n):
                vals = [str(i), f"{y[i]:.6f}"]
                vals += [f"{arrs[k][i]:.6f}" for k in
                         sorted(arrs, key=lambda t: (t[0], TAGS.index(t[1]), t[2]))]
                fh.write(",".join(vals) + "\n")

    # ---- markdown
    f = lambda v: "n/a" if v is None else f"{v:.6f}"
    L = ["### 四种状态（同一冻结权重，seed 0，train-only 诊断 + valid 仅评估）\n",
         "| checkpoint | ep | online train | O train | B train | T train(sub) | C train | O valid | C valid | C−O valid |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for tag in TAGS:
            s = summary[f"{arm}_{tag}"]
            L.append(f"| {arm}_{tag} | {s['epoch']} | {s['online_train_mae']:.6f} | "
                     f"{s['O_train']:.6f} | **{s['B_train']:.6f}** | {f(s['T_train_subset'])} | "
                     f"**{s['C_train']:.6f}** | {s['O_valid']:.6f} | **{s['C_valid']:.6f}** | "
                     f"{s['C_minus_O_valid']:+.6f} |")
    L += ["", "### 同子集配对（2048 train 分子，batch 128，O/B/C 同序同批）\n",
          "| checkpoint | O | B | B−O | B(第二排列) | C | C−O |",
          "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for tag in TAGS:
            s = summary[f"{arm}_{tag}"]
            L.append(f"| {arm}_{tag} | {s['O_subset']:.6f} | {s['B_subset']:.6f} | "
                     f"{s['B_minus_O_train']:+.6f} | {s['B_subset_second_permutation']:.6f} | "
                     f"{s['C_subset']:.6f} | {s['C_minus_O_train']:+.6f} |")
    L += ["", "### 检查摘要\n",
          "| checkpoint | 复现 O 参考(最大差) | 参数一致 | 只改 BN running | BN 计数 | 校准自洽(最大) | 重分批 | 源未改 |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for tag in TAGS:
            c = checks[f"{arm}_{tag}"]
            L.append(f"| {arm}_{tag} | {c['O_replay_max_abs_diff_vs_reference']:.1e} | "
                     f"{c['params_identical_after_calibration']} | "
                     f"{c['state_diff_only_running_stats']} | "
                     f"{c['bn_entry_counts']['passed']} | "
                     f"{c['calibration_self_consistency']['worst_rel_var_diff']:.1e} | "
                     f"{c['C_rebatch_invariance_max_abs_diff']:.1e} | "
                     f"{c['source_unchanged_by_diagnostics']} |")
    (ANALYSIS / "main_table.md").write_text("\n".join(L) + "\n")

    (ANALYSIS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (ANALYSIS / "checks.json").write_text(json.dumps(checks, indent=1))
    (ANALYSIS / "summary.json").write_text(json.dumps(summary, indent=1))
    (ANALYSIS / "bn_layers.json").write_text(json.dumps(layers, indent=1))
    print((ANALYSIS / "main_table.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
