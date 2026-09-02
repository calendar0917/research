"""
summarize_results.py — 结果汇总与审计

读取 results/registry.jsonl，生成:
1. 论文结果 vs 复现结果表 (results/paper_vs_reproduced.csv)
2. 每个方法/数据集/协议的 mean, std, median, CI
3. 成功/失败/OOM 统计
4. 最终判定标签

用法:
    python summarize_results.py
    python summarize_results.py --output results/paper_vs_reproduced.csv
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "results" / "registry.jsonl"

# 论文目标数字 (来自 HOD-GNN 论文表格；ROC-AUC/AP 统一为 0..1，和 OGB/LRGB evaluator 一致)
PAPER_TARGETS = {
    ("gps", "zinc-12k"): {"mean": 0.070, "std": 0.004},
    ("gps", "moltox21"): {"mean": 0.7570, "std": 0.0040},
    ("gps", "molhiv"): {"mean": 0.7880, "std": 0.0101},
    ("graphvit", "zinc-12k"): {"mean": 0.085, "std": 0.005},
    ("graphvit", "moltox21"): {"mean": 0.7851, "std": 0.0077},
    ("graphvit", "molhiv"): {"mean": 0.7792, "std": 0.0149},
    ("graphvit", "peptides-func"): {"mean": 0.6919, "std": 0.0085},
    ("graphvit", "peptides-struct"): {"mean": 0.2474, "std": 0.0016},
    ("full", "zinc-12k"): {"mean": 0.087, "std": 0.003},
    ("full", "moltox21"): {"mean": 0.7625, "std": 0.0112},
    ("full", "molbace"): {"mean": 0.7841, "std": 0.0194},
    ("full", "molhiv"): {"mean": 0.7654, "std": 0.0137},
    ("random", "zinc-12k"): {"mean": 0.102, "std": 0.003},
    ("random", "moltox21"): {"mean": 0.7662, "std": 0.0063},
    ("random", "molbace"): {"mean": 0.7814, "std": 0.0236},
    ("random", "molhiv"): {"mean": 0.7730, "std": 0.0256},
    ("policy_learn", "zinc-12k"): {"mean": 0.097, "std": 0.005},
    ("policy_learn", "moltox21"): {"mean": 0.7736, "std": 0.0060},
    ("policy_learn", "molbace"): {"mean": 0.7839, "std": 0.0228},
    ("policy_learn", "molhiv"): {"mean": 0.7849, "std": 0.0101},
    ("policy_learn", "peptides-func"): {"mean": 0.6459, "std": 0.0018},
    ("policy_learn", "peptides-struct"): {"mean": 0.2475, "std": 0.0011},
}


def load_registry():
    entries = []
    if not REGISTRY_PATH.exists():
        print(f"Registry not found: {REGISTRY_PATH}")
        return entries
    with open(REGISTRY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_ci(values, confidence=0.95):
    n = len(values)
    if n < 2:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    z = 1.96  # normal approximation
    margin = z * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def compute_stats(values):
    if not values:
        return None
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    median = statistics.median(values)
    ci = compute_ci(values)
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "median": round(median, 6),
        "ci_lower": round(ci[0], 6) if ci else None,
        "ci_upper": round(ci[1], 6) if ci else None,
        "raw": values,
    }


def determine_verdict(repro_mean, repro_std, paper_mean, paper_std):
    if repro_mean is None or paper_mean is None:
        return "inconclusive_due_to_missing_seed"
    diff = abs(repro_mean - paper_mean)
    combined_std = math.sqrt(repro_std**2 + paper_std**2) if repro_std and paper_std else 0
    if combined_std == 0:
        return "reproduced" if diff < 0.01 else "not_reproduced"
    if diff <= combined_std:
        return "reproduced"
    elif diff <= 2 * combined_std:
        return "partially_reproduced"
    else:
        return "not_reproduced"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "paper_vs_reproduced.csv"))
    args = parser.parse_args()

    entries = load_registry()
    if not entries:
        print("No entries in registry. Run experiments first.")
        return

    # Stats by (method, dataset, protocol)
    groups = defaultdict(list)
    for e in entries:
        key = (e["method"], e["dataset"], e["protocol"])
        if e["status"] == "success":
            groups[key].append(e["test_metric"])

    # Status counts
    status_counts = defaultdict(int)
    for e in entries:
        status_counts[e["status"]] += 1

    print(f"\n{'='*70}")
    print(f"RESULT SUMMARY")
    print(f"Total entries: {len(entries)}")
    print(f"Status breakdown: {dict(status_counts)}")
    print(f"{'='*70}\n")

    rows = []
    for (method, dataset, protocol), values in sorted(groups.items()):
        stats = compute_stats(values)
        paper = PAPER_TARGETS.get((method, dataset))
        verdict = determine_verdict(
            stats["mean"] if stats else None,
            stats["std"] if stats else None,
            paper["mean"] if paper else None,
            paper["std"] if paper else None,
        ) if paper else "unsupported"

        print(f"{method:15s} | {dataset:15s} | {protocol:25s} | "
              f"repro={stats['mean']:.4f}±{stats['std']:.4f} (n={stats['n']}) | "
              f"paper={paper['mean']:.4f}±{paper['std']:.4f} | "
              f"{verdict}")

        rows.append({
            "method": method,
            "dataset": dataset,
            "protocol": protocol,
            "repro_mean": stats["mean"],
            "repro_std": stats["std"],
            "repro_median": stats["median"],
            "repro_ci_lower": stats["ci_lower"],
            "repro_ci_upper": stats["ci_upper"],
            "repro_n": stats["n"],
            "paper_mean": paper["mean"] if paper else None,
            "paper_std": paper["std"] if paper else None,
            "verdict": verdict,
        })

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        header = "method,dataset,protocol,repro_mean,repro_std,repro_median,repro_ci_lower,repro_ci_upper,repro_n,paper_mean,paper_std,verdict\n"
        f.write(header)
        for row in rows:
            f.write(f"{row['method']},{row['dataset']},{row['protocol']},"
                    f"{row['repro_mean']},{row['repro_std']},{row['repro_median']},"
                    f"{row['repro_ci_lower']},{row['repro_ci_upper']},{row['repro_n']},"
                    f"{row['paper_mean']},{row['paper_std']},{row['verdict']}\n")

    print(f"\nSaved to {output_path}")

    # Final report summary
    print(f"\n{'='*70}")
    print("FINAL VERDICT BREAKDOWN")
    print(f"{'='*70}")
    verdict_counts = defaultdict(int)
    for row in rows:
        if row["protocol"] == "paper-seeds-provisional":
            verdict_counts[row["verdict"]] += 1
    for verdict, count in sorted(verdict_counts.items()):
        print(f"  {verdict}: {count}")


if __name__ == "__main__":
    main()
