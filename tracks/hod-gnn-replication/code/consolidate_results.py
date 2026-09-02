#!/usr/bin/env python3
"""Merge local/Kaggle registries into one auditable, de-duplicated view.

The raw registries remain untouched.  For each (method, dataset, protocol,
seed), a successful/unavailable terminal record wins over a failed retry; if
several terminal records exist, the source order below gives precedence to the
newer local/Kaggle run.  Source provenance is retained in each output row.
"""
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "consolidated"
SOURCES = [
    (ROOT / "results" / "kaggle" / "registry.jsonl", "kaggle-v16"),
    (ROOT / "results" / "local-seed2" / "registry.jsonl", "local-seed2"),
    (ROOT / "results" / "local-seed3" / "registry.jsonl", "local-seed3"),
    (ROOT / "kaggle" / "v20-unpacked" / "hod-gnn-results" / "registry.jsonl", "kaggle-v20"),
]
PAPER = {
    ("gps", "molhiv"): (0.7880, 0.0101),
    ("gps", "zinc"): (0.070, 0.004),
}


def read(path, source):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row["source_registry"] = source
        out.append(row)
    return out


def rank(row, source_index):
    # Terminal success/unavailable is the completed truth; failed/interrupted
    # rows remain useful only when no completed retry exists.  Among records
    # with the same terminal class, prefer the newest timestamp; this matters
    # for checkpointed retries where both the old and new attempt are timeout
    # records (the latest one contains more completed epochs).
    terminal = row.get("status") in {"success", "unavailable"}
    return (1 if terminal else 0, row.get("timestamp", ""), source_index)


def main():
    all_rows = []
    for i, (path, source) in enumerate(SOURCES):
        all_rows.extend((i, row) for row in read(path, source))

    chosen = {}
    for source_index, row in all_rows:
        key = (row.get("method"), row.get("dataset"), row.get("protocol"), row.get("seed"))
        candidate = dict(row)
        candidate["source_priority"] = source_index
        if key not in chosen or rank(candidate, source_index) >= rank(chosen[key], chosen[key].get("source_priority", -1)):
            chosen[key] = candidate

    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(chosen.values(), key=lambda r: (r.get("method", ""), r.get("dataset", ""), r.get("seed", -1)))
    with (OUT / "registry.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            row.pop("source_priority", None)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "success" and row.get("test_metric") is not None:
            groups[(row.get("method"), row.get("dataset"), row.get("protocol"))].append(row)
    with (OUT / "paper_vs_reproduced.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "protocol", "repro_mean", "repro_std", "n", "paper", "verdict"])
        for key, vals in sorted(groups.items()):
            method, dataset, protocol = key
            metrics = [v["test_metric"] for v in vals]
            mean = statistics.mean(metrics)
            std = statistics.stdev(metrics) if len(metrics) > 1 else 0.0
            target = PAPER.get((method, dataset))
            if target:
                combined = (std ** 2 + target[1] ** 2) ** 0.5
                verdict = "reproduced" if abs(mean - target[0]) <= combined else ("partial" if abs(mean - target[0]) <= 2 * combined else "not_reproduced")
                paper = f"{target[0]}±{target[1]}"
            else:
                verdict, paper = "?", "—"
            w.writerow([method, dataset, protocol, f"{mean:.4f}", f"{std:.4f}", len(metrics), paper, verdict])

    counts = Counter(row.get("status") for row in rows)
    print(f"wrote {len(rows)} unique units to {OUT}")
    print("status_counts", dict(counts))
    for key, vals in sorted(groups.items()):
        m = [v["test_metric"] for v in vals]
        print(key, f"mean={statistics.mean(m):.6f}", f"std={statistics.stdev(m) if len(m)>1 else 0:.6f}", f"n={len(m)}")


if __name__ == "__main__":
    main()
