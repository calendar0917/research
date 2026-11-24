"""Aggregate canonical artifacts and write observed_cells.parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sfec_common as C
import sfec_data as D


def concat_csv(paths: list[Path], out: Path) -> None:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 10:
            frame = pd.read_csv(path)
            if "dataset" not in frame.columns:
                frame.insert(0, "dataset", path.stem.split("_")[-1])
            frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)


def write_observed_cells(out: Path) -> dict[str, int]:
    schema = pa.schema([
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("row_id", pa.int64()),
        ("molecule_id", pa.int64()),
        ("cut_id", pa.int32()),
        ("bond_type", pa.int8()),
        ("C_id", pa.int32()),
        ("S_id", pa.int32()),
        ("nS", pa.int8()),
        ("nC", pa.int32()),
        ("y", pa.float64()),
    ])
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    counts = {}
    try:
        # PCQM train
        for name, prefix, split_name, label_kind in (
            ("pcqm", C.ROUND_DIR / "data" / "cells_pcqm_train_1m", "train", "pcqm"),
            ("pcqm", C.ROUND_DIR / "data" / "cells_pcqm_valid", "valid", "pcqm"),
        ):
            cells = D.load_cells(prefix)
            labels = C.load_pcqm_labels()
            n = len(cells["row_id"])
            for start in range(0, n, 400_000):
                end = min(n, start + 400_000)
                row = cells["row_id"][start:end].astype(np.int64)
                tab = pa.table({
                    "dataset": pa.array([name] * (end - start), type=pa.string()),
                    "split": pa.array([split_name] * (end - start), type=pa.string()),
                    "row_id": pa.array(row, type=pa.int64()),
                    "molecule_id": pa.array(cells["mol_id"][start:end].astype(np.int64), type=pa.int64()),
                    "cut_id": pa.array(cells["cut_id"][start:end].astype(np.int32), type=pa.int32()),
                    "bond_type": pa.array(cells["bond_type"][start:end].astype(np.int8), type=pa.int8()),
                    "C_id": pa.array(cells["C_id"][start:end].astype(np.int32), type=pa.int32()),
                    "S_id": pa.array(cells["S_id"][start:end].astype(np.int32), type=pa.int32()),
                    "nS": pa.array(cells["nS"][start:end].astype(np.int8), type=pa.int8()),
                    "nC": pa.array(cells["nC"][start:end].astype(np.int32), type=pa.int32()),
                    "y": pa.array(labels[row], type=pa.float64()),
                })
                writer.write_table(tab)
            counts[f"{name}_{split_name}"] = int(n)
        # ZINC
        cells = D.load_cells(C.ROUND_DIR / "data" / "cells_zinc")
        for split, code in (("train", 0), ("valid", 1), ("test", 2)):
            labels = C.zinc_labels(split)
            idx = np.flatnonzero(cells["split_id"] == code)
            for start in range(0, len(idx), 400_000):
                take = idx[start:start + 400_000]
                row = cells["row_id"][take].astype(np.int64)
                tab = pa.table({
                    "dataset": pa.array(["zinc"] * len(take), type=pa.string()),
                    "split": pa.array([split] * len(take), type=pa.string()),
                    "row_id": pa.array(row, type=pa.int64()),
                    "molecule_id": pa.array(cells["mol_id"][take].astype(np.int64), type=pa.int64()),
                    "cut_id": pa.array(cells["cut_id"][take].astype(np.int32), type=pa.int32()),
                    "bond_type": pa.array(cells["bond_type"][take].astype(np.int8), type=pa.int8()),
                    "C_id": pa.array(cells["C_id"][take].astype(np.int32), type=pa.int32()),
                    "S_id": pa.array(cells["S_id"][take].astype(np.int32), type=pa.int32()),
                    "nS": pa.array(cells["nS"][take].astype(np.int8), type=pa.int8()),
                    "nC": pa.array(cells["nC"][take].astype(np.int32), type=pa.int32()),
                    "y": pa.array(labels[row], type=pa.float64()),
                })
                writer.write_table(tab)
            counts[f"zinc_{split}"] = int(len(idx))
    finally:
        writer.close()
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(C.ANALYSIS_DIR))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # support
    support = {}
    for key, path in (
        ("pcqm_train_1m", C.ROUND_DIR / "data" / "cells_pcqm_train_1m.support.json"),
        ("pcqm_train_100k", C.ROUND_DIR / "data" / "cells_pcqm_train_100k.support.json"),
        ("zinc", C.ROUND_DIR / "data" / "cells_zinc.support.json"),
    ):
        support[key] = json.loads(path.read_text())
    C.write_json(out / "structural_support.json", support)
    # context split + rule
    config = C.load_config()
    context = {
        "rule": {
            "salt": "0|",
            "hash": "sha256 hex first 16 chars mod 100",
            "discovery": "< 80",
            "evaluation": ">= 80",
            "molecule_disjoint": "remove from discovery any molecule appearing in any evaluation context",
            "primary_evaluation_selection": "hash-order greedy, molecule-disjoint, requiring primary S and k>=2",
            "seed": config["statistics"]["bootstrap_seed"],
            "tau": config["datasets"]["pcqm"]["tau"],
        },
        "datasets": {k: v["context_split"] for k, v in support.items()},
    }
    C.write_json(out / "context_split.json", context)
    # rectangle statistics
    frames = []
    for name, path in (
        ("pcqm_train_1m", C.ROUND_DIR / "data" / "cells_pcqm_train_1m.rectangles.csv"),
        ("pcqm_train_100k", C.ROUND_DIR / "data" / "cells_pcqm_train_100k.rectangles.csv"),
        ("zinc", C.ROUND_DIR / "data" / "cells_zinc.rectangles.csv"),
    ):
        if path.exists():
            f = pd.read_csv(path)
            f.insert(0, "dataset", name)
            frames.append(f)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "rectangle_statistics.csv", index=False)
    # response transfer + candidates
    concat_csv([out / "response_transfer_pcqm.csv", out / "response_transfer_zinc.csv"], out / "response_transfer.csv")
    concat_csv([out / "functional_equivalence_candidates_pcqm.csv", out / "functional_equivalence_candidates_zinc.csv"], out / "functional_equivalence_candidates.csv")
    # composition
    concat_csv([out / "unseen_composition_results_pcqm.csv", out / "unseen_composition_results_zinc.csv"], out / "unseen_composition_results.csv")
    # context rule results: metrics rows + q support
    rows = []
    for name in ("pcqm", "zinc"):
        p = json.loads((out / f"p3_{name}.json").read_text())
        for subset, rec in p["metrics"].items():
            if "n_obs" not in rec:
                continue
            for model in ("SHARED", "LOCAL", "LOCAL_NULL"):
                if model not in rec:
                    continue
                rows.append({
                    "dataset": name, "subset": subset, "model": model,
                    "n_obs": rec["n_obs"], "n_ctx": rec["n_ctx"], "n_pairs": rec["n_pairs"],
                    "mae": rec[model]["mae"], "mae_boot": rec[model]["mae_boot"],
                    "mae_ci_lo": rec[model]["mae_ci"][0], "mae_ci_hi": rec[model]["mae_ci"][1],
                    "sign_accuracy": rec[model]["sign_accuracy"],
                    "relative_gain_local_vs_shared": rec.get("relative_gain_local_vs_shared"),
                    "relative_gain_local_vs_null": rec.get("relative_gain_local_vs_null"),
                    "diff_shared_minus_local": rec.get("diff_SHARED_minus_LOCAL", {}).get("point"),
                    "diff_local_minus_null": rec.get("diff_LOCAL_minus_NULL", {}).get("point"),
                })
        for q in p["q_support"]:
            rows.append({
                "dataset": name, "subset": "q_support", "model": "",
                "q_id": q["q_id"], "q": q["q"], "n_contexts": q["n_contexts"],
                "n_contexts_k_ge_2": q["n_contexts_k_ge_2"], "supported": q["supported"],
                "n_s_types": q["n_s_types"], "n_components": q["n_components"],
            })
    if rows:
        pd.DataFrame(rows).to_csv(out / "context_rule_results.csv", index=False)
    counts = write_observed_cells(out / "observed_cells.parquet")
    summary = {
        "counts": counts,
        "observed_cells_bytes": int((out / "observed_cells.parquet").stat().st_size),
        "files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    C.write_json(out / "aggregate_manifest.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())