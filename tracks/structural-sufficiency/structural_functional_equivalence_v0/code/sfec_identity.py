"""Write data_identity.json: hashes, split counts, row plan, identity methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import sfec_common as C


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_record(path: Path, full_hash: bool = True) -> dict:
    rec = {"path": str(path), "bytes": int(path.stat().st_size)}
    if full_hash:
        rec["sha256"] = sha256(path)
    return rec


def cache_record(path: Path) -> dict:
    gids = np.load(path / "gids.npy")
    smiles = path / "smiles.jsonl"
    n_lines = 0
    with smiles.open("rb") as handle:
        for _ in handle:
            n_lines += 1
    return {
        "path": str(path),
        "n_gids": int(len(gids)),
        "gids_unique": int(len(np.unique(gids))),
        "gids_sha256": sha256(path / "gids.npy"),
        "smiles_bytes": int(smiles.stat().st_size),
        "smiles_lines": int(n_lines),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "data_identity.json"))
    args = parser.parse_args(argv)
    round_dir = C.ROUND_DIR
    split = C.load_pcqm_split()
    zinc_sizes = {s: len(C.load_zinc_split(s)) for s in ("train", "valid", "test")}
    code_files = sorted((round_dir / "code").glob("*.py"))
    code = {p.name: sha256(p) for p in code_files}
    artifacts = {}
    for prefix_name in ("cells_zinc", "cells_pcqm_train_1m", "cells_pcqm_valid", "cells_pcqm_train_100k"):
        prefix = round_dir / "data" / prefix_name
        for suffix in (".npz", ".C.parquet", ".S.parquet", ".M.parquet", ".keys.json", ".universe.npz", ".support.json", ".rectangles.csv"):
            path = Path(str(prefix) + suffix)
            if path.exists():
                artifacts[path.name] = file_record(path)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=C.REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "round": "sfec_v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(C.REPO_ROOT),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "protocol": file_record(round_dir / "PROTOCOL.md"),
        "config": file_record(round_dir / "configs" / "sfec_v0.json"),
        "code_sha256": code,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
            "torch": __import__("torch").__version__,
            "pyg": __import__("torch_geometric").__version__,
            "pynauty": __import__("pynauty").__version__ if hasattr(__import__("pynauty"), "__version__") else "unknown",
        },
        "datasets": {
            "zinc": {
                "source": str(C.ZINC_ROOT),
                "official_subset": True,
                "split_sizes": zinc_sizes,
                "labels": "logP_SA_cycle_normalized (raw y)",
                "raw_files": {
                    name: file_record(C.ZINC_ROOT / "raw" / name)
                    for name in ("train.pickle", "val.pickle", "test.pickle", "train.index", "val.index", "test.index")
                },
                "identity": "pynauty colored-incidence certificate of the rooted fragment; WL hashes never used as identity",
                "known_limit": "5-6% of PCQM stereochemical reassemblies differ; graph reassembly is exact and reported separately",
            },
            "pcqm": {
                "source": str(C.PCQM_ROOT),
                "raw_files": {
                    "data.csv.gz": file_record(C.PCQM_ROOT / "raw" / "data.csv.gz"),
                    "split_dict.pt": file_record(C.PCQM_ROOT / "split_dict.pt"),
                },
                "official_split_counts": {k: int(len(v)) for k, v in split.items()},
                "rows_used": {"train_prefix": 1_000_000, "valid": 73_545, "test": 0},
                "labels": "homolumogap (eV, raw column; no normalization)",
                "test_labels_read": False,
                "cache": {
                    "train1m": cache_record(C.PCQM_TRAIN_CACHE),
                    "valid": cache_record(C.PCQM_VALID_CACHE),
                },
                "identity": "canonical isomeric fragment SMILES with explicit dummy atoms at original bond slots; WL hashes never used as identity",
                "known_limit": "stereochemistry is not used in the graph-level identity; ~8.9% sampled stereo reassemblies differ",
            },
        },
        "row_plan": {
            "pcqm_train_prefix_order": "stable cache order; 100k -> 1M; no expansion based on labels",
            "pcqm_train_rows": 1_000_000,
            "pcqm_valid_rows": 73_545,
            "pcqm_test_rows": 0,
            "zinc_rows": 12_000,
        },
        "artifacts": artifacts,
        "counts": {
            "zinc": {
                "cells": int(len(np.load(round_dir / "data" / "cells_zinc.npz")["row_id"])),
                "C_keys": int(len(pq.read_table(round_dir / "data" / "cells_zinc.C.parquet", columns=["C_id"]))),
                "S_keys": int(len(pq.read_table(round_dir / "data" / "cells_zinc.S.parquet", columns=["S_id"]))),
                "molecules": int(len(pq.read_table(round_dir / "data" / "cells_zinc.M.parquet", columns=["mol_id"]))),
            },
            "pcqm_train_1m": {
                "cells": int(len(np.load(round_dir / "data" / "cells_pcqm_train_1m.npz")["row_id"])),
                "C_keys": int(len(pq.read_table(round_dir / "data" / "cells_pcqm_train_1m.C.parquet", columns=["C_id"]))),
                "S_keys": int(len(pq.read_table(round_dir / "data" / "cells_pcqm_train_1m.S.parquet", columns=["S_id"]))),
                "molecules": int(len(pq.read_table(round_dir / "data" / "cells_pcqm_train_1m.M.parquet", columns=["mol_id"]))),
            },
            "pcqm_valid": {
                "cells": int(len(np.load(round_dir / "data" / "cells_pcqm_valid.npz")["row_id"])),
                "C_keys": int(len(pq.read_table(round_dir / "data" / "cells_pcqm_valid.C.parquet", columns=["C_id"]))),
                "S_keys": int(len(pq.read_table(round_dir / "data" / "cells_pcqm_valid.S.parquet", columns=["S_id"]))),
                "molecules": int(len(pq.read_table(round_dir / "data" / "cells_pcqm_valid.M.parquet", columns=["mol_id"]))),
            },
        },
        "identity_methods": {
            "zinc": "colored incidence certificate, root-marked (historical rd_fragment.py)",
            "pcqm": "RDKit canonical isomeric SMILES of rooted fragments with explicit dummies",
            "dedupe": "molecule key = whole-graph certificate/SMILES; duplicate rows keep labels but share one identity",
            "context": "C_key = root-marked environment fragment; S_key = root-marked small fragment",
        },
        "notes": [
            "PCQM only Train and Valid were parsed; Test labels were never read.",
            "PCQM duplicate full graphs can carry different labels; all duplicate rows share one molecule identity for split independence.",
            "ZINC official PyG subset; train/valid/test counts as stated.",
        ],
    }
    C.write_json(args.out, payload)
    print(json.dumps({k: payload[k] for k in ("round", "git_commit", "git_dirty")}, indent=2))
    print("artifacts", len(artifacts), "counts", payload["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())