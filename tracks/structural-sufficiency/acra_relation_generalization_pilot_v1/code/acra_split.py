"""Frozen, label-blind Train-sub / Dev scaffold split.

Only the official Train rows are used.  Scaffolds are RDKit Bemis--Murcko
scaffolds; acyclic molecules have an empty Murcko scaffold and are keyed by
their own SMILES so they are not all forced into one giant group.  Groups are
shuffled with seed 0 and greedily assigned to Dev until the target fraction is
reached without exceeding it.  No label is read.

The resulting JSON is committed; training never invokes RDKit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D


def split_path(cfg: Mapping[str, Any]) -> Path:
    return C.ROUND_DIR / str(cfg["split"]["path"])


def scaffold_key(smiles: str) -> str:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"__UNPARSED__{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else f"__ACYCLIC__{smiles}"


def build(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    raw = D.load_raw(cfg)
    train = np.asarray(raw["train_idx"], np.int64)
    smiles = np.asarray(raw["smiles"])
    groups: dict[str, list[int]] = defaultdict(list)
    for row in train.tolist():
        groups[scaffold_key(str(smiles[row]))].append(int(row))
    keys = sorted(groups)
    rng = np.random.default_rng(int(cfg["split"]["seed"]))
    order = rng.permutation(len(keys))
    target = int(round(float(cfg["split"]["fraction_dev"]) * train.size))
    dev: list[int] = []
    dev_scaffolds: list[str] = []
    for i in order.tolist():
        key = keys[i]
        if len(dev) + len(groups[key]) <= target:
            dev.extend(groups[key])
            dev_scaffolds.append(key)
        if len(dev) == target:
            break
    if len(dev) != target:
        raise RuntimeError(f"could not hit the frozen Dev target exactly: {len(dev)} != {target}")
    dev_set = set(dev)
    train_sub = [int(r) for r in train.tolist() if int(r) not in dev_set]
    if set(train_sub) & dev_set:
        raise AssertionError("Train-sub / Dev overlap")
    if sorted(train_sub + dev) != sorted(int(r) for r in train.tolist()):
        raise AssertionError("Train-sub and Dev do not cover the official Train split")
    scaffold_by_row = {str(r): scaffold_key(str(smiles[r])) for r in train.tolist()}
    payload: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "seed": int(cfg["split"]["seed"]),
        "fraction_dev_target": float(cfg["split"]["fraction_dev"]),
        "n_official_train": int(train.size),
        "n_train_sub": int(len(train_sub)),
        "n_dev": int(len(dev)),
        "n_scaffolds_official_train": int(len(keys)),
        "n_scaffolds_train_sub": int(len(set(scaffold_by_row[str(r)] for r in train_sub))),
        "n_scaffolds_dev": int(len(set(scaffold_by_row[str(r)] for r in dev))),
        "train_sub_rows": train_sub,
        "dev_rows": sorted(int(r) for r in dev),
        "dev_scaffolds": sorted(dev_scaffolds),
        "scaffold_by_row": scaffold_by_row,
        "label_free": True,
        "labels_read": False,
        "scaffold_definition": str(cfg["split"]["scaffold"]),
        "assignment_rule": "sorted scaffold keys -> numpy default_rng(seed).permutation -> "
                           "greedy add whole groups while not exceeding the target",
    }
    payload["split_sha256"] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ("train_sub_rows", "dev_rows", "seed")},
                   sort_keys=True).encode()).hexdigest()
    path = split_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log(f"wrote {path}: train_sub={len(train_sub)} dev={len(dev)} "
        f"scaffolds={len(keys)}")
    return payload


def load(cfg: Mapping[str, Any]) -> dict[str, Any]:
    path = split_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"frozen split missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = np.asarray(payload["train_sub_rows"], np.int64)
    dev = np.asarray(payload["dev_rows"], np.int64)
    if train.size != int(payload["n_train_sub"]) or dev.size != int(payload["n_dev"]):
        raise ValueError("split counts do not match the frozen payload")
    if set(train.tolist()) & set(dev.tolist()):
        raise ValueError("Train-sub / Dev overlap")
    return payload


def row_split(cfg: Mapping[str, Any], n_rows: int) -> np.ndarray:
    """0 = Train-sub, 1 = Dev, 2 = outside the official Train split."""
    payload = load(cfg)
    out = np.full(int(n_rows), 2, np.uint8)
    out[np.asarray(payload["train_sub_rows"], np.int64)] = 0
    out[np.asarray(payload["dev_rows"], np.int64)] = 1
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    build(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())