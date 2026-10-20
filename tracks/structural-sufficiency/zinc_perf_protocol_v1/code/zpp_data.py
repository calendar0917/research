"""Data layer for zinc-perf-protocol-v1.

Only official PyG ZINC subset=True graphs and labels are used. RRWP features are
deterministic functions of the raw graph and are cached once. The P3 calibration
additionally verifies the frozen object caches against the official split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import zpp_common as C

SPLITS = ("train", "valid", "test")


def pyg_zinc(split: str):
    from torch_geometric.datasets import ZINC

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pyg_split = "val" if split == "valid" else split
    return ZINC(root=str(C.ZINC_ROOT), subset=True, split=pyg_split)


def split_size(split: str) -> int:
    return {"train": 10000, "valid": 1000, "test": 1000}[split]


def _cast(d):
    # GraphGym TypeDict encoders index embeddings with x[:,0] / edge_attr.
    d.x = d.x.long()
    d.edge_attr = d.edge_attr.long()
    d.y = d.y.float().view(-1)
    return d


def build_data(split: str, *, ksteps: int = 21, rrwp: bool = True,
               force: bool = False) -> list:
    """Official split as a list of PyG Data, with cached deterministic RRWP."""
    C.CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"zinc_subset_{split}_rrwp{ksteps}" if rrwp else f"zinc_subset_{split}_norrwp"
    cache = C.CACHE / f"{tag}.pt"
    if cache.exists() and not force:
        return torch.load(cache, map_location="cpu", weights_only=False)

    ds = pyg_zinc(split)
    if len(ds) != split_size(split):
        raise RuntimeError(f"{split}: expected {split_size(split)} graphs, got {len(ds)}")
    data = []
    if rrwp:
        C.ensure_grit_runtime()
        from grit.transform.rrwp import add_full_rrwp
    for i in range(len(ds)):
        d = _cast(ds.get(i).clone())
        if rrwp:
            d = add_full_rrwp(d, walk_length=int(ksteps), add_identity=True)
        data.append(d)
    torch.save(data, cache)
    return data


def split_labels(data: Sequence) -> np.ndarray:
    return np.asarray([float(d.y.reshape(-1)[0]) for d in data], dtype=np.float64)


def load_p3_objects():
    return torch.load(C.P3_OBJECTS_TRAIN, map_location="cpu", weights_only=False)


def load_p3_valid_objects():
    return torch.load(C.P3_OBJECTS_VALID, map_location="cpu", weights_only=False)


def load_p3_train_y() -> np.ndarray:
    enc = torch.load(C.P3_ENCODED_TRAIN, map_location="cpu", weights_only=False)
    y = np.asarray([float(enc[i].y.reshape(-1)[0]) for i in range(10000)], dtype=np.float64)
    del enc
    return y


def verify_p3_inputs() -> dict:
    """Verify the frozen P3 inputs by content, not by trusting the old config."""
    expected = {
        C.P3_OBJECTS_TRAIN: "14326a7ae6f9bb673a5c8b35c7d54171a4008f77edb702f0591d3d1f21338e8d",
        C.P3_OBJECTS_VALID: "61fca36a097f859b72a29e5f656405bff87a54ecf09f74b2727aa9663fdb853c",
        C.P3_PREP: "d0fb86a349ebdd37af069cdaa4ee71ba5045a2b67365cb28b2dd734c594b1e2a",
        C.P3_SSL_S0: "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97",
        C.P3_SSL_S1: "ff8d6474896c274407a1433658b95c79561fbc9a1deb1f9885bd862b4e8b7a1a",
    }
    out = {}
    ok = True
    for path, want in expected.items():
        got = C.sha256_file(path) if path.exists() else "MISSING"
        passed = got == want
        ok = ok and passed
        out[str(path.relative_to(C.REPO))] = {"expected": want, "actual": got, "ok": passed}
    out["all_ok"] = bool(ok)
    if not ok:
        raise RuntimeError(f"frozen P3 input hash mismatch: {out}")
    return out


def audit_data(write: bool = True) -> dict:
    """Audit official splits and cross-check the frozen P3 object cache."""
    pyg_splits = {}
    for split in SPLITS:
        ds = pyg_zinc(split)
        ys = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float64)
        atom_types = set()
        bond_types = set()
        nodes = []
        for i in range(len(ds)):
            d = ds[i]
            atom_types.update(int(v) for v in torch.unique(d.x).tolist())
            bond_types.update(int(v) for v in torch.unique(d.edge_attr).tolist())
            nodes.append(int(d.num_nodes))
        pyg_splits[split] = {
            "n": int(len(ds)),
            "atom_types": sorted(atom_types),
            "bond_types": sorted(bond_types),
            "num_nodes_min": int(min(nodes)),
            "num_nodes_max": int(max(nodes)),
            "num_nodes_mean": float(np.mean(nodes)),
            "label_mean": float(ys.mean()),
            "label_std_sample": float(ys.std(ddof=1)),
            "label_std_population": float(ys.std(ddof=0)),
            "label_min": float(ys.min()),
            "label_max": float(ys.max()),
            "label_sha256": C.sha256_array(ys),
        }

    # Cross-check official train graphs and labels against the frozen P3 cache.
    C.ensure_p3_modules()
    import structural_relation_objects_v1 as sro  # noqa: F401

    train_ds = pyg_zinc("train")
    objs = load_p3_objects()
    if len(objs) != 10000:
        raise RuntimeError(f"frozen train objects: {len(objs)}")
    max_atom_diff = 0
    max_bond_diff = 0
    n_atoms_mismatch = 0
    for i, obj in enumerate(objs):
        d = train_ds[i]
        n = int(d.num_nodes)
        if int(obj.n_atoms) != n:
            n_atoms_mismatch += 1
            continue
        atom_types = np.argmax(np.asarray(obj.atom_feat)[:, :21], axis=1)
        x_types = d.x.reshape(-1).long().numpy()
        max_atom_diff = max(max_atom_diff, int(np.abs(atom_types - x_types).max()))
        bond_map = {(int(u), int(v)): int(t) for (u, v), t in zip(np.asarray(obj.bond_index), np.asarray(obj.bond_type))}
        for u, v, t in zip(d.edge_index[0].tolist(), d.edge_index[1].tolist(), d.edge_attr.reshape(-1).tolist()):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            if key not in bond_map or bond_map[key] != int(t):
                max_bond_diff += 1

    # Frozen label vector vs official train labels.
    frozen_y = load_p3_train_y()
    official_y = np.asarray([float(train_ds[i].y.reshape(-1)[0]) for i in range(10000)])
    label_max_abs = float(np.abs(frozen_y - official_y).max())

    # Historical valid predictions are an alignment check only.
    valid_labels = split_labels(build_data("valid", rrwp=False))
    alignment = {}
    for seed, path in C.P3_HISTORICAL_VALID_PRED.items():
        z = np.load(path)
        pred = np.asarray(z["pred"]).reshape(-1)
        value = C.mae(pred, valid_labels)
        alignment[str(seed)] = {
            "historical_valid_mae": value,
            "frozen_recorded_valid_mae": C.P3_FROZEN_VALID_MAE[seed],
            "abs_diff": abs(value - C.P3_FROZEN_VALID_MAE[seed]),
            "ok": abs(value - C.P3_FROZEN_VALID_MAE[seed]) < 1e-9,
            "n": int(pred.size),
        }
    if not all(v["ok"] for v in alignment.values()):
        raise RuntimeError(f"valid split alignment failed: {alignment}")

    out = {
        "protocol": C.PROTOCOL_ID,
        "source": "torch_geometric.datasets.ZINC(subset=True)",
        "root": str(C.ZINC_ROOT.relative_to(C.REPO)),
        "splits": pyg_splits,
        "frozen_p3_crosscheck": {
            "n_objects": len(objs),
            "n_atoms_mismatch": int(n_atoms_mismatch),
            "max_atom_type_abs_diff": int(max_atom_diff),
            "bond_type_mismatches": int(max_bond_diff),
            "label_max_abs_diff": label_max_abs,
            "ok": bool(n_atoms_mismatch == 0 and max_atom_diff == 0 and max_bond_diff == 0 and label_max_abs == 0.0),
            "frozen_p3_inputs": verify_p3_inputs(),
        },
        "historical_valid_prediction_alignment": alignment,
        "note": "valid/test labels are never used for feature or parameter fitting",
        "timestamp": C.utcnow(),
    }
    if not out["frozen_p3_crosscheck"]["ok"]:
        raise RuntimeError(f"frozen-cache crosscheck failed: {out['frozen_p3_crosscheck']}")
    if write:
        C.write_json(C.DATA_AUDIT, out)
    return out


def main(argv=None) -> int:
    import argparse
    import time

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["audit", "build"])
    p.add_argument("--split", default="train,valid")
    p.add_argument("--ksteps", type=int, default=21)
    p.add_argument("--rrwp", action="store_true", default=True)
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)
    if args.command == "audit":
        t0 = time.perf_counter()
        out = audit_data(write=not args.no_write)
        C.log(f"data audit ok in {time.perf_counter()-t0:.1f}s: "
              f"train {out['splits']['train']['n']}, valid {out['splits']['valid']['n']}, "
              f"test {out['splits']['test']['n']}")
    else:
        for split in args.split.split(","):
            t0 = time.perf_counter()
            data = build_data(split.strip(), ksteps=args.ksteps, rrwp=args.rrwp)
            C.log(f"{split}: {len(data)} graphs cached in {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
