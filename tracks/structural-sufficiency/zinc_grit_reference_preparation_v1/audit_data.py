#!/usr/bin/env python
"""Audit the official ZINC train split against the research track's cache.

Only the official train split is loaded.  The official valid/test datasets and
loaders are not constructed.  The audit is read-only and computes no model
score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import reference_common as rc


def _label(d) -> float:
    return float(d.y.view(-1)[0])


def _pyg_bonds(d) -> dict[tuple[int, int], int]:
    out = {}
    ei = d.edge_index.cpu()
    ea = d.edge_attr.cpu().view(-1)
    for k in range(ei.size(1)):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u < v:
            out[(u, v)] = int(ea[k])
    return out


def _object_bonds(o) -> dict[tuple[int, int], int]:
    out = {}
    for (u, v), bt in zip(o.bond_index, o.bond_type):
        a, b = int(u), int(v)
        out[(min(a, b), max(a, b))] = int(bt)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=rc.BUNDLE / "data_audit.json")
    p.add_argument("--max-graphs", type=int, default=0)
    args = p.parse_args(argv)

    from torch_geometric.datasets import ZINC

    # GraphGym appends the PyG dataset id, so the official GRIT config's
    # dataset.dir=<repo>/data resolves to <repo>/data/ZINC for PyG-ZINC.
    data_root = rc.REPO / "data" / "ZINC"
    ds = ZINC(root=str(data_root), subset=True, split="train")
    n = len(ds) if args.max_graphs <= 0 else min(args.max_graphs, len(ds))

    objs_path = (rc.REPO / "tracks/structure-contribution/results/"
                 "bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt")
    enc_path = (rc.REPO / "tracks/ksvd/results/"
                "zinc_static_dictionary_pair/cache/encoded_train.pt")
    code_dir = rc.REPO / "tracks/structure-contribution/code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    objects = torch.load(objs_path, map_location="cpu", weights_only=False)
    encoded = torch.load(enc_path, map_location="cpu", weights_only=False)

    atom_types = set()
    bond_types = set()
    nodes = []
    labels = []
    atom_mismatch = []
    bond_mismatch = []
    label_mismatch = []
    object_n_atoms_mismatch = []
    structure_match = 0
    label_match = 0
    max_abs_label_diff = 0.0

    for i in range(n):
        d = ds[i]
        o = objects[i]
        e = encoded[i]
        atom_types.update(int(v) for v in torch.unique(d.x[:, 0]).tolist())
        bond_types.update(int(v) for v in torch.unique(d.edge_attr).tolist())
        nodes.append(int(d.num_nodes))
        y = _label(d)
        labels.append(y)

        obj_atoms = np.argmax(o.atom_feat[:, :21], axis=1).astype(np.int64)
        pyg_atoms = np.asarray(d.x[:, 0].cpu(), dtype=np.int64)
        atoms_ok = bool(np.array_equal(obj_atoms, pyg_atoms))
        bonds_ok = _object_bonds(o) == _pyg_bonds(d)
        if not atoms_ok:
            atom_mismatch.append(i)
        if not bonds_ok:
            bond_mismatch.append(i)
        if atoms_ok and bonds_ok:
            structure_match += 1
        if int(o.n_atoms) != int(d.num_nodes):
            object_n_atoms_mismatch.append(i)

        ey = float(e.y.view(-1)[0])
        diff = abs(ey - y)
        max_abs_label_diff = max(max_abs_label_diff, diff)
        if diff <= 1e-6:
            label_match += 1
        else:
            label_mismatch.append(i)

    y = torch.tensor(labels, dtype=torch.float64)
    valid_labels = (rc.REPO / "tracks/structure-contribution/results/"
                    "bjoc_recipe_tuning_v1/valid_y_read_once.npy")
    payload = {
        "schema": "zinc_grit_reference_data_audit_v1",
        "source": "official PyG ZINC subset=True, split=train",
        "root": str(data_root),
        "test_read": False,
        "valid_dataset_constructed": False,
        "train": {
            "graphs": n,
            "atom_types": sorted(atom_types),
            "bond_types": sorted(bond_types),
            "num_nodes_min": min(nodes),
            "num_nodes_max": max(nodes),
            "num_nodes_mean": sum(nodes) / len(nodes),
            "label_mean": float(y.mean()),
            "label_std_population": float(y.std(unbiased=False)),
            "label_std_sample": float(y.std(unbiased=True)) if n > 1 else None,
            "label_min": float(y.min()),
            "label_max": float(y.max()),
        },
        "split_evidence": {
            "train_rows": len(ds),
            "valid_label_rows_from_cached_npy": int(
                np.load(valid_labels, mmap_mode="r").shape[0])
            if valid_labels.exists() else None,
            "valid_values_read_for_audit": False,
            "official_test_rows": None,
            "official_test_note": "not constructed or read in this preparation",
        },
        "research_cache_comparison": {
            "objects_path": str(objs_path.relative_to(rc.REPO)),
            "encoded_path": str(enc_path.relative_to(rc.REPO)),
            "graphs_compared": n,
            "structure_match": structure_match,
            "label_match": label_match,
            "atom_mismatch_graphs": atom_mismatch[:10],
            "bond_mismatch_graphs": bond_mismatch[:10],
            "label_mismatch_graphs": label_mismatch[:10],
            "object_n_atoms_mismatch_graphs": object_n_atoms_mismatch[:10],
            "max_abs_label_diff": max_abs_label_diff,
            "pass": bool(
                not atom_mismatch and not bond_mismatch and not label_mismatch
                and not object_n_atoms_mismatch
                and structure_match == n and label_match == n
            ),
        },
        "notes": [
            "Atom type is the 0-based index in PyG x[:,0]; the research cache "
            "uses the first 21 columns of atom_feat as its one-hot.",
            "Bond types are compared after undirected deduplication (u < v).",
            "No official valid or test graph is fed to a model here.",
        ],
    }
    rc.write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["research_cache_comparison"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
