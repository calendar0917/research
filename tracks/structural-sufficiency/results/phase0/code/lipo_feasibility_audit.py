"""Real-molecule feasibility audit for SSRC Phase 0 (CPU only, no training).

Checks, using structures only:

* Lipophilicity CSV -> canonical SMILES / Bemis--Murcko scaffolds;
* the frozen DeepChem-2.8.0 scaffold split rows and the label-free row manifest;
* whether a train-only scaffold-grouped internal split is feasible;
* whether the existing label-free ``objects_train.pt`` conversion is reusable;
* exact rooted-radius-r patch extraction on a deterministic 200-molecule
  sample, with timing and vocabulary statistics;
* which label files exist but must not be read.

The official valid/test labels are never loaded.  The split JSON is parsed with
``train_y`` and ``train_label_quantiles`` dropped at parse time.  ``labels.npy``,
``test_metrics.json`` and ``validation_metrics.json`` are only checked for
existence; they are not opened.

Run:

    uv run python tracks/structural-sufficiency/results/phase0/code/lipo_feasibility_audit.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PHASE0 = HERE.parent
ANALYSIS = PHASE0 / "analysis"
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles  # noqa: E402

import ssrc_toy as toy  # noqa: E402

LIPO = REPO / "data/lipophilicity/Lipophilicity.csv"
BJOC = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1"
SPLIT = BJOC / "split_lipo_v1.json"
ROW_MANIFEST = BJOC / "row_manifest.json"
OBJECTS_TRAIN = BJOC / "objects_train.pt"
LABEL_VAULT = REPO / "data/lipophilicity/labels.npy"
TEST_METRICS = BJOC / "test_metrics.json"
VALID_METRICS = BJOC / "validation_metrics.json"
TEST_LOCK = BJOC / "test_read.lock"
VALID_LOCK = BJOC / "validation_read.lock"
ZINC_TRAIN = REPO / "data/ZINC/subset/processed/train.pt"
ESOL_CANDIDATES = sorted(REPO.glob("data/**/*esol*")) + sorted(REPO.glob("data/**/*ESOL*"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_smiles_and_hash(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot canonicalise {smiles!r}")
    can = Chem.MolToSmiles(mol)
    return can, sha256_text(can)


def scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def drop_label_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Parse the split JSON while discarding label payloads immediately."""
    return {k: v for k, v in pairs if k not in {"train_y", "train_label_quantiles"}}


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT.read_text(), object_pairs_hook=drop_label_keys)


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO)),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime_utc": (
            __import__("datetime").datetime.fromtimestamp(path.stat().st_mtime, __import__("datetime").timezone.utc).isoformat()
            if path.exists()
            else None
        ),
    }


def split_and_manifest_audit() -> dict[str, Any]:
    split = load_split()
    manifest = json.loads(ROW_MANIFEST.read_text())
    df = pd.read_csv(LIPO, usecols=["CMPD_CHEMBLID", "smiles"])
    assert len(df) == manifest["n"] == 4200

    canonical, cann_hashes, scaffolds, scaffolds_hashes = [], [], [], []
    t0 = time.perf_counter()
    for smi in df["smiles"].tolist():
        can, can_hash = canonical_smiles_and_hash(smi)
        scaf = scaffold(smi)
        canonical.append(can)
        cann_hashes.append(can_hash)
        scaffolds.append(scaf)
        scaffolds_hashes.append(sha256_text(str(scaf)))
    elapsed = time.perf_counter() - t0

    assert len(set(canonical)) == 4200, "canonical SMILES duplicates"
    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        scaffold_groups[str(s)].append(i)

    mismatches = {
        "canonical_sha256": sum(a != b for a, b in zip(cann_hashes, manifest["canonical_smiles_sha256"])),
        "scaffold_sha256": sum(a != b for a, b in zip(scaffolds_hashes, manifest["scaffold_sha256"])),
        "n_atoms": sum(
            a != b
            for a, b in zip(
                [Chem.MolFromSmiles(s).GetNumAtoms() for s in df["smiles"].tolist()],
                manifest["n_atoms"],
            )
        ),
    }

    train = [int(x) for x in split["train_rows"]]
    valid = [int(x) for x in split["valid_rows"]]
    test = [int(x) for x in split["test_rows"]]
    assert sorted(train + valid + test) == list(range(4200))
    assert len(set(train)) == len(train) and len(set(valid)) == len(valid) and len(set(test)) == len(test)
    cross_split_scaffolds = [
        g for g in scaffold_groups.values() if len({0 if i in train else 1 if i in valid else 2 for i in g}) > 1
    ]
    # Canonical duplicates are asserted zero above, so no cross-split duplicate
    # canonical SMILES can exist.  Keep the field explicit for the report.
    cross_split_smiles = 0
    return {
        "n_total": 4200,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_scaffold_groups": len(scaffold_groups),
        "n_scaffold_groups_train": len({scaffolds[i] for i in train}),
        "n_scaffold_groups_valid": len({scaffolds[i] for i in valid}),
        "n_scaffold_groups_test": len({scaffolds[i] for i in test}),
        "cross_split_scaffold_groups": len(cross_split_scaffolds),
        "cross_split_canonical_smiles": cross_split_smiles,
        "canonical_smiles_duplicates": 4200 - len(set(canonical)),
        "manifest_mismatches": mismatches,
        "canonical_scaffold_seconds": elapsed,
        "split_reference_match": split.get("reference_match"),
        "split_sha256_rows": {
            "train": split.get("train_sha256_rows"),
            "valid": split.get("valid_sha256_rows"),
            "test": split.get("test_sha256_rows"),
        },
        "label_keys_loaded": False,
        "scaffold_group_size_histogram": dict(
            sorted(Counter(len(v) for v in scaffold_groups.values()).items())
        ),
        "train_scaffold_group_size_histogram": dict(
            sorted(Counter(len(v) for k, v in scaffold_groups.items() if any(i in train for i in v)).items())
        ),
        "_train_rows": train,
        "_scaffolds": scaffolds,
        "_scaffold_groups": scaffold_groups,
    }


def candidate_internal_split(
    train_rows: Sequence[int], scaffolds: Sequence[str]
) -> dict[str, Any]:
    """Group-disjoint 70/15/15 candidate over the official TRAIN rows only.

    This is deliberately labelled CANDIDATE_ONLY_NOT_A_PROTOCOL; it uses no
    labels and is not frozen.  A future pilot must freeze its own split before
    any training.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for row in train_rows:
        groups[str(scaffolds[row])].append(int(row))
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    targets = {"fit": int(round(0.70 * len(train_rows))), "valid": int(round(0.15 * len(train_rows))), "locked": int(round(0.15 * len(train_rows)))}
    assigned = {"fit": [], "valid": [], "locked": []}
    for scaf, rows in ordered:
        capacities = {k: targets[k] - len(assigned[k]) for k in targets}
        dest = max(("fit", "valid", "locked"), key=lambda k: (capacities[k], -("fit", "valid", "locked").index(k)))
        assigned[dest].extend(rows)
    for k in assigned:
        assigned[k].sort()
    all_rows = sorted(assigned["fit"] + assigned["valid"] + assigned["locked"])
    assert all_rows == sorted(int(r) for r in train_rows)
    # group disjointness
    group_sets = {k: {str(scaffolds[r]) for r in v} for k, v in assigned.items()}
    overlap = {
        "fit_valid": len(group_sets["fit"] & group_sets["valid"]),
        "fit_locked": len(group_sets["fit"] & group_sets["locked"]),
        "valid_locked": len(group_sets["valid"] & group_sets["locked"]),
    }
    return {
        "status": "CANDIDATE_ONLY_NOT_A_PROTOCOL",
        "algorithm": "greedy group-disjoint target 70/15/15; sort groups by (-size, scaffold); assign to largest remaining capacity",
        "targets": targets,
        "sizes": {k: len(v) for k, v in assigned.items()},
        "n_groups": {k: len(group_sets[k]) for k in assigned},
        "scaffold_group_overlap": overlap,
        "row_index_sha256": {
            k: hashlib.sha256(",".join(map(str, v)).encode()).hexdigest() for k, v in assigned.items()
        },
        "rows": {k: v for k, v in assigned.items()},
        "labels_used": False,
    }


def objects_audit() -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO / "tracks/structure-contribution/code"))
    objs = torch.load(OBJECTS_TRAIN, weights_only=False)
    attrs = sorted(objs[0].__dict__)
    no_label_attr = not any(k in {"y", "label", "target", "exp"} for k in attrs)
    # Compare the label-free conversion to a fresh RDKit parse for all rows.
    # ``MolObjects.smiles`` is the source (not canonical) SMILES, so compare
    # canonical forms against the source CSV row at ``mol_index``.
    df = pd.read_csv(LIPO, usecols=["smiles"])
    mismatches_atoms = 0
    mismatches_smiles = 0
    for o in objs:
        mol = Chem.MolFromSmiles(o.smiles)
        source = Chem.MolFromSmiles(df["smiles"].iloc[int(o.mol_index)])
        if mol is None or source is None:
            mismatches_smiles += 1
            continue
        mismatches_atoms += int(mol.GetNumAtoms() != o.n_atoms)
        mismatches_smiles += int(Chem.MolToSmiles(mol) != Chem.MolToSmiles(source))
    return {
        "n_objects": len(objs),
        "object_class": type(objs[0]).__name__,
        "attributes": attrs,
        "no_label_attribute": no_label_attr,
        "atom_feat_shape": list(objs[0].atom_feat.shape),
        "bond_feat_shape": list(objs[0].bond_feat.shape),
        "global_feat_shape": list(objs[0].global_feat.shape),
        "n_atoms_mismatches_vs_rdkit": mismatches_atoms,
        "canonical_smiles_mismatches_vs_source_csv": mismatches_smiles,
        "mol_index_first_10": [int(o.mol_index) for o in objs[:10]],
    }


def molecule_graph(mol: Chem.Mol) -> tuple[list[tuple[int, ...]], list[tuple[int, int, tuple[int, ...]]]]:
    """Build node/edge colour tuples for a molecular graph (no learned features)."""
    node_colors: list[tuple[int, ...]] = []
    for atom in mol.GetAtoms():
        node_colors.append(
            (
                int(atom.GetAtomicNum()),
                int(atom.GetFormalCharge()) + 2,
                int(atom.GetIsAromatic()),
                int(atom.GetTotalNumHs()),
                int(atom.IsInRing()),
            )
        )
    edges: list[tuple[int, int, tuple[int, ...]]] = []
    for bond in mol.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        bt_code = {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}.get(float(bt), 5)
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((min(u, v), max(u, v), (bt_code, int(bond.IsInRing()))))
    return node_colors, edges


def rooted_encoding_from_mol(mol: Chem.Mol, radius: int, return_sizes: bool = False):
    node_colors, edges = molecule_graph(mol)
    n = len(node_colors)
    adj: dict[int, list[tuple[int, tuple[int, ...]]]] = {v: [] for v in range(n)}
    for u, v, ec in edges:
        adj[u].append((v, ec))
        adj[v].append((u, ec))
    keys: list[bytes] = []
    sizes: list[int] = []
    for root in range(n):
        dist = {root: 0}
        queue = [root]
        for v in queue:
            for u, _ in adj[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        nodes = [root] + sorted(v for v, d in dist.items() if d <= radius and v != root)
        local = {v: i for i, v in enumerate(nodes)}
        local_colors = []
        for v in nodes:
            base = node_colors[v]
            local_colors.append(base + (1 if v == root else 0, int(dist[v])))
        local_edges = []
        for u, v, ec in edges:
            if u in local and v in local:
                iu, iv = local[u], local[v]
                local_edges.append((min(iu, iv), max(iu, iv), ec))
        keys.append(toy.pynauty_key_from_colors(local_colors, local_edges))
        sizes.append(len(nodes))
    if return_sizes:
        return tuple(sorted(keys)), sizes
    return tuple(sorted(keys))


def prototype_er_audit() -> dict[str, Any]:
    split = load_split()
    df = pd.read_csv(LIPO, usecols=["smiles"])
    train_rows = [int(x) for x in split["train_rows"]]
    sample_rows = train_rows[:200]
    sample = [Chem.MolFromSmiles(df["smiles"].iloc[r]) for r in sample_rows]
    assert all(m is not None for m in sample)

    out: dict[str, Any] = {
        "sample_rows": sample_rows,
        "sample_size": len(sample),
        "n_atoms": [m.GetNumAtoms() for m in sample],
        "by_radius": {},
    }
    for r in (2, 3):
        t0 = time.perf_counter()
        results = [rooted_encoding_from_mol(m, r, return_sizes=True) for m in sample]
        encodings = [keys for keys, _ in results]
        patch_sizes = [s for _, sizes in results for s in sizes]
        elapsed = time.perf_counter() - t0
        all_keys = [k for enc in encodings for k in enc]
        counts = Counter(all_keys)
        out["by_radius"][f"r{r}"] = {
            "seconds": elapsed,
            "seconds_per_molecule": elapsed / len(sample),
            "projected_seconds_3360": elapsed / len(sample) * 3360,
            "n_patches": len(all_keys),
            "n_unique_patch_keys": len(counts),
            "unique_fraction": len(counts) / len(all_keys),
            "singleton_patch_fraction": sum(1 for c in counts.values() if c == 1) / len(counts),
            "max_patch_nodes": max(patch_sizes),
            "mean_patch_nodes": sum(patch_sizes) / len(patch_sizes),
            "mean_patches_per_molecule": len(all_keys) / len(sample),
            "top_patch_counts": [int(c) for _, c in counts.most_common(10)],
        }

    # Relabelling invariance on a deterministic 20-molecule subset.
    rng = random.Random(20261018)
    failures: list[str] = []
    relabel_n = 20
    for idx in range(relabel_n):
        mol = sample[idx]
        n = mol.GetNumAtoms()
        order = list(range(n))
        rng.shuffle(order)
        mol2 = Chem.RenumberAtoms(mol, order)
        for r in (2, 3):
            if rooted_encoding_from_mol(mol, r) != rooted_encoding_from_mol(mol2, r):
                failures.append(f"mol={sample_rows[idx]} r={r}")
    out["relabel_invariance"] = {
        "n_molecules": relabel_n,
        "radii": [2, 3],
        "failures": failures,
        "passed": not failures,
    }
    return out


def resources_and_reuse() -> dict[str, Any]:
    import torch

    flop_counter_available = hasattr(__import__("torch.utils.flop_counter", fromlist=["FlopCounterMode"]), "FlopCounterMode")
    provenance = json.loads((BJOC / "provenance.json").read_text())
    return {
        "raw_graph_conversion_seconds_full_4200": provenance.get("raw_graph_build_seconds"),
        "rdkit_version": provenance.get("rdkit_version"),
        "torch_version": torch.__version__,
        "flop_counter_available": flop_counter_available,
        "zinc_train_exists": ZINC_TRAIN.exists(),
        "esol_candidates": [str(p.relative_to(REPO)) for p in ESOL_CANDIDATES],
        "official_valid_test_artifacts": {
            "label_vault": file_info(LABEL_VAULT),
            "test_metrics": file_info(TEST_METRICS),
            "validation_metrics": file_info(VALID_METRICS),
            "test_read_lock": file_info(TEST_LOCK),
            "validation_read_lock": file_info(VALID_LOCK),
        },
        "label_files_opened": False,
    }


def main() -> int:
    t0 = time.perf_counter()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    split_audit = split_and_manifest_audit()
    candidate = candidate_internal_split(split_audit["_train_rows"], split_audit["_scaffolds"])
    objects = objects_audit()
    er = prototype_er_audit()
    resources = resources_and_reuse()

    # Drop private helper fields from the serialised split audit.
    public_split = {k: v for k, v in split_audit.items() if not k.startswith("_")}
    payload: dict[str, Any] = {
        "protocol": "ssrc-phase0-lipo-feasibility-v1",
        "date_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "scope": {
            "cpu_only": True,
            "training": False,
            "official_valid_labels_read": False,
            "official_test_labels_read": False,
            "internal_locked_labels_read": False,
            "gpu_used": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rdkit": __import__("rdkit").__version__,
            "pandas": pd.__version__,
        },
        "split_manifest_audit": public_split,
        "candidate_internal_split": candidate,
        "objects_audit": objects,
        "prototype_rooted_encodings": er,
        "resources_and_reuse": resources,
        "runtime_seconds": time.perf_counter() - t0,
    }
    out = ANALYSIS / "lipo_feasibility_audit.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("SSRC Phase 0 Lipophilicity feasibility audit")
    print(f"  output: {out}")
    print(f"  split: {public_split['n_train']}/{public_split['n_valid']}/{public_split['n_test']} "
          f"scaffold_groups={public_split['n_scaffold_groups']} cross_split={public_split['cross_split_scaffold_groups']}")
    print(f"  manifest mismatches: {public_split['manifest_mismatches']}")
    print(f"  objects: {objects['n_objects']} no_label_attr={objects['no_label_attribute']} "
          f"atom_mismatch={objects['n_atoms_mismatches_vs_rdkit']} "
          f"smiles_mismatch={objects['canonical_smiles_mismatches_vs_source_csv']}")
    print(f"  candidate internal split: {candidate['sizes']} groups={candidate['n_groups']} overlap={candidate['scaffold_group_overlap']}")
    for r, row in er["by_radius"].items():
        print(f"  E_{r}: {row['seconds_per_molecule']:.4f}s/mol, patches={row['n_patches']}, "
              f"unique={row['n_unique_patch_keys']}, max_patch={row['max_patch_nodes']}")
    print(f"  relabel invariance: {er['relabel_invariance']['passed']} failures={len(er['relabel_invariance']['failures'])}")
    print(f"  official label vault present={resources['official_valid_test_artifacts']['label_vault']['exists']} "
          f"(not opened); test metrics present={resources['official_valid_test_artifacts']['test_metrics']['exists']} (not opened)")
    print(f"  ESOL candidates: {resources['esol_candidates'] or 'none'}")
    print(f"  runtime: {payload['runtime_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
