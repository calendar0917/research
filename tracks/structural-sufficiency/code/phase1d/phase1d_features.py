"""Phase 1d CPU structural features for the frozen ZINC Monitor molecules.

These features are structure-only.  They never read a label, the internal
Reserved split, or official ZINC valid/test.  The feature list and the grouping
rules are frozen in ``notes/phase1d_protocol.md`` before any Phase 1d GPU run.

The features are descriptive audit variables.  They are *not* model inputs and
are not a license to add a new module after seeing an error correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
for p in (HERE, PHASE1B):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssrc_data import CACHE_DIR, load_cache, sha256_file  # noqa: E402

CACHE_PATH = CACHE_DIR / "zinc_phase1b.pt"
SPLIT_PATH = REPO / "tracks/structural-sufficiency/results/phase1b/split_zinc_phase1b.json"
OUT_DIR = REPO / "tracks/structural-sufficiency/results/phase1d"
FEATURES_PATH = OUT_DIR / "structural_features.npz"
MANIFEST_PATH = OUT_DIR / "structural_features_manifest.json"

# ZINC-12k raw bond_type values are 1/2/3 in the 10k subset; the frozen
# cache is a 4-slot one-hot with slot 0 unused (no explicit aromatic type).
ZINC_BOND_TYPES = ("unused0", "single", "double", "triple")


# ---------------------------------------------------------------------------
# exact rooted-patch canonical key (pynauty, same convention as Phase 0)
# ---------------------------------------------------------------------------


def _pynauty_patch_key(node_colors: Sequence[tuple[int, ...]], edges: Sequence[tuple[int, int, tuple[int, ...]]]) -> bytes:
    import pynauty

    n = len(node_colors)
    m = len(edges)
    nv = n + m
    colors: list[tuple[int, ...]] = [(0,) + tuple(int(x) for x in c) for c in node_colors]
    adjacency: dict[int, list[int]] = {v: [] for v in range(nv)}
    for e, (u, v, edge_color) in enumerate(edges):
        edge_vertex = n + e
        colors.append((1,) + tuple(int(x) for x in edge_color))
        for endpoint in (u, v):
            adjacency[edge_vertex].append(int(endpoint))
            adjacency[int(endpoint)].append(edge_vertex)
    groups: dict[tuple[int, ...], list[int]] = {}
    for v, color in enumerate(colors):
        groups.setdefault(color, []).append(v)
    cell_keys = sorted(groups, key=repr)
    coloring = [set(groups[k]) for k in cell_keys]
    pg = pynauty.Graph(
        number_of_vertices=nv,
        directed=False,
        adjacency_dict={v: sorted(adj) for v, adj in adjacency.items()},
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(pg))
    label = [int(v) for v in pynauty.canon_label(pg)]
    if sorted(label) != list(range(nv)):
        raise RuntimeError("pynauty.canon_label is not a permutation")
    color_sequence = tuple(colors[v] for v in label)
    return certificate + b"|" + repr(color_sequence).encode()


def _entropy_bits(counts: Iterable[int]) -> float:
    c = np.asarray([int(x) for x in counts if int(x) > 0], dtype=np.float64)
    if c.size == 0 or c.sum() <= 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def _bfs_distances(adj: Sequence[Sequence[int]], src: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int64)
    dist[src] = 0
    q = deque([src])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def _rdkit_ring_features(
    n_atoms: int,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_type: np.ndarray,
) -> dict[str, float]:
    from rdkit import Chem

    rw = Chem.RWMol()
    for _ in range(n_atoms):
        rw.AddAtom(Chem.Atom(6))
    bt = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    for u, v, t in zip(bond_u.tolist(), bond_v.tolist(), bond_type.tolist()):
        rw.AddBond(int(u), int(v), bt[int(t) % 4])
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    ri = mol.GetRingInfo()
    atom_rings = [tuple(int(x) for x in r) for r in ri.AtomRings()]
    # RDKit's BondRings() returns one tuple per ring (ring -> bond ids), not
    # one tuple per bond.  Fusion needs multiplicities over ring memberships.
    ring_bond_membership = [bid for ring in ri.BondRings() for bid in ring]
    ring_atom_membership = [aid for ring in ri.AtomRings() for aid in ring]
    ring_sizes = [len(r) for r in atom_rings]
    fusion_bonds = sum(1 for c in Counter(ring_bond_membership).values() if c >= 2)
    shared_atoms = sum(1 for c in Counter(ring_atom_membership).values() if c >= 2)
    ring_bonds = len(set(ring_bond_membership))
    type_by_pair = {
        (int(min(u, v)), int(max(u, v))): int(t)
        for u, v, t in zip(bond_u.tolist(), bond_v.tolist(), bond_type.tolist())
    }
    kekule6 = 0
    for ring in atom_rings:
        if len(ring) != 6:
            continue
        types = []
        for i in range(6):
            a, b = int(ring[i]), int(ring[(i + 1) % 6])
            types.append(type_by_pair.get((min(a, b), max(a, b)), -1))
        if types.count(2) == 3 and types.count(1) == 3:
            kekule6 += 1
    return {
        "rings_sssr": float(len(atom_rings)),
        "max_ring_size": float(max(ring_sizes)) if ring_sizes else 0.0,
        "ring_fusion_bonds": float(fusion_bonds),
        "ring_shared_atoms": float(shared_atoms),
        "large_cycles_gt6": float(sum(1 for s in ring_sizes if s > 6)),
        "frac_ring_atoms": float(len(set(ring_atom_membership)) / n_atoms) if n_atoms else 0.0,
        "kekule6_ring_count": float(kekule6),
        "ring_bond_fraction": float(ring_bonds / max(len(bond_type), 1)),
    }


def graph_features(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    """Simple graph-level features, label-free, from the frozen cache."""
    rows = np.asarray(rows, dtype=np.int64)
    n = rows.size
    out: dict[str, list[float]] = {
        k: []
        for k in [
            "n_atoms",
            "n_bonds",
            "rings_cyclomatic",
            "rings_sssr",
            "max_ring_size",
            "ring_fusion_bonds",
            "ring_shared_atoms",
            "large_cycles_gt6",
            "frac_ring_atoms",
            "kekule6_ring_count",
            "ring_bond_fraction",
            "mean_degree",
            "max_degree",
            "min_degree",
            "frac_degree1",
            "diameter",
            "mean_pair_distance",
            "frac_pairs_distance_gt4",
            "unused0_bonds",
            "single_bonds",
            "double_bonds",
            "triple_bonds",
            "double_bond_fraction",
            "atom_type_entropy",
            "bond_type_entropy",
        ]
    }
    atom_feat = cache["atom_feat_flat"]
    root_src = cache["root_edge_src_flat"].numpy()
    root_dst = cache["root_edge_dst_flat"].numpy()
    root_attr = cache["root_edge_attr_flat"].numpy()
    atom_start = cache["atom_start"].numpy()
    edge_start = cache["root_edge_start"].numpy()
    mol_n_atoms = cache["mol_n_atoms"].numpy()

    for mol in rows.tolist():
        n_atoms = int(mol_n_atoms[mol])
        a0, a1 = int(atom_start[mol]), int(atom_start[mol + 1])
        atom_types = atom_feat[a0:a1].float().argmax(dim=1).numpy().astype(np.int64)
        s0, s1 = int(edge_start[mol]), int(edge_start[mol + 1])
        src = root_src[s0:s1]
        dst = root_dst[s0:s1]
        attr = root_attr[s0:s1]
        keep = src < dst
        bu = src[keep].astype(np.int64)
        bv = dst[keep].astype(np.int64)
        bt = attr[keep].argmax(axis=1).astype(np.int64)
        n_bonds = int(bu.size)

        adj: list[list[int]] = [[] for _ in range(n_atoms)]
        for u, v in zip(bu.tolist(), bv.tolist()):
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        degrees = np.asarray([len(x) for x in adj], dtype=np.float64)
        if n_atoms > 1:
            dists = np.stack([_bfs_distances(adj, i) for i in range(n_atoms)], axis=0)
            pair_mask = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
            pair_d = dists[pair_mask]
            diameter = float(pair_d.max())
            mean_pair = float(pair_d.mean())
            frac_gt4 = float(np.mean(pair_d > 4))
        else:
            diameter = 0.0
            mean_pair = 0.0
            frac_gt4 = 0.0

        ring = _rdkit_ring_features(n_atoms, bu, bv, bt)
        counts = Counter(bt.tolist())
        atom_counts = Counter(atom_types.tolist())
        out["n_atoms"].append(float(n_atoms))
        out["n_bonds"].append(float(n_bonds))
        out["rings_cyclomatic"].append(float(n_bonds - n_atoms + 1))
        out["rings_sssr"].append(ring["rings_sssr"])
        out["max_ring_size"].append(ring["max_ring_size"])
        out["ring_fusion_bonds"].append(ring["ring_fusion_bonds"])
        out["ring_shared_atoms"].append(ring["ring_shared_atoms"])
        out["large_cycles_gt6"].append(ring["large_cycles_gt6"])
        out["frac_ring_atoms"].append(ring["frac_ring_atoms"])
        out["kekule6_ring_count"].append(ring["kekule6_ring_count"])
        out["ring_bond_fraction"].append(ring["ring_bond_fraction"])
        out["mean_degree"].append(float(degrees.mean()) if degrees.size else 0.0)
        out["max_degree"].append(float(degrees.max()) if degrees.size else 0.0)
        out["min_degree"].append(float(degrees.min()) if degrees.size else 0.0)
        out["frac_degree1"].append(float(np.mean(degrees == 1)) if degrees.size else 0.0)
        out["diameter"].append(diameter)
        out["mean_pair_distance"].append(mean_pair)
        out["frac_pairs_distance_gt4"].append(frac_gt4)
        for k, name in enumerate(ZINC_BOND_TYPES):
            out[f"{name}_bonds"].append(float(counts.get(k, 0)))
        out["double_bond_fraction"].append(float(counts.get(2, 0) / n_bonds) if n_bonds else 0.0)
        out["atom_type_entropy"].append(_entropy_bits(atom_counts.values()))
        out["bond_type_entropy"].append(_entropy_bits([counts.get(1, 0), counts.get(2, 0), counts.get(3, 0)]))

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def patch_features(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    """E2 patch size/repetition/root-pooling features, label-free."""
    rows = np.asarray(rows, dtype=np.int64)
    out: dict[str, list[float]] = {
        k: []
        for k in [
            "patch_size_mean",
            "patch_size_max",
            "patch_size_std",
            "patch_whole_fraction",
            "patch_dup_rate",
            "patch_key_entropy",
            "patch_max_multiplicity",
            "patch_unique_count",
            "mean_pool_bits",
            "root_vs_patch_tv_mean",
            "patch_atom_entropy_mean",
            "first_shell_mean",
            "first_shell_max",
            "second_shell_mean",
        ]
    }
    mol_patch_start = cache["mol_patch_start"].numpy()
    node_start = cache["patch_node_start"].numpy()
    edge_start = cache["patch_edge_start"].numpy()
    patch_src = cache["patch_edge_src"].numpy()
    patch_dst = cache["patch_edge_dst"].numpy()
    patch_attr = cache["patch_edge_attr"].numpy()
    patch_feat = cache["patch_node_feat"].numpy() if "patch_node_feat" in cache else cache["patch_node_feat_flat"].numpy()
    mol_n_atoms = cache["mol_n_atoms"].numpy()

    for mol in rows.tolist():
        n_atoms = int(mol_n_atoms[mol])
        p0, p1 = int(mol_patch_start[mol]), int(mol_patch_start[mol + 1])
        sizes: list[int] = []
        first_shell: list[int] = []
        second_shell: list[int] = []
        whole = 0
        keys: list[bytes] = []
        tvs: list[float] = []
        patch_entropies: list[float] = []
        for pid in range(p0, p1):
            a0, a1 = int(node_start[pid]), int(node_start[pid + 1])
            e0, e1 = int(edge_start[pid]), int(edge_start[pid + 1])
            nf = patch_feat[a0:a1]
            n_nodes = int(nf.shape[0])
            sizes.append(n_nodes)
            if n_nodes == n_atoms:
                whole += 1
            atom_idx = nf[:, :21].argmax(axis=1).astype(np.int64)
            root_flag = (nf[:, 21] > 0.5).astype(np.int64)
            dist_idx = nf[:, 22:25].argmax(axis=1).astype(np.int64)
            first_shell.append(int(np.sum(dist_idx == 1)))
            second_shell.append(int(np.sum(dist_idx == 2)))
            # exact rooted node/edge-colour key
            node_colors = [
                (int(atom_idx[i]), int(root_flag[i]), int(dist_idx[i])) for i in range(n_nodes)
            ]
            edges: list[tuple[int, int, tuple[int, ...]]] = []
            for k in range(e0, e1):
                u, v = int(patch_src[k]), int(patch_dst[k])
                if u < v:
                    t = int(patch_attr[k].argmax())
                    edges.append((u, v, (t,)))
            keys.append(_pynauty_patch_key(node_colors, edges))
            # root-vs-patch composition divergence and within-patch entropy
            counts = np.bincount(atom_idx, minlength=21).astype(np.float64)
            p = counts / max(counts.sum(), 1.0)
            q = np.zeros_like(p)
            q[int(atom_idx[0])] = 1.0
            tvs.append(float(0.5 * np.abs(p - q).sum()))
            patch_entropies.append(_entropy_bits(counts.astype(int).tolist()))

        counts = Counter(keys)
        n_patches = max(len(keys), 1)
        mult = np.asarray(list(counts.values()), dtype=np.float64) if counts else np.zeros(0)
        out["patch_size_mean"].append(float(np.mean(sizes)) if sizes else 0.0)
        out["patch_size_max"].append(float(np.max(sizes)) if sizes else 0.0)
        out["patch_size_std"].append(float(np.std(sizes)) if sizes else 0.0)
        out["patch_whole_fraction"].append(float(whole / n_patches))
        out["patch_dup_rate"].append(float(1.0 - len(counts) / n_patches) if keys else 0.0)
        out["patch_key_entropy"].append(_entropy_bits(mult.astype(int).tolist()))
        out["patch_max_multiplicity"].append(float(mult.max()) if mult.size else 0.0)
        out["patch_unique_count"].append(float(len(counts)))
        out["mean_pool_bits"].append(float(np.mean(np.log2(np.maximum(sizes, 1)))) if sizes else 0.0)
        out["root_vs_patch_tv_mean"].append(float(np.mean(tvs)) if tvs else 0.0)
        out["patch_atom_entropy_mean"].append(float(np.mean(patch_entropies)) if patch_entropies else 0.0)
        out["first_shell_mean"].append(float(np.mean(first_shell)) if first_shell else 0.0)
        out["first_shell_max"].append(float(np.max(first_shell)) if first_shell else 0.0)
        out["second_shell_mean"].append(float(np.mean(second_shell)) if second_shell else 0.0)

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def all_features(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    g = graph_features(cache, rows)
    p = patch_features(cache, rows)
    g.update(p)
    return g


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", choices=["all", "monitor", "fit"], default="all")
    ap.add_argument("--out", default=str(FEATURES_PATH))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = ap.parse_args()

    cache = load_cache(CACHE_PATH)
    split = json.loads(SPLIT_PATH.read_text())
    if args.rows == "monitor":
        rows = [int(x) for x in split["rows"]["monitor"]]
    elif args.rows == "fit":
        rows = [int(x) for x in split["rows"]["fit"]]
    else:
        rows = list(range(10000))
    feats = all_features(cache, rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=np.asarray(rows, dtype=np.int64), **feats)
    manifest = {
        "protocol_id": "ssrc-phase1d-cpu-features",
        "scope": {
            "structure_only": True,
            "labels_read": False,
            "official_valid_test_read": False,
            "internal_reserved_read": False,
        },
        "cache": {"path": str(CACHE_PATH.relative_to(REPO)), "sha256": sha256_file(CACHE_PATH)},
        "split": {"path": str(SPLIT_PATH.relative_to(REPO)), "sha256": sha256_file(SPLIT_PATH)},
        "rows": {"kind": args.rows, "n": len(rows)},
        "features": sorted(feats),
        "feature_path": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
        "feature_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
