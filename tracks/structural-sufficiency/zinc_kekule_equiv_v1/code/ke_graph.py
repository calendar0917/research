"""Label-free graph / feature / inference helpers for the Kekule audit.

This module deliberately has **no RDKit dependency**: it is imported by the
remote inference job, which only consumes the verified representation artifact
produced locally by :mod:`ke_chem` / :mod:`ke_prep`.

Nothing here reads labels from the official test split.  The only label read is
the official-valid ``y`` used by the frozen-model scoring helpers, and the
caller may omit it entirely for feature construction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SS = REPO / "tracks/structural-sufficiency"

# import paths used by the frozen P3 / BF / SF implementations
_PATHS = (
    REPO / "tracks/structure-contribution/code",
    SS / "zinc_arch_discovery_v1/code",
    SS / "zinc_recipe_trunk_v2/code",
    SS / "zinc_branching_fragment_v1/code",
    SS / "zinc_local_factorization_v1/code",
    SS / "zinc_shared_factor_v1/code",
)
for _p in _PATHS:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAX_REPS = 8
MAX_EDGES = 64
MAX_ATOMS = 48


# ---------------------------------------------------------------------------
# official PyG graphs (train / valid only)
# ---------------------------------------------------------------------------


def load_pyg(split: str):
    from torch_geometric.datasets import ZINC

    if split not in ("train", "valid"):
        raise ValueError("only official train / valid may be loaded in this audit")
    return ZINC(root=str(REPO / "data/ZINC"), subset=True,
                split=("val" if split == "valid" else split))


def graph_arrays(d) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray,
                             np.ndarray, np.ndarray]:
    """Return ``(atom_categories, undirected_edges, bond_orders, edge_index, edge_attr)``.

    ``edges`` are sorted by ``(u, v)`` with ``u < v``; ``bond_orders`` are the
    committed PyG bond words (1/2/3).  The directed PyG arrays are kept so the
    remote job can rebuild the exact same edge list after changing orders.
    """
    cats = np.asarray(d.x).reshape(-1).astype(np.int64)
    ei = np.asarray(d.edge_index, dtype=np.int64)
    ea = np.asarray(d.edge_attr).reshape(-1).astype(np.int64)
    orders: dict[tuple[int, int], int] = {}
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            raise RuntimeError(f"self loop at atom {u}")
        if u > v:
            u, v = v, u
        t = int(ea[k])
        if t not in (1, 2, 3):
            raise RuntimeError(f"unsupported bond order {t}")
        old = orders.setdefault((u, v), t)
        if old != t:
            raise RuntimeError(f"conflicting directed bond orders for {(u, v)}")
    edges = sorted(orders)
    edge_orders = np.asarray([orders[e] for e in edges], dtype=np.int64)
    return cats, edges, edge_orders, ei, ea


def graph_sha256(cats: np.ndarray, ei: np.ndarray, ea: np.ndarray) -> str:
    """Stable per-molecule hash used to align the local artifact and remote data."""
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cats, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(ei, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(ea, dtype=np.int64).tobytes())
    return h.hexdigest()


def apply_orders(edge_index: np.ndarray, edge_attr: np.ndarray,
                 edges: Sequence[tuple[int, int]], orders: np.ndarray) -> np.ndarray:
    """Map an undirected order vector onto the directed PyG ``edge_attr`` array."""
    table = {tuple(map(int, e)): int(o) for e, o in zip(edges, orders)}
    out = np.asarray(edge_attr).reshape(-1).copy().astype(np.int64)
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u > v:
            u, v = v, u
        out[k] = table[(u, v)]
    return out


def replace_order(edges: Sequence[tuple[int, int]], orders: np.ndarray,
                  new: Mapping[tuple[int, int], int]) -> np.ndarray:
    out = np.asarray(orders).copy()
    index = {tuple(map(int, e)): k for k, e in enumerate(edges)}
    for e, o in new.items():
        out[index[tuple(map(int, e))]] = int(o)
    return out


# ---------------------------------------------------------------------------
# the exact old radius-2 content key
# ---------------------------------------------------------------------------


def radius_keys(cats: np.ndarray, edges: Sequence[tuple[int, int]],
                orders: np.ndarray) -> list[tuple]:
    """The radius-2 key used by ``zinc_generalization_audit_v1``.

    ``radius-1: (central category, sorted (bond type, neighbour category))``;
    ``radius-2: (central category, sorted (bond type, radius-1 neighbour))``.
    Exact nested tuples, no atom index, no label.
    """
    n = len(cats)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (u, v), o in zip(edges, orders):
        u, v, o = int(u), int(v), int(o)
        adj[u].append((v, o))
        adj[v].append((u, o))
    for a in adj:
        a.sort()
    s1 = [(int(cats[v]), tuple(sorted((t, int(cats[u])) for u, t in adj[v])))
          for v in range(n)]
    s2 = [(int(cats[v]), tuple(sorted((t, s1[u]) for u, t in adj[v])))
          for v in range(n)]
    return s2


# ---------------------------------------------------------------------------
# compact P3 cache + frozen model inference
# ---------------------------------------------------------------------------


def build_objects(cats: np.ndarray, edge_index: np.ndarray,
                  edge_attr: np.ndarray, mol_index: int):
    import structural_relation_objects_v1 as sro

    return sro.build_mol_objects(
        int(mol_index),
        np.asarray(cats, dtype=np.int64),
        np.asarray(edge_index, dtype=np.int64),
        np.asarray(edge_attr, dtype=np.int64),
    )


def compact_cache(objs: Sequence[Any], labels: np.ndarray | None, split: str) -> dict:
    """The subset of the frozen ``ad_data`` cache consumed by P3 / BF models.

    This is rebuilt from the modified raw graph for every representation; it is
    not a patch of the cached original features.
    """
    n = len(objs)
    atom_ptr = np.zeros(n + 1, dtype=np.int32)
    bond_ptr = np.zeros(n + 1, dtype=np.int32)
    junc_ptr = np.zeros(n + 1, dtype=np.int32)
    af, at, bf, bt, bi, jf, ji, jb, gf = [], [], [], [], [], [], [], [], []
    for i, o in enumerate(objs):
        atom_ptr[i + 1] = atom_ptr[i] + int(o.n_atoms)
        bond_ptr[i + 1] = bond_ptr[i] + int(len(o.bond_type))
        junc_ptr[i + 1] = junc_ptr[i] + int(len(o.junc_index))
        feat = np.asarray(o.atom_feat, dtype=np.float32)
        af.append(feat)
        at.append(np.argmax(feat[:, :28], axis=1).astype(np.int64))
        bf.append(np.asarray(o.bond_feat, dtype=np.float32))
        bt.append(np.asarray(o.bond_type, dtype=np.int64))
        bi.append(np.asarray(o.bond_index, dtype=np.int16))
        jf.append(np.asarray(o.junc_feat, dtype=np.float32))
        ji.append(np.asarray(o.junc_index, dtype=np.int16))
        jb.append(np.asarray(o.junc_bond, dtype=np.int16))
        gf.append(np.asarray(o.global_feat, dtype=np.float32))
    return {
        "split": str(split),
        "n_molecules": int(n),
        "labels": (np.zeros(n, dtype=np.float32) if labels is None
                   else np.asarray(labels, dtype=np.float32)),
        "atom_ptr": atom_ptr,
        "atom_feat": np.concatenate(af, axis=0),
        "atom_type": np.concatenate(at, axis=0),
        "bond_ptr": bond_ptr,
        "bond_feat": np.concatenate(bf, axis=0),
        "bond_type": np.concatenate(bt, axis=0),
        "bond_index": np.concatenate(bi, axis=0),
        "junc_ptr": junc_ptr,
        "junc_feat": np.concatenate(jf, axis=0),
        "junc_index": np.concatenate(ji, axis=0),
        "junc_bond": np.concatenate(jb, axis=0),
        "global_feat": np.stack(gf, axis=0),
    }


def load_models(device):
    """Load the two frozen checkpoints named by the round brief."""
    import torch
    import ad_data as D
    import ad2_models as A2
    import sf_models as SF

    norm = D.load_norm()
    p3, p3_info = A2.build_model("P3N", norm, 0, device)
    p3_ckpt = SS / "zinc_recipe_trunk_v2/runs/xBssl_P3N_s0/best.pt"
    p3_state = torch.load(p3_ckpt, map_location="cpu", weights_only=False)
    p3.load_state_dict(p3_state["state_dict"])
    p3.eval()

    t1, t1_info = SF.build_model("T", norm, 1, device)
    t1_ckpt = SS / "zinc_shared_factor_v1/runs/SF_T_s1/best.pt"
    t1_state = torch.load(t1_ckpt, map_location="cpu", weights_only=False)
    t1.load_state_dict(t1_state["state_dict"])
    t1.eval()
    return {
        "p3": p3,
        "t1": t1,
        "p3_ckpt": p3_ckpt,
        "t1_ckpt": t1_ckpt,
        "p3_epoch": int(p3_state.get("epoch", -1)),
        "t1_epoch": int(t1_state.get("epoch", -1)),
        "p3_valid_recorded": float(p3_state.get("valid_mae", np.nan)),
        "t1_valid_recorded": float(t1_state.get("valid_mae", np.nan)),
        "p3_parameters": int(p3_info["parameters"]),
        "t1_parameters": int(t1_info["parameters"]),
    }


def predict_p3(model, cache: Mapping[str, Any], rows: np.ndarray,
               device, batch_size: int = 256) -> np.ndarray:
    import torch
    import ad_train as T1

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.asarray(rows[start:start + int(batch_size)], dtype=np.int64)
            batch = T1._to_device(T1.collate(cache, chunk, "P3N", valid=True), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)


def predict_t1(model, cache: Mapping[str, Any], bf_cache: Mapping[str, Any],
               rows: np.ndarray, device, split: str,
               batch_size: int = 256) -> np.ndarray:
    import torch
    import ad_train as T1
    import bf_data as BFD
    import lf_models as LM

    table = LM.build_pattern_table(bf_cache, split=split)
    model.set_table(table.to(device))
    model.eval()
    BFD.register_cache(split, bf_cache)
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.asarray(rows[start:start + int(batch_size)], dtype=np.int64)
            batch = T1._to_device(BFD.collate_bf(cache, chunk), device)
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)


def build_bf_cache(objs: Sequence[Any], split: str) -> dict:
    import bf_data as BFD

    return BFD.build_from_objects(objs, split, {"source": "kekule_equiv_v1"}, verbose=False)


# ---------------------------------------------------------------------------
# artifact I/O
# ---------------------------------------------------------------------------


def save_reps_artifact(path: Path, records: Sequence[dict], meta: dict) -> None:
    """Save variable-length verified Kekule representations as padded arrays."""
    n = len(records)
    edge_u = np.full((n, MAX_EDGES), -1, dtype=np.int16)
    edge_v = np.full((n, MAX_EDGES), -1, dtype=np.int16)
    orig_order = np.zeros((n, MAX_EDGES), dtype=np.int8)
    aromatic_mask = np.zeros((n, MAX_EDGES), dtype=np.uint8)
    rep_orders = np.zeros((n, MAX_REPS, MAX_EDGES), dtype=np.int8)
    n_atoms = np.zeros(n, dtype=np.int16)
    n_edges = np.zeros(n, dtype=np.int16)
    n_reps = np.zeros(n, dtype=np.int8)
    status = np.zeros(n, dtype=np.int8)
    truncated = np.zeros(n, dtype=np.uint8)
    n_total = np.full(n, -1, dtype=np.int16)
    n_inferred_h = np.zeros(n, dtype=np.int16)
    graph_sha = np.empty(n, dtype="U64")
    status_code = {"exhaustive": 0, "no_alternative": 1, "truncated": 2,
                   "unsupported": 3}
    for i, r in enumerate(records):
        ne = len(r["edges"])
        nr = len(r["reps"])
        n_atoms[i] = int(r["n_atoms"])
        n_edges[i] = ne
        n_reps[i] = nr
        status[i] = status_code[r["status"]]
        truncated[i] = int(bool(r["truncated"]))
        n_total[i] = int(r["n_total"])
        n_inferred_h[i] = int(r.get("n_inferred_h", 0))
        graph_sha[i] = r["graph_sha256"]
        for k, (u, v) in enumerate(r["edges"]):
            edge_u[i, k] = int(u)
            edge_v[i, k] = int(v)
        orig_order[i, :ne] = np.asarray(r["orig_orders"], dtype=np.int8)
        aromatic_mask[i, :ne] = np.asarray(
            [1 if tuple(e) in r["aromatic_edge_set"] else 0 for e in r["edges"]],
            dtype=np.uint8)
        for j, o in enumerate(r["reps"]):
            rep_orders[i, j, :ne] = np.asarray(o, dtype=np.int8)
    np.savez_compressed(
        path,
        edge_u=edge_u, edge_v=edge_v, orig_order=orig_order,
        aromatic_mask=aromatic_mask, rep_orders=rep_orders,
        n_atoms=n_atoms, n_edges=n_edges, n_reps=n_reps,
        status=status, truncated=truncated, n_total=n_total,
        n_inferred_h=n_inferred_h, graph_sha256=graph_sha,
        meta_json=np.asarray([__import__("json").dumps(meta)], dtype="U"),
    )


def load_reps_artifact(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def graph_from_arrays(cats, edges, edge_index, orders):
    """Directed PyG-style ``edge_attr`` for one undirected order vector."""
    return apply_orders(edge_index, np.zeros(edge_index.shape[1], dtype=np.int64),
                        edges, orders)
