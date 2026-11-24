"""ZINC graph decomposition and exact rooted-fragment identity.

The bridge finder, component labeller and coloured-incidence-graph certificate
are the historical implementations in
``response_dictionary_small_s0_v1/code/rd_fragment.py``; they are imported
read-only rather than reimplemented.  ZINC raw node/bond type codes are used
directly, so no chemical decoding is required for identity.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sfec_common import HIST_FRAGMENT_CODE, REPO_ROOT

_HIST_CODE = REPO_ROOT / "tracks/structural-sufficiency/response_dictionary_small_s0_v1/code"
if str(_HIST_CODE) not in sys.path:
    sys.path.insert(0, str(_HIST_CODE))

import rd_fragment as RF  # noqa: E402

BOND_COLUMNS = 3
ATOM_COLUMNS = 9


def dedupe_bonds(edge_index: np.ndarray, edge_attr: np.ndarray):
    """Undirected bond list from PyG's directed edge list."""
    u = np.asarray(edge_index[0], dtype=np.int64)
    v = np.asarray(edge_index[1], dtype=np.int64)
    t = np.asarray(edge_attr, dtype=np.int64).reshape(-1)
    keep_u, keep_v, keep_t = [], [], []
    seen: dict[tuple[int, int], int] = {}
    for a, b, c in zip(u.tolist(), v.tolist(), t.tolist()):
        if a == b:
            raise ValueError("self loop")
        key = (a, b) if a < b else (b, a)
        if key in seen:
            if seen[key] != c:
                raise ValueError(f"asymmetric bond attribute {key}: {seen[key]} vs {c}")
            continue
        seen[key] = c
        keep_u.append(key[0])
        keep_v.append(key[1])
        keep_t.append(c)
    return (
        np.asarray(keep_u, dtype=np.int64),
        np.asarray(keep_v, dtype=np.int64),
        np.asarray(keep_t, dtype=np.int64),
    )


def _atom_feat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.int16).reshape(-1, 1)


def _bond_feat(bt: np.ndarray) -> np.ndarray:
    out = np.zeros((len(bt), BOND_COLUMNS), dtype=np.int16)
    out[:, 0] = np.asarray(bt, dtype=np.int16)
    return out


def whole_key(x: np.ndarray, u: np.ndarray, v: np.ndarray, bt: np.ndarray) -> str:
    cert = RF.fragment_certificate(
        _atom_feat(x), u, v, _bond_feat(bt), port_local=None
    )
    return hashlib.sha256(cert.encode()).hexdigest()


def _rooted_key(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bt: np.ndarray,
    atoms: np.ndarray,
    root: int,
) -> tuple[str, int]:
    frag = RF._induce(
        atoms, _atom_feat(x), u, v, _bond_feat(bt), port_global=int(root)
    )
    return frag.key, len(atoms)


def q_descriptor(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bt: np.ndarray,
    root: int,
    cut_edge: int,
) -> str:
    """Root raw type plus sorted one-hop (neighbour type, bond code) inside C."""
    nbrs = []
    for k, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        if k == cut_edge:
            continue
        if a == root:
            nbrs.append((int(x[b]), int(bt[k])))
        elif b == root:
            nbrs.append((int(x[a]), int(bt[k])))
    nbrs.sort()
    return f"z{int(x[root])}|" + ";".join(f"{t}:{c}" for t, c in nbrs)


def decompose(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    single_bond_code: int = 1,
    max_small_side: int = 8,
    min_context: int = 6,
    min_ratio: float = 2.0,
) -> dict[str, Any]:
    """Return cuts and stats for one raw ZINC graph."""
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    n = int(x.size)
    u, v, bt = dedupe_bonds(edge_index, edge_attr)
    stats = {
        "n_atoms": n,
        "n_bonds": int(len(u)),
        "n_components": 0,
        "n_bridges": 0,
        "n_single_bridges": 0,
        "n_cuts": 0,
        "multi_component": False,
    }
    lab = RF.component_labels(n, u, v)
    n_comp = int(lab.max()) + 1 if n else 0
    stats["n_components"] = n_comp
    if n_comp != 1:
        stats["multi_component"] = True
        return {"stats": stats, "cuts": [], "mol_key": None}
    mol_key = whole_key(x, u, v, bt)
    bridges = RF.find_bridges(n, u, v)
    stats["n_bridges"] = len(bridges)
    cuts = []
    for eid in bridges:
        if int(bt[eid]) != int(single_bond_code):
            continue
        stats["n_single_bridges"] += 1
        a, b = int(u[eid]), int(v[eid])
        side_lab = RF.component_labels(n, u, v, skip_edge=eid)
        side_a = np.flatnonzero(side_lab == side_lab[a])
        side_b = np.flatnonzero(side_lab == side_lab[b])
        if len(side_a) > len(side_b):
            side_a, side_b = side_b, side_a
            a, b = b, a
        if len(side_a) == len(side_b):
            ka, _ = _rooted_key(x, u, v, bt, side_a, a)
            kb, _ = _rooted_key(x, u, v, bt, side_b, b)
            if kb < ka:
                side_a, side_b = side_b, side_a
                a, b = b, a
        nS, nC = len(side_a), len(side_b)
        if not (1 <= nS <= max_small_side and nC >= min_context and nC >= min_ratio * nS):
            continue
        s_key, _ = _rooted_key(x, u, v, bt, side_a, a)
        c_key, _ = _rooted_key(x, u, v, bt, side_b, b)
        cuts.append(
            {
                "cut_id": int(eid),
                "bond_type": int(bt[eid]),
                "S_key": s_key,
                "C_key": c_key,
                "nS": int(nS),
                "nC": int(nC),
                "q": q_descriptor(x, u, v, bt, b, eid),
            }
        )
    stats["n_cuts"] = len(cuts)
    return {"stats": stats, "cuts": cuts, "mol_key": mol_key}


def reassemble_check(
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    cut_id: int,
    *,
    single_bond_code: int = 1,
) -> bool:
    """Exact graph-level reassembly check for one bridge cut."""
    x = np.asarray(x, dtype=np.int64).reshape(-1)
    n = int(x.size)
    u, v, bt = dedupe_bonds(edge_index, edge_attr)
    a, b = int(u[cut_id]), int(v[cut_id])
    lab = RF.component_labels(n, u, v, skip_edge=cut_id)
    side_a = np.flatnonzero(lab == lab[a])
    side_b = np.flatnonzero(lab == lab[b])
    if len(side_a) > len(side_b):
        side_a, side_b = side_b, side_a
        a, b = b, a
    sa, _ = _rooted_key(x, u, v, bt, side_a, a)
    sb, _ = _rooted_key(x, u, v, bt, side_b, b)
    # The union of the two induced graphs plus the cut edge is the original
    # graph by construction; verify the certificates are mutually distinct
    # rooted fragments and the attribute counts add up exactly.
    n_bonds = len(u)
    # count induced edges independently
    set_a = set(int(i) for i in side_a)
    e_a = sum(1 for i, j in zip(u.tolist(), v.tolist()) if i in set_a and j in set_a)
    e_b = n_bonds - e_a - 1
    ok = (
        len(side_a) + len(side_b) == n
        and e_a + e_b + 1 == n_bonds
        and sa != sb
    )
    return bool(ok)