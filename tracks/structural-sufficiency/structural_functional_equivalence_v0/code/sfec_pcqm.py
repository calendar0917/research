"""PCQM4Mv2 graph decomposition with RDKit exact rooted-fragment identity.

A cut fragment keeps one side plus the cut neighbour, which is turned into an
explicit dummy atom in its original bond slot.  The canonical isomeric SMILES of
that rooted fragment is the exact key.  The construction preserves formal
charge, aromaticity and stereo; the cut-spanning double-bond stereo is carried
by the dummy bond's direction in the key.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def find_bridges(n_atoms: int, bond_u: np.ndarray, bond_v: np.ndarray) -> list[int]:
    """Indices of bridges in a simple undirected graph (iterative Tarjan)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for k, (a, b) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        adj[a].append((b, k))
        adj[b].append((a, k))
    tin = np.full(n_atoms, -1, dtype=np.int64)
    low = np.zeros(n_atoms, dtype=np.int64)
    bridges: list[int] = []
    timer = 0
    for root in range(n_atoms):
        if tin[root] != -1:
            continue
        tin[root] = low[root] = timer
        timer += 1
        stack: list[list[int]] = [[root, -1, 0]]
        while stack:
            node, parent_edge, pos = stack[-1]
            if pos < len(adj[node]):
                nxt, edge_id = adj[node][pos]
                stack[-1][2] += 1
                if edge_id == parent_edge:
                    continue
                if tin[nxt] == -1:
                    tin[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append([nxt, edge_id, 0])
                else:
                    low[node] = min(low[node], tin[nxt])
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > tin[parent]:
                        bridges.append(int(parent_edge))
    return sorted(set(bridges))


def component_labels(
    n_atoms: int, bond_u: np.ndarray, bond_v: np.ndarray, skip_edge: int | None = None
) -> np.ndarray:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, (a, b) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        if skip_edge is not None and k == skip_edge:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n_atoms)]
    remap = {root: i for i, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[r] for r in roots], dtype=np.int32)


def bond_code(bond: Chem.Bond) -> int:
    t = bond.GetBondType()
    if t == Chem.BondType.SINGLE:
        return 1
    if t == Chem.BondType.DOUBLE:
        return 2
    if t == Chem.BondType.TRIPLE:
        return 3
    if t == Chem.BondType.AROMATIC:
        return 4
    return 5


def canon(mol: Chem.Mol, stereo: bool = True) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=bool(stereo), canonical=True)


def _fragment_mol(
    mol: Chem.Mol, side: np.ndarray, cut_nbr: int
) -> tuple[Chem.Mol, int, int]:
    """Keep ``side`` + the cut neighbour; turn the neighbour into a dummy."""
    keep = set(int(a) for a in side)
    keep.add(int(cut_nbr))
    ordered = sorted(keep)
    newidx = {old: i for i, old in enumerate(ordered)}
    rw = Chem.RWMol(mol)
    for old in range(rw.GetNumAtoms() - 1, -1, -1):
        if old not in keep:
            rw.RemoveAtom(old)
    d = newidx[int(cut_nbr)]
    atom = rw.GetAtomWithIdx(d)
    atom.SetAtomicNum(0)
    atom.SetFormalCharge(0)
    atom.SetIsAromatic(False)
    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetNumRadicalElectrons(0)
    atom.SetIsotope(0)
    atom.SetAtomMapNum(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    # the root is the unique neighbour of the dummy
    nbrs = m.GetAtomWithIdx(d).GetNeighbors()
    if len(nbrs) != 1:
        raise RuntimeError(f"dummy has {len(nbrs)} neighbours")
    return m, int(nbrs[0].GetIdx()), int(d)


def _rooted_key(mol: Chem.Mol, side: np.ndarray, cut_nbr: int) -> tuple[str, Chem.Mol, int]:
    m, root, dummy = _fragment_mol(mol, side, cut_nbr)
    return canon(m), m, root


def q_descriptor(mol: Chem.Mol, root: int, cut_bond: int) -> str:
    """Coarse PCQM interface descriptor: root plus one-hop inside C."""
    atom = mol.GetAtomWithIdx(int(root))
    root_desc = (int(atom.GetAtomicNum()), int(atom.GetFormalCharge()), int(atom.GetIsAromatic()))
    nbrs = []
    for bond in atom.GetBonds():
        if bond.GetIdx() == int(cut_bond):
            continue
        other = mol.GetAtomWithIdx(int(bond.GetOtherAtomIdx(int(root))))
        nbrs.append(
            (
                int(other.GetAtomicNum()),
                int(other.GetFormalCharge()),
                int(other.GetIsAromatic()),
                bond_code(bond),
            )
        )
    nbrs.sort()
    return (
        f"p{root_desc[0]},{root_desc[1]},{root_desc[2]}|"
        + ";".join(f"{z},{c},{a},{b}" for z, c, a, b in nbrs)
    )


def _scaffold(m: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        text = canon(scaf, stereo=False)
        return text if text else ""
    except Exception:
        return ""


def decompose(
    smiles: str,
    *,
    max_small_side: int = 8,
    min_context: int = 6,
    min_ratio: float = 2.0,
    do_reassembly: bool = False,
) -> dict[str, Any] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    stats = {
        "n_atoms": int(mol.GetNumAtoms()),
        "n_bonds": int(mol.GetNumBonds()),
        "n_components": int(len(Chem.GetMolFrags(mol))),
        "n_bridges": 0,
        "n_single_bridges": 0,
        "n_cuts": 0,
        "multi_component": False,
        "parse_failed": False,
    }
    if stats["n_components"] != 1:
        stats["multi_component"] = True
        return {"stats": stats, "cuts": [], "mol_key": None}
    n = int(mol.GetNumAtoms())
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    bu = np.asarray([a for a, _ in bonds], dtype=np.int64)
    bv = np.asarray([b for _, b in bonds], dtype=np.int64)
    bridges = find_bridges(n, bu, bv)
    stats["n_bridges"] = len(bridges)
    mol_key = canon(mol, stereo=True)
    mol_scaffold = _scaffold(mol)
    cuts = []
    for eid in bridges:
        bond = mol.GetBondWithIdx(eid)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        stats["n_single_bridges"] += 1
        a, b = int(bu[eid]), int(bv[eid])
        lab = component_labels(n, bu, bv, skip_edge=eid)
        side_a = np.flatnonzero(lab == lab[a])
        side_b = np.flatnonzero(lab == lab[b])
        if len(side_a) > len(side_b):
            side_a, side_b = side_b, side_a
            a, b = b, a
        if len(side_a) == len(side_b):
            ka, _, _ = _rooted_key(mol, side_a, b)
            kb, _, _ = _rooted_key(mol, side_b, a)
            if kb < ka:
                side_a, side_b = side_b, side_a
                a, b = b, a
        nS, nC = len(side_a), len(side_b)
        if not (1 <= nS <= max_small_side and nC >= min_context and nC >= min_ratio * nS):
            continue
        s_key, ms, _ = _rooted_key(mol, side_a, b)
        c_key, mc, _ = _rooted_key(mol, side_b, a)
        rec = {
            "cut_id": int(eid),
            "bond_type": int(bond_code(bond)),
            "S_key": s_key,
            "C_key": c_key,
            "nS": int(nS),
            "nC": int(nC),
            "q": q_descriptor(mol, b, eid),
            "C_scaffold": _scaffold(mc),
            "reassembly_graph_ok": None,
            "reassembly_stereo_ok": None,
        }
        if do_reassembly:
            graph_ok, stereo_ok = reassemble_check(mol, ms, mc, bond)
            rec["reassembly_graph_ok"] = bool(graph_ok)
            rec["reassembly_stereo_ok"] = bool(stereo_ok)
        cuts.append(rec)
    stats["n_cuts"] = len(cuts)
    return {
        "stats": stats,
        "cuts": cuts,
        "mol_key": mol_key,
        "mol_scaffold": mol_scaffold,
    }


def reassemble_check(
    original: Chem.Mol, ms: Chem.Mol, mc: Chem.Mol, cut_bond: Chem.Bond
) -> tuple[bool, bool]:
    """Reassemble the two rooted fragments and compare with the original.

    Returns (graph_exact, stereo_exact).  The graph check is independent of
    stereo; the stereo check compares canonical isomeric SMILES.  Stereo at the
    cut root can be represented differently when the dummy bond is replaced by
    the other root, so the two outcomes are reported separately.
    """
    comb = Chem.RWMol(Chem.CombineMols(ms, mc))
    dummies = [a.GetIdx() for a in comb.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return False, False
    roots = [comb.GetAtomWithIdx(d).GetNeighbors()[0].GetIdx() for d in dummies]
    for d in sorted(dummies, reverse=True):
        comb.RemoveAtom(d)
    roots = [r - sum(1 for d in dummies if d < r) for r in roots]
    if roots[0] == roots[1]:
        return False, False
    nb = comb.AddBond(roots[0], roots[1], cut_bond.GetBondType())
    new_bond = comb.GetBondWithIdx(nb - 1)
    new_bond.SetBondDir(cut_bond.GetBondDir())
    m2 = comb.GetMol()
    try:
        Chem.SanitizeMol(m2)
    except Exception:
        return False, False
    graph_ok = canon(m2, stereo=False) == canon(original, stereo=False)
    stereo_ok = canon(m2, stereo=True) == canon(original, stereo=True)
    return bool(graph_ok), bool(stereo_ok)