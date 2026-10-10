"""Chemical-graph alignment check: canonical SMILES vs PyG ZINC graph.

cscl-correctness-v1 (task 3.3).  The cscl-v0 check ``crosscheck_atom_counts``
compared only *approximate atom counts* parsed from the SMILES string; an
exact atom-count match does not establish that the SMILES and the PyG graph
describe the same chemical structure.  This module performs a real chemical
graph reconstruction and attribute correspondence check:

1. A minimal, strict SMILES parser builds the attributed molecular graph:
   atoms = (element, explicit H count, formal charge, aromatic flag), bonds =
   (order 1/2/3, or ``AROMATIC`` for bonds written by the SMILES aromatic
   model: between adjacent lowercase aromatic atoms without an explicit bond
   symbol).  Stereo ``@/@@`` is intentionally ignored (recorded limitation —
   the PyG graphs carry no stereochemistry either).
2. Aromatic bonds are kekulized: per-atom ring-π capacity follows the
   standard valence model ``capacity = default_valence(element) + charge −
   σ_bonds − explicit_H − exocyclic_π``; a maximum matching over capacity-≥1
   aromatic atoms assigns double bonds (matched aromatic bond → order 2,
   unmatched → 1).  If no perfect matching exists the molecule is flagged
   (tier-1 skipped, tier-2/3 still run).
3. The PyG graph is typed through the committed dataset dictionaries
   (``data/ZINC/raw/atom_dict.pickle`` / ``bond_dict.pickle``): ZINC atom
   types encode element + *explicit* H count + charge only (implicit H are
   not represented; aromatic CH / amide NH are plain ``C``/``N`` — verified
   empirically on ``data/ZINC``), bond types 1/2/3 = SINGLE/DOUBLE/TRIPLE
   (kekulized adjacency).
4. Both graphs are compared with VF2 attributed-graph isomorphism
   (``networkx.is_isomorphic``), in tiers:
   * tier 1 — exact bond orders + full atom typing;
   * tier 2 — aromatic-relaxed: the SMILES graph keeps ``AROMATIC`` bond
     marks (kekulization skipped) and an aromatic-side bond matches any
     order; captures kekulé-choice ambiguity, not chemical difference;
   * tier 3 — element-only topology (weakest).

   Atom-typing normalization (dataset-side information loss, documented):
   the committed ZINC atom dictionary contains no uncharged H-bearing
   N/O/S words (verified against ``atom_dict.idx2word``: only
   ``N H1 +``/``N H2 +``/``N H3 +``/``N H1 -``/``O H1 +``/``S H1 +``
   exist), so aromatic ``[nH]``-style atoms are stored as plain ``N``
   on the PyG side.  Uncharged explicit H on N/O/S is therefore blanked
   on the SMILES side before comparison; charged H (e.g. ``[nH+]``) and
   carbon stereocentre H (``C H1``) are compared exactly.

Interpretation discipline: attributed isomorphism is strong structural
evidence; failures are reported per tier and never silently dropped.
Signature/hash equality elsewhere in the codebase is *not* treated as
isomorphism proof.

Limitations (explicit): no RDKit in this environment (``rdkit`` absent from
``pyproject.toml``/``uv.lock`` — checked before writing this); the parser
targets the RDKit-canonical SMILES dialect of the committed ZINC table and
raises on constructs it does not understand instead of guessing; stereo and
isotopes are ignored on both sides; the valence model covers the elements of
the 28-entry ZINC atom dictionary only.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import networkx as nx

AROMATIC = 4  # internal bond-order code for SMILES-aromatic (not yet kekulized) bonds

_ZINC_ROOT = Path(__file__).resolve().parents[3] / "data/ZINC"

#: default valences for elements appearing in the 28-entry ZINC atom dictionary
DEFAULT_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}

_ORGANIC = {"B", "C", "N", "O", "P", "S", "F", "I", "Cl", "Br"}
_ORGANIC_AROMATIC = {"b", "c", "n", "o", "p", "s"}
_BOND_SYMBOL = {"-": 1, "=": 2, "#": 3, ":": AROMATIC, "/": 1, "\\": 1}


class SmilesParseError(ValueError):
    """Raised (never swallowed) when the SMILES uses unsupported syntax."""


@dataclass
class SmilesMolecule:
    atoms: list[tuple[str, int, int, bool]] = field(default_factory=list)  # (element, explicit H, charge, aromatic)
    bonds: list[tuple[int, int, int]] = field(default_factory=list)  # (u, v, order) with AROMATIC possible
    kekulization_failed: bool = False


# ---------------------------------------------------------------------------
# SMILES parsing
# ---------------------------------------------------------------------------


def _parse_bracket(s: str, i: int) -> tuple[str, int, int, bool, int]:
    """Parse ``[...]`` starting at ``s[i] == '['``; return (element, H, charge, aromatic, next_i)."""
    j = s.index("]", i)
    body = s[i + 1 : j]
    k = 0
    while k < len(body) and body[k].isdigit():  # optional isotope (ignored)
        k += 1
    if k >= len(body) or not body[k].isalpha():
        raise SmilesParseError(f"unsupported bracket atom [{body}]")
    el = body[k]
    k += 1
    if k < len(body) and (el + body[k]) in {"Cl", "Br", "Si", "Se", "As", "Te"}:
        el += body[k]
        k += 1
    aromatic = el[0].islower()
    el = el.capitalize() if aromatic else el
    if el not in DEFAULT_VALENCE:
        raise SmilesParseError(f"element {el!r} outside ZINC atom dictionary")
    h = 0
    if k < len(body) and body[k] == "@":  # chirality: @, @@, or @TH1/@AL1/... (ignored)
        k += 1
        if k < len(body) and body[k] == "@":
            k += 1
        else:
            for pref in ("TH", "AL", "SP", "TB", "OH"):
                if body.startswith(pref, k):
                    k += len(pref)
                    while k < len(body) and body[k].isdigit():
                        k += 1
                    break
    if k < len(body) and body[k] == "H":
        k += 1
        while k < len(body) and body[k].isdigit():
            h = h * 10 + int(body[k])
            k += 1
        h = max(h, 1)
    charge = 0
    while k < len(body):
        c = body[k]
        if c in "+-":
            sign = 1 if c == "+" else -1
            k += 1
            num, got = 0, False
            while k < len(body) and body[k].isdigit():
                num = num * 10 + int(body[k])
                k += 1
                got = True
            charge += sign * (num if got else 1)
        elif c == ":":  # atom class digits: ignored
            break
        else:
            raise SmilesParseError(f"unsupported bracket token {c!r} in [{body}]")
    return el, h, charge, aromatic, j + 1


def _default_order(mol: SmilesMolecule, u: int, v: int) -> int:
    if mol.atoms[u][3] and mol.atoms[v][3]:
        return AROMATIC
    return 1


def parse_smiles(smi: str) -> SmilesMolecule:
    """Strict parser for the RDKit-canonical dialect used by the ZINC table."""
    mol = SmilesMolecule()
    cur: int | None = None
    stack: list[int] = []
    pending_bond: int | None = None
    open_rings: dict[int, tuple[int, int | None]] = {}
    i, n = 0, len(smi)
    while i < n:
        c = smi[i]
        if c == "(":
            if cur is None:
                raise SmilesParseError("branch before first atom")
            stack.append(cur)
            i += 1
        elif c == ")":
            if not stack:
                raise SmilesParseError("unbalanced ')'")
            cur = stack.pop()
            pending_bond = None
            i += 1
        elif c == ".":
            cur = None
            pending_bond = None
            i += 1
        elif c in _BOND_SYMBOL:
            if pending_bond is not None:
                raise SmilesParseError(f"consecutive bond symbols at {i}")
            pending_bond = _BOND_SYMBOL[c]
            i += 1
        elif c == "[":
            el, h, q, ar, i = _parse_bracket(smi, i)
            idx = len(mol.atoms)
            mol.atoms.append((el, h, q, ar))
            if cur is not None:
                mol.bonds.append((cur, idx, pending_bond if pending_bond is not None else _default_order(mol, cur, idx)))
            cur, pending_bond = idx, None
        elif c.isalpha():
            two = smi[i : i + 2]
            if two in _ORGANIC:
                el, ar, i = two, False, i + 2
            elif c in _ORGANIC:
                el, ar, i = c, False, i + 1
            elif c in _ORGANIC_AROMATIC:
                el, ar, i = c.capitalize(), True, i + 1
            else:
                raise SmilesParseError(f"unsupported atom token {smi[i:i+2]!r} at {i}")
            idx = len(mol.atoms)
            mol.atoms.append((el, 0, 0, ar))
            if cur is not None:
                mol.bonds.append((cur, idx, pending_bond if pending_bond is not None else _default_order(mol, cur, idx)))
            cur, pending_bond = idx, None
        elif c.isdigit() or c == "%":
            raise SmilesParseError(f"ring closure without preceding atom at {i}")
        else:
            raise SmilesParseError(f"unsupported character {c!r} at {i}")
        # ring-closure tokens immediately after the current atom
        while i < n and (smi[i].isdigit() or smi[i] == "%" or smi[i] in _BOND_SYMBOL):
            bond: int | None = None
            if smi[i] in _BOND_SYMBOL:
                bond = _BOND_SYMBOL[smi[i]]
                i += 1
            if i < n and (smi[i].isdigit() or smi[i] == "%"):
                if smi[i] == "%":
                    if i + 2 >= n:
                        raise SmilesParseError("truncated %nn ring number")
                    num = int(smi[i + 1 : i + 3])
                    i += 3
                else:
                    num = int(smi[i])
                    i += 1
                if num in open_rings:
                    prev, prev_bond = open_rings.pop(num)
                    order = prev_bond if prev_bond is not None else bond
                    mol.bonds.append((prev, cur, order if order is not None else _default_order(mol, prev, cur)))
                else:
                    open_rings[num] = (cur, bond)
            elif bond is not None:
                pending_bond = bond  # symbol not followed by a digit: applies to next chain bond
                break
            else:
                break
    if open_rings:
        raise SmilesParseError(f"unclosed ring bonds: {sorted(open_rings)}")
    return mol


# ---------------------------------------------------------------------------
# kekulization (aromatic bonds -> alternating 1/2 via perfect matching)
# ---------------------------------------------------------------------------


def _needs_ring_double(mol: SmilesMolecule, v: int, sigma_arom: int, sigma_total: int, exo_pi: int) -> bool:
    """SMILES aromatic-model rule: does aromatic atom ``v`` take one ring double?

    * exocyclic π already written (e.g. ``O=c1...`` pyridone-type) → no;
    * plain/bracket ``c`` → yes (valence 4; fused junction c takes one too);
    * ``n`` plain, exactly 2 aromatic σ and no exocyclic σ (pyridine-type)
n      → yes; ``n`` with a substituent or 3 aromatic σ (pyrrole-type) → no;
    * ``[nH]`` (pyrrole N–H) → no; ``[nH+]`` / bracket ``n+`` without
      exocyclic bond (pyridinium-type) → yes; aromatic ``[n+]`` with an
      exocyclic bond (N-oxide write-out) → no (documented pragmatic rule);
    * aromatic ``o`` / ``s`` / others → no (furan/thiophene type).
    """
    if exo_pi > 0:
        return False
    el, h, q, _ar = mol.atoms[v]
    if el == "C":
        return True
    if el == "N":
        if h >= 1:
            return q > 0
        if sigma_arom == 2 and sigma_total == 2:
            return True
        return False
    return False


def kekulize(mol: SmilesMolecule) -> SmilesMolecule:
    """Replace AROMATIC orders by 1/2; sets ``kekulization_failed`` on failure."""
    arom_bonds = [(idx, u, v) for idx, (u, v, o) in enumerate(mol.bonds) if o == AROMATIC]
    if not arom_bonds:
        return mol
    sigma_arom = [0] * len(mol.atoms)
    sigma_total = [0] * len(mol.atoms)
    exo_pi = [0] * len(mol.atoms)
    for u, v, o in mol.bonds:
        sigma_total[u] += 1
        sigma_total[v] += 1
        if o == AROMATIC:
            sigma_arom[u] += 1
            sigma_arom[v] += 1
        else:
            exo_pi[u] += o - 1
            exo_pi[v] += o - 1
    needs = [
        v
        for v in range(len(mol.atoms))
        if mol.atoms[v][3] and _needs_ring_double(mol, v, sigma_arom[v], sigma_total[v], exo_pi[v])
    ]
    need_set = set(needs)
    mgraph = nx.Graph()
    for idx, u, v in arom_bonds:
        if u in need_set and v in need_set:
            mgraph.add_edge(u, v)
    matching = nx.max_weight_matching(mgraph, maxcardinality=True)
    covered = {node for edge in matching for node in edge}
    if covered != need_set:
        mol.kekulization_failed = True
        return mol
    doubled = {tuple(sorted(e)) for e in matching}
    for idx, u, v in arom_bonds:
        mol.bonds[idx] = (u, v, 2 if (u, v) in doubled or (v, u) in doubled else 1)
    return mol


# ---------------------------------------------------------------------------
# PyG-side graph (typed via committed dataset dictionaries)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _zinc_dicts() -> tuple[list[str], list[str]]:
    import __main__

    class Dictionary(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)

    __main__.Dictionary = Dictionary
    with open(_ZINC_ROOT / "raw/atom_dict.pickle", "rb") as f:
        atom_dict = pickle.load(f)
    with open(_ZINC_ROOT / "raw/bond_dict.pickle", "rb") as f:
        bond_dict = pickle.load(f)
    return list(atom_dict.idx2word), list(bond_dict.idx2word)


def parse_atom_word(word: str) -> tuple[str, int, int]:
    """``'C H1'/'N H1 +'/'O -'`` -> (element, explicit H count, charge)."""
    tokens = word.split()
    el = tokens[0]
    el = el.capitalize() if el[0].islower() else el
    h, charge = 0, 0
    for tok in tokens[1:]:
        if tok.startswith("H"):
            h = int(tok[1:]) if len(tok) > 1 else 1
        elif tok.endswith("+"):
            charge += int(tok[:-1]) if len(tok) > 1 else 1
        elif tok.endswith("-"):
            charge -= int(tok[:-1]) if len(tok) > 1 else 1
        else:
            raise ValueError(f"unsupported atom dictionary word {word!r}")
    return el, h, charge


def pyg_graph(x, edge_index, edge_attr) -> nx.Graph:
    """Attributed networkx graph of one PyG ZINC molecule (bonds deduped)."""
    idx2atom, _idx2bond = _zinc_dicts()
    g = nx.Graph()
    for i, t in enumerate(x.reshape(-1).tolist()):
        g.add_node(i, t=parse_atom_word(idx2atom[int(t)]))
    seen: dict[tuple[int, int], int] = {}
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == v:
            raise ValueError("self-loop in PyG graph")
        key = (u, v) if u < v else (v, u)
        bt = int(edge_attr.reshape(-1)[k])
        prev = seen.setdefault(key, bt)
        if prev != bt:
            raise ValueError(f"inconsistent bond type across directions for {key}")
    for (u, v), bt in sorted(seen.items()):
        if bt not in (1, 2, 3):
            raise ValueError(f"unsupported bond type {bt}")
        g.add_edge(u, v, o=bt)
    return g


# ---------------------------------------------------------------------------
# tiered attributed-isomorphism comparison
# ---------------------------------------------------------------------------


def smiles_graph(mol: SmilesMolecule) -> nx.Graph:
    g = nx.Graph()
    for i, (el, h, q, _ar) in enumerate(mol.atoms):
        g.add_node(i, t=(el, h, q))
    for u, v, o in mol.bonds:
        g.add_edge(u, v, o=o)
    return g


def _edge_match_exact(a: dict, b: dict) -> bool:
    return a["o"] == b["o"]


def _edge_match_arom_relaxed(a: dict, b: dict) -> bool:
    return a["o"] == b["o"] or a["o"] == AROMATIC


def _normalize_dataset_atom_typing(mol: SmilesMolecule) -> SmilesMolecule:
    """Blank uncharged explicit H on N/O/S (dataset cannot represent them).

    The committed ZINC atom dictionary has no uncharged ``N H1``/``O H1``/
    ``S H1`` words; the dataset conversion stores aromatic ``[nH]``-type
    atoms as plain ``N``/``O``/``S``.  Charged H (``[nH+]``) and carbon H
    (``[C@H]`` -> ``C H1``) are represented and stay exact.
    """
    atoms = []
    for el, h, q, ar in mol.atoms:
        if el in {"N", "O", "S"} and q == 0 and h > 0:
            h = 0
        atoms.append((el, h, q, ar))
    mol.atoms = atoms
    return mol


def check_alignment(smiles: str, x, edge_index, edge_attr) -> dict:
    """Full comparison for one molecule; returns per-tier results.

    Never raises on chemical mismatch; raises only on unsupported SMILES
    syntax (:class:`SmilesParseError`) or data-level inconsistencies.
    """
    mol = _normalize_dataset_atom_typing(parse_smiles(smiles))
    g_pyg = pyg_graph(x, edge_index, edge_attr)
    node_match = lambda a, b: a["t"] == b["t"]  # noqa: E731
    out: dict = {
        "n_atoms_smiles": len(mol.atoms),
        "n_atoms_pyg": g_pyg.number_of_nodes(),
        "n_bonds_smiles": len(mol.bonds),
        "n_bonds_pyg": g_pyg.number_of_edges(),
    }
    mol_kek = kekulize(parse_smiles(smiles))
    mol_kek = _normalize_dataset_atom_typing(mol_kek)
    out["kekulization_failed"] = mol_kek.kekulization_failed
    if mol_kek.kekulization_failed:
        out["tier1_isomorphic"] = None
    else:
        g_exact = smiles_graph(mol_kek)
        out["tier1_isomorphic"] = bool(
            nx.is_isomorphic(g_exact, g_pyg, node_match=node_match, edge_match=_edge_match_exact)
        )
    if not out.get("tier1_isomorphic"):
        g_arom = smiles_graph(mol)  # AROMATIC marks preserved for the relaxed tier
        out["tier2_isomorphic"] = bool(
            nx.is_isomorphic(g_arom, g_pyg, node_match=node_match, edge_match=_edge_match_arom_relaxed)
        )
    else:
        out["tier2_isomorphic"] = True
    if not out.get("tier1_isomorphic") and not out.get("tier2_isomorphic"):
        g_smiles_el = nx.Graph()
        g_smiles_el.add_nodes_from((i, {"t": a[0]}) for i, a in enumerate(mol.atoms))
        g_smiles_el.add_edges_from((u, v) for u, v, _ in mol.bonds)
        g_pyg_el = nx.Graph()
        g_pyg_el.add_nodes_from((i, {"t": d["t"][0]}) for i, d in g_pyg.nodes(data=True))
        g_pyg_el.add_edges_from(g_pyg.edges())
        out["tier3_isomorphic"] = bool(
            nx.is_isomorphic(g_smiles_el, g_pyg_el, node_match=lambda a, b: a["t"] == b["t"])
        )
    else:
        out["tier3_isomorphic"] = None
    return out
