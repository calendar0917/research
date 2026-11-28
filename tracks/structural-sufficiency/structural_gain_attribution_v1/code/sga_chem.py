"""Phase 0.1 — corrected *display* decoding of OGB categorical attribute indices.

The predecessor round (`MSDS-v0`) rendered structural registries and case
descriptions with ``msds_enum.atom_label`` / ``bond_label``, which interpreted
raw OGB **category indices** as if they were physical atomic numbers / charges /
bond orders.  Example: category index 5 of ``possible_atomic_num_list`` is the
element with atomic number 6 (carbon), not boron.

This module decodes the indices with OGB's own ``allowable_features`` table and is
the only place where chemistry names are produced in SGA-v1.  Nothing here is
ever read by a model: the canonical keys, the counts, the vocabulary and the
design matrices stay byte-identical (see ``run_p0_chem_errata.py``).

Facts pinned by ``run_p0_chem_errata.py``:

* every model input path in MSDS-v0 is built from raw integer indices and
  canonical tuples only; ``describe_key``/``atom_label``/``bond_label`` are used
  exclusively by registry CSV / case JSON writers;
* the per-atom chemical properties decoded here agree with RDKit for real
  molecules (atomic number, formal charge, aromaticity, degree, total H,
  hybridization, chirality presence, ring membership, bond order/stereo).
"""

from __future__ import annotations

from typing import Any, Sequence

from ogb.utils.features import allowable_features

ATOM_FIELDS = (
    "atomic_num",
    "chirality",
    "degree",
    "formal_charge",
    "num_h",
    "num_radical_e",
    "hybridization",
    "aromatic",
    "in_ring",
)
BOND_FIELDS = ("bond_type", "stereo", "conjugated")

_ATOM_TABLE = {
    "atomic_num": allowable_features["possible_atomic_num_list"],
    "chirality": allowable_features["possible_chirality_list"],
    "degree": allowable_features["possible_degree_list"],
    "formal_charge": allowable_features["possible_formal_charge_list"],
    "num_h": allowable_features["possible_numH_list"],
    "num_radical_e": allowable_features["possible_number_radical_e_list"],
    "hybridization": allowable_features["possible_hybridization_list"],
    "aromatic": allowable_features["possible_is_aromatic_list"],
    "in_ring": allowable_features["possible_is_in_ring_list"],
}
_BOND_TABLE = {
    "bond_type": allowable_features["possible_bond_type_list"],
    "stereo": allowable_features["possible_bond_stereo_list"],
    "conjugated": allowable_features["possible_is_conjugated_list"],
}

#: short one-letter-ish codes for readable structural formulas
_HYB_CODE = {"SP": "s", "SP2": "sp2", "SP3": "sp3", "SP3D": "sp3d", "SP3D2": "sp3d2", "misc": "h?"}
_CHI_CODE = {
    "CHI_UNSPECIFIED": "",
    "CHI_TETRAHEDRAL_CW": "R?",
    "CHI_TETRAHEDRAL_CCW": "S?",
    "CHI_OTHER": "chi?",
    "misc": "chi?",
}


def _lookup(table: Sequence[Any], index: int, field: str) -> Any:
    idx = int(index)
    if idx < 0 or idx >= len(table):
        return f"OUT_OF_RANGE[{field}={idx}]"
    return table[idx]


# ---------------------------------------------------------------------------
# raw-index -> chemistry
# ---------------------------------------------------------------------------


#: the raw OGB atom columns the COARSE granularity keeps (``msds_common``)
COARSE_ATOM_FIELDS = ("atomic_num", "formal_charge", "aromatic")


def decode_atom(values: Sequence[int], granularity: str = "FULL") -> dict[str, Any]:
    """Decode one OGB atom row (9 FULL categorical indices, or 3 for COARSE)."""
    out: dict[str, Any] = {}
    if granularity == "COARSE":
        if len(values) != len(COARSE_ATOM_FIELDS):
            raise ValueError(f"COARSE atom rows have {len(COARSE_ATOM_FIELDS)} columns, got {len(values)}")
        for field, raw in zip(COARSE_ATOM_FIELDS, values):
            out[field] = _lookup(_ATOM_TABLE[field], raw, field)
        for field in ATOM_FIELDS:
            out.setdefault(field, None)
    else:
        for field, raw in zip(ATOM_FIELDS, values):
            out[field] = _lookup(_ATOM_TABLE[field], raw, field)
    z = out["atomic_num"]
    out["symbol"] = _symbol(z) if isinstance(z, int) else str(z)
    out["is_misc"] = any(v == "misc" or (isinstance(v, str) and v.startswith("OUT_OF_RANGE"))
                         for v in out.values())
    return out


def decode_bond(values: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, raw in zip(BOND_FIELDS, values):
        out[field] = _lookup(_BOND_TABLE[field], raw, field)
    out["is_misc"] = any(v == "misc" or (isinstance(v, str) and v.startswith("OUT_OF_RANGE"))
                         for v in out.values())
    return out


def _symbol(z: Any) -> str:
    from rdkit import Chem

    if isinstance(z, int) and 0 < z <= 118:
        return Chem.GetPeriodicTable().GetElementSymbol(z)
    return f"Z{z}"


# ---------------------------------------------------------------------------
# compact labels used in readable structural formulas
# ---------------------------------------------------------------------------


def atom_label(values: Sequence[int], granularity: str) -> str:
    """Compact, chemistry-correct label of one atom.

    ``COARSE`` keeps what the COARSE granularity actually reads (atomic number,
    formal charge, aromatic flag); ``FULL`` keeps every decoded field.
    """
    d = decode_atom(values, granularity)
    sym = d["symbol"]
    if granularity == "COARSE":
        text = sym + ("ar" if d["aromatic"] else "")
        if d["formal_charge"] not in (0, "misc"):
            text += f"{int(d['formal_charge']):+d}"
        return text
    parts = [sym]
    if d["aromatic"]:
        parts.append("ar")
    if d["in_ring"]:
        parts.append("R")
    parts.append(_HYB_CODE.get(str(d["hybridization"]), str(d["hybridization"])))
    if d["num_h"] not in (0, "misc"):
        parts.append(f"H{int(d['num_h'])}")
    if d["formal_charge"] not in (0, "misc"):
        parts.append(f"{int(d['formal_charge']):+d}")
    chi = _CHI_CODE.get(str(d["chirality"]), str(d["chirality"]))
    if chi:
        parts.append(chi)
    if d["num_radical_e"] not in (0, "misc"):
        parts.append(f"rad{int(d['num_radical_e'])}")
    parts.append(f"deg{d['degree']}")
    return ".".join(parts)


def bond_label(values: Sequence[int], granularity: str) -> str:
    d = decode_bond(values)
    bt = str(d["bond_type"])
    if granularity == "COARSE":
        return bt
    text = bt
    if d["stereo"] not in ("STEREONONE", "misc"):
        text += f"/{d['stereo']}"
    if d["conjugated"]:
        text += "/cj"
    return text


def describe_key(key: tuple, granularity: str) -> dict[str, Any]:
    """Readable class-level summary of a canonical key (display only)."""
    node_part, edge_part = key
    labels = [atom_label(n, granularity) for n in node_part]
    class_of: dict[str, str] = {}
    for lab in labels:
        if lab not in class_of:
            class_of[lab] = chr(ord("a") + len(class_of))
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    node_sig = " ".join(f"{class_of[lab]}:{lab}x{n}" for lab, n in sorted(counts.items()))
    bond_counts: dict[tuple[str, str, str], int] = {}
    for i, j, battr in edge_part:
        a, b = class_of[labels[i]], class_of[labels[j]]
        bl = bond_label(battr, granularity)
        key2 = (min(a, b), max(a, b), bl)
        bond_counts[key2] = bond_counts.get(key2, 0) + 1
    edge_sig = " ".join(f"{a}-{b}:{bl}x{n}" for (a, b, bl), n in sorted(bond_counts.items()))
    return {
        "n_nodes": len(labels),
        "n_bonds": len(edge_part),
        "node_signature": node_sig,
        "bond_signature": edge_sig,
        "atom_labels": labels,
        "atom_details": [decode_atom(n, granularity) for n in node_part],
        "bond_labels": [bond_label(e[2], granularity) for e in edge_part],
        "bond_details": [decode_bond(e[2]) for e in edge_part],
    }


# ---------------------------------------------------------------------------
# the predecessor's (wrong) display mapping, kept only for the ERRATA
# ---------------------------------------------------------------------------

_WRONG_ELEMENTS_BY_INDEX = {  # msds_enum._ELEMENTS read as if the index were Z
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
}
_WRONG_HYBRID = {0: "?", 1: "s", 2: "sp", 3: "sp2", 4: "sp3"}
_WRONG_BOND = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROM"}


def predecessor_coarse_atom_label(values: Sequence[int]) -> str:
    """Reproduce MSDS-v0 ``atom_label`` (COARSE) verbatim, for the ERRATA table."""
    z, charge, aromatic = (int(a) for a in values)
    sym = _WRONG_ELEMENTS_BY_INDEX.get(z, f"Z{z}")
    text = sym + ("ar" if aromatic else "")
    if charge:
        text += f"{charge:+d}"
    return text
