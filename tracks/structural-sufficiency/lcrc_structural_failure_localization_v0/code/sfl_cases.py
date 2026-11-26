"""Phase A: locate the real failure cases on the official Valid split.

The three case sets are frozen from the *frozen* paired predictions of the
reference round; no label or prediction is re-derived here.  For every case the
union is reported once (``in_gain5`` / ``in_loss5`` / ``in_common5`` flags), and
each case carries its 2-D structure figure with canonical atom numbering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import sfl_common as C
import sfl_data as D


def case_rows(base: D.Base) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    valid = base.errors[base.errors.split == "valid"].copy()
    sets = C.freeze_case_sets(valid)
    order = [r for name in ("GAIN5", "LOSS5", "COMMON5") for r in sets[name]]
    seen: list[int] = []
    for r in order:
        if r not in seen:
            seen.append(r)

    mol = base.mol.set_index("row_id")
    err = base.errors.set_index("row_id")
    pats = C.frozen_patterns()
    factory, _fpath = C.basefeature_factory()
    n_train = int((base.mol.split == "train").sum())

    records = []
    for row in seen:
        rec = mol.loc[row]
        e = err.loc[row]
        desc = C.descriptors(rec["smiles"])
        hits = C.candidate_hits(rec["smiles"], pats)
        bf = C.basefeature_hits(rec["smiles"], factory)
        families = C.acid_families(hits)
        in_sets = [n for n in ("GAIN5", "LOSS5", "COMMON5") if row in sets[n]]
        view_a, view_b = encoder_field_view(rec["smiles"], hits)
        records.append(
            {
                "row_id": int(row),
                "in_sets": "+".join(in_sets),
                "in_gain5": int("GAIN5" in in_sets),
                "in_loss5": int("LOSS5" in in_sets),
                "in_common5": int("COMMON5" in in_sets),
                "smiles": rec["smiles"],
                "canonical_isomeric_smiles": rec["canon_smiles"],
                "y_true": float(rec["exp"]),
                "pred_A": float(e["pred_A"]),
                "pred_B": float(e["pred_B"]),
                "err_A": float(e["error_A"]),
                "err_B": float(e["error_B"]),
                "err_A_sq": float(e["e_A2"]),
                "err_B_sq": float(e["e_B2"]),
                "gain_A_minus_B": float(e["gain"]),
                **desc,
                "candidate_hits": ";".join(f"{k}@{v}" for k, v in sorted(hits.items())) or "none",
                "candidate_families": "+".join(families) or "none",
                "n_candidate_families": len(families),
                "basefeature_NegIonizable": bf["NegIonizable"],
                "basefeature_PosIonizable": bf["PosIonizable"],
                "scaffold": C.scaffold_key(rec["smiles"]),
                "A_retained_fields_for_candidate": view_a,
                "B_retained_fields_for_candidate": view_b,
            }
        )
    return pd.DataFrame(records), sets


#: what each encoder actually consumes, quoted from the frozen specifications
ENCODER_INPUTS: dict[str, dict[str, Any]] = {
    "A_FIXED1": {
        "atom_fields": [
            "atom_type one-hot(16: element x aromatic, P and S_aromatic present)",
            "formal_charge one-hot(6: -2,-1,0,+1,+2,UNK)",
            "n_hydrogens one-hot(6: 0,1,2,3,4,UNK)",
            "degree one-hot(6)", "in_ring", "min_ring_size one-hot(7)",
            "ring_bond_count_over_4", "ring_bond_fraction",
            "incident_bond_class_fraction(5)", "is_terminal", "is_fused_or_spiro",
        ],
        "atom_fields_absent": [
            "chirality (no stereo descriptor on atoms)",
            "hybridization", "num_radical_electrons",
            "explicit H count above 4 (bucket UNK)",
        ],
        "bond_fields": ["bond_class(5)", "is_ring", "exactly_one_endpoint_in_ring",
                        "both_endpoints_in_ring", "stereo(5)"],
        "local_structure": "S1/S2 rooted 2-path content: log1p_n_paths2, closure_ratio, "
                           "log1p_n_d2 + ring/bond-class fractions per atom",
        "receptive_field": "2 hops (S2 enumeration) + 2 fixed-source message stages",
        "global_fields": "33 graph-topology scalars (shared with B via post_local)",
        "solution_context": "none (no pH, no pKa, no solvent, no experimental context)",
    },
    "B_GINE": {
        "atom_fields": [
            "atomic_num index", "chirality", "degree", "formal_charge", "num_hs",
            "num_radical_electrons", "hybridization", "is_aromatic", "is_in_ring",
        ],
        "atom_fields_absent": [
            "element x aromatic class beyond the raw atomic_num index",
            "min-ring size as an atom label (ring index is only a separate embedding)",
        ],
        "bond_fields": ["bond_type", "bond_stereo", "bond_is_in_ring"],
        "local_structure": "4 x GINEConv message passing (4 hops), edge-conditioned",
        "receptive_field": "4 hops",
        "global_fields": "same 33 graph-topology scalars via the shared post_local",
        "solution_context": "none (no pH, no pKa, no solvent, no experimental context)",
    },
}


def _draw_mol(ax, mol, highlights: dict[int, tuple[float, float, float]], title: str) -> None:
    """Matplotlib-only 2-D depiction (RDKit 2-D coordinates, no X11/cairo deps)."""
    import matplotlib.pyplot as plt
    from rdkit.Chem import rdDepictor

    rdDepictor.Compute2DCoords(mol)
    m = mol
    conf = m.GetConformer()
    xy = {a.GetIdx(): (conf.GetAtomPosition(a.GetIdx()).x, conf.GetAtomPosition(a.GetIdx()).y)
          for a in m.GetAtoms()}
    ax.set_aspect("equal")
    ax.axis("off")
    for i, col in highlights.items():
        if i in xy:
            ax.add_patch(plt.Circle(xy[i], 0.34, color=col, alpha=0.28, zorder=0))
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        (x1, y1), (x2, y2) = xy[i], xy[j]
        order = b.GetBondTypeAsDouble()
        if b.GetIsAromatic():
            order = 1.0
        ax.plot([x1, x2], [y1, y2], color="0.15", lw=1.6, solid_capstyle="round", zorder=2)
        if order >= 2:
            dx, dy = x2 - x1, y2 - y1
            n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ox, oy = -dy / n * 0.12, dx / n * 0.12
            ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy], color="0.15", lw=1.2, zorder=2)
    for a in m.GetAtoms():
        i = a.GetIdx()
        x, y = xy[i]
        sym = a.GetSymbol()
        if sym != "C":
            ax.text(x, y, sym, ha="center", va="center", fontsize=9, weight="bold",
                    color="0.05", zorder=3,
                    bbox=dict(boxstyle="circle,pad=0.14", fc="white", ec="none", alpha=0.85))
        ax.text(x, y + 0.30, str(i), ha="center", va="bottom", fontsize=5.5,
                color="0.35", zorder=4)
    ax.set_title(title, fontsize=7.5)
    ax.relim()
    ax.autoscale_view()



def encoder_field_view(smiles: str, hits: Mapping[str, Sequence[int]]) -> tuple[str, str]:
    """What each encoder actually receives for this molecule's candidate atoms.

    A value is only reported if the field exists in the frozen specification
    (see ``ENCODER_INPUTS``); nothing is inferred from a hidden vector.
    """
    from rdkit import Chem

    m = C.mol_from_smiles(smiles)
    idxs = sorted({int(i) for v in hits.values() for i in v})
    if not idxs:
        note_a = "no frozen candidate atom; A receives the same 11 per-atom field groups for every atom"
        note_b = "no frozen candidate atom; B receives the same 9+1 per-atom columns for every atom"
        return note_a, note_b
    a_parts, b_parts = [], []
    for i in idxs:
        at = m.GetAtomWithIdx(i)
        el = at.GetSymbol()
        arom = int(at.GetIsAromatic())
        q = int(at.GetFormalCharge())
        nh = int(at.GetTotalNumHs())
        deg = int(at.GetDegree())
        bonds = {}
        for b in at.GetBonds():
            k = str(b.GetBondType()).split(".")[-1]
            bonds[k] = bonds.get(k, 0) + 1
        bl = "{" + ",".join(f"{k}:{v}" for k, v in sorted(bonds.items())) + "}"
        a_parts.append(
            f"{i}:{el}{'+arom' if arom else ''};q={q};nH={nh};deg={deg};bondclass={bl}"
        )
        b_parts.append(
            f"{i}:Z={at.GetAtomicNum()};chir={str(at.GetChiralTag()).split('.')[-1]};deg={deg};"
            f"q={q};nHs={nh};hyb={str(at.GetHybridization()).split('.')[-1]};"
            f"arom={arom};in_ring={int(at.IsInRing())}"
        )
    a_note = ("A per-atom fields used above: atom_type(element x aromatic), formal_charge, "
              "n_hydrogens, degree, incident_bond_class_fraction, in_ring "
              "(+ S1/S2 two-hop path content, receptive field 2 hops) || "
              + " | ".join(a_parts))
    b_note = ("B per-atom fields used above: atomic_num, chirality, degree, formal_charge, "
              "num_hs, hybridization, is_aromatic, is_in_ring (4 x GINEConv, receptive field 4 hops) || "
              + " | ".join(b_parts))
    return a_note, b_note


def draw_cases(cases: pd.DataFrame, out_dir: Path = C.FIG_DIR) -> list[dict[str, Any]]:
    """One PNG per case: 2-D structure, canonical atom indices, candidate atoms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pats = C.frozen_patterns()
    colors = {"PHOSPHO": (0.85, 0.33, 0.10), "CARBOXYL": (0.13, 0.47, 0.71), "SULFONIC": (0.47, 0.19, 0.62)}
    written = []
    for _, r in cases.iterrows():
        m = C.mol_from_smiles(r["smiles"])
        hit_atoms: dict[int, tuple[float, float, float]] = {}
        for fam, keys in C.ACID_FAMILIES:
            for k in keys:
                for match in m.GetSubstructMatches(pats[k]):
                    for i in match:
                        hit_atoms[int(i)] = colors[fam]
        fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=170)
        _draw_mol(
            ax,
            m,
            hit_atoms,
            f"row {int(r['row_id'])}  [{r['in_sets']}]   y={r['y_true']:.2f}   "
            f"A={r['pred_A']:.2f} (e={r['err_A']:+.2f})   B={r['pred_B']:.2f} (e={r['err_B']:+.2f})\n"
            f"families: {r['candidate_families']}   MolLogP={r['mol_logp']:.2f}",
        )
        path = out_dir / f"case_{int(r['row_id'])}.png"
        fig.tight_layout()
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        written.append(
            {
                "row_id": int(r["row_id"]),
                "figure": C.rel(path),
                "highlighted_atoms": sorted(hit_atoms),
                "families": r["candidate_families"],
                "note": "atom indices are canonical-SMILES atom indices; "
                        "highlight = frozen ionization candidate, not a measured pKa",
            }
        )
    return written
