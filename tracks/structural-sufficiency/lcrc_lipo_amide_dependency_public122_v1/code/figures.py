"""Three real original molecules: 2D structure with the cut bonds marked.

Local-only (RDKit + matplotlib).  RDKit's cairo drawing backend is not available
on this machine (missing ``libXrender``), so the depiction is drawn directly with
matplotlib from RDKit 2D coordinates: atom positions come from
``AllChem.Compute2DCoords`` and every bond/atom label is taken from the proven
raw graph (never from a re-parsed, re-ordered molecule).

Before drawing, the artifact's ``smiles`` is re-parsed and the derived raw
attributes must equal the artifact row byte for byte; otherwise the script stops
(``SMILES_GRAPH_MAPPING_FAILED``) instead of drawing the wrong bond.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import common as C

COLORS = {
    "C": "#222222",
    "N": "#1f4fd8",
    "O": "#d62728",
    "S": "#b8860b",
    "F": "#2ca02c",
    "Cl": "#2ca02c",
    "Br": "#8c564b",
    "I": "#7b1fa2",
    "P": "#e377c2",
    "Si": "#8c8c8c",
    "B": "#17becf",
    "H": "#777777",
}
ELEMENT_OF_CLASS = {i: name for i, (name, _arom) in enumerate(C.RD.ATOM_CLASSES)}


def verify_smiles_mapping(smiles: str, row_id: int, raw_row: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the raw attributes from SMILES and require exact equality."""
    derived = C.RD.smiles_to_raw_row(str(smiles), int(row_id))
    checks = {
        "n_atoms": int(len(derived["atom_class"])) == int(len(raw_row["atom_class"])),
        "atom_class": np.array_equal(np.asarray(derived["atom_class"]), np.asarray(raw_row["atom_class"])),
        "atom_charge": np.array_equal(np.asarray(derived["atom_charge"]), np.asarray(raw_row["atom_charge"])),
        "atom_h": np.array_equal(np.asarray(derived["atom_h"]), np.asarray(raw_row["atom_h"])),
        "bond_class_multiset": sorted(np.asarray(derived["edge_class"]).tolist())
        == sorted(np.asarray(raw_row["edge_class"]).tolist()),
        "bond_pairs": {tuple(sorted(p)) for p in np.asarray(derived["edge_index"]).T.tolist()}
        == {tuple(sorted(p)) for p in np.asarray(raw_row["edge_index"]).T.tolist()},
        "canonical_smiles": str(derived["canonical_smiles"]) == str(raw_row["canonical_smiles"]),
        "n_components": int(derived["n_components"]) == int(raw_row["n_components"]),
    }
    return {"all_equal": all(checks.values()), "fields": checks}


def _bond_segments(p0, p1, style: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """One, two or three parallel segments for single/double/triple/aromatic."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy)) or 1.0
    nx, ny = -dy / length, dx / length
    if style in ("single", "aromatic", "unknown"):
        return [((x0, y0), (x1, y1))]
    if style == "double":
        off = 0.075
        return [
            ((x0 + nx * off, y0 + ny * off), (x1 + nx * off, y1 + ny * off)),
            ((x0 - nx * off, y0 - ny * off), (x1 - nx * off, y1 - ny * off)),
        ]
    off = 0.085
    return [
        ((x0, y0), (x1, y1)),
        ((x0 + nx * off, y0 + ny * off), (x1 + nx * off, y1 + ny * off)),
        ((x0 - nx * off, y0 - ny * off), (x1 - nx * off, y1 - ny * off)),
    ]


_BOND_NAMES = {0: "single", 1: "double", 2: "triple", 3: "aromatic", 4: "unknown"}


def draw_case(case: Mapping[str, Any], raw_row: Mapping[str, Any], out_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = str(raw_row["smiles"])
    mapping = verify_smiles_mapping(smiles, int(case["row_id"]), raw_row)
    if not mapping["all_equal"]:
        raise SystemExit(f"SMILES_GRAPH_MAPPING_FAILED for row {case['row_id']}: {mapping['fields']}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"row {case['row_id']}: RDKit cannot parse {smiles!r}")
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {
        int(a.GetIdx()): (
            float(conf.GetAtomPosition(int(a.GetIdx())).x),
            float(conf.GetAtomPosition(int(a.GetIdx())).y),
        )
        for a in mol.GetAtoms()
    }
    atom_class = np.asarray(raw_row["atom_class"], dtype=np.int64)
    ei = np.asarray(raw_row["edge_index"], dtype=np.int64)
    ec = np.asarray(raw_row["edge_class"], dtype=np.int64)
    seen: dict[tuple[int, int], int] = {}
    for k in range(ei.shape[1]):
        key = (min(int(ei[0, k]), int(ei[1, k])), max(int(ei[0, k]), int(ei[1, k])))
        seen.setdefault(key, int(ec[k]))

    targets, controls, roles = [], [], {}
    for inst in case["instances"]:
        c, o, n, acyl = int(inst["c"]), int(inst["o"]), int(inst["n"]), int(inst["acyl_c"])
        targets.append((min(c, n), max(c, n)))
        controls.append((min(c, acyl), max(c, acyl)))
        roles[c] = "c"
        roles[o] = "O"
        roles[n] = "N"
        roles[acyl] = "acylC"

    fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=160)
    # highlighted bonds first (thick underlay)
    for pair, color in [(t, "#e02020") for t in targets] + [(k, "#2050e0") for k in controls]:
        p0, p1 = pos[pair[0]], pos[pair[1]]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=11, alpha=0.28, solid_capstyle="round", zorder=1
        )
    for (u, v), klass in seen.items():
        style = _BOND_NAMES.get(int(klass), "unknown")
        for a, b in _bond_segments(pos[u], pos[v], style):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", lw=1.9, zorder=2, solid_capstyle="round")
        if style == "aromatic":
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color="#333333", lw=1.2, ls=(0, (2.5, 2.5)), zorder=2, alpha=0.85)
    for idx, (x, y) in pos.items():
        element = ELEMENT_OF_CLASS.get(int(atom_class[idx]), "?")
        color = COLORS.get(element, "#444444")
        ax.scatter([x], [y], s=190, color="white", edgecolor=color, linewidths=1.8, zorder=3)
        label = element
        ax.text(x, y, label, ha="center", va="center", fontsize=11, color=color, fontweight="bold", zorder=4)
        ax.text(x + 0.10, y + 0.10, f"{idx}", ha="left", va="bottom", fontsize=7.2, color="#666666", zorder=4)
        if idx in roles:
            ax.text(
                x - 0.10,
                y - 0.14,
                roles[idx],
                ha="right",
                va="top",
                fontsize=9,
                color="#b02020" if roles[idx] in ("c", "N", "O") else "#2030a0",
                fontweight="bold",
                zorder=5,
            )

    lines = [
        f"row {case['row_id']}   y = {case['y']:.3f} logD   (Test, MATCHED, FULL)",
        f"f(O) = {case['instances'][0]['pred_O']:.3f}    f(T: cut c-N) = {case['instances'][0]['pred_T']:.3f}    "
        f"f(K: cut c-acylC) = {case['instances'][0]['pred_K']:.3f}",
        f"B = LT - L0 = {case['B']:.3f}    D = LT - LK = {case['D']:.3f}    "
        f"({case['tag']}, D rank {case['D_rank']}/{case['n_molecules']})",
        "red = target amide C-N bond (T),  blue = matched control C-C bond (K)",
    ]
    ax.text(
        0.01,
        -0.045,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9.2,
        va="top",
        ha="left",
        family="monospace",
    )
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 1.2
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad - 0.6, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "row_id": int(case["row_id"]),
        "tag": str(case["tag"]),
        "file": C.track_rel(out_path),
        "smiles": smiles,
        "mapping": mapping,
        "target_bonds": targets,
        "control_bonds": controls,
        "instances": case["instances"],
        "y": case["y"],
        "B": case["B"],
        "D": case["D"],
        "bond_styles": {f"{u}-{v}": _BOND_NAMES.get(int(k), "unknown") for (u, v), k in seen.items()},
    }


def main() -> int:
    cases_path = C.ANALYSIS / "cases.json"
    if not cases_path.exists():
        raise FileNotFoundError("cases.json missing; run analysis.py first")
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    drawn = []
    for case in payload["cases"]:
        out_path = C.CASES / f"case_{case['tag']}_row{case['row_id']}.png"
        drawn.append(draw_case(case, raw_rows[int(case["row_id"])], out_path))
        C.log(f"drew {drawn[-1]['file']} ({drawn[-1]['tag']})")
    C.write_json(
        C.CASES / "cases_manifest.json", {"main_population": payload["main_population"], "cases": drawn}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
