"""Three paired cases: original graph next to the T and K rewired graphs.

Local only (RDKit + matplotlib).  Each figure has three panels sharing one 2D
coordinate set: the measured original molecule, the ``T`` rewire (amide pattern
destroyed) and the ``K`` rewire (amide pattern kept), for the *same* donor bond.

The drawn bonds always come from the proven raw graph of that view (never from a
re-parsed SMILES), and the SMILES -> raw mapping is re-verified before drawing.
Removed bonds are drawn dashed and faint, added bonds thick orange, the target
amide bonds red and the kept control bond blue.

The rewired panels are **not** compounds with measured properties: the label
printed at the top is the original molecule's experimental logD and is used only
as the computational reference of ``D``/``B``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import graphlib as G
import rewire as R

ELEMENT_OF_CLASS = {i: name for i, (name, _arom) in enumerate(C.RD.ATOM_CLASSES)}
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
_BOND_NAMES = {0: "single", 1: "double", 2: "triple", 3: "aromatic", 4: "unknown"}


def _verify_mapping(smiles: str, row_id: int, raw_row: Mapping[str, Any]) -> dict[str, Any]:
    derived = C.RD.smiles_to_raw_row(str(smiles), int(row_id))
    checks = {
        "n_atoms": int(len(derived["atom_class"])) == int(len(raw_row["atom_class"])),
        "atom_class": np.array_equal(
            np.asarray(derived["atom_class"]), np.asarray(raw_row["atom_class"], dtype=np.int64)
        ),
        "atom_charge": np.array_equal(
            np.asarray(derived["atom_charge"]), np.asarray(raw_row["atom_charge"], dtype=np.int64)
        ),
        "atom_h": np.array_equal(
            np.asarray(derived["atom_h"]), np.asarray(raw_row["atom_h"], dtype=np.int64)
        ),
        "bond_pairs": {tuple(sorted(p)) for p in np.asarray(derived["edge_index"]).T.tolist()}
        == {tuple(sorted(p)) for p in np.asarray(raw_row["edge_index"]).T.tolist()},
        "bond_classes": sorted(np.asarray(derived["edge_class"]).tolist())
        == sorted(np.asarray(raw_row["edge_class"]).tolist()),
    }
    return {"all_equal": all(checks.values()), "fields": checks}


def _bond_segments(p0, p1, style: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
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


def _draw_panel(ax, *, title: str, row: Mapping[str, Any], pos, roles, remove, add, target_bonds, control_bonds):
    bu, bv, bc, _bs = G.physical_bonds(row)
    ac = np.asarray(row["atom_class"], dtype=np.int64)
    existing = {(int(u), int(v)): int(c) for u, v, c in zip(bu.tolist(), bv.tolist(), bc.tolist())}
    remove_set = {tuple(sorted((int(a), int(b)))) for a, b in remove}
    add_set = {tuple(sorted((int(a), int(b)))) for a, b in add}
    for pair, color in [(p, "#d02020") for p in target_bonds] + [(p, "#2040d0") for p in control_bonds]:
        if tuple(sorted(pair)) in existing:
            p0, p1 = pos[pair[0]], pos[pair[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=13, alpha=0.38,
                    solid_capstyle="round", zorder=1)
    for (u, v), klass in existing.items():
        style = _BOND_NAMES.get(int(klass), "unknown")
        for a, b in _bond_segments(pos[u], pos[v], style):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", lw=1.8, zorder=2, solid_capstyle="round")
    for a, b in add_set:
        p0, p1 = pos[a], pos[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#ff8c00", lw=4.0, zorder=3,
                solid_capstyle="round", alpha=0.95)
    target_set = {tuple(sorted(p)) for p in target_bonds}
    for a, b in remove_set:
        p0, p1 = pos[a], pos[b]
        is_target = tuple(sorted((a, b))) in target_set
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="#d02020" if is_target else "#8a8a8a",
            lw=2.4 if is_target else 1.8,
            ls=(0, (3, 2.2)),
            zorder=3,
            alpha=0.95,
        )
    for idx, (x, y) in pos.items():
        element = ELEMENT_OF_CLASS.get(int(ac[idx]), "?")
        color = COLORS.get(element, "#444444")
        ax.scatter([x], [y], s=150, color="white", edgecolor=color, linewidths=1.6, zorder=4)
        ax.text(x, y, element, ha="center", va="center", fontsize=9, color=color, fontweight="bold", zorder=5)
        ax.text(x + 0.10, y + 0.10, f"{idx}", ha="left", va="bottom", fontsize=6.5, color="#666666", zorder=5)
        if idx in roles:
            ax.text(x - 0.10, y - 0.14, roles[idx], ha="right", va="top", fontsize=8.5,
                    color="#b02020" if roles[idx] in ("c", "N", "O") else "#2030a0",
                    fontweight="bold", zorder=6)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5)


def draw_case(case: Mapping[str, Any], raw_row: Mapping[str, Any], out_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = str(raw_row["smiles"])
    mapping = _verify_mapping(smiles, int(case["row_id"]), raw_row)
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
    n_atoms = int(np.asarray(raw_row["atom_class"]).shape[0])
    if len(pos) != n_atoms:
        raise SystemExit(f"atom count mismatch row {case['row_id']}: {len(pos)} != {n_atoms}")

    # pick the first donor of the first instance that has FULL-scope numbers
    inst = case["instances"][0]
    donor = None
    for d in inst["donors"]:
        if d["scope"] == C.PRIMARY_SCOPE:
            donor = d
            break
    if donor is None:
        raise SystemExit(f"row {case['row_id']}: no primary-scope donor to draw")
    roles = {
        int(inst["c"]): "c",
        int(inst["o"]): "O",
        int(inst["n"]): "N",
        int(inst["acyl_c"]): "acylC",
        int(donor["donor_u"]): "u",
        int(donor["donor_v"]): "v",
    }
    t_remove = [(int(a), int(b)) for a, b in _parse(donor["t_remove"])]
    t_add = [(int(a), int(b)) for a, b in _parse(donor["t_add"])]
    k_remove = [(int(a), int(b)) for a, b in _parse(donor["k_remove"])]
    k_add = [(int(a), int(b)) for a, b in _parse(donor["k_add"])]
    base = {**dict(raw_row), "mol_index": int(case["row_id"])}
    t_row, _ = R.edit_row(base, t_remove, t_add)
    k_row, _ = R.edit_row(base, k_remove, k_add)
    target_bonds = [tuple(sorted((int(inst["c"]), int(inst["n"]))))]
    control_bonds = [tuple(sorted((int(inst["c"]), int(inst["acyl_c"]))))]

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.2), dpi=150)
    _draw_panel(
        axes[0],
        title="O — original measured molecule",
        row=base,
        pos=pos,
        roles=roles,
        remove=[],
        add=[],
        target_bonds=target_bonds,
        control_bonds=control_bonds,
    )
    _draw_panel(
        axes[1],
        title="T — amide C–N destroyed (rewired graph, no experimental label)",
        row=t_row,
        pos=pos,
        roles=roles,
        remove=t_remove,
        add=t_add,
        target_bonds=target_bonds,
        control_bonds=control_bonds,
    )
    _draw_panel(
        axes[2],
        title="K — amide C–N kept, same donor rewired (no experimental label)",
        row=k_row,
        pos=pos,
        roles=roles,
        remove=k_remove,
        add=k_add,
        target_bonds=target_bonds,
        control_bonds=control_bonds,
    )
    text = [
        f"row {case['row_id']}   y(original) = {case['y']:.3f} logD   "
        f"f(O) = {case['pred_O']:+.3f}   ({case['tag']}, D rank {case['D_rank']}/{case['n_molecules']}, "
        f"ranking scope FULL)",
        f"instance {inst['instance_id']}   donor {donor['pair_id'].split(':')[-1]} (u={donor['donor_u']}, "
        f"v={donor['donor_v']})   T removes {donor['t_remove']} adds {donor['t_add']}",
        f"K removes {donor['k_remove']} adds {donor['k_add']}   "
        f"solid black = bonds of that graph, dashed grey = removed, orange = added",
    ]
    for scope in (C.PRIMARY_SCOPE, "G_FIXED"):
        s = case["scopes"][scope]
        text.append(
            f"{scope:<8} L0={s['loss_O']:.3f} LT={s['loss_T']:.3f} LK={s['loss_K']:.3f}   "
            f"move_T={s['move_T']:.3f} move_K={s['move_K']:.3f}   J={s['J']:+.3f} D={s['D']:+.3f} B={s['B']:+.3f}"
        )
    text.append(
        "y above is the ORIGINAL molecule's measured logD and is used only as the computational reference for "
        "D/B; the T/K graphs were never synthesised, have no measured label and are not drawn as compounds."
    )
    fig.text(0.01, 0.015, "\n".join(text), fontsize=8.2, va="bottom", ha="left", family="monospace")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "tag": case["tag"],
        "row_id": int(case["row_id"]),
        "file": C.track_rel(out_path),
        "smiles": smiles,
        "mapping": mapping,
        "instance_id": inst["instance_id"],
        "donor": {k: donor[k] for k in ("pair_id", "donor_u", "donor_v", "t_remove", "t_add", "k_remove", "k_add")},
        "y_original": case["y"],
        "pred_O": case["pred_O"],
        "scopes": case["scopes"],
        "target_bonds": target_bonds,
        "control_bonds": control_bonds,
    }


def _parse(code: str) -> list[list[int]]:
    out = []
    for chunk in str(code).split("."):
        if not chunk:
            continue
        a, b = chunk.split("-")
        out.append([int(a), int(b)])
    return out


def main() -> int:
    cases_path = C.ANALYSIS / "cases.json"
    if not cases_path.exists():
        raise FileNotFoundError("cases.json missing; run analysis.py first")
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    drawn: list[dict[str, Any]] = []
    for case in payload["cases"]:
        out_path = C.CASES / f"case_{case['tag']}_row{case['row_id']}.png"
        drawn.append(draw_case(case, raw_rows[int(case["row_id"])], out_path))
        C.log(f"drew {drawn[-1]['file']} ({drawn[-1]['tag']})")
    C.write_json(
        C.CASES / "cases_manifest.json",
        {
            "main_population": payload["main_population"],
            "note": payload["note"],
            "cases": drawn,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
