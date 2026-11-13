"""Stage C figures: the six-cell applicability/effect matrix, the TRAINED vs INIT
calibrated-preference comparison, the mechanism (Mobius) matrix, and one median
case plus one counterexample per non-empty family.

All numbers come from the frozen analysis artefacts (``analysis/*.json`` /
``*.csv``); nothing is recomputed here and no prediction is re-run.  Case figures
are the only place that touches RDKit: they draw the *proven* raw graph of the O /
T / K0 views of one measured molecule (never a re-parsed SMILES), and the SMILES ->
raw mapping is re-verified before drawing.

The rewired T / K0 panels are **not** compounds with measured properties: the
label printed is the original molecule's experimental logD and is used only as the
computational reference of ``D`` / ``B``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C

W = 0.75  # bar width
COL_TRAINED = "#1f4fd8"
COL_INIT = "#d9741f"
COL_GOOD = "#2f7d32"
COL_BAD = "#b3261e"
COL_NEUTRAL = "#6b6b6b"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _point(results: Mapping[tuple, Mapping[str, Any]], key: tuple, metric: str) -> float:
    r = results.get(key)
    if r is None:
        return float("nan")
    v = r.get(metric)
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_tables() -> dict[str, Any]:
    d = C.ANALYSIS
    results: dict[tuple, dict[str, Any]] = {}
    import csv

    with open(d / "results.csv", "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            results[(row["population"], row["split"], row["family"], row["metric"])] = row
    return {
        "results": results,
        "uncertainty": json.loads((d / "uncertainty.json").read_text(encoding="utf-8")),
        "mechanism": json.loads((d / "mechanism.json").read_text(encoding="utf-8")),
        "checks": json.loads((d / "analysis_checks.json").read_text(encoding="utf-8")),
        "cases": json.loads((d / "cases.json").read_text(encoding="utf-8")),
    }


def _met(row, default=float("nan")) -> float:
    if not row:
        return default
    try:
        return float(row["value"])
    except Exception:
        return default


def _ci(row) -> tuple[float, float]:
    if not row:
        return (float("nan"), float("nan"))
    try:
        return float(row["ci95_low"]), float(row["ci95_high"])
    except Exception:
        return (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# figure 1: applicability x effect matrix
# ---------------------------------------------------------------------------


def figure_applicability(tables, out: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = tables["results"]
    unc = tables["uncertainty"]
    fams = list(C.FAMILIES)
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.6), dpi=150,
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.15]})

    # ---- (a) support: Test main queue and calibration coverage ----------
    ax = axes[0]
    n_pairs = [int(unc["family_support"][f]["main/test"]["pairs"]) for f in fams]
    n_cal = [int(unc["family_support"][f]["calibration/test"]["pairs"]) for f in fams]
    n_mol = [int(unc["family_support"][f]["calibration/test"]["molecules"]) for f in fams]
    x = np.arange(len(fams))
    ax.bar(x - W / 3, n_pairs, W / 1.5, color="#9db7d8", label="main queue pairs")
    ax.bar(x + W / 3, n_cal, W / 1.5, color=COL_TRAINED, label="dose-calipered pairs")
    for i, f in enumerate(fams):
        if n_pairs[i] == 0:
            ax.text(i, 5, "empty", ha="center", va="bottom", fontsize=9, color=COL_BAD, rotation=90)
        else:
            ax.text(i - W / 3, n_pairs[i] + 6, str(n_pairs[i]), ha="center", va="bottom", fontsize=7.5)
        if n_cal[i] == 0:
            ax.text(i + W / 3, 5, "no C", ha="center", va="bottom", fontsize=8, color=COL_BAD, rotation=90)
        else:
            ax.text(i + W / 3, n_cal[i] + 6, f"{n_cal[i]}\n({n_mol[i]} mol)", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([C.FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
    ax.set_ylabel("pairs (Test)")
    ax.set_title("(a) applicability — Test rewiring support", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, max(n_pairs + [1]) * 1.22)

    # ---- (b) rC_TRAINED vs rC_INIT with 95% interval --------------------
    ax = axes[1]
    for i, f in enumerate(fams):
        for off, metric, col, lab in (
            (-0.16, "rC", COL_TRAINED, "TRAINED"),
            (+0.16, "rC_init", COL_INIT, "INIT"),
        ):
            row = res.get(("calibration", "test", f, metric))
            v = _met(row)
            lo, hi = _ci(row)
            if not np.isfinite(v):
                ax.text(i, 0, "n/a", ha="center", va="center", fontsize=8, color=COL_NEUTRAL)
                continue
            ax.plot([i + off, i + off], [lo, hi], color=col, lw=2.2, solid_capstyle="round")
            ax.plot([i + off], [v], "o", color=col, ms=6.5, label=lab if i == 0 else None)
    ax.axhline(0, color="#999999", lw=1, ls=":")
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([C.FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
    ax.set_ylabel("rC = (aT − aC) / (aT + aC)")
    ax.set_title("(b) calibrated preference, Test (95% CI)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- (c) J_C / J_K / B ---------------------------------------------
    ax = axes[2]
    metrics = [("J_C_G_FIXED", "J_C (calipered)", COL_TRAINED), ("J_K_G_FIXED", "J_K (all)", "#5aa469"),
               ("B_G_FIXED", "B (loss vs O)", COL_NEUTRAL)]
    width = 0.26
    for j, (metric, lab, col) in enumerate(metrics):
        vals = [_met(res.get(("main", "test", f, metric))) for f in fams]
        ax.bar(np.arange(len(fams)) + (j - 1) * width, vals, width, color=col, label=lab)
    ax.axhline(0, color="#999999", lw=1, ls=":")
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([C.FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
    ax.set_ylabel("raw logD")
    ax.set_title("(c) response size, Test", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Six pre-registered connection families: support and selective response of the frozen LCRC-C1 model "
        "(zero training; G_FIXED scope)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"file": C.track_rel(out), "panel": "applicability x effect"}


# ---------------------------------------------------------------------------
# figure 2: parameter dependence (TRAINED vs INIT)
# ---------------------------------------------------------------------------


def figure_parameter_dependence(tables, out: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = tables["results"]
    fams = list(C.FAMILIES)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=150)

    for ax, split, title in ((axes[0], "test", "Test (main estimate)"), (axes[1], "valid", "Valid (direction check)")):
        for i, f in enumerate(fams):
            a = _met(res.get(("calibration", split, f, "rC_init")))
            b = _met(res.get(("calibration", split, f, "rC")))
            if not (np.isfinite(a) and np.isfinite(b)):
                ax.text(i, 0, "n/a", ha="center", va="center", fontsize=8, color=COL_NEUTRAL)
                continue
            lo_i, hi_i = _ci(res.get(("calibration", split, f, "rC_init")))
            lo_t, hi_t = _ci(res.get(("calibration", split, f, "rC")))
            ax.plot([i, i], [a, b], color="#bbbbbb", lw=3.0, zorder=1, solid_capstyle="round")
            ax.plot([i - 0.07, i - 0.07], [lo_i, hi_i], color=COL_INIT, lw=1.6, alpha=0.75)
            ax.plot([i + 0.07, i + 0.07], [lo_t, hi_t], color=COL_TRAINED, lw=1.6, alpha=0.75)
            ax.plot([i - 0.07], [a], "o", color=COL_INIT, ms=7,
                    label="INIT (verified training start)" if i == 0 else None, zorder=3)
            ax.plot([i + 0.07], [b], "o", color=COL_TRAINED, ms=7,
                    label="TRAINED (A_s0)" if i == 0 else None, zorder=3)
            ax.annotate(
                f"E={b - a:+.2f}",
                xy=(i, max(a, b)), xytext=(i, max(a, b) + 0.16),
                ha="center", fontsize=7.5,
                color=COL_GOOD if (b - a) > 0 else COL_BAD,
            )
        ax.axhline(0, color="#999999", lw=1, ls=":")
        ax.set_xticks(np.arange(len(fams)))
        ax.set_xticklabels([C.FAMILY_LABEL[f] for f in fams], rotation=20, ha="right")
        ax.set_ylabel("rC (calipered preference)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(
        "Does the calibrated preference depend on the trained parameters?  "
        "E = rC_TRAINED − rC_INIT on identical views, C set, weights, queue", fontsize=11, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"file": C.track_rel(out), "panel": "parameter dependence"}


# ---------------------------------------------------------------------------
# figure 3: mechanism (Mobius) matrix
# ---------------------------------------------------------------------------


TERM_LABEL = {"g1": "S1", "g2": "S2", "g3": "scalar", "g12": "S1×S2", "g13": "S1×S", "g23": "S2×S",
              "g123": "S1×S2×S"}


def figure_mechanism(tables, out: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mech = tables["mechanism"]["summary"]
    fams = [f for f in C.FAMILIES if f"{'test'}/calibration/{f}" in mech]
    terms = ("g1", "g2", "g3", "g12", "g13", "g23", "g123")
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.4), dpi=150)
    for row, (population, title) in enumerate(
        (("calibration", "T − C (dose-calipered control)"), ("main", "T − K0 (all pairs)"))
    ):
        for col, space in enumerate(("H", "L")):
            ax = axes[row, col]
            M = np.array([[mech[f"test/{population}/{f}"][space]["terms"][t] for t in terms] for f in fams], dtype=float)
            vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
            vmax = max(vmax, 1e-6)
            im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    if not np.isfinite(v):
                        continue
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.6,
                            color="white" if abs(v) > 0.55 * vmax else "#222222")
            ax.set_xticks(np.arange(len(terms)))
            ax.set_xticklabels([TERM_LABEL[t] for t in terms], rotation=25, ha="right")
            ax.set_yticks(np.arange(len(fams)))
            ax.set_yticklabels([C.FAMILY_LABEL[f] for f in fams])
            ax.set_title(f"{'H: |Δp|' if space == 'H' else 'L: |Δloss vs original label|'} — {title}", fontsize=9.5)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle(
        "Internal path: three-factor Möbius decomposition of the local-input change (G_FIXED, global42 = O, "
        "TRAINED only).  Seven terms sum to the full change (≤1e-6).", fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"file": C.track_rel(out), "panel": "mechanism"}


# ---------------------------------------------------------------------------
# case figures
# ---------------------------------------------------------------------------

ELEMENT_OF_CLASS = {i: name for i, (name, _arom) in enumerate(C.RD.ATOM_CLASSES)}
COLORS = {"C": "#222222", "N": "#1f4fd8", "O": "#d62728", "S": "#b8860b", "F": "#2ca02c", "Cl": "#2ca02c",
          "Br": "#8c564b", "I": "#7b1fa2", "P": "#e377c2", "Si": "#8c8c8c", "B": "#17becf", "H": "#777777"}
_BOND_NAMES = {0: "single", 1: "double", 2: "triple", 3: "aromatic", 4: "unknown"}


def _verify_mapping(smiles: str, row_id: int, raw_row: Mapping[str, Any]) -> dict[str, Any]:
    derived = C.RD.smiles_to_raw_row(str(smiles), int(row_id))
    checks = {
        "n_atoms": int(len(derived["atom_class"])) == int(len(raw_row["atom_class"])),
        "atom_class": np.array_equal(np.asarray(derived["atom_class"]), np.asarray(raw_row["atom_class"], dtype=np.int64)),
        "atom_charge": np.array_equal(np.asarray(derived["atom_charge"]), np.asarray(raw_row["atom_charge"], dtype=np.int64)),
        "atom_h": np.array_equal(np.asarray(derived["atom_h"]), np.asarray(raw_row["atom_h"], dtype=np.int64)),
        "bond_pairs": {tuple(sorted(p)) for p in np.asarray(derived["edge_index"]).T.tolist()}
        == {tuple(sorted(p)) for p in np.asarray(raw_row["edge_index"]).T.tolist()},
        "bond_classes": sorted(np.asarray(derived["edge_class"]).tolist())
        == sorted(np.asarray(raw_row["edge_class"]).tolist()),
    }
    return {"all_equal": all(checks.values()), "fields": checks}


def _bond_segments(p0, p1, style: str):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy)) or 1.0
    nx, ny = -dy / length, dx / length
    if style in ("single", "aromatic", "unknown"):
        return [((x0, y0), (x1, y1))]
    if style == "double":
        off = 0.075
        return [((x0 + nx * off, y0 + ny * off), (x1 + nx * off, y1 + ny * off)),
                ((x0 - nx * off, y0 - ny * off), (x1 - nx * off, y1 - ny * off))]
    off = 0.085
    return [((x0, y0), (x1, y1)),
            ((x0 + nx * off, y0 + ny * off), (x1 + nx * off, y1 + ny * off)),
            ((x0 - nx * off, y0 - ny * off), (x1 - nx * off, y1 - ny * off))]


def _draw_panel(ax, *, title, row, pos, roles, remove, add, target_bonds, control_bonds):
    import graphlib as G

    bu, bv, bc, _bs = G.physical_bonds(row)
    ac = np.asarray(row["atom_class"], dtype=np.int64)
    existing = {(int(u), int(v)): int(c) for u, v, c in zip(bu.tolist(), bv.tolist(), bc.tolist())}
    remove_set = {tuple(sorted((int(a), int(b)))) for a, b in remove}
    add_set = {tuple(sorted((int(a), int(b)))) for a, b in add}
    for pair, color in [(p, "#d02020") for p in target_bonds] + [(p, "#2040d0") for p in control_bonds]:
        if tuple(sorted(pair)) in existing:
            p0, p1 = pos[pair[0]], pos[pair[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=13, alpha=0.38, solid_capstyle="round", zorder=1)
    for (u, v), klass in existing.items():
        for a, b in _bond_segments(pos[u], pos[v], _BOND_NAMES.get(int(klass), "unknown")):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", lw=1.8, zorder=2, solid_capstyle="round")
    for a, b in add_set:
        p0, p1 = pos[a], pos[b]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#ff8c00", lw=4.0, zorder=3, solid_capstyle="round", alpha=0.95)
    target_set = {tuple(sorted(p)) for p in target_bonds}
    control_set = {tuple(sorted(p)) for p in control_bonds}
    for a, b in remove_set:
        p0, p1 = pos[a], pos[b]
        key = tuple(sorted((a, b)))
        if key in target_set:
            color, lw = "#d02020", 2.4
        elif key in control_set:
            color, lw = "#2040d0", 2.4
        else:
            color, lw = "#8a8a8a", 1.8
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=(0, (3, 2.2)), zorder=3, alpha=0.95)
    for idx, (x, y) in pos.items():
        element = ELEMENT_OF_CLASS.get(int(ac[idx]), "?")
        color = COLORS.get(element, "#444444")
        ax.scatter([x], [y], s=150, color="white", edgecolor=color, linewidths=1.6, zorder=4)
        ax.text(x, y, element, ha="center", va="center", fontsize=9, color=color, fontweight="bold", zorder=5)
        ax.text(x + 0.10, y + 0.10, f"{idx}", ha="left", va="bottom", fontsize=6.5, color="#666666", zorder=5)
        if idx in roles:
            ax.text(x - 0.10, y - 0.14, roles[idx], ha="right", va="top", fontsize=8.5,
                    color="#b02020" if roles[idx] in ("c", "t") else "#2030a0", fontweight="bold", zorder=6)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5)


def _parse(code: str) -> list[tuple[int, int]]:
    out = []
    for chunk in str(code).split("."):
        if not chunk:
            continue
        a, b = chunk.split("-")
        out.append((int(a), int(b)))
    return out


def draw_case(case: Mapping[str, Any], raw_row: Mapping[str, Any], out_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem

    import rewire as R

    smiles = str(raw_row["smiles"])
    mapping = _verify_mapping(smiles, int(case["row_id"]), raw_row)
    if not mapping["all_equal"]:
        raise SystemExit(f"SMILES_GRAPH_MAPPING_FAILED row {case['row_id']}: {mapping['fields']}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"row {case['row_id']}: RDKit cannot parse {smiles!r}")
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos = {int(a.GetIdx()): (float(conf.GetAtomPosition(int(a.GetIdx())).x),
                             float(conf.GetAtomPosition(int(a.GetIdx())).y)) for a in mol.GetAtoms()}
    n_atoms = int(np.asarray(raw_row["atom_class"]).shape[0])
    if len(pos) != n_atoms:
        raise SystemExit(f"atom count mismatch row {case['row_id']}: {len(pos)} != {n_atoms}")
    roles = {int(case["center"]): "c", int(case["target"]): "t", int(case["control"]): "k"}
    t_remove, t_add = _parse(case["t_remove"]), _parse(case["t_add"])
    k_remove, k_add = _parse(case["k_remove"]), _parse(case["k_add"])
    # donor atoms
    for a, b in t_remove:
        if (a, b) != (int(case["center"]), int(case["target"])) and (b, a) != (int(case["center"]), int(case["target"])):
            roles.setdefault(a, "u")
            roles.setdefault(b, "v")
    base = {**dict(raw_row), "mol_index": int(case["row_id"])}
    t_row, _ = R.edit_row(base, t_remove, t_add)
    k_row, _ = R.edit_row(base, k_remove, k_add)
    target_bonds = [tuple(sorted((int(case["center"]), int(case["target"]))))]
    control_bonds = [tuple(sorted((int(case["center"]), int(case["control"]))))]
    p = case["preds"]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.4), dpi=150)
    _draw_panel(axes[0], title="O — original measured molecule", row=base, pos=pos, roles=roles, remove=[], add=[],
                target_bonds=target_bonds, control_bonds=control_bonds)
    _draw_panel(axes[1], title=f"T — {C.FAMILY_LABEL[case['family']]} pattern destroyed (rewired, no measured label)",
                row=t_row, pos=pos, roles=roles, remove=t_remove, add=t_add,
                target_bonds=target_bonds, control_bonds=control_bonds)
    _draw_panel(axes[2], title=f"K0 — pattern kept, same donor rewired (no measured label)",
                row=k_row, pos=pos, roles=roles, remove=k_remove, add=k_add,
                target_bonds=target_bonds, control_bonds=control_bonds)
    lines = [
        f"row {case['row_id']}  family {case['family']}  instance {case['instance_id']}  ({case['case_kind']}, "
        f"rank metric {case['ranking_metric']})  scaffold {case['scaffold_key'][:60]}"
        f"{'…' if len(case['scaffold_key']) > 60 else ''} (seen in Train: {case['scaffold_seen_in_train']})",
        f"y(original) = {case['y']:.3f} logD   f(O) = {p['O']:+.3f}   "
        f"T = {p['T_G_FIXED']:+.3f}   K0 = {p['K0_G_FIXED']:+.3f}"
        + (f"   mean C = {p['C_mean_G_FIXED']:+.3f} (n={case['n_c_candidates']})" if p.get("C_mean_G_FIXED") is not None else "   (no dose-calipered C)"),
        f"G_FIXED:  aT = {case['aT_G_FIXED']:.3f}   aK0 = {case['aK_G_FIXED']:.3f}"
        + (f"   aC = {case['aC_G_FIXED']:.3f}" if case.get("aC_G_FIXED") is not None else "")
        + f"   J_C = {case['J_C_G_FIXED'] if case.get('J_C_G_FIXED') is not None else float('nan'):+.3f}"
        + f"   J_K0 = {case['J_K_G_FIXED']:+.3f}   B = {case['B_G_FIXED']:+.3f}",
        f"T removes {case['t_remove']} adds {case['t_add']}    K0 removes {case['k_remove']} adds {case['k_add']}",
        "solid black = bonds of that graph; thick orange = added; dashed red = removed target bond; "
        "dashed blue = removed control bond; dashed grey = removed donor bond",
        "the printed label is the ORIGINAL molecule's measured logD, used only as the computational reference of D/B; "
        "the T/K0 graphs were never synthesised and have no measured value.",
    ]
    fig.text(0.01, 0.015, "\n".join(lines), fontsize=8.0, va="bottom", ha="left", family="monospace")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"tag": case["tag"], "row_id": int(case["row_id"]), "file": C.track_rel(out_path), "mapping": mapping,
            "instance_id": case["instance_id"], "ranking_metric": case["ranking_metric"], "case_kind": case["case_kind"]}


def main() -> int:
    tables = load_tables()
    drawn = []
    drawn.append(figure_applicability(tables, C.ANALYSIS / "fig_applicability_matrix.png"))
    drawn.append(figure_parameter_dependence(tables, C.ANALYSIS / "fig_parameter_dependence.png"))
    drawn.append(figure_mechanism(tables, C.ANALYSIS / "fig_mechanism_matrix.png"))
    for f in drawn:
        C.log(f"drew {f['file']}")
    raw_rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    cases = []
    for case in tables["cases"]["cases"]:
        out = C.CASES / f"case_{case['tag']}_row{case['row_id']}.png"
        cases.append(draw_case(case, raw_rows[int(case["row_id"])], out))
        C.log(f"drew {cases[-1]['file']} ({cases[-1]['tag']})")
    C.write_json(
        C.CASES / "cases_manifest.json",
        {"main_population": tables["cases"]["main_population"], "note": tables["cases"]["note"],
         "families": tables["cases"]["families"], "cases": cases},
    )
    C.write_json(C.ANALYSIS / "figures_manifest.json", {"figures": drawn, "cases": cases})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
