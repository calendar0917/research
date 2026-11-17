"""Figures: highlighted molecules (RDKit) and effect distributions.

Only a bounded, deterministic selection is rendered: the first 8 selected Valid
molecules, the first 8 selected Test molecules (both ordered by official row
id), and 4 clearly-labelled result-driven diagnostic cases per dataset.  The
several hundred raw molecules are *not* committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import as_common as C

DPI = 160
VALID_TEST_N = 8
DIAGNOSTIC_N = 2


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------


_DRAW2D = None
_PRELOADED: list[str] = []


def _preload_so(soname: str) -> str | None:
    """Resolve a missing shared library from the local nix store (if present).

    This machine's RDKit wheel links against X render libraries that are not on the
    default loader path; the aggregate stage is CPU-only and local, so resolving
    them here keeps the frozen numerical pipeline untouched.  Every preloaded path
    is recorded in ``analysis/figures/index.json`` for provenance.
    """
    import ctypes
    import glob

    base = soname.split(".so")[0]
    for path in sorted(glob.glob(f"/nix/store/*/lib/{base}.so*")):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if path not in _PRELOADED:
            _PRELOADED.append(path)
        return path
    return None


def _draw2d():
    """``rdkit.Chem.Draw.rdMolDraw2D`` with on-demand dependency preloading."""
    global _DRAW2D
    if _DRAW2D is not None:
        return _DRAW2D
    import re

    last = None
    for _ in range(24):
        try:
            from rdkit.Chem.Draw import rdMolDraw2D

            _DRAW2D = rdMolDraw2D
            return rdMolDraw2D
        except ImportError as exc:           # missing .so for this machine
            last = exc
            match = re.search(r"(lib\w+\.so(?:\.\d+)*)", str(exc))
            if not match or _preload_so(match.group(1)) is None:
                raise
    raise ImportError(f"could not resolve the RDKit drawing backend: {last}")


def _rdkit():
    from rdkit import Chem

    return Chem, _draw2d()


def _mol_from_smiles(smiles: str, n_atoms: int, bond_index: Sequence[Sequence[int]]):
    """RDKit molecule plus a verification that cached atom/bond ids line up."""
    Chem, _draw = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, {"parsed": False, "reason": "RDKit could not parse the SMILES"}
    info = {"parsed": True, "n_atoms_match": int(mol.GetNumAtoms()) == int(n_atoms),
            "all_bonds_present": True}
    for (u, v) in bond_index:
        if mol.GetBondBetweenAtoms(int(u), int(v)) is None:
            info["all_bonds_present"] = False
            break
    info["usable"] = bool(info["n_atoms_match"] and info["all_bonds_present"])
    return mol, info


def render_molecule(
    mol_record: Mapping[str, Any],
    bond_index: Sequence[Sequence[int]],
    out_path: Path,
    *,
    title_lines: Sequence[str],
) -> dict[str, Any]:
    """One annotated 2-D structure with the Top-1/2/3 bond sets highlighted."""
    Chem, rdMolDraw2D = _rdkit()
    mol, info = _mol_from_smiles(mol_record.get("smiles", ""), mol_record["n_atoms"], bond_index)
    if mol is None or not info.get("usable"):
        return {"rendered": False, "reason": info.get("reason", "atom/bond identity mismatch"), "info": info}
    palette = [(0.85, 0.10, 0.10), (0.10, 0.45, 0.85), (0.10, 0.65, 0.25)]
    atom_colors: dict[int, tuple[float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float]] = {}
    atom_radius: dict[int, float] = {}
    for rank, rec in enumerate(mol_record.get("topk") or []):
        color = palette[min(rank, len(palette) - 1)]
        for a in rec["atoms"]:
            atom_colors[int(a)] = color
            atom_radius[int(a)] = 0.42
        for b in rec["bonds"]:
            u, v = int(bond_index[int(b)][0]), int(bond_index[int(b)][1])
            bond = mol.GetBondBetweenAtoms(u, v)
            if bond is not None:
                bond_colors[bond.GetIdx()] = color
    drawer = rdMolDraw2D.MolDraw2DCairo(560, 460)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.addStereoAnnotation = False
    drawer.drawOptions().annotationFontScale = 0.7
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=sorted(atom_colors) or None,
        highlightBonds=sorted(bond_colors) or None,
        highlightAtomColors=atom_colors or None,
        highlightBondColors=bond_colors or None,
        highlightAtomRadii=atom_radius or None,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)
    return {"rendered": True, "path": C.relpath(out_path), "info": info, "title": list(title_lines)}


def _title_lines(rec: Mapping[str, Any], *, tag: str, controls: Mapping[str, Any] | None) -> list[str]:
    top1 = (rec.get("topk") or [{}])[0]
    eff = (top1 or {}).get("effect", {})
    lines = [
        f"{rec['dataset']} {rec['split']} row_id={rec['row_id']}  {tag}",
        f"y={rec['y_raw']:.3f}  pred={rec['pred_raw_base']:.3f}  atoms={rec['n_atoms']} bonds={rec['n_bonds']}",
        f"TOP1 k={top1.get('n_bonds')} A={top1.get('score_A', float('nan')):.4f} "
        f"delta={eff.get('signed_delta', float('nan')):.4f} "
        f"|d|/std={eff.get('abs_delta_over_train_std', float('nan')):.4f}",
        f"  S1={eff.get('delta_L', float('nan')):.4f} "
        f"rho={eff.get('delta_P', float('nan')):.4f} "
        f"rho_share={eff.get('rho_share', float('nan')):.3f}",
    ]
    for rank, t in enumerate((rec.get("topk") or [])[1:], start=2):
        e = t["effect"]
        lines.append(f"TOP{rank} k={t['n_bonds']} A={t['score_A']:.4f} delta={e['signed_delta']:.4f}")
    if controls:
        lines.append(
            f"matched random n={controls.get('n_matched')} "
            f"median|d|={_fmt(controls.get('median_abs_delta'))} "
            f"ratio={_fmt(controls.get('effect_ratio'))}"
        )
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# selections
# ---------------------------------------------------------------------------


def figure_selection(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic Valid/Test picks plus the labelled diagnostic cases."""
    out: dict[str, Any] = {"main": [], "diagnostics": []}
    for split in ("valid", "test"):
        sub = sorted(
            [r for r in records if r["split"] == split],
            key=lambda r: int(r["row_id"]),
        )[:VALID_TEST_N]
        out["main"].extend(sub)
    graded = sorted(
        [r for r in records if r.get("topk")],
        key=lambda r: (-abs(r["topk"][0]["effect"]["signed_delta"]), int(r["row_id"])),
    )
    largest = graded[:DIAGNOSTIC_N]
    smallest = sorted(graded, key=lambda r: (abs(r["topk"][0]["effect"]["signed_delta"]), int(r["row_id"])))[
        :DIAGNOSTIC_N
    ]
    out["diagnostics"] = {"largest": largest, "smallest": smallest}
    return out


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


def distribution_figure(rows: Sequence[Mapping[str, Any]], dataset: str, out_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = [r for r in rows if r["dataset"] == dataset and r["has_top1"]]
    panels = [
        ("abs_delta_top1_over_train_std", "|delta| / train target std", True),
        ("top1_rho_share", "rho_share (full-graph readout share)", False),
        ("top1_cancellation", "cancellation", False),
        ("effect_ratio", "effect_ratio vs matched random", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
    for ax, (key, label, logx) in zip(axes.ravel(), panels):
        for split, color in (("valid", "#1f77b4"), ("test", "#d62728")):
            vals = [r[key] for r in sub if r["split"] == split and r.get(key) is not None]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                ax.hist(vals, bins=18, alpha=0.55, label=split, color=color)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        if logx:
            ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        if key == "effect_ratio":
            ax.axvline(1.0, color="k", lw=0.8, ls="--")
    fig.suptitle(f"{dataset}: Top-1 discovery vs CONDITIONAL_S1_S2_EVIDENCE_MASK", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return {"path": C.relpath(out_path)}


def readout_figure(rows: Sequence[Mapping[str, Any]], dataset: str, out_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = [r for r in rows if r["dataset"] == dataset and r["has_top1"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    for split, color in (("valid", "#1f77b4"), ("test", "#d62728")):
        xs = [r["top1_delta_L"] for r in sub if r["split"] == split]
        ys = [r["top1_delta_P"] for r in sub if r["split"] == split]
        ax.scatter(xs, ys, s=18, alpha=0.7, label=split, color=color)
    lim = max(
        [abs(v) for r in sub for v in (r["top1_delta_L"], r["top1_delta_P"])] + [1e-3]
    )
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":")
    ax.set_xlabel("delta_L (local sum readout)", fontsize=8)
    ax.set_ylabel("delta_P (nonlinear full-graph readout)", fontsize=8)
    ax.set_title("readout decomposition of the Top-1 mask", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax = axes[1]
    for split, color in (("valid", "#1f77b4"), ("test", "#d62728")):
        xs = [abs(r.get("top1_s1_only_signed_delta", 0.0)) for r in sub if r["split"] == split]
        ys = [abs(r.get("top1_s2_only_signed_delta", 0.0)) for r in sub if r["split"] == split]
        ax.scatter(xs, ys, s=18, alpha=0.7, label=split, color=color)
    ax.plot([0, lim], [0, lim], color="k", lw=0.6, ls=":")
    ax.set_xlabel("|delta| S1_ONLY", fontsize=8)
    ax.set_ylabel("|delta| S2_ONLY", fontsize=8)
    ax.set_title("S1 vs S2 response (same bond set)", fontsize=9)
    ax.tick_params(labelsize=7)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return {"path": C.relpath(out_path)}


__all__ = [
    "render_molecule",
    "figure_selection",
    "distribution_figure",
    "readout_figure",
    "VALID_TEST_N",
    "DIAGNOSTIC_N",
]
