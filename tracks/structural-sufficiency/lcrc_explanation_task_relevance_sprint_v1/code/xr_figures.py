"""Round figures: forest plots, prediction scatter, failure modes and real molecules.

The molecule panels are deliberately *result-driven* illustrations: the three
cases are selected after the formal run to show a success, a failure and a
model-inconsistent pair.  They are not a random sample and are labelled as such.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import xr_common as C
import xr_assets as A
import xr_models as M


COLORS = {"LCRC": "#1f5fbf", "GINE": "#1f9d55", "LOCAL_LCRC": "#d97706"}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _forest(comp: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    metrics = [
        ("atom_jaccard_adv_advantage", "Atom Jaccard\nadvantage"),
        ("atom_recall_adv_advantage", "Atom recall\nadvantage"),
        ("bond_recall_adv_advantage", "Bond recall\nadvantage"),
        ("hit_boundary_adv_advantage", "Attachment-boundary\nhit advantage"),
    ]
    models = ["LCRC", "GINE", "LOCAL_LCRC"]
    rows = []
    for metric, label in metrics:
        for model in models:
            hits = [r for r in comp if r["metric"] == metric and r["stratum"] == "all" and r["model"] == model]
            if hits:
                rows.append((label, model, _f(hits[0]["point"]), _f(hits[0]["ci_low"]), _f(hits[0]["ci_high"])))
    oracle = []
    for model in models:
        hits = [r for r in comp if r["metric"] == "oracle_enrichment" and r["stratum"] == "all" and r["model"] == model]
        if hits:
            oracle.append((model, _f(hits[0]["point"]), _f(hits[0]["ci_low"]), _f(hits[0]["ci_high"])))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [3.2, 1.0]})
    ax = axes[0]
    yticks = []
    ylabels = []
    y = 0
    for label, _metric_label in [(m[1], m[0]) for m in metrics]:
        for model in models:
            rec = next((r for r in rows if r[0] == label and r[1] == model), None)
            if rec is None:
                continue
            _lab, mod, point, lo, hi = rec
            ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color=COLORS[mod], capsize=3, ms=6)
            yticks.append(y)
            ylabels.append(f"{mod}")
            y += 1
        y += 0.6
    ax.axvline(0.0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Top-1 minus mean same-size random control")
    ax.set_title("Real-label structural localisation (all evaluable pairs)", fontsize=10)
    # group labels
    group_y = []
    start = 0
    for _metric, label in metrics:
        n = sum(1 for r in rows if r[0] == label)
        group_y.append(start + (n - 1) / 2)
        start += n + 0.6
    for gy, (_metric, label) in zip(group_y, metrics):
        ax.text(-0.085, gy, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.08, 0.18)

    ax = axes[1]
    ys = np.arange(len(oracle))
    for i, (model, point, lo, hi) in enumerate(oracle):
        ax.errorbar(point, i, xerr=[[point - lo], [hi - point]], fmt="o", color=COLORS[model], capsize=3, ms=7)
    ax.axvline(0.0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in oracle], fontsize=9)
    ax.set_xlabel("oracle enrichment (mass-capture units)")
    ax.set_title("WC_LOGP contribution capture", fontsize=10)
    ax.set_xlim(-0.02, 0.42)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"path": C.relpath(out), "n_real_label_rows": len(rows), "oracle": oracle}


def _scatter(pairs: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    for ax, model in zip(axes, ("LCRC", "GINE")):
        dy = np.asarray([_f(r["delta_y"]) for r in pairs], dtype=np.float64)
        dp = np.asarray([_f(r[f"delta_pred_{model}"]) for r in pairs], dtype=np.float64)
        mask = np.isfinite(dy) & np.isfinite(dp)
        ax.scatter(dy[mask], dp[mask], s=9, alpha=0.35, color=COLORS[model], linewidths=0)
        lim = np.nanmax(np.abs(np.concatenate([dy[mask], dp[mask]]))) + 0.2
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, ls="--", alpha=0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"real $\Delta y$ (logD)")
        ax.set_ylabel(r"predicted $\Delta \hat y$")
        ax.set_title(model, fontsize=11)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Pair-wise prediction change on the frozen 823 pairs", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"path": C.relpath(out)}


def _failure_modes(pairs: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    order = ["accurate_localized", "accurate_mislocalized", "inaccurate_localized", "inaccurate_mislocalized"]
    labels = ["accurate &\nlocalized", "accurate &\nmislocalized", "inaccurate &\nlocalized", "inaccurate &\nmislocalized"]
    colors = ["#1f9d55", "#f59e0b", "#60a5fa", "#dc2626"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(2)
    bottom = np.zeros(2)
    for key, label, color in zip(order, labels, colors):
        vals = np.asarray([
            sum(1 for r in pairs if r[f"failure_mode_{m}"] == key) for m in ("LCRC", "GINE")
        ], dtype=np.float64)
        ax.bar(x, vals, bottom=bottom, color=color, label=label, width=0.55)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["LCRC", "GINE"])
    ax.set_ylabel("pairs (823 total)")
    ax.set_title("Failure modes: pair accuracy (|Δŷ−Δy|≤0.25) vs localisation success", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    for i, total in enumerate(bottom):
        ax.text(i, total + 8, f"{int(total)}", ha="center", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"path": C.relpath(out)}


def _oracle_hist(oracle: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bins = np.linspace(-0.1, 1.0, 45)
    for model in ("LCRC", "GINE", "LOCAL_LCRC"):
        vals = np.asarray([_f(r["oracle_enrichment"]) for r in oracle if r["model"] == model], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, alpha=0.45, label=model, color=COLORS[model], density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("oracle enrichment (Top-1 − random median mass capture)")
    ax.set_ylabel("density (log scale)")
    ax.set_title("Zero-inflated WC_LOGP capture: many candidates miss all changed mass", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"path": C.relpath(out)}


# ---------------------------------------------------------------------------
# molecule panels
# ---------------------------------------------------------------------------


def _load_selected_endpoints(pair_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {pid: [] for pid in pair_ids}
    with open(C.RESULTS_DIR / "measured/endpoints.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = str(rec["meta"]["pair_id"])
            if pid in out:
                out[pid].append(rec)
    return out


def _select_cases(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success = [
        r for r in pairs
        if r["failure_mode_LCRC"] == "accurate_localized" and r["failure_mode_GINE"] == "accurate_localized"
        and (_f(r["atom_recall_LCRC"]) or 0) >= 0.5 and (_f(r["atom_recall_GINE"]) or 0) >= 0.5
        and (_f(r["oracle_enrichment_LCRC"]) or 0) > 0.3
    ]
    success.sort(key=lambda r: (-(_f(r["oracle_enrichment_LCRC"]) or 0), r["pair_id"]))
    failure = [
        r for r in pairs
        if r["failure_mode_LCRC"] == "accurate_mislocalized"
        and (_f(r["abs_delta_error_LCRC"]) or 9) <= 0.01
        and abs(_f(r["delta_y"]) or 0) >= 0.2
    ]
    failure.sort(key=lambda r: ((_f(r["atom_recall_LCRC"]) or 9), -abs(_f(r["delta_y"]) or 0), r["pair_id"]))
    inconsistent = [
        r for r in pairs
        if r["failure_mode_LCRC"] == "accurate_mislocalized" and r["failure_mode_GINE"] == "accurate_localized"
        and str(r["pair_accurate_LCRC"]).lower() in ("true", "1") and str(r["pair_accurate_GINE"]).lower() in ("true", "1")
    ]
    inconsistent.sort(key=lambda r: (-((_f(r["oracle_enrichment_GINE"]) or 0) - (_f(r["oracle_enrichment_LCRC"]) or 0)), r["pair_id"]))
    if not success or not failure or not inconsistent:
        raise RuntimeError("could not select the three illustrative cases")
    return {
        "success": success[0],
        "failure": failure[0],
        "inconsistent": inconsistent[0],
        "rules": {
            "success": "LCRC and GINE both accurate_localized; atom recall >= 0.5 both; highest LCRC oracle enrichment",
            "failure": "LCRC accurate_mislocalized; |Delta-y error| <= 0.01; |Delta y| >= 0.2; lowest LCRC atom recall",
            "inconsistent": "LCRC accurate_mislocalized and GINE accurate_localized; both pair-accurate; largest GINE-minus-LCRC oracle enrichment",
        },
    }


def _pick_side(records: Sequence[Mapping[str, Any]], case: str) -> Mapping[str, Any]:
    def score(r: Mapping[str, Any]) -> tuple[float, float]:
        l = _f(r.get("localisation", {}).get("LCRC", {}).get("top1", {}).get("atom_recall")) or 0.0
        g = _f(r.get("localisation", {}).get("GINE", {}).get("top1", {}).get("atom_recall")) or 0.0
        if case == "success":
            return (l + g, len(r["edit"]["edit_atoms"]))
        if case == "failure":
            return (-(l + g), len(r["edit"]["edit_atoms"]))
        return (g - l, len(r["edit"]["edit_atoms"]))

    return max(records, key=score)


def _rdkit_images(cases: Mapping[str, Any], endpoints: Mapping[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    """Real molecular graphs with real atom/bond highlights.

    RDKit is used for SMILES parsing, atom identity and 2-D coordinates only.
    The drawing itself is matplotlib so the figure does not depend on the
    optional XRender/Cairo RDKit drawing backend.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    C.install_source_paths()
    b = M.bundle(C.DATASET)
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 10.2))
    case_order = [
        ("success", "SUCCESS (edit-localised)"),
        ("failure", "FAILURE (accurate prediction, missed edit)"),
        ("inconsistent", "INCONSISTENT (LCRC misses, GINE hits)"),
    ]
    chosen = {}
    for row_idx, (case, title) in enumerate(case_order):
        pair = cases[case]
        recs = endpoints[str(pair["pair_id"])]
        rec = _pick_side(recs, case)
        chosen[case] = {"pair_id": pair["pair_id"], "side": rec["meta"]["side"], "smiles": rec["meta"]["smiles"], "row_id": rec["meta"]["row_id"]}
        mol = Chem.MolFromSmiles(rec["meta"]["smiles"])
        if mol is None or mol.GetNumAtoms() != int(rec["meta"]["n_atoms"]):
            raise RuntimeError(f"RDKit atom-order mismatch for {pair['pair_id']}")
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        xy = np.asarray([[conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())], dtype=np.float64)
        xy = (xy - xy.min(axis=0)) / np.maximum(1e-9, xy.max(axis=0) - xy.min(axis=0))
        cache_row = int(rec["meta"]["cache_row"])
        split = str(rec["meta"]["split"])
        global_row = int(np.asarray(b.global_row_ids[split], dtype=np.int64)[cache_row])
        raw = A._raw(b, global_row)
        bond_to_rd = {}
        for bid, (u, v) in enumerate(zip(raw["edge_u"].tolist(), raw["edge_v"].tolist())):
            rd = mol.GetBondBetweenAtoms(int(u), int(v))
            if rd is not None:
                bond_to_rd[int(bid)] = int(rd.GetIdx())

        edit_atoms = set(int(v) for v in rec["edit"]["edit_atoms"])
        edit_bonds = {tuple(sorted(int(x) for x in bb)) for bb in rec["edit"]["edit_bonds"]}
        edit_rd_bonds = {bond_to_rd[bb] for bb in edit_bonds if bb in bond_to_rd}

        def draw(ax, atoms, bonds, color, title_text, extra_atoms=None, extra_bonds=None):
            atoms = set(int(a) for a in atoms)
            bonds = set(int(bb) for bb in bonds)
            extra_atoms = set(int(a) for a in (extra_atoms or set()))
            extra_bonds = set(int(bb) for bb in (extra_bonds or set()))
            atom_color = {a: color for a in atoms}
            atom_color.update({a: (0.95, 0.55, 0.45) for a in extra_atoms if a not in atoms})
            bond_color = {bb: color for bb in bonds}
            bond_color.update({bb: (0.85, 0.15, 0.10) for bb in extra_bonds if bb not in bonds})
            for bond in mol.GetBonds():
                i, j = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
                bi = int(bond.GetIdx())
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=bond_color.get(bi, (0.72, 0.72, 0.72)), lw=3.0 if bi in bond_color else 1.4, zorder=1)
            for i in range(mol.GetNumAtoms()):
                fill = atom_color.get(i, (0.92, 0.92, 0.92))
                ax.scatter([xy[i, 0]], [xy[i, 1]], s=260, c=[fill], edgecolors="black", linewidths=0.6, zorder=2)
                symbol = mol.GetAtomWithIdx(i).GetSymbol()
                dark = max(fill) < 0.65
                ax.text(xy[i, 0], xy[i, 1], symbol, ha="center", va="center", fontsize=6.2, color="white" if dark else "black", zorder=3)
            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(-0.08, 1.08)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title_text, fontsize=8)

        def candidate(rec_model: str):
            top = rec.get("localisation", {}).get(rec_model, {}).get("top1", {})
            atoms = set(int(v) for v in top.get("candidate_atoms", []))
            bonds = set()
            for bb in top.get("candidate_bonds", []):
                key = tuple(sorted(int(x) for x in bb))
                if key in bond_to_rd:
                    bonds.add(bond_to_rd[key])
            return atoms, bonds

        draw(axes[row_idx, 0], edit_atoms, edit_rd_bonds, (0.85, 0.15, 0.10), "Edit region (red)")
        la, lb = candidate("LCRC")
        draw(axes[row_idx, 1], la, lb, (0.10, 0.30, 0.85), "LCRC Top-1 (blue); edit-only red", extra_atoms=edit_atoms - la, extra_bonds=edit_rd_bonds - lb)
        ga, gb = candidate("GINE")
        draw(axes[row_idx, 2], ga, gb, (0.10, 0.60, 0.20), "GINE Top-1 (green); edit-only red", extra_atoms=edit_atoms - ga, extra_bonds=edit_rd_bonds - gb)
        delta = np.abs(np.asarray(rec.get("oracle_delta", []), dtype=np.float64))
        if delta.size and np.nanmax(delta) > 0:
            top_atoms = set(int(i) for i in np.argsort(-np.nan_to_num(delta))[: min(8, int((delta > 0).sum()))])
        else:
            top_atoms = set()
        draw(axes[row_idx, 3], top_atoms, set(), (0.95, 0.55, 0.10), "WC_LOGP |delta c| high atoms (orange)")
        axes[row_idx, 0].text(-0.08, 0.5, title, transform=axes[row_idx, 0].transAxes, ha="right", va="center", fontsize=9, fontweight="bold", rotation=90)
    fig.suptitle("Result-driven illustrative cases: edit region vs auto-discovered connected subgraphs vs WC_LOGP contribution", fontsize=10)
    fig.text(0.5, 0.005, "Red: structural edit region. Blue: LCRC Top-1. Green: GINE Top-1. Orange: top-|delta c| WC_LOGP atoms. Cases selected after the run, not a random sample.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"path": C.relpath(out), "cases": chosen, "rules": cases["rules"]}


def main() -> int:
    import sys
    sys.path.insert(0, str(C.CODE_DIR))
    comp = _read_csv(C.ANALYSIS_DIR / "model_comparison.csv")
    pairs = _read_csv(C.ANALYSIS_DIR / "measured_pair_results.csv")
    oracle = _read_csv(C.ANALYSIS_DIR / "oracle_localization_results.csv")
    cases = _select_cases(pairs)
    endpoints = _load_selected_endpoints({str(cases[k]["pair_id"]) for k in ("success", "failure", "inconsistent")})
    info = {
        "forest": _forest(comp, C.FIGURES_DIR / "localisation_and_oracle_forest.png"),
        "scatter": _scatter(pairs, C.FIGURES_DIR / "pair_prediction_scatter.png"),
        "failure_modes": _failure_modes(pairs, C.FIGURES_DIR / "failure_modes.png"),
        "oracle_hist": _oracle_hist(oracle, C.FIGURES_DIR / "oracle_capture_distribution.png"),
        "molecules": _rdkit_images(cases, endpoints, C.FIGURES_DIR / "molecule_cases.png"),
    }
    C.write_json(C.ANALYSIS_DIR / "figure_cases.json", info)
    print(json.dumps(info, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
