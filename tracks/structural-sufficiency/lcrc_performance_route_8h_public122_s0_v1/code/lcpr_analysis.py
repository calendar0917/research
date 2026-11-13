"""Analysis: metrics, paired bootstrap, error ledger, mechanism panels, figures.

Everything here is read-only with respect to training artifacts.  Memory is kept
small on purpose: predictions are 420-row vectors, covariates come from the
label-free structure cache of one dataset at a time, and no Large cache
(ogbg-molhiv, 347 MB) is ever loaded locally.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import lcpr_common as C

# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2)))


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((pred - y) ** 2) / denom) if denom > 0 else float("nan")


def _auc(pred: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pred))


def _bce(pred: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(pred, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def compute_metrics(pred: np.ndarray, y: np.ndarray, task: str) -> dict[str, float]:
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    out = {"n": int(len(y))}
    if task == "regression":
        out.update({"rmse": _rmse(pred, y), "mae": _mae(pred, y), "r2": _r2(pred, y),
                    "bias": float(np.mean(pred - y))})
    else:
        out.update({"roc_auc": _auc(pred, y), "bce": _bce(pred, y),
                    "accuracy": float(np.mean((pred > 0.5) == (y > 0.5)))})
    return out


def primary_key(task: str) -> str:
    return "rmse" if task == "regression" else "roc_auc"


def gain(base: Mapping[str, float], arm: Mapping[str, float], task: str) -> float:
    """Positive = improvement (PROTOCOL §6)."""
    if task == "regression":
        return float((base["rmse"] - arm["rmse"]) / base["rmse"])
    return float(arm["roc_auc"] - base["roc_auc"])


# --------------------------------------------------------------------------- #
# paired bootstrap
# --------------------------------------------------------------------------- #


def _row_hash(smiles: Sequence[str]) -> np.ndarray:
    import hashlib

    return np.array([int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) for s in smiles],
                    dtype=np.int64)


def scaffold_clusters(smiles: Sequence[str], *, max_bytes: int = 400_000) -> np.ndarray:
    """Murcko scaffold clusters; empty scaffold falls back to canonical SMILES."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    ids: list[str] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ids.append(f"mol:{smi}")
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:  # pragma: no cover - defensive
            scaf = ""
        ids.append(scaf if scaf else f"mol:{Chem.MolToSmiles(mol)}")
    order: dict[str, int] = {}
    for key in ids:
        order.setdefault(key, len(order))
    return np.array([order[k] for k in ids], dtype=np.int64)


def paired_bootstrap(
    *,
    control: np.ndarray,
    candidate: np.ndarray,
    kind: str,
    index: np.ndarray | None = None,
    y: np.ndarray | None = None,
    replicates: int = 2000,
    seed: int = 20261113,
) -> dict[str, Any]:
    """Cluster/row bootstrap of a paired gain.

    ``kind`` is ``"rmse"`` or ``"auc"``.  ``index`` holds cluster ids for
    regression (scaffold clusters) and stratum ids for binary tasks
    (0 = negative row, 1 = positive row), i.e. the positive/negative row
    bootstrap.  The point estimate is the plug-in gain on all rows.
    """
    control = np.asarray(control, float)
    candidate = np.asarray(candidate, float)
    n = len(control)
    if index is None:
        index = np.arange(n, dtype=np.int64)
    index = np.asarray(index, dtype=np.int64)

    def _gain(c: np.ndarray, a: np.ndarray, rows: np.ndarray) -> float:
        if kind == "rmse":
            return _rmse(c[rows], y[rows]) - _rmse(a[rows], y[rows])
        if kind == "auc":
            if len(np.unique(y[rows])) < 2:
                return float("nan")
            return _auc(a[rows], y[rows]) - _auc(c[rows], y[rows])
        raise ValueError(kind)

    all_rows = np.arange(n)
    point = _gain(control, candidate, all_rows)
    rng = np.random.default_rng(int(seed))
    groups = np.unique(index)
    members = [np.where(index == g)[0] for g in groups]
    draws = np.empty(int(replicates), dtype=float)
    for b in range(int(replicates)):
        pick = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[i] for i in pick])
        draws[b] = _gain(control, candidate, rows)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return {"delta": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_two_sided": float("nan"), "n_valid_replicates": int(len(draws)),
                "positive_means": "improvement"}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    frac = float(np.mean(draws <= 0.0))
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(2 * min(frac, 1 - frac)),
        "n_valid_replicates": int(len(draws)),
        "positive_means": "improvement",
        "n_clusters": int(len(groups)),
    }


# --------------------------------------------------------------------------- #
# run collection
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def collect_runs(root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(root or C.RUNS)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/train.json")):
        payload = _load_json(path)
        if payload.get("status") != "completed":
            continue
        meta = payload.get("model", {})
        rec = payload.get("recipe", {})
        row = {
            "run": path.parent.name,
            "candidate": payload["candidate"],
            "dataset": payload["dataset"],
            "task": payload["task"],
            "recipe": rec.get("name"),
            "epochs": rec.get("epochs"),
            "peak_lr": rec.get("peak_lr"),
            "weight_decay": rec.get("weight_decay"),
            "parameters": meta.get("parameters"),
            "trainable_parameters": meta.get("trainable_parameters"),
            "best_epoch": payload.get("best_epoch"),
            "valid": payload["metrics"]["best"]["valid"],
            "test": payload["metrics"]["best"]["test"],
            "train": payload["metrics"]["best"]["train"],
            "wall_seconds": (payload.get("timing") or {}).get("process_wall_seconds"),
            "gpu_hours": (payload.get("timing") or {}).get("gpu_hours_charged"),
            "device": (payload.get("environment") or {}).get("device")
                      or (payload.get("gpu") or {}).get("device"),
            "commit": (payload.get("provenance") or {}).get("commit"),
            "seed": payload.get("seed", C.TRAIN_SEED),
            "mean_epoch_seconds": (payload.get("timing") or {}).get("mean_epoch_seconds"),
            "train_wall_seconds": (payload.get("timing") or {}).get("train_wall_seconds"),
            "predictions": str(path.parent / "predictions_best.npz"),
            "smoke": bool(payload.get("smoke", False)),
        }
        rows.append(row)
    return rows


def run_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(r["candidate"], r["dataset"]): r for r in rows if not r["smoke"]}


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k]) for k in z.files}


# --------------------------------------------------------------------------- #
# covariates (fixed, never re-thresholded)
# --------------------------------------------------------------------------- #


_SMARTS = {
    "fr_quatN": "[NX4+;+1]",
    "fr_COO": "[CX3](=O)[OX1-]",
    "fr_phos_acid": "[PX4](=O)([OX1-])",
}


def covariates(dataset: str) -> dict[str, Any]:
    """Fixed structural covariates for every row of ``dataset`` (all splits)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    cache = C.DD.PREP_DATA.load_cache(dataset) if hasattr(C.DD.PREP_DATA, "load_cache") else None
    import lcpr_data as D

    splits = {s: D.load_cache(dataset, s) for s in C.SPLITS}
    smiles = {s: list(splits[s]["smiles"]) for s in C.SPLITS}
    natoms = {s: np.asarray(splits[s]["atom_ptr"]) for s in C.SPLITS}
    counts = {s: np.diff(natoms[s]) for s in C.SPLITS}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {s: [gen.GetFingerprint(Chem.MolFromSmiles(m)) for m in smiles[s]] for s in C.SPLITS}

    # fixed Train reference subsample by row hash (<=4096)
    tr = smiles["train"]
    h = _row_hash(tr)
    order = np.argsort(h, kind="stable")
    ref_rows = order[: min(4096, len(order))]
    ref_fps = [fp["train"][i] for i in ref_rows]

    out: dict[str, Any] = {}
    for s in C.SPLITS:
        mx = np.zeros(len(smiles[s]), dtype=np.float64)
        for i, f in enumerate(fp[s]):
            if s == "train":
                peers = [ref_fps[j] for j, r in enumerate(ref_rows) if r != i]
            else:
                peers = ref_fps
            sims = DataStructs.BulkTanimotoSimilarity(f, peers) if peers else [0.0]
            mx[i] = float(max(sims))
        patt = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}
        hits = {}
        for k, p in patt.items():
            hits[k] = np.array([
                len(Chem.MolFromSmiles(m).GetSubstructMatches(p)) if Chem.MolFromSmiles(m) else 0
                for m in smiles[s]
            ], dtype=np.float64)
        formal = np.array([sum(a.GetFormalCharge() for a in Chem.MolFromSmiles(m).GetAtoms())
                           if Chem.MolFromSmiles(m) else 0.0 for m in smiles[s]], dtype=np.float64)
        out[s] = {"smiles": np.array(smiles[s], dtype=object), "n_atoms": counts[s].astype(float),
                  "formal_charge": formal, "max_tanimoto_to_train": mx, **hits}
    return out


def s2_support_by_molecule(dataset: str) -> dict[str, np.ndarray]:
    """Per-molecule mean/min Train support ``c`` over its S2 (root,bt2) keys."""
    import lcpr_data as D
    import torch

    support = D.load_support(dataset)
    keys = support["keys"].astype(np.int64)
    counts = support["counts"].astype(np.float64)
    n_codes = int(C.N_PATH_CODES) * int(C.N_NBR_CODES)
    flat_keys = keys[:, 0] * n_codes + keys[:, 1]
    out: dict[str, np.ndarray] = {}
    for s in C.SPLITS:
        cache = D.load_cache(dataset, s)
        pcat = torch.as_tensor(cache["path_cat"]).to(torch.int64)
        pend = torch.as_tensor(cache["path_end63"]).to(torch.int64)
        patom = torch.as_tensor(cache["path_atom"]).to(torch.int64)
        n = int(cache["n_molecules"])
        ptr = torch.as_tensor(cache["atom_ptr"]).to(torch.int64)
        flat = (pcat * n_codes + pend).numpy()
        pos = np.clip(np.searchsorted(flat_keys, flat), 0, max(len(counts) - 1, 0))
        ok = flat_keys[pos] == flat
        val = np.zeros(len(flat), dtype=np.float64)
        val[ok] = counts[pos[ok]]
        # reduce per molecule
        mol_of_path = np.searchsorted(ptr.numpy(), patom.numpy(), side="right") - 1
        mean = np.full(n, np.nan)
        mn = np.full(n, np.nan)
        for m in range(n):
            sel = mol_of_path == m
            if sel.any():
                mean[m] = val[sel].mean()
                mn[m] = val[sel].min()
        out[s] = np.stack([mean, mn], axis=1)
    return out


# --------------------------------------------------------------------------- #
# error ledger
# --------------------------------------------------------------------------- #


def error_ledger(dataset: str, base_run: Mapping[str, Any],
                 arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-row ledger on the aligned best-Valid rows (Test + Valid + Train)."""
    base_pred = load_predictions(base_run["predictions"])
    cov = covariates(dataset)
    supp = s2_support_by_molecule(dataset)
    rows: list[dict[str, Any]] = []
    for split in C.SPLITS:
        y = np.asarray(base_pred[f"{split}_y"], dtype=float)
        pb = np.asarray(base_pred[f"{split}_pred"], dtype=float)
        task = base_run["task"]
        eb = pb - y
        for arm in arms:
            pa = np.asarray(load_predictions(arm["predictions"])[f"{split}_pred"], dtype=float)
            ea = pa - y
            for i in range(len(y)):
                rec = {
                    "dataset": dataset, "split": split, "row": i,
                    "smiles": str(cov[split]["smiles"][i]),
                    "y": float(y[i]), "pred_base": float(pb[i]), "pred_arm": float(pa[i]),
                    "error_base": float(eb[i]), "error_arm": float(ea[i]),
                    "abs_error_base": float(abs(eb[i])), "abs_error_arm": float(abs(ea[i])),
                    "delta_abs_error": float(abs(ea[i]) - abs(eb[i])),
                    "candidate": arm["candidate"],
                    "n_atoms": float(cov[split]["n_atoms"][i]),
                    "formal_charge": float(cov[split]["formal_charge"][i]),
                    "fr_quatN": float(cov[split]["fr_quatN"][i]),
                    "fr_COO": float(cov[split]["fr_COO"][i]),
                    "fr_phos_acid": float(cov[split]["fr_phos_acid"][i]),
                    "max_tanimoto_to_train": float(cov[split]["max_tanimoto_to_train"][i]),
                    "s2_support_mean": float(supp[split][i, 0]),
                    "s2_support_min": float(supp[split][i, 1]),
                }
                if task != "regression":
                    rec["bce_base"] = float(-(y[i] * math.log(min(max(pb[i], 1e-7), 1 - 1e-7))
                                              + (1 - y[i]) * math.log(min(max(1 - pb[i], 1e-7), 1 - 1e-7))))
                    rec["bce_arm"] = float(-(y[i] * math.log(min(max(pa[i], 1e-7), 1 - 1e-7))
                                             + (1 - y[i]) * math.log(min(max(1 - pa[i], 1e-7), 1 - 1e-7))))
                rows.append(rec)
    return rows


def tail_view(base_pred: Mapping[str, np.ndarray], split: str,
              fractions: Sequence[float] = C.TAIL_FRACTIONS) -> dict[str, Any]:
    """Row sets defined **only** by the frozen base's absolute error."""
    y = np.asarray(base_pred[f"{split}_y"], dtype=float)
    pb = np.asarray(base_pred[f"{split}_pred"], dtype=float)
    ae = np.abs(pb - y)
    n = len(ae)
    out: dict[str, Any] = {"n": int(n), "top_row": int(np.argmax(ae)),
                           "top_row_abs_error": float(ae.max())}
    for frac in fractions:
        k = max(1, int(round(frac * n)))
        out[f"exclude_top_{frac:g}"] = np.argsort(ae)[:k].tolist()
    return out


# --------------------------------------------------------------------------- #
# selection trace
# --------------------------------------------------------------------------- #


def select_recipe(rows: Iterable[Mapping[str, Any]], dataset: str = "Lipophilicity") -> dict[str, Any]:
    idx = run_index(rows)
    table = []
    for name in ("R0", "R1", "R2", "R3"):
        rec = idx.get((f"C1_{name}", dataset))
        if rec is None:
            continue
        table.append({"recipe": name, "valid_rmse": rec["valid"]["rmse"],
                      "test_rmse": rec["test"]["rmse"], "valid_mae": rec["valid"]["mae"],
                      "test_mae": rec["test"]["mae"]})
    if not table:
        return {"status": "pending"}
    ranked = sorted(table, key=lambda r: (r["valid_rmse"], "R0 R1 R2 R3".split().index(r["recipe"])))
    return {"status": "frozen", "table": table, "best": ranked[0]["recipe"],
            "rule": "best Valid RMSE on Lipophilicity; ties R0,R1,R2,R3",
            "test_rmse_of_best": ranked[0]["test_rmse"]}


def family_summary(rows: Iterable[Mapping[str, Any]], dataset: str,
                   base_candidate: str) -> dict[str, Any]:
    idx = run_index(rows)
    base = idx.get((base_candidate, dataset))
    if base is None:
        return {"status": "pending", "base": base_candidate}
    task = base["task"]
    out: dict[str, Any] = {"base": base_candidate, "task": task, "arms": {}, "families": {}}
    for (cand, ds), rec in idx.items():
        if ds != dataset:
            continue
        if cand == base_candidate:
            continue
        g_valid = gain(base["valid"], rec["valid"], task)
        g_test = gain(base["test"], rec["test"], task)
        entry = {
            "valid": rec["valid"], "test": rec["test"],
            "gain_valid": g_valid, "gain_test": g_test,
            "balanced": min(g_valid, g_test),
            "parameters": rec["parameters"],
            "family": C.CANDIDATES.get(cand, {}).get("family", "?"),
            "recipe": rec["recipe"],
            "epochs": rec["epochs"],
            "gpu_hours": rec["gpu_hours"],
        }
        if task == "regression":
            entry["mae_gain_valid"] = float((base["valid"]["mae"] - rec["valid"]["mae"])
                                            / base["valid"]["mae"])
            entry["mae_gain_test"] = float((base["test"]["mae"] - rec["test"]["mae"])
                                           / base["test"]["mae"])
            entry["clear_signal"] = bool(
                g_test >= C.CLEAR_SIGNAL["test_rel_rmse_gain_min"]
                and g_valid >= C.CLEAR_SIGNAL["valid_rel_rmse_gain_min"]
                and entry["mae_gain_valid"] >= -C.CLEAR_SIGNAL["mae_degradation_max"]
                and entry["mae_gain_test"] >= -C.CLEAR_SIGNAL["mae_degradation_max"]
            )
        else:
            entry["clear_signal"] = bool(g_test >= C.CLEAR_SIGNAL["bace_auc_gain_min"] if False else g_test >= 0.005)
        out["arms"][cand] = entry
        out["families"].setdefault(entry["family"], []).append(cand)
    for fam, cands in out["families"].items():
        ranked = sorted(cands, key=lambda c: (-out["arms"][c]["balanced"],
                                              -0.5 * (out["arms"][c].get("mae_gain_valid", 0.0)
                                                      + out["arms"][c].get("mae_gain_test", 0.0)),
                                              out["arms"][c]["parameters"] or 0, c))
        out["families"][fam] = {"candidates": cands, "winner": ranked[0]}
    clear = [f for f, v in out["families"].items()
             if out["arms"][v["winner"]].get("clear_signal")]
    out["clear_families"] = sorted(clear, key=lambda f: (-out["arms"][out["families"][f]["winner"]]["balanced"], f))
    out["promoted"] = out["clear_families"][: C.MAX_TRANSFER_FAMILIES]
    if not out["promoted"]:
        best = max(out["arms"], key=lambda c: (out["arms"][c]["balanced"],
                                               -out["arms"][c]["parameters"] or 0))
        out["promoted"] = out["arms"][best]["family"]
        out["promoted_role"] = "diagnostic_transfer (no clear signal in any family)"
        out["promoted"] = [out["promoted"]]
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_performance_cost(summary: Mapping[str, Any], path: Path, reference: Mapping[str, float]) -> None:
    plt = _mpl()
    arms = summary.get("arms", {})
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=170)
    names, gains, costs = [], [], []
    for cand, e in sorted(arms.items(), key=lambda kv: -kv[1]["balanced"]):
        names.append(cand)
        gains.append(e["balanced"] * 100.0)
        cost = e.get("gpu_hours")
        costs.append(float(cost) if cost else 0.05)
    ax.scatter(costs, gains, s=28, c="#2b6cb0", zorder=3)
    for x, y, n in zip(costs, gains, names, strict=True):
        ax.annotate(n, (x, y), fontsize=5.6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    if "xgboost_valid" in reference:
        ax.axhline(reference["xgboost_valid"] * 100, color="#c53030", lw=0.9, ls="--")
        ax.annotate("ECFP+XGBoost (Valid)", (max(costs) * 0.98, reference["xgboost_valid"] * 100),
                    ha="right", va="bottom", fontsize=6, color="#c53030")
    if "dmpnn_valid" in reference:
        ax.axhline(reference["dmpnn_valid"] * 100, color="#2f855a", lw=0.9, ls="--")
        ax.annotate("D-MPNN (Valid)", (max(costs) * 0.98, reference["dmpnn_valid"] * 100),
                    ha="right", va="top", fontsize=6, color="#2f855a")
    ax.set_xlabel("charged GPU hours (wall x GPUs)")
    ax.set_ylabel("balanced gain vs C1@R*  [% relative RMSE]")
    ax.set_title("Lipophilicity random122 — performance vs cost (seed 0, one recipe each)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_error_distribution(ledger: Sequence[Mapping[str, Any]], path: Path,
                             candidates: Sequence[str]) -> None:
    plt = _mpl()
    rows = [r for r in ledger if r["split"] == "test" and r["candidate"] in candidates]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=170)
    data, labels = [], []
    for cand in candidates:
        sub = [r["delta_abs_error"] for r in rows if r["candidate"] == cand]
        if sub:
            data.append(sub)
            labels.append(cand)
    ax.boxplot(data, labels=labels, showfliers=False, widths=0.55)
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_ylabel("per-row |error| change (arm − base)")
    ax.set_title("Test-row error redistribution (negative = better)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_mechanism(mech: Mapping[str, Any], path: Path) -> None:
    plt = _mpl()
    s_block = mech.get("s_panel", {})
    points = s_block.get("rows", [])
    if not points:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4), dpi=170)
    sup = [p["s2_support_mean"] for p in points]
    dq = [p["delta_q_norm"] for p in points]
    ax[0].scatter(sup, dq, s=6, alpha=0.6, c="#2b6cb0")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("mean Train support of the row's S2 keys")
    ax[0].set_ylabel("|Δq| under α = 0 → 1")
    ax[0].set_title("J-block contraction sensitivity vs support")
    ds = [p["delta_se"] for p in points]
    ax[1].scatter(dq, ds, s=6, alpha=0.6, c="#c53030")
    ax[1].axhline(0, color="#999", lw=0.8, ls=":")
    ax[1].set_xlabel("|Δq| under α = 0 → 1")
    ax[1].set_ylabel("Δ squared error (α = 0 par → full)")
    ax[1].set_title("Does the mechanism move real error?")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_route_matrix(transfer: Mapping[str, Any], path: Path) -> None:
    plt = _mpl()
    fams = transfer.get("families", {})
    tasks = transfer.get("tasks", [])
    if not fams or not tasks:
        return
    mat = np.full((len(fams), len(tasks)), np.nan)
    for i, fam in enumerate(fams):
        for j, task in enumerate(tasks):
            v = fams[fam].get(task, {}).get("gain")
            if v is not None:
                mat[i, j] = float(v) * 100.0
    fig, ax = plt.subplots(figsize=(5.2, 2.9), dpi=170)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else -1,
                   vmax=np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1)
    ax.set_xticks(range(len(tasks)), tasks, fontsize=7)
    ax.set_yticks(range(len(fams)), fams, fontsize=7)
    for i in range(len(fams)):
        for j in range(len(tasks)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Route × task gain [% RMSE / AUC×100]")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
