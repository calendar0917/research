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
    for path in sorted(root.rglob("train.json")):
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


def load_predictions_raw(dataset: str, path: str | Path) -> dict[str, np.ndarray]:
    """Predictions decoded to the published target units (regression only)."""
    import lcpr_data as D

    z = load_predictions(path)
    out: dict[str, np.ndarray] = {}
    norm = None
    if D.dataset_meta(dataset)["task"] == "regression":
        norm = D.targets(dataset, include_test=False).get("target_norm")
    for split in C.SPLITS:
        y = np.asarray(z[f"{split}_y"], dtype=float)
        p = np.asarray(z[f"{split}_pred_std"], dtype=float)
        if norm is not None:
            p = p * float(norm["scale"]) + float(norm["mean"])
        out[f"{split}_y"] = y
        out[f"{split}_pred"] = p
        if f"{split}_rows" in z:
            out[f"{split}_rows"] = np.asarray(z[f"{split}_rows"])
    return out


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
    base_pred = load_predictions_raw(dataset, base_run["predictions"])
    cov = covariates(dataset)
    supp = s2_support_by_molecule(dataset)
    rows: list[dict[str, Any]] = []
    for split in C.SPLITS:
        y = np.asarray(base_pred[f"{split}_y"], dtype=float)
        pb = np.asarray(base_pred[f"{split}_pred"], dtype=float)
        task = base_run["task"]
        eb = pb - y
        for arm in arms:
            pa = np.asarray(load_predictions_raw(dataset, arm["predictions"])[f"{split}_pred"],
                            dtype=float)
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


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

XFGB_REFERENCE = {
    # frozen external reference, ECFP4+RDKit+XGBoost, Train-only fitting
    # source: lcrc_molnet_benchmark_pilot_v2/analysis/baseline
    "source": "lcrc_molnet_benchmark_pilot_v2/analysis/baseline (ECFP4 + RDKit descriptors + XGBoost, frozen config)",
    "Lipophilicity": {"valid_rmse": 0.6532, "test_rmse": 0.6477, "valid_r2": 0.68, "test_r2": 0.71},
    "ESOL": {"valid_rmse": 0.6425},
    "BACE": {"valid_roc_auc": 0.7065},
    "FreeSolv": {"valid_rmse": 1.1331},
}


def split_labels(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Published labels + SMILES for one split (the structure cache is label-free)."""
    import lcpr_data as D

    y = np.asarray(D.targets(dataset, include_test=True)["y"][split], dtype=float).reshape(-1)
    cache = D.load_cache(dataset, split)
    smiles = np.array(list(cache["smiles"]), dtype=object)
    if len(y) != len(smiles):
        raise ValueError(f"{dataset}/{split}: label and structure row counts differ")
    return y, smiles


_split_labels = split_labels


def _cluster_index(dataset: str, split: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    y, smiles = _split_labels(dataset, split)
    if task == "regression":
        return y, scaffold_clusters(list(smiles))
    return y, np.asarray(y > 0.5, dtype=np.int64)


def run_core() -> dict[str, Any]:
    rows = [r for r in collect_runs() if not r.get("smoke")]
    C.write_json(C.ANALYSIS / "run_table.json", {"created_utc": C.utcnow(), "runs": rows})
    cols = ["run", "candidate", "dataset", "task", "recipe", "epochs", "peak_lr", "weight_decay",
            "parameters", "best_epoch", "wall_seconds", "mean_epoch_seconds", "commit", "seed"]
    metric_cols: list[str] = []
    for split in C.SPLITS:
        for key in ("rmse", "mae", "r2", "roc_auc", "bce"):
            if any(key in (r.get(split) or {}) for r in rows):
                metric_cols.append(f"{split}_{key}")
    lines = [",".join(cols + metric_cols + ["smoke"])]
    for r in rows:
        vals = [str(r.get(c, "")) for c in cols]
        for mc in metric_cols:
            split, key = mc.split("_", 1)
            v = (r.get(split) or {}).get(key)
            vals.append("" if v is None else f"{float(v):.6f}")
        vals.append(str(r.get("smoke")))
        lines.append(",".join(vals))
    (C.ANALYSIS / "run_table.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    trace: dict[str, Any] = {"created_utc": C.utcnow(), "recipes": select_recipe(rows)}
    winner = trace["recipes"].get("best")
    trace["lipophilicity_base"] = f"C1_{winner}" if winner else None
    if winner:
        base_candidate = f"C1_{winner}"
        fam = family_summary(rows, "Lipophilicity", base_candidate)
        C.write_json(C.ANALYSIS / "family_summary.json", fam)
        trace["families"] = {k: v["winner"] for k, v in fam.get("families", {}).items()}
        trace["clear_families"] = fam.get("clear_families")
        trace["promoted"] = fam.get("promoted")
        trace["promoted_role"] = fam.get("promoted_role", "clear signal in >=1 family")

        # paired bootstrap for every Lipophilicity arm against the base
        idx = run_index(rows)
        base = idx[(base_candidate, "Lipophilicity")]
        comparisons: dict[str, Any] = {"base": base_candidate, "task": "regression",
                                       "replicates": C.BOOTSTRAP_REPLICATES if hasattr(C, "BOOTSTRAP_REPLICATES") else 2000,
                                       "seed": C.BOOTSTRAP_SEED, "splits": {}}
        for split in ("valid", "test"):
            y, index = _cluster_index("Lipophilicity", split, "regression")
            control = np.asarray(load_predictions_raw("Lipophilicity", base["predictions"])[f"{split}_pred"], float)
            for (cand, ds), rec in sorted(idx.items()):
                if ds != "Lipophilicity" or cand == base_candidate:
                    continue
                cand_pred = np.asarray(load_predictions_raw("Lipophilicity", rec["predictions"])[f"{split}_pred"], float)
                res = paired_bootstrap(control=control, candidate=cand_pred, kind="rmse",
                                       index=index, y=y, replicates=2000, seed=C.BOOTSTRAP_SEED)
                comparisons["splits"].setdefault(split, {})[cand] = res
        C.write_json(C.ANALYSIS / "paired_comparisons.json", comparisons)
    C.write_json(C.ANALYSIS / "selection_trace.json", trace)
    return trace


def run_ledger(base_candidate: str | None = None) -> dict[str, Any]:
    rows = collect_runs()
    idx = run_index(rows)
    if base_candidate is None:
        trace = json.loads((C.ANALYSIS / "selection_trace.json").read_text())
        base_candidate = trace.get("lipophilicity_base")
    if base_candidate is None or (base_candidate, "Lipophilicity") not in idx:
        return {"status": "pending"}
    base = idx[(base_candidate, "Lipophilicity")]
    arms = [r for (c, d), r in sorted(idx.items())
            if d == "Lipophilicity" and c != base_candidate]
    ledger = error_ledger("Lipophilicity", base, arms)
    import csv as _csv

    path = C.ANALYSIS / "error_ledger.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)

    pred = load_predictions_raw("Lipophilicity", base["predictions"])
    tails = {s: tail_view(pred, s) for s in C.SPLITS}
    summary: dict[str, Any] = {"base": base_candidate, "tails": {}, "tail_metrics": {}}
    for s, tv in tails.items():
        summary["tails"][s] = {"n": tv["n"], "top_row": tv["top_row"],
                               "top_row_abs_error": tv["top_row_abs_error"]}
        y = np.asarray(pred[f"{s}_y"], float)
        pb = np.asarray(pred[f"{s}_pred"], float)
        entry: dict[str, Any] = {}
        for frac in C.TAIL_FRACTIONS:
            drop = np.asarray(tv[f"exclude_top_{frac:g}"], dtype=np.int64)
            keep = np.setdiff1d(np.arange(len(y)), drop)
            entry[f"exclude_top_{frac:g}"] = {
                "base_rmse": _rmse(pb[keep], y[keep]),
                "n_kept": int(len(keep)),
            }
            for cand in summary.setdefault("_arms", []):
                pass
        summary["tail_metrics"][s] = entry
    # per-arm tail metrics (same frozen row sets as the base)
    arm_tail: dict[str, Any] = {}
    for arm in arms:
        ap = np.asarray(load_predictions_raw("Lipophilicity", arm["predictions"])["test_pred"], float)
        y = np.asarray(pred["test_y"], float)
        rec: dict[str, Any] = {"full_rmse": _rmse(ap, y), "base_full_rmse": _rmse(
            np.asarray(pred["test_pred"], float), y)}
        for frac in C.TAIL_FRACTIONS:
            drop = np.asarray(tails["test"][f"exclude_top_{frac:g}"], dtype=np.int64)
            keep = np.setdiff1d(np.arange(len(y)), drop)
            rec[f"exclude_top_{frac:g}"] = {"arm_rmse": _rmse(ap[keep], y[keep]),
                                            "n_kept": int(len(keep))}
        arm_tail[arm["candidate"]] = rec
    summary["arm_tail_test"] = arm_tail
    summary.pop("_arms", None)
    C.write_json(C.ANALYSIS / "tail_summary.json", summary)
    return {"rows": len(ledger), "arms": len(arms)}


def run_figures() -> dict[str, Any]:
    rows = collect_runs()
    trace_path = C.ANALYSIS / "selection_trace.json"
    if not trace_path.exists():
        return {"status": "pending"}
    trace = json.loads(trace_path.read_text())
    base_candidate = trace.get("lipophilicity_base")
    fam_path = C.ANALYSIS / "family_summary.json"
    fam = json.loads(fam_path.read_text()) if fam_path.exists() else {}
    ref = {"xgboost_valid": XFGB_REFERENCE["Lipophilicity"]["valid_rmse"]}
    idx = run_index(rows)
    dmp = [r for (c, d), r in idx.items() if d == "Lipophilicity" and str(c).startswith("DMPNN")]
    if dmp:
        best = min(dmp, key=lambda r: r["valid"]["rmse"])
        ref["dmpnn_valid"] = best["valid"]["rmse"]
    fig_dir = C.ANALYSIS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_performance_cost(fam, fig_dir / "performance_cost.png", ref)
    ledger_path = C.ANALYSIS / "error_ledger.csv"
    if ledger_path.exists():
        import csv as _csv

        with open(ledger_path, encoding="utf-8") as fh:
            ledger = list(_csv.DictReader(fh))
        for rec in ledger:
            rec["split"] = rec["split"]
            rec["delta_abs_error"] = float(rec["delta_abs_error"])
        cands = [c for c in (fam.get("promoted") or [])]
        if not cands:
            cands = sorted({r["candidate"] for r in ledger if r["split"] == "test"})[:3]
        figure_error_distribution(ledger, fig_dir / "error_distribution.png", cands)
    mech_path = C.ANALYSIS / "mechanism_summary.json"
    if mech_path.exists():
        figure_mechanism(json.loads(mech_path.read_text()), fig_dir / "mechanism_vs_gain.png")
    return {"figures": sorted(p.name for p in fig_dir.glob("*.png"))}


def run(stage: str = "all") -> dict[str, Any]:
    out: dict[str, Any] = {"created_utc": C.utcnow(), "stage": stage}
    if stage in ("core", "all"):
        out["core"] = run_core()
    if stage in ("ledger", "all"):
        out["ledger"] = run_ledger()
    if stage in ("figures", "all"):
        out["figures"] = run_figures()
    C.log(f"[analyze] {stage}: {str(out)[:400]}")
    return out
