"""Core analysis: main tables, paired comparisons, DiD, scaffold bootstrap,

mechanical route labels.  CPU post-processing of pulled per-run outputs under
``analysis/results/<run_name>/``; never trains and never selects a checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import se_common as C


def _SPLITS():
    return ("train", "valid", "test")


RESULTS = C.ANALYSIS / "results"


def run_dir(run_name: str) -> Path:
    return RESULTS / run_name


def has_run(run_name: str) -> bool:
    return (run_dir(run_name) / "metrics.json").is_file()


def load_run(run_name: str) -> dict[str, Any]:
    return C.read_json(run_dir(run_name) / "metrics.json")


def load_curve(run_name: str) -> list[dict[str, Any]]:
    import csv

    with open(run_dir(run_name) / "curve.csv", newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def load_pred(run_name: str, split: str, tag: str = "best") -> dict[str, np.ndarray]:
    with np.load(run_dir(run_name) / f"pred_{split}_{tag}.npz", allow_pickle=False) as h:
        return {k: np.asarray(h[k]) for k in h.files}


def load_grid(run_name: str) -> dict[str, Any]:
    return C.read_json(run_dir(run_name) / "grid.json")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def rmse(pred, y, w=None):
    err = np.asarray(pred, float) - np.asarray(y, float)
    if w is None:
        return float(np.sqrt(np.mean(err ** 2)))
    w = np.asarray(w, float)
    return float(np.sqrt(np.sum(w * err ** 2) / np.sum(w)))


def mae(pred, y, w=None):
    err = np.abs(np.asarray(pred, float) - np.asarray(y, float))
    if w is None:
        return float(np.mean(err))
    w = np.asarray(w, float)
    return float(np.sum(w * err) / np.sum(w))


def rel(control: float, candidate: float) -> float:
    return float((control - candidate) / control)


def point_metrics(run_name: str, split: str, tag: str) -> dict[str, float]:
    d = load_pred(run_name, split, tag)
    raw, y = d["raw"], d["y_true"]
    return {"rmse": rmse(raw, y), "mae": mae(raw, y),
            "n": int(y.size), "bias": float(np.mean(raw - y))}


# ---------------------------------------------------------------------------
# paired comparison of a family against its matched control
# ---------------------------------------------------------------------------


def compare(cand_run: str, ctrl_run: str, tag: str = "best") -> dict[str, Any]:
    out: dict[str, Any] = {"candidate": cand_run, "control": ctrl_run, "tag": tag}
    for split in _SPLITS():
        pm_c = point_metrics(ctrl_run, split, tag)
        pm_d = point_metrics(cand_run, split, tag)
        entry = {
            "control_rmse": pm_c["rmse"], "candidate_rmse": pm_d["rmse"],
            "rel_improvement": rel(pm_c["rmse"], pm_d["rmse"]),
            "control_mae": pm_c["mae"], "candidate_mae": pm_d["mae"],
            "rel_mae_improvement": rel(pm_c["mae"], pm_d["mae"]),
        }
        if split in ("valid", "test"):
            dc = load_pred(ctrl_run, split, tag)
            dd = load_pred(cand_run, split, tag)
            err_c = np.abs(dc["raw"] - dc["y_true"])
            err_d = np.abs(dd["raw"] - dd["y_true"])
            entry["win_rate"] = float((err_c - err_d > 0).mean())
            for frac in (0.01, 0.05):
                keep = _tail_keep(err_c, frac)
                entry[f"trim_{frac}"] = {
                    "rows_removed": int((~keep).sum()),
                    "control_rmse": rmse(dc["raw"][keep], dc["y_true"][keep]),
                    "candidate_rmse": rmse(dd["raw"][keep], dd["y_true"][keep]),
                    "rel_improvement": rel(rmse(dc["raw"][keep], dc["y_true"][keep]),
                                           rmse(dd["raw"][keep], dd["y_true"][keep])),
                }
        out[split] = entry
    return out


def _tail_keep(err, frac):
    n = err.size
    k = int(math.floor(frac * n))
    keep = np.ones(n, dtype=bool)
    if k > 0:
        order = np.argsort(-err, kind="stable")
        keep[order[:k]] = False
    return keep


def did(sem: Mapping[str, Any], perm: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"note": "DiD_semantics = Delta_sem - Delta_perm; positive => real semantics help"}
    for split in ("valid", "test"):
        ds = sem[split]["rel_improvement"]
        dp = perm[split]["rel_improvement"]
        out[split] = {"delta_sem": ds, "delta_perm": dp, "did": ds - dp,
                      "delta_sem_abs": sem[split]["control_rmse"] - sem[split]["candidate_rmse"],
                      "delta_perm_abs": perm[split]["control_rmse"] - perm[split]["candidate_rmse"]}
    return out


# ---------------------------------------------------------------------------
# scaffold bootstrap (paired, shared Exp(1) weights)
# ---------------------------------------------------------------------------


def scaffold_clusters(smiles: Sequence[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    keys = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            keys.append(str(smi))
            continue
        try:
            scf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scf = ""
        keys.append(scf if scf else Chem.MolToSmiles(mol, canonical=True))
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.asarray([uniq[k] for k in keys], dtype=np.int64)


def _model_smiles(split: str) -> list[str]:
    import gzip

    import lp_source as LPS

    root = C.HOME / "data/download/extracted/lipophilicity"
    mol = LPS.read_mol_csv(root)
    return list(mol["smiles"])


def bootstrap(runs: Mapping[str, str], *, replicates: int = 500, seed: int = 20261114) -> dict[str, Any]:
    """Paired scaffold-cluster bootstrap with shared Exp(1) weights.

    ``runs`` maps a model key -> run_name.  Recomputed jointly so DiD intervals
    share the same weights across all four models.
    """
    import lp_source as LPS

    rows = LPS.load_split()["rows"]
    smiles = _model_smiles("valid")
    rng = np.random.default_rng(int(seed))
    out: dict[str, Any] = {"replicates": replicates, "seed": seed, "splits": {}, "paired": {}, "did": {}}
    for split in ("valid", "test"):
        keys = scaffold_clusters([smiles[int(r)] for r in rows[split]])
        n_clusters = int(keys.max()) + 1
        weights = rng.exponential(scale=1.0, size=(replicates, n_clusters))
        errs, ys = {}, {}
        for name, run in runs.items():
            d = load_pred(run, split, "best")
            errs[name] = np.asarray(d["raw"], float) - np.asarray(d["y_true"], float)
            ys[name] = np.asarray(d["y_true"], float)
        split_out = {"n": int(len(rows[split])), "n_clusters": n_clusters, "models": {}}
        for name in runs:
            s_rmse = np.empty(replicates)
            for i in range(replicates):
                w = weights[i][keys]
                s_rmse[i] = math.sqrt(float(np.sum(w * errs[name] ** 2) / np.sum(w)))
            split_out["models"][name] = {"rmse_point": rmse(errs[name] + ys[name], ys[name]),
                                         "lo": float(np.percentile(s_rmse, 2.5)),
                                         "hi": float(np.percentile(s_rmse, 97.5))}
        out["splits"][split] = split_out
        # paired deltas for the semantic / perm pairs
        for name, (ctrl, cand) in {"FS_vs_LS": ("LS", "FS"), "FP_vs_LP": ("LP", "FP"),
                                   "FS_vs_B": ("B", "FS")}.items():
            if ctrl not in errs or cand not in errs:
                continue
            d_rmse = np.empty(replicates)
            for i in range(replicates):
                w = weights[i][keys]
                d_rmse[i] = math.sqrt(float(np.sum(w * errs[ctrl] ** 2) / np.sum(w))) - math.sqrt(
                    float(np.sum(w * errs[cand] ** 2) / np.sum(w)))
            out["paired"].setdefault(name, {})[split] = {
                "point": rmse(errs[ctrl] + ys[ctrl], ys[ctrl]) - rmse(errs[cand] + ys[cand], ys[cand]),
                "lo": float(np.percentile(d_rmse, 2.5)), "hi": float(np.percentile(d_rmse, 97.5)),
            }
        # DiD: (LS-FS) - (LP-FP), same weights
        if all(k in errs for k in ("LS", "FS", "LP", "FP")):
            s_rmse = {k: np.empty(replicates) for k in ("LS", "FS", "LP", "FP")}
            for i in range(replicates):
                w = weights[i][keys]
                for k in s_rmse:
                    s_rmse[k][i] = math.sqrt(float(np.sum(w * errs[k] ** 2) / np.sum(w)))
            did_samples = (s_rmse["LS"] - s_rmse["FS"]) - (s_rmse["LP"] - s_rmse["FP"])
            out["did"].setdefault("semantics", {})[split] = {
                "point": (rmse(errs["LS"] + ys["LS"], ys["LS"]) - rmse(errs["FS"] + ys["FS"], ys["FS"]))
                         - (rmse(errs["LP"] + ys["LP"], ys["LP"]) - rmse(errs["FP"] + ys["FP"], ys["FP"])),
                "lo": float(np.percentile(did_samples, 2.5)), "hi": float(np.percentile(did_samples, 97.5)),
            }
    return out


# ---------------------------------------------------------------------------
# mechanical route labels
# ---------------------------------------------------------------------------


def label_route(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered mechanical labels from the available numbers."""
    labels: list[str] = []
    sem = payload.get("did", {}).get("semantics", {})
    fs_ls = payload.get("pairs", {}).get("FS_vs_LS", {})
    fp_lp = payload.get("pairs", {}).get("FP_vs_LP", {})
    b_fs = payload.get("pairs", {}).get("FS_vs_B", {})
    bs = payload.get("bootstrap", {})

    def pos(block, split):
        try:
            return float(block[split]["rel_improvement"]) > 0
        except Exception:
            return None

    did_v = sem.get("valid", {}).get("did") if sem else None
    did_t = sem.get("test", {}).get("did") if sem else None
    fs_over_ls_v = pos(fs_ls, "valid")
    fs_over_ls_t = pos(fs_ls, "test")
    # INITIALIZATION_SIGNAL_ONLY: LS beats B, FS does not beat LS
    if b_fs and "valid" in b_fs:
        pass
    # GENERIC_REPARAMETERIZATION: Delta_sem ~ Delta_perm
    if did_v is not None and did_t is not None:
        if abs(did_v) < 1e-4 and abs(did_t) < 1e-4:
            labels.append("GENERIC_REPARAMETERIZATION")
    # SEMANTIC_SHARING_SUPPORTED_THIS_PANEL
    if did_v is not None and did_t is not None and did_v > 0 and did_t > 0 and fs_over_ls_t:
        labels.append("SEMANTIC_SHARING_SUPPORTED_THIS_PANEL")
    if not labels:
        labels.append("NO_USEFUL_SIGNAL")
    return {"labels": labels, "did_valid": did_v, "did_test": did_t}


def run_all(*, with_bootstrap: bool = True) -> dict[str, Any]:
    """Assemble every table that can be built from the present runs."""
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    default = {
        "LS": C.arm_name("LS", "R", "lipo", "N100"),
        "FS": C.arm_name("FS", "R", "lipo", "N100"),
        "LP": C.arm_name("LP", "R", "lipo", "N100"),
        "FP": C.arm_name("FP", "R", "lipo", "N100"),
    }
    present = {k: v for k, v in default.items() if has_run(v)}
    tables = {}
    for k, run in present.items():
        m = load_run(run)
        tables[k] = {"run": run, "parameters": m["model"]["parameters"],
                     "best_epoch": m["summary"]["best_valid"]["epoch"],
                     "summary": m["summary"]}
    C.write_json(C.ANALYSIS / "tables_main.json", tables)

    comparisons: dict[str, Any] = {}
    if all(k in present for k in ("LS", "FS")):
        comparisons["FS_vs_LS"] = compare(present["FS"], present["LS"])
    if all(k in present for k in ("LP", "FP")):
        comparisons["FP_vs_LP"] = compare(present["FP"], present["LP"])
    did_payload = {}
    if "FS_vs_LS" in comparisons and "FP_vs_LP" in comparisons:
        did_payload["semantics"] = did(comparisons["FS_vs_LS"], comparisons["FP_vs_LP"])
    C.write_json(C.ANALYSIS / "tables_comparisons.json", comparisons)
    C.write_json(C.ANALYSIS / "tables_did.json", did_payload)

    bs = bootstrap(present) if (with_bootstrap and len(present) >= 2) else {}
    if bs:
        C.write_json(C.ANALYSIS / "tables_bootstrap.json", bs)
    judgement = label_route({"did": did_payload, "pairs": comparisons, "bootstrap": bs})
    C.write_json(C.ANALYSIS / "tables_judgements.json", judgement)

    return {"metrics": {"n_runs": len(present), "labels": judgement["labels"]},
            "artifacts": [C.track_rel(C.ANALYSIS / f) for f in
                          ("tables_main.json", "tables_comparisons.json", "tables_did.json", "tables_judgements.json")]}


__all__ = [name for name in dir() if not name.startswith("_")]
