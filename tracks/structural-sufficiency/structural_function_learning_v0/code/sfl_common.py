"""Shared, side-effect-free utilities for structural-function-learning-v0.

Everything here reads *frozen, tracked* inputs (the SGA-v1 bundle, the tracked raw
MolHIV graph file used only for bond counts, and the frozen ZINC design) so the
same code runs locally and on the remote GPU/CPU hosts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

ROUND_ID = "structural-function-learning-v0"
SEED = 0
CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
ANALYSIS = ROUND / "analysis"
PREDICTIONS = ANALYSIS / "predictions"
FUNCTIONS = ANALYSIS / "functions"
CURVES = ANALYSIS / "curves"
MODELS = ANALYSIS / "models"
RUNS = ROUND / "runs"
REPO = ROUND.parents[2]
SS = REPO / "tracks" / "structural-sufficiency"
SGA = SS / "structural_gain_attribution_v1"
ISRD = SS / "interpretable_structural_rule_discovery_v1"
MSDS = SS / "molhiv_multigranular_structure_sharing_v0"

BUNDLE = SGA / "bundle" / "sav1_inputs.npz"
BUNDLE_META = SGA / "bundle" / "sav1_inputs_meta.json"
#: tracked raw OGB MolHIV graph file (atom/bond_ptr); used ONLY for n_bonds identity
RAW_MOLHIV = (SS / "lcrc_molnet_benchmark_pilot_v2" / "analysis" / "source" / "raw"
              / "lcrc_raw_v2" / "ogbg-molhiv.npz")
ZINC_DESIGN = SGA / "data" / "zinc_design.npz"
ZINC_META = SGA / "data" / "zinc_design_meta.json"
XGB_ALL_MODEL = ISRD / "analysis" / "models" / "XGB-ALL.json"
ISRD_PHASEC = ISRD / "analysis" / "phaseC.json"
ISRD_PRED = ISRD / "analysis" / "predictions"
SGA_RUNS = SGA / "runs"

REGISTRY = {
    "C": SGA / "analysis" / "registry_v1" / "structure_registry_coarse_v1.csv",
    "F": SGA / "analysis" / "registry_v1" / "structure_registry_full_v1.csv",
}
SCAFFOLDS = MSDS / "data" / "molhiv_scaffolds.npz"
MSDS_INSTANCES = MSDS / "data" / "phaseA_instances.npz"
MSDS_PRED = MSDS / "data" / "phaseB_predictions.npz"

SPLITS = ("train", "valid", "test")
SPLIT_N = {"train": 32901, "valid": 4113, "test": 4113}
HIV_N_C = 754
HIV_N_F = 3652
HIV_N_STRUCT = HIV_N_C + HIV_N_F          # 4406
HIV_WIDTH = 2 + HIV_N_STRUCT              # 4408
ZINC_SPLIT_N = {"train": 10000, "valid": 1000, "test": 1000}
ZINC_WIDTH = 4642 + 9 + 1250              # 5901
ZINC_N_C = 9
ZINC_N_F = 1250


# ---------------------------------------------------------------------------
# IO / hashing / provenance
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               default=_json_default) + "\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def git_state() -> dict[str, Any]:
    import subprocess

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked": bool(run("status", "--porcelain")),
            "status_short": [line for line in run("status", "--porcelain").splitlines()
                             if line.strip()]}


def environment() -> dict[str, Any]:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "numpy": np.__version__}
    for name in ("scipy", "sklearn", "pandas", "xgboost", "torch", "rdkit"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[name] = None
    try:
        import torch

        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["torch_cuda"] = None
        env["cuda_available"] = False
        env["gpu_name"] = None
    return env


# ---------------------------------------------------------------------------
# MolHIV data (frozen; n_bonds from the tracked raw graph file)
# ---------------------------------------------------------------------------


def _molhiv_n_bonds() -> np.ndarray:
    with np.load(RAW_MOLHIV) as z:
        nb = (np.asarray(z["bond_ptr"])[1:] - np.asarray(z["bond_ptr"])[:-1]) // 2
    return nb.astype(np.int64)


class HIVData:
    """Frozen 4408-column MolHIV design (size + C + F), no refitting of anything."""

    def __init__(self, with_scaffold: bool = False) -> None:
        with np.load(BUNDLE) as z:
            self.y = np.asarray(z["y"], dtype=np.int64)
            self.n_atoms = np.asarray(z["n_atoms"], dtype=np.int64)
            self.rows = {k: np.asarray(z[f"rows_{k}"], dtype=np.int64) for k in SPLITS}
            self.C = sp.csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]),
                                   shape=(self.y.size, HIV_N_C))
            self.F = sp.csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]),
                                   shape=(self.y.size, HIV_N_F))
        self.n_bonds = _molhiv_n_bonds()
        if self.n_bonds.size != self.y.size:
            raise ValueError("bond count length mismatch")
        self.n = int(self.y.size)
        for k in SPLITS:
            if self.rows[k].size != SPLIT_N[k]:
                raise ValueError(f"split size mismatch: {k}")
        self.size_log = np.column_stack([np.log1p(self.n_atoms.astype(np.float64)),
                                         np.log1p(self.n_bonds.astype(np.float64))])
        self.scaffold_code = None
        self.scaffold = None
        if with_scaffold:
            with np.load(SCAFFOLDS) as zs:
                self.scaffold = np.asarray(zs["scaffolds"])
            self.scaffold_code = _factorize(self.scaffold)

    def structures(self) -> sp.csr_matrix:
        return sp.hstack([self.C, self.F], format="csr")

    def design_sparse(self) -> sp.csr_matrix:
        """[log1p(n_atoms), log1p(n_bonds) | C | F] as CSR."""
        return sp.hstack([sp.csr_matrix(self.size_log), self.C, self.F], format="csr")

    def design_dense(self, rows: np.ndarray | None = None) -> np.ndarray:
        """float32 dense 4408-column design; rows=None is 41127 x 4408 (~725 MB)."""
        sel = np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)
        out = np.empty((sel.size, HIV_WIDTH), dtype=np.float32)
        out[:, 0] = self.size_log[sel, 0]
        out[:, 1] = self.size_log[sel, 1]
        out[:, 2:2 + HIV_N_C] = self.C[sel].toarray()
        out[:, 2 + HIV_N_C:] = self.F[sel].toarray()
        return out

    def nn_design_dense(self, rows: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Phase-C design: size terms Train-standardised (clip +-10), counts frozen."""
        sel = np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)
        train = self.rows["train"]
        mu = self.size_log[train].mean(axis=0)
        sd = self.size_log[train].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        out = self.design_dense(sel)
        out[:, :2] = np.clip((self.size_log[sel] - mu) / sd, -10.0, 10.0)
        meta = {"size_mean": mu, "size_std": sd, "fit_on": "Train only",
                "size_clip": 10.0, "structure_scale": "frozen count/sqrt(n_atoms)"}
        return out, meta

    def split_code(self) -> np.ndarray:
        code = np.zeros(self.n, dtype=np.int8)
        for i, k in enumerate(SPLITS):
            code[self.rows[k]] = i
        return code


def _factorize(values: Sequence[str]) -> np.ndarray:
    order: dict[str, int] = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        out[i] = order.setdefault(v, len(order))
    return out


def registry() -> dict[str, Any]:
    import pandas as pd

    return {g: pd.read_csv(p, keep_default_na=False) for g, p in REGISTRY.items()}


def full_parent_columns() -> np.ndarray:
    """COARSE column of every FULL column's parent (-1 if outside the vocabulary)."""
    reg = registry()
    with np.load(MSDS_INSTANCES, allow_pickle=True) as z:
        full_parent = np.asarray(z["full_parent"], dtype=np.int64)
    c_dtid = reg["C"]["discovered_type_id"].to_numpy().astype(np.int64)
    c_col_of = {int(d): i for i, d in enumerate(c_dtid)}
    f_dtid = reg["F"]["discovered_type_id"].to_numpy().astype(np.int64)
    return np.array([c_col_of.get(int(full_parent[d]), -1) for d in f_dtid], dtype=np.int64)


# ---------------------------------------------------------------------------
# ZINC data (frozen SGA-v1 design: Z + C + F, width 5901)
# ---------------------------------------------------------------------------


class ZincData:
    def __init__(self) -> None:
        with np.load(ZINC_DESIGN) as z:
            n = int(np.asarray(z["y"]).size)
            self.y = np.asarray(z["y"], dtype=np.float64)
            self.n_atoms = np.asarray(z["n_atoms"], dtype=np.int64)
            self.rows = {k: np.asarray(z[f"rows_{k}"], dtype=np.int64) for k in SPLITS}
            self.Z = self._csr(z, "Z", n)
            self.C = self._csr(z, "C", n)
            self.F = self._csr(z, "F", n)
        self.n = n
        for k in SPLITS:
            if self.rows[k].size != ZINC_SPLIT_N[k]:
                raise ValueError(f"ZINC split size mismatch: {k}")

    @staticmethod
    def _csr(z: Any, prefix: str, n_rows: int) -> sp.csr_matrix:
        return sp.csr_matrix(
            (np.asarray(z[f"{prefix}_data"], dtype=np.float32),
             np.asarray(z[f"{prefix}_indices"], dtype=np.int64),
             np.asarray(z[f"{prefix}_indptr"], dtype=np.int64)),
            shape=(n_rows, int(np.asarray(z[f"{prefix}_indices"]).max()) + 1),
        )

    def design_dense(self, rows: np.ndarray | None = None) -> np.ndarray:
        sel = np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)
        out = np.empty((sel.size, ZINC_WIDTH), dtype=np.float32)
        out[:, :self.Z.shape[1]] = self.Z[sel].toarray()
        out[:, self.Z.shape[1]:self.Z.shape[1] + self.C.shape[1]] = self.C[sel].toarray()
        out[:, self.Z.shape[1] + self.C.shape[1]:] = self.F[sel].toarray()
        return out

    def nn_design_dense(self, rows: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Identical to ZN1 preprocessing (size standardised, whole Z clipped +-10)."""
        sel = np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)
        tr = self.rows["train"]
        mu = self.Z[tr][:, [0, 1]].toarray().mean(axis=0)
        sd = self.Z[tr][:, [0, 1]].toarray().std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        out = self.design_dense(sel)
        z_cols = self.Z.shape[1]
        out[:, [0, 1]] = (out[:, [0, 1]] - mu) / sd
        np.clip(out[:, :z_cols], -10.0, 10.0, out=out[:, :z_cols])
        meta = {"size_mean": mu, "size_std": sd, "fit_on": "Train only",
                "size_clip": 10.0, "clip_scope": "Z block (4642 columns)",
                "C_F_scale": "frozen count/sqrt(n_atoms)"}
        return out, meta


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -60.0, 60.0)))


def auc_ap(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = sigmoid(np.asarray(score, dtype=np.float64))
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p))}


def hiv_metrics(score: np.ndarray, y: np.ndarray, rows: dict[str, np.ndarray],
                score_is_logit: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in SPLITS:
        r = rows[k]
        m = auc_ap(y[r], score[r])
        m.update({"n": int(r.size), "n_positive": int(y[r].sum())})
        m["evidence_label"] = ("DEVELOPMENT_TEST" if k == "test"
                               else ("VALID" if k == "valid" else "TRAIN"))
        out[k] = m
    return out


def reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    err = np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))
    rho = spearmanr(y, pred).statistic if y.size > 2 else float("nan")
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median_ae": float(np.median(err)), "spearman": float(rho),
            "n": int(y.size)}


def zinc_metrics(pred: np.ndarray, y: np.ndarray, rows: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in SPLITS:
        m = reg_metrics(y[rows[k]], pred[rows[k]])
        m["evidence_label"] = ("DEVELOPMENT_TEST" if k == "test"
                               else ("VALID" if k == "valid" else "TRAIN"))
        out[k] = m
    return out


def metric_from_predictions(pred: np.ndarray, y: np.ndarray, rows: dict[str, np.ndarray],
                            dataset: str) -> dict[str, Any]:
    return (hiv_metrics(pred, y, rows) if dataset == "hiv"
            else zinc_metrics(pred, y, rows))


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def _group_indices(codes: np.ndarray) -> list[np.ndarray]:
    uniq, inv = np.unique(codes, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(uniq.size + 1))
    return [order[bounds[g]:bounds[g + 1]] for g in range(uniq.size)]


def scaffold_bootstrap(idx: np.ndarray, y: np.ndarray, score_a: np.ndarray,
                       score_b: np.ndarray, scaffold_code: np.ndarray,
                       n_boot: int = 1000, seed: int = SEED) -> dict[str, Any]:
    """Paired MolHIV bootstrap resampling whole scaffolds (AUC and AP differences)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    groups = _group_indices(scaffold_code[idx])
    pa, pb = sigmoid(score_a[idx]), sigmoid(score_b[idx])
    yy = y[idx]
    rng = np.random.default_rng(seed)
    deltas = np.full((n_boot, 2), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        take = np.concatenate([groups[g] for g in pick])
        yb = yy[take]
        if yb.min() == yb.max():
            continue
        deltas[b, 0] = roc_auc_score(yb, pa[take]) - roc_auc_score(yb, pb[take])
        deltas[b, 1] = (average_precision_score(yb, pa[take])
                        - average_precision_score(yb, pb[take]))
    point = np.array([roc_auc_score(yy, pa) - roc_auc_score(yy, pb),
                      average_precision_score(yy, pa) - average_precision_score(yy, pb)])
    out: dict[str, Any] = {"n_groups": int(len(groups)), "n_boot": int(n_boot),
                           "n_valid_resamples": int(np.isfinite(deltas[:, 0]).sum()),
                           "delta_auc": float(point[0]), "delta_ap": float(point[1])}
    for j, name in enumerate(("auc", "ap")):
        v = deltas[np.isfinite(deltas[:, j]), j]
        out[f"delta_{name}_ci95"] = [float(np.percentile(v, 2.5)),
                                     float(np.percentile(v, 97.5))]
        out[f"delta_{name}_se"] = float(np.std(v, ddof=1))
    return out


def molecule_bootstrap_mae(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                           idx: np.ndarray, n_boot: int = 1000,
                           seed: int = SEED) -> dict[str, Any]:
    """Paired ZINC bootstrap over molecules; positive gain = MAE(B) - MAE(A)."""
    yy = y[idx]
    ea = np.abs(yy - pred_a[idx])
    eb = np.abs(yy - pred_b[idx])
    point = float(eb.mean() - ea.mean())
    rng = np.random.default_rng(seed)
    n = idx.size
    vals = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.integers(0, n, size=n)
        vals[b] = eb[take].mean() - ea[take].mean()
    return {"n_groups": int(n), "n_boot": int(n_boot),
            "n_valid_resamples": int(n_boot),
            "delta_mae_gain": point,
            "delta_mae_ci95": [float(np.percentile(vals, 2.5)),
                               float(np.percentile(vals, 97.5))],
            "delta_mae_se": float(np.std(vals, ddof=1))}


# ---------------------------------------------------------------------------
# ARD-like xgboost helpers (tree parsing used by Phase B/E)
# ---------------------------------------------------------------------------


def parse_tree_features(dump: str, feature_names: Sequence[str]) -> set[str]:
    """Feature names used by one dumped tree (``[c12<...]`` nodes)."""
    import re

    pattern = re.compile(r"\[(" + "|".join(re.escape(n) for n in feature_names) + r")<")
    return set(pattern.findall(dump))


def iteration_range_for(rounds: int) -> tuple[int, int]:
    return (0, int(rounds))
