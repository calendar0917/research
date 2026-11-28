"""Shared data, metric, bootstrap and offset-logistic utilities for ISRD-v0.

Round: ``tracks/structural-sufficiency/interpretable_structural_rule_discovery_v0``
Protocol id: ``interpretable-structural-rule-discovery-v0``

Everything here is read-only with respect to the historical rounds: the frozen C/F
design blocks, the frozen B2 coefficients/logits, the X-C / X-F boosters and the
corrected structure registry are consumed as they are.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

ROUND_ID = "interpretable-structural-rule-discovery-v0"
SEED = 0

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
ANALYSIS = ROUND / "analysis"
PREDICTIONS = ANALYSIS / "predictions"
CONFIGS = ROUND / "configs"
REPO = ROUND.parents[2]

SGA = REPO / "tracks" / "structural-sufficiency" / "structural_gain_attribution_v1"
MSDS = REPO / "tracks" / "structural-sufficiency" / "molhiv_multigranular_structure_sharing_v0"
HIST = REPO / "tracks" / "structural-sufficiency" / "lcrc_molnet_benchmark_pilot_v2"

BUNDLE = SGA / "bundle" / "sav1_inputs.npz"
BUNDLE_META = SGA / "bundle" / "sav1_inputs_meta.json"
BOOSTERS = {"X-C": SGA / "runs" / "phaseA" / "X-C_model.json",
            "X-F": SGA / "runs" / "phaseA" / "X-F_model.json"}
REGISTRY = {
    "C": SGA / "analysis" / "registry_v1" / "structure_registry_coarse_v1.csv",
    "F": SGA / "analysis" / "registry_v1" / "structure_registry_full_v1.csv",
}
CASE_STRUCTURES = SGA / "analysis" / "case_structures.json"
B2_COEF = MSDS / "data" / "phaseB_coefficients.npz"
B2_PRED = MSDS / "data" / "phaseB_predictions.npz"
PHASEB_JSON = MSDS / "analysis" / "phaseB_linear.json"
INSTANCES = MSDS / "data" / "phaseA_instances.npz"
SCAFFOLDS = MSDS / "data" / "molhiv_scaffolds.npz"
MORGAN = MSDS / "data" / "molhiv_morgan_r2_2048.npz"

SPLITS = ("train", "valid", "test")
H_WIDTH = 2258          # H = 2048 Morgan bits + 210 descriptors, offsets used by Phase A
N_C = 754
N_F = 3652
#: global design offsets of the structural blocks inside X-C (H+C) and X-F (H+F)
OFFSET = {"C": H_WIDTH, "F": H_WIDTH + N_C}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


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


def _default(obj: Any) -> Any:
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def git_state() -> dict[str, Any]:
    import subprocess

    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked": bool(run("status", "--porcelain"))}


def environment() -> dict[str, Any]:
    import platform

    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "cpu_count": os.cpu_count()}
    for name in ("scipy", "sklearn", "pandas", "xgboost", "rdkit", "ogb"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[name] = None
    return env


# ---------------------------------------------------------------------------
# frozen inputs
# ---------------------------------------------------------------------------


class Data:
    """Frozen MolHIV structure data as consumed by this round."""

    def __init__(self) -> None:
        with np.load(BUNDLE) as z:
            self.y = np.asarray(z["y"], dtype=np.int64)
            self.n_atoms = np.asarray(z["n_atoms"], dtype=np.int64)
            self.rows = {k: np.asarray(z[f"rows_{k}"], dtype=np.int64) for k in SPLITS}
            self.C = sp.csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]),
                                   shape=(self.y.size, N_C))
            self.F = sp.csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]),
                                   shape=(self.y.size, N_F))
        self.n_bonds = self._bond_counts()
        self.scaffold = np.load(SCAFFOLDS)["scaffolds"]
        self.scaffold_code = _factorize(self.scaffold)
        self.block = {"C": self.C, "F": self.F}
        self.offset = {"C": H_WIDTH, "F": H_WIDTH + N_C}  # GLOBAL design offsets (X-C / X-F)
        for k in SPLITS:
            assert self.rows[k].size == {"train": 32901, "valid": 4113, "test": 4113}[k]

    def _bond_counts(self) -> np.ndarray:
        sys.path.insert(0, str(MSDS / "code"))
        import msds_data as D

        m = D.load_molecules()
        nb = m.bond_offsets[1:] - m.bond_offsets[:-1]
        assert np.array_equal(m.n_atoms.astype(np.int64), self.n_atoms)
        assert np.array_equal(m.label.astype(np.int64), self.y)
        return nb.astype(np.int64)

    def col(self, gran: str, index: int) -> np.ndarray:
        """Dense frozen scaled count column (all 41 127 rows)."""
        mat = self.block[gran]
        return np.asarray(mat[:, index].toarray()).ravel().astype(np.float32)

    def sub(self, gran: str, index: int, rows: np.ndarray) -> np.ndarray:
        mat = self.block[gran][rows]
        return np.asarray(mat[:, index].toarray()).ravel().astype(np.float64)

    def train_mask(self) -> np.ndarray:
        m = np.zeros(self.y.size, dtype=bool)
        m[self.rows["train"]] = True
        return m


def _factorize(values: Sequence[str]) -> np.ndarray:
    order: dict[str, int] = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        out[i] = order.setdefault(v, len(order))
    return out


def registry() -> dict[str, Any]:
    import pandas as pd

    out = {}
    for gran, path in REGISTRY.items():
        df = pd.read_csv(path, keep_default_na=False)
        out[gran] = df
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -60.0, 60.0)))


def metrics(logit: np.ndarray, y: np.ndarray, ogb: bool = False) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = sigmoid(np.asarray(logit, dtype=np.float64))
    out = {"n": int(y.size), "n_positive": int(y.sum()),
           "roc_auc": float(roc_auc_score(y, p)),
           "average_precision": float(average_precision_score(y, p))}
    if ogb:
        from ogb.graphproppred import Evaluator

        auc = float(Evaluator(name="ogbg-molhiv").eval(
            {"y_true": y.reshape(-1, 1).astype(float),
             "y_pred": p.reshape(-1, 1)})["rocauc"])
        out["ogb_evaluator_rocauc"] = auc
        out["ogb_abs_diff"] = abs(auc - out["roc_auc"])
    return out


def all_metrics(logit: np.ndarray, y: np.ndarray, rows: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for k in SPLITS:
        r = rows[k]
        out[k] = metrics(logit[r], y[r], ogb=(k != "train"))
        if k != "train":
            out[k]["evidence_label"] = "DEVELOPMENT_TEST" if k == "test" else "VALID"
    return out


# ---------------------------------------------------------------------------
# paired scaffold bootstrap
# ---------------------------------------------------------------------------


def scaffold_bootstrap(idx: np.ndarray, y: np.ndarray, logit_a: np.ndarray, logit_b: np.ndarray,
                       scaffold_code: np.ndarray, n_boot: int = 1000,
                       seed: int = SEED) -> dict[str, Any]:
    """Paired scaffold bootstrap of AUC(a) - AUC(b) and AP(a) - AP(b) on one split."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    codes = scaffold_code[idx]
    uniq, inv = np.unique(codes, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(uniq.size)]
    pa, pb = sigmoid(logit_a[idx]), sigmoid(logit_b[idx])
    yy = y[idx]
    rng = np.random.default_rng(seed)
    deltas = np.full((n_boot, 2), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        take = np.concatenate([groups[g] for g in pick])
        yb = yy[take]
        if yb.min() == yb.max():
            continue
        deltas[b, 0] = roc_auc_score(yb, pa[take]) - roc_auc_score(yb, pb[take])
        deltas[b, 1] = average_precision_score(yb, pa[take]) - average_precision_score(yb, pb[take])
    point = np.array([roc_auc_score(yy, pa) - roc_auc_score(yy, pb),
                      average_precision_score(yy, pa) - average_precision_score(yy, pb)])
    out = {"n_scaffold_groups": int(uniq.size), "n_boot": int(n_boot),
           "n_valid_resamples": int(np.isfinite(deltas[:, 0]).sum()),
           "delta_auc": float(point[0]), "delta_ap": float(point[1])}
    for j, name in enumerate(("auc", "ap")):
        v = deltas[np.isfinite(deltas[:, j]), j]
        out[f"delta_{name}_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[f"delta_{name}_se"] = float(np.std(v, ddof=1))
    return out


# ---------------------------------------------------------------------------
# weighted logistic regression with optional offset (Newton, exact)
# ---------------------------------------------------------------------------


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    w = np.empty(n, dtype=np.float64)
    for c in (0.0, 1.0):
        m = y == c
        w[m] = n / (2.0 * m.sum())
    return w


def fit_logistic(X: np.ndarray, y: np.ndarray, sw: np.ndarray, C: float,
                 offset: np.ndarray | None = None, penalize_intercept: bool = True,
                 max_iter: int = 200, tol: float = 1e-10) -> dict[str, Any]:
    """Minimise ``sum_i sw_i BCE(y_i, offset_i + b + X_i w) + ||w||^2/(2C)``.

    Exact Newton with backtracking.  Returns coefficients, objective and diagnostics.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    beta = np.zeros(p + 1, dtype=np.float64)
    pen = np.full(p + 1, 1.0 / C)
    if not penalize_intercept:
        pen[0] = 0.0
    Xa = np.hstack([np.ones((n, 1)), X])

    def objective(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        z = off + Xa @ b
        loss = float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(pen * b * b))
        prob = sigmoid(z)
        g = Xa.T @ (sw * (prob - y)) + pen * b
        h = Xa.T @ (Xa * (sw * prob * (1.0 - prob))[:, None]) + np.diag(pen)
        return loss, g, h

    loss, g, H = objective(beta)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        if np.max(np.abs(g)) < tol:
            break
        step = None
        try:
            import scipy.linalg as la

            step = -la.solve(H, g, assume_a="pos")
        except Exception:
            step = -np.linalg.lstsq(H, g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            step = -g / np.maximum(np.diag(H), 1e-12)
        t = 1.0
        for _ in range(40):
            cand = beta + t * step
            new_loss, new_g, new_H = objective(cand)
            if np.isfinite(new_loss) and new_loss <= loss + 1e-12:
                break
            t *= 0.5
        else:
            break
        beta, loss, g, H = cand, new_loss, new_g, new_H
    z = off + Xa @ beta
    return {"intercept": float(beta[0]), "coef": beta[1:].copy(), "objective": loss,
            "grad_inf": float(np.max(np.abs(g))), "n_iter": int(n_iter),
            "logit": z, "converged": bool(np.max(np.abs(g)) < 1e-6 * max(1.0, n))}


def grad_check(X: np.ndarray, y: np.ndarray, sw: np.ndarray, C: float,
               offset: np.ndarray | None, penalize_intercept: bool, eps: float = 1e-6) -> float:
    """Max relative error between analytic and finite-difference gradient."""
    X = np.asarray(X, dtype=np.float64)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=np.float64)
    n, p = X.shape
    beta = np.linspace(-0.3, 0.4, p + 1)
    Xa = np.hstack([np.ones((n, 1)), X])
    pen = np.full(p + 1, 1.0 / C)
    if not penalize_intercept:
        pen[0] = 0.0

    def f(b: np.ndarray) -> float:
        z = off + Xa @ b
        return float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(pen * b * b))

    z = off + Xa @ beta
    analytic = Xa.T @ (sw * (sigmoid(z) - y)) + pen * beta
    numeric = np.empty_like(analytic)
    for j in range(p + 1):
        e = np.zeros_like(beta)
        e[j] = eps
        numeric[j] = (f(beta + e) - f(beta - e)) / (2 * eps)
    return float(np.max(np.abs(analytic - numeric)) / (1.0 + np.max(np.abs(numeric))))


def active_counts(rule_matrix: np.ndarray) -> dict[str, Any]:
    """Distribution of active rules per molecule."""
    k = np.asarray(rule_matrix).sum(axis=1)
    return {"mean": float(k.mean()), "median": float(np.median(k)),
            "p05": float(np.percentile(k, 5)), "p95": float(np.percentile(k, 95)),
            "min": int(k.min()), "max": int(k.max()),
            "histogram": {int(v): int((k == v).sum()) for v in np.unique(k)}}
