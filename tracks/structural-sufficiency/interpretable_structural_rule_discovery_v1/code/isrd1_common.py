"""Shared data, metric, bootstrap and logistic utilities for ISRD-v1.

Round: ``tracks/structural-sufficiency/interpretable_structural_rule_discovery_v1``

Read-only reuse of the frozen SGA-v1 input bundle (C/F design blocks, X-C/X-F
boosters, registry), the MSDS-v0 B2/B0 coefficients and predictions, the raw
instance cache and the Bemis-Murcko scaffolds.  Nothing historical is refitted.

Key v1 repair relative to v0: structural feature indices are decoded **per arm**
(``decode_feature``), because X-C is ``H + C`` (width 3012) while X-F is
``H + F`` (width 5910).  The v0 code decoded both boosters with one global
``H + C + F`` layout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

ROUND_ID = "interpretable-structural-rule-discovery-v1"
SEED = 0

CODE = Path(__file__).resolve().parent
ROUND = CODE.parent
ANALYSIS = ROUND / "analysis"
PREDICTIONS = ANALYSIS / "predictions"
MODELS = ANALYSIS / "models"
REPO = ROUND.parents[2]

SGA = REPO / "tracks" / "structural-sufficiency" / "structural_gain_attribution_v1"
MSDS = REPO / "tracks" / "structural-sufficiency" / "molhiv_multigranular_structure_sharing_v0"
V0 = REPO / "tracks" / "structural-sufficiency" / "interpretable_structural_rule_discovery_v0"

BUNDLE = SGA / "bundle" / "sav1_inputs.npz"
BUNDLE_META = SGA / "bundle" / "sav1_inputs_meta.json"
BOOSTERS = {"X-C": SGA / "runs" / "phaseA" / "X-C_model.json",
            "X-F": SGA / "runs" / "phaseA" / "X-F_model.json"}
BOOSTER_META = {"X-C": SGA / "runs" / "phaseA" / "X-C.json",
                "X-F": SGA / "runs" / "phaseA" / "X-F.json"}
REGISTRY = {
    "C": SGA / "analysis" / "registry_v1" / "structure_registry_coarse_v1.csv",
    "F": SGA / "analysis" / "registry_v1" / "structure_registry_full_v1.csv",
}
B2_COEF = MSDS / "data" / "phaseB_coefficients.npz"
B2_PRED = MSDS / "data" / "phaseB_predictions.npz"
INSTANCES = MSDS / "data" / "phaseA_instances.npz"
SCAFFOLDS = MSDS / "data" / "molhiv_scaffolds.npz"
MORGAN = MSDS / "data" / "molhiv_morgan_r2_2048.npz"

SPLITS = ("train", "valid", "test")
SPLIT_N = {"train": 32901, "valid": 4113, "test": 4113}
H_WIDTH = 2258            # H = 2048 ECFP4 bits + 210 RDKit descriptors
N_C = 754
N_F = 3652
N_STRUCT = N_C + N_F
#: arm -> (structural block, block width); the arm design is H + that single block
ARM_BLOCK = {"X-C": ("C", N_C), "X-F": ("F", N_F)}
ARM_WIDTH = {arm: H_WIDTH + w for arm, (_g, w) in ARM_BLOCK.items()}


# ---------------------------------------------------------------------------
# A1: arm-specific feature decoding (the v0 error)
# ---------------------------------------------------------------------------


def decode_feature(arm: str, feature_idx: int) -> tuple[str, int]:
    """Decode one booster feature index into ``(granularity, column)``.

    X-C consumes ``H + C`` and X-F consumes ``H + F``; a global feature index is
    therefore only meaningful together with its arm.  Any index outside the arm
    design raises instead of silently falling into another block.
    """
    if arm not in ARM_WIDTH:
        raise ValueError(f"unknown arm {arm!r}")
    width = ARM_WIDTH[arm]
    idx = int(feature_idx)
    if not (0 <= idx < width):
        raise ValueError(f"feature {idx} outside {arm} design width {width}")
    if idx < H_WIDTH:
        return ("H", idx)
    if arm == "X-C":
        return ("C", idx - H_WIDTH)
    if arm == "X-F":
        return ("F", idx - H_WIDTH)
    raise ValueError(arm)


def arm_offset(arm: str) -> int:
    """First structural column of the arm design (identical for both arms)."""
    if arm not in ARM_WIDTH:
        raise ValueError(arm)
    return H_WIDTH


def structure_feature(arm: str, gran: str, column: int) -> int:
    """Inverse of :func:`decode_feature` for a structural (granularity, column)."""
    if ARM_BLOCK[arm][0] != gran:
        raise ValueError(f"{arm} has no {gran} block")
    if not (0 <= int(column) < ARM_BLOCK[arm][1]):
        raise ValueError(f"{gran}[{column}] outside block width")
    return H_WIDTH + int(column)


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
        with np.load(SCAFFOLDS) as zs:
            self.scaffold = np.asarray(zs["scaffolds"])
        self.scaffold_code = _factorize(self.scaffold)
        self.block = {"C": self.C, "F": self.F}
        self.arm_block = {arm: ARM_BLOCK[arm][0] for arm in ARM_WIDTH}
        for k in SPLITS:
            assert self.rows[k].size == SPLIT_N[k], (k, self.rows[k].size)
        self._ints: sp.csr_matrix | None = None

    def _bond_counts(self) -> np.ndarray:
        sys.path.insert(0, str(MSDS / "code"))
        import msds_data as D

        m = D.load_molecules()
        nb = m.bond_offsets[1:] - m.bond_offsets[:-1]
        assert np.array_equal(m.n_atoms.astype(np.int64), self.n_atoms)
        assert np.array_equal(m.label.astype(np.int64), self.y)
        return nb.astype(np.int64)

    # -- structure blocks ---------------------------------------------------
    def col(self, gran: str, index: int) -> np.ndarray:
        """Dense frozen scaled count column (all 41 127 rows), float32 values."""
        return np.asarray(self.block[gran][:, index].toarray()).ravel().astype(np.float32)

    def sub(self, gran: str, index: int, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.block[gran][rows][:, index].toarray()).ravel().astype(np.float64)

    def sub32(self, gran: str, index: int, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.block[gran][rows][:, index].toarray()).ravel().astype(np.float32)

    def struct_matrix(self, gran: str) -> sp.csr_matrix:
        return self.block[gran]

    def all_structure_matrix(self) -> sp.csr_matrix:
        """``[C | F]`` scaled counts, 41 127 x 4 406 (CSR)."""
        return sp.hstack([self.C, self.F], format="csr")

    def size_matrix(self) -> np.ndarray:
        return np.column_stack([np.log1p(self.n_atoms.astype(np.float64)),
                                np.log1p(self.n_bonds.astype(np.float64))])

    def train_mask(self) -> np.ndarray:
        m = np.zeros(self.y.size, dtype=bool)
        m[self.rows["train"]] = True
        return m

    def split_code(self) -> np.ndarray:
        code = np.zeros(self.y.size, dtype=np.int8)
        for i, k in enumerate(SPLITS):
            code[self.rows[k]] = i
        return code

    # -- raw integer instance counts ---------------------------------------
    def integer_counts(self) -> sp.csr_matrix:
        """Raw integer occurrence counts per molecule from the MSDS-v0 instance cache.

        The cache stores every discovered type id per molecule together with its
        multiplicity; ``registry["*"]["discovered_type_id"]`` maps the frozen
        vocabulary column to that id.
        """
        if self._ints is not None:
            return self._ints
        reg = registry()
        with np.load(INSTANCES, allow_pickle=True) as z:
            n = int(z["n_molecules"])
            blocks = []
            for gran, pfx in (("C", "coarse"), ("F", "full")):
                indptr = np.asarray(z[f"{pfx}_indptr"], dtype=np.int64)
                indices = np.asarray(z[f"{pfx}_indices"], dtype=np.int64)
                data = np.asarray(z[f"{pfx}_data"], dtype=np.int64)
                width = int(reg[gran].shape[0])
                col_of_dtid = np.full(int(indices.max()) + 1, -1, dtype=np.int64)
                dtid = reg[gran]["discovered_type_id"].to_numpy().astype(np.int64)
                col_of_dtid[dtid] = np.arange(width)
                cols = col_of_dtid[indices]
                keep = cols >= 0
                mat = sp.csr_matrix((data[keep].astype(np.float32), (np.repeat(
                    np.arange(n), np.diff(indptr))[keep], cols[keep])), shape=(n, width))
                blocks.append(mat)
            out = sp.hstack(blocks, format="csr")
        self._ints = out
        return out

    def raw_count(self, gran: str, index: int) -> np.ndarray:
        """Dense raw integer count column (int64)."""
        mat = self.integer_counts()
        off = 0 if gran == "C" else N_C
        col = np.asarray(mat[:, off + index].toarray()).ravel()
        return np.rint(col).astype(np.int64)


def _factorize(values: Sequence[str]) -> np.ndarray:
    order: dict[str, int] = {}
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        out[i] = order.setdefault(v, len(order))
    return out


_REG_CACHE: dict[str, Any] = {}


def registry() -> dict[str, Any]:
    import pandas as pd

    if "reg" not in _REG_CACHE:
        _REG_CACHE["reg"] = {gran: pd.read_csv(path, keep_default_na=False)
                             for gran, path in REGISTRY.items()}
    return _REG_CACHE["reg"]


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
    """Paired scaffold bootstrap of AUC(a)-AUC(b) and AP(a)-AP(b) on one split."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    codes = scaffold_code[idx]
    uniq, inv = np.unique(codes, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(uniq.size + 1))
    groups = [order[bounds[g]:bounds[g + 1]] for g in range(uniq.size)]
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
# weighted logistic regression (dense Newton, identical to v0)
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

    Exact Newton with backtracking (unchanged from v0 so R0/R128 stay comparable).
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


# ---------------------------------------------------------------------------
# sparse equivalent of the standardised dense logistic (COUNT-ALL)
# ---------------------------------------------------------------------------


def fit_logistic_sparse_standardised(X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray,
                                     C: float, mean: np.ndarray, sd: np.ndarray,
                                     offset: np.ndarray | None = None,
                                     penalize_intercept: bool = True,
                                     max_iter: int = 60, tol: float = 1e-9,
                                     verbose: bool = False) -> dict[str, Any]:
    """Fit ``sw-BCE`` logistic on the Train-only standardised design ``(X-mean)/sd``.

    The standardised design is dense, but it is a sparse matrix ``A = X/sd`` plus a
    constant shift ``-mean/sd`` per row.  With ``b' = b - (mean/sd)·u`` the linear
    predictor is ``b' + A u`` and the (penalised) centred intercept is
    ``b = b' + m·u`` with ``m = mean/sd``.  The exact Newton system of the dense
    centred problem is therefore solved without materialising ``(X-mean)/sd``.
    """
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    m = mean / sd
    A = X.multiply(1.0 / sd[None, :]).tocsr()
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    theta = np.zeros(p + 1, dtype=np.float64)          # [b', u]
    pen_b = 1.0 / C if penalize_intercept else 0.0
    pen_u = np.full(p, 1.0 / C)

    def unpack(th: np.ndarray) -> tuple[float, np.ndarray]:
        return float(th[0]), th[1:]

    def zscore(th: np.ndarray) -> np.ndarray:
        bp, u = unpack(th)
        return off + bp + A @ u

    def parts(th: np.ndarray):
        bp, u = unpack(th)
        z = off + bp + A @ u
        prob = sigmoid(z)
        bc = bp + float(m @ u)                          # centred intercept
        loss = float(np.sum(sw * (np.logaddexp(0.0, z) - y * z))
                     + 0.5 * pen_b * bc * bc + 0.5 * np.sum(pen_u * u * u))
        r = sw * (prob - y)                             # dL/dz
        w = sw * prob * (1.0 - prob)
        g = np.empty(p + 1)
        g[0] = float(np.sum(r)) + pen_b * bc
        gb = A.T @ r
        g[1:] = gb + pen_u * u + pen_b * bc * m
        Hb = float(np.sum(w)) + pen_b
        Hbu = np.asarray(A.T @ w).ravel() + pen_b * m
        H = np.empty((p + 1, p + 1))
        H[0, 0] = Hb
        H[0, 1:] = Hbu
        H[1:, 0] = Hbu
        AW = A.multiply(w[:, None]).tocsr()
        Huu = np.asarray((A.T @ AW).todense())
        Huu += np.diag(pen_u)
        if pen_b:
            Huu += pen_b * np.outer(m, m)
        H[1:, 1:] = Huu
        return loss, g, H, z

    loss, g, H, z = parts(theta)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        if np.max(np.abs(g)) < tol:
            break
        try:
            import scipy.linalg as la

            step = -la.solve(H, g, assume_a="pos")
        except Exception:
            step = -np.linalg.lstsq(H, g, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            step = -g / np.maximum(np.diag(H), 1e-12)
        t = 1.0
        for _ in range(30):
            cand = theta + t * step
            loss2, g2, H2, z2 = parts(cand)
            if np.isfinite(loss2) and loss2 <= loss + 1e-12:
                break
            t *= 0.5
        else:
            break
        theta, loss, g, H, z = cand, loss2, g2, H2, z2
        if verbose:
            print(f"[sparse-fit] it={n_iter} loss={loss:.6g} grad={np.max(np.abs(g)):.3e}",
                  flush=True)
    bp, u = unpack(theta)
    b = bp + float(m @ u)
    return {"intercept": float(b), "coef": u.copy(), "objective": loss,
            "grad_inf": float(np.max(np.abs(g))), "n_iter": int(n_iter),
            "logit": z, "converged": bool(np.max(np.abs(g)) < tol),
            "parametrisation": "centred intercept b = b' + (mean/sd).u"}


def predict_standardised(X: sp.csr_matrix | np.ndarray, coef: np.ndarray, intercept: float,
                         mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mean) / sd
    return intercept + np.asarray(Z @ coef).ravel()


def active_counts(rule_matrix: np.ndarray) -> dict[str, Any]:
    k = np.asarray(rule_matrix).sum(axis=1)
    return {"mean": float(k.mean()), "median": float(np.median(k)),
            "p05": float(np.percentile(k, 5)), "p95": float(np.percentile(k, 95)),
            "min": int(k.min()), "max": int(k.max()),
            "histogram": {int(v): int((k == v).sum()) for v in np.unique(k)}}
