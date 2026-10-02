"""Standalone graph-level prototype dictionary and certified convex MAE fit.

No ZINC data, PyTorch model or pretrained backbone is included here.
Fit-only preprocessing and prototypes; test/validation are transform-only.
Nyström basis is folded into prototype values for a single direct readout.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

SEED = 20261002
N_PROTOTYPES = 256
LAMBDA_GRID = (1e-5, 1e-4, 1e-3)
FULL_BLOCK_WIDTHS = (289, 485, 32, 8)


def squared_distance(x, y):
    return np.maximum(np.sum(x*x, axis=1)[:, None]
                      + np.sum(y*y, axis=1)[None, :] - 2.0*x @ y.T, 0.0)


@dataclass
class BlockScaler:
    mean: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    weights: np.ndarray
    block_widths: tuple

    @classmethod
    def fit(cls, raw, block_widths=FULL_BLOCK_WIDTHS):
        if raw.ndim != 2 or raw.shape[1] != sum(block_widths):
            raise ValueError('reader input/block layout mismatch')
        z = np.arcsinh(np.asarray(raw, dtype=np.float64))
        mean, std = z.mean(0), z.std(0)
        keep = std > 1e-8
        scale = np.ones(len(std))
        weights = np.zeros(len(std))
        start = 0
        n_blocks = sum(np.any(keep[sum(block_widths[:b]):sum(block_widths[:b+1])])
                       for b in range(len(block_widths)))
        if n_blocks == 0:
            raise ValueError('constant representation has no prototype geometry')
        for width in block_widths:
            stop = start+width
            active = keep[start:stop]
            if active.any():
                floor = .05 * np.median(std[start:stop][active])
                scale[start:stop] = np.maximum(std[start:stop], floor)
                weights[start:stop] = 1.0/np.sqrt(int(active.sum())*n_blocks)
            start = stop
        return cls(mean, scale, keep, weights, tuple(block_widths))

    def transform(self, raw):
        z = (np.arcsinh(np.asarray(raw, dtype=np.float64)) - self.mean)/self.scale
        return (z*self.weights)[:, self.keep]


@dataclass
class PrototypeDictionary:
    scaler: BlockScaler
    centers: np.ndarray
    inverse_root: np.ndarray
    bandwidth_squared: float
    selected_rows: np.ndarray
    spectrum: np.ndarray

    @classmethod
    def fit(cls, raw, block_widths=FULL_BLOCK_WIDTHS, n_prototypes=N_PROTOTYPES,
            seed=SEED):
        scaler = BlockScaler.fit(raw, block_widths)
        x = scaler.transform(raw)
        if len(x) < n_prototypes:
            raise ValueError('not enough fit graphs for frozen dictionary width')
        rng = np.random.default_rng(seed)
        # Target-independent representative dictionary; no exact graph-ID lookup.
        selected = rng.choice(len(x), size=n_prototypes, replace=False)
        centers = x[selected].copy()
        a = rng.integers(len(x), size=4096)
        b = rng.integers(len(x), size=4096)
        d2 = np.sum((x[a]-x[b])**2, axis=1)
        positive = d2[d2 > 1e-12]
        if not len(positive):
            raise ValueError('no nonzero distances')
        bandwidth_squared = float(np.median(positive))
        w = np.exp(-squared_distance(centers, centers)/(2*bandwidth_squared))
        w = (w+w.T)/2
        eigenvalues, u = np.linalg.eigh(w)
        # Consistent truncated pseudo-inverse, including duplicate prototypes.
        keep = eigenvalues > 1e-8*max(float(eigenvalues[-1]), 1.0)
        inv = np.zeros_like(eigenvalues)
        inv[keep] = 1/np.sqrt(eigenvalues[keep])
        inverse_root = (u*inv) @ u.T
        return cls(scaler, centers, inverse_root, bandwidth_squared, selected,
                   eigenvalues)

    def kernel(self, raw):
        x = self.scaler.transform(raw)
        return np.exp(-squared_distance(x, self.centers)/(2*self.bandwidth_squared))

    def codes(self, raw):
        return self.kernel(raw) @ self.inverse_root

    def design(self, raw):
        return np.column_stack([np.ones(len(raw)), self.codes(raw)])


def mae_primal_dual(a, y, coef, dual, lam):
    n = len(y)
    # Dual max y^T v - ||A^T v||²/(2 lambda), |v_i| <= 1/n.
    feasible = np.clip(dual, -1/n, 1/n)
    primal = float(np.mean(np.abs(a@coef-y)) + .5*lam*(coef@coef))
    projection = a.T@feasible
    lower = float(y@feasible - .5*(projection@projection)/lam)
    return dict(primal=primal, dual=lower, gap=max(0.0, primal-lower),
                dual_box_max=float(np.abs(feasible).max()),
                dual_box_limit=1/n)


def fit_mae(a, y, lam, *, max_iterations=12000, gap_tolerance=1e-6,
            time_budget_s=180, check_every=20, rho_multiplier=10.0):
    """Exact MAE + lambda/2 ||coef||², ADMM with a valid dual lower bound.

    Includes a penalized constant atom, so the objective is strongly convex.
    Status must be checked; a time/iteration cap is not a scientific negative.
    """
    started = time.perf_counter()
    a = np.asarray(a, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (a.ndim == 2 and y.shape == (len(a),) and lam > 0):
        raise ValueError('invalid convex regression input')
    n, k = a.shape
    rho = rho_multiplier/n
    gram = a.T@a + (lam/rho)*np.eye(k)
    factor = cho_factor(gram, lower=True, check_finite=False)
    coef = np.zeros(k)
    slack, scaled_dual = np.zeros(n), np.zeros(n)
    certificate = None
    converged = False
    for iteration in range(1, max_iterations+1):
        coef = cho_solve(factor, a.T@(y+slack-scaled_dual), check_finite=False)
        residual = a@coef-y
        value = residual+scaled_dual
        slack = np.sign(value)*np.maximum(np.abs(value)-1/(n*rho), 0)
        scaled_dual = value-slack
        if iteration % check_every == 0 or iteration == max_iterations:
            certificate = mae_primal_dual(a,y,coef,-rho*scaled_dual,lam)
            if certificate['gap'] <= gap_tolerance:
                converged = True
                break
            if time.perf_counter()-started >= time_budget_s:
                break
    if certificate is None:
        certificate = mae_primal_dual(a,y,coef,-rho*scaled_dual,lam)
    return coef, dict(status='CONVERGED' if converged else 'INCOMPLETE_SOLVER',
                      iterations=iteration, seconds=time.perf_counter()-started,
                      lambda_value=lam, certificate=certificate)


def folded_predict(dictionary, raw, coef, target_median):
    # Single readout: basis whitening is absorbed into prototype values.
    values = dictionary.inverse_root @ coef[1:]
    return target_median+coef[0]+dictionary.kernel(raw)@values
