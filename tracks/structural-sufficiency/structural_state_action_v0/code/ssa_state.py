"""Exact whole-graph state ``U(G)`` and its nonlinear readout ``H``.

For every root representation ``h_v`` (width ``W = 192`` for W3) and every
attention head ``k = 1..4`` the per-root state feature is

``phi(h_v) = [1, f(h_v), h_v, h_v**2, exp(score_k - c_k),
              exp(score_k - c_k) * h_v for k = 1..4]``

and the whole-graph state is the sum ``U(G) = sum_v phi(h_v)``.  ``f`` is the
frozen per-atom contribution of W3 and ``score_k`` its frozen head scorer.

Shifting: the raw state contains exponentials.  A **single, head-specific
constant** ``c_k`` is used for every molecule of the round (calibrated once on a
fixed hash sample and stored in the report).  It cancels exactly in the pooled
vector ``P_k / Z_k``, but it does *not* cancel in ``U(B) - U(A)``; using a
per-molecule ``max`` shift instead would make two molecules' states
incomparable.  The audit reports that failure mode numerically.

``H`` rebuilds, from ``(U(G), g(G))`` alone: node count, ``sum f(h)``,
``mean h``, ``mean h^2``, ``log1p(n)``, each head's pooled vector, the ``rho``
output, the attention-residual output and the final prediction in eV.  This is
an *equivalence rewrite* of the existing computation, not a new model and not a
performance claim.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import ssa_common as C


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def _mlp2(x: np.ndarray, w0, b0, w1, b1) -> np.ndarray:
    return _linear(_silu(_linear(x, w0, b0)), w1, b1)


class StateMachine:
    """Frozen float64 re-implementation of the W3 post-``local`` computation."""

    def __init__(self, model: Any, spec: Any, norm: Mapping[str, Any],
                 label_norm: Mapping[str, Any]):
        self.model = model
        self.spec = spec
        self.norm = dict(norm)
        self.W = int(model.cfg["width"])
        self.H = int(model.readout.n_heads)
        self.phi_dim = 2 + 2 * self.W + self.H + self.H * self.W
        self.label_mean_eV = float(label_norm["mean_eV"])
        self.label_std_eV = float(label_norm["std_eV"])
        sd = {k: np.asarray(v.detach().cpu().numpy(), np.float64)
              for k, v in model.state_dict().items()}
        self.state = sd
        self.f = (sd["f.lins.0.weight"], sd["f.lins.0.bias"],
                  sd["f.lins.1.weight"], sd["f.lins.1.bias"])
        self.rho = (sd["rho.lins.0.weight"], sd["rho.lins.0.bias"],
                    sd["rho.lins.1.weight"], sd["rho.lins.1.bias"])
        self.scorers = [
            (sd[f"readout.heads.{k}.fc1.weight"], sd[f"readout.heads.{k}.fc1.bias"],
             sd[f"readout.heads.{k}.fc2.weight"], sd[f"readout.heads.{k}.fc2.bias"])
            for k in range(self.H)
        ]
        self.fc1 = (sd["readout.fc1.weight"], sd["readout.fc1.bias"])
        self.fc2 = (sd["readout.fc2.weight"], sd["readout.fc2.bias"])
        self.global_mean = sd["global_mean"]
        self.global_scale = sd["global_scale"]
        self.shift = np.zeros(self.H, np.float64)

    # -- per-atom quantities ------------------------------------------------

    def f_forward(self, h: np.ndarray) -> np.ndarray:
        w0, b0, w1, b1 = self.f
        return _mlp2(h, w0, b0, w1, b1)

    def scores(self, h: np.ndarray) -> np.ndarray:
        out = []
        for w0, b0, w1, b1 in self.scorers:
            out.append(_mlp2(h, w0, b0, w1, b1))
        return np.concatenate(out, axis=1)

    def phi_atoms(self, h: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
        h = np.asarray(h, np.float64)
        if scores is None:
            scores = self.scores(h)
        e = np.exp(scores - self.shift)
        parts = [
            np.ones((h.shape[0], 1)),
            self.f_forward(h),
            h,
            h * h,
            e,
        ]
        parts.extend(e[:, k:k + 1] * h for k in range(self.H))
        return np.concatenate(parts, axis=1)

    # -- whole-graph state --------------------------------------------------

    def graph_state(self, h: np.ndarray, scores: np.ndarray, atom_start: np.ndarray,
                    *, block_atoms: int = 40_000) -> np.ndarray:
        """``U(G)`` for a block of molecules whose atoms are in graph order.

        The molecule boundaries are respected exactly: a reduction block is a
        whole number of molecules, never a split one.
        """
        h = np.asarray(h)
        scores = np.asarray(scores, np.float64)
        atom_start = np.asarray(atom_start, np.int64)
        G = atom_start.size
        W, H = self.W, self.H
        U = np.zeros((G, self.phi_dim), np.float64)
        if G == 0:
            return U
        atom_stop = np.r_[atom_start[1:], h.shape[0]].astype(np.int64)
        U[:, 0] = atom_stop - atom_start
        g0 = 0
        while g0 < G:
            a0 = int(atom_start[g0])
            g1 = int(np.searchsorted(atom_start, a0 + block_atoms, side="right")) - 1
            g1 = min(max(g1, g0 + 1), G)
            a1 = int(atom_stop[g1 - 1])
            hb = np.asarray(h[a0:a1], np.float64)
            sb = scores[a0:a1]
            st = atom_start[g0:g1] - a0
            f = self.f_forward(hb)
            e = np.exp(sb - self.shift)
            U[g0:g1, 1] = np.add.reduceat(f, st, axis=0)[:, 0]
            U[g0:g1, 2:2 + W] = np.add.reduceat(hb, st, axis=0)
            U[g0:g1, 2 + W:2 + 2 * W] = np.add.reduceat(hb * hb, st, axis=0)
            U[g0:g1, 2 + 2 * W:2 + 2 * W + H] = np.add.reduceat(e, st, axis=0)
            for k in range(H):
                off = 2 + 2 * W + H + k * W
                U[g0:g1, off:off + W] = np.add.reduceat(e[:, k:k + 1] * hb, st, axis=0)
            g0 = g1
        return U

    def state_parts(self, U: np.ndarray) -> dict[str, np.ndarray]:
        W, H = self.W, self.H
        n = U[:, 0]
        fsum = U[:, 1]
        sum_h = U[:, 2:2 + W]
        sum_h2 = U[:, 2 + W:2 + 2 * W]
        Z = U[:, 2 + 2 * W:2 + 2 * W + H]
        P = [U[:, 2 + 2 * W + H + k * W: 2 + 2 * W + H + (k + 1) * W] for k in range(H)]
        return {"n": n, "sum_f": fsum, "sum_h": sum_h, "sum_h2": sum_h2, "Z": Z, "P": P}

    # -- nonlinear readout --------------------------------------------------

    def readout(self, U: np.ndarray, global_raw: np.ndarray) -> dict[str, np.ndarray]:
        """``H(U, g)`` -> every reconstructed intermediate and the eV output."""
        parts = self.state_parts(U)
        n = parts["n"]
        denom = np.maximum(n, 1.0)[:, None]
        mean_h = parts["sum_h"] / denom
        mean_h2 = parts["sum_h2"] / denom
        log1p_n = np.log1p(n)[:, None]
        g = (np.asarray(global_raw, np.float64) - self.global_mean) / self.global_scale
        rho_in = np.concatenate([g, mean_h, mean_h2, log1p_n], axis=1)
        w0, b0, w1, b1 = self.rho
        rho = _mlp2(rho_in, w0, b0, w1, b1)
        Z = parts["Z"]
        if (Z <= 0).any():
            raise RuntimeError("degenerate attention normaliser in the state")
        pooled = [parts["P"][k] / Z[:, k:k + 1] for k in range(self.H)]
        z_in = np.concatenate(pooled, axis=1)
        w1_, b1_ = self.fc1
        w2_, b2_ = self.fc2
        residual = _mlp2(z_in, w1_, b1_, w2_, b2_)
        contrib = parts["sum_f"][:, None]
        logits = contrib + rho + residual
        pred_eV = logits[:, 0] * self.label_std_eV + self.label_mean_eV
        return {
            "logits": logits[:, 0],
            "pred_eV": pred_eV,
            "contrib": contrib[:, 0],
            "rho": rho[:, 0],
            "residual": residual[:, 0],
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "log1p_n": log1p_n[:, 0],
            "n": n,
            "pooled": z_in,
            "g_norm": g,
        }


# ---------------------------------------------------------------------------
# model forward helpers
# ---------------------------------------------------------------------------


def forward_atoms(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """One frozen forward of a collated batch; returns ``h``, scores and aux."""
    import torch

    model.eval()
    with torch.no_grad():
        logits, aux = model(batch, return_aux=True)
        h = aux["h"]
        scores = torch.cat([head(h) for head in model.readout.heads], dim=1)
    return {
        "h": h.detach().cpu().numpy(),
        "scores": scores.detach().cpu().numpy().astype(np.float64),
        "logits": logits.detach().cpu().numpy().reshape(-1),
        "aux": {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in aux.items()},
        "atom_start": batch["atom_off"].detach().cpu().numpy().astype(np.int64),
        "atom_off": np.r_[batch["atom_off"].detach().cpu().numpy().astype(np.int64),
                          int(batch["atom_feat"].shape[0])],
    }


def computation_audit(model: Any, batch: Mapping[str, Any], EM: Any) -> dict[str, Any]:
    """Code facts about the frozen W3 computation graph (no re-derivation)."""
    calls = EM.count_module_calls(model, batch)
    local = model.local
    readout = model.readout
    cfg = dict(model.cfg)
    return {
        "model_class": type(model).__name__,
        "base_class": type(model).__mro__[1].__name__,
        "readout_class": type(readout).__name__,
        "variant": str(getattr(model, "variant", "")),
        "widths": {k: int(v) for k, v in cfg.items()},
        "reads_b_mean": bool(local.use_b_mean),
        "local_input_layout": {k: [int(v[0]), int(v[1])] for k, v in local.layout.items()},
        "local_input_dim": int(sum(v[1] - v[0] for v in local.layout.values())),
        "local_mlp_input_dim": int(local.mlp.lins[0].weight.shape[1]),
        "local_column_count": int(len(local.local_cols)),
        "local_columns_contiguous_from_zero": bool(
            list(local.local_cols) == list(range(len(local.local_cols)))),
        "module_calls_per_forward": {k: int(v) for k, v in calls.items()},
        "post_local_reads": ["h", "atom_batch", "n_graphs", "global"],
        "post_local_does_not_read": ["bond_index", "bond_feat", "bond_type",
                                     "nbr_atom", "nbr_code", "path_atom", "path_cat",
                                     "path_end", "path_end63", "path_closure",
                                     "atom_feat", "atom_scalar"],
        "n_heads": int(readout.n_heads),
        "head_scorer_hidden": int(readout.heads[0].fc1.out_features),
        "fc1": [int(readout.fc1.in_features), int(readout.fc1.out_features)],
        "fc2": [int(readout.fc2.in_features), int(readout.fc2.out_features)],
        "rho_input_dim": int(model.rho.lins[0].weight.shape[1]),
        "dropout_p": float(model.drop.p),
        "dropout_active_in_eval": False,
        "statement": (
            "local() is called exactly once per forward and is the only consumer "
            "of bond/nbr/path tensors; every later term is a function of h, the "
            "atom->graph map, the molecule count and the graph descriptor g."
        ),
    }


def calibrate_shift(machine: StateMachine, scores: np.ndarray) -> dict[str, Any]:
    """Fixed per-head shift ``c_k = max_k`` over a fixed, label-free sample."""
    c = np.max(scores, axis=0)
    machine.shift = np.asarray(c, np.float64)
    return {
        "policy": "one fixed constant per attention head, identical for every molecule",
        "c_k": [float(x) for x in machine.shift.tolist()],
        "source": "maximum head score over the fixed hash state sample",
        "cancels_in_pooled_vector": True,
        "required_for_cross_molecule_subtraction": True,
        "fp64_exp_range": {"underflow_ln": -745.13, "overflow_ln": 709.78,
                           "max_shifted_exponent": 0.0,
                           "min_shifted_exponent": float(np.min(scores - machine.shift))},
    }


__all__ = [
    "StateMachine",
    "calibrate_shift",
    "computation_audit",
    "forward_atoms",
]
