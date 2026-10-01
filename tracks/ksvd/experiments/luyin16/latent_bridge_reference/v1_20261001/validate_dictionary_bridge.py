"""Structural/numerical checks only; no molecular data or MAE validation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_dictionary_bridge_reference import (
    bridge, frame_initialization, ista_codes, normalized_dictionary,
    soft_threshold, symmetric_pair,
)


def main():
    rng = np.random.default_rng(0)
    report = {"scope": "synthetic structural and numerical audit only",
              "official_train_loaded": False, "official_valid_loaded": False,
              "official_test_loaded": False, "zinc_mae_claim": None,
              "repository_review_revision": "65d4b8fb82261e9ec0a94bbf02d7ccc36e14219f"}

    # Any linear scalar-to-vector map preserves this exact moment collision.
    a = np.array([0., 0., 3., 3.])[:, None]
    b = np.array([0., 1., 1., 4.])[:, None]
    projection = rng.normal(size=(1, 128))
    def moments(x):
        return np.concatenate([x.sum(0), x.mean(0), x.std(0)])
    collision = float(np.max(np.abs(moments(a @ projection) - moments(b @ projection))))
    nonlinear_a = float(np.maximum(a - 1., 0.).sum())
    nonlinear_b = float(np.maximum(b - 1., 0.).sum())
    report["linear_node_moment_witness"] = {
        "max_moment_difference": collision,
        "nonlinear_before_pool_sum": [nonlinear_a, nonlinear_b],
        "limitation": "node branch witness, not a collision proof for the complete HierRel model",
    }
    assert collision < 1e-12 and nonlinear_a != nonlinear_b

    # Hard top-1 is discontinuous at a rank tie; soft threshold is continuous.
    left = np.array([[1. + 1e-7, 1. - 1e-7]])
    right = left[:, ::-1].copy()
    def hard_top1(x):
        result = np.zeros_like(x)
        keep = np.argmax(np.abs(x), axis=1)
        result[np.arange(len(x)), keep] = x[np.arange(len(x)), keep]
        return result
    report["rank_tie_witness"] = {
        "input_l2_difference": float(np.linalg.norm(left - right)),
        "hard_top1_l2_difference": float(np.linalg.norm(hard_top1(left) - hard_top1(right))),
        "soft_threshold_l2_difference": float(np.linalg.norm(soft_threshold(left, .05) - soft_threshold(right, .05))),
        "limitation": "a solver boundary example, not causal evidence for the observed ZINC failure",
    }

    # A distant same-type node changes the old graph-group relation object.
    z = rng.normal(size=(3, 128))
    p = rng.choice([-1., 1.], size=(128, 32)) / np.sqrt(32.)
    def old_relation(zall):
        mu = zall.mean(0)
        m = mu @ p
        du, dv = (zall[:2] - mu) @ p
        return np.concatenate([symmetric_pair(m, m), symmetric_pair(du, dv),
                               m * (du + dv), np.array([1., 0., 0., 0.])])
    old_before = old_relation(z)
    old_after = old_relation(np.concatenate([z, 5 * rng.normal(size=(1, 128))]))
    report["graph_group_context_witness"] = {
        "local_endpoint_codes_changed": False,
        "relation_object_l2_change": float(np.linalg.norm(old_before - old_after)),
        "limitation": "demonstrates graph-context dependence, not that such context is intrinsically harmful",
    }

    # The proposed bridge protects the existing 48D interface at initialization.
    d, v = frame_initialization()
    h = rng.normal(size=(256, 48)) * np.exp(rng.normal(size=(256, 1)))
    h[0] = 0
    tight_error = float(np.max(np.abs(d @ d.T - 2 * np.eye(48))))
    unit_error = float(np.max(np.abs(np.linalg.norm(d, axis=0) - 1)))
    identity, _ = bridge(h, d, v, lambda1=0, lambda2=0)
    identity_error = float(np.max(np.abs(identity - h)))
    output, aux = bridge(h, d, v)
    relative = float(np.sum((h - output) ** 2) / np.sum(h ** 2))
    per_row = np.sum((h - output) ** 2, axis=1) / (np.sum(h ** 2, axis=1) + 1e-12)
    report["bridge_initialization"] = {
        "tight_frame_max_error": tight_error, "unit_column_max_error": unit_error,
        "zero_penalty_identity_max_error_float64": identity_error,
        "production_relative_squared_error": relative,
        "production_row_error_p95": float(np.quantile(per_row, .95)),
        "production_mean_l0": float(np.count_nonzero(aux["alpha"], axis=1).mean()),
        "note": "variable density; initial coefficients are deliberately not forced to top16",
    }
    assert unit_error < 1e-12 and tight_error < 1e-12 and identity_error < 1e-9
    assert relative < .01 and np.isfinite(output).all() and np.all(output[0] == 0)

    h32, d32, v32 = h.astype(np.float32), d.astype(np.float32), v.astype(np.float32)
    exact32, _ = bridge(h32, d32, v32, lambda1=0, lambda2=0)
    rel32 = float(np.linalg.norm(exact32 - h32) / np.linalg.norm(h32))
    report["float32_identity_relative_l2_error"] = rel32
    assert rel32 < 1e-6

    perm = rng.permutation(len(h))
    permuted, _ = bridge(h[perm], d, v)
    separate, _ = bridge(h[:17], d, v)
    pair_left = symmetric_pair(output[:16], output[16:32])
    pair_right = symmetric_pair(output[16:32], output[:16])
    report["invariance"] = {
        "node_relabel_max_difference": float(np.max(np.abs(permuted - output[perm]))),
        "batch_composition_max_difference": float(np.max(np.abs(separate - output[:17]))),
        "endpoint_swap_max_difference": float(np.max(np.abs(pair_left - pair_right))),
    }
    assert max(report["invariance"].values()) < 1e-10

    # Monotone convex inner objective even after a substantial dictionary change.
    changed = normalized_dictionary(d + .12 * rng.normal(size=d.shape))
    _, history = ista_codes(aux["normalized"][1:65], changed, return_history=True)
    largest_increase = float(np.max(np.diff(history, axis=0)))
    report["changed_dictionary_solver"] = {"largest_objective_step_increase": largest_increase}
    assert largest_increase <= 1e-10

    # Finite-difference task derivatives test dictionary and decoder activity.
    ds, vs = frame_initialization(dim=4)
    hs = rng.normal(size=(8, 4))
    target = np.arange(8, dtype=np.float64) / 7
    w = np.array([.2, -.3, .7, .5])
    def task(dictionary, values):
        decoded, _ = bridge(hs, dictionary, values)
        return float(np.mean(np.abs(decoded @ w - target)))
    def gradient(which):
        base = ds if which == "dictionary" else vs
        grad = np.zeros_like(base)
        step = 1e-6
        for index in np.ndindex(base.shape):
            plus, minus = base.copy(), base.copy()
            plus[index] += step
            minus[index] -= step
            loss_plus = task(plus, vs) if which == "dictionary" else task(ds, plus)
            loss_minus = task(minus, vs) if which == "dictionary" else task(ds, minus)
            grad[index] = (loss_plus - loss_minus) / (2 * step)
        return grad
    gd, gv = gradient("dictionary"), gradient("values")
    before, after = task(ds, vs), task(ds - 1e-3 * gd, vs - 1e-3 * gv)
    report["synthetic_task_gradient"] = {
        "dictionary_norm": float(np.linalg.norm(gd)), "values_norm": float(np.linalg.norm(gv)),
        "loss_before": before, "loss_after_small_step": after,
        "limitation": "finite differences in a synthetic task, not a PyTorch backward or ZINC training test",
    }
    assert np.linalg.norm(gd) > 1e-6 and np.linalg.norm(gv) > 1e-6 and after < before

    report["passed"] = True
    path = Path(__file__).with_name("dictionary_bridge_audit.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
