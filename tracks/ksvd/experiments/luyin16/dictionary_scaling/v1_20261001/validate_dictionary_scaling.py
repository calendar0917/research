"""Synthetic numerical audit, not a ZINC/MAE experiment. Run with NumPy only."""
from __future__ import annotations

from pathlib import Path
import copy
import json
import numpy as np

from dictionary_scale_reference import (WidthSpec, state_init, forward, embed_small,
                                       parameter_arrays)
from latent_dictionary_bridge_reference import (frame_initialization, bridge,
                                                ista_codes, normalized_dictionary)


def toy_objects(seed=17):
    rng = np.random.default_rng(seed)
    sizes = [8, 9, 7]
    batch = np.repeat(np.arange(3), sizes)
    pairs = []
    offset = 0
    for n in sizes:
        pairs.extend((offset + i, offset + j) for i in range(n) for j in range(i + 1, n))
        offset += n
    pairs = np.asarray(pairs, dtype=int).T
    buckets = np.arange(pairs.shape[1]) % 5
    relations = rng.normal(size=(pairs.shape[1], 15))
    relations[:, -1] = 0  # C6 removes path-count coordinate.
    fused = rng.normal(size=(len(batch), 446))
    auxiliary = rng.normal(size=(3, 40))
    return (fused, batch, pairs, buckets, relations, auxiliary)


def max_delta(a, b):
    return float(np.max(np.abs(a - b)))


def cast_arrays(state, dtype):
    return {k: (v.astype(dtype) if isinstance(v, np.ndarray) else
                cast_arrays(v, dtype) if isinstance(v, dict) else v)
            for k, v in state.items()}


def run():
    rng = np.random.default_rng(91)
    objects = toy_objects()
    report = {
        "scope": "NumPy reference; synthetic post-stem objects, no ZINC data/fit/PyTorch",
        "repository_revision": "00c203624682a9ca46d54a225721e25015d8b28c",
        "official_test_loaded": False,
        "widths": {}, "containment": {}, "invariance": {}, "task_signal": {},
    }
    fixed = {"structural_dictionary", "node_binding", "edge_binding", "node_slot_encoder",
             "edge_slot_encoder", "global_encoder", "topology_encoder"}
    for m in [1, 2, 3]:
        spec = WidthSpec(m)
        inventory = spec.inventory()
        state = state_init(spec)
        actual_variable = sum(x.size for x in parameter_arrays(state))
        expected_variable = sum(v for k, v in inventory.items() if k not in fixed)
        assert actual_variable == expected_variable
        h = rng.normal(size=(96, spec.d))
        D, V = frame_initialization(spec.d)
        out, aux = bridge(h, D, V)
        rel_sq = float(np.sum((out - h) ** 2) / np.sum(h ** 2))
        assert rel_sq < 0.005
        identity, _ = bridge(h, D, V, lambda1=0, lambda2=0)
        rel_identity = float(np.linalg.norm(identity - h) / np.linalg.norm(h))
        assert rel_identity < 1e-7
        modified_D = D + rng.normal(scale=0.07, size=D.shape)
        alpha, history = ista_codes(h / np.sqrt(np.mean(h * h, axis=1, keepdims=True)),
                                    modified_D, return_history=True)
        largest_increase = float(np.max(np.diff(history, axis=0)))
        assert largest_increase < 1e-10
        prediction, endpoint = forward(state, *objects, return_aux=True)
        assert endpoint["graph"].shape == (3, spec.reader_input)
        assert endpoint["E"].shape == (24, spec.d)
        report["widths"][str(m)] = dict(
            d=spec.d, K=spec.k, p=spec.p, parameters=spec.parameters(),
            inventory=inventory, variable_array_count=actual_variable,
            formal_init_relative_squared_error=rel_sq,
            zero_lambda_identity_relative_L2=rel_identity,
            max_ISTA_objective_increase=largest_increase,
            initial_code_nonzero_fraction=float(np.mean(aux["alpha"] != 0)),
        )
    assert report["widths"]["1"]["parameters"] == 106925
    assert report["widths"]["3"]["parameters"] == 408651

    # Use moved dictionary/value and normalization parameters: containment must
    # work for learned parameters too, not just identity-frame initialization.
    small = state_init(WidthSpec(1))
    small["D"] += rng.normal(scale=0.06, size=small["D"].shape)
    small["V"] += rng.normal(scale=0.05, size=small["V"].shape)
    for k in ["relation", "pair"]:
        small[k]["gamma"] += rng.normal(scale=0.1, size=small[k]["gamma"].shape)
        small[k]["beta"] += rng.normal(scale=0.1, size=small[k]["beta"].shape)
    baseline, b = forward(small, *objects, return_aux=True)
    for m in [2, 3]:
        widened = embed_small(small, m)
        y, a = forward(widened, *objects, return_aux=True)
        assert max_delta(y, baseline) < 1e-10
        assert max_delta(a["E"], np.tile(b["E"], (1, m))) < 1e-10
        assert max_delta(a["pair"], np.tile(b["pair"], (1, m))) < 1e-10
        # Also check the unmasked count semantics in the untrained reference.
        plain_small = forward(small, *objects, zero_count=False)
        plain_wide = forward(widened, *objects, zero_count=False)
        assert max_delta(plain_small, plain_wide) < 1e-10
        report["containment"][str(m)] = dict(
            prediction_max_delta=max_delta(y, baseline),
            environment_max_delta=max_delta(a["E"], np.tile(b["E"], (1, m))),
            pair_max_delta=max_delta(a["pair"], np.tile(b["pair"], (1, m))),
            plain_count_prediction_delta=max_delta(plain_small, plain_wide),
            note="eval-mode width embedding; not a proposed training initializer",
        )
        float_objects = tuple(x.astype(np.float32) if x.dtype.kind == "f" else x for x in objects)
        small32 = cast_arrays(small, np.float32)
        wide32 = cast_arrays(widened, np.float32)
        y32 = forward(wide32, *float_objects)
        baseline32 = forward(small32, *float_objects)
        float_delta = max_delta(y32, baseline32)
        assert float_delta < 1e-5
        report["containment"][str(m)]["float32_prediction_max_delta"] = float_delta

    spec, state = WidthSpec(3), state_init(WidthSpec(3))
    fused, batch, pairs, buckets, relations, auxiliary = objects
    prediction = forward(state, *objects)
    perm = rng.permutation(len(batch))
    inverse = np.argsort(perm)
    permuted = forward(state, fused[perm], batch[perm], inverse[pairs], buckets,
                       relations, auxiliary)
    swapped = forward(state, fused, batch, pairs[::-1], buckets, relations, auxiliary)
    doubled = forward(state, np.concatenate([fused, fused]),
                      np.concatenate([batch, batch + 3]),
                      np.concatenate([pairs, pairs + len(batch)], axis=1),
                      np.tile(buckets, 2), np.tile(relations, (2, 1)),
                      np.tile(auxiliary, (2, 1)))
    before = fused.copy()
    forward(state, *objects)
    report["invariance"] = dict(
        node_relabel_delta=max_delta(prediction, permuted),
        endpoint_exchange_delta=max_delta(prediction, swapped),
        batch_offset_delta=max_delta(np.tile(prediction, 2), doubled),
        input_writeback_delta=max_delta(fused, before),
    )
    assert max(report["invariance"].values()) < 1e-10

    # MAE-only directional derivatives through the full synthetic task path.
    # This is not a replacement for actual PyTorch backward on official data.
    targets = prediction + np.array([0.2, -0.1, 0.25])
    def objective(s):
        return float(np.mean(np.abs(forward(s, *objects) - targets)))
    epsilon = 1e-5
    directions, derivatives = {}, {}
    for name in ["D", "V"]:
        v = rng.normal(size=state[name].shape)
        v /= np.linalg.norm(v)
        plus, minus = copy.deepcopy(state), copy.deepcopy(state)
        plus[name] += epsilon * v
        minus[name] -= epsilon * v
        derivative = (objective(plus) - objective(minus)) / (2 * epsilon)
        assert np.isfinite(derivative) and abs(derivative) > 1e-8
        directions[name], derivatives[name] = v, float(derivative)
    moved = copy.deepcopy(state)
    for name in ["D", "V"]:
        moved[name] -= 0.01 * np.sign(derivatives[name]) * directions[name]
    before_loss, after_loss = objective(state), objective(moved)
    assert after_loss < before_loss
    report["task_signal"] = dict(MAE_directional_derivatives=derivatives,
                                 toy_MAE_before=before_loss, toy_MAE_after=after_loss,
                                 interpretation="synthetic trainability only")
    report["all_gates_passed"] = True
    return report


if __name__ == "__main__":
    result = run()
    out = Path(__file__).with_name("dictionary_scale_audit.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"all_gates_passed": result["all_gates_passed"],
                      "params": {k: v["parameters"] for k, v in result["widths"].items()},
                      "containment": result["containment"],
                      "invariance": result["invariance"],
                      "task_signal": result["task_signal"]}, indent=2))
