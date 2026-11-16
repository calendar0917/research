"""Targeted implementation checks for this round (no historical suite).

Scope is deliberately narrow: only the new models, the new data wiring and the
frozen reuse gates.  Every check returns ``{"pass": bool, ...}`` so the same
functions back both ``analysis/checks.json`` and ``tests/test_cr_round.py``.

Sections
--------
``data``      official splits, real file SHA-256, Train-only norm/scaler
``params``    C1 / trunk / readout parameter counts and the B,C budget rule
``armA``      our arm A vs the read-only ``DissectLCRC("C1")`` on one real batch
``init``      bitwise-shared trunk, bitwise-shared ``u_j``, cross-process repeat
``single``    ``q(0) = 0``, vectorised vs scalar-loop value *and* gradient
``dropout``   exactly one dropout call on ``h``, used by every local head
``roundtrip`` raw/standardised round-trip and the zero-readout constant predictor
``grad``      finite, reachable gradients on local / ``u`` / ``a`` paths
``invariance`` eval batching, molecule isolation, atom renumbering
``ledger``    contribution closure and the a_k/c_k scale-equivalence transform
``checkpoint`` save/reload replay of predictions and the ledger
``gine``      the frozen G reference reuse gates
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import cr_common as C
import cr_data as D
import cr_models as M
import cr_train as T

MOD = D.modules()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tol(pred: np.ndarray) -> np.ndarray:
    return C.trace_tolerance(pred)


def _close(a: np.ndarray, b: np.ndarray, *, tol: float) -> dict[str, Any]:
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "tolerance": float(tol),
        "pass": bool(diff.size == 0 or float(diff.max()) <= float(tol)),
    }


def _one_batch(bundle: D.Bundle, rows: Sequence[int], split: str = "train"):
    rows = np.asarray(rows, dtype=np.int64)
    batch = bundle.collate(bundle.caches[split], rows)
    T._attach_target(batch, rows, bundle.y_std[split], _torch())
    return batch


def _torch():
    import torch

    return torch


def _arm(bundle: D.Bundle, arm: str, canonical_state, **kwargs):
    context_indices = [D.field_index(bundle.spec, name) for name in C.CONTEXT_FIELDS]
    specs = M.arm_specs(bundle)
    return M.build_arm(
        arm,
        bundle.spec,
        bundle.norm,
        canonical_state,
        n_active=bundle.active_global["n_active"],
        hidden=specs[arm].hidden,
        active_indices=bundle.active_global["active_indices"],
        context_indices=context_indices,
        device="cpu",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_data(bundle: D.Bundle) -> dict[str, Any]:
    audit = D.audit_dataset(bundle.dataset)
    ok = (
        audit["files"]["all_match"]
        and audit["split_disjoint"]
        and audit["split_covers_all_rows"]
        and audit["split_sizes"] == C.EXPECTED_SPLIT_SIZES[bundle.dataset]
        and all(audit["row_ids_sorted_ascending"].values())
        and audit["norm"]["fit_split"] == "train"
        and audit["norm"]["valid_or_test_used"] is False
        and audit["target_scaler"]["fit_split"] == "train"
        and audit["target_scaler"]["target_roundtrip_max_abs_error"] < 1e-9
        and all(v["s2_sum_matches_pointer"] and v["s1_equals_two_bonds"] for v in audit["occurrence"].values())
    )
    return {
        "check": "data",
        "dataset": bundle.dataset,
        "pass": bool(ok),
        "split_sizes": audit["split_sizes"],
        "file_sha256": {name: entry["actual_sha256"] for name, entry in audit["files"]["files"].items()},
        "row_id_sha256": audit["row_id_sha256"],
        "norm_fingerprint": audit["norm"]["fingerprint"],
        "target_scaler": {k: audit["target_scaler"][k] for k in ("mean", "std", "n", "fit_split")},
        "n_active_global": audit["active_global"]["n_active"],
        "contexts_status": audit["contexts"]["status"],
        "occurrence": audit["occurrence"],
    }


def check_parameters(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    specs = M.arm_specs(bundle)
    models = {arm: _arm(bundle, arm, canonical_state) for arm in C.ARMS}
    reports = {arm: M.parameter_report(model) for arm, model in models.items()}
    expectations = {
        "A_readout": (reports["A"]["readout"], M.readout_params("A", 0, specs["A"].n_active)),
        "B_readout": (reports["B"]["readout"], M.readout_params("B", specs["B"].hidden, specs["B"].n_active)),
        "C_readout": (reports["C"]["readout"], M.readout_params("C", specs["C"].hidden, specs["C"].n_active)),
        "trunk": (reports["A"]["trunk"], M.EXPECTED_COUNTS["trunk"]),
        "A_total": (reports["A"]["total"], M.EXPECTED_COUNTS["full_c1"]),
    }
    mismatches = {k: v for k, v in expectations.items() if int(v[0]) != int(v[1])}
    forbidden = sorted(
        key
        for arm in ("B", "C")
        for key in reports[arm]["readout_parameter_names"]
        if key.split(".")[0] in ("f", "rho")
    )
    return {
        "check": "parameters",
        "dataset": bundle.dataset,
        "pass": bool(not mismatches and not forbidden),
        "counts": {k: {"actual": int(v[0]), "expected": int(v[1])} for k, v in expectations.items()},
        "mismatches": mismatches,
        "reports": reports,
        "budget": {arm: specs[arm].budget for arm in C.ARMS},
        "forbidden_f_or_rho_parameters_in_B_C": forbidden,
        "n_active_global": int(bundle.active_global["n_active"]),
    }


def check_armA_equivalence(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    """Our arm A must reproduce the read-only C1 module on the same state/batch."""
    torch = _torch()
    reference = M.DM.DissectLCRC(bundle.spec, bundle.norm, "C1")
    reference.load_state_dict({k: v.clone() for k, v in canonical_state.items()}, strict=True)
    reference.eval()
    ours = _arm(bundle, "A", canonical_state)
    ours.eval()
    rows = np.arange(0, min(64, int(bundle.caches["train"]["n_molecules"])), dtype=np.int64)
    batch = _one_batch(bundle, rows)
    with torch.no_grad():
        y_ref, aux_ref = reference(batch, return_aux=True)
        y_ours, aux_ours = ours(batch, return_aux=True)
    pred = _close(y_ours.numpy(), y_ref.numpy(), tol=float(_tol(y_ref.numpy()).max()))
    contrib = _close(aux_ours["local_sum"].numpy(), aux_ref["contrib"].numpy(), tol=float(_tol(aux_ref["contrib"].numpy()).max()))
    rho = _close(aux_ours["rho"].numpy(), aux_ref["rho"].numpy(), tol=float(_tol(aux_ref["rho"].numpy()).max()))
    return {
        "check": "armA_equivalence",
        "dataset": bundle.dataset,
        "pass": bool(pred["pass"] and contrib["pass"] and rho["pass"]),
        "prediction": pred,
        "local_sum": contrib,
        "rho": rho,
        "n_rows": int(rows.size),
        "reference_class": "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code/dissect_model.py::DissectLCRC('C1')",
    }


def check_shared_init(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    """Trunk bitwise-identical in A/B/C; u_j bitwise-identical in B/C; reproducible."""
    torch = _torch()
    models = {arm: _arm(bundle, arm, canonical_state) for arm in C.ARMS}
    trunk_hashes = {}
    for arm, model in models.items():
        trunk = {k[len(M.ARM_TRUNK_PREFIX) :]: v for k, v in model.state_dict().items() if k.startswith(M.ARM_TRUNK_PREFIX)}
        trunk_hashes[arm] = C.sha256_state(trunk)
    u_hashes = {
        arm: C.sha256_state(models[arm].global_functions.state_dict()) for arm in ("B", "C")
    }
    # explicit tensor equality, not just equal hashes
    ref_trunk = {k[len(M.ARM_TRUNK_PREFIX) :]: v for k, v in models["B"].state_dict().items() if k.startswith(M.ARM_TRUNK_PREFIX)}
    equality = {}
    for arm in C.ARMS:
        other = {k[len(M.ARM_TRUNK_PREFIX) :]: v for k, v in models[arm].state_dict().items() if k.startswith(M.ARM_TRUNK_PREFIX)}
        equality[arm] = {
            key: bool(torch.equal(other[key], ref_trunk[key])) for key in ref_trunk
        }
    u_equal = {
        key: bool(torch.equal(models["B"].global_functions.state_dict()[key], models["C"].global_functions.state_dict()[key]))
        for key in models["B"].global_functions.state_dict()
    }
    # independent process: rebuild the canonical state and hash the trunk
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import cr_common as C, cr_data as D, cr_models as M;"
        "b = D.build_bundle(%r);"
        "s = M.canonical_c1_state(b.spec, b.norm);"
        "print(M.state_hashes(s)['trunk_hash'])"
    ) % (str(C.CODE_DIR), bundle.dataset)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(C.CODE_DIR))
    subprocess_hash = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
    canonical_trunk = M.state_hashes(canonical_state)["trunk_hash"]
    return {
        "check": "shared_init",
        "dataset": bundle.dataset,
        "pass": bool(
            len(set(trunk_hashes.values())) == 1
            and all(all(v.values()) for v in equality.values())
            and len(set(u_hashes.values())) == 1
            and all(u_equal.values())
            and subprocess_hash == canonical_trunk
        ),
        "canonical_trunk_hash": canonical_trunk,
        "canonical_full_state_hash": M.state_hashes(canonical_state)["full_state_hash"],
        "arm_trunk_hashes": trunk_hashes,
        "arm_trunk_tensor_equality": equality,
        "u_hash_B_C": u_hashes,
        "u_tensor_equality_B_C": u_equal,
        "subprocess_trunk_hash": subprocess_hash,
        "subprocess_returncode": proc.returncode,
        "readout_stream_hashes": {
            arm: C.sha256_state(
                {k: v for k, v in models[arm].state_dict().items() if not k.startswith(M.ARM_TRUNK_PREFIX)}
            )
            for arm in C.ARMS
        },
    }


def check_single_variable_functions() -> dict[str, Any]:
    """``q(0) = 0`` by construction, vectorised == scalar loop, no dead output bias."""
    torch = _torch()
    fns = M.SingleVariableFunctions(5, generator=torch.Generator().manual_seed(C.INIT_STREAM_GLOBAL))
    zero_value = fns(torch.zeros(4, 5)).detach()
    z_nonzero = torch.tensor(
        [[1.0, -1.0, 0.5, -2.0, 2.5], [2.0, -0.5, 0.25, 1.5, -3.0]], dtype=torch.float32
    )
    vectorised = fns(z_nonzero)
    looped = fns.scalar_loop(z_nonzero)
    z_v = z_nonzero.clone().requires_grad_(True)
    grads_v = torch.autograd.grad(fns(z_v).sum(), [z_v] + list(fns.parameters()), retain_graph=True)
    z_l = z_nonzero.clone().requires_grad_(True)
    grads_l = torch.autograd.grad(fns.scalar_loop(z_l).sum(), [z_l] + list(fns.parameters()), retain_graph=True)
    grad_match = all(bool(torch.allclose(a, b, atol=1e-6, rtol=1e-6)) for a, b in zip(grads_v, grads_l))
    # b1 is trainable through the centring at any z != 0 (and is *meant* to have
    # zero derivative exactly at z = 0, which is why z = 0 is not the test point)
    b1_grad = torch.autograd.grad(fns(z_nonzero).sum(), [fns.b1], retain_graph=True)[0]
    value_close = _close(vectorised.detach().numpy(), looped.detach().numpy(), tol=1e-6)
    q_zero_max_abs = float(zero_value.abs().max())
    names = [name for name, _ in fns.named_parameters()]
    return {
        "check": "single_variable_functions",
        "pass": bool(
            q_zero_max_abs <= 1e-6
            and value_close["pass"]
            and grad_match
            and float(b1_grad.abs().max()) > 0.0
            and "bias" not in names
        ),
        "q_of_zero_max_abs": q_zero_max_abs,
        "q_of_zero_note": (
            "q(0) subtracts the identical SiLU(b1) term, so it is zero by construction; "
            "the measured residual is float32 ULP noise from the SiLU kernel, not a bias"
        ),
        "vectorised_vs_loop": value_close,
        "gradient_match_vectorised_vs_loop": bool(grad_match),
        "b1_grad_at_nonzero_z_max_abs": float(b1_grad.abs().max()),
        "has_output_bias": bool(any(name.endswith("b2") or name == "bias" for name in names)),
        "parameter_names": names,
        "params_per_function": int(sum(p.numel() for p in fns.parameters()) // fns.n_functions),
    }


def check_dropout_shared(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    torch = _torch()
    batch = _one_batch(bundle, np.arange(0, 32, dtype=np.int64))
    results = {}
    for arm in C.ARMS:
        model = _arm(bundle, arm, canonical_state)
        calls = {"n": 0}

        def hook(_module, _inputs, _output, _calls=calls):
            _calls["n"] += 1

        handle = model.trunk.drop.register_forward_hook(hook)
        model.train()
        torch.manual_seed(1234)
        with torch.no_grad():
            train_out = model(batch).numpy()
        train_calls = calls["n"]
        handle.remove()
        # eval: dropout off, repeated inference identical
        model.eval()
        with torch.no_grad():
            first = model(batch).numpy()
            second = model(batch).numpy()
        results[arm] = {
            "dropout_calls_per_train_forward": int(train_calls),
            "eval_repeat_identical": bool(np.array_equal(first, second)),
            "train_mode_differs_from_eval": bool(not np.array_equal(first, train_out)),
        }
    return {
        "check": "dropout_shared",
        "dataset": bundle.dataset,
        "pass": bool(all(v["dropout_calls_per_train_forward"] == 1 for v in results.values())
                     and all(v["eval_repeat_identical"] for v in results.values())),
        "per_arm": results,
        "note": "one dropout(0.10) call on h per forward; the same h feeds every local head of the arm",
    }


def check_roundtrip(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    torch = _torch()
    mean, std = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    worst = 0.0
    for split in C.SPLITS:
        raw = bundle.targets_raw[split]
        y_std = (raw - mean) / std
        worst = max(worst, float(np.abs(y_std * std + mean - raw).max()))
    detail: dict[str, Any] = {}
    ok = worst < 1e-9
    for arm in ("B", "C"):
        model = _arm(bundle, arm, canonical_state)
        model.eval()
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        out = T.evaluate(model, bundle, "valid", "cpu")
        ref_rmse = C.rmse(np.full_like(bundle.targets_raw["valid"], mean), bundle.targets_raw["valid"])
        detail[arm] = {
            "pred_raw_max_abs_minus_train_mean": float(np.abs(out["pred_raw"] - mean).max()),
            "rmse_raw": float(out["rmse_raw"]),
            "expected_constant_rmse": float(ref_rmse),
            "match": bool(abs(out["rmse_raw"] - ref_rmse) < 1e-9),
        }
        ok = ok and detail[arm]["match"] and detail[arm]["pred_raw_max_abs_minus_train_mean"] < 1e-9
    return {
        "check": "roundtrip_and_zero_readout",
        "dataset": bundle.dataset,
        "pass": bool(ok),
        "target_roundtrip_max_abs_error": worst,
        "zero_readout": detail,
        "note": "de-standardisation happens exactly once: pred_raw = mu + sigma * pred_std",
    }


def check_gradients(bundle: D.Bundle, canonical_state, *, n_batches: int = 8, batch_size: int = 64) -> dict[str, Any]:
    torch = _torch()
    train_rows = int(bundle.caches["train"]["n_molecules"])
    starts = np.linspace(0, max(0, train_rows - batch_size), num=n_batches).astype(np.int64)
    report: dict[str, Any] = {}
    ok = True
    for arm in ("B", "C"):
        model = _arm(bundle, arm, canonical_state)
        model.train()
        max_abs_z = np.zeros(int(bundle.active_global["n_active"]))
        grad_names = {"local": [], "global_functions": [], "context": [], "b_global": []}
        finite = True
        for start in starts:
            rows = np.arange(start, min(start + batch_size, train_rows), dtype=np.int64)
            batch = _one_batch(bundle, rows)
            model.zero_grad(set_to_none=True)
            pred, parts = model(batch, return_aux=True)
            loss = torch.nn.functional.mse_loss(pred, batch["y"].view(-1))
            loss.backward()
            max_abs_z = np.maximum(max_abs_z, np.abs(parts["z_active"].detach().numpy()).max(axis=0))
            for name, param in model.named_parameters():
                if param.grad is None:
                    finite = False
                    continue
                if not torch.isfinite(param.grad).all():
                    finite = False
                g = float(param.grad.abs().max())
                if name.startswith(M.ARM_TRUNK_PREFIX):
                    continue
                if name.startswith("global_functions") or name == "b_global":
                    grad_names["global_functions"].append(g)
                elif name.startswith("context_functions") or name == "w_context":
                    grad_names["context"].append(g)
                else:
                    grad_names["local"].append(g)
        summary = {
            "local_max_grad_over_batches": float(np.max(grad_names["local"]) if grad_names["local"] else 0.0),
            "u_max_grad_over_batches": float(np.max(grad_names["global_functions"]) if grad_names["global_functions"] else 0.0),
            "a_max_grad_over_batches": float(np.max(grad_names["context"]) if grad_names["context"] else 0.0),
            "finite": bool(finite),
            "active_columns_with_zero_z_everywhere": [int(bundle.active_global["active_indices"][i]) for i, v in enumerate(max_abs_z) if v == 0.0],
        }
        summary["pass"] = bool(
            summary["finite"]
            and summary["local_max_grad_over_batches"] > 0.0
            and summary["u_max_grad_over_batches"] > 0.0
            and (arm == "B" or summary["a_max_grad_over_batches"] > 0.0)
        )
        ok = ok and summary["pass"]
        report[arm] = summary
    return {
        "check": "gradients",
        "dataset": bundle.dataset,
        "pass": bool(ok),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "per_arm": report,
        "note": (
            "eight fixed Train batches distinguish a genuinely unreachable path from a z==0 column "
            "inside a single batch"
        ),
    }


def check_invariance(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    torch = _torch()
    mods = MOD
    split = "valid"
    rows = np.arange(0, min(16, int(bundle.caches[split]["n_molecules"])), dtype=np.int64)

    # (a) eval batching: whole split vs chunk 7 vs one molecule at a time
    model = _arm(bundle, "C", canonical_state)
    model.eval()
    base = T.evaluate(model, bundle, split, "cpu")
    chunked: list[np.ndarray] = []
    order, _row_ids, _targets = T.eval_rows(bundle, split)
    with torch.no_grad():
        for start in range(0, order.size, 7):
            batch = bundle.collate(bundle.caches[split], order[start : start + 7])
            chunked.append(model(batch).numpy())
    batched = np.concatenate(chunked)
    batch_close = _close(batched, base["pred_std"], tol=1e-5)

    # (b) molecule isolation: one molecule alone vs inside a mixed batch
    solo = []
    with torch.no_grad():
        for r in rows:
            batch = bundle.collate(bundle.caches[split], np.asarray([int(r)], dtype=np.int64))
            solo.append(float(model(batch).numpy().reshape(-1)[0]))
        mixed_batch = bundle.collate(bundle.caches[split], rows)
        mixed = model(mixed_batch).numpy().reshape(-1)
    solo_close = _close(np.asarray(solo), mixed, tol=1e-5)

    # (c) atom renumbering through a *correct* raw -> cache rebuild
    rng = np.random.default_rng(20261116)
    perm_report = []
    with torch.no_grad():
        for r in rows[:4]:
            base_batch = bundle.collate(bundle.caches[split], np.asarray([int(r)], dtype=np.int64))
            base_pred = float(model(base_batch).numpy().reshape(-1)[0])
            raw = mods["sq_data"].lipo_raw_row(bundle.lipo_raw, int(bundle.global_row_ids[split][r]))
            n_atoms = int(np.asarray(raw["atom_class"]).size)
            perm = rng.permutation(n_atoms)
            inv = np.argsort(perm)
            new_row = dict(raw)
            new_row["atom_class"] = np.asarray(raw["atom_class"])[perm]
            new_row["atom_charge"] = np.asarray(raw["atom_charge"])[perm]
            new_row["atom_h"] = np.asarray(raw["atom_h"])[perm]
            new_row["edge_index"] = np.stack(
                [inv[np.asarray(raw["edge_index"][0])], inv[np.asarray(raw["edge_index"][1])]]
            )
            graph = mods["adapters"].lipo_row_to_graph(new_row, int(r), bundle.spec)
            cache = mods["lcrc_data"].build_cache([graph], bundle.spec, split="permuted")
            batch = bundle.collate(cache, np.asarray([0], dtype=np.int64))
            perm_pred = float(model(batch).numpy().reshape(-1)[0])
            perm_report.append(
                {
                    "local_row": int(r),
                    "global_row_id": int(bundle.global_row_ids[split][r]),
                    "n_atoms": n_atoms,
                    "base_pred_std": base_pred,
                    "permuted_pred_std": perm_pred,
                    "abs_diff": abs(base_pred - perm_pred),
                    "pass": bool(abs(base_pred - perm_pred) <= 1e-5 * max(1.0, abs(base_pred))),
                }
            )
    return {
        "check": "invariance",
        "dataset": bundle.dataset,
        "pass": bool(batch_close["pass"] and solo_close["pass"] and all(r["pass"] for r in perm_report)),
        "eval_batching": batch_close,
        "molecule_isolation": solo_close,
        "atom_renumbering": perm_report,
        "note": "renumbering goes through raw row -> adapters -> build_cache, not a tensor permutation",
    }


def _reconstruct_ledger(blocks: Sequence[Mapping[str, np.ndarray]], arm: str) -> dict[str, np.ndarray]:
    pred = np.concatenate([b["pred_std"] for b in blocks])
    intercept = np.concatenate([b["intercept"] for b in blocks])
    sum_base = np.concatenate([b["sum_base"] for b in blocks])
    u_sum = np.concatenate([b["u_sum"] for b in blocks])
    if arm == "C":
        interaction = np.concatenate([b["interaction"] for b in blocks])
        total = intercept + sum_base + u_sum + interaction.sum(axis=1)
    else:
        interaction = np.zeros((pred.size, 0))
        total = intercept + sum_base + u_sum
    return {
        "pred": pred,
        "intercept": intercept,
        "sum_base": sum_base,
        "u_sum": u_sum,
        "interaction": interaction,
        "total": total,
    }


def check_ledger_closure(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    torch = _torch()
    report: dict[str, Any] = {}
    ok = True
    for arm in ("B", "C"):
        model = _arm(bundle, arm, canonical_state)
        model.eval()
        with torch.no_grad():
            out = T.evaluate(model, bundle, "valid", "cpu", with_ledger=True)
        rec = _reconstruct_ledger(out["ledger"], arm)
        closure = _close(rec["total"], rec["pred"], tol=float(_tol(rec["pred"]).max()))
        base_sums = np.concatenate(
            [
                np.bincount(b["atom_batch"], weights=b["base_root"], minlength=int(b["n_graphs"][0]))
                for b in out["ledger"]
            ]
        )
        base_check = _close(base_sums, rec["sum_base"], tol=1e-5)
        detail = {"closure": closure, "per_root_base_sum": base_check}
        if arm == "C":
            root_inter = np.concatenate(
                [
                    np.stack(
                        [
                            np.bincount(b["atom_batch"], weights=b["interaction_root"][:, k], minlength=int(b["n_graphs"][0]))
                            for k in range(b["interaction_root"].shape[1])
                        ],
                        axis=1,
                    )
                    for b in out["ledger"]
                ]
            )
            kgraph = np.abs(root_inter - rec["interaction"])
            detail["per_root_interaction"] = {
                "max_abs_diff": float(kgraph.max()),
                "pass": bool(float(kgraph.max()) <= 1e-5),
            }
        report[arm] = detail
        ok = ok and all(v["pass"] for v in detail.values())
    return {
        "check": "ledger_closure",
        "dataset": bundle.dataset,
        "pass": bool(ok),
        "per_arm": report,
        "note": "verified on the canonical initial state: closure is an implementation property, not a trained-weight property",
    }


def check_scale_equivalence(bundle: D.Bundle, canonical_state) -> dict[str, Any]:
    """Multiply a_k output by 2 and c_k output by 1/2: predictions must not move."""
    torch = _torch()
    model = _arm(bundle, "C", canonical_state)
    model.eval()
    order, _row_ids, _targets = T.eval_rows(bundle, "valid")
    chunk = order[: min(32, order.size)]
    batch = bundle.collate(bundle.caches["valid"], chunk)
    with torch.no_grad():
        pred_a, parts_a = model(batch, return_aux=True)
        base = {
            "pred": pred_a.numpy(),
            "per_root_interaction": parts_a["interaction_root"].numpy(),
            "a_values": parts_a["a_values"].numpy(),
            "context_root": parts_a["context_root"].numpy(),
        }
        # a_k * 2  <=> w2 of the context functions doubles
        model.context_functions.w2.data.mul_(2.0)
        # c_k / 2  <=> the bias-free context head weights halve
        model.w_context.data.mul_(0.5)
        pred_b, parts_b = model(batch, return_aux=True)
        scaled = {
            "pred": pred_b.numpy(),
            "per_root_interaction": parts_b["interaction_root"].numpy(),
            "a_values": parts_b["a_values"].numpy(),
            "context_root": parts_b["context_root"].numpy(),
        }
    pred_close = _close(scaled["pred"], base["pred"], tol=1e-5)
    root_close = _close(scaled["per_root_interaction"], base["per_root_interaction"], tol=1e-5)
    a_doubled = _close(scaled["a_values"], 2.0 * base["a_values"], tol=1e-5)
    c_halved = _close(scaled["context_root"], 0.5 * base["context_root"], tol=1e-5)
    return {
        "check": "scale_equivalence",
        "dataset": bundle.dataset,
        "pass": bool(pred_close["pass"] and root_close["pass"] and a_doubled["pass"] and c_halved["pass"]),
        "prediction": pred_close,
        "per_root_interaction": root_close,
        "a_doubled": a_doubled,
        "c_halved": c_halved,
        "n_rows": int(chunk.size),
        "note": "a_k and c_k individual parameter norms are therefore not importance measures",
    }


def check_checkpoint_reload(bundle: D.Bundle, canonical_state, tmp_dir) -> dict[str, Any]:
    torch = _torch()
    tmp_dir = C.ensure_dir(tmp_dir)
    report: dict[str, Any] = {}
    ok = True
    for arm in ("B", "C"):
        model = _arm(bundle, arm, canonical_state)
        model.eval()
        with torch.no_grad():
            out = T.evaluate(model, bundle, "valid", "cpu", with_ledger=True)
        path = tmp_dir / f"init_{arm}.pt"
        C.atomic_torch_save(
            path,
            {
                "round_id": C.ROUND_ID,
                "dataset": bundle.dataset,
                "arm": arm,
                "epoch": 0,
                "state_dict": model.state_dict(),
                "metrics": {"note": "initialisation diagnostic, never a scientific checkpoint"},
                "target_scaler": dict(bundle.scaler),
                "norm": {"fingerprint": bundle.extra["norm_fingerprint"]},
                "active_global_indices": list(bundle.active_global["active_indices"]),
                "context_fields": list(C.CONTEXT_FIELDS),
                "context_indices": [D.field_index(bundle.spec, name) for name in C.CONTEXT_FIELDS],
                "signature": "init-check",
                "init_audit": {
                    "hidden": None if arm == "A" else M.arm_specs(bundle)[arm].hidden,
                    "n_active_global": int(bundle.active_global["n_active"]),
                },
                "created_at": "init-check",
            },
        )
        reloaded, payload = T.load_arm_from_checkpoint(bundle, path, device="cpu")
        with torch.no_grad():
            again = T.evaluate(reloaded, bundle, "valid", "cpu", with_ledger=True)
        pred_close = _close(again["pred_std"], out["pred_std"], tol=1e-6)
        rec_a = _reconstruct_ledger(out["ledger"], arm)
        rec_b = _reconstruct_ledger(again["ledger"], arm)
        ledger_close = _close(rec_b["total"], rec_a["total"], tol=1e-6)
        report[arm] = {
            "reloaded_matches_stored": bool(payload.get("reloaded_matches_stored")),
            "prediction": pred_close,
            "ledger": ledger_close,
            "pass": bool(payload.get("reloaded_matches_stored") and pred_close["pass"] and ledger_close["pass"]),
        }
        ok = ok and report[arm]["pass"]
    return {
        "check": "checkpoint_reload",
        "dataset": bundle.dataset,
        "pass": bool(ok),
        "per_arm": report,
    }


# ---------------------------------------------------------------------------
# frozen GINE reference reuse gates
# ---------------------------------------------------------------------------


def check_gine_reference(dataset: str) -> dict[str, Any]:
    mods = MOD
    path = C.SQL_RESULTS / dataset / "method_G.json"
    payload = C.read_json(path)
    cache = mods["sq_data"].load_cache(dataset)
    scaler = C.target_scaler(np.asarray(cache["labels"])[mods["sq_data"].split_indices(cache, "train")])
    predictions = payload["selected_predictions"]
    selected = payload["selected"]
    per_split: dict[str, Any] = {}
    ok = True
    for split in C.SPLITS:
        rows = mods["sq_data"].split_indices(cache, split)
        block = predictions[split]
        pred_raw = np.asarray(block["pred_raw"], dtype=np.float64)
        pred_std = np.asarray(block["pred_std"], dtype=np.float64)
        target_raw = np.asarray(block["target_raw"], dtype=np.float64)
        row_ids = np.asarray(block["row_ids"], dtype=np.int64)
        labels = np.asarray(cache["labels"], dtype=np.float64)[rows]
        rmse = C.rmse(pred_raw, target_raw)
        mae = C.mae(pred_raw, target_raw)
        entry = {
            "n": int(row_ids.size),
            "row_ids_equal_official_split": bool(np.array_equal(row_ids, rows)),
            "target_raw_max_abs_diff_vs_cache_labels": float(np.abs(target_raw - labels).max()),
            "destandardisation_roundtrip_max_abs_diff": float(
                np.abs(pred_std * scaler["std"] + scaler["mean"] - pred_raw).max()
            ),
            "rmse_recomputed": rmse,
            "mae_recomputed": mae,
            "rmse_reported": float(selected[f"{split}_rmse_raw"]),
            "mae_reported": float(selected[f"{split}_mae_raw"]),
        }
        entry["rmse_abs_diff"] = abs(entry["rmse_recomputed"] - entry["rmse_reported"])
        entry["mae_abs_diff"] = abs(entry["mae_recomputed"] - entry["mae_reported"])
        entry["pass"] = bool(
            entry["row_ids_equal_official_split"]
            and entry["target_raw_max_abs_diff_vs_cache_labels"] <= 1e-6
            and entry["destandardisation_roundtrip_max_abs_diff"] <= 1e-6
            and entry["rmse_abs_diff"] <= 1e-6
            and entry["mae_abs_diff"] <= 1e-6
        )
        ok = ok and entry["pass"]
        per_split[split] = entry
    curves = payload["per_lr"][str(payload["selected_peak_lr"])]["curves"]
    epochs = [int(row["epoch"]) for row in curves]
    best_index = min(range(len(curves)), key=lambda i: (curves[i]["valid_rmse_raw"], i))
    recomputed = {
        "n_epochs": len(curves),
        "epochs_are_1_to_360": bool(epochs == list(range(1, 361))),
        "has_three_split_rmse": bool(all(k in curves[0] for k in ("train_rmse_raw", "valid_rmse_raw", "test_rmse_raw"))),
        "has_three_split_mae": bool(all(k in curves[0] for k in ("train_mae_raw", "valid_mae_raw", "test_mae_raw"))),
        "best_epoch": int(curves[best_index]["epoch"]),
        "best_valid_rmse_raw": float(curves[best_index]["valid_rmse_raw"]),
        "best_test_rmse_raw": float(curves[best_index]["test_rmse_raw"]),
        "final": {
            "epoch": int(curves[-1]["epoch"]),
            "valid_rmse_raw": float(curves[-1]["valid_rmse_raw"]),
            "test_rmse_raw": float(curves[-1]["test_rmse_raw"]),
        },
        "last20": {
            f"mean_last20_{split}_rmse_raw": float(np.mean([row[f"{split}_rmse_raw"] for row in curves[-20:]]))
            for split in C.SPLITS
        },
    }
    matches = {
        "best_epoch": recomputed["best_epoch"] == int(selected["epoch"]),
        "best_valid": abs(recomputed["best_valid_rmse_raw"] - float(selected["valid_rmse_raw"])) <= 1e-9,
        "best_test": abs(recomputed["best_test_rmse_raw"] - float(selected["test_rmse_raw"])) <= 1e-9,
        "final_round_selected": recomputed["final"]["epoch"] == int(payload["final_round_selected"]["epoch"]),
        "last20": all(
            abs(recomputed["last20"][k] - float(payload["last20_selected"][k])) <= 1e-9 for k in recomputed["last20"]
        ),
    }
    source = C.SQL_CODE / "sq_gine.py"
    source_text = source.read_text(encoding="utf-8")
    standardised = "standardised" in source_text or "standardized" in source_text
    ok = ok and all(matches.values()) and recomputed["epochs_are_1_to_360"] and recomputed["has_three_split_rmse"]
    return {
        "check": "gine_reference_reuse",
        "dataset": dataset,
        "pass": bool(ok),
        "status": "REUSED_FROZEN_REFERENCE",
        "path": str(path.relative_to(C.REPO_ROOT)),
        "sha256": C.sha256_file(path),
        "source": str(source.relative_to(C.REPO_ROOT)),
        "source_sha256": C.sha256_file(source),
        "source_declares_standardised_target": bool(standardised),
        "standardisation_fix_commit": "44300426",
        "selected_peak_lr": payload["selected_peak_lr"],
        "n_parameters": payload["n_parameters"],
        "per_split": per_split,
        "curve_recomputation": recomputed,
        "summary_matches": matches,
        "target_scaler": {k: scaler[k] for k in ("mean", "std")},
        "note": (
            "reused as a frozen performance coordinate only; no G trajectory was retrained "
            "because every reuse gate passed"
        ),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_all(
    *,
    datasets: Sequence[str] = C.DATASETS,
    tmp_dir=None,
    include_gine: bool = True,
) -> dict[str, Any]:
    tmp_dir = C.ensure_dir(tmp_dir or (C.ANALYSIS_DIR / "_checkpoints"))
    checks: list[dict[str, Any]] = []
    for dataset in datasets:
        bundle = D.build_bundle(dataset)
        canonical = M.canonical_c1_state(bundle.spec, bundle.norm)
        checks.append(check_data(bundle))
        checks.append(check_parameters(bundle, canonical))
        checks.append(check_armA_equivalence(bundle, canonical))
        checks.append(check_shared_init(bundle, canonical))
        checks.append(check_dropout_shared(bundle, canonical))
        checks.append(check_roundtrip(bundle, canonical))
        checks.append(check_gradients(bundle, canonical))
        checks.append(check_invariance(bundle, canonical))
        checks.append(check_ledger_closure(bundle, canonical))
        checks.append(check_scale_equivalence(bundle, canonical))
        checks.append(check_checkpoint_reload(bundle, canonical, tmp_dir / dataset))
        if include_gine:
            checks.append(check_gine_reference(dataset))
    checks.append(check_single_variable_functions())
    failed = [c["check"] + ":" + str(c.get("dataset")) for c in checks if not c.get("pass")]
    payload = {
        "round_id": C.ROUND_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": list(datasets),
        "n_checks": len(checks),
        "all_pass": bool(not failed),
        "failed": failed,
        "checks": checks,
        "environment": C.environment_snapshot(),
    }
    return payload


__all__ = [
    "check_data",
    "check_parameters",
    "check_armA_equivalence",
    "check_shared_init",
    "check_single_variable_functions",
    "check_dropout_shared",
    "check_roundtrip",
    "check_gradients",
    "check_invariance",
    "check_ledger_closure",
    "check_scale_equivalence",
    "check_checkpoint_reload",
    "check_gine_reference",
    "run_all",
]
