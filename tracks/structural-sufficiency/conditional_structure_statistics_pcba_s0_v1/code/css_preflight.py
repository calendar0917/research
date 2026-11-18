"""GPU / CPU preflight: the measurement gates that must pass before the main run.

Everything here is *engineering* verification, not a result:

1. canonical-initial-state audit (E/L/S parameter counts, tensor-identical
   initial states, zero query, one shared public fingerprint),
2. structural audit of real batches (S1 = 2 x bonds, per-root S2 count identity,
   middle / physical bond ids, closure, same-molecule membership),
3. mathematical gates: ``q=0`` makes E = L = S; equal keys inside a root and a
   single-occurrence root both force E = L; the two-occurrence non-exchange
   witness gives E = tanh(1)/sqrt(2) and L = 0,
4. atom/bond renumbering and batch reordering invariance; isolated atoms; no-S2
   molecules; triangle closure,
5. task-chunk equivalence for eval logits *and* Train loss / gradients,
6. gradient reachability at 1 / 3 / 10 / 100 real Train steps (query first, keys
   once the query leaves zero, plus values / root / graph MLP / head) with the
   recorded query norms and gate statistics,
7. checkpoint continuation equivalence (interrupted == uninterrupted, both on a
   deliberately truncated step budget so the check is cheap),
8. official OGB AP versus an independent scikit-learn recomputation,
9. full Train / Validation / Test evaluation row counts and label sources,
10. a measured cost projection with >= 20% headroom.

Profiling state is discarded: the formal run reloads the canonical initial state
and the formal dropout generator.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import css_common as C
import css_data as D
import css_model as M
import css_train as T

SMOKE_STEPS = 12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _batch(dev: D.DevTask, rows, device: torch.device) -> dict[str, Any]:
    batch = D.collate_batch(dev, rows)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def heaviest_batches(dev: D.DevTask, plan, *, n: int = 3) -> list[list[int]]:
    """Batches with the largest real object volume (atoms + S1 + S2), from the
    frozen plan itself so the probe measures the actual workload."""
    cache = dev.task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    s2 = D.s2_counts_per_molecule(dev)
    scored = []
    for record in plan[:2]:
        for rows in record["batches"]:
            r = np.asarray(rows, dtype=np.int64)
            atoms = int((atom_ptr[r + 1] - atom_ptr[r]).sum())
            bonds = int((bond_ptr[r + 1] - bond_ptr[r]).sum())
            scored.append((int(s2[r].sum()) + 2 * bonds + atoms, rows))
    scored.sort(key=lambda x: -x[0])
    return [list(rows) for _score, rows in scored[: int(n)]]


def _toy_batch(
    *,
    n_atoms: int,
    bond_index: Sequence[Sequence[int]],
    s1: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """A hand-built batch with no S2 occurrences (used by the algebra gates)."""
    bi = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2) if len(bond_index) else np.zeros((0, 2), np.int64)
    s1 = np.asarray(s1, dtype=np.int64).reshape(-1, 3) if len(s1) else np.zeros((0, 3), np.int64)
    n_bonds = int(bi.shape[0])
    counts = np.bincount(s1[:, 0], minlength=n_atoms).astype(np.int64) if len(s1) else np.zeros(n_atoms, np.int64)

    def t(arr, dtype):
        return torch.as_tensor(np.ascontiguousarray(np.asarray(arr, dtype=dtype)))

    return {
        "atom_feat": t(np.zeros((n_atoms, 51), np.float32), np.float32),
        "atom_scalar": t(np.zeros((n_atoms, 9), np.float32), np.float32),
        "atom_batch": t(np.zeros(n_atoms, np.int64), np.int64),
        "atom_counts": t([n_atoms], np.int64),
        "bond_feat": t(np.zeros((n_bonds, 13), np.float32), np.float32),
        "bond_index": t(bi, np.int64),
        "bond_batch": t(np.zeros(n_bonds, np.int64), np.int64),
        "global": t(np.zeros((1, 42), np.float32), np.float32),
        "s1_root": t(s1[:, 0] if len(s1) else [], np.int64),
        "s1_end": t(s1[:, 1] if len(s1) else [], np.int64),
        "s1_bond": t(s1[:, 2] if len(s1) else [], np.int64),
        "s1_counts": t(counts, np.int64),
        "s2_root": t([], np.int64),
        "s2_middle": t([], np.int64),
        "s2_end": t([], np.int64),
        "s2_bond1": t([], np.int64),
        "s2_bond2": t([], np.int64),
        "s2_closure": t([], np.float32),
        "s2_counts": t(np.zeros(n_atoms, np.int64), np.int64),
        "n_atoms": t([n_atoms], np.int64),
        "n_bonds": t([n_bonds], np.int64),
        "n_s1": t([int(len(s1))], np.int64),
        "n_s2": t([0], np.int64),
    }


def _vector_batch(n_atoms: int, s1: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Two-atom toy graph: one bond 0-1, two directed occurrences."""
    return _toy_batch(n_atoms=n_atoms, bond_index=[[0, 1]], s1=s1)


def _toy_model(arm: str, norm) -> M.CSSModel:
    model, _ = M.build_model(arm, norm, n_tasks=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def _z(model: M.CSSModel, enc: Mapping[str, torch.Tensor], batch, q) -> float:
    sums = model.root_sums(enc, batch)
    z1, _z2 = model.condition_and_aggregate(enc, batch, q, sums)
    return float(z1[0, 0, 0])


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_initial_states(dev: D.DevTask) -> dict[str, Any]:
    audit = M.audit_initial_states(dev.task.norm)
    return {"name": "initial_state_and_parameters", **audit}


def gate_structures(dev: D.DevTask, plan) -> dict[str, Any]:
    atom_ptr = np.asarray(dev.task.cache["atom_ptr"])
    n_atoms = atom_ptr[1:] - atom_ptr[:-1]
    n_paths = D.s2_counts_per_molecule(dev)
    row_sets: list[tuple[str, list[int]]] = [
        ("train_head", [int(x) for x in dev.train_rows[:32]]),
        ("largest_batch", [int(x) for x in max(plan[0]["batches"], key=len)]),
        ("smallest_molecules", [int(x) for x in np.argsort(n_atoms, kind="stable")[:16]]),
        ("valid_head", [int(x) for x in dev.valid_rows[:32]]),
        ("test_head", [int(x) for x in dev.test_rows[:32]]),
    ]
    for i, rows in enumerate(heaviest_batches(dev, plan, n=2)):
        row_sets.append((f"heaviest_{i}", rows))
    isolated = [int(r) for r in np.flatnonzero(n_atoms == 1)[:16]]
    if isolated:
        row_sets.append(("isolated_atoms", isolated))
    no_s2 = [int(r) for r in np.flatnonzero(n_paths == 0)[:16]]
    if no_s2:
        row_sets.append(("no_s2_molecules", no_s2))
    closed = [int(r) for r in np.flatnonzero(n_atoms == 3)[:16]]
    if closed:
        row_sets.append(("three_atom_molecules", closed))

    reports = {}
    failures = []
    for label, rows in row_sets:
        batch = D.collate_batch(dev, rows)
        rep = D.audit_batch_structures(dev, batch)
        reports[label] = {k: v for k, v in rep.items() if k != "problems"}
        reports[label]["nonzero_problems"] = {k: v for k, v in rep["problems"].items() if v}
        if not rep["passed"]:
            failures.append({"batch": label, "problems": reports[label]["nonzero_problems"]})
    return {
        "name": "structure_objects",
        "passed": not failures,
        "failures": failures,
        "n_batches": len(row_sets),
        "batches": reports,
        "checks": [
            "S1 == 2 x physical bonds, root-major, endpoints match the physical bond",
            "per-root S2 count == sum_{w in N(v)} (deg(w) - 1)",
            "S2 middle / physical bond ids / endpoints consistent and same-molecule",
            "S2 atoms pairwise distinct (simple paths, multiplicity preserved)",
            "closure recomputed from adjacency equals the frozen path_closure",
            "S1/S2 pointer arrays consistent with the per-atom counts",
        ],
    }


def gate_initial_function_equality(dev: D.DevTask, plan) -> dict[str, Any]:
    rows = list(plan[0]["batches"][0])
    batch = _batch(dev, rows, torch.device("cpu"))
    outs = {}
    for arm in C.ARM_ORDER:
        model, _ = M.build_model(arm, dev.task.norm)
        model.eval()
        with torch.no_grad():
            outs[arm] = model(batch)
    diffs = {
        "E_L": float((outs["E"] - outs["L"]).abs().max()),
        "E_S": float((outs["E"] - outs["S"]).abs().max()),
    }
    return {
        "name": "zero_query_equality",
        "passed": bool(max(diffs.values()) <= 1e-6),
        "E_equals_L_bitwise_at_init": bool(torch.equal(outs["E"], outs["L"])),
        "E_equals_S_bitwise_at_init": bool(torch.equal(outs["E"], outs["S"])),
        "max_abs_diff": diffs,
        "rows": len(rows),
    }


def gate_non_exchange_witness(norm) -> dict[str, Any]:
    """Two occurrences, scores (+1, -1), values (+1, -1), n = 2."""
    # atom 0 (root) - atom 1: S1 = [(0,1,e0), (1,0,e0)] -> root 0 has ONE
    # occurrence; to get two occurrences at the root we need two bonds:
    batch = _toy_batch(
        n_atoms=3,
        bond_index=[[0, 1], [0, 2]],
        s1=[[0, 1, 0], [0, 2, 1], [1, 0, 0], [2, 0, 1]],
    )
    q = torch.zeros(1, 1, 16)
    q[0, 0, 0] = 1.0
    keys = torch.zeros(4, 16)
    values = torch.zeros(4, 32)
    keys[0, 0] = 4.0   # score +1
    values[0, 0] = 1.0
    keys[1, 0] = -4.0  # score -1
    values[1, 0] = -1.0
    enc = {
        "a": torch.ones(3, 192),
        "b": torch.ones(2, 192),
        "k1": keys,
        "v1": values,
        "k2": torch.zeros(0, 16),
        "v2": torch.zeros(0, 32),
    }
    values_by_arm = {}
    for arm in ("E", "L"):
        model = _toy_model(arm, norm)
        values_by_arm[arm] = _z(model, enc, batch, q)
    expected = math.tanh(1.0) / math.sqrt(2.0)
    return {
        "name": "non_exchange_witness",
        "passed": bool(abs(values_by_arm["E"] - expected) <= 1e-6 and abs(values_by_arm["L"]) <= 1e-7),
        "E_value": values_by_arm["E"],
        "L_value": values_by_arm["L"],
        "E_expected_tanh1_over_sqrt2": expected,
        "roles": "two occurrences at one root, scores (+1,-1), values (+1,-1), n=2",
        "note": "E changes with the aggregation order; L cannot (n=2, mean key 0, value sum 0)",
    }


def gate_degenerate_equalities(norm) -> dict[str, Any]:
    """Equal keys inside a root, and a single occurrence, both force E = L."""
    out: dict[str, Any] = {}
    # (a) equal keys inside a root, different values
    batch = _toy_batch(
        n_atoms=3,
        bond_index=[[0, 1], [0, 2]],
        s1=[[0, 1, 0], [0, 2, 1], [1, 0, 0], [2, 0, 1]],
    )
    keys = torch.zeros(4, 16)
    keys[0, 0] = 3.0
    keys[1, 0] = 3.0
    values = torch.zeros(4, 32)
    values[0, 0] = 1.0
    values[1, 0] = -2.0
    enc = {
        "a": torch.ones(3, 192),
        "b": torch.ones(2, 192),
        "k1": keys,
        "v1": values,
        "k2": torch.zeros(0, 16),
        "v2": torch.zeros(0, 32),
    }
    q = torch.zeros(1, 1, 16)
    q[0, 0, 0] = 1.0
    pair = {arm: _z(_toy_model(arm, norm), enc, batch, q) for arm in ("E", "L")}
    out["equal_keys"] = {"E": pair["E"], "L": pair["L"], "abs_diff": abs(pair["E"] - pair["L"])}

    # (b) one occurrence at the root
    single_batch = _vector_batch(2, [[0, 1, 0], [1, 0, 0]])
    enc_single = {
        "a": torch.ones(2, 192),
        "b": torch.ones(1, 192),
        "k1": torch.zeros(2, 16),
        "v1": torch.zeros(2, 32),
        "k2": torch.zeros(0, 16),
        "v2": torch.zeros(0, 32),
    }
    enc_single["k1"][0, 0] = 2.5
    enc_single["v1"][0, 0] = 1.75
    single = {arm: _z(_toy_model(arm, norm), enc_single, single_batch, q) for arm in ("E", "L")}
    out["single_occurrence"] = {
        "E": single["E"],
        "L": single["L"],
        "abs_diff": abs(single["E"] - single["L"]),
    }
    return {
        "name": "degenerate_E_L_equalities",
        "passed": all(v["abs_diff"] <= 1e-6 for v in out.values()),
        "cases": out,
    }


def gate_invariances(dev: D.DevTask, plan) -> dict[str, Any]:
    rows = list(plan[0]["batches"][0])
    device = torch.device("cpu")
    model, _ = M.build_model("E", dev.task.norm)
    model.eval()
    batch = _batch(dev, rows, device)
    rng = np.random.default_rng(C.SEED)
    with torch.no_grad():
        base = model(batch)
        perm = rng.permutation(len(rows))
        rows2 = [rows[i] for i in perm.tolist()]
        batch2 = _batch(dev, rows2, device)
        reordered = model(batch2)
        # reordered[i] is base[perm[i]]
        reorder_diff = float((reordered[np.argsort(perm)] - base).abs().max())
        ren = _renumber(batch, np.random.default_rng(C.SEED + 1))
        renumber_diff = float((base - model(ren)).abs().max())
    atom_ptr = np.asarray(dev.task.cache["atom_ptr"])
    n_atoms = atom_ptr[1:] - atom_ptr[:-1]
    n_paths = D.s2_counts_per_molecule(dev)
    cases = {}
    for label, sel in (
        ("no_s2", np.flatnonzero(n_paths == 0)[:16]),
        ("isolated_atoms", np.flatnonzero(n_atoms == 1)[:16]),
        ("smallest", np.argsort(n_atoms, kind="stable")[:16]),
        ("largest", np.argsort(-n_atoms, kind="stable")[:16]),
    ):
        if sel.size == 0:
            continue
        b = _batch(dev, [int(x) for x in sel], device)
        with torch.no_grad():
            out = model(b)
        cases[label] = {
            "n_molecules": int(sel.size),
            "finite": bool(torch.isfinite(out).all()),
            "n_s2_min": int(b["s2_counts"].min()),
            "n_atoms_min": int(b["atom_counts"].min()),
            "n_atoms_max": int(b["atom_counts"].max()),
        }
    return {
        "name": "permutation_and_degenerate_graphs",
        "passed": bool(
            reorder_diff <= 1e-6 and renumber_diff <= 1e-6 and all(v["finite"] for v in cases.values())
        ),
        "batch_reorder_max_abs_diff": reorder_diff,
        "atom_bond_renumber_max_abs_diff": renumber_diff,
        "degenerate_cases": cases,
    }


def _renumber(batch: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Relabel atoms and physical bonds inside each molecule, rebuild indices."""
    atom_counts = batch["atom_counts"].numpy()
    n_atoms = int(atom_counts.sum())
    atom_perm = np.empty(n_atoms, dtype=np.int64)
    start = 0
    for c in atom_counts.tolist():
        atom_perm[start : start + c] = start + rng.permutation(c)
        start += c
    bi = batch["bond_index"].numpy()
    bond_counts = batch["bond_counts"].numpy()
    bond_perm = np.empty(int(bi.shape[0]), dtype=np.int64)
    start = 0
    for c in bond_counts.tolist():
        bond_perm[start : start + c] = start + rng.permutation(c)
        start += c
    bond_src = np.argsort(bond_perm, kind="stable")  # new bond position -> old bond id
    remap = bond_perm.copy()  # old bond id -> new bond id
    new_bi = np.sort(atom_perm[bi[bond_src]], axis=1)
    atom_src = np.argsort(atom_perm, kind="stable")  # new atom position -> old atom id
    out = dict(batch)
    out["atom_feat"] = batch["atom_feat"][torch.as_tensor(atom_src)]
    out["atom_scalar"] = batch["atom_scalar"][torch.as_tensor(atom_src)]
    out["bond_feat"] = batch["bond_feat"][torch.as_tensor(bond_src)]
    out["bond_index"] = torch.as_tensor(new_bi)
    out["s1_root"] = torch.as_tensor(atom_perm[batch["s1_root"].numpy()])
    out["s1_end"] = torch.as_tensor(atom_perm[batch["s1_end"].numpy()])
    out["s1_bond"] = torch.as_tensor(remap[batch["s1_bond"].numpy()])
    out["s2_root"] = torch.as_tensor(atom_perm[batch["s2_root"].numpy()])
    out["s2_middle"] = torch.as_tensor(atom_perm[batch["s2_middle"].numpy()])
    out["s2_end"] = torch.as_tensor(atom_perm[batch["s2_end"].numpy()])
    out["s2_bond1"] = torch.as_tensor(remap[batch["s2_bond1"].numpy()])
    out["s2_bond2"] = torch.as_tensor(remap[batch["s2_bond2"].numpy()])
    # per-atom occurrence counts are indexed by root atom -> permute with atoms
    out["s1_counts"] = batch["s1_counts"][torch.as_tensor(atom_src)]
    out["s2_counts"] = batch["s2_counts"][torch.as_tensor(atom_src)]
    # the per-root occurrence order is not part of the model: both aggregations
    # are order invariant, so the (renumbered) lists are kept as they are.
    return out


def gate_task_chunk(dev: D.DevTask, plan) -> dict[str, Any]:
    rows = list(plan[0]["batches"][0])
    device = torch.device("cpu")
    batch = _batch(dev, rows, device)
    y = torch.as_tensor(dev.training_targets(rows), dtype=torch.float32)

    def _eval_logits(chunk: int, arm: str = "E") -> np.ndarray:
        model, _ = M.build_model(arm, dev.task.norm, task_chunk=chunk)
        model.eval()
        with torch.no_grad():
            return model(batch).numpy()

    refs = {arm: _eval_logits(C.N_TASKS, arm) for arm in C.ARM_ORDER}
    eval_diffs = {}
    for arm in C.ARM_ORDER:
        for chunk in (16, 8, 4):
            eval_diffs[f"{arm}_chunk_{chunk}"] = float(np.abs(_eval_logits(chunk, arm) - refs[arm]).max())

    grad_diff = {}
    loss_diff = {}
    for arm in ("E", "L"):
        grads = {}
        losses = {}
        for chunk in (16, 4):
            model, _ = M.build_model(arm, dev.task.norm, task_chunk=chunk)
            gen = torch.Generator().manual_seed(C.DROPOUT_GENERATOR_SEED)
            mask = T.dropout_mask(
                int(batch["n_atoms"]), C.N_TASKS, 64, generator=gen, device=device, p=model.dropout_p
            )
            model.zero_grad(set_to_none=True)
            loss = C.MET.masked_bce_loss(model(batch, dropout_mask=mask), y)
            loss.backward()
            losses[chunk] = float(loss)
            grads[chunk] = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
        worst = 0.0
        for name in grads[16]:
            a, b = grads[16][name], grads[4][name]
            if a is None and b is None:
                continue
            worst = max(worst, float((a - b).abs().max()))
        grad_diff[arm] = worst
        loss_diff[arm] = abs(losses[16] - losses[4])
    return {
        "name": "task_chunk_equivalence",
        "passed": bool(
            max(eval_diffs.values()) <= 1e-6
            and max(loss_diff.values()) <= 1e-6
            and max(grad_diff.values()) <= 1e-6
        ),
        "eval_max_abs_diff": eval_diffs,
        "train_loss_abs_diff": loss_diff,
        "grad_max_abs_diff": grad_diff,
        "tolerance": {"cpu": 1e-6, "cuda": 1e-4},
    }


def gate_label_isolation(dev: D.DevTask) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        dev.training_targets(np.asarray(dev.test_rows[:2], dtype=np.int64))
        checks["test_row_training_access_refused"] = False
    except RuntimeError:
        checks["test_row_training_access_refused"] = True
    y_tr = dev.training_targets(dev.train_rows[:8])
    checks["train_targets_are_frozen"] = bool(
        np.array_equal(
            np.nan_to_num(y_tr, nan=-9.0),
            np.nan_to_num(np.asarray(dev.task.labels[dev.train_rows[:8]]), nan=-9.0),
        )
    )
    checks["frozen_artifact_test_all_nan"] = bool(not np.isfinite(dev.task.labels[dev.test_rows]).any())
    checks["development_test_observed_entries"] = int(np.isfinite(dev.labels_dev[dev.test_rows]).sum())
    keys = set(D.collate_batch(dev, dev.train_rows[:4]).keys())
    checks["no_label_tensor_in_model_input"] = not any("label" in k or k == "y" for k in keys)
    checks["normalisation_uses_train_rows_only"] = (
        int(dev.task.norm["fit_rows"]) == int(C.N_TRAIN) and dev.task.norm["fit_split"] == "train_only"
    )
    passed = (
        checks["test_row_training_access_refused"]
        and checks["train_targets_are_frozen"]
        and checks["frozen_artifact_test_all_nan"]
        and checks["development_test_observed_entries"] > 0
        and checks["no_label_tensor_in_model_input"]
        and checks["normalisation_uses_train_rows_only"]
    )
    return {"name": "label_isolation", "passed": bool(passed), "checks": checks}


def gate_gradients(dev: D.DevTask, plan, device: torch.device) -> dict[str, Any]:
    model, _ = M.build_model("E", dev.task.norm)
    model.to(device)
    optimizer, _ = T.build_optimizer(model)
    gen = T.make_dropout_generator(device)
    rows = list(plan[0]["batches"][0])
    batch = _batch(dev, rows, device)
    y = torch.as_tensor(dev.training_targets(rows), dtype=torch.float32, device=device)
    probes = {1, 3, 10, 100}
    records: list[dict[str, Any]] = []
    global_step = 0
    while global_step < max(probes):
        out = T.train_step(
            model,
            optimizer,
            batch,
            y,
            global_step=global_step,
            clip=float(C.RECIPE["gradient_clip"]),
            generator=gen,
        )
        global_step += 1
        if global_step in probes:
            with torch.no_grad():
                q = model.build_query(batch)[0]
                diag = M.ForwardDiagnostics()
                model.eval()
                model(batch, diagnostics=diag)
                model.train()
            records.append(
                {
                    "step": global_step,
                    "loss": out["loss"],
                    "grad_norm": out["grad_norm"],
                    "module_grad_norms": out["module_grad_norms"],
                    "q0_norm": float(model.q0.detach().norm()),
                    "qt_norm": float(model.qt.detach().norm()),
                    "Wc_norm": float(model.Wc.weight.detach().norm()),
                    "query_norm": float(q.norm(dim=-1).mean()),
                    "query_task_spread": float(q.norm(dim=-1).mean(dim=0).std()),
                    "value_weight_norm": float(model.value1.weight.detach().norm()),
                    "root_weight_norm": float(model.root_static.weight.detach().norm()),
                    "head_weight_norm": float(model.out.weight.detach().norm()),
                    "gate_summary": diag.summary(),
                }
            )
    first = records[0]["module_grad_norms"]
    last = records[-1]["module_grad_norms"]
    checks = {
        "query_gradient_at_step1": first["query"] > 0.0,
        "keys_gradient_at_step1_is_zero_by_construction": first["keys"] == 0.0,
        "keys_gradient_after_query_left_zero": last["keys"] > 0.0,
        "value_gradient_present": last["values"] > 0.0,
        "root_gradient_present": (last["root_static"] > 0.0 and last["root_stat"] > 0.0),
        "head_gradient_present": last["task_output"] > 0.0,
        "graph_mlp_gradient_present": last["graph_mlp"] > 0.0,
        "event_encoders_gradient_present": (last["s1_event"] > 0.0 and last["s2_event"] > 0.0),
        "query_moved_from_zero": records[-1]["q0_norm"] > 0.0 and records[-1]["qt_norm"] > 0.0,
        "gate_actually_moved": abs(records[-1]["gate_summary"]["gate_mean"] - 1.0) > 1e-6,
    }
    return {
        "name": "gradient_reachability",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "probes": records,
        "device": str(device),
    }


def gate_step_and_memory(dev: D.DevTask, plan, device: torch.device, *, steps: int = 30) -> dict[str, Any]:
    model, _ = M.build_model("E", dev.task.norm)
    model.to(device)
    optimizer, _ = T.build_optimizer(model)
    gen = T.make_dropout_generator(device)
    batches = [plan[0]["batches"][i] for i in range(min(steps, len(plan[0]["batches"])))]
    heaviest = heaviest_batches(dev, plan, n=1)[0]
    cache = dev.task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    r = np.asarray(heaviest, dtype=np.int64)
    # measured from the collated batch itself: the cache's path_ptr describes a
    # different object family, so it is never used for the reported volume
    probe = _batch(dev, heaviest, torch.device("cpu"))
    heaviest_volume = {
        "n_molecules": int(r.size),
        "n_atoms": int((atom_ptr[r + 1] - atom_ptr[r]).sum()),
        "n_bonds": int(probe["n_bonds"]),
        "n_s1": int(probe["n_s1"]),
        "n_s2": int(probe["n_s2"]),
    }
    del probe
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for i, rows in enumerate(batches):
        batch = _batch(dev, rows, device)
        y = torch.as_tensor(dev.training_targets(rows), dtype=torch.float32, device=device)
        t0 = time.perf_counter()
        T.train_step(
            model,
            optimizer,
            batch,
            y,
            global_step=i,
            clip=float(C.RECIPE["gradient_clip"]),
            generator=gen,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        for key in ("atom_feat", "bond_feat", "atom_batch"):
            del batch[key]
    steady = float(np.median(times[5:])) if len(times) > 5 else float(np.median(times))
    batch = _batch(dev, heaviest, device)
    y = torch.as_tensor(dev.training_targets(heaviest), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    T.train_step(
        model,
        optimizer,
        batch,
        y,
        global_step=len(batches),
        clip=float(C.RECIPE["gradient_clip"]),
        generator=gen,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    heaviest_ms = (time.perf_counter() - t0) * 1000.0
    peak = C.gpu_peak(device)
    return {
        "name": "step_and_memory",
        "passed": bool(steady > 0.0),
        "step_ms_median": steady,
        "step_ms_mean": float(np.mean(times)),
        "step_ms_p90": float(np.percentile(times, 90)),
        "n_steps": len(times),
        "heaviest_batch_step_ms": heaviest_ms,
        "heaviest_batch": heaviest_volume,
        "peak_allocated_bytes": peak["peak_allocated_bytes"],
        "peak_reserved_bytes": peak["peak_reserved_bytes"],
        "peak_reserved_mb": peak["peak_reserved_bytes"] / 1e6 if device.type == "cuda" else 0.0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


def gate_eval_cost(dev: D.DevTask, device: torch.device, *, n_rows: int = 4096) -> dict[str, Any]:
    model, _ = M.build_model("E", dev.task.norm)
    model.to(device)
    timings = {}
    for split in ("train", "valid", "test"):
        rows = dev.rows[split][: int(n_rows)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits = M.predict_logits(
            model, lambda r: D.collate_batch(dev, r), rows, device, batch_size=C.BATCH_SIZE
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
        timings[split] = {
            "rows": int(rows.size),
            "seconds": float(seconds),
            "ms_per_row": float(1000.0 * seconds / max(int(rows.size), 1)),
            "finite": bool(np.isfinite(logits).all()),
        }
    full_rows = {k: int(v.size) for k, v in dev.rows.items()}
    projected = {split: timings[split]["ms_per_row"] * full_rows[split] / 1000.0 for split in timings}
    return {
        "name": "evaluation_cost",
        "passed": all(v["finite"] for v in timings.values()),
        "subset": timings,
        "full_rows": full_rows,
        "projected_full_split_seconds": projected,
        "projected_three_split_seconds": float(sum(projected.values())),
        "device": str(device),
    }


def gate_resume(dev: D.DevTask, device: torch.device, out_dir: Path) -> dict[str, Any]:
    """Interrupted == uninterrupted on a truncated (cheap) step budget."""
    base = Path(out_dir) / "_preflight_resume"
    fields = ("train_ap", "valid_ap", "test_ap", "train_eval_bce", "valid_bce", "test_bce")
    plain = T.train(
        dev,
        arm="L",
        out_dir=base / "plain",
        device=device,
        epochs=2,
        resume=False,
        log_every=0,
        steps_per_epoch=SMOKE_STEPS,
    )
    T.train(
        dev,
        arm="L",
        out_dir=base / "resumed",
        device=device,
        epochs=1,
        resume=False,
        log_every=0,
        steps_per_epoch=SMOKE_STEPS,
    )
    cont = T.train(
        dev,
        arm="L",
        out_dir=base / "resumed",
        device=device,
        epochs=2,
        resume=True,
        log_every=0,
        steps_per_epoch=SMOKE_STEPS,
    )
    diffs = {f: abs(float(plain["curve"][-1][f]) - float(cont["curve"][-1][f])) for f in fields}
    return {
        "name": "resume_equivalence",
        "passed": bool(max(diffs.values()) <= 1e-6),
        "max_abs_diff": float(max(diffs.values())),
        "per_field": diffs,
        "steps_per_epoch": int(SMOKE_STEPS),
        "note": "1 epoch + resume to 2 epochs vs 2 uninterrupted epochs, both with a truncated step budget",
    }


def gate_full_split_eval(dev: D.DevTask, device: torch.device) -> dict[str, Any]:
    """Every split fully evaluated from the correct label source + AP agreement."""
    model, _ = M.build_model("E", dev.task.norm)
    model.to(device)
    splits: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    failures: list[str] = []
    expected = {"train": C.N_TRAIN, "valid": C.N_VALID, "test": C.N_TEST}
    for split in ("train", "valid", "test"):
        rep = T.evaluate_split(model, dev, split, device)
        y = rep["labels"]
        splits[split] = {
            "rows": rep["rows"],
            "evaluated_rows": rep["evaluated_rows"],
            "observed_entries": rep["observed_entries"],
            "positive_entries": int((np.asarray(y) == 1.0).sum()),
            "ap": rep["ap"],
            "bce": rep["bce"],
            "n_tasks_scored": rep["n_tasks_scored"],
            "logits_finite": bool(np.isfinite(rep["logits"]).all()),
            "prediction_seconds": rep["seconds"],
        }
        ag = C.MET.ap_agreement(rep["labels"], M.sigmoid(rep["logits"]))
        agreement[split] = {k: ag[k] for k in ("official_ap", "independent_ap", "abs_delta", "n_tasks_scored")}
        if rep["evaluated_rows"] != expected[split]:
            failures.append(f"{split} evaluated {rep['evaluated_rows']} != official {expected[split]}")
        if rep["observed_entries"] == 0:
            failures.append(f"{split} has no observed labels (inherited empty evaluation)")
        if not np.isfinite(rep["ap"]):
            failures.append(f"{split} AP is not finite")
        if ag["abs_delta"] > 1e-6:
            failures.append(f"{split} official AP and sklearn differ by {ag['abs_delta']}")
    return {
        "name": "full_split_evaluation",
        "passed": not failures,
        "failures": failures,
        "splits": splits,
        "ap_agreement": agreement,
        "label_view": dev.label_view_audit(),
    }


def gate_epoch0_shared(dev: D.DevTask, device: torch.device, *, n_rows: int = 4096) -> dict[str, Any]:
    """The three arms start as one function: logits equal on real rows."""
    rows = np.concatenate([dev.valid_rows[:n_rows], dev.test_rows[:n_rows]])
    outs = {}
    for arm in C.ARM_ORDER:
        model, _ = M.build_model(arm, dev.task.norm)
        model.to(device)
        outs[arm] = M.predict_logits(model, lambda r: D.collate_batch(dev, r), rows, device)
    diffs = {
        "E_L": float(np.abs(outs["E"] - outs["L"]).max()),
        "E_S": float(np.abs(outs["E"] - outs["S"]).max()),
    }
    return {
        "name": "epoch0_shared_function",
        "passed": bool(max(diffs.values()) <= 1e-6),
        "max_abs_diff": diffs,
        "n_rows": int(rows.size),
        "splits": ["valid[:n]", "test[:n]"],
        "note": "the shared initial-function evaluation is cached here, not repeated per arm",
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def cost_projection(preflight: Mapping[str, Any]) -> dict[str, Any]:
    step = preflight["step_and_memory"]
    evaluation = preflight["evaluation_cost"]
    seconds_per_epoch = (
        float(step["step_ms_median"]) * C.STEPS_PER_EPOCH / 1000.0
        + float(evaluation["projected_three_split_seconds"])
    )
    hours_per_arm = seconds_per_epoch * C.TOTAL_EPOCHS / 3600.0
    with_margin = hours_per_arm * 1.2
    return {
        "schema": "css-pcba-s0-v1-cost-projection-v1",
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "epochs": int(C.TOTAL_EPOCHS),
        "step_ms_median": float(step["step_ms_median"]),
        "heaviest_batch_step_ms": float(step["heaviest_batch_step_ms"]),
        "peak_reserved_mb": float(step["peak_reserved_mb"]),
        "projected_full_eval_seconds": float(evaluation["projected_three_split_seconds"]),
        "projected_epoch_seconds": float(seconds_per_epoch),
        "projected_arm_hours": float(hours_per_arm),
        "projected_arm_hours_with_20pct_margin": float(with_margin),
        "projected_two_arm_parallel_wall_hours": float(with_margin),
        "projected_two_arm_gpu_hours": float(2.0 * with_margin),
        "budget": dict(C.BUDGET),
        "within_gpu_soft_budget": bool(2.0 * with_margin <= C.BUDGET["gpu_soft_hours"]),
        "within_gpu_hard_budget": bool(2.0 * with_margin <= C.BUDGET["gpu_hard_hours"]),
        "within_wall_soft_budget": bool(with_margin + 1.0 <= C.BUDGET["wall_soft_hours"]),
        "measured_preflight_gpu_seconds": float(preflight.get("preflight_gpu_seconds", 0.0) or 0.0),
    }


def run_preflight(
    *,
    device_name: str = "cuda:0",
    out_dir: Path | str | None = None,
    gates: Sequence[str] | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    dev = D.load_task(with_cpc=True)
    plan = D.make_plan40(dev)
    out_dir = Path(out_dir) if out_dir else C.ANALYSIS
    tmp = out_dir / "_preflight_work"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    default_gates = [
        "initial_state",
        "structures",
        "initial_function",
        "witness",
        "degenerate",
        "invariances",
        "chunk",
        "labels",
        "gradients",
        "step_memory",
        "eval_cost",
        "resume",
        "full_eval",
        "epoch0",
    ]
    selected = list(gates) if gates else default_gates
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def _run(name: str, fn) -> None:
        if name not in selected:
            return
        t = time.perf_counter()
        results[name] = fn()
        timings[name] = time.perf_counter() - t
        C.log(f"[preflight] {name}: passed={results[name].get('passed')} ({timings[name]:.1f}s)")

    _run("initial_state", lambda: gate_initial_states(dev))
    _run("structures", lambda: gate_structures(dev, plan))
    _run("initial_function", lambda: gate_initial_function_equality(dev, plan))
    _run("witness", lambda: gate_non_exchange_witness(dev.task.norm))
    _run("degenerate", lambda: gate_degenerate_equalities(dev.task.norm))
    _run("invariances", lambda: gate_invariances(dev, plan))
    _run("chunk", lambda: gate_task_chunk(dev, plan))
    _run("labels", lambda: gate_label_isolation(dev))
    _run("gradients", lambda: gate_gradients(dev, plan, device))
    _run("step_memory", lambda: gate_step_and_memory(dev, plan, device))
    _run("eval_cost", lambda: gate_eval_cost(dev, device))
    _run("resume", lambda: gate_resume(dev, device, tmp))
    _run("full_eval", lambda: gate_full_split_eval(dev, device))
    _run("epoch0", lambda: gate_epoch0_shared(dev, device))

    elapsed = time.perf_counter() - t0
    gpu_seconds = elapsed if device.type == "cuda" else 0.0
    failures = [k for k, v in results.items() if not bool(v.get("passed", True))]
    payload = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "device": str(device),
        "host": C.host_facts(),
        "gates_run": sorted(results),
        "passed": not failures,
        "failing_gates": failures,
        "elapsed_seconds": float(elapsed),
        "preflight_gpu_seconds": float(gpu_seconds),
        "gate_seconds": timings,
        "gates": results,
        "identities": {
            "feature_spec_hash": C.FEATURE_SPEC_HASH,
            "norm_fingerprint": C.NORM_FINGERPRINT,
            "split_partition_sha256": C.SPLIT_PARTITION_SHA256,
            "cache_sha256": C.CACHE_SHA256,
            "label_artifact_sha256": C.LABEL_ARTIFACT_SHA256,
            "plan_global_sha256": C.PLAN["global_plan_sha256"],
            "plan_consumed40_prefix_sha256": C.PLAN["consumed40_prefix_sha256"],
        },
        "label_view": dev.label_view_audit(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "created_utc": C.utcnow(),
    }
    if {"step_and_memory", "evaluation_cost"} <= set(results):
        _synthetic = {
            "step_and_memory": results["step_and_memory"],
            "evaluation_cost": results["evaluation_cost"],
            "preflight_gpu_seconds": gpu_seconds,
        }
        payload["cost_projection"] = cost_projection(_synthetic)
    C.atomic_write_json(out_dir / "preflight.json", payload)
    if "cost_projection" in payload:
        C.atomic_write_json(out_dir / "cost_projection.json", payload["cost_projection"])
    return payload


__all__ = [
    "cost_projection",
    "gate_degenerate_equalities",
    "gate_epoch0_shared",
    "gate_eval_cost",
    "gate_full_split_eval",
    "gate_gradients",
    "gate_invariances",
    "gate_initial_function_equality",
    "gate_initial_states",
    "gate_label_isolation",
    "gate_non_exchange_witness",
    "gate_resume",
    "gate_step_and_memory",
    "gate_structures",
    "gate_task_chunk",
    "run_preflight",
]
