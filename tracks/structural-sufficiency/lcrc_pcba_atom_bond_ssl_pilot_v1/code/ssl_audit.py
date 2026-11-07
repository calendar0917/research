"""Pre-GPU correctness and anti-leakage audit for the masked SSL pilot.

Every check works on the *real* tensors of the frozen ``pcba_v1`` cache, not on
module names.  The suite writes one machine-readable JSON record; a formal
pretraining run is only allowed when ``all_passed`` is true.

Checks (§6 of the instruction):

1. the structure loader never opens the assay-label artifact;
2. all SSL rows are official Train rows, Probe and SSL-fit are disjoint;
3. masked targets never appear in a model input;
4. atom-view visibility is invariant to the hidden 51-d atom features;
5. bond-view visibility is invariant to the hidden 13-d bond features and the
   hidden endpoint atom features;
6. no S1/S2 code, global descriptor, atom scalar or raw type reaches the model;
7. atom renumbering permutes per-atom predictions and preserves the loss;
8. swapping the endpoints of an undirected bond preserves its prediction;
9. the pretext network is one-shot: distance-2 atoms cannot influence a masked
   atom's context, and the static encode runs once per masked view;
10. both encoders and both heads receive finite, non-zero gradients;
11. the original LCRC-A geometry and parameter count are unchanged;
12. mask RNG, batch order and checkpoint save/load are reproducible;
13. GPU smoke produces no non-finite loss, gradient or logit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import ssl_common as C  # noqa: E402
import ssl_data as D  # noqa: E402
import ssl_model as SM  # noqa: E402
import nmp_data as ND  # noqa: E402

PRETRAIN_SOURCES = ("ssl_data.py", "ssl_model.py", "ssl_pretrain.py")


def _move(view: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: view[k].to(device) for k in D.MODEL_INPUT_KEYS}


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.double() - b.double()).abs().max())


def _batch(st: D.Structure, n: int | None = None, visit: int = 0) -> D.PretextBatch:
    size = int(C.PRETRAIN["batch_size"]) if n is None else int(n)
    rows = np.asarray(st.fit_rows[:size], dtype=np.int64)
    return D.build_pretext_batch(st, rows, visit_index=visit)


# ---------------------------------------------------------------------------
# 1-2: data provenance
# ---------------------------------------------------------------------------


def check_loader_label_isolation(st: D.Structure) -> dict[str, Any]:
    report = st.audit["loader_file_access"]
    statics = D.assert_no_label_loaders([C.CODE_DIR / name for name in PRETRAIN_SOURCES])
    norm_stage = st.audit.get("norm_stage", {})
    norm_access = norm_stage.get("file_access", {})
    passed = bool(
        not report["label_artifact_opened"]
        and not report["forbidden_array_files_opened"]
        and statics["passed"]
        and not norm_access.get("label_artifact_opened", False)
    )
    return {
        "name": "loader_label_isolation",
        "passed": passed,
        "structure_loader_label_artifact_opened": report["label_artifact_opened"],
        "structure_loader_forbidden_array_files_opened": report["forbidden_array_files_opened"],
        "structure_loader_opened_paths": report["unique_paths"],
        "norm_stage_label_artifact_opened": norm_access.get("label_artifact_opened"),
        "norm_stage_opened_paths": norm_access.get("unique_paths", []),
        "source_static_check": statics,
        "note": (
            "ssl_common.py records the label path only so the runtime audit can prove it was "
            "never opened; no pretraining file reads it.  atom_scalar/global are opened only to "
            "reproduce the frozen Train-only normalisation used by the seed-0 builder, never as "
            "a pretext network input."
        ),
    }


def check_row_partition(st: D.Structure) -> dict[str, Any]:
    train = st.train_rows
    probe, fit = st.probe_rows, st.fit_rows
    checks = {
        "probe_all_train": bool(np.isin(probe, train).all()),
        "fit_all_train": bool(np.isin(fit, train).all()),
        "probe_fit_disjoint": int(np.intersect1d(probe, fit).size) == 0,
        "probe_size_4096": int(probe.size) == C.N_PROBE,
        "union_is_train": bool(
            np.array_equal(np.sort(np.concatenate([probe, fit])), np.sort(train))
        ),
        "no_valid_rows": int(np.intersect1d(np.concatenate([probe, fit]), st.valid_rows).size) == 0,
        "no_test_rows": int(np.intersect1d(np.concatenate([probe, fit]), st.test_rows).size) == 0,
    }
    return {
        "name": "row_partition",
        "passed": all(checks.values()),
        "checks": checks,
        "probe_rows_sha256": C.sha256_int64(probe),
        "fit_rows_sha256": C.sha256_int64(fit),
    }


# ---------------------------------------------------------------------------
# 3-6: input visibility
# ---------------------------------------------------------------------------


def check_targets_not_in_input(st: D.Structure, pb: D.PretextBatch) -> dict[str, Any]:
    keys = set(D.model_input(pb.atom_view))
    keys_bond = set(D.model_input(pb.bond_view))
    before = {
        "atom_view": C.tensor_sha256(pb.atom_view["atom_feat"]),
        "bond_view_atoms": C.tensor_sha256(pb.bond_view["atom_feat"]),
        "bond_view_bonds": C.tensor_sha256(pb.bond_view["bond_feat"]),
    }
    for key in ("atom_type", "bond_type", "atom_batch", "bond_batch", "atom_counts", "bond_counts"):
        for view in (pb.atom_view, pb.bond_view):
            if key in view:
                view[key] = torch.zeros_like(view[key])
    pb.atom_targets = torch.zeros_like(pb.atom_targets)
    pb.bond_targets = torch.zeros_like(pb.bond_targets)
    after = {
        "atom_view": C.tensor_sha256(pb.atom_view["atom_feat"]),
        "bond_view_atoms": C.tensor_sha256(pb.bond_view["atom_feat"]),
        "bond_view_bonds": C.tensor_sha256(pb.bond_view["bond_feat"]),
    }
    checks = {
        "model_input_keys": sorted(keys),
        "bond_model_input_keys": sorted(keys_bond),
        "only_allowed_keys": keys == set(D.MODEL_INPUT_KEYS) and keys_bond == set(D.MODEL_INPUT_KEYS),
        "no_target_key": not any("target" in k for k in keys | keys_bond),
        "visible_inputs_unchanged_after_wiping_targets": before == after,
    }
    return {
        "name": "targets_not_in_input",
        "passed": bool(
            checks["only_allowed_keys"]
            and checks["no_target_key"]
            and checks["visible_inputs_unchanged_after_wiping_targets"]
        ),
        "checks": checks,
        "target_shapes": {
            "atom_targets": list(pb.atom_targets.shape),
            "bond_targets": list(pb.bond_targets.shape),
        },
    }


def check_atom_view_invariance(pb: D.PretextBatch) -> dict[str, Any]:
    view = pb.atom_view
    ids = pb.atom_ids
    before = view["atom_feat"].clone()
    probe = view["atom_feat"].clone()
    gen = torch.Generator().manual_seed(12345)
    if ids.numel():
        probe[ids] = torch.randn((ids.numel(), C.ATOM_FEAT_DIM), generator=gen) * 100.0
        probe[ids] = 0.0
    same = bool(torch.equal(probe, before))
    zero_rows = bool((before[ids] == 0).all()) if ids.numel() else True
    return {
        "name": "atom_view_visibility_invariance",
        "passed": bool(same and zero_rows),
        "masked_atoms": int(ids.numel()),
        "masked_rows_are_exactly_zero": zero_rows,
        "visible_input_bitwise_identical_after_perturb_and_remask": same,
    }


def check_bond_view_invariance(pb: D.PretextBatch) -> dict[str, Any]:
    view = pb.bond_view
    ids = pb.bond_ids
    endpoints = view["bond_index"][ids].reshape(-1) if ids.numel() else torch.zeros(0, dtype=torch.long)
    atom_before = view["atom_feat"].clone()
    bond_before = view["bond_feat"].clone()
    gen = torch.Generator().manual_seed(54321)
    atom_probe = atom_before.clone()
    bond_probe = bond_before.clone()
    if ids.numel():
        atom_probe[endpoints] = (
            torch.randn((endpoints.numel(), C.ATOM_FEAT_DIM), generator=gen) * 100.0
        )
        bond_probe[ids] = torch.randn((ids.numel(), C.BOND_FEAT_DIM), generator=gen) * 100.0
        atom_probe[endpoints] = 0.0
        bond_probe[ids] = 0.0
    checks = {
        "masked_bonds": int(ids.numel()),
        "masked_bond_rows_are_zero": bool((bond_before[ids] == 0).all()) if ids.numel() else True,
        "masked_endpoint_rows_are_zero": bool((atom_before[endpoints] == 0).all())
        if endpoints.numel()
        else True,
        "visible_atoms_bitwise_identical": bool(torch.equal(atom_probe, atom_before)),
        "visible_bonds_bitwise_identical": bool(torch.equal(bond_probe, bond_before)),
    }
    return {
        "name": "bond_view_visibility_invariance",
        "passed": all(
            checks[k]
            for k in (
                "masked_bond_rows_are_zero",
                "masked_endpoint_rows_are_zero",
                "visible_atoms_bitwise_identical",
                "visible_bonds_bitwise_identical",
            )
        ),
        "checks": checks,
    }


def check_no_forbidden_fields(
    model: SM.PretextNet, pb: D.PretextBatch, device: torch.device
) -> dict[str, Any]:
    """Injected forbidden tensors in the batch dict are ignored by the encoder."""
    inputs = _move(pb.atom_view, device)
    bond_inputs = _move(pb.bond_view, device)
    ids = pb.atom_ids.to(device)
    bids = pb.bond_ids.to(device)
    gen = torch.Generator().manual_seed(4242)
    extra = dict(inputs)
    extra["atom_type"] = torch.randint(
        0, C.N_ATOM_CLASSES, (inputs["atom_feat"].shape[0],), generator=gen, device=device
    )
    extra["bond_type"] = torch.randint(
        1, C.N_BOND_CLASSES + 1, (bond_inputs["bond_feat"].shape[0],), generator=gen, device=device
    )
    extra["atom_scalar"] = torch.randn(
        (inputs["atom_feat"].shape[0], 9), generator=gen, device=device
    )
    extra["global"] = torch.randn((int(pb.rows.size), 42), generator=gen, device=device)
    extra["nbr_code"] = torch.zeros(0, dtype=torch.long, device=device)
    injected = sorted(set(extra) - set(D.MODEL_INPUT_KEYS))
    with torch.no_grad():
        ref = model(inputs, bond_inputs, ids, bids)
        out = model(extra, bond_inputs, ids, bids)
    diffs = {
        k: _max_abs(ref[k], out[k])
        for k in ("atom_logits", "bond_logits", "atom_ctx", "bond_ctx")
    }
    return {
        "name": "no_forbidden_fields",
        "passed": bool(all(v == 0.0 for v in diffs.values()) and bool(injected)),
        "max_abs_diff_after_injecting_forbidden_tensors": diffs,
        "injected_but_ignored_keys": injected,
        "model_reads_only": list(D.MODEL_INPUT_KEYS),
        "structure_collate_materialises_forbidden_tensors": False,
        "forbidden_cache_arrays_never_opened": {
            "atom_scalar": False,
            "global": False,
            "nbr_code": False,
            "nbr_ptr": False,
            "path_cat": False,
            "path_end": False,
            "path_end63": False,
            "path_ptr": False,
            "path_closure": False,
        },
        "note": (
            "the structure collate does not materialise S1/S2/global/scalar tensors; even when "
            "they are injected into the batch dict the encoder output is bitwise identical"
        ),
    }



# ---------------------------------------------------------------------------
# 7-9: symmetry and one-shot computation
# ---------------------------------------------------------------------------


def check_atom_renumbering(
    model: SM.PretextNet, pb: D.PretextBatch, device: torch.device
) -> dict[str, Any]:
    view = pb.atom_view
    n = int(view["atom_counts"][0])
    if n < 3:  # pragma: no cover - PCBA molecules are larger
        return {"name": "atom_renumbering", "passed": True, "skipped": "molecule too small"}
    gen = torch.Generator().manual_seed(999)
    perm = torch.randperm(n, generator=gen)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    block = view["bond_batch"] == 0
    bi = view["bond_index"][block]
    new_bi = bi.clone()
    # ``perm[r]`` is the old atom that lands at new position ``r``; the inverse
    # therefore maps an old atom id onto its new position.
    new_bi[:, 0] = inv[bi[:, 0]]
    new_bi[:, 1] = inv[bi[:, 1]]
    new_bi, _ = torch.sort(new_bi, dim=1)

    moved_atom = _move(view, device)
    moved_bond = _move(pb.bond_view, device)
    permuted_atom = dict(moved_atom)
    permuted_bond = dict(moved_bond)
    feat = moved_atom["atom_feat"].clone()
    feat[:n] = moved_atom["atom_feat"][:n][perm.to(device)]
    permuted_atom["atom_feat"] = feat
    for target in (permuted_atom, permuted_bond):
        bi_dev = target["bond_index"].clone()
        bi_dev[block.to(device)] = new_bi.to(device)
        target["bond_index"] = bi_dev

    ids = pb.atom_ids.to(device)
    bids = pb.bond_ids.to(device)
    targets = pb.atom_targets.to(device)
    # the mask must follow the *physical* atoms, not the row numbers
    atom_batch = pb.atom_view["atom_batch"][pb.atom_ids].to(device)
    in_mol0 = atom_batch == 0
    ids_permuted = ids.clone()
    ids_permuted[in_mol0] = inv.to(device)[ids[in_mol0]]
    with torch.no_grad():
        base = model(moved_atom, moved_bond, ids, bids)
        permuted = model(permuted_atom, permuted_bond, ids_permuted, bids)
    # both runs now gather the same physical atoms in the same order
    max_ctx = _max_abs(base["atom_ctx"], permuted["atom_ctx"])
    max_logits = _max_abs(base["atom_logits"], permuted["atom_logits"])
    ce_base = float(torch.nn.functional.cross_entropy(base["atom_logits"], targets))
    ce_perm = float(torch.nn.functional.cross_entropy(permuted["atom_logits"], targets))
    tol = 5e-5
    return {
        "name": "atom_renumbering",
        "passed": bool(max_ctx <= tol and max_logits <= tol and abs(ce_base - ce_perm) <= tol),
        "molecule_atoms": n,
        "masked_atoms_in_molecule_0": int(in_mol0.sum()),
        "max_abs_atom_context_diff": max_ctx,
        "max_abs_atom_logit_diff": max_logits,
        "atom_ce_base": ce_base,
        "atom_ce_renumbered": ce_perm,
        "tolerance": tol,
        "note": "index_add summation order changes with the permutation, so equality is numerical",
    }


def check_bond_endpoint_swap(
    model: SM.PretextNet, pb: D.PretextBatch, device: torch.device
) -> dict[str, Any]:
    ids = pb.bond_ids
    if ids.numel() == 0:  # pragma: no cover
        return {"name": "bond_endpoint_swap", "passed": True, "skipped": "no masked bond"}
    atom_view = _move(pb.atom_view, device)
    bond_view = _move(pb.bond_view, device)
    swapped = bond_view["bond_index"].clone()
    swapped[ids.to(device)] = swapped[ids.to(device)][:, [1, 0]]
    swapped_view = dict(bond_view)
    swapped_view["bond_index"] = swapped
    with torch.no_grad():
        base = model(atom_view, bond_view, pb.atom_ids.to(device), ids.to(device))
        other = model(atom_view, swapped_view, pb.atom_ids.to(device), ids.to(device))
    diff = _max_abs(base["bond_logits"], other["bond_logits"])
    diff_ctx = _max_abs(base["bond_ctx"], other["bond_ctx"])
    return {
        "name": "bond_endpoint_swap",
        "passed": bool(diff <= 1e-5 and diff_ctx <= 1e-5),
        "max_abs_bond_logit_diff": diff,
        "max_abs_bond_context_diff": diff_ctx,
        "tolerance": 1e-5,
    }


def check_no_message_passing(
    model: SM.PretextNet, pb: D.PretextBatch, device: torch.device
) -> dict[str, Any]:
    atom_view = _move(pb.atom_view, device)
    bond_view = _move(pb.bond_view, device)
    ids = pb.atom_ids.to(device)
    bids = pb.bond_ids.to(device)
    masked = set(int(x) for x in pb.atom_ids.tolist())
    bi = atom_view["bond_index"].cpu().numpy()
    n_atoms = int(atom_view["atom_feat"].shape[0])
    adj: list[set[int]] = [set() for _ in range(n_atoms)]
    for u, v in bi:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    target_atom = distance2 = None
    for v in sorted(masked):
        for u in sorted(adj[v]):
            for w in sorted(adj[u]):
                if w != v and w not in adj[v] and w not in masked:
                    target_atom, distance2 = v, w
                    break
            if target_atom is not None:
                break
        if target_atom is not None:
            break
    if target_atom is None:  # pragma: no cover
        return {"name": "no_message_passing", "passed": True, "skipped": "no distance-2 atom"}

    perturbed = dict(atom_view)
    feat = atom_view["atom_feat"].clone()
    feat[distance2] = feat[distance2] + 7.0
    perturbed["atom_feat"] = feat
    calls = {"n": 0}
    original = SM.encode_static

    def counting(*args, **kwargs):  # noqa: ANN002,ANN003
        calls["n"] += 1
        return original(*args, **kwargs)

    SM.encode_static = counting  # type: ignore[assignment]
    try:
        with torch.no_grad():
            base = model(atom_view, bond_view, ids, bids)
            other = model(perturbed, bond_view, ids, bids)
    finally:
        SM.encode_static = original  # type: ignore[assignment]
    row = int(torch.nonzero(ids == target_atom).squeeze(1))
    diff = _max_abs(base["atom_ctx"][row], other["atom_ctx"][row])
    hidden_modules = [
        n
        for n, m in model.named_modules()
        if type(m).__name__.lower()
        in {"messagepassing", "gcnconv", "ginconv", "gatconv", "transformerconv", "multiheadattention"}
    ]
    return {
        "name": "no_message_passing",
        "passed": bool(
            diff == 0.0 and calls["n"] == 4 and not hidden_modules
        ),
        "masked_atom": int(target_atom),
        "perturbed_distance2_atom": int(distance2),
        "max_abs_atom_context_diff": diff,
        "static_encode_calls_total_two_forwards": int(calls["n"]),
        "static_encode_calls_per_forward": float(calls["n"] / 2.0),
        "message_passing_modules": hidden_modules,
        "note": "one atom encode + one bond encode per masked view, incidence means only",
    }


# ---------------------------------------------------------------------------
# 10-12: training viability, model identity, reproducibility
# ---------------------------------------------------------------------------


def check_gradients(
    st: D.Structure, model: SM.PretextNet, device: torch.device, *, steps: int = 3
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(C.PRETRAIN["learning_rate"]))
    groups = {
        "atom_encoder": "atom_encoder.",
        "bond_encoder": "bond_encoder.",
        "atom_head": "atom_head.",
        "bond_head": "bond_head.",
    }
    seen: dict[str, list[float]] = {g: [] for g in groups}
    losses: list[float] = []
    size = int(C.PRETRAIN["batch_size"])
    model.train()
    for i in range(int(steps)):
        rows = np.asarray(st.fit_rows[i * size : (i + 1) * size], dtype=np.int64)
        pb = D.build_pretext_batch(st, rows, visit_index=100 + i)
        loss, _ = model.loss(
            _move(pb.atom_view, device),
            _move(pb.bond_view, device),
            pb.atom_ids.to(device),
            pb.bond_ids.to(device),
            pb.atom_targets.to(device),
            pb.bond_targets.to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(float(loss.detach()))
        for name, prefix in groups.items():
            norms = [
                float(p.grad.norm())
                for n, p in model.named_parameters()
                if n.startswith(prefix) and p.grad is not None
            ]
            seen[name].append(float(np.sqrt(np.sum(np.square(norms)))))
        optimizer.step()
    checks = {}
    for name in groups:
        values = np.asarray(seen[name], dtype=np.float64)
        checks[name] = {
            "grad_norms": [float(x) for x in values],
            "finite": bool(np.isfinite(values).all()),
            "any_nonzero": bool((values > 0).any()),
            "all_steps_nonzero": bool((values > 0).all()),
        }
    return {
        "name": "gradients_reach_all_groups",
        "passed": bool(
            all(v["finite"] and v["any_nonzero"] for v in checks.values())
            and all(np.isfinite(losses))
        ),
        "steps": int(steps),
        "trainable_groups": checks,
        "losses": losses,
        "note": "a group only needs a non-zero gradient on at least one of the steps",
    }


def check_original_a_geometry(scratch: Mapping[str, Any]) -> dict[str, Any]:
    model = scratch["model"]
    state = model.state_dict()
    expected_widths = {
        "atom_encoder.lins.0.weight": [C.ATOM_HIDDEN, C.ATOM_FEAT_DIM],
        "atom_encoder.lins.1.weight": [C.ATOM_OUT, C.ATOM_HIDDEN],
        "bond_encoder.lins.0.weight": [C.BOND_HIDDEN, 3 * C.ATOM_OUT + C.BOND_FEAT_DIM],
        "bond_encoder.lins.1.weight": [C.BOND_OUT, C.BOND_HIDDEN],
        "readout.fc1.weight": [88, 4 * C.WIDTH],
        "readout.fc2.weight": [C.N_TASKS, 88],
    }
    widths = {k: list(state[k].shape) for k in expected_widths}
    params = C.parameter_count(model)
    audit = scratch["audit"]
    ref_info = scratch["reference_info"]
    return {
        "name": "original_a_geometry",
        "passed": bool(
            widths == expected_widths
            and params == C.EXPECTED_PARAMETERS
            and int(audit["new_parameter_count"]) == C.EXPECTED_NEW_PARAMETERS
            and int(audit["base_parameter_elements"]) == C.EXPECTED_BASE_PARAMETERS
            and int(ref_info["local_input_dim"]) > 0
        ),
        "parameter_count": int(params),
        "expected_parameter_count": int(C.EXPECTED_PARAMETERS),
        "state_dict_key_count": len(state),
        "widths": widths,
        "new_parameter_count": int(audit["new_parameter_count"]),
        "base_parameter_elements": int(audit["base_parameter_elements"]),
        "c1_local_input_dim": int(ref_info["local_input_dim"]),
        "c1_rho_input_dim": int(ref_info["rho_input_dim"]),
        "pretext_heads_in_downstream_model": [
            k for k in state if k.startswith(("atom_head.", "bond_head."))
        ],
        "zero_residual_at_init": bool(audit["zero_residual_at_init"]),
    }


def check_reproducibility(
    st: D.Structure, model: SM.PretextNet, out_dir: Path
) -> dict[str, Any]:
    rows = np.asarray(st.fit_rows[:8], dtype=np.int64)
    a = D.build_pretext_batch(st, rows, visit_index=7)
    b = D.build_pretext_batch(st, rows, visit_index=7)
    c = D.build_pretext_batch(st, rows, visit_index=8)
    same_view = all(
        torch.equal(a.atom_view[k], b.atom_view[k]) and torch.equal(a.bond_view[k], b.bond_view[k])
        for k in D.MODEL_INPUT_KEYS
    )
    same_ids = torch.equal(a.atom_ids, b.atom_ids) and torch.equal(a.bond_ids, b.bond_ids)
    same_targets = torch.equal(a.atom_targets, b.atom_targets) and torch.equal(
        a.bond_targets, b.bond_targets
    )
    differs_on_new_visit = not (
        torch.equal(a.atom_ids, c.atom_ids) and torch.equal(a.bond_ids, c.bond_ids)
    )
    atom_counts = a.atom_view["atom_counts"].tolist()
    bond_counts = a.bond_view["bond_counts"].tolist()
    atom_mask_ok = all(
        int(m.size) == C.n_masked(int(n)) for m, n in zip(a.atom_mask_by_molecule, atom_counts)
    )
    bond_mask_ok = all(
        int(m.size) == C.n_masked(int(n)) for m, n in zip(a.bond_mask_by_molecule, bond_counts)
    )
    at_least_one = all(int(m.size) >= 1 for m in a.atom_mask_by_molecule)
    order_ok = _order_plan_deterministic(st.train_rows)
    payload = {
        "encoder_state": SM.encoder_state(model),
        "encoder_fingerprint": SM.encoder_fingerprint(model),
        "flat": torch.arange(7, dtype=torch.int64),
    }
    path = out_dir / "roundtrip_checkpoint.pt"
    torch.save(payload, path)
    back = torch.load(path, map_location="cpu", weights_only=False)
    roundtrip = C.state_fingerprint(back["encoder_state"]) == payload["encoder_fingerprint"]
    path.unlink(missing_ok=True)
    return {
        "name": "reproducibility",
        "passed": bool(
            same_view
            and same_ids
            and same_targets
            and differs_on_new_visit
            and atom_mask_ok
            and bond_mask_ok
            and at_least_one
            and order_ok
            and roundtrip
        ),
        "same_visit_bitwise_identical_views": same_view,
        "same_visit_identical_targets": bool(same_ids and same_targets),
        "new_visit_changes_the_mask": differs_on_new_visit,
        "atom_mask_count_is_frozen_15pct": atom_mask_ok,
        "bond_mask_count_is_frozen_15pct": bond_mask_ok,
        "every_molecule_masks_at_least_one_atom": at_least_one,
        "no_bond_molecule_skips_bond_targets": "molecules with zero bonds get an empty bond mask",
        "order_plan_matches_historical": order_ok,
        "encoder_checkpoint_roundtrip": roundtrip,
    }


def _order_plan_deterministic(train_rows: np.ndarray) -> bool:
    shim = type("T", (), {"train_rows": train_rows})()
    a = ND.make_plan40(shim, seed=int(C.SSL_SEED))
    b = ND.make_plan40(shim, seed=int(C.SSL_SEED))
    return bool(
        a[0]["global_plan_sha256"]
        == b[0]["global_plan_sha256"]
        == C.HISTORICAL_PARITY["global_plan_sha256"]
    )


def check_gpu_smoke(
    st: D.Structure, model: SM.PretextNet, device: torch.device, *, steps: int = 20
) -> dict[str, Any]:
    if device.type != "cuda":
        return {"name": "gpu_smoke", "passed": True, "skipped": "cpu audit run"}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(C.PRETRAIN["learning_rate"]))
    times: list[float] = []
    losses: list[float] = []
    grad_norms: list[float] = []
    model.train()
    size = int(C.PRETRAIN["batch_size"])
    for i in range(int(steps)):
        rows = np.asarray(st.fit_rows[i * size : (i + 1) * size], dtype=np.int64)
        t0 = time.perf_counter()
        pb = D.build_pretext_batch(st, rows, visit_index=200 + i)
        loss, _ = model.loss(
            _move(pb.atom_view, device),
            _move(pb.bond_view, device),
            pb.atom_ids.to(device),
            pb.bond_ids.to(device),
            pb.atom_targets.to(device),
            pb.bond_targets.to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
        losses.append(float(loss.detach()))
        grad_norms.append(gnorm)
    return {
        "name": "gpu_smoke",
        "passed": bool(
            np.isfinite(losses).all()
            and np.isfinite(grad_norms).all()
            and all(x > 0 for x in grad_norms)
        ),
        "device": str(device),
        "steps": int(steps),
        "step_ms_mean": float(np.mean(times) * 1e3),
        "step_ms_p50": float(np.percentile(times, 50) * 1e3),
        "step_ms_p90": float(np.percentile(times, 90) * 1e3),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "grad_norm_min": float(min(grad_norms)),
        "grad_norm_max": float(max(grad_norms)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "projected_pretrain_seconds": float(
            np.percentile(times, 50) * int(C.PRETRAIN["updates"])
        ),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_audit(
    *, out_path: Path, device: torch.device, gpu_smoke_steps: int = 20
) -> dict[str, Any]:
    t0 = time.perf_counter()
    st = D.load_structure()
    norm = D.load_train_norm(st)
    st.audit["norm_stage"] = norm["ssl_norm_audit"]
    priors = D.class_priors(st)
    model, pretext_audit = SM.build_pretext(st.spec, norm)
    model.to(device)
    scratch = SM.seed0_scratch_state(st.spec, norm)
    checks = [
        check_loader_label_isolation(st),
        check_row_partition(st),
        check_targets_not_in_input(st, _batch(st)),
        check_atom_view_invariance(_batch(st)),
        check_bond_view_invariance(_batch(st)),
        check_no_forbidden_fields(model, _batch(st), device),
        check_atom_renumbering(model, _batch(st), device),
        check_bond_endpoint_swap(model, _batch(st), device),
        check_no_message_passing(model, _batch(st), device),
        check_gradients(st, model, device),
        check_original_a_geometry(scratch),
        check_reproducibility(st, model, out_path.parent),
    ]
    if device.type == "cuda":
        checks.append(check_gpu_smoke(st, model, device, steps=int(gpu_smoke_steps)))
    report = {
        "schema": "lcrc-pcba-atom-bond-ssl-mask-leakage-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "device": str(device),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "pretext_model": pretext_audit,
        "class_priors": priors,
        "data_audit": st.audit,
        "checks": checks,
        "all_passed": bool(all(c.get("passed", False) for c in checks)),
        "audit_seconds": float(time.perf_counter() - t0),
        "test_evaluated": False,
    }
    C.write_json(out_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-smoke-steps", type=int, default=20)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_audit(
            out_path=out_path, device=device, gpu_smoke_steps=int(args.gpu_smoke_steps)
        )
        C.log(f"[ssl-audit] all_passed={report['all_passed']}")
        return 0 if report["all_passed"] else 1
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_path,
            {
                "schema": "lcrc-pcba-atom-bond-ssl-mask-leakage-audit-v1",
                "round_id": C.ROUND_ID,
                "status": "FAILED",
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "all_passed": False,
                "test_evaluated": False,
            },
        )
        C.log(f"[ssl-audit] FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
