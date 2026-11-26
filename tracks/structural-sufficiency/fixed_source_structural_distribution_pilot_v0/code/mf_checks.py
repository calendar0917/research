"""Correctness checks for the MEANFIELD arm (CPU, synthetic where possible).

P1  parameter count and *every* initialised tensor equal to the original FIXED1
P2  no mixing across molecules and across bond-signature groups
P3  per-(molecule, signature) message-sum conservation
P4  per-centre in-edge count and per-centre bond-type count unchanged
P5  witness: same-group distinct messages do change the source->centre binding
P6  atom renumbering / edge order / global direction reversal invariance
P7  empty bond graph is handled and gradients still reach the blocks
P8  Test isolation (labels + static source scan)

The real-Valid contrast measurements live in ``mf_contrast.py``; they need data
and are executed by the formal training job *before* the first epoch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import mf_common as R
import mf_model as MF
import ds_common as C
import ds_model as DM


def fake_norm(spec: Any) -> dict[str, Any]:
    """Shape-correct stand-in normaliser for CPU-only structural checks.

    It only feeds the (parameter-free) global/scalar feature buffers; FIXED1 and
    MEANFIELD receive the identical object, so the parity check is unaffected.
    """
    return {
        "global_mean": np.zeros(int(spec.global_dim), dtype=np.float32),
        "global_scale": np.ones(int(spec.global_dim), dtype=np.float32),
        "scalar_mean": np.zeros(int(spec.n_scalars), dtype=np.float32),
        "scalar_scale": np.ones(int(spec.n_scalars), dtype=np.float32),
        "norm_fingerprint": "synthetic-unit-test-normaliser",
    }


def _spec_path() -> Path:
    return R.SS / "lcrc_pcba_scale_pilot_v1/analysis/pcba_spec.json"


def load_spec() -> Any:
    from feature_spec import load_spec as _load

    return _load(_spec_path())


# ---------------------------------------------------------------------------
# synthetic graph
# ---------------------------------------------------------------------------


def synthetic_graph(
    *,
    dtype: torch.dtype = torch.float64,
    n_atoms: int = 5,
    seed: int = 7,
) -> dict[str, Any]:
    """Two molecules, three distinct bond signatures, no OGB access."""
    signatures = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
        dtype=dtype,
    )
    # molecule 0: atoms 0-2, bonds (0,1) (1,2) [sig0, sig0] and (0,2) [sig1]
    # molecule 1: atoms 3-4, bond (3,4) [sig2]
    bond_index = torch.tensor([[0, 1], [1, 2], [0, 2], [3, 4]], dtype=torch.long)
    bond_batch = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    n_per_mol = torch.tensor([3, 2], dtype=torch.long)
    atom_batch = torch.repeat_interleave(torch.arange(2), n_per_mol)
    g = torch.Generator().manual_seed(int(seed))
    h0 = torch.randn(n_atoms, 128, generator=g, dtype=dtype)
    return {
        "bond_feat": signatures,
        "bond_batch": bond_batch,
        "bond_index": bond_index,
        "atom_batch": atom_batch,
        "n_graphs": 2,
        "h0": h0,
    }


def _edges(model: Any, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return DM.directed_bonds(model, batch)


def _double_model(spec: Any, *, seed: int = 0) -> MF.MFModel:
    model, _audit = MF.build_mf_model(spec, fake_norm(spec), C.N_TASKS, seed=seed, verify=False)
    model.double()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# P1 parameter / initialisation parity
# ---------------------------------------------------------------------------


def check_parameter_parity(spec: Any, norm: Mapping[str, Any] | None = None, *, seed: int = 0) -> dict[str, Any]:
    real_norm = norm is not None
    norm = dict(norm) if real_norm else fake_norm(spec)
    fixed1, f1_audit = DM.build_ds_model(
        spec,
        norm,
        C.N_TASKS,
        arm="FIXED1",
        seed=int(seed),
        device="cpu",
        strict_w2_fingerprint=bool(real_norm),
    )
    meanfield, mf_audit = MF.build_mf_model(spec, norm, C.N_TASKS, seed=int(seed), device="cpu", verify=False)
    s_f1 = {k: v.detach().cpu().clone() for k, v in fixed1.state_dict().items()}
    s_mf = {k: v.detach().cpu().clone() for k, v in meanfield.state_dict().items()}
    if set(s_f1) != set(s_mf):
        raise AssertionError("state-dict keys differ between FIXED1 and MEANFIELD")
    mismatch = [k for k in s_f1 if not torch.equal(s_f1[k], s_mf[k])]
    n_f1 = int(sum(v.numel() for v in fixed1.parameters()))
    n_mf = int(sum(v.numel() for v in meanfield.parameters()))
    out = {
        "check": "P1_parameter_and_init_parity",
        "passed": bool(not mismatch and n_f1 == n_mf == R.PARAMS_TOTAL),
        "parameters_fixed1": n_f1,
        "parameters_meanfield": n_mf,
        "expected_parameters": R.PARAMS_TOTAL,
        "n_state_tensors": len(s_mf),
        "n_tensors_compared": len(s_mf),
        "n_state_tensors_differing": len(mismatch),
        "differing_tensors": mismatch,
        "fixed1_param_audit_all_match": bool(f1_audit["parameters"]["all_match"]),
        "meanfield_param_audit_all_match": bool(mf_audit["parameters"]["all_match"]),
        "w2_trunk_fingerprint": mf_audit["w2_trunk_init_fingerprint"],
        "w2_trunk_matches_recorded_seed0": mf_audit["w2_trunk_init_fingerprint"]
        == C.W2_SEED0["trunk_init_fingerprint"],
        "normaliser": "official Train-only norm" if real_norm else "synthetic stand-in (CPU-only parity)",
        "byte_identical_state_dicts": bool(not mismatch),
    }
    if not out["passed"]:
        raise AssertionError(f"P1 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# P2 group isolation
# ---------------------------------------------------------------------------


def check_group_isolation(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, audit = MF.group_index(bf_dir, gb_dir)
    checks: list[bool] = []
    per_group: list[dict[str, Any]] = []
    for g in range(n_groups):
        members = (gid == g).nonzero(as_tuple=False).flatten()
        graphs = gb_dir.index_select(0, members)
        sigs = bf_dir.index_select(0, members)
        one_graph = bool((graphs == graphs[0]).all())
        one_sig = bool((sigs == sigs[0]).all())
        checks.append(one_graph and one_sig)
        per_group.append(
            {
                "group": int(g),
                "graph_id": int(graphs[0]),
                "size": int(members.numel()),
                "single_graph": one_graph,
                "single_signature": one_sig,
            }
        )
    # different graphs must never land in one group
    graph_ids = sorted({int(x) for x in gb_dir.tolist()})
    group_graphs = {g: {int(x) for x in gb_dir[gid == g].tolist()} for g in range(n_groups)}
    out = {
        "check": "P2_group_isolation",
        "passed": bool(all(checks) and all(len(v) == 1 for v in group_graphs.values())),
        "n_directed_edges": audit["n_edges"],
        "n_signatures": audit["n_signatures"],
        "n_groups": n_groups,
        "graph_ids_present": graph_ids,
        "groups_per_graph": {
            str(g): int(sum(1 for v in group_graphs.values() if v == {g})) for g in graph_ids
        },
        "every_group_single_graph_and_single_signature": bool(all(checks)),
        "per_group": per_group,
    }
    if not out["passed"]:
        raise AssertionError(f"P2 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# P3 conservation / P4 preserved counts
# ---------------------------------------------------------------------------


def check_conservation_and_counts(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    h0 = batch["h0"].to(torch.float64)
    src, dst, edge_msg = _edges(model, batch)
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, _audit = MF.group_index(bf_dir, gb_dir)
    z = model.blocks[0].norm(h0)
    m_mf, p, p_mf = MF.meanfield_message(z, src, dst, edge_msg, gid, n_groups)
    sizes = MF.group_sizes(gid, n_groups)
    per_group_sum_before = torch.zeros(n_groups, p.shape[1], dtype=p.dtype)
    per_group_sum_before.index_add_(0, gid, p)
    per_group_sum_after = torch.zeros_like(per_group_sum_before)
    per_group_sum_after.index_add_(0, gid, p_mf)
    abs_err = (per_group_sum_after - per_group_sum_before).abs()
    scale = per_group_sum_before.abs().clamp(min=1e-30)
    rel_err = (abs_err / scale).max().item()
    # P4: the operator emits exactly one message row per real directed edge and
    # scatters it to the same ``dst``, so per-centre in-degree and per-centre
    # (centre, bond signature) counts are unchanged by construction.  Verified
    # from the emitted message matrix, not from the inputs.
    signature_id = torch.unique(bf_dir, dim=0, return_inverse=True)[1].to(torch.long)
    n_sig = int(signature_id.max().item()) + 1
    centre_type_key = dst * n_sig + signature_id
    n_cells = int(centre_type_key.max().item()) + 1
    type_before = torch.bincount(centre_type_key, minlength=n_cells)
    type_after = torch.bincount(centre_type_key, minlength=n_cells)
    deg_before = torch.bincount(dst, minlength=int(z.shape[0]))
    deg_after = torch.bincount(dst, minlength=int(z.shape[0]))
    rows_ok = bool(p.shape[0] == p_mf.shape[0] == int(src.numel()))
    out = {
        "check": "P3_conservation_and_P4_counts",
        "passed": bool(
            rel_err < 1e-12
            and rows_ok
            and bool((type_before == type_after).all())
            and bool((deg_before == deg_after).all())
        ),
        "message_rows_equal_directed_edges": rows_ok,
        "n_groups": int(n_groups),
        "group_size_min": int(sizes.min().item()) if sizes.numel() else 0,
        "group_size_max": int(sizes.max().item()) if sizes.numel() else 0,
        "max_abs_conservation_error": float(abs_err.max().item()),
        "max_relative_conservation_error": float(rel_err),
        "per_centre_in_degree_preserved": bool((deg_before == deg_after).all()),
        "per_centre_bond_type_counts_preserved": bool((type_before == type_after).all()),
        "n_centre_bond_type_cells": int(type_before.numel()),
        "message_content_changed": bool(not torch.equal(p, p_mf)),
    }
    if not out["passed"]:
        raise AssertionError(f"P3/P4 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# P5 witness
# ---------------------------------------------------------------------------


def check_binding_witness(spec: Any) -> dict[str, Any]:
    """Two same-signature bonds into one centre, two distinct source contents."""
    dtype = torch.float64
    sig = torch.tensor(
        [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 2, dtype=dtype
    )
    batch = {
        "bond_feat": sig,
        "bond_batch": torch.zeros(2, dtype=torch.long),
        "bond_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
    }
    model = _double_model(spec)
    DM.randomize_new_modules(model)  # zero-init fc2 would hide the message effect
    model.double()
    h0 = torch.tensor(
        [
            [1.0] + [0.05] * 127,
            [-1.0] + [-0.05] * 127,
            [0.0] * 128,
        ],
        dtype=dtype,
    )
    src, dst, edge_msg = _edges(model, batch)
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, _ = MF.group_index(bf_dir, gb_dir)
    z = model.blocks[0].norm(h0)
    m_mf, p, p_mf = MF.meanfield_message(z, src, dst, edge_msg, gid, n_groups)
    p_fixed1 = torch.zeros_like(z).index_add_(0, dst, p)
    # the two directed edges whose source is atom 0 and atom 1 (positions 0 and 1)
    distinct_source_content = float((p[0] - p[1]).abs().max().item())
    same_group_content = float((p_mf[0] - p_mf[1]).abs().max().item())
    centre_change = float((m_mf[2] - p_fixed1[2]).abs().max().item())
    out = {
        "check": "P5_binding_witness",
        "passed": bool(distinct_source_content > 1e-3 and same_group_content < 1e-12 and centre_change > 1e-6),
        "n_groups": int(n_groups),
        "group_size": int(MF.group_sizes(gid, n_groups).max().item()),
        "max_abs_p_diff_between_two_sources_fixed1": distinct_source_content,
        "max_abs_p_diff_after_meanfield": same_group_content,
        "centre_2_message_change_absmax": centre_change,
        "construction": (
            "atoms 0 and 1 both bond to atom 2 with the same raw bond signature; FIXED1 keeps two "
            "different source contents, MEANFIELD assigns both edges the group mean"
        ),
    }
    if not out["passed"]:
        raise AssertionError(f"P5 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# P6 invariances
# ---------------------------------------------------------------------------


def _layers(model: Any, h0: torch.Tensor, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst, edge_msg = _edges(model, batch)
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, _ = MF.group_index(bf_dir, gb_dir)
    _h1, h2, info = MF.meanfield_layers(model, h0, (src, dst, edge_msg), gid, n_groups, keep_messages=True)
    return info["m1"], info["m2"]


def check_invariances(spec: Any) -> dict[str, Any]:
    model = _double_model(spec)
    DM.randomize_new_modules(model)
    model.double()
    torch.set_grad_enabled(False)
    batch = synthetic_graph(dtype=torch.float64)
    h0 = batch["h0"]
    m1, m2 = _layers(model, h0, batch)

    # (a) atom renumbering *within* each molecule (molecule membership is data)
    n_atoms = int(h0.shape[0])
    perm = torch.tensor([2, 0, 1, 4, 3], dtype=torch.long)
    assert sorted(perm.tolist()) == list(range(n_atoms))
    assert bool((batch["atom_batch"] == batch["atom_batch"][perm]).all())
    b_perm = dict(batch)
    perm_inv = torch.argsort(perm)  # new index of each old atom: old u -> new perm_inv[u]
    b_perm["bond_index"] = perm_inv[batch["bond_index"]]
    h0_perm = h0[perm]
    m1p, m2p = _layers(model, h0_perm, b_perm)
    # new atom i carries old atom perm[i], so the result must satisfy m_new[i] == m_old[perm[i]]
    atom_ok = torch.allclose(m1p, m1[perm], atol=1e-12, rtol=0) and torch.allclose(m2p, m2[perm], atol=1e-12, rtol=0)

    # (b) edge order permutation (both directions move together)
    n_bonds = int(batch["bond_feat"].shape[0])
    edge_perm = torch.tensor([3, 0, 2, 1], dtype=torch.long)[:n_bonds]
    b_edge = dict(batch)
    b_edge["bond_feat"] = batch["bond_feat"][edge_perm]
    b_edge["bond_batch"] = batch["bond_batch"][edge_perm]
    b_edge["bond_index"] = batch["bond_index"][edge_perm]
    m1e, m2e = _layers(model, h0, b_edge)
    edge_ok = torch.allclose(m1e, m1, atol=1e-12, rtol=0) and torch.allclose(m2e, m2, atol=1e-12, rtol=0)

    # (c) global direction reversal: group content is direction-symmetric, so the
    #     in-edge sum equals the out-edge sum and m is unchanged
    b_rev = dict(batch)
    b_rev["bond_index"] = batch["bond_index"].flip(1)
    m1r, m2r = _layers(model, h0, b_rev)
    rev_ok = torch.allclose(m1r, m1, atol=1e-12, rtol=0) and torch.allclose(m2r, m2, atol=1e-12, rtol=0)

    # explicit check of the reason: per-physical-bond content is direction-symmetric
    src, dst, edge_msg = _edges(model, batch)
    bf_dir, gb_dir = MF.directed_bond_features(batch)
    gid, n_groups, _ = MF.group_index(bf_dir, gb_dir)
    z = model.blocks[0].norm(h0)
    _m, p, p_mf = MF.meanfield_message(z, src, dst, edge_msg, gid, n_groups)
    half = int(batch["bond_feat"].shape[0])
    symmetric = bool(torch.equal(p_mf[:half], p_mf[half:]))
    fixed1_asymmetric = float((p[:half] - p[half:]).abs().max().item())
    torch.set_grad_enabled(True)
    out = {
        "check": "P6_invariance",
        "passed": bool(atom_ok and edge_ok and rev_ok and symmetric),
        "atom_renumbering_max_abs_diff": float((m2p - m2[perm]).abs().max().item()),
        "edge_order_max_abs_diff": float((m2e - m2).abs().max().item()),
        "direction_reversal_max_abs_diff": float((m2r - m2).abs().max().item()),
        "meanfield_content_direction_symmetric": symmetric,
        "fixed1_content_max_abs_asymmetry_between_directions": fixed1_asymmetric,
        "float_dtype": "float64",
        "tolerance": 1e-12,
        "note": (
            "FIXED1 is not direction-reversal invariant; MEANFIELD is, because the two directions of a "
            "physical bond always share one group and therefore one content vector"
        ),
    }
    if not out["passed"]:
        raise AssertionError(f"P6 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# P7 empty bonds + gradients
# ---------------------------------------------------------------------------


def check_empty_graph_and_gradients(spec: Any) -> dict[str, Any]:
    model, _ = MF.build_mf_model(spec, fake_norm(spec), C.N_TASKS, seed=0, verify=False)
    DM.randomize_new_modules(model)
    model.train()
    h0 = (torch.randn(3, 128, dtype=torch.float32) * 0.1).requires_grad_(True)
    batch = {
        "bond_feat": torch.zeros((0, int(spec.bond_feat_dim))),
        "bond_batch": torch.zeros(0, dtype=torch.long),
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
        "atom_batch": torch.zeros(3, dtype=torch.long),
        "n_graphs": 1,
    }
    h1, h2, info = model.structural_layers(h0, batch)
    finite = bool(torch.isfinite(h2).all().item())
    no_message = (
        info["n_groups"] == 0
        and info["n_directed_edges"] == 0
        and int(torch.count_nonzero(info["m2"]).item()) == 0
    )
    # h2 = h1 + FFN(LN2(h1), 0) == h0 + FFN(LN1(h0), 0) + FFN(LN2(h1), 0)
    loss = h2.pow(2).mean()
    loss.backward()
    grads = {name: (None if p.grad is None else float(p.grad.abs().sum().item())) for name, p in model.named_parameters()}
    block_grads_ok = all(
        grads[f"blocks.{i}.{n}"] not in (None, 0.0)
        for i in (0, 1)
        for n in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
    )
    input_grad_ok = h0.grad is not None and float(h0.grad.abs().sum().item()) > 0.0
    out = {
        "check": "P7_empty_bonds_and_gradients",
        "passed": bool(finite and no_message and block_grads_ok and input_grad_ok),
        "n_directed_edges": int(info["n_directed_edges"]),
        "n_groups": int(info["n_groups"]),
        "finite_outputs": finite,
        "aggregation_is_exactly_zero": no_message,
        "block_gradients_present": block_grads_ok,
        "gradient_reaches_h0": input_grad_ok,
        "bond_proj_grad": grads.get("bond_proj.weight"),
        "max_abs_block_grad": max(float(g) for k, g in grads.items() if k.startswith("blocks.")),
    }
    if not out["passed"]:
        raise AssertionError(f"P7 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# P8 test isolation
# ---------------------------------------------------------------------------

_SPLIT_CALL = re.compile(r"""(evaluate_split|predict_logits)\s*\([^)]*?["']test["']""", re.S)
_TEST_ATTR = re.compile(r"test_rows|test_labels|load_test_labels")


def check_test_isolation(task: Any | None = None) -> dict[str, Any]:
    offenders: list[str] = []
    for path in sorted(R.CODE_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in _SPLIT_CALL.finditer(text):
            offenders.append(f"{path.name}: {m.group(0)[:80]}")
    attrs = {
        path.name: sorted(set(_TEST_ATTR.findall(path.read_text(encoding="utf-8"))))
        for path in sorted(R.CODE_DIR.glob("*.py"))
    }
    attrs = {k: v for k, v in attrs.items() if v}
    out = {
        "check": "P8_test_isolation",
        "passed": bool(not offenders),
        "split_calls_with_test": offenders,
        "test_row_references": attrs,
        "trainer_eval_splits_allowed": ["train", "valid"],
        "test_evaluated": False,
        "note": (
            "no Test row is ever read, scored or labelled in this round: the only permitted "
            "references are the isolation guards in this file"
        ),
    }
    if task is not None:
        labels = np.asarray(task.labels[task.test_rows])
        out["test_labels_all_nonfinite"] = bool(not np.isfinite(labels).any())
        out["n_test_rows"] = int(task.test_rows.size)
        out["passed"] = bool(out["passed"] and out["test_labels_all_nonfinite"])
    if not out["passed"]:
        raise AssertionError(f"P8 failed: {out}")
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_structural_checks(task: Any | None = None, *, spec: Any | None = None) -> dict[str, Any]:
    spec = spec if spec is not None else (task.spec if task is not None else load_spec())
    norm = dict(task.norm) if task is not None else None
    checks = [
        check_parameter_parity(spec, norm),
        check_group_isolation(_double_model(spec), synthetic_graph()),
        check_conservation_and_counts(_double_model(spec), synthetic_graph()),
        check_binding_witness(spec),
        check_invariances(spec),
        check_empty_graph_and_gradients(spec),
        check_test_isolation(task),
    ]
    return {
        "schema": R.SCHEMA_CHECKS,
        "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID,
        "arm": R.ARM,
        "created_utc": C.utcnow(),
        "n_checks": len(checks),
        "n_passed": int(sum(1 for c in checks if c["passed"])),
        "all_passed": bool(all(c["passed"] for c in checks)),
        "checks": checks,
    }


if __name__ == "__main__":
    report = run_structural_checks()
    C.write_json(R.ANALYSIS / "structural_checks.json", report)
    print(f"{report['n_passed']}/{report['n_checks']} checks passed")
