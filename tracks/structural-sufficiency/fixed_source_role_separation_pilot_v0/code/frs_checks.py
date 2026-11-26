"""FSR v0 correctness checks for the one change of this round.

``--mode quick`` runs on CPU with a stand-in normaliser and synthetic graphs
(no data cache).  ``--mode full`` additionally verifies the inherited data
identity and the real-batch equivalence/invariance.  The inherited FIXED1
arithmetic is not re-audited beyond the ``A = 0`` identity, per protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import frs_common as FC  # noqa: E402  (sets sys.path to the inherited code)
import ds_common as C  # noqa: E402
import ds_model as DSM  # noqa: E402
import ds_checks as DCHK  # noqa: E402
import frs_model as FM  # noqa: E402
import pc_metrics as MET  # noqa: E402

RECORDS: list[dict[str, Any]] = []


def record(name: str, passed: bool, detail: Mapping[str, Any]) -> None:
    RECORDS.append({"check": name, "passed": bool(passed), "detail": dict(detail)})
    FC.log(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:360]}")


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"max_abs_diff": float("inf"), "rel_to_scale": float("inf")}
    if a.numel() == 0:
        return {"max_abs_diff": 0.0, "rel_to_scale": 0.0}
    d = (a.detach().double() - b.detach().double()).abs()
    scale = b.detach().double().abs().max().clamp(min=1e-12)
    return {
        "max_abs_diff": float(d.max().item()) if d.numel() else 0.0,
        "rel_to_scale": float((d.max() / scale).item()) if d.numel() else 0.0,
    }


def close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6) -> bool:
    if a.shape != b.shape:
        return False
    return bool((a.detach().double() - b.detach().double()).abs().max().item() <= atol)


def initial_models(spec: Any, norm: Mapping[str, Any], *, verify: bool) -> tuple[dict[str, Any], Any]:
    models = {}
    for arm in FC.ARMS:
        m, audit = FM.build_frs_model(spec, norm, C.N_TASKS, arm=arm, seed=FC.SEED,
                                      device="cpu", verify=verify,
                                      strict_w2_fingerprint=False)
        models[arm] = (m, audit)
    fixed1, fixed1_audit = FM.build_fixed1_reference(spec, norm, C.N_TASKS, seed=FC.SEED)
    return models, (fixed1, fixed1_audit)


def check_parameters(models: Mapping[str, Any]) -> None:
    got = {arm: FC.parameter_groups(m) for arm, (m, _a) in models.items()}
    tops = {arm: int(g["total"]) for arm, g in got.items()}
    lows = {arm: int(g["lowrank_total"]) for arm, g in got.items()}
    detail = {
        "groups": got,
        "totals": tops,
        "lowrank_totals": lows,
        "expected_total": FC.PARAMS_TOTAL,
        "expected_lowrank": FC.PARAMS_ADD,
    }
    passed = (
        len(set(tops.values())) == 1
        and len(set(lows.values())) == 1
        and tops[FC.ARMS[0]] == FC.PARAMS_TOTAL
        and lows[FC.ARMS[0]] == FC.PARAMS_ADD
        and all(a["parameters"]["all_match"] for _m, a in models.values())
    )
    record("B1_parameters", passed, detail)


def check_init_identity(models: Mapping[str, Any], *, full: bool) -> None:
    a_state = {k: v.detach().cpu() for k, v in models["TIED"][0].state_dict().items()}
    b_state = {k: v.detach().cpu() for k, v in models["RSSC"][0].state_dict().items()}
    keys_equal = set(a_state) == set(b_state)
    diffs = {
        k: diff_stats(b_state[k], a_state[k])
        for k in a_state
        if k in b_state
    }
    max_diff = max((d["max_abs_diff"] for d in diffs.values()), default=float("inf"))
    fp_tied = FC.initial_fingerprints(models["TIED"][0])
    fp_rssc = FC.initial_fingerprints(models["RSSC"][0])
    detail = {
        "same_keys": bool(keys_equal),
        "max_abs_diff_over_all_tensors": float(max_diff),
        "ties_bitwise_identical": bool(keys_equal and max_diff == 0.0),
        "TIED_fingerprints": fp_tied,
        "RSSC_fingerprints": fp_rssc,
        "historical_FIXED1_init": FC.HIST_FIXED1_INIT,
        "base_matches_historical": fp_tied["base_state_sha256"] == FC.HIST_FIXED1_INIT["base_state_sha256"],
        "inherited_new_matches_historical": fp_tied["new_state_sha256"] == FC.HIST_FIXED1_INIT["new_state_sha256"],
        "mode": "full" if full else "quick",
    }
    passed = bool(keys_equal and max_diff == 0.0 and fp_tied == fp_rssc)
    if full:
        passed = passed and detail["base_matches_historical"] and detail["inherited_new_matches_historical"]
    record("B2_initial_identity", passed, detail)


def check_zero_A_equivalence(models: Mapping[str, Any], fixed1: Any, batch: Mapping[str, Any],
                             *, label: str) -> None:
    """A = 0 must reproduce the inherited FIXED1 forward exactly."""
    device = next(fixed1.parameters()).device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    fixed1.eval()
    with torch.inference_mode():
        ref_logits = fixed1(batch, return_states=True)
    ref_logits, ref_states = ref_logits
    out: dict[str, Any] = {"batch": label, "per_arm": {}}
    all_pass = True
    for arm, (model, _audit) in models.items():
        model.eval()
        with torch.inference_mode():
            logits, states = model(batch, return_states=True)
        d = {
            "logits": diff_stats(logits, ref_logits),
            "h1": diff_stats(states["h1"], ref_states["h1"]),
            "h2": diff_stats(states["h2"], ref_states["h2"]),
            "m2": diff_stats(states["info"]["m2"], ref_states["info"]["m2"]),
        }
        max_abs = max(v["max_abs_diff"] for v in d.values())
        d["exact"] = bool(max_abs == 0.0)
        out["per_arm"][arm] = d
        all_pass = all_pass and max_abs == 0.0
    record(f"B3_zero_A_FIXED1_equivalence_{label}", all_pass, out)


def check_role_input(models: Mapping[str, Any]) -> None:
    """Non-zero A (and a non-zero fc2): TIED ignores z-m, RSSC is sensitive to it."""
    torch.manual_seed(11)
    z = torch.randn(7, FC.WIDTH)
    m = torch.randn(7, FC.WIDTH) * 0.3
    delta = torch.randn(7, FC.WIDTH) * 0.5
    z2, m2 = z + delta, m - delta  # z+m unchanged up to fp32, z-m changed
    detail: dict[str, Any] = {
        "z_plus_m_max_abs_diff": float((z + m - (z2 + m2)).abs().max().item()),
        "z_minus_m_change_rms": float((z - m - (z2 - m2)).pow(2).mean().sqrt()),
        "note": "fc2 is zero at initialisation (inherited zero-residual rule), so the branch "
                "is made observable with a small deterministic non-zero fc2 for this check",
    }
    all_pass = True
    for arm, (model, _audit) in models.items():
        model.eval()
        with torch.no_grad():
            for i in (0, 1):
                model.lowrank[i].A.weight.normal_(0.0, 0.05)
                model.blocks[i].fc2.weight.normal_(0.0, 0.02)
                model.blocks[i].fc2.bias.zero_()
            o1 = [model.block_out(i, z, m) for i in (0, 1)]
            o2 = [model.block_out(i, z2, m2) for i in (0, 1)]
            a1 = [model.blocks[i].fc1(z + m) + model.lowrank[i](model.branch_input(z, m)) for i in (0, 1)]
            a2 = [model.blocks[i].fc1(z2 + m2) + model.lowrank[i](model.branch_input(z2, m2)) for i in (0, 1)]
        d = {
            f"block{i}_preactivation_max_abs_diff": float((a1[i] - a2[i]).abs().max().item())
            for i in (0, 1)
        }
        d.update({
            f"block{i}_output_max_abs_diff": float((o1[i] - o2[i]).abs().max().item())
            for i in (0, 1)
        })
        if arm == "TIED":
            # z + m is unchanged up to float32 rounding, so the block output is too
            d["invariant_within_fp32_rounding"] = all(
                d[f"block{i}_output_max_abs_diff"] <= 1e-6 for i in (0, 1)
            )
            d["invariant_as_expected"] = bool(d["invariant_within_fp32_rounding"])
        else:
            d["responds_as_expected"] = all(
                d[f"block{i}_preactivation_max_abs_diff"] > 1e-3 for i in (0, 1)
            )
        detail[arm] = d
        all_pass = all_pass and bool(d.get("invariant_as_expected", d.get("responds_as_expected")))
    record("B4_role_branch_input", all_pass, detail)


def check_dual_form(models: Mapping[str, Any]) -> None:
    """``W(z+m) + D(z-m) == (W+D)z + (W-D)m`` with the fc1 bias added once."""
    torch.manual_seed(13)
    z = torch.randn(5, FC.WIDTH, dtype=torch.float64)
    m = torch.randn(5, FC.WIDTH, dtype=torch.float64) * 0.4
    model = models["RSSC"][0]
    detail: dict[str, Any] = {"per_block": {}}
    all_pass = True
    for i in (0, 1):
        blk = model.blocks[i]
        W = blk.fc1.weight.detach().double()
        b = blk.fc1.bias.detach().double()
        D = model.lowrank[i].effective_weight().detach().double()
        direct = z @ W.t() + m @ W.t() + (z - m) @ D.t() + b
        factored = z @ (W + D).t() + m @ (W - D).t() + b
        err = (direct - factored).abs().max().item()
        rel = err / max(1e-12, direct.abs().max().item())
        detail["per_block"][str(i)] = {
            "max_abs_error": float(err),
            "relative_error": float(rel),
            "bias_added_times": 1,
        }
        all_pass = all_pass and rel < 1e-12
    record("B5_RSSC_dual_form_equivalence", all_pass, detail)


def check_fixed_source(models: Mapping[str, Any]) -> None:
    """Layer 2 neighbourhood aggregation must read LN2(h0), never LN2(h1)."""
    torch.manual_seed(17)
    n, width = 6, FC.WIDTH
    h0 = torch.randn(n, width)
    h1 = torch.randn(n, width)
    src = torch.tensor([0, 2, 3, 4, 5, 1, 2, 5])
    dst = torch.tensor([1, 1, 2, 2, 2, 4, 4, 5])
    edge_msg = torch.randn(src.numel(), width) * 0.2
    detail: dict[str, Any] = {"per_arm": {}}
    all_pass = True
    for arm, (model, _audit) in models.items():
        model.eval()
        with torch.no_grad():
            blk2 = model.blocks[1]
            h1a, m_a, _i = model.stage2(h1, h0, src, dst, edge_msg)
            h1b, m_b, _i = model.stage2(h1 + torch.randn(n, width) * 5.0, h0, src, dst, edge_msg)
            _h, m_c, _i = model.stage2(h1, h0 + torch.randn(n, width) * 5.0, src, dst, edge_msg)
            expected = DSM._agg_source(blk2.norm(h0), src, dst, edge_msg)
            block1_out = model.stage1(h0 + torch.randn(n, width), src, dst, edge_msg)[0]
        d = {
            "m2_unchanged_when_h1_perturbed": bool((m_a - m_b).abs().max().item() == 0.0),
            "m2_changed_when_h0_perturbed": bool((m_a - m_c).abs().max().item() > 0.0),
            "m2_equals_agg_of_LN2_h0": bool((m_a - expected).abs().max().item() == 0.0),
            "stage1_returns_finite": bool(torch.isfinite(block1_out).all().item()),
        }
        d["passed"] = all(d.values())
        detail["per_arm"][arm] = d
        all_pass = all_pass and d["passed"]
    record("B6_fixed_h0_neighbour_source", all_pass, detail)


def _permute_bonds(batch: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Reorder the physical bond list and swap the stored key direction."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    bi = out["bond_index"]
    n = int(bi.shape[0])
    if n == 0:
        return out
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    bi = bi[perm]
    flip = torch.rand(n, generator=g) < 0.5
    bi = torch.where(flip.unsqueeze(1), bi[:, [1, 0]], bi)
    out["bond_index"] = bi
    for key in ("bond_feat", "bond_type", "bond_batch", "bond_scalar"):
        if key in out and torch.is_tensor(out[key]) and out[key].shape[0] == n:
            out[key] = out[key][perm]
    return out


def check_invariance(models: Mapping[str, Any], batch: Mapping[str, Any], *, label: str) -> None:
    detail: dict[str, Any] = {"batch": label, "per_arm": {}}
    all_pass = True
    for arm, (model, _audit) in models.items():
        model.eval()
        with torch.inference_mode():
            base = model(batch)
            perm_batch, _p = DSM.BM.permute_atoms(batch, seed=20261108)
            perm = model(perm_batch)
            reversed_batch = _permute_bonds(batch, seed=7)
            rev = model(reversed_batch)
        d = {
            "atom_renumbering": diff_stats(perm, base),
            "edge_order_and_key_direction": diff_stats(rev, base),
        }
        d["passed"] = close(perm, base, atol=2e-6) and close(rev, base, atol=2e-6)
        detail["per_arm"][arm] = d
        all_pass = all_pass and d["passed"]
    record(f"B7_invariance_{label}", all_pass, detail)


def check_gradients(models: Mapping[str, Any], batch: Mapping[str, Any], *, label: str) -> None:
    """A/B gradients on the natural optimisation path (inherited zero-residual)."""
    detail: dict[str, Any] = {"batch": label, "per_arm": {}}
    all_pass = True
    g = torch.Generator().manual_seed(23)
    for arm, (model, _audit) in models.items():
        # reproduce the inherited initial residual state (fc2 = 0, A = 0); B4
        # deliberately made fc2 observable and is not part of this check
        with torch.no_grad():
            for i in (0, 1):
                model.blocks[i].fc2.weight.zero_()
                model.blocks[i].fc2.bias.zero_()
                model.lowrank[i].A.weight.zero_()
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        y = (torch.rand((1, C.N_TASKS), generator=g) > 0.6).float()
        logits = model(batch)
        y = (torch.rand(logits.shape, generator=g) > 0.6).float()
        steps: list[dict[str, Any]] = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = MET.masked_bce_loss(logits, y)
            loss.backward()
            grad_rec = {k: (None if p.grad is None else float(p.grad.norm().item()))
                        for k, p in model.named_parameters() if FC.is_lowrank(k)}
            steps.append({"step": step + 1, "loss": float(loss.detach()), "grad_norms": grad_rec,
                          "fc2_weight_norm_0": float(model.blocks[0].fc2.weight.norm().item())})
            opt.step()
        model.eval()
        first = steps[0]["grad_norms"]
        a_step2 = [steps[1]["grad_norms"][f"lowrank.{i}.A.weight"] for i in (0, 1)]
        b_step3 = [steps[2]["grad_norms"][f"lowrank.{i}.B.weight"] for i in (0, 1)]
        # explicit addressability: non-zero A makes B receive gradient immediately
        with torch.no_grad():
            for i in (0, 1):
                model.lowrank[i].A.weight.normal_(0.0, 0.05)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch), y)
        loss.backward()
        explicit = {k: (None if p.grad is None else float(p.grad.norm().item()))
                    for k, p in model.named_parameters() if FC.is_lowrank(k)}
        opt.zero_grad(set_to_none=True)
        model.eval()
        d = {
            "steps": steps,
            "A_grad_zero_at_step1_because_fc2_is_zero": all(v in (None, 0.0) for v in first.values()),
            "A_grad_nonzero_at_step2": all(v not in (None, 0.0) for v in a_step2),
            "B_grad_nonzero_at_step3": all(v not in (None, 0.0) for v in b_step3),
            "B_grad_with_explicit_nonzero_A": explicit,
            "B_grad_addressable": all(
                explicit[f"lowrank.{i}.B.weight"] not in (None, 0.0) for i in (0, 1)
            ),
            "note": "step-1 zero gradient is the inherited zero-initialised fc2, not a broken path",
        }
        d["passed"] = bool(
            d["A_grad_zero_at_step1_because_fc2_is_zero"]
            and d["A_grad_nonzero_at_step2"]
            and d["B_grad_nonzero_at_step3"]
            and d["B_grad_addressable"]
        )
        detail["per_arm"][arm] = d
        all_pass = all_pass and d["passed"]
    record(f"B8_lowrank_gradient_{label}", all_pass, detail)


def check_empty_neighbourhood(models: Mapping[str, Any], spec: Any, fixed1: Any) -> None:
    n_atoms = 1
    batch = {
        "n_graphs": 1,
        "atom_feat": torch.zeros((n_atoms, spec.atom_feat_dim)),
        "atom_scalar": torch.zeros((n_atoms, spec.n_scalars)),
        "atom_type": torch.zeros(n_atoms, dtype=torch.long),
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
        "bond_feat": torch.zeros((0, spec.bond_feat_dim)),
        "bond_type": torch.zeros(0, dtype=torch.long),
        "bond_batch": torch.zeros(0, dtype=torch.long),
        "global": torch.zeros((1, spec.global_dim)),
        "atom_counts": torch.tensor([n_atoms], dtype=torch.long),
        "bond_counts": torch.tensor([0], dtype=torch.long),
        "atom_batch": torch.zeros(n_atoms, dtype=torch.long),
        "atom_off": torch.zeros(1, dtype=torch.long),
        "nbr_atom": torch.zeros(0, dtype=torch.long),
        "nbr_code": torch.zeros(0, dtype=torch.long),
        "path_atom": torch.zeros(0, dtype=torch.long),
        "path_cat": torch.zeros(0, dtype=torch.long),
        "path_end63": torch.zeros(0, dtype=torch.long),
        "path_end": torch.zeros(0, dtype=torch.long),
        "path_closure": torch.zeros(0, dtype=torch.long),
        "path_graph": torch.zeros(0, dtype=torch.long),
    }
    detail: dict[str, Any] = {"per_arm": {}}
    all_pass = True
    for arm, (model, _audit) in models.items():
        model.eval()
        with torch.inference_mode():
            logits, states = model(batch, return_states=True)
            m2 = states["info"]["m2"]
        d = {
            "logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "m2_zero": bool(m2.abs().max().item() == 0.0),
            "h1_finite": bool(torch.isfinite(states["h1"]).all().item()),
            "h2_finite": bool(torch.isfinite(states["h2"]).all().item()),
        }
        d["passed"] = bool(d["logits_finite"] and d["m2_zero"] and d["h1_finite"] and d["h2_finite"])
        detail["per_arm"][arm] = d
        all_pass = all_pass and d["passed"]
    fixed1.eval()
    with torch.inference_mode():
        ref = fixed1(batch)
    detail["fixed1_reference_finite"] = bool(torch.isfinite(ref).all().item())
    record("B9_empty_neighbourhood", all_pass and detail["fixed1_reference_finite"], detail)


def check_no_test_access() -> None:
    here = Path(__file__).resolve().parent
    forbidden = []
    for name in ("frs_train.py", "frs_mechanism.py"):
        text = (here / name).read_text(encoding="utf-8")
        if 'evaluate_split(' in text and '"test"' in text:
            forbidden.append(f"{name}: evaluate_split with a test split")
        if "load_test_labels" in text:
            forbidden.append(f"{name}: load_test_labels")
        if "test_rows" in text:
            forbidden.append(f"{name}: test_rows")
    detail = {"files_scanned": ["frs_train.py", "frs_mechanism.py"], "violations": forbidden,
              "test_evaluated": False, "test_used_for_selection": False}
    record("B10_no_test_access_static", not forbidden, detail)


def check_data_identity(task: Any) -> None:
    hist = FC.read_json(Path(FC.ANALYSIS.parent / ".." / ".." / ".." / "tracks/structural-sufficiency"
                             / "direct_structural_computation_recurrence_v0/analysis/checks.json"))
    rec = next(r for r in hist["checks"] if r["check"].startswith("A2.1_data_identity"))
    d = rec["detail"]
    hist_identity = {
        "feature_spec_hash": d.get("expected_feature_spec_hash", d.get("feature_spec_hash")),
        "norm_fingerprint": d.get("expected_norm_fingerprint", d.get("norm_fingerprint")),
        "split_partition_sha256": d.get("expected_split_partition_sha256", d.get("split_partition_sha256")),
        "label_artifact_sha256": d.get("expected_label_artifact_sha256", d.get("label_artifact_sha256")),
    }
    identity = FC.data_identity(task, {"data_identity": hist_identity})
    split_ok = (
        d.get("expected_split_sizes") is not None
        and {k: int(v) for k, v in d["expected_split_sizes"].items()}
        == {"train": int(C.N_TRAIN), "valid": int(C.N_VALID), "test": int(C.N_TEST)}
    )
    identity["historical_expected_split_sizes"] = d.get("expected_split_sizes")
    record("B0_data_identity",
           bool(identity["all_match"] and split_ok and all(identity["count_checks"].values())),
           {"identity": identity, "split_sizes_match": split_ok,
            "historical_source": "dscr-v0 analysis/checks.json A2.1"})


def check_real_batch(models: Mapping[str, Any], fixed1: Any, task: Any, batch: Mapping[str, Any]) -> None:
    check_zero_A_equivalence(models, fixed1, batch, label="real_valid_8")
    check_invariance(models, batch, label="real_valid_8")
    check_gradients(models, batch, label="real_valid_8")
    rows = [int(x) for x in task.train_rows[:8]]
    train_batch = {k: (v.to(next(fixed1.parameters()).device) if torch.is_tensor(v) else v)
                   for k, v in C.PC.collate(task.cache, rows).items()}
    check_zero_A_equivalence(models, fixed1, train_batch, label="real_train_8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--out", default=str(FC.ANALYSIS / "checks.json"))
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    FC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    full = args.mode == "full"
    spec = DCHK.load_spec()
    norm = DCHK.fake_norm()
    task = None
    if full:
        task = C.SD.load_task()
        norm = task.norm
        check_no_test_access()
        check_data_identity(task)
    else:
        DSM.STRICT_W2_FINGERPRINT = False
        check_no_test_access()
    models, (fixed1, fixed1_audit) = initial_models(spec, norm, verify=full)
    check_parameters(models)
    check_init_identity(models, full=full)
    synth = DCHK.synthetic_star(spec)
    check_zero_A_equivalence(models, fixed1, synth, label="synthetic_star")
    check_role_input(models)
    check_dual_form(models)
    check_fixed_source(models)
    check_invariance(models, synth, label="synthetic_star")
    check_gradients(models, synth, label="synthetic_star")
    check_empty_neighbourhood(models, spec, fixed1)
    if full:
        rows = [int(x) for x in task.valid_rows[:8]]
        batch = {k: (v.to(torch.device("cpu")) if torch.is_tensor(v) else v)
                 for k, v in C.PC.collate(task.cache, rows).items()}
        check_real_batch(models, fixed1, task, batch)
    payload = {
        "schema": FC.SCHEMA_CHECKS,
        "round_id": FC.ROUND_ID,
        "mode": args.mode,
        "seconds": float(time.perf_counter() - t0),
        "n_checks": len(RECORDS),
        "n_passed": sum(1 for r in RECORDS if r["passed"]),
        "all_passed": all(r["passed"] for r in RECORDS),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "checks": RECORDS,
        "created_utc": C.utcnow(),
    }
    FC.write_json(Path(args.out), payload)
    FC.log(f"checks: {payload['n_passed']}/{payload['n_checks']} passed in {payload['seconds']:.1f}s")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
