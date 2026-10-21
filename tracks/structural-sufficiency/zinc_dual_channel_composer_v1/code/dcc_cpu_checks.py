"""Focused CPU gates for the dual-channel composer.

These checks are intentionally narrow: parameter/initial-state identity, the
organized/Null grouping identities, permutation and batch-composition
invariance, first/second-step gradient flow, checkpoint round-trip, and the
trainer's parameter coverage.  No training result is produced here.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SUFF = _HERE.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_V2 = _SUFF / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import dcc_models as M  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)
EXPECTED_PARAMS = 494_193


def add(checks: dict, name: str, passed: bool, detail) -> None:
    checks[name] = {"pass": bool(passed), "detail": detail}


def state_equal(a: M.DualChannel, b: M.DualChannel) -> tuple[bool, float]:
    sa, sb = a.state_dict(), b.state_dict()
    if list(sa) != list(sb):
        return False, float("inf")
    maxdiff = 0.0
    same = True
    for k in sa:
        if sa[k].shape != sb[k].shape:
            return False, float("inf")
        if not torch.equal(sa[k], sb[k]):
            same = False
            maxdiff = max(maxdiff, float((sa[k] - sb[k]).abs().max()))
    return same, maxdiff


def permute_single_graph(batch: dict, seed: int) -> dict:
    """Relabel atoms, bonds and junctions inside a one-graph batch."""
    gen = torch.Generator().manual_seed(int(seed))
    N = int(batch["atom_feat"].shape[0])
    E = int(batch["bond_feat"].shape[0])
    J = int(batch["junc_feat"].shape[0])
    a_order = torch.randperm(N, generator=gen)
    b_order = torch.randperm(E, generator=gen)
    j_order = torch.randperm(J, generator=gen)
    a_inv = torch.empty(N, dtype=torch.long)
    a_inv[a_order] = torch.arange(N)
    b_inv = torch.empty(E, dtype=torch.long)
    b_inv[b_order] = torch.arange(E)
    out = dict(batch)
    out["atom_feat"] = batch["atom_feat"][a_order]
    out["atom_type"] = batch["atom_type"][a_order]
    out["bond_feat"] = batch["bond_feat"][b_order]
    out["bond_type"] = batch["bond_type"][b_order]
    out["junc_feat"] = batch["junc_feat"][j_order]
    out["bond_index"] = a_inv[batch["bond_index"][b_order]]
    out["junc_index"] = a_inv[batch["junc_index"]][j_order]
    out["junc_bond"] = b_inv[batch["junc_bond"]][j_order]
    return out


def eval_pred(model: torch.nn.Module, batch: dict) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(batch).detach().cpu().numpy()


def main() -> int:
    checks: dict = {}
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    valid = D.load_cache("valid")
    norm = D.load_norm()
    rows = np.arange(64, dtype=np.int64)
    batch = T1.collate(valid, rows, "P3N", valid=True)

    # 1) Reference P3 and shared initial state.
    p3_ref, p3_info = A2.build_model("P3N", norm, 0, "cpu")
    shared, init_info = M.build_initial_state(norm, 0)
    p3_state = p3_ref.state_dict()
    shared_state = shared.state_dict()
    base_diff = 0.0
    for k, v in p3_state.items():
        if k.startswith("reader."):
            continue
        base_diff = max(base_diff, float((shared_state[k] - v).abs().max()))
    reader_old_diff = float(
        (shared_state["reader.lins.0.weight"][:, :321] - p3_state["reader.lins.0.weight"]).abs().max()
    )
    reader_new_max = float(shared_state["reader.lins.0.weight"][:, 321:].abs().max())
    add(checks, "p3_reference_state_hash_matches_builder",
        init_info["p3_reference_state_sha256"] == C.state_hash(p3_ref),
        {"expected": C.state_hash(p3_ref), "actual": init_info["p3_reference_state_sha256"]})
    add(checks, "p3_base_weights_exact_minus_reader", base_diff == 0.0, {"max_abs_diff": base_diff})
    add(checks, "p3_reader_first_321_columns_exact", reader_old_diff == 0.0, {"max_abs_diff": reader_old_diff})
    add(checks, "p3_reader_new_columns_zero", reader_new_max == 0.0, {"max_abs": reader_new_max})

    L = copy.deepcopy(shared)
    Cc = copy.deepcopy(shared)
    L.organized_local = True
    Cc.organized_local = False

    # 2) Parameters and identical initial state.
    pl = C.parameter_count(L)
    pc = C.parameter_count(Cc)
    same_state, state_diff = state_equal(L, Cc)
    add(checks, "parameter_count_494193", pl == EXPECTED_PARAMS and pc == EXPECTED_PARAMS,
        {"L": pl, "C": pc, "expected": EXPECTED_PARAMS, "headroom": 500000 - pl})
    add(checks, "same_keys_shapes_and_initial_state", same_state,
        {"state_dict_equal": same_state, "max_abs_diff": state_diff,
         "initial_state_sha256": init_info["initial_state_sha256"]})
    add(checks, "parameters_breakdown",
        init_info["parameters_breakdown"]["local_composer"] == 31_536
        and init_info["parameters_breakdown"]["reader"] == 197_633,
        init_info["parameters_breakdown"])

    # 3) Initial forward equals P3 for both arms.
    y_ref = eval_pred(p3_ref, batch)
    y_l = eval_pred(L, batch)
    y_c = eval_pred(Cc, batch)
    d_l = float(np.max(np.abs(y_l - y_ref)))
    d_c = float(np.max(np.abs(y_c - y_ref)))
    d_lc = float(np.max(np.abs(y_l - y_c)))
    add(checks, "initial_forward_L_vs_P3_le_1e-6", d_l <= 1e-6, {"max_abs_diff": d_l})
    add(checks, "initial_forward_C_vs_P3_le_1e-6", d_c <= 1e-6, {"max_abs_diff": d_c})
    add(checks, "initial_forward_L_equals_C", d_lc <= 1e-7, {"max_abs_diff": d_lc})

    # 4) Organized incidence identities and exact Null control.
    a, b, w = L.encoders(batch)
    Wl, Vl, n_b = L.local_grouping(w, batch, organized=True)
    Wn, Vn, n_null = L.local_grouping(w, batch, organized=False)
    Wg, Vg, n_g = p3_ref.group_to_bonds(w, batch)
    jb = batch["junc_bond"]
    E = int(batch["bond_feat"].shape[0])
    W_dense = torch.zeros_like(Wl)
    V_dense = torch.zeros_like(Vl)
    for j in range(jb.shape[0]):
        W_dense[jb[j, 0]] += w[j]
        W_dense[jb[j, 1]] += w[j]
        V_dense[jb[j, 0]] += w[j] * w[j]
        V_dense[jb[j, 1]] += w[j] * w[j]
    sum_ok = bool(torch.allclose(Wl.sum(0), 2.0 * w.sum(0), atol=1e-5, rtol=1e-5)
                  and torch.allclose(Vl.sum(0), 2.0 * (w * w).sum(0), atol=1e-5, rtol=1e-5))
    dense_diff = float((Wl - W_dense).abs().max())
    dense_v_diff = float((Vl - V_dense).abs().max())
    inc = torch.bincount(jb.reshape(-1), minlength=E)
    add(checks, "organized_each_junction_enters_two_bonds",
        bool((jb[:, 0] != jb[:, 1]).all() and int(inc.sum()) == 2 * int(jb.shape[0])),
        {"n_junctions": int(jb.shape[0]), "sum_bond_incidence": int(inc.sum()),
         "distinct_endpoint_pairs": bool((jb[:, 0] != jb[:, 1]).all())})
    add(checks, "organized_scatter_matches_dense",
        dense_diff <= 1e-4 and dense_v_diff <= 1e-4,
        {"W_max_abs_diff": dense_diff, "V_max_abs_diff": dense_v_diff})
    add(checks, "organized_sum_identity", sum_ok,
        {"sum_W_vs_2sum_w_max_abs": float((Wl.sum(0) - 2.0 * w.sum(0)).abs().max()),
         "sum_V_vs_2sum_w2_max_abs": float((Vl.sum(0) - 2.0 * (w * w).sum(0)).abs().max())})
    add(checks, "null_control_exact", bool(torch.equal(Wn, Wg) and torch.equal(Vn, Vg) and torch.equal(n_null, n_g)),
        {"W_equal": bool(torch.equal(Wn, Wg)), "V_equal": bool(torch.equal(Vn, Vg)),
         "n_equal": bool(torch.equal(n_null, n_g))})

    # 5) Empty-junction and single-junction boundaries.
    empty_w = torch.zeros(0, 96)
    empty_batch = {
        "n_graphs": 1,
        "bond_feat": torch.zeros(3, 7),
        "junc_bond": torch.zeros(0, 2, dtype=torch.long),
        "junc_batch": torch.zeros(0, dtype=torch.long),
        "bond_batch": torch.zeros(3, dtype=torch.long),
    }
    eW, eV, eN = L.local_grouping(empty_w, empty_batch, organized=True)
    eWn, eVn, eNn = p3_ref.group_to_bonds(empty_w, empty_batch)
    one_w = torch.randn(1, 96)
    one_batch = {
        "n_graphs": 1,
        "bond_feat": torch.zeros(3, 7),
        "junc_bond": torch.tensor([[0, 1]], dtype=torch.long),
        "junc_batch": torch.tensor([0], dtype=torch.long),
        "bond_batch": torch.zeros(3, dtype=torch.long),
    }
    oW, oV, oN = L.local_grouping(one_w, one_batch, organized=True)
    oWn, oVn, oNn = p3_ref.group_to_bonds(one_w, one_batch)
    add(checks, "empty_junction_finite_zero",
        bool(torch.isfinite(eW).all() and torch.isfinite(eV).all() and eW.abs().max() == 0 and eV.abs().max() == 0
             and torch.isfinite(eWn).all() and torch.isfinite(eVn).all()),
        {"organized_max": float(eW.abs().max()), "null_max": float(eWn.abs().max())})
    add(checks, "single_junction_finite_and_equals_null",
        bool(torch.isfinite(oW).all() and torch.isfinite(oV).all()
             and torch.allclose(oW, oWn) and torch.allclose(oV, oVn)),
        {"max_W_diff": float((oW - oWn).abs().max()), "max_V_diff": float((oV - oVn).abs().max())})

    # 6) Permutation and batch-composition invariance.
    single = T1.collate(valid, [17], "P3N", valid=True)
    pair = T1.collate(valid, [17, 42], "P3N", valid=True)
    p_single = eval_pred(L, single)[0]
    p_pair = eval_pred(L, pair)[0]
    p_perm = eval_pred(L, permute_single_graph(single, 12345))[0]
    add(checks, "batch_composition_invariance", abs(float(p_single) - float(p_pair)) <= 1e-6,
        {"single": float(p_single), "in_pair": float(p_pair), "abs_diff": abs(float(p_single) - float(p_pair))})
    add(checks, "relabeling_invariance", abs(float(p_single) - float(p_perm)) <= 1e-6,
        {"single": float(p_single), "relabeled": float(p_perm), "abs_diff": abs(float(p_single) - float(p_perm))})

    # 7) Gradient flow: reader new columns at step 1, local composer from step 2.
    gbatch = T1.collate(valid, np.arange(32), "P3N", valid=True)
    gy = torch.as_tensor(np.asarray(valid["labels"])[:32], dtype=torch.float32)
    opt = torch.optim.AdamW(L.parameters(), lr=1e-3, weight_decay=1e-5)
    L.train()
    opt.zero_grad(set_to_none=True)
    loss1 = F.l1_loss(L(gbatch).view(-1), gy)
    loss1.backward()
    new_col_grad = L.reader.lins[0].weight.grad[:, 321:513]
    new_col_nonzero = int(torch.count_nonzero(new_col_grad).item())
    new_col_finite = bool(torch.isfinite(new_col_grad).all())
    local_grad_norm_1 = float(torch.sqrt(sum((p.grad * p.grad).sum() for p in L.local_composer.parameters() if p.grad is not None)))
    local_nonzero_1 = int(sum(int(torch.count_nonzero(p.grad).item()) for p in L.local_composer.parameters() if p.grad is not None))
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss2 = F.l1_loss(L(gbatch).view(-1), gy)
    loss2.backward()
    local_grad_norm_2 = float(torch.sqrt(sum((p.grad * p.grad).sum() for p in L.local_composer.parameters() if p.grad is not None)))
    local_nonzero_2 = int(sum(int(torch.count_nonzero(p.grad).item()) for p in L.local_composer.parameters() if p.grad is not None))
    local_finite_2 = bool(all(torch.isfinite(p.grad).all() for p in L.local_composer.parameters() if p.grad is not None))
    add(checks, "reader_new_columns_grad_step1",
        new_col_nonzero > 0 and new_col_finite,
        {"nonzero_elements": new_col_nonzero, "finite": new_col_finite,
         "grad_norm": float(torch.sqrt((new_col_grad * new_col_grad).sum()))})
    add(checks, "local_composer_zero_grad_step1_expected",
        local_nonzero_1 == 0 and local_grad_norm_1 == 0.0,
        {"nonzero_elements": local_nonzero_1, "grad_norm": local_grad_norm_1})
    add(checks, "local_composer_nonzero_finite_grad_step2",
        local_nonzero_2 > 0 and local_finite_2 and np.isfinite(local_grad_norm_2),
        {"nonzero_elements": local_nonzero_2, "grad_norm": local_grad_norm_2, "finite": local_finite_2})

    # 8) Checkpoint round-trip.
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "best.pt"
        torch.save({"state_dict": C.cpu_state(L), "arm": "L", "epoch": 0}, ckpt)
        restored = M.DualChannel(norm)
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        restored.load_state_dict(obj["state_dict"])
        restored.organized_local = True
        p_restored = eval_pred(restored, gbatch)
        p_l = eval_pred(L, gbatch)
        ckpt_diff = float(np.max(np.abs(p_restored - p_l)))
    add(checks, "checkpoint_save_load_consistent", ckpt_diff == 0.0, {"max_abs_diff": ckpt_diff})

    # 9) Trainer parameter coverage and arm switch.
    opt_all = torch.optim.AdamW(L.parameters(), lr=1e-3, weight_decay=1e-5)
    audit = T1._param_audit(L, opt_all)
    add(checks, "trainer_optimizer_covers_all_parameters",
        audit["parameters"] == EXPECTED_PARAMS and audit["optimizer_parameters"] == EXPECTED_PARAMS
        and audit["all_parameters_require_grad"],
        audit)

    L2 = copy.deepcopy(shared)
    C2 = copy.deepcopy(shared)
    L2.organized_local = True
    C2.organized_local = False
    w_before = L2.reader.lins[0].weight.detach().clone()
    for model, flag in ((L2, True), (C2, False)):
        opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        model.train()
        opt2.zero_grad(set_to_none=True)
        F.l1_loss(model(gbatch).view(-1), gy).backward()
        opt2.step()
    arm_diff = float((L2.reader.lins[0].weight - C2.reader.lins[0].weight).abs().max())
    add(checks, "arm_switch_changes_first_update",
        C.state_hash(L2) != C.state_hash(C2) and arm_diff > 0.0,
        {"state_hashes_differ": C.state_hash(L2) != C.state_hash(C2), "reader_max_abs_diff": arm_diff,
         "L_organized": bool(L2.organized_local), "C_organized": bool(C2.organized_local)})

    # 10) Builder metadata and same initial hash from both arms.
    ml, il = M.build_model("DCC_L", norm, 0, "cpu", organized_local=True)
    mc, ic = M.build_model("DCC_C", norm, 0, "cpu", organized_local=False)
    add(checks, "builder_arm_metadata_and_same_init_hash",
        il["initial_state_sha256"] == ic["initial_state_sha256"] == init_info["initial_state_sha256"]
        and il["arm"] == "L" and ic["arm"] == "C",
        {"L_hash": il["initial_state_sha256"], "C_hash": ic["initial_state_sha256"],
         "shared_hash": init_info["initial_state_sha256"]})

    # 11) No official-test access in this track's code.
    forbidden = []
    for path in sorted((_HERE.parent / "code").glob("*.py")):
        if path.name == "dcc_cpu_checks.py":
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("ad_test", "path_test", "load_cache(\"test\")", "official_test_loaded\": true"):
            if token in text:
                forbidden.append(f"{path.name}:{token}")
    add(checks, "no_official_test_paths", not forbidden, {"forbidden_hits": forbidden})

    all_pass = all(v["pass"] for v in checks.values())
    out = {
        "protocol_id": "zinc-dual-channel-composer-v1",
        "all_pass": bool(all_pass),
        "official_test_loaded": False,
        "checks": checks,
    }
    C.write_json(ANALYSIS / "cpu_checks.json", out)
    C.write_json(ANALYSIS / "initial_state_hashes.json", {
        "seed": 0,
        "initial_state_sha256": init_info["initial_state_sha256"],
        "p3_reference_state_sha256": init_info["p3_reference_state_sha256"],
        "parameters": pl,
        "parameters_breakdown": init_info["parameters_breakdown"],
        "ssl": p3_info.get("ssl"),
    })
    print(json.dumps(out, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
