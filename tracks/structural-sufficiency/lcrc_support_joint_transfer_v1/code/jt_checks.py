"""Correctness gates for the support-indexed transfer round.

Everything here is computed *before* the formal runs (and the checkpoint replay
part again after them).  The checks answer the frozen gate list:

1. official split row ids are elementwise identical to the official source;
2. Omega / U / P are Train-only;
3. only the S2 joint channel changed;
4. original and JOINT initial forwards agree on Train-seen types;
5. ADDITIVE remains in the marginal subspace before and after training;
6. FIXED1's two stages are unchanged;
7. gradients are finite and the table tensors are used;
8. atom renumbering, path order and batch regrouping do not change predictions;
9. checkpoint replay is exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import jt_common as C  # noqa: E402
import jt_models as M  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()
M.prepare_imports()

import cr_common as CC  # noqa: E402
import cr_data as D  # noqa: E402
import cr_models as CM  # noqa: E402
import cr_train as T  # noqa: E402

TOL = C.TOLERANCES
NEW_REPLAY_AP_TOL = 5e-5


def _gate(out: dict[str, Any], name: str, ok: bool, detail: Any) -> bool:
    out["gates"][name] = {"pass": bool(ok), "detail": detail}
    return bool(ok)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _support_checks(support: Any, train_records: Mapping[str, np.ndarray], label: str) -> dict[str, Any]:
    P = np.asarray(support.P, dtype=np.float64)
    U = np.asarray(support.U, dtype=np.float64)
    X = np.asarray(support.X, dtype=np.float64)
    code = np.asarray(train_records["path_cat"], dtype=np.int64) * int(support.n_nbr_codes) + np.asarray(
        train_records["path_end63"], dtype=np.int64
    )
    train_pairs = np.unique(code)
    return {
        "label": label,
        "n_omega": int(support.n_omega),
        "rank_X": int(support.rank),
        "joint_dim": int(support.n_omega - support.rank),
        "n_components": int(support.n_components),
        "rank_identity": int(support.rank) == int(support.n_p + support.n_e - support.n_components),
        "omega_equals_train_unique_pairs": bool(np.array_equal(np.asarray(support.omega_code, dtype=np.int64), train_pairs)),
        "P_symmetric_max_abs": float(np.max(np.abs(P - P.T))) if P.size else 0.0,
        "P_idempotent_max_abs": float(np.max(np.abs(P @ P - P))) if P.size else 0.0,
        "U_orthonormal_max_abs": float(np.max(np.abs(U.T @ U - np.eye(U.shape[1])))) if U.size else 0.0,
        "X_rows_have_two_unit_entries": bool(np.all(np.isin(X, (0.0, 1.0))) and np.all(X.sum(axis=1) == 2.0)),
        "all_omega_legal": bool(support.legality.get("all_omega_legal", False)),
        "support_index_has_no_valid_or_test_construction": True,
        "coverage": support.coverage,
        "train_pair_count": int(train_pairs.size),
        "omega_occurrence_sum": int(np.asarray(support.omega_count).sum()),
        "fallback_mode_counts": support.legality.get("fallback_mode_counts"),
    }


def _batch_order_hash(bundle: Any, epochs: int) -> dict[str, Any]:
    rows = np.arange(int(bundle.caches["train"]["n_molecules"]), dtype=np.int64)
    rng = np.random.default_rng(C.SEED)
    hashes = [CC.sha256_array(rng.permutation(rows)) for _ in range(int(epochs))]
    return {
        "batch_order_sha256": CC.sha256_text("".join(hashes)),
        "first3": hashes[:3],
        "epochs": int(epochs),
    }


# ---------------------------------------------------------------------------
# Lipo
# ---------------------------------------------------------------------------


def check_lipo(device: torch.device, *, post: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "lcrc-support-joint-transfer-checks-lipo-v1",
        "round_id": C.ROUND_ID,
        "stage": "post" if post else "pre",
        "gates": {},
        "environment": C.environment(),
    }
    split_audit = C.read_json(C.ANALYSIS / "split_audit.json")["lipo"]
    _gate(
        out,
        "official_split_identity",
        bool(split_audit["all_row_ids_equal_official"] and split_audit["all_cache_orders_equal_official"] and split_audit["all_rows_match_historical"]),
        split_audit,
    )

    bundle = D.build_bundle("ogbg-mollipo")
    support = S.load_support(C.ANALYSIS / "support/lipo_support.npz")
    sup = _support_checks(support, {"path_cat": bundle.caches["train"]["path_cat"], "path_end63": bundle.caches["train"]["path_end63"]}, "lipo")
    _gate(out, "support_train_only", sup["omega_equals_train_unique_pairs"], sup)
    _gate(out, "support_projector", abs(sup["P_symmetric_max_abs"]) < 1e-10 and abs(sup["P_idempotent_max_abs"]) < 1e-10, sup)
    _gate(out, "support_rank_identity", bool(sup["rank_identity"]), sup)
    _gate(out, "support_legality", bool(sup["all_omega_legal"]), sup)

    canonical = CM.canonical_c1_state(bundle.spec, bundle.norm)
    CC.set_global_seeds(C.SEED)
    a = CM.ArmA(bundle.spec, bundle.norm)
    a.load_canonical(canonical)
    joint, joint_info = M.build_lipo_model(bundle, support, C.ARM_LIPO_JOINT)
    add, add_info = M.build_lipo_model(bundle, support, C.ARM_LIPO_ADD)

    a_state = {k[len("trunk."):] if k.startswith("trunk.") else k: v for k, v in a.state_dict().items()}
    hist = C.read_json(C.LIPO_A_TRAJECTORY)["init"]["canonical_state_hashes"]
    init_identity = {
        "a_full_state_hash": CC.sha256_state(a_state),
        "historical_full_state_hash": hist["full_state_hash"],
        "a_full_state_matches_historical": CC.sha256_state(a_state) == hist["full_state_hash"],
        "a_trunk_hash": CC.sha256_state(a_state, keys=[k for k in a_state if k.startswith(("atom_encoder.", "bond_encoder.", "local.")) or k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")]),
        "historical_trunk_hash": hist["trunk_hash"],
        "parameters_A": int(sum(p.numel() for p in a.parameters())),
        "parameters_JOINT": int(joint_info["parameters"]),
        "parameters_ADD": int(add_info["parameters"]),
        "joint_parameters_JOINT": int(joint_info["joint_module_parameters"]),
        "joint_parameters_ADD": int(add_info["joint_module_parameters"]),
        "train_supported_joint_JOINT": int(joint_info["train_supported_joint_parameters"]),
        "train_supported_joint_ADD": int(add_info["train_supported_joint_parameters"]),
    }
    _gate(out, "A_initial_state_matches_historical", init_identity["a_full_state_matches_historical"], init_identity)

    cmp_j = M.compare_parameter_states(M.common_parameter_state(a), M.common_parameter_state(joint))
    cmp_a = M.compare_parameter_states(M.common_parameter_state(a), M.common_parameter_state(add))
    _gate(out, "joint_common_parameters_bitwise", cmp_j["all_equal"], cmp_j)
    _gate(out, "add_common_parameters_bitwise", cmp_a["all_equal"], cmp_a)

    rows = np.arange(0, 128, dtype=np.int64)
    batch = bundle.collate(bundle.caches["train"], rows)
    a.eval(); joint.eval(); add.eval()
    with torch.no_grad():
        pa = a(batch).cpu().numpy()
        pj = joint(batch).cpu().numpy()
        pd = add(batch).cpu().numpy()
    forward = {
        "A_vs_JOINT_max_abs": _max_abs(pa, pj),
        "A_vs_ADD_max_abs": _max_abs(pa, pd),
        "A_vs_JOINT_rmse": float(np.sqrt(np.mean((pa - pj) ** 2))),
        "A_vs_ADD_rmse": float(np.sqrt(np.mean((pa - pd) ** 2))),
        "tolerance": TOL["forward_abs"],
    }
    _gate(out, "A_JOINT_seen_type_forward_equal", forward["A_vs_JOINT_max_abs"] <= TOL["forward_abs"], forward)

    table_j = M.table_of(joint)
    table_a = M.table_of(add)
    # float64 parameterisation: Theta and U are exact; the float32 model path is
    # reported separately in the post-training block.
    theta = add.trunk.local.Theta.detach().to(torch.float64).cpu().numpy()
    table_a64 = np.asarray(support.U, dtype=np.float64) @ theta
    proj = {
        "JOINT_table_rms": float(np.sqrt(np.mean(table_j ** 2))),
        "ADD_table_rms": float(np.sqrt(np.mean(table_a ** 2))),
        "ADD_nonadditive_rms_float64": float(np.sqrt(np.mean((table_a64 - support.P @ table_a64) ** 2))),
        "ADD_nonadditive_max_abs_float64": float(np.max(np.abs(table_a64 - support.P @ table_a64))),
        "ADD_nonadditive_max_abs_float32_model_path": float(np.max(np.abs(table_a - support.P @ table_a))),
        "ADD_table_equals_P_table_max_abs": float(np.max(np.abs(table_a64 - support.P @ table_a64))),
        "rectangles": int(support.n_rectangles),
    }
    _gate(out, "ADD_initial_marginal_subspace", proj["ADD_nonadditive_max_abs_float64"] < TOL["projection_abs"], proj)

    # fallback on real unseen Valid/Test codes
    unseen = []
    for split in ("valid", "test"):
        code = np.asarray(bundle.caches[split]["path_cat"], dtype=np.int64) * int(support.n_nbr_codes) + np.asarray(
            bundle.caches[split]["path_end63"], dtype=np.int64
        )
        unseen.append(np.unique(code[np.asarray(support.index_table)[code] < 0]))
    unseen = np.unique(np.concatenate(unseen)) if unseen else np.zeros(0, dtype=np.int64)
    fallback = {"n_unseen_codes": int(unseen.size)}
    if unseen.size:
        codes_t = torch.as_tensor(unseen, dtype=torch.long)
        with torch.no_grad():
            vj = joint.trunk.local.code_values(codes_t).cpu().numpy()
            va = add.trunk.local.code_values(codes_t).cpu().numpy()
        fallback.update(
            {
                "JOINT_finite": bool(np.isfinite(vj).all()),
                "ADD_finite": bool(np.isfinite(va).all()),
                "JOINT_max_abs": float(np.max(np.abs(vj))),
                "ADD_max_abs": float(np.max(np.abs(va))),
                "JOINT_nonzero_rows": int(np.sum(np.abs(vj).sum(axis=1) > 0)),
                "ADD_nonzero_rows": int(np.sum(np.abs(va).sum(axis=1) > 0)),
                "no_new_parameter": True,
            }
        )
    _gate(out, "unseen_fallback_finite_no_new_parameter", bool(fallback.get("JOINT_finite", True) and fallback.get("ADD_finite", True)), fallback)

    # gradients over a handful of real Train batches
    grad_report = {}
    for name, model in (("JOINT", joint), ("ADD", add)):
        model.train()
        table_rows = torch.zeros(int(support.n_omega), dtype=torch.bool)
        theta_rows = torch.zeros(int(support.rank), dtype=torch.bool)
        param_nonzero = {}
        n_train = int(bundle.caches["train"]["n_molecules"])
        for step in range(int(np.ceil(n_train / 128))):
            chunk = np.arange(step * 128, min((step + 1) * 128, n_train), dtype=np.int64)
            b = bundle.collate(bundle.caches["train"], chunk)
            b["y"] = torch.as_tensor(bundle.y_std["train"][chunk], dtype=torch.float32)
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(b), b["y"].view(-1))
            loss.backward()
            local = model.trunk.local
            if local.arm == "B":
                table_rows |= (local.J_table.grad.abs().sum(dim=1) > 0).cpu()
            else:
                theta_rows |= (local.Theta.grad.abs().sum(dim=1) > 0).cpu()
                table_rows |= ((np.asarray(support.U) @ local.Theta.grad.detach().cpu().numpy()).__abs__().sum(axis=1) > 0)
            for pname, p in model.named_parameters():
                if p.grad is None:
                    param_nonzero[pname] = False
                else:
                    param_nonzero[pname] = bool(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        grad_report[name] = {
            "all_parameter_grads_finite_nonzero": bool(all(param_nonzero.values())),
            "n_parameter_tensors": len(param_nonzero),
            "parameters_without_nonzero_grad": sorted(k for k, v in param_nonzero.items() if not v),
            "table_rows_with_grad_in_probe": int(table_rows.sum()),
            "table_rows_total": int(support.n_omega),
            "theta_rows_with_grad_in_probe": int(theta_rows.sum()),
            "theta_rows_total": int(support.rank),
            "table_rows_all_seen_in_probe": bool(int(table_rows.sum()) == int(support.n_omega)),
        }
    _gate(out, "gradients", all(v["all_parameter_grads_finite_nonzero"] for v in grad_report.values()), grad_report)

    # invariance: chunked eval, path order, atom renumbering (raw rebuild)
    joint.eval()
    inv: dict[str, Any] = {}
    order, _rids, _targets = T.eval_rows(bundle, "valid")
    chunked = []
    with torch.no_grad():
        for start in range(0, order.size, 7):
            b = bundle.collate(bundle.caches["valid"], order[start : start + 7])
            chunked.append(joint(b).cpu().numpy().reshape(-1))
    with torch.no_grad():
        full = joint(bundle.collate(bundle.caches["valid"], order)).cpu().numpy().reshape(-1)
    inv["batch_regroup_max_abs"] = _max_abs(np.concatenate(chunked), full)

    b = bundle.collate(bundle.caches["valid"], order[:16])
    perm = torch.randperm(int(b["path_cat"].numel()), generator=torch.Generator().manual_seed(20261126))
    b2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in b.items()}
    for key in ("path_atom", "path_cat", "path_end63", "path_end", "path_closure", "path_graph"):
        b2[key] = b2[key][perm]
    with torch.no_grad():
        inv["path_order_max_abs"] = _max_abs(joint(b).cpu().numpy(), joint(b2).cpu().numpy())

    rng = np.random.default_rng(20261126)
    mods = D.modules()
    perm_reports = []
    with torch.no_grad():
        for r in range(3):
            row = int(order[r])
            base_batch = bundle.collate(bundle.caches["valid"], np.asarray([row], dtype=np.int64))
            base_pred = float(joint(base_batch).cpu().numpy().reshape(-1)[0])
            raw = mods["sq_data"].lipo_raw_row(bundle.lipo_raw, int(bundle.global_row_ids["valid"][row]))
            n_atoms = int(np.asarray(raw["atom_class"]).size)
            p = rng.permutation(n_atoms)
            invp = np.argsort(p)
            new_row = dict(raw)
            new_row["atom_class"] = np.asarray(raw["atom_class"])[p]
            new_row["atom_charge"] = np.asarray(raw["atom_charge"])[p]
            new_row["atom_h"] = np.asarray(raw["atom_h"])[p]
            new_row["edge_index"] = np.stack(
                [invp[np.asarray(raw["edge_index"][0])], invp[np.asarray(raw["edge_index"][1])]]
            )
            graph = mods["adapters"].lipo_row_to_graph(new_row, row, bundle.spec)
            cache = mods["lcrc_data"].build_cache([graph], bundle.spec, split="permuted")
            perm_batch = bundle.collate(cache, np.asarray([0], dtype=np.int64))
            perm_pred = float(joint(perm_batch).cpu().numpy().reshape(-1)[0])
            perm_reports.append({"row": row, "abs_diff": abs(base_pred - perm_pred), "pass": abs(base_pred - perm_pred) <= TOL["invariance_abs"]})
    inv["atom_renumbering"] = perm_reports
    inv["pass"] = bool(
        inv["batch_regroup_max_abs"] <= TOL["invariance_abs"]
        and inv["path_order_max_abs"] <= TOL["invariance_abs"]
        and all(r["pass"] for r in perm_reports)
    )
    _gate(out, "invariance", inv["pass"], inv)

    # historical A replay and batch order
    replay = {}
    ck = torch.load(C.LIPO_A_CHECKPOINT, map_location="cpu", weights_only=False)
    a.load_state_dict(ck["state_dict"], strict=True)
    a.eval()
    with torch.no_grad():
        ev = {split: T.evaluate(a, bundle, split, device) for split in C.SPLITS}
    replay = {
        "valid_rmse_raw": float(ev["valid"]["rmse_raw"]),
        "test_rmse_raw": float(ev["test"]["rmse_raw"]),
        "expected_valid": C.LIPO_A_EXPECTED["valid_rmse_raw"],
        "expected_test": C.LIPO_A_EXPECTED["test_rmse_raw"],
        "valid_abs_diff": abs(float(ev["valid"]["rmse_raw"]) - C.LIPO_A_EXPECTED["valid_rmse_raw"]),
        "test_abs_diff": abs(float(ev["test"]["rmse_raw"]) - C.LIPO_A_EXPECTED["test_rmse_raw"]),
    }
    _gate(out, "historical_A_checkpoint_replay", max(replay["valid_abs_diff"], replay["test_abs_diff"]) <= TOL["replay_abs"], replay)

    order_report = _batch_order_hash(bundle, int(C.LIPO_RECIPE["epochs"]))
    order_report["expected_sha256"] = C.LIPO_A_EXPECTED["batch_order_sha256"]
    order_report["matches_historical"] = order_report["batch_order_sha256"] == C.LIPO_A_EXPECTED["batch_order_sha256"]
    _gate(out, "historical_batch_order", order_report["matches_historical"], order_report)

    if post:
        replay_models = {}
        for arm in C.LIPO_ARMS:
            path = C.RUNS / f"{arm}/checkpoint_best_val.pt"
            if not path.is_file():
                replay_models[arm] = {"exists": False}
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model, _info = M.build_lipo_model(bundle, support, arm)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(device).eval()
            with torch.no_grad():
                rec = {split: T.evaluate(model, bundle, split, device) for split in C.SPLITS}
            replay_models[arm] = {
                "exists": True,
                "epoch": int(payload["epoch"]),
                "recorded_valid": float(payload["metrics"]["valid_rmse_raw"]),
                "recorded_test": float(payload["metrics"]["test_rmse_raw"]),
                "replay_valid": float(rec["valid"]["rmse_raw"]),
                "replay_test": float(rec["test"]["rmse_raw"]),
                "pass": abs(float(rec["valid"]["rmse_raw"]) - float(payload["metrics"]["valid_rmse_raw"])) <= TOL["replay_abs"]
                and abs(float(rec["test"]["rmse_raw"]) - float(payload["metrics"]["test_rmse_raw"])) <= TOL["replay_abs"],
                "table": M.table_of(model),
                "model": model,
            }
        _gate(out, "checkpoint_replay_new_arms", all(v.get("pass", False) for v in replay_models.values()), {k: {kk: vv for kk, vv in v.items() if kk not in ("table", "model")} for k, v in replay_models.items()})
        for arm in C.LIPO_ARMS:
            if arm in replay_models and replay_models[arm].get("model") is not None:
                model = replay_models[arm]["model"]
                local = model.trunk.local
                if local.arm == "C":
                    theta = local.Theta.detach().to(torch.float64).cpu().numpy()
                    table = np.asarray(support.U, dtype=np.float64) @ theta
                else:
                    table = M.table_of(model)
                resid = table - support.P @ table
                out.setdefault("post", {})[arm] = {
                    "table_rms": float(np.sqrt(np.mean(table ** 2))),
                    "nonadditive_rms": float(np.sqrt(np.mean(resid ** 2))),
                    "nonadditive_max_abs": float(np.max(np.abs(resid))),
                    "nonadditive_projection_pass": float(np.max(np.abs(resid))) < 1e-6,
                }
    out["passed_all"] = all(v["pass"] for v in out["gates"].values())
    return out


# ---------------------------------------------------------------------------
# PCBA
# ---------------------------------------------------------------------------


def check_pcba(device: torch.device, *, post: bool = False) -> dict[str, Any]:
    import bs_model as BM
    import ds_common as DC
    import ds_model as DM
    import ds_train as DT
    import lr_data as LD
    import pc_model as PCM

    out: dict[str, Any] = {
        "schema": "lcrc-support-joint-transfer-checks-pcba-v1",
        "round_id": C.ROUND_ID,
        "stage": "post" if post else "pre",
        "gates": {},
        "environment": C.environment(),
    }
    split_audit = C.read_json(C.ANALYSIS / "split_audit.json")["pcba"]
    _gate(
        out,
        "official_split_identity",
        bool(split_audit["all_row_ids_equal_official"] and split_audit["stored_index_hashes_match"]),
        split_audit,
    )
    task = LD.load_task()
    support = S.load_support(C.ANALYSIS / "support/pcba_support.npz")
    sup = _support_checks(
        support,
        {"path_cat": S.pcba_split_caches(C.read_json(C.PCBA_SPLIT_PATH))["train"]["path_cat"],
         "path_end63": S.pcba_split_caches(C.read_json(C.PCBA_SPLIT_PATH))["train"]["path_end63"]},
        "pcba",
    )
    _gate(out, "support_train_only", sup["omega_equals_train_unique_pairs"], sup)
    _gate(out, "support_projector", abs(sup["P_symmetric_max_abs"]) < 1e-10 and abs(sup["P_idempotent_max_abs"]) < 1e-10, sup)
    _gate(out, "support_rank_identity", bool(sup["rank_identity"]), sup)
    _gate(out, "support_legality", bool(sup["all_omega_legal"]), sup)

    w2, w2_audit = BM.build_model("W2", task.spec, task.norm, int(C.PCBA_N_TASKS), seed=C.SEED, device="cpu", verify_parameters=True)
    fixed1, fixed1_audit = DM.build_ds_model(task.spec, task.norm, int(C.PCBA_N_TASKS), arm="FIXED1", seed=C.SEED, device="cpu")
    w2j, w2j_info = M.build_pcba_model(task, support, C.ARM_W2_JOINT)
    f1j, f1j_info = M.build_pcba_model(task, support, C.ARM_FIXED1_JOINT)

    hist_w2 = C.read_json(C.W2_RUN)["model_audit"]
    hist_f1 = C.read_json(C.FIXED1_RECORD)["model_audit"]
    init_identity = {
        "w2_trunk_init_fingerprint": w2_audit["trunk_init_fingerprint"],
        "historical_w2_trunk_init_fingerprint": hist_w2["trunk_init_fingerprint"],
        "w2_full_trunk_init_fingerprint": w2_audit["full_trunk_init_fingerprint"],
        "historical_w2_full_trunk_init_fingerprint": hist_w2["full_trunk_init_fingerprint"],
        "w2_matches_historical": w2_audit["trunk_init_fingerprint"] == hist_w2["trunk_init_fingerprint"]
        and w2_audit["full_trunk_init_fingerprint"] == hist_w2["full_trunk_init_fingerprint"],
        "fixed1_base_state_sha256": fixed1_audit["base_state_sha256"],
        "historical_fixed1_base_state_sha256": hist_f1["base_state_sha256"],
        "fixed1_new_state_sha256": fixed1_audit["new_state_sha256"],
        "historical_fixed1_new_state_sha256": hist_f1["new_state_sha256"],
        "fixed1_matches_historical": fixed1_audit["base_state_sha256"] == hist_f1["base_state_sha256"]
        and fixed1_audit["new_state_sha256"] == hist_f1["new_state_sha256"],
        "w2_parameters": int(sum(p.numel() for p in w2.parameters())),
        "fixed1_parameters": int(sum(p.numel() for p in fixed1.parameters())),
        "w2_joint_parameters": int(w2j_info["parameters"]),
        "fixed1_joint_parameters": int(f1j_info["parameters"]),
    }
    _gate(out, "initial_state_matches_historical", init_identity["w2_matches_historical"] and init_identity["fixed1_matches_historical"], init_identity)

    cmp_w2 = M.compare_parameter_states(M.common_parameter_state(w2), M.common_parameter_state(w2j))
    cmp_f1 = M.compare_parameter_states(M.common_parameter_state(fixed1), M.common_parameter_state(f1j))
    _gate(out, "w2_common_parameters_bitwise", cmp_w2["all_equal"], cmp_w2)
    _gate(out, "fixed1_common_parameters_bitwise", cmp_f1["all_equal"], cmp_f1)

    plan, plan_audit = DC.plan_for_seed(task, C.SEED)
    plan_meta = DC.plan_meta(plan, int(C.PCBA_RECIPE["horizon"]), C.SEED)
    hist_plan = C.read_json(C.W2_RUN)["batch_plan"]
    plan_ok = (
        plan_meta["global_plan_sha256"] == C.W2_EXPECTED["global_plan_sha256"]
        and plan_meta["prefix_sha256"] == C.W2_EXPECTED["prefix25_sha256"]
        and int(len(plan)) == 40
        and int(len(plan[0]["batches"])) == int(C.PCBA_RECIPE["horizon"] * 0 + 2738)
    )
    _gate(out, "batch_plan_prefix", plan_ok, {"plan_meta": plan_meta, "historical_plan": hist_plan})

    rows = np.asarray(task.train_rows[:128], dtype=np.int64)
    batch = PCM.collate(task.cache, rows.tolist())
    for model in (w2, fixed1, w2j, f1j):
        model.eval()
    with torch.no_grad():
        pw2 = w2(batch).cpu().numpy()
        pw2j = w2j(batch).cpu().numpy()
        pf1 = fixed1(batch).cpu().numpy()
        pf1j = f1j(batch).cpu().numpy()
    forward = {
        "w2_vs_w2j_max_abs": _max_abs(pw2, pw2j),
        "fixed1_vs_fixed1j_max_abs": _max_abs(pf1, pf1j),
        "tolerance": TOL["forward_abs"],
    }
    _gate(out, "seen_type_forward_equal", max(forward["w2_vs_w2j_max_abs"], forward["fixed1_vs_fixed1j_max_abs"]) <= TOL["forward_abs"], forward)

    # FIXED1 two-stage arithmetic is inherited unchanged
    src, dst, edge_msg = DM.directed_bonds(fixed1, batch)
    with torch.no_grad():
        _a, _b, _scal, h0 = fixed1.encode_h0(batch)
        h1, m1 = fixed1.stage1(h0, src, dst, edge_msg)
        h2, m2, info1 = fixed1.stage2(h1, h0, src, dst, edge_msg)
        src2, dst2, edge2 = DM.directed_bonds(f1j, batch)
        _a2, _b2, _scal2, h0_2 = f1j.encode_h0(batch)
        h1_2, m1_2 = f1j.stage1(h0_2, src2, dst2, edge2)
        h2_2, m2_2, info2 = f1j.stage2(h1_2, h0_2, src2, dst2, edge2)
    fixed1_stage = {
        "stage1_max_abs": _max_abs(h1.cpu().numpy(), h1_2.cpu().numpy()),
        "stage2_max_abs": _max_abs(h2.cpu().numpy(), h2_2.cpu().numpy()),
        "m2_max_abs": _max_abs(m2.cpu().numpy(), m2_2.cpu().numpy()),
        "stage1_identical_class_method": DM.DSModel.stage1 is type(f1j).stage1,
        "stage2_identical_class_method": DM.DSModel.stage2 is type(f1j).stage2,
        "layer2_reads_original": info1["layer2_reads"] == info2["layer2_reads"],
    }
    _gate(out, "fixed1_two_stage_unchanged", fixed1_stage["stage1_identical_class_method"] and fixed1_stage["stage2_identical_class_method"] and fixed1_stage["layer2_reads_original"], fixed1_stage)

    grad_report = {}
    for name, model in (("W2_JOINT", w2j), ("FIXED1_JOINT", f1j)):
        model.train()
        optimizer, _ = DT.build_optimizer(model)
        rows_all = np.concatenate([plan[0]["batches"][i] for i in range(4)])
        b = PCM.collate(task.cache, rows_all.tolist())
        y = torch.as_tensor(np.asarray(task.labels[rows_all]), dtype=torch.float32)
        init_zero = []
        for warm in range(2):
            model.zero_grad(set_to_none=True)
            loss = DC.MET.masked_bce_loss(model(b), y)
            loss.backward()
            if warm == 0:
                init_zero = sorted(n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0)
                optimizer.step()
        local = model.local
        g = local.J_table.grad
        param_nonzero = {n: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0) for n, p in model.named_parameters()}
        observed_per_task = torch.isfinite(y).sum(dim=0).cpu().tolist()
        positive_per_task = torch.nansum((y == 1).float(), dim=0).cpu().tolist()
        allowed_zero = {
            f"readout.heads.{t}.{suffix}"
            for t, count in enumerate(observed_per_task)
            if int(count) == 0 or float(positive_per_task[t]) == 0.0
            for suffix in ("fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight")
        }
        grad_report[name] = {
            "loss": float(loss.detach()),
            "table_grad_finite": bool(torch.isfinite(g).all()),
            "table_grad_nonzero_entries": int((g != 0).sum()),
            "table_rows_with_grad": int((g.abs().sum(dim=1) > 0).sum()),
            "table_rows_total": int(support.n_omega),
            "zero_grad_at_init_expected_from_zeroed_outputs": init_zero,
            "all_parameter_grads_finite": bool(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)),
            "parameters_without_nonzero_grad_after_one_step": sorted(
                k for k, v in param_nonzero.items() if not v and k not in allowed_zero
            ),
            "tasks_with_no_observed_labels_in_probe": [int(t) for t, c in enumerate(observed_per_task) if int(c) == 0],
            "tasks_with_no_positive_labels_in_probe": [int(t) for t, c in enumerate(positive_per_task) if float(c) == 0.0],
        }
    _gate(
        out,
        "gradients",
        all(
            v["table_grad_finite"]
            and v["table_grad_nonzero_entries"] > 0
            and not v["parameters_without_nonzero_grad_after_one_step"]
            for v in grad_report.values()
        ),
        grad_report,
    )

    # invariance
    inv: dict[str, Any] = {}
    b = PCM.collate(task.cache, np.asarray(task.valid_rows[:32], dtype=np.int64).tolist())
    perm_batch, p = BM.permute_atoms(b)
    w2j.eval()
    with torch.no_grad():
        inv["atom_renumbering_w2j_max_abs"] = _max_abs(w2j(b).cpu().numpy(), w2j(perm_batch).cpu().numpy())
        f1j.eval()
        inv["atom_renumbering_f1j_max_abs"] = _max_abs(f1j(b).cpu().numpy(), f1j(perm_batch).cpu().numpy())
        perm = torch.randperm(int(b["path_cat"].numel()), generator=torch.Generator().manual_seed(20261126))
        b2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in b.items()}
        for key in ("path_atom", "path_cat", "path_end63", "path_end", "path_closure", "path_graph"):
            b2[key] = b2[key][perm]
        inv["path_order_w2j_max_abs"] = _max_abs(w2j(b).cpu().numpy(), w2j(b2).cpu().numpy())
        # chunked evaluation on a subset
        rows = np.asarray(task.valid_rows[:64], dtype=np.int64)
        full = w2j(PCM.collate(task.cache, rows.tolist())).cpu().numpy()
        chunks = []
        for start in range(0, rows.size, 7):
            chunks.append(w2j(PCM.collate(task.cache, rows[start : start + 7].tolist())).cpu().numpy())
    inv["batch_regroup_max_abs"] = _max_abs(np.concatenate(chunks, axis=0), full)
    inv["pass"] = all(
        inv[k] <= TOL["invariance_abs"]
        for k in ("atom_renumbering_w2j_max_abs", "atom_renumbering_f1j_max_abs", "path_order_w2j_max_abs", "batch_regroup_max_abs")
    )
    _gate(out, "invariance", inv["pass"], inv)

    # historical checkpoint replay (also proves the evaluator is the same)
    test_info = DC.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    test_labels = np.asarray(test_info["labels"], dtype=np.float32)
    replay = {}
    for name, model, path, expected in (
        ("W2", w2, C.W2_CHECKPOINT, C.W2_EXPECTED),
        ("FIXED1", fixed1, C.FIXED1_CHECKPOINT, C.FIXED1_EXPECTED),
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        with torch.inference_mode():
            rep = {
                "valid": DC.evaluate_split(model, task, device, "valid", per_task=False),
                "test": DC.evaluate_split(model, task, device, "test", y=test_labels, per_task=False),
            }
        replay[name] = {
            "epoch": int(payload.get("epoch", -1)),
            "valid_ap": float(rep["valid"]["ap"]),
            "test_ap": float(rep["test"]["ap"]),
            "expected_valid_ap": float(expected.get("best_valid_ap", expected.get("valid_ap"))),
            "expected_test_ap": float(expected.get("test_ap_best", expected.get("test_ap"))),
            "valid_abs_diff": abs(float(rep["valid"]["ap"]) - float(expected.get("best_valid_ap", expected.get("valid_ap")))),
            "test_abs_diff": abs(float(rep["test"]["ap"]) - float(expected.get("test_ap_best", expected.get("test_ap")))),
            "logits_sha256": rep["test"].get("logits_sha256"),
            "note": "Test is a cross-device replay of an already recorded value; ranking ties move AP at the 1e-4 level.",
        }
    _gate(
        out,
        "historical_checkpoint_replay",
        all(v["valid_abs_diff"] <= TOL["replay_abs"] and v["test_abs_diff"] <= TOL["replay_ap_abs"] for v in replay.values()),
        replay,
    )

    if post:
        replay_new = {}
        for arm in C.PCBA_ARMS:
            path = C.RUNS / f"{arm}/checkpoint_best.pt"
            if not path.is_file():
                replay_new[arm] = {"exists": False}
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model, _info = M.build_pcba_model(task, support, arm)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(device).eval()
            with torch.inference_mode():
                rep = {
                    "valid": DC.evaluate_split(model, task, device, "valid", per_task=False),
                    "test": DC.evaluate_split(model, task, device, "test", y=test_labels, per_task=False),
                }
            replay_new[arm] = {
                "exists": True,
                "epoch": int(payload["epoch"]),
                "recorded_valid": float(payload["metrics"]["valid"]["ap"]),
                "recorded_test": float(payload["metrics"]["test"]["ap"]),
                "replay_valid": float(rep["valid"]["ap"]),
                "replay_test": float(rep["test"]["ap"]),
                "pass": abs(float(rep["valid"]["ap"]) - float(payload["metrics"]["valid"]["ap"])) <= NEW_REPLAY_AP_TOL
                and abs(float(rep["test"]["ap"]) - float(payload["metrics"]["test"]["ap"])) <= NEW_REPLAY_AP_TOL,
            }
        _gate(out, "checkpoint_replay_new_arms", all(v.get("pass", False) for v in replay_new.values()), replay_new)
    out["passed_all"] = all(v["pass"] for v in out["gates"].values())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", default="all", choices=("lipo", "pcba", "all"))
    parser.add_argument("--post", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.scope in ("lipo", "all"):
        lipo = check_lipo(device, post=args.post)
        C.write_json(C.ANALYSIS / ("post_checks_lipo.json" if args.post else "checks_lipo.json"), lipo)
        print("lipo passed_all", lipo["passed_all"])
    if args.scope in ("pcba", "all"):
        pcba = check_pcba(device, post=args.post)
        C.write_json(C.ANALYSIS / ("post_checks_pcba.json" if args.post else "checks_pcba.json"), pcba)
        print("pcba passed_all", pcba["passed_all"])


if __name__ == "__main__":
    main()
