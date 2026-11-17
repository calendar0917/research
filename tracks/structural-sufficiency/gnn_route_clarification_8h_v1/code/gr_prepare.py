"""Preflight: the only checks that can change a research conclusion.

Runs on CPU, writes ``analysis/prepare.json`` and ``analysis/protocol_freeze.json``.
Every check raises instead of quietly degrading.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

import gr_common as C
import gr_data as D
import gr_models as M


def check_data(dataset: str) -> dict[str, Any]:
    b = D.bundle(dataset)
    mods = D.modules()
    files = mods["cr_data"].verify_frozen_files(dataset)
    sizes = {split: int(b.global_row_ids[split].size) for split in C.SPLITS}
    if sizes != C.EXPECTED_SPLIT_SIZES[dataset]:
        raise RuntimeError(f"{dataset}: split sizes {sizes}")
    union = np.concatenate([b.global_row_ids[split] for split in C.SPLITS])
    if union.size != C.EXPECTED_TOTAL_ROWS[dataset] or np.unique(union).size != union.size:
        raise RuntimeError(f"{dataset}: splits are not a disjoint cover")
    if not np.array_equal(np.sort(union), np.arange(C.EXPECTED_TOTAL_ROWS[dataset])):
        raise RuntimeError(f"{dataset}: split union is not 0..N-1")
    g = D.gine_cache(dataset)
    order_audit = {split: g.audit_against_lcrc(b, split) for split in C.SPLITS}
    if not all(v["atom_order_shared"] for v in order_audit.values()):
        raise RuntimeError(f"{dataset}: LCRC and OGB atom order differ")
    train = np.asarray(b.caches["train"]["global"], dtype=np.float64)
    span = train.max(0) - train.min(0)
    active = np.flatnonzero(span > C.ACTIVE_SPAN_EPS)
    if not np.array_equal(active, np.asarray(b.active_global["active_indices"])):
        raise RuntimeError(f"{dataset}: active-global mask mismatch")
    # target scaler must be Train-only, ddof=0
    tr = b.payload["labels"][b.global_row_ids["train"]]
    if abs(float(tr.mean()) - b.scaler["mean"]) > 0 or abs(float(tr.std(ddof=0)) - b.scaler["std"]) > 0:
        raise RuntimeError(f"{dataset}: target scaler is not the Train-only ddof=0 statistic")
    return {
        "dataset": dataset,
        "split_sizes": sizes,
        "split_disjoint_cover": True,
        "frozen_files": files,
        "atom_order": order_audit,
        "n_active_global": int(active.size),
        "active_indices": active.tolist(),
        "inactive_indices": b.active_global["inactive_indices"],
        "target_scaler": b.scaler,
        "feature_spec": {"name": b.extra["spec_name"], "hash": b.extra["spec_hash"], "global_dim": int(b.spec.global_dim)},
        "norm_fingerprint": b.extra["norm_fingerprint"],
        "passed": True,
    }


def check_models(dataset: str) -> dict[str, Any]:
    import torch
    from torch_geometric.data import Batch, Data

    b = D.bundle(dataset)
    ms = M.ModelSet(b)
    g = D.gine_cache(dataset)
    out: dict[str, Any] = {"dataset": dataset}

    # 1. G0 reproduces the repository's original GINE forward exactly.
    sys.path.insert(0, str(C.SQL_CODE))
    import sq_gine

    torch.manual_seed(0)
    sq = sq_gine.GINEModel.build(g.category_counts())
    mine = ms.dense("G0")
    state = {("head.net." + k[5:] if k.startswith("head.") else "trunk." + k): v for k, v in sq.state_dict().items()}
    mine.load_state_dict(state, strict=True)
    mine.eval()
    sq.eval()
    rows = list(range(min(16, b.caches["train"]["n_molecules"])))
    payload = b.payload
    data = []
    for r in rows:
        gid = int(b.global_row_ids["train"][r])
        a0, a1 = int(payload["node_ptr"][gid]), int(payload["node_ptr"][gid + 1])
        b0, b1 = int(payload["bond_ptr"][gid]), int(payload["bond_ptr"][gid + 1])
        bu = payload["bond_u"][b0:b1].astype(np.int64)
        bv = payload["bond_v"][b0:b1].astype(np.int64)
        ei = torch.as_tensor(np.concatenate([np.stack([bu, bv]), np.stack([bv, bu])], axis=1), dtype=torch.long)
        ec = torch.as_tensor(np.concatenate([payload["bond_feat"][b0:b1], payload["bond_feat"][b0:b1]], axis=0), dtype=torch.long)
        data.append(
            Data(
                num_nodes=int(a1 - a0),
                x_cat=torch.as_tensor(payload["atom_feat"][a0:a1], dtype=torch.long),
                deg=torch.as_tensor(payload["degree"][a0:a1], dtype=torch.long),
                ring=torch.as_tensor(payload["min_ring"][a0:a1], dtype=torch.long),
                edge_index=ei,
                edge_cat=ec,
                n_atoms=torch.as_tensor([int(a1 - a0)]),
            )
        )
    batch = D.collate(b, "train", rows)
    with torch.no_grad():
        diff = float((sq(Batch.from_data_list(data)) - mine(batch)).abs().max())
    if diff > 0:
        raise RuntimeError(f"G0 does not reproduce sq_gine (max diff {diff})")
    out["g0_reproduces_sq_gine_max_abs_diff"] = diff

    # 2. physical bonds are two directed edges, never four
    cache = b.caches["train"]
    bond_ptr = np.asarray(cache["bond_ptr"])
    n_bonds = int(bond_ptr[-1])
    n_edges = int(batch["edge_index"].shape[1])
    expected = int(2 * np.asarray(cache["bond_counts"] if "bond_counts" in cache else np.concatenate([[0], np.cumsum(np.diff(bond_ptr))])).sum())
    # recompute directly for the sampled rows
    rows_arr = np.asarray(rows)
    exp_rows = int(2 * np.diff(bond_ptr)[rows_arr].sum())
    if n_edges != exp_rows:
        raise RuntimeError(f"edge direction duplication is wrong: {n_edges} != {exp_rows}")
    out["directed_edge_count"] = {"edges": n_edges, "physical_bonds": exp_rows // 2, "factor": 2}

    # 3. parameter counts and analytic additive budget
    dense = {"G0": ms.dense("G0"), "Gx": ms.dense("Gx")}
    counts = {k: int(sum(p.numel() for p in v.parameters())) for k, v in dense.items()}
    out["parameter_counts"] = counts
    out["readout_parameter_counts"] = {
        "G0": int(sum(p.numel() for p in dense["G0"].head.parameters())),
        "Gx": int(sum(p.numel() for p in dense["Gx"].head.parameters())),
    }
    if out["readout_parameter_counts"]["G0"] != C.G0_READOUT_PARAMS:
        raise RuntimeError("G0 readout parameter count mismatch")
    if out["readout_parameter_counts"]["Gx"] != C.GX_READOUT_PARAMS:
        raise RuntimeError("Gx readout parameter count mismatch")
    if counts["G0"] != 377094 and dataset == "ogbg-molesol":
        raise RuntimeError(f"G0 parameter count {counts['G0']} != 377094 for ESOL")
    if counts["G0"] != 377350 and dataset == "ogbg-mollipo":
        raise RuntimeError(f"G0 parameter count {counts['G0']} != 377350 for Lipo")
    if counts["Gx"] != 379014:
        raise RuntimeError(f"Gx parameter count {counts['Gx']} != 379014")
    if int(sum(p.numel() for p in ms.lcrc().parameters())) != C.LCRC_C1_PARAMS:
        raise RuntimeError("LCRC C1 parameter count mismatch")
    budgets = {}
    for kind in ("G0", "Gx"):
        active = [] if kind == "G0" else b.active_global["active_indices"]
        n_uni = 1 if kind == "G0" else len(active)
        head = ms._new_head(kind, n_active=n_uni, active_indices=active, size_scaler=D.size_scaler(b))
        params = int(sum(p.numel() for p in head.parameters()))
        formula = M.additive_params(head.hidden, n_uni)
        if params != formula:
            raise RuntimeError(f"additive parameter formula mismatch for {kind}: {params} != {formula}")
        budgets[kind] = {
            **head.budget,
            "actual_params": params,
            "delta_to_dense": params - out["readout_parameter_counts"][kind],
            "trunk_plus_head_total": params + int(sum(p.numel() for p in dense[kind].trunk.parameters())),
            "dense_total": counts[kind],
        }
    out["additive_budget"] = budgets

    # 4. canonical init identity across recipes + Gx inherits the G0 conv/BN trunk
    hashes = {
        "c1": C.sha256_state(ms.c1_state),
        "g0": C.sha256_state(ms.g0_state),
        "gx": C.sha256_state(ms.gx_state),
    }
    gx = ms.dense("Gx").state_dict()
    g0 = ms.dense("G0").state_dict()
    shared_keys = [k for k in gx if k.startswith("trunk.convs.") or k.startswith("trunk.norms.")]
    inherited = all(torch.equal(gx[k], g0[k]) for k in shared_keys)
    if not inherited:
        raise RuntimeError("Gx does not inherit the G0 conv/BN canonical trunk")
    out["init_hashes"] = hashes
    out["gx_inherits_g0_convs_and_bn"] = inherited
    out["scale"] = {
        "mean_h": None,
        "note": "additive readout initialised from isolated seed 0; trunk from the canonical state",
    }

    # 5. L wrapper equals the frozen original C1 module
    from dissect_model import DissectLCRC

    C.set_global_seeds(C.INIT_SEED_CANONICAL)
    original = DissectLCRC(b.spec, b.norm, "C1")
    original.load_state_dict(ms.c1_state, strict=True)
    original.eval()
    wrapped = ms.lcrc().eval()
    with torch.no_grad():
        p_orig, _aux = original(batch, return_aux=True)
        p_wrap = wrapped(batch)
    diff_c1 = float((p_orig - p_wrap).abs().max())
    if diff_c1 > 1e-6:
        raise RuntimeError(f"C1 wrapper differs from the frozen module ({diff_c1})")
    out["c1_wrapper_max_abs_diff"] = diff_c1

    # 6. no bypass: the additive head is exactly the five allowed tensors
    model = ms.additive("G0", active_indices=[], size_scaler=D.size_scaler(b))
    allowed_prefixes = ("head.", "trunk.")
    bad = [name for name, _ in model.named_parameters() if not name.startswith(allowed_prefixes)]
    if bad:
        raise RuntimeError(f"unexpected additive-model parameters: {bad}")
    head_names = sorted(name for name, _ in model.head.named_parameters())
    out["additive_head_parameters"] = head_names

    # 7. gradient really flows into the trunk during a training step
    model.train()
    batch = D.collate(b, "train", list(range(4)))
    batch["y"] = torch.zeros(4)
    loss = torch.nn.functional.mse_loss(model(batch), batch["y"])
    loss.backward()
    grads = {name: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for name, p in model.named_parameters()}
    trunk_grads = {k: v for k, v in grads.items() if k.startswith("trunk.convs.")}
    if not any(v > 0 for v in trunk_grads.values()):
        raise RuntimeError("no gradient reached the additive model's GINE trunk")
    out["gradient_flow"] = {
        "n_params_with_grad": int(sum(1 for v in grads.values() if v > 0)),
        "n_params": len(grads),
        "trunk_conv_grad_positive": int(sum(1 for v in trunk_grads.values() if v > 0)),
    }
    return out


def freeze_protocol(prepare: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "design_basis_commit": C.DESIGN_BASIS_COMMIT,
        "mainline_basis_commit": C.MAINLINE_BASIS_COMMIT,
        "authorised": {
            "datasets": list(C.DATASETS),
            "architectures_stage1": list(C.ARCHS),
            "recipes": {k: dict(v) for k, v in C.RECIPES.items()},
            "stage2_family": "A: dropout(SiLU(W h + b)) -> bias-free s_v, m_v; y = b + sum s_v + mean m_v + sum_j u_j(z_j)",
            "stage3": "same head on frozen final representation; teacher = T",
            "oracle": {"properties": list(C.ORACLE_PROPERTIES), "models": list(C.ORACLE_MODELS)},
            "epochs": C.EPOCHS,
            "batch_size": C.BATCH_SIZE,
            "eval_batch_size": C.EVAL_BATCH_SIZE,
            "dropout": C.DROPOUT,
            "seed": C.TRAIN_SEED,
            "selection": "minimum Val RMSE (stage 3: minimum Val teacher-fidelity RMSE); ties keep the earlier epoch / R3",
            "test_policy": "always; Test is observed every epoch and never used for selection",
            "gates": {
                "A_VAL_GATE": C.A_GATE_RATIO,
                "A_POINT_RETENTION": C.A_POINT_RATIO,
                "D_FIDELITY_SIGMA": C.D_FIDELITY_SIGMA,
                "D_REAL_RATIO": C.D_REAL_RATIO,
                "BRIDGE_R_GEO": C.BRIDGE_R_GEO,
                "BRIDGE_R_MAX": C.BRIDGE_R_MAX,
                "ORACLE_ACCURACY_SIGMA": C.ORACLE_ACCURACY_SIGMA,
                "ORACLE_COMPARABILITY_SIGMA": C.ORACLE_COMPARABILITY_SIGMA,
                "EXPLANATION_IMPROVEMENT": C.EXPLANATION_IMPROVEMENT,
                "IG_TOL_REL": C.IG_TOL_REL,
                "IG_TOL_ABS": C.IG_TOL_ABS,
                "LOW_SUPPORT_COMPONENTS": C.LOW_SUPPORT_COMPONENTS,
                "MAX_EXPLAIN_NONTRAIN": C.MAX_EXPLAIN_NONTRAIN,
                "MAX_EXPLAIN_TRAIN": C.MAX_EXPLAIN_TRAIN,
            },
            "budget_soft_gpu_hours": 14.0,
            "budget_hard_gpu_hours": 16.0,
        },
        "code_sha256": C.code_hashes(),
        "prepare": prepare,
    }
    payload["freeze_hash"] = C.sha256_json(payload)
    return payload


def main() -> int:
    t0 = time.time()
    prepare: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    prepare["data"] = {ds: check_data(ds) for ds in C.DATASETS}
    prepare["models"] = {ds: check_models(ds) for ds in C.DATASETS}
    prepare["environment"] = C.environment()
    prepare["seconds"] = time.time() - t0
    prepare["passed"] = True
    C.write_json(C.ANALYSIS_DIR / "prepare.json", prepare)
    freeze = freeze_protocol({k: v for k, v in prepare.items() if k not in ("data", "models")})
    C.write_json(C.ANALYSIS_DIR / "protocol_freeze.json", freeze)
    print(json.dumps({k: v for k, v in prepare.items() if k in ("passed", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
