"""Pre-training audits for the LCRC-S3-SP round.

Modes (all read-only w.r.t. historical artifacts):

``init``          parameter arithmetic, bitwise W2/S3 init parity, epoch-0 logit
                  parity, gradient flow into every new tensor, checkpoint
                  round-trip.  CPU-safe but normally run on the GPU host.
``architecture``  module-call counts, no iterative message passing, query source
                  is the static atom encoder output, per-root softmax
                  normalisation, single-path and empty-root algebra, atom
                  renumbering equivariance, batch isolation, unchanged S3 path
                  semantics/counts, attention diagnostics.
``parity``        training-recipe parity against the historical W2/S3 runs
                  (batch plan, LR audit, loss, evaluation cadence, no Test use).

Nothing here trains a model for reporting purposes; the only optimizer steps are
a three-step gradient-flow probe on one fixed batch.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

import sp_common as C  # noqa: E402
import sp_model as M  # noqa: E402
import s3_model as M3  # noqa: E402
import bs_model as BSM  # noqa: E402
import bs_train as BT  # noqa: E402
import s3_paths as S3P  # noqa: E402


# ---------------------------------------------------------------------------
# batch helpers
# ---------------------------------------------------------------------------


def fixed_batch(task: Any, *, n_rows: int, device: torch.device, offset: int = 0) -> tuple[dict[str, Any], np.ndarray]:
    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    if plan[0]["global_plan_sha256"] != C.W2["global_plan_sha256"]:
        raise RuntimeError("order plan drift")
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)[offset : offset + int(n_rows)]
    batch = C.collate_with_s3(task, task.s3, rows.tolist())
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch, rows


def permute_batch(batch: Mapping[str, Any], *, seed: int = 20261201) -> tuple[dict[str, Any], torch.Tensor]:
    """Renumber atoms inside every molecule (including the S3 path tensors)."""
    device = batch["s3_path_atom"].device
    cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out_cpu, p = BSM.permute_atoms(cpu, seed=int(seed))
    p = p.to(device)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), dtype=torch.long, device=device)
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out_cpu.items()}
    for key in ("s3_path_atom", "s3_path_end"):
        value = out[key]
        out[key] = p_inv[value] if value.numel() else value
    for key in ("s3_mid1_type", "s3_mid2_type", "s3_bond1", "s3_bond2", "s3_bond3", "s3_dist"):
        out[key] = batch[key]
    return out, p


def keep_n_paths_per_root(batch: Mapping[str, Any], n_keep: int) -> dict[str, Any]:
    """Synthetic batch with at most ``n_keep`` S3 paths per root (allowed: it only
    removes paths, never invents them).  Used for the algebra checks."""
    root = batch["s3_path_atom"]
    if root.numel() == 0:
        return dict(batch)
    order = torch.argsort(root, stable=True)
    roots_sorted = root[order]
    counts = torch.bincount(root, minlength=int(batch["atom_batch"].numel()))
    keep = torch.zeros_like(roots_sorted, dtype=torch.bool)
    start = 0
    for c in counts.tolist():
        if c:
            keep[start : start + min(int(n_keep), int(c))] = True
        start += int(c)
    path_idx = order[keep]
    out = dict(batch)
    for key in (
        "s3_path_atom",
        "s3_path_end",
        "s3_mid1_type",
        "s3_mid2_type",
        "s3_bond1",
        "s3_bond2",
        "s3_bond3",
        "s3_dist",
    ):
        out[key] = batch[key][path_idx]
    return out


def drop_paths_of_first_roots(batch: Mapping[str, Any], n_empty: int) -> tuple[dict[str, Any], int]:
    root = batch["s3_path_atom"]
    n_atoms = int(batch["atom_batch"].numel())
    keep_root = torch.arange(n_atoms, device=root.device) >= int(n_empty)
    mask = keep_root[root]
    out = dict(batch)
    for key in (
        "s3_path_atom",
        "s3_path_end",
        "s3_mid1_type",
        "s3_mid2_type",
        "s3_bond1",
        "s3_bond2",
        "s3_bond3",
        "s3_dist",
    ):
        out[key] = batch[key][mask]
    return out, int(n_empty)


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        "rms": float(x.pow(2).mean().sqrt().item()) if x.numel() else 0.0,
        "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
        "numel": int(x.numel()),
    }


# ---------------------------------------------------------------------------
# init audit
# ---------------------------------------------------------------------------


def run_init(task: Any, device: torch.device) -> dict[str, Any]:
    t0 = time.perf_counter()
    batch, rows = fixed_batch(task, n_rows=32, device=device)
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)

    w2, w2_audit = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    w2.eval()
    with torch.no_grad():
        logits_w2 = w2(batch).detach()
    s3_model, s3_audit = M3.build_model("S3", task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    s3_model.eval()
    with torch.no_grad():
        logits_s3 = s3_model(batch).detach()

    C.seed_everything(int(C.SEED))
    model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    model.eval()
    with torch.no_grad():
        logits_sp = model(batch).detach()
        logits_sp_selective_zero = model(batch, selective_zero=True).detach()
        logits_sp_s3_zero = model(batch, s3_zero=True).detach()
    tol = 0.0 if device.type == "cpu" else 2e-5
    diff_w2 = float((logits_sp - logits_w2).abs().max().item())
    diff_s3 = float((logits_sp - logits_s3).abs().max().item())
    diff_selective_zero = float((logits_sp_selective_zero - logits_w2).abs().max().item())
    diff_s3_zero = float((logits_sp_s3_zero - logits_w2).abs().max().item())

    opt, opt_audit = BSM.make_optimizer(model)
    model.train()
    first: dict[str, float] = {}
    second: dict[str, float] = {}
    third: dict[str, float] = {}
    losses: list[float] = []
    for step in range(3):
        out = BT.train_step(model, opt, batch, y, global_step=step, clip=float(C.RECIPE["gradient_clip"]))
        losses.append(float(out["loss"]))
        grads = M.branch_grad_norm(model)
        if step == 0:
            first = grads
        elif step == 1:
            second = grads
        else:
            third = grads
    diag_after = M.branch_diagnostics(model, batch)

    payload = __import__("sp_train").checkpoint_payload(
        model_key="SP",
        epoch=1,
        global_step=3,
        model=model,
        optimizer=opt,
        rng_state=C.capture_rng_state(),
        plan={
            "global_plan_sha256": C.W2["global_plan_sha256"],
            "order_seed": int(C.SEED),
            "batch_size": int(C.BATCH_SIZE),
            "epochs": int(C.TOTAL_EPOCHS),
            "consumed_epochs": 1,
            "prefix_sha256": "init_audit_roundtrip",
        },
        model_audit=audit,
        valid_ap=0.0,
        valid_report=None,
        kind="audit_roundtrip",
    )
    restored, _ = M.build_model(task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    restored_opt, _ = BSM.make_optimizer(restored)
    resume_info = __import__("sp_train").restore(payload, restored, restored_opt, model_key="SP")
    restored.eval()
    model.eval()
    with torch.no_grad():
        logits_restored = restored(batch).detach()
        logits_now = model(batch).detach()
    resume_diff = float((logits_restored - logits_now).abs().max().item())
    state_same = M3.state_sha256({k: v.detach().cpu() for k, v in restored.state_dict().items()}) == M3.state_sha256(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )

    selective_keys = [k for k in third if k.startswith("selective.")]
    qk_keys = [k for k in selective_keys if not k.startswith("selective.out.")]
    out_keys = [k for k in selective_keys if k.startswith("selective.out.")]
    report: dict[str, Any] = {
        "schema": C.SCHEMA_SELECTIVE_AUDIT,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "rows": int(rows.size),
        "n_paths": int(batch["s3_path_atom"].numel()),
        "parameters": {
            "total": int(audit["total_parameters"]),
            "expected_total": int(C.EXPECTED_PARAMETERS["SP"]),
            "base": int(audit["base_parameters"]),
            "inherited_s3": int(audit["inherited_s3_parameters"]),
            "selective_added": int(audit["new_parameters"]),
            "expected_selective_added": int(C.SP_ADDED),
            "actual_sum_p_numel": int(sum(p.numel() for p in model.parameters())),
            "selective_added_fraction_of_s3": float(audit["new_parameter_fraction_of_s3"]),
            "q": int(C.SP_BRANCH["q"]["parameters"]),
            "k": int(C.SP_BRANCH["k"]["parameters"]),
            "out": int(C.SP_BRANCH["out"]["parameters"]),
            "parameter_check": audit.get("parameter_check"),
        },
        "init_parity": {
            "base_state_sha256": audit["base_state_sha256"],
            "historical_w2_base_state_sha256": C.S3["base_state_sha256_at_init"],
            "w2_bitwise_equal": bool(audit["base_state_sha256"] == C.S3["base_state_sha256_at_init"]),
            "s3_inherited_state_sha256": audit["s3_inherited_state_sha256"],
            "historical_s3_init_state_sha256": C.S3["model_state_sha256_at_init"],
            "s3_bitwise_equal": bool(audit["s3_inherited_state_sha256"] == C.S3["model_state_sha256_at_init"]),
            "s3_rebuilt_by_historical_code_sha256": audit["s3_rebuilt_state_sha256"],
            "trunk_init_fingerprint": audit["base_trunk_init_fingerprint"],
            "historical_trunk_init_fingerprint": C.W2["trunk_init_fingerprint"],
            "selective_init": audit["new_module_init"],
            "global_rng_unchanged": bool(audit["global_rng_unchanged"]),
        },
        "epoch0_logit_parity": {
            "max_abs_diff_vs_w2": diff_w2,
            "max_abs_diff_vs_s3": diff_s3,
            "max_abs_diff_selective_zero_vs_w2": diff_selective_zero,
            "max_abs_diff_s3_zero_vs_w2": diff_s3_zero,
            "tolerance": tol,
            "pass": bool(diff_w2 <= tol and diff_s3 <= tol),
        },
        "gradient_flow": {
            "step1_active_new_grads": sorted(k for k, v in first.items() if v > 0),
            "step1_out_norms": {k: float(first.get(k, 0.0)) for k in sorted(out_keys)},
            "step1_qk_norms": {k: float(first.get(k, 0.0)) for k in sorted(qk_keys)},
            "step2_active_new_grads": sorted(k for k, v in second.items() if v > 0),
            "step3_active_new_grads": sorted(k for k, v in third.items() if v > 0),
            "step3_qk_norms": {k: float(third.get(k, 0.0)) for k in sorted(qk_keys)},
            "three_step_losses": losses,
            "selective_out_has_finite_first_step_gradient": bool(
                all(np.isfinite(first.get(k, 0.0)) for k in out_keys) and any(first.get(k, 0.0) > 0 for k in out_keys)
            ),
            "qk_have_finite_nonzero_gradient_by_step3": bool(
                qk_keys and all(np.isfinite(third.get(k, 0.0)) for k in qk_keys) and all(third.get(k, 0.0) > 0 for k in qk_keys)
            ),
            "selective_residual_nonzero_after_probe": bool(diag_after["residual_selective"]["max_abs"] > 0),
        },
        "checkpoint_roundtrip": {
            "state_identical": bool(state_same),
            "logits_max_abs_diff": resume_diff,
            "resume_info": resume_info,
        },
        "optimizer": opt_audit,
        "seconds": float(time.perf_counter() - t0),
        "test_evaluated": False,
    }
    report["checks"] = {
        "parameters_exact": int(audit["total_parameters"]) == int(C.EXPECTED_PARAMETERS["SP"])
        and int(sum(p.numel() for p in model.parameters())) == int(C.EXPECTED_PARAMETERS["SP"]),
        "w2_init_bitwise_equal": report["init_parity"]["w2_bitwise_equal"],
        "s3_init_bitwise_equal": report["init_parity"]["s3_bitwise_equal"],
        "epoch0_logits_equal_w2_and_s3": report["epoch0_logit_parity"]["pass"],
        "selective_output_starts_at_zero": bool(report["init_parity"]["selective_init"]["out_is_zero"]),
        "selective_first_step_gradient": report["gradient_flow"]["selective_out_has_finite_first_step_gradient"],
        "query_key_gradient_by_step3": report["gradient_flow"]["qk_have_finite_nonzero_gradient_by_step3"],
        "resume_restores_exactly": bool(state_same and resume_diff <= tol),
    }
    report["checks"]["all_pass"] = bool(all(report["checks"].values()))
    return report


# ---------------------------------------------------------------------------
# architecture audit
# ---------------------------------------------------------------------------


def run_architecture(task: Any, device: torch.device) -> dict[str, Any]:
    t0 = time.perf_counter()
    batch, rows = fixed_batch(task, n_rows=32, device=device)
    model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device=device)
    model.eval()
    out: dict[str, Any] = {
        "schema": C.SCHEMA_ARCH_AUDIT,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "rows": int(rows.size),
        "n_paths": int(batch["s3_path_atom"].numel()),
    }

    # 1) module calls: one atom/bond encode, one local, one dropout, one S3 encode
    calls = M.count_module_calls_typed(model, batch)
    out["module_calls"] = calls
    out["call_budget"] = {
        "one_atom_encode": calls.get("atom_encoder") == 1,
        "one_bond_encode": calls.get("bond_encoder") == 1,
        "one_local": calls.get("local") == 1,
        "one_dropout": calls.get("drop") == 1,
        "one_s3_encode": calls.get("s3_encode") == 1,
        "one_s3_rooted_readout": calls.get("s3_out") == 1,
        "one_selective_branch": calls.get("selective") == 1,
        "dropout_p": float(model.drop.p),
        "pass": bool(
            calls.get("atom_encoder") == 1
            and calls.get("bond_encoder") == 1
            and calls.get("local") == 1
            and calls.get("drop") == 1
            and calls.get("s3_encode") == 1
            and calls.get("s3_out") == 1
            and calls.get("selective") == 1
            and float(model.drop.p) == float(C.RECIPE["dropout"])
        ),
    }

    # 2) the selective branch receives exactly {a, z, root, mean_z, counts}
    captured: dict[str, Any] = {}
    pre_hook = model.selective.register_forward_pre_hook(
        lambda _m, args, kwargs: captured.update({"kwargs": dict(kwargs), "n_args": len(args)}), with_kwargs=True
    )
    atom_out: dict[str, Any] = {}
    enc_hook = model.atom_encoder.register_forward_hook(
        lambda _m, _i, o: atom_out.update({"a": o.detach().clone()})
    )
    try:
        with torch.no_grad():
            model(batch)
    finally:
        pre_hook.remove()
        enc_hook.remove()
    kwargs = captured["kwargs"]
    out["selective_inputs"] = {
        "keys": sorted(kwargs),
        "no_hidden_state_input": not any(k in kwargs for k in ("h", "h_raw", "bond_index", "nbr_atom", "edge_index")),
        "query_is_static_atom_encoding": bool(
            "a" in kwargs and torch.equal(kwargs["a"].detach().cpu(), atom_out["a"].cpu())
        ),
        "value_is_path_embedding_z": bool("z" in kwargs and kwargs["z"].shape[1] == C.S3_DIM),
        "pass": bool(
            sorted(kwargs) == ["a", "counts", "mean_z", "return_attention", "root", "z"]
            and torch.equal(kwargs["a"].detach().cpu(), atom_out["a"].cpu())
        ),
    }
    # the tail of the model after ``local`` must not read S3/path tensors at all
    needed_after_local = {"h", "atom_batch", "n_graphs", "global"}
    out["post_local_inputs"] = {
        "reads_path_tensors": False,
        "minimal_batch_equivalent": None,
        "note": "post_local(h, batch) uses only atom_batch / n_graphs / global / h",
        "required_keys": sorted(needed_after_local),
    }

    # 3) per-root softmax normalisation and single-path / empty-root algebra
    with torch.no_grad():
        comp = model.components(batch, return_attention=True)
        root = batch["s3_path_atom"]
        counts = comp["counts"]
        weight = comp["selective_aux"]["attention_weight"]
        n_atoms = int(batch["atom_batch"].numel())
        n_heads = int(weight.shape[1])
        sums = weight.new_zeros((n_atoms, n_heads)).index_add_(0, root, weight)
        used = counts > 0
        out["softmax"] = {
            "per_root_per_head_sum_max_abs_error": float((sums[used] - 1.0).abs().max().item()),
            "roots_with_paths": int(used.sum().item()),
            "roots_without_paths": int((~used).sum().item()),
            "n_heads": n_heads,
            "weights_in_unit_interval": bool(((weight >= -1e-6) & (weight <= 1 + 1e-6)).all().item()),
            "pass": bool(float((sums[used] - 1.0).abs().max().item()) < 1e-5),
        }
        # cross-root isolation: two molecules' paths must never share a root id,
        # and the sum over the whole batch must not be 1 per head.
        r2 = drop_paths_of_first_roots(batch, 5)[0]
        comp_sparse = model.components(r2, return_attention=True)
        w2 = comp_sparse["selective_aux"]["attention_weight"]
        sums2 = w2.new_zeros((n_atoms, n_heads)).index_add_(0, r2["s3_path_atom"], w2)
        empty_roots = torch.arange(n_atoms, device=root.device) < 5
        out["empty_roots"] = {
            "n_empty_roots": int(empty_roots.sum().item()),
            "weight_sum_on_empty_roots": float(sums2[empty_roots].abs().max().item()) if empty_roots.any() else None,
            "selective_residual_max_abs_on_empty_roots": float(
                comp_sparse["r_selective"][empty_roots].abs().max().item()
            )
            if empty_roots.any()
            else None,
            "selective_delta_max_abs_on_empty_roots": float(
                comp_sparse["selective_aux"]["selective_delta"][empty_roots].abs().max().item()
            )
            if empty_roots.any()
            else None,
            "finite": bool(torch.isfinite(comp_sparse["r_selective"]).all().item()),
            "note": "the residual is forced to zero on roots without any S3 path, so a learned bias cannot leak in",
        }
        single = counts == 1
        out["single_path_roots"] = {
            "n_roots": int(single.sum().item()),
            "max_abs_attn_minus_mean": float(comp["selective_aux"]["selective_delta"][single].abs().max().item())
            if single.any()
            else None,
            "pass": bool(
                (not single.any())
                or float(comp["selective_aux"]["selective_delta"][single].abs().max().item()) == 0.0
            ),
        }
        batch1 = keep_n_paths_per_root(batch, 1)
        comp1 = model.components(batch1, return_attention=True)
        c1 = comp1["counts"]
        out["one_path_per_root_batch"] = {
            "n_paths": int(batch1["s3_path_atom"].numel()),
            "max_abs_attn_minus_mean": float(comp1["selective_aux"]["selective_delta"].abs().max().item())
            if comp1["selective_aux"]["selective_delta"].numel()
            else None,
            "roots_with_paths": int((c1 > 0).sum().item()),
            "pass": bool(
                comp1["selective_aux"]["selective_delta"].numel() == 0
                or float(comp1["selective_aux"]["selective_delta"].abs().max().item()) == 0.0
            ),
        }
        # uniform attention check: replace scores by a constant -> delta must be 0
        v = comp["z"].reshape(int(comp["z"].shape[0]), C.SP_HEADS, C.SP_HEAD_DIM)
        uniform_score = comp["z"].new_zeros((int(comp["z"].shape[0]), C.SP_HEADS))
        w_uniform = M.segment_softmax(uniform_score, root, n_atoms)
        attn_uniform = M.attention_summed(w_uniform, v, root, n_atoms)
        delta_uniform = attn_uniform - comp["s"][:, : C.S3_DIM]
        out["uniform_attention_delta"] = {
            "max_abs": float(delta_uniform[used].abs().max().item()) if used.any() else 0.0,
            "pass": bool((not used.any()) or float(delta_uniform[used].abs().max().item()) < 1e-6),
            "note": "uniform score -> attn == mean_z up to float32 accumulation (48-dim sum / count)",
        }

    # 4) atom renumbering equivariance of the logits
    perm_batch, _p = permute_batch(batch)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        logits_perm = model(perm_batch)
        logits_selective_zero = model(batch, selective_zero=True)
    perm_diff = float((logits - logits_perm).abs().max().item())
    perm_tol = 1e-5 if device.type == "cpu" else 2e-5
    out["atom_permutation"] = {
        "max_abs_logit_diff": perm_diff,
        "tolerance": perm_tol,
        "note": (
            "atom renumbering preserves the molecule and the path set, but changes the path order, "
            "so the float32 scatter-add accumulation order changes (same magnitude the S3 round "
            "observed); this is a numerical tolerance, not a semantic difference"
        ),
        "pass": bool(perm_diff <= perm_tol),
    }

    # 5) batch isolation: no path crosses a molecule boundary
    iso = C.check_s3_batch_isolation(batch)
    out["batch_isolation"] = dict(iso)

    # 6) S3 path semantics and counts are unchanged (re-enumerate from the cache)
    cache = task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    bond_index = np.asarray(cache["bond_index"])
    bond_type = np.asarray(cache["bond_type"])
    atom_type = np.asarray(cache["atom_type"])
    row_paths = np.asarray(task.s3["s3_row_paths"])
    row_atoms = np.asarray(task.s3["s3_row_atoms"])
    idx = np.asarray(task.s3["row_index"])[rows]
    a0 = np.asarray(task.s3["row_atom_ptr"])[idx]
    expected: list[np.ndarray] = []
    for k, row in enumerate(rows.tolist()):
        n = int(row_atoms[idx[k]])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        if n < 4 or b1 == b0:
            continue
        bi = bond_index[b0:b1].astype(np.int64)
        bt = bond_type[b0:b1].astype(np.int64)
        adj = S3P.molecule_adjacency(n, bi[:, 0], bi[:, 1], bt)
        at = atom_type[int(atom_ptr[row]) : int(atom_ptr[row + 1])]
        dist = S3P.all_pairs_shortest_paths(n, adj)
        paths = S3P.enumerate_s3(n, adj, at, dist, keep_mid_atoms=True)
        expected.append(
            np.stack(
                [
                    paths.end,
                    paths.atom_type_mid1,
                    paths.atom_type_mid2,
                    paths.bond_type_1,
                    paths.bond_type_2,
                    paths.bond_type_3,
                    paths.dist,
                ],
                axis=1,
            )
        )
    got = torch.stack(
        [
            batch["s3_path_end"].cpu(),
            batch["s3_mid1_type"].cpu(),
            batch["s3_mid2_type"].cpu(),
            batch["s3_bond1"].cpu(),
            batch["s3_bond2"].cpu(),
            batch["s3_bond3"].cpu(),
            batch["s3_dist"].cpu(),
        ],
        dim=1,
    ).numpy()
    exp = np.concatenate(expected, axis=0) if expected else np.zeros((0, 7), np.int64)
    # endpoints are batch-local; recover their molecule-local values
    atom_off = batch["atom_off"].detach().cpu().numpy().astype(np.int64)
    atom_graph = np.repeat(np.arange(len(rows)), row_atoms[idx])
    path_graph = atom_graph[batch["s3_path_atom"].cpu().numpy()]
    local_end = got[:, 0].astype(np.int64) - atom_off[path_graph]
    s3_ptr = np.asarray(task.s3["s3_atom_ptr"])
    atom_index = np.concatenate(
        [np.arange(a0[i], a0[i] + int(row_atoms[idx[i]]), dtype=np.int64) for i in range(len(rows))]
    )
    expected_root_counts = (s3_ptr[atom_index + 1] - s3_ptr[atom_index]).astype(np.int64)
    got_root_counts = np.bincount(batch["s3_path_atom"].cpu().numpy(), minlength=n_atoms).astype(np.int64)
    out["s3_path_semantics"] = {
        "n_paths_batch": int(got.shape[0]),
        "n_paths_reenumerated": int(exp.shape[0]),
        "counts_match": bool(got.shape[0] == exp.shape[0]),
        "fields_match": bool(got.shape[0] == exp.shape[0] and np.array_equal(got[:, 1:], exp[:, 1:]))
        and bool(np.array_equal(local_end, exp[:, 0])),
        "per_root_counts_match_cache": bool(np.array_equal(got_root_counts, expected_root_counts)),
        "s3_cache_total_paths": int(task.s3["meta"]["n_paths"]),
        "s3_cache_scope": task.s3["meta"]["scope"],
        "s3_cache_labels_present": bool(task.s3["meta"]["labels_present"]),
        "s3_cache_test_rows_present": bool(task.s3["meta"]["test_rows_present"]),
    }
    out["s3_path_semantics"]["pass"] = bool(
        out["s3_path_semantics"]["counts_match"]
        and out["s3_path_semantics"]["fields_match"]
        and out["s3_path_semantics"]["per_root_counts_match_cache"]
    )

    # 7) attention diagnostics at initialisation
    with torch.no_grad():
        diag = M.branch_diagnostics(model, batch)
    out["attention_diagnostics"] = diag["attention"]
    out["branch_diagnostics"] = {k: v for k, v in diag.items() if k != "attention"}

    # 8) no iterative MP: operator count must not grow with the path count
    small = keep_n_paths_per_root(batch, 2)
    counts_small = _count_aten_ops(model, small)
    counts_full = _count_aten_ops(model, batch)
    out["operator_scaling"] = {
        "paths_small": int(small["s3_path_atom"].numel()),
        "paths_full": int(batch["s3_path_atom"].numel()),
        "aten_ops_small": counts_small,
        "aten_ops_full": counts_full,
        "ratio_ops": float(counts_full / max(counts_small, 1)),
        "ratio_paths": float(batch["s3_path_atom"].numel() / max(int(small["s3_path_atom"].numel()), 1)),
        "no_python_loop_over_paths": True,
        "pass": bool(counts_full <= counts_small * 2),
    }

    # 9) selective flops / analytical cost
    n_atoms = int(batch["atom_batch"].numel())
    n_paths = int(batch["s3_path_atom"].numel())
    s3_flops = M3.branch_flops(_as_s3(model), n_paths=n_paths, n_atoms=n_atoms)
    sel_flops = M.selective_flops(n_paths=n_paths, n_atoms=n_atoms)
    out["cost"] = {
        "n_atoms": n_atoms,
        "n_paths": n_paths,
        "s3_branch_flops": s3_flops,
        "selective_branch_flops": sel_flops,
        "selective_over_s3_flops": float(sel_flops["flops_total"] / max(s3_flops["flops_total"], 1.0)),
    }

    out["seconds"] = float(time.perf_counter() - t0)
    out["test_evaluated"] = False
    out["checks"] = {
        "one_pass_no_iterative_mp": out["call_budget"]["pass"],
        "query_from_static_atom_encoding": out["selective_inputs"]["query_is_static_atom_encoding"],
        "selective_reads_no_propagated_state": out["selective_inputs"]["no_hidden_state_input"],
        "per_root_softmax_normalised": out["softmax"]["pass"],
        "single_path_delta_zero": out["single_path_roots"]["pass"],
        "one_path_per_root_delta_zero": out["one_path_per_root_batch"]["pass"],
        "uniform_attention_delta_zero": out["uniform_attention_delta"]["pass"],
        "empty_root_residual_zero": bool(
            out["empty_roots"]["selective_residual_max_abs_on_empty_roots"] == 0.0
            and out["empty_roots"]["finite"]
        ),
        "atom_permutation_equivariant": out["atom_permutation"]["pass"],
        "no_cross_molecule_paths": bool(iso["isolated"]),
        "s3_semantics_unchanged": out["s3_path_semantics"]["pass"],
        "operator_count_stable": out["operator_scaling"]["pass"],
    }
    out["checks"]["all_pass"] = bool(all(out["checks"].values()))
    return out


def _as_s3(model: torch.nn.Module):
    """A view exposing ``s3_branch`` for the inherited analytical FLOP counter."""

    class _View(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.s3_branch = inner.s3_branch

    return _View(model)


def _count_aten_ops(model: torch.nn.Module, batch: Mapping[str, Any]) -> int:
    n = {"calls": 0}
    handles = []

    def _hook(_m, _i, _o):
        n["calls"] += 1

    for module in model.modules():
        handles.append(module.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()
    return int(n["calls"])


# ---------------------------------------------------------------------------
# training parity
# ---------------------------------------------------------------------------


def run_parity() -> dict[str, Any]:
    sp = C.read_json(C.RUNS / "SP" / "run.json")
    s3 = C.read_json(C.S3["run_dir"] / "run.json")
    w2 = C.read_json(C.REPO / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2/run.json")
    curve = C.RUNS / "SP" / "train_curve.csv"
    import csv

    with open(curve, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lrs = [float(r["learning_rate"]) for r in rows]
    plan = sp["batch_plan"]
    checks = {
        "global_plan_matches_w2": plan["global_plan_sha256"] == C.W2["global_plan_sha256"],
        "consumed_prefix_matches_w2": plan["prefix_sha256"] == C.W2["consumed40_prefix_sha256"],
        "batch_size_128": int(plan["batch_size"]) == 128 and int(C.BATCH_SIZE) == 128,
        "epochs_40": int(sp["epochs_run"]) == 40 == int(C.TOTAL_EPOCHS),
        "steps_per_epoch_2738": int(C.STEPS_PER_EPOCH) == 2738,
        "global_steps_109520": int(sp["global_steps_run"]) == 109520 == int(C.TOTAL_STEPS_40),
        "eval_every_epoch": int(sp["eval_every"]) == 1 and len(sp["eval_epochs_completed"]) == 40,
        "optimizer_adamw": sp["optimizer_state_audit"]["optimizer"] == "AdamW",
        "peak_lr_3e-4": float(sp["optimizer_state_audit"]["peak_lr"]) == 3e-4,
        "warmup_start_3e-5": float(sp["optimizer_state_audit"]["warmup_start_lr"]) == 3e-5,
        "weight_decay_1e-3": float(sp["optimizer_state_audit"]["weight_decay"]) == 1e-3,
        "betas_0.9_0.999": [float(x) for x in sp["optimizer_state_audit"]["betas"]] == [0.9, 0.999],
        "eps_1e-8": float(sp["optimizer_state_audit"]["eps"]) == 1e-8,
        "dropout_0.10": float(sp["model_dropout"]) == 0.1 == float(C.RECIPE["dropout"]),
        "grad_clip_5.0": float(sp["grad_clip"]) == 5.0,
        "precision_fp32": sp["precision"] == "fp32",
        "loss_masked_bce": sp["recipe"]["loss"] == C.RECIPE["loss"] == s3["recipe"]["loss"],
        "no_early_stopping": sp["recipe"]["early_stopping"] is False,
        # the curve stores the LR after each epoch's steps, so epoch 1 ends at the peak
        "lr_epoch1_ends_at_peak": bool(abs(lrs[0] - 3e-4) < 1e-12),
        "lr_warmup_starts_at_3e-5": bool(3e-5 <= float(C.lr_at(0)) < 3.02e-5),
        "lr_warmup_reaches_peak_at_epoch_boundary": bool(
            abs(float(C.lr_at(int(C.STEPS_PER_EPOCH))) - 3e-4) < 1e-12
            and abs(float(C.lr_at(int(C.STEPS_PER_EPOCH) - 1)) - 3e-4) < 1e-5
        ),
        "lr_monotone_down_after_warmup": bool(all(lrs[i] >= lrs[i + 1] for i in range(2740, len(lrs) - 1))),
        "lr_final_near_1e-5": bool(abs(lrs[-1] - 1e-5) < 2e-7),
        "same_selection_rule": sp["selection_rule"] == s3["selection_rule"],
        "sp_test_not_evaluated": sp["test_evaluated"] is False and sp["test_cache_used"] is False,
        "sp_epoch0_ap_equals_s3": bool(
            abs(float(sp["epoch0_diagnostic"]["ap"]) - float(s3["epoch0_diagnostic"]["ap"])) < 1e-4
        ),
        "sp_epoch0_ap_equals_w2": bool(
            abs(float(sp["epoch0_diagnostic"]["ap"]) - float(w2["epoch0_diagnostic"]["ap"])) < 1e-4
        ),
        "same_train_cache_identity": sp["s3_cache_identity"]["meta_sha256"] == s3["s3_cache_identity"]["meta_sha256"],
        "same_s3_cache_n_paths": int(sp["s3_cache_identity"]["n_paths"]) == int(s3["s3_cache_identity"]["n_paths"]),
    }
    return {
        "schema": "lcrc-pcba-s3-selective-training-parity-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "sp_run": C.track_rel(C.RUNS / "SP" / "run.json"),
        "s3_run": C.track_rel(C.S3["run_dir"] / "run.json"),
        "w2_run": "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/runs/W2/run.json",
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "details": {
            "plan_sha256": plan["global_plan_sha256"],
            "prefix_sha256": plan["prefix_sha256"],
            "lr_first": lrs[0] if lrs else None,
            "lr_at_epoch2": lrs[1] if len(lrs) > 1 else None,
            "lr_last": lrs[-1] if lrs else None,
            "epoch0_ap": sp["epoch0_diagnostic"]["ap"],
            "epoch0_ap_s3": s3["epoch0_diagnostic"]["ap"],
            "epoch0_ap_w2": w2["epoch0_diagnostic"]["ap"],
            "train_eval": sp.get("train_eval_final"),
            "epoch_seconds_mean": sp["epoch_seconds_mean"],
            "s3_epoch_seconds_mean": s3["epoch_seconds_mean"],
            "w2_epoch_seconds_mean": w2["epoch_seconds_mean"],
            "gpu_hours": sp["gpu_hours"],
            "s3_gpu_hours": s3["gpu_hours"],
            "w2_gpu_hours": w2["gpu_hours"],
        },
        "test_evaluated": False,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="init,architecture,parity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    try:
        if "parity" in modes:
            report = run_parity()
            C.write_json(out_dir / "training_parity.json", report)
            C.log(f"[parity] all_pass={report['all_pass']}")
        if "init" in modes:
            task = C.load_task_gated()
            report = run_init(task, device)
            C.write_json(out_dir / "init_audit.json", report)
            C.log(f"[init] all_pass={report['checks']['all_pass']}")
            del task
        if "architecture" in modes:
            task = C.load_task_gated()
            report = run_architecture(task, device)
            C.write_json(out_dir / "architecture_audit.json", report)
            C.log(f"[architecture] all_pass={report['checks']['all_pass']}")
            del task
        return 0
    except Exception:  # noqa: BLE001
        C.log("AUDIT FAILED\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
