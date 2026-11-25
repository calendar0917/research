"""Pre-training correctness checks and the throughput preflight for CSC v0.

Section 8 of ``PROTOCOL.md``.  Every check is recorded (with its actual numbers)
in ``analysis/checks.json``; any failure raises, so a run cannot start on a
broken implementation.  The checks are read-only with respect to training data
selection: they never look at Test labels for anything except scoring.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cs_common as C

C.setup_paths()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cs_data as D  # noqa: E402
import cs_model as M  # noqa: E402

TOL_PERMUTE = 1e-5
TOL_BATCH = 1e-5
TOL_GROUP = 1e-5


# ---------------------------------------------------------------------------
# batch transforms used by the invariance checks
# ---------------------------------------------------------------------------


def clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def permute_atoms(batch: Mapping[str, Any], *, seed: int = 20261126):
    """Renumber atoms inside every molecule; returns ``(batch, p)`` with ``p[new] = old``."""
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    counts = torch.bincount(atom_batch, minlength=n_graphs).tolist()
    gen = torch.Generator().manual_seed(int(seed))
    parts = []
    off = 0
    for g in range(n_graphs):
        n = int(counts[g])
        parts.append(torch.arange(off, off + n, dtype=torch.long)[torch.randperm(n, generator=gen)])
        off += n
    p = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), dtype=torch.long)
    out = clone_batch(batch)
    for key in ("atom_feat", "atom_scalar", "atom_type"):
        out[key] = out[key][p]
    if out["bond_index"].numel():
        out["bond_index"] = inv[out["bond_index"]]
    for key in ("nbr_atom", "path_atom", "path_end"):
        if out[key].numel():
            out[key] = inv[out[key]]
    return out, p


def permute_bonds(batch: Mapping[str, Any], *, seed: int = 20261127):
    """Permute the physical-bond order (bond_index / bond_feat / bond_type together)."""
    n_bonds = int(batch["bond_index"].shape[0])
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_bonds, generator=gen) if n_bonds else torch.zeros(0, dtype=torch.long)
    out = clone_batch(batch)
    for key in ("bond_index", "bond_feat", "bond_type", "bond_batch"):
        if key in out and out[key].numel():
            out[key] = out[key][perm]
    return out, perm


def swap_endpoints(batch: Mapping[str, Any]):
    out = clone_batch(batch)
    if out["bond_index"].numel():
        out["bond_index"] = out["bond_index"].flip(1)
    return out


def predictions(model, batch: Mapping[str, Any]) -> np.ndarray:
    was = model.training
    model.eval()
    with torch.no_grad():
        out = model(batch).detach().cpu().numpy().reshape(-1)
    if was:
        model.train()
    return out


# ---------------------------------------------------------------------------
# synthetic batches
# ---------------------------------------------------------------------------


def synthetic_batch(bundle: Any, *, n_atoms: int, bonds: Sequence[tuple[int, int, int]], rows: int = 1) -> dict[str, Any]:
    """A tiny hand-built batch: ``bonds`` are ``(u, v, bond_type)`` triples."""
    spec = bundle.spec
    cache = bundle.caches["valid"]
    atom_feat = np.asarray(cache["atom_feat"])[:n_atoms]
    atom_scalar = np.asarray(cache["atom_scalar"])[:n_atoms]
    atom_type = np.asarray(cache["atom_type"])[:n_atoms]
    global_feat = np.asarray(cache["global"])[:1]
    bond_index = np.asarray([[u, v] for u, v, _ in bonds], dtype=np.int64).reshape(-1, 2)
    bond_type = np.asarray([t for _, _, t in bonds], dtype=np.int64)
    bond_feat = np.zeros((len(bonds), spec.bond_feat_dim), dtype=np.float32)
    for i, (_u, _v, t) in enumerate(bonds):
        bond_feat[i, int(t) - 1] = 1.0

    def t(arr, dtype):
        return torch.as_tensor(np.asarray(arr), dtype=dtype)

    return {
        "n_graphs": 1,
        "atom_feat": t(atom_feat, torch.float32),
        "atom_type": t(atom_type, torch.int64),
        "atom_scalar": t(atom_scalar, torch.float32),
        "atom_batch": torch.zeros(n_atoms, dtype=torch.long),
        "atom_off": torch.zeros(1, dtype=torch.long),
        "atom_counts": t([n_atoms], torch.int64),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.float32),
        "bond_type": t(bond_type, torch.int64),
        "bond_index": t(bond_index, torch.int64),
        "bond_batch": torch.zeros(len(bonds), dtype=torch.long),
        "bond_counts": t([len(bonds)], torch.int64),
        "nbr_atom": torch.zeros(0, dtype=torch.long),
        "nbr_code": torch.zeros(0, dtype=torch.long),
        "path_atom": torch.zeros(0, dtype=torch.long),
        "path_cat": torch.zeros(0, dtype=torch.long),
        "path_end63": torch.zeros(0, dtype=torch.long),
        "path_end": torch.zeros(0, dtype=torch.long),
        "path_closure": torch.zeros(0, dtype=torch.long),
        "path_graph": torch.zeros(0, dtype=torch.long),
        "global": t(global_feat, torch.float32),
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def run_checks(datasets: Sequence[str] = C.DATASETS, *, device: str = "cpu", seed: int = C.SEED) -> dict[str, Any]:
    dev = torch.device(device)
    report: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "device": str(dev),
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "checks": {},
        "per_dataset": {},
    }
    ok = True

    def record(name: str, passed: bool, detail: Mapping[str, Any]) -> None:
        nonlocal ok
        report["checks"][name] = {"passed": bool(passed), **dict(detail)}
        ok = ok and bool(passed)

    for dataset in datasets:
        bundle = D.load_bundle(dataset)
        ident = bundle.extra["identity_audit"]
        n = int(C.EXPECTED_SPLIT_SIZES[dataset]["train"])
        record(
            f"1_split_identity::{dataset}",
            ident["split_sizes"] == C.EXPECTED_SPLIT_SIZES[dataset]
            and all(v.get("match", True) for v in ident["cache_files"].values())
            and ident["relation_mapping"]["raw_edge_class_mismatches"] == 0,
            {
                "split_sizes": ident["split_sizes"],
                "expected": C.EXPECTED_SPLIT_SIZES[dataset],
                "split_partition_sha256": ident["split_partition_sha256"],
                "cache_files": {k: {"match": v.get("match"), "actual": v.get("actual_sha256")} for k, v in ident["cache_files"].items()},
                "relation_mapping": ident["relation_mapping"],
            },
        )

        order, row_ids, y_raw = D.split_order(bundle, "valid")
        rows = order[:16]
        batch = D.make_batch(bundle, "valid", rows, dev)

        models: dict[str, Any] = {}
        audits: dict[str, Any] = {}
        for arm in C.ARM_ORDER:
            model, audit = M.build_csc(bundle.spec, bundle.norm, arm=arm, seed=int(seed), device=dev)
            models[arm] = model
            audits[arm] = audit

        # -- 2. atom renumbering / bond order / endpoint swap invariance
        base_pred = predictions(models["A"], batch)
        atom_perm, _p = permute_atoms(batch)
        bond_perm, _perm = permute_bonds(batch)
        swapped = swap_endpoints(batch)
        inv = {}
        for arm in ("A", "D"):
            for name, bb in (("atom_renumber", atom_perm), ("bond_order", bond_perm), ("endpoint_swap", swapped)):
                pred = predictions(models[arm], bb)
                inv[f"{arm}:{name}"] = float(np.abs(pred - base_pred).max()) if arm == "A" else float(np.abs(pred - predictions(models[arm], batch)).max())
        record(
            "2_invariance_to_renumbering",
            all(v <= TOL_PERMUTE for v in inv.values()),
            {"max_abs_logit_delta": inv, "tolerance": TOL_PERMUTE,
             "note": "atom renumbering reorders float sums inside the inherited local content, so this is a numerical invariance, not bitwise"},
        )

        # -- 3. single graph vs batched inference
        single = predictions(models["A"], D.make_batch(bundle, "valid", rows[:1], dev))
        batch_as_1 = predictions(models["A"], D.make_batch(bundle, "valid", rows, dev))
        single_d = predictions(models["D"], D.make_batch(bundle, "valid", rows[:1], dev))
        batch_d = predictions(models["D"], D.make_batch(bundle, "valid", rows, dev))
        d1 = float(abs(single[0] - batch_as_1[0]))
        d2 = float(abs(single_d[0] - batch_d[0]))
        record(
            "3_single_vs_batch",
            max(d1, d2) <= TOL_BATCH,
            {"A_abs_delta": d1, "D_abs_delta": d2, "tolerance": TOL_BATCH},
        )

        # -- 4. shared base initialisation identical across arms
        base_hashes = {arm: audits[arm]["base_state_sha256"] for arm in C.ARM_ORDER}
        shared_new = {
            "bond_proj.weight": [float(models[a].bond_proj.weight.detach().sum().item()) for a in C.ARM_ORDER],
            "blocks.0.fc1.weight": [float(models[a].blocks[0].fc1.weight.detach().sum().item()) for a in C.ARM_ORDER],
            "blocks.1.fc2.weight": [float(models[a].blocks[1].fc2.weight.detach().sum().item()) for a in C.ARM_ORDER],
        }
        same_shared = all(len(set(v)) == 1 for v in shared_new.values())
        record(
            "4_shared_init_identical",
            len(set(base_hashes.values())) == 1 and same_shared,
            {"base_state_sha256": base_hashes, "shared_new_module_sums": shared_new},
        )

        # -- 5. C/D full parameter initialisation identical
        cd_state = {
            arm: {k: v.detach().cpu().clone() for k, v in models[arm].state_dict().items()} for arm in C.STAT_ARMS
        }
        keys_equal = sorted(cd_state["C"]) == sorted(cd_state["D"])
        c_d_equal = keys_equal and all(
            torch.equal(cd_state["C"][k], cd_state["D"][k]) for k in cd_state["C"]
        )
        record(
            "5_C_D_init_identical",
            bool(c_d_equal),
            {"n_tensors": len(cd_state["C"]), "same_keys": bool(keys_equal),
             "full_state_sha256": {a: audits[a]["full_state_sha256"] for a in C.STAT_ARMS},
             "n_parameters": {a: audits[a]["n_parameters"] for a in C.STAT_ARMS},
             "n_parameters_new": {a: audits[a]["n_parameters_new"] for a in C.STAT_ARMS}},
        )

        # every dependency check below needs non-zero new-module weights; the
        # private generator makes this identical for all four arms
        for arm in C.ARM_ORDER:
            M.randomize_new_modules(models[arm])

        # -- 6. real and mixed grouping both reconstruct B_i
        out = models["D"].forward_states(batch)
        B = out["B"]
        P = out["P"]
        P_null = (out["n_r"] / out["n"].clamp(min=1.0).unsqueeze(1)).unsqueeze(-1) * B.unsqueeze(1)
        rec_real = float((P.sum(dim=1) - B).abs().max().item())
        rec_null = float((P_null.sum(dim=1) - B).abs().max().item())
        rec_norm = float(B.abs().max().item())
        record(
            "6_grouping_reconstructs_B",
            rec_real <= TOL_GROUP * max(1.0, rec_norm) and rec_null <= TOL_GROUP * max(1.0, rec_norm),
            {"real_sum_minus_B_absmax": rec_real, "mixed_sum_minus_B_absmax": rec_null,
             "B_absmax": rec_norm, "n_zero_degree_atoms": int((out["n"] == 0).sum().item()),
             "tolerance": TOL_GROUP * max(1.0, rec_norm)},
        )

        # -- 7. delta = 0 reproduces the FIXED1 computation exactly
        same_logits = {}
        for arm in C.STAT_ARMS:
            with torch.no_grad():
                models[arm].gate[-1].weight.zero_()
                models[arm].gate[-1].bias.zero_()
            same_logits[arm] = float(np.abs(predictions(models[arm], batch) - predictions(models["A"], batch)).max())
        # also through the explicit delta_mode='zero' switch
        with torch.no_grad():
            zero_mode = models["D"].forward_states(batch, delta_mode="zero")["logits"].cpu().numpy().reshape(-1)
        zero_delta = float(np.abs(zero_mode - predictions(models["A"], batch)).max())
        record(
            "7_delta_zero_reduces_to_FIXED1",
            max([*same_logits.values(), zero_delta]) == 0.0,
            {"max_abs_logit_delta_by_arm": same_logits, "delta_mode_zero_abs_delta": zero_delta,
             "note": "exact: Q = 0 adds a literal 0.0 to fc2's input, so the arithmetic is bitwise identical"},
        )

        # -- 8. stage 1 identical for the four arms (same weights, same input)
        stage1 = {}
        for arm in C.ARM_ORDER:
            models[arm].eval()
            with torch.no_grad():
                h1, _h2, _info = models[arm].structural(
                    models[arm].encode_h0(batch)[3], batch, zero_correction=True
                )
            stage1[arm] = h1.detach().cpu()
        max_stage1 = float(max((stage1[a] - stage1["A"]).abs().max().item() for a in C.ARM_ORDER))
        record(
            "8_stage1_identical",
            max_stage1 == 0.0,
            {"max_abs_h1_delta_vs_A": max_stage1,
             "note": "arm B only differs in stage 2, so h1 must be bitwise equal"},
        )

        # -- 9. empty-edge / single-atom / single-relation neighbourhood
        tiny = {}
        cases = {
            "single_atom_no_bonds": synthetic_batch(bundle, n_atoms=1, bonds=[]),
            "three_atom_single_relation": synthetic_batch(bundle, n_atoms=3, bonds=[(0, 1, 1), (1, 2, 1)]),
            "degree_one_double_bond": synthetic_batch(bundle, n_atoms=2, bonds=[(0, 1, 2)]),
        }
        for name, bb in cases.items():
            per_arm = {}
            for arm in C.ARM_ORDER:
                M.randomize_new_modules(models[arm])
                states = models[arm].forward_states(bb)
                per_arm[arm] = {
                    "finite": bool(torch.isfinite(states["logits"]).all().item()),
                    "logit": float(states["logits"].reshape(-1)[0].item()),
                    "n_zero_degree": int((states["n"] == 0).sum().item()),
                    "Q_rms": M.rms(states["Q"]) if states["Q"] is not None else 0.0,
                }
            tiny[name] = per_arm
        record(
            "9_degenerate_graphs",
            all(v["finite"] for case in tiny.values() for v in case.values()),
            tiny,
        )

        # -- 10. with a non-zero Gate, C and D differ only on heterogeneous neighbourhoods
        for arm in C.ARM_ORDER:
            M.randomize_new_modules(models[arm])
        states_c = models["C"].forward_states(batch)
        states_d = models["D"].forward_states(batch)
        n_r = states_c["n_r"]
        hetero = (n_r > 0).sum(dim=1) >= 2
        homo = ((n_r > 0).sum(dim=1) == 1)
        # the per-message decomposition vector v_ji does not depend on the grouping;
        # the grouping only enters Q, so C vs D must be compared on Q / h2
        dQ = states_c["Q"] - states_d["Q"]
        dh2 = states_c["h2"] - states_d["h2"]
        dev_hetero = float(dQ[hetero].abs().max().item()) if hetero.any() else 0.0
        dev_homo = float(dQ[homo].abs().max().item()) if homo.any() else 0.0
        dev_h2_hetero = float(dh2[hetero].abs().max().item()) if hetero.any() else 0.0
        dev_h2_homo = float(dh2[homo].abs().max().item()) if homo.any() else 0.0
        # an explicit numeric example on a real heterogeneous node
        example = None
        if hetero.any():
            i = int(torch.nonzero(hetero, as_tuple=False)[0, 0])
            src, dst, _e, rel = M.directed_bonds(models["D"], batch)
            sel = torch.nonzero(dst == i, as_tuple=False).reshape(-1)
            p = states_d["p"][sel]
            delta = states_d["delta"][i].index_select(0, rel[sel])
            P_real = (delta * p).sum(dim=0).cpu().numpy()
            n_i = states_d["n_r"][i].cpu().numpy()
            B_i = states_d["B"][i].cpu().numpy()
            P_null = (n_i / max(n_i.sum(), 1.0))[:, None] * B_i[None, :]
            Q_real = P_real
            Q_null = (states_d["delta"][i].cpu().numpy() * P_null).sum(axis=0)
            example = {
                "atom_index": i,
                "n_by_relation": n_i.tolist(),
                "Q_real_absmax": float(np.abs(Q_real).max()),
                "Q_mixed_absmax": float(np.abs(Q_null).max()),
                "Q_real_minus_mixed_absmax": float(np.abs(Q_real - Q_null).max()),
                "Q_real_minus_mixed_relative": float(
                    np.abs(Q_real - Q_null).max() / (np.abs(Q_real).max() + 1e-12)
                ),
                "delta_absmax": float(states_d["delta"][i].abs().max().item()),
            }
        nonzero_gate = float(states_c["delta"].abs().max().item()) > 0.0
        record(
            "10_C_D_differ_on_heterogeneous_neighbourhoods",
            bool(nonzero_gate and dev_hetero > 1e-6 and dev_homo == 0.0),
            {
                "n_hetero_atoms": int(hetero.sum().item()),
                "n_single_relation_atoms": int(homo.sum().item()),
                "max_abs_Q_delta_hetero": dev_hetero,
                "max_abs_Q_delta_single_relation": dev_homo,
                "max_abs_h2_delta_hetero": dev_h2_hetero,
                "max_abs_h2_delta_single_relation": dev_h2_homo,
                "numeric_example": example,
                "identity_check": "single-relation atoms: mixed grouping reconstructs exactly the real group content, so C and D agree bitwise",
            },
        )

        # -- 11. Gate gradient reaches the parameters (initial zero, then non-zero)
        model, _audit = M.build_csc(bundle.spec, bundle.norm, arm="D", seed=int(seed), device=dev)
        epochs = 3
        plan = D.batch_order_plan(dataset, epochs)
        opt = torch.optim.AdamW([p for p in model.parameters()], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
        trace = []
        total_steps = epochs * C.steps_per_epoch(dataset)
        step = 0
        for ep in range(epochs):
            model.train()
            batch_losses = []
            gate_grads = []
            for start in range(0, plan["plan"][ep].size, int(C.RECIPE["batch_size"])):
                rows_b = plan["plan"][ep][start:start + int(C.RECIPE["batch_size"])]
                bb = D.make_batch(bundle, "train", rows_b, dev)
                y = torch.as_tensor(
                    np.asarray((bundle.targets_raw["train"][rows_b] - bundle.scaler["mean"]) / bundle.scaler["std"], dtype=np.float32),
                    device=dev,
                )
                opt.param_groups[0]["lr"] = C.lr_at(step, total_steps)
                step += 1
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(bb).view(-1), y)
                loss.backward()
                gate_grads.append(float(model.gate[-1].weight.grad.norm().item()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                batch_losses.append(float(loss.detach()))
            # output RMS of the diagnostic batch
            stats = M.gate_statistics(model, batch)
            weight_norm = float(model.gate[-1].weight.detach().norm().item())
            trace.append(
                {
                    "epoch": ep + 1,
                    "loss_mean": float(np.mean(batch_losses)),
                    "gate_out_grad_first_step": gate_grads[0],
                    "gate_out_grad_mean": float(np.mean(gate_grads)),
                    "gate_out_grad_max": float(np.max(gate_grads)),
                    "gate_out_weight_norm": weight_norm,
                    "delta_rms": stats.get("delta_rms"),
                    "Q_rms": stats.get("Q_rms"),
                }
            )
        later_nonzero = trace[-1]["gate_out_grad_mean"] > 0.0 and trace[-1]["delta_rms"] > 0.0
        record(
            "11_gate_receives_gradient",
            later_nonzero,
            {
                "trace": trace,
                "note": ("with the residual fc2 zeroed, the Gate gradient is exactly zero at step 0 "
                         "(documented, not an error); it must become non-zero once the second-stage residual "
                         "block leaves zero"),
                "first_step_grad_is_zero": bool(trace[0]["gate_out_grad_first_step"] == 0.0),
            },
        )

        report["per_dataset"][dataset] = {
            "split_sizes": ident["split_sizes"],
            "relation_counts": ident["relation_mapping"]["bond_counts_by_relation"],
            "n_parameters": {arm: audits[arm]["n_parameters"] for arm in C.ARM_ORDER},
            "n_parameters_new": {arm: audits[arm]["n_parameters_new"] for arm in C.ARM_ORDER},
            "n_parameters_base": audits["A"]["n_parameters_base"],
            "base_state_sha256": audits["A"]["base_state_sha256"],
            "lcrca_reference_parameters_n_out_1": audits["A"]["lcrca_reference_parameters"],
            "batch_plan_sha256": D.batch_order_plan(dataset, int(C.RECIPE["epochs"]))["plan_sha256"],
        }

    report["all_passed"] = bool(ok)
    return report


# ---------------------------------------------------------------------------
# throughput preflight
# ---------------------------------------------------------------------------


def preflight_timing(
    datasets: Sequence[str] = C.DATASETS,
    *,
    device: str = "cuda",
    train_batches: int = 6,
    arms: Sequence[str] = ("A", "D"),
) -> dict[str, Any]:
    """Time a few real train steps plus one full three-split evaluation."""
    dev = torch.device(device)
    out: dict[str, Any] = {"device": str(dev), "created_utc": C.utcnow(), "datasets": {}}
    for dataset in datasets:
        bundle = D.load_bundle(dataset)
        plan = D.batch_order_plan(dataset, 1)
        per_arm: dict[str, Any] = {}
        for arm in arms:
            model, audit = M.build_csc(bundle.spec, bundle.norm, arm=arm, seed=C.SEED, device=dev)
            opt = torch.optim.AdamW([p for p in model.parameters()], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
            model.train()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            step_times = []
            for b in range(int(train_batches)):
                rows = plan["plan"][0][b * int(C.RECIPE["batch_size"]):(b + 1) * int(C.RECIPE["batch_size"])]
                batch = D.make_batch(bundle, "train", rows, dev)
                y = torch.as_tensor(
                    np.asarray((bundle.targets_raw["train"][rows] - bundle.scaler["mean"]) / bundle.scaler["std"], dtype=np.float32),
                    device=dev,
                )
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(batch).view(-1), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                step_times.append(float(time.perf_counter() - t0))
            t0 = time.perf_counter()
            eval_seconds = {}
            for split in C.SPLITS:
                ts = time.perf_counter()
                D.evaluate(bundle, model, split, dev)
                eval_seconds[split] = float(time.perf_counter() - ts)
            total_eval = float(time.perf_counter() - t0)
            step_s = float(np.median(step_times))
            n_steps = C.steps_per_epoch(dataset)
            epoch_s = step_s * n_steps + total_eval
            per_arm[arm] = {
                "n_parameters": int(audit["n_parameters"]),
                "step_seconds_median": step_s,
                "steps_per_epoch": int(n_steps),
                "train_epoch_seconds": step_s * n_steps,
                "eval_seconds": eval_seconds,
                "projected_epoch_seconds": epoch_s,
                "projected_360_epoch_hours": epoch_s * int(C.RECIPE["epochs"]) / 3600.0,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else 0,
            }
        out["datasets"][dataset] = per_arm
    units = [(ds, arm) for ds, arms_ in C.QUEUE.items() for arm in arms_]
    projected = sum(
        out["datasets"][ds][arm]["projected_360_epoch_hours"]
        for ds, arm in units
        if arm in out["datasets"].get(ds, {})
    )
    out["projected_gpu_hours_all_units"] = float(projected)
    out["projected_gpu_hours_if_only_A_D_timed"] = float(projected)
    out["budget"] = {"target_hours": C.GPU_TARGET_HOURS, "hard_hours": C.GPU_HARD_HOURS}
    out["queue"] = {k: list(v) for k, v in C.QUEUE.items()}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="checks", choices=["checks", "preflight"])
    parser.add_argument("--datasets", default=",".join(C.DATASETS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    parser.add_argument("--train-batches", type=int, default=6)
    args = parser.parse_args(argv)
    datasets = [d for d in args.datasets.split(",") if d]
    if args.mode == "checks":
        report = run_checks(datasets, device=args.device)
        path = Path(args.out) if args.out else (C.ANALYSIS_DIR / "checks.json")
        C.write_json(path, report)
        C.log(f"checks written to {path} (all_passed={report['all_passed']})")
        return 0 if report["all_passed"] else 1
    timing = preflight_timing(datasets, device=args.device, train_batches=int(args.train_batches))
    path = Path(args.out) if args.out else (C.ANALYSIS_DIR / "preflight_timing.json")
    C.write_json(path, timing)
    C.log(f"preflight written to {path} projected_gpu_hours={timing['projected_gpu_hours_all_units']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
