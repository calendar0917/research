"""Targeted preflight: structure/initialisation, source reach, channel usage,
evaluation isolation, determinism and cost - nothing else.

Sections (``--sections a,b,c,d,e,f``):

``a``  parameter counts, shared init, epoch-0 equivalence, an independent
       hand-written aggregation on a 3-atom graph, zero-bond finiteness,
       atom-renumbering invariance and molecule isolation.
``b``  independent h0 chain A-B-C: which ``h0`` token can reach ``h2_C`` for
       S / R1 / R2 / D2 (token level only, on a throw-away model).
``c``  the layer-2 source channel is actually used: gradient on every new
       module, ``fc2`` displacement, source and prediction response at
       optimisation steps 1/3/10/100.
``d``  20-batch replay with an inserted full-split evaluation, and a
       replay-determinism control.
``e``  determinism configuration and a bitwise repeat of forward/backward.
``f``  real training + three full-split evaluations per epoch: measured cost,
       conservative extrapolation and an independent NumPy metric recompute.

This module never touches the formal initial state of a run: b/c/d use
throw-away models on tiny or subsetted batches.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import lp_common as C
import lp_data as D
import lp_model as M

import dissect_model as DM
import lcrc_data as PREP_DATA


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _c1_state(view: D.View) -> dict[str, torch.Tensor]:
    full = DM.build_full_state(view.spec, view.train_norm, int(C.TRAIN_SEED))
    return DM.remap_state("C1", full)


def _arm_model(view: D.View, arm: str, *, randomise: bool = False, seed: int = 4242) -> M.LipoContextModel:
    state = _c1_state(view)
    model = M.build_arm(view.spec, view.train_norm, arm)
    model.base.load_state_dict(state, strict=True)
    if model.kind == "context":
        model.init_new_modules()
        if randomise:
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for _, param in model.new_parameters():
                    param.copy_(torch.empty_like(param).normal_(0.0, 0.35, generator=generator))
    return model


def _chain_struct(label_map: Mapping[int, int] | None = None) -> dict[str, torch.Tensor]:
    """A-B-C chain, bonds (0,1) and (1,2) expanded to both directions.

    ``label_map`` relabels atoms first (for the renumbering-invariance check);
    the physical bond list is re-sorted afterwards and its features carried
    along, so the (bond, feature) pairing stays correct.
    """
    bonds = [((0, 1), (1.0, 0.0)), ((1, 2), (0.0, 1.0))]
    items = []
    for (a, b), feat in bonds:
        if label_map is not None:
            a, b = int(label_map[a]), int(label_map[b])
        items.append((min(a, b), max(a, b), feat))
    items.sort(key=lambda t: (t[0], t[1]))
    u = torch.as_tensor([t[0] for t in items], dtype=torch.long)
    v = torch.as_tensor([t[1] for t in items], dtype=torch.long)
    bond_feat = torch.zeros((len(items), C.BOND_FEAT_DIM), dtype=torch.float32)
    for i, (_, _, feat) in enumerate(items):
        bond_feat[i, 0], bond_feat[i, 1] = feat
    deg = torch.zeros(3, dtype=torch.float32).index_add_(
        0, torch.cat([u, v]), torch.ones(2 * len(items), dtype=torch.float32)
    )
    return {
        # the model's directed-edge convention: all u->v, then all v->u
        "src": torch.cat([u, v]),
        "dst": torch.cat([v, u]),
        "bond_feat": bond_feat,
        "deg": deg,
        "n_atoms": 3,
    }


def _h2_at(model: M.LipoContextModel, h0: torch.Tensor, struct, source: str, atom: int) -> torch.Tensor:
    return model.message_layers(h0, struct, source)["h2"][atom]


def _sensitivity(model, h0, struct, source, atom: int, eps: float = 1e-3) -> float:
    base = _h2_at(model, h0, struct, source, atom=2)
    perturbed = h0.clone()
    perturbed[atom] = perturbed[atom] + eps
    other = _h2_at(model, perturbed, struct, source, atom=2)
    return float((base - other).abs().max() / eps)


# ---------------------------------------------------------------------------
# section a
# ---------------------------------------------------------------------------


def section_a(view: D.View) -> dict[str, Any]:
    device = torch.device("cpu")
    out: dict[str, Any] = {}
    batch = D.to_device(PREP_DATA.collate(view.caches["train"], list(range(8))), device)

    base = DM.DissectLCRC(view.spec, view.train_norm, "C1")
    base.load_state_dict(_c1_state(view), strict=True)
    base.eval()
    with torch.no_grad():
        y_base = base(batch)
    out["base_parameters"] = M.parameter_count(base)
    out["base_parameters_expected"] = C.EXPECTED_PARAMS["B"]
    out["base_parameters_match"] = out["base_parameters"] == C.EXPECTED_PARAMS["B"]

    per_arm: dict[str, Any] = {}
    drawn_hashes: dict[str, str] = {}
    deltas: dict[str, float] = {}
    for arm in C.ARM_SPECS:
        if arm == C.DM_ARM:
            continue
        model = _arm_model(view, arm)
        model.eval()
        with torch.no_grad():
            y = model(batch)
        delta = float((y - y_base).abs().max())
        deltas[arm] = delta
        entry: dict[str, Any] = {
            "parameters": M.parameter_count(model),
            "parameters_expected": C.EXPECTED_PARAMS[arm],
            "parameters_match": M.parameter_count(model) == C.EXPECTED_PARAMS[arm],
            "delta_vs_base_at_init": delta,
            "init_state_hash": D.state_hash(model.state_dict()),
            "new_module_params": sum(p.numel() for _, p in model.new_parameters()),
        }
        if model.kind == "context":
            info = model.init_new_modules()
            digest = C.sha256_bytes(
                b"".join(
                    np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes()
                    for _, v in sorted(info["drawn"].items())
                )
            )
            drawn_hashes[arm] = digest
            replay = model.init_new_modules()
            replay_digest = C.sha256_bytes(
                b"".join(
                    np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes()
                    for _, v in sorted(replay["drawn"].items())
                )
            )
            entry["private_init_replay_equal"] = digest == replay_digest
            entry["fc2_zero_after_init"] = all(
                float(b.fc2.weight.abs().max()) == 0.0 and float(b.fc2.bias.abs().max()) == 0.0 for b in model.blocks
            )
            entry["layer_norm_identity_after_init"] = all(
                bool(torch.equal(b.norm.weight, torch.ones_like(b.norm.weight)))
                and bool(torch.equal(b.norm.bias, torch.zeros_like(b.norm.bias)))
                for b in model.blocks
            )
            model.eval()
            with torch.no_grad():
                entry["delta_vs_base_after_reinit"] = float((model(batch) - y_base).abs().max())
        per_arm[arm] = entry
    out["arms"] = per_arm
    out["new_parameter_budget"] = {
        "bond_proj": C.BOND_PROJ_PARAMS,
        "message_ffn_each": C.MESSAGE_FFN_PARAMS,
        "total_added": C.NEW_PARAMS,
    }
    out["shared_new_init_across_context_arms"] = len(set(drawn_hashes.values())) == 1
    out["private_init_hashes"] = drawn_hashes
    out["private_init_does_not_advance_global_rng"] = _global_rng_untouched(view)
    out["epoch0_equivalence_max_abs_delta"] = float(max(deltas.values()))
    out["epoch0_equivalence_passed"] = out["epoch0_equivalence_max_abs_delta"] <= C.ZERO_DRIFT_TOL

    # ---- independent hand-written aggregation ---------------------------
    model = _arm_model(view, "D2", randomise=True)
    model.eval()
    struct = _chain_struct()
    generator = torch.Generator().manual_seed(7)
    h0 = torch.empty(3, C.WIDTH).normal_(0.0, 1.0, generator=generator)
    nbrs = {0: [1], 1: [0, 2], 2: [1]}
    with torch.no_grad():
        observed = model.message_layers(h0, struct, "dynamic")
        q = model.bond_proj(struct["bond_feat"])
        edge_q = {(0, 1): q[0], (1, 0): q[0], (1, 2): q[1], (2, 1): q[1]}
        states = [h0]
        for block in model.blocks:
            x = states[-1]
            z = block.norm(x)
            v = block.norm(x)
            msg = torch.zeros_like(z)
            for i, js in nbrs.items():
                acc = torch.zeros_like(z[i])
                for j in js:
                    acc = acc + torch.nn.functional.silu(v[j] + edge_q[(j, i)])
                msg[i] = acc
            states.append(x + block.fc2(torch.nn.functional.silu(block.fc1(z + msg))))
        h1_ref, h2_ref = states[1], states[2]
    out["independent_aggregation"] = {
        "layer1_max_abs_diff": float((observed["h1"] - h1_ref).abs().max()),
        "layer2_max_abs_diff": float((observed["h2"] - h2_ref).abs().max()),
    }
    out["independent_aggregation_passed"] = (
        out["independent_aggregation"]["layer1_max_abs_diff"] < 1e-6
        and out["independent_aggregation"]["layer2_max_abs_diff"] < 1e-6
    )

    # ---- zero-bond molecule --------------------------------------------
    zero_struct = {
        "src": torch.zeros(0, dtype=torch.long),
        "dst": torch.zeros(0, dtype=torch.long),
        "bond_feat": torch.zeros((0, C.BOND_FEAT_DIM)),
        "deg": torch.zeros(1),
        "n_atoms": 1,
    }
    h0_single = torch.ones(1, C.WIDTH)
    zero_bond: dict[str, Any] = {}
    for arm in C.CONTEXT_ARMS:
        m = _arm_model(view, arm, randomise=True)
        m.eval()
        with torch.no_grad():
            layers = m.message_layers(h0_single, zero_struct, m.spec["source"])
            extended = m._extended_source(h0_single, zero_struct["src"], zero_struct["dst"], zero_struct["deg"])
        zero_bond[arm] = {
            "h2_finite": bool(torch.isfinite(layers["h2"]).all()),
            "h2_equals_h0": float((layers["h2"] - h0_single).abs().max()) <= C.ZERO_DRIFT_TOL,
            "extended_equals_h0": float((extended - h0_single).abs().max()) <= C.ZERO_DRIFT_TOL,
        }
    out["zero_bond"] = zero_bond
    out["zero_bond_passed"] = all(v["h2_finite"] and v["extended_equals_h0"] for v in zero_bond.values())

    # ---- atom renumbering ----------------------------------------------
    perm = [2, 0, 1]                          # new atom i holds old atom perm[i]
    inv = {old: new for new, old in enumerate(perm)}
    h0_perm = h0[perm]
    struct_perm = _chain_struct(label_map=inv)
    renumber: dict[str, float] = {}
    for arm in C.CONTEXT_ARMS:
        m = _arm_model(view, arm, randomise=True)
        m.eval()
        with torch.no_grad():
            ref = m.message_layers(h0, struct, m.spec["source"])["h2"]
            alt = m.message_layers(h0_perm, struct_perm, m.spec["source"])["h2"]
        renumber[arm] = float((alt - ref[perm]).abs().max())
    out["renumbering_max_abs_diff"] = max(renumber.values())
    out["renumbering_by_arm"] = renumber
    out["renumbering_equivariant"] = out["renumbering_max_abs_diff"] < 1e-5

    # ---- molecule isolation --------------------------------------------
    rows = [int(view.rows["train"][0]), int(view.rows["train"][1])]
    model = _arm_model(view, "D2", randomise=True).to(device)
    model.eval()
    with torch.no_grad():
        single = []
        for r in rows:
            b = D.to_device(PREP_DATA.collate(view.caches["train"], [r]), device)
            single.append(float(model(b)))
        paired = D.to_device(PREP_DATA.collate(view.caches["train"], rows), device)
        paired_preds = model(paired)
    iso = float(max(abs(float(paired_preds[0]) - single[0]), abs(float(paired_preds[1]) - single[1])))
    out["molecule_isolation_max_abs_diff"] = iso
    out["molecule_isolation_passed"] = iso < 1e-5

    # ---- bond direction (both directions already cover the same q) ------
    with torch.no_grad():
        struct_rev = dict(struct)
        struct_rev["src"], struct_rev["dst"] = struct["dst"], struct["src"]
        a = model.message_layers(h0, struct, "dynamic")["h2"]
        b = model.message_layers(h0, struct_rev, "dynamic")["h2"]
    out["bond_direction_max_abs_diff"] = float((a - b).abs().max())
    out["bond_direction_invariant"] = out["bond_direction_max_abs_diff"] < 1e-6
    return out


def _global_rng_untouched(view: D.View) -> bool:
    """The private-init generator must not touch the shared CPU RNG."""
    model = _arm_model(view, "D2")
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    model.init_new_modules()
    after = torch.get_rng_state().clone()
    return bool(torch.equal(before, after))


# ---------------------------------------------------------------------------
# section b
# ---------------------------------------------------------------------------


def section_b(view: D.View) -> dict[str, Any]:
    struct = _chain_struct()
    generator = torch.Generator().manual_seed(11)
    h0 = torch.empty(3, C.WIDTH).normal_(0.0, 1.0, generator=generator)
    expectations = {
        "S": {"A": False, "B": False},
        "R1": {"A": False, "B": True},
        "R2": {"A": True, "B": True},
        "D2": {"A": True, "B": True},
    }
    out: dict[str, Any] = {}
    for arm in C.CONTEXT_ARMS:
        model = _arm_model(view, arm, randomise=True)
        model.eval()
        source = model.spec["source"]
        h = h0.clone().requires_grad_(True)
        value = model.message_layers(h, struct, source)["h2"][2]
        grad = torch.autograd.grad(value.sum(), h)[0]
        row: dict[str, Any] = {
            "source": source,
            "grad_norm_by_atom": [float(g.norm()) for g in grad],
            "sensitivity_A": _sensitivity(model, h0, struct, source, 0),
            "sensitivity_B": _sensitivity(model, h0, struct, source, 1),
        }
        row["reads_A"] = bool(row["grad_norm_by_atom"][0] > 1e-8)
        row["reads_B"] = bool(row["grad_norm_by_atom"][1] > 1e-8)
        row["expectation"] = expectations[arm]
        row["matches_expectation"] = bool(
            row["reads_A"] == expectations[arm]["A"] and row["reads_B"] == expectations[arm]["B"]
        )
        out[arm] = row
    out["note"] = (
        "token-level only: the tested quantity is h2 at atom C of a 3-atom chain A-B-C. "
        "No whole-molecule readout is involved and no formal initial state is touched."
    )
    out["passed"] = all(v["matches_expectation"] for v in out.values() if isinstance(v, dict) and "matches_expectation" in v)
    return out


# ---------------------------------------------------------------------------
# shared training-step helper (mirrors lp_train's loop)
# ---------------------------------------------------------------------------


def _train_steps(
    view: D.View,
    arm: str,
    steps: int,
    *,
    device,
    seed: int = int(C.TRAIN_SEED),
    eval_hook: Callable[[int, torch.nn.Module], None] | None = None,
    hooks: Mapping[int, Callable[..., dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    model = _arm_model(view, arm).to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.C1_RECIPE["peak_lr"]),
        betas=tuple(C.C1_RECIPE["adam_betas"]),
        eps=float(C.C1_RECIPE["adam_eps"]),
        weight_decay=float(C.C1_RECIPE["weight_decay"]),
    )
    y_scaled = view.y_scaled("train").to(device)
    plan = D.epoch_plan(int(view.caches["train"]["n_molecules"]), 1, seed)["plan"][0]
    batch_size = int(C.C1_RECIPE["batch_size"])
    perm = np.asarray(plan, dtype=np.int64)
    batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
    records: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        rows = batches[(step - 1) % len(batches)]
        batch = D.to_device(PREP_DATA.collate(view.caches["train"], rows), device)
        y = y_scaled[torch.as_tensor(rows, device=device)]
        model.train()
        pred = model(batch).reshape(-1)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if hooks is not None and step in hooks:
            records.append({"step": step, **hooks[step](model, optimizer, batch, loss)})
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.C1_RECIPE["gradient_clip"]))
        optimizer.step()
        if eval_hook is not None:
            eval_hook(step, model)
    return {"model": model, "optimizer": optimizer, "records": records, "state": D.state_hash(model.state_dict())}


# ---------------------------------------------------------------------------
# section c
# ---------------------------------------------------------------------------


def section_c(view: D.View, device, *, steps: int = 100) -> dict[str, Any]:
    probe = D.to_device(PREP_DATA.collate(view.caches["valid"], list(range(16))), device)
    watched = [s for s in (1, 3, 10, 100) if s <= steps]

    def make_hook(step: int):
        def inner(model, optimizer, batch, loss):
            grads = {}
            for name, param in model.new_parameters():
                grads[name] = float(param.grad.norm()) if param.grad is not None else 0.0
            base_grad = model.base.local.mlp.lins[0].weight.grad
            was_training = model.training
            model.eval()
            with torch.no_grad():
                normal = model(probe)
                reused = model(probe, source_override="reuse")
                fixed = model(probe, source_override="fixed")
                _, aux = model(probe, return_aux=True)
                _, aux_reuse = model(probe, source_override="reuse", return_aux=True)
            model.train(was_training)
            return {
                "grad_norm_bond_proj": grads.get("bond_proj.weight", 0.0),
                "grad_norm_block0_fc1": grads.get("blocks.0.fc1.weight", 0.0),
                "grad_norm_block1_fc1": grads.get("blocks.1.fc1.weight", 0.0),
                "grad_norm_block1_fc2": grads.get("blocks.1.fc2.weight", 0.0),
                "grad_norm_base_local": float(base_grad.norm()) if base_grad is not None else 0.0,
                "fc2_max_abs_weight": max(float(b.fc2.weight.abs().max()) for b in model.blocks),
                "message_norm_layer2": float(aux["h2"].sub(aux["h1"]).norm()),
                "prediction_response_reuse": float((normal - reused).abs().max()),
                "prediction_response_fixed": float((normal - fixed).abs().max()),
                "hidden_response_reuse": float((aux["h2"] - aux_reuse["h2"]).abs().max()),
            }

        return inner

    out = _train_steps(view, "D2", steps, device=device, hooks={s: make_hook(s) for s in watched})
    records = {int(r["step"]): {k: v for k, v in r.items() if k != "step"} for r in out["records"]}
    late = [s for s in records if s >= 10]
    checks = {
        "fc2_gradient_nonzero_at_every_watched_step": all(r["grad_norm_block1_fc2"] > 0 for r in records.values()),
        "bond_proj_gradient_nonzero_after_first_update": all(
            records[s]["grad_norm_bond_proj"] > 0 for s in records if s >= 3
        ),
        "fc2_moved_away_from_zero": all(r["fc2_max_abs_weight"] > 1e-4 for s, r in records.items() if s >= 3),
        "prediction_responds_to_layer2_source_late": all(records[s]["prediction_response_reuse"] > 0 for s in late),
        "second_layer_update_nonzero_late": all(records[s]["message_norm_layer2"] > 0 for s in late),
    }
    return {
        "arm": "D2",
        "steps_run": steps,
        "watched_steps": watched,
        "records": {str(k): v for k, v in sorted(records.items())},
        "final_state_hash": out["state"],
        "checks": checks,
        "channel_used": all(checks.values()),
        "note": (
            "with both fc2 layers zero-initialised the source channel reaches the loss only "
            "through fc2.weight at step 1, so the bond_proj / fc1 gradients are expected to "
            "become non-zero from the first weight update on"
        ),
    }


# ---------------------------------------------------------------------------
# section d
# ---------------------------------------------------------------------------


def section_d(view: D.View, device) -> dict[str, Any]:
    steps = 20
    plain = _train_steps(view, "D2", steps, device=device)
    inserted: list[int] = []

    def eval_hook(step, model):
        if step == 10:
            D.evaluate_all_splits(lambda split: D.predict_cache(model, view.caches[split], device), view, device)
            inserted.append(step)

    with_eval = _train_steps(view, "D2", steps, device=device, eval_hook=eval_hook)
    replay = _train_steps(view, "D2", steps, device=device)
    return {
        "steps": steps,
        "inserted_eval_at": inserted,
        "state_equal_with_inserted_eval": plain["state"] == with_eval["state"],
        "plain_state_hash": plain["state"],
        "eval_state_hash": with_eval["state"],
        "replay_state_hash": replay["state"],
        "replay_deterministic": plain["state"] == replay["state"],
        "note": "a full Train/Valid/Test evaluation inserted mid-epoch must not change the next step",
    }


# ---------------------------------------------------------------------------
# section e
# ---------------------------------------------------------------------------


def section_e(view: D.View, device) -> dict[str, Any]:
    config = C.setup_determinism()
    model = _arm_model(view, "D2").to(device)
    batch = D.to_device(PREP_DATA.collate(view.caches["train"], list(range(32))), device)
    y = view.y_scaled("train")[:32].to(device)
    grads: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    error = None
    snap = D.rng_snapshot(device)
    try:
        for _ in range(2):
            # identical stochastic state for both passes (dropout is active in train mode)
            D.rng_restore(snap, device)
            model.train()
            model.zero_grad(set_to_none=True)
            pred = model(batch).reshape(-1)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            preds.append(pred.detach().clone())
            grads.append(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]))
        D.rng_restore(snap, device)
        repeat_max = float((grads[0] - grads[1]).abs().max())
        pred_max = float((preds[0] - preds[1]).abs().max())
    except RuntimeError as exc:
        repeat_max = float("nan")
        pred_max = float("nan")
        error = f"{type(exc).__name__}: {exc}"
    passed = error is None and repeat_max == 0.0 and pred_max == 0.0
    return {
        "config": config,
        "forward_backward_repeat_max_abs_grad_diff": repeat_max,
        "forward_repeat_max_abs_pred_diff": pred_max,
        "error": error,
        "bitwise_repeatable": passed,
        "status": "PASSED" if passed else C.NUMERICAL_ALARM,
        "adaptation_used": "none",
    }


# ---------------------------------------------------------------------------
# section f
# ---------------------------------------------------------------------------


def section_f(view: D.View, device, *, epochs: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {"per_arm": {}}
    n_train = int(view.caches["train"]["n_molecules"])
    for arm in ("B", "D2"):
        model = _arm_model(view, arm).to(device)
        torch.manual_seed(int(C.TRAIN_SEED))
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(C.C1_RECIPE["peak_lr"]),
            betas=tuple(C.C1_RECIPE["adam_betas"]),
            eps=float(C.C1_RECIPE["adam_eps"]),
            weight_decay=float(C.C1_RECIPE["weight_decay"]),
        )
        total_updates = math.ceil(n_train / int(C.C1_RECIPE["batch_size"])) * C.arm_epochs(arm)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=float(C.C1_RECIPE["min_lr"])
        )
        plan = D.epoch_plan(n_train, C.arm_epochs(arm), int(C.TRAIN_SEED))["plan"]
        y_scaled = view.y_scaled("train").to(device)
        timings = []
        metrics: dict[str, dict[str, float]] = {}
        preds_by_split: dict[str, np.ndarray] = {}
        for epoch in range(epochs):
            model.train()
            t0 = time.perf_counter()
            perm = np.asarray(plan[epoch], dtype=np.int64)
            for start in range(0, len(perm), int(C.C1_RECIPE["batch_size"])):
                rows = perm[start : start + int(C.C1_RECIPE["batch_size"])]
                batch = D.to_device(PREP_DATA.collate(view.caches["train"], rows), device)
                y = y_scaled[torch.as_tensor(rows, device=device)]
                pred = model(batch).reshape(-1)
                loss = torch.nn.functional.mse_loss(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.C1_RECIPE["gradient_clip"]))
                optimizer.step()
                scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - t0
            t1 = time.perf_counter()
            metrics, preds_by_split = D.evaluate_all_splits(
                lambda split: D.predict_cache(model, view.caches[split], device),
                view,
                device,
                keep_predictions=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            eval_seconds = time.perf_counter() - t1
            timings.append({"epoch": epoch + 1, "train_seconds": train_seconds, "eval_seconds": eval_seconds})
        per_epoch = float(np.mean([t["train_seconds"] + t["eval_seconds"] for t in timings]))
        # independent NumPy recompute of the raw metrics
        recompute = {}
        for split in C.SPLITS:
            raw = view.to_raw(preds_by_split[split])
            err = raw - view.y[split]
            recompute[split] = {
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "reported_rmse": float(metrics[split]["rmse"]),
                "reported_mae": float(metrics[split]["mae"]),
                "rmse_abs_diff": abs(float(np.sqrt(np.mean(err**2))) - float(metrics[split]["rmse"])),
                "mae_abs_diff": abs(float(np.mean(np.abs(err))) - float(metrics[split]["mae"])),
            }
        out["per_arm"][arm] = {
            "timings": timings,
            "mean_train_seconds": float(np.mean([t["train_seconds"] for t in timings])),
            "mean_eval_seconds": float(np.mean([t["eval_seconds"] for t in timings])),
            "mean_epoch_seconds": per_epoch,
            "planned_epochs": C.arm_epochs(arm),
            "projected_gpu_hours": per_epoch * C.arm_epochs(arm) / 3600.0,
            "projected_with_25pct_margin": 1.25 * per_epoch * C.arm_epochs(arm) / 3600.0,
            "metrics_last_epoch": {k: dict(v) for k, v in metrics.items()},
            "numpy_recompute": recompute,
        }
    total = sum(v["projected_with_25pct_margin"] for v in out["per_arm"].values())
    out["projected_total_c1_arm_gpu_hours_with_margin"] = total
    out["projection_note"] = (
        "measured on real Train + full Train/Valid/Test evaluation per epoch; the D-MPNN arm is "
        "measured separately; the total charges 4 context arms + B + G against the 2.5 A100.h cap"
    )
    out["budget"] = {
        "soft_cap_gpu_hours": C.GPU_SOFT_CAP_HOURS,
        "hard_cap_gpu_hours": C.GPU_HARD_CAP_HOURS,
    }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


SECTIONS = ("a", "b", "c", "d", "e", "f")


def run_sections(view: D.View, device, sections: Sequence[str], *, steps: int = 100, f_epochs: int = 2) -> dict[str, Any]:
    C.setup_determinism()
    results: dict[str, Any] = {}
    started = time.perf_counter()
    if "a" in sections:
        results["a"] = section_a(view)
    if "b" in sections:
        results["b"] = section_b(view)
    if "c" in sections:
        results["c"] = section_c(view, device, steps=steps)
    if "d" in sections:
        results["d"] = section_d(view, device)
    if "e" in sections:
        results["e"] = section_e(view, device)
    if "f" in sections:
        results["f"] = section_f(view, device, epochs=f_epochs)
    return {"device": str(device), "sections": results, "seconds": time.perf_counter() - started}


def collect_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    sections = payload.get("sections", {})
    a = sections.get("a")
    if a:
        checks["a_parameters"] = all(v["parameters_match"] for v in a["arms"].values())
        checks["a_zero_residual_equivalence"] = a["epoch0_equivalence_passed"]
        checks["a_shared_private_init"] = a["shared_new_init_across_context_arms"]
        checks["a_private_init_keeps_global_rng"] = a["private_init_does_not_advance_global_rng"]
        checks["a_independent_aggregation"] = a["independent_aggregation_passed"]
        checks["a_zero_bond_finite"] = a["zero_bond_passed"]
        checks["a_renumbering"] = a["renumbering_equivariant"]
        checks["a_bond_direction"] = a["bond_direction_invariant"]
        checks["a_molecule_isolation"] = a["molecule_isolation_passed"]
    b = sections.get("b")
    if b:
        checks["b_source_reach"] = bool(b["passed"])
    c = sections.get("c")
    if c:
        checks["c_channel_used"] = bool(c["channel_used"])
    d = sections.get("d")
    if d:
        checks["d_eval_does_not_disturb_training"] = bool(d["state_equal_with_inserted_eval"])
        checks["d_replay_deterministic"] = bool(d["replay_deterministic"])
    e = sections.get("e")
    if e:
        checks["e_determinism"] = bool(e["bitwise_repeatable"])
    f = sections.get("f")
    if f:
        checks["f_numpy_metric_recompute"] = all(
            v["numpy_recompute"][split]["rmse_abs_diff"] < 1e-9 and v["numpy_recompute"][split]["mae_abs_diff"] < 1e-9
            for v in f["per_arm"].values()
            for split in C.SPLITS
        )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LCRC lipophilicity context arms: targeted preflight")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--f-epochs", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    C.setup_determinism()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    view = D.build_view()
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    payload = run_sections(view, device, sections, steps=args.steps, f_epochs=args.f_epochs)
    payload["environment"] = C.env_info()
    payload["git_commit"] = C.git_commit()
    payload["git_dirty_tracked"] = C.git_dirty()
    payload["data"] = view.audit()
    checks = collect_checks(payload)
    payload["checks"] = checks
    payload["all_passed"] = all(checks.values())
    payload["sections_run"] = sections
    if args.out:
        C.write_json(Path(args.out), payload)
    print(json.dumps({"checks": checks, "all_passed": payload["all_passed"], "seconds": payload["seconds"]}, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
