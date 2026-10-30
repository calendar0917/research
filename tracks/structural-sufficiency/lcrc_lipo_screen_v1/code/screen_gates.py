"""CPU correctness gates for the LCRC Lipophilicity screen (pre-GPU).

Each gate returns a dict with ``status`` in {PASS, FAIL} and machine-readable
evidence.  ``run_all`` writes ``analysis/cpu_gates.json``.  No gate reads a
Monitor/Reserved label, runs a formal training recipe, or uses a GPU.

The gates are also exercised by ``tests/``; this module is the single source of
truth for the evidence JSON.
"""

from __future__ import annotations

import ast
import copy
import traceback
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import access_guard
import baseline_ecfp_xgb as BASE
import evaluate
import fit_labels
import monitor_labels
import screen_common as C
import screen_data as D
import screen_models as M
import screen_training as T


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class GateFailure(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _tensor_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False
        return bool(a.shape == b.shape and torch.equal(a, b))
    return bool(a == b)


def _small_batch(cache: Mapping[str, Any], n: int = 8) -> dict[str, Any]:
    rows = list(range(min(n, int(cache["n_molecules"]))))
    return D.collate(cache, rows), rows


def _synthetic_target(batch: Mapping[str, Any], scale: float = 1.0) -> torch.Tensor:
    n = int(batch["n_graphs"])
    return torch.linspace(-scale, scale, n, dtype=torch.float32)


def _step(model, optimizer, scheduler, batch, *, clip: float = C.GRAD_CLIP) -> float:
    target = _synthetic_target(batch)
    pred = model(batch)
    loss = F.mse_loss(pred.view(-1), target.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    optimizer.step()
    scheduler.step()
    return gnorm


def _state_copy(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# G0 frozen artifacts
# ---------------------------------------------------------------------------


def gate_frozen_artifacts() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    split = C.load_frozen_split()
    caches = D.load_caches(spec)
    evidence: dict[str, Any] = {
        "feature_spec": {
            "sha256": C.sha256_file(C.PREP_ANALYSIS / "feature_spec_lipo.json"),
            "hash": spec.hash(),
            "dims": [spec.atom_feat_dim, spec.bond_feat_dim, spec.global_dim, spec.n_scalars],
        },
        "norm": {
            "sha256": C.sha256_file(C.PREP_ANALYSIS / "norm_lipo.json"),
            "fingerprint": C.prep_data.norm_fingerprint(norm),
            "valid_or_test_used": norm["valid_or_test_used"],
            "global_constant_indices": norm["global_constant_indices"],
        },
        "split": {
            "sha256": C.sha256_file(C.LIPO_SPLIT),
            "row_hashes": split["hashes"],
            "sizes": {k: len(v) for k, v in split["split"].items()},
            "scope": split["payload"]["scope"],
        },
        "caches": {},
    }
    _require(spec.hash() == C.EXPECTED_FEATURE_SPEC_HASH, "feature spec hash")
    _require(spec.atom_feat_dim == 51 and spec.bond_feat_dim == 13, "schema widths")
    _require(C.prep_data.norm_fingerprint(norm) == C.EXPECTED_NORM_FINGERPRINT, "norm fingerprint")
    _require(norm["valid_or_test_used"] is False, "norm is not Fit-only")
    _require(C.sha256_file(C.LIPO_LABELS) == C.EXPECTED_SHA256["labels_fit_monitor"], "labels sha")
    for split_name, payload in caches.items():
        cache = payload["cache"]
        evidence["caches"][split_name] = {
            "file": C.track_rel(payload["path"]),
            "file_sha256": payload["file_sha256"],
            "content_fingerprint": payload["content_fingerprint"],
            "rebuilt_from_raw": payload["rebuilt"],
            "n_molecules": payload["n_molecules"],
            "labels_present": cache["meta"]["labels_present"],
            "feature_spec_hash": cache["meta"]["feature_spec_hash"],
        }
        _require(payload["content_fingerprint"] == C.EXPECTED_CACHE_FINGERPRINT[split_name], "cache fingerprint")
        _require(cache["meta"]["labels_present"] is False, "cache labels_present")
        _require(cache["meta"]["feature_spec_hash"] == C.EXPECTED_FEATURE_SPEC_HASH, "cache spec hash")
        ids = [str(x) for x in cache["molecule_ids"]]
        _require(ids == [str(int(r)) for r in payload["rows"]], "cache row order")
    _require("reserved" not in caches, "reserved cache must not be loaded")
    evidence["reserved_cache_loaded"] = False
    evidence["gpu_available"] = bool(torch.cuda.is_available())
    return {"status": "PASS", "evidence": evidence}


# ---------------------------------------------------------------------------
# G1 label alignment and isolation
# ---------------------------------------------------------------------------


def gate_label_alignment() -> dict[str, Any]:
    split_info = C.load_frozen_split()
    fit_rows = split_info["split"]["fit"]
    monitor_rows = split_info["split"]["monitor"]
    fit = fit_labels.load_fit_labels()
    audit = fit["audit"]
    _require(int(fit["y"].size) == C.SPLIT_SIZES["fit"], "Fit label count")
    _require(audit["label_values_constructed"] == C.SPLIT_SIZES["fit"], "constructed Fit values")
    _require(audit["label_values_skipped"] == C.SPLIT_SIZES["monitor"], "skipped Monitor values")
    _require(audit["unrequested_values_constructed"] == 0, "unrequested labels constructed")
    _require(audit["reserved_rows_in_artifact"] == 0, "reserved rows present")
    _require(audit["official_valid_test_rows_in_artifact"] == 0, "official rows present")
    _require(audit["labels_sha256"] == C.EXPECTED_SHA256["labels_fit_monitor"], "label artifact sha")
    recomputed_row_hash = C.prep_common.row_sha256(sorted(fit_rows + monitor_rows))
    _require(recomputed_row_hash == C.EXPECTED_LABELS_ROW_HASH, "label artifact row hash")
    _require(len(set(fit_rows) & set(monitor_rows)) == 0, "Fit/Monitor row overlap")

    caches = D.load_caches()
    _require([int(x) for x in caches["fit"]["cache"]["molecule_ids"]] == fit_rows, "Fit cache order")
    _require([int(x) for x in caches["monitor"]["cache"]["molecule_ids"]] == monitor_rows, "Monitor cache order")

    # The training batch has no label field, and injecting one does not leak.
    batch, _ = _small_batch(caches["fit"]["cache"], 8)
    label_like = [k for k in batch if k.lower() in {"y", "label", "labels", "target", "exp", "targets"}]
    _require(not label_like, f"batch contains label-like keys {label_like}")
    poisoned = dict(caches["fit"]["cache"])
    poisoned["labels"] = np.zeros(int(poisoned["n_molecules"]), dtype=np.float32)
    poisoned["exp"] = np.zeros(int(poisoned["n_molecules"]), dtype=np.float32)
    batch_poisoned = D.collate(poisoned, list(range(8)))
    _require(set(batch) == set(batch_poisoned), "collate keys changed after label injection")
    for key, value in batch.items():
        _require(_tensor_equal(value, batch_poisoned[key]), f"batch tensor {key} changed after label injection")

    # Monitor access is blocked before the prediction freeze.
    freeze_blocked = False
    try:
        monitor_labels.verify_prediction_freeze()
    except monitor_labels.FreezeError:
        freeze_blocked = True
    _require(freeze_blocked, "Monitor freeze gate did not block")
    _require(not C.PREDICTION_MANIFEST.exists(), "a prediction manifest exists before any model was trained")

    # Static import boundary: only evaluate.py may import monitor_labels.
    importers = []
    for path in sorted((C.EXP_DIR / "code").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        if "monitor_labels" in names:
            importers.append(path.name)
    allowed_importers = {"evaluate.py", "screen_gates.py"}
    _require(set(importers) <= allowed_importers, f"monitor_labels importers {importers}")
    training_modules = {"screen_training.py", "screen_data.py", "screen_models.py", "screen_predict.py", "baseline_ecfp_xgb.py", "run_screen.py"}
    _require(not (set(importers) & training_modules), "a training/prediction module imports monitor_labels")
    evidence = {
        "fit_rows": len(fit_rows),
        "monitor_rows": len(monitor_rows),
        "fit_label_vector_sha256": audit["fit_label_vector_sha256"],
        "label_values_constructed": audit["label_values_constructed"],
        "label_values_skipped": audit["label_values_skipped"],
        "unrequested_values_constructed": audit["unrequested_values_constructed"],
        "labels_sha256": audit["labels_sha256"],
        "row_hash_recomputed": recomputed_row_hash,
        "monitor_freeze_blocked": freeze_blocked,
        "monitor_labels_importers": importers,
        "batch_keys": sorted(batch),
        "label_injection_does_not_change_batch": True,
    }
    return {"status": "PASS", "evidence": evidence}


# ---------------------------------------------------------------------------
# G2 LCRC-A / S2-off
# ---------------------------------------------------------------------------


def gate_lcrc_s2() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    caches = D.load_caches(spec)
    batch, _rows = _small_batch(caches["fit"]["cache"], 8)
    evidence: dict[str, Any] = {"seeds": {}, "shared_inputs": {}, "perturbation": {}, "gradients": {}}

    for seed in C.SEEDS:
        a, info_a = M.build_model("A", spec, norm, seed, "cpu")
        b, info_b = M.build_model("B", spec, norm, seed, "cpu")
        _require(info_a["state_hash"] == info_b["state_hash"], f"seed {seed}: A/B init differs")
        _require(info_a["parameters"] == info_b["parameters"] == C.LCRC_PARAMS_REFERENCE, "parameter count")
        evidence["seeds"][f"seed_{seed}"] = {
            "state_hash": info_a["state_hash"],
            "parameters_A": info_a["parameters"],
            "parameters_B": info_b["parameters"],
            "effective_parameters_B": info_b["effective_parameters"],
            "init_identical": True,
        }

    a, info_a = M.build_model("A", spec, norm, 0, "cpu")
    b, info_b = M.build_model("B", spec, norm, 0, "cpu")
    controller = b._s2_controller  # type: ignore[attr-defined]
    _require(controller is not None, "arm B has no S2 controller")
    _require((controller.start, controller.end, controller.in_dim) == (320, 520, 529), "S2 slice offsets")
    _require(info_b["s2_off"]["active_train"] and info_b["s2_off"]["active_eval"], "S2-off not frozen on")

    with M.LocalMLPInputCapture(a) as cap_a, M.LocalMLPInputCapture(b) as cap_b:
        ya = a(batch)
        yb = b(batch)
    _require(len(cap_a.inputs) == len(cap_b.inputs) == 1, "LocalMLP input capture count")
    ia, ib = cap_a.inputs[0], cap_b.inputs[0]
    s2 = slice(controller.start, controller.end)
    _require(_tensor_equal(ib[:, s2], torch.zeros_like(ib[:, s2])), "B S2 block not exactly zero")
    _require(_tensor_equal(ia[:, : controller.start], ib[:, : controller.start]), "B changed pre-S2 input")
    _require(_tensor_equal(ia[:, controller.end :], ib[:, controller.end :]), "B changed post-S2 input")
    _require(float(ia[:, s2].abs().max()) > 0.0, "A S2 block is identically zero on this batch")
    evidence["shared_inputs"] = {
        "slice": [controller.start, controller.end],
        "pre_s2_bitwise_equal": True,
        "post_s2_bitwise_equal": True,
        "B_s2_max_abs": 0.0,
        "A_s2_max_abs": float(ia[:, s2].abs().max()),
    }

    perturbed = dict(batch)
    perturbed["path_cat"] = torch.roll(batch["path_cat"], 1, 0)
    perturbed["path_end63"] = torch.roll(batch["path_end63"], 1, 0)
    perturbed["path_end"] = torch.roll(batch["path_end"], 1, 0)
    perturbed["path_atom"] = torch.roll(batch["path_atom"], 1, 0)
    perturbed["path_closure"] = 1 - batch["path_closure"]
    with torch.no_grad():
        ya_p = a(perturbed)
        yb_p = b(perturbed)
    delta_a = float((ya_p - ya).abs().max())
    delta_b = float((yb_p - yb).abs().max())
    _require(delta_a > 1e-6, "A is invariant to S2 perturbation (S2 path not used)")
    _require(delta_b == 0.0, f"B changed under S2 perturbation by {delta_b}")
    evidence["perturbation"] = {
        "A_max_abs_output_change": delta_a,
        "B_max_abs_output_change": delta_b,
        "B_bitwise_invariant": True,
    }
    _require(controller.calls == 2, f"B forward hook fired {controller.calls} times for two forwards")

    target = _synthetic_target(batch)
    for label, model in (("A", a), ("B", b)):
        model.zero_grad(set_to_none=True)
        F.mse_loss(model(batch).view(-1), target.view(-1)).backward()
        params = dict(model.named_parameters())
        names = M.dead_parameter_names_lcrc(model, controller)
        grads = {name: float(params[name].grad.abs().sum()) if params[name].grad is not None else 0.0 for name in sorted(names)}
        first_w = model.local.mlp.lins[0].weight.grad
        s2_cols = float(first_w[:, controller.start : controller.end].abs().sum())
        live_cols = float(first_w[:, : controller.start].abs().sum())
        live_names = {
            "atom_encoder": float(model.atom_encoder.lins[0].weight.grad.abs().sum()),
            "bond_encoder": float(model.bond_encoder.lins[0].weight.grad.abs().sum()),
            "local.mlp": float(model.local.mlp.lins[1].weight.grad.abs().sum()),
            "f": float(model.f.lins[0].weight.grad.abs().sum()),
            "rho": float(model.rho.lins[0].weight.grad.abs().sum()),
        }
        evidence["gradients"][label] = {
            "dead_module_grad_abs_sum": grads,
            "first_layer_s2_column_grad_abs_sum": s2_cols,
            "first_layer_live_column_grad_abs_sum": live_cols,
            "live_module_grad_abs_sum": live_names,
        }
        _require(all(np.isfinite(v) for v in grads.values()), f"{label}: non-finite S2 gradient")
        if label == "A":
            _require(all(v > 0 for v in grads.values()), "A: some S2 parameter received no gradient")
            _require(s2_cols > 0, "A: first-layer S2 columns received no gradient")
        else:
            _require(all(v == 0.0 for v in grads.values()), "B: an S2 parameter received a non-zero gradient")
            _require(s2_cols == 0.0, "B: first-layer S2 columns received a gradient")
        _require(live_cols > 0 and all(v > 0 for v in live_names.values()), f"{label}: live path has no gradient")

    evidence["soup_safety"] = M.soup_safety_report(a)
    _require(evidence["soup_safety"]["all_state_entries_float"], "state dict is not soup-safe")
    return {"status": "PASS", "evidence": evidence}


# ---------------------------------------------------------------------------
# G3 GINE
# ---------------------------------------------------------------------------


class _NoOpConv(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr):  # noqa: D401 - test helper
        return torch.zeros_like(x)


def gate_gine() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    caches = D.load_caches(spec)
    model, info = M.build_model("C", spec, norm, 0, "cpu")
    evidence: dict[str, Any] = {
        "parameters": info["parameters"],
        "expected_parameters": C.GINE_PARAMS_REFERENCE,
        "hidden": info["hidden"],
        "layers": info["layers"],
    }
    _require(info["parameters"] == C.GINE_PARAMS_REFERENCE, f"GINE parameter count {info['parameters']}")
    _require(abs(info["parameters"] - C.LCRC_PARAMS_REFERENCE) / C.LCRC_PARAMS_REFERENCE < 0.05, "GINE parameter budget off by >5%")

    # Batch-composition isolation.
    rows = list(range(8))
    batch = D.collate(caches["fit"]["cache"], rows)
    with torch.no_grad():
        batched = model(batch)
        singles = []
        for r in rows:
            singles.append(float(model(D.collate(caches["fit"]["cache"], [r]))[0]))
    singles = np.asarray(singles)
    isolation = float(np.abs(singles - batched.numpy()).max())
    _require(isolation < 1e-5, f"GINE batch isolation {isolation}")
    evidence["batch_isolation_max_abs_diff"] = isolation

    # Single-molecule rewiring / edge-feature dependency.
    single_row = next(
        r
        for r in range(int(caches["fit"]["cache"]["n_molecules"]))
        if D.atom_counts(caches["fit"]["cache"])[r] >= 6
        and int(caches["fit"]["cache"]["bond_ptr"][r + 1] - caches["fit"]["cache"]["bond_ptr"][r]) >= 5
    )
    one = D.collate(caches["fit"]["cache"], [single_row])
    with torch.no_grad():
        y0 = model(one)[0].item()
        rewired = dict(one)
        bi = one["bond_index"].clone()
        u, v = int(bi[0, 0]), int(bi[0, 1])
        n_atoms = int(one["atom_feat"].shape[0])
        candidates = [w for w in range(n_atoms) if w not in (u, v)]
        _require(bool(candidates), "no rewiring candidate")
        bi[0, 1] = candidates[-1]
        rewired["bond_index"] = bi
        y_rewire = model(rewired)[0].item()
        edge_pert = dict(one)
        bf = one["bond_feat"].clone()
        bf[0] = bf[0] + 0.5
        edge_pert["bond_feat"] = bf
        y_edge = model(edge_pert)[0].item()
        no_mp = copy.deepcopy(model)
        for i in range(len(no_mp.convs)):
            no_mp.convs[i] = _NoOpConv()
        y_no_mp = no_mp(one)[0].item()
        # Permutation invariance.
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(n_atoms, generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n_atoms)
        permuted = dict(one)
        permuted["atom_feat"] = one["atom_feat"][perm]
        permuted["bond_index"] = inv[one["bond_index"]]
        y_perm = model(permuted)[0].item()
    evidence["rewiring"] = {
        "row": int(single_row),
        "base_prediction": y0,
        "rewired_prediction": y_rewire,
        "max_abs_change": abs(y_rewire - y0),
        "edge_feature_perturbation_change": abs(y_edge - y0),
        "no_message_passing_change": abs(y_no_mp - y0),
        "permutation_invariance_change": abs(y_perm - y0),
    }
    _require(abs(y_rewire - y0) > 1e-4, "GINE ignores edge rewiring")
    _require(abs(y_edge - y0) > 1e-4, "GINE ignores bond features")
    _require(abs(y_no_mp - y0) > 1e-3, "GINE message passing has no effect")
    _require(abs(y_perm - y0) < 1e-4, f"GINE is not permutation invariant ({abs(y_perm - y0)})")

    # Messages actually travel along the physical adjacency: a 2-molecule batch
    # gives each molecule its own prediction regardless of the other molecule.
    model.zero_grad(set_to_none=True)
    target = _synthetic_target(batch)
    F.mse_loss(model(batch).view(-1), target.view(-1)).backward()
    grads = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in model.named_parameters()}
    _require(all(np.isfinite(v) for v in grads.values()), "GINE non-finite gradient")
    _require(all(v > 0 for v in grads.values()), "GINE parameter without gradient")
    evidence["gradient_abs_sum_min"] = min(grads.values())
    evidence["gradient_abs_sum_max"] = max(grads.values())
    evidence["soup_safety"] = M.soup_safety_report(model)
    _require(evidence["soup_safety"]["all_state_entries_float"], "GINE state dict is not soup-safe")
    return {"status": "PASS", "evidence": evidence}


# ---------------------------------------------------------------------------
# G4 training numerics, batch order, soup
# ---------------------------------------------------------------------------


def gate_training_numerics() -> dict[str, Any]:
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    caches = D.load_caches(spec)
    evidence: dict[str, Any] = {}
    _require(C.STEPS_PER_EPOCH == 19, "steps per epoch")
    _require(C.TOTAL_UPDATES == 6840, "total updates")
    hash1 = D.batch_order_hash(0)
    hash2 = D.batch_order_hash(0)
    _require(hash1 == hash2, "batch order hash is not deterministic")
    first_epoch = D.epoch_batches(0, 1)
    replay = torch.Generator().manual_seed(C.TRAIN_SHUFFLE_OFFSET)
    expected_first = D.shuffle_batches(C.SPLIT_SIZES["fit"], C.BATCH_SIZE, replay)
    _require(first_epoch == expected_first, "epoch_batches does not match the generator stream")
    _require(len(expected_first) == 19 and len(expected_first[-1]) == 48, "batch shapes")
    evidence["batch_order"] = {
        "hash": hash1,
        "deterministic": True,
        "steps_per_epoch": len(expected_first),
        "last_batch_size": len(expected_first[-1]),
        "total_updates": C.TOTAL_UPDATES,
    }

    # One optimizer step + save/restore reproduces the next step bitwise.
    model, _ = M.build_model("A", spec, norm, 0, "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.TOTAL_UPDATES, eta_min=C.LR_MIN)
    generator = torch.Generator().manual_seed(C.TRAIN_SHUFFLE_OFFSET)
    b1 = D.collate(caches["fit"]["cache"], expected_first[0])
    b2 = D.collate(caches["fit"]["cache"], expected_first[1])
    _step(model, optimizer, scheduler, b1)
    state_path = C.RUNS / "gate_state.pt"
    snap = T.save_training_state(state_path, model, optimizer, scheduler, epoch=1, generator=generator)
    _step(model, optimizer, scheduler, b2)
    live = _state_copy(model)

    model2, _ = M.build_model("A", spec, norm, 0, "cpu")
    opt2 = torch.optim.AdamW(model2.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=C.TOTAL_UPDATES, eta_min=C.LR_MIN)
    T.load_training_state(state_path, model2, opt2, sch2)
    _step(model2, opt2, sch2, b2)
    restored = _state_copy(model2)
    diffs = {k: float((live[k] - restored[k]).abs().max()) for k in live}
    _require(max(diffs.values()) == 0.0, f"save/restore did not reproduce the next step: {max(diffs.values())}")
    evidence["save_restore"] = {"max_abs_param_diff": 0.0, "checkpoint_sha256": snap["sha256"], "epoch": 1}

    # Parameter averaging.
    torch.manual_seed(11)
    states = [{k: v + torch.randn_like(v) * 0.01 for k, v in _state_copy(model).items()} for _ in range(C.SOUP_LAST)]
    soup = M.average_states(states)
    for key in soup:
        manual = torch.stack([s[key] for s in states]).mean(0)
        _require(_tensor_equal(soup[key], manual), f"soup mismatch at {key}")
    same = M.average_states([_state_copy(model) for _ in range(C.SOUP_LAST)])
    identity_diff = max(float((same[k] - model.state_dict()[k].float()).abs().max()) for k in same)
    _require(identity_diff < 1e-6, f"soup of identical states changed the model by {identity_diff}")
    evidence["soup"] = {
        "members": C.SOUP_LAST,
        "rule": "float mean of last 5 epoch-end state dicts",
        "matches_manual_mean": True,
        "identical_states_max_abs_diff": identity_diff,
        "mismatch_count": 0,
    }

    # Formal recipe is immutable but the authorization gate blocks GPU.
    formal = T.TrainConfig(arm="A", seed=0, device="cuda")
    formal.validate()
    overridden = T.TrainConfig(arm="A", seed=0, device="cuda", epochs=120)
    blocked = False
    try:
        overridden.validate()
    except ValueError:
        blocked = True
    _require(blocked, "formal epochs override was accepted")
    import run_screen

    gpu_blocked = False
    try:
        run_screen.require_gpu_authorization()
    except PermissionError:
        gpu_blocked = True
    _require(gpu_blocked, "GPU training is not blocked without an authorization file")
    _require(C.load_config()["budget"]["hard_cap_gpu_hours"] <= 1.5, "GPU cap above the frozen hard cap")
    evidence["recipe_guard"] = {
        "formal_validation_passes": True,
        "epochs_override_rejected": blocked,
        "gpu_authorization_required": gpu_blocked,
        "gpu_authorized": False,
    }
    return {"status": "PASS", "evidence": evidence}


# ---------------------------------------------------------------------------
# G5 CPU smoke: loss must actually decrease on a small Fit subset
# ---------------------------------------------------------------------------


def gate_smoke() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in C.NN_ARMS:
        cfg = T.TrainConfig(arm=arm, seed=0, device="cpu", smoke=True, fit_limit=256, epochs=12)
        payload = T.train_one(cfg)
        curve = payload["curve"]
        first = curve[0]["train_mse_normalised"]
        last = curve[-1]["train_mse_normalised"]
        best = min(c["train_mse_normalised"] for c in curve)
        ratio = last / first if first > 0 else float("inf")
        _require(np.isfinite(last) and np.isfinite(best), f"{arm}: non-finite smoke loss")
        _require(ratio < 0.8, f"{arm}: smoke train MSE did not drop enough ({first} -> {last})")
        _require(payload["monitor_labels_read"] is False, f"{arm}: smoke touched Monitor labels")
        _require(payload["recipe"]["total_updates"] == 24, f"{arm}: smoke update count")
        summary[arm] = {
            "run_id": payload["run_id"],
            "fit_limit": 256,
            "epochs": 12,
            "updates": payload["recipe"]["total_updates"],
            "train_mse_first": first,
            "train_mse_last": last,
            "train_mse_best": best,
            "last_over_first": ratio,
            "init_state_hash": payload["init_state_hash"],
            "soup_state_hash": payload["soup_state_hash"],
            "soup_members": payload["recipe"]["soup_members"],
            "batch_order_matches": payload["batch_order"]["matches"],
            "batch_order_hash": payload["batch_order"]["actual_hash"],
            "grad_norm_max": payload["anomalies"]["grad_norm_max"],
            "clip_events": payload["anomalies"]["grad_clip_events"],
            "nonfinite_events": payload["anomalies"].get("nonfinite_events"),
            "fit_diagnostics": payload["fit_diagnostics"],
            "wall_seconds": payload["wall_seconds"],
            "monitor_labels_read": payload["monitor_labels_read"],
            "recipe": payload["recipe"],
            "curve": curve,
        }
    C.write_json(C.ANALYSIS / "smoke.json", {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": "engineering smoke on 256 Fit molecules, 12 epochs, CPU only; not a scientific result",
        "gpu_used": False,
        "monitor_scores": None,
        "monitor_labels_read": False,
        "fit_limit": 256,
        "epochs": 12,
        "note": "three independent smoke runs, one per neural arm; losses are Fit-only and cannot rank arms scientifically",
        "arms": summary,
    })
    return {"status": "PASS", "evidence": summary}


# ---------------------------------------------------------------------------
# G6 baseline D
# ---------------------------------------------------------------------------


def gate_baseline() -> dict[str, Any]:
    cfg = C.load_config()
    split = C.load_frozen_split()
    fit_rows = [int(r) for r in split["split"]["fit"]]
    monitor_rows = [int(r) for r in split["split"]["monitor"]]
    fp_a = BASE.fingerprint_rows(fit_rows, cfg)
    fp_b = BASE.fingerprint_rows(fit_rows, cfg)
    _require(fp_a["feature_sha256"] == fp_b["feature_sha256"], "ECFP fingerprint is not deterministic")
    fp_m = BASE.fingerprint_rows(monitor_rows, cfg)
    _require(fp_a["X"].shape == (len(fit_rows), 2048), "Fit fingerprint shape")
    _require(fp_m["X"].shape == (len(monitor_rows), 2048), "Monitor fingerprint shape")
    _require(int(fp_a["X"].sum()) > 0 and int(fp_m["X"].sum()) > 0, "empty fingerprint")
    det = BASE.determinism_check(cfg)
    _require(det["max_abs_diff"] == 0.0, f"XGBoost re-fit is not deterministic: {det['max_abs_diff']}")
    _require(monitor_rows and all(0 <= r < 4200 for r in monitor_rows), "Monitor row range")
    return {
        "status": "PASS",
        "evidence": {
            "fingerprint": fp_a["spec"],
            "fit_feature_sha256": fp_a["feature_sha256"],
            "monitor_feature_sha256": fp_m["feature_sha256"],
            "fit_shape": list(fp_a["X"].shape),
            "monitor_shape": list(fp_m["X"].shape),
            "canonical_smiles_rows_verified": len(fit_rows) + len(monitor_rows),
            "xgb_determinism": det,
            "hyperparameter_search": False,
            "monitor_labels_read": False,
        },
    }


# ---------------------------------------------------------------------------
# G7 evaluation order and metric units
# ---------------------------------------------------------------------------


def gate_evaluation() -> dict[str, Any]:
    y = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pred = np.asarray([1.0, -1.0, 2.0, 0.0], dtype=np.float64)
    _require(abs(evaluate.rmse(pred, y) - np.sqrt(6 / 4)) < 1e-12, "rmse definition")
    _require(abs(evaluate.mae(pred, y) - 1.0) < 1e-12, "mae definition")
    ci = evaluate.bootstrap_ci_mean(np.asarray([1.0, 2.0, 3.0, 4.0]), 500, 0)
    _require(ci["ci95_low"] <= ci["estimate"] <= ci["ci95_high"], "bootstrap CI ordering")
    delta = evaluate.bootstrap_delta_rmse(pred, np.zeros(4), y, 500, 0)
    _require(delta["estimate"] > 0, "delta RMSE sign")

    # Frozen classification branches.
    _require(evaluate.classify_comparison(-0.10, -0.15, -0.05, 0.03) == "A_MATERIALLY_BETTER", "classify better")
    _require(evaluate.classify_comparison(-0.01, -0.05, 0.02, 0.03) == "COMPETITIVE", "classify competitive")
    _require(evaluate.classify_comparison(0.10, 0.05, 0.15, 0.03) == "A_MATERIALLY_WORSE", "classify worse")
    _require(evaluate.classify_comparison(0.02, -0.02, 0.06, 0.03) == "INCONCLUSIVE", "classify inconclusive")
    _require(evaluate.classify_s2([-0.05, -0.04, -0.06], -0.05, -0.01, 0.03) == "CLEAR_INCREMENT", "classify S2 clear")
    _require(evaluate.classify_s2([-0.01, -0.02, -0.005], -0.012, -0.001, 0.03) == "SUGGESTIVE_NOT_MATERIAL", "classify S2 suggestive")
    _require(evaluate.classify_s2([-0.01, 0.01, -0.005], -0.002, 0.01, 0.03) == "NO_CLEAR_INCREMENT", "classify S2 none")
    _require(
        evaluate.classify_case(True, "CLEAR_INCREMENT", "COMPETITIVE", "COMPETITIVE") == "CASE_1_S2_INCREMENT_SUPPORTED",
        "classify case 1",
    )
    _require(
        evaluate.classify_case(True, "NO_CLEAR_INCREMENT", "COMPETITIVE", "COMPETITIVE")
        == "CASE_2_LCRC_COMPETITIVE_NO_CLEAR_S2_INCREMENT",
        "classify case 2",
    )
    _require(
        evaluate.classify_case(False, "CLEAR_INCREMENT", "COMPETITIVE", "COMPETITIVE")
        == "CASE_3_A_WEAKER_OR_DID_NOT_LEARN",
        "classify case 3 (did not learn)",
    )

    # Evaluation cannot run before the prediction freeze.
    blocked = False
    try:
        evaluate.evaluate()
    except monitor_labels.FreezeError:
        blocked = True
    _require(blocked, "evaluate ran before the prediction freeze")
    _require(not C.PREDICTION_MANIFEST.exists(), "prediction manifest exists before GPU runs")
    partial = sorted(p.name for p in C.PREDICTIONS.glob("*.npz"))
    for path in sorted(C.PREDICTIONS.glob("*.json")):
        sidecar = C.read_json(path)
        _require(sidecar.get("monitor_labels_read") is False, f"{path.name}: not certified label-blind")
    return {
        "status": "PASS",
        "evidence": {
            "rmse_definition": "sqrt(mean((pred-y)^2))",
            "mae_definition": "mean(|pred-y|)",
            "bootstrap": "percentile, molecule resampling",
            "classification_branches_tested": 10,
            "evaluate_blocked_before_freeze": blocked,
            "prediction_manifest_present": False,
            "partial_prediction_files": partial,
            "partial_predictions_label_blind": True,
        },
    }


# ---------------------------------------------------------------------------
# G8 runtime access control
# ---------------------------------------------------------------------------


def gate_access_control() -> dict[str, Any]:
    access_guard.arm()
    blocked = {}
    for name, path in (
        ("csv_with_exp_column", C.LIPO_CSV),
        ("official_split_with_train_y", C.LIPO_OFFICIAL_SPLIT),
        ("reserved_cache", C.PREP_CACHE / "lipo_reserved.pt"),
        ("labels_without_grant", C.LIPO_LABELS),
    ):
        try:
            with open(path, "rb") as fh:
                fh.read(1)
            blocked[name] = False
        except PermissionError:
            blocked[name] = True
    _require(all(blocked.values()), f"access guard failed: {blocked}")
    access_guard.grant_once(C.LIPO_LABELS)
    ok = False
    try:
        with open(C.LIPO_LABELS, "rb") as fh:
            fh.read(1)
        ok = True
    except PermissionError:
        ok = False
    _require(ok, "one-shot label grant did not work")
    second_blocked = False
    try:
        with open(C.LIPO_LABELS, "rb") as fh:
            fh.read(1)
    except PermissionError:
        second_blocked = True
    _require(second_blocked, "label grant was not one-shot")
    status = access_guard.guard_status()
    access_guard.disarm()
    return {
        "status": "PASS",
        "evidence": {
            "forbidden_open_attempts": blocked,
            "label_grant_works_once": ok,
            "second_open_blocked": second_blocked,
            "guard": status,
            "no_label_values_constructed": True,
        },
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


GATES: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
    ("G0", "frozen_artifacts", gate_frozen_artifacts),
    ("G1", "label_alignment_and_isolation", gate_label_alignment),
    ("G2", "lcrc_s2_ablation", gate_lcrc_s2),
    ("G3", "gine_adjacency", gate_gine),
    ("G4", "training_numerics_and_soup", gate_training_numerics),
    ("G5", "cpu_smoke_loss_decrease", gate_smoke),
    ("G6", "baseline_ecfp_xgb", gate_baseline),
    ("G7", "evaluation_order_and_metrics", gate_evaluation),
    ("G8", "runtime_access_control", gate_access_control),
]


def run_all() -> dict[str, Any]:
    import time

    started = time.perf_counter()
    results: dict[str, Any] = {}
    for gate_id, name, fn in GATES:
        t0 = time.perf_counter()
        try:
            out = fn()
            results[gate_id] = {
                "id": gate_id,
                "name": name,
                "status": out.get("status", "PASS"),
                "seconds": time.perf_counter() - t0,
                "evidence": out.get("evidence", {}),
            }
        except Exception as exc:  # noqa: BLE001 - a gate failure must be recorded, not raised
            results[gate_id] = {
                "id": gate_id,
                "name": name,
                "status": "FAIL",
                "seconds": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        C.log(f"[gate {gate_id}] {results[gate_id]['status']} {name} ({results[gate_id]['seconds']:.2f}s)")
    final = "READY_FOR_GPU_AUTHORIZATION" if all(r["status"] == "PASS" for r in results.values()) else "BLOCKED"
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "gates": results,
        "final_status": final,
        "gpu_authorized": False,
        "gpu_used": False,
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
        "total_seconds": time.perf_counter() - started,
        "limitations": [
            "CPU gates validate implementation and data discipline, not predictive performance.",
            "No formal training run has been executed in this round.",
            "GINE is a practical performance reference, not an information-matched control.",
            "Arm B has equal nominal but lower effective parameter count than arm A.",
        ],
    }
    C.write_json(C.ANALYSIS / "cpu_gates.json", payload)
    return payload


__all__ = ["GATES", "run_all"]
