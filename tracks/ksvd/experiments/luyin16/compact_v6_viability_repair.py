"""Compact-v6 attribute-branch viability repair: diagnostics + gates.

This module is *narrow*.  It does not redesign compact-v6.  It answers one
question: why did the original ``factorized_role`` attribute branch collapse,
and can the *same* branch be made demonstrably active with a minimal fix?

Stages (``--stage``):

* ``original_trace``  -- single real training batch, forward stats for every
  attribute-branch layer, then ``loss.backward()`` and per-parameter gradient
  norms.  Writes ``original_single_batch_trace.json``.
* ``optimizer_audit`` -- every trainable parameter vs ``optimizer.param_groups``.
  Writes ``optimizer_parameter_audit.csv``.
* ``original_10step`` -- 10 optimizer steps on a fixed small subset, recording
  loss / branch norm / per-layer gradient norms.  Writes
  ``original_10step_trace.csv``.
* ``input_variance``  -- raw attribute representation uniqueness and an
  adversarial placement pair check.  Writes ``attribute_input_variance.json``
  and ``adversarial_representation_check.json``.
* ``repaired_100step`` -- 100 optimizer steps with the repaired initialisation,
  records ``repaired_100step_trace.csv`` and runs gates V1-V4 into
  ``repaired_viability_gate.json``.

The official test split is never loaded anywhere in this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as pp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.compact_v6_diagnostics import (
    _build_model,
    _load_config,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
)

DEFAULT_FACTORIZED_RUN = (
    REPO_ROOT / "tracks/ksvd/runs/2026/09/10/20260910-162801-fa8a41cf"
)
DEFAULT_OUTPUT = REPO_ROOT / "tracks/ksvd/results/compact_v6_viability"
CACHE_DIR = REPO_ROOT / "tracks/ksvd/results/compact_v6_viability/.cache"

#: Pre-registered repaired initialisation scale for the final fusion
#: projection (the "small nonzero" Fix 1).  Not swept.
REPAIRED_INIT_SCALE = 0.01


# --------------------------------------------------------------------------- #
# data extraction with an on-disk cache (extraction dominates wall-clock)
# --------------------------------------------------------------------------- #
def _extract_records(config: Mapping[str, Any]):
    data_root = pp._resolve(config["data"]["root"])
    representation = config.get("representation", {})
    model_config = config["model"]
    patch_radius = int(representation.get("patch_radius", pp.PATCH_RADIUS))
    context_radius = int(representation.get("context_radius", 0))
    tokenizer_version = pp.resolve_typed_tokenizer_version(
        representation.get("typed_tokenizer_version")
    )
    structural_mode = str(model_config.get("structural_context_mode", "none"))
    topology_mode = str(model_config.get("topology_mode", "none"))
    attribute_mode = str(model_config.get("attribute_mode", "none"))
    max_cycle_len = int(representation.get("structural_context_max_cycle_len", 10))
    certificate_cache: dict[bytes, bytes] = {}
    records: list[list[pp.GraphRecord]] = []
    for split in ("train", "val"):
        dataset = _load_zinc(data_root, split)
        matrix = None
        if topology_mode != "none":
            matrix, _frame, _meta = ztopo.matrices_for_split(
                "valid" if split == "val" else split,
                dataset,
                topology_mode,
                force=False,
                input_width=(
                    int(model_config.get("topology_input_width"))
                    if topology_mode == "capacity_control"
                    else None
                ),
            )
        split_records, _meta = pp._extract_split(
            dataset,
            split,
            certificate_cache,
            patch_radius=patch_radius,
            context_radius=context_radius,
            structural_mode=structural_mode,
            max_cycle_len=max_cycle_len,
            topology_mode=topology_mode,
            topology_matrix=matrix,
            tokenizer_version=tokenizer_version,
            attribute_mode=attribute_mode,
        )
        records.append(split_records)
    return records[0], records[1], patch_radius, context_radius


def _cache_key(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {
            "root": config["data"]["root"],
            "representation": config.get("representation", {}),
            "topology_mode": config["model"].get("topology_mode"),
            "structural_context_mode": config["model"].get("structural_context_mode"),
            "attribute_mode": config["model"].get("attribute_mode"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_records(config: Mapping[str, Any], *, use_cache: bool = True):
    """Return ``(train_records, valid_records, patch_radius, context_radius)``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"records_{_cache_key(config)}.pkl"
    if use_cache and path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    result = _extract_records(config)
    with path.open("wb") as handle:
        pickle.dump(result, handle)
    return result


def build_phase_data(config: Mapping[str, Any], *, use_cache: bool = True):
    """Encoded train/valid ``Data`` lists (train-fit vocab/standardizers)."""
    train_records, valid_records, patch_radius, context_radius = load_records(
        config, use_cache=use_cache
    )
    fit_data, valid_data, audit = pp._phase_data(
        train_records, valid_records, config=config
    )
    return fit_data, valid_data, audit, patch_radius, context_radius


def build_fresh_model(config, audit, patch_radius, context_radius, seed=0, **kwargs):
    pp._seed_everything(int(seed))
    model = _build_model(config, audit, patch_radius, context_radius, **kwargs)
    return model


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def _stat(name: str, value: torch.Tensor | np.ndarray | None) -> dict[str, Any]:
    if value is None:
        return {"name": name, "present": False}
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.asarray(value))
    else:
        tensor = value.detach()
    flat = tensor.reshape(-1).to(torch.float64)
    finite = torch.isfinite(flat)
    n = int(flat.numel())
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {
            "name": name,
            "present": True,
            "shape": list(tensor.shape),
            "numel": n,
            "fraction_finite": 0.0,
        }
    finite_values = flat[finite]
    return {
        "name": name,
        "present": True,
        "shape": list(tensor.shape),
        "numel": n,
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std(unbiased=False)) if n_finite > 1 else 0.0,
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "fraction_zero": float((finite_values == 0).to(torch.float64).mean()),
        "fraction_finite": float(n_finite / n),
    }


def _classify_grad(tensor: torch.Tensor | None, param: torch.Tensor | None = None) -> dict:
    if tensor is None:
        return {
            "grad_state": "None",
            "grad_norm": None,
            "param_norm": (
                float(param.detach().to(torch.float64).norm())
                if param is not None
                else None
            ),
            "grad_param_ratio": None,
        }
    grad_norm = float(tensor.detach().to(torch.float64).norm())
    param_norm = (
        float(param.detach().to(torch.float64).norm()) if param is not None else None
    )
    ratio = (
        grad_norm / param_norm
        if param_norm is not None and param_norm > 0
        else None
    )
    if grad_norm == 0.0:
        state = "exactly_zero"
    elif grad_norm < 1e-12:
        state = "numerically_tiny"
    else:
        state = "nonzero"
    return {
        "grad_state": state,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "grad_param_ratio": ratio,
    }


# --------------------------------------------------------------------------- #
# stage: original single-batch forward/gradient trace
# --------------------------------------------------------------------------- #
def stage_original_trace(
    config,
    fit_data,
    audit,
    patch_radius,
    context_radius,
    *,
    output_dir: Path,
    seed: int = 0,
    batch_size: int = 128,
) -> dict:
    model = build_fresh_model(config, audit, patch_radius, context_radius, seed=seed)
    model.eval()  # deterministic forward (no dropout) for the trace
    loader = pp._make_loader(fit_data, batch_size, True, seed + 91011)
    batch = next(iter(loader))
    batch = batch.to(torch.device("cpu"))

    captures: dict[str, Any] = {}
    handles: list[Any] = []
    if model.attribute_encoder is not None:
        enc = model.attribute_encoder

        def grab(tag: str, *, use_input: bool = False):
            def hook(module, inputs, output):
                captures[tag] = (
                    (inputs, output) if use_input else output
                )
            return hook

        handles.append(enc.atom_embedding.register_forward_hook(grab("atom_type_embedding")))
        handles.append(enc.bond_embedding.register_forward_hook(grab("bond_type_embedding")))
        handles.append(enc.atom_mlp.register_forward_hook(grab("atom_encoder_output")))
        handles.append(enc.bond_mlp.register_forward_hook(grab("bond_encoder_output")))
        handles.append(enc.atom_mlp[0].register_forward_hook(grab("atom_mlp_first_input", use_input=True)))
        handles.append(enc.bond_mlp[0].register_forward_hook(grab("bond_mlp_first_input", use_input=True)))
        handles.append(enc.fusion.register_forward_hook(grab("e_attribute")))
        handles.append(enc.fusion[0].register_forward_hook(grab("attr_fusion_hidden")))
        # keep the final e_attribute tensor for gradient probing
        frozen_e_attr: dict[str, torch.Tensor] = {}

        def e_attr_hook(module, inputs, output):
            frozen_e_attr["value"] = output
            captures["attr_fusion_input"] = inputs[0]
        handles.append(enc.fusion.register_forward_hook(e_attr_hook))

    # patch encoder first linear output
    def patch_first_hook(module, inputs, output):
        captures["concat_patch_input"] = inputs[0]
        captures["patch_encoder_first_output"] = output

    handles.append(model.patch_encoder.layers[0].register_forward_hook(patch_first_hook))

    prediction = model(batch)
    target = batch.y.view(-1)
    loss = torch.nn.functional.l1_loss(prediction, target)

    e_attr_tensor = frozen_e_attr.get("value")
    e_attr_grad: dict[str, Any] = {}
    if e_attr_tensor is not None:
        def _egrad(grad):
            e_attr_grad["grad"] = grad.detach()
            return grad
        e_attr_tensor.register_hook(_egrad)

    model.zero_grad(set_to_none=True)
    loss.backward()

    for handle in handles:
        handle.remove()

    forward_stats: list[dict[str, Any]] = []
    raw_atom_role = batch.attribute_atom_role
    raw_bond_role = torch.cat(
        [batch.attribute_bond_role_left, batch.attribute_bond_role_right], dim=1
    )
    forward_stats.append(_stat("01_raw_atom_role_features", raw_atom_role))
    forward_stats.append(_stat("02_raw_bond_role_features", raw_bond_role))
    forward_stats.append(_stat("03_atom_type_embeddings", captures.get("atom_type_embedding")))
    forward_stats.append(_stat("04_bond_type_embeddings", captures.get("bond_type_embedding")))
    forward_stats.append(_stat("05_atom_encoder_output", captures.get("atom_encoder_output")))
    forward_stats.append(_stat("06_bond_encoder_output", captures.get("bond_encoder_output")))
    # pooled / fusion-input are per patch (mean over patch occurrences)
    if e_attr_tensor is not None:
        fusion_input = captures.get("attr_fusion_input")
        forward_stats.append(_stat("07_atom_pooled_representation", None if fusion_input is None else fusion_input[:, : fusion_input.shape[1] // 2]))
        forward_stats.append(_stat("08_bond_pooled_representation", None if fusion_input is None else fusion_input[:, fusion_input.shape[1] // 2:]))
        forward_stats.append(_stat("09_attr_fusion_input", fusion_input))
    else:
        for label in ("07_atom_pooled_representation", "08_bond_pooled_representation", "09_attr_fusion_input"):
            forward_stats.append(_stat(label, None))
    forward_stats.append(_stat("10_attr_fusion_hidden", captures.get("attr_fusion_hidden")))
    forward_stats.append(_stat("11_final_e_attribute", e_attr_tensor))
    forward_stats.append(_stat("12_concatenated_patch_input", captures.get("concat_patch_input")))
    forward_stats.append(_stat("13_patch_encoder_first_output", captures.get("patch_encoder_first_output")))

    gradients: list[dict[str, Any]] = []
    named = dict(model.named_parameters())
    tracked_patterns = (
        "attribute_encoder",
        "patch_encoder",
    )
    for name, param in named.items():
        if not any(name.startswith(p) for p in tracked_patterns):
            continue
        entry = {"name": name, **_classify_grad(param.grad, param)}
        entry["shape"] = list(param.shape)
        gradients.append(entry)

    # dedicated upstream / W_attr probes
    special: dict[str, Any] = {}
    if model.attribute_encoder is not None:
        enc = model.attribute_encoder
        special["fusion_final_weight"] = {
            "name": "attribute_encoder.fusion.2.weight",
            **_classify_grad(model.attribute_encoder.fusion[-1].weight.grad, model.attribute_encoder.fusion[-1].weight),
        }
        special["fusion_final_bias"] = {
            "name": "attribute_encoder.fusion.2.bias",
            **_classify_grad(model.attribute_encoder.fusion[-1].bias.grad, model.attribute_encoder.fusion[-1].bias),
        }
        special["grad_e_attribute"] = _stat(
            "grad_e_attribute", e_attr_grad.get("grad")
        )
        # patch encoder columns that consume the 8D e_attribute:
        first_weight = model.patch_encoder.layers[0].weight  # [hidden, in]
        in_width = int(first_weight.shape[1])
        attr_cols = slice(in_width - int(model.attribute_input_width), in_width)
        w_attr = first_weight[:, attr_cols]
        g_attr = None if first_weight.grad is None else first_weight.grad[:, attr_cols]
        special["patch_encoder_attribute_columns"] = {
            "init_mean": float(w_attr.detach().mean()),
            "init_std": float(w_attr.detach().std(unbiased=False)),
            "init_norm": float(w_attr.detach().to(torch.float64).norm()),
            "grad_norm": None if g_attr is None else float(g_attr.detach().to(torch.float64).norm()),
            "grad_state": (
                "None"
                if g_attr is None
                else ("exactly_zero" if float(g_attr.detach().to(torch.float64).norm()) == 0 else "nonzero")
            ),
        }
        special["grad_upstream_encoder"] = {
            "atom_mlp_grad_norm": float(
                torch.cat(
                    [p.grad.reshape(-1) for p in enc.atom_mlp.parameters() if p.grad is not None]
                ).norm()
            ) if any(p.grad is not None for p in enc.atom_mlp.parameters()) else None,
            "bond_mlp_grad_norm": float(
                torch.cat(
                    [p.grad.reshape(-1) for p in enc.bond_mlp.parameters() if p.grad is not None]
                ).norm()
            ) if any(p.grad is not None for p in enc.bond_mlp.parameters()) else None,
        }

    result = {
        "stage": "original_single_batch_trace",
        "seed": int(seed),
        "batch_molecules": int(target.numel()),
        "loss_l1": float(loss.detach()),
        "prediction_mean": float(prediction.detach().mean()),
        "prediction_std": float(prediction.detach().std(unbiased=False)),
        "forward_stats": forward_stats,
        "gradients": gradients,
        "special_gradients": special,
        "e_attribute_norm": (
            None
            if e_attr_tensor is None
            else float(e_attr_tensor.detach().norm(dim=1).mean())
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "original_single_batch_trace.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


# --------------------------------------------------------------------------- #
# stage: optimizer parameter audit
# --------------------------------------------------------------------------- #
def stage_optimizer_audit(
    config,
    audit,
    model: pp.PatchPathModel | None = None,
    *,
    output_dir: Path,
    seed: int = 0,
    patch_radius: int | None = None,
    context_radius: int | None = None,
) -> dict:
    if model is None:
        model = build_fresh_model(config, audit, patch_radius, context_radius, seed=seed)
    model_config = config["model"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    group_index = {
        id(parameter): index
        for index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        gidx = group_index.get(id(parameter))
        group = optimizer.param_groups[gidx] if gidx is not None else None
        rows.append(
            {
                "parameter_name": name,
                "parameter_id": id(parameter),
                "shape": "x".join(str(d) for d in parameter.shape),
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "in_optimizer": gidx is not None,
                "optimizer_group": gidx,
                "weight_decay": None if group is None else float(group["weight_decay"]),
                "learning_rate": None if group is None else float(group["lr"]),
                "is_attribute_branch": name.startswith("attribute_encoder"),
            }
        )
    attribute_rows = [r for r in rows if r["is_attribute_branch"]]
    missing = [
        r["parameter_name"]
        for r in attribute_rows
        if (not r["in_optimizer"]) or (not r["requires_grad"])
    ]
    result = {
        "num_parameters": len(rows),
        "num_trainable": int(sum(1 for r in rows if r["requires_grad"])),
        "num_in_optimizer": int(sum(1 for r in rows if r["in_optimizer"])),
        "num_optimizer_groups": len(optimizer.param_groups),
        "attribute_branch_parameters": len(attribute_rows),
        "attribute_branch_all_registered": len(missing) == 0,
        "attribute_branch_missing": missing,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    import csv

    with (output_dir / "optimizer_parameter_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "optimizer_parameter_audit.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


# --------------------------------------------------------------------------- #
# micro / sanity step traces
# --------------------------------------------------------------------------- #
def _e_attribute_stats(model, batch) -> dict[str, float]:
    holder: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        holder["value"] = output.detach()

    handle = model.attribute_encoder.register_forward_hook(hook)
    with torch.no_grad():
        model(batch)
    handle.remove()
    value = holder["value"].to(torch.float64)
    return {
        "e_attribute_norm_mean": float(value.norm(dim=1).mean()),
        "e_attribute_norm_std": float(value.norm(dim=1).std(unbiased=False)),
        "e_attribute_patch_std": float(value.std(dim=0, unbiased=False).mean()),
        "e_attribute_element_std": float(value.std(unbiased=False)),
    }


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    pieces = [p.grad.reshape(-1) for p in params if p.grad is not None]
    if not pieces:
        return 0.0
    return float(torch.cat(pieces).to(torch.float64).norm())


def stage_step_trace(
    config,
    fit_data,
    audit,
    patch_radius,
    context_radius,
    *,
    output_dir: Path,
    seed: int = 0,
    steps: int = 10,
    subset_size: int = 1024,
    batch_size: int = 128,
    repair: bool = False,
    filename: str | None = None,
    run_gates: bool = False,
) -> dict:
    pp._seed_everything(int(seed))
    model = build_fresh_model(
        config, audit, patch_radius, context_radius, seed=seed,
        attribute_fusion_init=("small_normal" if repair else "zero"),
    )
    if repair:
        _check_repaired_init(model)
    model_config = config["model"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config.get("learning_rate", 1.0e-3)),
        weight_decay=float(model_config.get("weight_decay", 1.0e-5)),
    )
    # fixed small subset, fixed shuffle -> deterministic micro-sequence
    subset = fit_data[: int(subset_size)]
    loader = pp._make_loader(subset, batch_size, True, seed + 4242)
    batches = [b.to(torch.device("cpu")) for b in loader]
    first_batch = batches[0]

    enc = model.attribute_encoder
    patch_first_weight = model.patch_encoder.layers[0].weight
    in_width = int(patch_first_weight.shape[1])
    attr_cols = slice(in_width - int(model.attribute_input_width), in_width)

    rows: list[dict[str, Any]] = []
    step = 0
    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            model.train()
            prediction = model(batch)
            target = batch.y.view(-1)
            loss = torch.nn.functional.l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_w_final = _grad_norm([enc.fusion[-1].weight, enc.fusion[-1].bias])
            grad_upstream = _grad_norm(
                list(enc.atom_mlp.parameters()) + list(enc.bond_mlp.parameters())
            )
            grad_w_attr = _grad_norm([patch_first_weight])
            if patch_first_weight.grad is not None:
                grad_w_attr = float(
                    patch_first_weight.grad[:, attr_cols].to(torch.float64).norm()
                )
            grad_norm_total = _grad_norm(list(model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            stats = _e_attribute_stats(model, first_batch)
            w_norm = float(enc.fusion[-1].weight.detach().to(torch.float64).norm())
            b_norm = float(enc.fusion[-1].bias.detach().to(torch.float64).norm())
            rows.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "grad_norm_total": grad_norm_total,
                    "grad_norm_W_final": grad_w_final,
                    "grad_norm_attr_upstream": grad_upstream,
                    "grad_norm_W_attr_patch_encoder": grad_w_attr,
                    "W_final_norm": w_norm,
                    "b_final_norm": b_norm,
                    **stats,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            step += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    assert filename is not None
    import csv

    with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {"rows": rows, "repair": bool(repair)}
    if run_gates:
        gate = _viability_gates(model, batches[0], rows)
        (output_dir / "repaired_viability_gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["gate"] = gate
    return result


def _check_repaired_init(model) -> None:
    enc = model.attribute_encoder
    if enc is None:
        raise RuntimeError("repair requires attribute_mode != none")
    w = enc.fusion[-1].weight.detach()
    if float(w.to(torch.float64).norm()) == 0.0:
        raise RuntimeError("repaired init left the final projection at exactly zero")


def _viability_gates(model, batch, rows) -> dict[str, Any]:
    enc = model.attribute_encoder
    model.eval()

    def _e_attribute(batch_arg):
        holder: dict[str, torch.Tensor] = {}

        def hook(_m, _i, output):
            holder["value"] = output.detach()

        handle = enc.register_forward_hook(hook)
        with torch.no_grad():
            clean = model(batch_arg)
        handle.remove()
        return clean, holder["value"]

    clean_pred, e_attr = _e_attribute(batch)
    e_flat = e_attr.to(torch.float64)
    # V1 must measure *patch* variation: a constant vector with distinct
    # per-dimension values has a large element-wise std but is collapsed.
    v1_std = float(e_flat.std(dim=0, unbiased=False).mean())

    from tracks.ksvd.experiments.luyin16.compact_v6_diagnostics import _shuffle_graph

    role_shuffled = _shuffle_graph(batch, "role", 0, 0)
    with torch.no_grad():
        shuffled_pred = model(role_shuffled)

    # V4 branch ablation: force e_attribute to zero.
    def zero_hook(_m, _i, output):
        return torch.zeros_like(output)

    handle = enc.register_forward_hook(zero_hook)
    with torch.no_grad():
        ablated_pred = model(batch)
    handle.remove()

    clean_np = clean_pred.detach().numpy().reshape(-1)
    shuffled_np = shuffled_pred.detach().numpy().reshape(-1)
    ablated_np = ablated_pred.detach().numpy().reshape(-1)
    delta_shuffle = np.abs(clean_np - shuffled_np)
    delta_ablate = np.abs(clean_np - ablated_np)

    rows_last = rows[-5:]
    v2_w_final = all(r["grad_norm_W_final"] > 1e-9 for r in rows_last)
    v2_upstream = all(r["grad_norm_attr_upstream"] > 1e-9 for r in rows_last)
    v2_w_attr = all(r["grad_norm_W_attr_patch_encoder"] > 1e-9 for r in rows_last)
    gate = {
        "V1_non_collapse": {
            "patch_std_e_attribute": v1_std,
            "threshold": 1e-4,
            "pass": bool(v1_std > 1e-4),
            "e_attribute_norm_mean": float(e_flat.norm(dim=1).mean()),
            "e_attribute_norm_std": float(e_flat.norm(dim=1).std(unbiased=False)),
        },
        "V2_gradients_alive": {
            "final_projection_alive": bool(v2_w_final),
            "upstream_encoder_alive": bool(v2_upstream),
            "patch_encoder_attr_columns_alive": bool(v2_w_attr),
            "pass": bool(v2_w_final and v2_upstream and v2_w_attr),
        },
        "V3_prediction_sensitivity_role_shuffle": {
            "max_abs_delta": float(delta_shuffle.max()),
            "mean_abs_delta": float(delta_shuffle.mean()),
            "pass": bool(delta_shuffle.max() > 0.0),
        },
        "V4_branch_ablation": {
            "max_abs_delta": float(delta_ablate.max()),
            "mean_abs_delta": float(delta_ablate.mean()),
            "pass": bool(delta_ablate.max() > 0.0),
        },
    }
    gate["all_pass"] = bool(
        gate["V1_non_collapse"]["pass"]
        and gate["V2_gradients_alive"]["pass"]
        and gate["V3_prediction_sensitivity_role_shuffle"]["pass"]
        and gate["V4_branch_ablation"]["pass"]
    )
    return gate


# --------------------------------------------------------------------------- #
# stage: attribute input variance + adversarial representation check
# --------------------------------------------------------------------------- #
def stage_input_variance(
    config,
    fit_data,
    valid_data,
    *,
    output_dir: Path,
    n_samples: int = 1000,
) -> dict:
    """Raw pre-learned attribute representation variance / uniqueness."""
    def _collect(data_list, tag):
        atom_type_rows = []
        atom_role_rows = []
        occur = {"atom": [], "bond": []}
        for data in data_list[:n_samples]:
            atom_type_rows.append(data.attribute_atom_type.numpy())
            atom_role_rows.append(data.attribute_atom_role.numpy())
            occur["atom"].append(len(data.attribute_atom_type))
            occur["bond"].append(len(data.attribute_bond_type))
        atom_types = np.concatenate(atom_type_rows) if atom_type_rows else np.zeros(0, dtype=np.int64)
        atom_roles = np.concatenate(atom_role_rows) if atom_role_rows else np.zeros((0, 8), np.float32)
        unique_types = int(np.unique(atom_types).size)
        role_std = atom_roles.std(axis=0) if atom_roles.size else np.zeros(8)
        # a raw "representation" is the concatenation (type embedding is learned,
        # so we check the raw primitive tuple uniqueness)
        raw = np.concatenate(
            [atom_types.reshape(-1, 1).astype(np.float32), atom_roles], axis=1
        ) if atom_roles.size else np.zeros((0, 9), np.float32)
        raw_unique = int(np.unique(raw, axis=0).shape[0]) if raw.size else 0
        return {
            "tag": tag,
            "n_atom_occurrences": int(occur["atom"][:n_samples].__len__() and sum(occur["atom"])),
            "n_patches_sampled": len(data_list[:n_samples]),
            "unique_atom_types": unique_types,
            "atom_role_per_feature_std": role_std.tolist(),
            "atom_role_mean_std": float(role_std.mean()) if role_std.size else 0.0,
            "unique_raw_atom_primitives": raw_unique,
            "unique_raw_fraction": (
                float(raw_unique / raw.shape[0]) if raw.shape[0] else 0.0
            ),
        }

    train_stats = _collect(fit_data, "train")
    valid_stats = _collect(valid_data, "valid")
    result = {"train": train_stats, "valid": valid_stats, "n_samples": int(n_samples)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "attribute_input_variance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def stage_adversarial(
    config,
    *,
    output_dir: Path,
    seed: int = 0,
) -> dict:
    """Build an adversarial placement pair and verify the factorized view
    differs while the count view is identical (raw-representation level)."""
    radius = int(config.get("representation", {}).get("patch_radius", pp.PATCH_RADIUS))
    # a controlled placement pair on the exact role construction
    found = _adversarial_from_dataset(None, radius)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adversarial_representation_check.json").write_text(
        json.dumps(found, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return found


def _adversarial_from_dataset(dataset, radius):
    """Real-graph adversarial check: same topology + same atom/bond multiset
    on a substitution that changes *typed placement* must change the
    factorized raw primitive multiset but not the count view."""
    # We use the role primitive construction directly on a controlled
    # molecule: two different atom types swapped between a leaf and a branch
    # position of the same coarse topology.  The graph object and its
    # automorphism orbit structure are identical; only the type->role
    # assignment changes.
    import networkx as nx
    from tracks.ksvd.experiments.luyin16.compact_v6_attribute_roles import (
        patch_role_primitives,
    )

    # Build an explicit small tree, integer node labels: center 0 with two
    # neighbours 1 (leaf) and 2 (which forks into 3, 4).  Radius-2 patch at 0.
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (2, 3), (2, 4)])
    graph = _EdgeKeyGraph(graph)

    # Two atom-type assignments keeping the multiset {C,C,C,C,C} except a
    # single O (type 8): put the O on the branch atom (2) vs on the leaf (1).
    # Same coarse topology, same type multiset, different typed placement.
    base = {node: 6 for node in graph.nodes()}  # 6 == carbon
    assignment_a = dict(base); assignment_a[2] = 8  # O on branch
    assignment_b = dict(base); assignment_b[1] = 8  # O on leaf
    nodes = sorted(graph.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    types_a = np.zeros(len(nodes), dtype=np.int64)
    types_b = np.zeros(len(nodes), dtype=np.int64)
    for node in nodes:
        types_a[idx[node]] = assignment_a[node]
        types_b[idx[node]] = assignment_b[node]
    edge_types = {graph.edge_key(u, v): 1 for u, v in graph.edges()}

    prim_a = patch_role_primitives(graph, 0, types_a, edge_types, radius)
    prim_b = patch_role_primitives(graph, 0, types_b, edge_types, radius)

    factorized_a = _primitive_multiset(prim_a)
    factorized_b = _primitive_multiset(prim_b)
    count_a = sorted(prim_a.atom_types.tolist())
    count_b = sorted(prim_b.atom_types.tolist())
    return {
        "graph": "c-a, c-b, b-d, b-e (radius-2 patch at c)",
        "count_view_identical": bool(count_a == count_b),
        "factorized_view_differs": bool(factorized_a != factorized_b),
        "count_a": count_a,
        "count_b": count_b,
        "factorized_a_n": len(factorized_a),
        "factorized_b_n": len(factorized_b),
        "placement_pair_ok": bool(count_a == count_b and factorized_a != factorized_b),
    }


def _primitive_multiset(primitives) -> frozenset:
    rows = []
    for t, role in zip(primitives.atom_types.tolist(), primitives.atom_roles.tolist()):
        rows.append(("atom", int(t), tuple(round(x, 5) for x in role)))
    return frozenset(rows)


class _EdgeKeyGraph:
    """Minimal wrapper exposing the tiny interface ``patch_role_primitives``
    needs from the historical ``GraphRecord`` graph object."""

    def __init__(self, nx_graph):
        import networkx as nx

        self._g = nx_graph

    def neighbors(self, node):
        return self._g.neighbors(node)

    def nodes(self):
        return self._g.nodes()

    def edges(self):
        return self._g.edges()

    def induced(self, nodes):
        return _EdgeKeyGraph(self._g.subgraph(nodes).copy())

    def edge_key(self, u, v):
        u, v = int(u), int(v)
        return (u, v) if u <= v else (v, u)


# --------------------------------------------------------------------------- #
# stage: frozen-checkpoint mechanism diagnostics (clean / attr-zero / shuffles)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _predict_with_hook(model, graphs, hook=None, batch_size: int = 128):
    handle = None
    if hook is not None:
        handle = model.attribute_encoder.register_forward_hook(hook)
    model.eval()
    loader = pp._make_loader(graphs, int(batch_size), False, 0)
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        preds.append(model(batch).cpu().numpy().reshape(-1))
        targets.append(batch.y.view(-1).cpu().numpy())
    if handle is not None:
        handle.remove()
    return np.concatenate(targets).astype(np.float64), np.concatenate(preds).astype(np.float64)


def _branch_stats(model, graphs, batch_size: int = 128) -> dict[str, float]:
    holder: dict[str, list] = {"value": []}

    def hook(_m, _i, output):
        holder["value"].append(output.detach())

    handle = model.attribute_encoder.register_forward_hook(hook)
    model.eval()
    loader = pp._make_loader(graphs, int(batch_size), False, 0)
    with torch.no_grad():
        for batch in loader:
            model(batch)
    handle.remove()
    value = torch.cat(holder["value"], dim=0).to(torch.float64)
    first = model.patch_encoder.layers[0].weight
    in_width = int(first.shape[1])
    attr_w = first[:, in_width - int(model.attribute_input_width):]
    return {
        "e_attribute_patch_std": float(value.std(dim=0, unbiased=False).mean()),
        "e_attribute_element_std": float(value.std(unbiased=False)),
        "e_attribute_norm_mean": float(value.norm(dim=1).mean()),
        "e_attribute_norm_std": float(value.norm(dim=1).std(unbiased=False)),
        "patch_encoder_attr_column_weight_norm": float(attr_w.detach().to(torch.float64).norm()),
        "fusion_final_weight_norm": float(
            model.attribute_encoder.fusion[-1].weight.detach().to(torch.float64).norm()
        ),
    }


def stage_seed_mechanism(
    config,
    valid_data,
    run_dir: Path,
    *,
    output_dir: Path,
    seed: int = 0,
) -> dict:
    from tracks.ksvd.experiments.luyin16.compact_v6_diagnostics import _shuffle_graph

    train_records, valid_records, patch_radius, context_radius = load_records(config)
    _fit, encoded_valid, audit = pp._phase_data(
        train_records, valid_records, config=config
    )
    model = _build_model(config, audit, patch_radius, context_radius)
    state = torch.load(
        run_dir / "artifacts" / "legacy_full_result_selection_state.pt",
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)

    targets, clean = _predict_with_hook(model, encoded_valid)
    clean_mae = float(np.mean(np.abs(clean - targets)))

    def zero_hook(_m, _i, output):
        return torch.zeros_like(output)

    _t, attr_zero = _predict_with_hook(model, encoded_valid, hook=zero_hook)
    type_shuffled = [_shuffle_graph(d, "type", seed, i) for i, d in enumerate(encoded_valid)]
    role_shuffled = [_shuffle_graph(d, "role", seed, i) for i, d in enumerate(encoded_valid)]
    _t, type_pred = _predict_with_hook(model, type_shuffled)
    _t, role_pred = _predict_with_hook(model, role_shuffled)

    stats = _branch_stats(model, encoded_valid)
    result = {
        "stage": "seed_mechanism",
        "run_dir": str(run_dir),
        "seed": int(seed),
        "clean_mae": clean_mae,
        "attr_zero_mae": float(np.mean(np.abs(attr_zero - targets))),
        "attribute_type_shuffle_mae": float(np.mean(np.abs(type_pred - targets))),
        "role_association_shuffle_mae": float(np.mean(np.abs(role_pred - targets))),
        "attr_zero": _delta_stats(clean, attr_zero),
        "attribute_type_shuffle": _delta_stats(clean, type_pred),
        "role_association_shuffle": _delta_stats(clean, role_pred),
        "branch_stats": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed0_mechanism_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.save(output_dir / "seed0_clean_predictions.npy", clean)
    np.save(output_dir / "seed0_attr_zero_predictions.npy", attr_zero)
    np.save(output_dir / "seed0_role_shuffle_predictions.npy", role_pred)
    np.save(output_dir / "seed0_valid_targets.npy", targets)
    return result


def _delta_stats(clean: np.ndarray, other: np.ndarray) -> dict[str, float]:
    delta = np.abs(clean - other)
    return {
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "num_changed": int((delta > 1e-9).sum()),
        "fraction_changed": float((delta > 1e-9).mean()),
    }


def stage_seed_quintiles(
    config,
    valid_data,
    run_dir: Path,
    *,
    output_dir: Path,
    seed: int = 0,
) -> dict:
    """Difficulty-quintile bulk-safety table for a frozen checkpoint."""
    import csv as _csv

    train_records, valid_records, patch_radius, context_radius = load_records(config)
    _fit, encoded_valid, audit = pp._phase_data(train_records, valid_records, config=config)
    model = _build_model(config, audit, patch_radius, context_radius)
    state = torch.load(
        run_dir / "artifacts" / "legacy_full_result_selection_state.pt",
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    targets, clean = _predict_with_hook(model, encoded_valid)
    clean_err = np.abs(clean - targets)

    quint_path = (
        REPO_ROOT
        / "tracks/ksvd/results/baseline_difficulty_typed_refinement_audit/per_molecule_analysis.csv"
    )
    with quint_path.open(encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    quintile = np.asarray([int(r["difficulty_quintile"]) for r in rows], dtype=np.int64)
    ref_targets = np.asarray([float(r["target"]) for r in rows], dtype=np.float64)
    hist_err = np.asarray([float(r["err_hist_mean"]) for r in rows], dtype=np.float64)
    assert len(quintile) == len(targets), (len(quintile), len(targets))
    assert np.allclose(ref_targets, targets, atol=1e-4), "quintile file not aligned to valid order"

    table = []
    q1q2_v4 = q1q2_v6 = q1q2_n = 0.0
    for q in range(1, 6):
        mask = quintile == q
        v4 = float(hist_err[mask].mean())
        v6 = float(clean_err[mask].mean())
        table.append(
            {
                "quintile": q,
                "n": int(mask.sum()),
                "v4_mae": v4,
                "repaired_v6_mae": v6,
                "delta": float(v4 - v6),
            }
        )
        if q in (1, 2):
            q1q2_v4 += float(hist_err[mask].sum())
            q1q2_v6 += float(clean_err[mask].sum())
            q1q2_n += float(mask.sum())
    q1q2_v4 /= q1q2_n
    q1q2_v6 /= q1q2_n
    q1q2_degradation = q1q2_v6 - q1q2_v4
    summary = {
        "overall_v4_mae": float(hist_err.mean()),
        "overall_v6_mae": float(clean_err.mean()),
        "overall_delta_v4_minus_v6": float(hist_err.mean() - clean_err.mean()),
        "Q1Q2_v4_mae": q1q2_v4,
        "Q1Q2_v6_mae": q1q2_v6,
        # degradation = how much *worse* v6 is on the easy bulk (v6 - v4).
        "Q1Q2_combined_degradation": float(q1q2_degradation),
        "Q1Q2_safe_le_0.002": bool(q1q2_degradation <= 0.002),
        "table": table,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "seed0_difficulty_quintiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    np.save(output_dir / "seed0_best_clean_errors.npy", clean_err)
    (output_dir / "seed0_difficulty_quintiles.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _load_stage_config(run_dir: Path) -> dict:
    return _load_config(run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_FACTORIZED_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "cache",
            "original_trace",
            "optimizer_audit",
            "original_10step",
            "input_variance",
            "adversarial",
            "repaired_100step",
            "seed_mechanism",
            "seed_quintiles",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--subset-size", type=int, default=1024)
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore record cache")
    args = parser.parse_args(argv)

    config = _load_stage_config(args.run_dir)
    if args.stage == "adversarial":
        result = stage_adversarial(config, output_dir=args.output, seed=args.seed)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    fit_data, valid_data, audit, patch_radius, context_radius = build_phase_data(
        config, use_cache=not args.fresh
    )

    if args.stage == "cache":
        print(
            json.dumps(
                {
                    "train_encoded": len(fit_data),
                    "valid_encoded": len(valid_data),
                    "patch_radius": patch_radius,
                    "context_radius": context_radius,
                },
                indent=2,
            )
        )
        return 0

    if args.stage == "original_trace":
        result = stage_original_trace(
            config, fit_data, audit, patch_radius, context_radius,
            output_dir=args.output, seed=args.seed,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "gradients"}, indent=2, default=str))
        return 0

    if args.stage == "optimizer_audit":
        result = stage_optimizer_audit(
            config, audit, output_dir=args.output, seed=args.seed,
            patch_radius=patch_radius, context_radius=context_radius,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        return 0

    if args.stage == "original_10step":
        result = stage_step_trace(
            config, fit_data, audit, patch_radius, context_radius,
            output_dir=args.output, seed=args.seed, steps=args.steps,
            subset_size=args.subset_size, repair=False,
            filename=args.out_csv or "original_10step_trace.csv",
        )
        return 0

    if args.stage == "input_variance":
        result = stage_input_variance(
            config, fit_data, valid_data, output_dir=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.stage == "repaired_100step":
        result = stage_step_trace(
            config, fit_data, audit, patch_radius, context_radius,
            output_dir=args.output, seed=args.seed, steps=args.steps,
            subset_size=args.subset_size, repair=True,
            filename="repaired_100step_trace.csv", run_gates=True,
        )
        print(json.dumps(result.get("gate", {}), indent=2))
        return 0

    if args.stage == "seed_mechanism":
        result = stage_seed_mechanism(
            config, valid_data, args.run_dir, output_dir=args.output, seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.stage == "seed_quintiles":
        result = stage_seed_quintiles(
            config, valid_data, args.run_dir, output_dir=args.output, seed=args.seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
