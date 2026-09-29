"""E2E-DictEnv training-protocol audit — shared-prefix exact-fork core module.

Round ``e2e_dictenv_training_protocol_audit_v1`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_training_protocol_audit_v1_preregistration.md``.

Question.  How much of the current CSSD-q1 plateau (~0.128-0.130 Top-5 soup
valid MAE) comes from the frozen training protocol (fixed ``Adam lr = 1e-3``
for the full 320 epochs) rather than from the architecture?

Design.  Exactly one architecture (the frozen ``CSSDModel`` q1; 97727
parameters) is trained once to epoch 280, then the exact model/optimizer/RNG/
loader state is cloned and only the learning rate of epochs 281-320 changes:

* CONTROL: ``lr = 1e-3``
* LOW-LR : ``lr = 1e-4``

The module reuses, instead of duplicating: the CSSD model builder and coder
(``cssd``), the frozen dataset/cache pipeline and loader contract
(``zinc_e2e_dictenv_p1``), the frozen optimizer constants and dictionary loader
(``zinc_e2e_dictenv_p2_abs``), the frozen CPU evaluation and state hashing
(``e2e_dictenv_h1_clarity_audit``) and the frozen usage/reconstruction metrics
(``e2e_dictenv_dictionary_coder_audit_v1``).  No architecture is implemented or
modified here.

CPU only.  ``official_test_loaded = False`` in every payload; no import of this
module reads the official test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_dictionary_coder_audit_v1 as dca
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

PROTOCOL_VERSION = "e2e_dictenv_training_protocol_audit_v1"

# ---------------------------------------------------------------------------
# frozen design constants (preregistration sections 2-6)
# ---------------------------------------------------------------------------

PREFIX_EPOCHS = 280
TAIL_EPOCHS = 40
TOTAL_EPOCHS = PREFIX_EPOCHS + TAIL_EPOCHS
FORK_EPOCH = PREFIX_EPOCHS
CONTROL_LR = 1.0e-3
LOW_LR = 1.0e-4
SEED = 0
SOUP_K = int(p2run.SOUP_K)          # 5
SPARSITY = int(cssd.SPARSITY)       # 8

#: frozen architecture tag; the audit never builds anything else.
AUDIT_TAG = "TPA"
AUDIT_SPEC = cssd.CSSD_SPEC
AUDIT_MASK = cssd.CSSD_MASK
EXPECTED_PARAMETERS = 97727

#: module-level gradient / update diagnostics (exactly five groups).
MODULE_GROUPS: tuple[str, ...] = (
    "dictionary",
    "node_semantic_binding",
    "edge_semantic_binding",
    "pair_encoder",
    "reader",
)
GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "dictionary": ("D",),
    "node_semantic_binding": ("W_A_S",),
    "edge_semantic_binding": ("W_E_S",),
    "pair_encoder": ("pair_encoder",),
    "reader": ("reader",),
}

# ---------------------------------------------------------------------------
# frozen historical references (preregistration section 1)
# ---------------------------------------------------------------------------

HISTORICAL_SOUP_MAE = 0.13002798487985273
HISTORICAL_SOUP_MEMBERS = (282, 299, 307, 312, 313)
HISTORICAL_BEST_MAE = 0.13691346781601896
HISTORICAL_BEST_EPOCH = 307
HISTORICAL_EPOCH_MAE: dict[int, float] = {
    40: 0.21084321507340065,
    120: 0.1710996474697604,
    240: 0.154002405811043,
    280: 0.14115870875114342,
    320: 0.14931416233995695,
}
V2_LOW_RATE_WARM_SOUP = 0.1287231611124589
FINAL_CLEAN_SOUP = float(cssd.REF_SPARSE_SEED0_SOUP_MAE)

PREFIX_MAE_TOLERANCE = 0.02
PREFIX_MAE_MAX = HISTORICAL_EPOCH_MAE[FORK_EPOCH] + PREFIX_MAE_TOLERANCE

# ---------------------------------------------------------------------------
# frozen decision thresholds (preregistration sections 6-7)
# ---------------------------------------------------------------------------

NOISE_FLOOR = 6.0e-4
CASE_A_MAX = 0.002
CASE_B_MIN = 0.003
CASE_C_MIN = 0.005
CASE_C_ABS = 0.125
CASE_D_ABS = 0.120
CATASTROPHIC_MAE = 2.0

# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def official_test_blocker(payload: Mapping[str, Any]) -> None:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("official-test blocker violated (official_test_loaded must be False)")


def cpu_only_guard(device: Any) -> None:
    resolved = torch.device(device)
    if resolved.type != "cpu":
        raise RuntimeError(f"training-protocol audit is CPU-only, got device={resolved}")


# ---------------------------------------------------------------------------
# deterministic hashing helpers
# ---------------------------------------------------------------------------


def optimizer_state_sha256(optimizer: torch.optim.Adam) -> str:
    """Canonical SHA-256 over Adam state entries (tensors) and group settings."""
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    for index in sorted(state["state"], key=lambda key: int(key)):
        entry = state["state"][index]
        digest.update(f"p{int(index)}:".encode())
        for field in sorted(entry):
            value = entry[field]
            digest.update(f"{field}=".encode())
            if torch.is_tensor(value):
                digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
            else:
                digest.update(str(value).encode())
            digest.update(b";")
    for group in state["param_groups"]:
        settings = {key: value for key, value in group.items() if key != "params"}
        digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def optimizer_state_comparison(
    left: torch.optim.Adam, right: torch.optim.Adam, *, ignore_lr: bool = False
) -> dict[str, Any]:
    """Field-by-field comparison of two Adam optimizers loaded from one state.

    Returns the exact tensor-level equality report and the list of differing
    param-group fields (``ignore_lr=True`` drops ``lr`` from the diff, which is
    the post-LR-edit integrity requirement).
    """
    left_state = left.state_dict()
    right_state = right.state_dict()
    exp_avg_equal = exp_avg_sq_equal = step_equal = True
    compared = 0
    per_parameter: dict[str, dict[str, bool]] = {}
    for index in sorted(left_state["state"], key=lambda key: int(key)):
        left_entry = left_state["state"][index]
        right_entry = right_state["state"][index]
        flags = {
            "exp_avg": bool(torch.equal(left_entry["exp_avg"], right_entry["exp_avg"])),
            "exp_avg_sq": bool(torch.equal(left_entry["exp_avg_sq"], right_entry["exp_avg_sq"])),
            "step": bool(torch.equal(left_entry["step"], right_entry["step"])),
        }
        compared += 1
        exp_avg_equal &= flags["exp_avg"]
        exp_avg_sq_equal &= flags["exp_avg_sq"]
        step_equal &= flags["step"]
        per_parameter[str(int(index))] = flags
    group_fields: list[str] = []
    group_count = len(left_state["param_groups"])
    if group_count != len(right_state["param_groups"]):
        raise RuntimeError("optimizer param-group count changed")
    for left_group, right_group in zip(left_state["param_groups"], right_state["param_groups"]):
        keys = sorted(set(left_group) | set(right_group))
        for key in keys:
            if ignore_lr and key == "lr":
                continue
            if left_group.get(key) != right_group.get(key):
                group_fields.append(str(key))
    return {
        "compared_parameters": compared,
        "exp_avg_identical": bool(exp_avg_equal),
        "exp_avg_sq_identical": bool(exp_avg_sq_equal),
        "step_identical": bool(step_equal),
        "differing_param_group_fields": sorted(set(group_fields)),
        "per_parameter_identical": bool(
            all(all(flags.values()) for flags in per_parameter.values())
        ),
    }


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return audit.state_sha256(state)


def rng_fingerprint() -> str:
    return hashlib.sha256(bytes(torch.get_rng_state().numpy().tobytes())).hexdigest()


def generator_fingerprint(generator: torch.Generator) -> str:
    return hashlib.sha256(bytes(generator.get_state().numpy().tobytes())).hexdigest()


def tag_graph_ids(graphs: Sequence[Any]) -> None:
    """Attach a stable per-graph id so every batch order can be hashed."""
    for index, data in enumerate(graphs):
        data.gid = torch.tensor([int(index)], dtype=torch.long)


def _gid_digest_update(digest: "hashlib._Hash", batch: Any) -> None:
    gid = getattr(batch, "gid", None)
    if gid is None:
        raise RuntimeError("graphs are not tagged with gid; batch order cannot be hashed")
    digest.update(np.ascontiguousarray(gid.detach().cpu().numpy()).tobytes())


def replay_epoch_batch_order_hash(
    data: Sequence[Any], *, seed: int, loader_state: torch.Tensor, shuffle_seed: int | None = None
) -> dict[str, Any]:
    """Hash one epoch of batch order without touching the training loader."""
    loader = p1.make_env_loader(
        data,
        int(p2run.BATCH_SIZE),
        True,
        int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET) if shuffle_seed is None else int(shuffle_seed),
    )
    loader.generator.set_state(loader_state.clone())
    digest = hashlib.sha256()
    batches = 0
    for batch in loader:
        _gid_digest_update(digest, batch)
        batches += 1
    return {"sha256": digest.hexdigest(), "n_batches": batches}


# ---------------------------------------------------------------------------
# module-level gradient / update diagnostics
# ---------------------------------------------------------------------------


def _is_group_parameter(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def parameter_groups(model: torch.nn.Module) -> dict[str, list[str]]:
    """The five frozen diagnostic groups, guaranteed non-empty and disjoint.

    The groups are a read-only diagnostic partition of the interesting modules;
    they do not have to cover every trainable parameter (the global gradient and
    update norms are computed over all parameters).
    """
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups: dict[str, list[str]] = {}
    used: set[str] = set()
    for group in MODULE_GROUPS:
        members = [name for name in names if _is_group_parameter(name, GROUP_PREFIXES[group])]
        if not members:
            raise RuntimeError(f"diagnostic group {group!r} matched no parameters")
        overlap = used & set(members)
        if overlap:
            raise RuntimeError(f"diagnostic groups overlap on {sorted(overlap)}")
        used.update(members)
        groups[group] = members
    return groups


def _flat_norm(tensors: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        total += float(tensor.detach().pow(2).sum())
    return float(math.sqrt(max(total, 0.0)))


def _delta_norm(parameters: Sequence[torch.Tensor], before: Sequence[torch.Tensor]) -> float:
    total = 0.0
    for parameter, reference in zip(parameters, before):
        total += float((parameter.detach() - reference).pow(2).sum())
    return float(math.sqrt(max(total, 0.0)))


# ---------------------------------------------------------------------------
# soup keepers (frozen Top-5-by-valid-MAE rule, ties by earlier epoch)
# ---------------------------------------------------------------------------


def empty_keeper() -> dict[str, Any]:
    return {"entries": [], "states": {}}


def seed_keeper(entries: Sequence[tuple[int, float]], states: Mapping[int, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    """Rebuild a keeper from a previous segment (prefix -> continuation)."""
    ordered = sorted((int(epoch), float(mae)) for epoch, mae in entries)
    return {
        "entries": ordered,
        "states": {int(epoch): states[int(epoch)] for epoch, _ in ordered},
    }


def retain_epoch(
    keeper: Mapping[str, Any],
    *,
    epoch: int,
    valid_mae: float,
    state: Mapping[str, torch.Tensor],
    k: int = SOUP_K,
) -> dict[str, Any]:
    """Insert one epoch and keep the frozen top-``k`` by (valid MAE, epoch)."""
    entries = list(keeper["entries"]) + [(int(epoch), float(valid_mae))]
    entries.sort(key=lambda item: (item[1], item[0]))
    kept = entries[: int(k)]
    members = {member for member, _ in kept}
    states = {member: value for member, value in keeper["states"].items() if member in members}
    if int(epoch) in members:
        states[int(epoch)] = state
    return {"entries": kept, "states": states}


def keeper_members(keeper: Mapping[str, Any]) -> list[int]:
    return sorted(int(epoch) for epoch, _ in keeper["entries"])


def keeper_member_mae(keeper: Mapping[str, Any]) -> list[float]:
    return [float(mae) for _, mae in sorted(keeper["entries"], key=lambda item: item[0])]


def soup_state_from(
    keeper: Mapping[str, Any], members: Sequence[int] | None = None
) -> dict[str, torch.Tensor]:
    """Frozen weight soup: mean of the retained epoch states, key order preserved."""
    chosen = keeper_members(keeper) if members is None else sorted(int(m) for m in members)
    if not chosen:
        raise RuntimeError("cannot average an empty soup")
    return {
        key: torch.stack([keeper["states"][epoch][key].float() for epoch in chosen], dim=0).mean(0)
        for key in keeper["states"][chosen[0]]
    }


def _best_epoch(curve: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
    best = min(curve, key=lambda row: (float(row["valid_mae"]), int(row["epoch"])))
    return int(best["epoch"]), float(best["valid_mae"])


# ---------------------------------------------------------------------------
# the single training loop used for the prefix and both continuations
# ---------------------------------------------------------------------------


def run_epochs(
    *,
    tag: str,
    start_epoch: int,
    end_epoch: int,
    threads: int,
    out_dir: Path,
    subspace: cssd.CommonSubspace,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    seed: int = SEED,
    lr: float = CONTROL_LR,
    weight_decay: float = float(p2run.WEIGHT_DECAY),
    grad_clip: float = float(p2run.GRAD_CLIP),
    model_state: Mapping[str, torch.Tensor] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    rng_state: torch.Tensor | None = None,
    loader_state: torch.Tensor | None = None,
    soup_seed: Mapping[str, Any] | None = None,
    tail_seed: Mapping[str, Any] | None = None,
    best_seed: tuple[int, float] | None = None,
    track_tail: bool = False,
    model_factory: Callable[..., cssd.CSSDModel] | None = None,
    log: bool = True,
    save_states: bool = True,
) -> dict[str, Any]:
    """Run the frozen CSSD training loop over ``[start_epoch, end_epoch]``.

    From scratch (``model_state is None``) this is a faithful copy of the frozen
    ``cssd.train_cssd`` training path (test-asserted bit-equivalence), with the
    frozen optimizer constants read from ``zinc_e2e_dictenv_p2_abs``.  With a
    saved state it resumes the exact CPU/RNG/loader contract: the model and
    optimizer state are loaded unchanged, and only then is the requested
    ``lr`` applied to every parameter group (the fork's single edit).
    """
    device = audit.attach_cpu(int(threads))
    cpu_only_guard(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(end_epoch) < int(start_epoch):
        raise ValueError("end_epoch must be >= start_epoch")
    dictionary, dict_sha = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)
    factory = cssd.build_cssd_model if model_factory is None else model_factory
    resumed = model_state is not None
    if not resumed:
        p2run._seed_everything(int(seed))
        model = factory(dictionary, int(seed), subspace)
        initial_state = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
        initial_state_sha = state_fingerprint(initial_state)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        lr_contract = {
            "resumed": False,
            "lr_requested": float(lr),
            "lr_before": [float(group["lr"]) for group in optimizer.param_groups],
            "lr_after": [float(group["lr"]) for group in optimizer.param_groups],
        }
    else:
        model = factory(dictionary, int(seed), subspace)
        model.load_state_dict({key: value for key, value in model_state.items()})
        initial_state = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
        initial_state_sha = state_fingerprint(initial_state)
        if optimizer_state is None:
            raise ValueError("resumed runs require the saved optimizer state")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(CONTROL_LR), weight_decay=float(weight_decay)
        )
        optimizer.load_state_dict(optimizer_state)
        lr_before = [float(group["lr"]) for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = float(lr)
        lr_after = [float(group["lr"]) for group in optimizer.param_groups]
        lr_contract = {
            "resumed": True,
            "lr_requested": float(lr),
            "lr_before": lr_before,
            "lr_after": lr_after,
        }
    model = model.to(device)
    groups = parameter_groups(model)
    parameters = dict(model.named_parameters())
    group_parameters = {name: [parameters[key] for key in members] for name, members in groups.items()}

    train_loader = p1.make_env_loader(
        train_data, int(p2run.BATCH_SIZE), True, int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET)
    )
    eval_loader = p1.make_env_loader(
        valid_data, int(p2run.BATCH_SIZE), False, int(seed) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    if loader_state is not None:
        train_loader.generator.set_state(loader_state.to("cpu"))
    if rng_state is not None:
        torch.set_rng_state(rng_state.to("cpu"))
    loader_fingerprint_before = generator_fingerprint(train_loader.generator)

    mask = AUDIT_MASK
    lam = float(cm.H1_LAMBDA)
    keeper = empty_keeper() if soup_seed is None else seed_keeper(
        soup_seed["entries"], soup_seed["states"]
    )
    tail_keeper = empty_keeper() if tail_seed is None else seed_keeper(
        tail_seed["entries"], tail_seed["states"]
    )
    best_valid_mae = float("inf")
    best_epoch_value = int(start_epoch)
    if best_seed is not None:
        best_epoch_value, best_valid_mae = int(best_seed[0]), float(best_seed[1])
    curve: list[dict[str, Any]] = []
    batch_order_hashes: dict[int, str] = {}
    rss_start = audit._rss_mb()
    started = time.perf_counter()
    for epoch in range(int(start_epoch), int(end_epoch) + 1):
        epoch_started = time.perf_counter()
        model.train()
        task_sum = rec_sum = rec_term_sum = total_loss_sum = 0.0
        task_loss_sum = 0.0
        grad_global_sum = 0.0
        update_global_sum = 0.0
        group_grads = {name: 0.0 for name in MODULE_GROUPS}
        group_updates = {name: 0.0 for name in MODULE_GROUPS}
        n_mol = n_nodes = n_batches = 0
        order_digest = hashlib.sha256()
        for batch in train_loader:
            _gid_digest_update(order_digest, batch)
            batch = batch.to(device)
            prediction, aux = model(batch, mask=mask, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            task = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            loss = task + lam * rec
            optimizer.zero_grad()
            loss.backward()
            grad_global_sum += _flat_norm([parameter.grad for parameter in model.parameters()])
            for name in MODULE_GROUPS:
                group_grads[name] += _flat_norm(
                    [parameter.grad for parameter in group_parameters[name]]
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            before = {key: parameter.detach().clone() for key, parameter in parameters.items()}
            optimizer.step()
            update_global_sum += _delta_norm(list(parameters.values()), [before[key] for key in parameters])
            for name in MODULE_GROUPS:
                group_updates[name] += _delta_norm(
                    group_parameters[name], [before[key] for key in groups[name]]
                )
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            task_loss_sum += float(task.detach())
            total_loss_sum += float(loss.detach())
            rec_term_sum += float(rec.detach())
            n_batches += 1
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float(
                (((phi - phi_hat) ** 2).sum(dim=1) / ((phi**2).sum(dim=1) + float(v0.EPS))).sum()
            )
            n_nodes += int(phi.shape[0])
        steps = max(n_batches, 1)
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = audit._evaluate_model(model, eval_loader, device, mask)
        valid_mae = float(valid["mae"])
        row: dict[str, Any] = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_mae": train_mae,
            "train_task_loss": float(task_loss_sum / steps),
            "train_rec": train_rec,
            "train_rec_term": float(rec_term_sum / steps),
            "train_total_loss": float(total_loss_sum / steps),
            "valid_mae": valid_mae,
            "grad_norm_global": float(grad_global_sum / steps),
            "update_norm_global": float(update_global_sum / steps),
            "d_norm": float(model.D.detach().norm()),
            "batch_order_hash": order_digest.hexdigest(),
            "seconds": float(time.perf_counter() - epoch_started),
        }
        for name in MODULE_GROUPS:
            row[f"grad_norm_{name}"] = float(group_grads[name] / steps)
            row[f"update_norm_{name}"] = float(group_updates[name] / steps)
        curve.append(row)
        batch_order_hashes[int(epoch)] = order_digest.hexdigest()
        epoch_state = {
            key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()
        }
        keeper = retain_epoch(keeper, epoch=epoch, valid_mae=valid_mae, state=epoch_state)
        if track_tail:
            tail_keeper = retain_epoch(
                tail_keeper, epoch=epoch, valid_mae=valid_mae, state=epoch_state
            )
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_epoch_value = int(epoch)
        if log and (epoch == int(start_epoch) or epoch % 10 == 0 or epoch == int(end_epoch)):
            print(
                f"[{tag}] epoch={epoch:03d} lr={row['lr']:.0e} train={train_mae:.6f} "
                f"valid={valid_mae:.6f} best={best_valid_mae:.6f}@{best_epoch_value} "
                f"upd={row['update_norm_global']:.3e}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)

    members = keeper_members(keeper)
    soup_state = soup_state_from(keeper, members)
    soup_model = factory(dictionary, int(seed), subspace)
    soup_model.load_state_dict(soup_state)
    soup_model = soup_model.to(device)
    soup_valid = audit._evaluate_model(soup_model, eval_loader, device, mask)
    tail_payload: dict[str, Any] | None = None
    if track_tail:
        tail_members = keeper_members(tail_keeper)
        tail_state = soup_state_from(tail_keeper, tail_members)
        tail_model = factory(dictionary, int(seed), subspace)
        tail_model.load_state_dict(tail_state)
        tail_model = tail_model.to(device)
        tail_valid = audit._evaluate_model(tail_model, eval_loader, device, mask)
        tail_payload = {
            "members": tail_members,
            "member_valid_mae": keeper_member_mae(tail_keeper),
            "soup_valid_mae": float(tail_valid["mae"]),
            "window": [int(start_epoch), int(end_epoch)],
            "top_k_rule": "valid MAE, ties by earlier epoch",
        }
    final_state = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "tag": str(tag),
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "epochs_run": int(len(curve)),
        "resumed": bool(resumed),
        "lr_contract": lr_contract,
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "batch_size": int(p2run.BATCH_SIZE),
        "lambda_rec": lam,
        "mask": mask.as_dict() if mask is not None else None,
        "config": cm.H1_CONFIG.as_dict(),
        "common_dim": int(subspace.q),
        "subspace_kind": subspace.kind,
        "dictionary_sha256": dict_sha,
        "actual_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": state_fingerprint(final_state),
        "soup_state_sha256": state_fingerprint(soup_state),
        "rng_fingerprint_end": rng_fingerprint(),
        "loader_fingerprint_start": loader_fingerprint_before,
        "loader_fingerprint_end": generator_fingerprint(train_loader.generator),
        "curve": curve,
        "batch_order_hashes": {str(epoch): value for epoch, value in batch_order_hashes.items()},
        "soup": {
            "members": members,
            "member_valid_mae": keeper_member_mae(keeper),
            "soup_valid_mae": float(soup_valid["mae"]),
            "window": [1, int(end_epoch)],
            "top_k_rule": "valid MAE, ties by earlier epoch",
            "members_after_280": sorted(member for member in members if member > FORK_EPOCH),
            "n_members_after_280": int(sum(1 for member in members if member > FORK_EPOCH)),
        },
        "tail_soup": tail_payload,
        "best_valid_mae": float(best_valid_mae),
        "best_epoch": int(best_epoch_value),
        "final_valid_mae": float(curve[-1]["valid_mae"]),
        "final_train_mae": float(curve[-1]["train_mae"]),
        "wall_clock_s": wall,
        "seconds_per_epoch": float(wall / max(len(curve), 1)),
        "rss_start_mb": float(rss_start),
        "rss_end_mb": float(audit._rss_mb()),
        "peak_rss_mb": audit._peak_rss_mb(),
        "official_test_loaded": False,
    }
    payload["keeper"] = {
        "entries": [[int(epoch), float(mae)] for epoch, mae in keeper["entries"]],
        "states": keeper["states"],
    }
    payload["tail_keeper"] = (
        {
            "entries": [[int(epoch), float(mae)] for epoch, mae in tail_keeper["entries"]],
            "states": tail_keeper["states"],
        }
        if track_tail
        else None
    )
    if save_states:
        torch.save(final_state, out_dir / f"{tag}_final_state.pt")
        torch.save(soup_state, out_dir / f"{tag}_soup_state.pt")
        torch.save(
            {
                "protocol_version": PROTOCOL_VERSION,
                "tag": str(tag),
                "start_epoch": int(start_epoch),
                "end_epoch": int(end_epoch),
                "model_state": final_state,
                "optimizer_state": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "train_loader_state": train_loader.generator.get_state(),
                "keeper": payload["keeper"],
                "tail_keeper": payload["tail_keeper"],
                "best": [int(best_epoch_value), float(best_valid_mae)],
                "curve": curve,
                "official_test_loaded": False,
            },
            out_dir / f"{tag}_resume_checkpoint.pt",
        )
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# fork integrity
# ---------------------------------------------------------------------------


def _predictions(model: torch.nn.Module, loader: Any, device: torch.device) -> np.ndarray:
    return np.asarray(audit.evaluate_mask(model, loader, device, AUDIT_MASK)["predictions"])


def fork_integrity(
    *,
    checkpoint: Mapping[str, Any],
    dictionary: np.ndarray,
    subspace: cssd.CommonSubspace,
    train_data: Sequence[Any],
    valid_data: Sequence[Any],
    threads: int = 4,
    seed: int = SEED,
    probe_size: int = 256,
    model_factory: Callable[..., cssd.CSSDModel] | None = None,
) -> dict[str, Any]:
    """Load the fork checkpoint twice, prove exactness, then apply the LR edit.

    Returns the frozen ``fork_integrity.json`` payload.  The two probe arms are
    built with the *same* construction code as ``run_epochs`` resumes (fresh
    Adam at ``CONTROL_LR`` -> load prefix optimizer state -> set the arm LR), so
    the tensor-level optimizer comparison proves exactly what the continuations
    inherit.
    """
    device = audit.attach_cpu(int(threads))
    cpu_only_guard(device)
    factory = cssd.build_cssd_model if model_factory is None else model_factory
    model_state = checkpoint["model_state"]
    optimizer_state = checkpoint["optimizer_state"]
    rng_state = checkpoint["rng_state"].to("cpu")
    loader_state = checkpoint["train_loader_state"].to("cpu")

    def load_arm() -> tuple[cssd.CSSDModel, torch.optim.Adam]:
        model = factory(dictionary, int(seed), subspace)
        model.load_state_dict({key: value for key, value in model_state.items()})
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(CONTROL_LR),
            weight_decay=float(p2run.WEIGHT_DECAY),
        )
        optimizer.load_state_dict(optimizer_state)
        return model.to(device), optimizer

    control_model, control_optimizer = load_arm()
    low_model, low_optimizer = load_arm()

    # pre-edit: both arms must be exactly the checkpoint state
    model_hash_control = state_fingerprint(control_model.state_dict())
    model_hash_low = state_fingerprint(low_model.state_dict())
    optimizer_hash_control = optimizer_state_sha256(control_optimizer)
    optimizer_hash_low = optimizer_state_sha256(low_optimizer)
    optimizer_pre_edit = optimizer_state_comparison(control_optimizer, low_optimizer)

    # the fork's single edit
    lr_edit = {
        "control_lr": float(CONTROL_LR),
        "low_lr": float(LOW_LR),
        "control_lr_before": [float(group["lr"]) for group in control_optimizer.param_groups],
        "low_lr_before": [float(group["lr"]) for group in low_optimizer.param_groups],
    }
    for group in control_optimizer.param_groups:
        group["lr"] = float(CONTROL_LR)
    for group in low_optimizer.param_groups:
        group["lr"] = float(LOW_LR)
    optimizer_comparison = optimizer_state_comparison(
        control_optimizer, low_optimizer, ignore_lr=True
    )

    probe_data = list(valid_data)[: int(probe_size)]
    probe_loader = p1.make_env_loader(
        probe_data, int(p2run.BATCH_SIZE), False, int(seed) + int(p2run.EVAL_SHUFFLE_OFFSET)
    )
    torch.set_rng_state(rng_state)
    prediction_control = _predictions(control_model, probe_loader, device)
    torch.set_rng_state(rng_state)
    prediction_low = _predictions(low_model, probe_loader, device)
    prediction_max_abs_diff = (
        float(np.abs(prediction_control - prediction_low).max())
        if prediction_control.size
        else 0.0
    )

    train_probe = next(iter(probe_loader))
    train_probe = train_probe.to(device)
    control_model.train()
    low_model.train()
    torch.set_rng_state(rng_state)
    train_prediction_control = control_model(train_probe, mask=AUDIT_MASK).detach()
    torch.set_rng_state(rng_state)
    train_prediction_low = low_model(train_probe, mask=AUDIT_MASK).detach()
    train_mode_max_abs_diff = float(
        (train_prediction_control - train_prediction_low).abs().max()
    )

    # the RNG contract the two continuations will restore
    torch.set_rng_state(rng_state)
    rng_fingerprint_value = rng_fingerprint()
    loader_probe = p1.make_env_loader(
        train_data,
        int(p2run.BATCH_SIZE),
        True,
        int(seed) + int(p2run.TRAIN_SHUFFLE_OFFSET),
    )
    loader_probe.generator.set_state(loader_state.clone())
    loader_fingerprint_value = generator_fingerprint(loader_probe.generator)
    order = replay_epoch_batch_order_hash(train_data, seed=seed, loader_state=loader_state)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "device": "cpu",
        "threads": int(threads),
        "seed": int(seed),
        "epoch": FORK_EPOCH,
        "model_state_hash_control": model_hash_control,
        "model_state_hash_low_lr": model_hash_low,
        "model_state_identical": bool(model_hash_control == model_hash_low),
        "optimizer_state_hash_control": optimizer_hash_control,
        "optimizer_state_hash_low_lr": optimizer_hash_low,
        "optimizer_state_identical_before_lr_edit": bool(
            optimizer_hash_control == optimizer_hash_low
        ),
        "optimizer_field_comparison_pre_edit": optimizer_pre_edit,
        "optimizer_field_comparison_ignoring_lr": optimizer_comparison,
        "prediction_max_abs_diff": prediction_max_abs_diff,
        "train_mode_prediction_max_abs_diff": train_mode_max_abs_diff,
        "rng_fingerprint": rng_fingerprint_value,
        "loader_generator_fingerprint": loader_fingerprint_value,
        "next_epoch_batch_order_hash": order,
        "lr_edit": {
            **lr_edit,
            "control_lr_after": [float(group["lr"]) for group in control_optimizer.param_groups],
            "low_lr_after": [float(group["lr"]) for group in low_optimizer.param_groups],
        },
        "checks": {
            "model_states_identical": bool(model_hash_control == model_hash_low),
            "optimizer_states_identical_before_lr_edit": bool(
                optimizer_hash_control == optimizer_hash_low
            ),
            "optimizer_pre_edit_fully_identical": bool(optimizer_pre_edit["per_parameter_identical"]),
            "exp_avg_identical": bool(optimizer_comparison["exp_avg_identical"]),
            "exp_avg_sq_identical": bool(optimizer_comparison["exp_avg_sq_identical"]),
            "step_counters_identical": bool(optimizer_comparison["step_identical"]),
            "only_lr_differs_after_edit": bool(
                not optimizer_comparison["differing_param_group_fields"]
            ),
            "prediction_max_abs_diff_zero": bool(prediction_max_abs_diff == 0.0),
            "train_mode_prediction_max_abs_diff_zero": bool(train_mode_max_abs_diff == 0.0),
            "control_lr_is_1e3": bool(
                all(float(group["lr"]) == CONTROL_LR for group in control_optimizer.param_groups)
            ),
            "low_lr_is_1e4": bool(
                all(float(group["lr"]) == LOW_LR for group in low_optimizer.param_groups)
            ),
        },
        "official_test_loaded": False,
    }
    payload["passed"] = bool(all(payload["checks"].values()))
    payload["verdict"] = "FORK_INTEGRITY_OK" if payload["passed"] else "FORK_INTEGRITY_FAILURE"
    official_test_blocker(payload)
    return payload


# ---------------------------------------------------------------------------
# prefix gate and historical comparison
# ---------------------------------------------------------------------------


def prefix_gate(
    *,
    curve: Sequence[Mapping[str, Any]],
    initial_state_sha256: str,
    final_state_sha256: str,
    dictionary_sha256: str,
    expected_parameters: int = EXPECTED_PARAMETERS,
    actual_parameters: int,
    model_finite: bool,
    dictionary_moved: bool,
    historical_epoch_mae: float = HISTORICAL_EPOCH_MAE[FORK_EPOCH],
    tolerance: float = PREFIX_MAE_TOLERANCE,
) -> dict[str, Any]:
    """Frozen prefix failure gate (preregistration section 8)."""
    rows = {int(row["epoch"]): row for row in curve}
    if FORK_EPOCH not in rows:
        raise RuntimeError(f"prefix curve does not contain epoch {FORK_EPOCH}")
    epoch_mae = float(rows[FORK_EPOCH]["valid_mae"])
    train_mae = float(rows[FORK_EPOCH]["train_mae"])
    finite = all(
        np.isfinite(float(row[key]))
        for row in curve
        for key in ("train_mae", "valid_mae", "train_rec_term")
    )
    limit = float(historical_epoch_mae) + float(tolerance)
    checks = {
        "epoch280_valid_within_tolerance": bool(np.isfinite(epoch_mae) and epoch_mae <= limit),
        "epoch280_train_bounded": bool(np.isfinite(train_mae) and train_mae < CATASTROPHIC_MAE),
        "curves_finite": bool(finite),
        "model_finite": bool(model_finite),
        "dictionary_moved": bool(dictionary_moved),
        "parameter_count_exact": bool(int(actual_parameters) == int(expected_parameters)),
        "initial_state_hash_recorded": bool(isinstance(initial_state_sha256, str) and initial_state_sha256),
        "final_state_hash_recorded": bool(isinstance(final_state_sha256, str) and final_state_sha256),
        "dictionary_sha_recorded": bool(isinstance(dictionary_sha256, str) and dictionary_sha256),
    }
    passed = bool(all(checks.values()))
    return {
        "epoch": FORK_EPOCH,
        "epoch280_valid_mae": epoch_mae,
        "epoch280_train_mae": train_mae,
        "historical_epoch280_valid_mae": float(historical_epoch_mae),
        "tolerance": float(tolerance),
        "limit": limit,
        "checks": checks,
        "passed": passed,
        "verdict": "PREFIX_HEALTHY" if passed else "PREFIX_REGIME_MISMATCH",
        "official_test_loaded": False,
    }


def historical_comparison(
    curve: Sequence[Mapping[str, Any]],
    historical_curve: Sequence[Mapping[str, Any]] | None,
    *,
    checkpoints: Sequence[int] = (40, 120, 240, 280),
) -> dict[str, Any]:
    """Provenance-only drift statistics against the frozen historical curve."""
    now = {int(row["epoch"]): float(row["valid_mae"]) for row in curve}
    frozen = {int(row["epoch"]): float(row["valid_mae"]) for row in (historical_curve or [])}
    if not frozen:
        return {
            "available": False,
            "note": "historical per-epoch curve not present locally; frozen checkpoint constants used",
            "frozen_checkpoint_mae": {str(epoch): float(value) for epoch, value in HISTORICAL_EPOCH_MAE.items()},
            "official_test_loaded": False,
        }
    shared = sorted(set(now) & set(frozen))
    deltas = np.asarray([now[epoch] - frozen[epoch] for epoch in shared], dtype=np.float64)
    return {
        "available": True,
        "n_shared_epochs": int(len(shared)),
        "mean_abs_delta": float(np.abs(deltas).mean()),
        "median_abs_delta": float(np.median(np.abs(deltas))),
        "p90_abs_delta": float(np.quantile(np.abs(deltas), 0.90)),
        "max_abs_delta": float(np.abs(deltas).max()),
        "mean_delta": float(deltas.mean()),
        "checkpoints": {
            str(epoch): {
                "shared_prefix": now[epoch],
                "historical": frozen[epoch],
                "delta": now[epoch] - frozen[epoch],
            }
            for epoch in checkpoints
            if epoch in now and epoch in frozen
        },
        "frozen_checkpoint_mae": {str(epoch): float(value) for epoch, value in HISTORICAL_EPOCH_MAE.items()},
        "official_test_loaded": False,
    }


def load_historical_curve(path: Path) -> list[dict[str, Any]]:
    """Read the frozen CSSD curve (columns may be shuffled; ``epoch`` required)."""
    import csv

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({"epoch": int(row["epoch"]), "valid_mae": float(row["valid_mae"])})
            except (KeyError, ValueError):
                return []
    return rows


# ---------------------------------------------------------------------------
# dictionary health (minimal frozen set)
# ---------------------------------------------------------------------------


def dictionary_health(
    *,
    model: cssd.CSSDModel,
    phi_valid: np.ndarray,
    reference_D: np.ndarray | None = None,
    fork_D: np.ndarray | None = None,
    chunk: int = 65536,
) -> dict[str, Any]:
    """Minimal frozen health set on official valid (preregistration section 9)."""
    cssd.cpu_only_guard(torch.device("cpu"))
    alpha = cssd.residual_codes(model, phi_valid, chunk=int(chunk))
    usage = cssd.usage_payload(alpha)
    D_bar = np.asarray(model.residual_dictionary().detach().double().numpy(), dtype=np.float64)
    D_raw = np.asarray(model.D.detach().double().numpy(), dtype=np.float64)
    U = np.asarray(model.U.detach().double().numpy(), dtype=np.float64)
    phi = np.asarray(phi_valid, dtype=np.float64)
    residual = phi - (phi @ U) @ U.T
    reconstruction = dca.reconstruction_metrics(residual, alpha, D_bar)
    payload = {
        "active_atoms": int(usage["active_atoms"]),
        "effective_atoms": float(usage["effective_atoms"]),
        "top5_usage": float(usage["top5_share"]),
        "top8_usage": float(usage["top8_share"]),
        "max_activation_rate": float(usage["max_activation_rate"]),
        "reconstruction_frobenius": float(reconstruction["recon_frobenius"]),
        "reconstruction_mean_row_squared": float(reconstruction["recon_mean_row_squared"]),
        "dictionary_frobenius_norm": float(np.linalg.norm(D_raw)),
        "dictionary_movement_vs_init": (
            float(np.linalg.norm(D_raw - np.asarray(reference_D, dtype=np.float64)))
            if reference_D is not None
            else None
        ),
        "dictionary_movement_vs_fork": (
            float(np.linalg.norm(D_raw - np.asarray(fork_D, dtype=np.float64)))
            if fork_D is not None
            else None
        ),
        "official_test_loaded": False,
    }
    official_test_blocker(payload)
    return payload


def dictionary_finite(model: cssd.CSSDModel) -> bool:
    return bool(torch.isfinite(model.D).all())


def dictionary_moved(model: cssd.CSSDModel, reference: np.ndarray) -> bool:
    current = np.asarray(model.D.detach().cpu(), dtype=np.float64)
    return bool(np.linalg.norm(current - np.asarray(reference, dtype=np.float64)) > 0.0)


# ---------------------------------------------------------------------------
# training-dynamics and comparison tables
# ---------------------------------------------------------------------------


def dynamics_rows(
    control_curve: Sequence[Mapping[str, Any]],
    low_curve: Sequence[Mapping[str, Any]],
    *,
    epochs: Sequence[int] = (280, 285, 290, 300, 310, 320),
) -> list[dict[str, Any]]:
    control = {int(row["epoch"]): row for row in control_curve}
    low = {int(row["epoch"]): row for row in low_curve}
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        if epoch not in control or epoch not in low:
            continue
        rows.append(
            {
                "epoch": int(epoch),
                "control_valid_mae": float(control[epoch]["valid_mae"]),
                "low_lr_valid_mae": float(low[epoch]["valid_mae"]),
                "difference": float(control[epoch]["valid_mae"]) - float(low[epoch]["valid_mae"]),
                "control_update_norm": float(control[epoch]["update_norm_global"]),
                "low_lr_update_norm": float(low[epoch]["update_norm_global"]),
                "control_grad_norm": float(control[epoch]["grad_norm_global"]),
                "low_lr_grad_norm": float(low[epoch]["grad_norm_global"]),
                "control_train_mae": float(control[epoch]["train_mae"]),
                "low_lr_train_mae": float(low[epoch]["train_mae"]),
            }
        )
    return rows


def late_slope(curve: Sequence[Mapping[str, Any]], window: tuple[int, int] = (311, 320)) -> float:
    rows = [row for row in curve if int(window[0]) <= int(row["epoch"]) <= int(window[1])]
    if len(rows) < 2:
        return float("nan")
    x = np.asarray([float(row["epoch"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["valid_mae"]) for row in rows], dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def arm_summary(
    *,
    prefix_payload: Mapping[str, Any],
    continuation_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Full 1-320 arm summary: merged curve statistics plus the merged soup."""
    curve = list(prefix_payload["curve"]) + list(continuation_payload["curve"])
    best = min(curve, key=lambda row: (float(row["valid_mae"]), int(row["epoch"])))
    last10 = [float(row["valid_mae"]) for row in curve[-10:]]
    final = curve[-1]
    soup = continuation_payload["soup"]
    tail = continuation_payload["tail_soup"]
    return {
        "soup_valid_mae": float(soup["soup_valid_mae"]),
        "soup_members": list(soup["members"]),
        "soup_member_valid_mae": list(soup["member_valid_mae"]),
        "soup_members_after_280": list(soup["members_after_280"]),
        "n_soup_members_after_280": int(soup["n_members_after_280"]),
        "tail_soup_valid_mae": (float(tail["soup_valid_mae"]) if tail else None),
        "tail_soup_members": (list(tail["members"]) if tail else None),
        "best_valid_mae": float(best["valid_mae"]),
        "best_epoch": int(best["epoch"]),
        "last10_mean_valid_mae": float(np.mean(last10)),
        "epoch320_valid_mae": float(final["valid_mae"]),
        "epoch320_train_mae": float(final["train_mae"]),
        "epoch320_task_loss": float(final["train_task_loss"]),
        "epoch320_rec_term": float(final["train_rec_term"]),
        "epoch320_total_loss": float(final["train_total_loss"]),
        "generalization_gap_epoch320": float(final["valid_mae"]) - float(final["train_mae"]),
        "late_slope_311_320": late_slope(curve),
        "wall_clock_s": float(prefix_payload["wall_clock_s"]) + float(continuation_payload["wall_clock_s"]),
        "epochs_run": int(len(curve)),
        "official_test_loaded": False,
    }


def task_vs_valid_delta(
    prefix_payload: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    """Train/valid classification at epoch 280 -> 320 for one arm."""
    final_prefix = prefix_payload["curve"][-1]
    train_delta = float(arm["epoch320_train_mae"]) - float(final_prefix["train_mae"])
    valid_delta = float(arm["epoch320_valid_mae"]) - float(final_prefix["valid_mae"])
    if (not math.isfinite(train_delta)) or (not math.isfinite(valid_delta)):
        reading = "non_finite"
    elif valid_delta < 0.0 and train_delta < 0.0:
        reading = "joint_improvement_late_optimization"
    elif valid_delta < 0.0 <= train_delta:
        reading = "validation_improves_train_worsens_regularisation_or_stability"
    elif valid_delta >= 0.0 and train_delta < 0.0:
        reading = "overfitting_or_noise"
    else:
        reading = "both_worse"
    return {
        "epoch280_train_mae": float(final_prefix["train_mae"]),
        "epoch280_valid_mae": float(final_prefix["valid_mae"]),
        "epoch320_train_mae": float(arm["epoch320_train_mae"]),
        "epoch320_valid_mae": float(arm["epoch320_valid_mae"]),
        "train_mae_delta_280_to_320": train_delta,
        "valid_mae_delta_280_to_320": valid_delta,
        "reading": reading,
        "official_test_loaded": False,
    }


def reconstruction_balance(
    prefix_payload: Mapping[str, Any], arm: Mapping[str, Any]
) -> dict[str, Any]:
    """Frozen lambda_rec check: task vs reconstruction loss at 280 and 320."""
    final_prefix = prefix_payload["curve"][-1]
    return {
        "lambda_rec": float(prefix_payload["lambda_rec"]),
        "epoch280_task_loss": float(final_prefix["train_task_loss"]),
        "epoch320_task_loss": float(arm["epoch320_task_loss"]),
        "task_loss_delta": float(arm["epoch320_task_loss"]) - float(final_prefix["train_task_loss"]),
        "epoch280_rec_term": float(final_prefix["train_rec_term"]),
        "epoch320_rec_term": float(arm["epoch320_rec_term"]),
        "rec_term_delta": float(arm["epoch320_rec_term"]) - float(final_prefix["train_rec_term"]),
        "official_test_loaded": False,
    }


# ---------------------------------------------------------------------------
# frozen decision (preregistration section 7)
# ---------------------------------------------------------------------------


def training_protocol_decision(
    *, m_control: float, m_low_lr: float, low_lr_best_epoch: int, low_lr_late_slope: float
) -> dict[str, Any]:
    """Frozen decision thresholds and the single reported verdict."""
    g = float(m_control) - float(m_low_lr)
    case_a = bool(g < CASE_A_MAX)
    directional = bool(CASE_A_MAX <= g < CASE_B_MIN)
    case_b = bool(g >= CASE_B_MIN)
    case_c = bool(g >= CASE_C_MIN or float(m_low_lr) <= CASE_C_ABS)
    case_d = bool(float(m_low_lr) <= CASE_D_ABS)
    negative = bool(g < 0.0)
    if case_d:
        verdict = "BASELINE_UNDEROPTIMIZATION_WAS_SUBSTANTIAL"
    elif case_c:
        verdict = "TRAINING_PROTOCOL_MAJOR_FACTOR"
    elif case_b:
        verdict = "LOW_LR_TAIL_MATERIALLY_SUPPORTED"
    elif directional:
        verdict = "DIRECTIONAL_SMALL_TRAINING_EFFECT"
    elif negative:
        verdict = "LOW_LR_TAIL_HARMFUL"
    else:
        verdict = "TRAINING_PROTOCOL_NOT_PRIMARY_BOTTLENECK"
    adopt = bool(case_b or case_c or case_d)
    horizon_binding = bool(int(low_lr_best_epoch) == TOTAL_EPOCHS or float(low_lr_late_slope) < 0.0)
    return {
        "m_control": float(m_control),
        "m_low_lr": float(m_low_lr),
        "g_schedule": g,
        "g_over_noise_floor": float(g / NOISE_FLOOR),
        "noise_floor": float(NOISE_FLOOR),
        "case_a_no_material": case_a,
        "directional_small": directional,
        "case_b_material": case_b,
        "case_c_major": case_c,
        "case_d_new_band": case_d,
        "negative_effect": negative,
        "verdict": verdict,
        "canonical_protocol": (
            "280@1e-3 + 40@1e-4" if adopt else "320@1e-3"
        ),
        "adopt_low_lr_tail": adopt,
        "horizon_may_still_be_binding": horizon_binding,
        "low_lr_best_epoch": int(low_lr_best_epoch),
        "low_lr_late_slope_311_320": float(low_lr_late_slope),
        "thresholds": {
            "case_a_max": CASE_A_MAX,
            "case_b_min": CASE_B_MIN,
            "case_c_min": CASE_C_MIN,
            "case_c_abs": CASE_C_ABS,
            "case_d_abs": CASE_D_ABS,
        },
        "historical_references": {
            "cssd_q1_seed0_soup": HISTORICAL_SOUP_MAE,
            "v2_low_rate_warm_soup": V2_LOW_RATE_WARM_SOUP,
            "final_clean_seed0_soup": FINAL_CLEAN_SOUP,
        },
        "official_test_loaded": False,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "PREFIX_EPOCHS",
    "TAIL_EPOCHS",
    "TOTAL_EPOCHS",
    "FORK_EPOCH",
    "CONTROL_LR",
    "LOW_LR",
    "SEED",
    "SOUP_K",
    "SPARSITY",
    "AUDIT_TAG",
    "AUDIT_SPEC",
    "AUDIT_MASK",
    "EXPECTED_PARAMETERS",
    "MODULE_GROUPS",
    "GROUP_PREFIXES",
    "HISTORICAL_SOUP_MAE",
    "HISTORICAL_SOUP_MEMBERS",
    "HISTORICAL_BEST_MAE",
    "HISTORICAL_BEST_EPOCH",
    "HISTORICAL_EPOCH_MAE",
    "V2_LOW_RATE_WARM_SOUP",
    "FINAL_CLEAN_SOUP",
    "PREFIX_MAE_TOLERANCE",
    "PREFIX_MAE_MAX",
    "NOISE_FLOOR",
    "CASE_A_MAX",
    "CASE_B_MIN",
    "CASE_C_MIN",
    "CASE_C_ABS",
    "CASE_D_ABS",
    "CATASTROPHIC_MAE",
    "official_test_blocker",
    "cpu_only_guard",
    "optimizer_state_sha256",
    "optimizer_state_comparison",
    "state_fingerprint",
    "rng_fingerprint",
    "generator_fingerprint",
    "tag_graph_ids",
    "replay_epoch_batch_order_hash",
    "parameter_groups",
    "empty_keeper",
    "seed_keeper",
    "retain_epoch",
    "keeper_members",
    "keeper_member_mae",
    "soup_state_from",
    "run_epochs",
    "fork_integrity",
    "prefix_gate",
    "historical_comparison",
    "load_historical_curve",
    "dictionary_health",
    "dictionary_finite",
    "dictionary_moved",
    "dynamics_rows",
    "late_slope",
    "arm_summary",
    "task_vs_valid_delta",
    "reconstruction_balance",
    "training_protocol_decision",
]
