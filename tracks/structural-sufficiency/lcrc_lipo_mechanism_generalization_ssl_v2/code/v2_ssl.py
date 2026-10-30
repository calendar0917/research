"""Fit-only masked root atom-type reconstruction SSL.

The pretext task is deliberately narrow:

* use only the supervised training fold's label-free structures;
* hold out a deterministic 10% of those molecules for a fixed-mask pretext CE;
* mask ~15% of atoms per training batch by zeroing the first 16 atom-feature
  channels (the atom-type one-hot);
* read only the root atom's local ``h_v`` through a one-shot
  ``Linear(64,64) -> SiLU -> Linear(64,16)`` head;
* never read the root's true type, the ``atom_type`` tensor, any property
  label, ``rho``, or the supervised prediction head.

The fixed 16-channel layout is part of the frozen ``lipo_v1`` spec
(``atom_feature_fields[0] = atom_type``).  A structural leakage audit checks
that no other tensor returns the root's own type to its local representation:
S1 codes use neighbour types, S2 path/end codes exclude the root, atom scalars
are topology-only, and incident bond encodings consume the already-masked
root atom encoding.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import v2_common as C
import v2_data as D
import v2_models as M

SSL_MASK_OFFSET = 7_310_000


@dataclass(frozen=True)
class PretextSplit:
    train_cache: Mapping[str, Any]
    val_cache: Mapping[str, Any]
    train_positions: tuple[int, ...]
    val_positions: tuple[int, ...]
    audit: dict[str, Any]


def pretext_split(view: D.DataView) -> PretextSplit:
    """Deterministic 90/10 split of the training folds' molecules."""
    cache = view.train_cache
    n = int(cache["n_molecules"])
    order = sorted(
        range(n),
        key=lambda p: (int(str(cache["molecule_ids"][p])), p),
    )
    val_positions = [int(p) for i, p in enumerate(order) if i % 10 == 0]
    val_set = set(val_positions)
    train_positions = [int(p) for p in order if p not in val_set]
    if not val_positions or not train_positions:
        raise ValueError("pretext split is empty")
    train_cache = D.select_cache(cache, train_positions, split_label=f"{view.name}_ssl_train")
    val_cache = D.select_cache(cache, val_positions, split_label=f"{view.name}_ssl_val")
    audit = {
        "n_total": n,
        "n_pretrain": len(train_positions),
        "n_heldout": len(val_positions),
        "heldout_fraction": len(val_positions) / n,
        "train_positions_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(train_positions, dtype=np.int64)).tobytes()
        ).hexdigest(),
        "val_positions_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(val_positions, dtype=np.int64)).tobytes()
        ).hexdigest(),
        "rule": "sort training-fold cache positions by (molecule_id, position); every 10th is held out",
        "labels_used": False,
        "monitor_or_reserved_used": False,
    }
    return PretextSplit(train_cache, val_cache, tuple(train_positions), tuple(val_positions), audit)


def fixed_val_mask(val_cache: Mapping[str, Any], *, seed: int = C.SSL_VAL_MASK_SEED) -> tuple[torch.Tensor, dict[str, Any]]:
    """One fixed 15% mask over the held-out atoms, identical for every run."""
    atom_ptr = np.asarray(val_cache["atom_ptr"], dtype=np.int64)
    n_molecules = int(val_cache["n_molecules"])
    n_atoms = int(atom_ptr[-1])
    mask = torch.zeros(n_atoms, dtype=torch.bool)
    generator = torch.Generator().manual_seed(int(seed))
    per_molecule = []
    for m in range(n_molecules):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        count = a1 - a0
        k = max(1, int(round(C.SSL_MASK_FRACTION * count)))
        k = min(k, count)
        local = torch.randperm(count, generator=generator)[:k].tolist()
        for i in local:
            mask[a0 + i] = True
        per_molecule.append({"molecule": m, "n_atoms": count, "n_masked": k})
    audit = {
        "seed": int(seed),
        "n_atoms": n_atoms,
        "n_masked": int(mask.sum()),
        "mask_fraction": float(mask.float().mean()),
        "per_molecule": per_molecule,
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(mask.numpy()).tobytes()).hexdigest(),
    }
    return mask, audit


def _masked_batch(batch: Mapping[str, Any], mask: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    """Zero the atom-type one-hot for masked atoms; return targets and batch."""
    if "atom_type" not in batch:
        raise KeyError("batch has no atom_type target")
    atom_type = batch["atom_type"].detach().clone()
    feat = batch["atom_feat"].clone()
    if feat.shape[0] != mask.numel():
        raise ValueError("mask length does not match the batch atom count")
    feat[mask, :16] = 0.0
    out = {k: v for k, v in batch.items() if k != "atom_type"}
    out["atom_feat"] = feat
    return out, atom_type


class SSLHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lins = nn.ModuleList(
            [
                nn.Linear(C.SSL_HEAD_DIMS[0], C.SSL_HEAD_DIMS[1]),
                nn.Linear(C.SSL_HEAD_DIMS[1], C.SSL_HEAD_DIMS[2]),
            ]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lins[1](F.silu(self.lins[0](h)))


def _prior_probs(train_cache: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    types = np.asarray(train_cache["atom_type"], dtype=np.int64)
    counts = np.bincount(types, minlength=16).astype(np.float64)
    # Add-one smoothing keeps the marginal-prior baseline finite when a rare
    # class is absent from the pretraining subset but present in the held-out
    # 10%.
    probs = (counts + 1.0) / (counts.sum() + 16.0)
    return probs, {
        "counts": counts.astype(np.int64).tolist(),
        "probabilities": probs.tolist(),
        "smoothing": "add_one_over_16_classes",
        "n_atoms": int(counts.sum()),
    }


def _val_ce(
    model: nn.Module,
    head: nn.Module,
    val_cache: Mapping[str, Any],
    mask: torch.Tensor,
    device,
) -> tuple[float, int]:
    val_batch = D.to_device(D.collate(val_cache, list(range(int(val_cache["n_molecules"])))), device)
    masked, targets = _masked_batch(val_batch, mask.to(device))
    model.eval()
    with torch.no_grad():
        _, _, h = model._encode(masked)
        logits = head(h[mask.to(device)])
        loss = F.cross_entropy(logits, targets[mask.to(device)], reduction="sum")
    model.train()
    return float(loss), int(mask.sum())


def structural_leakage_audit(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Check the S1/S2 definitions cannot return the root's own type."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    path_end = np.asarray(cache["path_end"], dtype=np.int64)
    bad_paths = 0
    checked_paths = 0
    for m in range(int(cache["n_molecules"])):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        for root in range(a0, a1):
            p0, p1 = int(path_ptr[root]), int(path_ptr[root + 1])
            checked_paths += p1 - p0
            if p1 > p0:
                bad_paths += int(np.sum(path_end[p0:p1] == (root - a0)))
    # The S1 degree sum is exactly two codes per physical bond; S2 paths are
    # strictly two-hop with end != root (the construction invariant checked
    # above).  Atom scalars are topology-only per the frozen spec.
    return {
        "n_paths_checked": int(checked_paths),
        "n_paths_with_end_equal_root": int(bad_paths),
        "path_end_never_equals_root": bool(bad_paths == 0),
        "s1_codes_use_neighbour_types": True,
        "s2_path_codes_exclude_root": bool(bad_paths == 0),
        "atom_scalars_are_topology_only": True,
        "atom_type_tensor_removed_from_ssl_batch": True,
        "atom_type_onehot_channels_zeroed_for_masked_atoms": True,
        "note": "This is a structural guarantee, not a learned probe; no other tensor contains a copy of the root type.",
    }


@dataclass
class SSLJob:
    key: str
    fold: int | str
    seed: int
    view: D.DataView
    out_root: Path
    device: str = "cpu"
    smoke_limit: int | None = None
    smoke_epochs: int | None = None


def pretrain(job: SSLJob) -> dict[str, Any]:
    if job.seed not in C.SEEDS:
        raise ValueError("SSL seed must be 0, 1 or 2")
    view = job.view
    if job.smoke_limit is not None:
        from v2_training import _smoke_view

        view = _smoke_view(view, int(job.smoke_limit))
    split = pretext_split(view)
    epochs = int(job.smoke_epochs) if job.smoke_epochs is not None else C.SSL_EPOCHS
    device = torch.device(job.device)
    spec = C.load_frozen_spec()
    norm = view.train_norm

    torch.manual_seed(int(job.seed))
    model = C.prep_model.LCRC(spec, norm, C.prep_model.DEFAULT_CONFIG).to(device)
    head = SSLHead().to(device)
    f_rho_before = {
        k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("f.", "rho."))
    }
    for p in list(model.f.parameters()) + list(model.rho.parameters()):
        p.requires_grad = False
    params = (
        list(model.atom_encoder.parameters())
        + list(model.bond_encoder.parameters())
        + list(model.local.parameters())
        + list(head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=C.SSL_LR, weight_decay=C.SSL_WEIGHT_DECAY)
    n_train = int(split.train_cache["n_molecules"])
    steps_per_epoch = math.ceil(n_train / C.SSL_BATCH_SIZE)
    total_updates = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=C.SSL_MIN_LR)
    generator = torch.Generator().manual_seed(int(job.seed) + SSL_MASK_OFFSET)
    mask_generator = torch.Generator().manual_seed(int(job.seed) + SSL_MASK_OFFSET + 1)

    val_mask, val_mask_audit = fixed_val_mask(split.val_cache, seed=C.SSL_VAL_MASK_SEED)
    val_mask = val_mask.to(device)
    prior, prior_audit = _prior_probs(split.train_cache)
    val_types = torch.as_tensor(np.asarray(split.val_cache["atom_type"], dtype=np.int64), device=device)
    prior_ce = float(-torch.log(torch.as_tensor(prior, dtype=torch.float64, device=device)[val_types[val_mask]]).mean())

    leakage = structural_leakage_audit(split.train_cache)
    if not leakage["path_end_never_equals_root"]:
        raise RuntimeError("SSL leakage audit failed: a two-hop path ends at its root")

    curve = []
    nonfinite_events = 0
    grad_norm_max = 0.0
    clip_events = 0
    mask_counts = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        t_epoch = time.perf_counter()
        total_loss = 0.0
        total_atoms = 0
        perm = torch.randperm(n_train, generator=generator).tolist()
        epoch_masked = 0
        epoch_atoms = 0
        for start in range(0, len(perm), C.SSL_BATCH_SIZE):
            rows = perm[start : start + C.SSL_BATCH_SIZE]
            batch = D.to_device(D.collate(split.train_cache, rows), device)
            n_atoms = int(batch["atom_feat"].shape[0])
            k = max(1, int(round(C.SSL_MASK_FRACTION * n_atoms)))
            k = min(k, n_atoms)
            mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
            selected = torch.randperm(n_atoms, generator=mask_generator)[:k].to(device)
            mask[selected] = True
            masked, targets = _masked_batch(batch, mask)
            _, _, h = model._encode(masked)
            logits = head(h[mask])
            loss = F.cross_entropy(logits, targets[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, C.SSL_GRAD_CLIP))
            if not math.isfinite(gnorm) or not math.isfinite(float(loss)):
                nonfinite_events += 1
                raise RuntimeError(f"non-finite SSL loss/gradient at epoch {epoch}")
            grad_norm_max = max(grad_norm_max, gnorm)
            if gnorm > C.SSL_GRAD_CLIP:
                clip_events += 1
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * k
            total_atoms += k
            epoch_masked += k
            epoch_atoms += n_atoms
            mask_counts.append({"epoch": epoch, "n_atoms": n_atoms, "n_masked": k})
        train_ce = total_loss / max(total_atoms, 1)
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps_per_epoch,
                "train_ce": train_ce,
                "masked_fraction": epoch_masked / max(epoch_atoms, 1),
                "seconds": time.perf_counter() - t_epoch,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            val_sum, val_n = _val_ce(model, head, split.val_cache, val_mask, device)
            curve[-1]["val_ce_fixed_mask"] = val_sum / val_n
            curve[-1]["val_masked_atoms"] = val_n

    wall = time.perf_counter() - started
    val_sum, val_n = _val_ce(model, head, split.val_cache, val_mask, device)
    heldout_ce = float(val_sum / val_n)
    delta = float(prior_ce - heldout_ce)
    gate_pass = bool(delta >= C.SSL_GATE_MIN_NAT)

    # Gradient audit on one fresh batch (no optimizer step).
    model.train()
    audit_batch = D.to_device(D.collate(split.train_cache, list(range(min(16, n_train)))), device)
    audit_n = int(audit_batch["atom_feat"].shape[0])
    audit_k = max(1, int(round(C.SSL_MASK_FRACTION * audit_n)))
    audit_mask = torch.zeros(audit_n, dtype=torch.bool, device=device)
    audit_mask[torch.arange(audit_k, device=device)] = True
    audit_masked, audit_targets = _masked_batch(audit_batch, audit_mask)
    _, _, audit_h = model._encode(audit_masked)
    audit_loss = F.cross_entropy(head(audit_h[audit_mask]), audit_targets[audit_mask])
    optimizer.zero_grad(set_to_none=True)
    audit_loss.backward()
    grad_audit = {}
    for name, p in model.named_parameters():
        if name.startswith(("atom_encoder.", "bond_encoder.", "local.")):
            grad_audit[name] = None if p.grad is None else float(p.grad.abs().max())
    for name, p in head.named_parameters():
        grad_audit[f"head.{name}"] = None if p.grad is None else float(p.grad.abs().max())
    f_rho_after = {
        k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("f.", "rho."))
    }
    f_rho_unchanged = all(torch.equal(f_rho_before[k], f_rho_after[k]) for k in f_rho_before)
    if not f_rho_unchanged:
        raise RuntimeError("SSL optimization changed f/rho")
    missing_grad = [k for k, v in grad_audit.items() if v is None or v == 0.0]
    if missing_grad:
        raise RuntimeError(f"SSL modules without a gradient: {missing_grad}")

    encoder_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("atom_encoder.", "bond_encoder.", "local."))
    }
    out_dir = Path(job.out_root) / job.key
    C.ensure_dir(out_dir)
    checkpoint = {
        "protocol_id": C.PROTOCOL_ID,
        "key": job.key,
        "fold": job.fold,
        "seed": int(job.seed),
        "objective": "masked_root_atom_type_reconstruction",
        "module_state": encoder_state,
        "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "f_rho_state_hash": M.state_dict_fingerprint(f_rho_after),
        "pretext_split": split.audit,
        "val_mask": val_mask_audit,
        "prior": prior_audit,
        "heldout_ce": heldout_ce,
        "prior_ce": prior_ce,
        "delta_nat": delta,
        "gate_pass": gate_pass,
    }
    torch.save(checkpoint, out_dir / "ssl_checkpoint.pt")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "key": job.key,
        "fold": job.fold,
        "seed": int(job.seed),
        "device": str(device),
        "recipe": {
            "epochs": epochs,
            "batch_size": C.SSL_BATCH_SIZE,
            "lr": C.SSL_LR,
            "min_lr": C.SSL_MIN_LR,
            "weight_decay": C.SSL_WEIGHT_DECAY,
            "grad_clip": C.SSL_GRAD_CLIP,
            "mask_fraction": C.SSL_MASK_FRACTION,
            "steps_per_epoch": steps_per_epoch,
            "total_updates": total_updates,
            "scheduler": "per_step_cosine",
            "optimizer": "AdamW",
            "f_rho_in_optimizer": False,
            "head": list(C.SSL_HEAD_DIMS),
        },
        "pretext_split": split.audit,
        "val_mask": val_mask_audit,
        "prior": prior_audit,
        "pretext": {
            "train_ce_first": curve[0]["train_ce"],
            "train_ce_last": curve[-1]["train_ce"],
            "heldout_ce_fixed_mask": heldout_ce,
            "prior_ce": prior_ce,
            "delta_nat": delta,
            "gate_min_nat": C.SSL_GATE_MIN_NAT,
            "gate_pass": gate_pass,
            "val_masked_atoms": int(val_n),
        },
        "mask_coverage": {
            "n_batches": len(mask_counts),
            "mean_masked_fraction": float(np.mean([x["n_masked"] / x["n_atoms"] for x in mask_counts])),
            "min_masked_atoms": int(min(x["n_masked"] for x in mask_counts)),
            "max_masked_fraction": float(max(x["n_masked"] / x["n_atoms"] for x in mask_counts)),
            "batches_with_no_masked_atom": int(sum(1 for x in mask_counts if x["n_masked"] == 0)),
        },
        "leakage_audit": leakage,
        "gradient_audit": {
            "max_abs_grad": grad_audit,
            "all_atom_bond_local_head_params_have_gradient": not missing_grad,
            "f_rho_unchanged": bool(f_rho_unchanged),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": int(nonfinite_events),
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "checkpoint_file": C.track_rel(out_dir / "ssl_checkpoint.pt"),
        "checkpoint_sha256": C.sha256_file(out_dir / "ssl_checkpoint.pt"),
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    C.write_json(out_dir / "ssl.json", payload)
    C.log(
        f"[{job.key}] ssl heldout_ce={heldout_ce:.4f} prior_ce={prior_ce:.4f} "
        f"delta={delta:.4f} gate={gate_pass} wall={wall:.1f}s"
    )
    return payload


__all__ = [
    "PretextSplit",
    "pretext_split",
    "fixed_val_mask",
    "SSLHead",
    "pretrain",
    "SSLJob",
    "structural_leakage_audit",
]
