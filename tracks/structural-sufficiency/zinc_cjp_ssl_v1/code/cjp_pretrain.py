"""CJP-SSL pretraining: two matched arms (MARG / JOINT) on masked two-hop paths.

Frozen recipe: AdamW, LR 1e-3 constant, WD 1e-5, batch 128 molecules, clip 5,
7,560 updates, FP32, no dropout.  ``atom_encoder`` / ``bond_encoder`` are
frozen; only ``LocalContent`` + the throw-away pretext head receive gradients.
``f`` and ``rho`` do not participate in pretraining at all.

The final-step LocalContent weights are exported; the diagnostic structures
never select a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import cjp_common as C
import cjp_data as D
import cjp_models as CM

L = C.L


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return C.RUNS / f"pretrain_{prefix}{str(arm).lower()}_s{int(seed)}"


def _to_device(batch: Mapping[str, Any], query: Mapping[str, Any], device):
    b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
         for k, v in batch.items()}
    q = {k: v.to(device, non_blocking=True) for k, v in query.items()}
    return b, q


def targets_and_groups(topo: Mapping[str, Any], path_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Hidden 63-class target and the *visible* conditioning group per path.

    ``target = 3*type(u) + (bond2-1)``; ``group = 3*type(w) + (bond1-1)`` where
    ``w`` and ``bond1`` stay visible under the frozen mask.
    """
    path_ids = np.asarray(path_ids, dtype=np.int64)
    m = len(path_ids)
    targets = np.zeros(m, dtype=np.int64)
    groups = np.zeros(m, dtype=np.int64)
    for k, pid in enumerate(path_ids):
        u = int(topo["path_u"][pid])
        e2 = int(topo["path_e2"][pid])
        w = int(topo["path_w"][pid])
        e1 = int(topo["path_e1"][pid])
        targets[k] = C.N_BOND_TYPES * int(topo["x_raw"][u]) + int(topo["bond_type_raw"][e2]) - 1
        groups[k] = C.N_BOND_TYPES * int(topo["x_raw"][w]) + int(topo["bond_type_raw"][e1]) - 1
    return targets, groups


def collect_prior_targets(
    topo: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    targets, groups, per_mol = [], [], []
    for rid in indices:
        t = topo[rid]
        pid = D.sample_paths(t, seed=seed, molecule_id=rid, visit_index=0)
        tg, gp = targets_and_groups(t, pid)
        targets.append(tg)
        groups.append(gp)
        per_mol.append(len(tg))
    return {
        "targets": np.concatenate(targets) if targets else np.zeros(0, np.int64),
        "groups": np.concatenate(groups) if groups else np.zeros(0, np.int64),
        "per_molecule": np.asarray(per_mol, dtype=np.int64),
    }


def fit_priors(targets: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Laplace-smoothed class prior and (type(w), bond1)-conditional prior."""
    n_cls = C.N_TARGET_CLASSES
    prior = np.full(n_cls, 1.0, dtype=np.float64)
    np.add.at(prior, targets, 1.0)
    prior /= prior.sum()
    cond = np.ones((n_cls, n_cls), dtype=np.float64)
    np.add.at(cond, (groups, targets), 1.0)
    cond /= cond.sum(axis=1, keepdims=True)
    return {"class_prior": prior, "conditional_prior": cond}


def prior_ce(targets: np.ndarray, groups: np.ndarray, priors: Mapping[str, np.ndarray]) -> dict[str, float]:
    if len(targets) == 0 or priors is None:
        return {"class_prior_ce": float("nan"), "conditional_prior_ce": float("nan")}
    cp = priors["class_prior"]
    cond = priors["conditional_prior"]
    return {
        "class_prior_ce": float(-np.log(cp[targets]).mean()),
        "conditional_prior_ce": float(-np.log(cond[groups, targets]).mean()),
    }


def build_diagnostic_visits(
    topo: Mapping[int, Mapping[str, Any]],
    indices: Sequence[int],
    *,
    seed: int,
    visit_index: int = 0,
) -> list[dict[str, Any]]:
    visits = []
    for rid in indices:
        v = D.prepare_visit(topo[rid], seed=seed, molecule_id=rid, visit_index=visit_index)
        _tg, gp = targets_and_groups(topo[rid], v["path_ids"])
        visits.append({
            "mol": v["mol"],
            "queries": v["queries"],
            "stats": v["stats"],
            "masked_sha": v["masked_sha"],
            "groups": gp,
        })
    return visits


@torch.no_grad()
def evaluate_visits(
    model: torch.nn.Module,
    head: torch.nn.Module,
    visits: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    device,
    priors: Mapping[str, np.ndarray] | None = None,
    chunk: int = 256,
    want_activations: bool = False,
) -> dict[str, Any]:
    model.eval()
    n = len(visits)
    if n == 0:
        return {"n_molecules": 0}
    mol_ce = np.full(n, np.nan, dtype=np.float64)
    ce_all, tg_all, gp_all = [], [], []
    rms_terms: list[float] = []
    hs = [] if want_activations else None
    for start in range(0, n, chunk):
        part = visits[start:start + chunk]
        batch, query = D.batch_from_molecules([v["mol"] for v in part],
                                              [v["queries"] for v in part])
        batch, query = _to_device(batch, query, device)
        used, raw, h = CM.pretext_logits(model, head, batch, query, arm=arm)
        ce = CM.per_query_ce(used, query["target"])
        sums = torch.zeros(len(part), device=device, dtype=ce.dtype)
        counts = torch.zeros(len(part), device=device, dtype=ce.dtype)
        sums = sums.index_add(0, query["graph"], ce)
        counts = counts.index_add(0, query["graph"], torch.ones_like(ce))
        for i in range(len(part)):
            if counts[i] > 0:
                mol_ce[start + i] = float(sums[i] / counts[i])
        ce_all.append(ce.detach().cpu())
        tg_all.append(query["target"].detach().cpu())
        gp_all.append(torch.from_numpy(
            np.concatenate([np.asarray(v["groups"], dtype=np.int64) for v in part])
            if any(len(v["groups"]) for v in part) else np.zeros(0, np.int64)
        ))
        if raw.shape[0]:
            rms_terms.append(float(CM.interaction_residual(raw).pow(2).mean().sqrt()))
        if want_activations:
            hs.append(h.detach().cpu())
    valid = ~np.isnan(mol_ce)
    out: dict[str, Any] = {
        "n_molecules": int(n),
        "n_molecules_with_paths": int(valid.sum()),
        "n_queries": int(sum(len(v["queries"]) for v in visits)),
        "molecule_mean_ce": float(np.nanmean(mol_ce)) if valid.any() else float("nan"),
        "interaction_rms": float(np.mean(rms_terms)) if rms_terms else 0.0,
        "path_stats": D.mask_coverage([v["stats"] for v in visits]),
    }
    if ce_all:
        ce_t = torch.cat(ce_all)
        out["query_mean_ce"] = float(ce_t.mean().item()) if len(ce_t) else float("nan")
        out.update(prior_ce(torch.cat(tg_all).numpy(), torch.cat(gp_all).numpy(), priors))
    if want_activations and hs:
        out["representation"] = CM.effective_rank(torch.cat(hs, dim=0))
    model.train()
    return out


def _grad_norms(module: torch.nn.Module, prefix: str = "") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, p in module.named_parameters():
        out[f"{prefix}{name}"] = None if p.grad is None else float(p.grad.detach().norm().item())
    return out


@torch.no_grad()
def interventions(
    model: torch.nn.Module,
    head: torch.nn.Module,
    visits: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    device,
    seed: int,
) -> dict[str, float]:
    """Swap h_v <-> h_w and cross-molecule-shuffle the query representations."""
    batch, query = D.batch_from_molecules([v["mol"] for v in visits],
                                          [v["queries"] for v in visits])
    batch, query = _to_device(batch, query, device)
    _used, raw, h = CM.pretext_logits(model, head, batch, query, arm=arm)
    ce = CM.per_query_ce(_used, query["target"])
    base, _, _ = CM.molecule_mean_loss(ce, query, batch["n_graphs"])

    hv = h[query["v"]]
    hw = h[query["w"]]
    q_swap = torch.cat([hw, hv, hv * hw], dim=1)
    logits_swap = head(q_swap)
    used_swap = CM.marg_projection(logits_swap) if arm == "MARG" else logits_swap
    loss_swap, _, _ = CM.molecule_mean_loss(
        CM.per_query_ce(used_swap, query["target"]), query, batch["n_graphs"])

    gen = torch.Generator().manual_seed(int(seed) + 424242)
    perm = torch.randperm(len(query["target"]), generator=gen).to(device)
    q_shuf = torch.cat([hv[perm], hw[perm], hv[perm] * hw[perm]], dim=1)
    logits_shuf = head(q_shuf)
    used_shuf = CM.marg_projection(logits_shuf) if arm == "MARG" else logits_shuf
    loss_shuf, _, _ = CM.molecule_mean_loss(
        CM.per_query_ce(used_shuf, query["target"]), query, batch["n_graphs"])
    return {
        "base_loss": float(base.item()),
        "swap_v_w_loss": float(loss_swap.item()),
        "swap_v_w_delta": float(loss_swap.item() - base.item()),
        "cross_molecule_shuffle_loss": float(loss_shuf.item()),
        "cross_molecule_shuffle_delta": float(loss_shuf.item() - base.item()),
        "mean_interaction_rms": float(CM.interaction_residual(raw).pow(2).mean().sqrt().item()),
    }


# ---------------------------------------------------------------------------
# main pretraining loop
# ---------------------------------------------------------------------------


def pretrain(
    arm: str,
    seed: int,
    device: str,
    *,
    updates: int | None = None,
    batch_size: int | None = None,
    max_fit: int | None = None,
    tag: str | None = None,
    out_dir: Path | None = None,
    force: bool = False,
    diag_limit: int | None = None,
    eval_subset: int = 1000,
) -> dict[str, Any]:
    if arm not in ("MARG", "JOINT"):
        raise ValueError(arm)
    L.require_cuda(device)
    out = Path(out_dir) if out_dir is not None else run_dir(arm, seed, tag)
    C.ensure_dir(out)
    payload_path = out / "pretrain.json"
    if payload_path.exists() and not force:
        L.log(f"[pretrain {arm} s{seed}] already complete: {payload_path}")
        return C.read_json(payload_path)

    recipe = dict(C.PRETRAIN)
    n_updates = int(recipe["updates"] if updates is None else updates)
    bs = int(recipe["batch_size_molecules"] if batch_size is None else batch_size)
    timing_run = bool(n_updates != recipe["updates"] or bs != recipe["batch_size_molecules"])

    t_start = time.perf_counter()
    # ---- data ------------------------------------------------------------
    raws = D.load_raw_train()
    split = D.structure_split(raws, n_fit=recipe["n_fit"])
    iso = D.verify_split_isomorphism(raws, split)
    fit_all = [int(i) for i in split["fit_index"]]
    diag_all = [int(i) for i in split["diag_index"]]
    fit = fit_all[: int(max_fit)] if max_fit is not None else fit_all
    diag = diag_all[: int(diag_limit)] if diag_limit is not None else diag_all
    L.log(f"[pretrain {arm} s{seed}] fit={len(fit)} diag={len(diag)} "
          f"(unique fingerprints {split['n_unique_fingerprints']})")
    topo: dict[int, dict[str, Any]] = {}
    t_data = time.perf_counter()
    all_needed = sorted(set(fit) | set(diag))
    for k, rid in enumerate(all_needed):
        topo[rid] = D.build_topology(raws[rid])
        if (k + 1) % 2500 == 0:
            L.log(f"[pretrain {arm} s{seed}] topology {k + 1}/{len(all_needed)} "
                  f"{time.perf_counter() - t_data:.1f}s")
    prep_seconds = time.perf_counter() - t_data

    # ---- priors (fit structures only) ------------------------------------
    prior_targets = collect_prior_targets(topo, fit, seed=seed)
    priors = fit_priors(prior_targets["targets"], prior_targets["groups"])

    # ---- fixed diagnostic / eval visits ----------------------------------
    diag_visits = build_diagnostic_visits(topo, diag, seed=seed)
    eval_rows = fit[: int(eval_subset)]
    eval_visits = build_diagnostic_visits(topo, eval_rows, seed=seed)

    # ---- model -----------------------------------------------------------
    norm = C.D.load_norm()
    model, head, model_info = CM.build_ssl_model(norm, seed, device)
    encoder_before = {
        "atom_encoder": C.state_hash(model.atom_encoder),
        "bond_encoder": C.state_hash(model.bond_encoder),
    }

    diag_before = evaluate_visits(model, head, diag_visits, arm=arm, device=device,
                                  priors=priors, want_activations=True)
    fit_before = evaluate_visits(model, head, eval_visits, arm=arm, device=device,
                                 priors=priors, want_activations=True)

    # ---- optimisation ----------------------------------------------------
    params = [p for p in list(model.local.parameters()) + list(head.parameters())
              if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(recipe["lr"]),
                                  weight_decay=float(recipe["weight_decay"]))
    opt_params = int(sum(p.numel() for g in optimizer.param_groups for p in g["params"]))
    expected = int(model_info["parameters_local"] + model_info["parameters_head"])
    if opt_params != expected:
        raise RuntimeError(f"optimizer covers {opt_params} != {expected}")

    ss = np.random.SeedSequence([int(seed), int(C.BATCH_SEED_TAG)])
    batch_rng = np.random.Generator(np.random.PCG64(ss))
    visit_counts = {rid: 0 for rid in fit}

    curve: list[dict[str, Any]] = []
    masked_hash = hashlib.sha256()
    first_batch_sha = None
    mask_totals = {"atoms": 0, "bonds": 0, "masked_atoms": 0, "masked_bonds": 0,
                   "paths": 0, "molecules": 0, "no_path": 0}
    grad_first: dict[str, Any] | None = None
    grad_last: dict[str, Any] | None = None
    window_loss: list[float] = []
    window_ce: list[float] = []
    n_updates_done = 0
    t_train = time.perf_counter()
    model.train()
    head.train()
    for update in range(n_updates):
        rows = batch_rng.permutation(len(fit))[:bs]
        mols, queries = [], []
        for pos in rows:
            rid = fit[int(pos)]
            t = topo[rid]
            vi = visit_counts[rid]
            visit_counts[rid] = vi + 1
            v = D.prepare_visit(t, seed=seed, molecule_id=rid, visit_index=vi)
            mols.append(v["mol"])
            queries.append(v["queries"])
            masked_hash.update(v["masked_sha"].encode())
            mask_totals["atoms"] += int(v["stats"]["n_atoms"])
            mask_totals["bonds"] += int(v["stats"]["n_bonds"])
            mask_totals["masked_atoms"] += int(v["stats"]["atoms_masked"])
            mask_totals["masked_bonds"] += int(v["stats"]["bonds_masked"])
            mask_totals["paths"] += int(v["stats"]["n_selected_paths"])
            mask_totals["molecules"] += 1
            mask_totals["no_path"] += int(v["stats"]["n_selected_paths"] == 0)
        if first_batch_sha is None:
            h0 = hashlib.sha256()
            for q in queries:
                h0.update(np.ascontiguousarray(q).tobytes())
            first_batch_sha = h0.hexdigest()
        batch, query = D.batch_from_molecules(mols, queries)
        batch, query = _to_device(batch, query, device)
        used, _raw, _h = CM.pretext_logits(model, head, batch, query, arm=arm)
        ce = CM.per_query_ce(used, query["target"])
        loss, _sums, _counts = CM.molecule_mean_loss(ce, query, batch["n_graphs"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite pretraining loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update == 0:
            grad_first = {
                "local": _grad_norms(model.local, "local."),
                "head": _grad_norms(head, "head."),
                "f": _grad_norms(model.f, "f."),
                "rho": _grad_norms(model.rho, "rho."),
                "atom_encoder": _grad_norms(model.atom_encoder, "atom_encoder."),
                "bond_encoder": _grad_norms(model.bond_encoder, "bond_encoder."),
            }
        gn = torch.nn.utils.clip_grad_norm_(params, float(recipe["grad_clip"]))
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient at update {update}")
        if update == n_updates - 1:
            grad_last = {
                "local": _grad_norms(model.local, "local."),
                "head": _grad_norms(head, "head."),
            }
        optimizer.step()
        n_updates_done += 1
        window_loss.append(float(loss.item()))
        window_ce.append(float(ce.mean().item()))
        if (update + 1) % 200 == 0 or update == 0 or update + 1 == n_updates:
            entry = {
                "update": int(update + 1),
                "window_loss": float(np.mean(window_loss)),
                "window_query_ce": float(np.mean(window_ce)),
                "grad_norm": float(gn),
            }
            curve.append(entry)
            L.log(f"[pretrain {arm} s{seed}] upd={update + 1}/{n_updates} "
                  f"loss={entry['window_loss']:.6f} ce={entry['window_query_ce']:.6f} "
                  f"gn={entry['grad_norm']:.3f}")
            window_loss, window_ce = [], []
    train_seconds = time.perf_counter() - t_train

    # ---- post-training diagnostics ---------------------------------------
    diag_after = evaluate_visits(model, head, diag_visits, arm=arm, device=device,
                                 priors=priors, want_activations=True)
    fit_after = evaluate_visits(model, head, eval_visits, arm=arm, device=device,
                                priors=priors, want_activations=True)
    inter = interventions(model, head, eval_visits[:128], arm=arm, device=device, seed=seed)
    encoder_after = {
        "atom_encoder": C.state_hash(model.atom_encoder),
        "bond_encoder": C.state_hash(model.bond_encoder),
    }
    encoder_bit_identical = encoder_before == encoder_after

    # ---- export ----------------------------------------------------------
    local_path = out / "local_state.pt"
    head_path = out / "pretext_head_state.pt"
    L.atomic_save({"state_dict": {k: v.detach().cpu().clone()
                                  for k, v in model.local.state_dict().items()},
                   "arm": arm, "seed": int(seed), "updates": int(n_updates_done)},
                  local_path)
    L.atomic_save({"state_dict": {k: v.detach().cpu().clone()
                                  for k, v in head.state_dict().items()},
                   "arm": arm, "seed": int(seed), "updates": int(n_updates_done)},
                  head_path)
    local_sha = C.state_hash(model.local)
    head_sha = C.state_hash(head)

    wall = time.perf_counter() - t_start
    coverage = {
        "atom_mask_fraction": mask_totals["masked_atoms"] / max(mask_totals["atoms"], 1),
        "bond_mask_fraction": mask_totals["masked_bonds"] / max(mask_totals["bonds"], 1),
        "paths_per_molecule": mask_totals["paths"] / max(mask_totals["molecules"], 1),
        "molecules_with_no_path": int(mask_totals["no_path"]),
    }
    final_window = float(np.mean([c["window_query_ce"] for c in curve[-3:]])) if curve else float("nan")
    payload = {
        "round": C.ROUND_ID,
        "arm": arm,
        "seed": int(seed),
        "status": "completed",
        "timing_run": bool(timing_run),
        "git_commit": C.git_commit(),
        "environment": L.env_summary(),
        "data": {
            "source": "official PyG ZINC subset=True train raw x/edge_index/edge_attr",
            "labels_read": False,
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "n_molecules_available": int(len(raws)),
            "n_fit": int(len(fit)),
            "n_diag": int(len(diag)),
            "split": {
                "n_unique_fingerprints": int(split["n_unique_fingerprints"]),
                "n_duplicate_groups": int(split["n_duplicate_groups"]),
                "n_molecules_in_duplicate_groups": int(split["n_molecules_in_duplicate_groups"]),
                "fingerprint_sha256": split["fingerprint_sha"],
                "fingerprint_order_sha256": split["fingerprint_order_sha"],
                "fit_index_sha256": C.sha256_bytes(np.asarray(fit_all, np.int64).tobytes()),
                "diag_index_sha256": C.sha256_bytes(np.asarray(diag_all, np.int64).tobytes()),
                "isomorphism_check": iso,
                "n_fit_full": len(fit_all),
                "n_diag_full": len(diag_all),
            },
            "masked_batch_sha256": masked_hash.hexdigest(),
            "first_batch_query_sha256": first_batch_sha,
            "topology_prep_seconds": float(prep_seconds),
        },
        "protocol": {
            "mask_rng": "SeedSequence([seed, molecule_id, visit_index, 20261029])",
            "batch_rng": "SeedSequence([seed, 20261030])",
            "max_paths_per_molecule": C.MAX_PATHS_PER_MOLECULE,
            "atom_mask_token": C.ATOM_MASK_TOKEN,
            "bond_mask_token": C.BOND_MASK_TOKEN,
            "target": "3*raw_atom_type(u) + (raw_bond_type(w,u)-1) in [0,62]",
            "loss": recipe["path_loss"],
            "marg": "L_marg[a,b] = mean_b L + mean_a L - mean_ab L",
        },
        "model_info": model_info,
        "recipe": recipe,
        "optimizer_parameters": opt_params,
        "updates_done": int(n_updates_done),
        "prior_stats": {
            "n_fit_queries": int(len(prior_targets["targets"])),
            "class_counts": np.bincount(prior_targets["targets"],
                                        minlength=C.N_TARGET_CLASSES).tolist(),
            "group_counts": np.bincount(prior_targets["groups"],
                                        minlength=C.N_TARGET_CLASSES).tolist(),
            "laplace_smoothing": 1.0,
            "class_prior_ce_fit": float(
                -np.log(priors["class_prior"][prior_targets["targets"]]).mean()),
            "conditional_prior_ce_fit": float(
                -np.log(priors["conditional_prior"][prior_targets["groups"],
                                                    prior_targets["targets"]]).mean()),
        },
        "diagnostics_before": {"diag": diag_before, "fit_eval": fit_before},
        "diagnostics_after": {"diag": diag_after, "fit_eval": fit_after},
        "train": {
            "curve": curve,
            "final_window_query_ce": final_window,
            "grad_norms_first_update": grad_first,
            "grad_norms_last_update": grad_last,
            "mask_coverage": coverage,
            "mask_totals": mask_totals,
            "interventions": inter,
        },
        "encoder_sha256_before": encoder_before,
        "encoder_sha256_after": encoder_after,
        "encoder_bit_identical": bool(encoder_bit_identical),
        "local_state": {
            "path": C.track_rel(local_path),
            "sha256_file": C.sha256_file(local_path),
            "state_sha256": local_sha,
        },
        "pretext_head": {
            "path": C.track_rel(head_path),
            "sha256_file": C.sha256_file(head_path),
            "state_sha256": head_sha,
            "discarded": True,
        },
        "timing": {
            "wall_seconds": float(wall),
            "topology_seconds": float(prep_seconds),
            "train_seconds": float(train_seconds),
            "gpu_hours": C.gpu_hours(wall),
        },
    }
    C.write_json(payload_path, payload)
    L.log(f"[pretrain {arm} s{seed}] done wall={wall:.1f}s local_sha={local_sha[:12]} "
          f"diag_ce={diag_after['molecule_mean_ce']:.6f}")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CJP-SSL pretraining")
    parser.add_argument("--arm", required=True, choices=["MARG", "JOINT"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-fit", type=int, default=None)
    parser.add_argument("--diag-limit", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    res = pretrain(
        args.arm,
        args.seed,
        args.device,
        updates=args.updates,
        batch_size=args.batch_size,
        max_fit=args.max_fit,
        diag_limit=args.diag_limit,
        tag=args.tag,
        out_dir=None if args.out is None else Path(args.out),
        force=bool(args.force),
    )
    L.log(f"{res['arm']} s{res['seed']}: diag CE "
          f"{res['diagnostics_after']['diag']['molecule_mean_ce']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
