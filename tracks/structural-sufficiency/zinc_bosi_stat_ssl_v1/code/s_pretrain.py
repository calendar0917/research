"""Pretraining driver: statistics-only masked SSL for BOSI-B.

One seed, one fixed budget, no checkpoint selection on reconstruction accuracy
(the last-step state is the exported initialisation, exactly as in
``bjoc_masked_pretraining_v1``).

    python code/s_pretrain.py --seed 0 --device cuda:0 --mode pilot   # 1,890 updates
    python code/s_pretrain.py --seed 0 --device cuda:0 --mode full    # 7,560 updates

The pretraining process never receives the label array: it is handed the
structural cache only (``s_common.structural_cache``).  All masks come from
``SeedSequence([seed, molecule, visit_index])`` and are independent of the data
shuffle ``torch.Generator(seed + 202)``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402
import s_mask as SM  # noqa: E402
import s_pretext as SP  # noqa: E402
import ad_common as C  # noqa: E402

PROBE_VISIT = 1_000_000      # disjoint from every training visit index (1..120)
HEAD_SEED_OFFSET = 303
GRAD_STEPS = (1, 100, 630)   # plus the final step


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_topology(cache) -> tuple[dict, dict]:
    """Build (or load) the type-independent topology cache of the split."""
    SC.CACHE.mkdir(parents=True, exist_ok=True)
    path = SC.CACHE / "stat_ssl_topo_train.pt"
    if path.exists():
        topo = torch.load(path, map_location="cpu", weights_only=False)
        h = SM.topology_hash(topo)
        if h == topo.get("sha256"):
            return topo, {"source": SC.rel_or_abs(path), "reused": True, "sha256": h}
    t0 = time.perf_counter()
    topo = SM.build_topology(cache)
    topo["sha256"] = SM.topology_hash(topo)
    C.atomic_torch_save(topo, path)
    return topo, {"source": SC.rel_or_abs(path), "reused": False,
                  "sha256": topo["sha256"],
                  "build_seconds": float(time.perf_counter() - t0)}


def marginal_baselines(cache, rows: np.ndarray) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    a_counts = np.zeros(28, dtype=np.float64)
    b_counts = np.zeros(4, dtype=np.float64)
    for r in rows:
        a_counts += np.bincount(at[ap[r]:ap[r + 1]], minlength=28)
        b_counts += np.bincount(bt[bp[r]:bp[r + 1]], minlength=4)
    out = {}
    for name, counts in (("atom", a_counts), ("bond", b_counts)):
        p = counts / max(counts.sum(), 1.0)
        nz = p[p > 0]
        out[name] = {
            "n": int(counts.sum()),
            "counts": counts.astype(int).tolist(),
            "entropy_nats": float(-(nz * np.log(nz)).sum()),
            "marginal_ce_nats": float(-(nz * np.log(nz)).sum()),
            "majority_class": int(counts.argmax()),
            "majority_fraction": float(p.max()),
            "non_majority_fraction": float(1.0 - p.max()),
        }
    return out


def probe_batch(cache, rel, topo):
    rows = SC.probe_rows()
    mb = SM.masked_batch(cache, rel, topo, rows, seed=0, visit_index=PROBE_VISIT)
    return rows, mb


@torch.no_grad()
def evaluate(model, heads, cache, rel, topo, device, *, seed: int) -> dict:
    model.eval()
    heads.eval()
    rows, mb = probe_batch(cache, rel, topo)
    batch = _to_device(mb["batch"], device)
    at = mb["atom_targets"].to(device)
    bt = mb["bond_targets"].to(device)
    out = SP.pretext_loss(model, heads, batch, at, bt)
    clean = {
        "atom": SP.class_report(out["atom_logits"], at, 28),
        "bond": SP.class_report(out["bond_logits"], bt, 4),
    }
    probes = {"clean": clean}
    with SP.zero_moment_block(model):
        z = SP.pretext_loss(model, heads, batch, at, bt)
    probes["zero_M"] = {
        "atom": SP.class_report(z["atom_logits"], at, 28),
        "bond": SP.class_report(z["bond_logits"], bt, 4),
    }
    sb = _to_device(SP.shuffle_partner_stats(batch, seed=12345), device)
    s = SP.pretext_loss(model, heads, sb, at, bt)
    probes["shuffled_partner"] = {
        "atom": SP.class_report(s["atom_logits"], at, 28),
        "bond": SP.class_report(s["bond_logits"], bt, 4),
    }
    model.train()
    heads.train()
    return {
        "probe_rows": int(len(rows)),
        "probe_visit_index": PROBE_VISIT,
        "probe_rows_sha256": C.sha256_array(rows.astype(np.float64).astype(np.float32)),
        "n_masked_atoms": int(at.numel()),
        "n_masked_bonds": int(bt.numel()),
        "marginal": marginal_baselines(cache, SC.split_rows()["fit8k"]),
        "probes": probes,
        "delta_ce": {
            "atom_zero_M": clean["atom"]["ce"] - probes["zero_M"]["atom"]["ce"],
            "atom_shuffled_partner":
                clean["atom"]["ce"] - probes["shuffled_partner"]["atom"]["ce"],
            "bond_zero_M": clean["bond"]["ce"] - probes["zero_M"]["bond"]["ce"],
            "bond_shuffled_partner":
                clean["bond"]["ce"] - probes["shuffled_partner"]["bond"]["ce"],
            "atom_vs_marginal": clean["atom"]["marginal_ce"] - clean["atom"]["ce"],
            "bond_vs_marginal": clean["bond"]["marginal_ce"] - clean["bond"]["ce"],
        },
    }


def pretrain(seed: int, device, *, mode: str, force: bool = False) -> dict:
    updates = SC.PRETRAIN_PILOT_UPDATES if mode == "pilot" else SC.PRETRAIN_UPDATES
    out = SC.RUNS / f"S_pre{int(seed)}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pretrain.json").exists() and not force:
        return json.loads((out / "pretrain.json").read_text())

    cache_full = SC.load_train_cache()
    cache = SC.structural_cache(cache_full)
    del cache_full  # the labels are unreachable from here on
    rel = SC.load_relation_cache()
    topo, topo_info = load_topology(cache)
    rows = SC.split_rows()
    fit = rows["fit8k"]
    norm = SC.fit_only_norm(cache)

    torch.manual_seed(int(seed) + HEAD_SEED_OFFSET)
    heads = SP.PretextHeads().to(device)
    model, model_info = SP.build_backbone(norm, seed, device)
    frozen = SP.freeze_encoders(model)
    params = SP.pretrainable_parameters(model) + list(heads.parameters())
    n_trainable = int(sum(p.numel() for p in params))

    optimizer = torch.optim.Adam(params, lr=SC.PRETRAIN_LR, weight_decay=SC.PRETRAIN_WD)
    gen = torch.Generator().manual_seed(int(seed) + SC.PRETRAIN_SHUFFLE_OFFSET)
    steps_per_epoch = int(np.ceil(len(fit) / SC.PRETRAIN_BATCH))
    n_epochs = int(np.ceil(updates / steps_per_epoch))

    initial_state_sha = C.state_hash(model)
    if initial_state_sha != SC.HISTORICAL_INIT_SHA256[int(seed)]:
        raise RuntimeError(f"seed {seed}: backbone init {initial_state_sha} is not the "
                           f"historical BOSI-B init {SC.HISTORICAL_INIT_SHA256[int(seed)]}")

    curve: list[dict] = []
    grad_log: list[dict] = []
    global_step = 0
    started = time.perf_counter()
    data_seconds = 0.0
    model.train()
    heads.train()
    for epoch in range(1, n_epochs + 1):
        order = torch.randperm(len(fit), generator=gen).numpy()
        sums = {"loss": 0.0, "ce_atom": 0.0, "ce_bond": 0.0,
                "n_atom": 0, "n_bond": 0, "correct_atom": 0, "correct_bond": 0}
        batches = 0
        t_epoch = time.perf_counter()
        t_data = 0.0
        for start in range(0, len(order), SC.PRETRAIN_BATCH):
            if global_step >= updates:
                break
            batch_rows = fit[order[start:start + SC.PRETRAIN_BATCH]]
            t0 = time.perf_counter()
            mb = SM.masked_batch(cache, rel, topo, batch_rows, seed=seed,
                                 visit_index=epoch)
            t_data += time.perf_counter() - t0
            batch = _to_device(mb["batch"], device)
            at = mb["atom_targets"].to(device)
            bt = mb["bond_targets"].to(device)
            res = SP.pretext_loss(model, heads, batch, at, bt)

            optimizer.zero_grad(set_to_none=True)
            # per-head gradient attribution must be read *before* the joint
            # backward frees the graph; autograd.grad does not accumulate
            # into ``.grad``, so the optimisation step is unaffected.
            entry = None
            if global_step in GRAD_STEPS or global_step == updates - 1:
                entry = {"step": int(global_step), "epoch": int(epoch),
                         "loss": float(res["loss"]),
                         "ce_atom": float(res["ce_atom"]) if res["ce_atom"] is not None else None,
                         "ce_bond": float(res["ce_bond"]) if res["ce_bond"] is not None else None}
                for head in ("atom", "bond"):
                    if res[f"ce_{head}"] is None:
                        continue
                    g = torch.autograd.grad(res[f"ce_{head}"],
                                            SP.pretrainable_parameters(model),
                                            retain_graph=True, allow_unused=True)
                    entry[f"grad_from_ce_{head}"] = _module_norms(model, g)
            res["loss"].backward()
            if entry is not None:
                entry["total"] = SP.module_grad_norms(model, heads)
                grad_log.append(entry)
            gn = torch.nn.utils.clip_grad_norm_(params, SC.PRETRAIN_CLIP)
            if not torch.isfinite(torch.as_tensor(float(gn))):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            optimizer.step()
            global_step += 1

            with torch.no_grad():
                if res["ce_atom"] is not None:
                    sums["ce_atom"] += float(res["ce_atom"]) * int(at.numel())
                    sums["n_atom"] += int(at.numel())
                    sums["correct_atom"] += int(res["atom_logits"].argmax(1).eq(at).sum())
                if res["ce_bond"] is not None:
                    sums["ce_bond"] += float(res["ce_bond"]) * int(bt.numel())
                    sums["n_bond"] += int(bt.numel())
                    sums["correct_bond"] += int(res["bond_logits"].argmax(1).eq(bt).sum())
                sums["loss"] += float(res["loss"])
                batches += 1
        epoch_s = time.perf_counter() - t_epoch
        data_seconds += t_data
        entry = {
            "epoch": int(epoch), "step": int(global_step), "batches": int(batches),
            "seconds": float(epoch_s), "data_seconds": float(t_data),
            "loss": sums["loss"] / max(batches, 1),
            "ce_atom": sums["ce_atom"] / max(sums["n_atom"], 1),
            "ce_bond": sums["ce_bond"] / max(sums["n_bond"], 1),
            "acc_atom": sums["correct_atom"] / max(sums["n_atom"], 1),
            "acc_bond": sums["correct_bond"] / max(sums["n_bond"], 1),
        }
        curve.append(entry)
        if epoch % 10 == 0 or global_step >= updates:
            SC.log(f"[{SC.ROUND_ID} pre s{seed}/{mode}] ep={epoch} step={global_step} "
                   f"ce_atom={entry['ce_atom']:.4f} ce_bond={entry['ce_bond']:.4f} "
                   f"acc_a={entry['acc_atom']:.4f} acc_b={entry['acc_bond']:.4f} "
                   f"{entry['seconds']:.2f}s (data {entry['data_seconds']:.2f}s)")
        if global_step >= updates:
            break

    wall = time.perf_counter() - started
    final_state_sha = C.state_hash(model)
    probe = evaluate(model, heads, cache, rel, topo, device, seed=seed)

    export = {
        "half_encoder": {k: v.detach().cpu().clone()
                         for k, v in model.half_encoder.state_dict().items()},
        "side": {k: v.detach().cpu().clone() for k, v in model.side.state_dict().items()},
        "compose": {k: v.detach().cpu().clone() for k, v in model.compose.state_dict().items()},
    }
    state_path = out / SC.pretrain_state_name(seed)
    C.atomic_torch_save({
        "export": export,
        "seed": int(seed),
        "mode": mode,
        "updates": int(updates),
        "backbone_final_state_sha256": final_state_sha,
        "backbone_initial_state_sha256": initial_state_sha,
        "head_state": {k: v.detach().cpu() for k, v in heads.state_dict().items()},
    }, state_path)
    export_sha = _hash_state(export)

    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": SC.ROUND_ID,
        "kind": "pretrain",
        "mode": mode,
        "seed": int(seed),
        "updates": int(updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_fit_rows": int(len(fit)),
        "fit_rows_sha256": rows["hashes"]["fit8k"],
        "recipe": {
            "optimizer": "adam", "lr": SC.PRETRAIN_LR, "weight_decay": SC.PRETRAIN_WD,
            "grad_clip": SC.PRETRAIN_CLIP, "batch_size": SC.PRETRAIN_BATCH,
            "shuffle_offset": SC.PRETRAIN_SHUFFLE_OFFSET,
            "atom_mask_ratio": SC.ATOM_MASK_RATIO, "bond_mask_ratio": SC.BOND_MASK_RATIO,
            "atom_mask_token": SC.ATOM_MASK_TOKEN, "bond_mask_token": SC.BOND_MASK_TOKEN,
            "loss": "CE_atom(28) + CE_bond(4), weights 1.0/1.0",
            "head_hidden": SC.HEAD_HIDDEN,
            "lr_schedule": "constant",
            "checkpoint_rule": "last-step state only",
            "arm": "B",
            "topology_masking": "types only (edges are never deleted)",
        },
        "backbone_final_state_sha256": final_state_sha,
        "model": {
            "parameters": model_info["parameters"],
            "parameters_breakdown": model_info["parameters_breakdown"],
            "backbone_initial_state_sha256": initial_state_sha,
            "histinit_matches_historical": True,
            "ssl_encoders": model_info["ssl"],
            "frozen": frozen,
            "trainable_parameters": n_trainable,
            "head_seed_offset": HEAD_SEED_OFFSET,
        },
        "topology": topo_info,
        "probe": probe,
        "grad_log": grad_log,
        "curve": curve,
        "tail10_median_ce_atom": float(np.median([e["ce_atom"] for e in tail])),
        "tail10_median_ce_bond": float(np.median([e["ce_bond"] for e in tail])),
        "state_file": SC.rel_or_abs(state_path),
        "state_sha256": SC.sha256_file(state_path),
        "export_sha256": export_sha,
        "wall_seconds": float(wall),
        "data_seconds": float(data_seconds),
        "gpu_hours": float(wall / 3600.0),
        "device": str(device),
        "labels_read": False,
        "monitor_loaded": False,
        "locked_loaded": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    SC.write_json(out / "pretrain.json", info)
    SC.log(f"[{SC.ROUND_ID} pre s{seed}/{mode}] DONE ce_atom={curve[-1]['ce_atom']:.4f} "
           f"ce_bond={curve[-1]['ce_bond']:.4f} "
           f"probe_atom={probe['probes']['clean']['atom']['ce']:.4f} "
           f"gpu_h={info['gpu_hours']:.3f}")
    return info


def _module_norms(model, grads) -> dict:
    out: dict[str, float] = {}
    i = 0
    for name in ("half_encoder", "side", "compose"):
        acc = 0.0
        for p in getattr(model, name).parameters():
            g = grads[i]
            i += 1
            if g is not None:
                acc += float(g.detach().pow(2).sum())
        out[name] = float(np.sqrt(acc))
    return out


def _hash_state(export: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in ("half_encoder", "side", "compose"):
        for k, v in sorted(export[name].items()):
            h.update(f"{name}.{k}".encode())
            h.update(np.ascontiguousarray(v.numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=("pilot", "full"), default="full")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    info = pretrain(args.seed, torch.device(args.device), mode=args.mode, force=args.force)
    print(json.dumps({k: info[k] for k in ("mode", "seed", "updates", "epochs_run",
                                           "export_sha256", "gpu_hours",
                                           "probe")}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
