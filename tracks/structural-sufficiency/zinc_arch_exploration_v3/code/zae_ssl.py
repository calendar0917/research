"""Native masked structural pretraining for arm A (A-SSL).

The *new* pair-composition trunk is pretrained on the train split only, from a
fresh mask on every visit:

1. draw ``max(1, round(0.15 n))`` atom rows and physical bonds per molecule with
   a deterministic SeedSequence over ``(seed, molecule, epoch)``;
2. replace the raw atom/bond category codes with the dedicated mask tokens and
   **rebuild every derived feature from the masked raw graph** (the frozen
   ``structural_relation_objects_v1`` builder): neighbour-type histograms,
   radius-2 histograms, junction bond-type descriptors, typed pair features and
   ``global33`` therefore see the mask, not the answer;
3. predict the hidden raw atom class (28-way) from the per-atom pooled pair
   state and the hidden raw bond class (4-way) from the two directed pair states
   ``h_ij, h_ji`` — both read the *new* contraction output, so the pretext task
   trains the new composition rather than only the local encoders;
4. topology, distances, common-neighbour counts, ring/block flags and the
   spectral operators are pure functions of the *untyped* graph and are kept.

No ``y``, no target component, no valid/test molecule is used.  Reconstruction
heads are temporary: they are dropped before the supervised phase, and their
parameters are reported separately from the final predictor.
"""

from __future__ import annotations

import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SUF = _EXP.parent
_V1 = _SUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_SUF.parents[1] / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import bjoc_masked_pretraining_v1 as MP  # noqa: E402
import zae_data as ZD  # noqa: E402
import zae_models as ZM  # noqa: E402
import zae_train as ZT  # noqa: E402

SSL_SEED = 0
ATOM_CLASSES = 28   # P3 atom one-hot width, 27 = mask token
BOND_CLASSES = 4    # raw bond one-hot width, 0 = mask token
HOLDOUT_ROWS = 500  # last 500 train molecules: reconstruction curve only

_G: dict[str, Any] = {}


def _init_worker(objs, raws, pair_cache):
    _G["objs"] = objs
    _G["raws"] = raws
    _G["pair"] = pair_cache


def _worker_chunk(args):
    epoch, start, stop = args
    objs = _G["objs"]
    raws = _G["raws"]
    out = []
    for i in range(start, stop):
        base = objs[i]
        x, ei, ea = raws[i]
        atom_ids, bond_ids = MP.sample_mask(SSL_SEED, i, epoch, base.n_atoms,
                                           int(base.bond_index.shape[0]))
        x_m, ea_m = MP.mask_raw_arrays(x, ei, ea, np.asarray(base.bond_index, dtype=np.int64),
                                       atom_ids, bond_ids)
        import structural_relation_objects_v1 as sro
        masked = sro.build_mol_objects(i, x_m, ei, ea_m, smiles="")
        if not np.array_equal(np.asarray(masked.bond_index), np.asarray(base.bond_index)):
            raise RuntimeError("masked rebuild changed the physical bond table")
        atom_truth = np.asarray([int(np.asarray(x).reshape(-1)[k]) for k in atom_ids], dtype=np.int64)
        bond_truth = np.asarray([int(base.bond_type[k]) for k in bond_ids], dtype=np.int64)
        out.append((masked, np.asarray(atom_ids, dtype=np.int64), atom_truth,
                    np.asarray(bond_ids, dtype=np.int64), bond_truth))
    return out


def _raw_graphs(split: str):
    ds = D.pyg_zinc(split)
    raws = []
    for i in range(len(ds)):
        d = ds[i]
        raws.append((d.x.reshape(-1).long().numpy(),
                     d.edge_index.numpy(),
                     d.edge_attr.reshape(-1).long().numpy()))
    return raws


def masked_epoch(objs, raws, epoch: int, *, workers: int = 8) -> dict:
    """Build one fresh-mask epoch table (plain-process parallel, fork)."""
    n = len(objs)
    bounds = [(epoch, s, min(s + 250, n)) for s in range(0, n, 250)]
    if workers <= 1:
        _init_worker(objs, raws, None)
        parts = [_worker_chunk(b) for b in bounds]
    else:
        ctx = get_context("fork")
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(objs, raws, None)) as pool:
            parts = pool.map(_worker_chunk, bounds)
    samples = [s for part in parts for s in part]
    masked_objs = [s[0] for s in samples]
    atom_type = np.concatenate([np.asarray(s[0].atom_feat)[:, :ATOM_CLASSES].argmax(1) for s in samples])
    bond_type = np.concatenate([np.asarray(s[0].bond_type) for s in samples])
    packed = ZD.pack_objects(masked_objs, atom_type, bond_type)
    n_atoms = np.asarray(packed["n_atoms_graph"], dtype=np.int64)
    n_bonds = np.asarray(packed["n_bonds_graph"], dtype=np.int64)
    atom_ptr = np.asarray(packed["atom_ptr"])
    bond_ptr = np.asarray(packed["bond_ptr"])
    atom_flag = np.zeros(int(atom_ptr[-1]), dtype=np.float32)
    atom_target = np.zeros(int(atom_ptr[-1]), dtype=np.int64)
    bond_flag = np.zeros(int(bond_ptr[-1]), dtype=np.float32)
    bond_target = np.zeros(int(bond_ptr[-1]), dtype=np.int64)
    for i, (_m, aids, atruth, bids, btruth) in enumerate(samples):
        a0 = int(atom_ptr[i])
        b0 = int(bond_ptr[i])
        atom_flag[a0 + aids] = 1.0
        atom_target[a0 + aids] = atruth
        bond_flag[b0 + bids] = 1.0
        bond_target[b0 + bids] = btruth
    packed["atom_mask_flag"] = torch.from_numpy(atom_flag)
    packed["atom_mask_target"] = torch.from_numpy(atom_target)
    packed["bond_mask_flag"] = torch.from_numpy(bond_flag)
    packed["bond_mask_target"] = torch.from_numpy(bond_target)
    return packed


class SSLHeads(nn.Module):
    """Temporary reconstruction heads on top of the new pair composition."""

    def __init__(self, width: int = ZM.WIDTH):
        super().__init__()
        self.atom_head = nn.Linear(2 * width, ATOM_CLASSES)
        self.bond_head = nn.Linear(2 * width, BOND_CLASSES)


def _ssl_logits(model: ZM.PairComposition, heads: SSLHeads, batch: Mapping[str, torch.Tensor]):
    """Reconstruct atom/bond classes from the pair-composition state."""
    enc = model.encode_pairs(batch)
    h = enc["h"]
    B, D = int(batch["n_graphs"]), int(batch["Nmax"])
    dense = B * D * D
    h_dense = h.new_zeros((dense, h.shape[1])).index_copy(0, batch["pair_dense"], h)
    # atom state: diagonal pair + mean over the atom's off-diagonal row
    diag = h_dense.view(B, D, D, -1).diagonal(dim1=1, dim2=2)          # [B,D,W]
    diag = diag.reshape(dense // D, -1).index_select(0, batch["atom_slot"])  # [A,W]
    off = h * (1.0 - batch["pair_self"]).unsqueeze(1)
    row_sum = h.new_zeros((int(batch["atom_feat"].shape[0]), h.shape[1]))
    row_sum.index_add_(0, batch["pair_i"], off)
    denom = (batch["atom_n"] - 1.0).clamp(min=1.0).unsqueeze(1)
    atom_state = torch.cat([diag, row_sum / denom], dim=1)
    atom_logits = heads.atom_head(atom_state)
    bond_state = torch.cat(
        [h_dense.index_select(0, batch["bond_dense_ij"]),
         h_dense.index_select(0, batch["bond_dense_ji"])], dim=1
    )
    bond_logits = heads.bond_head(bond_state)
    return atom_logits, bond_logits


def ssl_loss(model, heads, batch):
    atom_logits, bond_logits = _ssl_logits(model, heads, batch)
    af = batch["atom_mask_flag"]
    bf = batch["bond_mask_flag"]
    ce_a = F.cross_entropy(atom_logits, batch["atom_mask_target"], reduction="none")
    ce_b = F.cross_entropy(bond_logits, batch["bond_mask_target"], reduction="none")
    a = (ce_a * af).sum() / af.sum().clamp(min=1.0)
    b = (ce_b * bf).sum() / bf.sum().clamp(min=1.0)
    return ZT.SSL_RECIPE["atom_loss_weight"] * a + ZT.SSL_RECIPE["bond_loss_weight"] * b, a, b


@torch.no_grad()
def ssl_eval(model, heads, batch) -> dict:
    model.eval()
    _loss, a, b = ssl_loss(model, heads, batch)
    atom_logits, bond_logits = _ssl_logits(model, heads, batch)
    af = batch["atom_mask_flag"]
    bf = batch["bond_mask_flag"]
    acc_a = ((atom_logits.argmax(-1) == batch["atom_mask_target"]).float() * af).sum() / af.sum().clamp(min=1.0)
    acc_b = ((bond_logits.argmax(-1) == batch["bond_mask_target"]).float() * bf).sum() / bf.sum().clamp(min=1.0)
    model.train()
    return {"atom_ce": float(a), "bond_ce": float(b), "atom_acc": float(acc_a), "bond_acc": float(acc_b)}


def pretrain(seed: int, device, *, epochs: int | None = None, out_dir: Path | None = None,
             force: bool = False, workers: int = 8, tag: str | None = None) -> dict:
    recipe = dict(ZT.SSL_RECIPE)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    prefix = f"{tag}_" if tag else ""
    out_dir = Path(out_dir if out_dir is not None else ZT.RUNS / f"{prefix}ASSL_pretrain_s{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "ssl.json").exists() and not force:
        C.log(f"[A-SSL s{seed}] pretraining already complete")
        return C.read_json(out_dir / "ssl.json")

    norm = D.load_norm()
    objs, obj_source = D.load_p3_objects("train")
    pair_cache = ZD.load_pair_split("train")
    raws = _raw_graphs("train")
    model, info = ZM.build_model("A", norm, seed, device)
    heads = SSLHeads().to(device)
    params_pred = C.parameter_count(model)
    params_heads = C.parameter_count(heads)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()),
                                  lr=float(recipe["lr"]),
                                  weight_decay=float(recipe["weight_decay"]))
    audit = ZT._param_audit(model, optimizer)  # optimizer covers model + heads
    rows = np.arange(len(objs), dtype=np.int64)
    hold = rows[-HOLDOUT_ROWS:]
    fit = rows[:-HOLDOUT_ROWS]
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(fit) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    gen = torch.Generator().manual_seed(int(seed) + 303)
    curve: list[dict] = []
    started = time.perf_counter()
    global_step = 0
    last_lr = float(recipe["lr"])
    best_hold = float("inf")
    best_epoch = -1
    best_state = None
    for epoch in range(1, n_epochs + 1):
        t_ep = time.perf_counter()
        table = masked_epoch(objs, raws, epoch, workers=workers)
        model.train()
        heads.train()
        order = torch.randperm(len(fit), generator=gen).numpy()
        sums = {"loss": 0.0, "atom_ce": 0.0, "bond_ce": 0.0}
        nb = 0
        for start in range(0, len(order), batch_size):
            idx = fit[order[start:start + batch_size]]
            batch = ZT._to_device(ZD.collate_pair(table, pair_cache, idx), device)
            loss, a, b = ssl_loss(model, heads, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()),
                                                float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite SSL gradient at epoch {epoch}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, total_updates, warmup,
                                              float(recipe["lr"]), float(recipe["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            sums["loss"] += float(loss.detach())
            sums["atom_ce"] += float(a.detach())
            sums["bond_ce"] += float(b.detach())
            nb += 1
        # fixed-row holdout reconstruction curve (train molecules only)
        hold_batch = ZT._to_device(ZD.collate_pair(table, pair_cache, hold), device)
        hold_metrics = ssl_eval(model, heads, hold_batch)
        entry = {
            "epoch": int(epoch),
            "fit_loss": sums["loss"] / max(nb, 1),
            "fit_atom_ce": sums["atom_ce"] / max(nb, 1),
            "fit_bond_ce": sums["bond_ce"] / max(nb, 1),
            "hold_atom_ce": hold_metrics["atom_ce"],
            "hold_bond_ce": hold_metrics["bond_ce"],
            "hold_atom_acc": hold_metrics["atom_acc"],
            "hold_bond_acc": hold_metrics["bond_acc"],
            "lr": float(last_lr),
            "seconds": float(time.perf_counter() - t_ep),
        }
        curve.append(entry)
        objective = entry["hold_atom_ce"] + entry["hold_bond_ce"]
        if objective < best_hold:
            best_hold = objective
            best_epoch = epoch
            best_state = C.cpu_state(model)
            C.atomic_torch_save(
                {"state_dict": best_state, "epoch": int(epoch), "hold_objective": float(best_hold),
                 "candidate": "A-SSL", "seed": int(seed)},
                out_dir / "pretrain_best.pt",
            )
        C.log(f"[A-SSL s{seed}] ep={epoch:03d} fit={entry['fit_loss']:.4f} "
              f"hold_atom={entry['hold_atom_ce']:.4f} hold_bond={entry['hold_bond_ce']:.4f} "
              f"acc={entry['hold_atom_acc']:.3f}/{entry['hold_bond_acc']:.3f} {entry['seconds']:.1f}s")
    wall = time.perf_counter() - started
    final_state = C.cpu_state(model)
    C.atomic_torch_save(
        {"state_dict": best_state, "epoch": int(best_epoch), "hold_objective": float(best_hold),
         "candidate": "A-SSL", "seed": int(seed), "final_state_dict": final_state,
         "final_epoch": int(n_epochs)},
        out_dir / "pretrain.pt",
    )
    payload = {
        "round": ZT.ROUND_ID,
        "protocol_id": ZT.PROTOCOL_ID,
        "candidate": "A-SSL-pretrain",
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train only",
            "n_fit": int(len(fit)),
            "n_holdout": int(len(hold)),
            "official_test_loaded": False,
            "official_valid_loaded": False,
            "labels_read": False,
            "object_source": obj_source,
            "pair_cache_meta": pair_cache["meta"],
        },
        "recipe": {**recipe, "epochs": int(n_epochs), "steps_per_epoch": int(steps_per_epoch),
                   "updates": int(total_updates), "warmup_steps_effective": int(warmup),
                   "actual_global_steps": int(global_step), "fresh_mask_every_epoch": True},
        "parameters": {"predictor_trunk": int(params_pred), "temporary_ssl_heads": int(params_heads),
                       "optimizer_parameters": int(audit["optimizer_parameters"])},
        "masking": {
            "source": "bjoc_masked_pretraining_v1.sample_mask + mask_raw_arrays (frozen)",
            "atom_ratio": MP.ATOM_MASK_RATIO, "bond_ratio": MP.BOND_MASK_RATIO,
            "atom_mask_token": MP.ATOM_MASK_TOKEN, "bond_mask_token": MP.BOND_MASK_TOKEN,
            "full_rebuild": True,
            "seed_sequence": "SeedSequence([ssl_seed, mol_id, epoch])",
        },
        "best_hold": {"epoch": int(best_epoch), "objective": float(best_hold)},
        "holdout_rule": ("best holdout reconstruction objective (atom CE + bond CE over the fixed "
                         "last 500 train molecules, masked with the same epoch mask); ties keep the "
                         "earlier epoch"),
        "curve": curve,
        "wall_seconds": float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "device": str(device),
    }
    C.write_json(out_dir / "ssl.json", payload)
    C.write_curve_csv(out_dir / "ssl_curve.csv", curve)
    C.write_json(out_dir / "artifact_hashes.json", {
        "pretrain.pt": C.sha256_file(out_dir / "pretrain.pt"),
        "pretrain_best.pt": C.sha256_file(out_dir / "pretrain_best.pt"),
        "ssl.json": C.sha256_file(out_dir / "ssl.json"),
        "ssl_curve.csv": C.sha256_file(out_dir / "ssl_curve.csv"),
    })
    C.log(f"[A-SSL s{seed}] pretrain done best_hold={best_hold:.4f}@{best_epoch} wall={wall:.1f}s")
    return payload


def load_pretrained(path: Path, device="cpu") -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.to(device) for k, v in blob["state_dict"].items()}
