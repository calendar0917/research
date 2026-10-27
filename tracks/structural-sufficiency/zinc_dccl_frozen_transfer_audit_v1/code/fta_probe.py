"""Phase B: train fresh fixed-shape readers on frozen representations.

24 primary cells = {fit4k,fit8k} x {seed 0,1} x {atom_global, local_global, full}
                   x reader-init {0,1}
5 diagnostic cells = 4 readout-recovery + 1 label-permutation null
1 reproducibility cell (repeat of fit8k_s0 / full / init 0)

Everything is CPU, single-threaded, FP32, and fully deterministic.  No trunk
weight is optimised.  Labels are read for probe_fit (training) and probe_sel
(selection) only; Monitor labels are untouched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import fta_common as K

DIM = 513
HIDDEN = 64


def build_head(seed: int) -> torch.nn.Sequential:
    torch.manual_seed(int(seed))
    return torch.nn.Sequential(
        torch.nn.Linear(DIM, HIDDEN),
        torch.nn.SiLU(),
        torch.nn.Linear(HIDDEN, 1),
    )


def tensor_state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(
            v.detach().numpy() if hasattr(v, "detach") else v, dtype=np.float32).tobytes())
    return h.hexdigest()


def head_state_hash(head: torch.nn.Module) -> str:
    return tensor_state_hash(head.state_dict())


@torch.no_grad()
def _l1(head, z: torch.Tensor, y: torch.Tensor, idx: np.ndarray, batch: int = 256) -> float:
    total, n = 0.0, 0
    for s in range(0, len(idx), batch):
        j = torch.from_numpy(np.asarray(idx[s:s + batch], dtype=np.int64))
        total += float((head(z[j]).view(-1) - y[j]).abs().sum())
        n += len(idx[s:s + batch])
    return total / max(n, 1)


def cell_id(condition: str, block: str, head_init: int, kind: str = "main") -> str:
    base = f"{condition}__{block}__i{head_init}"
    return base if kind == "main" else f"{base}__{kind}"


def cells(cfg: dict) -> list[dict]:
    out = []
    inits = [int(i) for i in cfg["reader"]["head_inits"]]
    for fit, seed in K.CONDITIONS:
        cid = f"{fit}_s{seed}"
        for block in ("atom_global", "local_global", "full"):
            for init in inits:
                out.append({"id": cell_id(cid, block, init), "condition": cid,
                            "block": block, "head_init": init, "kind": "main",
                            "target": "y"})
    for cid in cfg["controls"]["readout_recovery"]["conditions"]:
        out.append({"id": cell_id(cid, "full", 0, "recovery"), "condition": cid,
                    "block": "full", "head_init": 0, "kind": "recovery",
                    "target": "trunk_pred"})
    null = cfg["controls"]["label_permutation_null"]
    out.append({"id": cell_id(null["condition"], null["block"], int(null["head_init"]), "null"),
                "condition": null["condition"], "block": null["block"],
                "head_init": int(null["head_init"]), "kind": "null",
                "target": "y_permuted", "perm_seed": int(null["perm_seed"])})
    out.append({"id": cell_id("fit8k_s0", "full", 0, "repro"), "condition": "fit8k_s0",
                "block": "full", "head_init": 0, "kind": "repro", "target": "y"})
    return out


def train_cell(spec: dict, data: dict, block_mask: np.ndarray, pos: dict,
               targets: dict, cfg: dict) -> dict:
    r = cfg["reader"]
    idx_fit = pos["fit"]
    idx_sel = pos["sel"]
    z_std = K.apply_standardizer(data["z"], data["mu"], data["sd"])
    z_masked = np.ascontiguousarray(z_std * block_mask.astype(np.float32))

    target = targets[spec["target"]]
    zt = torch.from_numpy(z_masked)
    yt = torch.from_numpy(np.ascontiguousarray(target))
    head = build_head(spec["head_init"])
    init_hash = head_state_hash(head)
    opt = torch.optim.AdamW(head.parameters(), lr=float(r["lr"]),
                            weight_decay=float(r["weight_decay"]))
    gen = torch.Generator().manual_seed(int(spec["head_init"]) + int(r["batch_order_seed_offset"]))
    bs = int(r["batch_size"])
    updates = int(r["updates"])
    eval_every = int(r["eval_every"])
    spe = math.ceil(len(idx_fit) / bs)

    curve, best = [], {"probe_sel_l1": float("inf"), "step": -1}
    best_state = None
    block_grad_check = None
    order = None
    t0 = time.perf_counter()
    for step in range(1, updates + 1):
        k = (step - 1) % spe
        if k == 0:
            order = idx_fit[torch.randperm(len(idx_fit), generator=gen).numpy()]
        batch = torch.from_numpy(np.asarray(order[k * bs:(k + 1) * bs], dtype=np.int64))
        pred = head(zt[batch]).view(-1)
        loss = F.l1_loss(pred, yt[batch])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and not block_mask.all():
            block_grad_check = float(head[0].weight.grad[:, ~block_mask].abs().max())
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(r["grad_clip"]))
        opt.step()
        if step % eval_every == 0 or step == updates:
            sel = _l1(head, zt, yt, idx_sel)
            curve.append({"step": int(step), "probe_sel_l1": float(sel),
                          "online_l1": float(loss)})
            if sel < best["probe_sel_l1"]:
                best = {"probe_sel_l1": float(sel), "step": int(step)}
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    wall = time.perf_counter() - t0

    final_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    train_best = _l1(head, zt, yt, idx_fit)
    head.load_state_dict(final_state)
    train_final = _l1(head, zt, yt, idx_fit)
    final_sel = _l1(head, zt, yt, idx_sel)

    best_path = K.HEAD_DIR / f"{spec['id']}.best.pt"
    final_path = K.HEAD_DIR / f"{spec['id']}.final.pt"
    K.atomic_torch_save({"state_dict": best_state, "spec": spec,
                         "best": best, "reader_cfg": r}, best_path)
    K.atomic_torch_save({"state_dict": final_state, "spec": spec,
                         "reader_cfg": r}, final_path)

    return {
        "id": spec["id"], "kind": spec["kind"], "condition": spec["condition"],
        "block": spec["block"], "head_init": int(spec["head_init"]),
        "target": spec["target"], "n_features_used": int(block_mask.sum()),
        "n_probe_fit": int(len(idx_fit)), "n_probe_sel": int(len(idx_sel)),
        "reader_cfg": dict(r),
        "init_state_sha256": init_hash,
        "best": best, "final": {"probe_sel_l1": float(final_sel), "step": int(updates)},
        "train_l1_at_best": float(train_best), "train_l1_at_final": float(train_final),
        "curve": curve,
        "block_grad_leak_max_abs": block_grad_check,
        "best_state_sha256": K.sha256_file(best_path),
        "final_state_sha256": K.sha256_file(final_path),
        "best_state_hash": tensor_state_hash(best_state),
        "final_state_hash": tensor_state_hash(final_state),
        "wall_seconds": float(wall),
        "torch_threads": int(torch.get_num_threads()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None, help="comma separated cell ids")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    torch.set_num_threads(1)

    cfg = K.load_config()
    cache = K.load_train_cache()
    rows = K.split_rows()
    pf, ps, split_info = K.probe_split(cache, rows["fit4k"])
    freeze = json.loads((K.ANALYSIS / "representation_freeze.json").read_text())
    if split_info["probe_fit_sha256"] != freeze["split"]["probe_fit_sha256"]:
        raise RuntimeError("probe split differs from the exported one")
    if freeze["split"]["n_probe_fit"] != len(pf):
        raise RuntimeError("probe_fit n drift")

    export_rows = np.concatenate([pf, ps, rows["monitor"]])
    if freeze["export_rows"]["sha256"] != K.row_hash(export_rows):
        raise RuntimeError("export row hash drift")
    y_export = np.asarray(cache["labels"], dtype=np.float64)[export_rows]
    pos = {"fit": np.arange(0, len(pf)),
           "sel": np.arange(len(pf), len(pf) + len(ps)),
           "mon": np.arange(len(pf) + len(ps), len(export_rows))}

    rng = np.random.default_rng(int(cfg["controls"]["label_permutation_null"]["perm_seed"]))
    perm = rng.permutation(len(pf) + len(ps))
    y_perm = y_export[:len(pf) + len(ps)][perm]
    targets = {"y": y_export, "y_permuted": y_perm}

    specs = cells(cfg)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        specs = [s for s in specs if s["id"] in keep]

    data_cache: dict[str, dict] = {}
    out_dir = K.ANALYSIS / "probe_cells"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in specs:
        path = out_dir / f"{spec['id']}.json"
        if path.exists() and not args.force:
            summary.append(json.loads(path.read_text()))
            K.log(f"{spec['id']} cached")
            continue
        cid = spec["condition"]
        if cid not in data_cache:
            d = np.load(K.Z_DIR / f"{cid}.npz")
            if not np.array_equal(d["rows"], export_rows):
                raise RuntimeError(f"{cid}: exported row order mismatch")
            mu, sd = K.standardizer(d["z"][pos["fit"]])
            data_cache[cid] = {"z": d["z"], "pred": d["pred"], "mu": mu, "sd": sd}
        t = dict(targets)
        t["trunk_pred"] = data_cache[cid]["pred"]
        info = train_cell(spec, data_cache[cid], K.block_mask(spec["block"]),
                          pos, t, cfg)
        K.write_json(path, info)
        summary.append(info)
        K.ledger_append({"stage": "probe", "cell": spec["id"], "kind": spec["kind"],
                         "wall_seconds": info["wall_seconds"], "device": "cpu"})
        K.log(f"{spec['id']} best={info['best']['probe_sel_l1']:.6f}"
              f"@{info['best']['step']} final={info['final']['probe_sel_l1']:.6f}"
              f" train@best={info['train_l1_at_best']:.5f} {info['wall_seconds']:.1f}s")

    init_map: dict[int, set[str]] = {}
    for c in summary:
        init_map.setdefault(c["head_init"], set()).add(c["init_state_sha256"])
    init_identity = {str(k): {"n_distinct": len(v), "hashes": sorted(v)}
                     for k, v in sorted(init_map.items())}
    if any(v["n_distinct"] != 1 for v in init_identity.values()):
        raise RuntimeError(f"head init differs across cells: {init_identity}")

    by_id = {c["id"]: c for c in summary}
    base = by_id.get("fit8k_s0__full__i0")
    repro = by_id.get("fit8k_s0__full__i0__repro")
    repro_check = None
    if base and repro:
        repro_check = {
            "base_probe_sel_l1": base["best"]["probe_sel_l1"],
            "repro_probe_sel_l1": repro["best"]["probe_sel_l1"],
            "abs_diff": abs(base["best"]["probe_sel_l1"] - repro["best"]["probe_sel_l1"]),
            "best_state_identical": repro["best_state_hash"] == base["best_state_hash"],
            "base_state_hash": base["best_state_hash"],
            "repro_state_hash": repro["best_state_hash"],
        }
        if repro_check["abs_diff"] > 1e-6:
            raise RuntimeError(f"reproducibility failed: {repro_check}")

    payload = {
        "round": K.ROUND_ID,
        "n_cells": len(summary),
        "reader_cfg": cfg["reader"],
        "head_init_identity": init_identity,
        "reproducibility": repro_check,
        "label_permutation": {
            "seed": int(cfg["controls"]["label_permutation_null"]["perm_seed"]),
            "applied_to": "probe_fit ++ probe_sel labels, single global permutation",
            "probe_fit_target_mean_abs": float(np.abs(y_perm[:len(pf)]).mean()),
            "probe_sel_target_mean_abs": float(np.abs(y_perm[len(pf):]).mean()),
        },
        "cells": summary,
        "wall_seconds_total": float(sum(c["wall_seconds"] for c in summary)),
        "monitor_labels_read": False,
    }
    K.write_json(K.ANALYSIS / "probe_freeze.json", payload)
    K.log(f"froze {len(summary)} cells -> analysis/probe_freeze.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
