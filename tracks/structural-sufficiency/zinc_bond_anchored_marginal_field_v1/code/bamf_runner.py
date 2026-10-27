"""Runner for BAMF-v1 (G = exact global marginals, I = independent marginals).

The supervised loop follows the frozen DCC-L bottleneck recipe exactly:

* Fit8k rows only, no additional labels;
* fit-only standardisation (identical to the historical DCC-L runs);
* AdamW, lr 1e-3 -> 1e-5 per-step cosine, wd 1e-5, warmup 500, clip 5;
* batch 128, raw-y L1, 47,400 updates;
* checkpoint selection on the frozen Monitor1k split only.

Official valid and official test are never loaded.  The Locked1k split is not
read here at all; it is read once by the separate freeze/score step after both
seeds' predictions have been hashed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
_DCB = _SUFF / "zinc_dccl_bottleneck_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC), str(_DCB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bamf_inference as B  # noqa: E402
import bamf_models as BM  # noqa: E402
import dcb_train as DT  # noqa: E402

ROUND_ID = "zinc-bond-anchored-marginal-field-v1"
RUNS = _ROUND / "runs"
ANALYSIS = _ROUND / "analysis"
TOTAL_UPDATES = 47_400
RECIPE = dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
              warmup_steps=500, dropout=0.0, grad_clip=5.0, batch_size=128,
              valid_batch_size=256, loss="l1")


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{arm}_s{int(seed)}"


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def graphs_from_cache(cache: Mapping[str, np.ndarray]):
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"])
    out = []
    for r in range(len(ap) - 1):
        s, e = int(ap[r]), int(ap[r + 1])
        b0, b1 = int(bp[r]), int(bp[r + 1])
        edges = [(int(bi[k, 0]), int(bi[k, 1])) for k in range(b0, b1)]
        out.append((e - s, edges))
    return out


def plan_hash(plan: B.GlobalPlan) -> str:
    h = hashlib.sha256()
    for name in (
        "clique_scope", "clique_scope_size", "clique_parent", "clique_level",
        "node_clique", "node_pos", "edge_clique", "edge_pos_u", "edge_pos_v",
        "up_sep_child", "up_sep_parent", "sep_size", "graph_clique_ptr",
        "graph_node_ptr", "graph_edge_ptr", "widths",
    ):
        a = getattr(plan, name).detach().cpu().numpy()
        h.update(name.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build_plan(cache: Mapping[str, np.ndarray], *, verbose: bool = False) -> B.GlobalPlan:
    graphs = graphs_from_cache(cache)
    plan = B.build_global_plan(graphs, smax=4, verbose=verbose)
    if plan.max_width > 3:
        raise RuntimeError(f"unexpected induced width {plan.max_width} > 3")
    return plan


def make_builder(arm: str, norm: Mapping[str, object], plan_cpu: B.GlobalPlan | None,
                 state: dict):
    def builder(candidate, norm_, seed, device):
        plan_gpu = None
        if plan_cpu is not None:
            plan_gpu = copy.deepcopy(plan_cpu).to(device)
        model, info = BM.build_model(candidate, norm_, seed, device, arm=arm, plan=plan_gpu)
        state["model"] = model
        state["plan_gpu"] = plan_gpu
        return model, info

    return builder


def make_collate(arm: str, plan_cpu: B.GlobalPlan | None, device, state: dict):
    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        b = T1.collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)
        if arm == "G":
            rows_np = np.asarray(rows, dtype=np.int64)
            ap = np.asarray(cache["atom_ptr"])
            bp = np.asarray(cache["bond_ptr"])
            batch_plan = plan_cpu.select(
                rows_np,
                (ap[rows_np + 1] - ap[rows_np]).astype(np.int64),
                (bp[rows_np + 1] - bp[rows_np]).astype(np.int64),
            ).to(device)
            b["bamf_plan"] = batch_plan
        return b

    return collate


def _parameter_audit(model: torch.nn.Module) -> dict:
    named = {n: int(p.numel()) for n, p in model.named_parameters()}
    total = int(sum(named.values()))
    opt_like = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "parameters": total,
        "parameters_named_sum": int(sum(named.values())),
        "requires_grad_parameters": opt_like,
        "parameter_tensors": sum(1 for _ in model.parameters()),
        "requires_grad_tensors": sum(1 for p in model.parameters() if p.requires_grad),
        "named": named,
    }


def _batch_plan_stats(collate, cache, rows, device) -> dict:
    b = collate(cache, rows, "P3N", valid=True)
    p = b.get("bamf_plan")
    if p is None:
        return {}
    return {
        "batch_n_cliques": int(p.n_cliques),
        "batch_n_atoms": int(p.n_atoms),
        "batch_n_bonds": int(p.n_bonds),
        "batch_max_scope": int(p.clique_scope_size.max().item()) if p.n_cliques else 0,
    }


def train(arm: str, seed: int, device, *, total_updates: int = TOTAL_UPDATES,
          tag: str | None = None, force: bool = False, verbose: bool = False,
          plan: B.GlobalPlan | None = None, rr_job_id: str | None = None) -> dict:
    if arm not in ("G", "I"):
        raise ValueError(arm)
    split = DT.split_rows()
    train_rows = split["fit8k"]
    monitor_rows = split["monitor"]
    out = run_dir(arm, seed, tag)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        return json.loads((out / "train.json").read_text())

    train_cache = D.load_cache("train")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    norm = DT.fit_only_norm(train_cache, train_rows)
    if plan is None and arm == "G":
        t_plan = time.perf_counter()
        plan = build_plan(train_cache, verbose=verbose)
        plan_seconds = time.perf_counter() - t_plan
    else:
        plan_seconds = 0.0
    if arm == "G" and plan is None:
        raise RuntimeError("G arm needs a global plan")
    if plan is not None and plan.max_width > 3:
        raise RuntimeError("induced width above the frozen bound")

    state: dict = {}
    builder = make_builder(arm, norm, plan, state)
    collate = make_collate(arm, plan, device, state)
    model, model_info = builder("BAMF", norm, seed, device)
    params = C.parameter_count(model)
    if params > 500_000:
        raise RuntimeError(f"{params} > 500k")
    audit = _parameter_audit(model)
    if audit["parameters"] != 497_093:
        raise RuntimeError(f"unexpected parameter count {audit['parameters']} != 497093")

    batch_size = int(RECIPE["batch_size"])
    steps_per_epoch = math.ceil(len(train_rows) / batch_size)
    n_epochs = math.ceil(int(total_updates) / steps_per_epoch)
    warmup = min(int(RECIPE["warmup_steps"]), max(1, int(total_updates) // 20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                                  weight_decay=RECIPE["weight_decay"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"monitor_mae": float("inf"), "epoch": -1, "step": -1}
    best_state = None
    global_step = 0
    started = time.perf_counter()
    last_lr = float(RECIPE["lr"])
    first_grad_audit: dict = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            if global_step >= int(total_updates):
                break
            batch_rows = train_rows[order[start:start + batch_size]]
            batch = T1._to_device(collate(train_cache, batch_rows, "BAMF", valid=False), device)
            target = torch.as_tensor(y_train[batch_rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if global_step < 3:
                ga = model.atom_potential.weight.grad
                gb = model.bond_potential.weight.grad
                first_grad_audit[f"step_{global_step}"] = {
                    "atom_potential_grad_max": None if ga is None else float(ga.abs().max()),
                    "bond_potential_grad_max": None if gb is None else float(gb.abs().max()),
                    "new_columns_grad_max": float(
                        model.local_composer.lins[0].weight.grad[:, 296:].abs().max()
                    ),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["grad_clip"])
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at step {global_step}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(global_step, int(total_updates), warmup,
                                              float(RECIPE["lr"]), float(RECIPE["min_lr"]))
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(batch_rows)
        online = total_err / max(n_mol, 1)
        _, monitor_mae = T1.evaluate(
            model, train_cache, monitor_rows, candidate="BAMF", device=device,
            batch_size=int(RECIPE["valid_batch_size"]), collate_fn=collate,
        )
        entry = {"epoch": int(epoch), "step": int(global_step),
                 "online_train_mae": float(online), "monitor_mae": float(monitor_mae),
                 "lr": float(last_lr), "seconds": float(time.perf_counter() - t_ep)}
        curve.append(entry)
        if monitor_mae < best["monitor_mae"]:
            best = {"monitor_mae": float(monitor_mae), "epoch": int(epoch), "step": int(global_step)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save(
                {"state_dict": best_state, "epoch": int(epoch), "step": int(global_step),
                 "monitor_mae": float(monitor_mae), "candidate": "BAMF", "seed": int(seed),
                 "arm": arm, "fit": "fit8k", "selection": "monitor"},
                out / "best_monitor.pt",
            )
        if epoch % 10 == 0 or global_step >= int(total_updates):
            C.log(f"[{ROUND_ID} {arm} s{seed}] ep={epoch} step={global_step} "
                  f"monitor={monitor_mae:.6f} online={online:.6f} "
                  f"best={best['monitor_mae']:.6f}@{best['epoch']} lr={last_lr:.2e} "
                  f"{entry['seconds']:.2f}s")
        if global_step >= int(total_updates):
            break

    final_state = C.cpu_state(model)
    C.atomic_torch_save(
        {"state_dict": final_state, "epoch": int(len(curve)), "step": int(global_step),
         "candidate": "BAMF", "seed": int(seed), "arm": arm, "fit": "fit8k",
         "selection": "final"},
        out / "final.pt",
    )
    wall = time.perf_counter() - started

    # Monitor-only reads; Locked is deliberately not touched.
    reads = {}
    for checkpoint, st in (("best_monitor", best_state), ("final", final_state)):
        C.load_state(model, st, device)
        pred_mon, mae_mon = T1.evaluate(
            model, train_cache, monitor_rows, candidate="BAMF", device=device,
            batch_size=int(RECIPE["valid_batch_size"]), collate_fn=collate,
        )
        reads[checkpoint] = {"monitor_mae": float(mae_mon), "monitor_pred": pred_mon.tolist()}
        C.atomic_torch_save(
            {"monitor_pred": np.asarray(pred_mon), "monitor_rows": monitor_rows,
             "checkpoint": checkpoint, "arm": arm, "seed": int(seed), "fit": "fit8k"},
            out / f"predictions_{checkpoint}.pt",
        )
    tail = curve[-max(1, len(curve) // 10):]
    info = {
        "round": ROUND_ID,
        "arm": arm,
        "seed": int(seed),
        "fit": "fit8k",
        "parameters": int(params),
        "parameter_audit": audit,
        "model_info": model_info,
        "norm": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
        "recipe": dict(RECIPE),
        "total_updates": int(total_updates),
        "steps_per_epoch": int(steps_per_epoch),
        "epochs_run": int(len(curve)),
        "n_train_rows": int(len(train_rows)),
        "split_hashes": split["hashes"],
        "best_monitor": best,
        "tail10_median_monitor_mae": float(np.median([e["monitor_mae"] for e in tail])),
        "tail10_median_online_train_mae": float(np.median([e["online_train_mae"] for e in tail])),
        "eval_reads": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_pred")}
                       for k, v in reads.items()},
        "first_grad_audit": first_grad_audit,
        "plan": None if plan is None else {
            "smax": int(plan.smax),
            "max_width": int(plan.max_width),
            "max_clique_size": int(plan.max_clique_size),
            "n_cliques": int(plan.graph_clique_ptr[-1].item()),
            "n_graphs": int(plan.n_graphs),
            "sha256": plan_hash(plan),
            "build_seconds": float(plan_seconds),
            "width_counts": [int((plan.widths == w).sum()) for w in range(int(plan.max_width) + 1)],
        },
        "wall_seconds": float(wall),
        "gpu_hours": float(wall / 3600.0),
        "rr_job_id": rr_job_id,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_loaded": False,
        "curve": curve,
    }
    C.write_json(out / "train.json", info)
    with (out / "curve.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    C.log(f"[{ROUND_ID} {arm} s{seed}] DONE bestMonitor={best['monitor_mae']:.6f}"
          f"@{best['epoch']} params={params} plan={None if plan is None else plan.max_width}"
          f" gpu_h={info['gpu_hours']:.3f}")
    return info


def evaluate_checkpoint(arm: str, seed: int, rows_name: str, checkpoint: str, device, *,
                        plan: B.GlobalPlan | None = None, tag: str | None = None) -> dict:
    """Evaluate a frozen checkpoint on Monitor1k or Locked1k (explicit caller only)."""
    split = DT.split_rows()
    rows = split[rows_name]
    train_cache = D.load_cache("train")
    if plan is None and arm == "G":
        plan = build_plan(train_cache)
    state: dict = {}
    collate = make_collate(arm, plan, device, state)
    model, _ = BM.build_model("BAMF", DT.fit_only_norm(train_cache, split["fit8k"]),
                              seed, device, arm=arm,
                              plan=None if plan is None else copy.deepcopy(plan).to(device))
    ck = torch.load(run_dir(arm, seed, tag) / f"{checkpoint}.pt", map_location="cpu",
                    weights_only=False)
    C.load_state(model, ck["state_dict"], device)
    pred, mae = T1.evaluate(model, train_cache, rows, candidate="BAMF", device=device,
                            batch_size=int(RECIPE["valid_batch_size"]), collate_fn=collate)
    y = np.asarray(train_cache["labels"], dtype=np.float64)[rows]
    return {"arm": arm, "seed": int(seed), "rows": rows_name, "checkpoint": checkpoint,
            "mae": float(mae), "pred": pred, "y": y, "rows_idx": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "evaluate"])
    p.add_argument("--arm", required=True, choices=["G", "I"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--total-updates", type=int, default=TOTAL_UPDATES)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rows", default="monitor", choices=["monitor", "locked"])
    p.add_argument("--checkpoint", default="best_monitor", choices=["best_monitor", "final"])
    p.add_argument("--rr-job-id", default=None)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "train":
        info = train(args.arm, args.seed, device, total_updates=args.total_updates,
                     tag=args.tag, force=args.force, rr_job_id=args.rr_job_id)
        print(json.dumps({k: info[k] for k in (
            "arm", "seed", "parameters", "total_updates", "epochs_run",
            "best_monitor", "tail10_median_monitor_mae", "eval_reads",
            "plan", "gpu_hours", "first_grad_audit")}, indent=2))
    else:
        out = evaluate_checkpoint(args.arm, args.seed, args.rows, args.checkpoint, device,
                                  tag=args.tag)
        print(f"{args.arm} s{args.seed} {args.rows} {args.checkpoint} MAE={out['mae']:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
