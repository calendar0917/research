"""Frozen-representation readout probes (P_WARM / P_POST / P_PRE).

Extract the clean atom representation ``h`` (before dropout) of the frozen
``C1_R1`` fit100 U0-stage R_SEL checkpoint, then train **only readout heads** on
that frozen ``h``.  Encoders (``atom_encoder`` / ``bond_encoder`` / ``local``)
are never updated and are used in eval mode.

The three probes share the *same* warm-started, trainable ``f`` / ``rho`` and
differ only in where an additional nonlinearity feeds the extra head ``delta``:

* ``P_WARM``  ``y = sum_v f(h_v) + rho(z_pool)``
* ``P_POST``  ``y = ... + delta([z_pool(171), psi(mean_v h_v)(16)])``
* ``P_PRE``   ``y = ... + delta([z_pool(171), mean_v psi(h_v)(16)])``

``z_pool = [g42, mean_h64, mean_h2_64, log1p n]`` (171 dims).  ``psi``/``delta``
use a private seed 1101 and have **exactly 14705** new parameters
(psi 64->32->16, delta 187->64->1).  ``delta``'s final Linear starts at zero, so
at step 0 all three probes reproduce the warm-started ``P_WARM`` prediction.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lb_common as C
import lb_data as D
import lb_train as T
import lb_selection as SEL

import lcpr_common as PC
import lcpr_model as PM

PROCESS_T0 = time.perf_counter()

PROBE_GRID = 50


# ---------------------------------------------------------------------------
# h extraction
# ---------------------------------------------------------------------------


def hcache_path(dataset: str) -> Path:
    return C.CACHE / f"hcache_{dataset}.npy.npz"


def extract_h(*, dataset: str, ckpt: Path, device: torch.device, force: bool = False) -> dict[str, Any]:
    """Run the frozen trunk once and cache clean ``h`` + pooled inputs per split."""
    path = hcache_path(dataset)
    if path.exists() and not force:
        z = np.load(path, allow_pickle=False)
        return {"dataset": dataset, "path": C.track_rel(path), "reused": True,
                "n_splits": len([k for k in z.files if k.endswith("_h")])}
    spec = D.frozen_spec()
    snorm = D.structure_norm(dataset)
    tgt = D.targets(dataset, include_test=True)
    kind = "c1"
    model = PM.build_model(spec, snorm, kind, variant="C1")
    snap = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(snap["state_dict"])
    model.to(device)
    model.eval()
    gm = model.global_mean.detach().cpu().numpy()
    gs = model.global_scale.detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    for split in C.SPLITS:
        cache = D.load_cache(dataset, split, tgt["rows"][split])
        n = int(cache["n_molecules"])
        atom_ptr = np.asarray(cache["atom_ptr"], np.int64)
        h_all: list[np.ndarray] = []
        count_all = np.zeros(n, np.float64)
        m1 = np.zeros((n, 64), np.float64)
        m2 = np.zeros((n, 64), np.float64)
        snap_rng = T.rng_snapshot(device)
        with torch.no_grad():
            for start in range(0, n, C.EVAL_BATCH):
                rows = list(range(start, min(start + C.EVAL_BATCH, n)))
                batch = D.to_device(D.collate(cache, rows), device)
                aux = model(batch, return_aux=True)
                h = aux["h_clean"].detach().float().cpu().numpy()
                h_all.append(h)
                g = batch["atom_batch"].detach().cpu().numpy()
                m1[start:start + len(rows)] = aux["mean_h"].detach().float().cpu().numpy()
                m2[start:start + len(rows)] = aux["mean_h2"].detach().float().cpu().numpy()
                count_all[start:start + len(rows)] = aux["count"].detach().float().cpu().numpy()
        T.rng_restore(snap_rng, device)
        h_cat = np.concatenate(h_all, axis=0).astype(np.float32)
        out[f"{split}_h"] = h_cat
        out[f"{split}_atom_ptr"] = atom_ptr.astype(np.int64)
        out[f"{split}_atom_graph"] = np.repeat(np.arange(n, dtype=np.int64),
                                               np.diff(atom_ptr)) if n else np.zeros(0, np.int64)
        out[f"{split}_count"] = count_all.astype(np.float32)
        out[f"{split}_mean_h"] = m1.astype(np.float32)
        out[f"{split}_mean_h2"] = m2.astype(np.float32)
        gnorm = (np.asarray(cache["global"], np.float64) - gm) / gs
        out[f"{split}_g"] = gnorm.astype(np.float32)
        meta[split] = {"n_molecules": n, "n_atoms": int(atom_ptr[-1]),
                       "rows_sha256": C.row_sha256(tgt["rows"][split])}
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return {"dataset": dataset, "path": C.track_rel(path), "reused": False, "meta": meta,
            "source_ckpt": C.track_rel(ckpt)}


# ---------------------------------------------------------------------------
# probe model
# ---------------------------------------------------------------------------


class ProbeHead(nn.Module):
    def __init__(self, rule: str, f_state: Mapping[str, torch.Tensor],
                 rho_state: Mapping[str, torch.Tensor]):
        super().__init__()
        if rule not in C.PROBE_ARMS:
            raise KeyError(rule)
        self.rule = rule
        self.width = 64
        self.rho_dim = 171
        self.f = PM.DM.MLP([64, 64, 1]) if False else _MLP([64, 64, 1])
        self.rho = _MLP([171, 64, 1])
        self.f.load_state_dict({k[2:]: v for k, v in f_state.items()})
        self.rho.load_state_dict({k[4:]: v for k, v in rho_state.items()})
        if rule in ("P_POST", "P_PRE"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(C.PROBE_PSI_SEED))
                self.psi = _MLP([64, 32, 16])
                self.delta = _MLP([171 + 16, 64, 1])
                nn.init.zeros_(self.delta.lins[-1].weight)
                nn.init.zeros_(self.delta.lins[-1].bias)

    def n_new_params(self) -> int:
        if self.rule == "P_WARM":
            return 0
        return (int(sum(p.numel() for p in self.psi.parameters()))
                + int(sum(p.numel() for p in self.delta.parameters())))

    # -- pieces -----------------------------------------------------------
    def warm_only(self, h, atom_graph, z_pool, n_graphs):
        contrib_atom = self.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_graph, contrib_atom)
        return contrib + self.rho(z_pool).view(-1)

    def forward_frozen(self, h, atom_graph, z_pool, n_graphs):
        y = self.warm_only(h, atom_graph, z_pool, n_graphs)
        if self.rule == "P_WARM":
            return y
        if self.rule == "P_POST":
            n = h.new_zeros(n_graphs).index_add_(0, atom_graph, torch.ones_like(atom_graph, dtype=h.dtype))
            den = n.clamp(min=1.0).unsqueeze(1)
            mean_h = h.new_zeros((n_graphs, h.shape[1])).index_add_(0, atom_graph, h) / den
            extra = self.psi(mean_h)
        else:  # P_PRE
            ph = self.psi(h)
            n = h.new_zeros(n_graphs).index_add_(0, atom_graph, torch.ones_like(atom_graph, dtype=h.dtype))
            den = n.clamp(min=1.0).unsqueeze(1)
            extra = ph.new_zeros((n_graphs, ph.shape[1])).index_add_(0, atom_graph, ph) / den
        return y + self.delta(torch.cat([z_pool, extra], dim=1)).view(-1)


class _MLP(nn.Module):
    """``dims`` deep linear ``SiLU`` MLP (last layer linear); mirrors ``dissect_model.MLP``."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


# ---------------------------------------------------------------------------
# probe run
# ---------------------------------------------------------------------------


class HView:
    """Frozen-``h`` view over one split; provides per-batch tensors."""

    def __init__(self, z: Mapping[str, np.ndarray], split: str):
        self.h = np.asarray(z[f"{split}_h"], np.float32)
        self.atom_graph = np.asarray(z[f"{split}_atom_graph"], np.int64)
        self.atom_ptr = np.asarray(z[f"{split}_atom_ptr"], np.int64)
        self.count = np.asarray(z[f"{split}_count"], np.float32)
        self.mean_h = np.asarray(z[f"{split}_mean_h"], np.float32)
        self.mean_h2 = np.asarray(z[f"{split}_mean_h2"], np.float32)
        self.g = np.asarray(z[f"{split}_g"], np.float32)
        self.n = int(len(self.atom_ptr) - 1)

    def z_pool(self, mols: Sequence[int]) -> np.ndarray:
        mols = np.asarray(list(mols), np.int64)
        return np.concatenate([
            self.g[mols], self.mean_h[mols], self.mean_h2[mols],
            np.log1p(self.count[mols])[:, None].astype(np.float32),
        ], axis=1)

    def batch_tensors(self, mols: Sequence[int], device):
        mols = [int(m) for m in mols]
        counts = self.atom_ptr[np.asarray(mols, np.int64) + 1] - self.atom_ptr[np.asarray(mols, np.int64)] if mols \
            else np.zeros(0, np.int64)
        atom_idx = np.concatenate([np.arange(self.atom_ptr[m], self.atom_ptr[m + 1], dtype=np.int64)
                                   for m in mols]) if mols else np.zeros(0, np.int64)
        h = torch.from_numpy(self.h[atom_idx]).to(device)
        local_graph = np.repeat(np.arange(len(mols), dtype=np.int64), counts) if mols else np.zeros(0, np.int64)
        ag = torch.from_numpy(local_graph).to(device)
        zp = torch.from_numpy(self.z_pool(mols)).to(device)
        return h, ag, zp, len(mols)


def train_probe(*, rule: str, dataset: str, ckpt: Path, device: torch.device, out_dir: Path,
                smoke: bool = False, updates: int | None = None) -> dict[str, Any]:
    import lb_train as T  # local import to avoid a cycle at module import
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")
    task = D.dataset_meta(dataset)["task"]
    spec = D.frozen_spec()
    snorm = D.structure_norm(dataset)
    tgt = D.targets(dataset, include_test=True)
    tnorm_info = D.shared_target_norm(dataset)
    tnorm = tnorm_info["target_norm"]

    base = PM.build_model(spec, snorm, "c1", variant="C1")
    snap = torch.load(ckpt, map_location="cpu", weights_only=False)
    base.load_state_dict(snap["state_dict"])

    z = dict(np.load(hcache_path(dataset), allow_pickle=False))
    views = {s: HView(z, s) for s in C.SPLITS}

    subset = D.nested_subset_positions(dataset, "100")
    fit_positions = subset["positions"]              # official-Train positions
    # fit view is the *official Train* cache, restricted later to fit positions
    n_fit = len(fit_positions)
    y_fit_all = np.asarray(tgt["y"]["train"], np.float64).reshape(-1)
    y_fit = y_fit_all[np.asarray(fit_positions, np.int64)]

    f_state = {k: v for k, v in base.state_dict().items() if k.startswith("f.")}
    rho_state = {k: v for k, v in base.state_dict().items() if k.startswith("rho.")}
    head = ProbeHead(rule, f_state, rho_state).to(device)
    new_params = head.n_new_params()
    if rule in ("P_POST", "P_PRE") and new_params != C.PROBE_NEW_PARAMS:
        raise RuntimeError(f"{rule}: {new_params} new params != frozen {C.PROBE_NEW_PARAMS}")
    # step-0 equivalence fixture: at initialisation P_POST/P_PRE must reproduce the
    # frozen-base (P_WARM) prediction exactly (delta's last layer starts at zero).
    with torch.no_grad():
        _h0, _ag0, _zp0, _ng0 = HView(z, "valid").batch_tensors(list(range(32)), device)
        step0_equiv = (0.0 if rule == "P_WARM" else
                       float((head.forward_frozen(_h0, _ag0, _zp0, _ng0)
                              - head.warm_only(_h0, _ag0, _zp0, _ng0)).abs().max()))
    del _h0, _ag0, _zp0, _ng0

    n_updates = int(updates or C.PROBE_TRAIN_UPDATES)
    if smoke:
        n_updates = 4
    grid = sorted({0} | {int(round(j * n_updates / PROBE_GRID)) for j in range(1, PROBE_GRID + 1)})

    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=float(C.PROBE_LR), weight_decay=float(C.PROBE_WEIGHT_DECAY))
    plan = T.BatchPlan(n_fit, C.PROBE_BATCH)
    valid_mols = list(range(views["valid"].n))
    test_mols = list(range(views["test"].n))
    fit_mols = fit_positions                          # official-Train positions == fit view positions
    y_valid = np.asarray(tgt["y"]["valid"], np.float64).reshape(-1)
    y_test = np.asarray(tgt["y"]["test"], np.float64).reshape(-1)

    selector = SEL.ProbeSelector(task)

    def predict(view: HView, mols, batch_size=C.EVAL_BATCH):
        out = []
        for start in range(0, len(mols), batch_size):
            chunk = mols[start:start + batch_size]
            h, ag, zp, ng = view.batch_tensors(chunk, device)
            out.append(head.forward_frozen(h, ag, zp, ng).detach().float().cpu().numpy().reshape(-1))
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0, np.float64)

    def eval_all():
        was = head.training
        snap_rng = T.rng_snapshot(device)
        head.eval()
        try:
            p_fit = predict(views["train"], fit_mols)
            p_valid = predict(views["valid"], valid_mols)
            p_test = predict(views["test"], test_mols)
        finally:
            T.rng_restore(snap_rng, device)
            head.train(was)
        return {
            "train_fit": {**T.split_metrics(task, p_fit, y_fit, tnorm), "_pred": p_fit},
            "valid": {**T.split_metrics(task, p_valid, y_valid, tnorm), "_pred": p_valid},
            "test": {**T.split_metrics(task, p_test, y_test, tnorm), "_pred": p_test},
        }

    curves: list[dict[str, Any]] = []
    grid_preds: dict[str, list[np.ndarray]] = {"grid_updates": []}

    def strip(ev):
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in ev.items()}

    def do_eval(update, *, grid_point, is_epoch_end):
        ev = eval_all()
        s = strip(ev)
        rec = {"update": int(update), "is_grid": bool(grid_point), "is_epoch_end": bool(is_epoch_end)}
        for name in ("train_fit", "valid", "test"):
            for k, v in s[name].items():
                if isinstance(v, (int, float)):
                    rec[f"{name}_{k}"] = float(v)
        curves.append(rec)
        if grid_point:
            selector.consider(update, s["valid"], head.state_dict())
            grid_preds["grid_updates"].append(int(update))
            for name in ("train_fit", "valid", "test"):
                grid_preds.setdefault(name, []).append(np.asarray(ev[name]["_pred"], np.float64))

    started = time.perf_counter()
    train_seconds = 0.0
    eval_seconds = 0.0
    update = 0
    t0 = time.perf_counter(); do_eval(0, grid_point=True, is_epoch_end=True); eval_seconds += time.perf_counter() - t0
    while update < n_updates:
        head.train()
        last = plan.last_of_epoch()
        rows_local = plan.next_batch()
        update += 1
        mols = [fit_positions[i] for i in rows_local]
        h, ag, zp, ng = views["train"].batch_tensors(mols, device)
        pred = head.forward_frozen(h, ag, zp, ng)
        t0 = time.perf_counter()
        if task == "regression":
            zz = torch.as_tensor((y_fit[rows_local] - float(tnorm["mean"])) / float(tnorm["scale"]),
                                 dtype=torch.float32, device=device)
            loss = F.mse_loss(pred, zz)
        else:
            yy = torch.as_tensor(y_fit[rows_local], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(pred, yy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_seconds += time.perf_counter() - t0
        if update in set(grid):
            do_eval(update, grid_point=True, is_epoch_end=False)
        elif last:
            do_eval(update, grid_point=False, is_epoch_end=True)
    wall = time.perf_counter() - started

    rules = selector.freeze()
    retained = {}
    rules_json = {}
    for r, rec in rules.items():
        rules_json[r] = {k: v for k, v in rec.items() if k != "state"}
        if rec["has_state"]:
            path = out_dir / f"best_{r}.pt"
            torch.save({"state_dict": rec["state"], "update": rec["update"], "criterion": rec["criterion"],
                        "rule": r, "probe": rule, "arm_id": f"{dataset}_PROBE_{rule}"}, path)
            retained[r] = {"update": rec["update"], "criterion": rec["criterion"], "path": C.track_rel(path)}
        else:
            retained[r] = {"update": None, "criterion": None, "path": None}

    all_fields: list[str] = []
    for row in curves:
        for k in row:
            if k not in all_fields:
                all_fields.append(k)
    with open(out_dir / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields)
        w.writeheader(); w.writerows(curves)
    np.savez_compressed(out_dir / "grid_predictions.npz",
                        grid_updates=np.asarray(grid_preds["grid_updates"], np.int64),
                        fit_rows=np.asarray([tgt["rows"]["train"][p] for p in fit_positions], np.int64),
                        valid_rows=np.asarray(tgt["rows"]["valid"], np.int64),
                        test_rows=np.asarray(tgt["rows"]["test"], np.int64),
                        fit_y=y_fit, valid_y=y_valid, test_y=y_test,
                        **{name: np.stack(grid_preds[name], axis=0) for name in ("train_fit", "valid", "test")})

    payload = {
        "schema": "lcrc-bottleneck-diagnosis-probe-v1", "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID, "status": "completed", "smoke": bool(smoke),
        "probe": rule, "dataset": dataset, "task": task,
        "source_ckpt": C.track_rel(ckpt), "source_update": int(snap.get("update", -1)),
        "frozen_encoder": True, "new_params": int(new_params),
        "shared_head_warm_start": True,
        "train": {"updates": int(n_updates), "batch_size": C.PROBE_BATCH, "lr": C.PROBE_LR,
                  "weight_decay": C.PROBE_WEIGHT_DECAY, "scheduler": "none",
                  "n_fit": n_fit, "fit_positions_sha256": subset["positions_sha256"],
                  "grid_updates": grid, "psi_seed": C.PROBE_PSI_SEED,
                  "step0_selectable": True},
        "selection": {"rules": rules_json, "retained": retained},
        "step0_delta_equivalence_max_abs": step0_equiv,
        "cost": {"train_seconds": float(train_seconds), "eval_seconds": float(eval_seconds),
                 "wall_seconds": float(wall), "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
                 "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0)
                 if device.type == "cuda" else 0.0},
        "environment": {**C.env_info(), "device": str(device)},
        "provenance": {"git_commit": C.git_commit()},
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(f"[PROBE {rule}@{dataset}] updates={n_updates} new_params={new_params} wall={wall:.1f}s")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="frozen-readout probe")
    ap.add_argument("--rule", required=True, choices=list(C.PROBE_ARMS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if args.extract_only:
        print(extract_h(dataset=args.dataset, ckpt=Path(args.ckpt), device=device))
        return 0
    train_probe(rule=args.rule, dataset=args.dataset, ckpt=Path(args.ckpt), device=device,
                out_dir=Path(args.out_dir), smoke=args.smoke, updates=args.updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
