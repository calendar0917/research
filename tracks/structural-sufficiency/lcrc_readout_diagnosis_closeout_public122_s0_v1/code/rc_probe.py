"""Readout probe trainer for the closeout round (Parts 2 + 3).

Three frozen-representation readout probes on the *recovered* update-8165 source:

* ``P_WARM``  ``y = sum_v f(h_v) + rho(z_pool)``
* ``P_POST``  ``y = ... + delta([z_pool, psi(mean_v h_v)])``
* ``P_PRE``   ``y = ... + delta([z_pool, mean_v psi(h_v)])``

The science spec is unchanged from the upstream protocol §10.  The only
additions are the **branch-response measurements** required to decide whether
PRE/POST closeness means "the branch did not train" (it usually does not) or
"the extra channel formed a response but the held-out set did not improve".

The encoders (``atom_encoder`` / ``bond_encoder`` / ``local``) are frozen; only
``f``/``rho`` (warm-started) and ``psi``/``delta`` (private seed) are trained.
The clean ``h`` cache is identity-keyed and must match the recovered source state
hash, otherwise this refuses to run (fail-closed on source/cache mismatch).
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

import rc_common as R
import rc_source_audit as AUD
import lb_common as C
import lb_data as D
import lb_train as T
import lb_selection as SEL

PROCESS_T0 = time.perf_counter()

PRIMARY = "valid"          # probe grid is primary-criterion min Valid MSE
TRAIN_ROW_KEY = "train_fit"


# ---------------------------------------------------------------------------
# probe head (identical computation to the upstream probe)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """``dims`` deep linear ``SiLU`` MLP (last layer linear); mirrors dissect MLP."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


class ProbeHead(nn.Module):
    def __init__(self, rule: str, f_state: Mapping[str, torch.Tensor],
                 rho_state: Mapping[str, torch.Tensor]):
        super().__init__()
        if rule not in R.PROBE_ARMS:
            raise KeyError(rule)
        self.rule = rule
        self.width = 64
        self.rho_dim = 171
        self.f = _MLP([64, 64, 1])
        self.rho = _MLP([171, 64, 1])
        self.f.load_state_dict({k[2:]: v for k, v in f_state.items()})
        self.rho.load_state_dict({k[4:]: v for k, v in rho_state.items()})
        if rule in ("P_POST", "P_PRE"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(R.PROBE_PSI_SEED))
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

    def _counts(self, h, atom_graph, n_graphs):
        return h.new_zeros(n_graphs).index_add_(
            0, atom_graph, torch.ones_like(atom_graph, dtype=h.dtype))

    def extra_feature(self, h, atom_graph, n_graphs):
        """The 16-dim extra channel for POST/PRE; P_WARM has none."""
        if self.rule == "P_WARM":
            raise RuntimeError("P_WARM has no extra feature")
        n = self._counts(h, atom_graph, n_graphs)
        den = n.clamp(min=1.0).unsqueeze(1)
        if self.rule == "P_POST":
            mean_h = h.new_zeros((n_graphs, h.shape[1])).index_add_(0, atom_graph, h) / den
            return self.psi(mean_h)
        ph = self.psi(h)
        return ph.new_zeros((n_graphs, ph.shape[1])).index_add_(0, atom_graph, ph) / den

    def forward_frozen(self, h, atom_graph, z_pool, n_graphs):
        y = self.warm_only(h, atom_graph, z_pool, n_graphs)
        if self.rule == "P_WARM":
            return y
        extra = self.extra_feature(h, atom_graph, n_graphs)
        return y + self.delta(torch.cat([z_pool, extra], dim=1)).view(-1)

    def delta_out(self, z_pool, extra):
        return self.delta(torch.cat([z_pool, extra], dim=1)).view(-1)


# ---------------------------------------------------------------------------
# frozen-h view
# ---------------------------------------------------------------------------


class HView:
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
        counts = (self.atom_ptr[np.asarray(mols, np.int64) + 1] - self.atom_ptr[np.asarray(mols, np.int64)]
                  if mols else np.zeros(0, np.int64))
        atom_idx = (np.concatenate([np.arange(self.atom_ptr[m], self.atom_ptr[m + 1], dtype=np.int64)
                                    for m in mols]) if mols else np.zeros(0, np.int64))
        h = torch.from_numpy(self.h[atom_idx]).to(device)
        local_graph = np.repeat(np.arange(len(mols), dtype=np.int64), counts) if mols else np.zeros(0, np.int64)
        ag = torch.from_numpy(local_graph).to(device)
        zp = torch.from_numpy(self.z_pool(mols)).to(device)
        return h, ag, zp, len(mols)


# ---------------------------------------------------------------------------
# branch diagnostics (Part 3)
# ---------------------------------------------------------------------------


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().double().pow(2).sum().item())
    return math.sqrt(sq)


def _rel_disp(params: Sequence[torch.nn.Parameter], init: Sequence[torch.Tensor]) -> float:
    num = den = 0.0
    for p, i0 in zip(params, init):
        num += float((p.detach().double() - i0.double()).pow(2).sum().item())
        den += float(i0.double().pow(2).sum().item())
    return math.sqrt(num) / (math.sqrt(den) + 1e-12)


def _fixed_molecules(fit_positions: Sequence[int], global_rows: Sequence[int], n: int = R.DIAG_N_MOLECULES):
    """First ``n`` official-Train molecules ordered by *public* global row id (no labels)."""
    fit_positions = list(fit_positions)
    gr = np.asarray([int(global_rows[p]) for p in fit_positions], np.int64)
    order = np.argsort(gr, kind="stable")[:n]
    return [fit_positions[int(i)] for i in order], [int(gr[int(i)]) for i in order]


def branch_diagnostics(*, head, view: HView, mols, device, tnorm) -> dict[str, Any]:
    """Fixed-batch channel diagnostics; never changes mode/RNG of the training stream."""
    was = head.training
    st = T.rng_snapshot(device)
    head.eval()
    try:
        with torch.no_grad():
            h, ag, zp, ng = view.batch_tensors(mols, device)
            out: dict[str, Any] = {"n_molecules": int(ng)}
            if head.rule == "P_WARM":
                out["has_branch"] = False
                return out
            out["has_branch"] = True
            extra = head.extra_feature(h, ag, ng)
            n = head._counts(h, ag, ng)
            den = n.clamp(min=1.0).unsqueeze(1)
            mean_h = h.new_zeros((ng, h.shape[1])).index_add_(0, ag, h) / den
            psi_mean_h = head.psi(mean_h)
            psi_h = head.psi(h)
            mean_psi_h = psi_h.new_zeros((ng, psi_h.shape[1])).index_add_(0, ag, psi_h) / den

            # per-molecule difference between "mean psi(h)" and "psi(mean h)"
            diff = (mean_psi_h - psi_mean_h).detach().double().cpu().numpy()
            extra_scale = float(np.sqrt(np.mean(extra.detach().double().cpu().numpy() ** 2)))
            out["extra_feature_rms"] = extra_scale
            out["mean_psi_vs_psi_mean"] = {
                "per_mol_l2": np.linalg.norm(diff, axis=1).tolist(),
                "per_mol_l2_mean": float(np.linalg.norm(diff, axis=1).mean()),
                "per_mol_l2_median": float(np.median(np.linalg.norm(diff, axis=1))),
                "relative_to_extra_rms": float(np.linalg.norm(diff, axis=1).mean() / (extra_scale + 1e-12)),
                "nonzero_fraction": float(np.mean(np.abs(diff) > 1e-9)),
            }
            # full delta output and the internal channel-use response
            d_full = head.delta_out(zp, extra).detach().double().cpu().numpy()
            d_zero = head.delta_out(zp, torch.zeros_like(extra)).detach().double().cpu().numpy()
            resp = d_full - d_zero
            scale = float(tnorm["scale"])
            out["delta_full_std"] = {
                "rms": float(np.sqrt(np.mean(d_full ** 2))),
                "abs_mean": float(np.mean(np.abs(d_full))),
            }
            out["delta_channel_response"] = {
                "definition": "delta([z_pool, extra]) - delta([z_pool, 0])",
                "note": "internal channel-use diagnostic, NOT a legal molecular intervention or independent contribution",
                "std_rms": float(np.sqrt(np.mean(resp ** 2))),
                "raw_rms": float(np.sqrt(np.mean(resp ** 2)) * scale),
                "std_abs_mean": float(np.mean(np.abs(resp))),
                "raw_abs_mean": float(np.mean(np.abs(resp)) * scale),
                "relative_to_full_delta": float(
                    np.sqrt(np.mean(resp ** 2)) / (np.sqrt(np.mean(d_full ** 2)) + 1e-12)),
            }
            return out
    finally:
        T.rng_restore(st, device)
        head.train(was)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


def _load_hcache(identity_sha: str) -> tuple[dict[str, np.ndarray], Path]:
    ident = {"identity_sha256": identity_sha}
    path = R.hcache_path(ident)
    if not path.exists():
        raise FileNotFoundError(f"missing identity-keyed clean-h cache {path}; run rc_source_audit first")
    z = dict(np.load(path, allow_pickle=False))
    got = str(np.asarray(z["__identity_sha256"]).item())
    if got != identity_sha:
        raise RuntimeError(f"hcache identity {got} != expected {identity_sha}")
    return z, path


def _source_identity(ckpt_state_hash: str) -> dict[str, Any]:
    spec = D.frozen_spec()
    rows_hashes = {s: R.row_sha256(R.targets()["rows"][s]) for s in R.SPLITS}
    return R.hcache_identity(ckpt_state_hash=ckpt_state_hash, spec_hash=R.feature_spec_hash(spec),
                             norms=R.norm_hashes(), rows_hashes=rows_hashes)


def train_probe(*, rule: str, device: torch.device, out_dir: Path,
                smoke: bool = False, updates: int | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not smoke and R.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")

    src = R.read_json(R.ANALYSIS / "source_audit.json")
    if src.get("verdict") != "SOURCE_OK":
        raise SystemExit(f"source audit verdict {src.get('verdict')}; refusing formal probe run")
    ckpt_state_hash = src["source"]["state_hash"]

    spec = D.frozen_spec()
    snorm = R.structure_norm()
    tnorm = R.target_norm()
    tgt = R.targets()
    identity = _source_identity(ckpt_state_hash)
    z, hpath = _load_hcache(identity["identity_sha256"])
    views = {s: HView(z, s) for s in R.SPLITS}
    cache_file_sha = R.sha256_file(hpath)

    snap = torch.load(R.source_checkpoint(), map_location="cpu", weights_only=False)
    src_state = snap.get("state_dict", snap.get("model"))
    if src_state is None:
        raise KeyError("recovered source checkpoint has no state_dict/model")

    base = R.build_base_model(src_state, spec, snorm)
    f_state = {k: v for k, v in base.state_dict().items() if k.startswith("f.")}
    rho_state = {k: v for k, v in base.state_dict().items() if k.startswith("rho.")}
    del base

    head = ProbeHead(rule, f_state, rho_state).to(device)
    new_params = head.n_new_params()
    if rule in ("P_POST", "P_PRE") and new_params != R.PROBE_NEW_PARAMS:
        raise RuntimeError(f"{rule}: {new_params} new params != frozen {R.PROBE_NEW_PARAMS}")

    fit_info = R.fit_positions()
    fit_positions = fit_info["positions"]
    all_train_rows = tgt["rows"]["train"]
    n_fit = len(fit_positions)
    y_fit_all = np.asarray(tgt["y"]["train"], np.float64).reshape(-1)
    y_fit = y_fit_all[np.asarray(fit_positions, np.int64)]
    y_valid = np.asarray(tgt["y"]["valid"], np.float64).reshape(-1)
    y_test = np.asarray(tgt["y"]["test"], np.float64).reshape(-1)
    valid_mols = list(range(views["valid"].n))
    test_mols = list(range(views["test"].n))

    # step-0 equivalence + shared-init fixtures
    with torch.no_grad():
        _h, _ag, _zp, _ng = views["valid"].batch_tensors(list(range(32)), device)
        step0_equiv = (0.0 if rule == "P_WARM" else
                       float((head.forward_frozen(_h, _ag, _zp, _ng) - head.warm_only(_h, _ag, _zp, _ng)).abs().max()))
        del _h, _ag, _zp, _ng
    init_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    param_groups = {
        "f": list(head.f.parameters()),
        "rho": list(head.rho.parameters()),
    }
    if rule in ("P_POST", "P_PRE"):
        param_groups["psi"] = list(head.psi.parameters())
        param_groups["delta_first"] = [head.delta.lins[0].weight, head.delta.lins[0].bias]
        param_groups["delta_last"] = [head.delta.lins[1].weight, head.delta.lins[1].bias]
    init_by_group = {g: [p.detach().clone() for p in ps] for g, ps in param_groups.items()}

    n_updates = int(updates or R.PROBE_TRAIN_UPDATES)
    if smoke:
        n_updates = 4
    grid = sorted({0} | {int(round(j * n_updates / R.GRID_POINTS)) for j in range(1, R.GRID_POINTS + 1)})

    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=float(R.PROBE_LR), weight_decay=float(R.PROBE_WEIGHT_DECAY))
    plan = T.BatchPlan(n_fit, R.PROBE_BATCH)

    fixed_mols, fixed_rows = _fixed_molecules(fit_positions, all_train_rows)
    diag_updates = set(R.DIAG_UPDATES)

    def predict(view: HView, mols, batch_size=R.EVAL_BATCH):
        out = []
        for start in range(0, len(mols), batch_size):
            chunk = mols[start:start + batch_size]
            h, ag, zp, ng = view.batch_tensors(chunk, device)
            out.append(head.forward_frozen(h, ag, zp, ng).detach().float().cpu().numpy().reshape(-1))
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0, np.float64)

    def eval_all():
        was = head.training
        snap = T.rng_snapshot(device)
        head.eval()
        try:
            p_fit = predict(views["train"], fit_positions)
            p_valid = predict(views["valid"], valid_mols)
            p_test = predict(views["test"], test_mols)
        finally:
            T.rng_restore(snap, device)
            head.train(was)
        return {
            "train_fit": {**T.split_metrics("regression", p_fit, y_fit, tnorm), "_pred": p_fit},
            "valid": {**T.split_metrics("regression", p_valid, y_valid, tnorm), "_pred": p_valid},
            "test": {**T.split_metrics("regression", p_test, y_test, tnorm), "_pred": p_test},
        }

    selector = SEL.ProbeSelector("regression")
    curves: list[dict[str, Any]] = []
    grid_preds: dict[str, list[np.ndarray]] = {"grid_updates": []}
    learning_path: list[dict[str, Any]] = []

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
        zz = torch.as_tensor((y_fit[rows_local] - float(tnorm["mean"])) / float(tnorm["scale"]),
                             dtype=torch.float32, device=device)
        loss = F.mse_loss(pred, zz)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        if update in diag_updates:
            grad = {g: _grad_norm(ps) for g, ps in param_groups.items()}
            loss_val = float(loss.detach().cpu())
        opt.step()
        train_seconds += time.perf_counter() - t0

        if update in diag_updates:
            disp = {g: _rel_disp(ps, init_by_group[g]) for g, ps in param_groups.items()}
            rec = {
                "update": int(update),
                "loss_std_mse": loss_val,
                "optimized_target_scale": "standardised (z=(y-mean)/scale); loss is MSE in z-space",
                "grad_norm_preclip": grad,
                "relative_param_displacement": disp,
            }
            if rule in ("P_POST", "P_PRE"):
                rec["branch"] = branch_diagnostics(head=head, view=views["train"], mols=fixed_mols,
                                                   device=device, tnorm=tnorm)
                rec["branch"]["fixed_molecules_global_rows"] = fixed_rows
            learning_path.append(rec)

        if update in set(grid):
            do_eval(update, grid_point=True, is_epoch_end=False)
        elif last:
            do_eval(update, grid_point=False, is_epoch_end=True)

    wall = time.perf_counter() - started

    # ---- retention ------------------------------------------------------
    rules = selector.freeze()
    retained = {}
    rules_json = {}
    for r, rec in rules.items():
        rules_json[r] = {k: v for k, v in rec.items() if k != "state"}
        if rec["has_state"]:
            path = out_dir / f"best_{r}.pt"
            torch.save({"state_dict": rec["state"], "update": rec["update"], "criterion": rec["criterion"],
                        "rule": r, "probe": rule, "arm_id": f"{R.DATASET}_PROBE_{rule}",
                        "source_state_hash": ckpt_state_hash}, path)
            retained[r] = {"update": rec["update"], "criterion": rec["criterion"],
                           "path": R.track_rel(path), "rel": R.track_rel(path)}
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
    final_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    torch.save({"state_dict": final_state, "update": int(update), "probe": rule,
                "source_state_hash": ckpt_state_hash}, out_dir / "last.pt")

    payload = {
        "schema": "lcrc-readout-closeout-probe-v1", "round_id": R.ROUND_ID,
        "protocol_id": R.PROTOCOL_ID, "status_tag": R.STATUS_TAG,
        "status": "completed", "smoke": bool(smoke), "probe": rule, "dataset": R.DATASET,
        "task": "regression",
        "source": {"arm": R.SOURCE_ARM, "ckpt": R.track_rel(R.source_checkpoint()),
                   "internal_update": src["source"]["internal_update"],
                   "state_hash": ckpt_state_hash},
        "hcache": {"path": R.track_rel(hpath), "file_sha256": cache_file_sha,
                   "identity_sha256": identity["identity_sha256"]},
        "target_norm": tnorm, "norms": R.norm_hashes(),
        "frozen_encoder": True, "new_params": int(new_params), "shared_head_warm_start": True,
        "train": {"updates": int(n_updates), "batch_size": R.PROBE_BATCH, "lr": R.PROBE_LR,
                  "weight_decay": R.PROBE_WEIGHT_DECAY, "scheduler": "none", "n_fit": n_fit,
                  "fit_positions_sha256": fit_info["positions_sha256"],
                  "grid_updates": grid, "psi_seed": R.PROBE_PSI_SEED,
                  "step0_selectable": True, "dropout": False},
        "selection": {"rules": rules_json, "retained": retained},
        "step0_delta_equivalence_max_abs": step0_equiv,
        "param_hashes": {"init": R.state_hash(init_state), "final": R.state_hash(final_state)},
        "cost": {"train_seconds": float(train_seconds), "eval_seconds": float(eval_seconds),
                 "wall_seconds": float(wall),
                 "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
                 "gpu_hours_charged": float((time.perf_counter() - PROCESS_T0) / 3600.0)
                 if device.type == "cuda" else 0.0},
        "environment": {**R.env_info(), "device": str(device)},
        "provenance": {"git_commit": R.git_commit(), "upstream_hashes": R.upstream_hashes(),
                       "code_hashes": R.code_hashes()},
    }
    R.write_json(out_dir / "train.json", payload)
    R.write_json(out_dir / "learning_path.json",
                 {"probe": rule, "source_state_hash": ckpt_state_hash,
                  "diag_updates": list(R.DIAG_UPDATES), "records": learning_path})
    R.log(f"[PROBE {rule}] updates={n_updates} new_params={new_params} wall={wall:.1f}s")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="closeout frozen-readout probe")
    ap.add_argument("--rule", required=True, choices=list(R.PROBE_ARMS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    train_probe(rule=args.rule, device=device, out_dir=Path(args.out_dir),
                smoke=args.smoke, updates=args.updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
