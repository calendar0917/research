"""CSCL-v0 driver: audit / train / synthetic modes (research track
``structure-contribution``; protocol notes/v0_protocol.md).

Examples
--------
CPU label-blind unit audit::

    uv run python -m tracks.structure_contribution.code.run_cscl_v0 audit \
        --out tracks/structure_contribution/results/cscl_v0_units_audit

Train one arm/seed (GPU via pool res-gpu1 uses ``--device cuda:0``)::

    uv run python -m tracks.structure_contribution.code.run_cscl_v0 train \
        --arm relational --seed 0 --device cuda:0 --out <dir>

XGBoost reference::

    uv run python -m tracks.structure_contribution.code.run_cscl_v0 train \
        --arm xgb --seed 0 --device cpu --out <dir>

Synthetic ground-truth run::

    uv run python -m tracks.structure_contribution.code.run_cscl_v0 synthetic \
        --arm relational --seed 0 --device cpu --out <dir>

Official ZINC test is never loaded in any mode (v0_protocol §1).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import cscl_models as cm  # noqa: E402
import cscl_synthetic as csyn  # noqa: E402
from cscl_units import KIND_CHAIN, KIND_RING  # noqa: E402

ARMS = ("additive", "relational", "shuffled", "opaque", "xgb")
RESULTS_SUBDIR = "tracks/structure-contribution/results"


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------


class MolTensors:
    """Precomputed per-molecule tensors (type ids resolved once)."""

    __slots__ = ("type_ids", "desc", "rel_type_lo", "rel_type_hi", "rel_feat", "rel_unit_lo", "rel_unit_hi")

    def __init__(self, mol: cf.MolUnits, stats: cf.FitStats) -> None:
        tids = [stats.type_id(mol, k) for k in range(len(mol.unit_sigs))]
        self.type_ids = np.asarray(tids, dtype=np.int64)
        self.desc = mol.desc.astype(np.float32)
        lo, hi, flo, fhi = [], [], [], []
        for (a, b), feat in zip(mol.rel_pairs, mol.rel_feat):
            ta, tb = int(self.type_ids[a]), int(self.type_ids[b])
            if ta > tb:
                ta, tb = tb, ta
                a, b = b, a
            lo.append(a)
            hi.append(b)
            flo.append(ta)
            fhi.append(tb)
        self.rel_type_lo = np.asarray(flo, dtype=np.int64)
        self.rel_type_hi = np.asarray(fhi, dtype=np.int64)
        self.rel_unit_lo = np.asarray(lo, dtype=np.int64)
        self.rel_unit_hi = np.asarray(hi, dtype=np.int64)
        self.rel_feat = mol.rel_feat.astype(np.float32)


def build_tensors(mols: list[cf.MolUnits], stats: cf.FitStats) -> list[MolTensors]:
    return [MolTensors(m, stats) for m in mols]


def assemble(mols: list[MolTensors]) -> cm.Batch:
    type_ids, desc, unit_mol = [], [], []
    rlo, rhi, rfeat, rmol, rulo, ruhi = [], [], [], [], [], []
    n_units = 0
    for i, m in enumerate(mols):
        k = len(m.type_ids)
        type_ids.append(m.type_ids)
        desc.append(m.desc)
        unit_mol.append(np.full(k, i, dtype=np.int64))
        if len(m.rel_type_lo):
            rlo.append(m.rel_type_lo)
            rhi.append(m.rel_type_hi)
            rfeat.append(m.rel_feat)
            rmol.append(np.full(len(m.rel_type_lo), i, dtype=np.int64))
            rulo.append(m.rel_unit_lo + n_units)
            ruhi.append(m.rel_unit_hi + n_units)
        n_units += k
    return cm.Batch(
        n_mols=len(mols),
        type_ids=torch.from_numpy(np.concatenate(type_ids)),
        desc=torch.from_numpy(np.concatenate(desc)),
        unit_mol=torch.from_numpy(np.concatenate(unit_mol)),
        rel_type_lo=torch.from_numpy(np.concatenate(rlo) if rlo else np.zeros(0, np.int64)),
        rel_type_hi=torch.from_numpy(np.concatenate(rhi) if rhi else np.zeros(0, np.int64)),
        rel_feat=torch.from_numpy(np.concatenate(rfeat) if rfeat else np.zeros((0, cf.REL_DIM), np.float32)),
        rel_mol=torch.from_numpy(np.concatenate(rmol) if rmol else np.zeros(0, np.int64)),
        rel_unit_lo=torch.from_numpy(np.concatenate(rulo) if rulo else np.zeros(0, np.int64)),
        rel_unit_hi=torch.from_numpy(np.concatenate(ruhi) if ruhi else np.zeros(0, np.int64)),
    )


# ---------------------------------------------------------------------------
# training (A/B/C/D)
# ---------------------------------------------------------------------------


def set_model_centering(model, stats: cf.FitStats) -> None:
    type_means = {t: torch.from_numpy(mu) for t, mu in stats.type_means.items()}
    global_mean = torch.from_numpy(stats.global_mean)
    rel_mean = torch.from_numpy(stats.rel_mean)
    model.set_centering(type_means, global_mean, rel_mean)


def train_torch_arm(arm: str, seed: int, device: str, data: dict, epochs: int = 300, patience: int = 30, batch_size: int = 128, lr: float = 1e-3, wd: float = 1e-5, log=print) -> dict:
    """Shared training loop; data holds fit_inner/monitor/dev MolTensors + y."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device)

    fit_ts, mon_ts, dev_ts = data["fit_inner"], data["monitor"], data["dev"]
    y = data["y"]  # raw [N,1] float
    stats: cf.FitStats = data["stats"]

    model = cm.build_model(arm, stats.vocab.n_total, seed).to(dev)
    set_model_centering(model, stats)

    y_inner = y[data["fit_inner_idx"], 0]
    y_mon = y[data["monitor_idx"], 0]
    y_dev = y[data["dev_idx"], 0]
    y_mean, y_std = stats.y_mean, stats.y_std

    fit_batches = [assemble(fit_ts[i : i + batch_size]) for i in range(0, len(fit_ts), batch_size)]
    mon_batch = assemble(mon_ts)
    dev_batch = assemble(dev_ts)
    mon_y = torch.from_numpy((y_mon - y_mean) / y_std).to(dev)
    dev_y = torch.from_numpy((y_dev - y_mean) / y_std).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen_train = torch.Generator().manual_seed(int(seed))
    gen_eval = torch.Generator().manual_seed(int(seed) + 777)

    history: list[dict] = []
    ckpts: list[tuple[float, dict]] = []  # (monitor mae, state)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(fit_batches), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for bi in order.tolist():
            batch = fit_batches[bi].to(dev)
            target = batch.desc.new_zeros(batch.n_mols)
            # standardized target for this batch
            ys = torch.from_numpy(
                (y[data["fit_inner_idx"][bi * batch_size : (bi + 1) * batch_size], 0] - y_mean) / y_std
            ).to(dev)
            loss, _ = cm.train_loss(model, batch, ys, arm, gen_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mb = mon_batch.to(dev)
            mon_out = cm.forward_arm(model, mb, arm, gen_eval)
            mon_mae = float((mon_out.pred - mon_y).abs().mean()) * y_std if not isinstance(model, cm.OpaqueModel) else float((model(mb) - mon_y).abs().mean()) * y_std
        history.append({"epoch": epoch, "monitor_mae": mon_mae})
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ckpts.append((mon_mae, state))
        ckpts.sort(key=lambda x: x[0])
        if len(ckpts) > 5:
            ckpts.pop()
        if epoch % 25 == 0 or epoch == 1:
            log(f"[{arm} s{seed}] epoch {epoch} monitor_mae {mon_mae:.5f} (raw units)")
        # early stop: no monitor improvement for `patience` epochs
        best_epoch = int(np.argmin([h["monitor_mae"] for h in history]))
        if epoch - (best_epoch + 1) >= patience:
            break

    # top-5 soup (v0_protocol §3) — primary performance metric; NOTE: soup
    # averages embedding tables across epochs, which destroys attribution
    # geometry, so the *interpretation object* is the best-epoch checkpoint
    # (see notes/v0_protocol_addendum.md)
    soup_states = [s for _, s in ckpts]
    avg = {k: torch.stack([s[k].float() for s in soup_states]).mean(0) for k in soup_states[0]}
    best_state = dict(ckpts[0][1])
    model.load_state_dict(avg)
    model.eval()

    wall = time.time() - t0
    with torch.no_grad():
        db = dev_batch.to(dev)
        if isinstance(model, cm.OpaqueModel):
            dev_pred_std = model(db)
        else:
            dev_pred_std = cm.forward_arm(model, db, arm, gen_eval).pred
        dev_mae = float((dev_pred_std - dev_y).abs().mean()) * y_std
        # best-epoch (non-soup) dev MAE — the interpretation object's metric
        model.load_state_dict(best_state)
        model.eval()
        if isinstance(model, cm.OpaqueModel):
            dev_pred_std_b = model(db)
        else:
            dev_pred_std_b = cm.forward_arm(model, db, arm, gen_eval).pred
        best_dev_mae = float((dev_pred_std_b - dev_y).abs().mean()) * y_std
        fb_best = torch.tensor(0.0)
        # fit MAE on fit_inner (first 2000 rows to bound cost; full fit reported too)
        fit_eval_batches = [assemble(fit_ts[i : i + 512]).to(dev) for i in range(0, len(fit_ts), 512)]
        preds = []
        for fb in fit_eval_batches:
            p = model(fb) if isinstance(model, cm.OpaqueModel) else cm.forward_arm(model, fb, arm, gen_eval).pred
            preds.append(p)
        fit_pred_std = torch.cat(preds)
        fit_mae = float((fit_pred_std - torch.from_numpy((y_inner - y_mean) / y_std).to(dev)).abs().mean()) * y_std
        best_mon = min(h["monitor_mae"] for h in history)
    n_params = int(sum(p.numel() for p in model.parameters()))
    out = {
        "arm": arm,
        "seed": seed,
        "device": str(dev),
        "soup_dev_mae": dev_mae,
        "best_epoch_dev_mae": best_dev_mae,
        "fit_inner_mae": fit_mae,
        "best_monitor_mae": best_mon,
        "best_epoch": int(np.argmin([h["monitor_mae"] for h in history])) + 1,
        "epochs_run": len(history),
        "n_params": n_params,
        "wall_seconds": wall,
        "y_mean": y_mean,
        "y_std": y_std,
        "soup_members_monitor_mae": [float(m) for m, _ in ckpts],
    }
    # per-molecule dev predictions (soup model) for paired comparisons
    with torch.no_grad():
        model.load_state_dict(avg)
        model.eval()
        out["_dev_pred"] = dev_pred_std.detach().cpu().numpy().astype(np.float32)
        out["_dev_row"] = data["dev_idx"].astype(np.int64)
    # keep soup + best states for save/load round-trip tests and export
    out["_state"] = {k: v for k, v in avg.items()}
    out["_best_state"] = best_state
    log(f"[{arm} s{seed}] soup dev MAE {dev_mae:.5f} | fit MAE {fit_mae:.5f} | params {n_params} | {wall:.0f}s")
    return out


# ---------------------------------------------------------------------------
# contribution export (v0_protocol §6.1)
# ---------------------------------------------------------------------------


def export_contributions(model, arm: str, seed: int, data: dict, device: str, out_dir: Path, max_mols: int | None = None) -> dict:
    """Per-molecule/per-unit/per-relation contribution table + completeness."""
    dev = torch.device(device)
    stats: cf.FitStats = data["stats"]
    mols = data["dev_mols"]
    ts = data["dev"]
    y_dev = data["y"][data["dev_idx"], 0]
    if max_mols is not None:
        mols = mols[:max_mols]
        ts = ts[:max_mols]
        y_dev = y_dev[:max_mols]
    gen = torch.Generator().manual_seed(int(seed) + 777)

    model = model.to(dev)
    model.eval()
    rows_unit: list[dict] = []
    rows_rel: list[dict] = []
    worst = 0.0
    all_pred: list[np.ndarray] = []
    n = len(ts)
    for start in range(0, n, 256):
        chunk = ts[start : start + 256]
        batch = assemble(chunk).to(dev)
        with torch.no_grad():
            out = cm.forward_arm(model, batch, arm, gen)
        # per-molecule sums recomputed from contributions
        alpha = out.alpha.cpu().numpy()
        delta = out.delta.cpu().numpy()
        gamma = out.gamma.cpu().numpy()
        umol = batch.unit_mol.cpu().numpy()
        rmol = batch.rel_mol.cpu().numpy()
        pred = out.pred.cpu().numpy()
        all_pred.append(pred)
        b = float(out.bias.reshape(-1)[0])
        for i in range(len(chunk)):
            mu = umol == i
            mr = rmol == i
            recon = b + float((alpha[mu] + delta[mu]).sum()) + float(gamma[mr].sum())
            worst = max(worst, abs(recon - pred[i]))
            m = mols[start + i]
            k = 0
            for j in np.where(mu)[0]:
                kk = int(j)
                rows_unit.append(
                    {
                        "mol_row": int(data["dev_idx"][start + i]),
                        "unit_id": k,
                        "type_id": int(batch.type_ids[kk]),
                        "kind": m.unit_kinds[k],
                        "n_atoms": len(m.unit_atoms[k]),
                        "alpha": float(alpha[kk]),
                        "delta": float(delta[kk]),
                    }
                )
                k += 1
            for local_slot, rr in enumerate(np.where(mr)[0]):
                rid = int(rr)
                pair = m.rel_pairs[local_slot]
                rows_rel.append(
                    {
                        "mol_row": int(data["dev_idx"][start + i]),
                        "rel_slot": local_slot,
                        "unit_a": int(pair[0]),
                        "unit_b": int(pair[1]),
                        "gamma": float(gamma[rid]),
                    }
                )
    metrics = {
        "additivity_max_abs_error": float(worst),
        "n_dev_mols": int(n),
        "n_unit_rows": len(rows_unit),
        "n_rel_rows": len(rows_rel),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "contributions_dev.npz", pred=np.concatenate(all_pred), y=y_dev)
        with (out_dir / "contributions_units.jsonl").open("w") as f:
            for r in rows_unit:
                f.write(json.dumps(r) + "\n")
        with (out_dir / "contributions_relations.jsonl").open("w") as f:
            for r in rows_rel:
                f.write(json.dumps(r) + "\n")
        (out_dir / "export_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# XGBoost arm (E)
# ---------------------------------------------------------------------------


def xgb_features(ts: list[MolTensors], vocab, mols: list[cf.MolUnits]) -> np.ndarray:
    """Graph-level flat features: type counts + unit-desc aggregates +
    relation aggregates + global composition scalars (label-free)."""
    n_types = vocab.n_total + 1
    feats = np.zeros((len(ts), n_types + 3 * cf.DESC_DIM + cf.REL_DIM + cf.ATOM_CATEGORIES + cf.BOND_CATEGORIES + 4), dtype=np.float32)
    for i, (t, m) in enumerate(zip(ts, mols)):
        for tid in t.type_ids:
            feats[i, tid] += 1.0
        d = t.desc
        base = n_types
        feats[i, base : base + cf.DESC_DIM] = d.mean(axis=0)
        feats[i, base + cf.DESC_DIM : base + 2 * cf.DESC_DIM] = d.sum(axis=0)
        feats[i, base + 2 * cf.DESC_DIM : base + 3 * cf.DESC_DIM] = d.std(axis=0) if len(d) > 1 else 0.0
        b2 = base + 3 * cf.DESC_DIM
        if len(t.rel_feat):
            feats[i, b2 : b2 + cf.REL_DIM] = t.rel_feat.sum(axis=0)
        b3 = b2 + cf.REL_DIM
        atom_hist = d[:, : cf.ATOM_CATEGORIES].sum(axis=0)
        feats[i, b3 : b3 + cf.ATOM_CATEGORIES] = atom_hist
        feats[i, b3 + cf.ATOM_CATEGORIES : b3 + cf.ATOM_CATEGORIES + cf.BOND_CATEGORIES] = (
            d[:, cf.ATOM_CATEGORIES : cf.ATOM_CATEGORIES + cf.BOND_CATEGORIES].sum(axis=0)
        )
        feats[i, -4] = len(t.type_ids)
        feats[i, -3] = len(t.rel_type_lo)
        feats[i, -2] = float(m.n_atoms)
        feats[i, -1] = float(np.log1p(len(t.type_ids)))
    return feats


def train_xgb(seed: int, data: dict, log=print) -> dict:
    import xgboost as xgb

    stats: cf.FitStats = data["stats"]
    t0 = time.time()
    X_in = xgb_features(data["fit_inner"], stats.vocab, data["fit_inner_mols"])
    X_mon = xgb_features(data["monitor"], stats.vocab, data["monitor_mols"])
    X_dev = xgb_features(data["dev"], stats.vocab, data["dev_mols"])
    y_in = data["y"][data["fit_inner_idx"], 0]
    y_mon = data["y"][data["monitor_idx"], 0]
    y_dev = data["y"][data["dev_idx"], 0]
    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=int(seed),
        n_jobs=8,
        early_stopping_rounds=30,
    )
    model.fit(X_in, y_in, eval_set=[(X_mon, y_mon)], verbose=False)
    dev_pred = model.predict(X_dev)
    in_pred = model.predict(X_in)
    out = {
        "arm": "xgb",
        "seed": seed,
        "device": "cpu",
        "soup_dev_mae": float(np.abs(dev_pred - y_dev).mean()),
        "fit_inner_mae": float(np.abs(in_pred - y_in).mean()),
        "best_monitor_mae": float(model.best_score) if hasattr(model, "best_score") else float("nan"),
        "best_epoch": int(getattr(model, "best_iteration", -1)) + 1,
        "epochs_run": int(getattr(model, "best_iteration", -1)) + 1,
        "n_params": -1,
        "wall_seconds": time.time() - t0,
    }
    log(f"[xgb s{seed}] soup dev MAE {out['soup_dev_mae']:.5f} | fit MAE {out['fit_inner_mae']:.5f} | {out['wall_seconds']:.0f}s")
    return out


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def prepare_data(cache: Path | None = None) -> dict:
    """Extract units (with optional disk cache) and build all splits/stats.

    cscl-correctness-v1 note: the cache is version-guarded
    (:func:`cscl_features.load_or_build_units_cache`); the pre-fix
    ``cscl_v0_units.pt`` is rejected instead of silently mixing v0 units with
    the fixed signature code.
    """
    mols, y, smiles = cf.load_or_build_units_cache(cache, cf.extract_all)
    idx = cf.build_split_indices(smiles)

    fit_inner_mols = [mols[i] for i in idx["fit_inner"]]
    stats = cf.FitStats().fit(fit_inner_mols, y)
    data = {
        "idx": idx,
        "stats": stats,
        "y": y,
        "smiles": smiles,
        "fit_inner": build_tensors(fit_inner_mols, stats),
        "monitor": build_tensors([mols[i] for i in idx["monitor"]], stats),
        "dev": build_tensors([mols[i] for i in idx["dev"]], stats),
        "fit_inner_mols": fit_inner_mols,
        "monitor_mols": [mols[i] for i in idx["monitor"]],
        "dev_mols": [mols[i] for i in idx["dev"]],
        "fit_inner_idx": idx["fit_inner"],
        "monitor_idx": idx["monitor"],
        "dev_idx": idx["dev"],
    }
    return data


# ---------------------------------------------------------------------------
# audit (label-blind, v0_protocol §8)
# ---------------------------------------------------------------------------


def run_audit(out_dir: Path, log=print) -> dict:
    mols, y, smiles = cf.extract_all()
    idx = cf.build_split_indices(smiles)
    fit_inner = [mols[i] for i in idx["fit_inner"]]
    dev = [mols[i] for i in idx["dev"]]

    stats = cf.FitStats(min_count=3).fit(fit_inner, y)

    def summarize(name: str, group: list[cf.MolUnits]) -> dict:
        sizes = [len(a) for m in group for a in m.unit_atoms]
        n_units = [len(m.unit_sigs) for m in group]
        n_rel = [len(m.rel_pairs) for m in group]
        sigs = [s for m in group for s in m.unit_sigs]
        tids = {}
        unk = 0
        for m in group:
            for k in range(len(m.unit_sigs)):
                tid = stats.type_id(m, k)
                tids[tid] = tids.get(tid, 0) + 1
                if not stats.vocab.is_known(m.unit_sigs[k]):
                    unk += 1
        counts = sorted(tids.values(), reverse=True)
        return {
            "n_mols": len(group),
            "units_per_mol": {"mean": float(np.mean(n_units)), "p50": float(np.percentile(n_units, 50)), "p95": float(np.percentile(n_units, 95)), "max": int(np.max(n_units))},
            "unit_size": {"mean": float(np.mean(sizes)), "p50": float(np.percentile(sizes, 50)), "p95": float(np.percentile(sizes, 95)), "max": int(np.max(sizes))},
            "unit_total": int(sum(n_units)),
            "kind_ring_vs_chain": [sum(1 for m in group for k in m.unit_kinds if k == KIND_RING), sum(1 for m in group for k in m.unit_kinds if k == KIND_CHAIN)],
            "rels_per_mol": {"mean": float(np.mean(n_rel)), "p50": float(np.percentile(n_rel, 50)), "p95": float(np.percentile(n_rel, 95)), "max": int(np.max(n_rel))},
            "zero_relation_mols_frac": float(np.mean([r == 0 for r in n_rel])),
            "vocab_known_types": stats.vocab.n_known,
            "vocab_total_ids": stats.vocab.n_total,
            "unk_unit_frac": float(unk / max(len(sigs), 1)),
            "top10_type_unit_coverage": float(sum(counts[:10]) / max(len(sigs), 1)),
            "singleton_type_frac_of_units": float(sum(1 for c in counts if c == 1) / max(len(sigs), 1)),
        }

    report = {
        "protocol": "cscl-v0",
        "label_blind": True,
        "official_test_loaded": False,
        "fit_inner": summarize("fit_inner", fit_inner),
        "dev": summarize("dev", dev),
        "smiles_atom_count_crosscheck": crosscheck_atom_counts(mols, smiles),
    }
    fit = report["fit_inner"]
    trigger = {
        "singleton_type_frac_gt_0.30": fit["singleton_type_frac_of_units"] > 0.30,
        "zero_relation_mols_gt_0.50": fit["zero_relation_mols_frac"] > 0.50,
        "unit_size_p95_gt_15": fit["unit_size"]["p95"] > 15,
    }
    report["fallback_trigger_lines"] = trigger
    report["fallback_triggered"] = any(trigger.values())

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report["fit_inner"], indent=2))
    log(f"fallback triggered: {report['fallback_triggered']} ({trigger})")
    return report


def crosscheck_atom_counts(mols: list[cf.MolUnits], smiles: list[str]) -> dict:
    """Cheap alignment check between committed SMILES and the PyG graphs
    (label-blind; approximate atom counting from the SMILES string, including
    lowercase aromatic atoms and bracket atoms)."""
    aromatic = set("bcnops")
    bad = 0
    worst = 0
    for m, s in zip(mols, smiles):
        n = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "[":
                j = s.index("]", i)
                inner = s[i + 1 : j]
                if any(c.isalpha() for c in inner):
                    n += 1
                i = j + 1
                continue
            if ch.isupper() and ch in "BCNOPSFI":
                n += 1
            elif ch in aromatic:
                n += 1
            i += 1
        d = abs(n - m.n_atoms)
        worst = max(worst, d)
        if d > 2:
            bad += 1
    return {"rows_checked": len(mols), "rows_diff_gt2": bad, "max_abs_diff": worst}


# ---------------------------------------------------------------------------
# synthetic mode (v0_protocol §6.3)
# ---------------------------------------------------------------------------


def run_synthetic(arm: str, seed: int, device: str, out_dir: Path, log=print) -> dict:
    data = prepare_data(cache=REPO_ROOT / "data/cache/cscl_v0_units.pt")
    stats: cf.FitStats = data["stats"]
    mols_all = [None] * 10000
    for slot in ("fit_inner_mols", "monitor_mols", "dev_mols"):
        rows = {"fit_inner_mols": "fit_inner_idx", "monitor_mols": "monitor_idx", "dev_mols": "dev_idx"}[slot]
        for r, m in zip(data[rows], data[slot]):
            mols_all[int(r)] = m
    eff = csyn.build_synthetic_effect(data["fit_inner_mols"], mols_all, stats, seed=20261010)

    syn_data = dict(data)
    syn_data["y"] = np.zeros_like(data["y"])
    for slot, rows in (("fit_inner", "fit_inner_idx"), ("monitor", "monitor_idx"), ("dev", "dev_idx")):
        syn_data["y"][data[rows]] = eff["y"][data[rows]][:, None]
    # stats (vocab/centering/y-scaler) remain the label-blind unit stats; y
    # scaler recomputed on synthetic fit_inner
    y_inner = eff["y"][data["fit_inner_idx"]]
    stats.y_mean = float(y_inner.mean())
    stats.y_std = float(y_inner.std() + 1e-12)

    if arm == "xgb":
        res = train_xgb(seed, syn_data, log=log)
    else:
        res = train_torch_arm(arm, seed, device, syn_data, log=log)

    # --- interpretation diagnostics on the trained model -------------------
    # interpretation object = best-epoch checkpoint (soup destroys embedding
    # attribution geometry; see notes/v0_protocol_addendum.md)
    model = None
    if arm != "xgb":
        model = cm.build_model(arm, stats.vocab.n_total, seed)
        model.load_state_dict(res["_best_state"])
        model = model.to(torch.device(device))
        model.eval()
        set_model_centering(model, stats)

        dev_ts = syn_data["dev"]
        gen = torch.Generator().manual_seed(seed + 777)
        dev_batch = assemble(dev_ts)
        with torch.no_grad():
            out = cm.forward_arm(model, dev_batch, arm, gen)
        # per-type alpha (unary path) aggregated over all fit+dev molecules
        alpha_by_type: dict[int, list[float]] = {}
        gamma_by_pair: dict[tuple[int, int], list[float]] = {}
        for slot_ts, slot_mols in ((syn_data["fit_inner"], syn_data["fit_inner_mols"]), (dev_ts, syn_data["dev_mols"])):
            for start in range(0, len(slot_ts), 512):
                chunk_ts = slot_ts[start : start + 512]
                chunk_mols = slot_mols[start : start + 512]
                b = assemble(chunk_ts)
                with torch.no_grad():
                    o = cm.forward_arm(model, b, arm, gen)
                u_off = 0
                r_off = 0
                for mol in chunk_mols:
                    for k, sig in enumerate(mol.unit_sigs):
                        t = stats.vocab.to_id(sig, mol.unit_kinds[k], len(mol.unit_atoms[k]))
                        alpha_by_type.setdefault(t, []).append(float(o.alpha[u_off + k]))
                    u_off += len(mol.unit_sigs)
                    for j, (a, bb) in enumerate(mol.rel_pairs):
                        pair = tuple(sorted((stats.type_id(mol, a), stats.type_id(mol, bb))))
                        gamma_by_pair.setdefault(pair, []).append(float(o.gamma[r_off + j]))
                    r_off += len(mol.rel_pairs)
        theta = eff["theta"]
        types_eval = sorted(set(theta) & set(alpha_by_type))
        th = np.array([theta[t] for t in types_eval])
        al = np.array([float(np.mean(alpha_by_type[t])) for t in types_eval])
        dir_agree = float(np.mean(np.sign(th) == np.sign(al)))
        from scipy.stats import spearmanr

        rho_unary = float(spearmanr(th, al).statistic)

        real_pairs = [tuple(p) for p in eff["theta_pair"]]
        trap_pairs = [tuple(p) for p in eff["trap_pairs"]]
        real_g = [float(np.mean(np.abs(gamma_by_pair[p]))) for p in real_pairs if p in gamma_by_pair]
        trap_g = [float(np.mean(np.abs(gamma_by_pair[p]))) for p in trap_pairs if p in gamma_by_pair]
        diag = {
            "unary_direction_agreement": dir_agree,
            "unary_spearman": rho_unary,
            "real_pair_gamma_mean": float(np.mean(real_g)) if real_g else None,
            "trap_pair_gamma_mean": float(np.mean(trap_g)) if trap_g else None,
            "n_real_pairs_eval": len(real_g),
            "n_trap_pairs_eval": len(trap_g),
        }
        # dev transfer: interactions on unseen combinations at fit time
        fit_pairs_seen = set()
        for m in syn_data["fit_inner_mols"]:
            tset = sorted({stats.type_id(m, k) for k in range(len(m.unit_sigs))})
            for i in range(len(tset)):
                for j in range(i + 1, len(tset)):
                    fit_pairs_seen.add((tset[i], tset[j]))
        nov_g = [
            float(np.mean(np.abs(gamma_by_pair[p])))
            for p in real_pairs
            if p in gamma_by_pair and p not in fit_pairs_seen
        ]
        diag["novel_real_pair_gamma_mean"] = float(np.mean(nov_g)) if nov_g else None
        diag["n_novel_real_pairs"] = len(nov_g)
    else:
        diag = {"note": "xgb: per-type contributions read from gain not implemented in v0"}

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arm": arm, "seed": seed, "synthetic_performance": {k: v for k, v in res.items() if not k.startswith("_")}, "diagnostics": diag}
    (out_dir / f"synthetic_{arm}_s{seed}.json").write_text(json.dumps(payload, indent=2))
    log(json.dumps(diag, indent=2))
    return payload


# ---------------------------------------------------------------------------
# train mode
# ---------------------------------------------------------------------------


def run_train(arm: str, seed: int, device: str, out_dir: Path, log=print, export: bool = True) -> dict:
    data = prepare_data(cache=REPO_ROOT / "data/cache/cscl_v0_units.pt")
    if arm == "xgb":
        res = train_xgb(seed, data, log=log)
    else:
        res = train_torch_arm(arm, seed, device, data, log=log)
    payload = {k: v for k, v in res.items() if not k.startswith("_")}
    payload["official_test_loaded"] = False
    payload["official_valid_loaded"] = False
    payload["supervision"] = "raw y only"
    payload["protocol"] = "cscl-v0"
    payload["python"] = platform.python_version()
    payload["torch"] = torch.__version__
    payload["cuda_device_name"] = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    out_dir.mkdir(parents=True, exist_ok=True)
    if arm in {"additive", "relational", "shuffled"}:
        metrics = export_contributions(
            # export from the best-epoch model (interpretation object)
            _reload_model(arm, seed, res["_best_state"], data),
            arm,
            seed,
            data,
            device,
            out_dir,
            max_mols=None,
        )
        payload["export"] = metrics
    (out_dir / f"train_{arm}_s{seed}.json").write_text(json.dumps(payload, indent=2))
    if "_state" in res:
        torch.save(res["_state"], out_dir / f"state_{arm}_s{seed}.pt")
        torch.save(res["_best_state"], out_dir / f"best_state_{arm}_s{seed}.pt")
    return payload


def _reload_model(arm: str, seed: int, state: dict, data: dict):
    model = cm.build_model(arm, data["stats"].vocab.n_total, seed)
    model.load_state_dict(state)
    set_model_centering(model, data["stats"])
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "train", "synthetic"])
    ap.add_argument("--arm", default="relational", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    print(
        f"[cscl driver] feature code = {cf.FEATURE_VERSION} (signature {cf.SIGNATURE_VERSION}); "
        "the published cscl-v0 REPORT numbers belong to commit 6549c04 (pre-fix code)",
        flush=True,
    )

    if args.mode == "audit":
        run_audit(args.out)
    elif args.mode == "synthetic":
        run_synthetic(args.arm, args.seed, args.device, args.out)
    else:
        run_train(args.arm, args.seed, args.device, args.out, export=not args.no_export)


if __name__ == "__main__":
    main()
