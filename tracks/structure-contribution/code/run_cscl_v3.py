"""cscl-v3 driver: SB-CSCL structured head (S) vs matched opaque head (O).

Protocol: ``configs/strong_backbone_contribution_v3.yaml`` (frozen before any
training).  Both heads read the SAME frozen Full-Y seed-0 features (see
``strong_backbone_attribution.py``); S must produce its prediction as the
strict sum of unit contributions (shared type alpha + context delta), explicit
cross-unit relation contributions, and disclosed per-feature global
context/topology contributions.  O is the capacity-matched opaque control over
the identical information.  seed 0 only; official valid/test never loaded; the
frozen encoder's prediction is never an input or a target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402

PROTOCOL_ID = "cscl-v3-strong-backbone-contribution"
OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3"

EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
TRAIN_SHUFFLE_OFFSET = 101
HIDDEN = 64
TYPE_EMB = 16
PEN_COEF = 1.0
N_HINGES = 3
ZDIM = 2 * sba.ENV_DIM + 2 * sba.PAIR_DIM          # 384
RELIN = 3 * ZDIM + 2 * sba.PAIR_DIM + sba.N_BUCKETS + sba.BOND_CATS + 1 + sba.REL_USED  # 1273


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def mlp(dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# batch collation from frozen features
# ---------------------------------------------------------------------------


class V3Batch:
    """Flat tensors for a set of molecules (unit rows, relation rows, graphs)."""

    def __init__(self, unit_mol, unit_type, unit_feat, unit_ctx, rel_mol, rel_feat, graph_feat):
        self.unit_mol = unit_mol
        self.unit_type = unit_type
        self.unit_feat = unit_feat
        self.unit_ctx = unit_ctx
        self.rel_mol = rel_mol
        self.rel_feat = rel_feat
        self.graph_feat = graph_feat
        self.n_mols = int(graph_feat.shape[0])

    def to(self, device):
        for k in ("unit_mol", "unit_type", "unit_feat", "unit_ctx", "rel_mol", "rel_feat", "graph_feat"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def build_fit_stats(features, fit_rows, type_means, global_mean, rel_mean):
    """Frozen hinge knots (fit quantiles) for the additive global banks."""
    fit = [features[i] for i in _index_by_row(features, fit_rows)]
    ctx = np.stack([m.ctx_structure for m in fit])
    topo = np.stack([m.topology for m in fit])
    qs = [0.25, 0.5, 0.75]
    tau_ctx = np.percentile(ctx, [q * 100 for q in qs], axis=0).T.astype(np.float32)   # [30,3]
    tau_topo = np.percentile(topo, [q * 100 for q in qs], axis=0).T.astype(np.float32)  # [25,3]
    return tau_ctx, tau_topo


def _index_by_row(features: Sequence[sba.MolFrozen], rows: Sequence[int]) -> list[int]:
    lookup = {m.row: i for i, m in enumerate(features)}
    return [lookup[int(r)] for r in rows]


def collate(features: Sequence[sba.MolFrozen], picks: Sequence[int], vocab, type_means, global_mean, rel_mean) -> V3Batch:
    """Pack molecules (by feature-list index) into flat tensors.

    Symmetry: relation feature rows are built from symmetric functions of
    (z_lo, z_hi) with lo<hi by construction, so endpoint swap is identity.
    """
    unit_mol, unit_type, unit_feat, unit_ctx = [], [], [], []
    rel_mol, rel_feat = [], []
    graph_feat = np.zeros((len(picks), sba.CTX_STRUCTURE + sba.TOPO_DIM), dtype=np.float32)
    for mi, fi in enumerate(picks):
        m = features[fi]
        n_u = m.n_units
        cnt_roots = np.asarray([len(a) for a in m.unit_atoms], dtype=np.float32)[:, None]
        cnt_pv = np.maximum(m.unit_n_pv, 0.0)[:, None]
        mean_e = m.unit_sum_e / np.maximum(cnt_roots, 1.0)
        rms_e = np.sqrt(np.maximum(m.unit_sum_e2, 0.0) / np.maximum(cnt_roots, 1.0))
        mean_pv = m.unit_sum_pv / np.maximum(cnt_pv, 1.0)
        rms_pv = np.sqrt(np.maximum(m.unit_sum_pv2, 0.0) / np.maximum(cnt_pv, 1.0))
        z = np.concatenate([mean_e, rms_e, mean_pv, rms_pv], axis=1).astype(np.float32)
        assert z.shape == (n_u, ZDIM)
        tids = m.type_ids(vocab)
        ctxd = np.stack(
            [m.unit_desc[k] - type_means.get(int(tids[k]), global_mean) for k in range(n_u)]
        ).astype(np.float32)
        unit_mol.append(np.full(n_u, mi, dtype=np.int64))
        unit_type.append(tids)
        unit_feat.append(z)
        unit_ctx.append(ctxd)

        n_r = len(m.rel_lo)
        if n_r:
            zlo = z[m.rel_lo]
            zhi = z[m.rel_hi]
            pair_mom = np.concatenate(
                [
                    m.rel_sum_pv / np.maximum(m.rel_count, 1.0)[:, None],
                    np.sqrt(np.maximum(m.rel_sum_pv2, 0.0) / np.maximum(m.rel_count, 1.0)[:, None]),
                ],
                axis=1,
            )
            rows = np.concatenate(
                [
                    zlo + zhi,
                    np.abs(zlo - zhi),
                    zlo * zhi,
                    pair_mom,
                    m.rel_bucket_counts,
                    m.rel_bond_hist,
                    np.log1p(m.rel_count)[:, None],
                    m.rel_mean_relation - rel_mean[None, :],
                ],
                axis=1,
            ).astype(np.float32)
            assert rows.shape == (n_r, RELIN)
            rel_mol.append(np.full(n_r, mi, dtype=np.int64))
            rel_feat.append(rows)
        graph_feat[mi, : sba.CTX_STRUCTURE] = m.ctx_structure
        graph_feat[mi, sba.CTX_STRUCTURE :] = m.topology

    to_cat = lambda parts, dim: (np.concatenate(parts, 0) if parts else np.zeros((0, dim), np.float32))  # noqa: E731
    unit_mol = np.concatenate(unit_mol) if unit_mol else np.zeros(0, np.int64)
    unit_type = np.concatenate(unit_type) if unit_type else np.zeros(0, np.int64)
    unit_feat = to_cat(unit_feat, ZDIM)
    unit_ctx = to_cat(unit_ctx, sba.DESC_DIM)
    rel_mol = np.concatenate(rel_mol) if rel_mol else np.zeros(0, np.int64)
    rel_feat = to_cat(rel_feat, RELIN)
    return V3Batch(
        torch.from_numpy(unit_mol),
        torch.from_numpy(unit_type),
        torch.from_numpy(unit_feat),
        torch.from_numpy(unit_ctx),
        torch.from_numpy(rel_mol),
        torch.from_numpy(rel_feat),
        torch.from_numpy(graph_feat),
    )


# ---------------------------------------------------------------------------
# S: structured contribution head
# ---------------------------------------------------------------------------


class SBContribHead(nn.Module):
    """prediction = bias + sum_u (alpha_type + delta) + sum_r gamma + g_ctx + g_topo."""

    def __init__(self, vocab_size: int, tau_ctx: np.ndarray, tau_topo: np.ndarray):
        super().__init__()
        self.alpha = nn.Embedding(vocab_size, 1)
        self.delta = mlp([ZDIM + sba.DESC_DIM, HIDDEN, HIDDEN, 1])
        self.gamma = mlp([RELIN, HIDDEN, HIDDEN, 1])
        assert tau_ctx.shape == (sba.CTX_STRUCTURE, N_HINGES)
        assert tau_topo.shape == (sba.TOPO_DIM, N_HINGES)
        self.register_buffer("tau_ctx", torch.from_numpy(tau_ctx.copy()))
        self.register_buffer("tau_topo", torch.from_numpy(tau_topo.copy()))
        self.w_ctx = nn.Parameter(torch.zeros(sba.CTX_STRUCTURE, 1 + N_HINGES))
        self.w_topo = nn.Parameter(torch.zeros(sba.TOPO_DIM, 1 + N_HINGES))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.alpha.weight, std=0.01)

    def unit_terms(self, batch: V3Batch):
        alpha = self.alpha(batch.unit_type).squeeze(-1)
        delta = self.delta(torch.cat([batch.unit_feat, batch.unit_ctx], dim=1)).squeeze(-1)
        return alpha, delta

    def relation_terms(self, batch: V3Batch):
        if batch.rel_feat.shape[0] == 0:
            return torch.zeros(0, device=batch.graph_feat.device)
        return self.gamma(batch.rel_feat).squeeze(-1)

    @staticmethod
    def _bank(x: torch.Tensor, tau: torch.Tensor, w: nn.Parameter) -> torch.Tensor:
        """Per-feature additive bank: linear term + hinges at frozen knots.

        x [B,F], tau [F,K], w [F,1+K] -> per-molecule contribution [B] and
        per-feature contributions [B,F] (exported for audits).
        """
        feats = [x]
        for k in range(tau.shape[1]):
            feats.append(torch.relu(x - tau[:, k].unsqueeze(0)))
        H = torch.stack(feats, dim=2)              # [B,F,1+K]
        per_feature = (H * w.unsqueeze(0)).sum(dim=2)  # [B,F]
        return per_feature.sum(dim=1), per_feature

    def forward(self, batch: V3Batch):
        alpha, delta = self.unit_terms(batch)
        gamma = self.relation_terms(batch)
        g_ctx, ctx_pf = self._bank(batch.graph_feat[:, : sba.CTX_STRUCTURE], self.tau_ctx, self.w_ctx)
        g_topo, topo_pf = self._bank(batch.graph_feat[:, sba.CTX_STRUCTURE :], self.tau_topo, self.w_topo)
        unit_sum = torch.zeros(batch.n_mols, device=alpha.device).index_add(0, batch.unit_mol, alpha + delta)
        rel_sum = torch.zeros(batch.n_mols, device=alpha.device).index_add(0, batch.rel_mol, gamma)
        pred = self.bias + unit_sum + rel_sum + g_ctx + g_topo
        return {
            "pred": pred,
            "alpha": alpha,
            "delta": delta,
            "gamma": gamma,
            "g_ctx": g_ctx,
            "g_topo": g_topo,
            "g_ctx_per_feature": ctx_pf,
            "g_topo_per_feature": topo_pf,
            "bias": self.bias.detach(),
        }

    def identifiability_penalty(self, out: Mapping[str, torch.Tensor], unit_type: torch.Tensor | None = None) -> torch.Tensor:
        """Identifiability constraints (research_core sect. 5):

        * per-type mean of delta over the batch must be ~0: the type's average
          contribution belongs to ``alpha``, delta only carries the deviation
          at the unit's actual context (input centering makes the descriptor
          zero at the fit-typical context; this penalty enforces the same
          convention on the learned function);
        * batch mean of gamma must be ~0: the average relation level belongs
          to the bias.

        Variation is never penalised.  Weighted average of squared per-type
        means (weights = type frequencies in the batch).
        """
        pen = out["delta"].new_zeros(())
        delta = out["delta"]
        if delta.numel():
            assert unit_type is not None, "per-type delta centering needs unit_type"
            sums = torch.zeros(int(self.alpha.num_embeddings), device=delta.device, dtype=delta.dtype)
            counts = torch.zeros_like(sums)
            sums.index_add_(0, unit_type, delta)
            counts.index_add_(0, unit_type, torch.ones_like(delta))
            means = sums / torch.clamp(counts, min=1.0)
            present = counts > 0
            weights = counts[present] / counts.sum().clamp(min=1.0)
            pen = pen + (weights * means[present].pow(2)).sum()
        gamma = out["gamma"]
        if gamma.numel():
            pen = pen + gamma.mean().pow(2)
        return pen


# ---------------------------------------------------------------------------
# O: matched opaque head (same frozen information, no additive structure)
# ---------------------------------------------------------------------------


class OpaqueHead(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.type_emb = nn.Embedding(vocab_size, TYPE_EMB)
        self.unit_enc = mlp([ZDIM + sba.DESC_DIM + TYPE_EMB, HIDDEN, HIDDEN, TYPE_EMB])
        self.rel_enc = mlp([RELIN, HIDDEN, HIDDEN, TYPE_EMB])
        self.glob_enc = mlp([sba.CTX_STRUCTURE + sba.TOPO_DIM, TYPE_EMB, TYPE_EMB])
        self.head = mlp([5 * TYPE_EMB, HIDDEN, 1])  # units mean+sum (32) + rel mean+sum (32) + global (16)
        nn.init.normal_(self.type_emb.weight, std=0.1)

    def _pool(self, emb: torch.Tensor, mol: torch.Tensor, n_mols: int) -> torch.Tensor:
        total = torch.zeros(n_mols, emb.shape[1], device=emb.device).index_add(0, mol, emb)
        cnt = torch.zeros(n_mols, device=emb.device).index_add(0, mol, torch.ones_like(mol, dtype=emb.dtype))
        mean = total / torch.clamp(cnt, min=1.0)[:, None]
        return torch.cat([mean, total], dim=1)     # [B, 2*TYPE_EMB]

    def forward(self, batch: V3Batch):
        u_in = torch.cat([batch.unit_feat, batch.unit_ctx, self.type_emb(batch.unit_type)], dim=1)
        u_emb = self.unit_enc(u_in)
        r_emb = self.rel_enc(batch.rel_feat) if batch.rel_feat.shape[0] else u_emb.new_zeros((0, TYPE_EMB))
        g = self.glob_enc(batch.graph_feat)
        pooled = torch.cat(
            [self._pool(u_emb, batch.unit_mol, batch.n_mols), self._pool(r_emb, batch.rel_mol, batch.n_mols), g],
            dim=1,
        )
        return {"pred": self.head(pooled).squeeze(-1)}


# ---------------------------------------------------------------------------
# data preparation (cached)
# ---------------------------------------------------------------------------


def prepare_all(log=print, device="cpu"):
    """Load frozen features (building the cache if absent) + split + stats."""
    blob = sba.load_features(log=log)
    if blob is None:
        log("[features] cache missing; building frozen features from committed artifacts ...")
        import torch

        from run_prediction_feasibility_v2 import apply_prep_fit7200, verify_row_alignment

        model, prep, blob10 = sba.load_frozen_backbone(log=log)
        train = sba.load_train_rows(log=log)
        idx0 = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
        verify_row_alignment(train, log=log)
        apply_prep_fit7200(train, prep, blob10)
        dataset = cf.load_official_train()
        smiles = [str(s) for s in cf.load_canonical_smiles()]
        alignment = sba.check_atom_alignment(train, dataset, None, log=log)
        mols_raw = sba.raw_ringchain(dataset, smiles)
        ys = np.asarray([float(dataset[i].y.reshape(-1)[0]) for i in range(10000)])
        features, audit = sba.extract_all_features(dataset, train, mols_raw, ys, model=model, device=device, log=log)
        sba.save_features(sba.FEAT_CACHE, features, {**audit, **alignment})
        log(f"[features] saved cache {sba.FEAT_CACHE}")
        blob = sba.load_features(log=log)
        assert blob is not None
    features = blob["features"]
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    vocab, type_means, global_mean, rel_mean = sba.fit_unit_vocab(features, idx["fit_inner"])
    tau_ctx, tau_topo = build_fit_stats(features, idx["fit_inner"], type_means, global_mean, rel_mean)
    y = np.asarray([m.y for m in features], dtype=np.float64)
    rows = np.asarray([m.row for m in features], dtype=np.int64)
    assert np.array_equal(np.sort(rows), np.arange(len(rows))) and len(rows) == 10000
    return SimpleNamespace(
        features=features, idx=idx, vocab=vocab, type_means=type_means,
        global_mean=global_mean, rel_mean=rel_mean, tau_ctx=tau_ctx,
        tau_topo=tau_topo, y=y, audit=blob.get("audit", {}),
    )


from types import SimpleNamespace  # noqa: E402


def _batches(n: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    order = torch.randperm(n, generator=generator).tolist() if shuffle else list(range(n))
    return [order[s : s + BATCH_SIZE] for s in range(0, n, BATCH_SIZE)]


def eval_mae(model, batch: V3Batch, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(batch)["pred"].cpu().numpy().astype(np.float64)
    return float(np.abs(pred - y).mean()), pred


# ---------------------------------------------------------------------------
# training (frozen protocol)
# ---------------------------------------------------------------------------


def train_arm(arm: str, ns, device: torch.device, out_dir: Path, seed: int = 0, epochs: int = EPOCHS, log=print):
    assert seed == 0, "this round is seed-0 only"
    assert arm in ("S", "O")
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    fit_batch = collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    mon_batch = collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    y_fit = ns.y[ns.idx["fit_inner"]]
    y_mon = ns.y[ns.idx["monitor"]]
    y_dev = ns.y[ns.idx["dev"]]

    seed_everything(seed)
    if arm == "S":
        model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo).to(device)
    else:
        model = OpaqueHead(ns.vocab.n_total).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)
    target = torch.as_tensor(y_fit, dtype=torch.float32, device=device)

    curve = []
    ckpts: list[tuple[float, int, dict]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
            if arm == "S":
                loss = loss + PEN_COEF * model.identifiability_penalty(out, b.unit_type)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        mon_mae, _ = eval_mae(model, mon_batch, y_mon)
        curve.append({"epoch": epoch, "monitor_mae": mon_mae})
        ckpts.append((mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        ckpts.sort(key=lambda t: (t[0], t[1]))
        if len(ckpts) > 5:
            ckpts.pop()
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            log(f"[{arm} s{seed}] ep={epoch:03d} mon={mon_mae:.5f} {time.perf_counter()-started:.0f}s")
    wall = time.perf_counter() - started

    def evaluate(state):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        f, _ = eval_mae(model, fit_batch, y_fit)
        m, _ = eval_mae(model, mon_batch, y_mon)
        d, dpred = eval_mae(model, dev_batch, y_dev)
        return {"fit_mae": f, "monitor_mae": m, "dev_mae": d}, dpred

    best_state = dict(ckpts[0][2])
    soup_state = {k: torch.stack([s[k].float() for _, _, s in ckpts]).mean(0) for k in ckpts[0][2]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    res_best, dev_pred_best = evaluate(best_state)
    res_soup, dev_pred_soup = evaluate(soup_state)
    res_last, _ = evaluate(last_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"{arm}_s{seed}_best_state.pt")
    np.savez_compressed(
        out_dir / f"{arm}_s{seed}_dev_preds.npz",
        row=np.asarray(ns.idx["dev"], dtype=np.int64),
        pred=dev_pred_best.astype(np.float32),
        pred_soup=dev_pred_soup.astype(np.float32),
        y=y_dev.astype(np.float32),
    )
    result = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": seed,
        "git_commit": git_commit(),
        "n_params": n_params,
        "epochs": epochs,
        "wall_seconds": wall,
        "device": str(device),
        "primary_best_monitor": res_best,
        "secondary_soup": res_soup,
        "tertiary_last": res_last,
        "best_epoch": int(ckpts[0][1]),
        "soup_epochs": sorted(e for _, e, _ in ckpts),
        "curve_tail": curve[-5:],
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "pen_coef": PEN_COEF if arm == "S" else 0.0,
            "hidden": HIDDEN, "optimizer": "Adam", "loss": "L1",
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / f"train_{arm}_s{seed}.json", result)
    log(f"[{arm} s{seed}] primary(dev)={res_best['dev_mae']:.5f} soup={res_soup['dev_mae']:.5f} params={n_params} wall={wall:.0f}s")
    return result


# ---------------------------------------------------------------------------
# contribution export + strict-additivity verification
# ---------------------------------------------------------------------------


@torch.no_grad()
def export_contributions(arm: str, model, ns, picks: Sequence[int], device, path: Path):
    """Per-molecule contribution export; verifies the strict sum identity from
    the exported values alone (numpy, independent of the forward pass)."""
    batch = collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    model.eval()
    if arm == "S":
        out = model(batch)
        pred = out["pred"].cpu().numpy().astype(np.float64)
        alpha = out["alpha"].cpu().numpy().astype(np.float64)
        delta = out["delta"].cpu().numpy().astype(np.float64)
        gamma = out["gamma"].cpu().numpy().astype(np.float64)
        g_ctx = out["g_ctx"].cpu().numpy().astype(np.float64)
        g_topo = out["g_topo"].cpu().numpy().astype(np.float64)
        bias = float(out["bias"].item())
        usum = np.zeros(batch.n_mols)
        np.add.at(usum, batch.unit_mol.cpu().numpy(), alpha + delta)
        rsum = np.zeros(batch.n_mols)
        if len(gamma):
            np.add.at(rsum, batch.rel_mol.cpu().numpy(), gamma)
        recon = bias + usum + rsum + g_ctx + g_topo
        resid = float(np.abs(recon - pred).max())
        per_feature_ctx = out["g_ctx_per_feature"].cpu().numpy().astype(np.float64)
        per_feature_topo = out["g_topo_per_feature"].cpu().numpy().astype(np.float64)
        # per-feature sum identity for the global banks
        gf_resid = float(
            np.abs(per_feature_ctx.sum(1) + per_feature_topo.sum(1) - (g_ctx + g_topo)).max()
        )
        payload = {
            "arm": arm,
            "bias": bias,
            "rows": np.asarray([ns.features[i].row for i in picks], dtype=np.int64),
            "pred": pred,
            "unit_mol": batch.unit_mol.cpu().numpy(),
            "unit_type": batch.unit_type.cpu().numpy(),
            "alpha": alpha,
            "delta": delta,
            "rel_mol": batch.rel_mol.cpu().numpy(),
            "gamma": gamma,
            "g_ctx": g_ctx,
            "g_topo": g_topo,
            "g_ctx_per_feature": per_feature_ctx,
            "g_topo_per_feature": per_feature_topo,
        }
    else:
        pred = model(batch)["pred"].cpu().numpy().astype(np.float64)
        payload = {
            "arm": arm,
            "rows": np.asarray([ns.features[i].row for i in picks], dtype=np.int64),
            "pred": pred,
        }
        resid, gf_resid = 0.0, 0.0
    np.savez_compressed(path, **payload)
    return {"additivity_max_abs_residual": resid, "global_bank_sum_residual": gf_resid}


# ---------------------------------------------------------------------------
# paired bootstrap (SIGNED difference; erratum-consistent statistic)
# ---------------------------------------------------------------------------


def paired_bootstrap(e_a: np.ndarray, e_b: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict:
    diff = e_a - e_b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.empty(resamples)
    for b in range(resamples):
        smp = rng.integers(0, n, n)
        boots[b] = (e_a[smp] - e_b[smp]).mean()      # signed MAE difference
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "resamples": resamples,
        "seed": seed,
        "statistic": "signed (e_a - e_b).mean() per resample",
        "frac_a_better": float((diff < 0).mean()),
    }


def subgroup_table(rows: np.ndarray, y: np.ndarray, arms: Mapping[str, np.ndarray], dataset) -> dict:
    """Subgroup MAEs by y-band / cyclomatic / ring_units / n_atoms."""
    import networkx as nx

    cyc = np.zeros(len(rows))
    n_atoms = np.zeros(len(rows))
    ring_units = np.zeros(len(rows))
    for i, row in enumerate(rows):
        data = dataset[int(row)]
        n_atoms[i] = data.x.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(int(data.x.shape[0])))
        ei = data.edge_index.numpy()
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u != v:
                g.add_edge(u, v)
        cyc[i] = max(ei.shape[1] // 2 - int(data.x.shape[0]) + nx.number_connected_components(g), 0)
        ring_units[i] = sum(1 for _ in nx.cycle_basis(g))
    bands = {
        "y_le_-3": y <= -3,
        "y_-3_to_-1": (y > -3) & (y <= -1),
        "y_-1_to_1": (y > -1) & (y <= 1),
        "y_gt_1": y > 1,
    }
    bands.update({
        "cyc_0_1": cyc < 1,
        "cyc_1_2": (cyc >= 1) & (cyc < 2),
        "cyc_2_3": (cyc >= 2) & (cyc < 3),
        "cyc_ge_3": cyc >= 3,
        "atoms_lt_22": n_atoms < 22,
        "atoms_ge_22": n_atoms >= 22,
    })
    out = {}
    for name, mask in bands.items():
        if mask.sum() == 0:
            continue
        out[name] = {"n": int(mask.sum())}
        for arm, pred in arms.items():
            out[name][f"mae_{arm}"] = float(np.abs(pred[mask] - y[mask]).mean())
    return out


def analyze(ns, out_dir: Path, device, log=print) -> dict:
    """F/O/S comparison, contribution audits (S), subgroup tables."""
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    rows = np.asarray(ns.idx["dev"], dtype=np.int64)
    y_dev = ns.y[rows]
    arms: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}

    # F: frozen Full-Y committed dev predictions
    saved = np.load(sba.DEV_PREDS_NPZ)
    assert np.array_equal(saved["row"], rows)
    arms["F"] = saved["pred"].astype(np.float64)

    for arm in ("O", "S"):
        state = torch.load(out_dir / f"{arm}_s0_best_state.pt", map_location="cpu", weights_only=True)
        if arm == "S":
            model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        else:
            model = OpaqueHead(ns.vocab.n_total)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
        with torch.no_grad():
            arms[arm] = model(batch)["pred"].cpu().numpy().astype(np.float64)
        info[f"{arm}_additivity"] = export_contributions(
            arm, model, ns, dev_pos, device, out_dir / f"{arm}_s0_dev_contribs.npz"
        )

    summary: dict[str, Any] = {
        "mae": {a: float(np.abs(p - y_dev).mean()) for a, p in arms.items()},
        "additivity": info,
    }
    for a, b in (("S", "O"), ("S", "F"), ("O", "F")):
        summary[f"paired_{a}_minus_{b}"] = paired_bootstrap(
            np.abs(arms[a] - y_dev), np.abs(arms[b] - y_dev)
        )
    dataset = cf.load_official_train()
    summary["subgroups"] = subgroup_table(rows, y_dev, arms, dataset)

    # ---- S contribution audit ----
    z = np.load(out_dir / "S_s0_dev_contribs.npz")
    alpha, delta, gamma = z["alpha"], z["delta"], z["gamma"]
    unit_type, unit_mol = z["unit_type"], z["unit_mol"]
    alpha_table = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    fit_counts = {}
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    for fi in fit_pos:
        for t in ns.features[fi].type_ids(ns.vocab):
            fit_counts[int(t)] = fit_counts.get(int(t), 0) + 1
    freq = sorted(fit_counts, key=lambda t: -fit_counts[t])[:20]
    summary["S_audit"] = {
        "bias": float(z["bias"]),
        "alpha_top20_frequent_types": {
            str(t): {"fit_count": fit_counts[t], "alpha": float(alpha_table[t])} for t in freq
        },
        "alpha_std_across_known": float(np.std([alpha_table[t] for t in fit_counts])),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_per_type_top20": {
            str(t): {
                "mean": float(delta[unit_type == t].mean()) if (unit_type == t).any() else 0.0,
                "std": float(delta[unit_type == t].std()) if (unit_type == t).sum() > 1 else 0.0,
                "n": int((unit_type == t).sum()),
            }
            for t in freq
        },
        "gamma_mean": float(gamma.mean()) if len(gamma) else 0.0,
        "gamma_std": float(gamma.std()) if len(gamma) else 0.0,
        "gamma_abs_le_0.01_frac": float((np.abs(gamma) <= 0.01).mean()) if len(gamma) else 1.0,
        "g_ctx_mean_abs": float(np.abs(z["g_ctx"]).mean()),
        "g_topo_mean_abs": float(np.abs(z["g_topo"]).mean()),
    }
    # channel variance shares over dev molecules (molecule-level sums)
    chans = {
        "unit": np.zeros(len(rows)), "rel": np.zeros(len(rows)),
        "g_ctx": z["g_ctx"], "g_topo": z["g_topo"],
    }
    np.add.at(chans["unit"], unit_mol, alpha + delta)
    if len(gamma):
        np.add.at(chans["rel"], z["rel_mol"], gamma)
    shares = {}
    pred = z["pred"]
    for name, v in chans.items():
        shares[name] = {
            "var_share": float(np.var(v) / max(np.var(pred), 1e-12)),
            "mean_abs": float(np.abs(v).mean()),
            "frac_pos": float((v > 0).mean()),
        }
    summary["S_channel_shares"] = shares
    # per-feature global contributions (context 30 / topology 25), fit-name free
    c_pf, t_pf = z["g_ctx_per_feature"], z["g_topo_per_feature"]
    summary["S_global_per_feature"] = {
        "ctx_feature_idx_by_abs_weight": np.argsort(-np.abs(c_pf).mean(0)).tolist(),
        "ctx_mean_abs_top5": np.sort(np.abs(c_pf).mean(0))[::-1][:5].tolist(),
        "topo_feature_idx_by_abs_weight": np.argsort(-np.abs(t_pf).mean(0)).tolist(),
        "topo_mean_abs_top5": np.sort(np.abs(t_pf).mean(0))[::-1][:5].tolist(),
        "ctx_total_mean_abs": float(np.abs(c_pf).mean()),
        "topo_total_mean_abs": float(np.abs(t_pf).mean()),
    }
    write_json(out_dir / "analysis_s0.json", summary)
    log("[analyze] " + json.dumps(summary["mae"]))
    return summary


# ---------------------------------------------------------------------------
# synthetic known-truth task (CPU, seed 0) — mechanism evidence, not ZINC claims
# ---------------------------------------------------------------------------


def _hash_value(payload: str) -> float:
    h = hashlib.sha256(payload.encode()).hexdigest()
    return (int(h[:8], 16) % 2000) / 1000.0 - 1.0


def synthetic_task(ns, out_dir: Path, device, epochs: int = 240, log=print) -> dict:
    """y_syn = sum_u [alpha*(t) + delta*(u)] + sum_pairs gamma*(t1,t2) + noise.

    Known truth on REAL frozen features: per-type main effects (mixed signs),
    a context effect driven by the type-centered attach count, three true
    interaction pairs, a co-occurring confuser type with NO interaction, and
    seed-0 noise.  Trains a fresh SBContribHead (seed 0, same hyper).
    """
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])

    fit_type_counts: dict[int, int] = {}
    for fi in fit_pos:
        for t in ns.features[fi].type_ids(ns.vocab):
            fit_type_counts[int(t)] = fit_type_counts.get(int(t), 0) + 1
    freq_types = sorted(fit_type_counts, key=lambda t: -fit_type_counts[t])
    top32 = freq_types[:32]
    alpha_star = {t: _hash_value(f"alpha|{t}") for t in top32}

    # fit relation counts by type pair
    rel_type_count: dict[tuple[int, int], int] = {}
    for fi in fit_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for lo, hi in zip(m.rel_lo, m.rel_hi):
            key = (int(tids[lo]), int(tids[hi]))
            rel_type_count[key] = rel_type_count.get(key, 0) + 1
    cand = [k for k, v in sorted(rel_type_count.items(), key=lambda item: -item[1]) if v >= 50]
    if len(cand) < 4:
        cand = [k for k, _ in sorted(rel_type_count.items(), key=lambda item: -item[1])][:4]
    pair_A, pair_B, pair_C = cand[0], cand[1], cand[2]
    gamma_star = {pair_A: 0.9, pair_B: -0.7, pair_C: 0.5}
    # confuser: frequent type pair with no true interaction, co-occurring with A
    type_of_A = {pair_A[0], pair_A[1]}
    confusers = [k for k in cand[3:] if k[0] in type_of_A or k[1] in type_of_A]
    pair_D = confusers[0] if confusers else cand[3]

    # fit-only attach-count centering per type (matches the head's context input)
    attach_sum, attach_n = {}, {}
    for fi in fit_pos:
        m = ns.features[fi]
        for k, t in enumerate(m.type_ids(ns.vocab)):
            attach_sum[int(t)] = attach_sum.get(int(t), 0.0) + float(m.unit_desc[k, 4])
            attach_n[int(t)] = attach_n.get(int(t), 0) + 1
    attach_mean = {t: attach_sum[t] / attach_n[t] for t in attach_n}
    global_attach = float(np.mean([float(ns.features[fi].unit_desc[:, 4].mean()) for fi in fit_pos]))

    def syn_labels(picks):
        ys, meta = [], []
        for fi in picks:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            total = 0.0
            for k, t in enumerate(tids):
                total += alpha_star.get(int(t), 0.0)
                z = (float(m.unit_desc[k, 4]) - attach_mean.get(int(t), global_attach))
                total += 0.8 * z
            for lo, hi in zip(m.rel_lo, m.rel_hi):
                key = (int(tids[lo]), int(tids[hi]))
                total += gamma_star.get(key, 0.0)
            ys.append(total)
        return np.asarray(ys, dtype=np.float64)

    rng = np.random.default_rng(0)
    y_syn_fit = syn_labels(fit_pos) + 0.05 * rng.standard_normal(len(fit_pos))
    y_syn_mon = syn_labels(mon_pos)
    y_syn_dev = syn_labels(dev_pos)

    seed_everything(0)
    model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(101)
    fit_batch = collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    mon_batch = collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    target = torch.as_tensor(y_syn_fit, dtype=torch.float32, device=device)
    best = (1e9, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
            loss = loss + PEN_COEF * model.identifiability_penalty(out, b.unit_type)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            mon_pred = model(mon_batch)["pred"].cpu().numpy().astype(np.float64)
        mon_mae = float(np.abs(mon_pred - y_syn_mon).mean())
        if mon_mae < best[0]:
            best = (mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict({k: v.to(device) for k, v in best[2].items()})
    model.eval()

    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    with torch.no_grad():
        out = model(dev_batch)
    alpha_hat = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    delta_hat = out["delta"].cpu().numpy().astype(np.float64)
    gamma_hat = out["gamma"].cpu().numpy().astype(np.float64)

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    # alpha recovery on the 32 known types
    a_true = np.array([alpha_star[t] for t in top32])
    a_hat = np.array([alpha_hat[t] for t in top32])
    sign_match = float((np.sign(a_true) == np.sign(a_hat)).mean())
    # delta recovery: true delta is 0.8 * centered attach count
    # delta recovery: true delta is 0.8 * type-centered attach count; align
    # hat rows to (molecule, unit) via the flat unit_mol map
    true_delta, hat_delta, alpha_of_unit = [], [], []
    um = dev_batch.unit_mol.cpu().numpy()
    for mi, fi in enumerate(dev_pos):
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        hats = delta_hat[um == mi]
        assert len(hats) == m.n_units
        for k, t in enumerate(tids):
            true_delta.append(0.8 * (float(m.unit_desc[k, 4]) - attach_mean.get(int(t), global_attach)))
            hat_delta.append(float(hats[k]))
            alpha_of_unit.append(alpha_star.get(int(t), 0.0))
    true_delta = np.asarray(true_delta)
    hat_delta = np.asarray(hat_delta)
    alpha_of_unit = np.asarray(alpha_of_unit)

    # gamma recovery per selected pair
    # gamma recovery per selected pair (collate order: relations are appended
    # per molecule in rel_lo/rel_hi order)
    def gamma_mean_for(pair_key):
        vals = []
        idx_rel = 0
        for fi in dev_pos:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            for j in range(len(m.rel_lo)):
                key = (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]]))
                if key == pair_key:
                    vals.append(float(gamma_hat[idx_rel + j]))
            idx_rel += len(m.rel_lo)
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    gamma_check = {}
    for name, key, star in (("pair_A", pair_A, 0.9), ("pair_B", pair_B, -0.7), ("pair_C", pair_C, 0.5), ("pair_D_confuser", pair_D, 0.0)):
        mean_hat, n_obs = gamma_mean_for(key)
        gamma_check[name] = {
            "types": [int(key[0]), int(key[1])], "gamma_star": star,
            "gamma_hat_mean": mean_hat, "sign_recovered": bool(np.sign(mean_hat) == np.sign(star)),
            "n_dev_relations": n_obs,
        }
    # unseen combinations: type pairs absent in fit, present in dev
    unseen_hat, seen_hat = [], []
    idx_rel = 0
    for fi in dev_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for j in range(len(m.rel_lo)):
            key = (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]]))
            (unseen_hat if key not in rel_type_count else seen_hat).append(float(gamma_hat[idx_rel + j]))
        idx_rel += len(m.rel_lo)

    result = {
        "protocol": PROTOCOL_ID,
        "task": "synthetic known-truth on real frozen features (unit/context/interaction + noise 0.05, seed 0)",
        "best_monitor_epoch": int(best[1]),
        "monitor_mae_syn": float(best[0]),
        "dev_mae_syn": float(np.abs(out["pred"].cpu().numpy().astype(np.float64) - y_syn_dev).mean()),
        "alpha_sign_recovery_top32": sign_match,
        "alpha_spearman_top32": _spearman(a_true, a_hat),
        "alpha_true_value_range": [float(a_true.min()), float(a_true.max())],
        "delta_recovery_pearson": float(np.corrcoef(true_delta, hat_delta)[0, 1]),
        "delta_hat_vs_alpha_star_confusion": float(np.corrcoef(hat_delta, alpha_of_unit)[0, 1]),
        "gamma_checks": gamma_check,
        "unseen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(unseen_hat))) if unseen_hat else None,
        "seen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(seen_hat))) if seen_hat else None,
        "n_unseen_dev_relations": len(unseen_hat),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "official_valid_loaded": False,
    }
    write_json(out_dir / "synthetic_s0.json", result)
    log("[synthetic] alpha sign %.2f spearman %.2f delta-r %.2f" % (sign_match, result["alpha_spearman_top32"], result["delta_recovery_pearson"]))
    return result


# ---------------------------------------------------------------------------
# Phase-A checks + CLI
# ---------------------------------------------------------------------------


def run_checks(out_dir: Path, device="cpu", log=print) -> dict:
    """Phase A end-to-end: alignment, extraction audit (fresh, first 640 rows),
    frozen-soup replay, cache consistency."""
    model, prep, blob10 = sba.load_frozen_backbone(log=log)
    train = sba.load_train_rows(log=log)
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    from run_prediction_feasibility_v2 import apply_prep_fit7200

    apply_prep_fit7200(train, prep, blob10)
    dataset = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    alignment = sba.check_atom_alignment(train, dataset, None, log=log)
    mols_raw = sba.raw_ringchain(dataset, smiles)
    ys = np.array([float(dataset[i].y.reshape(-1)[0]) for i in range(10000)])
    _feats, audit = sba.extract_all_features(dataset, train, mols_raw, ys, model=model, rows=list(range(640)), log=log)
    replay = sba.replay_predictions(model, train, idx, device=device, log=log)
    out = {"alignment": alignment, "extraction_audit_640": audit, "replay": replay, "ok": True}
    write_json(out_dir / "phase_a_checks.json", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("checks", "train", "analyze", "synthetic", "all"))
    parser.add_argument("--arm", choices=("S", "O"), default=None)
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda *a, **k: print(*a, **k, flush=True)  # noqa: E731

    if args.mode == "checks":
        run_checks(args.out, device="cpu", log=log)
        return 0
    ns = prepare_all(log=log)
    if args.mode == "train":
        assert args.arm is not None, "--arm required"
        train_arm(args.arm, ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        return 0
    if args.mode == "analyze":
        analyze(ns, args.out, device, log=log)
        return 0
    if args.mode == "synthetic":
        synthetic_task(ns, args.out, torch.device("cpu"), log=log)
        return 0
    if args.mode == "all":
        log("[all] Phase-A checks on the run device ...")
        run_checks(args.out, device=str(device), log=log)
        ns = prepare_all(log=log, device=str(device))
        log("[all] training S (seed 0) ...")
        train_arm("S", ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        log("[all] training O (seed 0, sequential) ...")
        train_arm("O", ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        analyze(ns, args.out, device, log=log)
        synthetic_task(ns, args.out, torch.device("cpu"), log=log)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
