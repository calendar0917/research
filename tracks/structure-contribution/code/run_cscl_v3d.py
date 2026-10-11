"""cscl-v3d driver: bucket-complete S/O arms (seed 0), same enhanced inputs.

Protocol: ``configs/cscl_v3d_bucket.yaml`` (frozen before any formal training).

Research question: v3/v3b/v3c inputs keep only bucket-MARGINAL pair moments
while the frozen Full-Y reader consumes per-distance-bucket first/second
moments.  This round restores the complete distance-conditioned statistics
(``strong_backbone_bucket_features``) and feeds them through the SAME additive
unit/relation channels to BOTH arms:

  S_bucket: pred = bias + sum_u(alpha_t + delta_u) + sum_r gamma_r + g_ctx + g_topo
            delta input [z 384 | ctx 40 | type_emb 16 | unit bucket 485] = 925
            gamma input [v3 rel row 1273 | rel bucket 480]               = 1753
  O_bucket: identical input extension on the v3 opaque DeepSets head
            (unit_enc 440 -> 925, rel_enc 1273 -> 1753)

Everything else is held at the historical configuration: per-type scalar alpha,
16-D type embedding (delta only, S arm), fit-only prediction-invariant gauge
(S arm), v3b/v3c objective (S: L1 + 1.0*gamma batch-mean penalty; O: L1),
Adam 1e-3 / wd 1e-5 / clip 5 / batch 128 / 240 epochs, shuffle generator 101
(same batch sequence as v3/v3b/v3c), best-monitor single-checkpoint primary +
pre-registered top-5 monitor soup secondary.  Intra-unit bucket info enters
ONLY unit contributions; cross-unit bucket info ONLY the corresponding unit-pair
contributions (AST-pinned).  seed 0 only; official valid/test never loaded; the
frozen Full-Y prediction is never an input or a supervision target.

Initialization discipline: the untrained v3c TypeCondContribHead (resp. v3
OpaqueHead) is built under seed 0 and every shape-unchanged parameter is copied
bit-exact; the new input columns (485 unit-side / 480 relation-side for BOTH
arms) are ZERO-initialized, so both arms get the SAME +61,760 parameters and
the untrained predictions equal the reference heads exactly.  The new heads
train from scratch (historical trained checkpoints are never loaded).  dev is
NEVER evaluated inside training; both arms are trained sequentially first, and
dev metrics are computed only afterwards in ``analyze``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import contribution_gauge as cg  # noqa: E402
import cscl_features as cf  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
import strong_backbone_bucket_features as bf  # noqa: E402
from run_cscl_v3 import (  # noqa: E402  (reuse: collate/split/stats/utils/heads)
    BATCH_SIZE,
    EPOCHS,
    GRAD_CLIP,
    HIDDEN,
    LR,
    PEN_COEF,
    RELIN,
    TRAIN_SHUFFLE_OFFSET,
    TYPE_EMB,
    V3Batch,
    WEIGHT_DECAY,
    ZDIM,
    _batches,
    _index_by_row,
    collate,
    git_commit,
    mlp,
    paired_bootstrap,
    seed_everything,
    write_json,
)
from run_cscl_v3b import (  # noqa: E402
    attach_counts,
    gamma_identifiability_penalty,
    subgroup_table_v3b,
)

PROTOCOL_ID = "cscl-v3d-distance-conditioned-contribution"
OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3d"
V3C_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3c"
V3B_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b"
V3_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3"

#: appended input-block widths (protocol-frozen representation)
UNIT_BUCKET_DIM = bf.UNIT_BUCKET_DIM   # 485 = 5 * (1 + 48 + 48)
REL_BUCKET_DIM = bf.REL_BUCKET_DIM     # 480 = 5 * (48 + 48)

#: head input widths
S_DELTA_IN = ZDIM + sba.DESC_DIM + TYPE_EMB + UNIT_BUCKET_DIM   # 925
S_GAMMA_IN = RELIN + REL_BUCKET_DIM                             # 1753
O_UNIT_IN = ZDIM + sba.DESC_DIM + TYPE_EMB + UNIT_BUCKET_DIM    # 925
O_REL_IN = RELIN + REL_BUCKET_DIM                               # 1753

JUDGMENT_BANDS = {
    "case_A": "S and O both improve AND S_bucket-O_bucket <= 0.02 AND S_bucket-F <= 0.02 with intact mechanism -> structured readout approaches the strong backbone; next step INDEPENDENT VALIDATION (new authorization, not this round)",
    "case_B": "O_bucket approaches F but S_bucket-O_bucket > 0.02 -> additive exit underuses the restored information; reflect on expression limits; do NOT grow encoders",
    "case_C": "neither arm improves materially -> bucket compression is NOT the main O-F explanation; study the Full-Y reader's nonlinear mixing next; stop claiming the compression story",
    "case_D": "S_bucket improves but O_bucket improves more -> bucket information has predictive value but does not close the structured-vs-opaque gap; no success claim from an S record alone",
    "case_E": "prediction ok but mechanism checks fail -> predictive improvement only, no interpretability claim",
    "disclosure": "arms keep their historical loss difference (S gamma penalty); F checkpoint rule differs (soup); both vs-historical comparisons carry the SAME +61,760 params of capacity confound",
}

RECORDED_MAE = {"F": 0.11641, "O": 0.14129, "S_v3": 0.19551, "S_gauge": 0.17317, "S_typecond": 0.15209}


# ---------------------------------------------------------------------------
# batch: V3Batch + appended bucket blocks (SAME tensors for both arms)
# ---------------------------------------------------------------------------


class BucketBatch(V3Batch):
    """V3Batch + ``unit_bucket`` [U,485] + ``rel_bucket`` [R,480]."""

    def __init__(self, unit_mol, unit_type, unit_feat, unit_ctx, rel_mol, rel_feat, graph_feat,
                 unit_bucket, rel_bucket):
        super().__init__(unit_mol, unit_type, unit_feat, unit_ctx, rel_mol, rel_feat, graph_feat)
        self.unit_bucket = unit_bucket
        self.rel_bucket = rel_bucket

    def to(self, device):
        super().to(device)
        for k in ("unit_bucket", "rel_bucket"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def collate_bucket(features: Sequence[Any], picks: Sequence[int], vocab, type_means, global_mean, rel_mean) -> BucketBatch:
    """Pack molecules into a BucketBatch.

    The legacy columns are produced by the UNCHANGED v3 collate (bit-identical
    semantics); the bucket blocks are appended deterministically (bucket-major:
    per bucket [count, mean48, rms48] for units; per bucket [mean48, rms48] for
    relations; zero-count buckets -> all zeros).  S and O receive the SAME
    tensors from this single function (test-pinned).  The audit-only
    ``pair_ref_*`` arrays of the cache are never read here (test-pinned).
    """
    base = collate(features, picks, vocab, type_means, global_mean, rel_mean)
    unit_parts: list[np.ndarray] = []
    rel_parts: list[np.ndarray] = []
    for mi, fi in enumerate(picks):
        m = features[fi]
        cnt = np.maximum(m.unit_bucket_counts, 0.0).astype(np.float32)[:, :, None]      # [n_u,5,1]
        mean = m.unit_bucket_sum_pv / np.maximum(cnt, 1.0)                              # [n_u,5,48]
        rms = np.sqrt(np.maximum(m.unit_bucket_sum_pv2, 0.0) / np.maximum(cnt, 1.0))    # [n_u,5,48]
        unit_parts.append(np.concatenate([cnt, mean, rms], axis=2).reshape(m.n_units, UNIT_BUCKET_DIM).astype(np.float32))
        n_r = len(m.rel_lo)
        if n_r:
            rc = np.maximum(m.rel_bucket_counts, 0.0).astype(np.float32)[:, :, None]
            rmean = m.rel_bucket_sum_pv / np.maximum(rc, 1.0)
            rrms = np.sqrt(np.maximum(m.rel_bucket_sum_pv2, 0.0) / np.maximum(rc, 1.0))
            rel_parts.append(np.concatenate([rmean, rrms], axis=2).reshape(n_r, REL_BUCKET_DIM).astype(np.float32))
    cat = lambda parts, dim: (np.concatenate(parts, 0) if parts else np.zeros((0, dim), np.float32))  # noqa: E731
    unit_bucket = cat(unit_parts, UNIT_BUCKET_DIM)
    rel_bucket = cat(rel_parts, REL_BUCKET_DIM)
    assert unit_bucket.shape == (base.unit_feat.shape[0], UNIT_BUCKET_DIM)
    assert rel_bucket.shape == (base.rel_feat.shape[0], REL_BUCKET_DIM)
    return BucketBatch(
        base.unit_mol, base.unit_type, base.unit_feat, base.unit_ctx,
        base.rel_mol, base.rel_feat, base.graph_feat,
        torch.from_numpy(unit_bucket), torch.from_numpy(rel_bucket),
    )


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------


class SBucketContribHead(nn.Module):
    """v3c TypeCondContribHead + appended bucket input columns.

    prediction = bias + sum_u(alpha_t + delta_u) + sum_r gamma_r + g_ctx + g_topo.
    delta reads [z | ctx | type_emb | unit_bucket]; gamma reads [rel_row | rel_bucket].
    Intra-unit bucket info reaches the prediction ONLY through delta; cross-unit
    bucket info ONLY through the corresponding gamma (AST-pinned by tests).
    Module construction order is identical to TypeCondContribHead so the seed-0
    bit-exact copy is reproducible.
    """

    def __init__(self, vocab_size: int, tau_ctx: np.ndarray, tau_topo: np.ndarray):
        super().__init__()
        self.alpha = nn.Embedding(vocab_size, 1)
        self.delta = mlp([S_DELTA_IN, HIDDEN, HIDDEN, 1])
        self.gamma = mlp([S_GAMMA_IN, HIDDEN, HIDDEN, 1])
        assert tau_ctx.shape == (sba.CTX_STRUCTURE, 3) and tau_topo.shape == (sba.TOPO_DIM, 3)
        self.register_buffer("tau_ctx", torch.from_numpy(tau_ctx.copy()))
        self.register_buffer("tau_topo", torch.from_numpy(tau_topo.copy()))
        self.w_ctx = nn.Parameter(torch.zeros(sba.CTX_STRUCTURE, 1 + 3))
        self.w_topo = nn.Parameter(torch.zeros(sba.TOPO_DIM, 1 + 3))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.alpha.weight, std=0.01)
        self.type_emb = nn.Embedding(vocab_size, TYPE_EMB)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.type_emb.weight.copy_(torch.randn(vocab_size, TYPE_EMB, generator=gen) * 0.1)

    def unit_terms(self, batch: BucketBatch):
        alpha = self.alpha(batch.unit_type).squeeze(-1)
        z = torch.cat([batch.unit_feat, batch.unit_ctx, self.type_emb(batch.unit_type), batch.unit_bucket], dim=1)
        delta = self.delta(z).squeeze(-1)
        return alpha, delta

    def relation_terms(self, batch: BucketBatch):
        if batch.rel_feat.shape[0] == 0:
            return torch.zeros(0, device=batch.graph_feat.device)
        return self.gamma(torch.cat([batch.rel_feat, batch.rel_bucket], dim=1)).squeeze(-1)

    @staticmethod
    def _bank(x: torch.Tensor, tau: torch.Tensor, w: nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [x]
        for k in range(tau.shape[1]):
            feats.append(torch.relu(x - tau[:, k].unsqueeze(0)))
        H = torch.stack(feats, dim=2)
        per_feature = (H * w.unsqueeze(0)).sum(dim=2)
        return per_feature.sum(dim=1), per_feature

    def forward(self, batch: BucketBatch):
        alpha, delta = self.unit_terms(batch)
        gamma = self.relation_terms(batch)
        g_ctx, ctx_pf = self._bank(batch.graph_feat[:, : sba.CTX_STRUCTURE], self.tau_ctx, self.w_ctx)
        g_topo, topo_pf = self._bank(batch.graph_feat[:, sba.CTX_STRUCTURE :], self.tau_topo, self.w_topo)
        unit_sum = torch.zeros(batch.n_mols, device=alpha.device).index_add(0, batch.unit_mol, alpha + delta)
        rel_sum = torch.zeros(batch.n_mols, device=alpha.device).index_add(0, batch.rel_mol, gamma)
        pred = self.bias + unit_sum + rel_sum + g_ctx + g_topo
        return {
            "pred": pred, "alpha": alpha, "delta": delta, "gamma": gamma,
            "g_ctx": g_ctx, "g_topo": g_topo,
            "g_ctx_per_feature": ctx_pf, "g_topo_per_feature": topo_pf,
            "bias": self.bias.detach(),
        }


class OBucketHead(nn.Module):
    """v3 OpaqueHead + appended bucket input columns (no other change)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.type_emb = nn.Embedding(vocab_size, TYPE_EMB)
        self.unit_enc = mlp([O_UNIT_IN, HIDDEN, HIDDEN, TYPE_EMB])
        self.rel_enc = mlp([O_REL_IN, HIDDEN, HIDDEN, TYPE_EMB])
        self.glob_enc = mlp([sba.CTX_STRUCTURE + sba.TOPO_DIM, TYPE_EMB, TYPE_EMB])
        self.head = mlp([5 * TYPE_EMB, HIDDEN, 1])
        nn.init.normal_(self.type_emb.weight, std=0.1)

    def _pool(self, emb: torch.Tensor, mol: torch.Tensor, n_mols: int) -> torch.Tensor:
        total = torch.zeros(n_mols, emb.shape[1], device=emb.device).index_add(0, mol, emb)
        cnt = torch.zeros(n_mols, device=emb.device).index_add(0, mol, torch.ones_like(mol, dtype=emb.dtype))
        mean = total / torch.clamp(cnt, min=1.0)[:, None]
        return torch.cat([mean, total], dim=1)

    def forward(self, batch: BucketBatch):
        u_in = torch.cat([batch.unit_feat, batch.unit_ctx, self.type_emb(batch.unit_type), batch.unit_bucket], dim=1)
        u_emb = self.unit_enc(u_in)
        r_emb = (
            self.rel_enc(torch.cat([batch.rel_feat, batch.rel_bucket], dim=1))
            if batch.rel_feat.shape[0]
            else u_emb.new_zeros((0, TYPE_EMB))
        )
        g = self.glob_enc(batch.graph_feat)
        pooled = torch.cat(
            [self._pool(u_emb, batch.unit_mol, batch.n_mols), self._pool(r_emb, batch.rel_mol, batch.n_mols), g],
            dim=1,
        )
        return {"pred": self.head(pooled).squeeze(-1)}


# ---------------------------------------------------------------------------
# initialization discipline (deterministic seed-0, zero-filled new columns)
# ---------------------------------------------------------------------------


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def build_head_pair(arm: str, ns, log=print):
    """(ref_untrained_historical_head, new_head, report) for arm in {S, O}."""
    assert arm in ("S", "O")
    if arm == "S":
        from run_cscl_v3c import TypeCondContribHead

        seed_everything(0)
        ref = TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        seed_everything(0)
        model = SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        old_din = ZDIM + sba.DESC_DIM + TYPE_EMB       # 440
        old_gin = RELIN                                # 1273
    else:
        from run_cscl_v3 import OpaqueHead

        seed_everything(0)
        ref = OpaqueHead(ns.vocab.n_total)
        seed_everything(0)
        model = OBucketHead(ns.vocab.n_total)
        old_din = ZDIM + sba.DESC_DIM + TYPE_EMB       # 440 (unit_enc)
        old_gin = RELIN                                # 1273 (rel_enc)

    with torch.no_grad():
        if arm == "S":
            model.alpha.weight.copy_(ref.alpha.weight)
            model.delta[0].weight[:, :old_din].copy_(ref.delta[0].weight)
            model.delta[0].weight[:, old_din:].zero_()
            model.delta[0].bias.copy_(ref.delta[0].bias)
            for li in (2, 4):
                model.delta[li].weight.copy_(ref.delta[li].weight)
                model.delta[li].bias.copy_(ref.delta[li].bias)
            model.gamma[0].weight[:, :old_gin].copy_(ref.gamma[0].weight)
            model.gamma[0].weight[:, old_gin:].zero_()
            model.gamma[0].bias.copy_(ref.gamma[0].bias)
            for gi in (2, 4):
                model.gamma[gi].weight.copy_(ref.gamma[gi].weight)
                model.gamma[gi].bias.copy_(ref.gamma[gi].bias)
            model.w_ctx.copy_(ref.w_ctx)
            model.w_topo.copy_(ref.w_topo)
            model.bias.copy_(ref.bias)
            model.tau_ctx.copy_(ref.tau_ctx)
            model.tau_topo.copy_(ref.tau_topo)
            assert torch.equal(model.type_emb.weight, ref.type_emb.weight), "type emb stream drifted"
        else:
            model.type_emb.weight.copy_(ref.type_emb.weight)
            model.unit_enc[0].weight[:, :old_din].copy_(ref.unit_enc[0].weight)
            model.unit_enc[0].weight[:, old_din:].zero_()
            model.unit_enc[0].bias.copy_(ref.unit_enc[0].bias)
            for li in (2, 4):
                model.unit_enc[li].weight.copy_(ref.unit_enc[li].weight)
                model.unit_enc[li].bias.copy_(ref.unit_enc[li].bias)
            model.rel_enc[0].weight[:, :old_gin].copy_(ref.rel_enc[0].weight)
            model.rel_enc[0].weight[:, old_gin:].zero_()
            model.rel_enc[0].bias.copy_(ref.rel_enc[0].bias)
            for li in (2, 4):
                model.rel_enc[li].weight.copy_(ref.rel_enc[li].weight)
                model.rel_enc[li].bias.copy_(ref.rel_enc[li].bias)
            for mod_a, mod_b in ((model.glob_enc, ref.glob_enc), (model.head, ref.head)):
                for la, lb in zip(mod_a, mod_b):
                    if hasattr(la, "weight"):
                        la.weight.copy_(lb.weight)
                    if hasattr(la, "bias"):
                        la.bias.copy_(lb.bias)

    def _slice(name: str, q: torch.Tensor) -> torch.Tensor:
        if name in ("delta.0.weight", "unit_enc.0.weight"):
            return q[:, :old_din]
        if name in ("gamma.0.weight", "rel_enc.0.weight"):
            return q[:, :old_gin]
        return q

    md = dict(model.named_parameters())
    shared_ok = all(torch.equal(p, _slice(name, md[name])) for name, p in ref.named_parameters())
    report = {
        "arm": f"{arm}-bucket",
        "init_rule": {
            "shared_modules": f"bit-exact copy of the untrained seed-0 v3c {'TypeCondContribHead' if arm == 'S' else 'OpaqueHead'}",
            "first_layer_old_columns": f"delta/unit_enc columns [0:{old_din}] copied bit-exact; gamma/rel_enc columns [0:{old_gin}] copied bit-exact",
            "first_layer_new_columns": f"{UNIT_BUCKET_DIM} unit-side + {REL_BUCKET_DIM} relation-side ZERO-initialized columns; SAME +{UNIT_BUCKET_DIM * HIDDEN + REL_BUCKET_DIM * HIDDEN} params for BOTH arms",
            "trained_historical_checkpoint_used": False,
        },
        "n_params_ref": int(sum(p.numel() for p in ref.parameters())),
        "n_params_new": int(sum(p.numel() for p in model.parameters())),
        "n_params_added": int(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in ref.parameters())),
        "added_breakdown": {
            "unit_side_new_columns": int(UNIT_BUCKET_DIM * HIDDEN),
            "relation_side_new_columns": int(REL_BUCKET_DIM * HIDDEN),
        },
        "shared_init_bit_exact": bool(shared_ok),
        "new_columns_all_zero": bool(
            (model.delta[0].weight[:, old_din:].abs().sum().item() == 0.0 and model.gamma[0].weight[:, old_gin:].abs().sum().item() == 0.0)
            if arm == "S"
            else (model.unit_enc[0].weight[:, old_din:].abs().sum().item() == 0.0 and model.rel_enc[0].weight[:, old_gin:].abs().sum().item() == 0.0)
        ),
        "shared_init_hashes": {
            name: {"ref": _hash_tensor(p), "new": _hash_tensor(_slice(name, md[name]))}
            for name, p in list(ref.named_parameters())[:6]
        },
    }
    log(
        f"[init:{arm}-bucket] ref {report['n_params_ref']} -> new {report['n_params_new']} "
        f"(+{report['n_params_added']}); shared bit-exact={shared_ok}"
    )
    return ref, model, report


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------


def prepare_all(log=print, device="cpu"):
    """v3d bucket features + split + fit-only stats."""
    blob = bf.load_bucket_cache(log=log)
    if blob is None:
        log("[features] v3d cache missing; building bucket-complete features ...")
        bf.build_bucket_cache(device=device, log=log)
        blob = bf.load_bucket_cache(log=log)
        assert blob is not None
    features = blob["features"]
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    vocab, type_means, global_mean, rel_mean = sba.fit_unit_vocab(features, idx["fit_inner"])
    from run_cscl_v3 import build_fit_stats

    tau_ctx, tau_topo = build_fit_stats(features, idx["fit_inner"], type_means, global_mean, rel_mean)
    y = np.asarray([m.y for m in features], dtype=np.float64)
    rows = np.asarray([m.row for m in features], dtype=np.int64)
    assert np.array_equal(np.sort(rows), np.arange(len(rows))) and len(rows) == 10000
    from types import SimpleNamespace

    return SimpleNamespace(
        features=features, idx=idx, vocab=vocab, type_means=type_means,
        global_mean=global_mean, rel_mean=rel_mean, tau_ctx=tau_ctx,
        tau_topo=tau_topo, y=y, audit=blob.get("audit", {}),
        provenance={k: v for k, v in blob.items() if k != "features"},
    )


# ---------------------------------------------------------------------------
# training (sequential arms; dev is NOT evaluated here)
# ---------------------------------------------------------------------------


def _eval_mae(model, batch: BucketBatch, y: np.ndarray) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred = model(batch)["pred"].cpu().numpy().astype(np.float64)
    return float(np.abs(pred - y).mean()), pred


def train_bucket_arm(arm: str, ns, device: torch.device, out_dir: Path, seed: int = 0, epochs: int = EPOCHS, log=print):
    """Train one arm; monitor-based checkpoint selection only (no dev read)."""
    assert seed == 0, "this round is seed-0 only"
    assert arm in ("S", "O")
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    fit_batch = collate_bucket(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    mon_batch = collate_bucket(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    y_fit = ns.y[ns.idx["fit_inner"]]
    y_mon = ns.y[ns.idx["monitor"]]

    _ref, model, init_report = build_head_pair(arm, ns, log=log)
    model = model.to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)  # same batch sequence as v3/v3b/v3c
    target = torch.as_tensor(y_fit, dtype=torch.float32, device=device)

    curve = []
    ckpts: list[tuple[float, int, dict]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate_bucket(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
            if arm == "S":
                loss = loss + PEN_COEF * gamma_identifiability_penalty(out)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        mon_mae, _ = _eval_mae(model, mon_batch, y_mon)
        curve.append({"epoch": epoch, "monitor_mae": mon_mae})
        ckpts.append((mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        ckpts.sort(key=lambda t: (t[0], t[1]))
        if len(ckpts) > 5:
            ckpts.pop()
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            log(f"[{arm}bucket s{seed}] ep={epoch:03d} mon={mon_mae:.5f} {time.perf_counter()-started:.0f}s")
    wall = time.perf_counter() - started

    def evaluate_monitor(state):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        f, _ = _eval_mae(model, fit_batch, y_fit)
        m, _ = _eval_mae(model, mon_batch, y_mon)
        return {"fit_mae": f, "monitor_mae": m}

    best_state = dict(ckpts[0][2])
    best_epoch = int(ckpts[0][1])
    soup_state = {k: torch.stack([s[k].float() for _, _, s in ckpts]).mean(0) for k in ckpts[0][2]}
    res_best = evaluate_monitor(best_state)
    res_soup = evaluate_monitor(soup_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"{arm}_bucket_s{seed}_best_state.pt")
    write_json(out_dir / f"init_report_{arm}_bucket_s{seed}.json", init_report)
    result = {
        "protocol": PROTOCOL_ID,
        "arm": f"{arm}-bucket",
        "seed": seed,
        "git_commit": git_commit(),
        "n_params": n_params,
        "n_params_added_vs_historical": init_report["n_params_added"],
        "epochs": epochs,
        "wall_seconds": wall,
        "device": str(device),
        "primary_best_monitor": {**res_best, "dev_mae": None},   # filled by analyze (post both-arms)
        "secondary_soup": {**res_soup, "dev_mae": None},
        "best_epoch": best_epoch,
        "soup_epochs": sorted(e for _, e, _ in ckpts),
        "curve": curve,
        "objective": {
            "S": "L1(raw y) + 1.0 * gamma batch-mean penalty (v3b/v3c verbatim; no delta penalty, no embedding regulariser)",
            "O": "L1(raw y) only (historical O objective)",
        }[arm],
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "hidden": HIDDEN, "optimizer": "Adam", "loss": "L1",
            "shuffle_generator_seed": seed + TRAIN_SHUFFLE_OFFSET,
        },
        "discipline": {
            "dev_used_for_selection": False,
            "dev_evaluated_in_train": False,
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "frozen_full_y_prediction_used": False,
        },
        "judgment_bands": JUDGMENT_BANDS,
    }
    write_json(out_dir / f"train_{arm}_bucket_s{seed}.json", result)
    log(f"[{arm}bucket s{seed}] best-monitor fit={res_best['fit_mae']:.5f} mon={res_best['monitor_mae']:.5f} (dev deferred) params={n_params} wall={wall:.0f}s")
    return result


# ---------------------------------------------------------------------------
# gauge (S arm only; reuses contribution_gauge via GaugedHead delegation)
# ---------------------------------------------------------------------------


def gauge_and_verify(ns, device, out_dir: Path, log=print) -> dict:
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    raw_model = SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    raw_model.load_state_dict(torch.load(out_dir / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True))
    raw_model = raw_model.to(torch.device(device)).eval()

    offsets, meta = _compute_fit_offsets_bucket(
        raw_model, ns, fit_pos, device, log=log
    )
    gauged = cg.GaugedHead(raw_model, offsets).to(torch.device(device)).eval()

    batches = {
        "fit": collate_bucket(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
        "monitor": collate_bucket(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
        "dev": collate_bucket(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
    }
    checks: dict[str, Any] = cg.verify_gauge_invariance(raw_model, gauged, batches, log=log)

    with torch.no_grad():
        full = gauged(batches["dev"])
        ha = gauged(collate_bucket(ns.features, dev_pos[: len(dev_pos) // 2], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device))
        hb = gauged(collate_bucket(ns.features, dev_pos[len(dev_pos) // 2 :], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device))
    comp_gap = float(max(
        (torch.cat([ha["delta"], hb["delta"]]) - full["delta"]).abs().max(),
        (torch.cat([ha["alpha"], hb["alpha"]]) - full["alpha"]).abs().max(),
        (torch.cat([ha["pred"], hb["pred"]]) - full["pred"]).abs().max(),
    ))
    checks["batch_composition_max_abs_gap"] = comp_gap
    checks["batch_composition_ok"] = bool(comp_gap <= cg.GAUGE_ATOL)
    checks["all_gates_ok"] = bool(checks["all_gates_ok"] and checks["batch_composition_ok"])

    zero_fit_ids = set(meta["zero_fit_instance_type_ids"])
    cov = {}
    for name, pos in (("fit", fit_pos), ("monitor", mon_pos), ("dev", dev_pos)):
        ut = np.concatenate([ns.features[fi].type_ids(ns.vocab) for fi in pos])
        known = np.asarray([ns.vocab.is_known(s) for fi in pos for s in ns.features[fi].unit_sigs])
        cov[name] = {
            "n_units": int(len(ut)),
            "n_units_on_zero_fit_instance_types": int(np.isin(ut, list(zero_fit_ids)).sum()) if zero_fit_ids else 0,
            "n_units_unknown_signature_unk_buckets": int((~known).sum()),
        }
    checks["gauge_coverage"] = cov

    y = {"fit": ns.y[ns.idx["fit_inner"]], "monitor": ns.y[ns.idx["monitor"]], "dev": ns.y[ns.idx["dev"]]}
    mae_inv = {}
    for name, b in batches.items():
        with torch.no_grad():
            pr = raw_model(b)["pred"].cpu().numpy().astype(np.float64)
            pc = gauged(b)["pred"].cpu().numpy().astype(np.float64)
        mae_inv[name] = {
            "mae_raw": float(np.abs(pr - y[name]).mean()),
            "mae_canonical": float(np.abs(pc - y[name]).mean()),
            "max_abs_pred_diff": float(np.abs(pr - pc).max()),
        }
    checks["mae_invariance"] = mae_inv

    cg.save_gauge(out_dir / "gauge_offsets_s0.pt", offsets, meta, raw_model.state_dict(), protocol_id=PROTOCOL_ID)
    write_json(out_dir / "gauge_meta_s0.json", meta)
    export = export_gauged_contributions(gauged, ns, dev_pos, device, out_dir / "S_bucket_s0_dev_contribs.npz")
    checks["additivity_export"] = export
    with torch.no_grad():
        pred_dev = gauged(batches["dev"])["pred"].cpu().numpy().astype(np.float64)
    np.savez_compressed(
        out_dir / "S_bucket_s0_dev_preds.npz",
        row=np.asarray(ns.idx["dev"], dtype=np.int64),
        pred=pred_dev.astype(np.float32),
        y=ns.y[ns.idx["dev"]].astype(np.float32),
    )
    write_json(out_dir / "gauge_verification_s0.json", checks)
    return checks


@torch.no_grad()
def _compute_fit_offsets_bucket(model, ns, fit_pos, device="cpu", log=print):
    """cg.compute_fit_offsets with the bucket collate (same algebra)."""
    from contribution_gauge import _hash_array

    was_training = model.training
    model.eval()
    offsets = np.zeros(ns.vocab.n_total, dtype=np.float64)
    counts = np.zeros(ns.vocab.n_total, dtype=np.int64)
    batch = collate_bucket(ns.features, list(fit_pos), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    delta = model(batch)["delta"].detach().cpu().numpy().astype(np.float64)
    unit_type = batch.unit_type.cpu().numpy()
    sums = np.zeros(ns.vocab.n_total, dtype=np.float64)
    np.add.at(sums, unit_type, delta)
    np.add.at(counts, unit_type, 1)
    present = counts > 0
    offsets[present] = sums[present] / counts[present]
    zero_ids = np.flatnonzero(~present)
    meta = {
        "n_fit_units": int(counts.sum()),
        "n_types_total": int(ns.vocab.n_total),
        "n_types_with_fit_instances": int(present.sum()),
        "n_types_zero_fit_instances": int((~present).sum()),
        "zero_fit_instance_type_ids": zero_ids.tolist(),
        "fallback_value": 0.0,
        "fallback_rule": "m_t = 0.0 for type buckets with zero fit_inner instances",
        "fit_unit_type_hash": _hash_array(unit_type),
        "fit_rows_hash": _hash_array(np.asarray(sorted(int(r) for r in ns.idx["fit_inner"]), dtype=np.int64)),
        "counts_hash": _hash_array(counts),
        "offsets_hash": _hash_array(offsets),
        "min_fit_count": int(counts.min()),
        "max_fit_count": int(counts.max()),
    }
    log(
        f"[gauge] m_t computed on {meta['n_fit_units']} fit units; "
        f"{meta['n_types_with_fit_instances']}/{meta['n_types_total']} types have fit instances; "
        f"fallback ids {meta['zero_fit_instance_type_ids']}"
    )
    if was_training:
        model.train()
    return offsets, meta


@torch.no_grad()
def export_gauged_contributions(gauged, ns, picks: Sequence[int], device, path: Path) -> dict:
    """Strict-additivity export (canonical AND raw) with unit atom membership."""
    batch = collate_bucket(ns.features, list(picks), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    out = gauged(batch)
    pred = out["pred"].cpu().numpy().astype(np.float64)
    alpha = out["alpha"].cpu().numpy().astype(np.float64)
    delta = out["delta"].cpu().numpy().astype(np.float64)
    alpha_raw = out["alpha_raw"].cpu().numpy().astype(np.float64)
    delta_raw = out["delta_raw"].cpu().numpy().astype(np.float64)
    gamma = out["gamma"].cpu().numpy().astype(np.float64)
    g_ctx = out["g_ctx"].cpu().numpy().astype(np.float64)
    g_topo = out["g_topo"].cpu().numpy().astype(np.float64)
    bias = float(out["bias"].item())
    off = gauged.offset.detach().cpu().numpy().astype(np.float64)
    unit_type = batch.unit_type.cpu().numpy()
    unit_mol = batch.unit_mol.cpu().numpy()
    usum = np.zeros(batch.n_mols)
    np.add.at(usum, unit_mol, alpha + delta)
    usum_raw = np.zeros(batch.n_mols)
    np.add.at(usum_raw, unit_mol, alpha_raw + delta_raw)
    rsum = np.zeros(batch.n_mols)
    if len(gamma):
        np.add.at(rsum, batch.rel_mol.cpu().numpy(), gamma)
    recon = bias + usum + rsum + g_ctx + g_topo
    recon_raw = bias + usum_raw + rsum + g_ctx + g_topo
    atom_flat: list[int] = []
    atom_ptr = [0]
    unit_rows: list[int] = []
    for mi, fi in enumerate(picks):
        m = ns.features[fi]
        for atoms in m.unit_atoms:
            atom_flat.extend(atoms)
            atom_ptr.append(len(atom_flat))
        unit_rows.extend([mi] * m.n_units)
    assert np.array_equal(np.asarray(unit_rows), unit_mol)
    np.savez_compressed(
        path,
        arm="S_bucket",
        protocol=PROTOCOL_ID,
        bias=bias,
        rows=np.asarray([ns.features[i].row for i in picks], dtype=np.int64),
        pred=pred,
        unit_mol=unit_mol,
        unit_type=unit_type,
        unit_offset=off[unit_type],
        alpha=alpha, delta=delta,               # canonical
        alpha_raw=alpha_raw, delta_raw=delta_raw,
        unit_atom_flat=np.asarray(atom_flat, dtype=np.int64),
        unit_atom_ptr=np.asarray(atom_ptr, dtype=np.int64),
        rel_mol=batch.rel_mol.cpu().numpy(),
        gamma=gamma,
        g_ctx=g_ctx, g_topo=g_topo,
        g_ctx_per_feature=out["g_ctx_per_feature"].cpu().numpy().astype(np.float64),
        g_topo_per_feature=out["g_topo_per_feature"].cpu().numpy().astype(np.float64),
    )
    return {
        "additivity_max_abs_residual_canonical": float(np.abs(recon - pred).max()),
        "additivity_max_abs_residual_raw": float(np.abs(recon_raw - pred).max()),
        "global_bank_sum_residual": float(np.abs(
            out["g_ctx_per_feature"].cpu().numpy().astype(np.float64).sum(1)
            + out["g_topo_per_feature"].cpu().numpy().astype(np.float64).sum(1)
            - (g_ctx + g_topo)
        ).max()),
    }


# ---------------------------------------------------------------------------
# split-half gauge stability of the new trained checkpoint (v3c procedure)
# ---------------------------------------------------------------------------

MIN_PER_HALF = 10   # pre-frozen reliability threshold
HALF_SEED = 0       # fixed subsample seed (NOT a training seed)


def split_half_gauge_stability(ns, model, device="cpu", log=print) -> dict:
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    model = model.to(torch.device(device)).eval()

    def per_type(picks):
        with torch.no_grad():
            batch = collate_bucket(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            delta = model(batch)["delta"].detach().cpu().numpy().astype(np.float64)
        ut = batch.unit_type.cpu().numpy()
        sums = np.zeros(ns.vocab.n_total)
        counts = np.zeros(ns.vocab.n_total, dtype=np.int64)
        np.add.at(sums, ut, delta)
        np.add.at(counts, ut, 1)
        return sums, counts

    rng = np.random.default_rng(HALF_SEED)
    order = rng.permutation(len(fit_pos))
    half_a = [fit_pos[i] for i in order[: len(fit_pos) // 2]]
    half_b = [fit_pos[i] for i in order[len(fit_pos) // 2 :]]
    sums_a, cnt_a = per_type(half_a)
    sums_b, cnt_b = per_type(half_b)
    m_a = np.where(cnt_a > 0, sums_a / np.maximum(cnt_a, 1), 0.0)
    m_b = np.where(cnt_b > 0, sums_b / np.maximum(cnt_b, 1), 0.0)

    with torch.no_grad():
        alpha_raw = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    alpha_can_a = alpha_raw + np.where(cnt_a > 0, m_a, 0.0)
    alpha_can_b = alpha_raw + np.where(cnt_b > 0, m_b, 0.0)

    present = (cnt_a > 0) & (cnt_b > 0)
    both_cnt = cnt_a + cnt_b
    reliable = present & (cnt_a >= MIN_PER_HALF) & (cnt_b >= MIN_PER_HALF)
    rare = present & ~reliable

    strat = {}
    for name, mask in (
        ("count_lt_20", both_cnt < 20),
        ("count_20_99", (both_cnt >= 20) & (both_cnt < 100)),
        ("count_100_499", (both_cnt >= 100) & (both_cnt < 500)),
        ("count_ge_500", both_cnt >= 500),
    ):
        sel = present & mask
        strat[name] = {
            "n_types": int(sel.sum()),
            "mean_abs_m_a_minus_m_b": float(np.abs(m_a[sel] - m_b[sel]).mean()) if sel.any() else None,
            "p90_abs_m_a_minus_m_b": float(np.percentile(np.abs(m_a[sel] - m_b[sel]), 90)) if sel.sum() >= 5 else None,
        }

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    out = {
        "question": "statistical stability of the fit-only gauge offsets of ONE trained S-bucket checkpoint across fit subsamples (NOT cross-seed training stability)",
        "half_seed": HALF_SEED,
        "n_types_present_both_halves": int(present.sum()),
        "n_types_reliable_ge_thresh_each_half": int(reliable.sum()),
        "n_types_present_but_below_threshold": int(rare.sum()),
        "n_types_total": ns.vocab.n_total,
        "reliable_types": {
            "mean_abs_m_a_minus_m_b": float(np.abs(m_a[reliable] - m_b[reliable]).mean()) if reliable.any() else None,
            "p90_abs_m_a_minus_m_b": float(np.percentile(np.abs(m_a[reliable] - m_b[reliable]), 90)) if reliable.any() else None,
            "mean_abs_alpha_canonical_split_gap": float(np.abs(alpha_can_a[reliable] - alpha_can_b[reliable]).mean()) if reliable.any() else None,
            "alpha_ordering_spearman_across_halves": _spearman(alpha_can_a[reliable], alpha_can_b[reliable]) if int(reliable.sum()) >= 2 else None,
        },
        "stratified_by_total_fit_count": strat,
        "interpretation_guard": "half-sampling noise of per-type delta means on ONE trained model; does not measure training-seed variability",
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    r = out["reliable_types"]
    log(
        "[split-half] reliable types=%d  mean|m_t^A-m_t^B|=%s (p90 %s)  spearman %s"
        % (int(reliable.sum()),
           f"{r['mean_abs_m_a_minus_m_b']:.4f}" if r["mean_abs_m_a_minus_m_b"] is not None else "n/a",
           f"{r['p90_abs_m_a_minus_m_b']:.4f}" if r["p90_abs_m_a_minus_m_b"] is not None else "n/a",
           f"{r['alpha_ordering_spearman_across_halves']:.4f}" if r["alpha_ordering_spearman_across_halves"] is not None else "n/a")
    )
    return out


# ---------------------------------------------------------------------------
# analysis: unified dev evaluation AFTER both arms are trained + selected
# ---------------------------------------------------------------------------

RECORDED_REF_FILES = {
    "F": ("feasibility_v2", "full_y_seed0_dev_preds.npz"),
}


def _load_reference_arms(rows: np.ndarray) -> dict[str, np.ndarray]:
    arms: dict[str, np.ndarray] = {}
    saved_f = np.load(sba.DEV_PREDS_NPZ)
    assert np.array_equal(saved_f["row"], rows)
    arms["F"] = saved_f["pred"].astype(np.float64)
    for name, path in (
        ("O", V3_DIR / "O_s0_dev_preds.npz"),
        ("S_v3", V3_DIR / "S_s0_dev_preds.npz"),
        ("S_gauge", V3B_DIR / "S_gauge_s0_dev_preds.npz"),
        ("S_typecond", V3C_DIR / "S_typecond_s0_dev_preds.npz"),
    ):
        z = np.load(path)
        assert np.array_equal(z["row"], rows), f"{name} row order mismatch"
        arms[name] = z["pred"].astype(np.float64)
    return arms


@torch.no_grad()
def analyze(ns, out_dir: Path, device, log=print) -> dict:
    """The unified post-both-arms dev evaluation (first dev read of the round)."""
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    rows = np.asarray(ns.idx["dev"], dtype=np.int64)
    y_dev = ns.y[rows]

    arms = _load_reference_arms(rows)
    s_model = SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    s_model.load_state_dict(torch.load(out_dir / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True))
    gauge_blob = torch.load(out_dir / "gauge_offsets_s0.pt", map_location="cpu", weights_only=False)
    gauged = cg.GaugedHead(s_model, np.asarray(gauge_blob["offsets"])).to(device).eval()
    batch_dev = collate_bucket(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    arms["S_bucket"] = gauged(batch_dev)["pred"].cpu().numpy().astype(np.float64)

    o_model = OBucketHead(ns.vocab.n_total)
    o_model.load_state_dict(torch.load(out_dir / "O_bucket_s0_best_state.pt", map_location="cpu", weights_only=True))
    o_model = o_model.to(device).eval()
    arms["O_bucket"] = o_model(batch_dev)["pred"].cpu().numpy().astype(np.float64)

    mae = {a: float(np.abs(p - y_dev).mean()) for a, p in arms.items()}
    check = {a: bool(abs(mae[a] - RECORDED_MAE[a]) <= 1e-4) for a in RECORDED_MAE}
    assert all(check.values()), f"reference files do not reproduce recorded MAEs: {check}"

    tr_s = json.loads((out_dir / "train_S_bucket_s0.json").read_text())
    tr_o = json.loads((out_dir / "train_O_bucket_s0.json").read_text())
    # fill the deferred dev metrics into the train records (post-selection)
    tr_s["primary_best_monitor"]["dev_mae"] = mae["S_bucket"]
    tr_o["primary_best_monitor"]["dev_mae"] = mae["O_bucket"]
    write_json(out_dir / "train_S_bucket_s0.json", tr_s)
    write_json(out_dir / "train_O_bucket_s0.json", tr_o)

    gaps = {
        f"{a}_minus_{b}": paired_bootstrap(np.abs(arms[a] - y_dev), np.abs(arms[b] - y_dev))
        for a, b in (
            ("S_bucket", "O_bucket"), ("S_bucket", "F"), ("O_bucket", "F"),
            ("S_bucket", "S_typecond"), ("O_bucket", "O"),
        )
    }
    s_o = mae["S_bucket"] - mae["O_bucket"]
    s_f = mae["S_bucket"] - mae["F"]
    o_f = mae["O_bucket"] - mae["F"]
    s_hist = mae["S_bucket"] - mae["S_typecond"]
    o_hist = mae["O_bucket"] - mae["O"]
    if s_o <= 0.02 and s_f <= 0.02:
        band = "case_A"
    elif o_hist <= 0.005 and s_o > 0.02:
        band = "case_B"
    elif s_hist >= -0.002 and o_hist >= -0.002:
        band = "case_C"
    elif o_hist < s_hist - 1e-9:
        band = "case_D"
    else:
        band = "indeterminate"

    summary: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "primary_rule": "best-monitor single checkpoint; S_bucket uses canonical (gauged) predictions; gauge is prediction-invariant",
        "mae": mae,
        "recorded_reference_check": check,
        "gap_values": {"S_bucket-O_bucket": s_o, "S_bucket-F": s_f, "O_bucket-F": o_f,
                       "S_bucket-S_v3c": s_hist, "O_bucket-O_v3": o_hist},
        "judgment_gaps": gaps,
        "judgment_band": band,
        "judgment_bands": JUDGMENT_BANDS,
        "secondary_soup": {"S_bucket": tr_s["secondary_soup"], "O_bucket": tr_o["secondary_soup"]},
        "capacity_disclosure": {
            "params_S_bucket": tr_s["n_params"], "params_O_bucket": tr_o["n_params"],
            "params_added_each": tr_s["n_params_added_vs_historical"],
            "note": "both arms received the SAME +61,760 params; vs-historical deltas confound information with capacity",
        },
        "loss_disclosure": "S_bucket keeps the historical gamma penalty (1.0); O_bucket uses L1 only; F's checkpoint rule is monitor top-5 soup (differs from best-monitor primaries)",
        "fit_monitor_dev": {
            "S_bucket": tr_s["primary_best_monitor"], "O_bucket": tr_o["primary_best_monitor"],
        },
        "epochs_best": {"S_bucket": tr_s["best_epoch"], "O_bucket": tr_o["best_epoch"]},
        "wall_seconds": {"S_bucket": tr_s["wall_seconds"], "O_bucket": tr_o["wall_seconds"]},
        "official_valid_loaded": False, "official_test_loaded": False,
    }

    dataset = cf.load_official_train()
    summary["subgroups"] = subgroup_table_v3b(rows, y_dev, arms, ns, dataset)

    # ---- S_bucket contribution audit ----
    z = np.load(out_dir / "S_bucket_s0_dev_contribs.npz")
    n_mols = len(rows)
    ut, um = z["unit_type"], z["unit_mol"]
    summary["channel_var_shares"] = {}
    for tag, a_key, d_key in (("canonical", "alpha", "delta"), ("raw", "alpha_raw", "delta_raw")):
        unit = np.zeros(n_mols)
        np.add.at(unit, um, z[a_key] + z[d_key])
        rel = np.zeros(n_mols)
        if len(z["gamma"]):
            np.add.at(rel, z["rel_mol"], z["gamma"])
        ch = {"unit": unit, "rel": rel, "g_ctx": z["g_ctx"], "g_topo": z["g_topo"]}
        summary[f"channel_var_shares_{tag}"] = {k: float(np.var(v) / max(np.var(z["pred"]), 1e-12)) for k, v in ch.items()}
        if tag == "canonical":
            C = np.stack([ch[k] for k in ("unit", "rel", "g_ctx", "g_topo")])
            summary["channel_corr_matrix_canonical"] = np.corrcoef(C).tolist()

    # gamma bucket-usage audit on real ZINC (relation-level)
    rel_bond_hist = np.concatenate([
        ns.features[fi].rel_bond_hist for fi in dev_pos if len(ns.features[fi].rel_lo)
    ], axis=0)
    rel_counts = np.concatenate([ns.features[fi].rel_count for fi in dev_pos if len(ns.features[fi].rel_lo)])
    rel_bcount = np.concatenate([ns.features[fi].rel_bucket_counts for fi in dev_pos if len(ns.features[fi].rel_lo)], axis=0)
    gamma = z["gamma"]
    bonded = rel_bond_hist.sum(1) > 0
    frac0 = rel_bcount[:, 0] / np.maximum(rel_counts, 1.0)
    summary["gamma_audit"] = {
        "n_relations": int(len(gamma)),
        "var_share_canonical": summary["channel_var_shares_canonical"]["rel"],
        "abs_gamma_mean_bonded": float(np.abs(gamma[bonded]).mean()),
        "abs_gamma_mean_not_bonded": float(np.abs(gamma[~bonded]).mean()),
        "frac_abs_le_0.01": float((np.abs(gamma) <= 0.01).mean()),
        "corr_abs_gamma_frac_bucket0": float(np.corrcoef(np.abs(gamma), frac0)[0, 1]),
        "corr_abs_gamma_frac_bucket0_bonded_only": float(np.corrcoef(np.abs(gamma[bonded]), frac0[bonded])[0, 1]) if bonded.sum() > 10 else None,
        "gamma_std": float(gamma.std()),
        "note": "gamma var share is descriptive, not an independent explained fraction; usage of bucket info does not imply chemical causality",
    }

    # UNK stratified MAE for ALL arms on identical molecules
    known_unit = np.concatenate([
        np.asarray([ns.vocab.is_known(s) for s in ns.features[fi].unit_sigs]) for fi in dev_pos
    ])
    unit_mol_all = np.concatenate([np.full(ns.features[fi].n_units, i) for i, fi in enumerate(dev_pos)])
    has_unk = np.zeros(n_mols, dtype=bool)
    np.add.at(has_unk, unit_mol_all[~known_unit], 1)
    out_unk = {"n_with_unk": int(has_unk.sum()), "n_without_unk": int((~has_unk).sum())}
    for grp_name, mask in (("with_unk", has_unk), ("without_unk", ~has_unk)):
        out_unk[f"mae_{grp_name}"] = {a: float(np.abs(p[mask] - y_dev[mask]).mean()) for a, p in arms.items()}
        out_unk[f"paired_{grp_name}"] = {
            f"{a}_minus_{b}": paired_bootstrap(np.abs(arms[a][mask] - y_dev[mask]), np.abs(arms[b][mask] - y_dev[mask]))
            for a, b in (("S_bucket", "O_bucket"), ("S_bucket", "F"), ("O_bucket", "F"))
        }
    summary["unk_audit"] = out_unk

    write_json(out_dir / "analysis_s0.json", summary)
    log("[analyze] " + json.dumps({k: round(v, 5) for k, v in mae.items()}))
    log(f"[analyze] gaps: S-O {s_o:+.5f}  S-F {s_f:+.5f}  O-F {o_f:+.5f}  band={band}")
    return summary


# ---------------------------------------------------------------------------
# synthetic known-truth tasks (CPU seed 0): A = v3b/v3c verbatim; B = bucket-specific
# ---------------------------------------------------------------------------


def _hash_value(payload: str) -> float:
    h = hashlib.sha256(payload.encode()).hexdigest()
    return (int(h[:8], 16) % 2000) / 1000.0 - 1.0


def _syn_truth_context(ns, fit_pos):
    """Shared truth scaffolding of the v3b/v3c corrected task (verbatim)."""
    fit_type_counts: dict[int, int] = {}
    for fi in fit_pos:
        for t in ns.features[fi].type_ids(ns.vocab):
            fit_type_counts[int(t)] = fit_type_counts.get(int(t), 0) + 1
    freq_types = sorted(fit_type_counts, key=lambda t: -fit_type_counts[t])
    top32 = freq_types[:32]
    alpha_star = {t: _hash_value(f"alpha|{t}") for t in top32}

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
    type_of_A = {pair_A[0], pair_A[1]}
    confusers = [k for k in cand[3:] if k[0] in type_of_A or k[1] in type_of_A]
    pair_D = confusers[0] if confusers else cand[3]

    attach_sum, attach_n = {}, {}
    for fi in fit_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        ac = attach_counts(m)
        for k, t in enumerate(tids):
            attach_sum[int(t)] = attach_sum.get(int(t), 0.0) + float(ac[k])
            attach_n[int(t)] = attach_n.get(int(t), 0) + 1
    attach_mean = {t: attach_sum[t] / attach_n[t] for t in attach_n}
    global_attach = float(np.mean([float(attach_counts(ns.features[fi]).mean()) for fi in fit_pos]))
    return top32, alpha_star, rel_type_count, (pair_A, pair_B, pair_C, pair_D), attach_mean, global_attach


def _syn_labels(picks, ns, alpha_star, attach_mean, global_attach, gamma_fn):
    ys = []
    for fi in picks:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        ac = attach_counts(m)
        total = 0.0
        for k, t in enumerate(tids):
            total += alpha_star.get(int(t), 0.0)
            total += 0.8 * (float(ac[k]) - attach_mean.get(int(t), global_attach))
        for j, (lo, hi) in enumerate(zip(m.rel_lo, m.rel_hi)):
            key = (int(tids[lo]), int(tids[hi]))
            total += gamma_fn(key, j, m)
        ys.append(total)
    return np.asarray(ys, dtype=np.float64)


def _train_synthetic(ns, device, y_syn_fit, y_syn_mon, log):
    """Fresh S-bucket head on the synthetic target; monitor best checkpoint."""
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    seed_everything(0)
    _ref, model, _ = build_head_pair("S", ns, log=lambda *a, **k: None)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(101)
    mon_batch = collate_bucket(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    target = torch.as_tensor(y_syn_fit, dtype=torch.float32, device=device)
    best = (1e9, 0, None)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate_bucket(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
            loss = loss + PEN_COEF * gamma_identifiability_penalty(out)
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
    return model, best


def _syn_recovery(ns, device, model, dev_pos, top32, alpha_star, rel_type_count, attach_mean, global_attach, pairs):
    """Shared recovery metrics: alpha, delta (raw/canonical), gamma per pair."""
    pair_A, pair_B, pair_C, pair_D = pairs
    dev_batch = collate_bucket(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    with torch.no_grad():
        out = model(dev_batch)
    alpha_hat = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    delta_hat = out["delta"].cpu().numpy().astype(np.float64)
    gamma_hat = out["gamma"].cpu().numpy().astype(np.float64)
    offsets, meta = _compute_fit_offsets_bucket(
        model, ns, _index_by_row(ns.features, ns.idx["fit_inner"]), device, log=lambda *a, **k: None
    )
    gauged = cg.GaugedHead(model, offsets).to(device).eval()
    with torch.no_grad():
        gout = gauged(dev_batch)
        pred_raw_syn = out["pred"].cpu().numpy().astype(np.float64)
        pred_can_syn = gout["pred"].cpu().numpy().astype(np.float64)
    gauge_pred_gap = float(np.abs(pred_raw_syn - pred_can_syn).max())
    delta_can = gout["delta"].cpu().numpy().astype(np.float64)

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    a_true = np.array([alpha_star[t] for t in top32])
    a_hat = np.array([alpha_hat[t] for t in top32])
    sign_match = float((np.sign(a_true) == np.sign(a_hat)).mean())

    um = dev_batch.unit_mol.cpu().numpy()
    true_delta, hat_raw, hat_can, alpha_of_unit = [], [], [], []
    for mi, fi in enumerate(dev_pos):
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        ac = attach_counts(m)
        hats = delta_hat[um == mi]
        hats_can = delta_can[um == mi]
        assert len(hats) == m.n_units
        for k, t in enumerate(tids):
            true_delta.append(0.8 * (float(ac[k]) - attach_mean.get(int(t), global_attach)))
            hat_raw.append(float(hats[k]))
            hat_can.append(float(hats_can[k]))
            alpha_of_unit.append(alpha_star.get(int(t), 0.0))
    true_delta = np.asarray(true_delta)
    hat_raw = np.asarray(hat_raw)
    hat_can = np.asarray(hat_can)
    alpha_of_unit = np.asarray(alpha_of_unit)

    frac0_by_pair: dict[tuple[int, int], list[float]] = {p: [] for p in (pair_A, pair_B, pair_C, pair_D)}
    ghat_by_pair: dict[tuple[int, int], list[float]] = {p: [] for p in (pair_A, pair_B, pair_C, pair_D)}
    unseen_hat, seen_hat = [], []
    idx_rel = 0
    for fi in dev_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for j, (lo, hi) in enumerate(zip(m.rel_lo, m.rel_hi)):
            key = (int(tids[lo]), int(tids[hi]))
            if key in ghat_by_pair:
                frac0_by_pair[key].append(float(m.rel_bucket_counts[j, 0]) / max(float(m.rel_count[j]), 1.0))
                ghat_by_pair[key].append(float(gamma_hat[idx_rel + j]))
            (unseen_hat if key not in rel_type_count else seen_hat).append(float(gamma_hat[idx_rel + j]))
        idx_rel += len(m.rel_lo)

    def _slope(xs, ys_):
        if len(xs) < 5 or np.std(xs) < 1e-9:
            return None, None
        return float(np.polyfit(np.asarray(xs), np.asarray(ys_), 1)[0]), float(np.corrcoef(np.asarray(xs), np.asarray(ys_))[0, 1])

    stars = {"pair_A": 0.9, "pair_B": -0.7, "pair_C": 0.5, "pair_D_confuser": 0.0}
    gamma_checks = {}
    for name, key in (("pair_A", pair_A), ("pair_B", pair_B), ("pair_C", pair_C), ("pair_D_confuser", pair_D)):
        g = ghat_by_pair[key]
        f = frac0_by_pair[key]
        slope, corr = _slope(f, g)
        gamma_checks[name] = {
            "types": [int(key[0]), int(key[1])],
            "n_dev_relations": len(g),
            "gamma_hat_mean": float(np.mean(g)) if g else None,
            "gamma_star_constant_part": stars[name],
            "bucket_modulation_slope": slope,
            "bucket_modulation_corr": corr,
        }

    # alpha/gamma confusion (canonical), following the v3c synthetic
    dev_ut = dev_batch.unit_type.cpu().numpy()
    a_can_by_type = {int(t): float(gauged.alpha_weight_canonical(dev_ut, t)) for t in np.unique(dev_ut)} if False else None
    alpha_can_flat = gout["alpha"].cpu().numpy().astype(np.float64)
    a_can_by_type = {int(t): float(alpha_can_flat[dev_ut == t][0]) for t in np.unique(dev_ut)}
    per_type_gamma: dict[int, list[float]] = {}
    idx_rel = 0
    for fi in dev_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for j in range(len(m.rel_lo)):
            for tt in (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]])):
                per_type_gamma.setdefault(tt, []).append(float(gamma_hat[idx_rel + j]))
        idx_rel += len(m.rel_lo)
    common = [t for t in per_type_gamma if t in a_can_by_type]
    alpha_gamma_confusion = float(np.corrcoef(
        np.array([a_can_by_type[t] for t in common]),
        np.array([np.mean(per_type_gamma[t]) for t in common]),
    )[0, 1]) if len(common) >= 3 else None

    return {
        "alpha_sign_recovery_top32": sign_match,
        "alpha_spearman_top32_raw": _spearman(a_true, a_hat),
        "delta_recovery_pearson_raw": float(np.corrcoef(true_delta, hat_raw)[0, 1]),
        "delta_recovery_pearson_canonical": float(np.corrcoef(true_delta, hat_can)[0, 1]),
        "delta_hat_vs_alpha_star_confusion_raw": float(np.corrcoef(hat_raw, alpha_of_unit)[0, 1]),
        "delta_hat_vs_alpha_star_confusion_canonical": float(np.corrcoef(hat_can, alpha_of_unit)[0, 1]),
        "alpha_gamma_confusion_corr_canonical": alpha_gamma_confusion,
        "gauge_max_abs_pred_diff": gauge_pred_gap,
        "gamma_checks": gamma_checks,
        "unseen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(unseen_hat))) if unseen_hat else None,
        "seen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(seen_hat))) if seen_hat else None,
        "n_unseen_dev_relations": len(unseen_hat),
        "gauge_meta_fit_units": meta["n_fit_units"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "official_valid_loaded": False, "official_test_loaded": False,
    }


def synthetic_task_A(ns, out_dir: Path, device, log=print) -> dict:
    """Task A: the v3b/v3c corrected attachment task, verbatim truth."""
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    top32, alpha_star, rel_type_count, pairs, attach_mean, global_attach = _syn_truth_context(ns, fit_pos)
    pair_A = pairs[0]

    def gamma_fn(key, j, m):
        return 0.9 if key == pair_A else (-0.7 if key == pairs[1] else (0.5 if key == pairs[2] else 0.0))

    rng = np.random.default_rng(0)
    y_fit = _syn_labels(fit_pos, ns, alpha_star, attach_mean, global_attach, gamma_fn) + 0.05 * rng.standard_normal(len(fit_pos))
    y_mon = _syn_labels(mon_pos, ns, alpha_star, attach_mean, global_attach, gamma_fn)
    y_dev = _syn_labels(dev_pos, ns, alpha_star, attach_mean, global_attach, gamma_fn)
    model, best = _train_synthetic(ns, device, y_fit, y_mon, log)
    dev_batch = collate_bucket(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    with torch.no_grad():
        dev_pred = model(dev_batch)["pred"].cpu().numpy().astype(np.float64)
    res = _syn_recovery(ns, device, model, dev_pos, top32, alpha_star, rel_type_count, attach_mean, global_attach, pairs)
    res.update({
        "protocol": PROTOCOL_ID,
        "task": "corrected synthetic attachment task (v3b/v3c verbatim truth; gamma constant per pair; noise 0.05; seed 0)",
        "attachment_index": cf.D_HIST + 4,
        "comparability": "same truth generator/data/seeds as results/cscl_v3b/synthetic_gauge_s0.json and results/cscl_v3c/synthetic_typecond_s0.json; task B (bucket-modulated) reported separately",
        "best_monitor_epoch": int(best[1]),
        "monitor_mae_syn": float(best[0]),
        "dev_mae_syn": float(np.abs(dev_pred - y_dev).mean()),
    })
    write_json(out_dir / "synthetic_A_attachment_s0.json", res)
    log("[synthetic-A] delta-r(raw) %.2f -> can %.3f; gamma_A mean %s" % (
        res["delta_recovery_pearson_raw"], res["delta_recovery_pearson_canonical"],
        res["gamma_checks"]["pair_A"]["gamma_hat_mean"]))
    return res


def synthetic_task_B(ns, out_dir: Path, device, log=print) -> dict:
    """Task B (pre-frozen): pair_A's contribution is bucket-modulated,
    gamma_r = 0.9 * frac_0(r), frac_0(r) = rel_bucket_counts[r,0]/rel_count[r];
    pair_B constant -0.7; pair_C constant +0.5.  Tests whether gamma uses the
    restored distance-conditioned information to distinguish the SAME type pair
    at different distance compositions."""
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    top32, alpha_star, rel_type_count, pairs, attach_mean, global_attach = _syn_truth_context(ns, fit_pos)
    pair_A, pair_B, pair_C = pairs[0], pairs[1], pairs[2]

    def gamma_fn(key, j, m):
        if key == pair_A:
            cnt0 = float(m.rel_bucket_counts[j, 0])
            tot = max(float(m.rel_count[j]), 1.0)
            return 0.9 * (cnt0 / tot)
        return -0.7 if key == pair_B else (0.5 if key == pair_C else 0.0)

    rng = np.random.default_rng(0)
    y_fit = _syn_labels(fit_pos, ns, alpha_star, attach_mean, global_attach, gamma_fn) + 0.05 * rng.standard_normal(len(fit_pos))
    y_mon = _syn_labels(mon_pos, ns, alpha_star, attach_mean, global_attach, gamma_fn)
    y_dev = _syn_labels(dev_pos, ns, alpha_star, attach_mean, global_attach, gamma_fn)
    model, best = _train_synthetic(ns, device, y_fit, y_mon, log)
    dev_batch = collate_bucket(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    with torch.no_grad():
        dev_pred = model(dev_batch)["pred"].cpu().numpy().astype(np.float64)
    res = _syn_recovery(ns, device, model, dev_pos, top32, alpha_star, rel_type_count, attach_mean, global_attach, pairs)
    res.update({
        "protocol": PROTOCOL_ID,
        "task": "bucket-modulated interaction task (PRE-FROZEN truth): pair_A gamma = 0.9*frac_0(r); pair_B -0.7; pair_C +0.5; noise 0.05; seed 0",
        "truth": {"pair_A": "0.9 * rel_bucket_counts[r,0]/rel_count[r]", "pair_B": -0.7, "pair_C": 0.5, "pair_D": 0.0},
        "interpretation": {
            "expected": "pair_A bucket_modulation_slope ~ 1.0 with high corr (gamma uses bucket info); pair_B/C slopes ~ 0; confuser ~ 0",
            "note": "separate from task A; no generator tuning; failures are reported as failures",
        },
        "best_monitor_epoch": int(best[1]),
        "monitor_mae_syn": float(best[0]),
        "dev_mae_syn": float(np.abs(dev_pred - y_dev).mean()),
    })
    write_json(out_dir / "synthetic_B_bucket_modulated_s0.json", res)
    a_check = res["gamma_checks"]["pair_A"]
    log("[synthetic-B] pair_A slope %s corr %s; B/C slopes %s/%s" % (
        a_check["bucket_modulation_slope"], a_check["bucket_modulation_corr"],
        res["gamma_checks"]["pair_B"]["bucket_modulation_slope"],
        res["gamma_checks"]["pair_C"]["bucket_modulation_slope"]))
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _untrained_pred_check(arm: str, ns, ref, model) -> dict:
    dev64 = _index_by_row(ns.features, ns.idx["dev"][:64])
    b0 = collate_bucket(ns.features, dev64, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    ref.eval(); model.eval()
    with torch.no_grad():
        p_ref = ref(b0)["pred"].numpy().astype(np.float64)
        p_new = model(b0)["pred"].numpy().astype(np.float64)
    return {
        "untrained_max_abs_pred_diff_vs_ref": float(np.abs(p_new - p_ref).max()),
        "untrained_pred_exact_equal": bool(np.array_equal(p_new, p_ref)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("audit-init", "train-S", "train-O", "gauge", "analyze", "splithalf", "synthetic", "all"))
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda *a, **k: print(*a, **k, flush=True)  # noqa: E731

    if args.mode == "audit-init":
        ns = prepare_all(log=log)
        for arm in ("S", "O"):
            ref, model, report = build_head_pair(arm, ns, log=log)
            report.update(_untrained_pred_check(arm, ns, ref, model))
            write_json(args.out / f"init_report_{arm}_bucket_s0.json", report)
            log(f"[audit-init:{arm}] untrained pred diff {report['untrained_max_abs_pred_diff_vs_ref']:.3e} (exact={report['untrained_pred_exact_equal']})")
        return 0
    if args.mode == "train-S":
        ns = prepare_all(log=log, device=str(device))
        train_bucket_arm("S", ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        return 0
    if args.mode == "train-O":
        ns = prepare_all(log=log, device=str(device))
        train_bucket_arm("O", ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        return 0
    if args.mode == "gauge":
        ns = prepare_all(log=log)
        gauge_and_verify(ns, device, args.out, log=log)
        return 0
    if args.mode == "analyze":
        ns = prepare_all(log=log)
        analyze(ns, args.out, device, log=log)
        return 0
    if args.mode == "splithalf":
        ns = prepare_all(log=log)
        model = SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        model.load_state_dict(torch.load(args.out / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True))
        write_json(args.out / "split_half_gauge_stability_s0.json", split_half_gauge_stability(ns, model, device="cpu", log=log))
        return 0
    if args.mode == "synthetic":
        ns = prepare_all(log=log)
        synthetic_task_A(ns, args.out, torch.device("cpu"), log=log)
        synthetic_task_B(ns, args.out, torch.device("cpu"), log=log)
        return 0
    if args.mode == "all":
        log("[all] train S-bucket (seed 0) ...")
        ns = prepare_all(log=log, device=str(device))
        train_bucket_arm("S", ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        log("[all] train O-bucket (seed 0, sequential) ...")
        train_bucket_arm("O", ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        log("[all] fit-only gauge + verification (S arm) ...")
        gauge_and_verify(ns, device, args.out, log=log)
        log("[all] unified dev analysis (F/O/S_v3/S_gauge/S_v3c/O_bucket/S_bucket) ...")
        analyze(ns, args.out, device, log=log)
        log("[all] split-half gauge stability ...")
        model = SBucketContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        model.load_state_dict(torch.load(args.out / "S_bucket_s0_best_state.pt", map_location="cpu", weights_only=True))
        write_json(args.out / "split_half_gauge_stability_s0.json", split_half_gauge_stability(ns, model, device="cpu", log=log))
        log("[all] synthetic task A (corrected attachment, verbatim truth) ...")
        synthetic_task_A(ns, args.out, torch.device("cpu"), log=log)
        log("[all] synthetic task B (bucket-modulated, pre-frozen truth) ...")
        synthetic_task_B(ns, args.out, torch.device("cpu"), log=log)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
