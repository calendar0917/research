"""cscl-v3c driver: Type-Conditioned Gauge head (S-typecond-gauge), seed 0.

Protocol: ``configs/cscl_v3c_type_conditioned_gauge.yaml`` (frozen before any
formal training).

Research question (single mechanism intervention on top of cscl-v3b): the
v3b S-gauge contextual correction is ``delta_u = h(z_u, c_u)`` -- the shared
structural type ``t_u`` enters ONLY through the per-type scalar ``alpha(t_u)``,
while the opaque control O feeds a 16-D type embedding into its nonlinear unit
encoder.  v3c makes the contextual correction explicitly type-conditional,

    delta_u = h(z_u, c_u, Emb(t_u)),      Emb = Embedding(vocab_size, 16),

while everything else is held at the v3b configuration: per-type scalar alpha,
unchanged gamma (input/params/width), unchanged global context/topology banks
and bias, unchanged frozen Full-Y encoder, unchanged split, raw-y L1 +
1.0 * gamma batch-mean penalty (NO delta penalty, NO embedding regulariser),
same optimizer/checkpoint rule, and the SAME fit-only prediction-invariant
gauge normalization (alpha(t) += m_t, delta(u) -= m_{t(u)}).

Initialization discipline (maximal control of init confounds): the untrained
v3b SBContribHead is built under seed 0 and every shape-unchanged parameter is
copied bit-exact; the delta first layer keeps the old input columns bit-exact
and the 16 new columns are zero-initialized; the type embedding is initialized
from an independent deterministic generator stream.  The new model is trained
from scratch (the v3b TRAINED checkpoint is never loaded).  The extra capacity
(+10,176 params) is an unavoidable confound and is disclosed.

seed 0 only; official valid/test never loaded; the frozen Full-Y prediction is
never an input or a supervision target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_CODE_DIR))

import contribution_gauge as cg  # noqa: E402
import cscl_features as cf  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
from run_cscl_v3 import (  # noqa: E402  (reuse: collate/split/stats/utils)
    SBContribHead,
    V3Batch,
    BATCH_SIZE,
    EPOCHS,
    GRAD_CLIP,
    HIDDEN,
    LR,
    PEN_COEF,
    RELIN,
    TRAIN_SHUFFLE_OFFSET,
    TYPE_EMB,
    WEIGHT_DECAY,
    ZDIM,
    _batches,
    _index_by_row,
    collate,
    git_commit,
    mlp,
    paired_bootstrap,
    prepare_all,
    seed_everything,
    write_json,
)
from run_cscl_v3b import (  # noqa: E402  (reuse: v3b objective + audit helpers)
    attach_counts,
    gamma_identifiability_penalty,
    subgroup_table_v3b,
)

PROTOCOL_ID = "cscl-v3c-type-conditioned-gauge"
OUT_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3c"
V3B_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b"
V3_DIR = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3"

#: width of the NEW shared type-conditional embedding entering delta only
#: (same 16-D convention as the opaque control's unit-encoder type channel)
TYPE_EMB_DIM = TYPE_EMB
#: independent deterministic stream for the new embedding (does not touch the
#: seed-0 stream that defines the copied shared init)
TYPE_EMB_SEED = 0

JUDGMENT_BANDS = {
    "case_A": "dev_mae <= 0.16 with intact mechanism (strict additivity, lossless canonicalization, no alpha split-half degradation, synthetic recovery preserved, no gamma/global bypass) -> supports the type-conditioning hypothesis; next round (NOT this one) may unify distance-bucket moments with type-conditioned contributions",
    "case_B": "0.16 < dev_mae < 0.17 -> limited improvement; inspect frequent-type concentration, low-frequency overfit, delta absorbing type main effects, alpha split stability",
    "case_C": "dev_mae >= 0.17 -> the added type conditioning does not close the gap; do NOT grow the embedding; next round pivots to the full per-distance-bucket pair-moment separable readout",
    "special_case": "better MAE with worse interpretation (unstable canonical alpha, failed synthetic recovery, gamma/global compensation) -> predictive improvement only, no methodological claim",
}


# ---------------------------------------------------------------------------
# the ONLY new architecture: Type-Conditioned Delta head
# ---------------------------------------------------------------------------


class TypeCondContribHead(nn.Module):
    """prediction = bias + sum_u (alpha_type + delta(z,c,Emb(type)))
    + sum_r gamma + g_ctx + g_topo.

    Identical to SBContribHead except: delta's input is
    ``[unit_feat | unit_ctx | type_emb(unit_type)]`` (440 = 384 + 40 + 16;
    the v3/v3b delta input is 424 = 384 + 40 -- the '36-D/420' figure in older
    reports is a documentation error, DESC_DIM = 40 in code).

    Only delta reads the type embedding: alpha stays ``Embedding(vocab, 1)``
    (one shared scalar per type); gamma / global banks / bias are untouched;
    there is NO post-sum fusion MLP.
    """

    def __init__(self, vocab_size: int, tau_ctx: np.ndarray, tau_topo: np.ndarray):
        super().__init__()
        # Same module construction order as SBContribHead (alpha, delta, gamma,
        # buffers, banks, bias, alpha re-init) so the copied seed-0 init is
        # reproducible; the NEW embedding is created LAST and initialized from
        # an independent generator, never consuming the shared stream.
        self.alpha = nn.Embedding(vocab_size, 1)
        self.delta = mlp([ZDIM + sba.DESC_DIM + TYPE_EMB_DIM, HIDDEN, HIDDEN, 1])
        self.gamma = mlp([RELIN, HIDDEN, HIDDEN, 1])
        assert tau_ctx.shape == (sba.CTX_STRUCTURE, 3) and tau_topo.shape == (sba.TOPO_DIM, 3)
        self.register_buffer("tau_ctx", torch.from_numpy(tau_ctx.copy()))
        self.register_buffer("tau_topo", torch.from_numpy(tau_topo.copy()))
        self.w_ctx = nn.Parameter(torch.zeros(sba.CTX_STRUCTURE, 1 + 3))
        self.w_topo = nn.Parameter(torch.zeros(sba.TOPO_DIM, 1 + 3))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.alpha.weight, std=0.01)
        # NEW: shared type embedding, independent deterministic stream
        self.type_emb = nn.Embedding(vocab_size, TYPE_EMB_DIM)
        gen = torch.Generator().manual_seed(TYPE_EMB_SEED)
        with torch.no_grad():
            self.type_emb.weight.copy_(torch.randn(vocab_size, TYPE_EMB_DIM, generator=gen) * 0.1)

    def unit_terms(self, batch: V3Batch):
        alpha = self.alpha(batch.unit_type).squeeze(-1)
        z = torch.cat([batch.unit_feat, batch.unit_ctx, self.type_emb(batch.unit_type)], dim=1)
        delta = self.delta(z).squeeze(-1)
        return alpha, delta

    # relation_terms / _bank / forward are structurally identical to
    # SBContribHead; duplicated explicitly so AST tests can pin that type_emb
    # is read ONLY inside unit_terms (never in gamma / global / post-sum).
    def relation_terms(self, batch: V3Batch):
        if batch.rel_feat.shape[0] == 0:
            return torch.zeros(0, device=batch.graph_feat.device)
        return self.gamma(batch.rel_feat).squeeze(-1)

    @staticmethod
    def _bank(x: torch.Tensor, tau: torch.Tensor, w: nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [x]
        for k in range(tau.shape[1]):
            feats.append(torch.relu(x - tau[:, k].unsqueeze(0)))
        H = torch.stack(feats, dim=2)
        per_feature = (H * w.unsqueeze(0)).sum(dim=2)
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
            "pred": pred, "alpha": alpha, "delta": delta, "gamma": gamma,
            "g_ctx": g_ctx, "g_topo": g_topo,
            "g_ctx_per_feature": ctx_pf, "g_topo_per_feature": topo_pf,
            "bias": self.bias.detach(),
        }


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def build_head_pair(ns, log=print):
    """(ref_untrained_v3b_head, new_head) under the frozen init discipline.

    1. seed 0 -> untrained SBContribHead (the v3b architecture, untrained).
    2. seed 0 -> TypeCondContribHead.
    3. every shape-unchanged parameter is copied BIT-EXACT from (1) into (2);
       the delta first layer keeps the 424 legacy input columns bit-exact and
       zero-fills the 16 new columns; the type embedding keeps its independent
       deterministic init (verified against a fresh stream).
    """
    seed_everything(0)
    ref = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    seed_everything(0)
    model = TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    zin = ZDIM + sba.DESC_DIM  # 424 legacy delta input width
    with torch.no_grad():
        model.alpha.weight.copy_(ref.alpha.weight)
        model.delta[0].weight[:, :zin].copy_(ref.delta[0].weight)
        model.delta[0].weight[:, zin:].zero_()
        model.delta[0].bias.copy_(ref.delta[0].bias)
        for li in (2, 4):
            model.delta[li].weight.copy_(ref.delta[li].weight)
            model.delta[li].bias.copy_(ref.delta[li].bias)
        for gi in (0, 2, 4):
            model.gamma[gi].weight.copy_(ref.gamma[gi].weight)
            model.gamma[gi].bias.copy_(ref.gamma[gi].bias)
        model.w_ctx.copy_(ref.w_ctx)
        model.w_topo.copy_(ref.w_topo)
        model.bias.copy_(ref.bias)
        model.tau_ctx.copy_(ref.tau_ctx)
        model.tau_topo.copy_(ref.tau_topo)
        gen = torch.Generator().manual_seed(TYPE_EMB_SEED)
        emb_ref = torch.randn(ns.vocab.n_total, TYPE_EMB_DIM, generator=gen) * 0.1
        assert torch.equal(model.type_emb.weight, emb_ref), "type emb init stream drifted"

    zin = ZDIM + sba.DESC_DIM  # 424 legacy delta input width

    def _shared_bit_exact() -> bool:
        md = dict(model.named_parameters())
        for name, p in ref.named_parameters():
            q = md[name]
            if name == "delta.0.weight":
                if not torch.equal(p, q[:, :zin]):
                    return False
            elif not torch.equal(p, q):
                return False
        return True

    report = {
        "init_rule": {
            "shared_modules": "bit-exact copy of the untrained seed-0 v3b SBContribHead",
            "delta_first_layer": f"columns [0:{zin}] copied bit-exact; columns [{zin}:{zin + TYPE_EMB_DIM}] zero-initialized (frozen rule)",
            "type_embedding": f"Embedding(vocab={ns.vocab.n_total}, {TYPE_EMB_DIM}) ~ N(0, 0.1^2) from an independent torch.Generator(seed={TYPE_EMB_SEED}); the shared seed-0 stream is not consumed",
            "trained_v3b_checkpoint_used": False,
        },
        "n_params_v3b": int(sum(p.numel() for p in ref.parameters())),
        "n_params_v3c": int(sum(p.numel() for p in model.parameters())),
        "n_params_added": int(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in ref.parameters())),
        "added_breakdown": {
            "delta_first_layer_new_columns": int(TYPE_EMB_DIM * HIDDEN),
            "type_embedding_table": int(ns.vocab.n_total * TYPE_EMB_DIM),
        },
        "shared_init_hashes": {
            name: {
                "v3b_ref": _hash_tensor(p),
                "v3c": _hash_tensor(q[:, :zin] if name == "delta.0.weight" else q),
            }
            for name, p, q in (
                (n, p, dict(model.named_parameters())[n]) for n, p in ref.named_parameters()
            )
        },
        "shared_init_bit_exact": _shared_bit_exact(),
        "delta_new_columns_all_zero": bool(model.delta[0].weight[:, zin:].abs().sum().item() == 0.0),
        "delta_legacy_columns_bit_exact": bool(
            torch.equal(model.delta[0].weight[:, :zin], ref.delta[0].weight)
        ),
        "alpha_is_per_type_scalar_shape": list(model.alpha.weight.shape),
    }

    # untrained prediction equality vs the v3b-architecture reference head
    # (zero-init of the new columns must make the two heads identical before
    # any training; device float noise tolerated, exact equality expected)
    dev64 = _index_by_row(ns.features, ns.idx["dev"][:64])
    b0 = collate(ns.features, dev64, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)
    ref.eval()
    model.eval()
    with torch.no_grad():
        p_ref = ref(b0)["pred"].numpy().astype(np.float64)
        p_new = model(b0)["pred"].numpy().astype(np.float64)
        d_ref = ref(b0)["delta"].numpy().astype(np.float64)
        d_new = model(b0)["delta"].numpy().astype(np.float64)
    report["untrained_max_abs_pred_diff_vs_v3b_ref"] = float(np.abs(p_new - p_ref).max())
    report["untrained_max_abs_delta_diff_vs_v3b_ref"] = float(np.abs(d_new - d_ref).max())
    report["untrained_pred_exact_equal"] = bool(np.array_equal(p_new, p_ref))
    report["untrained_delta_exact_equal"] = bool(np.array_equal(d_new, d_ref))
    log(
        f"[init] v3b {report['n_params_v3b']} params -> v3c {report['n_params_v3c']} "
        f"(+{report['n_params_added']}: emb {report['added_breakdown']['type_embedding_table']} "
        f"+ W cols {report['added_breakdown']['delta_first_layer_new_columns']}); "
        f"shared bit-exact={report['shared_init_bit_exact']}"
    )
    return ref, model, report


# ---------------------------------------------------------------------------
# training (identical to v3b train_gauge except the head construction)
# ---------------------------------------------------------------------------


def _eval_mae(model, batch: V3Batch, y: np.ndarray) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pred = model(batch)["pred"].cpu().numpy().astype(np.float64)
    return float(np.abs(pred - y).mean()), pred


def train_typecond(ns, device: torch.device, out_dir: Path, seed: int = 0, epochs: int = EPOCHS, log=print):
    """S-typecond-gauge: v3b objective, v3b schedule, type-conditioned delta."""
    assert seed == 0, "this round is seed-0 only"
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    fit_batch = collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    mon_batch = collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    y_fit = ns.y[ns.idx["fit_inner"]]
    y_mon = ns.y[ns.idx["monitor"]]
    y_dev = ns.y[ns.idx["dev"]]

    _ref, model, init_report = build_head_pair(ns, log=log)
    model = model.to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)  # same batch sequence as v3/v3b
    target = torch.as_tensor(y_fit, dtype=torch.float32, device=device)

    curve = []
    ckpts: list[tuple[float, int, dict]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
            out = model(b)
            # v3b objective verbatim: L1 + 1.0 * gamma batch-mean penalty.
            # No delta penalty, no embedding regulariser, no alpha anchoring.
            loss = torch.nn.functional.l1_loss(out["pred"], target[torch.as_tensor(picks)])
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
            log(f"[Stypecond s{seed}] ep={epoch:03d} mon={mon_mae:.5f} {time.perf_counter()-started:.0f}s")
    wall = time.perf_counter() - started

    def evaluate(state):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        f, _ = _eval_mae(model, fit_batch, y_fit)
        m, _ = _eval_mae(model, mon_batch, y_mon)
        d, dpred = _eval_mae(model, dev_batch, y_dev)
        return {"fit_mae": f, "monitor_mae": m, "dev_mae": d}, dpred

    best_state = dict(ckpts[0][2])
    best_epoch = int(ckpts[0][1])
    soup_state = {k: torch.stack([s[k].float() for _, _, s in ckpts]).mean(0) for k in ckpts[0][2]}
    res_best, dev_pred_best = evaluate(best_state)
    res_soup, dev_pred_soup = evaluate(soup_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "S_typecond_s0_best_state.pt")
    write_json(out_dir / "init_report_s0.json", init_report)
    np.savez_compressed(
        out_dir / "S_typecond_s0_dev_preds_raw.npz",
        row=np.asarray(ns.idx["dev"], dtype=np.int64),
        pred=dev_pred_best.astype(np.float32),
        pred_soup=dev_pred_soup.astype(np.float32),
        y=y_dev.astype(np.float32),
    )
    result = {
        "protocol": PROTOCOL_ID,
        "model": "S-typecond-gauge-seed0",
        "seed": seed,
        "git_commit": git_commit(),
        "n_params": n_params,
        "n_params_added_vs_v3b": init_report["n_params_added"],
        "epochs": epochs,
        "wall_seconds": wall,
        "device": str(device),
        "primary_best_monitor": res_best,
        "secondary_soup": res_soup,
        "best_epoch": best_epoch,
        "soup_epochs": sorted(e for _, e, _ in ckpts),
        "curve": curve,
        "objective": {
            "loss": "L1(raw y)",
            "gamma_batch_penalty_coef": PEN_COEF,
            "delta_batch_type_penalty": 0.0,
            "embedding_regularization": "none",
            "alpha_anchoring": "none",
            "distillation": "none",
            "note": "identical to the v3b objective; the only design change is the 16-D type embedding entering delta",
        },
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "hidden": HIDDEN, "optimizer": "Adam", "loss": "L1",
            "shuffle_generator_seed": seed + TRAIN_SHUFFLE_OFFSET,
        },
        "judgment_bands": JUDGMENT_BANDS,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "train_S_typecond_s0.json", result)
    log(f"[Stypecond s{seed}] primary(dev)={res_best['dev_mae']:.5f} soup={res_soup['dev_mae']:.5f} params={n_params} wall={wall:.0f}s")
    return result


# ---------------------------------------------------------------------------
# gauge normalization (reuses contribution_gauge verbatim) + verification
# ---------------------------------------------------------------------------


def gauge_and_verify(ns, device, out_dir: Path, log=print) -> dict:
    """Fit-only offsets on the trained S-typecond checkpoint; all v3b gates;
    canonical dev contributions + predictions."""
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    mon_pos = _index_by_row(ns.features, ns.idx["monitor"])
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    raw_model = TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    raw_model.load_state_dict(torch.load(out_dir / "S_typecond_s0_best_state.pt", map_location="cpu", weights_only=True))
    raw_model = raw_model.to(torch.device(device)).eval()

    offsets, meta = cg.compute_fit_offsets(
        raw_model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device=device, log=log,
    )
    gauged = cg.GaugedHead(raw_model, offsets).to(torch.device(device)).eval()

    batches = {
        "fit": collate(ns.features, fit_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
        "monitor": collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
        "dev": collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device),
    }
    checks: dict[str, Any] = cg.verify_gauge_invariance(raw_model, gauged, batches, log=log)

    with torch.no_grad():
        full = gauged(batches["dev"])
        ha = gauged(collate(ns.features, dev_pos[: len(dev_pos) // 2], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device))
        hb = gauged(collate(ns.features, dev_pos[len(dev_pos) // 2 :], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device))
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
    export = export_gauged_contributions(gauged, ns, dev_pos, device, out_dir / "S_typecond_s0_dev_contribs.npz")
    checks["additivity_export"] = export
    with torch.no_grad():
        pred_dev = gauged(batches["dev"])["pred"].cpu().numpy().astype(np.float64)
    np.savez_compressed(
        out_dir / "S_typecond_s0_dev_preds.npz",
        row=np.asarray(ns.idx["dev"], dtype=np.int64),
        pred=pred_dev.astype(np.float32),
        y=ns.y[ns.idx["dev"]].astype(np.float32),
    )
    write_json(out_dir / "gauge_verification_s0.json", checks)
    return checks


@torch.no_grad()
def export_gauged_contributions(gauged, ns, picks: Sequence[int], device, path: Path) -> dict:
    """Strict-additivity export (canonical AND raw) with unit atom membership
    and the per-unit gauge offset; both identities verified from the exported
    values alone (numpy, independent of the forward pass)."""
    batch = collate(ns.features, list(picks), ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
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
    # atom membership (unit -> atom list) as flat + ptr arrays
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
        arm="S_typecond",
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
# split-half gauge stability of the NEW trained checkpoint (same procedure as
# the Phase A audit of v3b; ONE checkpoint, fit subsamples -- NOT cross-seed)
# ---------------------------------------------------------------------------

MIN_PER_HALF = 10   # pre-frozen reliability threshold (Phase A)
HALF_SEED = 0       # fixed subsample seed (NOT a training seed)


def split_half_gauge_stability(ns, model, device="cpu", log=print) -> dict:
    """Per-type m_t^A vs m_t^B over fixed fit halves + canonical alpha split
    gaps, stratified by type frequency (statistical gauge stability)."""
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    model = model.to(torch.device(device)).eval()

    def per_type(picks):
        with torch.no_grad():
            batch = collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
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
            "max_abs_m_a_minus_m_b": float(np.abs(m_a[sel] - m_b[sel]).max()) if sel.any() else None,
        }

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    out = {
        "question": "statistical stability of the fit-only gauge offsets of ONE trained S-typecond checkpoint across fit subsamples (NOT cross-seed training stability)",
        "half_seed": HALF_SEED,
        "n_mols_half_a": len(half_a), "n_mols_half_b": len(half_b),
        "n_units_half_a": int(cnt_a.sum()), "n_units_half_b": int(cnt_b.sum()),
        "min_per_half_threshold": MIN_PER_HALF,
        "n_types_present_both_halves": int(present.sum()),
        "n_types_reliable_ge_thresh_each_half": int(reliable.sum()),
        "n_types_present_but_below_threshold": int(rare.sum()),
        "n_types_total": ns.vocab.n_total,
        "frac_types_below_threshold_among_present": float(rare.sum() / max(present.sum(), 1)),
        "reliable_types": {
            "mean_abs_m_a_minus_m_b": float(np.abs(m_a[reliable] - m_b[reliable]).mean()),
            "p90_abs_m_a_minus_m_b": float(np.percentile(np.abs(m_a[reliable] - m_b[reliable]), 90)),
            "max_abs_m_a_minus_m_b": float(np.abs(m_a[reliable] - m_b[reliable]).max()),
            "mean_abs_alpha_canonical_split_gap": float(np.abs(alpha_can_a[reliable] - alpha_can_b[reliable]).mean()),
            "p90_abs_alpha_canonical_split_gap": float(np.percentile(np.abs(alpha_can_a[reliable] - alpha_can_b[reliable]), 90)),
            "alpha_ordering_spearman_across_halves": _spearman(alpha_can_a[reliable], alpha_can_b[reliable]),
        },
        "stratified_by_total_fit_count": strat,
        "interpretation_guard": "half-sampling noise of per-type delta means on ONE trained model; does not measure training-seed variability",
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    r = out["reliable_types"]
    log(
        "[split-half] reliable types=%d  mean|m_t^A-m_t^B|=%.4f (p90 %.4f)  alpha split gap mean %.4f  spearman %.4f"
        % (int(reliable.sum()), r["mean_abs_m_a_minus_m_b"], r["p90_abs_m_a_minus_m_b"],
           r["mean_abs_alpha_canonical_split_gap"], r["alpha_ordering_spearman_across_halves"])
    )
    return out


# ---------------------------------------------------------------------------
# analysis: comparison vs frozen F / O / S(v3) / S-gauge(v3b) + audits
# ---------------------------------------------------------------------------


def _load_reference_arms(rows: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    arms: dict[str, np.ndarray] = {}
    avail: dict[str, bool] = {}
    saved_f = np.load(sba.DEV_PREDS_NPZ)
    assert np.array_equal(saved_f["row"], rows)
    arms["F"] = saved_f["pred"].astype(np.float64)
    for name, path in (
        ("O", V3_DIR / "O_s0_dev_preds.npz"),
        ("S_v3", V3_DIR / "S_s0_dev_preds.npz"),
        ("S_gauge", V3B_DIR / "S_gauge_s0_dev_preds.npz"),
    ):
        avail[name] = path.exists()
        if path.exists():
            z = np.load(path)
            assert np.array_equal(z["row"], rows)
            arms[name] = z["pred"].astype(np.float64)
    return arms, avail


@torch.no_grad()
def analyze(ns, out_dir: Path, device, log=print) -> dict:
    """MAEs + paired bootstrap + contribution audit (raw AND canonical) +
    type-conditioned response audit + UNK audit + subgroup table."""
    dev_pos = _index_by_row(ns.features, ns.idx["dev"])
    rows = np.asarray(ns.idx["dev"], dtype=np.int64)
    y_dev = ns.y[rows]

    zg = np.load(out_dir / "S_typecond_s0_dev_preds.npz")
    assert np.array_equal(zg["row"], rows)
    arms, avail = _load_reference_arms(rows)
    arms["S_typecond"] = zg["pred"].astype(np.float64)
    train_res = json.loads((out_dir / "train_S_typecond_s0.json").read_text())
    mae_new = float(np.abs(arms["S_typecond"] - y_dev).mean())

    summary: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "model": "S-typecond-gauge-seed0",
        "reference_files_available": avail,
        "mae": {a: float(np.abs(p - y_dev).mean()) for a, p in arms.items()},
        "mae_S_typecond_fit_monitor_dev": {
            "fit": train_res["primary_best_monitor"]["fit_mae"],
            "monitor": train_res["primary_best_monitor"]["monitor_mae"],
            "dev": train_res["primary_best_monitor"]["dev_mae"],
        },
        "soup_dev_secondary": train_res["secondary_soup"]["dev_mae"],
        "delta_dev_mae_vs": {
            key: mae_new - float(np.abs(arms[ref] - y_dev).mean())
            for key, ref in (("S_gauge", "S_gauge"), ("S_v3", "S_v3"), ("O", "O"), ("F", "F")) if ref in arms
        },
        "judgment_band": (
            "case_A" if mae_new <= 0.16 else ("case_B" if mae_new < 0.17 else "case_C")
        ),
    }
    for a, b in (("S_typecond", "S_gauge"), ("S_typecond", "S_v3"), ("S_typecond", "O"), ("S_typecond", "F")):
        if b in arms:
            summary[f"paired_{a}_minus_{b}"] = paired_bootstrap(np.abs(arms[a] - y_dev), np.abs(arms[b] - y_dev))

    dataset = cf.load_official_train()
    summary["subgroups"] = subgroup_table_v3b(rows, y_dev, arms, ns, dataset)

    # ---- contribution audit: canonical AND raw (from the export npz) ----
    z = np.load(out_dir / "S_typecond_s0_dev_contribs.npz")
    offsets = np.asarray(torch.load(out_dir / "gauge_offsets_s0.pt", map_location="cpu", weights_only=False)["offsets"])
    ut, um = z["unit_type"], z["unit_mol"]
    n_mols = len(rows)
    fit_counts: dict[int, int] = {}
    for fi in _index_by_row(ns.features, ns.idx["fit_inner"]):
        for t in ns.features[fi].type_ids(ns.vocab):
            fit_counts[int(t)] = fit_counts.get(int(t), 0) + 1

    def channels(a_key, d_key):
        unit = np.zeros(n_mols)
        np.add.at(unit, um, z[a_key] + z[d_key])
        rel = np.zeros(n_mols)
        if len(z["gamma"]):
            np.add.at(rel, z["rel_mol"], z["gamma"])
        return {"unit": unit, "rel": rel, "g_ctx": z["g_ctx"], "g_topo": z["g_topo"]}

    audit: dict[str, Any] = {}
    for tag, a_key, d_key in (("canonical", "alpha", "delta"), ("raw", "alpha_raw", "delta_raw")):
        ch = channels(a_key, d_key)
        pred = z["pred"]
        audit[f"channel_var_shares_{tag}"] = {k: float(np.var(v) / max(np.var(pred), 1e-12)) for k, v in ch.items()}
        C = np.stack([ch[k] for k in ("unit", "rel", "g_ctx", "g_topo")])
        audit[f"channel_corr_matrix_{tag}"] = np.corrcoef(C).tolist()
        audit[f"delta_{tag}_dev"] = {"mean": float(z[d_key].mean()), "std": float(z[d_key].std())}
    audit["gamma"] = {
        "mean": float(z["gamma"].mean()) if len(z["gamma"]) else 0.0,
        "std": float(z["gamma"].std()) if len(z["gamma"]) else 0.0,
        "frac_abs_le_0.01": float((np.abs(z["gamma"]) <= 0.01).mean()) if len(z["gamma"]) else 1.0,
    }

    # m_t / alpha behaviour (canonical alpha is convention-defined, not a
    # context-independent chemical constant -- §5.3 boundary)
    present = np.array([fit_counts.get(t, 0) for t in range(len(offsets))]) > 0
    audit["m_t_distribution"] = {
        "n_types_with_fit_instances": int(present.sum()),
        "mean_abs_m_t": float(np.abs(offsets[present]).mean()),
        "p50_abs": float(np.percentile(np.abs(offsets[present]), 50)),
        "p90_abs": float(np.percentile(np.abs(offsets[present]), 90)),
        "max_abs": float(np.abs(offsets[present]).max()),
        "n_zero_fit_types_fallback_0": int((~present).sum()),
    }
    freq_ids = sorted(fit_counts, key=lambda t: -fit_counts[t])[:20]
    a_can_by_type, a_raw_by_type = {}, {}
    for t in np.unique(ut):
        m = ut == t
        a_raw_by_type[int(t)] = float(z["alpha_raw"][m][0])
        a_can_by_type[int(t)] = float(z["alpha"][m][0])
    audit["alpha_top20_frequent_types"] = {
        str(int(t)): {
            "fit_count": fit_counts[t],
            "alpha_raw": a_raw_by_type.get(int(t)),
            "alpha_canonical": a_can_by_type.get(int(t)),
        }
        for t in freq_ids if int(t) in a_can_by_type
    }
    known_ids = np.asarray(sorted(fit_counts.keys()))
    audit["alpha_summary"] = {
        "raw_std_across_fit_known_types": float(np.std([a_raw_by_type.get(int(t), np.nan) for t in known_ids if int(t) in a_raw_by_type])),
        "canonical_std_across_dev_types": float(np.std(list(a_can_by_type.values()))),
        "canonical_value_range": [float(min(a_can_by_type.values())), float(max(a_can_by_type.values()))],
    }

    # per-type canonical delta on dev: within-type variance + attach-count
    # response slopes (type-conditioned contextual response check)
    attach_by_unit = []
    for fi in dev_pos:
        m = ns.features[fi]
        attach_by_unit.extend(attach_counts(m).tolist())
    attach_by_unit = np.asarray(attach_by_unit)
    assert len(attach_by_unit) == len(ut)
    per_type = {}
    for t in np.unique(ut):
        m = ut == t
        n = int(m.sum())
        if n < 5 or fit_counts.get(int(t), 0) == 0:
            continue
        d_can = z["delta"][m]
        entry = {
            "n_dev_units": n,
            "delta_canonical_std": float(d_can.std()) if n > 1 else 0.0,
            "delta_canonical_mean": float(d_can.mean()),
            "fit_count": fit_counts[int(t)],
        }
        if n >= 10 and np.std(attach_by_unit[m]) > 1e-9:
            entry["delta_vs_attach_slope"] = float(np.polyfit(attach_by_unit[m], d_can, 1)[0])
        per_type[int(t)] = entry
    top = sorted(per_type, key=lambda t: -per_type[t]["fit_count"])[:20]
    slopes = [per_type[t]["delta_vs_attach_slope"] for t in top if "delta_vs_attach_slope" in per_type[t]]
    audit["delta_per_type_top20"] = {str(int(t)): per_type[t] for t in top}
    audit["delta_attach_slope_dispersion_top20"] = {
        "n_types_with_slope": len(slopes),
        "slope_std_across_types": float(np.std(slopes)) if len(slopes) > 1 else None,
        "slope_min": float(np.min(slopes)) if slopes else None,
        "slope_max": float(np.max(slopes)) if slopes else None,
        "note": "the synthetic truth uses a TYPE-INDEPENDENT slope (0.8); slope dispersion on real ZINC shows how much type-conditioning the model actually uses",
    }
    audit["delta_within_type_std_instance_weighted"] = float(np.average(
        [per_type[t]["delta_canonical_std"] for t in per_type],
        weights=[per_type[t]["n_dev_units"] for t in per_type],
    ))

    # UNK audit: units whose signature is unknown (UNK buckets)
    known_unit = np.concatenate([
        np.asarray([ns.vocab.is_known(s) for s in ns.features[fi].unit_sigs])
        for fi in dev_pos
    ])
    abs_unit_contrib = np.abs(z["alpha"] + z["delta"])
    mol_has_unk = np.zeros(n_mols, dtype=bool)
    np.add.at(mol_has_unk, um[~known_unit], 1)
    err = np.abs(arms["S_typecond"] - y_dev)
    audit["unk_audit"] = {
        "n_unk_units_dev": int((~known_unit).sum()),
        "mean_abs_canonical_unit_contrib_unk": float(abs_unit_contrib[~known_unit].mean()) if (~known_unit).any() else None,
        "mean_abs_canonical_unit_contrib_known": float(abs_unit_contrib[known_unit].mean()),
        "mae_molecules_with_unk": float(err[mol_has_unk].mean()) if mol_has_unk.any() else None,
        "mae_molecules_without_unk": float(err[~mol_has_unk].mean()),
        "n_molecules_with_unk": int(mol_has_unk.sum()),
    }
    summary["S_typecond_audit"] = audit
    write_json(out_dir / "analysis_s0.json", summary)
    log("[analyze] " + json.dumps(summary["mae"]))
    return summary


# ---------------------------------------------------------------------------
# corrected synthetic known-truth task (CPU, seed 0) -- SAME generator as v3b
# ---------------------------------------------------------------------------


def _hash_value(payload: str) -> float:
    h = hashlib.sha256(payload.encode()).hexdigest()
    return (int(h[:8], 16) % 2000) / 1000.0 - 1.0


def synthetic_task_typecond(ns, out_dir: Path, device, epochs: int = EPOCHS, log=print) -> dict:
    """y_syn = sum_u alpha*(t) + 0.8*(attach - fit type mean) + sum_pairs gamma* + noise.

    Identical truth generator / data / seeds as the v3b corrected synthetic
    (`run_cscl_v3b.synthetic_task_gauge`); only the head differs (the 16-D
    type-conditioned delta).  Numbers are therefore SAME-PROTOCOL comparable
    to results/cscl_v3b/synthetic_gauge_s0.json (mechanism comparison), while
    both remain not comparable to the v3 synthetic (older truth definition).
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

    def syn_labels(picks):
        ys = []
        for fi in picks:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            ac = attach_counts(m)
            total = 0.0
            for k, t in enumerate(tids):
                total += alpha_star.get(int(t), 0.0)
                total += 0.8 * (float(ac[k]) - attach_mean.get(int(t), global_attach))
            for lo, hi in zip(m.rel_lo, m.rel_hi):
                total += gamma_star.get((int(tids[lo]), int(tids[hi])), 0.0)
            ys.append(total)
        return np.asarray(ys, dtype=np.float64)

    rng = np.random.default_rng(0)
    y_syn_fit = syn_labels(fit_pos) + 0.05 * rng.standard_normal(len(fit_pos))
    y_syn_mon = syn_labels(mon_pos)
    y_syn_dev = syn_labels(dev_pos)

    seed_everything(0)
    _ref, model, _ = build_head_pair(ns, log=lambda *a, **k: None)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(101)
    mon_batch = collate(ns.features, mon_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    dev_batch = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
    target = torch.as_tensor(y_syn_fit, dtype=torch.float32, device=device)
    best = (1e9, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        for picks in _batches(len(fit_pos), gen, True):
            b = collate(ns.features, [fit_pos[i] for i in picks], ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
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

    with torch.no_grad():
        out = model(dev_batch)
    alpha_hat = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    delta_hat = out["delta"].cpu().numpy().astype(np.float64)
    gamma_hat = out["gamma"].cpu().numpy().astype(np.float64)

    offsets, meta = cg.compute_fit_offsets(
        model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device=device, log=log,
    )
    gauged = cg.GaugedHead(model, offsets).to(torch.device(device)).eval()
    with torch.no_grad():
        gout = gauged(dev_batch)
        pred_raw_syn = out["pred"].cpu().numpy().astype(np.float64)
        pred_can_syn = gout["pred"].cpu().numpy().astype(np.float64)
    gauge_pred_gap = float(np.abs(pred_raw_syn - pred_can_syn).max())
    delta_can = gout["delta"].cpu().numpy().astype(np.float64)
    alpha_can = gout["alpha"].cpu().numpy().astype(np.float64)

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    a_true = np.array([alpha_star[t] for t in top32])
    a_hat = np.array([alpha_hat[t] for t in top32])
    sign_match = float((np.sign(a_true) == np.sign(a_hat)).mean())

    true_delta, hat_pairs, alpha_of_unit = [], [], []
    um = dev_batch.unit_mol.cpu().numpy()
    for mi, fi in enumerate(dev_pos):
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        ac = attach_counts(m)
        hats = delta_hat[um == mi]
        hats_can = delta_can[um == mi]
        assert len(hats) == m.n_units
        for k, t in enumerate(tids):
            true_delta.append(0.8 * (float(ac[k]) - attach_mean.get(int(t), global_attach)))
            hat_pairs.append((float(hats[k]), float(hats_can[k])))
            alpha_of_unit.append(alpha_star.get(int(t), 0.0))
    true_delta = np.asarray(true_delta)
    hat_delta_raw = np.asarray([h[0] for h in hat_pairs])
    hat_delta_can = np.asarray([h[1] for h in hat_pairs])
    alpha_of_unit = np.asarray(alpha_of_unit)

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
    unseen_hat, seen_hat = [], []
    idx_rel = 0
    for fi in dev_pos:
        m = ns.features[fi]
        tids = m.type_ids(ns.vocab)
        for j in range(len(m.rel_lo)):
            key = (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]]))
            (unseen_hat if key not in rel_type_count else seen_hat).append(float(gamma_hat[idx_rel + j]))
        idx_rel += len(m.rel_lo)

    def alpha_gamma_confusion(a_vec_by_type):
        per_type_gamma: dict[int, list[float]] = {}
        idx_rel = 0
        for fi in dev_pos:
            m = ns.features[fi]
            tids = m.type_ids(ns.vocab)
            for j in range(len(m.rel_lo)):
                for tt in (int(tids[m.rel_lo[j]]), int(tids[m.rel_hi[j]])):
                    per_type_gamma.setdefault(tt, []).append(float(gamma_hat[idx_rel + j]))
            idx_rel += len(m.rel_lo)
        common = [t for t in per_type_gamma if t in a_vec_by_type]
        if len(common) < 3:
            return None
        xs = np.array([a_vec_by_type[t] for t in common])
        gs = np.array([np.mean(per_type_gamma[t]) for t in common])
        return float(np.corrcoef(xs, gs)[0, 1])

    dev_ut = dev_batch.unit_type.cpu().numpy()
    a_can_by_type = {int(t): float(alpha_can[dev_ut == t][0]) for t in np.unique(dev_ut)}
    result = {
        "protocol": PROTOCOL_ID,
        "task": "corrected synthetic known-truth on real frozen features (attach idx D_HIST+4=36; gamma-only penalty; noise 0.05; seed 0)",
        "attachment_index": cf.D_HIST + 4,
        "comparability": "same truth generator/data/seeds as results/cscl_v3b/synthetic_gauge_s0.json (only the head differs) -> same-protocol comparison allowed; NOT comparable to the v3 synthetic",
        "best_monitor_epoch": int(best[1]),
        "monitor_mae_syn": float(best[0]),
        "dev_mae_syn_raw": float(np.abs(pred_raw_syn - y_syn_dev).mean()),
        "dev_mae_syn_canonical": float(np.abs(pred_can_syn - y_syn_dev).mean()),
        "gauge_max_abs_pred_diff_synthetic": gauge_pred_gap,
        "alpha_sign_recovery_top32": sign_match,
        "alpha_spearman_top32_raw": _spearman(a_true, a_hat),
        "alpha_true_value_range": [float(a_true.min()), float(a_true.max())],
        "delta_recovery_pearson_raw": float(np.corrcoef(true_delta, hat_delta_raw)[0, 1]),
        "delta_recovery_pearson_canonical": float(np.corrcoef(true_delta, hat_delta_can)[0, 1]),
        "delta_hat_vs_alpha_star_confusion_raw": float(np.corrcoef(hat_delta_raw, alpha_of_unit)[0, 1]),
        "delta_hat_vs_alpha_star_confusion_canonical": float(np.corrcoef(hat_delta_can, alpha_of_unit)[0, 1]),
        "delta_amp_ratio_raw_vs_truth": float(hat_delta_raw.std() / max(np.abs(true_delta).std(), 1e-12)),
        "alpha_gamma_confusion_corr_canonical": alpha_gamma_confusion(a_can_by_type),
        "gamma_checks": gamma_check,
        "unseen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(unseen_hat))) if unseen_hat else None,
        "seen_pair_mean_abs_gamma_hat": float(np.mean(np.abs(seen_hat))) if seen_hat else None,
        "n_unseen_dev_relations": len(unseen_hat),
        "gauge_meta_fit_units": meta["n_fit_units"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "synthetic_typecond_s0.json", result)
    log("[synthetic-typecond] alpha sign %.2f spearman %.2f delta-r(raw) %.2f delta-r(can) %.2f gauge-pred-gap %.2e"
        % (sign_match, result["alpha_spearman_top32_raw"], result["delta_recovery_pearson_raw"],
           result["delta_recovery_pearson_canonical"], gauge_pred_gap))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("audit-init", "train", "gauge", "analyze", "splithalf", "synthetic", "all"))
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
        _ref, model, report = build_head_pair(ns, log=log)
        write_json(args.out / "init_report_s0.json", report)
        log(f"[audit-init] untrained pred diff {report['untrained_max_abs_pred_diff_vs_v3b_ref']:.3e} (exact={report['untrained_pred_exact_equal']})")
        return 0
    if args.mode == "train":
        ns = prepare_all(log=log)
        train_typecond(ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
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
        model = TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        model.load_state_dict(torch.load(args.out / "S_typecond_s0_best_state.pt", map_location="cpu", weights_only=True))
        res = split_half_gauge_stability(ns, model, device="cpu", log=log)
        write_json(args.out / "split_half_gauge_stability_s0.json", res)
        return 0
    if args.mode == "synthetic":
        ns = prepare_all(log=log)
        synthetic_task_typecond(ns, args.out, torch.device("cpu"), log=log)
        return 0
    if args.mode == "all":
        log("[all] train S-typecond (seed 0) ...")
        ns = prepare_all(log=log, device=str(device))
        train_typecond(ns, device, args.out, seed=args.seed, epochs=args.epochs, log=log)
        log("[all] fit-only gauge + verification ...")
        gauge_and_verify(ns, device, args.out, log=log)
        log("[all] analyze vs frozen F/O/S(v3)/S-gauge(v3b) ...")
        analyze(ns, args.out, device, log=log)
        log("[all] split-half gauge stability ...")
        model = TypeCondContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
        model.load_state_dict(torch.load(args.out / "S_typecond_s0_best_state.pt", map_location="cpu", weights_only=True))
        write_json(args.out / "split_half_gauge_stability_s0.json", split_half_gauge_stability(ns, model, device="cpu", log=log))
        log("[all] corrected synthetic known-truth (CPU, seed 0) ...")
        synthetic_task_typecond(ns, args.out, torch.device("cpu"), log=log)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
