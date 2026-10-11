"""cscl-v3c Phase A CPU audits (no training, no GPU):

A.1  Exact type-access / parameterization audit of v3b S-gauge vs O (from the
     real modules + trained checkpoints, not from config descriptions).
A.2  Split-half gauge-normalization stability of the TRAINED v3b S-gauge
     checkpoint: fit_inner split into two halves by a fixed seed; per-type
     m_t^A / m_t^B vs m_t^all; canonical-alpha split differences; stratified
     by type occurrence count.  This is statistical stability of the gauge
     offsets of ONE trained checkpoint across fit subsamples -- NOT cross-seed
     training stability.

Label-free: uses model outputs and fit structure only; official valid/test
never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contribution_gauge as cg  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
from run_cscl_v3 import (  # noqa: E402
    OpaqueHead,
    SBContribHead,
    ZDIM,
    _index_by_row,
    collate,
    prepare_all,
)

OUT_PATH = REPO_ROOT / "tracks/structure-contribution/results/cscl_v3c/phase_a_type_access_and_gauge_stability_s0.json"

#: pre-frozen reliability threshold for the split-half analysis (task 3.2):
#: a type is "frequent enough" only if it has >= 10 unit instances in EACH half
MIN_PER_HALF = 10
#: fixed seed for the fit half split (NOT a training seed)
HALF_SEED = 0


def _state_hash(sd) -> dict:
    import hashlib

    return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()[:16] for k, v in sd.items()}


# ---------------------------------------------------------------------------
# A.1  type access / parameterization of S-gauge vs O (real code + checkpoints)
# ---------------------------------------------------------------------------


@torch.no_grad()
def type_access_audit(ns, out: dict) -> None:
    v3b_state = torch.load(
        REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b/S_gauge_s0_best_state.pt",
        map_location="cpu", weights_only=True,
    )
    o_state = torch.load(
        REPO_ROOT / "tracks/structure-contribution/results/cscl_v3/O_s0_best_state.pt",
        map_location="cpu", weights_only=True,
    )
    s = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    s.load_state_dict(v3b_state, strict=True)
    o = OpaqueHead(ns.vocab.n_total)
    o.load_state_dict(o_state, strict=True)
    s.eval()
    o.eval()

    # ---- module inventory (shapes from the actual state dicts) ----
    def inventory(model):
        inv = {}
        for name, p in model.named_parameters():
            inv[name] = {"shape": list(p.shape), "numel": int(p.numel())}
        return inv

    inv_s, inv_o = inventory(s), inventory(o)

    # ---- forward-path information flow, verified on a real dev batch ----
    dev_pos = _index_by_row(ns.features, ns.idx["dev"][:16])
    b = collate(ns.features, dev_pos, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean)

    # S: alpha depends ONLY on unit_type; delta ONLY on [feat, ctx]
    t1 = b.unit_type.clone()
    delta_in = torch.cat([b.unit_feat, b.unit_ctx], dim=1)
    # NOTE (Phase A documentation finding): the v3/v3b reports and the v3c task
    # brief state a "36-D context descriptor / 420-D delta input", but the
    # actual code (cscl_features.DESC_DIM = D_HIST 32 + D_SCALARS 8 = 40) gives
    # a 40-D descriptor and a 424-D delta input; param count 118,168 pins this.
    assert delta_in.shape[1] == ZDIM + sba.DESC_DIM == 384 + 40 == 424  # NO type channel
    a1 = s.alpha(t1).squeeze(-1)
    t2 = t1.clone()
    torch.manual_seed(0)
    t2[torch.randperm(len(t2), generator=torch.Generator().manual_seed(1))[:5]] = (
        (t2[torch.randperm(len(t2), generator=torch.Generator().manual_seed(2))[:5]] + 7) % ns.vocab.n_total
    )
    a2 = s.alpha(t2).squeeze(-1)
    alpha_depends_on_type_only = bool(torch.equal(a1[t1 == t2], a2[t1 == t2])) and not torch.equal(a1, a2)
    d1 = s.delta(delta_in).squeeze(-1)
    # shuffle type ids -> delta must be unchanged (no type input)
    perm = torch.randperm(len(b.unit_type), generator=torch.Generator().manual_seed(3))
    b_perm_types = type(b)(
        b.unit_mol, b.unit_type[perm], b.unit_feat, b.unit_ctx, b.rel_mol, b.rel_feat, b.graph_feat
    )
    d2 = s.delta(torch.cat([b_perm_types.unit_feat, b_perm_types.unit_ctx], dim=1)).squeeze(-1)
    # rows with identical (feat, ctx) but different type must have identical delta
    delta_depends_only_on_feat_ctx = bool(torch.equal(d1, d2))

    # O: unit encoder consumes the 16-D type embedding
    u_in_dim = o.unit_enc[0].weight.shape[1]
    emb = o.type_emb(b.unit_type)
    o_in = torch.cat([b.unit_feat, b.unit_ctx, emb], dim=1)
    assert o_in.shape[1] == u_in_dim == 384 + 40 + 16 == 440
    e1 = o.unit_enc(o_in)
    o_perm = type(b)(
        b.unit_mol, b.unit_type[perm], b.unit_feat, b.unit_ctx, b.rel_mol, b.rel_feat, b.graph_feat
    )
    e2 = o.unit_enc(torch.cat([o_perm.unit_feat, o_perm.unit_ctx, o.type_emb(o_perm.unit_type)], dim=1))
    same_feat_ctx_rows = (b.unit_feat == o_perm.unit_feat).all(1) & (b.unit_ctx == o_perm.unit_ctx).all(1)
    type_changed = b.unit_type != o_perm.unit_type
    probe = same_feat_ctx_rows & type_changed
    o_unit_enc_reads_type = bool((e1[probe] - e2[probe]).abs().max() > 0) if probe.any() else None

    # gamma input: identical for S and O; contains z sums/diffs/products + pair
    # moments + bucket/bond histograms + centered relation means (NO type id,
    # NO type embedding in either head)
    relin = s.gamma[0].weight.shape[1]
    assert relin == o.rel_enc[0].weight.shape[1] == 1273
    assert not any("type" in k for k in inv_s if k.startswith("gamma"))
    assert not any("type" in k for k in inv_o if k.startswith("rel_enc"))

    # global channels: O's glob_enc reads the same 55 = 30 ctx + 25 topo graph
    # features; S reads them through the per-feature hinge banks on the same
    # graph_feat columns (S: additive per-feature bank; O: opaque MLP)
    assert o.glob_enc[0].weight.shape[1] == sba.CTX_STRUCTURE + sba.TOPO_DIM == 55
    assert inv_s["w_ctx"]["shape"] == [sba.CTX_STRUCTURE, 1 + 3]
    assert inv_s["w_topo"]["shape"] == [sba.TOPO_DIM, 1 + 3]

    out["type_access"] = {
        "S_gauge": {
            "alpha": "nn.Embedding(vocab_size, 1) indexed by unit_type; one shared scalar per type",
            "alpha_param_shape": inv_s["alpha.weight"],
            "alpha_depends_on_type_only": alpha_depends_on_type_only,
            "delta_input": "[unit_feat 384 | unit_ctx 40] = 424; NO type id / NO type embedding (docs' '36-D/420' is a recording error; code + param count authoritative)",
            "delta_input_dim": int(delta_in.shape[1]),
            "delta_unchanged_under_type_shuffle": delta_depends_only_on_feat_ctx,
            "delta_params": {k: v for k, v in inv_s.items() if k.startswith("delta")},
            "gamma_input_dim": int(relin),
            "gamma_type_inputs": "none (z sums/diffs/products, pair moments, bucket/bond hist, log1p count, centered relation means)",
            "type_embedding_params": "none",
        },
        "O": {
            "type_embedding": "nn.Embedding(vocab_size, 16) -> unit encoder input",
            "type_emb_param_shape": inv_o["type_emb.weight"],
            "unit_encoder_input_dim": int(u_in_dim),  # 384 + 40 + 16 = 440 (docs' 420/436 inherit the same recording error)
            "unit_encoder_reads_type_verified": o_unit_enc_reads_type,
            "unit_encoder_params": {k: v for k, v in inv_o.items() if k.startswith("unit_enc")},
            "gamma_input_dim": int(o.rel_enc[0].weight.shape[1]),
            "gamma_type_inputs": "none (same 1273-D rows as S)",
            "head": "DeepSets mean+sum pooling over 5x16 slots + MLP (no additive constraint)",
        },
        "shared_inputs": {
            "unit_feat": "z_u = [mean E, rms E, mean pv, rms pv] (384-D), identical collate",
            "unit_ctx": "36-D descriptor, fit-only per-type mean centered, identical collate",
            "relation_rows": "identical collate (same rel_feat builder in run_cscl_v3.collate)",
            "global": "graph_feat = [ctx_structure 30 | topology 25] for both heads",
        },
        "n_params": {"S_gauge": int(sum(p.numel() for p in s.parameters())), "O": int(sum(p.numel() for p in o.parameters()))},
        "vocab": {"n_known": ns.vocab.n_known, "n_total": ns.vocab.n_total},
        "param_hash_s_gauge_trained": _state_hash(v3b_state),
    }


# ---------------------------------------------------------------------------
# A.2  split-half gauge stability of the TRAINED v3b S-gauge (label-free)
# ---------------------------------------------------------------------------


@torch.no_grad()
def gauge_split_half(ns, out: dict, device="cpu", log=print) -> None:
    fit_pos = _index_by_row(ns.features, ns.idx["fit_inner"])
    model = SBContribHead(ns.vocab.n_total, ns.tau_ctx, ns.tau_topo)
    model.load_state_dict(
        torch.load(
            REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b/S_gauge_s0_best_state.pt",
            map_location="cpu", weights_only=True,
        )
    )
    model = model.to(torch.device(device)).eval()

    # fit-only offsets on the FULL fit (reference m_t^all, = the shipped gauge)
    offsets_all, meta_all = cg.compute_fit_offsets(
        model, ns.features, ns.idx["fit_inner"], fit_pos, ns.vocab,
        ns.type_means, ns.global_mean, ns.rel_mean, device=device, log=log,
    )

    # fixed-seed half split of the 7200 fit MOLECULES (molecule-level halves,
    # so both halves are ordinary fit subsamples; unit instances follow)
    rng = np.random.default_rng(HALF_SEED)
    order = rng.permutation(len(fit_pos))
    half_a = [fit_pos[i] for i in order[: len(fit_pos) // 2]]
    half_b = [fit_pos[i] for i in order[len(fit_pos) // 2 :]]

    def per_type(picks):
        batch = collate(ns.features, picks, ns.vocab, ns.type_means, ns.global_mean, ns.rel_mean).to(device)
        delta = model(batch)["delta"].detach().cpu().numpy().astype(np.float64)
        ut = batch.unit_type.cpu().numpy()
        sums = np.zeros(ns.vocab.n_total)
        counts = np.zeros(ns.vocab.n_total, dtype=np.int64)
        np.add.at(sums, ut, delta)
        np.add.at(counts, ut, 1)
        return sums, counts, ut

    sums_a, cnt_a, _ = per_type(half_a)
    sums_b, cnt_b, _ = per_type(half_b)
    m_a = np.where(cnt_a > 0, sums_a / np.maximum(cnt_a, 1), 0.0)
    m_b = np.where(cnt_b > 0, sums_b / np.maximum(cnt_b, 1), 0.0)

    # raw alpha table of the trained model
    alpha_raw = model.alpha.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    alpha_can_a = alpha_raw + np.where(cnt_a > 0, m_a, 0.0)
    alpha_can_b = alpha_raw + np.where(cnt_b > 0, m_b, 0.0)

    present = (cnt_a > 0) & (cnt_b > 0)
    both_cnt = cnt_a + cnt_b
    reliable = present & (cnt_a >= MIN_PER_HALF) & (cnt_b >= MIN_PER_HALF)
    rare = present & ~reliable

    def _sd(x):
        return float(np.std(x)) if len(x) else 0.0

    strat: dict[str, dict] = {}
    bins = (("count_lt_20", both_cnt < 20), ("count_20_99", (both_cnt >= 20) & (both_cnt < 100)),
            ("count_100_499", (both_cnt >= 100) & (both_cnt < 500)), ("count_ge_500", both_cnt >= 500))
    for name, mask in bins:
        sel = present & mask
        strat[name] = {
            "n_types": int(sel.sum()),
            "mean_abs_m_a_minus_m_b": float(np.abs(m_a[sel] - m_b[sel]).mean()) if sel.any() else None,
            "p90_abs_m_a_minus_m_b": float(np.percentile(np.abs(m_a[sel] - m_b[sel]), 90)) if sel.sum() >= 5 else None,
            "max_abs_m_a_minus_m_b": float(np.abs(m_a[sel] - m_b[sel]).max()) if sel.any() else None,
            "mean_abs_alpha_can_split_gap": float(np.abs(alpha_can_a[sel] - alpha_can_b[sel]).mean()) if sel.any() else None,
        }

    # alpha ordering stability across the split (Spearman over reliable types)
    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    alpha_order_spearman_reliable = _spearman(alpha_can_a[reliable], alpha_can_b[reliable])

    out["gauge_split_half_stability"] = {
        "question": "statistical stability of the fit-only gauge offsets of ONE trained v3b checkpoint across fit subsamples (NOT cross-seed training stability)",
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
            "alpha_ordering_spearman_across_halves": alpha_order_spearman_reliable,
        },
        "stratified_by_total_fit_count": strat,
        "full_fit_reference": {
            "m_t_abs_mean": float(np.abs(offsets_all[present | (cnt_a > 0) | (cnt_b > 0)]).mean()),
            "shipped_gauge_device_check": {
                "note": "the shipped gauge offsets were computed on the GPU formal run; this audit recomputes them on CPU, so exact equality is not expected -- only device-level float32 forward noise (~1e-7)",
                "max_abs_cpu_vs_shipped_gpu": float(np.abs(
                    offsets_all - np.asarray(torch.load(
                        REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b/gauge_offsets_s0.pt",
                        map_location="cpu", weights_only=False)["offsets"], dtype=np.float64)
                ).max()),
                "atol": cg.GAUGE_ATOL,
                "ok": bool(np.abs(
                    offsets_all - np.asarray(torch.load(
                        REPO_ROOT / "tracks/structure-contribution/results/cscl_v3b/gauge_offsets_s0.pt",
                        map_location="cpu", weights_only=False)["offsets"], dtype=np.float64)
                ).max() <= cg.GAUGE_ATOL),
            },
        },
        "interpretation_guard": "m_t^A vs m_t^B differences quantify sampling noise of the per-type delta mean on ONE trained model; they do not measure training-seed variability",
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    r = out["gauge_split_half_stability"]["reliable_types"]
    log(
        "[split-half] reliable types=%d  mean|m_t^A-m_t^B|=%.4f (p90 %.4f)  alpha split gap mean %.4f  spearman %.4f"
        % (int(reliable.sum()), r["mean_abs_m_a_minus_m_b"], r["p90_abs_m_a_minus_m_b"],
           r["mean_abs_alpha_canonical_split_gap"], r["alpha_ordering_spearman_across_halves"])
    )


def main() -> int:
    torch.set_num_threads(8)
    log = lambda *a, **k: print(*a, **k, flush=True)  # noqa: E731
    out: dict = {"protocol": "cscl-v3c-type-conditioned-gauge", "phase": "A CPU audits"}
    ns = prepare_all(log=log)
    type_access_audit(ns, out)
    log("[audit] type-access table verified from real modules + trained checkpoints")
    gauge_split_half(ns, out, device="cpu", log=log)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[audit] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
