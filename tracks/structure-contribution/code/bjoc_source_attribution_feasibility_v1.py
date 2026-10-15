"""bjoc-source-attribution-feasibility-v1 — frozen-model junction source attribution.

Namespace: ``bjoc_source_attribution_feasibility_v1`` (track ``structure-contribution``).
Frozen protocol: ``results/bjoc_source_attribution_feasibility_v1/PROTOCOL.md``.

Subcommands: erratum / prepare / synthetic / attribute / finalize.

CPU-only, label-blind, ``eval()`` weights frozen. The only gradients are input
gradients of the frozen prediction used by the attribution method itself.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_structure_explanation_audit_v1 as v1  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

NAMESPACE = "bjoc_source_attribution_feasibility_v1"
OUT = REPO_ROOT / "tracks/structure-contribution/results" / NAMESPACE
V1_DIR = REPO_ROOT / "tracks/structure-contribution/results/bjoc_structure_explanation_audit_v1"

SEEDS = (0, 1)
IG_STEPS = 32
IG_CHUNK = 12
TOP_K = 4
RANDOM_DRAWS = 64
SUB_ALPHAS = (0.5, 1.0)
SYNTH_SMOKE = 16
SPEARMAN_TOP_THRESHOLD = 0.5
BASELINE_STABILITY_THRESHOLD = 0.7
ALPHA_STABILITY_THRESHOLD = 0.7
IG_OCC_SPEARMAN_THRESHOLD = 0.8
POINTWISE_REL_THRESHOLD = 0.05  # replaces the mis-specified COMPLETENESS_REL_THRESHOLD (see PROTOCOL.md R1)
COMPLETENESS_REL_THRESHOLD = 0.10  # original pre-registration, mathematically wrong reference for one-at-a-time IG; retained for the record


def log(msg: str) -> None:
    print(msg, flush=True)


def mean_abs(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(np.abs(a).mean()) if a.size else float("nan")


def exact_binomial_p(k: int, n: int) -> float:
    """Two-sided exact sign-test p-value for k successes in n paired trials."""
    if n == 0:
        return float("nan")
    k, n = int(k), int(n)
    lo = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, lo + 1))
    log_p = math.log(2 * tail) - n * math.log(2.0)
    return float(min(1.0, math.exp(min(0.0, log_p))))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def sign_test(deltas: np.ndarray) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = len(d)
    k = int((d > 0).sum())
    return {
        "n": n,
        "n_positive": k,
        "n_negative": int((d < 0).sum()),
        "mean": float(d.mean()) if n else float("nan"),
        "median": float(np.median(d)) if n else float("nan"),
        "frac_positive": float(k / n) if n else float("nan"),
        "p_two_sided_exact": exact_binomial_p(k, n),
    }


# ---------------------------------------------------------------------------
# per-molecule pack + Null/Organized pipeline helpers
# ---------------------------------------------------------------------------


def pack_from_features(
    model: src.BJOC,
    batch: Mapping[str, torch.Tensor],
    feats: Mapping[str, torch.Tensor],
    g: int,
) -> dict[str, Any]:
    atom_mask = batch["atom_batch"] == g
    bond_mask = batch["bond_batch"] == g
    junc_mask = batch["junc_batch"] == g
    n_a = int(atom_mask.sum())
    n_b = int(bond_mask.sum())
    n_j = int(junc_mask.sum())
    a_off = int(torch.nonzero(atom_mask)[0, 0])
    b_off = int(torch.nonzero(bond_mask)[0, 0])
    j_off = int(torch.nonzero(junc_mask)[0, 0])
    return {
        "a": feats["a"][a_off : a_off + n_a],
        "b": feats["b"][b_off : b_off + n_b],
        "w": feats["w"][j_off : j_off + n_j],
        "a_sum": feats["a"][a_off : a_off + n_a].sum(0),
        "a_sq": (feats["a"][a_off : a_off + n_a] ** 2).sum(0),
        "g": feats["reader_in"][g, 288:],
        "n_b": feats["n_b"][b_off : b_off + n_b],
        "bond_feat": batch["bond_feat"][b_off : b_off + n_b],
        "junc_index": batch["junc_index"][j_off : j_off + n_j],
        "junc_bond": batch["junc_bond"][j_off : j_off + n_j] - b_off,
        "n_atoms": n_a,
        "n_bonds": n_b,
        "n_junctions": n_j,
    }


def _reader_from_components(model: src.BJOC, pack: Mapping[str, Any], W: torch.Tensor, V: torch.Tensor, n_b: torch.Tensor) -> torch.Tensor:
    comp = torch.cat([pack["b"], W, V, torch.log1p(n_b).unsqueeze(1), pack["bond_feat"]], dim=1)
    c = model.group_composer(comp)
    z = c.sum(0, keepdim=True)
    reader_in = torch.cat(
        [z, pack["a_sum"].view(1, -1), pack["a_sq"].view(1, -1), pack["g"].view(1, -1)], dim=1
    )
    return model.reader(reader_in).view(())


def predict_null(
    model: src.BJOC,
    pack: Mapping[str, Any],
    w: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    nu: torch.Tensor | None = None,
) -> torch.Tensor:
    w = pack["w"] if w is None else w
    mu = w.mean(0) if mu is None else mu
    nu = (w * w).mean(0) if nu is None else nu
    n_b = pack["n_b"]
    return _reader_from_components(model, pack, n_b.unsqueeze(1) * mu.unsqueeze(0), n_b.unsqueeze(1) * nu.unsqueeze(0), n_b)


def predict_from_mu_nu(model: src.BJOC, pack: Mapping[str, Any], mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    n_b = pack["n_b"]
    return _reader_from_components(model, pack, n_b.unsqueeze(1) * mu.unsqueeze(0), n_b.unsqueeze(1) * nu.unsqueeze(0), n_b)


def predict_organized(model: src.BJOC, pack: Mapping[str, Any], w: torch.Tensor) -> torch.Tensor:
    """Organized grouping (junction -> its own incident bonds) with frozen composer/reader."""
    jb = pack["junc_bond"]
    W = torch.zeros((pack["n_bonds"], w.shape[1]), dtype=w.dtype)
    V = torch.zeros_like(W)
    for col in range(jb.shape[1]):
        W.index_add_(0, jb[:, col], w)
        V.index_add_(0, jb[:, col], w * w)
    n_b = torch.zeros(pack["n_bonds"], dtype=w.dtype)
    n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), dtype=w.dtype))
    return _reader_from_components(model, pack, W, V, n_b)


def ig_scores(model: src.BJOC, pack: Mapping[str, Any], baseline: torch.Tensor, K: int = IG_STEPS, chunk: int = IG_CHUNK) -> dict[str, Any]:
    """Integrated gradients of the frozen prediction w.r.t. each encoded ``w_j``.

    The Null model depends on ``w`` only through ``mu = mean_j w_j`` and
    ``nu = mean_j w_j^2``, so the path integral is computed exactly by
    differentiating the composer+reader w.r.t. the batched path statistics.
    """
    w = pack["w"]
    n = w.shape[0]
    mu0 = w.mean(0)
    nu0 = (w * w).mean(0)
    u = w - baseline.unsqueeze(0)
    ts = (torch.arange(K, dtype=w.dtype) + 0.5) / K
    ig = torch.zeros(n, dtype=w.dtype)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        m = end - start
        wjk = baseline.view(1, 1, -1) + ts.view(K, 1, 1) * u[start:end].view(1, m, -1)
        wc = w[start:end]
        mu_path = mu0.view(1, 1, -1) + (wjk - wc.view(1, m, -1)) / float(n)
        nu_path = nu0.view(1, 1, -1) + (wjk * wjk - (wc * wc).view(1, m, -1)) / float(n)
        p = K * m
        mu_rows = mu_path.reshape(p, -1).clone().requires_grad_(True)
        nu_rows = nu_path.reshape(p, -1).clone().requires_grad_(True)
        n_b = pack["n_b"]
        n_bonds = n_b.shape[0]
        W = (n_b.view(1, n_bonds, 1) * mu_rows.view(p, 1, -1)).expand(p, n_bonds, -1)
        V = (n_b.view(1, n_bonds, 1) * nu_rows.view(p, 1, -1)).expand(p, n_bonds, -1)
        b_rep = pack["b"].view(1, n_bonds, -1).expand(p, -1, -1)
        f_rep = pack["bond_feat"].view(1, n_bonds, -1).expand(p, -1, -1)
        l_rep = torch.log1p(n_b).view(1, n_bonds, 1).expand(p, -1, -1)
        comp = torch.cat([b_rep, W, V, l_rep, f_rep], dim=2).reshape(p * n_bonds, -1)
        c = model.group_composer(comp).view(p, n_bonds, -1)
        z = c.sum(1)
        reader_in = torch.cat(
            [z, pack["a_sum"].view(1, -1).expand(p, -1), pack["a_sq"].view(1, -1).expand(p, -1), pack["g"].view(1, -1).expand(p, -1)],
            dim=1,
        )
        pred = model.reader(reader_in).view(p)
        g_mu_rows, g_nu_rows = torch.autograd.grad(pred.sum(), [mu_rows, nu_rows], retain_graph=False)
        g_mu = g_mu_rows.view(K, m, -1)
        g_nu = g_nu_rows.view(K, m, -1)
        dmu = u[start:end] / float(n)
        dnu_fixed = 2.0 * baseline.unsqueeze(0) * u[start:end] / float(n)
        dnu_time = 2.0 * ts.view(K, 1, 1) * (u[start:end].view(1, m, -1) ** 2) / float(n)
        integrand = (g_mu * dmu.view(1, m, -1) + g_nu * (dnu_fixed.view(1, m, -1) + dnu_time)).sum(-1)
        ig[start:end] = integrand.mean(0)
    with torch.no_grad():
        pred_nat = predict_null(model, pack)
        pred_all_b = predict_from_mu_nu(model, pack, baseline, baseline * baseline)
    total = float(ig.sum().item())
    delta = float(pred_nat.item() - pred_all_b.item())
    return {
        "ig": ig,
        "pred_nat": float(pred_nat.item()),
        "pred_all_baseline": float(pred_all_b.item()),
        "ig_sum": total,
        "sum_abs_ig": float(ig.abs().sum().item()),
        "joint_reset_delta": delta,
        "joint_reset_nonadditivity_gap": total - delta,
        "joint_reset_nonadditivity_rel": abs(total - delta) / max(abs(delta), 1e-9),
    }


def occlusion_scores(model: src.BJOC, pack: Mapping[str, Any], baseline: torch.Tensor) -> torch.Tensor:
    """Exact single-junction occlusion: ``w_j <- baseline``, full Null recompute."""
    w = pack["w"]
    out = torch.zeros(w.shape[0], dtype=w.dtype)
    with torch.no_grad():
        pred_nat = predict_null(model, pack)
        for j in range(w.shape[0]):
            w2 = w.clone()
            w2[j] = baseline
            out[j] = predict_null(model, pack, w=w2) - pred_nat
    return out


def substitution_scores(model: src.BJOC, pack: Mapping[str, Any], w_ref: torch.Tensor, alpha: float) -> torch.Tensor:
    """Matched natural-encoding substitution per junction; full Null recompute."""
    w = pack["w"]
    out = torch.zeros(w.shape[0], dtype=w.dtype)
    with torch.no_grad():
        pred_nat = predict_null(model, pack)
        for j in range(w.shape[0]):
            w2 = w.clone()
            w2[j] = w[j] + float(alpha) * (w_ref[j % w_ref.shape[0]] - w[j])
            out[j] = predict_null(model, pack, w=w2) - pred_nat
    return out


# ---------------------------------------------------------------------------
# junction adjacency / subset helpers
# ---------------------------------------------------------------------------


def junction_share_matrix(pack: Mapping[str, Any]) -> np.ndarray:
    jb = pack["junc_bond"].numpy()
    n = jb.shape[0]
    share = np.zeros((n, n), dtype=bool)
    for i in range(n):
        si = set(int(x) for x in jb[i])
        for j in range(i + 1, n):
            if si & set(int(x) for x in jb[j]):
                share[i, j] = share[j, i] = True
    return share


def internal_links_from_share(share: np.ndarray, subset: Sequence[int]) -> int:
    s = np.asarray(sorted(set(int(x) for x in subset)), dtype=np.int64)
    if s.size < 2:
        return 0
    return int(share[np.ix_(s, s)].sum() // 2)


def top_k_indices(score: torch.Tensor, k: int) -> list[int]:
    values = score.detach().abs().numpy()
    return [int(i) for i in np.argsort(-values, kind="stable")[:k]]


def w_dtype(feats: Mapping[str, torch.Tensor]) -> torch.dtype:
    return feats["c"].dtype


# ---------------------------------------------------------------------------
# erratum recalcs (task 1)
# ---------------------------------------------------------------------------


def cmd_erratum(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    objs = v1.load_objects()
    v1_manifest = v1.load_sample_manifest()
    v1_b3 = v1.read_json(V1_DIR / "phase_b_nonlocal.json")
    v1_c = v1.read_json(V1_DIR / "phase_c_interface.json")
    norm = v1.load_norm()
    b3_rows = v1_manifest["diagnostic_rows"][:64]
    c_rows = v1_manifest["diagnostic_rows"][:32]
    payload: dict[str, Any] = {
        "namespace": NAMESPACE,
        "purpose": "independent recalculation for ERRATUM points of bjoc_structure_explanation_audit_v1",
        "history_overwritten": False,
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    # ---- point 4: proper Organized-mode perturbation baseline (B3 fix) ----
    org_per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        model = v1.load_model(seed, norm)
        null_deltas, org_proper, org_v1_style = [], [], []
        for row in b3_rows:
            obj = objs[int(row)]
            if int(obj.junc_index.shape[0]) < 2 or int(obj.bond_index.shape[0]) < 2:
                continue
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = pack_from_features(model, batch, feats, 0)
            w = pack["w"]
            dist = v1.all_pairs_distances(obj)
            bjd = v1.bond_junction_distances(obj, dist)
            e_star, j_star = np.unravel_index(int(np.argmax(bjd)), bjd.shape)
            delta = w.std(dim=0, unbiased=False)
            w2 = w.clone()
            w2[int(j_star)] = w2[int(j_star)] + delta
            pred_nat_null = predict_null(model, pack).item()
            pred_pert_null = predict_null(model, pack, w=w2).item()
            pred_nat_org = predict_organized(model, pack, w).item()
            pred_pert_org = predict_organized(model, pack, w2).item()
            null_deltas.append(pred_pert_null - pred_nat_null)
            org_proper.append(pred_pert_org - pred_nat_org)
            org_v1_style.append(pred_pert_org - pred_nat_null)
        org_per_seed[str(seed)] = {
            "n_molecules": len(null_deltas),
            "v1_recorded": {
                "null_dose_alpha_1.0_mean_abs_delta_pred": v1_b3["per_seed"][str(seed)]["dose_response"]["mean_abs_delta_pred"]["1.0"],
                "organized_mean_abs_delta_pred": v1_b3["per_seed"][str(seed)]["organized_counterfactual"]["mean_abs_delta_pred"],
            },
            "null_mean_abs_delta_pred_recalc": mean_abs(np.asarray(null_deltas)),
            "organized_proper_mean_abs_delta_pred": mean_abs(np.asarray(org_proper)),
            "organized_v1_style_mean_abs_delta_pred": mean_abs(np.asarray(org_v1_style)),
            "organized_v1_style_minus_proper": mean_abs(np.asarray(org_v1_style)) - mean_abs(np.asarray(org_proper)),
            "spearman_null_vs_organized_proper": v1.spearman(np.asarray(null_deltas), np.asarray(org_proper)),
            "frac_abs_org_proper_below_abs_null": float(np.mean(np.abs(org_proper) < np.abs(null_deltas))),
            "organized_proper_signed_mean": float(np.mean(org_proper)) if org_proper else float("nan"),
            "null_signed_mean": float(np.mean(null_deltas)) if null_deltas else float("nan"),
        }
    payload["point4_organized_baseline"] = {
        "description": (
            "v1 B3 used pred_null_natural as the baseline for the Organized-mode perturbation; "
            "the Organized mode's own natural prediction is the correct within-mode baseline"
        ),
        "per_seed": org_per_seed,
    }

    # ---- point 2: abs(mean(delta)) vs mean(abs(delta)) in the Phase-C probe ----
    c_per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        model = v1.load_model(seed, norm)
        conn_single, rand_conn_signed, rand_arb_signed = [], [], []
        for row in c_rows:
            obj = objs[int(row)]
            if int(obj.bond_index.shape[0]) < 3:
                continue
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = pack_from_features(model, batch, feats, 0)
            pred_nat = predict_null(model, pack).item()
            n_b = pack["n_bonds"]
            k = min(8, n_b)
            bonds = batch["bond_index"][batch["bond_batch"] == 0]
            bond_adj = [[] for _ in range(n_b)]
            for i in range(n_b):
                si = set(int(x) for x in bonds[i])
                for j in range(i + 1, n_b):
                    if si & set(int(x) for x in bonds[j]):
                        bond_adj[i].append(j)
                        bond_adj[j].append(i)
            connected = v1.bfs_bond_ball(bond_adj, [0], k)
            rng = np.random.default_rng(1000 + seed * 100 + int(row))
            conn_rand = [v1.bfs_bond_ball(bond_adj, [int(rng.integers(0, n_b))], k) for _ in range(8)]
            arb_rand = [sorted(int(x) for x in rng.choice(n_b, size=k, replace=False)) for _ in range(8)]

            def z_masked(subset: Sequence[int]) -> float:
                mask = torch.zeros(n_b, dtype=w_dtype(feats))
                for b in subset:
                    mask[int(b)] = 1.0
                c = feats["c"][batch["bond_batch"] == 0]
                z2 = (c * (1.0 - mask).unsqueeze(1)).sum(0, keepdim=True)
                reader_in = torch.cat([z2, pack["a_sum"].view(1, -1), pack["a_sq"].view(1, -1), pack["g"].view(1, -1)], dim=1)
                return float(model.reader(reader_in).item() - pred_nat)

            conn_single.append(z_masked(connected))
            rand_conn_signed.append([z_masked(s) for s in conn_rand])
            rand_arb_signed.append([z_masked(s) for s in arb_rand])
        rc = np.asarray(rand_conn_signed, dtype=np.float64)
        ra = np.asarray(rand_arb_signed, dtype=np.float64)
        cs = np.asarray(conn_single, dtype=np.float64)
        paired = np.abs(cs) - np.abs(rc).mean(axis=1)
        c_per_seed[str(seed)] = {
            "n_molecules": len(conn_single),
            "v1_recorded": {
                "mean_abs_delta_connected": v1_c["per_seed"][str(seed)]["mean_abs_delta_connected"],
                "mean_abs_delta_random_connected": v1_c["per_seed"][str(seed)]["mean_abs_delta_random_connected"],
                "mean_abs_delta_random_arbitrary": v1_c["per_seed"][str(seed)]["mean_abs_delta_random_arbitrary"],
            },
            "v1_style_random_connected_abs_mean_of_signed": mean_abs(rc.mean(axis=1)),
            "corrected_random_connected_mean_abs": float(np.abs(rc).mean()),
            "connected_mean_abs_single": mean_abs(cs),
            "paired_corrected_connected_minus_random": {
                "mean": float(paired.mean()),
                "median": float(np.median(paired)),
                "frac_positive": float((paired > 0).mean()),
            },
            "v1_style_random_arbitrary_abs_mean_of_signed": mean_abs(ra.mean(axis=1)),
            "corrected_random_arbitrary_mean_abs": float(np.abs(ra).mean()),
        }
    payload["point2_abs_mean_vs_mean_abs"] = {
        "description": (
            "v1 Phase C compared mean|delta| of the connected subset against the magnitude of the mean of "
            "signed deltas of the random draws (abs(mean) rather than mean(abs)); both statistics are "
            "recomputed here on the same molecules with the same draws"
        ),
        "per_seed": c_per_seed,
    }

    # ---- point 3: semantics of norm(c-c_local)/norm(c) + zero-W/V effect ----
    sem_per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        model = v1.load_model(seed, norm)
        fracs, zero_effects = [], []
        for row in c_rows:
            obj = objs[int(row)]
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = pack_from_features(model, batch, feats, 0)
            zeros = torch.zeros_like(pack["n_b"])
            c_local = model.group_composer(
                torch.cat([pack["b"], zeros.unsqueeze(1).expand(-1, 96), zeros.unsqueeze(1).expand(-1, 96), torch.log1p(pack["n_b"]).unsqueeze(1), pack["bond_feat"]], dim=1)
            )
            c = model.group_composer(
                torch.cat([pack["b"], pack["n_b"].unsqueeze(1) * pack["w"].mean(0), pack["n_b"].unsqueeze(1) * (pack["w"] ** 2).mean(0), torch.log1p(pack["n_b"]).unsqueeze(1), pack["bond_feat"]], dim=1)
            )
            fracs.append(float(((c - c_local).norm(dim=1) / c.norm(dim=1).clamp(min=1e-12)).mean().item()))
            zero_effects.append(abs(predict_from_mu_nu(model, pack, torch.zeros(96), torch.zeros(96)).item() - predict_null(model, pack).item()))
        sem_per_seed[str(seed)] = {
            "mean_norm_c_minus_c_local_over_norm_c": float(np.mean(fracs)),
            "mean_abs_pred_delta_zero_WV_counterfactual": float(np.mean(zero_effects)),
        }
    payload["point3_semantics"] = {
        "description": (
            "norm(c-c_local)/norm(c) is the relative change of a zero-W/V counterfactual, not a strict "
            "percentage of global contribution; the measurable quantity is the counterfactual prediction effect"
        ),
        "per_seed": sem_per_seed,
    }

    v1.write_json(OUT / "erratum_recalcs.json", payload)
    log("[erratum] recalcs written")
    return 0


# ---------------------------------------------------------------------------
# prepare / reproduction
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v1_manifest = v1.load_sample_manifest()
    sample = {
        "rule": v1_manifest["rule"],
        "source": "reused verbatim from bjoc_structure_explanation_audit_v1/sample_manifest.json",
        "diagnostic_rows": v1_manifest["diagnostic_rows"],
        "smoke_rows": v1_manifest["smoke_rows"],
        "reference_rows": v1_manifest["reference_rows"],
        "diagnostic_key_sha256": v1_manifest["diagnostic_key_sha256"],
        "reference_key_sha256": v1_manifest["reference_key_sha256"],
        "diagnostic_stats": v1_manifest["diagnostic_stats"],
        "reference_stats": v1_manifest["reference_stats"],
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    v1.write_json(OUT / "sample_manifest.json", sample)

    sources = {
        "namespace": NAMESPACE,
        "git_commit": v1.git_commit(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoints": {
            str(seed): {"path": str(v1.SOUP_PATHS[seed].relative_to(REPO_ROOT)), "sha256": v1.file_sha256(v1.SOUP_PATHS[seed])}
            for seed in SEEDS
        },
        "cache": {
            "objects_train10k_sha256": v1.file_sha256(v1.OBJECTS_TRAIN),
            "prep_train10k_sha256": v1.file_sha256(v1.PREP_TRAIN),
        },
        "code_sha256": {
            "driver": v1.file_sha256(Path(__file__)),
            "protocol": v1.file_sha256(OUT / "PROTOCOL.md"),
            "v1_audit_driver": v1.file_sha256(_CODE_DIR / "bjoc_structure_explanation_audit_v1.py"),
            "composer_module": v1.file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
            "objects_module": v1.file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
        },
        "environment": v1.env_info(),
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    expected = {
        str(v1.SOUP_PATHS[0]): "a021d36e3c07039cf191895f8037d09bf2e734101863dccf8993a69de737f782",
        str(v1.SOUP_PATHS[1]): "b622824a8eb08043777391d7ed617da2639658b413dd83a05c57f81df53d03d9",
        str(v1.OBJECTS_TRAIN): "14326a7ae6f9bb673a5c8b35c7d54171a4008f77edb702f0591d3d1f21338e8d",
        str(v1.PREP_TRAIN): "d0fb86a349ebdd37af069cdaa4ee71ba5045a2b67365cb28b2dd734c594b1e2a",
    }
    checks = {
        "soups_and_cache_match_records": all(
            sources["checkpoints"][str(s)]["sha256"] == expected[str(v1.SOUP_PATHS[s])] for s in SEEDS
        )
        and sources["cache"]["objects_train10k_sha256"] == expected[str(v1.OBJECTS_TRAIN)]
        and sources["cache"]["prep_train10k_sha256"] == expected[str(v1.PREP_TRAIN)]
    }
    sources["historical_hash_checks"] = checks

    objs = v1.load_objects()
    norm = v1.load_norm()
    smoke = sample["smoke_rows"]
    v1_baseline = v1.read_json(V1_DIR / "phase_b_baseline.json")
    repro: dict[str, Any] = {}
    for seed in SEEDS:
        model = v1.load_model(seed, norm)
        batch = v1.make_batch(objs, smoke)
        with torch.no_grad():
            feats = model.forward_features(batch)
            pred = model.reader(feats["reader_in"]).view(-1)
            relabelled = np.asarray(
                [
                    v1.run_pred(model, sro.batch_objects([v1.atom_relabel(objs[int(r)], list(reversed(range(objs[int(r)].n_atoms))))]))[0]
                    for r in smoke
                ],
                dtype=np.float64,
            )
            single = np.asarray([v1.run_pred(model, v1.make_batch(objs, [int(r)]))[0] for r in smoke], dtype=np.float64)
        recorded = v1_baseline["per_seed"][str(seed)]
        repro[str(seed)] = {
            "params": int(sum(p.numel() for p in model.parameters())),
            "reader_in_dim": int(feats["reader_in"].shape[1]),
            "forward_vs_reader_max_abs": float((pred - model(batch)).abs().max()),
            "atom_relabel_max_abs": float(np.abs(pred.numpy() - relabelled).max()),
            "single_vs_batch_max_abs": float(np.abs(pred.numpy() - single).max()),
            "recorded_forward_vs_reader_max_abs": recorded["forward_vs_reader_of_features_max_abs"],
            "recorded_atom_relabel_max_abs": recorded["atom_relabel_max_abs"],
            "recorded_single_vs_batch_max_abs": recorded["single_vs_batch_max_abs"],
        }
    sources["reproduction_gate"] = repro
    sources["reproduction_pass"] = all(
        r["params"] == 413505
        and r["reader_in_dim"] == 321
        and r["forward_vs_reader_max_abs"] <= 1e-6
        and r["atom_relabel_max_abs"] <= 1e-5
        and r["single_vs_batch_max_abs"] <= 1e-5
        and abs(r["forward_vs_reader_max_abs"] - r["recorded_forward_vs_reader_max_abs"]) <= 1e-6
        and abs(r["atom_relabel_max_abs"] - r["recorded_atom_relabel_max_abs"]) <= 1e-6
        and abs(r["single_vs_batch_max_abs"] - r["recorded_single_vs_batch_max_abs"]) <= 1e-6
        for r in repro.values()
    )
    v1.write_json(OUT / "source_manifest.json", sources)
    log(f"[prepare] hash checks {checks}; reproduction_pass={sources['reproduction_pass']}")
    if not sources["historical_hash_checks"] or not sources["reproduction_pass"]:
        raise SystemExit("source/reproduction gate failed — STOP")
    return 0


# ---------------------------------------------------------------------------
# synthetic / known-truth correctness checks
# ---------------------------------------------------------------------------


def permute_junctions_with_perm(batch: Mapping[str, torch.Tensor], seed: int) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    """Permute junction rows within each molecule; return (batch, order) with new[r] = old[order[r]]."""
    rng = np.random.default_rng(int(seed))
    jb = batch["junc_batch"].numpy()
    order = np.arange(len(jb))
    for g in np.unique(jb):
        idx = np.where(jb == g)[0]
        order[idx] = rng.permutation(idx)
    out = dict(batch)
    for k in ("junc_feat", "junc_index", "junc_bond", "junc_batch"):
        if k in batch:
            out[k] = batch[k][torch.as_tensor(order)]
    return out, order


def synthetic_known_truth(model: src.BJOC) -> dict[str, Any]:
    """Inject one strongly deviating junction; check top-1 by IG and by exact occlusion."""
    objs = v1.load_objects()
    row0 = int(v1.read_json(OUT / "sample_manifest.json")["smoke_rows"][0])
    batch = v1.make_batch(objs, [row0])
    feats = v1.run_features(model, batch)
    pack = pack_from_features(model, batch, feats, 0)
    n = pack["n_junctions"]
    if n < 2:
        return {"top1_hit": False, "n_junctions": n, "note": "skipped"}
    w = pack["w"].clone()
    deviation = (w - w.mean(0)).norm(dim=1)
    target = int(torch.argmax(deviation).item())
    w[target] = w[target] + 5.0 * float(deviation.median().item() + 1e-6)
    mu = w.mean(0)
    ig = ig_scores(model, {**pack, "w": w}, mu)
    occ = occlusion_scores(model, {**pack, "w": w}, mu)
    top_ig = top_k_indices(ig["ig"], 1)[0]
    top_occ = top_k_indices(occ, 1)[0]
    return {
        "smoke_row": row0,
        "n_junctions": n,
        "injected_junction": target,
        "top1_ig": top_ig,
        "top1_occlusion": top_occ,
        "top1_hit": bool(top_ig == target and top_occ == target),
        "ig_target": float(ig["ig"][target].item()),
        "ig_mean_abs_all": float(ig["ig"].abs().mean().item()),
    }


def cmd_synthetic(args: argparse.Namespace) -> int:
    objs = v1.load_objects()
    sample = v1.read_json(OUT / "sample_manifest.json")
    norm = v1.load_norm()
    smoke = sample["smoke_rows"][:SYNTH_SMOKE]
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        model = v1.load_model(seed, norm)
        pooled_ig, pooled_occ = [], []
        records = []
        perm_ok = True
        sym_ok = True
        completeness = []
        for row in smoke:
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = pack_from_features(model, batch, feats, 0)
            n = pack["n_junctions"]
            if n < 2:
                continue
            mu = pack["w"].mean(0)
            ig = ig_scores(model, pack, mu)
            occ = occlusion_scores(model, pack, mu)
            pooled_ig.extend(ig["ig"].tolist())
            pooled_occ.extend(occ.tolist())
            pointwise_rel = float((ig["ig"] + occ).abs().max().item() / max(float(occ.abs().mean().item()), 1e-9))
            records.append(
                {
                    "row": int(row),
                    "n_junctions": n,
                    "spearman_ig_vs_neg_occ": v1.spearman(ig["ig"].numpy(), -occ.numpy()),
                    "topk_jaccard": jaccard(top_k_indices(ig["ig"], min(TOP_K, n)), top_k_indices(occ, min(TOP_K, n))),
                    "pointwise_rel_error": pointwise_rel,
                    "joint_reset_nonadditivity_gap": ig["joint_reset_nonadditivity_gap"],
                    "joint_reset_nonadditivity_rel": ig["joint_reset_nonadditivity_rel"],
                    "joint_reset_delta": ig["joint_reset_delta"],
                    "ig_sum": ig["ig_sum"],
                    "sum_abs_ig": ig["sum_abs_ig"],
                }
            )
            completeness.append(ig["joint_reset_nonadditivity_rel"])
            perm_batch, order = permute_junctions_with_perm(batch, seed=1000 + seed * 17 + int(row))
            perm_feats = v1.run_features(model, perm_batch)
            perm_pack = pack_from_features(model, perm_batch, perm_feats, 0)
            ig_perm = ig_scores(model, perm_pack, mu)
            diff = float(np.abs(ig_perm["ig"].numpy() - ig["ig"].numpy()[order]).max())
            if diff > 1e-5:
                perm_ok = False
            w = pack["w"].clone()
            w[1] = w[0]
            sym_scores = ig_scores(model, {**pack, "w": w}, w.mean(0))["ig"]
            if float(abs(sym_scores[0].item() - sym_scores[1].item())) > 1e-6:
                sym_ok = False
        per_seed[str(seed)] = {
            "n_molecules": len(records),
            "pooled_spearman_ig_vs_neg_occ": v1.spearman(np.asarray(pooled_ig), -np.asarray(pooled_occ)),
            "mean_per_molecule_spearman_ig_vs_neg_occ": float(np.mean([m["spearman_ig_vs_neg_occ"] for m in records])),
            "min_per_molecule_spearman_ig_vs_neg_occ": float(np.min([m["spearman_ig_vs_neg_occ"] for m in records])),
            "mean_topk_jaccard_ig_vs_occ": float(np.mean([m["topk_jaccard"] for m in records])),
            "permutation_equivariance_pass": bool(perm_ok),
            "symmetry_pass": bool(sym_ok),
            "max_pointwise_rel_error": float(np.max([m["pointwise_rel_error"] for m in records])),
            "mean_joint_reset_nonadditivity_rel": float(np.mean(completeness)),
            "max_joint_reset_nonadditivity_rel": float(np.max(completeness)),
            "known_truth_witness": synthetic_known_truth(model),
            "records": records,
        }
    checks = {
        "permutation_equivariance": all(p["permutation_equivariance_pass"] for p in per_seed.values()),
        "symmetry": all(p["symmetry_pass"] for p in per_seed.values()),
        "ig_vs_occlusion_identity": all(
            p["max_pointwise_rel_error"] <= POINTWISE_REL_THRESHOLD and p["min_per_molecule_spearman_ig_vs_neg_occ"] >= IG_OCC_SPEARMAN_THRESHOLD for p in per_seed.values()
        ),
        "ig_vs_occlusion_topk_jaccard": all(p["mean_topk_jaccard_ig_vs_occ"] >= 0.5 for p in per_seed.values()),
        "known_truth_witness_top1": all(p["known_truth_witness"]["top1_hit"] for p in per_seed.values()),
    }
    payload = {
        "namespace": NAMESPACE,
        "git_commit": v1.git_commit(),
        "per_seed": per_seed,
        "checks": checks,
        "all_pass": all(checks.values()),
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    v1.write_json(OUT / "synthetic_checks.json", payload)
    log(f"[synthetic] checks {checks}")
    return 0


# ---------------------------------------------------------------------------
# main attribution: IG ranking + independent matched-natural-encoding validation
# ---------------------------------------------------------------------------


def _reference_w(model: src.BJOC, objs: Sequence[sro.MolObjects], ref_rows: Sequence[int]) -> dict[int, torch.Tensor]:
    batch = v1.make_batch(objs, ref_rows)
    feats = v1.run_features(model, batch)
    out: dict[int, torch.Tensor] = {}
    for g, row in enumerate(ref_rows):
        w = feats["w"][batch["junc_batch"] == g]
        if w.shape[0] >= 1:
            out[int(row)] = w
    return out


def cmd_attribute(args: argparse.Namespace) -> int:
    synth = v1.read_json(OUT / "synthetic_checks.json")
    if not synth["all_pass"]:
        raise SystemExit("synthetic checks failed; attribution blocked (protocol stop rule)")
    objs = v1.load_objects()
    sample = v1.read_json(OUT / "sample_manifest.json")
    norm = v1.load_norm()
    diag = sample["diagnostic_rows"][: int(args.molecules)]
    ref_rows = sample["reference_rows"]
    per_seed: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    included_rows: list[int] = []
    for seed in SEEDS:
        model = v1.load_model(seed, norm)
        ref_w = _reference_w(model, objs, ref_rows)
        ref_valid = [(int(r), sample["reference_stats"][str(int(r))], ref_w[int(r)]) for r in ref_rows if int(r) in ref_w]
        records = []
        pooled: dict[str, list[np.ndarray]] = {k: [] for k in ("ig_mean", "ig_zero", "occ_mean", "d_sub10", "d_sub05")}
        zero_block_effects = {k: [] for k in ("z_bond", "sum_atom", "sum_atom_squared", "global_topology")}
        junction_total_effect = []
        per_row_arrays: dict[str, list[np.ndarray]] = {k: [] for k in pooled}
        for row in diag:
            stats = sample["diagnostic_stats"][str(int(row))]
            ref_row, _, w_ref = min(
                ref_valid,
                key=lambda t: (
                    abs(t[1]["n_atoms"] - stats["n_atoms"]) + abs(t[1]["n_bonds"] - stats["n_bonds"]) + abs(t[1]["n_junctions"] - stats["n_junctions"]),
                    t[0],
                ),
            )
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = pack_from_features(model, batch, feats, 0)
            n = pack["n_junctions"]
            if n < 1:
                continue
            if seed == 0:
                included_rows.append(int(row))
            mu = pack["w"].mean(0)
            ig_mean = ig_scores(model, pack, mu)
            ig_zero = ig_scores(model, pack, torch.zeros_like(mu))
            with torch.no_grad():
                pred_nat = predict_null(model, pack)
                occ = occlusion_scores(model, pack, mu)
                d10 = substitution_scores(model, pack, w_ref, 1.0)
                d05 = substitution_scores(model, pack, w_ref, 0.5)
                ridx = feats["reader_in"][0]
                for name, (lo, hi) in (("z_bond", (0, 96)), ("sum_atom", (96, 192)), ("sum_atom_squared", (192, 288)), ("global_topology", (288, 321))):
                    mod = ridx.clone()
                    mod[lo:hi] = 0.0
                    zero_block_effects[name].append(abs(float(model.reader(mod.view(1, -1)).item()) - float(pred_nat.item())))
                junction_total_effect.append(abs(float(predict_from_mu_nu(model, pack, torch.zeros(96), torch.zeros(96)).item()) - float(pred_nat.item())))
            for key, val in (("ig_mean", ig_mean["ig"]), ("ig_zero", ig_zero["ig"]), ("occ_mean", occ), ("d_sub10", d10), ("d_sub05", d05)):
                pooled[key].append(val.numpy())
                per_row_arrays[key].append(val.numpy())
            k = min(TOP_K, n)
            top = top_k_indices(ig_mean["ig"], k)
            complement = [i for i in range(n) if i not in set(top)]
            share = junction_share_matrix(pack)
            rng = np.random.default_rng(7_000_000 + seed * 100_000 + int(row))
            random_means_10, random_means_05, random_links = [], [], []
            for _ in range(RANDOM_DRAWS):
                if complement:
                    draw = np.asarray(rng.choice(complement, size=min(k, len(complement)), replace=False), dtype=np.int64)
                    random_means_10.append(float(d10[torch.as_tensor(draw)].abs().mean()))
                    random_means_05.append(float(d05[torch.as_tensor(draw)].abs().mean()))
                random_links.append(internal_links_from_share(share, rng.permutation(n)[:k]))
            top_links = internal_links_from_share(share, top)
            random_links_arr = np.asarray(random_links, dtype=np.float64)
            # S1 (exploratory, PROTOCOL R2): top-ranked adjacent junction pair vs random pairs
            s1: dict[str, Any] = {}
            pair_adj = [(i, j) for i in range(n) for j in range(i + 1, n) if share[i, j]]
            pair_dis = [(i, j) for i in range(n) for j in range(i + 1, n) if not share[i, j]]
            if pair_adj and pair_dis:
                abs_ig = np.abs(ig_mean["ig"].numpy())
                pair_scores = np.asarray([(abs_ig[i] + abs_ig[j]) / 2.0 for i, j in pair_adj])
                top_pair = pair_adj[int(np.argmax(pair_scores))]

                def joint_delta(pair: tuple[int, int]) -> float:
                    w2 = pack["w"].clone()
                    for jj in pair:
                        w2[jj] = w2[jj] + 1.0 * (w_ref[jj % w_ref.shape[0]] - w2[jj])
                    with torch.no_grad():
                        return float(predict_null(model, pack, w=w2)) - float(pred_nat)

                rng_s1 = np.random.default_rng(11_000_000 + seed * 100_000 + int(row))
                adj_draws = [joint_delta(pair_adj[int(k)]) for k in rng_s1.integers(0, len(pair_adj), size=RANDOM_DRAWS)]
                dis_draws = [joint_delta(pair_dis[int(k)]) for k in rng_s1.integers(0, len(pair_dis), size=RANDOM_DRAWS)]
                top_joint = joint_delta(top_pair)
                single_sum = float(d10[top_pair[0]]) + float(d10[top_pair[1]])
                s1 = {
                    "top_pair": [int(x) for x in top_pair],
                    "top_joint": float(top_joint),
                    "random_adjacent_mean_abs": float(np.mean(np.abs(adj_draws))),
                    "random_disconnected_mean_abs": float(np.mean(np.abs(dis_draws))),
                    "paired_adjacent_diff": abs(top_joint) - float(np.mean(np.abs(adj_draws))),
                    "paired_disconnected_diff": abs(top_joint) - float(np.mean(np.abs(dis_draws))),
                    "interaction_signed": float(top_joint) - single_sum,
                    "single_sum_signed": single_sum,
                }
            records.append(
                {
                    "row": int(row),
                    "reference_row": int(ref_row),
                    "n_junctions": n,
                    "n_bonds": pack["n_bonds"],
                    "k": k,
                    "top_k": [int(x) for x in top],
                    "top_k_zero": top_k_indices(ig_zero["ig"], k),
                    "top_mean_abs_d10": float(d10[torch.as_tensor(top)].abs().mean()),
                    "random_mean_abs_d10": float(np.mean(random_means_10)) if random_means_10 else float("nan"),
                    "mean_abs_d10_all": float(d10.abs().mean()),
                    "top_mean_abs_d05": float(d05[torch.as_tensor(top)].abs().mean()),
                    "random_mean_abs_d05": float(np.mean(random_means_05)) if random_means_05 else float("nan"),
                    "mean_abs_d05_all": float(d05.abs().mean()),
                    "top_internal_links": int(top_links),
                    "random_internal_links_mean": float(random_links_arr.mean()),
                    "random_internal_links_p95": float(np.quantile(random_links_arr, 0.95)),
                    "frac_random_draws_below_top": float((random_links_arr < top_links).mean()),
                    "joint_reset_nonadditivity_gap": ig_mean["joint_reset_nonadditivity_gap"],
                    "joint_reset_nonadditivity_rel": ig_mean["joint_reset_nonadditivity_rel"],
                    "joint_reset_delta": ig_mean["joint_reset_delta"],
                    "ig_sum": ig_mean["ig_sum"],
                    "sum_abs_ig": ig_mean["sum_abs_ig"],
                    "s1": s1,
                }
            )
        row_d10 = np.asarray([r["top_mean_abs_d10"] - r["random_mean_abs_d10"] for r in records], dtype=np.float64)
        row_d05 = np.asarray([r["top_mean_abs_d05"] - r["random_mean_abs_d05"] for r in records], dtype=np.float64)
        frac_above_p95 = np.asarray([r["frac_random_draws_below_top"] for r in records], dtype=np.float64)
        s1_recs = [r["s1"] for r in records if r.get("s1")]
        s1_payload: dict[str, Any] = {}
        if s1_recs:
            s1_payload = {
                "n_molecules": len(s1_recs),
                "adjacent_pair": sign_test(np.asarray([x["paired_adjacent_diff"] for x in s1_recs], dtype=np.float64)),
                "disconnected_pair": sign_test(np.asarray([x["paired_disconnected_diff"] for x in s1_recs], dtype=np.float64)),
                "interaction_signed": sign_test(np.asarray([x["interaction_signed"] for x in s1_recs], dtype=np.float64)),
                "mean_top_joint_abs": float(np.mean([abs(x["top_joint"]) for x in s1_recs])),
                "mean_random_adjacent_abs": float(np.mean([x["random_adjacent_mean_abs"] for x in s1_recs])),
                "mean_random_disconnected_abs": float(np.mean([x["random_disconnected_mean_abs"] for x in s1_recs])),
                "mean_single_sum_signed": float(np.mean([x["single_sum_signed"] for x in s1_recs])),
            }
        per_seed[str(seed)] = {
            "n_molecules": len(records),
            "pooled_spearman_ig_mean_vs_neg_occ_mean": v1.spearman(np.concatenate(pooled["ig_mean"]), -np.concatenate(pooled["occ_mean"])),
            "top_vs_random": {
                "alpha_1.0": sign_test(row_d10),
                "alpha_0.5": sign_test(row_d05),
            },
            "connectivity": {
                "mean_top_internal_links": float(np.mean([r["top_internal_links"] for r in records])),
                "mean_random_internal_links": float(np.mean([r["random_internal_links_mean"] for r in records])),
                "mean_frac_random_draws_below_top": float(frac_above_p95.mean()),
                "median_frac_random_draws_below_top": float(np.median(frac_above_p95)),
                "frac_molecules_top_above_random_p95": float((frac_above_p95 >= 0.95).mean()),
            },
            "stability": {
                "spearman_ig_mean_vs_ig_zero": v1.spearman(np.concatenate(pooled["ig_mean"]), np.concatenate(pooled["ig_zero"])),
                "spearman_d_sub05_vs_d_sub10": v1.spearman(np.concatenate(pooled["d_sub05"]), np.concatenate(pooled["d_sub10"])),
            },
            "channel": {
                "mean_abs_junction_total_effect_zero_WV": float(np.mean(junction_total_effect)),
                "mean_abs_ig_sum": float(np.mean([abs(r["ig_sum"]) for r in records])),
                "median_abs_ig_sum": float(np.median([abs(r["ig_sum"]) for r in records])),
                "mean_sum_abs_ig": float(np.mean([r["sum_abs_ig"] for r in records])),
                "mean_abs_joint_reset_delta": float(np.mean([abs(r["joint_reset_delta"]) for r in records])),
                "mean_joint_reset_nonadditivity_rel": float(np.mean([r["joint_reset_nonadditivity_rel"] for r in records])),
                "mean_abs_zero_block_effects": {k: float(np.mean(v)) for k, v in zero_block_effects.items()},
            },
            "records": records,
            "s1_adjacent_pair_probe": s1_payload,
        }
        for key, vals in per_row_arrays.items():
            arrays[f"seed{seed}__{key}"] = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)
        arrays[f"seed{seed}__row"] = np.repeat(np.asarray(included_rows, dtype=np.int64), [len(v) for v in per_row_arrays["ig_mean"]])
    # cross-seed alignment (same included rows and junction order in both seeds)
    cross: dict[str, Any] = {}
    for key in ("ig_mean", "d_sub10"):
        a0, a1 = arrays[f"seed0__{key}"], arrays[f"seed1__{key}"]
        cross[f"pooled_spearman_seed0_vs_seed1_{key}"] = v1.spearman(a0, a1) if a0.shape == a1.shape else float("nan")
    jac_obs, jac_null = [], []
    for r0, r1 in zip(per_seed["0"]["records"], per_seed["1"]["records"]):
        obs = jaccard(r0["top_k"], r1["top_k"])
        jac_obs.append(obs)
        rng = np.random.default_rng(9_000_000 + r0["row"])
        n, k = r0["n_junctions"], r0["k"]
        nulls = []
        for _ in range(RANDOM_DRAWS):
            nulls.append(jaccard(rng.permutation(n)[:k], rng.permutation(n)[:k]))
        jac_null.append({"mean": float(np.mean(nulls)), "p95": float(np.quantile(nulls, 0.95)), "obs": float(obs)})
    cross["top_k_jaccard_seed0_vs_seed1"] = {
        "mean_observed": float(np.mean(jac_obs)),
        "mean_null": float(np.mean([x["mean"] for x in jac_null])),
        "frac_observed_above_null_p95": float(np.mean([x["obs"] > x["p95"] for x in jac_null])),
    }
    payload = {
        "namespace": NAMESPACE,
        "git_commit": v1.git_commit(),
        "protocol": {
            "ig_steps": IG_STEPS,
            "ig_chunk": IG_CHUNK,
            "top_k": TOP_K,
            "random_draws": RANDOM_DRAWS,
            "sub_alphas": list(SUB_ALPHAS),
            "primary_baseline": "graph_mean(mu_G)",
            "secondary_baseline": "zero",
            "validation_intervention": "matched natural encoding from structure-matched reference molecule (positional junction correspondence), full Null recompute",
        },
        "n_diagnostic": len(included_rows),
        "per_seed": per_seed,
        "cross_seed": cross,
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    v1.write_json(OUT / "attribution.json", payload)
    np.savez_compressed(OUT / "attribution_arrays.npz", **arrays)
    log(f"[attribute] done: {len(included_rows)} molecules x {len(SEEDS)} seeds")
    return 0


# ---------------------------------------------------------------------------
# finalize: decision labels
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    metrics: dict[str, Any] = {
        "namespace": NAMESPACE,
        "git_commit": v1.git_commit(),
        "labels_used": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "decisions": {},
        "criteria": {
            "sign_test_p<0.01_and_positive": "top-k |delta| > random junction |delta| in both seeds and both alphas",
            "connectivity": "majority of molecules have top-k internal links above the random 95th percentile, both seeds",
            "seed_stability": "pooled Spearman >= 0.5 and observed top-k Jaccard above random null p95 for majority",
            "baseline_stability": "Spearman(ig_mean, ig_zero) >= 0.7",
            "alpha_stability": "Spearman(d05, d10) >= 0.7",
            "method_validity": "all synthetic checks pass",
        },
    }
    synth = v1.read_json(OUT / "synthetic_checks.json")
    attr = v1.read_json(OUT / "attribution.json")
    src_manifest = v1.read_json(OUT / "source_manifest.json")
    metrics["source_manifest"] = {
        "historical_hash_checks": src_manifest["historical_hash_checks"],
        "reproduction_pass": src_manifest["reproduction_pass"],
    }
    metrics["method_validity"] = {"all_synthetic_checks_pass": synth["all_pass"], "checks": synth["checks"]}
    per_seed = attr["per_seed"]
    tr = {s: {a: per_seed[s]["top_vs_random"][a] for a in ("alpha_1.0", "alpha_0.5")} for s in ("0", "1")}
    metrics["top_vs_random"] = tr
    descriptives: dict[str, Any] = {}
    for s in ("0", "1"):
        recs = per_seed[s]["records"]
        descriptives[s] = {
            "n_molecules": len(recs),
            "mean_n_junctions": float(np.mean([r["n_junctions"] for r in recs])),
            "mean_n_bonds": float(np.mean([r["n_bonds"] for r in recs])),
            "mean_top_mean_abs_d10": float(np.mean([r["top_mean_abs_d10"] for r in recs])),
            "mean_random_mean_abs_d10": float(np.mean([r["random_mean_abs_d10"] for r in recs])),
            "mean_abs_d10_all": float(np.mean([r["mean_abs_d10_all"] for r in recs])),
            "top_share_of_mass_d10": float(
                np.mean([r["top_mean_abs_d10"] for r in recs]) / max(np.mean([r["mean_abs_d10_all"] for r in recs]), 1e-12)
            ),
            "mean_top_mean_abs_d05": float(np.mean([r["top_mean_abs_d05"] for r in recs])),
            "mean_random_mean_abs_d05": float(np.mean([r["random_mean_abs_d05"] for r in recs])),
            "mean_abs_d05_all": float(np.mean([r["mean_abs_d05_all"] for r in recs])),
            "top_share_of_mass_d05": float(
                np.mean([r["top_mean_abs_d05"] for r in recs]) / max(np.mean([r["mean_abs_d05_all"] for r in recs]), 1e-12)
            ),
            "mean_top_k_baseline_jaccard": float(np.mean([jaccard(r["top_k"], r["top_k_zero"]) for r in recs])),
            "mean_abs_ig_sum": per_seed[s]["channel"]["mean_abs_ig_sum"],
            "mean_sum_abs_ig": per_seed[s]["channel"]["mean_sum_abs_ig"],
            "mean_abs_joint_reset_delta": per_seed[s]["channel"]["mean_abs_joint_reset_delta"],
        }
    metrics["descriptives"] = descriptives
    metrics["exploratory_s1"] = {s: per_seed[s]["s1_adjacent_pair_probe"] for s in ("0", "1")}
    top_supported = all(
        tr[s][a]["mean"] > 0 and tr[s][a]["median"] > 0 and tr[s][a]["p_two_sided_exact"] < 0.01 for s in ("0", "1") for a in ("alpha_1.0", "alpha_0.5")
    )
    conn_supported = all(
        per_seed[s]["connectivity"]["frac_molecules_top_above_random_p95"] > 0.5 for s in ("0", "1")
    )
    cross = attr["cross_seed"]
    seed_stable = (
        all(np.isfinite(cross[f"pooled_spearman_seed0_vs_seed1_{k}"]) and cross[f"pooled_spearman_seed0_vs_seed1_{k}"] >= SPEARMAN_TOP_THRESHOLD for k in ("ig_mean", "d_sub10"))
        and cross["top_k_jaccard_seed0_vs_seed1"]["frac_observed_above_null_p95"] > 0.5
    )
    baseline_stable = all(per_seed[s]["stability"]["spearman_ig_mean_vs_ig_zero"] >= BASELINE_STABILITY_THRESHOLD for s in ("0", "1"))
    alpha_stable = all(per_seed[s]["stability"]["spearman_d_sub05_vs_d_sub10"] >= ALPHA_STABILITY_THRESHOLD for s in ("0", "1"))
    metrics["decisions"] = {
        "METHOD_VALIDATED": bool(synth["all_pass"]),
        "TOP_JUNCTIONS_RESPONSE_EXCEEDS_RANDOM": bool(top_supported),
        "TOP_JUNCTIONS_FORM_CONNECTED_COMBINATIONS": bool(conn_supported),
        "SEED_STABLE": bool(seed_stable),
        "BASELINE_STABLE": bool(baseline_stable),
        "INTERVENTION_STRENGTH_STABLE": bool(alpha_stable),
        "channel_summary": {
            str(s): {
                "junction_total_effect_zero_WV": per_seed[s]["channel"]["mean_abs_junction_total_effect_zero_WV"],
                "within_molecule_attributed_abs_ig_sum": per_seed[s]["channel"]["mean_abs_ig_sum"],
                "attribution_mass_sum_abs_ig": per_seed[s]["channel"]["mean_sum_abs_ig"],
                "joint_reset_delta": per_seed[s]["channel"]["mean_abs_joint_reset_delta"],
                "joint_reset_nonadditivity_rel": per_seed[s]["channel"]["mean_joint_reset_nonadditivity_rel"],
                "other_zero_block_effects": per_seed[s]["channel"]["mean_abs_zero_block_effects"],
            }
            for s in ("0", "1")
        },
    }
    v1.write_json(OUT / "metrics.json", metrics)
    log("[finalize] metrics written")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BJOC frozen-model junction source attribution feasibility v1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("erratum", cmd_erratum), ("prepare", cmd_prepare), ("synthetic", cmd_synthetic), ("attribute", cmd_attribute), ("finalize", cmd_finalize)):
        p = sub.add_parser(name)
        if name == "attribute":
            p.add_argument("--molecules", type=int, default=256)
        p.set_defaults(fn=fn)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
