"""CPU gates for ``zinc_bosi_joint_moment_v1`` — 12 mandatory checks, no GPU.

Run before any formal training:

    python tracks/.../zinc_bosi_joint_moment_v1/code/bjm_checks.py [--rebuild]

Writes ``analysis/cpu_checks.json`` with ``verdict`` = ``CPU_GO`` only when all
12 gates pass.  Official test is never touched; official valid is opened only
for the shape/forward/checkpoint gates (structure and labels are not used to
build any feature, scale or initialisation).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bjm_common as BC  # noqa: E402
import bjm_models as BJM  # noqa: E402
import bosi_data as BD  # noqa: E402

TOL = 1.0e-5
N_REL = BD.N_REL
RANK = BC.RANK
JM_RANK = BJM.JM_RANK


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def layernorm_np(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Biased-variance LayerNorm without affine, matching ``F.layer_norm``."""
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def partner_row_lists(tgt: np.ndarray, n_half: int) -> list[np.ndarray]:
    """For each half-edge, the **partner-row indices** ``p`` with ``tgt[p] == h``.

    Row indices index ``partner_half`` / ``partner_rel`` / the gathered u, w;
    they are NOT half-edge ids.  ``lag[rows]`` gives the partner half-edges.
    """
    order = np.argsort(tgt, kind="stable")
    tgt_s = tgt[order]
    starts = np.searchsorted(tgt_s, np.arange(n_half), side="left")
    stops = np.searchsorted(tgt_s, np.arange(n_half), side="right")
    return [order[starts[h]:stops[h]] for h in range(n_half)]


def ref_joint_block(x: np.ndarray, lag: np.ndarray, rows_by_half: list[np.ndarray],
                    U: np.ndarray, V: np.ndarray, mode: str) -> np.ndarray:
    """Independent float64 NumPy implementation of the JM / CM statistic.

    ``x`` is ``[H, 32]`` (every half-edge), ``lag[p]`` the partner half-edge of
    partner row ``p``, ``rows_by_half[h]`` the partner rows of half-edge ``h``.
    Returns ``[H, 144]``.
    """
    z = layernorm_np(np.asarray(x, dtype=np.float64))
    zr = z[np.asarray(lag, dtype=np.int64)]          # one row per partner
    out = np.zeros((len(rows_by_half), JM_RANK * JM_RANK), dtype=np.float64)
    for h, P in enumerate(rows_by_half):
        if len(P) == 0:
            continue
        u = np.tanh(zr[P] @ np.asarray(U, np.float64).T)
        w = np.tanh(zr[P] @ np.asarray(V, np.float64).T)
        if mode == "JM":
            acc = np.zeros((JM_RANK, JM_RANK), dtype=np.float64)
            for i in range(len(P)):
                acc += np.outer(u[i], w[i])
        else:
            acc = np.outer(u.sum(0), w.sum(0)) / max(len(P), 1)
        out[h] = acc.reshape(-1)
    return out


def ref_pairing_2d(pairs: np.ndarray, mode: str) -> np.ndarray:
    """2-D ``(u, w)`` partner set -> the 2x2 joint statistic, flattened (float64).

    Independent of the torch model: explicit float64 loops over the partner
    rows, ``JM`` pairs each partner with itself before summing, ``CM`` sums the
    two coordinates first and pairs the sums (divided by ``max(n, 1)``).
    """
    uv = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    u, w = uv[:, 0], uv[:, 1]
    acc = np.zeros((2, 2), dtype=np.float64)
    if mode == "JM":
        for i in range(len(u)):
            acc += np.outer([u[i], 0.0], [w[i], 0.0])
        return acc.reshape(-1)
    acc = np.outer([u.sum(), 0.0], [w.sum(), 0.0]) / max(len(u), 1)
    return acc.reshape(-1)


def spec_from_cache(cache, row: int) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    b0, b1 = int(bp[row]), int(bp[row + 1])
    return {
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1].astype(np.float32),
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].astype(np.int64).reshape(-1, 2),
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1].astype(np.float32),
        "global": np.asarray(cache["global_feat"])[row].astype(np.float32),
    }


def path_spec(n_atoms: int, base: int = 0) -> dict:
    """A simple path graph: every atom has degree <= 2, so every half-edge has
    at most one partner."""
    idx = np.stack([np.arange(n_atoms - 1), np.arange(1, n_atoms)], axis=1).astype(np.int64)
    rng = np.random.default_rng(1000 + n_atoms + base)
    af = np.zeros((n_atoms, BC.N_ATOM_FEAT), dtype=np.float32)
    af[:, :5] = rng.normal(0.0, 1.0, size=(n_atoms, 5))
    bf = np.zeros((len(idx), BC.BOND_FEAT), dtype=np.float32)
    bf[:, :4] = rng.normal(0.0, 1.0, size=(len(idx), 4))
    gf = rng.normal(0.0, 1.0, size=(BC.GLOBAL_DIM,)).astype(np.float32)
    return {"atom_feat": af, "bond_index": idx, "bond_feat": bf, "global": gf}


def forward_y(model, batch) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(batch).detach().cpu().numpy().reshape(-1)


def joint_reference_check(model, batch, mode: str) -> dict:
    """Compare the model's own J/C block with the float64 NumPy reference."""
    model.eval()
    with torch.no_grad():
        a, b = model.encode_tokens(batch)
        x = model.half_edges(batch, a, b)
        jb = model.joint_blocks(x, batch["partner_tgt"], batch["partner_half"],
                                batch["partner_half"])
    tgt = batch["partner_tgt"].numpy()
    lag = batch["partner_half"].numpy()
    rows_by_half = partner_row_lists(tgt, x.shape[0])
    ref = []
    for U, V in ((model.joint_u1, model.joint_v1), (model.joint_u2, model.joint_v2)):
        ref.append(ref_joint_block(x.numpy(), lag, rows_by_half,
                                   U.weight.detach().numpy(),
                                   V.weight.detach().numpy(), mode))
    out = {}
    for j, blk in enumerate(jb["blocks"]):
        got = blk.numpy().astype(np.float64)
        d = np.abs(got - ref[j])
        out[f"block{j + 1}_max_abs_diff"] = float(d.max())
        out[f"block{j + 1}_max_ref_abs"] = float(np.abs(ref[j]).max())
    out["max_abs_diff"] = max(v for k, v in out.items() if k.endswith("abs_diff"))
    return out


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate1_parameters(norm) -> dict:
    out = {"checks": {}, "pass": True}
    total_all = {}
    for arm in BJM.ARMS:
        model, info = BJM.build_model(norm, arm, 0, "cpu")
        named = {k: int(v.numel()) for k, v in model.named_parameters()}
        total = int(sum(named.values()))
        total_all[arm] = total
        item = {
            "parameters_from_named_parameters": total,
            "expected": BC.EXPECTED_PARAMETERS[arm],
            "parameter_cap": BC.PARAM_CAP,
            "within_cap": total <= BC.PARAM_CAP,
            "breakdown": info["parameters_breakdown"],
            "side_in_features": int(model.side.lins[0].in_features),
            "side_out_features": int(model.side.lins[-1].out_features),
            "reader_in_features": int(model.reader.lins[0].in_features),
            "half_out": int(model.half_encoder.lins[-1].out_features),
        }
        item["side_in_expected"] = BC.SIDE_IN
        item["side_in_ok"] = item["side_in_features"] == BC.SIDE_IN
        item["side_out_ok"] = item["side_out_features"] == BC.WIDTH
        item["reader_in_ok"] = item["reader_in_features"] == (
            4 * BC.WIDTH + BC.GLOBAL_DIM)
        item["half_out_ok"] = item["half_out"] == BC.RANK
        if arm != "B":
            item["joint_projections"] = {
                k: {"shape": list(getattr(model, k).weight.shape),
                    "parameters": int(getattr(model, k).weight.numel()),
                    "bias": getattr(model, k).bias is not None}
                for k in ("joint_u1", "joint_v1", "joint_u2", "joint_v2")
            }
        out["checks"][arm] = item
        ok = (total == BC.EXPECTED_PARAMETERS[arm] and item["within_cap"]
              and item["side_in_ok"] and item["side_out_ok"]
              and item["reader_in_ok"] and item["half_out_ok"])
        if not ok:
            out["pass"] = False

    # forward shapes on a real valid batch
    cache = BC.load_cache("valid")
    rel = BC.load_rel("valid")
    batch = BD.collate(cache, rel, np.arange(4))
    shapes = {}
    for arm in BJM.ARMS:
        model, _ = BJM.build_model(norm, arm, 0, "cpu")
        y, st = model(batch, return_stats=True)
        shapes[arm] = {
            "y": list(y.shape), "M": list(st["M"].shape), "n": list(st["n"].shape),
            "z": list(st["z"].shape), "s": list(st["s"].shape),
            "c": list(st["c"].shape), "M_dim_expected": BC.M_DIM,
            "M_dim_ok": int(st["M"].shape[1]) == BC.M_DIM,
            "finite": bool(torch.isfinite(y).all() and torch.isfinite(st["M"]).all()),
        }
        if not (shapes[arm]["M_dim_ok"] and shapes[arm]["finite"]
                and shapes[arm]["y"] == [4]):
            out["pass"] = False
    out["forward_shapes"] = shapes
    out["total_parameters"] = total_all
    out["jm_minus_b_parameters"] = total_all["JM"] - total_all["B"]
    out["jm_equals_cm_parameters"] = total_all["JM"] == total_all["CM"]
    if not out["jm_equals_cm_parameters"]:
        out["pass"] = False
    return out


def gate2_initial_state(norm) -> dict:
    out = {"checks": {}, "pass": True}
    for seed in (0, 1):
        states = {}
        for arm in BJM.ARMS:
            model, _ = BJM.build_model(norm, arm, seed, "cpu")
            states[arm] = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
        jm, cm, b = states["JM"], states["CM"], states["B"]
        keys_equal = set(jm) == set(cm)
        diffs = [float((jm[k] - cm[k]).abs().max()) for k in jm] if keys_equal else [float("inf")]
        common_jm = BJM.common_state(jm)
        common_cm = BJM.common_state(cm)
        b_keys = set(b)
        common_keys_equal = set(common_jm) == b_keys == set(common_cm)
        common_diffs = [float((common_jm[k] - b[k]).abs().max())
                        for k in common_jm] if common_keys_equal else [float("inf")]
        proj_stats = {}
        for k in ("joint_u1", "joint_v1", "joint_u2", "joint_v2"):
            w = jm[f"{k}.weight"]
            proj_stats[k] = {"weights": int(w.numel()), "shape": list(w.shape),
                             "std": float(w.std()), "bias": False}
        item = {
            "jm_cm_keys_equal": keys_equal,
            "jm_cm_max_abs_diff": max(diffs) if diffs else None,
            "jm_cm_full_state_sha256": [BC.state_hash(jm), BC.state_hash(cm)],
            "jm_cm_full_state_identical": bool(keys_equal and max(diffs) == 0.0),
            "common_vs_B_keys_equal": common_keys_equal,
            "common_vs_B_max_abs_diff": max(common_diffs) if common_diffs else None,
            "common_state_sha256": {
                "JM": BC.state_hash(common_jm), "CM": BC.state_hash(common_cm),
                "B": BC.state_hash(b)},
            "common_vs_B_identical": bool(common_keys_equal and max(common_diffs) == 0.0),
            "joint_projection_init": proj_stats,
            "joint_projection_init_nonzero": all(v["std"] > 0 for v in proj_stats.values()),
        }
        ok = (item["jm_cm_full_state_identical"] and item["common_vs_B_identical"]
              and item["joint_projection_init_nonzero"])
        item["pass"] = ok
        if not ok:
            out["pass"] = False
        out["checks"][f"seed{seed}"] = item
    return out


def gate3_small_n(norm) -> dict:
    """``n <= 1``: JM and CM must be the same statistic."""
    out = {"checks": {}, "pass": True}
    specs = [path_spec(n, base=i) for i, n in enumerate((2, 3, 5, 8, 12))]
    batch = BD.batch_from_specs(specs)
    result = {}
    ys = {}
    blocks = {}
    for mode in ("JM", "CM"):
        model, _ = BJM.build_model(norm, mode, 0, "cpu")
        model.eval()
        with torch.no_grad():
            y, st = model(batch, return_stats=True)
            a, b = model.encode_tokens(batch)
            x = model.half_edges(batch, a, b)
            jb = model.joint_blocks(x, batch["partner_tgt"], batch["partner_half"],
                                    batch["partner_half"])
        ys[mode] = y.numpy()
        blocks[mode] = torch.cat(jb["blocks"], dim=1).numpy()
        result[f"{mode}_n_partners_max"] = int(jb["n_tot"].max().item())
        result[f"{mode}_n_partners_zero_frac"] = float((jb["n_tot"] == 0).float().mean())
    blk_diff = float(np.abs(blocks["JM"] - blocks["CM"]).max())
    y_diff = float(np.abs(ys["JM"] - ys["CM"]).max())
    result["synthetic_jc_block_max_abs_diff"] = blk_diff
    result["synthetic_y_max_abs_diff"] = y_diff
    result["synthetic_pass"] = blk_diff <= 1e-7 and y_diff <= 1e-7

    # real molecules: restrict to the half-edges whose partner count is <= 1
    cache = BC.load_cache("valid")
    rel = BC.load_rel("valid")
    rb = BD.collate(cache, rel, np.arange(32))
    real = {}
    for mode in ("JM", "CM"):
        model, _ = BJM.build_model(norm, mode, 0, "cpu")
        model.eval()
        with torch.no_grad():
            a, b = model.encode_tokens(rb)
            x = model.half_edges(rb, a, b)
            jb = model.joint_blocks(x, rb["partner_tgt"], rb["partner_half"],
                                    rb["partner_half"])
        real[mode] = torch.cat(jb["blocks"], dim=1).numpy()
        n_tot = jb["n_tot"].numpy().reshape(-1)
    mask = n_tot <= 1
    real["n_le1_rows"] = int(mask.sum())
    real["n_le1_frac"] = float(mask.mean())
    d = float(np.abs(real["JM"][mask] - real["CM"][mask]).max())
    real["max_abs_diff_on_n_le1"] = d
    real["n_ge2_rows"] = int((n_tot >= 2).sum())
    real["max_abs_diff_on_n_ge2"] = float(np.abs(real["JM"][~mask]
                                                 - real["CM"][~mask]).max())
    real["pass"] = d <= 1e-7
    # drop the raw [H, 288] blocks: only the summary numbers are evidence
    real = {k: v for k, v in real.items() if k not in ("JM", "CM")}
    out["synthetic"] = result
    out["real_valid_batch"] = real
    if not (result["synthetic_pass"] and real["pass"]):
        out["pass"] = False
    return out


def gate4_centered_identity(norm) -> dict:
    """``J - C == sum_f (u_f - mean u)(w_f - mean w)^T`` for every n >= 2 row."""
    cache = BC.load_cache("train")
    rel = BC.load_rel("train")
    rows = np.arange(64)
    batch = BD.collate(cache, rel, rows)
    model, _ = BJM.build_model(norm, "JM", 0, "cpu")
    model.eval()
    model_cm, _ = BJM.build_model(norm, "CM", 0, "cpu")
    model_cm.load_state_dict(model.state_dict())
    model_cm.eval()
    with torch.no_grad():
        a, b = model.encode_tokens(batch)
        x = model.half_edges(batch, a, b)
        jb = model.joint_blocks(x, batch["partner_tgt"], batch["partner_half"],
                                batch["partner_half"])
        cm = model_cm.joint_blocks(x, batch["partner_tgt"], batch["partner_half"],
                                   batch["partner_half"])
    jm_blocks = [blk.numpy().astype(np.float64) for blk in jb["blocks"]]
    cm_blocks = [blk.numpy().astype(np.float64) for blk in cm["blocks"]]
    tgt = batch["partner_tgt"].numpy()
    rows_by_half = partner_row_lists(tgt, x.shape[0])
    n_tot = jb["n_tot"].numpy().reshape(-1)
    rows_ge2 = np.where(n_tot >= 2)[0]

    out = {"n_rows_ge2": int(len(rows_ge2)), "blocks": {}}
    worst = 0.0
    for j in range(2):
        u = jb["u"][j].numpy().astype(np.float64)
        w = jb["w"][j].numpy().astype(np.float64)
        diff = jm_blocks[j] - cm_blocks[j]
        ref = np.zeros_like(diff)
        for h in rows_ge2:
            P = rows_by_half[h]
            if len(P) < 2:
                continue
            uu = u[P]
            ww = w[P]
            uc = uu - uu.mean(0, keepdims=True)
            wc = ww - ww.mean(0, keepdims=True)
            ref[h] = (uc.T @ wc).reshape(-1)
        d = float(np.abs(diff[rows_ge2] - ref[rows_ge2]).max())
        scale = float(np.abs(ref[rows_ge2]).max())
        out["blocks"][f"j{j + 1}"] = {"max_abs_diff": d, "max_abs_ref": scale,
                                      "relative": d / max(scale, 1e-12)}
        worst = max(worst, d)
    out["max_abs_diff"] = worst
    out["tolerance"] = TOL
    out["pass"] = bool(worst <= TOL)
    return out


def gate5_pairing_counterexample(norm) -> dict:
    """Two partner sets with identical per-coordinate marginals: JM separates
    them, CM cannot.  Cross-checked against an independent NumPy reference and
    against the model's own statistic on real molecules with a
    marginal-preserving re-pairing."""
    out = {"cases": {}, "pass": True}

    def marginal_moments(pairs, order=6):
        u = np.asarray(pairs, np.float64)[:, 0]
        w = np.asarray(pairs, np.float64)[:, 1]
        return [np.array([np.mean(u ** k) for k in range(order + 1)]),
                np.array([np.mean(w ** k) for k in range(order + 1)])]

    cases = {
        "n2_antipodal": (np.array([[0.5, 0.5], [-0.5, -0.5]]),
                         np.array([[0.5, -0.5], [-0.5, 0.5]])),
        "n3_permuted_ranks": (np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
                              np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])),
    }
    for name, (s1, s2) in cases.items():
        m1, m2 = marginal_moments(s1), marginal_moments(s2)
        marg_same = bool(all(np.allclose(a, b, atol=0.0, rtol=0.0) for a, b in zip(m1, m2)))
        jm1, jm2 = ref_pairing_2d(s1, "JM"), ref_pairing_2d(s2, "JM")
        cm1, cm2 = ref_pairing_2d(s1, "CM"), ref_pairing_2d(s2, "CM")
        item = {
            "set1": s1.tolist(), "set2": s2.tolist(),
            "per_coordinate_moments_identical_order_0_to_6": marg_same,
            "JM": {"set1": jm1.tolist(), "set2": jm2.tolist(),
                   "max_abs_separation": float(np.abs(jm1 - jm2).max())},
            "CM": {"set1": cm1.tolist(), "set2": cm2.tolist(),
                   "max_abs_separation": float(np.abs(cm1 - cm2).max())},
        }
        item["JM_separates"] = item["JM"]["max_abs_separation"] > 0.1
        item["CM_does_not_separate"] = item["CM"]["max_abs_separation"] <= 1e-12
        item["pass"] = bool(marg_same and item["JM_separates"]
                            and item["CM_does_not_separate"])
        if not item["pass"]:
            out["pass"] = False
        out["cases"][name] = item

    # model-level, on real molecules: re-pair w with a permuted partner inside
    # each half-edge group.  Both marginal multisets are preserved exactly, so
    # CM must be invariant and JM must move.
    cache = BC.load_cache("valid")
    rel = BC.load_rel("valid")
    batch = BD.collate(cache, rel, np.arange(48))
    tgt = batch["partner_tgt"].numpy()
    lag = batch["partner_half"].numpy()
    rows_by_half = partner_row_lists(tgt, int(batch["n_half"]))
    perm = lag.copy()
    n_repaired = 0
    for rows in rows_by_half:
        if len(rows) >= 2:
            perm[rows] = np.roll(lag[rows], 1)
            n_repaired += 1
    perm_t = torch.from_numpy(perm.astype(np.int64))
    real = {"n_half_edges_repaird": int(n_repaired),
            "marginal_multisets_preserved": bool(
                all(sorted(lag[rows].tolist()) == sorted(perm[rows].tolist())
                    for rows in rows_by_half))}
    for mode in ("JM", "CM"):
        model, _ = BJM.build_model(norm, mode, 0, "cpu")
        model.eval()
        y0 = forward_y(model, batch)
        model._diag_pairing = perm_t
        y1 = forward_y(model, batch)
        # identical-path guard: _diag_pairing=None must equal a fresh forward
        model._diag_pairing = None
        y2 = forward_y(model, batch)
        real[mode] = {
            "y_max_abs_diff_repair": float(np.abs(y1 - y0).max()),
            "y_moved_fraction": float(np.mean(np.abs(y1 - y0) > 1e-9)),
            "default_path_bitwise_identical": float(np.abs(y2 - y0).max()),
        }
    real["JM_moves"] = real["JM"]["y_max_abs_diff_repair"] > 1e-6
    real["CM_invariant"] = real["CM"]["y_max_abs_diff_repair"] <= 1e-6
    real["pass"] = bool(real["JM_moves"] and real["CM_invariant"]
                        and real["marginal_multisets_preserved"]
                        and real["JM"]["default_path_bitwise_identical"] == 0.0
                        and real["CM"]["default_path_bitwise_identical"] == 0.0)
    out["real_molecule_repairing"] = real
    if not real["pass"]:
        out["pass"] = False
    return out


def gate6_real_statistics(norm) -> dict:
    """Label-free statistics over the full official train 10k."""
    rel = BC.load_rel("train")
    pp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    n_part = np.diff(pp)
    deg = np.asarray(rel["half_deg"], dtype=np.float64)
    n_half = int(len(rel["half_atom_raw"]))
    hist = np.bincount(n_part, minlength=4).astype(np.int64)
    out = {
        "n_half_edges": n_half,
        "partner_count_hist": {str(i): int(hist[i]) for i in range(len(hist))},
        "partner_count_fraction": {str(i): float(hist[i]) / n_half
                                   for i in range(len(hist))},
        "max_atom_degree": int(deg.max()),
        "mean_partner_count": float(n_part.mean()),
        "eligible_half_edge_fraction_n_ge_2": float(np.mean(n_part >= 2)),
        "n0_fraction": float(np.mean(n_part == 0)),
        "n1_fraction": float(np.mean(n_part == 1)),
        "sum_n_equals_deg_minus_1": bool(np.all(n_part == deg - 1)),
        "relation_count_fractions": {},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    pr = np.asarray(rel["partner_rel"], dtype=np.int64)
    cnt = np.bincount(pr, minlength=N_REL)
    out["relation_count_fractions"] = {n: float(c) / int(cnt.sum())
                                       for n, c in zip(BD.REL_NAMES, cnt)}

    # JM vs CM input difference and activation RMS at initialisation
    cache = BC.load_cache("train")
    rows = np.arange(0, 10000, 16)           # 625 molecules, deterministic stride
    batch = BD.collate(cache, rel, rows)
    stats = {}
    full_M = {}
    for mode in ("JM", "CM"):
        model, _ = BJM.build_model(norm, mode, 0, "cpu")
        model.eval()
        with torch.no_grad():
            a, b = model.encode_tokens(batch)
            x = model.half_edges(batch, a, b)
            jb = model.joint_blocks(x, batch["partner_tgt"], batch["partner_half"],
                                    batch["partner_half"])
            Mfull, _ = model.relation_stats(batch, x)
        M = torch.cat(jb["blocks"], dim=1).numpy()
        full_M[mode] = Mfull.numpy()
        stats[mode] = {
            "M_joint_block": M,
            "n_tot": jb["n_tot"].numpy().reshape(-1),
            "a_rms": float(a.pow(2).mean().sqrt()),
            "b_rms": float(b.pow(2).mean().sqrt()),
            "x_rms": float(x.pow(2).mean().sqrt()),
            "j1_rms": float(jb["blocks"][0].pow(2).mean().sqrt()),
            "j2_rms": float(jb["blocks"][1].pow(2).mean().sqrt()),
        }
    diff = np.abs(stats["JM"]["M_joint_block"] - stats["CM"]["M_joint_block"])
    scale = np.abs(stats["CM"]["M_joint_block"]).max()
    nmask = stats["JM"]["n_tot"] >= 2
    out["jm_cm_input_difference"] = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "relative_to_cm_max": float(diff.max() / max(scale, 1e-12)),
        "changed_fraction_gt_1e-6": float(np.mean(diff.max(1) > 1e-6)),
        "changed_fraction_gt_1e-6_on_n_ge_2": float(np.mean(diff.max(1)[nmask] > 1e-6)),
        "n_ge_2_rows": int(nmask.sum()),
    }
    out["activation_rms_at_init"] = {
        mode: {k: v for k, v in s.items() if k not in ("M_joint_block", "n_tot")}
        for mode, s in stats.items()
    }
    out["A_block_rms_of_JM"] = float(
        np.sqrt(np.mean(full_M["JM"][:, :96].astype(np.float64) ** 2)))
    out["A_block_rms_of_CM"] = float(
        np.sqrt(np.mean(full_M["CM"][:, :96].astype(np.float64) ** 2)))
    out["A_block_shared"] = bool(np.array_equal(full_M["JM"][:, :96],
                                                full_M["CM"][:, :96]))
    out["M_384_shape_ok"] = bool(full_M["JM"].shape[1] == BC.M_DIM
                                 and full_M["CM"].shape[1] == BC.M_DIM)
    # B arm, same batch: the 384-dim block it feeds to Side
    model_b, _ = BJM.build_model(norm, "B", 0, "cpu")
    model_b.eval()
    with torch.no_grad():
        yb, sb = model_b(batch, return_stats=True)
    out["B_M_rms"] = float(sb["M"].pow(2).mean().sqrt())
    out["B_M_max"] = float(sb["M"].abs().max())
    out["M_384_max_abs_vs_B"] = float(np.abs(full_M["JM"] - sb["M"].numpy()).max())
    out["JM_vs_CM_moves"] = bool(
        out["jm_cm_input_difference"]["relative_to_cm_max"] > 0.01
        and out["jm_cm_input_difference"]["changed_fraction_gt_1e-6_on_n_ge_2"] > 0.90)
    out["exact_symmetry_rows_note"] = (
        "half-edges whose partners are structurally identical (two equivalent "
        "ring/substituent bonds at the same centre) have J == C exactly by "
        "symmetry; that is a property of the molecule, not of the mechanism")
    out["pass"] = bool(out["sum_n_equals_deg_minus_1"] and out["JM_vs_CM_moves"]
                       and out["jm_cm_input_difference"]["n_ge_2_rows"] > 0)
    return out


def gate7_invariance(norm) -> dict:
    cache = BC.load_cache("valid")
    rows = [3, 11, 42, 77, 108, 512]
    specs = [spec_from_cache(cache, r) for r in rows]
    rng = np.random.default_rng(7)

    model, _ = BJM.build_model(norm, "JM", 0, "cpu")
    base = forward_y(model, BD.batch_from_specs(specs))

    relab = []
    for s in specs:
        n = len(s["atom_feat"])
        p = rng.permutation(n)
        new = dict(s)
        new["atom_feat"] = s["atom_feat"][np.argsort(p)]
        new["bond_index"] = p[s["bond_index"]]
        relab.append(new)
    y_relab = forward_y(model, BD.batch_from_specs(relab))

    rev = []
    for s in specs:
        bi = s["bond_index"].copy()
        flip = rng.random(len(bi)) < 0.5
        bi[flip] = bi[flip][:, ::-1]
        new = dict(s)
        new["bond_index"] = bi
        rev.append(new)
    y_rev = forward_y(model, BD.batch_from_specs(rev))

    bord = []
    for s in specs:
        e = len(s["bond_index"])
        p = rng.permutation(e)
        new = dict(s)
        new["bond_index"] = s["bond_index"][p]
        new["bond_feat"] = s["bond_feat"][p]
        bord.append(new)
    y_bord = forward_y(model, BD.batch_from_specs(bord))

    y_batch_rev = forward_y(model, BD.batch_from_specs(list(reversed(specs))))[::-1]
    half = len(specs) // 2
    y_split = np.concatenate([
        forward_y(model, BD.batch_from_specs(specs[:half])),
        forward_y(model, BD.batch_from_specs(specs[half:])),
    ])

    res = {
        "atom_relabel_max_abs_diff": float(np.abs(y_relab - base).max()),
        "bond_reverse_max_abs_diff": float(np.abs(y_rev - base).max()),
        "bond_order_max_abs_diff": float(np.abs(y_bord - base).max()),
        "batch_order_max_abs_diff": float(np.abs(y_batch_rev - base).max()),
        "batch_recombination_max_abs_diff": float(np.abs(y_split - base).max()),
        "tolerance": TOL,
    }
    res["pass"] = bool(max(v for k, v in res.items() if k.endswith("_diff")) <= TOL)
    return res


def gate8_gradients(norm) -> dict:
    import torch.nn.functional as F

    cache = BC.load_cache("train")
    rel = BC.load_rel("train")
    rows = np.arange(64)
    batch = BD.collate(cache, rel, rows)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    target = torch.as_tensor(y, dtype=torch.float32)
    out = {"arms": {}, "pass": True}
    expected = {
        "B": {"atom_encoder", "bond_encoder", "half_encoder", "side", "compose", "reader"},
        "JM": {"atom_encoder", "bond_encoder", "half_encoder", "side", "compose", "reader",
               "joint_u1", "joint_v1", "joint_u2", "joint_v2"},
        "CM": {"atom_encoder", "bond_encoder", "half_encoder", "side", "compose", "reader",
               "joint_u1", "joint_v1", "joint_u2", "joint_v2"},
    }
    for arm in BJM.ARMS:
        torch.manual_seed(0)
        model, _ = BJM.build_model(norm, arm, 0, "cpu")
        model.train()
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        norms, none_grad, nonfinite = {}, [], []
        for name, p in model.named_parameters():
            if p.grad is None:
                none_grad.append(name)
                continue
            if not torch.isfinite(p.grad).all():
                nonfinite.append(name)
            k = name.split(".")[0]
            norms[k] = norms.get(k, 0.0) + float(p.grad.detach().pow(2).sum())
        norms = {k: math.sqrt(v) for k, v in sorted(norms.items())}
        item = {
            "loss": float(loss), "module_grad_norms": norms,
            "parameters_without_grad": none_grad,
            "non_finite_gradients": nonfinite,
            "modules_with_grad": sorted(norms),
            "expected_modules": sorted(expected[arm]),
        }
        item["all_expected_modules_present"] = set(norms) == expected[arm]
        item["all_grads_nonzero"] = all(v > 0 for v in norms.values())
        if arm != "B":
            item["joint_projection_grads_nonzero"] = all(
                norms.get(k, 0.0) > 0 for k in
                ("joint_u1", "joint_v1", "joint_u2", "joint_v2"))
            # CM has no multiplicative path from the two joint blocks into u/w
            # for n<=1 rows, so only the global gradient needs to be finite.
        ok = (item["all_expected_modules_present"] and item["all_grads_nonzero"]
              and not none_grad and not nonfinite)
        item["pass"] = bool(ok)
        if not ok:
            out["pass"] = False
        out["arms"][arm] = item
    return out


def gate9_batch_integrity(norm) -> dict:
    cache = BC.load_cache("train")
    rel = BC.load_rel("train")
    out = {"checks": {}, "pass": True}
    for rows in (np.arange(128), np.array([0, 417, 9999]), np.arange(63, 191)):
        batch = BD.collate(cache, rel, rows)
        H = int(batch["n_half"])
        tgt = batch["partner_tgt"].numpy()
        lag = batch["partner_half"].numpy()
        hb = batch["half_of_bond"].numpy()
        half_atom = batch["half_atom"].numpy()
        half_deg = batch["half_deg"].numpy()
        # molecule id per half-edge: the batch order of the folded rows
        atom_batch = batch["atom_batch"].numpy()
        mol_of_atom = atom_batch[half_atom]
        plist = partner_row_lists(tgt, H)
        cross, dup, nsum_bad = 0, 0, 0
        for h in range(H):
            P = plist[h]
            if len(P):
                if not np.all(mol_of_atom[lag[P]] == mol_of_atom[h]):
                    cross += 1
                if len(set(lag[P].tolist())) != len(P):
                    dup += 1
            if len(P) != int(round(half_deg[h] - 1)):
                nsum_bad += 1
        # per-relation counts sum to the partner count
        model, _ = BJM.build_model(norm, "JM", 0, "cpu")
        model.eval()
        with torch.no_grad():
            a, b = model.encode_tokens(batch)
            x = model.half_edges(batch, a, b)
            M, n = model.relation_stats(batch, x)
        n_np = n.numpy()
        rel_sum_bad = int((n_np.sum(1) != np.array([len(p) for p in plist])).sum())
        item = {
            "n_molecules": int(batch["n_graphs"]),
            "n_half_edges": H,
            "cross_molecule_partner_rows": int(cross),
            "duplicate_partner_rows": int(dup),
            "half_edges_with_wrong_partner_count": int(nsum_bad),
            "half_edges_where_sum_r_n_ne_partners": rel_sum_bad,
            "max_partners_per_half_edge": int(max((len(p) for p in plist), default=0)),
            "half_of_bond_shape_ok": list(hb.shape) == [int(batch["bond_index"].shape[0]), 2],
        }
        item["pass"] = bool(cross == 0 and dup == 0 and nsum_bad == 0
                            and rel_sum_bad == 0 and item["half_of_bond_shape_ok"])
        if not item["pass"]:
            out["pass"] = False
        out["checks"][f"rows_{int(rows[0])}_{int(rows[-1])}"] = item
    return out


def gate10_provenance(norm) -> dict:
    out = {"checks": {}, "pass": True}
    train = BC.load_cache("train")
    valid = BC.load_cache("valid")
    out["checks"]["train_cache"] = {
        "path": BC.rel_or_abs(BC.DATA_TRAIN),
        "sha256": BC.sha256_file(BC.DATA_TRAIN),
        "split": train.get("split"),
        "n_molecules": int(len(np.asarray(train["atom_ptr"])) - 1),
        "labels_finite": bool(np.isfinite(np.asarray(train["labels"], np.float64)).all()),
        "meta_feature_version": train.get("meta", {}).get("feature_version"),
        "meta_object_source": train.get("meta", {}).get("object_source", {}).get("path"),
    }
    out["checks"]["valid_cache"] = {
        "path": BC.rel_or_abs(BC.DATA_VALID),
        "sha256": BC.sha256_file(BC.DATA_VALID),
        "split": valid.get("split"),
        "n_molecules": int(len(np.asarray(valid["atom_ptr"])) - 1),
        "role": "per-epoch evaluation and best-checkpoint selection only",
    }
    # normalisation provenance: recompute the frozen train-only convention
    # (biased std over the 10k train rows; a coordinate with zero variance gets
    # scale 1.0 -- one such coordinate, the all-zero global-33 slot 19).
    g = np.asarray(train["global_feat"], dtype=np.float64)
    recomputed_mean = g.mean(0)
    recomputed_scale = g.std(0)
    degenerate = recomputed_scale < 1e-6
    recomputed_scale = np.where(degenerate, 1.0, recomputed_scale)
    mean_diff = float(np.abs(recomputed_mean - norm["global_mean"].astype(np.float64)).max())
    scale_diff = float(np.abs(recomputed_scale - norm["global_scale"].astype(np.float64)).max())
    out["checks"]["norm"] = {
        "path": norm["path"], "sha256": norm["sha256"], "source": norm["source"],
        "fit_split": norm["fit_split"],
        "valid_or_test_used": bool(norm["valid_or_test_used"]),
        "max_abs_diff_vs_recomputed_train10k_mean": mean_diff,
        "max_abs_diff_vs_recomputed_train10k_scale": scale_diff,
        "n_train_rows": int(g.shape[0]),
        "degenerate_coordinates": [int(i) for i in np.where(degenerate)[0]],
        "degenerate_convention": "zero-variance coordinate -> scale 1.0",
        "recomputed_from_train_only": bool(mean_diff < 1e-5 and scale_diff < 1e-5
                                           and g.shape[0] == 10_000),
    }
    # SSL checkpoint provenance
    ssl = {}
    for seed in (0, 1):
        p = BC.ssl_path(seed)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        keys = sorted(blob.keys()) if isinstance(blob, dict) else []
        ssl[f"seed{seed}"] = {
            "path": BC.rel_or_abs(p), "sha256": BC.sha256_file(p),
            "top_level_keys": keys,
            "has_atom_encoder": "atom_encoder" in keys,
            "has_bond_encoder": "bond_encoder" in keys,
            "has_reader": "reader" in keys,
            "has_side": "side" in keys,
            "has_compose": "compose" in keys,
            "supervised_weights_loaded": False,
        }
        ssl[f"seed{seed}"]["ssl_only"] = bool(
            "atom_encoder" in keys and "bond_encoder" in keys
            and not any(k in keys for k in ("reader", "side", "compose")))
    out["checks"]["ssl"] = ssl
    # static audit: no module names the official test cache, and the official
    # valid labels are never used outside the evaluation/selection lines.
    code_dir = BC.ROUND / "code"
    files = sorted(p.name for p in code_dir.glob("*.py"))
    text = {p.name: p.read_text() for p in code_dir.glob("*.py")}
    # the auditor itself necessarily contains the tokens it searches for, so it
    # is audited by the explicit offset list below instead of by raw substring
    scan = {n: t for n, t in text.items() if n != "bjm_checks.py"}
    out["checks"]["code_audit"] = {
        "files": files,
        "files_scanned_for_test_access": sorted(scan),
        "mentions_ad_test": {n: ("ad_test" in t) for n, t in scan.items()},
        "mentions_test_cache": {n: ("DATA_TEST" in t or "cache/ad_test" in t)
                                for n, t in scan.items()},
        "auditor_only_occurrences": [
            ln.strip() for ln in text["bjm_checks.py"].splitlines()
            if "ad_test" in ln or "DATA_TEST" in ln],
    }
    out["checks"]["code_audit"]["no_test_access"] = not any(
        out["checks"]["code_audit"]["mentions_ad_test"].values()) and not any(
        out["checks"]["code_audit"]["mentions_test_cache"].values())
    trainer = text["bjm_train.py"]
    yval_lines = [ln.strip() for ln in trainer.splitlines() if "y_valid" in ln]
    allowed = ("y_valid = np.asarray(vcache", "len(y_valid)",
               "BC.mae(valid_pred, y_valid)", "BC.mae(pv, y_valid)",
               "BC.sha256_array(y_valid")
    out["checks"]["valid_label_usage"] = {
        "lines_with_y_valid": yval_lines,
        "n_lines": len(yval_lines),
        "usage_is_eval_and_selection_only": bool(
            len(yval_lines) == 5
            and all(any(a in ln for a in allowed) for ln in yval_lines)),
        "feature_scale_from_valid": False,
        "pretraining_from_valid": False,
    }
    ok = (out["checks"]["norm"]["recomputed_from_train_only"]
          and all(v["ssl_only"] and not v["supervised_weights_loaded"]
                  for v in ssl.values())
          and out["checks"]["code_audit"]["no_test_access"]
          and out["checks"]["valid_label_usage"]["usage_is_eval_and_selection_only"]
          and out["checks"]["train_cache"]["n_molecules"] == 10_000
          and out["checks"]["valid_cache"]["n_molecules"] == 1_000
          and out["checks"]["train_cache"]["split"] == "train"
          and out["checks"]["valid_cache"]["split"] == "valid")
    out["pass"] = bool(ok)
    return out


def gate11_checkpoint_roundtrip(norm) -> dict:
    cache = BC.load_cache("valid")
    rel = BC.load_rel("valid")
    rows = np.arange(200)
    out = {"arms": {}, "pass": True}
    tmp = BC.CACHE / "check_roundtrip.pt"
    for arm in BJM.ARMS:
        model, _ = BJM.build_model(norm, arm, 0, "cpu")
        # perturb so the state is not the initial state
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        y0 = BC.predict(model, cache, rel, rows, device="cpu")
        torch.save({"state_dict": {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}}, tmp)
        fresh, _ = BJM.build_model(norm, arm, 0, "cpu")
        blob = torch.load(tmp, map_location="cpu", weights_only=False)
        fresh.load_state_dict(blob["state_dict"])
        y1 = BC.predict(fresh, cache, rel, rows, device="cpu")
        item = {"max_abs_pred_diff": float(np.abs(y0 - y1).max()),
                "predictions_identical": bool(np.array_equal(y0, y1))}
        item["pass"] = item["predictions_identical"]
        if not item["pass"]:
            out["pass"] = False
        out["arms"][arm] = item
    tmp.unlink(missing_ok=True)
    return out


def gate12_full_cache_forward(norm, rebuild: bool) -> dict:
    import bosi_data as BD

    out = {"pass": True, "splits": {}}
    for split in ("train", "valid"):
        path = BC.rel_path(split)
        rel = BC.load_rel(split)
        content = BD.relation_content_hash(rel)
        info = {
            "path": BC.rel_or_abs(path),
            "file_sha256": BC.sha256_file(path),
            "content_sha256": content,
            "n_molecules": int(rel["n_molecules"]),
        }
        if rebuild:
            t0 = time.perf_counter()
            fresh = BD.build_relation_cache(BC.load_cache(split))
            info["rebuild_content_sha256"] = BD.relation_content_hash(fresh)
            info["rebuild_identical"] = info["rebuild_content_sha256"] == content
            info["rebuild_seconds"] = float(time.perf_counter() - t0)
            if not info["rebuild_identical"]:
                out["pass"] = False
        out["splits"][split] = info

    cache = BC.load_cache("train")
    rel = BC.load_rel("train")
    rows = np.arange(10_000)
    y = np.asarray(cache["labels"], dtype=np.float64)
    forwards = {}
    for arm in BJM.ARMS:
        model, _ = BJM.build_model(norm, arm, 0, "cpu")
        t0 = time.perf_counter()
        pred = BC.predict(model, cache, rel, rows, device="cpu", batch_size=256)
        item = {
            "forward_seconds_full_train10k": float(time.perf_counter() - t0),
            "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
            "pred_min": float(pred.min()), "pred_max": float(pred.max()),
            "finite": bool(np.isfinite(pred).all()),
            "initial_mae_vs_train_labels_reference_only": BC.mae(pred, y),
        }
        item["pass"] = item["finite"]
        if not item["pass"]:
            out["pass"] = False
        forwards[arm] = item
    out["initial_forward_train10k"] = forwards
    out["note"] = ("initial-state forward only; the MAE column is a reference for "
                   "trainability sanity, not a result")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild", action="store_true",
                   help="also rebuild both relation caches in memory and compare hashes")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    norm = BC.load_norm()
    BC.log(f"[bjm-checks] data train={BC.sha256_file(BC.DATA_TRAIN)[:8]} "
           f"valid={BC.sha256_file(BC.DATA_VALID)[:8]} norm={norm['source']}")

    gates = [
        ("G1_parameters_and_shapes", lambda: gate1_parameters(norm)),
        ("G2_initial_state", lambda: gate2_initial_state(norm)),
        ("G3_jm_equals_cm_for_n_le_1", lambda: gate3_small_n(norm)),
        ("G4_centered_joint_identity", lambda: gate4_centered_identity(norm)),
        ("G5_pairing_counterexample", lambda: gate5_pairing_counterexample(norm)),
        ("G6_real_train_statistics", lambda: gate6_real_statistics(norm)),
        ("G7_structural_invariance", lambda: gate7_invariance(norm)),
        ("G8_gradients", lambda: gate8_gradients(norm)),
        ("G9_batch_integrity", lambda: gate9_batch_integrity(norm)),
        ("G10_provenance", lambda: gate10_provenance(norm)),
        ("G11_checkpoint_roundtrip", lambda: gate11_checkpoint_roundtrip(norm)),
        ("G12_full_cache_forward", lambda: gate12_full_cache_forward(norm, args.rebuild)),
    ]
    results = {}
    for name, fn in gates:
        t = time.perf_counter()
        try:
            res = fn()
        except Exception as exc:  # pragma: no cover - gate failure path
            import traceback

            res = {"pass": False, "exception": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        res["seconds"] = float(time.perf_counter() - t)
        results[name] = res
        BC.log(f"[bjm-checks] {name:<32} {'PASS' if res['pass'] else 'FAIL'} "
               f"({res['seconds']:.1f}s)")

    n_pass = sum(1 for r in results.values() if r["pass"])
    verdict = "CPU_GO" if n_pass == len(gates) else "CPU_NO_GO"
    payload = {
        "round": BC.ROUND_ID,
        "written": BC.now(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "official_valid_loaded": True,
        "official_valid_role": "shape / forward / checkpoint gates only",
        "official_test_loaded": False,
        "tolerance": TOL,
        "gates_passed": int(n_pass),
        "gates_total": int(len(gates)),
        "verdict": verdict,
        "total_seconds": float(time.perf_counter() - t0),
        "gates": results,
    }
    out = Path(args.out) if args.out else (BC.ANALYSIS / "cpu_checks.json")
    BC.write_json(out, payload)
    BC.log(f"[bjm-checks] {n_pass}/{len(gates)} PASS -> {verdict} "
           f"({payload['total_seconds']:.1f}s) -> {BC.rel_or_abs(out)}")
    return 0 if verdict == "CPU_GO" else 1


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(BC.REPO), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
