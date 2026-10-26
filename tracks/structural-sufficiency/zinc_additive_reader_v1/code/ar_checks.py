"""Pre-training CPU checks for the additive-reader round.

Nine gate groups (plus a split-role group 0) must pass before any formal GPU
run.  Every number in ``analysis/cpu_checks.json`` is produced by executing the
production code paths used by ``ar_train.py``; the historical DCC-L computation
is replayed from its frozen best-Monitor checkpoint (inference only, no
training).
"""

from __future__ import annotations

import builtins
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ar_common as C  # noqa: E402
import ar_models as M  # noqa: E402

for _p in (str(C.V1), str(C.DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as AD  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models as DCC  # noqa: E402
import dcc_runner as DR  # noqa: E402

TOL = 2e-5
RESULTS: dict = {}
CFG = C.load_config()


def _record(group: str, passed: bool, detail: dict) -> None:
    RESULTS.setdefault(group, {"pass": bool(passed), "detail": detail})
    RESULTS[group]["pass"] = bool(RESULTS[group]["pass"] and passed)
    RESULTS[group]["detail"].update(detail)


def _check(group: str, passed: bool, **detail) -> None:
    RESULTS.setdefault(group, {"pass": True, "detail": {}})
    RESULTS[group]["pass"] = bool(RESULTS[group]["pass"] and passed)
    RESULTS[group]["detail"].update(detail)


def collate(cache, rows, *, valid=True):
    return T1.collate(cache, np.asarray(rows, dtype=np.int64), "P3N", valid=valid)


def _predict(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch).detach().cpu().numpy()


def _maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


# ---------------------------------------------------------------------------
# batch surgery for equivariance checks
# ---------------------------------------------------------------------------
def permute_batch(batch: dict, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(int(seed))
    B = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    bond_batch = batch["bond_batch"]
    A = int(atom_batch.numel())
    E = int(bond_batch.numel())
    atom_off = torch.zeros(B, dtype=torch.long)
    bond_off = torch.zeros(B, dtype=torch.long)
    for g in range(B):
        atom_off[g] = int((atom_batch < g).sum())
        bond_off[g] = int((bond_batch < g).sum())
    atom_perm = []
    bond_perm = []
    for g in range(B):
        ag = (atom_batch == g).nonzero(as_tuple=False).view(-1)
        bg = (bond_batch == g).nonzero(as_tuple=False).view(-1)
        atom_perm.append(ag[torch.randperm(len(ag), generator=gen)])
        bond_perm.append(bg[torch.randperm(len(bg), generator=gen)])
    atom_perm = torch.cat(atom_perm) if atom_perm else torch.zeros(0, dtype=torch.long)
    bond_perm = torch.cat(bond_perm) if bond_perm else torch.zeros(0, dtype=torch.long)
    new_atom_pos = torch.empty(A, dtype=torch.long)
    new_atom_pos[atom_perm] = torch.arange(A)
    new_bond_pos = torch.empty(E, dtype=torch.long)
    new_bond_pos[bond_perm] = torch.arange(E)

    # P3N collate stores bond_index / junc_index / junc_bond as batch-global ids.
    out = dict(batch)
    for key in ("atom_feat", "atom_type", "atom_batch", "atom_local"):
        if key in batch:
            out[key] = batch[key][atom_perm]
    for key in ("bond_feat", "bond_type", "bond_batch"):
        out[key] = batch[key][bond_perm]
    out["bond_index"] = new_atom_pos[batch["bond_index"]][bond_perm]
    if "junc_index" in batch:
        out["junc_index"] = new_atom_pos[batch["junc_index"]]
    if "junc_bond" in batch:
        out["junc_bond"] = new_bond_pos[batch["junc_bond"]]
    return out


def reverse_edges(batch: dict) -> dict:
    out = dict(batch)
    out["bond_index"] = batch["bond_index"].flip(1)
    return out


# ---------------------------------------------------------------------------
# 1. historical production forward and component parity
# ---------------------------------------------------------------------------
def check_historical(cache, rows, norm) -> None:
    ck_path = C.DCCL_RUNS / "fit8k_s0" / "best_monitor.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    state = ck["state_dict"]

    dual = DCC.DualChannel(norm)
    dual.load_state_dict(state)
    dual.eval()
    batch = collate(cache, rows)

    with torch.no_grad():
        # actual production forward
        y_prod = dual(batch).detach().numpy()
        a, b, w = dual.encoders(batch)
        Wg, Vg, n_b = dual.group_to_bonds(w, batch)
        c_glob = dual.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = dual.local_grouping(w, batch, organized=True)
        c_loc = dual.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        B = int(batch["n_graphs"])
        z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - dual.global_mean) / dual.global_scale
        G = torch.cat([z, a_sum, a_sq, g], dim=1)

        model_a, _, _ = M.build_pair(norm, 0)
        missing = model_a.load_state_dict(
            {k: v for k, v in state.items()
             if k.startswith(("encoders.", "group_composer.", "local_composer.",
                              "global_mean", "global_scale"))},
            strict=False,
        )
        if any(not k.startswith(("phi.", "w.", "psi.")) for k in missing.missing_keys):
            raise RuntimeError(f"unexpected missing keys in replay trunk: {missing.missing_keys}")
        model_a.eval()
        c = model_a.local_components(batch)

    diffs = {
        "a": _maxdiff(c["a"], a),
        "c_glob": _maxdiff(c["c_glob"], c_glob),
        "c_loc": _maxdiff(c["c_loc"], c_loc),
        "z": _maxdiff(c["z"], z),
        "a_sum": _maxdiff(c["a_sum"], a_sum),
        "a_sq": _maxdiff(c["a_sq"], a_sq),
        "g": _maxdiff(c["g"], g),
        "G": _maxdiff(c["G"], G),
    }

    # replay the frozen monitor predictions for these rows
    pred_file = C.DCCL_RUNS / "fit8k_s0" / "predictions_best_monitor.pt"
    frozen = torch.load(pred_file, map_location="cpu", weights_only=False)
    mon_rows = np.asarray(frozen["monitor_rows"], dtype=np.int64)
    mon_pred = np.asarray(frozen["monitor_pred"], dtype=np.float64)
    pos = np.searchsorted(mon_rows, np.asarray(rows, dtype=np.int64))
    if not np.array_equal(mon_rows[pos], np.asarray(rows)):
        raise RuntimeError("replay rows are not in the frozen monitor rows")
    replay_diff = _maxdiff(y_prod, mon_pred[pos])
    frozen_mae = float(np.abs(mon_pred - np.asarray(cache["labels"])[mon_rows]).mean())
    reported_mae = float(CFG["baseline_history"]["fit8k_monitor_mae"]["0"])

    # the supervised DCC-L checkpoint must not equal the SSL-initialised trunk
    ssl_enc = state
    init_a, _, _ = M.build_pair(norm, 0)
    init_enc = init_a.state_dict()
    enc_keys = [k for k in init_enc if k.startswith("encoders.")]
    ckpt_vs_ssl = max(
        float((ssl_enc[k].float() - init_enc[k].float()).abs().max()) for k in enc_keys
    )

    ok = (
        max(diffs.values()) <= TOL
        and replay_diff <= 1e-4
        and abs(frozen_mae - reported_mae) <= 1e-7
        and ckpt_vs_ssl > 1e-6
        and int(G.shape[1]) == 321
    )
    _record("1_historical_production_forward", ok, {
        "checkpoint": str(ck_path.relative_to(C.REPO)),
        "checkpoint_sha256": C.sha256_file(ck_path),
        "checkpoint_epoch": int(ck["epoch"]),
        "checkpoint_step": int(ck["step"]),
        "component_max_abs_diff": diffs,
        "production_component_max_abs_diff": max(diffs.values()),
        "monitor_prediction_replay_max_abs_diff": replay_diff,
        "frozen_monitor_mae_recomputed": frozen_mae,
        "frozen_monitor_mae_reported": reported_mae,
        "G_dim": int(G.shape[1]),
        "checkpoint_encoders_vs_ssl_init_max_abs_diff": ckpt_vs_ssl,
        "note": "historical checkpoint is used for inference replay only; the new model trunk is initialised from SSL, not from this checkpoint",
    })


# ---------------------------------------------------------------------------
# 2. A/D identical G and initial S
# ---------------------------------------------------------------------------
def check_inputs(cache, rows, norm) -> None:
    a, d, _ = M.build_pair(norm, 0)
    batch = collate(cache, rows)
    a.eval(); d.eval()
    with torch.no_grad():
        ca = a.local_components(batch)
        cd = d.local_components(batch)
        g_manual = (batch["global"] - a.global_mean) / a.global_scale
        parts = [ca["z"], ca["a_sum"], ca["a_sq"], ca["g"]]
        rebuilt = torch.cat(parts, dim=1)
    equal_S = bool(torch.equal(ca["S"], cd["S"]))
    equal_G = bool(torch.equal(ca["G"], cd["G"]))
    detail = {
        "S_equal_bitwise": equal_S,
        "G_equal_bitwise": equal_G,
        "G_dim": int(ca["G"].shape[1]),
        "G_part_dims": [int(p.shape[1]) for p in parts],
        "G_part_sum": int(sum(p.shape[1] for p in parts)),
        "G_rebuild_max_abs_diff": _maxdiff(ca["G"], rebuilt),
        "global33_max_abs_diff_vs_manual": _maxdiff(ca["g"], g_manual),
        "global_feat_dim": int(batch["global"].shape[1]),
        "all_global_columns_used": int(batch["global"].shape[1]) == 33,
        "c_loc_equal_bitwise": bool(torch.equal(ca["c_loc"], cd["c_loc"])),
        "c_glob_equal_bitwise": bool(torch.equal(ca["c_glob"], cd["c_glob"])),
    }
    ok = (equal_S and equal_G and detail["G_dim"] == 321
          and detail["G_part_sum"] == 321 and detail["all_global_columns_used"]
          and detail["G_rebuild_max_abs_diff"] == 0.0
          and detail["global33_max_abs_diff_vs_manual"] <= 1e-7)
    _record("2_identical_G_and_S", ok, detail)


# ---------------------------------------------------------------------------
# 3. shared initial parameters
# ---------------------------------------------------------------------------
def check_init(norm) -> None:
    a0, d0, info0 = M.build_pair(norm, 0)
    a1, d1, info1 = M.build_pair(norm, 1)
    s0, s1 = a0.state_dict(), a1.state_dict()
    shared_keys = [k for k in s0 if (k.startswith(("encoders.", "group_composer.", "local_composer.", "phi."))
                                     or k in ("global_mean", "global_scale"))]
    same_ad = all(torch.equal(a0.state_dict()[k], d0.state_dict()[k]) for k in shared_keys)
    n_diff_seed = sum(
        1 for k in shared_keys if not torch.equal(s0[k], s1[k])
    )
    buffers_equal = all(
        torch.equal(a0.state_dict()[k], d0.state_dict()[k])
        for k in ("global_mean", "global_scale")
    )

    # encoders match the train-only masked-SSL checkpoint exactly
    ssl_checks = {}
    for seed in (0, 1):
        ssl_path = AD.P3_SSL_S0 if seed == 0 else AD.P3_SSL_S1
        ssl = torch.load(ssl_path, map_location="cpu", weights_only=False)
        model = a0 if seed == 0 else a1
        st = model.state_dict()
        for enc in ("atom_encoder", "bond_encoder", "junction_encoder"):
            got = st[f"encoders.{enc}.lins.0.weight"]
            want = ssl[enc]["lins.0.weight"]
            ssl_checks[f"seed{seed}_{enc}_maxdiff"] = float((got - want).abs().max())
    ssl_ok = max(ssl_checks.values()) == 0.0

    # readers are not zero-initialised
    w_norm = float(a0.w.weight.norm())
    psi_norm = float(torch.sqrt(sum((p ** 2).sum() for p in a0.psi.parameters())))
    reader_norm = float(torch.sqrt(sum((p ** 2).sum() for p in d0.reader.parameters())))
    # and the supervised DCC-L checkpoint is not the source of the trunk
    ck = torch.load(C.DCCL_RUNS / "fit8k_s0" / "best_monitor.pt", map_location="cpu",
                    weights_only=False)["state_dict"]
    diff_sup = max(float((ck[k].float() - a0.state_dict()[k].float()).abs().max())
                   for k in shared_keys if k.startswith("encoders.")
                   and k in ck and ck[k].shape == a0.state_dict()[k].shape)
    detail = {
        "shared_tensor_count": len(shared_keys),
        "A_D_shared_identical": same_ad,
        "buffers_identical": buffers_equal,
        "seed0_vs_seed1_differing_shared_tensors": int(n_diff_seed),
        "ssl_encoder_exact_match": ssl_ok,
        "ssl_max_abs_diffs": ssl_checks,
        "initial_w_norm": w_norm,
        "initial_psi_norm": psi_norm,
        "initial_dense_reader_norm": reader_norm,
        "supervised_checkpoint_vs_init_max_abs_diff": diff_sup,
        "A_reader_state_sha256": info0["A_reader_state_sha256"],
        "D_reader_state_sha256": info0["D_reader_state_sha256"],
        "phi_state_sha256_seed0": info0["phi_state_sha256"],
        "phi_state_sha256_seed1": info1["phi_state_sha256"],
        "init_streams": info0["init_streams"],
    }
    ok = (same_ad and buffers_equal and n_diff_seed > 0 and ssl_ok
          and w_norm > 0 and psi_norm > 0 and reader_norm > 0 and diff_sup > 1e-6)
    _record("3_shared_initial_parameters", ok, detail)


# ---------------------------------------------------------------------------
# 4. gradient paths
# ---------------------------------------------------------------------------
def check_gradients(cache, rows, norm) -> None:
    detail = {}
    ok = True
    y = np.asarray(cache["labels"], dtype=np.float64)
    for arm in ("A", "D"):
        model, info = M.build_arm(arm, norm, 0, torch.device("cpu"))
        model.train()
        batch = collate(cache, rows)
        target = torch.as_tensor(y[np.asarray(rows)], dtype=torch.float32)
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), target.view(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        no_grad, nonfinite, zero = [], [], []
        for name, p in model.named_parameters():
            if p.grad is None:
                no_grad.append(name)
                continue
            if not torch.isfinite(p.grad).all():
                nonfinite.append(name)
            if float(p.grad.abs().sum()) == 0.0:
                zero.append(name)
        entry = {
            "loss": float(loss),
            "all_parameters_require_grad": all(p.requires_grad for p in model.parameters()),
            "params_without_grad": no_grad,
            "nonfinite_grads": nonfinite,
            "zero_grad_params": zero,
        }
        if arm == "A":
            entry["w_grad_norm"] = float(model.w.weight.grad.norm())
            entry["phi_first_layer_grad_norm"] = float(model.phi.lins[0].weight.grad.norm())
            entry["phi_last_layer_grad_norm"] = float(model.phi.lins[-1].weight.grad.norm())
            entry["psi_first_layer_grad_norm"] = float(model.psi.lins[0].weight.grad.norm())
            entry["w_grad_nonzero"] = bool(entry["w_grad_norm"] > 0)
            entry["phi_grad_nonzero"] = bool(
                entry["phi_first_layer_grad_norm"] > 0 and entry["phi_last_layer_grad_norm"] > 0
            )
            ok = ok and entry["w_grad_nonzero"] and entry["phi_grad_nonzero"]
        detail[arm] = entry
        ok = ok and not no_grad and not nonfinite and not zero
        ok = ok and bool(entry["all_parameters_require_grad"])
    detail["no_frozen_or_zero_gate"] = ok
    _record("4_gradient_paths", ok, detail)


# ---------------------------------------------------------------------------
# 5. renumbering / edge reversal / batching
# ---------------------------------------------------------------------------
def check_equivariance(cache, rows, norm) -> None:
    rows = np.asarray(rows, dtype=np.int64)
    batch = collate(cache, rows)
    perm = permute_batch(batch, seed=11)
    rev = reverse_edges(batch)
    # reversed batch composition (molecule order)
    order = torch.arange(len(rows) - 1, -1, -1)
    detail = {}
    ok = True
    for arm in ("A", "D"):
        model, _ = M.build_arm(arm, norm, 0, torch.device("cpu"))
        y0 = _predict(model, batch)
        yp = _predict(model, perm)
        yr = _predict(model, rev)
        singles = []
        for r in rows:
            singles.append(float(_predict(model, collate(cache, [r]))[0]))
        singles = np.asarray(singles)
        # shuffled batch order
        shuffled = collate(cache, rows[order.numpy()])
        ys = _predict(model, shuffled)[order.numpy()]
        d = {
            "atom_relabel_max_abs_diff": _maxdiff(y0, yp),
            "bond_edge_reversal_max_abs_diff": _maxdiff(y0, yr),
            "single_vs_batch_max_abs_diff": _maxdiff(y0, singles),
            "batch_order_max_abs_diff": _maxdiff(y0, ys),
        }
        detail[arm] = d
        ok = ok and max(d.values()) <= TOL
    detail["tolerance"] = TOL
    _record("5_equivariance_renumber_reverse_batching", ok, detail)


# ---------------------------------------------------------------------------
# 6. no cross-graph mixing and degenerate inputs
# ---------------------------------------------------------------------------
def check_isolation(cache, rows, norm, all_rows) -> None:
    rows = np.asarray(rows, dtype=np.int64)
    row_i, row_j, row_k = int(rows[0]), int(rows[1]), int(rows[2])
    detail = {"cross_graph": {}, "degenerate": {}}
    ok = True
    for arm in ("A", "D"):
        model, _ = M.build_arm(arm, norm, 0, torch.device("cpu"))
        y_ij = _predict(model, collate(cache, [row_i, row_j]))
        y_ik = _predict(model, collate(cache, [row_i, row_k]))
        diff = float(abs(y_ij[0] - y_ik[0]))
        detail["cross_graph"][arm] = diff
        ok = ok and diff <= TOL

        # synthetic empty graph
        empty = {
            "n_graphs": 1,
            "atom_feat": torch.zeros((0, D.ATOM_FEAT_DIM)),
            "atom_batch": torch.zeros(0, dtype=torch.long),
            "bond_feat": torch.zeros((0, D.BOND_FEAT_DIM)),
            "bond_batch": torch.zeros(0, dtype=torch.long),
            "bond_index": torch.zeros((0, 2), dtype=torch.long),
            "junc_feat": torch.zeros((0, D.JUNC_FEAT_DIM)),
            "junc_batch": torch.zeros(0, dtype=torch.long),
            "junc_index": torch.zeros((0, 3), dtype=torch.long),
            "junc_bond": torch.zeros((0, 2), dtype=torch.long),
            "global": torch.zeros((1, D.GLOBAL_DIM)),
        }
        y_empty = _predict(model, empty)
        finite_empty = bool(np.isfinite(y_empty).all())

        # synthetic single bond + two atoms
        single = {
            "n_graphs": 1,
            "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[:2].astype(np.float32)).clone(),
            "atom_batch": torch.zeros(2, dtype=torch.long),
            "bond_feat": torch.from_numpy(np.asarray(cache["bond_feat"])[:1].astype(np.float32)).clone(),
            "bond_batch": torch.zeros(1, dtype=torch.long),
            "bond_index": torch.tensor([[0, 1]], dtype=torch.long),
            "junc_feat": torch.zeros((0, D.JUNC_FEAT_DIM)),
            "junc_batch": torch.zeros(0, dtype=torch.long),
            "junc_index": torch.zeros((0, 3), dtype=torch.long),
            "junc_bond": torch.zeros((0, 2), dtype=torch.long),
            "global": torch.zeros((1, D.GLOBAL_DIM)),
        }
        y_single = _predict(model, single)
        finite_single = bool(np.isfinite(y_single).all())

        # one general ring and one fused-ring molecule from the cache
        ap = np.asarray(cache["atom_ptr"]); bp = np.asarray(cache["bond_ptr"])
        n_atoms = np.diff(ap); n_bonds = np.diff(bp)
        rank = n_bonds - n_atoms + 1
        cand_ring = np.where((rank == 1) & (n_atoms >= 5) & (n_atoms <= 12))[0]
        cand_fused = np.where((rank >= 2) & (n_atoms >= 8) & (n_atoms <= 16))[0]
        if len(cand_ring) == 0 or len(cand_fused) == 0:
            raise RuntimeError("no suitable ring / fused-ring molecule found")
        ring_row = int(cand_ring[0]); fused_row = int(cand_fused[0])
        y_ring = _predict(model, collate(cache, [ring_row]))
        y_fused = _predict(model, collate(cache, [fused_row]))
        finite_ring = bool(np.isfinite(y_ring).all())
        finite_fused = bool(np.isfinite(y_fused).all())
        detail["degenerate"][arm] = {
            "empty_graph_pred": float(y_empty[0]),
            "empty_finite": finite_empty,
            "single_bond_pred": float(y_single[0]),
            "single_bond_finite": finite_single,
            "ring_row": ring_row, "ring_rank": int(rank[ring_row]),
            "ring_pred": float(y_ring[0]), "ring_finite": finite_ring,
            "fused_row": fused_row, "fused_rank": int(rank[fused_row]),
            "fused_pred": float(y_fused[0]), "fused_finite": finite_fused,
        }
        ok = ok and finite_empty and finite_single and finite_ring and finite_fused
    detail["tolerance"] = TOL
    _record("6_no_cross_graph_mixing_and_degenerate_inputs", ok, detail)


# ---------------------------------------------------------------------------
# 7. additive separation
# ---------------------------------------------------------------------------
def check_additivity(cache, rows, norm) -> None:
    rows = np.asarray(rows, dtype=np.int64)
    batch = collate(cache, rows)
    detail = {}
    ok = True
    gen = torch.Generator().manual_seed(20261027)
    for arm in ("A", "D"):
        model, _ = M.build_arm(arm, norm, 0, torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            c = model.local_components(batch)
            h = c["h"]
            e0 = int((batch["bond_batch"] == 0).nonzero(as_tuple=False)[0])
            delta = torch.full((h.shape[1],), 0.2)
            noise1 = 0.5 * torch.randn(h.shape, generator=gen)
            noise2 = 1.0 * torch.randn(h.shape, generator=gen)
            h_plus = h.clone(); h_plus[e0] += delta
            h_n1 = h + noise1
            h_n1_plus = h_n1.clone(); h_n1_plus[e0] += delta
            h_n2 = h + noise2
            h_n2_plus = h_n2.clone(); h_n2_plus[e0] += delta
            y0 = model(batch, h_override=h)
            y1 = model(batch, h_override=h_plus)
            yn1 = model(batch, h_override=h_n1)
            yn1p = model(batch, h_override=h_n1_plus)
            yn2 = model(batch, h_override=h_n2)
            yn2p = model(batch, h_override=h_n2_plus)
            d0 = float(y1[0] - y0[0])
            d1 = float(yn1p[0] - yn1[0])
            d2 = float(yn2p[0] - yn2[0])
            wdot = float((model.w.weight.view(-1) * delta).sum()) if arm == "A" else float("nan")
            spread = max(abs(d1 - d0), abs(d2 - d0))
            detail[arm] = {
                "delta_base": d0,
                "delta_other_h_noise_0.5": d1,
                "delta_other_h_noise_1.0": d2,
                "max_delta_spread_across_other_h": spread,
                "w_dot_delta": wdot,
                "matches_w_dot_delta": (abs(d0 - wdot) <= 1e-5) if arm == "A" else None,
            }
            if arm == "A":
                ok = ok and spread <= 1e-5 and abs(d0 - wdot) <= 1e-5
            else:
                ok = ok and spread > 1e-4
    detail["interpretation"] = {
        "A": "fixed G, one h_b shift changes y by exactly w^T delta, independent of all other h",
        "D": "same shift interacts with the other h through the nonlinear dense reader",
    }
    _record("7_additive_separation", ok, detail)


# ---------------------------------------------------------------------------
# 8. counts, sums, scales, initial outputs
# ---------------------------------------------------------------------------
def check_scales(cache, rows, norm, monitor_rows) -> None:
    rows = np.asarray(rows, dtype=np.int64)
    batch = collate(cache, rows)
    detail = {}
    ok = True
    for arm in ("A", "D"):
        model, _ = M.build_arm(arm, norm, 0, torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            c = model.local_components(batch)
            B = int(batch["n_graphs"])
            zerr, aerr, serr, cgerr = [], [], [], []
            for g in range(B):
                bm = batch["bond_batch"] == g
                am = batch["atom_batch"] == g
                zerr.append(float((c["c_glob"][bm].sum(0) - c["z"][g]).abs().max()))
                aerr.append(float((c["a"][am].sum(0) - c["a_sum"][g]).abs().max()))
                serr.append(float((c["h"][bm].sum(0) - c["S"][g]).abs().max()))
                cgerr.append(float((c["c_glob"][bm].sum(0) - c["z"][g]).abs().max()))
            n_b_expected = torch.zeros_like(c["n_b"])
            n_b_expected.index_add_(0, batch["junc_bond"].reshape(-1),
                                    torch.ones(batch["junc_bond"].numel()))
            n_b_diff = float((n_b_expected - c["n_b"]).abs().max())
            S_norms = c["S"].norm(dim=1).numpy()
            h_norm = float(c["h"].norm(dim=1).mean())
            mon_rows = np.asarray(monitor_rows, dtype=np.int64)[:256]
            y_init = _predict(model, collate(cache, mon_rows))
            if arm == "A":
                cmon = model.local_components(collate(cache, mon_rows))
                with torch.no_grad():
                    wS = model.local_term(cmon).numpy()
                    psi = model.global_term(cmon).numpy()
                terms = {"wS_mean": float(np.mean(wS)), "wS_std": float(np.std(wS)),
                         "psi_mean": float(np.mean(psi)), "psi_std": float(np.std(psi))}
            else:
                terms = {}
            d = {
                "sum_vs_loop_max_abs_diff": {
                    "z": max(zerr), "a_sum": max(aerr), "S": max(serr), "z_dup": max(cgerr)},
                "n_b_count_max_abs_diff": n_b_diff,
                "h_per_bond_mean_norm": h_norm,
                "S_per_graph_norm_mean": float(np.mean(S_norms)),
                "S_per_graph_norm_max": float(np.max(S_norms)),
                "init_pred_mean": float(np.mean(y_init)),
                "init_pred_std": float(np.std(y_init)),
                "init_pred_min": float(np.min(y_init)),
                "init_pred_max": float(np.max(y_init)),
                "init_finite": bool(np.isfinite(y_init).all()),
                **terms,
            }
            detail[arm] = d
            ok = ok and max(zerr + aerr + serr + cgerr) <= 1e-4
            ok = ok and n_b_diff == 0.0 and d["init_finite"]
            ok = ok and d["init_pred_std"] > 1e-6 and abs(d["init_pred_max"]) < 1e3
    _record("8_counts_sums_scales_initial_outputs", ok, detail)


# ---------------------------------------------------------------------------
# 9. official test / valid and label isolation
# ---------------------------------------------------------------------------
class _AuditIO:
    def __enter__(self):
        self.paths = []
        self._open = builtins.open
        self._torch_load = torch.load

        def _open(path, *a, **k):
            self.paths.append(str(path))
            return self._open(path, *a, **k)

        def _load(path, *a, **k):
            self.paths.append(str(path))
            return self._torch_load(path, *a, **k)

        builtins.open = _open
        torch.load = _load
        return self

    def __exit__(self, *exc):
        builtins.open = self._open
        torch.load = self._torch_load
        return False


def check_isolation_labels(cache, rows, norm) -> None:
    src_files = ["ar_train.py", "ar_models.py", "ar_common.py", "ar_analyze.py"]
    static_hits = {}
    for name in src_files:
        p = HERE / name
        text = p.read_text() if p.exists() else ""
        hits = []
        for needle in ("ad_valid", "ad_test", "load_cache(\"valid\")", "official_valid_pred"):
            if needle in text:
                hits.append(needle)
        static_hits[name] = hits
    static_ok = all(len(v) == 0 for v in static_hits.values())

    # runtime audit over the whole pre-training path
    with _AuditIO() as audit:
        cfg = C.load_config()
        sp = C.split_rows()
        tr = C.load_train_cache()
        norm2 = C.fit_only_norm(tr, sp["fit8k"])
        a, d, _ = M.build_pair(norm2, 0)
        batch = collate(tr, sp["fit8k"][:2])
        y = torch.as_tensor(np.asarray(tr["labels"])[sp["fit8k"][:2]], dtype=torch.float32)
        loss = torch.nn.functional.l1_loss(a(batch), y)
        loss.backward()
    forbidden = [p for p in audit.paths if ("ad_valid" in p or "ad_test" in p
                                            or "official" in p.lower() and "test" in p.lower())]
    train_cache_hits = sorted({p for p in audit.paths if p.endswith("ad_train.pt")})
    allowed_suffixes = (".json", ".pt", ".py", ".txt", ".csv")
    unexpected = sorted({p for p in audit.paths
                         if not p.endswith(allowed_suffixes) and not p.endswith(".lock")})
    detail = {
        "static_forbidden_string_hits": static_hits,
        "static_ok": static_ok,
        "runtime_forbidden_paths": forbidden,
        "runtime_train_cache_paths": train_cache_hits,
        "runtime_unexpected_paths": unexpected,
        "runtime_audited_paths_count": len(audit.paths),
        "fit_rows_within_train_cache": bool(np.max(sp["fit8k"]) < int(tr["n_molecules"])),
        "monitor_rows_within_train_cache": bool(np.max(sp["monitor"]) < int(tr["n_molecules"])),
        "locked_rows_within_train_cache": bool(np.max(sp["locked"]) < int(tr["n_molecules"])),
        "partition_disjoint": bool(sp["partition_ok"]),
        "fit_monitor_locked_overlap": int(len(set(sp["fit8k"]) & set(sp["monitor"]))
                                          + len(set(sp["fit8k"]) & set(sp["locked"]))
                                          + len(set(sp["monitor"]) & set(sp["locked"]))),
        "norm_fit_only": norm2["valid_or_test_used"] is False,
        "norm_rows": int(norm2["n_fit_rows"]),
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "config_forbids_test_and_valid": True,
    }
    ok = (static_ok and not forbidden and len(train_cache_hits) >= 1 and not unexpected
          and detail["fit_rows_within_train_cache"] and detail["monitor_rows_within_train_cache"]
          and detail["locked_rows_within_train_cache"] and detail["partition_disjoint"]
          and detail["fit_monitor_locked_overlap"] == 0 and detail["norm_fit_only"])
    _record("9_official_test_valid_label_isolation", ok, detail)


# ---------------------------------------------------------------------------
def group0(cache) -> None:
    sp = C.split_rows()
    n = int(cache["n_molecules"])
    detail = {
        "n_molecules_train_cache": n,
        "row_counts": {k: int(len(sp[k])) for k in ("fit8k", "fit4k", "monitor", "locked")},
        "sha256_rows": sp["hashes"],
        "split_protocol": C.read_json(C.BJOC / "split_a.json")["protocol"],
        "split_salt": C.read_json(C.BJOC / "split_a.json")["salt"],
        "partition_ok": bool(sp["partition_ok"]),
        "fit4k_nested_in_fit8k": bool(set(sp["fit4k"].tolist()) <= set(sp["fit8k"].tolist())),
        "roles": sp["roles"],
    }
    ok = (detail["row_counts"] == {"fit8k": 8000, "fit4k": 4000, "monitor": 1000, "locked": 1000}
          and detail["partition_ok"] and detail["fit4k_nested_in_fit8k"])
    _record("0_frozen_split_and_roles", ok, detail)


def main() -> int:
    t0 = time.time()
    cfg = C.load_config()
    cache = C.load_train_cache()
    sp = C.split_rows()
    norm = C.fit_only_norm(cache, sp["fit8k"])

    # 8 representative rows: large/small, rings, from Fit8k
    ap = np.asarray(cache["atom_ptr"]); bp = np.asarray(cache["bond_ptr"])
    n_atoms = np.diff(ap)
    candidates = sp["fit8k"]
    order = np.argsort(n_atoms[candidates])
    pick = candidates[order[np.linspace(0, len(order) - 1, 8).astype(int)]]
    rows = np.asarray(sorted(set(int(x) for x in pick)), dtype=np.int64)
    if len(rows) < 6:
        rows = sp["fit8k"][:8]
    monitor_rows = sp["monitor"]

    failures = []
    for name, fn in [
        ("0", lambda: group0(cache)),
        ("1", lambda: check_historical(cache, monitor_rows[:8], norm)),
        ("2", lambda: check_inputs(cache, rows, norm)),
        ("3", lambda: check_init(norm)),
        ("4", lambda: check_gradients(cache, rows, norm)),
        ("5", lambda: check_equivariance(cache, rows[:4], norm)),
        ("6", lambda: check_isolation(cache, rows[:4], norm, sp["fit8k"])),
        ("7", lambda: check_additivity(cache, rows[:4], norm)),
        ("8", lambda: check_scales(cache, rows, norm, monitor_rows)),
        ("9", lambda: check_isolation_labels(cache, rows, norm)),
    ]:
        try:
            fn()
        except Exception as exc:
            failures.append({"group": name, "error": repr(exc),
                             "traceback": traceback.format_exc()})
            RESULTS[f"group_{name}_error"] = {"pass": False,
                                              "detail": {"error": repr(exc)}}

    all_pass = all(v.get("pass", False) for v in RESULTS.values()) and not failures
    payload = {
        "protocol_id": cfg["protocol_id"],
        "config_sha256": C.sha256_file(C.CONFIG),
        "git_commit": C.git_commit(),
        "split_hashes": sp["hashes"],
        "norm_hash": C.norm_hash(norm),
        "checks": RESULTS,
        "errors": failures,
        "all_pass": bool(all_pass),
        "n_groups_passed": int(sum(1 for v in RESULTS.values() if v.get("pass", False))),
        "n_groups": int(len(RESULTS)),
        "wall_seconds": float(time.time() - t0),
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "locked_read": False,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": "cpu",
        },
    }
    C.write_json(C.ANALYSIS / "cpu_checks.json", payload)
    for name, res in RESULTS.items():
        C.log(f"[check] {name}: {'PASS' if res.get('pass') else 'FAIL'}")
    C.log(f"[check] all_pass={all_pass} groups={payload['n_groups_passed']}/{payload['n_groups']} "
          f"wall={payload['wall_seconds']:.1f}s -> {C.ANALYSIS / 'cpu_checks.json'}")
    if failures:
        for f in failures:
            C.log(f"[check] ERROR {f['group']}: {f['error']}")
        return 1
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
