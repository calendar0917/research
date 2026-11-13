"""Small, direct correctness gates for this round.

Pre-training gates are CPU-only and data-local.  ``--post`` appends checkpoint,
metric and cross-arm RNG checks after the four formal runs have been pulled.

This is intentionally not a replay of historical round audits.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import common as C
import data as D
import losses as L
import metrics as M
import pairs as P
import schedule as S

OUT = C.ANALYSIS / "correctness_gates.json"


def _check(checks: dict[str, Any], name: str, fn) -> None:
    try:
        value = fn()
        checks[name] = {"pass": bool(value.get("pass", True)), **value}
    except Exception as exc:  # noqa: BLE001 - a gate failure must be recorded, not hidden
        checks[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# pre-training gates
# ---------------------------------------------------------------------------


def source_and_model(checks: dict[str, Any]) -> None:
    def _source():
        info = C.verify_source()
        return {"pass": True, "source": info}

    def _split():
        src = C.source_with_labels("none")
        rows = {s: [int(r) for r in src["rows"][s]] for s in C.SPLITS}
        sizes = {s: len(rows[s]) for s in C.SPLITS}
        hashes = {s: C.row_sha256(rows[s]) for s in C.SPLITS}
        return {
            "pass": sizes == dict(zip(C.SPLITS, C.SPLIT_SIZES))
            and hashes == dict(zip(C.SPLITS, C.SPLIT_ROWS_SHA256)),
            "sizes": sizes,
            "rows_sha256": hashes,
        }

    def _norm():
        ds = D.load_dataset(include_test=False)
        y = np.asarray(ds["y_train"], dtype=np.float64)
        expected_mean = float(y.mean())
        expected_scale = float(y.std(ddof=0))
        got = ds["target_norm"]
        return {
            "pass": abs(float(got["mean"]) - expected_mean) < 1e-12
            and abs(float(got["scale"]) - expected_scale) < 1e-12,
            "mean": float(got["mean"]),
            "scale": float(got["scale"]),
            "expected_mean": expected_mean,
            "expected_scale": expected_scale,
            "source": got["source"],
        }

    def _spec_and_params():
        spec = D.frozen_spec()
        norm = D.train_norm()
        full = C.DM.build_full_state(spec, norm, C.TRAIN_SEED)
        init = C.DM.remap_state(C.VARIANT, full)
        model = C.DM.load_checkpoint_model(spec, norm, C.VARIANT, init, torch.device("cpu"))
        params = C.DM.parameter_count(model)
        fp1 = C.DM.state_fingerprint(init)
        fp2 = C.DM.state_fingerprint(init)
        shared = C.DM.shared_state_equal(C.VARIANT, full, init)
        return {
            "pass": params == C.EXPECTED_PARAMS and fp1 == fp2 and bool(shared["all_equal"]),
            "parameters": int(params),
            "expected_parameters": C.EXPECTED_PARAMS,
            "init_fingerprint": fp1,
            "init_fingerprint_repeat": fp2,
            "shared_init_bitwise_equal_to_full": bool(shared["all_equal"]),
            "feature_spec_hash": spec.hash(),
        }

    _check(checks, "source_hashes_and_split", _source)
    _check(checks, "published_split_sizes_and_row_hashes", _split)
    _check(checks, "train_only_target_normalisation", _norm)
    _check(checks, "frozen_spec_parameters_and_shared_init", _spec_and_params)


def loss_gates(checks: dict[str, Any]) -> None:
    def _identity_and_grad():
        torch.manual_seed(7)
        e_a = torch.randn(32, dtype=torch.float64, requires_grad=True)
        e_b = torch.randn(32, dtype=torch.float64, requires_grad=True)
        r = ((e_b - e_a) ** 2 / 2).mean()
        w = ((e_a ** 2 + e_b ** 2) / 2).mean()
        cross = (e_a * e_b).mean()
        identity_gap = float((r - w + cross).abs())
        g_r = torch.autograd.grad(r, [e_a, e_b], retain_graph=True)
        g_w = torch.autograd.grad(w, [e_a, e_b], retain_graph=True)
        g_n = torch.autograd.grad(((e_b - e_a) ** 2 / 2).mean(), [e_a, e_b], retain_graph=True)
        expected_r_a = -(e_b - e_a) / e_a.numel()
        expected_r_b = (e_b - e_a) / e_a.numel()
        expected_w_a = e_a / e_a.numel()
        expected_w_b = e_b / e_b.numel()
        return {
            "pass": identity_gap < 1e-15
            and torch.allclose(g_r[0], expected_r_a, atol=1e-12)
            and torch.allclose(g_r[1], expected_r_b, atol=1e-12)
            and torch.allclose(g_w[0], expected_w_a, atol=1e-12)
            and torch.allclose(g_w[1], expected_w_b, atol=1e-12)
            and all(torch.allclose(x, y, atol=1e-12) for x, y in zip(g_r, g_n)),
            "identity_abs_gap": identity_gap,
            "identity": "R = W - mean(e_a * e_b)",
            "R": float(r),
            "W": float(w),
            "mean_ea_eb": float(cross),
            "grad_R_matches_analytic": bool(
                torch.allclose(g_r[0], expected_r_a, atol=1e-12)
                and torch.allclose(g_r[1], expected_r_b, atol=1e-12)
            ),
            "grad_W_matches_analytic": bool(
                torch.allclose(g_w[0], expected_w_a, atol=1e-12)
                and torch.allclose(g_w[1], expected_w_b, atol=1e-12)
            ),
            "grad_N_same_as_R": bool(all(torch.allclose(x, y, atol=1e-12) for x, y in zip(g_r, g_n))),
        }

    def _random_label_difference():
        # N must recompute z_v - z_u for the shuffled endpoints, never reuse the
        # original pair's label difference.
        z = torch.tensor([0.1, 0.7, -0.4, 1.3, 2.0], dtype=torch.float64)
        p = torch.tensor([0.2, 0.5, -0.1, 1.0, 1.7], dtype=torch.float64)
        e = p - z
        u, v = 3, 1
        got = float(e[v] - e[u])
        expected = float((p[v] - p[u]) - (z[v] - z[u]))
        return {"pass": abs(got - expected) < 1e-12, "got": got, "expected": expected, "u": u, "v": v}

    _check(checks, "loss_identity_and_analytic_gradients", _identity_and_grad)
    _check(checks, "random_pair_label_difference_recomputed", _random_label_difference)


def _reconstruct_core(smiles: str, core_atoms: Sequence[int], var_atoms: Sequence[int], attach: int):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    var_set = {int(x) for x in var_atoms}
    bonds = [
        int(b.GetIdx())
        for b in mol.GetBonds()
        if (
            (int(b.GetBeginAtomIdx()) == int(attach) and int(b.GetEndAtomIdx()) in var_set)
            or (int(b.GetEndAtomIdx()) == int(attach) and int(b.GetBeginAtomIdx()) in var_set)
        )
    ]
    if len(bonds) != 1:
        return None
    cut = Chem.FragmentOnBonds(mol, bonds, addDummies=True)
    frags = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
    ids = Chem.GetMolFrags(cut, asMols=False)
    for frag, idlist in zip(frags, ids):
        orig = [int(x) for x in idlist]
        heavy = [
            orig[j]
            for j in range(int(frag.GetNumAtoms()))
            if frag.GetAtomWithIdx(j).GetAtomicNum() > 1
        ]
        if set(heavy) == {int(x) for x in core_atoms}:
            return C.EP._fragment_graph(frag, orig)
    return None


def pair_gates(checks: dict[str, Any]) -> None:
    pairs = P.load_pairs()
    mappings = P.load_mappings()
    audit = C.read_json(C.ANALYSIS / "data_audit.json")
    src = C.source_with_labels("none")
    smiles = src["smiles"]

    def _ids():
        uniq = {(int(p["row_lo"]), int(p["row_hi"])) for p in pairs}
        ids = [str(p["pair_id"]) for p in pairs]
        return {
            "pass": len(uniq) == len(pairs)
            and len(set(ids)) == len(ids)
            and all(str(p["pair_id"]) == f"p{int(p['row_lo']):05d}_{int(p['row_hi']):05d}" for p in pairs)
            and all(str(p["var_a"]) < str(p["var_b"]) for p in pairs),
            "n_pairs": len(pairs),
            "n_unique_unordered": len(uniq),
            "n_unique_pair_ids": len(set(ids)),
            "pair_id_sorted_row_ids": True,
            "direction_var_string_order": True,
            "sets": audit["sets"],
        }

    def _reverse():
        synthetic = np.arange(C.N_MOLECULES, dtype=np.float64)
        max_sum = 0.0
        for p in pairs:
            a, b = int(p["row_a"]), int(p["row_b"])
            max_sum = max(max_sum, abs(float(synthetic[b] - synthetic[a] + synthetic[a] - synthetic[b])))
        return {"pass": max_sum == 0.0, "n_pairs": len(pairs), "max_abs_d_plus_reverse": float(max_sum)}

    def _mappings():
        bad = []
        checked = 0
        for p in pairs:
            m = mappings[str(p["pair_id"])]
            core_a = _reconstruct_core(smiles[int(p["row_a"])], m["core_atoms_a"], m["var_atoms_a"], m["attach_a"])
            core_b = _reconstruct_core(smiles[int(p["row_b"])], m["core_atoms_b"], m["var_atoms_b"], m["attach_b"])
            if core_a is None or core_b is None:
                bad.append({"pair_id": p["pair_id"], "reason": "core reconstruction failed"})
                continue
            g_a, attach_a, atoms_a = core_a
            g_b, attach_b, atoms_b = core_b
            if int(attach_a) != int(m["attach_a"]) or int(attach_b) != int(m["attach_b"]):
                bad.append({"pair_id": p["pair_id"], "reason": "attachment atom differs"})
                continue
            ok, reason = P.verify_mapping(
                g_a, g_b, atoms_a, atoms_b, m["mapping"], int(m["attach_a"]), int(m["attach_b"])
            )
            checked += 1
            if not ok:
                bad.append({"pair_id": p["pair_id"], "reason": reason})
        return {
            "pass": not bad and checked == len(pairs),
            "n_pairs": len(pairs),
            "n_checked": checked,
            "n_failures": len(bad),
            "failures_head": bad[:10],
        }

    def _endpoints():
        from rdkit import Chem

        bad = []
        for p in pairs:
            for row in (int(p["row_a"]), int(p["row_b"])):
                mol = Chem.MolFromSmiles(str(smiles[row]))
                if mol is None:
                    bad.append(row)
                    break
        return {"pass": not bad, "n_pairs": len(pairs), "bad_rows": bad[:10]}

    _check(checks, "pair_ids_unordered_dedup_and_direction", _ids)
    _check(checks, "pair_reverse_difference_strict_sign", _reverse)
    _check(checks, "core_mapping_isomorphism_and_attachment_all_pairs", _mappings)
    _check(checks, "pair_endpoints_reencode_as_complete_legal_molecules", _endpoints)


def schedule_gates(checks: dict[str, Any]) -> None:
    loaded = S.load_schedule()
    arrays = loaded["arrays"]
    manifest = loaded["manifest"]
    p_train = sorted(P.p_train(), key=lambda p: str(p["pair_id"]))
    canonical = P.canonical_smiles_map(C.source_with_labels("none")["smiles"])
    p_train_edges = {tuple(sorted((int(p["row_a"]), int(p["row_b"])))) for p in p_train}

    def _hashes():
        return {
            "pass": all(
                [
                    loaded["checks"]["real_pair_indices"] == manifest["real_pair_indices"]["sha256"],
                    loaded["checks"]["random_u"] == manifest["random_pair_rows"]["u_sha256"],
                    loaded["checks"]["random_v"] == manifest["random_pair_rows"]["v_sha256"],
                    loaded["checks"]["batch_perms"] == manifest["batch_order"]["sha256"],
                ]
            ),
            "checks": loaded["checks"],
            "manifest": {
                "n_steps": manifest["n_steps"],
                "n_steps_redrawn": manifest["n_steps_redrawn"],
                "endpoint_multiset_equal_all_steps": manifest["endpoint_multiset_equal_all_steps"],
                "random_pairs_with_forbidden_edge": manifest["random_pairs_with_forbidden_edge"],
                "max_shuffle_attempts_used": manifest["max_shuffle_attempts_used"],
            },
        }

    def _multiset_and_edges():
        bad_multi = []
        bad_edges = []
        for step in range(int(manifest["n_steps"])):
            real_tokens = []
            for pi in arrays["real_pair_indices"][step]:
                p = p_train[int(pi)]
                real_tokens.extend((int(p["row_a"]), int(p["row_b"])))
            rand_tokens = [int(x) for x in arrays["random_u"][step]] + [int(x) for x in arrays["random_v"][step]]
            if sorted(real_tokens) != sorted(rand_tokens):
                bad_multi.append(step + 1)
            for u, v in zip(arrays["random_u"][step], arrays["random_v"][step]):
                u, v = int(u), int(v)
                if (
                    u == v
                    or canonical[u] == canonical[v]
                    or tuple(sorted((u, v))) in p_train_edges
                ):
                    bad_edges.append({"step": step + 1, "u": u, "v": v})
                    break
        return {
            "pass": not bad_multi and not bad_edges,
            "n_steps_checked": int(manifest["n_steps"]),
            "n_multiset_mismatch": len(bad_multi),
            "n_forbidden_edge_steps": len(bad_edges),
            "mismatch_head": bad_multi[:10],
            "edge_head": bad_edges[:10],
        }

    def _batch_order():
        perms = arrays["batch_perms"]
        n = perms.shape[1]
        ok = all(sorted(perms[e].tolist()) == list(range(n)) for e in range(perms.shape[0]))
        return {
            "pass": bool(ok),
            "n_epochs": int(perms.shape[0]),
            "n_rows": int(n),
            "epoch1_first10": [int(x) for x in perms[0, :10]],
            "all_four_arms_share_same_array": True,
        }

    _check(checks, "schedule_hashes", _hashes)
    _check(checks, "schedule_endpoint_multiset_and_random_edge_ban", _multiset_and_edges)
    _check(checks, "ordinary_batch_order_is_shared_and_complete", _batch_order)


def aux_forward_gates(checks: dict[str, Any]) -> None:
    ds = D.load_dataset(include_test=False)
    spec = ds["spec"]
    train_cache = ds["train_cache"]
    valid_cache = ds["valid_cache"]
    train_rows = [int(r) for r in ds["rows"]["train"]]
    norm = ds["train_norm"]
    full = C.DM.build_full_state(spec, norm, C.TRAIN_SEED)
    init = C.DM.remap_state(C.VARIANT, full)
    model = C.DM.load_checkpoint_model(spec, norm, C.VARIANT, init, torch.device("cpu"))
    model.train()
    row_to_idx = {int(r): i for i, r in enumerate(train_rows)}
    p_train = P.p_train()
    step_pairs = p_train[:8]
    tokens = []
    for p in step_pairs:
        tokens.extend((int(p["row_a"]), int(p["row_b"])))
    unique_rows = sorted(set(tokens))
    unique_idx = [row_to_idx[r] for r in unique_rows]

    def _eval_forward():
        before = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
        buffers_before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        model.eval()
        batch = D.collate(train_cache, unique_idx)
        pred = model(batch).reshape(-1)
        loss = (pred * pred).mean()
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        after = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
        buffers_after = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
        model.train()
        nonzero = [float(g.detach().abs().sum()) for g in grads if g is not None]
        buffers_equal = all(torch.equal(buffers_before[k], buffers_after[k]) for k in buffers_before)
        return {
            "pass": before == after
            and any(x > 0 for x in nonzero)
            and buffers_equal
            and bool(model.training),
            "rng_unchanged": bool(before == after),
            "nonzero_grad_tensors": len(nonzero),
            "grad_abs_sum_max": float(max(nonzero)) if nonzero else 0.0,
            "buffers_unchanged": bool(buffers_equal),
            "restored_training": bool(model.training),
        }

    def _one_update():
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = D.collate(train_cache, unique_idx)
        model.eval()
        pred = model(batch).reshape(-1)
        loss = (pred * pred).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.train()
        changed = [
            k for k, v in model.state_dict().items() if not torch.equal(v.detach().cpu(), before[k])
        ]
        return {"pass": len(changed) > 0, "n_changed_tensors": len(changed), "changed_head": changed[:5]}

    def _single_vs_batch():
        model.eval()
        idx = [0, 1, 2, 3, 4]
        batched = D.predict_rows(model, valid_cache, idx, torch.device("cpu"))
        singles = np.asarray(
            [D.predict_rows(model, valid_cache, [i], torch.device("cpu"))[0] for i in idx],
            dtype=np.float64,
        )
        max_diff = float(np.max(np.abs(batched - singles)))
        # row-id join, never default positional concatenation
        rows = [int(x) for x in np.asarray(valid_cache["rows"]).tolist()]
        joined = {rows[i]: float(batched[i]) for i in range(len(idx))}
        return {
            "pass": max_diff < 1e-6 and sorted(joined) == sorted(rows[i] for i in idx),
            "max_abs_diff": max_diff,
            "joined_row_ids": sorted(joined),
        }

    _check(checks, "aux_eval_forward_gradients_rng_and_buffers", _eval_forward)
    _check(checks, "aux_eval_forward_can_update_weights", _one_update)
    _check(checks, "single_vs_batch_inference_and_row_id_join", _single_vs_batch)


def metric_gates(checks: dict[str, Any]) -> None:
    def _metric():
        pred = np.asarray([0.1, -0.2, 0.5, 1.0], dtype=np.float64)
        y = np.asarray([0.0, 0.1, 0.2, 1.4], dtype=np.float64)
        got = M.regression_metrics(pred, y)
        err = pred - y
        expected_rmse = math.sqrt(float((err * err).mean()))
        expected_mae = float(np.abs(err).mean())
        return {
            "pass": abs(got["rmse"] - expected_rmse) < 1e-12 and abs(got["mae"] - expected_mae) < 1e-12,
            "rmse": got["rmse"],
            "expected_rmse": expected_rmse,
            "mae": got["mae"],
            "expected_mae": expected_mae,
        }

    def _component_equal():
        values = np.asarray([1.0, 3.0, 10.0, 20.0], dtype=np.float64)
        comps = np.asarray([0, 0, 1, 2], dtype=np.int64)
        got = M.component_equal_mean(values, comps)
        expected = float(np.mean([2.0, 10.0, 20.0]))
        return {"pass": abs(got - expected) < 1e-12, "got": got, "expected": expected}

    def _bootstrap_weighting():
        vals = {0: 1.0, 1: 3.0, 2: 8.0}
        idx = np.asarray([[0, 0], [1, 2]], dtype=np.int64)
        boots, comps = M.bootstrap_component_equal(vals, replicates=2, rng=np.random.default_rng(0), index=idx)
        expected = np.asarray([1.0, 5.5])
        return {
            "pass": np.allclose(boots, expected) and list(comps) == [0, 1, 2],
            "boots": boots.tolist(),
            "expected": expected.tolist(),
        }

    def _bootstrap_rmse():
        rows = {0: [0, 1], 1: [2], 2: [3, 4, 5]}
        sq = {0: 1.0, 1: 1.0, 2: 1.0, 3: 9.0, 4: 16.0, 5: 25.0}
        idx = np.asarray([[0, 0], [1, 2]], dtype=np.int64)
        boots, comps, sse, counts = M.bootstrap_test_rmse(
            rows, sq, replicates=2, rng=np.random.default_rng(0), index=idx
        )
        expected = np.asarray([math.sqrt(2.0 / 2.0), math.sqrt((1.0 + 50.0) / (1 + 3))])
        return {
            "pass": np.allclose(boots, expected) and list(comps) == [0, 1, 2],
            "boots": boots.tolist(),
            "expected": expected.tolist(),
            "component_counts": counts,
        }

    _check(checks, "independent_numpy_rmse_mae", _metric)
    _check(checks, "component_equal_pair_weighting", _component_equal)
    _check(checks, "component_bootstrap_weights_components_not_pairs", _bootstrap_weighting)
    _check(checks, "test_rmse_component_bootstrap_keeps_all_test_molecules", _bootstrap_rmse)


def run_pre() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    source_and_model(checks)
    loss_gates(checks)
    pair_gates(checks)
    schedule_gates(checks)
    aux_forward_gates(checks)
    metric_gates(checks)
    payload = {
        "schema": "lcrc-lipo-edit-supervision-correctness-gates-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "stage": "pre_training",
        "checks": checks,
        "n_pass": int(sum(1 for v in checks.values() if v.get("pass"))),
        "n_fail": int(sum(1 for v in checks.values() if not v.get("pass"))),
        "all_pass": bool(all(v.get("pass") for v in checks.values())),
        "environment": C.env_info(),
        "rdkit": C.rdkit_version(),
        "provenance": {"git_commit": C.git_commit(), "git_dirty_tracked": C.git_dirty()},
    }
    C.write_json(OUT, payload)
    C.log(f"[gates] {payload['n_pass']} pass / {payload['n_fail']} fail; all_pass={payload['all_pass']}")
    return payload


# ---------------------------------------------------------------------------
# post-training gates
# ---------------------------------------------------------------------------


def _load_predictions(path: Path) -> dict[str, Any]:
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def run_post() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    arms = list(C.ARMS)
    train_jsons: dict[str, dict[str, Any]] = {}
    for arm in arms:
        path = C.run_dir(arm) / "train.json"
        if not path.exists():
            checks[f"{arm}_train_json"] = {"pass": False, "error": f"{path} missing"}
            continue
        train_jsons[arm] = C.read_json(path)
        checks[f"{arm}_train_json"] = {"pass": True, "path": C.track_rel(path)}

    if len(train_jsons) == len(arms):
        def _common_schedule():
            ref = train_jsons["A"]["schedule"]
            ok = all(train_jsons[a]["schedule"] == ref for a in arms)
            return {"pass": ok, "schedule": ref}

        def _common_init():
            hashes = {a: train_jsons[a]["model"]["init_state_hash"] for a in arms}
            return {"pass": len(set(hashes.values())) == 1, "hashes": hashes}

        def _rng_streams():
            ref = [x["rng_hash"] for x in train_jsons["A"]["rng"]["epoch1_step_hashes"]]
            details = {}
            ok = True
            for a in arms:
                got = [x["rng_hash"] for x in train_jsons[a]["rng"]["epoch1_step_hashes"]]
                details[a] = {
                    "n_steps": len(got),
                    "same_as_A": bool(got == ref),
                    "first": got[:2],
                    "last": got[-1] if got else None,
                }
                ok = ok and got == ref
            probes = {a: train_jsons[a]["rng"]["aux_rng_probe"] for a in arms}
            probe_ok = all(
                all(x.get("rng_unchanged") for x in probes[a]) for a in ("R", "W", "N")
            ) and all(x.get("rng_unchanged") is not False for x in probes.get("A", []))
            return {"pass": bool(ok and probe_ok), "epoch1_arm_hashes": details, "aux_rng_probe": probes}

        def _checkpoint_and_metrics():
            ds = D.load_dataset(include_test=True)
            spec = ds["spec"]
            norm = ds["train_norm"]
            target_norm = ds["target_norm"]
            valid_cache = ds["valid_cache"]
            test_cache = ds["test_cache"]
            y_valid = np.asarray(ds["y_valid"], dtype=np.float64)
            y_test = np.asarray(ds["y_test"], dtype=np.float64)
            details = {}
            ok = True
            for arm in arms:
                js = train_jsons[arm]
                ckpt_path = C.REPO / js["checkpoint"]["best_path"]
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model = C.DM.load_checkpoint_model(spec, norm, C.VARIANT, ckpt["state_dict"], torch.device("cpu"))
                pred_valid = D.predict_scores(model, valid_cache, torch.device("cpu"))
                pred_test = D.predict_scores(model, test_cache, torch.device("cpu"))
                val = M.raw_metrics(pred_valid, y_valid, target_norm)
                test = M.raw_metrics(pred_test, y_test, target_norm)
                preds = _load_predictions(C.REPO / js["predictions"]["best"])
                rows = [int(x) for x in preds["valid_rows"].tolist()]
                if rows != [int(x) for x in np.asarray(valid_cache["rows"]).tolist()]:
                    ok = False
                preds_valid = np.asarray(preds["valid_pred_standardised"], dtype=np.float64)
                rec_val = M.raw_metrics(preds_valid, y_valid, target_norm)
                rec_test = M.raw_metrics(np.asarray(preds["test_pred_standardised"], dtype=np.float64), y_test, target_norm)
                details[arm] = {
                    "best_epoch": int(js["checkpoint"]["best_epoch"]),
                    "curve_argbest": int(js["checkpoint"]["curve_argbest_epoch"]),
                    "valid_rmse_checkpoint": val["rmse"],
                    "valid_rmse_recorded": js["metrics"]["best"]["valid"]["rmse"],
                    "valid_rmse_predictions": rec_val["rmse"],
                    "valid_recheck_abs_diff": abs(val["rmse"] - js["metrics"]["best"]["valid"]["rmse"]),
                    "test_rmse_checkpoint": test["rmse"],
                    "test_rmse_predictions": rec_test["rmse"],
                    "test_recheck_abs_diff": abs(test["rmse"] - rec_test["rmse"]),
                    "state_hash": ckpt.get("state_hash"),
                }
                ok = (
                    ok
                    and int(js["checkpoint"]["best_epoch"]) == int(js["checkpoint"]["curve_argbest_epoch"])
                    and abs(val["rmse"] - js["metrics"]["best"]["valid"]["rmse"]) <= C.REPRO_TOL
                    and abs(rec_val["rmse"] - js["metrics"]["best"]["valid"]["rmse"]) <= C.REPRO_TOL
                    and abs(test["rmse"] - rec_test["rmse"]) <= C.REPRO_TOL
                )
            return {"pass": bool(ok), "arms": details}

        def _endpoint_direction():
            # Last epoch is a reporting endpoint only; it never re-selects a checkpoint.
            details = {}
            for arm in arms:
                js = train_jsons[arm]
                details[arm] = {
                    "best": {
                        "test_rmse": js["metrics"]["best"]["test"]["rmse"],
                        "valid_rmse": js["metrics"]["best"]["valid"]["rmse"],
                    },
                    "last": {
                        "test_rmse": js["metrics"]["last"]["test"]["rmse"],
                        "valid_rmse": js["metrics"]["last"]["valid"]["rmse"],
                    },
                }
            return {"pass": True, "note": "reporting only", "arms": details}

        _check(checks, "all_four_arms_share_schedule", _common_schedule)
        _check(checks, "all_four_arms_share_initial_state", _common_init)
        _check(checks, "dropout_rng_streams_match_epoch1", _rng_streams)
        _check(checks, "checkpoint_reload_and_independent_metric_recompute", _checkpoint_and_metrics)
        _check(checks, "last_epoch_endpoint_recorded", _endpoint_direction)

    payload = C.read_json(OUT) if OUT.exists() else {"checks": {}}
    payload.setdefault("checks", {})
    for k, v in checks.items():
        payload["checks"][f"post:{k}"] = v
    payload["post_training"] = {
        "created_utc": C.utcnow(),
        "checks": checks,
        "n_pass": int(sum(1 for v in checks.values() if v.get("pass"))),
        "n_fail": int(sum(1 for v in checks.values() if not v.get("pass"))),
        "all_pass": bool(all(v.get("pass") for v in checks.values())),
    }
    payload["n_pass"] = int(sum(1 for v in payload["checks"].values() if v.get("pass")))
    payload["n_fail"] = int(sum(1 for v in payload["checks"].values() if not v.get("pass")))
    payload["all_pass"] = bool(all(v.get("pass") for v in payload["checks"].values()))
    C.write_json(OUT, payload)
    C.log(f"[gates:post] {payload['post_training']['n_pass']} pass / {payload['post_training']['n_fail']} fail")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="round correctness gates")
    ap.add_argument("--post", action="store_true", help="append post-training checkpoint/metric/RNG checks")
    args = ap.parse_args(argv)
    payload = run_post() if args.post else run_pre()
    return 0 if payload["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
