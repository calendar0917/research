"""CPU correctness gates for the S2 count-mechanism round.

The script writes ``analysis/correctness_gates.json``.  It must pass before any
formal GPU training is launched.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import pc_common as C

GUARD = C.install_test_access_guard()

import pc_data as PD  # noqa: E402
import pc_p1_model as PM  # noqa: E402
import pc_s2 as S  # noqa: E402
from bp_data import collate  # noqa: E402


def hand_graph_gate() -> dict:
    atom_raw = np.zeros((4, 9), dtype=np.int64)
    atom_raw[:, 0] = [5, 6, 7, 8]
    bond_index = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    bond_raw = np.zeros((3, 3), dtype=np.int64)
    bond_raw[:, 0] = [0, 1, 2]
    paths = S.enumerate_paths(atom_raw, bond_raw, bond_index)
    expected = {(5, 0, 6, 1, 7), (6, 1, 7, 2, 8), (7, 1, 6, 0, 5), (8, 2, 7, 1, 6)}
    got = {tuple(int(x) for x in row) for row in S.type_keys(atom_raw, bond_raw, paths).tolist()}
    chain_ok = paths["n_paths"] == 4 and got == expected
    # Triangle: every 2-hop path has endpoint != root; 2 paths per root.
    t_atom = np.zeros((3, 9), dtype=np.int64)
    t_atom[:, 0] = [6, 6, 6]
    t_bonds = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
    t_raw = np.zeros((3, 3), dtype=np.int64)
    t_paths = S.enumerate_paths(t_atom, t_raw, t_bonds)
    triangle_ok = t_paths["n_paths"] == 6 and bool(np.all(t_paths["end"] != t_paths["root"]))
    # FULL key layout: root 9, bond1 3, middle 9, bond2 3, endpoint 9.
    full = S.full_keys(atom_raw, bond_raw, paths)
    layout_ok = full.shape == (4, 33) and bool(
        np.array_equal(full[0, :9], atom_raw[0])
        and np.array_equal(full[0, 9:12], bond_raw[0])
        and np.array_equal(full[0, 12:21], atom_raw[1])
        and np.array_equal(full[0, 21:24], bond_raw[1])
        and np.array_equal(full[0, 24:], atom_raw[2])
    )
    return {
        "chain_paths": int(paths["n_paths"]),
        "chain_type_keys": sorted(got),
        "chain_expected": sorted(expected),
        "chain_ok": bool(chain_ok),
        "triangle_paths": int(t_paths["n_paths"]),
        "triangle_no_root_closure": bool(np.all(t_paths["end"] != t_paths["root"])),
        "triangle_ok": bool(triangle_ok),
        "full_key_layout_33": bool(layout_ok),
        "status": "PASS" if chain_ok and triangle_ok and layout_ok else "FAIL",
    }


def random_graph(rng: np.random.Generator, n: int = 10) -> dict:
    atom_raw = np.zeros((n, 9), dtype=np.int64)
    atom_raw[:, 0] = rng.choice([5, 6, 7, 8, 16], size=n)
    for c, k in enumerate(C.ATOM_VOCABS[1:], start=1):
        atom_raw[:, c] = rng.integers(0, k, size=n)
    pairs = set()
    for i in range(1, n):
        pairs.add((int(rng.integers(0, i)), i))
    while len(pairs) < 14:
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        pairs.add((int(a), int(b)))
    pairs = sorted(pairs)
    bond_index = np.asarray(pairs, dtype=np.int64)
    bond_raw = np.stack(
        [rng.integers(0, k, size=len(pairs)) for k in C.BOND_VOCABS], axis=1
    ).astype(np.int64)
    return {
        "gid": 0,
        "atom_raw": atom_raw,
        "bond_raw": bond_raw,
        "bond_index": bond_index,
        "n_paths_cache": S.enumerate_paths(atom_raw, bond_raw, bond_index)["n_paths"],
    }


def invariance_gate() -> dict:
    rng = np.random.default_rng(0)
    g = random_graph(rng)
    base = S.graph_features(g)
    n = g["atom_raw"].shape[0]
    perm = rng.permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    atom2 = g["atom_raw"][perm]
    pairs2, bond2 = [], []
    for (a, b), ba in zip(g["bond_index"].tolist(), g["bond_raw"].tolist()):
        x, y = int(inv[a]), int(inv[b])
        if x > y:
            x, y = y, x
        pairs2.append((x, y))
        bond2.append(ba)
    order = np.lexsort((np.asarray(pairs2)[:, 1], np.asarray(pairs2)[:, 0]))
    pairs2 = [pairs2[i] for i in order]
    bond2 = np.asarray(bond2, dtype=np.int64)[order]
    g2 = {
        "gid": 0,
        "atom_raw": atom2,
        "bond_raw": bond2,
        "bond_index": np.asarray(pairs2, dtype=np.int64),
        "n_paths_cache": base["n_paths"],
    }
    relabeled = S.graph_features(g2)
    order = rng.permutation(g["bond_index"].shape[0])
    g3 = {
        "gid": 0,
        "atom_raw": g["atom_raw"],
        "bond_raw": g["bond_raw"][order],
        "bond_index": g["bond_index"][order],
        "n_paths_cache": base["n_paths"],
    }
    shuffled = S.graph_features(g3)
    checks = {}
    for key in ("counts_type", "counts_full", "counts_null"):
        checks[f"relabel_{key}"] = bool(np.array_equal(base[key], relabeled[key]))
        checks[f"bond_order_{key}"] = bool(np.array_equal(base[key], shuffled[key]))
    for key in ("counts_type", "counts_full", "counts_null"):
        checks[f"relabel_{key}_max_abs_diff"] = float(np.abs(base[key] - relabeled[key]).max())
        checks[f"bond_order_{key}_max_abs_diff"] = float(np.abs(base[key] - shuffled[key]).max())
    return {
        "n_atoms": int(n),
        "n_paths": int(base["n_paths"]),
        "checks": checks,
        "status": "PASS" if all(v for k, v in checks.items() if not k.endswith("_max_abs_diff")) else "FAIL",
    }


def full_null_gate() -> dict:
    rng = np.random.default_rng(1)
    graphs = [random_graph(rng, n=12) for _ in range(3)]
    rows = []
    ok = True
    for g in graphs:
        f = S.graph_features(g)
        full_keys = S.full_keys(g["atom_raw"], g["bond_raw"], S.enumerate_paths(g["atom_raw"], g["bond_raw"], g["bond_index"]))
        order = S.canonical_path_order(full_keys)
        full_keys = full_keys[order]
        first, second = S.split_halves(full_keys)
        seed = S.canonical_full_seed(full_keys)
        nkeys, perm = S.null_keys(first, second, seed)
        first_ok = bool(np.array_equal(first, nkeys[:, :21]))
        second_ok = bool(np.array_equal(np.sort(second, axis=0), np.sort(second[perm], axis=0)))
        rows_ok = bool(np.all(perm != np.arange(g["n_paths_cache"]))) if g["n_paths_cache"] > 1 else True
        total_ok = int(full_keys.shape[0]) == int(nkeys.shape[0]) == int(f["n_paths"])
        joint_different = int(np.any(full_keys != nkeys, axis=1).sum())
        vector_different = bool(np.any(f["counts_full"] != f["counts_null"]))
        ok = ok and first_ok and second_ok and rows_ok and total_ok and vector_different
        rows.append(
            {
                "n_paths": int(f["n_paths"]),
                "first_marginal_equal": first_ok,
                "second_marginal_equal": second_ok,
                "derangement": rows_ok,
                "total_equal": total_ok,
                "paths_with_different_joint_key": joint_different,
                "bucket_vectors_differ": vector_different,
            }
        )
    return {
        "graphs": rows,
        "status": "PASS" if ok else "FAIL",
        "rule": "canonical lexicographic FULL-key path order; per-graph PCG64 derangement of second halves; identity only for one path",
    }


def hash_cross_process_gate() -> dict:
    keys = [bytes(((i * 37 + j * 11) % 256) for j in range(33)) for i in range(4)]
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(C.CODE_DIR)!r}); "
        "import pc_common as C; "
        f"keys = {[k.hex() for k in keys]!r}; "
        "out = [[k, C.stable_digest_int(bytes.fromhex(k)), C.stable_bucket(bytes.fromhex(k))] for k in keys]; "
        "print(json.dumps(out, sort_keys=True))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(C.CODE_DIR)
    runs = [
        subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
        for _ in range(2)
    ]
    local = [[k.hex(), C.stable_digest_int(k), C.stable_bucket(k)] for k in keys]
    local_json = json.dumps(local, sort_keys=True)
    consistent = runs[0] == runs[1] == local_json
    return {
        "keys": [k.hex() for k in keys],
        "local": local,
        "subprocess_runs_equal": bool(runs[0] == runs[1]),
        "matches_local": bool(runs[0] == local_json),
        "status": "PASS" if consistent else "FAIL",
    }


def model_gate() -> dict:
    task = PD.load_c1_task()
    models: dict[str, object] = {}
    infos: dict[str, dict] = {}
    for arm in ("A0", "A1", "A2", "A3"):
        models[arm], infos[arm] = PM.build_arm(task, arm, 0, "cpu")
    base_state = models["A0"].state_dict()
    common_init = {}
    for arm in ("A1", "A2", "A3"):
        st = models[arm].base.state_dict()
        common_init[arm] = bool(all(torch_equal(st[k], base_state[k]) for k in base_state))
    rows = task["valid_rows"][:64]
    batch = collate(task["valid_cache"], rows.tolist())
    y = np.asarray(task["y_valid"][rows], dtype=np.float32)
    branch_grad = {}
    init_delta = {}
    for arm in ("A1", "A2", "A3"):
        counts = PM.load_counts(PM.ARM_SPECS[arm]["counts"])["valid"]
        cb = torch_tensor(counts[rows])
        models[arm].eval()
        with_no_grad = getattr(__import__("torch"), "no_grad")
        with with_no_grad():
            base_logits = models[arm].base(batch)
            full_logits = models[arm](batch, cb)
        init_delta[arm] = float((full_logits - base_logits).abs().max().item())
        models[arm].train()
        logits = models[arm](batch, cb)
        loss = torch_bce(logits, torch_tensor(y))
        loss.backward()
        grad = models[arm].count_linear.weight.grad
        branch_grad[arm] = {
            "norm": float(grad.norm().item()) if grad is not None else 0.0,
            "finite": bool(grad is not None and bool(__import__("torch").isfinite(grad).all().item())),
            "nonzero": bool(grad is not None and float(grad.norm().item()) > 0.0),
        }
    # Single-graph vs batched forward for A0 and A2.
    single = {}
    for arm in ("A0", "A2"):
        counts = PM.load_counts(PM.ARM_SPECS[arm]["counts"])["valid"] if arm == "A2" else None
        model = models[arm].eval()
        single_rows = task["valid_rows"][:5]
        singles = []
        for r in single_rows:
            b = collate(task["valid_cache"], [int(r)])
            cb = torch_tensor(counts[[r]]) if counts is not None else None
            with __import__("torch").no_grad():
                singles.append(model(b, cb) if cb is not None else model(b))
        batched_b = collate(task["valid_cache"], [int(r) for r in single_rows])
        cb = torch_tensor(counts[single_rows]) if counts is not None else None
        with __import__("torch").no_grad():
            batched = model(batched_b, cb) if cb is not None else model(batched_b)
        diff = float((batched - __import__("torch").cat(singles, dim=0)).abs().max().item())
        single[arm] = {"max_abs_diff": diff, "ok": bool(diff <= 1e-5)}
    return {
        "parameters": {arm: int(infos[arm]["total_parameters"]) for arm in infos},
        "base_parameters": {arm: int(infos[arm]["base_parameters"]) for arm in infos},
        "branch_parameters": {arm: int(infos[arm]["branch_parameters"]) for arm in infos},
        "base_init_matches_history": {arm: bool(infos[arm]["base_init_matches_history"]) for arm in infos},
        "common_base_init_equal": common_init,
        "branch_init_zero": {
            arm: bool(infos[arm]["branch_init_zero"]) for arm in ("A1", "A2", "A3")
        },
        "init_output_delta": init_delta,
        "first_step_branch_grad": branch_grad,
        "single_vs_batch": single,
        "status": "PASS"
        if (
            all(v for v in common_init.values())
            and all(v == 0.0 for v in init_delta.values())
            and all(v["finite"] and v["nonzero"] for v in branch_grad.values())
            and all(v["ok"] for v in single.values())
        )
        else "FAIL",
    }


def torch_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


def torch_tensor(x) -> object:
    import torch

    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def torch_bce(logits, y):
    import torch.nn.functional as F

    return F.binary_cross_entropy_with_logits(logits, y)


def smoke_reload_gate() -> dict:
    rows = {}
    ok = True
    for arm in ("A0", "A1", "A2", "A3"):
        path = C.RUNS / f"smoke_{arm}_cpu" / "run.json"
        if not path.is_file():
            rows[arm] = {"available": False}
            ok = False
            continue
        run = C.read_json(path)
        pred_path = C.RUNS / f"smoke_{arm}_cpu" / "predictions.npz"
        with np.load(pred_path) as z:
            y = np.asarray(z["y_valid"], dtype=np.float64)
            p = np.asarray(z["valid_probs"], dtype=np.float64)
        from sklearn.metrics import average_precision_score

        aps = [float(average_precision_score(y[:, j], p[:, j])) for j in range(C.N_LABELS)]
        macro = float(np.mean(aps))
        delta = abs(macro - float(run["valid_macro_ap"]))
        finite = bool(np.isfinite(p).all())
        rows[arm] = {
            "available": True,
            "status": run["status"],
            "reload_delta": float(run["reload_delta"]),
            "independent_ap": macro,
            "stored_ap": float(run["valid_macro_ap"]),
            "independent_ap_delta": delta,
            "finite_probabilities": finite,
            "first_epoch_rows_sha256": run["first_epoch_rows_sha256"],
            "label_hashes": run["label_hashes"],
            "branch_post_gate": run.get("branch_post_gate"),
        }
        ok = ok and run["status"] == "complete_no_test" and run["reload_delta"] <= 1e-4 and delta <= 1e-6 and finite
    batch_orders = {v.get("first_epoch_rows_sha256") for v in rows.values() if v.get("available")}
    labels = {json.dumps(v.get("label_hashes"), sort_keys=True) for v in rows.values() if v.get("available")}
    return {
        "runs": rows,
        "same_first_epoch_batch_order": len(batch_orders) == 1,
        "same_labels": len(labels) == 1,
        "status": "PASS" if ok and len(batch_orders) == 1 and len(labels) == 1 else "FAIL",
    }


def artifact_gate() -> dict:
    features = C.read_json(C.ANALYSIS / "p0_features_manifest.json")
    ecfp = C.read_json(C.ANALYSIS / "ecfp_validation.json")
    lgbm = C.read_json(C.ANALYSIS / "p0_lgbm.json")
    mlp = C.read_json(C.ANALYSIS / "p0_mlp.json")
    train = features["splits"]["train"]
    valid = features["splits"]["valid"]
    checks = {
        "feature_train_graphs_10873": train["n_graphs"] == 10873,
        "feature_valid_graphs_2331": valid["n_graphs"] == 2331,
        "feature_train_paths_4492788": train["n_paths_total"] == 4492788,
        "feature_valid_paths_949954": valid["n_paths_total"] == 949954,
        "feature_null_total_equal_train": train["n_paths_total"] == train["null_audit"]["n_paths_total_null"],
        "feature_null_total_equal_valid": valid["n_paths_total"] == valid["null_audit"]["n_paths_total_null"],
        "feature_null_marginal_failures_zero": train["null_audit"]["n_graphs_with_second_half_marginal_failure"] == 0
        and valid["null_audit"]["n_graphs_with_second_half_marginal_failure"] == 0,
        "feature_null_derangement_failures_zero": train["null_audit"]["n_graphs_with_derangement_failure"] == 0
        and valid["null_audit"]["n_graphs_with_derangement_failure"] == 0,
        "feature_null_bucket_vectors_differ_all_graphs": train["null_audit"]["n_graphs_full_vs_null_bucket_vectors_equal"] == 0
        and valid["null_audit"]["n_graphs_full_vs_null_bucket_vectors_equal"] == 0,
        "feature_labels_not_used": features.get("labels_used") is False,
        "ecfp_reproduction_ok": bool(ecfp["reproduction_ok"]),
        "ecfp_macro_ap_within_0.01pp": abs(ecfp["valid_macro_ap_pct"] - 72.98) <= 0.01,
        "ecfp_nnz_train_1105722": ecfp["fingerprint"]["nnz_train"] == 1105722,
        "ecfp_nnz_valid_235742": ecfp["fingerprint"]["nnz_valid"] == 235742,
        "ecfp_no_test_rows_parsed": ecfp["mixed_csv_access"]["test_rows_parsed"] == 0,
        "ecfp_no_test_labels_read": ecfp["mixed_csv_access"]["test_labels_read"] is False,
        "lgbm_three_features": set(lgbm["results"]) == {"type", "full", "full_binary"},
        "mlp_status_complete": mlp["status"] == "complete_no_test",
        "mlp_reload_delta": mlp["reload_delta"] <= 1e-6,
        "no_test_attempts_features": not features["test_access"]["attempts"],
        "no_test_attempts_ecfp": not ecfp["test_access"]["attempts"],
        "no_test_attempts_lgbm": not lgbm["test_access"]["attempts"],
        "no_test_attempts_mlp": not mlp["test_access"]["attempts"],
        "test_evaluated_false": lgbm["test_evaluated"] is False and mlp["test_evaluated"] is False,
    }
    return {
        "checks": checks,
        "p0_lgbm_valid_macro_ap_pct": {
            k: v["valid_macro_ap_pct"] for k, v in lgbm["results"].items()
        },
        "ecfp_valid_macro_ap_pct": ecfp["valid_macro_ap_pct"],
        "mlp_valid_macro_ap_pct": mlp["valid_macro_ap_pct"],
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    t_start = time.perf_counter()
    result: dict = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "entry_point": "code/pc_gates.py",
        "gates": {},
        "environment": C.environment(),
    }
    result["gates"]["hand_graph_s2"] = hand_graph_gate()
    result["gates"]["invariance"] = invariance_gate()
    result["gates"]["full_null"] = full_null_gate()
    result["gates"]["hash_cross_process"] = hash_cross_process_gate()
    result["gates"]["row_identity_train"] = PD.row_identity("train")
    result["gates"]["row_identity_valid"] = PD.row_identity("valid")
    result["gates"]["model"] = model_gate()
    result["gates"]["smoke_checkpoint_reload"] = smoke_reload_gate()
    result["gates"]["artifacts"] = artifact_gate()
    failed = [name for name, gate in result["gates"].items() if gate.get("status") != "PASS"]
    result["failed_gates"] = failed
    result["status"] = "PASS" if not failed else "FAIL"
    result["guard"] = GUARD
    result["seconds"] = float(time.perf_counter() - t_start)
    C.write_json(C.ANALYSIS / "correctness_gates.json", result)
    C.log(f"correctness gates: {result['status']} (failed={failed}) in {result['seconds']:.1f}s")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
