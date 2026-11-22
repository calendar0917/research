"""All 13 engineering checks, executed on the real computation path.

The test suite (`tests/test_acra.py`) is the primary executable form of these
checks; this script re-runs it and adds full-dataset coverage, arm-sharing and
gradient-coverage records.  Output: `analysis/checks.json`.

CPU only.  No Test labels are read.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K
import acra_model as M

ROUND = C.ROUND_DIR
TESTS = ROUND / "tests" / "test_acra.py"


def _item(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def full_dataset_coverage(cfg: Mapping[str, Any], raw: Mapping[str, Any],
                          pack: Mapping[str, Any]) -> dict[str, Any]:
    n_mol = int(pack["n_molecules"])
    atom_counts = np.diff(pack["atom_ptr"])
    bond_counts = np.diff(pack["ev_ptr"]) // 2
    raw_nodes = np.diff(np.asarray(raw["node_ptr"]))
    raw_bonds = np.diff(np.asarray(raw["bond_ptr"]))
    ev_ok = bool(np.all(np.diff(pack["ev_ptr"]) == 2 * raw_bonds))
    atom_ok = bool(np.array_equal(atom_counts, raw_nodes))
    feat_ok = bool(np.array_equal(pack["atom_feat"], np.asarray(raw["atom_feat"])))
    bond_counts_ok = bool(np.array_equal(bond_counts, raw_bonds))
    pair_ok = bool(np.array_equal(np.diff(pack["pair_ptr"]),
                                  bond_counts * (bond_counts - 1) // 2))
    ev_mol = np.repeat(np.arange(n_mol), np.diff(pack["ev_ptr"]))
    per_mol_bridges = np.bincount(ev_mol, weights=pack["ev_bridge"],
                                  minlength=n_mol).astype(np.int64) // 2
    n_bridge_events = int(pack["ev_bridge"].sum())
    n_ring_events = int(pack["ev_in_ring"].sum())
    n_mol_without_bridge = int(np.sum((bond_counts > 0) & (per_mol_bridges == 0)))
    empty_ev = int(np.sum(np.diff(pack["ev_ptr"]) == 0))
    max_mult = int(pack["ent_mult"].max(initial=0))
    return {
        "n_molecules": n_mol,
        "n_bonds": int(raw_bonds.sum()),
        "n_events": int(pack["ev_key_s"].size),
        "every_bond_has_two_events": ev_ok,
        "every_molecule_atom_count_matches_raw": atom_ok,
        "atom_attribute_table_equals_raw": feat_ok,
        "bond_count_per_molecule_matches_raw": bond_counts_ok,
        "bond_pair_table_covers_all_unordered_pairs": pair_ok,
        "n_bridge_directed_events": n_bridge_events,
        "n_ring_directed_events": n_ring_events,
        "n_molecules_with_no_bridge_bond": n_mol_without_bridge,
        "n_molecules_with_no_events": empty_ev,
        "max_frame_multiplicity": max_mult,
        "n_frame_entries": int(pack["ent_pat"].size),
        "n_frames_with_multiplicity": int(pack["blk_count"].sum()),
        "n_frame_patterns": int(pack["pat_k"].size),
        "unique_frame_compression": float(pack["ent_pat"].size)
        / max(1.0, float(pack["blk_count"].sum())),
    }


def pytest_run() -> dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        ["uv", "run", "pytest", "-q", str(TESTS)], cwd=str(C.REPO_ROOT),
        capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    return {
        "command": f"uv run pytest -q {TESTS.relative_to(C.REPO_ROOT)}",
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "output_tail": tail,
        "seconds": float(time.time() - t0),
    }


def gradient_coverage(cfg: Mapping[str, Any], raw: Mapping[str, Any],
                      dims, log=print) -> dict[str, Any]:
    import torch

    rows = list(range(24))
    pack = K.build_pack(cfg, raw, rows=rows, log=lambda _m: None)
    out: dict[str, Any] = {"arms": {}}
    # synthetic mechanics targets: no labels of any split are read here
    y = torch.sin(torch.arange(len(rows), dtype=torch.float32) * 0.7)
    for arm in M.ARMS:
        model = M.ACRAModel(cfg, dims[0], dims[1], arm)
        model.load_pack(pack)
        source = D.BatchSource(pack, "cpu")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        losses = []
        for _step in range(6):
            batch = source.make([i for i in range(len(rows))])
            res = model.forward_batch(batch)
            terms = model.loss(res, batch, y)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            opt.step()
            losses.append({k: float(v.detach()) for k, v in terms.items()})
        batch = source.make([i for i in range(len(rows))])
        res = model.forward_batch(batch)
        terms = model.loss(res, batch, y)
        model.zero_grad(set_to_none=True)
        terms["loss"].backward()
        grads: dict[str, float] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            top = name.split(".")[0]
            if param.grad is None:
                grads[top] = float("nan")
            else:
                grads[top] = grads.get(top, 0.0) + float(param.grad.abs().sum())
        dead = [k for k, v in grads.items() if not np.isfinite(v) or v == 0.0]
        out["arms"][arm] = {
            "params": model.parameter_ledger(),
            "branch_grad_abs_sum_after_6_steps": grads,
            "branches_without_gradient": dead,
            "finite_loss": all(np.isfinite(v) for x in losses for v in x.values()),
            "loss_first": losses[0],
            "loss_last": losses[-1],
        }
        log(f"  gradient coverage {arm}: dead={dead}")
    # shared modules must be bit-identical at initialisation across arms
    hashes = {}
    for arm in M.ARMS:
        model = M.ACRAModel(cfg, dims[0], dims[1], arm)
        hashes[arm] = C.module_hashes(model, ["frame", "proj_s", "proj_c", "mlp_s", "mlp_c",
                                              "bond_emb", "flag_bridge", "flag_ring",
                                              "atom_head", "scalar_s", "scalar_c", "beta"])
    shared_equal = all(hashes[a][k] == hashes[M.ARMS[0]][k]
                       for a in M.ARMS for k in hashes[M.ARMS[0]])
    out["shared_init_hashes_equal_across_arms"] = bool(shared_equal)
    out["init_hashes"] = hashes
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = C.load_config()
    raw = D.load_raw(cfg)
    dims = C.official_feature_dims()
    pack = K.ensure_pack(cfg)

    items: list[dict[str, Any]] = []
    coverage = full_dataset_coverage(cfg, raw, pack)
    items.append(_item("1_all_atoms_and_bonds_used", all([
        coverage["every_bond_has_two_events"],
        coverage["every_molecule_atom_count_matches_raw"],
        coverage["atom_attribute_table_equals_raw"],
        coverage["bond_count_per_molecule_matches_raw"],
    ]), coverage))
    items.append(_item("2_two_directed_events_per_undirected_bond",
                       coverage["every_bond_has_two_events"], coverage))
    items.append(_item("3_S_C_adjacency_and_raw_attributes",
                       False, "covered by pytest::test_local_balls_match_independent_bfs "
                              "and pytest::test_rooted_frames_match_bruteforce"))
    items.append(_item("4_ring_edges_never_dropped",
                       coverage["n_ring_directed_events"] > 0,
                       "covered by pytest::test_ring_bonds_keep_the_rest_of_the_ring; "
                       f"{coverage['n_ring_directed_events']} directed ring events"))
    items.append(_item("5_rooted_frame_enumeration_and_multiplicity",
                       False, "covered by pytest::test_rooted_frames_match_bruteforce and "
                              "pytest::test_synthetic_frames_and_multiplicity"))
    items.append(_item("6_atom_renumbering_invariance", False,
                       "covered by pytest::test_atom_renumbering_invariance"))
    items.append(_item("7_edge_direction_swap_invariance", False,
                       "covered by pytest::test_edge_direction_swap_invariance"))
    items.append(_item("8_single_vs_batch_prediction", False,
                       "covered by pytest::test_batch_equals_single"))
    items.append(_item("9_prediction_equals_sum_of_terms", False,
                       "covered by pytest::test_prediction_is_exactly_the_sum_of_terms"))
    items.append(_item("10_relation_term_pair_symmetry", False,
                       "covered by pytest::test_relation_term_is_symmetric"))
    grad = gradient_coverage(cfg, raw, dims)
    items.append(_item("11_all_trainable_branches_receive_gradient",
                       all(not rec["branches_without_gradient"] for rec in grad["arms"].values())
                       and grad["shared_init_hashes_equal_across_arms"], grad))
    items.append(_item("12_no_valid_test_labels_in_cache_scaler_vocab", False,
                       "covered by pytest::test_cache_and_scaler_are_train_only"))
    items.append(_item("13_no_nan_inf", False, "covered by pytest::test_no_nan_inf"))

    tests = pytest_run()
    for rec in items:
        if rec["detail"] is False or (isinstance(rec["detail"], str)
                                      and rec["detail"].startswith("covered by pytest")):
            rec["passed"] = bool(tests["passed"])
    out = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack.get("cache_sha256"),
        "cache_meta": json.loads(str(pack["meta_json"])),
        "pytest": tests,
        "checks": items,
        "all_passed": bool(tests["passed"]) and all(rec["passed"] for rec in items),
        "wall_seconds": float(time.time() - t0),
        "test_labels_read": False,
    }
    path = C.analysis_dir() / "checks.json"
    C.write_json(path, out)
    print(f"wrote {path}")
    for rec in items:
        print(f"  [{'ok' if rec['passed'] else 'FAIL'}] {rec['name']}")
    print(f"  pytest: {tests['returncode']} ({tests['seconds']:.1f}s)")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())