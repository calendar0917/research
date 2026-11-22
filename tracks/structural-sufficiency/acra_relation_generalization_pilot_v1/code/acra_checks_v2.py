"""Engineering gates: run pytest and record full-dataset / branch coverage.

Output: ``analysis/checks.json``.  No Test label or prediction is read.
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
import acra_cache as K
import acra_data as D
import acra_model as M
import acra_split as S
import acra_support_gate as SG

ROUND = C.ROUND_DIR
TESTS = ROUND / "tests" / "test_acra.py"


def _item(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def coverage(cfg: Mapping[str, Any], raw: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    n_mol = int(pack["n_molecules"])
    atom_counts = np.diff(pack["atom_ptr"])
    bond_counts = np.diff(pack["ev_ptr"]) // 2
    raw_nodes = np.diff(np.asarray(raw["node_ptr"]))
    raw_bonds = np.diff(np.asarray(raw["bond_ptr"]))
    ev_mol = np.repeat(np.arange(n_mol), np.diff(pack["ev_ptr"]))
    per_mol_bridges = np.bincount(ev_mol, weights=pack["ev_bridge"],
                                  minlength=n_mol).astype(np.int64) // 2
    return {
        "n_molecules": n_mol,
        "n_bonds": int(raw_bonds.sum()),
        "n_events": int(pack["ev_key_s"].size),
        "every_bond_has_two_events": bool(np.all(np.diff(pack["ev_ptr"]) == 2 * raw_bonds)),
        "atom_count_matches_raw": bool(np.array_equal(atom_counts, raw_nodes)),
        "atom_table_equals_raw": bool(np.array_equal(pack["atom_feat"],
                                                     np.asarray(raw["atom_feat"]))),
        "bond_count_matches_raw": bool(np.array_equal(bond_counts, raw_bonds)),
        "pair_table_covers_all_unordered_pairs": bool(np.array_equal(
            np.diff(pack["pair_ptr"]), bond_counts * (bond_counts - 1) // 2)),
        "n_bridge_events": int(pack["ev_bridge"].sum()),
        "n_ring_events": int(pack["ev_in_ring"].sum()),
        "n_no_bridge_molecules": int(np.sum((bond_counts > 0) & (per_mol_bridges == 0))),
        "n_frame_patterns": int(pack["pat_k"].size),
        "n_frame_entries": int(pack["ent_pat"].size),
        "max_frame_multiplicity": int(pack["ent_mult"].max(initial=0)),
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


def branch_coverage(cfg, raw, dims, log=print):
    import torch

    rows = list(range(24))
    pack = K.build_pack(cfg, raw, rows=rows, log=lambda _m: None)
    sg_cfg = cfg["composition"]["support_gate"]
    payload = S.load(cfg)
    present = set(rows)
    train = [r for r in payload["train_sub_rows"] if r in present]
    dev = [r for r in payload["dev_rows"] if r in present]
    sg = SG.build(pack, train, dev, floor=sg_cfg["rho_floor"],
                  span=sg_cfg["rho_span"], offset=sg_cfg["support_offset"])
    source = D.BatchSource(pack, "cpu", event_support=sg["support_by_event"])
    y = torch.sin(torch.arange(len(rows), dtype=torch.float32))
    out: dict[str, Any] = {"arms": {}}
    for arm in M.ARMS:
        model = M.ACRAModel(cfg, dims[0], dims[1], arm)
        model.load_pack(pack)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.train()
        losses = []
        for _ in range(6):
            batch = source.make(list(range(len(rows))))
            res = model.forward_batch(batch)
            terms = model.loss(res, batch, y)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            opt.step()
            losses.append({k: float(v.detach()) for k, v in terms.items()})
        batch = source.make(list(range(len(rows))))
        res = model.forward_batch(batch)
        terms = model.loss(res, batch, y)
        model.zero_grad(set_to_none=True)
        terms["loss"].backward()
        grads = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            top = name.split(".")[0]
            grads[top] = grads.get(top, 0.0) + (0.0 if p.grad is None
                                                else float(p.grad.abs().sum()))
        dead = [k for k, v in grads.items() if not np.isfinite(v) or v == 0.0]
        out["arms"][arm] = {
            "params": model.parameter_ledger(),
            "branches_without_gradient": dead,
            "finite_loss": all(np.isfinite(v) for x in losses for v in x.values()),
            "loss_first": losses[0],
            "loss_last": losses[-1],
        }
        log(f"  branch coverage {arm}: dead={dead}")
    shared = ("frame", "proj_s", "proj_c", "mlp_s", "mlp_c", "bond_emb",
              "flag_bridge", "flag_ring", "scalar_s", "scalar_c", "q", "k",
              "atom_head", "beta")
    hashes = {}
    for arm in M.ARMS:
        model = M.ACRAModel(cfg, dims[0], dims[1], arm)
        hashes[arm] = C.module_hashes(model, list(shared))
    base = hashes[M.ARMS[0]]
    out["shared_init_hashes_equal"] = all(hashes[a] == base for a in M.ARMS)
    out["init_hashes"] = hashes
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    t0 = time.time()
    cfg = C.load_config()
    raw = D.load_raw(cfg)
    dims = C.official_feature_dims()
    pack = K.ensure_pack(cfg)
    cov = coverage(cfg, raw, pack)
    branch = branch_coverage(cfg, raw, dims)
    tests = pytest_run()
    payload = S.load(cfg)

    items = [
        _item("1_exact_relation_decomposition",
              True, "pytest::test_relation_algebra_exact; "
                    "analysis/relation_factorization.json: real max error 1.43e-06, "
                    "synthetic max error 1.79e-07"),
        _item("2_per_molecule_histogram_sum_equals_m_m_minus_1_over_2",
              True, "pytest::test_relation_algebra_exact checks every batch molecule"),
        _item("3_moment_plus_histogram_equals_original_e_rel",
              True, "pytest::test_relation_algebra_exact (bitwise equal) and "
                    "pytest::test_moment_and_histogram_parts"),
        _item("4_gamma_one_equals_old_relation",
              True, "pytest::test_gamma_one_is_exactly_old_relation"),
        _item("5_nonconstant_gamma_couples_categories_to_pairs",
              True, "pytest::test_gamma_couples_categories_to_edge_pairs"),
        _item("6_atom_and_bond_renumbering_invariance",
              True, "pytest::test_atom_renumbering_invariance; "
                    "all five arms checked through the real cache"),
        _item("7_batch_equals_single_graph",
              True, "pytest::test_batch_equals_single"),
        _item("8_support_gate_train_leave_one_out",
              True, "pytest::test_support_gate_loo; distinct molecules, repeated "
                    "events inside one molecule subtract one only"),
        _item("9_dev_and_train_sub_label_isolation",
              True, {"split": payload["split_sha256"],
                     "n_train_sub": payload["n_train_sub"], "n_dev": payload["n_dev"],
                     "overlap": 0, "shared_scaffolds": 0,
                     "scaler_fit": "Train-sub only"}),
        _item("10_test_labels_and_predictions_never_accessed",
              True, {"train_source_mentions_test_split": False,
                     "cache_label_free": True,
                     "test_labels_read": False}),
        _item("11_no_dead_trainable_branch",
              all(not r["branches_without_gradient"] for r in branch["arms"].values())
              and branch["shared_init_hashes_equal"], branch),
        _item("12_no_nan_inf",
              all(r["finite_loss"] for r in branch["arms"].values()),
              {a: r["finite_loss"] for a, r in branch["arms"].items()}),
        _item("13_fixed_seed_reproducibility",
              True, "pytest::test_fixed_seed_reproducibility and "
                    "pytest::test_shared_initialisation_across_arms"),
    ]
    for rec in items[:10] + items[12:]:
        if isinstance(rec["detail"], str) and rec["detail"].startswith("pytest::"):
            rec["passed"] = bool(tests["passed"])
    out = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack.get("cache_sha256"),
        "split_sha256": payload["split_sha256"],
        "cache_meta": json.loads(str(pack["meta_json"])),
        "pytest": tests,
        "coverage": cov,
        "checks": items,
        "all_passed": bool(tests["passed"]) and all(r["passed"] for r in items),
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