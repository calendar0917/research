"""Pre-training engineering checks for task-conditioned word discovery v0.

Writes ``analysis/checks.json``.  No formal GPU run is allowed until every
critical check is PASS.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import tc_common as C
import tc_cpu as P
import tc_data as D
import tc_neural as N


def _ok(condition: bool) -> str:
    return "PASS" if bool(condition) else "FAIL"


def main() -> dict[str, Any]:
    cfg = C.load_config()
    checks: list[dict[str, Any]] = []

    def add(name: str, condition: bool, detail: Any) -> None:
        checks.append({"check": name, "status": _ok(condition), "detail": C.jsonable(detail)})

    # A -- frozen identities
    pins = C.assert_source_pins()
    add("A1_source_pins", True, pins)
    bundle = C.split_bundle("D0")
    add("A2_D0_identity", str(bundle.split["identity_sha256"]) ==
        cfg["data"]["splits"]["D0"]["identity_sha256"], C.split_meta(bundle))
    b1 = C.split_bundle("D1")
    add("A3_D1_identity", str(b1.split["identity_sha256"]) ==
        cfg["data"]["splits"]["D1"]["identity_sha256"], C.split_meta(b1))
    add("A4_valid_labels_nan", not np.isfinite(bundle.labels[bundle.official_valid]).any(),
        {"n_valid": int(bundle.official_valid.size),
         "n_finite_valid": int(np.isfinite(bundle.labels[bundle.official_valid]).sum())})
    add("A5_no_test_idx_member", "test_idx" not in bundle.raw,
        {"raw_keys": sorted(bundle.raw.keys())})
    add("A6_train_labels_3360", int(bundle.labels[np.union1d(bundle.fit_rows, bundle.eval_rows)].size) == 3360,
        {"n_train": int(np.union1d(bundle.fit_rows, bundle.eval_rows).size)})

    # B -- candidate freeze / dedupe / folds
    payloads = {s: C.read_json(C.ANALYSIS_DIR / f"candidates_{s}.json") for s in ("D0", "D1")}
    for split, payload in payloads.items():
        b = C.split_bundle(split)
        add(f"B1_{split}_split_match",
            str(payload["split_meta"]["identity_sha256"]) == str(b.split["identity_sha256"]),
            {"split": split})
        add(f"B2_{split}_candidate_counts",
            payload["n_candidates_dedup"] == len(payload["candidates"]) > 0,
            {"n_qualified": payload["n_candidates_qualified"],
             "n_dedup": payload["n_candidates_dedup"],
             "n_equivalence_groups": payload["n_equivalence_groups"]})
        add(f"B3_{split}_thresholds",
            all(c["df"] >= cfg["candidate"]["min_molecules"]
                and c["scaffold_df"] >= cfg["candidate"]["min_scaffolds"]
                for c in payload["candidates"]),
            {"min_molecules": cfg["candidate"]["min_molecules"],
             "min_scaffolds": cfg["candidate"]["min_scaffolds"]})
        add(f"B4_{split}_word_sizes", all(int(c["size"]) in (3, 4) for c in payload["candidates"]),
            {"size_gran": dict(Counter((c["size"], c["gran"]) for c in payload["candidates"]))})
        add(f"B5_{split}_dedup_members",
            all(sorted(c["equiv_members"]) == sorted(set(c["equiv_members"]))
                for c in payload["candidates"]),
            {"n_groups_multi": sum(1 for g in payload["equivalence_groups"] if g["n_members"] > 1)})
        x = C.sha256_file(C.ANALYSIS_DIR / f"candidate_columns_{split}.npz")
        with np.load(C.ANALYSIS_DIR / f"candidate_columns_{split}.npz",
                    allow_pickle=False) as z:
            x0f = C.array_sha256(np.asarray(z["x0_fit"]))
            x0e = C.array_sha256(np.asarray(z["x0_eval"]))
        add(f"B6_{split}_column_hash",
            x == payload["candidate_columns_sha256"]
            and x0f == payload["x0_fit_sha256"]
            and x0e == payload["x0_eval_sha256"],
            {"candidate_columns_sha256": x,
             "recorded": payload["candidate_columns_sha256"],
             "x0_fit_sha256": x0f, "x0_eval_sha256": x0e})
        fold_of_row, fold_info = P.scaffold_folds(split, b.fit_rows, int(cfg["folds"]["seed"]))
        sizes = np.bincount(fold_of_row, minlength=3)
        add(f"B7_{split}_folds", int(sizes.sum()) == int(b.fit_rows.size)
            and all(int(s) > 0 for s in sizes),
            {"fold_sizes": sizes.tolist(), **fold_info})

    # C -- selection / frequency control / positive control
    for split in ("D0", "D1"):
        sel = C.read_json(C.ANALYSIS_DIR / f"selection_scores_{split}.json")
        control = C.read_json(C.ANALYSIS_DIR / f"positive_control_{split}.json")["control"]
        add(f"C1_{split}_K", 32 <= len(sel["task_columns"]) <= cfg["selection"]["K"],
            {"n_task": len(sel["task_columns"]), "n_freq": len(sel["freq_columns"]), "K": cfg["selection"]["K"]})
        # exact joint distribution match
        rows = sel["rows"]
        task = [rows[i] for i in sel["task_columns"]]
        freq = [rows[i] for i in sel["freq_columns"]]
        task_cells = Counter((r["size"], r["gran"], r["df_bucket"]) for r in task)
        freq_cells = Counter((r["size"], r["gran"], r["df_bucket"]) for r in freq)
        add(f"C2_{split}_freq_distribution", task_cells == freq_cells,
            {"task_cells": {str(k): v for k, v in sorted(task_cells.items())},
             "freq_cells": {str(k): v for k, v in sorted(freq_cells.items())}})
        add(f"C3_{split}_task_direction",
            all(r["direction_consistent"] and r["score"] is not None
                and (r["direction"] == 1 and all(c > 0 for c in r["cond_corr_folds"])
                     or r["direction"] == -1 and all(c < 0 for c in r["cond_corr_folds"]))
                for r in task),
            {"n_task": len(task),
             "min_score": min(r["score"] for r in task),
             "max_score": max(r["score"] for r in task)})
        add(f"C4_{split}_positive_control", bool(control.get("passed")),
            {"status": control.get("status"),
             "planted": control.get("planted", {}).get("key"),
             "planted_in_task": control.get("planted_in_synthetic_task_vocabulary")})
        ridge = C.read_json(C.ANALYSIS_DIR / f"ridge_arms_{split}.json")
        arms = [a["arm"] for a in ridge["arms"]]
        add(f"C5_{split}_ridge_arms",
            arms == ["LOW_RIDGE", "ALL_RIDGE", "TASK_RIDGE", "FREQ_RIDGE"],
            {"ridge_arms": ridge["arms"]})

    # D -- neural shape / init / vocabulary plumbing
    atom_dims, bond_dims = C.atom_bond_dims()
    sel_d0 = C.read_json(C.ANALYSIS_DIR / "selection_scores_D0.json")
    K = len(sel_d0["task_columns"])
    models = {arm: N.TCWordModel(atom_dims, bond_dims, K, cfg["neural"]["dims"])
              for arm in N.ARMS}
    hashes = {arm: m.init_parameters(0) for arm, m in models.items()}
    ledgers = {arm: m.parameter_ledger() for arm, m in models.items()}
    add("D1_params_equal", len(set(json.dumps(v, sort_keys=True) for v in ledgers.values())) == 1,
        {"ledgers": ledgers})
    add("D2_init_equal", len(set(json.dumps(v, sort_keys=True) for v in hashes.values())) == 1,
        {"init_hashes": hashes})
    add("D3_dims", all(ledgers[a]["total"] > 0 for a in ledgers)
        and models["BASE"].sentence_in_dim == 176
        and models["BASE"].pair_in_dim == 72
        and models["BASE"].readout_in_dim == 297,
        {"sentence_input": models["BASE"].sentence_in_dim,
         "pair_input": models["BASE"].pair_in_dim,
         "readout_input": models["BASE"].readout_in_dim,
         "total_params": ledgers["BASE"]["total"]})
    data = D.build_split_data("D0", atom_dims, bond_dims)
    add("D4_data_K", data.n_task == K and data.n_freq == K
        and data.arm_columns["D_TASK"].size == K and data.arm_columns["D_FREQ"].size == K,
        {"n_task": data.n_task, "n_freq": data.n_freq,
         "union": data.hashes["n_union_words"]})
    task_keys = set(C.read_json(C.ANALYSIS_DIR / "vocab_D0_task.json")[i]["key"] for i in range(K))
    freq_keys = set(C.read_json(C.ANALYSIS_DIR / "vocab_D0_freq.json")[i]["key"] for i in range(K))
    add("D5_vocab_keys_match",
        all(data.union_vocab[int(c)]["key"] in task_keys for c in data.arm_columns["D_TASK"])
        and all(data.union_vocab[int(c)]["key"] in freq_keys for c in data.arm_columns["D_FREQ"]),
        {"task_vocab_sha256": data.hashes["task_vocab_sha256"],
         "freq_vocab_sha256": data.hashes["freq_vocab_sha256"]})
    # BASE output must not depend on the word-count tensor
    batch = D.collate(data, data.fit_index[:4].tolist(), "BASE", "cpu")
    batch2 = dict(batch)
    batch2["word_counts"] = torch_rand_like(batch["word_counts"])
    with with_torch_no_grad(models["BASE"]):
        y1 = models["BASE"](batch, "BASE")
        y2 = models["BASE"](batch2, "BASE")
    add("D6_base_ignores_words", bool(torch_allclose(y1, y2)),
        {"max_abs_diff": float((y1 - y2).abs().max())})
    # D_TASK uses only its selected columns (word width is exactly K)
    batch_t = D.collate(data, data.fit_index[:4].tolist(), "D_TASK", "cpu")
    add("D7_task_word_width", int(batch_t["word_counts"].shape[1]) == K,
        {"word_width": int(batch_t["word_counts"].shape[1])})
    # recipe equality with the frozen R2 recipe
    frozen_recipe = dict(C.RC.frozen_config()["recipe"])
    our = dict(cfg["neural"]["recipe"])
    keys = ("optimizer", "peak_lr", "min_lr", "weight_decay", "gradient_clip",
            "batch_molecules", "warmup_epochs", "schedule", "epochs", "eval_batch",
            "patience", "min_epochs", "objective")
    same = all((str(frozen_recipe.get(k)) == str(our.get(k))) for k in keys)
    add("D8_recipe_matches_r2", same, {"frozen": {k: frozen_recipe.get(k) for k in keys},
                                       "ours": {k: our.get(k) for k in keys}})
    add("D9_no_synthetic_in_training",
        "synthetic" not in Path(__file__).parent.joinpath("tc_train.py").read_text(encoding="utf-8")
        and "positive_control" not in Path(__file__).parent.joinpath("tc_train.py").read_text(encoding="utf-8"),
        {"files": ["tc_train.py", "tc_data.py"]})

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    payload = {
        "protocol_id": cfg["protocol_id"], "version": cfg["version"],
        "n_checks": len(checks), "n_pass": int(n_pass),
        "all_pass": bool(n_pass == len(checks)),
        "checks": checks, "frozen_utc": C.now_utc(),
    }
    C.write_json(C.ANALYSIS_DIR / "checks.json", payload)
    print(json.dumps({"n_checks": len(checks), "n_pass": n_pass,
                      "all_pass": payload["all_pass"]}, indent=1))
    for c in checks:
        if c["status"] != "PASS":
            print("FAIL", c["check"], c["detail"])
    return payload


# tiny torch shims keep the check body readable
def torch_rand_like(x):
    import torch
    return torch.rand_like(x)


def torch_allclose(a, b):
    import torch
    return torch.allclose(a, b, atol=1e-7, rtol=0)


class with_torch_no_grad:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        import torch
        self._ctx = torch.no_grad()
        self._ctx.__enter__()
        self.model.eval()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


if __name__ == "__main__":
    main()