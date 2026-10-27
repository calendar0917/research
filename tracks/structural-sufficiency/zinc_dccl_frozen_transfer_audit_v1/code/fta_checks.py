"""CPU gates for the frozen-transfer audit.

Everything here is derived from the frozen artefacts; no label is read except
the already-frozen Monitor MAE cross-check that ``fta_score.py`` recorded.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import fta_common as K


def add(checks: list, name: str, ok: bool, detail=None) -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    t0 = time.perf_counter()
    cfg = K.load_config()
    checks: list[dict] = []
    freeze = json.loads((K.ANALYSIS / "representation_freeze.json").read_text())
    probe = json.loads((K.ANALYSIS / "probe_freeze.json").read_text())
    metrics = json.loads((K.ANALYSIS / "probe_monitor_metrics.json").read_text())

    cache = K.load_train_cache()
    rows = K.split_rows()
    pf, ps, split_info = K.probe_split(cache, rows["fit4k"])

    # ---- C1 split integrity
    add(checks, "split_fit4k_rows", len(rows["fit4k"]) == 4000, len(rows["fit4k"]))
    add(checks, "split_fit8k_rows", len(rows["fit8k"]) == 8000, len(rows["fit8k"]))
    add(checks, "split_monitor_rows", len(rows["monitor"]) == 1000, len(rows["monitor"]))
    add(checks, "split_monitor_disjoint_from_fits",
        len(set(rows["monitor"]) & (set(rows["fit4k"]) | set(rows["fit8k"]))) == 0)
    add(checks, "split_fit4k_nested_in_fit8k",
        set(rows["fit4k"].tolist()) <= set(rows["fit8k"].tolist()))
    add(checks, "probe_split_sizes", (len(pf), len(ps)) == (3000, 1000), [len(pf), len(ps)])
    add(checks, "probe_split_partition",
        set(pf.tolist()) | set(ps.tolist()) == set(rows["fit4k"].tolist())
        and not (set(pf.tolist()) & set(ps.tolist())))
    add(checks, "probe_split_hashes",
        split_info["probe_fit_sha256"] == cfg["probe_split"]["probe_fit_sha256"]
        and split_info["probe_sel_sha256"] == cfg["probe_split"]["probe_sel_sha256"])

    # ---- C2 the split cannot see labels
    cache_nolabel = dict(cache)
    cache_nolabel["labels"] = np.zeros_like(np.asarray(cache["labels"]))
    pf2, ps2, info2 = K.probe_split(cache_nolabel, rows["fit4k"])
    add(checks, "probe_split_label_free",
        np.array_equal(pf, pf2) and np.array_equal(ps, ps2)
        and info2["probe_fit_sha256"] == split_info["probe_fit_sha256"])

    # ---- C3 export gates
    gate_keys = ["E2_hook_vs_independent_recompute_max_abs",
                 "E3_reader_replay_max_abs",
                 "E4_batch64_vs_batch256_max_abs"]
    for cid, meta in freeze["conditions"].items():
        add(checks, f"export_gates_{cid}", bool(meta["gates_pass"]),
            {k: meta["checks"][k] for k in gate_keys})
    export_rows = np.concatenate([pf, ps, rows["monitor"]])
    add(checks, "export_rows_hash", K.row_hash(export_rows) == freeze["export_rows"]["sha256"],
        freeze["export_rows"]["n"])

    # ---- C4 z layout: 513 dims, contiguous blocks, g block == cache normalisation
    dim = int(cfg["representation"]["dim"])
    edges = [0] + [b for _, b in K.BLOCKS.values()]
    add(checks, "z_blocks_contiguous",
        edges == [0, 96, 192, 288, 321, 417, 513] and dim == 513, edges)
    gmean_gmax = 0.0
    for fit, seed in K.CONDITIONS:
        cid = f"{fit}_s{seed}"
        z = np.load(K.Z_DIR / f"{cid}.npz")["z"]
        ck = torch.load(K.trunk_paths(fit, seed)[0], map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        g = (np.asarray(cache["global_feat"], dtype=np.float32)[export_rows]
             - sd["global_mean"].numpy()) / sd["global_scale"].numpy()
        d = float(np.abs(z[:, 288:321] - g).max())
        gmean_gmax = max(gmean_gmax, d)
        add(checks, f"z_global33_matches_cache_{cid}", d <= 1e-5, d)
        add(checks, f"z_finite_{cid}", bool(np.isfinite(z).all()), z.shape)

    # ---- C5 reader shape / capacity / block isolation
    head = torch.nn.Sequential(torch.nn.Linear(513, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
    n_params = int(sum(p.numel() for p in head.parameters()))
    add(checks, "reader_parameter_count", n_params == 32961, n_params)
    add(checks, "reader_arch", cfg["reader"]["arch"] == "Linear(513,64)-SiLU-Linear(64,1)",
        cfg["reader"]["arch"])
    expected_features = {"atom_global": 225, "local_global": 225, "full": 513}
    for c in probe["cells"]:
        add(checks, f"cell_features_{c['id']}",
            c["n_features_used"] == expected_features[c["block"]], c["n_features_used"])
        if c["kind"] == "main" and c["block"] != "full":
            add(checks, f"cell_block_grad_isolation_{c['id']}",
                c["block_grad_leak_max_abs"] == 0.0, c["block_grad_leak_max_abs"])

    # ---- C6 head initialisation identity and reproducibility
    for init, info in probe["head_init_identity"].items():
        add(checks, f"head_init_identity_{init}", info["n_distinct"] == 1,
            info["hashes"])
    add(checks, "probe_reproducibility", bool(probe["reproducibility"]["abs_diff"] <= 1e-6
                                              and probe["reproducibility"]["best_state_identical"]),
        probe["reproducibility"])
    add(checks, "n_cells", probe["n_cells"] == 30, probe["n_cells"])

    # ---- C7 selection rule respected (argmin over the recorded curve, ties earlier)
    bad = []
    for c in probe["cells"]:
        curve = c["curve"]
        best_val = min(e["probe_sel_l1"] for e in curve)
        arg = next(e for e in curve if e["probe_sel_l1"] == best_val)
        if abs(c["best"]["probe_sel_l1"] - best_val) > 0 or c["best"]["step"] != arg["step"]:
            bad.append(c["id"])
        if c["curve"][-1]["step"] != int(cfg["reader"]["updates"]):
            bad.append(c["id"] + ":last_step")
    add(checks, "selection_rule_argmin_ties_earlier", not bad, bad[:5])

    # ---- C8 E5 export-vs-frozen-trajectory
    for cid, v in metrics["e5_export_vs_frozen_trajectory"].items():
        add(checks, f"E5_raw_monitor_mae_{cid}", v["abs_diff"] <= 1e-6, v)

    # ---- C9 no official valid/test / locked access in code
    offenders = []
    pat = "ad_" + "(valid|test)" + r"\.pt"   # built at runtime: this file must not self-match
    for p in sorted((K.ROUND / "code").glob("*.py")):
        if p.name == Path(__file__).name:
            continue
        txt = p.read_text()
        if re.search(pat, txt):
            offenders.append(p.name)
    add(checks, "no_valid_test_paths_in_code", not offenders, offenders)
    add(checks, "score_flags", metrics["official_valid_loaded"] is False
        and metrics["official_test_loaded"] is False and metrics["locked_loaded"] is False,
        {"official_valid_loaded": metrics["official_valid_loaded"],
         "official_test_loaded": metrics["official_test_loaded"],
         "locked_loaded": metrics["locked_loaded"]})
    add(checks, "monitor_rows_not_in_any_fit",
        len(set(rows["monitor"]) & (set(rows["fit4k"]) | set(rows["fit8k"]))) == 0)

    # ---- C10 statistics unit checks
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 7))
    Y = X @ rng.normal(size=(7, 5)) + rng.normal(size=(200, 5)) * 0.1
    add(checks, "cka_identity", abs(K.linear_cka(X, X) - 1.0) < 1e-9, K.linear_cka(X, X))
    add(checks, "cka_symmetry", abs(K.linear_cka(X, Y) - K.linear_cka(Y, X)) < 1e-12)
    add(checks, "cka_range", 0.0 <= K.linear_cka(X, rng.normal(size=X.shape)) <= 1.0)
    kk = np.arange(50, dtype=np.float64)
    Z = np.zeros((50, 4))
    Z[:, 1] = np.cos(2 * np.pi * kk / 50.0)
    Z[:, 2] = np.sin(2 * np.pi * kk / 50.0)
    spec = K.spectral_summary(Z)
    pr = K.participation_ratio(np.linalg.svd(Z - Z.mean(0), compute_uv=False))
    add(checks, "pr_two_equal_directions", abs(pr - 2.0) < 1e-9, pr)
    add(checks, "numerical_rank_two", spec["numerical_rank_tol_1e-06"] == 2,
        spec["numerical_rank_tol_1e-06"])
    add(checks, "block_mask_sizes",
        int(K.block_mask("atom_global").sum()) == 225
        and int(K.block_mask("local_global").sum()) == 225
        and int(K.block_mask("full").sum()) == 513)

    # ---- C11 the split key is a *labelled-graph* fingerprint:
    #      invariant to the storage order of the edge list, sensitive to content.
    g = K.molecule_graphs(cache, [rows["monitor"][0]])[0]
    atoms, bonds = g
    rng2 = np.random.default_rng(7)
    shuffled = [bonds[i] for i in rng2.permutation(len(bonds))]
    add(checks, "fingerprint_bond_order_invariant",
        K.structure_fingerprint((atoms, bonds)) == K.structure_fingerprint((atoms, shuffled)))
    other = list(atoms)
    other[0] = (int(other[0]) + 1) % 21
    add(checks, "fingerprint_atom_type_sensitive",
        K.structure_fingerprint((atoms, bonds)) != K.structure_fingerprint((other, bonds)))

    payload = {
        "round": K.ROUND_ID,
        "n_checks": len(checks),
        "n_failed": int(sum(1 for c in checks if not c["pass"])),
        "all_pass": bool(all(c["pass"] for c in checks)),
        "checks": checks,
        "wall_seconds": float(time.perf_counter() - t0),
    }
    K.write_json(K.ANALYSIS / "cpu_checks.json", payload)
    K.ledger_append({"stage": "checks", "wall_seconds": payload["wall_seconds"], "device": "cpu"})
    K.log(f"CPU checks {len(checks) - payload['n_failed']}/{len(checks)} pass")
    failed = [c["check"] for c in checks if not c["pass"]]
    if failed:
        K.log(f"FAILED: {failed}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
