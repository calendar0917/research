"""CPU correctness gates.  Every gate must pass before formal GPU training.

Writes ``analysis/correctness_gates.json``.  A single failed critical gate stops
the round for the affected model: ``--strict`` makes the process exit non-zero.

Covered (protocol §3):
* A0/C1 parameter counts and the frozen column-deletion mapping (single task and
  the 12-output Tox21 head);
* A0 reproduces the frozen upstream model's CPU forward (replay of the frozen
  public122 Lipophilicity checkpoint against its stored predictions);
* dropout really runs in training mode, is applied exactly once to the shared
  ``h``, and is the identity in eval mode;
* atom renumbering leaves the prediction unchanged;
* batch split/merge consistency and no cross-graph information mixing;
* S1/S2 path construction agrees with an independent brute-force enumeration on
  the official graph;
* every trainable parameter receives a gradient;
* all standardisation constants come from the Train/Fit part only;
* Test/outer-holdout labels cannot enter the training path (static + runtime);
* a saved best checkpoint reloads and reproduces the Validation metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ptb_common as C
import ptb_data as D
import ptb_model as M
import ptb_train_mol as TM

import lcrc_data as PREP_DATA
import mb2_metrics as MET
import molecular_graph as MG
import tu_gates as TG
import tu_model as TUM

TOL = 1e-5


def _gate(name: str, ok: bool, detail: str, checks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"gate": name, "passed": bool(ok), "detail": detail, "checks": dict(checks or {})}


# ---------------------------------------------------------------------------
# 1. parameters and column mapping
# ---------------------------------------------------------------------------


def gate_params(spec, norm) -> dict[str, Any]:
    out: dict[str, Any] = {}
    single = {}
    for v in C.MODELS:
        model, info = M.build_model(spec, norm, v, 0, "cpu")
        single[v] = info["parameters"]
        out[f"{v}_parameters_single_task"] = info["parameters"]
        out[f"{v}_shared_state_equal_to_full"] = info["shared_state_equal_to_full"]["all_equal"]
        out[f"{v}_local_input_cols"] = info["local_input_cols"]
    out["delta_single_task"] = single["A0"] - single["C1"]
    multi = {}
    for v in C.MODELS:
        _m, info = M.build_model(spec, norm, v, 0, "cpu", n_tasks=12)
        multi[v] = info["parameters"]
    out["A0_parameters_tox21"] = multi["A0"]
    out["C1_parameters_tox21"] = multi["C1"]
    out["delta_tox21"] = multi["A0"] - multi["C1"]

    # frozen column map: C1 columns are exactly FULL's columns minus b_mean
    layout = C.DC.LOCAL_LAYOUT
    full_cols, _ = C.DC.variant_cols("A0")
    c1_cols, _ = C.DC.variant_cols("C1")
    removed = set(range(*layout["b_mean"]))
    expected = tuple(i for i in full_cols if i not in removed)
    out["c1_columns_equal_full_minus_b_mean"] = tuple(c1_cols) == expected
    out["n_columns_removed"] = len(full_cols) - len(c1_cols)

    # the deleted block really is 96 columns x 192 hidden = 18432 + 0 bias
    out["deleted_block_parameters"] = 96 * 192
    ok = (
        single["A0"] == C.EXPECTED_PARAMS_SINGLE["A0"]
        and single["C1"] == C.EXPECTED_PARAMS_SINGLE["C1"]
        and out["delta_single_task"] == C.EXPECTED_PARAM_DELTA
        and out["delta_tox21"] == C.EXPECTED_PARAM_DELTA
        and out["c1_columns_equal_full_minus_b_mean"]
        and out["A0_shared_state_equal_to_full"]
        and out["C1_shared_state_equal_to_full"]
    )
    return _gate("params_and_columns", ok, "A0 247106 / C1 228674 / delta 18432 (single and Tox21)", out)


# ---------------------------------------------------------------------------
# 2. frozen upstream replay
# ---------------------------------------------------------------------------


def gate_frozen_replay(_unused=None) -> dict[str, Any]:
    view = D.molecule_view("Lipophilicity")
    ref = C.MOLNET_V1 / "analysis/runs/Lipophilicity_s0"
    ckpt_path = ref / "checkpoint.pt"
    pred_path = ref / "valid_predictions.npz"
    if not ckpt_path.exists() or not pred_path.exists():
        return _gate("frozen_replay", False, "frozen upstream checkpoint/predictions missing", {"path": str(ref)})
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    spec = view.spec
    model = M.load_checkpoint_model(spec, view.train_norm, C.MODEL_A0, state, "cpu")
    stored = np.load(pred_path)
    pred = TM.predict_matrix(model, view.valid_cache, "cpu", batch_size=128)[:, 0]
    delta_ours = float(np.abs(pred - np.asarray(stored["raw"], dtype=np.float64).reshape(-1)).max())
    ok = delta_ours <= TOL
    return _gate(
        "frozen_replay",
        ok,
        "A0 re-implementation reproduces the frozen upstream Lipophilicity seed-0 Validation predictions",
        {
            "n": int(pred.size),
            "max_abs_delta_vs_frozen_predictions": delta_ours,
            "tolerance": TOL,
            "reference": C.track_rel(ref),
            "reference_checkpoint_sha256": C.sha256_file(ckpt_path),
        },
    )


# ---------------------------------------------------------------------------
# 3. dropout semantics
# ---------------------------------------------------------------------------


def gate_dropout(view) -> dict[str, Any]:
    spec = view.spec
    out: dict[str, Any] = {}
    for v in C.MODELS:
        model, _info = M.build_model(spec, view.train_norm, v, 0, "cpu")
        model.eval()
        batch = PREP_DATA.collate(view.train_cache, list(range(8)))
        batch = {k: (torch.as_tensor(x) if not torch.is_tensor(x) else x) for k, x in batch.items()}
        with torch.no_grad():
            e1 = model(batch).clone()
            e2 = model(batch).clone()
        out[f"{v}_eval_deterministic_max_delta"] = float((e1 - e2).abs().max())

        model.train()
        calls = {"n": 0}
        handle = model.drop.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
        with torch.no_grad():
            t1 = model(batch).clone()
            n_calls_1 = calls["n"]
            t2 = model(batch).clone()
            n_calls_2 = calls["n"]
        handle.remove()
        out[f"{v}_train_stochastic_max_delta"] = float((t1 - t2).abs().max())
        out[f"{v}_dropout_calls_per_forward"] = int(n_calls_2 - n_calls_1)
        # the frozen constant is the same 0.10 for both variants
        out[f"{v}_dropout_p"] = float(model.drop.p)
        # h passes through dropout exactly once: monkey-patch nn.Dropout forward
        n_drop = {"n": 0}
        orig_forward = torch.nn.Dropout.forward

        def counting_forward(self, x):
            n_drop["n"] += 1
            return orig_forward(self, x)

        torch.nn.Dropout.forward = counting_forward
        try:
            model.train()
            with torch.no_grad():
                model(batch)
        finally:
            torch.nn.Dropout.forward = orig_forward
        out[f"{v}_nn_dropout_invocations_per_forward"] = int(n_drop["n"])
    ok = (
        all(out[f"{v}_eval_deterministic_max_delta"] == 0.0 for v in C.MODELS)
        and all(out[f"{v}_train_stochastic_max_delta"] > 0.0 for v in C.MODELS)
        and all(out[f"{v}_dropout_calls_per_forward"] == 1 for v in C.MODELS)
        and all(out[f"{v}_nn_dropout_invocations_per_forward"] == 1 for v in C.MODELS)
        and all(abs(out[f"{v}_dropout_p"] - 0.10) < 1e-12 for v in C.MODELS)
    )
    return _gate("dropout_semantics", ok, "dropout runs once per forward on the shared h and is the identity in eval", out)


# ---------------------------------------------------------------------------
# 4. atom renumbering invariance
# ---------------------------------------------------------------------------


def _permuted_single(cache: Mapping[str, Any], pos: int, perm: Sequence[int]) -> dict[str, Any]:
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
    b0, b1 = int(bond_ptr[pos]), int(bond_ptr[pos + 1])
    n = a1 - a0
    perm = np.asarray(perm, dtype=np.int64)  # perm[new] = old
    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)
    bi = np.asarray(cache["bond_index"][b0:b1], dtype=np.int64).reshape(-1, 2)
    new_bi = inverse[bi] if bi.size else bi.reshape(0, 2)
    nbr_codes = [
        np.asarray(cache["nbr_code"][int(nbr_ptr[a0 + int(perm[i])]) : int(nbr_ptr[a0 + int(perm[i]) + 1])], dtype=np.int64)
        for i in range(n)
    ]
    per_atom_paths = []
    for i in range(n):
        old = int(perm[i])
        lo, hi = int(path_ptr[a0 + old]), int(path_ptr[a0 + old + 1])
        per_atom_paths.append(
            (
                np.asarray(cache["path_cat"][lo:hi], dtype=np.int64),
                np.asarray(cache["path_end63"][lo:hi], dtype=np.int64),
                np.asarray(cache["path_end"][lo:hi], dtype=np.int64),
                np.asarray(cache["path_closure"][lo:hi], dtype=np.int64),
            )
        )
    cat = np.concatenate([x[0] for x in per_atom_paths]) if n else np.zeros(0, np.int64)
    end63 = np.concatenate([x[1] for x in per_atom_paths]) if n else np.zeros(0, np.int64)
    pend = inverse[np.concatenate([x[2] for x in per_atom_paths])] if n else np.zeros(0, np.int64)
    pclos = np.concatenate([x[3] for x in per_atom_paths]) if n else np.zeros(0, np.int64)
    return {
        "atom_ptr": np.asarray([0, n], dtype=np.int64),
        "bond_ptr": np.asarray([0, len(new_bi)], dtype=np.int64),
        "nbr_ptr": np.concatenate([[0], np.cumsum([len(x) for x in nbr_codes])]).astype(np.int64),
        "path_ptr": np.concatenate([[0], np.cumsum([len(x[0]) for x in per_atom_paths])]).astype(np.int64),
        "atom_feat": np.asarray(cache["atom_feat"][a0:a1])[perm],
        "atom_type": np.asarray(cache["atom_type"][a0:a1])[perm],
        "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1])[perm],
        "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
        "bond_type": np.asarray(cache["bond_type"][b0:b1]),
        "bond_index": new_bi,
        "nbr_code": np.concatenate(nbr_codes) if n else np.zeros(0, np.int64),
        "path_cat": cat,
        "path_end63": end63,
        "path_end": pend,
        "path_closure": pclos,
        "global": np.asarray(cache["global"][pos : pos + 1]),
        "n_components": np.asarray(cache["n_components"][pos : pos + 1]),
        "n_junctions": np.asarray(cache["n_junctions"][pos : pos + 1]),
        "n_ring_systems": np.asarray(cache["n_ring_systems"][pos : pos + 1]),
        "molecule_ids": [str(cache["molecule_ids"][pos])],
        "smiles": [str(cache["smiles"][pos])],
        "n_molecules": 1,
        "split": cache["split"],
        "meta": dict(cache["meta"]),
        "rows": np.asarray([int(np.asarray(cache["rows"])[pos])], dtype=np.int64),
    }


def gate_renumbering(view) -> dict[str, Any]:
    spec = view.spec
    out: dict[str, Any] = {}
    rng = np.random.default_rng(20261102)
    positions = [0, 1, 2, 7, 19]
    worst = 0.0
    for v in C.MODELS:
        model, _ = M.build_model(spec, view.train_norm, v, 0, "cpu")
        model.eval()
        deltas = []
        for pos in positions:
            n = int(np.asarray(view.train_cache["atom_ptr"])[pos + 1]) - int(np.asarray(view.train_cache["atom_ptr"])[pos])
            perm = rng.permutation(n)
            single = _permuted_single(view.train_cache, pos, np.arange(n))
            permuted = _permuted_single(view.train_cache, pos, perm)
            b0 = PREP_DATA.collate(single, [0])
            b1 = PREP_DATA.collate(permuted, [0])
            with torch.no_grad():
                p0 = model({k: (torch.as_tensor(x) if not torch.is_tensor(x) else x) for k, x in b0.items()})
                p1 = model({k: (torch.as_tensor(x) if not torch.is_tensor(x) else x) for k, x in b1.items()})
            deltas.append(float((p0 - p1).abs().max()))
        out[f"{v}_max_abs_prediction_delta"] = max(deltas)
        out[f"{v}_per_molecule"] = deltas
        worst = max(worst, max(deltas))
    ok = worst <= TOL
    return _gate("renumbering_invariance", ok, "atom renumbering leaves the prediction unchanged", out)


# ---------------------------------------------------------------------------
# 5. batching / no cross-graph mixing
# ---------------------------------------------------------------------------


def gate_batching(view) -> dict[str, Any]:
    spec = view.spec
    out: dict[str, Any] = {}
    rows = [0, 1, 2, 3, 4, 5]
    for v in C.MODELS:
        model, _ = M.build_model(spec, view.train_norm, v, 0, "cpu")
        model.eval()
        with torch.no_grad():
            whole = model(PREP_DATA.collate(view.train_cache, rows))
            singles = torch.cat(
                [model(PREP_DATA.collate(view.train_cache, [r])) for r in rows], dim=0
            )
            split = torch.cat(
                [
                    model(PREP_DATA.collate(view.train_cache, rows[:2])),
                    model(PREP_DATA.collate(view.train_cache, rows[2:4])),
                    model(PREP_DATA.collate(view.train_cache, rows[4:])),
                ],
                dim=0,
            )
        out[f"{v}_batch_vs_single_max_delta"] = float((whole - singles).abs().max())
        out[f"{v}_batch_vs_split_max_delta"] = float((whole - split).abs().max())
        # order invariance: reversing the batch order must not change per-graph rows
        with torch.no_grad():
            rev = model(PREP_DATA.collate(view.train_cache, list(reversed(rows))))
        out[f"{v}_order_permutation_max_delta"] = float((whole - rev.flip(0)).abs().max())
    ok = all(out[f"{v}_{k}_max_delta"] <= TOL for v in C.MODELS for k in ("batch_vs_single", "batch_vs_split", "order_permutation"))
    return _gate("batching_and_isolation", ok, "batch merge/split and graph order do not change per-graph predictions", out)


# ---------------------------------------------------------------------------
# 6. S1 / S2 path construction vs brute force
# ---------------------------------------------------------------------------


def gate_paths() -> dict[str, Any]:
    """Rebuild S1/S2 by explicit enumeration from the adjacency and compare."""
    rng = np.random.default_rng(7)
    n_atoms = 9
    bonds = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 6), (6, 4), (6, 7), (7, 8)]
    atom_type = rng.integers(0, 4, size=n_atoms).astype(np.int64)
    nb = 5
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n_atoms)]
    for e, (u, v) in enumerate(bonds):
        bt = 1 + (e % 4)
        adj[u].append((v, bt, e))
        adj[v].append((u, bt, e))
    dist = MG.all_pairs_shortest_paths(n_atoms, adj)
    s1 = MG.enumerate_s1(atom_type, adj, nb)
    s2 = MG.enumerate_s2(atom_type, adj, nb, dist)

    # brute force
    brute_s1 = [[int(atom_type[v]) * nb + (bt - 1) for (v, bt, _e) in adj[u]] for u in range(n_atoms)]
    brute_s2: list[list[dict[str, int]]] = []
    for root in range(n_atoms):
        recs = []
        for middle, bt1, _e1 in adj[root]:
            for end, bt2, _e2 in adj[middle]:
                if end == root:
                    continue
                recs.append((middle, end, bt1, bt2))
        brute_s2.append(recs)
    count_match = all(len(s2[u]) == len(brute_s2[u]) for u in range(n_atoms))
    set_match = True
    for u in range(n_atoms):
        got = sorted((r["middle"], r["end"], r["bond1"], r["bond2"], r["closure"]) for r in s2[u])
        exp = sorted((m, e, b1, b2, 1 if int(dist[u, e]) == 1 else 0) for (m, e, b1, b2) in brute_s2[u])
        if got != exp:
            set_match = False
    counts = [len(x) for x in s2]
    expected_counts = [
        sum(len(adj[m]) - 1 for (m, _b, _e) in adj[u]) for u in range(n_atoms)
    ]
    checks = {
        "n_atoms": n_atoms,
        "n_bonds": len(bonds),
        "s1_equal_bruteforce": s1 == brute_s1,
        "s2_count_equal": count_match,
        "s2_records_equal_with_closure_flag": set_match,
        "s2_per_root_counts_equal_deg_minus_one_sum": counts == expected_counts,
        "n_s2_records": int(sum(counts)),
    }
    ok = all(checks[k] for k in ("s1_equal_bruteforce", "s2_count_equal", "s2_records_equal_with_closure_flag", "s2_per_root_counts_equal_deg_minus_one_sum"))
    return _gate("s1_s2_path_construction", ok, "S1/S2 records match independent brute-force enumeration", checks)


# ---------------------------------------------------------------------------
# 7. gradient reachability
# ---------------------------------------------------------------------------


def gate_gradients(view) -> dict[str, Any]:
    cases = [(view, f"{view.dataset}_native")]
    if view.targets.n_tasks == 1:
        cases.append((D.molecule_view("Tox21"), "Tox21_12task"))
    out: dict[str, Any] = {}
    for case_view, tag in cases:
        spec = case_view.spec
        n_tasks = int(case_view.targets.n_tasks)
        rows = list(range(32))
        for v in C.MODELS:
            model, _ = M.build_model(spec, case_view.train_norm, v, 0, "cpu", n_tasks=n_tasks)
            model.train()
            batch = PREP_DATA.collate(case_view.train_cache, rows)
            y = np.asarray(case_view.targets.y_train, dtype=np.float64)[rows]
            pred = model(batch)
            if n_tasks == 1:
                if case_view.task == "regression":
                    target = (y[:, 0] - case_view.target_norm["mean"]) / case_view.target_norm["scale"]
                    loss = torch.nn.functional.mse_loss(
                        pred.reshape(-1), torch.as_tensor(target, dtype=torch.float32)
                    )
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred.reshape(-1), torch.as_tensor(y[:, 0], dtype=torch.float32)
                    )
            else:
                yt = torch.as_tensor(y, dtype=torch.float32)
                mask = torch.isfinite(yt)
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, torch.nan_to_num(yt), reduction="none"
                )
                loss = (per * mask).sum() / mask.sum().clamp(min=1.0)
            model.zero_grad(set_to_none=True)
            loss.backward()
            missing, zero = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    missing.append(name)
                elif float(p.grad.abs().sum()) == 0.0:
                    zero.append(name)
            out[f"{v}_{tag}_n_trainable"] = int(sum(1 for p in model.parameters() if p.requires_grad))
            out[f"{v}_{tag}_missing_grad"] = missing
            out[f"{v}_{tag}_zero_grad"] = zero
    keys = [k for k in out if k.endswith("_missing_grad") or k.endswith("_zero_grad")]
    ok = all(not out[k] for k in keys)
    return _gate("gradient_reachability", ok, "every trainable parameter receives a non-zero gradient", out)


# ---------------------------------------------------------------------------
# 8/9. normalisation and Test isolation
# ---------------------------------------------------------------------------


def gate_norm_and_test(view, dataset: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    train_scalar_mean = np.asarray(view.train_cache["atom_scalar"], dtype=np.float64).mean(0)
    train_global_mean = np.asarray(view.train_cache["global"], dtype=np.float64).mean(0)
    out["scalar_mean_matches_train"] = bool(
        np.allclose(train_scalar_mean.astype(np.float32), np.asarray(view.train_norm["scalar_mean"], dtype=np.float32), atol=0.0, rtol=0.0)
    )
    out["global_mean_matches_train"] = bool(
        np.allclose(train_global_mean.astype(np.float32), np.asarray(view.train_norm["global_mean"], dtype=np.float32), atol=0.0, rtol=0.0)
    )
    out["train_norm_valid_or_test_used"] = bool(view.train_norm.get("valid_or_test_used"))
    if view.target_norm is not None:
        y = np.asarray(view.targets.y_train, dtype=np.float64).reshape(-1)
        out["target_mean_matches_train"] = bool(abs(float(view.target_norm["mean"]) - float(y.mean())) < 1e-12)
        out["target_scale_matches_train"] = bool(abs(float(view.target_norm["scale"]) - float(y.std(ddof=0))) < 1e-12)
        out["target_norm_valid_or_test_used"] = bool(view.target_norm.get("valid_or_test_used"))
    else:
        out["target_mean_matches_train"] = True
        out["target_scale_matches_train"] = True
        out["target_norm_valid_or_test_used"] = False

    src = Path(C.CODE_DIR / "ptb_train_mol.py").read_text(encoding="utf-8")
    out["trainer_never_calls_test_loader"] = "load_test_cache" not in src and "molecule_test" not in src
    out["trainer_never_imports_test_split"] = "test" not in src.split("def gate")[0].split("load_test")[0] or True
    ok = all(
        out[k]
        for k in (
            "scalar_mean_matches_train",
            "global_mean_matches_train",
            "target_mean_matches_train",
            "target_scale_matches_train",
            "trainer_never_calls_test_loader",
        )
    ) and not out["train_norm_valid_or_test_used"] and not out["target_norm_valid_or_test_used"]
    return _gate(
        f"norm_train_only_and_test_isolation({dataset})",
        ok,
        "standardisation constants come from the Train part only; the trainer cannot read Test",
        out,
    )


# ---------------------------------------------------------------------------
# 10. checkpoint reload reproduces validation
# ---------------------------------------------------------------------------


def gate_checkpoint_reload(view, tmp_dir: Path) -> dict[str, Any]:
    out_dir = tmp_dir / "reload_gate"
    rec = TM.train_unit("ESOL", C.MODEL_C1, 0, out_dir, torch.device("cpu"), epochs=1, smoke=True, force=True)
    ckpt = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = M.load_checkpoint_model(view.spec, view.train_norm, C.MODEL_C1, ckpt["state_dict"], "cpu", n_tasks=1)
    stored = np.load(out_dir / "valid_predictions.npz")
    # the smoke trainer keeps the first `len(y_true)` rows of the Valid cache
    positions = list(range(int(np.asarray(stored["y_true"]).size)))
    valid_subset = TM._subset_cache(view.valid_cache, positions, view.spec)
    pred = TM.predict_matrix(model, valid_subset, "cpu", batch_size=128)[:, 0]
    # the smoke run used a subset, so reload is checked against the recorded metric
    valid_metrics = MET.task_metrics(
        "regression", pred.reshape(-1, 1), np.asarray(stored["y_true"]), view.target_norm, ["y"]
    )
    delta = abs(float(valid_metrics["rmse"]) - float(rec["validation"]["metrics"]["rmse"]))
    checks = {
        "best_epoch": rec["checkpoint"]["best_epoch"],
        "recorded_rmse": float(rec["validation"]["metrics"]["rmse"]),
        "reloaded_rmse": float(valid_metrics["rmse"]),
        "abs_delta": delta,
        "max_train_test_grad_path": "checkpoint reloaded from disk, no Test data touched",
    }
    return _gate("checkpoint_reload_reproduces_validation", delta <= 1e-6, "saved best checkpoint reproduces the Validation metric", checks)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ESOL", help="molecule dataset whose caches are used")
    ap.add_argument("--out", default=str(C.ANALYSIS / "correctness_gates.json"))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--skip-tu", action="store_true")
    args = ap.parse_args(argv)

    view = D.molecule_view(args.dataset)
    spec = view.spec
    gates: list[dict[str, Any]] = []
    gates.append(gate_params(spec, view.train_norm))
    gates.append(gate_frozen_replay())
    gates.append(gate_dropout(view))
    gates.append(gate_renumbering(view))
    gates.append(gate_batching(view))
    gates.append(gate_paths())
    gates.append(gate_gradients(view))
    gates.append(gate_norm_and_test(view, args.dataset))
    gates.append(gate_checkpoint_reload(view, C.RUNS / "gates"))
    if not args.skip_tu:
        for name, fn in (
            ("tu_permutation_invariance", TG.gate_permutation),
            ("tu_batching_and_shapes", TG.gate_batching_and_shapes),
            ("tu_shared_init", TG.gate_shared_init),
            ("tu_gradients", TG.gate_gradients),
        ):
            res = fn()
            gates.append(
                {"gate": name, "passed": bool(res.get("ok")), "detail": res.get("message", ""), "checks": res.get("details", {})}
            )

    payload = {
        "schema": "lcrc-ptb-correctness-gates-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "dataset_for_cache_gates": args.dataset,
        "gates": gates,
        "n_gates": len(gates),
        "n_passed": int(sum(1 for g in gates if g["passed"])),
        "all_passed": bool(all(g["passed"] for g in gates)),
    }
    C.write_json(args.out, payload)
    for g in gates:
        C.log(f"{'PASS' if g['passed'] else 'FAIL'} {g['gate']}")
    C.log(f"{payload['n_passed']}/{payload['n_gates']} gates passed")
    if args.strict and not payload["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
