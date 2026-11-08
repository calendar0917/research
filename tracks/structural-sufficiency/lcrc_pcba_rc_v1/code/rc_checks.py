"""The eleven-plus-one pre-training correctness checks of the RC round.

Every check is a pure function returning ``{"name", "passed", "details"}`` so
that the same code produces both the pytest assertions (``tests/``) and the
machine-readable ``analysis/preflight.json`` record written by
``rc_audit.py checks``.  Nothing here trains a formal model.

Check list (numbering follows PROTOCOL.md section 4):

 1. W2 init identical to the historical seed-0 construction; REAL == BLIND W2 part
 2. REAL and BLIND new parameter names / shapes / counts identical
 3. zero-init branch reproduces the W2 forward (numerically)
 4. atom renumbering leaves the prediction unchanged
 5. two molecules in one batch never share a role co-occurrence
 6. distance buckets are mutually exclusive and cover every same-component pair
 7. disconnected components never produce a finite distance
 8. uniform role distributions make REAL and BLIND inputs identical
 9. a role<->distance rebinding changes REAL but leaves BLIND unchanged
10. the new branch really is on the gradient path (after a few steps)
11. brute-force CPU pair enumeration equals the batched GPU implementation
12. masked BCE / official AP / Train-only isolation match the inherited W2 code
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rc_common as C  # noqa: E402  (sets up the inherited code dirs)
import bs_model as BSM  # noqa: E402
import rc_dist as D  # noqa: E402
import rc_model as M  # noqa: E402
import rc_relations as REL  # noqa: E402


def _result(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def rebuild_pairs_from_bonds(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the packed distance payload from the batch's own bond_index.

    Used to keep a *relabelled* batch internally consistent (the production
    cache is built once in the original numbering).
    """
    atom_batch = batch["atom_batch"].to(torch.int64).cpu().numpy()
    counts = batch["atom_counts"].to(torch.int64).cpu().numpy()
    off = np.concatenate([[0], np.cumsum(counts)])[:-1].astype(np.int64)
    n_graphs = int(batch["n_graphs"])
    edges_all = batch["bond_index"].to(torch.int64).cpu().numpy().reshape(-1, 2)
    bond_graph = atom_batch[edges_all[:, 0]] if edges_all.size else np.zeros(0, dtype=np.int64)
    parts = []
    pair_counts = []
    for g in range(n_graphs):
        n = int(counts[g])
        sel = bond_graph == g
        local = edges_all[sel] - off[g]
        codes = D.molecule_bucket_codes(n, local)
        parts.append(codes)
        pair_counts.append(int(codes.size))
    packed = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    return np.ascontiguousarray(packed, dtype=np.uint8), np.asarray(pair_counts, dtype=np.int64)


def set_pairs_from_bonds(batch: dict[str, Any]) -> dict[str, Any]:
    packed, counts = rebuild_pairs_from_bonds(batch)
    batch["pair_bucket"] = torch.from_numpy(packed)
    batch["pair_counts"] = torch.from_numpy(counts)
    return batch


# ---------------------------------------------------------------------------
# 1 / 2 -- initialisation identity
# ---------------------------------------------------------------------------


def check_init(spec, norm, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    base, base_audit = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=int(seed), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, spec, norm, C.N_TASKS, seed=int(seed), device=device)
        models[key] = model
        audits[key] = audit

    w2_equal = True
    per_arm_base_equal = {}
    for key, model in models.items():
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        same = all(
            torch.equal(state[k], base_state[k])
            for k in base_state
            if not k.startswith("branch.")
        )
        per_arm_base_equal[key] = bool(same)
        w2_equal &= bool(same)

    inv = {key: audits[key]["new_parameter_inventory"] for key in audits}
    same_names = sorted(inv["REAL"]) == sorted(inv["BLIND"])
    same_shapes = all(inv["REAL"][k]["shape"] == inv["BLIND"][k]["shape"] for k in inv["REAL"])
    same_counts = all(inv["REAL"][k]["numel"] == inv["BLIND"][k]["numel"] for k in inv["REAL"])
    same_totals = audits["REAL"]["total_parameters"] == audits["BLIND"]["total_parameters"]

    details = {
        "base_state_sha256": {key: audits[key]["base_state_sha256"] for key in audits},
        "fresh_w2_state_sha256": M.state_sha256(base_state),
        "historical_state_sha256": C.W2["base_state_sha256_at_init"],
        "base_parameters": {key: audits[key]["base_parameters"] for key in audits},
        "new_parameters": {key: audits[key]["new_parameters"] for key in audits},
        "total_parameters": {key: audits[key]["total_parameters"] for key in audits},
        "parameter_ceiling": int(C.PARAMETER_CEILING),
        "new_state_keys": {"REAL": audits["REAL"]["new_state_keys"], "BLIND": audits["BLIND"]["new_state_keys"]},
        "new_parameter_inventory": inv,
        "w2_prefix_equal": bool(w2_equal),
        "per_arm_base_bitwise_equal": per_arm_base_equal,
        "new_names_identical": bool(same_names),
        "new_shapes_identical": bool(same_shapes),
        "new_counts_identical": bool(same_counts),
        "totals_identical": bool(same_totals),
        "historical_init_check": {key: audits[key].get("historical_init_check") for key in audits},
        "global_rng_unchanged": all(audits[key]["new_module_init"]["global_rng_unchanged"] for key in audits),
        "branch_fc2_zero_init": all(audits[key]["new_module_init"]["fc2_is_zero"] for key in audits),
        "base_trunk_init_fingerprint": audits["REAL"]["base_trunk_init_fingerprint"],
        "historical_trunk_init_fingerprint": C.W2["trunk_init_fingerprint"],
    }
    passed = bool(
        w2_equal
        and same_names
        and same_shapes
        and same_counts
        and same_totals
        and all(audits[k]["total_parameters"] <= C.PARAMETER_CEILING for k in audits)
        and all(audits[k]["new_module_init"]["fc2_is_zero"] for k in audits)
    )
    return _result("01_02_init_and_parameter_identity", passed, **details)


# ---------------------------------------------------------------------------
# 3 -- zero-init forward equals W2
# ---------------------------------------------------------------------------


def check_zero_init_forward(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=int(seed), device=device)
    base.eval()
    with torch.no_grad():
        ref = base(batch)
    results = {}
    passed = True
    for key in C.MODEL_ORDER:
        model, _ = M.build_model(key, spec, norm, C.N_TASKS, seed=int(seed), device=device)
        model.eval()
        with torch.no_grad():
            logits, aux = model(batch, return_aux=True)
            zeroed = model(batch, residual_zero=True)
        resid_max = float(aux["residual_logits"].abs().max().item())
        d_zero = float((logits - ref).abs().max().item())
        d_flag = float((zeroed - ref).abs().max().item())
        scale = float(ref.abs().max().item())
        ok = d_zero == 0.0 and d_flag == 0.0 and resid_max == 0.0
        passed &= bool(ok)
        results[key] = {
            "max_abs_diff": d_zero,
            "max_abs_diff_residual_zero_path": d_flag,
            "residual_max_abs": resid_max,
            "reference_max_abs": scale,
            "exact": bool(ok),
        }
    return _result("03_zero_init_forward_equals_w2", passed, per_arm=results)


# ---------------------------------------------------------------------------
# 4 -- atom renumbering
# ---------------------------------------------------------------------------


def permute_batch_consistent(batch: Mapping[str, Any], *, seed: int = 20261109) -> tuple[dict[str, Any], torch.Tensor]:
    """``bs_model.permute_atoms`` + a matching rebuild of the pair payload."""
    permuted, p = BSM.permute_atoms(batch, seed=seed)
    permuted = set_pairs_from_bonds(permuted)
    return permuted, p


def check_permutation(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    results = {}
    passed = True
    permuted, _p = permute_batch_consistent(batch)
    for key in ("REAL", "BLIND"):
        model, _ = M.build_model(key, spec, norm, C.N_TASKS, seed=int(seed), device=device)
        model.eval()
        with torch.no_grad():
            a = model(batch)
            b = model(permuted)
        diff = float((a - b).abs().max().item())
        scale = float(a.abs().max().item())
        rel = diff / max(scale, 1e-12)
        ok = rel <= 1e-4
        passed &= bool(ok)
        results[key] = {"max_abs_diff": diff, "max_abs_reference": scale, "relative": rel, "tolerance": 1e-4}
    return _result("04_atom_renumbering_invariance", passed, per_arm=results)


# ---------------------------------------------------------------------------
# 5 -- no cross-molecule co-occurrence
# ---------------------------------------------------------------------------


def random_roles(n_atoms: int, *, seed: int = 3, device: Any = "cpu") -> torch.Tensor:
    """Deterministic per-atom role distribution (test helper)."""
    gen = torch.Generator().manual_seed(int(seed))
    logits = torch.randn(int(n_atoms), C.N_ROLES, generator=gen)
    return torch.softmax(logits, dim=-1).to(device)


def _relation_x(model: M.RCModel, batch: Mapping[str, Any], R: torch.Tensor) -> torch.Tensor:
    x, _aux = model.branch.relation_input(batch, R)
    return x


def check_no_cross_molecule(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=int(seed), device=device)
    model.eval()
    n_graphs = int(batch["n_graphs"])
    counts = batch["atom_counts"].tolist()
    off = batch["atom_off"].to(torch.int64).tolist()
    results = []
    passed = True
    R_full = random_roles(int(batch["atom_feat"].shape[0]), device=device)
    with torch.no_grad():
        full = _relation_x(model, batch, R_full)
        for g in range(n_graphs):
            single = _clone_batch(batch)
            a0, a1 = int(off[g]), int(off[g]) + int(counts[g])
            atom_sel = torch.arange(a0, a1)
            remap = torch.full((int(batch["atom_feat"].shape[0]),), -1, dtype=torch.long)
            remap[atom_sel] = torch.arange(a1 - a0)
            single["atom_batch"] = torch.zeros(a1 - a0, dtype=torch.long)
            single["atom_counts"] = torch.tensor([int(counts[g])], dtype=torch.long)
            single["atom_off"] = torch.tensor([0], dtype=torch.long)
            single["n_graphs"] = 1
            for k in ("atom_feat", "atom_scalar", "atom_type"):
                single[k] = batch[k][atom_sel]
            for k in ("global",):
                single[k] = batch[k][g : g + 1]
            # pairs of this molecule only
            pair_off = int(torch.cumsum(batch["pair_counts"].to(torch.int64), 0)[g - 1]) if g else 0
            n_pairs = int(batch["pair_counts"][g])
            single["pair_counts"] = torch.tensor([n_pairs], dtype=torch.long)
            single["pair_bucket"] = batch["pair_bucket"][pair_off : pair_off + n_pairs].clone()
            x_single = _relation_x(model, single, R_full[atom_sel])
            diff = float((x_single[0] - full[g]).abs().max().item())
            ok = diff <= 1e-5
            passed &= bool(ok)
            results.append({"graph": g, "max_abs_diff": diff, "pairs": n_pairs, "exact_tolerance": 1e-5})
    return _result("05_no_cross_molecule_cooccurrence", passed, per_graph=results)


# ---------------------------------------------------------------------------
# 6 / 7 -- bucket partition and disconnected components
# ---------------------------------------------------------------------------


#: Small reference graphs used by several checks and by the unit tests.
TINY_GRAPHS: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "chain4_plus_chord": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
    "chain5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    "ring6": (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
    "two_components": (6, [(0, 1), (1, 2), (3, 4)]),
    "iso_atom": (3, [(0, 1)]),
}


def check_bucket_partition(dist: Mapping[str, Any] | None, *, n_sample: int = 200, seed: int = 20261110) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    synthetic = {}
    for name, (n, edges) in TINY_GRAPHS.items():
        codes = D.molecule_bucket_codes(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        ref = D.brute_force_distances(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        iu = np.triu_indices(n, 1)
        ref_codes = D.distance_to_code(ref[iu])
        ok = bool(codes.size == n * (n - 1) // 2) and bool(np.array_equal(codes, ref_codes))
        synthetic[name] = {"n_atoms": n, "n_pairs": int(codes.size), "codes_match": ok}
        if not ok:
            failures.append({"graph": name})

    real = {"checked_rows": 0, "mismatched_rows": 0}
    if dist is not None:
        gen = np.random.default_rng(int(seed))
        rows = np.sort(gen.choice(int(dist["meta"]["n_rows"]), size=min(n_sample, int(dist["meta"]["n_rows"])), replace=False))
        import lr_common as LC

        cache_dir = LC.CACHE_PATH
        atom_ptr = np.load(cache_dir / "atom_ptr.npy")
        bond_ptr = np.load(cache_dir / "bond_ptr.npy")
        bond_index = np.load(cache_dir / "bond_index.npy", mmap_mode="r")
        for row in rows:
            a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
            b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
            edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
            n = a1 - a0
            packed = dist["bucket"][int(dist["ptr"][row]) : int(dist["ptr"][row + 1])]
            ref = D.brute_force_distances(n, edges)
            iu = np.triu_indices(n, 1)
            ref_codes = D.distance_to_code(ref[iu])
            real["checked_rows"] += 1
            if not np.array_equal(packed, ref_codes):
                real["mismatched_rows"] += 1
                failures.append({"row": int(row)})
            if np.any(ref_codes == C.UNREACHABLE) != np.any(packed == C.UNREACHABLE):
                failures.append({"row": int(row), "note": "connectivity sentinel mismatch"})
            # exclusivity: codes are single-valued by construction; check the
            # multiset of distances maps to the declared buckets
            d = ref[iu]
            for index, (_label, lo, hi) in enumerate(C.BUCKETS):
                sel = np.isfinite(d) & (d >= lo) & ((d <= hi) if hi is not None else True)
                if not np.array_equal(sel, packed == index):
                    failures.append({"row": int(row), "bucket": index})
    details = {
        "synthetic_graphs": synthetic,
        "real_cache_sample": real,
        "checked_rows_total": real["checked_rows"],
        "mismatched_rows": real["mismatched_rows"],
        "bucket_definitions": [list(b) for b in C.BUCKETS],
        "unreachable_code": int(C.UNREACHABLE),
        "exclusivity": "every unordered pair carries exactly one byte, so exactly one bucket",
        "coverage": "every same-component pair is covered; different components use the 255 sentinel",
    }
    passed = not failures and all(v["codes_match"] for v in synthetic.values())
    details["failures"] = failures[:10]
    return _result("06_07_bucket_partition_and_disconnected", passed, **details)


# ---------------------------------------------------------------------------
# 8 / 9 -- REAL vs BLIND behaviour
# ---------------------------------------------------------------------------


def check_uniform_roles_equivalence(*, atol: float = 1e-6) -> dict[str, Any]:
    n_graphs = 2
    atom_counts = torch.tensor([5, 4], dtype=torch.long)
    pairs = [
        (0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3),
        (1, 2, 0), (1, 3, 1), (1, 4, 2), (2, 3, 0), (2, 4, 1), (3, 4, 0),
        (5, 6, 0), (5, 7, 1), (5, 8, 2), (6, 7, 0), (6, 8, 1), (7, 8, 0),
    ]
    R = torch.full((9, C.N_ROLES), 1.0 / C.N_ROLES)
    xs = {}
    for kind in ("real", "blind"):
        x, aux = REL.relation_input_from_roles(R, n_graphs=n_graphs, atom_counts=atom_counts, pairs=pairs, kind=kind)
        xs[kind] = x
    diff = float((xs["real"] - xs["blind"]).abs().max().item())
    return _result(
        "08_uniform_roles_real_equals_blind",
        diff <= atol,
        max_abs_diff=diff,
        tolerance=atol,
        note="with identical roles every T_d collapses to 144*r r^T, so the bucket binding carries no information",
    )


def check_binding_change() -> dict[str, Any]:
    """Same role multiset, same bucket counts, different role<->distance binding."""
    n_graphs = 1
    atom_counts = torch.tensor([5], dtype=torch.long)
    # 5 atoms: 4 d1 pairs (chain), 3 d2, 2 d3, 1 d4 -> the same counts in both arms
    roles = torch.tensor(
        [
            [0.70, 0.10, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.005, 0.005],
            [0.05, 0.60, 0.10, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01],
            [0.05, 0.05, 0.55, 0.10, 0.08, 0.05, 0.04, 0.02, 0.02, 0.02, 0.01, 0.01],
            [0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05],
        ],
        dtype=torch.float64,
    )
    roles = roles / roles.sum(dim=1, keepdim=True)
    # binding A: distances 1..4 along the chain 0-1-2-3-4
    chain_pairs = [
        (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
        (0, 2, 1), (1, 3, 1), (2, 4, 1),
        (0, 3, 2), (1, 4, 2),
        (0, 4, 3),
    ]
    # binding B: the same bucket sizes, different atom assignment
    # (d1 pairs now involve atom 4 instead of atom 0 wherever possible)
    swapped = list(range(5))
    swapped[0], swapped[4] = 4, 0
    rebinding = [(swapped[u], swapped[v], b) for (u, v, b) in chain_pairs]
    rebinding = [(min(u, v), max(u, v), b) for (u, v, b) in rebinding]
    out = {}
    for kind in ("real", "blind"):
        xa, auxa = REL.relation_input_from_roles(
            roles, n_graphs=n_graphs, atom_counts=atom_counts, pairs=chain_pairs, kind=kind
        )
        xb, auxb = REL.relation_input_from_roles(
            roles, n_graphs=n_graphs, atom_counts=atom_counts, pairs=rebinding, kind=kind
        )
        counts_equal = bool(torch.equal(auxa["ordered_counts"], auxb["ordered_counts"]))
        w_equal = bool(torch.allclose(auxa["w"], auxb["w"], atol=0.0, rtol=0.0))
        out[kind] = {
            "max_abs_diff": float((xa - xb).abs().max().item()),
            "bucket_counts_identical": counts_equal,
            "w_identical": w_equal,
        }
    passed = (
        out["real"]["max_abs_diff"] > 1e-3
        and out["blind"]["max_abs_diff"] <= 1e-7
        and out["real"]["bucket_counts_identical"]
        and out["real"]["w_identical"]
        and out["blind"]["bucket_counts_identical"]
    )
    return _result("09_binding_change_real_differs_blind_same", passed, per_arm=out)


# ---------------------------------------------------------------------------
# 10 -- gradient path
# ---------------------------------------------------------------------------


def check_gradient_path(spec, norm, batches: Sequence[Mapping[str, Any]], *, steps: int = 3, device: str = "cpu") -> dict[str, Any]:
    import bs_train as BT
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=C.SEED, device=device)
    optimizer, _ = BSM.make_optimizer(model)
    model.train()
    trace = []
    for step in range(int(steps)):
        batch = batches[step % len(batches)]
        y = torch.zeros((int(batch["n_graphs"]), C.N_TASKS))
        y[:, :] = float("nan")
        y[:, 0] = torch.randint(0, 2, (int(batch["n_graphs"]),)).float()
        y[:, 1] = 1.0
        out = BT.train_step(model, optimizer, batch, y, global_step=step, clip=5.0)
        grads = {}
        for name, param in model.named_parameters():
            if name.startswith("branch.") and param.grad is not None:
                grads[name] = float(param.grad.detach().float().norm().item())
        trace.append({"step": step, "loss": out["loss"], "branch_grad_norm": grads})
    final = trace[-1]["branch_grad_norm"]
    role_ok = final.get("branch.role_proj.weight", 0.0) > 0.0
    combiner_ok = final.get("branch.combiner.fc1.weight", 0.0) > 0.0
    readout_ok = final.get("branch.combiner.fc2.weight", 0.0) > 0.0
    task_ok = final.get("branch.task_embedding", 0.0) > 0.0
    passed = bool(role_ok and combiner_ok and readout_ok and task_ok)
    return _result(
        "10_gradient_path_active",
        passed,
        steps=int(steps),
        trace=trace,
        role_projection_grad_nonzero=bool(role_ok),
        combiner_fc1_grad_nonzero=bool(combiner_ok),
        combiner_fc2_grad_nonzero=bool(readout_ok),
        task_embedding_grad_nonzero=bool(task_ok),
        note="fc2 is zero at step 0, so the branch trunk only starts receiving gradient from step 1",
    )


# ---------------------------------------------------------------------------
# 11 -- brute force vs batched
# ---------------------------------------------------------------------------


def brute_force_relation_input(
    R: np.ndarray,
    atom_counts: Sequence[int],
    pairs: Sequence[tuple[int, int, int]],
    kind: str,
) -> np.ndarray:
    """Explicit double loop over every atom pair; reference for check 11."""
    R = np.asarray(R, dtype=np.float64)
    n_graphs = len(atom_counts)
    off = np.concatenate([[0], np.cumsum(atom_counts)])[:-1].astype(int)
    out = np.zeros((n_graphs, C.X_DIM), dtype=np.float64)
    rows, cols = np.triu_indices(C.N_ROLES, 0)
    for g in range(n_graphs):
        n = int(atom_counts[g])
        a0 = int(off[g])
        A = np.zeros((C.N_BUCKETS, n, n), dtype=np.float64)
        for u, v, b in pairs:
            u = int(u)
            if not (a0 <= u < a0 + n):
                continue
            if int(b) < C.N_BUCKETS:
                lu, lv = u - a0, int(v) - a0
                A[int(b), lu, lv] = 1.0
                A[int(b), lv, lu] = 1.0
        Rg = R[a0 : a0 + n]
        counts = np.asarray([A[d].sum() for d in range(C.N_BUCKETS)], dtype=np.float64)
        n_all = counts.sum()
        w = counts / max(n_all, 1.0)
        T_all = 144.0 * (Rg.T @ A.sum(axis=0) @ Rg) / max(n_all, 1.0)
        feats = []
        for d in range(C.N_BUCKETS):
            if counts[d] > 0:
                T = 144.0 * (Rg.T @ A[d] @ Rg) / counts[d]
            else:
                T = np.zeros((C.N_ROLES, C.N_ROLES))
            if kind == "real":
                F = np.sqrt(w[d]) * T
            else:
                F = np.sqrt(w[d]) * T_all if counts[d] > 0 else np.zeros_like(T)
            feats.append(F[rows, cols])
        out[g] = np.concatenate(feats + [w, Rg.mean(axis=0), [np.log1p(n)]])
    return out


def check_cpu_vs_batched(spec, norm, batch, *, device: str = "cpu", seed: int = C.SEED) -> dict[str, Any]:
    """The batched implementation must equal an explicit pair-by-pair enumeration."""
    geom = REL.pair_geometry(batch)
    u = geom["u"]
    v = geom["v"]
    b = geom["bucket"]
    pairs = [(int(u[i]), int(v[i]), int(b[i])) for i in range(u.numel())]
    counts = batch["atom_counts"].tolist()
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"].to(torch.int64)
    atom_counts = batch["atom_counts"].to(torch.int64)
    R_cpu = random_roles(int(batch["atom_feat"].shape[0]), seed=seed)
    results: dict[str, Any] = {}
    passed = True
    for kind in ("real", "blind"):
        x_cpu, _aux = REL.relation_input(
            R_cpu, u, v, geom["pair_graph"], b, atom_batch, atom_counts, n_graphs, kind=kind
        )
        ref = brute_force_relation_input(R_cpu.numpy(), counts, pairs, kind)
        diff = float(np.abs(x_cpu.numpy() - ref).max())
        denom = max(float(np.abs(ref).max()), 1e-12)
        ok = diff / denom <= 1e-6
        passed &= bool(ok)
        results[f"cpu_{kind}"] = {"max_abs_diff": diff, "relative": diff / denom, "tolerance": 1e-6}
        if device != "cpu":
            batch_d = {k: (val.to(device) if torch.is_tensor(val) else val) for k, val in batch.items()}
            geom_d = REL.pair_geometry(batch_d)
            R_d = R_cpu.to(device)
            x_gpu, _ = REL.relation_input(
                R_d,
                geom_d["u"],
                geom_d["v"],
                geom_d["pair_graph"],
                geom_d["bucket"],
                batch_d["atom_batch"].to(torch.int64),
                batch_d["atom_counts"].to(torch.int64),
                n_graphs,
                kind=kind,
            )
            diff_g = float(np.abs(x_gpu.detach().cpu().numpy() - ref).max())
            ok_g = diff_g / denom <= 1e-6
            passed &= bool(ok_g)
            results[f"gpu_{kind}"] = {"max_abs_diff": diff_g, "relative": diff_g / denom, "tolerance": 1e-6}
    return _result("11_bruteforce_equals_batched", passed, per_case=results)


# ---------------------------------------------------------------------------
# 12 -- metric / isolation parity
# ---------------------------------------------------------------------------


def check_metrics_and_isolation(task: Any | None, batch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import pc_metrics as MET

    gen = torch.Generator().manual_seed(17)
    logits = torch.randn(64, C.N_TASKS, generator=gen)
    y = torch.randint(0, 2, (64, C.N_TASKS), generator=gen).float()
    y[torch.rand_like(y) < 0.6] = float("nan")
    loss = MET.masked_bce_loss(logits, y)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logits[torch.isfinite(y)], y[torch.isfinite(y)])
    loss_equal = float((loss - manual).abs()) == 0.0

    probs = torch.sigmoid(logits).numpy()
    y_np = y.numpy()
    official = MET.official_ap(y_np, probs)
    independent = MET.per_task_ap(y_np, probs)
    ap_delta = abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))

    details: dict[str, Any] = {
        "masked_bce_equals_manual": bool(loss_equal),
        "masked_bce_value": float(loss),
        "official_ap": float(official["value"]),
        "independent_ap": float(independent["mean_over_scored_tasks"]),
        "official_vs_independent_abs_delta": float(ap_delta),
        "official_source": official["source"],
        "collate_keys_are_inherited_plus_pairs": sorted(
            set(batch or {}) - {"pair_bucket", "pair_counts"}
        ),
    }
    passed = bool(loss_equal and ap_delta < 1e-9)
    if task is not None:
        test_labels_nan = not bool(np.isfinite(task.labels[np.asarray(task.test_rows, dtype=np.int64)]).any())
        train_valid_disjoint = (
            np.intersect1d(task.train_rows, task.valid_rows).size == 0
            and np.intersect1d(task.train_rows, task.test_rows).size == 0
            and np.intersect1d(task.valid_rows, task.test_rows).size == 0
        )
        plan = C.BC.SD.make_plan40(task, seed=C.SEED)
        plan_sha_ok = plan[0]["global_plan_sha256"] == C.W2["global_plan_sha256"]
        train_sorted = np.sort(np.asarray(task.train_rows, dtype=np.int64))
        epochs_ok = all(np.array_equal(np.sort(np.asarray(p["rows"], np.int64)), train_sorted) for p in plan)
        details.update(
            {
                "test_labels_all_nan_in_training_artifact": bool(test_labels_nan),
                "split_partitions_disjoint": bool(train_valid_disjoint),
                "order_plan_sha256_matches_w2": bool(plan_sha_ok),
                "every_epoch_is_a_train_permutation": bool(epochs_ok),
                "global_plan_sha256": plan[0]["global_plan_sha256"],
            }
        )
        passed &= bool(test_labels_nan and train_valid_disjoint and plan_sha_ok and epochs_ok)
    return _result("12_metrics_and_train_only_isolation", passed, **details)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_all(
    spec,
    norm,
    batch: Mapping[str, Any],
    *,
    task: Any | None = None,
    dist: Mapping[str, Any] | None = None,
    device: str = "cpu",
    grad_batches: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    results = [
        check_init(spec, norm, device=device),
        check_zero_init_forward(spec, norm, batch, device=device),
        check_permutation(spec, norm, batch, device=device),
        check_no_cross_molecule(spec, norm, batch, device=device),
        check_bucket_partition(dist),
        check_uniform_roles_equivalence(),
        check_binding_change(),
        check_gradient_path(spec, norm, list(grad_batches or [batch]), steps=3, device=device),
        check_cpu_vs_batched(spec, norm, batch, device=device),
        check_metrics_and_isolation(task, batch),
    ]
    return {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "torch": torch.__version__,
        "checks": results,
        "all_passed": bool(all(r["passed"] for r in results)),
        "n_checks": len(results),
        "n_passed": int(sum(1 for r in results if r["passed"])),
    }


__all__ = [
    "TINY_GRAPHS",
    "brute_force_relation_input",
    "random_roles",
    "check_binding_change",
    "check_bucket_partition",
    "check_cpu_vs_batched",
    "check_gradient_path",
    "check_init",
    "check_metrics_and_isolation",
    "check_no_cross_molecule",
    "check_permutation",
    "check_uniform_roles_equivalence",
    "check_zero_init_forward",
    "permute_batch_consistent",
    "rebuild_pairs_from_bonds",
    "run_all",
    "set_pairs_from_bonds",
]
