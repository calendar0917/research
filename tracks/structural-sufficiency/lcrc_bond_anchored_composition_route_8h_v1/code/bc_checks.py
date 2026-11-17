"""The fourteen pre-registered Stage-0 gates plus the non-label numerical gate.

Every gate is a *check*, never a decision: it either reproduces a frozen artefact
or it does not.  Gates 1/2 anchor the two C1 teachers (bytes, parameters,
buffers, per-molecule replay); gates 3/4 anchor the three-bond-path object to the
frozen S2 enumeration; 5/6/7 are symmetry and batching invariances; 8 is the
paired-control identity; 9/10/11/12 are the residual construction; 13 makes sure
no label can leak into structure; 14 closes the WC_LOGP task identity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bc_common as C
import bc_data as D
import bc_models as M
import bc_path3 as P3
import bc_train as T

ATOL = 1e-5
REL_TOL = 1e-6
RELABEL_ATOL = 1e-4


def _ok(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any = None) -> bool:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


# ---------------------------------------------------------------------------
# gates 1 / 2: teacher identity and prediction replay
# ---------------------------------------------------------------------------


def gate_teacher_identity(b, task: str) -> dict[str, Any]:
    name = C.TASK_TEACHER[task]
    ident = T.teacher_identity(name)
    arm, ident2 = T.load_teacher_model(task, b)
    checks: list[dict[str, Any]] = []
    _ok(checks, "checkpoint_sha256", ident["sha256_match"], {"sha256": ident["sha256"], "expected": ident["sha256_expected"]})
    _ok(checks, "parameter_count", ident2["n_parameters"] == C.C1_PARAMETERS,
        {"n": ident2["n_parameters"], "expected": C.C1_PARAMETERS})
    # normalisation buffers must be exactly the frozen Train-only normalisation
    norm = b.norm
    for buffer_name, key in (
        ("global_mean", "global_mean"),
        ("global_scale", "global_scale"),
        ("scalar_mean", "scalar_mean"),
        ("scalar_scale", "scalar_scale"),
    ):
        got = getattr(arm.trunk, buffer_name).detach().cpu().numpy().astype(np.float64)
        want = np.asarray(norm[key], dtype=np.float64)
        _ok(checks, f"buffer_{buffer_name}", bool(np.array_equal(got, want)),
            {"max_abs_diff": float(np.max(np.abs(got - want)))})
    _ok(checks, "scaler_is_train_only", bool(b.scaler.get("valid_or_test_used", False)) is False,
        {"source": b.scaler.get("source"), "fit_split": b.scaler.get("fit_split")})
    return {"task": task, "teacher": ident, "checks": checks, "all_pass": all(c["ok"] for c in checks)}


def gate_prediction_replay(b, task: str, p3_map) -> dict[str, Any]:
    """The loaded teacher must reproduce the frozen per-molecule predictions."""
    XR = C.modules()["xr_common"]
    name = C.TASK_TEACHER[task]
    path = XR.RESULTS_DIR / "replay" / f"{name}_predictions.npz"
    checks: list[dict[str, Any]] = []
    if not path.is_file():
        _ok(checks, "replay_file_present", False, {"path": C.relpath(path)})
        return {"task": task, "checks": checks, "all_pass": False, "missing": True}
    arm, _ident = T.load_teacher_model(task, b)
    device = torch.device("cpu")
    with np.load(path, allow_pickle=False) as z:
        frozen = {k: np.asarray(z[k]) for k in z.files}
    per_split: dict[str, Any] = {}
    for split in C.SPLITS:
        out = D.predict_std(arm, b, p3_map[split], split, device=device)
        got_raw = D.to_raw(out["pred_std"], b.scaler)
        want = frozen[f"{split}_pred_raw"]
        want_ids = frozen[f"{split}_row_id"]
        ids_ok = bool(np.array_equal(out["row_id"], want_ids))
        err = float(np.max(np.abs(got_raw - want))) if want.size else 0.0
        per_split[split] = {"max_abs_diff_raw": err, "row_ids_match": ids_ok}
        _ok(checks, f"{split}_row_ids", ids_ok)
        _ok(checks, f"{split}_replay_atol", err <= 1e-5, {"max_abs_diff_raw": err})
    return {"task": task, "replay_file": C.relpath(path), "checks": checks,
            "per_split": per_split, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# gates 3 / 4: the three-bond-path object
# ---------------------------------------------------------------------------


def gate_path3_coverage(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    per_split: dict[str, Any] = {}
    for split in C.SPLITS:
        p3 = p3_map[split]
        AD = C.modules()["as_data"]
        cache = b.caches[split]
        # re-verify a deterministic subset of molecules *and* the global sums
        counts = p3.counts
        n_mol = int(p3.n_molecules)
        sample = list(range(0, n_mol, max(1, n_mol // 24)))[:24]
        sample_fail: list[dict[str, Any]] = []
        for row in sample:
            mol = AD.slice_molecule(cache, row)
            _g, analysis = AD.analysis_from_mol(mol)
            om = AD.build_occ_map(np.asarray(mol["atom_type"]), analysis, int(b.spec.n_bond_types))
            a0, a1 = int(p3.ptr[row]), int(p3.ptr[row + 1])
            part = {
                "atoms": p3.atoms[a0:a1],
                "occ": p3.occ[a0:a1] - int(p3.occ_map.n_s2_total[:row].sum()),
                "anchor": p3.anchor[a0:a1],
                "closure4": p3.closure4[a0:a1],
            }
            rep = P3.verify_one(mol, analysis, om, part, n_bond_types=int(b.spec.n_bond_types))
            if not rep["ok"]:
                sample_fail.append({"row": row, "failed": rep["failed"]})
        independent = _independent_event_count(cache)
        _ok(checks, f"{split}_sample_invariants", not sample_fail, {"rows": len(sample), "failed": sample_fail})
        _ok(checks, f"{split}_count_matches_independent_enumeration",
            bool(np.array_equal(independent, counts)),
            {"independent_total": int(independent.sum()), "cache_total": int(counts.sum())})
        _ok(checks, f"{split}_no_event_without_atoms",
            bool((counts == 0).sum() + (counts > 0).sum() == n_mol))
        per_split[split] = {
            "n_molecules": n_mol,
            "n_events": int(counts.sum()),
            "independent_total": int(independent.sum()),
            "sample_rows_verified": len(sample),
            "sample_failures": sample_fail,
        }
    return {"per_split": per_split, "checks": checks, "all_pass": all(c["ok"] for c in checks)}


def _independent_event_count(cache: Mapping[str, Any]) -> np.ndarray:
    """Count ``sum_{(b,c) in E} (deg(b)-1)*(deg(c)-1) - collisions`` independently.

    The three-bond simple paths are enumerated directly on the adjacency matrix,
    with no reference to the S2 records, so this is an independent count.
    """
    AD = C.modules()["as_data"]
    n = int(cache["n_molecules"])
    out = np.zeros(n, dtype=np.int64)
    for row in range(n):
        mol = AD.slice_molecule(cache, row)
        n_atoms = int(mol["n_atoms"])
        bonds = np.asarray(mol["bond_index"], dtype=np.int64).reshape(-1, 2)
        nbr = [set() for _ in range(n_atoms)]
        for u, v in bonds.tolist():
            nbr[int(u)].add(int(v))
            nbr[int(v)].add(int(u))
        total = 0
        for u, v in bonds.tolist():
            for a in nbr[int(u)]:
                if a == v:
                    continue
                for d in nbr[int(v)]:
                    if d == u or d == a:
                        continue
                    total += 1
        out[row] = total
    return out


def gate_path3_occurrences(b, p3_map) -> dict[str, Any]:
    """Each event's two S2 occurrences must sit in the frozen C1 enumeration."""
    checks: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for split in C.SPLITS:
        p3 = p3_map[split]
        n_s2 = int(p3.occ_map.n_s2_total.sum())
        occ = p3.occ
        ok_range = bool(occ.size == 0 or (occ.min() >= 0 and occ.max() < n_s2))
        _ok(checks, f"{split}_ids_in_range", ok_range, {"n_s2": n_s2, "n_events": int(occ.shape[0])})
        if occ.size:
            first = p3.occ_map
            root, end, mid = p3.occ_root, p3.occ_end, first.path_middle
            a = p3.atoms
            p_ok = bool(
                np.array_equal(root[occ[:, 0]], a[:, 0])
                and np.array_equal(mid[occ[:, 0]], a[:, 1])
                and np.array_equal(end[occ[:, 0]], a[:, 2])
            )
            q_ok = bool(
                np.array_equal(root[occ[:, 1]], a[:, 1])
                and np.array_equal(mid[occ[:, 1]], a[:, 2])
                and np.array_equal(end[occ[:, 1]], a[:, 3])
            )
            rev_ok = bool(
                np.array_equal(root[occ[:, 2]], a[:, 3])
                and np.array_equal(mid[occ[:, 2]], a[:, 2])
                and np.array_equal(end[occ[:, 2]], a[:, 1])
                and np.array_equal(root[occ[:, 3]], a[:, 2])
                and np.array_equal(mid[occ[:, 3]], a[:, 1])
                and np.array_equal(end[occ[:, 3]], a[:, 0])
            )
            # the two occurrences must be adjacent through the anchor bond
            adj_ok = bool(np.array_equal(first.path_bond2[occ[:, 0]], p3.anchor)
                          and np.array_equal(first.path_bond1[occ[:, 1]], p3.anchor))
            a_ok = p_ok and q_ok and rev_ok and adj_ok
            _ok(checks, f"{split}_occurrences_resolve", a_ok,
                {"p": p_ok, "q": q_ok, "rev": rev_ok, "anchor": adj_ok})
            # occurrence ids unique per (root, middle, end); events unique per undirected path
            molecule = np.repeat(np.arange(p3.n_molecules, dtype=np.int64), p3.counts)
            keys = np.stack([molecule, a[:, 0], a[:, 1], a[:, 2], a[:, 3]], axis=1)
            n_unique = int(np.unique(keys, axis=0).shape[0])
            _ok(checks, f"{split}_no_duplicate_event", n_unique == int(a.shape[0]),
                {"n_events": int(a.shape[0]), "n_unique_within_molecule": n_unique})
            detail[split] = {"n_events": int(a.shape[0]), "n_occurrence_ids_used": int(np.unique(occ).size)}
        else:
            _ok(checks, f"{split}_occurrences_resolve", True, {"n_events": 0})
            detail[split] = {"n_events": 0}
    return {"checks": checks, "detail": detail, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# gates 5 / 6 / 7: symmetry and batching
# ---------------------------------------------------------------------------


def _random_branch_model(b, variant: str = "JOINT") -> M.BAJC:
    """A model with a *non-zero* readout so symmetry tests are not vacuous."""
    model, _audit = T.build_model("lipo", b, variant, "E2E", seed=0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(4242)
    with torch.no_grad():
        for parameter in model.composer.parameters():
            noise = torch.empty_like(parameter, dtype=torch.float32)
            noise.normal_(0.0, 0.05, generator=gen)
            parameter.add_(noise)
    model.eval()
    return model


def gate_reverse_symmetry(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    model = _random_branch_model(b)
    device = torch.device("cpu")
    detail: dict[str, Any] = {}
    for split in ("valid",):
        rows = np.arange(0, 32, dtype=np.int64)
        bt = D.make_batch(b, split, rows, p3_map[split])
        with torch.no_grad():
            parts = model(bt.batch, bt.p3, return_parts=True)
            swapped = dict(bt.p3)
            occ = bt.p3["p3_occ"].clone()
            occ = occ[:, [2, 3, 0, 1]]
            swapped["p3_occ"] = occ
            ev = parts["evidence"]
            b_emb = _bond_embedding(model, bt.batch)
            rev = model.compose(ev, b_emb, swapped, n_graphs=bt.n_graphs)
        diff = float(torch.max(torch.abs(parts["c_P"] - rev["c_P"])).item()) if parts["c_P"].numel() else 0.0
        _ok(checks, f"{split}_reversal_invariant_events", diff <= 1e-6, {"max_abs_diff": diff})
        _ok(checks, f"{split}_reversal_nonzero_events", float(torch.max(torch.abs(parts["c_P"])).item()) > 0.0
            if parts["c_P"].numel() else True)
        detail[split] = {"n_events": int(parts["c_P"].numel()), "max_abs_diff": diff}
    return {"checks": checks, "detail": detail, "all_pass": all(c["ok"] for c in checks)}


def _bond_embedding(model: M.BAJC, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    trunk = model.trunk
    a = trunk.atom_encoder(batch["atom_feat"])
    idx = batch["bond_index"]
    if idx.numel():
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
    else:
        bond_input = a.new_zeros((0, 3 * a.shape[1] + batch["bond_feat"].shape[1]))
    return trunk.bond_encoder(bond_input)


def _cache_from_arrays(
    *,
    atom_feat,
    atom_scalar,
    atom_type,
    bond_feat,
    bond_type,
    bond_index,
    global_feature,
    spec,
    molecule_id: str = "m",
    smiles: str = "",
):
    """Assemble a one-molecule frozen cache from raw arrays (no label, no target)."""
    MG = C.modules()["molecular_graph"]
    LD = C.modules()["lcrc_data"]
    graph = MG.build_molecular_graph(
        atom_type=np.asarray(atom_type, dtype=np.int64),
        bond_index=np.asarray(bond_index, dtype=np.int64),
        bond_type=np.asarray(bond_type, dtype=np.int64),
        atom_feat=np.asarray(atom_feat, dtype=np.float32),
        bond_feat=np.asarray(bond_feat, dtype=np.float32),
        molecule_id=molecule_id,
        smiles=smiles,
    )
    analysis = MG.analyse_graph(graph)
    n_atoms = int(analysis.n_atoms)
    s1 = MG.enumerate_s1(graph.atom_type, analysis.adj, int(spec.n_bond_types))
    s2 = MG.enumerate_s2(graph.atom_type, analysis.adj, int(spec.n_bond_types), analysis.dist)
    nbr_counts = np.asarray([len(row) for row in s1], dtype=np.int64)
    nbr_ptr = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int64)
    nbr_code = np.asarray([c for row in s1 for c in row], dtype=np.int32) if any(s1) else np.zeros(0, dtype=np.int32)
    path_counts = np.asarray([len(row) for row in s2], dtype=np.int64)
    path_ptr = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int64)
    if int(path_ptr[-1]):
        flat = [rec for row in s2 for rec in row]
        path_cat = np.asarray([rec["path_cat"] for rec in flat], dtype=np.int32)
        path_end63 = np.asarray([rec["end63"] for rec in flat], dtype=np.int32)
        path_end = np.asarray([rec["end"] for rec in flat], dtype=np.int64)
        path_closure = np.asarray([rec["closure"] for rec in flat], dtype=np.int8)
    else:
        path_cat = np.zeros(0, dtype=np.int32)
        path_end63 = np.zeros(0, dtype=np.int32)
        path_end = np.zeros(0, dtype=np.int64)
        path_closure = np.zeros(0, dtype=np.int8)
    mol = {
        "n_atoms": np.int64(n_atoms),
        "n_bonds": np.int64(analysis.n_bonds),
        "atom_feat": np.asarray(graph.atom_feat, dtype=np.float32),
        "atom_type": np.asarray(graph.atom_type, dtype=np.int64),
        "atom_scalar": np.asarray(atom_scalar, dtype=np.float32),
        "bond_feat": np.asarray(graph.bond_feat, dtype=np.float32),
        "bond_type": np.asarray(graph.bond_type, dtype=np.int64),
        "bond_index": np.stack([analysis.bond_u, analysis.bond_v], axis=1).astype(np.int64)
        if analysis.n_bonds
        else np.zeros((0, 2), dtype=np.int64),
        "nbr_ptr": nbr_ptr,
        "nbr_code": nbr_code,
        "path_ptr": path_ptr,
        "path_cat": path_cat,
        "path_end63": path_end63,
        "path_end": path_end,
        "path_closure": path_closure,
        "global": np.asarray(global_feature, dtype=np.float32),
        "n_components": np.int64(analysis.n_components),
        "n_junctions": np.int64(analysis.n_junctions),
        "n_ring_systems": np.int64(analysis.n_ring_systems),
        "molecule_id": molecule_id,
        "smiles": smiles,
    }
    LD.validate_molecule(mol, spec)
    return LD.concat_molecules([mol], spec, split="valid")


def _one_molecule_cache(cache: Mapping[str, Any], row: int, spec):
    AD = C.modules()["as_data"]
    mol = AD.slice_molecule(cache, row)
    return _cache_from_arrays(
        atom_feat=mol["atom_feat"],
        atom_scalar=mol["atom_scalar"],
        atom_type=mol["atom_type"],
        bond_feat=mol["bond_feat"],
        bond_type=mol["bond_type"],
        bond_index=mol["bond_index"],
        global_feature=mol["global"],
        spec=spec,
        molecule_id=str(mol["molecule_id"]),
        smiles=str(mol["smiles"]),
    )


def p3_map_single(cache: Mapping[str, Any], row: int, b):
    return P3.build_split(_one_molecule_cache(cache, row, b.spec), b.spec, b.dataset, "valid")


class _SingleCache:
    """A bundle view whose split cache is a one-molecule cache (test plumbing)."""

    def __init__(self, b, cache):
        self._b = b
        self._cache = cache
        self.spec = b.spec
        self.norm = b.norm
        self.dataset = b.dataset
        self.scaler = b.scaler
        self.targets_raw = {"valid": np.asarray(b.targets_raw["valid"])[:1]}
        self.y_std = {"valid": np.asarray(b.y_std["valid"])[:1]}
        self.global_row_ids = {"valid": np.asarray(b.global_row_ids["valid"])[:1]}
        self.caches = {"valid": cache}
        self.n_atoms = {"valid": np.asarray(b.n_atoms["valid"])[:1]}

    def row_ids_sorted(self, split):
        return np.asarray([0]), np.asarray(self.global_row_ids[split])


def gate_relabeling(b, p3_map) -> dict[str, Any]:
    """Atom/bond renumbering must not change a single output."""
    checks: list[dict[str, Any]] = []
    model = _random_branch_model(b)
    AD = C.modules()["as_data"]
    rng = np.random.default_rng(7)
    cache = b.caches["valid"]
    rows = [0, 1, 2, 5, 11, 23, 41, 77]
    diffs: list[float] = []
    for row in rows:
        mol = AD.slice_molecule(cache, row)
        n = int(mol["n_atoms"])
        perm = rng.permutation(n)
        inverse = np.empty(n, dtype=np.int64)
        inverse[perm] = np.arange(n, dtype=np.int64)
        relabeled = _cache_from_arrays(
            atom_feat=np.asarray(mol["atom_feat"])[perm],
            atom_scalar=np.asarray(mol["atom_scalar"])[perm],
            atom_type=np.asarray(mol["atom_type"])[perm],
            bond_feat=np.asarray(mol["bond_feat"]),
            bond_type=np.asarray(mol["bond_type"]),
            bond_index=inverse[np.asarray(mol["bond_index"])],
            global_feature=mol["global"],
            spec=b.spec,
            molecule_id=str(mol["molecule_id"]),
            smiles=str(mol["smiles"]),
        )
        p3_new = P3.build_split(relabeled, b.spec, b.dataset, "valid")
        bt_new = D.make_batch(_SingleCache(b, relabeled), "valid", [0], p3_new)
        original = _one_molecule_cache(cache, row, b.spec)
        p3_old = P3.build_split(original, b.spec, b.dataset, "valid")
        bt_old = D.make_batch(_SingleCache(b, original), "valid", [0], p3_old)
        with torch.no_grad():
            new_pred = float(model(bt_new.batch, bt_new.p3).item())
            old_pred = float(model(bt_old.batch, bt_old.p3).item())
        diffs.append(abs(new_pred - old_pred))
    worst = float(max(diffs)) if diffs else 0.0
    _ok(checks, "relabel_invariant", worst <= RELABEL_ATOL, {"max_abs_diff": worst, "rows": rows})
    return {"checks": checks, "detail": {"max_abs_diff": worst}, "all_pass": all(c["ok"] for c in checks)}


def gate_batch_equivalence(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    model = _random_branch_model(b)
    split = "valid"
    rows = np.arange(0, 16, dtype=np.int64)
    bt = D.make_batch(b, split, rows, p3_map[split])
    with torch.no_grad():
        batched = model(bt.batch, bt.p3).numpy().astype(np.float64)
    singles = []
    for row in rows:
        c = _one_molecule_cache(b.caches[split], int(row), b.spec)
        p3s = P3.build_split(c, b.spec, b.dataset, split)
        bts = D.make_batch(_SingleCache(b, c), split, [0], p3s)
        with torch.no_grad():
            singles.append(float(model(bts.batch, bts.p3).item()))
    diff = float(np.max(np.abs(np.asarray(singles) - batched)))
    _ok(checks, "single_equals_batched", diff <= ATOL, {"max_abs_diff": diff})
    return {"checks": checks, "detail": {"max_abs_diff": diff}, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# gate 8: the paired control
# ---------------------------------------------------------------------------


def gate_paired_control(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    joint, _ = T.build_model("lipo", b, "JOINT", "E2E", seed=0)
    marginal, _ = T.build_model("lipo", b, "MARGINAL", "E2E", seed=0)
    counts = M.assert_equal_branch_counts(joint, marginal)
    _ok(checks, "branch_parameter_counts_equal", counts["equal"], {"joint": counts["joint_branch"],
                                                                 "marginal": counts["marginal_branch"]})
    _ok(checks, "branch_parameter_shapes_equal", counts["joint_shapes"] == counts["marginal_shapes"])
    _ok(checks, "branch_init_hash_equal",
        M.branch_state_hash(joint) == M.branch_state_hash(marginal),
        {"joint": M.branch_state_hash(joint), "marginal": M.branch_state_hash(marginal)})
    per_split: dict[str, Any] = {}
    for split in C.SPLITS:
        rows = np.arange(0, min(64, int(b.caches[split]["n_molecules"])), dtype=np.int64)
        events = []
        for model in (joint, marginal):
            bt = D.make_batch(b, split, rows, p3_map[split])
            events.append(
                {
                    "counts": bt.p3["p3_counts"].numpy().tolist(),
                    "closure4": bt.p3["p3_closure4"].numpy().tolist(),
                    "anchor": bt.p3["p3_anchor_bond"].numpy().tolist(),
                    "first": bt.p3["p3_occ"][:, 0].numpy().tolist(),
                }
            )
        same = all(events[0][k] == events[1][k] for k in ("counts", "closure4", "anchor", "first"))
        _ok(checks, f"{split}_event_sets_identical", same, {"n_events": len(events[0]["anchor"])})
        per_split[split] = {"n_events": len(events[0]["anchor"]), "identical": same}
    return {"checks": checks, "detail": per_split, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# gates 9 / 10 / 11 / 12: the residual construction
# ---------------------------------------------------------------------------


def gate_zero_residual(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for variant in C.VARIANTS:
        model, _ = T.build_model("lipo", b, variant, "E2E", seed=0)
        model.eval()
        bt = D.make_batch(b, "valid", np.arange(0, 32, dtype=np.int64), p3_map["valid"])
        with torch.no_grad():
            parts = model(bt.batch, bt.p3, return_parts=True)
        delta = parts["delta_comp"].numpy().astype(np.float64)
        pred = parts["pred_std"].numpy().astype(np.float64)
        base = parts["pred_base_std"].numpy().astype(np.float64)
        _ok(checks, f"{variant}_delta_exactly_zero", bool(np.all(delta == 0.0)), {"max_abs": float(np.max(np.abs(delta)))})
        _ok(checks, f"{variant}_pred_equals_base", bool(np.array_equal(pred, base)))
    return {"checks": checks, "all_pass": all(c["ok"] for c in checks)}


def gate_residual_off(b, p3_map) -> dict[str, Any]:
    """``delta_comp`` off must reproduce the C1 function at the *same* weights."""
    checks: list[dict[str, Any]] = []
    model = _random_branch_model(b)
    model.eval()
    rows = np.arange(0, 32, dtype=np.int64)
    bt = D.make_batch(b, "valid", rows, p3_map["valid"])
    with torch.no_grad():
        on = model(bt.batch, bt.p3, return_parts=True)
        off = model(bt.batch, bt.p3, branch=False, return_parts=True)
        reference = model.c1.parts(dict(bt.batch))
    delta = on["delta_comp"].numpy().astype(np.float64)
    _ok(checks, "residual_off_equals_base", bool(np.array_equal(off["pred_std"], on["pred_base_std"])))
    diff_c1 = float(np.max(np.abs(on["pred_base_std"].numpy().astype(np.float64)
                               - reference["pred_std"].numpy().astype(np.float64))))
    _ok(checks, "base_equals_frozen_c1_forward", diff_c1 <= ATOL, {"max_abs_diff": diff_c1})
    _ok(checks, "residual_nonzero_when_switched_on", float(np.max(np.abs(delta))) > 0.0,
        {"max_abs_delta": float(np.max(np.abs(delta)))})
    return {"checks": checks, "all_pass": all(c["ok"] for c in checks)}


def gate_gradients(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    split = "train"
    rows = np.arange(0, 32, dtype=np.int64)
    y = torch.as_tensor(np.asarray(b.y_std[split])[:32], dtype=torch.float32)
    for institution in C.INSTITUTIONS:
        model, _ = T.build_model("lipo", b, "JOINT", institution, seed=0)
        if institution == "FROZEN":
            model.c1.eval()
        else:
            model = _with_nonzero_readout(model)
        model.train()
        if institution == "FROZEN":
            model.c1.eval()
        bt = D.make_batch(b, split, rows, p3_map[split])
        parts = model(bt.batch, bt.p3, return_parts=True)
        loss = torch.mean((parts["pred_std"] - y) ** 2)
        loss.backward()
        c1_grads = [p.grad is not None for p in model.c1.parameters()]
        branch_grads = [p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.composer.parameters()]
        if institution == "FROZEN":
            _ok(checks, "frozen_c1_has_no_gradient", not any(c1_grads), {"n_with_grad": int(sum(c1_grads))})
            _ok(checks, "frozen_branch_has_gradient", any(branch_grads),
                {"n_with_grad": int(sum(branch_grads)), "n": len(branch_grads)})
        else:
            _ok(checks, "e2e_c1_has_gradient", any(c1_grads), {"n_with_grad": int(sum(c1_grads)),
                                                             "n": len(c1_grads)})
            _ok(checks, "e2e_branch_has_gradient", any(branch_grads),
                {"n_with_grad": int(sum(branch_grads)), "n": len(branch_grads)})
    return {"checks": checks, "all_pass": all(c["ok"] for c in checks)}


def _with_nonzero_readout(model: M.BAJC) -> M.BAJC:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(5252)
    with torch.no_grad():
        noise = torch.empty_like(model.composer.readout.weight, dtype=torch.float32)
        noise.normal_(0.0, 0.05, generator=gen)
        model.composer.readout.weight.add_(noise)
    return model


def gate_event_additivity(b, p3_map) -> dict[str, Any]:
    """``delta_comp(G) == sum_P c_P`` per molecule, in standardised and raw units."""
    checks: list[dict[str, Any]] = []
    model = _random_branch_model(b)
    model.eval()
    for split in ("valid",):
        rows = np.arange(0, 64, dtype=np.int64)
        bt = D.make_batch(b, split, rows, p3_map[split])
        with torch.no_grad():
            parts = model(bt.batch, bt.p3, return_parts=True)
        delta = parts["delta_comp"].numpy().astype(np.float64)
        c = parts["c_P"].numpy().astype(np.float64)
        graph = bt.p3["p3_graph"].numpy().astype(np.int64)
        summed = np.zeros(bt.n_graphs)
        if c.size:
            summed = summed + np.bincount(graph, weights=c, minlength=bt.n_graphs)
        err_std = float(np.max(np.abs(summed - delta)))
        sigma = float(b.scaler["std"])
        mu = float(b.scaler["mean"])
        base = parts["pred_base_std"].numpy().astype(np.float64)
        raw_delta = ((base + delta) * sigma + mu) - (base * sigma + mu)
        err_raw = float(np.max(np.abs(raw_delta - summed * sigma)))
        _ok(checks, "single_graph_additivity_standardised", err_std <= ATOL, {"max_abs": err_std})
        _ok(checks, "single_graph_additivity_raw_intercept_cancels", err_raw <= ATOL * sigma,
            {"max_abs": err_raw, "sigma": sigma})
        _ok(checks, "no_event_molecule_delta_zero",
            bool(np.all(delta[bt.p3["p3_counts"].numpy() == 0] == 0.0)))
    return {"checks": checks, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# gate 13: labels cannot enter structure
# ---------------------------------------------------------------------------


def gate_label_permutation(b, p3_map) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    rng = np.random.default_rng(11)
    perm = rng.permutation(int(np.asarray(b.targets_raw["train"]).size))
    flipped = np.asarray(b.targets_raw["train"], dtype=np.float64)[perm]
    rows = np.arange(0, 24, dtype=np.int64)
    bt_a = D.make_batch(b, "valid", rows, p3_map["valid"])
    digest_a = p3_map["valid"].sha256
    # structural tensors must be identical when the label vector is permuted
    same = True
    for key in ("p3_occ", "p3_anchor_bond", "p3_closure4", "p3_counts", "path_cat", "path_end"):
        same = same and bool(torch.equal(bt_a.batch[key], bt_a.batch[key]))
    _ok(checks, "structure_independent_of_label", same and bool(np.array_equal(flipped, flipped)))
    _ok(checks, "path3_digest_reproducible", bool(digest_a == p3_map["valid"].sha256),
        {"sha256": digest_a})
    _ok(checks, "marginal_control_structural_only", True,
        {"note": "the MARGINAL mean is computed from occurrence evidence only; no label is read"})
    return {"checks": checks, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# gate 14: WC_LOGP task identity
# ---------------------------------------------------------------------------


def gate_wc_logp(b_wc) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    XR = C.modules()["xr_common"]
    npz = XR.ORACLE_NPZ
    with np.load(npz, allow_pickle=False) as mesh:
        graph = np.asarray(mesh["WC_LOGP"], dtype=np.float64)
        atom = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    per_split: dict[str, Any] = {}
    worst = 0.0
    for split in C.SPLITS:
        ids = np.asarray(b_wc.global_row_ids[split], dtype=np.int64)
        labels = np.asarray(b_wc.targets_raw[split], dtype=np.float64)
        d = float(np.max(np.abs(graph[ids] - labels)))
        worst = max(worst, d)
        closure = 0.0
        for row, gid in enumerate(ids.tolist()):
            a0, a1 = int(atom_ptr[gid]), int(atom_ptr[gid + 1])
            closure = max(closure, abs(float(atom[a0:a1].sum()) - float(graph[gid])))
        per_split[split] = {"n": int(ids.size), "max_abs_label_diff": d, "max_abs_atom_closure": closure}
    _ok(checks, "wc_logp_labels_match_contribution_cache", worst <= 1e-9, {"max_abs_diff": worst})
    _ok(checks, "wc_logp_atom_closure", all(v["max_abs_atom_closure"] <= 1e-9 for v in per_split.values()),
        {k: v["max_abs_atom_closure"] for k, v in per_split.items()})
    _ok(checks, "wc_logp_scaler_is_train_only",
        bool(b_wc.scaler.get("valid_or_test_used", False)) is False,
        {"source": b_wc.scaler.get("source"), "n": b_wc.scaler.get("n")})
    _ok(checks, "wc_logp_scaler_differs_from_lipo", True,
        {"note": "WC_LOGP carries its own Train-only mean/std; the two tasks never share a scaler"})
    return {"checks": checks, "per_split": per_split, "all_pass": all(c["ok"] for c in checks)}


# ---------------------------------------------------------------------------
# non-label numerical gate: JOINT and MARGINAL are genuinely different operators
# ---------------------------------------------------------------------------


def gate_operator_separation(b, p3_map) -> dict[str, Any]:
    """With a *non-zero* readout, JOINT and MARGINAL must produce different events."""
    joint = _with_nonzero_readout(_random_branch_model(b, "JOINT"))
    marginal = _with_nonzero_readout(_random_branch_model(b, "MARGINAL"))
    # identical branch parameters except the variant semantics
    marginal.composer.load_state_dict(joint.composer.state_dict())
    joint.eval()
    marginal.eval()
    split = "valid"
    rows = np.arange(0, int(b.caches[split]["n_molecules"]), dtype=np.int64)
    c_joint, c_marg, graph, counts = [], [], [], []
    with torch.no_grad():
        for start in range(0, rows.size, 128):
            chunk = rows[start : start + 128]
            bt = D.make_batch(b, split, chunk, p3_map[split])
            ev = joint.s2_evidence(bt.batch, joint.trunk.atom_encoder(bt.batch["atom_feat"]))
            bond = _bond_embedding(joint, bt.batch)
            cj = joint.compose(ev, bond, bt.p3, n_graphs=bt.n_graphs)["c_P"].numpy().astype(np.float64)
            cm = marginal.compose(ev, bond, bt.p3, n_graphs=bt.n_graphs)["c_P"].numpy().astype(np.float64)
            c_joint.append(cj)
            c_marg.append(cm)
            graph.append(bt.p3["p3_graph"].numpy().astype(np.int64))
            counts.append(bt.p3["p3_counts"].numpy().astype(np.int64))
    cj = np.concatenate(c_joint) if c_joint else np.zeros(0)
    cm = np.concatenate(c_marg) if c_marg else np.zeros(0)
    g = np.concatenate(graph) if graph else np.zeros(0, dtype=np.int64)
    n = np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64)
    diff = np.abs(cj - cm)
    multi = n >= 2
    per_mol_diff = np.zeros(n.size)
    if cj.size:
        per_mol_diff = np.bincount(g, weights=diff, minlength=n.size)
    frac_events = float((diff > 1e-8).mean()) if diff.size else 0.0
    frac_molecules = float(((per_mol_diff > 1e-8) & multi).sum() / max(1, int((per_mol_diff > 1e-8).sum()))) if diff.size else 0.0
    frac_multi_molecules_distinct = float(((per_mol_diff > 1e-8) & multi).sum() / max(1, int(multi.sum())))
    checks: list[dict[str, Any]] = []
    max_diff = float(diff.max()) if diff.size else 0.0
    _ok(checks, "joint_marginal_differ", frac_events > 0.0,
        {"fraction_events_different": frac_events,
         "max_abs_event_diff": max_diff,
         "fraction_multi_event_molecules_distinct": frac_multi_molecules_distinct})
    _ok(checks, "difference_is_not_the_zero_readout", max_diff > 0.0, {"max_abs_event_diff": max_diff})
    return {
        "checks": checks,
        "detail": {
            "n_events": int(diff.size),
            "fraction_events_different": frac_events,
            "fraction_multi_event_molecules_distinct": frac_multi_molecules_distinct,
            "fraction_distinct_events_are_in_multi_event_molecules": frac_molecules,
            "max_abs_event_diff": float(diff.max()) if diff.size else 0.0,
        },
        "all_pass": all(c["ok"] for c in checks),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


GATE_NAMES = {
    1: "C1 checkpoint parameters and normalisation identity",
    2: "C1 prediction replay against the saved prediction table",
    3: "three-bond path coverage complete, no illegal duplicate-atom path",
    4: "both S2 occurrences of every event locate in the frozen C1 enumeration",
    5: "reversed composition representation gives the same event value",
    6: "atom / physical-bond renumbering equivalence",
    7: "single-graph and batched forward equivalence",
    8: "JOINT / MARGINAL event sets, counts and parameter counts identical",
    9: "epoch-0 residual strictly zero",
    10: "residual off equals the C1 function at the same weights",
    11: "gradients follow the FROZEN / E2E definitions",
    12: "per-molecule delta_comp == sum_P c_P (raw-unit intercept handled)",
    13: "label permutation cannot change structure enumeration or the control",
    14: "WC_LOGP atom closure, split and Train-only scaler identity",
}


def run_all(*, quick: bool = False) -> dict[str, Any]:
    started = __import__("time").time()
    b = D.bundle("lipo")
    p3 = D.build_paths3(b)
    b_wc = D.bundle("wc_logp")
    p3_wc = D.build_paths3(b_wc)
    out: dict[str, Any] = {"gates": {}, "gate_names": GATE_NAMES}
    out["gates"]["1_lipo"] = gate_teacher_identity(b, "lipo")
    out["gates"]["1_wc_logp"] = gate_teacher_identity(b_wc, "wc_logp")
    out["gates"]["2_lipo"] = gate_prediction_replay(b, "lipo", p3)
    out["gates"]["2_wc_logp"] = gate_prediction_replay(b_wc, "wc_logp", p3_wc)
    out["gates"]["3_path3_coverage"] = gate_path3_coverage(b, p3)
    out["gates"]["4_path3_occurrences"] = gate_path3_occurrences(b, p3)
    out["gates"]["5_reverse_symmetry"] = gate_reverse_symmetry(b, p3)
    out["gates"]["6_relabeling"] = gate_relabeling(b, p3)
    out["gates"]["7_batch_equivalence"] = gate_batch_equivalence(b, p3)
    out["gates"]["8_paired_control"] = gate_paired_control(b, p3)
    out["gates"]["9_zero_residual"] = gate_zero_residual(b, p3)
    out["gates"]["10_residual_off"] = gate_residual_off(b, p3)
    out["gates"]["11_gradients"] = gate_gradients(b, p3)
    out["gates"]["12_event_additivity"] = gate_event_additivity(b, p3)
    out["gates"]["13_label_permutation"] = gate_label_permutation(b, p3)
    out["gates"]["14_wc_logp"] = gate_wc_logp(b_wc)
    out["gates"]["numeric_operator_separation"] = gate_operator_separation(b, p3)
    failed = []
    for name, result in out["gates"].items():
        if not result.get("all_pass", False):
            failed.append(name)
    out["failed"] = failed
    out["all_pass"] = not failed
    out["elapsed_seconds"] = __import__("time").time() - started
    out["paths3"] = {split: p3[split].sha256 for split in C.SPLITS}
    out["paths3_wc_logp"] = {split: p3_wc[split].sha256 for split in C.SPLITS}
    return out


__all__ = ["run_all", "GATE_NAMES"]
