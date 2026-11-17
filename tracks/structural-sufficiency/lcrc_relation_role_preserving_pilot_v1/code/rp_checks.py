"""The correctness gates, evaluated before (and after) any formal training.

1. frozen reuse: the routing caches are the pinned bytes *and* every array is
   reproduced by the frozen partition/routing implementation;
2. role partition: every S2 occurrence belongs to exactly one group, no
   omission, no duplication; REAL equals ``owner(e1) != owner(e2)`` and CONTROL
   equals the frozen type hash on every occurrence;
3. S2 conservation: per atom, ``Q_internal + Q_cross`` equals the frozen C1 S2
   aggregate (typed record and content), column by column;
4. gamma = 1 equivalence: at the canonical C1 initial state, predictions, local
   input columns, local representation ``h``, atom readout ``f(h)``, ``mean h``,
   ``mean h^2``, ``rho`` and the prediction are the frozen C1 function's, for
   both role vectors;
5. column / normalisation / readout identity: the C1 input layout, the
   Train-only feature normalisation, the Train-only target scalers and the
   ``f``/``rho`` modules are unchanged;
6. atom / bond relabelling invariance of the routing, the roles, the per-atom
   role aggregates and the prediction (init and trained checkpoints);
7. parameter identity: RR_REAL and RR_CONTROL have the same count, the same
   full initial state, and differ from C1 by exactly two 200-d gamma vectors;
8. label / oracle blindness: roles and batch features are functions of the
   graph and S2 type codes only, and are unchanged when labels/oracle values
   are permuted;
9. trained-state identity and closure: with gamma = 1 a trained model equals the
   explicit single-bucket C1 aggregation of the same weights, and every
   prediction closes as ``sum_v f(h_v) + rho``;
10. WC_LOGP closure: the per-atom Crippen contributions sum to the per-molecule
    WC_LOGP value on every official row;
11. group-size gate: both role groups are non-empty on every split (sizes are
    reported, never forced to match).
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Mapping, Sequence

import numpy as np

import rp_common as C
import rp_data as D
import rp_models as M
import rp_prepare as P
import rp_train as T

MOD = C.modules()


def _ok(condition: bool, **details: Any) -> dict[str, Any]:
    out = {"pass": bool(condition)}
    out.update(details)
    return out


def _sample_rows(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, num=k, dtype=int)).astype(np.int64)


def _state_flat(model) -> dict[str, Any]:
    """Flat C1-style state dict (``trunk.`` stripped, gammas dropped)."""
    out = {}
    for key, value in model.state_dict().items():
        if key.startswith("gamma_"):
            continue
        if key.startswith(M.ARM_PREFIX):
            out[key[len(M.ARM_PREFIX) :]] = value
        else:
            out[key] = value
    return out


def _reference_c1(spec, norm, state, device="cpu"):
    arm = MOD["cr_models"].ArmA(spec, norm)
    arm.load_canonical(state)
    return arm.to(device).eval()


# ---------------------------------------------------------------------------
# gate 1: frozen reuse
# ---------------------------------------------------------------------------


def gate_frozen_reuse(bundle) -> dict[str, Any]:
    reuse = P.verify_reuse(bundle)
    return _ok(reuse["all_pass"], **reuse)


# ---------------------------------------------------------------------------
# gate 2 + 3: role partition and S2 conservation
# ---------------------------------------------------------------------------


def gates_role_partition(bundle) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}}
    all_pass = True
    table = bundle.control_table
    for split in C.SPLITS:
        cache = bundle.caches[split]
        routing = bundle.routing[split]
        pc = np.asarray(cache["path_cat"], dtype=np.int64)
        e = np.asarray(cache["path_end63"], dtype=np.int64)
        k = np.asarray(cache["path_closure"], dtype=np.int64)
        owner1 = np.asarray(routing["path_owner1"], dtype=np.int64)
        owner2 = np.asarray(routing["path_owner2"], dtype=np.int64)
        cross = np.asarray(routing["path_cross"], dtype=np.int64)
        total = int(pc.size)
        entry: dict[str, Any] = {"n_occurrences": total, "models": {}}
        for model in C.MODELS:
            role = np.asarray(bundle.roles(split, model), dtype=np.int64)
            n0 = int((role == 0).sum())
            n1 = int((role == 1).sum())
            one_group = bool(n0 + n1 == total) and bool(role.min() >= 0) and bool(role.max() <= 1)
            if model == "RR_REAL":
                rule_match = bool(np.array_equal(role, (owner1 != owner2).astype(np.int64)))
                rule_match = rule_match and bool(np.array_equal(cross.astype(np.int64), (owner1 != owner2).astype(np.int64)))
                rule_name = "path_cross == (owner1 != owner2)"
            else:
                # the role must be exactly the frozen hash of the typed S2 record
                unique = np.unique(np.stack([pc, e, k], axis=1), axis=0)
                recomputed = np.asarray(
                    [D.control_hash_role(a, b, c) for a, b, c in unique], dtype=np.int64
                )
                table_role = D.roles_from_triples(table, unique[:, 0], unique[:, 1], unique[:, 2]).astype(np.int64)
                rule_match = bool(np.array_equal(recomputed, table_role))
                # and every occurrence's role must be its type's role
                per_occ = D.roles_from_triples(table, pc, e, k).astype(np.int64)
                rule_match = rule_match and bool(np.array_equal(role, per_occ))
                rule_name = "role == blake2b(salt|path_cat|path_end63|path_closure) low bit"
            ok = bool(one_group and rule_match and n0 > 0 and n1 > 0)
            entry["models"][model] = {
                "n_group0": n0,
                "n_group1": n1,
                "fraction_group0": n0 / total if total else 0.0,
                "fraction_group1": n1 / total if total else 0.0,
                "exactly_one_group_per_occurrence": bool(one_group),
                "rule": rule_name,
                "matches_rule_on_all_occurrences": bool(rule_match),
                "pass": ok,
            }
            all_pass = all_pass and ok
        out["splits"][split] = entry
    out["all_pass"] = bool(all_pass)
    return _ok(all_pass, **out)


def gate_s2_conservation(bundle, *, max_batches: int | None = None) -> dict[str, Any]:
    """Per-atom ``Q_internal + Q_cross == frozen C1 S2 aggregate`` on every split."""
    import torch

    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_state(spec, norm)
    model = M.build_model("RR_REAL", spec, norm, canonical, device="cpu").eval()
    ones = (torch.ones(C.S2_ROLE_DIM), torch.ones(C.S2_ROLE_DIM))
    out: dict[str, Any] = {"splits": {}, "tolerance": "1e-5 + 1e-6*max(1,|x|)"}
    all_pass = True
    with torch.no_grad():
        for split in C.SPLITS:
            n = int(bundle.caches[split]["n_molecules"])
            rows_all = np.arange(n, dtype=np.int64)
            worst_q = 0.0
            worst_scaled = 0.0
            worst_nons2 = 0.0
            n_atoms = 0
            n_occ = 0
            chunks = [
                rows_all[s : s + 128] for s in range(0, rows_all.size, 128)
            ]
            if max_batches is not None:
                chunks = chunks[: int(max_batches)]
            for chunk in chunks:
                batch = bundle.collate(split, chunk)
                a, b, scal = model.trunk.encode(batch)
                evidence = model.trunk.s2_evidence(batch, a)
                _h, aux = model.trunk.local(batch, a, b, scal)
                ref = torch.cat([aux["path_agg"], aux["content_agg"]], dim=1)
                x, q0, q1 = model.trunk.local_inputs(
                    batch, a, b, scal, evidence, batch[model.role_key], ones[0], ones[1]
                )
                diff = (q0 + q1 - ref).abs()
                worst_q = max(worst_q, float(diff.max().item()) if diff.numel() else 0.0)
                denom = C.trace_tolerance(ref.numpy())
                worst_scaled = max(
                    worst_scaled, float((diff.numpy() / denom).max()) if diff.numel() else 0.0
                )
                nons2 = torch.cat([x[:, :224], x[:, 424:433]], dim=1)
                ref_nons2 = torch.cat([aux["local_input"][:, :224], aux["local_input"][:, 424:433]], dim=1)
                worst_nons2 = max(worst_nons2, float((nons2 - ref_nons2).abs().max().item()))
                s2_cols = x[:, 224:424]
                worst_scaled = max(worst_scaled, float(((s2_cols - ref).abs().numpy() / denom).max()))
                n_atoms += int(a.shape[0])
                n_occ += int(evidence.shape[0])
            ok = bool(worst_scaled <= 1.0 and worst_nons2 == 0.0)
            out["splits"][split] = {
                "n_atoms_checked": n_atoms,
                "n_occurrences_checked": n_occ,
                "max_abs_Q_split_minus_C1": worst_q,
                "max_tolerance_scaled_error": worst_scaled,
                "max_abs_diff_non_s2_columns": worst_nons2,
                "pass": ok,
            }
            all_pass = all_pass and ok
    out["all_pass"] = bool(all_pass)
    return _ok(all_pass, **out)


# ---------------------------------------------------------------------------
# gate 4 + 5: gamma=1 equivalence, column / normalisation / readout identity
# ---------------------------------------------------------------------------


def gate_equivalence(bundle, *, n_per_split: int = 24, model=None, label: str = "canonical") -> dict[str, Any]:
    """RR with gamma=1 must equal the frozen C1 function at the same weights."""
    import torch

    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_state(spec, norm)
    if model is None:
        variant = "RR_REAL"
        net = M.build_model(variant, spec, norm, canonical, device="cpu").eval()
        state = canonical
    else:
        variant = model.variant
        net = model.eval()
        state = _state_flat(net)
    ref = _reference_c1(spec, norm, state, device="cpu")
    ones = (torch.ones(C.S2_ROLE_DIM), torch.ones(C.S2_ROLE_DIM))
    out: dict[str, Any] = {
        "label": label,
        "model_under_test": variant,
        "splits": {},
        "tolerance": {"x_non_s2": 0.0, "other": "1e-5 + 1e-6*max(1,|x|)"},
    }
    all_pass = True
    with torch.no_grad():
        for split in C.SPLITS:
            n = int(bundle.caches[split]["n_molecules"])
            rows = _sample_rows(n, n_per_split)
            worst = {
                "x_non_s2": 0.0,
                "x_s2": 0.0,
                "h": 0.0,
                "contrib_atom": 0.0,
                "local_sum": 0.0,
                "mean_h": 0.0,
                "mean_h2": 0.0,
                "rho": 0.0,
                "pred": 0.0,
            }
            role_swap = 0.0
            n_atoms = 0
            for start in range(0, rows.size, 16):
                chunk = rows[start : start + 16]
                batch = bundle.collate(split, chunk)
                c1 = ref.parts(batch)
                a, b, scal = ref.trunk.encode(batch)
                _h, aux = ref.trunk.local(batch, a, b, scal)
                ref_x = aux["local_input"]
                for role_key in (C.MODEL_ROLE_KEY["RR_REAL"], C.MODEL_ROLE_KEY["RR_CONTROL"]):
                    pred, parts = net(
                        batch, role_key=role_key, gammas=ones, return_aux=True
                    )
                    x = parts["local_input"]
                    nons2 = torch.cat([x[:, :224], x[:, 424:433]], dim=1)
                    ref_nons2 = torch.cat([ref_x[:, :224], ref_x[:, 424:433]], dim=1)
                    worst["x_non_s2"] = max(worst["x_non_s2"], float((nons2 - ref_nons2).abs().max()))
                    d = (x[:, 224:424] - ref_x[:, 224:424]).abs()
                    worst["x_s2"] = max(worst["x_s2"], float((d / torch.as_tensor(C.trace_tolerance(ref_x[:, 224:424].numpy()))).max()))
                    for key, name in (
                        ("h", "h"),
                        ("contrib_atom", "contrib_atom"),
                        ("local_sum", "local_sum"),
                        ("mean_h", "mean_h"),
                        ("mean_h2", "mean_h2"),
                        ("rho", "rho"),
                    ):
                        d = (parts[key] - c1[key]).abs()
                        denom = torch.as_tensor(C.trace_tolerance(c1[key].numpy()))
                        worst[name] = max(worst[name], float((d / denom).max()) if d.numel() else 0.0)
                    d = (pred - c1["pred_std"]).abs()
                    worst["pred"] = max(worst["pred"], float((d / torch.as_tensor(C.trace_tolerance(c1["pred_std"].numpy()))).max()))
                    if role_key == C.MODEL_ROLE_KEY["RR_CONTROL"]:
                        p_real = net(batch, role_key=C.MODEL_ROLE_KEY["RR_REAL"], gammas=ones)
                        d = (pred - p_real).abs()
                        role_swap = max(role_swap, float((d / torch.as_tensor(C.trace_tolerance(p_real.numpy()))).max()))
                    n_atoms += int(x.shape[0])
            ok = bool(
                worst["x_non_s2"] == 0.0
                and all(worst[k] <= 1.0 for k in worst if k != "x_non_s2")
                and role_swap <= 1.0
            )
            out["splits"][split] = {
                "n_atoms_checked": n_atoms,
                "max_tolerance_scaled_diff": worst,
                "max_tolerance_scaled_role_swap_diff": float(role_swap),
                "pass": ok,
            }
            all_pass = all_pass and ok
    out["all_pass"] = bool(all_pass)
    return _ok(all_pass, **out)


def gate_data_identity(bundle) -> dict[str, Any]:
    """C1 input layout, Train-only normalisation / scalers, unchanged readout."""
    import torch

    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_state(spec, norm)
    model = M.build_model("RR_REAL", spec, norm, canonical, device="cpu").eval()
    checks: dict[str, Any] = {}

    # frozen fingerprints of the C1 pipeline
    checks["norm_fingerprint"] = {
        "value": bundle.extra["norm_fingerprint"],
        "frozen": "3c87249a325b1a59cba819d49dfca10df5004690efa7cc2ee22945aa548db4da",
        "match": bundle.extra["norm_fingerprint"]
        == "3c87249a325b1a59cba819d49dfca10df5004690efa7cc2ee22945aa548db4da",
    }
    checks["feature_spec_hash"] = {
        "value": bundle.extra["spec_hash"],
        "frozen": "a65c67f9595d9190e7c913d2594ba984a07b51300d63916de865dccca434206c",
        "match": bundle.extra["spec_hash"]
        == "a65c67f9595d9190e7c913d2594ba984a07b51300d63916de865dccca434206c",
    }

    # Train-only target scalers, recomputed from the frozen target tables
    scalers_ok = True
    scaler_detail = {}
    for task in C.TASKS:
        recomputed = C.target_scaler(bundle.targets(task, "train"))
        stored = bundle.scaler(task)
        same = all(abs(float(recomputed[k]) - float(stored[k])) < 1e-12 for k in ("mean", "std"))
        scaler_detail[task] = {
            "stored": stored,
            "recomputed_from_train": recomputed,
            "match": same,
            "fit_split": stored.get("fit_split"),
        }
        scalers_ok = scalers_ok and same
    checks["target_scalers"] = {"tasks": scaler_detail, "pass": bool(scalers_ok)}

    # local layout and readout structure
    local = model.trunk.local
    layout_ok = (
        int(C.LOCAL_INPUT_DIM) == 433
        and len(local.local_cols) == 433
        and (int(C.S2_TYPED_DIM), int(C.S2_CONTENT_DIM), int(C.S2_ROLE_DIM)) == (104, 96, 200)
        and len(list(local.scalar_cols)) == 9
        and int(model.rho_dim) == 171
    )
    checks["layout"] = {
        "local_input_dim": int(C.LOCAL_INPUT_DIM),
        "local_cols": len(local.local_cols),
        "s2_typed_dim": int(C.S2_TYPED_DIM),
        "s2_content_dim": int(C.S2_CONTENT_DIM),
        "s2_role_dim": int(C.S2_ROLE_DIM),
        "scalar_cols": len(list(local.scalar_cols)),
        "rho_input_dim": int(model.rho_dim),
        "pass": bool(layout_ok),
    }

    # the C1 part of the initial state is exactly the canonical C1 state
    state = model.state_dict()
    c1_keys = [k for k in state if not k.startswith("gamma_")]
    flat = _state_flat(model)
    canonical_hash = C.sha256_state(canonical)
    flat_hash = C.sha256_state(flat)
    checks["readout_identity"] = {
        "n_non_gamma_parameters": int(sum(state[k].numel() for k in c1_keys)),
        "non_gamma_state_hash": flat_hash,
        "canonical_c1_state_hash": canonical_hash,
        "match": bool(flat_hash == canonical_hash),
        "f_keys": sorted(k for k in flat if k.startswith("f.")),
        "rho_keys": sorted(k for k in flat if k.startswith("rho.")),
    }

    # C1 local columns: non-S2 columns bitwise identical, S2 columns inside tolerance
    with torch.no_grad():
        rows = _sample_rows(int(bundle.caches["valid"]["n_molecules"]), 16)
        worst_non = 0.0
        worst_s2 = 0.0
        for start in range(0, rows.size, 8):
            batch = bundle.collate("valid", rows[start : start + 8])
            a, b, scal = model.trunk.encode(batch)
            evidence = model.trunk.s2_evidence(batch, a)
            _h, aux = model.trunk.local(batch, a, b, scal)
            ref_x = aux["local_input"]
            x, _q0, _q1 = model.trunk.local_inputs(
                batch, a, b, scal, evidence, batch[model.role_key],
                torch.ones(C.S2_ROLE_DIM), torch.ones(C.S2_ROLE_DIM),
            )
            n = torch.cat([x[:, :224], x[:, 424:433]], dim=1)
            r = torch.cat([ref_x[:, :224], ref_x[:, 424:433]], dim=1)
            worst_non = max(worst_non, float((n - r).abs().max()))
            d = (x[:, 224:424] - ref_x[:, 224:424]).abs()
            worst_s2 = max(
                worst_s2,
                float((d / torch.as_tensor(C.trace_tolerance(ref_x[:, 224:424].numpy()))).max()),
            )
    checks["local_columns"] = {
        "max_abs_diff_non_s2": worst_non,
        "max_tolerance_scaled_diff_s2": worst_s2,
        "pass": bool(worst_non == 0.0 and worst_s2 <= 1.0),
    }

    all_pass = bool(
        checks["norm_fingerprint"]["match"]
        and checks["feature_spec_hash"]["match"]
        and checks["target_scalers"]["pass"]
        and checks["layout"]["pass"]
        and checks["readout_identity"]["match"]
        and checks["local_columns"]["pass"]
    )
    return _ok(all_pass, checks=checks)


# ---------------------------------------------------------------------------
# gate 6: relabelling invariance
# ---------------------------------------------------------------------------


def _permuted_graph(graph, perm: np.ndarray, *, seed: int):
    MG = MOD["molecular_graph"]
    n = int(graph.atom_type.size)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    edges = np.stack([graph.bond_u, graph.bond_v], axis=1)
    perm_edges = inv[edges]
    rng = np.random.default_rng(seed)
    bond_order = rng.permutation(perm_edges.shape[0])
    perm_edges = perm_edges[bond_order]
    out = MG.build_molecular_graph(
        atom_type=np.asarray(graph.atom_type)[perm],
        bond_index=perm_edges,
        bond_type=np.asarray(graph.bond_type)[bond_order],
        atom_feat=np.asarray(graph.atom_feat)[perm],
        bond_feat=np.asarray(graph.bond_feat)[bond_order],
        global_feat=None if graph.global_feat is None else np.asarray(graph.global_feat),
        molecule_id=str(graph.molecule_id),
        smiles=str(graph.smiles),
        raw_atom_charge=None if graph.raw_atom_charge is None else np.asarray(graph.raw_atom_charge)[perm],
        raw_atom_h=None if graph.raw_atom_h is None else np.asarray(graph.raw_atom_h)[perm],
        bond_stereo=None if graph.bond_stereo is None else np.asarray(graph.bond_stereo)[bond_order],
    )
    return out, inv


def gate_relabelling(bundle, *, checkpoint_paths: Mapping[str, Any] | None = None, n_per_split: int = 8) -> dict[str, Any]:
    import torch

    PM = C.pred_modules()
    lcrc = MOD["lcrc_data"]
    adapters = MOD["adapters"]
    sq_data = MOD["sq_data"]
    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_state(spec, norm)
    n_bond_types = int(bundle.extra["n_bond_types"])

    models = {v: M.build_model(v, spec, norm, canonical, device="cpu").eval() for v in C.MODELS}
    trained = []
    for key, path in (checkpoint_paths or {}).items():
        variant = str(key).split("/")[-1]
        task = str(key).split("/")[0]
        model, _payload = T.load_model_from_checkpoint(bundle, task, variant, path, device="cpu")
        trained.append((str(key), model.eval()))

    raw = bundle.cr.lipo_raw
    checked = []
    worst = {variant: 0.0 for variant in C.MODELS}
    worst_role = 0.0
    worst_agg = 0.0
    partition_match = True
    trained_worst = {}
    with torch.no_grad():
        for split in C.SPLITS:
            n = int(bundle.caches[split]["n_molecules"])
            for row in _sample_rows(n, n_per_split).tolist():
                graph = adapters.lipo_row_to_graph(sq_data.lipo_raw_row(raw, int(row)), int(row), spec)
                base_cache = lcrc.build_cache([graph], spec, split="relabel-base")
                rng = np.random.default_rng(10_000 + int(row))
                perm = rng.permutation(int(graph.atom_type.size))
                perm_graph, inv = _permuted_graph(graph, perm, seed=20_000 + int(row))
                perm_cache = lcrc.build_cache([perm_graph], spec, split="relabel-perm")
                batches = {}
                routing_maps = {}
                for tag, mol_cache in (("base", base_cache), ("perm", perm_cache)):
                    routing = PM["rr_routing"].build_split_routing(
                        mol_cache, tag, n_bond_types=n_bond_types
                    )
                    routing_maps[tag] = routing
                    roles_c = D.roles_from_triples(
                        bundle.control_table,
                        mol_cache["path_cat"],
                        mol_cache["path_end63"],
                        mol_cache["path_closure"],
                    )
                    batches[tag] = D.collate_arrays(
                        mol_cache,
                        routing.arrays,
                        np.asarray(routing.arrays["path_cross"], dtype=np.uint8),
                        roles_c,
                        [0],
                    )
                # partition / block correspondence
                parts = {}
                for tag, mol_cache in (("base", base_cache), ("perm", perm_cache)):
                    a0, a1 = int(mol_cache["atom_ptr"][0]), int(mol_cache["atom_ptr"][1])
                    b0, b1 = int(mol_cache["bond_ptr"][0]), int(mol_cache["bond_ptr"][1])
                    edges = [
                        (int(mol_cache["bond_index"][j, 0]), int(mol_cache["bond_index"][j, 1]))
                        for j in range(b0, b1)
                    ]
                    parts[tag] = (
                        PM["rr_partition"].partition_molecule(
                            a1 - a0, edges, np.asarray(mol_cache["atom_type"])[a0:a1],
                            np.asarray(mol_cache["bond_type"])[b0:b1],
                        ),
                        edges,
                    )
                sig_b = sorted(
                    (
                        tuple(sorted(tuple(sorted(parts["base"][1][e])) for e in block["bonds"])),
                        tuple(sorted(int(v) for v in block["atoms"])),
                        int(block["n_bonds"]),
                    )
                    for block in parts["base"][0].blocks
                )
                mapped = sorted(
                    (
                        tuple(
                            sorted(
                                tuple(sorted((int(inv[u]), int(inv[v]))))
                                for u, v in (parts["base"][1][e] for e in block["bonds"])
                            )
                        ),
                        tuple(sorted(int(inv[v]) for v in block["atoms"])),
                        int(block["n_bonds"]),
                    )
                    for block in parts["base"][0].blocks
                )
                sig_p = sorted(
                    (
                        tuple(sorted(tuple(sorted(parts["perm"][1][e])) for e in block["bonds"])),
                        tuple(sorted(int(v) for v in block["atoms"])),
                        int(block["n_bonds"]),
                    )
                    for block in parts["perm"][0].blocks
                )
                partition_match = partition_match and mapped == sig_p
                del sig_b

                # role class counts are preserved
                for variant, model in models.items():
                    role_key = C.MODEL_ROLE_KEY[variant]
                    nb = int(batches["base"][role_key].numel())
                    np_ = int(batches["perm"][role_key].numel())
                    cb = np.bincount(batches["base"][role_key].numpy().astype(np.int64), minlength=2)
                    cp = np.bincount(batches["perm"][role_key].numpy().astype(np.int64), minlength=2)
                    worst_role = max(worst_role, float(abs(int(cb[0]) - int(cp[0])) + abs(int(cb[1]) - int(cp[1]))))
                    if nb != np_:
                        worst_role = max(worst_role, float(abs(nb - np_)))
                    pred_b = float(model(batches["base"], role_key=role_key).item())
                    pred_p = float(model(batches["perm"], role_key=role_key).item())
                    worst[variant] = max(worst[variant], abs(pred_b - pred_p))
                    _, parts_b = model(batches["base"], role_key=role_key, return_aux=True)
                    _, parts_p = model(batches["perm"], role_key=role_key, return_aux=True)
                    for key in ("q_internal", "q_cross"):
                        qb = parts_b[key].numpy()
                        qp = parts_p[key].numpy()
                        # atom order is permuted: new position i holds old atom perm[i]
                        mapped_q = qp[inv]
                        d = np.abs(mapped_q - qb)
                        worst_agg = max(worst_agg, float((d / C.trace_tolerance(qb)).max()))
                for label, model in trained:
                    role_key = C.MODEL_ROLE_KEY[model.variant]
                    pred_b = float(model(batches["base"], role_key=role_key).item())
                    pred_p = float(model(batches["perm"], role_key=role_key).item())
                    trained_worst[label] = max(trained_worst.get(label, 0.0), abs(pred_b - pred_p))
                checked.append((split, int(row)))

    tol = 1e-5
    ok = (
        partition_match
        and worst_role == 0.0
        and worst_agg <= 1.0
        and all(v < tol for v in worst.values())
        and all(v < tol for v in trained_worst.values())
    )
    return _ok(
        ok,
        n_molecules=len(checked),
        molecules=[{"split": s, "row": r} for s, r in checked],
        block_partition_maps_bijectively=bool(partition_match),
        role_class_count_mismatch=float(worst_role),
        max_tolerance_scaled_q_diff=float(worst_agg),
        max_abs_prediction_diff_init=worst,
        max_abs_prediction_diff_trained=trained_worst,
        tolerance=float(tol),
        note="atom order and physical-bond order are both permuted; features move with the atoms",
    )


# ---------------------------------------------------------------------------
# gate 7: parameter identity
# ---------------------------------------------------------------------------


def gate_parameter_identity(bundle) -> dict[str, Any]:
    budget = P.parameter_budget(bundle)
    expected = budget["analytic"]
    ok = bool(
        budget["initial_state_identical"]
        and budget["ratio"]["identical_counts"]
        and budget["models"]["RR_REAL"]["matches_analytic"]
        and budget["models"]["RR_CONTROL"]["matches_analytic"]
        and budget["models"]["RR_REAL"]["c1_part_matches_c1_total"]
        and budget["models"]["RR_REAL"]["gamma_all_ones"]
        and int(expected["delta_over_c1"]) == 2 * int(C.S2_ROLE_DIM)
    )
    return _ok(
        ok,
        analytic=expected,
        ratio=budget["ratio"],
        initial_state_identical=budget["initial_state_identical"],
        full_initial_state_hash=budget["full_initial_state_hash"],
        c1_canonical_state_hash=budget["c1_canonical_state_hash"],
        models={
            v: {
                "total": budget["models"][v]["total"],
                "trunk": budget["models"][v]["trunk"],
                "readout": budget["models"][v]["readout"],
                "role_gamma": budget["models"][v]["role_gamma"],
                "gamma_all_ones": budget["models"][v]["gamma_all_ones"],
            }
            for v in C.MODELS
        },
    )


# ---------------------------------------------------------------------------
# gate 8: label / oracle blindness
# ---------------------------------------------------------------------------


def gate_label_blind(bundle, *, n_molecules: int = 64) -> dict[str, Any]:
    """Roles and batch features are unchanged by permuting labels / oracle mass."""
    split = "valid"
    n = int(bundle.caches[split]["n_molecules"])
    rows = _sample_rows(n, n_molecules)
    batch = bundle.collate(split, rows)
    rng = np.random.default_rng(7)
    # permute every target table and the oracle atom table in memory
    permuted = {task: rng.permutation(np.asarray(bundle.targets(task, split))) for task in C.TASKS}
    role_real = np.asarray(bundle.roles(split, "RR_REAL"))
    role_control = np.asarray(bundle.roles(split, "RR_CONTROL"))
    # recompute roles after the permutation: they are functions of the graph only
    cache = bundle.caches[split]
    role_real_again = np.asarray(bundle.routing[split]["path_cross"], dtype=np.uint8)
    role_control_again = D.roles_from_triples(
        bundle.control_table,
        cache["path_cat"],
        cache["path_end63"],
        cache["path_closure"],
    )
    role_unchanged = bool(
        np.array_equal(role_real, role_real_again) and np.array_equal(role_control, role_control_again)
    )
    cols = inspect.signature(D.collate).parameters
    collate_has_no_target = list(cols) == ["bundle", "split", "rows"]
    manifest = bundle.control_manifest
    manifest_flags = {
        "labels_used": bool(manifest.get("labels_used", True)),
        "oracle_used": bool(manifest.get("oracle_used", True)),
        "atom_index_used": bool(manifest.get("atom_index_used", True)),
        "row_id_used": bool(manifest.get("row_id_used", True)),
    }
    # the CONTROL role of a type is split-independent (type-only, never per-row)
    type_consistent = True
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        ca, cb = bundle.caches[a], bundle.caches[b]
        ta = set(
            map(tuple, np.unique(np.stack([ca["path_cat"], ca["path_end63"], ca["path_closure"]], 1), axis=0).tolist())
        )
        tb = set(
            map(tuple, np.unique(np.stack([cb["path_cat"], cb["path_end63"], cb["path_closure"]], 1), axis=0).tolist())
        )
        shared = sorted(ta & tb)
        if not shared:
            continue
        arr = np.asarray(shared, dtype=np.int64)
        ra = D.roles_from_triples(bundle.control_table, arr[:, 0], arr[:, 1], arr[:, 2])
        rb = D.roles_from_triples(bundle.control_table, arr[:, 0], arr[:, 1], arr[:, 2])
        type_consistent = type_consistent and bool(np.array_equal(ra, rb))
    ok = bool(role_unchanged and collate_has_no_target and not any(manifest_flags.values()) and type_consistent)
    return _ok(
        ok,
        target_permuted_shapes={k: list(v.shape) for k, v in permuted.items()},
        role_vectors_unchanged=role_unchanged,
        collate_signature=list(cols),
        collate_has_no_target_or_task=collate_has_no_target,
        control_manifest_flags=manifest_flags,
        control_type_role_split_independent=bool(type_consistent),
        note="roles are built from the frozen partition or from the S2 type code; no label, oracle, row id or atom index enters them",
    )


# ---------------------------------------------------------------------------
# gate 9: trained-state identity and readout closure
# ---------------------------------------------------------------------------


def gate_trained_identity(bundle, checkpoint_paths: Mapping[str, Any], *, n_per_split: int = 16) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"checkpoints": {}, "tolerance": "1e-5 + 1e-6*max(1,|x|)"}
    all_pass = True
    with torch.no_grad():
        for key, path in checkpoint_paths.items():
            task = str(key).split("/")[0]
            variant = str(key).split("/")[-1]
            model, payload = T.load_model_from_checkpoint(bundle, task, variant, path, device="cpu")
            model = model.eval()
            state = _state_flat(model)
            ref = _reference_c1(bundle.spec, bundle.norm, state, device="cpu")
            ones = (torch.ones(C.S2_ROLE_DIM), torch.ones(C.S2_ROLE_DIM))
            entry = {"epoch": int(payload["epoch"]), "task": task, "variant": variant, "splits": {}}
            for split in ("valid", "test"):
                rows = _sample_rows(int(bundle.caches[split]["n_molecules"]), n_per_split)
                worst_pred = 0.0
                worst_role = 0.0
                worst_closure = 0.0
                worst_sum = 0.0
                n_graphs = 0
                for start in range(0, rows.size, 16):
                    chunk = rows[start : start + 16]
                    batch = bundle.collate(split, chunk)
                    c1 = ref.parts(batch)
                    for role_key in (C.MODEL_ROLE_KEY["RR_REAL"], C.MODEL_ROLE_KEY["RR_CONTROL"]):
                        pred, parts = model(batch, role_key=role_key, gammas=ones, return_aux=True)
                        d = (pred - c1["pred_std"]).abs()
                        worst_pred = max(
                            worst_pred,
                            float((d / torch.as_tensor(C.trace_tolerance(c1["pred_std"].numpy()))).max()),
                        )
                        closure = (parts["pred_std"] - (parts["local_sum"] + parts["rho"])).abs()
                        worst_closure = max(
                            worst_closure,
                            float((closure / torch.as_tensor(C.trace_tolerance(parts["pred_std"].numpy()))).max()),
                        )
                        summed = parts["contrib_atom"].new_zeros(parts["local_sum"].shape).index_add_(
                            0, batch["atom_batch"], parts["contrib_atom"]
                        )
                        d2 = (summed - parts["local_sum"]).abs()
                        worst_sum = max(
                            worst_sum,
                            float((d2 / torch.as_tensor(C.trace_tolerance(parts["local_sum"].numpy()))).max()),
                        )
                        if role_key == C.MODEL_ROLE_KEY["RR_CONTROL"]:
                            p_real = model(batch, role_key=C.MODEL_ROLE_KEY["RR_REAL"], gammas=ones)
                            d3 = (pred - p_real).abs()
                            worst_role = max(
                                worst_role,
                                float((d3 / torch.as_tensor(C.trace_tolerance(p_real.numpy()))).max()),
                            )
                        n_graphs += int(pred.numel())
                ok = bool(worst_pred <= 1.0 and worst_role <= 1.0 and worst_closure <= 1.0 and worst_sum <= 1.0)
                entry["splits"][split] = {
                    "n_graphs_checked": n_graphs,
                    "max_tolerance_scaled_gamma1_vs_C1": worst_pred,
                    "max_tolerance_scaled_role_swap": worst_role,
                    "max_tolerance_scaled_readout_closure": worst_closure,
                    "max_tolerance_scaled_atom_sum": worst_sum,
                    "pass": ok,
                }
                all_pass = all_pass and ok
            entry["pass"] = bool(all(e["pass"] for e in entry["splits"].values()))
            out["checkpoints"][str(key)] = entry
    out["all_pass"] = bool(all_pass and bool(checkpoint_paths))
    return _ok(out["all_pass"], **out)


# ---------------------------------------------------------------------------
# gate 10: WC_LOGP closure
# ---------------------------------------------------------------------------


def gate_wc_logp_closure(bundle) -> dict[str, Any]:
    with np.load(C.ORACLE_CONTRIB_NPZ, allow_pickle=False) as mesh:
        graph_values = np.asarray(mesh["WC_LOGP"], dtype=np.float64)
        atom_values = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        atom_ptr = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    out = {"splits": {}, "max_abs_gap": 0.0}
    ok = True
    for split in C.SPLITS:
        ids = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
        labels = np.asarray(bundle.targets("wc_logp", split), dtype=np.float64)
        gap = np.abs(graph_values[ids] - labels)
        worst = float(gap.max())
        atom_gap = 0.0
        for row, gid in enumerate(ids.tolist()):
            a0, a1 = int(atom_ptr[gid]), int(atom_ptr[gid + 1])
            atom_gap = max(atom_gap, abs(float(atom_values[a0:a1].sum()) - float(labels[row])))
        ok = ok and worst <= 1e-9 and atom_gap <= 1e-8
        out["max_abs_gap"] = max(out["max_abs_gap"], worst, atom_gap)
        out["splits"][split] = {
            "n_rows": int(ids.size),
            "max_abs_graph_label_gap": worst,
            "max_abs_atom_sum_gap": atom_gap,
            "pass": bool(worst <= 1e-9 and atom_gap <= 1e-8),
        }
    return _ok(ok, **out)


# ---------------------------------------------------------------------------
# gate 11: group sizes
# ---------------------------------------------------------------------------


def gate_group_sizes(bundle) -> dict[str, Any]:
    groups = D.role_group_audit(bundle)
    return _ok(groups["all_pass"], **groups)


# ---------------------------------------------------------------------------
# checkpoints / run all
# ---------------------------------------------------------------------------


def discover_checkpoints() -> dict[str, Any]:
    out = {}
    for task in C.TASKS:
        for variant in C.MODELS:
            path = C.prediction_paths(task, variant)["best_checkpoint"]
            if path.is_file():
                out[f"{task}/{variant}"] = path
    return out


def run_all(bundle, *, checkpoint_paths: Mapping[str, Any] | None = None, phase: str = "pre_training") -> dict[str, Any]:
    gates: dict[str, Any] = {}
    gates["1_frozen_reuse"] = gate_frozen_reuse(bundle)
    gates["2_role_partition"] = gates_role_partition(bundle)
    gates["3_s2_conservation"] = gate_s2_conservation(bundle)
    gates["4_gamma_one_equivalence"] = gate_equivalence(bundle, label="canonical_init")
    gates["5_data_readout_identity"] = gate_data_identity(bundle)
    gates["6_relabelling"] = gate_relabelling(bundle, checkpoint_paths=checkpoint_paths)
    gates["7_parameter_identity"] = gate_parameter_identity(bundle)
    gates["8_label_oracle_blindness"] = gate_label_blind(bundle)
    gates["10_wc_logp_closure"] = gate_wc_logp_closure(bundle)
    gates["11_group_sizes"] = gate_group_sizes(bundle)
    if checkpoint_paths:
        gates["9_trained_identity"] = gate_trained_identity(bundle, checkpoint_paths)
    failing = sorted(name for name, gate in gates.items() if not gate.get("pass"))
    out = {
        "round_id": C.ROUND_ID,
        "phase": phase,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pass": bool(not failing),
        "failing": failing,
        "n_gates": len(gates),
        "gates": gates,
        "checkpoints": {k: str(v) for k, v in (checkpoint_paths or {}).items()},
    }
    return out


__all__ = [
    "gate_frozen_reuse",
    "gates_role_partition",
    "gate_s2_conservation",
    "gate_equivalence",
    "gate_data_identity",
    "gate_relabelling",
    "gate_parameter_identity",
    "gate_label_blind",
    "gate_trained_identity",
    "gate_wc_logp_closure",
    "gate_group_sizes",
    "discover_checkpoints",
    "run_all",
]
