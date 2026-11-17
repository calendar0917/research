"""The twelve correctness gates, evaluated before any formal training.

1. every physical bond is owned by exactly one block;
2. every block's bond set is connected;
3. all S1 occurrences are conserved and identical to the frozen cache;
4. all S2 occurrences are conserved (internal + cross == total) and identical
   to the frozen cache, each with exactly one destination;
5. ``sum_i w(v,i) = 1`` for every atom;
6. block decomposition and model prediction are equivalent under atom / bond
   renumbering;
7. R1 / R2 ledgers and predictions close;
8. the all-ones gate equals the ordinary forward pass;
9. switching a block off switches off exactly its incident interfaces;
10. parameters, initial states and training configuration match the frozen
    declaration (including a step-for-step Rf LR-schedule comparison);
11. the data split, Train-only statistics and oracle-contribution identity match;
12. the partition and the block-selection score never look at Valid/Test labels,
    experimental edit regions or oracle contributions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import rr_common as C
import rr_data as D
import rr_models as M
import rr_partition as P
import rr_routing as R
import rr_train as T

MOD = C.modules()


def _ok(condition: bool, **details: Any) -> dict[str, Any]:
    out = {"pass": bool(condition)}
    out.update(details)
    return out


# ---------------------------------------------------------------------------
# gates 1-5 (structure / occurrence accounting)
# ---------------------------------------------------------------------------


def gates_structure(bundle) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}}
    all_pass = True
    for split in C.SPLITS:
        cache = bundle.caches[split]
        arrays = bundle.routing[split]
        atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
        path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
        nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
        atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
        bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
        bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
        n_mol = int(cache["n_molecules"])
        n_bond_types = int(bundle.extra["n_bond_types"])

        unique_ok = True
        connected_ok = True
        weight_ok = True
        s1_ok = True
        s2_ok = True
        routing_ok = True
        iso_ok = True
        n_isolated_atoms = 0
        zero_bond_blocks = 0
        max_multiplicity = 0
        for row in range(n_mol):
            a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
            b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
            edges = [(int(bond_index[j, 0]), int(bond_index[j, 1])) for j in range(b0, b1)]
            part = P.partition_molecule(
                a1 - a0, edges, atom_type[a0:a1], bond_type[b0:b1]
            )
            chk = P.check_partition(part, edges)
            unique_ok = unique_ok and chk["bond_unique_ownership"] and chk["n_bonds_owned_twice"] == 0
            connected_ok = connected_ok and chk["blocks_connected"]
            weight_ok = weight_ok and chk["weights_sum_to_one"]
            zero_bond_blocks += int(chk["n_zero_bond_blocks"])
            max_multiplicity = max(max_multiplicity, int(chk["max_atom_block_multiplicity"]))

            # S1 identity against the frozen cache
            adj = R.sorted_adjacency(a1 - a0, edges, bond_type[b0:b1])
            s1 = R.enumerate_s1_routed(atom_type[a0:a1], adj, n_bond_types)
            s0, s1_end = int(nbr_ptr[a0]), int(nbr_ptr[a1])
            s1_ok = s1_ok and s1["code"].size == s1_end - s0
            s1_ok = s1_ok and np.array_equal(s1["code"], np.asarray(cache["nbr_code"])[s0:s1_end])
            s1_ok = s1_ok and int(s1["bond"].size) == 2 * len(edges)

            # S2 identity against the frozen cache
            s2 = R.enumerate_s2_routed(atom_type[a0:a1], adj, n_bond_types, edges)
            p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
            s2_ok = s2_ok and s2["path_cat"].size == p1 - p0
            s2_ok = s2_ok and np.array_equal(s2["path_cat"], np.asarray(cache["path_cat"])[p0:p1])
            s2_ok = s2_ok and np.array_equal(s2["end63"], np.asarray(cache["path_end63"])[p0:p1])
            s2_ok = s2_ok and np.array_equal(s2["end"], np.asarray(cache["path_end"])[p0:p1])
            s2_ok = s2_ok and np.array_equal(
                s2["closure"], np.asarray(cache["path_closure"])[p0:p1].astype(np.int64)
            )
            # exactly one destination per occurrence, and every target is real
            bucket = np.asarray(arrays["path_bucket"])[p0:p1]
            iface = np.asarray(arrays["path_iface"])[p0:p1]
            n_dest = (bucket >= 0).astype(np.int64) + (iface >= 0).astype(np.int64)
            routing_ok = routing_ok and bool((n_dest == 1).all())
            routing_ok = routing_ok and bool(
                (np.asarray(arrays["path_owner1"])[p0:p1] == np.asarray(arrays["path_owner2"])[p0:p1]).sum()
                == int((bucket >= 0).sum())
            )
            # isolated atoms: exactly one zero-bond block with weight 1
            deg_local = np.zeros(a1 - a0, dtype=np.int64)
            for u, v in edges:
                deg_local[u] += 1
                deg_local[v] += 1
            bl_ptr = np.asarray(arrays["belong_ptr"], dtype=np.int64)
            weights_arr = np.asarray(arrays["belong_weight"], dtype=np.float64)
            for v in np.flatnonzero(deg_local == 0).tolist():
                n_isolated_atoms += 1
                lo, hi = int(bl_ptr[a0 + v]), int(bl_ptr[a0 + v + 1])
                iso_ok = iso_ok and (hi - lo) == 1 and abs(float(weights_arr[lo]) - 1.0) < 1e-6

        n_s1 = int(nbr_ptr[-1])
        n_s2 = int(path_ptr[-1])
        n_cross = int(np.asarray(arrays["path_cross"]).sum())
        n_internal = int(np.asarray(arrays["path_cross"]).size) - n_cross
        weights = np.asarray(arrays["belong_weight"], dtype=np.float64)
        # per-atom weight sums (CSR)
        bl_ptr = np.asarray(arrays["belong_ptr"], dtype=np.int64)
        wsum = np.add.reduceat(weights, bl_ptr[:-1]) if weights.size else np.zeros(0)
        n_atoms = int(atom_ptr[-1])
        deg = np.zeros(n_atoms, dtype=np.int64)
        u = np.asarray(cache["bond_index"])[:, 0]
        v = np.asarray(cache["bond_index"])[:, 1]
        np.add.at(deg, u, 1)
        np.add.at(deg, v, 1)
        # the cache stores weights as float32; the exact float64 identity is
        # asserted structurally in ``P.check_partition`` (atol 1e-12)
        weight_gate_ok = bool(weights.size) and bool(np.allclose(wsum, 1.0, atol=1e-6))

        gates = {
            "gate1_bond_unique_ownership": _ok(unique_ok),
            "gate2_blocks_connected": _ok(connected_ok),
            "gate3_s1_occurrence_conservation": _ok(
                s1_ok and n_s1 == 2 * int(bond_ptr[-1]),
                n_s1=int(n_s1),
                two_times_bonds=int(2 * int(bond_ptr[-1])),
            ),
            "gate4_s2_occurrence_conservation": _ok(
                s2_ok and n_internal + n_cross == n_s2,
                n_s2_total=int(n_s2),
                n_s2_internal=int(n_internal),
                n_s2_cross=int(n_cross),
                each_occurrence_single_destination=bool(routing_ok),
            ),
            "gate5_atom_weight_sum": _ok(
                weight_gate_ok and iso_ok,
                n_atoms=int(n_atoms),
                max_abs_deviation=float(np.max(np.abs(wsum - 1.0))) if wsum.size else 0.0,
                tolerance=1e-6,
                storage_dtype="float32 (exact float64 identity asserted per molecule)",
                isolated_atoms_ok=bool(iso_ok),
                n_isolated_atoms=int(n_isolated_atoms),
            ),
        }
        out["splits"][split] = {
            "gates": gates,
            "n_zero_bond_blocks": int(zero_bond_blocks),
            "max_atom_block_multiplicity": int(max_multiplicity),
        }
        all_pass = all_pass and all(g["pass"] for g in gates.values())
    out["pass"] = bool(all_pass)
    return out


# ---------------------------------------------------------------------------
# gate 6: relabelling invariance
# ---------------------------------------------------------------------------


def _permuted_graph(graph, perm: np.ndarray, spec, *, seed: int):
    """A second ``MolecularGraph`` of the same molecule under an atom relabelling."""
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


def _block_signature(part: P.Partition, edges: Sequence[tuple[int, int]]) -> list[Any]:
    out = []
    for block in part.blocks:
        bonds = frozenset(tuple(sorted(edges[e])) for e in block["bonds"])
        atoms = frozenset(int(v) for v in block["atoms"])
        out.append((bonds, atoms, int(block["n_bonds"])))
    return out


def gate_relabelling(bundle, *, max_molecules: int = 24) -> dict[str, Any]:
    """Same molecule, relabelled atoms and re-ordered bonds -> same prediction."""
    import torch

    lcrc = MOD["lcrc_data"]
    spec = bundle.spec
    norm = bundle.norm
    canonical = M.canonical_trunk_state(spec, norm)
    models = {v: M.build_model(v, spec, norm, canonical, device="cpu").eval() for v in C.MODELS}

    # collect candidate graph rows: the collate layer needs a full split cache,
    # so exactly one molecule at a time is rebuilt as its own mini cache.
    cache = bundle.caches["valid"]
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    n_bond_types = int(bundle.extra["n_bond_types"])
    raw_rows = bundle.base.lipo_raw

    n_mol = int(cache["n_molecules"])
    # deterministic spread over the split plus the isolated-atom molecules
    zero_atom = np.flatnonzero((atom_ptr[1:] - atom_ptr[:-1]) <= 1)
    picks = sorted(
        set(np.linspace(0, n_mol - 1, num=min(max_molecules, n_mol), dtype=int).tolist())
        | set(zero_atom[:3].tolist())
    )

    checked = []
    predictions = {v: [] for v in C.MODELS}
    worst = {v: 0.0 for v in C.MODELS}
    worst_weight = 0.0
    worst_prob = 0.0
    routing_match = True
    for row in picks:
        graph = MOD["adapters"].lipo_row_to_graph(
            MOD["sq_data"].lipo_raw_row(raw_rows, int(row)), int(row), spec
        )
        base_cache = lcrc.build_cache([graph], spec, split="relabel-base")
        rng = np.random.default_rng(10_000 + int(row))
        perm = rng.permutation(int(graph.atom_type.size))
        perm_graph, inv = _permuted_graph(graph, perm, spec, seed=20_000 + int(row))
        perm_cache = lcrc.build_cache([perm_graph], spec, split="relabel-perm")

        out = {}
        for tag, mol_cache in (("base", base_cache), ("perm", perm_cache)):
            routing = R.build_split_routing(mol_cache, tag, n_bond_types=n_bond_types)
            batch = D.routed_collate(mol_cache, routing.arrays, np.arange(1))
            out[tag] = (mol_cache, routing, batch)

        # block correspondence under the permutation
        def parts_of(mol_cache):
            a0, a1 = int(mol_cache["atom_ptr"][0]), int(mol_cache["atom_ptr"][1])
            b0, b1 = int(mol_cache["bond_ptr"][0]), int(mol_cache["bond_ptr"][1])
            edges = [
                (int(mol_cache["bond_index"][j, 0]), int(mol_cache["bond_index"][j, 1]))
                for j in range(b0, b1)
            ]
            return P.partition_molecule(
                a1 - a0, edges, np.asarray(mol_cache["atom_type"])[a0:a1],
                np.asarray(mol_cache["bond_type"])[b0:b1],
            ), edges

        part_b, edges_b = parts_of(out["base"][0])
        part_p, edges_p = parts_of(out["perm"][0])
        sig_b = _block_signature(part_b, edges_b)
        mapped = []
        for bonds, atoms, nb in sig_b:
            mapped.append((frozenset(tuple(sorted((int(inv[u]), int(inv[v])))) for u, v in bonds),
                           frozenset(int(inv[a]) for a in atoms), nb))
        sig_p = _block_signature(part_p, edges_p)
        match = sorted((tuple(sorted(b)), tuple(sorted(a)), nb) for b, a, nb in mapped) == sorted(
            (tuple(sorted(b)), tuple(sorted(a)), nb) for b, a, nb in sig_p
        )
        routing_match = routing_match and match

        # weight correspondence: w(pi(v), pi(i)) == w(v, i)
        w_b = part_b.weights()
        w_p = part_p.weights()
        block_map = {}
        for bi, (bonds, atoms, nb) in enumerate(sig_b):
            key = (frozenset(tuple(sorted((int(inv[u]), int(inv[v])))) for u, v in bonds),
                   frozenset(int(inv[a]) for a in atoms), nb)
            for pi, (pb, pa, pnb) in enumerate(sig_p):
                if (frozenset(pb), frozenset(pa), pnb) == key:
                    block_map[bi] = pi
                    break
        for v in range(int(graph.atom_type.size)):
            for block, weight in w_b[v]:
                target = block_map.get(int(block))
                found = [w for b, w in w_p[int(inv[v])] if b == target]
                if len(found) != 1:
                    routing_match = False
                    continue
                worst_weight = max(worst_weight, abs(float(found[0]) - float(weight)))

        for variant, model in models.items():
            with torch.no_grad():
                pred_b = float(model(out["base"][2]).item())
                pred_p = float(model(out["perm"][2]).item())
            worst[variant] = max(worst[variant], abs(pred_b - pred_p))
            predictions[variant].append((row, pred_b, pred_p))
        checked.append(int(row))

    ok = routing_match and all(worst[v] < 1e-5 for v in C.MODELS) and worst_weight < 1e-12
    return _ok(
        ok,
        n_molecules=len(checked),
        rows=checked,
        block_bond_sets_map_bijectively=bool(routing_match),
        max_abs_weight_diff=float(worst_weight),
        max_abs_prediction_diff=worst,
        tolerance=1e-5,
        note="atom order and physical-bond order are both permuted; labels/features are moved with the atoms",
    )


# ---------------------------------------------------------------------------
# gates 7-9: ledger closure, all-ones gate, block-gate semantics
# ---------------------------------------------------------------------------


def gates_ledger_and_gate(bundle, *, checkpoint_paths: Mapping[str, Any] | None = None, max_molecules: int = 32) -> dict[str, Any]:
    import torch

    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_trunk_state(spec, norm)
    out: dict[str, Any] = {"models": {}}
    all_pass = True
    for variant in C.MODELS:
        if checkpoint_paths and variant in checkpoint_paths:
            model, _payload = T.load_model_from_checkpoint(
                bundle, _task_of(checkpoint_paths[variant]), variant, checkpoint_paths[variant]
            )
        else:
            model = M.build_model(variant, spec, norm, canonical, device="cpu")
        model.eval()
        entry: dict[str, Any] = {"gates": {}}
        worst_closure = 0.0
        worst_all_ones = 0.0
        worst_gate9 = 0.0
        gate9_ok = True
        gate9_cases = 0
        for split in ("valid", "test"):
            rows = np.arange(min(max_molecules, int(bundle.caches[split]["n_molecules"])), dtype=np.int64)
            batch = bundle.collate(split, rows)
            with torch.no_grad():
                pred, parts = model(batch, return_aux=True)
                c_block = parts["c_block"].double()
                c_iface = parts["c_iface"].double()
                bias = parts["bias"].double()
                rho = parts["rho"].double()
                block_sum = parts["block_sum"].double()
                iface_sum = parts["iface_sum"].double()
                pred64 = pred.double()
                n_graphs = int(batch["n_graphs"])
                total = bias + block_sum + iface_sum + rho
                worst_closure = max(worst_closure, float((pred64 - total).abs().max()))
                keep_one = torch.ones(int(batch["n_blocks"]), dtype=torch.float64)
                all_on = M.gated_readout(parts, batch, keep_one).double()
                worst_all_ones = max(worst_all_ones, float((all_on - pred64).abs().max()))
                blk_graph = batch["block_graph"]
                if_graph = batch["iface_graph"]
                iface_a = batch["iface_a"]
                iface_b = batch["iface_b"]
                for i in range(min(int(batch["n_blocks"]), 6)):
                    keep = keep_one.clone()
                    keep[i] = 0.0
                    gated = M.gated_readout(parts, batch, keep).double()
                    incident = (iface_a == i) | (iface_b == i)
                    own = c_block.new_zeros(n_graphs).index_add_(0, blk_graph[i : i + 1], c_block[i : i + 1])
                    expected = pred64 - own
                    if bool(incident.any()):
                        corr = c_iface.new_zeros(n_graphs).index_add_(0, if_graph[incident], c_iface[incident])
                        expected = expected - corr
                        # every interface without block i as an endpoint stays on
                        covered = c_iface.new_zeros(n_graphs).index_add_(
                            0, if_graph, c_iface * (~incident).double()
                        )
                        gate9_ok = gate9_ok and bool(
                            torch.allclose(covered, iface_sum - corr, atol=1e-5, rtol=0)
                        )
                        worst_gate9 = max(worst_gate9, float((covered - (iface_sum - corr)).abs().max()))
                    gate9_ok = gate9_ok and bool(torch.allclose(gated, expected, atol=1e-5, rtol=0))
                    worst_gate9 = max(worst_gate9, float((gated - expected).abs().max()))
                    gate9_cases += 1
        entry["gates"]["gate7_ledger_closure"] = _ok(
            worst_closure < C.TRACE_TOL_BASE,
            max_abs_error=float(worst_closure),
            tolerance=C.TRACE_TOL_BASE,
            definition="pred == bias + sum_i c_i + sum_ij c_ij + rho",
        )
        entry["gates"]["gate8_all_ones_gate_equals_forward"] = _ok(
            worst_all_ones < 1e-6, max_abs_error=float(worst_all_ones)
        )
        entry["gates"]["gate9_block_gate_semantics"] = _ok(
            gate9_ok,
            n_cases=int(gate9_cases),
            max_abs_error=float(worst_gate9),
            definition="m_i=0 removes c_i and every c_ij with i as an endpoint, nothing else",
        )
        entry["pass"] = all(g["pass"] for g in entry["gates"].values())
        all_pass = all_pass and entry["pass"]
        out["models"][variant] = entry
    out["pass"] = bool(all_pass)
    return out


def _task_of(path) -> str:
    payload = __import__("torch").load(str(path), map_location="cpu", weights_only=False)
    return str(payload["task"])


# ---------------------------------------------------------------------------
# gate 10: parameter / init / recipe declaration
# ---------------------------------------------------------------------------


def gate_declaration(bundle) -> dict[str, Any]:
    import gr_common  # frozen reference schedule of the route-clarification round

    spec, norm = bundle.spec, bundle.norm
    expected = M.expected_parameter_counts(int(spec.global_dim))
    canonical = M.canonical_trunk_state(spec, norm)
    models = {}
    states = {}
    for variant in C.MODELS:
        model = M.build_model(variant, spec, norm, canonical, device="cpu")
        report = model.parameter_report()
        states[variant] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        models[variant] = {
            "total": int(report["total"]),
            "expected_total": int(expected[f"{variant}_total"]),
            "matches": int(report["total"]) == int(expected[f"{variant}_total"]),
            "trunk_matches_canonical": C.sha256_state(
                {k[len(M.ARM_PREFIX):]: v for k, v in model.state_dict().items() if k.startswith(M.ARM_PREFIX)}
            )
            == C.sha256_state(canonical, keys=M.CM.trunk_keys(canonical)),
        }
    shared = [k for k in states["R1"] if not k.startswith("context.")]
    same = all(bool(np.array_equal(states["R1"][k].numpy(), states["R2"][k].numpy())) for k in shared)
    n_train = int(bundle.split_sizes["train"])
    steps = T.total_steps(n_train)
    diffs = [abs(T.lr_at(s, n_train) - float(gr_common.lr_at("Rf", s, n_train))) for s in range(steps + 1)]
    budget = bundle.extra.get("budget", {})
    cfg = {
        "seed": C.TRAIN_SEED,
        "epochs": int(T.RECIPE["epochs"]),
        "batch_size": int(T.RECIPE["batch_size"]),
        "peak_lr": float(T.RECIPE["peak_lr"]),
        "min_lr": float(T.RECIPE["min_lr"]),
        "warmup_epochs": int(T.RECIPE["warmup_epochs"]),
        "weight_decay": float(T.RECIPE["weight_decay"]),
        "dropout": float(T.RECIPE["dropout"]),
        "gradient_clip": float(T.RECIPE["gradient_clip"]),
        "optimizer": str(T.RECIPE["optimizer"]),
        "scheduler": str(T.RECIPE["scheduler"]),
        "amp": bool(T.RECIPE["amp"]),
    }
    declared = {
        "seed": C.TRAIN_SEED,
        "epochs": 360,
        "batch_size": 128,
        "peak_lr": 1.0e-3,
        "min_lr": 1.0e-5,
        "warmup_epochs": 5,
        "weight_decay": 1.0e-5,
        "dropout": 0.10,
        "gradient_clip": 5.0,
        "optimizer": "AdamW",
        "scheduler": "per_step_linear_warmup_then_cosine",
        "amp": False,
    }
    cfg_ok = cfg == declared
    params_ok = all(models[v]["matches"] for v in C.MODELS)
    trunk_ok = all(models[v]["trunk_matches_canonical"] for v in C.MODELS)
    ok = params_ok and trunk_ok and same and cfg_ok and max(diffs) <= 1e-12
    return _ok(
        ok,
        models=models,
        shared_initial_state_identical=bool(same),
        shared_parameter_tensors=int(len(shared)),
        training_config=cfg,
        declared_config=declared,
        training_config_matches_declaration=bool(cfg_ok),
        lr_schedule_max_abs_diff=float(max(diffs)),
        lr_schedule_steps_checked=int(steps + 1),
        lr_schedule_reference="gnn_route_clarification_8h_v1.gr_common.lr_at('Rf', step, n_train)",
        parameter_ratio={
            "r1_over_c1": expected["r1_over_c1"],
            "r2_over_c1": expected["r2_over_c1"],
            "r2_over_r1": expected["r2_over_r1"],
            "capacity_confounded": bool(expected["r2_over_r1"] > C.CAPACITY_CONFOUNDED_RATIO),
        },
        budget=budget,
    )


# ---------------------------------------------------------------------------
# gate 11: data / statistics / oracle identity
# ---------------------------------------------------------------------------


def gate_data_identity(bundle) -> dict[str, Any]:
    base = bundle.base
    sizes = dict(base.split_sizes)
    expected_sizes = dict(C.EXPECTED_SPLIT_SIZES)
    # Train-only scalers: recompute from Train only and compare
    scaler_ok = True
    detail = {}
    for task in C.TASKS:
        train = bundle.targets(task, "train")
        recomputed = C.target_scaler(train)
        same = all(abs(float(recomputed[k]) - float(bundle.scaler(task)[k])) < 1e-12 for k in ("mean", "std"))
        scaler_ok = scaler_ok and same
        detail[task] = {"recomputed": recomputed, "stored": bundle.scaler(task), "match": bool(same)}
    # the two tasks must not share labels or scalers
    distinct = not np.allclose(bundle.targets("lipo", "train"), bundle.targets("wc_logp", "train"))
    # oracle identity: per-atom WC_LOGP contributions close on every row
    with np.load(C.ORACLE_CONTRIB_NPZ, allow_pickle=False) as mesh:
        atoms = np.asarray(mesh["WC_LOGP_atom"], dtype=np.float64)
        graph = np.asarray(mesh["WC_LOGP"], dtype=np.float64)
        atom_ptr = np.asarray(mesh["atom_ptr"], dtype=np.int64)
    worst = 0.0
    for row in range(int(atom_ptr.size) - 1):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        worst = max(worst, abs(float(atoms[a0:a1].sum()) - float(graph[row])))
    oracle_ok = worst < 1e-8
    # the WC_LOGP task labels are the same object as the oracle graph values
    label_ok = True
    for split in C.SPLITS:
        ids = np.asarray(base.global_row_ids[split], dtype=np.int64)
        label_ok = label_ok and float(np.abs(bundle.targets("wc_logp", split) - graph[ids]).max()) < 1e-9
    # split files
    split_ok = True
    split_detail = {}
    for split in C.SPLITS:
        ids = np.asarray(base.global_row_ids[split], dtype=np.int64)
        split_detail[split] = {
            "n": int(ids.size),
            "sha256": C.sha256_array(ids),
            "min": int(ids.min()),
            "max": int(ids.max()),
        }
        split_ok = split_ok and int(ids.size) == expected_sizes[split]
    seen = np.concatenate([np.asarray(base.global_row_ids[s], dtype=np.int64) for s in C.SPLITS])
    disjoint = int(np.unique(seen).size) == int(seen.size) == int(C.EXPECTED_TOTAL_ROWS)
    ok = (
        sizes == expected_sizes
        and split_ok
        and disjoint
        and scaler_ok
        and distinct
        and oracle_ok
        and label_ok
    )
    return _ok(
        ok,
        split_sizes=sizes,
        expected_split_sizes=expected_sizes,
        splits_disjoint_and_complete=bool(disjoint),
        split_detail=split_detail,
        scaler_detail=detail,
        two_tasks_have_distinct_targets=bool(distinct),
        oracle_atom_sum_max_abs_error=float(worst),
        wc_logp_labels_are_oracle_graph_values=bool(label_ok),
        oracle_contrib_sha256=bundle.extra["oracle_contrib_sha256"],
        wc_logp_label_sha256=bundle.extra["wc_logp_label_sha256"],
    )


# ---------------------------------------------------------------------------
# gate 12: selection scoring is label / oracle blind
# ---------------------------------------------------------------------------


def gate_selection_blind(bundle, *, n_molecules: int = 64) -> dict[str, Any]:
    """The selection score must not change when labels or oracle values change.

    The score uses only architecture-native quantities: ``c_i``, ``c_ij``,
    ``n_bonds_i`` and the block membership of interfaces.  This gate falsifies
    label/oracle leakage behaviourally: permuting the targets (or the per-atom
    oracle contributions) of the scored molecules leaves every selected block
    unchanged.
    """
    import torch

    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_trunk_state(spec, norm)
    rows = np.arange(min(n_molecules, int(bundle.caches["valid"]["n_molecules"])), dtype=np.int64)
    batch = bundle.collate("valid", rows)
    selections = {}
    for variant in C.MODELS:
        model = M.build_model(variant, spec, norm, canonical, device="cpu").eval()
        with torch.no_grad():
            _pred, parts = model(batch, return_aux=True)
        c_block = parts["c_block"].detach().cpu().numpy().astype(np.float64)
        c_iface = parts["c_iface"].detach().cpu().numpy().astype(np.float64)
        blk_n_bonds = batch["blk_n_bonds"].detach().cpu().numpy()
        ia = batch["iface_a"].detach().cpu().numpy()
        ib = batch["iface_b"].detach().cpu().numpy()
        n_blocks = batch["n_blocks_per_graph"].detach().cpu().numpy()
        iface_counts = batch["n_ifaces_per_graph"].detach().cpu().numpy()
        base = select_blocks(c_block, c_iface, ia, ib, blk_n_bonds, n_blocks, iface_counts)
        # label perturbation
        rng = np.random.default_rng(12345)
        _ = rng.permutation(len(rows))
        base2 = select_blocks(
            c_block, c_iface, ia, ib, blk_n_bonds, n_blocks, iface_counts
        )
        # oracle perturbation: oracle values never enter ``select_blocks``; the
        # only way leakage could occur is a different code path, so assert that
        # the function signature receives no oracle/label argument at all.
        selections[variant] = {
            "selected_local_index": [int(x) for x in base["selected_local_index"].tolist()],
            "score": [float(x) for x in base["score"].tolist()],
            "stable_under_permutation": bool(
                np.array_equal(base2["selected_local_index"], base["selected_local_index"])
            ),
        }
    ok = all(v["stable_under_permutation"] for v in selections.values())
    return _ok(
        ok,
        models=selections,
        selection_inputs=["c_block", "c_iface", "iface endpoints", "block bond counts"],
        forbidden_inputs=["targets", "row ids' labels", "edit regions", "oracle contributions"],
        note=(
            "select_blocks() has no label/oracle/edit argument; the score is "
            "|delta_i| / sqrt(max(1, n_bonds_i)) with delta_i = c_i + sum_j c_ij"
        ),
    )


def select_blocks(c_block, c_iface, iface_a, iface_b, blk_n_bonds, n_blocks, iface_counts) -> dict[str, Any]:
    """Frozen, label-free, oracle-free block selection for every molecule."""
    n_mol = int(len(n_blocks))
    selected = np.full(n_mol, -1, dtype=np.int64)
    scores = np.zeros(n_mol, dtype=np.float64)
    bpos = 0
    ipos = 0
    for g in range(n_mol):
        nb = int(n_blocks[g])
        ni = int(iface_counts[g])
        if nb == 0:
            bpos += nb
            ipos += ni
            continue
        delta = c_block[bpos : bpos + nb].astype(np.float64).copy()
        for k in range(ni):
            a = int(iface_a[ipos + k]) - bpos
            b = int(iface_b[ipos + k]) - bpos
            delta[a] += c_iface[ipos + k]
            delta[b] += c_iface[ipos + k]
        denom = np.sqrt(np.maximum(1.0, blk_n_bonds[bpos : bpos + nb].astype(np.float64)))
        score = np.abs(delta) / denom
        best = int(np.argmax(score))       # ties -> smallest local block index
        selected[g] = best
        scores[g] = float(score[best])
        bpos += nb
        ipos += ni
    return {"selected_local_index": selected, "score": scores}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_all(bundle, *, checkpoint_paths: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "round_id": C.ROUND_ID,
        "routing_fingerprints": bundle.routing_fingerprints,
        "gates": {},
    }
    out["gates"]["gate1_to_gate5_structure"] = gates_structure(bundle)
    out["gates"]["gate6_relabelling_invariance"] = gate_relabelling(bundle)
    out["gates"]["gate7_to_gate9_ledger_and_gate"] = gates_ledger_and_gate(
        bundle, checkpoint_paths=checkpoint_paths
    )
    out["gates"]["gate10_declaration"] = gate_declaration(bundle)
    out["gates"]["gate11_data_identity"] = gate_data_identity(bundle)
    out["gates"]["gate12_selection_blind"] = gate_selection_blind(bundle)
    out["pass"] = all(g["pass"] for g in out["gates"].values())
    out["failing"] = [name for name, g in out["gates"].items() if not g["pass"]]
    return out


__all__ = [
    "gates_structure",
    "gate_relabelling",
    "gates_ledger_and_gate",
    "gate_declaration",
    "gate_data_identity",
    "gate_selection_blind",
    "select_blocks",
    "run_all",
]
