"""CPU correctness gates shared by the audit CLI and the pytest suite.

Every gate returns an explicit ``PASS`` / ``FAIL`` / ``BLOCKED`` status.  The
functions only read label-free structures and run CPU eval forwards; no gate
creates an optimizer, calls backward, reads a label or writes a historical
artifact.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import adapters as A
import lcrc_common as C
import lcrc_data as LD
import lcrc_model as LM
import molecular_graph as MG
from feature_spec import FeatureSpec


# ---------------------------------------------------------------------------
# result helper
# ---------------------------------------------------------------------------


def _gate(gate: str, name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        details = fn()
        status = str(details.pop("status", "PASS"))
    except Exception as exc:  # pragma: no cover - recorded as evidence
        details = {"error": f"{type(exc).__name__}: {exc}"}
        status = "FAIL"
    return {
        "gate": gate,
        "name": name,
        "status": status,
        "seconds": float(time.perf_counter() - t0),
        "details": details,
    }


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# synthetic raw rows and reference enumeration
# ---------------------------------------------------------------------------


def _n_components(n_atoms: int, bonds: Sequence[tuple[int, int, int]]) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, _bt in bonds:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    return len({find(i) for i in range(n_atoms)})


def make_raw_lipo_row(
    atom_classes: Sequence[int],
    bonds: Sequence[tuple[int, int, int]],
    *,
    charges: Sequence[int] | None = None,
    hydrogens: Sequence[int] | None = None,
    stereo: Sequence[int] | None = None,
    smiles: str = "",
) -> dict[str, Any]:
    n = len(atom_classes)
    if charges is None:
        charges = [2] * n  # raw bucket 2 == formal charge 0
    if hydrogens is None:
        hydrogens = [0] * n
    if stereo is None:
        stereo = [0] * len(bonds)
    if not (len(charges) == n and len(hydrogens) == n and len(stereo) == len(bonds)):
        raise ValueError("synthetic raw row attribute length mismatch")
    ei0: list[int] = []
    ei1: list[int] = []
    ec: list[int] = []
    es: list[int] = []
    for k, (u, v, bt) in enumerate(bonds):
        u, v, bt = int(u), int(v), int(bt)
        ei0.extend([u, v])
        ei1.extend([v, u])
        ec.extend([bt, bt])
        es.extend([int(stereo[k]), int(stereo[k])])
    return {
        "mol_index": -1,
        "smiles": smiles,
        "atom_class": [int(x) for x in atom_classes],
        "atom_charge": [int(x) for x in charges],
        "atom_h": [int(x) for x in hydrogens],
        "atom_chirality": [0] * n,
        "edge_index": [ei0, ei1],
        "edge_class": ec,
        "edge_stereo": es,
        "n_components": _n_components(n, bonds),
    }


def synthetic_cases(spec: FeatureSpec) -> dict[str, MG.MolecularGraph]:
    """Small hand-built cases covering the edge conditions required by the round."""
    cases: dict[str, dict[str, Any]] = {
        "single_atom": make_raw_lipo_row([2], []),
        "two_atom_single": make_raw_lipo_row([2, 2], [(0, 1, 0)]),
        "three_atom_path": make_raw_lipo_row([2, 2, 2], [(0, 1, 0), (1, 2, 0)]),
        "four_atom_linear": make_raw_lipo_row([2, 2, 2, 2], [(0, 1, 0), (1, 2, 0), (2, 3, 0)]),
        "triangle": make_raw_lipo_row([2, 2, 2], [(0, 1, 0), (1, 2, 0), (0, 2, 0)]),
        "benzene": make_raw_lipo_row([3] * 6, [(0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3), (4, 5, 3), (5, 0, 3)]),
        "branched": make_raw_lipo_row([2, 2, 2, 2], [(0, 1, 0), (0, 2, 0), (0, 3, 0)]),
        "charged": make_raw_lipo_row([2, 7], [(0, 1, 0)], charges=[3, 1]),
        "two_components": make_raw_lipo_row([2, 2, 7], [(0, 1, 0)]),
        "isolated_in_component": make_raw_lipo_row([2, 2, 2], [(0, 1, 0)]),
        "unknown_atom_and_bond": make_raw_lipo_row([15, 15, 15], [(0, 1, 4), (1, 2, 4)]),
    }
    out: dict[str, MG.MolecularGraph] = {}
    for i, (name, row) in enumerate(cases.items()):
        out[name] = A.lipo_row_to_graph(row, i, spec)
    return out


def reference_enumeration(
    atom_type: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_type: np.ndarray,
    n_bond_types: int,
) -> dict[str, Any]:
    """Simple independent S1/S2 enumeration written without production helpers."""
    n = int(len(atom_type))
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, bt in zip(np.asarray(bond_u).tolist(), np.asarray(bond_v).tolist(), np.asarray(bond_type).tolist()):
        adj[int(u)].append((int(v), int(bt)))
        adj[int(v)].append((int(u), int(bt)))
    for row in adj:
        row.sort()
    degree = np.asarray([len(row) for row in adj], dtype=np.int64)
    s1: list[list[int]] = []
    s2: list[list[tuple[int, int, int, int]]] = []
    nb = int(n_bond_types)
    for root in range(n):
        s1.append(
            sorted(int(atom_type[middle]) * nb + (int(bt) - 1) for middle, bt in adj[root])
        )
        nbr_set = {middle for middle, _bt in adj[root]}
        records: list[tuple[int, int, int, int]] = []
        for middle, bt1 in adj[root]:
            for end, bt2 in adj[middle]:
                if end == root:
                    continue
                records.append(
                    (
                        int(atom_type[middle]) * nb * nb + (int(bt1) - 1) * nb + (int(bt2) - 1),
                        int(end),
                        int(atom_type[end]) * nb + (int(bt2) - 1),
                        1 if int(end) in nbr_set else 0,
                    )
                )
        s2.append(sorted(records))
    return {
        "degree": degree,
        "n_bonds": int(len(bond_u)),
        "s1": s1,
        "s2": s2,
    }


def mol_s1_rows(mol: Mapping[str, Any]) -> list[list[int]]:
    ptr = np.asarray(mol["nbr_ptr"], dtype=np.int64)
    code = np.asarray(mol["nbr_code"], dtype=np.int64)
    return [sorted(code[int(ptr[v]) : int(ptr[v + 1])].tolist()) for v in range(int(mol["n_atoms"]))]


def mol_s2_records(mol: Mapping[str, Any]) -> list[list[tuple[int, int, int, int]]]:
    ptr = np.asarray(mol["path_ptr"], dtype=np.int64)
    cat = np.asarray(mol["path_cat"], dtype=np.int64)
    end = np.asarray(mol["path_end"], dtype=np.int64)
    end63 = np.asarray(mol["path_end63"], dtype=np.int64)
    closure = np.asarray(mol["path_closure"], dtype=np.int64)
    out: list[list[tuple[int, int, int, int]]] = []
    for v in range(int(mol["n_atoms"])):
        lo, hi = int(ptr[v]), int(ptr[v + 1])
        out.append(sorted(zip(cat[lo:hi].tolist(), end[lo:hi].tolist(), end63[lo:hi].tolist(), closure[lo:hi].tolist())))
    return out


def compare_to_reference(graph: MG.MolecularGraph, mol: Mapping[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    ref = reference_enumeration(graph.atom_type, graph.bond_u, graph.bond_v, graph.bond_type, spec.n_bond_types)
    s1_prod = mol_s1_rows(mol)
    s2_prod = mol_s2_records(mol)
    degree_prod = np.diff(np.asarray(mol["nbr_ptr"], dtype=np.int64))
    return {
        "n_bonds_equal": bool(ref["n_bonds"] == int(mol["n_bonds"])),
        "degree_equal": bool(np.array_equal(ref["degree"], degree_prod)),
        "s1_equal": bool(ref["s1"] == s1_prod),
        "s2_equal": bool(ref["s2"] == s2_prod),
        "n_s1": int(sum(len(x) for x in s1_prod)),
        "n_s2": int(sum(len(x) for x in s2_prod)),
        "max_abs_degree_diff": int(np.abs(ref["degree"] - degree_prod).max()) if len(degree_prod) else 0,
    }


# ---------------------------------------------------------------------------
# graph transformations
# ---------------------------------------------------------------------------


def permute_graph(graph: MG.MolecularGraph, perm: np.ndarray) -> MG.MolecularGraph:
    n = graph.n_atoms
    perm = np.asarray(perm, dtype=np.int64)
    if sorted(perm.tolist()) != list(range(n)):
        raise ValueError("perm must be a permutation")
    inv = np.argsort(perm)
    records = []
    for k in range(graph.n_bonds):
        new_u = int(perm[int(graph.bond_u[k])])
        new_v = int(perm[int(graph.bond_v[k])])
        st = None if graph.bond_stereo is None else int(graph.bond_stereo[k])
        records.append(
            (
                min(new_u, new_v),
                max(new_u, new_v),
                int(graph.bond_type[k]),
                np.asarray(graph.bond_feat[k], dtype=np.float32).copy(),
                st,
            )
        )
    records.sort(key=lambda x: (x[0], x[1]))
    bond_index = np.asarray([(r[0], r[1]) for r in records], dtype=np.int64) if records else np.zeros((0, 2), np.int64)
    bond_feat = (
        np.stack([r[3] for r in records], axis=0).astype(np.float32)
        if records
        else np.zeros((0, graph.bond_feat.shape[1]), np.float32)
    )
    bond_type = np.asarray([r[2] for r in records], dtype=np.int64)
    stereo = np.asarray([r[4] for r in records], dtype=np.int64) if graph.bond_stereo is not None else None
    return MG.build_molecular_graph(
        atom_type=graph.atom_type[inv],
        bond_index=bond_index,
        bond_type=bond_type,
        atom_feat=graph.atom_feat[inv],
        bond_feat=bond_feat,
        global_feat=graph.global_feat,
        molecule_id=graph.molecule_id,
        smiles=graph.smiles,
        raw_atom_charge=None if graph.raw_atom_charge is None else graph.raw_atom_charge[inv],
        raw_atom_h=None if graph.raw_atom_h is None else graph.raw_atom_h[inv],
        bond_stereo=stereo,
        raw_metadata=graph.raw_metadata,
    )


def shuffle_bond_records(graph: MG.MolecularGraph, rng: np.random.Generator) -> MG.MolecularGraph:
    records = []
    for k in range(graph.n_bonds):
        u, v = int(graph.bond_u[k]), int(graph.bond_v[k])
        if rng.random() < 0.5:
            u, v = v, u
        st = None if graph.bond_stereo is None else int(graph.bond_stereo[k])
        records.append((u, v, int(graph.bond_type[k]), np.asarray(graph.bond_feat[k], np.float32).copy(), st))
    order = rng.permutation(len(records))
    records = [records[int(i)] for i in order]
    bond_index = np.asarray([(r[0], r[1]) for r in records], dtype=np.int64) if records else np.zeros((0, 2), np.int64)
    bond_feat = (
        np.stack([r[3] for r in records], axis=0).astype(np.float32)
        if records
        else np.zeros((0, graph.bond_feat.shape[1]), np.float32)
    )
    bond_type = np.asarray([r[2] for r in records], dtype=np.int64)
    stereo = np.asarray([r[4] for r in records], dtype=np.int64) if graph.bond_stereo is not None else None
    return MG.build_molecular_graph(
        atom_type=graph.atom_type.copy(),
        bond_index=bond_index,
        bond_type=bond_type,
        atom_feat=graph.atom_feat.copy(),
        bond_feat=bond_feat,
        global_feat=graph.global_feat,
        molecule_id=graph.molecule_id,
        smiles=graph.smiles,
        raw_atom_charge=None if graph.raw_atom_charge is None else graph.raw_atom_charge.copy(),
        raw_atom_h=None if graph.raw_atom_h is None else graph.raw_atom_h.copy(),
        bond_stereo=stereo,
        raw_metadata=graph.raw_metadata,
    )


def shuffle_path_records(mol: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mol.items()}
    ptr = np.asarray(out["path_ptr"], dtype=np.int64)
    arrays = {key: np.asarray(out[key]).copy() for key in ("path_cat", "path_end63", "path_end", "path_closure")}
    for v in range(int(out["n_atoms"])):
        lo, hi = int(ptr[v]), int(ptr[v + 1])
        count = hi - lo
        if count > 1:
            perm = rng.permutation(count)
            for key, arr in arrays.items():
                arr[lo:hi] = arr[lo:hi][perm]
    out.update(arrays)
    return out


def single_batch(mol: Mapping[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    cache = LD.concat_molecules([mol], spec, split="single")
    return LD.collate(cache, [0])


def model_predict(model: torch.nn.Module, batch: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        y, aux = model(batch, return_aux=True)
    return y.detach().cpu().numpy(), {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in aux.items()}


def validate_batch_indices(batch: Mapping[str, Any]) -> dict[str, Any]:
    atom_batch = batch["atom_batch"].cpu().numpy()
    bond_batch = batch["bond_batch"].cpu().numpy()
    path_graph = batch["path_graph"].cpu().numpy()
    bi = batch["bond_index"].cpu().numpy()
    pe = batch["path_end"].cpu().numpy()
    n_atoms = int(atom_batch.shape[0])
    n_graphs = int(batch["n_graphs"])
    checks = {
        "atom_batch_range": bool(n_atoms == 0 or (atom_batch.min() >= 0 and atom_batch.max() < n_graphs)),
        "bond_batch_range": bool(len(bond_batch) == 0 or (bond_batch.min() >= 0 and bond_batch.max() < n_graphs)),
        "path_graph_range": bool(len(path_graph) == 0 or (path_graph.min() >= 0 and path_graph.max() < n_graphs)),
        "bond_endpoints_in_range": bool(len(bi) == 0 or (bi.min() >= 0 and bi.max() < n_atoms)),
        "path_end_in_range": bool(len(pe) == 0 or (pe.min() >= 0 and pe.max() < n_atoms)),
        "nbr_atom_in_range": bool(len(batch["nbr_atom"]) == 0 or (batch["nbr_atom"].min() >= 0 and batch["nbr_atom"].max() < n_atoms)),
        "path_atom_in_range": bool(len(batch["path_atom"]) == 0 or (batch["path_atom"].min() >= 0 and batch["path_atom"].max() < n_atoms)),
        "bond_endpoint_graph_consistent": bool(
            len(bi) == 0
            or (
                np.array_equal(atom_batch[bi[:, 0]], bond_batch)
                and np.array_equal(atom_batch[bi[:, 1]], bond_batch)
            )
        ),
        "path_end_graph_consistent": bool(len(pe) == 0 or np.array_equal(atom_batch[pe], path_graph)),
    }
    atom_off = batch["atom_off"].cpu().numpy()
    atom_counts = batch["atom_counts"].cpu().numpy()
    if n_graphs:
        ends = np.concatenate([atom_off, np.asarray([n_atoms], dtype=np.int64)])
        checks["atom_off_consistent"] = bool(
            atom_off[0] == 0
            and np.array_equal(np.diff(ends), atom_counts)
        )
    else:
        checks["atom_off_consistent"] = bool(len(atom_off) == 0)
    if not all(checks.values()):
        raise AssertionError(f"batch index check failed: {checks}")
    return {"checks": checks, "n_graphs": n_graphs, "n_atoms": n_atoms, "n_bonds": int(len(bi)), "n_paths": int(len(pe))}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_sources() -> dict[str, Any]:
    manifest = A.source_manifest()
    expected = C.EXPECTED_HASHES
    checks = {
        "raw_graph_hash": manifest["hashes"]["raw_graph"] == expected["lipo_raw"],
        "split_hash": manifest["hashes"]["frozen_split"] == expected["lipo_split"],
        "row_manifest_hash": manifest["hashes"]["row_manifest"] == expected["row_manifest"],
        "csv_hash_only": manifest["hashes"]["csv"] == expected["lipo_csv"],
        "split_counts": manifest["split"]["counts"] == {"fit": 2352, "monitor": 504, "reserved": 504},
        "split_no_overlap": all(v == 0 for v in manifest["split"]["overlap"].values()),
        "union_3360": manifest["split"]["union_size"] == 3360,
        "union_equals_official_train": manifest["split"]["union_equals_official_train_rows"] is True,
        "scaffold_overlap_zero": all(v == 0 for v in manifest["split"]["scaffold_group_overlap"].values()),
        "scaffold_counts_declared": manifest["split"]["scaffold_group_counts"] == manifest["split"]["scaffold_group_counts_declared"],
        "labels_not_read": manifest["label_isolation"]["label_values_parsed"] is False,
        "row_1646_not_silently_dropped": manifest["split"]["row_1646_in_frozen_union"] is False
        and manifest["split"]["row_1646_official_membership"] in {"valid", "test"},
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "manifest": manifest,
    }


def gate_row_manifest_alignment(
    raw_rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int]],
    row_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    n_checked = 0
    canonical_checked = 0
    canonical_mismatches = 0
    rdkit_available = True
    for split_name in ("fit", "monitor", "reserved"):
        for row_index in split[split_name]:
            i = int(row_index)
            row = raw_rows[i]
            n_checked += 1
            if row["compound_id"] != row_manifest["compound_ids"][i]:
                mismatches.append({"row": i, "field": "compound_id"})
            if len(row["atom_class"]) != int(row_manifest["n_atoms"][i]):
                mismatches.append({"row": i, "field": "n_atoms"})
            if int(row["n_components"]) != int(row_manifest["n_components"][i]):
                mismatches.append({"row": i, "field": "n_components"})
            expected_hash = row_manifest["canonical_smiles_sha256"][i]
            actual = A.canonical_smiles_sha256(str(row["smiles"]))
            if actual is None:
                rdkit_available = False
            else:
                canonical_checked += 1
                if actual != expected_hash:
                    canonical_mismatches += 1
                    if len(mismatches) < 20:
                        mismatches.append({"row": i, "field": "canonical_smiles_sha256"})
    status = "PASS" if not mismatches else "FAIL"
    if not rdkit_available:
        status = "BLOCKED"
    return {
        "status": status,
        "n_rows_checked": n_checked,
        "n_canonical_smiles_checked": canonical_checked,
        "n_canonical_smiles_mismatches": canonical_mismatches,
        "rdkit_available": rdkit_available,
        "mismatches": mismatches,
    }


def gate_feature_spec(spec: FeatureSpec) -> dict[str, Any]:
    d = spec.to_dict()
    loaded = FeatureSpec.from_dict(d)
    roundtrip = loaded.hash() == spec.hash()
    checks = {
        "serialisation_roundtrip": roundtrip,
        "names_present": all(spec.atom_type_names) and all(spec.bond_type_names),
        "unknown_policy_recorded": bool(spec.atom_unknown_policy) and bool(spec.bond_unknown_policy),
        "atom_fields_sum": spec.atom_feat_dim == sum(f.dim for f in spec.atom_feature_fields),
        "bond_fields_sum": spec.bond_feat_dim == sum(f.dim for f in spec.bond_feature_fields),
        "scalar_fields_sum": spec.n_scalars == sum(f.dim for f in spec.atom_scalar_fields),
        "global_fields_sum": spec.global_dim == sum(f.dim for f in spec.global_feature_fields),
        "code_dtype_safe": np.dtype(spec.structural_code_dtype).itemsize >= 4,
        "path_vocab_fits_dtype": spec.n_path_codes <= spec.code_max + 1,
        "disconnected_policy_recorded": bool(spec.disconnected_policy),
        "no_label_field_names": not any(
            f.name.lower() in {"exp", "label", "labels", "target", "y", "logp"}
            for group in (spec.atom_feature_fields, spec.bond_feature_fields, spec.atom_scalar_fields, spec.global_feature_fields)
            for f in group
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "spec_hash": spec.hash(),
        "dimensions": d["dimensions"],
    }


def gate_overflow(spec: FeatureSpec) -> dict[str, Any]:
    # A synthetic path with the largest atom and bond categories.
    row = make_raw_lipo_row([15, 15, 15], [(0, 1, 4), (1, 2, 4)])
    graph = A.lipo_row_to_graph(row, 0, spec)
    mol = LD.build_molecule(graph, spec)
    max_path = int(np.max(mol["path_cat"])) if len(mol["path_cat"]) else -1
    max_end63 = int(np.max(mol["path_end63"])) if len(mol["path_end63"]) else -1
    max_nbr = int(np.max(mol["nbr_code"])) if len(mol["nbr_code"]) else -1
    # NumPy 2.x raises on np.uint8(399); cast an array to observe the wrap.
    uint8_wrap_value = int(np.asarray([max_path], dtype=np.int64).astype(np.uint8)[0])
    checks = {
        "path_code_max_is_vocab_max": max_path == spec.n_path_codes - 1,
        "path_code_above_255": max_path > 255,
        "end63_code_max_is_vocab_max": max_end63 == spec.n_nbr_codes - 1,
        "nbr_code_max_is_vocab_max": max_nbr == spec.n_nbr_codes - 1,
        "path_dtype_at_least_4_bytes": np.asarray(mol["path_cat"]).dtype.itemsize >= 4,
        "nbr_dtype_at_least_4_bytes": np.asarray(mol["nbr_code"]).dtype.itemsize >= 4,
        "uint8_would_wrap": uint8_wrap_value != max_path,
        "all_codes_below_vocab": bool(np.all(np.asarray(mol["path_cat"]) < spec.n_path_codes)),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_path_code": max_path,
        "uint8_wrap_value": uint8_wrap_value,
        "n_path_codes": spec.n_path_codes,
        "n_nbr_codes": spec.n_nbr_codes,
        "dtype": str(np.asarray(mol["path_cat"]).dtype),
    }


def gate_reference_enumeration(
    graphs: Sequence[MG.MolecularGraph],
    spec: FeatureSpec,
    *,
    label: str = "sample",
) -> dict[str, Any]:
    results = []
    for graph in graphs:
        mol = LD.build_molecule(graph, spec)
        cmp = compare_to_reference(graph, mol, spec)
        results.append(cmp)
    all_ok = all(
        r["n_bonds_equal"] and r["degree_equal"] and r["s1_equal"] and r["s2_equal"] for r in results
    )
    return {
        "status": "PASS" if all_ok else "FAIL",
        "label": label,
        "n_graphs": len(graphs),
        "all_equal": bool(all_ok),
        "total_s1": int(sum(r["n_s1"] for r in results)),
        "total_s2": int(sum(r["n_s2"] for r in results)),
        "max_abs_degree_diff": int(max((r["max_abs_degree_diff"] for r in results), default=0)),
        "per_graph": results[:20],
    }


def gate_edge_cases(spec: FeatureSpec, model: torch.nn.Module) -> dict[str, Any]:
    cases = synthetic_cases(spec)
    records: dict[str, Any] = {}
    ok = True
    for name, graph in cases.items():
        mol = LD.build_molecule(graph, spec)
        cmp = compare_to_reference(graph, mol, spec)
        batch = single_batch(mol, spec)
        y, aux = model_predict(model, batch)
        finite = bool(np.isfinite(y).all() and np.isfinite(aux["h"]).all())
        decomp = float(np.max(np.abs(y - (aux["contrib"] + aux["rho"]))))
        expected_paths = {
            "single_atom": 0,
            "two_atom_single": 0,
            "three_atom_path": 2,
            "four_atom_linear": 4,
            "triangle": 6,
            "benzene": 12,
            "branched": 6,
            "charged": 0,
            "two_components": 0,
            "isolated_in_component": 0,
            "unknown_atom_and_bond": 2,
        }[name]
        case_ok = (
            cmp["s1_equal"]
            and cmp["s2_equal"]
            and cmp["degree_equal"]
            and cmp["n_bonds_equal"]
            and int(cmp["n_s2"]) == expected_paths
            and finite
            and decomp < 1e-6
        )
        ok = ok and case_ok
        records[name] = {
            "n_atoms": graph.n_atoms,
            "n_bonds": graph.n_bonds,
            "n_components": graph.n_components,
            "n_s1": int(cmp["n_s1"]),
            "n_s2": int(cmp["n_s2"]),
            "reference_equal": bool(cmp["s1_equal"] and cmp["s2_equal"] and cmp["degree_equal"]),
            "expected_paths": expected_paths,
            "y": float(y[0]),
            "finite": finite,
            "decomposition_error": decomp,
            "pass": bool(case_ok),
        }
    return {"status": "PASS" if ok else "FAIL", "cases": records}


def _atom_feature_perm_back(mol: Mapping[str, Any], perm: np.ndarray) -> np.ndarray:
    return np.asarray(mol["atom_feat"])[perm]


def _s1_rows_perm_back(mol: Mapping[str, Any], perm: np.ndarray) -> list[list[int]]:
    rows = mol_s1_rows(mol)
    return [rows[int(perm[v])] for v in range(len(perm))]


def _s2_records_perm_back(mol: Mapping[str, Any], perm: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    inv = np.argsort(perm)
    rows = mol_s2_records(mol)
    out: list[list[tuple[int, int, int, int]]] = []
    for v in range(len(perm)):
        new_root = int(perm[v])
        mapped = []
        for cat, end, end63, closure in rows[new_root]:
            mapped.append((int(cat), int(inv[int(end)]), int(end63), int(closure)))
        out.append(sorted(mapped))
    return out


def gate_permutation_invariance(
    graphs: Sequence[MG.MolecularGraph],
    spec: FeatureSpec,
    model: torch.nn.Module,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    max_y = 0.0
    max_feat = 0.0
    max_scalar = 0.0
    max_global = 0.0
    all_ok = True
    per_graph = []
    for graph in graphs:
        n = graph.n_atoms
        perm = rng.permutation(n)
        old_mol = LD.build_molecule(graph, spec)
        new_graph = permute_graph(graph, perm)
        new_mol = LD.build_molecule(new_graph, spec)
        feat_err = float(np.max(np.abs(_atom_feature_perm_back(new_mol, perm) - old_mol["atom_feat"])))
        scalar_err = float(np.max(np.abs(np.asarray(new_mol["atom_scalar"])[perm] - old_mol["atom_scalar"])))
        global_err = float(np.max(np.abs(np.asarray(new_mol["global"]) - old_mol["global"])))
        type_ok = bool(np.array_equal(np.asarray(new_mol["atom_type"])[perm], old_mol["atom_type"]))
        s1_ok = _s1_rows_perm_back(new_mol, perm) == mol_s1_rows(old_mol)
        s2_ok = _s2_records_perm_back(new_mol, perm) == mol_s2_records(old_mol)
        old_y, _ = model_predict(model, single_batch(old_mol, spec))
        new_y, _ = model_predict(model, single_batch(new_mol, spec))
        y_err = float(abs(float(old_y[0]) - float(new_y[0])))
        max_y = max(max_y, y_err)
        max_feat = max(max_feat, feat_err)
        max_scalar = max(max_scalar, scalar_err)
        max_global = max(max_global, global_err)
        ok = type_ok and s1_ok and s2_ok and feat_err <= 1e-6 and scalar_err <= 1e-6 and global_err <= 1e-6 and y_err <= 1e-5
        all_ok = all_ok and ok
        per_graph.append(
            {
                "molecule_id": graph.molecule_id,
                "n_atoms": n,
                "type_ok": type_ok,
                "s1_ok": s1_ok,
                "s2_ok": s2_ok,
                "atom_feat_max_abs": feat_err,
                "scalar_max_abs": scalar_err,
                "global_max_abs": global_err,
                "y_max_abs": y_err,
                "pass": bool(ok),
            }
        )
    return {
        "status": "PASS" if all_ok else "FAIL",
        "n_graphs": len(graphs),
        "max_atom_feat_abs": max_feat,
        "max_scalar_abs": max_scalar,
        "max_global_abs": max_global,
        "max_y_abs": max_y,
        "per_graph": per_graph[:20],
    }


def gate_bond_order_invariance(
    graphs: Sequence[MG.MolecularGraph],
    spec: FeatureSpec,
    model: torch.nn.Module,
    *,
    seed: int = 1,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_ok = True
    max_y = 0.0
    per_graph = []
    for graph in graphs:
        old_mol = LD.build_molecule(graph, spec)
        new_graph = shuffle_bond_records(graph, rng)
        new_mol = LD.build_molecule(new_graph, spec)
        arrays_ok = all(
            np.array_equal(np.asarray(old_mol[k]), np.asarray(new_mol[k]))
            for k in ("atom_feat", "atom_type", "atom_scalar", "bond_feat", "bond_type", "bond_index", "nbr_ptr", "nbr_code", "path_ptr", "path_cat", "path_end63", "path_end", "path_closure", "global")
        )
        old_y, _ = model_predict(model, single_batch(old_mol, spec))
        new_y, _ = model_predict(model, single_batch(new_mol, spec))
        y_err = float(abs(float(old_y[0]) - float(new_y[0])))
        max_y = max(max_y, y_err)
        ok = arrays_ok and y_err <= 1e-6
        all_ok = all_ok and ok
        per_graph.append({"molecule_id": graph.molecule_id, "arrays_exact": arrays_ok, "y_abs": y_err, "pass": bool(ok)})
    return {"status": "PASS" if all_ok else "FAIL", "n_graphs": len(graphs), "max_y_abs": max_y, "per_graph": per_graph[:20]}


def gate_path_order_invariance(
    graphs: Sequence[MG.MolecularGraph],
    spec: FeatureSpec,
    model: torch.nn.Module,
    *,
    seed: int = 2,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_ok = True
    max_y = 0.0
    per_graph = []
    for graph in graphs:
        old_mol = LD.build_molecule(graph, spec)
        new_mol = shuffle_path_records(old_mol, rng)
        old_y, _ = model_predict(model, single_batch(old_mol, spec))
        new_y, _ = model_predict(model, single_batch(new_mol, spec))
        y_err = float(abs(float(old_y[0]) - float(new_y[0])))
        s1_ok = mol_s1_rows(old_mol) == mol_s1_rows(new_mol)
        s2_multiset_ok = all(
            sorted(np.asarray(old_mol[k]).tolist()) == sorted(np.asarray(new_mol[k]).tolist())
            for k in ("path_cat", "path_end63", "path_end", "path_closure")
        )
        max_y = max(max_y, y_err)
        ok = s1_ok and s2_multiset_ok and y_err <= 1e-5
        all_ok = all_ok and ok
        per_graph.append({"molecule_id": graph.molecule_id, "s1_ok": s1_ok, "s2_multiset_ok": s2_multiset_ok, "y_abs": y_err, "pass": bool(ok)})
    return {"status": "PASS" if all_ok else "FAIL", "n_graphs": len(graphs), "max_y_abs": max_y, "per_graph": per_graph[:20]}


def gate_batch_invariance(
    cache: Mapping[str, Any],
    spec: FeatureSpec,
    model: torch.nn.Module,
    *,
    rows: Sequence[int],
    seed: int = 3,
) -> dict[str, Any]:
    rows = [int(x) for x in rows]
    full_batch = LD.collate(cache, rows)
    index_report = validate_batch_indices(full_batch)
    full_y, full_aux = model_predict(model, full_batch)
    full_by_row = {r: float(full_y[i]) for i, r in enumerate(rows)}
    singles = {}
    max_single = 0.0
    for i, r in enumerate(rows):
        y, _ = model_predict(model, LD.collate(cache, [r]))
        singles[r] = float(y[0])
        max_single = max(max_single, abs(singles[r] - full_by_row[r]))
    rng = np.random.default_rng(seed)
    subset_errors = {}
    for size in (2, 3, 5):
        if size > len(rows):
            continue
        subset = [rows[int(i)] for i in rng.choice(len(rows), size=size, replace=False)]
        y, _ = model_predict(model, LD.collate(cache, subset))
        err = max(abs(float(y[i]) - full_by_row[r]) for i, r in enumerate(subset))
        subset_errors[str(size)] = float(err)
    max_subset = max(subset_errors.values(), default=0.0)
    decomp = float(np.max(np.abs(full_y - (full_aux["contrib"] + full_aux["rho"]))))
    ok = max_single <= 1e-6 and max_subset <= 1e-6 and decomp <= 1e-6
    return {
        "status": "PASS" if ok else "FAIL",
        "n_molecules": len(rows),
        "max_single_vs_batch_abs": max_single,
        "max_subset_vs_batch_abs": max_subset,
        "subset_errors": subset_errors,
        "decomposition_error": decomp,
        "index_report": index_report,
        "finite": bool(np.isfinite(full_y).all()),
    }


def gate_numeric_validity(cache: Mapping[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    report = LD.validate_cache(cache, spec)
    path_cat = np.asarray(cache["path_cat"])
    nbr_code = np.asarray(cache["nbr_code"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    zero_path = 0
    zero_nbr_atoms = 0
    for g in range(int(cache["n_molecules"])):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        if int(path_ptr[a1] - path_ptr[a0]) == 0:
            zero_path += 1
        zero_nbr_atoms += int(np.count_nonzero(np.diff(nbr_ptr[a0 : a1 + 1]) == 0))
    checks = {
        "finite": report["finite"],
        "index_ranges_valid": report["index_ranges_valid"],
        "nbr_code_dtype_ok": nbr_code.dtype.itemsize >= 4,
        "path_cat_dtype_ok": path_cat.dtype.itemsize >= 4,
        "nbr_code_in_range": bool(nbr_code.size == 0 or (nbr_code.min() >= 0 and nbr_code.max() < spec.n_nbr_codes)),
        "path_cat_in_range": bool(path_cat.size == 0 or (path_cat.min() >= 0 and path_cat.max() < spec.n_path_codes)),
        "pointers_monotonic": bool(
            np.all(np.diff(atom_ptr) >= 0)
            and np.all(np.diff(bond_ptr) >= 0)
            and np.all(np.diff(nbr_ptr) >= 0)
            and np.all(np.diff(path_ptr) >= 0)
        ),
        "no_negative_counts": bool(np.all(np.diff(atom_ptr) >= 0) and np.all(np.diff(bond_ptr) >= 0) and np.all(np.diff(nbr_ptr) >= 0) and np.all(np.diff(path_ptr) >= 0)),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cache_report": report,
        "zero_path_molecules": int(zero_path),
        "isolated_atoms": int(zero_nbr_atoms),
        "max_path_code": int(path_cat.max()) if path_cat.size else -1,
        "max_nbr_code": int(nbr_code.max()) if nbr_code.size else -1,
        "n_path_codes": spec.n_path_codes,
        "n_nbr_codes": spec.n_nbr_codes,
    }


def gate_label_independence(
    cache: Mapping[str, Any],
    spec: FeatureSpec,
    *,
    rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    rows = list(range(min(8, int(cache["n_molecules"])))) if rows is None else [int(x) for x in rows]
    batch = LD.collate(cache, rows)
    keys = set(batch.keys())
    forbidden = {
        k
        for k in keys
        if k.lower() in {"label", "labels", "target", "targets", "exp", "y", "logp"}
        or any(t in k.lower() for t in ("label", "target"))
    }
    injected = dict(cache)
    injected["labels"] = np.arange(int(cache["n_molecules"]), dtype=np.float32)
    injected["exp"] = np.arange(int(cache["n_molecules"]), dtype=np.float32) + 1000.0
    batch_injected = LD.collate(injected, rows)
    same = set(batch.keys()) == set(batch_injected.keys()) and all(
        torch.equal(batch[k], batch_injected[k]) for k in batch if torch.is_tensor(batch[k])
    )
    code_text = ""
    for path in (C.CODE_DIR / "adapters.py", C.CODE_DIR / "lcrc_data.py", C.CODE_DIR / "lcrc_model.py", C.CODE_DIR / "audit_cpu.py"):
        if path.exists():
            code_text += path.read_text(encoding="utf-8")
    import re

    forbidden_read = bool(
        re.search(
            r"(?:open|read_json|torch\.load|np\.load)\s*\([^)\n]*(?:labels_fit_monitor|labels\.npy|LabelVault)",
            code_text,
        )
    )
    checks = {
        "no_label_keys_in_batch": not forbidden,
        "dummy_label_injection_ignored": bool(same),
        "no_label_reader_in_code": not forbidden_read,
        "cache_meta_labels_false": cache.get("meta", {}).get("labels_present") is False,
        "no_training_path": not any(k in code_text for k in ("optimizer.step", "loss.backward", "backward()")),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "batch_keys": sorted(keys),
        "rows": rows,
    }


def gate_model_forward(
    cache: Mapping[str, Any],
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    *,
    seed: int = 0,
    rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    rows = list(range(min(64, int(cache["n_molecules"])))) if rows is None else [int(x) for x in rows]
    model, info = LM.build_model(spec, norm, seed=seed, device="cpu", load_ssl=False)
    model.eval()
    batch = LD.collate(cache, rows)
    y, aux = model_predict(model, batch)
    contrib_sum = np.zeros(len(rows), dtype=np.float64)
    # aux["contrib_atom"] is per atom; sum by atom_batch.
    atom_batch = batch["atom_batch"].cpu().numpy()
    np.add.at(contrib_sum, atom_batch, aux["contrib_atom"])
    checks = {
        "y_shape": tuple(y.shape) == (len(rows),),
        "y_finite": bool(np.isfinite(y).all()),
        "h_finite": bool(np.isfinite(aux["h"]).all()),
        "mean_h_finite": bool(np.isfinite(aux["mean_h"]).all()),
        "mean_h2_finite": bool(np.isfinite(aux["mean_h2"]).all()),
        "decomposition": float(np.max(np.abs(y - (aux["contrib"] + aux["rho"])))) <= 1e-6,
        "contrib_grouping": float(np.max(np.abs(contrib_sum - aux["contrib"]))) <= 1e-5,
        "no_relation_head": not any("rel." in k for k in model.state_dict()),
        "no_junction_module": not any("junction" in k for k in model.state_dict()),
        "no_attention_module": not any("attention" in k for k in model.state_dict()),
        "model_eval_mode": model.training is False,
        "parameter_count_positive": info["parameters"] > 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model_info": info,
        "n_molecules": len(rows),
        "decomposition_max_abs_error": float(np.max(np.abs(y - (aux["contrib"] + aux["rho"])))),
        "contrib_grouping_max_abs_error": float(np.max(np.abs(contrib_sum - aux["contrib"]))),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
    }


__all__ = [
    "make_raw_lipo_row",
    "synthetic_cases",
    "reference_enumeration",
    "mol_s1_rows",
    "mol_s2_records",
    "compare_to_reference",
    "permute_graph",
    "shuffle_bond_records",
    "shuffle_path_records",
    "single_batch",
    "model_predict",
    "validate_batch_indices",
    "gate_sources",
    "gate_row_manifest_alignment",
    "gate_feature_spec",
    "gate_overflow",
    "gate_reference_enumeration",
    "gate_edge_cases",
    "gate_permutation_invariance",
    "gate_bond_order_invariance",
    "gate_path_order_invariance",
    "gate_batch_invariance",
    "gate_numeric_validity",
    "gate_label_independence",
    "gate_model_forward",
]
