"""CPU-only, label-free audit for the LCRC cross-task preparation round.

Usage
-----
    uv run python tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/audit_cpu.py \
        --dataset lipo --scope official-train-only --device cpu

The script builds the explicit Lipophilicity feature spec and structural
caches from the frozen Phase 1a Fit/Monitor/Reserved row lists, runs the
CPU correctness gates, writes ``analysis/*.json`` and (unless disabled)
replays the frozen ZINC LCRC-A checkpoint against the legacy adapter.

It never trains, never creates an optimizer, never reads a label and never
touches the official valid/test labels.  All outputs are evidence about the
*interface*, not about downstream predictive performance.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import adapters as A
import cpu_gates as G
import lcrc_common as C
import lcrc_data as LD
import lcrc_model as LM
from feature_spec import FeatureSpec, legacy_zinc_spec, lipo_v1_spec


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _field_slice(fields, name: str) -> slice:
    start = 0
    for f in fields:
        if f.name == name:
            return slice(start, start + f.dim)
        start += f.dim
    raise KeyError(name)


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return C.track_rel(obj)
    return obj


def _save_json(name: str, obj: Any) -> Path:
    path = C.ANALYSIS / name
    C.write_json(path, _as_jsonable(obj))
    return path


def _scan_fit_vocabulary(raw_rows: Sequence[Mapping[str, Any]], split_rows: Sequence[int]) -> dict[str, Any]:
    atoms: set[int] = set()
    bonds: set[int] = set()
    mask_seen = 0
    for i in split_rows:
        row = raw_rows[int(i)]
        for a in row["atom_class"]:
            a = int(a)
            if a == 16:
                mask_seen += 1
            else:
                atoms.add(a)
        for b in row["edge_class"]:
            bonds.add(int(b))
    return {
        "fit_observed_atom_classes": sorted(atoms),
        "fit_observed_bond_classes": sorted(bonds),
        "fit_mask_sentinel_occurrences": int(mask_seen),
        "source": "Fit row structures only; no labels",
    }


def _build_graphs(
    raw_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[int],
    spec: FeatureSpec,
    *,
    label: str,
) -> list[Any]:
    t0 = time.perf_counter()
    graphs = []
    for i, row_index in enumerate(split_rows):
        graphs.append(A.lipo_row_to_graph(raw_rows[int(row_index)], int(row_index), spec))
        if (i + 1) % 1000 == 0:
            C.log(f"[graphs {label}] {i + 1}/{len(split_rows)} {time.perf_counter() - t0:.1f}s")
    return graphs


def _cache_summary(cache: Mapping[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    report = LD.validate_cache(cache, spec)
    return {
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms": int(len(cache["atom_type"])),
        "n_bonds": int(len(cache["bond_type"])),
        "n_s1_codes": int(len(cache["nbr_code"])),
        "n_s2_paths": int(len(cache["path_cat"])),
        "content_fingerprint": C.content_fingerprint(cache),
        "validate": report,
        "dtypes": {
            key: str(np.asarray(cache[key]).dtype)
            for key in (
                "atom_feat",
                "bond_feat",
                "atom_scalar",
                "global",
                "atom_type",
                "bond_type",
                "nbr_code",
                "path_cat",
                "path_end63",
                "path_end",
                "path_closure",
                "bond_index",
            )
        },
        "meta": dict(cache.get("meta", {})),
    }


def _coverage(cache: Mapping[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    atom_type = np.asarray(cache["atom_type"])
    bond_type = np.asarray(cache["bond_type"])
    atom_scalar = np.asarray(cache["atom_scalar"])
    atom_feat = np.asarray(cache["atom_feat"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    n_mol = int(cache["n_molecules"])

    charge_sl = _field_slice(spec.atom_feature_fields, "formal_charge")
    h_sl = _field_slice(spec.atom_feature_fields, "n_hydrogens")
    charge_block = atom_feat[:, charge_sl]
    h_block = atom_feat[:, h_sl]
    known_charge_nonzero = float(charge_block[:, [0, 1, 3, 4]].sum())
    unknown_charge = float(charge_block[:, 5].sum())
    unknown_h = float(h_block[:, 5].sum())

    aromatic_ids = {int(m["index"]) for m in spec.atom_type_metadata if m.get("aromatic")}
    aromatic_atoms = int(np.isin(atom_type, sorted(aromatic_ids)).sum())
    unknown_atoms = int((atom_type == int(spec.atom_unk_index)).sum()) if spec.atom_unk_index is not None else 0
    unknown_bonds = int((bond_type == int(spec.bond_unk_id)).sum()) if spec.bond_unk_id is not None else 0

    in_ring_idx = list(spec.atom_scalar_fields).index(next(f for f in spec.atom_scalar_fields if f.name == "in_ring"))
    fused_idx = list(spec.atom_scalar_fields).index(next(f for f in spec.atom_scalar_fields if f.name == "is_fused_or_spiro"))
    ring_atoms = int(np.asarray(atom_scalar[:, in_ring_idx] > 0.5).sum())
    fused_atoms = int(np.asarray(atom_scalar[:, fused_idx] > 0.5).sum())

    zero_path = 0
    isolated = 0
    for g in range(n_mol):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        if int(path_ptr[a1] - path_ptr[a0]) == 0:
            zero_path += 1
        isolated += int(np.count_nonzero(np.diff(nbr_ptr[a0 : a1 + 1]) == 0))

    comp = np.asarray(cache["n_components"])
    comp_hist = {str(int(k)): int(v) for k, v in zip(*np.unique(comp, return_counts=True))}
    bond_hist = {str(int(k)): int(v) for k, v in zip(*np.unique(bond_type, return_counts=True))}
    atom_hist = {str(int(k)): int(v) for k, v in zip(*np.unique(atom_type, return_counts=True))}
    path_cat = np.asarray(cache["path_cat"])
    nbr_code = np.asarray(cache["nbr_code"])
    return {
        "n_molecules": n_mol,
        "n_atoms": int(len(atom_type)),
        "n_bonds": int(len(bond_type)),
        "n_s1_codes": int(len(nbr_code)),
        "n_s2_paths": int(len(path_cat)),
        "n_zero_path_molecules": int(zero_path),
        "n_isolated_atoms": int(isolated),
        "n_disconnected_molecules": int((comp > 1).sum()),
        "n_ring_atoms": ring_atoms,
        "n_fused_or_spiro_atoms": fused_atoms,
        "n_aromatic_atoms": aromatic_atoms,
        "n_unknown_atoms": unknown_atoms,
        "n_unknown_bonds": unknown_bonds,
        "n_known_nonzero_charge_atoms": int(round(known_charge_nonzero)),
        "n_unknown_charge_atoms": int(round(unknown_charge)),
        "n_unknown_hydrogen_atoms": int(round(unknown_h)),
        "unknown_atom_fraction": float(unknown_atoms / max(1, len(atom_type))),
        "unknown_bond_fraction": float(unknown_bonds / max(1, len(bond_type))),
        "component_histogram": comp_hist,
        "bond_type_histogram": bond_hist,
        "atom_type_histogram": atom_hist,
        "max_path_cat": int(path_cat.max()) if path_cat.size else -1,
        "max_nbr_code": int(nbr_code.max()) if nbr_code.size else -1,
        "dtype_path_cat": str(path_cat.dtype),
        "dtype_nbr_code": str(nbr_code.dtype),
    }


def _representative_indices(graphs: Sequence[Any], spec: FeatureSpec, n: int = 32, seed: int = 0) -> list[int]:
    if len(graphs) <= n:
        return list(range(len(graphs)))
    n_atoms = np.asarray([g.n_atoms for g in graphs])
    n_bonds = np.asarray([g.n_bonds for g in graphs])
    aromatic = np.asarray([int(np.sum(g.bond_type == 4)) for g in graphs])
    unknown = np.asarray([int(np.sum(g.atom_type == spec.atom_unk_index)) for g in graphs])
    charged = np.asarray(
        [
            int(np.sum(np.isin(g.raw_atom_charge, [0, 1, 3, 4])))
            if g.raw_atom_charge is not None
            else 0
            for g in graphs
        ]
    )
    components = np.asarray([g.n_components for g in graphs])
    cyclomatic = np.asarray([max(0, g.n_bonds - g.n_atoms + g.n_components) for g in graphs])
    picks: list[int] = []
    for arr in (n_atoms, n_bonds, aromatic, unknown, charged, components, cyclomatic):
        picks.append(int(np.argmax(arr)))
    rng = np.random.default_rng(seed)
    picks.extend(int(x) for x in rng.choice(len(graphs), size=min(n, len(graphs)), replace=False))
    unique = sorted(set(int(x) for x in picks))
    if len(unique) > n:
        # Keep the extremal picks and fill with the random sample.
        extremal = sorted(set(int(np.argmax(arr)) for arr in (n_atoms, n_bonds, aromatic, unknown, charged, components, cyclomatic)))
        rest = [x for x in unique if x not in extremal]
        unique = sorted(set(extremal + rest[: max(0, n - len(extremal))]))
    return unique


def _aggregate(gates: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(g.get("status", "FAIL")) for g in gates]
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s == "BLOCKED" for s in statuses):
        return "BLOCKED"
    return "PASS"


def _gate(gate_id: str, name: str, fn) -> dict[str, Any]:
    return G._gate(gate_id, name, fn)


# ---------------------------------------------------------------------------
# Lipophilicity audit
# ---------------------------------------------------------------------------


def audit_lipo(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    C.log("[audit] loading frozen label-free sources")
    raw_payload = A.load_lipo_raw_artifact()
    raw_rows = raw_payload["rows"]
    split = A.load_lipo_frozen_split()
    row_manifest = A.load_lipo_row_manifest()
    source = A.source_manifest()

    C.log("[audit] source gate + row-manifest alignment")
    source_gate = _gate("G1a", "source hashes, split union, scaffold isolation", G.gate_sources)
    alignment_gate = _gate(
        "G1b",
        "Fit/Monitor/Reserved rows align with the label-free row manifest",
        lambda: G.gate_row_manifest_alignment(raw_rows, split["split"], row_manifest),
    )

    C.log("[audit] scanning Fit-only observed vocabulary")
    env = C.env_info()
    vocab_scan = _scan_fit_vocabulary(raw_rows, split["split"]["fit"])
    if vocab_scan["fit_mask_sentinel_occurrences"]:
        raise RuntimeError("raw MASK sentinel 16 occurs in Fit structures; refusing to map it silently")
    provenance = {
        "round_id": C.ROUND_ID,
        "protocol": "ssrc-phase1a-lipo-v1",
        "raw_graph_sha256": source["hashes"]["raw_graph"],
        "frozen_split_sha256": source["hashes"]["frozen_split"],
        "row_manifest_sha256": source["hashes"]["row_manifest"],
        "csv_sha256": source["hashes"]["csv"],
        "official_split_sha256": source["hashes"]["official_split"],
        "row_hashes": source["hashes"]["row_hashes"],
        "vocabulary_policy": "fixed_source_classes_0..15_and_bond_classes_0..4",
        "fit_observed_atom_classes": vocab_scan["fit_observed_atom_classes"],
        "fit_observed_bond_classes": vocab_scan["fit_observed_bond_classes"],
        "rdkit_version": env.get("rdkit"),
        "python_version": env.get("python"),
        "numpy_version": env.get("numpy"),
        "torch_version": env.get("torch"),
    }
    spec = lipo_v1_spec(provenance)
    legacy_spec = legacy_zinc_spec({"round_id": C.ROUND_ID, "source": "frozen LCRC-v1/P3 interface"})
    C.log(f"[audit] lipo_v1 spec hash {spec.hash()}")
    _save_json("feature_spec_lipo.json", spec.to_dict())
    _save_json("feature_spec_legacy_zinc.json", legacy_spec.to_dict())
    source_out = dict(source)
    source_out["row_manifest_alignment"] = alignment_gate["details"]
    source_out["fit_vocabulary_scan"] = vocab_scan
    source_out["feature_spec_lipo_sha256"] = C.sha256_file(C.ANALYSIS / "feature_spec_lipo.json")
    source_out["reader_policy"] = {
        "csv": "sha256 byte stream only; exp column never parsed",
        "official_split": "row-index arrays read by a streaming JSON reader; label tail never parsed",
        "labels_fit_monitor": "never opened",
        "reserved_labels": "never opened",
        "official_valid_test_labels": "never opened",
    }
    _save_json("source_manifest.json", source_out)

    splits = ("fit", "monitor", "reserved")
    graphs: dict[str, list[Any]] = {}
    caches: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    cache_paths: dict[str, Any] = {}
    for split_name in splits:
        rows = split["split"][split_name]
        C.log(f"[audit] building {split_name} graphs ({len(rows)} molecules)")
        graphs[split_name] = _build_graphs(raw_rows, rows, spec, label=split_name)
        C.log(f"[audit] building {split_name} structural cache")
        cache = LD.build_cache(graphs[split_name], spec, split=split_name)
        LD.validate_cache(cache, spec)
        caches[split_name] = cache
        summaries[split_name] = _cache_summary(cache, spec)
        coverage[split_name] = _coverage(cache, spec)
        if args.write_cache:
            path = C.CACHE / f"lipo_{split_name}.pt"
            sha = LD.save_cache(cache, path)
            cache_paths[split_name] = {"path": C.track_rel(path), "sha256": sha, "bytes": path.stat().st_size}
            C.log(f"[audit] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    C.log("[audit] deterministic Fit rebuild")
    t0 = time.perf_counter()
    rebuilt_fit = LD.build_cache(graphs["fit"], spec, split="fit")
    det = {
        "same_content_fingerprint": C.content_fingerprint(rebuilt_fit) == summaries["fit"]["content_fingerprint"],
        "fingerprint": C.content_fingerprint(rebuilt_fit),
        "seconds": float(time.perf_counter() - t0),
    }
    del rebuilt_fit

    C.log("[audit] Fit-only normalisation")
    norm = LD.fit_norm(caches["fit"], spec)
    norm_info = {
        "path": C.track_rel(C.ANALYSIS / "norm_lipo.json"),
        "fingerprint": LD.norm_fingerprint(norm),
        "source": norm["source"],
        "fit_split": norm["fit_split"],
        "valid_or_test_used": norm["valid_or_test_used"],
        "zero_variance_policy": norm["zero_variance_policy"],
        "scalar_constant_indices": norm["scalar_constant_indices"],
        "global_constant_indices": norm["global_constant_indices"],
    }
    C.write_json(C.ANALYSIS / "norm_lipo.json", _as_jsonable(norm))

    C.log("[audit] building schema-driven LCRC-A")
    model, model_info = LM.build_model(spec, norm, seed=args.seed, device="cpu", load_ssl=False)
    model.eval()

    rep_fit = _representative_indices(graphs["fit"], spec, n=32, seed=args.seed)
    rep_monitor = _representative_indices(graphs["monitor"], spec, n=16, seed=args.seed + 1)
    rep_reserved = _representative_indices(graphs["reserved"], spec, n=16, seed=args.seed + 2)
    rep_graphs = [graphs["fit"][i] for i in rep_fit]

    C.log("[audit] running G2/G3/G4/G5/G6/G7 gates")
    feature_gate = _gate("G2a", "feature-spec serialisation, dimensions and dtype", lambda: G.gate_feature_spec(spec))
    overflow_gate = _gate("G4a", "path-code vocabulary and integer overflow", lambda: G.gate_overflow(spec))
    reference_gate = _gate(
        "G3a",
        "independent reference enumeration on representative real molecules",
        lambda: G.gate_reference_enumeration(rep_graphs, spec, label="fit_representative"),
    )
    synthetic = G.synthetic_cases(spec)
    synthetic_gate = _gate(
        "G3b",
        "independent reference enumeration on hand-built edge cases",
        lambda: G.gate_reference_enumeration(list(synthetic.values()), spec, label="synthetic"),
    )
    perm_gate = _gate(
        "G2b",
        "atom-permutation invariance of features/S1/S2/prediction",
        lambda: G.gate_permutation_invariance(rep_graphs[:16], spec, model, seed=args.seed),
    )
    bond_gate = _gate(
        "G2c",
        "bond endpoint swap + list shuffle invariance",
        lambda: G.gate_bond_order_invariance(rep_graphs[:16], spec, model, seed=args.seed + 1),
    )
    path_gate = _gate(
        "G2d",
        "within-root path list shuffle invariance",
        lambda: G.gate_path_order_invariance(rep_graphs[:16], spec, model, seed=args.seed + 2),
    )
    numeric_fit = _gate("G4b", "Fit cache numeric validity", lambda: G.gate_numeric_validity(caches["fit"], spec))
    numeric_monitor = _gate("G4c", "Monitor cache numeric validity", lambda: G.gate_numeric_validity(caches["monitor"], spec))
    numeric_reserved = _gate("G4d", "Reserved cache numeric validity", lambda: G.gate_numeric_validity(caches["reserved"], spec))
    batch_gate = _gate(
        "G5a",
        "single-vs-batch invariance and index isolation",
        lambda: G.gate_batch_invariance(
            caches["fit"], spec, model, rows=rep_fit[:24], seed=args.seed
        ),
    )
    edge_gate = _gate(
        "G6a",
        "single atom, small paths, rings, charge, disconnected and UNK cases",
        lambda: G.gate_edge_cases(spec, model),
    )
    forward_fit = _gate(
        "G7a",
        "LCRC-A forward and sum(f(h_v))+rho decomposition on Fit",
        lambda: G.gate_model_forward(caches["fit"], spec, norm, seed=args.seed, rows=rep_fit),
    )
    forward_monitor = _gate(
        "G7b",
        "LCRC-A forward and decomposition on Monitor",
        lambda: G.gate_model_forward(caches["monitor"], spec, norm, seed=args.seed, rows=rep_monitor),
    )
    forward_reserved = _gate(
        "G7c",
        "LCRC-A forward and decomposition on Reserved",
        lambda: G.gate_model_forward(caches["reserved"], spec, norm, seed=args.seed, rows=rep_reserved),
    )
    label_gate = _gate(
        "G7d",
        "label isolation and no training path",
        lambda: G.gate_label_independence(caches["fit"], spec, rows=rep_fit[:16]),
    )

    gates_all = {
        "G1_source_and_split_integrity": {
            "status": _aggregate([source_gate, alignment_gate]),
            "components": [source_gate, alignment_gate],
        },
        "G2_feature_semantics_and_structural_invariance": {
            "status": _aggregate([feature_gate, perm_gate, bond_gate, path_gate]),
            "components": [feature_gate, perm_gate, bond_gate, path_gate],
        },
        "G3_S1_S2_independent_enumeration": {
            "status": _aggregate([reference_gate, synthetic_gate]),
            "components": [reference_gate, synthetic_gate],
        },
        "G4_numeric_validity_and_overflow": {
            "status": _aggregate([overflow_gate, numeric_fit, numeric_monitor, numeric_reserved]),
            "components": [overflow_gate, numeric_fit, numeric_monitor, numeric_reserved],
        },
        "G5_batch_and_collate_isolation": {
            "status": _aggregate([batch_gate]),
            "components": [batch_gate],
        },
        "G6_edge_cases": {
            "status": _aggregate([edge_gate]),
            "components": [edge_gate],
        },
        "G7_forward_and_decomposition": {
            "status": _aggregate([forward_fit, forward_monitor, forward_reserved, label_gate]),
            "components": [forward_fit, forward_monitor, forward_reserved, label_gate],
        },
    }
    gate_status = _aggregate([v for v in gates_all.values()])
    if not det["same_content_fingerprint"]:
        gate_status = "FAIL"
        gates_all["G0_determinism"] = {
            "status": "FAIL",
            "components": [{"gate": "G0", "name": "repeat Fit cache build", "status": "FAIL", "details": det}],
        }
    else:
        gates_all["G0_determinism"] = {
            "status": "PASS",
            "components": [{"gate": "G0", "name": "repeat Fit cache build", "status": "PASS", "details": det}],
        }

    final = "READY" if gate_status == "PASS" else ("BLOCKED" if gate_status == "BLOCKED" else "NOT_READY")
    result = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "scope": {
            "dataset": "lipophilicity",
            "scope": args.scope,
            "device": "cpu",
            "training": False,
            "optimizer_steps": 0,
            "labels_read": False,
            "official_valid_test_labels_read": False,
            "official_valid_test_structures_converted": False,
            "raw_artifact_rows_loaded": len(raw_rows),
            "in_scope_rows_processed": int(sum(len(split["split"][k]) for k in splits)),
            "gpu_used": False,
            "frozen_zinc_artifacts_modified": False,
        },
        "environment": C.env_info(),
        "source_hashes": source["hashes"],
        "source_manifest": {
            "path": C.track_rel(C.ANALYSIS / "source_manifest.json"),
            "sha256": C.sha256_file(C.ANALYSIS / "source_manifest.json"),
        },
        "label_isolation": source["label_isolation"],
        "vocabulary_scan": vocab_scan,
        "feature_spec": {
            "name": spec.name,
            "path": C.track_rel(C.ANALYSIS / "feature_spec_lipo.json"),
            "sha256": C.sha256_file(C.ANALYSIS / "feature_spec_lipo.json"),
            "hash": spec.hash(),
            "dimensions": spec.to_dict()["dimensions"],
            "n_atom_types": spec.n_atom_types,
            "n_bond_types": spec.n_bond_types,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
            "structural_code_dtype": spec.structural_code_dtype,
            "supports_disconnected": spec.supports_disconnected,
        },
        "splits": {
            "counts": {k: len(split["split"][k]) for k in splits},
            "row_hashes": split["hashes"],
            "summaries": summaries,
            "coverage": coverage,
            "cache_files": cache_paths,
        },
        "normalisation": norm_info,
        "model": {
            **model_info,
            "role": "random-initialised CPU interface probe; not the frozen ZINC checkpoint and not trained on Lipophilicity",
        },
        "gates": gates_all,
        "gate_status": gate_status,
        "final_status": final,
        "timing": {
            "seconds": float(time.perf_counter() - t_start),
            "peak_rss_mb": C.peak_rss_mb(),
        },
        "limitations": [
            "No training, fine-tuning, hyper-parameter search or performance comparison was performed.",
            "The Lipophilicity model is randomly initialised; no claim is made that LCRC-A predicts lipophilicity.",
            "The frozen ZINC checkpoint is used only for a CPU compatibility replay under the legacy_zinc schema.",
            "Only Fit/Monitor/Reserved structures (the official train 3360 rows) are processed; official valid/test labels are never read.",
        ],
    }
    _save_json("cpu_audit.json", result)
    return result


# ---------------------------------------------------------------------------
# ZINC legacy replay
# ---------------------------------------------------------------------------


def audit_zinc(args: argparse.Namespace) -> dict[str, Any]:
    import zinc_legacy as Z

    legacy_spec = legacy_zinc_spec({"round_id": C.ROUND_ID, "source": "frozen LCRC-v1/P3 interface"})
    max_molecules = args.zinc_max if not args.quick else min(args.zinc_max, 256)
    collate_n = min(256, max_molecules) if not args.quick else 32
    replay_n = min(256, max_molecules) if not args.quick else 32
    C.log(f"[zinc] replaying up to {max_molecules} P3 objects; checkpoint n={replay_n}")
    result = Z.run_compatibility(
        legacy_spec,
        max_molecules=max_molecules,
        collate_n=collate_n,
        replay_n=replay_n,
    )
    result["scope"]["quick"] = bool(args.quick)
    result["feature_spec"] = {
        "path": C.track_rel(C.ANALYSIS / "feature_spec_legacy_zinc.json"),
        "hash": legacy_spec.hash(),
        "dimensions": legacy_spec.to_dict()["dimensions"],
        "n_atom_types": legacy_spec.n_atom_types,
        "n_bond_types": legacy_spec.n_bond_types,
        "n_nbr_codes": legacy_spec.n_nbr_codes,
        "n_path_codes": legacy_spec.n_path_codes,
    }
    result["final_status"] = result["status"]
    _save_json("zinc_compatibility.json", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("lipo", "zinc-legacy", "all"), default="lipo")
    p.add_argument("--scope", default="official-train-only")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="small ZINC replay sample; lipo still uses all Fit rows")
    p.add_argument("--skip-zinc", action="store_true", help="do not run the ZINC legacy replay")
    p.add_argument("--zinc-max", type=int, default=10000)
    p.add_argument("--no-write-cache", dest="write_cache", action="store_false", default=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device != "cpu":
        raise SystemExit("this round is CPU-only; --device must be cpu")
    if args.scope != "official-train-only":
        raise SystemExit("only --scope official-train-only is permitted")
    if not torch.cuda.is_available():
        C.log("[audit] CUDA not available; running CPU-only as required")
    out: dict[str, Any] = {}
    if args.dataset in ("lipo", "all"):
        out["lipo"] = audit_lipo(args)
    if args.dataset == "zinc-legacy" or (args.dataset in ("lipo", "all") and not args.skip_zinc):
        out["zinc"] = audit_zinc(args)
    C.log("[audit] done")
    for key, value in out.items():
        C.log(f"[audit] {key}: {value.get('final_status', value.get('status'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
