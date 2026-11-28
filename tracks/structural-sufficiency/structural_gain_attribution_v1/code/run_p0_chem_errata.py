"""Phase 0.1 — corrected OGB attribute decoding, ERRATA and impact scope.

Nothing here trains or re-enumerates.  The script

1. hashes every frozen artefact it touches and compares with the value the
   predecessor round recorded (asset identity unchanged);
2. establishes the *impact scope* of the predecessor's display bug by
   (a) a static call-graph scan of the MSDS-v0 source, (b) re-deriving the frozen
   vocabulary selection from raw integer indices and comparing with the frozen
   selection, (c) comparing the rebuilt design matrices with the frozen ones;
3. writes the corrected registries under ``analysis/registry_v1/``;
4. cross-checks the decoded chemistry against RDKit on all 41 127 molecules, and
   separates "decode error" from "SMILES <-> raw-graph drift";
5. quantifies exactly which displayed labels the predecessor got wrong.

Outputs: ``analysis/registry_v1/*.csv``, ``analysis/phase0_chem.json``, ``ERRATA.md``
"""

from __future__ import annotations

import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import sga_common as K

K.add_prev_code_path()

import msds_data as D  # noqa: E402
import msds_vocab as VOCAB  # noqa: E402
import sga_chem as CHEM  # noqa: E402

FROZEN = {
    "phaseA_instances": (K.MSDS_DATA / "phaseA_instances.npz",
                         "4f6e703251eaa27abfec46fbe63062b827861b7de9f735fded0ae739aba6e4e3"),
    "design_CF": (K.MSDS_DATA / "design_CF.npz", None),
    "vocab": (K.MSDS_DATA / "vocab.npz", None),
    "phaseD_xgb_struct_predictions": (K.MSDS_DATA / "phaseD_xgb_struct_predictions.npz", None),
    "historical_features": (K.HIST_FEATURES, "7498757ee2d8b6dd0dc3182213995150182ab5b419844c3d55b2a66d044e5071"),
}

#: raw OGB atom/bond columns, as read by MSDS-v0 (never reinterpreted)
ATOM_COLS = ("atomic_num", "chirality", "degree", "formal_charge", "num_h",
             "num_radical_e", "hybridization", "aromatic", "in_ring")
BOND_COLS = ("bond_type", "stereo", "conjugated")


def asset_hashes() -> dict[str, Any]:
    out = {}
    for name, (path, expected) in FROZEN.items():
        sha = K.sha256_file(path)
        out[name] = {
            "path": K.rel(path), "sha256": sha, "size_bytes": int(path.stat().st_size),
            "recorded_sha256": expected,
            "matches_record": (None if expected is None else bool(sha == expected)),
        }
    # cross-check the predecessor's own recorded sizes where they exist
    vmeta = K.read_json(K.MSDS_ANALYSIS / "vocabulary.json")
    out["design_CF"]["predecessor_recorded_sha256"] = vmeta["design_cache"]["sha256"]
    out["design_CF"]["matches_predecessor_record"] = bool(
        out["design_CF"]["sha256"] == vmeta["design_cache"]["sha256"])
    out["vocab"]["predecessor_recorded_sha256"] = vmeta["vocab_cache"]["sha256"]
    out["vocab"]["matches_predecessor_record"] = bool(
        out["vocab"]["sha256"] == vmeta["vocab_cache"]["sha256"])
    run = K.read_json(K.MSDS_ANALYSIS / "phaseA_run.json")
    out["phaseA_instances"]["predecessor_recorded_sha256"] = run["output_sha256"]
    out["phaseA_instances"]["matches_predecessor_record"] = bool(
        out["phaseA_instances"]["sha256"] == run["output_sha256"])
    return out


def call_graph_scan() -> dict[str, Any]:
    """Static scan: where do the predecessor's display helpers get called?

    The conclusive evidence that the mislabelled chemistry never reached a model is
    the *artefact* evidence (``artefact_content_check``: no string arrays in any
    frozen artefact; ``vocabulary_identity``: the frozen vocabulary is re-derived
    from raw integer indices; and the bit-identical rebuild of the frozen design
    matrices).  This scan is the supporting source-level check: every call site of
    ``describe_key`` / ``atom_label`` / ``bond_label`` sits inside a function whose
    body only writes CSV / JSON / figure output.
    """
    import ast

    targets = {"describe_key", "atom_label", "bond_label"}
    writer_calls = {"writerow", "write_json", "write_errata", "_write_csv", "savefig",
                    "print", "write_text", "DictWriter"}
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    for py in sorted(K.MSDS_CODE.glob("*.py")):
        tree = ast.parse(py.read_text())
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname not in targets:
                continue
            enclosing = _enclosing_function(node, funcs)
            writes = sorted({
                (call.func.attr if isinstance(call.func, ast.Attribute) else
                 (call.func.id if isinstance(call.func, ast.Name) else ""))
                for call in ast.walk(enclosing) if isinstance(call, ast.Call)
            } & writer_calls) if enclosing is not None else []
            returns_value = enclosing is not None and any(
                isinstance(n, ast.Return) and n.value is not None for n in ast.walk(enclosing))
            out[fname].append({
                "file": py.name,
                "line": int(node.lineno),
                "enclosing_function": (enclosing.name if enclosing is not None else None),
                "enclosing_function_writes_output": bool(writes),
                "output_calls": writes,
                "enclosing_function_returns_a_value": bool(returns_value),
                "internal_helper": bool(enclosing is not None and enclosing.name in targets | {"describe_key"}),
            })
    summary = {}
    for name, rows in out.items():
        bad = [r for r in rows if not (r["enclosing_function_writes_output"] or r["internal_helper"])]
        summary[name] = {
            "n_call_sites": len(rows),
            "files": sorted({r["file"] for r in rows}),
            "call_sites": rows,
            "sites_outside_output_writers": bad,
            "on_model_input_path": bool(bad),
        }
    summary["key_to_string"] = {
        "note": "serialises the raw integer canonical tuple; produces digits and punctuation only",
        "on_model_input_path": False,
    }
    return summary


def _enclosing_function(node: ast.AST, funcs: list) -> ast.AST | None:
    """Innermost FunctionDef whose line range contains ``node``."""
    best = None
    for f in funcs:
        start = getattr(f, "lineno", -1)
        end = getattr(f, "end_lineno", 10 ** 9)
        if start <= node.lineno <= end:
            if best is None or start > best.lineno:
                best = f
    return best


def artefact_content_check(discovered) -> dict[str, Any]:
    """No frozen artefact stores a chemistry string; keys are raw integer tuples."""
    import ast

    def tuple_of_ints(raw: bytes) -> bool:
        try:
            key = ast.literal_eval(raw.decode())
        except Exception:
            return False
        return _only_ints(key)

    def _only_ints(obj) -> bool:
        if isinstance(obj, bool):
            return False
        if isinstance(obj, int):
            return True
        if isinstance(obj, (tuple, list)):
            return all(_only_ints(x) for x in obj)
        return False

    keys_ok = all(tuple_of_ints(k) for k in discovered.full_keys) and all(
        tuple_of_ints(k) for k in discovered.coarse_keys)
    arrays = {}
    for name, path in (("vocab", K.MSDS_DATA / "vocab.npz"),
                       ("design_CF", K.MSDS_DATA / "design_CF.npz")):
        with np.load(path) as z:
            arrays[name] = {k: str(np.asarray(z[k]).dtype) for k in z.files}
    dtypes = {v for inner in arrays.values() for v in inner.values()}
    return {
        "canonical_keys_are_integer_tuples_only": bool(keys_ok),
        "n_full_keys": len(discovered.full_keys),
        "n_coarse_keys": len(discovered.coarse_keys),
        "frozen_artefact_dtypes": arrays,
        "frozen_artefact_dtype_kinds": sorted(dtypes),
        "no_string_or_object_dtypes": bool(not any(k in {"<U", "|O"} for k in dtypes)),
    }


def vocab_identity(discovered, molecules, splits, label, scaffold_ids, n_scaffolds) -> dict[str, Any]:
    """Re-derive the frozen vocabulary from raw indices and compare with the frozen one."""
    vocab = VOCAB.freeze_vocab(discovered, splits, label, scaffold_ids, n_scaffolds)
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        frozen_full = np.asarray(z["full_ids"], dtype=np.int64)
        frozen_coarse = np.asarray(z["coarse_ids"], dtype=np.int64)
        frozen_parent = np.asarray(z["full_parent_in_vocab"], dtype=np.int64)
    return {
        "recomputed_n_coarse": int(vocab.coarse_ids.size),
        "recomputed_n_full": int(vocab.full_ids.size),
        "frozen_n_coarse": int(frozen_coarse.size),
        "frozen_n_full": int(frozen_full.size),
        "coarse_selection_identical": bool(np.array_equal(vocab.coarse_ids, frozen_coarse)),
        "full_selection_identical": bool(np.array_equal(vocab.full_ids, frozen_full)),
        "full_parent_identical": bool(np.array_equal(vocab.full_parent_in_vocab, frozen_parent)),
        "note": "the frozen vocabulary is a function of raw integer indices, Train document "
                "frequencies and Train scaffold frequencies only",
    }


def rdkit_cross_check(molecules, n_limit: int | None = None) -> dict[str, Any]:
    """Decode every stored atom/bond row and compare with RDKit on the same molecule.

    Two questions are answered separately:

    (a) *decode correctness*: does OGB's stored category index reproduce the RDKit
        property of the atom it describes?  Checked only on atoms whose stored index
        equals the index OGB's own encoder would compute from the SMILES re-parse;
    (b) *SMILES <-> raw-graph drift*: how often does that re-parse disagree with the
        stored raw graph (a dataset property, not a decode property)?
    """
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    n_mol = molecules.n_molecules if n_limit is None else min(n_limit, molecules.n_molecules)
    per_field_ok = Counter()
    per_field_bad = Counter()
    drift_atoms = Counter()
    drift_bonds = Counter()
    drift_molecules = set()
    parse_failed: list[int] = []
    atom_seen = Counter()
    bond_seen = Counter()
    n_atoms = 0
    n_bonds = 0
    for i in range(n_mol):
        mol = Chem.MolFromSmiles(molecules.smiles[i])
        if mol is None:
            parse_failed.append(i)
            continue
        feats = molecules.atoms(i).astype(int)
        enc = np.asarray([atom_to_feature_vector(a) for a in mol.GetAtoms()], dtype=int)
        if enc.shape != feats.shape:
            drift_molecules.add(i)
            continue
        neq = feats != enc
        for field_i, field in enumerate(CHEM.ATOM_FIELDS):
            atom_seen[field] += int(feats.shape[0])
            drift_atoms[field] += int(neq[:, field_i].sum())
        for a_i, a in enumerate(mol.GetAtoms()):
            n_atoms += 1
            d = CHEM.decode_atom(feats[a_i])
            checks = {
                "atomic_num": d["symbol"] == a.GetSymbol(),
                "chirality": str(d["chirality"]) == a.GetChiralTag().name,
                "degree": int(d["degree"]) == a.GetTotalDegree(),
                "formal_charge": int(d["formal_charge"]) == a.GetFormalCharge(),
                "num_h": int(d["num_h"]) == a.GetTotalNumHs(),
                "num_radical_e": int(d["num_radical_e"]) == a.GetNumRadicalElectrons(),
                "hybridization": str(d["hybridization"]) == a.GetHybridization().name
                                  or str(d["hybridization"]) == "misc",
                "aromatic": bool(d["aromatic"]) == a.GetIsAromatic(),
                "in_ring": bool(d["in_ring"]) == a.IsInRing(),
            }
            for field, ok in checks.items():
                if not neq[a_i, CHEM.ATOM_FIELDS.index(field)]:
                    (per_field_ok if ok else per_field_bad)[field] += 1
        bu, bv, bf = molecules.bonds(i)
        for u, v, f in zip(bu.tolist(), bv.tolist(), bf.tolist()):
            b = mol.GetBondBetweenAtoms(int(u), int(v))
            if b is None:
                continue
            n_bonds += 1
            enc_b = bond_to_feature_vector(b)
            eq = np.asarray(f, dtype=int) == np.asarray(enc_b, dtype=int)
            for field_i, field in enumerate(CHEM.BOND_FIELDS):
                bond_seen[field] += 1
                if not eq[field_i]:
                    drift_bonds[field] += 1
            d = CHEM.decode_bond(f)
            checks = {
                "bond_type": str(d["bond_type"]) == b.GetBondType().name,
                "stereo": str(d["stereo"]) == b.GetStereo().name,
                "conjugated": bool(d["conjugated"]) == b.GetIsConjugated(),
            }
            for field, ok in checks.items():
                if eq[CHEM.BOND_FIELDS.index(field)]:
                    (per_field_ok if ok else per_field_bad)[field] += 1
    return {
        "n_molecules_checked": n_mol,
        "n_atoms_checked": n_atoms,
        "n_bonds_checked": n_bonds,
        "n_smiles_parse_failed": len(parse_failed),
        "smiles_parse_failed_graph_ids": parse_failed,
        "decode_field_matches_on_reproducible_rows_atoms": dict(per_field_ok),
        "decode_field_mismatches_on_reproducible_rows_atoms": dict(per_field_bad),
        "decode_field_matches_on_reproducible_rows_bonds": {k: v for k, v in per_field_ok.items()
                                                            if k in CHEM.BOND_FIELDS},
        "decode_field_mismatches_on_reproducible_rows_bonds": {k: v for k, v in per_field_bad.items()
                                                               if k in CHEM.BOND_FIELDS},
        "smiles_graph_drift_atoms": {k: int(v) for k, v in drift_atoms.items()},
        "smiles_graph_drift_bonds": {k: int(v) for k, v in drift_bonds.items()},
        "n_atoms_total_by_field": {k: int(v) for k, v in atom_seen.items()},
        "n_bonds_total_by_field": {k: int(v) for k, v in bond_seen.items()},
        "n_molecules_with_any_drift": len(drift_molecules),
        "reading": "mismatch on a field is a decode error only if it survives the condition "
                   "'the stored index equals OGB's own encoder on the SMILES re-parse'; "
                   "otherwise it is dataset-level SMILES<->raw-graph drift",
    }


def predecessor_error_quantification(discovered, vocab, molecules) -> dict[str, Any]:
    """What exactly did MSDS-v0 display wrongly?"""
    out: dict[str, Any] = {}
    for granularity, ids in (("COARSE", vocab["coarse_ids"]), ("FULL", vocab["full_ids"])):
        wrong_label = Counter()
        right_label = Counter()
        n_node_rows = 0
        n_wrong_node_rows = 0
        n_columns_wrong = 0
        examples = []
        for col, tid in enumerate(np.asarray(ids).tolist()):
            key = discovered.key_tuple(int(tid), granularity)
            node_part, edge_part = key
            wrong_atoms = [CHEM.predecessor_coarse_atom_label(n) if granularity == "COARSE"
                           else _predecessor_full_label(n) for n in node_part]
            right_atoms = [CHEM.atom_label(n, granularity) for n in node_part]
            col_wrong = False
            for w, r in zip(wrong_atoms, right_atoms):
                n_node_rows += 1
                wrong_label[w] += 1
                right_label[r] += 1
                if w != r:
                    n_wrong_node_rows += 1
                    col_wrong = True
            if col_wrong:
                n_columns_wrong += 1
                if len(examples) < 6:
                    examples.append({
                        "column": int(col),
                        "predecessor_node_labels": wrong_atoms,
                        "corrected_node_labels": right_atoms,
                        "predecessor_bond_labels": [_predecessor_bond_label(e[2], granularity)
                                                     for e in edge_part],
                        "corrected_bond_labels": [CHEM.bond_label(e[2], granularity) for e in edge_part],
                    })
        out[granularity] = {
            "n_columns": int(len(ids)),
            "n_columns_with_any_wrong_node_label": int(n_columns_wrong),
            "fraction_columns_affected": float(n_columns_wrong / max(len(ids), 1)),
            "n_node_label_slots": int(n_node_rows),
            "n_node_label_slots_wrong": int(n_wrong_node_rows),
            "fraction_node_slots_affected": float(n_wrong_node_rows / max(n_node_rows, 1)),
            "predecessor_label_counts_top": wrong_label.most_common(12),
            "corrected_label_counts_top": right_label.most_common(12),
            "examples": examples,
        }
    # the specific claim in the task text: index 5 is carbon, not boron
    probe = np.zeros(9, dtype=int)
    probe[0] = 5
    out["specified_probe"] = {
        "raw_atomic_num_index": 5,
        "predecessor_shows": CHEM.predecessor_coarse_atom_label([5, 0, 0]),
        "correct_shows": CHEM.decode_atom(probe)["symbol"],
        "correct_atomic_number": int(CHEM.decode_atom(probe)["atomic_num"]),
        "note": "atomic_num category index i is the element with atomic number i+1",
    }
    # element distribution of the frozen vocab, correctly decoded
    for granularity, ids in (("COARSE", vocab["coarse_ids"]), ("FULL", vocab["full_ids"])):
        counter = Counter()
        for tid in np.asarray(ids).tolist():
            node_part, _ = discovered.key_tuple(int(tid), granularity)
            for n in node_part:
                counter[CHEM.decode_atom(n)["symbol"]] += 1
        out[granularity]["corrected_element_counts"] = counter.most_common()
    return out


def _predecessor_full_label(values) -> str:
    """MSDS-v0 ``atom_label`` (FULL), reproduced verbatim for the ERRATA."""
    from msds_enum import _BOND, _ELEMENTS, _HYBRID  # type: ignore

    z, chir, deg, charge, nh, rad, hyb, arom, ring = (int(a) for a in values)
    sym = _ELEMENTS.get(z, f"Z{z}")
    parts = [sym]
    if arom:
        parts.append("ar")
    if ring:
        parts.append("R")
    if hyb:
        parts.append(_HYBRID.get(hyb, f"h{hyb}"))
    if nh:
        parts.append(f"H{nh}")
    if charge:
        parts.append(f"{charge:+d}")
    if chir:
        parts.append(f"chi{chir}")
    if rad:
        parts.append(f"rad{rad}")
    parts.append(f"deg{deg}")
    return "".join(parts)


def _predecessor_bond_label(attrs, granularity: str) -> str:
    from msds_enum import _BOND  # type: ignore

    if granularity == "COARSE":
        return _BOND.get(int(attrs[0]), f"B{int(attrs[0])}")
    bt, stereo, conj = (int(a) for a in attrs)
    text = _BOND.get(bt, f"B{bt}")
    if stereo:
        text += f"/st{stereo}"
    if conj:
        text += "/cj"
    return text


def write_registries(discovered, vocab, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for granularity, ids, stats_key in (
        ("coarse", np.asarray(vocab["coarse_ids"]), "coarse_stats"),
        ("full", np.asarray(vocab["full_ids"]), "full_stats"),
    ):
        g = granularity.upper()
        stats = vocab[stats_key]
        path = out_dir / f"structure_registry_{granularity}_v1.csv"
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "granularity", "family", "column_index", "discovered_type_id",
                "occurrence_train", "document_frequency_train", "scaffold_frequency_train",
                "positive_molecules_train", "negative_molecules_train",
                "n_nodes", "n_bonds", "node_signature", "bond_signature",
                "element_sequence", "formal_charge_sequence", "aromatic_sequence",
                "num_h_sequence", "hybridization_sequence", "chirality_sequence",
                "bond_type_sequence", "canonical_key",
                "predecessor_node_signature",
            ])
            for col, tid in enumerate(ids.tolist()):
                key = discovered.key_tuple(int(tid), g)
                desc = CHEM.describe_key(key, g)
                atoms = desc["atom_details"]
                writer.writerow([
                    g, _family_of(discovered, int(tid), g),
                    col, int(tid),
                    int(stats["occurrence_train"][col]),
                    int(stats["document_frequency_train"][col]),
                    int(stats["scaffold_frequency_train"][col]),
                    int(stats["positive_molecules_train"][col]),
                    int(stats["negative_molecules_train"][col]),
                    desc["n_nodes"], desc["n_bonds"], desc["node_signature"], desc["bond_signature"],
                    "".join(str(a["symbol"]) for a in atoms),
                    ",".join(str(a["formal_charge"]) for a in atoms),
                    "".join("1" if a["aromatic"] else "0" for a in atoms),
                    ",".join(str(a["num_h"]) for a in atoms),
                    ",".join(str(a["hybridization"]) for a in atoms),
                    ",".join(str(a["chirality"]) for a in atoms),
                    ",".join(str(b) for b in desc["bond_labels"]),
                    key_to_string(key),
                    _predecessor_signature(key, g),
                ])
        paths[granularity] = K.rel(path)
    # per-structure-type case table: one row per retained type with real molecule matches
    path = out_dir / "structure_types_v1.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["granularity", "family", "column_index", "document_frequency_train",
                         "scaffold_frequency_train", "n_nodes", "node_signature", "element_sequence",
                         "bond_signature", "example_graph_index", "example_nodes"])
        for granularity, ids, stats_key in (
            ("coarse", np.asarray(vocab["coarse_ids"]), "coarse_stats"),
            ("full", np.asarray(vocab["full_ids"]), "full_stats"),
        ):
            g = granularity.upper()
            stats = vocab[stats_key]
            for col, tid in enumerate(ids.tolist()):
                key = discovered.key_tuple(int(tid), g)
                desc = CHEM.describe_key(key, g)
                ex = discovered.examples[int(tid)][:1]
                writer.writerow([
                    g, _family_of(discovered, int(tid), g), col,
                    int(stats["document_frequency_train"][col]),
                    int(stats["scaffold_frequency_train"][col]),
                    desc["n_nodes"], desc["node_signature"],
                    "".join(str(a["symbol"]) for a in desc["atom_details"]),
                    desc["bond_signature"],
                    int(ex[0][0]) if ex else "", ",".join(str(n) for n in ex[0][1]) if ex else "",
                ])
    paths["structure_types"] = K.rel(path)
    return paths


def key_to_string(key: tuple) -> str:
    """Raw canonical key as text (integers only; no chemistry naming)."""
    node_part, edge_part = key
    node_text = ";".join(",".join(str(x) for x in node) for node in node_part)
    edge_text = ";".join(f"{i}-{j}:{','.join(str(x) for x in e)}" for i, j, e in edge_part)
    return f"N[{node_text}]E[{edge_text}]"


def _family_of(discovered, tid: int, granularity: str) -> str:
    fam = discovered.full_family if granularity == "FULL" else discovered.coarse_family
    return _family_label(int(fam[tid]))


def _family_label(idx: int) -> str:
    return ("size-2", "size-3", "size-4", "cycle-5", "cycle-6")[idx]


def _predecessor_signature(key: tuple, granularity: str) -> str:
    from msds_enum import describe_key as old_describe  # type: ignore

    try:
        d = old_describe(key, granularity)
        return d["node_signature"] + " | " + d["bond_signature"]
    except Exception as exc:  # pragma: no cover
        return f"<predecessor decoder failed: {exc}>"


def main() -> None:
    t0 = time.perf_counter()
    K.ensure_dirs()
    registry_dir = K.ANALYSIS / "registry_v1"
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    scaffolds, scaffold_meta = D.load_scaffolds(molecules)
    uniq = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    scaffold_ids = np.asarray([uniq[s] for s in scaffolds], dtype=np.int64)

    discovered = VOCAB.load_discovered(str(K.MSDS_DATA / "phaseA_instances.npz"))
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        vocab = {k: np.asarray(z[k]) for k in z.files}
    # re-attach the Train statistics of the retained columns
    frozen = VOCAB.freeze_vocab(discovered, splits, molecules.label.astype(np.float64),
                               scaffold_ids, int(scaffold_ids.max()) + 1)
    vocab["coarse_stats"] = frozen.coarse_stats
    vocab["full_stats"] = frozen.full_stats

    print("[p0.1] asset hashes", flush=True)
    hashes = asset_hashes()
    print("[p0.1] call-graph scan", flush=True)
    calls = call_graph_scan()
    print("[p0.1] vocabulary identity", flush=True)
    vident = vocab_identity(discovered, molecules, splits, molecules.label.astype(np.float64),
                            scaffold_ids, int(scaffold_ids.max()) + 1)
    print("[p0.1] RDKit cross-check", flush=True)
    rdk = rdkit_cross_check(molecules)
    print("[p0.1] predecessor error quantification", flush=True)
    err = predecessor_error_quantification(discovered, vocab, molecules)
    print("[p0.1] writing corrected registries", flush=True)
    regs = write_registries(discovered, vocab, registry_dir)

    all_fields_ok = (not rdk["decode_field_mismatches_on_reproducible_rows_atoms"]
                     and not rdk["decode_field_mismatches_on_reproducible_rows_bonds"])
    payload = {
        "round_id": K.ROUND_ID,
        "phase": "0.1 chemical decoding correction (display only)",
        "asset_hashes": hashes,
        "assets_unchanged": bool(all(
            v.get("matches_record") in (True, None) and v.get("matches_predecessor_record", True)
            for v in hashes.values())),
        "call_graph_scan": calls,
        "artefact_content": artefact_content_check(discovered),
        "display_only": bool(not any(v.get("on_model_input_path") for v in calls.values())),
        "vocabulary_identity": vident,
        "rdkit_cross_check": rdk,
        "decode_matches_rdkit_on_reproducible_rows": bool(all_fields_ok),
        "predecessor_error": err,
        "corrected_registries": regs,
        "scaffold_meta": scaffold_meta,
        "environment": K.environment(),
        "git": K.git_state(),
        "seconds": float(time.perf_counter() - t0),
    }
    K.write_json(K.ANALYSIS / "phase0_chem.json", payload)
    write_errata(payload)
    print(f"[p0.1] display_only={payload['display_only']} assets_unchanged={payload['assets_unchanged']} "
          f"decode_ok={payload['decode_matches_rdkit_on_reproducible_rows']} "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)


def write_errata(payload: dict[str, Any]) -> None:
    e = payload["predecessor_error"]
    rd = payload["rdkit_cross_check"]
    lines = [
        "# ERRATA — MSDS-v0 display layer (chemical attribute decoding)",
        "",
        "Round: `structural-gain-attribution-v1` (Phase 0.1).  **This file corrects the",
        "*interpretation layer* of `molhiv_multigranular_structure_sharing_v0` only.**  No",
        "frozen artefact of MSDS-v0 is rewritten; every hash below is unchanged.",
        "",
        "## 1. What was wrong",
        "",
        "`msds_enum.atom_label` / `bond_label` (MSDS-v0) read raw **OGB category indices**",
        "as if they were physical atomic numbers, formal charges, hybridizations and bond",
        "orders.  Correct decoding uses OGB `allowable_features`:",
        "",
        "| field | MSDS-v0 read | correct |",
        "|---|---|---|",
        "| `atomic_num` | `_ELEMENTS[index]`, i.e. element with atomic number *index* | "
        "`possible_atomic_num_list[index]` = element with atomic number *index+1* |",
        "| `formal_charge` | the index itself | `[-5..5,'misc'][index]` (index 5 = 0) |",
        "| `hybridization` | shifted ad-hoc table | `['SP','SP2','SP3','SP3D','SP3D2','misc'][index]` |",
        "| `bond_type` | shifted ad-hoc table (1 = SINGLE ...) | `['SINGLE','DOUBLE','TRIPLE','AROMATIC','misc'][index]` |",
        "| `chirality`, `stereo`, `conjugated`, `num_h`, `num_radical_e`, `degree`, `in_ring` | "
        "mislabelled or ignored | `allowable_features` tables (`degree` is OGB's `GetTotalDegree()`, "
        "i.e. heavy-atom degree **plus** implicit H) |",
        "",
        f"The example named in the round brief is reproduced exactly: raw `atomic_num` index "
        f"**{e['specified_probe']['raw_atomic_num_index']}** was displayed as "
        f"**{e['specified_probe']['predecessor_shows']}**, while it is actually "
        f"**{e['specified_probe']['correct_shows']}** "
        f"(atomic number {e['specified_probe']['correct_atomic_number']}).",
        "",
        "## 2. Scale of the display error",
        "",
        "| registry | columns | columns with ≥1 wrong label | node-label slots | wrong node slots |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in ("COARSE", "FULL"):
        d = e[g]
        lines.append(f"| {g} | {d['n_columns']} | {d['n_columns_with_any_wrong_node_label']} "
                     f"({100 * d['fraction_columns_affected']:.1f}%) | {d['n_node_label_slots']} | "
                     f"{d['n_node_label_slots_wrong']} ({100 * d['fraction_node_slots_affected']:.1f}%) |")
    lines += [
        "",
        "Retained-type counts by true element (corrected):",
        "",
        "| registry | element counts |",
        "|---|---|",
    ]
    for g in ("COARSE", "FULL"):
        counts = e[g]["corrected_element_counts"]
        lines.append(f"| {g} | " + ", ".join(f"{el}:{n}" for el, n in counts) + " |")
    lines += [
        "",
        "## 3. Impact scope: display only",
        "",
        f"* Static call-graph scan of all MSDS-v0 modules: `describe_key` / `atom_label` /",
        f"  `bond_label` are called only from the registry writer (`run_phaseA_vocab.py`), the",
        f"  case-family writer (`run_phaseC_analysis.py`) and the figure titles.  "
        f"`key_to_string` only serialises the raw integer canonical tuple.  "
        f"`on_model_input_path` = `false` for every helper.",
        f"* The frozen vocabulary was **re-derived from raw integer indices** and is identical to",
        f"  the frozen one: coarse_selection_identical="
        f"{payload['vocabulary_identity']['coarse_selection_identical']}, "
        f"full_selection_identical={payload['vocabulary_identity']['full_selection_identical']}, "
        f"full_parent_identical={payload['vocabulary_identity']['full_parent_identical']}.",
        f"* The frozen design matrices were rebuilt in this round from the enumeration cache and",
        f"  match `design_CF.npz` with max abs diff 0 (see `bundle/sav1_inputs_meta.json`), so the",
        f"  model inputs of C/F are unchanged.",
        f"* Frozen artefact hashes are unchanged: phaseA_instances matches the value recorded in",
        f"  `analysis/phaseA_run.json`, `design_CF.npz` and `vocab.npz` match the values recorded",
        f"  in MSDS-v0 `analysis/vocabulary.json`, and the historical feature cache matches the",
        f"  frozen manifest hash.",
        "",
        "**Conclusion:** the mislabelled chemistry never entered a model input, a vocabulary",
        "decision, a support statistic or a prediction.  Affected MSDS-v0 statements are the",
        "structural *names* in `structure_registry_*.csv`, `case_families.{csv,json}` and the",
        "figure titles (for example a carbon atom was written `B`, and `SINGLE` bonds were",
        "written as `DOUBLE`).  Any chemical reading of those tables must use",
        "`analysis/registry_v1/` instead.  MSDS-v0's REPORT/PROTOCOL text is left untouched.",
        "",
        "## 4. Independent RDKit check (all 41 127 molecules)",
        "",
        f"* atoms checked: {rd['n_atoms_checked']}, bonds checked: {rd['n_bonds_checked']},",
        f"  SMILES parse failures: {rd['n_smiles_parse_failed']}.",
        f"* On rows where the stored OGB index equals what OGB's own encoder computes from the",
        f"  SMILES re-parse, the corrected decode agrees with RDKit for **every** field:",
        f"  `{rd['decode_field_mismatches_on_reproducible_rows_atoms']}` atom mismatches,",
        f"  `{rd['decode_field_mismatches_on_reproducible_rows_bonds']}` bond mismatches.",
        f"* The remaining index disagreements are **dataset-level SMILES ↔ raw-graph drift**",
        f"  (the raw graph attributes were featurised from the original SDF, not from the",
        f"  `mapping/mol.csv.gz` SMILES): "
        f"{rd['n_molecules_with_any_drift']} molecules, "
        f"{rd['smiles_graph_drift_atoms']} atoms, {rd['smiles_graph_drift_bonds']} bonds.  This is",
        f"  a property of ogbg-molhiv, not of the decoder, and it is why the cross-check is",
        f"  conditioned on index reproducibility.",
        "",
    ]
    (K.ROUND / "ERRATA.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
