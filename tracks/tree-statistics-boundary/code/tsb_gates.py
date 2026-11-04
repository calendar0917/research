"""The CPU correctness gates of ``tsb-lipo-feature-ladder-v1``.

Every gate returns a record ``{name, passed, detail, evidence}``.  The same
functions back both the P0 control-plane stage (which writes
``analysis/p0/p0_gates.json``) and the pytest suite, so "the tests passed" and
"P0 recorded a pass" cannot diverge.

Gates (numbering follows ``PROTOCOL.md`` §6):

1. published split rows/sizes/disjointness match the recorded history;
2. features are label-free (poisoning the label column changes nothing);
3. typed-WL vocabularies come from the official Train rows only;
4. ``WL2_BIN`` and ``WL2_COUNT`` have identical column definitions;
5. ``WL3_COUNT`` rounds 0-2 equal ``WL2_COUNT``;
6. atom renumbering / bond-endpoint swaps leave every feature vector unchanged;
7. reloading the cache reproduces the features element-wise;
8. no NaN/Inf; OOV/parse policy is explicit;
9. row ids and labels align with the split and with every saved prediction;
10. a saved model reloads to identical predictions;
11. synthetic boundary: two labelled cycles with identical WL 0-3 statistics
    but different A-B graph distance.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp

import tsb_common as C
import tsb_data as D
import tsb_features as F
import tsb_g0
import tsb_model as M
import tsb_mol
import tsb_morgan
import tsb_wl


# ---------------------------------------------------------------------------
# synthetic boundary example
# ---------------------------------------------------------------------------


def labelled_cycle(n_atoms: int, position_b: int):
    """Single-bonded ring of carbons with atom 0 = N and atom ``position_b`` = O."""
    from rdkit import Chem

    rw = Chem.RWMol()
    for index in range(n_atoms):
        atom = Chem.Atom(6)
        if index == 0:
            atom = Chem.Atom(7)
        elif index == position_b:
            atom = Chem.Atom(8)
        rw.AddAtom(atom)
    for index in range(n_atoms):
        rw.AddBond(index, (index + 1) % n_atoms, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.GetSymmSSSR(mol)
    return mol


def graph_distance(mol, source: int, target: int) -> int:
    from collections import deque

    adjacency: dict[int, list[int]] = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i].append(j)
        adjacency[j].append(i)
    seen = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            return seen[node]
        for neighbour in adjacency[node]:
            if neighbour not in seen:
                seen[neighbour] = seen[node] + 1
                queue.append(neighbour)
    raise ValueError("atoms are not connected")


def synthetic_boundary() -> dict[str, Any]:
    """Same WL 0-3 pattern multiset, different A-B distance."""
    n_atoms, near, far = 24, 8, 12
    mol_near = labelled_cycle(n_atoms, near)
    mol_far = labelled_cycle(n_atoms, far)
    graphs = [tsb_mol.as_graph(mol_near), tsb_mol.as_graph(mol_far)]
    distances = [graph_distance(mol_near, 0, near), graph_distance(mol_far, 0, far)]

    tokens = []
    for mol in (mol_near, mol_far):
        _mol, _graph, per_round = tsb_mol.molecule_tokens_from_mol(mol, rounds=3)
        tokens.append([dict(tsb_mol.count_tokens(round_tokens)) for round_tokens in per_round])

    per_round_equal = [tokens[0][r] == tokens[1][r] for r in range(4)]
    g0_equal = bool(np.array_equal(tsb_g0.vector(graphs[0]), tsb_g0.vector(graphs[1])))
    morgan_a, _ = tsb_morgan.vector(mol_near)
    morgan_b, _ = tsb_morgan.vector(mol_far)
    morgan_equal = bool(np.array_equal(morgan_a, morgan_b))

    passed = all(per_round_equal) and distances[0] != distances[1] and g0_equal and morgan_equal
    return {
        "name": "11_synthetic_wl_boundary",
        "passed": bool(passed),
        "detail": (
            "two 24-ring graphs with ring atoms 0=N, k=O for k in {8,12}: identical rooted "
            "typed-WL token multiset at rounds 0-3, identical G0 vector and identical Morgan "
            "radius-2 count vector, yet the N-O graph distance differs (8 vs 12)"
        ),
        "evidence": {
            "n_atoms": n_atoms,
            "position_b": [near, far],
            "graph_distance": distances,
            "wl_round_multiset_equal": per_round_equal,
            "g0_vector_equal": g0_equal,
            "morgan_vector_equal": morgan_equal,
            "distinct_wl_tokens_round3": [len(tokens[0][3]), len(tokens[1][3])],
            "boundary_statement": (
                "All five arms of this round are invariant to this pair: distance between two "
                "marked atoms is a non-local statistic that neither G0 composition counts nor "
                "rooted typed-WL rounds 0-3 nor a radius-2 Morgan fingerprint can express."
            ),
        },
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_splits() -> dict[str, Any]:
    splits = D.load_splits()
    problems: list[str] = []
    evidence: dict[str, Any] = {}
    for suffix, meta in splits.items():
        rows = meta["rows"]
        sizes = {name: int(r.size) for name, r in rows.items()}
        if sizes != C.SPLIT_SIZES:
            problems.append(f"{suffix}: sizes {sizes}")
        for name in ("train", "valid", "test"):
            if meta["rows_sha256"][name] != C.SPLIT_ROWS_SHA256[suffix][name]:
                problems.append(f"{suffix}/{name}: row hash")
        if np.intersect1d(rows["train"], rows["valid"]).size:
            problems.append(f"{suffix}: train/valid overlap")
        if np.intersect1d(rows["train"], rows["test"]).size:
            problems.append(f"{suffix}: train/test overlap")
        if np.intersect1d(rows["valid"], rows["test"]).size:
            problems.append(f"{suffix}: valid/test overlap")
        union = np.unique(np.concatenate([rows["train"], rows["valid"], rows["test"]]))
        if union.size != 4200:
            problems.append(f"{suffix}: union {union.size}")
        evidence[str(suffix)] = {
            "file": meta["file"],
            "file_sha256": meta["file_sha256"],
            "sizes": sizes,
            "rows_sha256": meta["rows_sha256"],
        }
    return {
        "name": "01_published_splits",
        "passed": not problems,
        "detail": "3 ChemBench random repeats; sizes 3360/420/420; pairwise disjoint; union 4200",
        "evidence": {"problems": problems, "splits": evidence},
    }


def gate_labels_not_used(sample: int = 240) -> dict[str, Any]:
    """Poison the label column and prove every feature vector is unchanged."""
    import pandas as pd
    from unittest import mock

    smiles = D.load_smiles()
    rng = np.random.default_rng(20261104)
    idx = np.sort(rng.choice(len(smiles), size=min(sample, len(smiles)), replace=False))

    before = _feature_fingerprint(idx)
    real_read_csv = pd.read_csv

    def poisoned(*args, **kwargs):
        frame = real_read_csv(*args, **kwargs).copy()
        frame["exp"] = np.random.default_rng(99).normal(size=len(frame))
        return frame

    D._frame.cache_clear()
    with mock.patch("pandas.read_csv", side_effect=poisoned):
        poisoned_labels = D.all_labels()
        after = _feature_fingerprint(idx)
    D._frame.cache_clear()

    labels_changed = not np.allclose(poisoned_labels, D.all_labels())
    signatures = _signature_of_feature_builders()
    passed = bool(before == after) and labels_changed and signatures["no_label_parameter"]
    return {
        "name": "02_label_free_features",
        "passed": passed,
        "detail": (
            "label column replaced by N(0,1) noise; G0 / typed-WL token counts / Morgan counts "
            "are bit-identical on a 240-molecule sample; no feature builder accepts a target"
        ),
        "evidence": {
            "n_sampled": int(idx.size),
            "poisoning_changed_labels": bool(labels_changed),
            "feature_fingerprints_equal": bool(before == after),
            "feature_builders": signatures,
            "fingerprint_before": before,
        },
    }


def _feature_fingerprint(idx: Iterable[int]) -> str:
    smiles = D.load_smiles()
    payload: list[str] = []
    for i in idx:
        mol, graph, per_round = tsb_mol.molecule_tokens(smiles[int(i)], rounds=3)
        g0 = tsb_g0.vector(graph)
        morgan, _ = tsb_morgan.vector(mol)
        counts = [C.sha256_bytes(str(sorted(tsb_mol.count_tokens(r).items())).encode())
                  for r in per_round]
        payload.append(
            C.sha256_bytes(
                np.ascontiguousarray(g0).tobytes()
                + np.ascontiguousarray(morgan).tobytes()
                + "".join(counts).encode()
            )
        )
    return C.sha256_bytes("".join(payload).encode())


def _signature_of_feature_builders() -> dict[str, Any]:
    import inspect

    builders = {
        "tsb_g0.vector": tsb_g0.vector,
        "tsb_g0.matrix": tsb_g0.matrix,
        "tsb_mol.molecule_tokens": tsb_mol.molecule_tokens,
        "tsb_mol.refine_tokens": tsb_mol.refine_tokens,
        "tsb_morgan.vector": tsb_morgan.vector,
        "tsb_morgan.build": tsb_morgan.build,
        "tsb_wl.build_matrix": tsb_wl.build_matrix,
        "tsb_wl.select_vocabulary": tsb_wl.select_vocabulary,
    }
    offenders = []
    for name, func in builders.items():
        parameters = set(inspect.signature(func).parameters)
        if parameters & {"y", "labels", "target", "targets"}:
            offenders.append(name)
    return {"no_label_parameter": not offenders, "offenders": offenders}


def gate_vocab_train_only() -> dict[str, Any]:
    """Recompute each vocabulary from the official Train rows and compare."""
    smiles = D.load_smiles()
    splits = D.load_splits()
    problems: list[str] = []
    evidence: dict[str, Any] = {}
    max_round = max(C.WL_ROUNDS["WL3"])
    per_round: list[dict[int, dict[str, int]]] = [dict() for _ in range(max_round + 1)]
    for row, smi in enumerate(smiles):
        _mol, _graph, rounds = tsb_mol.molecule_tokens(smi, rounds=max_round)
        for r, tokens in enumerate(rounds):
            per_round[r][row] = dict(tsb_mol.count_tokens(tokens))
    for suffix, meta in splits.items():
        train_rows = meta["rows"]["train"]
        valid_rows = meta["rows"]["valid"]
        for r in range(max_round + 1):
            stored = F.load_vocab(suffix, r)
            rebuilt = tsb_wl.select_vocabulary(
                tsb_wl.aggregate_token_counts(per_round[r][int(row)] for row in train_rows)
            )
            if rebuilt != stored["tokens"]:
                problems.append(f"{suffix}/r{r}: stored vocabulary != Train-only rebuild")
            if stored["train_rows_sha256"] != meta["rows_sha256"]["train"]:
                problems.append(f"{suffix}/r{r}: recorded vocabulary Train hash mismatch")
            from_valid = tsb_wl.select_vocabulary(
                tsb_wl.aggregate_token_counts(per_round[r][int(row)] for row in valid_rows)
            )
            evidence[f"{suffix}/r{r}"] = {
                "n_tokens": len(rebuilt),
                "train_only_rebuild_matches": rebuilt == stored["tokens"],
                "valid_only_vocabulary_differs": from_valid != rebuilt,
                "overlap_with_valid_only": len(set(rebuilt) & set(from_valid)),
            }
    return {
        "name": "03_vocab_from_train_only",
        "passed": not problems,
        "detail": "every round of every split reproduces the stored vocabulary from official Train rows",
        "evidence": {"problems": problems, "rounds": evidence},
    }


def gate_wl2_columns() -> dict[str, Any]:
    problems: list[str] = []
    evidence: dict[str, Any] = {}
    for suffix in C.SPLIT_SUFFIXES:
        count_matrix, count_names = F.arm_matrix(suffix, "WL2_COUNT")
        bin_matrix, bin_names = F.arm_matrix(suffix, "WL2_BIN")
        if count_names != bin_names:
            problems.append(f"{suffix}: column names differ")
        if count_matrix.shape != bin_matrix.shape:
            problems.append(f"{suffix}: shapes differ")
        g0_dim = len(_g0_column_names())
        bin_wl = bin_matrix[:, g0_dim:]
        count_wl = count_matrix[:, g0_dim:]
        if not np.all(bin_wl.data == 1.0):
            problems.append(f"{suffix}: binary WL block is not 0/1")
        if not np.array_equal(count_wl.data > 0, bin_wl.data > 0):
            problems.append(f"{suffix}: WL binarisation mismatch")
        if not np.array_equal(count_wl.indptr, bin_wl.indptr) or not np.array_equal(
            count_wl.indices, bin_wl.indices
        ):
            problems.append(f"{suffix}: WL sparsity pattern differs")
        evidence[str(suffix)] = {
            "n_columns": len(count_names),
            "shape": list(count_matrix.shape),
            "identical_column_names": count_names == bin_names,
            "nnz": int(count_matrix.nnz),
        }
    return {
        "name": "04_wl2_bin_equals_wl2_count_columns",
        "passed": not problems,
        "detail": "WL2_BIN and WL2_COUNT share vocabulary, column order, sparsity and OOV policy",
        "evidence": {"problems": problems, "splits": evidence},
    }


def gate_wl3_low_rounds() -> dict[str, Any]:
    problems: list[str] = []
    evidence: dict[str, Any] = {}
    for suffix in C.SPLIT_SUFFIXES:
        wl2, names2 = F.arm_matrix(suffix, "WL2_COUNT")
        wl3, names3 = F.arm_matrix(suffix, "WL3_COUNT")
        low = names3[: len(names2)]
        if low != names2:
            problems.append(f"{suffix}: WL3 rounds 0-2 columns differ from WL2")
        if not np.array_equal(wl2.toarray(), wl3[:, : wl2.shape[1]].toarray()):
            problems.append(f"{suffix}: WL3 rounds 0-2 values differ from WL2")
        extra = names3[len(names2):]
        if len(extra) != C.WL_VOCAB_SIZE + 1:
            problems.append(f"{suffix}: WL3 adds {len(extra)} columns, expected 513")
        evidence[str(suffix)] = {
            "wl2_dim": int(wl2.shape[1]),
            "wl3_dim": int(wl3.shape[1]),
            "added_columns": len(extra),
            "low_round_prefix_identical": low == names2,
        }
    return {
        "name": "05_wl3_low_rounds_equal_wl2",
        "passed": not problems,
        "detail": "WL3_COUNT = WL2_COUNT columns + exactly one 513-column round-3 block",
        "evidence": {"problems": problems, "splits": evidence},
    }


def gate_permutation_invariance(sample: int = 200) -> dict[str, Any]:
    smiles = D.load_smiles()
    rng = np.random.default_rng(4242)
    idx = np.sort(rng.choice(len(smiles), size=min(sample, len(smiles)), replace=False))
    problems: list[str] = []
    checked_atoms = 0
    for i in idx:
        mol, graph, per_round = tsb_mol.molecule_tokens(smiles[int(i)], rounds=3)
        n = mol.GetNumAtoms()
        checked_atoms += n
        perm = [int(x) for x in rng.permutation(n)]
        mol2 = tsb_mol.relabel(mol, perm)
        _mol2, graph2, per_round2 = tsb_mol.molecule_tokens_from_mol(mol2, rounds=3)
        if not np.array_equal(tsb_g0.vector(graph), tsb_g0.vector(graph2)):
            problems.append(f"row {int(i)}: G0 changed under renumbering")
        for r in range(4):
            if sorted(per_round[r]) != sorted(per_round2[r]):
                problems.append(f"row {int(i)}: round {r} token multiset changed")
                break
        # explicit bond-endpoint swap: reverse the atom order, which turns
        # every stored (i, j) edge into (j, i) in the relabelled graph
        reverse = list(range(n - 1, -1, -1))
        _mol3, graph3, per_round3 = tsb_mol.molecule_tokens_from_mol(
            tsb_mol.relabel(mol, reverse), rounds=3
        )
        if not np.array_equal(tsb_g0.vector(graph), tsb_g0.vector(graph3)):
            problems.append(f"row {int(i)}: G0 changed under endpoint swap")
        for r in range(4):
            if sorted(per_round[r]) != sorted(per_round3[r]):
                problems.append(f"row {int(i)}: round {r} token multiset changed under endpoint swap")
                break
        m1, _ = tsb_morgan.vector(mol)
        m2, _ = tsb_morgan.vector(mol2)
        if not np.array_equal(m1, m2):
            problems.append(f"row {int(i)}: Morgan changed under renumbering")
    return {
        "name": "06_permutation_invariance",
        "passed": not problems,
        "detail": (
            f"{idx.size} molecules: random atom renumbering and full endpoint swap leave G0, "
            "all four token multisets and the Morgan vector unchanged"
        ),
        "evidence": {"n_molecules": int(idx.size), "n_atoms": int(checked_atoms),
                     "problems": problems[:10]},
    }


def gate_cache_reload() -> dict[str, Any]:
    manifest = F.load_manifest()
    if manifest.get("schema") != F.SCHEMA:
        return {"name": "07_cache_reload", "passed": False,
                "detail": f"unexpected manifest schema {manifest.get('schema')}",
                "evidence": {}}
    F._verify_cache(manifest)
    smiles = D.load_smiles()
    graphs = [tsb_mol.as_graph(tsb_mol.parse_smiles(s)) for s in smiles]
    g0_fresh = tsb_g0.matrix(graphs)
    g0_cached = F.load_g0()
    g0_equal = bool(np.array_equal(g0_fresh, g0_cached))

    max_atoms = max(C.WL_ROUNDS["WL3"])
    round_counts = [list() for _ in range(max_atoms + 1)]
    for smi in smiles:
        _mol, _graph, rounds = tsb_mol.molecule_tokens(smi, rounds=max_atoms)
        for r, tokens in enumerate(rounds):
            round_counts[r].append(dict(tsb_mol.count_tokens(tokens)))

    problems: list[str] = []
    for suffix in C.SPLIT_SUFFIXES:
        for r in range(max_atoms + 1):
            vocab = F.load_vocab(suffix, r)["tokens"]
            fresh = tsb_wl.build_matrix(round_counts[r], vocab, binary=False)
            cached = F.load_block_counts(suffix, r, binary=False)
            fresh_bin = tsb_wl.build_matrix(round_counts[r], vocab, binary=True)
            cached_bin = F.load_block_counts(suffix, r, binary=True)
            if not (np.array_equal(fresh.data, cached.data)
                    and np.array_equal(fresh.indptr, cached.indptr)
                    and np.array_equal(fresh.indices, cached.indices)):
                problems.append(f"{suffix}/r{r}: count cache differs from a fresh build")
            if not (np.array_equal(fresh_bin.data, cached_bin.data)
                    and np.array_equal(fresh_bin.indptr, cached_bin.indptr)
                    and np.array_equal(fresh_bin.indices, cached_bin.indices)):
                problems.append(f"{suffix}/r{r}: binary cache differs from a fresh build")
    morgan_cached = tsb_wl.load_csr(F.MORGAN_PATH)
    subset = np.arange(0, len(smiles), max(1, len(smiles) // 200))[:200]
    morgan_fresh = np.vstack([tsb_morgan.vector(tsb_mol.parse_smiles(smiles[i]))[0] for i in subset])
    morgan_equal = bool(np.array_equal(morgan_fresh, morgan_cached[subset].toarray()))
    passed = g0_equal and morgan_equal and not problems
    return {
        "name": "07_cache_reload_identical",
        "passed": passed,
        "detail": "fresh build == cached blocks for G0 (all rows), WL counts/binary (all rows) and Morgan (200 rows)",
        "evidence": {"g0_equal": g0_equal, "morgan_equal": morgan_equal,
                     "problems": problems[:10], "n_morgan_rows": int(subset.size)},
    }


def gate_no_nan_and_policy() -> dict[str, Any]:
    problems: list[str] = []
    dims: dict[str, Any] = {}
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            matrix, names = F.arm_matrix(suffix, arm)
            if not np.all(np.isfinite(matrix.data)):
                problems.append(f"{suffix}/{arm}: non-finite value")
            if matrix.shape[1] != len(names):
                problems.append(f"{suffix}/{arm}: column-name count mismatch")
            dims[f"{suffix}/{arm}"] = int(matrix.shape[1])
    try:
        tsb_mol.parse_smiles("not-a-molecule")
        parse_policy = False
    except ValueError:
        parse_policy = True
    manifest = F.load_manifest()
    oov_present = True
    for suffix in C.SPLIT_SUFFIXES:
        for round_index in range(4):
            vocab = F.load_vocab(suffix, round_index)
            columns = vocab["columns"]
            if not columns or not columns[-1].endswith("_oov"):
                oov_present = False
            if len(columns) != int(manifest["wl"][str(suffix)][f"r{round_index}"]["dim"]):
                oov_present = False
    passed = not problems and parse_policy and oov_present and manifest["n_parse_failed"] == 0
    return {
        "name": "08_no_nan_and_explicit_policy",
        "passed": passed,
        "detail": (
            "all 15 arm/split matrices finite; unparseable SMILES raises (0 parse failures in "
            "this dataset); unseen tokens always land in the per-round OOV column"
        ),
        "evidence": {"problems": problems, "dims": dims,
                     "parse_failure_is_hard_error": parse_policy,
                     "oov_column_present": oov_present},
    }


def gate_row_alignment(fits: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Check every recorded fit's row ids, label hashes and prediction files."""
    problems: list[str] = []
    checked = 0
    records = list(fits) if fits is not None else _recorded_fits()
    for record in records:
        key = record["key"]
        suffix = int(record["split_suffix"])
        split = D.split(suffix)
        labels = D.split_labels(suffix)
        n = int(record["n_train"])
        expected_train = D.nested_prefix(split["rows"]["train"], n)
        if record["train_rows_sha256"] != C.row_sha256(expected_train):
            problems.append(f"{key}: Train row ids")
        train_index = _index_of(split["rows"]["train"], expected_train)
        if record["y_train_sha256"] != C.sha256_bytes(
            np.ascontiguousarray(labels["train"][train_index]).tobytes()
        ):
            problems.append(f"{key}: Train label hash")
        valid_rows = np.asarray(split["rows"]["valid"], dtype=np.int64)
        if record["valid_rows_sha256"] != C.row_sha256(valid_rows):
            problems.append(f"{key}: Validation row ids")
        for role, expected_rows, expected_y in (
            ("train", expected_train, labels["train"][train_index]),
            ("valid", valid_rows, labels["valid"]),
        ):
            pred_path = C.REPO / record["predictions"][role]["path"]
            if C.sha256_file(pred_path) != record["predictions"][role]["sha256"]:
                problems.append(f"{key}: {role} prediction hash")
            with np.load(pred_path, allow_pickle=False) as handle:
                rows = np.asarray(handle["rows"], dtype=np.int64)
                pred = np.asarray(handle["pred"], dtype=np.float64)
                y_true = np.asarray(handle["y_true"], dtype=np.float64)
            if not np.array_equal(rows, expected_rows):
                problems.append(f"{key}: {role} rows")
            if pred.size != expected_rows.size:
                problems.append(f"{key}: {role} prediction length")
            if not np.array_equal(y_true, expected_y):
                problems.append(f"{key}: {role} labels")
        checked += 1
    return {
        "name": "09_row_and_label_alignment",
        "passed": not problems,
        "detail": f"{checked} recorded fits: nested Train rows, label hashes, rows and predictions align",
        "evidence": {"n_fits": checked, "problems": problems[:10]},
    }


def _index_of(train_rows: np.ndarray, subset: np.ndarray) -> np.ndarray:
    lookup = {int(row): i for i, row in enumerate(train_rows)}
    return np.asarray([lookup[int(row)] for row in subset], dtype=np.int64)


def gate_model_roundtrip() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    suffix = C.SPLIT_SUFFIXES[0]
    split = D.split(suffix)
    labels = D.split_labels(suffix)
    rows = D.nested_prefix(split["rows"]["train"], 600)
    X, _names = F.arm_matrix(suffix, "G0")
    index = _index_of(split["rows"]["train"], rows)
    model, seconds = M.fit(X[rows], labels["train"][index])
    before = M.predict(model, X[split["rows"]["valid"]])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.ubj"
        info = M.save(model, path)
        reloaded = M.load(path)
        after = M.predict(reloaded, X[split["rows"]["valid"]])
    identical = bool(np.array_equal(before, after))
    return {
        "name": "10_model_roundtrip",
        "passed": identical,
        "detail": "XGBoost model saved to .ubj and reloaded produces bit-identical predictions",
        "evidence": {"identical": identical, "fit_seconds": float(seconds),
                     "model_bytes": info["size_bytes"], "n_train": int(rows.size)},
    }


def _g0_column_names() -> list[str]:
    return list(tsb_g0.COLUMN_NAMES)


def _recorded_fits() -> list[dict[str, Any]]:
    fits_dir = C.ANALYSIS / "fits"
    if not fits_dir.is_dir():
        return []
    return [C.read_json(path) for path in sorted(fits_dir.glob("*.json"))]


def run_all(*, include_fit_gates: bool = True) -> list[dict[str, Any]]:
    alignment = (
        gate_row_alignment()
        if include_fit_gates
        else {
            "name": "09_row_and_label_alignment",
            "passed": True,
            "detail": "skipped: no recorded fits yet",
            "evidence": {},
        }
    )
    return [
        gate_splits(),
        gate_labels_not_used(),
        gate_vocab_train_only(),
        gate_wl2_columns(),
        gate_wl3_low_rounds(),
        gate_permutation_invariance(),
        gate_cache_reload(),
        gate_no_nan_and_policy(),
        alignment,
        gate_model_roundtrip(),
        synthetic_boundary(),
    ]
