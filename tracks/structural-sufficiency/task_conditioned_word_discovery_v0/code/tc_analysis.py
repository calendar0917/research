"""Post-selection analysis artifacts and the frozen identity manifest.

This script never changes a selection or a vocab.  It only aggregates the
already-frozen artifacts, computes coverage/repetition/structure displays, and
writes hashes for the GPU launch commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

import tc_common as C
import tc_data as D
import tc_vocab as V


def _read(split: str, name: str) -> Any:
    return C.read_json(C.ANALYSIS_DIR / f"{name}_{split}.json")


def combined_selected() -> dict[str, Any]:
    out = []
    for split in ("D0", "D1"):
        for entry in _read(split, "selected_words"):
            out.append({"split": split, **entry})
    C.write_json(C.ANALYSIS_DIR / "selected_words.json", out)
    C.write_csv(C.ANALYSIS_DIR / "selected_words.csv", out)
    return {"n": len(out), "sha256": C.sha256_file(C.ANALYSIS_DIR / "selected_words.json")}


def combined_scores() -> dict[str, Any]:
    rows = []
    for split in ("D0", "D1"):
        payload = _read(split, "selection_scores")
        for row in payload["rows"]:
            rows.append({"split": split, **row})
    C.write_json(C.ANALYSIS_DIR / "selection_scores.json", rows)
    C.write_csv(C.ANALYSIS_DIR / "selection_scores.csv", rows)
    return {"n": len(rows), "sha256": C.sha256_file(C.ANALYSIS_DIR / "selection_scores.json")}


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (x - mu) / sd, sd


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    za, _ = _standardise(a)
    zb, _ = _standardise(b)
    return (za.T @ zb) / a.shape[0]


def coverage_and_repetition(split: str, atom_dims: Sequence[int],
                            bond_dims: Sequence[int]) -> dict[str, Any]:
    payload, x_fit, x_eval, x0_fit, x0_eval = V.load_candidates(split)
    data = D.build_split_data(split, atom_dims, bond_dims)
    task_cols = [int(x) for x in _read(split, "selection_scores")["task_columns"]]
    freq_cols = [int(x) for x in _read(split, "selection_scores")["freq_columns"]]
    selected = _read(split, "selected_words")
    meta = payload["candidates"]
    atom_ptr = np.zeros(data.rows.size + 1, np.int64)
    atom_ptr[1:] = np.cumsum(data.n_atoms)
    fit_idx = data.fit_index
    fit_atoms = np.concatenate([np.arange(atom_ptr[i], atom_ptr[i + 1])
                                for i in fit_idx]) if fit_idx.size else np.zeros(0, np.int64)
    fit_mol_of_atom = np.concatenate([np.full(int(data.n_atoms[i]), k)
                                      for k, i in enumerate(fit_idx)]) if fit_idx.size \
        else np.zeros(0, np.int64)
    cov: dict[str, Any] = {}
    per_arm: dict[str, Any] = {}
    for arm, cols, key in (("D_TASK", data.arm_columns["D_TASK"], "task"),
                           ("D_FREQ", data.arm_columns["D_FREQ"], "freq")):
        cfit = np.concatenate([np.asarray(data.counts[int(i)][:, cols], np.float32)
                               for i in fit_idx], axis=0) if fit_idx.size else \
            np.zeros((0, len(cols)), np.float32)
        any_word = (cfit > 0).any(axis=1)
        counts_nonzero = (cfit > 0).sum(axis=1)
        mol_any = np.zeros(len(fit_idx), bool)
        if fit_idx.size:
            for m in np.unique(fit_mol_of_atom):
                mol_any[m] = any_word[fit_mol_of_atom == m].any()
        per_word = []
        for j, col in enumerate(cols.tolist()):
            hits = cfit[:, j] > 0
            per_word.append({
                "rank": j, "column": int(col),
                "key": data.union_vocab[int(col)]["key"],
                "root_coverage_fit": float(hits.mean()) if hits.size else 0.0,
                "occurrence_fit_vocab": int(hits.sum()),
            })
        per_arm[arm] = {
            "root_coverage_fit": float(any_word.mean()) if any_word.size else 0.0,
            "molecule_coverage_fit": float(mol_any.mean()) if mol_any.size else 0.0,
            "effective_words_per_root_mean": float(counts_nonzero.mean()) if counts_nonzero.size else 0.0,
            "effective_words_per_root_median": float(np.median(counts_nonzero)) if counts_nonzero.size else 0.0,
            "effective_words_per_root_max": int(counts_nonzero.max()) if counts_nonzero.size else 0,
            "mean_total_occurrences_per_root": float(cfit.sum(axis=1).mean()) if cfit.size else 0.0,
            "per_word": per_word,
        }

    # selected-word vs low-order feature correlation on Fit
    x0 = np.asarray(x0_fit, np.float64)
    xsel = np.asarray(x_fit[:, task_cols].toarray(), np.float64)
    corr = _corr_matrix(xsel, x0)
    names = payload["x0_columns"]
    repetition = []
    for j, col in enumerate(task_cols):
        row = corr[j]
        order = np.argsort(-np.abs(row))[:3]
        repetition.append({
            "column": int(col), "key": meta[col]["key"],
            "max_abs_corr_low_order": float(np.abs(row).max()),
            "top_low_order": [{"column": int(i), "name": names[int(i)],
                               "corr": C.safe_float(row[i])} for i in order],
        })
    # pairwise correlations among the selected task and frequency words
    both = sorted(set(task_cols) | set(freq_cols))
    xb = np.asarray(x_fit[:, both].toarray(), np.float64)
    cb = _corr_matrix(xb, xb)
    pairs = []
    for a in range(len(both)):
        for b in range(a + 1, len(both)):
            if abs(cb[a, b]) >= 0.95:
                pairs.append({"a_key": meta[both[a]]["key"],
                              "b_key": meta[both[b]]["key"],
                              "corr": C.safe_float(cb[a, b])})
    pairs.sort(key=lambda p: -abs(float(p["corr"])))
    equiv_selected = []
    for e in selected:
        if int(e["structure"].get("size", 0)) in (3, 4):
            group = [x for x in e["equiv_members"]]
            if len(group) > 1:
                equiv_selected.append({"arm": e["arm"], "rank": e["rank"],
                                       "key": e["key"],
                                       "equiv_member_keys": e["equiv_member_keys"]})
    return {
        "coverage": per_arm,
        "low_order_repetition": repetition,
        "selected_pair_corr_abs_ge_0.95": pairs,
        "n_pairs_abs_ge_0.95": int(len(pairs)),
        "selected_words_with_equivalent_columns": equiv_selected,
        "note": "Fit correlations only; descriptive, not chemical contributions",
    }


def _occurrences_for_family(split: str, family: str, entries: Sequence[Mapping[str, Any]],
                            max_per_word: int = 5) -> dict[str, list[dict[str, Any]]]:
    gran, size_s = family.split("_")
    size = int(size_s)
    atom_dims, bond_dims = C.atom_bond_dims()
    want = {str(e["key"]): C.signature_bytes(e["signature"]) for e in entries}
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in want}
    total = Counter()
    bundle = C.split_bundle(split)
    for row in np.asarray(bundle.fit_rows, np.int64).tolist():
        st = C.build_struct(bundle, int(row))
        sig = V._signatures_for(st, size, gran, atom_dims, bond_dims)
        if sig.shape[0] == 0:
            continue
        supports = np.asarray(st.atoms[size], np.int64)
        for i, s in enumerate(sig):
            key = C.signature_key(gran, size, [int(x) for x in s])
            if key not in want:
                continue
            total[key] += 1
            if len(out[key]) >= max_per_word:
                continue
            support = supports[i, :size].tolist()
            root = int(support[0])
            atoms = [{"slot": j, "local_atom_id": int(a),
                      "fields9": np.asarray(st.atom_feat[int(a)], np.int64).tolist()}
                     for j, a in enumerate(support)]
            bonds = []
            for j in range(int(st.bonds[size].shape[1])):
                bid = int(st.bonds[size][i, j])
                if bid < 0:
                    continue
                slots = [int(x) for x in np.asarray(st.bslot[size][i, j])]
                bonds.append({"slots": slots, "bond_id": bid,
                              "fields3": np.asarray(st.bond_feat[bid], np.int64).tolist()})
            out[key].append({
                "row": int(row),
                "smiles": str(C.GD.molecule_view(bundle.raw, int(row))["smiles"]),
                "root_local_atom_id": root,
                "support_local_atom_ids": support,
                "atoms": atoms,
                "internal_bonds": bonds,
            })
    return {k: {"n_occurrences_fit": int(total[k]), "examples": v} for k, v in out.items()}


def structure_top20(split: str) -> dict[str, Any]:
    selected = _read(split, "selected_words")
    task = [e for e in selected if e["arm"] == "D_TASK"]
    task.sort(key=lambda e: (-(float(e["score"]) if e["score"] is not None else -1.0),
                             str(e["key"])))
    top = task[:20]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for e in top:
        by_family.setdefault(f"{e['gran']}_{e['size']}", []).append(e)
    atom_dims, bond_dims = C.atom_bond_dims()
    result = {
        "split": split, "rule": "top 20 D_TASK words by frozen score, key ascending",
        "words": [],
    }
    for family, entries in sorted(by_family.items()):
        occ = _occurrences_for_family(split, family, entries)
        for e in entries:
            result["words"].append({
                "rank": int(e["rank"]), "score": e["score"],
                "size": int(e["size"]), "gran": str(e["gran"]),
                "key": str(e["key"]), "signature": [int(x) for x in e["signature"]],
                "df": int(e["df"]), "scaffold_df": int(e["scaffold_df"]),
                "cond_corr_folds": e["cond_corr_folds"],
                "raw_corr_folds": e["raw_corr_folds"],
                "structure": C.decode_word(e, atom_dims, bond_dims),
                "occurrences": occ[str(e["key"])],
            })
    result["words"].sort(key=lambda w: (w["rank"], str(w["key"])))
    C.write_json(C.ANALYSIS_DIR / f"structure_top20_{split}.json", result)
    return result


def freeze_manifest() -> dict[str, Any]:
    files = []
    for pattern in ("*.json", "*.csv", "*.npz"):
        for path in sorted(C.ANALYSIS_DIR.glob(pattern)):
            if path.name == "frozen_manifest.json":
                continue
            files.append({"path": str(path.relative_to(C.ROUND_DIR)),
                          "sha256": C.sha256_file(path),
                          "bytes": int(path.stat().st_size)})
    cfg_sha = C.sha256_file(C.CONFIG_PATH)
    proto_sha = C.sha256_file(C.ROUND_DIR / "PROTOCOL.md")
    code_files = ["tc_common.py", "tc_vocab.py", "tc_cpu.py", "tc_data.py",
                  "tc_neural.py", "tc_train.py", "tc_checks.py", "tc_analysis.py"]
    payload = {
        "protocol_id": C.load_config()["protocol_id"],
        "version": C.load_config()["version"],
        "frozen_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "config_sha256": cfg_sha,
        "protocol_sha256": proto_sha,
        "code_sha256": C.code_sha(code_files),
        "checks": C.read_json(C.ANALYSIS_DIR / "checks.json"),
        "analysis_files": files,
        "folds": {s: {"row_sha256": C.read_json(C.ANALYSIS_DIR / f"selection_scores_{s}.json")["folds"]["fold_of_row_sha256"],
                      "scaffold_sha256": C.read_json(C.ANALYSIS_DIR / f"selection_scores_{s}.json")["folds"]["scaffold_groups_sha256"]}
                  for s in ("D0", "D1")},
        "vocab_sha256": {s: {"task": C.sha256_file(C.ANALYSIS_DIR / f"vocab_{s}_task.json"),
                             "freq": C.sha256_file(C.ANALYSIS_DIR / f"vocab_{s}_freq.json")}
                         for s in ("D0", "D1")},
        "note": "selection rules were frozen before D1 Dev prediction and before any formal GPU run",
    }
    C.write_json(C.ANALYSIS_DIR / "frozen_manifest.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("all", "combined", "structure", "freeze"))
    args = ap.parse_args()
    atom_dims, bond_dims = C.atom_bond_dims()
    if args.command in ("all", "combined"):
        print(combined_selected())
        print(combined_scores())
        stats = {}
        for split in ("D0", "D1"):
            stats[split] = coverage_and_repetition(split, atom_dims, bond_dims)
        C.write_json(C.ANALYSIS_DIR / "vocabulary_stats.json", stats)
        print("vocabulary_stats written")
    if args.command in ("all", "structure"):
        for split in ("D0", "D1"):
            r = structure_top20(split)
            print(split, "top20", len(r["words"]))
    if args.command in ("all", "freeze"):
        print(C.jsonable(freeze_manifest()))


if __name__ == "__main__":
    main()