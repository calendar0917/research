"""Corpus structure statistics and token statistics for CSOI-v1.

Nothing here trains anything: it is descriptive accounting over the frozen cache,
run locally on CPU.  Outputs
``analysis/structure_stats.json`` and ``analysis/token_stats.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D


def _frac(x: np.ndarray, mask: np.ndarray) -> float:
    return float(mask.mean()) if x.size else 0.0


def structure_stats(cfg: dict, pack: dict, raw) -> dict[str, Any]:
    atom_table = np.asarray(pack["atom_table"])
    bond_table = np.asarray(pack["bond_table"])
    frame_atom = np.asarray(pack["frame_atom"], np.int64)
    frame_bond = np.asarray(pack["frame_bond"], np.int64)
    frame_k = np.asarray(pack["frame_k"], np.int64)
    frag_ptr = np.asarray(pack["frag_ptr"], np.int64)
    frag_natoms = np.asarray(pack["frag_natoms"], np.int64)
    frag_nbonds = np.asarray(pack["frag_nbonds"], np.int64)
    frag_ncomp = np.asarray(pack["frag_ncomp"], np.int64)
    frag_port = np.asarray(pack["frag_port"], np.int64)
    frag_bucket = np.asarray(pack["frag_bucket"], np.int64)
    frag_weight = np.asarray(pack["frag_weight"], np.float64)
    frag_counts = np.asarray(pack["frag_counts"], np.float64)
    keys = np.asarray(pack["fragment_keys"])
    counts = np.diff(frag_ptr)
    token_k = frame_k[np.asarray(pack["frag_frame"], np.int64)]

    # ---- object universe --------------------------------------------------
    uniq_frames = np.unique(np.asarray(pack["frag_frame"], np.int64))
    k_of_uniq = frame_k[uniq_frames]
    frame_hist = np.bincount(k_of_uniq, minlength=5).tolist()
    # atom & bond attribute vocabulary actually used by frames
    mask = np.arange(4)[None, :] < frame_k[uniq_frames][:, None]
    used_atom_rows = np.unique(frame_atom[uniq_frames][mask])
    atom_elements = {int(k): int(v) for k, v in
                     zip(*np.unique(atom_table[used_atom_rows, 0], return_counts=True))}
    # is the object's bond pattern reachable from the official bond table?
    bond_rows_used = np.unique(frame_bond[uniq_frames][frame_bond[uniq_frames] >= 0])
    obj = {
        "n_unique_objects": int(uniq_frames.size),
        "n_objects_by_size": {f"k={k}": int(v) for k, v in enumerate(frame_hist) if v},
        "n_objects_by_size_share": {f"k={k}": float(v / max(1, uniq_frames.size))
                                    for k, v in enumerate(frame_hist) if v},
        "n_atom_attribute_rows": int(atom_table.shape[0]),
        "n_bond_attribute_rows": int(bond_table.shape[0]),
        "n_atom_rows_used_by_objects": int(used_atom_rows.size),
        "n_bond_rows_used_by_objects": int(bond_rows_used.size),
        "object_multiplicity": C.quantiles(frag_weight),
        "object_multiplicity_by_size": {
            f"k={k}": C.quantiles(frag_weight[token_k == k])
            for k in range(1, 5) if (token_k == k).any()},
        "object_distance_bucket_hist": np.bincount(frag_bucket, minlength=C.N_BUCKETS).tolist(),
        "object_distance_bucket_share": (np.bincount(frag_bucket, minlength=C.N_BUCKETS)
                                        / max(1, frag_bucket.size)).round(6).tolist(),
    }
    obj["atom_element_counts_used_by_objects"] = atom_elements

    # ---- fragments / words ------------------------------------------------
    word = {
        "n_fragments": int(frag_natoms.size),
        "fragment_natoms": C.quantiles(frag_natoms),
        "fragment_nbonds": C.quantiles(frag_nbonds),
        "fragment_ncomponents": {int(k): int(v) for k, v in
                                 zip(*np.unique(frag_ncomp, return_counts=True))},
        "fragment_n_ports": {int(k): int(v) for k, v in
                             zip(*np.unique(frag_port, return_counts=True))},
        "fragment_tokens": C.quantiles(counts),
        "fragment_token_mass": C.quantiles(np.asarray(
            [frag_weight[frag_ptr[i]:frag_ptr[i + 1]].sum() for i in range(frag_natoms.size)])),
        "fragment_reuse": {
            "n_keys": int(keys.size),
            "n_duplicate_keys": int(keys.size - np.unique(keys).size),
            "top": [[str(k)[:28] + ("..." if len(str(k)) > 28 else ""), int(v)]
                    for k, v in Counter(keys.tolist()).most_common(10)],
        },
        "unique_objects_per_token": float(uniq_frames.size / max(1, int(counts.sum()))),
        "tokens_total": int(counts.sum()),
    }

    # ---- molecules / parses ----------------------------------------------
    mol_parse_ptr = np.asarray(pack["mol_parse_ptr"], np.int64)
    parse_mol = np.asarray(pack["parse_mol"], np.int64)
    parse_S = np.asarray(pack["parse_S"], np.int64)
    parse_C = np.asarray(pack["parse_C"], np.int64)
    parse_kind = np.asarray(pack["parse_kind"], np.int64)
    split = np.asarray(pack["molecule_split"], np.uint8)
    n_parses_per_mol = np.diff(mol_parse_ptr)
    has_c = parse_C >= 0
    mol = {
        "n_molecules": int(pack["n_molecules"]),
        "split_sizes": {"train": int((split == 0).sum()), "valid": int((split == 1).sum()),
                        "test_kept_unused": int((split == 2).sum())},
        "atoms_per_molecule": C.quantiles(np.asarray(
            [C.PD.raw_molecule(raw, m).n_atoms for m in range(0, 4200, 7)])),
        "parses_per_molecule": C.quantiles(n_parses_per_mol),
        "n_molecules_without_cuts": int(((mol_parse_ptr[1:] - mol_parse_ptr[:-1]) == 0).sum()),
        "fallback_rows": [int(x) for x in np.asarray(pack["fallback_rows"]).tolist()],
        "parse_kind_share": {"real_cut": float((parse_kind == 0).mean()),
                             "fallback_no_cut": float((parse_kind == 1).mean())},
        "parse_has_context": float(has_c.mean()),
        "S_side_objects_per_parse": C.quantiles(counts[parse_S]),
        "C_side_objects_per_parse": C.quantiles(counts[parse_C[has_c]]),
        "cut_bond_rows": {int(k): int(v) for k, v in
                          zip(*np.unique(np.asarray(pack["parse_cut_bond_row"])[parse_kind == 0],
                                         return_counts=True))},
        "attention_entries_per_parse": float(np.mean(
            counts[parse_S].astype(np.float64)
            * np.where(has_c, counts[np.where(has_c, parse_C, 0)], 0).astype(np.float64))),
        "attention_entries_total": float(np.sum(
            counts[parse_S].astype(np.float64)
            * np.where(has_c, counts[np.where(has_c, parse_C, 0)], 0).astype(np.float64))),
    }
    # per-split attention cost (train pass is what training pays each epoch)
    per_split = {}
    for name, code in (("train", 0), ("valid", 1)):
        sel = split[parse_mol] == code
        per_split[name] = {
            "n_parses": int(sel.sum()),
            "attention_entries": float(np.sum(
                counts[parse_S[sel]].astype(np.float64)
                * np.where(has_c[sel], counts[np.where(has_c[sel], parse_C[sel], 0)], 0)
                .astype(np.float64))),
        }
    mol["attention_entries_by_split"] = per_split

    # ---- pairs ------------------------------------------------------------
    pairs = {}
    if "pair_mol_a" in pack:
        cell = np.asarray(pack["pair_cell"], np.uint8)
        pairs = {
            "n_pairs": int(cell.size),
            "cells": {int(k): int(v) for k, v in zip(*np.unique(cell, return_counts=True))},
            "n_transforms": int(np.unique(np.asarray(pack["pair_transform"])).size),
            "pair_delta": C.quantiles(np.abs(
                np.asarray(pack["labels"])[np.asarray(pack["pair_mol_b"], np.int64)]
                - np.asarray(pack["labels"])[np.asarray(pack["pair_mol_a"], np.int64)])),
        }
    return {"objects": obj, "fragments": word, "molecules": mol, "pairs": pairs}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.load_pack(cfg, args.dataset)
    raw = C.PD.load_raw_cache(cfg, args.dataset)
    struct = structure_stats(cfg, pack, raw)
    C.write_json(C.analysis_dir() / "structure_stats.json", struct)
    tok = D.token_statistics(pack, cfg, log=print)
    tok["cache_sha256"] = pack["cache_sha256"]
    tok["content_sha256"] = pack["content_sha256"]
    C.write_json(C.analysis_dir() / "token_stats.json", tok)
    print(f"structure: {C.analysis_dir() / 'structure_stats.json'}")
    print(f"tokens:    {C.analysis_dir() / 'token_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
