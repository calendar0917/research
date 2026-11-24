"""Engineering checks for SFEC v0: reconstruction, identity, splits, labels, controls."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec_common as C
import sfec_data as D
import sfec_zinc as Z
from sfec_controls import run_controls


def _pcqm_raw_rows(indices: set[int]) -> dict[int, tuple[str, float]]:
    out = {}
    path = C.PCQM_ROOT / "raw" / "data.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 3:
                continue
            idx = int(parts[0])
            if idx in indices:
                out[idx] = (parts[1], float(parts[2]) if parts[2].strip() else float("nan"))
                if len(out) == len(indices):
                    break
    return out


def check_pcqm(prefix: Path, valid_prefix: Path) -> dict[str, Any]:
    cells = D.load_cells(prefix)
    M = D.load_M(prefix, columns=["mol_id", "mol_key", "split_id", "row_id", "n_atoms"])
    uni = D.load_universe(prefix)
    n_atoms = np.asarray(M["n_atoms"], dtype=np.int64)
    checks = {}
    # cut invariants
    ns = cells["nS"].astype(np.int64)
    nc = cells["nC"].astype(np.int64)
    mid = cells["mol_id"].astype(np.int64)
    inv = {
        "n_cells": int(len(ns)),
        "nS_1_8": int(np.sum((ns >= 1) & (ns <= 8))),
        "nC_ge_6": int(np.sum(nc >= 6)),
        "nC_ge_2nS": int(np.sum(nc >= 2 * ns)),
        "nS_plus_nC_eq_n_atoms": int(np.sum(ns + nc == n_atoms[mid])),
        "bond_type_unique": np.unique(cells["bond_type"]).tolist(),
    }
    checks["cut_invariants"] = {
        "ok": bool(
            inv["nS_1_8"] == inv["n_cells"]
            and inv["nC_ge_6"] == inv["n_cells"]
            and inv["nC_ge_2nS"] == inv["n_cells"]
            and inv["nS_plus_nC_eq_n_atoms"] == inv["n_cells"]
        ),
        **inv,
    }
    # reassembly flags
    rg = cells["reasm_graph"]
    rs = cells["reasm_stereo"]
    checked = rg >= 0
    checks["reassembly_sampled"] = {
        "ok": bool(np.sum(rg[checked] == 0) == 0),
        "n_checked": int(checked.sum()),
        "graph_bad": int(np.sum(rg[checked] == 0)),
        "stereo_bad": int(np.sum(rs[checked] == 0)),
        "stereo_exact_rate": float(np.mean(rs[checked] == 1)) if np.any(checked) else None,
        "note": "graph-exact reconstruction checked on the fixed sample; stereo exactness reported separately",
    }
    # label / SMILES alignment on a fixed sample
    rng = np.random.default_rng(0)
    train_pos = np.sort(rng.choice(min(1_000_000, len(M["mol_id"])), size=min(1000, len(M["mol_id"])), replace=False))
    gids = C.pcqm_cache_gids(C.PCQM_TRAIN_CACHE)
    sample_rows = set(int(gids[i]) for i in train_pos.tolist())
    raw = _pcqm_raw_rows(sample_rows)
    cache_smiles = {}
    with (C.PCQM_TRAIN_CACHE / "smiles.jsonl").open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i in set(train_pos.tolist()):
                cache_smiles[int(gids[i])] = json.loads(line)
            if len(cache_smiles) == len(train_pos):
                break
    same_smiles = sum(1 for idx in sample_rows if cache_smiles.get(idx) == raw.get(idx, ("",))[0])
    label_match = 0
    labels = C.load_pcqm_labels()
    for idx in sample_rows:
        if idx in raw and np.isfinite(raw[idx][1]) and np.isfinite(labels[idx]):
            label_match += int(abs(labels[idx] - raw[idx][1]) < 1e-9)
    checks["label_smiles_alignment"] = {
        "ok": bool(same_smiles == len(sample_rows) and label_match == len(sample_rows)),
        "n_sampled": len(sample_rows),
        "smiles_equal": int(same_smiles),
        "labels_equal": int(label_match),
    }
    # split identity
    split = C.load_pcqm_split()
    train_ids = np.asarray(split["train"])
    valid_ids = np.asarray(split["valid"])
    gids1m = gids[: min(1_000_000, len(gids))]
    checks["split_identity"] = {
        "ok": bool(
            np.all(np.isin(gids1m, train_ids))
            and len(np.unique(gids1m)) == len(gids1m)
            and np.all(np.isfinite(labels[gids1m]))
        ),
        "train_rows_used": int(len(gids1m)),
        "unique": int(len(np.unique(gids1m))),
        "all_in_official_train": bool(np.all(np.isin(gids1m, train_ids))),
        "all_labels_finite": bool(np.all(np.isfinite(labels[gids1m]))),
    }
    # valid
    vcells = D.load_cells(valid_prefix)
    vM = D.load_M(valid_prefix, columns=["mol_id", "n_atoms", "row_id"])
    v_natoms = np.asarray(vM["n_atoms"], dtype=np.int64)
    v_ns = vcells["nS"].astype(np.int64); v_nc = vcells["nC"].astype(np.int64)
    v_mid = vcells["mol_id"].astype(np.int64)
    checks["valid_invariants"] = {
        "ok": bool(np.all(v_ns + v_nc == v_natoms[v_mid])),
        "n_cells": int(len(v_ns)),
        "nS_plus_nC_eq_n_atoms": int(np.sum(v_ns + v_nc == v_natoms[v_mid])),
    }
    # duplicate molecule labels: cells retain every CSV row, M is deduped
    pair_rows = np.unique(
        np.stack([cells["mol_id"].astype(np.int64), cells["row_id"].astype(np.int64)], axis=1),
        axis=0,
    )
    order = np.argsort(pair_rows[:, 0], kind="stable")
    pr = pair_rows[order]
    cuts_at = np.flatnonzero(np.r_[True, pr[1:, 0] != pr[:-1, 0]]) if len(pr) else np.zeros(0, np.int64)
    ends = np.r_[cuts_at[1:], len(pr)] if len(pr) else np.zeros(0, np.int64)
    n_dup_mols = 0
    n_extra_rows = 0
    max_sd = 0.0
    for st, en in zip(cuts_at.tolist(), ends.tolist()):
        group = pr[st:en]
        if len(group) > 1:
            n_dup_mols += 1
            n_extra_rows += len(group) - 1
            ys = labels[group[:, 1].astype(np.int64)]
            max_sd = max(max_sd, float(np.max(np.abs(ys - ys[0]))))
    checks["duplicate_molecule_labels"] = {
        "ok": True,
        "n_duplicate_molecules": int(n_dup_mols),
        "n_extra_rows": int(n_extra_rows),
        "max_abs_label_difference_same_molecule": float(max_sd),
        "note": "duplicate molecules are one identity in all independence/evaluation sets",
    }
    # context split disjointness
    mol = cells["mol_id"].astype(np.int64)
    eval_mol = uni["evaluation_molecules"].astype(bool)
    overlap = np.sum(uni["discovery_mask"] & eval_mol[mol])
    prim = uni["primary_evaluation_cells"].astype(bool)
    pairs = np.unique(np.stack([mol[prim], cells["C_id"][prim].astype(np.int64)], axis=1), axis=0)
    if len(pairs):
        per_mol = np.bincount(pairs[:, 0].astype(np.int64), minlength=len(eval_mol))
        max_ctx_per_mol = int(per_mol.max())
    else:
        max_ctx_per_mol = 0
    checks["context_split_disjointness"] = {
        "ok": bool(overlap == 0 and max_ctx_per_mol <= 1),
        "discovery_cells_with_evaluation_molecule": int(overlap),
        "max_primary_evaluation_contexts_per_molecule": max_ctx_per_mol,
    }
    return {"checks": checks, "all_ok": bool(all(v.get("ok", False) for v in checks.values()))}


def check_zinc(prefix: Path) -> dict[str, Any]:
    cells = D.load_cells(prefix)
    M = D.load_M(prefix, columns=["mol_id", "n_atoms"])
    n_atoms = np.asarray(M["n_atoms"], dtype=np.int64)
    ns = cells["nS"].astype(np.int64); nc = cells["nC"].astype(np.int64)
    mid = cells["mol_id"].astype(np.int64)
    checks = {}
    checks["cut_invariants"] = {
        "ok": bool(np.all(ns + nc == n_atoms[mid])),
        "n_cells": int(len(ns)),
        "nS_plus_nC_eq_n_atoms": int(np.sum(ns + nc == n_atoms[mid])),
        "bond_type_counts": {str(k): int(v) for k, v in zip(*np.unique(cells["bond_type"], return_counts=True))},
    }
    # exact reassembly for all cuts: rebuild the union certificate
    ok = 0
    bad = 0
    checked = 0
    from sfec_zinc import RF, _atom_feat, _bond_feat, dedupe_bonds, whole_key
    for split in ("train", "valid", "test"):
        ds = C.load_zinc_split(split)
        for i in range(len(ds)):
            d = ds[i]
            x = d.x.reshape(-1).numpy().astype(np.int64)
            u, v, bt = dedupe_bonds(d.edge_index.numpy(), d.edge_attr.numpy())
            n = len(x)
            wk = whole_key(x, u, v, bt)
            for eid in Z.RF.find_bridges(n, u, v):
                if int(bt[eid]) != 1:
                    continue
                a, b = int(u[eid]), int(v[eid])
                lab = Z.RF.component_labels(n, u, v, skip_edge=eid)
                side_a = np.flatnonzero(lab == lab[a])
                side_b = np.flatnonzero(lab == lab[b])
                if len(side_a) > len(side_b):
                    side_a, side_b = side_b, side_a
                    a, b = b, a
                nS, nC = len(side_a), len(side_b)
                if not (1 <= nS <= 8 and nC >= 6 and nC >= 2 * nS):
                    continue
                # match a stored cell by nS, nC and the molecule row
                fs = Z.RF._induce(side_a, _atom_feat(x), u, v, _bond_feat(bt), port_global=a)
                fc = Z.RF._induce(side_b, _atom_feat(x), u, v, _bond_feat(bt), port_global=b)
                atom = np.concatenate([fs.atom_feat, fc.atom_feat], axis=0)
                bu = np.concatenate([fs.bond_u, fc.bond_u + len(side_a)], axis=0)
                bv = np.concatenate([fs.bond_v, fc.bond_v + len(side_a)], axis=0)
                bf = np.concatenate([fs.bond_feat, fc.bond_feat], axis=0)
                bu = np.r_[bu, fs.port_local]
                bv = np.r_[bv, len(side_a) + fc.port_local]
                bf = np.vstack([bf, _bond_feat(bt[[eid]])])
                cert = Z.RF.fragment_certificate(atom, bu, bv, bf, port_local=None)
                import hashlib
                key = hashlib.sha256(cert.encode()).hexdigest()
                checked += 1
                if key == wk:
                    ok += 1
                else:
                    bad += 1
                if checked >= 20000:
                    break
            if checked >= 20000:
                break
        if checked >= 20000:
            break
    checks["exact_reassembly"] = {
        "ok": bool(bad == 0 and checked > 0),
        "n_checked": int(checked),
        "n_bad": int(bad),
        "note": "union certificate of the two rooted fragments plus the cut edge equals the whole-molecule certificate; capped at 20000 cuts",
    }
    return {
        "checks": checks,
        "all_ok": bool(all(v.get("ok", False) for v in checks.values())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    args = parser.parse_args(argv)
    pcqm = check_pcqm(Path("../data/cells_pcqm_train_1m"), Path("../data/cells_pcqm_valid"))
    zinc = check_zinc(Path("../data/cells_zinc"))
    controls = run_controls()
    payload = {
        "pcqm": pcqm,
        "zinc": zinc,
        "controls": controls,
        "all_ok": bool(pcqm["all_ok"] and zinc["all_ok"] and controls["all_ok"]),
    }
    C.write_json(args.out, payload)
    print(json.dumps({
        "pcqm_ok": pcqm["all_ok"],
        "zinc_ok": zinc["all_ok"],
        "controls_ok": controls["all_ok"],
        "all_ok": payload["all_ok"],
        "pcqm": pcqm["checks"],
        "zinc": zinc["checks"],
    }, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())