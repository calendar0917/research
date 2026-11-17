"""Deterministic graph-target oracles and their specified atom contributions.

Two graph properties of the *same* 4200 Lipo molecules:

``WC_LOGP``  RDKit Wildman-Crippen logP (``includeHs=True, force=True``).
``MOLWT``    RDKit average molecular weight (``Descriptors.MolWt``).

For each property we also produce the per-original-input-node contribution that
the specified generating rule assigns:

* WC: Crippen atom contributions (``_CalcCrippenContribs``) on the explicit-H
  molecule; each added H contribution is assigned to its unique heavy neighbour;
* MW: ``Atom.GetMass()`` on the explicit-H molecule, same H aggregation.

``sum_v c_v`` must reproduce the graph value for all rows (1e-6 / 1e-5).  This
is a *specified generation decomposition*, never an experimental causal truth.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

import gr_common as C
import gr_data as D


def _rdkit():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    return Chem, Crippen, Descriptors, rdMolDescriptors


def atom_order_check(smiles: str, raw_row: Mapping[str, Any], mol) -> dict[str, Any]:
    """Verify the parsed RDKit atom order is exactly the frozen input atom order."""
    n0 = mol.GetNumAtoms()
    classes = np.asarray(raw_row["atom_class"])[:n0]
    charges = np.asarray(raw_row["atom_charge"])[:n0]
    hs = np.asarray(raw_row["atom_h"])[:n0]
    from rdkit import Chem

    index_of = {pair: k for k, pair in enumerate(_lipo_classes())}
    ok = True
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        expected = index_of.get((atom.GetSymbol(), bool(atom.GetIsAromatic())), len(_lipo_classes()))
        if int(classes[idx]) != expected:
            ok = False
        if int(charges[idx]) != _charge_code(int(atom.GetFormalCharge())):
            ok = False
        if int(hs[idx]) != _h_code(int(atom.GetTotalNumHs())):
            ok = False
    edges = sorted(
        (min(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), max(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())))
        for b in mol.GetBonds()
    )
    raw_edges = sorted(
        {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in zip(np.asarray(raw_row["edge_u"]), np.asarray(raw_row["edge_v"]))
        }
    )
    return {"n_atoms": n0, "atom_labels_match": bool(ok), "bond_set_match": bool(edges == raw_edges)}


def _lipo_classes():
    import sys

    sys.path.insert(0, str(C.SQL_CODE))
    import sq_data

    return sq_data.LIPO_RAW_ATOM_CLASSES


def _charge_code(charge: int) -> int:
    import sys

    sys.path.insert(0, str(C.SQL_CODE))
    import sq_data

    return sq_data._lipo_charge_code(charge)


def _h_code(h: int) -> int:
    import sys

    sys.path.insert(0, str(C.SQL_CODE))
    import sq_data

    return sq_data._lipo_h_code(h)


def build_contributions(dataset: str = "ogbg-mollipo") -> dict[str, Any]:
    Chem, Crippen, Descriptors, rdMolDescriptors = _rdkit()
    b = D.bundle(dataset)
    lipo = b.lipo_raw
    smiles = [str(s) for s in lipo["smiles"]]
    atom_ptr = np.asarray(lipo["atom_ptr"], dtype=np.int64)
    n = len(smiles)
    wc_atoms = np.zeros(int(atom_ptr[-1]), dtype=np.float64)
    mw_atoms = np.zeros(int(atom_ptr[-1]), dtype=np.float64)
    wc_graph = np.zeros(n, dtype=np.float64)
    mw_graph = np.zeros(n, dtype=np.float64)
    wc_heavy_sum = np.zeros(n, dtype=np.float64)
    mw_heavy_sum = np.zeros(n, dtype=np.float64)
    checks = {"atom_order": [], "failures": []}
    for row in range(n):
        mol = Chem.MolFromSmiles(smiles[row])
        if mol is None:
            checks["failures"].append({"row": row, "reason": "smiles_parse"})
            continue
        n0 = mol.GetNumAtoms()
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        if a1 - a0 != n0:
            checks["failures"].append({"row": row, "reason": "atom_count", "raw": a1 - a0, "rdkit": n0})
            continue
        raw_row = {
            "atom_class": np.asarray(lipo["atom_class"])[a0:a1],
            "atom_charge": np.asarray(lipo["atom_charge"])[a0:a1],
            "atom_h": np.asarray(lipo["atom_h"])[a0:a1],
            "edge_u": np.asarray(lipo["edge_u"])[int(np.asarray(lipo["edge_ptr"])[row]) : int(np.asarray(lipo["edge_ptr"])[row + 1])],
            "edge_v": np.asarray(lipo["edge_v"])[int(np.asarray(lipo["edge_ptr"])[row]) : int(np.asarray(lipo["edge_ptr"])[row + 1])],
        }
        check = atom_order_check(smiles[row], raw_row, mol)
        if len(checks["atom_order"]) < 4200:
            checks["atom_order"].append({"row": row, **check})
        if not (check["atom_labels_match"] and check["bond_set_match"]):
            checks["failures"].append({"row": row, "reason": "atom_order"})
        mh = Chem.AddHs(mol)
        for idx in range(n0):
            sym = int(mh.GetAtomWithIdx(idx).GetSymbol() == "H")
            if sym:
                checks["failures"].append({"row": row, "reason": "explicit_H_in_input"})
        contribs = rdMolDescriptors._CalcCrippenContribs(mh, force=True)
        wc_all = np.asarray([float(c[0]) for c in contribs], dtype=np.float64)
        mw_all = np.asarray([float(a.GetMass()) for a in mh.GetAtoms()], dtype=np.float64)
        wc_node = wc_all[:n0].copy()
        mw_node = mw_all[:n0].copy()
        for idx in range(n0, mh.GetNumAtoms()):
            neighbours = [int(nb.GetIdx()) for nb in mh.GetAtomWithIdx(idx).GetNeighbors()]
            if len(neighbours) != 1:
                checks["failures"].append({"row": row, "reason": "h_neighbours", "n": len(neighbours)})
                continue
            wc_node[neighbours[0]] += wc_all[idx]
            mw_node[neighbours[0]] += mw_all[idx]
        wc_atoms[a0:a1] = wc_node
        mw_atoms[a0:a1] = mw_node
        wc_graph[row] = float(Crippen.MolLogP(mh, includeHs=True, force=True))
        mw_graph[row] = float(Descriptors.MolWt(mol))
        wc_heavy_sum[row] = float(wc_node.sum())
        mw_heavy_sum[row] = float(mw_node.sum())
    err_wc = np.abs(wc_heavy_sum - wc_graph)
    err_mw = np.abs(mw_heavy_sum - mw_graph)
    if float(err_wc.max()) > 1e-6:
        raise RuntimeError(f"WC contribution conservation failed: max {err_wc.max():.3e} at row {int(err_wc.argmax())}")
    if float(err_mw.max()) > 1e-5:
        raise RuntimeError(f"MW contribution conservation failed: max {err_mw.max():.3e} at row {int(err_mw.argmax())}")

    # independent rebuild under a permuted atom order for >=32 rows
    reorder = _reorder_check(smiles, atom_ptr, wc_graph, mw_graph, wc_atoms, mw_atoms)
    summary = {
        "dataset": dataset,
        "n_rows": n,
        "n_atoms": int(atom_ptr[-1]),
        "WC_LOGP": {
            "max_abs_error": float(err_wc.max()),
            "mean_abs_error": float(err_wc.mean()),
            "graph_mean": float(wc_graph.mean()),
            "graph_std": float(wc_graph.std(ddof=0)),
            "atom_contrib_sum_max_abs_error": float(np.abs(wc_atoms.sum() - wc_graph.sum())),
        },
        "MOLWT": {
            "max_abs_error": float(err_mw.max()),
            "mean_abs_error": float(err_mw.mean()),
            "graph_mean": float(mw_graph.mean()),
            "graph_std": float(mw_graph.std(ddof=0)),
        },
        "atom_order_check": {
            "n_checked": len(checks["atom_order"]),
            "all_match": all(x["atom_labels_match"] and x["bond_set_match"] for x in checks["atom_order"]),
            "examples": checks["atom_order"][:3],
        },
        "failures": checks["failures"][:10],
        "n_failures": len(checks["failures"]),
        "reorder_check": reorder,
        "note": "specified generating decomposition of a computed property; not a causal truth",
    }
    return {
        "WC_LOGP": wc_graph,
        "MOLWT": mw_graph,
        "WC_LOGP_atom": wc_atoms,
        "MOLWT_atom": mw_atoms,
        "atom_ptr": atom_ptr,
        "summary": summary,
    }


def _reorder_check(smiles, atom_ptr, wc_graph, mw_graph, wc_atoms, mw_atoms) -> dict[str, Any]:
    """Rebuild >=32 rows under an independent (permuted) SMILES atom order."""
    Chem, Crippen, Descriptors, rdMolDescriptors = _rdkit()
    rng = np.random.default_rng(C.TRAIN_SEED)
    rows = rng.choice(len(smiles), size=min(64, len(smiles)), replace=False)
    worst = {"wc": 0.0, "mw": 0.0, "node_wc": 0.0, "node_mw": 0.0}
    for row in rows.tolist():
        mol = Chem.MolFromSmiles(smiles[row])
        n0 = mol.GetNumAtoms()
        order = rng.permutation(n0)
        inverse = np.argsort(order)
        perm = Chem.RenumberAtoms(mol, [int(i) for i in order])
        mh = Chem.AddHs(perm)
        contribs = rdMolDescriptors._CalcCrippenContribs(mh, force=True)
        wc_all = np.asarray([float(c[0]) for c in contribs])
        mw_all = np.asarray([float(a.GetMass()) for a in mh.GetAtoms()])
        wc_node = wc_all[:n0].copy()
        mw_node = mw_all[:n0].copy()
        for idx in range(n0, mh.GetNumAtoms()):
            nb = [int(x.GetIdx()) for x in mh.GetAtomWithIdx(idx).GetNeighbors()]
            if len(nb) != 1:
                raise RuntimeError(f"added H at {idx} has {len(nb)} neighbours")
            wc_node[nb[0]] += wc_all[idx]
            mw_node[nb[0]] += mw_all[idx]
        wc = float(Crippen.MolLogP(mh, includeHs=True, force=True))
        mw = float(Descriptors.MolWt(perm))
        worst["wc"] = max(worst["wc"], abs(wc - wc_graph[row]))
        worst["mw"] = max(worst["mw"], abs(mw - mw_graph[row]))
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        # permuted index i corresponds to original index order[i]
        back_wc = np.empty(n0)
        back_mw = np.empty(n0)
        back_wc[order] = wc_node
        back_mw[order] = mw_node
        worst["node_wc"] = max(worst["node_wc"], float(np.abs(back_wc - wc_atoms[a0:a1]).max()))
        worst["node_mw"] = max(worst["node_mw"], float(np.abs(back_mw - mw_atoms[a0:a1]).max()))
    return {
        "n_rows": int(len(rows)),
        "worst_graph_wc": float(worst["wc"]),
        "worst_graph_mw": float(worst["mw"]),
        "worst_node_wc": float(worst["node_wc"]),
        "worst_node_mw": float(worst["node_mw"]),
        "note": "independent AddHs + Crippen/MW rebuild under a permuted atom order, mapped back through the permutation",
    }


def prepare(dataset: str = "ogbg-mollipo") -> dict[str, Any]:
    out_dir = C.ANALYSIS_DIR / "oracle_contrib"
    out_dir.mkdir(parents=True, exist_ok=True)
    built = build_contributions(dataset)
    path = out_dir / f"{dataset}.npz"
    C.atomic_write_npz(
        path,
        {
            "WC_LOGP": built["WC_LOGP"],
            "MOLWT": built["MOLWT"],
            "WC_LOGP_atom": built["WC_LOGP_atom"],
            "MOLWT_atom": built["MOLWT_atom"],
            "atom_ptr": built["atom_ptr"],
        },
    )
    summary = dict(built["summary"])
    summary["path"] = str(path.relative_to(C.REPO_ROOT))
    summary["sha256"] = C.sha256_file(path)
    C.write_json(C.ANALYSIS_DIR / "oracle_contrib" / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=1))
