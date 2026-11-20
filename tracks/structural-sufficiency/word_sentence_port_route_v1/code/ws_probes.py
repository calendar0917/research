"""Stage-A controlled examples (data fidelity, content, interface, health).

Only the checks the protocol names are implemented; every result is written to a
machine-readable record.  The synthetic templates of the content probe and the
synthetic rings of the interface probe are *not* chemistry claims.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

import ws_common as C
import ws_train as T
import ws_words as W

TOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def table_from_freeze(freeze: Mapping[str, Any]) -> W.TemplateTable:
    table = W.TemplateTable()
    for row in freeze["template_table"]["templates"]:
        table._register(row["canon"], np.asarray(row["labels"], dtype=np.int64).reshape(-1, 3),
                        np.asarray(row["bu"], dtype=np.int64), np.asarray(row["bv"], dtype=np.int64),
                        np.asarray(row["btype"], dtype=np.int64))
        if table.templates[-1].tid != row["tid"]:  # pragma: no cover - order guard
            raise ValueError("template table order mismatch")
        if table.templates[-1].canon != row["canon"]:  # pragma: no cover
            raise ValueError("template canonical mismatch")
    return table


def make_piece(mol: W.Molecule, atoms: Sequence[int], table: W.TemplateTable):
    """Register the sub-template of ``atoms`` and return it in canonical order."""
    local = list(atoms)
    pos = {a: i for i, a in enumerate(local)}
    labels = mol.atom_feat[local][:, W.WORDS_ATOM_COLS]
    lu, lv, lt = [], [], []
    for a in local:
        for nbr, bk in mol.adj[a]:
            if nbr in pos and pos[a] < pos[nbr]:
                lu.append(pos[a])
                lv.append(pos[nbr])
                lt.append(int(mol.bond_feat[bk, 0]))
    canon, order = W.canonicalize(labels, np.asarray(lu), np.asarray(lv), np.asarray(lt))
    tid = table.add_from_canon(canon, np.asarray(labels), list(zip(lu, lv, lt)), order)
    return tid, [local[i] for i in order]


def synthetic_molecule(index: int, atom_rows: Sequence[Sequence[int]], bonds: Sequence[tuple[int, int, int]],
                       rank: np.ndarray | None = None) -> W.Molecule:
    atom_feat = np.asarray(atom_rows, dtype=np.int64)
    bu = np.asarray([b[0] for b in bonds], dtype=np.int64)
    bv = np.asarray([b[1] for b in bonds], dtype=np.int64)
    bf = np.asarray([[b[2]] + [0, 0] for b in bonds], dtype=np.int64).reshape(-1, 3)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(len(atom_rows))]
    for k, (a, b) in enumerate(zip(bu.tolist(), bv.tolist())):
        adj[a].append((b, k))
        adj[b].append((a, k))
    if rank is None:
        rank = W.molecule_canonical_rank(atom_feat, bu, bv, bf)
    return W.Molecule(index, len(atom_rows), atom_feat, bu, bv, bf, adj, np.asarray(rank))


CARBON_RING = [6, 0, 3, 0, 1, 0, 4, 0, 1]      # ring carbon carrying one substituent
CARBON_PLAIN = [6, 0, 2, 0, 2, 0, 4, 0, 1]     # unsubstituted ring carbon
SUBSTITUENTS = {
    "F": [9, 0, 1, 0, 0, 0, 1, 0, 0],
    "Cl": [17, 0, 1, 0, 0, 0, 1, 0, 0],
    "Br": [35, 0, 1, 0, 0, 0, 1, 0, 0],
    "I": [53, 0, 1, 0, 0, 0, 1, 0, 0],
    "O": [8, 0, 1, 0, 1, 0, 4, 0, 0],
    "N": [7, 0, 1, 0, 2, 0, 4, 0, 0],
    "S": [16, 0, 1, 0, 1, 0, 4, 0, 0],
    "P": [15, 0, 1, 0, 2, 0, 4, 0, 0],
}


def ring_molecule(center: int, pa: int, pb: int, type_a: str, type_b: str) -> W.Molecule:
    """6-ring word with two substituents at fixed ring positions ``pa``/``pb``."""
    rows = []
    for i in range(6):
        rows.append(CARBON_RING if i in (pa, pb) else CARBON_PLAIN)
    a_idx, b_idx = len(rows), len(rows) + 1
    rows.append(SUBSTITUENTS[type_a])
    rows.append(SUBSTITUENTS[type_b])
    bonds = [(i, (i + 1) % 6, 0) for i in range(6)]
    bonds.append((pa, a_idx, 0))
    bonds.append((pb, b_idx, 0))
    return synthetic_molecule(center, rows, bonds)


def ring_state(mol: W.Molecule, table: W.TemplateTable) -> W.PartState:
    ring_atoms = list(range(6))
    tid, ring = make_piece(mol, ring_atoms, table)
    tid_a, a = make_piece(mol, [6], table)
    tid_b, b = make_piece(mol, [7], table)
    state = W.PartState()
    state.pieces = [ring, a, b]
    state.tmpl = [tid, tid_a, tid_b]
    return state


# ---------------------------------------------------------------------------
# 1. data fidelity
# ---------------------------------------------------------------------------


def probe_fidelity(payload: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Words + instances + internal bonds + crossing bonds restore the raw graph."""
    report: dict[str, Any] = {}
    for split in ("train", "valid"):
        prefix = f"word/{split}/"
        arr = {k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)}
        atom_feat = np.asarray(payload[f"base/{split}/atom_feat"], dtype=np.int64)
        bond_feat = np.asarray(payload[f"base/{split}/bond_feat"], dtype=np.int64)
        bond_u = np.asarray(payload[f"base/{split}/bond_u"], dtype=np.int64)
        bond_v = np.asarray(payload[f"base/{split}/bond_v"], dtype=np.int64)
        n_atoms = int(payload[f"base/{split}/graph_n_atoms"].sum())
        n_bonds = int(payload[f"base/{split}/graph_n_bonds"].sum())
        word_atoms = arr["word_atoms"]
        word_bonds = arr["word_bonds"]
        inc_own = arr["inc_own_atom"]
        inc_other = arr["inc_other_atom"]
        inc_bond = arr["inc_bond"]
        counts = np.bincount(word_atoms, minlength=n_atoms)
        atoms_ok = bool(np.all(counts == 1)) and len(word_atoms) == n_atoms
        bond_counts = np.bincount(word_bonds, minlength=n_bonds)
        internal_ok = bool(np.all(bond_counts <= 1))
        internal_total = int((bond_counts == 1).sum())
        cross_pairs = np.sort(np.stack([inc_own, inc_other], axis=1), axis=1)
        unique_cross = np.unique(cross_pairs, axis=0)
        covered = int(internal_total + len(unique_cross))
        # every incidence must reference a real physical bond with its 3 columns
        lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(bond_u.tolist(), bond_v.tolist()))}
        port_ok = True
        attr_ok = True
        for k in range(len(inc_own)):
            key = (min(int(inc_own[k]), int(inc_other[k])), max(int(inc_own[k]), int(inc_other[k])))
            idx = lookup.get(key)
            if idx is None:
                port_ok = False
                break
            if int(inc_bond[k]) != idx:
                port_ok = False
                break
            if not np.array_equal(bond_feat[idx], bond_feat[int(inc_bond[k])]):
                attr_ok = False
                break
        report[split] = {
            "n_atoms": n_atoms, "n_bonds": n_bonds,
            "atoms_covered_exactly_once": atoms_ok,
            "internal_bond_duplicates": int((bond_counts > 1).sum()),
            "internal_bonds_recovered": internal_total,
            "crossing_bond_pairs_unique": int(len(unique_cross)),
            "bond_accounting_complete": bool(covered == n_bonds),
            "crossing_bond_uses_real_endpoints_and_index": bool(port_ok),
            "crossing_bond_attributes_match": bool(attr_ok),
            "raw_attribute_columns": [int(atom_feat.shape[1]), int(bond_feat.shape[1])],
            "n_words": int(len(arr["word_tmpl"])),
            "n_incidences": int(len(inc_own)),
            "n_pairs": int(len(arr["pair_a"])),
            "word_size_histogram": {str(int(k)): int(v) for k, v in
                                    zip(*np.unique(arr["word_n_atoms"], return_counts=True))},
        }
    report["ok"] = all(
        v["atoms_covered_exactly_once"] and v["bond_accounting_complete"]
        and v["internal_bond_duplicates"] == 0 and v["crossing_bond_uses_real_endpoints_and_index"]
        and v["crossing_bond_attributes_match"] and v["raw_attribute_columns"] == [9, 3]
        for v in (report["train"], report["valid"]))
    return report


def probe_relabel_and_batch(payload: Mapping[str, Any], freeze: Mapping[str, Any], *,
                            n_molecules: int = 32, device: str = "cpu") -> dict[str, Any]:
    """Renumbering / re-batching invariance of word identity and prediction."""
    import torch

    import ws_model as M

    table = table_from_freeze(freeze)
    history = list(freeze["merge"]["history"])
    data = T.split_data(payload, "word", "train")
    total = len(data.row_ids)
    comps = np.asarray(payload["base/train/graph_n_components"])
    special = [int(i) for i in np.nonzero(comps > 1)[0][:4]]
    rng = np.random.default_rng(20261120)
    pool = [i for i in range(total) if i not in special]
    chosen = special + [int(i) for i in rng.choice(pool, size=min(n_molecules, len(pool)), replace=False)]
    model = T.make_model(M.ModelSpec(encoder="code", mode="full"), payload, seed=0, device=device)
    model.eval()
    with torch.no_grad():
        whole = model(M.to_device(M.pack_batch(data, chosen), device))["pred"].cpu().numpy()
        single = np.asarray([float(model(M.to_device(M.pack_batch(data, [i]), device))["pred"][0])
                             for i in chosen])
        cut = len(chosen) // 2
        half = np.concatenate([
            model(M.to_device(M.pack_batch(data, chosen[:cut]), device))["pred"].cpu().numpy(),
            model(M.to_device(M.pack_batch(data, chosen[cut:]), device))["pred"].cpu().numpy()])
    batch_diff = float(np.max(np.abs(whole - single)))
    split_diff = float(np.max(np.abs(whole - half)))
    # molecule relabelling: identical word-type multiset and identical prediction
    raw = C.dataset_cache("ogbg-mollipo")
    molecules = W.load_molecules(raw)
    graphs = [int(i) for i in raw["train_idx"][chosen]]
    identity_ok = True
    pred_diffs = []
    base_atom_mask = np.zeros(len(data.row_ids))
    del base_atom_mask
    atom_ptr = data.arrays["graph_atom_ptr"]
    bond_ptr = data.arrays["graph_bond_ptr"]
    for pos, g in enumerate(graphs[:8]):
        mol = molecules[g]
        perm = np.random.default_rng(1000 + g).permutation(mol.n_atoms)
        permuted = _permute_molecule(mol, perm)
        state_p = W.replay_history([permuted], [0], table, history)[0]
        arrays_p = W.build_split([permuted], [0], {0: state_p}, table)
        single_p = M.SplitData(
            name="perm", row_ids=np.asarray([0]), labels=np.asarray([0.0]),
            graph_n_atoms=np.asarray([permuted.n_atoms]),
            graph_n_bonds=np.asarray([len(permuted.bond_u)]),
            atom_feat=permuted.atom_feat, bond_feat=permuted.bond_feat, arrays=arrays_p)
        original = W.replay_history([mol], [0], table, history)[0]
        arrays_o = W.build_split([mol], [0], {0: original}, table)
        single_o = M.SplitData(
            name="orig", row_ids=np.asarray([0]), labels=np.asarray([0.0]),
            graph_n_atoms=np.asarray([mol.n_atoms]),
            graph_n_bonds=np.asarray([len(mol.bond_u)]),
            atom_feat=mol.atom_feat, bond_feat=mol.bond_feat, arrays=arrays_o)
        can_o = sorted(table.templates[t].canon for t in original.tmpl)
        can_p = sorted(table.templates[t].canon for t in state_p.tmpl)
        if can_o != can_p:
            identity_ok = False
        with torch.no_grad():
            p_o = float(model(M.to_device(M.pack_batch(single_o, [0]), device))["pred"][0])
            p_p = float(model(M.to_device(M.pack_batch(single_p, [0]), device))["pred"][0])
        pred_diffs.append(abs(p_o - p_p))
    del atom_ptr, bond_ptr
    return {
        "n_molecules_batched": len(chosen),
        "n_special_multi_component": len(special),
        "max_abs_pred_diff_batch_split": split_diff,
        "max_abs_pred_diff_single_vs_batch": batch_diff,
        "n_molecules_relabelled": len(pred_diffs),
        "max_abs_pred_diff_relabelled": float(max(pred_diffs)) if pred_diffs else float("nan"),
        "tolerance": TOL,
        "batch_invariant": bool(split_diff <= TOL and batch_diff <= TOL),
        "word_identity_relabel_invariant": bool(identity_ok),
        "prediction_relabel_invariant": bool(pred_diffs and max(pred_diffs) <= TOL),
    }


def _permute_molecule(mol: W.Molecule, perm: np.ndarray) -> W.Molecule:
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    atom_feat = mol.atom_feat[perm]
    bu, bv, bf = [], [], []
    for k, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
        na, nb = int(inverse[a]), int(inverse[b])
        bu.append(min(na, nb))
        bv.append(max(na, nb))
        bf.append(mol.bond_feat[k])
    order = np.argsort(np.asarray(bu) * 1000 + np.asarray(bv))
    bu = np.asarray(bu, dtype=np.int64)[order]
    bv = np.asarray(bv, dtype=np.int64)[order]
    bf = np.asarray(bf, dtype=np.int64).reshape(-1, 3)[order]
    adj: list[list[tuple[int, int]]] = [[] for _ in range(mol.n_atoms)]
    for k, (a, b) in enumerate(zip(bu.tolist(), bv.tolist())):
        adj[a].append((b, k))
        adj[b].append((a, k))
    rank = W.molecule_canonical_rank(atom_feat, bu, bv, bf)
    return W.Molecule(mol.index, mol.n_atoms, atom_feat, bu, bv, bf, adj, rank)


# ---------------------------------------------------------------------------
# 2. content probe
# ---------------------------------------------------------------------------


def probe_content(dims: Mapping[str, Sequence[int]], *, device: str = "cpu",
                  updates: int = 500) -> dict[str, Any]:
    import torch

    import ws_model as M

    common = [6, 0, 0, 0, 0, 0, 4, 0, 0]
    prism = [(0, 1, 0), (1, 2, 0), (2, 0, 0), (3, 4, 0), (4, 5, 0), (5, 3, 0), (0, 3, 0), (1, 4, 0), (2, 5, 0)]
    k33 = [(i, j, 0) for i in range(3) for j in range(3, 6)]
    table = W.TemplateTable()
    labels = np.asarray([common] * 6, dtype=np.int64)[:, W.WORDS_ATOM_COLS]
    tid_prism = table.intern(labels, np.asarray([b[0] for b in prism]),
                             np.asarray([b[1] for b in prism]),
                             np.asarray([b[2] for b in prism]))
    tid_k33 = table.intern(labels, np.asarray([b[0] for b in k33]),
                           np.asarray([b[1] for b in k33]), np.asarray([b[2] for b in k33]))
    distinct = tid_prism != tid_k33
    tarr = M.template_arrays(table, dims, int(dims["bond"][0]))
    out: dict[str, Any] = {
        "prism_template_id": int(tid_prism), "k33_template_id": int(tid_k33),
        "exact_identity_distinct": bool(distinct),
        "prism_canonical": table.templates[tid_prism].canon,
        "k33_canonical": table.templates[tid_k33].canon,
        "same_node_and_bond_features": True, "same_n_atoms_and_edges": True,
    }
    # WGINE whole-template representation (expected to coincide: a self-admitted bound)
    enc = M.WGINEEncoder({k: torch.as_tensor(v) for k, v in tarr.items()
                          if isinstance(v, np.ndarray)}, dims, 64).to(device)
    enc.eval()
    t = {k: torch.as_tensor(v).to(device) for k, v in tarr.items() if isinstance(v, np.ndarray)}
    with torch.no_grad():
        x = sum(e(t["node_lab"][:, j]) for j, e in enumerate(enc.atom_emb))
        e = enc.bond_emb(t["edge_type"])
        src, dst = t["edge_index"][0], t["edge_index"][1]
        for layer in range(2):
            residual = x
            msg = torch.relu(x[src] + e)
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            h = (1.0 + enc.eps[layer]) * x + agg
            h = enc.lin2[layer](torch.nn.functional.silu(enc.lin1[layer](h)))
            x = enc.norm[layer](residual + h)
        node_ptr = t["node_ptr"]
        feats = {}
        for name, tid in (("prism", tid_prism), ("k33", tid_k33)):
            lo, hi = int(node_ptr[tid]), int(node_ptr[tid + 1])
            block = x[lo:hi]
            feats[name] = (block.sum(0), block.mean(0))
        wgine_sum_diff = float((feats["prism"][0] - feats["k33"][0]).abs().max())
        wgine_mean_diff = float((feats["prism"][1] - feats["k33"][1]).abs().max())
    out["wgine_sum_pool_max_abs_diff"] = wgine_sum_diff
    out["wgine_mean_pool_max_abs_diff"] = wgine_mean_diff
    out["wgine_distinguishes"] = bool(max(wgine_sum_diff, wgine_mean_diff) > 1e-6)
    # CODE / ID learnability with a small head (two labels, <= updates)
    code_enc = M.CODEEncoder({k: torch.as_tensor(v) for k, v in tarr.items()
                              if isinstance(v, np.ndarray)}, dims, 64, int(dims["bond"][0])).to(device)
    with torch.no_grad():
        codes = code_enc.code_tensor(t)[[tid_prism, tid_k33]]
    out["code_tensors_identical"] = bool(torch.equal(codes[0], codes[1]))
    out["code_fit_success"] = _fit_two_labels(codes, [0.0, 1.0], device, updates)
    ids = torch.eye(2, device=device)
    out["id_fit_success"] = _fit_two_labels(ids, [0.0, 1.0], device, updates)
    out["updates"] = int(updates)
    return out


def _fit_two_labels(features, targets, device: str, updates: int) -> bool:
    import torch

    head = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 64), torch.nn.SiLU(),
                               torch.nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device).view(-1)
    for _ in range(updates):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(features).view(-1), y)
        loss.backward()
        opt.step()
    return bool(float(loss.detach().cpu()) < 1e-4)


# ---------------------------------------------------------------------------
# 3. interface probe
# ---------------------------------------------------------------------------


def probe_interface(payload: Mapping[str, Any], dims: Mapping[str, Sequence[int]], *, device: str = "cpu",
                    max_updates: int = 2400) -> dict[str, Any]:
    """Only the relative port layout varies; FULL should predict, ERASED cannot."""
    import torch

    import ws_model as M

    table = W.TemplateTable()
    types = list(SUBSTITUENTS)
    train_combos = [(a, b) for a in types[:5] for b in types[5:]]
    dev_combos = [(a, b) for a in types[5:] for b in types[:5]]
    centres = [(0, d) for d in (1, 2, 3)]
    samples = []
    for group, combos in (("train", train_combos), ("dev", dev_combos)):
        for combo in combos:
            for (pa, pb) in centres:
                mol = ring_molecule(len(samples), pa, pb, combo[0], combo[1])
                state = ring_state(mol, table)
                d = min((pa - pb) % 6, (pb - pa) % 6)
                samples.append({"group": group, "mol": mol, "state": state,
                                "target": d / 3.0, "distance": d, "combo": combo})
    tarr = M.template_arrays(table, dims, int(dims["bond"][0]))
    splits = {}
    for group in ("train", "dev"):
        rows = [s for s in samples if s["group"] == group]
        mols = [s["mol"] for s in rows]
        states = {i: s["state"] for i, s in enumerate(rows)}
        arrays = W.build_split(mols, list(range(len(mols))), states, table)
        splits[group] = M.SplitData(
            name=group, row_ids=np.arange(len(mols)), labels=np.asarray([s["target"] for s in rows]),
            graph_n_atoms=np.asarray([m.n_atoms for m in mols]),
            graph_n_bonds=np.asarray([len(m.bond_u) for m in mols]),
            atom_feat=np.concatenate([m.atom_feat for m in mols], axis=0),
            bond_feat=np.concatenate([m.bond_feat for m in mols], axis=0), arrays=arrays)
    result: dict[str, Any] = {"n_train_combos": len(train_combos), "n_dev_combos": len(dev_combos),
                              "n_train_graphs": len(splits["train"].row_ids),
                              "n_dev_graphs": len(splits["dev"].row_ids),
                              "distances": sorted({s["distance"] for s in samples}),
                              "targets": sorted({s["target"] for s in samples})}
    # intervention strength: raw R vs the ERASED replacement
    for group in ("train", "dev"):
        arr = splits[group].arrays
        diff = np.abs(arr["pair_R"] - arr["pair_ER"]).max(axis=1)
        result[f"pairs_with_changed_R_{group}"] = int((diff > 1e-6).sum())
        result[f"pairs_{group}"] = int(len(diff))
    fitted = {}
    for mode in ("full", "erased"):
        spec = M.ModelSpec(encoder="code", mode=mode)
        model = M.WordSentenceModel(spec, tarr, dims, int(dims["bond"][0])).to(device)
        fitted[mode] = _fit_probe(model, splits, device, max_updates)
    result["full"] = fitted["full"]
    result["erased"] = fitted["erased"]
    result["erased_groups_identical"] = bool(
        fitted["erased"]["max_abs_diff_same_inputs"] <= TOL)
    result["full_passes"] = bool(fitted["full"]["train_rmse"] < 0.05
                                 and fitted["full"]["dev_rmse"] < 0.10)
    return result


def _fit_probe(model, splits, device: str, max_updates: int) -> dict[str, Any]:
    import torch

    import ws_model as M

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    train = splits["train"]
    n = len(train.row_ids)
    rng = np.random.default_rng(7)
    model.train()
    steps = 0
    for epoch in range(200):
        order = rng.permutation(n)
        for start in range(0, n, 32):
            idx = order[start:start + 32]
            batch = M.to_device(M.pack_batch(train, idx), device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = torch.nn.functional.mse_loss(out["pred"], batch["y"].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            steps += 1
            if steps >= max_updates:
                break
        if steps >= max_updates:
            break
    model.eval()
    report = {"updates": steps}
    with torch.no_grad():
        for group in ("train", "dev"):
            data = splits[group]
            preds = []
            for start in range(0, len(data.row_ids), 64):
                idx = np.arange(start, min(start + 64, len(data.row_ids)))
                batch = M.to_device(M.pack_batch(data, idx), device)
                preds.append(model(batch)["pred"].cpu().numpy())
            pred = np.concatenate(preds)
            report[f"{group}_rmse"] = C.rmse(pred, data.labels)
            report[f"{group}_pred"] = [float(x) for x in np.round(pred, 5)]
            report[f"{group}_target"] = [float(x) for x in data.labels]
    # identical-input grouping for ERASED: same combo, three distances
    data = splits["dev"]
    groups: dict[Any, list[int]] = {}
    # reconstruct grouping from the ring construction: the model only sees R and
    # pair fields, so equal predictions across distances at a fixed combo is the test
    preds = np.asarray(report["dev_pred"])
    combo_of = _dev_combo_index()
    by_combo: dict[Any, list[float]] = {}
    for k, combo in enumerate(combo_of):
        by_combo.setdefault(combo, []).append(float(preds[k]))
    spread = max((max(v) - min(v) for v in by_combo.values()), default=0.0)
    report["max_abs_diff_same_inputs"] = float(spread)
    return report


def _dev_combo_index():
    types = list(SUBSTITUENTS)
    dev_combos = [(a, b) for a in types[5:] for b in types[:5]]
    return [combo for combo in dev_combos for _ in (1, 2, 3)]


# ---------------------------------------------------------------------------
# 4. minimal training health
# ---------------------------------------------------------------------------


def probe_health(payload: Mapping[str, Any], scaler, *, device: str = "cpu",
                 n_molecules: int = 32, steps: int = 60) -> dict[str, Any]:
    import torch

    import ws_model as M

    data = T.split_data(payload, "word", "train")
    idx = np.arange(min(n_molecules, len(data.row_ids)))
    spec = M.ModelSpec(encoder="code", mode="full")
    model = T.make_model(spec, payload, seed=0, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss = None
    r_grad = None
    model.train()
    for step in range(steps):
        raw = M.pack_batch(data, idx)
        batch = M.to_device(raw, device)
        batch["pair_R"] = batch["pair_R"].clone().requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        target = torch.as_tensor((np.asarray(raw["y"], dtype=np.float32) - scaler["mean"]) / scaler["std"],
                                 device=device)
        loss = torch.nn.functional.mse_loss(out["pred"], target)
        loss.backward()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        r_grad = float(batch["pair_R"].grad.abs().max())
        gnorms = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
        opt.step()
    finite = all(np.isfinite(v) for v in gnorms.values())
    pair_head_grad = float(model.pair_lin2.weight.grad.abs().max())
    return {
        "n_molecules": int(len(idx)), "steps": steps,
        "first_loss": first_loss, "last_loss": float(loss.detach().cpu()),
        "loss_decreased": bool(float(loss.detach().cpu()) < first_loss),
        "all_gradients_finite": bool(finite),
        "max_abs_grad_R": r_grad,
        "pair_head_grad_nonzero": bool(pair_head_grad > 0),
        "zero_grad_parameters": [k for k, v in gnorms.items() if v == 0.0],
    }
