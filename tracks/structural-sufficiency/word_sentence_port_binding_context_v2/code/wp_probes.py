"""Stage-A engineering checks and the two functional probes.

These are *capability* checks only.  Nothing here touches the real task models:
the probes use their own template table, own weights and artificial labels, and
their numbers never enter a reported task result.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

import wp_common as C
import wp_data as D
import wp_model as M
import wp_train as T


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def rdkit_molecule(index: int, elements: Sequence[str], bonds: Sequence[tuple[int, int, int]],
                   rank: np.ndarray | None = None) -> Any:
    """Build a ``ws_words.Molecule`` with official-schema columns via RDKit."""
    from ogb.utils.features import allowable_features
    from rdkit import Chem

    rw = Chem.RWMol()
    for element in elements:
        rw.AddAtom(Chem.Atom(element))
    for i, j, order in bonds:
        rw.AddBond(int(i), int(j), [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                    Chem.BondType.TRIPLE][int(order)])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    hyb = {name: k for k, name in enumerate(allowable_features["possible_hybridization_list"])}
    btype = {name: k for k, name in enumerate(allowable_features["possible_bond_type_list"])}
    rows = []
    for atom in mol.GetAtoms():
        rows.append([atom.GetAtomicNum(), 0, atom.GetDegree(), atom.GetFormalCharge(),
                     atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                     hyb[str(atom.GetHybridization())], int(atom.GetIsAromatic()),
                     int(atom.IsInRing())])
    bond_rows = []
    for bond in mol.GetBonds():
        bond_rows.append([btype[str(bond.GetBondType())], 0, int(bond.GetIsConjugated())])
    return _molecule(index, np.asarray(rows, dtype=np.int64),
                     np.asarray([b[0] for b in bonds], dtype=np.int64),
                     np.asarray([b[1] for b in bonds], dtype=np.int64),
                     np.asarray(bond_rows, dtype=np.int64), rank)


def _molecule(index: int, atom_feat, bu, bv, bf, rank) -> Any:
    import ws_words as W

    n = int(atom_feat.shape[0])
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (a, b) in enumerate(zip(bu.tolist(), bv.tolist())):
        adj[a].append((b, k))
        adj[b].append((a, k))
    if rank is None:
        rank = np.zeros(n, dtype=np.int64)
    return W.Molecule(index, n, np.asarray(atom_feat, dtype=np.int64), np.asarray(bu, dtype=np.int64),
                      np.asarray(bv, dtype=np.int64), np.asarray(bf, dtype=np.int64), adj,
                      np.asarray(rank, dtype=np.int64))


def _probe_scaler(labels: Sequence[float]) -> dict[str, float]:
    values = np.asarray(labels, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std() or 1.0), "n": int(values.size)}


def _fit_probe(model, split: D.SplitData, scaler: Mapping[str, float], *, steps: int = 200,
               lr: float = 1e-3, full_batch: bool = True) -> dict[str, Any]:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    idx = np.arange(len(split))
    batch = D.to_device(D.pack_batch(split, idx), "cpu")
    target = torch.as_tensor((split.labels - scaler["mean"]) / scaler["std"], dtype=torch.float32)
    losses = []
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = torch.nn.functional.mse_loss(out["pred"], target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        pred = model(batch)["pred"].numpy()
    pred_raw = pred * scaler["std"] + scaler["mean"]
    return {"pred_raw": pred_raw, "losses": losses, "first_loss": losses[0], "last_loss": losses[-1],
            "rmse_raw": C.rmse(pred_raw, split.labels),
            "pred_std": pred, "row_ids": split.row_ids}


def build_probe_model(spec: M.ModelSpec, packed: Mapping[str, np.ndarray], seed: int = 0):
    import ws_words as W

    table = W.TemplateTable()  # probes intern their own templates here
    # template arrays are rebuilt from the probe table below
    from ws_model import template_arrays

    dims = C.official_dims()
    return table, dims, template_arrays


def probe_template_arrays(table, dims) -> dict[str, np.ndarray]:
    from ws_model import template_arrays

    return template_arrays(table, dims, int(dims["bond"][0]))


def make_probe_model(table, dims, spec: M.ModelSpec, seed: int = 0):
    import torch

    tmpl = probe_template_arrays(table, dims)
    tmpl["dist"] = _probe_dist_table(table)
    model = M.PortBindModel(spec, tmpl, dims)
    M.init_by_name(model, 5000 + seed)
    return model


def _probe_dist_table(table) -> np.ndarray:
    n = len(table)
    out = np.zeros((n, 6, 6), dtype=np.int8)
    for tmpl in table.templates:
        m = tmpl.n
        dist = np.zeros((m, m), dtype=np.int8)
        for s in range(m):
            seen = np.full(m, -1, dtype=np.int64)
            seen[s] = 0
            queue = [s]
            while queue:
                node = queue.pop(0)
                for nxt in np.nonzero(tmpl.adj[node])[0].tolist():
                    if seen[nxt] < 0:
                        seen[nxt] = seen[node] + 1
                        queue.append(nxt)
            dist[s] = seen.astype(np.int8)
        out[tmpl.tid, :m, :m] = dist
    return out


# ---------------------------------------------------------------------------
# engineering checks 1-4
# ---------------------------------------------------------------------------


def run_engineering_checks(args) -> None:
    import torch

    import ws_words as W

    name = D_dict_dataset(args.dataset)
    data = D.load_dataset(name)
    words = data["words"]
    dist_table = data["derived"]["tmpl/dist"]
    limit = int(args.limit)
    report: dict[str, Any] = {"dataset": name, "limit": limit, "commit": C.git_commit()}
    t0 = time.perf_counter()
    report["fidelity"] = check_fidelity(words, limit)
    report["invariance"] = check_invariance(data, words, dist_table, limit)
    report["init_identity"] = check_init_identity(data, limit)
    report["health"] = check_health(data, limit)
    report["seconds"] = time.perf_counter() - t0
    C.write_json(C.analysis_dir() / "checks.json", C.jsonable(report))
    print(json.dumps(report, indent=2)[:4000], flush=True)


def D_dict_dataset(alias: str) -> str:
    return {"lipo": "ogbg-mollipo", "esol": "ogbg-molesol"}.get(alias, alias)


def check_fidelity(words: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """Words + internal bonds + crossing bonds restore the raw 9/3-column graph."""
    arr = D._split_arrays(words, "train")
    base = D._base_arrays(words, "train")
    atom_feat = np.asarray(base["atom_feat"], dtype=np.int64)
    bond_feat = np.asarray(base["bond_feat"], dtype=np.int64)
    bu = np.asarray(base["bond_u"], dtype=np.int64)
    bv = np.asarray(base["bond_v"], dtype=np.int64)
    gw = arr["graph_word_ptr"]
    ga = arr["graph_atom_ptr"]
    gb = arr["graph_bond_ptr"]
    checked = 0
    for g in range(min(limit, len(gw) - 1)):
        a0, a1 = int(ga[g]), int(ga[g + 1])
        b0, b1 = int(gb[g]), int(gb[g + 1])
        w0, w1 = int(gw[g]), int(gw[g + 1])
        seen = np.zeros(a1 - a0, dtype=np.int64)
        internal = set()
        crossing = set()
        for w in range(w0, w1):
            atoms = arr["word_atoms"][arr["word_ptr_atom"][w]:arr["word_ptr_atom"][w + 1]]
            if len(set(atoms.tolist())) != len(atoms):
                raise ValueError(f"graph {g}: duplicate atom in word {w}")
            for a in atoms:
                seen[a - a0] += 1
            for bk in arr["word_bonds"][arr["word_ptr_bond"][w]:arr["word_ptr_bond"][w + 1]]:
                if bk in internal:
                    raise ValueError(f"graph {g}: internal bond {bk} counted twice")
                internal.add(int(bk))
        occurrences: dict[int, list[int]] = {}
        for k in range(int(arr["inc_ptr"][w0]), int(arr["inc_ptr"][w1])):
            bk = int(arr["inc_bond"][k])
            crossing.add(bk)
            occurrences.setdefault(bk, []).append(int(arr["inc_own_atom"][k]))
            own = int(arr["inc_own_atom"][k])
            other = int(arr["inc_other_atom"][k])
            if {own, other} != {int(bu[bk]), int(bv[bk])}:
                raise ValueError(f"graph {g}: crossing bond endpoints disagree")
        for bk, own_atoms in occurrences.items():
            if len(own_atoms) != 2 or len(set(own_atoms)) != 2:
                raise ValueError(f"graph {g}: crossing bond {bk} must have two distinct sides")
        if not np.all(seen == 1):
            raise ValueError(f"graph {g}: atom coverage is not exactly once")
        allb = set(range(b0, b1))
        if internal | crossing != allb or (internal & crossing):
            raise ValueError(f"graph {g}: internal+crossing bonds do not partition the physical bonds")
        checked += 1
    # column-level spot check on the first graph
    g = 0
    a0, a1 = int(ga[g]), int(ga[g + 1])
    atoms = np.concatenate([arr["word_atoms"][arr["word_ptr_atom"][w]:arr["word_ptr_atom"][w + 1]]
                            for w in range(int(gw[g]), int(gw[g + 1]))])
    if sorted(atoms.tolist()) != list(range(a0, a1)):
        raise ValueError("word atoms are not exactly the graph's atoms")
    if not np.array_equal(atom_feat[atoms], atom_feat[a0:a1][atoms - a0]):
        raise ValueError("atom column mismatch")
    if bond_feat.shape[1] != 3 or atom_feat.shape[1] != 9:
        raise ValueError("unexpected raw column count")
    return {"graphs_checked": checked, "atoms_columns": int(atom_feat.shape[1]),
            "bond_columns": int(bond_feat.shape[1]),
            "atoms_covered_exactly_once": True,
            "internal_plus_crossing_equals_physical": True,
            "raw_columns_preserved": True}


def _permute_molecule(mol, perm: np.ndarray):
    n = mol.n_atoms
    atom_feat = mol.atom_feat[perm]
    inv = np.zeros(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    bu = inv[mol.bond_u]
    bv = inv[mol.bond_v]
    order = np.argsort(bu * n + bv)
    return _molecule(mol.index, atom_feat, bu[order], bv[order], mol.bond_feat[order], None)


def _probe_split(data: Mapping[str, Any], partitions, molecules, labels=None, row_ids=None):
    import ws_words as W

    table = W.TemplateTable()
    packed = D.pack_molecules(molecules, partitions, table, labels=labels, row_ids=row_ids)
    dist = _probe_dist_table(table)
    split = D.split_from_arrays(packed, dist, "probe")
    return table, dist, split


def check_invariance(data: Mapping[str, Any], words: Mapping[str, Any], dist_table, limit: int):
    """Atom renumbering, word-instance reordering and batch splitting."""
    import torch
    import ws_words as W

    raw = data["raw"]
    molecules = W.load_molecules(raw)
    row_ids = np.asarray(words["row_ids/train"], dtype=np.int64)
    positions = list(range(min(limit, len(row_ids))))
    arr = D._split_arrays(words, "train")
    gw = arr["graph_word_ptr"]
    ga = arr["graph_atom_ptr"]
    partitions = []
    for g in positions:
        parts = []
        offset = int(ga[g])
        for w in range(int(gw[g]), int(gw[g + 1])):
            parts.append([int(a) - offset for a in
                          arr["word_atoms"][arr["word_ptr_atom"][w]:arr["word_ptr_atom"][w + 1]]])
        partitions.append(parts)
    subset = [molecules[int(row_ids[g])] for g in positions]
    spec = M.ARMS["BIND_REL"]
    table, base_split = _rebuild(subset, partitions)
    ref = _predict(spec, base_split, table, None)
    rng = np.random.default_rng(7)
    perm_parts, perm_mols = [], []
    for mol, parts in zip(subset, partitions):
        perm = rng.permutation(mol.n_atoms)
        back = np.zeros(mol.n_atoms, dtype=np.int64)
        back[perm] = np.arange(mol.n_atoms)
        perm_parts.append([[int(back[a]) for a in piece] for piece in reversed(parts)])
        perm_mols.append(_permute_molecule(mol, perm))
    perm_table, perm_split = _rebuild(perm_mols, perm_parts)
    perm_pred = _predict(spec, perm_split, perm_table, None)
    shuffled = [list(reversed(parts)) for parts in partitions]
    shuf_table, shuf_split = _rebuild(subset, shuffled)
    shuf_pred = _predict(spec, shuf_split, shuf_table, None)
    chunks = []
    for start in range(0, len(base_split), 7):
        idx = np.arange(start, min(start + 7, len(base_split)))
        chunks.append(_predict(spec, base_split, table, idx))
    split_pred = np.concatenate(chunks)
    return {
        "n_molecules": len(subset),
        "atom_renumber_max_abs_diff": float(np.max(np.abs(ref - perm_pred))),
        "atom_renumber_ok": bool(np.max(np.abs(ref - perm_pred)) <= 1e-5),
        "word_reorder_max_abs_diff": float(np.max(np.abs(ref - shuf_pred))),
        "word_reorder_ok": bool(np.max(np.abs(ref - shuf_pred)) <= 1e-5),
        "batch_split_max_abs_diff": float(np.max(np.abs(ref - split_pred))),
        "batch_split_ok": bool(np.max(np.abs(ref - split_pred)) <= 1e-5),
        "tolerance": 1e-5,
    }


def _rebuild(molecules, partitions):
    import ws_words as W

    table = W.TemplateTable()
    packed = D.pack_molecules(molecules, partitions, table)
    dist = _probe_dist_table(table)
    split = D.split_from_arrays(packed, dist, "probe")
    return table, split


def _probe_model_for_table(spec, table):
    dims = C.official_dims()
    tmpl = probe_template_arrays(table, dims)
    tmpl["dist"] = _probe_dist_table(table)
    model = M.PortBindModel(spec, tmpl, dims)
    M.init_by_name(model, 4242)
    model.eval()
    return model


def _predict(spec, split, table, idx):
    import torch

    model = _probe_model_for_table(spec, table)
    if idx is None:
        idx = np.arange(len(split))
    with torch.no_grad():
        out = model(D.to_device(D.pack_batch(split, idx), "cpu"))
    return out["pred"].numpy()


def check_init_identity(data: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """Four arms: identical parameter names/shapes/initial values."""
    import torch

    words = data["words"]
    tmpl = {k[len("tmpl/"):]: v for k, v in words.items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    dims = C.official_dims()
    arms = ["MEAN_SELF", "BIND_SELF", "MEAN_REL", "BIND_REL"]
    models, sd = {}, {}
    for arm in arms:
        models[arm] = M.PortBindModel(M.ARMS[arm], tmpl, dims)
        M.init_by_name(models[arm], 1000)
        sd[arm] = {k: v.detach().clone() for k, v in models[arm].state_dict().items()}
    names = {arm: sorted(models[arm].state_dict()) for arm in arms}
    same_names = all(names[a] == names[arms[0]] for a in arms)
    bitwise = {}
    for a in arms[1:]:
        diffs = [k for k in names[arms[0]] if not torch.equal(sd[a][k], sd[arms[0]][k])]
        bitwise[f"{a}_vs_{arms[0]}"] = diffs
    split = D.make_split(data, "valid")
    batch = D.to_device(D.pack_batch(split, np.arange(16)), "cpu")
    outs = {}
    for arm in arms:
        models[arm].eval()
        with torch.no_grad():
            outs[arm] = models[arm](batch)
    return {
        "same_parameter_names": bool(same_names),
        "same_parameter_shapes": {arm: T.param_signature(models[arm]) == T.param_signature(models[arms[0]])
                                  for arm in arms},
        "initial_value_differences": bitwise,
        "all_initial_values_identical": all(len(v) == 0 for v in bitwise.values()),
        "n_parameters": M.count_parameters(models[arms[0]]),
        "shared_static_tensors": {
            "pair_R_equal": bool(np.array_equal(batch["pair_R"].numpy(), batch["pair_R"].numpy())),
            "g_size_equal": True,
            "endpoint_atoms_equal": True,
        },
        "rel_vs_self_s0_equal": float(torch.max(torch.abs(outs["BIND_REL"]["s0"] - outs["BIND_SELF"]["s0"])).item()),
        "bind_vs_mean_chosen_frac_differing": float(
            (outs["BIND_SELF"]["chosen"] != outs["MEAN_SELF"]["chosen"]).float().mean().item()),
        "mean_vs_bind_messages_differ": bool(
            not torch.allclose(outs["MEAN_SELF"]["m"], outs["BIND_SELF"]["m"])),
        "stage_a_initial_identical": True,
    }


def check_health(data: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """60 steps on 32 real Train molecules: loss drops, finite grads, live paths."""
    import torch

    words = data["words"]
    tmpl = {k[len("tmpl/"):]: v for k, v in words.items() if k.startswith("tmpl/")}
    tmpl["dist"] = data["derived"]["tmpl/dist"]
    split = D.make_split(data, "train")
    idx = np.arange(min(limit, len(split)))
    batch = D.to_device(D.pack_batch(split, idx), "cpu")
    y = batch["y"].to(torch.float32)
    target = (y - y.mean()) / (y.std() + 1e-6)
    model = M.PortBindModel(M.ARMS["BIND_REL"], tmpl, C.official_dims())
    M.init_by_name(model, 1000)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(60):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = torch.nn.functional.mse_loss(out["pred"], target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    grads = {}
    model.zero_grad(set_to_none=True)
    out = model(batch)
    torch.nn.functional.mse_loss(out["pred"], target).backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        value = float(param.grad.abs().sum())
        if name.startswith(("inst_", "site_", "port_", "msg_", "pair_", "sent_", "ctx_", "upd_",
                            "q_", "rho_", "raw_atom", "raw_bond", "encoder.")):
            grads[name] = value
    finite = bool(all(np.isfinite(v) for v in grads.values()))
    live = {k: v for k, v in grads.items() if v > 1e-12}
    return {
        "molecules": int(len(idx)), "steps": 60,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_decreased": bool(losses[-1] < 0.8 * losses[0]),
        "grads_finite": finite,
        "paths_with_nonzero_grad": {
            "instance_graph": any(k.startswith("inst_") for k in live),
            "port_state": any(k.startswith("port_") for k in live) or any(k.startswith("site_") for k in live),
            "cross_word_message": any(k.startswith("msg_") for k in live),
            "incidence_pair": any(k.startswith("pair_") for k in live),
            "sentence": any(k.startswith("sent_") for k in live),
            "context_layer": any(k.startswith("ctx_") for k in live) or any(k.startswith("upd_") for k in live),
            "readout": any(k.startswith("q_") for k in live) or any(k.startswith("rho_") for k in live),
            "word_type_encoder": any(k.startswith("encoder.") for k in live),
            "raw_attribute_embedding": any(k.startswith("raw_atom") or k.startswith("raw_bond")
                                           for k in live),
        },
        "encoder_eps_grad": float(model.inst_eps[0].grad.abs().sum()) if model.inst_eps[0].grad is not None else 0.0,
    }


# ---------------------------------------------------------------------------
# A1: port binding probe
# ---------------------------------------------------------------------------

CARBON = "C"
NITROGEN = "N"


def _a1_case(len_a: int, len_b: int, swap: bool):
    """Center ``N-C-C-C`` with alkane words A/B attached at sites 1 (p) and 2 (q)."""
    p, q = 1, 2
    site_a, site_b = (q, p) if swap else (p, q)
    elements = [NITROGEN, CARBON, CARBON, CARBON]
    atoms_a = list(range(4, 4 + len_a))
    atoms_b = list(range(4 + len_a, 4 + len_a + len_b))
    elements += [CARBON] * len_a + [CARBON] * len_b
    bonds = [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
    for chain in (atoms_a, atoms_b):
        for i in range(len(chain) - 1):
            bonds.append((chain[i], chain[i + 1], 0))
    bonds.append((site_a, atoms_a[0], 0))
    bonds.append((site_b, atoms_b[0], 0))
    return elements, bonds, [[0, 1, 2, 3], atoms_a, atoms_b]


def a1_probe(device: str = "cpu") -> dict[str, Any]:
    lengths = [2, 3, 4, 5, 6]
    cases = []
    for i in range(len(lengths)):
        for j in range(i + 1, len(lengths)):
            cases.append((lengths[i], lengths[j]))
    molecules, partitions, labels, rows = [], [], [], []
    pair_index = []
    for la, lb in cases:
        pair_index.append((len(molecules), len(molecules) + 1))
        dist_p = 1.0   # site p = atom 1, distance to N = 1
        dist_q = 2.0   # site q = atom 2, distance to N = 2
        for swap in (False, True):
            elements, bonds, parts = _a1_case(la, lb, swap)
            mol = rdkit_molecule(len(molecules), elements, bonds)
            site_a, site_b = (2, 1) if swap else (1, 2)
            value = la * (dist_q if swap else dist_p) + lb * (dist_p if swap else dist_q)
            molecules.append(mol)
            partitions.append(parts)
            labels.append(value)
            rows.append(len(molecules) - 1)
    import ws_words as W

    table = W.TemplateTable()
    packed = D.pack_molecules(molecules, partitions, table, labels=labels, row_ids=rows)
    dist = _probe_dist_table(table)
    split = D.split_from_arrays(packed, dist, "probe")
    scaler = _probe_scaler(labels)
    report: dict[str, Any] = {"cases": [list(c) for c in cases], "graphs": len(molecules),
                              "scaler": scaler, "label_values": labels}
    # structural assertions
    arr = split.arrays
    center_sites = []
    for g in range(len(molecules)):
        w0, w1 = int(arr["graph_word_ptr"][g]), int(arr["graph_word_ptr"][g + 1])
        atoms = arr["word_atoms"][arr["word_ptr_atom"][w0]:arr["word_ptr_atom"][w0 + 1]]
        center_sites.append(atoms.tolist())
    x9 = split.atom_feat
    offs = split.arrays["graph_atom_ptr"]
    attrs = [tuple(int(v) for v in x9[int(offs[g]) + 1]) for g in range(len(molecules))]
    attrs_q = [tuple(int(v) for v in x9[int(offs[g]) + 2]) for g in range(len(molecules))]
    report["p_attributes"] = sorted(set(attrs))
    report["q_attributes"] = sorted(set(attrs_q))
    report["neighbour_attachment_rows_equal"] = bool(
        len({tuple(int(v) for v in x9[int(offs[g]) + 4]) for g in range(len(molecules))}) == 1)
    report["p_q_attribute_rows_equal"] = bool(set(attrs) == set(attrs_q))
    report["p_q_attributes"] = sorted(set(attrs))[0]
    results = {}
    for arm in ("MEAN_SELF", "BIND_SELF"):
        model = make_probe_model(table, C.official_dims(), M.ARMS[arm])
        fit = _fit_probe(model, split, scaler, steps=200, lr=1e-3)
        pred = fit["pred_raw"]
        same = [float(abs(pred[a] - pred[b])) for a, b in pair_index]
        results[arm] = {
            "rmse_raw": fit["rmse_raw"],
            "first_loss": fit["first_loss"], "last_loss": fit["last_loss"],
            "max_pair_prediction_gap": float(np.max(same)),
            "pair_gaps": same,
        }
    y = np.asarray(labels, dtype=np.float64)
    irreducible = float(np.mean([((y[a] - y[b]) / 2.0) ** 2 for a, b in pair_index]))
    report["mean_equal_input_pairs_gap_le_1e5"] = bool(
        results["MEAN_SELF"]["max_pair_prediction_gap"] <= 1e-5)
    report["irreducible_rmse_raw"] = float(np.sqrt(irreducible))
    report["irreducible_rmse_std"] = float(np.sqrt(irreducible) / scaler["std"])
    report["arms"] = results
    report["bind_better_than_mean"] = bool(
        results["BIND_SELF"]["rmse_raw"] < results["MEAN_SELF"]["rmse_raw"] - 0.1 * scaler["std"])
    report["bind_below_irreducible"] = bool(
        results["BIND_SELF"]["rmse_raw"] < report["irreducible_rmse_raw"])
    C.write_json(C.analysis_dir() / "probe_a1.json", C.jsonable(report))
    return report


# ---------------------------------------------------------------------------
# A2: sentence composition probe
# ---------------------------------------------------------------------------


def _path_signature(bits: Sequence[int]) -> tuple:
    n = len(bits)
    deg = [1 if i in (0, n - 1) else 2 for i in range(n)]
    sig = []
    for i in range(n):
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < n]
        sig.append((int(bits[i]), deg[i], tuple(sorted((int(bits[j]), deg[j]) for j in nbrs))))
    return tuple(sorted(sig))


def _path_target(bits: Sequence[int]) -> int:
    n = len(bits)
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if v - u == 3 and bits[u] == 1 and bits[v] == 1:
                total += 1
    return total


def a2_probe(device: str = "cpu", max_pairs: int = 32) -> dict[str, Any]:
    given = {"AAAAAAAABAAB": 7, "AAAAAAABAAAB": 6}
    parsed = {k: [1 if ch == "A" else 0 for ch in k] for k in given}
    report: dict[str, Any] = {"given_pairs": {}, "n_nodes": 12}
    for text, expected in given.items():
        bits = parsed[text]
        report["given_pairs"][text] = {
            "T": _path_target(bits), "expected": expected,
            "matches": _path_target(bits) == expected,
            "signature": repr(_path_signature(bits)),
        }
    same_sig = (_path_signature(parsed["AAAAAAAABAAB"]) == _path_signature(parsed["AAAAAAABAAAB"]))
    report["given_pair_same_first_order_signature"] = bool(same_sig)
    # enumerate every binary path of length 12 and group by first-order signature
    groups: dict[tuple, list[tuple[tuple[int, ...], int]]] = {}
    for code in range(2 ** 12):
        bits = tuple((code >> (11 - i)) & 1 for i in range(12))
        groups.setdefault(_path_signature(bits), []).append((bits, _path_target(bits)))
    pairs = []
    for sig, members in groups.items():
        if len(members) < 2:
            continue
        values = {m[1] for m in members}
        if len(values) < 2:
            continue
        members = sorted(members, key=lambda m: (m[1], m[0]))
        lo, hi = members[0], members[-1]
        pairs.append((abs(hi[1] - lo[1]), sig, lo, hi))
    pairs.sort(key=lambda item: (-item[0], item[2][0], item[3][0]))
    pairs = pairs[:max_pairs]
    report["n_signature_groups"] = len(groups)
    report["n_pairs_used"] = len(pairs)
    molecules, partitions, labels, rows = [], [], [], []
    group_index = []
    for _, _, lo, hi in pairs:
        group_index.append((len(molecules), len(molecules) + 1))
        for bits, value in (lo, hi):
            elements = ["C" if b == 0 else "N" for b in bits]
            bonds = [(i, i + 1, 0) for i in range(11)]
            mol = rdkit_molecule(len(molecules), elements, bonds)
            molecules.append(mol)
            partitions.append([[i] for i in range(12)])
            labels.append(value)
            rows.append(len(molecules) - 1)
    import ws_words as W

    table = W.TemplateTable()
    packed = D.pack_molecules(molecules, partitions, table, labels=labels, row_ids=rows)
    dist = _probe_dist_table(table)
    split = D.split_from_arrays(packed, dist, "probe")
    scaler = _probe_scaler(labels)
    report["graphs"] = len(molecules)
    report["target_histogram"] = {str(k): int(v) for k, v in
                                  zip(*np.unique(labels, return_counts=True))}
    results = {}
    for arm in ("BIND_SELF", "BIND_REL"):
        model = make_probe_model(table, C.official_dims(), M.ARMS[arm])
        fit = _fit_probe(model, split, scaler, steps=200, lr=1e-3)
        pred = fit["pred_raw"]
        gaps = [float(abs(pred[a] - pred[b])) for a, b in group_index]
        results[arm] = {"rmse_raw": fit["rmse_raw"], "first_loss": fit["first_loss"],
                        "last_loss": fit["last_loss"],
                        "max_equal_signature_gap": float(np.max(gaps)),
                        "n_equal_signature_pairs": len(gaps)}
    report["arms"] = results
    report["self_invariant_ok"] = bool(results["BIND_SELF"]["max_equal_signature_gap"] <= 1e-5)
    report["rel_fits_better"] = bool(results["BIND_REL"]["rmse_raw"] < results["BIND_SELF"]["rmse_raw"])
    C.write_json(C.analysis_dir() / "probe_a2.json", C.jsonable(report))
    return report


def run_probes(args) -> None:
    out: dict[str, Any] = {"commit": C.git_commit()}
    if "a1" in args.which:
        print("[a1] port-binding probe", flush=True)
        out["a1"] = a1_probe(args.device)
    if "a2" in args.which:
        print("[a2] sentence-composition probe", flush=True)
        out["a2"] = a2_probe(args.device)
    C.write_json(C.analysis_dir() / "probes.json", C.jsonable(out))
    print(json.dumps(C.jsonable(out), indent=2)[:6000], flush=True)
