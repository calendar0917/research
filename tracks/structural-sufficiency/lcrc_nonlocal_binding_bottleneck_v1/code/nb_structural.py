"""Phase A -- constructive structural blind-spot witness (CPU, label-free).

Two 18-heavy-atom single rings of SINGLE bonds, one O and one N:

* **A**: atom 0 = O, atom 6 = N, the other 16 atoms are C.
* **B**: atom 0 = O, atom 8 = N, the other 16 atoms are C.

Both are ``C16H33NO``; the N-O shortest-path distance is 6 in A and 8 in B, and
every atom has an identical radius-2 labelled environment multiset.  The two
molecules are therefore *not* isomorphic, but ECFP4 (2048-bit, radius 2) and the
frozen 42-d ``lipo_v1`` global block cannot separate them.  The question this
phase answers is narrower: **can the production A0/C1 forward pass separate
them?**  If not, the round has a constructive witness of an output-level
collision, and the type-distance statistics of Section B do separate them.

Nothing here reads a property label.  No model is trained.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import nb_common as C

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import nb_features as F

import dissect_model as DM
import lcrc_data as PREP_DATA
import mb2_rdkit as RDK
import ptb_model as PM


# ---------------------------------------------------------------------------
# molecule construction
# ---------------------------------------------------------------------------


def build_ring_molecule(nitrogen_index: int, *, n_atoms: int = C.RING_SIZE, oxygen_index: int = C.OXYGEN_INDEX):
    """Build the frozen 18-ring directly in index order (O at 0, N as asked)."""
    mol = Chem.RWMol()
    for i in range(int(n_atoms)):
        symbol = "O" if i == int(oxygen_index) else ("N" if i == int(nitrogen_index) else "C")
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(False)
        mol.AddAtom(atom)
    for i in range(int(n_atoms)):
        mol.AddBond(i, (i + 1) % int(n_atoms), Chem.BondType.SINGLE)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    return out


def molecule_record(mol, name: str) -> dict[str, Any]:
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    charges = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
    explicit_h = [int(a.GetNumExplicitHs()) for a in mol.GetAtoms()]
    bonds = [
        [int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), str(b.GetBondType())] for b in mol.GetBonds()
    ]
    dm = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float64)
    n_idx = [i for i, s in enumerate(symbols) if s == "N"]
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]
    if len(n_idx) != 1 or len(o_idx) != 1:
        raise ValueError(f"{name}: expected exactly one N and one O, got N={n_idx} O={o_idx}")
    return {
        "name": name,
        "construction_nitrogen_index": int(n_idx[0]),
        "construction_oxygen_index": int(o_idx[0]),
        "n_heavy_atoms": int(mol.GetNumAtoms()),
        "n_bonds": int(mol.GetNumBonds()),
        "symbols": symbols,
        "formal_charges": charges,
        "explicit_hydrogens": explicit_h,
        "atom_counts": {s: symbols.count(s) for s in sorted(set(symbols))},
        "bonds": bonds,
        "bond_types": sorted({str(b.GetBondType()) for b in mol.GetBonds()}),
        "canonical_smiles": str(Chem.MolToSmiles(mol)),
        "formula": str(rdMolDescriptors.CalcMolFormula(mol)),
        "degree_sequence": sorted(int(a.GetDegree()) for a in mol.GetAtoms()),
        "total_hydrogens": int(sum(a.GetTotalNumHs() for a in mol.GetAtoms())),
        "distance_matrix": dm.astype(np.int64).tolist(),
        "n_o_shortest_distance": int(round(float(dm[n_idx[0], o_idx[0]]))),
    }


# ---------------------------------------------------------------------------
# isomorphism (explicit labelled backtracking, independent of SMILES strings)
# ---------------------------------------------------------------------------


def labelled_isomorphic(mol_a, mol_b, *, max_nodes: int = 400_000) -> dict[str, Any]:
    """Exact labelled-graph isomorphism test (element labels + bond types).

    A full backtracking search with degree/label pruning.  The result is a proof
    obligation, so the routine does not stop at the first mismatch; it exhausts
    the search space (bounded by ``max_nodes``) and reports the number of nodes.
    """
    na, nb = mol_a.GetNumAtoms(), mol_b.GetNumAtoms()
    if na != nb:
        return {"isomorphic": False, "reason": "atom_count_differs", "search_nodes": 0}
    label_a = [a.GetSymbol() for a in mol_a.GetAtoms()]
    label_b = [a.GetSymbol() for a in mol_b.GetAtoms()]
    if sorted(label_a) != sorted(label_b):
        return {"isomorphic": False, "reason": "label_multiset_differs", "search_nodes": 0}

    def adj(mol):
        mat: list[dict[int, str]] = [dict() for _ in range(mol.GetNumAtoms())]
        for bond in mol.GetBonds():
            u, v = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
            mat[u][v] = str(bond.GetBondType())
            mat[v][u] = str(bond.GetBondType())
        return mat

    adj_a, adj_b = adj(mol_a), adj(mol_b)
    deg_a = [len(x) for x in adj_a]
    deg_b = [len(x) for x in adj_b]
    if sorted(deg_a) != sorted(deg_b):
        return {"isomorphic": False, "reason": "degree_sequence_differs", "search_nodes": 0}

    # Most-constrained-first ordering: high degree, then high label rarity.
    order = sorted(range(na), key=lambda i: (-deg_a[i], label_a[i], i))
    mapping = [-1] * na
    used = [False] * nb
    nodes = 0

    def candidates(pos: int):
        i = order[pos]
        for j in range(nb):
            if used[j] or label_b[j] != label_a[i] or deg_b[j] != deg_a[i]:
                continue
            ok = True
            for (u, bt) in adj_a[i].items():
                uu = mapping[u]
                if uu < 0:
                    continue
                if adj_b[j].get(uu) != bt:
                    ok = False
                    break
            if ok:
                yield j

    result = {"isomorphic": False, "reason": "search_exhausted", "search_nodes": 0}

    def backtrack(pos: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            result["reason"] = "search_budget_exhausted"
            return False
        if pos == na:
            result["isomorphic"] = True
            result["reason"] = "match_found"
            result["witness_mapping"] = [mapping[order[k]] for k in range(na)]
            return True
        i = order[pos]
        for j in candidates(pos):
            mapping[i] = j
            used[j] = True
            if backtrack(pos + 1):
                return True
            used[j] = False
            mapping[i] = -1
        return False

    found = backtrack(0)
    result["search_nodes"] = int(nodes)
    result["found"] = bool(found)
    return result


# ---------------------------------------------------------------------------
# radius-2 local environment multiset (RDKit Morgan environments)
# ---------------------------------------------------------------------------


def rooted_environment_multiset(mol, *, radius: int = C.MORGAN_RADIUS) -> dict[str, Any]:
    """Multiset of per-atom radius<=2 labelled environments.

    Uses RDKit's Morgan environment identifiers (32-bit hashes of the atom/bond
    label multiset) with ``bitInfo`` attribution, so each root atom gets its own
    ``{(environment_id, radius)}`` signature.  Two molecules with the same
    multiset of root signatures have the same radius-2 local structure up to the
    standard Morgan 32-bit environment hash.
    """
    info: dict[int, list[tuple[int, int]]] = {}
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=int(radius), bitInfo=info, useChirality=False)
    per_root: list[list[tuple[int, int]]] = [[] for _ in range(mol.GetNumAtoms())]
    for key, occurrences in info.items():
        for (atom_idx, r) in occurrences:
            per_root[int(atom_idx)].append((int(key), int(r)))
    signatures = [tuple(sorted(rows)) for rows in per_root]
    counts: dict[str, int] = {}
    for sig in signatures:
        counts[repr(sig)] = counts.get(repr(sig), 0) + 1
    return {
        "radius": int(radius),
        "n_roots": int(mol.GetNumAtoms()),
        "per_root_signatures": [list(sig) for sig in signatures],
        "signature_multiset": dict(sorted(counts.items())),
        "environment_counts": {str(k): int(v) for k, v in sorted(dict(fp.GetNonzeroElements()).items())},
    }


def morgan_2048_bits(mol, *, radius: int = C.MORGAN_RADIUS, n_bits: int = C.MORGAN_NBITS) -> np.ndarray:
    """The frozen round's 2048-bit Morgan radius-2 fingerprint (no chirality)."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits), useChirality=False)
    out = np.zeros(int(n_bits), dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, out)
    return out


# ---------------------------------------------------------------------------
# production lipo_v1 tensors
# ---------------------------------------------------------------------------


def production_tensors(canonical_smiles: str, spec) -> dict[str, Any]:
    """Frozen path: SMILES -> raw attributes -> lipo_v1 graph -> LCRC arrays."""
    raw = RDK.smiles_to_raw_row(canonical_smiles, 0)
    graph = RDK.raw_row_to_graph(raw, 0, spec)
    mol = PREP_DATA.build_molecule(graph, spec)
    return {"raw": raw, "graph": graph, "mol": mol}


ROW_TENSORS = (
    "atom_feat",
    "atom_type",
    "atom_scalar",
    "bond_feat",
    "bond_type",
    "bond_index",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "nbr_code",
)


def row_multiset_equal(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"equal": False, "reason": "shape", "shape_a": list(a.shape), "shape_b": list(b.shape)}
    ra = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(-1, 1)
    rb = b.reshape(b.shape[0], -1) if b.ndim > 1 else b.reshape(-1, 1)
    order_a = np.lexsort(ra.T[::-1])
    order_b = np.lexsort(rb.T[::-1])
    return {
        "equal": bool(np.array_equal(ra[order_a], rb[order_b])),
        "n_rows": int(ra.shape[0]),
        "row_order_identical": bool(np.array_equal(ra, rb)),
    }


def tensors_observation(mol_a: Mapping[str, Any], mol_b: Mapping[str, Any]) -> dict[str, Any]:
    """Informational: which production per-row tensors coincide as multisets."""
    out: dict[str, Any] = {}
    for key in ROW_TENSORS:
        out[key] = row_multiset_equal(np.asarray(mol_a[key]), np.asarray(mol_b[key]))
    out["global"] = {
        "equal": bool(np.array_equal(np.asarray(mol_a["global"]), np.asarray(mol_b["global"]))),
        "max_abs_diff": float(np.abs(np.asarray(mol_a["global"], dtype=np.float64)
                                     - np.asarray(mol_b["global"], dtype=np.float64)).max()),
    }
    return out


# ---------------------------------------------------------------------------
# LCRC forward
# ---------------------------------------------------------------------------


def lipo_train_norm() -> dict[str, Any]:
    """The frozen Lipophilicity Train-only structural normalisation.

    Recomputed from the frozen structure cache (structures only; the frozen
    loader never places a label in a cache) and gated on the fingerprint that the
    executed dissection/PTB runs recorded.
    """
    import dissect_source as S

    spec = C.frozen_spec()
    src = S.load_lipop(C.LIPO_REPEAT, labels="none")
    cache = MD_load_train_cache(C.LIPO_REPEAT, src, spec)
    norm = PREP_DATA.fit_norm(cache, spec, source=f"Lipophilicity_r{C.LIPO_REPEAT}_train_structures_only")
    fingerprint = PREP_DATA.norm_fingerprint(norm)
    return {
        "norm": norm,
        "fingerprint": fingerprint,
        "expected_fingerprint": C.LIPO_TRAIN_NORM_FINGERPRINT,
        "matches_executed_run": bool(fingerprint == C.LIPO_TRAIN_NORM_FINGERPRINT),
        "cache_fingerprint": str(cache["meta"].get("content_fingerprint")),
        "expected_cache_fingerprint": C.LIPO_TRAIN_CACHE_FINGERPRINT,
        "cache_matches_executed_run": bool(str(cache["meta"].get("content_fingerprint")) == C.LIPO_TRAIN_CACHE_FINGERPRINT),
        "n_train_structures": int(cache["n_molecules"]),
        "labels_present": bool(cache["meta"].get("labels_present")),
    }


def MD_load_train_cache(repeat: int, src: Mapping[str, Any], spec) -> Mapping[str, Any]:
    import dissect_data as DD

    return DD.load_or_build_split_cache(
        int(repeat), "train", src["rows"]["train"], src["smiles"], spec, rebuild=False
    )


def two_molecule_cache(payloads: Sequence[Mapping[str, Any]], spec) -> dict[str, Any]:
    return PREP_DATA.concat_molecules(list(payloads), spec, split="phase_a_witness")


def forward_record(
    model,
    cache: Mapping[str, Any],
    rows: Sequence[int],
    *,
    spec,
) -> dict[str, Any]:
    batch = PREP_DATA.collate(cache, list(rows))
    model.eval()
    with torch.no_grad():
        y, aux = model(batch, return_aux=True)
    h = aux["h"].detach().cpu().numpy().astype(np.float64)
    atom_counts = batch["atom_counts"].detach().cpu().numpy().astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(atom_counts)]).astype(np.int64)
    h_graphs = [h[int(offs[i]) : int(offs[i + 1])] for i in range(len(atom_counts))]
    local_input = aux.get("local_input")
    local_graphs = None
    if local_input is not None:
        li = local_input.detach().cpu().numpy().astype(np.float64)
        local_graphs = [li[int(offs[i]) : int(offs[i + 1])] for i in range(len(atom_counts))]
    out = {
        "y": y.detach().cpu().numpy().astype(np.float64),
        "contrib": aux["contrib"].detach().cpu().numpy().astype(np.float64),
        "rho": aux["rho"].detach().cpu().numpy().astype(np.float64),
        "mean_h": aux["mean_h"].detach().cpu().numpy().astype(np.float64),
        "mean_h2": aux["mean_h2"].detach().cpu().numpy().astype(np.float64),
        "h": h,
        "h_graphs": h_graphs,
        "local_input_graphs": local_graphs,
        "global_raw": batch["global"].detach().cpu().numpy().astype(np.float64),
        "atom_counts": atom_counts,
    }
    return out


def minimum_cost_row_matching(h_a: np.ndarray, h_b: np.ndarray) -> dict[str, Any]:
    """Minimum-cost bipartite matching between two equal-size sets of ``h`` rows.

    Atom numbering is arbitrary, so the rows are compared as a multiset: the
    optimal assignment under the L1 cost is found with the Hungarian algorithm.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(h_a, dtype=np.float64)
    b = np.asarray(h_b, dtype=np.float64)
    if a.shape != b.shape:
        return {"comparable": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    cost = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    matched = cost[rows, cols]
    return {
        "comparable": True,
        "total_cost": float(matched.sum()),
        "max_matched_cost": float(matched.max()),
        "identical": bool(np.all(matched == 0.0)),
        "permutation": [int(c) for c in cols],
        "exact_multiset_equal": bool(np.array_equal(np.sort(a, axis=0), np.sort(b, axis=0))),
    }


def build_witness_models(spec, norm: Mapping[str, Any]):
    models = {}
    for variant in C.FORWARD_VARIANTS:
        for seed in C.FORWARD_SEEDS:
            model, info = PM.build_model(spec, norm, variant, int(seed), torch.device("cpu"))
            model.eval()
            for param in model.parameters():
                if param.dtype != torch.float32:
                    raise ValueError("model is not float32")
            models[(variant, int(seed))] = {"model": model, "info": info}
    return models


# ---------------------------------------------------------------------------
# phase A driver
# ---------------------------------------------------------------------------


def run_phase_a(*, renumber_seed: int = 0) -> dict[str, Any]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    spec = C.frozen_spec()

    mol_a = build_ring_molecule(C.NITROGEN_INDEX_A)
    mol_b = build_ring_molecule(C.NITROGEN_INDEX_B)
    rec_a = molecule_record(mol_a, "A")
    rec_b = molecule_record(mol_b, "B")

    checks: dict[str, Any] = {}
    checks["sanitize_ok"] = {
        "A": bool(mol_a is not None and mol_a.GetNumAtoms() == C.RING_SIZE),
        "B": bool(mol_b is not None and mol_b.GetNumAtoms() == C.RING_SIZE),
    }
    checks["formula"] = {
        "A": rec_a["formula"],
        "B": rec_b["formula"],
        "expected": C.EXPECTED_FORMULA,
        "pass": bool(rec_a["formula"] == rec_b["formula"] == C.EXPECTED_FORMULA),
    }
    checks["canonical_smiles"] = {
        "A": rec_a["canonical_smiles"],
        "B": rec_b["canonical_smiles"],
        "expected_A": C.EXPECTED_SMILES_A,
        "expected_B": C.EXPECTED_SMILES_B,
        "pass": bool(
            rec_a["canonical_smiles"] == C.EXPECTED_SMILES_A and rec_b["canonical_smiles"] == C.EXPECTED_SMILES_B
        ),
    }
    iso = labelled_isomorphic(mol_a, mol_b)
    checks["labelled_isomorphism"] = {
        "isomorphic": bool(iso.get("isomorphic", False)),
        "reason": iso.get("reason"),
        "search_nodes": iso.get("search_nodes"),
        "pass": bool(not iso.get("isomorphic", True)),
        "note": (
            "exhaustive labelled backtracking: node labels are element symbols, edge labels are bond types; "
            "no SMILES string is compared"
        ),
        "cyclic_distance_invariant": {
            "A": rec_a["n_o_shortest_distance"],
            "B": rec_b["n_o_shortest_distance"],
            "argument": (
                "both graphs are a single 18-cycle, so every graph isomorphism is a dihedral symmetry of the "
                "cycle and preserves the cyclic N-O distance; 6 != 8 proves non-isomorphism independently"
            ),
        },
    }

    env_a = rooted_environment_multiset(mol_a)
    env_b = rooted_environment_multiset(mol_b)
    checks["radius2_local_multiset"] = {
        "identical": bool(env_a["signature_multiset"] == env_b["signature_multiset"]),
        "n_distinct_root_signatures": len(env_a["signature_multiset"]),
        "pass": bool(env_a["signature_multiset"] == env_b["signature_multiset"]),
    }

    fp_a = morgan_2048_bits(mol_a)
    fp_b = morgan_2048_bits(mol_b)
    checks["morgan_2048_equal"] = {
        "identical": bool(np.array_equal(fp_a, fp_b)),
        "n_on_bits_A": int(fp_a.sum()),
        "n_on_bits_B": int(fp_b.sum()),
        "pass": bool(np.array_equal(fp_a, fp_b)),
    }
    checks["n_o_distance"] = {
        "A": rec_a["n_o_shortest_distance"],
        "B": rec_b["n_o_shortest_distance"],
        "expected": [C.EXPECTED_NO_DISTANCE_A, C.EXPECTED_NO_DISTANCE_B],
        "pass": bool(
            [rec_a["n_o_shortest_distance"], rec_b["n_o_shortest_distance"]]
            == [C.EXPECTED_NO_DISTANCE_A, C.EXPECTED_NO_DISTANCE_B]
        ),
        "source": "rdkit.Chem.GetDistanceMatrix on the sanitised molecule",
    }

    # ---- production lipo_v1 tensors --------------------------------------
    tensors_a = production_tensors(rec_a["canonical_smiles"], spec)
    tensors_b = production_tensors(rec_b["canonical_smiles"], spec)
    mol_a_p, mol_b_p = tensors_a["mol"], tensors_b["mol"]
    g_a = np.asarray(mol_a_p["global"], dtype=np.float32)
    g_b = np.asarray(mol_b_p["global"], dtype=np.float32)
    checks["global42_identical"] = {
        "identical": bool(np.array_equal(g_a, g_b)),
        "max_abs_diff": float(np.abs(g_a.astype(np.float64) - g_b.astype(np.float64)).max()),
        "values_A": g_a.tolist(),
        "values_B": g_b.tolist(),
        "n_differing_columns": int((g_a != g_b).sum()),
        "pass": bool(np.array_equal(g_a, g_b)),
    }
    input_obs = tensors_observation(mol_a_p, mol_b_p)

    # ---- real type-distance features (Section B definition) --------------
    pair_a = F.pair_feature_molecule(rec_a["canonical_smiles"])
    pair_b = F.pair_feature_molecule(rec_b["canonical_smiles"])
    def nonzero(vec):
        v = np.asarray(vec)
        return [
            {
                "type_pair": list(C.TYPE_PAIRS[p]),
                "type_pair_names": [C.ELEMENT_CLASSES[C.TYPE_PAIRS[p][0]], C.ELEMENT_CLASSES[C.TYPE_PAIRS[p][1]]],
                "distance_bin": C.DIST_BIN_LABELS[k],
                "value": float(v[p * C.N_DIST_BINS + k]),
            }
            for p in range(C.N_TYPE_PAIRS)
            for k in range(C.N_DIST_BINS)
            if v[p * C.N_DIST_BINS + k] != 0
        ]

    no_pair = (C.ELEMENT_CLASSES.index("N"), C.ELEMENT_CLASSES.index("O"))
    no_pair_idx = C.TYPE_PAIRS.index(no_pair)
    checks["type_distance"] = {
        "real_differ": bool(not np.array_equal(pair_a["real"], pair_b["real"])),
        "real_l1_distance": float(np.abs(pair_a["real"] - pair_b["real"]).sum()),
        "null_identical": bool(np.array_equal(pair_a["null"], pair_b["null"])),
        "null_max_abs_diff": float(np.abs(pair_a["null"] - pair_b["null"]).max()),
        "n_o_pair_index": int(no_pair_idx),
        "n_o_bins": {
            "A": {
                label: float(pair_a["counts"][no_pair_idx, k]) for k, label in enumerate(C.DIST_BIN_LABELS)
            },
            "B": {
                label: float(pair_b["counts"][no_pair_idx, k]) for k, label in enumerate(C.DIST_BIN_LABELS)
            },
        },
        "n_o_expected_bin": {
            "A": C.DIST_BIN_LABELS[F.distance_bin(C.EXPECTED_NO_DISTANCE_A)],
            "B": C.DIST_BIN_LABELS[F.distance_bin(C.EXPECTED_NO_DISTANCE_B)],
        },
        "n_o_present_in_expected_bin": {
            "A": bool(pair_a["counts"][no_pair_idx, F.distance_bin(C.EXPECTED_NO_DISTANCE_A)] == 1),
            "B": bool(pair_b["counts"][no_pair_idx, F.distance_bin(C.EXPECTED_NO_DISTANCE_B)] == 1),
        },
        "nonzero_A": nonzero(pair_a["real"]),
        "nonzero_B": nonzero(pair_b["real"]),
        "totals": {"A": int(pair_a["total"]), "B": int(pair_b["total"])},
        "row_margin_equal": bool(np.array_equal(pair_a["row_margin"], pair_b["row_margin"])),
        "col_margin_equal": bool(np.array_equal(pair_a["col_margin"], pair_b["col_margin"])),
    }
    checks["type_distance"]["pass"] = bool(
        checks["type_distance"]["real_differ"]
        and checks["type_distance"]["null_identical"]
        and checks["type_distance"]["n_o_present_in_expected_bin"]["A"]
        and checks["type_distance"]["n_o_present_in_expected_bin"]["B"]
    )

    # ---- production forward --------------------------------------------
    norm_info = lipo_train_norm()
    norm = norm_info["norm"]
    cache = two_molecule_cache([mol_a_p, mol_b_p], spec)
    PREP_DATA.validate_cache(cache, spec)

    models = build_witness_models(spec, norm)

    # renumbering invariance: rebuild A through a permuted but equivalent SMILES
    rng = np.random.default_rng(int(renumber_seed))
    perm = rng.permutation(C.RING_SIZE)
    renum_mol = Chem.RenumberAtoms(mol_a, [int(x) for x in perm])
    perm_smiles = str(Chem.MolToSmiles(renum_mol, canonical=False))
    tensors_a_perm = production_tensors(perm_smiles, spec)
    perm_order = [int(a.GetIdx()) for a in Chem.MolFromSmiles(perm_smiles).GetAtoms()]
    cache_perm = two_molecule_cache([tensors_a_perm["mol"]], spec)

    forward: dict[str, Any] = {}
    for (variant, seed), payload in models.items():
        model = payload["model"]
        res = forward_record(model, cache, [0, 1], spec=spec)
        h_a, h_b = res["h_graphs"][0], res["h_graphs"][1]
        match = minimum_cost_row_matching(h_a, h_b)
        local_match = None
        if res["local_input_graphs"] is not None:
            local_match = minimum_cost_row_matching(
                res["local_input_graphs"][0], res["local_input_graphs"][1]
            )
        ren = forward_record(model, cache_perm, [0], spec=spec)
        ren_match = minimum_cost_row_matching(res["h_graphs"][0], ren["h_graphs"][0])
        d_contrib = float(abs(res["contrib"][0] - res["contrib"][1]))
        d_rho = float(abs(res["rho"][0] - res["rho"][1]))
        d_y = float(abs(res["y"][0] - res["y"][1]))
        d_mean_h = float(np.abs(res["mean_h"][0] - res["mean_h"][1]).max())
        d_mean_h2 = float(np.abs(res["mean_h2"][0] - res["mean_h2"][1]).max())
        forward[f"{variant}_s{seed}"] = {
            "variant": variant,
            "seed": int(seed),
            "parameters": int(payload["info"]["parameters"]),
            "init_state_hash": payload["info"]["init_state_hash"],
            "shared_state_equal_to_full": payload["info"]["shared_state_equal_to_full"],
            "dtype": "float32",
            "eval_mode": True,
            "global_raw_identical": bool(np.array_equal(res["global_raw"][0], res["global_raw"][1])),
            "h_row_matching": match,
            "local_input_row_matching": local_match,
            "mean_h_max_abs_diff": d_mean_h,
            "mean_h2_max_abs_diff": d_mean_h2,
            "mean_h_identical": bool(d_mean_h == 0.0),
            "mean_h2_identical": bool(d_mean_h2 == 0.0),
            "mean_h_within_tol": bool(d_mean_h <= C.PREDICTION_TOL),
            "mean_h2_within_tol": bool(d_mean_h2 <= C.PREDICTION_TOL),
            "contrib_abs_diff": d_contrib,
            "rho_abs_diff": d_rho,
            "y_abs_diff": d_y,
            "y_A": float(res["y"][0]),
            "y_B": float(res["y"][1]),
            "y_equal_within_tol": bool(d_y <= C.PREDICTION_TOL),
            "renumber_invariance": {
                "permutation_seed": int(renumber_seed),
                "permutation": [int(x) for x in perm],
                "permuted_smiles": perm_smiles,
                "n_atoms_round_trip": len(perm_order),
                "h_row_matching": ren_match,
                "y_original": float(res["y"][0]),
                "y_renumbered": float(ren["y"][0]),
                "y_abs_diff": float(abs(res["y"][0] - ren["y"][0])),
                "pass": bool(
                    ren_match.get("identical", False) and abs(float(res["y"][0]) - float(ren["y"][0])) <= C.PREDICTION_TOL
                ),
            },
        }

    # gate: every variant/seed must reproduce A == B and be renumber-invariant
    forward_pass = all(
        rec["h_row_matching"].get("identical", False)
        and rec["contrib_abs_diff"] <= C.PREDICTION_TOL
        and rec["rho_abs_diff"] <= C.PREDICTION_TOL
        and rec["y_equal_within_tol"]
        and rec["global_raw_identical"]
        and rec["renumber_invariance"]["pass"]
        for rec in forward.values()
    )
    failures = {
        key: [
            field
            for field, ok in (
                ("h_row_matching", rec["h_row_matching"].get("identical", False)),
                ("contrib", rec["contrib_abs_diff"] <= C.PREDICTION_TOL),
                ("rho", rec["rho_abs_diff"] <= C.PREDICTION_TOL),
                ("y", rec["y_equal_within_tol"]),
                ("global", rec["global_raw_identical"]),
                ("renumber", rec["renumber_invariance"]["pass"]),
            )
            if not ok
        ]
        for key, rec in forward.items()
    }
    failures = {k: v for k, v in failures.items() if v}

    return {
        "round_id": C.ROUND_ID,
        "phase": "A",
        "created_utc": C.utcnow(),
        "labels_read": False,
        "training_performed": False,
        "gpu_used": False,
        "molecules": {"A": rec_a, "B": rec_b},
        "checks": checks,
        "production_input_tensors_observation": input_obs,
        "normalisation": {
            "fingerprint": norm_info["fingerprint"],
            "matches_executed_dissection_run": norm_info["matches_executed_run"],
            "cache_matches_executed_dissection_run": norm_info["cache_matches_executed_run"],
            "n_train_structures": norm_info["n_train_structures"],
            "labels_present": norm_info["labels_present"],
            "source": norm_info["norm"].get("source"),
        },
        "forward": forward,
        "gates": {
            "sanitize": checks["sanitize_ok"]["A"] and checks["sanitize_ok"]["B"],
            "formula": checks["formula"]["pass"],
            "canonical_smiles": checks["canonical_smiles"]["pass"],
            "non_isomorphic": checks["labelled_isomorphism"]["pass"],
            "radius2_multiset_identical": checks["radius2_local_multiset"]["pass"],
            "morgan2048_identical": checks["morgan_2048_equal"]["pass"],
            "n_o_distance": checks["n_o_distance"]["pass"],
            "global42_identical": checks["global42_identical"]["pass"],
            "type_distance_separates": checks["type_distance"]["pass"],
            "forward_equivalent": bool(forward_pass),
        },
        "forward_failures": failures,
        "verdict": {
            "production_forward_blind": bool(forward_pass),
            "type_distance_separates": checks["type_distance"]["pass"],
            "summary": (
                "A0/C1 produce numerically identical outputs for A and B under the frozen Train-only "
                "normalisation, while the type-distance statistics separate them"
                if forward_pass
                else "forward equivalence FAILED -- see forward_failures"
            ),
        },
    }


__all__ = [
    "build_ring_molecule",
    "molecule_record",
    "labelled_isomorphic",
    "rooted_environment_multiset",
    "morgan_2048_bits",
    "production_tensors",
    "tensors_observation",
    "lipo_train_norm",
    "forward_record",
    "minimum_cost_row_matching",
    "run_phase_a",
]
