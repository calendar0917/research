"""structure-context-effect-gate-v1 (Stage B): frozen-model Bond-source replacement response.

Protocol: ``../PROTOCOL.md`` (frozen before any model response was read).

Question: for a frozen BJOC-Lipo-SSL predictor (Arm P soup, seeds 0/1), does the model's
response to replacing one Bond encoding carry a *real context* component beyond ``S``
identity and global composition controls (B0 < B1 < B2), and is that component stable
across reference definitions, model seeds and held-out scaffolds?

Boundaries (enforced by construction, asserted by tests):
* only the frozen label-free Lipophilicity train objects + frozen soup_P checkpoints;
* no valid/test labels, no ZINC test/Locked, no weight updates, no training;
* all results are "frozen-model Bond-source replacement responses", not chemical effects.

Run::

    uv run --group lipo python \
        tracks/structure-contribution/results/structure_context_effect_gate_v1/code/structure_context_effect_gate_v1.py all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
RESULT_DIR = CODE_DIR.parent
REPO_ROOT = CODE_DIR.parents[4]
assert (REPO_ROOT / "AGENT.md").exists(), REPO_ROOT

TRACK_CODE = REPO_ROOT / "tracks/structure-contribution/code"
STAGE_A = REPO_ROOT / "tracks/structure-contribution/results/structure_context_support_v1"
LIPO = REPO_ROOT / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1"
OUT = RESULT_DIR
ANALYSIS = OUT / "analysis"

for _p in (str(TRACK_CODE), str(STAGE_A / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structural_relation_composer_v1 as src  # noqa: E402
import bjoc_lipophilicity_adapter_v1 as lip  # noqa: E402
from structural_relation_objects_v1 import MolObjects  # noqa: E402

# ---------------------------------------------------------------------------
# frozen inputs / constants (PROTOCOL.md §1, §4.2, §5, §6)
# ---------------------------------------------------------------------------

INSTANCES_NPZ = STAGE_A / "analysis/instances.npz"
INSTANCES_SHA256 = "80970087965952baf63821e965418380aca88fc0463286bb3ff6bef10175d327"
STAGE_A_SUMMARY = STAGE_A / "summary.json"
OBJECTS_TRAIN = LIPO / "objects_train.pt"
OBJECTS_TRAIN_SHA256 = "6d681cb0eaaab8c448564993c3740b24204e52e9e430f8c366189aa0061c7c86"
SOUP_SHA256 = {
    0: "1a8292acf0477ce89899d48d913f0c50d651eab00c4d92d757580f847836a304",
    1: "505f840ea90783bdbd6761e2290e4f875670bcdf3f293a8aa5d9225582c542b0",
}
MODEL_SEEDS = (0, 1)
PRIMARY_SEED = 0
SCHEMES = ("anchor", "mean", "random")
PRIMARY_SCHEME = "anchor"
REF_RANDOM_DRAWS = 8
REF_RANDOM_BASE_SEED = 20261017
N_SHUFFLE = 100
BOOTSTRAP_SEED = 20261018
N_BOOTSTRAP = 10000
CV_SPLITS = 5
ZERO_TOL = 1e-6
TARGET_MEAN = 2.162904761904762  # frozen train-only target normalisation
TARGET_SCALE = 1.210992810122257
PREDICTOR_PARAMS = dict(
    max_iter=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=0,
    early_stopping=False,
)
SUPPORT_GATE = dict(min_scaffolds=30, min_cross_scaffold_types=10, min_two_scaffold_groups=30)
ELEMENT_LIST = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si")
ELEMENT_INDEX = {e: i for i, e in enumerate(ELEMENT_LIST)}
N_ELEM = len(ELEMENT_LIST)
N_ATOM_CLASSES = lip.ATOM_UNK + 1  # 15 real classes + UNK
N_BOND_CLASSES = 4
N_G_BASE = 7
CT_DIM = N_ATOM_CLASSES + 1 + N_BOND_CLASSES + 4 + 4 + 2 + 1
HETERO = frozenset({"N", "O", "S", "F", "Cl", "Br", "I", "P"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with Path(path).open() as fh:
        return json.load(fh)


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, sort_keys=False)


def log(msg: str) -> None:
    print(f"[gate] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data loading / verification
# ---------------------------------------------------------------------------


def load_stage_a() -> dict:
    with np.load(INSTANCES_NPZ, allow_pickle=True) as raw:
        return {key: raw[key] for key in raw.files}


def load_objects():
    import torch

    objs = torch.load(OBJECTS_TRAIN, map_location="cpu", weights_only=False)
    assert len(objs) == 3360 and all(isinstance(o, MolObjects) for o in objs)
    assert not any(hasattr(o, "y") for o in objs), "objects must be label-free"
    return objs


def load_model(seed: int):
    import torch

    norm = read_json(LIPO / "standardizer_v1.json")
    model = lip.build_lipo_model(
        lip.LipoConfig(),
        seed=int(seed),
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["mean"], dtype=np.float32),
        global_scale=np.asarray(norm["scale"], dtype=np.float32),
        device="cpu",
    )
    state = torch.load(LIPO / f"soup_P_s{int(seed)}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def verify_frozen_inputs() -> dict:
    out = {
        "instances_npz": {
            "path": str(INSTANCES_NPZ.relative_to(REPO_ROOT)),
            "sha256": sha256_file(INSTANCES_NPZ),
            "expected": INSTANCES_SHA256,
        },
        "objects_train": {
            "path": str(OBJECTS_TRAIN.relative_to(REPO_ROOT)),
            "sha256": sha256_file(OBJECTS_TRAIN),
            "expected": OBJECTS_TRAIN_SHA256,
        },
        "trained_model_soup": {
            f"seed{seed}": {
                "path": str((LIPO / f"soup_P_s{seed}.pt").relative_to(REPO_ROOT)),
                "sha256": sha256_file(LIPO / f"soup_P_s{seed}.pt"),
                "expected": SOUP_SHA256[seed],
            }
            for seed in MODEL_SEEDS
        },
        "standardizer_v1": {
            "path": str((LIPO / "standardizer_v1.json").relative_to(REPO_ROOT)),
            "sha256": sha256_file(LIPO / "standardizer_v1.json"),
        },
        "stage_a_summary": {
            "path": str(STAGE_A_SUMMARY.relative_to(REPO_ROOT)),
            "sha256": sha256_file(STAGE_A_SUMMARY),
        },
        "labels_loaded": False,
        "valid_test_labels_loaded": False,
        "weights_updated": False,
    }
    ok = (
        out["instances_npz"]["sha256"] == INSTANCES_SHA256
        and out["objects_train"]["sha256"] == OBJECTS_TRAIN_SHA256
        and all(v["sha256"] == v["expected"] for v in out["trained_model_soup"].values())
    )
    out["hash_check_passed"] = bool(ok)
    return out


def verify_prediction_replay(seed: int) -> dict:
    """Replay valid predictions with the frozen checkpoint; compare to frozen files (no labels)."""
    import torch

    model = load_model(seed)
    objs = torch.load(LIPO / "objects_valid.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        preds = []
        for chunk in range(0, len(objs), 256):
            batch = lip.batch_lipo_objects(objs[chunk : chunk + 256])
            preds.append(model(batch).view(-1))
        norm_pred = torch.cat(preds).numpy()
    raw = norm_pred * TARGET_SCALE + TARGET_MEAN
    frozen = np.load(LIPO / "predictions" / f"valid_P_s{seed}.npz")["pred"]
    manifest = read_json(LIPO / "prediction_manifest_valid.json")
    rec = next(r for r in manifest["records"] if r["arm"] == "P" and r["seed"] == seed)
    file_sha = sha256_file(LIPO / "predictions" / f"valid_P_s{seed}.npz")
    return {
        "seed": seed,
        "n": int(len(raw)),
        "max_abs_diff": float(np.abs(raw - frozen).max()),
        "file_sha256": file_sha,
        "manifest_sha256": rec["sha256"],
        "sha_match": file_sha == rec["sha256"],
        "pass": bool(np.abs(raw - frozen).max() <= 5e-6 and file_sha == rec["sha256"]),
    }


# ---------------------------------------------------------------------------
# RDKit structure helpers (identical semantics to Stage A)
# ---------------------------------------------------------------------------


def heavy_formula(mol) -> str:
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    return "".join(f"{sym}{counts[sym]}" for sym in sorted(counts))


def murcko_scaffold(mol) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "ACYCLIC"


def ring_system_membership(mol):
    """(rings, system id per atom, system size by id) — same semantics as Stage A."""
    from rdkit import Chem

    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    parent = {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ring in rings:
        root = find(ring[0])
        for atom in ring[1:]:
            other = find(atom)
            if other != root:
                parent[max(root, other)] = min(root, other)
                root = find(root)
    sys_id = {a.GetIdx(): find(a.GetIdx()) for a in mol.GetAtoms()}
    sizes = Counter(sys_id[a] for ring in rings for a in ring)
    return rings, sys_id, sizes


def rdkit_molecules(smiles_list):
    from rdkit import Chem

    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"RDKit could not parse {smi!r}")
        mols.append(mol)
    return mols


def bond_code_from_rdkit(bond) -> int:
    bt = bond.GetBondTypeAsDouble()
    if bt == 1.5:
        return 3
    mapping = {1.0: 0, 2.0: 1, 3.0: 2}
    if bt not in mapping:
        raise ValueError(f"unsupported RDKit bond type {bond.GetBondType()!r}")
    return mapping[bt]


def build_bond_correspondence(objs, mols) -> dict:
    """Verify RDKit graph <-> model Bond objects (endpoints, numbering, bond type)."""
    per_mol = []
    stats = Counter()
    for uid, (obj, mol) in enumerate(zip(objs, mols)):
        assert obj.n_atoms == mol.GetNumAtoms(), (uid, obj.n_atoms, mol.GetNumAtoms())
        stats["molecules_checked"] += 1
        stats["atoms_checked"] += int(obj.n_atoms)
        rd = {}
        for b in mol.GetBonds():
            u, v = int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())
            if u > v:
                u, v = v, u
            rd[(u, v)] = bond_code_from_rdkit(b)
        model_idx = {}
        for k, row in enumerate(np.asarray(obj.bond_index).tolist()):
            key = (int(row[0]), int(row[1]))
            assert row[0] < row[1], (uid, row)
            assert key not in model_idx, (uid, key)
            model_idx[key] = k
        assert set(rd) == set(model_idx), (uid, sorted(set(rd) ^ set(model_idx))[:5])
        for key, code in rd.items():
            assert int(obj.bond_type[model_idx[key]]) == code, (uid, key, code, obj.bond_type[model_idx[key]])
            stats["bonds_checked"] += 1
        for j, (i, c, k) in enumerate(np.asarray(obj.junc_index).tolist()):
            b1, b2 = (int(x) for x in obj.junc_bond[j])
            for b, other in ((b1, i), (b2, k)):
                u, v = obj.bond_index[b]
                assert c in (int(u), int(v)) and other in (int(u), int(v)), (uid, j, b)
            stats["junctions_checked"] += 1
        per_mol.append(model_idx)
    return {"per_molecule": per_mol, "stats": dict(stats), "pass": True}


def bond_index_array(blob, pool, per_molecule) -> np.ndarray:
    """Model bond index of every analysis-pool instance (validated by correspondence)."""
    out = []
    for pi in pool["pool_inst"].tolist():
        uid = int(blob["lipo__mol_uid"][pi])
        u, v = int(blob["lipo__u"][pi]), int(blob["lipo__v"][pi])
        out.append(per_molecule[uid][(min(u, v), max(u, v))])
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# analysis pool (PROTOCOL.md §3)
# ---------------------------------------------------------------------------


def build_pool(blob, objs, mols) -> dict:
    from rdkit import Chem

    summary = read_json(STAGE_A_SUMMARY)
    c4_keys = set(summary["datasets"]["lipo__T1__primary"]["decision_stats"]["exact_stratum_type_keys"])
    s_keys = [str(x) for x in blob["lipo__s_keys"]]
    ct_keys = [str(x) for x in blob["lipo__ct_keys"]]
    s_id = blob["lipo__s_id"]
    ct_id = blob["lipo__ct_id"]
    mol_uid = blob["lipo__mol_uid"]
    mol_stats = blob["lipo__mol_stats"]
    stored_scaffold = [str(x) for x in blob["lipo__mol_scaffold"]]
    stored_smiles = [str(x) for x in blob["lipo__mol_smiles"]]

    for uid, mol in enumerate(mols):
        assert stored_smiles[uid] == str(objs[uid].smiles)
        assert int(mol_stats[uid, 0]) == mol.GetNumHeavyAtoms()
        assert int(mol_stats[uid, 2]) == len(Chem.GetSymmSSSR(mol))
        assert stored_scaffold[uid] == murcko_scaffold(mol), uid
    formulas = [heavy_formula(m) for m in mols]
    scaffolds = stored_scaffold

    c4_idx = [i for i in range(len(s_id)) if s_keys[s_id[i]] in c4_keys]
    c4_types_present = sorted({s_keys[s_id[i]] for i in c4_idx})

    ct_mols: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    inst_of: dict[tuple, list[int]] = defaultdict(list)
    for i in c4_idx:
        mu = int(mol_uid[i])
        stratum = (int(mol_stats[mu, 0]), formulas[mu], int(mol_stats[mu, 2]))
        key = (s_keys[s_id[i]], stratum)
        ct_mols[key][ct_keys[ct_id[i]]].add(mu)
        inst_of[key].append(i)

    eligible: dict[tuple, dict[str, set]] = {}
    for key, cts in ct_mols.items():
        good = {ct: ms for ct, ms in cts.items() if len(ms) >= 2}
        if len(good) >= 2:
            eligible[key] = good

    pool_inst = []
    for key, good in eligible.items():
        for i in inst_of[key]:
            if ct_keys[ct_id[i]] in good:
                pool_inst.append(i)
    pool_inst.sort()

    type_ct_scaf: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    groups_two_scaf_cts = 0
    for key, good in eligible.items():
        scafs = {ct: {scaffolds[m] for m in ms} for ct, ms in good.items()}
        if sum(1 for s in scafs.values() if len(s) >= 2) >= 2:
            groups_two_scaf_cts += 1
        for ct, sset in scafs.items():
            type_ct_scaf[key[0]][ct] |= sset
    n_types_cross = 0
    for _t, cts in type_ct_scaf.items():
        items = list(cts)
        if any(
            cts[items[a]].isdisjoint(cts[items[b]]) for a in range(len(items)) for b in range(a + 1, len(items))
        ):
            n_types_cross += 1

    units = []
    unit_map = {}
    for i in pool_inst:
        mu = int(mol_uid[i])
        stratum = (int(mol_stats[mu, 0]), formulas[mu], int(mol_stats[mu, 2]))
        ukey = (mu, s_keys[s_id[i]], ct_keys[ct_id[i]])
        if ukey not in unit_map:
            unit_map[ukey] = len(units)
            units.append(
                {
                    "mol_uid": mu,
                    "s_key": ukey[1],
                    "ct_key": ukey[2],
                    "stratum": stratum,
                    "scaffold": scaffolds[mu],
                    "instances": [],
                }
            )
        units[unit_map[ukey]]["instances"].append(i)

    support = {
        "n_c4_types": len(c4_types_present),
        "n_pool_instances": len(pool_inst),
        "n_units": len(units),
        "n_molecules": len({u["mol_uid"] for u in units}),
        "n_scaffolds": len({u["scaffold"] for u in units}),
        "n_eligible_groups": len(eligible),
        "n_groups_two_scaffold_cts": groups_two_scaf_cts,
        "n_types_cross_scaffold_contrast": n_types_cross,
    }
    gate = {
        "min_scaffolds": SUPPORT_GATE["min_scaffolds"],
        "min_cross_scaffold_types": SUPPORT_GATE["min_cross_scaffold_types"],
        "min_two_scaffold_groups": SUPPORT_GATE["min_two_scaffold_groups"],
        "scaffolds_ok": support["n_scaffolds"] >= SUPPORT_GATE["min_scaffolds"],
        "cross_scaffold_types_ok": n_types_cross >= SUPPORT_GATE["min_cross_scaffold_types"],
        "two_scaffold_groups_ok": groups_two_scaf_cts >= SUPPORT_GATE["min_two_scaffold_groups"],
    }
    gate["pass"] = bool(gate["scaffolds_ok"] and gate["cross_scaffold_types_ok"] and gate["two_scaffold_groups_ok"])
    return {
        "c4_keys": sorted(c4_keys),
        "pool_inst": np.asarray(pool_inst, dtype=np.int64),
        "units": units,
        "unit_map": unit_map,
        "support": support,
        "gate": gate,
        "formulas": formulas,
        "scaffolds": scaffolds,
    }


def unit_arrays(pool) -> dict:
    units = pool["units"]
    s_order = sorted({u["s_key"] for u in units})
    ct_order = sorted({u["ct_key"] for u in units})
    scaf_order = sorted({u["scaffold"] for u in units})
    strat_order = sorted({u["stratum"] for u in units})
    s_index = {k: i for i, k in enumerate(s_order)}
    ct_index = {k: i for i, k in enumerate(ct_order)}
    scaf_index = {k: i for i, k in enumerate(scaf_order)}
    strat_index = {k: i for i, k in enumerate(strat_order)}
    return {
        "s_order": s_order,
        "ct_order": ct_order,
        "scaf_order": scaf_order,
        "s_id": np.asarray([s_index[u["s_key"]] for u in units], dtype=np.int64),
        "ct_id": np.asarray([ct_index[u["ct_key"]] for u in units], dtype=np.int64),
        "scaf_id": np.asarray([scaf_index[u["scaffold"]] for u in units], dtype=np.int64),
        "stratum_id": np.asarray([strat_index[u["stratum"]] for u in units], dtype=np.int64),
        "mol_id": np.asarray([u["mol_uid"] for u in units], dtype=np.int64),
        "n_bonds": np.asarray([len(u["instances"]) for u in units], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# features (PROTOCOL.md §5)
# ---------------------------------------------------------------------------


def global_features(unit: dict, mols) -> np.ndarray:
    from rdkit import Chem

    mol = mols[unit["mol_uid"]]
    n_heavy = mol.GetNumHeavyAtoms()
    n_bonds = mol.GetNumBonds()
    rings, _sys_id, _sizes = ring_system_membership(mol)
    n_rings = len(rings)
    n_sys = len(set(_sys_id.values())) if rings else 0
    bond_by_pair = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b for b in mol.GetBonds()}
    n_arom_rings = 0
    for ring in rings:
        pairs = [tuple(sorted((ring[i], ring[(i + 1) % len(ring)]))) for i in range(len(ring))]
        if all(bond_by_pair[p].GetIsAromatic() for p in pairs):
            n_arom_rings += 1
    elem_counts = np.zeros(N_ELEM, dtype=np.float32)
    for a in mol.GetAtoms():
        idx = ELEMENT_INDEX.get(a.GetSymbol())
        if idx is not None:
            elem_counts[idx] += 1.0
    n_components = len(Chem.GetMolFrags(mol))
    base = np.asarray(
        [
            n_heavy,
            np.log1p(n_heavy),
            n_bonds,
            n_rings,
            n_sys,
            n_arom_rings,
            n_bonds - mol.GetNumAtoms() + n_components,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, elem_counts])


def _adjacency(mol):
    adj = [[] for _ in range(mol.GetNumAtoms())]
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[a1].append((a2, b))
        adj[a2].append((a1, b))
    return adj


def _bfs_distances(n, adj):
    dist = np.full((n, n), -1, dtype=np.int32)
    for s0 in range(n):
        dist[s0, s0] = 0
        queue = [s0]
        head = 0
        while head < len(queue):
            a = queue[head]
            head += 1
            for j, _ in adj[a]:
                if dist[s0, j] < 0:
                    dist[s0, j] = dist[s0, a] + 1
                    queue.append(j)
    return dist


def ct_features(unit: dict, mols, atom_class_of, u: int, v: int) -> np.ndarray:
    """Outer-shell + relation-axis features strictly outside the bond-anchored patch."""
    from rdkit import Chem

    mol = mols[unit["mol_uid"]]
    adj = _adjacency(mol)
    patch = {u, v}
    patch |= {j for j, _ in adj[u]}
    patch |= {j for j, _ in adj[v]}
    outer = set()
    outer_bond_class = np.zeros(N_BOND_CLASSES, dtype=np.float32)
    for a in patch:
        for j, b in adj[a]:
            if j in patch:
                continue
            outer.add(j)
            outer_bond_class[bond_code_from_rdkit(b)] += 1.0
    outer = sorted(outer)
    outer_class_counts = np.zeros(N_ATOM_CLASSES, dtype=np.float32)
    ext_degree_hist = np.zeros(4, dtype=np.float32)
    ext = patch | set(outer)
    for a in outer:
        outer_class_counts[atom_class_of(a, mol)] += 1.0
        d = sum(1 for j, _ in adj[a] if j not in ext)
        ext_degree_hist[min(d, 3)] += 1.0

    rings, sys_id, sys_sizes = ring_system_membership(mol)
    ring_bond_sizes = []
    for ring in rings:
        pairs = {tuple(sorted((ring[j], ring[(j + 1) % len(ring)]))) for j in range(len(ring))}
        if (min(u, v), max(u, v)) in pairs:
            ring_bond_sizes.append(len(ring))
    anchor_in_ring = float(bool(ring_bond_sizes))
    min_ring = min(ring_bond_sizes) if ring_bond_sizes else 0
    ring_size_bucket = 0.0 if min_ring == 0 else float(min(min_ring, 8) - 2)
    n_rings_anchor = float(len(ring_bond_sizes))
    sys_size = float(sys_sizes.get(sys_id[u], 0)) if anchor_in_ring else 0.0
    ring_axis = np.asarray([anchor_in_ring, ring_size_bucket, n_rings_anchor, sys_size], dtype=np.float32)

    n = mol.GetNumAtoms()
    dist = _bfs_distances(n, adj)
    patch_list = sorted(patch)
    het_out = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in HETERO and a.GetIdx() not in patch]
    if het_out:
        d_het = min(min(int(dist[a, x]) for x in het_out) for a in patch_list)
        n_near = sum(1 for x in het_out if min(int(dist[a, x]) for a in patch_list) <= 2)
        het_axis = np.asarray([min(d_het, 4), min(n_near, 3)], dtype=np.float32)
    else:
        het_axis = np.asarray([5.0, 0.0], dtype=np.float32)
    ring_atoms = {a for r in rings for a in r}
    ring_out = [a for a in ring_atoms if a not in patch]
    if ring_out:
        r_d = min(min(int(dist[a, x]) for x in ring_out) for a in patch_list)
        ring_dist = float(min(r_d, 4))
    else:
        ring_dist = -1.0
    rd_axis = np.asarray([ring_dist], dtype=np.float32)
    return np.concatenate(
        [outer_class_counts, [float(len(outer))], outer_bond_class, ext_degree_hist, ring_axis, het_axis, rd_axis]
    )


def build_features(pool, blob, mols) -> dict:
    """X_S / X_G / X_CT for every unit (all features are label/response-free)."""
    arrays = unit_arrays(pool)
    s_index = {k: i for i, k in enumerate(arrays["s_order"])}
    n_units = len(pool["units"])
    X_S = np.zeros((n_units, len(arrays["s_order"])), dtype=np.float32)
    X_G = np.zeros((n_units, N_G_BASE + N_ELEM), dtype=np.float32)
    X_CT = np.zeros((n_units, CT_DIM), dtype=np.float32)
    u_all = blob["lipo__u"]
    v_all = blob["lipo__v"]

    def atom_class_of(a_idx, mol):
        a = mol.GetAtomWithIdx(a_idx)
        return lip._ELEMENT_TO_CLASS.get((a.GetSymbol(), bool(a.GetIsAromatic())), lip.ATOM_UNK)

    for k, unit in enumerate(pool["units"]):
        i0 = unit["instances"][0]
        u, v = int(u_all[i0]), int(v_all[i0])
        X_S[k, s_index[unit["s_key"]]] = 1.0
        X_G[k] = global_features(unit, mols)
        X_CT[k] = ct_features(unit, mols, atom_class_of, u, v)
    return {"X_S": X_S, "X_G": X_G, "X_CT": X_CT, "arrays": arrays}


# ---------------------------------------------------------------------------
# frozen-model forward with a Bond-encoding override
# ---------------------------------------------------------------------------


def forward_with_overrides(model, batch, overrides=None):
    """Bit-identical to ``model.forward`` modulo the explicit Bond overrides."""
    import torch

    a = model.encode_atoms(batch)
    b = model.encode_bonds(a, batch)
    if overrides:
        b = b.clone()
        for idx, vec in overrides.items():
            b[int(idx)] = vec
    w = model.encode_junctions(a, b, batch)
    W, V, n_b = model.group_to_bonds(w, batch, model.organization_mode)
    compose_in = torch.cat([b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    c = model.group_composer(compose_in)
    n_graphs = int(batch["n_graphs"])
    z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
    z.index_add_(0, batch["bond_batch"], c)
    a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
    a_sum.index_add_(0, batch["atom_batch"], a)
    a_sq = torch.zeros_like(a_sum)
    a_sq.index_add_(0, batch["atom_batch"], a * a)
    g = (batch["global_feat"].to(a.dtype) - model.global_mean) / model.global_scale
    reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
    pred = model.reader(reader_in).view(-1)
    return pred, {"a": a, "b": b, "w": w, "W": W, "V": V, "n_b": n_b, "c": c, "reader_in": reader_in}


def natural_bond_encodings(model, objs, batch_size: int = 128) -> list:
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(objs), batch_size):
            chunk = objs[start : start + batch_size]
            batch = lip.batch_lipo_objects(chunk)
            a = model.encode_atoms(batch)
            b = model.encode_bonds(a, batch).numpy()
            off = 0
            for obj in chunk:
                nb = int(obj.bond_index.shape[0])
                out.append(b[off : off + nb].copy())
                off += nb
    return out


def build_references(encodings: list, objs) -> dict:
    """Class-level natural references plus donor pools (label/response-free)."""
    pools: dict[int, list] = defaultdict(list)
    for uid, obj in enumerate(objs):
        for k, bt in enumerate(np.asarray(obj.bond_type).tolist()):
            pools[int(bt)].append((uid, k))
    means, anchors = {}, {}
    for c, items in pools.items():
        vecs = np.stack([encodings[uid][k] for uid, k in items]).astype(np.float64)
        mu = vecs.mean(axis=0)
        d = ((vecs - mu) ** 2).sum(axis=1)
        uids = np.asarray([uid for uid, _ in items])
        kids = np.asarray([k for _, k in items])
        order = np.lexsort((kids, uids))
        best = int(order[np.argmin(d[order])])
        means[c] = mu.astype(np.float32)
        anchors[c] = vecs[best].astype(np.float32)
    return {"pools": dict(pools), "means": means, "anchors": anchors}


def reference_for_instance(refs: dict, encodings: list, uid: int, bidx: int, bond_class: int, scheme: str):
    if scheme == "anchor":
        return [refs["anchors"][bond_class]]
    if scheme == "mean":
        return [refs["means"][bond_class]]
    if scheme == "random":
        pool = refs["pools"][bond_class]
        eligible = [(u, k) for (u, k) in pool if u != uid]
        rng = np.random.default_rng(
            np.random.SeedSequence([REF_RANDOM_BASE_SEED, int(uid), int(bidx), int(bond_class)])
        )
        take = rng.choice(len(eligible), size=min(REF_RANDOM_DRAWS, len(eligible)), replace=False)
        return [encodings[eligible[int(j)][0]][eligible[int(j)][1]] for j in take]
    raise ValueError(scheme)


def compute_responses(seed: int, pool, blob, objs, refs: dict, limit_mols: int | None = None) -> dict:
    """One forward per (molecule, intervention); identity and localisation checks per molecule."""
    import torch

    torch.set_num_threads(1)
    model = load_model(seed)
    encodings = natural_bond_encodings(model, objs)
    mol_uid = blob["lipo__mol_uid"]
    bond_idx = pool["_bond_idx"]

    by_mol = defaultdict(list)
    for pi in pool["pool_inst"].tolist():
        by_mol[int(mol_uid[pi])].append(int(pi))
    mol_list = sorted(by_mol)
    if limit_mols is not None:
        mol_list = mol_list[:limit_mols]

    n_pool = len(pool["pool_inst"])
    pos_of = {int(pi): k for k, pi in enumerate(pool["pool_inst"].tolist())}
    R = {s: np.full(n_pool, np.nan, dtype=np.float64) for s in SCHEMES}
    dist = {s: np.full(n_pool, np.nan, dtype=np.float64) for s in SCHEMES}
    rstd = {s: np.full(n_pool, np.nan, dtype=np.float64) for s in SCHEMES}
    diag = Counter(
        {
            k: 0
            for k in (
                "localisation_violations",
                "nonb_modified_violations",
                "override_not_applied",
                "target_junctions_unchanged",
                "identity_violations",
                "identity_max_abs",
                "w_unexpected_max",
                "b_unexpected_max",
                "molecules_checked",
            )
        }
    )

    for uid in mol_list:
        obj = objs[uid]
        n_b = int(obj.bond_index.shape[0])
        plans = []  # (pool_pos, scheme, donor_vec)
        for pi in by_mol[uid]:
            bidx = int(bond_idx[pos_of[pi]])
            bclass = int(obj.bond_type[bidx])
            for scheme in SCHEMES:
                for vec in reference_for_instance(refs, encodings, uid, bidx, bclass, scheme):
                    plans.append((pi, scheme, np.asarray(vec, dtype=np.float32)))
        if not plans:
            continue
        batch = lip.batch_lipo_objects([obj] * (1 + len(plans)))
        overrides = {}
        for g, (pi, _scheme, vec) in enumerate(plans, start=1):
            bidx = int(bond_idx[pos_of[pi]])
            overrides[g * n_b + bidx] = torch.as_tensor(vec, dtype=torch.float32)
        with torch.no_grad():
            preds, feats = forward_with_overrides(model, batch, overrides)
        preds = preds.detach().numpy().astype(np.float64)
        base = float(preds[0])

        # --- localisation checks: every planned intervention ---
        n_j = int(obj.junc_index.shape[0])
        w_all = feats["w"].detach().numpy()
        b_all = feats["b"].detach().numpy()
        w_base, b_base = w_all[:n_j], b_all[:n_b]
        jb = np.asarray(obj.junc_bond)
        for g, (pi, _scheme, vec) in enumerate(plans, start=1):
            bidx = int(bond_idx[pos_of[pi]])
            w_g = w_all[g * n_j : (g + 1) * n_j]
            rowmax = np.abs(w_g - w_base).max(axis=1)
            expected = set(np.where((jb[:, 0] == bidx) | (jb[:, 1] == bidx))[0].tolist())
            unexpected = np.setdiff1d(np.arange(n_j), list(expected))
            if float(rowmax[unexpected].max(initial=0.0)) > 1e-4:
                diag["localisation_violations"] += 1
            diag["w_unexpected_max"] = max(diag["w_unexpected_max"], float(rowmax[unexpected].max(initial=0.0)))
            donor_dist = float(np.linalg.norm(np.asarray(vec, dtype=np.float64) - encodings[uid][bidx].astype(np.float64)))
            if expected and donor_dist > 0.05 and float(rowmax[list(expected)].max()) < 1e-4:
                diag["target_junctions_unchanged"] += 1
            b_g = b_all[g * n_b : (g + 1) * n_b]
            if float(np.abs(b_g[bidx] - np.asarray(vec, dtype=np.float32)).max()) > 1e-6:
                diag["override_not_applied"] += 1
            diff = np.abs(b_g - b_base).max(axis=1)
            diff[bidx] = 0.0
            diag["b_unexpected_max"] = max(diag["b_unexpected_max"], float(diff.max()))
            if float(diff.max()) > 1e-5:
                diag["nonb_modified_violations"] += 1
        # --- identity control: own encoding must leave the prediction unchanged ---
        pi0 = by_mol[uid][0]
        bidx0 = int(bond_idx[pos_of[pi0]])
        overrides_id = {n_b + bidx0: torch.as_tensor(encodings[uid][bidx0])}
        with torch.no_grad():
            pred_id, _feats_id = forward_with_overrides(model, batch, overrides_id)
        identity_gap = float(abs(pred_id[1].item() - base))
        diag["identity_max_abs"] = max(diag["identity_max_abs"], identity_gap)
        if identity_gap > 1e-7:
            diag["identity_violations"] += 1
        diag["molecules_checked"] += 1

        for g, (pi, scheme, vec) in enumerate(plans, start=1):
            k = pos_of[pi]
            r = float(preds[g] - base)
            d = float(np.linalg.norm(vec.astype(np.float64) - encodings[uid][int(bond_idx[k])].astype(np.float64)))
            if scheme == "random":
                if np.isnan(R[scheme][k]):
                    R[scheme][k] = 0.0
                    dist[scheme][k] = 0.0
                R[scheme][k] += r
                dist[scheme][k] += d
            else:
                R[scheme][k] = r
                dist[scheme][k] = d
        for pi in by_mol[uid]:
            k = pos_of[pi]
            if np.isnan(rstd["random"][k]) and not np.isnan(R["random"][k]):
                vals = [
                    float(preds[g] - base)
                    for g, (p, sch, _v) in enumerate(plans, start=1)
                    if p == pi and sch == "random"
                ]
                n_draws = len(vals)
                R["random"][k] /= n_draws
                dist["random"][k] /= n_draws
                rstd["random"][k] = float(np.std(vals)) if n_draws >= 2 else 0.0

    return {"R": R, "dist": dist, "rstd": rstd, "diag": dict(diag), "n_pool": n_pool}


def build_units(blob, pool, responses_by_seed) -> dict:
    """Aggregate bond responses into (mol, s, ct) units (mean over exact-duplicate bonds)."""
    mol_uid = blob["lipo__mol_uid"]
    s_keys = [str(x) for x in blob["lipo__s_keys"]]
    ct_keys = [str(x) for x in blob["lipo__ct_keys"]]
    s_id = blob["lipo__s_id"]
    ct_id = blob["lipo__ct_id"]
    pooled = pool["pool_inst"]
    unit_map = pool["unit_map"]
    keep = np.asarray(
        [
            unit_map[(int(mol_uid[pi]), s_keys[s_id[pi]], ct_keys[ct_id[pi]])]
            for pi in pooled.tolist()
        ],
        dtype=np.int64,
    )
    n_units = len(pool["units"])
    out = {"unit_index": keep}
    for seed, resp in responses_by_seed.items():
        for scheme in SCHEMES:
            y = np.full(n_units, np.nan, dtype=np.float64)
            d = np.full(n_units, np.nan, dtype=np.float64)
            spread = np.full(n_units, np.nan, dtype=np.float64)
            for j, u in enumerate(keep.tolist()):
                y[u] = resp["R"][scheme][j]
                d[u] = resp["dist"][scheme][j]
                spread[u] = resp["rstd"]["random"][j] if scheme == "random" else 0.0
            out[f"y_seed{seed}_{scheme}"] = y
            out[f"dist_seed{seed}_{scheme}"] = d
            if scheme == "random":
                out[f"rstd_seed{seed}"] = spread
    return out


# ---------------------------------------------------------------------------
# predictor analysis (PROTOCOL.md §5-§7)
# ---------------------------------------------------------------------------


def fit_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: dict = PREDICTOR_PARAMS, n_splits=CV_SPLITS):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = HistGradientBoostingRegressor(**params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return oof


def fit_oof_target_encoding(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cats: np.ndarray, params=PREDICTOR_PARAMS, n_splits=CV_SPLITS, smooth=10.0):
    """B2x descriptive upper bound: B1 features + out-of-fold ct target encoding."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        glob = float(y[tr].mean())
        stats = {}
        for c in np.unique(cats[tr]):
            vals = y[tr][cats[tr] == c]
            stats[int(c)] = float((vals.sum() + smooth * glob) / (len(vals) + smooth))
        te_vals = np.asarray([stats.get(int(c), glob) for c in cats[te]], dtype=np.float32)
        Xtr = np.column_stack([X[tr], np.asarray([stats.get(int(c), glob) for c in cats[tr]], dtype=np.float32)])
        Xte = np.column_stack([X[te], te_vals])
        model = HistGradientBoostingRegressor(**params)
        model.fit(Xtr, y[tr])
        oof[te] = model.predict(Xte)
    return oof


def metric_bundle(y, oof):
    err = oof - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def bootstrap_delta(y, oof_b1, oof_b2, clusters: np.ndarray, n_draws=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_draws, dtype=np.float64)
    n = len(uniq)
    for b in range(n_draws):
        take = rng.integers(0, n, size=n)
        idx = np.concatenate([members[uniq[t]] for t in take])
        e1 = np.abs(oof_b1[idx] - y[idx])
        e2 = np.abs(oof_b2[idx] - y[idx])
        deltas[b] = float(e1.mean() - e2.mean())
    return {
        "mean": float(deltas.mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_gt0": float(np.mean(deltas > 0)),
        "n_draws": int(n_draws),
    }


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def sign_agreement(a, b, tol=ZERO_TOL) -> float:
    za = np.where(np.abs(a) < tol, 0, np.sign(a))
    zb = np.where(np.abs(b) < tol, 0, np.sign(b))
    mask = np.isfinite(za) & np.isfinite(zb)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(za[mask] == zb[mask]))


def shuffle_ct(X_CT, s_ids, strata_ids, seed):
    rng = np.random.default_rng(seed)
    out = X_CT.copy()
    groups = defaultdict(list)
    for i in range(len(s_ids)):
        groups[(int(s_ids[i]), int(strata_ids[i]))].append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        out[np.asarray(idx)[perm]] = X_CT[np.asarray(idx)]
    return out


def per_cluster_table(y, oof_b1, oof_b2, clusters):
    out = {}
    for c in np.unique(clusters):
        mask = clusters == c
        out[int(c)] = {
            "n": int(mask.sum()),
            "mae_b1": float(np.mean(np.abs(oof_b1[mask] - y[mask]))),
            "mae_b2": float(np.mean(np.abs(oof_b2[mask] - y[mask]))),
            "delta": float(np.mean(np.abs(oof_b1[mask] - y[mask])) - np.mean(np.abs(oof_b2[mask] - y[mask]))),
        }
    return out


def eta2(y: np.ndarray, groups: np.ndarray) -> float:
    grand = float(y.mean())
    ss_tot = float(np.sum((y - grand) ** 2))
    if ss_tot <= 0:
        return float("nan")
    ss_b = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_b += float(m.sum()) * (float(y[m].mean()) - grand) ** 2
    return ss_b / ss_tot


def variance_decomposition(y, s_ids, ct_ids):
    """Descriptive: how much response variance is between S vs between CT within S."""
    e_s = eta2(y, s_ids)
    grand = float(y.mean())
    ss_tot = float(np.sum((y - grand) ** 2))
    ss_s = float(e_s) * ss_tot if np.isfinite(e_s) else float("nan")
    e_ct = eta2(y, ct_ids)
    ss_ct = float(e_ct) * ss_tot if np.isfinite(e_ct) else float("nan")
    nested = (ss_ct - ss_s) / (ss_tot - ss_s) if ss_tot > ss_s and np.isfinite(ss_ct) else float("nan")
    return {"eta2_S": float(e_s), "eta2_CT_total": float(e_ct), "eta2_CT_given_S": float(nested)}


def improvement_share(y, oof_b1, oof_b2, s_ids, ks=(1, 3, 5)):
    contrib = np.abs(oof_b1 - y) - np.abs(oof_b2 - y)
    total = float(contrib.sum())
    counts = Counter(int(s) for s in s_ids)
    order = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    out = {}
    for k in ks:
        sel = np.isin(s_ids, order[:k])
        out[str(k)] = float(contrib[sel].sum() / total) if total != 0 else float("nan")
    return out


def leave_out_delta(y, arrays, features, drop_ids: np.ndarray, groups) -> dict:
    """Refit B1/B2 on the subset that excludes units belonging to ``drop_ids``."""
    mask = ~np.isin(np.asarray(groups), np.asarray(drop_ids))
    X_S, X_G, X_CT = features["X_S"][mask], features["X_G"][mask], features["X_CT"][mask]
    y1 = y[mask]
    g1 = np.asarray(groups)[mask]
    if len(np.unique(g1)) < CV_SPLITS:
        return {"n_units": int(mask.sum()), "delta": float("nan")}
    oof_b1 = fit_oof(np.concatenate([X_S, X_G], axis=1), y1, g1)
    oof_b2 = fit_oof(np.concatenate([X_S, X_G, X_CT], axis=1), y1, g1)
    m1 = metric_bundle(y1, oof_b1)
    m2 = metric_bundle(y1, oof_b2)
    return {"n_units": int(mask.sum()), "delta": float(m1["mae"] - m2["mae"]), "mae_b1": m1["mae"], "mae_b2": m2["mae"]}


def analysis_for_scheme_seed(y, arrays, features, seed, scheme, distances, shuffle_null=False):
    X_S, X_G, X_CT = features["X_S"], features["X_G"], features["X_CT"]
    groups = arrays["scaf_id"]
    X_extra = np.column_stack([np.nan_to_num(distances), np.log1p(np.abs(np.nan_to_num(distances)))]).astype(np.float32)
    X_B1 = np.concatenate([X_S, X_G], axis=1)
    X_B1d = np.concatenate([X_S, X_G, X_extra], axis=1)
    X_B2 = np.concatenate([X_S, X_G, X_CT], axis=1)
    oof = {
        "B0": fit_oof(X_S, y, groups),
        "B1": fit_oof(X_B1, y, groups),
        "B1d": fit_oof(X_B1d, y, groups),
        "B2": fit_oof(X_B2, y, groups),
    }
    oof["B2x"] = fit_oof_target_encoding(X_B1, y, groups, arrays["ct_id"])
    result = {name: {"metrics": metric_bundle(y, oof[name])} for name in oof}
    delta = result["B1"]["metrics"]["mae"] - result["B2"]["metrics"]["mae"]
    result["delta_b2_minus_b1"] = float(delta)
    result["bootstrap"] = bootstrap_delta(y, oof["B1"], oof["B2"], clusters=groups)
    result["delta_b2_minus_b1d"] = float(result["B1d"]["metrics"]["mae"] - result["B2"]["metrics"]["mae"])
    result["delta_b2x_minus_b1"] = float(result["B1"]["metrics"]["mae"] - result["B2x"]["metrics"]["mae"])
    result["delta_b2_minus_b0"] = float(result["B0"]["metrics"]["mae"] - result["B2"]["metrics"]["mae"])
    result["per_scaffold"] = per_cluster_table(y, oof["B1"], oof["B2"], groups)
    result["per_type"] = per_cluster_table(y, oof["B1"], oof["B2"], arrays["s_id"])
    result["response_distance_spearman"] = spearman(y, distances)
    result["variance"] = variance_decomposition(y, arrays["s_id"], arrays["ct_id"])
    result["improvement_share_top_types"] = improvement_share(y, oof["B1"], oof["B2"], arrays["s_id"])
    counts = Counter(int(s) for s in arrays["s_id"])
    top3_types = np.asarray([t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]], dtype=np.int64)
    scaf_counts = Counter(int(s) for s in arrays["scaf_id"])
    top3_scafs = np.asarray([t for t, _ in sorted(scaf_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]], dtype=np.int64)
    result["leave_top3_types"] = leave_out_delta(y, arrays, features, top3_types, groups)
    result["leave_top3_scaffolds"] = leave_out_delta(y, arrays, features, top3_scafs, groups)
    result["top3_types_id"] = [int(x) for x in top3_types]
    result["top3_scaffolds_id"] = [int(x) for x in top3_scafs]
    if shuffle_null:
        deltas = []
        for sh in range(N_SHUFFLE):
            X_sh = shuffle_ct(X_CT, arrays["s_id"], arrays["stratum_id"], sh)
            oof_sh = fit_oof(np.concatenate([X_S, X_G, X_sh], axis=1), y, groups)
            deltas.append(float(np.mean(np.abs(oof["B1"] - y)) - np.mean(np.abs(oof_sh - y))))
        deltas = np.asarray(deltas)
        result["shuffle"] = {
            "n": int(len(deltas)),
            "deltas": deltas.tolist(),
            "mean": float(deltas.mean()),
            "p95": float(np.percentile(deltas, 95)),
            "p_shuffle": float((1 + int(np.sum(deltas >= delta))) / (len(deltas) + 1)),
        }
    return result


def within_s_varying_features(X_CT, s_ids) -> int:
    count = 0
    for j in range(X_CT.shape[1]):
        v = X_CT[:, j]
        for s in np.unique(s_ids):
            if np.var(v[s_ids == s]) > 1e-12:
                count += 1
                break
    return count


# ---------------------------------------------------------------------------
# command drivers
# ---------------------------------------------------------------------------


def _structs():
    blob = load_stage_a()
    objs = load_objects()
    mols = rdkit_molecules(blob["lipo__mol_smiles"])
    return blob, objs, mols


def _attach_bond_idx(pool, blob, per_molecule):
    pool["_bond_idx"] = bond_index_array(blob, pool, per_molecule)
    return pool


def assert_pool_frozen(pool) -> None:
    """The analysis pool must equal the pre-response frozen artifact (PROTOCOL.md §3)."""
    path = ANALYSIS / "analysis_pool.npz"
    if not path.exists():
        raise SystemExit("analysis_pool.npz missing: run the `pool` command first")
    with np.load(path, allow_pickle=True) as npz:
        frozen_inst = npz["pool_inst"]
        frozen_mol = npz["unit_mol"]
        frozen_s = npz["unit_s"]
        frozen_ct = npz["unit_ct"]
    if not np.array_equal(frozen_inst, pool["pool_inst"]):
        raise SystemExit("recomputed pool instances differ from the frozen analysis_pool.npz")
    if list(frozen_mol) != [u["mol_uid"] for u in pool["units"]]:
        raise SystemExit("recomputed pool units differ from the frozen analysis_pool.npz")
    if [str(x) for x in frozen_s] != [u["s_key"] for u in pool["units"]]:
        raise SystemExit("recomputed pool unit S identities differ from the frozen artifact")
    if [str(x) for x in frozen_ct] != [u["ct_key"] for u in pool["units"]]:
        raise SystemExit("recomputed pool unit CT identities differ from the frozen artifact")


def cmd_pool(args) -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    integrity = verify_frozen_inputs()
    if not integrity["hash_check_passed"]:
        raise SystemExit("frozen input hash check failed")
    blob, objs, mols = _structs()
    pool = build_pool(blob, objs, mols)
    np.savez_compressed(
        ANALYSIS / "analysis_pool.npz",
        pool_inst=pool["pool_inst"],
        c4_keys=np.asarray(pool["c4_keys"], dtype=object),
        unit_mol=np.asarray([u["mol_uid"] for u in pool["units"]], dtype=np.int64),
        unit_s=np.asarray([u["s_key"] for u in pool["units"]], dtype=object),
        unit_ct=np.asarray([u["ct_key"] for u in pool["units"]], dtype=object),
        unit_scaffold=np.asarray([u["scaffold"] for u in pool["units"]], dtype=object),
        unit_nbond=np.asarray([len(u["instances"]) for u in pool["units"]], dtype=np.int64),
    )
    write_json(
        ANALYSIS / "pool_manifest.json",
        {
            "protocol": "structure-context-effect-gate-v1",
            "created_before_responses": True,
            "support": pool["support"],
            "gate": pool["gate"],
            "c4_keys": pool["c4_keys"],
            "analysis_pool_sha256": sha256_file(ANALYSIS / "analysis_pool.npz"),
            "integrity": integrity,
        },
    )
    log(f"pool: {pool['support']} gate={pool['gate']['pass']}")
    return 0


def cmd_correspondence(args) -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    blob, objs, mols = _structs()
    corr = build_bond_correspondence(objs, mols)
    write_json(ANALYSIS / "bond_correspondence.json", {"pass": corr["pass"], "stats": corr["stats"]})
    log(f"correspondence: {corr['stats']}")
    return 0


def cmd_responses(args) -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    blob, objs, mols = _structs()
    pool = build_pool(blob, objs, mols)
    corr = build_bond_correspondence(objs, mols)
    _attach_bond_idx(pool, blob, corr["per_molecule"])
    assert_pool_frozen(pool)
    for seed in MODEL_SEEDS:
        t0 = time.perf_counter()
        model = load_model(seed)
        encodings = natural_bond_encodings(model, objs)
        refs = build_references(encodings, objs)
        resp = compute_responses(seed, pool, blob, objs, refs, limit_mols=args.limit_mols)
        np.savez_compressed(
            ANALYSIS / f"responses_seed{seed}.npz",
            **{s: resp["R"][s] for s in SCHEMES},
            **{f"{s}_dist": resp["dist"][s] for s in SCHEMES},
            random_rstd=resp["rstd"]["random"],
        )
        write_json(ANALYSIS / f"responses_seed{seed}_diag.json", resp["diag"])
        log(f"responses seed {seed}: {time.perf_counter() - t0:.1f}s diag={resp['diag']}")
    return 0


def cmd_predict(args) -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    blob, objs, mols = _structs()
    pool = build_pool(blob, objs, mols)
    corr = build_bond_correspondence(objs, mols)
    _attach_bond_idx(pool, blob, corr["per_molecule"])
    assert_pool_frozen(pool)
    features = build_features(pool, blob, mols)
    arrays = features["arrays"]
    resp = {seed: {} for seed in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        with np.load(ANALYSIS / f"responses_seed{seed}.npz") as npz:
            resp[seed] = {
                "R": {s: npz[s] for s in SCHEMES},
                "dist": {s: npz[f"{s}_dist"] for s in SCHEMES},
                "rstd": {"random": npz["random_rstd"]},
            }
    unit_data = build_units(blob, pool, resp)
    for seed in MODEL_SEEDS:
        if not np.isfinite(unit_data[f"y_seed{seed}_{PRIMARY_SCHEME}"]).all():
            raise SystemExit("responses incomplete (smoke --limit-mols artifact?); rerun `responses`")
    results = {
        "meta": {
            "n_units": len(pool["units"]),
            "n_molecules": pool["support"]["n_molecules"],
            "n_scaffolds": pool["support"]["n_scaffolds"],
            "n_types": len(arrays["s_order"]),
            "n_pool_instances": int(len(pool["pool_inst"])),
            "schemes": list(SCHEMES),
            "seeds": list(MODEL_SEEDS),
            "cv": "GroupKFold(5) by Murcko scaffold",
            "ct_dim": CT_DIM,
            "ct_within_s_varying_features": within_s_varying_features(features["X_CT"], arrays["s_id"]),
            "negative_control_note": "responses are frozen-model Bond-source replacement differences",
        },
        "support": pool["support"],
        "gate": pool["gate"],
        "seed0": {},
        "seed1": {},
        "cross": {},
    }
    for seed in MODEL_SEEDS:
        for scheme in SCHEMES:
            y = unit_data[f"y_seed{seed}_{scheme}"]
            res = analysis_for_scheme_seed(
                y, arrays, features, seed, scheme, unit_data[f"dist_seed{seed}_{scheme}"],
                shuffle_null=args.shuffle,
            )
            results[f"seed{seed}"][scheme] = res
            log(
                f"predict seed{seed} {scheme}: delta={res['delta_b2_minus_b1']:.6f} "
                f"ci=({res['bootstrap']['ci_lo']:.6f},{res['bootstrap']['ci_hi']:.6f}) "
                f"b0={res['B0']['metrics']['mae']:.6f} b1={res['B1']['metrics']['mae']:.6f} b2={res['B2']['metrics']['mae']:.6f}"
            )
    for scheme in SCHEMES:
        a = unit_data[f"y_seed0_{scheme}"]
        b = unit_data[f"y_seed1_{scheme}"]
        results["cross"].setdefault("seed_corr", {})[scheme] = {
            "spearman": spearman(a, b),
            "sign_agreement": sign_agreement(a, b),
            "seed0_mean": float(np.nanmean(a)),
            "seed1_mean": float(np.nanmean(b)),
            "seed0_std": float(np.nanstd(a)),
        }
    for i, s1 in enumerate(SCHEMES):
        for s2 in SCHEMES[i + 1 :]:
            for seed in MODEL_SEEDS:
                a = unit_data[f"y_seed{seed}_{s1}"]
                b = unit_data[f"y_seed{seed}_{s2}"]
                results["cross"].setdefault("scheme_corr", {})[f"seed{seed}:{s1}:{s2}"] = {
                    "spearman": spearman(a, b),
                    "sign_agreement": sign_agreement(a, b),
                }
    np.savez_compressed(
        ANALYSIS / "features.npz",
        X_S=features["X_S"],
        X_G=features["X_G"],
        X_CT=features["X_CT"],
        **{k: v for k, v in unit_data.items()},
        **{f"arr_{k}": v for k, v in arrays.items() if isinstance(v, np.ndarray)},
    )
    write_json(ANALYSIS / "predictor_metrics.json", results)
    log("predict: written")
    return 0


def model_verification() -> dict:
    """Parameter budget + valid-prediction replay against the frozen manifest (no labels)."""
    out = {"parameter_count": {}, "replay": {}}
    for seed in MODEL_SEEDS:
        model = load_model(seed)
        n = int(sum(p.numel() for p in model.parameters()))
        out["parameter_count"][f"seed{seed}"] = {"n": n, "expected": 419713, "match": n == 419713}
        out["replay"][f"seed{seed}"] = verify_prediction_replay(seed)
    out["pass"] = bool(
        all(v["match"] for v in out["parameter_count"].values())
        and all(v["pass"] for v in out["replay"].values())
    )
    return out


def _all_cell_stability(features_npz) -> dict:
    """All pairwise (scheme x seed) response stability (frozen §6.5 / §7)."""
    cells = [(s, seed) for s in SCHEMES for seed in MODEL_SEEDS]
    vals = {c: np.asarray(features_npz[f"y_seed{c[1]}_{c[0]}"], dtype=np.float64) for c in cells}
    pairs = {}
    rhos, signs = [], []
    for i, c1 in enumerate(cells):
        for c2 in cells[i + 1 :]:
            rho = spearman(vals[c1], vals[c2])
            sa = sign_agreement(vals[c1], vals[c2])
            pairs[f"{c1[0]}@s{c1[1]}__{c2[0]}@s{c2[1]}"] = {"spearman": rho, "sign_agreement": sa}
            if np.isfinite(rho):
                rhos.append(rho)
            if np.isfinite(sa):
                signs.append(sa)
    within_seed = {}
    for key, val in pairs.items():
        left, right = key.split("__")
        if left.split("@s")[1] == right.split("@s")[1]:
            within_seed[key] = val
    seed_rhos = [
        pairs[f"{s}@s0__{s}@s1"]["spearman"] for s in SCHEMES if f"{s}@s0__{s}@s1" in pairs
    ]
    seed_signs = [
        pairs[f"{s}@s0__{s}@s1"]["sign_agreement"] for s in SCHEMES if f"{s}@s0__{s}@s1" in pairs
    ]
    return {
        "all_pairs": pairs,
        "n_pairs": len(pairs),
        "min_spearman": float(np.min(rhos)) if rhos else float("nan"),
        "min_sign_agreement": float(np.min(signs)) if signs else float("nan"),
        "within_seed_min_spearman": float(np.min([v["spearman"] for v in within_seed.values()]))
        if within_seed
        else float("nan"),
        "within_seed_min_sign_agreement": float(np.min([v["sign_agreement"] for v in within_seed.values()]))
        if within_seed
        else float("nan"),
        "cross_seed_median_spearman": float(np.median(seed_rhos)) if seed_rhos else float("nan"),
        "cross_seed_median_sign_agreement": float(np.median(seed_signs)) if seed_signs else float("nan"),
    }


def compute_verdict(
    metrics: dict,
    stability: dict,
    support_gate: dict,
    verification: dict,
    response_diag: dict,
    correspondence: dict,
) -> dict:
    """Frozen decision rule (PROTOCOL.md §7); no post-hoc thresholds."""
    diag_ok = True
    for seed in MODEL_SEEDS:
        d = response_diag[f"seed{seed}"]
        diag_ok = diag_ok and float(d.get("w_unexpected_max", 1.0)) <= 1e-4
        diag_ok = diag_ok and float(d.get("b_unexpected_max", 1.0)) <= 1e-5
        diag_ok = diag_ok and float(d.get("identity_max_abs", 1.0)) <= 1e-7
        diag_ok = diag_ok and int(d.get("molecules_checked", 0)) > 0
        for k in (
            "localisation_violations",
            "nonb_modified_violations",
            "override_not_applied",
            "identity_violations",
        ):
            diag_ok = diag_ok and int(d.get(k, 1)) == 0
    impl_ok = bool(verification["pass"] and correspondence["pass"] and diag_ok)
    cross_seed_ok = bool(
        np.isfinite(stability["cross_seed_median_spearman"])
        and stability["cross_seed_median_spearman"] > 0.5
        and stability["cross_seed_median_sign_agreement"] >= 0.7
    )
    response_defined = bool(impl_ok and cross_seed_ok)
    reference_stable = bool(
        np.isfinite(stability["min_spearman"])
        and stability["min_spearman"] >= 0.5
        and stability["min_sign_agreement"] >= 0.7
    )
    improvement = {}
    for seed in MODEL_SEEDS:
        hits = 0
        for scheme in SCHEMES:
            r = metrics[f"seed{seed}"][scheme]
            if r["delta_b2_minus_b1"] > 0 and r["bootstrap"]["ci_lo"] > 0:
                hits += 1
        improvement[f"seed{seed}"] = hits >= 2
    a = metrics[f"seed{PRIMARY_SEED}"][PRIMARY_SCHEME]
    both_seeds = bool(improvement["seed0"] and improvement["seed1"])
    any_positive = any(
        metrics[f"seed{seed}"][scheme]["delta_b2_minus_b1"] > 0
        for seed in MODEL_SEEDS
        for scheme in SCHEMES
    )
    null_exceeded = {}
    nontrivial = {}
    for seed in MODEL_SEEDS:
        r = metrics[f"seed{seed}"][PRIMARY_SCHEME]
        sh = r.get("shuffle", {})
        null_exceeded[f"seed{seed}"] = bool(
            "p95" in sh and r["delta_b2_minus_b1"] > sh["p95"]
        )
        hits = 0
        for scheme in SCHEMES:
            rr = metrics[f"seed{seed}"][scheme]
            ssh = rr.get("shuffle", {})
            if "p95" not in ssh:
                continue
            thr = max(0.02 * rr["B1"]["metrics"]["mae"], ssh["p95"])
            if rr["delta_b2_minus_b1"] >= thr:
                hits += 1
        nontrivial[f"seed{seed}"] = hits >= 2
    support_ok = bool(support_gate["pass"])

    if not response_defined:
        verdict = "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"
    elif not reference_stable:
        verdict = "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"
    elif not any_positive:
        verdict = "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"
    elif not both_seeds:
        verdict = "CONTEXT_EFFECT_INCONCLUSIVE"
    elif not support_ok:
        verdict = "CONTEXT_EFFECT_INCONCLUSIVE"
    elif not (null_exceeded["seed0"] and null_exceeded["seed1"] and nontrivial["seed0"] and nontrivial["seed1"]):
        verdict = "CONTEXT_EFFECT_INCONCLUSIVE"
    else:
        verdict = "CONTEXT_EFFECT_SIGNAL_PRESENT"
    return {
        "verdict": verdict,
        "components": {
            "impl_valid": impl_ok,
            "cross_seed_reproducibility_ok": cross_seed_ok,
            "response_defined": response_defined,
            "reference_stable_all_pairs": reference_stable,
            "improvement": improvement,
            "both_seeds": both_seeds,
            "any_positive": any_positive,
            "null_exceeded": null_exceeded,
            "nontrivial": nontrivial,
            "support_ok": support_ok,
        },
        "rule": (
            "PROTOCOL.md §7: response_defined && reference_stable && any_positive, else NO_JUSTIFIED; "
            "both_seeds && support_ok && null_exceeded && nontrivial, else INCONCLUSIVE; else SIGNAL_PRESENT."
        ),
    }


def cmd_decide(args) -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics = read_json(ANALYSIS / "predictor_metrics.json")
    pool_manifest = read_json(ANALYSIS / "pool_manifest.json")
    correspondence = read_json(ANALYSIS / "bond_correspondence.json")
    response_diag = {
        f"seed{seed}": read_json(ANALYSIS / f"responses_seed{seed}_diag.json") for seed in MODEL_SEEDS
    }
    with np.load(ANALYSIS / "features.npz", allow_pickle=True) as npz:
        features_npz = {k: npz[k] for k in npz.files}
    stability = _all_cell_stability(features_npz)
    verification = model_verification()
    write_json(ANALYSIS / "model_verification.json", verification)
    verdict = compute_verdict(
        metrics, stability, pool_manifest["gate"], verification, response_diag, correspondence
    )

    def block(seed, scheme):
        r = metrics[f"seed{seed}"][scheme]
        return {
            "B0_mae": r["B0"]["metrics"]["mae"],
            "B1_mae": r["B1"]["metrics"]["mae"],
            "B2_mae": r["B2"]["metrics"]["mae"],
            "B1d_mae": r["B1d"]["metrics"]["mae"],
            "B2x_mae": r["B2x"]["metrics"]["mae"],
            "delta_b2_minus_b1": r["delta_b2_minus_b1"],
            "delta_ci": [r["bootstrap"]["ci_lo"], r["bootstrap"]["ci_hi"]],
            "shuffle_p": r.get("shuffle", {}).get("p_shuffle"),
            "shuffle_p95": r.get("shuffle", {}).get("p95"),
            "delta_b2_minus_b1d": r["delta_b2_minus_b1d"],
            "delta_b2x_minus_b1": r["delta_b2x_minus_b1"],
            "response_distance_spearman": r["response_distance_spearman"],
            "variance": r["variance"],
            "improvement_share_top_types": r["improvement_share_top_types"],
            "leave_top3_types": r["leave_top3_types"],
            "leave_top3_scaffolds": r["leave_top3_scaffolds"],
        }

    summary = {
        "protocol": "structure-context-effect-gate-v1",
        "verdict": verdict["verdict"],
        "decision": verdict,
        "boundaries": {
            "labels_loaded": False,
            "valid_test_labels_loaded": False,
            "weights_updated": False,
            "official_zinc_test_or_locked_loaded": False,
            "result_name": "frozen-model Bond-source replacement response",
        },
        "valid_instances": {
            "n_pool_bond_instances": metrics["meta"]["n_pool_instances"],
            "n_analysis_units_mol_s_ct": metrics["meta"]["n_units"],
            "n_molecules": metrics["meta"]["n_molecules"],
            "n_scaffolds": metrics["meta"]["n_scaffolds"],
            "n_c4_types": metrics["meta"]["n_types"],
            "ct_dim": metrics["meta"]["ct_dim"],
            "ct_features_with_within_S_variance": metrics["meta"]["ct_within_s_varying_features"],
        },
        "support": pool_manifest["support"],
        "support_gate": pool_manifest["gate"],
        "model_verification": verification,
        "bond_correspondence": correspondence,
        "response_diagnostics": response_diag,
        "predictors": {
            f"seed{seed}": {s: block(seed, s) for s in SCHEMES} for seed in MODEL_SEEDS
        },
        "cross_stability": stability,
        "frozen_inputs": pool_manifest["integrity"],
        "reproduce": {
            "pool": "structure_context_effect_gate_v1.py pool",
            "responses": "structure_context_effect_gate_v1.py responses",
            "predict": "structure_context_effect_gate_v1.py predict --shuffle",
            "decide": "structure_context_effect_gate_v1.py decide",
        },
    }
    write_json(OUT / "summary.json", summary)
    write_json(
        ANALYSIS / "shuffle_null.json",
        {
            f"seed{seed}": {scheme: metrics[f"seed{seed}"][scheme].get("shuffle", {}) for scheme in SCHEMES}
            for seed in MODEL_SEEDS
        },
    )
    log(f"decide: {verdict['verdict']} components={verdict['components']}")
    return 0


def cmd_all(args) -> int:
    t0 = time.perf_counter()
    cmd_pool(args)
    cmd_correspondence(args)
    cmd_responses(args)
    cmd_predict(args)
    cmd_decide(args)
    log(f"all done in {time.perf_counter() - t0:.1f}s")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pool", "correspondence", "responses", "predict", "decide", "all"):
        p = sub.add_parser(name)
        p.add_argument("--limit-mols", type=int, default=None)
        p.add_argument("--shuffle", action="store_true")
        p.set_defaults(func=globals()[f"cmd_{name}"])
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
