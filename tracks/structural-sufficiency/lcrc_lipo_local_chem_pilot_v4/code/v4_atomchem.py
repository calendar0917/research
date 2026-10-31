"""AtomChem4: fixed atom-level chemistry features, alignment proofs and fold extras.

The four features, in this frozen order
---------------------------------------
1. ``crippen_logp_atom`` — RDKit Crippen atom logP contribution.  RDKit's
   molecule-level ``Crippen.MolLogP`` adds hydrogen contributions.  We compute
   the contribution table on ``Chem.AddHs(mol)`` and fold each added hydrogen's
   contribution onto its unique parent heavy atom.  The heavy-atom sum is then
   *identical* (to float precision) to ``Crippen.MolLogP(mol)``.
2. ``tpsa_atom`` — ``rdMolDescriptors._CalcTPSAContribs(mol)``.  The sum is
   identical to ``Descriptors.TPSA(mol)`` (the default S/P-excluded convention).
3. ``is_hbd`` — 1 iff the atom is the single atom of an RDKit
   ``BaseFeatures.fdef`` feature of family ``Donor``.
4. ``is_hba`` — same for family ``Acceptor``.

The two binary flags use the version-fixed RDKit chemical feature definitions.
This module *requires* every Donor/Acceptor feature in the frozen Fit set to be
single-atom; multi-atom definitions would make the atom flag ambiguous and
trigger a hard error.

Atom-order discipline
---------------------
The frozen raw-graph artifact was built by iterating RDKit atoms in index order
(``atom_class[i] = ...`` with ``i = atom.GetIdx()``).  The ``lipo_v1`` cache
keeps that index.  This module does not assume the correspondence: for every
Fit molecule it checks, index by index,

* cache atom type vs the frozen raw artifact's ``atom_class`` (and the raw
  artifact's SMILES against the cache SMILES);
* cache charge/hydrogen one-hots vs the raw artifact's codes;
* cache physical bonds vs the raw directed edge records;
* RDKit element / aromaticity / formal charge / hydrogen bucket against the
  cache, and the RDKit physical bond set against the cache bond set.

Any mismatch raises.  Nothing here reads a property label.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

import v4_common as C

FS = C.C.feature_spec  # frozen ``feature_spec`` module from the prep round


#: Charge/hydrogen bucket conventions of the frozen raw artifact.
CHARGE_VALUES = (-2, -1, 0, 1, 2)
CHARGE_UNK_CODE = 5
H_UNK_CODE = 5
RAW_BOND_CLASSES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
RAW_BOND_UNK = 4


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def feature_factory() -> dict[str, Any]:
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    path = f"{RDConfig.RDDataDir}/BaseFeatures.fdef"
    factory = ChemicalFeatures.BuildFeatureFactory(path)
    families = tuple(factory.GetFeatureFamilies())
    for needed in ("Donor", "Acceptor"):
        if needed not in families:
            raise RuntimeError(f"RDKit BaseFeatures.fdef lacks the {needed} family")
    return {
        "path": path,
        "sha256": C.sha256_file(path),
        "factory": factory,
        "families": list(families),
    }


def _charge_code(value: int) -> int:
    value = int(value)
    return CHARGE_VALUES.index(value) if value in CHARGE_VALUES else CHARGE_UNK_CODE


def _h_code(value: int) -> int:
    value = int(value)
    return value if 0 <= value <= 4 else H_UNK_CODE


def atom_type_map() -> dict[tuple[str, bool], int]:
    base = C.load_frozen_spec()
    return {
        (str(row["element"]), bool(row["aromatic"])): int(row["index"])
        for row in base.atom_type_metadata
    }


def _bond_class(bond) -> int:
    name = str(bond.GetBondType())
    return RAW_BOND_CLASSES.index(name) if name in RAW_BOND_CLASSES else RAW_BOND_UNK


def rdkit_atomchem(smiles: str) -> dict[str, Any]:
    """Compute AtomChem4 for one molecule in RDKit atom order (no labels)."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse frozen SMILES {smiles!r}")
    n = int(mol.GetNumAtoms())
    if n == 0:
        raise ValueError("empty molecule")

    # -- Crippen atom contributions, with added-H contributions folded on the
    # -- parent heavy atom.  Heavy-atom indices are preserved by AddHs.
    mol_h = Chem.AddHs(mol)
    n_h = int(mol_h.GetNumAtoms())
    for i in range(n):
        a, b = mol.GetAtomWithIdx(i), mol_h.GetAtomWithIdx(i)
        if a.GetSymbol() != b.GetSymbol() or bool(a.GetIsAromatic()) != bool(b.GetIsAromatic()):
            raise RuntimeError(f"AddHs reordered heavy atom {i} for {smiles!r}")
    contribs = rdMolDescriptors._CalcCrippenContribs(mol_h)
    if len(contribs) != n_h:
        raise RuntimeError("Crippen contribution table length mismatch")
    logp = np.zeros(n, dtype=np.float64)
    mr = np.zeros(n, dtype=np.float64)
    n_h_added = 0
    for atom in mol_h.GetAtoms():
        idx = int(atom.GetIdx())
        p, m = contribs[idx]
        if idx < n:
            logp[idx] += float(p)
            mr[idx] += float(m)
        else:
            n_h_added += 1
            nbrs = list(atom.GetNeighbors())
            if len(nbrs) != 1 or int(nbrs[0].GetIdx()) >= n:
                raise RuntimeError(f"added hydrogen {idx} does not have exactly one heavy parent in {smiles!r}")
            parent = int(nbrs[0].GetIdx())
            logp[parent] += float(p)
            mr[parent] += float(m)

    # -- TPSA atom contributions.
    tpsa = np.asarray(rdMolDescriptors._CalcTPSAContribs(mol), dtype=np.float64)
    if tpsa.shape != (n,):
        raise RuntimeError(f"TPSA contribution table has shape {tpsa.shape}, expected {(n,)}")

    # -- Donor / acceptor atom flags from the version-fixed feature definitions.
    ff = feature_factory()["factory"]
    hbd = np.zeros(n, dtype=np.float64)
    hba = np.zeros(n, dtype=np.float64)
    n_donor_features = 0
    n_acceptor_features = 0
    for feat in ff.GetFeaturesForMol(mol):
        family = str(feat.GetFamily())
        if family not in ("Donor", "Acceptor"):
            continue
        ids = [int(x) for x in feat.GetAtomIds()]
        if len(ids) != 1:
            raise RuntimeError(
                f"ambiguous {family} feature {feat.GetType()!r} with atoms {ids} in {smiles!r}; "
                "the atom flag would not be well defined"
            )
        if ids[0] < 0 or ids[0] >= n:
            raise RuntimeError("feature atom id outside the molecule")
        if family == "Donor":
            hbd[ids[0]] = 1.0
            n_donor_features += 1
        else:
            hba[ids[0]] = 1.0
            n_acceptor_features += 1

    feat = np.stack([logp, tpsa, hbd, hba], axis=1)
    if not np.isfinite(feat).all():
        raise RuntimeError(f"non-finite AtomChem4 value for {smiles!r}")

    mol_logp = float(Crippen.MolLogP(mol))
    mol_mr = float(Crippen.MolMR(mol))
    mol_tpsa = float(Descriptors.TPSA(mol))
    return {
        "features": feat,
        "n_atoms": n,
        "n_h_added": n_h_added,
        "sum_logp": float(feat[:, 0].sum()),
        "sum_mr": float(mr.sum()),
        "sum_tpsa": float(feat[:, 1].sum()),
        "sum_hbd": float(feat[:, 2].sum()),
        "sum_hba": float(feat[:, 3].sum()),
        "mol_logp": mol_logp,
        "mol_mr": mol_mr,
        "mol_tpsa": mol_tpsa,
        "n_donor_features": n_donor_features,
        "n_acceptor_features": n_acceptor_features,
        "rdkit_atoms": [
            {
                "index": int(a.GetIdx()),
                "symbol": str(a.GetSymbol()),
                "aromatic": bool(a.GetIsAromatic()),
                "charge": int(a.GetFormalCharge()),
                "h_bucket": int(_h_code(int(a.GetTotalNumHs()))),
            }
            for a in mol.GetAtoms()
        ],
        "rdkit_bonds": sorted(
            [
                (min(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), max(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), _bond_class(b))
                for b in mol.GetBonds()
            ]
        ),
        "canonical_smiles_sha256": hashlib.sha256(
            Chem.MolToSmiles(mol, canonical=True).encode("utf-8")
        ).hexdigest(),
    }


# ---------------------------------------------------------------------------
# frozen sources
# ---------------------------------------------------------------------------


def _fit_rows() -> list[int]:
    split = C.load_frozen_split()
    rows = [int(r) for r in split["split"]["fit"]]
    if len(rows) != C.SPLIT_SIZES["fit"] or len(set(rows)) != len(rows):
        raise ValueError("frozen Fit split is not 2352 unique rows")
    return rows


def _raw_artifact() -> tuple[dict[str, Any], dict[str, Any], list[int], dict[str, Any]]:
    """Load the label-free raw graph artifact (never a label file)."""
    raw, manifest = C.B._raw_and_manifest()
    rows = _fit_rows()
    return raw, manifest, rows, raw.get("meta", {})


def _decode_onehot(block: np.ndarray, n: int, what: str) -> np.ndarray:
    block = np.asarray(block, dtype=np.float64)
    if block.shape[0] != n:
        raise ValueError(f"{what} one-hot row count mismatch")
    sums = block.sum(axis=1)
    if not np.allclose(sums, 1.0):
        raise ValueError(f"{what} one-hot rows do not sum to 1")
    return np.argmax(block, axis=1).astype(np.int64)


def _raw_bond_reference(row: Mapping[str, Any], n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    edge_index = np.asarray(row["edge_index"], dtype=np.int64)
    edge_class = np.asarray(row["edge_class"], dtype=np.int64).reshape(-1)
    if edge_index.shape[0] != 2 or edge_index.shape[1] != edge_class.shape[0]:
        raise ValueError("raw edge arrays have inconsistent shapes")
    seen: dict[tuple[int, int], int] = {}
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if not (0 <= u < n_atoms and 0 <= v < n_atoms) or u == v:
            raise ValueError("raw edge endpoint outside the atom range")
        key = (min(u, v), max(u, v))
        val = int(edge_class[k])
        if key in seen and seen[key] != val:
            raise ValueError(f"inconsistent directed bond records for {key}")
        seen[key] = val
    keys = sorted(seen)
    bu = np.asarray([k[0] for k in keys], dtype=np.int64)
    bv = np.asarray([k[1] for k in keys], dtype=np.int64)
    bt = np.asarray([seen[k] + 1 if seen[k] <= 3 else 5 for k in keys], dtype=np.int64)
    return np.stack([bu, bv], axis=1) if keys else np.zeros((0, 2), dtype=np.int64), bt


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomChemStore:
    fit_rows: list[int]
    features: np.ndarray  # [n_atom_fit, 4] float64, cache atom order
    atom_ptr: np.ndarray  # [2353] int64
    molecule_ids: list[str]
    smiles: list[str]
    audit: dict[str, Any]


@lru_cache(maxsize=1)
def build_store() -> AtomChemStore:
    import rdkit

    cache_info = C.load_fit_cache()
    cache = cache_info["cache"]
    raw, manifest, fit_rows, _raw_meta = _raw_artifact()
    if raw.get("meta", {}).get("label_free") is not True:
        raise PermissionError("raw graph artifact is not certified label_free=true")
    if len(fit_rows) != int(cache["n_molecules"]):
        raise ValueError("Fit split size and cache size disagree")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    if atom_ptr.shape != (len(fit_rows) + 1,):
        raise ValueError("cache atom_ptr has the wrong length")
    n_total = int(atom_ptr[-1])
    if n_total != int(cache["atom_feat"].shape[0]):
        raise ValueError("cache atom_feat row count disagrees with atom_ptr")

    tmap = atom_type_map()
    features = np.zeros((n_total, C.N_ATOM_EXTRA), dtype=np.float64)
    per_mol: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for pos, row_id in enumerate(fit_rows):
        a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
        rec = raw["rows"][int(row_id)]
        if "exp" in rec:
            raise PermissionError(f"raw row {row_id} unexpectedly contains a label")
        smiles = str(rec["smiles"])
        if smiles != str(cache["smiles"][pos]):
            mismatches.append({"position": pos, "row": int(row_id), "kind": "smiles", "cache": cache["smiles"][pos], "raw": smiles})
            continue
        n = a1 - a0
        raw_class = np.asarray(rec["atom_class"], dtype=np.int64).reshape(-1)
        raw_charge = np.asarray(rec["atom_charge"], dtype=np.int64).reshape(-1)
        raw_h = np.asarray(rec["atom_h"], dtype=np.int64).reshape(-1)
        if raw_class.shape != (n,) or raw_charge.shape != (n,) or raw_h.shape != (n,):
            mismatches.append({"position": pos, "row": int(row_id), "kind": "raw_atom_lengths"})
            continue
        cache_type = np.asarray(cache["atom_type"])[a0:a1].astype(np.int64)
        feat_block = np.asarray(cache["atom_feat"])[a0:a1]
        cache_charge = _decode_onehot(feat_block[:, 16:22], n, "charge")
        cache_h = _decode_onehot(feat_block[:, 22:28], n, "hydrogen")
        raw_bonds, raw_btypes = _raw_bond_reference(rec, n)
        cache_bonds = np.asarray(cache["bond_index"])[int(cache["bond_ptr"][pos]) : int(cache["bond_ptr"][pos + 1])]
        cache_btypes = np.asarray(cache["bond_type"])[int(cache["bond_ptr"][pos]) : int(cache["bond_ptr"][pos + 1])]

        # RDKit parse of the *same* SMILES, index by index.
        res = rdkit_atomchem(smiles)
        if res["canonical_smiles_sha256"] != str(manifest["canonical_smiles_sha256"][int(row_id)]):
            mismatches.append({"position": pos, "row": int(row_id), "kind": "canonical_smiles_sha256"})
            continue
        if res["n_atoms"] != n:
            mismatches.append({"position": pos, "row": int(row_id), "kind": "n_atoms", "cache": n, "rdkit": res["n_atoms"]})
            continue
        rd_atoms = res["rdkit_atoms"]
        atom_ok = True
        for i, a in enumerate(rd_atoms):
            expected_type = tmap.get((a["symbol"], a["aromatic"]), 15)
            expected_charge = _charge_code(a["charge"])
            expected_h = int(a["h_bucket"])
            if (
                int(raw_class[i]) != int(cache_type[i])
                or int(raw_charge[i]) != int(cache_charge[i])
                or int(raw_h[i]) != int(cache_h[i])
                or expected_type != int(cache_type[i])
                or expected_charge != int(cache_charge[i])
                or expected_h != int(cache_h[i])
            ):
                atom_ok = False
                mismatches.append(
                    {
                        "position": pos,
                        "row": int(row_id),
                        "kind": "atom_index",
                        "index": i,
                        "raw": [int(raw_class[i]), int(raw_charge[i]), int(raw_h[i])],
                        "cache": [int(cache_type[i]), int(cache_charge[i]), int(cache_h[i])],
                        "rdkit": [expected_type, expected_charge, expected_h],
                    }
                )
                break
        rd_bonds = np.asarray(res["rdkit_bonds"], dtype=np.int64).reshape(-1, 3)
        rd_btypes = np.where(rd_bonds[:, 2] <= 3, rd_bonds[:, 2] + 1, 5)
        cache_bonds = np.asarray(cache_bonds, dtype=np.int64).reshape(-1, 2)
        bonds_ok = (
            np.array_equal(rd_bonds[:, :2], raw_bonds)
            and np.array_equal(rd_btypes, raw_btypes)
            and np.array_equal(raw_bonds, cache_bonds)
            and np.array_equal(raw_btypes, np.asarray(cache_btypes, dtype=np.int64))
        )
        if not bonds_ok:
            mismatches.append({"position": pos, "row": int(row_id), "kind": "bonds"})
        if not (atom_ok and bonds_ok):
            continue

        # Contribution-sum conventions.
        logp_ok = abs(res["sum_logp"] - res["mol_logp"]) <= 1e-8
        mr_ok = abs(res["sum_mr"] - res["mol_mr"]) <= 1e-8
        tpsa_ok = abs(res["sum_tpsa"] - res["mol_tpsa"]) <= 1e-8
        if not (logp_ok and mr_ok and tpsa_ok):
            mismatches.append(
                {
                    "position": pos,
                    "row": int(row_id),
                    "kind": "contribution_sum",
                    "logp": [res["sum_logp"], res["mol_logp"]],
                    "mr": [res["sum_mr"], res["mol_mr"]],
                    "tpsa": [res["sum_tpsa"], res["mol_tpsa"]],
                }
            )
            continue

        features[a0:a1] = res["features"]
        per_mol.append(
            {
                "position": pos,
                "row": int(row_id),
                "n_atoms": int(n),
                "sum_logp": res["sum_logp"],
                "mol_logp": res["mol_logp"],
                "sum_mr": res["sum_mr"],
                "mol_mr": res["mol_mr"],
                "sum_tpsa": res["sum_tpsa"],
                "mol_tpsa": res["mol_tpsa"],
                "n_donor_features": res["n_donor_features"],
                "n_acceptor_features": res["n_acceptor_features"],
                "n_h_added": res["n_h_added"],
                "shared_atoms_flagged": int(np.sum((res["features"][:, 2] > 0) & (res["features"][:, 3] > 0))),
            }
        )

    if mismatches:
        raise RuntimeError(f"AtomChem4 alignment audit failed with {len(mismatches)} mismatches: {mismatches[:5]}")
    if len(per_mol) != len(fit_rows):
        raise RuntimeError("AtomChem4 audit did not cover every Fit molecule")

    max_logp = max(abs(m["sum_logp"] - m["mol_logp"]) for m in per_mol)
    max_mr = max(abs(m["sum_mr"] - m["mol_mr"]) for m in per_mol)
    max_tpsa = max(abs(m["sum_tpsa"] - m["mol_tpsa"]) for m in per_mol)
    ff = feature_factory()
    audit = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "rdkit_version": str(rdkit.__version__),
        "features": list(C.ATOMCHEM_NAMES),
        "raw_feature_names": list(C.ATOMCHEM_RAW_NAMES),
        "n_molecules": len(fit_rows),
        "n_atoms": int(features.shape[0]),
        "fit_rows_sha256": C.row_sha256_bytes(fit_rows),
        "atom_ptr_sha256": C.array_sha256(atom_ptr.astype(np.int64)),
        "feature_matrix_sha256": C.array_sha256(features),
        "atom_order_chain": [
            "frozen raw-graph artifact built by iterating RDKit atoms in GetIdx() order",
            "lipo_v1 cache keeps raw atom index i",
            "AtomChem4 computed on Chem.MolFromSmiles(cache smiles) in GetIdx() order",
        ],
        "alignment": {
            "raw_adapter_source_sha256": C.sha256_file(
                C.REPO / "tracks/structure-contribution/code/bjoc_lipophilicity_adapter_v1.py"
            ),
            "raw_artifact_sha256": C.sha256_file(C.C.v1c.LIPO_RAW),
            "row_manifest_sha256": C.sha256_file(C.C.v1c.LIPO_ROW_MANIFEST),
            "checked_per_molecule": [
                "raw row smiles == cache smiles",
                "cache atom type == raw atom_class",
                "cache charge/hydrogen one-hot decoded == raw charge/H codes",
                "RDKit element/aromatic/charge/H bucket == cache, index by index",
                "RDKit physical bond set == raw physical bonds == cache physical bonds",
                "sum of folded-H Crippen logP == Crippen.MolLogP",
                "sum of atom TPSA contributions == Descriptors.TPSA",
                "sum of folded-H Crippen MR == Crippen.MolMR (audit only)",
            ],
            "n_mismatches": 0,
            "identity_mapping_proved_by_index": True,
            "mapping_assumption": "none; every atom index is checked attribute by attribute",
        },
        "crippen": {
            "convention": "AddHs(mol); each added H contribution folded onto its unique parent heavy atom",
            "max_abs_sum_minus_MolLogP": float(max_logp),
            "tolerance": 1e-8,
            "max_abs_sum_minus_MolMR": float(max_mr),
            "all_equal": bool(max_logp <= 1e-8 and max_mr <= 1e-8),
        },
        "tpsa": {
            "api": "rdkit.Chem.rdMolDescriptors._CalcTPSAContribs",
            "molecule_level": "rdkit.Chem.Descriptors.TPSA (includeSandP=False)",
            "max_abs_sum_minus_TPSA": float(max_tpsa),
            "tolerance": 1e-8,
            "all_equal": bool(max_tpsa <= 1e-8),
        },
        "hbd_hba": {
            "source": "RDKit BaseFeatures.fdef chemical feature definitions",
            "fdef_path_is_rdkit_data_dir": True,
            "fdef_sha256": ff["sha256"],
            "families_present": ff["families"],
            "multi_atom_donor_acceptor_features": 0,
            "single_atom_only": True,
            "total_donor_feature_atoms": int(sum(m["n_donor_features"] for m in per_mol)),
            "total_acceptor_feature_atoms": int(sum(m["n_acceptor_features"] for m in per_mol)),
            "atoms_flagged_hbd": int(np.sum(features[:, 2] > 0)),
            "atoms_flagged_hba": int(np.sum(features[:, 3] > 0)),
            "atoms_flagged_both": int(sum(m["shared_atoms_flagged"] for m in per_mol)),
        },
        "raw_value_stats": {
            C.ATOMCHEM_NAMES[j]: {
                "min": float(features[:, j].min()),
                "max": float(features[:, j].max()),
                "mean": float(features[:, j].mean()),
                "std": float(features[:, j].std()),
                "n_nonzero": int(np.count_nonzero(features[:, j])),
            }
            for j in range(C.N_ATOM_EXTRA)
        },
        "labels_used": False,
        "monitor_used": False,
        "reserved_used": False,
        "official_valid_test_used": False,
    }
    return AtomChemStore(
        fit_rows=fit_rows,
        features=features,
        atom_ptr=atom_ptr,
        molecule_ids=[str(x) for x in cache["molecule_ids"]],
        smiles=[str(x) for x in cache["smiles"]],
        audit=audit,
    )


def write_store_cache(store: AtomChemStore) -> dict[str, Any]:
    C.ensure_dir(C.CACHE)
    npz = C.CACHE / "atomchem4_fit.npz"
    meta = C.CACHE / "atomchem4_fit.json"
    np.savez_compressed(
        npz,
        fit_rows=np.asarray(store.fit_rows, dtype=np.int64),
        atom_ptr=np.asarray(store.atom_ptr, dtype=np.int64),
        features=np.asarray(store.features, dtype=np.float64),
    )
    payload = {**store.audit, "cache_npz": C.track_rel(npz), "cache_npz_sha256": C.sha256_file(npz)}
    C.write_json(meta, payload)
    return payload


# ---------------------------------------------------------------------------
# augmented spec (metadata only; safe on hosts without RDKit)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def augmented_spec():
    base = C.load_frozen_spec()
    if base.name != "lipo_v1" or base.atom_feat_dim != C.ATOM_DIM_BASE:
        raise ValueError("frozen base spec is not lipo_v1 with a 51-wide atom input")
    d = base.to_dict()
    d.pop("dimensions", None)
    d.pop("spec_hash", None)
    d["name"] = "lipo_v1_atomchem4"
    d["description"] = (
        base.description
        + " | v4 pilot: append four label-free RDKit atom-level chemistry fields "
        "(Crippen logP contribution, TPSA contribution, HBD flag, HBA flag); "
        "bond13 / scalar9 / global42 / S1 / S2 semantics unchanged."
    )
    d["atom_feature_fields"] = list(d["atom_feature_fields"]) + [
        {
            "name": name,
            "dim": 1,
            "dtype": "float32",
            "description": C.ATOMCHEM_DESCRIPTIONS[name],
            "source": "v4 AtomChem4 append from the frozen label-free SMILES",
        }
        for name in C.ATOMCHEM_NAMES
    ]
    d["atom_feature_source"] = (
        str(d.get("atom_feature_source", ""))
        + " | v4 append: RDKit Crippen/_CalcTPSAContribs/BaseFeatures.fdef Donor+Acceptor (no labels)"
    )
    d["source_provenance"] = {
        **dict(d.get("source_provenance", {})),
        "v4_atomchem4": {
            "names": list(C.ATOMCHEM_NAMES),
            "base_spec_hash": base.hash(),
            "rule": "append four one-dimensional atom fields only",
            "atom_order": "identity with the frozen raw-graph index",
            "labels_used": False,
        },
    }
    d["output_tensors"] = [
        t.to_dict()
        for t in FS._output_tensors(
            base.atom_feat_dim + C.N_ATOM_EXTRA,
            base.bond_feat_dim,
            base.n_scalars,
            base.global_dim,
        )
    ]
    spec = FS.FeatureSpec.from_dict(d)
    if spec.atom_feat_dim != C.ATOM_DIM_AUGMENTED or spec.bond_feat_dim != base.bond_feat_dim:
        raise ValueError("augmented spec widths are wrong")
    if spec.global_dim != C.GLOBAL_DIM_BASE or spec.n_scalars != base.n_scalars:
        raise ValueError("augmented spec changed global/scalar widths")
    for attr in ("bond_feature_fields", "atom_scalar_fields", "global_feature_fields"):
        if getattr(spec, attr) != getattr(base, attr):
            raise ValueError(f"augmented spec changed frozen {attr}")
    return spec


# ---------------------------------------------------------------------------
# fold extras
# ---------------------------------------------------------------------------


def standardise_fit(features: np.ndarray, train_atom_mask: np.ndarray) -> dict[str, Any]:
    """Mean/std from the Fold-1 *training* atoms only; zero variance -> scale 1."""
    X = np.asarray(features, dtype=np.float64)
    mask = np.asarray(train_atom_mask, dtype=bool)
    if mask.shape != (X.shape[0],) or not mask.any():
        raise ValueError("invalid training atom mask")
    train = X[mask]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    constant = np.flatnonzero(std <= 1e-12)
    scale = np.where(std > 1e-12, std, 1.0)
    return {
        "mean": mean,
        "scale": scale,
        "constant_indices": constant,
        "n_train_atoms": int(mask.sum()),
        "train_mean_after_standardisation_max_abs": float(np.max(np.abs(((train - mean) / scale).mean(axis=0)))),
        "train_std_after_standardisation_max_abs_deviation": float(
            np.max(np.abs(((train - mean) / scale).std(axis=0) - 1.0))
        ),
    }


@dataclass(frozen=True)
class FoldAtomExtras:
    fold: int
    spec: Any
    raw_fit: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    constant_indices: tuple[int, ...]
    train_std: np.ndarray  # float32, cache atom order for the fold train subset
    dev_std: np.ndarray
    train_broadcast_std: np.ndarray
    dev_broadcast_std: np.ndarray
    train_positions: tuple[int, ...]
    dev_positions: tuple[int, ...]
    train_rows: tuple[int, ...]
    dev_rows: tuple[int, ...]
    train_mol_ptr: np.ndarray
    dev_mol_ptr: np.ndarray
    train_molecule_ids: tuple[str, ...]
    dev_molecule_ids: tuple[str, ...]
    audit: dict[str, Any]


def _molecule_block(atom_ptr: np.ndarray, positions: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    ptr = [0]
    chunks = []
    for p in positions:
        a0, a1 = int(atom_ptr[int(p)]), int(atom_ptr[int(p) + 1])
        chunks.append(np.arange(a0, a1, dtype=np.int64))
        ptr.append(ptr[-1] + (a1 - a0))
    idx = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    return idx, np.asarray(ptr, dtype=np.int64)


def _broadcast(matrix: np.ndarray, mol_ptr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix)
    for m in range(len(mol_ptr) - 1):
        a0, a1 = int(mol_ptr[m]), int(mol_ptr[m + 1])
        if a1 <= a0:
            raise ValueError("empty molecule block")
        out[a0:a1] = matrix[a0:a1].mean(axis=0)
    return out


@lru_cache(maxsize=1)
def build_fold_extras(fold: int = C.FOLD) -> FoldAtomExtras:
    if int(fold) != C.FOLD:
        raise ValueError(f"v4 only uses Fold {C.FOLD}")
    store = build_store()
    folds = C.F.load_folds()
    train_pos = tuple(int(p) for p in C.F.train_positions(folds, int(fold)))
    dev_pos = tuple(int(p) for p in C.F.fold_positions(folds, int(fold)))
    if set(train_pos) & set(dev_pos):
        raise ValueError("fold train/dev overlap")
    if sorted(train_pos + dev_pos) != list(range(C.SPLIT_SIZES["fit"])):
        raise ValueError("fold train+dev does not cover Fit2352")

    cache_info = C.load_fit_cache()
    cache = cache_info["cache"]
    view = C.D.build_fold_view(
        int(fold),
        cache=cache,
        y_fit=np.asarray(C.load_fit_labels()["y"], dtype=np.float64),
        folds=folds,
    )
    if tuple(view.train_positions) != train_pos or tuple(view.dev_positions) != dev_pos:
        raise ValueError("v2 fold view positions disagree with the frozen fold file")
    if [str(x) for x in cache["molecule_ids"]] != store.molecule_ids:
        raise ValueError("cache molecule ids changed")
    if [str(x) for x in cache["smiles"]] != store.smiles:
        raise ValueError("cache smiles changed")

    atom_ptr = np.asarray(store.atom_ptr, dtype=np.int64)
    train_idx, train_ptr = _molecule_block(atom_ptr, train_pos)
    dev_idx, dev_ptr = _molecule_block(atom_ptr, dev_pos)
    train_mask = np.zeros(atom_ptr[-1], dtype=bool)
    train_mask[train_idx] = True

    norm = standardise_fit(store.features, train_mask)
    mean, scale = norm["mean"], norm["scale"]
    z = (store.features - mean) / scale
    train_std = z[train_idx].astype(np.float32)
    dev_std = z[dev_idx].astype(np.float32)
    train_bcast = _broadcast(train_std, train_ptr)
    dev_bcast = _broadcast(dev_std, dev_ptr)

    train_ids = tuple(store.molecule_ids[p] for p in train_pos)
    dev_ids = tuple(store.molecule_ids[p] for p in dev_pos)
    if tuple(str(x) for x in view.train_cache["molecule_ids"]) != train_ids:
        raise ValueError("train-cache molecule ids do not match the extras order")
    if tuple(str(x) for x in view.dev_cache["molecule_ids"]) != dev_ids:
        raise ValueError("dev-cache molecule ids do not match the extras order")
    if int(view.train_cache["atom_ptr"][-1]) != train_std.shape[0]:
        raise ValueError("train extras atom count differs from the train cache")
    if int(view.dev_cache["atom_ptr"][-1]) != dev_std.shape[0]:
        raise ValueError("dev extras atom count differs from the dev cache")

    # Exact per-molecule correspondence inside the subset caches: the extras
    # row (a0 + i) must belong to cache molecule m and use the same global atom.
    for name, cache_key, ptr, idx in (
        ("train", "train_cache", train_ptr, train_idx),
        ("dev", "dev_cache", dev_ptr, dev_idx),
    ):
        sub = view.__getattribute__(cache_key)
        sub_ptr = np.asarray(sub["atom_ptr"], dtype=np.int64)
        if not np.array_equal(sub_ptr, ptr):
            raise ValueError(f"{name}: subset atom_ptr differs from the pretrained extras layout")
        if int(idx.size) != int(ptr[-1]):
            raise ValueError(f"{name}: extras atom index size mismatch")

    audit = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(fold),
        "names": list(C.ATOMCHEM_NAMES),
        "n_train_molecules": int(len(train_pos)),
        "n_dev_molecules": int(len(dev_pos)),
        "n_train_atoms": int(train_std.shape[0]),
        "n_dev_atoms": int(dev_std.shape[0]),
        "train_positions_sha256": C.row_sha256_bytes(train_pos),
        "dev_positions_sha256": C.row_sha256_bytes(dev_pos),
        "train_rows_sha256": C.row_sha256_bytes(
            [int(folds["rows"][str(store.fit_rows[p])]["global_row_id"]) for p in train_pos]
        ),
        "dev_rows_sha256": C.row_sha256_bytes(
            [int(folds["rows"][str(store.fit_rows[p])]["global_row_id"]) for p in dev_pos]
        ),
        "train_positions": [int(x) for x in train_pos],
        "dev_positions": [int(x) for x in dev_pos],
        "train_molecule_ids": list(train_ids),
        "dev_molecule_ids": list(dev_ids),
        "train_atom_ptr_sha256": C.array_sha256(train_ptr.astype(np.int64)),
        "dev_atom_ptr_sha256": C.array_sha256(dev_ptr.astype(np.int64)),
        "train_std_sha256": C.array_sha256(train_std.astype(np.float32)),
        "dev_std_sha256": C.array_sha256(dev_std.astype(np.float32)),
        "train_broadcast_sha256": C.array_sha256(train_bcast.astype(np.float32)),
        "dev_broadcast_sha256": C.array_sha256(dev_bcast.astype(np.float32)),
        "raw_fit_sha256": C.array_sha256(store.features.astype(np.float64)),
        "standardisation": {
            "fit_scope": "fold1_train_atoms_only",
            "n_train_atoms": int(norm["n_train_atoms"]),
            "n_dev_atoms": int(dev_idx.size),
            "mean": [float(x) for x in mean],
            "scale": [float(x) for x in scale],
            "zero_variance_indices": [int(x) for x in norm["constant_indices"]],
            "zero_variance_policy": "scale 1.0",
            "dev_used_for_fit": False,
            "labels_used": False,
            "train_mean_after_standardisation_max_abs": norm["train_mean_after_standardisation_max_abs"],
            "train_std_after_standardisation_max_abs_deviation": norm["train_std_after_standardisation_max_abs_deviation"],
        },
        "broadcast_control": {
            "definition": "per-molecule arithmetic mean of the standardised AtomChem4 vector, broadcast to every atom",
            "same_spec": True,
            "same_parameter_count": True,
            "same_training_recipe": True,
            "per_molecule_mean_of_broadcast_equals_broadcast": True,
        },
        "data_identity": {
            "train_context": "v2 fold view train cache",
            "dev_context": "v2 fold view dev cache",
            "subset_atom_ptr_equals_extras_layout": True,
            "molecule_ids_checked": int(len(train_ids) + len(dev_ids)),
            "atom_counts_checked": int(train_std.shape[0] + dev_std.shape[0]),
            "labels_read_for_features": False,
            "monitor_structures_used": False,
            "reserved_structures_used": False,
            "official_valid_test_used": False,
        },
        "spec_hash": augmented_spec().hash(),
        "base_spec_hash": C.load_frozen_spec().hash(),
        "atom_dim_base": C.ATOM_DIM_BASE,
        "atom_dim_augmented": C.ATOM_DIM_AUGMENTED,
    }
    return FoldAtomExtras(
        fold=int(fold),
        spec=augmented_spec(),
        raw_fit=store.features,
        mean=mean,
        scale=scale,
        constant_indices=tuple(int(x) for x in norm["constant_indices"]),
        train_std=train_std,
        dev_std=dev_std,
        train_broadcast_std=train_bcast,
        dev_broadcast_std=dev_bcast,
        train_positions=train_pos,
        dev_positions=dev_pos,
        train_rows=tuple(int(folds["rows"][str(store.fit_rows[p])]["global_row_id"]) for p in train_pos),
        dev_rows=tuple(int(folds["rows"][str(store.fit_rows[p])]["global_row_id"]) for p in dev_pos),
        train_mol_ptr=train_ptr,
        dev_mol_ptr=dev_ptr,
        train_molecule_ids=train_ids,
        dev_molecule_ids=dev_ids,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# frozen extras (the only chemistry artifact the GPU host needs)
# ---------------------------------------------------------------------------

FROZEN_EXTRAS_NPZ = C.ANALYSIS / "frozen_extras_v4.npz"
FROZEN_EXTRAS_JSON = C.ANALYSIS / "frozen_extras_v4.json"


def export_frozen_extras() -> dict[str, Any]:
    fx = build_fold_extras(C.FOLD)
    arrays = {
        "raw_fit": np.asarray(fx.raw_fit, dtype=np.float64),
        "mean": np.asarray(fx.mean, dtype=np.float64),
        "scale": np.asarray(fx.scale, dtype=np.float64),
        "train_std": np.asarray(fx.train_std, dtype=np.float32),
        "dev_std": np.asarray(fx.dev_std, dtype=np.float32),
        "train_broadcast_std": np.asarray(fx.train_broadcast_std, dtype=np.float32),
        "dev_broadcast_std": np.asarray(fx.dev_broadcast_std, dtype=np.float32),
        "train_positions": np.asarray(fx.train_positions, dtype=np.int64),
        "dev_positions": np.asarray(fx.dev_positions, dtype=np.int64),
        "train_rows": np.asarray(fx.train_rows, dtype=np.int64),
        "dev_rows": np.asarray(fx.dev_rows, dtype=np.int64),
        "train_mol_ptr": np.asarray(fx.train_mol_ptr, dtype=np.int64),
        "dev_mol_ptr": np.asarray(fx.dev_mol_ptr, dtype=np.int64),
        "constant_indices": np.asarray(fx.constant_indices, dtype=np.int64),
    }
    C.ensure_dir(C.ANALYSIS)
    np.savez_compressed(FROZEN_EXTRAS_NPZ, **arrays)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold": int(C.FOLD),
        "seed": int(C.SEED),
        "names": list(C.ATOMCHEM_NAMES),
        "raw_feature_names": list(C.ATOMCHEM_RAW_NAMES),
        "atom_dim_base": C.ATOM_DIM_BASE,
        "atom_dim_augmented": C.ATOM_DIM_AUGMENTED,
        "n_atom_extra": C.N_ATOM_EXTRA,
        "spec_hash": fx.spec.hash(),
        "base_spec_hash": C.load_frozen_spec().hash(),
        "sha256": {
            key: C.array_sha256(np.asarray(val))
            for key, val in arrays.items()
        },
        "shapes": {key: list(np.asarray(val).shape) for key, val in arrays.items()},
        "train_molecule_ids": list(fx.train_molecule_ids),
        "dev_molecule_ids": list(fx.dev_molecule_ids),
        "audit": fx.audit,
        "npz_sha256": C.sha256_file(FROZEN_EXTRAS_NPZ),
        "npz_path": C.track_rel(FROZEN_EXTRAS_NPZ),
        "labels_used": False,
        "monitor_used": False,
        "reserved_used": False,
        "official_valid_test_used": False,
    }
    C.write_json(FROZEN_EXTRAS_JSON, payload)
    return payload


@lru_cache(maxsize=1)
def load_frozen_extras() -> dict[str, Any]:
    if not FROZEN_EXTRAS_NPZ.exists() or not FROZEN_EXTRAS_JSON.exists():
        raise FileNotFoundError("missing frozen v4 extras; run `run_v4.py freeze-extras` on CPU")
    payload = C.read_json(FROZEN_EXTRAS_JSON)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise ValueError("frozen extras protocol id mismatch")
    if C.sha256_file(FROZEN_EXTRAS_NPZ) != payload.get("npz_sha256"):
        raise PermissionError("frozen extras npz hash mismatch")
    spec = augmented_spec()
    if spec.hash() != payload["spec_hash"]:
        raise PermissionError("frozen extras spec hash mismatch")
    data = np.load(FROZEN_EXTRAS_NPZ)
    arrays = {key: data[key] for key in data.files}
    for key, expected in payload["sha256"].items():
        if C.array_sha256(np.asarray(arrays[key])) != expected:
            raise PermissionError(f"frozen extras array {key} hash mismatch")
    return {
        "spec": spec,
        "audit": payload,
        "raw_fit": np.asarray(arrays["raw_fit"], dtype=np.float64),
        "mean": np.asarray(arrays["mean"], dtype=np.float64),
        "scale": np.asarray(arrays["scale"], dtype=np.float64),
        "train_std": np.asarray(arrays["train_std"], dtype=np.float32),
        "dev_std": np.asarray(arrays["dev_std"], dtype=np.float32),
        "train_broadcast_std": np.asarray(arrays["train_broadcast_std"], dtype=np.float32),
        "dev_broadcast_std": np.asarray(arrays["dev_broadcast_std"], dtype=np.float32),
        "train_positions": tuple(int(x) for x in arrays["train_positions"].tolist()),
        "dev_positions": tuple(int(x) for x in arrays["dev_positions"].tolist()),
        "train_rows": tuple(int(x) for x in arrays["train_rows"].tolist()),
        "dev_rows": tuple(int(x) for x in arrays["dev_rows"].tolist()),
        "train_mol_ptr": np.asarray(arrays["train_mol_ptr"], dtype=np.int64),
        "dev_mol_ptr": np.asarray(arrays["dev_mol_ptr"], dtype=np.int64),
        "constant_indices": tuple(int(x) for x in arrays["constant_indices"].tolist()),
    }


def _variant_payload(
    *,
    variant: str,
    train_std: np.ndarray,
    dev_std: np.ndarray,
    train_broadcast_std: np.ndarray,
    dev_broadcast_std: np.ndarray,
    train_positions,
    dev_positions,
    train_rows,
    dev_rows,
    train_molecule_ids,
    dev_molecule_ids,
    train_mol_ptr,
    dev_mol_ptr,
    spec,
    mean,
    scale,
) -> dict[str, Any]:
    if variant not in C.ARM_VARIANT.values():
        raise ValueError(f"unknown v4 feature variant {variant!r}")
    if variant == "atomchem4":
        train = np.asarray(train_std, dtype=np.float32)
        dev = np.asarray(dev_std, dtype=np.float32)
    else:
        train = np.asarray(train_broadcast_std, dtype=np.float32)
        dev = np.asarray(dev_broadcast_std, dtype=np.float32)
    return {
        "train": train,
        "dev": dev,
        "train_positions": tuple(train_positions),
        "dev_positions": tuple(dev_positions),
        "train_rows": tuple(train_rows),
        "dev_rows": tuple(dev_rows),
        "train_molecule_ids": tuple(str(x) for x in train_molecule_ids),
        "dev_molecule_ids": tuple(str(x) for x in dev_molecule_ids),
        "train_mol_ptr": np.asarray(train_mol_ptr, dtype=np.int64),
        "dev_mol_ptr": np.asarray(dev_mol_ptr, dtype=np.int64),
        "spec": spec,
        "mean": np.asarray(mean, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
    }


def atom_arrays_from_fold_extras(fx: FoldAtomExtras, variant: str) -> dict[str, Any]:
    return _variant_payload(
        variant=variant,
        train_std=fx.train_std,
        dev_std=fx.dev_std,
        train_broadcast_std=fx.train_broadcast_std,
        dev_broadcast_std=fx.dev_broadcast_std,
        train_positions=fx.train_positions,
        dev_positions=fx.dev_positions,
        train_rows=fx.train_rows,
        dev_rows=fx.dev_rows,
        train_molecule_ids=fx.train_molecule_ids,
        dev_molecule_ids=fx.dev_molecule_ids,
        train_mol_ptr=fx.train_mol_ptr,
        dev_mol_ptr=fx.dev_mol_ptr,
        spec=fx.spec,
        mean=fx.mean,
        scale=fx.scale,
    )


def atom_arrays_for_variant(variant: str, extras: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fx = load_frozen_extras() if extras is None else extras
    return _variant_payload(
        variant=variant,
        train_std=fx["train_std"],
        dev_std=fx["dev_std"],
        train_broadcast_std=fx["train_broadcast_std"],
        dev_broadcast_std=fx["dev_broadcast_std"],
        train_positions=fx["train_positions"],
        dev_positions=fx["dev_positions"],
        train_rows=fx["train_rows"],
        dev_rows=fx["dev_rows"],
        train_molecule_ids=fx["audit"]["train_molecule_ids"],
        dev_molecule_ids=fx["audit"]["dev_molecule_ids"],
        train_mol_ptr=fx["train_mol_ptr"],
        dev_mol_ptr=fx["dev_mol_ptr"],
        spec=fx["spec"],
        mean=fx["mean"],
        scale=fx["scale"],
    )


__all__ = [
    "feature_factory",
    "atom_type_map",
    "rdkit_atomchem",
    "AtomChemStore",
    "build_store",
    "write_store_cache",
    "augmented_spec",
    "standardise_fit",
    "FoldAtomExtras",
    "build_fold_extras",
    "FROZEN_EXTRAS_NPZ",
    "FROZEN_EXTRAS_JSON",
    "export_frozen_extras",
    "load_frozen_extras",
    "atom_arrays_from_fold_extras",
    "atom_arrays_for_variant",
]
