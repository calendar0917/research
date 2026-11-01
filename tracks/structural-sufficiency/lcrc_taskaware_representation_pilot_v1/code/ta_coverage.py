"""P0.2 -- per-dataset structural coverage audit (seven tasks, structure only).

Everything here is **label-free**.  Two independent sources are used:

1. the committed raw-attribute artifact (exactly what the model consumes), for
   vocabulary coverage, UNK counts, formal-charge and hydrogen buckets,
   multi-component counts and the S1/S2 code coverage of each dataset's cache;
2. a fresh RDKit parse of the *locked* SMILES, for information the raw adapter
   does not materialise (tetrahedral configuration, E/Z specification, HBD/HBA,
   TPSA, molecular weight, Crippen logP).

Point 2 is an *audit recomputation*: it never touches an old cache and never
changes the frozen FeatureSpec.  Rows whose strict RDKit sanitisation fails use
the same documented relaxed fallback as the frozen raw adapter, and are counted.

Usage
-----
    uv run python code/ta_coverage.py                 # all seven, write JSON
    uv run python code/ta_coverage.py --only BBBP
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

import mb2_rdkit as RDK
import mb2_source as S

OUT = C.ANALYSIS / "feature_coverage.json"

#: Rows used for the expensive "potential stereo" scan of the largest dataset.
POTENTIAL_STEREO_SAMPLE = 5000
POTENTIAL_STEREO_SEED = 0

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _dist(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(arr, q))
    return out


def _molecule_facts(mol) -> dict[str, Any]:
    """Label-free RDKit facts for one molecule (audit only)."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors

    chiral_tagged = 0
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            chiral_tagged += 1
    assigned = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    bonds_with_stereo = 0
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            bonds_with_stereo += 1
    charge = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
    return {
        "n_atoms": int(mol.GetNumAtoms()),
        "n_heavy": int(mol.GetNumHeavyAtoms()),
        "n_components": int(len(Chem.GetMolFrags(mol))),
        "n_chiral_tagged_atoms": int(chiral_tagged),
        "n_assigned_chiral_centers": int(len(assigned)),
        "n_bonds_with_stereo": int(bonds_with_stereo),
        "n_formal_charged_atoms": int(sum(1 for c in charge if c != 0)),
        "max_abs_formal_charge": int(max((abs(c) for c in charge), default=0)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "mw": float(rdMolDescriptors.CalcExactMolWt(mol)),
        "rotatable": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }


def _potential_stereo(mol) -> tuple[int, int]:
    """(potential tetrahedral centres incl. unassigned, potential E/Z double bonds)."""
    from rdkit import Chem

    info = Chem.FindPotentialStereo(mol, cleanIt=False, flagPossible=True)
    n_atom = 0
    n_bond = 0
    for entry in info:
        kind = str(entry.type)
        if "Atom" in kind and "Bond" not in kind:
            n_atom += 1
        elif "Bond" in kind:
            n_bond += 1
    return n_atom, n_bond


def dataset_coverage(dataset: str) -> dict[str, Any]:
    from rdkit import Chem

    raw = __import__("mb2_raw_artifact", fromlist=["load_arrays"]).load_arrays(dataset, verify=True)
    smiles = [str(x) for x in raw["smiles"].tolist()]
    atom_class = np.asarray(raw["atom_class"], dtype=np.int64)
    atom_charge = np.asarray(raw["atom_charge"], dtype=np.int64)
    atom_h = np.asarray(raw["atom_h"], dtype=np.int64)
    edge_class = np.asarray(raw["edge_class"], dtype=np.int64)
    edge_stereo = np.asarray(raw["edge_stereo"], dtype=np.int64)
    atom_ptr = np.asarray(raw["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], dtype=np.int64)
    n_comp = np.asarray(raw["n_components"], dtype=np.int64)

    # -- artifact-level vocabulary coverage (what the model actually sees) ----
    n_mol = len(smiles)
    mol_class_presence: dict[int, int] = {}
    mol_bond_presence: dict[int, int] = {}
    for i in range(n_mol):
        for cls in set(atom_class[atom_ptr[i] : atom_ptr[i + 1]].tolist()):
            mol_class_presence[int(cls)] = mol_class_presence.get(int(cls), 0) + 1
        for cls in set(edge_class[bond_ptr[i] : bond_ptr[i + 1]].tolist()):
            mol_bond_presence[int(cls)] = mol_bond_presence.get(int(cls), 0) + 1

    class_names = [f"{name}" + ("_aromatic" if arom else "") for name, arom in RDK.ATOM_CLASSES] + ["UNK"]
    bond_names = list(RDK.BOND_CLASSES) + ["UNK"]
    charge_counts = np.bincount(atom_charge, minlength=RDK.CHARGE_UNK + 1)
    h_counts = np.bincount(atom_h, minlength=RDK.H_UNK + 1)
    stereo_counts = np.bincount(edge_stereo, minlength=4)

    # -- fresh RDKit recomputation on the locked SMILES ----------------------
    facts: list[dict[str, Any]] = []
    parse_methods: dict[str, int] = {}
    parse_failures: list[dict[str, Any]] = []
    potential_sample = (
        set(np.random.default_rng(POTENTIAL_STEREO_SEED).choice(n_mol, size=min(n_mol, POTENTIAL_STEREO_SAMPLE), replace=False).tolist())
        if n_mol > POTENTIAL_STEREO_SAMPLE
        else set(range(n_mol))
    )
    potential_atom = 0
    potential_bond = 0
    for i, smi in enumerate(smiles):
        try:
            row, method = RDK.smiles_to_raw_row_relaxed(smi, i)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:  # relaxed fallback, mirroring the frozen adapter
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                Chem.GetSymmSSSR(mol)
        except Exception as exc:  # recorded, never silently dropped
            parse_failures.append({"row": i, "error": f"{type(exc).__name__}: {exc}"})
            continue
        parse_methods[method] = parse_methods.get(method, 0) + 1
        f = _molecule_facts(mol)
        f["row"] = i
        facts.append(f)
        if i in potential_sample:
            a, b = _potential_stereo(mol)
            potential_atom += a
            potential_bond += b
            f["potential_chiral_centers"] = a
            f["potential_stereo_bonds"] = b

    def frac(pred) -> float:
        return float(sum(1 for f in facts if pred(f)) / max(len(facts), 1))

    def total(key: str) -> int:
        return int(sum(int(f[key]) for f in facts))

    coverage = {
        "dataset": dataset,
        "n_molecules": int(n_mol),
        "n_atoms_total": int(atom_class.size),
        "n_physical_bonds_total": int(edge_class.size // 2),
        "raw_artifact": {
            "atom_class_counts": {class_names[i]: int(c) for i, c in enumerate(np.bincount(atom_class, minlength=16).tolist())},
            "atom_class_molecule_coverage": {
                class_names[i]: float(mol_class_presence.get(i, 0) / max(n_mol, 1)) for i in range(16)
            },
            "bond_class_counts": {bond_names[i]: int(c) for i, c in enumerate(np.bincount(edge_class, minlength=5).tolist())},
            "bond_class_molecule_coverage": {
                bond_names[i]: float(mol_bond_presence.get(i, 0) / max(n_mol, 1)) for i in range(5)
            },
            "atom_unk_count": int((atom_class == RDK.ATOM_UNK).sum()),
            "atom_unk_fraction_atoms": float((atom_class == RDK.ATOM_UNK).mean()),
            "atom_unk_molecule_fraction": float((np.asarray([(atom_class[atom_ptr[i]:atom_ptr[i+1]] == RDK.ATOM_UNK).any() for i in range(n_mol)])).mean()),
            "bond_unk_count": int((edge_class == RDK.BOND_UNK).sum()),
            "bond_unk_fraction_records": float((edge_class == RDK.BOND_UNK).mean()),
            "charge_bucket_counts": {str(k): int(v) for k, v in enumerate(charge_counts.tolist())},
            "h_bucket_counts": {str(k): int(v) for k, v in enumerate(h_counts.tolist())},
            "edge_stereo_counts": {
                "NONE": int(stereo_counts[0]),
                "E": int(stereo_counts[1]),
                "Z": int(stereo_counts[2]),
                "OTHER": int(stereo_counts[3]),
            },
            "raw_charge_unk_fraction": float((atom_charge == RDK.CHARGE_UNK).mean()),
            "raw_h_unk_fraction": float((atom_h == RDK.H_UNK).mean()),
        },
        "multi_component": {
            "n_molecules_with_gt1_component": int((n_comp > 1).sum()),
            "fraction": float((n_comp > 1).mean()),
            "max_components": int(n_comp.max()) if n_comp.size else 0,
            "component_count_histogram": {
                str(k): int(v) for k, v in zip(*np.unique(n_comp, return_counts=True))
            },
        },
        "rdkit_audit": {
            "n_parsed": len(facts),
            "parse_methods": parse_methods,
            "parse_failures": len(parse_failures),
            "parse_failure_rows": parse_failures[:20],
            "fraction_molecules_with_formal_charge": frac(lambda f: f["n_formal_charged_atoms"] > 0),
            "fraction_atoms_formally_charged": float(total("n_formal_charged_atoms") / max(atom_class.size, 1)),
            "fraction_molecules_with_assigned_chiral_center": frac(lambda f: f["n_assigned_chiral_centers"] > 0),
            "fraction_molecules_with_chiral_tag": frac(lambda f: f["n_chiral_tagged_atoms"] > 0),
            "n_molecules_with_assigned_chiral_center": int(sum(1 for f in facts if f["n_assigned_chiral_centers"] > 0)),
            "n_assigned_chiral_centers_total": total("n_assigned_chiral_centers"),
            "fraction_molecules_with_bond_stereo": frac(lambda f: f["n_bonds_with_stereo"] > 0),
            "n_molecules_with_bond_stereo": int(sum(1 for f in facts if f["n_bonds_with_stereo"] > 0)),
            "n_stereo_bonds_total": total("n_bonds_with_stereo"),
            "potential_stereo_sample": {
                "n_rows_scanned": len(potential_sample),
                "n_potential_chiral_centers": int(potential_atom),
                "n_potential_stereo_bonds": int(potential_bond),
                "fraction_scanned_rows_with_potential_center": float(
                    sum(1 for f in facts if f.get("potential_chiral_centers", 0) > 0) / max(len(potential_sample), 1)
                ),
                "fraction_scanned_rows_with_potential_stereo_bond": float(
                    sum(1 for f in facts if f.get("potential_stereo_bonds", 0) > 0) / max(len(potential_sample), 1)
                ),
            },
            "hbd": _dist([f["hbd"] for f in facts]),
            "hba": _dist([f["hba"] for f in facts]),
            "tpsa": _dist([f["tpsa"] for f in facts]),
            "mw": _dist([f["mw"] for f in facts]),
            "clogp": _dist([f["clogp"] for f in facts]),
            "rotatable": _dist([f["rotatable"] for f in facts]),
            "num_rings": _dist([f["num_rings"] for f in facts]),
            "num_aromatic_rings": _dist([f["num_aromatic_rings"] for f in facts]),
            "fraction_molecules_with_hbd_gt0": frac(lambda f: f["hbd"] > 0),
            "fraction_molecules_with_hba_gt0": frac(lambda f: f["hba"] > 0),
            "fraction_molecules_tpsa_gt0": frac(lambda f: f["tpsa"] > 0),
            "fraction_molecules_with_ring": frac(lambda f: f["num_rings"] > 0),
            "fraction_molecules_with_aromatic_ring": frac(lambda f: f["num_aromatic_rings"] > 0),
        },
    }
    return coverage


def cache_coverage(dataset: str) -> dict[str, Any]:
    """S1/S2 code coverage of the frozen train cache (structure only)."""
    import ta_audit as A

    return A.vocabulary_utilisation(dataset, "train")


def run(only: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "scope": (
            "structure only, no label is read; RDKit facts are an audit recomputation on the locked "
            "SMILES and do not modify the frozen caches or the frozen FeatureSpec"
        ),
        "potential_stereo_sample": {"limit": POTENTIAL_STEREO_SAMPLE, "seed": POTENTIAL_STEREO_SEED},
        "datasets": {},
        "environment": {"rdkit": __import__("rdkit").__version__},
    }
    for dataset in (C.DATASETS if only is None else (only,)):
        C.log(f"[P0.2] coverage {dataset}")
        entry = dataset_coverage(dataset)
        entry["cache"] = cache_coverage(dataset)
        payload["datasets"][dataset] = entry
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)}")
    return payload


def summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "| dataset | n_mol | UNK atoms | charged mol | chiral mol | assigned centres | E/Z mol | multi-comp | HBD mean | TPSA mean | MW mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d, e in payload["datasets"].items():
        r = e["rdkit_audit"]
        raw = e["raw_artifact"]
        lines.append(
            f"| {d} | {e['n_molecules']} | {raw['atom_unk_count']} ({raw['atom_unk_fraction_atoms']*100:.3f}%) | "
            f"{r['fraction_molecules_with_formal_charge']*100:.1f}% | "
            f"{r['fraction_molecules_with_assigned_chiral_center']*100:.2f}% | "
            f"{r['n_assigned_chiral_centers_total']} | "
            f"{r['fraction_molecules_with_bond_stereo']*100:.2f}% | "
            f"{e['multi_component']['fraction']*100:.2f}% | "
            f"{r['hbd']['mean']:.2f} | {r['tpsa']['mean']:.1f} | {r['mw']['mean']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0.2 seven-dataset structural coverage audit")
    parser.add_argument("--only", default=None, choices=list(C.DATASETS))
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.only)
    if args.print:
        print(summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
