"""Identity, frozen constants, frozen SMARTS and small helpers.

Round: ``lcrc-structural-failure-localization-v0``.

Everything scientific here is frozen *before* any population statistic is
computed:

* the three case sets (GAIN5 / LOSS5 / COMMON5) and their tie rule,
* the ionization-candidate SMARTS (B3),
* the Train median cut points for the size / lipophilicity control (C1).

No model is trained, no threshold is searched, no label is used for grouping.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROUND_ID = "lcrc-structural-failure-localization-v0"

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
TRACK_DIR = ROUND_DIR.parent
REPO_ROOT = TRACK_DIR.parents[1]

ANALYSIS_DIR = ROUND_DIR / "analysis"
FIG_DIR = ANALYSIS_DIR / "figures"

EBR_DIR = TRACK_DIR / "lcrc_encoder_boundary_reference_v0"
EBR_PAIRED = EBR_DIR / "analysis/paired_errors.csv"
EDITSUP_DIR = TRACK_DIR / "lcrc_lipo_edit_supervision_public122_s0_v1"
EDITSUP_PAIRS = EDITSUP_DIR / "analysis/pairs.csv"
OGB_DIR = REPO_ROOT / "data/ogb_reader/ogbg_mollipo"

DATASET = "ogbg-mollipo"
N_MOLECULES = 4200
EXPECTED_SPLIT_SIZES = {"train": 3360, "valid": 420}

#: historical reference values this round must reproduce (tolerance 1e-5)
REF_RMSE = {"valid": {"A": 0.726678, "B": 0.678075}, "train": {"A": None, "B": None}}
REFATION_TOL = 1e-5

#: case-set sizes (frozen)
TOP_K = 5
#: Tanimoto support thresholds (frozen)
SIM_HI = 0.60
SIM_VHI = 0.80


# ---------------------------------------------------------------------------
# frozen ionization-candidate definitions (B3)
# ---------------------------------------------------------------------------
#: Every SMARTS below is frozen in this file before any population statistic is
#: produced.  They describe *structural ionization candidates only*: they say
#: nothing about the actual pKa or the actual protonation state at pH 7.4.
FROZEN_SMARTS: dict[str, str] = {
    # acidic P-OH: a tetrahedral P that carries a hydroxyl oxygen.  A phosphate
    # *triester* has no P-OH, only P-O-C, so it cannot match (explicit H1).
    "P_OH": "[P;X4][O;X2;H1]",
    # anionic phosphate oxygen (formal charge present in the SMILES)
    "P_O_MINUS": "[P;X4][O;X1;-1]",
    # phosphate tris-ester: background/negative control, deliberately NOT acidic
    "P_TRIESTER": "[P;X4](=[OX1])([OX2][#6])([OX2][#6])([OX2][#6])",
    # carboxylic acid (neutral) and carboxylate (charged)
    "C_ACID_OH": "[C;X3](=[OX1])[O;X2;H1]",
    "C_ACID_O_MINUS": "[C;X3](=[OX1])[O;X1;-1]",
    # carboxylic ester: the non-ionizable neighbour used to read off "ester vs acid"
    "C_ESTER": "[C;X3](=[OX1])[O;X2][#6]",
    # sulfonic acid / sulfonate
    "S_ACID_OH": "[S;X4](=[OX1])(=[OX1])[O;X2;H1]",
    "S_ACID_O_MINUS": "[S;X4](=[OX1])(=[OX1])[O;X1;-1]",
}

#: the three oxygen-acid families checked individually
ACID_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PHOSPHO", ("P_OH", "P_O_MINUS")),
    ("CARBOXYL", ("C_ACID_OH", "C_ACID_O_MINUS")),
    ("SULFONIC", ("S_ACID_OH", "S_ACID_O_MINUS")),
)

#: RDKit BaseFeatures families used as the secondary (unchosen) background control
BASEFEATURE_FAMILIES = ("NegIonizable", "PosIonizable")


def setup_paths() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_smarts() -> str:
    return sha256_text(json.dumps(FROZEN_SMARTS, sort_keys=True))


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# frozen case-set definitions
# ---------------------------------------------------------------------------


def freeze_case_sets(valid: "Any") -> dict[str, list[int]]:
    """GAIN5 / LOSS5 / COMMON5, ties broken by ascending official row id."""
    import pandas as pd

    d = valid.copy()
    d["e_A2"] = d["error_A"] ** 2
    d["e_B2"] = d["error_B"] ** 2
    d["gain"] = d["e_A2"] - d["e_B2"]
    d["common"] = d[["e_A2", "e_B2"]].min(axis=1)
    out: dict[str, list[int]] = {}
    for name, col, asc in (("GAIN5", "gain", False), ("LOSS5", "gain", True), ("COMMON5", "common", False)):
        ordered = d.sort_values([col, "row_id"], ascending=[asc, True], kind="mergesort")
        out[name] = [int(x) for x in ordered["row_id"].head(TOP_K)]
    return out


# ---------------------------------------------------------------------------
# chemistry helpers (RDKit)
# ---------------------------------------------------------------------------


def mol_from_smiles(smiles: str):
    from rdkit import Chem

    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        raise ValueError(f"unparsable SMILES: {smiles}")
    return m


def canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol_from_smiles(smiles))


def frozen_patterns() -> dict[str, Any]:
    """Compile the frozen SMARTS once, after a small positive/negative audit."""
    from rdkit import Chem

    pats = {k: Chem.MolFromSmarts(v) for k, v in FROZEN_SMARTS.items()}
    missing = [k for k, v in pats.items() if v is None]
    if missing:
        raise RuntimeError(f"frozen SMARTS do not compile: {missing}")
    return pats


#: hand-checked semantics of the frozen SMARTS (run before any population stat)
SMARTS_PROBES: tuple[tuple[str, str, dict[str, bool]], ...] = (
    (
        "phosphate monoester (row-1425 motif)",
        "CCOP(=O)(O)O",
        {"P_OH": True, "P_O_MINUS": False, "P_TRIESTER": False, "C_ACID_OH": False},
    ),
    (
        "phosphate diester",
        "CCOP(=O)(O)OCC",
        {"P_OH": True, "P_O_MINUS": False, "P_TRIESTER": False},
    ),
    (
        "phosphate triester (must NOT be acidic P-OH)",
        "COP(=O)(OC)OC",
        {"P_OH": False, "P_O_MINUS": False, "P_TRIESTER": True},
    ),
    (
        "phosphate dianion fragment",
        "CCOP(=O)([O-])[O-]",
        {"P_OH": False, "P_O_MINUS": True},
    ),
    ("acetic acid", "CC(=O)O", {"C_ACID_OH": True, "C_ESTER": False}),
    ("acetate", "CC(=O)[O-]", {"C_ACID_O_MINUS": True, "C_ACID_OH": False}),
    ("methyl acetate (ester, not acid)", "CC(=O)OC", {"C_ACID_OH": False, "C_ESTER": True}),
    ("methanesulfonic acid", "CS(=O)(=O)O", {"S_ACID_OH": True}),
    ("methyl methanesulfonate", "CS(=O)(=O)OC", {"S_ACID_OH": False}),
    ("benzamide (no acidic OH)", "c1ccccc1C(N)=O", {"C_ACID_OH": False, "P_OH": False}),
)


def smarts_audit() -> dict[str, Any]:
    pats = frozen_patterns()
    rows = []
    ok_all = True
    for name, smi, expect in SMARTS_PROBES:
        m = mol_from_smiles(smi)
        got = {k: bool(m.HasSubstructMatch(pats[k])) for k in expect}
        ok = all(got[k] == v for k, v in expect.items())
        ok_all = ok_all and ok
        rows.append({"probe": name, "smiles": smi, "expected": expect, "observed": got, "pass": bool(ok)})
    return {"probes": rows, "all_pass": bool(ok_all)}


def candidate_hits(smiles: str, pats: Mapping[str, Any] | None = None) -> dict[str, list[int]]:
    """Frozen-candidate hits per molecule, with canonical-SMILES atom indices."""
    from rdkit import Chem

    pats = pats or frozen_patterns()
    m = mol_from_smiles(smiles)
    out: dict[str, list[int]] = {}
    for key, pat in pats.items():
        idxs = sorted({int(i) for match in m.GetSubstructMatches(pat) for i in match})
        if idxs:
            out[key] = idxs
    return out


def acid_families(hits: Mapping[str, Sequence[int]]) -> list[str]:
    fams = []
    for fam, keys in ACID_FAMILIES:
        if any(k in hits for k in keys):
            fams.append(fam)
    return fams


def basefeature_hits(smiles: str, factory: Any) -> dict[str, int]:
    from rdkit import Chem

    m = mol_from_smiles(smiles)
    out: dict[str, int] = {}
    feats = factory.GetFeaturesForMol(m)
    for fam in BASEFEATURE_FAMILIES:
        n = sum(1 for f in feats if f.GetFamily() == fam)
        out[fam] = int(n)
    return out


def basefeature_factory() -> Any:
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    path = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(path)), str(path)


def descriptors(smiles: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    m = mol_from_smiles(smiles)
    counts: dict[str, int] = {}
    for a in m.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    formula = "".join(f"{k}{v}" for k, v in sorted(counts.items()))
    return {
        "n_heavy": int(m.GetNumHeavyAtoms()),
        "formula": formula,
        "n_C": int(counts.get("C", 0)),
        "n_N": int(counts.get("N", 0)),
        "n_O": int(counts.get("O", 0)),
        "n_S": int(counts.get("S", 0)),
        "n_P": int(counts.get("P", 0)),
        "n_halogen": int(sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))),
        "formal_charge": int(sum(a.GetFormalCharge() for a in m.GetAtoms())),
        "n_charged_atoms": int(sum(1 for a in m.GetAtoms() if a.GetFormalCharge() != 0)),
        "n_rings": int(rdMolDescriptors.CalcNumRings(m)),
        "n_ring_systems": int(len(m.GetRingInfo().AtomRings())),
        "n_arom_rings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(m)),
        "hba": int(rdMolDescriptors.CalcNumHBA(m)),
        "mol_logp": float(Crippen.MolLogP(m)),
        "n_stereo_centers": int(rdMolDescriptors.CalcNumAtomStereoCenters(m)),
        "n_unspecified_stereo": int(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m)),
    }


def scaffold_key(smiles: str) -> str:
    """Bemis-Murcko scaffold, used only to count distinct scaffolds per group."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    m = mol_from_smiles(smiles)
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc)
    except Exception:  # pragma: no cover - defensive
        return ""


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------


def rmse(e: Sequence[float]) -> float:
    a = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(a ** 2)))


def median_iqr(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None, "mean": None}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "mean": float(a.mean()),
    }


def components(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> int:
    parent = {int(n): int(n) for n in nodes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(int(n)) for n in nodes})
