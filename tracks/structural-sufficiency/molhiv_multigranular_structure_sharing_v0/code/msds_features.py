"""Morgan fingerprint block ``M`` (frozen recipe) and shared metric helpers.

``M`` = 2048-bit binary Morgan/ECFP4 (radius 2, RDKit
``GetMorganFingerprintAsBitVect`` -- the historical ``mb2_baseline.py`` call, *not*
a count fingerprint) concatenated with ``log1p(n_atoms)`` and ``log1p(n_bonds)``
standardised with Train-only mean/std.

The two count columns are standardised from Train only; the 2048 bit columns stay
0/1 (no scaling), exactly as declared in PROTOCOL.md §5.1.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix

import msds_common as C
import msds_data as D


def _mol_robust(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "strict"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "failed"
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        return None, "failed"
    return mol, "unsanitized_fallback"


def build_morgan(molecules: D.Molecules) -> tuple[np.ndarray, dict[str, Any]]:
    """Binary Morgan bits, packed to save disk, cached with a checksum."""
    cache = C.CACHE / "molhiv_morgan_r2_2048.npz"
    if cache.exists():
        with np.load(cache) as z:
            packed = z["packed"]
            n_bits = int(z["n_bits"])
        meta = C.read_json(C.CACHE / "molhiv_morgan_meta.json")
        bits = np.unpackbits(packed, axis=1)[:, :n_bits]
        return bits.astype(np.uint8), meta

    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    RDLogger.DisableLog("rdApp.*")
    n = molecules.n_molecules
    bits = np.zeros((n, 2048), dtype=np.uint8)
    methods: list[str] = []
    n_failed = 0
    for i, smiles in enumerate(molecules.smiles):
        mol, method = _mol_robust(smiles)
        methods.append(method)
        if mol is None:
            n_failed += 1
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.uint8)
        ConvertToNumpyArray(fp, arr)
        bits[i] = arr
    C.CACHE.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(bits, axis=1)
    np.savez_compressed(cache, packed=packed, n_bits=np.int64(2048))
    meta = {
        "impl": "rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)",
        "radius": 2,
        "n_bits": 2048,
        "folded": "bit-OR (binary), NOT count folded",
        "n_molecules": n,
        "n_parse_failed": n_failed,
        "n_unsanitized_fallback": methods.count("unsanitized_fallback"),
        "n_parse_failed_handling": "rows kept with all-zero fingerprint; recorded, never dropped",
        "bit_density_mean": float(bits.mean()),
        "cache": C.rel(cache),
        "cache_sha256": C.sha256_file(cache),
        "rdkit_version": __import__("rdkit").__version__,
    }
    C.write_json(C.CACHE / "molhiv_morgan_meta.json", meta)
    return bits, meta


def design_M(
    molecules: D.Molecules, bits: np.ndarray, train_rows: np.ndarray
) -> tuple[csr_matrix, dict[str, Any]]:
    """Sparse ``M`` with Train-fitted standardisation of the two count columns."""
    import scipy.sparse as sp

    n_atoms = molecules.n_atoms.astype(np.float64)
    n_bonds = (molecules.bond_offsets[1:] - molecules.bond_offsets[:-1]).astype(np.float64)
    a = np.log1p(n_atoms)
    b = np.log1p(n_bonds)
    mu_a, sd_a = float(a[train_rows].mean()), float(a[train_rows].std())
    mu_b, sd_b = float(b[train_rows].mean()), float(b[train_rows].std())
    za = (a - mu_a) / (sd_a if sd_a > 0 else 1.0)
    zb = (b - mu_b) / (sd_b if sd_b > 0 else 1.0)
    extra = sp.csr_matrix(np.stack([za, zb], axis=1))
    M = sp.hstack([sp.csr_matrix(bits.astype(np.float64)), extra], format="csr")
    meta = {
        "n_columns": int(M.shape[1]),
        "bits_columns": 2048,
        "extra_columns": ["log1p(n_atoms) z-scored on Train", "log1p(n_bonds) z-scored on Train"],
        "train_mean": {"log1p_n_atoms": mu_a, "log1p_n_bonds": mu_b},
        "train_std": {"log1p_n_atoms": sd_a, "log1p_n_bonds": sd_b},
        "standardisation_fit": "Train only",
        "nnz": int(M.nnz),
    }
    return M, meta
