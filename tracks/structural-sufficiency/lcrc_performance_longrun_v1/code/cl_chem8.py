"""The CHEMLOCAL8 atom channels and the derived ``lipo_v1_chem8`` spec.

Eight label-free, structure-determined atom channels are appended to the frozen
``lipo_v1`` atom feature matrix, in this fixed order:

    1. CIP_R
    2. CIP_S
    3. CIP_UNASSIGNED_TETRA
    4. HBD
    5. HBA
    6. HYB_SP
    7. HYB_SP2
    8. HYB_SP3

Definitions (no labels, no external descriptors):

* **CIP_R / CIP_S / CIP_UNASSIGNED_TETRA** -- ``Chem.FindMolChiralCenters``
  after an explicit ``AssignStereochemistry(cleanIt=True, force=True)``.  A
  potential tetrahedral centre whose configuration is unspecified is the third
  channel; an atom with an assigned CIP code is never simultaneously marked
  unassigned.
* **HBD / HBA** -- atom-level matches of RDKit's own Lipinski SMARTS
  (``rdkit.Chem.Lipinski._HDonors`` / ``_HAcceptors``), taking the first atom of
  each match.  This is an atom role, not a molecule count.
* **HYB_SP / HYB_SP2 / HYB_SP3** -- RDKit ``GetHybridization()``.  Every other
  value (SP3D, SP3D2, UNSPECIFIED) sets all three channels to zero.

The channel artefact is built once, label-free, from the frozen source SMILES and
committed under ``analysis/source/chem8/``.  The runtime never re-parses SMILES;
it only verifies the artefact's row ids, atom offsets and SMILES against the
frozen structure cache.  That keeps the extension bit-stable across RDKit
versions and CPU/GPU hosts.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

import cl_common as C  # noqa: I001

import lcrc_common as PREP_COMMON
import feature_spec as FS
from feature_spec import FeatureField, FeatureSpec, TensorSpec

CHANNEL_NAMES: tuple[str, ...] = (
    "CIP_R",
    "CIP_S",
    "CIP_UNASSIGNED_TETRA",
    "HBD",
    "HBA",
    "HYB_SP",
    "HYB_SP2",
    "HYB_SP3",
)

CHEM8_FEATURE_FIELDS: tuple[FeatureField, ...] = tuple(
    FeatureField(
        name=name,
        dim=1,
        dtype=FS.FEATURE_DTYPE,
        description=desc,
        source="RDKit stereochemistry / Lipinski SMARTS / hybridisation (label-free)",
    )
    for name, desc in (
        ("CIP_R", "atom carries an assigned CIP R descriptor"),
        ("CIP_S", "atom carries an assigned CIP S descriptor"),
        ("CIP_UNASSIGNED_TETRA", "potential tetrahedral stereocentre with unspecified configuration"),
        ("HBD", "atom is an RDKit Lipinski hydrogen-bond donor"),
        ("HBA", "atom is an RDKit Lipinski hydrogen-bond acceptor"),
        ("HYB_SP", "RDKit hybridisation is SP"),
        ("HYB_SP2", "RDKit hybridisation is SP2"),
        ("HYB_SP3", "RDKit hybridisation is SP3"),
    )
)


def chem8_spec(base: FeatureSpec) -> FeatureSpec:
    """Append the eight channels to a copy of the frozen spec.

    Only the atom-feature contract changes.  Atom/bond vocabularies, scalars,
    globals, S1/S2 formulas and every structural semantic are inherited
    unchanged, so the module names and state-dict keys of ``LCRC`` stay the same
    (except the widened first ``atom_encoder`` linear).
    """
    if tuple(f.name for f in base.atom_feature_fields[-8:]) == CHANNEL_NAMES:
        return base
    atom_fields = tuple(base.atom_feature_fields) + CHEM8_FEATURE_FIELDS
    atom_dim = sum(f.dim for f in atom_fields)
    tensors = tuple(
        TensorSpec("atom_feat", f"[N, {atom_dim}]", FS.FEATURE_DTYPE, "per-atom feature matrix (lipo_v1 + chem8)")
        if t.name == "atom_feat"
        else t
        for t in base.output_tensors
    )
    provenance = dict(base.source_provenance)
    provenance["chem8"] = {
        "channels": list(CHANNEL_NAMES),
        "definition": "committed label-free RDKit channel artefact; see analysis/chem8_manifest.json",
        "base_spec_name": base.name,
        "base_spec_hash": base.hash(),
    }
    new = dataclasses.replace(
        base,
        name=f"{base.name}_chem8",
        description=(
            f"{base.description} Plus eight explicit, label-free local chemistry channels "
            "(CIP R/S/unassigned-tetra, HBD, HBA, SP/SP2/SP3)."
        ),
        atom_feature_fields=atom_fields,
        source_provenance=provenance,
        output_tensors=tensors,
    )
    new.validate()
    if new.atom_feat_dim != base.atom_feat_dim + 8:
        raise AssertionError("chem8 spec width is not base + 8")
    return new


# ---------------------------------------------------------------------------
# per-molecule extraction (build time only)
# ---------------------------------------------------------------------------


def _parse_like_frozen_adapter(smiles: str):
    """Strict parse first; the documented unsanitized fallback second.

    This mirrors ``mb2_rdkit.smiles_to_raw_row_relaxed`` so that atom order and
    aromaticity agree with the cache builder.  The returned method is recorded
    per row; the fallback is never silent.
    """
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit returned None")
        Chem.SanitizeMol(mol)
        return mol, "strict"
    except Exception as strict_error:  # noqa: BLE001 - recorded
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError(f"RDKit could not parse {smiles!r} (strict error: {strict_error})") from strict_error
        mol.UpdatePropertyCache(strict=False)
        return mol, "unsanitized_fallback"


def chem8_for_mol(mol) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``[n_atoms, 8]`` float32 channels for one parsed RDKit molecule."""
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    n = int(mol.GetNumAtoms())
    out = np.zeros((n, len(CHANNEL_NAMES)), dtype=np.float32)
    notes: dict[str, Any] = {"n_atoms": n, "cip_method": None, "donor_acceptor_error": None}

    # CIP and unassigned tetrahedral centres.
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
        centers = Chem.FindMolChiralCenters(
            mol,
            force=True,
            includeUnassigned=True,
            includeCIP=True,
            useLegacyImplementation=False,
        )
        notes["cip_method"] = "FindMolChiralCenters(includeUnassigned=True)"
        for idx, code in centers:
            idx = int(idx)
            if code == "R":
                out[idx, 0] = 1.0
            elif code == "S":
                out[idx, 1] = 1.0
            elif code == "?":
                out[idx, 2] = 1.0
            else:  # pragma: no cover - RDKit only returns R/S/?
                raise ValueError(f"unexpected chiral-centre code {code!r}")
    except Exception as exc:  # noqa: BLE001 - fallback molecules are recorded
        notes["cip_method"] = f"unavailable: {type(exc).__name__}: {exc}"

    # Atom-level HBD / HBA roles.
    try:
        for match in Lipinski._HDonors(mol):
            out[int(match[0]), 3] = 1.0
        for match in Lipinski._HAcceptors(mol):
            out[int(match[0]), 4] = 1.0
        notes["donor_acceptor_method"] = "rdkit.Chem.Lipinski._HDonors/_HAcceptors SMARTS, first atom"
    except Exception as exc:  # noqa: BLE001
        notes["donor_acceptor_error"] = f"{type(exc).__name__}: {exc}"

    # Hybridisation.
    for atom in mol.GetAtoms():
        i = int(atom.GetIdx())
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP:
            out[i, 5] = 1.0
        elif hyb == Chem.HybridizationType.SP2:
            out[i, 6] = 1.0
        elif hyb == Chem.HybridizationType.SP3:
            out[i, 7] = 1.0
    notes["hybridisation_method"] = "RDKit GetHybridization (SP/SP2/SP3 only)"
    return out, notes


def build_artifact(dataset: str, *, verbose: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one dataset's label-free chem8 arrays from the frozen SMILES."""
    import mb2_source as S

    smiles = S.all_smiles(dataset)
    rows = np.arange(len(smiles), dtype=np.int64)
    atom_ptr = [0]
    chunks: list[np.ndarray] = []
    methods: list[int] = []
    fallback_rows: list[dict[str, Any]] = []
    for i, smi in enumerate(smiles):
        mol, method = _parse_like_frozen_adapter(str(smi))
        channels, notes = chem8_for_mol(mol)
        if channels.shape != (int(mol.GetNumAtoms()), 8):
            raise RuntimeError(f"row {i}: unexpected channel shape {channels.shape}")
        chunks.append(channels)
        atom_ptr.append(atom_ptr[-1] + int(channels.shape[0]))
        methods.append(0 if method == "strict" else 1)
        if method != "strict":
            fallback_rows.append({"row": i, "method": method, "smiles": str(smi), "n_atoms": int(channels.shape[0])})
        if verbose and (i + 1) % 5000 == 0:
            C.log(f"[chem8 {dataset}] {i + 1}/{len(smiles)}")
    chem8 = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 8), dtype=np.float32)
    if np.any(~np.isfinite(chem8)):
        raise ValueError("chem8 contains NaN/Inf")
    if np.any((chem8 < 0) | (chem8 > 1)):
        raise ValueError("chem8 channels must be in {0,1}")
    arrays = {
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "chem8": chem8.astype(np.uint8),
        "smiles": np.asarray([str(s) for s in smiles], dtype=np.str_),
        "mol_index": rows.astype(np.int32),
        "method": np.asarray(methods, dtype=np.int8),
        "channel_names": np.asarray(CHANNEL_NAMES, dtype=np.str_),
    }
    coverage = coverage_stats(arrays, dataset)
    meta = {
        "dataset": dataset,
        "n_molecules": int(len(smiles)),
        "n_atoms": int(atom_ptr[-1]),
        "channel_names": list(CHANNEL_NAMES),
        "fallback_rows": fallback_rows,
        "n_fallback": len(fallback_rows),
        "coverage": coverage,
        "label_free": True,
        "source": "frozen source SMILES via mb2_source.all_smiles",
        "parser": "strict Chem.MolFromSmiles + SanitizeMol; documented unsanitized fallback",
        "created_utc": C.utcnow(),
    }
    return arrays, meta


def coverage_stats(arrays: Mapping[str, np.ndarray], dataset: str) -> dict[str, Any]:
    chem8 = np.asarray(arrays["chem8"])
    atom_ptr = np.asarray(arrays["atom_ptr"])
    n_atoms = int(chem8.shape[0])
    per_channel = {name: int(chem8[:, i].sum()) for i, name in enumerate(CHANNEL_NAMES) if n_atoms}
    per_molecule = np.add.reduceat(chem8, atom_ptr[:-1], axis=0) if n_atoms else np.zeros((len(atom_ptr) - 1, 8))
    counts = {
        "molecules_with_any_chiral": int((per_molecule[:, 0] + per_molecule[:, 1] > 0).sum()),
        "molecules_with_unassigned_tetra": int((per_molecule[:, 2] > 0).sum()),
        "molecules_with_hbd": int((per_molecule[:, 3] > 0).sum()),
        "molecules_with_hba": int((per_molecule[:, 4] > 0).sum()),
        "molecules_with_any_hyb": int((per_molecule[:, 5:].sum(axis=1) > 0).sum()),
    }
    return {
        "dataset": dataset,
        "n_atoms": n_atoms,
        "n_molecules": int(len(atom_ptr) - 1),
        "per_channel_atom_counts": per_channel,
        "per_channel_atom_fraction": {k: float(v / max(n_atoms, 1)) for k, v in per_channel.items()},
        **counts,
    }


def artifact_path(dataset: str):
    return C.SOURCE_CHEM8 / f"{dataset}.npz"


def load_artifact(dataset: str, *, verify: bool = True) -> dict[str, np.ndarray]:
    path = artifact_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; build it with "
            f"`uv run python code/cl_chem8_artifact.py --dataset {dataset}`"
        )
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: np.asarray(handle[key]) for key in handle.files}
    if verify:
        manifest_path = C.CHEM8_MANIFEST
        manifest = C.read_json(manifest_path)
        entry = manifest["datasets"][dataset]
        actual = C.sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{path}: sha256 {actual} != manifest {entry['sha256']}")
        if list(arrays["channel_names"]) != list(CHANNEL_NAMES):
            raise ValueError(f"{path}: channel order differs from CHANNEL_NAMES")
        if np.asarray(arrays["atom_ptr"]).shape != (int(arrays["mol_index"].size) + 1,):
            raise ValueError(f"{path}: atom_ptr shape mismatch")
        if int(arrays["chem8"].shape[0]) != int(arrays["atom_ptr"][-1]):
            raise ValueError(f"{path}: chem8 rows != atom_ptr[-1]")
    return arrays


def channels_for_rows(dataset: str, rows: Sequence[int], *, smiles: Sequence[str] | None = None) -> np.ndarray:
    """Concatenate the per-atom channels for the requested global row ids."""
    arrays = load_artifact(dataset)
    atom_ptr = np.asarray(arrays["atom_ptr"])
    chem8 = np.asarray(arrays["chem8"], dtype=np.float32)
    art_smiles = [str(x) for x in np.asarray(arrays["smiles"])]
    rows = [int(r) for r in rows]
    if len(set(rows)) != len(rows):
        raise ValueError("duplicate rows")
    pieces: list[np.ndarray] = []
    if smiles is not None and len(smiles) != len(rows):
        raise ValueError("smiles must be aligned with rows")
    for i, row in enumerate(rows):
        if row < 0 or row >= len(art_smiles):
            raise IndexError(f"{dataset}: row {row} outside chem8 artefact")
        if smiles is not None and str(smiles[i]) != art_smiles[row]:
            raise ValueError(f"{dataset}: row {row} source SMILES differs from the chem8 artefact")
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        pieces.append(chem8[a0:a1])
    if not pieces:
        return np.zeros((0, len(CHANNEL_NAMES)), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def extend_cache(
    cache: Mapping[str, Any],
    spec: FeatureSpec,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Return a copy of ``cache`` whose ``atom_feat`` has the 8 appended channels."""
    rows = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    smiles = [str(x) for x in cache["smiles"]]
    # The cache's `smiles` is indexed by cache-local position; the artefact is
    # indexed by global row id.  Verify both against each other.
    arrs = load_artifact(dataset)
    atom_ptr = np.asarray(arrs["atom_ptr"])
    chem8 = np.asarray(arrs["chem8"], dtype=np.float32)
    art_smiles = [str(x) for x in np.asarray(arrs["smiles"])]
    pieces: list[np.ndarray] = []
    for local, row in enumerate(rows):
        if row < 0 or row >= len(art_smiles):
            raise IndexError(f"{dataset}: row {row} outside chem8 artefact")
        if art_smiles[row] != smiles[local]:
            raise ValueError(f"{dataset}: row {row} cache SMILES differs from chem8 artefact")
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        pieces.append(chem8[a0:a1])
    channels = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 8), dtype=np.float32)
    base = np.asarray(cache["atom_feat"], dtype=np.float32)
    if base.shape[0] != channels.shape[0]:
        raise ValueError(f"{dataset}: atom_feat rows {base.shape[0]} != chem8 rows {channels.shape[0]}")
    if base.shape[1] + 8 != spec.atom_feat_dim:
        raise ValueError(f"{dataset}: base atom width {base.shape[1]} + 8 != spec {spec.atom_feat_dim}")
    extended = dict(cache)
    extended["atom_feat"] = np.concatenate([base, channels], axis=1).astype(np.float32)
    meta = dict(cache.get("meta", {}))
    meta["base_content_fingerprint"] = meta.get("content_fingerprint")
    meta["chem8_artifact"] = {
        "path": C.track_rel(artifact_path(dataset)),
        "sha256": C.sha256_file(artifact_path(dataset)),
        "channels": list(CHANNEL_NAMES),
    }
    meta["feature_spec_name"] = spec.name
    meta["feature_spec_hash"] = spec.hash()
    extended["meta"] = meta
    # A structural validation pass (width is checked explicitly below).
    import lcrc_data as PREP_DATA

    PREP_DATA.validate_cache(extended, spec)
    if extended["atom_feat"].shape[1] != spec.atom_feat_dim:
        raise ValueError("extended cache width does not match the spec")
    extended["meta"]["content_fingerprint"] = PREP_COMMON.content_fingerprint(extended)
    return extended


__all__ = [
    "CHANNEL_NAMES",
    "CHEM8_FEATURE_FIELDS",
    "chem8_spec",
    "build_artifact",
    "coverage_stats",
    "artifact_path",
    "load_artifact",
    "channels_for_rows",
    "extend_cache",
]
