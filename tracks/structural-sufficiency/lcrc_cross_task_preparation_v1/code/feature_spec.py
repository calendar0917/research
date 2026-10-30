"""Explicit, serialisable feature specifications for LCRC-A inputs.

Two independent schemas are defined here:

``legacy_zinc``
    The frozen ZINC-LCRC-A interface: P3 108-d atom descriptors, P3 7-d bond
    descriptors, 21 categorical atom types, 3 one-based bond types, 33 global
    topology statistics and 9 train-standardised atom scalars.  It exists so
    that the old ZINC model/checkpoint can be replayed bit-for-bit.

``lipo_v1``
    A new Lipophilicity schema with explicitly named chemical fields.  It is
    **not** a re-scaling or zero-padding of ``legacy_zinc``: atom/bond/global
    dimensions, vocabularies and scalar semantics are different and recorded
    here.  The LCRC-A computational mechanism (one local encode + per-atom
    contribution + global correction) is unchanged.

Every field records a name, dimension, dtype, chemical/topological meaning and
provenance.  The whole specification is canonical-JSON hashed.  No label,
target or fitted statistic is part of a spec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "lcrc-feature-spec-v1"

#: Code dtype shared by both schemas.  ``int32`` is deliberate: the
#: Lipophilicity path vocabulary has 400 categories, so the historical
#: ``uint8`` path code would silently wrap at 256.  The tests assert that a
#: code above 255 is represented exactly.
STRUCTURAL_CODE_DTYPE = "int32"
STRUCTURAL_INDEX_DTYPE = "int64"
FEATURE_DTYPE = "float32"


@dataclass(frozen=True)
class FeatureField:
    name: str
    dim: int
    dtype: str
    description: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": int(self.dim),
            "dtype": self.dtype,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureField":
        return cls(
            name=str(d["name"]),
            dim=int(d["dim"]),
            dtype=str(d["dtype"]),
            description=str(d["description"]),
            source=str(d["source"]),
        )


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: str
    dtype: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TensorSpec":
        return cls(
            name=str(d["name"]),
            shape=str(d["shape"]),
            dtype=str(d["dtype"]),
            description=str(d["description"]),
        )


@dataclass(frozen=True)
class FeatureSpec:
    schema_version: str
    name: str
    description: str
    atom_type_names: tuple[str, ...]
    atom_type_metadata: tuple[dict[str, Any], ...]
    atom_unk_index: int | None
    atom_unknown_policy: str
    atom_type_encoding: str
    bond_type_names: tuple[str, ...]
    bond_type_metadata: tuple[dict[str, Any], ...]
    bond_unk_id: int | None
    bond_unknown_policy: str
    bond_type_encoding: str
    atom_feature_fields: tuple[FeatureField, ...]
    bond_feature_fields: tuple[FeatureField, ...]
    atom_scalar_fields: tuple[FeatureField, ...]
    global_feature_fields: tuple[FeatureField, ...]
    nbr_code_formula: str
    path_code_formula: str
    path_end_code_formula: str
    structural_code_dtype: str
    structural_index_dtype: str
    feature_dtype: str
    supports_disconnected: bool
    disconnected_policy: str
    atom_feature_source: str
    bond_feature_source: str
    global_feature_source: str
    source_provenance: dict[str, Any] = field(default_factory=dict)
    output_tensors: tuple[TensorSpec, ...] = ()

    # -- derived widths ----------------------------------------------------
    @property
    def n_atom_types(self) -> int:
        return len(self.atom_type_names)

    @property
    def n_bond_types(self) -> int:
        return len(self.bond_type_names)

    @property
    def n_nbr_codes(self) -> int:
        return self.n_atom_types * self.n_bond_types

    @property
    def n_path_codes(self) -> int:
        return self.n_atom_types * self.n_bond_types * self.n_bond_types

    @property
    def atom_feat_dim(self) -> int:
        return sum(f.dim for f in self.atom_feature_fields)

    @property
    def bond_feat_dim(self) -> int:
        return sum(f.dim for f in self.bond_feature_fields)

    @property
    def n_scalars(self) -> int:
        return sum(f.dim for f in self.atom_scalar_fields)

    @property
    def global_dim(self) -> int:
        return sum(f.dim for f in self.global_feature_fields)

    @property
    def code_max(self) -> int:
        return int(np.iinfo(np.dtype(self.structural_code_dtype)).max)

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema version {self.schema_version!r}")
        if not self.name:
            raise ValueError("spec name is empty")
        if self.n_atom_types < 1 or self.n_bond_types < 1:
            raise ValueError("atom/bond vocabularies must be non-empty")
        if len(self.atom_type_metadata) != self.n_atom_types:
            raise ValueError("atom_type_metadata length mismatch")
        if len(self.bond_type_metadata) != self.n_bond_types:
            raise ValueError("bond_type_metadata length mismatch")
        if len(set(self.atom_type_names)) != self.n_atom_types:
            raise ValueError("duplicate atom type names")
        if len(set(self.bond_type_names)) != self.n_bond_types:
            raise ValueError("duplicate bond type names")
        if self.atom_unk_index is not None and not (0 <= self.atom_unk_index < self.n_atom_types):
            raise ValueError("atom_unk_index outside vocabulary")
        # Bond ids are one-based by construction.
        if self.bond_unk_id is not None and not (1 <= self.bond_unk_id <= self.n_bond_types):
            raise ValueError("bond_unk_id outside vocabulary")
        for group, label in (
            (self.atom_feature_fields, "atom feature"),
            (self.bond_feature_fields, "bond feature"),
            (self.atom_scalar_fields, "atom scalar"),
            (self.global_feature_fields, "global feature"),
        ):
            if not group:
                raise ValueError(f"{label} fields are empty")
            for f in group:
                if f.dim <= 0:
                    raise ValueError(f"{label} {f.name!r} has non-positive dim")
                if f.dtype != self.feature_dtype:
                    raise ValueError(f"{label} {f.name!r} dtype {f.dtype!r} != {self.feature_dtype!r}")
            if len({f.name for f in group}) != len(group):
                raise ValueError(f"duplicate {label} names")
        if self.structural_code_dtype not in {"int32", "int64"}:
            raise ValueError("structural_code_dtype must be int32 or int64")
        if np.dtype(self.structural_code_dtype).itemsize < 4:
            raise ValueError("structural code dtype must be at least 4 bytes")
        if np.dtype(self.structural_index_dtype).itemsize < 4:
            raise ValueError("structural index dtype must be at least 4 bytes")
        if self.n_path_codes > self.code_max + 1:
            raise ValueError(
                f"path vocabulary {self.n_path_codes} exceeds dtype {self.structural_code_dtype} range"
            )
        if self.supports_disconnected and "finite" not in self.disconnected_policy.lower():
            raise ValueError("disconnected policy must explicitly describe finite treatment")
        if not self.supports_disconnected and "reject" not in self.disconnected_policy.lower():
            raise ValueError("connected-only schema must reject disconnected molecules")

    # -- serialisation -----------------------------------------------------
    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "atom_type_names": list(self.atom_type_names),
            "atom_type_metadata": list(self.atom_type_metadata),
            "atom_unk_index": self.atom_unk_index,
            "atom_unknown_policy": self.atom_unknown_policy,
            "atom_type_encoding": self.atom_type_encoding,
            "bond_type_names": list(self.bond_type_names),
            "bond_type_metadata": list(self.bond_type_metadata),
            "bond_unk_id": self.bond_unk_id,
            "bond_unknown_policy": self.bond_unknown_policy,
            "bond_type_encoding": self.bond_type_encoding,
            "atom_feature_fields": [f.to_dict() for f in self.atom_feature_fields],
            "bond_feature_fields": [f.to_dict() for f in self.bond_feature_fields],
            "atom_scalar_fields": [f.to_dict() for f in self.atom_scalar_fields],
            "global_feature_fields": [f.to_dict() for f in self.global_feature_fields],
            "nbr_code_formula": self.nbr_code_formula,
            "path_code_formula": self.path_code_formula,
            "path_end_code_formula": self.path_end_code_formula,
            "structural_code_dtype": self.structural_code_dtype,
            "structural_index_dtype": self.structural_index_dtype,
            "feature_dtype": self.feature_dtype,
            "supports_disconnected": bool(self.supports_disconnected),
            "disconnected_policy": self.disconnected_policy,
            "atom_feature_source": self.atom_feature_source,
            "bond_feature_source": self.bond_feature_source,
            "global_feature_source": self.global_feature_source,
            "source_provenance": dict(self.source_provenance),
            "output_tensors": [t.to_dict() for t in self.output_tensors],
            "dimensions": {
                "n_atom_types": self.n_atom_types,
                "n_bond_types": self.n_bond_types,
                "n_nbr_codes": self.n_nbr_codes,
                "n_path_codes": self.n_path_codes,
                "atom_feat_dim": self.atom_feat_dim,
                "bond_feat_dim": self.bond_feat_dim,
                "n_scalars": self.n_scalars,
                "global_dim": self.global_dim,
                "code_max": self.code_max,
            },
        }
        if include_hash:
            d["spec_hash"] = self.hash()
        return d

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeatureSpec":
        if "dimensions" in d:
            d = {k: v for k, v in d.items() if k not in {"dimensions", "spec_hash"}}
        spec = cls(
            schema_version=str(d["schema_version"]),
            name=str(d["name"]),
            description=str(d["description"]),
            atom_type_names=tuple(str(x) for x in d["atom_type_names"]),
            atom_type_metadata=tuple(dict(x) for x in d["atom_type_metadata"]),
            atom_unk_index=None if d.get("atom_unk_index") is None else int(d["atom_unk_index"]),
            atom_unknown_policy=str(d["atom_unknown_policy"]),
            atom_type_encoding=str(d["atom_type_encoding"]),
            bond_type_names=tuple(str(x) for x in d["bond_type_names"]),
            bond_type_metadata=tuple(dict(x) for x in d["bond_type_metadata"]),
            bond_unk_id=None if d.get("bond_unk_id") is None else int(d["bond_unk_id"]),
            bond_unknown_policy=str(d["bond_unknown_policy"]),
            bond_type_encoding=str(d["bond_type_encoding"]),
            atom_feature_fields=tuple(FeatureField.from_dict(x) for x in d["atom_feature_fields"]),
            bond_feature_fields=tuple(FeatureField.from_dict(x) for x in d["bond_feature_fields"]),
            atom_scalar_fields=tuple(FeatureField.from_dict(x) for x in d["atom_scalar_fields"]),
            global_feature_fields=tuple(FeatureField.from_dict(x) for x in d["global_feature_fields"]),
            nbr_code_formula=str(d["nbr_code_formula"]),
            path_code_formula=str(d["path_code_formula"]),
            path_end_code_formula=str(d["path_end_code_formula"]),
            structural_code_dtype=str(d["structural_code_dtype"]),
            structural_index_dtype=str(d["structural_index_dtype"]),
            feature_dtype=str(d["feature_dtype"]),
            supports_disconnected=bool(d["supports_disconnected"]),
            disconnected_policy=str(d["disconnected_policy"]),
            atom_feature_source=str(d["atom_feature_source"]),
            bond_feature_source=str(d["bond_feature_source"]),
            global_feature_source=str(d["global_feature_source"]),
            source_provenance=dict(d.get("source_provenance", {})),
            output_tensors=tuple(TensorSpec.from_dict(x) for x in d.get("output_tensors", [])),
        )
        spec.validate()
        if "spec_hash" in d and str(d["spec_hash"]) != spec.hash():
            raise ValueError("spec hash mismatch on load")
        return spec


# ---------------------------------------------------------------------------
# output tensor contract (dtype/shape documentation, not a runtime container)
# ---------------------------------------------------------------------------


def _output_tensors(atom_dim: int, bond_dim: int, scalar_dim: int, global_dim: int) -> tuple[TensorSpec, ...]:
    code = STRUCTURAL_CODE_DTYPE
    idx = STRUCTURAL_INDEX_DTYPE
    f32 = FEATURE_DTYPE
    return (
        TensorSpec("atom_feat", f"[N, {atom_dim}]", f32, "per-atom feature matrix"),
        TensorSpec("atom_type", "[N]", idx, "categorical atom type id (0-based)"),
        TensorSpec("atom_scalar", f"[N, {scalar_dim}]", f32, "per-atom scalar vector before Fit-only standardisation"),
        TensorSpec("bond_feat", f"[M, {bond_dim}]", f32, "per-physical-bond feature matrix"),
        TensorSpec("bond_type", "[M]", idx, "categorical bond type id (1-based)"),
        TensorSpec("bond_index", "[M, 2]", idx, "physical bond endpoints, u < v, sorted"),
        TensorSpec("atom_ptr", "[B + 1]", idx, "molecule atom offsets"),
        TensorSpec("bond_ptr", "[B + 1]", idx, "molecule bond offsets"),
        TensorSpec("nbr_ptr", "[N_total + 1]", idx, "per-atom S1 code offsets"),
        TensorSpec("nbr_code", "[nnz_S1]", code, "joint (neighbour atom type, bond type) code"),
        TensorSpec("path_ptr", "[N_total + 1]", idx, "per-root S2 record offsets"),
        TensorSpec("path_cat", "[nnz_S2]", code, "joint (middle type, bond1, bond2) code"),
        TensorSpec("path_end63", "[nnz_S2]", code, "joint (end atom type, bond2) code"),
        TensorSpec("path_end", "[nnz_S2]", idx, "absolute end-atom index in the batch"),
        TensorSpec("path_closure", "[nnz_S2]", "int8", "1 iff the end atom is adjacent to the root"),
        TensorSpec("global", f"[B, {global_dim}]", f32, "graph-level structural statistics before Fit-only standardisation"),
    )


# ---------------------------------------------------------------------------
# legacy ZINC schema
# ---------------------------------------------------------------------------

_LEGACY_ATOM_NAMES = (
    "C",
    "O",
    "N",
    "F",
    "C_H1",
    "S",
    "Cl",
    "O_minus",
    "N_H1_plus",
    "Br",
    "N_H3_plus",
    "N_H2_plus",
    "N_plus",
    "N_minus",
    "S_minus",
    "I",
    "P",
    "O_H1_plus",
    "N_H1_minus",
    "O_plus",
    "S_plus",
)

_LEGACY_GLOBAL_FIELDS = (
    ("log1p_n_atoms", "log1p(number of atoms)"),
    ("log1p_n_bonds", "log1p(number of physical bonds)"),
    ("log1p_n_junctions", "log1p(number of unordered neighbour pairs at a centre)"),
    ("cyclomatic_rank", "n_bonds - n_atoms + 1 (ZINC subset molecules are connected)"),
    ("ring_atom_fraction", "fraction of atoms on a cycle"),
    ("ring_bond_fraction", "fraction of bonds on a cycle"),
    ("n_ring_systems", "number of biconnected blocks with at least three edges"),
    ("diameter", "maximum finite shortest-path distance"),
    ("mean_distance", "mean finite unordered-pair shortest-path distance"),
    ("frac_dist_ge3", "fraction of ordered pairs at finite distance >= 3"),
    ("bonds_per_atom", "n_bonds / n_atoms"),
    ("log1p_n_ring_atoms", "log1p(number of ring atoms)"),
    ("max_degree", "maximum heavy-atom degree"),
    ("terminal_atom_fraction", "fraction of degree-1 atoms"),
    ("junction_per_bond", "n_junctions / n_bonds"),
    ("frac_junction_centers_ring", "fraction of junction centres on a cycle"),
    ("frac_deg2_junctions", "fraction of junctions whose centre has degree 2"),
    ("frac_deg3_junctions", "fraction of junctions whose centre has degree 3"),
    ("frac_deg4_junctions", "fraction of junctions whose centre has degree >= 4"),
    ("bond_type0_frac", "fraction of bonds with P3 bond type 0 (unused in ZINC)"),
    ("bond_type1_frac", "fraction of single bonds"),
    ("bond_type2_frac", "fraction of double bonds"),
    ("bond_type3_frac", "fraction of triple bonds"),
    ("ring_size3_frac", "ring-atom fraction with minimum ring size 3"),
    ("ring_size4_frac", "ring-atom fraction with minimum ring size 4"),
    ("ring_size5_frac", "ring-atom fraction with minimum ring size 5"),
    ("ring_size6_frac", "ring-atom fraction with minimum ring size 6"),
    ("ring_size7_frac", "ring-atom fraction with minimum ring size 7"),
    ("ring_size8plus_frac", "ring-atom fraction with minimum ring size >= 8"),
    ("fused_ring_atom_fraction", "fraction of atoms with >= 3 ring bonds"),
    ("mean_pair_distance", "historical duplicate of mean_distance"),
    ("distinct_bond_types", "number of distinct bond types present"),
    ("distinct_atom_types", "number of distinct P3 atom types present"),
)


def legacy_zinc_spec(source_provenance: Mapping[str, Any] | None = None) -> FeatureSpec:
    atom_fields = (
        FeatureField("atom_type", 28, FEATURE_DTYPE, "P3/ZINC atom-type one-hot (28-slot source vocabulary; LCRC uses the first 21)", "P3 atom_feat block"),
        FeatureField("degree", 5, FEATURE_DTYPE, "degree one-hot 0..4 with degree >= 4 collapsed to the last slot (legacy behaviour)", "P3 atom_feat block"),
        FeatureField("in_ring", 1, FEATURE_DTYPE, "atom lies on a cycle", "P3 atom_feat block"),
        FeatureField("min_ring_size", 6, FEATURE_DTYPE, "minimum ring size 3..8+ one-hot; all-zero for acyclic atoms (legacy behaviour)", "P3 atom_feat block"),
        FeatureField("ring_bonds", 2, FEATURE_DTYPE, "ring-bond count and ring-bond fraction", "P3 atom_feat block"),
        FeatureField("bond_type_counts", 4, FEATURE_DTYPE, "incident bond-type counts over the 4 P3 bond slots", "P3 atom_feat block"),
        FeatureField("neighbour_types", 28, FEATURE_DTYPE, "radius-1 neighbour-type histogram", "P3 atom_feat block"),
        FeatureField("radius2_types", 28, FEATURE_DTYPE, "radius-2 atom-type histogram", "P3 atom_feat block"),
        FeatureField("radius2_stats", 4, FEATURE_DTYPE, "log1p(n_d2), log1p(2-paths), triangles, fused/spiro flag", "P3 atom_feat block"),
        FeatureField("roles", 2, FEATURE_DTYPE, "terminal flag and small-ring/fused role flag", "P3 atom_feat block"),
    )
    bond_fields = (
        FeatureField("bond_type", 4, FEATURE_DTYPE, "P3 bond-type one-hot over slots 0..3; ZINC uses 1..3", "P3 bond_feat block"),
        FeatureField("is_ring", 1, FEATURE_DTYPE, "bond lies on a cycle", "P3 bond_feat block"),
        FeatureField("exactly_one_endpoint_in_ring", 1, FEATURE_DTYPE, "exactly one endpoint is a ring atom", "P3 bond_feat block"),
        FeatureField("both_endpoints_in_ring", 1, FEATURE_DTYPE, "both endpoints are ring atoms", "P3 bond_feat block"),
    )
    scalar_fields = (
        FeatureField("log1p_n_d2", 1, FEATURE_DTYPE, "log1p(number of atoms at graph distance exactly 2)", "legacy graph statistic"),
        FeatureField("log1p_n_paths2", 1, FEATURE_DTYPE, "log1p(number of ordered non-backtracking 2-paths rooted at the atom)", "legacy S2 count"),
        FeatureField("closure_ratio", 1, FEATURE_DTYPE, "fraction of rooted 2-paths whose end atom is adjacent to the root", "legacy S2 count"),
        FeatureField("degree_over_4", 1, FEATURE_DTYPE, "degree / 4 (legacy scalar)", "legacy graph statistic"),
        FeatureField("in_ring", 1, FEATURE_DTYPE, "P3 in_ring bit", "P3 atom_feat block"),
        FeatureField("min_ring_size_over_8", 1, FEATURE_DTYPE, "minimum ring size / 8 (0 if acyclic)", "P3 atom_feat block"),
        FeatureField("ring_bond_count_over_4", 1, FEATURE_DTYPE, "ring-bond count / 4", "P3 atom_feat block"),
        FeatureField("terminal", 1, FEATURE_DTYPE, "P3 terminal role bit", "P3 atom_feat block"),
        FeatureField("fused_or_spiro", 1, FEATURE_DTYPE, "ring-bond count >= 3", "P3 atom_feat block"),
    )
    global_fields = tuple(
        FeatureField(name, 1, FEATURE_DTYPE, desc, "P3 global_feat block")
        for name, desc in _LEGACY_GLOBAL_FIELDS
    )
    spec = FeatureSpec(
        schema_version=SCHEMA_VERSION,
        name="legacy_zinc",
        description=(
            "Frozen ZINC-LCRC-A interface.  Replays the historical P3 atom/bond/global "
            "tensors and the historical S1/S2 codes exactly; connected molecules only."
        ),
        atom_type_names=_LEGACY_ATOM_NAMES,
        atom_type_metadata=tuple(
            {
                "index": i,
                "name": name,
                "source": "ZINC atom_dict.pickle idx2word",
                "is_unknown": False,
            }
            for i, name in enumerate(_LEGACY_ATOM_NAMES)
        ),
        atom_unk_index=None,
        atom_unknown_policy="reject: categorical atom type outside 0..20 is an error",
        atom_type_encoding="atom_type = argmax(P3 atom_feat[:, :21]); 0-based",
        bond_type_names=("SINGLE", "DOUBLE", "TRIPLE"),
        bond_type_metadata=tuple(
            {"id": i + 1, "name": name, "raw_index": i + 1, "source": "ZINC bond_dict.pickle"}
            for i, name in enumerate(("SINGLE", "DOUBLE", "TRIPLE"))
        ),
        bond_unk_id=None,
        bond_unknown_policy="reject: bond type outside 1..3 is an error",
        bond_type_encoding="bond_type is the historical one-based P3 value 1..3",
        atom_feature_fields=atom_fields,
        bond_feature_fields=bond_fields,
        atom_scalar_fields=scalar_fields,
        global_feature_fields=global_fields,
        nbr_code_formula="atom_type[neighbour] * 3 + (bond_type - 1)",
        path_code_formula="atom_type[middle] * 9 + (bond_type_1 - 1) * 3 + (bond_type_2 - 1)",
        path_end_code_formula="atom_type[end] * 3 + (bond_type_2 - 1)",
        structural_code_dtype=STRUCTURAL_CODE_DTYPE,
        structural_index_dtype=STRUCTURAL_INDEX_DTYPE,
        feature_dtype=FEATURE_DTYPE,
        supports_disconnected=False,
        disconnected_policy="reject: legacy global33 uses connected-graph formulas; all ZINC subset train/valid molecules are connected",
        atom_feature_source="frozen P3 MolObjects.atom_feat (108-d)",
        bond_feature_source="frozen P3 MolObjects.bond_feat (7-d)",
        global_feature_source="frozen P3 MolObjects.global_feat (33-d)",
        source_provenance=dict(source_provenance or {}),
        output_tensors=_output_tensors(108, 7, 9, 33),
    )
    spec.validate()
    return spec


# ---------------------------------------------------------------------------
# lipo_v1 schema
# ---------------------------------------------------------------------------

_LIPO_ATOM_NAMES = (
    "B",
    "Br",
    "C",
    "C_aromatic",
    "Cl",
    "F",
    "I",
    "N",
    "N_aromatic",
    "O",
    "O_aromatic",
    "P",
    "S",
    "S_aromatic",
    "Si",
    "UNK",
)

_LIPO_ATOM_METADATA = (
    {"index": 0, "name": "B", "element": "B", "aromatic": False, "raw_index": 0, "is_unknown": False},
    {"index": 1, "name": "Br", "element": "Br", "aromatic": False, "raw_index": 1, "is_unknown": False},
    {"index": 2, "name": "C", "element": "C", "aromatic": False, "raw_index": 2, "is_unknown": False},
    {"index": 3, "name": "C_aromatic", "element": "C", "aromatic": True, "raw_index": 3, "is_unknown": False},
    {"index": 4, "name": "Cl", "element": "Cl", "aromatic": False, "raw_index": 4, "is_unknown": False},
    {"index": 5, "name": "F", "element": "F", "aromatic": False, "raw_index": 5, "is_unknown": False},
    {"index": 6, "name": "I", "element": "I", "aromatic": False, "raw_index": 6, "is_unknown": False},
    {"index": 7, "name": "N", "element": "N", "aromatic": False, "raw_index": 7, "is_unknown": False},
    {"index": 8, "name": "N_aromatic", "element": "N", "aromatic": True, "raw_index": 8, "is_unknown": False},
    {"index": 9, "name": "O", "element": "O", "aromatic": False, "raw_index": 9, "is_unknown": False},
    {"index": 10, "name": "O_aromatic", "element": "O", "aromatic": True, "raw_index": 10, "is_unknown": False},
    {"index": 11, "name": "P", "element": "P", "aromatic": False, "raw_index": 11, "is_unknown": False},
    {"index": 12, "name": "S", "element": "S", "aromatic": False, "raw_index": 12, "is_unknown": False},
    {"index": 13, "name": "S_aromatic", "element": "S", "aromatic": True, "raw_index": 13, "is_unknown": False},
    {"index": 14, "name": "Si", "element": "Si", "aromatic": False, "raw_index": 14, "is_unknown": False},
    {"index": 15, "name": "UNK", "element": None, "aromatic": False, "raw_index": 15, "is_unknown": True},
)

_LIPO_BOND_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "UNK")
_LIPO_BOND_METADATA = (
    {"id": 1, "name": "SINGLE", "raw_index": 0, "is_unknown": False},
    {"id": 2, "name": "DOUBLE", "raw_index": 1, "is_unknown": False},
    {"id": 3, "name": "TRIPLE", "raw_index": 2, "is_unknown": False},
    {"id": 4, "name": "AROMATIC", "raw_index": 3, "is_unknown": False},
    {"id": 5, "name": "UNK", "raw_index": 4, "is_unknown": True},
)

_LIPO_ATOM_FIELDS = (
    FeatureField("atom_type", 16, FEATURE_DTYPE, "one-hot over element x aromatic class (15 source classes + UNK)", "raw atom_class"),
    FeatureField("formal_charge", 6, FEATURE_DTYPE, "one-hot over -2,-1,0,+1,+2,UNK", "raw atom_charge index"),
    FeatureField("n_hydrogens", 6, FEATURE_DTYPE, "one-hot over 0,1,2,3,4,UNK explicit+implicit H count", "raw atom_h index"),
    FeatureField("degree", 6, FEATURE_DTYPE, "heavy-atom degree one-hot 0..4,>=5", "physical-bond graph"),
    FeatureField("in_ring", 1, FEATURE_DTYPE, "atom lies on at least one cycle", "physical-bond graph"),
    FeatureField("min_ring_size", 7, FEATURE_DTYPE, "one-hot acyclic,3,4,5,6,7,>=8 minimum ring size", "physical-bond graph"),
    FeatureField("ring_bond_count_over_4", 1, FEATURE_DTYPE, "ring-bond count / 4", "physical-bond graph"),
    FeatureField("ring_bond_fraction", 1, FEATURE_DTYPE, "ring-bond count / degree (0 for isolated atoms)", "physical-bond graph"),
    FeatureField("incident_bond_class_fraction", 5, FEATURE_DTYPE, "fraction of incident bonds in SINGLE,DOUBLE,TRIPLE,AROMATIC,UNK", "raw edge_class"),
    FeatureField("is_terminal", 1, FEATURE_DTYPE, "degree == 1", "physical-bond graph"),
    FeatureField("is_fused_or_spiro", 1, FEATURE_DTYPE, "ring-bond count >= 3", "physical-bond graph"),
)

_LIPO_BOND_FIELDS = (
    FeatureField("bond_class", 5, FEATURE_DTYPE, "one-hot SINGLE,DOUBLE,TRIPLE,AROMATIC,UNK", "raw edge_class"),
    FeatureField("is_ring", 1, FEATURE_DTYPE, "bond lies on at least one cycle", "physical-bond graph"),
    FeatureField("exactly_one_endpoint_in_ring", 1, FEATURE_DTYPE, "exactly one endpoint is a ring atom", "physical-bond graph"),
    FeatureField("both_endpoints_in_ring", 1, FEATURE_DTYPE, "both endpoints are ring atoms", "physical-bond graph"),
    FeatureField("stereo", 5, FEATURE_DTYPE, "one-hot NONE,E,Z,OTHER,UNK; encoder-only, not part of S1/S2 joint codes", "raw edge_stereo"),
)

_LIPO_SCALAR_FIELDS = (
    FeatureField("log1p_n_d2", 1, FEATURE_DTYPE, "log1p(number of atoms at graph distance exactly 2)", "physical-bond graph"),
    FeatureField("log1p_n_paths2", 1, FEATURE_DTYPE, "log1p(number of ordered non-backtracking 2-paths rooted at the atom)", "S2 enumeration"),
    FeatureField("closure_ratio", 1, FEATURE_DTYPE, "fraction of rooted 2-paths whose end atom is adjacent to the root", "S2 enumeration"),
    FeatureField("log1p_degree", 1, FEATURE_DTYPE, "log1p(heavy-atom degree)", "physical-bond graph"),
    FeatureField("in_ring", 1, FEATURE_DTYPE, "ring-atom flag", "physical-bond graph"),
    FeatureField("min_ring_size_over_8", 1, FEATURE_DTYPE, "minimum ring size / 8 (0 if acyclic)", "physical-bond graph"),
    FeatureField("ring_bond_count_over_4", 1, FEATURE_DTYPE, "ring-bond count / 4", "physical-bond graph"),
    FeatureField("is_terminal", 1, FEATURE_DTYPE, "degree == 1", "physical-bond graph"),
    FeatureField("is_fused_or_spiro", 1, FEATURE_DTYPE, "ring-bond count >= 3", "physical-bond graph"),
)

_LIPO_GLOBAL_FIELDS = (
    FeatureField("log1p_n_atoms", 1, FEATURE_DTYPE, "log1p(number of atoms)", "molecular graph"),
    FeatureField("log1p_n_bonds", 1, FEATURE_DTYPE, "log1p(number of physical bonds)", "molecular graph"),
    FeatureField("log1p_n_junctions", 1, FEATURE_DTYPE, "log1p(number of unordered neighbour pairs at a centre)", "molecular graph"),
    FeatureField("cyclomatic_number", 1, FEATURE_DTYPE, "n_bonds - n_atoms + n_components (first Betti number)", "molecular graph"),
    FeatureField("n_components", 1, FEATURE_DTYPE, "number of connected components", "molecular graph"),
    FeatureField("ring_atom_fraction", 1, FEATURE_DTYPE, "fraction of atoms on a cycle", "ring analysis"),
    FeatureField("ring_bond_fraction", 1, FEATURE_DTYPE, "fraction of bonds on a cycle", "ring analysis"),
    FeatureField("n_ring_systems", 1, FEATURE_DTYPE, "number of biconnected blocks with at least three edges", "ring analysis"),
    FeatureField("diameter_max_finite", 1, FEATURE_DTYPE, "maximum finite shortest-path distance within components (0 for a single atom)", "all-pairs BFS"),
    FeatureField("mean_finite_pair_distance", 1, FEATURE_DTYPE, "mean finite unordered-pair shortest-path distance; unreachable pairs excluded", "all-pairs BFS"),
    FeatureField("frac_unreachable_pairs", 1, FEATURE_DTYPE, "unreachable unordered pairs / all unordered pairs (0 for connected molecules)", "all-pairs BFS"),
    FeatureField("frac_finite_pairs_ge3", 1, FEATURE_DTYPE, "finite unordered pairs at distance >= 3 / all unordered pairs; unreachable pairs are NOT counted as >= 3", "all-pairs BFS"),
    FeatureField("bonds_per_atom", 1, FEATURE_DTYPE, "n_bonds / n_atoms", "molecular graph"),
    FeatureField("log1p_n_ring_atoms", 1, FEATURE_DTYPE, "log1p(number of ring atoms)", "ring analysis"),
    FeatureField("max_degree", 1, FEATURE_DTYPE, "maximum heavy-atom degree", "molecular graph"),
    FeatureField("terminal_atom_fraction", 1, FEATURE_DTYPE, "fraction of degree-1 atoms", "molecular graph"),
    FeatureField("junction_per_bond", 1, FEATURE_DTYPE, "n_junctions / n_bonds", "molecular graph"),
    FeatureField("frac_junction_centers_ring", 1, FEATURE_DTYPE, "fraction of junction centres on a cycle", "ring analysis"),
    FeatureField("frac_deg2_junctions", 1, FEATURE_DTYPE, "fraction of junctions whose centre has degree 2", "molecular graph"),
    FeatureField("frac_deg3plus_junctions", 1, FEATURE_DTYPE, "fraction of junctions whose centre has degree >= 3", "molecular graph"),
    FeatureField("bond_single_fraction", 1, FEATURE_DTYPE, "fraction of SINGLE bonds", "raw edge_class"),
    FeatureField("bond_double_fraction", 1, FEATURE_DTYPE, "fraction of DOUBLE bonds", "raw edge_class"),
    FeatureField("bond_triple_fraction", 1, FEATURE_DTYPE, "fraction of TRIPLE bonds", "raw edge_class"),
    FeatureField("bond_aromatic_fraction", 1, FEATURE_DTYPE, "fraction of AROMATIC bonds", "raw edge_class"),
    FeatureField("bond_unknown_fraction", 1, FEATURE_DTYPE, "fraction of UNK bonds", "raw edge_class"),
    FeatureField("ring_size3_fraction", 1, FEATURE_DTYPE, "ring-atom fraction with minimum ring size 3", "ring analysis"),
    FeatureField("ring_size4_fraction", 1, FEATURE_DTYPE, "ring-atom fraction with minimum ring size 4", "ring analysis"),
    FeatureField("ring_size5_fraction", 1, FEATURE_DTYPE, "ring-atom fraction with minimum ring size 5", "ring analysis"),
    FeatureField("ring_size6_fraction", 1, FEATURE_DTYPE, "ring-atom fraction with minimum ring size 6", "ring analysis"),
    FeatureField("ring_size7_fraction", 1, FEATURE_DTYPE, "ring-atom fraction with minimum ring size 7", "ring analysis"),
    FeatureField("ring_size8plus_fraction", 1, FEATURE_DTYPE, "ring-atom fraction with minimum ring size >= 8", "ring analysis"),
    FeatureField("fused_ring_atom_fraction", 1, FEATURE_DTYPE, "fraction of atoms with >= 3 ring bonds", "ring analysis"),
    FeatureField("aromatic_atom_fraction", 1, FEATURE_DTYPE, "fraction of atoms whose type is aromatic", "raw atom_class"),
    FeatureField("unknown_atom_fraction", 1, FEATURE_DTYPE, "fraction of atoms mapped to UNK", "raw atom_class"),
    FeatureField("charged_atom_fraction", 1, FEATURE_DTYPE, "fraction of atoms with non-zero known formal charge", "raw atom_charge"),
    FeatureField("mean_formal_charge", 1, FEATURE_DTYPE, "mean known formal charge; UNK charge treated as 0", "raw atom_charge"),
    FeatureField("mean_hydrogens", 1, FEATURE_DTYPE, "mean known hydrogen count; UNK H treated as 0", "raw atom_h"),
    FeatureField("distinct_atom_types", 1, FEATURE_DTYPE, "number of distinct categorical atom types present", "atom_type"),
    FeatureField("distinct_bond_types", 1, FEATURE_DTYPE, "number of distinct categorical bond types present", "bond_type"),
    FeatureField("max_abs_formal_charge", 1, FEATURE_DTYPE, "maximum absolute known formal charge; UNK charge treated as 0", "raw atom_charge"),
    FeatureField("unknown_charge_fraction", 1, FEATURE_DTYPE, "fraction of atoms with UNK charge bucket", "raw atom_charge"),
    FeatureField("unknown_hydrogen_fraction", 1, FEATURE_DTYPE, "fraction of atoms with UNK hydrogen bucket", "raw atom_h"),
)


def lipo_v1_spec(source_provenance: Mapping[str, Any] | None = None) -> FeatureSpec:
    spec = FeatureSpec(
        schema_version=SCHEMA_VERSION,
        name="lipo_v1",
        description=(
            "Explicit Lipophilicity structure schema built from the committed label-free raw graph. "
            "Atom/bond/global widths and vocabularies are independent of legacy_zinc.  "
            "Disconnected molecules and isolated atoms are supported with finite statistics."
        ),
        atom_type_names=_LIPO_ATOM_NAMES,
        atom_type_metadata=_LIPO_ATOM_METADATA,
        atom_unk_index=15,
        atom_unknown_policy="map_to_UNK: raw atom_class 15 is the source UNK category; raw classes 0..14 keep their chemical name; MASK (16) is rejected",
        atom_type_encoding="atom_type = raw atom_class (0..14 named element x aromatic class, 15 UNK); 0-based",
        bond_type_names=_LIPO_BOND_NAMES,
        bond_type_metadata=_LIPO_BOND_METADATA,
        bond_unk_id=5,
        bond_unknown_policy="map_to_UNK: raw edge_class 4 maps to UNK id 5; raw classes 0..3 map to ids 1..4",
        bond_type_encoding="bond_type = raw edge_class + 1 (SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4, UNK=5); one-based",
        atom_feature_fields=_LIPO_ATOM_FIELDS,
        bond_feature_fields=_LIPO_BOND_FIELDS,
        atom_scalar_fields=_LIPO_SCALAR_FIELDS,
        global_feature_fields=_LIPO_GLOBAL_FIELDS,
        nbr_code_formula="atom_type[neighbour] * 5 + (bond_type - 1)",
        path_code_formula="atom_type[middle] * 25 + (bond_type_1 - 1) * 5 + (bond_type_2 - 1)",
        path_end_code_formula="atom_type[end] * 5 + (bond_type_2 - 1)",
        structural_code_dtype=STRUCTURAL_CODE_DTYPE,
        structural_index_dtype=STRUCTURAL_INDEX_DTYPE,
        feature_dtype=FEATURE_DTYPE,
        supports_disconnected=True,
        disconnected_policy=(
            "supported with finite statistics: unreachable pairs are excluded from diameter/mean distance, "
            "reported separately via frac_unreachable_pairs, and never counted as a finite distance >= 3"
        ),
        atom_feature_source="lipo_v1 adapter from raw atom_class/atom_charge/atom_h + physical-bond graph",
        bond_feature_source="lipo_v1 adapter from raw edge_class/edge_stereo + physical-bond graph",
        global_feature_source="lipo_v1 graph statistics with explicit disconnected handling",
        source_provenance=dict(source_provenance or {}),
        output_tensors=_output_tensors(51, 13, 9, len(_LIPO_GLOBAL_FIELDS)),
    )
    spec.validate()
    return spec


# ---------------------------------------------------------------------------
# registry / serialisation helpers
# ---------------------------------------------------------------------------

BUILDERS = {
    "legacy_zinc": legacy_zinc_spec,
    "lipo_v1": lipo_v1_spec,
}


def build_spec(name: str, source_provenance: Mapping[str, Any] | None = None) -> FeatureSpec:
    if name not in BUILDERS:
        raise KeyError(f"unknown feature spec {name!r}; known: {sorted(BUILDERS)}")
    return BUILDERS[name](source_provenance)


def save_spec(spec: FeatureSpec, path) -> str:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec.hash()


def load_spec(path) -> FeatureSpec:
    from pathlib import Path

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureSpec.from_dict(d)


__all__ = [
    "SCHEMA_VERSION",
    "STRUCTURAL_CODE_DTYPE",
    "STRUCTURAL_INDEX_DTYPE",
    "FEATURE_DTYPE",
    "FeatureField",
    "TensorSpec",
    "FeatureSpec",
    "legacy_zinc_spec",
    "lipo_v1_spec",
    "build_spec",
    "save_spec",
    "load_spec",
]
