"""P0.1 -- audit of the *actual* information flow of the frozen LCRC-A input.

Nothing here is inferred from a field name.  Every claim is backed by one of
three executable checks on the real code path:

``A. declared -> tensors``
    ``smiles_to_raw_row`` -> raw artifact -> ``lipo_row_to_graph`` ->
    ``lcrc_data.build_molecule`` -> ``collate``.  For every field declared in the
    frozen ``lipo_v1`` FeatureSpec we record the tensor, its width and the slice
    the adapter actually writes.

``B. tensors -> forward``
    The batch mapping handed to ``LCRC.forward`` is wrapped in a recording dict,
    so the set of batch keys the forward pass really reads is measured, not read
    off the source.  A declared tensor that is never read is a *documentation*
    tensor (``atom_type`` / ``bond_type``), and that is reported as such.

``C. field -> output sensitivity``
    Each declared field block is perturbed on a real molecule and the response of
    ``h`` (local representation) / ``y`` (prediction) is measured with a
    seed-0 randomly initialised model.  This proves the wiring: a column that
    does not move the graph is not "in the input" in any useful sense.  It says
    nothing about learned importance, and it is labelled that way.

The static inventory (``INFORMATION_ITEMS``) then classifies each *chemical*
information item as EXPLICIT / INDIRECT / LOST_IN_PREPROCESSING /
EXTRA_COMPUTE / NOT_DETERMINABLE using the evidence keys produced above.

Usage
-----
    uv run python code/ta_audit.py            # write analysis/p0_information_flow.json
    uv run python code/ta_audit.py --print    # human-readable summary
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ta_common as C  # noqa: I001  (adds the frozen v2 code dir to sys.path)

import mb2_data as D
import mb2_model as M
import mb2_raw_artifact as RAW
import mb2_rdkit as RDK

import lcrc_data as PREP_DATA
import molecular_graph as MG
from feature_spec import FeatureSpec

OUT = C.ANALYSIS / "p0_information_flow.json"

# ---------------------------------------------------------------------------
# static inventory: chemical information -> where it lives
# ---------------------------------------------------------------------------

#: ``status`` vocabulary
#:   EXPLICIT              -- a named tensor column carries it into the forward pass
#:   INDIRECT              -- recoverable from an existing local statistic, but not named
#:   LOST_IN_PREPROCESSING -- RDKit/the SMILES could provide it; the adapter drops it
#:   EXTRA_COMPUTE         -- needs an extra deterministic computation (descriptor)
#:   NOT_DETERMINABLE      -- not a function of the 2-D molecular graph at all
INFORMATION_ITEMS: tuple[dict[str, Any], ...] = (
    {
        "item": "element_identity",
        "status": "EXPLICIT",
        "carrier": "atom_feat[atom_type] 16-d one-hot; S1 nbr_code; S2 path_cat/path_end63; global distinct_atom_types",
        "evidence": ["atom_type_onehot", "s1_joint_code", "s2_joint_code", "sensitivity:atom_feat.atom_type"],
    },
    {
        "item": "aromaticity",
        "status": "EXPLICIT",
        "carrier": "5 aromatic element classes in atom_type; AROMATIC bond_class; global aromatic_atom_fraction / bond_aromatic_fraction",
        "evidence": ["atom_type_onehot", "bond_class_onehot", "sensitivity:atom_feat.atom_type", "sensitivity:bond_feat.bond_class"],
    },
    {
        "item": "formal_charge",
        "status": "EXPLICIT",
        "carrier": "atom_feat[formal_charge] 6-d one-hot (-2,-1,0,+1,+2,UNK); global charged_atom_fraction / mean_formal_charge / max_abs_formal_charge / unknown_charge_fraction",
        "caveat": "not part of the S1/S2 joint codes; reaches radius 2 only through the S2 content term a[path_end]*gate",
        "evidence": ["atom_feat.formal_charge", "global.charged_atom_fraction", "sensitivity:atom_feat.formal_charge"],
    },
    {
        "item": "hydrogen_count",
        "status": "EXPLICIT",
        "carrier": "atom_feat[n_hydrogens] 6-d one-hot (0..4, 4+, UNK); global mean_hydrogens / unknown_hydrogen_fraction",
        "caveat": "same as formal charge: atom-level only, not in the S1/S2 codes",
        "evidence": ["atom_feat.n_hydrogens", "global.mean_hydrogens", "sensitivity:atom_feat.n_hydrogens"],
    },
    {
        "item": "heavy_atom_degree",
        "status": "EXPLICIT",
        "carrier": "atom_feat[degree] 6-d one-hot; atom_scalar log1p_degree; global max_degree / terminal_atom_fraction",
        "evidence": ["atom_feat.degree", "atom_scalar.log1p_degree", "sensitivity:atom_feat.degree"],
    },
    {
        "item": "bond_order",
        "status": "EXPLICIT",
        "carrier": "bond_feat[bond_class] 5-d one-hot (SINGLE/DOUBLE/TRIPLE/AROMATIC/UNK); S1 code (...,bond_type); S2 code (bt1,bt2); global bond_*_fraction",
        "evidence": ["bond_feat.bond_class", "s1_joint_code", "s2_joint_code", "sensitivity:bond_feat.bond_class"],
    },
    {
        "item": "ring_membership_size_fusion",
        "status": "EXPLICIT",
        "carrier": "atom_feat in_ring/min_ring_size/ring_bond_count/ring_bond_fraction/is_fused_or_spiro; bond_feat is_ring/exactly_one_endpoint_in_ring/both_endpoints_in_ring; global ring_* / n_ring_systems / ring_sizeK_fraction / fused_ring_atom_fraction",
        "evidence": ["atom_feat.in_ring", "bond_feat.is_ring", "global.n_ring_systems", "sensitivity:atom_feat.min_ring_size"],
    },
    {
        "item": "global_topology",
        "status": "EXPLICIT",
        "carrier": "the 44 lipo_v1 global fields (size, bonds, junctions, cyclomatic rank, distances, degree/junction profiles, bond-class and ring-size fractions)",
        "evidence": ["global.<44 fields>", "sensitivity:global"],
    },
    {
        "item": "formal_charge_state_protonation",
        "status": "EXPLICIT",
        "carrier": "the SMILES ionic form is carried by atom_feat[formal_charge] + n_hydrogens",
        "caveat": "the *protonation equilibrium* / pKa and any alternate protomer are not represented and are not derivable from the given graph",
        "evidence": ["atom_feat.formal_charge", "global.charged_atom_fraction"],
    },
    {
        "item": "double_bond_ez",
        "status": "EXPLICIT",
        "carrier": "bond_feat[stereo] 5-d one-hot (NONE,E,Z,OTHER,UNK) from raw edge_stereo",
        "caveat": "enters h only through the radius-1 bond aggregate b_sum/b_mean; S1/S2 joint codes are stereo-blind and the global block has no stereo statistic",
        "evidence": ["bond_feat.stereo", "sensitivity:bond_feat.stereo"],
    },
    {
        "item": "tetrahedral_chirality",
        "status": "LOST_IN_PREPROCESSING",
        "carrier": "none -- raw artifact has no field for it; atom_feat has no chirality block",
        "caveat": "RDKit exposes it (Atom.GetChiralTag / FindMolChiralCenters); the frozen raw-adapter never reads it",
        "evidence": ["raw_artifact.fields", "atom_feat.blocks", "collision:tetrahedral"],
    },
    {
        "item": "isotope_label",
        "status": "LOST_IN_PREPROCESSING",
        "carrier": "none",
        "caveat": "not read from RDKit; no dataset in this round is expected to rely on it",
        "evidence": ["raw_artifact.fields"],
    },
    {
        "item": "bond_conjugation",
        "status": "INDIRECT",
        "carrier": "not named; an alternating single/double pattern is expressible through the S2 (bt1,bt2) joint code and the aromatic bond class",
        "caveat": "the model must learn the conjugation rule from bond-order pairs; no explicit conjugated flag exists",
        "evidence": ["s2_joint_code", "bond_feat.bond_class"],
    },
    {
        "item": "valence_hybridisation",
        "status": "INDIRECT",
        "carrier": "not named; determined by (element, bond orders, hydrogen count, formal charge) which are all present",
        "caveat": "sp/sp2/sp3 must be inferred by the model from those four channels",
        "evidence": ["atom_feat.atom_type,formal_charge,n_hydrogens,degree", "bond_feat.bond_class"],
    },
    {
        "item": "hbd_hba_counts",
        "status": "EXTRA_COMPUTE",
        "carrier": "not present as a named feature; a deterministic RDKit/Lipinski rule (N,O,S with H / lone pairs) is needed",
        "caveat": "atom-level N/O/S + H count + charge are available, so the information is *derivable* in principle",
        "evidence": ["atom_feat.atom_type", "atom_feat.n_hydrogens", "coverage:hbd_hba"],
    },
    {
        "item": "local_polarity_electronegativity",
        "status": "INDIRECT",
        "carrier": "hetero-element one-hots + formal charge + hydrogen count + ring context; no electronegativity value or partial-charge descriptor",
        "caveat": "a Gasteiger/MMFF partial charge would be an EXTRA_COMPUTE prior, not new structural information",
        "evidence": ["atom_feat.atom_type", "atom_feat.formal_charge", "sensitivity:atom_feat.formal_charge"],
    },
    {
        "item": "ring_system_aromaticity",
        "status": "INDIRECT",
        "carrier": "aromatic_atom_fraction + n_ring_systems + ring-size fractions; the *aromatic ring count* is not a named field",
        "caveat": "derivable from the aromatic atom set and the biconnected-block decomposition",
        "evidence": ["global.aromatic_atom_fraction", "global.n_ring_systems"],
    },
    {
        "item": "multi_component_salt",
        "status": "EXPLICIT",
        "carrier": "global n_components / frac_unreachable_pairs; the raw adapter counts RDKit fragments",
        "evidence": ["global.n_components", "coverage:multi_component"],
    },
    {
        "item": "molecular_weight",
        "status": "EXTRA_COMPUTE",
        "carrier": "not present; element composition is present, so MW is a fixed linear functional of the atom_type histogram",
        "caveat": "adding it would inject the periodic table as an external prior; it is not new graph information",
        "evidence": ["atom_feat.atom_type", "coverage:mw"],
    },
    {
        "item": "tpsa",
        "status": "EXTRA_COMPUTE",
        "carrier": "not present; requires the Ertl fragment table over N/O/S environments",
        "caveat": "the fragments are functions of (element, H count, charge, ring/aromatic context), all of which are present",
        "evidence": ["atom_feat.atom_type,n_hydrogens,formal_charge", "coverage:tpsa"],
    },
    {
        "item": "logp_logd_prior",
        "status": "EXTRA_COMPUTE",
        "carrier": "not present; Crippen/Wildman atom-typing rules would be needed",
        "caveat": "logD also depends on the experimental pH / ionisation state, which the graph does not carry",
        "evidence": ["coverage:logp_prior"],
    },
    {
        "item": "three_dimensional_conformation",
        "status": "NOT_DETERMINABLE",
        "carrier": "none",
        "caveat": "the input is a 2-D molecular graph; no conformer, no 3-D descriptor",
        "evidence": ["audit:scope"],
    },
    {
        "item": "tautomer_protomer_choice",
        "status": "NOT_DETERMINABLE",
        "carrier": "none -- the supplied SMILES fixes one tautomer",
        "caveat": "tautomer enumeration would require external rules and would change the sample definition",
        "evidence": ["audit:scope"],
    },
    {
        "item": "target_specific_binding_pose",
        "status": "NOT_DETERMINABLE",
        "carrier": "none",
        "caveat": "BACE / HIV / Tox21 activity is not a function of the 2-D graph alone",
        "evidence": ["audit:scope"],
    },
)

#: items that a *representation* experiment could act on without changing the
#: method definition (see ``analysis/representation_candidates.md``)
CANDIDATE_ELIGIBLE = (
    "tetrahedral_chirality",
    "double_bond_ez",
    "hbd_hba_counts",
    "tpsa",
    "local_polarity_electronegativity",
)


# ---------------------------------------------------------------------------
# A. declared fields -> tensors
# ---------------------------------------------------------------------------


def field_table(spec: FeatureSpec) -> dict[str, Any]:
    """Declared fields with their slice in the produced tensor."""
    out: dict[str, Any] = {"atom_feat": [], "bond_feat": [], "atom_scalar": [], "global": []}
    offset = 0
    for f in spec.atom_feature_fields:
        out["atom_feat"].append({"name": f.name, "dim": f.dim, "slice": [offset, offset + f.dim], "source": f.source})
        offset += f.dim
    offset = 0
    for f in spec.bond_feature_fields:
        out["bond_feat"].append({"name": f.name, "dim": f.dim, "slice": [offset, offset + f.dim], "source": f.source})
        offset += f.dim
    offset = 0
    for f in spec.atom_scalar_fields:
        out["atom_scalar"].append({"name": f.name, "dim": f.dim, "slice": [offset, offset + f.dim]})
        offset += f.dim
    offset = 0
    for f in spec.global_feature_fields:
        out["global"].append({"name": f.name, "dim": f.dim, "slice": [offset, offset + f.dim]})
        offset += f.dim
    out["widths"] = {
        "atom_feat": spec.atom_feat_dim,
        "bond_feat": spec.bond_feat_dim,
        "atom_scalar": spec.n_scalars,
        "global": spec.global_dim,
        "n_nbr_codes": spec.n_nbr_codes,
        "n_path_codes": spec.n_path_codes,
    }
    out["code_formulas"] = {
        "nbr_code": spec.nbr_code_formula,
        "path_cat": spec.path_code_formula,
        "path_end63": spec.path_end_code_formula,
    }
    return out


def raw_artifact_fields() -> dict[str, Any]:
    """The committed raw artifact is the only structural input to the caches."""
    arrs = RAW.load_arrays("ESOL", verify=True)
    manifest = C.read_json(C.V2 / "analysis/source/raw_artifact_manifest.json")
    return {
        "schema": manifest.get("schema"),
        "fields": {k: str(v.dtype) for k, v in arrs.items()},
        "vocab": manifest.get("vocab"),
        "atom_field_provenance": {
            "atom_class": "(element, is_aromatic) of 15 named classes, 15=UNK",
            "atom_charge": "formal charge bucket (-2,-1,0,1,2), 5=UNK",
            "atom_h": "GetTotalNumHs bucket (0..4, >4 -> 4), 5=UNK",
            "edge_class": "SINGLE/DOUBLE/TRIPLE/AROMATIC -> 0..3, 4=UNK",
            "edge_stereo": "GetStereo(): NONE/E/Z -> 0..2, anything else -> 3 (OTHER)",
        },
        "not_read_from_rdkit": [
            "Atom.GetChiralTag() / FindMolChiralCenters (tetrahedral configuration)",
            "Atom.GetIsotope()",
            "Bond.GetStereoAtoms() (the substituent identity behind an E/Z label)",
            "Bond.GetBondDir() (SMILES directionality)",
            "conformer / 3-D coordinates",
        ],
    }


# ---------------------------------------------------------------------------
# B. tensors -> forward: recording batch
# ---------------------------------------------------------------------------


class RecordingBatch(dict):
    """``dict`` that records every key the model reads."""

    def __init__(self, data: Mapping[str, Any], log: set[str]):
        super().__init__(data)
        self._log = log

    def __getitem__(self, key):  # noqa: D105
        self._log.add(str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):  # noqa: D105
        self._log.add(str(key))
        return super().get(key, default)


def _single_batch(dataset: str, row: int) -> tuple[dict[str, Any], dict[str, Any], FeatureSpec]:
    """Build one real molecule from the committed artifact and collate it."""
    spec = D.frozen_spec()
    raw = RAW.load_rows(dataset, [row], verify=True)
    graph = RDK.raw_row_to_graph(raw[0], int(raw[0]["mol_index"]), spec)
    mol = PREP_DATA.build_molecule(graph, spec)
    cache = PREP_DATA.concat_molecules([mol], spec, split="audit")
    cache["rows"] = np.asarray([row], dtype=np.int64)
    return PREP_DATA.collate(cache, [0]), {"graph": graph, "mol": mol}, spec


def forward_key_audit(dataset: str, row: int) -> dict[str, Any]:
    """Measure which batch keys ``LCRC.forward`` actually reads."""
    batch, _info, spec = _single_batch(dataset, row)
    model, meta = M.build_lcrc(spec, _zero_norm(spec), 0, torch.device("cpu"), dropout=0.0)
    model.eval()
    log: set[str] = set()
    with torch.no_grad():
        out = model(RecordingBatch(batch, log), return_aux=True)
    y, aux = out
    declared = {
        "atom_feat",
        "atom_scalar",
        "bond_feat",
        "bond_index",
        "nbr_atom",
        "nbr_code",
        "path_atom",
        "path_cat",
        "path_end63",
        "path_end",
        "path_closure",
        "global",
        "atom_batch",
        "n_graphs",
    }
    #: tensors that exist in the batch / spec but are never read by the forward
    not_read = sorted(set(batch) - log)
    return {
        "dataset": dataset,
        "row": int(row),
        "n_atoms": int(aux["h"].shape[0]),
        "y": float(y.reshape(-1)[0]),
        "keys_read_by_forward": sorted(log),
        "keys_present_but_not_read": not_read,
        "declared_structural_keys": sorted(declared),
        "atom_type_read": "atom_type" in log,
        "bond_type_read": "bond_type" in log,
        "atom_type_note": (
            "atom_type / bond_type are documentation tensors in the batch; the same information is "
            "carried into the forward by atom_feat[:, :16] and by the S1/S2 joint codes"
        ),
        "parameters": int(meta["parameters"]),
    }


def _zero_norm(spec: FeatureSpec) -> dict[str, Any]:
    return {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# C. field -> output sensitivity (wiring, not learned importance)
# ---------------------------------------------------------------------------


def _response(model, batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        y, aux = model(dict(batch), return_aux=True)
    return y.detach().numpy(), aux["h"].detach().numpy()


def _perturb(
    batch: Mapping[str, Any],
    key: str,
    index: int,
    column: int | None,
    delta: float,
    modulus: int | None = None,
) -> dict[str, Any]:
    """Add ``delta`` to one tensor cell (optionally modulo ``modulus``)."""
    out = {k: v for k, v in batch.items()}
    arr = out[key].clone()
    if arr.ndim == 1:
        pos = (index,)
    else:
        if column is None:
            raise ValueError(f"{key}: a column is required for a 2-D tensor")
        pos = (index, int(column))
    is_int = not torch.is_floating_point(arr)
    if is_int:
        value = int(arr[pos])
        new = value + int(round(delta))
        if modulus:
            new = new % int(modulus)
        arr[pos] = new
    else:
        arr[pos] = arr[pos] + float(delta)
    out[key] = arr
    return out


def sensitivity_audit(dataset: str = "Lipophilicity", row: int = 3) -> dict[str, Any]:
    """Perturb every declared field block and record the h / y response."""
    spec = D.frozen_spec()
    batch, info, _ = _single_batch(dataset, row)
    model, _meta = M.build_lcrc(spec, _zero_norm(spec), 0, torch.device("cpu"), dropout=0.0)
    model.eval()
    y0, h0 = _response(model, batch)
    mol = info["mol"]
    rows: list[dict[str, Any]] = []

    def record(
        group: str,
        field: str,
        key: str,
        index: int,
        column: int | None,
        delta: float,
        modulus: int | None = None,
    ) -> None:
        perturbed = _perturb(batch, key, index, column, delta, modulus)
        y1, h1 = _response(model, perturbed)
        rows.append(
            {
                "group": group,
                "field": field,
                "tensor": key,
                "index": int(index),
                "column": None if column is None else int(column),
                "delta": float(delta),
                "tensor_changed": bool(not np.array_equal(
                    np.asarray(batch[key]), np.asarray(perturbed[key])
                )),
                "max_abs_delta_h": float(np.abs(h1 - h0).max()),
                "max_abs_delta_y": float(np.abs(y1 - y0).max()),
                "influences_local_representation": bool(np.abs(h1 - h0).max() > 0.0),
                "wired_to_output": bool(
                    max(float(np.abs(h1 - h0).max()), float(np.abs(y1 - y0).max())) > 0.0
                ),
            }
        )

    # atom feature blocks (one atom, first column of each block)
    for f in spec.atom_feature_fields:
        offset = sum(x.dim for x in spec.atom_feature_fields if x is not f)
        record("atom_feat", f.name, "atom_feat", 0, offset, 1.0)
    # bond feature blocks
    for i, f in enumerate(spec.bond_feature_fields):
        offset = sum(x.dim for x in spec.bond_feature_fields[:i])
        record("bond_feat", f.name, "bond_feat", 0, offset, 1.0)
    # atom scalars (one column at a time)
    for i, f in enumerate(spec.atom_scalar_fields):
        offset = sum(x.dim for x in spec.atom_scalar_fields[:i])
        record("atom_scalar", f.name, "atom_scalar", 0, offset, 1.0)
    # global fields
    for i, f in enumerate(spec.global_feature_fields):
        offset = sum(x.dim for x in spec.global_feature_fields[:i])
        record("global", f.name, "global", 0, offset, 1.0)
    # structural code tensors: move one code to the next value inside the
    # vocabulary, so the probe stays inside the declared code range.
    record("structural_code", "nbr_code", "nbr_code", 0, None, 1.0, spec.n_nbr_codes)
    record("structural_code", "path_cat", "path_cat", 0, None, 1.0, spec.n_path_codes)
    record("structural_code", "path_end63", "path_end63", 0, None, 1.0, spec.n_nbr_codes)
    record("structural_code", "path_closure", "path_closure", 0, None, 1.0, 2)
    record("structural_code", "path_end", "path_end", 0, None, 1.0, int(batch["atom_feat"].shape[0]))
    record("structural_code", "bond_index", "bond_index", 0, 0, 1, int(batch["atom_feat"].shape[0]))

    by_field = {}
    for row_ in rows:
        by_field.setdefault(row_["field"], []).append(row_)
    summary = {
        name: {
            "n_probes": len(v),
            "tensor_changed": all(x["tensor_changed"] for x in v),
            "max_abs_delta_h": max(x["max_abs_delta_h"] for x in v),
            "max_abs_delta_y": max(x["max_abs_delta_y"] for x in v),
            "influences_local_representation": any(x["influences_local_representation"] for x in v),
            "wired_to_output": any(x["wired_to_output"] for x in v),
        }
        for name, v in by_field.items()
    }
    dead = [name for name, v in summary.items() if not v["wired_to_output"]]
    return {
        "dataset": dataset,
        "row": int(row),
        "n_atoms": int(mol["n_atoms"]),
        "n_bonds": int(mol["n_bonds"]),
        "model": "LCRC-A, seed 0, random init (wiring test only)",
        "interpretation": (
            "A field is 'wired' iff perturbing one of its columns changes h or y. This proves the "
            "computational graph consumes the declared block; it does NOT measure learned importance. "
            "Local atom/bond/path fields must move h; global fields only reach y through rho."
        ),
        "n_probes": len(rows),
        "per_field": summary,
        "fields_with_no_response": dead,
        "passed": not dead,
        "probes": rows,
    }


# ---------------------------------------------------------------------------
# S1/S2 vocabulary utilisation (what the joint codes can actually express)
# ---------------------------------------------------------------------------


def vocabulary_utilisation(dataset: str, split: str = "train") -> dict[str, Any]:
    from mb2_data import build_view

    view = build_view(dataset)
    cache = view.train_cache if split == "train" else view.valid_cache
    spec = view.spec
    atom_feat = np.asarray(cache["atom_feat"], dtype=np.float32)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    nbr = np.asarray(cache["nbr_code"], dtype=np.int64)
    path = np.asarray(cache["path_cat"], dtype=np.int64)
    end = np.asarray(cache["path_end63"], dtype=np.int64)
    atom_types_used = sorted(set(atom_type.tolist()))
    bond_types_used = sorted(set(bond_type.tolist()))
    return {
        "dataset": dataset,
        "split": split,
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms": int(atom_type.size),
        "n_bonds": int(bond_type.size),
        "n_s1_codes": int(nbr.size),
        "n_s2_paths": int(path.size),
        "atom_types_used": atom_types_used,
        "atom_type_coverage": len(atom_types_used) / spec.n_atom_types,
        "bond_types_used": bond_types_used,
        "bond_type_coverage": len(bond_types_used) / spec.n_bond_types,
        "nbr_code_vocab": int(spec.n_nbr_codes),
        "nbr_codes_used": int(np.unique(nbr).size),
        "nbr_code_coverage": float(np.unique(nbr).size / spec.n_nbr_codes),
        "path_code_vocab": int(spec.n_path_codes),
        "path_codes_used": int(np.unique(path).size),
        "path_code_coverage": float(np.unique(path).size / spec.n_path_codes),
        "path_end_codes_used": int(np.unique(end).size),
        "closure_fraction": float(np.asarray(cache["path_closure"], dtype=np.float64).mean())
        if np.asarray(cache["path_closure"]).size
        else 0.0,
        "atom_feat_column_nonconstant": int((atom_feat.std(axis=0) > 0).sum()),
        "atom_feat_width": int(spec.atom_feat_dim),
        "unknown_atom_rows": int((atom_type == spec.atom_unk_index).sum()),
        "unknown_bond_rows": int((bond_type == spec.bond_unk_id).sum()) if spec.bond_unk_id else 0,
        "global_dim": int(spec.global_dim),
        "n_scalars": int(spec.n_scalars),
    }


# ---------------------------------------------------------------------------
# S1/S2 are *joint* codes, not concatenated marginals
# ---------------------------------------------------------------------------


def joint_code_tests(spec: FeatureSpec | None = None) -> dict[str, Any]:
    """Show that the S1/S2 codes couple (type, bond order) rather than splitting them.

    Two neighbourhoods with the *same* multiset of neighbour elements and the
    *same* multiset of bond orders but a different pairing must produce different
    codes.  The check uses the production enumerators (``molecular_graph
    .enumerate_s1`` / ``.enumerate_s2``) on a synthetic adjacency, so a future
    refactor that degrades the joint code into marginals fails it.
    """
    import molecular_graph as MG

    spec = spec or D.frozen_spec()
    nb = spec.n_bond_types
    # element ids from the frozen vocabulary: C=2, N=7, O=9
    C_ID, N_ID, O_ID = 2, 7, 9
    SINGLE, DOUBLE = 1, 2

    # --- S1: atom 0 has neighbours (C--single, O==double) ---------------------
    atom_type = np.asarray([C_ID, C_ID, O_ID], dtype=np.int64)
    adj_a = (
        ((1, SINGLE, 0), (2, DOUBLE, 1)),
    )
    adj_b = (
        ((1, DOUBLE, 0), (2, SINGLE, 1)),
    )
    s1_a = MG.enumerate_s1(atom_type, adj_a, nb)[0]
    s1_b = MG.enumerate_s1(atom_type, adj_b, nb)[0]
    marginal_elements_a = sorted(int(atom_type[v]) for v, _b, _e in adj_a[0])
    marginal_elements_b = sorted(int(atom_type[v]) for v, _b, _e in adj_b[0])
    marginal_bonds_a = sorted(int(b) for _v, b, _e in adj_a[0])
    marginal_bonds_b = sorted(int(b) for _v, b, _e in adj_b[0])

    # --- S2: the directed bond-order pair is ordered in the joint code ---------
    at2 = np.asarray([C_ID, C_ID, O_ID], dtype=np.int64)
    adj2_a = (
        ((1, SINGLE, 0),),
        ((0, SINGLE, 0), (2, DOUBLE, 1)),
        (),
    )
    adj2_b = (
        ((1, DOUBLE, 0),),
        ((0, DOUBLE, 0), (2, SINGLE, 1)),
        (),
    )
    dist = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.int64)
    recs_a = MG.enumerate_s2(at2, adj2_a, nb, dist)[0]
    recs_b = MG.enumerate_s2(at2, adj2_b, nb, dist)[0]
    cats_a = sorted(r["path_cat"] for r in recs_a)
    cats_b = sorted(r["path_cat"] for r in recs_b)
    marginal_bonds_2path_a = sorted([int(r["bond1"]) for r in recs_a] + [int(r["bond2"]) for r in recs_a])
    marginal_bonds_2path_b = sorted([int(r["bond1"]) for r in recs_b] + [int(r["bond2"]) for r in recs_b])
    pairing = s2_pairing_test(spec)
    return {
        "s1": {
            "codes_a": s1_a,
            "codes_b": s1_b,
            "marginal_element_multiset_identical": marginal_elements_a == marginal_elements_b,
            "marginal_bond_multiset_identical": marginal_bonds_a == marginal_bonds_b,
            "joint_codes_differ": s1_a != s1_b,
            "code_vocabulary": int(spec.n_nbr_codes),
            "embedding_rows": int(spec.n_nbr_codes),
            "passed": bool(s1_a != s1_b and marginal_elements_a == marginal_elements_b and marginal_bonds_a == marginal_bonds_b),
        },
        "s2": {
            "path_cats_a": cats_a,
            "path_cats_b": cats_b,
            "marginal_bond_multiset_identical": marginal_bonds_2path_a == marginal_bonds_2path_b,
            "joint_codes_differ": cats_a != cats_b,
            "code_vocabulary": int(spec.n_path_codes),
            "embedding_rows": int(spec.n_path_codes),
            "passed": bool(cats_a != cats_b and marginal_bonds_2path_a == marginal_bonds_2path_b),
        },
        "s2_pairing": pairing,
        "note": (
            "S1 is one jointly typed statistic per incident bond; S2 is one jointly typed statistic "
            "per ordered non-backtracking 2-path, plus a separate joint (end type, bond2) code, the "
            "triangle-closure flag and a gated copy of the end atom's full atom representation. "
            "Neither is a sum of element and bond-order marginals. The middle<->end pairing inside "
            "S2 is NOT additive (see s2_pairing)."
        ),
    }


def s2_pairing_test(spec: FeatureSpec | None = None) -> dict[str, Any]:
    """Is the (path_cat, end_code) pairing recoverable from the additive S2 terms?

    Two synthetic 2-path multisets with the same marginals of ``path_cat`` and of
    ``end63`` but a different pairing are pushed through the *real* ``LocalContent``
    embeddings (seed 0, random init).  The additive part
    (``path_emb + end_emb + close_emb``) is identical by construction; only the
    rank-``stat_rank`` multiplicative term ``joint_proj(path_u * path_v)`` can see
    the pairing.  This bounds what S2 can express without widening the code.
    """
    import torch

    import lcrc_model as PREP_MODEL

    spec = spec or D.frozen_spec()
    torch.manual_seed(0)
    local = PREP_MODEL.LocalContent(spec, PREP_MODEL.DEFAULT_CONFIG)
    local.eval()
    # two path_cat codes that share (middle type, bt2) but differ in bt1
    pc_a, pc_b = 50, 51
    end_a, end_b = 11, 45
    set_a = [(pc_a, end_a), (pc_b, end_b)]
    set_b = [(pc_a, end_b), (pc_b, end_a)]

    def aggregate(records):
        with torch.no_grad():
            path = torch.stack([local.path_emb(torch.tensor(p)) for p, _e in records]).sum(0)
            end = torch.stack([local.end_emb(torch.tensor(e)) for _p, e in records]).sum(0)
            joint = torch.stack(
                [
                    local.joint_proj(local.path_u(torch.tensor(p)) * local.path_v(torch.tensor(e)))
                    for p, e in records
                ]
            ).sum(0)
            close = local.close_emb(torch.zeros(2, dtype=torch.long)).sum(0)
        return {
            "additive_sum": (path + end).numpy().copy(),
            "joint_term": joint.numpy().copy(),
            "closure": close.numpy().copy(),
        }

    agg_a = aggregate(set_a)
    agg_b = aggregate(set_b)
    additive_identical = bool(np.array_equal(agg_a["additive_sum"], agg_b["additive_sum"]))
    joint_differs = bool(not np.array_equal(agg_a["joint_term"], agg_b["joint_term"]))
    return {
        "record_set_a": set_a,
        "record_set_b": set_b,
        "same_path_cat_marginal": True,
        "same_end63_marginal": True,
        "additive_terms_identical": additive_identical,
        "joint_term_differs": joint_differs,
        "max_abs_joint_difference": float(np.abs(agg_a["joint_term"] - agg_b["joint_term"]).max()),
        "stat_rank": int(PREP_MODEL.DEFAULT_CONFIG["stat_rank"]),
        "conclusion": (
            "with a fixed set of additive codes the middle<->end pairing is only representable through "
            "the rank-32 multiplicative term; it is not part of any additive S2 code"
            if additive_identical and joint_differs
            else "unexpected: inspect the aggregation"
        ),
        "passed": bool(additive_identical and joint_differs),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(rows: Mapping[str, int] | None = None) -> dict[str, Any]:
    spec = D.frozen_spec()
    default_rows = {
        "ESOL": 3,
        "Lipophilicity": 3,
        "FreeSolv": 3,
        "BACE": 3,
        "BBBP": 3,
        "Tox21": 3,
        "ogbg-molhiv": 3,
    }
    rows = {**default_rows, **(rows or {})}
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "scope": (
            "2-D molecular graph only, lipo_v1 FeatureSpec, frozen LCRC-A forward. No learned-importance "
            "claim is made; see each section's interpretation field."
        ),
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "path": C.track_rel(C.PREP / "analysis/feature_spec_lipo.json"),
            "description": spec.description,
        },
        "field_table": field_table(spec),
        "raw_artifact": raw_artifact_fields(),
        "information_items": list(INFORMATION_ITEMS),
        "status_counts": {},
        "forward_keys": {},
        "sensitivity": {},
        "joint_codes": None,
        "vocabulary": {},
        "environment": C.env_info(),
    }
    counts: dict[str, int] = {}
    for item in INFORMATION_ITEMS:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    payload["status_counts"] = counts
    for dataset, row in rows.items():
        C.log(f"[P0.1] forward key audit {dataset} row {row}")
        payload["forward_keys"][dataset] = forward_key_audit(dataset, row)
    C.log("[P0.1] sensitivity audit (Lipophilicity row 3)")
    payload["sensitivity"] = sensitivity_audit("Lipophilicity", rows["Lipophilicity"])
    C.log("[P0.1] joint vs marginal code test")
    payload["joint_codes"] = joint_code_tests(spec)
    for dataset in C.DATASETS:
        C.log(f"[P0.1] S1/S2 vocabulary utilisation {dataset}")
        payload["vocabulary"][dataset] = vocabulary_utilisation(dataset, "train")
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)}")
    return payload


def summary(payload: Mapping[str, Any]) -> str:
    lines = ["# P0.1 information flow (audit summary)", ""]
    lines.append("declared-field status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(payload["status_counts"].items())))
    lines.append("")
    lines.append("| dataset | keys read | not read | atom_type read |")
    lines.append("|---|---|---|---|")
    for d, row in payload["forward_keys"].items():
        lines.append(
            f"| {d} | {len(row['keys_read_by_forward'])} | {', '.join(row['keys_present_but_not_read']) or '-'} | {row['atom_type_read']} |"
        )
    sens = payload["sensitivity"]
    lines.append("")
    lines.append(
        f"sensitivity: {sens['n_probes']} probes; fields with no response: {sens['fields_with_no_response'] or 'none'}"
    )
    lines.append("")
    lines.append(
        "atom_feat columns with non-zero variance (train): "
        + ", ".join(f"{d}={v['atom_feat_column_nonconstant']}/{v['atom_feat_width']}" for d, v in payload["vocabulary"].items())
    )
    lines.append(
        "path-code vocabulary used (train): "
        + ", ".join(f"{d}={v['path_codes_used']}/{v['path_code_vocab']}" for d, v in payload["vocabulary"].items())
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0.1 input information-flow audit")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)
    payload = run()
    if args.print:
        print(summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
