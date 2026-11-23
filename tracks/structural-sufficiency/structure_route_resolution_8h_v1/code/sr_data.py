"""Label-safe split / molecule layer for the route-resolution sprint.

D0 and D1 are the two frozen scaffold splits of the 3,360 official Train
molecules.  Their identity hashes are re-verified by the pinned
``r2_sentence_replication_v1`` loader; this module only re-exposes the pieces the
new models need and never touches a Valid/Test label (they stay NaN).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as C

SPLIT_PINS = {
    "D0": {
        "file": C.STRUCT / "acra_relation_generalization_pilot_v1" / "data"
        / "split_train_sub_dev.json",
        "identity": C.D0_IDENTITY,
    },
    "D1": {
        "file": C.STRUCT / "structural_path_composition_route_v1" / "data"
        / "split_train_sub_dev_d1.json",
        "identity": C.D1_IDENTITY,
    },
}


def _r2():
    import r2_data as RD  # noqa: PLC0415  (pinned parent loader)

    return RD


def load_bundle(split: str) -> Any:
    if split not in SPLIT_PINS:
        raise KeyError(f"unknown split {split!r}")
    bundle = _r2().load_bundle(split, "fit")
    spec = bundle.split
    if str(spec["identity_sha256"]) != SPLIT_PINS[split]["identity"]:
        raise ValueError(f"{split} identity mismatch: {spec['sha256']}")
    if int(bundle.fit_rows.size) != 2856 or int(bundle.eval_rows.size) != 504:
        raise ValueError(f"{split} sizes changed")
    if np.isfinite(bundle.labels[bundle.official_valid]).any():
        raise ValueError("official Valid label leaked")
    if not np.isfinite(bundle.labels[np.union1d(bundle.fit_rows, bundle.eval_rows)]).all():
        raise ValueError("non-finite label inside the chosen split")
    return bundle


def all_rows(bundle: Any) -> np.ndarray:
    return np.union1d(bundle.fit_rows, bundle.eval_rows).astype(np.int64)


def view(bundle: Any, row: int) -> dict[str, Any]:
    import gwic_data as GD  # noqa: PLC0415

    return GD.molecule_view(bundle.raw, int(row))


def mol_arrays(bundle: Any, row: int) -> dict[str, Any]:
    """Field layout expected by ``rice_structure.build_molecule``."""
    mol = view(bundle, row)
    return {
        "row": int(row),
        "n_atoms": int(mol["n_atoms"]),
        "n_bonds": int(mol["n_bonds"]),
        "atom_feat": np.asarray(mol["atom_feat"], np.int64),
        "bond_feat": np.asarray(mol["bond_feat"], np.int64).reshape(mol["n_bonds"], 3),
        "bond_u": np.asarray(mol["bond_u"], np.int64),
        "bond_v": np.asarray(mol["bond_v"], np.int64),
        "smiles": str(mol["smiles"]),
        "graph_scalars": bundle.graph_scalars[int(row)].copy(),
    }


def norm_of(bundle: Any) -> dict[str, np.ndarray]:
    return {"graph_mean": bundle.graph_mean, "graph_scale": bundle.graph_scale}


def split_identity(bundle: Any) -> dict[str, Any]:
    return {
        "split": bundle.split["name"],
        "identity_sha256": bundle.split["identity_sha256"],
        "file_sha256": bundle.split["file_sha256"],
        "fit_rows_sha256": bundle.split["fit_rows_sha256"],
        "eval_rows_sha256": bundle.split["eval_rows_sha256"],
        "n_fit": int(bundle.fit_rows.size),
        "n_eval": int(bundle.eval_rows.size),
        "scaler": dict(bundle.scaler),
        "raw_cache_sha256": bundle.hashes["raw_cache"]["sha256"],
        "labels_sha256": bundle.hashes["train_labels"]["sha256"],
    }


def feature_dims() -> tuple[list[int], list[int]]:
    import gwic_data as GD  # noqa: PLC0415

    return [int(x) for x in GD.atom_bond_dims()[0]], [int(x) for x in GD.atom_bond_dims()[1]]


def support(bundle: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import gwic_data as GD  # noqa: PLC0415

    atom_dims, bond_dims = feature_dims()
    return GD.category_support(bundle.raw, bundle.fit_rows, atom_dims, bond_dims)


def unseen(bundle: Any, rows: Sequence[int],
           atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray]) -> dict[str, int]:
    import gwic_data as GD  # noqa: PLC0415

    return GD.unseen_category_counts(bundle.raw, rows, atom_ok, bond_ok)


def graph_scalars_all(bundle: Any) -> np.ndarray:
    return np.asarray(bundle.graph_scalars, np.float64)