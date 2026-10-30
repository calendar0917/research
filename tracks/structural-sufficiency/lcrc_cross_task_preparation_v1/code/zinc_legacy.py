"""Read-only ZINC legacy compatibility replay.

The round never trains or re-selects a ZINC model.  It checks that the new
schema-driven interface reproduces the historical label-free LCRC-A inputs and
that the frozen LCRC-A checkpoint can be loaded and executed on CPU with the
same forward decomposition.

Old modules are imported only here.  They are read-only; no historical file is
written.  The comparison uses the frozen P3 ``MolObjects`` cache (label-free)
and the historical ``lcrcA_s0/best.pt`` checkpoint.  ZINC labels are never
loaded, and no MAE/RMSE is computed.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

import adapters as A
import lcrc_common as C
import lcrc_data as LD
import lcrc_model as LM
import molecular_graph as MG
from feature_spec import FeatureSpec

#: Fields that LCRC-A actually consumes, plus the pointer/code arrays that
#: define the structural interface.
LCRC_A_FIELDS = (
    "atom_feat",
    "atom_type",
    "atom_scalar",
    "bond_feat",
    "bond_type",
    "bond_index",
    "nbr_ptr",
    "nbr_code",
    "path_ptr",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "global",
)

INTEGER_FIELDS = {
    "atom_type",
    "bond_type",
    "bond_index",
    "nbr_ptr",
    "nbr_code",
    "path_ptr",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
}

PAIR_FIELDS = ("pair_i", "pair_j", "pair_cat")


def _old_modules():
    old_code = str(C.LEGACY_CODE)
    p3_code = str(C.P3_CODE)
    for p in (p3_code, old_code):
        if p not in sys.path:
            sys.path.insert(0, p)
    import lc_data as old_data  # type: ignore
    import lc_models as old_models  # type: ignore

    return old_data, old_models


def _compare_arrays(old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    old = np.asarray(old)
    new = np.asarray(new)
    if old.shape != new.shape:
        return {
            "shape_ok": False,
            "old_shape": list(old.shape),
            "new_shape": list(new.shape),
            "exact": False,
            "max_abs_diff": None,
        }
    if old.dtype.kind in "iu" and new.dtype.kind in "iu":
        diff = np.abs(old.astype(np.int64) - new.astype(np.int64))
        max_abs = int(diff.max()) if diff.size else 0
        return {
            "shape_ok": True,
            "exact": bool(max_abs == 0),
            "max_abs_diff": max_abs,
            "n_different": int(np.count_nonzero(diff)),
        }
    oldf = old.astype(np.float64)
    newf = new.astype(np.float64)
    diff = np.abs(oldf - newf)
    max_abs = float(diff.max()) if diff.size else 0.0
    return {
        "shape_ok": True,
        "exact": bool(max_abs == 0.0),
        "max_abs_diff": max_abs,
        "n_different": int(np.count_nonzero(diff)),
    }


def _new_pair_arrays(graph: MG.MolecularGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    analysis = MG.analyse_graph(graph)
    return MG.relation_pairs(analysis.n_atoms, analysis.dist, analysis.ring_block_membership)


def compare_molecule(old: Mapping[str, Any], new: Mapping[str, Any], graph: MG.MolecularGraph) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in LCRC_A_FIELDS:
        out[field] = _compare_arrays(old[field], new[field])
    # Relation-pair arrays are R/N-only but are part of the historical
    # per-molecule builder output; compare them as an extra non-LCRC-A check.
    pi, pj, pc = _new_pair_arrays(graph)
    out["pair_i"] = _compare_arrays(old["pair_i"], pi)
    out["pair_j"] = _compare_arrays(old["pair_j"], pj)
    out["pair_cat"] = _compare_arrays(old["pair_cat"], pc)
    return out


def compare_p3_objects(
    spec: FeatureSpec,
    split: str = "train",
    max_molecules: int | None = None,
    *,
    progress_every: int = 2000,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    objs, obj_info = A.load_legacy_p3_objects(split)
    old_data, _old_models = _old_modules()
    if max_molecules is not None:
        objs = objs[: int(max_molecules)]
        obj_info = dict(obj_info)
        obj_info["max_molecules"] = int(max_molecules)
    field_summary: dict[str, dict[str, Any]] = {
        f: {"n_molecules": 0, "n_exact": 0, "n_shape_mismatch": 0, "max_abs_diff": 0, "n_different": 0}
        for f in (*LCRC_A_FIELDS, *PAIR_FIELDS)
    }
    first_mismatch: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = []
    for i, obj in enumerate(objs):
        graph = A.legacy_object_to_graph(obj, spec)
        old = old_data.build_molecule_from_object(obj)
        new = LD.build_molecule(graph, spec)
        cmp = compare_molecule(old, new, graph)
        for field, result in cmp.items():
            s = field_summary[field]
            s["n_molecules"] += 1
            if result.get("shape_ok") is False:
                s["n_shape_mismatch"] += 1
                if first_mismatch is None:
                    first_mismatch = {"molecule": i, "field": field, **result}
                failures.append({"molecule": i, "field": field, **result})
                continue
            if result["exact"]:
                s["n_exact"] += 1
            else:
                if first_mismatch is None:
                    first_mismatch = {"molecule": i, "field": field, **result}
                if len(failures) < 20:
                    failures.append({"molecule": i, "field": field, **result})
            if result["max_abs_diff"] is not None:
                s["max_abs_diff"] = max(s["max_abs_diff"], result["max_abs_diff"])
            s["n_different"] += int(result.get("n_different", 0))
        if progress_every and (i + 1) % progress_every == 0:
            C.log(f"[zinc compat] {i + 1}/{len(objs)} molecules {time.perf_counter() - t0:.1f}s")
    all_exact = all(
        s["n_shape_mismatch"] == 0 and s["n_exact"] == s["n_molecules"]
        for s in field_summary.values()
    )
    return {
        "status": "PASS" if all_exact else "FAIL",
        "split": split,
        "n_molecules": len(objs),
        "objects": obj_info,
        "fields": field_summary,
        "all_fields_exact": bool(all_exact),
        "first_mismatch": first_mismatch,
        "failures_sample": failures,
        "seconds": float(time.perf_counter() - t0),
        "labels_read": False,
    }


def compare_collate(spec: FeatureSpec, n_molecules: int = 64) -> dict[str, Any]:
    objs, obj_info = A.load_legacy_p3_objects("train")
    objs = objs[: int(n_molecules)]
    old_data, _old_models = _old_modules()
    old_mols = [old_data.build_molecule_from_object(obj) for obj in objs]
    new_graphs = [A.legacy_object_to_graph(obj, spec) for obj in objs]
    old_cache = old_data._concat_molecules(old_mols)
    new_cache = LD.concat_molecules(
        [LD.build_molecule(g, spec) for g in new_graphs],
        spec,
        split="legacy_collate_probe",
    )
    rows = list(range(len(objs)))
    old_batch = old_data.collate(old_cache, rows)
    new_batch = LD.collate(new_cache, rows)
    common = [
        "atom_feat",
        "atom_type",
        "atom_scalar",
        "atom_batch",
        "atom_off",
        "bond_feat",
        "bond_type",
        "bond_index",
        "bond_batch",
        "nbr_atom",
        "nbr_code",
        "path_atom",
        "path_cat",
        "path_end63",
        "path_end",
        "path_closure",
        "global",
        "atom_counts",
        "bond_counts",
    ]
    fields: dict[str, Any] = {}
    for key in common:
        old_arr = old_batch[key].cpu().numpy() if torch.is_tensor(old_batch[key]) else np.asarray(old_batch[key])
        new_arr = new_batch[key].cpu().numpy() if torch.is_tensor(new_batch[key]) else np.asarray(new_batch[key])
        fields[key] = _compare_arrays(old_arr, new_arr)
    all_exact = all(v["exact"] for v in fields.values())
    return {
        "status": "PASS" if all_exact else "FAIL",
        "n_molecules": int(n_molecules),
        "objects": obj_info,
        "fields": fields,
        "all_fields_exact": bool(all_exact),
        "note": "R/N-only pair_* keys are not part of the new LCRC-A collate; they are compared at the per-molecule level above.",
        "labels_read": False,
    }


def replay_frozen_checkpoint(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    n_molecules: int = 64,
) -> dict[str, Any]:
    if not C.LEGACY_CKPT.exists():
        return {
            "status": "BLOCKED",
            "missing": C.track_rel(C.LEGACY_CKPT),
            "labels_read": False,
        }
    objs, obj_info = A.load_legacy_p3_objects("train")
    objs = objs[: int(n_molecules)]
    old_data, old_models = _old_modules()
    ck = A.load_legacy_checkpoint(C.LEGACY_CKPT)
    state = ck["state_dict"]
    old_model = old_models.LCRC("A", norm)
    old_model.load_state_dict(state, strict=True)
    old_model.eval()
    new_model = LM.load_model_from_state(spec, norm, state, device="cpu")
    new_model.eval()

    graphs = [A.legacy_object_to_graph(obj, spec) for obj in objs]
    cache = LD.concat_molecules([LD.build_molecule(g, spec) for g in graphs], spec, split="legacy_replay")
    batch = LD.collate(cache, list(range(len(objs))))
    with torch.no_grad():
        old_y, old_aux = old_model(batch, return_aux=True)
        new_y, new_aux = new_model(batch, return_aux=True)
    fields = {
        "y": _compare_arrays(old_y.detach().cpu().numpy(), new_y.detach().cpu().numpy()),
        "contrib": _compare_arrays(
            old_aux["contrib"].detach().cpu().numpy(), new_aux["contrib"].detach().cpu().numpy()
        ),
        "rho": _compare_arrays(old_aux["rho"].detach().cpu().numpy(), new_aux["rho"].detach().cpu().numpy()),
        "h": _compare_arrays(old_aux["h"].detach().cpu().numpy(), new_aux["h"].detach().cpu().numpy()),
    }
    decomposition_error = float(
        torch.max(torch.abs(new_y - (new_aux["contrib"] + new_aux["rho"]))).item()
    )
    old_state_keys = set(old_model.state_dict().keys())
    new_state_keys = set(new_model.state_dict().keys())
    all_match = all(v["exact"] for v in fields.values())
    return {
        "status": "PASS" if all_match else "FAIL",
        "checkpoint": {**C.file_info(C.LEGACY_CKPT)},
        "checkpoint_meta": {
            "arm": ck.get("arm"),
            "seed": ck.get("seed"),
            "epoch": ck.get("epoch"),
            "metrics_omitted": True,
            "note": "historical metric fields were omitted; this round did not read ZINC labels or compute a metric",
        },
        "objects": obj_info,
        "n_molecules": int(n_molecules),
        "state_keys_equal": old_state_keys == new_state_keys,
        "n_state_keys": len(state.keys()),
        "parameters_old": LM.parameter_count(old_model),
        "parameters_new": LM.parameter_count(new_model),
        "state_hash_old": LM.state_hash(old_model),
        "state_hash_new": LM.state_hash(new_model),
        "fields": fields,
        "decomposition_max_abs_error": decomposition_error,
        "all_fields_exact": bool(all_match),
        "device": "cpu",
        "training": False,
        "optimizer_steps": 0,
        "labels_read": False,
    }


def run_compatibility(
    spec: FeatureSpec,
    *,
    max_molecules: int | None = 10000,
    collate_n: int = 64,
    replay_n: int = 64,
) -> dict[str, Any]:
    norm = A.load_legacy_norm()
    result: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "schema": spec.name,
        "schema_hash": spec.hash(),
        "scope": {
            "device": "cpu",
            "training": False,
            "optimizer_steps": 0,
            "zinc_labels_read": False,
            "official_valid_test_loaded": False,
            "checkpoint_weights_modified": False,
        },
        "norm": {
            "path": C.track_rel(C.LEGACY_NORM),
            "sha256": C.sha256_file(C.LEGACY_NORM),
            "source": norm["source"],
            "fingerprint": LD.norm_fingerprint(norm),
        },
    }
    try:
        result["input_replay"] = compare_p3_objects(spec, "train", max_molecules=max_molecules)
    except Exception as exc:  # pragma: no cover - recorded as a blocked/failed artifact
        result["input_replay"] = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    try:
        result["collate_replay"] = compare_collate(spec, n_molecules=collate_n)
    except Exception as exc:
        result["collate_replay"] = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    try:
        result["checkpoint_replay"] = replay_frozen_checkpoint(spec, norm, n_molecules=replay_n)
    except Exception as exc:
        result["checkpoint_replay"] = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    statuses = [
        result["input_replay"].get("status"),
        result["collate_replay"].get("status"),
        result["checkpoint_replay"].get("status"),
    ]
    if any(s == "FAIL" for s in statuses):
        result["status"] = "FAIL"
    elif any(s == "BLOCKED" for s in statuses):
        result["status"] = "BLOCKED"
    else:
        result["status"] = "PASS"
    result["environment"] = C.env_info()
    result["peak_rss_mb"] = C.peak_rss_mb()
    return result


__all__ = [
    "LCRC_A_FIELDS",
    "PAIR_FIELDS",
    "compare_p3_objects",
    "compare_collate",
    "replay_frozen_checkpoint",
    "run_compatibility",
]
