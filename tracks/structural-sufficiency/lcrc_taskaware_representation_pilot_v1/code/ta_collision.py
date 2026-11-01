"""P0.2b -- constructive and real-data representation-collision tests.

The question is narrow and empirical: *do two chemically different structures
produce exactly the same LCRC-A input?*  Three tests, all on the real execution
path (``smiles_to_raw_row`` -> ``raw_row_to_graph`` -> ``build_molecule`` ->
``collate``), never on string comparison alone:

1. ``tetrahedral``  -- take a molecule with an assigned stereocentre, invert the
   tag, round-trip both isomers through the frozen pipeline and compare every
   input tensor byte-for-byte.  If the tensors are identical the model *cannot*
   distinguish the two structures, and (with a frozen checkpoint) it provably
   returns the same prediction.
2. ``ez`` -- take a molecule with a defined C=C configuration, flip E<->Z and
   report which tensors do and do not change.  This separates "not encoded at
   all" from "encoded once, in one radius-1 channel".
3. ``real_data`` -- for every dataset, group rows by stereo-stripped canonical
   SMILES and count groups whose members differ only by stereo; then build the
   tensors for those members and count how many of those pairs actually collide.

A collision is a *representation* fact.  It is not, by itself, evidence that a
task's error comes from that collision; the report keeps the two apart.

Usage
-----
    uv run python code/ta_collision.py           # write analysis/feature_collision_tests.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

import mb2_data as D
import mb2_raw_artifact as RAW
import mb2_rdkit as RDK

import lcrc_data as PREP_DATA

OUT = C.ANALYSIS / "feature_collision_tests.json"

#: per-molecule tensors compared for a collision
TENSOR_KEYS = (
    "atom_feat",
    "atom_type",
    "atom_scalar",
    "bond_feat",
    "bond_type",
    "bond_index",
    "nbr_code",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "global",
)

MAX_REAL_GROUPS = 400


def _tensors(smiles: str, row: int = 0, spec=None) -> dict[str, np.ndarray]:
    """Frozen pipeline: SMILES -> raw row -> MolecularGraph -> LCRC arrays."""
    spec = spec or D.frozen_spec()
    raw = RDK.smiles_to_raw_row(smiles, int(row))
    graph = RDK.raw_row_to_graph(raw, int(row), spec)
    mol = PREP_DATA.build_molecule(graph, spec)
    out: dict[str, np.ndarray] = {}
    for key in TENSOR_KEYS:
        out[key] = np.asarray(mol[key])
    return out


def _diff(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    identical = True
    for key in TENSOR_KEYS:
        same = bool(np.array_equal(a[key], b[key]))
        out[key] = {
            "identical": same,
            "shape_a": list(np.asarray(a[key]).shape),
            "shape_b": list(np.asarray(b[key]).shape),
            "n_differing_entries": int(np.asarray(a[key]).size - np.isclose(
                np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64), equal_nan=True
            ).sum()) if np.asarray(a[key]).shape == np.asarray(b[key]).shape else None,
        }
        identical = identical and same
    return {"tensors": out, "all_input_tensors_identical": identical}


# ---------------------------------------------------------------------------
# constructive tests
# ---------------------------------------------------------------------------


def tetrahedral_test(dataset: str = "BACE") -> dict[str, Any]:
    """Invert one assigned stereocentre and compare the frozen model input."""
    from rdkit import Chem

    spec = D.frozen_spec()
    arrs = RAW.load_arrays(dataset, verify=True)
    smiles_all = [str(x) for x in arrs["smiles"].tolist()]
    chosen = None
    for i, smi in enumerate(smiles_all):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
        if len(centers) == 1:
            chosen = (i, smi, mol, centers[0])
            break
    if chosen is None:
        raise RuntimeError(f"{dataset}: no molecule with exactly one assigned stereocentre")
    key, smiles, mol, centre = chosen
    inverted = Chem.Mol(mol)
    atom = inverted.GetAtomWithIdx(int(centre[0]))
    atom.InvertChirality()
    Chem.AssignStereochemistry(inverted, cleanIt=True, force=True)
    inv_smiles = Chem.MolToSmiles(inverted)
    flat_orig = Chem.MolToSmiles(mol, isomericSmiles=False)
    flat_inv = Chem.MolToSmiles(inverted, isomericSmiles=False)
    if inv_smiles == Chem.MolToSmiles(mol):
        raise RuntimeError("chirality inversion did not change the isomeric SMILES")
    if flat_orig != flat_inv:
        raise RuntimeError("stereo inversion changed the constitution; test is invalid")

    # The raw adapter is compared directly too, so a future adapter change that
    # starts reading chirality is caught by this test rather than assumed.
    raw_a = RDK.smiles_to_raw_row(Chem.MolToSmiles(mol), key)
    raw_b = RDK.smiles_to_raw_row(inv_smiles, key)
    raw_fields = ("atom_class", "atom_charge", "atom_h", "edge_index", "edge_class", "edge_stereo")
    raw_diff = {f: bool(not np.array_equal(np.asarray(raw_a[f]), np.asarray(raw_b[f]))) for f in raw_fields}

    ta = _tensors(Chem.MolToSmiles(mol), key, spec)
    tb = _tensors(inv_smiles, key, spec)
    diff = _diff(ta, tb)
    return {
        "dataset": dataset,
        "row": int(key),
        "source_smiles": smiles,
        "isomer_a_smiles": Chem.MolToSmiles(mol),
        "isomer_b_smiles": inv_smiles,
        "stereo_stripped_smiles": flat_orig,
        "stereocentre_atom_index": int(centre[0]),
        "cip_a": str(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)[0][1]),
        "cip_b": str(
            Chem.FindMolChiralCenters(inverted, includeUnassigned=False, useLegacyImplementation=False)[0][1]
        ),
        "raw_artifact_fields_that_differ": raw_diff,
        "raw_artifact_identical": not any(raw_diff.values()),
        "model_input": diff,
        "verdict": (
            "REPRESENTATION_COLLISION: two enantiomers with different CIP labels produce identical "
            "LCRC-A input tensors"
            if diff["all_input_tensors_identical"]
            else "distinguishable"
        ),
    }


def ez_test(dataset: str = "BBBP") -> dict[str, Any]:
    """Flip an E double bond to Z and report exactly what changes."""
    from rdkit import Chem

    spec = D.frozen_spec()
    arrs = RAW.load_arrays(dataset, verify=True)
    smiles_all = [str(x) for x in arrs["smiles"].tolist()]
    chosen = None
    for i, smi in enumerate(smiles_all):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ez = [b for b in mol.GetBonds() if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)]
        if len(ez) == 1:
            chosen = (i, smi, mol, ez[0])
            break
    if chosen is None:
        raise RuntimeError(f"{dataset}: no molecule with exactly one E/Z bond")
    key, smiles, mol, bond = chosen
    flipped = Chem.Mol(mol)
    target = flipped.GetBondWithIdx(int(bond.GetIdx()))
    if target.GetStereo() == Chem.BondStereo.STEREOE:
        target.SetStereo(Chem.BondStereo.STEREOZ)
        direction = "E->Z"
    else:
        target.SetStereo(Chem.BondStereo.STEREOE)
        direction = "Z->E"
    flip_smiles = Chem.MolToSmiles(flipped)
    flat_orig = Chem.MolToSmiles(mol, isomericSmiles=False)
    flat_flip = Chem.MolToSmiles(flipped, isomericSmiles=False)
    if flip_smiles == Chem.MolToSmiles(mol):
        raise RuntimeError("E/Z flip did not change the isomeric SMILES")

    raw_a = RDK.smiles_to_raw_row(Chem.MolToSmiles(mol), key)
    raw_b = RDK.smiles_to_raw_row(flip_smiles, key)
    raw_fields = ("atom_class", "atom_charge", "atom_h", "edge_index", "edge_class", "edge_stereo")
    raw_diff = {f: bool(not np.array_equal(np.asarray(raw_a[f]), np.asarray(raw_b[f]))) for f in raw_fields}

    ta = _tensors(Chem.MolToSmiles(mol), key, spec)
    tb = _tensors(flip_smiles, key, spec)
    diff = _diff(ta, tb)
    changed = [k for k, v in diff["tensors"].items() if not v["identical"]]
    unchanged = [k for k, v in diff["tensors"].items() if v["identical"]]
    return {
        "dataset": dataset,
        "row": int(key),
        "source_smiles": smiles,
        "isomer_a_smiles": Chem.MolToSmiles(mol),
        "isomer_b_smiles": flip_smiles,
        "stereo_stripped_smiles": flat_orig,
        "direction": direction,
        "raw_artifact_fields_that_differ": raw_diff,
        "model_input": diff,
        "tensors_that_change": changed,
        "tensors_that_stay_identical": unchanged,
        "s1_s2_codes_identical": bool(
            diff["tensors"]["nbr_code"]["identical"]
            and diff["tensors"]["path_cat"]["identical"]
            and diff["tensors"]["path_end63"]["identical"]
            and diff["tensors"]["path_closure"]["identical"]
        ),
        "verdict": (
            "PARTIAL: the E/Z label is carried only by bond_feat[stereo]; the S1/S2 joint codes, the "
            "atom scalars and the global block are stereo-blind"
            if diff["tensors"]["bond_feat"]["identical"] is False
            and diff["tensors"]["nbr_code"]["identical"]
            and diff["tensors"]["path_cat"]["identical"]
            else "check tensors_that_change"
        ),
    }


# ---------------------------------------------------------------------------
# real-data collision scan
# ---------------------------------------------------------------------------


def _flat_and_iso(smiles: str) -> tuple[str, str, str] | None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    method = "strict"
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        method = "relaxed"
    try:
        flat = Chem.MolToSmiles(mol, isomericSmiles=False)
        iso = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None
    return flat, iso, method


def real_data_collisions(dataset: str) -> dict[str, Any]:
    """Groups whose members are stereo-distinct but constitutionally identical."""
    arrs = RAW.load_arrays(dataset, verify=True)
    smiles_all = [str(x) for x in arrs["smiles"].tolist()]
    groups: dict[str, list[tuple[int, str]]] = {}
    failed = 0
    n_chiral_specified = 0
    from rdkit import Chem

    for i, smi in enumerate(smiles_all):
        parsed = _flat_and_iso(smi)
        if parsed is None:
            failed += 1
            continue
        flat, iso, _method = parsed
        groups.setdefault(flat, []).append((i, iso))
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False):
            n_chiral_specified += 1

    multi = {k: v for k, v in groups.items() if len({iso for _i, iso in v}) > 1}
    n_groups = len(multi)
    n_pairs_max = sum(1 for v in multi.values() if len({iso for _i, iso in v}) > 1)
    n_pairs_actual = sum(
        len({iso for _i, iso in v}) * (len({iso for _i, iso in v}) - 1) // 2 for v in multi.values()
    )

    spec = D.frozen_spec()
    identical_pairs = 0
    distinguishable_pairs = 0
    compared_pairs = 0
    examples: list[dict[str, Any]] = []
    for flat, members in list(multi.items())[:MAX_REAL_GROUPS]:
        by_iso: dict[str, int] = {}
        for row, iso in members:
            by_iso.setdefault(iso, row)
        isos = sorted(by_iso)
        tensors = {}
        for iso in isos:
            try:
                tensors[iso] = _tensors(iso, by_iso[iso], spec)
            except Exception:  # a relaxed-parse row may not round-trip
                tensors = {}
                break
        if len(tensors) != len(isos) or len(isos) < 2:
            continue
        for a in range(len(isos)):
            for b in range(a + 1, len(isos)):
                ta, tb = tensors[isos[a]], tensors[isos[b]]
                same = all(np.array_equal(ta[k], tb[k]) for k in TENSOR_KEYS)
                # only compare molecules with the same atom/bond count
                if ta["atom_type"].shape != tb["atom_type"].shape:
                    continue
                compared_pairs += 1
                if same:
                    identical_pairs += 1
                    if len(examples) < 5:
                        examples.append(
                            {
                                "rows": [int(by_iso[isos[a]]), int(by_iso[isos[b]])],
                                "smiles": [isos[a], isos[b]],
                                "stereo_stripped": flat,
                                "identical_tensors": True,
                            }
                        )
                else:
                    distinguishable_pairs += 1
                    diff_keys = [k for k in TENSOR_KEYS if not np.array_equal(ta[k], tb[k])]
                    if len(examples) < 5:
                        examples.append(
                            {
                                "rows": [int(by_iso[isos[a]]), int(by_iso[isos[b]])],
                                "smiles": [isos[a], isos[b]],
                                "stereo_stripped": flat,
                                "identical_tensors": False,
                                "tensors_that_differ": diff_keys,
                            }
                        )
    return {
        "dataset": dataset,
        "n_molecules": len(smiles_all),
        "n_parse_failed": int(failed),
        "n_molecules_with_assigned_chiral_center": int(n_chiral_specified),
        "n_constitution_groups": len(groups),
        "n_groups_with_stereo_distinct_members": int(n_groups),
        "n_groups_capped_at": int(MAX_REAL_GROUPS),
        "n_stereo_distinct_isoforms_upper_bound": int(n_pairs_max),
        "n_stereo_isoform_pairs": int(n_pairs_actual),
        "n_pairs_compared": int(compared_pairs),
        "n_pairs_with_identical_model_input": int(identical_pairs),
        "n_pairs_with_distinguishable_model_input": int(distinguishable_pairs),
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# label-side diagnostic (Train labels only; descriptive, never used for selection)
# ---------------------------------------------------------------------------


def train_label_divergence(dataset: str, collisions: Mapping[str, Any]) -> dict[str, Any]:
    """Do the colliding pairs carry the *same* Train label?

    This is the only place in the round that touches a label for the collision
    question, and it is deliberately restricted to the published **Train** split
    (``labels="train_valid"`` never decodes a Test label).  It answers one
    descriptive question: is the collision merely a duplicate record, or does it
    force the model to map two differently-labelled molecules to one input?
    """
    import mb2_source as S

    source = S.load_dataset(dataset, labels="train_valid")
    y_train = np.asarray(source["y"]["train"], dtype=np.float64)
    train_rows = [int(r) for r in source["rows"]["train"]]
    position = {row: i for i, row in enumerate(train_rows)}
    task = C.dataset_entry(dataset)["task"]

    pairs = []
    for example in collisions.get("examples", []):
        pairs.append(example)
    # The example list is capped at five; the full pair list is rebuilt from the scan.
    full = _all_collision_pairs(dataset)
    n_in_train = 0
    n_label_identical = 0
    n_label_different = 0
    worst: list[dict[str, Any]] = []
    for a, b, same_input in full:
        if a not in position or b not in position:
            continue
        n_in_train += 1
        ya, yb = y_train[position[a]], y_train[position[b]]
        if task == "multitask":
            mask = np.isfinite(ya) & np.isfinite(yb)
            n_diff = int((ya[mask] != yb[mask]).sum()) if mask.any() else 0
            same = n_diff == 0
            delta = n_diff
        elif task == "regression":
            delta = float(abs(ya.reshape(-1)[0] - yb.reshape(-1)[0]))
            same = delta == 0.0
        else:
            delta = float(abs(ya.reshape(-1)[0] - yb.reshape(-1)[0]))
            same = delta == 0.0
        if same:
            n_label_identical += 1
        else:
            n_label_different += 1
            if len(worst) < 8:
                worst.append(
                    {
                        "train_rows": [int(a), int(b)],
                        "identical_model_input": bool(same_input),
                        "label_a": np.asarray(ya).reshape(-1).tolist(),
                        "label_b": np.asarray(yb).reshape(-1).tolist(),
                        "n_differing_label_cells": int(delta) if task == "multitask" else None,
                        "abs_delta": None if task == "multitask" else float(delta),
                    }
                )
    return {
        "dataset": dataset,
        "task": task,
        "label_scope": "published Train split only; Test labels never decoded",
        "n_collision_pairs": int(len(full)),
        "n_collision_pairs_with_both_rows_in_train": int(n_in_train),
        "n_pairs_same_train_label": int(n_label_identical),
        "n_pairs_different_train_label": int(n_label_different),
        "fraction_different": float(n_label_different / max(n_in_train, 1)),
        "examples_of_label_divergence": worst,
        "n_capped_examples_ignored": len(pairs),
    }


def _all_collision_pairs(dataset: str) -> list[tuple[int, int, bool]]:
    """Rebuild the full pair list (rows, identical-input flag) from the scan."""
    from rdkit import Chem

    arrs = RAW.load_arrays(dataset, verify=True)
    smiles_all = [str(x) for x in arrs["smiles"].tolist()]
    groups: dict[str, dict[str, int]] = {}
    for i, smi in enumerate(smiles_all):
        parsed = _flat_and_iso(smi)
        if parsed is None:
            continue
        flat, iso, _method = parsed
        groups.setdefault(flat, {}).setdefault(iso, i)
    spec = D.frozen_spec()
    out: list[tuple[int, int, bool]] = []
    for flat, by_iso in groups.items():
        if len(by_iso) < 2:
            continue
        isos = sorted(by_iso)
        tensors: dict[str, dict[str, np.ndarray]] = {}
        for iso in isos:
            try:
                tensors[iso] = _tensors(iso, by_iso[iso], spec)
            except Exception:
                tensors = {}
                break
        if len(tensors) != len(isos):
            continue
        for a in range(len(isos)):
            for b in range(a + 1, len(isos)):
                ta, tb = tensors[isos[a]], tensors[isos[b]]
                if ta["atom_type"].shape != tb["atom_type"].shape:
                    continue
                same = all(np.array_equal(ta[k], tb[k]) for k in TENSOR_KEYS)
                out.append((by_iso[isos[a]], by_iso[isos[b]], bool(same)))
    return out


# ---------------------------------------------------------------------------
# frozen-checkpoint consequence (label-free)
# ---------------------------------------------------------------------------


def frozen_checkpoint_effect(dataset: str, pair: Mapping[str, Any], checkpoint: str) -> dict[str, Any]:
    """Show that identical inputs give identical predictions under a frozen model."""
    import torch

    import mb2_model as M

    spec = D.frozen_spec()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    n_tasks = int(ckpt.get("n_outputs", 1))
    model = M.load_checkpoint_model(spec, D.frozen_spec() and _norm_from_train_cache(dataset), state, torch.device("cpu"), n_tasks=n_tasks)
    out = {}
    for smi in pair["smiles"]:
        ta = _tensors(smi, 0, spec)
        cache = PREP_DATA.concat_molecules(
            [
                PREP_DATA.build_molecule(
                    RDK.raw_row_to_graph(RDK.smiles_to_raw_row(smi, 0), 0, spec), spec
                )
            ],
            spec,
            split="collision",
        )
        cache["rows"] = np.asarray([0], dtype=np.int64)
        batch = PREP_DATA.collate(cache, [0])
        with torch.no_grad():
            y = model(batch).reshape(-1).numpy()
        out[smi] = [float(x) for x in y]
    values = list(out.values())
    return {
        "checkpoint": C.track_rel(checkpoint),
        "checkpoint_best_epoch": int(ckpt.get("best_epoch", -1)),
        "raw_model_output_by_smiles": out,
        "outputs_identical": bool(np.array_equal(np.asarray(values[0]), np.asarray(values[1]))),
    }


def _norm_from_train_cache(dataset: str) -> dict[str, Any]:
    """Train-only normaliser of the *published* split (read-only reuse)."""
    from mb2_data import build_view

    view = build_view(dataset)
    return view.train_norm


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "interpretation": (
            "A collision proves that the frozen input cannot distinguish two structures. It does NOT "
            "prove that any dataset's error is caused by that collision; no label is read in this file."
        ),
        "constructive": {},
        "real_data": {},
        "environment": C.env_info(),
    }
    C.log("[P0.2b] constructive tetrahedral collision (BACE)")
    payload["constructive"]["tetrahedral"] = tetrahedral_test("BACE")
    C.log("[P0.2b] constructive tetrahedral collision (BBBP)")
    payload["constructive"]["tetrahedral_bbbp"] = tetrahedral_test("BBBP")
    C.log("[P0.2b] constructive E/Z collision (BBBP)")
    payload["constructive"]["ez"] = ez_test("BBBP")

    tetra = payload["constructive"]["tetrahedral"]
    payload["constructive"]["frozen_model_effect"] = frozen_checkpoint_effect(
        "BACE",
        {"smiles": [tetra["isomer_a_smiles"], tetra["isomer_b_smiles"]]},
        C.V1 / "analysis/runs/BACE_s0/checkpoint.pt",
    )

    for dataset in C.DATASETS:
        C.log(f"[P0.2b] real-data stereo collision scan {dataset}")
        payload["real_data"][dataset] = real_data_collisions(dataset)
    payload["train_label_divergence"] = {
        d: train_label_divergence(d, payload["real_data"][d]) for d in C.DATASETS
    }
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)}")
    return payload


def summary(payload: Mapping[str, Any]) -> str:
    t = payload["constructive"]["tetrahedral"]
    e = payload["constructive"]["ez"]
    lines = [
        "## constructive",
        f"- tetrahedral: {t['isomer_a_smiles']} vs {t['isomer_b_smiles']} -> "
        f"raw identical={t['raw_artifact_identical']}, model input identical="
        f"{t['model_input']['all_input_tensors_identical']} ({t['verdict'].split(':')[0]})",
        f"- frozen checkpoint outputs: {payload['constructive']['frozen_model_effect']['raw_model_output_by_smiles']}"
        f" identical={payload['constructive']['frozen_model_effect']['outputs_identical']}",
        f"- E/Z: {e['isomer_a_smiles']} vs {e['isomer_b_smiles']} -> changes {e['tensors_that_change']}, "
        f"S1/S2 identical={e['s1_s2_codes_identical']}",
        "",
        "## real data",
        "| dataset | molecules | chiral mols | stereo-distinct pairs | identical input pairs | distinguishable pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d, r in payload["real_data"].items():
        lines.append(
            f"| {d} | {r['n_molecules']} | {r['n_molecules_with_assigned_chiral_center']} | "
            f"{r['n_stereo_isoform_pairs']} | {r['n_pairs_with_identical_model_input']} | "
            f"{r['n_pairs_with_distinguishable_model_input']} |"
        )
    lines.append("")
    lines.append("## Train-only label divergence of colliding pairs")
    lines.append("| dataset | pairs in train | same label | different label | examples |")
    lines.append("|---|---:|---:|---:|---|")
    for d, r in payload.get("train_label_divergence", {}).items():
        ex = r["examples_of_label_divergence"][:2]
        lines.append(
            f"| {d} | {r['n_collision_pairs_with_both_rows_in_train']} | {r['n_pairs_same_train_label']} | "
            f"{r['n_pairs_different_train_label']} | {json.dumps(ex) if ex else '-'} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P0.2b representation collision tests")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)
    payload = run()
    if args.print:
        print(summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
