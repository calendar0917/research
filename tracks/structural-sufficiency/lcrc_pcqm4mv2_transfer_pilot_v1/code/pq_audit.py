"""Input-parsing / OOV audits over a merged feature cache.

Everything here is label-blind and reads the cache (plus the stored SMILES for
the handful of molecules that contain an out-of-vocabulary atom).  It answers
the questions the protocol requires before a formal run:

* which atom classes / bond classes actually occur, and how much of the data is
  mapped to ``UNK``,
* *which elements* the ``UNK`` bucket receives (so a mapping that is unsuitable
  for this dataset cannot hide),
* the parse-mode ledger (strict vs documented fallback) recorded by the build,
* the largest S2 record count (the per-molecule path budget),
* whether any structural feature is non-finite, out of range or degenerate,
* disconnected molecules (the ``lipo_v1`` schema supports them explicitly).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402

import feature_spec as FS  # noqa: E402

UNK_ATOM_INDEX = 15
UNK_BOND_ID = 5


def oov_audit(cache: Mapping[str, Any], spec: FS.FeatureSpec, *, max_unk_molecules: int = 400) -> dict[str, Any]:
    """Vocabulary coverage of the *stored* cache, plus the UNK element breakdown."""
    from rdkit import Chem

    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    components = np.asarray(cache["n_components"], dtype=np.int64)

    atom_counts = np.bincount(atom_type, minlength=spec.n_atom_types)
    bond_counts = np.bincount(bond_type, minlength=spec.n_bond_types + 1)
    n_atoms = int(atom_type.size)
    n_bonds = int(bond_type.size)
    s2_per_atom = np.diff(path_ptr)
    s2_per_molecule = np.concatenate([[0], np.cumsum(s2_per_atom)])[atom_ptr]
    s2_per_molecule = np.diff(s2_per_molecule)
    n_atoms_per_molecule = np.diff(atom_ptr)

    unk_atom_idx = np.flatnonzero(atom_type == UNK_ATOM_INDEX)
    unk_molecules = np.unique(np.searchsorted(atom_ptr, unk_atom_idx, side="right") - 1) if unk_atom_idx.size else np.zeros(0, np.int64)
    element_counts: dict[str, int] = {}
    inspected = 0
    parse_failures: list[dict[str, Any]] = []
    for row in unk_molecules[: int(max_unk_molecules)]:
        row = int(row)
        smi = cache["smiles"][row]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            parse_failures.append({"row": row, "smiles": smi})
            continue
        a0 = int(atom_ptr[row])
        local = atom_type[a0 : int(atom_ptr[row + 1])]
        for idx in np.flatnonzero(local == UNK_ATOM_INDEX):
            symbol = mol.GetAtomWithIdx(int(idx)).GetSymbol()
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        inspected += 1

    nonfinite = {}
    for name in ("atom_feat", "atom_scalar", "bond_feat", "global"):
        arr = np.asarray(cache[name])
        nonfinite[name] = int((~np.isfinite(arr)).sum()) if arr.size else 0
    const_scalar = np.flatnonzero(np.asarray(cache["atom_scalar"]).std(0) == 0).tolist() if n_atoms else []
    const_global = np.flatnonzero(np.asarray(cache["global"]).std(0) == 0).tolist()

    payload = {
        "schema": "lcrc-pcqm4mv2-oov-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "feature_spec": {"name": spec.name, "hash": spec.hash()},
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "atom_class_counts": {
            spec.atom_type_names[i] if i < spec.n_atom_types else str(i): int(c)
            for i, c in enumerate(atom_counts.tolist())
        },
        "atom_unk_count": int(atom_counts[UNK_ATOM_INDEX]) if atom_counts.size > UNK_ATOM_INDEX else 0,
        "atom_unk_fraction": float(atom_counts[UNK_ATOM_INDEX] / max(n_atoms, 1))
        if atom_counts.size > UNK_ATOM_INDEX
        else 0.0,
        "atom_unk_molecules": int(unk_molecules.size),
        "atom_unk_molecule_fraction": float(unk_molecules.size / max(int(cache["n_molecules"]), 1)),
        "atom_unk_elements": dict(sorted(element_counts.items(), key=lambda kv: -kv[1])),
        "atom_unk_molecules_inspected": int(inspected),
        "atom_unk_inspection_failures": parse_failures,
        "bond_class_counts": {
            (spec.bond_type_names[i - 1] if 1 <= i <= spec.n_bond_types else f"id{i}"): int(c)
            for i, c in enumerate(bond_counts.tolist())
            if i > 0
        },
        "bond_unk_count": int(bond_counts[UNK_BOND_ID]) if bond_counts.size > UNK_BOND_ID else 0,
        "bond_unk_fraction": float(bond_counts[UNK_BOND_ID] / max(n_bonds, 1))
        if bond_counts.size > UNK_BOND_ID
        else 0.0,
        "s2_records": {
            "total": int(s2_per_atom.sum()),
            "per_molecule_max": int(s2_per_molecule.max()) if s2_per_molecule.size else 0,
            "per_molecule_p99": float(np.percentile(s2_per_molecule, 99)) if s2_per_molecule.size else 0.0,
            "per_molecule_mean": float(s2_per_molecule.mean()) if s2_per_molecule.size else 0.0,
        },
        "atoms_per_molecule": {
            "max": int(n_atoms_per_molecule.max()),
            "p99": float(np.percentile(n_atoms_per_molecule, 99)),
            "mean": float(n_atoms_per_molecule.mean()),
        },
        "s1_codes": int(np.asarray(nbr_ptr)[-1]),
        "s1_equals_2_bonds": bool(int(np.asarray(nbr_ptr)[-1]) == 2 * n_bonds),
        "n_multi_component_molecules": int((components > 1).sum()),
        "max_components": int(components.max()) if components.size else 0,
        "non_finite_values": nonfinite,
        "constant_atom_scalar_columns": const_scalar,
        "constant_global_columns": const_global,
        "mapping_suitability": {
            "atom_unk_is_material": bool(atom_counts[UNK_ATOM_INDEX] / max(n_atoms, 1) > 1e-3)
            if atom_counts.size > UNK_ATOM_INDEX
            else False,
            "bond_unk_is_material": bool(
                (bond_counts[UNK_BOND_ID] / max(n_bonds, 1) > 1e-3) if bond_counts.size > UNK_BOND_ID else False
            ),
            "note": (
                "lipo_v1 maps unsupported elements to a single UNK atom class (never dropped); "
                "the recorded elements say which chemistry that affects on this dataset"
            ),
        },
    }
    payload["status"] = (
        "OK"
        if all(v == 0 for v in nonfinite.values()) and payload["s1_equals_2_bonds"]
        else "FAILED"
    )
    return payload


def _scan_chunk(task: tuple[list[int], list[str]]) -> dict[str, Any]:
    """Parse-mode ledger for one chunk of stored SMILES (label-blind)."""
    import mb2_rdkit as R

    rows, smiles = task
    modes = {"strict": 0, "unsanitized_fallback": 0}
    fallback: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row, smi in zip(rows, smiles):
        try:
            raw, mode = R.smiles_to_raw_row_relaxed(smi, int(row))
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            failures.append({"row": int(row), "smiles": smi, "error": f"{type(exc).__name__}: {exc}"})
            continue
        modes[mode] += 1
        if mode != "strict":
            fallback.append(
                {
                    "row": int(row),
                    "mode": str(mode),
                    "strict_error": str(raw.get("strict_error", ""))[:240],
                    "smiles": smi,
                }
            )
    return {"modes": modes, "fallback": fallback, "failures": failures, "n": len(rows)}


def parse_ledger(tag: str, *, workers: int = 12, chunk: int = 20_000) -> dict[str, Any]:
    """Full-coverage per-row parse ledger, recomputed from the stored SMILES."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    t0 = time.perf_counter()
    cache = D.load_cache(C.CACHE / tag)
    n = int(cache["n_molecules"])
    smiles = list(cache["smiles"])
    tasks = [
        (list(range(i, min(i + chunk, n))), smiles[i : min(i + chunk, n)])
        for i in range(0, n, chunk)
    ]
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as pool:
        for res in pool.map(_scan_chunk, tasks):
            results.append(res)
    modes = {
        "strict": int(sum(r["modes"]["strict"] for r in results)),
        "unsanitized_fallback": int(sum(r["modes"]["unsanitized_fallback"] for r in results)),
    }
    fallback_rows = sorted([x for r in results for x in r["fallback"]], key=lambda x: x["row"])
    failures = [x for r in results for x in r["failures"]]
    lead = C.read_json(C.ANALYSIS / f"{tag}_cache_build.json") if (
        C.ANALYSIS / f"{tag}_cache_build.json"
    ).exists() else {}
    build_modes = lead.get("parse_modes") or {}
    payload: dict[str, Any] = {
        "schema": "lcrc-pcqm4mv2-parse-ledger-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "tag": tag,
        "n_rows": int(n),
        "coverage": "every row of the cache, recomputed independently from the stored SMILES",
        "n_strict": modes["strict"],
        "n_unsanitized_fallback": modes["unsanitized_fallback"],
        "fallback_fraction": float(modes["unsanitized_fallback"] / max(n, 1)),
        "n_failures": int(len(failures)),
        "failures": failures[:50],
        "policy": (
            "one documented fallback (sanitize=False + UpdatePropertyCache(strict=False)) is used only "
            "after strict sanitisation failed; every affected row is listed here and no row is dropped, "
            "substituted or renumbered"
        ),
        "fallback_rows": fallback_rows,
        "build_summary_parse_modes": build_modes or None,
        "agrees_with_build_summary": (
            None
            if not build_modes
            else bool(
                int(build_modes.get("strict", -1)) == modes["strict"]
                and int(build_modes.get("unsanitized_fallback", -1)) == modes["unsanitized_fallback"]
            )
        ),
        "workers": int(workers),
        "seconds": float(time.perf_counter() - t0),
    }
    payload["status"] = (
        "OK" if not failures and payload["n_strict"] + payload["n_unsanitized_fallback"] == n else "FAILED"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="pcqm_train1m")
    parser.add_argument("--mode", default="oov", choices=["oov", "parse", "both"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    status = 0
    if args.mode in ("oov", "both"):
        t0 = time.perf_counter()
        spec = FS.load_spec(C.ANALYSIS / "pcqm_spec.json")
        cache = D.load_cache(C.CACHE / args.tag)
        payload = oov_audit(cache, spec)
        payload["seconds"] = float(time.perf_counter() - t0)
        out = Path(args.out) if args.out else C.ANALYSIS / f"{args.tag}_oov_audit.json"
        C.write_json(out, payload)
        C.log(f"[audit] {args.tag} oov: {payload['status']} -> {C.track_rel(out)}")
        status |= 0 if payload["status"] == "OK" else 1
    if args.mode in ("parse", "both"):
        payload = parse_ledger(args.tag, workers=int(args.workers))
        out = C.ANALYSIS / f"{args.tag}_parse_ledger_scan.json"
        C.write_json(out, payload)
        C.log(
            f"[audit] {args.tag} parse: {payload['status']} "
            f"strict={payload['n_strict']} fallback={payload['n_unsanitized_fallback']} "
            f"-> {C.track_rel(out)}"
        )
        status |= 0 if payload["status"] == "OK" else 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
