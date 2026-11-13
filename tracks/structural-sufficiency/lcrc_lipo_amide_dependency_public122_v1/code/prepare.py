"""Structure-only preparation: target catalog, populations and view manifest.

Everything here is label-blind and prediction-blind.  The step writes the
frozen protocol inputs that the (remote) inference step consumes:

* ``analysis/inputs_manifest.json`` - byte hashes, split rows, checkpoint,
  feature spec, code hashes, structural coverage.
* ``analysis/motif_catalog.csv`` - one row per restricted amide instance.
* ``analysis/molecule_catalog.csv`` - one row per molecule with >= 1 instance
  (split, scaffold key, population membership).
* ``analysis/r_candidates.csv`` - every remote ``q-r`` candidate of every
  instance, with the frozen cost tuple and the minimal-cost tie flag.
* ``analysis/view_manifest.csv`` - one row per unique (row_id, cut set) view
  for Valid+Test, with the scopes that must be predicted.
* ``analysis/reference_o_predictions.csv`` - the frozen checkpoint's
  per-molecule Valid/Test predictions (baseline identity input).
* ``analysis/prepare_checks.json`` - pre-registered counts, structural gates,
  zero-cut re-encoding identity against the frozen caches.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import motif as M
import views as V

#: Pre-registered structural counts (protocol section 3/5).  A mismatch means an
#: implementation bug; the protocol is not adjusted.
EXPECTED_COUNTS = {
    "ALL": {
        "molecules": {"train": 1438, "valid": 194, "test": 185},
        "instances": {"train": 1657, "valid": 234, "test": 208},
    },
    "MATCHED": {
        "molecules": {"train": 343, "valid": 52, "test": 42},
        "instances": {"train": 354, "valid": 54, "test": 43},
    },
    "BALANCED": {"molecules": {"train": 172, "valid": 28, "test": 23}},
    "R": {"molecules": {"valid": 49, "test": 43}},
}

SCOPE_FULL_ONLY = "FULL"
SCOPE_BOTH = "FULL,G_FIXED"


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
    return p


def read_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# view identifiers (deduplicated by row + physical cut set)
# ---------------------------------------------------------------------------


def view_id(row_id: int, kind: str, cut_a: int | None = None, cut_b: int | None = None) -> str:
    if kind == "O":
        return f"{int(row_id)}:O"
    if cut_a is None or cut_b is None:
        raise ValueError("cut atoms are required for a non-O view")
    return f"{int(row_id)}:X{min(int(cut_a), int(cut_b))}-{max(int(cut_a), int(cut_b))}"


# ---------------------------------------------------------------------------
# catalog construction
# ---------------------------------------------------------------------------


def build_catalogs(
    payload: Mapping[str, Any],
    src: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = payload["rows"]["Lipophilicity"]
    splits = C.split_of_row(src)
    instance_rows: list[dict[str, Any]] = []
    molecule_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    graph_cache: dict[int, M.RawGraph] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {
        p: {"molecules": {s: 0 for s in C.SPLITS}, "instances": {s: 0 for s in C.SPLITS}}
        for p in ("ALL", "MATCHED", "BALANCED", "R")
    }
    for rid in range(C.N_MOLECULES):
        g = M.summarise_raw(rows[rid], rid)
        graph_cache[rid] = g
        insts = M.find_instances(g)
        if not insts:
            continue
        split = splits[rid]
        n_all = len(insts)
        n_matched = sum(1 for i in insts if i.matched)
        n_balanced = sum(1 for i in insts if abs(i.n_side - i.acyl_side) <= C.MATCHED_SIDE_TOL)
        n_r = 0
        for inst in insts:
            rc = M.remote_candidates(g, inst, same_component=True)
            rc_any = M.remote_candidates(g, inst, same_component=False)
            base = inst.as_row()
            base["instance_id"] = f"{rid}:{inst.c}"
            base["split"] = split
            base["view_t"] = view_id(rid, "T", inst.c, inst.n)
            base["view_k"] = view_id(rid, "K", inst.c, inst.acyl_c)
            if rc["n_candidates"]:
                n_r += 1
                vids = [view_id(rid, "R", t.q, t.r) for t in rc["ties"]]
                base["view_r"] = ";".join(vids)
                base["r_n_candidates"] = rc["n_candidates"]
                base["r_n_candidates_any_component"] = rc_any["n_candidates"]
                base["r_n_ties"] = rc["n_ties"]
                base["r_best_cost"] = "|".join(str(int(x)) for x in rc["best_cost"])
                base["r_best_is_amide"] = int(any(t.is_amide_like for t in rc["ties"]))
                base["r_best_q"] = ";".join(str(t.q) for t in rc["ties"])
                base["r_best_r"] = ";".join(str(t.r) for t in rc["ties"])
                base["r_best_sides"] = ";".join(str(t.side_small) for t in rc["ties"])
                for rank, t in enumerate(rc["ties"]):
                    row = {
                        "instance_id": base["instance_id"],
                        "row_id": rid,
                        "split": split,
                        "is_best_tie": 1,
                        "tie_rank": rank,
                        **t.as_row(),
                    }
                    candidate_rows.append(row)
                for t in rc["ties"]:
                    if t.is_amide_like:
                        base["r_best_amide_like"] = 1
            else:
                base["view_r"] = ""
                base["r_n_candidates"] = 0
                base["r_n_candidates_any_component"] = rc_any["n_candidates"]
                base["r_n_ties"] = 0
            instance_rows.append(base)
        counts["ALL"]["molecules"][split] += 1
        counts["ALL"]["instances"][split] += n_all
        counts["MATCHED"]["molecules"][split] += int(n_matched > 0)
        counts["MATCHED"]["instances"][split] += n_matched
        counts["BALANCED"]["molecules"][split] += int(n_balanced > 0)
        counts["BALANCED"]["instances"][split] += n_balanced
        counts["R"]["molecules"][split] += int(n_r > 0)
        counts["R"]["instances"][split] += n_r
        molecule_rows.append(
            {
                "row_id": rid,
                "split": split,
                "smiles": str(rows[rid]["smiles"]),
                "canonical_smiles": str(rows[rid]["canonical_smiles"]),
                "n_atoms": int(g.n_atoms),
                "n_bonds": int(g.n_bonds),
                "n_components": int(g.component.max()) + 1 if g.n_atoms else 0,
                "scaffold_key": M.bemis_murcko_key(str(rows[rid]["smiles"])),
                "n_instances": n_all,
                "n_matched": n_matched,
                "n_balanced": n_balanced,
                "n_r_instances": n_r,
                "pop_all": 1,
                "pop_matched": int(n_matched > 0),
                "pop_balanced": int(n_balanced > 0),
                "pop_r": int(n_r > 0),
            }
        )
    return instance_rows, molecule_rows, candidate_rows, {"counts": counts, "graphs": graph_cache}


# ---------------------------------------------------------------------------
# populations
# ---------------------------------------------------------------------------


def instance_population(row: Mapping[str, Any], population: str) -> bool:
    if population == "ALL":
        return True
    if population == "MATCHED":
        return bool(int(row["matched"]))
    if population == "BALANCED":
        return bool(int(row["balanced"]))
    if population == "R":
        return bool(str(row.get("view_r", "")).strip())
    raise KeyError(population)


def population_instances(
    instance_rows: Sequence[Mapping[str, Any]], population: str
) -> dict[int, list[dict[str, Any]]]:
    """Molecules of a population -> only the instances qualified by that population."""
    out: dict[int, list[dict[str, Any]]] = {}
    for row in instance_rows:
        if not instance_population(row, population):
            continue
        out.setdefault(int(row["row_id"]), []).append(dict(row))
    for v in out.values():
        v.sort(key=lambda r: int(r["c"]))
    return out


# ---------------------------------------------------------------------------
# view manifest
# ---------------------------------------------------------------------------


def build_view_manifest(
    instance_rows: Sequence[Mapping[str, Any]],
    split_rows: Mapping[str, Sequence[int]],
    cut_splits: Sequence[str] = ("valid", "test"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Unique (row_id, cut set) views required by the analysis populations.

    ``O`` views cover **every** molecule of all 4200 rows: the frozen baseline is
    reproduced on the full Valid/Test 420/420 and the Train O predictions are
    needed for the natural-pair endpoints and the input-range check.  ``T`` /
    ``K`` / ``R`` views exist only for Valid/Test molecules that carry an
    instance of the corresponding population (no Train masked view is run).
    """
    views: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        for rid in [int(x) for x in split_rows[split]]:
            vid_o = view_id(rid, "O")
            views.setdefault(
                vid_o,
                {
                    "view_id": vid_o,
                    "row_id": rid,
                    "split": split,
                    "kind": "O",
                    "cut_a": -1,
                    "cut_b": -1,
                    "kinds": set(),
                    "instances": set(),
                },
            )
            views[vid_o]["kinds"].add("O")
    for split in cut_splits:
        for row in instance_rows:
            if row["split"] != split:
                continue
            rid = int(row["row_id"])
            for kind, key in (("T", "view_t"), ("K", "view_k"), ("R", "view_r")):
                raw = str(row.get(key, "") or "").strip()
                if not raw:
                    continue
                for vid in [x for x in raw.split(";") if x]:
                    cut = vid.split(":X")[1]
                    a, b = (int(x) for x in cut.split("-"))
                    views.setdefault(
                        vid,
                        {
                            "view_id": vid,
                            "row_id": rid,
                            "split": split,
                            "kind": kind,
                            "cut_a": a,
                            "cut_b": b,
                            "kinds": set(),
                            "instances": set(),
                        },
                    )
                    views[vid]["kinds"].add(kind)
                    views[vid]["instances"].add(str(row["instance_id"]))
    out: list[dict[str, Any]] = []
    for vid in sorted(views, key=lambda s: (int(s.split(":")[0]), s)):
        v = views[vid]
        kinds = sorted(v["kinds"])
        out.append(
            {
                "view_id": vid,
                "row_id": v["row_id"],
                "split": v["split"],
                "cut_a": v["cut_a"],
                "cut_b": v["cut_b"],
                "kinds": "+".join(kinds),
                "n_instances": len(v["instances"]),
                "instances": ";".join(sorted(v["instances"])),
                "scopes": SCOPE_FULL_ONLY if kinds == ["O"] else SCOPE_BOTH,
            }
        )
    summary = {
        "n_views": len(out),
        "per_split": {
            s: {
                "n_views": sum(1 for r in out if r["split"] == s),
                "n_O": sum(1 for r in out if r["split"] == s and r["kinds"] == "O"),
                "n_cut_views": sum(1 for r in out if r["split"] == s and r["kinds"] != "O"),
                "n_scoped_predictions": sum(
                    (1 if r["scopes"] == SCOPE_FULL_ONLY else 2) for r in out if r["split"] == s
                ),
            }
            for s in ("train", "valid", "test")
        },
    }
    return out, summary


# ---------------------------------------------------------------------------
# frozen reference predictions (from the executed round's npz)
# ---------------------------------------------------------------------------


def reference_o_predictions() -> list[dict[str, Any]]:
    path = Path(C.REFERENCE_PREDICTIONS)
    if not path.exists():
        raise FileNotFoundError(f"frozen per-molecule predictions missing: {C.track_rel(path)}")
    d = np.load(path, allow_pickle=True)
    out: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        rows = np.asarray(d[f"{split}_rows"], dtype=np.int64)
        std = np.asarray(d[f"{split}_pred_standardised"], dtype=np.float64)
        raw = np.asarray(d[f"{split}_pred_raw"], dtype=np.float64)
        y = np.asarray(d[f"{split}_y"], dtype=np.float64)
        for r, s, p, yy in zip(rows.tolist(), std.tolist(), raw.tolist(), y.tolist()):
            out.append(
                {
                    "row_id": int(r),
                    "split": split,
                    "pred_standardised": float(s),
                    "pred_raw": float(p),
                    "y": float(yy),
                }
            )
    return out


# ---------------------------------------------------------------------------
# structural gates
# ---------------------------------------------------------------------------


def _nx_bridges(n: int, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(int(n)))
    g.add_edges_from((int(u), int(v)) for u, v in zip(bu.tolist(), bv.tolist()))
    found = {tuple(sorted((int(a), int(b)))) for a, b in nx.bridges(g)}
    return np.asarray([tuple(sorted((int(u), int(v)))) in found for u, v in zip(bu.tolist(), bv.tolist())])


def check_structure(rows: Sequence[Mapping[str, Any]], graphs: Mapping[int, M.RawGraph]) -> dict[str, Any]:
    """Bridge definition, ring flags and graph/analysis agreement."""
    import molecular_graph as MG

    n_bridge_checked = 0
    bridge_mismatch: list[int] = []
    ring_mismatch: list[int] = []
    for rid, g in graphs.items():
        if g.n_atoms == 0 or g.n_bonds == 0:
            continue
        nx_flag = _nx_bridges(g.n_atoms, g.bu, g.bv)
        n_bridge_checked += 1
        if not np.array_equal(nx_flag, g.is_bridge):
            bridge_mismatch.append(int(rid))
        graph = V.view_graph(V.row_from_artifact(rows[rid]), C.DD.frozen_spec())
        analysis = MG.analyse_graph(graph)
        if not np.array_equal(analysis.ring_atom, g.ring_atom):
            ring_mismatch.append(int(rid))
        if int(analysis.n_components) != int(g.component.max()) + 1:
            ring_mismatch.append(int(rid))
    return {
        "n_molecules_bridge_crosschecked": n_bridge_checked,
        "bridge_mismatch_rows": bridge_mismatch,
        "ring_or_component_mismatch_rows": ring_mismatch,
        "pass": not bridge_mismatch and not ring_mismatch,
        "definitions": {
            "bridge": "Tarjan bridge on the original undirected physical graph; cross-checked with networkx.bridges",
            "ring_atom": "atom with at least one incident non-bridge bond; cross-checked with molecular_graph.bincount ring_atom",
        },
    }


def check_zero_cut_identity(
    rows: Sequence[Mapping[str, Any]], spec, *, splits: Sequence[str] = ("valid", "test")
) -> dict[str, Any]:
    """Zero-cut rebuild must equal the frozen structure cache byte for byte."""
    src = C.load_source("none")
    out: dict[str, Any] = {"splits": {}, "pass": True}
    for split in splits:
        cache = C.DD.load_or_build_split_cache(
            C.DATA_REPEAT, split, src["rows"][split], src["smiles"], spec, rebuild=False
        )
        bad: list[int] = []
        for i, rid in enumerate(np.asarray(cache["rows"]).tolist()):
            row = V.row_from_artifact(rows[int(rid)])
            row["n_components"] = V.n_components_of_row(row)
            graph = V.view_graph(row, spec)
            mol = C.DD.PREP_DATA.build_molecule(graph, spec)
            rebuilt = {
                "n_atoms": int(mol["n_atoms"]),
                "n_bonds": int(mol["n_bonds"]),
                "atom_feat": mol["atom_feat"],
                "atom_type": mol["atom_type"],
                "atom_scalar": mol["atom_scalar"],
                "bond_feat": mol["bond_feat"],
                "bond_type": mol["bond_type"],
                "bond_index": mol["bond_index"],
                "nbr_ptr": mol["nbr_ptr"],
                "nbr_code": mol["nbr_code"],
                "path_ptr": mol["path_ptr"],
                "path_cat": mol["path_cat"],
                "path_end63": mol["path_end63"],
                "path_end": mol["path_end"],
                "path_closure": mol["path_closure"],
                "global": mol["global"],
                "n_components": int(mol["n_components"]),
                "n_junctions": int(mol["n_junctions"]),
                "n_ring_systems": int(mol["n_ring_systems"]),
            }
            cmp = V.compare_cached_molecule(rebuilt, V.cache_slice(cache, i))
            if not cmp["all_equal"]:
                bad.append(int(rid))
        ok = not bad
        out["splits"][split] = {
            "n_checked": int(cache["n_molecules"]),
            "mismatch_rows": bad[:20],
            "n_mismatch": len(bad),
            "pass": ok,
            "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
        }
        out["pass"] = out["pass"] and ok
    return out


def check_permutation_catalog(
    rows: Sequence[Mapping[str, Any]],
    instance_rows: Sequence[Mapping[str, Any]],
    *,
    n_molecules: int = 3,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Atom-relabelling invariance of the structural catalog (no model involved)."""
    matched_test = [r for r in instance_rows if r["split"] == "test" and int(r["matched"]) == 1]
    take_rows = sorted({int(r["row_id"]) for r in matched_test})[:n_molecules]
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for rid in take_rows:
        row = V.row_from_artifact(rows[rid])
        n = int(row["atom_class"].shape[0])
        perm = rng.permutation(n)
        prows = V.permute_row(row, perm)
        g = M.summarise_raw(prows, rid)
        found = {(i.c, i.o, i.n, i.acyl_c) for i in M.find_instances(g)}
        expected = {
            (
                int(perm[int(r["c"])]),
                int(perm[int(r["o"])]),
                int(perm[int(r["n"])]),
                int(perm[int(r["acyl_c"])]),
            )
            for r in instance_rows
            if int(r["row_id"]) == rid
        }
        out.append(
            {
                "row_id": int(rid),
                "permutation": perm.tolist(),
                "match": found == expected,
                "n_found": len(found),
                "n_expected": len(expected),
            }
        )
    return {"molecules": out, "pass": all(o["match"] for o in out)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    spec = C.DD.frozen_spec()
    src = C.load_source("all")
    payload = C.load_raw_rows()
    rows = payload["rows"]["Lipophilicity"]

    ck = C.load_checkpoint()
    norm = C.frozen_norm(ck)
    tgt = C.target_norm(ck)

    C.log("building structure-only catalogs ...")
    instance_rows, molecule_rows, candidate_rows, extra = build_catalogs(payload, src)
    counts = extra["counts"]
    graphs = extra["graphs"]

    mismatches = {}
    for pop, expected in EXPECTED_COUNTS.items():
        for field, per_split in expected.items():
            for split, want in per_split.items():
                got = int(counts[pop][field][split])
                if got != want:
                    mismatches[f"{pop}/{field}/{split}"] = {"expected": want, "actual": got}
    if mismatches:
        raise SystemExit(f"STRUCTURE_COUNT_MISMATCH: {mismatches}")

    C.log("structural gates ...")
    structure_check = check_structure(rows, graphs)
    if not structure_check["pass"]:
        raise SystemExit(f"STRUCTURE_CHECK_FAILED: {structure_check}")

    C.log("zero-cut re-encoding identity vs frozen caches ...")
    zero_cut = check_zero_cut_identity(rows, spec)
    if not zero_cut["pass"]:
        raise SystemExit(f"ZERO_CUT_IDENTITY_FAILED: {zero_cut}")

    C.log("atom-permutation catalog invariance ...")
    perm_check = check_permutation_catalog(rows, instance_rows)

    C.log("view manifest ...")
    view_rows, view_summary = build_view_manifest(instance_rows, src["rows"])

    # reference predictions
    ref_rows = reference_o_predictions()

    instance_columns = [
        "instance_id",
        "row_id",
        "split",
        "c",
        "o",
        "n",
        "acyl_c",
        "degree_n",
        "degree_acyl",
        "ring_n",
        "ring_acyl",
        "matched",
        "n_side",
        "acyl_side",
        "component_size",
        "side_diff",
        "balanced",
        "small_side_target",
        "bond_cn",
        "bond_c_acyl",
        "bond_c_o",
        "view_t",
        "view_k",
        "view_r",
        "r_n_candidates",
        "r_n_candidates_any_component",
        "r_n_ties",
        "r_best_cost",
        "r_best_is_amide",
        "r_best_q",
        "r_best_r",
        "r_best_sides",
    ]
    molecule_columns = [
        "row_id",
        "split",
        "smiles",
        "canonical_smiles",
        "n_atoms",
        "n_bonds",
        "n_components",
        "scaffold_key",
        "n_instances",
        "n_matched",
        "n_balanced",
        "n_r_instances",
        "pop_all",
        "pop_matched",
        "pop_balanced",
        "pop_r",
    ]
    candidate_columns = [
        "instance_id",
        "row_id",
        "split",
        "is_best_tie",
        "tie_rank",
        "q",
        "r",
        "bond",
        "cost_degree",
        "cost_h",
        "cost_size",
        "is_amide_like",
        "degree_q",
        "atom_h_q",
        "side_small",
        "side_q",
    ]
    view_columns = [
        "view_id",
        "row_id",
        "split",
        "cut_a",
        "cut_b",
        "kinds",
        "n_instances",
        "instances",
        "scopes",
    ]

    written = {
        "motif_catalog.csv": write_csv(C.ANALYSIS / "motif_catalog.csv", instance_rows, instance_columns),
        "molecule_catalog.csv": write_csv(
            C.ANALYSIS / "molecule_catalog.csv", molecule_rows, molecule_columns
        ),
        "r_candidates.csv": write_csv(C.ANALYSIS / "r_candidates.csv", candidate_rows, candidate_columns),
        "view_manifest.csv": write_csv(C.ANALYSIS / "view_manifest.csv", view_rows, view_columns),
        "reference_o_predictions.csv": write_csv(
            C.ANALYSIS / "reference_o_predictions.csv",
            ref_rows,
            ["row_id", "split", "pred_standardised", "pred_raw", "y"],
        ),
    }
    written_hashes = {name: C.sha256_file(path) for name, path in written.items()}

    manifest = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "generated_utc": C.utcnow(),
        "config": C.config_snapshot(),
        "source_files": {k: C.track_rel(v) for k, v in C.resolve_source_files().items()},
        "data": {
            "source_commit": C.SOURCE_COMMIT,
            "csv_sha256": C.LIPOP_CSV_SHA256,
            "split_file": C.SPLIT_FILE,
            "split_file_sha256": C.SPLIT_FILE_SHA256,
            "split_rows_sha256": dict(C.SPLIT_ROWS_SHA256),
            "raw_artifact_sha256": C.RAW_ARTIFACT_SHA256,
            "rows": {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS},
            "row_sha256_recomputed": {s: C.row_sha256(src["rows"][s]) for s in C.SPLITS},
        },
        "model": {
            "checkpoint": C.CHECKPOINT_REL,
            "checkpoint_sha256": C.CHECKPOINT_SHA256,
            "state_hash": C.STATE_HASH,
            "variant": C.VARIANT,
            "parameter_count": C.EXPECTED_PARAMS,
            "target_norm": tgt,
            "norm_fingerprint": C.DD.PREP_DATA.norm_fingerprint(norm),
            "norm_source": "checkpoint buffers (frozen Train-only standardisation)",
            "reference_metrics": dict(C.REFERENCE),
        },
        "frozen_structure_caches": {s: torch_cache_info(C.DATA_REPEAT, s) for s in C.SPLITS},
        "structural_counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "coverage": {
            "n_molecules_with_instances": len(molecule_rows),
            "n_instances": len(instance_rows),
            "per_split_molecules": {s: sum(1 for r in molecule_rows if r["split"] == s) for s in C.SPLITS},
            "n_scaffold_clusters": len({r["scaffold_key"] for r in molecule_rows}),
        },
        "views": view_summary,
        "code_hashes": C.code_hashes(),
        "artifacts": {
            "files": written_hashes,
            "note": "sha256 of the committed protocol inputs consumed by infer.py",
        },
    }
    C.write_json(C.ANALYSIS / "inputs_manifest.json", manifest)

    checks = {
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "precounts_match_protocol": True,
        "observed_counts": counts,
        "structure": structure_check,
        "zero_cut_identity": zero_cut,
        "permutation_catalog": perm_check,
        "checkpoint": {
            "sha256": C.CHECKPOINT_SHA256,
            "state_hash": C.STATE_HASH,
            "variant": C.VARIANT,
            "parameter_count": C.EXPECTED_PARAMS,
            "best_epoch": C.BEST_EPOCH,
            "pass": True,
        },
        "view_summary": view_summary,
    }
    C.write_json(C.ANALYSIS / "prepare_checks.json", checks)

    C.log(f"instances={len(instance_rows)} molecules={len(molecule_rows)} views={len(view_rows)}")
    for s in C.SPLITS:
        C.log(
            f"{s}: ALL {counts['ALL']['molecules'][s]}mol/{counts['ALL']['instances'][s]}inst; "
            f"MATCHED {counts['MATCHED']['molecules'][s]}/{counts['MATCHED']['instances'][s]}; "
            f"BALANCED {counts['BALANCED']['molecules'][s]}/{counts['BALANCED']['instances'][s]}; "
            f"R {counts['R']['molecules'][s]}/{counts['R']['instances'][s]}"
        )
    C.log(f"views: {view_summary['per_split']}")
    return 0


def torch_cache_info(repeat: int, split: str) -> dict[str, Any]:
    import torch

    path = C.DD.cache_path(repeat, split)
    if not path.exists():
        return {"path": C.track_rel(path), "exists": False}
    cache = torch.load(path, map_location="cpu", weights_only=False)
    src = C.load_source("none")
    rows_ok = [int(x) for x in np.asarray(cache["rows"]).tolist()] == [int(x) for x in src["rows"][split]]
    return {
        "path": C.track_rel(path),
        "exists": True,
        "sha256": C.sha256_file(path),
        "n_molecules": int(cache["n_molecules"]),
        "content_fingerprint": cache["meta"].get("content_fingerprint"),
        "feature_spec_hash": cache["meta"].get("feature_spec_hash"),
        "labels_present": bool(cache["meta"].get("labels_present")),
        "rows_match_published_split": bool(rows_ok),
        "rows_sha256": C.row_sha256(np.asarray(cache["rows"]).tolist()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
