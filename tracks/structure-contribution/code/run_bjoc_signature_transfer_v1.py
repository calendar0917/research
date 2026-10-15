"""cscl-bjoc-signature-transfer-v1 — driver (CPU only, no labels, frozen models).

Commands
--------
verify-inputs   hash and presence check of every frozen input
freeze          write protocol_freeze.json (git commit + config/code hashes)
discovery       molecule-capped signature importance table from Phase-0 records
structure       RDKit dataset-graph SMILES / Murcko scaffolds / duplicate audit
validate        per-Junction IG, encodings, natural-replacement responses for
                official-valid structures and the Phase-0 diagnostic replay
analyze         statistics, strata, gates and the frozen three-way decision
semantics       RDKit chemical-semantics check of the pre-selected classes
finalize        artifact hashes; refuses to run if the freeze is broken

Discipline: no prediction labels, no official test, no Locked split, no new
training, no GPU, no search.  Historical artifacts are read-only.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_signature_transfer_v1 as bst  # noqa: E402
import bjoc_source_attribution_feasibility_v1 as bsaf  # noqa: E402
import bjoc_source_ownership_v1 as own  # noqa: E402
import bjoc_structure_explanation_audit_v1 as v1  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = bst.PROTOCOL_ID
OUT = bst.OUT
CONFIG_PATH = bst.CONFIG_PATH
FREEZE = OUT / "protocol_freeze.json"
DISCOVERY_TABLE = OUT / "discovery_table.json"
DISCOVERY_FOLDS = OUT / "discovery_folds.json"
STRUCTURE = OUT / "structure_overlap.json"
ANALYSIS = OUT / "validation_analysis.json"
SEMANTICS = OUT / "semantics_check.json"
DECISION = OUT / "final_decision.json"
FINALIZE = OUT / "finalize.json"

PHASE0 = REPO_ROOT / "tracks/structure-contribution/results/bjoc_source_ownership_v1/phase0_results.json"
SAMPLE = bst.BSAF_DIR / "sample_manifest.json"
OBJ_TRAIN = bst.F10K / "objects_train10k.pt"
OBJ_VALID = bst.F10K / "objects_valid.pt"
PREP_TRAIN = bst.F10K / "prep_train10k.json"
SOUPS = {0: bst.REC / "soup_R3_seed0.pt", 1: bst.REC / "soup_R3_seed1.pt"}
ZINC_ATOM_DICT = bst.ZINC_ROOT / "raw/atom_dict.pickle"
ZINC_BOND_DICT = bst.ZINC_ROOT / "raw/bond_dict.pickle"

PHASE0_RANDOM_RNG_BASE = 70_000_000
PHASE0_RANDOM_DRAWS = 16
KS = (1, 2, 4, 8)

DISCIPLINE = {
    "labels_read": False,
    "official_valid_structures_only": True,
    "official_valid_exposed_dev_set": True,
    "official_test_loaded": False,
    "locked_split_read": False,
    "new_training": False,
    "gpu_used": False,
}


def log(msg: str) -> None:
    print(f"[bst] {msg}", flush=True)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_paths() -> list[Path]:
    return [
        CONFIG_PATH,
        _CODE_DIR / "bjoc_signature_transfer_v1.py",
        Path(__file__),
        REPO_ROOT / "tracks/structure-contribution/tests/test_bjoc_signature_transfer_v1.py",
    ]


def code_hashes() -> dict[str, str]:
    return {str(p.relative_to(REPO_ROOT)): bst.file_sha256(p) for p in code_paths()}


def audit_inputs() -> dict[str, Any]:
    paths = {
        "phase0_results": PHASE0,
        "sample_manifest": SAMPLE,
        "objects_train10k": OBJ_TRAIN,
        "objects_valid": OBJ_VALID,
        "prep_train10k": PREP_TRAIN,
        "r3_soup0": SOUPS[0],
        "r3_soup1": SOUPS[1],
        "zinc_atom_dict": ZINC_ATOM_DICT,
        "zinc_bond_dict": ZINC_BOND_DICT,
    }
    inputs = {
        name: {
            "path": str(p.relative_to(REPO_ROOT)),
            "exists": p.exists(),
            "sha256": bst.file_sha256(p) if p.exists() else None,
        }
        for name, p in paths.items()
    }
    phase0 = bst.read_json(PHASE0)
    sample = bst.read_json(SAMPLE)
    inputs["phase0_shape"] = {
        "n_records_c": len(phase0.get("records_c", [])),
        "n_records_b": len(phase0.get("records_b", [])),
        "n_diagnostic_rows": len(sample.get("diagnostic_rows", [])),
        "n_reference_rows": len(sample.get("reference_rows", [])),
    }
    inputs["all_present"] = all(inputs[name]["exists"] for name in paths)
    return inputs


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    payload = audit_inputs()
    bst.write_json(OUT / "input_audit.json", payload)
    if not payload["all_present"]:
        log("verify-inputs: MISSING INPUTS")
        return 1
    log(f"verify-inputs: all present; {payload['phase0_shape']}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    if not audit["all_present"]:
        raise SystemExit("cannot freeze: missing inputs")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": utc_now(),
        "config_sha256": bst.file_sha256(CONFIG_PATH),
        "code_sha256": code_hashes(),
        "input_audit": audit,
        "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "discipline": DISCIPLINE,
    }
    bst.write_json(FREEZE, payload)
    log(f"freeze: written at {payload['git_commit'][:10]}")
    return 0


def verify_freeze() -> dict[str, Any]:
    if not FREEZE.exists():
        raise SystemExit("protocol not frozen: run freeze first")
    frozen = bst.read_json(FREEZE)
    for path, sha in frozen["code_sha256"].items():
        if bst.file_sha256(REPO_ROOT / path) != sha:
            raise SystemExit(f"frozen code changed: {path} (amend via a recorded protocol amendment)")
    if bst.file_sha256(CONFIG_PATH) != frozen["config_sha256"]:
        raise SystemExit("frozen config changed")
    return frozen


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def cmd_discovery(args: argparse.Namespace) -> int:
    verify_freeze()
    records = bst.read_json(PHASE0)["records_c"]
    table = bst.build_discovery_table(records, bst.DISCOVERY_FLOOR_MOLECULES)
    table.update(
        {
            "created_utc": utc_now(),
            "source": str(PHASE0.relative_to(REPO_ROOT)),
            "source_sha256": bst.file_sha256(PHASE0),
            "discipline": DISCIPLINE,
        }
    )
    bst.write_json(DISCOVERY_TABLE, table)
    log(
        f"discovery: {table['n_classes']} classes, {table['n_candidates']} candidates "
        f"(floor {table['floor_molecules']}), discovery junction coverage "
        f"{table['candidate_molecule_junction_coverage']:.4f}"
    )

    sample = bst.read_json(SAMPLE)
    diag = sorted(int(r) for r in sample["diagnostic_rows"])
    fold_a, fold_b = diag[0::2], diag[1::2]
    set_a = set(fold_a)
    folds: dict[str, Any] = {
        "rule": "diagnostic rows sorted ascending; even index -> fold A, odd index -> fold B",
        "fold_a_rows": fold_a,
        "fold_b_rows": fold_b,
        "internal_floor_molecules": bst.INTERNAL_CONTROL_FLOOR_MOLECULES,
        "created_utc": utc_now(),
        "discipline": DISCIPLINE,
    }
    for name, rows in (("A", set(fold_a)), ("B", set(fold_b))):
        recs = [r for r in records if int(r["row"]) in rows]
        folds[f"table{name}"] = bst.build_discovery_table(recs, bst.INTERNAL_CONTROL_FLOOR_MOLECULES)
    bst.write_json(DISCOVERY_FOLDS, folds)
    for name in ("A", "B"):
        t = folds[f"table{name}"]
        log(f"discovery fold {name}: {t['n_molecules']} rows, {t['n_candidates']} candidates")
    return 0


# ---------------------------------------------------------------------------
# structure audit (label-free)
# ---------------------------------------------------------------------------


def _zinc_split(name: str):
    from torch_geometric.datasets import ZINC

    return ZINC(root=str(bst.ZINC_ROOT), subset=True, split=name)


def _build_graph_index(ds, n: int) -> dict[str, Any]:
    smiles, scaffolds, tiers, n_atoms, n_bonds = [], [], [], [], []
    for i in range(n):
        d = ds[i]
        smi, scaf, tier = bst.canonical_smiles_and_scaffold(
            d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()
        )
        smiles.append(smi)
        scaffolds.append(scaf)
        tiers.append(tier)
        n_atoms.append(int(d.x.shape[0]))
        n_bonds.append(int(len({(min(int(d.edge_index[0, k]), int(d.edge_index[1, k])), max(int(d.edge_index[0, k]), int(d.edge_index[1, k]))) for k in range(d.edge_index.shape[1])})))
    return {"smiles": smiles, "scaffolds": scaffolds, "tiers": tiers, "n_atoms": n_atoms, "n_bonds": n_bonds}


def cmd_structure(args: argparse.Namespace) -> int:
    verify_freeze()
    ds_train = _zinc_split("train")
    ds_valid = _zinc_split("val")
    train = _build_graph_index(ds_train, len(ds_train))
    valid = _build_graph_index(ds_valid, len(ds_valid))
    log(f"structure: built {len(train['smiles'])} train / {len(valid['smiles'])} valid dataset graphs")

    sample = bst.read_json(SAMPLE)
    diag_rows = sorted(int(r) for r in sample["diagnostic_rows"])
    discovery_scaffolds = sorted({train["scaffolds"][r] for r in diag_rows})
    discovery_smiles = {train["smiles"][r] for r in diag_rows}
    train_smiles = {s for s in train["smiles"] if s}
    train_scaffolds = set(train["scaffolds"])

    objs_valid = torch.load(OBJ_VALID, map_location="cpu", weights_only=False)
    entries = []
    graph_match = 0
    for i in range(len(valid["smiles"])):
        obj = objs_valid[i]
        n_junc = int(obj.junc_index.shape[0])
        n_ring_bonds = int((obj.bond_feat[:, 4] > 0.5).sum())
        match = int(valid["n_atoms"][i] == int(obj.n_atoms)) and int(valid["n_bonds"][i]) == int(obj.bond_index.shape[0])
        graph_match += int(match)
        entries.append(
            {
                "row": i,
                "smiles": valid["smiles"][i],
                "scaffold": valid["scaffolds"][i],
                "sanitize_tier": valid["tiers"][i],
                "dup_train": bool(valid["smiles"][i] in train_smiles),
                "dup_discovery": bool(valid["smiles"][i] in discovery_smiles),
                "scaffold_in_discovery": bool(valid["scaffolds"][i] in discovery_scaffolds),
                "scaffold_in_train": bool(valid["scaffolds"][i] in train_scaffolds),
                "n_junctions": n_junc,
                "n_ring_bonds": n_ring_bonds,
                "graph_matches_objects": bool(match),
            }
        )

    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": DISCIPLINE,
        "n_train": len(train["smiles"]),
        "n_valid": len(valid["smiles"]),
        "n_discovery": len(diag_rows),
        "sanitize_tier_counts": dict(Counter(train["tiers"] + valid["tiers"])),
        "n_unique_train_smiles": len(train_smiles),
        "n_unique_valid_smiles": len({s for s in valid["smiles"] if s}),
        "n_internal_train_duplicates": len(train["smiles"]) - len(train_smiles),
        "valid_dup_train_rows": [e["row"] for e in entries if e["dup_train"]],
        "valid_dup_discovery_rows": [e["row"] for e in entries if e["dup_discovery"]],
        "n_valid_scaffold_in_discovery": sum(1 for e in entries if e["scaffold_in_discovery"]),
        "n_valid_scaffold_in_train": sum(1 for e in entries if e["scaffold_in_train"]),
        "n_unique_discovery_scaffolds": len(discovery_scaffolds),
        "n_valid_graph_match_objects": int(graph_match),
        "valid": entries,
        "discovery_rows": diag_rows,
        "discovery_scaffolds": discovery_scaffolds,
        "identity_note": (
            "canonical SMILES of the dataset graph (element word + charge + explicit H of the ZINC "
            "atom dictionary, kekulized bond orders); the dictionary cannot express uncharged N/O/S "
            "H counts, so this is a cache-level graph identity, not a full chemical SMILES"
        ),
    }
    bst.write_json(STRUCTURE, payload)
    log(
        f"structure: dup_train={len(payload['valid_dup_train_rows'])} "
        f"dup_discovery={len(payload['valid_dup_discovery_rows'])} "
        f"scaffold_in_discovery={payload['n_valid_scaffold_in_discovery']} "
        f"graph_match={payload['n_valid_graph_match_objects']}/1000"
    )
    return 0


# ---------------------------------------------------------------------------
# validation compute
# ---------------------------------------------------------------------------


def _obj_stats(obj: sro.MolObjects) -> dict[str, int]:
    return {
        "n_atoms": int(obj.n_atoms),
        "n_bonds": int(obj.bond_index.shape[0]),
        "n_junctions": int(obj.junc_index.shape[0]),
    }


def _reference_w_list(model, objs, rows: Sequence[int], stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    batch = v1.make_batch(objs, rows)
    feats = v1.run_features(model, batch)
    jb = batch["junc_batch"]
    out = []
    for g, row in enumerate(rows):
        w = feats["w"][jb == g]
        out.append(
            {
                "row": int(row),
                "n_atoms": int(stats[str(int(row))]["n_atoms"]),
                "n_bonds": int(stats[str(int(row))]["n_bonds"]),
                "n_junctions": int(stats[str(int(row))]["n_junctions"]),
                "w": w,
            }
        )
    return out


def _molecule_record(
    model,
    obj: sro.MolObjects,
    row: int,
    table: Mapping[str, Any],
    ref_list: Sequence[Mapping[str, Any]],
    seed: int,
    store_signatures: bool,
) -> dict[str, Any]:
    n_j = int(obj.junc_index.shape[0])
    arrs = bst.junction_arrays(obj)
    stats = _obj_stats(obj)
    ref = bst.nearest_reference(stats, ref_list)
    batch = v1.make_batch([obj], [0])
    feats = v1.run_features(model, batch)
    pack = bsaf.pack_from_features(model, batch, feats, 0)
    w = pack["w"]
    ig = bsaf.ig_scores(model, pack, w.mean(0))
    ig_abs = np.abs(ig["ig"].numpy())
    pct = bst.average_rank_percentile(ig_abs)
    dev = own.dev_ranking(w).numpy()
    rare = np.asarray([bst.rarity_score(sig, table["rarity"]) for sig in arrs["signatures"]], dtype=np.float64)
    deltas = bst.all_junction_responses(model, pack, ref["w"])

    cand_index = bst.table_candidate_index(table)
    cand_values = bst.table_candidate_values(table)
    codes = np.asarray([cand_index.get(sig, -1) for sig in arrs["signatures"]], dtype=np.int64)
    sig_arr = bst.signature_scores(arrs["signatures"], table, "primary")
    sig_min_arr = bst.signature_scores(arrs["signatures"], table, "min")

    scores = {
        "SIG": sig_arr,
        "SIG_MIN": sig_min_arr,
        "RING": arrs["in_ring"].astype(np.float64),
        "DEGREE": arrs["degree"].astype(np.float64),
        "RING_DEGREE": arrs["in_ring"] * 100.0 + arrs["degree"].astype(np.float64),
        "DEV": dev,
        "RARE": rare,
        "ORACLE": ig_abs,
    }
    top8 = {name: bst.top_k_indices(score, 8) for name, score in scores.items()}
    rand = bst.random_draw_sets(n_j, 4, bst.RANDOM_RNG_BASE + bst.RANDOM_RNG_STRIDE_SEED * seed + int(row))
    n_cand = int(cand_values.size)
    shuf = np.stack(
        [
            bst.shuffle_top_sets(
                cand_values,
                codes,
                np.random.default_rng(bst.SHUFFLE_RNG_BASE + bst.SHUFFLE_RNG_STRIDE_SEED * seed + s).permutation(n_cand),
                4,
            )
            for s in range(bst.SHUFFLE_PERMUTATIONS)
        ]
    ).astype(np.int64)

    rec: dict[str, Any] = {
        "row": int(row),
        "n_j": int(n_j),
        "sig_code": codes.astype(np.int16),
        "pct": pct.astype(np.float32),
        "ig_abs": ig_abs.astype(np.float64),
        "ig_signed": ig["ig"].numpy().astype(np.float64),
        "dev": dev.astype(np.float32),
        "rare": rare.astype(np.float32),
        "delta": deltas.astype(np.float64),
        "degree": arrs["degree"].astype(np.int16),
        "in_ring": arrs["in_ring"].astype(np.int16),
        "endpoint_ring_count": arrs["endpoint_ring_count"].astype(np.int16),
        "tc": sig_arr.astype(np.float32),
        "top8": {name: v.astype(np.int16) for name, v in top8.items()},
        "rand": rand.astype(np.int16),
        "shuf": shuf,
        "reference_row": int(ref["row"]),
    }
    if store_signatures:
        rec["signatures"] = [bst.sig_json(s) for s in arrs["signatures"]]
    return rec


def _phase0_random_replay(mol: Mapping[str, Any], seed: int, oracle_top4: np.ndarray) -> float:
    n_j = int(mol["n_j"])
    rng = np.random.default_rng(PHASE0_RANDOM_RNG_BASE + 100_000 * seed + int(mol["row"]))
    complement = np.asarray([j for j in range(n_j) if j not in set(int(x) for x in oracle_top4)], dtype=np.int64)
    vals: list[float] = []
    delta = np.asarray(mol["delta"], dtype=np.float64)
    for _ in range(PHASE0_RANDOM_DRAWS):
        if complement.size:
            draw = rng.choice(complement, size=min(4, complement.size), replace=False)
            vals.extend(np.abs(delta[draw]).tolist())
    return float(np.mean(vals)) if vals else float("nan")


def cmd_validate(args: argparse.Namespace) -> int:
    verify_freeze()
    table = bst.read_json(DISCOVERY_TABLE)
    sample = bst.read_json(SAMPLE)
    diag_rows = [int(r) for r in sample["diagnostic_rows"]]
    ref_rows = [int(r) for r in sample["reference_rows"]]
    records_b = {(int(r["seed"]), int(r["row"])): r for r in bst.read_json(PHASE0)["records_b"]}
    records_c = defaultdict(dict)
    for r in bst.read_json(PHASE0)["records_c"]:
        records_c[(int(r["seed"]), int(r["row"]))][int(r["j"])] = r

    objs_train = v1.load_objects()
    objs_valid = torch.load(OBJ_VALID, map_location="cpu", weights_only=False)
    norm = bst.read_json(PREP_TRAIN)
    valid_rows = list(range(len(objs_valid)))

    replay: dict[str, Any] = {
        "tolerance": bst.REPLAY_TOLERANCE,
        "reference_row_mismatches": 0,
        "oracle_top4_set_mismatches": 0,
        "dev_top4_set_mismatches": 0,
        "max_ig_abs_diff": 0.0,
        "max_dev_diff": 0.0,
        "signature_mismatches": 0,
        "degree_mismatches": 0,
        "in_ring_mismatches": 0,
        "max_rare_diff_definitional": 0.0,
        "max_ig_top4_response_diff": 0.0,
        "max_random_response_diff": 0.0,
        "n_diagnostic_molecules": 0,
        "n_diagnostic_junctions_seed0": 0,
    }
    provenance: dict[str, Any] = {"protocol": PROTOCOL_ID, "created_utc": utc_now(), "discipline": DISCIPLINE, "per_seed": {}}

    for seed in (0, 1):
        t0 = time.time()
        model = v1.load_model(seed, norm)
        ref_list = _reference_w_list(model, objs_train, ref_rows, sample["reference_stats"])
        mols_valid: list[dict[str, Any]] = []
        mols_diag: list[dict[str, Any]] = []
        n_skipped = 0
        for zone, rows, objs, target, store_sig in (
            ("valid", valid_rows, objs_valid, mols_valid, False),
            ("diag", diag_rows, objs_train, mols_diag, True),
        ):
            for row in rows:
                obj = objs[int(row)]
                if int(obj.junc_index.shape[0]) == 0:
                    n_skipped += 1
                    continue
                rec = _molecule_record(model, obj, int(row), table, ref_list, seed, store_sig)
                target.append(rec)
                if zone == "diag":
                    replay["n_diagnostic_molecules"] += 1
                    if seed == 0:
                        replay["n_diagnostic_junctions_seed0"] += int(rec["n_j"])
                    exp_b = records_b.get((seed, int(row)))
                    if exp_b is not None:
                        if int(exp_b["reference_row"]) != int(rec["reference_row"]):
                            replay["reference_row_mismatches"] += 1
                        oracle = rec["top8"]["ORACLE"][:4]
                        if [int(x) for x in oracle] != [int(x) for x in exp_b["top4_sets"]["IG"]]:
                            replay["oracle_top4_set_mismatches"] += 1
                        if [int(x) for x in rec["top8"]["DEV"][:4]] != [int(x) for x in exp_b["top4_sets"]["dev"]]:
                            replay["dev_top4_set_mismatches"] += 1
                        resp = float(np.abs(np.asarray(rec["delta"], dtype=np.float64)[oracle]).mean())
                        replay["max_ig_top4_response_diff"] = max(
                            replay["max_ig_top4_response_diff"], abs(resp - float(exp_b["top4_response"]["IG"]))
                        )
                        rand = _phase0_random_replay(rec, seed, oracle)
                        if exp_b["random_response"] is not None and not np.isnan(rand):
                            replay["max_random_response_diff"] = max(
                                replay["max_random_response_diff"], abs(rand - float(exp_b["random_response"]))
                            )
                    recs = records_c.get((seed, int(row)), {})
                    for j in range(int(rec["n_j"])):
                        exp = recs.get(j)
                        if exp is None:
                            replay["signature_mismatches"] += 1
                            continue
                        if bst.sig_json(bst.sig_key(exp["signature"])) != bst.sig_json(bst.sig_key(rec["signatures"][j])):
                            replay["signature_mismatches"] += 1
                        if int(exp["degree"]) != int(rec["degree"][j]):
                            replay["degree_mismatches"] += 1
                        if int(exp["in_ring"]) != int(rec["in_ring"][j]):
                            replay["in_ring_mismatches"] += 1
                        replay["max_ig_abs_diff"] = max(replay["max_ig_abs_diff"], abs(float(exp["ig_abs"]) - float(rec["ig_abs"][j])))
                        replay["max_dev_diff"] = max(replay["max_dev_diff"], abs(float(exp["dev"]) - float(rec["dev"][j])))
                        replay["max_rare_diff_definitional"] = max(replay["max_rare_diff_definitional"], abs(float(exp["rare"]) - float(rec["rare"][j])))
        np.savez_compressed(
            OUT / f"validation_raw_seed{seed}.npz",
            rows_valid=np.asarray(valid_rows, dtype=np.int64),
            mols_valid=np.array(mols_valid, dtype=object),
            rows_diag=np.asarray(diag_rows, dtype=np.int64),
            mols_diag=np.array(mols_diag, dtype=object),
        )
        provenance["per_seed"][str(seed)] = {
            "runtime_seconds": round(time.time() - t0, 2),
            "n_valid_molecules": len(mols_valid),
            "n_diagnostic_molecules": len(mols_diag),
            "n_skipped_empty_junctions": n_skipped,
            "raw_sha256": bst.file_sha256(OUT / f"validation_raw_seed{seed}.npz"),
            "reference_choices_sha256": __import__("hashlib").sha256(
                json.dumps([int(m["reference_row"]) for m in mols_valid], sort_keys=True).encode()
            ).hexdigest(),
        }
        log(f"validate seed {seed}: {len(mols_valid)} valid / {len(mols_diag)} diagnostic, {provenance['per_seed'][str(seed)]['runtime_seconds']}s")

    hard_ok = (
        replay["reference_row_mismatches"] == 0
        and replay["oracle_top4_set_mismatches"] == 0
        and replay["dev_top4_set_mismatches"] == 0
        and replay["signature_mismatches"] == 0
        and replay["degree_mismatches"] == 0
        and replay["in_ring_mismatches"] == 0
        and replay["max_ig_abs_diff"] <= bst.REPLAY_TOLERANCE
        and replay["max_dev_diff"] <= bst.REPLAY_TOLERANCE
        and replay["max_ig_top4_response_diff"] <= 1e-9
        and replay["max_random_response_diff"] <= 1e-9
    )
    replay["hard_ok"] = bool(hard_ok)
    replay["discovery_table_sha256"] = bst.file_sha256(DISCOVERY_TABLE)
    replay["structure_sha256"] = bst.file_sha256(STRUCTURE)
    bst.write_json(OUT / "discovery_replay.json", replay)
    bst.write_json(OUT / "validation_provenance.json", provenance)
    log(
        f"validate: replay {'OK' if hard_ok else 'FAILED'} "
        f"(max|dIG|={replay['max_ig_abs_diff']:.2e}, oracle sets={replay['oracle_top4_set_mismatches']}, "
        f"ref rows={replay['reference_row_mismatches']}, resp={replay['max_ig_top4_response_diff']:.2e}, "
        f"random={replay['max_random_response_diff']:.2e})"
    )
    if not hard_ok:
        raise SystemExit("Phase-0 replay guard failed; aborting before any validation analysis")
    return 0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _load_raw(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    z = np.load(OUT / f"validation_raw_seed{seed}.npz", allow_pickle=True)
    return list(z["mols_valid"]), list(z["mols_diag"])


def _A_top(mol: Mapping[str, Any], method: str, k: int) -> float:
    idx = np.asarray(mol["top8"][method], dtype=np.int64)[: min(int(k), int(mol["n_j"]))]
    return float(np.abs(np.asarray(mol["delta"], dtype=np.float64)[idx]).mean())


def _A_random(mol: Mapping[str, Any]) -> float:
    delta = np.abs(np.asarray(mol["delta"], dtype=np.float64))
    return float(delta[np.asarray(mol["rand"], dtype=np.int64)].mean())


def _A_shuffle(mol: Mapping[str, Any]) -> np.ndarray:
    delta = np.abs(np.asarray(mol["delta"], dtype=np.float64))
    return delta[np.asarray(mol["shuf"], dtype=np.int64)].mean(1)


def _arrays(mols: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for method in bst.METHODS:
        out[method] = {k: np.asarray([_A_top(m, method, k) for m in mols], dtype=np.float64) for k in KS}
    out["RANDOM"] = {4: np.asarray([_A_random(m) for m in mols], dtype=np.float64)}
    shuf = np.asarray([_A_shuffle(m) for m in mols], dtype=np.float64)
    out["SHUFFLE"] = {4: shuf.mean(1)}
    return out, shuf


def _contrast(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    boot = bst.bootstrap_mean(d)
    sign = bst.sign_test_one_sided(d)
    return {
        "mean": boot["mean"],
        "ci95_lower": boot["ci95_lower"],
        "ci95_upper": boot["ci95_upper"],
        "sign_test": sign,
        "n_molecules": int(d.size),
        "fraction_positive": float(np.mean(d > 0)) if d.size else float("nan"),
    }


def _class_matrix(mols: Sequence[Mapping[str, Any]], n_cand: int) -> np.ndarray:
    mat = np.full((len(mols), n_cand), np.nan, dtype=np.float64)
    for i, m in enumerate(mols):
        codes = np.asarray(m["sig_code"], dtype=np.int64)
        pct = np.asarray(m["pct"], dtype=np.float64)
        for c in np.unique(codes[codes >= 0]):
            mat[i, int(c)] = float(pct[codes == c].mean())
    return mat


def _nanmean_safe(mat: np.ndarray, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    sub = mat if idx is None else mat[idx]
    sums = np.nansum(sub, axis=0)
    counts = np.sum(~np.isnan(sub), axis=0)
    return sums / np.maximum(counts, 1), counts


def _class_level(mats: list[np.ndarray], cand_values: np.ndarray, min_molecules: int = 5) -> dict[str, Any]:
    stacked = np.stack(mats)
    pooled = np.nanmean(stacked, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vals, counts = _nanmean_safe(pooled)
        sel = np.where(counts >= int(min_molecules))[0]
        rho = bst.spearman(cand_values[sel], vals[sel])
        r = bst.pearson(cand_values[sel], vals[sel])
        rng = np.random.default_rng(bst.BOOTSTRAP_SEED)
        draws = []
        chunk = 100
        done = 0
        while done < bst.BOOTSTRAP_N:
            m = min(chunk, bst.BOOTSTRAP_N - done)
            idx = rng.integers(0, pooled.shape[0], size=(m, pooled.shape[0]))
            for row in idx:
                sub_vals, _sub_counts = _nanmean_safe(pooled, row)
                draws.append(bst.spearman(cand_values[sel], sub_vals[sel]))
            done += m
    draws = np.asarray(draws, dtype=np.float64)
    per_seed = {}
    for s, mat in enumerate(mats):
        vals_s, counts_s = _nanmean_safe(mat)
        sel_s = np.where(counts_s >= int(min_molecules))[0]
        per_seed[str(s)] = {
            "n_classes": int(sel_s.size),
            "spearman": bst.spearman(cand_values[sel_s], vals_s[sel_s]),
            "pearson": bst.pearson(cand_values[sel_s], vals_s[sel_s]),
        }
    return {
        "n_classes_used": int(sel.size),
        "min_molecules": int(min_molecules),
        "spearman": rho,
        "pearson": r,
        "spearman_ci95_lower": float(np.nanquantile(draws, 0.025)),
        "spearman_ci95_upper": float(np.nanquantile(draws, 0.975)),
        "per_seed": per_seed,
        "class_values": {
            str(int(c)): {"T": float(cand_values[c]), "valid_pct_mean": float(vals[c]), "valid_molecules": int(counts[c])}
            for c in sel
        },
    }


def _context_stability(
    mols_by_seed: list[Sequence[Mapping[str, Any]]],
    structure: Mapping[str, Any],
    table: Mapping[str, Any],
) -> dict[str, Any]:
    entries = structure["valid"]
    n_junc = np.asarray([e["n_junctions"] for e in entries], dtype=np.float64)
    n_ring = np.asarray([e["n_ring_bonds"] for e in entries], dtype=np.float64)
    size_q = np.quantile(n_junc, [1 / 3, 2 / 3])
    ring_q = np.quantile(n_ring, [1 / 3, 2 / 3])

    def size_bin(row: int) -> str:
        v = n_junc[row]
        return "low" if v <= size_q[0] else ("mid" if v <= size_q[1] else "high")

    def ring_bin(row: int) -> str:
        v = n_ring[row]
        return "low" if v <= ring_q[0] else ("mid" if v <= ring_q[1] else "high")

    def degree_bin(deg: int) -> str:
        return "2" if deg <= 2 else ("3" if deg == 3 else "4+")

    acc: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    molsets: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for seed, mols in enumerate(mols_by_seed):
        for m in mols:
            codes = np.asarray(m["sig_code"], dtype=np.int64)
            pct = np.asarray(m["pct"], dtype=np.float64)
            absd = np.abs(np.asarray(m["delta"], dtype=np.float64))
            erc = np.asarray(m["endpoint_ring_count"], dtype=np.int64)
            deg = np.asarray(m["degree"], dtype=np.int64)
            row = int(m["row"])
            for j in range(int(m["n_j"])):
                c = int(codes[j])
                if c < 0:
                    continue
                pairs = {
                    "endpoint_ring_count": str(int(erc[j])),
                    "center_degree": degree_bin(int(deg[j])),
                    "size_tercile": size_bin(row),
                    "mol_ring_tercile": ring_bin(row),
                }
                for var, val in pairs.items():
                    acc[c][var][val].append(float(pct[j]))
                    acc[c][var][val].append(float(absd[j]))
                    molsets[c][var].add(row)
    out: dict[str, Any] = {"tercile_cutoffs": {"n_junctions": [float(x) for x in size_q], "n_ring_bonds": [float(x) for x in ring_q]}, "classes": {}}
    cand_values = bst.table_candidate_values(table)
    for c, vars_ in sorted(acc.items()):
        cls = {
            "T": float(cand_values[c]),
            "support_molecules_discovery": int(table["candidates"][c]["support_molecules"]),
            "by": {},
        }
        for var, values in vars_.items():
            cls["by"][var] = {}
            for val, flat in sorted(values.items()):
                arr = np.asarray(flat, dtype=np.float64)
                cls["by"][var][val] = {
                    "mean_pct": float(arr[0::2].mean()) if arr.size else float("nan"),
                    "mean_abs_delta": float(arr[1::2].mean()) if arr.size else float("nan"),
                    "n_junction_seed_units": int(arr.size // 2),
                }
        out["classes"][str(c)] = cls
    return out


def cmd_analyze(args: argparse.Namespace) -> int:
    verify_freeze()
    table = bst.read_json(DISCOVERY_TABLE)
    structure = bst.read_json(STRUCTURE)
    folds = bst.read_json(DISCOVERY_FOLDS)
    cand_values = bst.table_candidate_values(table)
    n_cand = int(cand_values.size)
    mols0, diag0 = _load_raw(0)
    mols1, diag1 = _load_raw(1)
    arr0, shuf0 = _arrays(mols0)
    arr1, shuf1 = _arrays(mols1)

    pooled: dict[str, dict[int, np.ndarray]] = {}
    for method in arr0:
        pooled[method] = {k: (arr0[method][k] + arr1[method][k]) / 2.0 for k in arr0[method]}

    contrasts: dict[str, dict[str, Any]] = {}
    for base in ("RANDOM", "RING", "DEGREE", "RING_DEGREE", "DEV", "RARE", "SHUFFLE", "ORACLE"):
        contrasts[f"SIG_minus_{base}"] = _contrast(pooled["SIG"][4], pooled[base][4])
    per_seed: dict[str, Any] = {}
    for s, arr in enumerate((arr0, arr1)):
        per_seed[str(s)] = {
            "method_means_k4": {m: float(arr[m][4].mean()) for m in arr},
            "SIG_minus_RANDOM": _contrast(arr["SIG"][4], arr["RANDOM"][4]),
            "SIG_minus_RING": _contrast(arr["SIG"][4], arr["RING"][4]),
            "SIG_minus_DEV": _contrast(arr["SIG"][4], arr["DEV"][4]),
        }
    k_sensitivity = {
        str(k): {m: float(pooled[m][k].mean()) for m in bst.METHODS} for k in KS
    }
    k_sensitivity["4"]["RANDOM"] = float(pooled["RANDOM"][4].mean())
    k_sensitivity["4"]["SHUFFLE"] = float(pooled["SHUFFLE"][4].mean())
    oracle_fraction = float(
        (pooled["SIG"][4].mean() - pooled["RANDOM"][4].mean())
        / max(pooled["ORACLE"][4].mean() - pooled["RANDOM"][4].mean(), 1e-12)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ranking_quality = {}
        for s, mols in enumerate((mols0, mols1)):
            for method in ("SIG", "RING", "DEGREE", "RING_DEGREE", "DEV", "RARE", "ORACLE"):
                vals = []
                for m in mols:
                    score = {
                        "SIG": np.asarray(m["tc"], dtype=np.float64),
                        "RING": np.asarray(m["in_ring"], dtype=np.float64),
                        "DEGREE": np.asarray(m["degree"], dtype=np.float64),
                        "RING_DEGREE": np.asarray(m["in_ring"], dtype=np.float64) * 100.0 + np.asarray(m["degree"], dtype=np.float64),
                        "DEV": np.asarray(m["dev"], dtype=np.float64),
                        "RARE": np.asarray(m["rare"], dtype=np.float64),
                        "ORACLE": np.asarray(m["ig_abs"], dtype=np.float64),
                    }[method]
                    rho = bst.spearman(score, np.asarray(m["ig_abs"], dtype=np.float64))
                    if not np.isnan(rho):
                        vals.append(rho)
                ranking_quality[f"seed{s}_{method}"] = {"mean_spearman_vs_abs_ig": float(np.mean(vals)), "n_molecules": len(vals)}

    class_level = _class_level([_class_matrix(mols0, n_cand), _class_matrix(mols1, n_cand)], cand_values)

    flags = {int(m["row"]): structure["valid"][int(m["row"])] for m in mols0}
    masks = {
        "all": np.ones(len(mols0), dtype=bool),
        "scaffold_unseen_discovery": np.asarray([not flags[m["row"]]["scaffold_in_discovery"] for m in mols0], dtype=bool),
        "scaffold_seen_discovery": np.asarray([flags[m["row"]]["scaffold_in_discovery"] for m in mols0], dtype=bool),
        "no_exact_dup_train": np.asarray([not flags[m["row"]]["dup_train"] for m in mols0], dtype=bool),
    }
    masks["both"] = masks["scaffold_unseen_discovery"] & masks["no_exact_dup_train"]
    subsets = {}
    for name, mask in masks.items():
        subsets[name] = {
            "n_molecules": int(mask.sum()),
            "SIG_mean": float(pooled["SIG"][4][mask].mean()) if mask.any() else float("nan"),
            "RANDOM_mean": float(pooled["RANDOM"][4][mask].mean()) if mask.any() else float("nan"),
            "SIG_minus_RANDOM": _contrast(pooled["SIG"][4][mask], pooled["RANDOM"][4][mask]) if mask.any() else None,
            "SIG_minus_RING": _contrast(pooled["SIG"][4][mask], pooled["RING"][4][mask]) if mask.any() else None,
        }
    seed_corr = {
        "sig_top4_pearson": bst.pearson(arr0["SIG"][4], arr1["SIG"][4]),
        "sig_top4_spearman": bst.spearman(arr0["SIG"][4], arr1["SIG"][4]),
        "random_pearson": bst.pearson(arr0["RANDOM"][4], arr1["RANDOM"][4]),
    }

    # top-4 SIG class composition
    comp: Counter = Counter()
    total_known = 0
    total_sel = 0
    for mols in (mols0, mols1):
        for m in mols:
            codes = np.asarray(m["sig_code"], dtype=np.int64)
            for j in np.asarray(m["top8"]["SIG"], dtype=np.int64)[: min(4, int(m["n_j"]))]:
                total_sel += 1
                if codes[j] >= 0:
                    total_known += 1
                    comp[int(codes[j])] += 1
    coverage = {
        "top4_sig_known_fraction": float(total_known / max(total_sel, 1)),
        "top4_sig_n_selections": int(total_sel),
        "junction_known_fraction_seed0": float(
            np.mean([np.mean(np.asarray(m["sig_code"]) >= 0) for m in mols0])
        ),
        "top_classes": [
            {"class_index": int(c), "signature": table["candidates"][c]["signature"], "T": float(cand_values[c]), "count": int(n)}
            for c, n in comp.most_common(10)
        ],
    }

    strata = _context_stability([mols0, mols1], structure, table)

    # internal fold control
    def _control_for(mols: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
        fold_a = set(int(r) for r in folds["fold_a_rows"])
        rows: list[float] = []
        rand_rows: list[float] = []
        ring_rows: list[float] = []
        cov_hits = cov_tot = 0
        for m in mols:
            row = int(m["row"])
            tbl = folds["tableB"] if row in fold_a else folds["tableA"]
            sigs = [bst.sig_key(s) for s in m["signatures"]]
            scores = bst.signature_scores(sigs, tbl, "primary")
            top4 = bst.top_k_indices(scores, 4)
            delta = np.abs(np.asarray(m["delta"], dtype=np.float64))
            rows.append(float(delta[top4].mean()))
            rand_rows.append(_A_random(m))
            deg = np.asarray(m["degree"], dtype=np.float64)
            ring = np.asarray(m["in_ring"], dtype=np.float64)
            ring_scores = np.asarray(m["in_ring"], dtype=np.float64) * 100.0 + deg
            r_top = bst.top_k_indices(ring_scores, 4)
            ring_rows.append(float(delta[r_top].mean()))
            ci = bst.table_candidate_index(tbl)
            for j in top4:
                cov_tot += 1
                if ci.get(sigs[int(j)]) is not None:
                    cov_hits += 1
        a = np.asarray(rows, dtype=np.float64)
        b = np.asarray(rand_rows, dtype=np.float64)
        c = np.asarray(ring_rows, dtype=np.float64)
        return {
            "n_molecules": int(a.size),
            "SIG_mean": float(a.mean()),
            "RANDOM_mean": float(b.mean()),
            "RING_mean": float(c.mean()),
            "SIG_minus_RANDOM": _contrast(a, b),
            "SIG_minus_RING": _contrast(a, c),
            "top4_known_fraction": float(cov_hits / max(cov_tot, 1)),
        }

    internal_control = {"seed0": _control_for(diag0, 0), "seed1": _control_for(diag1, 1)}
    # pooled across seeds at the molecule level
    fold_vals = []
    for mols in (diag0, diag1):
        fold_a = set(int(r) for r in folds["fold_a_rows"])
        for m in mols:
            tbl = folds["tableB"] if int(m["row"]) in fold_a else folds["tableA"]
            sigs = [bst.sig_key(s) for s in m["signatures"]]
            scores = bst.signature_scores(sigs, tbl, "primary")
            top4 = bst.top_k_indices(scores, 4)
            delta = np.abs(np.asarray(m["delta"], dtype=np.float64))
            fold_vals.append((float(delta[top4].mean()), _A_random(m)))
    fv = np.asarray([v[0] for v in fold_vals])
    fr = np.asarray([v[1] for v in fold_vals])
    internal_control["pooled"] = {"SIG_minus_RANDOM": _contrast(fv, fr)}

    # signed secondary context
    signed = {
        "note": "signed delta of SIG top-4, context only; never a positive/negative chemical contribution",
        "per_seed": {},
    }
    for s, mols in enumerate((mols0, mols1)):
        vals = []
        for m in mols:
            idx = np.asarray(m["top8"]["SIG"], dtype=np.int64)[: min(4, int(m["n_j"]))]
            vals.append(float(np.asarray(m["delta"], dtype=np.float64)[idx].mean()))
        signed["per_seed"][str(s)] = {"mean_signed_delta": float(np.mean(vals))}

    gates = {
        "M0": bool(coverage["top4_sig_known_fraction"] >= 0.5),
        "G1": bool(
            contrasts["SIG_minus_RANDOM"]["mean"] > 0
            and contrasts["SIG_minus_RANDOM"]["ci95_lower"] > 0
            and contrasts["SIG_minus_RANDOM"]["sign_test"]["p_one_sided"] < 0.01
        ),
        "G2": bool(contrasts["SIG_minus_RING"]["ci95_lower"] > 0),
        "G3": bool(
            contrasts["SIG_minus_DEV"]["ci95_lower"] > 0
            and contrasts["SIG_minus_RARE"]["mean"] > 0
            and contrasts["SIG_minus_RING_DEGREE"]["mean"] > 0
            and contrasts["SIG_minus_DEGREE"]["mean"] > 0
        ),
        "G4": bool(
            subsets["scaffold_unseen_discovery"]["SIG_minus_RANDOM"] is not None
            and subsets["scaffold_unseen_discovery"]["SIG_minus_RANDOM"]["mean"] > 0
            and subsets["scaffold_unseen_discovery"]["SIG_minus_RANDOM"]["ci95_lower"] > 0
        ),
        "G5": bool(
            per_seed["0"]["SIG_minus_RANDOM"]["mean"] > 0 and per_seed["1"]["SIG_minus_RANDOM"]["mean"] > 0
        ),
        "G6": bool(contrasts["SIG_minus_SHUFFLE"]["mean"] > 0 and contrasts["SIG_minus_SHUFFLE"]["ci95_lower"] > 0),
        "G7": bool(oracle_fraction >= bst.ORACLE_FRACTION_MIN),
        "G8": bool(class_level["spearman"] > 0 and class_level["spearman_ci95_lower"] > 0),
    }
    label = bst.decide_from_gates(gates)

    analysis = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": DISCIPLINE,
        "primary_k": 4,
        "method_means_k4_pooled": {m: float(pooled[m][4].mean()) for m in pooled},
        "contrasts_pooled": contrasts,
        "per_seed": per_seed,
        "k_sensitivity_means": k_sensitivity,
        "oracle_fraction": oracle_fraction,
        "oracle_fraction_min": bst.ORACLE_FRACTION_MIN,
        "ranking_quality": ranking_quality,
        "class_level": class_level,
        "subsets": subsets,
        "seed_correlation": seed_corr,
        "coverage": coverage,
        "context_stability": strata,
        "internal_fold_control": internal_control,
        "signed_secondary": signed,
        "gates": gates,
        "label": label,
        "raw_sha256": {
            "seed0": bst.file_sha256(OUT / "validation_raw_seed0.npz"),
            "seed1": bst.file_sha256(OUT / "validation_raw_seed1.npz"),
        },
    }
    bst.write_json(ANALYSIS, analysis)
    bst.write_json(
        DECISION,
        {
            "protocol": PROTOCOL_ID,
            "created_utc": utc_now(),
            "label": label,
            "gates": gates,
            "oracle_fraction": oracle_fraction,
            "primary_contrast": contrasts["SIG_minus_RANDOM"],
            "discipline": DISCIPLINE,
            "analysis_sha256": bst.file_sha256(ANALYSIS),
        },
    )
    np.savez_compressed(
        OUT / "row_metrics.npz",
        **{f"seed{s}_{m}_k{k}": (arr0 if s == 0 else arr1)[m][k] for s in (0, 1) for m in arr0 for k in arr0[m]},
        **{f"pooled_{m}_k{k}": pooled[m][k] for m in pooled for k in pooled[m]},
    )
    log(f"analyze: label={label}")
    log(f"analyze: gates={gates}")
    log(
        f"analyze: SIG-RANDOM {contrasts['SIG_minus_RANDOM']['mean']:+.5f} "
        f"[{contrasts['SIG_minus_RANDOM']['ci95_lower']:+.5f},{contrasts['SIG_minus_RANDOM']['ci95_upper']:+.5f}] "
        f"p={contrasts['SIG_minus_RANDOM']['sign_test']['p_one_sided']:.2e} "
        f"oracle_frac={oracle_fraction:.3f} rho_class={class_level['spearman']:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# semantics
# ---------------------------------------------------------------------------


def cmd_semantics(args: argparse.Namespace) -> int:
    verify_freeze()
    table = bst.read_json(DISCOVERY_TABLE)
    candidate_rows = table["candidates"]
    support = np.asarray([int(c["support_molecules"]) for c in candidate_rows], dtype=np.int64)
    T = bst.table_candidate_values(table)
    eligible = np.where(support >= 50)[0]
    top3 = eligible[np.argsort(-T[eligible], kind="stable")[:3]]
    bottom3 = eligible[np.argsort(T[eligible], kind="stable")[:3]]
    most_frequent = int(np.argmax(support))
    selected_idx = list(dict.fromkeys([int(x) for x in np.concatenate([top3, bottom3, [most_frequent]])]))
    mols0, _ = _load_raw(0)
    mols1, _ = _load_raw(1)
    objs_valid = torch.load(OBJ_VALID, map_location="cpu", weights_only=False)
    ds = _zinc_split("val")

    def rdkit_view(row: int, j: int) -> dict[str, Any]:
        obj = objs_valid[row]
        d = ds[row]
        mol, tier = bst.build_rdkit_mol(d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        i, c, k = (int(v) for v in obj.junc_index[j])
        bonds = [int(b) for b in obj.junc_bond[j]]
        bond_info = []
        for b in bonds:
            u, v = (int(x) for x in obj.bond_index[b])
            rd_bond = mol.GetBondBetweenAtoms(u, v)
            bond_info.append(
                {
                    "atoms": [u, v],
                    "order": str(rd_bond.GetBondType()) if rd_bond is not None else None,
                    "in_ring_rdkit": bool(rd_bond.IsInRing()) if rd_bond is not None else None,
                }
            )
        return {
            "sanitize_tier": tier,
            "center_atom": c,
            "endpoint_atoms": [i, k],
            "center_degree_rdkit": int(mol.GetAtomWithIdx(c).GetDegree()),
            "center_in_ring_rdkit": bool(mol.GetAtomWithIdx(c).IsInRing()),
            "center_symbol": mol.GetAtomWithIdx(c).GetSymbol(),
            "bond_info": bond_info,
        }

    def instance(row: int, j: int) -> dict[str, Any]:
        obj = objs_valid[row]
        sig = own.junction_signature(obj, j)
        return {
            "valid_row": int(row),
            "junction": int(j),
            "signature": bst.sig_json(sig),
            "ig_pct": float(by_row[row]["pct"][j]),
            "delta_signed": float(by_row[row]["delta"][j]),
            "delta_abs": float(abs(by_row[row]["delta"][j])),
            "endpoint_ring_count": int(by_row[row]["endpoint_ring_count"][j]),
            "degree": int(by_row[row]["degree"][j]),
            "rdkit": rdkit_view(row, j),
        }

    awords, bwords = bst.zinc_words()
    by_row = {int(m["row"]): m for m in mols0}
    selected: list[dict[str, Any]] = []
    for ci in selected_idx:
        ci = int(ci)
        rows_here = sorted(
            int(m["row"]) for m in mols0 if np.any(np.asarray(m["sig_code"], dtype=np.int64) == ci)
        )
        chosen_rows = rows_here[:3]
        chosen = []
        for row in chosen_rows:
            idx = np.where(np.asarray(by_row[row]["sig_code"], dtype=np.int64) == ci)[0]
            pct = np.asarray(by_row[row]["pct"], dtype=np.float64)[idx]
            order = np.argsort(pct, kind="stable")
            j = int(idx[order[len(order) // 2]])
            chosen.append(instance(row, j))
        all_inst = []
        for row in rows_here:
            idx = np.where(np.asarray(by_row[row]["sig_code"], dtype=np.int64) == ci)[0]
            for j in idx:
                all_inst.append(instance(row, int(j)))
        counter = None
        if all_inst:
            order = sorted(range(len(all_inst)), key=lambda t: all_inst[t]["ig_pct"])
            for t in order:
                if all_inst[t]["valid_row"] not in chosen_rows:
                    counter = all_inst[t]
                    break
            if counter is None:
                counter = all_inst[order[0]]
        ctx_examples: dict[str, Any] = {}
        for inst in all_inst:
            key = str(inst["endpoint_ring_count"])
            if key not in ctx_examples:
                ctx_examples[key] = {"valid_row": inst["valid_row"], "junction": inst["junction"], "ig_pct": inst["ig_pct"], "delta_signed": inst["delta_signed"]}
        selected.append(
            {
                "class_index": ci,
                "signature": candidate_rows[ci]["signature"],
                "words": bst.signature_words(bst.sig_key(candidate_rows[ci]["signature"]), awords, bwords),
                "T": float(T[ci]),
                "support_molecules_discovery": int(support[ci]),
                "n_valid_molecules": int(len(rows_here)),
                "n_valid_junctions": int(len(all_inst)),
                "instances": chosen,
                "counterexample": counter,
                "endpoint_ring_count_examples": ctx_examples,
            }
        )

    # RDKit agreement over all instances of the selected classes
    agree = {
        "center_in_ring_total": 0,
        "center_in_ring_match": 0,
        "ring_bonds_total": 0,
        "ring_bonds_match": 0,
        "bond_order_total": 0,
        "bond_order_match": 0,
    }
    sel_set = set(int(x) for x in selected_idx)
    for row, m in enumerate(mols0):
        row = int(m["row"])
        codes = np.asarray(m["sig_code"], dtype=np.int64)
        obj = objs_valid[row]
        d = ds[row]
        mol, _tier = bst.build_rdkit_mol(d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        for j in np.where(np.isin(codes, list(sel_set)))[0]:
            sig = own.junction_signature(obj, int(j))
            i, c, k = (int(v) for v in obj.junc_index[int(j)])
            agree["center_in_ring_total"] += 1
            agree["center_in_ring_match"] += int(bool(mol.GetAtomWithIdx(c).IsInRing()) == bool(sig[3]))
            ring_hits = 0
            for b in [int(x) for x in obj.junc_bond[int(j)]]:
                u, v = (int(x) for x in obj.bond_index[b])
                rb = mol.GetBondBetweenAtoms(u, v)
                ring_hits += int(bool(rb.IsInRing()))
                agree["bond_order_total"] += 1
                agree["bond_order_match"] += int(
                    str(rb.GetBondType()) == {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE"}.get(int(obj.bond_type[b]), "?")
                )
            agree["ring_bonds_total"] += 1
            agree["ring_bonds_match"] += int(ring_hits == int(sig[4]))

    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "discipline": DISCIPLINE,
        "selection_rule": "top-3 by T with support>=50, bottom-3 by T with support>=50, most frequent candidate",
        "selected_indices": selected_idx,
        "selected": selected,
        "rdkit_agreement": agree,
        "agreement_note": (
            "BJOC ring flags use biconnected blocks (>=3 edges); RDKit uses its own ring perception. "
            "Small disagreements are definitional, not chemistry errors."
        ),
        "scope_note": (
            "two-bond signatures are not chemical group names; separate (a) signature reuse across "
            "molecules, (b) context effects; higher-order connected-structure discovery is out of scope"
        ),
    }
    bst.write_json(SEMANTICS, payload)
    log(f"semantics: selected {len(selected_idx)} classes; rdkit agreement {agree}")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    frozen = verify_freeze()
    decision = bst.read_json(DECISION)
    artifacts = [
        "PROTOCOL.md",
        "protocol_freeze.json",
        "input_audit.json",
        "discovery_table.json",
        "discovery_folds.json",
        "structure_overlap.json",
        "validation_raw_seed0.npz",
        "validation_raw_seed1.npz",
        "discovery_replay.json",
        "validation_provenance.json",
        "validation_analysis.json",
        "row_metrics.npz",
        "semantics_check.json",
        "final_decision.json",
        "REPORT.md",
    ]
    hashes = {}
    missing = []
    for name in artifacts:
        p = OUT / name
        if p.exists():
            hashes[name] = bst.file_sha256(p)
        else:
            missing.append(name)
    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utc_now(),
        "label": decision["label"],
        "gates": decision["gates"],
        "artifact_sha256": hashes,
        "missing_artifacts": missing,
        "freeze_git_commit": frozen["git_commit"],
        "freeze_ok": True,
        "discipline": DISCIPLINE,
    }
    bst.write_json(FINALIZE, payload)
    if missing:
        log(f"finalize: missing artifacts: {missing}")
        return 1
    log(f"finalize: complete; label={decision['label']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("verify-inputs", cmd_verify_inputs),
        ("freeze", cmd_freeze),
        ("discovery", cmd_discovery),
        ("structure", cmd_structure),
        ("validate", cmd_validate),
        ("analyze", cmd_analyze),
        ("semantics", cmd_semantics),
        ("finalize", cmd_finalize),
    ):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
