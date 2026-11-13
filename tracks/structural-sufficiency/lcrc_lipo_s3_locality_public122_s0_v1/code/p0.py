"""P0: rebuild the provable local signature, verify the alias candidates.

CPU only.  Everything here is structural or uses the frozen A checkpoint; no
target byte defines a signature or a group.

Outputs (all under ``analysis/``):

* ``alias_audit.json``        - groups, isomorphism classes, A forward replication
* ``alias_pairs.csv``         - the five frozen pairs with labels/predictions/splits
* ``feature_separation.json`` - U3 / U2 separation per pair and per group
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import common as C
import data as D
import features as F
import signature as SIG

import mb_raw_artifact as RAW


# ---------------------------------------------------------------------------
# labelled graph isomorphism on the committed raw attributes
# ---------------------------------------------------------------------------


def raw_graph(row: Mapping[str, Any]) -> dict[str, Any]:
    atom_class = np.asarray(row["atom_class"], dtype=np.int64)
    atom_charge = np.asarray(row["atom_charge"], dtype=np.int64)
    atom_h = np.asarray(row["atom_h"], dtype=np.int64)
    ej = np.asarray(row["edge_index"], dtype=np.int64)
    ec = np.asarray(row["edge_class"], dtype=np.int64)
    es = np.asarray(row["edge_stereo"], dtype=np.int64)
    edges: dict[tuple[int, int], tuple[int, int]] = {}
    for k in range(int(ec.size)):
        u, v = int(ej[0, k]), int(ej[1, k])
        edges[(min(u, v), max(u, v))] = (int(ec[k]), int(es[k]))
    return {
        "n_atoms": int(atom_class.size),
        "atom_class": atom_class,
        "atom_charge": atom_charge,
        "atom_h": atom_h,
        "edges": edges,
    }


def to_networkx(g: Mapping[str, Any], *, stereo: bool, node_extra: bool = True):
    import networkx as nx

    graph = nx.Graph()
    for i in range(int(g["n_atoms"])):
        if node_extra:
            label = (int(g["atom_class"][i]), int(g["atom_charge"][i]), int(g["atom_h"][i]))
        else:
            label = (int(g["atom_class"][i]),)
        graph.add_node(i, label=label)
    for (u, v), (ec, es) in g["edges"].items():
        label = (int(ec), int(es)) if stereo else (int(ec),)
        graph.add_edge(int(u), int(v), label=label)
    return graph


def isomorphic(
    ga: Mapping[str, Any],
    gb: Mapping[str, Any],
    *,
    stereo: bool = True,
    node_extra: bool = True,
) -> bool:
    import networkx as nx

    if int(ga["n_atoms"]) != int(gb["n_atoms"]) or len(ga["edges"]) != len(gb["edges"]):
        return False
    A = to_networkx(ga, stereo=stereo, node_extra=node_extra)
    B = to_networkx(gb, stereo=stereo, node_extra=node_extra)
    match = nx.algorithms.isomorphism.GraphMatcher(
        A, B, node_match=lambda x, y: x["label"] == y["label"], edge_match=lambda x, y: x["label"] == y["label"]
    )
    return bool(match.is_isomorphic())


def isomorphism_class(ga: Mapping[str, Any], gb: Mapping[str, Any]) -> str:
    """Explicit labelled-graph decision (WL is never used here)."""
    if isomorphic(ga, gb, stereo=True, node_extra=True):
        return "RAW_ATTRIBUTE_GRAPH_ISOMORPHIC"
    if isomorphic(ga, gb, stereo=False, node_extra=True):
        return "STEREO_LABEL_ONLY_DIFFERENCE"
    if isomorphic(ga, gb, stereo=True, node_extra=False):
        return "ATOM_CLASS_ONLY_IGNORING_CHARGE_H"
    return "CONNECTIVITY_OR_ATTRIBUTE_NONISOMORPHIC"


# ---------------------------------------------------------------------------
# evidence helpers
# ---------------------------------------------------------------------------


def shortest_path(n_atoms: int, bond_index: np.ndarray, s: int, t: int) -> list[int]:
    adj = F.molecule_adjacency(n_atoms, bond_index)
    prev = {int(s): -1}
    frontier = [int(s)]
    while frontier and int(t) not in prev:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    if int(t) not in prev:
        return []
    path = [int(t)]
    while path[-1] != int(s):
        path.append(prev[path[-1]])
    return list(reversed(path))


def vector_bytes(vec: Sequence[float]) -> bytes:
    return np.ascontiguousarray(np.asarray(vec, dtype=np.float32)).tobytes()


def paired_multiset(
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> dict[str, Any]:
    """Compare the multiset of ``(signature digest, structural vector)`` pairs."""
    da = SIG.molecule_digests(view_a)
    db = SIG.molecule_digests(view_b)
    a0 = int(view_a["atom_base"])
    b0 = int(view_b["atom_base"])
    pa = sorted((da[v], vector_bytes(vec_a[a0 + v])) for v in range(len(da)))
    pb = sorted((db[v], vector_bytes(vec_b[b0 + v])) for v in range(len(db)))
    same = bool(pa == pb)
    classes: dict[str, dict[str, list[bytes]]] = {}
    for side, digests, base, vec in (("a", da, a0, vec_a), ("b", db, b0, vec_b)):
        for v, d in enumerate(digests):
            classes.setdefault(d, {"a": [], "b": []})[side].append(vector_bytes(vec[base + v]))
    differing = []
    for d, entry in classes.items():
        if Counter(entry["a"]) != Counter(entry["b"]):
            differing.append(
                {
                    "signature_digest": d,
                    "a_vectors": [list(np.frombuffer(x, dtype=np.float32).tolist()) for x in sorted(entry["a"])],
                    "b_vectors": [list(np.frombuffer(x, dtype=np.float32).tolist()) for x in sorted(entry["b"])],
                }
            )
    return {
        "separated": not same,
        "n_roots_a": len(da),
        "n_roots_b": len(db),
        "n_differing_signature_classes": len(differing),
        "differing_classes": differing,
    }


def _multiset_rows(arr: np.ndarray) -> np.ndarray:
    rows = [tuple(np.asarray(r, dtype=np.float64).tolist()) for r in arr]
    return np.asarray(sorted(rows), dtype=np.float64)


def multiset_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float | None:
    """Order-independent multiset distance: optimal row assignment, max abs diff."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    if a.shape[0] == 0:
        return 0.0
    from scipy.optimize import linear_sum_assignment

    # L-infinity cost per row pair, then the exact minimum-cost assignment
    cost = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].max())


# ---------------------------------------------------------------------------
# A-baseline forward replication
# ---------------------------------------------------------------------------


def load_baseline_per_molecule() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with open(C.BASELINE_PER_MOLECULE, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != "A":
                continue
            out[int(row["row_id"])] = {
                "split": row["split"],
                "y_true": float(row["y_true"]),
                "pred": float(row["pred"]),
                "eval_component": int(row["eval_component"]),
            }
    return out


def load_baseline_model(device):
    import torch

    dataset = D.load_dataset(include_test=True)
    spec = dataset["spec"]
    norm = dataset["train_norm"]
    ckpt = torch.load(C.BASELINE_BEST, map_location="cpu", weights_only=False)
    if int(ckpt["parameters"]) != C.BASE_PARAMS:
        raise RuntimeError(f"baseline checkpoint has {ckpt['parameters']} parameters")
    if str(ckpt["variant"]) != C.VARIANT:
        raise RuntimeError(f"baseline checkpoint variant {ckpt['variant']} != {C.VARIANT}")
    model = C.DM.load_checkpoint_model(spec, norm, C.VARIANT, ckpt["state_dict"], device)
    model.eval()
    return model, ckpt, dataset


def baseline_predictions(device) -> dict[str, Any]:
    model, ckpt, dataset = load_baseline_model(device)
    per_mol = load_baseline_per_molecule()
    scale = float(dataset["target_norm"]["scale"])
    mean = float(dataset["target_norm"]["mean"])
    report: dict[str, Any] = {
        "checkpoint": {
            "path": C.track_rel(C.BASELINE_BEST),
            "sha256": C.sha256_file(C.BASELINE_BEST),
            "state_hash": ckpt.get("state_hash"),
            "best_epoch": int(ckpt["best_epoch"]),
            "best_metric_value": float(ckpt["best_metric_value"]),
            "best_metric_name": ckpt.get("best_metric_name"),
            "parameters": int(ckpt["parameters"]),
            "protocol_id": ckpt.get("protocol_id"),
            "variant": ckpt.get("variant"),
            "init_state_hash": ckpt.get("init_state_hash"),
            "feature_spec_hash": ckpt.get("feature_spec_hash"),
        },
        "splits": {},
        "predictions": {},
        "raw_predictions": {},
        "rows": {},
        "y": {},
        "design_note_predictions": {},
    }
    max_join = 0.0
    for split in C.SPLITS:
        rows = [int(r) for r in dataset["rows"][split]]
        y = np.asarray(dataset[f"y_{split}"], dtype=np.float64)
        pred = D.predict_scores(model, dataset[f"{split}_cache"], device) * scale + mean
        err = pred - y
        joins = [abs(float(p) - float(per_mol[int(r)]["pred"])) for r, p in zip(rows, pred) if int(r) in per_mol]
        if joins:
            max_join = max(max_join, float(np.max(joins)))
        report["splits"][split] = {
            "n": int(len(pred)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "sse": float(np.sum(err ** 2)),
            "row_id_join_max_abs_diff_vs_baseline_per_molecule": float(np.max(joins)) if joins else None,
            "parameters": int(ckpt["parameters"]),
        }
        report["predictions"][split] = {str(r): float(p) for r, p in zip(rows, pred)}
        report["raw_predictions"][split] = pred
        report["rows"][split] = rows
        report["y"][split] = y
    max_note = 0.0
    note = {}
    for row, expected in C.ALIAS_EXPECTED_PRED.items():
        split = next(s for s in C.SPLITS if row in set(report["rows"][s]))
        pos = report["rows"][split].index(row)
        pred = float(report["raw_predictions"][split][pos])
        note[str(row)] = {"pred": pred, "design_note": float(expected), "abs_diff": abs(pred - float(expected))}
        max_note = max(max_note, abs(pred - float(expected)))
    report["design_note_predictions"] = note
    report["design_note_predictions_max_abs_diff"] = float(max_note)
    report["design_note_predictions_tolerance"] = float(C.ALIAS_PRED_TOL)
    report["row_id_join_max_abs_diff"] = float(max_join)
    report["dataset"] = {
        "rows": report["rows"],
        "y": report["y"],
        "target_norm": dict(dataset["target_norm"]),
        "train_norm_fingerprint": C.PREP_DATA.norm_fingerprint(dataset["train_norm"]),
    }
    return report


def baseline_reference(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Small committed provenance record of the reused ordinary-C1 baseline.

    ``runs/`` is git-ignored, so the epoch-1 dropout-RNG trace and the exact
    checkpoint identity of the reused A run are recorded here once, with the
    sha256 of every local artefact they came from.
    """
    ref: dict[str, Any] = {
        "schema": "lcrc-lipo-s3-locality-baseline-reference-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "baseline_round": C.track_rel(C.BASELINE_ROUND),
        "checkpoint": {
            **baseline["checkpoint"],
            "per_molecule_csv_sha256": C.sha256_file(C.BASELINE_PER_MOLECULE),
            "per_molecule_csv": C.track_rel(C.BASELINE_PER_MOLECULE),
        },
        "replication": {
            "splits": baseline["splits"],
            "row_id_join_max_abs_diff": baseline["row_id_join_max_abs_diff"],
            "design_note_predictions_max_abs_diff": baseline["design_note_predictions_max_abs_diff"],
            "design_note_tolerance": baseline["design_note_predictions_tolerance"],
            "device": "cpu (local P0 replication; GPU values in the baseline run)",
        },
        "gpu_hours": 0.0,
    }
    if C.BASELINE_TRAIN_JSON.exists():
        tj = C.read_json(C.BASELINE_TRAIN_JSON)
        ref["rng"] = {
            "initial_hash_after_reset": tj.get("rng", {}).get("initial_hash_after_reset"),
            "epoch1_step_hashes": tj.get("rng", {}).get("epoch1_step_hashes"),
            "source": C.track_rel(C.BASELINE_TRAIN_JSON),
            "source_sha256": C.sha256_file(C.BASELINE_TRAIN_JSON),
        }
        ref["gpu_hours"] = float(tj.get("gpu", {}).get("gpu_hours_charged", 0.0))
        ref["baseline_metrics"] = {
            "best": tj.get("metrics", {}).get("best"),
            "initial": tj.get("metrics", {}).get("initial"),
            "target_norm": tj.get("data", {}).get("target_norm"),
            "train_norm_fingerprint": tj.get("data", {}).get("train_norm_fingerprint"),
            "train_cache_fingerprint": tj.get("data", {}).get("train_cache_fingerprint"),
            "dropout_calls_in_run": tj.get("model", {}).get("dropout_calls_in_run"),
        }
    return ref


# ---------------------------------------------------------------------------
# group enumeration
# ---------------------------------------------------------------------------


def enumerate_groups(caches: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_mol: list[dict[str, Any]] = []
    digest_set: set[str] = set()
    for split in C.SPLITS:
        cache = caches[split]
        rows = np.asarray(cache["rows"], dtype=np.int64)
        for mol in range(int(cache["n_molecules"])):
            view = SIG.molecule_view(cache, mol)
            digests = SIG.molecule_digests(view)
            digest_set.update(digests)
            per_mol.append(
                {
                    "split": split,
                    "row": int(rows[mol]),
                    "cache_index": int(mol),
                    "n_atoms": int(view["n_atoms"]),
                    "multiset": SIG.multiset_digest(digests),
                    "global": SIG.global_digest(view),
                    "root_digests": digests,
                }
            )
    by_local: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_full: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in per_mol:
        by_local[rec["multiset"]].append(rec)
        by_full[f"{rec['multiset']}|{rec['global']}"].append(rec)
    local_groups = sorted(
        (sorted(g, key=lambda r: r["row"]) for g in by_local.values() if len(g) >= 2), key=lambda g: g[0]["row"]
    )
    full_groups = sorted(
        (sorted(g, key=lambda r: r["row"]) for g in by_full.values() if len(g) >= 2), key=lambda g: g[0]["row"]
    )
    return {
        "n_molecules": len(per_mol),
        "n_atoms": int(sum(r["n_atoms"] for r in per_mol)),
        "n_distinct_local_signatures_per_molecule": len({r["multiset"] for r in per_mol}),
        "n_distinct_root_signatures": len(digest_set),
        "n_local_groups": len(local_groups),
        "n_local_group_molecules": int(sum(len(g) for g in local_groups)),
        "n_full_groups": len(full_groups),
        "n_full_group_molecules": int(sum(len(g) for g in full_groups)),
        "local_groups": local_groups,
        "full_groups": full_groups,
        "per_molecule": per_mol,
    }


def view_of(caches: Mapping[str, Mapping[str, Any]], rec: Mapping[str, Any]) -> dict[str, Any]:
    return SIG.molecule_view(caches[rec["split"]], int(rec["cache_index"]))


def group_report(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    caches: Mapping[str, Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    y_by_row: Mapping[int, float],
) -> dict[str, Any]:
    """Exact signature verification + labelled-isomorphism classification.

    Isomorphism is an equivalence relation, so classifying every member against
    the group representative is exact and cheaper than all pairs.
    """
    classes: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    n_sig_verified = 0
    total_sse = 0.0
    for gi, group in enumerate(groups):
        view_rep = view_of(caches, group[0])
        verifs = [SIG.compare_molecule_signatures(view_rep, view_of(caches, other)) for other in group[1:]]
        sig_ok = bool(all(v["multiset_equal"] for v in verifs))
        n_sig_verified += int(sig_ok)
        base_graph = raw_graph(raw_rows[int(group[0]["row"])])
        class_against_rep = {
            str(other["row"]): isomorphism_class(base_graph, raw_graph(raw_rows[int(other["row"])]))
            for other in group[1:]
        }
        for cls in class_against_rep.values():
            classes[cls] += 1
        ys = np.asarray([float(y_by_row[int(r["row"])]) for r in group], dtype=np.float64)
        sse = float(np.sum((ys - ys.mean()) ** 2))
        total_sse += sse
        records.append(
            {
                "group_id": int(gi),
                "label": label,
                "rows": [int(r["row"]) for r in group],
                "splits": [str(r["split"]) for r in group],
                "n_atoms": [int(r["n_atoms"]) for r in group],
                "signature_multiset_verified_equal": sig_ok,
                "per_block_all_equal": bool(all(v["per_block_all_equal"] for v in verifs)),
                "isomorphism_class_vs_representative": class_against_rep,
                "empirical_min_sse": sse,
                "n_rows": len(group),
            }
        )
    return {
        "label": label,
        "n_groups": len(groups),
        "n_molecules_in_groups": int(sum(len(g) for g in groups)),
        "n_molecules_compared_to_representative": int(sum(len(g) - 1 for g in groups)),
        "n_groups_signature_verified": int(n_sig_verified),
        "isomorphism_class_counts": dict(classes),
        "empirical_min_sse_total": total_sse,
        "empirical_min_sse_note": (
            "sum over groups of sum_i (y_i - mean_group(y))^2; describes the fit constraint implied by "
            "identical inputs on this sample, not irreducible property noise"
        ),
        "groups": records,
    }


# ---------------------------------------------------------------------------
# the five frozen pairs
# ---------------------------------------------------------------------------


def alias_pair_audit(
    *,
    caches: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    per_mol: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    rows_of = baseline["rows"]
    split_by_row = {int(r): s for s in C.SPLITS for r in rows_of[s]}
    y_by_row = {int(r): float(y) for s in C.SPLITS for r, y in zip(rows_of[s], baseline["y"][s])}
    pred_by_row = {
        int(r): float(p) for s in C.SPLITS for r, p in baseline["predictions"][s].items()
    }
    s3 = {s: np.asarray(feats[f"{s}_s3"], dtype=np.float32) for s in C.SPLITS}
    s2 = {s: np.asarray(feats[f"{s}_s2"], dtype=np.float32) for s in C.SPLITS}
    records: list[dict[str, Any]] = []
    separation_pairs: list[dict[str, Any]] = []
    for ra, rb in C.ALIAS_PAIRS:
        sa, sb = split_by_row[ra], split_by_row[rb]
        va = SIG.molecule_view(caches[sa], rows_of[sa].index(ra))
        vb = SIG.molecule_view(caches[sb], rows_of[sb].index(rb))
        sig = SIG.compare_molecule_signatures(va, vb)
        glob_equal = SIG.global_bytes(va) == SIG.global_bytes(vb)
        iso = isomorphism_class(raw_graph(raw_rows[ra]), raw_graph(raw_rows[rb]))
        u3_sep = paired_multiset(va, vb, s3[sa], s3[sb])
        u2_sep = paired_multiset(va, vb, s2[sa], s2[sb])
        pa, pb = float(pred_by_row[ra]), float(pred_by_row[rb])
        records.append(
            {
                "row_a": ra,
                "row_b": rb,
                "split_a": sa,
                "split_b": sb,
                "y_a": float(y_by_row[ra]),
                "y_b": float(y_by_row[rb]),
                "y_diff": float(y_by_row[rb] - y_by_row[ra]),
                "pred_a": pa,
                "pred_b": pb,
                "pred_diff": pb - pa,
                "pred_abs_diff": abs(pb - pa),
                "pred_abs_diff_le_2e5": bool(abs(pb - pa) <= C.ALIAS_PRED_TOL),
                "design_note_pred_diff": float(C.ALIAS_EXPECTED_PRED[rb] - C.ALIAS_EXPECTED_PRED[ra]),
                "signature_multiset_equal": bool(sig["multiset_equal"]),
                "signature_digest_multiset_equal": bool(sig["digest_multiset_equal"]),
                "signature_verified_root_pairs": int(sig["n_verified_root_pairs"]),
                "global42_equal": bool(glob_equal),
                "atom_count_equal": int(va["n_atoms"]) == int(vb["n_atoms"]),
                "isomorphism_class": iso,
                "raw_attribute_graph_non_isomorphic": iso != "RAW_ATTRIBUTE_GRAPH_ISOMORPHIC",
                "u3_separated": bool(u3_sep["separated"]),
                "u2_separated": bool(u2_sep["separated"]),
                "u3_n_differing_signature_classes": int(u3_sep["n_differing_signature_classes"]),
                "u2_n_differing_signature_classes": int(u2_sep["n_differing_signature_classes"]),
                "baseline_eval_component_a": int(per_mol[ra]["eval_component"]),
                "baseline_eval_component_b": int(per_mol[rb]["eval_component"]),
            }
        )
        separation_pairs.append(
            {
                "row_a": ra,
                "row_b": rb,
                "u3": u3_sep,
                "u2": u2_sep,
                "expected_u3_separated": (ra, rb) in C.ALIAS_EXPECTED_SEPARATED,
            }
        )
    summary = {
        "n_pairs": len(records),
        "n_signature_multiset_equal": int(sum(r["signature_multiset_equal"] for r in records)),
        "n_global42_equal": int(sum(r["global42_equal"] for r in records)),
        "n_prediction_equal_within_tol": int(sum(r["pred_abs_diff_le_2e5"] for r in records)),
        "n_non_isomorphic_raw_graph": int(sum(r["raw_attribute_graph_non_isomorphic"] for r in records)),
        "n_u3_separated": int(sum(r["u3_separated"] for r in records)),
        "n_u2_separated": int(sum(r["u2_separated"] for r in records)),
        "expected_u3_separated": len(C.ALIAS_EXPECTED_SEPARATED),
        "expected_u2_separated": 0,
        "u3_matches_design_expectation": bool(
            all(
                (r["u3_separated"] == ((r["row_a"], r["row_b"]) in C.ALIAS_EXPECTED_SEPARATED))
                for r in records
            )
        ),
    }
    return {"pairs": records, "separation_pairs": separation_pairs, "summary": summary}


# ---------------------------------------------------------------------------
# forward-level identity: same signature => same local input and same h
# ---------------------------------------------------------------------------


def forward_identity_checks(
    caches: Mapping[str, Mapping[str, Any]],
    feats_by_split: Mapping[str, Mapping[str, Any]],
    dataset: Mapping[str, Any],
    device,
) -> dict[str, Any]:
    import torch

    model, _ckpt, _dataset = load_baseline_model(device)
    rows_of = dataset["rows"]
    split_by_row = {int(r): s for s in C.SPLITS for r in rows_of[s]}
    out: dict[str, Any] = {"pairs": [], "sample": {}}
    with torch.no_grad():
        for ra, rb in C.ALIAS_PAIRS:
            sa, sb = split_by_row[ra], split_by_row[rb]
            ba = D.to_device(D.collate(caches[sa], [rows_of[sa].index(ra)], feats_by_split[sa]), device)
            bb = D.to_device(D.collate(caches[sb], [rows_of[sb].index(rb)], feats_by_split[sb]), device)
            ya, aux_a = model(ba, return_aux=True)
            yb, aux_b = model(bb, return_aux=True)
            xa = aux_a["local_input"].detach().cpu().numpy()
            xb = aux_b["local_input"].detach().cpu().numpy()
            ha = aux_a["h"].detach().cpu().numpy()
            hb = aux_b["h"].detach().cpu().numpy()
            out["pairs"].append(
                {
                    "row_a": ra,
                    "row_b": rb,
                    "n_atoms_equal": int(xa.shape[0]) == int(xb.shape[0]),
                    "local_input_multiset_max_abs_diff": multiset_max_abs_diff(xa, xb),
                    "h_multiset_max_abs_diff": multiset_max_abs_diff(ha, hb),
                    "eval_pred_diff": float(ya.item() - yb.item()),
                    "local_shape": list(xa.shape),
                }
            )
        # cross-molecule signature-matched root sample
        root_groups: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for split in C.SPLITS:
            cache = caches[split]
            for mol in range(int(cache["n_molecules"])):
                view = SIG.molecule_view(cache, mol)
                for v in range(int(view["n_atoms"])):
                    root_groups[SIG.root_digest_hex(view, v)].append((split, mol, v))
        candidates = [k for k, vals in root_groups.items() if len({x[1] for x in vals}) > 1]
        rng = np.random.default_rng(20261113)
        rng.shuffle(candidates)
        max_x = 0.0
        max_h = 0.0
        n_checked = 0
        for d in candidates[:1000]:
            vals = root_groups[d]
            i, j = rng.choice(len(vals), size=2, replace=False)
            (split_a, mol_a, root_a), (split_b, mol_b, root_b) = vals[int(i)], vals[int(j)]
            ba = D.to_device(D.collate(caches[split_a], [mol_a], feats_by_split[split_a]), device)
            bb = D.to_device(D.collate(caches[split_b], [mol_b], feats_by_split[split_b]), device)
            _, aux_a = model(ba, return_aux=True)
            _, aux_b = model(bb, return_aux=True)
            xa = aux_a["local_input"].detach().cpu().numpy()[root_a]
            xb = aux_b["local_input"].detach().cpu().numpy()[root_b]
            ha = aux_a["h"].detach().cpu().numpy()[root_a]
            hb = aux_b["h"].detach().cpu().numpy()[root_b]
            max_x = max(max_x, float(np.max(np.abs(xa - xb))))
            max_h = max(max_h, float(np.max(np.abs(ha - hb))))
            n_checked += 1
        out["sample"] = {
            "n_distinct_root_signatures": len(root_groups),
            "n_cross_molecule_signatures": len(candidates),
            "n_root_pairs_checked": int(n_checked),
            "max_abs_local_input_diff": float(max_x),
            "max_abs_h_diff": float(max_h),
            "tolerance": float(C.ALIAS_PRED_TOL),
            "passed": bool(max_x <= C.ALIAS_PRED_TOL and max_h <= C.ALIAS_PRED_TOL),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(device_name: str = "cpu") -> dict[str, Any]:
    import torch

    device = torch.device(device_name)
    source = C.verify_source()
    C.log("[p0] source verified")

    caches = {split: D.load_cache(split, require=False) for split in C.SPLITS}
    feats = F.load_features()
    feats_by_split = {split: F.split_features(feats, split, caches[split]) for split in C.SPLITS}

    raw = RAW.load(verify=True)
    raw_rows = raw["rows"]["Lipophilicity"]

    baseline = baseline_predictions(device)
    per_mol = load_baseline_per_molecule()
    C.log(
        f"[p0] A replicated epoch={baseline['checkpoint']['best_epoch']} "
        f"valid={baseline['splits']['valid']['rmse']:.9f} test={baseline['splits']['test']['rmse']:.9f} "
        f"join_max={baseline['row_id_join_max_abs_diff']:.2e} "
        f"design_note_max={baseline['design_note_predictions_max_abs_diff']:.2e}"
    )

    groups = enumerate_groups(caches)
    C.log(
        f"[p0] local-full-signature groups={groups['n_local_groups']} full-input groups={groups['n_full_groups']} "
        f"root signatures={groups['n_distinct_root_signatures']}"
    )

    y_by_row = {int(r): float(y) for s in C.SPLITS for r, y in zip(baseline["rows"][s], baseline["y"][s])}
    full_report = group_report(
        groups["full_groups"], caches, raw_rows, label="same_local_signature_multiset_and_global42", y_by_row=y_by_row
    )
    local_report = group_report(
        groups["local_groups"], caches, raw_rows, label="same_local_signature_multiset", y_by_row=y_by_row
    )
    alias = alias_pair_audit(
        caches=caches, feats=feats, raw_rows=raw_rows, baseline=baseline, per_mol=per_mol
    )
    forward = forward_identity_checks(caches, feats_by_split, baseline["dataset"], device)

    sep_summary = {"u3_separated_groups": 0, "u2_separated_groups": 0, "n_groups": 0}
    for group in groups["full_groups"]:
        sep_summary["n_groups"] += 1
        base = group[0]
        view_base = view_of(caches, base)
        if any(
            paired_multiset(
                view_base, view_of(caches, other), feats_by_split[base["split"]]["s3"], feats_by_split[other["split"]]["s3"]
            )["separated"]
            for other in group[1:]
        ):
            sep_summary["u3_separated_groups"] += 1
        if any(
            paired_multiset(
                view_base, view_of(caches, other), feats_by_split[base["split"]]["s2"], feats_by_split[other["split"]]["s2"]
            )["separated"]
            for other in group[1:]
        ):
            sep_summary["u2_separated_groups"] += 1

    def non_iso_groups(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            g
            for g in report["groups"]
            if any(v != "RAW_ATTRIBUTE_GRAPH_ISOMORPHIC" for v in g["isomorphism_class_vs_representative"].values())
        ]

    local_non_iso = non_iso_groups(local_report)
    full_non_iso = non_iso_groups(full_report)
    full_keys = {tuple(g["rows"]) for g in full_non_iso}
    local_only = [g for g in local_non_iso if tuple(g["rows"]) not in full_keys]

    audit = {
        "schema": "lcrc-lipo-s3-locality-alias-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "source": source,
        "feature_spec_hash": C.FEATURE_SPEC_HASH,
        "signature_definition": SIG.signature_field_audit(),
        "baseline_A": {k: v for k, v in baseline.items() if k != "dataset"},
        "groups": {
            "local_signature_only": {
                "n_groups": local_report["n_groups"],
                "n_molecules_in_groups": local_report["n_molecules_in_groups"],
                "n_groups_signature_verified": local_report["n_groups_signature_verified"],
                "isomorphism_class_counts": local_report["isomorphism_class_counts"],
                "empirical_min_sse_total": local_report["empirical_min_sse_total"],
            },
            "local_plus_global42": {
                "n_groups": full_report["n_groups"],
                "n_molecules_in_groups": full_report["n_molecules_in_groups"],
                "n_groups_signature_verified": full_report["n_groups_signature_verified"],
                "isomorphism_class_counts": full_report["isomorphism_class_counts"],
                "empirical_min_sse_total": full_report["empirical_min_sse_total"],
                "u3_u2_group_separation": sep_summary,
            },
            "enumeration": {
                "n_molecules": groups["n_molecules"],
                "n_atoms": groups["n_atoms"],
                "n_distinct_local_signatures_per_molecule": groups["n_distinct_local_signatures_per_molecule"],
                "n_distinct_root_signatures": groups["n_distinct_root_signatures"],
                "n_local_groups": groups["n_local_groups"],
                "n_full_groups": groups["n_full_groups"],
            },
            "non_isomorphic_alias_groups": {
                "local_signature_only": [
                    {"rows": g["rows"], "splits": g["splits"], "empirical_min_sse": g["empirical_min_sse"]}
                    for g in local_non_iso
                ],
                "local_plus_global42": [
                    {"rows": g["rows"], "splits": g["splits"], "empirical_min_sse": g["empirical_min_sse"]}
                    for g in full_non_iso
                ],
                "local_only_but_separated_by_global42": [
                    {
                        "rows": g["rows"],
                        "splits": g["splits"],
                        "empirical_min_sse": g["empirical_min_sse"],
                        "note": "same local signature multiset but different global42, therefore NOT a full-input alias",
                    }
                    for g in local_only
                ],
                "n_additional_full_input_alias_pairs_beyond_the_five": int(len(full_non_iso) - len(C.ALIAS_PAIRS)),
            },
            "detail": {
                "local_groups": local_report["groups"],
                "full_groups": full_report["groups"],
            },
        },
        "alias_pairs": alias["pairs"],
        "alias_pair_summary": alias["summary"],
        "forward_identity": forward,
        "cache_manifest": C.read_json(C.CACHE / "manifest.json"),        "batch_order_manifest": C.read_json(C.ANALYSIS / "batch_order_manifest.json"),
        "features_manifest": {"norm_s3": feats["norm_s3"], "norm_s2": feats["norm_s2"]},
    }
    C.write_json(C.ANALYSIS / "alias_audit.json", audit)
    C.write_json(C.ANALYSIS / "baseline_reference.json", baseline_reference(baseline))

    with open(C.ANALYSIS / "alias_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [
            "row_a", "row_b", "split_a", "split_b", "y_a", "y_b", "y_diff", "pred_a", "pred_b",
            "pred_diff", "pred_abs_diff", "signature_multiset_equal", "global42_equal",
            "isomorphism_class", "u3_separated", "u2_separated", "baseline_eval_component_a",
            "baseline_eval_component_b",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in alias["pairs"]:
            writer.writerow(rec)

    C.write_json(
        C.ANALYSIS / "feature_separation.json",
        {
            "schema": "lcrc-lipo-s3-locality-feature-separation-v1",
            "round_id": C.ROUND_ID,
            "definition": {
                "u3": "count of distinct atoms at exact shortest-path distance 3, per atom type",
                "u2": "count of S2 path occurrences rooted at v by path_end63 // 5 (end atom type)",
                "separated": "the multiset of (local signature digest, structural vector) pairs differs",
                "grouping": "roots are compared inside the identical local-signature classes",
            },
            "pairs": alias["separation_pairs"],
            "summary": alias["summary"],
            "full_groups": sep_summary,
            "norm_s3": feats["norm_s3"],
            "norm_s2": feats["norm_s2"],
        },
    )
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0 alias audit (CPU)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    audit = run(args.device)
    s = audit["alias_pair_summary"]
    print(
        "P0: signature_equal={n_signature_multiset_equal}/{n_pairs} "
        "global_equal={n_global42_equal}/{n_pairs} pred_equal={n_prediction_equal_within_tol}/{n_pairs} "
        "raw_noniso={n_non_isomorphic_raw_graph}/{n_pairs} u3_sep={n_u3_separated}/{n_pairs} "
        "u2_sep={n_u2_separated}/{n_pairs}".format(**s)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
