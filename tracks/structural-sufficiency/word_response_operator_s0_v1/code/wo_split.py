"""Pre-registered Fit/H label-holdout construction for the word-response-operator s0.

Only structure is read here: the frozen edit-pair catalogue (``pairs.csv``) and
the official split identity of the raw cache.  No lipophilicity label is touched,
so the holdout decision cannot depend on it.

Rule (frozen before any metric)
-------------------------------
* Train-Train pairs of the fixed catalogue become graph edges over official row
  ids.  Connected components of *that* graph (not the old catalogue numbering)
  are the components.
* component key = sorted official row ids joined by ``,``;
  digest = SHA256("word-op-s0|" + key), component is a candidate when digest % 5 == 0.
* H = every molecule of the candidate components; Fit = the remaining official
  Train molecules (isolated molecules stay in Fit).
* A transform is *evaluable* on the holdout when it appears in at least 2
  distinct Fit pair components, at least one H-internal pair exists, and the
  pair's exact retained-core key never appears on a Fit pair.
* H is enabled only when 50 <= |H| <= 840, evaluable pairs >= 40, H components
  with an evaluable pair >= 10, distinct evaluable transforms >= 8.
  Otherwise H is disabled once and Fit becomes the whole official Train
  (``TRANSFER_SUPPORT_INSUFFICIENT``).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

import wo_common as C

SPLITS = C.SPLITS


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        parent = self.parent
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)
            self.parent[min(ra, rb)] = min(ra, rb)


def read_pairs() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(C.PAIRS_CSV.open(newline="", encoding="utf-8")))
    out = []
    for row in rows:
        out.append({
            "pair_id": str(row["pair_id"]),
            "row_a": int(row["row_a"]),
            "row_b": int(row["row_b"]),
            "row_lo": int(row["row_lo"]),
            "row_hi": int(row["row_hi"]),
            "split_a": str(row["split_a"]),
            "split_b": str(row["split_b"]),
            "core_id": str(row["core_id"]),
            "core_uid": int(row["core_uid"]),
            "s_key_a": str(row["s_key_a"]),
            "s_key_b": str(row["s_key_b"]),
            "transform_id": str(row["transform_id"]),
        })
    return out


def official_split(payload: Mapping[str, Any]) -> np.ndarray:
    out = np.empty(len(payload["smiles"]), dtype=object)
    for split in SPLITS:
        out[np.asarray(payload[f"{split}_idx"], dtype=np.int64)] = split
    return out


def pair_direction_ok(pairs) -> bool:
    return all(p["s_key_a"] < p["s_key_b"] for p in pairs)


def select_split() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = C.load_config()
    rules = cfg["split"]
    payload = C.dataset_cache(cfg["dataset"])
    split_of = official_split(payload)
    n_rows = len(payload["smiles"])
    train_rows = sorted(int(r) for r in np.asarray(payload["train_idx"]))
    valid_rows = sorted(int(r) for r in np.asarray(payload["valid_idx"]))
    pairs = read_pairs()

    # stored split columns must agree with the official split (structure only)
    mismatch = [p["pair_id"] for p in pairs
                if (p["split_a"], p["split_b"]) != (split_of[p["row_a"]], split_of[p["row_b"]])]
    if mismatch:
        raise ValueError(f"pairs.csv split columns disagree with the official split: {mismatch[:3]}")
    if not pair_direction_ok(pairs):
        raise ValueError("pairs.csv is not in canonical s_key direction")

    train_train = [p for p in pairs if p["split_a"] == "train" and p["split_b"] == "train"]
    uf = UnionFind()
    for row in train_rows:
        uf.find(row)
    for p in train_train:
        uf.union(p["row_a"], p["row_b"])
    groups: dict[int, list[int]] = defaultdict(list)
    for row in train_rows:
        groups[uf.find(row)].append(row)

    tag = rules["hash_tag"]
    comps = []
    for root, rows in groups.items():
        rows = sorted(int(r) for r in rows)
        key = ",".join(str(r) for r in rows)
        digest = C.sha256_bytes((tag + key).encode("utf-8"))
        value = int(digest, 16)
        comps.append({
            "component_key": key,
            "component_hash": digest,
            "hash_remainder": int(value % int(rules["hash_modulus"])),
            "rows": rows,
            "n_molecules": len(rows),
        })
    comps.sort(key=lambda item: item["rows"][0])
    h_comps = [c for c in comps if c["hash_remainder"] == int(rules["hash_remainder"])]
    h_rows = sorted({int(r) for c in h_comps for r in c["rows"]})
    fit_rows = sorted(set(train_rows) - set(h_rows))

    # Fit pair components and transform support (structure only)
    fit_set = set(fit_rows)
    fit_pairs = [p for p in pairs if p["row_lo"] in fit_set and p["row_hi"] in fit_set]
    h_pairs = [p for p in pairs if p["row_lo"] in set(h_rows) and p["row_hi"] in set(h_rows)]
    fuf = UnionFind()
    for row in fit_rows:
        fuf.find(row)
    for p in fit_pairs:
        fuf.union(p["row_a"], p["row_b"])
    fit_comp = {row: fuf.find(row) for row in fit_rows}
    transform_fit_comps: dict[str, set[int]] = defaultdict(set)
    for p in fit_pairs:
        transform_fit_comps[p["transform_id"]].add(fit_comp[p["row_lo"]])
    fit_core_keys = {p["core_id"] for p in fit_pairs}
    transform_n_fit_comps = {t: len(v) for t, v in transform_fit_comps.items()}
    supported = {t for t, n in transform_n_fit_comps.items()
                 if n >= int(rules["min_fit_transform_components"])}

    h_rows_set = set(h_rows)
    evaluable = []
    for p in h_pairs:
        if p["transform_id"] not in supported:
            continue
        if p["core_id"] in fit_core_keys:
            continue
        evaluable.append(p)
    eval_transforms = sorted({p["transform_id"] for p in evaluable})
    h_comp_of: dict[int, str] = {}
    for c in h_comps:
        for row in c["rows"]:
            h_comp_of[int(row)] = c["component_key"]
    eval_comps = sorted({h_comp_of[p["row_lo"]] for p in evaluable})

    conditions = {
        "n_h_molecules_in_range": bool(int(rules["h_min_molecules"]) <= len(h_rows)
                                      <= int(rules["h_max_molecules"])),
        "n_evaluable_pairs_ok": len(evaluable) >= int(rules["min_evaluable_pairs"]),
        "n_h_components_ok": len(eval_comps) >= int(rules["min_h_components"]),
        "n_transforms_ok": len(eval_transforms) >= int(rules["min_transforms"]),
    }
    h_enabled = all(conditions.values())
    flag = "H_ENABLED" if h_enabled else "TRANSFER_SUPPORT_INSUFFICIENT"
    if not h_enabled:
        fit_rows = train_rows
        h_rows = []
        evaluable = []
        eval_transforms = []
        eval_comps = []

    record = {
        "protocol_id": cfg["protocol_id"],
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "pairs_csv": str(C.PAIRS_CSV.relative_to(C.REPO_ROOT)),
        "pairs_csv_sha256": C.sha256_file(C.PAIRS_CSV),
        "official_cache_sha256": payload["cache_sha256"],
        "n_rows": n_rows,
        "n_train_pairs_total": len(pairs),
        "n_train_train_pairs": len(train_train),
        "n_train_components": len(comps),
        "n_h_candidate_components": len(h_comps),
        "n_h_molecules": len(h_rows),
        "n_fit_molecules": len(fit_rows),
        "n_fit_pairs": len(fit_pairs),
        "n_h_internal_pairs": len(h_pairs),
        "n_fit_transforms": len(transform_fit_comps),
        "n_fit_transforms_with_2_components": len(supported),
        "n_fit_core_keys": len(fit_core_keys),
        "n_evaluable_pairs": len(evaluable),
        "n_evaluable_h_components": len(eval_comps),
        "n_evaluable_transforms": len(eval_transforms),
        "conditions": conditions,
        "flag": flag,
        "h_enabled": bool(h_enabled),
        "labels_read": False,
        "is_scaffold_holdout": False,
        "fit_row_ids": [int(r) for r in fit_rows],
        "h_row_ids": [int(r) for r in h_rows],
        "valid_row_ids": [int(r) for r in valid_rows],
        "evaluable_pairs": [
            {
                "pair_id": p["pair_id"],
                "row_a": p["row_a"],
                "row_b": p["row_b"],
                "s_key_a": p["s_key_a"],
                "s_key_b": p["s_key_b"],
                "transform_id": p["transform_id"],
                "core_id": p["core_id"],
                "h_component_key": h_comp_of[p["row_lo"]],
                "fit_transform_components": int(transform_n_fit_comps[p["transform_id"]]),
            }
            for p in evaluable
        ],
        "components": [
            {
                "component_key": c["component_key"],
                "component_hash": c["component_hash"],
                "hash_remainder": c["hash_remainder"],
                "n_molecules": c["n_molecules"],
                "selected": bool(c["hash_remainder"] == int(rules["hash_remainder"])),
            }
            for c in comps
        ],
    }
    record["split_sha256"] = C.sha256_json({
        "fit": record["fit_row_ids"],
        "h": record["h_row_ids"],
        "valid": record["valid_row_ids"],
        "flag": record["flag"],
    })
    stats = {
        "n_h_molecules": len(h_rows),
        "n_fit_molecules": len(fit_rows),
        "n_evaluable_pairs": len(evaluable),
        "n_evaluable_h_components": len(eval_comps),
        "n_evaluable_transforms": len(eval_transforms),
        "conditions": conditions,
        "flag": flag,
    }
    return record, stats


def cmd_select(args) -> None:
    record, stats = select_split()
    path = C.analysis_dir() / "split_freeze.json"
    C.write_json(path, record)
    print(C.jsonable({"path": str(path), **stats}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").set_defaults(func=cmd_select)
    ns = parser.parse_args()
    ns.func(ns)
