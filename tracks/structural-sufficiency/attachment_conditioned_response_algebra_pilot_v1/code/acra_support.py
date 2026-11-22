"""Local structural support over *all* directed connection events (11.1).

Definitions (frozen in PROTOCOL.md):

* identity of a local ball = exact coloured canonical SMILES of the rooted
  induced subgraph (all raw atom/bond attributes, root marked);
* combination identity = (S identity, C identity, raw 3-column cut-bond
  attributes);
* "seen" is defined **only** by Train events; Valid reuse is what is measured.

Test structural statistics are not reported.  CPU only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K


def _split_of_molecules(raw: Mapping[str, Any], pack: Mapping[str, Any]) -> np.ndarray:
    n = int(pack["n_molecules"])
    split = np.full(n, 2, np.uint8)
    for name, value in (("train", 0), ("valid", 1)):
        rows = D.official_splits(raw)[name]
        split[np.asarray(rows, np.int64)] = value
    order = np.asarray(pack["row_ids"], np.int64)
    return split[order]


def _histogram(counts: np.ndarray) -> dict[str, int]:
    edges = [(1, 1), (2, 4), (5, 9), (10, 49), (50, 999), (1000, 10 ** 18)]
    out: dict[str, int] = {}
    for lo, hi in edges:
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 18 else f"{lo}+")
        out[label] = int(np.sum((counts >= lo) & (counts <= hi)))
    return out


def _support_block(key_ids: np.ndarray, train_mask: np.ndarray, valid_mask: np.ndarray,
                   n_keys: int, name: str, keys: np.ndarray | None = None) -> dict[str, Any]:
    train_counts = np.bincount(key_ids[train_mask], minlength=n_keys)
    train_total = int(train_mask.sum())
    valid_ids = key_ids[valid_mask]
    seen = train_counts[valid_ids] > 0
    out: dict[str, Any] = {
        "name": name,
        "n_keys_total": int(n_keys),
        "n_train_events": train_total,
        "n_train_unique_keys": int(np.count_nonzero(train_counts)),
        "train_repeat_rate": float(1.0 - np.count_nonzero(train_counts) / max(1, train_total)),
        "train_key_frequency_histogram": _histogram(train_counts[train_counts > 0]),
        "n_valid_events": int(valid_ids.size),
        "n_valid_unique_keys": int(np.unique(valid_ids).size),
        "valid_seen_rate": float(np.mean(seen)) if valid_ids.size else 0.0,
        "valid_seen_event_counts": int(np.sum(seen)),
        "valid_singleton_key_fraction": float(np.mean(train_counts[valid_ids] == 1))
        if valid_ids.size else 0.0,
        "valid_median_train_support_of_key": float(np.median(train_counts[valid_ids]))
        if valid_ids.size else 0.0,
    }
    if keys is not None and train_counts.size:
        order = np.argsort(-train_counts)[:8]
        out["top_train_keys"] = [
            {"key": str(keys[i])[:180], "train_count": int(train_counts[i])}
            for i in order if train_counts[i] > 0
        ]
    return out


def _stratum(key_ids_s: np.ndarray, key_ids_c: np.ndarray, key_ids_combo: np.ndarray,
             train_mask: np.ndarray, valid_mask: np.ndarray, mask_extra: np.ndarray,
             n_s: int, n_c: int, n_combo: int, name: str) -> dict[str, Any]:
    tr = train_mask & mask_extra
    va = valid_mask & mask_extra
    return {
        "name": name,
        "n_train_events": int(tr.sum()),
        "n_valid_events": int(va.sum()),
        "S": _support_block(key_ids_s, tr, va, n_s, "S"),
        "C": _support_block(key_ids_c, tr, va, n_c, "C"),
        "combo": _support_block(key_ids_combo, tr, va, n_combo, "combo"),
    }


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    pack = K.ensure_pack(cfg, log=log)
    raw = D.load_raw(cfg)
    mol_split = _split_of_molecules(raw, pack)
    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    split = mol_split[ev_mol]
    train_mask = split == 0
    valid_mask = split == 1
    test_mask = split == 2

    s_ids = np.asarray(pack["ev_key_s"], np.int64)
    c_ids = np.asarray(pack["ev_key_c"], np.int64)
    combo_ids = np.asarray(pack["ev_combo"], np.int64)
    n_s = int(pack["key_strings"].size)
    n_c = n_s
    n_combo = int(pack["combo_s"].size)
    bridge = np.asarray(pack["ev_bridge"], np.uint8) == 1
    ring = np.asarray(pack["ev_in_ring"], np.uint8) == 1

    block_overall = {
        "definition": {
            "S_identity": "exact coloured canonical SMILES of the rooted radius-1 ball in G - bond",
            "C_identity": "exact coloured canonical SMILES of the rooted radius-2 ball in G - bond",
            "combo_identity": "(S identity, C identity, raw 3-column cut bond attributes)",
            "seen": "key occurs in at least one Train directed event",
            "note": "in a simple undirected graph an edge is a bridge iff it is in no ring, "
                    "so the bridge and ring strata coincide exactly; both are reported",
        },
        "counts": {
            "n_molecules": int(pack["n_molecules"]),
            "n_directed_events": int(s_ids.size),
            "n_train_directed_events": int(train_mask.sum()),
            "n_valid_directed_events": int(valid_mask.sum()),
            "n_test_directed_events_not_analyzed": int(test_mask.sum()),
            "n_bridge_directed_events": int(bridge.sum()),
            "n_ring_directed_events": int(ring.sum()),
            "bridge_equals_not_ring": bool(np.array_equal(bridge, ~ring)),
        },
        "overall": {
            "S": _support_block(s_ids, train_mask, valid_mask, n_s, "S",
                                np.asarray(pack["key_strings"])),
            "C": _support_block(c_ids, train_mask, valid_mask, n_c, "C",
                                np.asarray(pack["key_strings"])),
            "combo": _support_block(combo_ids, train_mask, valid_mask, n_combo, "combo"),
        },
        "strata": {
            "bridge": _stratum(s_ids, c_ids, combo_ids, train_mask, valid_mask, bridge,
                               n_s, n_c, n_combo, "bridge"),
            "non_bridge": _stratum(s_ids, c_ids, combo_ids, train_mask, valid_mask, ~bridge,
                                   n_s, n_c, n_combo, "non_bridge"),
            "ring": _stratum(s_ids, c_ids, combo_ids, train_mask, valid_mask, ring,
                             n_s, n_c, n_combo, "ring"),
            "non_ring": _stratum(s_ids, c_ids, combo_ids, train_mask, valid_mask, ~ring,
                                 n_s, n_c, n_combo, "non_ring"),
        },
    }

    # per-molecule Valid support profile (fraction of events whose combo key is unseen)
    train_combo_counts = np.bincount(combo_ids[train_mask], minlength=n_combo)
    per_mol = []
    for m in range(int(pack["n_molecules"])):
        if mol_split[m] != 1:
            continue
        ids = combo_ids[ev_ptr[m]:ev_ptr[m + 1]]
        if ids.size == 0:
            continue
        cnt = train_combo_counts[ids]
        per_mol.append({
            "row_id": int(pack["row_ids"][m]),
            "n_events": int(ids.size),
            "seen_fraction": float(np.mean(cnt > 0)),
            "median_support": float(np.median(cnt)),
        })
    seen_fractions = np.asarray([r["seen_fraction"] for r in per_mol], np.float64)
    block_overall["valid_molecules"] = {
        "n": int(seen_fractions.size),
        "seen_fraction_quantiles": C.quantiles(seen_fractions, (0, 5, 25, 50, 75, 95, 100)),
        "molecules_with_all_events_seen": int(np.sum(seen_fractions == 1.0)),
        "molecules_with_no_event_seen": int(np.sum(seen_fractions == 0.0)),
    }

    # joint support: S and C both seen, combo seen, and the exact 2x2 table
    s_train = np.bincount(s_ids[train_mask], minlength=n_s)
    c_train = np.bincount(c_ids[train_mask], minlength=n_c)
    va_ids_s, va_ids_c, va_ids_combo = s_ids[valid_mask], c_ids[valid_mask], combo_ids[valid_mask]
    seen_s = s_train[va_ids_s] > 0
    seen_c = c_train[va_ids_c] > 0
    seen_combo = train_combo_counts[va_ids_combo] > 0
    block_overall["joint_valid"] = {
        "n": int(valid_mask.sum()),
        "S_and_C_seen": float(np.mean(seen_s & seen_c)),
        "combo_seen": float(np.mean(seen_combo)),
        "combo_seen_given_S_and_C_seen": float(np.mean(seen_combo[seen_s & seen_c]))
        if np.any(seen_s & seen_c) else 0.0,
        "S_seen_true_C_seen_false": int(np.sum(seen_s & ~seen_c)),
        "C_seen_true_S_seen_false": int(np.sum(~seen_s & seen_c)),
        "neither_seen": int(np.sum(~seen_s & ~seen_c)),
    }

    # occurrence counts of local structures themselves (Train)
    block_overall["occurrence_counts"] = {
        "S": _histogram(s_train[s_train > 0]),
        "C": _histogram(c_train[c_train > 0]),
        "combo": _histogram(train_combo_counts[train_combo_counts > 0]),
        "S_max_count": int(s_train.max(initial=0)),
        "C_max_count": int(c_train.max(initial=0)),
        "combo_max_count": int(train_combo_counts.max(initial=0)),
    }
    top = np.argsort(-train_combo_counts)[:10]
    block_overall["top_combos"] = []
    for i in top:
        if train_combo_counts[i] == 0:
            continue
        block_overall["top_combos"].append({
            "combo_id": int(i),
            "train_count": int(train_combo_counts[i]),
            "S_identity": str(pack["key_strings"][int(pack["combo_s"][i])])[:150],
            "C_identity": str(pack["key_strings"][int(pack["combo_c"][i])])[:150],
            "cut_bond_attributes": [int(x) for x in pack["combo_bond"][i]],
        })

    out = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack.get("cache_sha256"),
        "test_structural_statistics_reported": False,
        **block_overall,
    }
    C.write_json(C.analysis_dir() / "structural_support.json", out)
    log(f"wrote structural_support.json: valid S seen={out['overall']['S']['valid_seen_rate']:.3f} "
        f"C seen={out['overall']['C']['valid_seen_rate']:.3f} "
        f"combo seen={out['overall']['combo']['valid_seen_rate']:.3f}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())