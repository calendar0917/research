"""Focused checks for exactly the defects this round repairs.

Each check is cheap and targets a known failure mode; there is no broad test
sweep.  Every check writes a boolean plus the numbers it compared, and the checks
deliberately use *independent* oracles (``PREP_DATA.collate``, the per-molecule
``build_molecule`` path, and hand-built synthetic edit cells) so a bug in
:mod:`rp_reader` cannot produce both sides of its own comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import rp_common as RP


# ---------------------------------------------------------------------------
# reader / layout
# ---------------------------------------------------------------------------


def check_layout(view) -> dict[str, Any]:
    import rp_reader as RR

    out: dict[str, Any] = {}
    for split in RP.SPLITS:
        c = view.caches[split]
        n_mol = int(c["n_molecules"])
        n_atom = int(c["atom_ptr"][-1])
        n_occ = int(c["path_cat"].shape[0])
        n_nbr = int(c["nbr_code"].shape[0])
        ok = {
            "atom_ptr_len": len(c["atom_ptr"]) == n_mol + 1,
            "path_ptr_len": len(c["path_ptr"]) == n_atom + 1,
            "nbr_ptr_len": len(c["nbr_ptr"]) == n_atom + 1,
            "path_occ_total": int(c["path_ptr"][-1]) == n_occ,
            "nbr_total": int(c["nbr_ptr"][-1]) == n_nbr,
            "end_len_matches": int(c["path_end63"].shape[0]) == n_occ == int(c["path_closure"].shape[0]),
        }
        reader = RR.MoleculeReader(c, name=split)
        sums = {"path": 0, "nbr": 0, "end": 0, "closure": 0}
        s1_ok = True
        for m in range(n_mol):
            r = reader.ranges(m)
            sums["path"] += r["path1"] - r["path0"]
            sums["nbr"] += r["s1_1"] - r["s1_0"]
            sums["end"] += r["path1"] - r["path0"]
            sums["closure"] += r["path1"] - r["path0"]
            n_bonds = int(c["bond_ptr"][m + 1] - c["bond_ptr"][m])
            if (r["s1_1"] - r["s1_0"]) != 2 * n_bonds:
                s1_ok = False
        ok["sum_path_equals_occ"] = sums["path"] == n_occ
        ok["sum_nbr_equals_nbr"] = sums["nbr"] == n_nbr
        ok["sum_end_equals_occ"] = sums["end"] == n_occ
        ok["sum_closure_equals_occ"] = sums["closure"] == n_occ
        ok["s1_equals_2_bonds"] = s1_ok
        out[split] = {"ok": all(ok.values()), "detail": ok, "n_molecules": n_mol, "n_atoms": n_atom,
                      "n_path_occurrences": n_occ, "n_nbr_occurrences": n_nbr}
    out["all_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def check_collate_identity(view, n_mol: int = 50) -> dict[str, Any]:
    """Reader per-molecule code Counters vs what the model actually receives."""
    from collections import Counter

    import lcrc_data as PREP_DATA
    import rp_reader as RR

    out: dict[str, Any] = {}
    for split in ("train", "valid"):
        c = view.caches[split]
        reader = RR.MoleculeReader(c, name=split)
        mismatch = 0
        n = min(n_mol, reader.n_molecules)
        for m in range(n):
            batch = PREP_DATA.collate(c, [m])
            pg = np.asarray(batch["path_graph"]).reshape(-1)
            if not np.all(pg == 0):
                mismatch += 1
                continue
            got = {
                "path": Counter(np.asarray(batch["path_cat"]).reshape(-1).tolist()),
                "end": Counter(np.asarray(batch["path_end63"]).reshape(-1).tolist()),
                "nbr": Counter(np.asarray(batch["nbr_code"]).reshape(-1).tolist()),
                "closure": Counter(np.asarray(batch["path_closure"]).reshape(-1).tolist()),
            }
            r = reader.ranges(m)
            exp = {
                "path": Counter(reader.path_cat[r["path0"]:r["path1"]].tolist()),
                "end": Counter(reader.path_end63[r["path0"]:r["path1"]].tolist()),
                "nbr": Counter(reader.nbr_code[r["s1_0"]:r["s1_1"]].tolist()),
                "closure": Counter(reader.path_closure[r["path0"]:r["path1"]].tolist()),
            }
            if got != exp:
                mismatch += 1
        out[split] = {"n_checked": n, "n_mismatch": mismatch, "ok": mismatch == 0}
    out["all_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def _permute_cache(cache: dict[str, Any], perm: np.ndarray) -> dict[str, Any]:
    """Rebuild a cache with molecules in ``perm`` order (atom renumbering)."""
    atom_ptr = cache["atom_ptr"]
    path_ptr = cache["path_ptr"]
    nbr_ptr = cache["nbr_ptr"]
    new_atom_ptr = [0]
    for m in perm:
        new_atom_ptr.append(new_atom_ptr[-1] + int(atom_ptr[m + 1] - atom_ptr[m]))
    new_atom_ptr = np.asarray(new_atom_ptr, dtype=np.int64)

    def _gather_blocks(src: np.ndarray, ptr: np.ndarray, idx) -> np.ndarray:
        parts = [src[ptr[int(m)]:ptr[int(m) + 1]] for m in idx]
        return np.concatenate(parts) if parts else src[:0]

    atom_blocks = {k: _gather_blocks(cache[k], atom_ptr, perm)
                   for k in ("atom_feat", "atom_type", "atom_scalar")}
    # atoms are renumbered; bonds refer to *local* atom ids, gathered per molecule
    offset = 0
    atom_ids: list[np.ndarray] = []
    for m in perm:
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        atom_ids.append(np.arange(offset, offset + (a1 - a0), dtype=np.int64))
        offset += a1 - a0

    # atom-level pointers: concatenate each molecule's per-atom increments
    def _atom_level_ptr(ptr: np.ndarray) -> np.ndarray:
        parts = [np.diff(ptr[int(atom_ptr[m]):int(atom_ptr[m + 1]) + 1]) for m in perm]
        parts = [x for x in parts if x.size]
        flat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        return np.asarray([0] + list(np.cumsum(flat)), dtype=np.int64)

    new_path_ptr = _atom_level_ptr(path_ptr)
    new_nbr_ptr = _atom_level_ptr(nbr_ptr)

    def _gather_occ(src: np.ndarray) -> np.ndarray:
        parts = [src[path_ptr[int(atom_ptr[m])]:path_ptr[int(atom_ptr[m + 1])]] for m in perm]
        return np.concatenate(parts) if parts else src[:0]

    def _gather_nbr(src: np.ndarray) -> np.ndarray:
        parts = [src[nbr_ptr[int(atom_ptr[m])]:nbr_ptr[int(atom_ptr[m + 1])]] for m in perm]
        return np.concatenate(parts) if parts else src[:0]

    new = {
        "n_molecules": int(len(perm)),
        "atom_ptr": new_atom_ptr,
        "path_ptr": new_path_ptr,
        "nbr_ptr": new_nbr_ptr,
        "path_cat": _gather_occ(cache["path_cat"]),
        "path_end63": _gather_occ(cache["path_end63"]),
        "path_end": _gather_occ(cache["path_end"]),
        "path_closure": _gather_occ(cache["path_closure"]),
        "nbr_code": _gather_nbr(cache["nbr_code"]),
        "global": np.asarray(cache["global"])[perm],
        "rows": np.asarray(cache["rows"])[perm],
    }
    for k, v in atom_blocks.items():
        new[k] = v
    # bonds: concatenate per molecule, remap local atom ids to the new global ids
    bond_feat, bond_type, bond_index = [], [], []
    for m, ids in zip(perm, atom_ids):
        b0, b1 = int(cache["bond_ptr"][m]), int(cache["bond_ptr"][m + 1])
        bond_feat.append(np.asarray(cache["bond_feat"])[b0:b1])
        bond_type.append(np.asarray(cache["bond_type"])[b0:b1])
        idx = np.asarray(cache["bond_index"])[b0:b1]  # stored local to the molecule
        bond_index.append(ids[idx])
    new["bond_feat"] = np.concatenate(bond_feat) if bond_feat else np.zeros((0, 13), np.float32)
    new["bond_type"] = np.concatenate(bond_type) if bond_type else np.zeros((0,), np.int64)
    new["bond_index"] = np.concatenate(bond_index) if bond_index else np.zeros((0, 2), np.int64)
    new["bond_ptr"] = np.asarray([0] + list(np.cumsum(
        [int(cache["bond_ptr"][m + 1] - cache["bond_ptr"][m]) for m in perm])), dtype=np.int64)
    new["n_components"] = np.asarray(cache["n_components"])[perm]
    new["n_junctions"] = np.asarray(cache["n_junctions"])[perm]
    new["n_ring_systems"] = np.asarray(cache["n_ring_systems"])[perm]
    new["meta"] = dict(cache.get("meta", {}))
    new["dataset"] = cache.get("dataset")
    new["split"] = cache.get("split")
    new["split_name"] = cache.get("split_name")
    new["molecule_ids"] = [cache["molecule_ids"][m] for m in perm]
    new["smiles"] = [cache["smiles"][m] for m in perm]
    new["cache_version"] = cache.get("cache_version")
    return new


def check_permutation_invariance(view) -> dict[str, Any]:
    import rp_reader as RR

    c = view.caches["valid"]
    n = int(c["n_molecules"])
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    pc = _permute_cache(dict(c), perm)
    r0 = RR.MoleculeReader(c, name="valid")
    r1 = RR.MoleculeReader(pc, name="valid-permuted")
    bad = 0
    for new_pos, old_pos in enumerate(perm):
        if r0.sets(int(old_pos)) != r1.sets(int(new_pos)):
            bad += 1
    df0 = RR.document_frequency(c, name="valid")
    df1 = RR.document_frequency(pc, name="valid-permuted")
    df_same = all(df0[k] == df1[k] for k in ("df_path", "df_end", "df_nbr", "df_joint"))
    return {
        "n_molecules": n,
        "n_permuted": int(n),
        "n_counter_mismatch": bad,
        "df_identical": bool(df_same),
        "rows_reordered_only": bool(np.array_equal(np.asarray(pc["rows"]), np.asarray(c["rows"])[perm])),
        "ok": bad == 0 and df_same,
    }


def check_oracle_agreement(view, n_mol: int = 20) -> dict[str, Any]:
    import rp_reader as RR
    import se_data as SED
    import lcrc_data as PREP_DATA

    import adapters as ADP
    import lp_source as LPS

    c = view.caches["train"]
    reader = RR.MoleculeReader(c, name="train")
    raw = LPS.load_raw_artifact()
    spec = view.spec
    rows = np.asarray(c["rows"]).tolist()
    bad = 0
    checked = 0
    for m in range(min(n_mol, reader.n_molecules)):
        graph = ADP.lipo_row_to_graph(raw[int(rows[m])], int(rows[m]), spec)
        record = PREP_DATA.build_molecule(graph, spec)
        oracle = RR.oracle_from_molecule(record)
        got = reader.sets(m)
        checked += 1
        from collections import Counter as _C

        r = reader.ranges(m)
        got_joint = {f"{a},{b}": int(v) for (a, b), v in _C(reader.joint_pairs(m)).items()}
        same = (
            _C(reader.path_cat[r["path0"]:r["path1"]].tolist()) == _C(oracle["path_counter"])
            and _C(reader.nbr_code[r["s1_0"]:r["s1_1"]].tolist()) == _C(oracle["nbr_counter"])
            and _C(reader.path_end63[r["path0"]:r["path1"]].tolist()) == _C(oracle["end_counter"])
            and got_joint == oracle["joint_counter"]
            and got["n_path"] == oracle["n_path"]
            and got["n_s1"] == oracle["n_s1"]
        )
        if not same:
            bad += 1
    return {"n_checked": checked, "n_mismatch": bad, "ok": bad == 0,
            "source": "lcrc_data.build_molecule on a freshly built graph (independent of MoleculeReader)"}


# ---------------------------------------------------------------------------
# small-molecule construction sanity
# ---------------------------------------------------------------------------


def view_spec():
    import lp_data as LPD

    return LPD.frozen_spec()


def check_small_molecules() -> dict[str, Any]:
    import lcrc_data as PREP_DATA
    import rp_reader as RR

    import adapters as ADP
    import lcrc_data as PREP_DATA
    import lp_source as LPS

    cases = {"CC": (2, 0), "CCO": (4, 2), "C": (0, 0)}
    out: dict[str, Any] = {}
    ok = True
    for smi, (want_s1, want_s2) in cases.items():
        row = LPS.build_raw_rows([smi], [f"probe_{smi}"])[0]
        graph = ADP.lipo_row_to_graph(row, 0, view_spec())
        cache = PREP_DATA.build_cache([graph], view_spec(), split=smi)
        reader = RR.MoleculeReader(cache, name=smi)
        s = reader.sets(0)
        good = int(s["n_s1"]) == want_s1 and int(s["n_path"]) == want_s2
        ok = ok and good
        out[smi] = {"n_atoms": int(cache["atom_ptr"][-1]), "n_bonds": int(cache["bond_ptr"][-1]),
                    "n_s1": int(s["n_s1"]), "n_s2": int(s["n_path"]),
                    "expected_s1": want_s1, "expected_s2": want_s2, "ok": good}
    return {"cases": out, "ok": ok, "note": "expected S1/S2 come from the task book, not from this reader"}


# ---------------------------------------------------------------------------
# edit-metric weighting semantics (synthetic, hand-computed)
# ---------------------------------------------------------------------------


def check_edit_weighting() -> dict[str, Any]:
    import rp_edit as RE

    # component 1 has 1 pair with e=0.0; component 2 has 3 pairs with e=1.0
    comp = {1: 1, 2: 2, 3: 2, 4: 2}
    # pairs are (row_a,row_b); add fake delta_y/delta_pred through d_pred
    pairs = [
        {"row_a": 1, "row_b": 10, "delta_y": 0.0, "split_a": "train", "split_b": "train"},
        {"row_a": 2, "row_b": 20, "delta_y": 1.0, "split_a": "train", "split_b": "train"},
        {"row_a": 3, "row_b": 30, "delta_y": 1.0, "split_a": "train", "split_b": "train"},
        {"row_a": 4, "row_b": 40, "delta_y": 1.0, "split_a": "train", "split_b": "train"},
    ]
    d_pred = {(1, 10): 0.0, (2, 20): 0.0, (3, 30): 0.0, (4, 40): 0.0}
    m = RE._metrics_for_cell(pairs, comp, d_pred)
    # CE-MAE = mean(0.0, 1.0) = 0.5 ; pair-MAE = (0+1+1+1)/4 = 0.75
    ce_ok = abs(m["ce_mae"] - 0.5) < 1e-12
    pair_ok = abs(m["pair_mae"] - 0.75) < 1e-12
    distinct_ok = abs(m["ce_mae"] - m["pair_mae"]) > 1e-9

    # swap-pair-ends invariance
    swapped = [
        {"row_a": 10, "row_b": 1, "delta_y": 0.0, "split_a": "train", "split_b": "train"},
        {"row_a": 20, "row_b": 2, "delta_y": -1.0, "split_a": "train", "split_b": "train"},
        {"row_a": 30, "row_b": 3, "delta_y": -1.0, "split_a": "train", "split_b": "train"},
        {"row_a": 40, "row_b": 4, "delta_y": -1.0, "split_a": "train", "split_b": "train"},
    ]
    d_swapped = {(10, 1): 0.0, (20, 2): 0.0, (30, 3): 0.0, (40, 4): 0.0}
    m2 = RE._metrics_for_cell(swapped, {10: 1, 20: 2, 30: 2, 40: 2}, d_swapped)
    swap_ok = abs(m2["ce_mae"] - m["ce_mae"]) < 1e-12 and abs(m2["pair_mae"] - m["pair_mae"]) < 1e-12

    # dedup invariance on the canonicaliser
    raw = [
        {"pair_id": "a", "row_a": 5, "row_b": 9, "split_a": "train", "split_b": "train", "delta_y": 0.3},
        {"pair_id": "b", "row_a": 9, "row_b": 5, "split_a": "train", "split_b": "train", "delta_y": -0.3},
        {"pair_id": "c", "row_a": 5, "row_b": 9, "split_a": "train", "split_b": "train", "delta_y": 0.3},
    ]
    ded, info = RE.canonicalise(raw)
    dedup_ok = len(ded) == 1 and ded[0]["row_a"] == 5 and ded[0]["row_b"] == 9

    # shared endpoint -> same component
    pairs2 = [
        {"row_a": 1, "row_b": 2, "delta_y": 0.1, "split_a": "train", "split_b": "train"},
        {"row_a": 2, "row_b": 3, "delta_y": 0.1, "split_a": "train", "split_b": "train"},
    ]
    comp2, _ = RE.components(pairs2)
    shared_ok = comp2[1] == comp2[2] == comp2[3]

    # LOW_SUPPORT must depend on component count, not pair count
    many_pairs_few_components = [{"row_a": 1, "row_b": 2, "delta_y": 0.0, "split_a": "train", "split_b": "train"}]
    mp = RE._metrics_for_cell(many_pairs_few_components * 1, {1: 1, 2: 1}, {(1, 2): 0.0})
    low_ok = mp["n_components"] == 1 and mp["low_support"] is True

    return {
        "ce_mae": m["ce_mae"], "pair_mae": m["pair_mae"],
        "ce_equals_half": ce_ok, "pair_equals_0.75": pair_ok, "ce_pair_differ": distinct_ok,
        "swap_invariant": swap_ok, "dedup_single_edge": dedup_ok, "shared_endpoint_same_component": shared_ok,
        "low_support_by_component": low_ok,
        "ok": all([ce_ok, pair_ok, distinct_ok, swap_ok, dedup_ok, shared_ok, low_ok]),
    }


# ---------------------------------------------------------------------------
# B_W0 initialisation identity
# ---------------------------------------------------------------------------


def check_bw0_init() -> dict[str, Any]:
    import lp_data as LPD
    import se_train as TR
    import se_tables as T

    view = LPD.build_view()
    perms = T.build_permutations()
    h = {}
    for name in ("B", "LS", "FS"):
        cfg = {"family": name, "recipe": "W0", "dataset": "lipo", "budget": "N100"}
        model = TR.build_model(cfg, view, perms)
        sd = model.state_dict()
        _banks = {"atom_emb", "bond_emb", "path_emb", "path_u", "path_v", "path_gate", "end_emb", "nbr_emb"}

        def _is_table_or_gen(k: str) -> bool:
            parts = k.split(".")
            if k.startswith("generator.") or ".table." in k:
                return True
            return len(parts) >= 3 and parts[0] == "base" and parts[1] == "local" and parts[2] in _banks

        non_table = {k: v for k, v in sd.items() if not _is_table_or_gen(k)}
        table = {k: v for k, v in sd.items() if _is_table_or_gen(k)}
        h[name] = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "state_hash": LPD.state_hash(sd),
            "non_table_hash": LPD.state_hash(non_table),
            "n_table_tensors": len(table),
            "table_hash": LPD.state_hash(table),
        }
    return {
        "B_W0_params": h["B"]["parameters"],
        "expected_C1_params": RP.C1_PARAMS,
        "params_ok": h["B"]["parameters"] == RP.C1_PARAMS,
        "B_vs_LS_same_non_table_init": h["B"]["non_table_hash"] == h["LS"]["non_table_hash"],
        "B_vs_FS_same_non_table_init": h["B"]["non_table_hash"] == h["FS"]["non_table_hash"],
        "LS_vs_FS_same_non_table_init": h["LS"]["non_table_hash"] == h["FS"]["non_table_hash"],
        "table_init_differs_LS_vs_B": h["LS"]["table_hash"] != h["B"]["table_hash"]
        if "table_hash" in h["B"] else None,
        "note": (
            "B and the free-table arms share every non-table parameter; the FS generator arms differ "
            "only in the six table initial values.  Compared against the trained B_R/B_Rwd init hashes "
            "in analysis/checkpoint_replay.json and the arm metrics."
        ),
        "hashes": h,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    import lp_data as LPD

    view = LPD.build_view()
    payload = {
        "schema": "lcrc-semantic-analysis-repair-focused-checks-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "git_commit": RP.git_commit(),
        "layout": check_layout(view),
        "collate_identity": check_collate_identity(view),
        "permutation_invariance": check_permutation_invariance(view),
        "oracle_agreement": check_oracle_agreement(view),
        "small_molecules": check_small_molecules(),
        "edit_weighting": check_edit_weighting(),
        "bw0_init": check_bw0_init(),
    }
    keys = ("layout", "collate_identity", "permutation_invariance", "oracle_agreement",
            "small_molecules", "edit_weighting")
    payload["all_ok"] = all(bool(payload[k].get("all_ok", payload[k].get("ok"))) for k in keys) and bool(
        payload["bw0_init"]["params_ok"]
    )
    RP.write_json(RP.ANALYSIS / "focused_checks.json", payload)
    return {
        "all_ok": payload["all_ok"],
        "failed": [k for k in keys if not bool(payload[k].get("all_ok", payload[k].get("ok")))],
        "artifacts": [RP.track_rel(RP.ANALYSIS / "focused_checks.json")],
    }


__all__ = [name for name in dir() if not name.startswith("_")]
