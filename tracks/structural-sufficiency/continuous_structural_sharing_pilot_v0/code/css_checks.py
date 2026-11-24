"""Pre-training correctness checks for CSS-v0.

Gates (all computed from the frozen cache, the preparation artifact and freshly
built seed-0 models; no formal-training result is read):

C1  official row ids / 3360-420-420 split; 50% Fit is a true subset of 100% Fit
C2  every reference and every fitted statistic comes from the corresponding Fit
C3  atom renumbering + bond-order change leave z_v, q_v and the prediction intact
C4  q_v is finite, non-negative and sums to one
C5  identical structure descriptors produce identical reference weights
C6  DIRECT/SHARE parameter counts, initial state and batch plan are identical
C7  the new branch has a finite non-zero gradient and the original trunk updates
D   reference weights are not degenerate (fixed design, no temperature search)

Formal training must not start unless every gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import css_common as C
import css_data as D
import css_descriptors as DD
import css_fit as F
import css_model as M
import css_train as TR
import ses_data as SD
import ses_structure as SS
import ses_tensor as ST

CHECKS: list[dict[str, Any]] = []


def add(name: str, passed: bool, detail: Any) -> None:
    CHECKS.append({"name": name, "status": "PASS" if passed else "FAIL",
                   "detail": C.jsonable(detail)})


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def load_artifact(prep_dir: Path, size: int, structs: Mapping[int, SS.Struct],
                  n_mol: int) -> dict[str, Any]:
    with (prep_dir / f"fitstats_{size}.pkl").open("rb") as handle:
        fitstats = pickle.load(handle)
    fits = np.load(prep_dir / f"fit_{size}.npz")
    plan = np.load(prep_dir / f"plan_{size}.npy")
    return {"fitstats": fitstats, "fits": fits, "plan": plan,
            "ptr": TR.row_atom_ptr(structs, n_mol)}


def _permuted_raw(raw: Mapping[str, Any], row: int, rng: np.random.Generator
                  ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    n = a1 - a0
    m = b1 - b0
    new = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
           for k, v in raw.items()}
    perm = rng.permutation(n)
    old2new = np.empty(n, np.int64)
    old2new[perm] = np.arange(n, dtype=np.int64)
    new["atom_feat"][a0:a1] = np.asarray(raw["atom_feat"])[a0:a1][perm]
    for key in ("degree", "min_ring", "in_cycle"):
        new[key][a0:a1] = np.asarray(raw[key])[a0:a1][perm]
    border = rng.permutation(m)
    old_u = np.asarray(raw["bond_u"])[b0:b1][border]
    old_v = np.asarray(raw["bond_v"])[b0:b1][border]
    new["bond_u"][b0:b1] = old2new[old_u]
    new["bond_v"][b0:b1] = old2new[old_v]
    new["bond_feat"][b0:b1] = np.asarray(raw["bond_feat"])[b0:b1][border]
    return new, perm, border


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------


def check_split(bundle: Any, manifest: Mapping[str, Any],
                artifacts: Mapping[int, Mapping[str, Any]]) -> None:
    train = np.asarray(bundle.train_rows, np.int64)
    valid = np.asarray(bundle.valid_rows, np.int64)
    test = np.asarray(bundle.test_rows, np.int64)
    labels = np.asarray(bundle.labels, np.float64)
    ok = bool(train.size == 3360 and valid.size == 420 and test.size == 420)
    ok &= bool(np.intersect1d(train, valid).size == 0
               and np.intersect1d(train, test).size == 0
               and np.intersect1d(valid, test).size == 0)
    ok &= bool(np.union1d(np.union1d(train, valid), test).size == 4200)
    ok &= bool(np.array_equal(np.asarray(bundle.raw["row_ids"], np.int64),
                              np.arange(4200)))
    ok &= bool(np.isfinite(labels).all())
    fit100 = np.sort(np.asarray(artifacts[100]["fitstats"]["fit_rows"], np.int64))
    fit50 = np.sort(np.asarray(artifacts[50]["fitstats"]["fit_rows"], np.int64))
    held50 = np.sort(np.asarray(artifacts[50]["fitstats"]["heldout_rows"], np.int64))
    recomputed_fit, recomputed_held = D.fit_rows_for(50, train)
    ok &= bool(np.array_equal(fit100, train))
    ok &= bool(fit50.size == 1680 and held50.size == 1680)
    ok &= bool(np.isin(fit50, train).all() and np.intersect1d(fit50, held50).size == 0)
    ok &= bool(np.array_equal(np.union1d(fit50, held50), train))
    ok &= bool(np.array_equal(fit50, recomputed_fit)
               and np.array_equal(held50, recomputed_held))
    scaler_detail = {}
    for size in (100, 50):
        fs = artifacts[size]["fitstats"]
        rows = np.sort(np.asarray(fs["fit_rows"], np.int64))
        ok &= bool(np.array_equal(np.sort(np.asarray(fs["valid_rows"], np.int64)), valid))
        ok &= bool(np.array_equal(np.sort(np.asarray(fs["test_rows"], np.int64)), test))
        ok &= C.array_sha256(rows) == manifest["sizes"][str(size)]["fit_rows_sha256"]
        y = labels[rows]
        expect_mean, expect_std = float(y.mean()), float(y.std(ddof=0))
        ok &= abs(float(fs["target_scaler"]["mean"]) - expect_mean) < 1e-12
        ok &= abs(float(fs["target_scaler"]["std"]) - expect_std) < 1e-12
        scaler_detail[str(size)] = {"mean": expect_mean, "std": expect_std,
                                    "n": int(rows.size)}
    add("C1_official_rows_and_fit_subsets", bool(ok), {
        "train": int(train.size), "valid": int(valid.size), "test": int(test.size),
        "fit100": int(fit100.size), "fit50": int(fit50.size),
        "heldout50": int(held50.size),
        "fit50_is_train_subset": bool(np.isin(fit50, train).all()),
        "fit50_hash_selection_reproduced": bool(np.array_equal(fit50, recomputed_fit)),
        "fit_scalers": scaler_detail,
        "valid_rows_sha256": C.array_sha256(valid),
        "test_rows_sha256": C.array_sha256(test),
        "label_vault": bundle.hashes.get("label_vault"),
    })


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------


def check_fit_only(manifest: Mapping[str, Any], descriptions: Mapping[str, Any],
                   artifacts: Mapping[int, Mapping[str, Any]],
                   structs: Mapping[int, SS.Struct], n_mol: int
                   ) -> tuple[dict[int, F.FrozenProjection], bool]:
    desc = descriptions
    root_row = np.asarray(desc["root_row"], np.int64)
    n_roots = int(root_row.size)
    frozen_maps: dict[int, F.FrozenProjection] = {}
    detail: dict[str, Any] = {}
    ok = True
    for size in (100, 50):
        art = artifacts[size]
        fit_rows = np.sort(np.asarray(art["fitstats"]["fit_rows"], np.int64))
        held_rows = np.sort(np.asarray(art["fitstats"]["heldout_rows"], np.int64))
        fit_mask = np.isin(root_row, fit_rows)
        fit_index = np.flatnonzero(fit_mask)
        frozen, info = F.fit_projection(desc, fit_rows, fit_mask, n_roots, seed=0)
        frozen_maps[size] = frozen
        saved_z = np.asarray(art["fits"]["z"], np.float32)
        saved_q = np.asarray(art["fits"]["q"], np.float32)
        saved_ref_z = np.asarray(art["fits"]["ref_z"], np.float32)
        ref_roots = np.asarray(art["fits"]["ref_root_index"], np.int64)
        expected_ref_roots = fit_index[np.asarray(info["ref_positions"], np.int64)]
        saved_norm_d = F.branch_normalisation(np.asarray(saved_z[fit_mask], np.float64))
        saved_norm_s = F.branch_normalisation(np.asarray(saved_q[fit_mask], np.float64))
        bd, bs = F.canonical_branch_tables(np.asarray(saved_z), np.asarray(saved_q),
                                           saved_norm_d, saved_norm_s, structs,
                                           art["ptr"], n_mol)
        recomputed = frozen.project(desc, n_roots)
        tau_manifest = float(manifest["sizes"][str(size)]["temperature"]["tau"])
        d2 = F.squared_distances(np.asarray(saved_z[fit_mask], np.float64),
                                 np.asarray(saved_ref_z, np.float64))
        tau_from_artifact = max(1e-6, float(np.median(np.partition(d2, 7, axis=1)[:, 7])))
        checks = {
            "vocab_sha256_match": bool(info["vocab_sha256"]
                                       == manifest["sizes"][str(size)]["projection"]["vocab_sha256"]),
            "components_sha256_match": bool(info["components_sha256"]
                                            == manifest["sizes"][str(size)]["projection"]["components_sha256"]),
            "fit_z_hash_match": bool(C.array_sha256(np.asarray(saved_z[fit_mask]))
                                     == info["z_fit_sha256"]),
            "fit_q_hash_match": bool(C.array_sha256(np.asarray(saved_q[fit_mask]))
                                     == info["q_fit_sha256"]),
            "all_z_recomputed_identical": bool(_max_abs_diff(recomputed["z"], saved_z) == 0.0),
            "all_q_recomputed_identical": bool(_max_abs_diff(recomputed["q"], saved_q) <= 1e-7),
            "reference_roots_in_fit": bool(np.isin(root_row[ref_roots], fit_rows).all()),
            "reference_roots_reproduced": bool(np.array_equal(ref_roots, expected_ref_roots)),
            "reference_roots_not_heldout": bool(np.isin(root_row[ref_roots], held_rows).sum() == 0),
            "reference_z_match": bool(_max_abs_diff(saved_ref_z, frozen.z_ref) == 0.0),
            "tau_manifest_matches_recompute": bool(abs(tau_manifest
                                                       - float(info["temperature"]["tau"])) < 1e-15),
            "tau_equals_fit_median_d8": bool(abs(tau_manifest - tau_from_artifact) < 1e-12),
            "branch_mean_direct_match": bool(_max_abs_diff(saved_norm_d["mean"],
                                                           frozen.norm_direct["mean"]) == 0.0),
            "branch_mean_share_match": bool(_max_abs_diff(saved_norm_s["mean"],
                                                          frozen.norm_share["mean"]) == 0.0),
            "branch_rms_match": bool(abs(float(saved_norm_d["rms"])
                                         - float(frozen.norm_direct["rms"])) < 1e-12
                                     and abs(float(saved_norm_s["rms"])
                                             - float(frozen.norm_share["rms"])) < 1e-12),
            "branch_tables_match": bool(
                C.array_sha256(bd) == C.array_sha256(np.asarray(art["fits"]["branch_DIRECT"]))
                and C.array_sha256(bs) == C.array_sha256(np.asarray(art["fits"]["branch_SHARE"]))),
            "branch_tables_match_manifest": bool(
                C.array_sha256(bd) == manifest["sizes"][str(size)]["branch_table_sha256"]["DIRECT"]
                and C.array_sha256(bs) == manifest["sizes"][str(size)]["branch_table_sha256"]["SHARE"]),
        }
        size_ok = all(checks.values())
        detail[str(size)] = {"checks": checks,
                             "n_ref": int(ref_roots.size),
                             "tau": tau_manifest,
                             "z_all_max_abs_diff": _max_abs_diff(recomputed["z"], saved_z),
                             "q_all_max_abs_diff": _max_abs_diff(recomputed["q"], saved_q)}
        ok &= bool(size_ok)
    detail["fit_50_differs_from_fit_100"] = bool(
        manifest["sizes"]["100"]["projection"]["vocab_sha256"]
        != manifest["sizes"]["50"]["projection"]["vocab_sha256"])
    ok &= detail["fit_50_differs_from_fit_100"]
    add("C2_all_fitted_statistics_from_fit_only", bool(ok), detail)
    return frozen_maps, bool(ok)


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------


def check_relabelling(bundle: Any, artifacts: Mapping[int, Mapping[str, Any]],
                      descriptions: Mapping[str, Any],
                      frozen_maps: Mapping[int, F.FrozenProjection],
                      structs: Mapping[int, SS.Struct], rng: np.random.Generator) -> None:
    rows = [int(r) for r in rng.choice(4200, size=6, replace=False).tolist()]
    detail: dict[str, Any] = {"rows": rows}
    ok = True
    atom_dims, bond_dims = SD.atom_bond_dims()
    cfg = C.load_config()
    for size in (100, 50):
        art = artifacts[size]
        z_all = np.asarray(art["fits"]["z"], np.float32)
        q_all = np.asarray(art["fits"]["q"], np.float32)
        bd = np.asarray(art["fits"]["branch_DIRECT"], np.float32)
        bs = np.asarray(art["fits"]["branch_SHARE"], np.float32)
        frozen = frozen_maps[size]
        row_ptr = np.asarray(art["ptr"], np.int64)
        fs = art["fitstats"]
        norm = {"atom_scalar_mean": np.asarray(fs["atom_scalar_mean"]),
                "atom_scalar_scale": np.asarray(fs["atom_scalar_scale"]),
                "graph_mean": np.asarray(fs["graph_mean"]),
                "graph_scale": np.asarray(fs["graph_scale"])}
        model, _ = M.build_css_model(atom_dims, bond_dims, 0, norm, cfg)
        model.eval()
        records = []
        for row in rows:
            st = structs[row]
            n = int(st.n_atoms)
            base = int(row_ptr[row])
            order = np.asarray(st.support_atoms, np.int64)
            new_raw, perm, border = _permuted_raw(bundle.raw, row, rng)
            node_ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
            bond_ptr = np.asarray(bundle.raw["bond_ptr"], np.int64)
            a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
            b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
            st_new = SS.canonical_full_struct(new_raw, row)
            keys_to_compare = ("atom_feat", "bond_feat", "bond_u", "bond_v",
                               "atom_scalars", "dist1_atoms", "dist2_atoms",
                               "path_cat", "path_end_type", "path_closure",
                               "r2_bond", "graph_scalars")
            arrays_equal = all(
                np.array_equal(np.asarray(getattr(st, key)), np.asarray(getattr(st_new, key)))
                for key in keys_to_compare)
            bu = np.asarray(new_raw["bond_u"][b0:b1], np.int64)
            bv = np.asarray(new_raw["bond_v"][b0:b1], np.int64)
            pair = np.stack([np.minimum(bu, bv), np.maximum(bu, bv)], axis=1)
            bond_order = np.lexsort((pair[:, 1], pair[:, 0]))
            keys, counts, lptr = DD.molecule_descriptors(
                np.asarray(new_raw["atom_feat"])[a0:a1],
                np.asarray(new_raw["bond_feat"])[b0:b1][bond_order],
                pair[bond_order, 0], pair[bond_order, 1])
            subset = {"keys": keys, "counts": counts, "root_ptr": lptr,
                      "root_row": np.full(n, row, np.int64),
                      "root_atom": np.arange(n, dtype=np.int64)}
            out = frozen.project(subset, n)
            rank = np.asarray(st_new.support_atoms, np.int64)
            q_new = np.asarray(out["q"])[rank]
            z_new = np.asarray(out["z"])[rank]
            q_old = q_all[base + order]
            z_old = z_all[base + order]
            q_diff = _max_abs_diff(q_new, q_old)
            z_diff = _max_abs_diff(z_new, z_old)
            batch = ST.collate([st_new], "cpu")
            with torch.no_grad():
                batch["branch"] = torch.as_tensor(
                    F.normalise_branch(z_new, frozen.norm_direct), dtype=torch.float32)
                pd_new = float(model(batch, arm="SENTENCE_BASE")["yhat_std"][0])
                batch["branch"] = torch.as_tensor(
                    F.normalise_branch(q_new, frozen.norm_share), dtype=torch.float32)
                ps_new = float(model(batch, arm="SENTENCE_BASE")["yhat_std"][0])
                batch["branch"] = torch.as_tensor(bd[base:base + n], dtype=torch.float32)
                pd_old = float(model(batch, arm="SENTENCE_BASE")["yhat_std"][0])
                batch["branch"] = torch.as_tensor(bs[base:base + n], dtype=torch.float32)
                ps_old = float(model(batch, arm="SENTENCE_BASE")["yhat_std"][0])
            pdiff = max(abs(pd_new - pd_old), abs(ps_new - ps_old))
            record = {"row": row, "q_max_abs_diff": q_diff,
                      "z_max_abs_diff": z_diff,
                      "prediction_max_abs_diff": float(pdiff),
                      "canonical_struct_identical": bool(arrays_equal)}
            records.append(record)
            ok &= bool(arrays_equal and q_diff <= 1e-6 and z_diff <= 1e-6
                       and pdiff <= 1e-6)
        detail[str(size)] = records
    add("C3_atom_renumbering_and_edge_order_invariance", bool(ok), detail)


# ---------------------------------------------------------------------------
# C4 / C5
# ---------------------------------------------------------------------------


def check_q_properties(manifest: Mapping[str, Any],
                       artifacts: Mapping[int, Mapping[str, Any]]) -> None:
    detail: dict[str, Any] = {}
    ok = True
    for size in (100, 50):
        q = np.asarray(artifacts[size]["fits"]["q"], np.float32)
        finite = bool(np.isfinite(q).all())
        nonneg = bool((q >= 0).all())
        rowsum = np.abs(q.sum(axis=1) - 1.0)
        good = bool(finite and nonneg and float(rowsum.max()) < 1e-6)
        detail[str(size)] = {"shape": list(q.shape), "finite": finite,
                             "nonnegative": nonneg,
                             "max_row_sum_error": float(rowsum.max()),
                             "prep_q_stats": manifest["sizes"][str(size)]["q_stats"]}
        ok &= good
    add("C4_q_finite_nonnegative_normalised", bool(ok), detail)


def check_same_descriptor(descriptions: Mapping[str, Any],
                          artifacts: Mapping[int, Mapping[str, Any]],
                          n_roots: int) -> None:
    keys = np.asarray(descriptions["keys"], np.int64)
    counts = np.asarray(descriptions["counts"], np.int32)
    ptr = np.asarray(descriptions["root_ptr"], np.int64)
    root_row = np.asarray(descriptions["root_row"], np.int64)
    fingerprints = np.empty(n_roots, dtype=object)
    for root in range(n_roots):
        lo, hi = int(ptr[root]), int(ptr[root + 1])
        h = hashlib.sha256()
        h.update(keys[lo:hi].tobytes())
        h.update(counts[lo:hi].tobytes())
        fingerprints[root] = h.hexdigest()[:32]
    fingerprints = np.asarray(fingerprints)
    uniq, inverse, group_sizes = np.unique(fingerprints, return_inverse=True,
                                           return_counts=True)
    groups = np.flatnonzero(group_sizes > 1)
    max_checked = 5000
    checked = groups[:max_checked]
    detail: dict[str, Any] = {
        "n_roots": int(n_roots),
        "n_distinct_descriptors": int(uniq.size),
        "n_repeated_descriptor_groups": int(groups.size),
        "n_roots_in_repeated_groups": int(group_sizes[groups].sum()),
        "max_group_size": int(group_sizes.max()),
        "groups_checked": int(checked.size),
    }
    ok = bool(groups.size > 0)
    for size in (100, 50):
        z = np.asarray(artifacts[size]["fits"]["z"], np.float64)
        q = np.asarray(artifacts[size]["fits"]["q"], np.float64)
        q_diff = 0.0
        z_diff = 0.0
        cross = 0
        for g in checked.tolist():
            member = np.flatnonzero(inverse == g)
            q_diff = max(q_diff, _max_abs_diff(q[member], np.repeat(q[member[:1]], member.size, axis=0)))
            z_diff = max(z_diff, _max_abs_diff(z[member], np.repeat(z[member[:1]], member.size, axis=0)))
            if np.unique(root_row[member]).size > 1:
                cross += 1
        detail[str(size)] = {"q_max_abs_diff": q_diff, "z_max_abs_diff": z_diff,
                             "cross_molecule_groups_checked": int(cross)}
        ok &= bool(q_diff <= 1e-7 and z_diff <= 1e-7)
    add("C5_identical_descriptor_identical_reference_weights", bool(ok), detail)


# ---------------------------------------------------------------------------
# C6 / C7 / D
# ---------------------------------------------------------------------------


def check_model_identity(artifacts: Mapping[int, Mapping[str, Any]],
                         manifest: Mapping[str, Any]) -> None:
    atom_dims, bond_dims = SD.atom_bond_dims()
    cfg = C.load_config()
    norm = {"atom_scalar_mean": np.zeros(5), "atom_scalar_scale": np.ones(5),
            "graph_mean": np.zeros(8), "graph_scale": np.ones(8)}
    m1, i1 = M.build_css_model(atom_dims, bond_dims, 0, norm, cfg)
    m2, i2 = M.build_css_model(atom_dims, bond_dims, 0, norm, cfg)
    trunk_ok = (int(i1["parameters"]["total"]) - int(i1["branch_parameters"])
                == C.FROZEN_PREDICTOR_PARAMS)
    plan_ok = True
    plan_detail = {}
    for size in (100, 50):
        plan = np.asarray(artifacts[size]["plan"])
        plan_detail[str(size)] = {
            "sha256": C.array_sha256(plan),
            "prep_sha256": manifest["batch_plans"][str(size)]["sha256"],
            "epoch0_sha256": C.array_sha256(plan[0]),
            "shape": list(plan.shape),
        }
        plan_ok &= C.array_sha256(plan) == manifest["batch_plans"][str(size)]["sha256"]
        plan_ok &= plan.shape[1] == int(artifacts[size]["fitstats"]["fit_rows"].size)
    branches_differ = all(
        C.array_sha256(np.asarray(artifacts[size]["fits"]["branch_DIRECT"]))
        != C.array_sha256(np.asarray(artifacts[size]["fits"]["branch_SHARE"]))
        for size in (100, 50))
    ok = bool(i1["parameters"] == i2["parameters"]
              and i1["init_hashes"] == i2["init_hashes"]
              and trunk_ok and plan_ok and branches_differ)
    add("C6_matching_parameters_initial_state_batch_plan", ok, {
        "ledger": i1["parameters"],
        "branch_parameters": int(i1["branch_parameters"]),
        "trunk_parameters": int(i1["parameters"]["total"]) - int(i1["branch_parameters"]),
        "init_hashes_identical": bool(i1["init_hashes"] == i2["init_hashes"]),
        "batch_plans": plan_detail,
        "branch_inputs_differ": bool(branches_differ),
    })


def check_gradients(bundle: Any, artifacts: Mapping[int, Mapping[str, Any]],
                    structs: Mapping[int, SS.Struct]) -> None:
    atom_dims, bond_dims = SD.atom_bond_dims()
    cfg = C.load_config()
    labels = np.asarray(bundle.labels, np.float64)
    detail: dict[str, Any] = {}
    ok = True
    for size in (100, 50):
        art = artifacts[size]
        fs = art["fitstats"]
        fit_rows = np.sort(np.asarray(fs["fit_rows"], np.int64))
        plan = np.asarray(art["plan"])
        norm = {"atom_scalar_mean": np.asarray(fs["atom_scalar_mean"]),
                "atom_scalar_scale": np.asarray(fs["atom_scalar_scale"]),
                "graph_mean": np.asarray(fs["graph_mean"]),
                "graph_scale": np.asarray(fs["graph_scale"])}
        mean, std = float(fs["target_scaler"]["mean"]), float(fs["target_scaler"]["std"])
        for arm in C.ARMS:
            table = np.asarray(art["fits"][f"branch_{arm}"], np.float32)
            torch.manual_seed(0)
            model, _ = M.build_css_model(atom_dims, bond_dims, 0, norm, cfg)
            model.zero_unseen(fs["atom_ok"], fs["bond_ok"], atom_dims, bond_dims)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            rows = fit_rows[plan[0, :16].astype(np.int64)]
            batch = ST.collate([structs[int(r)] for r in rows.tolist()], "cpu")
            batch["branch"] = M.branch_gather(rows, structs, table, art["ptr"], "cpu")
            target = torch.as_tensor((labels[rows] - mean) / std, dtype=torch.float32)
            out = model(batch, arm="SENTENCE_BASE")
            loss = torch.mean((out["yhat_std"] - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            branch_grad = float(model.branch_proj.weight.grad.norm())
            trunk_grad = float(model.atom_emb[0].weight.grad.norm())
            before = model.atom_emb[0].weight.detach().clone()
            opt.step()
            delta = float((model.atom_emb[0].weight.detach() - before).abs().max())
            good = (np.isfinite(float(loss)) and np.isfinite(branch_grad)
                    and branch_grad > 0 and np.isfinite(trunk_grad) and trunk_grad > 0
                    and delta > 0)
            detail[f"{size}_{arm}"] = {
                "loss": float(loss), "branch_grad_norm": branch_grad,
                "trunk_grad_norm": trunk_grad,
                "trunk_weight_max_delta_after_step": delta,
                "finite_nonzero": bool(good)}
            ok &= bool(good)
    add("C7_branch_and_trunk_gradients_finite_nonzero", bool(ok), detail)


def check_degeneracy(manifest: Mapping[str, Any]) -> None:
    detail: dict[str, Any] = {}
    ok = True
    for size in (100, 50):
        qs = manifest["sizes"][str(size)]["q_stats"]
        ts = manifest["sizes"][str(size)]["temperature"]
        gate = (float(qs["mean_effective_number"]) > 1.01
                and float(qs["mean_max_weight"]) < 0.999
                and float(qs["spread_from_mean_l2"]) > 1e-6
                and float(ts["tau"]) > 0
                and int(qs["nonfinite"]) == 0
                and float(qs["row_sum_min"]) > 1.0 - 1e-6
                and float(qs["row_sum_max"]) < 1.0 + 1e-6)
        detail[str(size)] = {
            "mean_entropy": qs["mean_entropy"], "max_entropy": qs["max_entropy"],
            "mean_effective_number": qs["mean_effective_number"],
            "max_effective_number": qs["max_effective_number"],
            "mean_max_weight": qs["mean_max_weight"],
            "spread_from_mean_l2": qs["spread_from_mean_l2"],
            "nearest_squared_median": ts["nearest_squared_median"],
            "d8_squared_median": ts["d8_squared_median"],
            "tau": ts["tau"],
            "mean_reference_weights": manifest["sizes"][str(size)]["q_mean_column_weight"],
            "degenerate": not bool(gate),
        }
        ok &= bool(gate)
    add("D_reference_weights_not_degenerate", bool(ok), detail)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(prep_dir: str = str(C.PREP_DIR), out: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    prep_path = Path(prep_dir)
    manifest = C.read_json(prep_path / "prep_manifest.json")
    if manifest.get("format") != "css-v0-prep-v1":
        raise ValueError("prep artifact format mismatch")
    topology = C.assert_frozen_sources()
    bundle = SD.load_bundle()
    n_mol = int(bundle.labels.size)
    with (prep_path / "structs.pkl").open("rb") as handle:
        structs = pickle.load(handle)["structs"]
    if len(structs) != n_mol:
        raise ValueError("struct count mismatch")
    artifacts = {size: load_artifact(prep_path, size, structs, n_mol)
                 for size in (100, 50)}
    descriptions = DD.build_all(bundle.raw, range(n_mol))
    n_roots = int(np.asarray(descriptions["root_row"]).size)
    if n_roots != int(manifest["n_roots"]):
        raise ValueError("descriptor root count mismatch")

    rng = np.random.default_rng(0)
    check_split(bundle, manifest, artifacts)
    frozen_maps, _ = check_fit_only(manifest, descriptions, artifacts, structs, n_mol)
    check_relabelling(bundle, artifacts, descriptions, frozen_maps, structs, rng)
    check_q_properties(manifest, artifacts)
    check_same_descriptor(descriptions, artifacts, n_roots)
    check_model_identity(artifacts, manifest)
    check_gradients(bundle, artifacts, structs)
    check_degeneracy(manifest)

    payload = {
        "format": "css-v0-checks-v1",
        "protocol_id": manifest["protocol_id"],
        "prep_commit": manifest.get("commit"),
        "host": socket.gethostname(),
        "now_utc": C.now_utc(),
        "topology": topology,
        "checks": CHECKS,
        "n_checks": len(CHECKS),
        "n_pass": int(sum(c["status"] == "PASS" for c in CHECKS)),
        "all_pass": bool(all(c["status"] == "PASS" for c in CHECKS)),
        "wall_seconds": float(time.time() - t0),
    }
    out_path = Path(out) if out else (C.ANALYSIS_DIR / "checks.json")
    C.write_json(out_path, payload)
    print(f"checks: {payload['n_pass']}/{payload['n_checks']} PASS "
          f"({payload['wall_seconds']:.1f}s) -> {out_path}", flush=True)
    for check in CHECKS:
        print(f"  [{check['status']}] {check['name']}", flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", default=str(C.PREP_DIR))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    payload = main(args.prep_dir, args.out)
    if not payload["all_pass"]:
        raise SystemExit(1)
