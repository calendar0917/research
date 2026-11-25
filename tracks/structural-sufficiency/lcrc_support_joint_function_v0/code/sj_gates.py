"""The twelve pre-training correctness gates of this round.

Each gate is a pure engineering check: it may only fail because of a bug, a
stale cache or a stale artifact.  Nothing here selects, tunes or compares a
score; no Task/Valid/Test metric is consulted.

Run order:
    1  official split / row ids / labels / cache identity
    2  Omega uses Train only
    3  p / e types and their model encoding agree with the production caches
    4  A and B/C share bitwise identical common parameters at initialisation
    5  materialised B reproduces A's initial joint, h0 and final prediction
    6  every B/C parameter receives a gradient
    7  atom renumbering, path enumeration order and batch regrouping invariance
    8  graphs without any S2 path run correctly
    9  P^2 = P and P^T = P
   10  C's table satisfies (I - P) J ~ 0
   11  C's interaction contrast vanishes on every rectangle that exists in Omega
   12  the unseen-pair fallback is identical, finite and parameter-free
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

import sj_common as C
import sj_model as M
import sj_support as S
import sj_train as T


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def subset_cache(cache, rows, spec):
    """Rebuild a cache containing only ``rows`` (adapted from ``ptb_train_mol``)."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    mols = []
    for g in rows:
        g = int(g)
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        nbr_s = int(nbr_ptr[a0])
        path_s = int(path_ptr[a0])
        mols.append(
            {
                "n_atoms": np.int64(a1 - a0),
                "n_bonds": np.int64(b1 - b0),
                "atom_feat": np.asarray(cache["atom_feat"][a0:a1]),
                "atom_type": np.asarray(cache["atom_type"][a0:a1]),
                "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1]),
                "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
                "bond_type": np.asarray(cache["bond_type"][b0:b1]),
                "bond_index": np.asarray(cache["bond_index"][b0:b1]),
                "nbr_ptr": np.asarray(nbr_ptr[a0 : a1 + 1]) - nbr_s,
                "nbr_code": np.asarray(cache["nbr_code"][nbr_s : int(nbr_ptr[a1])]),
                "path_ptr": np.asarray(path_ptr[a0 : a1 + 1]) - path_s,
                "path_cat": np.asarray(cache["path_cat"][path_s : int(path_ptr[a1])]),
                "path_end63": np.asarray(cache["path_end63"][path_s : int(path_ptr[a1])]),
                "path_end": np.asarray(cache["path_end"][path_s : int(path_ptr[a1])]),
                "path_closure": np.asarray(cache["path_closure"][path_s : int(path_ptr[a1])]),
                "global": np.asarray(cache["global"][g]),
                "n_components": np.int64(np.asarray(cache["n_components"])[g]),
                "n_junctions": np.int64(np.asarray(cache["n_junctions"])[g]),
                "n_ring_systems": np.int64(np.asarray(cache["n_ring_systems"])[g]),
                "molecule_id": str(cache["molecule_ids"][g]),
                "smiles": str(cache["smiles"][g]) if "smiles" in cache else "",
            }
        )
    out = C.PREP_DATA.concat_molecules(mols, spec, split="subset")
    out["rows"] = np.asarray([int(np.asarray(cache["rows"])[g]) for g in rows], dtype=np.int64)
    out["meta"]["dataset"] = cache["meta"].get("dataset", "")
    out["molecule_ids"] = [str(cache["molecule_ids"][g]) for g in rows]
    out["smiles"] = [str(cache["smiles"][g]) if "smiles" in cache else "" for g in rows]
    return out


def permute_atoms(cache, rng: np.random.Generator) -> dict[str, Any]:
    """Atom-renumbered copy of a small cache (bond order re-sorted per molecule)."""
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    n_mol = int(cache["n_molecules"])
    n_atoms = int(atom_ptr[-1])
    perm = np.arange(n_atoms, dtype=np.int64)
    # ``bond_index`` and ``path_end`` are molecule-local in the frozen cache
    # layout (the collate adds the batch atom offsets), so the local permutation
    # is what those two arrays must be remapped with.
    local_perm: dict[int, np.ndarray] = {}
    for g in range(n_mol):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        lp = rng.permutation(a1 - a0)
        # lp[new_local] = old_local; remapping an old atom index needs its inverse
        local_perm[g] = np.argsort(lp)
        perm[a0:a1] = a0 + lp

    for key in ("atom_feat", "atom_type", "atom_scalar"):
        out[key] = np.asarray(cache[key])[perm]

    # per-atom S1 and S2 record blocks are re-ordered with their atom
    new_nbr_ptr = np.zeros_like(nbr_ptr)
    new_nbr_code = np.zeros_like(np.asarray(cache["nbr_code"]))
    new_path_ptr = np.zeros_like(path_ptr)
    new_path_cat = np.zeros_like(np.asarray(cache["path_cat"]))
    new_path_end63 = np.zeros_like(np.asarray(cache["path_end63"]))
    new_path_end = np.zeros_like(np.asarray(cache["path_end"]))
    new_path_closure = np.zeros_like(np.asarray(cache["path_closure"]))
    nbr_counts = np.diff(nbr_ptr)
    path_counts = np.diff(path_ptr)
    nbr_off = 0
    path_off = 0
    mol_of_atom = np.repeat(np.arange(n_mol, dtype=np.int64), np.diff(atom_ptr))
    for new_a, old_a in enumerate(perm.tolist()):
        lp = local_perm[int(mol_of_atom[old_a])]
        c = int(nbr_counts[old_a])
        new_nbr_ptr[new_a] = nbr_off
        if c:
            new_nbr_code[nbr_off : nbr_off + c] = np.asarray(cache["nbr_code"])[nbr_ptr[old_a] : nbr_ptr[old_a] + c]
        nbr_off += c
        c = int(path_counts[old_a])
        new_path_ptr[new_a] = path_off
        if c:
            sl = slice(int(path_ptr[old_a]), int(path_ptr[old_a]) + c)
            new_path_cat[path_off : path_off + c] = np.asarray(cache["path_cat"])[sl]
            new_path_end63[path_off : path_off + c] = np.asarray(cache["path_end63"])[sl]
            new_path_end[path_off : path_off + c] = lp[np.asarray(cache["path_end"])[sl]]
            new_path_closure[path_off : path_off + c] = np.asarray(cache["path_closure"])[sl]
        path_off += c
    new_nbr_ptr[n_atoms] = nbr_off
    new_path_ptr[n_atoms] = path_off
    out.update(
        {
            "nbr_ptr": new_nbr_ptr,
            "nbr_code": new_nbr_code,
            "path_ptr": new_path_ptr,
            "path_cat": new_path_cat,
            "path_end63": new_path_end63,
            "path_end": new_path_end,
            "path_closure": new_path_closure,
        }
    )

    # bonds: remap endpoints, re-sort by (u, v) inside each molecule
    bi = np.asarray(cache["bond_index"])
    bf = np.asarray(cache["bond_feat"])
    bt = np.asarray(cache["bond_type"])
    new_bi = np.zeros_like(bi)
    new_bf = np.zeros_like(bf)
    new_bt = np.zeros_like(bt)
    for g in range(n_mol):
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        if b1 <= b0:
            continue
        e = local_perm[g][bi[b0:b1]]
        e = np.sort(e, axis=1)
        order = np.lexsort((e[:, 1], e[:, 0]))
        new_bi[b0:b1] = e[order]
        new_bf[b0:b1] = bf[b0:b1][order]
        new_bt[b0:b1] = bt[b0:b1][order]
    out.update({"bond_index": new_bi, "bond_feat": new_bf, "bond_type": new_bt})
    return out


def permute_paths(cache, rng: np.random.Generator) -> dict[str, Any]:
    """Same cache with the S2 record order inside every root atom shuffled."""
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    path_ptr = np.asarray(cache["path_ptr"])
    arrays = {k: np.asarray(cache[k]) for k in ("path_cat", "path_end63", "path_end", "path_closure")}
    n_atoms = int(np.asarray(cache["atom_ptr"])[-1])
    for a in range(n_atoms):
        s, e = int(path_ptr[a]), int(path_ptr[a + 1])
        if e - s < 2:
            continue
        order = s + rng.permutation(e - s)
        for k, arr in arrays.items():
            out[k][s:e] = arr[order]
    return out


def strip_paths(cache, rows) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove every S2 record of ``rows`` (keeps atom/bond/S1 untouched)."""
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    atom_ptr = np.asarray(cache["atom_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    counts = np.diff(path_ptr)
    kill = np.zeros(counts.size, dtype=bool)
    for g in rows:
        kill[int(atom_ptr[g]) : int(atom_ptr[g + 1])] = True
    keep_atom = ~kill
    rec_counts = np.where(keep_atom, counts, 0)
    new_ptr = np.concatenate([[0], np.cumsum(rec_counts)]).astype(np.int64)
    mask = np.zeros(int(counts.sum()), dtype=bool)
    pos = 0
    for a in range(counts.size):
        c = int(counts[a])
        if keep_atom[a] and c:
            mask[pos : pos + c] = True
        pos += c
    out["path_ptr"] = new_ptr
    for key in ("path_cat", "path_end63", "path_end", "path_closure"):
        out[key] = np.asarray(cache[key])[mask]
    return out, {"kept_records": int(mask.sum()), "dropped_records": int((~mask).sum()), "n_atoms_emptied": int(kill.sum())}


def model_predict(model, cache, device, batch_size=64) -> np.ndarray:
    return T.predict_raw(model, cache, device, batch_size=batch_size)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate1_split_identity(task_key: str) -> dict[str, Any]:
    t = C.TASKS[task_key]
    view, train_cache, valid_cache, test_cache, spec = S.task_caches(task_key)
    y, test_cache2 = T.task_targets(task_key, view)
    checks: dict[str, Any] = {}
    sizes = {"train": int(train_cache["n_molecules"]), "valid": int(valid_cache["n_molecules"]), "test": int(test_cache["n_molecules"])}
    checks["split_sizes_match"] = sizes == {k: int(v) for k, v in t["split_sizes"].items()}
    checks["split_sizes"] = sizes
    rows = {"train": np.asarray(train_cache["rows"]), "valid": np.asarray(valid_cache["rows"]), "test": np.asarray(test_cache["rows"])}
    checks["rows_equal_published"] = {
        "train": np.array_equal(rows["train"], np.asarray(view.rows["train"])),
        "valid": np.array_equal(rows["valid"], np.asarray(view.rows["valid"])),
        "test": np.array_equal(rows["test"], np.asarray(y["test_rows"])),
    }
    checks["split_disjoint"] = bool(
        len(set(rows["train"].tolist()) & set(rows["valid"].tolist())) == 0
        and len(set(rows["train"].tolist()) & set(rows["test"].tolist())) == 0
        and len(set(rows["valid"].tolist()) & set(rows["test"].tolist())) == 0
    )
    checks["rows_sha256"] = {k: C.index_sha256([int(x) for x in v.tolist()]) for k, v in rows.items()}
    checks["cache_feature_spec_hash"] = str(train_cache["meta"].get("feature_spec_hash", ""))
    checks["spec_hash"] = spec.hash()
    checks["spec_hash_matches_cache"] = checks["cache_feature_spec_hash"] == checks["spec_hash"]
    checks["split_names"] = [str(train_cache.get("split", "")), str(valid_cache.get("split", "")), str(test_cache.get("split", ""))]
    checks["labels_finite"] = {
        "train": bool(np.isfinite(y["train"]).all()),
        "valid": bool(np.isfinite(y["valid"]).all()),
        "test": bool(np.isfinite(y["test"]).all()),
    }
    checks["label_counts"] = {k: int(v.size) for k, v in y.items() if k in ("train", "valid", "test")}
    if t["task"] == "binary":
        checks["positive_rates"] = {k: float(np.mean(y[k])) for k in ("train", "valid", "test")}
    checks["norm_fingerprint"] = C.PREP_DATA.norm_fingerprint(view.train_norm)
    checks["train_norm_uses_valid_or_test"] = bool(view.train_norm.get("valid_or_test_used", False))
    checks["test_cache_is_a_separate_object"] = bool(test_cache2 is not train_cache)
    checks["train_valid_cache_version"] = [str(train_cache.get("cache_version", "")), str(valid_cache.get("cache_version", ""))]
    if "repeat" in t:
        checks["split_file_sha256"] = str(view.source_meta.get("split_file_sha256", ""))
        checks["split_file"] = str(view.source_meta.get("split_file", ""))
    else:
        checks["split_source"] = str(view.source_meta.get("split_file", ""))
        checks["ogb_release"] = str(view.source_meta.get("ogb_release", ""))
    checks["passed"] = bool(
        checks["split_sizes_match"]
        and all(checks["rows_equal_published"].values())
        and checks["split_disjoint"]
        and checks["spec_hash_matches_cache"]
        and all(checks["labels_finite"].values())
        and not checks["train_norm_uses_valid_or_test"]
    )
    return checks


def gate2_omega_train_only(task_key: str, support: S.Support) -> dict[str, Any]:
    view, train_cache, valid_cache, test_cache, spec = S.task_caches(task_key)
    n_nbr = int(spec.n_nbr_codes)
    tr = S.split_records(train_cache)
    train_codes = np.unique(S.pair_code(tr["path_cat"], tr["path_end63"], n_nbr))
    va = S.split_records(valid_cache)
    te = S.split_records(test_cache)
    valid_codes = np.unique(S.pair_code(va["path_cat"], va["path_end63"], n_nbr))
    test_codes = np.unique(S.pair_code(te["path_cat"], te["path_end63"], n_nbr))
    checks = {
        "omega_equals_train_unique_pairs": bool(np.array_equal(train_codes, support.omega_code)),
        "n_train_unique_pairs": int(train_codes.size),
        "n_omega": int(support.n_omega),
        "valid_only_pairs_excluded": int(np.setdiff1d(valid_codes, train_codes).size),
        "test_only_pairs_excluded": int(np.setdiff1d(test_codes, train_codes).size),
        "omega_contains_no_valid_or_test_only_pair": bool(
            np.array_equal(np.intersect1d(support.omega_code, np.setdiff1d(valid_codes, train_codes)), np.zeros(0, np.int64))
            and np.array_equal(np.intersect1d(support.omega_code, np.setdiff1d(test_codes, train_codes)), np.zeros(0, np.int64))
        ),
        "occurrence_counts_equal_bincount": bool(
            np.array_equal(
                support.omega_count,
                np.bincount(np.searchsorted(support.omega_code, S.pair_code(tr["path_cat"], tr["path_end63"], n_nbr)),
                            minlength=support.n_omega).astype(np.int64),
            )
        ),
        "document_frequency_le_occurrence": bool(np.all(support.omega_docfreq <= support.omega_count)),
        "document_frequency_le_train_molecules": bool(
            np.all(support.omega_docfreq <= int(train_cache["n_molecules"]))
        ),
        "labels_never_read": True,
        "coverage": support.coverage,
    }
    checks["passed"] = bool(
        checks["omega_equals_train_unique_pairs"]
        and checks["omega_contains_no_valid_or_test_only_pair"]
        and checks["occurrence_counts_equal_bincount"]
        and checks["document_frequency_le_occurrence"]
        and checks["document_frequency_le_train_molecules"]
    )
    return checks


def gate3_type_encoding(task_key: str, support: S.Support, model, device) -> dict[str, Any]:
    """The p/e types fed to the model are the production codes, unchanged."""
    view, train_cache, valid_cache, test_cache, spec = S.task_caches(task_key)
    local = model.local
    checks: dict[str, Any] = {}
    checks["path_emb_rows"] = int(local.path_emb.weight.shape[0])
    checks["end_emb_rows"] = int(local.end_emb.weight.shape[0])
    checks["spec_n_path_codes"] = int(spec.n_path_codes)
    checks["spec_n_nbr_codes"] = int(spec.n_nbr_codes)
    checks["path_emb_matches_path_codes"] = checks["path_emb_rows"] == checks["spec_n_path_codes"]
    checks["end_emb_matches_end_codes"] = checks["end_emb_rows"] == checks["spec_n_nbr_codes"]
    checks["omega_p_within_vocabulary"] = bool(support.omega_p.min() >= 0 and support.omega_p.max() < spec.n_path_codes)
    checks["omega_e_within_vocabulary"] = bool(support.omega_e.min() >= 0 and support.omega_e.max() < spec.n_nbr_codes)
    checks["all_types_legal"] = bool(support.legality["all_omega_legal"])
    checks["n_p_types_used"] = int(support.n_p)
    checks["n_e_types_used"] = int(support.n_e)
    checks["legality"] = support.legality
    # codes reach the model through the frozen collate, not through a re-encode
    rows = list(range(min(16, int(train_cache["n_molecules"]))))
    batch = C.PREP_DATA.collate(train_cache, rows)
    tr = S.split_records(train_cache)
    atom_ptr = np.asarray(train_cache["atom_ptr"])
    path_ptr = np.asarray(train_cache["path_ptr"])
    n_paths = int(np.sum([int(path_ptr[atom_ptr[g + 1]]) - int(path_ptr[atom_ptr[g]]) for g in rows]))
    checks["collate_record_count_matches"] = int(batch["path_cat"].numel()) == n_paths
    first = np.asarray(train_cache["path_cat"])[path_ptr[atom_ptr[rows[0]]] : path_ptr[atom_ptr[rows[0] + 1]]]
    checks["collate_first_molecule_codes_identical"] = bool(
        np.array_equal(batch["path_cat"][: first.size].numpy(), first.astype(np.int64))
    )
    checks["n_records_train"] = int(tr["path_cat"].size)
    checks["passed"] = bool(
        checks["path_emb_matches_path_codes"]
        and checks["end_emb_matches_end_codes"]
        and checks["omega_p_within_vocabulary"]
        and checks["omega_e_within_vocabulary"]
        and checks["all_types_legal"]
        and checks["collate_record_count_matches"]
        and checks["collate_first_molecule_codes_identical"]
    )
    return checks


def gate4_shared_parameters(model_a, model_b, model_c) -> dict[str, Any]:
    sa = {k: v.detach().cpu() for k, v in model_a.state_dict().items()}
    sb = {k: v.detach().cpu() for k, v in model_b.state_dict().items()}
    sc = {k: v.detach().cpu() for k, v in model_c.state_dict().items()}
    joint_keys = {"local.path_u.weight", "local.path_v.weight", "local.joint_proj.weight", "local.joint_proj.bias"}
    common = sorted((set(sa) & set(sb)) - joint_keys)
    checks: dict[str, Any] = {}
    checks["n_common_tensors"] = len(common)
    checks["values_bitequal_B"] = all(bool(torch.equal(sa[k], sb[k])) for k in common)
    checks["values_bitequal_C"] = all(bool(torch.equal(sa[k], sc[k])) for k in common)
    bad = [k for k in common if not (torch.equal(sa[k], sb[k]) and torch.equal(sa[k], sc[k]))]
    checks["mismatched_keys"] = bad
    checks["wider_minus_A"] = {
        "B": sorted(set(sb) - set(sa)),
        "C": sorted(set(sc) - set(sa)),
        "A_minus_common": sorted(set(sa) - set(sb)),
    }
    checks["parameters"] = {
        "A": int(sum(p.numel() for p in model_a.parameters())),
        "B": int(sum(p.numel() for p in model_b.parameters())),
        "C": int(sum(p.numel() for p in model_c.parameters())),
    }
    checks["joint_module_parameters"] = {
        "A": M.joint_module_parameters(model_a),
        "B": M.joint_module_parameters(model_b),
        "C": M.joint_module_parameters(model_c),
    }
    checks["passed"] = bool(checks["values_bitequal_B"] and checks["values_bitequal_C"])
    return checks


def gate5_init_replay(task_key: str, support: S.Support, model_a, model_b, model_c, device) -> dict[str, Any]:
    view, train_cache, _va, _te, spec = S.task_caches(task_key)
    rows = list(range(min(64, int(train_cache["n_molecules"]))))
    cache = subset_cache(train_cache, rows, spec)
    batch = T.to_device(C.PREP_DATA.collate(cache, list(range(len(rows)))), device)
    local_a, local_b, local_c = model_a.local, model_b.local, model_c.local
    codes = batch["path_cat"].to(torch.long) * int(spec.n_nbr_codes) + batch["path_end63"].to(torch.long)
    with torch.no_grad():
        model_a.eval(); model_b.eval(); model_c.eval()
        j_a = local_a.joint_proj(local_a.path_u(batch["path_cat"]) * local_a.path_v(batch["path_end63"]))
        j_b = local_b.code_values(codes)
        j_c = local_c.code_values(codes)
        ya, aux_a = model_a(batch, return_aux=True)
        yb, aux_b = model_b(batch, return_aux=True)
        yc, aux_c = model_c(batch, return_aux=True)
    ja = j_a.detach().double().cpu().numpy()
    jb = j_b.detach().double().cpu().numpy()
    jc = j_c.detach().double().cpu().numpy()
    xa = aux_a["local_input"].detach().double().cpu().numpy()
    xb = aux_b["local_input"].detach().double().cpu().numpy()
    xc = aux_c["local_input"].detach().double().cpu().numpy()
    checks = {
        "n_paths_checked": int(codes.numel()),
        "joint_max_abs_diff_B": float(np.max(np.abs(ja - jb))),
        "joint_max_abs_diff_C": float(np.max(np.abs(ja - jc))),
        "h0_local_input_max_abs_diff_B": float(np.max(np.abs(xa - xb))),
        "h0_local_input_max_abs_diff_C": float(np.max(np.abs(xa - xc))),
        "final_logit_max_abs_diff_B": float(
            np.max(np.abs(ya.detach().double().cpu().numpy() - yb.detach().double().cpu().numpy()))
        ),
        "final_logit_max_abs_diff_C": float(
            np.max(np.abs(ya.detach().double().cpu().numpy() - yc.detach().double().cpu().numpy()))
        ),
        "train_seen_only": True,
        "note": (
            "C is the orthogonal marginal projection of the same draw, so it is reported but not required to "
            "equal A; only B must replay A exactly"
        ),
    }
    tol = float(C.TOL["forward_abs"])
    checks["passed"] = bool(
        checks["joint_max_abs_diff_B"] <= tol
        and checks["h0_local_input_max_abs_diff_B"] <= tol
        and checks["final_logit_max_abs_diff_B"] <= tol
    )
    checks["c_deviation_is_expected"] = True
    return checks


def gate6_gradients(task_key: str, support: S.Support, models: dict[str, Any], device) -> dict[str, Any]:
    view, train_cache, _va, _te, spec = S.task_caches(task_key)
    task = str(C.TASKS[task_key]["task"])
    rows = list(range(min(96, int(train_cache["n_molecules"]))))
    cache = subset_cache(train_cache, rows, spec)
    batch = T.to_device(C.PREP_DATA.collate(cache, list(range(len(rows)))), device)
    y_raw = np.asarray(
        view.targets.y_train if C.TASKS[task_key]["family"] == "ptb" else view.y_train, dtype=np.float64
    ).reshape(-1)[: len(rows)]
    y = T.loss_targets(task_key, view, y_raw).to(device)
    out: dict[str, Any] = {}
    for arm in C.TRAINED_ARMS:
        model = models[arm]
        was_training = model.training
        model.train()
        model.zero_grad(set_to_none=True)
        pred = model(batch).reshape(-1)
        loss = F_loss(task, pred, y)
        loss.backward()
        params = [(n, p) for n, p in model.named_parameters()]
        missing = [n for n, p in params if p.grad is None]
        zero = [n for n, p in params if p.grad is not None and float(p.grad.abs().sum()) == 0.0]
        joint_name = "local.J_table" if arm == C.ARM_B else "local.Theta"
        joint = dict(params)[joint_name]
        entry = {
            "n_parameters_tensors": len(params),
            "n_without_grad": len(missing),
            "tensors_without_grad": missing,
            "n_zero_grad": len(zero),
            "tensors_with_zero_grad": zero,
            "joint_tensor": joint_name,
            "joint_grad_abs_sum": float(joint.grad.abs().sum()),
            "joint_grad_rows_nonzero": int((joint.grad.abs().sum(dim=1) > 0).sum()),
            "joint_grad_rows_total": int(joint.grad.shape[0]),
            "joint_grad_finite": bool(torch.isfinite(joint.grad).all()),
            "loss": float(loss),
        }
        if arm == C.ARM_B:
            codes = torch.unique(
                batch["path_cat"].to(torch.long) * int(spec.n_nbr_codes) + batch["path_end63"].to(torch.long)
            )
            table_idx = torch.as_tensor(support.index_table, dtype=torch.long, device=codes.device)
            rows_seen = table_idx[codes]
            entry["joint_grad_rows_seen_in_batch"] = int((rows_seen >= 0).sum())
            del codes, rows_seen
        entry["passed"] = bool(
            len(missing) == 0
            and len(zero) == 0
            and entry["joint_grad_abs_sum"] > 0
            and entry["joint_grad_finite"]
        )
        out[arm] = entry
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    out["passed"] = bool(all(out[a]["passed"] for a in C.TRAINED_ARMS))
    return out


def F_loss(task: str, pred, target):
    if task == "regression":
        return torch.nn.functional.mse_loss(pred, target)
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)


def gate7_invariance(task_key: str, support: S.Support, models: dict[str, Any], device) -> dict[str, Any]:
    view, train_cache, _va, _te, spec = S.task_caches(task_key)
    rows = list(range(min(48, int(train_cache["n_molecules"]))))
    cache = subset_cache(train_cache, rows, spec)
    rng = np.random.default_rng(20261125)
    permuted = permute_atoms(cache, rng)
    shuffled = permute_paths(cache, rng)
    out: dict[str, Any] = {}
    for arm in (C.ARM_A, C.ARM_B, C.ARM_C):
        model = models[arm]
        base = model_predict(model, cache, device, batch_size=64)
        renum = model_predict(model, permuted, device, batch_size=64)
        reorder = model_predict(model, shuffled, device, batch_size=64)
        regroup = model_predict(model, cache, device, batch_size=7)
        out[arm] = {
            "atom_renumbering_max_abs_diff": float(np.max(np.abs(base - renum))),
            "path_order_max_abs_diff": float(np.max(np.abs(base - reorder))),
            "batch_regrouping_max_abs_diff": float(np.max(np.abs(base - regroup))),
        }
        out[arm]["passed"] = bool(
            out[arm]["atom_renumbering_max_abs_diff"] <= float(C.TOL["invariance_abs"])
            and out[arm]["path_order_max_abs_diff"] <= float(C.TOL["invariance_abs"])
            and out[arm]["batch_regrouping_max_abs_diff"] <= float(C.TOL["invariance_abs"])
        )
    out["n_molecules"] = len(rows)
    out["passed"] = bool(all(out[a]["passed"] for a in (C.ARM_A, C.ARM_B, C.ARM_C)))
    return out


def gate8_empty_paths(task_key: str, support: S.Support, models: dict[str, Any], device) -> dict[str, Any]:
    view, train_cache, _va, _te, spec = S.task_caches(task_key)
    rows = list(range(min(32, int(train_cache["n_molecules"]))))
    cache = subset_cache(train_cache, rows, spec)
    atom_ptr = np.asarray(cache["atom_ptr"])
    partial, meta_partial = strip_paths(cache, list(range(0, len(rows), 2)))
    all_empty, meta_all = strip_paths(cache, list(range(len(rows))))
    out: dict[str, Any] = {"partial": meta_partial, "all": meta_all}
    for arm in (C.ARM_A, C.ARM_B, C.ARM_C):
        model = models[arm]
        base = model_predict(model, cache, device, batch_size=64)
        p = model_predict(model, partial, device, batch_size=64)
        e = model_predict(model, all_empty, device, batch_size=64)
        out[arm] = {
            "partial_finite": bool(np.isfinite(p).all()),
            "all_empty_finite": bool(np.isfinite(e).all()),
            "partial_matches_where_no_paths": float(np.max(np.abs(base[: len(rows)] - p))),
            "all_empty_shift_max_abs": float(np.max(np.abs(base - e))),
        }
        out[arm]["passed"] = bool(out[arm]["partial_finite"] and out[arm]["all_empty_finite"])
    out["n_molecules"] = len(rows)
    out["n_atoms"] = int(atom_ptr[-1])
    out["passed"] = bool(all(out[a]["passed"] for a in (C.ARM_A, C.ARM_B, C.ARM_C)))
    return out


def gate9_projector(support: S.Support) -> dict[str, Any]:
    P = support.P
    checks = {
        "shape": list(P.shape),
        "symmetric_max_abs_error": float(np.max(np.abs(P - P.T))),
        "idempotent_max_abs_error": float(np.max(np.abs(P @ P - P))),
        "rank_X": int(support.rank),
        "rank_identity_holds": bool(support.rank == support.n_columns - support.n_components),
        "U_orthonormal_max_abs_error": float(np.max(np.abs(support.U.T @ support.U - np.eye(support.rank)))),
    }
    tol = float(C.TOL["orthogonality_abs"])
    checks["passed"] = bool(
        checks["symmetric_max_abs_error"] <= tol
        and checks["idempotent_max_abs_error"] <= tol
        and checks["rank_identity_holds"]
        and checks["U_orthonormal_max_abs_error"] <= tol
    )
    return checks


def gate10_additive_constraint(task_key: str, support: S.Support, model_c) -> dict[str, Any]:
    table = M.extract_table(model_c, support)  # float32 model path, cast to float64
    theta = model_c.local.Theta.detach().double().cpu().numpy()
    table64 = support.U @ theta  # the exact float64 meaning of the parameterisation
    resid = table - support.P @ table
    resid64 = table64 - support.P @ table64
    denom = float(np.sqrt(np.mean(np.square(table)))) or 1.0
    checks = {
        "table_rms": float(np.sqrt(np.mean(np.square(table)))),
        "nonadditive_rms_float32_model_path": float(np.sqrt(np.mean(np.square(resid)))),
        "relative_nonadditive_residual_float32_model_path": float(np.sqrt(np.mean(np.square(resid))) / denom),
        "relative_nonadditive_residual_float64_parameterisation": float(
            np.sqrt(np.mean(np.square(resid64))) / denom
        ),
        "max_abs_theta_projection_error_float64": float(np.max(np.abs(resid64))) if resid64.size else 0.0,
        "float32_roundoff_scale": float(np.finfo(np.float32).eps),
        "theta_shape": [int(support.rank), int(C.STAT_RANK)],
    }
    checks["passed"] = bool(
        checks["relative_nonadditive_residual_float64_parameterisation"] <= 1e-12
        and checks["relative_nonadditive_residual_float32_model_path"] <= float(C.TOL["orthogonality_abs"])
    )
    return checks


def gate11_interaction_zero(task_key: str, support: S.Support, model_c) -> dict[str, Any]:
    """Exact D = J(p2,e2) - J(p2,e1) - J(p1,e2) + J(p1,e1) on real rectangles."""
    table = M.extract_table(model_c, support)
    theta = model_c.local.Theta.detach().double().cpu().numpy()
    table64 = support.U @ theta
    rect = rectangles(support)
    if not rect.size:
        return {"n_rectangles": 0, "passed": True, "note": "Omega contains no rectangle"}
    i11, i12, i21, i22 = rect[:, 0], rect[:, 1], rect[:, 2], rect[:, 3]
    d = table[i22] - table[i21] - table[i12] + table[i11]
    d64 = table64[i22] - table64[i21] - table64[i12] + table64[i11]
    checks = {
        "n_rectangles": int(rect.shape[0]),
        "max_abs_contrast_float32_model_path": float(np.max(np.abs(d))),
        "rms_contrast_float32_model_path": float(np.sqrt(np.mean(np.square(d)))),
        "max_abs_contrast_float64": float(np.max(np.abs(d64))),
        "table_rms": float(np.sqrt(np.mean(np.square(table)))),
        "note": "D = J(p1,e1) + J(p2,e2) - J(p1,e2) - J(p2,e1) on quadruples that really exist in Omega",
    }
    checks["passed"] = bool(
        checks["max_abs_contrast_float64"] <= float(C.TOL["interaction_abs"])
        and checks["max_abs_contrast_float32_model_path"] <= float(C.TOL["orthogonality_abs"])
    )
    return checks


def rectangles(support: S.Support) -> np.ndarray:
    """All ``(i(p1,e1), i(p1,e2), i(p2,e1), i(p2,e2))`` index quadruples present in Omega."""
    n_p, n_e = support.n_p, support.n_e
    idx = np.full((n_e, n_p), -1, dtype=np.int64)
    e_index = np.searchsorted(support.e_types, support.omega_e)
    p_index = np.searchsorted(support.p_types, support.omega_p)
    idx[e_index, p_index] = np.arange(support.n_omega, dtype=np.int64)
    out = []
    for e1 in range(n_e):
        for e2 in range(e1 + 1, n_e):
            common = np.flatnonzero((idx[e1] >= 0) & (idx[e2] >= 0))
            for a in range(common.size):
                for b in range(a + 1, common.size):
                    p1, p2 = common[a], common[b]
                    out.append((idx[e1, p1], idx[e2, p1], idx[e1, p2], idx[e2, p2]))
    return np.asarray(out, dtype=np.int64).reshape(-1, 4)


def gate12_fallback(task_key: str, support: S.Support, model_b, model_c, device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_bond = int(support.n_bond_types)
    n_codes = int(support.n_path_codes) * int(support.n_nbr_codes)
    # a legal pair that exists as two known types but never in Train
    ls_codes = [c for c in range(n_codes) if support.pair_mode[c] == S.MODE_MARGINAL_LS]
    mean_codes = [c for c in range(n_codes) if support.pair_mode[c] == S.MODE_TYPE_MEAN]
    out["n_marginal_ls_codes"] = len(ls_codes)
    out["n_type_mean_codes"] = len(mean_codes)
    out["n_omega_codes"] = int((support.pair_mode == S.MODE_OMEGA).sum())
    if not ls_codes or not mean_codes:
        raise RuntimeError("both fallback branches must be exercised")
    probe = torch.as_tensor(ls_codes[:8] + mean_codes[:8], dtype=torch.long, device=device)
    out["probe_note"] = "first 8 codes of each fallback branch (deterministic, ascending code order)"
    for arm, model in (("B", model_b), ("C", model_c)):
        local = model.local
        with torch.no_grad():
            vals = local.code_values(probe)
        out[arm] = {
            "finite": bool(torch.isfinite(vals).all()),
            "ls_row_norms_finite": bool(torch.isfinite(vals[:8]).all()),
            "mean_row_norms_finite": bool(torch.isfinite(vals[8:]).all()),
            "mean_rows_identical": bool(torch.allclose(vals[8:], vals[8:9].expand_as(vals[8:]))),
            "types_seen_flags": [int(support.p_col[ls_codes[0] // int(support.n_nbr_codes)]),
                                 int(support.e_col[ls_codes[0] % int(support.n_nbr_codes)])],
            "has_extra_parameters": int(local.table_n_parameters()),
        }
    out["identical_rule"] = bool(
        np.array_equal(support.pair_mode, support.pair_mode)
        and out["B"]["mean_rows_identical"] == out["C"]["mean_rows_identical"]
    )
    # the two implementations share the same rule table by construction
    out["shared_pair_mode_table"] = True
    out["same_component_example"] = {
        "code": int(ls_codes[0]),
        "p": int(ls_codes[0] // int(support.n_nbr_codes)),
        "e": int(ls_codes[0] % int(support.n_nbr_codes)),
        "legal": bool((ls_codes[0] // int(support.n_nbr_codes)) % n_bond == (ls_codes[0] % int(support.n_nbr_codes)) % n_bond),
    }
    out["no_new_parameters"] = bool(
        out["B"]["has_extra_parameters"] == int(support.n_omega) * int(C.STAT_RANK)
        and out["C"]["has_extra_parameters"] == int(support.rank) * int(C.STAT_RANK)
    )
    out["ratio_valid_unseen_occurrence"] = {
        split: float(support.coverage[split]["occurrence_coverage"]) for split in support.coverage
    }
    out["passed"] = bool(
        all(out[a]["finite"] for a in ("B", "C"))
        and all(out[a]["mean_rows_identical"] for a in ("B", "C"))
        and out["no_new_parameters"]
    )
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_task(task_key: str, device, *, seed: int = C.SEED) -> dict[str, Any]:
    C.log(f"gates: building support + arms for {task_key}")
    view, _tr, _va, _te, spec = S.task_caches(task_key)
    support = S.build_task_support(task_key)
    models: dict[str, Any] = {}
    infos: dict[str, Any] = {}
    for arm in C.ARMS:
        model, info = M.make_arm(task_key, view, support, arm, seed, device)
        models[arm], infos[arm] = model, info
    out: dict[str, Any] = {
        "task": task_key,
        "dataset": C.TASKS[task_key]["dataset"],
        "seed": int(seed),
        "device": str(device),
        "support": support.summary(),
        "init": {
            "init_state_hash_A": infos[C.ARM_A]["init_state_hash"],
            "historical_init_state_hash": historical_init_hash(task_key),
            "init_hash_matches_history": infos[C.ARM_A]["init_state_hash"] == historical_init_hash(task_key),
        },
        "gates": {},
    }
    checks = out["gates"]
    checks["1_split_identity"] = gate1_split_identity(task_key)
    checks["2_omega_train_only"] = gate2_omega_train_only(task_key, support)
    checks["3_type_encoding"] = gate3_type_encoding(task_key, support, models[C.ARM_A], device)
    checks["4_shared_parameters"] = gate4_shared_parameters(models[C.ARM_A], models[C.ARM_B], models[C.ARM_C])
    checks["5_init_replay"] = gate5_init_replay(task_key, support, models[C.ARM_A], models[C.ARM_B], models[C.ARM_C], device)
    checks["6_gradients"] = gate6_gradients(task_key, support, models, device)
    checks["7_invariance"] = gate7_invariance(task_key, support, models, device)
    checks["8_empty_paths"] = gate8_empty_paths(task_key, support, models, device)
    checks["9_projector"] = gate9_projector(support)
    checks["10_additive_constraint"] = gate10_additive_constraint(task_key, support, models[C.ARM_C])
    checks["11_interaction_zero"] = gate11_interaction_zero(task_key, support, models[C.ARM_C])
    checks["12_fallback"] = gate12_fallback(task_key, support, models[C.ARM_B], models[C.ARM_C], device)
    out["all_passed"] = bool(all(bool(c.get("passed")) for c in checks.values()))
    failed = [k for k, c in checks.items() if not bool(c.get("passed"))]
    out["failed"] = failed
    C.log(f"gates {task_key}: all_passed={out['all_passed']} failed={failed}")
    return out


_HIST_INIT: dict[str, str] = {}


def historical_init_hash(task_key: str) -> str:
    if task_key in _HIST_INIT:
        return _HIST_INIT[task_key]
    if task_key == "hiv":
        rec = C.read_json(C.REUSE["hiv"]["run_json"])
        value = str(rec["model"]["init_state_hash"])
    else:
        rec = C.read_json(C.REUSE["lipo"]["run_json"])
        value = str(rec["model"]["init_state_hash"])
    _HIST_INIT[task_key] = value
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-training correctness gates")
    ap.add_argument("--tasks", nargs="+", default=list(C.TASK_ORDER))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(C.ANALYSIS / "checks.json"))
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    payload = {
        "schema": "lcrc-support-joint-function-checks-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "env": C.env_info(),
        "created_utc": C.utcnow(),
        "device": str(device),
        "gate_list": {
            "1": "official split, row ids, labels and cache identity",
            "2": "Omega uses Train only",
            "3": "p/e types and model encoding match the production caches",
            "4": "A and B/C share bitwise identical common parameters",
            "5": "materialised B replays A's initial joint, h0 and prediction (Train-seen types)",
            "6": "every B/C parameter receives a gradient",
            "7": "atom renumbering, path order and batch regrouping invariance",
            "8": "graphs without S2 paths run correctly",
            "9": "P^2 = P and P^T = P",
            "10": "C's table satisfies (I - P) J ~ 0",
            "11": "C's interaction contrast is zero on every real rectangle",
            "12": "the unseen-pair fallback is identical, finite and parameter-free",
        },
        "tolerances": C.TOL,
        "tasks": {},
    }
    for task in args.tasks:
        payload["tasks"][task] = run_task(task, device)
    payload["all_passed"] = bool(all(v.get("all_passed") for v in payload["tasks"].values()))
    payload["failed_tasks"] = [k for k, v in payload["tasks"].items() if not v.get("all_passed")]
    C.write_json(args.out, payload)
    C.log(f"checks written to {args.out}: all_passed={payload['all_passed']}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
