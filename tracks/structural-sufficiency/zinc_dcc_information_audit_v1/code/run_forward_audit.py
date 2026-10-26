#!/usr/bin/env python
"""A1: forward audit of the frozen DCC-L computation.

CPU only, no gradient step, no official valid/test cache.  Captures a, b, w,
c_glob, c_loc, the 513-d reader input z and the prediction; verifies grouping,
slice layout, reader parity and structural invariances; reports non-degeneracy
spectra for at least two frozen seeds.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_common as AC  # noqa: E402
import joint_stats as JS  # noqa: E402

ROWS_SEED = 20261026
N_AUDIT_ROWS = 1000


def _t(x) -> np.ndarray:
    return x.detach().cpu().numpy()


def _maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))) if np.size(a) else 0.0


def _molecule_batch(cache, row: int) -> dict:
    return AC.collate_rows(cache, [row], device="cpu")


def _clone_batch(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def atom_relabel(batch: dict, perm: np.ndarray) -> dict:
    """New atom i is old atom perm[i]."""
    out = _clone_batch(batch)
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    inv = torch.empty_like(perm_t)
    inv[perm_t] = torch.arange(len(perm), dtype=torch.long)
    out["atom_feat"] = batch["atom_feat"][perm_t]
    out["atom_type"] = batch["atom_type"][perm_t]
    out["bond_index"] = inv[batch["bond_index"].long()]
    out["junc_index"] = inv[batch["junc_index"].long()]
    return out


def bond_permutation(batch: dict, perm: np.ndarray) -> dict:
    """New bond i is old bond perm[i]."""
    out = _clone_batch(batch)
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    inv = torch.empty_like(perm_t)
    inv[perm_t] = torch.arange(len(perm), dtype=torch.long)
    out["bond_feat"] = batch["bond_feat"][perm_t]
    out["bond_type"] = batch["bond_type"][perm_t]
    out["bond_index"] = batch["bond_index"][perm_t]
    out["junc_bond"] = inv[batch["junc_bond"].long()]
    return out


def endpoint_reversal(batch: dict) -> dict:
    out = _clone_batch(batch)
    out["bond_index"] = batch["bond_index"][:, [1, 0]]
    out["junc_index"] = batch["junc_index"][:, [2, 1, 0]]
    out["junc_bond"] = batch["junc_bond"][:, [1, 0]]
    return out


def independent_grouping(model, batch, cap):
    """Recompute Wg/Vg/Wl/Vl from captured w with an independent index_add."""
    w = cap["w"]
    jb = batch["junc_bond"].long()
    n_b = torch.zeros(int(batch["bond_feat"].shape[0]), dtype=w.dtype, device=w.device)
    n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), dtype=w.dtype, device=w.device))

    Wl = torch.zeros((int(batch["bond_feat"].shape[0]), w.shape[1]), dtype=w.dtype, device=w.device)
    Vl = torch.zeros_like(Wl)
    Wl.index_add_(0, jb[:, 0], w)
    Wl.index_add_(0, jb[:, 1], w)
    w2 = w * w
    Vl.index_add_(0, jb[:, 0], w2)
    Vl.index_add_(0, jb[:, 1], w2)

    B = int(batch["n_graphs"])
    w_sum = torch.zeros((B, w.shape[1]), dtype=w.dtype, device=w.device).index_add_(0, batch["junc_batch"], w)
    w2_sum = torch.zeros((B, w.shape[1]), dtype=w.dtype, device=w.device).index_add_(0, batch["junc_batch"], w2)
    cnt = torch.bincount(batch["junc_batch"], minlength=B).clamp(min=1).to(w.dtype)
    mean_w = w_sum / cnt.unsqueeze(1)
    mean_w2 = w2_sum / cnt.unsqueeze(1)
    Wg = n_b.unsqueeze(1) * mean_w[batch["bond_batch"]]
    Vg = n_b.unsqueeze(1) * mean_w2[batch["bond_batch"]]
    return Wg, Vg, Wl, Vl, n_b


def main() -> int:
    AC.install_load_guard()
    cfg = AC.load_audit_config()
    cache = AC.load_train_cache()
    norm = AC.load_norm()
    rng = np.random.default_rng(ROWS_SEED)
    rows = np.sort(rng.choice(int(cache["n_molecules"]), N_AUDIT_ROWS, replace=False)).astype(np.int64)
    AC.ANALYSIS.mkdir(parents=True, exist_ok=True)

    provenance = {
        "track": "zinc-dcc-information-audit-v1",
        "frozen_at_utc": cfg["frozen_at_utc"],
        "repo_head": cfg["repo_head"],
        "audit_plan_sha256": cfg["audit_plan_sha256"],
        "config_sha256": AC.sha256_file(AC.CONFIGS / "audit_v1.json"),
        "train_cache": str(AC.TRAIN_CACHE),
        "train_cache_sha256": AC.sha256_file(AC.TRAIN_CACHE),
        "train_cache_meta": cache.get("meta"),
        "norm_path": str(AC.NORM_PATH),
        "norm_sha256": AC.sha256_file(AC.NORM_PATH),
        "norm_source": AC.read_json(AC.NORM_PATH).get("source"),
        "p3_objects_sha256": AC.sha256_file(AC.P3_OBJECTS) if AC.P3_OBJECTS.exists() else None,
        "p3_prep_sha256": AC.sha256_file(AC.P3_PREP) if AC.P3_PREP.exists() else None,
        "rows_seed": ROWS_SEED,
        "rows": rows.tolist(),
        "rows_sha256": hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest(),
        "n_audit_rows": int(len(rows)),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "bond_type_observed": np.unique(np.asarray(cache["bond_type"])).tolist(),
    }
    AC.write_json(AC.ANALYSIS / "provenance.json", provenance)

    audit = {"protocol": cfg["protocol_id"], "checkpoints": {}, "parity": {}}
    nondeg = {"checkpoints": {}}

    # rows for invariance: a handful of real molecules of varying size
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    sizes = ap[1:] - ap[:-1]
    inv_rows = [int(np.where(sizes == q)[0][0]) for q in (8, 17, 25, 33) if np.any(sizes == q)]
    inv_rows = inv_rows[:4]

    for name in ("D0", "D1"):
        model, info = AC.load_checkpoint(name, norm, device="cpu", organized_local=True)
        model.eval()
        with torch.no_grad():
            # capture on the audit rows in batches
            preds = []
            caps_acc = {}
            parity_max = 0.0
            grouping = {}
            slice_info = {}
            for start in range(0, len(rows), 128):
                part = rows[start:start + 128]
                batch = AC.collate_rows(cache, part, device="cpu")
                cap = AC.capture_forward(model, batch)
                direct = model(batch)
                parity_max = max(parity_max, _maxdiff(_t(cap["pred"]), _t(direct)))
                # independent grouping on the first batch only
                if start == 0:
                    Wg, Vg, Wl, Vl, n_b = independent_grouping(model, batch, cap)
                    grouping = {
                        "max_abs_Wg": _maxdiff(_t(Wg), _t(cap["Wg"])),
                        "max_abs_Vg": _maxdiff(_t(Vg), _t(cap["Vg"])),
                        "max_abs_Wl": _maxdiff(_t(Wl), _t(cap["Wl"])),
                        "max_abs_Vl": _maxdiff(_t(Vl), _t(cap["Vl"])),
                        "max_abs_n_b": _maxdiff(_t(n_b), _t(cap["n_b"])),
                        "n_bonds": int(batch["bond_feat"].shape[0]),
                        "n_junctions": int(batch["junc_feat"].shape[0]),
                    }
                    # independent c_glob / c_loc composers
                    cg = model.group_composer(torch.cat(
                        [cap["b"], Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1))
                    cl = model.local_composer(torch.cat(
                        [cap["b"], Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1))
                    grouping["max_abs_c_glob"] = _maxdiff(_t(cg), _t(cap["c_glob"]))
                    grouping["max_abs_c_loc"] = _maxdiff(_t(cl), _t(cap["c_loc"]))
                    # slice layout
                    z = cap["z"]
                    slices = {
                        "z_glob": (0, 96), "a_sum": (96, 192), "a_sq": (192, 288),
                        "g": (288, 321), "zl": (321, 417), "zl2": (417, 513),
                    }
                    for key, (lo, hi) in slices.items():
                        slice_info[key] = {
                            "shape": list(z[:, lo:hi].shape),
                            "mean_abs": float(z[:, lo:hi].abs().mean()),
                            "std": float(z[:, lo:hi].std()),
                            "max_abs": float(z[:, lo:hi].abs().max()),
                        }
                    slice_info["reassembled_maxdiff"] = _maxdiff(
                        _t(z),
                        _t(torch.cat([cap["z_glob"], cap["a_sum"], cap["a_sq"], cap["g"],
                                      cap["zl"], cap["zl2"]], dim=1)),
                    )
                    slice_info["zl_matches_sum_c_loc"] = _maxdiff(_t(cap["zl"]), _t(cap["c_loc"].new_zeros(
                        (int(batch["n_graphs"]), cap["c_loc"].shape[1])).index_add_(
                        0, batch["bond_batch"], cap["c_loc"])))
                    slice_info["zl2_matches_sum_c_loc_sq"] = _maxdiff(_t(cap["zl2"]), _t(cap["c_loc"].new_zeros(
                        (int(batch["n_graphs"]), cap["c_loc"].shape[1])).index_add_(
                        0, batch["bond_batch"], cap["c_loc"] * cap["c_loc"])))
                    slice_info["reader_matches_pred"] = _maxdiff(
                        _t(model.reader(z).view(-1)), _t(cap["pred"]))
                preds.append(_t(cap["pred"]))
                for k, v in cap.items():
                    caps_acc.setdefault(k, []).append(_t(v))
            caps = {k: np.concatenate(v, axis=0) for k, v in caps_acc.items()}
            pred = np.concatenate(preds, axis=0)

            # invariance tests
            inv = {}
            for r in inv_rows:
                b0 = _molecule_batch(cache, r)
                with torch.no_grad():
                    p0 = float(model(b0).detach().cpu().numpy()[0])
                    n_atoms = int(b0["atom_feat"].shape[0])
                    n_bonds = int(b0["bond_feat"].shape[0])
                    n_junc = int(b0["junc_feat"].shape[0])
                    rng_a = np.random.default_rng(1000 + r)
                    perm_a = rng_a.permutation(n_atoms)
                    perm_b = rng_a.permutation(n_bonds)
                    variants = {
                        "atom_relabel": atom_relabel(b0, perm_a),
                        "bond_permutation": bond_permutation(b0, perm_b),
                        "endpoint_reversal": endpoint_reversal(b0),
                    }
                    # batch regrouping: same molecule with a companion, both orders
                    comp = _molecule_batch(cache, int(inv_rows[(inv_rows.index(r) + 1) % len(inv_rows)]))
                    # merge two single-molecule batches manually
                    def merge(b1, b2):
                        out = {}
                        for k in b1:
                            if k == "n_graphs":
                                out[k] = 2
                            elif torch.is_tensor(b1[k]):
                                if k in ("atom_batch", "bond_batch", "junc_batch"):
                                    out[k] = torch.cat([b1[k], b2[k] + int(b1[k].max()) + 1])
                                elif k in ("bond_index", "junc_index", "junc_bond"):
                                    continue
                                elif k == "global":
                                    out[k] = torch.cat([b1[k], b2[k]], dim=0)
                                else:
                                    out[k] = torch.cat([b1[k], b2[k]], dim=0)
                            else:
                                out[k] = b1[k]
                        # rebuild local indices for the concatenated batch
                        a_off = 0
                        b_off = 0
                        out["bond_index"] = torch.cat([b1["bond_index"], b2["bond_index"] + int(b1["atom_feat"].shape[0])], dim=0)
                        out["junc_index"] = torch.cat([b1["junc_index"], b2["junc_index"] + int(b1["atom_feat"].shape[0])], dim=0)
                        out["junc_bond"] = torch.cat([b1["junc_bond"], b2["junc_bond"] + int(b1["bond_feat"].shape[0])], dim=0)
                        # recompute batch vectors for the concatenated layout
                        out["atom_batch"] = torch.cat([torch.zeros(int(b1["atom_feat"].shape[0]), dtype=torch.long),
                                                       torch.ones(int(b2["atom_feat"].shape[0]), dtype=torch.long)])
                        out["bond_batch"] = torch.cat([torch.zeros(int(b1["bond_feat"].shape[0]), dtype=torch.long),
                                                       torch.ones(int(b2["bond_feat"].shape[0]), dtype=torch.long)])
                        out["junc_batch"] = torch.cat([torch.zeros(int(b1["junc_feat"].shape[0]), dtype=torch.long),
                                                       torch.ones(int(b2["junc_feat"].shape[0]), dtype=torch.long)])
                        return out
                    merged = merge(b0, comp)
                    p_merged = float(model(merged).detach().cpu().numpy()[0])
                    variants["batch_regrouping"] = merged
                    row = {"n_atoms": n_atoms, "n_bonds": n_bonds, "n_junctions": n_junc,
                           "pred_base": p0}
                    for k, v in variants.items():
                        with torch.no_grad():
                            pv = float(model(v).detach().cpu().numpy()[0])
                        row[k] = {"pred": pv, "abs_delta": abs(pv - p0)}
                    inv[str(int(r))] = row
            audit["checkpoints"][name] = {
                "checkpoint": info,
                "parity_max_abs_diff": parity_max,
                "grouping": grouping,
                "slices": slice_info,
                "invariance": inv,
                "finite": bool(np.isfinite(pred).all()),
                "pred_mean": float(pred.mean()),
                "pred_std": float(pred.std()),
            }

            # non-degeneracy
            nd = {}
            nd["a"] = AC.spectrum_summary(caps["a"])
            nd["b"] = AC.spectrum_summary(caps["b"])
            nd["w"] = AC.spectrum_summary(caps["w"])
            nd["c_glob"] = AC.spectrum_summary(caps["c_glob"])
            nd["c_loc"] = AC.spectrum_summary(caps["c_loc"])
            nd["z"] = AC.spectrum_summary(caps["z"])
            audit_bond_ptr = np.concatenate([[0], np.cumsum([
                int(bp[int(r) + 1] - bp[int(r)]) for r in rows
            ])]).astype(np.int64)
            nd["b_within"] = AC.within_molecule_bond_summary(caps["b"], audit_bond_ptr, max_molecules=len(rows))
            nd["c_glob_within"] = AC.within_molecule_bond_summary(caps["c_glob"], audit_bond_ptr, max_molecules=len(rows))
            nd["c_loc_within"] = AC.within_molecule_bond_summary(caps["c_loc"], audit_bond_ptr, max_molecules=len(rows))
            nd["b_within_pooled"] = AC.pooled_within_spectrum(caps["b"], audit_bond_ptr)
            nd["c_glob_within_pooled"] = AC.pooled_within_spectrum(caps["c_glob"], audit_bond_ptr)
            nd["c_loc_within_pooled"] = AC.pooled_within_spectrum(caps["c_loc"], audit_bond_ptr)
            nd["a_constant_channel_fraction"] = AC.constant_channel_fraction(caps["a"])
            nd["b_constant_channel_fraction"] = AC.constant_channel_fraction(caps["b"])
            nd["w_constant_channel_fraction"] = AC.constant_channel_fraction(caps["w"])
            nd["c_glob_constant_channel_fraction"] = AC.constant_channel_fraction(caps["c_glob"])
            nd["c_loc_constant_channel_fraction"] = AC.constant_channel_fraction(caps["c_loc"])
            nd["z_constant_channel_fraction"] = AC.constant_channel_fraction(caps["z"])
            # in-molecule c_loc variation for the exact 4 relation stats
            rel_rows = rows[:200]
            jl = []
            for i in rel_rows:
                b = np.asarray(cache["bond_index"])[bp[i]:bp[i+1]]
                n = int(ap[i+1] - ap[i])
                rel = JS.molecule_relations(b, n)
                # caps indexing is by audit-row order, not molecule id; find position
                pos = int(np.where(rows == i)[0][0])
                off = int(np.sum([bp[int(r) + 1] - bp[int(r)] for r in rows[:pos]]))
                c = caps["c_loc"][off:off + int(bp[int(i) + 1] - bp[int(i)])]
                jl.append(JS.j_latent(c, rel))
            jl = np.asarray(jl)
            nd["j_latent_sample"] = {
                "n": int(len(jl)),
                "mean": jl.mean(axis=0).tolist(),
                "std": jl.std(axis=0).tolist(),
                "per_dim_nonzero_fraction": (np.abs(jl) > 1e-12).mean(axis=0).tolist(),
            }
            nondeg["checkpoints"][name] = nd

    AC.write_json(AC.ANALYSIS / "forward_audit.json", AC.jsonable(audit))
    AC.write_json(AC.ANALYSIS / "nondegeneracy.json", AC.jsonable(nondeg))
    AC.write_json(AC.ANALYSIS / "label_access_audit.json", {
        "loaded": AC.loaded_paths(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "labels_read_by_this_script": False,
        "notes": "A1 reads the official-train structural cache only; no y tensor is touched.",
    })
    print(json.dumps({
        "checkpoints": {k: {"parity": v["parity_max_abs_diff"],
                            "params": v["checkpoint"]["parameters"],
                            "c_loc_PR": nondeg["checkpoints"][k]["c_loc"]["participation_ratio"],
                            "c_loc_within_PR": nondeg["checkpoints"][k]["c_loc_within"]["per_molecule_participation_ratio"]["median"]}
                        for k, v in audit["checkpoints"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
