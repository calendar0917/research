"""P1 - complete statistics traceback for every molecule pair.

For a fixed trained model (no re-training, no re-weighting) and a pair ``A -> B``
defined by the P0 single-bond-cut matching we recompute **both full structures**
and split the standardised prediction difference into the three parts that the
frozen forward pass actually produces:

    L      = sum_v f(h_v)                                  (standardised)
    rho    = rho([g, mean_v h_v, mean_v h_v^2, log1p n])    (standardised)
    y_hat  = L + rho
    pred   = mu + sigma * (L + rho)                        (raw logD)   mu cancels

    dL_var  = sum_{v in B var}  f(h_v) - sum_{v in A var}  f(h_v)
    dL_core = sum_{v in B core} f(h_v) - sum_{v in A core} f(h_v)
    dR      = rho(B) - rho(A)
    d_pred  = sigma * (dL_var + dL_core + dR)

``dL_core`` is a *sum over matched core atoms*, so it does not depend on which
element of the core automorphism group the atomic correspondence uses.  Per-atom
statements are only made on atoms with a unique correspondence, or after
aggregating over automorphism orbits; otherwise the mapping is reported as
ambiguous.

The decomposition is a computational decomposition of the current function, not
an allocation of observed error to physics.  ``analysis/gauge_shift.json`` shows
a coordinate change (constant per-atom shift compensated inside ``rho``) that
leaves every prediction unchanged but moves the atom terms, so the absolute zero
point of a contribution is not a physical quantity.

Every statistic used to *explain* a change is recomputed from the legitimate A
and B molecules - nothing is transported from a single-molecule intervention.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import edit_common as C
import edit_data as ED

import dissect_common as DC
import dissect_model as M

FULL_TO_BLOCK: dict[int, str] = {}
for _name, (_lo, _hi) in DC.LOCAL_LAYOUT.items():
    for _c in range(_lo, _hi):
        FULL_TO_BLOCK[_c] = _name

SCALAR_NAMES = (
    "log1p_n_d2",
    "log1p_n_paths2",
    "closure_ratio",
    "log1p_degree",
    "in_ring",
    "min_ring_size/8",
    "ring_bond_count/4",
    "is_terminal",
    "is_fused_or_spiro",
)

#: input blocks compared channel-by-channel when a core atom's signature moves
CHANNELS = (
    "a",
    "b_sum",
    "b_mean",
    "s1_joint",
    "s2_type",
    "s2_end",
    "s2_joint",
    "s2_closure",
    "s2_content",
    "atom_scalar",
)

_BLOCK_CACHE: dict[tuple[int, ...], dict[str, np.ndarray]] = {}


def block_indices(model) -> dict[str, np.ndarray]:
    cols = tuple(int(c) for c in model.local_cols)
    if cols not in _BLOCK_CACHE:
        arr = np.asarray(cols)
        _BLOCK_CACHE[cols] = {
            block: np.flatnonzero(np.asarray([FULL_TO_BLOCK[int(c)] == block for c in arr]))
            for block in CHANNELS
        }
    return _BLOCK_CACHE[cols]


# ---------------------------------------------------------------------------
# per-molecule view of the *actual* model inputs
# ---------------------------------------------------------------------------


class MolView:
    """Everything the forward pass reads, sliced per atom, for one molecule."""

    def __init__(self, cache: Mapping[str, Any], pos: int, model) -> None:
        atom_ptr = np.asarray(cache["atom_ptr"])
        bond_ptr = np.asarray(cache["bond_ptr"])
        nbr_ptr = np.asarray(cache["nbr_ptr"])
        path_ptr = np.asarray(cache["path_ptr"])
        a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
        b0, b1 = int(bond_ptr[pos]), int(bond_ptr[pos + 1])
        n = a1 - a0
        self.n = n
        self.row = int(np.asarray(cache["rows"])[pos])
        self.smiles = str(cache["smiles"][pos])
        self.pos = int(pos)
        self.atom_feat = np.asarray(cache["atom_feat"][a0:a1], dtype=np.float32)
        self.atom_type = np.asarray(cache["atom_type"][a0:a1], dtype=np.int64)
        scal = np.asarray(cache["atom_scalar"][a0:a1], dtype=np.float32)
        mean = model.scalar_mean.detach().cpu().numpy()
        scale = model.scalar_scale.detach().cpu().numpy()
        self.atom_scalar = ((scal - mean) / scale).astype(np.float32)
        self.bond_feat = np.asarray(cache["bond_feat"][b0:b1], dtype=np.float32)
        # ``bond_index`` and ``path_end`` are stored per molecule in local atom
        # indices; ``collate`` adds the per-molecule atom offset.
        self.bond_index = np.asarray(cache["bond_index"][b0:b1], dtype=np.int64).reshape(-1, 2)
        incident: list[list[int]] = [[] for _ in range(n)]
        for e, (u, v) in enumerate(self.bond_index.tolist()):
            incident[int(u)].append(e)
            incident[int(v)].append(e)
        self.incident = [tuple(sorted(x)) for x in incident]
        self.nbr = [
            tuple(int(x) for x in np.asarray(cache["nbr_code"][int(nbr_ptr[a0 + i]) : int(nbr_ptr[a0 + i + 1])]))
            for i in range(n)
        ]
        path_end = np.asarray(cache["path_end"])
        path_cat = np.asarray(cache["path_cat"])
        path_end63 = np.asarray(cache["path_end63"])
        path_closure = np.asarray(cache["path_closure"])
        self.paths: list[tuple[tuple[int, int, int, bytes], ...]] = []
        for i in range(n):
            lo, hi = int(path_ptr[a0 + i]), int(path_ptr[a0 + i + 1])
            recs = []
            for j in range(lo, hi):
                end_local = int(path_end[j])
                recs.append(
                    (int(path_cat[j]), int(path_end63[j]), int(path_closure[j]), self.atom_feat[end_local].tobytes())
                )
            self.paths.append(tuple(sorted(recs)))
        self.n_paths = tuple(len(p) for p in self.paths)

    # -- signature: covers the complete forward input of one atom ---------
    def incident_entries(self, atom: int) -> tuple[tuple[bytes, bytes], ...]:
        out = []
        for e in self.incident[atom]:
            u, v = int(self.bond_index[e][0]), int(self.bond_index[e][1])
            other = v if u == atom else u
            out.append((self.bond_feat[e].tobytes(), self.atom_feat[other].tobytes()))
        return tuple(sorted(out))

    def signature(self, atom: int) -> tuple:
        return (
            self.atom_feat[atom].tobytes(),
            self.atom_scalar[atom].tobytes(),
            self.incident_entries(atom),
            self.nbr[atom],
            self.paths[atom],
        )

    def describe_difference(self, other: "MolView", a_self: int, a_other: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not np.array_equal(self.atom_feat[a_self], other.atom_feat[a_other]):
            out["atom_type_a"] = int(self.atom_type[a_self])
            out["atom_type_b"] = int(other.atom_type[a_other])
            out["atom_feat_changed_indices"] = [
                int(x) for x in np.flatnonzero(self.atom_feat[a_self] != other.atom_feat[a_other])
            ]
        if not np.array_equal(self.atom_scalar[a_self], other.atom_scalar[a_other]):
            diff = [int(x) for x in np.flatnonzero(self.atom_scalar[a_self] != other.atom_scalar[a_other])]
            out["atom_scalar_changed"] = [SCALAR_NAMES[i] for i in diff]
            out["atom_scalar_delta"] = [float(other.atom_scalar[a_other][i]) - float(self.atom_scalar[a_self][i]) for i in diff]
        bs, bo = self.incident_entries(a_self), other.incident_entries(a_other)
        if bs != bo:
            out["incident_bonds_changed"] = {
                "degree_a": len(bs),
                "degree_b": len(bo),
                "n_multiset_symbol_difference": len(set(bs) ^ set(bo)),
            }
        if self.nbr[a_self] != other.nbr[a_other]:
            out["one_hop_codes_changed"] = {"a": sorted(self.nbr[a_self]), "b": sorted(other.nbr[a_other])}
        if self.paths[a_self] != other.paths[a_other]:
            pa, pb = set(self.paths[a_self]), set(other.paths[a_other])
            out["two_hop_paths_changed"] = {
                "n_a": len(self.paths[a_self]),
                "n_b": len(other.paths[a_other]),
                "added": _path_reprs(pb - pa),
                "removed": _path_reprs(pa - pb),
            }
        return out


def _path_reprs(paths: Any, limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "path_cat": int(cat),
            "path_end63": int(end63),
            "closure": int(closure),
            "end_atom_feat_sha8": hashlib.sha256(endfeat).hexdigest()[:8],
        }
        for cat, end63, closure, endfeat in sorted(paths)[:limit]
    ]


# ---------------------------------------------------------------------------
# pairs / checkpoints
# ---------------------------------------------------------------------------


def load_pairs() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(open(C.ANALYSIS / "pairs.csv", encoding="utf-8")))
    mappings = {m["pair_id"]: m for m in C.read_json(C.ANALYSIS / "pair_mappings.json")["pairs"]}
    split = C.read_json(C.ANALYSIS / "component_split.json")
    partition_of: dict[int, str] = {}
    for part in C.PARTITIONS:
        for row in split["rows_by_partition"][part]:
            partition_of[int(row)] = part
    classes = {int(c["core_class"]): c for c in C.read_json(C.ANALYSIS / "core_classes.json")["classes"]}
    out = []
    for row in rows:
        pid = row["pair_id"]
        m = mappings[pid]
        a, b = int(row["row_a"]), int(row["row_b"])
        if partition_of[a] != partition_of[b]:
            raise SystemExit(f"{pid}: pair straddles partitions")
        out.append(
            {
                "pair_id": pid,
                "row_a": a,
                "row_b": b,
                "partition": partition_of[a],
                "class_id": row["class_id"],
                "core_class": int(row["core_class"]),
                "core_heavy": int(row["core_heavy"]),
                "var_a": row["var_a"],
                "var_b": row["var_b"],
                "var_heavy_a": int(row["var_heavy_a"]),
                "var_heavy_b": int(row["var_heavy_b"]),
                "same_var_heavy": row["same_var_heavy"] == "1",
                "core_atoms_a": [int(x) for x in json.loads(m["core_atoms_a"])] if isinstance(m["core_atoms_a"], str) else [int(x) for x in m["core_atoms_a"]],
                "core_atoms_b": [int(x) for x in json.loads(m["core_atoms_b"])] if isinstance(m["core_atoms_b"], str) else [int(x) for x in m["core_atoms_b"]],
                "var_atoms_a": [int(x) for x in json.loads(m["var_atoms_a"])] if isinstance(m["var_atoms_a"], str) else [int(x) for x in m["var_atoms_a"]],
                "var_atoms_b": [int(x) for x in json.loads(m["var_atoms_b"])] if isinstance(m["var_atoms_b"], str) else [int(x) for x in m["var_atoms_b"]],
                "attach_a": int(m["attach_a"]),
                "attach_b": int(m["attach_b"]),
                "mapping": {int(k): int(v) for k, v in m["mapping"].items()},
                "mapping_unique": bool(classes[int(row["core_class"])]["mapping_unique"]),
                "n_automorphisms_capped": int(classes[int(row["core_class"])]["n_automorphisms_capped"]),
                "automorphism_truncated": bool(classes[int(row["core_class"])]["automorphism_truncated"]),
            }
        )
    return out


def load_model(model_variant: str, seed: int, device: str = "cpu"):
    spec = ED.frozen_spec()
    norm = ED.train_norm()
    path = C.run_dir(model_variant, seed) / "checkpoint.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = M.load_checkpoint_model(spec, norm, model_variant, ckpt["state_dict"], torch.device(device))
    model.eval()
    return model, ckpt


def _pair_positions(cache: Mapping[str, Any]) -> dict[int, int]:
    return {int(r): i for i, r in enumerate(np.asarray(cache["rows"]).tolist())}


def encode(views: Sequence[MolView], cache, model, device: str = "cpu") -> dict[str, Any]:
    batch = ED.to_device(ED.collate(cache, [v.pos for v in views]), torch.device(device))
    with torch.no_grad():
        y, aux = model(batch, return_aux=True)
    return {
        "y": y.detach().cpu().numpy(),
        "h": aux["h"].detach().cpu().numpy(),
        "contrib_atom": aux["contrib_atom"].detach().cpu().numpy(),
        "contrib": aux["contrib"].detach().cpu().numpy(),
        "rho": aux["rho"].detach().cpu().numpy(),
        "rho_input": aux["rho_input"].detach().cpu().numpy(),
        "local_input": aux["local_input"].detach().cpu().numpy(),
        "mean_h": aux["mean_h"].detach().cpu().numpy(),
        "mean_h2": aux["mean_h2"].detach().cpu().numpy(),
        "offsets": np.cumsum([0] + [v.n for v in views])[:-1],
    }


# ---------------------------------------------------------------------------
# one pair
# ---------------------------------------------------------------------------


def _block_deltas(model, x_a: np.ndarray, x_b: np.ndarray) -> dict[str, dict[str, Any]]:
    idx = block_indices(model)
    delta = x_b - x_a
    out: dict[str, dict[str, Any]] = {}
    for block in CHANNELS:
        ii = idx[block]
        if ii.size == 0:
            out[block] = {"present": False, "l1": 0.0, "linf": 0.0}
            continue
        sub = delta[ii]
        out[block] = {"present": True, "l1": float(np.abs(sub).sum()), "linf": float(np.abs(sub).max())}
    return out


def trace_pair(
    pair: Mapping[str, Any], cache, model, seed: int, sigma: float, top_atoms: int = 6
) -> tuple[dict[str, Any], dict[str, Any]]:
    pos = _pair_positions(cache)
    va = MolView(cache, pos[pair["row_a"]], model)
    vb = MolView(cache, pos[pair["row_b"]], model)
    enc = encode([va, vb], cache, model)
    n_a, n_b = va.n, vb.n
    hA, hB = enc["h"][:n_a], enc["h"][n_a : n_a + n_b]
    fA, fB = enc["contrib_atom"][:n_a], enc["contrib_atom"][n_a : n_a + n_b]
    xA, xB = enc["local_input"][:n_a], enc["local_input"][n_a : n_a + n_b]

    var_a = sorted(int(i) for i in pair["var_atoms_a"])
    var_b = sorted(int(i) for i in pair["var_atoms_b"])
    core_a = sorted(int(i) for i in pair["core_atoms_a"])
    core_b = sorted(int(i) for i in pair["core_atoms_b"])
    if sorted(var_a + core_a) != list(range(n_a)) or sorted(var_b + core_b) != list(range(n_b)):
        raise SystemExit(f"{pair['pair_id']}: core+variable atoms do not partition the molecule")

    dL_var = float(fB[var_b].sum() - fA[var_a].sum())
    dL_core = float(fB[core_b].sum() - fA[core_a].sum())
    dR = float(enc["rho"][1] - enc["rho"][0])
    d_pred_std = float(enc["y"][1] - enc["y"][0])
    d_pred_raw = sigma * d_pred_std
    comp_sum_raw = sigma * (dL_var + dL_core + dR)
    resid = d_pred_raw - comp_sum_raw
    tol = max(C.IDENTITY_ABS_TOL, C.IDENTITY_REL_TOL * abs(d_pred_raw))

    mapping = pair["mapping"]
    changed, unchanged = [], []
    for ia in core_a:
        ib = int(mapping[int(ia)])
        same = va.signature(ia) == vb.signature(ib)
        rec = {
            "atom_a": int(ia),
            "atom_b": ib,
            "dh_linf": float(np.abs(hB[ib] - hA[ia]).max()),
            "df": float(abs(float(fB[ib]) - float(fA[ia]))),
            "blocks": _block_deltas(model, xA[ia], xB[ib]),
        }
        if same:
            unchanged.append(rec)
        else:
            rec["difference"] = va.describe_difference(vb, ia, ib)
            changed.append(rec)

    g_delta = enc["rho_input"][1, : DC.GLOBAL_DIM] - enc["rho_input"][0, : DC.GLOBAL_DIM]
    rec_out: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "partition": pair["partition"],
        "model": model.variant,
        "seed": int(seed),
        "row_a": pair["row_a"],
        "row_b": pair["row_b"],
        "class_id": pair["class_id"],
        "core_class": pair["core_class"],
        "core_heavy": pair["core_heavy"],
        "var_heavy_a": pair["var_heavy_a"],
        "var_heavy_b": pair["var_heavy_b"],
        "same_var_heavy": int(pair["same_var_heavy"]),
        "n_var_atoms_a": len(var_a),
        "n_var_atoms_b": len(var_b),
        "n_atoms_a": n_a,
        "n_atoms_b": n_b,
        "pred_a_raw": float(enc["y"][0] * sigma),
        "pred_b_raw": float(enc["y"][1] * sigma),
        "L_a_raw": float(enc["contrib"][0] * sigma),
        "L_b_raw": float(enc["contrib"][1] * sigma),
        "rho_a_raw": float(enc["rho"][0] * sigma),
        "rho_b_raw": float(enc["rho"][1] * sigma),
        "d_pred_raw": d_pred_raw,
        "dL_var_raw": sigma * dL_var,
        "dL_core_raw": sigma * dL_core,
        "dR_raw": sigma * dR,
        "component_sum_raw": comp_sum_raw,
        "identity_residual_raw": resid,
        "identity_residual_rel": float(abs(resid) / max(abs(d_pred_raw), 1e-12)),
        "identity_ok": bool(abs(resid) <= tol),
        "n_core_atoms": len(core_a),
        "n_core_atoms_changed": len(changed),
        "n_core_atoms_unchanged": len(unchanged),
        "max_abs_dh_unchanged": float(max((r["dh_linf"] for r in unchanged), default=0.0)),
        "max_abs_df_unchanged": float(max((r["df"] for r in unchanged), default=0.0)),
        "mean_abs_df_changed": float(np.mean([r["df"] for r in changed])) if changed else 0.0,
        "max_abs_df_changed": float(max((r["df"] for r in changed), default=0.0)),
        "mean_dh_changed": float(np.mean([r["dh_linf"] for r in changed])) if changed else 0.0,
        "max_dh_changed": float(max((r["dh_linf"] for r in changed), default=0.0)),
        "mapping_unique": int(pair["mapping_unique"]),
        "n_automorphisms_capped": pair["n_automorphisms_capped"],
        "automorphism_truncated": int(pair["automorphism_truncated"]),
        "g_delta_l1": float(np.abs(g_delta).sum()),
        "g_delta_linf": float(np.abs(g_delta).max()),
        "g_n_changed": int((np.abs(g_delta) > 0).sum()),
        "mean_h_delta_linf": float(np.abs(enc["mean_h"][1] - enc["mean_h"][0]).max()),
        "mean_h2_delta_linf": float(np.abs(enc["mean_h2"][1] - enc["mean_h2"][0]).max()),
        "n_paths_a": int(sum(va.n_paths)),
        "n_paths_b": int(sum(vb.n_paths)),
    }
    for block in CHANNELS:
        rec_out[f"chg_{block}_l1"] = float(sum(r["blocks"][block]["l1"] for r in changed))
        rec_out[f"chg_{block}_linf"] = float(max((r["blocks"][block]["linf"] for r in changed), default=0.0))
        rec_out[f"chg_{block}_present"] = int(all(r["blocks"][block]["present"] for r in changed)) if changed else 1

    top = sorted(changed, key=lambda r: (-r["df"], r["atom_a"]))[:top_atoms]
    detail = {
        "pair_id": pair["pair_id"],
        "partition": pair["partition"],
        "model": model.variant,
        "seed": int(seed),
        "row_a": pair["row_a"],
        "row_b": pair["row_b"],
        "mapping_unique": pair["mapping_unique"],
        "n_automorphisms_capped": pair["n_automorphisms_capped"],
        "automorphism_truncated": pair["automorphism_truncated"],
        "n_unchanged": len(unchanged),
        "n_changed": len(changed),
        "n_unchanged_with_df_above_1e-6": int(sum(1 for r in unchanged if r["df"] > 1e-6)),
        "n_unchanged_with_dh_above_1e-6": int(sum(1 for r in unchanged if r["dh_linf"] > 1e-6)),
        "top_changed_atoms": top,
    }
    return rec_out, detail


# ---------------------------------------------------------------------------
# global verifications
# ---------------------------------------------------------------------------


def permuted_cache_view(cache: Mapping[str, Any], pos: int, perm: Sequence[int]) -> dict[str, Any]:
    """Single-molecule cache with atoms re-ordered so that ``new = perm[old_index_map]``."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
    b0, b1 = int(bond_ptr[pos]), int(bond_ptr[pos + 1])
    n = a1 - a0
    perm = np.asarray(perm, dtype=np.int64)  # perm[new] = old
    inverse = np.empty(n, dtype=np.int64)
    inverse[perm] = np.arange(n)
    bi = np.asarray(cache["bond_index"][b0:b1], dtype=np.int64).reshape(-1, 2)
    new_bi = inverse[bi] if bi.size else bi.reshape(0, 2)
    nbr_codes = [
        np.asarray(cache["nbr_code"][int(nbr_ptr[a0 + int(perm[i])]) : int(nbr_ptr[a0 + int(perm[i]) + 1])], dtype=np.int64)
        for i in range(n)
    ]
    per_atom_paths = []
    for i in range(n):
        old = int(perm[i])
        lo, hi = int(path_ptr[a0 + old]), int(path_ptr[a0 + old + 1])
        per_atom_paths.append(
            (
                np.asarray(cache["path_cat"][lo:hi], dtype=np.int64),
                np.asarray(cache["path_end63"][lo:hi], dtype=np.int64),
                np.asarray(cache["path_end"][lo:hi], dtype=np.int64),
                np.asarray(cache["path_closure"][lo:hi], dtype=np.int64),
            )
        )
    cat = np.concatenate([x[0] for x in per_atom_paths]) if n else np.zeros(0, np.int64)
    end63 = np.concatenate([x[1] for x in per_atom_paths]) if n else np.zeros(0, np.int64)
    pend = inverse[np.concatenate([x[2] for x in per_atom_paths])] if n else np.zeros(0, np.int64)
    pclos = np.concatenate([x[3] for x in per_atom_paths]) if n else np.zeros(0, np.int64)
    new_cache = {
        "atom_ptr": np.asarray([0, n], dtype=np.int64),
        "bond_ptr": np.asarray([0, len(new_bi)], dtype=np.int64),
        "nbr_ptr": np.concatenate([[0], np.cumsum([len(x) for x in nbr_codes])]).astype(np.int64),
        "path_ptr": np.concatenate([[0], np.cumsum([len(x[0]) for x in per_atom_paths])]).astype(np.int64),
        "atom_feat": np.asarray(cache["atom_feat"][a0:a1])[perm],
        "atom_type": np.asarray(cache["atom_type"][a0:a1])[perm],
        "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1])[perm],
        "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
        "bond_type": np.asarray(cache["bond_type"][b0:b1]),
        "bond_index": new_bi,
        "nbr_code": np.concatenate(nbr_codes) if n else np.zeros(0, np.int64),
        "path_cat": cat,
        "path_end63": end63,
        "path_end": pend,
        "path_closure": pclos,
        "global": np.asarray(cache["global"][pos : pos + 1]),
        "n_components": np.asarray(cache["n_components"][pos : pos + 1]),
        "n_junctions": np.asarray(cache["n_junctions"][pos : pos + 1]),
        "n_ring_systems": np.asarray(cache["n_ring_systems"][pos : pos + 1]),
        "molecule_ids": [str(cache["molecule_ids"][pos])],
        "smiles": [str(cache["smiles"][pos])],
        "n_molecules": 1,
        "split": cache["split"],
        "meta": dict(cache["meta"]),
        "rows": np.asarray([int(np.asarray(cache["rows"])[pos])], dtype=np.int64),
        "dataset": cache.get("dataset"),
        "split_name": cache.get("split_name"),
        "repeat": cache.get("repeat"),
        "cache_version": cache.get("cache_version"),
        "source_commit": cache.get("source_commit"),
    }
    return new_cache


def _swapped(pair: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(pair)
    out["row_a"], out["row_b"] = pair["row_b"], pair["row_a"]
    out["var_atoms_a"], out["var_atoms_b"] = pair["var_atoms_b"], pair["var_atoms_a"]
    out["core_atoms_a"], out["core_atoms_b"] = pair["core_atoms_b"], pair["core_atoms_a"]
    out["mapping"] = {int(v): int(k) for k, v in pair["mapping"].items()}
    out["var_heavy_a"], out["var_heavy_b"] = pair["var_heavy_b"], pair["var_heavy_a"]
    return out


def verify_all(pairs: Sequence[Mapping[str, Any]], model, caches: Mapping[str, Any], sigma: float) -> dict[str, Any]:
    flip_max = 0.0
    flip_fail = 0
    single: dict[int, float] = {}
    split_max = 0.0
    for pair in pairs:
        cache = caches[pair["partition"]]
        fwd, _ = trace_pair(pair, cache, model, 0, sigma)
        rev, _ = trace_pair(_swapped(pair), cache, model, 0, sigma)
        err = max(
            abs(fwd["d_pred_raw"] + rev["d_pred_raw"]),
            abs(fwd["dL_var_raw"] + rev["dL_var_raw"]),
            abs(fwd["dL_core_raw"] + rev["dL_core_raw"]),
            abs(fwd["dR_raw"] + rev["dR_raw"]),
        )
        flip_max = max(flip_max, err)
        if err > 1e-6:
            flip_fail += 1
        pos = _pair_positions(cache)
        views = [MolView(cache, pos[pair["row_a"]], model), MolView(cache, pos[pair["row_b"]], model)]
        enc_pair = encode(views, cache, model)
        for k, view in enumerate(views):
            if view.row not in single:
                single[view.row] = float(encode([MolView(cache, pos[view.row], model)], cache, model)["y"][0])
            split_max = max(split_max, abs(float(enc_pair["y"][k]) - single[view.row]))

    renumber = {"y": 0.0, "rho": 0.0, "h": 0.0, "f": 0.0}
    n_mol = 0
    for row in sorted({int(p["row_a"]) for p in pairs} | {int(p["row_b"]) for p in pairs}):
        part = next(p["partition"] for p in pairs if int(p["row_a"]) == row or int(p["row_b"]) == row)
        cache = caches[part]
        pos = _pair_positions(cache)
        base = MolView(cache, pos[row], model)
        enc0 = encode([base], cache, model)
        perm = list(range(base.n))[::-1]  # new order = reversed old order
        sub = permuted_cache_view(cache, pos[row], perm)
        enc1 = encode([MolView(sub, 0, model)], sub, model)
        renumber["y"] = max(renumber["y"], float(abs(enc1["y"][0] - enc0["y"][0])))
        renumber["rho"] = max(renumber["rho"], float(abs(enc1["rho"][0] - enc0["rho"][0])))
        renumber["h"] = max(renumber["h"], float(np.abs(enc1["h"] - enc0["h"][np.asarray(perm)]).max()))
        renumber["f"] = max(renumber["f"], float(np.abs(enc1["contrib_atom"] - enc0["contrib_atom"][np.asarray(perm)]).max()))
        n_mol += 1
    return {
        "n_pairs_checked": len(pairs),
        "reverse_sign_flip_max_abs_raw": float(flip_max),
        "reverse_sign_flip_failures": int(flip_fail),
        "batch_split_merge_max_abs_standardised": float(split_max),
        "atom_renumbering": {"n_molecules": n_mol, **{k: float(v) for k, v in renumber.items()}},
    }


# ---------------------------------------------------------------------------
# gauge demonstration
# ---------------------------------------------------------------------------


def gauge_demo(pairs: Sequence[Mapping[str, Any]], model, cache, sigma: float, model_variant: str, seed: int) -> dict[str, Any]:
    """A coordinate change that leaves every prediction unchanged."""
    k_raw = C.GAUGE_K
    rows = []
    for pair in pairs:
        rec, _ = trace_pair(pair, cache, model, seed, sigma)
        n_var_a, n_var_b = rec["n_var_atoms_a"], rec["n_var_atoms_b"]
        n_a, n_b = rec["n_atoms_a"], rec["n_atoms_b"]
        shifted = {
            "dL_var_raw": rec["dL_var_raw"] + k_raw * (n_var_b - n_var_a),
            "dL_core_raw": rec["dL_core_raw"],
            "dR_raw": rec["dR_raw"] - k_raw * (n_b - n_a),
        }
        total_shifted = shifted["dL_var_raw"] + shifted["dL_core_raw"] + shifted["dR_raw"]
        rows.append(
            {
                "pair_id": rec["pair_id"],
                "d_pred_raw": rec["d_pred_raw"],
                "original_components": {
                    "dL_var_raw": rec["dL_var_raw"],
                    "dL_core_raw": rec["dL_core_raw"],
                    "dR_raw": rec["dR_raw"],
                },
                "after_uniform_shift": shifted,
                "components_sum_before": rec["component_sum_raw"],
                "components_sum_after": total_shifted,
                "abs_change_in_d_pred": float(abs(total_shifted - rec["component_sum_raw"])),
                "same_var_heavy": bool(rec["same_var_heavy"]),
            }
        )
    return {
        "model": model_variant,
        "seed": int(seed),
        "shift_per_atom_raw_logD": k_raw,
        "rule": (
            "c'_v = c_v + k for every atom and rho' = rho - k*n_atoms keeps "
            "pred = mu + sigma*(sum_v c_v + rho) identical for every molecule, hence d_pred too, "
            "while the split between dL_var and dR moves by k*(n_var_B - n_var_A). Absolute atom "
            "contribution values are therefore gauge-dependent; only the total d_pred is anchored."
        ),
        "max_abs_change_in_d_pred": float(max((r["abs_change_in_d_pred"] for r in rows), default=0.0)),
        "n_pairs": len(rows),
        "pairs": rows,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P1: per-pair statistics traceback")
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    sigma = float(ED.target_norm()["scale"])
    caches = {p: ED.load_cache(p) for p in C.PARTITIONS}

    all_rows: list[dict[str, Any]] = []
    verify: dict[str, Any] = {}
    gauge: dict[str, Any] = {}
    details_path = C.ANALYSIS / "pair_atom_changes.jsonl.gz"
    with gzip.open(details_path, "wt", encoding="utf-8") as detail_fh:
        for model_variant in args.models:
            for seed in args.seeds:
                model, ckpt = load_model(model_variant, int(seed))
                key = f"{model_variant}:s{seed}"
                C.log(f"[P1] {key} checkpoint={ckpt['best_state_hash'][:12]} best_epoch={ckpt['best_epoch']}")
                for pair in pairs:
                    rec, detail = trace_pair(pair, caches[pair["partition"]], model, int(seed), sigma)
                    all_rows.append(rec)
                    if pair["partition"] in ("diagnose", "audit"):
                        detail_fh.write(json.dumps(detail, sort_keys=True) + "\n")
                bad = [r["pair_id"] for r in all_rows if not r["identity_ok"]]
                if bad:
                    raise SystemExit(f"{len(bad)} pair traces violate the identity, e.g. {bad[:5]}")
                if not args.skip_verify:
                    verify[key] = verify_all(pairs, model, caches, sigma)
                    C.log(f"[P1] verify {key}: {verify[key]}")
                    gauge[key] = gauge_demo(
                        [p for p in pairs if p["partition"] == "diagnose"], model, caches["diagnose"], sigma, model_variant, int(seed)
                    )

    out_csv = C.ANALYSIS / "pair_traces.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    C.write_json(
        C.ANALYSIS / "trace_verification.json",
        {
            "schema": "lcrc-local-edit-trace-verification-v1",
            "created_utc": C.utcnow(),
            "identity_threshold": {"abs": C.IDENTITY_ABS_TOL, "rel": C.IDENTITY_REL_TOL},
            "max_identity_residual_raw": float(max(abs(r["identity_residual_raw"]) for r in all_rows)),
            "n_pair_traces": len(all_rows),
            "n_identity_violations": int(sum(0 if r["identity_ok"] else 1 for r in all_rows)),
            "n_unchanged_core_atoms": int(sum(r["n_core_atoms_unchanged"] for r in all_rows)),
            "n_changed_core_atoms": int(sum(r["n_core_atoms_changed"] for r in all_rows)),
            "max_abs_df_on_unchanged_atoms": float(max(r["max_abs_df_unchanged"] for r in all_rows)),
            "max_abs_dh_on_unchanged_atoms": float(max(r["max_abs_dh_unchanged"] for r in all_rows)),
            "per_model_seed": verify,
            "atom_detail_file": C.track_rel(details_path),
            "atom_detail_sha256": C.sha256_file(details_path),
            "scope": "all 509 pairs (fit/select/diagnose/audit) x 6 model-seed units",
        },
    )
    if gauge:
        C.write_json(C.ANALYSIS / "gauge_shift.json", {"schema": "lcrc-local-edit-gauge-shift-v1", "demos": gauge})
    C.log(f"[P1] wrote {len(all_rows)} pair traces -> {C.track_rel(out_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
