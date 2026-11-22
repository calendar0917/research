"""Data identity, token sets and Train-only support bookkeeping.

The structural-object cache is built with the parent round's
``trc_prepare.build_dataset`` (same rooted 1-4 objects, same root-fixed frames,
same bridge cuts, same exact coloured-graph keys, same Train-only replacement
directory).  This module adds the *token view* that CSOI consumes: one token per
``(frame id, root-to-cut distance bucket)`` inside a fragment, with the frame
multiplicity mass that the parent CSR already stores.

Nothing here reads labels for structure, support, buckets or tokenisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import csoi_common as C


def cache_path(cfg: Mapping[str, Any], dataset: str = "lipo") -> Path:
    return C.data_dir() / f"csoi_cache_{dataset}.npz"


def build_cache(cfg: Mapping[str, Any], dataset: str = "lipo", log=print) -> dict[str, Any]:
    """Build the cache locally (needs rdkit) with the parent's builder."""
    pack = C.PP.build_dataset(cfg, dataset, log=log)
    raw = C.PD.load_raw_cache(cfg, dataset)
    if int(raw["node_ptr"].size - 1) != int(pack["n_molecules"]):
        raise ValueError(
            "raw cache mismatch: the parent loader picks the ESOL cache for any "
            f"dataset string other than 'lipo' (got {raw['cache_path']})")
    path = cache_path(cfg, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = C.PP.save_pack(path, pack)
    log(f"wrote {path} sha256={digest}")
    return load_pack(cfg, dataset)


def content_digest(path: Path | str, skip: Sequence[str] = ("timing_json",)) -> str:
    """Order-independent content identity of a prepared pack.

    ``np.savez`` stores zip timestamps, so two builds of the same cache have
    different *file* hashes.  This digest covers name, dtype, shape and bytes of
    every array except the build-timing string.
    """
    import hashlib

    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for name in sorted(z.files):
            if name in skip:
                continue
            arr = np.ascontiguousarray(z[name])
            h.update(str(name).encode())
            h.update(arr.dtype.str.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def load_pack(cfg: Mapping[str, Any], dataset: str = "lipo") -> dict[str, Any]:
    path = cache_path(cfg, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"prepared cache missing: {path}")
    with np.load(path, allow_pickle=False) as handle:
        pack = {key: handle[key] for key in handle.files}
    pack["cache_path"] = str(path)
    pack["cache_sha256"] = C.sha256_file(path)
    pack["content_sha256"] = content_digest(path)
    expected = (cfg.get("expected_content_sha256") or {}).get(dataset)
    if expected and pack["content_sha256"] != expected:
        raise ValueError(
            "cache identity mismatch: structural objects differ from the parent "
            f"round ({pack['content_sha256']} != {expected})")
    return pack


# ---------------------------------------------------------------------------
# token view
# ---------------------------------------------------------------------------


class TokenView:
    """Ragged token view over fragments (frame + distance bucket + mass)."""

    def __init__(self, pack: Mapping[str, Any]) -> None:
        self.frag_ptr = np.asarray(pack["frag_ptr"], np.int64)
        self.frag_bucket = np.asarray(pack["frag_bucket"], np.uint8)
        self.frag_frame = np.asarray(pack["frag_frame"], np.int32)
        self.frag_weight = np.asarray(pack["frag_weight"], np.float32)
        self.n_fragments = int(pack["n_fragments"])
        self.token_count = np.diff(self.frag_ptr)
        self.natoms = np.asarray(pack["frag_natoms"], np.int64)

    def slices(self, frag_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(frag_ids, np.int64)
        starts = self.frag_ptr[ids]
        return starts, self.frag_ptr[ids + 1] - starts

    def tokens_of(self, frag_ids: np.ndarray) -> dict[str, np.ndarray]:
        ids = np.asarray(frag_ids, np.int64)
        starts, counts = self.slices(ids)
        owner = np.repeat(np.arange(ids.size, dtype=np.int64), counts)
        total = int(counts.sum())
        local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        gidx = np.repeat(starts, counts) + local
        return {
            "owner": owner,
            "frame": self.frag_frame[gidx].astype(np.int64),
            "bucket": self.frag_bucket[gidx].astype(np.int64),
            "weight": self.frag_weight[gidx].astype(np.float64),
            "n_tokens": counts,
        }

    def object_mass(self, frag_ids: np.ndarray) -> np.ndarray:
        """Total rooted-object mass of each fragment (every object counts once)."""
        ids = np.asarray(frag_ids, np.int64)
        starts, counts = self.slices(ids)
        total = int(counts.sum())
        mass = np.zeros(ids.size, np.float64)
        if total == 0:
            return mass
        local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        gidx = np.repeat(starts, counts) + local
        owner = np.repeat(np.arange(ids.size, dtype=np.int64), counts)
        mass = np.bincount(owner, weights=self.frag_weight[gidx].astype(np.float64),
                           minlength=ids.size)
        return mass


# ---------------------------------------------------------------------------
# token statistics and the 128-token guard
# ---------------------------------------------------------------------------


def _stratified_cap(bucket: np.ndarray, weight: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic distance-bucket stratified keep mask with a token budget.

    Quota per bucket is proportional to the bucket's token count (largest
    remainder), at least one token per non-empty bucket; inside a bucket the
    highest-multiplicity tokens are kept (ties broken by token order).
    """
    n = bucket.size
    if n <= cap:
        return np.ones(n, bool)
    counts = np.bincount(bucket, minlength=C.N_BUCKETS)
    present = np.nonzero(counts)[0]
    quota = np.zeros(C.N_BUCKETS, np.int64)
    if cap < present.size:  # degenerate; keep the largest buckets
        order = present[np.argsort(-counts[present])]
        quota[order[:cap]] = 1
    else:
        quota[present] = 1
        left = cap - present.size
        share = left * counts[present] / counts[present].sum()
        base = np.floor(share).astype(np.int64)
        quota[present] += base
        rest = cap - int(quota.sum())
        if rest > 0:
            frac = share - base
            order = np.lexsort((-counts[present], -frac))
            quota[present[order[:rest]]] += 1
    keep = np.zeros(n, bool)
    order = np.lexsort((np.arange(n), -weight, bucket))
    used = np.zeros(C.N_BUCKETS, np.int64)
    for i in order.tolist():
        b = int(bucket[i])
        if used[b] < quota[b]:
            keep[i] = True
            used[b] += 1
    return keep


def token_statistics(pack: Mapping[str, Any], cfg: Mapping[str, Any],
                     log=print) -> dict[str, Any]:
    """Token counts per fragment + retained mass under the frozen 128 guard."""
    view = TokenView(pack)
    counts = view.token_count
    cap = int(cfg["csoi"]["guard_token_samples"])
    parse_S = np.asarray(pack["parse_S"], np.int64)
    parse_C = np.asarray(pack["parse_C"], np.int64)
    parse_mol = np.asarray(pack["parse_mol"], np.int64)
    split_of = np.zeros(int(pack["n_molecules"]), np.uint8) + 3
    split_of[np.asarray(pack["train_idx"], np.int64)] = 0
    split_of[np.asarray(pack["valid_idx"], np.int64)] = 1
    split_of[np.asarray(pack["test_idx"], np.int64)] = 2

    out: dict[str, Any] = {
        "n_fragments": int(view.n_fragments),
        "tokens_per_fragment": C.quantiles(counts),
        "tokens_per_fragment_by_size": {},
        "guard": {"cap": cap, "applied": bool(np.percentile(counts, 99) > cap)},
        "splits": {},
    }
    for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 32), (33, 200)):
        sel = (view.natoms >= lo) & (view.natoms <= hi)
        if sel.any():
            out["tokens_per_fragment_by_size"][f"atoms_{lo}_{hi}"] = {
                "n_fragments": int(sel.sum()), **C.quantiles(counts[sel])}
    # word side (S) vs environment side (C) actually used by parses
    s_counts = counts[parse_S] if parse_S.size else np.zeros(0)
    has_c = parse_C >= 0
    c_counts = counts[parse_C[has_c]] if has_c.any() else np.zeros(0)
    out["parse_side_tokens"] = {
        "S": C.quantiles(s_counts), "C": C.quantiles(c_counts)}

    # retained mass of the frozen guard (documentation only: the model uses the
    # full token set through the chunked attention path)
    kept_mass = np.zeros(view.n_fragments)
    total_mass = np.zeros(view.n_fragments)
    under95 = 0
    for fid in range(view.n_fragments):
        tok = view.tokens_of(np.asarray([fid]))
        w = tok["weight"]
        b = tok["bucket"]
        total_mass[fid] = w.sum()
        if w.size <= cap:
            kept_mass[fid] = total_mass[fid]
        else:
            keep = _stratified_cap(b, w, cap)
            kept_mass[fid] = w[keep].sum()
            if total_mass[fid] > 0 and kept_mass[fid] / total_mass[fid] < 0.95:
                under95 += 1
    frac = np.where(total_mass > 0, kept_mass / np.maximum(total_mass, 1e-12), 1.0)
    out["guard"].update({
        "retained_mass": C.quantiles(frac),
        "fragments_below_095": int(under95),
        "fraction_below_095": float(under95 / max(1, view.n_fragments)),
        "note": ("the guard is not used for training: the ragged attention path "
                 "processes the full token set in entry chunks, so no structural "
                 "object is dropped"),
    })
    for split, code in (("train", 0), ("valid", 1)):
        sel = split_of[parse_mol] == code
        out["splits"][split] = {
            "n_molecules": int((split_of == code).sum()),
            "n_parses": int(sel.sum()),
            "S_tokens_per_parse": C.quantiles(s_counts[sel]) if sel.any() else {},
            "C_tokens_per_parse": C.quantiles(counts[parse_C[sel & has_c]]) if (sel & has_c).any() else {},
            "attention_entries_per_parse": (
                float(np.sum(s_counts[sel].astype(np.float64)
                             * counts[np.where(has_c[sel], parse_C[sel], 0)].astype(np.float64)))
                if sel.any() else 0.0),
        }
    return out


# ---------------------------------------------------------------------------
# Train-only support
# ---------------------------------------------------------------------------


class SupportIndex:
    """Everything is defined on the *Train* parses only."""

    def __init__(self, pack: Mapping[str, Any]) -> None:
        self.parse_mol = np.asarray(pack["parse_mol"], np.int64)
        self.parse_S = np.asarray(pack["parse_S"], np.int64)
        self.parse_C = np.asarray(pack["parse_C"], np.int64)
        self.parse_bond = np.asarray(pack["parse_cut_bond_row"], np.int64)
        self.parse_kind = np.asarray(pack["parse_kind"], np.uint8)
        self.split_of = np.zeros(int(pack["n_molecules"]), np.uint8) + 3
        self.split_of[np.asarray(pack["train_idx"], np.int64)] = 0
        self.split_of[np.asarray(pack["valid_idx"], np.int64)] = 1
        self.split_of[np.asarray(pack["test_idx"], np.int64)] = 2
        train = self.split_of[self.parse_mol] == 0
        valid = self.split_of[self.parse_mol] == 1
        self.train_parse = np.nonzero(train)[0]
        self.valid_parse = np.nonzero(valid)[0]
        bond = np.maximum(self.parse_bond, -1)
        self.seen_S = set(self.parse_S[self.train_parse].tolist())
        real = train & (self.parse_kind == 0)
        self.seen_C = set(zip(self.parse_C[real].tolist(), bond[real].tolist()))
        self.seen_combo = set(zip(self.parse_S[train].tolist(),
                                  self.parse_C[train].tolist(),
                                  bond[train].tolist()))
        # rooted-object support from the Train molecule universe
        mol_ptr = np.asarray(pack["mol_ptr"], np.int64)
        mol_frame = np.asarray(pack["mol_frame"], np.int64)
        train_mols = np.nonzero(self.split_of == 0)[0]
        parts = [mol_frame[mol_ptr[m]:mol_ptr[m + 1]] for m in train_mols.tolist()]
        self.seen_objects = set(np.concatenate(parts).tolist()) if parts else set()

    def stratum(self, parse_idx: np.ndarray) -> np.ndarray:
        """0=(seen S, seen C) 1=(seen S, unseen C) 2=(unseen S, seen C) 3=(unseen S, unseen C)."""
        idx = np.asarray(parse_idx, np.int64)
        s_seen = np.isin(self.parse_S[idx], list(self.seen_S) or [-1])
        bond = np.maximum(self.parse_bond[idx], -1)
        keys = list(zip(self.parse_C[idx].tolist(), bond.tolist()))
        c_seen = np.asarray([k in self.seen_C for k in keys], bool) if keys else np.zeros(0, bool)
        return (~s_seen).astype(np.int64) * 2 + (~c_seen).astype(np.int64)

    def coverage(self) -> dict[str, Any]:
        idx = self.valid_parse
        bond = np.maximum(self.parse_bond[idx], -1)
        s_seen = np.isin(self.parse_S[idx], list(self.seen_S) or [-1])
        c_seen = np.asarray([k in self.seen_C for k in
                             zip(self.parse_C[idx].tolist(), bond.tolist())], bool)
        combo_seen = np.asarray([k in self.seen_combo for k in zip(
            self.parse_S[idx].tolist(), self.parse_C[idx].tolist(), bond.tolist())], bool)
        return {
            "n_valid_parses": int(idx.size),
            "valid_S_seen_rate": float(s_seen.mean()) if idx.size else 0.0,
            "valid_C_seen_rate": float(c_seen.mean()) if idx.size else 0.0,
            "valid_combo_seen_rate": float(combo_seen.mean()) if idx.size else 0.0,
            "n_train_S_keys": len(self.seen_S),
            "n_train_C_keys": len(self.seen_C),
            "n_train_combo_keys": len(self.seen_combo),
            "n_train_object_types": len(self.seen_objects),
            "strata_counts": np.bincount(self.stratum(idx), minlength=4).tolist(),
        }

    def object_coverage(self, pack: Mapping[str, Any]) -> dict[str, Any]:
        mol_ptr = np.asarray(pack["mol_ptr"], np.int64)
        mol_frame = np.asarray(pack["mol_frame"], np.int64)
        mol_weight = np.asarray(pack["mol_weight"], np.float64)
        out: dict[str, Any] = {}
        for split, code in (("train", 0), ("valid", 1)):
            mols = np.nonzero(self.split_of == code)[0]
            total = 0.0
            unseen = 0.0
            n_types = 0
            types: set[int] = set()
            for m in mols.tolist():
                lo, hi = int(mol_ptr[m]), int(mol_ptr[m + 1])
                f = mol_frame[lo:hi]
                w = mol_weight[lo:hi]
                types.update(f.tolist())
                total += float(w.sum())
                unseen += float(w[~np.isin(f, list(self.seen_objects) or [-1])].sum())
            n_types = len(types)
            out[split] = {
                "n_molecules": int(mols.size),
                "n_object_types": int(n_types),
                "object_mass": total,
                "unseen_object_mass_fraction": (unseen / total) if total else 0.0,
            }
        return out
