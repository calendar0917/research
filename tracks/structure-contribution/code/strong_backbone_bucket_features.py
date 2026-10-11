"""cscl-v3d bucket-complete frozen features (distance-conditioned restoration).

Research track: ``structure-contribution``, round ``cscl-v3d-distance-conditioned-
contribution``.  Protocol: ``configs/cscl_v3d_bucket.yaml`` (frozen before any
build/training).

What v3/v3b/v3c lost (verified against code, results/cscl_v3b/phase_a_boundaries.json)
-------------------------------------------------------------------------------------
``strong_backbone_attribution.extract_batch`` computes, but does NOT store into
``MolFrozen``:
  * intra-unit per-distance-bucket pair moments  (intra_b_counts / sum / sq)
  * cross-unit per-distance-bucket pair moments  (cross_b_counts / sum / sq)
while the frozen Full-Y reader consumes exactly those per-bucket first/second
moment blocks (``v0.pool_pair_moments``: per bucket [sum, sumsq, log1p count]).
The stored S/O inputs keep only bucket-marginal moments, so distance-conditioned
magnitude information is lost.

What this module does
---------------------
* Re-runs the frozen forward with the SAME operations in the SAME order as
  ``sba.extract_batch`` (``strong_backbone_attribution.py`` is NOT modified, so
  the v3 cache fingerprint and all v3/v3b/v3c replays stay valid) and stores the
  bucket-complete aggregates:
      unit:  count_intra[u,b], sum_pv_intra[u,b,:], sum_pv2_intra[u,b,:]
      rel:   count_cross (already rel_bucket_counts), sum_pv_cross, sum_pv2_cross
* Stores audit-only reference blocks of the Full-Y reader pair pool
  (first/second moments + log1p counts per bucket) so the STORED cache itself
  can be verified to rebuild Full-Y's pooled pair moments -- the training
  collate never reads those audit fields.
* Freezes full provenance (code fingerprints, checkpoint/prep/split hashes,
  bucket rule) and enforces conservation gates at build time.

Label discipline: y is never used by the extraction.  Official valid/test are
never loaded.  The old cache ``cscl_v3_frozen_features.pt`` is never touched.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import strong_backbone_attribution as sba  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    e2e_dictenv_clean_mechanism_v1 as cm,
    e2e_dictenv_h1_clarity_audit as audit,
    e2e_dictenv_p1 as p1,
    e2e_dictenv_v0 as v0,
)

PROTOCOL_ID = "cscl-v3d-distance-conditioned-contribution"
FEATURE_VERSION = "cscl-v3d-bucket-features-v1"

BUCKET_CACHE = REPO_ROOT / "data/cache/cscl_v3d_bucket_features.pt"
OLD_CACHE = sba.FEAT_CACHE

ENV_DIM = sba.ENV_DIM            # 144
PAIR_DIM = sba.PAIR_DIM          # 48
N_BUCKETS = sba.N_BUCKETS        # 5
REL_USED = sba.REL_USED          # 15
CTX_STRUCTURE = sba.CTX_STRUCTURE  # 30
TOPO_DIM = sba.TOPO_DIM          # 25
BOND_CATS = sba.BOND_CATS        # 4
EXTRACT_BATCH = sba.EXTRACT_BATCH

#: training-input block widths derived from the stored bucket arrays
UNIT_BUCKET_DIM = N_BUCKETS * (1 + 2 * PAIR_DIM)   # 485: per bucket [count, mean, rms]
REL_BUCKET_DIM = N_BUCKETS * (2 * PAIR_DIM)        # 480: per bucket [mean, rms] (counts already old cols)

POOL_RTOL = sba.POOL_RTOL   # 2e-4
POOL_ATOL = sba.POOL_ATOL   # 2e-3

PAIR_BUCKET_RULE = (
    "pair_bucket = min(max(topological graph distance,1),5)-1 in {0,1,2,3,4}; "
    "equals argmax(pair_relation[:5]); bucket 0 = bonded (distance 1)"
)


# ---------------------------------------------------------------------------
# per-molecule record
# ---------------------------------------------------------------------------


@dataclass
class MolBucketFeatures:
    """v3 ``MolFrozen`` fields (identical names/shapes) + bucket-complete arrays.

    The bucket arrays make the stored cache sufficient to rebuild Full-Y's
    per-bucket pair pooled first/second moments.  ``pair_ref_first`` /
    ``pair_ref_second`` / ``pair_ref_logcount`` are AUDIT-ONLY copies of the
    Full-Y reader pool blocks; the S/O collate must not read them (test-pinned).
    """

    row: int
    y: float
    n_atoms: int
    n_units: int
    n_pairs: int
    unit_sigs: list[str]
    unit_kinds: list[int]
    unit_atoms: list[list[int]]
    unit_desc: np.ndarray              # [n_units, 40]
    unit_sum_e: np.ndarray             # [n_units, 144]
    unit_sum_e2: np.ndarray            # [n_units, 144]
    unit_sum_pv: np.ndarray            # [n_units, 48]  intra, bucket-marginal
    unit_sum_pv2: np.ndarray           # [n_units, 48]
    unit_n_pv: np.ndarray              # [n_units]
    rel_lo: np.ndarray                 # [n_rel]
    rel_hi: np.ndarray
    rel_sum_pv: np.ndarray             # [n_rel, 48]    cross, bucket-marginal
    rel_sum_pv2: np.ndarray            # [n_rel, 48]
    rel_count: np.ndarray              # [n_rel]
    rel_bucket_counts: np.ndarray      # [n_rel, 5]
    rel_bond_hist: np.ndarray          # [n_rel, 4]
    rel_mean_relation: np.ndarray      # [n_rel, 15]
    ctx_structure: np.ndarray          # [30]
    topology: np.ndarray               # [25]
    # ---- NEW bucket-complete arrays (v3d) ----
    unit_bucket_counts: np.ndarray     # [n_units, 5]
    unit_bucket_sum_pv: np.ndarray     # [n_units, 5, 48]
    unit_bucket_sum_pv2: np.ndarray    # [n_units, 5, 48]
    rel_bucket_sum_pv: np.ndarray      # [n_rel, 5, 48]
    rel_bucket_sum_pv2: np.ndarray     # [n_rel, 5, 48]
    # ---- AUDIT-ONLY Full-Y reader reference blocks (not model inputs) ----
    pair_ref_first: np.ndarray         # [5, 48]
    pair_ref_second: np.ndarray        # [5, 48]
    pair_ref_logcount: np.ndarray      # [5]

    def type_ids(self, vocab) -> np.ndarray:
        return np.asarray(
            [
                vocab.to_id(sig, kind, len(atoms))
                for sig, kind, atoms in zip(self.unit_sigs, self.unit_kinds, self.unit_atoms)
            ],
            dtype=np.int64,
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> str:
    return hashlib.sha256((_CODE_DIR / "strong_backbone_bucket_features.py").read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# extraction (compatible extension of sba.extract_batch: same forward order)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_batch_bucket(model, batch, mols_raw: Sequence[Any], ys: Sequence[float], device):
    """Frozen forward over one batch; v3 fields + bucket-complete aggregates.

    The forward and the per-pair assignment replicate
    ``strong_backbone_attribution.extract_batch`` exactly (same operations,
    same order, same float64 accumulation of float32 rows); only the stored
    fields are extended.
    """
    batch = batch.to(device)
    n_graphs = int(batch.global_context.shape[0])
    coord = model.code(batch.dict_phi)
    E = model.environments_masked(coord, batch, audit.AuditMask())
    pair_value, relation_used = sba._replicate_pair_path(model, E, batch)

    unary_ref = v0.pool_moments(E, batch.batch, n_graphs)
    pair_graph = batch.batch[batch.pair_index[0]]
    pair_ref = v0.pool_pair_moments(pair_value, pair_graph, batch.pair_bucket, n_graphs)
    _pred_masked, aux = model(batch, mask=cm.C6_MASK, return_aux=True)

    E_np = E.detach().cpu().numpy().astype(np.float64)
    PV_np = pair_value.detach().cpu().numpy().astype(np.float64)
    unary_ref_np = unary_ref.detach().cpu().numpy().astype(np.float64)
    pair_ref_np = pair_ref.detach().cpu().numpy().astype(np.float64)
    unary_m_np = aux["unary"].detach().cpu().numpy().astype(np.float64)
    pair_m_np = aux["relation_readout"].detach().cpu().numpy().astype(np.float64)
    rel_used_np = relation_used.detach().cpu().numpy().astype(np.float64)

    ptr = batch.ptr.cpu().numpy()
    src = batch.pair_index[0].cpu().numpy()
    dst = batch.pair_index[1].cpu().numpy()
    buckets = batch.pair_bucket.cpu().numpy()
    pair_graph_np = pair_graph.cpu().numpy()

    out: list[MolBucketFeatures] = []
    aud = {
        "unary_masked_vs_ref": 0.0, "pair_masked_vs_ref": 0.0,
        "unary_first_recon": 0.0, "unary_second_recon": 0.0,
        "pair_first_recon": 0.0, "pair_second_recon": 0.0,
        "log_count_recon": 0.0, "masked_count_blocks_max": 0.0,
        "pair_total_count_gap": 0.0,
        "unit_marginal_containment": 0.0, "rel_marginal_containment": 0.0,
        "unit_bucket_count_containment": 0.0, "rel_bucket_count_containment": 0.0,
    }
    blk = 2 * PAIR_DIM + 1
    for k, mu in enumerate(mols_raw):
        lo, hi = int(ptr[k]), int(ptr[k + 1])
        n, n_u = mu.n_atoms, len(mu.unit_sigs)
        assert hi - lo == n
        rows = np.arange(lo, hi)
        p_lo = int(np.searchsorted(pair_graph_np, k, "left"))
        p_hi = int(np.searchsorted(pair_graph_np, k, "right"))
        p_src, p_dst = src[p_lo:p_hi] - lo, dst[p_lo:p_hi] - lo
        p_bucket = buckets[p_lo:p_hi]
        n_pairs = n * (n - 1) // 2
        assert len(p_src) == n_pairs and np.all(p_src < p_dst)

        # atom -> unit (each atom exactly once)
        atom_unit = np.full(n, -1, dtype=np.int64)
        for uid, atoms in enumerate(mu.unit_atoms):
            atom_unit[np.asarray(atoms, dtype=np.int64)] = uid
        assert (atom_unit >= 0).all() and len(mu.unit_sigs) == n_u

        a_src, a_dst = atom_unit[p_src], atom_unit[p_dst]
        intra = a_src == a_dst
        pv_k = PV_np[p_lo:p_hi]
        E_k = E_np[rows]

        # ---- unit aggregates (float64 accumulation of float32 rows) ----
        unit_n_roots = np.bincount(atom_unit, minlength=n_u).astype(np.float64)
        assert float(unit_n_roots.sum()) == n, "atom count conservation failed"
        aud["log_count_recon"] = max(aud["log_count_recon"], abs(math.log1p(n) - unary_ref_np[k, 2 * ENV_DIM]))
        unit_sum_e = np.zeros((n_u, ENV_DIM))
        unit_sum_e2 = np.zeros((n_u, ENV_DIM))
        np.add.at(unit_sum_e, atom_unit, E_k)
        np.add.at(unit_sum_e2, atom_unit, E_k**2)

        unit_sum_pv = np.zeros((n_u, PAIR_DIM))
        unit_sum_pv2 = np.zeros((n_u, PAIR_DIM))
        unit_n_pv = np.zeros(n_u)
        unit_bucket_counts = np.zeros((n_u, N_BUCKETS))
        unit_bucket_sum = np.zeros((n_u, N_BUCKETS, PAIR_DIM))
        unit_bucket_sum2 = np.zeros((n_u, N_BUCKETS, PAIR_DIM))
        if intra.any():
            ai = a_src[intra]
            np.add.at(unit_sum_pv, ai, pv_k[intra])
            np.add.at(unit_sum_pv2, ai, pv_k[intra] ** 2)
            np.add.at(unit_n_pv, ai, 1.0)
            for b in range(N_BUCKETS):
                m = intra & (p_bucket == b)
                if m.any():
                    np.add.at(unit_bucket_counts[:, b], a_src[m], 1.0)
                    np.add.at(unit_bucket_sum[:, b, :], a_src[m], pv_k[m])
                    np.add.at(unit_bucket_sum2[:, b, :], a_src[m], pv_k[m] ** 2)

        # ---- cross-unit groups (unordered unit pair, lo<hi) ----
        lo_pair = np.minimum(a_src, a_dst)[~intra]
        hi_pair = np.maximum(a_src, a_dst)[~intra]
        uniq, inverse = np.unique(np.stack([lo_pair, hi_pair], 1), axis=0, return_inverse=True)
        n_rel = len(uniq)
        pv_x = pv_k[~intra]
        rel_sum_pv = np.zeros((n_rel, PAIR_DIM))
        rel_sum_pv2 = np.zeros((n_rel, PAIR_DIM))
        rel_count = np.zeros(n_rel)
        rel_bucket_counts = np.zeros((n_rel, N_BUCKETS))
        rel_bucket_sum = np.zeros((n_rel, N_BUCKETS, PAIR_DIM))
        rel_bucket_sum2 = np.zeros((n_rel, N_BUCKETS, PAIR_DIM))
        rel_mean_relation = np.zeros((n_rel, REL_USED))
        np.add.at(rel_sum_pv, inverse, pv_x)
        np.add.at(rel_sum_pv2, inverse, pv_x**2)
        np.add.at(rel_count, inverse, 1.0)
        np.add.at(rel_mean_relation, inverse, rel_used_np[p_lo:p_hi][~intra])
        rel_mean_relation /= np.maximum(rel_count[:, None], 1.0)
        for b in range(N_BUCKETS):
            m = (p_bucket[~intra]) == b
            np.add.at(rel_bucket_counts[:, b], inverse[m], 1.0)
            if m.any():
                np.add.at(rel_bucket_sum[:, b, :], inverse[m], pv_x[m])
                np.add.at(rel_bucket_sum2[:, b, :], inverse[m], pv_x[m] ** 2)

        # physical inter-unit bonds per observed unit pair (0 if none)
        bond_lookup = {pr: mu.rel_feat[j, :BOND_CATS].copy() for j, pr in enumerate(mu.rel_pairs)}
        rel_bond_hist = np.zeros((n_rel, BOND_CATS))
        for j in range(n_rel):
            key = (int(uniq[j, 0]), int(uniq[j, 1]))
            if key in bond_lookup:
                rel_bond_hist[j] = bond_lookup[key]
        observed = {(int(a), int(b)) for a, b in uniq}
        for pr in mu.rel_pairs:
            assert pr in observed, ("bonded unit pair missing from cross pairs", mu.mol_index, pr)

        # ---- reconstruction + conservation checks (float64 vs float32 pool) ----
        ref1 = unary_ref_np[k, :ENV_DIM]
        ref2 = unary_ref_np[k, ENV_DIM : 2 * ENV_DIM]
        aud["unary_first_recon"] = max(aud["unary_first_recon"], float(np.abs(unit_sum_e.sum(0) - ref1).max() / max(np.abs(ref1).max(), 1.0)))
        aud["unary_second_recon"] = max(aud["unary_second_recon"], float(np.abs(unit_sum_e2.sum(0) - ref2).max() / max(np.abs(ref2).max(), 1.0)))
        total_bucket_pairs = 0
        for b in range(N_BUCKETS):
            base = b * blk
            # first/second moment reconstruction from the STORED bucket arrays
            rec1 = unit_bucket_sum[:, b, :].sum(0) + rel_bucket_sum[:, b, :].sum(0)
            rec2 = unit_bucket_sum2[:, b, :].sum(0) + rel_bucket_sum2[:, b, :].sum(0)
            refb1 = pair_ref_np[k, base : base + PAIR_DIM]
            refb2 = pair_ref_np[k, base + PAIR_DIM : base + 2 * PAIR_DIM]
            aud["pair_first_recon"] = max(aud["pair_first_recon"], float(np.abs(rec1 - refb1).max() / max(np.abs(refb1).max(), 1.0)))
            aud["pair_second_recon"] = max(aud["pair_second_recon"], float(np.abs(rec2 - refb2).max() / max(np.abs(refb2).max(), 1.0)))
            b_intra = float(unit_bucket_counts[:, b].sum())
            b_cross = float(rel_bucket_counts[:, b].sum())
            total_bucket_pairs += int(b_intra + b_cross)
            aud["log_count_recon"] = max(aud["log_count_recon"], abs(math.log1p(int(b_intra + b_cross)) - pair_ref_np[k, base + 2 * PAIR_DIM]))
        aud["pair_total_count_gap"] = max(aud["pair_total_count_gap"], abs(total_bucket_pairs - n_pairs))
        # the new bucket arrays strictly contain the old marginal fields
        aud["unit_marginal_containment"] = max(
            aud["unit_marginal_containment"],
            float(np.abs(unit_bucket_sum.sum(1) - unit_sum_pv).max()),
            float(np.abs(unit_bucket_sum2.sum(1) - unit_sum_pv2).max()),
        )
        aud["unit_bucket_count_containment"] = max(
            aud["unit_bucket_count_containment"], float(np.abs(unit_bucket_counts.sum(1) - unit_n_pv).max())
        )
        if n_rel:
            aud["rel_marginal_containment"] = max(
                aud["rel_marginal_containment"],
                float(np.abs(rel_bucket_sum.sum(1) - rel_sum_pv).max()),
                float(np.abs(rel_bucket_sum2.sum(1) - rel_sum_pv2).max()),
            )
            aud["rel_bucket_count_containment"] = max(
                aud["rel_bucket_count_containment"], float(np.abs(rel_bucket_counts.sum(1) - rel_count).max())
            )
        aud["unary_masked_vs_ref"] = max(aud["unary_masked_vs_ref"], float(np.abs(unary_m_np[k, : 2 * ENV_DIM] - unary_ref_np[k, : 2 * ENV_DIM]).max()))
        masked_pair = pair_m_np[k].reshape(N_BUCKETS, blk)[:, : 2 * PAIR_DIM]
        ref_pair = pair_ref_np[k].reshape(N_BUCKETS, blk)[:, : 2 * PAIR_DIM]
        aud["pair_masked_vs_ref"] = max(aud["pair_masked_vs_ref"], float(np.abs(masked_pair - ref_pair).max()))
        aud["masked_count_blocks_max"] = max(
            aud["masked_count_blocks_max"],
            float(np.abs(unary_m_np[k, 2 * ENV_DIM]).max()),
            float(np.abs(pair_m_np[k].reshape(N_BUCKETS, blk)[:, -1]).max()),
        )

        ref_blocks = pair_ref_np[k].reshape(N_BUCKETS, blk)
        out.append(
            MolBucketFeatures(
                row=int(mu.mol_index),
                y=float(ys[k]),
                n_atoms=n,
                n_units=n_u,
                n_pairs=n_pairs,
                unit_sigs=list(mu.unit_sigs),
                unit_kinds=[int(x) for x in mu.unit_kinds],
                unit_atoms=[list(map(int, a)) for a in mu.unit_atoms],
                unit_desc=mu.desc.astype(np.float32),
                unit_sum_e=unit_sum_e.astype(np.float32),
                unit_sum_e2=unit_sum_e2.astype(np.float32),
                unit_sum_pv=unit_sum_pv.astype(np.float32),
                unit_sum_pv2=unit_sum_pv2.astype(np.float32),
                unit_n_pv=unit_n_pv.astype(np.float32),
                rel_lo=uniq[:, 0].astype(np.int64),
                rel_hi=uniq[:, 1].astype(np.int64),
                rel_sum_pv=rel_sum_pv.astype(np.float32),
                rel_sum_pv2=rel_sum_pv2.astype(np.float32),
                rel_count=rel_count.astype(np.float32),
                rel_bucket_counts=rel_bucket_counts.astype(np.float32),
                rel_bond_hist=rel_bond_hist.astype(np.float32),
                rel_mean_relation=rel_mean_relation.astype(np.float32),
                ctx_structure=batch.global_context.detach().cpu().numpy()[k, :CTX_STRUCTURE].astype(np.float32),
                topology=batch.topology_features.detach().cpu().numpy()[k].astype(np.float32),
                unit_bucket_counts=unit_bucket_counts.astype(np.float32),
                unit_bucket_sum_pv=unit_bucket_sum.astype(np.float32),
                unit_bucket_sum_pv2=unit_bucket_sum2.astype(np.float32),
                rel_bucket_sum_pv=rel_bucket_sum.astype(np.float32),
                rel_bucket_sum_pv2=rel_bucket_sum2.astype(np.float32),
                pair_ref_first=ref_blocks[:, :PAIR_DIM].astype(np.float32),
                pair_ref_second=ref_blocks[:, PAIR_DIM : 2 * PAIR_DIM].astype(np.float32),
                pair_ref_logcount=ref_blocks[:, -1].astype(np.float32),
            )
        )
    return out, aud


def extract_all_bucket_features(
    dataset,
    train_rows,
    mols_raw,
    ys: np.ndarray,
    *,
    device="cpu",
    batch_size: int = EXTRACT_BATCH,
    rows: Sequence[int] | None = None,
    model=None,
    log=print,
):
    """Frozen bucket-complete extraction over ``rows`` (default all 10000)."""
    if model is None:
        model, _prep, _b10 = sba.load_frozen_backbone(log=log)
    if rows is None:
        rows = list(range(len(train_rows)))
    rows = [int(r) for r in rows]
    features: list[MolBucketFeatures] = []
    keys = (
        "unary_masked_vs_ref", "pair_masked_vs_ref", "unary_first_recon",
        "unary_second_recon", "pair_first_recon", "pair_second_recon",
        "log_count_recon", "masked_count_blocks_max", "pair_total_count_gap",
        "unit_marginal_containment", "rel_marginal_containment",
        "unit_bucket_count_containment", "rel_bucket_count_containment",
    )
    totals = {k: 0.0 for k in keys}
    started = time.perf_counter()
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        lists = [train_rows[i] for i in chunk]
        batch = p1.env_collate(lists)
        batch.y = torch.zeros(len(lists))
        mols, aud = extract_batch_bucket(model, batch, [mols_raw[i] for i in chunk], [float(ys[i]) for i in chunk], device)
        features.extend(mols)
        for k, v in aud.items():
            totals[k] = max(totals[k], v)
        if (s // batch_size) % 10 == 0:
            log(f"[extract-bucket] {s + len(chunk)}/{len(rows)} rows  ({time.perf_counter() - started:.0f}s)")
    log(f"[extract-bucket] done {len(features)} molecules in {time.perf_counter() - started:.0f}s")
    log("[extract-bucket] audit=" + json_dumps_compact(totals))
    failed = {k: v for k, v in totals.items() if (
        v > POOL_RTOL if k.endswith("_recon") or k.endswith("_containment") else v > POOL_ATOL
    )}
    if failed:
        raise RuntimeError(f"bucket reconstruction/conservation thresholds exceeded: {failed}")
    return features, totals


def json_dumps_compact(d: dict) -> str:
    return "{" + ", ".join(f"'{k}': {v:.3e}" for k, v in d.items()) + "}"


# ---------------------------------------------------------------------------
# cache save/load + stored-cache-only verification
# ---------------------------------------------------------------------------


def _split_hashes() -> dict[str, Any]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)
    out = {}
    for name, rows in idx.items():
        h = hashlib.sha256(np.ascontiguousarray(np.sort(rows)).tobytes()).hexdigest()[:16]
        out[f"rows_{name}_sha16"] = h
    out["smiles_table_sha16"] = hashlib.sha256(cf.SMILES_TABLE.read_bytes()).hexdigest()[:16]
    return out


def build_provenance() -> dict[str, Any]:
    from cscl_units import __file__ as _cu_file

    prov = {
        "feature_version": FEATURE_VERSION,
        "protocol": PROTOCOL_ID,
        "code_fp_bucket_module": code_fingerprint(),
        "code_fp_strong_backbone_attribution": sba.code_fingerprint(),
        "code_fp_cscl_units_features": cf.code_fingerprint(),
        "ringchain_feature_version": cf.FEATURE_VERSION,
        "ringchain_signature_version": cf.SIGNATURE_VERSION,
        "frozen_checkpoint_sha16": _sha256_file(sba.SOUP_STATE),
        "prep_fit7200_sha16": _sha256_file(sba.PREP_NPZ),
        "legacy_prep_sha16": _sha256_file(sba.LEGACY_PREP),
        "pair_bucket_rule": PAIR_BUCKET_RULE,
        "n_distance_buckets": N_BUCKETS,
        "env_dim": ENV_DIM,
        "pair_dim": PAIR_DIM,
        "git_commit": _git_commit(),
        **_split_hashes(),
    }
    del _cu_file
    return prov


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def save_bucket_cache(path: Path, features: Sequence[MolBucketFeatures], audit: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **build_provenance(),
            "audit": dict(audit),
            "features": list(features),
        },
        path,
    )


def load_bucket_cache(path: Path | None = None, log=print, verify: bool = True):
    """Load the v3d cache; optionally verify reconstruction from STORED arrays."""
    path = Path(path) if path is not None else BUCKET_CACHE
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("feature_version") != FEATURE_VERSION or blob.get("code_fp_bucket_module") != code_fingerprint():
        log(f"[cache] stale v3d bucket cache at {path}; rebuilding")
        return None
    log(f"[cache] loaded v3d bucket features ({len(blob['features'])} molecules) from {path.name}")
    if verify:
        res = verify_stored_cache(blob, log=log)
        if not res["ok"]:
            raise RuntimeError(f"stored-cache reconstruction failed: {res}")
    return blob


def verify_stored_cache(blob: dict, log=print, max_mols: int | None = None) -> dict[str, Any]:
    """Rebuild Full-Y's pair pooled first/second moments from the STORED cache
    arrays alone (float64) and check the frozen gates.  This is the
    information-fidelity gate: the training inputs are exactly the stored
    unit/relation bucket arrays, so proving reconstruction from storage (not
    from a live forward) closes the v3 information-loss gap."""
    feats = blob["features"] if max_mols is None else blob["features"][:max_mols]
    worst = {"first_rel": 0.0, "second_rel": 0.0, "logcount_abs": 0.0,
             "unit_marginal": 0.0, "rel_marginal": 0.0,
             "unit_count_marginal": 0.0, "rel_count_marginal": 0.0,
             "total_count_gap": 0.0}
    n_pairs_total = 0
    per_bucket_total = np.zeros(N_BUCKETS, dtype=np.int64)
    for m in feats:
        cnt = m.unit_bucket_counts.astype(np.float64)
        s1 = m.unit_bucket_sum_pv.astype(np.float64)
        s2 = m.unit_bucket_sum_pv2.astype(np.float64)
        rcnt = m.rel_bucket_counts.astype(np.float64)
        r1 = m.rel_bucket_sum_pv.astype(np.float64)
        r2 = m.rel_bucket_sum_pv2.astype(np.float64)
        ref1 = m.pair_ref_first.astype(np.float64)
        ref2 = m.pair_ref_second.astype(np.float64)
        reflc = m.pair_ref_logcount.astype(np.float64)
        for b in range(N_BUCKETS):
            rec1 = s1[:, b, :].sum(0) + r1[:, b, :].sum(0)
            rec2 = s2[:, b, :].sum(0) + r2[:, b, :].sum(0)
            worst["first_rel"] = max(worst["first_rel"], float(np.abs(rec1 - ref1[b]).max() / max(np.abs(ref1[b]).max(), 1.0)))
            worst["second_rel"] = max(worst["second_rel"], float(np.abs(rec2 - ref2[b]).max() / max(np.abs(ref2[b]).max(), 1.0)))
            bc = cnt[:, b].sum() + rcnt[:, b].sum()
            worst["logcount_abs"] = max(worst["logcount_abs"], abs(math.log1p(bc) - reflc[b]))
            per_bucket_total[b] += int(bc)
        worst["unit_marginal"] = max(worst["unit_marginal"], float(np.abs(s1.sum(1) - m.unit_sum_pv).max()), float(np.abs(s2.sum(1) - m.unit_sum_pv2).max()))
        if len(rcnt):
            worst["rel_marginal"] = max(worst["rel_marginal"], float(np.abs(r1.sum(1) - m.rel_sum_pv).max()), float(np.abs(r2.sum(1) - m.rel_sum_pv2).max()))
            worst["rel_count_marginal"] = max(worst["rel_count_marginal"], float(np.abs(rcnt.sum(1) - m.rel_count).max()))
        worst["unit_count_marginal"] = max(worst["unit_count_marginal"], float(np.abs(cnt.sum(1) - m.unit_n_pv).max()))
        worst["total_count_gap"] = max(worst["total_count_gap"], abs(int(cnt.sum() + rcnt.sum()) - m.n_pairs))
        n_pairs_total += int(m.n_pairs)
    gates = {
        "first_rel_ok": worst["first_rel"] <= POOL_RTOL,
        "second_rel_ok": worst["second_rel"] <= POOL_RTOL,
        "logcount_ok": worst["logcount_abs"] <= POOL_ATOL,
        "unit_marginal_ok": worst["unit_marginal"] <= POOL_ATOL,
        "rel_marginal_ok": worst["rel_marginal"] <= POOL_ATOL,
        "count_marginal_ok": worst["unit_count_marginal"] <= POOL_ATOL and worst["rel_count_marginal"] <= POOL_ATOL,
        "total_count_ok": worst["total_count_gap"] == 0,
    }
    res = {
        "n_molecules": len(feats),
        "n_pairs_total": n_pairs_total,
        "per_bucket_pair_totals": per_bucket_total.tolist(),
        "worst": worst,
        **gates,
        "ok": bool(all(gates.values())),
    }
    log(f"[verify-stored] ok={res['ok']} worst={ {k: f'{v:.2e}' for k, v in worst.items()} }")
    return res


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------


def compare_with_old_cache(features: Sequence[MolBucketFeatures], log=print, max_mols: int | None = None) -> dict[str, Any]:
    """Compatibility check: shared fields of the new extraction match the
    committed v3 cache within float tolerance (device differences allowed);
    exact structural fields must match exactly."""
    blob = torch.load(OLD_CACHE, map_location="cpu", weights_only=False)
    old = {m.row: m for m in blob["features"]}
    exact_bad = 0
    float_worst = 0.0
    checked = 0
    for m in features if max_mols is None else features[:max_mols]:
        o = old[m.row]
        checked += 1
        exact_ok = (
            o.n_atoms == m.n_atoms and o.n_units == m.n_units and o.n_pairs == m.n_pairs
            and o.unit_sigs == m.unit_sigs and o.unit_kinds == m.unit_kinds
            and o.unit_atoms == m.unit_atoms
            and np.array_equal(o.rel_lo, m.rel_lo) and np.array_equal(o.rel_hi, m.rel_hi)
            and np.array_equal(np.round(o.y, 6), np.round(m.y, 6))
        )
        if not exact_ok:
            exact_bad += 1
        for name in (
            "unit_desc", "unit_sum_e", "unit_sum_e2", "unit_sum_pv", "unit_sum_pv2",
            "unit_n_pv", "rel_sum_pv", "rel_sum_pv2", "rel_count", "rel_bucket_counts",
            "rel_bond_hist", "rel_mean_relation", "ctx_structure", "topology",
        ):
            a, b = getattr(o, name), getattr(m, name)
            if a.shape != b.shape or a.size == 0:
                if a.shape != b.shape:
                    exact_bad += 1
                continue
            denom = max(np.abs(a).max(), 1e-6)
            float_worst = max(float_worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / denom))
    res = {
        "checked_molecules": checked,
        "exact_field_mismatches": int(exact_bad),
        "float_field_worst_rel": float(float_worst),
        "ok": bool(exact_bad == 0 and float_worst <= 1e-3),
    }
    log(f"[old-cache-compat] {res}")
    if not res["ok"]:
        raise RuntimeError("new extraction is not compatible with the committed v3 cache")
    return res


def build_bucket_cache(device: str = "cpu", log=print, rows: Sequence[int] | None = None) -> dict:
    """Full label-free build: alignment -> extraction -> gates -> save."""
    from run_prediction_feasibility_v2 import apply_prep_fit7200, verify_row_alignment

    model, prep, blob10 = sba.load_frozen_backbone(log=log)
    model = model.to(torch.device(device))
    train = sba.load_train_rows(log=log)
    verify_row_alignment(train, log=log)
    apply_prep_fit7200(train, prep, blob10)
    dataset = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    alignment = sba.check_atom_alignment(train, dataset, None, log=log)
    mols_raw = sba.raw_ringchain(dataset, smiles)
    ys = np.asarray([float(dataset[i].y.reshape(-1)[0]) for i in range(10000)])
    features, audit = extract_all_bucket_features(dataset, train, mols_raw, ys, model=model, device=device, rows=rows, log=log)
    compat = compare_with_old_cache(features, log=log, max_mols=None if rows is None else min(len(features), 640))
    save_bucket_cache(BUCKET_CACHE, features, audit)
    log(f"[cache] saved {BUCKET_CACHE}")
    return {"audit": audit, "alignment": alignment, "compat": compat, "n": len(features)}


if __name__ == "__main__":
    # Rebuild through the canonical module identity so the cached dataclasses
    # pickle as ``strong_backbone_bucket_features.MolBucketFeatures`` (loading
    # from any other entry point must keep working).
    import importlib

    torch.set_num_threads(8)
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    _mod = importlib.import_module(__name__ if __name__ != "__main__" else "strong_backbone_bucket_features")
    if __name__ == "__main__":
        _self = importlib.import_module("strong_backbone_bucket_features")
        _self.build_bucket_cache(device=dev)
    else:
        build_bucket_cache(device=dev)
