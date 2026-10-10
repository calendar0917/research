"""SB-CSCL (cscl-v3) attribution core: frozen Full-Y backbone -> RINGCHAIN units.

Research track: ``structure-contribution``, round ``cscl-v3`` (strong-backbone
structural contribution learning).  Protocol:
``configs/strong_backbone_contribution_v3.yaml`` (frozen before any training).

Question
--------
Can the frozen Full-Y seed-0 encoder (dev MAE 0.11641, non-GNN, committed in
``results/feasibility_v2_gpu/``) be decomposed, without retraining it, into
RINGCHAIN structural units and explicit unit pairs, such that (a) every rooted
atom environment and every node pair of the original computation is assigned
to exactly one unit / one unit-internal or cross-unit relation, and (b) the
graph-level pooled tensors consumed by the frozen reader can be reconstructed
from unit/relation-level statistics inside float tolerance?

Mapping evidence chain (Phase A; asserted per molecule in this module)
----------------------------------------------------------------------
1. ``dict_phi`` row i == patch rooted at atom i: ``prepare_env`` asserts
   ``n == num_nodes == graph.n == phi.shape[0]`` and ``node_types ==
   molecule.atom_idx``; ``fsar_r2_ar0.build_phi`` emits one row per graph node
   (= PyG atom order).  Re-checked numerically on sampled molecules.
2. ``dict_atom[i] == raw x[i]`` (checked on every molecule).
3. ``pair_index`` covers all atom pairs i<j exactly once per molecule
   (``_graph_record`` loops over all center pairs; re-checked per molecule);
   ``pair_bucket == argmax(relation[:5])``.
4. Batch assembly: PyG offsets ``pair_index`` by ``num_nodes`` (``__inc__``);
   ``env_collate`` offsets occ/bond fields by ``ptr``.  Batch-invariance of
   aggregates is tested separately.
5. RINGCHAIN partition (cscl-correctness-v1): every atom in exactly one unit;
   every physical bond intra- or inter-unit; hash 1-WL signatures.
6. Pooled reconstruction: unary [sum E, sum E^2, log1p(count)] and per-bucket
   pair moments rebuild from unit/relation aggregates; counts and bucket
   histograms conserve; C6-masked reader inputs reproduced as consumed.

This module never reads official valid/test, never reads
``target_decomposition`` / ``g`` / ``ell`` / ``s`` / ``c``, and never uses the
frozen model's prediction as an input or a supervision target of the new heads.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import cscl_units as cu  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    e2e_dictenv_clean_mechanism_v1 as cm,
    e2e_dictenv_common_subspace_dictionary_v1 as cssd,
    e2e_dictenv_h1_clarity_audit as audit,
    e2e_dictenv_p1 as p1,
    e2e_dictenv_scale_v1 as sc,
    e2e_dictenv_v0 as v0,
)

PROTOCOL_ID = "cscl-v3-strong-backbone-attribution"
FEATURE_VERSION = "cscl-v3-frozen-features-v1"

FEAT_CACHE = REPO_ROOT / "data/cache/cscl_v3_frozen_features.pt"
SOUP_STATE = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_soup_state.pt"
PREP_NPZ = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/prep_fit7200.npz"
LEGACY_PREP = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/all_train_prep.npz"
DEV_PREDS_NPZ = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_dev_preds.npz"
MON_PREDS_NPZ = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_monitor_preds.npz"

ENV_DIM = int(sc.FULL.d)               # 144
PAIR_DIM = int(sc.FULL.p)              # 48
N_BUCKETS = int(p1.DISTANCE_BUCKETS)   # 5
REL_USED = int(p1.RELATION_WIDTH)      # 15
CTX_DIM = int(p1.GLOBAL_WIDTH)         # 62
CTX_STRUCTURE = 30            # structure_short + structure_long (C6 keeps these)
TOPO_DIM = int(p1.TOPOLOGY_IN)         # 25
DESC_DIM = int(cf.DESC_DIM)            # 36
BOND_CATS = int(cu.BOND_CATEGORIES)    # 4
EXTRACT_BATCH = 128

REPLAY_ATOL = 5e-3        # float32 CPU forward vs saved float32 GPU forward
REPLAY_MAE_TOL = 1e-3
POOL_RTOL = 2e-4          # float64 aggregate reconstruction vs float32 pooling
POOL_ATOL = 2e-3


# ---------------------------------------------------------------------------
# frozen backbone
# ---------------------------------------------------------------------------


def load_frozen_backbone(prep_path=None, soup_path=None, log=print):
    """Build the FULL model, load the committed seed-0 soup state, freeze it."""
    import torch

    prep_path = Path(prep_path) if prep_path is not None else PREP_NPZ
    soup_path = Path(soup_path) if soup_path is not None else SOUP_STATE
    for p in (prep_path, soup_path, LEGACY_PREP):
        if not Path(p).exists():
            raise FileNotFoundError(f"required artifact missing: {p}")
    prep = {k: z for k, z in np.load(prep_path, allow_pickle=False).items()}
    blob10 = {k: z for k, z in np.load(LEGACY_PREP, allow_pickle=False).items()}
    subspace = cssd.CommonSubspace(
        components=np.asarray(prep["U_components"], np.float32),
        rms=np.asarray(prep["U_rms"], np.float32),
    )
    model = sc.build_scale_model(np.asarray(prep["D_fit"], np.float32), 0, subspace, sc.FULL, scale_seed=0)
    state = torch.load(soup_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    log(f"[backbone] frozen Full-Y soup loaded ({Path(soup_path).name}); parameters frozen")
    return model, prep, blob10


def load_train_rows(log=print):
    """Encoded official-train cache + env attachment (v2 driver path)."""
    from run_prediction_feasibility_v2 import load_train_rows as _load

    return _load(log=log)


def raw_ringchain(dataset, smiles, rows=None):
    """RINGCHAIN summaries (cscl-correctness-v1) for the given rows, label-free."""
    if rows is None:
        rows = range(len(dataset))
    return [
        cf.molecule_units(
            int(i),
            dataset[i].x.numpy(),
            dataset[i].edge_index.numpy(),
            dataset[i].edge_attr.numpy(),
            str(smiles[i]),
        )
        for i in rows
    ]


# ---------------------------------------------------------------------------
# per-molecule frozen features
# ---------------------------------------------------------------------------


@dataclass
class MolFrozen:
    """Frozen-feature summary of one molecule.

    Training inputs of the S/O heads are derived only from these aggregates;
    the per-root ``E`` rows and per-pair ``pair_value`` rows of the frozen
    encoder are not stored (deterministically recomputable from committed
    artifacts + code).  Audit-only arrays (bucket split) are not stored.
    """

    row: int
    y: float
    n_atoms: int
    n_units: int
    n_pairs: int
    unit_sigs: list[str]
    unit_kinds: list[int]
    unit_atoms: list[list[int]]
    unit_desc: np.ndarray              # [n_units, 36]
    unit_sum_e: np.ndarray             # [n_units, 144]
    unit_sum_e2: np.ndarray            # [n_units, 144]
    unit_sum_pv: np.ndarray            # [n_units, 48]  intra-unit pair values
    unit_sum_pv2: np.ndarray           # [n_units, 48]
    unit_n_pv: np.ndarray              # [n_units]
    rel_lo: np.ndarray                 # [n_rel] unordered unit pair, lo < hi
    rel_hi: np.ndarray
    rel_sum_pv: np.ndarray             # [n_rel, 48]
    rel_sum_pv2: np.ndarray            # [n_rel, 48]
    rel_count: np.ndarray              # [n_rel] Full-Y pair instances
    rel_bucket_counts: np.ndarray      # [n_rel, 5]
    rel_bond_hist: np.ndarray          # [n_rel, 4] physical inter-unit bonds
    rel_mean_relation: np.ndarray      # [n_rel, 15] mean P1 relation slice
    ctx_structure: np.ndarray          # [30]
    topology: np.ndarray               # [25]

    def type_ids(self, vocab) -> np.ndarray:
        return np.asarray(
            [
                vocab.to_id(sig, kind, len(atoms))
                for sig, kind, atoms in zip(self.unit_sigs, self.unit_kinds, self.unit_atoms)
            ],
            dtype=np.int64,
        )


def _replicate_pair_path(model, E, batch):
    """Exact replication of the frozen pair path (AuditModel.forward lines)."""
    import torch

    source = batch.pair_index[0]
    target = batch.pair_index[1]
    u = model.pair_projection(E)
    left = u[source]
    right = u[target]
    relation_input = batch.pair_relation[:, list(p1.P1_RELATION_INDICES)]
    relation = model.relation_encoder(relation_input)
    product = left * right
    gate = 1.0 + torch.tanh(model.distance_gate(batch.pair_bucket))
    pair_input = torch.cat(
        [left + right, torch.abs(left - right), product * gate, relation], dim=1
    )
    return model.pair_encoder(pair_input), relation_input


@torch.no_grad()
def extract_batch(model, batch, mols_raw: Sequence[Any], ys: Sequence[float], device):
    """Frozen forward over one batch + per-molecule assignment and checks.

    ``mols_raw[k]`` is the RINGCHAIN summary (``cf.MolUnits``) of the k-th
    molecule in the batch (order preserved).  Returns ``(list[MolFrozen],
    audit_dict)``.
    """
    import torch

    batch = batch.to(device)
    n_graphs = int(batch.global_context.shape[0])
    coord = model.code(batch.dict_phi)
    E = model.environments_masked(coord, batch, audit.AuditMask())
    pair_value, relation_used = _replicate_pair_path(model, E, batch)

    unary_ref = v0.pool_moments(E, batch.batch, n_graphs)
    pair_graph = batch.batch[batch.pair_index[0]]
    pair_ref = v0.pool_pair_moments(pair_value, pair_graph, batch.pair_bucket, n_graphs)
    _, aux = model(batch, mask=cm.C6_MASK, return_aux=True)

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

    out: list[MolFrozen] = []
    aud = {
        "unary_masked_vs_ref": 0.0, "pair_masked_vs_ref": 0.0,
        "unary_first_recon": 0.0, "unary_second_recon": 0.0,
        "pair_first_recon": 0.0, "pair_second_recon": 0.0,
        "log_count_recon": 0.0, "masked_count_blocks_max": 0.0,
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
        intra_b_counts = np.zeros((n_u, N_BUCKETS))
        intra_b_sum = np.zeros((n_u, N_BUCKETS, PAIR_DIM))
        intra_b_sq = np.zeros((n_u, N_BUCKETS, PAIR_DIM))
        if intra.any():
            ai = a_src[intra]
            np.add.at(unit_sum_pv, ai, pv_k[intra])
            np.add.at(unit_sum_pv2, ai, pv_k[intra] ** 2)
            np.add.at(unit_n_pv, ai, 1.0)
            for b in range(N_BUCKETS):
                m = intra & (p_bucket == b)
                if m.any():
                    np.add.at(intra_b_counts[:, b], a_src[m], 1.0)
                    np.add.at(intra_b_sum[:, b, :], a_src[m], pv_k[m])
                    np.add.at(intra_b_sq[:, b, :], a_src[m], pv_k[m] ** 2)

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
        rel_mean_relation = np.zeros((n_rel, REL_USED))
        cross_b_counts = np.zeros(N_BUCKETS)
        cross_b_sum = np.zeros((N_BUCKETS, PAIR_DIM))
        cross_b_sq = np.zeros((N_BUCKETS, PAIR_DIM))
        np.add.at(rel_sum_pv, inverse, pv_x)
        np.add.at(rel_sum_pv2, inverse, pv_x**2)
        np.add.at(rel_count, inverse, 1.0)
        np.add.at(rel_mean_relation, inverse, rel_used_np[p_lo:p_hi][~intra])
        rel_mean_relation /= np.maximum(rel_count[:, None], 1.0)
        for b in range(N_BUCKETS):
            m = (p_bucket[~intra]) == b
            np.add.at(rel_bucket_counts[:, b], inverse[m], 1.0)
            cross_b_counts[b] = float(m.sum())
            if m.any():
                cross_b_sum[b] = pv_x[m].sum(0)
                cross_b_sq[b] = (pv_x[m] ** 2).sum(0)

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

        # ---- reconstruction checks (float64 aggregates vs float32 pool) ----
        ref1 = unary_ref_np[k, :ENV_DIM]
        ref2 = unary_ref_np[k, ENV_DIM : 2 * ENV_DIM]
        aud["unary_first_recon"] = max(aud["unary_first_recon"], float(np.abs(unit_sum_e.sum(0) - ref1).max() / max(np.abs(ref1).max(), 1.0)))
        aud["unary_second_recon"] = max(aud["unary_second_recon"], float(np.abs(unit_sum_e2.sum(0) - ref2).max() / max(np.abs(ref2).max(), 1.0)))
        # per-bucket pair counts (log1p of the total per-bucket count) are
        # checked inside the bucket loop below; the unary count block was
        # checked above against log1p(n_atoms).  Counts are always
        # re-aggregated from totals, never by summing local log1p values.
        for b in range(N_BUCKETS):
            base = b * blk
            rec1 = intra_b_sum[:, b, :].sum(0) + cross_b_sum[b]
            rec2 = intra_b_sq[:, b, :].sum(0) + cross_b_sq[b]
            refb1 = pair_ref_np[k, base : base + PAIR_DIM]
            refb2 = pair_ref_np[k, base + PAIR_DIM : base + 2 * PAIR_DIM]
            aud["pair_first_recon"] = max(aud["pair_first_recon"], float(np.abs(rec1 - refb1).max() / max(np.abs(refb1).max(), 1.0)))
            aud["pair_second_recon"] = max(aud["pair_second_recon"], float(np.abs(rec2 - refb2).max() / max(np.abs(refb2).max(), 1.0)))
            aud["log_count_recon"] = max(aud["log_count_recon"], abs(math.log1p(int((p_bucket == b).sum())) - pair_ref_np[k, base + 2 * PAIR_DIM]))
        aud["unary_masked_vs_ref"] = max(aud["unary_masked_vs_ref"], float(np.abs(unary_m_np[k, : 2 * ENV_DIM] - unary_ref_np[k, : 2 * ENV_DIM]).max()))
        masked_pair = pair_m_np[k].reshape(N_BUCKETS, blk)[:, : 2 * PAIR_DIM]
        ref_pair = pair_ref_np[k].reshape(N_BUCKETS, blk)[:, : 2 * PAIR_DIM]
        aud["pair_masked_vs_ref"] = max(aud["pair_masked_vs_ref"], float(np.abs(masked_pair - ref_pair).max()))
        aud["masked_count_blocks_max"] = max(
            aud["masked_count_blocks_max"],
            float(np.abs(unary_m_np[k, 2 * ENV_DIM]).max()),
            float(np.abs(pair_m_np[k].reshape(N_BUCKETS, blk)[:, -1]).max()),
        )

        out.append(
            MolFrozen(
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
            )
        )
    return out, aud


# ---------------------------------------------------------------------------
# dataset-level extraction + alignment checks
# ---------------------------------------------------------------------------


def check_atom_alignment(train_rows, dataset, mols_raw, n_phi_rebuild: int = 24, log=print) -> dict[str, Any]:
    """Per-molecule root-order evidence (mapping prerequisites 1-3).

    * ``dict_atom[i] == raw x[i]`` on **all** molecules;
    * ``dict_phi`` row i == ``fsar_r2_ar0.build_phi(graph)[i]`` on sampled
      molecules (root order = atom order);
    * RINGCHAIN partition signatures reproduce ``cf.molecule_units`` (the
      same deterministic partition function is used by the extractor).
    """
    import torch

    from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _data_to_graph

    n_bad_atom = 0
    for i in range(len(train_rows)):
        if not np.array_equal(
            train_rows[i].dict_atom.numpy(), dataset[i].x.numpy().reshape(-1).astype(np.int64)
        ):
            n_bad_atom += 1
    phi_max = 0.0
    step = max(1, len(train_rows) // n_phi_rebuild)
    checked = 0
    for i in range(0, len(train_rows), step):
        graph, _nt, _et = _data_to_graph(dataset[i])
        fresh = r2.build_phi(graph).astype(np.float32)
        phi_max = max(phi_max, float(np.abs(fresh - train_rows[i].dict_phi.numpy()).max()))
        checked += 1
    ok = bool(n_bad_atom == 0 and phi_max == 0.0)
    log(f"[align] dict_atom==x on all rows: {n_bad_atom == 0}; phi rebuild max|d|={phi_max} on {checked} molecules")
    if not ok:
        raise RuntimeError("atom/root alignment failed; refusing to build unit attribution")
    return {"dict_atom_mismatches": n_bad_atom, "phi_rebuild_max_abs_diff": phi_max, "phi_rebuild_molecules": checked, "ok": ok}


def extract_all_features(
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
    """Frozen-feature extraction over ``rows`` (default all 10000, fixed order).

    Rows are batched sequentially (deterministic); each batch runs the frozen
    encoder once and emits one :class:`MolFrozen` per molecule.  All per-pair
    and per-root assignments are checked inside :func:`extract_batch`.
    """
    import torch

    if model is None:
        model, _prep, _b10 = load_frozen_backbone(log=log)
    if rows is None:
        rows = list(range(len(train_rows)))
    rows = [int(r) for r in rows]
    features: list[MolFrozen] = []
    totals = {k: 0.0 for k in (
        "unary_masked_vs_ref", "pair_masked_vs_ref", "unary_first_recon",
        "unary_second_recon", "pair_first_recon", "pair_second_recon",
        "log_count_recon", "masked_count_blocks_max",
    )}
    started = time.perf_counter()
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        lists = [train_rows[i] for i in chunk]
        batch = p1.env_collate(lists)
        batch.y = torch.zeros(len(lists))
        mols, aud = extract_batch(model, batch, [mols_raw[i] for i in chunk], [float(ys[i]) for i in chunk], device)
        features.extend(mols)
        for k, v in aud.items():
            totals[k] = max(totals[k], v)
        if (s // batch_size) % 10 == 0:
            log(f"[extract] {s + len(chunk)}/{len(rows)} rows  ({time.perf_counter() - started:.0f}s)")
    log(f"[extract] done {len(features)} molecules in {time.perf_counter() - started:.0f}s; audit={ {k: f'{v:.3e}' for k, v in totals.items()} }")
    failed = {k: v for k, v in totals.items() if (
        v > POOL_RTOL if k.endswith("_recon") else v > POOL_ATOL
    )}
    if failed:
        raise RuntimeError(f"pooled reconstruction/audit thresholds exceeded: {failed}")
    return features, totals


def fit_unit_vocab(features: Sequence[MolFrozen], fit_rows: Sequence[int], min_count: int = 3):
    """Fit-only type vocabulary + per-type descriptor means (v0_protocol sect. 2.4/3)."""
    fit_set = set(int(r) for r in fit_rows)
    fit_mols = [m for m in features if m.row in fit_set]
    vocab = cf.UnitVocab(min_count=min_count)
    vocab.fit([s for m in fit_mols for s in m.unit_sigs])
    acc: dict[int, list[np.ndarray]] = {}
    for m in fit_mols:
        for k, sig in enumerate(m.unit_sigs):
            acc.setdefault(vocab.to_id(sig, m.unit_kinds[k], len(m.unit_atoms[k])), []).append(m.unit_desc[k])
    type_means = {t: np.mean(np.stack(v), axis=0).astype(np.float32) for t, v in acc.items()}
    global_mean = np.mean(np.concatenate([m.unit_desc for m in fit_mols], 0), axis=0).astype(np.float32)
    rel_acc = np.concatenate(
        [m.rel_mean_relation - 0.0 for m in fit_mols if len(m.rel_lo)], axis=0
    )
    rel_mean = rel_acc.mean(0).astype(np.float32) if len(rel_acc) else np.zeros(REL_USED, np.float32)
    return vocab, type_means, global_mean, rel_mean


# ---------------------------------------------------------------------------
# cache + replay
# ---------------------------------------------------------------------------


def code_fingerprint() -> str:
    import hashlib

    return hashlib.sha256((_CODE_DIR / "strong_backbone_attribution.py").read_bytes()).hexdigest()[:16]


def save_features(path: Path, features: Sequence[MolFrozen], audit: Mapping[str, float]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_version": FEATURE_VERSION,
            "code_fp": code_fingerprint(),
            "protocol": PROTOCOL_ID,
            "audit": dict(audit),
            "features": list(features),
        },
        path,
    )


def load_features(path: Path | None = None, log=print):
    import torch

    path = Path(path) if path is not None else FEAT_CACHE
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("feature_version") != FEATURE_VERSION or blob.get("code_fp") != code_fingerprint():
        log(f"[cache] stale frozen-feature cache at {path}; rebuilding")
        return None
    log(f"[cache] loaded frozen features ({len(blob['features'])} molecules) from {path.name}")
    return blob


def replay_predictions(model, train_rows, idx, device="cpu", log=print) -> dict[str, float]:
    """Re-run the frozen soup on monitor+dev exactly like the committed run.

    Compares against the saved per-molecule exports (row, pred, y).  Device
    and library differences are allowed within REPLAY_* tolerances; no
    checkpoint or data modification is permitted on mismatch.
    """
    import torch

    from run_prediction_feasibility_v2 import predict_raw

    out: dict[str, float] = {}
    for name, path, rows_key in (
        ("dev", DEV_PREDS_NPZ, "dev"),
        ("monitor", MON_PREDS_NPZ, "monitor"),
    ):
        saved = np.load(path)
        rows = [int(i) for i in idx[rows_key]]
        assert np.array_equal(saved["row"], np.asarray(rows)), f"{name} row order mismatch"
        lists = [train_rows[i] for i in rows]
        pred = predict_raw(model, lists, torch.device(device))
        dy = float(np.abs(pred - saved["pred"].astype(np.float64)).max())
        mae_new = float(np.abs(pred - saved["y"].astype(np.float64)).mean())
        mae_saved = float(np.abs(saved["pred"].astype(np.float64) - saved["y"].astype(np.float64)).mean())
        out[f"{name}_max_abs_pred_diff"] = dy
        out[f"{name}_mae_replayed"] = mae_new
        out[f"{name}_mae_saved"] = mae_saved
        ok = dy <= REPLAY_ATOL and abs(mae_new - mae_saved) <= REPLAY_MAE_TOL
        out[f"{name}_ok"] = bool(ok)
        log(f"[replay] {name}: max|dpred|={dy:.3e} mae {mae_new:.5f} vs saved {mae_saved:.5f} ok={ok}")
        if not ok:
            raise RuntimeError(f"frozen-soup replay failed on {name}; do not modify the checkpoint - investigate first")
    return out
