"""Structural source/query partitions for the composition-transfer round.

For every training batch ``B`` (the same 128-molecule batch the ERM arm sees):

* ``R`` is a balanced random half split from a fixed RNG;
* ``S`` starts from ``R`` and applies a deterministic greedy pairwise swap that
  raises the *symmetric* cross-group unseen rate

      d(Q|A) = |{v in Q : k2(v) eligible, k2(v) absent from A, k1(v) present in A}|
               / |atoms(Q)|
      score  = (d(Q|A) + d(A|Q)) / 2

  where "eligible" means the radius-2 key occurs in 2..16 *different molecules of
  that batch* (batch-local support, never a valid-derived statistic).  Swaps are
  proposed from a fixed RNG and accepted only if the score rises and the declared
  coarse constraints hold (radius-1 unseen rate may rise by at most 0.01
  absolute; standardized mean differences of molecule size and ring rank may not
  exceed max(R's own difference, 0.25)).  No label, model prediction or gradient
  is ever read when building a partition.

The same partition list is used by the R and S arms of a seed, so the only
difference between those arms is the partition itself.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

CACHE = _TASK / "cache"

ELIG_LO, ELIG_HI = 2, 16
MAX_SWAPS = 256          # declared proposal budget (after the single authorised 64 -> 256 fix)
PREFIX_SWAPS = 64        # the pre-fix measurement, kept for the strength artifact
R1_TOL = 0.01
SMD_FLOOR = 0.25


# ---------------------------------------------------------------------------
# static tables
# ---------------------------------------------------------------------------


def molecule_tables(keys: dict, train: dict) -> dict:
    """Per-molecule unique key lists and structure covariates for the train split."""
    n = int(train["n_molecules"])
    off = np.asarray(keys["mol_k2_off"], dtype=np.int64)
    mk2 = np.asarray(keys["mol_k2"], dtype=np.int64)
    mk1 = np.asarray(keys["mol_k1"], dtype=np.int64)
    off1 = np.asarray(keys["mol_k1_off"], dtype=np.int64)
    u2_off = off[: n + 1].copy()
    u1_off = off1[: n + 1].copy()
    ap = np.asarray(train["atom_ptr"], dtype=np.int64)
    bp = np.asarray(train["bond_ptr"], dtype=np.int64)
    n_atoms = np.diff(ap).astype(np.int64)
    n_bonds = np.diff(bp).astype(np.int64)
    ring_rank = n_bonds - n_atoms + 1
    return {
        "n_molecules": n,
        "atom_start": ap[:n].copy(),
        "atom_count": n_atoms,
        "mol_u2_start": u2_off[:-1].copy(),
        "mol_u2_count": np.diff(u2_off),
        "u2": mk2[u2_off[0]:u2_off[n]],
        "mol_u1_start": u1_off[:-1].copy(),
        "mol_u1_count": np.diff(u1_off),
        "u1": mk1[u1_off[0]:u1_off[n]],
        "n_atoms": n_atoms,
        "ring_rank": ring_rank,
        "size_scale": float(np.std(n_atoms.astype(np.float64))) or 1.0,
        "ring_scale": float(np.std(ring_rank.astype(np.float64))) or 1.0,
        "n_key2": int(keys["n_key2"]),
        "n_key1": int(keys["n_key1"]),
        "atom_key1": np.asarray(keys["atom_key1"], dtype=np.int64),
        "atom_key2": np.asarray(keys["atom_key2"], dtype=np.int64),
    }


class BatchView:
    """Batch-local view of one molecule batch: atoms, keys, per-molecule key sets."""

    __slots__ = (
        "rows", "n", "atom_k2", "atom_k1", "atom_mol", "nloc2", "nloc1",
        "elig", "atom_elig", "pair_mol2", "pair_k2", "pair_mol1", "pair_k1",
        "mol_u2_idx", "mol_u2_ptr", "mol_u1_idx", "mol_u1_ptr",
    )

    def __init__(self, tables: dict, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.int64)
        self.rows = rows
        self.n = int(len(rows))
        counts = tables["atom_count"][rows]
        starts = tables["atom_start"][rows]
        total = int(counts.sum())
        offsets = np.cumsum(counts) - counts
        atom_idx = (np.arange(total, dtype=np.int64)
                    - np.repeat(offsets, counts)
                    + np.repeat(starts, counts))
        self.atom_mol = np.repeat(np.arange(self.n, dtype=np.int64), counts)
        k2_global = tables["atom_key2"][atom_idx]
        k1_global = tables["atom_key1"][atom_idx]
        _, inv2 = np.unique(k2_global, return_inverse=True)
        _, inv1 = np.unique(k1_global, return_inverse=True)
        self.atom_k2 = inv2.astype(np.int64)
        self.atom_k1 = inv1.astype(np.int64)
        self.nloc2 = int(inv2.max(initial=0)) + 1
        self.nloc1 = int(inv1.max(initial=0)) + 1

        # unique (molecule, key) pairs
        n2 = self.nloc2
        comb2 = np.unique(self.atom_mol * n2 + self.atom_k2)
        self.pair_mol2 = (comb2 // n2).astype(np.int64)
        self.pair_k2 = (comb2 % n2).astype(np.int64)
        n1 = self.nloc1
        comb1 = np.unique(self.atom_mol * n1 + self.atom_k1)
        self.pair_mol1 = (comb1 // n1).astype(np.int64)
        self.pair_k1 = (comb1 % n1).astype(np.int64)

        cnt2 = np.bincount(self.pair_k2, minlength=n2)
        self.elig = (cnt2 >= ELIG_LO) & (cnt2 <= ELIG_HI)
        self.atom_elig = self.elig[self.atom_k2]

        self.mol_u2_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.pair_mol2, minlength=self.n))])
        self.mol_u2_idx = self.pair_k2
        self.mol_u1_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.pair_mol1, minlength=self.n))])
        self.mol_u1_idx = self.pair_k1

    # -- side bookkeeping -------------------------------------------------

    def counts(self, q_mol: np.ndarray):
        q_pair2 = q_mol[self.pair_mol2]
        cnt_a = np.bincount(self.pair_k2[~q_pair2], minlength=self.nloc2)
        cnt_q = np.bincount(self.pair_k2[q_pair2], minlength=self.nloc2)
        q_pair1 = q_mol[self.pair_mol1]
        k1_a = np.bincount(self.pair_k1[~q_pair1], minlength=self.nloc1)
        k1_q = np.bincount(self.pair_k1[q_pair1], minlength=self.nloc1)
        return cnt_a, cnt_q, k1_a, k1_q

    def metrics(self, q_mol: np.ndarray, tabs: dict | None = None,
                counts: tuple | None = None) -> dict:
        cnt_a, cnt_q, k1_a, k1_q = self.counts(q_mol) if counts is None else counts
        qatom = q_mol[self.atom_mol]
        elig = self.elig[self.atom_k2]
        not_in_a = cnt_a[self.atom_k2] == 0
        not_in_q = cnt_q[self.atom_k2] == 0
        k1_in_a = k1_a[self.atom_k1] > 0
        k1_in_q = k1_q[self.atom_k1] > 0
        dq_num = int(np.count_nonzero(qatom & elig & not_in_a & k1_in_a))
        da_num = int(np.count_nonzero(~qatom & elig & not_in_q & k1_in_q))
        q_den = int(np.count_nonzero(qatom))
        a_den = int(np.count_nonzero(~qatom))
        q_elig = int(np.count_nonzero(qatom & elig))
        a_elig = int(np.count_nonzero(~qatom & elig))
        r1q = int(np.count_nonzero(qatom & ~k1_in_a))
        r1a = int(np.count_nonzero(~qatom & ~k1_in_q))
        full_q = int(np.count_nonzero(qatom & not_in_a))
        full_a = int(np.count_nonzero(~qatom & not_in_q))
        out = {
            "d_sym": 0.5 * (dq_num / max(q_den, 1) + da_num / max(a_den, 1)),
            "d_elig": 0.5 * (dq_num / max(q_elig, 1) + da_num / max(a_elig, 1)),
            "r1_unseen": 0.5 * (r1q / max(q_den, 1) + r1a / max(a_den, 1)),
            "full_unseen": 0.5 * (full_q / max(q_den, 1) + full_a / max(a_den, 1)),
            "elig_atom_frac": 0.5 * (q_elig / max(q_den, 1) + a_elig / max(a_den, 1)),
            "d_q_given_a": dq_num / max(q_den, 1),
            "d_a_given_q": da_num / max(a_den, 1),
            "n_query": int(np.count_nonzero(q_mol)),
            "n_source": int(self.n - np.count_nonzero(q_mol)),
        }
        if tabs is not None:
            rows = self.rows
            out["smd_atoms"] = smd(tabs["n_atoms"][rows], q_mol, tabs["size_scale"])
            out["smd_rings"] = smd(tabs["ring_rank"][rows], q_mol, tabs["ring_scale"])
        return out


def smd(values: np.ndarray, q_mol: np.ndarray, scale: float) -> float:
    a = np.asarray(values, dtype=np.float64)[~q_mol]
    q = np.asarray(values, dtype=np.float64)[q_mol]
    if len(a) == 0 or len(q) == 0:
        return 0.0
    return float(abs(a.mean() - q.mean()) / max(scale, 1e-9))


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------


def random_partition(n: int, rng: np.random.Generator) -> np.ndarray:
    q = np.zeros(n, dtype=bool)
    perm = rng.permutation(n)
    q[perm[: n // 2]] = True
    return q


def _fast_scores(view: BatchView, q_mol: np.ndarray, cnt_a: np.ndarray, cnt_q: np.ndarray,
                 k1_a: np.ndarray, k1_q: np.ndarray) -> tuple[float, float, int, int, int, int]:
    """(d_sym, r1_unseen, dq_num, da_num, q_den, a_den) -- search-path scorer."""
    qatom = q_mol[view.atom_mol]
    ak2 = view.atom_k2
    ak1 = view.atom_k1
    k1_in_a = k1_a[ak1] > 0
    k1_in_q = k1_q[ak1] > 0
    dq_num = int(np.count_nonzero(qatom & view.atom_elig & (cnt_a[ak2] == 0) & k1_in_a))
    da_num = int(np.count_nonzero(~qatom & view.atom_elig & (cnt_q[ak2] == 0) & k1_in_q))
    q_den = int(np.count_nonzero(qatom))
    a_den = int(np.count_nonzero(~qatom))
    r1q = int(np.count_nonzero(qatom & ~k1_in_a))
    r1a = int(np.count_nonzero(~qatom & ~k1_in_q))
    d_sym = 0.5 * (dq_num / max(q_den, 1) + da_num / max(a_den, 1))
    r1 = 0.5 * (r1q / max(q_den, 1) + r1a / max(a_den, 1))
    return d_sym, r1, dq_num, da_num, q_den, a_den


def improve(view: BatchView, q_mol: np.ndarray, rng: np.random.Generator,
            tabs: dict, base: dict, max_swaps: int = MAX_SWAPS) -> tuple[np.ndarray, list]:
    """Greedy pairwise swaps from ``q_mol`` that raise the symmetric cross-group rate.

    A *proposal* is one evaluated candidate swap ``(i in source, j in query)``;
    at most ``max_swaps`` proposals are evaluated per batch.  Proposals are drawn
    from a fixed RNG stream of molecule pairs, so the construction is
    deterministic given ``(seed, epoch, batch)`` and reads no label, prediction
    or gradient.  A swap is accepted only if the score strictly rises and the
    coarse constraints still hold.
    """
    q = q_mol.copy()
    cnt_a, cnt_q, k1_a, k1_q = view.counts(q)
    cur_d, cur_r1, *_ = _fast_scores(view, q, cnt_a, cnt_q, k1_a, k1_q)
    values = np.asarray(tabs["n_atoms"][view.rows], dtype=np.float64)
    rings = np.asarray(tabs["ring_rank"][view.rows], dtype=np.float64)
    size_scale = tabs["size_scale"]
    ring_scale = tabs["ring_scale"]
    smd_atoms_cap = max(base["smd_atoms"], SMD_FLOOR)
    smd_rings_cap = max(base["smd_rings"], SMD_FLOOR)
    r1_cap = base["r1_unseen"] + R1_TOL
    accepted: list[tuple[int, int]] = []
    n = view.n
    u2_ptr, u2_idx = view.mol_u2_ptr, view.mol_u2_idx
    u1_ptr, u1_idx = view.mol_u1_ptr, view.mol_u1_idx
    stream = rng.integers(0, n, size=(max_swaps * 8, 2))
    evaluated = 0
    for i, j in stream:
        if evaluated >= max_swaps:
            break
        i, j = int(i), int(j)
        if q[i] or not q[j] or i == j:
            continue
        evaluated += 1
        for m, d_a, d_q in ((i, -1, 1), (j, 1, -1)):
            s2 = slice(int(u2_ptr[m]), int(u2_ptr[m + 1]))
            s1 = slice(int(u1_ptr[m]), int(u1_ptr[m + 1]))
            cnt_a[u2_idx[s2]] += d_a
            cnt_q[u2_idx[s2]] += d_q
            k1_a[u1_idx[s1]] += d_a
            k1_q[u1_idx[s1]] += d_q
        q[i] = True
        q[j] = False
        d_sym, r1, *_ = _fast_scores(view, q, cnt_a, cnt_q, k1_a, k1_q)
        ok = d_sym > cur_d + 1e-9 and r1 <= r1_cap
        if ok:
            ok = (smd(values, q, size_scale) <= smd_atoms_cap
                  and smd(rings, q, ring_scale) <= smd_rings_cap)
        if ok:
            cur_d = d_sym
            accepted.append((i, j))
        else:
            for m, d_a, d_q in ((i, 1, -1), (j, -1, 1)):
                s2 = slice(int(u2_ptr[m]), int(u2_ptr[m + 1]))
                s1 = slice(int(u1_ptr[m]), int(u1_ptr[m + 1]))
                cnt_a[u2_idx[s2]] += d_a
                cnt_q[u2_idx[s2]] += d_q
                k1_a[u1_idx[s1]] += d_a
                k1_q[u1_idx[s1]] += d_q
            q[i] = False
            q[j] = True
    return q, accepted


def partitions_for_epoch(tables: dict, order: np.ndarray, epoch: int, seed: int,
                         batch_size: int = 128, max_swaps: int = MAX_SWAPS,
                         collect_metrics: bool = False) -> dict:
    """Build R and S partitions for every batch of one epoch (deterministic)."""
    batches = []
    for b, start in enumerate(range(0, len(order), batch_size)):
        rows = np.asarray(order[start:start + batch_size], dtype=np.int64)
        view = BatchView(tables, rows)
        rng = np.random.default_rng([int(seed), int(epoch), int(b)])
        r = random_partition(view.n, rng)
        base = view.metrics(r, tables)
        s, swaps = improve(view, r, rng, tables, base, max_swaps=max_swaps)
        entry = {"rows": rows, "r": r, "s": s, "swaps": swaps}
        if collect_metrics:
            fin = view.metrics(s, tables)
            entry["metrics"] = {"R": base, "S": fin, "n_swaps": len(swaps)}
        batches.append(entry)
    return {"epoch": epoch, "order": order, "batches": batches}


def order_hash(order: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(order), dtype=np.int64).tobytes()).hexdigest()


def build_all(seed: int, epochs: int = 600, batch_size: int = 128,
              max_swaps: int = MAX_SWAPS, keys: dict | None = None,
              train: dict | None = None, collect_metrics: bool = True) -> dict:
    """Precompute every partition of one training seed and store compact masks."""
    import torch

    import ad_data as D
    import zct_keys as K

    keys = K.load() if keys is None else keys
    train = D.load_cache("train") if train is None else train
    tables = molecule_tables(keys, train)
    n = int(train["n_molecules"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    swaps_a, swaps_b, swap_off = [], [], [0]
    rec = {k: [] for k in ("d_sym", "d_elig", "r1_unseen", "full_unseen",
                           "elig_atom_frac", "smd_atoms", "smd_rings", "n_swaps",
                           "d_sym_R", "d_elig_R", "r1_unseen_R", "smd_atoms_R", "smd_rings_R",
                           "full_unseen_R")}
    epoch_hashes = []
    batch_rows = []
    r_pack, s_pack = [], []
    n_swaps_total = 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=gen).numpy()
        epoch_hashes.append(order_hash(order))
        res = partitions_for_epoch(tables, order, epoch, seed, batch_size, max_swaps, collect_metrics)
        for entry in res["batches"]:
            nb = len(entry["rows"])
            swaps_a.extend([int(i) for i, _ in entry["swaps"]])
            swaps_b.extend([int(j) for _, j in entry["swaps"]])
            swap_off.append(len(swaps_a))
            batch_rows.append(nb)
            n_swaps_total += len(entry["swaps"])
            # the stored S mask must be exactly R plus the recorded swaps
            check = entry["r"].copy()
            for i, j in entry["swaps"]:
                check[int(i)] = True
                check[int(j)] = False
            if not np.array_equal(check, entry["s"]):
                raise RuntimeError(f"seed {seed} epoch {epoch}: S mask != R + swaps")
            pad = batch_size - nb
            r_pack.append(np.packbits(np.concatenate([entry["r"], np.zeros(pad, bool)])))
            s_pack.append(np.packbits(np.concatenate([entry["s"], np.zeros(pad, bool)])))
            if collect_metrics:
                m = entry["metrics"]
                for key in ("d_sym", "d_elig", "r1_unseen", "full_unseen", "elig_atom_frac"):
                    rec[key].append(float(m["S"][key]))
                rec["smd_atoms"].append(float(m["S"]["smd_atoms"]))
                rec["smd_rings"].append(float(m["S"]["smd_rings"]))
                rec["n_swaps"].append(int(m["n_swaps"]))
                for key in ("d_sym", "d_elig", "r1_unseen", "smd_atoms", "smd_rings", "full_unseen"):
                    rec[key + "_R"].append(float(m["R"][key]))
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"partitions_s{int(seed)}.npz"
    np.savez_compressed(
        path,
        swap_off=np.asarray(swap_off, dtype=np.int64),
        swap_i=np.asarray(swaps_a, dtype=np.int16),
        swap_j=np.asarray(swaps_b, dtype=np.int16),
        batch_rows=np.asarray(batch_rows, dtype=np.int16),
        r_packed=np.stack(r_pack).astype(np.uint8),
        s_packed=np.stack(s_pack).astype(np.uint8),
        **{k: np.asarray(v, dtype=np.float64) for k, v in rec.items()},
    )
    summary = {
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "max_proposals_per_batch": int(max_swaps),
        "n_batches": int(len(batch_rows)),
        "steps_per_epoch": int(math.ceil(n / batch_size)),
        "total_proposal_budget": int(len(batch_rows) * max_swaps),
        "mean_accepted_swaps": float(np.mean(rec["n_swaps"])),
        "mean_d_sym_R": float(np.mean(rec["d_sym_R"])),
        "mean_d_sym_S": float(np.mean(rec["d_sym"])),
        "delta_d_sym": float(np.mean(rec["d_sym"]) - np.mean(rec["d_sym_R"])),
        "mean_d_elig_R": float(np.mean(rec["d_elig_R"])),
        "mean_d_elig_S": float(np.mean(rec["d_elig"])),
        "delta_d_elig": float(np.mean(rec["d_elig"]) - np.mean(rec["d_elig_R"])),
        "mean_r1_unseen_R": float(np.mean(rec["r1_unseen_R"])),
        "mean_r1_unseen_S": float(np.mean(rec["r1_unseen"])),
        "mean_full_unseen_R": float(np.mean(rec["full_unseen_R"])),
        "mean_full_unseen_S": float(np.mean(rec["full_unseen"])),
        "mean_elig_atom_frac_S": float(np.mean(rec["elig_atom_frac"])),
        "mean_smd_atoms_R": float(np.mean(rec["smd_atoms_R"])),
        "mean_smd_atoms_S": float(np.mean(rec["smd_atoms"])),
        "mean_smd_rings_R": float(np.mean(rec["smd_rings_R"])),
        "mean_smd_rings_S": float(np.mean(rec["smd_rings"])),
        "frac_batches_S_gt_R": float(np.mean(np.asarray(rec["d_sym"]) > np.asarray(rec["d_sym_R"]))),
        "epoch_order_sha256": epoch_hashes,
        "npz": str(path),
        "npz_sha256": C.sha256_file(path),
        "constraints": {
            "elig_lo": ELIG_LO, "elig_hi": ELIG_HI, "r1_tol": R1_TOL, "smd_floor": SMD_FLOOR,
            "checked_per_batch": "S.r1_unseen <= R.r1_unseen + 0.01 and S.smd <= max(R.smd, 0.25)",
        },
        "n_swaps_total": int(n_swaps_total),
    }
    C.write_json(CACHE / f"partitions_s{int(seed)}.json", summary)
    return summary


def load_partitions(seed: int) -> dict:
    path = CACHE / f"partitions_s{int(seed)}.npz"
    d = dict(np.load(path))
    d["summary"] = C.read_json(CACHE / f"partitions_s{int(seed)}.json")
    return d


def mask_for(part: dict, epoch: int, batch: int, arm: str, n: int,
             steps_per_epoch: int | None = None) -> np.ndarray:
    """Exact frozen mask for (epoch, batch); ``arm`` is "R" or "S"."""
    key = "s_packed" if arm == "S" else "r_packed"
    steps = int(steps_per_epoch if steps_per_epoch is not None
                else part["summary"]["steps_per_epoch"])
    epochs = len(part[key]) // steps
    idx = (int(epoch) - 1) * steps + int(batch)
    if idx >= len(part[key]):
        raise IndexError(f"partition index {idx} out of range ({epochs} epochs, {steps} batches)")
    packed = np.asarray(part[key][idx], dtype=np.uint8)
    return np.unpackbits(packed)[:n].astype(bool)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--max-swaps", type=int, default=MAX_SWAPS)
    p.add_argument("--calibrate", action="store_true",
                   help="only the first 256 batches, no npz (strength gate)")
    args = p.parse_args(argv)
    out = {}
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        if args.calibrate:
            out[str(seed)] = calibrate(seed, n_batches=256, max_swaps=args.max_swaps)
        else:
            out[str(seed)] = build_all(seed, epochs=args.epochs, max_swaps=args.max_swaps)
        C.log(out[str(seed)])
    return 0


def calibrate(seed: int, n_batches: int = 256, max_swaps: int = MAX_SWAPS) -> dict:
    """Intervention-strength check over the first ``n_batches`` batches of seed 0."""
    import torch

    import ad_data as D
    import zct_keys as K

    keys = K.load()
    train = D.load_cache("train")
    tables = molecule_tables(keys, train)
    n = int(train["n_molecules"])
    gen = torch.Generator().manual_seed(int(seed) + 101)
    rec = {k: [] for k in ("d_sym_R", "d_sym_S", "d_elig_R", "d_elig_S",
                           "r1_R", "r1_S", "full_R", "full_S", "smd_atoms_R", "smd_atoms_S",
                           "smd_rings_R", "smd_rings_S", "n_swaps", "elig_frac")}
    seen = 0
    epoch = 0
    while seen < n_batches:
        epoch += 1
        order = torch.randperm(n, generator=gen).numpy()
        res = partitions_for_epoch(tables, order, epoch, seed, 128, max_swaps, True)
        for entry in res["batches"]:
            if seen >= n_batches:
                break
            m = entry["metrics"]
            rec["d_sym_R"].append(m["R"]["d_sym"])
            rec["d_sym_S"].append(m["S"]["d_sym"])
            rec["d_elig_R"].append(m["R"]["d_elig"])
            rec["d_elig_S"].append(m["S"]["d_elig"])
            rec["r1_R"].append(m["R"]["r1_unseen"])
            rec["r1_S"].append(m["S"]["r1_unseen"])
            rec["full_R"].append(m["R"]["full_unseen"])
            rec["full_S"].append(m["S"]["full_unseen"])
            rec["smd_atoms_R"].append(m["R"]["smd_atoms"])
            rec["smd_atoms_S"].append(m["S"]["smd_atoms"])
            rec["smd_rings_R"].append(m["R"]["smd_rings"])
            rec["smd_rings_S"].append(m["S"]["smd_rings"])
            rec["n_swaps"].append(m["n_swaps"])
            rec["elig_frac"].append(m["S"]["elig_atom_frac"])
            seen += 1
    def mean(key):
        return float(np.mean(rec[key]))
    summary = {
        "seed": int(seed),
        "n_batches": int(seen),
        "max_swaps": int(max_swaps),
        "mean_n_swaps": mean("n_swaps"),
        "mean_d_sym_R": mean("d_sym_R"),
        "mean_d_sym_S": mean("d_sym_S"),
        "delta_d_sym": mean("d_sym_S") - mean("d_sym_R"),
        "mean_d_elig_R": mean("d_elig_R"),
        "mean_d_elig_S": mean("d_elig_S"),
        "delta_d_elig": mean("d_elig_S") - mean("d_elig_R"),
        "mean_r1_R": mean("r1_R"),
        "mean_r1_S": mean("r1_S"),
        "mean_smd_atoms_R": mean("smd_atoms_R"),
        "mean_smd_atoms_S": mean("smd_atoms_S"),
        "mean_smd_rings_R": mean("smd_rings_R"),
        "mean_smd_rings_S": mean("smd_rings_S"),
        "mean_full_unseen_R": mean("full_R"),
        "mean_full_unseen_S": mean("full_S"),
        "frac_batches_S_gt_R": float(np.mean(np.asarray(rec["d_sym_S"]) > np.asarray(rec["d_sym_R"]))),
        "passes": {
            "delta_d_sym_ge_0.02": bool(mean("d_sym_S") - mean("d_sym_R") >= 0.02),
            "delta_d_elig_ge_0.10": bool(mean("d_elig_S") - mean("d_elig_R") >= 0.10),
            "r1_increase_le_0.01": bool(mean("r1_S") - mean("r1_R") <= R1_TOL + 1e-12),
        },
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
