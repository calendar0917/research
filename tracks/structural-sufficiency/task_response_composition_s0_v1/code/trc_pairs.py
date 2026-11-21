"""Real replacement directory: support, sampling distribution and ``tau2``.

The directory is built from *real* observed molecules under an identical exact
context key.  Only pairs whose two endpoints both belong to the current training
set are used; nothing here reads validation or test labels.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import trc_common as C


def subset_rows(row_ids: np.ndarray, train_rows: np.ndarray, fraction: float) -> np.ndarray:
    """Nested fixed-hash subset of the official training split."""
    order = sorted(train_rows.tolist(),
                   key=lambda r: hashlib.sha256(f"trc-s0:{int(row_ids[r])}".encode()).hexdigest())
    take = int(np.floor(len(order) * float(fraction)))
    return np.asarray(sorted(order[:take]), np.int64)


def _components(n_mol: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parent = list(range(int(n_mol)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(int(x)), find(int(y))
        if rx != ry:
            parent[rx] = ry
    return np.asarray([find(i) for i in range(int(n_mol))], np.int32)


class PairStore:
    """Filtered train-only pair directory with a fixed sampling distribution."""

    def __init__(self, cfg: Mapping[str, Any], pack: Mapping[str, Any],
                 labels: np.ndarray, _unused: Any, train_rows: np.ndarray,
                 row_ids: np.ndarray, log=print) -> None:
        self.cfg = cfg
        self.n_mol = int(pack["n_molecules"])
        self.labels = np.asarray(labels, np.float64)
        self.sigma = float(np.std(self.labels[np.asarray(train_rows, np.int64)]))
        train_set = np.zeros(self.n_mol, bool)
        train_set[np.asarray(train_rows, np.int64)] = True
        pa = np.asarray(pack["pair_mol_a"], np.int64)
        pb = np.asarray(pack["pair_mol_b"], np.int64)
        keep = train_set[pa] & train_set[pb]
        self.index = np.nonzero(keep)[0]                     # original pair ids
        self.a = pa[keep]
        self.b = pb[keep]
        self.C = np.asarray(pack["pair_C"], np.int64)[keep]
        self.Sa = np.asarray(pack["pair_Sa"], np.int64)[keep]
        self.Sb = np.asarray(pack["pair_Sb"], np.int64)[keep]
        self.bond = np.asarray(pack["pair_cut_bond_row"], np.int64)[keep]
        self.transform = np.asarray(pack["pair_transform"], np.int64)[keep]
        self.delta = self.labels[self.b] - self.labels[self.a]
        self.delta_std = self.delta / self.sigma if self.sigma > 0 else self.delta
        comp = _components(self.n_mol, self.a, self.b)
        self.component = comp[self.a]
        self.weights = {
            "repeat_pool_weight": float(cfg["pair_directory"]["repeat_pool_weight"]),
            "all_pool_weight": float(cfg["pair_directory"]["all_pool_weight"]),
        }
        self._build_pools()
        self.tau2 = self._compute_tau2()
        self._rng = np.random.default_rng(int(cfg["pair_directory"]["sampler_seed"]))

    # -- pools --------------------------------------------------------------
    def _build_pools(self) -> None:
        by_transform: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, (t, c) in enumerate(zip(self.transform.tolist(), self.component.tolist())):
            by_transform[int(t)][int(c)].append(i)
        self.transform_groups = {t: {c: np.asarray(v, np.int64) for c, v in g.items()}
                                 for t, g in by_transform.items()}
        repeat = sorted(t for t, g in self.transform_groups.items() if len(g) >= 2)
        allt = sorted(self.transform_groups)
        self.pools: list[tuple[float, list[int]]] = []
        self.repeat_transforms = repeat
        self.all_transforms = allt
        w_all = float(self.cfg["pair_directory"]["all_pool_weight"])
        if repeat:
            self.pools = [(1.0 - w_all, repeat), (w_all, allt)]
        else:
            self.pools = [(1.0, allt)]

    def npairs(self) -> int:
        return int(self.a.size)

    def support(self) -> dict[str, Any]:
        counts = np.bincount(np.asarray([len(g) for g in self.transform_groups.values()], np.int64),
                             minlength=8) if self.transform_groups else np.zeros(8, np.int64)
        return {
            "n_train_pairs": self.npairs(),
            "n_train_transforms": len(self.all_transforms),
            "n_repeat_transforms": len(self.repeat_transforms),
            "n_train_pair_components": len(set(self.component.tolist())),
            "n_molecules_in_pairs": int(len(set(self.a.tolist()) | set(self.b.tolist()))),
            "transform_component_count_hist": counts.tolist(),
            "pools": {name: len(pool) for name, (w, pool) in zip(("repeat", "all"), self.pools)},
            "sigma_y": self.sigma,
            "tau2": self.tau2,
            "delta_y_abs_mean": float(np.mean(np.abs(self.delta))) if self.a.size else 0.0,
            "delta_y_std": float(np.std(self.delta)) if self.a.size else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        return self.support()

    def tensor_pairs(self, idx: np.ndarray, device) -> dict:
        import torch

        sel = np.asarray(idx, np.int64)
        return {
            "C": torch.as_tensor(self.C[sel], device=device, dtype=torch.long),
            "Sa": torch.as_tensor(self.Sa[sel], device=device, dtype=torch.long),
            "Sb": torch.as_tensor(self.Sb[sel], device=device, dtype=torch.long),
            "bond": torch.as_tensor(self.bond[sel], device=device, dtype=torch.long),
        }

    def delta_std_of(self, idx: np.ndarray) -> np.ndarray:
        return self.delta_std[np.asarray(idx, np.int64)]

    def labels_of_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.delta, self.delta_std

    # -- tau2 ---------------------------------------------------------------
    def _compute_tau2(self) -> float:
        total = 0.0
        for weight, pool in self.pools:
            acc = 0.0
            for t in pool:
                groups = self.transform_groups[t]
                ncomp = len(groups)
                inner = 0.0
                for c, idxs in groups.items():
                    inner += float(np.mean(self.delta_std[idxs] ** 2))
                acc += inner / ncomp
            if pool:
                acc /= len(pool)
            total += weight * acc
        return float(max(float(self.cfg["recipe"]["aux_tau2_floor"]), total))

    # -- sampling -----------------------------------------------------------
    def sampler(self) -> "PairSampler":
        return PairSampler(self)


class PairSampler:
    def __init__(self, store: PairStore) -> None:
        self.store = store
        self.rng = np.random.default_rng(int(store.cfg["pair_directory"]["sampler_seed"]))

    def sample(self, k: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        st = self.store
        if st.npairs() == 0:
            raise RuntimeError("empty train pair directory")
        picks = np.empty(int(k), np.int64)
        weights = np.asarray([w for w, _p in st.pools], np.float64)
        weights = weights / weights.sum()
        pool_choice = self.rng.choice(len(st.pools), size=int(k), p=weights)
        for i in range(int(k)):
            _w, pool = st.pools[int(pool_choice[i])]
            t = pool[int(self.rng.integers(len(pool)))]
            groups = st.transform_groups[t]
            keys = sorted(groups)
            c = keys[int(self.rng.integers(len(keys)))]
            idxs = groups[c]
            picks[i] = idxs[int(self.rng.integers(idxs.size))]
        return picks, (st.a[picks], st.b[picks])

    def probs(self) -> np.ndarray:
        st = self.store
        p = np.zeros(st.npairs(), np.float64)
        weights = np.asarray([w for w, _pp in st.pools], np.float64)
        weights = weights / weights.sum()
        for (w, pool), pw in zip(st.pools, weights):
            if not pool:
                continue
            per_t = pw / len(pool)
            for t in pool:
                groups = st.transform_groups[t]
                for c, idxs in groups.items():
                    p[idxs] += per_t / len(groups) / idxs.size
        return p
