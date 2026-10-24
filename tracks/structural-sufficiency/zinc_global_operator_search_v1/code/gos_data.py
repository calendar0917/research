"""Bond-object relation cache for zinc-global-operator-search-v1.

One undirected physical bond is one global object.  The static relation between
two bonds ``b={u,v}`` and ``c={p,q}`` is the symmetric lift of the frozen
DBT atom-pair simple-path counts and shortest distances:

    lift_l(b,c) = (S_l(u,p) + S_l(u,q) + S_l(v,p) + S_l(v,q)) / 2

which is exactly ``M S_l M^T`` with ``M[b, :] = 1/sqrt(2)`` on the two
endpoints of ``b``.  It is a *structural association statistic* between two
bond objects, not the number of bond-to-bond simple paths of the original graph.

Everything here is a deterministic, label-free function of the raw graph
(through the frozen P3/DBT caches).  Scaling constants are fitted on train only.
Official test is never touched by this module.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
EXP = _HERE.parent
SSUF = EXP.parent
REPO = SSUF.parents[1]
_V1 = SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = SSUF / "zinc_directed_bond_transformer_v1"
for _p in (str(_HERE), str(_V1), str(_DBT / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402

CACHE = EXP / "cache"
ANALYSIS = EXP / "analysis"

DBT_PATH = {s: _DBT / "cache" / f"dbt-path-v1-L12_{s}.pt" for s in ("train", "valid")}
DBT_OBJ = {s: _DBT / "cache" / f"dbt-objects-v1_{s}.pt" for s in ("train", "valid")}

REL_VERSION = "gos-bondrel-v1"
L_MAX = 12
DIST_EXACT_MAX = 12
DIST_LONG = 13
DIST_UNREACH = 14

N_PATH = L_MAX            # 12 log1p lifted path statistics
REL_DIM = N_PATH + 4      # + min endpoint distance, shared endpoints, self, unreachable
SELF_COL = N_PATH + 2
UNREACH_COL = N_PATH + 3


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _mol_lift(S: np.ndarray, Dm: np.ndarray, u: np.ndarray, v: np.ndarray):
    """S [n,n,L], Dm [n,n], bond endpoints u,v [B] -> lifted tables."""
    Su, Sv = S[u], S[v]
    lift_sum = Su[:, u] + Su[:, v] + Sv[:, u] + Sv[:, v]          # [B,B,L]
    Du, Dv = Dm[u], Dm[v]
    lift_dist = np.minimum(np.minimum(Du[:, u], Du[:, v]),
                           np.minimum(Dv[:, u], Dv[:, v]))         # [B,B]
    shared = ((u[:, None] == u[None, :]).astype(np.uint8)
              + (u[:, None] == v[None, :]).astype(np.uint8)
              + (v[:, None] == u[None, :]).astype(np.uint8)
              + (v[:, None] == v[None, :]).astype(np.uint8))
    return lift_sum, lift_dist, shared


def _resolve_split(split: str) -> dict[str, Any]:
    """Atom-pair tables + bond tables for a split (rebuilds if the DBT cache is absent)."""
    src = {"path_source": None, "path_sha256": None, "objects_sha256": None}
    if split in DBT_PATH and DBT_PATH[split].exists() and DBT_OBJ[split].exists():
        path_d = torch.load(DBT_PATH[split], map_location="cpu", weights_only=False)
        obj_d = torch.load(DBT_OBJ[split], map_location="cpu", weights_only=False)
        src["path_source"] = str(DBT_PATH[split].relative_to(REPO))
        src["path_sha256"] = C.sha256_file(DBT_PATH[split])
        src["objects_sha256"] = C.sha256_file(DBT_OBJ[split])
    else:
        import dbt_data as DB  # only used when a split cache must be rebuilt

        pth = DB.cache_path(split)
        if pth.exists():
            path_d = torch.load(pth, map_location="cpu", weights_only=False)
        else:
            path_d = DB.build_relation_split(split)
            DB.save_relation_split(path_d)
        md = D.load_cache(split)
        obj_d = {"bond_ptr": md["bond_ptr"], "bond_index": md["bond_index"]}
        src["path_source"] = f"rebuilt:{pth.relative_to(REPO)}"
    n_mol = int(path_d["n_molecules"])
    if len(np.asarray(path_d["pair_ptr"])) != n_mol + 1:
        raise RuntimeError("pair pointer / molecule count mismatch")
    return {
        "pair_ptr": path_d["pair_ptr"], "path": path_d["path"], "dist": path_d["dist"],
        "bond_ptr": np.asarray(obj_d["bond_ptr"]),
        "bond_index": np.asarray(obj_d["bond_index"]).astype(np.int64),
        "n_mol": n_mol, "src": src,
    }


def build_rel(split: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = _resolve_split(split)
    pair_ptr = np.asarray(r["pair_ptr"])
    path = np.asarray(r["path"])
    dist = np.asarray(r["dist"])
    bond_ptr = r["bond_ptr"]
    bond_index = r["bond_index"]
    n_mol = int(r["n_mol"])

    sums, dsts, shs = [], [], []
    nb = np.zeros(n_mol, dtype=np.int32)
    ptr = np.zeros(n_mol + 1, dtype=np.int64)
    for i in range(n_mol):
        s, e = int(pair_ptr[i]), int(pair_ptr[i + 1])
        n = int(round(float(np.sqrt(e - s))))
        if n * n != e - s:
            raise RuntimeError(f"{split}[{i}]: pair table is not square")
        S = path[s * L_MAX:e * L_MAX].reshape(n, n, L_MAX)
        Dm = dist[s:e].reshape(n, n)
        bs, be = int(bond_ptr[i]), int(bond_ptr[i + 1])
        bi = bond_index[bs:be]
        B = bi.shape[0]
        nb[i] = B
        if B == 0:
            ptr[i + 1] = ptr[i]
            continue
        ls, ld, sh = _mol_lift(S, Dm, bi[:, 0], bi[:, 1])
        if int(ls.max()) > 255:
            raise RuntimeError("lifted path count exceeds uint8")
        sums.append(ls.astype(np.uint8).reshape(-1))
        dsts.append(ld.astype(np.uint8).reshape(-1))
        shs.append(sh.reshape(-1))
        ptr[i + 1] = ptr[i] + B * B

    lift_sum = np.concatenate(sums) if sums else np.zeros(0, np.uint8)
    lift_dist = np.concatenate(dsts) if dsts else np.zeros(0, np.uint8)
    lift_shared = np.concatenate(shs) if shs else np.zeros(0, np.uint8)
    out = {
        "split": split,
        "n_molecules": n_mol,
        "n_bonds_graph": nb,
        "ptr": ptr,
        "lift_sum": lift_sum,
        "lift_dist": lift_dist,
        "lift_shared": lift_shared,
        "meta": {
            "version": REL_VERSION,
            "l_max": L_MAX,
            "path_source": r["src"]["path_source"],
            "path_sha256": r["src"]["path_sha256"],
            "objects_sha256": r["src"]["objects_sha256"],
            "definition": "lift_l(b,c) = (S_l(u,p)+S_l(u,q)+S_l(v,p)+S_l(v,q))/2",
            "build_seconds": None,
            "labels_used": False,
        },
    }
    out["meta"]["build_seconds"] = round(time.perf_counter() - t0, 3)
    return out


def rel_cache_path(split: str) -> Path:
    return CACHE / f"{REL_VERSION}_{split}.pt"


def save_rel(d: Mapping[str, Any]) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = rel_cache_path(d["split"])
    C.atomic_torch_save(dict(d), p)
    return p


def load_rel(split: str, *, rebuild: bool = False) -> dict:
    p = rel_cache_path(split)
    if p.exists() and not rebuild:
        return torch.load(p, map_location="cpu", weights_only=False)
    d = build_rel(split)
    save_rel(d)
    return d


# ---------------------------------------------------------------------------
# train-only scaling
# ---------------------------------------------------------------------------


def fit_scale(train_rel: Mapping[str, Any]) -> dict[str, Any]:
    ptr = np.asarray(train_rel["ptr"])
    ls = np.asarray(train_rel["lift_sum"])
    ld = np.asarray(train_rel["lift_dist"]).astype(np.float64)
    sh = np.asarray(train_rel["lift_shared"]).astype(np.float64)
    n_path, s1, s2 = 0, np.zeros(L_MAX), np.zeros(L_MAX)
    d1 = d2 = 0.0
    q1 = q2 = 0.0
    counts = np.zeros(L_MAX, dtype=np.int64)
    for i in range(int(train_rel["n_molecules"])):
        s, e = int(ptr[i]), int(ptr[i + 1])
        B = int(np.sqrt(e - s))
        if B == 0:
            continue
        block = ls[s * L_MAX:e * L_MAX].reshape(B, B, L_MAX).astype(np.float64) / 2.0
        f = np.log1p(block).reshape(-1, L_MAX)
        s1 += f.sum(0)
        s2 += (f * f).sum(0)
        counts += f.shape[0]
        n_path += f.shape[0]
        dd = ld[s:e].astype(np.float64)
        d1 += dd.sum()
        d2 += (dd * dd).sum()
        ss = sh[s:e].astype(np.float64) / 2.0
        q1 += ss.sum()
        q2 += (ss * ss).sum()
    mean_path = s1 / n_path
    var_path = np.maximum(s2 / n_path - mean_path ** 2, 1e-8)
    mean_d = d1 / n_path
    var_d = max(d2 / n_path - mean_d ** 2, 1e-8)
    mean_s = q1 / n_path
    var_s = max(q2 / n_path - mean_s ** 2, 1e-8)
    return {
        "version": REL_VERSION,
        "source": "train split only",
        "n_pairs": int(n_path),
        "path_mean": mean_path.tolist(),
        "path_std": np.sqrt(var_path).tolist(),
        "dist_mean": float(mean_d),
        "dist_std": float(np.sqrt(var_d)),
        "shared_mean": float(mean_s),
        "shared_std": float(np.sqrt(var_s)),
    }


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------


def _rel_tensors(rel: Mapping[str, Any], rows: Sequence[int],
                 scale: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    nb = np.asarray(rel["n_bonds_graph"])[rows].astype(np.int64)
    bb = int(nb.max()) if B else 0
    feat = np.zeros((B, bb, bb, REL_DIM), dtype=np.float32)
    ptr = np.asarray(rel["ptr"])
    ls = np.asarray(rel["lift_sum"])
    ld = np.asarray(rel["lift_dist"])
    sh = np.asarray(rel["lift_shared"])
    pm = np.asarray(scale["path_mean"], dtype=np.float32)
    ps = np.asarray(scale["path_std"], dtype=np.float32)
    dm, ds = float(scale["dist_mean"]), float(scale["dist_std"])
    sm, ss = float(scale["shared_mean"]), float(scale["shared_std"])
    for i, row in enumerate(rows):
        n = int(nb[i])
        if n == 0:
            continue
        s, e = int(ptr[row]), int(ptr[row + 1])
        p = ls[s * L_MAX:e * L_MAX].reshape(n, n, L_MAX).astype(np.float32) / 2.0
        dd = ld[s:e].reshape(n, n).astype(np.float32)
        shd = sh[s:e].reshape(n, n).astype(np.float32)
        f = feat[i, :n, :n]
        f[..., :L_MAX] = (np.log1p(p) - pm) / ps
        f[..., L_MAX] = (dd - dm) / ds
        f[..., L_MAX + 1] = (shd / 2.0 - sm) / ss
        f[..., SELF_COL] = (shd == 2).astype(np.float32)
        f[..., UNREACH_COL] = (dd >= DIST_UNREACH).astype(np.float32)
    mask = torch.arange(bb).view(1, bb) < torch.from_numpy(nb).view(B, 1)
    return {
        "rel_feat": torch.from_numpy(feat),
        "rel_mask": mask,
        "n_bonds_pad": int(bb),
    }


def make_collate(rel_train: Mapping[str, Any], rel_valid: Mapping[str, Any],
                 scale: Mapping[str, Any], rel_test: Mapping[str, Any] | None = None):
    """P3N batch + bond-object relation tensors.

    The split is read from the ad cache ``meta.split`` (authoritative); the
    label-count fallback exists only for older caches without the meta tag.
    """
    cache_rel = {"train": rel_train, "valid": rel_valid}
    if rel_test is not None:
        cache_rel["test"] = rel_test
    by_count = {int(rel_train["n_molecules"]): rel_train, int(rel_valid["n_molecules"]): rel_valid}
    if rel_test is not None:
        by_count.setdefault(int(rel_test["n_molecules"]), rel_test)

    def collate(cache, rows, candidate, *, valid: bool = False, walk_cols=None):
        split = str(cache.get("meta", {}).get("split", "")) if isinstance(cache.get("meta"), Mapping) else ""
        if split in cache_rel:
            rel = cache_rel[split]
        else:
            n_mol = int(cache["labels"].shape[0])
            if n_mol not in by_count:
                raise KeyError(f"no relation cache for split {split!r} with {n_mol} molecules")
            rel = by_count[n_mol]
        batch = T1.collate(cache, rows, "P3N", valid=valid, walk_cols=walk_cols)
        batch.update(_rel_tensors(rel, rows, scale))
        bond_ptr = np.asarray(cache["bond_ptr"])
        rows_arr = np.asarray(rows, dtype=np.int64)
        counts = (bond_ptr[rows_arr + 1] - bond_ptr[rows_arr]).astype(np.int64)
        local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts]) if counts.sum() else np.zeros(0, np.int64)
        batch["bond_local"] = torch.from_numpy(local)
        return batch

    return collate


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "scale", "audit"])
    p.add_argument("--split", default=None)
    args = p.parse_args(argv)
    if args.command in ("build", "audit"):
        splits = [args.split] if args.split else ["train", "valid"]
        for s in splits:
            d = load_rel(s, rebuild=args.command == "build")
            print(s, "molecules", d["n_molecules"], "pairs", int(d["ptr"][-1]),
                  "bonds", int(np.asarray(d["n_bonds_graph"]).sum()),
                  "seconds", d["meta"]["build_seconds"])
    if args.command == "scale":
        scale = fit_scale(load_rel("train"))
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        C.write_json(ANALYSIS / "rel_scale.json", scale)
        print(json.dumps(scale, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
