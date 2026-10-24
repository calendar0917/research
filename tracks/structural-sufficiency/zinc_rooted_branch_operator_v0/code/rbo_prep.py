"""Build the rooted-branch structure cache for RBO (train split only).

For every *directed* branch ``v -> u`` (both directions of every bond) this
stores what RBO is allowed to see, from raw structure only:

  centre atom id, neighbour atom id, bond type,
  hist(u minus v) : counts over (bond type 1..3) x (atom type 0..20) of u's
                    remaining incidences, plus the remaining degree.

and, for every unordered pair of branches sharing one root ``v``:

  rel = extra_bond_type(u_i, u_j) in {0,1,2,3}
        + 4 * min(|N(u_i) & N(u_j) - {v}|, 3)
        + 16 * clip(dist_{G-v}(u_i, u_j) - 2, 0, 3)   -> id 0..63

The distance bucket is the ring-separation of the two branches: 0 = they are
2-hop neighbours (4-ring through the shared neighbour), 1 = 3-hop (5-ring),
2 = 4-hop (6-ring), 3 = 5+ hops or different components.  It is pure
structure, so every pair relation is auditable without labels.

No labels, no radius-2 key, no learned/recursive neighbour state.  Everything is
per-molecule local so batching is index arithmetic (no Python loop in training).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_CACHE = _SSUF / "zinc_arch_discovery_v1" / "cache"
OUT = _TASK / "cache"
N_ATOM = 21
N_BOND = 3
HIST_DIM = N_BOND * N_ATOM + 1  # 64
N_REL = 64  # 4 extra-edge types x 4 shared-neighbour buckets x 4 distance buckets


def build(cache: dict) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    n_mol = int(cache["n_molecules"])

    br_ptr = [0]
    br_src_l, br_dst_l, br_bt, br_bond_l, br_mol = [], [], [], [], []
    hist = []
    pair_ptr = [0]
    pr_root_l, pr_i_l, pr_j_l, pr_rel, pr_mol = [], [], [], [], []
    n_atoms_per_mol = np.zeros(n_mol, dtype=np.int64)

    for m in range(n_mol):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        n = a1 - a0
        n_atoms_per_mol[m] = n
        types = at[a0:a1].tolist()
        adj = [[] for _ in range(n)]
        for (u, v), t in zip(bi[b0:b1].tolist(), bt[b0:b1].tolist()):
            adj[u].append((v, int(t)))
            adj[v].append((u, int(t)))
        for a in adj:
            a.sort()

        # directed branches, local ids within this molecule
        out_br = [[] for _ in range(n)]
        m_src, m_dst = [], []
        for bond_id, ((u, v), t) in enumerate(zip(bi[b0:b1].tolist(), bt[b0:b1].tolist())):
            for src, dst in ((u, v), (v, u)):
                bid = len(m_src)
                br_bond_l.append(bond_id)
                m_src.append(src)
                m_dst.append(dst)
                br_src_l.append(src)
                br_dst_l.append(dst)
                br_bt.append(int(t))
                br_mol.append(m)
                h = np.zeros(HIST_DIM, dtype=np.uint8)
                deg = 0
                for w, tw in adj[dst]:
                    if w == src:
                        continue
                    h[(tw - 1) * N_ATOM + types[w]] += 1
                    deg += 1
                h[HIST_DIM - 1] = deg
                hist.append(h)
                out_br[src].append(bid)
        br_ptr.append(len(br_src_l))

        # same-root branch pairs with a structural relation id
        for v in range(n):
            bs = out_br[v]
            if len(bs) < 2:
                continue
            # BFS in G - {v} from every neighbour of v
            dist = {}
            for i in bs:
                src = m_dst[i]
                d = {src: 0}
                queue = [src]
                while queue:
                    x = queue.pop(0)
                    for w, _t in adj[x]:
                        if w == v or w in d:
                            continue
                        d[w] = d[x] + 1
                        queue.append(w)
                dist[i] = d
            for i, j in itertools.combinations(bs, 2):
                ui, uj = m_dst[i], m_dst[j]
                extra = 0
                for w, tw in adj[ui]:
                    if w == uj:
                        extra = tw
                        break
                shared = len({w for w, _ in adj[ui]} & {w for w, _ in adj[uj]} - {v})
                d = dist[i].get(uj, 99)
                dbucket = max(0, min(int(d) - 2, 3))
                pr_root_l.append(v)
                pr_i_l.append(i)
                pr_j_l.append(j)
                pr_rel.append(int(extra) + 4 * min(shared, 3) + 16 * dbucket)
                pr_mol.append(m)
        pair_ptr.append(len(pr_root_l))

    out = {
        "br_ptr": np.asarray(br_ptr, dtype=np.int64),
        "br_src_local": np.asarray(br_src_l, dtype=np.int16),
        "br_dst_local": np.asarray(br_dst_l, dtype=np.int16),
        "br_bt": np.asarray(br_bt, dtype=np.int8),
        "br_bond_local": np.asarray(br_bond_l, dtype=np.int16),
        "bond_ptr": np.asarray(bp, dtype=np.int64),
        "br_hist": np.stack(hist, 0) if hist else np.zeros((0, HIST_DIM), np.uint8),
        "pair_ptr": np.asarray(pair_ptr, dtype=np.int64),
        "pair_root_local": np.asarray(pr_root_l, dtype=np.int16),
        "pair_i_local": np.asarray(pr_i_l, dtype=np.int16),
        "pair_j_local": np.asarray(pr_j_l, dtype=np.int16),
        "pair_rel": np.asarray(pr_rel, dtype=np.int8),
        "n_atoms_per_mol": n_atoms_per_mol,
        "atom_ptr": ap,
        "n_molecules": n_mol,
        "meta": {
            "hist_dim": HIST_DIM,
            "n_rel": N_REL,
            "n_branches": int(len(br_src_l)),
            "n_pairs": int(len(pr_root_l)),
            "atom_vocab": N_ATOM,
            "bond_vocab": N_BOND,
            "rel_definition": "extra_bond_type(0..3) + 4*min(shared_neighbours_excluding_root,3) + 16*clip(dist_{G-v}(u_i,u_j)-2,0,3)",
            "source_cache": "zinc_arch_discovery_v1/cache/ad_train.pt",
        },
    }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = torch.load(_CACHE / "ad_train.pt", map_location="cpu", weights_only=False)
    d = build(cache)
    torch.save(d, OUT / "rbo_train.pt")
    rel = np.bincount(d["pair_rel"].astype(np.int64), minlength=N_REL)
    stats = {
        **d["meta"],
        "rel_hist": rel.tolist(),
        "rel_frac": (rel / max(int(rel.sum()), 1)).tolist(),
        "mean_branches_per_atom": float(d["meta"]["n_branches"] / int(d["n_atoms_per_mol"].sum())),
        "mean_pairs_per_atom": float(d["meta"]["n_pairs"] / int(d["n_atoms_per_mol"].sum())),
    }
    (_TASK / "analysis").mkdir(exist_ok=True)
    (_TASK / "analysis" / "t0_prep.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
