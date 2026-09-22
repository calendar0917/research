#!/usr/bin/env python
"""TCCD-v2 CHEM-CONT: label-free chemical continuity regularizer on the local latent.

Single diagnostic intervention on top of the frozen TCCD-v2 Prototype-REL
architecture. Pre-registration:
``tracks/ksvd/notes/tccd_v2_chemcont_preregistration.md``.

The architecture, optimizer, split, stopping protocol and all existing
regularizers are reused unchanged. The only change is one added label-free term

    L_chem = mean softplus(cos(z_i, z_neg) - cos(z_i, z_pos))

over deterministic cross-molecule training triplets, where the positive is
"VeryNear" (identical radius-1 chemistry, different canonical key) and the
negative is a matched "HardNegative" (same root type/degree, comparable patch
size, >=2 radius-1 edits). ``lambda_chem`` is calibrated once from detached
initial task/chem magnitudes (0.05 initial task share) and then frozen.

The chemistry decoder and tier definitions are lifted verbatim from the frozen
continuity audit ``tracks/ksvd/audit/tccd_v2_continuity/run_tccd_v2_continuity_audit.py``
so training and audit share one similarity definition.
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v2 as V

PROTOCOL_VERSION = "tccd_v2_chemcont"
RESULTS_DIR = V.RESULTS_DIR.parent / "tccd_v2_chemcont"
CACHE_DIR = RESULTS_DIR / "cache"

# --- frozen intervention constants -----------------------------------------
PAIR_SEED = T.SPLIT_SEED
MAX_PAIRS_PER_ANCHOR = 4
PAIR_ATTEMPTS = 128
LAMBDA_SHARE = 0.05


# ===========================================================================
# chemistry decoding (same construction as the frozen continuity audit)
# ===========================================================================
def decode_chemistry(
    records: Sequence[Mapping[str, Any]],
    graph_indices: Sequence[int],
    *,
    layout: T.PatchLayout | None = None,
    log=print,
) -> dict[str, Any]:
    """Decode deterministic structure-only chemistry for every patch of the graphs.

    Returns flat patch-level arrays plus a ``c1`` helper. No target ``y`` is read.
    """
    layout = V.frozen_layout() if layout is None else layout
    cap = layout.capacity
    na = layout.n_atom
    nb = layout.n_bond
    sl = layout.block_slices()
    pi, pj = layout.pair_i, layout.pair_j
    pid = np.full((cap, cap), -1, dtype=np.int64)
    pid[pi, pj] = np.arange(len(pi))

    gid_list: list[int] = []
    key_list: list[bytes] = []
    root_l: list[int] = []
    degree_l: list[int] = []
    n_l: list[int] = []
    l1_l: list[int] = []
    s1_l: list[np.ndarray] = []
    rb_l: list[np.ndarray] = []
    l1map: dict[Any, int] = {}

    def l1_key(atom, shell, edges):
        sh1 = [s for s in range(len(shell)) if shell[s] == 1]
        rb, inner = {}, {}
        for a, b, bt in edges:
            if a == 0 and b in sh1:
                rb[b] = bt
            elif b == 0 and a in sh1:
                rb[a] = bt
            elif a in sh1 and b in sh1:
                inner[(min(a, b), max(a, b))] = bt
        best = None
        for perm in itertools.permutations(sh1):
            pos = {v: i for i, v in enumerate(perm)}
            ent = []
            for v in perm:
                sub = tuple(
                    sorted(
                        (inner[(min(v, w), max(v, w))], pos[w])
                        for w in sh1
                        if w != v and (min(v, w), max(v, w)) in inner
                    )
                )
                ent.append((int(atom[v]), int(rb.get(v, -1)), sub))
            cand = (int(atom[0]), tuple(ent))
            if best is None or cand < best:
                best = cand
        return best

    t0 = time.time()
    for gpos, gi in enumerate(graph_indices):
        rec = records[int(gi)]
        n = int(rec["n"])
        X = np.asarray(rec["X"], dtype=np.float32)
        keys = rec["keys"]
        for v in range(n):
            row = X[v]
            mask = row[sl["mask"]] > 0.5
            ni = int(mask.sum())
            atom = row[sl["atom"]].reshape(cap, na).argmax(1)[:ni].astype(np.int64)
            shell = row[sl["shell"]].reshape(cap, 3).argmax(1)[:ni].astype(np.int64)
            topo = row[sl["topology"]] > 0.5
            bond = row[sl["bond"]].reshape(nb, len(pi)).argmax(0)
            edges = []
            for s in range(ni):
                for t in range(s + 1, ni):
                    q = pid[s, t]
                    if topo[q]:
                        edges.append((s, t, int(bond[q])))
            s1 = np.zeros(na, dtype=np.int64)
            rb = np.zeros(nb, dtype=np.int64)
            for s in range(ni):
                if shell[s] == 1:
                    s1[int(atom[s])] += 1
            for a, b, bt in edges:
                if a == 0 and shell[b] == 1:
                    rb[bt] += 1
                elif b == 0 and shell[a] == 1:
                    rb[bt] += 1
            k = l1_key(atom, shell, edges)
            if k not in l1map:
                l1map[k] = len(l1map)
            gid_list.append(gpos)
            key_list.append(keys[v])
            root_l.append(int(atom[0]))
            n_l.append(ni)
            degree_l.append(int(s1.sum()))
            s1_l.append(s1)
            rb_l.append(rb)
            l1_l.append(l1map[k])
    out = {
        "n_patches": len(root_l),
        "graph_of": np.asarray(gid_list, dtype=np.int64),
        "keys": key_list,
        "root": np.asarray(root_l, dtype=np.int64),
        "degree": np.asarray(degree_l, dtype=np.int64),
        "n_atoms": np.asarray(n_l, dtype=np.int64),
        "l1id": np.asarray(l1_l, dtype=np.int64),
        "s1atom": np.asarray(s1_l, dtype=np.int64),
        "rbond": np.asarray(rb_l, dtype=np.int64),
        "n_unique_l1": len(l1map),
    }
    log(
        f"[chem] decoded {out['n_patches']} train patches from {len(graph_indices)} graphs "
        f"({out['n_unique_l1']} radius-1 keys) in {time.time() - t0:.1f}s"
    )
    return out


def key_ids(keys: Sequence[bytes]) -> np.ndarray:
    kmap: dict[bytes, int] = {}
    out = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in kmap:
            kmap[k] = len(kmap)
        out[i] = kmap[k]
    return out


def c1_edits(s1atom: np.ndarray, rbond: np.ndarray, i: int, j: int) -> int:
    return int(0.5 * np.abs(s1atom[i] - s1atom[j]).sum() + 0.5 * np.abs(rbond[i] - rbond[j]).sum())


# ===========================================================================
# deterministic training pair cache (training split only)
# ===========================================================================
def _sample_positive(i, cand, kkeys, graph_of, rng, max_pairs, attempts=PAIR_ATTEMPTS):
    out: list[int] = []
    seen: set[int] = set()
    if len(cand) < 2:
        return out
    for _ in range(attempts):
        if len(out) >= max_pairs:
            break
        j = int(cand[rng.integers(0, len(cand))])
        if j == i or kkeys[j] == kkeys[i] or graph_of[j] == graph_of[i]:
            continue
        if j in seen:
            continue
        seen.add(j)
        out.append(j)
    return out


def _sample_negative(i, cand, n_atoms, s1atom, rbond, graph_of, rng, max_pairs, attempts=PAIR_ATTEMPTS):
    out: list[int] = []
    seen: set[int] = set()
    if len(cand) < 2:
        return out
    for _ in range(attempts):
        if len(out) >= max_pairs:
            break
        j = int(cand[rng.integers(0, len(cand))])
        if j == i or graph_of[j] == graph_of[i]:
            continue
        if abs(int(n_atoms[j]) - int(n_atoms[i])) > 1:
            continue
        if c1_edits(s1atom, rbond, i, j) < 2:
            continue
        if j in seen:
            continue
        seen.add(j)
        out.append(j)
    return out


def build_pairs(
    records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    layout: T.PatchLayout | None = None,
    seed: int = PAIR_SEED,
    max_pairs: int = MAX_PAIRS_PER_ANCHOR,
    log=print,
) -> dict[str, Any]:
    """Build the deterministic CHEM-CONT triplet cache from training graphs only.

    Does not read ``records[...]['y']`` anywhere.
    """
    train_indices = [int(i) for i in train_indices]
    chem = decode_chemistry(records, train_indices, layout=layout, log=log)
    P = int(chem["n_patches"])
    graph_of = chem["graph_of"]
    root = chem["root"]
    degree = chem["degree"]
    n_atoms = chem["n_atoms"]
    l1id = chem["l1id"]
    s1atom = chem["s1atom"]
    rbond = chem["rbond"]
    kkeys = key_ids(chem["keys"])

    l1_groups: dict[int, list[int]] = collections.defaultdict(list)
    rd_groups: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for i in range(P):
        l1_groups[int(l1id[i])].append(i)
        rd_groups[(int(root[i]), int(degree[i]))].append(i)
    l1_arrs = {k: np.asarray(v, dtype=np.int64) for k, v in l1_groups.items()}
    rd_arrs = {k: np.asarray(v, dtype=np.int64) for k, v in rd_groups.items()}

    rng = np.random.default_rng(int(seed))
    A: list[int] = []
    Pp: list[int] = []
    Nn: list[int] = []
    Apos: list[int] = []
    unmatched = 0
    n_nopos = 0
    n_noneg = 0
    t0 = time.time()
    for i in range(P):
        pos = _sample_positive(i, l1_arrs[int(l1id[i])], kkeys, graph_of, rng, max_pairs)
        if not pos:
            n_nopos += 1
            unmatched += 1
            continue
        neg = _sample_negative(
            i, rd_arrs[(int(root[i]), int(degree[i]))], n_atoms, s1atom, rbond, graph_of, rng, max_pairs
        )
        if not neg:
            n_noneg += 1
            unmatched += 1
            continue
        m = min(len(pos), len(neg))
        for t in range(m):
            A.append(i)
            Pp.append(pos[t])
            Nn.append(neg[t])
            Apos.append(int(graph_of[i]))

    anchor = np.asarray(A, dtype=np.int64)
    pos = np.asarray(Pp, dtype=np.int64)
    neg = np.asarray(Nn, dtype=np.int64)
    anchor_pos = np.asarray(Apos, dtype=np.int64)
    log(
        f"[pairs] {P} candidate anchors -> {anchor.size} triplets "
        f"({int(np.unique(anchor).size)} anchors used, {unmatched} unmatched, {time.time() - t0:.1f}s)"
    )

    # tier sanity (must be 100% on the sampled partners)
    pos_l1 = float((l1id[pos] == l1id[anchor]).mean()) if anchor.size else 0.0
    pos_keydiff = float((kkeys[pos] != kkeys[anchor]).mean()) if anchor.size else 0.0
    neg_rd = float(((root[neg] == root[anchor]) & (degree[neg] == degree[anchor])).mean()) if anchor.size else 0.0
    neg_size = float((np.abs(n_atoms[neg] - n_atoms[anchor]) <= 1).mean()) if anchor.size else 0.0
    neg_c1 = np.asarray(
        [c1_edits(s1atom, rbond, int(a), int(b)) for a, b in zip(anchor, neg)], dtype=np.int64
    )
    cross = (
        float(((graph_of[pos] != graph_of[anchor]) & (graph_of[neg] != graph_of[anchor])).mean())
        if anchor.size
        else 0.0
    )
    stats: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "seed": int(seed),
        "max_pairs_per_anchor": int(max_pairs),
        "attempts": PAIR_ATTEMPTS,
        "n_train_graphs": len(train_indices),
        "n_patches": P,
        "n_unique_l1": int(chem["n_unique_l1"]),
        "n_anchors_total": P,
        "n_anchors_used": int(np.unique(anchor).size),
        "n_triplets": int(anchor.size),
        "n_positive_pairs": int(anchor.size),
        "n_negative_pairs": int(anchor.size),
        "unmatched_anchor_count": int(unmatched),
        "unmatched_anchor_fraction": float(unmatched / max(P, 1)),
        "anchors_without_positive": int(n_nopos),
        "anchors_without_negative": int(n_noneg),
        "graph_coverage": float(np.unique(anchor_pos).size / max(len(train_indices), 1)),
        "positive_l1_equal_rate": pos_l1,
        "positive_key_diff_rate": pos_keydiff,
        "negative_root_degree_match_rate": neg_rd,
        "negative_size_match_rate": neg_size,
        "negative_c1_ge2_rate": float((neg_c1 >= 2).mean()) if neg_c1.size else 0.0,
        "cross_molecule_rate": cross,
        "negative_c1_edits_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(neg_c1, return_counts=True))}
        if neg_c1.size
        else {},
        "official_test_loaded": False,
        "uses_target_label": False,
        "wall_s": time.time() - t0,
    }
    log(
        f"[pairs] sanity pos_l1={pos_l1:.4f} pos_keydiff={pos_keydiff:.4f} "
        f"neg_rd={neg_rd:.4f} neg_size={neg_size:.4f} neg_c1>=2={stats['negative_c1_ge2_rate']:.4f} "
        f"cross={cross:.4f}"
    )
    return {
        "anchor": anchor,
        "positive": pos,
        "negative": neg,
        "anchor_pos": anchor_pos,
        "stats": stats,
    }


def pairs_cache_paths(seed: int = PAIR_SEED) -> tuple[Path, Path]:
    return (
        CACHE_DIR / f"chemcont_pairs_seed{seed}.npz",
        CACHE_DIR / f"chemcont_pairs_seed{seed}_stats.json",
    )


def save_pairs(pairs: Mapping[str, Any], seed: int = PAIR_SEED) -> tuple[Path, Path]:
    npz, js = pairs_cache_paths(seed)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz,
        anchor=pairs["anchor"],
        positive=pairs["positive"],
        negative=pairs["negative"],
        anchor_pos=pairs["anchor_pos"],
    )
    js.write_text(json.dumps(pairs["stats"], indent=2, sort_keys=True, default=float))
    return npz, js


def load_pairs(seed: int = PAIR_SEED) -> dict[str, Any] | None:
    npz, js = pairs_cache_paths(seed)
    if not npz.exists() or not js.exists():
        return None
    obj = np.load(npz)
    return {
        "anchor": obj["anchor"],
        "positive": obj["positive"],
        "negative": obj["negative"],
        "anchor_pos": obj["anchor_pos"],
        "stats": json.loads(js.read_text()),
    }


# ===========================================================================
# continuity loss + training loop
# ===========================================================================
def _gather_rows(trip_by_pos: Sequence[np.ndarray], positions: Sequence[int]) -> np.ndarray:
    chunks = [trip_by_pos[int(p)] for p in positions if trip_by_pos[int(p)].size]
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chunks)


def _trips_by_position(anchor_pos: np.ndarray, n_positions: int) -> list[np.ndarray]:
    out: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_positions)]
    order = np.argsort(anchor_pos, kind="stable")
    if order.size == 0:
        return out
    ua, starts = np.unique(anchor_pos[order], return_index=True)
    for k, pos in enumerate(ua):
        st = int(starts[k])
        en = int(starts[k + 1]) if k + 1 < len(starts) else int(order.size)
        out[int(pos)] = order[st:en]
    return out


def _patch_matrix(records: Sequence[Mapping[str, Any]], train_indices: Sequence[int], device: str):
    torch = V._torch()
    X = np.concatenate(
        [np.asarray(records[int(gi)]["X"], dtype=np.float32) for gi in train_indices], axis=0
    )
    return torch.as_tensor(X, dtype=torch.float32, device=device)


def _chem_loss(za, zp, zn):
    torch = V._torch()
    F = torch.nn.functional
    za = F.normalize(za, dim=-1, eps=V.EPS)
    zp = F.normalize(zp, dim=-1, eps=V.EPS)
    zn = F.normalize(zn, dim=-1, eps=V.EPS)
    s_pos = (za * zp).sum(dim=-1)
    s_neg = (za * zn).sum(dim=-1)
    return F.softplus(s_neg - s_pos).mean()


def _chem_loss_rows(model, Xpatch, tri_a, tri_p, tri_n):
    za = Xpatch.index_select(0, tri_a) @ model.W
    zp = Xpatch.index_select(0, tri_p) @ model.W
    zn = Xpatch.index_select(0, tri_n) @ model.W
    return _chem_loss(za, zp, zn)


def initial_regularization_chem(
    model,
    records_train,
    train_indices,
    device,
    *,
    batch,
    Xpatch,
    tri_a,
    tri_p,
    tri_n,
    trip_by_pos,
    log=print,
) -> dict[str, Any]:
    """Reuse the frozen TCCD-v2 calibration, then calibrate lambda_chem once."""
    base = V._initial_regularization(model, records_train, train_indices, device, batch=batch, log=log)
    positions = list(range(min(int(batch), len(train_indices))))
    rows = _gather_rows(trip_by_pos, positions)
    torch = V._torch()
    with torch.no_grad():
        if rows.size:
            rt = torch.as_tensor(rows, dtype=torch.long, device=device)
            chem = float(_chem_loss_rows(model, Xpatch, tri_a[rt], tri_p[rt], tri_n[rt]))
        else:
            chem = 0.0
    lam = LAMBDA_SHARE * float(base["initial_task"]) / max(chem, T.EPS)
    base.update(
        {
            "initial_chem": chem,
            "lambda_chem": float(lam),
            "lambda_chem_share": LAMBDA_SHARE,
            "initial_chem_contribution": float(lam * chem),
            "calibration_triplets": int(rows.size),
        }
    )
    log(
        "  [chem-calibration] "
        + str(
            {
                "initial_task": round(base["initial_task"], 6),
                "initial_chem": round(chem, 6),
                "lambda_chem": round(lam, 6),
                "initial_chem_contribution": round(lam * chem, 6),
                "calibration_triplets": int(rows.size),
            }
        )
    )
    return base


@dataclass
class ChemContTrainResult:
    best_valid: float
    best_epoch: int
    soup_valid: float | None
    soup_members: list[int]
    train_history: list[dict[str, Any]] = field(default_factory=list)
    state_best: Any = None
    state_soup: Any = None
    regularization: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    peak_mem_mb: float | None = None


def train_model_chemcont(
    model,
    records_train,
    records_dev,
    train_indices,
    dev_indices,
    device,
    *,
    pairs: Mapping[str, Any],
    seed: int = 0,
    max_epochs: int = T.MAX_EPOCHS,
    patience: int = T.PATIENCE,
    batch: int = T.BATCH,
    lr: float = T.LR,
    wd: float = T.WD,
    clip: float = T.CLIP,
    lambda_chem: float | None = None,
    log=print,
) -> ChemContTrainResult:
    """TCCD-v2 Prototype-REL training plus one frozen chemical-continuity term.

    With ``lambda_chem=0`` every operation is identical to ``tccd_v2.train_model``
    for the same seed/init (the chem term is skipped and consumes no RNG).
    """
    torch = V._torch()
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed) + 91011)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    Xpatch = _patch_matrix(records_train, train_indices, device)
    trip_by_pos = _trips_by_position(np.asarray(pairs["anchor_pos"], dtype=np.int64), len(train_indices))
    tri_a = torch.as_tensor(np.asarray(pairs["anchor"], dtype=np.int64), dtype=torch.long, device=device)
    tri_p = torch.as_tensor(np.asarray(pairs["positive"], dtype=np.int64), dtype=torch.long, device=device)
    tri_n = torch.as_tensor(np.asarray(pairs["negative"], dtype=np.int64), dtype=torch.long, device=device)

    reg = initial_regularization_chem(
        model,
        records_train,
        train_indices,
        device,
        batch=batch,
        Xpatch=Xpatch,
        tri_a=tri_a,
        tri_p=tri_p,
        tri_n=tri_n,
        trip_by_pos=trip_by_pos,
        log=log,
    )
    lam_local = float(reg["lambda_local"])
    lam_balance = float(reg["lambda_balance"])
    lam_chem = float(reg["lambda_chem"]) if lambda_chem is None else float(lambda_chem)
    reg["lambda_chem_applied"] = lam_chem
    reg["initial_chem_contribution_applied"] = lam_chem * float(reg["initial_chem"])

    best = math.inf
    best_epoch = -1
    best_state = None
    top: list[tuple[float, int, Any]] = []
    history: list[dict[str, Any]] = []
    stale = 0
    t0 = time.time()
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = rng.permutation(len(train_indices))
        ep_total = ep_task = ep_local = ep_balance = ep_chem = 0.0
        ep_n = 0
        for start in range(0, len(order), batch):
            positions = order[start : start + batch]
            sel = [train_indices[i] for i in positions]
            b = V.make_batch(records_train, sel, device)
            opt.zero_grad(set_to_none=True)
            pred, C_flat, _ = model.forward_padded(b)
            valid = b["valid"].reshape(-1)
            task = (pred - b["y"]).abs().mean()
            local = V.local_entropy(C_flat, valid)
            balance = V.balance_kl(C_flat, valid)
            if lam_chem > 0.0:
                rows = _gather_rows(trip_by_pos, positions)
                if rows.size:
                    rt = torch.as_tensor(rows, dtype=torch.long, device=device)
                    chem = _chem_loss_rows(model, Xpatch, tri_a[rt], tri_p[rt], tri_n[rt])
                else:
                    chem = torch.zeros((), device=device)
            else:
                chem = torch.zeros((), device=device)
            loss = task + lam_local * local + lam_balance * balance + lam_chem * chem
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            model.renormalize_()
            count = len(sel)
            ep_total += float(loss.detach()) * count
            ep_task += float(task.detach()) * count
            ep_local += float(local.detach()) * count
            ep_balance += float(balance.detach()) * count
            ep_chem += float(chem.detach()) * count
            ep_n += count
        dev_mae = V.evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        row = {
            "epoch": epoch,
            "train_loss": ep_total / max(ep_n, 1),
            "train_task": ep_task / max(ep_n, 1),
            "train_local_entropy": ep_local / max(ep_n, 1),
            "train_balance_kl": ep_balance / max(ep_n, 1),
            "train_chem": ep_chem / max(ep_n, 1),
            "valid": dev_mae,
            "temperature": float(model.temperature().detach()),
        }
        history.append(row)
        log(
            f"  epoch={epoch:03d} train={row['train_loss']:.6f} task={row['train_task']:.6f} "
            f"chem={row['train_chem']:.6f} valid={dev_mae:.6f} tau={row['temperature']:.5f} best@{best_epoch}"
        )
        if dev_mae < best - 1e-9:
            best = dev_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if len(top) < T.TOP_K_SOUP or dev_mae < max(t[0] for t in top):
            top.append((dev_mae, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}))
            top.sort(key=lambda t: t[0])
            top = top[: T.TOP_K_SOUP]
        if stale >= int(patience):
            log(f"  early stop at epoch {epoch}")
            break

    soup = None
    soup_mae = None
    members: list[int] = []
    if top:
        members = [e for _, e, _ in top]
        keys = top[0][2].keys()
        soup = {k: sum(st[k].float() for _, _, st in top) / len(top) for k in keys}
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(soup)
        soup_mae = V.evaluate_mae(model, records_dev, dev_indices, device, batch=64)
        model.load_state_dict(backup)
    if best_state is not None:
        model.load_state_dict(best_state)
    return ChemContTrainResult(
        best_valid=float(best),
        best_epoch=int(best_epoch),
        soup_valid=None if soup_mae is None else float(soup_mae),
        soup_members=members,
        train_history=history,
        state_best=best_state,
        state_soup=soup,
        regularization=reg,
        wall_s=time.time() - t0,
    )
