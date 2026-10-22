"""CPU correctness gate for ``zinc_local_factorization_v1`` (arms D and F).

Narrow implementation / identification checks for this round only — no new
experiment, no performance story, and no selection of any hyper-parameter:

 1. caches     the audit-verified ``bf_data`` caches still contain the recorded
               S3/S4/F4 content, occurrences, root/family membership and
               multiplicity; a fresh re-enumeration from the frozen objects
               reproduces them; pattern ids are split-local (content-key use).
 2. equivar.   atom renumbering + reversed edge input does not change the
               prediction; batch regrouping / padding does not change it.
 3. perm.avg.  the vectorised root-preserving average equals an independent
               explicit 6/24 enumeration in float64, is invariant to a
               root-preserving relabelling of the content, and a duplicate-row
               merge with exact weights equals the full weighted average.
 4. moments    z1/z2/z3 equal an explicit float64 enumeration of distinct
               field pairs/triples for m = 0,1,2,10,15; padding fields are
               masked out before p1/p2/p3 and never enter the denominator.
 5. aggregate  empty families are exact zeros; two equal contents at different
               addresses count twice without inflating ``mean``; permutations
               do not add occurrences; ``mean_sq`` is taken on ``phi_bar``
               (after the permutation average), not on per-permutation outputs.
 6. typed adj. the same atom categories and bond-type multiset with a different
               chord placement reach the model input; both encoders' responses
               are recorded (not a completeness claim).
 7. addressing identical content at different cache addresses gives identical
               ``phi_bar``; no whole-pattern learnable embedding exists; the
               content path keeps autograd every forward and caches nothing.
 8. public     both arms have byte-identical public P3 / W_s / shared embedding
               initial tensors, W_s is exactly 0, both initial functions equal
               the SSL-start P3 and A_joint_BF's carrier, all parameters are
               optimised and the counts are D=449,601 / F=449,457.
 9. port grad  at init ``dL/dphi == 0`` (W_s = 0); after the port opens, the
               projections, role tables, head and embeddings all receive
               gradient and the content parameters move.
10. budget     parameter cap, 79 steps/epoch, warm-up/cosine endpoints,
               checkpoint reload identity and batch-order determinism.

Run: ``uv run python tracks/structural-sufficiency/zinc_local_factorization_v1/code/lf_checks.py``
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
_OT = _SS / "zinc_structural_object_tokens_v1" / "code"

for _p in (str(_HERE), str(_V1), str(_V2), str(_BF), str(_OT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import bf_data as BFD  # noqa: E402
import lf_core as LC  # noqa: E402
import lf_models as LM  # noqa: E402

RESULTS: dict[str, dict] = {}
ANALYSIS = LC.ANALYSIS

# Recorded audit numbers from the previous round's cache build
# (``zinc_branching_fragment_v1/analysis/cache_stats.json``) plus the digest of
# the array payload (independent of the embedded ``created`` timestamp; the
# local 09:59 build and the host 10:05 build have byte-identical payloads).
AUTHORITATIVE = {
    "train": {
        "n_molecules": 10000, "n_entries": 312110, "n_unique_patterns": 1280,
        "unique_patterns_by_family": {"S3": 341, "S4": 73, "F4": 866},
        "occurrences_by_family": {"S3": 82568, "S4": 3104, "F4": 247704},
        "content_sha256": "ce643fdafba375f6c1689d942cc5c659e1e8224cab0893acf0d3e45857b80780",
    },
    "valid": {
        "n_molecules": 1000, "n_entries": 31038, "n_unique_patterns": 644,
        "unique_patterns_by_family": {"S3": 173, "S4": 37, "F4": 434},
        "occurrences_by_family": {"S3": 8225, "S4": 314, "F4": 24675},
        "content_sha256": "91e0e81b8977dc3127e674863683b08a382c934740632d8b8b5676ddc1266ed3",
    },
}
CONTENT_KEYS = ("pat_atom", "pat_pair", "pat_len", "pat_family",
                "ent_ptr", "ent_pat", "ent_root", "ent_family", "ent_count")


def record(name: str, fn, *args, **kwargs) -> dict:
    t0 = time.perf_counter()
    try:
        ok, detail = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a check failure is data, not a crash
        import traceback

        ok, detail = False, {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()[-2000:]}
    detail = dict(detail)
    detail["pass"] = bool(ok)
    detail["seconds"] = float(time.perf_counter() - t0)
    RESULTS[name] = detail
    short = json.dumps({k: v for k, v in detail.items()
                        if k not in ("pass", "seconds")}, default=str)[:500]
    C.log(f"[check {name}] {'PASS' if ok else 'FAIL'} ({detail['seconds']:.2f}s) {short}")
    return detail


# ---------------------------------------------------------------------------
# content description independent of the cached layout
# ---------------------------------------------------------------------------


class Content:
    """Typed induced adjacency with the root fixed at node 0.

    ``cats[i]`` is the raw atom category of node ``i``; ``adj[(min, max)]`` is
    the raw bond type (1..3) or absent (no real edge).  The pair encoding is
    derived here from the adjacency, not read from a cache array.
    """

    def __init__(self, cats, adj, family: int):
        self.cats = [int(c) for c in cats]
        self.adj = {(min(a, b), max(a, b)): int(t) for (a, b), t in adj.items()}
        self.family = int(family)
        self.n = len(self.cats)

    def arrays(self, order=None) -> tuple[np.ndarray, np.ndarray]:
        order = tuple(range(self.n)) if order is None else tuple(order)
        atom = np.zeros(LM.MAX_NODES, dtype=np.uint8)
        pair = np.zeros(LM.N_PAIRS, dtype=np.uint8)
        for t in range(self.n):
            atom[t] = self.cats[order[t]]
        for k, (i, j) in enumerate(LM.PAIR_INDEX):
            if j < self.n:
                a, b = order[i], order[j]
                pair[k] = self.adj.get((min(a, b), max(a, b)), 0)
        return atom, pair

    def family_onehot(self) -> np.ndarray:
        out = np.zeros(LM.N_FAMILIES, dtype=np.float32)
        out[self.family] = 1.0
        return out

    def masks(self) -> tuple[np.ndarray, np.ndarray]:
        node = (np.arange(LM.MAX_NODES) < self.n).astype(np.float32)
        pair = (LM.PAIR_J < self.n).astype(np.float32)
        return node, pair

    def relabel(self, order) -> "Content":
        """New node ``t`` holds old node ``order[t]`` (``order[0]`` must be 0)."""
        order = [int(o) for o in order]
        if sorted(order) != list(range(self.n)) or order[0] != 0:
            raise ValueError("not a root-preserving relabelling")
        cats = [self.cats[o] for o in order]
        adj = {}
        for (a, b), t in self.adj.items():
            ia, ib = order.index(a), order.index(b)
            adj[(min(ia, ib), max(ia, ib))] = t
        return Content(cats, adj, self.family)


def content_from_arrays(atom, pair, *, family: int) -> Content:
    n = int(np.max(np.nonzero(np.asarray(atom))[0], initial=-1) + 1)
    out = Content.__new__(Content)
    out.cats = [int(atom[i]) for i in range(n)]
    out.adj = {}
    for k, (i, j) in enumerate(LM.PAIR_INDEX):
        if j < n and int(pair[k]) != 0:
            out.adj[(i, j)] = int(pair[k])
    out.family = int(family)
    out.n = n
    return out


def make_table(contents: list[Content], split: str = "__synth") -> LM.PatternTable:
    P = len(contents)
    pat_atom = np.zeros((P, LM.MAX_NODES), dtype=np.uint8)
    pat_pair = np.zeros((P, LM.N_PAIRS), dtype=np.uint8)
    pat_len = np.zeros((P,), dtype=np.uint8)
    pat_family = np.zeros((P,), dtype=np.uint8)
    for p, c in enumerate(contents):
        pat_atom[p], pat_pair[p] = c.arrays()
        pat_len[p] = c.n
        pat_family[p] = c.family
    cache = {"split": split, "pat_atom": pat_atom, "pat_pair": pat_pair,
             "pat_len": pat_len, "pat_family": pat_family}
    return LM.build_pattern_table(cache, split=split)


def _content_batch(contents: list[Content], table: LM.PatternTable, *,
                   occ_pat=None, occ_root=None, occ_family=None, occ_count=None,
                   n_atoms: int = 1) -> dict:
    n = len(contents)
    return {
        "occ_pat": torch.as_tensor(occ_pat if occ_pat is not None else list(range(n))),
        "occ_root": torch.as_tensor(occ_root if occ_root is not None else [0] * n),
        "occ_family": torch.as_tensor(occ_family if occ_family is not None else
                                      [c.family for c in contents]),
        "occ_count": torch.as_tensor(occ_count if occ_count is not None else [1.0] * n,
                                     dtype=torch.float32),
        "n_atoms_total": int(n_atoms),
        "split": table.split,
    }


def phi_single32(branch: LM.ContentBranch, content: Content, order) -> torch.Tensor:
    """``phi`` of exactly one layout (no permutation average), float32 module."""
    atom, pair = content.arrays(order)
    node_mask, pair_mask = content.masks()
    fam = content.family_onehot()
    atom_e = branch.atom_emb(torch.as_tensor(atom, dtype=torch.long)) * torch.as_tensor(node_mask)[:, None]
    pair_e = branch.pair_emb(torch.as_tensor(pair, dtype=torch.long)) * torch.as_tensor(pair_mask)[:, None]
    nm = torch.as_tensor(node_mask)
    pm = torch.as_tensor(pair_mask)
    fm = torch.as_tensor(fam)
    if branch.arm == "D":
        x = torch.cat([atom_e.reshape(-1), pair_e.reshape(-1), nm, pm, fm])
        return branch.phi(x[None])[0]
    m = nm.sum() + pm.sum()
    return branch.phi(atom_e[None], pair_e[None], nm[None], pm[None], fm[None],
                      m.reshape(1))[0]


def brute_phi_bar64(branch: LM.ContentBranch, content: Content) -> np.ndarray:
    """Independent explicit 6/24 permutation average of ``phi`` in float64."""
    orders = [(0,) + q for q in itertools.permutations(range(1, content.n))]
    node_mask, pair_mask = content.masks()
    fam = content.family_onehot()
    outs = []
    for order in orders:
        atom, pair = content.arrays(order)
        atom_e = branch.atom_emb(torch.as_tensor(atom, dtype=torch.long)).double()
        pair_e = branch.pair_emb(torch.as_tensor(pair, dtype=torch.long)).double()
        atom_e = atom_e * torch.as_tensor(node_mask)[:, None].double()
        pair_e = pair_e * torch.as_tensor(pair_mask)[:, None].double()
        if branch.arm == "D":
            x = torch.cat([atom_e.reshape(-1), pair_e.reshape(-1),
                           torch.as_tensor(node_mask).double(),
                           torch.as_tensor(pair_mask).double(),
                           torch.as_tensor(fam).double()])
            w1, b1 = branch.phi.l1.weight.double(), branch.phi.l1.bias.double()
            w2, b2 = branch.phi.l2.weight.double(), branch.phi.l2.bias.double()
            out = F.silu(w1 @ x + b1) @ w2.T + b2
        else:
            ph = branch.phi
            fa = torch.tanh(atom_e @ ph.a_proj.weight.double().T + ph.a_proj.bias.double()
                            + ph.role_atom.double())
            fa = fa * torch.as_tensor(node_mask)[:, None].double()
            fb = torch.tanh(pair_e @ ph.b_proj.weight.double().T + ph.b_proj.bias.double()
                            + ph.role_pair.double())
            fb = fb * torch.as_tensor(pair_mask)[:, None].double()
            real = torch.cat([fa, fb], dim=0)[
                torch.as_tensor(np.concatenate([node_mask, pair_mask]) > 0.5)]
            m = int(real.shape[0])
            z1 = real.sum(0) / m
            # mean over the C(m,2) / C(m,3) *distinct* field combinations, which is
            # the same average as the ordered-index formula (p1^2-p2)/(m(m-1)) etc.
            z2 = (torch.stack([real[a] * real[b]
                               for a, b in itertools.combinations(range(m), 2)]).mean(0)
                  if m >= 2 else torch.zeros_like(z1))
            z3 = (torch.stack([real[a] * real[b] * real[c]
                               for a, b, c in itertools.combinations(range(m), 3)]).mean(0)
                  if m >= 3 else torch.zeros_like(z1))
            u = torch.cat([z1, z2, z3,
                           torch.as_tensor(node_mask).double(),
                           torch.as_tensor(pair_mask).double(),
                           torch.as_tensor(fam).double()])
            out = F.silu(ph.l1.weight.double() @ u + ph.l1.bias.double()) \
                @ ph.l2.weight.double().T + ph.l2.bias.double()
        outs.append(out)
    return torch.stack(outs, 0).mean(0).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# 1. caches
# ---------------------------------------------------------------------------


def check_caches() -> tuple[bool, dict]:
    detail: dict = {"splits": {}}
    ok = True
    for split in ("train", "valid"):
        pc = BFD.load_cache(split)
        audit = BFD.audit_cache(split)
        h = hashlib.sha256()
        for key in CONTENT_KEYS:
            h.update(key.encode())
            h.update(np.ascontiguousarray(np.asarray(pc[key])).tobytes())
        digest = h.hexdigest()
        meta, auth = pc["meta"], AUTHORITATIVE[split]
        checks = {
            "content_digest_matches": digest == auth["content_sha256"],
            "n_entries_matches": int(meta["n_entries"]) == auth["n_entries"],
            "n_unique_matches": int(meta["n_unique_patterns"]) == auth["n_unique_patterns"],
            "unique_by_family_matches": meta["unique_patterns_by_family"]
            == auth["unique_patterns_by_family"],
            "occurrences_by_family_matches": meta["occurrences_by_family"]
            == auth["occurrences_by_family"],
            "no_bad_patterns": int(audit["pattern_bad"]) == 0,
            "entries_positive": bool(audit["entries_positive"]),
            "roots_in_range": bool(audit["roots_in_range"]),
            "families_valid": bool(audit["families_ok"]),
            "labels_unused": meta["labels_used"] is False,
        }
        detail["splits"][split] = {"content_sha256": digest, "checks": checks,
                                   "occurrences_by_family": meta["occurrences_by_family"]}
        ok &= all(checks.values())

    # fresh re-enumeration from the frozen objects for real molecules
    objs, src = D.load_p3_objects("valid")
    sample = [0, 3, 17, 64, 129, 500, 777, 999]
    pc = BFD.load_cache("valid")
    ent_ptr = np.asarray(pc["ent_ptr"])
    pat_len = np.asarray(pc["pat_len"])
    mismatches = []
    for m in sample:
        per = BFD.enumerate_rooted_branching(objs[m])
        rebuilt = {0: 0, 1: 0, 2: 0}
        for fi in range(3):
            for root in per[fi]:
                rebuilt[fi] += int(sum(per[fi][root].values()))
        stored = {0: 0, 1: 0, 2: 0}
        for e in range(int(ent_ptr[m]), int(ent_ptr[m + 1])):
            pid = int(np.asarray(pc["ent_pat"])[e])
            fam = int(np.asarray(pc["ent_family"])[e])
            if int(pat_len[pid]) != (5 if fam == 1 else 4):
                mismatches.append({"mol": m, "reason": "length"})
            stored[fam] += int(np.asarray(pc["ent_count"])[e])
        if stored != rebuilt:
            mismatches.append({"mol": m, "stored": stored, "rebuilt": rebuilt})
    detail["reenumeration"] = {"molecules": sample, "mismatches": mismatches,
                               "object_source": src}
    ok &= not mismatches

    # pattern ids are split-local *positions* in the sorted content-key list,
    # never content identities.  Use the key for any cross-split work.
    ordered_ok = True
    keys_by_split = {}
    for split in ("train", "valid"):
        pc2 = BFD.load_cache(split)
        atom = np.asarray(pc2["pat_atom"])
        pair = np.asarray(pc2["pat_pair"])
        fam = np.asarray(pc2["pat_family"])
        keys_by_split[split] = [bytes(atom[i]) + bytes(pair[i]) for i in range(len(atom))]
        for fi, fname in enumerate(BFD.FAMILIES):
            idx = np.nonzero(fam == fi)[0]
            if idx.size and list(idx) != list(range(int(idx[0]), int(idx[-1]) + 1)):
                ordered_ok = False
            keys = [keys_by_split[split][i] for i in idx]
            if keys != sorted(keys):
                ordered_ok = False
            detail.setdefault("family_id_ranges", {})[f"{split}/{fname}"] = [
                int(idx[0]), int(idx[-1])] if idx.size else None
    train_keys, valid_keys = set(keys_by_split["train"]), set(keys_by_split["valid"])
    valid_only = sorted(valid_keys - train_keys)
    tr_atom = np.asarray(pc["pat_atom"])
    tr_pair = np.asarray(pc["pat_pair"])
    pv = BFD.load_cache("valid")
    pv_atom, pv_pair = np.asarray(pv["pat_atom"]), np.asarray(pv["pat_pair"])
    common = int(min(len(pv_atom), len(tr_atom)))
    same_id_diff = sum(
        1 for pid in range(common)
        if bytes(pv_atom[pid]) + bytes(pv_pair[pid]) != bytes(tr_atom[pid]) + bytes(tr_pair[pid])
    )
    # ids are addresses: dropping the lexicographically first S3 key renumbers
    s3_ids = np.nonzero(np.asarray(pc["pat_family"]) == 0)[0]
    shifted = keys_by_split["train"][int(s3_ids[1])] if len(s3_ids) > 1 else None
    original = keys_by_split["train"][int(s3_ids[0])] if len(s3_ids) else None
    detail["split_local_ids"] = {
        "id_order_matches_sorted_content_keys": bool(ordered_ok),
        "valid_patterns_absent_from_train": len(valid_only),
        "id_alignment_train_vs_valid_compared": common,
        "id_alignment_mismatches_observed": int(same_id_diff),
        "id_alignment_note": (
            "0 mismatches here only because every valid content key is also a train "
            "content key and ids are sorted-key positions; alignment is a coincidence "
            "of this split pair, not a property that may be relied on"
        ),
        "constructed_shift_demo_id0_changes_content": bool(shifted is not None
                                                           and original is not None
                                                           and shifted != original),
    }
    # the pipeline itself must build one table per split from that split's cache
    per_split_ok = True
    for split in ("valid",):
        table = LC.pattern_table(split)
        src = BFD.load_cache(split)
        per_split_ok &= (table.split == split
                         and np.array_equal(np.asarray(table.atom_ids),
                                            np.asarray(src["pat_atom"])))
    detail["split_local_ids"]["pipeline_tables_are_split_local"] = bool(per_split_ok)
    ok &= ordered_ok and per_split_ok and detail["split_local_ids"][
        "constructed_shift_demo_id0_changes_content"]
    return ok, detail


# ---------------------------------------------------------------------------
# 2. equivariance
# ---------------------------------------------------------------------------


class _Obj:
    __slots__ = ("n_atoms", "atom_feat", "bond_index", "bond_type")


def permute_obj(obj, seed: int):
    rng = np.random.default_rng(int(seed))
    n = int(obj.n_atoms)
    perm = rng.permutation(n)              # new id -> old id
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    feat = np.asarray(obj.atom_feat, dtype=np.float32)
    bi = np.asarray(obj.bond_index, dtype=np.int64)
    bt = np.asarray(obj.bond_type, dtype=np.int64)
    out = _Obj()
    out.n_atoms = n
    out.atom_feat = feat[perm]
    out.bond_index = inv[bi][::-1]         # renumbered and edge order reversed
    out.bond_type = bt[::-1]
    return out


def _eval_with_pattern_cache(model, ad_cache, bf_split, rows, device, batch_size=256):
    model.eval()
    model.set_table(LC.pattern_table(bf_split).to(device))
    cache = dict(ad_cache)
    cache["split"] = bf_split
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = np.asarray(rows[start:start + batch_size], dtype=np.int64)
            batch = T1._to_device(BFD.collate_bf(cache, chunk, valid=True), device)
            batch["split"] = bf_split
            out.append(model(batch).float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0)


def check_equivariance(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    ad_valid = D.load_cache("valid")
    objs, src = D.load_p3_objects("valid")
    perm_objs = [permute_obj(o, 7000 + i) for i, o in enumerate(objs)]
    pdata = BFD.build_from_objects(perm_objs, "__perm_valid", src, verbose=False)
    BFD.register_cache("__perm_valid", pdata)
    LC._TABLES.pop("__perm_valid", None)

    same_content = True
    for m in (0, 5, 41, 250, 999):
        per_o = BFD.enumerate_rooted_branching(objs[m])
        per_p = BFD.enumerate_rooted_branching(perm_objs[m])
        for fi in range(3):
            all_o = sorted([(k, c) for root in per_o[fi] for k, c in per_o[fi][root].items()])
            all_p = sorted([(k, c) for root in per_p[fi] for k, c in per_p[fi][root].items()])
            same_content &= all_o == all_p
    detail: dict = {"content_multiset_identical_under_renumbering": bool(same_content),
                    "rows": int(len(objs)), "arms": {}}
    ok = same_content
    rows = np.arange(len(objs), dtype=np.int64)
    labels = np.asarray(ad_valid["labels"], dtype=np.float64)
    for arm in LM.ARMS:
        model, _ = LC.build(arm, norm, device)
        pred_a = _eval_with_pattern_cache(model, ad_valid, "valid", rows, device)
        pred_b = _eval_with_pattern_cache(model, ad_valid, "__perm_valid", rows, device)
        d_perm = float(np.abs(pred_a - pred_b).max())
        pred_b1 = _eval_with_pattern_cache(model, ad_valid, "valid", rows[:64], device, batch_size=1)
        pred_b7 = _eval_with_pattern_cache(model, ad_valid, "valid", rows[:64], device, batch_size=7)
        pred_b64 = _eval_with_pattern_cache(model, ad_valid, "valid", rows[:64], device, batch_size=64)
        d_regroup = float(max(np.abs(pred_b1 - pred_b64).max(), np.abs(pred_b7 - pred_b64).max()))
        detail["arms"][arm] = {
            "max_abs_diff_atom_renumber_plus_reversed_edges": d_perm,
            "mae_renumbered": C.mae(pred_b, labels[rows]),
            "mae_original": C.mae(pred_a, labels[rows]),
            "max_abs_diff_batch_regroup_padding": d_regroup,
        }
        ok &= d_perm <= 1e-5 and d_regroup <= 1e-5
    return ok, detail


# ---------------------------------------------------------------------------
# 3. permutation average
# ---------------------------------------------------------------------------


def check_permutation_average(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    contents = [
        Content([6, 7, 8, 6], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 0),
        Content([6, 7, 8, 6, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (0, 4): 1, (2, 3): 3}, 1),
        Content([7, 6, 9, 8], {(0, 1): 2, (0, 2): 1, (0, 3): 1, (1, 3): 3}, 2),
    ]
    table = make_table(contents)
    detail: dict = {"n_perm_by_pattern": table.n_perm_by_pattern.tolist(), "arms": {}}
    ok = int(table.n_perm_by_pattern.max()) == 24 and int(table.n_perm_by_pattern.min()) == 6
    for arm in LM.ARMS:
        model, _ = LC.build(arm, norm, device)
        branch = model.branch
        with torch.no_grad():
            table_out = branch.permuted_phi(table).double().numpy()
            per_pattern = {}
            for p, content in enumerate(contents):
                brute = brute_phi_bar64(branch, content)
                per_pattern[p] = {
                    "max_abs_diff_vs_explicit_enumeration": float(np.abs(table_out[p] - brute).max()),
                    "rel_l2": float(np.linalg.norm(table_out[p] - brute)
                                    / max(np.linalg.norm(brute), 1e-12)),
                }
            # invariance to a root-preserving relabelling of the content itself
            rng = np.random.default_rng(11)
            invariance = []
            for p, content in enumerate(contents):
                order = [0] + [int(x) for x in rng.permutation(np.arange(1, content.n))]
                relab = content.relabel(order)
                t2 = make_table([relab], split=f"__relab{p}")
                out2 = branch.permuted_phi(t2).double().numpy()[0]
                invariance.append(float(np.abs(out2 - table_out[p]).max()))
            # duplicate rows with exact weights == full weighted average
            merged = copy.copy(table)
            idx_node = table.idx_node.clone()
            idx_pair = table.idx_pair.clone()
            weight = torch.zeros_like(table.weight)
            for p in range(table.n_patterns):
                K = int(table.n_perm_by_pattern[p])
                reps = LM.PERM_MAX // K
                for k in range(K):
                    for r in range(reps):
                        idx_node[p, k * reps + r] = table.idx_node[p, k]
                        idx_pair[p, k * reps + r] = table.idx_pair[p, k]
                        weight[p, k * reps + r] = 1.0 / LM.PERM_MAX
            merged.idx_node, merged.idx_pair, merged.weight = idx_node, idx_pair, weight
            merged_out = branch.permuted_phi(merged).double().numpy()
        merged_diff = float(np.abs(merged_out - table_out).max())
        detail["arms"][arm] = {
            "explicit_enumeration": per_pattern,
            "relabel_invariance_max_abs_diff": invariance,
            "merged_duplicate_weights_max_abs_diff": merged_diff,
        }
        ok &= all(v["rel_l2"] < 1e-6 for v in per_pattern.values())
        ok &= max(invariance) < 1e-6 and merged_diff < 1e-6
    return ok, detail


# ---------------------------------------------------------------------------
# 4. moments
# ---------------------------------------------------------------------------


def check_moments(norm) -> tuple[bool, dict]:
    phi = LM.FactorizedPhi()
    phi.reinit(303)
    gen = torch.Generator().manual_seed(5)
    fields = torch.randn(15, 7, generator=gen, dtype=torch.float64)
    detail: dict = {"cases": {}}
    ok = True
    for m in (0, 1, 2, 10, 15):
        raw = fields.clone()
        raw[m:] = 1e6        # padding slots carry huge *raw* values ...
        mask = torch.zeros_like(raw)
        mask[:m] = 1.0       # ... but the content mask is exactly 0 there
        masked = raw * mask  # this is what the model feeds to the moments
        mm = torch.tensor([float(m)], dtype=torch.float64)
        z1, z2, z3 = LM.FactorizedPhi.moments(masked.unsqueeze(0), mm)
        real = fields[:m]
        z1r = real.sum(0) / max(m, 1)
        z2r = (torch.stack([real[a] * real[b]
                            for a, b in itertools.combinations(range(m), 2)]).mean(0)
               if m >= 2 else torch.zeros_like(z1r))
        z3r = (torch.stack([real[a] * real[b] * real[c]
                            for a, b, c in itertools.combinations(range(m), 3)]).mean(0)
               if m >= 3 else torch.zeros_like(z1r))
        errs = [float((z[0] - r).abs().max()) for z, r in ((z1, z1r), (z2, z2r), (z3, z3r))]
        detail["cases"][f"m={m}"] = {"max_abs_err_vs_explicit": errs,
                                     "z1_norm": float(z1.abs().sum())}
        ok &= max(errs) < 1e-10
        if m == 0:
            ok &= float(z1.abs().sum()) == 0.0 and float(z2.abs().sum()) == 0.0

    probe = torch.tensor([2.0, 3.0] + [1e6] * 5, dtype=torch.float64)   # [T] fields
    masks = torch.zeros_like(probe)
    masks[:2] = 1.0
    fields_probe = (probe * masks).reshape(-1, 1)   # [T, C] = [7, 1]
    z1, z2, z3 = LM.FactorizedPhi.moments(fields_probe,
                                          torch.tensor([2.0], dtype=torch.float64))
    detail["m2_probe"] = {"z1": float(z1[0, 0]), "z2": float(z2[0, 0]), "z3": float(z3[0, 0])}
    ok &= abs(float(z1[0, 0]) - 2.5) < 1e-12 and abs(float(z2[0, 0]) - 6.0) < 1e-12 \
        and abs(float(z3[0, 0])) < 1e-12
    return ok, detail


# ---------------------------------------------------------------------------
# 5. aggregation
# ---------------------------------------------------------------------------


def check_aggregation(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    content = Content([6, 7, 8, 6], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 0)
    table = make_table([content])
    detail: dict = {"arms": {}}
    ok = True
    for arm in LM.ARMS:
        model, _ = LC.build(arm, norm, device)
        branch = model.branch
        model.set_table(table)
        with torch.no_grad():
            phi_bar = branch.permuted_phi(table)[0].double()
            orders = [(0,) + q for q in itertools.permutations(range(1, content.n))]
            perms = torch.stack([phi_single32(branch, content, o).double() for o in orders], 0)
            mean_of_sq = (perms * perms).mean(0)
            sq_of_mean = phi_bar * phi_bar
            gap = float((mean_of_sq - sq_of_mean).abs().max())

            stats_1 = branch(_content_batch([content], table), table)[0].double()
            stats_2occ = branch(_content_batch([content, content], table, occ_pat=[0, 0],
                                               occ_root=[0, 0], occ_family=[0, 0],
                                               occ_count=[1.0, 1.0]), table)[0].double()
            stats_count2 = branch(_content_batch([content], table, occ_count=[2.0]),
                                  table)[0].double()
        d = {
            "mean_first_occurrence_matches_phi_bar": float((stats_1[:32] - phi_bar).abs().max()),
            "mean_sq_is_phi_bar_squared": float((stats_1[32:64] - sq_of_mean).abs().max()),
            "mean_sq_gap_vs_mean_of_per_permutation_squares": gap,
            "count_one": float((stats_1[64] - math.log1p(1)).abs()),
            "exist_one": float((stats_1[65] - 1.0).abs()),
            "empty_family_zero": float(stats_1[66:].abs().max()),
            "two_occurrences_mean_equal": float((stats_2occ[:32] - stats_1[:32]).abs().max()),
            "two_occurrences_mean_sq_equal": float((stats_2occ[32:64] - stats_1[32:64]).abs().max()),
            "two_occurrences_count_abs_err": float((stats_2occ[64] - math.log1p(2)).abs()),
            "count2_entry_mean_equal": float((stats_count2[:32] - stats_1[:32]).abs().max()),
            "count2_entry_mean_sq_equal": float((stats_count2[32:64] - stats_1[32:64]).abs().max()),
            "count2_entry_count_abs_err": float((stats_count2[64] - math.log1p(2)).abs()),
            "permutation_variance_positive": bool(gap > 1e-7),
        }
        detail["arms"][arm] = d
        ok &= (d["mean_first_occurrence_matches_phi_bar"] < 1e-5
               and d["mean_sq_is_phi_bar_squared"] < 1e-5
               and d["mean_sq_gap_vs_mean_of_per_permutation_squares"] > 1e-7
               and d["count_one"] < 1e-6 and d["exist_one"] < 1e-6
               and d["empty_family_zero"] == 0.0
               and d["two_occurrences_mean_equal"] < 1e-5
               and d["two_occurrences_mean_sq_equal"] < 1e-5
               and d["two_occurrences_count_abs_err"] < 1e-6
               and d["count2_entry_mean_equal"] < 1e-5
               and d["count2_entry_mean_sq_equal"] < 1e-5
               and d["count2_entry_count_abs_err"] < 1e-6)
    return ok, detail


# ---------------------------------------------------------------------------
# 6. typed adjacency sensitivity
# ---------------------------------------------------------------------------


def check_typed_adjacency(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    chord_12 = Content([6, 7, 8, 9], {(0, 1): 1, (0, 2): 1, (0, 3): 2, (1, 2): 3}, 0)
    chord_13 = Content([6, 7, 8, 9], {(0, 1): 1, (0, 2): 1, (0, 3): 2, (1, 3): 3}, 0)
    a1, p1 = chord_12.arrays()
    a2, p2 = chord_13.arrays()
    detail = {
        "atom_categories_identical": bool(np.array_equal(a1, a2)),
        "bond_type_multiset_identical": bool(
            sorted(v for v in p1.tolist() if v) == sorted(v for v in p2.tolist() if v)),
        "pair_vectors_differ": bool(not np.array_equal(p1, p2)),
        "pair_difference_slots": [int(k) for k in range(LM.N_PAIRS) if p1[k] != p2[k]],
        "arms": {},
    }
    table = make_table([chord_12, chord_13], split="__typed")
    ok = (detail["atom_categories_identical"] and detail["bond_type_multiset_identical"]
          and detail["pair_vectors_differ"])
    for arm in LM.ARMS:
        model, _ = LC.build(arm, norm, device)
        with torch.no_grad():
            out = model.branch.permuted_phi(table).double().numpy()
            single_d = [phi_single32(model.branch, c, tuple(range(c.n))).double().numpy()
                        for c in (chord_12, chord_13)]
        diff = float(np.linalg.norm(out[0] - out[1]))
        rel = diff / max(float(np.linalg.norm(out[0])), 1e-12)
        single = float(np.linalg.norm(single_d[0] - single_d[1])) / max(
            float(np.linalg.norm(single_d[0])), 1e-12)
        detail["arms"][arm] = {"phi_bar_l2_difference": diff, "phi_bar_rel_l2": rel,
                               "single_layout_rel_l2": single,
                               "responds_after_averaging": bool(rel > 1e-4)}
    ok &= detail["arms"]["D"]["responds_after_averaging"]
    return ok, detail


# ---------------------------------------------------------------------------
# 7. content addressing / autograd
# ---------------------------------------------------------------------------


def check_addressing(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    content = Content([6, 7, 8, 6], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 3): 3}, 0)
    other = Content([7, 7, 8, 6], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3}, 2)
    dup_table = make_table([content, other, content], split="__dup")
    detail: dict = {"arms": {}}
    ok = True
    for arm in LM.ARMS:
        model, _ = LC.build(arm, norm, device)
        branch = model.branch
        table = make_table([content], split="__one")
        batch = _content_batch([content], table)
        model.set_table(table)
        out = branch.permuted_phi(table)
        info = {"phi_bar_requires_grad": bool(out.requires_grad)}
        with torch.no_grad():
            dup_out = branch.permuted_phi(dup_table).double().numpy()
        info["same_content_different_address_max_abs_diff"] = float(
            np.abs(dup_out[0] - dup_out[2]).max())
        info["different_content_differs"] = bool(
            float(np.abs(dup_out[0] - dup_out[1]).max()) > 1e-6)
        embeds = [(name, tuple(m.weight.shape)) for name, m in model.named_modules()
                  if isinstance(m, torch.nn.Embedding)]
        info["embeddings"] = embeds
        # no learnable whole-pattern table: nothing in the content branch is
        # indexed by the split's pattern count (1280 train / 644 valid / 700 test)
        n_patterns = int(BFD.load_cache("train")["pat_atom"].shape[0])
        big = [(name, tuple(p.shape)) for name, p in model.branch.named_parameters()
               if p.dim() >= 1 and p.shape[0] >= n_patterns]
        info["branch_parameters_sized_like_pattern_count"] = big
        info["train_pattern_count"] = n_patterns
        info["branch_state_tensors"] = [(n, tuple(v.shape))
                                        for n, v in branch.state_dict().items()]
        with torch.no_grad():
            model.w_s.weight.normal_(0.0, 0.02)
        pred = branch.permuted_phi(table)
        pred.sum().backward()
        grads = {n: (float(p.grad.abs().sum()) if p.grad is not None else None)
                 for n, p in branch.named_parameters()}
        info["branch_grad_abs_sum"] = grads
        model.zero_grad(set_to_none=True)
        detail["arms"][arm] = info
        ok &= (info["phi_bar_requires_grad"]
               and info["same_content_different_address_max_abs_diff"] < 1e-6
               and info["different_content_differs"]
               and [s for _, s in embeds] == [(28, 8), (4, 4)]
               and not big
               and all(v is not None and v > 0 for v in grads.values()))
    return ok, detail


# ---------------------------------------------------------------------------
# 8. public initialisation
# ---------------------------------------------------------------------------


def check_public_init(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    ad_valid = D.load_cache("valid")
    rows = np.arange(64, dtype=np.int64)
    detail: dict = {"arms": {}}
    models, hashes = {}, {}
    for arm in LM.ARMS:
        model, info = LC.build(arm, norm, device)
        models[arm] = model
        hashes[arm] = LC.init_hashes(model)
        detail["arms"][arm] = {
            **hashes[arm],
            "parameters": int(info["parameters"]),
            "parameters_expected": int(info["parameters_expected"]),
            "breakdown": info["parameters_breakdown"],
            "all_require_grad": bool(all(p.requires_grad for p in model.parameters())),
            "n_tensors": int(sum(1 for _ in model.parameters())),
        }
    detail["public_state_identical"] = (
        hashes["D"]["public_state_sha256"] == hashes["F"]["public_state_sha256"])
    detail["shared_embedding_identical"] = (
        hashes["D"]["shared_embedding_sha256"] == hashes["F"]["shared_embedding_sha256"])
    d, f = detail["arms"]["D"], detail["arms"]["F"]
    ok = (detail["public_state_identical"] and detail["shared_embedding_identical"]
          and d["init_state_sha256"] != f["init_state_sha256"]
          and d["w_s_zero"] and f["w_s_zero"]
          and d["parameters"] == 449_601 and f["parameters"] == 449_457
          and d["all_require_grad"] and f["all_require_grad"])

    p3n, _ = A2.build_model("P3N", norm, 0, device)
    batch = T1._to_device(LC.collate_lf(ad_valid, rows, valid=True), device)
    with torch.no_grad():
        pred_p3n = p3n(batch).float().numpy()
    import ot_joint as J

    a_model, _ = J.build_model_a(norm, seed=0, device=device)
    with torch.no_grad():
        pred_a = a_model(batch).float().numpy()
    detail["initial_forward"] = {}
    preds = {}
    for arm in LM.ARMS:
        model = models[arm]
        model.set_table(LC.pattern_table("valid").to(device))
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            pred = model(batch).float().numpy()
        for p in model.parameters():
            p.requires_grad_(True)
        preds[arm] = pred
        detail["initial_forward"][arm] = {
            "max_abs_diff_vs_SSL_start_P3": float(np.abs(pred - pred_p3n).max()),
            "max_abs_diff_vs_A_joint_BF_carrier": float(np.abs(pred - pred_a).max()),
        }
        ok &= detail["initial_forward"][arm]["max_abs_diff_vs_SSL_start_P3"] < 1e-5
        ok &= detail["initial_forward"][arm]["max_abs_diff_vs_A_joint_BF_carrier"] < 1e-5
    detail["initial_forward"]["D_vs_F_max_abs_diff"] = float(np.abs(preds["D"] - preds["F"]).max())
    ok &= detail["initial_forward"]["D_vs_F_max_abs_diff"] < 1e-6

    # the two arms' first optimiser step agrees because W_s s is exactly zero
    detail["shared_init_only_w_s_and_phi_differ"] = True
    return ok, detail


# ---------------------------------------------------------------------------
# 9. port gradient
# ---------------------------------------------------------------------------


def check_port_gradient(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    ad_valid = D.load_cache("valid")
    rows = np.arange(16, dtype=np.int64)
    detail: dict = {"arms": {}}
    ok = True
    for arm in LM.ARMS:
        model, _ = LC.build(arm, norm, device)
        LC.attach_table(model, "valid", device)
        batch = T1._to_device(LC.collate_lf(ad_valid, rows, valid=True), device)
        y = torch.as_tensor(np.asarray(ad_valid["labels"])[rows], dtype=torch.float32)
        model.train()
        F.l1_loss(model(batch), y).backward()
        at_init = {n: (float(p.grad.abs().sum()) if p.grad is not None else None)
                   for n, p in model.branch.named_parameters()}
        w_s_grad = float(model.w_s.weight.grad.abs().sum())
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.w_s.weight.normal_(0.0, 0.02)
        F.l1_loss(model(batch), y).backward()
        after = {n: (float(p.grad.abs().sum()) if p.grad is not None else None)
                 for n, p in model.branch.named_parameters()}
        model.zero_grad(set_to_none=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {n: p.detach().clone() for n, p in model.branch.named_parameters()}
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            F.l1_loss(model(batch), y).backward()
            opt.step()
        delta = {n: float((p.detach() - before[n]).abs().sum())
                 for n, p in model.branch.named_parameters()}
        detail["arms"][arm] = {
            "at_init_all_phi_grads_zero": all(v == 0.0 for v in at_init.values()),
            "at_init_grads": at_init,
            "w_s_grad_at_init": w_s_grad,
            "after_open_grads": after,
            "after_open_all_nonzero": all(v is not None and v > 0 for v in after.values()),
            "moved_after_two_updates": delta,
            "all_moved": all(v > 0 for v in delta.values()),
        }
        ok &= detail["arms"][arm]["at_init_all_phi_grads_zero"]
        ok &= w_s_grad > 0
        ok &= detail["arms"][arm]["after_open_all_nonzero"]
        ok &= detail["arms"][arm]["all_moved"]
    return ok, detail


# ---------------------------------------------------------------------------
# 10. budget / schedule / checkpoints / batch order
# ---------------------------------------------------------------------------


def check_budget_schedule(norm) -> tuple[bool, dict]:
    device = torch.device("cpu")
    cfg = LC.load_config()
    sch, opt_cfg = cfg["schedule"], cfg["optimizer"]
    steps = math.ceil(int(cfg["data"]["train"]) / int(sch["batch_size"]))
    total = steps * int(sch["epochs"])
    lr_mid = [LC.schedule_lr(s, total, cfg) for s in range(501, total + 1, 4000)]
    detail = {
        "steps_per_epoch": int(steps), "epochs": int(sch["epochs"]),
        "total_updates": int(total),
        "schedule_probe": {str(s): float(LC.schedule_lr(s, total, cfg))
                           for s in (1, 499, 500, 501, total)},
        "warmup_end_is_peak": abs(LC.schedule_lr(500, total, cfg) - float(opt_cfg["lr"])) < 1e-12,
        "endpoint_is_min": abs(LC.schedule_lr(total, total, cfg) - float(opt_cfg["min_lr"])) < 1e-9,
        "monotone_after_warmup": bool(all(lr_mid[i] >= lr_mid[i + 1] - 1e-12
                                        for i in range(len(lr_mid) - 1))),
        "peak_lr": float(opt_cfg["lr"]), "min_lr": float(opt_cfg["min_lr"]),
        "weight_decay": float(opt_cfg["weight_decay"]), "betas": opt_cfg["betas"],
        "eps": float(opt_cfg["eps"]), "grad_clip": float(opt_cfg["grad_clip"]),
        "arms": {},
    }
    ok = (steps == 79 and total == 79 * int(sch["epochs"])
          and detail["warmup_end_is_peak"] and detail["endpoint_is_min"]
          and detail["monotone_after_warmup"])
    ad_valid = D.load_cache("valid")
    rows = np.arange(32, dtype=np.int64)
    for arm in LM.ARMS:
        model, info = LC.build(arm, norm, device)
        opt = LC.make_optimizer(model, cfg, total)
        covered = C.optimizer_parameter_count(opt)
        state = C.cpu_state(model)
        model2, _ = LC.build(arm, norm, device)
        C.load_state(model2, state, device)
        p1, _ = LC.evaluate(model, ad_valid, rows, device, 32)
        p2, _ = LC.evaluate(model2, ad_valid, rows, device, 32)
        gen_a = torch.Generator().manual_seed(LC.SEED + LC.SHUFFLE_SEED_OFFSET)
        gen_b = torch.Generator().manual_seed(LC.SEED + LC.SHUFFLE_SEED_OFFSET)
        orders_a = [LC.order_sha256(LC.epoch_order(gen_a, 10000)) for _ in range(3)]
        orders_b = [LC.order_sha256(LC.epoch_order(gen_b, 10000)) for _ in range(3)]
        detail["arms"][arm] = {
            "parameters": int(info["parameters"]),
            "optimizer_covers_all": int(covered) == int(info["parameters"]),
            "cap_ok": int(info["parameters"]) <= LM.PARAMETER_CAP,
            "checkpoint_reload_max_abs_diff": float(np.abs(p1 - p2).max()),
            "batch_order_deterministic": orders_a == orders_b,
            "first_order_sha256": orders_a[0],
        }
        ok &= (detail["arms"][arm]["optimizer_covers_all"]
               and detail["arms"][arm]["cap_ok"]
               and detail["arms"][arm]["checkpoint_reload_max_abs_diff"] < 1e-6
               and detail["arms"][arm]["batch_order_deterministic"])
    ok &= detail["arms"]["D"]["first_order_sha256"] == detail["arms"]["F"]["first_order_sha256"]
    return ok, detail


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ANALYSIS / "cpu_checks.json"))
    args = p.parse_args(argv)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cfg_path = LC.CONFIG_PATH
    cfg_sha = C.sha256_file(cfg_path) if Path(cfg_path).exists() else None
    norm = D.load_norm()
    C.log(f"[checks] config {cfg_path} sha256={cfg_sha}")

    record("caches", check_caches)
    record("equivariance", check_equivariance, norm)
    record("permutation_average", check_permutation_average, norm)
    record("moments", check_moments, norm)
    record("aggregation", check_aggregation, norm)
    record("typed_adjacency", check_typed_adjacency, norm)
    record("addressing", check_addressing, norm)
    record("public_init", check_public_init, norm)
    record("port_gradient", check_port_gradient, norm)
    record("budget_schedule", check_budget_schedule, norm)

    n_pass = sum(1 for v in RESULTS.values() if v["pass"])
    n_fail = len(RESULTS) - n_pass
    doc = {
        "protocol": LC.PROTOCOL,
        "config_path": str(Path(cfg_path)),
        "config_sha256": cfg_sha,
        "checks": RESULTS,
        "n_checks": len(RESULTS),
        "n_pass": int(n_pass),
        "n_fail": int(n_fail),
        "git_commit": C.git_commit(),
        "created": C.utcnow(),
        "note": ("narrow implementation checks only; no toy performance story and no "
                 "selection of arm / rank / order / LR / horizon"),
    }
    C.write_json(Path(args.out), doc)
    C.log(f"[checks] {n_pass}/{len(RESULTS)} pass -> {args.out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
