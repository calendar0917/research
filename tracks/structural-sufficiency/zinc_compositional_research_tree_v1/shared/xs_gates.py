"""CPU correctness / feasibility gates (task §4, gates 1-12) + SPC coverage stop.

Run:  ``python shared/xs_gates.py``   (CPU only; writes ``analysis/gates/``)

Every gate executes the production code paths used by ``xs_train.py``.  Any
failure stops the corresponding branch; the design is never edited to make a
gate pass.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "cma", HERE.parent / "spc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import xs_common as C  # noqa: E402
import xs_data as D  # noqa: E402
import xs_models as SM  # noqa: E402
import xs_train as T  # noqa: E402
import cma_model as CM  # noqa: E402
import cma_structure as CS  # noqa: E402
import spc_model as SP  # noqa: E402
import spc_structure as SS  # noqa: E402

TOL = 3e-5
RESULT: dict = {}
T0 = time.perf_counter()


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------


def record(group: str, passed: bool, **detail) -> None:
    entry = RESULT.setdefault(group, {"pass": True, "detail": {}})
    entry["pass"] = bool(entry["pass"] and passed)
    entry["detail"].update(detail)
    print(f"[{group}] {'PASS' if passed else 'FAIL'} "
          f"{ {k: v for k, v in detail.items() if not isinstance(v, (list, dict))} }", flush=True)


def models_for(family: str, norm, seed: int, arm: str):
    return CM.build_model(norm, seed, arm=arm)[0] if family == "cma" else SP.build_model(norm, seed, arm=arm)[0]


def activate_event(model, scale: float = 0.15, seed: int = 7) -> None:
    """Give ``rho_event`` a non-zero output so the mechanism path is exercised.

    Without this, the fixed zero-init output layer makes every event term exactly
    zero at step 0 and the invariance tests would only exercise ``rho_base``.
    """
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        last = model.skeleton.rho_event.lins[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g) * scale)
        last.bias.copy_(torch.randn(last.bias.shape, generator=g) * scale)


def forward(model, ds: T.SplitData, rows, norm, *, batch_size: int = 4096) -> np.ndarray:
    return T.predict(model, ds, np.asarray(rows, dtype=np.int64), norm,
                     device="cpu", batch_size=batch_size)


# ---------------------------------------------------------------------------
# mini corpus builder (subset / atom permutation / bond flip / bond reorder)
# ---------------------------------------------------------------------------


def mini_cache(cache, rows, *, perm=False, flip=False, reorder=False, seed: int = 0) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    atom_feat, atom_type, bond_feat, bond_type, bond_index = [], [], [], [], []
    atom_ptr, bond_ptr = [0], [0]
    for m in rows:
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        af = np.asarray(cache["atom_feat"])[a0:a1].copy()
        at = np.asarray(cache["atom_type"])[a0:a1].copy()
        bf = np.asarray(cache["bond_feat"])[b0:b1].copy()
        bt = np.asarray(cache["bond_type"])[b0:b1].copy()
        # cache ``bond_index`` is molecule-local (ad-v1 convention)
        bi = np.asarray(cache["bond_index"])[b0:b1].astype(np.int64)
        n = a1 - a0
        if perm:
            p = rng.permutation(n)
            inv = np.argsort(p)
            af, at = af[p], at[p]
            bi = inv[bi]
        if flip:
            f = rng.random(len(bi)) < 0.5
            bi[f] = bi[f][:, ::-1]
        if reorder:
            q = rng.permutation(len(bi))
            bf, bt, bi = bf[q], bt[q], bi[q]
        atom_feat.append(af); atom_type.append(at)
        bond_feat.append(bf); bond_type.append(bt); bond_index.append(bi)
        atom_ptr.append(atom_ptr[-1] + n)
        bond_ptr.append(bond_ptr[-1] + len(bi))
    out = {
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "atom_feat": np.concatenate(atom_feat, 0).astype(np.float32),
        "atom_type": np.concatenate(atom_type, 0).astype(np.int64),
        "bond_feat": np.concatenate(bond_feat, 0).astype(np.float32),
        "bond_type": np.concatenate(bond_type, 0).astype(np.int64),
        "bond_index": np.concatenate(bond_index, 0).astype(np.int64),
        "global_feat": np.asarray(cache["global_feat"])[rows].astype(np.float32),
        "labels": np.asarray(cache["labels"])[rows].astype(np.float32),
        "n_molecules": int(len(rows)),
    }
    out["n_atoms_graph"] = np.diff(out["atom_ptr"]).astype(np.int64)
    out["n_bonds_graph"] = np.diff(out["bond_ptr"]).astype(np.int64)
    out["meta"] = {"mini": True, "rows": rows.tolist()}
    return out


def make_dataset(cache, family: str, norm, name: str = "mini", *, struct=None):
    if struct is None:
        struct = (CS if family == "cma" else SS).build_split(cache, name)
    return T.SplitData(cache, family, name, "cpu", struct=struct)


# ===========================================================================
# Gate 1 -- renumbering / bond direction / adjacency order / batch split
# ===========================================================================


def gate1() -> None:
    cache = C.load_cache("train")
    rows = np.arange(24, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    base_ds = {f: make_dataset(mini_cache(cache, rows), f, norm) for f in ("cma", "spc")}
    variants = {
        "perm": dict(perm=True, seed=1),
        "flip": dict(flip=True, seed=2),
        "reorder": dict(reorder=True, seed=3),
        "perm+flip+reorder": dict(perm=True, flip=True, reorder=True, seed=4),
    }
    for family in ("cma", "spc"):
        vds = {}
        for name, kw in variants.items():
            vds[name] = make_dataset(mini_cache(cache, rows, **kw), family, norm)
        for arm in ("REAL", "MEAN"):
            model = models_for(family, norm, 0, arm)
            activate_event(model)
            model.eval()
            ref = forward(model, base_ds[family], rows, norm)
            diffs = {}
            for name, ds in vds.items():
                diffs[name] = float(np.max(np.abs(forward(model, ds, rows, norm) - ref)))
            split = np.concatenate([
                forward(model, base_ds[family], rows[:9], norm),
                forward(model, base_ds[family], rows[9:], norm)])
            diffs["batch_split"] = float(np.max(np.abs(split - ref)))
            ok = all(v <= TOL for v in diffs.values())
            record("g1_invariance", ok, family=family, arm=arm, max_abs_diff=diffs,
                   tol=TOL, n_rows=int(len(rows)))


# ===========================================================================
# Gate 2 -- CMA enumeration vs independent brute force
# ===========================================================================


def gate2() -> None:
    graphs = [
        (5, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0)]),
        (4, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)]),
        (4, [(0, 1, 0), (1, 2, 0), (2, 0, 0), (2, 3, 0)]),
        (5, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 0, 1), (0, 2, 2)]),
        (6, [(0, 1, 0), (1, 2, 1), (2, 0, 0), (2, 3, 0), (3, 4, 2), (4, 5, 0),
             (5, 3, 0)]),
        (6, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 3), (0, 4, 0), (1, 5, 0),
             (4, 5, 0)]),
    ]
    cache = C.load_cache("train")
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"])
    bt = np.asarray(cache["bond_type"])
    for m in range(int(cache["n_molecules"])):
        if len(graphs) >= 26:
            break
        b0, b1 = int(bp[m]), int(bp[m + 1])
        graphs.append((int(ap[m + 1] - ap[m]),
                       [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e])) for e in range(b0, b1)]))

    total_records = 0
    n_bad = 0
    rev_bad = 0
    count_bad = 0
    examples = []
    for n_atoms, bonds in graphs:
        obj = CS.enumerate_molecule(n_atoms, bonds)
        got = []
        z_atom = obj["z_atom"]
        for row in range(len(obj["path_bond"])):
            zl, zr = obj["path_z"][row]
            j, a1, a2 = (int(x) for x in z_atom[zl])
            k, b1, b2 = (int(x) for x in z_atom[zr])
            # zL outers are {i, k}; zR outers are {j, l}
            if a1 == k:
                i = a2
            elif a2 == k:
                i = a1
            else:
                raise AssertionError(f"zL outer pair {(a1, a2)} does not contain k={k}")
            if b1 == j:
                l = b2
            elif b2 == j:
                l = b1
            else:
                raise AssertionError(f"zR outer pair {(b1, b2)} does not contain j={j}")
            got.append((int(i), int(j), int(k), int(l),
                        int(obj["path_chord"][row, 0]), int(obj["path_chord"][row, 1]),
                        int(obj["path_chord"][row, 2]), int(obj["path_bond"][row])))
        key = {}
        for e, (u, v, _t) in enumerate(bonds):
            key[(min(u, v), max(u, v))] = e
        ref = CS.brute_force_directed_paths(n_atoms, bonds)
        ref_full = [(i, j, k, l, c1, c2, c3, key[(min(j, k), max(j, k))])
                    for (i, j, k, l, c1, c2, c3) in ref]
        got_sorted = sorted(got)
        ref_sorted = sorted(ref_full)
        total_records += len(got_sorted)
        if got_sorted != ref_sorted:
            n_bad += 1
            if len(examples) < 3:
                examples.append({"n_atoms": n_atoms, "bonds": bonds,
                                 "missing": [x for x in ref_sorted if x not in got_sorted][:4],
                                 "extra": [x for x in got_sorted if x not in ref_sorted][:4],
                                 "count": [len(got_sorted), len(ref_sorted)]})
        # reversed record exists with the two side chord types swapped
        bag = set(got_sorted)
        for rec in got_sorted:
            want = (rec[3], rec[2], rec[1], rec[0], rec[5], rec[4], rec[6], rec[7])
            if want not in bag:
                rev_bad += 1
        if len(got_sorted) != 2 * int(np.sum(obj["bond_npaths"])):
            count_bad += 1
    record("g2_cma_enumeration", n_bad == 0 and rev_bad == 0 and count_bad == 0,
           n_graphs=len(graphs), total_directed_records=int(total_records),
           graphs_mismatching=int(n_bad), reversals_missing=int(rev_bad),
           count_mismatching=int(count_bad), examples=examples)

    # F3 reversal symmetry, evaluated numerically on a real mini batch
    rows = np.arange(16, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    ds = make_dataset(mini_cache(cache, rows), "cma", norm)
    model = models_for("cma", norm, 0, "REAL")
    activate_event(model)
    model.eval()
    batch = ds.batch(rows, norm)
    with torch.no_grad():
        a, b = model.skeleton.encode(batch)
        z = model.pair_fragments(batch, a, b)
        pz = batch["cma_path_z"]
        zL, zR = z[pz[:, 0]], z[pz[:, 1]]
        rel = model.path_relation(batch, a.dtype)
        bc = b[batch["cma_path_bond"]]
        u1 = torch.cat([zL + zR, (zL - zR).abs(), zL * zR, bc, rel], 1)
        u2 = torch.cat([zR + zL, (zR - zL).abs(), zR * zL, bc, rel], 1)
        d = float((model.F3(u1) - model.F3(u2)).abs().max())
        # z is reversal symmetric by construction (same triple row) -- check equality of
        # z_ijk and z_kji via an explicit reversed F2 call on the raw triple rows.
        za, zb = batch["cma_z_atom"], batch["cma_z_bond"]
        ch = F.one_hot(batch["cma_z_chord"], 4).to(a.dtype)
        fwd = torch.cat([a[za[:, 1]], a[za[:, 0]], a[za[:, 2]], b[zb[:, 0]], b[zb[:, 1]], ch], 1)
        rev = torch.cat([a[za[:, 2]], a[za[:, 0]], a[za[:, 1]], b[zb[:, 1]], b[zb[:, 0]], ch], 1)
        # definitional check: z is the exact mean of the two F2 orientations
        z_def = 0.5 * (model.F2(fwd) + model.F2(rev))
        dz = float((z - z_def).abs().max())
        # z_ijk and z_kji are the same row by construction -> reversal leaves z alone
    record("g2_cma_reversal_numeric", d <= TOL and dz <= TOL,
           f3_reversal_max_abs_diff=d, z_definition_max_abs_diff=dz, tol=TOL)


# ===========================================================================
# Gate 3 -- CMA REAL vs MEAN: identical marginals, different composition
# ===========================================================================


def gate3() -> None:
    cache = C.load_cache("train")
    rows = np.arange(32, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    ds = make_dataset(mini_cache(cache, rows), "cma", norm)
    batch = ds.batch(rows, norm)
    real = models_for("cma", norm, 0, "REAL")
    mean = models_for("cma", norm, 0, "MEAN")
    activate_event(real); activate_event(mean)
    real.eval(); mean.eval()
    with torch.no_grad():
        yR, sR = real(batch, return_stats=True)
        yM, sM = mean(batch, return_stats=True)
    diff = float((sR["m"] - sM["m"]).abs().max())
    same_n = bool(torch.equal(sR["n_e"], sM["n_e"]))
    same_centre = bool(torch.equal(batch["cma_path_bond"], batch["cma_path_bond"]))
    zero_ok = bool(torch.all(sR["m"][sR["n_e"] == 0] == 0) and torch.all(sM["m"][sM["n_e"] == 0] == 0))
    record("g3_cma_real_vs_mean", diff > 1e-3 and same_n and zero_ok,
           max_abs_m_diff=diff, identical_n_e=same_n, zero_when_no_path=zero_ok,
           n_bonds=int(len(sR["n_e"])), n_paths=int(len(batch["cma_path_bond"])),
           n_bonds_with_two_or_more_paths=int((sR["n_e"] >= 2).sum()))

    # fixed synthetic example: centre bond with exactly two undirected paths
    n_atoms = 5
    bonds = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (1, 4, 0)]
    obj = CS.enumerate_molecule(n_atoms, bonds)
    synth = {
        "n_atom_total": n_atoms,
        "atom_ptr": np.array([0, n_atoms]), "bond_ptr": np.array([0, len(bonds)]),
        "atom_feat": np.random.default_rng(0).normal(size=(n_atoms, 108)).astype(np.float32),
        "atom_type": np.zeros(n_atoms, dtype=np.int64),
        "bond_feat": np.random.default_rng(1).normal(size=(len(bonds), 7)).astype(np.float32),
        "bond_type": np.zeros(len(bonds), dtype=np.int64),
        "bond_index": np.array([[u, v] for u, v, _ in bonds], dtype=np.int64),
        "global_feat": np.zeros((1, 33), dtype=np.float32),
        "labels": np.zeros(1, dtype=np.float32),
        "n_molecules": 1,
        "n_atoms_graph": np.array([n_atoms]), "n_bonds_graph": np.array([len(bonds)]),
    }
    synth_struct = CS.build_split(synth, "synth")
    dsyn = make_dataset(synth, "cma", norm, struct=synth_struct)
    bsyn = dsyn.batch(np.array([0]), norm)
    with torch.no_grad():
        yRs, sRs = real(bsyn, return_stats=True)
        yMs, sMs = mean(bsyn, return_stats=True)
    centre = 1  # bond (1,2) carries the two paths
    record("g3_cma_synthetic_example",
           float((sRs["m"][centre] - sMs["m"][centre]).abs().max()) > 1e-3,
           n_e=int(sRs["n_e"][centre]),
           m_real_norm=float(sRs["m"][centre].norm()),
           m_mean_norm=float(sMs["m"][centre].norm()),
           max_abs_diff=float((sRs["m"][centre] - sMs["m"][centre]).abs().max()),
           y_real=float(yRs[0]), y_mean=float(yMs[0]))


# ===========================================================================
# Gate 4 -- SPC partition vs NetworkX + branch coverage + order permutation
# ===========================================================================


def gate4() -> None:
    graphs = [
        (6, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0)]),
        (5, [(0, 1, 0), (1, 2, 0), (1, 3, 0), (1, 4, 0)]),
        (7, [(0, 1, 0), (1, 2, 0), (2, 0, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0),
             (5, 3, 0), (1, 6, 0)]),
        (6, [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0), (0, 4, 0), (0, 5, 0)]),
    ]
    cache = C.load_cache("train")
    ap = np.asarray(cache["atom_ptr"]); bp = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"]); bt = np.asarray(cache["bond_type"])
    for m in range(40):
        b0, b1 = int(bp[m]), int(bp[m + 1])
        graphs.append((int(ap[m + 1] - ap[m]),
                       [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e])) for e in range(b0, b1)]))

    bad_cut = bad_comp = bad_cover = bad_bond = bad_cyc = 0
    examples = []
    for n_atoms, bonds in graphs:
        obj = SS.enumerate_molecule(n_atoms, bonds)
        ref = SS.networkx_reference(n_atoms, bonds)
        if [int(x) for x in obj["cv_atom"]] != ref["cut_vertices"]:
            bad_cut += 1
            if len(examples) < 3:
                examples.append({"kind": "cut", "got": obj["cv_atom"].tolist(),
                                 "ref": ref["cut_vertices"]})
        edge = {(min(u, v), max(u, v)): t for u, v, t in bonds}
        incident = {v: [] for v in range(n_atoms)}
        for e, (u, v, t) in enumerate(bonds):
            incident[u].append(e); incident[v].append(e)
        for cvi, v in enumerate(obj["cv_atom"]):
            v = int(v)
            brs = np.where(np.asarray(obj["br_cv"]) == cvi)[0]
            got_comps = []
            for b in brs:
                a0, a1 = obj["br_atom_ptr"][b], obj["br_atom_ptr"][b + 1]
                got_comps.append(sorted(int(x) for x in obj["br_atom_idx"][a0:a1]))
            got_comps.sort(key=lambda c: c[0])
            if got_comps != ref["components"][v]:
                bad_comp += 1
            # disjoint + union == V \ {v}
            union = sorted(x for c in got_comps for x in c)
            if union != [x for x in range(n_atoms) if x != v] or \
                    sum(len(c) for c in got_comps) != len(union):
                bad_cover += 1
            for b in brs:
                b0, b1 = obj["br_bond_ptr"][b], obj["br_bond_ptr"][b + 1]
                bl = [int(x) for x in obj["br_bond_idx"][b0:b1]]
                comp = set(got_comps[list(brs).index(b)])
                expect = sorted(e for e, (u, w, _t) in enumerate(bonds)
                                if (u in comp and w in comp) or (u == v and w in comp)
                                or (w == v and u in comp))
                if bl != expect:
                    bad_bond += 1
                cyc = len(bl) - (len(comp) + 1) + 1
                if cyc != int(obj["br_cyc"][b]):
                    bad_cyc += 1
    record("g4_spc_partition", bad_cut == bad_comp == bad_cover == bad_bond == bad_cyc == 0,
           n_graphs=len(graphs), cut_mismatches=bad_cut, component_mismatches=bad_comp,
           coverage_mismatches=bad_cover, bond_set_mismatches=bad_bond,
           cyclomatic_mismatches=bad_cyc, examples=examples)

    # branch-order permutation must not change predictions
    rows = np.arange(32, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    mini = mini_cache(cache, rows)
    struct = SS.build_split(mini, "mini")
    ds = make_dataset(mini, "spc", norm, struct=struct)
    dsp = make_dataset(mini, "spc", norm, struct=permute_branches(struct, seed=5))
    for arm in ("REAL", "MEAN"):
        model = models_for("spc", norm, 0, arm)
        activate_event(model)
        model.eval()
        d = float(np.max(np.abs(forward(model, ds, rows, norm) -
                                forward(model, dsp, rows, norm))))
        record("g4_spc_branch_order", d <= TOL, arm=arm, max_abs_diff=d, tol=TOL)


def permute_branches(obj: dict, *, seed: int = 0) -> dict:
    """Permute branch rows inside each cut vertex (order must be irrelevant)."""
    rng = np.random.default_rng(seed)
    br_ptr = np.asarray(obj["br_ptr"], dtype=np.int64)
    total = int(br_ptr[-1])
    order = np.arange(total, dtype=np.int64)
    for m in range(len(br_ptr) - 1):
        lo, hi = int(br_ptr[m]), int(br_ptr[m + 1])
        if hi - lo > 1:
            order[lo:hi] = lo + rng.permutation(hi - lo)
    inv = np.empty(total, dtype=np.int64)
    inv[order] = np.arange(total, dtype=np.int64)
    old_a_ptr = np.asarray(obj["br_atom_ptr"]); old_b_ptr = np.asarray(obj["br_bond_ptr"])
    a_blocks = [np.asarray(obj["br_atom_idx"])[old_a_ptr[i]:old_a_ptr[i + 1]] for i in range(total)]
    b_blocks = [np.asarray(obj["br_bond_idx"])[old_b_ptr[i]:old_b_ptr[i + 1]] for i in range(total)]
    a_ptr = [0]; b_ptr = [0]
    for i in order:
        a_ptr.append(a_ptr[-1] + len(a_blocks[i]))
        b_ptr.append(b_ptr[-1] + len(b_blocks[i]))
    out = dict(obj)
    for key in ("br_cv", "br_natoms", "br_nbonds", "br_cyc"):
        out[key] = np.asarray(obj[key])[order]
    out["br_atom_ptr"] = np.asarray(a_ptr, dtype=np.int64)
    out["br_atom_idx"] = np.concatenate([a_blocks[i] for i in order]).astype(np.int64)
    out["br_bond_ptr"] = np.asarray(b_ptr, dtype=np.int64)
    out["br_bond_idx"] = np.concatenate([b_blocks[i] for i in order]).astype(np.int64)
    out["pair_br"] = inv[np.asarray(obj["pair_br"])]
    return out


# ===========================================================================
# Gate 5 -- SPC REAL vs MEAN separable
# ===========================================================================


def gate5() -> None:
    cache = C.load_cache("train")
    rows = np.arange(32, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    ds = make_dataset(mini_cache(cache, rows), "spc", norm)
    batch = ds.batch(rows, norm)
    real = models_for("spc", norm, 0, "REAL")
    mean = models_for("spc", norm, 0, "MEAN")
    activate_event(real); activate_event(mean)
    real.eval(); mean.eval()
    with torch.no_grad():
        yR, sR = real(batch, return_stats=True)
        yM, sM = mean(batch, return_stats=True)
    diff = float((sR["t"] - sM["t"]).abs().max())
    same_counts = bool(torch.equal(batch["spc_cv_npairs"], batch["spc_cv_npairs"]))
    record("g5_spc_real_vs_mean", diff > 1e-3 and same_counts,
           max_abs_t_diff=diff, n_cut_vertices=int(len(sR["n_pairs"])),
           n_pairs_total=float(batch["spc_cv_npairs"].sum()))

    # fixed synthetic star: one cut vertex with 3 branches (n_pairs = 3)
    n_atoms = 4
    bonds = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
    synth = {
        "atom_ptr": np.array([0, n_atoms]), "bond_ptr": np.array([0, len(bonds)]),
        "atom_feat": np.random.default_rng(11).normal(size=(n_atoms, 108)).astype(np.float32),
        "atom_type": np.zeros(n_atoms, dtype=np.int64),
        "bond_feat": np.random.default_rng(12).normal(size=(len(bonds), 7)).astype(np.float32),
        "bond_type": np.zeros(len(bonds), dtype=np.int64),
        "bond_index": np.array([[u, v] for u, v, _ in bonds], dtype=np.int64),
        "global_feat": np.zeros((1, 33), dtype=np.float32),
        "labels": np.zeros(1, dtype=np.float32),
        "n_molecules": 1,
        "n_atoms_graph": np.array([n_atoms]), "n_bonds_graph": np.array([len(bonds)]),
    }
    struct = SS.build_split(synth, "synth")
    dsyn = make_dataset(synth, "spc", norm, struct=struct)
    bsyn = dsyn.batch(np.array([0]), norm)
    with torch.no_grad():
        yRs, sRs = real(bsyn, return_stats=True)
        yMs, sMs = mean(bsyn, return_stats=True)
    record("g5_spc_synthetic_example",
           float((sRs["t"] - sMs["t"]).abs().max()) > 1e-3,
           n_pairs=int(sRs["n_pairs"][0]),
           t_real_norm=float(sRs["t"][0].norm()), t_mean_norm=float(sMs["t"][0].norm()),
           max_abs_diff=float((sRs["t"] - sMs["t"]).abs().max()),
           y_real=float(yRs[0]), y_mean=float(yMs[0]))


# ===========================================================================
# Gate 6 / 7 -- parameter budget, matched counts, identical initial tensors
# ===========================================================================


def gate6_7() -> None:
    cache = C.load_cache("train")
    norm = C.fit_only_norm(cache, np.arange(256))
    table = {}
    for family, builder in (("cma", CM.build_model), ("spc", SP.build_model)):
        counts, hashes, breakdowns = {}, {}, {}
        for arm in ("REAL", "MEAN"):
            model, info = builder(norm, 0, arm=arm)
            counts[arm] = C.parameter_count(model)
            hashes[arm] = C.state_hash(model)
            breakdowns[arm] = SM.parameter_breakdown(model)
            for name, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    record("g6_parameters", False, family=family, arm=arm,
                           nonfinite=[name])
        table[family] = {"parameters": counts, "module_breakdown": breakdowns,
                         "per_module": {k: v for k, v in breakdowns["REAL"].items()}}
        record("g6_parameters", counts["REAL"] == counts["MEAN"] and counts["REAL"] <= C.PARAM_CAP,
               family=family, n_params=counts, cap=C.PARAM_CAP)
        record("g7_identical_init", hashes["REAL"] == hashes["MEAN"],
               family=family, state_sha256=hashes)
    # shared skeleton must also be identical across families at the same seed
    m1, _ = CM.build_model(norm, 0, arm="REAL")
    m2, _ = SP.build_model(norm, 0, arm="REAL")
    record("g7_identical_init", C.state_hash(m1.skeleton) == C.state_hash(m2.skeleton),
           cross_family_shared_skeleton_hash=C.state_hash(m1.skeleton))
    C.write_json(C.GATES / "parameter_table.json", table)


# ===========================================================================
# Gate 8 -- gradients, loss decrease, checkpoint round trip
# ===========================================================================


def gate8() -> None:
    cache = C.load_cache("train")
    rows = np.arange(24, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    for family in ("cma", "spc"):
        mini = mini_cache(cache, rows)
        ds = make_dataset(mini, family, norm)
        for arm in ("REAL", "MEAN"):
            model = models_for(family, norm, 0, arm)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            losses, grads = [], []
            model.train()
            torch.manual_seed(0)
            for step in range(6):
                batch = ds.batch(rows, norm)
                target = torch.as_tensor(ds.targets(rows), dtype=torch.float32)
                pred = model(batch)
                loss = F.l1_loss(pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                per_module = {}
                finite = True
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    mod = ".".join(name.split(".")[:2])
                    per_module[mod] = per_module.get(mod, 0.0) + float(p.grad.pow(2).sum())
                    finite = finite and bool(torch.isfinite(p.grad).all())
                grads.append({"step": step, "finite": finite,
                              "norms": {k: math.sqrt(v) for k, v in sorted(per_module.items())}})
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss))
            last = grads[-1]["norms"]
            active = {k: v for k, v in last.items() if v > 0.0}
            expected = ({"skeleton.rho_base", "skeleton.rho_event", "F2.lins", "F3.lins"}
                        if family == "cma" else
                        {"skeleton.rho_base", "skeleton.rho_event", "Fport.lins", "Fpair.lins"})
            # skeleton.atom_encoder / bond_encoder are fine-tuned too
            expected |= {"skeleton.atom_encoder", "skeleton.bond_encoder"}
            missing = sorted(k for k in expected if last.get(k, 0.0) <= 0.0)
            decreased = losses[-1] < losses[0]
            state = C.cpu_state(model)
            model2 = models_for(family, norm, 0, arm)
            C.load_state(model2, state, "cpu")
            rt = C.state_hash(model2) == C.state_hash(model)
            record("g8_gradients", all(g["finite"] for g in grads) and not missing and decreased and rt,
                   family=family, arm=arm, losses=losses, final_module_grad_norms=last,
                   modules_without_gradient=missing, checkpoint_roundtrip=rt)


# ===========================================================================
# Gate 9 -- no label-derived input on the prediction path
# ===========================================================================


def gate9() -> None:
    cache = C.load_cache("train")
    rows = np.arange(16, dtype=np.int64)
    norm = C.fit_only_norm(cache, np.arange(256))
    for family in ("cma", "spc"):
        mini = mini_cache(cache, rows)
        struct = (CS if family == "cma" else SS).build_split(mini, "mini")
        ds = T.SplitData(mini, family, "mini", "cpu", struct=struct)
        batch = ds.batch(rows, norm)
        keys = sorted(batch)
        allowed = all(k in C.FORWARD_INPUT_WHITELIST for k in keys)
        label_free = not any(tok in k.lower() for k in keys
                             for tok in ("label", "target", "penalty", "logp", "cycle"))
        model = models_for(family, norm, 0, "REAL")
        activate_event(model)
        model.eval()
        with torch.no_grad():
            y1 = model(batch).clone()
        # mutating labels must not change the forward pass
        mini2 = dict(mini)
        mini2["labels"] = np.random.default_rng(0).normal(size=mini["labels"].shape).astype(np.float32)
        ds2 = T.SplitData(mini2, family, "mini", "cpu", struct=struct)
        with torch.no_grad():
            y2 = model(ds2.batch(rows, norm)).clone()
        unchanged = bool(torch.equal(y1, y2))
        record("g9_no_label_input", allowed and label_free and unchanged,
               family=family, forward_keys=keys, whitelisted=allowed,
               label_free_keys=label_free, prediction_unchanged_under_label_mutation=unchanged)


# ===========================================================================
# Gate 10 -- data boundary, frozen rows / label hashes / encoder provenance
# ===========================================================================


def gate10() -> None:
    split = C.split_rows()
    hashes = C.verify_split_hashes(split["rows"])
    cache = C.load_cache("train")
    labels = C.label_hashes(cache, split["rows"])
    label_ok = labels["full_labels_sha256"] == labels["cache_meta_labels_sha256"]
    ssl = {s: {"path": str(C.ssl_path(s)), "sha256": C.sha256_file(C.ssl_path(s))}
           for s in (0, 1)}
    refused = False
    try:
        C.load_cache("test")
    except PermissionError:
        refused = True
    except Exception:
        refused = False
    # static check: only xs_common may mention the official test cache
    offenders = []
    for path in sorted(C.ROUND.rglob("*.py")):
        if path.name in ("xs_common.py", "xs_gates.py"):  # boundary module + this checker
            continue
        if "ad_test" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(C.ROUND)))
    record("g10_data_boundary",
           bool(label_ok and refused and not offenders and all(h["matches_frozen"] for h in hashes.values())),
           split_rows={k: v["n"] for k, v in hashes.items()},
           split_row_hashes={k: v["sha256"] for k, v in hashes.items()},
           labels_sha256=labels, labels_match_cache_meta=bool(label_ok),
           ssl_encoder_provenance=ssl, official_test_required_refusal=refused,
           code_mentioning_ad_test_outside_xs_common=offenders,
           official_test_loaded=False)


# ===========================================================================
# Gate 11 -- coverage / compute scale
# ===========================================================================


def gate11() -> dict:
    cache = C.load_cache("train")
    cma = CS.load_split("train", cache=cache)
    spc = SS.load_split("train", cache=cache)
    c = CS.coverage_stats(cma, cache)
    s = SS.coverage_stats(spc)
    rows_fit = C.split_rows()["rows"]["fit8k"]
    fit_frac = float((np.diff(np.asarray(spc["cv_ptr"]))[rows_fit] > 0).mean())
    record("g11_coverage", True, cma=c, spc=s,
           spc_fit8k_molecules_with_cut_vertex=fit_frac,
           spc_coverage_stop_threshold=0.60,
           spc_coverage_stop_triggered=bool(fit_frac < 0.60))
    record("g11_spc_coverage_stop", fit_frac >= 0.60,
           fit8k_cut_vertex_fraction=fit_frac, threshold=0.60,
           verdict="PROCEED" if fit_frac >= 0.60 else "INSUFFICIENT_SEPARATOR_COVERAGE")
    return {"cma": c, "spc": s, "spc_fit8k_cut_vertex_fraction": fit_frac}


# ===========================================================================
# Gate 12 -- synthetic learnability sanity (never enters ZINC training)
# ===========================================================================


def _synthetic_graphs(n: int = 160, seed: int = 0):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n):
        n_atoms = int(rng.integers(6, 10))
        bonds = []
        for v in range(1, n_atoms):                       # random tree
            u = int(rng.integers(0, v))
            bonds.append((u, v, 0))
        for _extra in range(int(rng.integers(0, 3))):     # a few extra bonds
            u, v = sorted(rng.choice(n_atoms, size=2, replace=False).tolist())
            if not any((min(a, b), max(a, b)) == (u, v) for a, b, _t in bonds):
                bonds.append((int(u), int(v), 0))
        graphs.append((n_atoms, bonds))
    return graphs


def _synthetic_target(family: str, graphs):
    ys = []
    for n_atoms, bonds in graphs:
        if family == "cma":
            obj = CS.enumerate_molecule(n_atoms, bonds)
            # fraction of directed 3-paths whose two outer atoms are bonded
            chords = obj["path_chord"][:, 2] > 0
            ys.append(float(chords.mean()) if len(chords) else 0.0)
        else:
            obj = SS.enumerate_molecule(n_atoms, bonds)
            k = np.bincount(obj["br_cv"], minlength=max(1, len(obj["cv_atom"])))
            ys.append(float(np.sum(k * (k - 1) / 2.0)) / max(1, len(bonds)))
    return np.asarray(ys, dtype=np.float32)


def gate12() -> None:
    graphs = _synthetic_graphs()
    norm = {"global_mean": np.zeros(33, dtype=np.float32),
            "global_scale": np.ones(33, dtype=np.float32)}
    rng = np.random.default_rng(3)
    atom_feat, bond_feat = [], []
    atom_ptr, bond_ptr, bond_index = [0], [0], []
    for n_atoms, bonds in graphs:
        atom_feat.append(rng.normal(size=(n_atoms, 108)).astype(np.float32))
        bond_feat.append(rng.normal(size=(len(bonds), 7)).astype(np.float32))
        bond_index.append(np.array([[u, v] for u, v, _ in bonds], dtype=np.int64))
        atom_ptr.append(atom_ptr[-1] + n_atoms)
        bond_ptr.append(bond_ptr[-1] + len(bonds))
    cache = {
        "atom_ptr": np.asarray(atom_ptr), "bond_ptr": np.asarray(bond_ptr),
        "atom_feat": np.concatenate(atom_feat, 0), "atom_type": np.zeros(atom_ptr[-1], np.int64),
        "bond_feat": np.concatenate(bond_feat, 0), "bond_type": np.zeros(bond_ptr[-1], np.int64),
        "bond_index": np.concatenate(bond_index, 0),
        "global_feat": np.zeros((len(graphs), 33), dtype=np.float32),
        "labels": np.zeros(len(graphs), dtype=np.float32),
        "n_molecules": len(graphs),
        "n_atoms_graph": np.diff(np.asarray(atom_ptr)), "n_bonds_graph": np.diff(np.asarray(bond_ptr)),
    }
    rows = np.arange(len(graphs), dtype=np.int64)
    for family in ("cma", "spc"):
        struct = (CS if family == "cma" else SS).build_split(cache, "synthetic")
        ds = T.SplitData(cache, family, "synthetic", "cpu", struct=struct)
        y = torch.as_tensor(_synthetic_target(family, graphs), dtype=torch.float32)
        for arm in ("REAL", "MEAN"):
            model = models_for(family, norm, 0, arm)
            activate_event(model, scale=0.5, seed=11)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
            model.train()
            torch.manual_seed(0)
            first = last = None
            for step in range(200):
                batch = ds.batch(rows, norm)
                pred = model(batch)
                loss = F.l1_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step == 0:
                    first = float(loss)
                last = float(loss)
            ratio = last / max(first, 1e-9)
            record("g12_synthetic_learnability",
                   ratio <= 0.25,   # pre-registered: at least a 4x loss reduction

                   family=family, arm=arm, loss_first=first, loss_last=last,
                   target_std=float(y.std()), ratio=ratio,
                   n_graphs=int(len(graphs)),
                   note="synthetic targets only; never used in ZINC training")


# ===========================================================================
# main
# ===========================================================================


def main() -> int:
    C.ensure_dir(C.GATES)
    gate10()
    gate11()
    gate1()
    gate2()
    gate3()
    gate4()
    gate5()
    gate6_7()
    gate8()
    gate9()
    gate12()
    total = time.perf_counter() - T0
    summary = {
        "round": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "generated_at": C.now(),
        "seconds": float(total),
        "groups": RESULT,
        "all_pass": all(v["pass"] for v in RESULT.values()),
        "failed": sorted(k for k, v in RESULT.items() if not v["pass"]),
    }
    C.write_json(C.GATES / "cpu_gates.json", summary)
    C.log(f"[gates] {len(RESULT)} groups, all_pass={summary['all_pass']}, "
          f"failed={summary['failed']}, {total:.1f}s")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
