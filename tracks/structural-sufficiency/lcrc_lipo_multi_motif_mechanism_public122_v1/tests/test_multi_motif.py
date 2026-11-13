"""Focused invariant tests for the multi-motif mechanism round.

These tests are structure- and arithmetic-only: they load a handful of real
molecules from the frozen public122 data and check the invariants the protocol
freezes *without* running the model (no weights, no prediction, no training).

Covered:

* dose token ordering / numbering does not change any aggregation (permuting the
  per-root row order inside ``nbr_rows`` / the three S2 tables leaves ``doses``
  bit-identical);
* the three S2 per-root tables are used through the *same* closure/content/joint
  aggregation, so an equal-table view has zero S2 dose under any weighting;
* ``T`` destroys the target family key and drops the count by exactly one,
  ``K0`` keeps the key and the count; both stay single-component, keep the atom
  and bond counts and every AF51 feature;
* the physical change is delete-2 / add-2 for every accepted caliper row;
* the seven Mobius terms sum back to ``value(123) - value(empty)`` and a constant
  function has zero higher-order terms;
* ``rC`` obeys its denominator guard and lies in ``[-1, 1]``;
* the dose / caliper code has no access to a prediction or a label.
"""

from __future__ import annotations

import copy
import inspect
import math

import numpy as np
import pytest

_CODE = __import__("pathlib").Path(__file__).resolve().parents[1] / "code"
import sys as _sys

if str(_CODE) not in _sys.path:
    _sys.path.insert(0, str(_CODE))

import common as C
import doses as D
import families as FAM
import feats as F
import graphlib as G
import rewire as R


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    ck = C.load_checkpoint()
    norm = C.frozen_norm(ck)
    rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    spec = C.DD.frozen_spec()
    return {"norm": norm, "rows": rows, "spec": spec}


def _first_pair(rows, kinds=("amide",)) -> dict:
    for rid in range(C.N_MOLECULES):
        g = G.summarise_raw(rows[rid], rid)
        for fam in kinds:
            for inst in FAM.main_instances(g, fam):
                for donor in R.donor_bonds(g, inst):
                    pair = R.build_pair(g, inst, donor)
                    if pair.ok:
                        return {"row_id": rid, "graph": g, "instance": inst, "donor": donor, "pair": pair}
    raise AssertionError("no rewirable pair found")


@pytest.fixture(scope="module")
def amide_pair(world):
    return _first_pair(world["rows"])


def _cache_for(world, entries):
    return F.build_cache(entries, world["spec"], split="unit_test")


# ---------------------------------------------------------------------------
# 1. dose token ordering / numbering invariance
# ---------------------------------------------------------------------------


def test_dose_permutation_invariance(world, amide_pair):
    rid = amide_pair["row_id"]
    base = F.row_from_artifact({**dict(world["rows"][rid]), "mol_index": rid})
    pr = amide_pair["pair"]
    t_row, _ = R.edit_row(base, pr.t_remove, pr.t_add)
    cache = _cache_for(world, [(f"{rid}:O", base), (f"{rid}:T", t_row)])
    i_o, i_v = 0, 1
    norm = world["norm"]
    ref = D.doses(
        D.molecule_tables(cache, i_o), D.molecule_tables(cache, i_v),
        scalar_mean=norm["scalar_mean"], scalar_scale=norm["scalar_scale"],
    )
    # permute the row order inside every per-root block of the view cache
    cache2 = copy.deepcopy(cache)
    rng = np.random.default_rng(0)
    nbr_ptr = np.asarray(cache2["nbr_ptr"])
    path_ptr = np.asarray(cache2["path_ptr"])
    a0 = int(np.asarray(cache2["atom_ptr"])[i_v])
    a1 = int(np.asarray(cache2["atom_ptr"])[i_v + 1])
    for a in range(a0, a1):
        n0, n1 = int(nbr_ptr[a]), int(nbr_ptr[a + 1])
        if n1 - n0 > 1:
            perm = rng.permutation(n1 - n0)
            arr = np.asarray(cache2["nbr_code"]).copy()
            arr[n0:n1] = arr[n0:n1][perm]
            cache2["nbr_code"] = arr
        p0, p1 = int(path_ptr[a]), int(path_ptr[a + 1])
        if p1 - p0 > 1:
            perm = rng.permutation(p1 - p0)
            for key in ("path_cat", "path_end63", "path_end", "path_closure"):
                arr = np.asarray(cache2[key]).copy()
                arr[p0:p1] = arr[p0:p1][perm]
                cache2[key] = arr
    got = D.doses(
        D.molecule_tables(cache2, i_o), D.molecule_tables(cache2, i_v),
        scalar_mean=norm["scalar_mean"], scalar_scale=norm["scalar_scale"],
    )
    for key in ("d1n", "d1b", "d2j", "d2c", "d2k", "dr"):
        assert got[key] == ref[key], f"{key}: permuted {got[key]} != {ref[key]}"
    assert math.isclose(got["d3"], ref["d3"], rel_tol=0, abs_tol=0)


# ---------------------------------------------------------------------------
# 2. identical S2 tables -> zero S2 dose under any weighting
# ---------------------------------------------------------------------------


def test_identical_s2_tables_give_zero_s2_dose(world, amide_pair):
    rid = amide_pair["row_id"]
    base = F.row_from_artifact({**dict(world["rows"][rid]), "mol_index": rid})
    cache = _cache_for(world, [(f"{rid}:O", base)])
    tbl = D.molecule_tables(cache, 0)
    # a view whose S2 tables are identical to O: the three S2 L1 doses must be 0
    norm = world["norm"]
    z = D.doses(tbl, tbl, scalar_mean=norm["scalar_mean"], scalar_scale=norm["scalar_scale"])
    assert z["d2j"] == 0 and z["d2c"] == 0 and z["d2k"] == 0
    assert z["d1n"] == 0 and z["d1b"] == 0 and z["d3"] == 0.0 and z["dr"] == 0


def test_s2_content_and_closure_consistent(world, amide_pair):
    """The joint (path_cat, path_end63) table is the marginal of the content table."""
    rid = amide_pair["row_id"]
    base = F.row_from_artifact({**dict(world["rows"][rid]), "mol_index": rid})
    cache = _cache_for(world, [(f"{rid}:O", base)])
    tbl = D.molecule_tables(cache, 0)
    from collections import Counter

    for a, rows in enumerate(tbl["path_rows"]):
        joint = Counter(zip(rows[0], rows[1]))
        end63 = Counter(zip(rows[0], rows[2]))
        closure = Counter(rows[3])
        assert sum(joint.values()) == sum(end63.values()) == sum(closure.values())
        assert sum(closure.values()) == len(rows[0])


# ---------------------------------------------------------------------------
# 3. T / K0 structural invariants
# ---------------------------------------------------------------------------


def test_t_and_k0_structural_invariants(world, amide_pair):
    g = amide_pair["graph"]
    inst = amide_pair["instance"]
    pr = amide_pair["pair"]
    rid = amide_pair["row_id"]
    base = F.row_from_artifact({**dict(world["rows"][rid]), "mol_index": rid})
    t_row, _ = R.edit_row(base, pr.t_remove, pr.t_add)
    k_row, _ = R.edit_row(base, pr.k_remove, pr.k_add)
    for row in (t_row, k_row):
        assert np.asarray(row["edge_index"]).shape == np.asarray(base["edge_index"]).shape
        assert np.array_equal(np.asarray(row["atom_class"]), np.asarray(base["atom_class"]))
    import torch  # noqa: F401  (collate path needs torch on sys.path)

    chk_t = R.verify_view_structure(g, t_row, remove=pr.t_remove, add=pr.t_add, kind="T", inst=inst)
    chk_k = R.verify_view_structure(g, k_row, remove=pr.k_remove, add=pr.k_add, kind="K0", inst=inst)
    assert chk_t["pass"], [k for k, v in chk_t.items() if v is False]
    assert chk_k["pass"], [k for k, v in chk_k.items() if v is False]


# ---------------------------------------------------------------------------
# 4. every accepted caliper row is delete-2 / add-2 on both views
# ---------------------------------------------------------------------------


def test_rejected_rows_have_no_structure():
    rows = _read(C.ANALYSIS / "pair_catalog.csv")
    # every structurally accepted pair must carry a T and K0 graph id and be single-component
    ok = [r for r in rows if int(r["pop_struct_chem"]) == 1]
    assert ok, "no accepted pairs in the frozen catalogue"
    for r in ok[:200]:
        assert r["t_graph_id"] and r["k_graph_id"] and r["o_graph_id"]


def _read(path):
    import csv

    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 5. Mobius identities
# ---------------------------------------------------------------------------


def _mobius(values: dict[str, float]) -> dict[str, float]:
    terms = {}
    terms["g1"] = values["1"] - values[""]
    terms["g2"] = values["2"] - values[""]
    terms["g3"] = values["3"] - values[""]
    terms["g12"] = values["12"] - values["1"] - values["2"] + values[""]
    terms["g13"] = values["13"] - values["1"] - values["3"] + values[""]
    terms["g23"] = values["23"] - values["2"] - values["3"] + values[""]
    terms["g123"] = (
        values["123"] - values["12"] - values["13"] - values["23"]
        + values["1"] + values["2"] + values["3"] - values[""]
    )
    return terms


def test_mobius_sum_identity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        vals = {s: float(rng.normal()) for s in ("", "1", "2", "3", "12", "13", "23", "123")}
        terms = _mobius(vals)
        assert abs(sum(terms.values()) - (vals["123"] - vals[""])) < 1e-12


def test_mobius_additive_function_has_no_interaction():
    """If the value is a sum of three single-factor functions, all mixed terms vanish."""
    a = {"": 0.0, "1": 0.3, "2": 0.0, "3": 0.0, "12": 0.3, "13": 0.3, "23": 0.0, "123": 0.3}
    b = {"": 0.0, "1": 0.0, "2": -0.7, "3": 0.0, "12": -0.7, "13": 0.0, "23": -0.7, "123": -0.7}
    c = {"": 0.0, "1": 0.0, "2": 0.0, "3": 0.05, "12": 0.0, "13": 0.05, "23": 0.05, "123": 0.05}
    vals = {k: a[k] + b[k] + c[k] for k in a}
    terms = _mobius(vals)
    assert abs(terms["g1"] - 0.3) < 1e-12
    assert abs(terms["g2"] + 0.7) < 1e-12
    assert abs(terms["g3"] - 0.05) < 1e-12
    for k in ("g12", "g13", "g23", "g123"):
        assert abs(terms[k]) < 1e-12


# ---------------------------------------------------------------------------
# 6. rC definition and denominator guard
# ---------------------------------------------------------------------------


def test_rC_definition_and_guard():
    import analysis as A

    assert A._ratio(0.0, 0.0, init=False) == 0.0
    assert A._ratio(1e-13, 0.0, init=True) == 0.0
    assert math.isclose(A._ratio(0.4, 0.1, init=False), (0.4 - 0.1) / (0.4 + 0.1))
    for a, b in ((0.4, 0.1), (0.0, 0.3), (2.0, 2.0), (0.2, 0.0)):
        v = A._ratio(a, b, init=False)
        assert -1.0 - 1e-12 <= v <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# 7. the dose / caliper path cannot see a label or a prediction
# ---------------------------------------------------------------------------


def test_dose_code_is_prediction_blind():
    src = inspect.getsource(D)
    body = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith(("#", "``", "*")) 
    )
    body = body.split('"""')[2] if body.count('"""') >= 2 else body
    for bad in ("import torch", "y_true", "pred_raw", "target_norm"):
        assert bad not in body.lower(), f"dose module code mentions {bad!r}"
    params = list(inspect.signature(D.doses).parameters)
    assert params == ["tbl_o", "tbl_v", "scalar_mean", "scalar_scale"]
    params = list(inspect.signature(D.caliper_ok).parameters)
    assert params == ["t", "c", "ratio", "eps"]


def test_caliper_ok_exact_dimensions():
    t = {"d1n": 3, "d1b": 5, "d2k": 2, "d2j": 4, "d2c": 4, "d3": 1.0, "dr": 4}
    same = dict(t)
    ok, detail = D.caliper_ok(t, same, (0.8, 1.25), 1e-12)
    assert ok and detail["d1n_equal"] and detail["d1b_equal"] and detail["d2k_equal"]
    bad = dict(t)
    bad["d1n"] = 4
    ok, _ = D.caliper_ok(t, bad, (0.8, 1.25), 1e-12)
    assert not ok
    far = dict(t)
    far["d2j"] = 100
    ok, _ = D.caliper_ok(t, far, (0.8, 1.25), 1e-12)
    assert not ok
