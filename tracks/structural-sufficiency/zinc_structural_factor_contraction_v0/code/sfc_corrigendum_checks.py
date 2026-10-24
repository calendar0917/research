"""SFC-v0 CORRIGENDUM — failing checks for Bug A (evaluation T) and Bug B (T gradient).

Run against the pre-fix snapshot *and* against the fixed modules.  The very same
file is used for both; only ``--code-dir`` changes.  Every check group must be
red on ``code/legacy_pre_fix`` and green on ``code/``.

Check groups
------------
``feature_consistency``
    The production SFC feature path (``sfc_synth._sfc_batch``) must return the
    same ``[H, T]`` numbers in training mode (grad on, trainable parameters) and
    in evaluation mode (``torch.no_grad``, detached parameters).  Also: the
    cached frozen-factor features must equal the recomputed ones, and
    ``sfc_forward(..., with_marginals=False)`` must **refuse** ``.T`` instead of
    silently returning ``logZ``.

``grad_H`` / ``grad_T`` / ``grad_downstream``
    float64 central finite differences of the *value* of ``H``, ``T`` and of a
    downstream binary-cross-entropy loss, versus the autograd gradient.  The
    finite difference is the definition of the derivative; autograd must agree.

``forest_zero_gradient``
    On an isolated node, a single-edge tree and random trees, ``T`` is
    identically zero **for every theta**, so both ``T`` and ``dT/dtheta`` must
    vanish.  The value alone is not enough: the pre-fix code has ``|T| = 0``
    but ``|dT/dtheta|`` up to 4.0.

``cycle_nondegenerate``
    On graphs with a closed cycle, ``T`` must be non-zero and its autograd
    gradient must equal the finite-difference gradient.

``srand_frozen_factors``
    For a tiny assembled split, the cached features of the frozen-random arm,
    the train-time features recomputed from the arm's own buffers, and the
    eval-time features must agree item by item.

The pre-fix failure modes are recorded in
``analysis/t0_corrigendum_prefix.json``; the post-fix pass in
``analysis/t0_corrigendum_postfix.json``.

Run:
  uv run python tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/sfc_corrigendum_checks.py \
      --code-dir tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/legacy_pre_fix \
      --out tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/analysis/t0_corrigendum_prefix.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
EXP = _HERE.parent
ANALYSIS = EXP / "analysis"

N_ATOM = 3          # small category space keeps the finite differences cheap
N_BOND = 4          # slots 0..3, slot 0 unused
EPS_FD = 1e-6
TOL_VALUE = 1e-10
TOL_GRAD_ABS = 1e-7
TOL_GRAD_REL = 1e-4
TOL_FOREST_GRAD = 1e-9
TOL_FOREST_FD = 1e-7


# ---------------------------------------------------------------------------
# module loading (by path, never by sys.path shadowing)
# ---------------------------------------------------------------------------


def load_modules(code_dir: Path):
    code_dir = code_dir.resolve()
    for name in ("sfc_core", "sfc_synth"):
        sys.modules.pop(name, None)

    def _load(name: str):
        path = code_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    core = _load("sfc_core")
    synth = _load("sfc_synth")
    if Path(core.__file__).resolve() != (code_dir / "sfc_core.py"):
        raise RuntimeError(f"loaded wrong sfc_core: {core.__file__}")
    if Path(synth.__file__).resolve() != (code_dir / "sfc_synth.py"):
        raise RuntimeError(f"loaded wrong sfc_synth: {synth.__file__}")
    return core, synth


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# graphs / parameters
# ---------------------------------------------------------------------------


def typed(core, n, atom, eu, ev, eb):
    return core.TypedGraph(n, np.asarray(atom, np.int64), np.asarray(eu, np.int64),
                           np.asarray(ev, np.int64), np.asarray(eb, np.int64))


def g_isolated(core):
    return typed(core, 1, [0], [], [], [])


def g_single_edge(core):
    return typed(core, 2, [0, 1], [0], [1], [2])


def g_path(core, n):
    atom = np.arange(n) % N_ATOM
    return typed(core, n, atom, np.arange(n - 1), np.arange(1, n),
                 (np.arange(n - 1) % 3) + 1)


def g_random_tree(core, n, seed=0):
    rng = np.random.default_rng(seed)
    atom = rng.integers(0, N_ATOM, n)
    eu = np.array([int(rng.integers(0, i)) for i in range(1, n)], np.int64)
    ev = np.arange(1, n, dtype=np.int64)
    eb = rng.integers(1, 4, n - 1)
    return typed(core, n, atom, eu, ev, eb)


def g_ring(core, n, seed=0):
    rng = np.random.default_rng(seed + 100)
    atom = rng.integers(0, N_ATOM, n)
    return typed(core, n, atom, np.arange(n), (np.arange(n) + 1) % n,
                 rng.integers(1, 4, n))


def g_square_with_pendant(core):
    # 0-1-2-3-0 square plus pendant 4 on vertex 1
    return typed(core, 5, [0, 1, 2, 0, 1],
                 [0, 1, 2, 3, 1], [1, 2, 3, 0, 4], [1, 2, 1, 3, 2])


def make_params(seed=0, C=1, scale=1.2, requires_grad=False):
    gen = torch.Generator().manual_seed(seed)
    A = (torch.randn(N_ATOM, C, 2, generator=gen, dtype=torch.float64) * scale)
    S = (torch.randn(N_BOND, C, 3, generator=gen, dtype=torch.float64) * scale)
    return A.requires_grad_(requires_grad), S.requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# finite differences of a scalar function of leaf tensors
# ---------------------------------------------------------------------------


def fd_grad(fn, tensors):
    """Central differences of ``fn()`` w.r.t. every entry of every tensor."""
    out = []
    for t in tensors:
        num = torch.zeros_like(t)
        for idx in np.ndindex(*t.shape):
            old = float(t[idx])
            with torch.no_grad():
                t[idx] = old + EPS_FD
            fp = float(fn())
            with torch.no_grad():
                t[idx] = old - EPS_FD
            fm = float(fn())
            with torch.no_grad():
                t[idx] = old
            num[idx] = (fp - fm) / (2 * EPS_FD)
        out.append(num)
    return out


def grad_metrics(ana, num):
    ana = torch.as_tensor(ana, dtype=torch.float64)
    num = torch.as_tensor(num, dtype=torch.float64)
    diff = (ana - num).abs()
    allowed = TOL_GRAD_ABS + TOL_GRAD_REL * num.abs()
    return {
        "max_abs_err": float(diff.max()) if diff.numel() else 0.0,
        "max_rel_err": float((diff / num.abs().clamp_min(1e-12)).max())
        if diff.numel() else 0.0,
        "global_rel_err": float(diff.max() / num.abs().max().clamp_min(1e-12))
        if diff.numel() else 0.0,
        "max_abs_grad": float(num.abs().max()) if num.numel() else 0.0,
        "n_entries": int(num.numel()),
        "pass": bool(diff.numel() == 0 or bool((diff <= allowed).all())),
    }


def autograd_of(fn, tensors):
    with torch.enable_grad():
        val = fn()
        grads = torch.autograd.grad(val, tensors, allow_unused=True)
    out = []
    for t, gr in zip(tensors, grads):
        out.append(torch.zeros_like(t) if gr is None else gr.detach().clone())
    return out


# ---------------------------------------------------------------------------
# check 1 — train/eval feature consistency + explicit T refusal + Srand cache
# ---------------------------------------------------------------------------


def check_feature_consistency(core, synth, K=2) -> dict:
    graphs = [g_ring(core, 4, seed=1), g_ring(core, 5, seed=2),
              g_random_tree(core, 6, seed=3), g_single_edge(core), g_isolated(core)]
    A, S = make_params(seed=11, C=K, scale=1.2)
    A_req = A.clone().requires_grad_(True)
    S_req = S.clone().requires_grad_(True)

    with torch.enable_grad():
        train_feat = synth._sfc_batch(graphs, A_req, S_req, need_grad=True)
    with torch.no_grad():
        eval_feat = synth._sfc_batch(graphs, A.detach(), S.detach(), need_grad=False)
    d_te = float((train_feat.detach() - eval_feat).abs().max())

    # the cache path (used for the frozen-random arm) must agree as well
    with torch.no_grad():
        cache_feat = torch.stack([
            torch.as_tensor(synth.sfc_features(g, A, S)) for g in graphs])
    d_tc = float((train_feat.detach() - cache_feat).abs().max())
    d_ce = float((cache_feat - eval_feat).abs().max())

    # explicit contract: without exact marginals, T must be refused, H must work
    refusal = {}
    for name, g in (("ring4", graphs[0]), ("tree6", graphs[2])):
        out = core.sfc_forward(g, A, S, with_marginals=False)
        h = out.H.detach().clone()
        try:
            _ = out.T
            refused = False
            err = None
        except (RuntimeError, ValueError) as exc:      # explicit refusal
            refused = True
            err = f"{type(exc).__name__}: {exc}"[:200]
        refusal[name] = {"refused_T": refused, "error": err,
                         "H_available": bool(torch.isfinite(h).all())}

    ok = (d_te == 0.0 and d_tc == 0.0 and d_ce == 0.0
          and all(v["refused_T"] and v["H_available"] for v in refusal.values()))

    # evaluation runs under torch.no_grad; the exact marginals must still be
    # computed (local enable_grad) and detached, and must match dense enumeration
    ngrad = {}
    for name, g in (("ring4", graphs[0]), ("ring5", graphs[1])):
        try:
            with torch.no_grad():
                o = core.sfc_forward(g, A, S, with_marginals=True)
            _, node_ref, edge_ref = core.marginals_dense(g, A, S)
            ngrad[name] = {
                "node_max_abs_err": float((o.node_p - node_ref).abs().max()),
                "edge_max_abs_err": float((o.edge_p - edge_ref).abs().max()),
                "T_finite": bool(torch.isfinite(o.T).all()),
                "error": None,
            }
            ngrad[name]["pass"] = bool(
                ngrad[name]["node_max_abs_err"] <= 1e-10
                and ngrad[name]["edge_max_abs_err"] <= 1e-10
                and ngrad[name]["T_finite"] and not o.T.requires_grad)
        except Exception as exc:                     # pre-fix: autograd disabled
            ngrad[name] = {"pass": False,
                           "error": f"{type(exc).__name__}: {exc}"[:200]}
    ok = ok and all(v["pass"] for v in ngrad.values())

    return {
        "train_vs_eval_max_abs_diff": d_te,
        "train_vs_cache_max_abs_diff": d_tc,
        "cache_vs_eval_max_abs_diff": d_ce,
        "T_refusal": refusal,
        "no_grad_marginals": ngrad,
        "pass": bool(ok),
        "interpretation": ("training and evaluation must use the same [H, T] "
                           "numbers; T must never be silently replaced by logZ"),
    }


# ---------------------------------------------------------------------------
# check 2 — H / T / downstream-loss gradient vs central finite differences
# ---------------------------------------------------------------------------


def _downstream_loss(features, w, b, y):
    logits = features @ w + b
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def check_gradients(core: dict) -> dict:
    g = g_ring(core, 4, seed=7)
    A, S = make_params(seed=21, C=1, scale=1.2, requires_grad=True)

    def feats():
        with torch.enable_grad():
            out = core.sfc_forward(g, A, S, with_marginals=True)
            return torch.cat([out.H, out.T])

    def f_H():
        return feats()[0]

    def f_T():
        return feats()[1]

    w = torch.tensor([0.7, -1.1], dtype=torch.float64)
    b = torch.tensor(0.2, dtype=torch.float64)
    y = torch.tensor(1.0, dtype=torch.float64)

    def f_down():
        return _downstream_loss(feats(), w, b, y)

    results = {}
    for name, fn in (("H", f_H), ("T", f_T), ("downstream_BCE", f_down)):
        ana_A, ana_S = autograd_of(fn, [A, S])
        num_A, num_S = fd_grad(fn, [A, S])
        mA = grad_metrics(ana_A, num_A)
        mS = grad_metrics(ana_S, num_S)
        results[name] = {
            "A": mA, "S": mS,
            "pass": bool(mA["pass"] and mS["pass"]),
            "value": float(fn()),
        }
        # restore leaves (fd_grad already does, but be explicit)
        with torch.no_grad():
            pass

    # multi-channel path: the replay trains K=8 channels, so verify T there too
    A8, S8 = make_params(seed=22, C=8, scale=1.2, requires_grad=True)

    def f_T8():
        with torch.enable_grad():
            return core.sfc_forward(g, A8, S8, with_marginals=True).T.sum()

    ana_A, ana_S = autograd_of(f_T8, [A8, S8])
    num_A, num_S = fd_grad(f_T8, [A8, S8])
    mA = grad_metrics(ana_A, num_A)
    mS = grad_metrics(ana_S, num_S)
    results["T_C8_ring4"] = {"A": mA, "S": mS,
                            "pass": bool(mA["pass"] and mS["pass"]),
                            "value": float(f_T8())}
    results["all_pass"] = bool(all(results[k]["pass"] for k in
                                    ("H", "T", "downstream_BCE", "T_C8_ring4")))
    results["pass"] = results["all_pass"]
    results["note"] = ("H is the control (correct before and after); T and the "
                       "downstream loss are the corrigendum targets.")
    return results


# ---------------------------------------------------------------------------
# check 3 — forest: T == 0 AND dT/dtheta == 0
# ---------------------------------------------------------------------------


def check_forest_zero_gradient(core, seed=31) -> dict:
    graphs = {
        "isolated_node": g_isolated(core),
        "single_edge_tree": g_single_edge(core),
        "path_6": g_path(core, 6),
        "random_tree_9": g_random_tree(core, 9, seed=5),
        "forest_2_components": typed(
            core, 7, [0, 1, 2, 0, 1, 2, 0],
            [0, 2, 3, 4, 5], [1, 3, 4, 5, 6], [1, 2, 1, 3, 2]),
    }
    rows = {}
    for i, (name, g) in enumerate(graphs.items()):
        assert g.is_forest(), name
        A, S = make_params(seed=seed + i, C=1, scale=1.2, requires_grad=True)
        with torch.enable_grad():
            T = core.sfc_forward(g, A, S, with_marginals=True).T.sum()
        gA, gS = torch.autograd.grad(T, [A, S], allow_unused=True)
        gA = torch.zeros_like(A) if gA is None else gA
        gS = torch.zeros_like(S) if gS is None else gS
        try:
            num_A, num_S = fd_grad(
                lambda: core.sfc_forward(g, A, S, with_marginals=True).T.sum(),
                [A, S])
            fd_max = float(max(num_A.abs().max(), num_S.abs().max()))
        except Exception as exc:                        # pragma: no cover
            fd_max = float("nan")
            fd_err = str(exc)[:200]
        rows[name] = {
            "abs_T": float(T.abs()),
            "max_abs_grad_autograd": float(max(gA.abs().max(), gS.abs().max())),
            "max_abs_grad_fd": fd_max,
            "pass": bool(float(T.abs()) <= TOL_VALUE
                         and float(max(gA.abs().max(), gS.abs().max())) <= TOL_FOREST_GRAD
                         and fd_max <= TOL_FOREST_FD),
        }
    return {"per_graph": rows, "pass": bool(all(v["pass"] for v in rows.values())),
            "note": ("T is identically zero on every forest, so the gradient must "
                     "vanish too; checking only the value misses Bug B.")}


# ---------------------------------------------------------------------------
# check 4 — cycle: T != 0 and dT/dtheta matches finite differences
# ---------------------------------------------------------------------------


def check_cycle_nondegenerate(core) -> dict:
    rows = {}
    for i, (name, g) in enumerate((("ring_4", g_ring(core, 4, seed=11)),
                                   ("ring_5", g_ring(core, 5, seed=12)),
                                   ("square_pendant", g_square_with_pendant(core)))):
        assert not g.is_forest(), name
        A, S = make_params(seed=41 + i, C=1, scale=1.2, requires_grad=True)

        def f():
            return core.sfc_forward(g, A, S, with_marginals=True).T.sum()

        with torch.enable_grad():
            val = float(f())
        ana_A, ana_S = autograd_of(f, [A, S])
        num_A, num_S = fd_grad(f, [A, S])
        mA = grad_metrics(ana_A, num_A)
        mS = grad_metrics(ana_S, num_S)
        rows[name] = {
            "cyclic_number": int(g.cyclomatic()),
            "T_value": val,
            "A": mA, "S": mS,
            "pass": bool(abs(val) > 1e-8 and mA["pass"] and mS["pass"]),
        }
    return {"per_graph": rows, "pass": bool(all(v["pass"] for v in rows.values())),
            "note": "the cycle residual must be non-degenerate and correctly differentiable"}


# ---------------------------------------------------------------------------
# check 5 — Srand: cached == train-time == eval-time, from its own buffers
# ---------------------------------------------------------------------------


def _clear_plan_cache(synth) -> None:
    """The pre-fix ``_plan_for`` caches plans by ``id(graph)``; CPython reuses ids.

    Freeing graphs from an earlier check can make a *different* graph hit a stale
    plan (crash or wrong values), which would hide the defect under test.  Clear
    the cache so the legacy run reaches the actual check.
    """
    cache = getattr(synth, "_PLAN_CACHE", None)
    if cache is not None:
        cache.clear()


def check_srand_frozen_factors(core, synth, task="reach") -> dict:
    _clear_plan_cache(synth)
    split = synth.assemble(task, n_seen=24, n_unseen=12, seed=0)
    plan = split["sfc_plan"]
    graphs = split["graphs_train"]
    n = len(graphs)
    cache = torch.as_tensor(split["sfc_train"])

    A = torch.as_tensor(plan["A"], dtype=torch.float64)
    S = torch.as_tensor(plan["S"], dtype=torch.float64)

    # the actual frozen buffers the Srand arm would carry
    model = synth.Arm("Srand", comp_dim=int(split["comp_train"].shape[1]),
                      hidden=8, sfc_plan=plan)
    bufA = model.A.detach().to(torch.float64)
    bufS = model.S.detach().to(torch.float64)

    with torch.enable_grad():
        train_feat = synth._sfc_batch(graphs, bufA.clone().requires_grad_(True),
                                      bufS.clone().requires_grad_(True), need_grad=True)
    with torch.no_grad():
        eval_feat = synth._sfc_batch(graphs, bufA, bufS, need_grad=False)
    with torch.no_grad():
        plan_feat = torch.stack([
            torch.as_tensor(synth.sfc_features(g, A, S)) for g in graphs])

    def cmp(x, y):
        d = (x.detach() - y.detach()).abs()
        return {"max_abs_diff": float(d.max()) if d.numel() else 0.0,
                "bit_identical": bool(torch.equal(x.detach(), y.detach()))}

    r = {
        "task": task,
        "n_graphs": n,
        "cache_vs_train": cmp(cache[:n], train_feat),
        "cache_vs_eval": cmp(cache[:n], eval_feat),
        "train_vs_eval": cmp(train_feat, eval_feat),
        "cache_vs_plan_params": cmp(cache[:n], plan_feat),
        "buffer_matches_plan": {
            "A": bool(torch.equal(bufA, A)), "S": bool(torch.equal(bufS, S))},
    }
    r["pass"] = bool(
        r["cache_vs_train"]["bit_identical"]
        and r["cache_vs_eval"]["bit_identical"]
        and r["train_vs_eval"]["bit_identical"]
        and r["cache_vs_plan_params"]["bit_identical"]
        and r["buffer_matches_plan"]["A"] and r["buffer_matches_plan"]["S"])
    return r


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _fingerprint(plan) -> str:
    h = hashlib.sha256()
    for key in ("A", "S"):
        t = plan[key].detach().to(torch.float64).contiguous()
        h.update(np.asarray(t.numpy()).tobytes())
    return h.hexdigest()


def check_shared_plan_immutability(core, synth) -> dict:
    """Auxiliary code must not mutate the frozen factors shared by every arm.

    ``torch.as_tensor`` on a float64 tensor returns the *same* object, so an
    in-place ``requires_grad_()`` or an optimiser step inside the fit diagnostic
    rewrites ``split["sfc_plan"]`` and with it the Srand cache.  This check runs
    the diagnostic and then verifies (a) the plan fingerprint is unchanged, and
    (b) the arm built before/after the diagnostic is bit-identical.
    """
    _clear_plan_cache(synth)
    split = synth.assemble("reach", n_seen=64, n_unseen=16, seed=0)
    fp0 = _fingerprint(split["sfc_plan"])

    def snapshot():
        torch.manual_seed(0)
        comp_dim = int(split["comp_train"].shape[1])
        hidden, _npar, plan = synth.fit_hidden("S", comp_dim, split["sfc_plan"], 0)
        m = synth.Arm("S", comp_dim, hidden, plan)
        flat = torch.cat([p.detach().flatten() for p in m.parameters()])
        with torch.no_grad():
            feats = synth._sfc_batch(split["graphs_train"][:8],
                                     m.A.detach().to(torch.float64),
                                     m.S.detach().to(torch.float64), False)
        return hidden, flat, feats

    h0, p0, f0 = snapshot()
    diag = synth.sfc_fit_diagnostic("reach", split, steps=2, n_graphs=8)
    h1, p1, f1 = snapshot()
    fp1 = _fingerprint(split["sfc_plan"])
    out = {
        "plan_fingerprint_before": fp0,
        "plan_fingerprint_after": fp1,
        "fingerprint_stable": bool(fp0 == fp1),
        "diagnostic_reports_mutation": bool(diag.get("plan_mutated", True)),
        "hidden_width_stable": bool(h0 == h1),
        "init_params_bit_identical": bool(torch.equal(p0, p1)),
        "features_bit_identical": bool(torch.equal(f0, f1)),
    }
    out["pass"] = bool(out["fingerprint_stable"] and not out["diagnostic_reports_mutation"]
                       and out["hidden_width_stable"]
                       and out["init_params_bit_identical"]
                       and out["features_bit_identical"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", default=str(_HERE))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    code_dir = Path(args.code_dir).resolve()
    torch.set_num_threads(4)

    t0 = time.perf_counter()
    core, synth = load_modules(code_dir)

    checks = {}
    checks["feature_consistency"] = check_feature_consistency(core, synth)
    checks["gradients"] = check_gradients(core)
    checks["forest_zero_gradient"] = check_forest_zero_gradient(core)
    checks["cycle_nondegenerate"] = check_cycle_nondegenerate(core)
    checks["srand_frozen_factors"] = check_srand_frozen_factors(core, synth)
    checks["shared_plan_immutability"] = check_shared_plan_immutability(core, synth)

    n_pass = sum(1 for v in checks.values() if v["pass"])
    payload = {
        "protocol": "sfc-v0-cpu-audit",
        "stage": "T0-corrigendum-checks",
        "code_dir": str(code_dir),
        "module_sha256": {
            "sfc_core.py": sha256(code_dir / "sfc_core.py"),
            "sfc_synth.py": sha256(code_dir / "sfc_synth.py"),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "threads": int(torch.get_num_threads()),
            "dtype": "float64",
        },
        "tolerances": {
            "fd_eps": EPS_FD,
            "value": TOL_VALUE,
            "grad_abs": TOL_GRAD_ABS,
            "grad_rel": TOL_GRAD_REL,
            "forest_grad": TOL_FOREST_GRAD,
            "forest_fd": TOL_FOREST_FD,
        },
        "checks": checks,
        "n_pass": n_pass,
        "n_total": len(checks),
        "all_pass": bool(n_pass == len(checks)),
        "seconds": time.perf_counter() - t0,
    }
    out = Path(args.out) if args.out else ANALYSIS / "t0_corrigendum_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"pass": v["pass"]} for k, v in checks.items()}, indent=2))
    print(f"{n_pass}/{len(checks)} check groups pass -> {out} "
          f"({payload['seconds']:.1f}s)")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
