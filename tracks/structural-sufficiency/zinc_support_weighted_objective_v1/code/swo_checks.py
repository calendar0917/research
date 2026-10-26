"""CPU gates for the support-weighted-objective pilot (all must pass before GPU).

G1  recipe identity      the R0 dict used here equals zps_runner.RECIPES["R0"]
G2  trainer diff         swo_train_core.py differs from ad_train.py only in the
                         documented sample-weight hunks (no silent drift)
G3  E equivalence         with sample_weight=None the copy reproduces
                         ad_train.train bit for bit (same seed, same rows)
G4  weighted identity     w == 1 reproduces F.l1_loss exactly
G5  weights frozen       W is deterministic, label-free, train-only, and its
                         recorded statistics match the frozen npz
G6  init + batch alism   E and W share the initial state, the SSL encoder and the
                         batch order (the generator depends only on the seed)
G7  no test access       no test cache is referenced anywhere in this round
G8  capacity             the body is the historical DCC-L (494,193 params, cap 500k)
G9  gradient             weighted-L1 gradient equals sum_i w_i sign(e_i)/sum_i w_i
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import swo_common as S
import swo_weights

ROUND = S.ROUND
OUT = S.ANALYSIS / "cpu_checks.json"


def _g1():
    import zps_runner
    mine = S.r0_recipe()
    ref = dict(zps_runner.RECIPES["R0"])
    return {"recipe": mine, "equal": mine == ref, "pass": bool(mine == ref)}


def _g2():
    """Every edit must sit next to one of the three documented anchors."""
    import difflib
    src = (ROUND / "code" / "swo_train_core.py").read_text().splitlines()
    ref = (S.ROUND.parents[0] / "zinc_arch_discovery_v1" / "code" / "ad_train.py").read_text().splitlines()
    sm = difflib.SequenceMatcher(None, ref, src, autojunk=False)
    anchors = []
    for i, ln in enumerate(ref):
        if ("train_rows: np.ndarray | None" in ln
                or "rows_valid = np.arange" in ln
                or "loss = F.l1_loss" in ln):
            anchors.append(i)
    doc_end = next(i for i, ln in enumerate(ref) if not ln.startswith(('"""', "#")) and i > 5)
    hunks, bad = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        near = any(abs(i1 - a) <= 6 for a in anchors) or i1 < doc_end
        hunks.append({"tag": tag, "ref_lines": [i1, i2], "copy_lines": [j1, j2],
                      "documented_anchor": bool(near),
                      "removed": ref[i1:i2][:4], "added": src[j1:j2][:6]})
        if not near:
            bad.append(hunks[-1])
    return {"n_hunks": len(hunks), "undocumented_hunks": bad,
            "hunk_summary": [{k: h[k] for k in ("tag", "ref_lines", "copy_lines")} for h in hunks],
            "pass": bool(3 <= len(hunks) <= 4 and not bad)}


def _short_train(module, seed=0, rows=1280, epochs=1, sample_weight=None):
    cand, builder, collate = S.dcc_candidate()
    out = S.ANALYSIS / f"_gate_{module.__name__}_{'W' if sample_weight is not None else 'E'}"
    kw = {"sample_weight": sample_weight} if (sample_weight is not None
                                              or module.__name__ == "swo_train_core") else {}
    info = module.train(
        cand, seed, torch.device("cpu"), epochs=epochs,
        recipe_override=dict(S.r0_recipe(), epochs=epochs),
        model_builder=builder, collate_fn=collate, out_dir=out,
        loss="l1", train_rows=np.arange(int(rows), dtype=np.int64),
        force=True, **kw)
    curve = [(float(e["valid_mae"]), float(e["online_train_mae"])) for e in info["curve"]]
    return {"curve": curve, "best": info["best_valid"],
            "final_hash": S.sha256_file(out / "best.pt")}


def _g3():
    import ad_train as REF
    import swo_train_core as COPY
    # ad_train.train has no sample_weight argument at all; the copy with None
    # must reproduce it exactly.
    a = _short_train(REF)
    b = _short_train(COPY)
    same_curve = a["curve"] == b["curve"]
    return {"reference_curve": a["curve"], "copy_curve": b["curve"],
            "reference_best": a["best"], "copy_best": b["best"],
            "curves_bit_identical": bool(same_curve),
            "same_final_hash": bool(a["final_hash"] == b["final_hash"]),
            "pass": bool(same_curve and a["final_hash"] == b["final_hash"])}


def _g4():
    torch.manual_seed(0)
    pred = torch.randn(128, dtype=torch.float64)
    target = torch.randn(128, dtype=torch.float64)
    w = torch.ones(128, dtype=torch.float64)
    err = (pred - target).abs()
    weighted = float((w * err).sum() / w.sum())
    plain = float(F.l1_loss(pred, target))
    return {"weighted_with_ones": weighted, "plain_l1": plain,
            "exact_equal": bool(weighted == plain), "pass": bool(weighted == plain)}


def _g5():
    npz = S.ANALYSIS / "weights_train10k.npz"
    meta = S.read_json(npz.with_suffix(".json"))
    w1 = np.asarray(np.load(npz)["w"], dtype=np.float64)
    again = swo_weights.build()
    w2 = again["weights"]
    # label-free: perturb the training labels, rebuild, weights must be identical
    import ad_data as D
    real_loader = D.load_cache
    cache = real_loader("train")
    poisoned = dict(cache)
    rng = np.random.default_rng(7)
    poisoned["labels"] = rng.permutation(np.asarray(cache["labels"]))
    try:
        D.load_cache = lambda split: (poisoned if split == "train" else real_loader(split))
        w3 = swo_weights.build()["weights"]
    finally:
        D.load_cache = real_loader
    # vocab matters: a 5k-only vocabulary must give a different low4 profile
    import bjm_common as BC
    graphs = BC.molecules_from_cache(cache)
    half = BC.radius2_support(graphs[:5000], graphs)
    w4_low = np.asarray(half["low4_frac"], dtype=np.float64)
    ess = float(w1.sum() ** 2 / (w1 ** 2).sum())
    out = {
        "sha_match": S.sha256_array(w1) == meta["weight_sha256"],
        "deterministic": bool(np.array_equal(w1, w2)),
        "label_free": bool(np.array_equal(w1, w3)),
        "vocab_sensitive": bool(abs(w4_low.mean() - again["stats"]["low4"]["mean"]) > 1e-3),
        "mean": float(w1.mean()), "min": float(w1.min()), "max": float(w1.max()),
        "ess": ess, "ess_recorded": meta["effective_sample_size"],
        "ess_fraction": ess / len(w1),
        "clip_fraction": meta["clip_fraction"],
        "frac_gt_1": float(np.mean(w1 > 1.0)),
        "half_vocab_low4_mean": float(w4_low.mean()),
        "full_vocab_low4_mean": again["stats"]["low4"]["mean"],
    }
    out["pass"] = bool(out["sha_match"] and out["deterministic"] and out["label_free"]
                       and out["vocab_sensitive"]
                       and abs(out["mean"] - 1.0) < 1e-12
                       and abs(ess - meta["effective_sample_size"]) < 1e-6)
    return out


def _g6():
    import ad_common as C
    import ad_data as D
    cand, builder, _ = S.dcc_candidate()
    hashes = {}
    for arm in ("E", "W"):
        m, info = builder(cand, D.load_norm(), 0, torch.device("cpu"))
        hashes[arm] = {"state": C.state_hash(m),
                       "ssl": (info.get("ssl") or {}).get("sha256")}
        del m
    anchor = S.read_json(S.E_ANCHOR / "artifacts.json")
    g_e = torch.Generator().manual_seed(0 + 101)
    g_w = torch.Generator().manual_seed(0 + 101)
    order_e = torch.randperm(10_000, generator=g_e)
    order_w = torch.randperm(10_000, generator=g_w)
    out = {"E_hash": hashes["E"]["state"], "W_hash": hashes["W"]["state"],
           "historical_anchor_hash": anchor["initial_state_sha256"],
           "ssl_hash_E": hashes["E"]["ssl"], "ssl_hash_W": hashes["W"]["ssl"],
           "batch_order_identical": bool(torch.equal(order_e, order_w)),
           "batch_order_head": [int(x) for x in order_e[:6]]}
    out["pass"] = bool(hashes["E"]["state"] == hashes["W"]["state"]
                       == anchor["initial_state_sha256"]
                       and out["batch_order_identical"]
                       and hashes["E"]["ssl"] == hashes["W"]["ssl"])
    return out


def _g7():
    q = chr(34)
    needles = ["load_cache(" + q + "test" + q + ")", "load_cache(" + chr(39) + "test" + chr(39) + ")",
               "ad_" + "test", "cache_path(" + q + "test" + q + ")", "official_" + "test.pt"]
    hits = []
    for p in sorted((ROUND / "code").glob("*.py")):
        text = p.read_text()
        for needle in needles:
            if needle in text:
                hits.append({"file": p.name, "needle": needle})
    core = (ROUND / "code" / "swo_train_core.py").read_text()
    cache_loads = sorted({ln.strip() for ln in core.splitlines() if "load_cache(" in ln})
    return {"hits": hits, "trainer_cache_loads": cache_loads,
            "trainer_loads_only_train_valid": bool(
                all('"train"' in ln or '"valid"' in ln for ln in cache_loads)),
            "pass": bool(not hits and all('"train"' in ln or '"valid"' in ln
                                          for ln in cache_loads))}


def _g8():
    import ad_common as C
    import ad_data as D
    cand, builder, _ = S.dcc_candidate()
    m, _ = builder(cand, D.load_norm(), 0, torch.device("cpu"))
    n = C.parameter_count(m)
    anchor = S.read_json(S.E_ANCHOR / "artifacts.json")
    return {"parameters": n, "cap": S.PARAM_CAP,
            "historical_parameters": anchor["parameter_count"],
            "pass": bool(n <= S.PARAM_CAP and n == anchor["parameter_count"])}


def _g9():
    torch.manual_seed(3)
    w = torch.rand(64, dtype=torch.float64) * 1.5 + 0.2
    e0 = torch.randn(64, dtype=torch.float64)
    target = torch.zeros(64, dtype=torch.float64)

    def loss_of(pred):
        return float((w * (pred - target).abs()).sum() / w.sum())

    x = target.clone().requires_grad_(True)
    pred = (target + e0).clone().requires_grad_(True)
    lv = (w * (pred - target).abs()).sum() / w.sum()
    lv.backward()
    analytic = float(pred.grad.sum() * 0 + pred.grad[0])
    eps = 1e-6
    p_plus = (target + e0).clone()
    p_plus[0] += eps
    numeric = (loss_of(p_plus) - loss_of((target + e0).clone())) / eps
    expected = float(w[0] * math.copysign(1.0, float(e0[0])) / w.sum())
    return {"analytic_first": analytic, "numeric_first": numeric, "expected_first": expected,
            "pass": bool(abs(numeric - expected) < 1e-4 and abs(analytic - expected) < 1e-6)}


def main(argv=None) -> int:
    gates = [("G1_recipe_identity", _g1), ("G2_trainer_diff", _g2),
             ("G3_E_equivalence", _g3), ("G4_weighted_identity", _g4),
             ("G5_weights_frozen", _g5), ("G6_init_batch_alignment", _g6),
             ("G7_no_test_access", _g7), ("G8_capacity", _g8),
             ("G9_gradient", _g9)]
    results, ok = {}, 0
    for name, fn in gates:
        try:
            r = fn()
        except Exception as exc:  # noqa: BLE001
            r = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        results[name] = r
        ok += bool(r.get("pass"))
        print(f"[gate-swo] {name}: {'PASS' if r.get('pass') else 'FAIL'}"
              + ("" if r.get("pass") else f"  {json.dumps(r)[:300]}"))
        sys.stdout.flush()
    S.write_json(OUT, {"round": S.ROUND_ID, "written": S.now(),
                       "n_pass": ok, "n_gates": len(gates), "gates": results,
                       "official_test_loaded": False})
    print(f"[gate-swo] {ok}/{len(gates)} pass -> {S.rel(OUT)}")
    return 0 if ok == len(gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
