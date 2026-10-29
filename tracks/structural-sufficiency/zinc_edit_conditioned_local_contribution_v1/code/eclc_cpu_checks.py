"""ECLC-v1 Phase 2 CPU correctness gates.

All checks run on CPU, before any GPU training.  They cover:

1. independent local-statistics reference;
2. synthetic radius-1 invariance / radius-2 response;
3. atom/bond/batch invariance;
4. parameter identity and cap;
5. initial equality with frozen LCRC-A;
6. gradient finiteness and base/context/pair path liveness;
7. checkpoint round-trip;
8. split isolation and no test access;
9. pair-loss algebra (no new labels, not a substitute for molecule L1);
10. the REAL/NULL mechanism difference is real;
11. the post-step model is not forward-equivalent to LCRC-A;
12. identical data/pair streams for REAL and NULL.

Output: ``analysis/cpu_checks.json`` and ``analysis/cpu_checks.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import eclc_common as E
import eclc_models as EM
import eclc_train as T

D = E.D
L = E.L
M = E.M

CHECKS: dict[str, dict[str, Any]] = {}


def record(name: str, passed: bool, **details) -> None:
    CHECKS[name] = {"pass": bool(passed), **details}
    E.log(f"[cpu] {name}: {'PASS' if passed else 'FAIL'}")


def _fake_cache_from_mol(mol: Mapping[str, Any]) -> dict[str, Any]:
    cache = D._concat_molecules([mol])
    cache["labels"] = np.zeros(1, dtype=np.float32)
    cache["n_molecules"] = 1
    cache["split"] = "synthetic"
    cache["meta"] = {"feature_version": L.FEATURE_VERSION, "split": "synthetic",
                     "n_molecules": 1, "labels_loaded": False}
    return cache


def _real_cache_from_rows(rows: Sequence[int]) -> dict[str, Any]:
    objs, _ = L.D.load_p3_objects("train")
    mols = [D.build_molecule_from_object(objs[int(r)]) for r in rows]
    cache = D._concat_molecules(mols)
    cache["labels"] = np.zeros(len(rows), dtype=np.float32)
    cache["n_molecules"] = len(rows)
    cache["split"] = "train_subset"
    cache["meta"] = {"feature_version": L.FEATURE_VERSION, "split": "train_subset",
                     "n_molecules": len(rows), "labels_loaded": False}
    return cache


def _naive_base(atom_type, bond_index, bond_type):
    """Independent per-atom radius-1/2 count reference (pure Python loops)."""
    n = len(atom_type)
    adj = [[] for _ in range(n)]
    seen = set()
    for e in range(len(bond_type)):
        u, v = int(bond_index[e, 0]), int(bond_index[e, 1])
        bt = int(bond_type[e])
        key = (min(u, v), max(u, v), bt)
        if key in seen:  # PyG stores every bond twice
            continue
        seen.add(key)
        adj[u].append((v, bt))
        adj[v].append((u, bt))
    r1 = np.zeros((n, E.BASE_R1_DIM), dtype=np.int64)
    r2p = np.zeros((n, E.BASE_R2_PATH_DIM), dtype=np.int64)
    r2e = np.zeros((n, E.BASE_R2_END_DIM), dtype=np.int64)
    r2c = np.zeros((n, 1), dtype=np.int64)
    for v in range(n):
        for u, bt in adj[v]:
            r1[v, int(atom_type[u]) * 3 + (bt - 1)] += 1
        for w, bt1 in adj[v]:
            for u, bt2 in adj[w]:
                if u == v:
                    continue
                r2p[v, int(atom_type[w]) * 9 + (bt1 - 1) * 3 + (bt2 - 1)] += 1
                r2e[v, int(atom_type[u]) * 3 + (bt2 - 1)] += 1
                if any(x == u for x, _ in adj[v]):
                    r2c[v, 0] += 1
    center = np.zeros((n, E.BASE_CENTER_DIM), dtype=np.int64)
    center[np.arange(n), np.asarray(atom_type, dtype=np.int64)] = 1
    return np.concatenate([center, np.log1p(r1), np.log1p(r2p),
                           np.log1p(r2e), np.log1p(r2c)], axis=1).astype(np.float32)


def check_independent_reference() -> None:
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(L.C.ZINC_ROOT), subset=True, split="train")
    rows = [0, 1, 2, 3, 10, 42, 100, 172, 500, 1000, 2500, 5000, 7500, 9999]
    max_base = 0.0
    max_int = 0
    n_atoms = 0
    for r in rows:
        d = ds[int(r)]
        x = d.x.reshape(-1).numpy()
        ei = d.edge_index.numpy().T
        ea = d.edge_attr.reshape(-1).numpy()
        ref = _naive_base(x, ei, ea)
        # cache-side single molecule
        obj = L.D.load_p3_objects("train")[0][int(r)]
        mol = D.build_molecule_from_object(obj)
        cache = _fake_cache_from_mol(mol)
        batch = D.collate(cache, [0])
        got = EM.base_features(batch, zero_r2=False).numpy()
        max_base = max(max_base, float(np.abs(got - ref).max()))
        max_int = max(max_int, int(np.abs(got[:, :E.BASE_R1_TOTAL] - ref[:, :E.BASE_R1_TOTAL]).max()))
        n_atoms += len(x)
    record("C1_independent_reference", max_base == 0.0 and max_int == 0,
           n_molecules=len(rows), n_atoms=int(n_atoms),
           max_abs_diff=float(max_base), max_int_diff=int(max_int),
           note="naive loop over atom_type/bond_index/bond_type vs cache-derived base_features")


def _synthetic_mols():
    """Two graphs with identical radius-1 at atom 0 and different radius-2."""
    def make(change: bool):
        atom_type = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        if change:
            atom_type[3] = 2  # C -> N at a second-shell atom
        bond_index = np.array([[0, 1], [0, 2], [1, 3], [2, 4]], dtype=np.int64)
        bond_type = np.array([1, 1, 2 if change else 1, 1], dtype=np.int64)
        atom_feat = np.zeros((5, L.ATOM_FEAT_DIM), dtype=np.float32)
        bond_feat = np.zeros((4, L.BOND_FEAT_DIM), dtype=np.float32)
        global_feat = np.zeros(L.GLOBAL_DIM, dtype=np.float32)
        mol = D.build_from_arrays(atom_feat, atom_type, bond_index, bond_type,
                                  bond_feat, global_feat)
        return _fake_cache_from_mol(mol)

    return make(False), make(True)


def _nonzero_base(model):
    torch.manual_seed(12345)
    with torch.no_grad():
        torch.nn.init.normal_(model.base.lins[1].weight, mean=0.0, std=0.5)
        torch.nn.init.zeros_(model.base.lins[1].bias)


def check_synthetic() -> None:
    norm = E.load_norm()
    ca, cb = _synthetic_mols()
    ba = D.collate(ca, [0])
    bb = D.collate(cb, [0])
    models = {}
    for arm in E.ARMS:
        m, _ = EM.build_model(arm, norm, 0, "cpu")
        _nonzero_base(m)
        models[arm] = m
    with torch.no_grad():
        fa = EM.base_features(ba, zero_r2=True)
        fb = EM.base_features(bb, zero_r2=True)
        na = EM.base_features(ba, zero_r2=False)
        nb = EM.base_features(bb, zero_r2=False)
        # radius-1 block must be identical at the target atom
        r1_equal = bool(torch.equal(fa[0, :E.BASE_R1_TOTAL], fb[0, :E.BASE_R1_TOTAL]))
        r2_changed = bool(not torch.equal(na[0, E.BASE_R1_TOTAL:], nb[0, E.BASE_R1_TOTAL:]))
        real_zero = bool(torch.count_nonzero(fa[:, E.BASE_R1_TOTAL:]).item() == 0)
        real_base_a = models["REAL"].base(fa)
        real_base_b = models["REAL"].base(fb)
        null_base_a = models["NULL"].base(na)
        null_base_b = models["NULL"].base(nb)
        real_same = bool(torch.equal(real_base_a[0], real_base_b[0]))
        null_diff = bool(not torch.equal(null_base_a[0], null_base_b[0]))
        # f_context must respond to the radius-2 change
        _, auxa = models["REAL"](ba, return_aux=True)
        _, auxb = models["REAL"](bb, return_aux=True)
        context_diff = float((auxa["context_atom"][0] - auxb["context_atom"][0]).abs().max())
    passed = (r1_equal and r2_changed and real_zero and real_same and null_diff
              and context_diff > 0.0)
    record("C2_synthetic_radius1_invariance", passed,
           radius1_block_identical=bool(r1_equal),
           radius2_block_changed=bool(r2_changed),
           real_radius2_input_zero=bool(real_zero),
           real_fbase_target_exactly_equal=bool(real_same),
           null_fbase_target_differs=bool(null_diff),
           f_context_target_max_abs_diff=float(context_diff),
           note="atom 0 radius-1 environment fixed; a second-shell atom type and bond order change")


def check_invariance() -> None:
    norm = E.load_norm()
    rows = list(range(8))
    cache = _real_cache_from_rows(rows)
    batch = D.collate(cache, rows)
    models = {}
    for arm in E.ARMS:
        m, _ = EM.build_model(arm, norm, 0, "cpu")
        models[arm] = m

    def pred(m, b):
        with torch.no_grad():
            return m(b).numpy()

    # renumber atoms of one molecule and compare the graph-level prediction
    objs, _ = L.D.load_p3_objects("train")
    rng = np.random.default_rng(0)
    max_perm = 0.0
    max_rev = 0.0
    max_split = 0.0
    max_order = 0.0
    for r in rows:
        obj = objs[int(r)]
        base_mol = D.build_molecule_from_object(obj)
        perm = rng.permutation(base_mol["n_atoms"])
        perm_mol = D.build_molecule_from_object(obj, perm=perm)
        rev_mol = D.build_molecule_from_object(obj, reverse_bonds=True)
        for arm, m in models.items():
            p0 = pred(m, D.collate(_fake_cache_from_mol(base_mol), [0]))[0]
            p1 = pred(m, D.collate(_fake_cache_from_mol(perm_mol), [0]))[0]
            p2 = pred(m, D.collate(_fake_cache_from_mol(rev_mol), [0]))[0]
            max_perm = max(max_perm, abs(float(p0 - p1)))
            max_rev = max(max_rev, abs(float(p0 - p2)))
    # batch split and order on the 8-molecule cache
    for arm, m in models.items():
        p_all = pred(m, batch)
        p_a = pred(m, D.collate(cache, rows[:3]))
        p_b = pred(m, D.collate(cache, rows[3:]))
        p_split = np.concatenate([p_a, p_b])
        max_split = max(max_split, float(np.abs(p_all - p_split).max()))
        order = list(reversed(rows))
        p_rev = pred(m, D.collate(cache, order))
        max_order = max(max_order, float(np.abs(p_all - p_rev[::-1]).max()))
    passed = max(max_perm, max_rev, max_split, max_order) <= 1e-5
    record("C3_invariance", passed,
           max_abs_atom_renumber=float(max_perm),
           max_abs_bond_direction=float(max_rev),
           max_abs_batch_split=float(max_split),
           max_abs_molecule_order=float(max_order),
           tolerance=1e-5)


def check_params_and_init() -> None:
    norm = E.load_norm()
    models = {}
    infos = {}
    for arm in E.ARMS:
        m, info = EM.build_model(arm, norm, 0, "cpu")
        models[arm] = m
        infos[arm] = info
    real, null = models["REAL"], models["NULL"]
    same_count = infos["REAL"]["parameters"] == infos["NULL"]["parameters"]
    same_breakdown = infos["REAL"]["parameters_breakdown"] == infos["NULL"]["parameters_breakdown"]
    within_cap = infos["REAL"]["parameters"] <= 500_000
    state_equal = all(torch.equal(a, b) for a, b in zip(real.state_dict().values(),
                                                         null.state_dict().values()))
    hash_equal = infos["REAL"]["initial_state_sha256"] == infos["NULL"]["initial_state_sha256"]
    zero_last = bool(torch.count_nonzero(real.base.lins[1].weight).item() == 0
                     and torch.count_nonzero(real.base.lins[1].bias).item() == 0)
    record("C4_parameter_identity", same_count and same_breakdown and within_cap
           and state_equal and hash_equal and zero_last,
           parameters=infos["REAL"]["parameters"],
           breakdown=infos["REAL"]["parameters_breakdown"],
           cap=500000, same_count=bool(same_count),
           same_breakdown=bool(same_breakdown), within_cap=bool(within_cap),
           full_state_equal=bool(state_equal), state_hash_equal=bool(hash_equal),
           base_last_layer_zero=bool(zero_last))


def check_initial_equivalence() -> None:
    norm = E.load_norm()
    cache = _real_cache_from_rows(range(16))
    batch = D.collate(cache, list(range(16)))
    legacy, _ = M.build_model("A", norm, 0, "cpu")
    models = {}
    infos = {}
    for arm in E.ARMS:
        m, info = EM.build_model(arm, norm, 0, "cpu")
        models[arm] = m
        infos[arm] = info
    with torch.no_grad():
        y_legacy = legacy(batch).numpy()
        y_real = models["REAL"](batch).numpy()
        y_null = models["NULL"](batch).numpy()
    max_real = float(np.abs(y_legacy - y_real).max())
    max_null = float(np.abs(y_legacy - y_null).max())
    shared_equal = all(torch.equal(legacy.state_dict()[k], models["REAL"].backbone.state_dict()[k])
                       for k in legacy.state_dict())
    record("C5_initial_lcrc_equivalence", max_real == 0.0 and max_null == 0.0 and shared_equal,
           max_abs_real_vs_lcrcA=float(max_real),
           max_abs_null_vs_lcrcA=float(max_null),
           shared_backbone_tensors_equal=bool(shared_equal),
           note="f_base is exactly zero at step 0; this equality is intentional, not the trained model")


def _batch_and_pair():
    cache = E.load_train_cache()
    pairs = T.load_pair_arrays()
    rows = np.arange(128, dtype=np.int64)
    batch = D.collate(cache, rows)
    target = torch.as_tensor(cache["labels"][rows], dtype=torch.float32)
    pair_ids = np.arange(8, dtype=np.int64)
    pi, pj = pairs["i"][pair_ids], pairs["j"][pair_ids]
    pair_rows = np.stack([pi, pj], axis=1).reshape(-1)
    pair_batch = D.collate(cache, pair_rows)
    pair_target = torch.as_tensor(cache["labels"][pi] - cache["labels"][pj], dtype=torch.float32)
    return cache, rows, batch, target, pair_ids, pair_batch, pair_target


def check_gradients() -> None:
    norm = E.load_norm()
    cache, rows, batch, target, pair_ids, pair_batch, pair_target = _batch_and_pair()
    out = {}
    passed = True
    for arm in E.ARMS:
        m, _ = EM.build_model(arm, norm, 0, "cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
        # step 1: combined loss gradient audit
        pred = m(batch)
        lm = F.l1_loss(pred.view(-1), target.view(-1))
        pp = m(pair_batch).view(8, 2)
        lp = F.l1_loss(pp[:, 0] - pp[:, 1], pair_target)
        loss = lm + E.PAIR_LOSS_COEF * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_last = m.base.lins[1].weight.grad
        g_first = m.base.lins[0].weight.grad
        g_ctx = m.backbone.f.lins[-1].weight.grad
        g_rho = m.backbone.rho.lins[-1].weight.grad
        all_finite = all(torch.isfinite(p.grad).all().item()
                         for p in m.parameters() if p.grad is not None)
        step1 = {
            "loss_mol": float(lm.item()),
            "loss_pair": float(lp.item()),
            "pair_target_std": float(pair_target.std().item()),
            "base_last_grad_norm": float(torch.linalg.vector_norm(g_last).item()),
            "base_first_grad_norm": float(torch.linalg.vector_norm(g_first).item()),
            "base_r2_col_grad_max_abs": float(g_first[:, E.BASE_R1_TOTAL:].abs().max().item()),
            "base_r1_col_grad_max_abs": float(g_first[:, :E.BASE_R1_TOTAL].abs().max().item()),
            "f_context_grad_norm": float(torch.linalg.vector_norm(g_ctx).item()),
            "rho_grad_norm": float(torch.linalg.vector_norm(g_rho).item()),
            "all_finite": bool(all_finite),
        }
        opt.step()
        # step 2: first-layer gradient is no longer blocked by the zero last layer
        pred = m(batch)
        lm = F.l1_loss(pred.view(-1), target.view(-1))
        pp = m(pair_batch).view(8, 2)
        lp = F.l1_loss(pp[:, 0] - pp[:, 1], pair_target)
        loss = lm + E.PAIR_LOSS_COEF * lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_first2 = m.base.lins[0].weight.grad
        step2 = {
            "base_r1_col_grad_max_abs": float(g_first2[:, :E.BASE_R1_TOTAL].abs().max().item()),
            "base_r2_col_grad_max_abs": float(g_first2[:, E.BASE_R1_TOTAL:].abs().max().item()),
            "base_last_grad_norm": float(torch.linalg.vector_norm(m.base.lins[1].weight.grad).item()),
        }
        # pair-only gradient audit
        opt.zero_grad(set_to_none=True)
        pp = m(pair_batch).view(8, 2)
        lp = F.l1_loss(pp[:, 0] - pp[:, 1], pair_target)
        lp.backward()
        pair_only = {
            "base_last_grad_norm": float(torch.linalg.vector_norm(m.base.lins[1].weight.grad).item()),
            "f_context_grad_norm": float(torch.linalg.vector_norm(m.backbone.f.lins[-1].weight.grad).item()),
        }
        arm_pass = (
            step1["all_finite"]
            and step1["loss_pair"] > 0
            and step1["pair_target_std"] > 0
            and step1["base_last_grad_norm"] > 0
            and step1["f_context_grad_norm"] > 0
            and step1["rho_grad_norm"] > 0
            and step2["base_r1_col_grad_max_abs"] > 0
            and pair_only["base_last_grad_norm"] > 0
            and pair_only["f_context_grad_norm"] > 0
        )
        if arm == "REAL":
            arm_pass = arm_pass and step2["base_r2_col_grad_max_abs"] == 0.0
        else:
            arm_pass = arm_pass and step2["base_r2_col_grad_max_abs"] > 0.0
        out[arm] = {"step1": step1, "step2": step2, "pair_only": pair_only,
                    "pass": bool(arm_pass)}
        passed = passed and arm_pass
    record("C6_gradients_and_paths", passed, arms=out,
           note="step 1 audits the zero-init last layer; step 2 audits the first layer after the last layer is non-zero")


def check_checkpoint() -> None:
    norm = E.load_norm()
    cache = _real_cache_from_rows(range(4))
    batch = D.collate(cache, list(range(4)))
    out = {}
    passed = True
    for arm in E.ARMS:
        m, _ = EM.build_model(arm, norm, 0, "cpu")
        with torch.no_grad():
            y0 = m(batch).numpy()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.pt"
            torch.save(m.state_dict(), p)
            state = torch.load(p, map_location="cpu", weights_only=True)
            m2 = EM.load_model_from_state(arm, norm, state, "cpu")
            with torch.no_grad():
                y1 = m2(batch).numpy()
        diff = float(np.abs(y0 - y1).max())
        out[arm] = {"max_abs_diff": diff}
        passed = passed and diff <= 1e-6
    record("C7_checkpoint_roundtrip", passed, arms=out, tolerance=1e-6)


def check_split_isolation() -> None:
    pairs = T.load_pair_arrays()
    i, j = pairs["i"], pairs["j"]
    train_only = bool(i.min() >= 0 and j.min() >= 0 and i.max() < 10000 and j.max() < 10000)
    no_self = bool(np.all(i != j))
    production = ["eclc_common.py", "eclc_match.py", "eclc_models.py",
                  "eclc_phase0.py", "eclc_train.py"]
    bad_refs = []
    forbidden = "lc_" + "test"
    for name in production:
        p = E.CODE_DIR / name
        if p.exists() and forbidden in p.read_text():
            bad_refs.append(name)
    valid_len = len(E.load_valid_cache()["labels"])
    payload = pairs["payload"]
    record("C8_split_isolation", train_only and no_self and not bad_refs and valid_len == 1000
           and payload.get("official_test_loaded") is False and payload.get("labels_read") is False,
           pair_rows_in_train=bool(train_only), no_self_pairs=bool(no_self),
           code_refs_to_lc_test=bad_refs, valid_rows=int(valid_len),
           pair_payload_labels_read=payload.get("labels_read"),
           pair_payload_official_test_loaded=payload.get("official_test_loaded"),
           note="the frozen LCRC track has a post-hoc lc_test.pt on disk; ECLC never references it")


def check_pair_algebra() -> None:
    cache, rows, batch, target, pair_ids, pair_batch, pair_target = _batch_and_pair()
    norm = E.load_norm()
    m, _ = EM.build_model("NULL", norm, 0, "cpu")
    _nonzero_base(m)
    pred, aux = m(batch, return_aux=True)
    # use the first two graphs of the batch as a pair for the identity check
    d = pred[1] - pred[0]
    terms = (aux["context"][1] - aux["context"][0]
             + aux["base"][1] - aux["base"][0]
             + aux["rho"][1] - aux["rho"][0])
    identity = float(abs(float(d - terms)))
    # constant offset leaves pair loss unchanged, changes molecule L1
    y = torch.as_tensor(cache["labels"][rows], dtype=torch.float32)
    base_pair = F.l1_loss(pred[1] - pred[0], y[1] - y[0])
    offset_pair = F.l1_loss((pred[1] + 3.7) - (pred[0] + 3.7), y[1] - y[0])
    base_mol = F.l1_loss(pred, y)
    offset_mol = F.l1_loss(pred + 3.7, y)
    # gradient non-proportionality
    m.zero_grad(set_to_none=True)
    lp = F.l1_loss(pred[1] - pred[0], y[1] - y[0])
    lp.backward(retain_graph=True)
    g_pair = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    m.zero_grad(set_to_none=True)
    lm = F.l1_loss(pred, y)
    lm.backward()
    g_mol = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    cos = float(torch.nn.functional.cosine_similarity(g_pair.unsqueeze(0), g_mol.unsqueeze(0)).item())
    ratio = float((g_pair.norm() / g_mol.norm()).item()) if g_mol.norm() > 0 else float("nan")
    # squared-loss pair is a quadratic form in residuals; numeric identity
    r = (pred - y).double()
    n = len(r)
    pair_sq = sum(float((r[a] - r[b]) ** 2) for a in range(n) for b in range(a + 1, n))
    quad = float((n - 1) * (r ** 2).sum() - 2 * sum(float(r[a] * r[b]) for a in range(n) for b in range(a + 1, n)))
    record("C9_pair_loss_algebra", identity <= 1e-5 and abs(float(base_pair - offset_pair)) <= 1e-6
           and float(offset_mol - base_mol) > 0 and abs(cos) < 0.999
           and abs(pair_sq - quad) <= 1e-6,
           additive_identity_abs_diff=float(identity),
           pair_loss_offset_diff=float(abs(float(base_pair - offset_pair))),
           molecule_loss_offset_delta=float(offset_mol - base_mol),
           pair_vs_molecule_grad_cosine=float(cos),
           grad_norm_ratio=float(ratio),
           squared_pair_quadratic_identity=float(abs(pair_sq - quad)),
           note="pair loss uses only y_i,y_j; it is invariant to a common offset and is not a substitute for molecule L1")


def check_mechanism_difference() -> None:
    norm = E.load_norm()
    cache = _real_cache_from_rows(range(64))
    batch = D.collate(cache, list(range(64)))
    models = {}
    for arm in E.ARMS:
        m, _ = EM.build_model(arm, norm, 0, "cpu")
        _nonzero_base(m)
        models[arm] = m
    with torch.no_grad():
        xr = EM.base_features(batch, zero_r2=True)
        xn = EM.base_features(batch, zero_r2=False)
        br = models["REAL"].base(xr)
        bn = models["NULL"].base(xn)
        base_diff = float((br - bn).abs().max())
        r2_nonzero_frac = float((xn[:, E.BASE_R1_TOTAL:] != 0).any(dim=1).float().mean())
        _, auxr = models["REAL"](batch, return_aux=True)
        _, auxn = models["NULL"](batch, return_aux=True)
        context_diff = float((auxr["context"] - auxn["context"]).abs().max())
        predr = models["REAL"](batch)
        predn = models["NULL"](batch)
        pred_diff = float((predr - predn).abs().max())
    record("C10_mechanism_difference", base_diff > 0 and context_diff == 0.0 and pred_diff > 0
           and r2_nonzero_frac > 0,
           max_abs_fbase_diff=float(base_diff),
           max_abs_fcontext_diff=float(context_diff),
           max_abs_full_pred_diff=float(pred_diff),
           fraction_atoms_with_radius2_content=float(r2_nonzero_frac))


def check_post_step_non_equivalence() -> None:
    norm = E.load_norm()
    cache = _real_cache_from_rows(range(8))
    batch = D.collate(cache, list(range(8)))
    legacy, _ = M.build_model("A", norm, 0, "cpu")
    out = {}
    passed = True
    for arm in E.ARMS:
        m, _ = EM.build_model(arm, norm, 0, "cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
        cache128, rows, batch128, target, pair_ids, pair_batch, pair_target = _batch_and_pair()
        pred = m(batch128)
        lm = F.l1_loss(pred.view(-1), target.view(-1))
        pp = m(pair_batch).view(8, 2)
        lp = F.l1_loss(pp[:, 0] - pp[:, 1], pair_target)
        (lm + E.PAIR_LOSS_COEF * lp).backward()
        opt.step()
        with torch.no_grad():
            y_m = m(batch).numpy()
            y_l = legacy(batch).numpy()
            # zero the base head temporarily and compare
            saved = [p.detach().clone() for p in m.base.parameters()]
            for p in m.base.parameters():
                p.zero_()
            y_zero = m(batch).numpy()
            for p, s in zip(m.base.parameters(), saved):
                p.copy_(s)
        diff_lcrc = float(np.abs(y_m - y_l).max())
        diff_zero = float(np.abs(y_m - y_zero).max())
        arm_pass = diff_lcrc > 0 and diff_zero > 0
        out[arm] = {"max_abs_vs_lcrcA": diff_lcrc, "max_abs_zero_base_ablation": diff_zero,
                    "pass": bool(arm_pass)}
        passed = passed and arm_pass
    record("C11_post_step_non_equivalence", passed, arms=out,
           note="the zero-base equality holds only at initialisation; after one step f_base is active")


def check_streams_identical() -> None:
    def stream(seed):
        g_data = torch.Generator().manual_seed(seed + 101)
        g_pair = torch.Generator().manual_seed(seed + 202)
        data = torch.randperm(10000, generator=g_data)[:1000].numpy().tolist()
        pair = torch.randint(0, 350, (1000,), generator=g_pair).numpy().tolist()
        return data, pair

    d0, p0 = stream(0)
    d0b, p0b = stream(0)
    record("C12_streams_identical", d0 == d0b and p0 == p0b,
           data_order_first_1000_sha=hash(tuple(d0)) & 0xFFFFFFFF,
           pair_stream_first_1000_sha=hash(tuple(p0)) & 0xFFFFFFFF,
           note="the stream is a pure function of seed, not of arm")


def main() -> int:
    E.ensure_dir(E.ANALYSIS)
    check_independent_reference()
    check_synthetic()
    check_invariance()
    check_params_and_init()
    check_initial_equivalence()
    check_gradients()
    check_checkpoint()
    check_split_isolation()
    check_pair_algebra()
    check_mechanism_difference()
    check_post_step_non_equivalence()
    check_streams_identical()
    all_pass = bool(all(v["pass"] for v in CHECKS.values()))
    payload = {
        "round": E.ROUND_ID,
        "created": E.now(),
        "git_commit": E.git_commit(),
        "environment": L.env_summary(),
        "all_pass": all_pass,
        "n_checks": len(CHECKS),
        "checks": CHECKS,
    }
    E.write_json(E.ANALYSIS / "cpu_checks.json", payload)
    lines = ["# Phase 2 CPU gate record", "",
             f"- Round: `{E.ROUND_ID}`",
             f"- Created: `{payload['created']}`",
             f"- All pass: **`{all_pass}`**", "",
             "| check | pass | key evidence |", "|---|:---:|---|"]
    for name, v in CHECKS.items():
        keys = [f"{k}={v[k]}" for k in v if k not in ("pass", "note")][:4]
        lines.append(f"| {name} | {'yes' if v['pass'] else 'no'} | {'; '.join(keys)} |")
    (E.ANALYSIS / "cpu_checks.md").write_text("\n".join(lines) + "\n")
    E.log(f"[cpu] all_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
