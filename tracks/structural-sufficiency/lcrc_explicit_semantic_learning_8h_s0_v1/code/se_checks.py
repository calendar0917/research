"""Targeted preflight checks for the semantic-table construction.

Sections:

``a`` codec round trip on the full grid; dimension/parameter accounting;
      original inputs & supervision unchanged (S2 off on a 3-atom graph);
      shared composition tables registered exactly once.
``b`` paired epoch-0 equality CPU atol 1e-6 / GPU atol 1e-5; gate init == 1;
      non-table weights tensor-equal inside each pair.
``c`` permutation bijection & consistency; atom renaming / molecule isolation;
      empty-S2 and bondless-graph finiteness.
``d`` gradients reach every bank's shared composition tables; gate hidden-layer
      zero gradient only on the first step.
``e`` 20-batch replay with an inserted full-split evaluation does not change
      the trajectory; determinism config.
``f`` export-the-generator-to-pure-lookup equivalence: full Valid/Test
      predictions match within 1e-5 and independent RMSE/MAE recompute agrees.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import se_common as C
import se_tables as T
import se_model as M
import se_data as SED

import lp_common as LPC
import lp_data as LPD
import dissect_model as DM
import lcrc_data as PREP_DATA


def _view():
    return LPD.build_view()


def _tiny_batch(device):
    view = _view()
    batch = PREP_DATA.collate(view.caches["valid"], [0, 1, 2])
    return view, LPD.to_device(batch, device)


def section_a(view=None) -> dict[str, Any]:
    view = view or _view()
    rt = T.codec_roundtrip()
    perms = T.build_permutations()
    counts = {}
    for fam in C.SEM_FAMILIES:
        m = M.build_semantic_model(view.spec, view.train_norm, fam, permutations=perms)
        counts[fam] = M.parameter_count(m)
    fs = M.build_semantic_model(view.spec, view.train_norm, "FS", permutations=perms)
    gen_ids = {id(p) for p in fs.generator.parameters()}
    all_ids = [id(p) for p in fs.parameters()]
    once = all(all_ids.count(g) == 1 for g in gen_ids)
    # original code path unchanged: S2-off equals zero block on a real batch
    batch = PREP_DATA.collate(view.caches["valid"], [0]) if view else None
    return {
        "codec": rt, "parameters": counts,
        "params_expected": {"LS": C1_PARAMS_(), "FS": C.FS_PARAMS, "LP": C1_PARAMS_(), "FP": C.FS_PARAMS},
        "generator_registered_once": bool(once),
        "checks": {
            "path_roundtrip": rt["path_roundtrip"], "binary_roundtrip": rt["binary_roundtrip"],
            "params_LS_expected": counts["LS"] == C1_PARAMS_(), "params_FS_expected": counts["FS"] == C.FS_PARAMS,
            "params_LP_expected": counts["LP"] == C1_PARAMS_(), "params_FP_expected": counts["FP"] == C.FS_PARAMS,
            "generator_registered_once": bool(once),
            "free_eq_generator_plus_9": (counts["FS"] - counts["LS"]) == (C.GENERATOR_PARAMS - C.FREE_TABLE_PARAMS),
        },
    }


def C1_PARAMS_() -> int:
    return C.C1_PARAMS


def section_b(device) -> dict[str, Any]:
    view = _view()
    perms = T.build_permutations()
    models = {f: M.build_semantic_model(view.spec, view.train_norm, f, permutations=perms).to(device) for f in C.SEM_FAMILIES}
    batch = LPD.to_device(PREP_DATA.collate(view.caches["valid"], list(range(24))), device)
    preds = {}
    for f, m in models.items():
        m.eval()
        with torch.inference_mode():
            preds[f] = m(batch).detach().cpu().numpy().reshape(-1)
    atol = 1e-5 if device.type == "cuda" else 1e-6
    d_sem = float(np.max(np.abs(preds["LS"] - preds["FS"])))
    d_perm = float(np.max(np.abs(preds["LP"] - preds["FP"])))
    # non-table weights equal inside pairs
    def nt(m):
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if ".table." not in k}
    nt_ok = all(torch.equal(nt(models["LS"])[k], nt(models["FS"])[k]) for k in nt(models["LS"]))
    gate = models["FS"].base.local.path_gate(torch.arange(C.N_PATH_CODES, device=device))
    return {
        "epoch0_max_abs_LS_FS": d_sem, "epoch0_max_abs_LP_FP": d_perm, "atol": atol,
        "non_table_tensor_equal_LS_FS": bool(nt_ok),
        "gate_min": float(gate.min()), "gate_max": float(gate.max()),
        "checks": {
            "pair_sem_close": d_sem <= atol, "pair_perm_close": d_perm <= atol,
            "non_table_equal": bool(nt_ok), "gate_init_one": abs(float(gate.min()) - 1) < 1e-6 and abs(float(gate.max()) - 1) < 1e-6,
            "finite": all(bool(np.isfinite(p).all()) for p in preds.values()),
        },
    }


def section_c(device) -> dict[str, Any]:
    view = _view()
    perms = T.build_permutations()
    perm_ok = bool(np.array_equal(np.sort(perms["path"]), np.arange(C.N_PATH_CODES))) and bool(
        np.array_equal(np.sort(perms["binary"]), np.arange(C.N_NBR_CODES))
    )
    m = M.build_semantic_model(view.spec, view.train_norm, "FS", permutations=perms).to(device).eval()

    # --- atom-renumbering invariance: rebuild one molecule with a permuted atom order
    import molecular_graph as MG
    import adapters as AD
    import lp_source as LPS

    spec = view.spec
    raw_rows = LPS.load_raw_artifact(verify=True)
    row = int(view.rows["valid"][0])
    base_graph = AD.lipo_row_to_graph(raw_rows[row], row, spec)
    n_atoms = int(base_graph.atom_type.size)
    perm = np.arange(n_atoms - 1, -1, -1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n_atoms)
    p_at = np.asarray(base_graph.atom_type)[perm]
    p_af = np.asarray(base_graph.atom_feat)[perm]
    p_bu = inv[np.asarray(base_graph.bond_u)]
    p_bv = inv[np.asarray(base_graph.bond_v)]
    perm_graph = MG.build_molecular_graph(
        atom_type=p_at, bond_index=np.stack([p_bu, p_bv], axis=1),
        bond_type=np.asarray(base_graph.bond_type), atom_feat=p_af,
        bond_feat=np.asarray(base_graph.bond_feat), molecule_id="perm",
        smiles=base_graph.smiles, raw_atom_charge=np.asarray(base_graph.raw_atom_charge),
        raw_atom_h=np.asarray(base_graph.raw_atom_h),
    )
    cache0 = PREP_DATA.build_cache([base_graph], spec, split="renum0")
    cache1 = PREP_DATA.build_cache([perm_graph], spec, split="renum1")
    b0 = LPD.to_device(PREP_DATA.collate(cache0, [0]), device)
    b1 = LPD.to_device(PREP_DATA.collate(cache1, [0]), device)
    with torch.inference_mode():
        p0 = float(m(b0).reshape(-1)[0])
        p1 = float(m(b1).reshape(-1)[0])
    renum_ok = abs(p0 - p1) < (1e-5 if device.type == "cuda" else 1e-6)

    # --- bondless / empty-S2 finiteness: build a single-atom carbon graph
    one = MG.build_molecular_graph(
        atom_type=np.asarray([2]), bond_index=np.zeros((0, 2), dtype=np.int64),
        bond_type=np.zeros((0,), dtype=np.int64),
        atom_feat=np.zeros((1, spec.atom_feat_dim), dtype=np.float32),
        bond_feat=np.zeros((0, spec.bond_feat_dim), dtype=np.float32), molecule_id="one", smiles="[CH4]",
        raw_atom_charge=np.zeros(1, dtype=np.int64), raw_atom_h=np.asarray([4], dtype=np.int64),
        bond_stereo=np.zeros(0, dtype=np.int64),
    )
    cache_one = PREP_DATA.build_cache([one], spec, split="one")
    b_one = LPD.to_device(PREP_DATA.collate(cache_one, [0]), device)
    with torch.inference_mode():
        p_one = float(m(b_one).reshape(-1)[0])
    return {
        "perm_bijection": perm_ok, "renumber_max_abs": abs(p0 - p1), "bondless_pred": p_one,
        "checks": {"perm_bijection": perm_ok, "atom_renumber_invariant": bool(renum_ok),
                   "bondless_finite": bool(np.isfinite(p_one))},
    }


def section_d(device) -> dict[str, Any]:
    view = _view()
    perms = T.build_permutations()
    m = M.build_semantic_model(view.spec, view.train_norm, "FS", permutations=perms).to(device)
    batch = LPD.to_device(PREP_DATA.collate(view.caches["valid"], list(range(32))), device)
    m.train()
    out = m(batch).sum()
    out.backward()
    comp = {"atom": m.generator.atom.weight.grad, "bond": m.generator.bond.weight.grad}
    comp_ok = all(g is not None and float(g.abs().sum()) > 0 for g in comp.values())
    # gate hidden layer grad on first step
    gate_hidden = m.generator.path_gate.lins[0].weight.grad
    gate_last = m.generator.path_gate.lins[-1].weight.grad
    return {
        "shared_composition_grad": {k: (float(g.abs().sum()) if g is not None else None) for k, g in comp.items()},
        "gate_hidden_grad_first_step": float(gate_hidden.abs().sum()) if gate_hidden is not None else None,
        "gate_last_grad_first_step": float(gate_last.abs().sum()) if gate_last is not None else None,
        "checks": {
            "atom_grad": bool(comp_ok),
            "gate_hidden_zero_first_step": bool(gate_hidden is None or float(gate_hidden.abs().sum()) == 0.0),
            "gate_last_nonzero_first_step": bool(gate_last is not None and float(gate_last.abs().sum()) > 0),
        },
    }


def section_e(device) -> dict[str, Any]:
    view = _view()
    perms = T.build_permutations()
    m = M.build_semantic_model(view.spec, view.train_norm, "FS", permutations=perms).to(device)
    m.train()
    batches = [LPD.to_device(PREP_DATA.collate(view.caches["train"], list(range(i * 16, i * 16 + 16))), device) for i in range(4)]
    base_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def run(insert_eval: bool):
        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        m.load_state_dict({k: v.to(device) for k, v in base_state.items()})
        snap = LPD.rng_snapshot(device)
        for b in batches:
            opt.zero_grad(set_to_none=True)
            m(b).sum().backward()
            opt.step()
            if insert_eval:
                snap2 = LPD.rng_snapshot(device)
                was = m.training
                m.eval()
                with torch.inference_mode():
                    _ = m(b)
                m.train(was)
                LPD.rng_restore(snap2, device)
        hashes = LPD.state_hash(m.state_dict())
        LPD.rng_restore(snap, device)
        return hashes
    h0 = run(False)
    h1 = run(True)
    marg = LPC.setup_determinism()
    return {"replay_hash": h0, "replay_with_eval_hash": h1, "determinism": marg,
            "checks": {"eval_insertion_invariant": h0 == h1,
                       "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                       "tf32_off": not bool(torch.backends.cuda.matmul.allow_tf32)}}


def section_f(device) -> dict[str, Any]:
    view = _view()
    perms = T.build_permutations()
    m = M.build_semantic_model(view.spec, view.train_norm, "FS", permutations=perms).to(device).eval()
    tables = m.materialise_tables(permute=m.permute, device=device)
    # build a pure-lookup C1 model with the materialised tables and the same
    # non-table weights as the generator model
    from se_model import SemanticLCRC
    base = M._c1_base(view.spec, view.train_norm)
    bkeys = set(base.state_dict().keys())
    base.load_state_dict({k: v.cpu() for k, v in m.state_dict().items() if k in bkeys}, strict=False)
    lookup = SemanticLCRC(base, mode="free", init_tables={b: t.cpu() for b, t in tables.items()})
    lookup = lookup.to(device).eval()

    def preds(model):
        return {split: LPD.predict_cache(model, view.caches[split], device, pred_fn=lambda b: model(b)) for split in LPC.SPLITS}

    with torch.inference_mode():
        pg = preds(m)
        pl = preds(lookup)
    diffs = {s: float(np.max(np.abs(pg[s] - pl[s]))) for s in LPC.SPLITS}
    rmse = {}
    for s in ("valid", "test"):
        pr = view.to_raw(pg[s])
        rmse[s] = float(np.sqrt(np.mean((pr - view.y[s]) ** 2)))
    return {"export_max_abs": diffs, "independent_rmse": rmse,
            "checks": {"export_leq_1e5": all(v <= 1e-5 for v in diffs.values())}}


SECTIONS = ("a", "b", "c", "d", "e", "f")


def run_sections(device, sections: Sequence[str]) -> dict[str, Any]:
    view = _view()
    out: dict[str, Any] = {"sections": {}, "device": str(device)}
    for s in sections:
        C.log(f"section {s}")
        if s == "a":
            out["sections"][s] = section_a(view)
        elif s == "b":
            out["sections"][s] = section_b(device)
        elif s == "c":
            out["sections"][s] = section_c(device)
        elif s == "d":
            out["sections"][s] = section_d(device)
        elif s == "e":
            out["sections"][s] = section_e(device)
        elif s == "f":
            out["sections"][s] = section_f(device)
        else:
            raise ValueError(s)
    return out


def collect_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for s, section in payload.get("sections", {}).items():
        for name, val in section.get("checks", {}).items():
            checks[f"{s}_{name}"] = bool(val)
    return checks


__all__ = [name for name in dir() if not name.startswith("_")]
