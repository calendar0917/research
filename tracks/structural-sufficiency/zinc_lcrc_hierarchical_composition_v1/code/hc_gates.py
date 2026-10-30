"""CPU correctness gates for LCRC-HC3-v1.

Run:
    uv run python tracks/structural-sufficiency/zinc_lcrc_hierarchical_composition_v1/code/hc_gates.py

Writes ``analysis/gates/cpu_gates.json`` and raises on any failed check.  No
GPU, no labels in any forward pass, official test never constructed.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import hc_common as C
import hc_data as D
import hc_models as M
import hc_structure as HS

torch.set_grad_enabled(True)
RESULTS: dict = {"round": C.ROUND_ID, "created": C.now(), "checks": {}}


def check(group: str, name: str, ok: bool, detail) -> None:
    RESULTS["checks"].setdefault(group, {})[name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        raise AssertionError(f"{group}.{name} failed: {detail}")


def _one_mol_cache(mol, label=0.0):
    data = C.LD._concat_molecules([mol])
    data["labels"] = np.asarray([label], dtype=np.float32)
    data["split"] = "gate"
    data["n_molecules"] = 1
    data["meta"] = {"feature_version": C.LC.FEATURE_VERSION}
    return data


def _raw_molecule(cache, row):
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    b0, b1 = int(bp[row]), int(bp[row + 1])
    return dict(
        atom_feat=np.asarray(cache["atom_feat"])[a0:a1],
        atom_type=np.asarray(cache["atom_type"])[a0:a1],
        bond_index=np.asarray(cache["bond_index"])[b0:b1],
        bond_type=np.asarray(cache["bond_type"])[b0:b1],
        bond_feat=np.asarray(cache["bond_feat"])[b0:b1],
        global_feat=np.asarray(cache["global"])[row],
    )


def _build_mol(raw, *, perm=None, reverse_bonds=False, bond_order=None):
    bi = raw["bond_index"]
    bt = raw["bond_type"]
    bf = raw["bond_feat"]
    if bond_order is not None:
        bi = bi[bond_order]
        bt = bt[bond_order]
        bf = bf[bond_order]
    return C.LD.build_from_arrays(
        raw["atom_feat"], raw["atom_type"], bi, bt, bf, raw["global_feat"],
        perm=perm, reverse_bonds=reverse_bonds,
    )


def _predict_mol(model, mol, norm, device="cpu", batch_size=64):
    cache = _one_mol_cache(mol)
    hs = HS.build_split(cache, "train")
    ds = D.SplitData("train", cache=cache, struct=hs, device=device)
    return D.predict(model, ds, [0], batch_size=batch_size)


def _shuffled_struct(struct, rng):
    keys = ["z_atom", "z_bond", "z_chord", "path_atom", "path_bonds", "path_bond",
            "path_chord", "path_z"]
    out = dict(struct)
    for k in keys:
        out[k] = np.asarray(struct[k]).copy()
    z_ptr = np.asarray(struct["z_ptr"])
    p_ptr = np.asarray(struct["path_ptr"])
    for m in range(len(z_ptr) - 1):
        z0, z1 = int(z_ptr[m]), int(z_ptr[m + 1])
        nz = z1 - z0
        if nz:
            perm = rng.permutation(nz)
            inv = np.empty(nz, dtype=np.int64)
            inv[perm] = np.arange(nz)
            for k in ("z_atom", "z_bond", "z_chord"):
                out[k][z0:z1] = np.asarray(struct[k])[z0:z1][perm]
        p0, p1 = int(p_ptr[m]), int(p_ptr[m + 1])
        npth = p1 - p0
        if npth:
            perm = rng.permutation(npth)
            for k in ("path_atom", "path_bonds", "path_bond", "path_chord"):
                out[k][p0:p1] = np.asarray(struct[k])[p0:p1][perm]
            pz = np.asarray(struct["path_z"])[p0:p1][perm]
            out["path_z"][p0:p1] = pz + z0
            out["path_z"][p0:p1] -= z0
            out["path_z"][p0:p1] = inv[pz]
    return out


def gate_parameters():
    tr = C.load_cache("train")
    split = C.split_rows()
    norm = C.fit8k_norm(tr, split["rows"]["fit8k"])
    table = {}
    models = {}
    for kind in M.KINDS:
        model, info = M.build_model(kind, norm, 0, "cpu")
        models[kind] = model
        table[kind] = {"parameters": info["parameters"],
                       "breakdown": info["parameters_breakdown"]}
    exp = {"B": 237_519, "H": 323_391, "F": 323_384}
    check("g1_parameter_table", "counts", all(table[k]["parameters"] == exp[k] for k in exp),
          {"table": table, "expected": exp})
    check("g1_parameter_table", "H_minus_F_7",
          table["H"]["parameters"] - table["F"]["parameters"] == 7,
          table["H"]["parameters"] - table["F"]["parameters"])
    check("g1_parameter_table", "additions",
          table["H"]["breakdown"]["comp_proj"] == 4160
          and table["F"]["breakdown"]["comp_proj"] == 4160
          and table["H"]["breakdown"]["F2"] + table["H"]["breakdown"]["F3"] == 81_712
          and table["F"]["breakdown"]["Fflat"] == 81_705,
          {"H": table["H"]["breakdown"], "F": table["F"]["breakdown"]})
    check("g1_parameter_table", "cap", all(v["parameters"] <= C.PARAM_CAP for v in table.values()),
          C.PARAM_CAP)
    return models, norm


def gate_base_state(models, norm):
    out = {}
    for seed in (0, 1):
        full_norm = C.load_full_norm()
        fb, fib = C.LM.build_model("A", full_norm, seed, "cpu")
        h, hi = M.build_model("H", full_norm, seed, "cpu")
        f, fi = M.build_model("F", full_norm, seed, "cpu")
        sb, sh, sf = fb.state_dict(), h.state_dict(), f.state_dict()
        keys_ok_h = all(torch.equal(sb[k], sh[k]) for k in sb)
        keys_ok_f = all(torch.equal(sb[k], sf[k]) for k in sb)
        hist = C.read_json(C.LCRC / "runs" / f"lcrcA_s{seed}" / "train.json")
        hist_hash = hist["model_info"]["initial_state_sha256"]
        out[seed] = {
            "H_base_matches_frozen_tensorwise": keys_ok_h,
            "F_base_matches_frozen_tensorwise": keys_ok_f,
            "B_hash_matches_historical": fib["initial_state_sha256"] == hist_hash,
            "H_base_hash_equals_B": hi["base_state_sha256"] == fib["initial_state_sha256"],
            "F_base_hash_equals_B": fi["base_state_sha256"] == fib["initial_state_sha256"],
            "comp_proj_H_equals_F": torch.equal(h.comp_proj.weight, f.comp_proj.weight)
            and torch.equal(h.comp_proj.bias, f.comp_proj.bias),
            "comp_proj_zero": bool(torch.count_nonzero(h.comp_proj.weight) == 0
                                   and torch.count_nonzero(h.comp_proj.bias) == 0),
            "hist_hash": hist_hash,
        }
    ok = all(v["H_base_matches_frozen_tensorwise"] and v["F_base_matches_frozen_tensorwise"]
             and v["B_hash_matches_historical"] and v["H_base_hash_equals_B"]
             and v["F_base_hash_equals_B"] and v["comp_proj_H_equals_F"]
             and v["comp_proj_zero"] for v in out.values())
    check("g2_initial_state", "tensorwise_vs_frozen", ok, out)


def gate_step0(models, norm):
    tr = C.load_cache("train")
    split = C.split_rows()
    rng = np.random.default_rng(7)
    rows = split["rows"]["fit8k"][rng.choice(8000, 96, replace=False)]
    ds = D.SplitData("train", cache=tr)
    batch = ds.batch(rows)
    out = {}
    with torch.no_grad():
        for seed in (0, 1):
            b, _ = M.build_model("B", norm, seed, "cpu")
            h, _ = M.build_model("H", norm, seed, "cpu")
            f, _ = M.build_model("F", norm, seed, "cpu")
            b.eval(); h.eval(); f.eval()
            yb = b(batch); yh = h(batch); yf = f(batch)
            out[seed] = {
                "H_minus_B_maxabs": float((yh - yb).abs().max()),
                "F_minus_B_maxabs": float((yf - yb).abs().max()),
                "H_bitwise": bool(torch.equal(yh, yb)),
                "F_bitwise": bool(torch.equal(yf, yb)),
            }
    ok = all(v["H_bitwise"] and v["F_bitwise"] for v in out.values())
    check("g3_step0_forward", "strict_reproduction", ok, out)


def gate_structure():
    # hand-built graphs: chain, star, 5-cycle, fused bicyclic, chord triangle
    hand = [
        (4, [(0, 1, 0), (1, 2, 0), (2, 3, 1)]),
        (5, [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 0)]),
        (5, [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 2), (4, 0, 0)]),
        (6, [(0, 1, 0), (1, 2, 0), (2, 0, 1), (2, 3, 0), (3, 4, 1), (4, 2, 2)]),
        (4, [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 2), (0, 2, 0)]),
    ]
    tr = C.load_cache("train")
    va = C.load_cache("valid")
    mols = hand + [HS._molecules(tr)[i] for i in (0, 1, 2, 3, 17, 42, 101, 707, 9999)] \
        + [HS._molecules(va)[i] for i in (0, 3, 11, 201, 999)]
    mismatches = []
    for n_atoms, bonds in mols:
        obj = HS.enumerate_molecule(n_atoms, bonds)
        ref = HS.brute_force_directed_paths(n_atoms, bonds)
        got = set(map(tuple, np.asarray(obj["path_atom"]).tolist()))
        ref_set = set((i, j, k, l) for i, j, k, l, _c1, _c2, _c3 in ref)
        if got != ref_set:
            mismatches.append({"n": n_atoms, "kind": "path_set"})
        # reverse path present
        for i, j, k, l in got:
            if (l, k, j, i) not in got:
                mismatches.append({"path": (i, j, k, l), "kind": "reverse"})
                break
        # chords by record
        by_key = {}
        for row, key in enumerate(map(tuple, np.asarray(obj["path_atom"]).tolist())):
            by_key[key] = tuple(np.asarray(obj["path_chord"])[row].tolist())
        for i, j, k, l, c_ik, c_jl, c_il in ref:
            if by_key[(i, j, k, l)] != (c_ik, c_jl, c_il):
                mismatches.append({"path": (i, j, k, l), "kind": "chord"})
                break
        # n_e == directed/2 per bond, and centre bond of each record
        counts = np.zeros(len(bonds), dtype=np.int64)
        for row, key in enumerate(map(tuple, np.asarray(obj["path_atom"]).tolist())):
            e = int(np.asarray(obj["path_bonds"])[row, 1])
            counts[e] += 1
            # the centre bond must be the unique edge {j,k}
            j, k = key[1], key[2]
            e_ref = [eid for eid, (u, v, _t) in enumerate(bonds)
                     if {int(u), int(v)} == {j, k}]
            if e_ref != [e]:
                mismatches.append({"path": key, "kind": "centre_bond"})
                break
        if not np.array_equal(counts // 2, np.asarray(obj["bond_npaths"])):
            mismatches.append({"n": n_atoms, "kind": "n_e"})
        if not np.all(counts % 2 == 0):
            mismatches.append({"n": n_atoms, "kind": "odd"})
    check("g4_enumeration", "brute_force", not mismatches,
          {"n_graphs": len(mols), "mismatches": mismatches[:10]})


def gate_cma_equivalence():
    old_round = C.SUFF / "zinc_compositional_research_tree_v1"
    sys.path.insert(0, str(old_round / "cma"))
    sys.path.insert(0, str(old_round / "shared"))
    try:
        import cma_structure as CS
    except Exception as exc:  # pragma: no cover
        check("g5_cma_semantics", "import", False, repr(exc))
        return
    tr = C.load_cache("train")
    mols = HS._molecules(tr)[:60]
    bad = []
    for n_atoms, bonds in mols:
        ours = HS.enumerate_molecule(n_atoms, bonds)
        ref = CS.enumerate_molecule(n_atoms, bonds)
        # z triples and their chord / bond rows
        ours_z = {(int(j), int(a), int(b)): (int(c), tuple(int(x) for x in bo))
                  for (j, a, b), c, bo in zip(ours["z_atom"], ours["z_chord"], ours["z_bond"])}
        ref_z = {(int(j), int(a), int(b)): (int(c), tuple(int(x) for x in bo))
                 for (j, a, b), c, bo in zip(ref["z_atom"], ref["z_chord"], ref["z_bond"])}
        if ours_z != ref_z:
            bad.append({"n": n_atoms, "kind": "z"})
            continue
        # Build the reference records in the same canonical atom order using
        # the brute-force record list and CMA z lookup.
        ref_by_key = {}
        for row, (i, j, k, l, c_ik, c_jl, c_il) in enumerate(
                HS.brute_force_directed_paths(n_atoms, bonds)):
            zl = (j, min(i, k), max(i, k))
            zr = (k, min(j, l), max(j, l))
            e = [eid for eid, (u, v, _t) in enumerate(bonds) if {int(u), int(v)} == {j, k}]
            if len(e) != 1:
                continue
            ref_by_key[(i, j, k, l)] = (
                (zr[0], zr[1], zr[2]), (zl[0], zl[1], zl[2]), e[0],
                (c_ik, c_jl, c_il),
            )
        ours_key = {}
        for row in range(len(ours["path_atom"])):
            i, j, k, l = map(int, ours["path_atom"][row])
            zl = tuple(int(x) for x in ours["z_atom"][ours["path_z"][row, 0]])
            zr = tuple(int(x) for x in ours["z_atom"][ours["path_z"][row, 1]])
            ours_key[(i, j, k, l)] = (zr, zl, int(ours["path_bonds"][row, 1]),
                                      tuple(int(x) for x in ours["path_chord"][row]))
        if ours_key != ref_by_key:
            bad.append({"n": n_atoms, "kind": "path_z"})
            continue
        # CMA's own path_chord / path_bond records must be the same multiset.
        cma_set = sorted((int(ref["path_bond"][r]), tuple(int(x) for x in ref["path_chord"][r]))
                         for r in range(len(ref["path_bond"])))
        our_set = sorted((int(ours["path_bonds"][r, 1]),
                          tuple(int(x) for x in ours["path_chord"][r]))
                         for r in range(len(ours["path_bonds"])))
        if cma_set != our_set:
            bad.append({"n": n_atoms, "kind": "cma_path_attrs"})
        if not np.array_equal(np.asarray(ref["bond_npaths"]), np.asarray(ours["bond_npaths"])):
            bad.append({"n": n_atoms, "kind": "cma_n_e"})
    check("g5_cma_semantics", "sets_match", not bad,
          {"n_molecules": len(mols), "mismatches": bad[:10]})


def gate_learned_vs_secondhop():
    src = (C.SUFF / "zinc_secondhop_composition_pilot_v1/code/sh_models.py").read_text()
    old_fixed = ('register_buffer("P_b"' in src and 'register_buffer("P_a"' in src
                 and "requires_grad_(False)" in src)
    en2 = Path(C.SUFF / "zinc_secondhop_composition_pilot_v1/code/sh_common.py").read_text()
    old_two_hop = ("SECOND_HOP_DIM" in en2 and "PROJ_DIM" in en2)
    norms = C.fit8k_norm(C.load_cache("train"), C.split_rows()["rows"]["fit8k"])
    h, hi = M.build_model("H", norms, 0, "cpu")
    f, fi = M.build_model("F", norms, 0, "cpu")
    hq = sum(p.numel() for p in h.path_encoder.parameters())
    fq = sum(p.numel() for p in f.path_encoder.parameters())
    trainable = all(p.requires_grad for p in h.path_encoder.parameters()) and \
        all(p.requires_grad for p in f.path_encoder.parameters())
    tr = C.load_cache("train")
    raw = _raw_molecule(tr, 17)
    mol_cache = _one_mol_cache(_build_mol(raw))
    obj = HS.build_split(mol_cache, "train")
    four_distinct = bool(np.all(np.diff(np.sort(np.asarray(obj["path_atom"]), axis=1),
                                       axis=1) > 0))
    three_bonds = np.asarray(obj["path_bonds"]).shape[1] == 3
    # q_path is actually used by a learned nonlinear module: perturbing F3/Fflat
    # changes the path embedding, while the old fixed projections have no
    # parameters at all.
    with torch.no_grad():
        cache = mol_cache
        hs = obj
        ds = D.SplitData("train", cache=cache, struct=hs)
        batch = ds.batch([0])
        a = h.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        b = h.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], 1))
        q0h = h.path_encoder(batch, a, b)
        q0f = f.path_encoder(batch, a, b)
        for p in h.path_encoder.F3.parameters():
            p.add_(0.01)
        for p in f.path_encoder.Fflat.parameters():
            p.add_(0.01)
        q1h = h.path_encoder(batch, a, b)
        q1f = f.path_encoder(batch, a, b)
        dh = float((q1h - q0h).abs().max())
        df = float((q1f - q0f).abs().max())
    check("g6_learned_not_fixed_secondhop", "contrast",
          old_fixed and old_two_hop and trainable and four_distinct and three_bonds
          and hq == 81_712 and fq == 81_705 and dh > 0 and df > 0,
          {"old_fixed_projection": old_fixed, "old_two_hop_names": old_two_hop,
           "new_trainable": trainable, "four_distinct_atoms": four_distinct,
           "three_bonds": three_bonds, "H_path_params": hq, "F_path_params": fq,
           "q_perturbation_H": dh, "q_perturbation_F": df,
           "q_dim": int(q0h.shape[1])})


def gate_shared_content(models, norm):
    tr = C.load_cache("train")
    split = C.split_rows()
    rows = split["rows"]["fit8k"][:16]
    ds = D.SplitData("train", cache=tr)
    b = ds.batch(rows)
    h, _ = M.build_model("H", norm, 0, "cpu")
    f, _ = M.build_model("F", norm, 0, "cpu")
    bm, _ = M.build_model("B", norm, 0, "cpu")
    with torch.no_grad():
        ah, bh, _ = h._encode(b)
        af, bf, _ = f._encode(b)
        ab, bb, _ = bm._encode(b)
    same = {
        "a_H_eq_F": bool(torch.equal(ah, af)),
        "b_H_eq_F": bool(torch.equal(bh, bf)),
        "a_eq_B": bool(torch.equal(ah, ab)),
        "b_eq_B": bool(torch.equal(bh, bb)),
    }
    # raw path content: four atoms distinct, three bond rows, chord in 0..3
    atoms = b["hc_path_atom"]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    distinct = all(bool((atoms[:, i] != atoms[:, j]).all()) for i, j in pairs)
    chords_ok = bool(b["hc_path_chord"].min() >= 0 and b["hc_path_chord"].max() <= 3)
    check("g7_shared_content", "atom_bond_and_paths",
          all(same.values()) and distinct and chords_ok,
          {**same, "four_atoms_distinct": distinct, "chord_range_0_3": chords_ok})


def gate_gradients(norm):
    tr = C.load_cache("train")
    split = C.split_rows()
    rows = split["rows"]["fit8k"][:64]
    ds = D.SplitData("train", cache=tr)
    batch = ds.batch(rows)
    target = ds.target_tensor(rows, "cpu")
    out = {}
    for kind in ("H", "F"):
        model, _ = M.build_model(kind, norm, 0, "cpu")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        steps = []
        for step in range(3):
            pred = model(batch)
            loss = torch.nn.functional.l1_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            groups = {}
            for name, p in model.named_parameters():
                if not ("path_encoder" in name or "comp_proj" in name):
                    continue
                key = ".".join(name.split(".")[:2])
                groups[key] = groups.get(key, 0.0) + float(
                    (p.grad.detach() ** 2).sum()) if p.grad is not None else groups.get(key, 0.0)
            groups = {k: math.sqrt(v) for k, v in groups.items()}
            steps.append({"step": step, "loss": float(loss), "grad": groups})
            opt.step()
        comp0 = steps[0]["grad"].get("comp_proj.weight", 0.0)
        path0 = {k: v for k, v in steps[0]["grad"].items() if k.startswith("path_encoder")}
        path1 = {k: v for k, v in steps[1]["grad"].items() if k.startswith("path_encoder")}
        out[kind] = {
            "comp_grad_step0": comp0,
            "path_grad_step0": path0,
            "path_grad_step1": path1,
            "ok": comp0 > 0 and all(v == 0.0 for v in path0.values())
            and all(v > 0 for v in path1.values())
            and all(math.isfinite(v) for v in path1.values()),
        }
    check("g8_gradient_path", "injection_then_encoder", all(v["ok"] for v in out.values()), out)


def gate_invariance(models, norm):
    tr = C.load_cache("train")
    split = C.split_rows()
    rows = split["rows"]["fit8k"][[0, 1, 2, 3, 17, 42, 101, 707]]
    ds = D.SplitData("train", cache=tr)
    rng = np.random.default_rng(11)
    out = {}
    for kind in ("B", "H", "F"):
        model, _ = M.build_model(kind, norm, 0, "cpu")
        model.eval()
        base = D.predict(model, ds, rows, batch_size=8)
        # batch split / merge
        chunks = np.concatenate([D.predict(model, ds, rows[i:i + 3], batch_size=3)
                                 for i in range(0, len(rows), 3)])
        d_batch = float(np.max(np.abs(base - chunks)))
        # atom renumbering, bond reversal, bond-order permutation (one molecule)
        raw = _raw_molecule(tr, int(rows[4]))
        perm = rng.permutation(raw["atom_feat"].shape[0])
        bond_order = rng.permutation(raw["bond_index"].shape[0])
        preds = {
            "orig": _predict_mol(model, _build_mol(raw), norm),
            "perm": _predict_mol(model, _build_mol(raw, perm=perm), norm),
            "reversed": _predict_mol(model, _build_mol(raw, reverse_bonds=True), norm),
            "bond_order": _predict_mol(model, _build_mol(raw, bond_order=bond_order), norm),
        }
        deltas = {k: float(np.max(np.abs(v - preds["orig"]))) for k, v in preds.items()}
        # path enumeration order (struct rows shuffled, references remapped)
        hs = HS.load_split("train", cache=tr)
        sh = _shuffled_struct(hs, np.random.default_rng(23))
        ds_sh = D.SplitData("train", cache=tr, struct=sh)
        pred_sh = D.predict(model, ds_sh, rows, batch_size=8)
        d_sh = float(np.max(np.abs(base - pred_sh)))
        out[kind] = {"batch_split_maxabs": d_batch, "renumber_reverse_bondorder": deltas,
                     "path_order_maxabs": d_sh}
    ok = all(v["batch_split_maxabs"] <= 3e-5 and v["path_order_maxabs"] <= 3e-5
             and max(v["renumber_reverse_bondorder"].values()) <= 3e-5 for v in out.values())
    check("g9_invariance", "renumber/reverse/reorder/split", ok, out)


def gate_reversal_aggregation(norm):
    tr = C.load_cache("train")
    rows = C.split_rows()["rows"]["fit8k"][:8]
    ds = D.SplitData("train", cache=tr)
    batch = ds.batch(rows)
    out = {}
    for kind in ("H", "F"):
        model, _ = M.build_model(kind, norm, 0, "cpu")
        with torch.no_grad():
            a = model.atom_encoder(batch["atom_feat"])
            idx = batch["bond_index"]
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            b = model.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av,
                                              batch["bond_feat"]], dim=1))
            q = model.path_encoder(batch, a, b)
            m = model.bond_m(batch, q)
            n_e = batch["hc_event_n"]
            zero_ok = bool(torch.count_nonzero(m[n_e == 0]) == 0)
            # match reverse path records within the first molecule
            atoms = batch["hc_path_atom"].cpu().numpy()
            pz = batch["hc_path_z"].cpu().numpy()
            # invert per-molecule path atom tuple -> row
            index = {}
            for r in range(len(atoms)):
                index[tuple(atoms[r])] = r
            diffs = []
            for r in range(len(atoms)):
                i, j, k, l = map(int, atoms[r])
                rr = index.get((l, k, j, i))
                if rr is None:
                    continue
                diffs.append(float((q[r] - q[rr]).abs().max()))
            out[kind] = {
                "n_e_zero_rows_strict_zero": zero_ok,
                "n_reverse_pairs": len(diffs),
                "max_forward_reverse_q_diff": max(diffs) if diffs else None,
            }
    ok = all(v["n_e_zero_rows_strict_zero"] and v["n_reverse_pairs"] > 0
             and (v["max_forward_reverse_q_diff"] or 0.0) <= 1e-6 for v in out.values())
    check("g10_reversal_and_zero", "q_reverse_and_n_e_zero", ok, out)


def gate_labels_and_boundary(norm):
    tr = C.load_cache("train")
    split = C.split_rows()
    rows = split["rows"]["fit8k"][:16]
    ds = D.SplitData("train", cache=tr)
    model, _ = M.build_model("H", norm, 0, "cpu")
    model.eval()
    pred0 = D.predict(model, ds, rows, batch_size=16)
    mutated = dict(tr)
    mutated["labels"] = np.asarray(tr["labels"]).copy()
    mutated["labels"][rows] += 123.0
    ds_mut = D.SplitData("train", cache=mutated, struct=ds.struct)
    pred1 = D.predict(model, ds_mut, rows, batch_size=16)
    batch = ds.batch(rows)
    keys = sorted(batch.keys())
    forbidden = [k for k in keys if "label" in k.lower() or k in ("y", "target")]
    refused = {}
    for fn, arg in ((C.load_cache, "test"), (C.cache_path, "test"),
                    (HS.load_split, "test"), (C.LCRC, None)):
        if arg is None:
            continue
        try:
            fn(arg)
            refused[str(getattr(fn, "__name__", fn))] = False
        except Exception:
            refused[str(getattr(fn, "__name__", fn))] = True
    check("g11_label_and_boundary", "no_label_leak_test_refused",
          bool(np.max(np.abs(pred0 - pred1)) == 0.0) and not forbidden
          and all(refused.values()),
          {"prediction_unchanged": bool(np.max(np.abs(pred0 - pred1)) == 0.0),
           "forbidden_keys": forbidden, "refused": refused, "n_keys": len(keys)})


def gate_recipe_and_resume(norm):
    # LR schedule
    peak, floor = C.RECIPE["lr"], C.RECIPE["min_lr"]
    lr = [C.cosine_with_warmup_lr(s, 47_400, 500, peak, floor)
          for s in (0, 1, 250, 499, 500, 501, 23700, 47_400)]
    warm_ok = (lr[0] == 0.0 and abs(lr[1] - peak / 500) < 1e-15
               and abs(lr[3] - peak * 499 / 500) < 1e-12
               and abs(lr[4] - peak) < 1e-12)
    end_ok = abs(lr[-1] - floor) < 1e-12
    mono = all(lr[i] >= lr[i + 1] for i in range(4, len(lr) - 1))
    # parameter/optimizer audit + checkpoint round-trip
    model, _ = M.build_model("H", norm, 0, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    audit_ok = C.parameter_count(model) == sum(
        p.numel() for g in opt.param_groups for p in g["params"]) == 323_391
    tr = C.load_cache("train")
    rows = C.split_rows()["rows"]["fit8k"][:8]
    ds = D.SplitData("train", cache=tr)
    p0 = D.predict(model, ds, rows)
    state = C.cpu_state(model)
    fresh, _ = M.build_model("H", norm, 0, "cpu")
    C.load_state(fresh, state, "cpu")
    p1 = D.predict(fresh, ds, rows)
    roundtrip = bool(np.array_equal(p0, p1))

    # resume: 63 steps, then resume to 126, versus uninterrupted 126
    base = C.RUNS / "cpu_gate_resume"
    if base.exists():
        shutil.rmtree(base)
    C.ensure_dir(base)
    try:
        import hc_train as T
        T.train("H", 0, "fit8k", "cpu", total_updates=126, stop_after_steps=63,
                force=True, tag="cpu_gate_resume/seg")
        hs1 = C.sha256_file(base / "seg" / "last_state.pt")
        T.train("H", 0, "fit8k", "cpu", total_updates=126, resume=True,
                force=True, tag="cpu_gate_resume/seg")
        h_res = C.read_json(base / "seg" / "train.json")
        T.train("H", 0, "fit8k", "cpu", total_updates=126, force=True,
                tag="cpu_gate_resume/full")
        h_full = C.read_json(base / "full" / "train.json")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    resume_ok = (h_res["initial_state_sha256"] == h_full["initial_state_sha256"]
                 and h_res["reads"]["final"]["state_hash"] == h_full["reads"]["final"]["state_hash"]
                 and abs(h_res["curve"][-1]["monitor_mae"]
                         - h_full["curve"][-1]["monitor_mae"]) < 1e-9
                 and len(h_res["curve"]) == len(h_full["curve"]) == 2
                 and h_res["total_updates"] == h_full["total_updates"] == 126)
    check("g12_recipe_and_checkpoint", "lr_optimizer_roundtrip_resume",
          warm_ok and end_ok and mono and audit_ok and roundtrip and resume_ok,
          {"lr": lr, "warm_ok": warm_ok, "end_ok": end_ok, "mono": mono,
           "audit_ok": audit_ok, "roundtrip": roundtrip, "resume_ok": resume_ok,
           "resume_final_hash": h_res["reads"]["final"]["state_hash"],
           "full_final_hash": h_full["reads"]["final"]["state_hash"]})


def gate_norm_and_hashes():
    tr = C.load_cache("train")
    split = C.split_rows()
    C.verify_split_hashes(split["rows"])
    fit = C.fit8k_norm(tr, split["rows"]["fit8k"])
    mon = C.fit8k_norm(tr, split["rows"]["monitor"])
    fit_ok = not np.allclose(fit["global_mean"], mon["global_mean"]) and \
        not np.allclose(fit["scalar_mean"], mon["scalar_mean"])
    full = C.load_full_norm()
    hist = C.read_json(C.LCRC / "runs" / "lcrcA_s0" / "train.json")
    full_ok = C.norm_fingerprint(full) == hist["norm"]["fingerprint"]
    lab = C.label_hashes(tr, split["rows"])
    hist_lab = C.read_json(C.SUFF / "zinc_compositional_research_tree_v1/analysis/frozen_baseline.json")
    labels_ok = lab["full_labels_sha256"] == hist_lab["data_boundary"]["label_hashes"]["full_labels_sha256"]
    check("g13_data_and_norm", "fit_only_and_historical", fit_ok and full_ok and labels_ok,
          {"fit8k_norm_differs_from_monitor": fit_ok, "full_norm_fingerprint_matches_history": full_ok,
           "label_hash_matches_history": labels_ok, "norm_fingerprint": C.norm_fingerprint(full)})


def gate_n_e_nonzero_stats():
    st = HS.load_split("train")
    n = np.asarray(st["bond_npaths"])
    check("g14_path_coverage", "n_e_distribution",
          n.min() == 0 and n.max() >= 2 and int((n > 0).sum()) > 0,
          {"n_bonds": int(len(n)), "n_e_zero": int((n == 0).sum()),
           "n_e_gt0": int((n > 0).sum()), "max_n_e": int(n.max())})


def gate_historical_baseline_replay():
    """Stage C baseline: replay the historical LCRC-A train10k valid predictions."""
    norm = C.load_full_norm()
    tr = C.load_cache("train")
    va = C.load_cache("valid")
    normalised = C.fit8k_norm(tr, C.split_rows()["rows"]["fit8k"])  # unused; keeps path warm
    out = {}
    for seed in (0, 1):
        run = C.LCRC / "runs" / f"lcrcA_s{seed}"
        hist = C.read_json(run / "train.json")
        state = C.atomic_torch_load(run / "best.pt")["state_dict"]
        model, _ = M.build_model("B", norm, seed, "cpu")
        C.load_state(model, state, "cpu")
        ds = D.SplitData("valid", cache=va, device="cpu")
        pred = D.predict(model, ds, np.arange(1000), batch_size=256)
        saved = np.load(run / "predictions" / "valid_best.npz")
        order = np.argsort(np.asarray(saved["row"], dtype=np.int64))
        saved_pred = np.asarray(saved["pred"], dtype=np.float64)[order]
        maxdiff = float(np.max(np.abs(pred.astype(np.float64) - saved_pred)))
        mae = D.mae(pred, np.asarray(va["labels"], dtype=np.float64))
        out[seed] = {
            "checkpoint": str(run / "best.pt"),
            "checkpoint_sha256": C.sha256_file(run / "best.pt"),
            "historical_best_valid_mae": hist["best"]["valid_mae"],
            "replayed_valid_mae": mae,
            "maxdiff_vs_saved_prediction": maxdiff,
            "cache_content_sha256": C.sha256_file(C.cache_path("valid")),
            "label_sha256": C.sha256_array(np.asarray(va["labels"], dtype=np.float64), "<f8"),
        }
    ok = all(v["maxdiff_vs_saved_prediction"] <= 1e-5
             and abs(v["replayed_valid_mae"] - v["historical_best_valid_mae"]) <= 2e-6
             for v in out.values())
    check("g15_stageC_baseline_replay", "historical_lcrcA_valid", ok, out)


def main():
    C.ensure_dir(C.GATES)
    models, norm = gate_parameters()
    gate_base_state(models, norm)
    gate_step0(models, norm)
    gate_structure()
    gate_cma_equivalence()
    gate_learned_vs_secondhop()
    gate_shared_content(models, norm)
    gate_gradients(norm)
    gate_invariance(models, norm)
    gate_reversal_aggregation(norm)
    gate_labels_and_boundary(norm)
    gate_recipe_and_resume(norm)
    gate_norm_and_hashes()
    gate_n_e_nonzero_stats()
    gate_historical_baseline_replay()
    RESULTS["all_pass"] = all(v["pass"] for g in RESULTS["checks"].values()
                              for v in g.values())
    RESULTS["git_commit"] = C.git_commit()
    RESULTS["environment"] = {"python": sys.version.split()[0], "torch": torch.__version__}
    C.write_json(C.GATES / "cpu_gates.json", RESULTS)
    C.log(json.dumps({"all_pass": RESULTS["all_pass"],
                      "groups": {g: all(v["pass"] for v in vals.values())
                                 for g, vals in RESULTS["checks"].items()}}, indent=1))
    if not RESULTS["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
