"""Engineering and mathematics checks for the word-response-operator s0.

Only the checks the task book asks for: structure audit on 16 real molecules,
atom-space vs boundary-assembly equivalence (FP64/FP32), batch-split invariance,
shared initialisation, gradient health on a small subset with a finite-difference
spot check, and the parameter-count width selection for the atomic control.
"""

from __future__ import annotations

import itertools
import sys
from typing import Any, Mapping, Sequence

import numpy as np

import wo_common as C
import wo_data as D
import wo_model as M
import wo_struct as S
import wo_train as T


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def to_numpy(packed: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    out = {}
    for key, value in packed.items():
        out[key] = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    return out


def model_terms(model, packed: Mapping[str, Any], dtype):
    import torch

    cast = {}
    for key, value in packed.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            cast[key] = value.to(dtype)
        else:
            cast[key] = value
    with torch.no_grad():
        base = model.atom_bond_pair(cast)
        bval = model.word_source(cast, base["bbase"])
    out = {"d": base["d"].detach().cpu().numpy().astype(np.float64),
           "k": base["k"].detach().cpu().numpy().astype(np.float64),
           "b": bval.detach().cpu().numpy().astype(np.float64),
           "lambda": None}
    if base["lambda"] is not None:
        out["lambda"] = base["lambda"].detach().cpu().numpy().astype(np.float64)
    return out


def ref_full(npk, terms, m: int, channel: int) -> float:
    n = int(npk["n_atoms"][m])
    d, k, lam, b = terms["d"], terms["k"], terms["lambda"], terms["b"]
    Q = np.zeros((n, n))
    Q[np.arange(n), np.arange(n)] = d[m, :n, channel]
    sel = npk["bond_mol"] == m
    for u, v, kk in zip(npk["bond_u_row"][sel], npk["bond_v_row"][sel], k[sel, channel]):
        Q[u, u] += kk
        Q[v, v] += kk
        Q[u, v] -= kk
        Q[v, u] -= kk
    if lam is not None:
        sel = npk["pair_mol"] == m
        for u, v, ll in zip(npk["pair_u_row"][sel], npk["pair_v_row"][sel], lam[sel, channel]):
            Q[u, u] += ll
            Q[v, v] += ll
            Q[u, v] -= ll
            Q[v, u] -= ll
    sign, logdet = np.linalg.slogdet(Q)
    if sign <= 0:
        raise ValueError("reference full matrix is not positive definite")
    bvec = b[m, :n, channel]
    return 0.5 * logdet - 0.5 * float(bvec @ np.linalg.solve(Q, bvec))


def _word_matrices(npk, terms, m: int, channel: int):
    d, k, lam = terms["d"], terms["k"], terms["lambda"]
    words = np.where(npk["word_mol"] == m)[0]
    out = []
    for w in words:
        n_w = int(npk["word_n_atoms"][w])
        sites = npk["word_atoms_row"][w][:n_w]
        Q = np.zeros((n_w, n_w))
        Q[np.arange(n_w), np.arange(n_w)] = d[m, sites, channel]
        sel = npk["bond_word_u"] == w
        sel &= npk["bond_word_v"] == w
        for u, v, kk in zip(npk["bond_site_u"][sel], npk["bond_site_v"][sel], k[sel, channel]):
            Q[u, u] += kk
            Q[v, v] += kk
            Q[u, v] -= kk
            Q[v, u] -= kk
        if lam is not None:
            sel = npk["pair_word"] == w
            for u, v, ll in zip(npk["pair_site_u"][sel], npk["pair_site_v"][sel],
                                lam[sel, channel]):
                Q[u, u] += ll
                Q[v, v] += ll
                Q[u, v] -= ll
                Q[v, u] -= ll
        bvec = terms["b"][m, sites, channel]
        bnd = npk["word_site_bnd_local"][w][:n_w]
        bnd_slots = np.full(n_w, -1, dtype=np.int64)
        for site in range(n_w):
            if bnd[site] >= 0:
                bnd_slots[bnd[site]] = site
        boundary = [int(x) for x in bnd_slots if x >= 0]
        internal = [s for s in range(n_w) if s not in boundary]
        # crossing diagonal for this word
        D = np.zeros(n_w)
        sel = npk["cross_word_u"] == w
        for local, kk in zip(npk["cross_local_u"][sel], k[npk["cross_bond_sel"]][sel, channel]):
            D[local] += kk
        sel = npk["cross_word_v"] == w
        for local, kk in zip(npk["cross_local_v"][sel], k[npk["cross_bond_sel"]][sel, channel]):
            D[local] += kk
        out.append({"w": int(w), "n": n_w, "sites": sites, "Q": Q, "b": bvec,
                    "boundary": boundary, "internal": internal, "D": D})
    return out


def _reduce(word):
    Q, b = word["Q"], word["b"]
    I, B = word["internal"], word["boundary"]
    if I:
        QII = Q[np.ix_(I, I)]
        if B:
            QIB = Q[np.ix_(I, B)]
            QBI = QIB.T
            QBB = Q[np.ix_(B, B)]
            sol = np.linalg.solve(QII, QIB)
            S = QBB - QBI @ sol
            h = b[B] - QBI @ np.linalg.solve(QII, b[I])
        else:
            S = np.zeros((0, 0))
            h = np.zeros(0)
        sign, logdet = np.linalg.slogdet(QII)
        if sign <= 0:
            raise ValueError("Q_II is not positive definite")
        c = 0.5 * logdet - 0.5 * float(b[I] @ np.linalg.solve(QII, b[I]))
    else:
        S = Q[np.ix_(B, B)] if B else np.zeros((0, 0))
        h = b[B] if B else np.zeros(0)
        c = 0.0
    return S, h, c


def ref_local(npk, terms, m: int, channel: int) -> float:
    total = 0.0
    for word in _word_matrices(npk, terms, m, channel):
        S, h, c = _reduce(word)
        if len(word["boundary"]) == 0:
            total += c
            continue
        Sd = S + np.diag(word["D"][:len(word["boundary"])])
        sign, logdet = np.linalg.slogdet(Sd)
        if sign <= 0:
            raise ValueError("local boundary matrix is not positive definite")
        total += c + 0.5 * logdet - 0.5 * float(h @ np.linalg.solve(Sd, h))
    return total


def ref_portdiag(npk, terms, m: int, channel: int) -> float:
    n_b = int(npk["n_boundary"][m])
    K = np.zeros((n_b, n_b))
    h = np.zeros(n_b)
    csum = 0.0
    for word in _word_matrices(npk, terms, m, channel):
        S, hw, c = _reduce(word)
        csum += c
        bnd = word["boundary"]
        start = int(npk["word_bnd_row"][word["w"]])
        S = np.diag(np.diag(S)) if S.size else S   # PORT_DIAG keeps only diag(S)
        for p in range(len(bnd)):
            K[start + p, start + p] += S[p, p]
            h[start + p] += hw[p]
    # crossing physical bonds
    sel = npk["cross_mol"] == m
    for su, sv, kk in zip(npk["cross_slot_u"][sel], npk["cross_slot_v"][sel],
                          terms["k"][npk["cross_bond_sel"]][sel, channel]):
        K[su, su] += kk
        K[sv, sv] += kk
        K[su, sv] -= kk
        K[sv, su] -= kk
    sign, logdet = np.linalg.slogdet(K)
    if sign <= 0:
        raise ValueError("port-diagonal boundary matrix is not positive definite")
    return csum + 0.5 * logdet - 0.5 * float(h @ np.linalg.solve(K, h))


# ---------------------------------------------------------------------------
# structural audit
# ---------------------------------------------------------------------------


def choose_molecules(fit: D.SplitData) -> list[int]:
    n_mol = fit.n_molecules
    n_bnd = fit.graph_n_boundary
    cross_counts = np.diff(fit.arrays["graph_ptr_cross"])
    comps = fit.arrays["graph_n_components"]
    chosen: list[int] = []
    buckets = {
        "no_boundary": np.where(n_bnd == 0)[0],
        "one_boundary": np.where(n_bnd == 1)[0],
        "multi_boundary": np.where(n_bnd >= 2)[0],
        "multi_component": np.where(comps > 1)[0],
        "many_cross": np.where(cross_counts >= 6)[0],
    }
    per = max(1, 16 // len(buckets))
    for name, ids in buckets.items():
        for x in ids[:per]:
            if int(x) not in chosen:
                chosen.append(int(x))
    for x in range(n_mol):
        if len(chosen) >= 16:
            break
        if int(x) not in chosen:
            chosen.append(int(x))
    return chosen[:16]


def structure_audit(data: Mapping[str, Any]) -> dict[str, Any]:
    fit = data["fit"]
    payload = C.dataset_cache(data["record"].get("dataset", "ogbg-mollipo"))
    ids = choose_molecules(fit)
    packed = to_numpy(fit.pack(np.asarray(ids)))
    issues = []
    rows = fit.row_ids[ids]
    for local, row in enumerate(rows.tolist()):
        n = int(packed["n_atoms"][local])
        a0 = int(payload["node_ptr"][row])
        raw_atoms = payload["atom_feat"][a0:a0 + n]
        atoms = packed["atom_cat"][local][:n]
        if not np.array_equal(atoms, raw_atoms):
            issues.append(f"row {row}: atom columns changed")
        b0 = int(payload["bond_ptr"][row])
        b1 = int(payload["bond_ptr"][row + 1])
        sel = packed["bond_mol"] == local
        got = packed["bond_cat"][sel]
        want = payload["bond_feat"][b0:b1]
        if got.shape != want.shape or not np.array_equal(got, want):
            issues.append(f"row {row}: bond columns changed")
        # every physical bond exactly once, split into internal and crossing
        cross = int(np.sum(sel & (packed["bond_word_u"] != packed["bond_word_v"])))
        if int(np.sum(sel)) != b1 - b0:
            issues.append(f"row {row}: physical bond count mismatch")
        # internal bonds of the words + crossing bonds partition the bond set
        internal = 0
        for w in np.where(packed["word_mol"] == local)[0]:
            internal += int(np.sum((packed["bond_word_u"] == w) & (packed["bond_word_v"] == w)))
        if internal + cross != int(np.sum(sel)):
            issues.append(f"row {row}: internal+cross != all bonds")
        # boundary slots are exactly the distinct crossing endpoints
        endpoints = set(packed["cross_u_row"][packed["cross_mol"] == local].tolist())
        endpoints |= set(packed["cross_v_row"][packed["cross_mol"] == local].tolist())
        if len(endpoints) != int(packed["n_boundary"][local]):
            issues.append(f"row {row}: boundary atom count mismatch")
        # batch offsets: atom rows are inside the molecule
        if sel.any() and (packed["bond_u_row"][sel].max() >= n
                          or packed["bond_v_row"][sel].max() >= n):
            issues.append(f"row {row}: bond row outside molecule")
        if packed["pair_u_row"].size:
            psel = packed["pair_mol"] == local
            if psel.any() and (packed["pair_u_row"][psel].max() >= n):
                issues.append(f"row {row}: pair row outside molecule")
    # boundary slot consistency: every crossing bond must land on the word slots
    start = packed["word_bnd_start"]
    for w in range(len(packed["word_ids"])):
        n_w = int(packed["word_n_atoms"][w])
        bnd = packed["word_site_bnd_local"][w][:n_w]
        for site in range(n_w):
            if bnd[site] < 0:
                continue
            expected = int(start[w]) + int(bnd[site])
            mol = int(packed["word_mol"][w])
            row = int(packed["word_atoms_row"][w][site])
            su = packed["cross_slot_u"][packed["cross_mol"] == mol]
            sv = packed["cross_slot_v"][packed["cross_mol"] == mol]
            cu = packed["cross_u_row"][packed["cross_mol"] == mol]
            cv = packed["cross_v_row"][packed["cross_mol"] == mol]
            slots = {int(x) for x, r in zip(su, cu) if int(r) == row}
            slots |= {int(x) for x, r in zip(sv, cv) if int(r) == row}
            if slots and slots != {expected}:
                issues.append(f"word {w} site {site}: boundary slot {sorted(slots)} != {expected}")
    return {"molecules": [int(r) for r in rows.tolist()], "issues": issues,
            "n_checked": len(ids), "ok": not issues}


# ---------------------------------------------------------------------------
# equivalence: atom-space vs boundary assembly, and the two variants
# ---------------------------------------------------------------------------


def equivalence(data: Mapping[str, Any], dtype_name: str = "float64") -> dict[str, Any]:
    import torch

    cfg = C.load_config()
    fit = data["fit"]
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    ids = choose_molecules(fit)
    models = {}
    base = None
    for arm in ("FULL_WORD_OP", "FULL_BOUNDARY_CHECK", "LOCAL_WORD_OP", "PORT_DIAG_WORD_OP"):
        model = T.build_model(cfg, data, arm).to(dtype)
        if base is None:
            base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(base)
        models[arm] = model
    packed = fit.pack(np.asarray(ids), device="cpu")
    cast = {k: (v.to(dtype) if torch.is_tensor(v) and v.dtype.is_floating_point else v)
            for k, v in packed.items()}
    npk = to_numpy(cast)
    npk["cross_bond_sel"] = npk["bond_cross"]
    # boundary slot start of every word, relative to its own molecule
    split_word_ids = npk["word_ids"]
    split_mol_of_word = fit.arrays["word_mol"][split_word_ids].astype(np.int64)
    npk["word_bnd_row"] = (fit.arrays["word_boundary_start"][split_word_ids].astype(np.int64)
                           - fit.graph_ptr_boundary[split_mol_of_word])
    terms = model_terms(models["FULL_WORD_OP"], packed, dtype)
    refs = {"FULL": [], "LOCAL": [], "PORT_DIAG": []}
    for m in range(len(ids)):
        for c in range(cfg["operator"]["channels"]):
            refs["FULL"].append(ref_full(npk, terms, m, c))
            refs["LOCAL"].append(ref_local(npk, terms, m, c))
            refs["PORT_DIAG"].append(ref_portdiag(npk, terms, m, c))
    torch_dtype = dtype
    out_vals = {}
    with torch.no_grad():
        for arm, model in models.items():
            res = model(cast)
            out_vals[arm] = res["F"].detach().cpu().numpy().reshape(-1)
    refs_flat = {k: np.asarray(v, dtype=np.float64) for k, v in refs.items()}
    errors = {}
    arm_name = {"FULL": "FULL_WORD_OP", "LOCAL": "LOCAL_WORD_OP",
                "PORT_DIAG": "PORT_DIAG_WORD_OP"}
    for key in ("FULL", "LOCAL", "PORT_DIAG"):
        errors[key] = float(np.max(np.abs(out_vals[arm_name[key]] - refs_flat[key])))
    errors["FULL_vs_boundary"] = float(np.max(np.abs(out_vals["FULL_WORD_OP"]
                                                       - out_vals["FULL_BOUNDARY_CHECK"])))
    # per-channel spread used to judge the tolerance
    magnitude = float(np.max(np.abs(refs_flat["FULL"])))
    return {"dtype": dtype_name, "n_molecules": len(ids), "molecules": [int(x) for x in ids],
            "max_abs_error": errors, "magnitude": magnitude,
            "tolerance": 1e-8 if dtype_name == "float64" else 1e-4,
            "pass": all(v <= (1e-8 if dtype_name == "float64" else 1e-4)
                        for v in errors.values())}


def batch_split_invariance(data: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    cfg = C.load_config()
    fit = data["fit"]
    ids = choose_molecules(fit)[:8]
    model = T.build_model(cfg, data, "FULL_WORD_OP").double()
    model.eval()

    def run(ids_):
        packed = fit.pack(np.asarray(ids_), device="cpu")
        cast = {k: (v.to(torch.float64) if torch.is_tensor(v) and v.dtype.is_floating_point
                    else v) for k, v in packed.items()}
        with torch.no_grad():
            return model(cast)["yhat"].numpy()

    whole = run(ids)
    singles = np.concatenate([run([int(x)]) for x in ids])
    rev = run(ids[::-1])[::-1]
    return {"max_abs_whole_vs_single": float(np.max(np.abs(whole - singles))),
            "max_abs_whole_vs_reversed": float(np.max(np.abs(whole - rev))),
            "pass": float(np.max(np.abs(whole - singles))) <= 1e-9
                    and float(np.max(np.abs(whole - rev))) <= 1e-9}


def shared_init(data: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    cfg = C.load_config()
    hashes = {}
    for arm in ("FULL_WORD_OP", "LOCAL_WORD_OP", "PORT_DIAG_WORD_OP"):
        model = T.build_model(cfg, data, arm)
        digest = T._state_hash(model)
        hashes[arm] = digest
        if arm == "FULL_WORD_OP":
            full_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            local_state = model.state_dict()
            equal = all(torch.equal(full_state[k], local_state[k]) for k in full_state)
            hashes[f"{arm}_bitwise_equal"] = bool(equal)
    return {"hashes": hashes, "all_equal": len({v for k, v in hashes.items()
                                                if isinstance(v, str)}) == 1,
            "pass": len({v for k, v in hashes.items() if isinstance(v, str)}) == 1}


def width_selection(data: Mapping[str, Any]) -> dict[str, Any]:
    cfg = C.load_config()
    record = data["record"]
    result = M.select_atomic_width(cfg, record["dims"], record["n_templates"],
                                   int(record["n_site_orbits"]), int(record["n_pair_orbits"]),
                                   int(data["fit"].z_std.shape[1]))
    result["pass"] = bool(result["within_20pct"])
    C.write_json(C.analysis_dir() / "width_selection.json", result)
    return result


# ---------------------------------------------------------------------------
# gradient health and finite differences
# ---------------------------------------------------------------------------


def health(data: Mapping[str, Any], steps: int = 60, device: str = "cpu") -> dict[str, Any]:
    import torch

    cfg = C.load_config()
    fit = data["fit"]
    ids = np.arange(32)
    model = T.build_model(cfg, data, "FULL_WORD_OP").to(device)
    target_all = (fit.labels - fit.target_mean) / fit.target_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["recipe"]["peak_lr"]),
                                  weight_decay=float(cfg["recipe"]["weight_decay"]))
    losses = []
    grad = {}
    for step in range(int(steps)):
        model.train()
        start = (step * 64) % max(1, fit.n_molecules - 64)
        idx = np.arange(start, start + 32)
        packed = fit.pack(idx, device=device)
        target = torch.as_tensor(target_all[idx], device=device)
        optimizer.zero_grad(set_to_none=True)
        out = model(packed)
        loss, parts = T.loss_terms(model, out, target, data["templates"], data)
        loss.backward()
        if step == 0:
            grad = T._grad_stats(model)
        losses.append(float(loss.detach()))
        optimizer.step()
    fd = finite_difference(data, model, idx, device)
    return {"steps": int(steps), "loss_first": losses[0], "loss_last": losses[-1],
            "loss_min": float(np.min(losses)), "loss_max": float(np.max(losses)),
            "grad_abs_sum": grad, "finite_difference": fd,
            "no_nan": bool(np.all(np.isfinite(losses))),
            "loss_decreased": bool(losses[-1] < losses[0])}


def finite_difference(data: Mapping[str, Any], model, idx, device: str) -> dict[str, Any]:
    import torch

    fit = data["fit"]
    target_all = (fit.labels - fit.target_mean) / fit.target_std
    packed = fit.pack(idx, device=device)
    target = torch.as_tensor(target_all[idx], device=device)
    model.eval()
    checks = {}

    def loss_value() -> float:
        with torch.no_grad():
            out = model(packed)
            loss, _ = T.loss_terms(model, out, target, data["templates"], data)
        return float(loss)

    for name, index in (("theta", (0, 0)), ("psi", (1, 0)), ("alpha", (2,)), ("pair.4.bias", None)):
        if name == "alpha":
            param = model.alpha
            sel = (index[0],)
        elif name == "pair.4.bias":
            if not hasattr(model, "pair"):
                continue
            param = model.pair[-1].bias
            sel = (0,)
        else:
            param = getattr(model, name)
            sel = index
        eps = 1e-4
        if sel is None:
            sel = (0,)
        flat = param.data.view(-1)
        shape = tuple(int(x) for x in param.shape)
        if len(shape) == 1:
            pos = int(sel[0])
        else:
            pos = int(np.ravel_multi_index(tuple(sel), shape))
        original = float(flat[pos])
        flat[pos] = original + eps
        plus = loss_value()
        flat[pos] = original - eps
        minus = loss_value()
        flat[pos] = original
        param.grad = None
        out = model(packed)
        loss, _ = T.loss_terms(model, out, target, data["templates"], data)
        loss.backward()
        analytic = float(param.grad.view(-1)[pos])
        param.grad = None
        numeric = (plus - minus) / (2 * eps)
        checks[name] = {"analytic": analytic, "numeric": numeric,
                        "abs_diff": abs(analytic - numeric),
                        "rel_diff": abs(analytic - numeric) / max(1e-12, abs(numeric))}
    return checks


def replay_check(data: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the old frozen word partition for the whole official Train/Valid."""
    old_path = C.OLD_ROUND / "data" / "ogbg-mollipo_words_ws-word-route-v1.npz"
    if not old_path.is_file():
        return {"skipped": True, "reason": "old word cache not present on this host"}
    old = C.load_npz(old_path)
    freeze = C.old_freeze()
    table, history = S.load_word_table(freeze)
    payload = C.dataset_cache("ogbg-mollipo")
    molecules = C.W.load_molecules(payload)
    out = {}
    for split in ("train", "valid"):
        graphs = [int(g) for g in np.asarray(payload[f"{split}_idx"])]
        structs = S.graph_structures(molecules, graphs, table, history)
        ptr_word = np.concatenate([[0], np.cumsum([len(r["words"]) for r in structs])])
        n_atoms = np.concatenate([[0], np.cumsum([r["mol"].n_atoms for r in structs])])
        n_bonds = np.concatenate([[0], np.cumsum([len(r["mol"].bond_u) for r in structs])])
        tmpl, natoms, atoms, bonds = [], [], [], []
        ptr_bond = [0]
        for m, rec in enumerate(structs):
            for w, word in enumerate(rec["words"]):
                tmpl.append(word["tid"])
                natoms.append(word["n"])
                atoms.extend(n_atoms[m] + a for a in word["atoms"])
                ids = []
                for u, v, t in word["internal"]:
                    # map word-local bond to the split-global physical bond id
                    a = word["atoms"][u]
                    b = word["atoms"][v]
                    for k in range(len(rec["mol"].bond_u)):
                        if {int(rec["mol"].bond_u[k]), int(rec["mol"].bond_v[k])} == {a, b}:
                            ids.append(int(n_bonds[m] + k))
                            break
                bonds.extend(sorted(ids))
                ptr_bond.append(len(bonds))
        checks = {
            "graph_word_ptr": np.array_equal(ptr_word, old[f"word/{split}/graph_word_ptr"]),
            "word_tmpl": np.array_equal(np.asarray(tmpl, np.int64),
                                        old[f"word/{split}/word_tmpl"].astype(np.int64)),
            "word_n_atoms": np.array_equal(np.asarray(natoms, np.int64),
                                           old[f"word/{split}/word_n_atoms"].astype(np.int64)),
            "word_atoms": np.array_equal(np.asarray(atoms, np.int64),
                                         old[f"word/{split}/word_atoms"].astype(np.int64)),
            "word_bonds": np.array_equal(np.asarray(bonds, np.int64),
                                         old[f"word/{split}/word_bonds"].astype(np.int64)),
        }
        out[split] = {"checks": checks, "pass": all(checks.values())}
    return {"skipped": False, **out}


def cmd_all(args) -> None:
    data = T.load_all()
    results = {}
    results["structure_audit"] = structure_audit(data)
    results["equivalence_fp64"] = equivalence(data, "float64")
    results["equivalence_fp32"] = equivalence(data, "float32")
    results["batch_split"] = batch_split_invariance(data)
    results["shared_init"] = shared_init(data)
    results["width"] = width_selection(data)
    results["replay"] = replay_check(data)
    results["health"] = health(data, steps=int(args.steps), device=args.device)
    path = C.analysis_dir() / "checks.json"
    C.write_json(path, C.jsonable(results))
    summary = {key: (value.get("pass", value.get("ok")) if isinstance(value, dict) else value)
               for key, value in results.items()}
    summary["equivalence_errors"] = {
        "fp64": results["equivalence_fp64"]["max_abs_error"],
        "fp32": results["equivalence_fp32"]["max_abs_error"],
    }
    summary["width_selected"] = results["width"]["selected_width"]
    summary["health"] = {k: results["health"][k] for k in
                         ("loss_first", "loss_last", "loss_decreased", "no_nan",
                          "grad_abs_sum")}
    summary["finite_difference"] = results["health"]["finite_difference"]
    print(C.jsonable(summary))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("all")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_all)
    ns = parser.parse_args()
    ns.func(ns)


def merge_check(data, channel: int = 0, n_molecules: int = 4) -> dict:
    """合并两个相连词再消元 vs 逐词消元后组装: 整图 F 必须一致."""
    import numpy as np
    import torch

    import wo_train as T

    fit = data["fit"]
    ids = choose_molecules(fit)[:n_molecules]
    model = T.build_model(C.load_config(), data, "FULL_WORD_OP", hidden=32)
    packed = fit.pack(np.asarray(ids), device="cpu")
    npk = to_numpy(packed)
    npk["cross_bond_sel"] = npk["bond_cross"]
    terms = model_terms(model, packed, torch.float32)
    report = []
    for m in range(len(ids)):
        mats = _word_matrices(npk, terms, m, channel)
        by_word = {int(x["w"]): x for x in mats}
        csel = npk["cross_mol"] == m
        cbu = npk["cross_u_row"][csel]
        cbv = npk["cross_v_row"][csel]
        cbk = terms["k"][npk["cross_bond_sel"]][csel, channel]
        cbw = list(zip(npk["cross_word_u"][csel].tolist(), npk["cross_word_v"][csel].tolist()))
        bonds = [(int(u), int(v), float(k), int(a), int(b))
                 for u, v, k, (a, b) in zip(cbu, cbv, cbk, cbw)]
        f_std = _assemble(mats, bonds, channel, merge=None)
        pair = next(((a, b) for a, b in cbw if a != b), None)
        if pair is None:
            continue
        f_merge = _assemble(mats, bonds, channel, merge=pair)
        report.append({"molecule": int(ids[m]), "merged_words": [int(pair[0]), int(pair[1])],
                       "F_standard": float(f_std), "F_merged": float(f_merge),
                       "abs_diff": float(abs(f_std - f_merge))})
    worst = max((r["abs_diff"] for r in report), default=0.0)
    return {"channel": channel, "rows": report, "max_abs_diff": worst,
            "pass": bool(worst <= 1e-9)}


def _assemble(mats, bonds, channel, merge=None):
    """Boundary assembly; ``merge`` merges two word factors into one block."""
    from collections import OrderedDict

    import numpy as np

    factors = []
    if merge is None:
        for word in mats:
            factors.append({"atoms": {int(word["sites"][s]): s for s in range(word["n"])},
                            "words": [word]})
    else:
        a, b = merge
        wa, wb = None, None
        kept = []
        for word in mats:
            if int(word["w"]) == int(a):
                wa = word
            elif int(word["w"]) == int(b):
                wb = word
            else:
                kept.append(word)
        atoms = {}
        for word in (wa, wb):
            for s in range(word["n"]):
                atoms[int(word["sites"][s])] = (word["w"], s)
        factors.append({"atoms": atoms, "words": [wa, wb]})
        for word in kept:
            factors.append({"atoms": {int(word["sites"][s]): s for s in range(word["n"])},
                            "words": [word]})
    owner = {}
    for idx, f in enumerate(factors):
        for atom in f["atoms"]:
            owner[atom] = idx
    boundary = OrderedDict()
    for u, v, k, a, b in bonds:
        if owner[u] != owner[v]:
            for x in (u, v):
                boundary.setdefault(x, len(boundary))
    slots = dict(boundary)
    total_c = 0.0
    K = np.zeros((len(slots), len(slots)))
    h = np.zeros(len(slots))
    for f in factors:
        atoms = sorted(f["atoms"], key=lambda x: f["atoms"][x] if isinstance(f["atoms"][x], int) else -1)
        atoms = sorted(f["atoms"])
        words = f["words"]
        if len(words) == 1:
            Q, bvec, sites = words[0]["Q"], words[0]["b"], words[0]["sites"]
            n = len(sites)
            Qf, bf = Q, bvec
            atom_list = [int(x) for x in sites[:n]]
        else:
            atom_list = sorted(f["atoms"])
            n = len(atom_list)
            Qf = np.zeros((n, n))
            bf = np.zeros(n)
            pos = {atom: i for i, atom in enumerate(atom_list)}
            for word in words:
                idx = [pos[int(word["sites"][s])] for s in range(word["n"])]
                Qf[np.ix_(idx, idx)] = word["Q"]
                bf[idx] = word["b"]
            for u, v, kk, a, bb in bonds:
                if u in pos and v in pos:
                    i, j = pos[u], pos[v]
                    Qf[i, i] += kk
                    Qf[j, j] += kk
                    Qf[i, j] -= kk
                    Qf[j, i] -= kk
        B = [i for i, atom in enumerate(atom_list) if atom in slots]
        I = [i for i in range(n) if i not in set(B)]
        if not I:
            S, hh, cc = Qf, bf, 0.0
        else:
            QII = Qf[np.ix_(I, I)]
            QIB = Qf[np.ix_(I, B)] if B else np.zeros((len(I), 0))
            QBB = Qf[np.ix_(B, B)]
            sign, ld = np.linalg.slogdet(QII)
            sol = np.linalg.solve(QII, QIB) if B else None
            S = QBB - QIB.T @ sol
            hh = bf[B] - (sol.T @ bf[I] if B else np.zeros(0))
            cc = 0.5 * ld - 0.5 * float(bf[I] @ np.linalg.solve(QII, bf[I]))
        total_c += cc
        for i, atom in enumerate(atom_list):
            if atom in slots:
                K[slots[atom], slots[atom]] += S[B.index(i), B.index(i)]
                h[slots[atom]] += hh[B.index(i)]
                for j in B:
                    if j != i:
                        K[slots[atom], slots[atom_list[j]]] += S[B.index(i), B.index(j)]
    for u, v, k, a, b in bonds:
        if u in slots and v in slots:
            i, j = slots[u], slots[v]
            K[i, i] += k
            K[j, j] += k
            K[i, j] -= k
            K[j, i] -= k
    sign, ld = np.linalg.slogdet(K) if len(slots) else (1, 0.0)
    return total_c + 0.5 * ld - 0.5 * float(h @ np.linalg.solve(K, h)) if len(slots) else total_c


# ---------------------------------------------------------------------------
# structural case reports (Section 10)
# ---------------------------------------------------------------------------


def _case_mol_ids(data, n_fixed: int = 2):
    """Deterministic multi-port Valid molecules, fixed before labels are read."""
    valid = data["valid"]
    a = valid.arrays
    cross = np.diff(a["graph_ptr_cross"])
    n_words = np.diff(a["graph_ptr_word"])
    score = [(int(min(cross[m], 4)), int(min(n_words[m], 8)), -int(m))
             for m in range(valid.n_molecules) if cross[m] >= 2 and n_words[m] >= 3]
    score.sort(reverse=True)
    return [int(-x[2]) for x in score[:n_fixed]]


def case_report(data, arm: str, mol_ids, channel: int = 0) -> list[dict]:
    import torch

    import wo_train as T

    path = C.results_root() / f"lipo_{arm}" / "best.pt"
    if not path.is_file():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = T.build_model(C.load_config(), data, arm)
    model.load_state_dict(payload["state"])
    model.eval()
    out = []
    for split_name in ("valid", "h"):
        split = data[split_name]
        if split.labels.size == 0:
            continue
        for mol in mol_ids:
            if mol >= split.n_molecules:
                continue
            packed = split.pack([mol], device="cpu")
            npk = to_numpy(packed)
            npk["cross_bond_sel"] = npk["bond_cross"]
            terms = model_terms(model, packed, torch.float32)
            with torch.no_grad():
                pred = float(model(packed)["yhat"].item())
            words = _word_matrices(npk, terms, 0, channel)
            word_rows = []
            for word in words:
                S, h, c = _reduce(word)
                word_rows.append({
                    "type": int(npk["word_tmpl"][word["w"]]),
                    "n_atoms": int(word["n"]), "n_ports": len(word["boundary"]),
                    "ports": [int(word["sites"][s]) for s in word["boundary"]],
                    "c": float(c), "h": [float(x) for x in h],
                    "S": [[float(x) for x in row] for row in S],
                })
            out.append({
                "split": split_name, "molecule": int(mol),
                "row_id": int(split.row_ids[mol]), "n_atoms": int(split.graph_n_atoms[mol]),
                "n_words": int(len(words)), "channel": channel,
                "pred_raw": pred, "label_raw": float(split.labels[mol]),
                "n_crossing_bonds": int((npk["cross_mol"] == 0).sum()),
                "words": word_rows,
            })
    return out


def cases(data, n: int = 4) -> list[dict]:
    fixed = _case_mol_ids(data, 2)
    out = case_report(data, "FULL_WORD_OP", fixed)
    # two more after the freeze: best / worst FULL-vs-LOCAL valid improvement
    try:
        import torch

        import wo_train as T

        valid = data["valid"]
        preds = {}
        for arm in ("FULL_WORD_OP", "LOCAL_WORD_OP"):
            path = C.results_root() / f"lipo_{arm}" / "best.pt"
            if not path.is_file():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = T.build_model(C.load_config(), data, arm)
            model.load_state_dict(payload["state"])
            model.eval()
            ev = T.evaluate(model, data["valid"], "cpu")
            preds[arm] = ev["pred_raw"]
        if len(preds) == 2:
            err = np.abs(preds["FULL_WORD_OP"] - valid.labels) - \
                np.abs(preds["LOCAL_WORD_OP"] - valid.labels)
            order = np.argsort(err)
            picks = [int(order[0]), int(order[-1])]
            out += case_report(data, "FULL_WORD_OP", picks)
            out += [{"split": "valid", "molecule": picks[0], "role": "best_full_vs_local"}]
            out += [{"split": "valid", "molecule": picks[1], "role": "worst_full_vs_local"}]
    except Exception as exc:  # pragma: no cover
        out.append({"error": str(exc)})
    return out
