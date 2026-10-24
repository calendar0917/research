"""Minimal correctness gates for zinc-global-operator-search-v1.

Only checks that can invalidate this round's result:

G1  parameter counts of C0 / A / K
G2  relation cache: lift == explicit M S M^T on a real molecule; symmetry
G3  molecule-order / padding invariance and no cross-molecule mixing
G4  atom renaming + bond reordering invariance of the full forward
G5  zero-bond / all-padding default: finite output, no softmax NaN
G6  finite-difference check of the final L1 loss w.r.t. one atom feature
G7  train/eval determinism with dropout off, at the reader input
G8  rebuild of the relation cache is byte-identical

Run: python gos_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import gos_data as GD  # noqa: E402
import gos_models as GM  # noqa: E402
import gos_runner as GR  # noqa: E402

OUT = _EXP / "analysis" / "cpu_checks.json"
RESULTS: dict[str, object] = {}


def gate(name, ok, **info):
    RESULTS[name] = {"ok": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(info)[:400], flush=True)
    if not ok:
        raise SystemExit(f"gate failed: {name}")


def main() -> int:
    torch.set_num_threads(4)
    rel_train, rel_valid, scale, mean_nb = GR.prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    train_cache = D.load_cache("train")
    norm = D.load_norm()

    # G1 -----------------------------------------------------------------
    counts = {}
    for route in ("C0", "A", "K"):
        m, info = GM.build_model(route, norm, 0, torch.device("cpu"), mean_n_bonds=mean_nb)
        counts[route] = int(info["parameters"])
        RESULTS.setdefault("param_breakdown", {})[route] = info["parameters_breakdown"]
    gate("G1_param_counts", counts == {"C0": 378993, "A": 460017, "K": 452457},
         counts=counts, expected={"C0": 378993, "A": 460017, "K": 452457})
    RESULTS["mean_train_n_bonds"] = mean_nb

    # G2 -----------------------------------------------------------------
    path_d = torch.load(GD.DBT_PATH["train"], map_location="cpu", weights_only=False)
    obj_d = torch.load(GD.DBT_OBJ["train"], map_location="cpu", weights_only=False)
    pair_ptr = np.asarray(path_d["pair_ptr"])
    path = np.asarray(path_d["path"])
    bond_ptr = np.asarray(obj_d["bond_ptr"])
    bond_index = np.asarray(obj_d["bond_index"]).astype(np.int64)
    worst = 0.0
    sym_err = 0
    for row in range(5):
        s, e = int(pair_ptr[row]), int(pair_ptr[row + 1])
        n = int(round(np.sqrt(e - s)))
        S = path[s * GD.L_MAX:e * GD.L_MAX].reshape(n, n, GD.L_MAX).astype(np.float64)
        bs, be = int(bond_ptr[row]), int(bond_ptr[row + 1])
        bi = bond_index[bs:be]
        B = bi.shape[0]
        M = np.zeros((B, n))
        M[np.arange(B), bi[:, 0]] = 1.0 / np.sqrt(2.0)
        M[np.arange(B), bi[:, 1]] = 1.0 / np.sqrt(2.0)
        ps, pe = int(rel_train["ptr"][row]), int(rel_train["ptr"][row + 1])
        got = np.asarray(rel_train["lift_sum"])[ps * GD.L_MAX:pe * GD.L_MAX].astype(np.float64).reshape(B, B, GD.L_MAX)
        for l in range(GD.L_MAX):
            ref = M @ S[:, :, l] @ M.T
            # the cache stores 2x the lifted value (sum, /2 applied at feature time)
            worst = max(worst, float(np.abs(got[:, :, l] / 2.0 - ref).max()))
            sym_err = max(sym_err, int(np.abs(got[:, :, l] - got[:, :, l].T).max()))
    gate("G2_lift_matches_matrix_product", worst < 1e-9 and sym_err == 0,
         max_abs_diff=worst, max_symmetry_diff=sym_err)

    # G3 -----------------------------------------------------------------
    rows = np.arange(32, dtype=np.int64)
    y = np.asarray(train_cache["labels"], dtype=np.float32)[rows]
    batches = {}
    preds = {}
    for route in ("C0", "A", "K"):
        model, _ = GM.build_model(route, norm, 0, torch.device("cpu"), mean_n_bonds=mean_nb)
        model.eval()
        b1 = collate(train_cache, rows, "GOS", valid=True)
        preds[route] = model(b1).detach().numpy()
        perm = np.random.default_rng(7).permutation(len(rows))
        b2 = collate(train_cache, rows[perm], "GOS", valid=True)
        p2 = model(b2).detach().numpy()
        d = float(np.abs(p2 - preds[route][perm]).max())
        # single molecule alone must agree with its batch position
        solo = collate(train_cache, rows[[3]], "GOS", valid=True)
        p_solo = float(model(solo).detach().numpy()[0])
        d_solo = abs(p_solo - float(preds[route][3]))
        gate(f"G3_batch_invariance_{route}", d < 1e-5 and d_solo < 1e-5,
             reorder_max_diff=d, solo_max_diff=d_solo)
    RESULTS["pred_spread_across_routes_at_init"] = {
        r: float(np.abs(preds[r] - preds["C0"]).max()) for r in ("A", "K")}

    # G4 atom renaming + bond reordering ---------------------------------
    # New object position k holds old object perm_b[k]; new atom row i holds
    # old atom perm_a[i].  Every table is rewritten consistently.
    for route in ("A", "K"):
        model, _ = GM.build_model(route, norm, 1, torch.device("cpu"), mean_n_bonds=mean_nb)
        model.eval()
        b = collate(train_cache, rows, "GOS", valid=True)
        p0 = model(b).detach().numpy()
        bp = np.asarray(train_cache["bond_ptr"])
        ap = np.asarray(train_cache["atom_ptr"])
        a_counts = np.diff(ap)[rows]
        b_counts = np.diff(bp)[rows]
        i = 5
        n = int(a_counts[i])
        e = int(b_counts[i])
        a_off = int(np.concatenate([[0], np.cumsum(a_counts)[:-1]])[i])
        b_off = int(np.concatenate([[0], np.cumsum(b_counts)[:-1]])[i])
        junc_b = b["junc_bond"]
        junc_in_mol = (junc_b[:, 0] >= b_off) & (junc_b[:, 0] < b_off + e)
        j_idx = torch.nonzero(junc_in_mol).reshape(-1)
        j_counts = len(j_idx)
        g = torch.Generator().manual_seed(11)
        perm_a = torch.randperm(n, generator=g)
        perm_b = torch.randperm(e, generator=g)
        inv_a = torch.empty(n, dtype=torch.long)
        inv_a[perm_a] = torch.arange(n)
        inv_b = torch.empty(e, dtype=torch.long)
        inv_b[perm_b] = torch.arange(e)
        bundle = {k: v.clone() if torch.is_tensor(v) else v for k, v in b.items()}
        # atoms
        ai = bundle["atom_feat"].clone()
        ai[a_off:a_off + n] = ai[a_off:a_off + n][perm_a]
        bundle["atom_feat"] = ai
        # bonds: object order perm_b, atom ids renamed
        bi = bundle["bond_index"].clone()
        obi = b["bond_index"][b_off:b_off + e].clone()
        nb_idx = obi[perm_b].clone()
        nb_idx[:, 0] = torch.tensor([int(inv_a[int(x) - a_off]) + a_off for x in nb_idx[:, 0]])
        nb_idx[:, 1] = torch.tensor([int(inv_a[int(x) - a_off]) + a_off for x in nb_idx[:, 1]])
        bi[b_off:b_off + e] = nb_idx
        bundle["bond_index"] = bi
        bf = bundle["bond_feat"].clone()
        bf[b_off:b_off + e] = b["bond_feat"][b_off:b_off + e][perm_b]
        bundle["bond_feat"] = bf
        # junctions: rows kept, bond ids and atom ids renamed
        jbi = bundle["junc_bond"].clone()
        jii = bundle["junc_index"].clone()
        for t in range(j_counts):
            r = int(j_idx[t])
            jbi[r, 0] = inv_b[int(jbi[r, 0]) - b_off] + b_off
            jbi[r, 1] = inv_b[int(jbi[r, 1]) - b_off] + b_off
            for c in range(3):
                jii[r, c] = int(inv_a[int(jii[r, c]) - a_off]) + a_off
        bundle["junc_bond"] = jbi
        bundle["junc_index"] = jii
        # bond-object relation grid
        rel = bundle["rel_feat"].clone()
        rel[i, :e, :e] = b["rel_feat"][i, :e, :e][perm_b][:, perm_b]
        bundle["rel_feat"] = rel
        p1 = model(bundle).detach().numpy()
        d = float(np.abs(p1 - p0).max())
        gate(f"G4_relabel_invariance_{route}", d < 1e-4, max_diff=d,
             atoms=int(n), bonds=int(e), junctions=int(j_counts))

    # G5 zero bonds ------------------------------------------------------
    nb = np.asarray(train_cache["n_bonds_graph"])
    rows0 = np.nonzero(nb == 0)[0][:4]
    RESULTS["zero_bond_molecules_in_train"] = int(len(rows0))
    if not len(rows0):
        print("NOTE no zero-bond molecule in train; padding case covered by G5_padding_*", flush=True)
    if len(rows0):
        b0 = collate(train_cache, rows0, "GOS", valid=True)
        z = float(b0["n_bonds_pad"])
        for route in ("A", "K", "C0"):
            model, _ = GM.build_model(route, norm, 0, torch.device("cpu"), mean_n_bonds=mean_nb)
            model.eval()
            out = model(b0).detach().numpy()
            gate(f"G5_zero_bond_{route}", np.isfinite(out).all(), n_zero_bond_rows=int(len(rows0)),
                 n_bonds_pad=z, values=out.tolist())
    # synthetic all-padding row inside the core
    for route in ("A", "K"):
        model, _ = GM.build_model(route, norm, 0, torch.device("cpu"), mean_n_bonds=mean_nb)
        model.eval()
        h = torch.zeros(2, 3, GM.TOKEN_DIM)
        rel = torch.zeros(2, 3, 3, GM.REL_REL_DIM)
        mask = torch.tensor([[True, False, False], [True, True, False]])
        if route == "A":
            out = model.layers[0](h, rel, mask)
        else:
            out = model.layers[0](h, rel, mask, 1.0 / mean_nb)
        gate(f"G5_padding_{route}", torch.isfinite(out).all().item(), max_abs=float(out.abs().max()))

    # G6 finite difference on the final loss (float64 forward) ----------
    rows = np.arange(16, dtype=np.int64)
    y = torch.tensor(np.asarray(train_cache["labels"], dtype=np.float64)[rows])
    for route in ("A", "K"):
        model, _ = GM.build_model(route, norm, 2, torch.device("cpu"), mean_n_bonds=mean_nb)
        model.eval().double()
        b = collate(train_cache, rows, "GOS", valid=True)
        b64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in b.items()}
        af = b64["atom_feat"].clone().requires_grad_(True)
        b2 = dict(b64)
        b2["atom_feat"] = af
        loss = F.l1_loss(model(b2).view(-1), y)
        loss.backward()
        ap = np.asarray(train_cache["atom_ptr"])
        counts = np.diff(ap)[rows]
        off = int(np.concatenate([[0], np.cumsum(counts)[:-1]])[2])
        j = off + 1
        grad = float(af.grad[j, 3])
        eps = 1e-6
        with torch.no_grad():
            fp = b64["atom_feat"].clone(); fp[j, 3] += eps
            fm = b64["atom_feat"].clone(); fm[j, 3] -= eps
            bp = dict(b64); bp["atom_feat"] = fp
            bm = dict(b64); bm["atom_feat"] = fm
            lp = float(F.l1_loss(model(bp).view(-1), y))
            lm = float(F.l1_loss(model(bm).view(-1), y))
        fd = (lp - lm) / (2 * eps)
        denom = max(1e-12, abs(grad), abs(fd))
        gate(f"G6_finite_difference_{route}", abs(grad - fd) / denom < 1e-4,
             autograd=grad, finite_diff=fd, rel_err=abs(grad - fd) / denom)

    # G7 determinism -----------------------------------------------------
    for route in ("A", "K"):
        model, _ = GM.build_model(route, norm, 3, torch.device("cpu"), mean_n_bonds=mean_nb)
        b = collate(train_cache, rows, "GOS", valid=True)
        model.eval()
        p1 = model(b).detach().clone(); p2 = model(b).detach().clone()
        model.train()
        p3 = model(b).detach().clone(); p4 = model(b).detach().clone()
        gate(f"G7_determinism_{route}",
             float((p1 - p2).abs().max()) == 0.0 and float((p3 - p4).abs().max()) == 0.0,
             eval_diff=float((p1 - p2).abs().max()), train_diff=float((p3 - p4).abs().max()))
        # reader feature check: every scale is finite and non-degenerate
        with torch.no_grad():
            feats = []
            orig = model.reader.forward
            model.reader.forward = lambda x: (feats.append(x.detach().clone()), orig(x))[1]
            model(b)
            model.reader.forward = orig
            f = feats[0]
        gate(f"G7_reader_features_{route}", torch.isfinite(f).all().item(),
             dim=int(f.shape[1]), mean_abs=float(f.abs().mean()),
             z0_scale=float(f[:, :513].abs().mean()),
             hsum_scale=float(f[:, 513:561].abs().mean()),
             hsq_scale=float(f[:, 561:].abs().mean()))

    # G8 rebuild determinism ---------------------------------------------
    d2 = GD.build_rel("valid")
    ref = GD.load_rel("valid")
    same = all(np.array_equal(np.asarray(d2[k]), np.asarray(ref[k]))
               for k in ("ptr", "lift_sum", "lift_dist", "lift_shared", "n_bonds_graph"))
    gate("G8_rel_cache_rebuild_deterministic", same)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT, RESULTS)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
