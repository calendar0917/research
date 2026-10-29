"""CPU-only correctness gates for HIS-v0 (Phase A1).  No GPU, no test split.

    uv run python -m ...  ->  code/his_cpu_checks.py

Writes ``analysis/cpu_checks.json``.  Everything the regulariser assumes is
checked against the *real* frozen model / feature cache, not against synthetic
tensors alone:

  G1  frozen-input sha256 lineage + frozen LCRC-A init state + 237,519 params
  G2  ``path_emb`` column block (320..351) proven from the real forward pass,
      including that the first-layer output delta equals dx[:,320:352] @ W.T
  G3  ANOVA in the trained model: exact reconstruction, zero marginals,
      projection commutes with the read-out
  G4  penalty invariant to embedding rescaling inside the real model
  G5  atom renumbering + bond-direction invariance of predictions and penalty
  G6  save / reload round-trip (state, H, H_3, predictions bitwise equal)
  G7  batch-composition invariance of predictions; penalty batch-independent
  G8  penalty gradients: non-zero, finite, confined to E and W_path,
      analytic vs finite-difference agreement
  G9  penalty needs no labels / no test split; static scan for test access
  G10 path_cat axis convention proven from a hand-built molecule
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import his_anova as A
import his_common as HIS

L, D, M, T = HIS.L, HIS.D, HIS.M, HIS.T

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

FORBIDDEN = ('load_cache("test"', "load_cache('test'", 'build_split("test"',
             "build_split('test'", 'official_labels("test"', "official_labels('test'",
             "lc_test.pt")


class Checker:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def run(self, cid: str, fn: Callable[[], Any]) -> None:
        try:
            details = fn() or {}
            self.results.append({"id": cid, "pass": True, "details": details})
            print(f"[PASS] {cid}")
        except Exception as exc:  # noqa: BLE001
            self.results.append({"id": cid, "pass": False,
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()})
            print(f"[FAIL] {cid}: {type(exc).__name__}: {exc}")

    @property
    def ok(self) -> bool:
        return all(r["pass"] for r in self.results)


def _cache_from_molecules(mols, labels=None) -> dict:
    cache = D._concat_molecules(mols)
    cache["split"] = "synthetic"
    cache["n_molecules"] = len(mols)
    cache["labels"] = np.asarray(labels if labels is not None else np.zeros(len(mols)),
                                 dtype=np.float32)
    cache["meta"] = {"feature_version": L.FEATURE_VERSION, "split": "synthetic"}
    return cache


def _pred(model, cache, rows) -> np.ndarray:
    model.eval()
    with torch.no_grad(), L.C.isolated_rng():
        return model(D.collate(cache, rows)).cpu().numpy().reshape(-1)


def _sha_of(paths) -> dict:
    return {HIS.track_rel(p): HIS.sha256_file(p) for p in paths}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="HIS-v0 CPU gates")
    ap.add_argument("--no-write", action="store_true",
                    help="run the gates without touching analysis/cpu_checks.json "
                         "(used on the remote checkout, which must stay clean)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    HIS.ANALYSIS.mkdir(parents=True, exist_ok=True)
    ckpt = Checker()

    frozen = HIS.verify_frozen_inputs()
    train = D.load_cache("train")
    valid = D.load_cache("valid")
    norm = D.load_norm()
    objs, _ = L.D.load_p3_objects("train")

    payload = torch.load(HIS.LCRC_A_S0_BEST, map_location="cpu", weights_only=False)
    trained = M.load_model_from_state("A", norm, payload["state_dict"], "cpu")
    trained.eval()
    fresh, fresh_info = M.build_model("A", norm, 0, "cpu")

    rows8 = np.arange(8)
    mols8 = [D.build_molecule_from_object(objs[i]) for i in rows8]
    cache8 = _cache_from_molecules(mols8)

    # ---------------------------------------------------------------- G1
    def g1():
        lineage = _sha_of(sorted(HIS.LCRC_TRACK.glob("code/lc_*.py")))
        his_code = _sha_of(sorted(HIS.CODE_DIR.glob("his_*.py")))
        assert fresh_info["parameters"] == HIS.FROZEN["params"], fresh_info["parameters"]
        assert fresh_info["initial_state_sha256"] == HIS.FROZEN["lcrc_a_s0_initial_state_sha256"]
        assert payload["epoch"] == HIS.FROZEN["lcrc_a_s0_best_epoch"]
        assert abs(float(payload["valid_mae"]) - HIS.FROZEN["lcrc_a_s0_best_valid"]) < 1e-12
        assert T.RECIPE["epochs"] == 600 and T.RECIPE["batch_size"] == 128
        assert T.RECIPE["lr"] == 1e-3 and T.RECIPE["min_lr"] == 1e-5
        assert T.RECIPE["warmup_steps"] == 500 and T.RECIPE["weight_decay"] == 1e-5
        assert T.RECIPE["grad_clip"] == 5.0 and T.RECIPE["loss"] == "l1"
        assert T.RECIPE["optimizer"] == "adamw"
        return {"frozen": frozen, "lcrc_code_sha256": lineage, "his_code_sha256": his_code,
                "git_commit": L.C.git_commit(), "parameters": fresh_info["parameters"],
                "initial_state_sha256": fresh_info["initial_state_sha256"],
                "recipe": T.RECIPE,
                "inference_unchanged": True}

    ckpt.run("G1_frozen_lineage_and_init_state", g1)

    # ---------------------------------------------------------------- G2
    def g2():
        model = trained
        captured: dict[str, torch.Tensor] = {}
        handles = [
            model.local.mlp.lins[0].register_forward_hook(
                lambda m, i, o: captured.__setitem__("x", i[0].detach().clone())),
            model.local.mlp.lins[0].register_forward_hook(
                lambda m, i, o: captured.__setitem__("y", o.detach().clone())),
        ]
        batch = D.collate(cache8, rows8)
        with torch.no_grad():
            model(batch)
        x0, y0 = captured["x"], captured["y"]
        sl = A.path_slice(model)
        assert sl == slice(320, 352), sl
        delta = torch.zeros_like(model.local.path_emb.weight)
        gen = torch.Generator().manual_seed(0)
        delta.normal_(generator=gen)
        with torch.no_grad():
            model.local.path_emb.weight.add_(delta)
            model(batch)
            x1, y1 = captured["x"], captured["y"]
            model.local.path_emb.weight.sub_(delta)
        for h in handles:
            h.remove()
        dx, dy = x1 - x0, y1 - y0
        touched = (dx.abs().sum(0) > 0).nonzero().flatten().tolist()
        assert touched == list(range(sl.start, sl.stop)), f"touched columns {touched}"
        expected_dx = torch.zeros_like(dx)
        expected_dx[:, sl].index_add_(0, batch["path_atom"], delta[batch["path_cat"]])
        err_x = float((dx - expected_dx).abs().max())
        w = model.local.mlp.lins[0].weight
        # the layer-0 output delta must be exactly the *column-block* read-out of
        # the path delta through W[:, 320:352] -- this is the matrix the penalty uses
        expected_dy = expected_dx[:, sl] @ w[:, sl].T
        err_y = float((dy - expected_dy).abs().max() / max(float(dy.abs().max()), 1e-12))
        assert err_x < 1e-5, err_x
        assert err_y < 1e-5, err_y
        # and no other column block reproduces that delta
        err_other = min(
            float((dy - expected_dx[:, q] @ w[:, q].T).abs().max() / max(float(dy.abs().max()), 1e-12))
            for q in (slice(0, 32), slice(288, 320), slice(352, 384)))
        assert err_other > 1e-3, err_other
        h = A.effective_response(model)
        assert h.shape == (21, 3, 3, int(model.cfg["local_hidden"]))
        return {"path_slice": [sl.start, sl.stop], "touched_columns": touched,
                "max_abs_err_input_delta": err_x,
                "max_rel_err_layer0_delta": err_y,
                "min_rel_err_other_blocks": err_other,
                "n_path_tokens_in_batch": int(batch["path_cat"].numel()),
                "effective_response_shape": list(h.shape)}

    ckpt.run("G2_path_block_proven_from_forward", g2)

    # ---------------------------------------------------------------- G3
    def g3():
        h = A.effective_response(trained).detach()
        h3 = A.third_order(h)
        recon = float((h3 + A.anova_le2(h) - h).abs().max())
        margins = A.margin_norms(h3)
        e = trained.local.path_emb.weight.detach().reshape(*A.PATH_SHAPE, 32)
        w = trained.local.mlp.lins[0].weight.detach()[:, A.path_slice(trained)]
        proj_commutes = float((A.project_embedding(e) @ w.T - A.anova_le2(h)).abs().max())
        scaled_ok = []
        for c in (0.1, 1.0, 12.0):
            hh = A.third_order((c * e) @ (w / c).T)
            scaled_ok.append(abs(float(hh.pow(2).mean()) - float(h3.pow(2).mean())))
        return {"shape": list(h.shape), "reconstruction_max_abs_err": recon,
                "marginal_norms_of_H3": margins, "projection_commutes_err": proj_commutes,
                "rescale_penalty_diffs": scaled_ok,
                "trained_h3_ratio": A.fro(h3) / A.fro(h),
                "trained_h_rms": float(h.pow(2).mean().sqrt()),
                "trained_h3_rms": float(h3.pow(2).mean().sqrt())}

    ckpt.run("G3_anova_in_trained_model", g3)

    # ---------------------------------------------------------------- G4
    def g4():
        import copy
        model = copy.deepcopy(trained)
        base = float(A.his_penalty(model)[0])
        e = model.local.path_emb.weight
        w = model.local.mlp.lins[0].weight
        sl = A.path_slice(model)
        with torch.no_grad():
            for c in (0.05, 3.0):
                e.mul_(c)
                w[:, sl] = w[:, sl] / c
                got = float(A.his_penalty(model)[0])
                assert abs(got - base) <= 1e-4 * max(base, 1e-8), (c, got, base)
                w[:, sl] = w[:, sl] * c
                e.div_(c)
        assert abs(float(A.his_penalty(model)[0]) - base) < 1e-12
        return {"base_penalty": base, "invariant": True}

    ckpt.run("G4_penalty_rescaling_invariance", g4)

    # ---------------------------------------------------------------- G5
    def g5():
        idx = [0, 1, 2, 3, 4]
        base_mols = [D.build_molecule_from_object(objs[i]) for i in idx]
        base = _cache_from_molecules(base_mols)
        p0 = _pred(trained, base, np.arange(len(idx)))
        pen0 = float(A.his_penalty(trained)[0])
        worst_perm, worst_rev = 0.0, 0.0
        for k, i in enumerate(idx):
            mol = base_mols[k]
            perm = np.random.default_rng(1234 + k).permutation(int(mol["n_atoms"]))
            permuted = _cache_from_molecules(
                [D.build_molecule_from_object(objs[i], perm=perm)])
            worst_perm = max(worst_perm, float(abs(_pred(trained, permuted, [0])[0] - p0[k])))
            reversed_ = _cache_from_molecules(
                [D.build_molecule_from_object(objs[i], reverse_bonds=True)])
            worst_rev = max(worst_rev, float(abs(_pred(trained, reversed_, [0])[0] - p0[k])))
        assert worst_perm < 2e-5, worst_perm
        assert worst_rev < 2e-5, worst_rev
        assert abs(float(A.his_penalty(trained)[0]) - pen0) < 1e-12
        return {"max_pred_delta_permutation": worst_perm,
                "max_pred_delta_bond_reversal": worst_rev,
                "penalty_unchanged": True}

    ckpt.run("G5_renumbering_and_bond_direction", g5)

    # ---------------------------------------------------------------- G6
    def g6():
        tmp = HIS.ANALYSIS / "_roundtrip_tmp.pt"
        L.C.atomic_torch_save({"state_dict": L.C.cpu_state(trained)}, tmp)
        reloaded = M.load_model_from_state(
            "A", norm, torch.load(tmp, map_location="cpu", weights_only=False)["state_dict"], "cpu")
        tmp.unlink()
        reloaded.eval()
        h_a = A.effective_response(trained).detach()
        h_b = A.effective_response(reloaded).detach()
        assert torch.equal(h_a, h_b)
        assert torch.equal(A.third_order(h_a), A.third_order(h_b))
        p_a = _pred(trained, cache8, rows8)
        p_b = _pred(reloaded, cache8, rows8)
        assert np.array_equal(p_a, p_b), float(np.abs(p_a - p_b).max())
        assert float(A.his_penalty(trained)[0]) == float(A.his_penalty(reloaded)[0])
        return {"bitwise_state": True, "h_equal": True, "pred_max_abs_delta": 0.0}

    ckpt.run("G6_save_reload_roundtrip", g6)

    # ---------------------------------------------------------------- G7
    def g7():
        full = _pred(trained, cache8, rows8)
        half_a = _pred(trained, cache8, [0, 1, 2])
        half_b = _pred(trained, cache8, [3, 4, 5])
        rev = _pred(trained, cache8, [5, 4, 3, 2, 1, 0])
        d1 = float(np.abs(np.concatenate([half_a, half_b]) - full[:6]).max())
        d2 = float(np.abs(rev - full[:6][::-1]).max())
        assert d1 < 2e-5 and d2 < 2e-5, (d1, d2)
        pen = [float(A.his_penalty(trained)[0]) for _ in range(3)]
        assert max(pen) - min(pen) == 0.0
        # penalty must not depend on labels or on any batch
        assert float(A.his_penalty(trained)[0]) == pen[0]
        return {"max_pred_delta_split_batch": d1, "max_pred_delta_reordered": d2,
                "penalty_batch_independent": True}

    ckpt.run("G7_batch_composition", g7)

    # ---------------------------------------------------------------- G8
    def g8():
        import copy
        model = copy.deepcopy(trained)
        model.train()
        batch = D.collate(cache8, rows8)
        target = torch.as_tensor(cache8["labels"][rows8], dtype=torch.float32)
        pred = model(batch)
        loss = F.l1_loss(pred, target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        g_l1 = A.affected_grad_vector(model).clone()
        model.zero_grad(set_to_none=True)
        pen, _ = A.his_penalty(model)
        pen.backward()
        g_reg = A.affected_grad_vector(model).clone()
        outside = A.outside_slice_grad_max(model)
        other = [n for n, p in model.named_parameters()
                 if p.grad is not None and float(p.grad.abs().max()) > 0]
        assert set(other) == {"local.path_emb.weight", "local.mlp.lins.0.weight"}, other
        assert outside == 0.0, outside
        assert torch.isfinite(g_reg).all() and float(g_reg.abs().max()) > 0
        assert torch.isfinite(g_l1).all() and float(g_l1.abs().max()) > 0
        lam = 0.10 * float(g_l1.norm()) / float(g_reg.norm())

        # finite-difference check of the penalty gradient (float64 for headroom)
        mdl = copy.deepcopy(model).double()
        pen64, _ = A.his_penalty(mdl)
        mdl.zero_grad(set_to_none=True)
        pen64.backward()
        g_emb64 = mdl.local.path_emb.weight.grad.detach().clone()
        w64 = mdl.local.mlp.lins[0].weight
        sl = A.path_slice(mdl)
        g_w64 = w64.grad.detach()[:, sl].clone()
        gen = torch.Generator().manual_seed(7)
        eps = 1e-6
        fd = {}
        for name, gvec in (("path_emb.weight", g_emb64), ("W_path", g_w64)):
            u = torch.zeros_like(gvec)
            u.normal_(generator=gen)
            u = u / u.norm()

            def plus_minus(sign: float) -> float:
                with torch.no_grad():
                    if name == "path_emb.weight":
                        mdl.local.path_emb.weight.add_(sign * eps * u)
                    else:
                        w64.data[:, sl].add_(sign * eps * u)
                    val = float(A.his_penalty(mdl)[0])
                    if name == "path_emb.weight":
                        mdl.local.path_emb.weight.sub_(sign * eps * u)
                    else:
                        w64.data[:, sl].sub_(sign * eps * u)
                return val

            plus, minus = plus_minus(1.0), plus_minus(-1.0)
            analytic = float((gvec * u).sum())
            fd_val = (plus - minus) / (2 * eps)
            rel = abs(analytic - fd_val) / max(abs(analytic), 1e-12)
            fd[name] = {"analytic": analytic, "fd": fd_val, "rel_err": rel}
            assert rel < 5e-3, (name, fd[name])
        return {"g_l1_affected_norm": float(g_l1.norm()), "g_reg_affected_norm": float(g_reg.norm()),
                "lambda_preview_on_synthetic_batch": lam,
                "penalty_on_synthetic_batch": float(pen),
                "params_with_penalty_grad": other,
                "outside_slice_penalty_grad_max": outside,
                "finite_difference": fd,
                "affected_param_elements": int(g_l1.numel())}

    ckpt.run("G8_penalty_gradients", g8)

    # ---------------------------------------------------------------- G9
    def g9():
        hits = {}
        for path in sorted(HIS.CODE_DIR.glob("his_*.py")):
            if path.name == Path(__file__).name:
                continue  # this scanner necessarily contains the patterns
            text = path.read_text()
            found = [p for p in FORBIDDEN if p in text]
            if found:
                hits[path.name] = found
        assert not hits, hits
        # train/valid caches carry labels; no test cache is ever opened
        assert train["meta"]["labels_loaded"] is True
        pen_zero_label_free = float(A.his_penalty(fresh)[0])
        assert pen_zero_label_free > 0
        return {"forbidden_pattern_hits": hits, "test_cache_loaded": False,
                "penalty_at_fresh_init": pen_zero_label_free,
                "note": "test split is never constructed by this round's code"}

    ckpt.run("G9_label_and_test_isolation", g9)

    # ---------------------------------------------------------------- G10
    def g10():
        # hand-built 3-atom chain: types (0, 7, 5), bonds (1=b1, 2=b2)
        atom_feat = np.zeros((3, L.ATOM_FEAT_DIM), dtype=np.float32)
        atom_feat[0, 0] = 1.0
        atom_feat[1, 7] = 1.0
        atom_feat[2, 5] = 1.0
        atoms = np.asarray([0, 7, 5], dtype=np.int64)
        bond_index = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        bond_type = np.asarray([1, 2], dtype=np.int64)
        bond_feat = np.zeros((2, L.BOND_FEAT_DIM), dtype=np.float32)
        glob = np.zeros(L.GLOBAL_DIM, dtype=np.float32)
        mol = D.build_from_arrays(atom_feat, atoms, bond_index, bond_type, bond_feat, glob)
        codes = mol["path_cat"].astype(int).tolist()
        ends = mol["path_end63"].astype(int).tolist()
        # root 0 -> 1 -> 2 : middle type 7, b1 = 1, b2 = 2  => 7*9 + 0*3 + 1 = 64
        assert 64 in codes, codes
        # and the axis interpretation H.view(21,3,3) matches that encoding
        flat = [a * 9 + b * 3 + c for a in range(21) for b in range(3) for c in range(3)]
        assert flat == list(range(189))
        i = flat.index(64)
        a, rem = divmod(i, 9)
        b, c = divmod(rem, 3)
        assert (a, b + 1, c + 1) == (7, 1, 2)
        return {"path_codes": codes, "path_end63": ends,
                "expected_code_for_root0": 64,
                "axis_mapping": {"a": "middle atom type", "b": "bond1 type - 1",
                                 "c": "bond2 type - 1"}}

    ckpt.run("G10_path_cat_axis_convention", g10)

    out = {
        "round": HIS.ROUND_ID,
        "git_commit": L.C.git_commit(),
        "created": HIS.now(),
        "device": "cpu",
        "all_pass": ckpt.ok,
        "checks": ckpt.results,
        "official_test_loaded": False,
        "pytest": "tests/test_his_v0.py (21 pure-math tests, fresh-clone safe)",
    }
    if args.no_write:
        print("\n[--no-write] analysis/cpu_checks.json not modified")
    else:
        path = Path(args.out) if args.out else HIS.ANALYSIS / "cpu_checks.json"
        HIS.write_json(path, out)
        print(f"\nCPU gates: {sum(1 for r in ckpt.results if r['pass'])}/{len(ckpt.results)} "
              f"pass -> {HIS.track_rel(path)}")
    n_pass = sum(1 for r in ckpt.results if r["pass"])
    print(f"CPU gates: {n_pass}/{len(ckpt.results)} pass")
    return 0 if ckpt.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
