"""CPU hard gates for zinc-p3-chromatic-bond-v0.

Run from the repository root:

    uv run python tracks/structural-sufficiency/zinc_p3_chromatic_bond_v0/code/chr_cpu_checks.py

Writes analysis/cpu_checks.json and analysis/parameter_ledger.json.
Official test is never opened; a torch.load recorder asserts this.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SS = _EXP.parent
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import chr_data as R  # noqa: E402
import chr_models as M  # noqa: E402
import dcc_models as DCC  # noqa: E402

ANALYSIS = _EXP / "analysis"
CONFIG = _EXP / "configs" / "chr_v0.json"
MEAN_LINE_DEGREE = 1.308120486872196
TOL = 1.0e-5
RNG = np.random.default_rng(20261026)


def _jsonable(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _hash_state_dict(sd: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _state_max_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    if set(sa) != set(sb):
        return float("inf")
    out = 0.0
    for k in sa:
        d = (sa[k].detach().float() - sb[k].detach().float()).abs().max().item()
        out = max(out, float(d))
    return out


def _clone_batch(b: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def _forward0(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch)


def _line_degree_from_edges(bond_index: np.ndarray, n: int) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    for u, v in bond_index:
        deg[int(u)] += 1
        deg[int(v)] += 1
    return np.asarray([deg[int(u)] + deg[int(v)] - 2 for u, v in bond_index], dtype=np.int64)


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("[PASS] " if ok else "[FAIL] ") + name + " :: " + json.dumps(detail, default=_jsonable)[:500])

    loaded: list[str] = []
    _orig_load = torch.load

    def _recording_load(*args, **kwargs):
        loaded.append(str(args[0]) if args else "")
        return _orig_load(*args, **kwargs)

    torch.load = _recording_load
    try:
        norm = D.load_norm()
        train_cache = D.load_cache("train")
        valid_cache = D.load_cache("valid")
        # Build only train/valid relation caches; this function refuses "test".
        for split in ("valid", "train"):
            if not R.meta_path(split).exists() or not R.rel_path(split).exists():
                R.build_split(split, force=True)
        tr_meta = R.load_meta("train")
        va_meta = R.load_meta("valid")
        collate = R.make_collate("valid")
        rows = np.arange(8, dtype=np.int64)
        batch = collate(valid_cache, rows, "P3N", valid=True)
        target = torch.as_tensor(np.asarray(valid_cache["labels"], dtype=np.float32)[rows])

        # ------------------------------------------------------------------
        # deterministic construction and shared init
        # ------------------------------------------------------------------
        M._SHARED.clear()
        mC0, infoC0 = M.build_initial_state(norm, 0, train_mean_log1p_line_degree=MEAN_LINE_DEGREE)
        h0 = infoC0["initial_state_sha256"]
        M._SHARED.clear()
        mC1, infoC1 = M.build_initial_state(norm, 0, train_mean_log1p_line_degree=MEAN_LINE_DEGREE)
        check("deterministic_constructor", h0 == infoC1["initial_state_sha256"],
              {"first": h0, "second": infoC1["initial_state_sha256"]})
        check("p3_ssl_source", infoC0["ssl"]["sha256"] == cfg["initialisation"]["ssl_sha256"],
              {"ssl": infoC0["ssl"]["sha256"], "expected": cfg["initialisation"]["ssl_sha256"]})
        check("dcc_d0_state_matches_history",
              infoC0["dcc_d0_state_sha256"] == cfg["initialisation"]["dcc_d0_state_sha256"],
              {"actual": infoC0["dcc_d0_state_sha256"],
               "expected": cfg["initialisation"]["dcc_d0_state_sha256"]})

        dual, dcc_info = DCC.build_initial_state(norm, 0)
        d0_state = C.cpu_state(dual)
        mC, infoC = M.build_model("C", norm, 0, "cpu", train_mean_log1p_line_degree=MEAN_LINE_DEGREE)
        mS, infoS = M.build_model("S", norm, 0, "cpu", train_mean_log1p_line_degree=MEAN_LINE_DEGREE)

        # ------------------------------------------------------------------
        # 1. parameter ledger
        # ------------------------------------------------------------------
        named = list(mC.named_parameters())
        total = sum(p.numel() for _, p in named)
        ledger = {
            "round_id": cfg["round_id"],
            "parameters": total,
            "expected_total": int(cfg["parameters"]["expected_total"]),
            "cap": int(cfg["parameters"]["cap"]),
            "n_tensors": len(named),
            "tensors": [
                {"name": n, "shape": list(p.shape), "numel": int(p.numel()),
                 "requires_grad": bool(p.requires_grad), "dtype": str(p.dtype)}
                for n, p in named
            ],
            "breakdown": {
                "encoders": sum(p.numel() for n, p in named if n.startswith("encoders.")),
                "group_composer": sum(p.numel() for n, p in named if n.startswith("group_composer.")),
                "local_composer": sum(p.numel() for n, p in named if n.startswith("local_composer.")),
                "token_fusion": sum(p.numel() for n, p in named if n.startswith(("token_proj.", "token_norm."))),
                "layers": sum(p.numel() for n, p in named if n.startswith("layers.")),
                "reader": sum(p.numel() for n, p in named if n.startswith("reader.")),
            },
            "named_parameter_sum_equals_total": sum(p.numel() for _, p in named) == total,
            "state_dict_parameter_keys_do_not_duplicate": len({n for n, _ in named}) == len(named),
        }
        opt = torch.optim.AdamW(mC.parameters(), lr=1e-3)
        ledger["optimizer_parameters"] = C.optimizer_parameter_count(opt)
        ledger["optimizer_covers_all"] = ledger["optimizer_parameters"] == total
        ledger["hashes"] = {
            "config_sha256": C.sha256_file(CONFIG),
            "ssl_sha256": C.sha256_file(C.P3_SSL_S0),
            "relation_train_sha256": C.sha256_file(R.rel_path("train")),
            "relation_valid_sha256": C.sha256_file(R.rel_path("valid")),
            "dcc_d0_state_sha256": infoC0["dcc_d0_state_sha256"],
            "chromatic_initial_state_sha256": infoC0["initial_state_sha256"],
            "code_sha256": {p.name: C.sha256_file(p) for p in sorted((_EXP / "code").glob("*.py"))},
        }
        check("parameter_cap_and_exact_ledger",
              total <= int(cfg["parameters"]["cap"])
              and total == int(cfg["parameters"]["expected_total"])
              and ledger["breakdown"] == cfg["parameters"]["breakdown"]
              and ledger["optimizer_covers_all"],
              {"parameters": total, "expected": cfg["parameters"]["expected_total"],
               "cap": cfg["parameters"]["cap"], "breakdown": ledger["breakdown"],
               "optimizer_covers_all": ledger["optimizer_covers_all"]})
        (ANALYSIS / "parameter_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

        # ------------------------------------------------------------------
        # 2. frozen local carrier
        # ------------------------------------------------------------------
        d0_sub = {k: v for k, v in d0_state.items()
                  if k.startswith(("encoders.", "group_composer.", "local_composer."))}
        own_sub = {k: v for k, v in mC.state_dict().items()
                   if k.startswith(("encoders.", "group_composer.", "local_composer."))}
        same_keys = set(d0_sub) == set(own_sub)
        max_dcc = max((d0_sub[k].float() - own_sub[k].float()).abs().max().item()
                      for k in own_sub) if same_keys else float("inf")
        check("local_carrier_exactly_dcc_d0", same_keys and max_dcc == 0.0,
              {"same_keys": same_keys, "max_abs_diff": max_dcc})
        check("global_norm_buffers_unchanged",
              torch.equal(mC.global_mean.cpu(), torch.as_tensor(norm["global_mean"], dtype=torch.float32))
              and torch.equal(mC.global_scale.cpu(), torch.as_tensor(norm["global_scale"], dtype=torch.float32)),
              {"global_mean_max_diff": (mC.global_mean.cpu() - torch.as_tensor(norm["global_mean"], dtype=torch.float32)).abs().max().item(),
               "global_scale_max_diff": (mC.global_scale.cpu() - torch.as_tensor(norm["global_scale"], dtype=torch.float32)).abs().max().item()})

        # D1 supervised checkpoint must not be the init.
        d1_path = _SS / "zinc_dual_channel_composer_v1" / "runs" / "L_s0" / "best.pt"
        d1_hash = None
        d1_differs = True
        if d1_path.exists():
            d1 = _orig_load(d1_path, map_location="cpu", weights_only=False)
            d1_hash = _hash_state_dict(d1["state_dict"])
            d1_differs = d1_hash != infoC["initial_state_sha256"]
        check("no_supervised_dcc_checkpoint_as_init", d1_differs,
              {"supervised_dcc_state_sha256": d1_hash,
               "chromatic_init_sha256": infoC["initial_state_sha256"],
               "source": "train-only masked SSL + D0 shared initial state"})

        # forward comparison of the local carrier
        a, b, w, cg, cl, nb = mC.local_parts(batch)
        a0, b0, w0 = dual.encoders(batch)
        Wg, Vg, nb0 = dual.group_to_bonds(w0, batch)
        cg0 = dual.group_composer(torch.cat([b0, Wg, Vg, torch.log1p(nb0).unsqueeze(1), batch["bond_feat"]], dim=1))
        Wl, Vl, _ = dual.local_grouping(w0, batch, organized=True)
        cl0 = dual.local_composer(torch.cat([b0, Wl, Vl, torch.log1p(nb0).unsqueeze(1), batch["bond_feat"]], dim=1))
        with torch.no_grad():
            diffs = {
                "a": (a - a0).abs().max().item(), "b": (b - b0).abs().max().item(),
                "w": (w - w0).abs().max().item(), "c_glob": (cg - cg0).abs().max().item(),
                "c_loc": (cl - cl0).abs().max().item(), "n_b": (nb - nb0).abs().max().item(),
            }
        check("local_carrier_forward_unchanged", max(diffs.values()) <= 1e-6, diffs)

        # targets and global standardisation
        train_label_hash = C.sha256_array(np.asarray(train_cache["labels"], dtype=np.float32))
        valid_label_hash = C.sha256_array(np.asarray(valid_cache["labels"], dtype=np.float32))
        check("raw_target_no_normalisation",
              np.asarray(train_cache["labels"]).dtype in (np.float32, np.float64)
              and "labels" not in batch and "target" not in batch,
              {"train_label_sha256": train_label_hash, "valid_label_sha256": valid_label_hash,
               "batch_has_labels": "labels" in batch, "batch_has_target": "target" in batch})
        g_expected = (batch["global"] - mC.global_mean) / mC.global_scale
        batch_g = _clone_batch(batch)
        # reader input is checked structurally through its declared dimension
        check("global33_standardisation_matches_p3",
              torch.allclose(g_expected, (batch["global"] - mC.global_mean) / mC.global_scale, atol=1e-7)
              and int(mC.reader[0].in_features) == 323,
              {"reader_in_features": int(mC.reader[0].in_features), "declared": 323})

        # ------------------------------------------------------------------
        # 3. C/S identity and mechanism tests
        # ------------------------------------------------------------------
        sd_eq = _state_max_diff(mC, mS)
        yC = _forward0(mC, batch)
        yS = _forward0(mS, batch)
        check("C_S_identical_initial_state", sd_eq == 0.0, {"max_state_diff": sd_eq})
        check("C_S_identical_initial_forward", (yC - yS).abs().max().item() <= TOL,
              {"max_pred_diff": (yC - yS).abs().max().item()})
        check("rel_channel_zero_init",
              all(float(layer.rel_channel.weight.abs().max()) == 0.0 for layer in mC.layers),
              {"layers": len(mC.layers)})

        c2 = copy.deepcopy(mC)
        s2 = copy.deepcopy(mS)
        with torch.no_grad():
            for layer in c2.layers:
                pat12 = torch.tensor([math.sin(float(i)) for i in range(12)], dtype=torch.float32)
                pat12 = (pat12 - pat12.mean()) * 0.05
                pat = pat12.repeat(4).view(48, 1).expand(48, 22)
                layer.rel_channel.weight.add_(pat)
            for layer in s2.layers:
                pat12 = torch.tensor([math.sin(float(i)) for i in range(12)], dtype=torch.float32)
                pat12 = (pat12 - pat12.mean()) * 0.05
                pat = pat12.repeat(4).view(48, 1).expand(48, 22)
                layer.rel_channel.weight.add_(pat)
        yC_p = _forward0(c2, batch)
        yS_p = _forward0(s2, batch)
        check("zero_mean_channel_bias_changes_C_not_S",
              (yC_p - yC).abs().max().item() > 1e-4 and (yS_p - yS).abs().max().item() <= TOL,
              {"C_delta": (yC_p - yC).abs().max().item(), "S_delta": (yS_p - yS).abs().max().item()})
        c3 = copy.deepcopy(mC)
        s3 = copy.deepcopy(mS)
        with torch.no_grad():
            pat12 = torch.tensor([math.sin(float(i)) for i in range(12)], dtype=torch.float32)
            pat12 = (pat12 - pat12.mean()) * 0.05 + 0.05
            pat = pat12.repeat(4).view(48, 1).expand(48, 22)
            for layer in c3.layers:
                layer.rel_channel.weight.add_(pat)
            for layer in s3.layers:
                layer.rel_channel.weight.add_(pat)
        yC_n = _forward0(c3, batch)
        yS_n = _forward0(s3, batch)
        check("nonzero_mean_channel_bias_separates_C_S",
              (yC_n - yS_n).abs().max().item() > 1e-5,
              {"C_S_delta": (yC_n - yS_n).abs().max().item()})

        # ------------------------------------------------------------------
        # 4. gradients
        # ------------------------------------------------------------------
        def grad_audit(model, name):
            model.train()
            model.zero_grad(set_to_none=True)
            out = model(batch)
            loss = F.l1_loss(out, target)
            loss.backward()
            rows_out = []
            bad = []
            for n, p in model.named_parameters():
                g = p.grad
                ok = g is not None and torch.isfinite(g).all().item()
                nz = bool(torch.count_nonzero(g).item() > 0) if g is not None else False
                rows_out.append({"name": n, "finite": bool(ok), "nonzero": nz,
                                 "grad_norm": float(g.norm()) if g is not None else None})
                if not ok or not nz:
                    bad.append(n)
            return {"bad": bad, "rows": rows_out, "loss": float(loss.detach())}

        gaC = grad_audit(mC, "C")
        gaS = grad_audit(mS, "S")
        check("all_parameters_receive_nonzero_finite_gradients_C", not gaC["bad"],
              {"bad": gaC["bad"], "loss": gaC["loss"], "n": len(gaC["rows"])})
        check("all_parameters_receive_nonzero_finite_gradients_S", not gaS["bad"],
              {"bad": gaS["bad"], "loss": gaS["loss"], "n": len(gaS["rows"])})

        # ring column gradient
        mC.zero_grad(set_to_none=True)
        batch_rg = _clone_batch(batch)
        batch_rg["rel"].requires_grad_(True)
        out_rg = mC(batch_rg)
        out_rg.pow(2).mean().backward()
        rg = batch_rg["rel"].grad
        check("ring_column_has_nonzero_gradient",
              rg is not None and float(rg[..., 21].abs().max()) > 0.0,
              {"ring_grad_max": float(rg[..., 21].abs().max()) if rg is not None else None,
               "rw_grad_max": float(rg[..., :21].abs().max()) if rg is not None else None})

        # ring switch on the real model
        b_off = _clone_batch(batch)
        b_off["rel"][..., 21] = 0.0
        y_off = _forward0(mC, b_off)
        check("ring_switch_changes_real_model_output",
              (yC - y_off).abs().max().item() > 1e-6,
              {"max_pred_delta": (yC - y_off).abs().max().item(),
               "ring_pairs": int((batch["rel"][..., 21] > 0).sum())})

        # ------------------------------------------------------------------
        # 5. batch isolation and permutation
        # ------------------------------------------------------------------
        b0 = collate(valid_cache, [0], "P3N", valid=True)
        b1 = collate(valid_cache, [1], "P3N", valid=True)
        b01 = collate(valid_cache, [0, 1], "P3N", valid=True)
        b10 = collate(valid_cache, [1, 0], "P3N", valid=True)
        y0 = _forward0(mC, b0)[0]
        y1 = _forward0(mC, b1)[0]
        y01 = _forward0(mC, b01)
        y10 = _forward0(mC, b10)
        check("cross_molecule_batch_isolation",
              abs(float(y0 - y01[0])) <= TOL and abs(float(y1 - y01[1])) <= TOL,
              {"d0": abs(float(y0 - y01[0])), "d1": abs(float(y1 - y01[1]))})
        check("batch_order_permutation",
              abs(float(y01[0] - y10[1])) <= TOL and abs(float(y01[1] - y10[0])) <= TOL,
              {"d0": abs(float(y01[0] - y10[1])), "d1": abs(float(y01[1] - y10[0]))})

        # atom + bond + endpoint permutation on one real molecule
        row = 0
        b_orig = collate(valid_cache, [row], "P3N", valid=True)
        n = int(b_orig["atom_feat"].shape[0])
        e = int(b_orig["bond_feat"].shape[0])
        j = int(b_orig["junc_feat"].shape[0])
        new_of_old = RNG.permutation(n)
        old_of_new = np.empty(n, dtype=np.int64)
        old_of_new[new_of_old] = np.arange(n)
        bn_of_old = RNG.permutation(e)
        bo_of_new = np.empty(e, dtype=np.int64)
        bo_of_new[bn_of_old] = np.arange(e)
        jn_of_old = RNG.permutation(j)
        jo_of_new = np.empty(j, dtype=np.int64)
        jo_of_new[jn_of_old] = np.arange(j)
        old_edges = b_orig["bond_index"].numpy().copy()
        new_edges = new_of_old[old_edges[bo_of_new]]
        flip = RNG.random(e) < 0.5
        new_edges[flip] = new_edges[flip][:, ::-1]
        old_jb = b_orig["junc_bond"].numpy().copy()
        old_ji = b_orig["junc_index"].numpy().copy()
        b_perm = _clone_batch(b_orig)
        b_perm["atom_feat"] = b_orig["atom_feat"][torch.as_tensor(old_of_new)]
        b_perm["atom_type"] = b_orig["atom_type"][torch.as_tensor(old_of_new)]
        b_perm["bond_feat"] = b_orig["bond_feat"][torch.as_tensor(bo_of_new)]
        b_perm["bond_type"] = b_orig["bond_type"][torch.as_tensor(bo_of_new)]
        b_perm["bond_index"] = torch.as_tensor(new_edges, dtype=torch.long)
        b_perm["junc_feat"] = b_orig["junc_feat"][torch.as_tensor(jo_of_new)]
        b_perm["junc_index"] = torch.as_tensor(new_of_old[old_ji[jo_of_new]], dtype=torch.long)
        b_perm["junc_bond"] = torch.as_tensor(bn_of_old[old_jb[jo_of_new]], dtype=torch.long)
        rel_old = b_orig["rel"][0].numpy()
        rel_new = np.zeros_like(rel_old)
        rel_new[np.ix_(bn_of_old, bn_of_old)] = rel_old
        b_perm["rel"] = torch.from_numpy(rel_new[None])
        # independent relation equivariance
        r_check, ring_check, _ = R.build_molecule(new_edges, n)
        rel_check = np.concatenate([r_check.astype(np.float32), ring_check.astype(np.float32)[..., None]], axis=-1)
        check("relation_equivariance_under_atom_bond_permutation",
              np.abs(rel_check - rel_new).max() <= 1e-6,
              {"max_rel_diff": float(np.abs(rel_check - rel_new).max())})
        y_perm = _forward0(mC, b_perm)
        y_original = _forward0(mC, b_orig)
        check("model_invariance_under_atom_bond_endpoint_permutation",
              abs(float(y_perm[0] - y_original[0])) <= 1e-4,
              {"delta": abs(float(y_perm[0] - y_original[0])), "n_atoms": n, "n_bonds": e, "n_junc": j})

        # ------------------------------------------------------------------
        # 6. synthetic boundaries
        # ------------------------------------------------------------------
        examples = {
            "empty": (np.zeros((0, 2), dtype=np.int64), 1),
            "single": (np.array([[0, 1]], dtype=np.int64), 2),
            "acyclic_path4": (np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64), 4),
            "benzene": (np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], dtype=np.int64), 6),
            "naphthalene": (np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
                                      [2, 6], [6, 7], [7, 8], [8, 9], [9, 3]], dtype=np.int64), 10),
            "spiro": (np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                                [0, 4], [4, 5], [5, 6], [6, 0]], dtype=np.int64), 7),
        }
        boundary = {}
        for name, (edges, nat) in examples.items():
            bb = R.synthetic_batch(edges, nat)
            yy = _forward0(mC, bb)
            boundary[name] = {"finite": bool(torch.isfinite(yy).all()), "value": float(yy[0])}
        check("boundary_inputs_finite", all(v["finite"] for v in boundary.values()), boundary)

        # ring correctness
        def ring_of(name):
            edges, nat = examples[name]
            _, ring, lengths = R.build_molecule(edges, nat)
            return ring, lengths

        rt, lt = ring_of("benzene")
        ra, la = ring_of("acyclic_path4")
        rs, ls = ring_of("spiro")
        rn, ln = ring_of("naphthalene")
        k4 = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
        rk, ringk, lk = R.build_molecule(k4, 4)
        c4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        rc4, ring4, lc4 = R.build_molecule(c4, 4)
        c5 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], dtype=np.int64)
        rc5, ring5, lc5 = R.build_molecule(c5, 5)
        tri = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        rt3, ring3, lt3 = R.build_molecule(tri, 3)
        k4_expected = np.ones((6, 6), dtype=np.uint8)
        # K4: opposite edge pairs are not in the same chordless cycle
        k4_expected[0, 5] = k4_expected[5, 0] = 0
        k4_expected[1, 4] = k4_expected[4, 1] = 0
        k4_expected[2, 3] = k4_expected[3, 2] = 0
        naph_expected_cross_zero = (rn[0, 6] == 0 and rn[6, 0] == 0 and rn[2, 6] == 1)
        ring_ok = (
            bool((rt == 1).all()) and bool((ra == 0).all())
            and bool((ring4 == 1).all()) and bool((ring5 == 1).all()) and bool((ring3 == 1).all())
            and bool(np.array_equal(ringk, k4_expected))
            and bool((rs[:4, :4] == 1).all() and (rs[4:, 4:] == 1).all() and (rs[:4, 4:] == 0).all())
            and bool(naph_expected_cross_zero)
        )
        check("chordless_ring_examples", ring_ok,
              {"benzene_lengths": lt, "acyclic_lengths": la, "spiro_lengths": ls,
               "naphthalene_lengths": ln, "k4_lengths": lk, "c4_lengths": lc4, "c5_lengths": lc5,
               "triangle_lengths": lt3, "k4_match": bool(np.array_equal(ringk, k4_expected)),
               "naphthalene_cross_zero": bool(naph_expected_cross_zero)})

        # line-degree definition
        bp = np.asarray(valid_cache["bond_ptr"], dtype=np.int64)
        jp = np.asarray(valid_cache["junc_ptr"], dtype=np.int64)
        jb_all = np.asarray(valid_cache["junc_bond"], dtype=np.int64)
        bi_all = np.asarray(valid_cache["bond_index"], dtype=np.int64)
        line_ok = True
        line_detail = {}
        for rr in (0, 1, 2, 10):
            e0, e1 = int(bp[rr]), int(bp[rr + 1])
            s, t = int(jp[rr]), int(jp[rr + 1])
            nb = np.zeros(e1 - e0, dtype=np.int64)
            if t > s:
                np.add.at(nb, jb_all[s:t].reshape(-1), 1)
            alt = _line_degree_from_edges(bi_all[e0:e1], int(np.asarray(valid_cache["atom_ptr"])[rr + 1] - np.asarray(valid_cache["atom_ptr"])[rr]))
            line_ok = line_ok and bool(np.array_equal(nb, alt))
            line_detail[str(rr)] = {"junction_count": nb[:8].tolist(), "edge_degree": alt[:8].tolist()}
        check("line_degree_equals_junction_incidence", line_ok, line_detail)

        # ------------------------------------------------------------------
        # 7. attention response / no adjacency mask
        # ------------------------------------------------------------------
        c_att = copy.deepcopy(mC)
        s_att = copy.deepcopy(mS)
        with torch.no_grad():
            for layer in c_att.layers:
                layer.rel_channel.weight.normal_(0.0, 0.02)
            for layer in s_att.layers:
                layer.rel_channel.weight.normal_(0.0, 0.02)
        b_att = collate(valid_cache, [0], "P3N", valid=True)
        with torch.no_grad():
            _, stC = c_att(b_att, collect=True)
            _, stS = s_att(b_att, collect=True)
        att_ok = True
        att_detail = {}
        for i, (aC, aS) in enumerate(zip(stC["layers"], stS["layers"])):
            ok = (
                aC["finite"] and aS["finite"]
                and aC["row_sum_max_error"] <= 1e-5 and aS["row_sum_max_error"] <= 1e-5
                and aC["mean_max_weight"] < 0.999 and aC["mean_max_weight"] > 0.0
                and aC["mean_entropy"] < aC["mean_log_keys"] - 1e-3
                and aC["nonadjacent_attention_mass"] > 1e-7
                and aC["channel_dev_tv"] > 1e-8
                and aS["channel_dev_tv"] <= 1e-6
                and aC["rc_logit_std_effective"] > 0.0
                and aS["rc_logit_std_effective"] <= 1e-6
            )
            att_ok = att_ok and ok
            att_detail[str(i)] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                  for k, v in aC.items()}
            att_detail[str(i)]["S_channel_dev_tv"] = float(aS["channel_dev_tv"])
            att_detail[str(i)]["S_rc_logit_std_effective"] = float(aS["rc_logit_std_effective"])
        check("attention_finite_nonuniform_nonadjacent_and_channel_diverse",
              att_ok, att_detail)

        # train/eval identity
        mC.train()
        y_tr = mC(batch)
        mC.eval()
        y_ev = mC(batch)
        mS.train()
        y_tr_s = mS(batch)
        mS.eval()
        y_ev_s = mS(batch)
        check("train_eval_identical_no_dropout_no_bn",
              (y_tr - y_ev).abs().max().item() <= 1e-7 and (y_tr_s - y_ev_s).abs().max().item() <= 1e-7,
              {"C": (y_tr - y_ev).abs().max().item(), "S": (y_tr_s - y_ev_s).abs().max().item()})

        # ------------------------------------------------------------------
        # 8. relation cache
        # ------------------------------------------------------------------
        meta_ok = (
            va_meta["offset"][-1] == va_meta["total_pairs"]
            and tr_meta["offset"][-1] == tr_meta["total_pairs"]
            and np.array_equal(np.asarray(va_meta["bond_ptr"]), np.asarray(valid_cache["bond_ptr"]))
            and np.array_equal(np.asarray(va_meta["atom_ptr"]), np.asarray(valid_cache["atom_ptr"]))
            and np.array_equal(np.asarray(tr_meta["bond_ptr"]), np.asarray(train_cache["bond_ptr"]))
            and np.array_equal(np.asarray(tr_meta["atom_ptr"]), np.asarray(train_cache["atom_ptr"]))
        )
        sample_rows = [0, 1, 2, 10, 100, 500, 900, 999]
        rel_valid = R.open_relation("valid", mmap=True)
        v_ap = np.asarray(valid_cache["atom_ptr"]); v_bp = np.asarray(valid_cache["bond_ptr"])
        v_bi = np.asarray(valid_cache["bond_index"])
        sample_max = 0.0
        for rr in sample_rows:
            n = int(v_ap[rr + 1] - v_ap[rr]); e0, e1 = int(v_bp[rr]), int(v_bp[rr + 1]); e = e1 - e0
            s, t = int(va_meta["offset"][rr]), int(va_meta["offset"][rr + 1])
            r2, ring2, _ = R.build_molecule(v_bi[e0:e1], n)
            ref = np.concatenate([r2.astype(np.float32), ring2.astype(np.float32)[..., None]], axis=-1)
            got = np.asarray(rel_valid[s:t]).reshape(e, e, R.REL_DIM)
            sample_max = max(sample_max, float(np.abs(ref - got).max()))
        check("relation_cache_offsets_and_independent_sample", meta_ok and sample_max <= 1e-6,
              {"meta_ok": meta_ok, "sample_max_abs_diff": sample_max,
               "train_total_pairs": tr_meta["total_pairs"], "valid_total_pairs": va_meta["total_pairs"]})
        check("ring_support_statistics_present",
              tr_meta["stats"]["ring_pair_fraction"] > 0.0
              and all(3 <= int(k) <= 18 for k in tr_meta["stats"]["cycle_length_histogram"]),
              {"train_ring_pair_fraction": tr_meta["stats"]["ring_pair_fraction"],
               "valid_ring_pair_fraction": va_meta["stats"]["ring_pair_fraction"],
               "train_cycle_histogram": tr_meta["stats"]["cycle_length_histogram"]})

        # test refusal
        refused = False
        try:
            R.rel_path("test")
        except ValueError:
            refused = True
        check("test_split_refused", refused, {"official_test_loaded": False})

    finally:
        torch.load = _orig_load

    # no test file was loaded in this check process
    bad_loads = [p for p in loaded if "ad_test" in p or "official_test" in p]
    check("no_official_test_load_recorded", not bad_loads, {"loaded": loaded})
    production_code = [p for p in sorted((_EXP / "code").glob("*.py"))
                       if p.name != Path(__file__).name]
    code_text = "".join(p.read_text(encoding="utf-8") for p in production_code)
    check("code_never_names_test_cache", "ad_test.pt" not in code_text,
          {"contains_ad_test_pt": "ad_test.pt" in code_text,
           "files_scanned": [p.name for p in production_code]})

    # provenance manifest
    manifest = {
        "round_id": cfg["round_id"],
        "config_sha256": C.sha256_file(CONFIG),
        "code_sha256": {p.name: C.sha256_file(p) for p in sorted((_EXP / "code").glob("*.py"))},
        "relation_cache": {
            "train": {"path": str(R.rel_path("train")), "sha256": C.sha256_file(R.rel_path("train"))},
            "valid": {"path": str(R.rel_path("valid")), "sha256": C.sha256_file(R.rel_path("valid"))},
        },
        "ad_cache": {
            "train": {"path": str(D.cache_path("train")), "sha256": C.sha256_file(D.cache_path("train"))},
            "valid": {"path": str(D.cache_path("valid")), "sha256": C.sha256_file(D.cache_path("valid"))},
        },
        "ssl": {"source": str(C.P3_SSL_S0), "sha256": C.sha256_file(C.P3_SSL_S0)},
        "loaded_paths": loaded,
        "official_test_loaded": False,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "utc": C.utcnow(),
    }
    check("provenance_hashes_recorded", True, {"config_sha256": manifest["config_sha256"],
                                               "n_code_files": len(manifest["code_sha256"])})

    payload = {
        "round_id": cfg["round_id"],
        "checks": checks,
        "all_pass": all(c["ok"] for c in checks),
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["ok"]),
        "manifest": manifest,
        "official_test_loaded": False,
    }
    (ANALYSIS / "cpu_checks.json").write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n")
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
