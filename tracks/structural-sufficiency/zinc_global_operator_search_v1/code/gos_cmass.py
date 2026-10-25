"""Constant-mass control for GOS-v1 (one arm, one seed).

This module only supplies (a) the train-only initialisation of the 12 rho
scalars and (b) the small correctness gate for the new arithmetic.  Training
itself goes through the existing gos_runner / ad_train recipe.

    python code/gos_cmass.py init
    python code/gos_cmass.py check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import gos_data as GD  # noqa: E402
import gos_models as GM  # noqa: E402
import gos_runner as GR  # noqa: E402

ANALYSIS = _EXP / "analysis"
INIT_PATH = ANALYSIS / "cmass_init.json"
CHECKS_PATH = ANALYSIS / "cmass_checks.json"
EXPECTED_COMMON_INIT = "4b8c59f8eb54f6a0d2d63422fd92dfe9644466e8bbddcc4925316c0aa936ffd9"


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if np.any(y <= 0):
        raise ValueError("mean mass must be positive")
    return y + np.log(-np.expm1(-y))


def _capture_layer_mass(model):
    layers = list(model.layers)
    recs = [[] for _ in layers]
    restored = []
    for li, layer in enumerate(layers):
        orig = layer._attend
        restored.append((layer, orig))

        def wrap(*args, _orig=orig, _li=li):
            out = _orig(*args)
            recs[_li].append(out["mass"].detach().float().cpu().numpy())
            return out

        layer._attend = wrap
    return recs, restored


def _restore(restored):
    for layer, orig in restored:
        del layer._attend


def init() -> int:
    rel_train, rel_valid, scale, mean_nb = GR.prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    cache = D.load_cache("train")
    keep, hold = GR.holdout_split(1000, n_train=10000)
    norm = D.load_norm()
    model, info = GM.build_model("A", norm, 0, "cpu", mean_n_bonds=mean_nb,
                                 mix_mode="legacy")
    if info["initial_state_sha256"] != EXPECTED_COMMON_INIT:
        raise RuntimeError("common init hash changed: "
                           f"{info['initial_state_sha256']} != {EXPECTED_COMMON_INIT}")
    model.eval()
    recs, restored = _capture_layer_mass(model)
    sums = np.zeros((3, 4), dtype=np.float64)
    counts = np.zeros((3, 4), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(keep), 128):
            rows = keep[start:start + 128]
            batch = collate(cache, rows, "P3N", valid=False)
            _ = model(batch)
            nb = batch["rel_mask"].sum(1).numpy().astype(int)
            for li in range(3):
                m = recs[li][-1]                       # [B,H,N,1]
                for h in range(4):
                    for i in range(len(rows)):
                        n = int(nb[i])
                        sums[li, h] += float(m[i, h, :n, 0].sum())
                        counts[li, h] += n
    _restore(restored)
    mean_mass = sums / counts
    rho = _softplus_inv(mean_mass)
    out = {
        "kind": "constant-mass rho init from train-only unlabeled structure",
        "protocol": "zinc-global-operator-search-v1",
        "model": "route A, seed 0, common random+SSL init (no checkpoint loaded)",
        "common_init_sha256": info["initial_state_sha256"],
        "train_rows": int(len(keep)),
        "holdout_rows": int(len(hold)),
        "holdout_seed": GR.HOLDOUT_SEED,
        "n_queries_per_cell": counts.tolist(),
        "mean_mass": mean_mass.tolist(),
        "rho_init": rho.tolist(),
        "softplus_formula": "softplus(rho) = mean_mass",
        "labels_used": False,
        "valid_or_test_used": False,
        "note": "avg over all bond queries in the 9,000 training rows; the 1,000-row "
                "holdout is excluded, and no labels are read.",
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(INIT_PATH, out)
    print(json.dumps(out, indent=2))
    return 0


def _check(cond, name, detail):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return {"name": name, "pass": bool(cond), "detail": detail}


def check() -> int:
    init_rec = C.read_json(INIT_PATH) if INIT_PATH.exists() else None
    if init_rec is None:
        raise FileNotFoundError(INIT_PATH)
    rel_train, rel_valid, scale, mean_nb = GR.prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    norm = D.load_norm()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    checks = []

    # 1. same common initialisation (all tensors except rho)
    legacy, l_info = GM.build_model("A", norm, 0, "cpu", mean_n_bonds=mean_nb,
                                    mix_mode="legacy")
    cmass, c_info = GM.build_model("A", norm, 0, "cpu", mean_n_bonds=mean_nb,
                                   mix_mode="cmass")
    lsd, csd = legacy.state_dict(), cmass.state_dict()
    common_keys = [k for k in lsd if not k.endswith(".rho")]
    max_init_diff = max(float((lsd[k] - csd[k]).abs().max()) for k in common_keys)
    checks.append(_check(
        max_init_diff == 0.0 and l_info["initial_state_sha256"] == EXPECTED_COMMON_INIT,
        "same public init",
        f"legacy sha {l_info['initial_state_sha256'][:16]}, "
        f"max non-rho tensor diff {max_init_diff:.3g}, "
        f"cmass params {c_info['parameters']} (legacy {l_info['parameters']})"))

    # 2. formula / fp64 equivalence to (a*g)/sum(a*g) times the learned scalar
    legacy.double()
    cmass.double()
    torch.manual_seed(7)
    B, N = 2, 6
    h = torch.randn(B, N, GM.TOKEN_DIM, dtype=torch.float64)
    rel = torch.randn(B, N, N, GM.REL_REL_DIM, dtype=torch.float64)
    mask = torch.ones(B, N, dtype=torch.bool)
    with torch.no_grad():
        lo = legacy.layers[0].mix(h, rel, mask)
        co = cmass.layers[0].mix(h, rel, mask)
        scale = torch.nn.functional.softplus(cmass.layers[0].rho).view(1, GM.HEADS, 1, 1)
        expected = scale * lo["ctx"] / lo["mass"]
        diff = (co["ctx"] - expected).abs().max().item()
        mass_err = (co["mass"] - scale).abs().max().item()
        attn_err = (co["attn"].sum(-1) - 1.0).abs().max().item()
    checks.append(_check(diff < 1e-10 and mass_err == 0.0 and attn_err < 1e-12,
                         "formula equivalence (fp64)",
                         f"max ctx diff {diff:.3g}, mass-scale diff {mass_err:.3g}, "
                         f"row-sum error {attn_err:.3g}"))

    # 3. padding / batch invariance on a real valid batch
    cmass.float()
    ck = torch.load(_EXP / "runs" / "ho_A_s0" / "best.pt", map_location="cpu",
                    weights_only=False)
    # load the common non-rho weights, then the frozen rho init
    cmass.load_state_dict({k: v for k, v in ck["state_dict"].items()
                           if not k.endswith(".rho")}, strict=False)
    GR._apply_cmass_init(cmass, {}, "cmass")
    cmass.eval()
    rows = np.arange(16, dtype=np.int64)
    with torch.no_grad():
        full = cmass(collate(valid_cache, rows, "P3N", valid=True)).numpy()
        singles = np.asarray([
            cmass(collate(valid_cache, np.asarray([r]), "P3N", valid=True)).item()
            for r in rows])
    pad_diff = float(np.abs(full - singles).max())
    checks.append(_check(pad_diff < 2e-5, "padding / batch invariance",
                         f"max |batch - single| {pad_diff:.3g}"))

    # 4. finite gradients, including rho
    cmass.train()
    batch = collate(train_cache, np.arange(32, dtype=np.int64), "P3N", valid=False)
    pred = cmass(batch)
    y = torch.as_tensor(np.asarray(train_cache["labels"])[:32], dtype=torch.float32)
    loss = torch.nn.functional.l1_loss(pred, y)
    loss.backward()
    bad = [n for n, p in cmass.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    rho_grad_ok = all(torch.isfinite(layer.rho.grad).all().item()
                      for layer in cmass.layers)
    checks.append(_check(len(bad) == 0 and rho_grad_ok, "finite gradients",
                         f"non-finite tensors {bad}, rho grad finite {rho_grad_ok}, "
                         f"loss {float(loss):.6g}"))

    # 5. checkpoint reload
    cmass.eval()
    with torch.no_grad():
        before = cmass(collate(valid_cache, np.arange(32, dtype=np.int64),
                               "P3N", valid=True)).numpy()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cmass.pt"
        torch.save({"state_dict": C.cpu_state(cmass), "epoch": 0}, p)
        fresh, _ = GM.build_model("A", norm, 0, "cpu", mean_n_bonds=mean_nb,
                                  mix_mode="cmass")
        fresh.load_state_dict(torch.load(p, map_location="cpu",
                                         weights_only=False)["state_dict"])
        fresh.eval()
        with torch.no_grad():
            after = fresh(collate(valid_cache, np.arange(32, dtype=np.int64),
                                  "P3N", valid=True)).numpy()
    reload_diff = float(np.abs(before - after).max())
    checks.append(_check(reload_diff == 0.0, "checkpoint reload",
                         f"max prediction diff {reload_diff:.3g}"))

    out = {
        "protocol": "zinc-global-operator-search-v1",
        "kind": "constant-mass focused correctness gate (not the 25-item erratum gate)",
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "init_file": str(INIT_PATH.relative_to(C.REPO)),
        "params": int(c_info["parameters"]),
    }
    C.write_json(CHECKS_PATH, out)
    print(f"all_pass={out['all_pass']} params={out['params']}")
    return 0 if out["all_pass"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["init", "check"])
    args = p.parse_args(argv)
    return init() if args.command == "init" else check()


if __name__ == "__main__":
    raise SystemExit(main())
