"""Local CPU self-checks for the semantic-table construction (no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se_common as C  # noqa: E402
import se_tables as T  # noqa: E402
import se_model as M  # noqa: E402

import lp_common as LPC  # noqa: E402
import lp_data as LPD  # noqa: E402
import dissect_model as DM  # noqa: E402
import lcrc_data as PREP_DATA  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    LPC.setup_determinism(strict=False)
    view = LPD.build_view()
    spec, norm = view.spec, view.train_norm
    report: dict = {}

    # codec
    rt = T.codec_roundtrip()
    assert rt["path_roundtrip"] and rt["binary_roundtrip"], rt
    report["codec"] = rt

    # parameter counts
    perms = T.build_permutations()
    models = {}
    for fam in C.SEM_FAMILIES:
        models[fam] = M.build_semantic_model(spec, norm, fam, permutations=perms)
    counts = {fam: M.parameter_count(m) for fam, m in models.items()}
    report["parameters"] = counts
    report["expected_free_table_params"] = C.FREE_TABLE_PARAMS
    report["expected_generator_params"] = C.GENERATOR_PARAMS
    report["expected_FS_params"] = C.FS_PARAMS

    # shared generator registered once
    fs = models["FS"]
    gen_ids = {id(p) for p in fs.generator.parameters()}
    all_ids = [id(p) for p in fs.parameters()]
    report["generator_registered_once"] = all(all_ids.count(g) == 1 for g in gen_ids)
    report["n_generator_params_registered"] = int(sum(1 for i in all_ids if i in gen_ids))

    # epoch-0 prediction equality inside each pair
    batch = PREP_DATA.collate(view.caches["valid"], list(range(16)))
    batch = LPD.to_device(batch, torch.device("cpu"))
    preds = {}
    for fam, m in models.items():
        m.eval()
        with torch.inference_mode():
            preds[fam] = m(batch).detach().cpu().numpy().reshape(-1)
    d_sem = float(np.max(np.abs(preds["LS"] - preds["FS"])))
    d_perm = float(np.max(np.abs(preds["LP"] - preds["FP"])))
    report["epoch0_max_abs_LS_FS"] = d_sem
    report["epoch0_max_abs_LP_FP"] = d_perm
    report["epoch0_max_abs_LS_LP"] = float(np.max(np.abs(preds["LS"] - preds["LP"])))

    # gate init = 1
    fs.eval()
    gate = fs.base.local.path_gate(torch.arange(C.N_PATH_CODES, dtype=torch.long))
    report["gate_init"] = {"min": float(gate.min()), "max": float(gate.max())}

    # all four finite
    report["finite"] = {fam: bool(np.isfinite(p).all()) for fam, p in preds.items()}

    # permutation bijection / consistency
    report["perm"] = {
        "path_is_perm": bool(np.array_equal(np.sort(perms["path"]), np.arange(C.N_PATH_CODES))),
        "binary_is_perm": bool(np.array_equal(np.sort(perms["binary"]), np.arange(C.N_NBR_CODES))),
    }

    # gradient reaches shared composition tables from every bank
    fs.train()
    out = fs(batch).sum()
    out.backward()
    grads = {
        "atom": float(fs.generator.atom.weight.grad.abs().sum()) if fs.generator.atom.weight.grad is not None else 0.0,
        "bond": float(fs.generator.bond.weight.grad.abs().sum()) if fs.generator.bond.weight.grad is not None else 0.0,
    }
    report["shared_grad"] = grads
    for name, p in fs.generator.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            report.setdefault("nonfinite_or_missing_grad", []).append(name)

    print("=== report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    ok = (
        d_sem < 1e-6 and d_perm < 1e-6 and np.isfinite(d_sem) and np.isfinite(d_perm)
        and report["finite"] == {k: True for k in preds}
        and report["generator_registered_once"]
    )
    print("SELFCHECK_OK", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
