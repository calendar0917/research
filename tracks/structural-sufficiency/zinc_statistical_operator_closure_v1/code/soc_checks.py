"""CPU-only gates for SOC-v1.  No GPU, no official test, no official valid labels.

Run:  uv run python .../soc_checks.py
Writes analysis/cpu_gates.json and exits non-zero if a hard gate fails.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code"),
           str(_SUFF / "zinc_dual_channel_composer_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dcc_models as DM  # noqa: E402
import soc_common as SC  # noqa: E402
import soc_models as SM  # noqa: E402

RESULTS: dict = {}


def gate(name: str, ok: bool, detail) -> None:
    RESULTS[name] = {"ok": bool(ok), "detail": detail}
    SC.log(f"[gate] {name}: {'PASS' if ok else 'FAIL'} {detail}")


def max_abs(a, b) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def main() -> int:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cfg = SC.load_config()
    cache = D.load_cache("train")
    norm = D.load_norm()
    rows = np.arange(32, dtype=np.int64)
    batch = T1.collate(cache, rows, "P3N", valid=True)

    # ---- G0: parameter counts ------------------------------------------------
    m, info = SM.build_initial_state(norm, 0)
    ctrl, cinfo = SM.build_control_state(norm, 0, "cpu", reader_hidden=int(cfg["control_reader_hidden"]))
    gate("G0_params",
         int(info["parameters"]) == int(cfg["expected_params"]["R"]) and
         int(cinfo["parameters"]) == int(cfg["expected_params"]["CTRL"]),
         {"soc": int(info["parameters"]), "ctrl": int(cinfo["parameters"]),
          "ratio": float(cinfo["parameters"] / info["parameters"])})

    # ---- G1: exact DCC-L initialisation -------------------------------------
    dm, _ = DM.build_initial_state(norm, 0)
    dm.eval()
    m.eval()
    ctrl.eval()
    with torch.no_grad():
        base = dm(batch)
        m.closure.gate.fill_(-30.0)
        got_final = m(batch)
        m_off = m
        mmean, _ = SM.build_initial_state(norm, 0, readout="mean")
        mmean.eval()
        mmean.closure.gate.fill_(-30.0)
        got_mean = mmean(batch)
        got_ctrl = ctrl(batch)
    d_final = max_abs(base, got_final)
    d_mean = max_abs(base, got_mean)
    d_ctrl = max_abs(base, got_ctrl)
    gate("G1_init_equivalence", max(d_final, d_mean) <= 1e-8 and d_ctrl <= 1e-5,
         {"final": d_final, "mean": d_mean, "ctrl": d_ctrl, "tol": "1e-8 / 1e-5"})

    # ---- G2: finite forward/backward and gradient to the closure ------------
    m.train()
    with torch.no_grad():
        m.closure.gate.fill_(-2.0)
    out = m(batch)
    loss = out.abs().mean()
    m.zero_grad(set_to_none=True)
    loss.backward()
    grads = {name: float(p.grad.abs().sum()) for name, p in m.closure.named_parameters()}
    finite = bool(torch.isfinite(out).all() and all(np.isfinite(list(grads.values()))))
    gate("G2_gradients", finite and grads["proj.weight"] > 0 and grads["core.0.weight"] > 0
         and grads["gate"] > 0, {"loss": float(loss), "grads": grads})

    # ---- G3: batch-composition invariance (no cross-molecule leakage) -------
    m.eval()
    with torch.no_grad():
        m.closure.gate.fill_(-2.0)
    p16, _ = SC.evaluate(m, cache, np.arange(16), device="cpu")
    p32, _ = SC.evaluate(m, cache, np.arange(32), device="cpu")
    d = float(np.max(np.abs(p16 - p32[:16])))
    gate("G3_batch_invariance", d <= 1e-5, {"max_abs": d, "n": 16})

    # ---- G4: activation actually changes the computation --------------------
    m.eval()
    with torch.no_grad():
        m.closure.gate.fill_(-30.0)
        off = m(batch)
        m.closure.gate.fill_(-2.0)
        on = m(batch)
        (a0, b0, w0), traj = m.closure(*m.encoders(batch), batch, return_trajectory=True)
    drift = float(torch.max(torch.abs(traj[-1][1] - traj[0][1])).item())
    gate("G4_activation", float(torch.max(torch.abs(on - off)).item()) > 1e-4 and drift > 1e-5,
         {"pred_delta": float(torch.max(torch.abs(on - off)).item()),
          "bond_state_drift": drift, "K": int(info["K"])})

    # ---- G5: weight tying / K-independence of parameter count ---------------
    m1, i1 = SM.build_initial_state(norm, 0, K=1)
    m3, i3 = SM.build_initial_state(norm, 0, K=3)
    gate("G5_weight_tying", int(i1["parameters"]) == int(i3["parameters"]),
         {"K1_params": int(i1["parameters"]), "K3_params": int(i3["parameters"]),
          "closure_modules": int(sum(1 for _ in m3.closure.modules()))})

    # ---- G6: overfit a tiny subset ------------------------------------------
    tiny = np.arange(8, dtype=np.int64)
    tb = T1.collate(cache, tiny, "P3N", valid=False)
    ty = torch.as_tensor(np.asarray(cache["labels"])[tiny], dtype=torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    m.train()
    t0 = time.perf_counter()
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        pred = m(tb)
        l = F.l1_loss(pred.view(-1), ty.view(-1))
        l.backward()
        opt.step()
        if step == 0:
            first = float(l)
    last = float(l)
    gate("G6_overfit", last < 0.6 * first, {"first": first, "last": last,
                                            "seconds": time.perf_counter() - t0})

    # ---- G7: stress at a fully open gate ------------------------------------
    ms, _ = SM.build_initial_state(norm, 0, K=3)
    ms.eval()
    with torch.no_grad():
        ms.closure.gate.fill_(10.0)
        big = ms(batch)
    gate("G7_open_gate_finite", bool(torch.isfinite(big).all()),
         {"max_abs": float(torch.max(torch.abs(big)).item())})

    # ---- G8: readout semantics ---------------------------------------------
    mm, _ = SM.build_initial_state(norm, 0, readout="mean")
    mm.eval()
    with torch.no_grad():
        mm.closure.gate.fill_(-2.0)
        mean_out = mm(batch)
        m.closure.gate.fill_(-2.0)
        final_out = m(batch)
    gate("G8_readouts_distinct", float(torch.max(torch.abs(mean_out - final_out)).item()) > 1e-5,
         {"delta": float(torch.max(torch.abs(mean_out - final_out)).item())})

    RESULTS["meta"] = {"parameters": int(info["parameters"]),
                       "parameters_closure": int(info["parameters_closure"]),
                       "K": int(info["K"]), "rank": int(info["rank"]),
                       "readout_final": int(mm.readout == "mean"),
                       "config": cfg,
                       "torch": torch.__version__}
    out = SC.ANALYSIS / "cpu_gates.json"
    SC.write_json(out, RESULTS)
    hard = [k for k, v in RESULTS.items() if k.startswith("G") and not v["ok"]]
    SC.log(f"[gates] {len(RESULTS) - 1} gates, failures={hard}")
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
