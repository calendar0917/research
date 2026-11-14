"""No-data toy schedule trace: verify realised lr endpoints before any formal run.

Checks (recorded to ``analysis/toy_schedule.json``):

* C1: step 0 lr = peak, lr at U0 = min, and constant min afterwards (the E2 leg);
* D-MPNN: real ``param_group['lr']`` (not the declared schedule) goes
  init_lr -> peak at the frozen warmup step -> min at U0, then constant min.

The historical ERRATA-E4 bug was exactly a declared-correct but realised-wrong lr,
so this trace reads the optimiser's actual param-group lr after each ``step()``.
"""

from __future__ import annotations

from typing import Any

import torch

import lb_common as C
import lb_data as D
import lb_train as T

import lcpr_dmpnn as PDN


def _trace(lambda_or_cosine, *, total: int, peak: float, minlr: float, optim: str,
           init: float | None = None, warmup: int = 0) -> list[float]:
    p = torch.nn.Parameter(torch.zeros(1))
    if optim == "adamw":
        opt = torch.optim.AdamW([p], lr=peak)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1), eta_min=minlr)
        lrs = [opt.param_groups[0]["lr"]]
        for s in range(total + 5):
            opt.step()
            if s < total:
                sch.step()
            else:
                for g in opt.param_groups:
                    g["lr"] = minlr
            lrs.append(opt.param_groups[0]["lr"])
        return lrs
    base = float(init)
    opt = torch.optim.Adam([p], lr=base)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: PDN._noam_lambda(s, warmup, max(total - warmup, 1), init, peak, minlr))
    lrs = [opt.param_groups[0]["lr"]]
    for s in range(total + 5):
        opt.step()
        if s < total:
            sch.step()
        else:
            for g in opt.param_groups:
                g["lr"] = minlr
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


def toy_trace() -> dict[str, Any]:
    out: dict[str, Any] = {"created_utc": C.utcnow(), "checks": {}}
    for model, ds in (("C1_R1", "Lipophilicity"), ("C1_R1", "BACE"),
                      ("DMPNN_D2F", "Lipophilicity"), ("DMPNN_D2F", "BACE")):
        U0 = C.u0(model, ds)
        r = C.C1_R1 if model.startswith("C1") else C.DMPNN_D2F
        if model.startswith("C1"):
            lrs = _trace(None, total=U0, peak=r["peak_lr"], minlr=r["min_lr"], optim="adamw")
            warm = 0
        else:
            warm = C.dmpnn_warmup_steps(ds)
            lrs = _trace(None, total=U0, peak=r["peak_lr"], minlr=r["min_lr"], optim="adam",
                         init=r["init_lr"], warmup=warm)
        entry = {
            "U0": U0, "warmup_steps": warm,
            "lr_step0": lrs[0],
            "lr_at_warmup_end": lrs[warm] if warm else None,
            "lr_at_U0": lrs[U0],
            "lr_after_U0_plus5": lrs[-1],
            "peak_declared": r["peak_lr"], "init_declared": r.get("init_lr"), "min_declared": r["min_lr"],
        }
        # assertions that catch a declared-vs-realised mismatch
        if model.startswith("C1"):
            assert abs(entry["lr_step0"] - r["peak_lr"]) < 1e-9
            assert abs(entry["lr_at_U0"] - r["min_lr"]) < 1e-6
        else:
            assert abs(entry["lr_step0"] - r["init_lr"]) < 1e-6, entry["lr_step0"]
            assert abs(entry["lr_at_warmup_end"] - r["peak_lr"]) < 1e-3, entry["lr_at_warmup_end"]
            assert abs(entry["lr_at_U0"] - r["min_lr"]) < 1e-6, entry["lr_at_U0"]
        assert abs(entry["lr_after_U0_plus5"] - r["min_lr"]) < 1e-12
        out["checks"][f"{model}@{ds}"] = entry
    C.write_json(C.ANALYSIS / "toy_schedule.json", out)
    return out


if __name__ == "__main__":
    print(toy_trace())
