"""Phase 0.2 — mechanism-scope verification for the Full and the X175/D prototype.

Read-only over the historical prep blob.  Builds the canonical Full (arm F) and
the joint dictionary prototype (arm D) from the SAVED post-prep initial state and
checks, with `named_parameters` / `requires_grad` / optimizer parameter IDs and
one real backward pass, which dictionaries are end-to-end trainable and which
losses drive them.  This is an *initial-state* check: it does not and cannot
verify the final soup weights (no soup checkpoint was saved).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

OUT = Path(__file__).resolve().parent


def grads_for(model, batch, kind, arm):
    model.zero_grad(set_to_none=True)
    pred, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    if kind == "task":
        loss = torch.nn.functional.l1_loss(pred.view(-1), batch.y.view(-1))
    elif arm == "D":
        loss = zjd.relative_recon(aux["x175"], aux["alpha"], model.local.dictionary)
    else:
        loss = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss.backward()
    out = {}
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.norm() > 0:
            out[name] = float(p.grad.norm())
    return float(loss.detach()), out


def main() -> int:
    torch.set_num_threads(8)
    blob = zjd.load_prep_blob()
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    zjd.apply_prep_blob(train_data, valid_data, blob)
    fit_data, dev_data = zjd._build_fit_dev(train_data, {"fit_idx": blob["fit_idx"], "dev_idx": blob["dev_idx"]})
    batch = next(iter(torch.utils.data.DataLoader(list(fit_data[:32]), batch_size=32, collate_fn=p1.env_collate)))

    report = {"official_test_loaded": False, "note": "post-prep INITIAL state; no soup checkpoint exists"}
    for arm in ("F", "D"):
        model = zjd.make_arm_model(arm, blob).to("cpu")
        named = {n: {"shape": list(p.shape), "requires_grad": bool(p.requires_grad)}
                 for n, p in model.named_parameters()}
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt_ids = {id(p) for group in opt.param_groups for p in group["params"]}
        dict_names = [n for n in named if ("local_dictionary_bridge" in n) or n == "D"
                      or n.endswith("local.dictionary") or n.endswith("local.value.weight")]
        dict_ids = {id(p) for n, p in model.named_parameters() if n in dict_names}
        task_loss, task_grads = grads_for(model, batch, "task", arm)
        rec_loss, rec_grads = grads_for(model, batch, "rec", arm)
        report[arm] = {
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "dictionary_params": {n: named[n] for n in dict_names},
            "all_dict_params_requires_grad": all(named[n]["requires_grad"] for n in dict_names),
            "all_dict_params_in_optimizer": dict_ids <= opt_ids,
            "task_loss": task_loss,
            "reconstruction_loss": rec_loss,
            "task_grad_norm_on_dict": {n: task_grads.get(n) for n in dict_names},
            "rec_grad_norm_on_dict": {n: rec_grads.get(n) for n in dict_names},
        }
        print(arm, json.dumps({k: v for k, v in report[arm].items()
                               if k not in ("task_grad_norm_on_dict", "rec_grad_norm_on_dict")}, indent=2))
        print(arm, "task grads", report[arm]["task_grad_norm_on_dict"])
        print(arm, "rec  grads", report[arm]["rec_grad_norm_on_dict"])

    (OUT / "mechanism_check.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
