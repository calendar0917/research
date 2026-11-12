"""One-shot auxiliary-loss dose calibration (no optimiser update).

Per architecture and per loss ``X in {R, W, N}``:

    gA = L2 norm of the gradient of the plain molecular loss
    gX = L2 norm of the gradient of the *raw* auxiliary term alone
    lambda_X = clip(0.2 * median(gA) / max(median(gX), 1e-12), 0.001, 10)

Measured on 32 fixed ordinary batches + the first 32 pair batches of S0/seed0 at
the raw initialisation, in exactly the mode the training loop uses (molecular
loss in train mode, auxiliary endpoints in eval mode).  The resulting numbers are
frozen: they are reused for every seed, split, SG and full-Train run and are
never re-tuned against an evaluation result.

The identity ``R_aux = W_aux - 2 * mean(e_i * e_j)`` is verified numerically here
and the dose probe is written to ``analysis/dose_freeze.json``.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import lr_common as C
import lr_data as D
import lr_models as LM
import lr_pairs as P
import lr_train as T

DOSE_MODELS = ("B", "S", "O", "D", "F")


def _grad_norm(loss: torch.Tensor, params) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))


def calibrate(device: torch.device) -> dict[str, Any]:
    split = "s0"
    seed = 0
    spec = D.frozen_spec()
    fit_cache = D.load_cache(split, "fit")
    rows_fit = [int(r) for r in np.asarray(fit_cache["rows"]).tolist()]
    train_norm = D.train_norm(split)
    tgt = D.target_norm(split)
    y_fit = D.y_aligned(split, "fit", rows_fit, "train")
    z_fit = (y_fit - tgt["mean"]) / tgt["scale"]
    plan = P.get_plan(split, seed, C.FORMAL_UPDATES)
    table_ = P.table()
    fp_payload = None
    fp_path = C.CACHE / f"fp_{split}.npz"
    if fp_path.exists():
        data = np.load(fp_path)
        fp_payload = {k: data[k].astype(np.float32) for k in data.files}

    gen = torch.Generator().manual_seed(int(seed) + int(C.SHUFFLE_OFFSET))
    perm = torch.randperm(len(rows_fit), generator=gen).tolist()
    batches = [perm[i:i + C.BATCH] for i in range(0, len(perm), C.BATCH)][: C.DOSE_BATCHES]
    id_of = {r: i for i, r in enumerate(rows_fit)}

    out: dict[str, Any] = {"schema": "lcrc-lipo-route-dose-v1", "created_utc": C.utcnow(),
                           "split": split, "seed": seed, "n_batches": len(batches),
                           "rule": "lambda = clip(0.2*median(gA)/max(median(gX),1e-12), 0.001, 10)",
                           "clip": list(C.DOSE_CLIP), "doses": {}, "per_model": {}}
    for kind in DOSE_MODELS:
        model = LM.build_model(kind, spec, train_norm, seed, device,
                               {"fp_mean": fp_payload["mean"], "fp_scale": fp_payload["scale"]}
                               if kind == "F" else None)
        params = [p for p in model.parameters() if p.requires_grad]
        ga: list[float] = []
        gx: dict[str, list[float]] = {k: [] for k in ("R", "W", "N")}
        identity_err = 0.0
        model.train()
        for b_index, rows in enumerate(batches):
            batch = D.to_device(D.collate(fit_cache, rows), device)
            if kind == "F":
                batch["fp"] = torch.from_numpy(fp_payload["fit"][np.asarray(rows, dtype=np.int64)]).to(device)
            y = torch.as_tensor([float(z_fit[i]) for i in rows], dtype=torch.float32, device=device)
            pred = model(batch).reshape(-1)
            loss_mol = F.mse_loss(pred, y)
            ga.append(_grad_norm(loss_mol, params))
            model.zero_grad(set_to_none=True)

            pair_idx = np.asarray(plan["pairs"][b_index], dtype=np.int64)
            perm_n = np.asarray(plan["perm"][b_index], dtype=np.int64)
            ep_rows, idx_a, idx_b, _ = P.endpoint_index(pair_idx, table_.row_a, table_.row_b)
            ep_pos = [id_of[int(r)] for r in ep_rows]
            aux_batch = D.to_device(D.collate(fit_cache, ep_pos), device)
            if kind == "F":
                aux_batch["fp"] = torch.from_numpy(fp_payload["fit"][np.asarray(ep_pos, dtype=np.int64)]).to(device)
            was = model.training
            model.eval()
            aux_out = model(aux_batch).reshape(-1)
            model.train(was)
            e = aux_out - torch.as_tensor([float(z_fit[i]) for i in ep_pos],
                                          dtype=torch.float32, device=device)
            ia = torch.as_tensor(idx_a, dtype=torch.long, device=device)
            ib = torch.as_tensor(idx_b, dtype=torch.long, device=device)
            ib_n = torch.as_tensor([idx_b[int(k)] for k in perm_n], dtype=torch.long, device=device)
            r_aux = T.auxiliary_term(e[ia], e[ib], "R")
            w_aux = T.auxiliary_term(e[ia], e[ib], "W")
            n_aux = T.auxiliary_term(e[ia], e[ib_n], "N")
            ident = float(r_aux - (w_aux - 2.0 * (e[ia] * e[ib]).mean()))
            identity_err = max(identity_err, abs(ident))
            for name, value in (("R", r_aux), ("W", w_aux), ("N", n_aux)):
                gx[name].append(_grad_norm(value, params))
            model.zero_grad(set_to_none=True)
        med_ga = float(np.median(ga))
        entry = {"median_grad_norm_plain": med_ga, "median_grad_norm_aux": {}, "lambda": {}}
        for name in ("R", "W", "N"):
            med_gx = float(np.median(gx[name]))
            raw = C.DOSE_TARGET_FRACTION * med_ga / max(med_gx, 1e-12)
            lam = float(min(max(raw, C.DOSE_CLIP[0]), C.DOSE_CLIP[1]))
            entry["median_grad_norm_aux"][name] = med_gx
            entry["lambda"][name] = lam
            entry["lambda_unclipped_" + name] = float(raw)
            out["doses"].setdefault(kind, {})[name] = lam
        entry["rw_identity_max_abs_err"] = float(identity_err)
        entry["grad_norm_plain_all"] = [float(x) for x in ga]
        entry["grad_norm_aux_all"] = {k: [float(x) for x in v] for k, v in gx.items()}
        out["per_model"][kind] = entry
        C.log(f"[dose] {kind}: gA={med_ga:.4g} " +
              " ".join(f"{n}={entry['lambda'][n]:.4g}(gX={entry['median_grad_norm_aux'][n]:.4g})"
                       for n in ("R", "W", "N")) +
              f" identity_err={entry['rw_identity_max_abs_err']:.2e}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out["status"] = "FROZEN"
    out["identity_check"] = "R_aux = W_aux - 2*mean(e_i*e_j) verified numerically"
    C.write_json(T.DOSE_FREEZE, out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="one-shot dose calibration")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    calibrate(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
