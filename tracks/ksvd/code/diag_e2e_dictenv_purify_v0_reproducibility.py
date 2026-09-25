"""Post-STOP reproducibility diagnosis for E2E-DictEnv-Purify-v0 (read-only, GPU1).

The frozen pre-registration (section 10.2) stopped the round with
``REFERENCE_REPRODUCTION_FAILURE``: the fresh reference seed-0 soup
(0.129878) is +0.006329 above the historical H1 soup (0.123549) on the same
machine, with an identical data cache, identical dictionary sha256, identical
model/training code (no commits touched either module since H1's commit),
identical init path (``p2.build_model`` calls ``torch.manual_seed(0)``),
identical lambda, optimizer, batch order (generator-seeded) and *no dropout*.

This script isolates the cause.  It trains nothing to completion and writes no
round artifact; it answers two questions:

probe A  is the formal training protocol bit-reproducible at all?  Two runs of
         the same code, same seed, same GPU, two epochs each, compared epoch by
         epoch (epoch body copied verbatim from the formal ``train_arm`` loop).

probe B  does an IHT top-``s`` support survive a repeated forward pass?  The
         pooled read-outs use ``index_add_``, whose CUDA accumulation order is
         not deterministic; if that float noise flips top-``s`` code supports,
         the discrete support makes the trajectory chaotic and run-to-run
         dispersion at the 0.005 level is expected rather than exceptional.

probe C  the same forward comparison on CPU (expected exactly deterministic),
         as the contrast that proves the effect is the CUDA reduction regime.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_purify_v0 as pur
from tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_purify_v0 import (
    BATCH_SIZE,
    EVAL_SHUFFLE_OFFSET,
    GRAD_CLIP,
    LEARNING_RATE,
    TRAIN_SHUFFLE_OFFSET,
    WEIGHT_DECAY,
    dictionary,
    evaluate as evaluate_model,
    load_split,
    resolve_device,
)


def _batch(device: torch.device) -> object:
    data = load_split("train", subset=BATCH_SIZE)
    loader = p1.make_env_loader(data, BATCH_SIZE, False, 0)
    return next(iter(loader)).to(device)


def probe_forward(device: torch.device, n_molecules: int = 128) -> dict[str, object]:
    D, _sha = dictionary()
    data = load_split("train", subset=int(n_molecules))
    loader = p1.make_env_loader(data, int(n_molecules), False, 0)
    batch = next(iter(loader)).to(device)
    model = pur.build_model(pur.reference_config(), D, seed=0).to(device).eval()
    with torch.no_grad():
        first, first_aux = model(batch, return_aux=True)
        second, second_aux = model(batch, return_aux=True)
    codes_a, codes_b = first_aux["coord"], second_aux["coord"]
    support_a, support_b = codes_a > 0, codes_b > 0
    flips = (support_a != support_b).any(dim=1)
    scalar = {}
    for key in ("pair_value", "unary", "relation_readout", "unified", "E"):
        if key in first_aux:
            left = first_aux[key].float()
            right = second_aux[key].float()
            scalar[key] = float((left - right).abs().max().item()) if left.shape == right.shape else None
    return {
        "device": str(device),
        "n_nodes": int(codes_a.shape[0]),
        "prediction_max_abs_diff": float((first - second).abs().max().item()),
        "codes_max_abs_diff": float((codes_a - codes_b).abs().max().item()),
        "codes_entries_gt_1e_6": int(((codes_a - codes_b).abs() > 1.0e-6).sum().item()),
        "support_flip_nodes": int(flips.sum().item()),
        "support_entries_flipped": int((support_a != support_b).sum().item()),
        "scalar_max_abs_diff": scalar,
    }


def probe_two_epochs(seed: int, device: torch.device, epochs: int = 2) -> dict[str, object]:
    """Epoch body copied verbatim from the formal ``train_arm`` loop."""
    D, _sha = dictionary()
    train_data = load_split("train")
    valid_data = load_split("valid")
    config = pur.reference_config()
    model = pur.build_model(config, D, seed=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    lam = float(config.lambda_rec)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        for batch in loader:
            batch = batch.to(device)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum().item())
            n_mol += int(batch.y.numel())
            rec_sum += float(((aux["phi"] - model.reconstruct(aux["coord"])).pow(2).sum(1) / (aux["phi"].pow(2).sum(1) + 1e-12)).sum().item())
            n_nodes += int(aux["phi"].shape[0])
        valid = evaluate_model(model, eval_loader, device)
        curve.append({"epoch": epoch, "train_mae": task_sum / max(n_mol, 1), "valid_mae": float(valid["mae"])})
    drift = float(max((model.state_dict()[k].float() - v.float()).abs().max().item() for k, v in initial.items()))
    return {"curve": curve, "weight_max_abs_from_init": drift}


def main() -> int:
    started = time.perf_counter()
    device = resolve_device("cuda")
    print(f"device={device}", flush=True)
    print(f"[B/cuda] {probe_forward(device)}", flush=True)
    print(f"[C/cpu ] {probe_forward(torch.device('cpu'), n_molecules=64)}", flush=True)
    first = probe_two_epochs(0, device)
    second = probe_two_epochs(0, device)
    for tag, run in (("A1", first), ("A2", second)):
        print(f"[{tag}] {run['curve']} weight_drift_from_init={run['weight_max_abs_from_init']:.6f}", flush=True)
    for epoch in range(2):
        left, right = first["curve"][epoch], second["curve"][epoch]
        print(
            f"[A] epoch={epoch + 1} train_mae {left['train_mae']:.8f} vs {right['train_mae']:.8f} "
            f"(diff {right['train_mae'] - left['train_mae']:+.8f}) "
            f"valid_mae {left['valid_mae']:.8f} vs {right['valid_mae']:.8f} "
            f"(diff {right['valid_mae'] - left['valid_mae']:+.8f})",
            flush=True,
        )
    print(f"wall_clock_s={time.perf_counter() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
