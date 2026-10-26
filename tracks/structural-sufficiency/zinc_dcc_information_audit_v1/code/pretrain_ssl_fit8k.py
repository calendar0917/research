#!/usr/bin/env python
"""Fit8k-only masked structural SSL for the audit's DCC-L reference.

Reuses the frozen BJOC masked-pretraining mechanism verbatim:
15% atom + 15% physical-bond masks, full feature rebuild from the masked raw
graph, CE + CE, Adam 1e-3 / wd 1e-5 / clip 5, batch 128, 7,560 updates.
Restricted to the label-free Fit8k rows of this round's split.  No labels,
no official valid, no official test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
_SC = _REPO / "tracks" / "structure-contribution" / "code"
_V1 = _REPO / "tracks" / "structural-sufficiency" / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_SC), str(_V1), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import bjoc_masked_pretraining_v1 as MP  # noqa: E402
import structural_relation_composer_v1 as SRC  # noqa: E402
import structural_relation_objects_v1 as SRO  # noqa: E402

DEFAULT_OUT = _TRACK / "analysis" / "weights"
UPDATES = 7560
BATCH = 128
LR = 1e-3
WD = 1e-5
CLIP = 5.0


def _rows_sha(rows) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.out) if args.out else DEFAULT_OUT / f"pretrain_encoder_fit8k_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    split = json.loads((_TRACK / "configs" / "split_fit8k_probe_audit.json").read_text())
    fit_rows = np.asarray(split["rows"]["fit"], dtype=np.int64)
    assert len(fit_rows) == 8000
    objs, obj_source = D.load_p3_objects("train")
    ds = D.pyg_zinc("train")
    # Only structure fields are touched.  y is never read.
    raws = []
    for i in fit_rows:
        d = ds[int(i)]
        raws.append((d.x.reshape(-1).long().numpy(),
                     d.edge_index.numpy(),
                     d.edge_attr.reshape(-1).long().numpy()))

    cfg = SRC.BJOCConfig()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    backbone = SRC.build_bjoc(
        cfg,
        seed=int(args.seed),
        organization_mode=SRC.NULL,
        global_mean=np.zeros(SRO.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(SRO.GLOBAL_DIM, dtype=np.float32),
        device=device,
    )
    pretext = MP.PretextModel(backbone).to(device)
    opt = torch.optim.Adam(pretext.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(int(args.seed) + MP.PRETRAIN_SHUFFLE_OFFSET)

    steps = 0
    visit = 0
    curve = []
    started = time.perf_counter()
    while steps < int(args.updates):
        order = torch.randperm(len(fit_rows), generator=gen).numpy()
        t_ep = time.perf_counter()
        sums = {"loss": 0.0, "ce_atom": 0.0, "ce_bond": 0.0, "acc_atom": 0.0, "acc_bond": 0.0}
        n_steps = n_a = n_b = 0
        for start in range(0, len(order), int(args.batch_size)):
            if steps >= int(args.updates):
                break
            idxs = order[start:start + int(args.batch_size)]
            pre = []
            for local in idxs:
                row = int(fit_rows[int(local)])
                x, ei, ea = raws[int(local)]
                pre.append(MP.build_masked_sample(objs[row], x, ei, ea,
                                                  seed=int(args.seed), visit_index=visit))
            batch, ta, tb, _, _ = MP.batch_pretext_samples(pre)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ta = ta.to(device)
            tb = tb.to(device)
            loss, parts = pretext.loss(batch, ta, tb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretext.parameters(), CLIP)
            opt.step()
            steps += 1
            n_steps += 1
            sums["loss"] += float(loss.detach())
            if "ce_atom" in parts:
                sums["ce_atom"] += float(parts["ce_atom"].detach())
                with torch.no_grad():
                    logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                    sums["acc_atom"] += float((logits["atom_logits"].argmax(1) == ta).float().mean())
                n_a += 1
            if "ce_bond" in parts:
                sums["ce_bond"] += float(parts["ce_bond"].detach())
                with torch.no_grad():
                    logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                    sums["acc_bond"] += float((logits["bond_logits"].argmax(1) == tb).float().mean())
                n_b += 1
        curve.append({
            "visit": visit, "steps": steps,
            "loss": sums["loss"] / max(n_steps, 1),
            "ce_atom": sums["ce_atom"] / max(n_a, 1),
            "ce_bond": sums["ce_bond"] / max(n_b, 1),
            "acc_atom": sums["acc_atom"] / max(n_a, 1),
            "acc_bond": sums["acc_bond"] / max(n_b, 1),
            "seconds": time.perf_counter() - t_ep,
        })
        if visit % 5 == 0 or steps >= int(args.updates):
            print(f"[ssl-fit8k s{args.seed}] visit={visit} steps={steps}/{args.updates} "
                  f"ce_a={curve[-1]['ce_atom']:.4f} acc_a={curve[-1]['acc_atom']:.4f} "
                  f"ce_b={curve[-1]['ce_bond']:.4f} acc_b={curve[-1]['acc_bond']:.4f}", flush=True)
        visit += 1

    state = MP.encoder_state(backbone)
    torch.save(state, out)
    payload = {
        "protocol": "zinc-dcc-information-audit-v1",
        "arm": "fit8k_masked_ssl",
        "seed": int(args.seed),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "updates": steps,
        "batch_size": int(args.batch_size),
        "visits": visit,
        "n_fit_rows": int(len(fit_rows)),
        "fit_rows_sha256": _rows_sha(fit_rows),
        "molecule_visits": int(steps * int(args.batch_size)),
        "hyper": {"optimizer": "Adam", "lr": LR, "weight_decay": WD,
                  "grad_clip": CLIP, "loss": "CE atom + CE bond (1/1)",
                  "checkpoint_rule": "last step only"},
        "mask": {"atom_ratio": MP.ATOM_MASK_RATIO, "bond_ratio": MP.BOND_MASK_RATIO,
                 "atom_token": MP.ATOM_MASK_TOKEN, "bond_token": MP.BOND_MASK_TOKEN,
                 "mask_seed": "SeedSequence([seed, molecule_id, visit])"},
        "object_source": obj_source,
        "curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "output": str(out),
        "output_sha256": C.sha256_file(out),
    }
    (_TRACK / "analysis" / f"ssl_fit8k_s{args.seed}.json").write_text(json.dumps(_jsonable(payload), indent=2))
    print(json.dumps({"seed": args.seed, "updates": steps, "wall_seconds": payload["wall_seconds"],
                      "output_sha256": payload["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
