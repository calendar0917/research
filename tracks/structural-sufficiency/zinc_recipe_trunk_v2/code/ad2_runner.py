"""Runner for zinc-recipe-trunk-v2.

Two jobs:

1. **Recipe cross** (Phase I): run an existing architecture under the *other*
   recipe through the unchanged v1 trainer.  ``--candidate T --recipe A`` and
   ``--candidate P3N --recipe B``.  Only the named recipe dict changes; the
   model builder, data cache, shuffle seed, best-valid selection and evaluation
   are the v1 ones.

2. **New trunks** (Phase II): ``M``/``S`` with the supervised loss, and
   ``Maux``/``Saux`` which add ``lambda * (mean CE_atom + mean CE_bond)`` on a
   masked view of the same molecules.  The masked branch never sees ``y`` and
   the clean branch predicts ``y``.

Everything is delegated to ``ad_train.train``; this module only supplies the
model registry, the named recipe, the auxiliary-loss closure and the run
directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import ad2_masked as MK  # noqa: E402

EXP2 = _HERE.parent
RUNS2 = EXP2 / "runs"
ANALYSIS2 = EXP2 / "analysis"
AUX_WEIGHT = 0.05  # declared before the first joint run; no grid, no balancing


def run_dir(candidate: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS2 / f"{prefix}{candidate}_s{int(seed)}"


def make_aux(visits: int, weight: float = AUX_WEIGHT) -> tuple[dict, dict]:
    """Return (aux spec for ad_train, per-epoch metric sink)."""
    masked = MK.load(visits)
    train_cache = D.load_cache("train")
    atom_ptr = np.asarray(train_cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(train_cache["bond_ptr"], dtype=np.int64)
    sink: dict[str, list] = {"epoch": [], "ce_atom": [], "ce_bond": [], "loss": []}
    epoch_meter: dict[str, list] = {}
    cur = {"epoch": None}

    def flush() -> None:
        if cur["epoch"] is None or not epoch_meter:
            return
        sink["epoch"].append(int(cur["epoch"]))
        for k in ("ce_atom", "ce_bond", "loss"):
            sink[k].append(float(np.mean(epoch_meter.get(k, [float("nan")]))))
        epoch_meter.clear()

    def loss_fn(model, rows, device, epoch):
        if cur["epoch"] != int(epoch):
            flush()
            cur["epoch"] = int(epoch)
        v = (int(epoch) - 1) % int(visits)
        view = dict(train_cache)
        view["atom_feat"] = masked["atom_feat_v"][v]
        view["bond_feat"] = masked["bond_feat_v"][v]
        view["junc_feat"] = masked["junc_feat_v"][v]
        view["global_feat"] = masked["global_feat_v"][v]
        batch = T1._to_device(T1.collate(view, rows, "Maux"), device)

        rows = np.asarray(rows, dtype=np.int64)
        a_cnt = np.diff(atom_ptr)[rows]
        b_cnt = np.diff(bond_ptr)[rows]
        a_off = np.concatenate([[0], np.cumsum(a_cnt)[:-1]])
        b_off = np.concatenate([[0], np.cumsum(b_cnt)[:-1]])
        a_ptr = np.asarray(masked["mask_atom_ptr"])[v]
        b_ptr = np.asarray(masked["mask_bond_ptr"])[v]
        a_ids_v = np.asarray(masked["mask_atom_ids"][v])
        a_tgt_v = np.asarray(masked["mask_atom_target"][v])
        b_ids_v = np.asarray(masked["mask_bond_ids"][v])
        b_tgt_v = np.asarray(masked["mask_bond_target"][v])
        a_ids, a_tgt, b_ids, b_tgt = [], [], [], []
        for k, row in enumerate(rows):
            s, e = int(a_ptr[row]), int(a_ptr[row + 1])
            a_ids.append(a_ids_v[s:e] + a_off[k])
            a_tgt.append(a_tgt_v[s:e])
            s, e = int(b_ptr[row]), int(b_ptr[row + 1])
            b_ids.append(b_ids_v[s:e] + b_off[k])
            b_tgt.append(b_tgt_v[s:e])
        atom_ids = torch.as_tensor(np.concatenate(a_ids), dtype=torch.long, device=device)
        atom_tgt = torch.as_tensor(np.concatenate(a_tgt), dtype=torch.long, device=device)
        bond_ids = torch.as_tensor(np.concatenate(b_ids), dtype=torch.long, device=device)
        bond_tgt = torch.as_tensor(np.concatenate(b_tgt), dtype=torch.long, device=device)
        n_atoms = int(batch["atom_feat"].shape[0])
        n_bonds = int(batch["bond_feat"].shape[0])
        if int(atom_ids.max()) >= n_atoms or int(bond_ids.max()) >= n_bonds:
            raise RuntimeError(
                f"auxiliary index out of range: atoms {int(atom_ids.max())}/{n_atoms}, "
                f"bonds {int(bond_ids.max())}/{n_bonds}"
            )
        if int(atom_tgt.max()) >= A2.ATOM_CE_CLASSES or int(bond_tgt.max()) >= A2.BOND_CE_CLASSES:
            raise RuntimeError("auxiliary target out of range")

        total, parts = model.aux_loss(batch, atom_ids, atom_tgt, bond_ids, bond_tgt)
        for k, t in parts.items():
            epoch_meter.setdefault(k, []).append(float(t.detach()))
        epoch_meter.setdefault("loss", []).append(float(total.detach()))
        return total, parts

    spec = {
        "loss": loss_fn,
        "weight": float(weight),
        "provenance": {
            "construction": "P3 masked-SSL rebuild (ad2_masked.py)",
            "visits": int(visits),
            "mask_seed": int(masked["mask_seed"]),
            "atom_mask_ratio": 0.15,
            "bond_mask_ratio": 0.15,
            "targets": "raw atom type (28 classes) / raw bond type (4 classes)",
            "schedule": "visit = (epoch - 1) % visits",
            "weight": float(weight),
        },
    }
    spec["_finish"] = flush
    spec["_sink"] = sink
    return spec, sink


def train(candidate: str, seed: int, device, *, recipe: str | None = None,
          aux: bool = False, visits: int = 4, tag: str | None = None,
          epochs: int | None = None, force: bool = False) -> dict:
    out = run_dir(candidate, seed, tag)
    spec = None
    if aux:
        spec, sink = make_aux(visits)
    builder = A2.build_model if candidate in A2.CANDIDATES_V2 else T1.M.build_model
    info = T1.train(
        candidate, seed, device, epochs=epochs, tag=tag, force=force,
        recipe_variant=recipe, model_builder=builder, out_dir=out,
        aux=spec,
    )
    if aux:
        spec["_finish"]()
        C.write_json(
            out / "aux_curve.json",
            {"epochs": spec["_sink"]["epoch"], "ce_atom": spec["_sink"]["ce_atom"],
             "ce_bond": spec["_sink"]["ce_bond"], "loss": spec["_sink"]["loss"],
             "weight": AUX_WEIGHT, "visits": int(visits)},
        )
    return info


def timing(candidates, device, *, epochs: int = 2, recipe: str | None = None,
           aux: bool = False, visits: int = 4) -> dict:
    out = {}
    for cand in candidates:
        is_aux = aux or cand.endswith("aux")
        t0 = time.perf_counter()
        info = train(cand, 0, device, recipe=recipe, aux=is_aux, visits=visits,
                     tag="timing", epochs=epochs, force=True)
        sec = time.perf_counter() - t0
        per_epoch = float(np.mean([e["seconds"] for e in info["curve"]]))
        out[cand] = {
            "timed_seconds": sec,
            "mean_epoch_seconds_including_valid": per_epoch,
            "parameters": info["parameter_count"],
            "optimizer_parameters": info["parameter_audit"]["optimizer_parameters"],
            "all_parameters_require_grad": info["parameter_audit"]["all_parameters_require_grad"],
            "gradient_audit": info["gradient_audit"],
            "auxiliary": None if info.get("auxiliary") is None else {
                "weight": info["auxiliary"]["weight"],
                "epoch_mean": info["auxiliary"]["epoch_mean"],
            },
        }
        C.log(f"[timing] {cand}: {per_epoch:.2f}s/epoch, {info['parameter_count']} params")
    ANALYSIS2.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS2 / "timing.json", out)
    C.log(out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing"])
    p.add_argument("--candidate", default="M")
    p.add_argument("--candidates", default="M,S")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--recipe", default=None, choices=[None, "A", "B"])
    p.add_argument("--aux", action="store_true")
    p.add_argument("--visits", type=int, default=4)
    p.add_argument("--tag", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "timing":
        timing([x for x in args.candidates.split(",") if x], device,
               epochs=args.epochs or 2, recipe=args.recipe, visits=args.visits)
    else:
        train(args.candidate, args.seed, device, recipe=args.recipe, aux=args.aux,
              visits=args.visits, tag=args.tag, epochs=args.epochs, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
