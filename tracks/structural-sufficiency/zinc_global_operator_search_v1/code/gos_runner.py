"""Runner for zinc-global-operator-search-v1 (routes C0 / A / K).

Everything in the supervised loop is the existing ``ad_train.train``: recipe B
(AdamW, lr 1e-3 -> 1e-5, wd 1e-5, 600 epochs, 500-step warm-up, clip 5, raw-y
L1, batch 128, full valid every epoch, strict best-valid selection).  This
module only supplies the model builder, the relation collate and artifact
hashing.  Official test is never constructed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

RUNS = _EXP / "runs"
ANALYSIS = _EXP / "analysis"
RECIPE = "B"


def run_dir(route: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{route}_s{int(seed)}"


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def prepare() -> tuple[dict, dict, dict, float]:
    """Load train/valid caches, relation caches and the train-only scale."""
    for p in (C.CACHE / "ad_train.pt", C.CACHE / "ad_valid.pt"):
        if not p.exists():
            raise FileNotFoundError(f"missing train/valid cache {p}")
    rel_train = GD.load_rel("train")
    rel_valid = GD.load_rel("valid")
    sp = ANALYSIS / "rel_scale.json"
    if sp.exists():
        scale = C.read_json(sp)
    else:
        scale = GD.fit_scale(rel_train)
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        C.write_json(sp, scale)
    mean_nb = float(np.asarray(rel_train["n_bonds_graph"]).astype(np.float64).mean())
    return rel_train, rel_valid, scale, mean_nb


def _apply_cmass_init(model, info: dict, mix: str) -> None:
    """Set the 12 constant-mass scales from the frozen train-only init file.

    The common non-rho initialisation is untouched: rho is created with zeros
    (no RNG consumption) and overwritten here.
    """
    if mix != "cmass":
        return
    path = ANALYSIS / "cmass_init.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run code/gos_cmass.py init first")
    init = C.read_json(path)
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            if getattr(layer, "rho", None) is None:
                raise RuntimeError("cmass model layer has no rho parameter")
            layer.rho.copy_(torch.as_tensor(init["rho_init"][li], dtype=layer.rho.dtype))
    info["cmass_init"] = init
    info["cmass_rho_init"] = init["rho_init"]


def build_arm_model(route: str, seed: int, device, *, mix: str = "legacy",
                    mean_n_bonds: float, dropout: float = 0.0,
                    reader_hidden: int = GM.READER_HIDDEN, norm=None):
    """Single constructor for every model instance of an arm, so the trainer, the
    best-checkpoint re-evaluation and the holdout pass cannot disagree about the
    operator."""
    model, info = GM.build_model(route, D.load_norm() if norm is None else norm, int(seed), device,
                                 mean_n_bonds=float(mean_n_bonds), dropout=float(dropout),
                                 reader_hidden=int(reader_hidden), mix_mode=str(mix))
    _apply_cmass_init(model, info, str(mix))
    return model, info


def make_builder(route: str, mean_n_bonds: float, dropout: float = 0.0,
                 reader_hidden: int = GM.READER_HIDDEN, mix: str = "legacy"):
    def builder(candidate, norm, seed, device):
        return build_arm_model(route, seed, device, mix=mix, mean_n_bonds=mean_n_bonds,
                               dropout=dropout, reader_hidden=reader_hidden, norm=norm)

    return builder


def write_artifacts(out: Path, info: dict, *, mix: str = "legacy") -> dict:
    best = out / "best.pt"
    final = out / "final.pt"
    pred = out / "predictions" / "valid_best.npz"
    art = {
        "run_dir": str(out.relative_to(C.REPO)),
        "candidate": info.get("candidate"),
        "route": info.get("model_info", {}).get("route"),
        "mix_mode": mix,
        "seed": info.get("seed"),
        "recipe_variant": info.get("recipe_variant"),
        "parameter_count": info.get("parameter_count"),
        "parameters_breakdown": info.get("model_info", {}).get("parameters_breakdown"),
        "initial_state_sha256": info.get("model_info", {}).get("initial_state_sha256"),
        "best_valid": info.get("best_valid"),
        "fixed_objects": info.get("fixed_objects"),
        "gpu_hours_run_level": info.get("gpu_hours"),
        "wall_seconds": info.get("wall_seconds"),
        "sha256": {
            "best_pt": _sha(best),
            "final_pt": _sha(final),
            "valid_best_npz": _sha(pred),
            "train_json": _sha(out / "train.json"),
            "curve_csv": _sha(out / "curve.csv"),
        },
    }
    C.write_json(out / "artifacts.json", art)
    return art


HOLDOUT_SEED = 20261025


def holdout_split(n_holdout: int, n_train: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train-internal holdout: a fresh 1,000-molecule sample that
    no development decision in this round ever looked at."""
    perm = np.random.default_rng(HOLDOUT_SEED).permutation(n_train)
    hold = np.sort(perm[:n_holdout]).astype(np.int64)
    keep = np.sort(perm[n_holdout:]).astype(np.int64)
    return keep, hold


def train(route: str, seed: int, device, *, recipe: str = RECIPE,
          epochs: int | None = None, tag: str | None = None, force: bool = False,
          lr: float | None = None, weight_decay: float | None = None,
          dropout: float | None = None,
          holdout: int | None = None,
          reader_hidden: int | None = None,
          mix: str = "legacy") -> dict:
    rel_train, rel_valid, scale, mean_nb = prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    keep = None
    hold = None
    if holdout:
        keep, hold = holdout_split(int(holdout))
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        C.write_json(ANALYSIS / f"holdout_split_{int(holdout)}.json", {
            "seed": HOLDOUT_SEED, "n_holdout": int(holdout),
            "holdout_rows_sha256": C.sha256_array(hold),
            "holdout_first20": hold[:20].tolist(),
            "train_rows": int(len(keep)), "source": "official train split only",
        })
    out = run_dir(route, seed, tag)
    label = f"GOS_{route}" if mix == "legacy" else f"GOS_{route}[{mix}]"
    info = T1.train(
        label,
        seed,
        device,
        epochs=epochs,
        tag=tag,
        force=force,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        recipe_variant=recipe,
        model_builder=make_builder(route, mean_nb, float(dropout or 0.0),
                                   int(reader_hidden or GM.READER_HIDDEN), mix),
        out_dir=out,
        collate_fn=collate,
        train_rows=keep,
    )
    art = write_artifacts(out, info, mix=mix)
    if holdout:
        norm = D.load_norm()
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model, _ = build_arm_model(route, int(seed), device, mix=mix, mean_n_bonds=mean_nb,
                                   dropout=float(dropout or 0.0),
                                   reader_hidden=int(reader_hidden or GM.READER_HIDDEN),
                                   norm=norm)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        train_cache = D.load_cache("train")
        valid_cache = D.load_cache("valid")
        rows_valid = np.arange(int(valid_cache["labels"].shape[0]), dtype=np.int64)
        p_hold, mae_hold = T1.evaluate(model, train_cache, hold, candidate="GOS",
                                       device=device, batch_size=256, collate_fn=collate)
        _p, mae_valid = T1.evaluate(model, valid_cache, rows_valid, candidate="GOS",
                                    device=device, batch_size=256, collate_fn=collate)
        p_keep, mae_keep = T1.evaluate(model, train_cache, keep, candidate="GOS",
                                       device=device, batch_size=256, collate_fn=collate)
        np.savez(out / "predictions" / "holdout.npz", row=hold, pred=p_hold)
        rec = {
            "protocol": "zinc-global-operator-search-v1",
            "kind": "train-internal holdout (1000 molecules; not fresh any more)",
            "route": route,
            "mix_mode": mix,
            "seed": int(seed), "holdout_size": int(len(hold)),
            "holdout_seed": HOLDOUT_SEED,
            "holdout_sha256": C.sha256_array(hold),
            "epoch": int(ck.get("epoch", -1)),
            "valid_official_mae": float(mae_valid),
            "holdout_mae": float(mae_hold),
            "train_kept_mae": float(mae_keep),
            "n_train_rows": int(len(keep)),
            "checkpoint": str((out / "best.pt").relative_to(C.REPO)),
            "checkpoint_sha256": _sha(out / "best.pt"),
        }
        C.write_json(ANALYSIS / f"holdout_{route}{'' if mix == 'legacy' else '_' + mix}_s{int(seed)}.json", rec)
        C.log(f"[holdout-{route}:{mix}] seed={seed} holdout={mae_hold:.6f} valid={mae_valid:.6f} "
              f"kept={mae_keep:.6f} ep={rec['epoch']}")
    bo = info["fixed_objects"]["best"]
    C.log(
        f"[gos-{route}:{mix}] seed={seed} best={info['best_valid']['valid_mae']:.9f}"
        f"@{info['best_valid']['epoch']} final={info['fixed_objects']['final']['valid_mae']:.9f} "
        f"params={info['parameter_count']} wall={info['wall_seconds']:.1f}s "
        f"init={info['model_info']['initial_state_sha256'][:16]} sha={art['sha256']['best_pt']}"
    )
    return info


def timing(route: str, seed: int, device, *, epochs: int = 3, recipe: str = RECIPE) -> dict:
    t0 = time.perf_counter()
    info = train(route, seed, device, recipe=recipe, epochs=epochs, tag="timing", force=True)
    total_seconds = time.perf_counter() - t0
    curve = info["curve"]
    epoch_seconds = [float(e["seconds"]) for e in curve]
    sum_epoch = float(np.sum(epoch_seconds))
    mean_epoch = float(np.mean(epoch_seconds))
    steady = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else mean_epoch
    final_eval_seconds = float(info["wall_seconds"]) - sum_epoch
    startup = float(total_seconds) - float(info["wall_seconds"])
    projected = 600.0 * mean_epoch
    total_proj = 1.15 * projected + startup + final_eval_seconds + 30.0
    out = {
        "protocol_id": "zinc-global-operator-search-v1",
        "route": route,
        "seed": int(seed),
        "recipe": recipe,
        "rows": "full official train 10000, full valid 1000",
        "epochs_measured": int(len(epoch_seconds)),
        "steps_per_epoch": int(info["recipe"]["steps_per_epoch"]),
        "epoch_seconds": epoch_seconds,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady,
        "startup_seconds": startup,
        "final_eval_seconds": final_eval_seconds,
        "projected_600ep_seconds_with_overheads": total_proj,
        "projected_600ep_gpu_hours": total_proj / 3600.0,
        "parameters": int(info["parameter_count"]),
        "parameter_audit": info["parameter_audit"],
        "gradient_audit": info["gradient_audit"],
        "official_test_loaded": False,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / f"timing_{route}.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing", "prep"])
    p.add_argument("--route", default="A", choices=["C0", "A", "K", "Asm", "Ksm", "N0"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--recipe", default=RECIPE, choices=["A", "B", "C"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--reader-hidden", type=int, default=None,
                   help="reader width (default 128; 160 is the widest under the 500k cap)")
    p.add_argument("--holdout", type=int, default=None,
                   help="train on 10000-N rows and score the held-out N (fresh sample)")
    p.add_argument("--mix", default="legacy", choices=list(GM.MIX_MODES),
                   help="mixing arithmetic; only cmass adds 12 parameters (see ERRATUM.md)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command == "prep":
        rel_train, rel_valid, scale, mean_nb = prepare()
        C.log(f"prep ok: train pairs={int(rel_train['ptr'][-1])} valid pairs={int(rel_valid['ptr'][-1])} "
              f"mean_n_bonds={mean_nb:.6f}")
        return 0
    device = torch.device(args.device)
    if args.command == "timing":
        timing(args.route, args.seed, device, epochs=args.epochs or 3, recipe=args.recipe)
    else:
        train(args.route, args.seed, device, recipe=args.recipe, epochs=args.epochs,
              tag=args.tag, force=args.force, lr=args.lr,
              weight_decay=args.weight_decay, dropout=args.dropout,
              holdout=args.holdout, reader_hidden=args.reader_hidden, mix=args.mix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
