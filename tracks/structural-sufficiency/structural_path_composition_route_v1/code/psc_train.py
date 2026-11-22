"""Formal training entry point for the PSC route.

Families
--------
PSC   path encoder + SUM/POOL readout, target MOLECULE / PAIR / DIAG
ACRA  the frozen ACRA-v2 relation model, parent objective (+ optional PAIR)
GINE  the frozen raw-feature GINE reference recipe (external performance anchor)

Every run writes ``manifest.json``, ``config.json``, ``curves.csv`` and its
checkpoints under ``results/<run_id>/``.  Whole-molecule evaluation RMSE on the
current held-out side selects the best checkpoint for D0/D1; ``FULL`` trains a
fixed number of epochs and never evaluates official Valid inside training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import psc_common as C
import psc_data as D
import psc_model as M
import psc_paths as PP

RESULTS_DIR = C.ROUND_DIR / "results"


def lr_at(step: int, warmup_steps: int, total_steps: int, peak: float, minimum: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _arm_cfg(cfg: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm not in cfg["arms"]:
        raise KeyError(f"arm {arm} not in config")
    return dict(cfg["arms"][arm])


def _rows_hash(rows: np.ndarray) -> str:
    return C.array_sha256(np.asarray(rows, np.int64))


# ---------------------------------------------------------------------------
# PSC
# ---------------------------------------------------------------------------


def train_psc(args: argparse.Namespace, cfg: Mapping[str, Any], spec: Mapping[str, Any],
              bundle: D.DataBundle, split: Mapping[str, Any], log: Callable[[str], None],
              out: Path) -> dict[str, Any]:
    import torch

    fit_rows = np.asarray(split["fit_rows"], np.int64)
    eval_rows = np.asarray(split["eval_rows"], np.int64)
    n_mol = int(bundle.labels.size)
    y_std, scaler = D.labels_standardized(bundle, fit_rows)
    atom_dims, bond_dims = _feature_dims()
    L = int(cfg["paths"]["max_len"])
    cache_rows = np.union1d(bundle.official_train, bundle.official_valid)
    cache_meta = PP.ensure_cache(bundle.raw, cache_rows, L, log=log)
    cache = PP.PathCache(cache_meta, device=args.device)
    c_l = PP.c_l_from_cache(cache, fit_rows)
    log(f"  cache: {cache.n_paths} keys, L={L}, compat={cache_meta['compat']}")
    log(f"  c_l={np.round(c_l, 6).tolist()}")

    atom_ok, bond_ok = D.category_support(bundle.raw, fit_rows, atom_dims, bond_dims)
    model, init_hashes = M.build_model(atom_dims, bond_dims, L, str(spec["readout"]),
                                       int(args.seed), cfg)
    zeroed = zero_unseen(model, atom_ok, bond_ok)
    model = model.to(args.device)
    ledger = M.parameter_ledger(model)
    log(f"  params={ledger}")
    log(f"  shared_init={json.dumps(init_hashes, sort_keys=True)}")
    log(f"  zeroed_unseen={zeroed}")

    target = str(spec["target"])
    targets = None
    sampler = None
    if target in ("PAIR", "DIAG"):
        pair_dir = ensure_pair_directory(bundle, log=log)
        targets = D.build_pair_targets(pair_dir, fit_rows, n_mol)
        sampler = D.PairSampler(targets, seed=int(args.seed) * 7919 + 13)
        log(f"  pair targets: {json.dumps(targets.checks, sort_keys=True)}")

    recipe = dict(cfg["recipe"])
    epochs = int(args.epochs or recipe["epochs"])
    batch_mol = int(recipe["batch_molecules"])
    n_pairs = int(recipe["aux_pairs"])
    n_fit = int(fit_rows.size)
    steps_per_epoch = int(math.ceil(n_fit / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    rng = np.random.default_rng(int(args.seed) + 101)
    eval_during_train = split["name"] in ("D0", "D1")

    curves: list[dict[str, Any]] = []
    best: dict[str, Any] = {"eval_rmse": float("inf"), "epoch": -1}
    step = 0
    peak_vram = 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_fit)
        losses: list[dict[str, float]] = []
        for start in range(0, n_fit, batch_mol):
            main_rows = fit_rows[order[start:start + batch_mol]]
            if sampler is not None:
                aux_a, aux_b = sampler.sample(n_pairs)
                aux_rows = np.concatenate([aux_a, aux_b])
                rows = np.union1d(main_rows, aux_rows)
            else:
                rows = main_rows
            batch = cache.gather(rows, c_l, bag=bool(spec["bag"]))
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out_batch = model.forward_batch(batch)
            e = out_batch["yhat_std"] - torch.as_tensor(
                y_std[rows], device=args.device, dtype=torch.float32)
            pos = {int(r): i for i, r in enumerate(rows.tolist())}
            main_idx = torch.as_tensor([pos[int(r)] for r in main_rows.tolist()],
                                       device=args.device)
            l_mol = torch.mean(e[main_idx] ** 2)
            terms = {"loss": l_mol, "mol": l_mol}
            if sampler is not None:
                ia = torch.as_tensor([pos[int(r)] for r in aux_a.tolist()], device=args.device)
                ib = torch.as_tensor([pos[int(r)] for r in aux_b.tolist()], device=args.device)
                ea, eb = e[ia], e[ib]
                if target == "PAIR":
                    extra = torch.mean((eb - ea) ** 2)
                else:
                    extra = torch.mean(ea ** 2 + eb ** 2)
                terms["loss"] = l_mol + float(recipe["lambda_aux"]) * extra
                terms["aux"] = extra
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(recipe["clip_norm"]))
            opt.step()
            step += 1
            losses.append({k: float(v.detach()) for k, v in terms.items()} | {
                "grad_norm": float(grad_norm),
                "n_encoded": float(batch["atom"].shape[0]),
                "n_entries": float(batch["n_entries"]),
            })
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            if step == 1:
                log(f"  step1 loss={losses[-1]['loss']:.4f} mol={losses[-1]['mol']:.4f} "
                    f"grad={losses[-1]['grad_norm']:.3f} entries={int(batch['n_entries'])} "
                    f"encoded={int(batch['atom'].shape[0])} mols={int(batch['n_molecules'])}")
        row: dict[str, Any] = {
            "run_id": args.run_id, "arm": args.arm, "family": "PSC", "epoch": epoch,
            "step": step, "lr": lr_at(step - 1, warmup_steps, total_steps,
                                      float(recipe["peak_lr"]), float(recipe["min_lr"])),
            "epoch_seconds": time.time() - t_epoch,
        }
        for key in ("loss", "mol", "aux", "grad_norm"):
            vals = [x[key] for x in losses if key in x]
            if vals:
                row[f"train_{key}"] = float(np.mean(vals))
        if eval_during_train:
            p_std, _ = predict_psc(model, cache, c_l, fit_rows, args.device,
                                   bool(spec["bag"]), int(recipe["eval_batch"]))
            pred = to_raw(p_std, scaler)
            row["fit_rmse"] = C.rmse(pred, bundle.labels[fit_rows])
            row["fit_mae"] = C.mae(pred, bundle.labels[fit_rows])
            p_std, _ = predict_psc(model, cache, c_l, eval_rows, args.device,
                                   bool(spec["bag"]), int(recipe["eval_batch"]))
            pred = to_raw(p_std, scaler)
            row["eval_rmse"] = C.rmse(pred, bundle.labels[eval_rows])
            row["eval_mae"] = C.mae(pred, bundle.labels[eval_rows])
            row["generalization_gap_rmse"] = row["eval_rmse"] - row["fit_rmse"]
        curves.append(row)
        C.write_csv(out / "curves.csv", curves)
        if eval_during_train:
            better = row["eval_rmse"] < float(best["eval_rmse"]) - 1e-12
            if better:
                best = {"eval_rmse": row["eval_rmse"], "eval_mae": row["eval_mae"],
                        "fit_rmse": row["fit_rmse"], "epoch": epoch}
                torch.save({"state": model.state_dict(), "epoch": epoch, "row": row,
                            "scaler": scaler, "c_l": c_l, "ledger": ledger,
                            "shared_init": init_hashes, "arm": args.arm,
                            "spec": dict(spec), "zeroed": zeroed,
                            "cache_sha256": cache_meta["cache_sha256"],
                            "split_sha256": split["sha256"]}, out / "state_best.pt")
            log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} fit={row['fit_rmse']:.4f} "
                f"eval={row['eval_rmse']:.4f} loss={row['train_loss']:.4f} {row['epoch_seconds']:.1f}s")
        elif epoch % 20 == 0 or epoch == epochs:
            log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} loss={row['train_loss']:.4f} "
                f"{row['epoch_seconds']:.1f}s")

    torch.save({"state": model.state_dict(), "epoch": epochs,
                "row": curves[-1], "scaler": scaler, "c_l": c_l, "ledger": ledger,
                "shared_init": init_hashes, "arm": args.arm, "spec": dict(spec),
                "zeroed": zeroed, "cache_sha256": cache_meta["cache_sha256"],
                "split_sha256": split["sha256"]}, out / "state_final.pt")
    # predictions (no labels are read here; labels come from the frozen Train file)
    for tag in (["best", "final"] if (split["name"] in ("D0", "D1")
                                      and (out / "state_best.pt").is_file()) else ["final"]):
        if tag == "best":
            state = torch.load(out / "state_best.pt", map_location=args.device,
                              weights_only=False)
            model.load_state_dict(state["state"])
        for name, rows in (("fit", fit_rows), ("eval", eval_rows)):
            pred_std, _ = predict_psc(model, cache, c_l, rows, args.device, bool(spec["bag"]),
                                      int(recipe["eval_batch"]))
            np.savez(out / f"predictions_{tag}_{name}.npz", row_id=bundle.row_id[rows],
                     pred=to_raw(pred_std, scaler), pred_std=pred_std)
    wall = time.time() - t_start
    extra_manifest = {
        "target": target,
        "readout": spec["readout"],
        "input": "BAG" if spec["bag"] else "ORDER",
        "best": best if eval_during_train else None,
        "final": curves[-1],
        "c_l": c_l.tolist(),
        "path_cache": {"path": str(C.PATH_CACHE.relative_to(C.REPO_ROOT)),
                       "sha256": cache_meta["cache_sha256"],
                       "schema": cache_meta["schema"], "max_len": L,
                       "n_paths": int(cache.n_paths),
                       "n_entries": int(cache.ent_path.size)},
        "zeroed_categories": zeroed,
        "pair_targets": None if targets is None else targets.checks,
        "n_encoded_paths_step1": int(curves[0].get("train_n_encoded", 0)),
    }
    manifest = _manifest(args, cfg, spec, bundle, split, scaler, ledger, init_hashes,
                         wall, peak_vram, extra_manifest)
    return manifest


def rows_of(split: Mapping[str, Any], which: str) -> np.ndarray:
    return np.asarray(split["fit_rows"] if which == "fit" else split["eval_rows"], np.int64)


def zero_unseen(model: M.PSCModel, atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray]
                ) -> dict[str, int]:
    import torch

    zeroed: dict[str, int] = {}

    def zero(module_list, support, tag: str) -> None:
        for i, emb in enumerate(module_list):
            mask = torch.as_tensor(np.asarray(support[i], bool), device=emb.weight.device)
            if mask.numel() != emb.weight.shape[0]:
                raise ValueError(f"{tag} support width mismatch")
            zeroed[f"{tag}_field_{i}"] = int((~mask).sum().item())
            with torch.no_grad():
                emb.weight[~mask] = 0.0

    zero(model.encoder.atom_emb, atom_ok, "atom")
    zero(model.encoder.bond_emb, bond_ok, "bond")
    return zeroed


def to_raw(pred_std: np.ndarray, scaler: Mapping[str, float]) -> np.ndarray:
    return np.asarray(pred_std, np.float64) * float(scaler["std"]) + float(scaler["mean"])


def predict_psc(model: M.PSCModel, cache: PP.PathCache, c_l: np.ndarray,
                rows: Sequence[int], device: Any, bag: bool, batch: int = 256
                ) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    model.eval()
    rows = np.asarray(rows, np.int64)
    preds = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch)):
            part = rows[start:start + int(batch)]
            b = cache.gather(part, c_l, bag=bag)
            out = model.forward_batch(b)
            preds[start:start + part.size] = out["yhat_std"].detach().double().cpu().numpy()
    return preds, {"n": int(rows.size)}


def _feature_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != C.ATOM_COLUMNS or len(bond) != C.BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature dimensions")
    return atom, bond


# ---------------------------------------------------------------------------
# pair directory caching
# ---------------------------------------------------------------------------

PAIR_CACHE = C.ROUND_DIR / "data" / "pair_directory_lipo.npz"


def ensure_pair_directory(bundle: D.DataBundle, log: Callable[[str], None] = print
                          ) -> D.PairDirectory:
    if PAIR_CACHE.is_file():
        with np.load(PAIR_CACHE, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            if (meta.get("trc_sha256") == C.sha256_file(C.TRC_CACHE)
                    and meta.get("acra_sha256") == C.sha256_file(C.ACRA_CACHE)):
                dir_ = D.PairDirectory.__new__(D.PairDirectory)
                dir_.table = None
                dir_.trc = None
                dir_.acra = None
                dir_.n_fragments = int(meta["n_fragments"])
                dir_.mol_a = np.asarray(z["mol_a"], np.int64)
                dir_.mol_b = np.asarray(z["mol_b"], np.int64)
                dir_.word_a = np.asarray(z["word_a"], np.int64)
                dir_.word_b = np.asarray(z["word_b"], np.int64)
                dir_.bond_id = np.asarray(z["bond_id"], np.int64)
                dir_.T_id = np.asarray(z["T_id"], np.int64)
                dir_.T_key = np.asarray(z["T_key"], np.int64)
                dir_.checks = dict(meta["checks"])
                log(f"  pair directory reused: {dir_.mol_a.size} cut rows")
                return dir_
    dir_ = D.PairDirectory(bundle)
    meta = {"trc_sha256": C.sha256_file(C.TRC_CACHE),
            "acra_sha256": C.sha256_file(C.ACRA_CACHE),
            "n_fragments": int(dir_.n_fragments), "checks": dir_.checks,
            "built_utc": C.now_utc()}
    np.savez(PAIR_CACHE, mol_a=dir_.mol_a, mol_b=dir_.mol_b, word_a=dir_.word_a,
             word_b=dir_.word_b, bond_id=dir_.bond_id, T_id=dir_.T_id,
             T_key=dir_.T_key, meta_json=np.asarray(json.dumps(C.jsonable(meta))))
    log(f"  pair directory built and cached: {dir_.mol_a.size} cut rows")
    return dir_


# ---------------------------------------------------------------------------
# ACRA
# ---------------------------------------------------------------------------

ACRA2_CODE = C.ACRA2_DIR / "code"


def _acra_modules():
    if str(ACRA2_CODE) not in sys.path:
        sys.path.insert(0, str(ACRA2_CODE))
    import acra_common as AC
    import acra_model as AM
    import acra_data as AD
    import acra_cache as AK

    return AC, AM, AD, AK


ACRA_PACK = C.ROUND_DIR / "data" / "acra_pack_lipo_psc.npz"


def ensure_acra_pack(cfg_acc: Mapping[str, Any], bundle: D.DataBundle,
                     log: Callable[[str], None] = print) -> dict[str, Any]:
    AC, AM, AD, AK = _acra_modules()
    code_hash = C.code_sha([ACRA2_CODE / "acra_cache.py", ACRA2_CODE / "acra_data.py"])
    if ACRA_PACK.is_file():
        with np.load(ACRA_PACK, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            pack = {k: np.asarray(z[k]) for k in z.files if k != "meta_json"}
        if (meta.get("raw_sha256") == str(bundle.raw["cache_sha256"])
                and meta.get("acra_sha256") == C.sha256_file(C.ACRA_CACHE)
                and meta.get("code_sha256") == code_hash):
            pack["cache_sha256"] = C.sha256_file(ACRA_PACK)
            pack["cache_path"] = str(ACRA_PACK)
            pack["compat"] = "reused"
            log(f"  ACRA pack reused: {pack['ent_pat'].size} frame entries")
            return pack
    # label-free raw dict (the whitelist loader never touches labels)
    raw = {k: bundle.raw[k] for k in ("node_ptr", "bond_ptr", "atom_feat", "bond_feat",
                                      "bond_u", "bond_v", "smiles", "row_ids")}
    raw["cache_sha256"] = str(bundle.raw["cache_sha256"])
    t0 = time.time()
    pack = AK.build_pack(cfg_acc, raw, log=log)
    meta = {"raw_sha256": str(bundle.raw["cache_sha256"]),
            "acra_sha256": C.sha256_file(C.ACRA_CACHE),
            "code_sha256": code_hash, "built_utc": C.now_utc(),
            "build_seconds": time.time() - t0}
    save = {k: np.asarray(v) for k, v in pack.items() if not isinstance(v, str)}
    np.savez_compressed(ACRA_PACK, meta_json=np.asarray(json.dumps(C.jsonable(meta))), **save)
    pack["cache_sha256"] = C.sha256_file(ACRA_PACK)
    pack["cache_path"] = str(ACRA_PACK)
    pack["compat"] = "built"
    log(f"  ACRA pack built: {pack['ent_pat'].size} frame entries")
    return pack


def acra_masked_loss(model, out, batch, y_std, main_mask, aux=None, lam=0.25):
    """Parent ACRA objective restricted to the uniform main batch (+ optional PAIR).

    The parent regularizers keep their original normalisation but are evaluated
    only on the molecules/events of the main minibatch.
    """
    import torch

    n_mol = int(batch["n_molecules"])
    e = out["yhat_std"] - y_std
    task = torch.mean(e[main_mask] ** 2)
    site_sq = torch.cat([out["atoms"] ** 2, out["r"] ** 2])
    owner = torch.cat([batch["atom_mol"], batch["ev_mol"]])
    owner_main = main_mask[owner]
    per_mol = torch.zeros(n_mol, device=task.device).index_add_(
        0, owner[owner_main], site_sq[owner_main])
    n_sites = (batch["n_atoms_per_mol"] + 2 * batch["n_bonds"]).float()
    site = torch.mean((per_mol / n_sites.clamp(min=1.0))[main_mask])
    ev_main = main_mask[batch["ev_mol"]]
    rel = torch.mean(out["e_rel"][ev_main] ** 2)
    total = (task
             + float(model.cfg["composition"]["site_loss_weight"]) * site
             + float(model.cfg["composition"]["relation_loss_weight"]) * rel)
    terms = {"loss": total, "task": task, "site": site, "rel": rel}
    if aux is not None:
        ia, ib = aux
        pair_term = torch.mean((e[ib] - e[ia]) ** 2)
        terms["pair"] = pair_term
        terms["loss"] = total + float(lam) * pair_term
    return terms


def train_acra(args: argparse.Namespace, cfg: Mapping[str, Any], spec: Mapping[str, Any],
               bundle: D.DataBundle, split: Mapping[str, Any], log: Callable[[str], None],
               out: Path) -> dict[str, Any]:
    import torch

    AC, AM, AD, AK = _acra_modules()
    acc_cfg = json.loads((C.ACRA2_DIR / "configs" / "acra_v2.json").read_text(encoding="utf-8"))
    acc_cfg["recipe"]["seed"] = int(args.seed)
    pack = ensure_acra_pack(acc_cfg, bundle, log=log)
    source = AD.BatchSource(pack, torch.device(args.device))
    fit_rows = np.asarray(split["fit_rows"], np.int64)
    eval_rows = np.asarray(split["eval_rows"], np.int64)
    n_mol = int(bundle.labels.size)
    y_std, scaler = D.labels_standardized(bundle, fit_rows)
    atom_dims, bond_dims = _feature_dims()
    atom_ok, bond_ok = D.category_support(bundle.raw, fit_rows, atom_dims, bond_dims)
    model = AM.ACRAModel(acc_cfg, atom_dims, bond_dims, "ACRA_REL_REF")
    model.load_pack(pack)
    zeroed = AM.apply_support(model, atom_ok, bond_ok)
    model = model.to(args.device)
    ledger = model.parameter_ledger()
    hashes = C.sha256_json({k: round(float(v.abs().sum()), 6)
                            for k, v in model.state_dict().items() if v.dtype.is_floating_point})
    log(f"  params={ledger} zeroed={zeroed} init_hash={hashes[:16]}")

    target = str(spec["target"])
    targets = None
    sampler = None
    if target in ("PAIR", "DIAG"):
        pair_dir = ensure_pair_directory(bundle, log=log)
        targets = D.build_pair_targets(pair_dir, fit_rows, n_mol)
        sampler = D.PairSampler(targets, seed=int(args.seed) * 7919 + 13)
        log(f"  pair targets: {json.dumps(targets.checks, sort_keys=True)}")

    recipe = dict(acc_cfg["recipe"])
    epochs = int(args.epochs or cfg["recipe"]["epochs"])
    recipe["epochs"] = epochs
    batch_mol = int(cfg["recipe"]["batch_molecules"])
    n_aux = int(cfg["recipe"]["aux_pairs"])
    n_fit = int(fit_rows.size)
    steps_per_epoch = int(math.ceil(n_fit / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    rng = np.random.default_rng(int(args.seed) + 101)
    eval_during_train = split["name"] in ("D0", "D1")
    curves: list[dict[str, Any]] = []
    best = {"eval_rmse": float("inf"), "epoch": -1}
    step = 0
    peak_vram = 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_fit)
        losses: list[dict[str, float]] = []
        for start in range(0, n_fit, batch_mol):
            main_rows = fit_rows[order[start:start + batch_mol]]
            if sampler is not None:
                aux_a, aux_b = sampler.sample(n_aux)
                rows = np.union1d(main_rows, np.concatenate([aux_a, aux_b]))
            else:
                aux_a = aux_b = None
                rows = main_rows
            batch = source.make(rows)
            y = torch.as_tensor(y_std[rows], device=args.device, dtype=torch.float32)
            main_set = set(int(r) for r in main_rows.tolist())
            main_mask = torch.as_tensor([int(r) in main_set for r in rows.tolist()],
                                        device=args.device)
            aux = None
            if sampler is not None:
                pos = {int(r): i for i, r in enumerate(rows.tolist())}
                aux = (torch.as_tensor([pos[int(r)] for r in aux_a.tolist()], device=args.device),
                       torch.as_tensor([pos[int(r)] for r in aux_b.tolist()], device=args.device))
            lr = lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                       float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            out_batch = model.forward_batch(batch)
            terms = acra_masked_loss(model, out_batch, batch, y, main_mask, aux=aux,
                                     lam=float(cfg["recipe"]["lambda_aux"]))
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       float(recipe["clip_norm"]))
            opt.step()
            step += 1
            losses.append({k: float(v.detach()) for k, v in terms.items()} | {
                "grad_norm": float(grad_norm)})
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            if step == 1:
                log(f"  step1 loss={losses[-1]['loss']:.4f} task={losses[-1]['task']:.4f} "
                    f"site={losses[-1]['site']:.6f} rel={losses[-1]['rel']:.6f} "
                    f"grad={losses[-1]['grad_norm']:.3f} mols={int(batch['n_molecules'])} "
                    f"events={int(batch['n_events'])}")
        row: dict[str, Any] = {
            "run_id": args.run_id, "arm": args.arm, "family": "ACRA", "epoch": epoch,
            "step": step, "lr": lr_at(step - 1, warmup_steps, total_steps,
                                      float(recipe["peak_lr"]), float(recipe["min_lr"])),
            "epoch_seconds": time.time() - t_epoch}
        for key in ("loss", "task", "site", "rel", "pair", "grad_norm"):
            vals = [x[key] for x in losses if key in x]
            if vals:
                row[f"train_{key}"] = float(np.mean(vals))
        if eval_during_train:
            p_fit = AM.predict_rows(model, source, fit_rows, model.device,
                                    int(recipe["eval_batch"]))["yhat_std"]
            p_eval = AM.predict_rows(model, source, eval_rows, model.device,
                                     int(recipe["eval_batch"]))["yhat_std"]
            tr = p_fit * scaler["std"] + scaler["mean"]
            ev = p_eval * scaler["std"] + scaler["mean"]
            row["fit_rmse"] = C.rmse(tr, bundle.labels[fit_rows])
            row["fit_mae"] = C.mae(tr, bundle.labels[fit_rows])
            row["eval_rmse"] = C.rmse(ev, bundle.labels[eval_rows])
            row["eval_mae"] = C.mae(ev, bundle.labels[eval_rows])
            row["generalization_gap_rmse"] = row["eval_rmse"] - row["fit_rmse"]
        curves.append(row)
        C.write_csv(out / "curves.csv", curves)
        if eval_during_train:
            if row["eval_rmse"] < float(best["eval_rmse"]) - 1e-12:
                best = {"eval_rmse": row["eval_rmse"], "eval_mae": row["eval_mae"],
                        "fit_rmse": row["fit_rmse"], "epoch": epoch}
                torch.save({"state": model.state_dict(), "epoch": epoch, "row": row,
                            "scaler": scaler, "arm": "ACRA_REL_REF", "spec": dict(spec),
                            "zeroed": zeroed, "pack_sha256": pack["cache_sha256"],
                            "split_sha256": split["sha256"]}, out / "state_best.pt")
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} fit={row['fit_rmse']:.4f} "
                    f"eval={row['eval_rmse']:.4f} loss={row['train_loss']:.4f} "
                    f"{row['epoch_seconds']:.1f}s")
        elif epoch % 20 == 0 or epoch == epochs:
            log(f"epoch {epoch:3d}/{epochs} lr={row['lr']:.2e} loss={row['train_loss']:.4f} "
                f"{row['epoch_seconds']:.1f}s")

    torch.save({"state": model.state_dict(), "epoch": epochs, "row": curves[-1],
                "scaler": scaler, "arm": "ACRA_REL_REF", "spec": dict(spec),
                "zeroed": zeroed, "pack_sha256": pack["cache_sha256"],
                "split_sha256": split["sha256"]}, out / "state_final.pt")
    for tag in (["best", "final"] if eval_during_train else ["final"]):
        state = torch.load(out / f"state_{tag}.pt", map_location=args.device,
                           weights_only=False)
        model.load_state_dict(state["state"])
        for name, rows in (("fit", fit_rows), ("eval", eval_rows)):
            res = AM.predict_rows(model, source, rows, model.device,
                                  int(recipe["eval_batch"]))
            pred = res["yhat_std"] * scaler["std"] + scaler["mean"]
            np.savez(out / f"predictions_{tag}_{name}.npz", row_id=bundle.row_id[rows],
                     pred=pred, pred_std=res["yhat_std"])
    wall = time.time() - t_start
    extra = {
        "target": target,
        "readout": spec["readout"],
        "input": "ORDER",
        "best": best if eval_during_train else None,
        "final": curves[-1],
        "zeroed_categories": zeroed,
        "init_hash": hashes,
        "pair_targets": None if targets is None else targets.checks,
        "acra_pack": {"path": str(ACRA_PACK.relative_to(C.REPO_ROOT)),
                      "sha256": pack["cache_sha256"], "compat": pack.get("compat")},
        "parent_recipe": {k: v for k, v in recipe.items()},
        "acra_cfg_sha256": C.sha256_file(C.ACRA2_DIR / "configs" / "acra_v2.json"),
    }
    return _manifest(args, cfg, spec, bundle, split, scaler, ledger, {}, wall, peak_vram, extra)


# ---------------------------------------------------------------------------
# GINE reference
# ---------------------------------------------------------------------------

GINE_RECIPE = {"batch": 128, "eval_batch": 512, "peak_lr": 1e-3, "min_lr": 1e-5,
               "weight_decay": 1e-5, "clip": 5.0, "warmup_epochs": 5}


def _gine_factory():
    if str(C.WP_DIR / "code") not in sys.path:
        sys.path.insert(0, str(C.WP_DIR / "code"))
    import wp_train as WT

    return WT.build_gine_model


def _gine_data(bundle: D.DataBundle, rows: np.ndarray, scaler: Mapping[str, float]):
    import torch
    from torch_geometric.data import Data

    raw = bundle.raw
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom = np.asarray(raw["atom_feat"], np.int64)
    bond = np.asarray(raw["bond_feat"], np.int64)
    bu = np.asarray(raw["bond_u"], np.int64)
    bv = np.asarray(raw["bond_v"], np.int64)
    out = []
    for row in np.asarray(rows, np.int64).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        src = np.concatenate([bu[b0:b1], bv[b0:b1]])
        dst = np.concatenate([bv[b0:b1], bu[b0:b1]])
        eattr = np.concatenate([bond[b0:b1], bond[b0:b1]])
        out.append(Data(x_cat=torch.as_tensor(np.ascontiguousarray(atom[a0:a1])).long(),
                        edge_index=torch.as_tensor(np.stack([src, dst])).long(),
                        edge_cat=torch.as_tensor(np.ascontiguousarray(eattr)).long(),
                        n_atoms=torch.as_tensor([n]).long(), num_nodes=n))
    return out


def train_gine(args: argparse.Namespace, cfg: Mapping[str, Any], spec: Mapping[str, Any],
               bundle: D.DataBundle, split: Mapping[str, Any], log: Callable[[str], None],
               out: Path) -> dict[str, Any]:
    import torch
    from torch_geometric.data import Batch

    C.set_deterministic(int(args.seed))  # seeds are set before the factory runs
    factory = _gine_factory()
    fit_rows = np.asarray(split["fit_rows"], np.int64)
    eval_rows = np.asarray(split["eval_rows"], np.int64)
    y_std, scaler = D.labels_standardized(bundle, fit_rows)
    fit_data = _gine_data(bundle, fit_rows, scaler)
    eval_data = _gine_data(bundle, eval_rows, scaler)
    model = factory({}, int(args.seed), args.device, raw_features=True)
    ledger = {"total": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    epochs = int(args.epochs or cfg["recipe"]["epochs"])
    recipe = dict(GINE_RECIPE)
    n_fit = int(fit_rows.size)
    steps_per_epoch = int(math.ceil(n_fit / recipe["batch"]))
    total_steps = epochs * steps_per_epoch
    warmup = recipe["warmup_epochs"] * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=recipe["peak_lr"],
                            weight_decay=recipe["weight_decay"])
    rng = np.random.default_rng(int(args.seed) + 101)
    eval_during_train = split["name"] in ("D0", "D1")
    curves: list[dict[str, Any]] = []
    best = {"eval_rmse": float("inf"), "epoch": -1}
    peak_vram = 0
    step = 0
    t_start = time.time()

    def predict(data, batch_size=512):
        model.eval()
        out = np.zeros(len(data), np.float64)
        with torch.no_grad():
            for start in range(0, len(data), int(batch_size)):
                chunk = data[start:start + int(batch_size)]
                b = Batch.from_data_list(chunk).to(args.device)
                out[start:start + len(chunk)] = model(b).detach().double().cpu().numpy()
        return out * scaler["std"] + scaler["mean"]

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        order = rng.permutation(n_fit)
        loss_sum = 0.0
        for start in range(0, n_fit, recipe["batch"]):
            idx = order[start:start + recipe["batch"]]
            rows = fit_rows[idx]
            b = Batch.from_data_list([fit_data[int(i)] for i in idx]).to(args.device)
            y = torch.as_tensor(y_std[rows], device=args.device, dtype=torch.float32)
            lr = lr_at(step, warmup, total_steps, recipe["peak_lr"], recipe["min_lr"])
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            pred = model(b)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["clip"])
            opt.step()
            step += 1
            loss_sum += float(loss.detach().cpu()) * int(b.num_graphs)
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
        row: dict[str, Any] = {"run_id": args.run_id, "arm": args.arm, "family": "GINE",
                               "epoch": epoch, "step": step,
                               "lr": lr_at(step - 1, warmup, total_steps,
                                           recipe["peak_lr"], recipe["min_lr"]),
                               "train_loss": loss_sum / max(1, n_fit),
                               "epoch_seconds": time.time() - t_epoch}
        if eval_during_train:
            p_fit = predict(fit_data)
            p_eval = predict(eval_data)
            row["fit_rmse"] = C.rmse(p_fit, bundle.labels[fit_rows])
            row["fit_mae"] = C.mae(p_fit, bundle.labels[fit_rows])
            row["eval_rmse"] = C.rmse(p_eval, bundle.labels[eval_rows])
            row["eval_mae"] = C.mae(p_eval, bundle.labels[eval_rows])
        curves.append(row)
        C.write_csv(out / "curves.csv", curves)
        if eval_during_train:
            if row["eval_rmse"] < float(best["eval_rmse"]) - 1e-12:
                best = {"eval_rmse": row["eval_rmse"], "eval_mae": row["eval_mae"],
                        "fit_rmse": row["fit_rmse"], "epoch": epoch}
                torch.save({"state": model.state_dict(), "epoch": epoch, "row": row,
                            "scaler": scaler, "recipe": recipe,
                            "split_sha256": split["sha256"]}, out / "state_best.pt")
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                log(f"epoch {epoch:3d}/{epochs} fit={row['fit_rmse']:.4f} "
                    f"eval={row['eval_rmse']:.4f} loss={row['train_loss']:.4f} "
                    f"{row['epoch_seconds']:.1f}s")
        elif epoch % 20 == 0 or epoch == epochs:
            log(f"epoch {epoch:3d}/{epochs} loss={row['train_loss']:.4f} "
                f"{row['epoch_seconds']:.1f}s")
    torch.save({"state": model.state_dict(), "epoch": epochs, "row": curves[-1],
                "scaler": scaler, "recipe": recipe, "split_sha256": split["sha256"]},
               out / "state_final.pt")
    for tag in (["best", "final"] if eval_during_train else ["final"]):
        state = torch.load(out / f"state_{tag}.pt", map_location=args.device,
                           weights_only=False)
        model.load_state_dict(state["state"])
        for name, rows, data in (("fit", fit_rows, fit_data), ("eval", eval_rows, eval_data)):
            pred = predict(data, int(recipe["eval_batch"]))
            np.savez(out / f"predictions_{tag}_{name}.npz", row_id=bundle.row_id[rows],
                     pred=pred, pred_std=(pred - scaler["mean"]) / scaler["std"])
    wall = time.time() - t_start
    extra = {"target": "MOLECULE", "readout": "GINE", "input": "graph",
             "best": best if eval_during_train else None, "final": curves[-1],
             "gine_recipe": recipe,
             "factory": str((C.WP_DIR / "code" / "wp_train.py").relative_to(C.REPO_ROOT)),
             "recipe_note": "frozen native raw-feature GINE recipe; not parameter/FLOP matched"}
    return _manifest(args, cfg, spec, bundle, split, scaler, ledger, {}, wall, peak_vram, extra)


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def _manifest(args, cfg, spec, bundle, split, scaler, ledger, init_hashes, wall,
              peak_vram, extra) -> dict[str, Any]:
    manifest = {
        "run_id": args.run_id,
        "arm": args.arm,
        "family": spec["family"],
        "dataset": "ogbg-mollipo",
        "protocol_id": cfg["protocol_id"],
        "seed": int(args.seed),
        "split": {"name": split["name"], "sha256": split["sha256"],
                  "file_sha256": split.get("file_sha256", ""),
                  "source": split["source"], "role": split["role"],
                  "n_fit": int(np.asarray(split["fit_rows"]).size),
                  "n_eval": int(np.asarray(split["eval_rows"]).size),
                  "fit_rows_sha256": _rows_hash(np.asarray(split["fit_rows"], np.int64)),
                  "eval_rows_sha256": _rows_hash(np.asarray(split["eval_rows"], np.int64))},
        "selection": ("best whole-molecule RMSE on the held-out side"
                      if split["name"] in ("D0", "D1") else
                      "fixed epoch; final checkpoint (no Valid-based selection)"),
        "test_labels_read": False,
        "valid_labels_read": False,
        "label_access": {
            "train_labels": bundle.hashes["train_labels"],
            "labels_masked_outside_official_train": True,
            "allowed_label_rows": "official Train rows only",
            "valid_labels": "NOT read during training (final evaluator only)",
        },
        "scaler": scaler,
        "params": ledger,
        "shared_init_hashes": init_hashes,
        "hashes": {k: v for k, v in bundle.hashes.items()},
        "recipe": dict(cfg["recipe"]),
        "wall_seconds": float(wall),
        "gpu_hours": float(wall) / 3600.0,
        "max_vram_bytes": int(peak_vram),
        "environment": C.environment(),
        "provenance": C.code_provenance(),
        "argv": list(getattr(args, "argv", [])),
        "finished_utc": C.now_utc(),
    }
    manifest.update(extra)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="D0", choices=("D0", "D1", "FULL"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else []
    cfg = C.load_config()
    spec = _arm_cfg(cfg, args.arm)
    out = RESULTS_DIR / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train.log")
    C.set_deterministic(int(args.seed))
    log(f"run={args.run_id} arm={args.arm} family={spec['family']} split={args.split} "
        f"seed={args.seed} device={args.device}")
    log(f"  commit={C.git_commit()} host={C.environment()['hostname']} "
        f"torch={C.environment()['torch']}")
    bundle = D.load_bundle()
    f1 = None
    if args.split == "D1":
        if not C.SPLIT_F1.is_file():
            raise FileNotFoundError(f"D1 split missing: {C.SPLIT_F1}")
        f1 = json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
        f1["file_sha256"] = C.sha256_file(C.SPLIT_F1)
    split = C.split_spec(args.split, bundle.raw, f1=f1)
    log(f"  split={split['name']} n_fit={len(split['fit_rows'])} "
        f"n_eval={len(split['eval_rows'])} sha={split['sha256']}")
    log(f"  train_labels sha={bundle.hashes['train_labels']['sha256']} "
        f"n={int(bundle.labels[np.isfinite(bundle.labels)].size)}")
    t0 = time.time()
    if spec["family"] == "PSC":
        manifest = train_psc(args, cfg, spec, bundle, split, log, out)
    elif spec["family"] == "ACRA":
        manifest = train_acra(args, cfg, spec, bundle, split, log, out)
    elif spec["family"] == "GINE":
        manifest = train_gine(args, cfg, spec, bundle, split, log, out)
    else:
        raise ValueError(f"unknown family {spec['family']}")
    manifest["main_seconds"] = time.time() - t0
    C.write_json(out / "manifest.json", manifest)
    C.write_json(out / "config.json", {"protocol_id": cfg["protocol_id"],
                                       "arm": args.arm, "spec": spec, "config": cfg})
    log(f"done wall={manifest['wall_seconds'] / 60:.2f}min "
        f"best={manifest.get('best')}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())