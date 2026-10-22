"""Supervised-support intervention on official-train molecules only (§7).

Question: with the same model, initialisation and recipe, does *adding labelled
molecules that contain a target local environment* change error on a held-out
query set that contains the same environment, compared with adding the same
number of structurally matched molecules that contain none of the target keys?

Construction is structure-only (no labels, no model errors):

* candidate radius-2 target keys: molecule support in 4..25 in official train,
  radius-1 parent key present in >= 50 molecules;
* ``H`` = train molecules containing any selected key, grown by a fixed
  structural greedy until ``H`` is in 800..1500 with >= 40 keys;
* ``Q_target`` (~300-500) and ``D_plus`` from ``H``, every target key with >= 1
  query molecule and >= 2 donor molecules;
* ``Q_background`` (to make |Q| = 1000) and ``M`` (1000) from molecules without
  any target key; neither is ever used for fitting;
* ``D_minus``: same number as ``D_plus``, matched on size, ring rank, atom
  composition and radius-1 key histogram; contains zero target keys;
* ``B`` = everything left; ``F_plus = B + D_plus``, ``F_minus = B + D_minus``.

The split is fixed before any internal query error is computed.  Labels of Q are
read only after every planned checkpoint is written and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code"),
           str(_SSUF / "zinc_recipe_trunk_v2" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import zct_keys as K  # noqa: E402
import zct_train as ZT  # noqa: E402

CACHE = _TASK / "cache"
ANALYSIS = _TASK / "analysis"
RUNS = _TASK / "runs"

KEY_SUP_LO, KEY_SUP_HI = 4, 25
PARENT_MIN = 50
H_LO, H_HI = 800, 1500
MIN_KEYS = 40
Q_TARGET_LO, Q_TARGET_HI = 300, 500
D_PLUS_LO, D_PLUS_HI = 400, 1000
Q_TOTAL = 1000
M_SIZE = 1000
MIN_FIT = 6000


def _mol_sets(mol_list: np.ndarray, off: np.ndarray) -> list[np.ndarray]:
    return [mol_list[int(off[i]):int(off[i + 1])] for i in range(len(off) - 1)]


def build_support(seed: int = 20261101, verbose: bool = True) -> dict:
    keys = K.load()
    train = D.load_cache("train")
    n = int(train["n_molecules"])
    off2 = np.asarray(keys["mol_k2_off"], dtype=np.int64)[: n + 1]
    off1 = np.asarray(keys["mol_k1_off"], dtype=np.int64)[: n + 1]
    mol2 = _mol_sets(np.asarray(keys["mol_k2"], dtype=np.int64), off2)
    mol1 = _mol_sets(np.asarray(keys["mol_k1"], dtype=np.int64), off1)
    n_keys2 = int(keys["n_key2"])
    n_keys1 = int(keys["n_key1"])

    sup2 = np.zeros(n_keys2, dtype=np.int64)
    sup1 = np.zeros(n_keys1, dtype=np.int64)
    inv2: list[list[int]] = [[] for _ in range(n_keys2)]
    for m, s2 in enumerate(mol2):
        sup2[s2] += 1
        for k in s2.tolist():
            inv2[k].append(m)
    for s1 in mol1:
        sup1[s1] += 1
    inv2 = [np.asarray(x, dtype=np.int64) for x in inv2]
    if any((len(x) == 0) for x in np.asarray(inv2, dtype=object)[sup2 > 0]):
        raise RuntimeError("radius-2 key with train support has no molecule (inverse index broken)")
    atom_k2 = np.asarray(keys["atom_key2"], dtype=np.int64)
    atom_k1 = np.asarray(keys["atom_key1"], dtype=np.int64)
    ap = np.asarray(train["atom_ptr"], dtype=np.int64)
    parent = np.full(n_keys2, -1, dtype=np.int64)
    for i in range(n):
        s, e = int(ap[i]), int(ap[i + 1])
        parent[atom_k2[s:e]] = atom_k1[s:e]
    if (parent[sup2 > 0] < 0).any():
        raise RuntimeError("some radius-2 key with train support has no radius-1 parent")

    cand = np.flatnonzero((sup2 >= KEY_SUP_LO) & (sup2 <= KEY_SUP_HI)
                          & (sup1[np.maximum(parent, 0)] >= PARENT_MIN))
    rng = np.random.default_rng(int(seed))
    order = cand[rng.permutation(len(cand))]   # fixed-RNG structural order (no y)
    selected: list[int] = []
    h = np.zeros(0, dtype=np.int64)
    for k in order:
        trial = np.unique(np.concatenate([h, inv2[int(k)]]))
        if len(trial) > H_HI - 100:
            continue
        selected.append(int(k))
        h = trial
        if len(h) >= H_LO and len(selected) >= MIN_KEYS:
            break
    if len(h) < H_LO or len(selected) < MIN_KEYS:
        raise RuntimeError(f"target-key greedy failed: |H|={len(h)} keys={len(selected)}")

    # ---- Q_target / D_plus ------------------------------------------------
    q_target: list[int] = []
    for k in selected:
        q_target.append(int(inv2[k].min()))
    q_target = sorted(set(q_target))
    # leave repair slack: Q_target ~45% of H, and never below Q_TARGET_LO-10
    target_q = min(max(int(round(0.45 * len(h))), Q_TARGET_LO + 50), Q_TARGET_HI)
    target_q = max(target_q, Q_TARGET_LO + 50)
    if len(q_target) < target_q:
        used = set(q_target)
        rest = np.asarray([m for m in h.tolist() if m not in used], dtype=np.int64)
        rest = rest[rng.permutation(len(rest))][: target_q - len(q_target)]
        q_target = sorted(set(q_target + rest.tolist()))
    q_t = np.asarray(q_target, dtype=np.int64)
    d_plus = np.setdiff1d(h, q_t)

    def fix_donors(q_t, d_plus):
        q_set = set(q_t.tolist())
        d_set = set(d_plus.tolist())
        for k in selected:
            ms = set(inv2[k].tolist())
            while len(ms & d_set) < 2 and len(q_set) > Q_TARGET_LO - 10:
                movable = sorted(ms & q_set)
                if not movable:
                    raise RuntimeError(f"target key {k} cannot reach 2 donors")
                m = movable[-1]
                q_set.discard(m)
                d_set.add(m)
            if len(ms & q_set) < 1 or len(ms & d_set) < 2:
                raise RuntimeError(f"target key {k}: query/donor support insufficient")
        return np.asarray(sorted(q_set), dtype=np.int64), np.asarray(sorted(d_set), dtype=np.int64)

    q_t, d_plus = fix_donors(q_t, d_plus)

    has_target = np.zeros(n, dtype=bool)
    for k in selected:
        has_target[inv2[k]] = True
    if int(has_target.sum()) != len(h):
        raise RuntimeError("target-key molecule union does not match H")
    free = np.flatnonzero(~has_target)
    if len(free) < (Q_TOTAL - len(q_t)) + M_SIZE + len(d_plus):
        raise RuntimeError("not enough non-target molecules for Q/M/D_minus")
    perm = free[rng.permutation(len(free))]
    n_qb = Q_TOTAL - len(q_t)
    q_background = np.sort(perm[:n_qb])
    monitor = np.sort(perm[n_qb:n_qb + M_SIZE])
    cand_minus = perm[n_qb + M_SIZE:]

    # ---- structure matching for D_minus ----------------------------------
    stats = K.molecule_stats(train)
    atom_types = np.asarray(train["atom_type"], dtype=np.int64)
    n_types = int(atom_types.max()) + 1
    top_atoms = np.argsort(-np.bincount(atom_types, minlength=n_types))[:6]
    k1_atom = np.asarray(keys["atom_key1"], dtype=np.int64)
    k1_counts_flat = np.bincount(k1_atom[: int(ap[-1])], minlength=n_keys1)
    top_k1 = np.argsort(-k1_counts_flat)[:24]

    def features(mols: np.ndarray) -> np.ndarray:
        f = np.zeros((len(mols), 4 + len(top_atoms) + len(top_k1)), dtype=np.float64)
        for i, m in enumerate(mols):
            s, e = int(ap[m]), int(ap[m + 1])
            nat = max(e - s, 1)
            f[i, 0] = math.log1p(stats["n_atoms"][m])
            f[i, 1] = math.log1p(stats["ring_rank"][m])
            f[i, 2] = math.log1p(stats["n_bonds"][m])
            f[i, 3] = stats["ring_rank"][m] / nat
            cnt = np.bincount(atom_types[s:e], minlength=n_types)
            f[i, 4:4 + len(top_atoms)] = cnt[top_atoms] / nat
            c1 = np.bincount(k1_atom[s:e], minlength=n_keys1)
            f[i, 4 + len(top_atoms):] = c1[top_k1] / nat
        return f

    fit_feat = features(np.arange(n))
    mu = fit_feat.mean(0)
    sd = fit_feat.std(0) + 1e-9
    dp = np.asarray(d_plus, dtype=np.int64)
    cm = np.asarray(cand_minus, dtype=np.int64)
    fd = (features(dp) - mu) / sd
    fc = (features(cm) - mu) / sd
    dmat = ((fd[:, None, :] - fc[None, :, :]) ** 2).sum(2)
    taken = np.zeros(len(cm), dtype=bool)
    pick = -np.ones(len(dp), dtype=np.int64)
    for i in range(len(dp)):
        j = int(np.argmin(np.where(taken, np.inf, dmat[i])))
        pick[i] = j
        taken[j] = True
    d_minus = np.sort(cm[pick])
    match_dist = np.sqrt(dmat[np.arange(len(dp)), pick])
    diff = fd.mean(0) - fc[pick].mean(0)
    b_rows = np.setdiff1d(np.arange(n), np.concatenate([q_t, q_background, monitor, dp, d_minus]))
    fit_plus = np.sort(np.concatenate([b_rows, dp]))
    fit_minus = np.sort(np.concatenate([b_rows, d_minus]))

    # ---- assertions -------------------------------------------------------
    def assert_disjoint(*arrays):
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                if np.intersect1d(arrays[i], arrays[j]).size:
                    raise RuntimeError(f"support sets {i} and {j} overlap")

    q_all = np.sort(np.concatenate([q_t, q_background]))
    assert_disjoint(q_all, monitor, dp, d_minus, b_rows)
    if set(dp.tolist()) & set(fit_minus.tolist()):
        raise RuntimeError("D_plus leaked into F_minus")
    if set(d_minus.tolist()) & set(fit_plus.tolist()):
        raise RuntimeError("D_minus leaked into F_plus")
    sup_plus = np.zeros(n_keys2, dtype=np.int64)
    sup_minus = np.zeros(n_keys2, dtype=np.int64)
    for m in fit_plus:
        sup_plus[mol2[m]] += 1
    for m in fit_minus:
        sup_minus[mol2[m]] += 1
    sel = np.asarray(selected, dtype=np.int64)
    if int(sup_minus[sel].sum()) != 0:
        raise RuntimeError("F_minus contains a target key")
    if int((sup_plus[sel] == 0).sum()) != 0:
        raise RuntimeError("F_plus misses a target key")
    if len(fit_plus) != len(fit_minus) or len(fit_plus) < MIN_FIT:
        raise RuntimeError("fit sizes invalid")
    if bool(has_target[q_background].any()) or bool(has_target[monitor].any()):
        raise RuntimeError("Q_background or M contains a target key")

    manifest = {
        "seed": int(seed),
        "target_keys": [int(k) for k in selected],
        "n_target_keys": int(len(selected)),
        "key_support_train": {str(int(k)): int(sup2[k]) for k in selected},
        "key_parent_support_train": {str(int(k)): int(sup1[parent[k]]) for k in selected},
        "key_support_in_fit_plus": {str(int(k)): int(sup_plus[k]) for k in selected},
        "key_support_in_fit_minus": {str(int(k)): int(sup_minus[k]) for k in selected},
        "key_donor_count": {str(int(k)): int(len(set(inv2[int(k)].tolist()) & set(dp.tolist())))
                            for k in selected},
        "key_query_count": {str(int(k)): int(len(set(inv2[int(k)].tolist()) & set(q_t.tolist())))
                            for k in selected},
        "sizes": {
            "H": int(len(h)), "Q_target": int(len(q_t)), "D_plus": int(len(dp)),
            "Q_background": int(len(q_background)), "Q_total": int(len(q_all)),
            "M": int(len(monitor)), "D_minus": int(len(d_minus)), "B": int(len(b_rows)),
            "F_plus": int(len(fit_plus)), "F_minus": int(len(fit_minus)),
        },
        "matching": {
            "covariates": ["log1p_atoms", "log1p_ring_rank", "log1p_bonds", "ring_rank/atoms"]
                          + [f"atom_type_{int(t)}_frac" for t in top_atoms]
                          + [f"k1_{int(t)}_frac" for t in top_k1],
            "mean_distance": float(match_dist.mean()),
            "median_distance": float(np.median(match_dist)),
            "max_distance": float(match_dist.max()),
            "smd_D_plus_vs_D_minus": [float(x) for x in diff],
            "max_abs_smd": float(np.max(np.abs(diff))),
            "note": "matching reduces but never removes structural differences; no y or model error used",
        },
        "checks": {
            "disjoint": True,
            "fit_minus_target_support_zero": True,
            "fit_plus_target_support_positive": True,
            "sizes_equal": bool(len(fit_plus) == len(fit_minus)),
            "no_target_key_in_Q_background_or_M": True,
        },
    }
    path = CACHE / "support_sets.npz"
    np.savez_compressed(
        path,
        target_keys=sel,
        H=np.asarray(h, dtype=np.int64),
        q_target=np.sort(q_t),
        q_background=np.asarray(q_background, dtype=np.int64),
        monitor=np.asarray(monitor, dtype=np.int64),
        d_plus=dp,
        d_minus=d_minus,
        b_rows=np.asarray(b_rows, dtype=np.int64),
        fit_plus=fit_plus,
        fit_minus=fit_minus,
    )
    manifest["npz"] = str(path)
    manifest["npz_sha256"] = C.sha256_file(path)
    C.write_json(CACHE / "support_manifest.json", manifest)
    if verbose:
        C.log(manifest["sizes"])
        C.log({"n_target_keys": manifest["n_target_keys"],
               "max_abs_smd": manifest["matching"]["max_abs_smd"],
               "mean_match_distance": manifest["matching"]["mean_distance"]})
    return manifest


def load_support() -> tuple[dict, dict]:
    d = dict(np.load(CACHE / "support_sets.npz"))
    return d, C.read_json(CACHE / "support_manifest.json")


# ---------------------------------------------------------------------------
# training on F_plus / F_minus
# ---------------------------------------------------------------------------


def fit_batches(base_rows: np.ndarray, donor_rows: np.ndarray, epoch: int, seed: int,
                batch_size: int = 128) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic batches: shared B rows in a fixed order, donors in fixed slots."""
    rng = np.random.default_rng([int(seed), int(epoch), 7])
    b_perm = base_rows[rng.permutation(len(base_rows))]
    d_perm = donor_rows[rng.permutation(len(donor_rows))]
    n_b, n_d = len(b_perm), len(d_perm)
    steps = math.ceil((n_b + n_d) / batch_size)
    d_sizes = np.full(steps, n_d // steps, dtype=np.int64)
    d_sizes[: n_d % steps] += 1
    b_sizes = np.full(steps, n_b // steps, dtype=np.int64)
    b_sizes[: n_b % steps] += 1
    out, bo, do = [], 0, 0
    for k in range(steps):
        nb, nd = int(b_sizes[k]), int(d_sizes[k])
        out.append((b_perm[bo:bo + nb], d_perm[do:do + nd]))
        bo += nb
        do += nd
    if bo != n_b or do != n_d:
        raise RuntimeError("batch partition lost rows")
    return out


def train_fit(fit: str, seed: int, device, out: Path, epochs: int = 600) -> dict:
    sup, manifest = load_support()
    rows_fit = np.asarray(sup["fit_plus" if fit == "plus" else "fit_minus"], dtype=np.int64)
    base_rows = np.asarray(sup["b_rows"], dtype=np.int64)
    donor_rows = np.asarray(sup["d_plus" if fit == "plus" else "d_minus"], dtype=np.int64)
    monitor = np.asarray(sup["monitor"], dtype=np.int64)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    norm = D.load_norm()
    cache = D.load_cache("train")
    y = np.asarray(cache["labels"], dtype=np.float64)
    model, info = ZT.build_model(norm, seed, device)
    init_hash = C.state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=ZT.LR, betas=ZT.BETAS, eps=ZT.EPS,
                            weight_decay=ZT.WD)
    steps = math.ceil(len(rows_fit) / ZT.BATCH_SIZE)
    total = steps * epochs
    curve = []
    best = {"mae": float("inf"), "epoch": -1}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        batches = fit_batches(base_rows, donor_rows, epoch, seed)
        err, cnt = 0.0, 0
        for k_batch, (b_rows, d_rows) in enumerate(batches):
            rows = np.concatenate([b_rows, d_rows])
            batch = T1._to_device(T1.collate(cache, rows, "P3N"), device)
            target = torch.as_tensor(y[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), ZT.CLIP)
            if not math.isfinite(float(gn)):
                raise RuntimeError("non-finite gradient")
            curve_step = (epoch - 1) * steps + k_batch + 1
            rate = C.cosine_with_warmup_lr(curve_step, total, ZT.WARMUP, ZT.LR, ZT.MIN_LR)
            for group in opt.param_groups:
                group["lr"] = rate
            opt.step()
            err += float((pred.detach().view(-1) - target).abs().sum())
            cnt += len(rows)
        monitor_mae = T1.evaluate(model, cache, monitor, candidate="P3N", device=device,
                                  batch_size=256)[1]
        sec = time.perf_counter() - started
        curve.append({"epoch": epoch, "online_train_mae": err / cnt,
                      "monitor_mae": float(monitor_mae), "lr": rate, "steps": steps})
        if monitor_mae < best["mae"]:
            best = {"mae": float(monitor_mae), "epoch": int(epoch)}
            C.atomic_torch_save({"state_dict": C.cpu_state(model), "epoch": epoch,
                                 "monitor_mae": float(monitor_mae), "fit": fit, "seed": seed},
                                out / "best_monitor.pt")
        C.log(f"[sup {fit} s{seed}] ep={epoch:03d} monitor={monitor_mae:.6f} "
              f"train={err / cnt:.6f} elapsed={sec:.1f}s")
    C.atomic_torch_save({"state_dict": C.cpu_state(model), "epoch": epochs,
                         "monitor_mae": float(monitor_mae), "fit": fit, "seed": seed}, out / "final.pt")
    result = {
        "fit": fit, "seed": int(seed), "rows": int(len(rows_fit)), "base_rows": int(len(base_rows)),
        "donor_rows": int(len(donor_rows)), "steps_per_epoch": int(steps), "epochs": int(epochs),
        "updates": int(total), "best_monitor": best, "curve": curve,
        "init_state_sha256": init_hash,
        "model_info": info, "manifest_sha256": manifest["npz_sha256"],
        "wall_seconds": float(time.perf_counter() - started),
        "gpu_hours": None if str(device).startswith("cpu") else float(time.perf_counter() - started) / 3600.0,
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {"fit_rows": "official train only", "monitor_rows": "official train only",
                 "q_rows_used_for_training": False, "official_valid_or_test_read": False},
    }
    C.write_json(out / "train.json", result)
    # freeze per-molecule monitor + Q predictions is done separately after the
    # checkpoint hashes are recorded (see freeze_predictions)
    return result


def freeze_predictions(run_dirs: list[Path], out_path: Path) -> dict:
    """Write monitor/Q predictions for every frozen checkpoint; returns hashes."""
    sup, manifest = load_support()
    cache = D.load_cache("train")
    norm = D.load_norm()
    groups = {
        "monitor": np.asarray(sup["monitor"], dtype=np.int64),
        "q_target": np.asarray(sup["q_target"], dtype=np.int64),
        "q_background": np.asarray(sup["q_background"], dtype=np.int64),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"checkpoints": [], "groups": {k: len(v) for k, v in groups.items()},
           "manifest_sha256": manifest["npz_sha256"]}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        meta = C.read_json(run_dir / "train.json")
        for which in ("best_monitor", "final"):
            ckpt = run_dir / f"{which}.pt"
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model, _ = ZT.build_model(norm, meta["seed"], torch.device("cpu"))
            C.load_state(model, state["state_dict"], torch.device("cpu"))
            preds = {}
            for name, rows in groups.items():
                pred, mae = T1.evaluate(model, cache, rows, candidate="P3N",
                                        device=torch.device("cpu"), batch_size=256)
                preds[name] = pred
                out.setdefault("mae_by_run", {}).setdefault(f"{meta['fit']}_s{meta['seed']}", {})[
                    which] = {}
                out["mae_by_run"][f"{meta['fit']}_s{meta['seed']}"][which][name] = float(mae)
            np.savez(out_path.with_name(out_path.stem + f"_{meta['fit']}_s{meta['seed']}_{which}.npz"),
                     **{k: v for k, v in preds.items()})
            out["checkpoints"].append({
                "fit": meta["fit"], "seed": meta["seed"], "which": which,
                "epoch": int(state["epoch"]),
                "sha256": C.sha256_file(ckpt),
            })
    C.write_json(out_path, out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["build", "train", "freeze"])
    p.add_argument("--fit", default="plus", choices=["plus", "minus"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--out", default=None)
    p.add_argument("--runs", default="")
    args = p.parse_args(argv)
    if args.command == "build":
        build_support()
    elif args.command == "train":
        out = Path(args.out) if args.out else RUNS / f"sup_{args.fit}_s{args.seed}"
        train_fit(args.fit, args.seed, torch.device(args.device), out, epochs=args.epochs)
    else:
        dirs = [Path(x) for x in args.runs.split(",") if x]
        (ANALYSIS / "support_predictions.json").parent.mkdir(parents=True, exist_ok=True)
        freeze_predictions(dirs, CACHE / "artifacts" / "support_predictions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
