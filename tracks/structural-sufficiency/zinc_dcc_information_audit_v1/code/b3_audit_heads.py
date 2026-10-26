#!/usr/bin/env python
"""B3: freeze Backbone, fit J_hat and the Z/ZJ/ZN audit heads on ProbeFit,
and write AuditEval predictions + hashes.  AuditEval labels are guarded and
must not be indexed here.

Run once per supervised seed after the corresponding final DCC-L checkpoint
exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import KFold

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
_V1 = _REPO / "tracks" / "structural-sufficiency" / "zinc_arch_discovery_v1" / "code"
_DCC = _REPO / "tracks" / "structural-sufficiency" / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DCC), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import audit_common as AC  # noqa: E402
import dcc_models as DM  # noqa: E402
import joint_stats as JS  # noqa: E402

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
CV_SEED = 20261026
N_FOLDS = 5


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


class GuardedLabels:
    def __init__(self, arr, forbidden):
        self.arr = np.asarray(arr)
        self.forbidden = set(int(x) for x in forbidden)
        self.log = []

    def __getitem__(self, idx):
        rows = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        bad = [int(x) for x in rows if int(x) in self.forbidden]
        if bad:
            raise RuntimeError(f"forbidden AuditEval label access: {bad[:5]}")
        self.log.append(rows.copy())
        return self.arr[rows]


def fit8k_norm(cache, fit_rows):
    norm = dict(D.load_norm())
    gfit = np.asarray(cache["global_feat"], dtype=np.float64)[fit_rows]
    norm["global_mean"] = gfit.mean(0).astype(np.float32)
    norm["global_scale"] = np.maximum(gfit.std(0).astype(np.float32), 1e-6)
    return norm


def load_backbone(path, seed, norm, device):
    model, info = DM.build_model("DCC_L", norm, int(seed), device, organized_local=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    C.load_state(model, state, device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    if params != 494193:
        raise RuntimeError(f"unexpected parameter count {params}")
    return model, {"checkpoint": str(path), "checkpoint_sha256": C.sha256_file(path),
                   "parameters": int(params), "checkpoint_epoch": ck.get("epoch") if isinstance(ck, dict) else None,
                   "checkpoint_global_step": ck.get("global_step") if isinstance(ck, dict) else None}


@torch.no_grad()
def extract_all(model, cache, n, device, batch_size=128):
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    bi_all = np.asarray(cache["bond_index"])
    bt_all = np.asarray(cache["bond_type"])
    z = np.zeros((n, 513), dtype=np.float32)
    jl = np.zeros((n, JS.N_LATENT_DIMS), dtype=np.float32)
    pred = np.zeros((n,), dtype=np.float32)
    for start in range(0, n, batch_size):
        rows = np.arange(start, min(start + batch_size, n), dtype=np.int64)
        batch = AC.collate_rows(cache, rows, device=device)
        cap = AC.capture_forward(model, batch)
        z[rows] = cap["z"].detach().cpu().numpy()
        pred[rows] = cap["pred"].detach().cpu().numpy()
        c_loc = cap["c_loc"].detach().cpu().numpy()
        off = 0
        for row in rows:
            nb = int(bp[row + 1] - bp[row])
            c = c_loc[off:off + nb]
            off += nb
            rel = JS.molecule_relations(bi_all[bp[row]:bp[row + 1]], int(ap[row + 1] - ap[row]))
            jl[row] = JS.j_latent(c, rel)
    return z, jl, pred


def fit_ridge_cv(X, Y, alphas=ALPHAS, n_folds=N_FOLDS, seed=CV_SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = {}
    for alpha in alphas:
        mses = []
        for tr, va in kf.split(X):
            m = Ridge(alpha=float(alpha))
            m.fit(X[tr], Y[tr])
            mses.append(float(np.mean((m.predict(X[va]) - Y[va]) ** 2)))
        scores[float(alpha)] = float(np.mean(mses))
    best = min(scores, key=scores.get)
    m = Ridge(alpha=float(best))
    m.fit(X, Y)
    return m, {"best_alpha": float(best), "cv_mse": scores}


def std_fit(x):
    mu = np.asarray(x, dtype=np.float64).mean(0)
    sd = np.asarray(x, dtype=np.float64).std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def std_apply(x, mu, sd):
    return (np.asarray(x, dtype=np.float64) - mu) / sd


def save_model(path, model, xmu, xsd, ymu, ysd, info):
    np.savez(path, coef=model.coef_, intercept=model.intercept_,
             xmu=xmu, xsd=xsd, ymu=ymu, ysd=ysd, alpha=info["best_alpha"])
    return C.sha256_file(path)


def build_graph(cache, row):
    ap = np.asarray(cache["atom_ptr"])
    bp = np.asarray(cache["bond_ptr"])
    at = np.asarray(cache["atom_type"])
    bi = np.asarray(cache["bond_index"])
    bt = np.asarray(cache["bond_type"])
    g = nx.Graph()
    for j, a in enumerate(at[ap[row]:ap[row + 1]]):
        g.add_node(int(j), atom=int(a))
    for (u, v), t in zip(bi[bp[row]:bp[row + 1]], bt[bp[row]:bp[row + 1]]):
        g.add_edge(int(u), int(v), bond=int(t))
    return g


def radius2_keys(g):
    hashes = nx.weisfeiler_lehman_subgraph_hashes(
        g, node_attr="atom", edge_attr="bond", iterations=2,
        digest_size=16, include_initial_labels=True,
    )
    return [hashes[int(n)][-1] for n in range(g.number_of_nodes())]


def support_audit(cache, fit_rows, audit_rows):
    support = defaultdict(set)
    for r in fit_rows:
        g = build_graph(cache, int(r))
        for k in radius2_keys(g):
            support[k].add(int(r))
    atom_ptr = [0]
    atom_counts = []
    min_support = []
    n_unseen = []
    n_low = []
    for r in audit_rows:
        g = build_graph(cache, int(r))
        counts = [len(support.get(k, ())) for k in radius2_keys(g)]
        atom_counts.extend(counts)
        atom_ptr.append(len(atom_counts))
        min_support.append(int(min(counts)) if counts else 0)
        n_unseen.append(int(sum(1 for c in counts if c == 0)))
        n_low.append(int(sum(1 for c in counts if 1 <= c <= 4)))
    cls = []
    for u, lo in zip(n_unseen, n_low):
        cls.append("unseen" if u > 0 else ("low_support" if lo > 0 else "full_support"))
    return {
        "row": np.asarray(audit_rows, dtype=np.int64),
        "class": np.asarray(cls),
        "min_support": np.asarray(min_support, dtype=np.int64),
        "n_unseen_atoms": np.asarray(n_unseen, dtype=np.int64),
        "n_low_atoms": np.asarray(n_low, dtype=np.int64),
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "atom_counts": np.asarray(atom_counts, dtype=np.int64),
        "n_fit_environments": int(len(support)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    out = Path(args.out_dir) if args.out_dir else _TRACK / "analysis" / "b" / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    split = json.loads((_TRACK / "configs" / "split_fit8k_probe_audit.json").read_text())
    fit = np.asarray(split["rows"]["fit"], dtype=np.int64)
    probe = np.asarray(split["rows"]["probe"], dtype=np.int64)
    audit = np.asarray(split["rows"]["audit"], dtype=np.int64)
    cache = D.load_cache("train")
    n = int(cache["n_molecules"])
    j_raw_path = _TRACK / "analysis" / "j_raw.npy"
    if j_raw_path.exists():
        j_raw = np.load(j_raw_path).astype(np.float64)
    else:
        # Structure-only fallback: regenerate the frozen J_raw table from the
        # train cache rather than depending on an ignored .npy artifact.
        ap = np.asarray(cache["atom_ptr"])
        bp = np.asarray(cache["bond_ptr"])
        bi_all = np.asarray(cache["bond_index"])
        bt_all = np.asarray(cache["bond_type"])
        j_raw = np.zeros((n, JS.N_RAW_DIMS), dtype=np.float64)
        for i in range(n):
            rel = JS.molecule_relations(bi_all[bp[i]:bp[i + 1]], int(ap[i + 1] - ap[i]))
            j_raw[i] = JS.j_raw(bt_all[bp[i]:bp[i + 1]], rel)
    assert j_raw.shape == (n, JS.N_RAW_DIMS)
    # remove labels from the cache, then wrap them so AuditEval rows are
    # inaccessible in this process.
    labels = np.asarray(cache.pop("labels"), dtype=np.float64)
    guard = GuardedLabels(labels, forbidden=audit)
    y_probe = guard[probe]

    norm = fit8k_norm(cache, fit)
    model, model_info = load_backbone(args.checkpoint, args.seed, norm, device)
    z_all, jl_all, pred_all = extract_all(model, cache, n, device)
    np.save(out / "z_all.npy", z_all)
    np.save(out / "j_latent_all.npy", jl_all)
    np.save(out / "pred_all.npy", pred_all)

    # ---- J_hat fitted on Fit8k only (no labels) --------------------------
    Xf = z_all[fit]
    xmu, xsd = std_fit(Xf)
    Xf_s = std_apply(Xf, xmu, xsd)
    jhat = {}
    for name, Y in (("raw", j_raw[fit]), ("latent", jl_all[fit])):
        ymu, ysd = std_fit(Y)
        Ys = std_apply(Y, ymu, ysd)
        m, info = fit_ridge_cv(Xf_s, Ys)
        jhat[name] = {
            "model": m, "xmu": xmu, "xsd": xsd, "ymu": ymu, "ysd": ysd, "info": info,
            "probe": std_apply(z_all[probe], xmu, xsd) @ m.coef_.T + m.intercept_,
            "audit": std_apply(z_all[audit], xmu, xsd) @ m.coef_.T + m.intercept_,
        }
        jhat[name]["probe"] = jhat[name]["probe"] * ysd + ymu
        jhat[name]["audit"] = jhat[name]["audit"] * ysd + ymu
    jhat_raw_probe, jhat_raw_audit = jhat["raw"]["probe"], jhat["raw"]["audit"]
    jhat_lat_probe, jhat_lat_audit = jhat["latent"]["probe"], jhat["latent"]["audit"]
    jhat_files = {}
    for name in ("raw", "latent"):
        path = out / f"jhat_{name}.npz"
        h = save_model(path, jhat[name]["model"], jhat[name]["xmu"], jhat[name]["xsd"],
                       jhat[name]["ymu"], jhat[name]["ysd"], jhat[name]["info"])
        jhat_files[name] = {"path": str(path), "sha256": h}

    # ---- audit heads on ProbeFit -----------------------------------------
    resid_probe = y_probe - pred_all[probe]
    inputs_probe = {
        "Z": z_all[probe],
        "ZJ": np.concatenate([z_all[probe], j_raw[probe], jl_all[probe]], axis=1),
        "ZN": np.concatenate([z_all[probe], jhat_raw_probe, jhat_lat_probe], axis=1),
    }
    inputs_audit = {
        "Z": z_all[audit],
        "ZJ": np.concatenate([z_all[audit], j_raw[audit], jl_all[audit]], axis=1),
        "ZN": np.concatenate([z_all[audit], jhat_raw_audit, jhat_lat_audit], axis=1),
    }
    heads = {}
    preds = {}
    for arm in ("Z", "ZJ", "ZN"):
        Xp = inputs_probe[arm]
        xmu, xsd = std_fit(Xp)
        Xp_s = std_apply(Xp, xmu, xsd)
        ymu, ysd = std_fit(resid_probe)
        yp_s = std_apply(resid_probe, ymu, ysd)
        m, info = fit_ridge_cv(Xp_s, yp_s)
        Xa_s = std_apply(inputs_audit[arm], xmu, xsd)
        Xp2_s = std_apply(inputs_probe[arm], xmu, xsd)
        preds[f"Z{'' if arm == 'Z' else arm[1:]}_probe"] = pred_all[probe] + (m.predict(Xp2_s) * ysd + ymu)
        preds[f"Z{'' if arm == 'Z' else arm[1:]}_audit"] = pred_all[audit] + (m.predict(Xa_s) * ysd + ymu)
        path = out / f"head_{arm}.npz"
        h = save_model(path, m, xmu, xsd, ymu, ysd, info)
        heads[arm] = {"path": str(path), "sha256": h, "input_dim": int(Xp.shape[1]),
                      "alpha": info, "cv_fold_seed": CV_SEED}
        # one fixed robust sensitivity, no tuning
        try:
            hub = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=300)
            hub.fit(Xp_s, yp_s)
            hub_pred = pred_all[audit] + (hub.predict(Xa_s) * ysd + ymu)
            np.save(out / f"huber_{arm}_audit.npy", hub_pred)
            heads[arm]["huber"] = {"epsilon": 1.35, "alpha": 1e-4, "max_iter": 300,
                                   "prediction_sha256": sha_array(hub_pred),
                                   "n_iter": int(getattr(hub, "n_iter_", -1))}
        except Exception as exc:  # sensitivity only; never replaces the main head
            heads[arm]["huber"] = {"error": repr(exc)}
    preds["raw_probe"] = pred_all[probe]
    preds["raw_audit"] = pred_all[audit]

    # ---- label-free support strata for AuditEval -------------------------
    sup = support_audit(cache, fit, audit)
    sp = out / "support_audit.npz"
    np.savez(sp, **sup)
    support_info = {"path": str(sp), "sha256": C.sha256_file(sp),
                    "n_fit_environments": sup["n_fit_environments"],
                    "class_counts": {k: int(v) for k, v in zip(*np.unique(sup["class"], return_counts=True))}}

    # ---- freeze + hashes -------------------------------------------------
    arrays = {}
    for name, arr in (("z_all", z_all), ("j_latent_all", jl_all), ("pred_all", pred_all),
                      ("y_probe", y_probe), ("j_raw", j_raw)):
        arrays[name] = {"sha256": sha_array(arr), "shape": list(arr.shape)}
    for name in ("Z_probe", "Z_audit", "ZJ_probe", "ZJ_audit", "ZN_probe", "ZN_audit",
                 "raw_probe", "raw_audit"):
        arrays[name] = {"sha256": sha_array(preds[name]), "shape": list(preds[name].shape)}
    # save the canonical AuditEval and ProbeFit prediction arrays explicitly
    for name in ("raw_audit", "Z_audit", "ZJ_audit", "ZN_audit"):
        np.save(out / f"pred_{name}.npy", preds[name])
    for name in ("raw_probe", "Z_probe", "ZJ_probe", "ZN_probe"):
        np.save(out / f"pred_{name}.npy", preds[name])
    prediction_hashes = {}
    for name in ("raw_audit", "Z_audit", "ZJ_audit", "ZN_audit",
                 "raw_probe", "Z_probe", "ZJ_probe", "ZN_probe"):
        prediction_hashes[name] = C.sha256_file(out / f"pred_{name}.npy")

    payload = {
        "protocol": "zinc-dcc-information-audit-v1",
        "phase": "B3-freeze",
        "seed": int(args.seed),
        "git_commit": C.git_commit(),
        "checkpoint": model_info,
        "rows": {"fit_sha256": sha_array(fit), "probe_sha256": sha_array(probe),
                 "audit_sha256": sha_array(audit), "n_fit": len(fit),
                 "n_probe": len(probe), "n_audit": len(audit)},
        "normalization": {"global33": "Fit8k only",
                          "global_mean_sha256": C.sha256_array(norm["global_mean"]),
                          "global_scale_sha256": C.sha256_array(norm["global_scale"])},
        "arrays": arrays,
        "jhat": jhat_files,
        "heads": heads,
        "predictions": prediction_hashes,
        "support": support_info,
        "label_guard": {"forbidden_audit_rows": int(len(audit)),
                        "accessed_row_blocks": int(len(guard.log)),
                        "accessed_unique_rows": int(len(set(int(x) for a in guard.log for x in a))),
                        "audit_labels_accessed": False,
                        "note": "AuditEval labels are not in this process's accessible label array"},
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "runtime_seconds": float(time.time() - t0),
    }
    (out / "freeze.json").write_text(json.dumps(AC.jsonable(payload), indent=2))
    print(json.dumps({"seed": args.seed, "runtime_seconds": payload["runtime_seconds"],
                      "support": support_info["class_counts"],
                      "prediction_hashes": prediction_hashes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
