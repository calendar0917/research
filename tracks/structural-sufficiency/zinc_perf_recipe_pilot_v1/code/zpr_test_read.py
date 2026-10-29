"""ZPR-v1 terminal official-test read — **exactly once**, after the decision is frozen.

    python zpr_test_read.py --verdict NO_COMPELLING_RECIPE_GAIN --device cuda:0

Order of operations (any failure before step 4 aborts without spending test):

1. refuse if ``runs/test_read/test_read.json`` already exists (read-once guard);
2. freeze first: checkpoint SHA-256 for all four ZPR checkpoints, their valid-MAE
   replay against ``train.json``, the decision file, the git commit;
3. self-check the test grouping rule: the radius-2 typed-environment support
   computed here must reproduce the frozen valid anchor exactly (251/251 unseen);
   otherwise abort — the test groups would not be comparable to the valid groups;
4. build the label-free test feature cache, then read the official test labels
   **once** (``lc_data.official_labels('test')``) and attach them in memory only;
5. evaluate the four checkpoints, plus a reproduction check of the two published
   historical baselines straight from their frozen per-molecule prediction CSVs
   (this validates the label order end to end without loading those models);
6. write ``runs/test_read/test_read.json`` + ``test_per_molecule.csv``.

Everything written here is ``TEST_AS_DEV_DESCRIPTIVE``.  The official test split
is already spent in this repository; these numbers are a descriptive record and
must not drive beta, EMA, epoch, or architecture choices.  Nothing in this
script selects anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

import zpr_common as Z

L, D, M, T = Z.L, Z.D, Z.M, Z.T

HIST_LCRC_TEST_CSV = (Z.LCRC_TRACK / "analysis" / "test_per_molecule_lcrcA_s0.csv")
HIST_DCC_TEST_CSV = (Z.REPO / "tracks/structural-sufficiency/zinc_performance_search_v1"
                     / "analysis" / "test_pred_B-R0-s0_s0.csv")
PUBLISHED = {"lcrcA_s0": 0.06678237025253475, "dcc_l_s0": 0.066267}

CHECKPOINTS = {
    "G0_online": ("G0", "best.pt"),
    "G0_ema": ("G0", "ema_best.pt"),
    "G1_online": ("G1", "best.pt"),
    "G1_ema": ("G1", "ema_best.pt"),
}


def _load_ckpt(path: Path) -> tuple[dict, dict]:
    """Frozen checkpoint payload -> (state_dict, payload); config comes from the file."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["state_dict"], payload


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _anchor_selfcheck() -> dict:
    """Does the radius-2 support rule used for the test groups reproduce the anchor?"""
    import zpr_phase_a as P

    train_graphs, _ = P.build_split_graphs("train")
    valid_graphs, _ = P.build_split_graphs("valid")
    support = P.train_support(train_graphs)
    unseen = np.array([P.support_features(g, support)["r2_unseen_atoms"] > 0
                       for g in valid_graphs])
    ref = np.asarray(Z.read_anchor()["unseen"], dtype=bool)
    return {
        "n_valid": int(len(unseen)),
        "n_unseen_rule": int(unseen.sum()),
        "n_unseen_anchor": int(ref.sum()),
        "agreement": float((unseen == ref).mean()),
        "exact": bool(np.array_equal(unseen, ref)),
        "support": support,
        "train_graphs": train_graphs,
    }


def _test_groups(support: dict, train_graphs: list, y: np.ndarray) -> tuple[dict, dict]:
    import zpr_phase_a as P

    test_graphs, _ = P.build_split_graphs("test")
    if len(test_graphs) != len(y):
        raise RuntimeError(f"test graphs {len(test_graphs)} vs labels {len(y)}")
    unseen = np.array([P.support_features(g, support)["r2_unseen_atoms"] > 0
                       for g in test_graphs])
    n_atoms = np.array([g.n for g in test_graphs])
    ordinary = y > -3.0
    masks = {
        "full": np.ones(len(y), dtype=bool),
        "ordinary": ordinary,
        "r2_unseen": unseen,
        "r2_unseen_ordinary": unseen & ordinary,
        "seen_ordinary": (~unseen) & ordinary,
        "tail": ~ordinary,
    }
    info = {
        "n": int(len(y)),
        "n_ordinary": int(ordinary.sum()),
        "n_tail": int((~ordinary).sum()),
        "n_r2_unseen": int(unseen.sum()),
        "n_r2_unseen_ordinary": int((unseen & ordinary).sum()),
        "n_seen_ordinary": int(((~unseen) & ordinary).sum()),
        "mean_atoms": float(n_atoms.mean()),
    }
    return masks, info


def _group_mae(pred: np.ndarray, y: np.ndarray, masks: dict) -> dict:
    ae = np.abs(pred - y)
    return {k: (float(ae[m].mean()) if m.any() else None) for k, m in masks.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ZPR-v1 terminal official-test read (once)")
    ap.add_argument("--verdict", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Z.RUNS / "test_read"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    Z.ensure_dir(out)
    report_path = out / "test_read.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists: the official test was already read once "
                         f"for this round; refusing a second read")

    decision = Z.read_json(Z.ANALYSIS / "decision.json")
    if decision["verdict"] != args.verdict:
        raise SystemExit(f"verdict mismatch: decision.json says {decision['verdict']!r}, "
                         f"CLI says {args.verdict!r}")
    if decision.get("official_test_read"):
        raise SystemExit("decision.json already records a test read")
    L.require_cuda(args.device)

    # ---- 1. freeze before touching test -----------------------------------
    norm = D.load_norm()
    valid_cache = D.load_cache("valid")
    rows_valid = np.arange(valid_cache["n_molecules"])
    y_valid = np.asarray(valid_cache["labels"], dtype=np.float64)
    freeze = {"checkpoints": {}, "valid_replay": {}}
    for name, (arm, fname) in CHECKPOINTS.items():
        path = Z.RUNS / f"{arm}_s0" / fname
        if not path.exists():
            raise SystemExit(f"missing frozen checkpoint {Z.track_rel(path)}")
        state, payload_ck = _load_ckpt(path)
        model = M.load_model_from_state("A", norm, state, args.device,
                                        payload_ck.get("config") or dict(M.DEFAULT_CONFIG))
        _, valid_mae = T.evaluate(model, valid_cache, rows_valid, device=args.device,
                                 batch_size=256)
        del model
        meta = Z.read_json(Z.RUNS / f"{arm}_s0" / "train.json")
        expect = (meta["best"]["valid_mae"] if name.endswith("online")
                  else meta["ema"]["best"]["valid_mae"])
        freeze["checkpoints"][name] = {
            "path": Z.track_rel(path),
            "sha256": _sha256(path),
            "valid_mae_replay": float(valid_mae),
            "valid_mae_recorded": float(expect),
            "abs_diff": abs(float(valid_mae) - float(expect)),
            "parameters": int(meta["param_audit"]["parameters"]),
        }
    replay_ok = all(v["abs_diff"] < 1e-6 for v in freeze["checkpoints"].values())
    if not replay_ok:
        raise SystemExit(f"checkpoint/valid replay mismatch before test: {freeze['checkpoints']}")
    freeze.update({
        "frozen_before_test": True,
        "official_test_loaded_before_freeze": False,
        "git_commit": L.C.git_commit(),
        "decision": {"verdict": decision["verdict"],
                     "decision_inputs": decision["decision_inputs"],
                     "decision_stage": decision["decision_stage"]},
        "decision_sha256": _sha256(Z.ANALYSIS / "decision.json"),
        "valid_labels_sha256": Z.sha256_file(D.cache_path("valid")),
        "created": Z.now(),
    })
    Z.write_json(out / "test_freeze.json", freeze)
    Z.log(f"[test] freeze written before any test access: {Z.track_rel(out / 'test_freeze.json')}")

    # ---- 2. grouping rule self-check (aborts before the test read) ---------
    anchor = _anchor_selfcheck()
    support, train_graphs = anchor["support"], anchor["train_graphs"]
    anchor_ok = anchor["exact"]
    anchor_summary = {k: v for k, v in anchor.items() if k not in ("support", "train_graphs")}
    if not anchor_ok:
        raise SystemExit(f"radius-2 grouping rule does not reproduce the frozen valid anchor: "
                         f"{anchor_summary}")

    # ---- 3. the single official test read ---------------------------------
    test_path = D.cache_path("test")
    if test_path.exists():
        test = D.load_cache("test")
        test_source = "existing_label_free_cache"
    else:
        test = D.build_split("test", with_labels=False)
        D.save_cache(test, "test")
        test = D.load_cache("test")
        test_source = "built_from_official_pyg_test"
    if "labels" in test:
        raise SystemExit("test cache unexpectedly carries labels; refusing")
    labels = np.asarray(L.D.official_labels("test"), dtype=np.float64)
    if len(labels) != test["n_molecules"]:
        raise RuntimeError("test label/prediction length mismatch")
    n_test = len(labels)
    test["labels"] = labels.astype(np.float32)  # in memory only, never saved
    rows_test = np.arange(n_test)
    Z.log(f"[test] official test labels read once: n={n_test} source={test_source}")

    # ---- 4. evaluate the four frozen checkpoints --------------------------
    preds = {}
    for name, (arm, fname) in CHECKPOINTS.items():
        state, payload_ck = _load_ckpt(Z.RUNS / f"{arm}_s0" / fname)
        model = M.load_model_from_state("A", norm, state, args.device,
                                        payload_ck.get("config") or dict(M.DEFAULT_CONFIG))
        pred, _ = T.evaluate(model, test, rows_test, device=args.device, batch_size=256)
        del model
        preds[name] = pred.astype(np.float64)

    masks, group_info = _test_groups(support, train_graphs, labels)
    results = {}
    for name, pred in preds.items():
        ae = np.abs(pred - labels)
        results[name] = {
            "test_mae": float(ae.mean()),
            "test_rmse": float(np.sqrt(np.mean((pred - labels) ** 2))),
            "test_p90_ae": float(np.percentile(ae, 90)),
            "test_max_ae": float(ae.max()),
            "pearson_r": float(np.corrcoef(pred, labels)[0, 1]),
            "groups": _group_mae(pred, labels, masks),
            "worst10_rows": [int(r) for r in np.argsort(ae)[-10:][::-1]],
            "worst10_ae": [float(ae[r]) for r in np.argsort(ae)[-10:][::-1]],
        }

    # ---- 5. reproduction of the two published historical baselines --------
    hist = {}
    for tag, path in (("lcrcA_s0", HIST_LCRC_TEST_CSV), ("dcc_l_s0", HIST_DCC_TEST_CSV)):
        if not path.exists():
            hist[tag] = {"status": "missing", "path": Z.track_rel(path)}
            continue
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        idx = np.array([int(r["row"]) for r in rows], dtype=np.int64)
        hp = np.array([float(r["pred"]) for r in rows], dtype=np.float64)
        if len(idx) != n_test or not np.array_equal(idx, rows_test):
            hist[tag] = {"status": "row_mismatch", "n_rows": int(len(idx))}
            continue
        mae = float(np.abs(hp - labels).mean())
        hist[tag] = {"status": "reproduced", "test_mae": mae,
                     "published": PUBLISHED[tag],
                     "abs_diff": abs(mae - PUBLISHED[tag]),
                     "matches_published": bool(abs(mae - PUBLISHED[tag]) < 1e-6)}
        preds[f"hist_{tag}"] = hp

    # ---- 6. paired descriptive contrasts ---------------------------------
    def contrast(a: str, b: str) -> dict:
        ae_a = np.abs(preds[a] - labels)
        ae_b = np.abs(preds[b] - labels)
        d = ae_b - ae_a
        return {"mae_a": float(ae_a.mean()), "mae_b": float(ae_b.mean()),
                "delta_b_minus_a": float(d.mean()),
                "wins_b": int((ae_b < ae_a - 1e-9).sum()),
                "wins_a": int((ae_b < ae_a + 1e-9).sum())}

    contrasts = {
        "smooth_l1_vs_l1_online": contrast("G0_online", "G1_online"),
        "smooth_l1_vs_l1_ema": contrast("G0_ema", "G1_ema"),
        "ema_vs_online_G0": contrast("G0_online", "G0_ema"),
        "ema_vs_online_G1": contrast("G1_online", "G1_ema"),
    }
    if "hist_lcrcA_s0" in preds:
        contrasts["G0_online_vs_hist_lcrcA_s0"] = contrast("hist_lcrcA_s0", "G0_online")

    payload = {
        "round": Z.ROUND_ID,
        "label": "TEST_AS_DEV_DESCRIPTIVE",
        "descriptive_only": True,
        "test_is_spent": True,
        "read_count": 1,
        "official_test_loaded_before_freeze": False,
        "read_policy": "one read, after checkpoint selection and the §2.4 decision were frozen; "
                       "no beta, EMA, epoch or architecture choice follows from it",
        "git_commit": L.C.git_commit(),
        "environment": L.env_summary(),
        "n_test": int(n_test),
        "test_cache": {"path": Z.track_rel(test_path), "source": test_source},
        "label_source": "lc_data.official_labels('test') (terminal reader)",
        "checkpoints": freeze["checkpoints"],
        "grouping_rule_selfcheck": anchor_summary,
        "group_info": group_info,
        "results": results,
        "historical_reproduction": hist,
        "contrasts": contrasts,
        "created": Z.now(),
    }
    Z.write_json(report_path, payload)

    header = ["row", "y"] + list(preds) + [f"ae_{k}" for k in preds]
    with open(out / "test_per_molecule.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in rows_test:
            w.writerow([int(i), float(labels[i])] + [float(preds[k][i]) for k in preds]
                       + [abs(float(preds[k][i]) - float(labels[i])) for k in preds])

    # the decision file now records that test has been read
    decision["official_test_read"] = True
    decision["test_read_label"] = "TEST_AS_DEV_DESCRIPTIVE"
    decision["test_read_path"] = Z.track_rel(report_path)
    decision["created"] = Z.now()
    Z.write_json(Z.ANALYSIS / "decision.json", decision)

    Z.log("[test] " + ", ".join(f"{k}={v['test_mae']:.6f}" for k, v in results.items()))
    Z.log("[test] historical reproduction: "
          + ", ".join(f"{k}={v.get('test_mae')}" for k, v in hist.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
