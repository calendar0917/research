"""Terminal, reporting-only official-test read for the frozen shortlist.

Order of operations (the freeze is written *before* any test label is read):

1. read ``analysis/frozen_shortlist.json`` (committed at the freeze);
2. for every entry, re-load the checkpoint, re-score the full 1,000 valid
   molecules and require the recomputed MAE to equal the recorded best-valid
   MAE (1e-6) -- so the thing being tested is the thing that was ranked;
3. write ``analysis/test_freeze.json`` with the checkpoint sha256s;
4. only then read the official test labels once and report descriptions.

No test number selects a checkpoint, an epoch, a recipe or a direction, and
this script never writes back into ``runs/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_GRIT = _SS / "zinc_performance_search_v1" / "code"
_DBT = _SS / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_GRIT), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
AUDIT = _SS / "zinc_generalization_audit_v1" / "analysis" / "per_molecule.csv"

CAND = {"A": "P3N", "B": "DCC_L"}


def build(entry: dict, device):
    arm, seed = entry["arm"], int(entry["seed"])
    if arm in CAND:
        import ad2_models as A2
        model, _ = A2.build_model(CAND[arm], D.load_norm(), seed, device)
        return model, "ad"
    if arm == "C":
        import dbt_data as BD
        import dbt_models as BM
        scales = C.read_json(_SS / "zinc_directed_bond_transformer_v1" / "analysis" / "graph_scales.json")
        model, _ = BM.build_model(D.load_norm(), seed, device, graph_scales=scales,
                                  dropout=float(entry.get("dropout", 0.05)))
        return model, "dbt"
    raise ValueError(arm)


def predict(model, kind, arm, split: str, device, batch_size: int = 256):
    if kind == "ad":
        cache = D.load_cache(split)
        rows = np.arange(len(cache["labels"]), dtype=np.int64)
        pred, mae = T.evaluate(model, cache, rows, candidate=CAND[arm], device=device,
                              batch_size=batch_size)
        return pred, np.asarray(cache["labels"], np.float64)
    import dbt_data as BD
    import dbt_train as BT
    cache = BD.load_object_cache(split)
    rel = BD.load_relation_split(split)
    rows = np.arange(len(cache["labels"]), dtype=np.int64)
    pred, mae = BT.evaluate(model, cache, rel, rows, device=device, batch_size=batch_size)
    return pred, np.asarray(cache["labels"], np.float64)


def describe(pred, y) -> dict:
    ae = np.abs(pred - y)
    signed = pred - y
    return {"mae": float(ae.mean()), "rmse": float(np.sqrt((signed ** 2).mean())),
            "median_ae": float(np.median(ae)), "p90_ae": float(np.quantile(ae, 0.90)),
            "p95_ae": float(np.quantile(ae, 0.95)), "max_ae": float(ae.max()),
            "mean_signed": float(signed.mean()), "n": int(len(y))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shortlist", default=str(ANALYSIS / "frozen_shortlist.json"))
    p.add_argument("--freeze-out", default=str(ANALYSIS / "test_freeze.json"))
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    shortlist = C.read_json(Path(args.shortlist))
    entries = shortlist["entries"]

    # -- 1) reproduce valid for every frozen entry ---------------------------
    reproduced = []
    for e in entries:
        ckpt = C.REPO / e["checkpoint"]
        sha = C.sha256_file(ckpt)
        if e.get("checkpoint_sha256") and sha != e["checkpoint_sha256"]:
            raise RuntimeError(f"{e['name']}: checkpoint sha256 mismatch")
        model, kind = build(e, device)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if int(state["epoch"]) != int(e["epoch"]):
            raise RuntimeError(f"{e['name']}: epoch mismatch")
        C.load_state(model, state["state_dict"], device)
        pred, y = predict(model, kind, e["arm"], "valid", device, args.batch_size)
        mae = C.mae(pred, y)
        ok = abs(mae - float(e["valid_mae"])) <= 1e-6
        C.log(f"[test] {e['name']} valid {mae:.9f} recorded {e['valid_mae']:.9f} "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise RuntimeError(f"{e['name']}: valid reproduction failed")
        reproduced.append({**e, "checkpoint_sha256": sha, "valid_mae_recomputed": float(mae)})

    freeze = {
        "protocol": "zinc-performance-search-v1 terminal test read",
        "frozen_at": C.utcnow(),
        "git_commit": C.git_commit(),
        "entries": reproduced,
        "selection": shortlist.get("selection"),
        "test_loaded_before_this_file": False,
    }
    C.write_json(Path(args.freeze_out), freeze)
    C.log(f"[freeze] wrote {args.freeze_out} for {len(reproduced)} entries")
    if args.no_test:
        C.log("[freeze] --no-test: stopping before any test label is read")
        return 0

    # -- 2) the one terminal test read --------------------------------------
    groups = None
    if AUDIT.exists():
        import csv
        with AUDIT.open() as fh:
            groups = list(csv.DictReader(fh))
    table = []
    per_mol = {}
    for e in reproduced:
        model, kind = build(e, device)
        state = torch.load(C.REPO / e["checkpoint"], map_location="cpu", weights_only=False)
        C.load_state(model, state["state_dict"], device)
        pred, y = predict(model, kind, e["arm"], "test", device, args.batch_size)
        d = describe(pred, y)
        d.update({"name": e["name"], "arm": e["arm"], "recipe": e["recipe"],
                  "policy": e["policy"], "seed": int(e["seed"]),
                  "epoch": int(e["epoch"]), "valid_mae": float(e["valid_mae"]),
                  "params": int(e["params"]), "checkpoint": e["checkpoint"]})
        ae = np.abs(pred - y)
        d["mae_drop_row172"] = float(np.delete(ae, 172).mean())
        d["mae_drop_top10"] = float(np.sort(ae)[10:].mean())
        table.append(d)
        per_mol[e["name"]] = {"pred": pred, "y": y}
        C.log(f"[test] {e['name']} s{e['seed']} test MAE {d['mae']:.6f} "
              f"(valid {d['valid_mae']:.6f})")

    rows = []
    if groups:
        import csv
        keys = [k for k in groups[0] if k not in ("row", "y_raw")]
        for e in reproduced:
            pm = per_mol[e["name"]]
            ae = np.abs(pm["pred"] - pm["y"])
            rec = {"name": e["name"], "seed": int(e["seed"])}
            for k in keys:
                vals = np.array([g[k] for g in groups])
                try:
                    vals = vals.astype(float)
                except ValueError:
                    continue
                m = vals > 0
                if m.sum() == 0:
                    continue
                rec[k] = float(ae[m].mean())
            rows.append(rec)
    out = {"freeze": str(Path(args.freeze_out)), "test": table,
           "test_by_group": rows, "bootstrap": _bootstrap(per_mol)}
    C.write_json(ANALYSIS / "test_table.json", out)
    _markdown(table, rows, out["bootstrap"])
    C.log(f"[test] wrote {ANALYSIS / 'test_table.json'} and test_table.md")
    return 0


def _bootstrap(per_mol) -> list:
    rng = np.random.default_rng(20261023)
    out = []
    names = list(per_mol)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = np.abs(per_mol[a]["pred"] - per_mol[a]["y"])
            db = np.abs(per_mol[b]["pred"] - per_mol[b]["y"])
            n = len(da)
            idx = rng.integers(0, n, size=(10000, n))
            diff = da[idx].mean(1) - db[idx].mean(1)
            out.append({"a": a, "b": b, "delta_mae": float(da.mean() - db.mean()),
                        "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
                        "p_a_better": float((diff < 0).mean())})
    return out


def _markdown(table, rows, boot) -> None:
    lines = ["# Terminal official-test read (frozen shortlist)", "",
             "Test is read once, after the freeze; no post-test tuning.", "",
             "| model | seed | policy | epoch | params | valid MAE | test MAE | test RMSE | p90 AE | drop row172 | drop top10 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in table:
        lines.append(
            f"| {d['name']} | {d['seed']} | {d['policy']} | {d['epoch']} | {d['params']:,} "
            f"| {d['valid_mae']:.6f} | {d['mae']:.6f} | {d['rmse']:.4f} | {d['p90_ae']:.4f} "
            f"| {d['mae_drop_row172']:.6f} | {d['mae_drop_top10']:.6f} |")
    if rows:
        keys = [k for k in rows[0] if k not in ("name", "seed")]
        lines += ["", "## Test MAE by frozen group", "",
                  "| model | seed | " + " | ".join(keys) + " |",
                  "|---|---|" + "---|" * len(keys)]
        for r in rows:
            lines.append(f"| {r['name']} | {r['seed']} | "
                         + " | ".join(f"{r.get(k, float('nan')):.5f}" for k in keys) + " |")
    if boot:
        lines += ["", "## Paired row bootstrap on test (10k, seed 20261023)", "",
                  "| a | b | delta MAE (a-b) | 95% CI | P(a better) |", "|---|---|---|---|---|"]
        for b in boot:
            lines.append(f"| {b['a']} | {b['b']} | {b['delta_mae']:+.6f} "
                         f"| [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | {b['p_a_better']:.3f} |")
    (ANALYSIS / "test_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
