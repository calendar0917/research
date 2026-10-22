"""Kappa calibration on the shared epoch-50 prefix (train-only, no valid tuning).

For kappa in {0.1, 1, 10, 100} on a fixed set of 32 train-only probe batches and
both frozen partitions (R and S), one virtual source step is taken and two things
are measured on the *same* episodes:

* the fraction of episodes whose source L1 does not increase;
* the median per-molecule mean |f(theta') - f(theta)| on the query half, in raw-y
  units.

A kappa is admissible if >= 75% of episodes do not increase the source loss and
the median displacement lies in [0.005, 0.05]; among admissible values pick the
one closest to 0.02 raw-y (ties -> smaller kappa).  If the coarse grid jumps over
that window, at most three log-scale bisections are taken inside the bracketing
interval.  Nothing here reads the valid split, and no full training is run.
"""

from __future__ import annotations

import argparse
import sys
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
import zct_train as ZT  # noqa: E402

CACHE = _TASK / "cache"
ANALYSIS = _TASK / "analysis"


class Prober:
    def __init__(self, seed: int, device):
        self.seed = int(seed)
        ep = np.load(CACHE / "probe_episodes.npz")
        self.rows = np.asarray(ep["probe_rows"], dtype=np.int64)
        self.q = {"R": np.asarray(ep["probe_q"], dtype=bool),
                  "S": np.asarray(ep["probe_q_structural"], dtype=bool)}
        self.cache = D.load_cache("train")
        self.y = np.asarray(self.cache["labels"], dtype=np.float64)
        self.device = device
        self.model, self.info = ZT.build_model(D.load_norm(), self.seed, device)

    def load_state(self, path: Path) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        C.load_state(self.model, state["state_dict"], self.device)
        return {"state_epoch": int(state["epoch"]), "state_sha256": C.sha256_file(path),
                "model_state_sha256": C.state_hash(self.model)}

    def _predict(self, params, batch):
        return torch.func.functional_call(self.model, params, (batch,)).view(-1)

    def episode(self, k: int, part: str, alpha: float, batch_cache: dict) -> dict:
        rows = self.rows[k]
        q = self.q[part][k]
        s_rows, q_rows = rows[~q], rows[q]
        key = (k, part)
        if key not in batch_cache:
            bs = T1._to_device(T1.collate(self.cache, s_rows, "P3N"), self.device)
            bq = T1._to_device(T1.collate(self.cache, q_rows, "P3N"), self.device)
            ts = torch.as_tensor(self.y[s_rows], dtype=torch.float32, device=self.device)
            tq = torch.as_tensor(self.y[q_rows], dtype=torch.float32, device=self.device)
            batch_cache[key] = (bs, bq, ts, tq)
        bs, bq, ts, tq = batch_cache[key]
        params = dict(self.model.named_parameters())
        values = list(params.values())
        pred_s = self._predict(params, bs)
        l_s = F.l1_loss(pred_s, ts)
        g = torch.autograd.grad(l_s, values, create_graph=False, allow_unused=False)
        gn = torch.sqrt(sum((x * x).sum() for x in g))
        c = torch.clamp(ZT.CLIP / (gn + ZT.EPS), max=1.0)
        shifted = {n: p - alpha * c.detach() * gi.detach()
                   for n, p, gi in zip(params, values, g)}
        l_s2 = F.l1_loss(self._predict(shifted, bs), ts)
        pq0 = self._predict(params, bq)
        pq1 = self._predict(shifted, bq)
        per_mol = (pq1 - pq0).abs()   # one prediction per molecule
        return {
            "source_delta": float(l_s2 - l_s),
            "source_mae_theta": float(l_s),
            "query_shift_median_mol": float(per_mol.median()),
            "query_shift_mean": float(per_mol.mean()),
            "param_step_norm": float(alpha * c * gn),
            "g_norm": float(gn),
            "clip_c": float(c),
        }

    def evaluate(self, kappa: float, alpha_ref: float, batch_cache: dict) -> dict:
        alpha = float(kappa) * alpha_ref
        eps = []
        for k in range(len(self.rows)):
            for part in ("R", "S"):
                eps.append({"part": part, **self.episode(k, part, alpha, batch_cache)})
        src = np.asarray([e["source_delta"] for e in eps])
        shift = np.asarray([e["query_shift_median_mol"] for e in eps])
        per_part = {}
        for part in ("R", "S"):
            sub = [e for e in eps if e["part"] == part]
            per_part[part] = {
                "frac_source_not_worse": float(np.mean([e["source_delta"] <= 0 for e in sub])),
                "median_query_shift": float(np.median([e["query_shift_median_mol"] for e in sub])),
            }
        return {
            "kappa": float(kappa),
            "alpha": alpha,
            "n_episodes": len(eps),
            "frac_source_not_worse": float(np.mean(src <= 0)),
            "mean_source_delta": float(src.mean()),
            "median_query_shift": float(np.median(shift)),
            "mean_query_shift": float(shift.mean()),
            "p10_query_shift": float(np.percentile(shift, 10)),
            "p90_query_shift": float(np.percentile(shift, 90)),
            "mean_param_step_norm": float(np.mean([e["param_step_norm"] for e in eps])),
            "per_partition": per_part,
            "source_stable": bool(np.mean(src <= 0) >= 0.75),
            "shift_in_window": bool(0.005 <= float(np.median(shift)) <= 0.05),
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork", required=True, help="epoch-50 prefix checkpoint (fork.pt)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kappas", default="0.1,1,10,100")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    prober = Prober(args.seed, device)
    fork = prober.load_state(Path(args.fork))
    total_updates = 79 * ZT.EPOCHS
    alpha_ref = ZT.lr_at(50 * 79, total_updates)   # lr at the last prefix update
    batch_cache: dict = {}
    results = []
    kappas = [float(x) for x in args.kappas.split(",") if x]

    def add(k):
        r = prober.evaluate(k, alpha_ref, batch_cache)
        results.append(r)
        C.log({kk: (round(v, 5) if isinstance(v, float) else v)
               for kk, v in r.items() if kk != "per_partition"})
        return r

    for k in kappas:
        add(k)

    def admissible(r):
        return r["source_stable"] and r["shift_in_window"]

    bisections = 0
    while bisections < 3:
        ok = [r["kappa"] for r in results if admissible(r)]
        if ok:
            break
        ordered = sorted(results, key=lambda r: r["kappa"])
        bracket = None
        for a, b in zip(ordered, ordered[1:]):
            if a["median_query_shift"] < 0.005 and b["median_query_shift"] > 0.05:
                bracket = (a["kappa"], b["kappa"])
                break
            if b["kappa"] / a["kappa"] > 1.5 and b["kappa"] / a["kappa"] <= 11:
                if (a["median_query_shift"] < 0.005 and b["median_query_shift"] >= 0.005) or \
                   (a["median_query_shift"] <= 0.05 and b["median_query_shift"] > 0.05):
                    bracket = (a["kappa"], b["kappa"])
                    break
        if bracket is None:
            break
        new_k = float(np.sqrt(bracket[0] * bracket[1]))
        add(new_k)
        bisections += 1

    adm = [r for r in results if admissible(r)]
    if adm:
        chosen = sorted(adm, key=lambda r: (abs(r["median_query_shift"] - 0.02), r["kappa"]))[0]
        decision = "SELECTED"
    else:
        chosen = None
        decision = "NO_ADMISSIBLE_KAPPA"
    payload = {
        "fork": fork,
        "alpha_reference_lr": float(alpha_ref),
        "alpha_reference_step": int(50 * 79),
        "grid": [float(x) for x in kappas],
        "results": results,
        "chosen_kappa": None if chosen is None else float(chosen["kappa"]),
        "chosen": chosen,
        "decision": decision,
        "bisections_used": int(bisections),
        "rules": {
            "source_stable": ">= 75% of episodes have source L1 not increasing",
            "shift_window": "median per-molecule mean |f(theta')-f(theta)| in [0.005, 0.05] raw-y",
            "selection": "closest median displacement to 0.02; ties -> smaller kappa",
            "pooling": "32 train-only batches x both frozen partitions (R and S)",
        },
        "data": {"split": "official train only", "valid_or_test_read": False},
        "environment": C.environment(),
    }
    out = ANALYSIS / "kappa_calibration.json" if args.out is None else Path(args.out)
    C.write_json(out, payload)
    if chosen is not None:
        C.write_json(CACHE / "kappa.json", {
            "kappa": float(chosen["kappa"]),
            "calibration_sha256": C.sha256_file(out),
            "fork_state_sha256": fork["state_sha256"],
            "alpha_reference_lr": float(alpha_ref),
        })
    C.log({"decision": decision, "chosen_kappa": payload["chosen_kappa"],
           "bisections": bisections})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
