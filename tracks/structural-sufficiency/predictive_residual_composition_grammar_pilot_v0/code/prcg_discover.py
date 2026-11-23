"""PRCG-v0 stage 1 driver: learn the two merge grammars from D0 Fit only.

Writes ``grammar/predictive_grammar.json``, ``grammar/frequency_grammar.json``
(exactly as many rules as the predictive grammar accepted),
``grammar/discovery_trace.json`` and ``grammar/stats.json``.  If the predictive
grammar cannot accept a single legal new rule the round stops with
``NO_TRANSFERABLE_PREDICTIVE_MERGE`` and nothing else is run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import prcg_common as C
import prcg_data as D
import prcg_grammar as GD
import prcg_graph as G


def rules_to_grammar(payload: Sequence[dict[str, Any]]) -> G.Grammar:
    rules = [G.Rule(index=int(r["index"]), left_cert=str(r["left_cert"]),
                    right_cert=str(r["right_cert"]), pattern=str(r["pattern"]),
                    parent_cert=str(r["parent_cert"]), parent_size=int(r["parent_size"]),
                    n_crossing=int(r["n_crossing_bonds"]), origin=str(r["origin"]))
             for r in payload]
    return G.Grammar(rules)


def load_grammar(path: Path) -> G.Grammar:
    payload = C.read_json(path)
    return rules_to_grammar(payload["rules"])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.GRAMMAR_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = C.load_config()
    gcfg = dict(cfg["grammar"])
    bundle = D.load_bundle()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    all_rows = np.concatenate([fit_rows, dev_rows])
    molecules = D.build_molecules(bundle.raw, all_rows)
    y_std = bundle.labels_std()
    disc = GD.GrammarDiscovery(
        molecules, fit_rows, y_std[fit_rows], bundle.graph_scalars[fit_rows],
        bundle.scaffold_by_row, log=lambda m: print(m, flush=True))
    pred_grammar, pred_steps = disc.run("predictive", max_rules=int(gcfg["max_rules"]))
    if len(pred_grammar) == 0:
        summary = {"verdict": "NO_TRANSFERABLE_PREDICTIVE_MERGE",
                   "n_rules": 0, "wall_seconds": time.time() - t0,
                   "data_hashes": bundle.hashes,
                   "fold_sizes": np.bincount(disc.fold).tolist()}
        C.write_json(out_dir / "summary.json", summary)
        print(json.dumps(C.jsonable(summary), indent=1))
        return 2
    freq_grammar, freq_steps = disc.run("frequency", max_rules=int(gcfg["max_rules"]),
                                        n_rules=len(pred_grammar))
    stats = {}
    for name, grammar in (("predictive", pred_grammar), ("frequency", freq_grammar)):
        stats[name] = {
            "fit": GD.grammar_stats(molecules, fit_rows, grammar, bundle.scaffold_by_row),
            "dev": GD.grammar_stats(molecules, dev_rows, grammar, bundle.scaffold_by_row),
            "rules": grammar.to_dict(),
        }
        for side in ("fit", "dev"):
            stats[name][side].pop("type_support", None)
            stats[name][side].pop("type_scaffold_support", None)
            stats[name][side].pop("type_size", None)
    payload = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "n_rules": len(pred_grammar),
        "fold_sizes": [int(x) for x in np.bincount(disc.fold).tolist()],
        "fold_scaffold_counts": [
            len({disc.scaffold[r] for i, r in enumerate(fit_rows.tolist())
                 if disc.fold[i] == side}) for side in (0, 1)],
        "settings": gcfg,
        "data_hashes": bundle.hashes,
        "target_scaler": bundle.scaler,
        "fit_rows_sha256": C.array_sha256(fit_rows),
        "dev_rows_sha256": C.array_sha256(dev_rows),
        "wall_seconds": time.time() - t0,
    }
    C.write_json(out_dir / "predictive_grammar.json",
                 {"rules": pred_grammar.to_dict(), **payload, "origin": "predictive"})
    C.write_json(out_dir / "frequency_grammar.json",
                 {"rules": freq_grammar.to_dict(), **payload, "origin": "frequency"})
    C.write_json(out_dir / "discovery_trace.json",
                 {"predictive": pred_steps, "frequency": freq_steps})
    C.write_json(out_dir / "stats.json", stats)
    print(json.dumps({"n_rules": len(pred_grammar), "wall_seconds": payload["wall_seconds"],
                      "predictive_sizes": [s["rule"]["parent_size"] for s in pred_steps],
                      "frequency_sizes": [s["rule"]["parent_size"] for s in freq_steps]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())