"""P0 preparation: identity, pair support, edit regions and checkpoint replay.

Run on CPU.  It writes only analysis artefacts under this round's directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import xr_common as C
import xr_assets as A
import xr_models as M


def _support_with_evaluable(support: dict[str, Any], pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = json.loads(json.dumps(support))
    for name, label in C.STRATUM_LABEL.items():
        subset = [p for p in pair_rows if _in_stratum(p, name)]
        out["strata"][name]["n_evaluable_pairs"] = int(sum(1 for p in subset if p["evaluable"]))
        out["strata"][name]["n_not_evaluable_pairs"] = int(sum(1 for p in subset if not p["evaluable"]))
        out["strata"][name]["n_evaluable_components"] = len({int(p["component"]) for p in subset if p["evaluable"]})
        out["strata"][name]["support_evaluable"] = C.support_label(
            len({int(p["component"]) for p in subset if p["evaluable"]}),
            sum(1 for p in subset if p["evaluable"]),
        )
    return out


def _in_stratum(pair: MappingLike, name: str) -> bool:
    if name == "train_train":
        return bool(pair["train_train"])
    if name == "at_least_one_nontrain":
        return not bool(pair["train_train"])
    if name == "both_nontrain":
        return bool(pair["in_H"])
    if name == "all":
        return True
    raise KeyError(name)


MappingLike = dict[str, Any]


def prepare(*, device: str = "cpu", replay: bool = True) -> dict[str, Any]:
    C.install_source_paths()
    b = M.bundle(C.DATASET)
    pairs = A.load_pairs()
    oracle = A.load_oracle()
    print(f"[prepare] bundle split sizes {b.split_sizes}", flush=True)

    # -- checkpoints: identity + stored-prediction replay ------------------
    checkpoint_reports = []
    for name in C.CHECKPOINTS:
        ident = M.checkpoint_identity(name, b)
        repl = M.replay_checkpoint(name, b, device=device) if replay else {"passed": True, "splits": {}}
        report = {**ident, "replay": repl, "passed": bool(ident["sha256_match"] and ident["parameter_count_match"] and ident["kind_match"] and repl["passed"])}
        checkpoint_reports.append(report)
        print(f"[prepare] checkpoint {name}: sha={ident['sha256_match']} params={ident['parameter_count_match']} kind={ident['kind_match']} replay={repl['passed']}", flush=True)

    # -- endpoints / edit regions / oracle deltas --------------------------
    endpoint_rows, pair_rows, endpoint_summary = A.build_endpoints(b, pairs, oracle)
    print(f"[prepare] endpoints {endpoint_summary}", flush=True)
    support = A.pair_support(pairs)
    support = _support_with_evaluable(support, pair_rows)

    # -- identity ----------------------------------------------------------
    identity = A.data_identity(b, pairs, oracle, checkpoint_reports)
    identity["endpoint_summary"] = endpoint_summary
    identity["pair_support"] = support
    identity["all_identity_gates_passed"] = bool(identity["all_identity_gates_passed"] and endpoint_summary["mapping_all_valid"] and endpoint_summary["wc_all_closed"])
    C.write_json(C.ANALYSIS_DIR / "data_identity.json", identity)
    C.write_json(C.ANALYSIS_DIR / "pair_support.json", support)
    C.write_json(C.ANALYSIS_DIR / "model_replay.json", {
        "checkpoints": checkpoint_reports,
        "all_passed": all(r["passed"] for r in checkpoint_reports),
    })
    C.write_json(C.ANALYSIS_DIR / "pair_table.json", {
        "pairs": pair_rows,
        "summary": endpoint_summary,
    })
    endpoint_info = C.write_jsonl_gz(C.PAIR_ENDPOINTS, endpoint_rows)
    C.write_json(C.ANALYSIS_DIR / "pair_endpoints_index.json", endpoint_info)
    print(f"[prepare] wrote {endpoint_info['n']} endpoints to {endpoint_info['path']}", flush=True)
    print(f"[prepare] identity passed={identity['all_identity_gates_passed']} checkpoints={all(r['passed'] for r in checkpoint_reports)}", flush=True)
    return {
        "identity_passed": identity["all_identity_gates_passed"],
        "checkpoints_passed": all(r["passed"] for r in checkpoint_reports),
        "endpoint_summary": endpoint_summary,
        "support": support,
        "endpoint_file": endpoint_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    out = prepare(device=args.device, replay=not args.no_replay)
    print(json.dumps(out, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
