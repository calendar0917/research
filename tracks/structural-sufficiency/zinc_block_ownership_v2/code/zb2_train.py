"""Formal-run driver for ``zinc_block_ownership_v2``.

The training loop, model, blocks, collate, recipe and selection rule come from
the frozen v1 modules and are executed unchanged (``zbo_train.train``).  This
wrapper only:

1. re-verifies that the v1 frozen inputs (config, gates, code, caches) are
   byte-identical to what v2 pre-registered;
2. verifies v2's own Phase A + gate fingerprints and that the requested
   (arm, seed) is authorised by ``configs/zb2_v1.json`` -- Phase C additionally
   requires ``analysis/summary.json`` to carry ``proceed_to_phase_c: true``;
3. redirects the output directory to ``v2/runs``;
4. writes an additive ``zb2.json`` next to v1's untouched ``train.json``.

Usage:
    python zb2_train.py --arm S --seed 0 --device cuda:0        # Phase B
    python zb2_train.py --arm R --seed 1 --device cuda:0        # Phase C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zb2_common as Z  # noqa: E402

BC = Z.BC
ZT = Z.ZT


def _load_config() -> dict:
    if not Z.CONFIG.exists():
        raise SystemExit(f"missing v2 config {Z.CONFIG}")
    cfg = Z.read_json(Z.CONFIG)
    if not cfg.get("cpu_gates", {}).get("all_pass", False):
        raise SystemExit("v2 CPU gates did not pass; S arm is not authorised")
    if not cfg.get("s_arm_authorized", False):
        raise SystemExit("v2 config does not authorise the S arm")
    for key in ("phase_a", "cpu_gates"):
        p = Z.REPO / cfg[key]["path"]
        got = BC.sha256_file(p)
        if got != cfg[key]["sha256"]:
            raise SystemExit(f"{key} artifact changed after the freeze: {got} != "
                             f"{cfg[key]['sha256']}")
    return cfg


def _authorised(cfg: dict, arm: str, seed: int) -> dict:
    plan_b = {(int(r["seed"]), str(r["arm"])) for r in cfg["run_plan"]["phase_b"]}
    plan_c = {(int(r["seed"]), str(r["arm"]))
              for r in cfg["run_plan"]["phase_c_iff_phase_b_proceed"]}
    key = (int(seed), str(arm))
    if key in plan_b:
        return {"phase": "B", "reason": "pre-registered Phase B purchase"}
    if key in plan_c:
        summary = Z.ANALYSIS / "summary.json"
        if not summary.exists():
            raise SystemExit("Phase C requires analysis/summary.json from Phase B")
        s = Z.read_json(summary)
        if not s.get("proceed_to_phase_c", False):
            raise SystemExit(f"Phase C is not authorised: phase B verdict "
                             f"{s.get('phase_b_label')!r} did not proceed")
        return {"phase": "C", "reason": "Phase B verdict authorised confirmation",
                "phase_b_label": s.get("phase_b_label")}
    raise SystemExit(f"({arm}, seed {seed}) is not in the pre-registered run plan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["R", "N", "S"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--total-updates", type=int, default=None)
    args = p.parse_args(argv)

    integrity = Z.require_v1_integrity()
    cfg = _load_config()
    auth = _authorised(cfg, args.arm, args.seed)
    tag = f"{args.arm}_s{args.seed}"
    Z.log(f"[zb2] arm={args.arm} seed={args.seed} tag={tag} phase={auth['phase']}")
    Z.log(f"[zb2] v1 integrity ok (config/gates/code/caches byte-identical); "
          f"out={BC.rel_or_abs(Z.RUNS / tag)}")

    total = args.total_updates
    if args.smoke and total is None:
        total = 79 * 2          # v1's smoke convention: two epochs
    info = ZT.train(args.arm, args.seed, args.device,
                    total_updates=total, smoke=args.smoke)

    out = (Z.RUNS_SMOKE if args.smoke else Z.RUNS) / tag
    Z.write_json(out / "zb2.json", Z.jsonable({
        "round": Z.ROUND_ID,
        "driver_round": Z.DRIVER_ROUND_ID,
        "tag": tag, "arm": args.arm, "seed": int(args.seed),
        "written": Z.now(),
        "official_test_loaded": False,
        "phase": auth,
        "v2_config": {"path": BC.rel_or_abs(Z.CONFIG),
                      "sha256": BC.sha256_file(Z.CONFIG)},
        "v1_integrity": integrity,
        "train_json": {"path": BC.rel_or_abs(out / "train.json"),
                       "sha256": BC.sha256_file(out / "train.json"),
                       "git_head": info.get("git_head"),
                       "git_dirty": info.get("git_dirty"),
                       "best_valid": info["best"]["valid_mae"],
                       "total_updates": info.get("total_updates"),
                       "parameters": info.get("parameters")},
        "note": ("train.json is exactly what the frozen v1 trainer wrote; this "
                 "file is additive provenance for the v2 round"),
    }))
    Z.log(f"[zb2] DONE {tag} bestValid={info['best']['valid_mae']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
