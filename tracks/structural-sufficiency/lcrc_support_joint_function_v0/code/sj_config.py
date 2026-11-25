"""Write / verify the frozen experiment configuration."""

from __future__ import annotations

import argparse
import json

import sj_common as C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="freeze / verify configs/experiment.json")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    path = C.CONFIGS / "experiment.json"
    payload = C.config_payload()
    if args.write:
        payload["reuse"] = {
            k: {
                "round": v["round"],
                "run_dir": C.track_rel(v["run_dir"]),
                "checkpoint_sha256": C.sha256_file(v["checkpoint"]),
                "expected": v["expected"],
            }
            for k, v in C.REUSE.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {C.track_rel(path)} sha256={C.sha256_file(path)}")
        return 0
    frozen = C.read_json(path)
    same = C.stable_json(frozen) == C.stable_json(payload)
    print(C.stable_json({"frozen_matches_code": same, "path": C.track_rel(path),
                         "sha256": C.sha256_file(path)}))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
