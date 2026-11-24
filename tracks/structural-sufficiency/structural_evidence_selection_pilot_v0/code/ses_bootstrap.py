"""One-shot pre-training bootstrap: prepare -> checks -> throughput preflight.

The prepared candidate artifact is shared by the three formal arms.  The
throughput preflight builds and discards its own models; formal runs are
separate processes.  Any failed check or infeasible 60-epoch horizon stops here.
"""

from __future__ import annotations

import json

import ses_prepare
import ses_checks
import ses_preflight


def main() -> dict:
    prepare_stats = ses_prepare.main()
    checks = ses_checks.main()
    if not checks["all_pass"]:
        raise SystemExit("SES bootstrap stopped: checks failed")
    preflight = ses_preflight.main()
    if not preflight["feasible"]:
        raise SystemExit("SES bootstrap stopped: even 60 epochs exceeds the budget")
    payload = {
        "prepare": prepare_stats,
        "checks_all_pass": bool(checks["all_pass"]),
        "n_checks": int(checks["n_checks"]),
        "preflight_decision": preflight["decision"],
        "preflight_feasible": bool(preflight["feasible"]),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
