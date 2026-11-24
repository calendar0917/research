"""Run the two analysis blocks and write the single round summary."""

from __future__ import annotations

import json
import time

import sed_common as C

import pcqm_response
import ses_correction


def main() -> dict:
    t0 = time.perf_counter()
    summary = {
        "round": "structural_effect_diagnosis_v0",
        "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "scope": (
            "Numerical diagnosis only: SES selection erratum + multi-candidate "
            "complementarity, and frozen W3 response transfer to SFEC v1 Valid "
            "replacements. No training, no fine-tuning, no hyper-parameter search."
        ),
        "ses": ses_correction.run(),
        "pcqm": pcqm_response.run(),
        "wall_seconds": float(time.perf_counter() - t0),
        "peak_rss_gb": C.peak_rss_gb(),
    }
    C.write_json(C.ROUND_DIR / "summary.json", summary)
    print(json.dumps({
        "status": summary["pcqm"].get("status"),
        "wall_seconds": summary["wall_seconds"],
        "peak_rss_gb": summary["peak_rss_gb"],
        "ses_erratum": {
            s: summary["ses"]["erratum"]["splits"][s]["contrasts"]["A_vs_B_correct"]
            for s in ("valid", "test")
        },
        "pcqm_all_evaluable": summary["pcqm"].get("subsets", {}).get("all_evaluable", {}).get("models"),
    }, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
