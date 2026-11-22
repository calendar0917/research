"""Normalise the analysis JSON deliverables to strict JSON.

Non-finite floats are genuine "undefined" values (a zero-variance term has no
Pearson correlation / eta-squared, and a disabled relation gate has no gate
value).  They are stored as ``null`` so the deliverables can be read by any
strict JSON parser.  Raw per-run ``manifest.json`` files produced by the
training script are left untouched as execution artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import acra_common as C


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(log=print) -> list[str]:
    targets = sorted(C.analysis_dir().glob("*.json"))
    targets.append(C.results_root() / "summary.json")
    written = []
    for path in targets:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        text = json.dumps(_clean(payload), indent=2)
        if "NaN" in text or "Infinity" in text:  # pragma: no cover - guard
            raise ValueError(f"{path} still contains a non-finite literal")
        path.write_text(text + "\n", encoding="utf-8")
        written.append(str(path.relative_to(C.ROUND_DIR)))
    log(f"normalised {len(written)} JSON files: {', '.join(written)}")
    return written


if __name__ == "__main__":
    run()