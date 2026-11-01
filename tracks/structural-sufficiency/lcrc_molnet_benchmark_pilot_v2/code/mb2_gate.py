"""The Test-access gate: which module bytes may and may not block a Test read.

This lives outside ``mb2_common.py`` on purpose.  ``mb2_common.py`` was already
frozen and hashed by the training runs, so adding a constant to it would have
made the *training-time* and *freeze-time* bytes of a scoring-critical module
differ, and the gate would have had to accept an explanation instead of a proof.
Keeping the gate policy in its own module means:

* ``SCORING_CRITICAL_MODULES`` is not itself scoring-critical, so editing it can
  never invalidate a checkpoint;
* the assertions in ``mb2_freeze``/``mb2_eval`` can require **zero** drift in the
  modules that actually produce a score.

The set covers everything that decides which rows exist, which graph is built,
which parameters are trained, which Validation value is frozen, and which
checkpoint is written.
"""

from __future__ import annotations

from typing import Any

import mb2_common as C

#: Modules whose bytes determine the checkpoint -> metric chain.
SCORING_CRITICAL_MODULES: tuple[str, ...] = (
    "v2/mb2_common.py",
    "v2/mb2_source.py",
    "v2/mb2_rdkit.py",
    "v2/mb2_data.py",
    "v2/mb2_model.py",
    "v2/mb2_metrics.py",
    "v2/mb2_train.py",
    "prep/adapters.py",
    "prep/feature_spec.py",
    "prep/lcrc_common.py",
    "prep/lcrc_data.py",
    "prep/lcrc_model.py",
    "prep/molecular_graph.py",
    "v1/mb_rdkit.py",
)


def scoring_critical_hashes() -> dict[str, str]:
    """The subset of ``mb2_common.code_hashes()`` that the Test gate cares about."""
    everything = C.code_hashes()
    missing = [k for k in SCORING_CRITICAL_MODULES if k not in everything]
    if missing:
        raise KeyError(f"scoring-critical modules missing from code_hashes(): {missing}")
    return {k: everything[k] for k in SCORING_CRITICAL_MODULES}


def drift(critical_snapshot: dict[str, Any], *, label: str) -> dict[str, tuple[Any, Any]]:
    """Modules whose current bytes differ from the snapshot (``{name: (old, new)}``)."""
    current = scoring_critical_hashes()
    out = {
        k: (critical_snapshot.get(k), v)
        for k, v in current.items()
        if critical_snapshot.get(k) not in (None, v)
    }
    if out:
        C.log(f"{label}: scoring-critical drift in {sorted(out)}")
    return out


__all__ = ["SCORING_CRITICAL_MODULES", "scoring_critical_hashes", "drift"]
