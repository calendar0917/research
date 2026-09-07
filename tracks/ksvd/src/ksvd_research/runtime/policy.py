"""Test-access policy: the control plane decides whether a run may touch an
official test split.

The scientific rule lives in the protocol (``test_policy: terminal``).  This
module is the single place that interprets that rule into a tri-state
``test_access`` outcome.  Runners never re-derive the policy: they only obey
``context.test_access``.

Semantics:

* ``protocol is None``                          -> ``unguarded`` (no rule)
* ``test_policy == terminal`` and mode != terminal -> ``blocked``
* ``test_policy == terminal`` and mode == terminal -> ``granted``
* ``test_policy == always``                     -> ``granted`` (legacy: test may
  be evaluated in any mode)
* any other non-empty ``test_policy``           -> hard error: unknown policy
  must never silently permit high-risk access
"""

from __future__ import annotations

from typing import Any, Mapping

BLOCKED = "blocked"
GRANTED = "granted"
UNGUARDED = "unguarded"

_NON_POLICY_MODES = "terminal"


def resolve_test_access(
    protocol: Mapping[str, Any] | None,
    mode: str,
) -> str:
    """Return ``blocked`` / ``granted`` / ``unguarded`` for one run."""
    if protocol is None:
        return UNGUARDED
    policy = protocol.get("test_policy")
    if policy is None:
        return UNGUARDED
    if policy == "terminal":
        return BLOCKED if mode != _NON_POLICY_MODES else GRANTED
    if policy == "always":
        return GRANTED
    raise ValueError(
        f"unknown test_policy {policy!r} in protocol; refusing to guess access rights"
    )


def is_blocked(test_access: str) -> bool:
    return test_access == BLOCKED
