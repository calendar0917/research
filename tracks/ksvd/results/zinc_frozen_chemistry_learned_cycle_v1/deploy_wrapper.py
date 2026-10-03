"""Deployable wrapper for zinc-frozen-chemistry-learned-cycle-v1.

Real inference = frozen Full ``O`` branch ``h(x)`` + learned cycle head ``q(T)``
+ fit-only calibration ``b_P``:

    p(x) = h(x) + q(T) + b_P

``forward(batch, topo25=None)`` accepts existing graph data and the existing
Full ``topology25`` model input.  It **never** receives or looks up ``c``, ``g``,
``k``, ``y``, group id or molecule id label tables.

Usage
-----
    from tracks.ksvd.results.zinc_frozen_chemistry_learned_cycle_v1.deploy_wrapper import load_predictor
    predictor = load_predictor(".../deploy_bundle_seed0.pt")
    p = predictor(batch, topo25)

``topo25`` must be the existing Full topology25 model input (same 8000-fit
standardization as the frozen O branch).  ``C6_MASK`` and the frozen full
architecture come from ``e2e_dictenv_clean_mechanism_v1`` / the frozen runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracks.ksvd.experiments.luyin16.zinc_frozen_chemistry_learned_cycle_v1 import (  # noqa: E402
    LearnedCyclePredictor,
    load_predictor as _load_predictor,
    load_prep_and_targets,
)

__all__ = ["LearnedCyclePredictor", "load_predictor"]


def load_predictor(bundle_path: str | Path) -> LearnedCyclePredictor:
    blob, _ = load_prep_and_targets()
    return _load_predictor(Path(bundle_path), blob)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the deployable frozen-chemistry learned-cycle predictor")
    parser.add_argument("bundle", help="path to deploy_bundle_seed{0,1}.pt")
    parser.add_argument("--print-bias", action="store_true")
    args = parser.parse_args()
    predictor = load_predictor(args.bundle)
    print(f"loaded predictor: b_P={predictor.b_P:.8f}")
    if args.print_bias:
        print(f"head parameters: {sum(p.numel() for p in predictor.head.parameters())}")
