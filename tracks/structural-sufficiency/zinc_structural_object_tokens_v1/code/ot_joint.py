"""A_joint_BF: the exact P3-BF graph, trained jointly from the train-only SSL start.

Nothing about the model changes relative to ``zinc_branching_fragment_v1``:
the same S3/S4/F4 typed rooted objects, complete typed induced adjacency, root
marker, multiplicity, exact empty set, phi (98 -> 64 -> 32) and the same
198-dim per-root statistics entering the P3 atom encoder as a zero-initialised
second linear term (``w_s``).  Total parameters stay 447,505.

What changes is the *initialisation* and the *training*:

1. ``torch.manual_seed(seed)`` then the P3 core modules (``P3Encoders``,
   ``group_composer``, ``reader``) are constructed in the historical order;
2. only the seed-matched **train-only masked-SSL** atom/bond/junction encoders
   are loaded from ``pretrain_encoder_s0.pt`` (sha256 checked);
3. then the new BF modules (``w_s`` zero-initialised + ``branch``) are appended;
4. every parameter is trained with one peak LR (recipe B), full 600 epochs.

The supervised P3B ``best.pt``/``final.pt`` and the old BF epoch-100 state are
never loaded.  The initial state therefore matches the *SSL-start* P3, not the
supervised P3B.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
_V2 = _SS / "zinc_recipe_trunk_v2" / "code"
_PERF = _SS / "zinc_perf_protocol_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF), str(_V2), str(_PERF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import bf_models as BFM  # noqa: E402

SSL_SHA256 = "006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97"
EXPECTED_PARAMETERS = 447_505
P3_INHERITED = ("encoders.atom_encoder", "encoders.bond_encoder",
                "encoders.junction_encoder", "group_composer", "reader")


def load_ssl_encoders(model, device="cpu") -> dict:
    """Load the seed-0 train-only masked-SSL encoders (and nothing else)."""
    path = C.P3_SSL_S0
    if not path.exists():
        raise FileNotFoundError(
            f"missing train-only SSL encoder {path}; refusing a random fallback"
        )
    sha = C.sha256_file(path)
    if sha != SSL_SHA256:
        raise RuntimeError(f"SSL encoder sha256 {sha} != {SSL_SHA256}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if sorted(state.keys()) != ["atom_encoder", "bond_encoder", "junction_encoder"]:
        raise RuntimeError(f"unexpected SSL encoder keys: {sorted(state.keys())}")
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        getattr(model.encoders, name).load_state_dict(
            {k: v.to(device) for k, v in state[name].items()}, strict=True
        )
    return {
        "source": str(path.relative_to(C.REPO)),
        "sha256": sha,
        "loaded": sorted(state.keys()),
        "note": "atom/bond/junction encoders only; composer and reader keep the seed-0 init",
    }


def build_model_a(norm: Mapping[str, object], seed: int = 0, device="cpu"):
    """Construct P3 (seed order) -> load SSL encoders -> append BF modules."""
    torch.manual_seed(int(seed))
    model = BFM.BranchingFragmentModel(norm)
    ssl = load_ssl_encoders(model, device="cpu")
    model = model.to(device)
    total = C.parameter_count(model)
    if total != EXPECTED_PARAMETERS:
        raise RuntimeError(f"A has {total} parameters != {EXPECTED_PARAMETERS}")
    info = {
        "candidate": "A_joint_BF",
        "seed": int(seed),
        "parameters": total,
        "parameters_breakdown": BFM.parameter_breakdown(model),
        "ssl_encoders": ssl,
        "init_order": [
            "torch.manual_seed(seed)",
            "P3Encoders + group_composer + reader (historical order)",
            "load train-only SSL atom/bond/junction encoders",
            "append w_s (zeros) + BF branch",
        ],
        "w_s_zero": bool(model.w_s.weight.abs().sum().item() == 0.0),
        "all_require_grad": bool(all(p.requires_grad for p in model.parameters())),
    }
    return model, info


def p3_core_state(model) -> dict:
    """The 24 P3 tensors with the ``encoders.`` prefix stripped."""
    out = {}
    for key, value in model.state_dict().items():
        if key.startswith("encoders."):
            out[key[len("encoders."):]] = value
        elif key.startswith(("group_composer.", "reader.")):
            out[key] = value
        elif key in ("global_mean", "global_scale"):
            out[key] = value
    return out


def historical_p3_ssl_init() -> dict:
    """Historical P3 builder: ``build_arm("B", seed 0)`` + ``load_encoder``."""
    import zpp_p3  # noqa: F401  (registers import side effects of the old round)
    import esr_models as E

    prep = C.read_json(C.P3_PREP)
    bjoc = E.build_arm("B", 0, prep, None, 0.0, torch.device("cpu"))
    E.load_encoder(bjoc, 0, C.P3_SSL_S0, torch.device("cpu"))
    return bjoc.state_dict()
