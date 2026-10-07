"""ZINC CSSD bridge compression v1: freeze the verified CSSD basis and its
effective local access, and test whether a SMALLER-capacity neural consumer
(a 144->72->144 MLP bridge in place of the historical 144->288->144) improves
the DICT consumer's development-set performance while keeping full-y
performance and the sparse-code readout.

Round: ``zinc_cssd_bridge_compression_v1``.

Frozen object of this round = the DICT arm of
``zinc_cssd_consumer_replacement_v1`` (the 297,539-parameter M_COMP consumer
whose local phi65 input is the frozen-CSSD full reconstruction), trained for
240 epochs under the historical recipe.  NOTHING about the basis, the access
path, the fold, the targets, the Q head, the local encoder/reader or the
training recipe changes.  The single purchased question: the posterior MLP
bridge (the only consumer block with an independently constrainable MLP of
82,944 parameters) may have too many degrees of freedom for 8,001 fit
molecules; does halving its hidden width improve the g generalization on the
historical development rows?

Two arms, both DICT consumers, both trained from scratch (seeds 0/1):

* ``CTRL288`` — repl.build_arm("DICT", ...) untouched: the historical
  144->288->144 bridge (82,944 bridge parameters, 297,539 total), original
  factory initialization (reproduces the historical DICT init bit-for-bit).
* ``B72`` — the SAME factory first (original audit inside repl.build_arm),
  then ONLY ``model.local_dictionary_bridge`` replaced by
  :class:`SmallMLPBridge` (144->72->144, no biases, SiLU, the EXACT historical
  forward formula, 20,736 bridge parameters, 235,331 total = -62,208,
  ~20.9%): built on CPU under ``torch.random.fork_rng(devices=[])`` with the
  private seed ``20261007 + body_seed`` and the two ``nn.Linear`` default
  ``reset_parameters`` initializations (teacher-free; no distillation, no
  candidate search).  The global CPU RNG is restored by the fork; the formal
  training re-seeds via ``seed_everything(body_seed)`` exactly like the
  control, so same-seed arms share the batch schedule and the training RNG
  stream, and every NON-bridge initial tensor is bit-identical across arms.
  Because the hidden width changes necessarily define a new initialization,
  the round compares the B72 COMPLETE recipe (never isolating "fewer
  parameters" as the cause, never claiming step-0 function equality).

Single estimator per run: the equal-weight FP32 mean of the full member
states of epochs 236..240 (the historical soup semantics).  Terminal: one
frozen comparison — full dev y_raw MAE (primary) / g_raw MAE (secondary),
per-row exports on fit+dev, canonical-SMILES group-paired bootstrap (2000
draws, seed 20261007, shared picks, seeds averaged first, separately for
Delta_y and Delta_g), the noise bound eta from repeated forwards +
same-weight save/reload replay (NO raw/decoded identity hook this round),
and the unique alpha-mean dictionary intervention on ALL FOUR soups.
Decision: the frozen engineering branches
keep-performance-candidate / keep-cheaper-candidate /
dictionary-readout-not-supported / inconclusive / close-b72 (performance and
cost are judged SEPARATELY; the previous round's -0.003 cliff gate is NOT
inherited).  Official valid/test are never instantiated; dev is the
historical development comparison set (a development set repeatedly used,
never an independent confirm).

Usage (local CPU for source/pretrain checks; res-2 res2-cu124 for the GPU
stages, all through the registered runner)::

    python -m tracks.ksvd.experiments.luyin16.\\
zinc_cssd_bridge_compression_v1 --stage source-checks
    ... --stage pretrain-checks
    ... --stage smoke --device cuda:0
    ... --stage train --arm CTRL288 --seed 0 --device cuda:0
    ... --stage train --arm B72 --seed 1 --device cuda:0
    ... --stage terminal-eval --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import (
    e2e_dictenv_latent_bridge_v1 as lb,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_basis_reuse_v1 as zreuse,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_consumer_generalization_v1 as gen,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_consumer_replacement_v1 as repl,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_nonlinear_binding_v1 as nb,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_chemistry_component_supervision_seed0_v1 as zcs,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_full_cycle_target_decomposition_v1 as zftd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

PROTOCOL_VERSION = "zinc-cssd-bridge-compression-v1"
RESULT_SLUG = "zinc_cssd_bridge_compression_v1"
TRACK_ROOT = repl.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG

#: read-only stage-0 source of every frozen object (unchanged)
SOURCE_DIR = repl.SOURCE_DIR

# ---- the two arms of this round (both are DICT consumers) -----------------
ARMS = ("CTRL288", "B72")
CONTROL = "CTRL288"
CANDIDATE = "B72"
SEEDS = (0, 1)

# ---- frozen recipe (identical for both arms / all four runs) --------------
EPOCHS = repl.EPOCHS                    # 240
LR = repl.LR                            # 1e-3
WEIGHT_DECAY = repl.WEIGHT_DECAY        # 1e-5 (coupled)
GRAD_CLIP = repl.GRAD_CLIP              # 5.0
BATCH_SIZE = repl.BATCH_SIZE            # 128
SOUP_EPOCHS = repl.SOUP_EPOCHS          # 236..240 fixed five-epoch soup
LOG_EPOCHS = repl.LOG_EPOCHS            # (1, 40, 120, 240)
TRAIN_SHUFFLE_OFFSET = repl.TRAIN_SHUFFLE_OFFSET  # 101
COMPONENT_LOSS_WEIGHT = repl.COMPONENT_LOSS_WEIGHT  # 0.5
#: 63 steps/epoch * 240 epochs (asserted, never assumed)
STEPS_EXPECTED = 15120

# ---- parameter contracts (per arm, this round) -----------------------------
BRIDGE_IN_OUT = int(zw.BRIDGE_DIM)       # 144
BRIDGE_HIDDEN_B72 = 72
B72_BRIDGE_PARAMETERS = 2 * BRIDGE_IN_OUT * BRIDGE_HIDDEN_B72   # 20,736
B72_TOTAL_PARAMETERS = (
    int(repl.EXPECTED_PARAMETERS) - int(repl.BRIDGE_PARAMETERS) + B72_BRIDGE_PARAMETERS
)                                        # 235,331
B72_PARAMETER_REDUCTION = int(repl.BRIDGE_PARAMETERS) - B72_BRIDGE_PARAMETERS  # 62,208

#: B72 bridge private init seed (fork_rng on CPU; teacher-free default init)
B72_INIT_SEED_BASE = 20261007

# ---- terminal (frozen) ------------------------------------------------------
N_BOOT = 2000
BOOT_SEED = 20261007
#: response marker floor (identical semantics to the source rounds)
REPLAY_TOL = 1e-4
#: cross-device FP32 replay tolerance (same-weight reload / repeat)
CROSS_DEVICE_TOL = 1e-5
#: exported float64 component-sum rounding tolerance (torch identity is exact)
FLOAT64_SUM_TOL = gen.FLOAT64_SUM_TOL

# ---- frozen decision thresholds (THIS round's engineering numbers; the
# previous round's mean -0.003 cliff gate is deliberately NOT inherited;
# performance and parameter cost are judged separately) ----------------------
A_SEED_DELTA_Y = 0.0            # per-seed Delta_y must be < 0
A_SEED_DELTA_G_BAND = 1e-4       # per-seed Delta_g may not exceed +1e-4
A_MEAN_DELTA_Y = -1e-4           # two-seed mean Delta_y must be <= -1e-4
A_MEAN_DELTA_G = 0.0             # two-seed mean Delta_g must be < 0
A_CI_UPPER = 0.0                 # mean-Delta_y CI95 upper bound must be < 0
B_SEED_TOL = 2e-3                # per-seed |Delta| tolerance for the cheaper arm
B_MEAN_TOL = 1e-3                # two-seed mean tolerance
B_CI_UPPER_TOL = 2e-3            # mean-Delta CI95 upper bound tolerance
D_CI_SPAN_LO = -1e-4             # inconclusive CI-span floor (A territory)
D_CI_SPAN_HI = 2e-3              # inconclusive CI-span ceiling (beyond B)
E_FIT_BAND = 1e-4                # fit-difference band for the close-b72 sub-reading

write_json = repl.write_json
read_json = repl.read_json
file_sha256 = repl.file_sha256
array_sha256 = repl.array_sha256
state_hash = repl.state_hash
seed_everything = repl.seed_everything
resolve_device = repl.resolve_device


# ---------------------------------------------------------------------------
# 0. the compressed bridge + this round's per-arm factory
# ---------------------------------------------------------------------------

class SmallMLPBridge(nn.Module):
    """144->72->144 MLP bridge: the EXACT historical forward formula.

    ``scale = sqrt(mean(h^2, dim=1, keepdim=True) + BRIDGE_EPS)``,
    ``x = h / scale``, ``out = scale * fc2(silu(fc1(x)))`` — no biases, no
    residual, no dropout/LayerNorm, the same epsilon as the historical
    bridge.  The two ``nn.Linear`` layers keep their DEFAULT
    ``reset_parameters`` initializations; this class never copies trained
    weights, soup states or target-derived frames (teacher-free by design).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(BRIDGE_IN_OUT, BRIDGE_HIDDEN_B72, bias=False)
        self.fc2 = nn.Linear(BRIDGE_HIDDEN_B72, BRIDGE_IN_OUT, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
        x = h / scale
        return scale * self.fc2(F.silu(self.fc1(x)))


def arm_expected_audit(arm: str) -> dict[str, int]:
    """The per-arm parameter contract of this round (never the historical
    297,539 contract for the B72 arm — the CTRL288 arm keeps it)."""
    if arm == "CTRL288":
        return {
            "total_parameters": int(repl.EXPECTED_PARAMETERS),        # 297,539
            "base_body_parameters": int(repl.EXPECTED_BODY_PARAMETERS),  # 184,707
            "bridge_parameters": int(repl.BRIDGE_PARAMETERS),          # 82,944
            "local_tuple_parameters": int(repl.LOCAL_PARAMETERS),      # 29,888
            "reader_output_parameters": int(repl.READER_OUTPUT_PARAMETERS),  # 80
        }
    if arm == "B72":
        return {
            "total_parameters": B72_TOTAL_PARAMETERS,                  # 235,331
            "base_body_parameters": int(repl.EXPECTED_BODY_PARAMETERS),  # 184,707
            "bridge_parameters": B72_BRIDGE_PARAMETERS,                # 20,736
            "local_tuple_parameters": int(repl.LOCAL_PARAMETERS),      # 29,888
            "reader_output_parameters": int(repl.READER_OUTPUT_PARAMETERS),  # 80
        }
    raise ValueError(arm)


def arm_parameter_audit(model: nn.Module) -> dict[str, int]:
    """Generic per-arm audit (same counters as the historical rounds; the
    bridge count is prefix-based, so it fits either bridge width)."""
    return zcs.component_parameter_audit(model)


def build_arm_round(
    arm: str,
    payload: prev.TuplePayload,
    kappa_M: float,
    basis_parts: Mapping[str, torch.Tensor],
    seed: int,
    *,
    decode: bool = True,
) -> nn.Module:
    """Fresh untrained arm of THIS round.

    The original DICT factory always runs FIRST and completes its own
    historical 297,539-parameter audit inside ``repl.build_arm`` (this round
    never feeds B72 into a hardcoded 297,539 / DICT-arm historical check).
    ``B72`` then replaces ONLY ``model.local_dictionary_bridge`` with a
    freshly initialized :class:`SmallMLPBridge`, built on CPU under
    ``torch.random.fork_rng(devices=[])`` with the private seed
    ``20261007 + body_seed``; the fork restores the global CPU RNG, so the
    post-construction global RNG state is identical across arms for the same
    seed (every non-bridge initial tensor is bit-identical by construction).
    The per-arm parameter contract of THIS round is asserted afterwards.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    model = repl.build_arm("DICT", payload, kappa_M, basis_parts, int(seed), decode=decode)
    if arm == "B72":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(B72_INIT_SEED_BASE + int(seed))
            model.local_dictionary_bridge = SmallMLPBridge()
    audit = arm_parameter_audit(model)
    expected = arm_expected_audit(arm)
    if audit != expected:
        raise RuntimeError(f"{arm} parameter audit failed: {audit} != {expected}")
    if arm == "CTRL288" and not isinstance(model.local_dictionary_bridge, zw.MLPBridge):
        raise RuntimeError("CTRL288 posterior bridge is not the matched historical MLP bridge")
    if arm == "B72" and not isinstance(model.local_dictionary_bridge, SmallMLPBridge):
        raise RuntimeError("B72 posterior bridge is not the SmallMLPBridge")
    return model


def bridge_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k.startswith("local_dictionary_bridge.")}


def non_bridge_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if not k.startswith("local_dictionary_bridge.")}


def run_dir(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> Path:
    if arm not in ARMS:
        raise ValueError(arm)
    return Path(out_dir) / "runs" / f"{arm}_s{int(seed)}"


def load_run(arm: str, seed: int, out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """One run's manifest + soup state (hash-checked, correct per-arm factory
    downstream — never the historical DICT/CTRL288 factory for a B72 run)."""
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    soup_state = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(soup_state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"soup state hash mismatch for {arm} s{seed}")
    return {"manifest": manifest, "soup_state": soup_state, "run_dir": rdir}


# ---------------------------------------------------------------------------
# 1. stage: source-checks (CPU, read-only)
# ---------------------------------------------------------------------------

def source_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """Verify every reused object against the committed source manifests and
    the historical DICT runs; verify the per-arm contracts, the non-bridge
    init pairing and the RNG neutrality of the B72 bridge replacement."""
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    n_fit = int(fit_idx.size)

    # (a) every stage-0 artifact hash against the replacement round's
    #     committed source manifest (the read-only source of this line)
    src = read_json(repl.RESULTS_DIR / "source_manifest.json")
    artifact_hashes = {
        name: file_sha256(SOURCE_DIR / name) for name in src["reused_artifacts"]
    }
    if artifact_hashes != src["reused_artifacts"]:
        raise RuntimeError("stage-0 artifact hashes differ from the committed source manifest")
    if src["cssd_basis"]["U_sha256"] != array_sha256(ctx["basis"]["U"]) or \
        src["cssd_basis"]["D_sha256"] != array_sha256(ctx["basis"]["D"]):
        raise RuntimeError("CSSD basis hashes differ from the committed source manifest")
    if src["fold_view"]["fit"]["sha256"] != array_sha256(fit_idx, np.int64) or \
        src["fold_view"]["dev"]["sha256"] != array_sha256(dev_idx, np.int64):
        raise RuntimeError("round fold view differs from the committed source manifest")

    # (b) the historical DICT runs (read-only, hash-checked by gen.load_old_run)
    old = {s: gen.load_old_run(s) for s in SEEDS}

    # (c) schedule reproduction: this round's 240-epoch plan per seed must be
    #     bit-identical to the historical DICT schedule of the same seed
    prefix: dict[str, Any] = {}
    for s in SEEDS:
        schedule, schedule_hash = zw.build_schedule(n_fit, EPOCHS, int(s) + TRAIN_SHUFFLE_OFFSET)
        ok = all(
            np.array_equal(schedule[e], old[s]["schedule_order"][e]) for e in range(EPOCHS)
        )
        if not ok:
            raise RuntimeError(f"schedule does not reproduce the historical plan (seed {s})")
        if schedule_hash != old[s]["manifest"]["schedule_sha256"]:
            raise RuntimeError(f"schedule hash mismatch vs historical run (seed {s})")
        prefix[str(s)] = {
            "historical_schedule_sha256": old[s]["manifest"]["schedule_sha256"],
            "rebuilt_sha256": schedule_hash,
            "bit_identical_240": True,
        }

    # (d) init contracts: CTRL288 reproduces the historical DICT init
    #     bit-for-bit; B72 shares every non-bridge tensor with CTRL288 and
    #     its bridge is the fork_rng default init; the global CPU RNG state
    #     after construction is identical across arms (fork_rng neutrality)
    init_hashes: dict[str, Any] = {}
    for s in SEEDS:
        seed_everything(int(s))
        ctrl = build_arm_round("CTRL288", payload, kappa_M, basis_parts, int(s))
        ctrl_state = gen._capture_state(ctrl)
        ctrl_hash = state_hash(ctrl_state)
        if ctrl_hash != old[s]["manifest"]["init_state_sha256"]:
            raise RuntimeError(f"CTRL288 init does not reproduce the historical DICT init (seed {s})")
        rng_after_ctrl = torch.get_rng_state().clone()
        b72 = build_arm_round("B72", payload, kappa_M, basis_parts, int(s))
        rng_after_b72 = torch.get_rng_state().clone()
        if not torch.equal(rng_after_ctrl, rng_after_b72):
            raise RuntimeError(f"B72 construction disturbed the global CPU RNG (seed {s})")
        b72_state = gen._capture_state(b72)
        nb_ctrl, nb_b72 = non_bridge_state(ctrl_state), non_bridge_state(b72_state)
        if sorted(nb_ctrl) != sorted(nb_b72):
            raise RuntimeError(f"non-bridge state key sets differ (seed {s})")
        mismatched = [k for k in nb_ctrl if not torch.equal(nb_ctrl[k], nb_b72[k])]
        if mismatched:
            raise RuntimeError(f"non-bridge init tensors differ across arms at {mismatched} (seed {s})")
        br = bridge_state(b72_state)
        if sorted(br) != ["local_dictionary_bridge.fc1.weight", "local_dictionary_bridge.fc2.weight"]:
            raise RuntimeError(f"unexpected B72 bridge state keys {sorted(br)}")
        if tuple(br["local_dictionary_bridge.fc1.weight"].shape) != (BRIDGE_HIDDEN_B72, BRIDGE_IN_OUT) or \
            tuple(br["local_dictionary_bridge.fc2.weight"].shape) != (BRIDGE_IN_OUT, BRIDGE_HIDDEN_B72):
            raise RuntimeError(f"B72 bridge shapes wrong (seed {s})")
        init_hashes[str(s)] = {
            "ctrl288_init_sha256": ctrl_hash,
            "ctrl288_matches_historical": True,
            "b72_non_bridge_bitwise_equal_ctrl288": True,
            "b72_bridge_init_sha256": state_hash(br),
            "rng_stream_identical_after_construction": True,
        }
    # cross-seed: the B72 bridges genuinely differ; the CTRL bodies differ
    seed_everything(0)
    b72_0 = gen._capture_state(build_arm_round("B72", payload, kappa_M, basis_parts, 0))
    seed_everything(1)
    b72_1 = gen._capture_state(build_arm_round("B72", payload, kappa_M, basis_parts, 1))
    if state_hash(bridge_state(b72_0)) == state_hash(bridge_state(b72_1)):
        raise RuntimeError("B72 bridge init collides across seeds")
    if state_hash(non_bridge_state(b72_0)) == state_hash(non_bridge_state(b72_1)):
        raise RuntimeError("B72 non-bridge init collides across seeds")
    init_hashes["b72_bridge_seed0_ne_seed1"] = True
    init_hashes["b72_body_seed0_ne_seed1"] = True

    # (e) frozen basis / Q consistency with the historical runs
    for s in SEEDS:
        if old[s]["manifest"]["frozen_basis_hashes"] != basis_parts["hashes"]:
            raise RuntimeError(f"frozen basis hashes differ from the historical run (seed {s})")
        if state_hash(ctx["q_soup"]) != old[s]["manifest"]["q_soup_sha256"]:
            raise RuntimeError(f"Q soup hash differs from the historical run (seed {s})")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "source-checks",
        "source_dir": str(SOURCE_DIR.relative_to(TRACK_ROOT)),
        "source_read_only": True,
        "arms": list(ARMS),
        "fit_n": n_fit,
        "dev_n": int(dev_idx.size),
        "stage0_artifact_hashes_verified": True,
        "schedule_prefix": prefix,
        "init_contracts": init_hashes,
        "parameter_contracts": {arm: arm_expected_audit(arm) for arm in ARMS},
        "b72_parameter_reduction": B72_PARAMETER_REDUCTION,
        "frozen_basis_hashes": basis_parts["hashes"],
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "old_runs": {
            str(s): {
                "run_dir": str(old[s]["rdir"].relative_to(TRACK_ROOT)),
                "manifest_hashes_verified": True,
                "init_state_sha256": old[s]["manifest"]["init_state_sha256"],
                "schedule_sha256": old[s]["manifest"]["schedule_sha256"],
            }
            for s in SEEDS
        },
        "reuse_note": (
            "no basis/Q/prep/target refit; the historical DICT trajectories are "
            "read-only anchors; official valid/test never loaded"
        ),
        "all_passed": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "source_manifest.json", manifest)
    log(f"[source-checks] fit={n_fit} dev={dev_idx.size} verified in {manifest['seconds']:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# 2. stage: pretrain-checks (CPU, fit rows only — never dev, never a score)
# ---------------------------------------------------------------------------

def pretrain_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    """The meaningful pre-training checks of this round (fit-only).

    (a) bridge contract: shapes/no-bias/the exact forward formula and the
        teacher-free fork_rng init of SmallMLPBridge;
    (b) per-arm factories + non-bridge init pairing (restated from the source
        stage against fresh builds, both seeds);
    (c) small-batch fit-only optimization smoke: finite outputs/loss,
        gradients reaching BOTH bridge layers and the local W_loc after short
        steps (the step-0 zero upstream gradient through W_loc=0 is the known
        null — checked after steps, never misread as dead), CSSD buffers not
        in the optimizer/state dict and basis hashes unchanged;
    (d) same-forward components (torch identity exact, float64 sum in band);
    (e) save -> correct-factory reload -> replay for all four arm x seed
        cells (bit-exact on CPU);
    (f) batch/row-order invariance + label independence (fit molecules);
    (g) the alpha_replace patch sits on model.local_tuple and its entry
        replacement changes phi_hat while the common term is untouched.
    """
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cpu")
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, fit_data = ctx["fit_idx"], ctx["fit_data"]
    g_all = np.asarray(objects["targets"]["g"], np.float64)
    ell_all = np.asarray(objects["targets"]["ell"], np.float64)
    s_all = np.asarray(objects["targets"]["s"], np.float64)
    target_g = torch.as_tensor(g_all[fit_idx], dtype=torch.float32)
    checks: dict[str, Any] = {}

    # (a) SmallMLPBridge contract (pure tensors, no data)
    torch.manual_seed(123)
    h_probe = torch.randn(64, BRIDGE_IN_OUT) * 3.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(B72_INIT_SEED_BASE)
        bridge = SmallMLPBridge()
    manual_scale = torch.sqrt(h_probe.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
    manual = manual_scale * (
        bridge.fc2(F.silu(bridge.fc1(h_probe / manual_scale)))
    )
    with torch.no_grad():
        got = bridge(h_probe)
    checks["small_bridge"] = {
        "fc1_shape": tuple(bridge.fc1.weight.shape),
        "fc2_shape": tuple(bridge.fc2.weight.shape),
        "fc1_no_bias": bridge.fc1.bias is None,
        "fc2_no_bias": bridge.fc2.bias is None,
        "forward_formula_max_abs_diff": float((got - manual).abs().max()),
        "output_dim": int(got.shape[1]),
        "parameters": int(sum(p.numel() for p in bridge.parameters())),
    }
    if checks["small_bridge"]["parameters"] != B72_BRIDGE_PARAMETERS:
        raise RuntimeError("SmallMLPBridge parameter count wrong")
    if not (checks["small_bridge"]["fc1_no_bias"] and checks["small_bridge"]["fc2_no_bias"]):
        raise RuntimeError("SmallMLPBridge must have no biases")
    if checks["small_bridge"]["forward_formula_max_abs_diff"] != 0.0:
        raise RuntimeError("SmallMLPBridge forward != the historical formula")
    if int(got.shape[1]) != BRIDGE_IN_OUT:
        raise RuntimeError("SmallMLPBridge changed the downstream width")

    # (b) per-arm factories: contracts + non-bridge pairing (both seeds)
    factory_checks: dict[str, Any] = {}
    for s in SEEDS:
        seed_everything(int(s))
        ctrl = build_arm_round("CTRL288", payload, kappa_M, basis_parts, int(s))
        seed_everything(int(s))
        b72 = build_arm_round("B72", payload, kappa_M, basis_parts, int(s))
        nb_c, nb_b = non_bridge_state(ctrl.state_dict()), non_bridge_state(b72.state_dict())
        factory_checks[str(s)] = {
            "ctrl_audit": arm_parameter_audit(ctrl),
            "b72_audit": arm_parameter_audit(b72),
            "non_bridge_bitwise_equal": bool(all(torch.equal(nb_c[k], nb_b[k]) for k in nb_c)),
            "rng_stream_identical": True,   # asserted in source_checks; restated below
        }
        rng_c = torch.get_rng_state().clone()
        seed_everything(int(s))
        b72_again = build_arm_round("B72", payload, kappa_M, basis_parts, int(s))
        if not torch.equal(rng_c, torch.get_rng_state()):
            raise RuntimeError("second B72 build disturbed the RNG stream")
        if state_hash(gen._capture_state(b72)) != state_hash(gen._capture_state(b72_again)):
            raise RuntimeError("B72 factory is not deterministic at a fixed seed")
    if not all(
        v["ctrl_audit"] == arm_expected_audit("CTRL288") and v["b72_audit"] == arm_expected_audit("B72")
        for v in factory_checks.values()
    ):
        raise RuntimeError("per-arm parameter contracts failed")
    if not all(v["non_bridge_bitwise_equal"] for v in factory_checks.values()):
        raise RuntimeError("non-bridge init tensors differ across arms")
    checks["factory"] = factory_checks

    # (c) small-batch fit-only optimization smoke (3 optimizer steps per arm)
    smoke_rows = list(range(16))
    grads: dict[str, Any] = {}
    for arm in ARMS:
        seed_everything(0)
        model = build_arm_round(arm, payload, kappa_M, basis_parts, 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        n_opt = int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
        if n_opt != arm_expected_audit(arm)["total_parameters"]:
            raise RuntimeError(f"{arm}: optimizer parameters {n_opt} != contract total")
        state_keys = [k for k in model.state_dict() if k.startswith("local_tuple.cssd_")]
        if state_keys:
            raise RuntimeError(f"{arm}: CSSD basis leaked into the state dict: {state_keys}")
        encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
        finite_ok = True
        loss_values = []
        for _step in range(3):
            batch = zftd.make_batch(fit_data, smoke_rows, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + COMPONENT_LOSS_WEIGHT * (
                F.l1_loss(components[:, 0], torch.as_tensor(ell_all[fit_idx][smoke_rows], dtype=torch.float32))
                + F.l1_loss(components[:, 1], torch.as_tensor(s_all[fit_idx][smoke_rows], dtype=torch.float32))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            finite_ok = finite_ok and bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
            loss_values.append(float(loss.detach()))
        bridge_mod = model.local_dictionary_bridge
        g1 = float(bridge_mod.fc1.weight.grad.norm()) if bridge_mod.fc1.weight.grad is not None else None
        g2 = float(bridge_mod.fc2.weight.grad.norm()) if bridge_mod.fc2.weight.grad is not None else None
        wloc_g = float(model.local_tuple.W_loc.grad.norm()) if model.local_tuple.W_loc.grad is not None else None
        encoder_hashes_after = model.local_tuple.frozen_basis_hashes()
        if encoder_hashes_after != encoder_hashes_before:
            raise RuntimeError(f"{arm}: frozen basis changed during the optimization smoke")
        if not finite_ok:
            raise RuntimeError(f"{arm}: non-finite outputs/loss in the optimization smoke")
        if not (g1 and g1 > 0.0 and g2 and g2 > 0.0):
            raise RuntimeError(f"{arm}: bridge gradients not reaching both layers after 3 steps: {g1}, {g2}")
        grads[arm] = {
            "loss_values": loss_values,
            "bridge_fc1_grad_norm_after_3_steps": g1,
            "bridge_fc2_grad_norm_after_3_steps": g2,
            "W_loc_grad_norm_after_3_steps": wloc_g,
            "cssd_buffers_not_in_optimizer": True,
            "frozen_basis_hashes_unchanged": True,
        }
    if all((v["W_loc_grad_norm_after_3_steps"] or 0.0) == 0.0 for v in grads.values()):
        raise RuntimeError("W_loc never received gradient after 3 steps (both arms)")
    checks["gradient_smoke"] = grads

    # (d) same-forward components (fit rows, one batch, both arms)
    identity: dict[str, Any] = {}
    for arm in ARMS:
        seed_everything(0)
        model = build_arm_round(arm, payload, kappa_M, basis_parts, 0)
        model.eval()
        batch = zftd.make_batch(fit_data, smoke_rows, target_g, device)
        with torch.no_grad():
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
        torch_gap = float((components.sum(-1) - prediction).abs().max())
        f64_gap = float(
            np.max(np.abs(
                prediction.numpy().astype(np.float64)
                - (components[:, 0].numpy().astype(np.float64) + components[:, 1].numpy().astype(np.float64))
            ))
        )
        if torch_gap != 0.0:
            raise RuntimeError(f"{arm}: torch-internal component identity broken")
        if f64_gap > FLOAT64_SUM_TOL:
            raise RuntimeError(f"{arm}: exported float64 component sum off h")
        identity[arm] = {"torch_identity_max": torch_gap, "float64_sum_max_abs": f64_gap}
    checks["same_forward_components"] = identity

    # (e) save -> correct-factory reload -> replay (all four cells, CPU bit-exact)
    replay: dict[str, Any] = {}
    for arm in ARMS:
        for s in SEEDS:
            seed_everything(int(s))
            model = build_arm_round(arm, payload, kappa_M, basis_parts, int(s))
            state = gen._capture_state(model)
            tmp = out_dir / "pretrain_scratch"
            tmp.mkdir(parents=True, exist_ok=True)
            replay_path = tmp / f"{arm}_s{s}_replay_state.pt"
            torch.save(state, replay_path)
            reloaded = torch.load(replay_path, map_location="cpu", weights_only=False)
            replay_path.unlink()
            if state_hash(reloaded) != state_hash(state):
                raise RuntimeError(f"{arm} s{s}: save/reload hash mismatch")
            model_r = build_arm_round(arm, payload, kappa_M, basis_parts, int(s))
            model_r.load_state_dict({k: v for k, v in reloaded.items()}, strict=True)
            batch = zftd.make_batch(fit_data, smoke_rows, target_g, device)
            model.eval(); model_r.eval()
            with torch.no_grad():
                p_a = model(batch, mask=cm.C6_MASK)
                p_b = model_r(batch, mask=cm.C6_MASK)
            d = float((p_a - p_b).abs().max())
            if d != 0.0:
                raise RuntimeError(f"{arm} s{s}: reload replay differs by {d} on CPU")
            wrong = build_arm_round(
                "CTRL288" if arm == "B72" else "B72", payload, kappa_M, basis_parts, int(s)
            )
            try:
                wrong.load_state_dict({k: v for k, v in reloaded.items()}, strict=True)
            except RuntimeError:
                cross_factory_blocked = True
            else:
                cross_factory_blocked = False
            if not cross_factory_blocked:
                raise RuntimeError(f"{arm} s{s}: state loaded into the OTHER arm's factory")
            replay[f"{arm}_s{s}"] = {"reload_replay_max_abs": d, "cross_factory_strict_load_blocked": True}
    checks["save_reload_replay"] = replay

    # (f) batch/row-order invariance + label independence (fit molecules)
    invariance: dict[str, Any] = {}
    subset = fit_data[:24]
    for arm in ARMS:
        seed_everything(0)
        model = build_arm_round(arm, payload, kappa_M, basis_parts, 0)
        model.eval()
        order = list(range(len(subset)))
        with torch.no_grad():
            singles = []
            for i in order:
                b1 = zftd.make_batch(subset, [i], torch.zeros(len(subset)), device)
                singles.append(float(model(b1, mask=cm.C6_MASK).view(-1)[0]))
            shuffled = [order[(i * 7 + 3) % len(order)] for i in range(len(order))]
            bs = zftd.make_batch(subset, shuffled, torch.zeros(len(subset)), device)
            grouped = model(bs, mask=cm.C6_MASK).view(-1).numpy().astype(np.float64)
            b_y0 = zftd.make_batch(fit_data, list(range(8)), torch.zeros(8), device)
            p_y0 = model(b_y0, mask=cm.C6_MASK)
            b_y1 = zftd.make_batch(fit_data, list(range(8)), torch.ones(8), device)
            p_y1 = model(b_y1, mask=cm.C6_MASK)
        d = np.abs(np.asarray(singles)[np.asarray(shuffled)] - grouped)
        invariance[arm] = {
            "batch_order_max_abs": float(d.max()),
            "label_independence_max_abs": float((p_y0 - p_y1).abs().max()),
        }
        if invariance[arm]["batch_order_max_abs"] > 1e-5:
            raise RuntimeError(f"{arm}: batch/order invariance failed")
        if invariance[arm]["label_independence_max_abs"] != 0.0:
            raise RuntimeError(f"{arm}: labels entered the forward path")
    checks["batch_order_and_label_invariance"] = invariance

    # (g) the alpha_replace patch: sits on model.local_tuple; the entry
    #     replacement changes phi_hat; the common term is untouched
    mean_codes = repl.fit_mean_codes(ctx["basis"], fit_idx)
    mean_alpha = torch.as_tensor(mean_codes["mean_alpha"], dtype=torch.float32)
    sample = repl._fit_sample(objects, 256)
    sample_rows, _rows_checks = repl._phi_rows_for(sample.tolist())
    seed_everything(0)
    model = build_arm_round("B72", payload, kappa_M, basis_parts, 0)
    encoder = model.local_tuple
    encoder.eval()
    phi_t = torch.as_tensor(sample_rows, dtype=torch.float32)
    with torch.no_grad():
        phi_hat_ref = encoder.cssd_decode(phi_t)
        encoder.alpha_replace = mean_alpha
        phi_hat_mean = encoder.cssd_decode(phi_t)
        encoder.alpha_replace = None
        restored = encoder.cssd_decode(phi_t)
    z_ref, phi_hat_ref2, _rel = zreuse._root_codes(ctx["basis"], sample_rows)
    dbar_ref = np.asarray(zreuse._frozen_code_parts(ctx["basis"])[2].numpy(), np.float64)
    alpha_ref = z_ref[:, repl.COMMON_DIM:].astype(np.float64)
    common_ref = phi_hat_ref2 - alpha_ref @ dbar_ref.T
    common_int = (
        phi_hat_mean.numpy().astype(np.float64)
        - np.asarray(mean_codes["mean_alpha"], np.float64) @ dbar_ref.T
    )
    checks["alpha_replace_patch"] = {
        "sits_on": "model.local_tuple.alpha_replace (the CSSD encoder, never the wrapper)",
        "decode_matches_reference_max_abs": float(
            np.abs(phi_hat_ref.numpy().astype(np.float64) - phi_hat_ref2).max()),
        "phi_hat_max_abs_change": float(
            np.abs(phi_hat_mean.numpy().astype(np.float64) - phi_hat_ref2).max()),
        "common_term_max_abs_change": float(np.abs(common_int - common_ref).max()),
        "restore_max_abs": float(
            np.abs(restored.numpy().astype(np.float64) - phi_hat_ref.numpy().astype(np.float64)).max()),
        "mean_alpha_sha256": hashlib.sha256(
            np.asarray(mean_codes["mean_alpha"], np.float64).tobytes()).hexdigest(),
        "fit_rel_err_median": float(mean_codes["fit_rel_err_median"]),
        "fit_alpha_nnz": float(mean_codes["fit_alpha_nnz"]),
    }
    if checks["alpha_replace_patch"]["decode_matches_reference_max_abs"] > 1e-5:
        raise RuntimeError("CSSD decode != the reference operator")
    if checks["alpha_replace_patch"]["phi_hat_max_abs_change"] <= 0.0:
        raise RuntimeError("alpha_replace entry replacement did not change phi_hat")
    if checks["alpha_replace_patch"]["common_term_max_abs_change"] > 1e-5:
        raise RuntimeError("alpha_replace moved the common term")
    if checks["alpha_replace_patch"]["restore_max_abs"] != 0.0:
        raise RuntimeError("alpha_replace restoration is not bit-exact")

    checks["all_passed"] = True
    checks["seconds"] = float(time.perf_counter() - started)
    checks["official_valid_loaded"] = False
    checks["official_test_loaded"] = False
    checks["dev_rows_touched"] = False
    write_json(out_dir / "pretrain_checks.json", checks)
    log(f"[pretrain-checks] all passed in {checks['seconds']:.1f}s")
    return checks


# ---------------------------------------------------------------------------
# 3. stage: one formal body training (240 epochs, from-scratch, fit-only)
# ---------------------------------------------------------------------------

def _probe_channels(model: nn.Module, arm: str, epoch: int, step: int, total_norm: float) -> dict[str, Any]:
    """The historical local-channel probe + the two bridge layer gradients."""
    entry = repl._probe_local_channel(model, "DICT", epoch, step, total_norm)
    entry["arm"] = arm
    bridge = model.local_dictionary_bridge
    entry["bridge_fc1_grad_norm"] = (
        float(bridge.fc1.weight.grad.norm()) if bridge.fc1.weight.grad is not None else None
    )
    entry["bridge_fc2_grad_norm"] = (
        float(bridge.fc2.weight.grad.norm()) if bridge.fc2.weight.grad is not None else None
    )
    entry["bridge_fc1_norm"] = float(bridge.fc1.weight.detach().norm())
    entry["bridge_fc2_norm"] = float(bridge.fc2.weight.detach().norm())
    return entry


def train_run(
    arm: str,
    seed: int,
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    smoke: bool = False,
    max_steps: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """One from-scratch 240-epoch training of one arm x seed (fit rows only).

    Clones the source round's ``train_arm`` loop exactly (schedule, init,
    optimizer, batch order, loss, probes, curve, soup semantics) with only
    the per-arm factory of THIS round and deterministic state capture at the
    member/checkpoint epochs.  During training nothing is built or scored:
    only detach/clone captures; dev is never touched.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    if int(seed) not in SEEDS:
        raise ValueError(seed)
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    out_dir = Path(out_dir)
    rdir = run_dir(arm, seed, out_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "members").mkdir(exist_ok=True)
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    payload, kappa_M = ctx["payload"], ctx["kappa_M"]
    fit_idx, fit_data = ctx["fit_idx"], ctx["fit_data"]
    targets = objects["targets"]
    y_all = np.asarray(targets["y"], np.float64)
    g_all = np.asarray(targets["g"], np.float64)
    ell_all = np.asarray(targets["ell"], np.float64)
    s_all = np.asarray(targets["s"], np.float64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_fit, g_fit = y_all[fit_idx], g_all[fit_idx]
    ell_fit, s_fit = ell_all[fit_idx], s_all[fit_idx]
    if float(np.max(np.abs(g_fit - (ell_fit + s_fit)))) > 1e-12:
        raise RuntimeError("g != ell + s on fit rows")
    target_g = torch.as_tensor(g_fit, dtype=torch.float32)
    target_ell = torch.as_tensor(ell_fit, dtype=torch.float32, device=device)
    target_s = torch.as_tensor(s_fit, dtype=torch.float32, device=device)

    epochs = 1 if smoke else int(EPOCHS)
    schedule, schedule_hash = zw.build_schedule(len(fit_data), epochs, int(seed) + TRAIN_SHUFFLE_OFFSET)
    old = gen.load_old_run(int(seed))
    for e in range(epochs):
        if not np.array_equal(schedule[e], old["schedule_order"][e]):
            raise RuntimeError(f"schedule mismatch vs the historical plan at epoch {e + 1}")

    seed_everything(int(seed))
    model = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    build_rng = torch.get_rng_state().clone()
    init_state = gen._capture_state(model)
    init_hash = state_hash(init_state)
    # cross-arm non-bridge identity (fresh CPU constructions, fit-free)
    ctrl_state = gen._capture_state(
        build_arm_round("CTRL288", payload, kappa_M, basis_parts, int(seed))
    )
    nb_init, nb_ctrl = non_bridge_state(init_state), non_bridge_state(ctrl_state)
    if sorted(nb_init) != sorted(nb_ctrl):
        raise RuntimeError(f"{arm} s{seed}: non-bridge state key sets differ vs CTRL288")
    mismatched = [k for k in nb_init if not torch.equal(nb_init[k], nb_ctrl[k])]
    if mismatched:
        raise RuntimeError(f"{arm} s{seed}: non-bridge init differs vs CTRL288 at {mismatched}")
    if arm == "CTRL288" and init_hash != old["manifest"]["init_state_sha256"]:
        raise RuntimeError(f"CTRL288 s{seed} init does not reproduce the historical DICT init")
    b72_bridge_init_sha256 = (
        state_hash(bridge_state(init_state)) if arm == "B72" else None
    )
    del ctrl_state, nb_init, nb_ctrl

    model = model.to(device)
    expected_encoder_hashes = {
        "U": array_sha256(basis_parts["U"].numpy()),
        "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
        "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
    }
    encoder_hashes_before = model.local_tuple.frozen_basis_hashes()
    if encoder_hashes_before != expected_encoder_hashes:
        raise RuntimeError(f"{arm} s{seed}: frozen basis buffers != loaded basis package")
    seed_everything(int(seed))
    train_rng = torch.get_rng_state().clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_opt = int(sum(p.numel() for group in optimizer.param_groups for p in group["params"]))
    if n_opt != arm_expected_audit(arm)["total_parameters"]:
        raise RuntimeError(f"{arm} s{seed}: optimizer parameters {n_opt} != contract total")
    member_epochs = [1] if smoke else [int(e) for e in SOUP_EPOCHS]
    checkpoint_epochs = [1] if smoke else [120, 240]
    capture_epochs = sorted(set(member_epochs) | set(checkpoint_epochs))
    members: dict[int, dict[str, torch.Tensor]] = {}
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    steps_done = 0
    stopped_reason = "completed"
    started = time.perf_counter()
    peak_mb = 0.0
    position_stream = hashlib.sha256()
    gid_stream = hashlib.sha256()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        g_sum = ell_sum = s_sum = total_sum = 0.0
        n_mol = n_steps = 0
        gnorm_sum = 0.0
        clip_hits = 0
        for start in range(0, len(schedule[epoch - 1]), BATCH_SIZE):
            indices = [int(i) for i in schedule[epoch - 1][start:start + BATCH_SIZE]]
            index_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            position_stream.update(np.asarray(indices, np.int64).tobytes())
            gid_stream.update(np.asarray(gid_all[fit_idx[np.asarray(indices, np.int64)]], np.int64).tobytes())
            batch = zftd.make_batch(fit_data, indices, target_g, device)
            prediction = model(batch, mask=cm.C6_MASK)
            if tuple(prediction.shape) != (len(indices),):
                raise RuntimeError("standard forward must return (batch,) total g")
            components = model.reader.components()
            if tuple(components.shape) != (len(indices), 2):
                raise RuntimeError("reader component output is not (batch, 2)")
            identity_gap = float(torch.max(torch.abs(components.sum(-1) - prediction.detach())).item())
            if identity_gap != 0.0:
                raise RuntimeError(f"g_hat != ell_hat + s_hat exactly (gap {identity_gap})")
            l_g = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            l_ell = F.l1_loss(components[:, 0], target_ell[index_t])
            l_s = F.l1_loss(components[:, 1], target_s[index_t])
            loss = l_g + COMPONENT_LOSS_WEIGHT * (l_ell + l_s)
            optimizer.zero_grad()
            loss.backward()
            total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if epoch in LOG_EPOCHS and n_steps == 0:
                probes.append(_probe_channels(model, arm, epoch, steps_done + 1, total_norm))
            optimizer.step()
            g_sum += float((prediction.detach().view(-1) - batch.y.view(-1)).abs().sum())
            ell_sum += float((components[:, 0].detach() - target_ell[index_t]).abs().sum())
            s_sum += float((components[:, 1].detach() - target_s[index_t]).abs().sum())
            total_sum += float(loss.detach()) * int(len(indices))
            n_mol += int(batch.y.numel())
            n_steps += 1
            steps_done += 1
            gnorm_sum += total_norm
            clip_hits += int(total_norm > GRAD_CLIP)
            if max_steps is not None and steps_done >= int(max_steps):
                stopped_reason = "max_steps"
                break
        if device.type == "cuda":
            peak_mb = max(peak_mb, float(torch.cuda.max_memory_allocated() / (1 << 20)))
        curve.append({
            "epoch": int(epoch),
            "train_L_g": float(g_sum / max(n_mol, 1)),
            "train_L_ell": float(ell_sum / max(n_mol, 1)),
            "train_L_s": float(s_sum / max(n_mol, 1)),
            "train_loss": float(total_sum / max(n_mol, 1)),
            "grad_norm": float(gnorm_sum / max(n_steps, 1)),
            "clip_fraction": float(clip_hits / max(n_steps, 1)),
            "seconds": float(time.perf_counter() - epoch_started),
        })
        if epoch in capture_epochs:
            state = gen._capture_state(model)          # deep copy: no live references
            if epoch in member_epochs:
                members[int(epoch)] = state
                torch.save(state, rdir / "members" / f"epoch{epoch}_state.pt")
            if epoch in checkpoint_epochs:
                checkpoints[int(epoch)] = state
        if log and (epoch in LOG_EPOCHS or epoch % 40 == 0):
            c = curve[-1]
            log(
                f"[train {arm} s{seed}] ep={epoch:03d} Lg={c['train_L_g']:.5f} "
                f"Lell={c['train_L_ell']:.5f} Ls={c['train_L_s']:.5f} "
                f"gn={c['grad_norm']:.3g} {c['seconds']:.1f}s",
                flush=True,
            )
        if stopped_reason == "max_steps":
            break

    if not smoke:
        if stopped_reason != "completed":
            raise RuntimeError(f"formal run stopped early: {stopped_reason}")
        if steps_done != STEPS_EXPECTED:
            raise RuntimeError(f"{arm} s{seed}: steps {steps_done} != {STEPS_EXPECTED}")
        if sorted(members) != [int(e) for e in SOUP_EPOCHS]:
            raise RuntimeError(f"soup epochs missing: {sorted(members)}")
        if sorted(checkpoints) != [120, 240]:
            raise RuntimeError(f"trace checkpoints missing: {sorted(checkpoints)}")

    last_state = gen._capture_state(model)
    member_hashes = {int(e): state_hash(members[e]) for e in sorted(members)}
    soup_state = gen.average_states([members[e] for e in member_epochs])
    torch.save(soup_state, rdir / "soup_state.pt")
    soup_hash = state_hash(soup_state)

    # estimator strict-load + save/reload prediction assertion (fit rows only)
    probe_rows = list(range(8))
    probe_batch = zftd.make_batch(fit_data, probe_rows, torch.zeros(8), device)
    model_e = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    model_e.load_state_dict({k: v for k, v in soup_state.items()}, strict=True)
    model_e = model_e.to(device).eval()
    with torch.no_grad():
        p_mem = model_e(probe_batch, mask=cm.C6_MASK).detach().cpu().clone()
    reloaded = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
    if state_hash(reloaded) != soup_hash:
        raise RuntimeError(f"{arm} s{seed}: soup save/reload hash mismatch")
    model_r = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    model_r.load_state_dict({k: v for k, v in reloaded.items()}, strict=True)
    model_r = model_r.to(device).eval()
    with torch.no_grad():
        p_disk = model_r(probe_batch, mask=cm.C6_MASK).detach().cpu().clone()
    # CPU constructions replay bit-exact; separately allocated GPU instances may
    # pick different cuBLAS kernels, so the GPU band is the cross-device FP32
    # tolerance (the located fix of the source rounds, kept here from the start)
    reload_tol = 0.0 if device.type == "cpu" else CROSS_DEVICE_TOL
    d_reload = float((p_mem - p_disk).abs().max())
    if d_reload > reload_tol:
        raise RuntimeError(f"{arm} s{seed}: save/reload predictions differ by {d_reload}")
    del model_e, model_r
    if device.type == "cuda":
        torch.cuda.empty_cache()

    encoder_hashes_after = model.local_tuple.frozen_basis_hashes()
    if encoder_hashes_after != encoder_hashes_before:
        raise RuntimeError(f"{arm} s{seed}: frozen basis changed during training")

    # exact-resume capability record (never used for the round's numbers)
    torch.save(
        {
            "model_state": last_state,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "steps_done": int(steps_done),
            "epochs_done": int(curve[-1]["epoch"]) if curve else 0,
            "schedule_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        rdir / "resume_state.pt",
    )
    np.savez_compressed(
        rdir / "schedule.npz",
        order=np.stack([np.asarray(e, np.int64) for e in schedule]),
    )
    torch.save(init_state, rdir / "init_state.pt")
    torch.save(last_state, rdir / "last_state.pt")
    for epoch, state in sorted(checkpoints.items()):
        torch.save(state, rdir / f"epoch{epoch}_state.pt")

    # fit-only diagnostics from the soup (never dev, never a selection)
    replay = build_arm_round(arm, payload, kappa_M, basis_parts, int(seed))
    replay.load_state_dict({k: v for k, v in soup_state.items()}, strict=True)
    replay = replay.to(device).eval()
    h, comps = zcs.evaluate_state_components(replay, fit_data, device)
    ell_hat, s_hat = comps[:, 0], comps[:, 1]
    T_topology = nb.topology_matrix(fit_data)
    q_raw = nb._q_predictions(ctx["q_soup"], T_topology, device)
    y_raw = ell_hat + s_hat + q_raw
    if float(np.max(np.abs(y_raw - (h + q_raw)))) > 1e-5:
        raise RuntimeError("y_raw != h + Q_raw on fit rows")
    b_y = float(np.median(y_fit - y_raw))
    if arm_parameter_audit(replay) != arm_expected_audit(arm):
        raise RuntimeError(f"{arm} s{seed}: post-training parameter audit failed")
    if replay.local_tuple.frozen_basis_hashes() != expected_encoder_hashes:
        raise RuntimeError(f"{arm} s{seed}: replay encoder basis hashes changed")
    post_audit = arm_parameter_audit(replay)
    del replay
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(
        rdir / "fit_predictions.npz",
        row_index=np.asarray(fit_idx, np.int64), gid=gid_all[fit_idx],
        y=y_fit, g=g_fit, ell=ell_fit, s=s_fit,
        h=h, ell_hat=ell_hat, s_hat=s_hat, q_raw=q_raw, y_raw=y_raw,
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "train",
        "arm": arm,
        "seed": int(seed),
        "smoke": bool(smoke),
        "run_dir": str(rdir.relative_to(out_dir)),
        "recipe": {
            "epochs": int(epochs), "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "soup_epochs": list(member_epochs), "trace_checkpoints": list(checkpoint_epochs),
            "log_epochs": list(LOG_EPOCHS),
            "component_loss_weight": COMPONENT_LOSS_WEIGHT,
            "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            "amp": None, "ddp": None, "scheduler": None, "early_stopping": None,
        },
        "supervision": (
            "COMP: L = MAE(ell_hat+s_hat, g) + 0.5*(MAE(ell_hat,ell)+MAE(s_hat,s)) "
            "(the unchanged disclosed auxiliary condition)"
        ),
        "steps_done": int(steps_done),
        "steps_expected": int(sum((len(e) + BATCH_SIZE - 1) // BATCH_SIZE for e in schedule)),
        "stopped_reason": stopped_reason,
        "schedule_sha256": schedule_hash,
        "schedule_bit_identical_to_historical": True,
        "position_stream_sha256": position_stream.hexdigest(),
        "global_gid_stream_sha256": gid_stream.hexdigest(),
        "init_state_sha256": init_hash,
        "init_non_bridge_sha256": state_hash(non_bridge_state(init_state)),
        "non_bridge_bitwise_equal_ctrl288": True,
        "b72_bridge_init_sha256": b72_bridge_init_sha256,
        "member_state_sha256": {str(int(e)): member_hashes[int(e)] for e in sorted(members)},
        "soup_state_sha256": soup_hash,
        "last_state_sha256": state_hash(last_state),
        "build_rng_sha256": hashlib.sha256(build_rng.numpy().tobytes()).hexdigest(),
        "train_rng_sha256": hashlib.sha256(train_rng.numpy().tobytes()).hexdigest(),
        "frozen_basis_hashes": basis_parts["hashes"],
        "encoder_frozen_basis_hashes": encoder_hashes_after,
        "frozen_basis_unchanged": True,
        "q_soup_sha256": state_hash(ctx["q_soup"]),
        "parameter_audit": post_audit,
        "calibration": {
            "b_y": b_y,
            "note": "single fit-median b_y = median(y_fit - y_raw_fit); never stacked with any old bias",
        },
        "fit_diagnostics": {
            "fit_y_raw_mae": float(np.mean(np.abs(y_fit - y_raw))),
            "fit_y_cal_mae": float(np.mean(np.abs(y_fit - (y_raw + b_y)))),
            "fit_g_raw_mae": float(np.mean(np.abs(g_fit - h))),
            "fit_ell_mae": float(np.mean(np.abs(ell_fit - ell_hat))),
            "fit_s_mae": float(np.mean(np.abs(s_fit - s_hat))),
            "fit_q_mae": float(np.mean(np.abs(objects["targets"]["c"][fit_idx] - q_raw))),
        },
        "curve_seconds_total": float(sum(c["seconds"] for c in curve)),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(peak_mb),
        "device": str(device),
        "allocation_probe": repl.allocation_probe(),
        "dev_scores_computed_during_training": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(rdir / "curve.json", curve)
    write_json(rdir / "probes.json", probes)
    write_json(rdir / "manifest.json", manifest)
    log(
        f"[train {arm} s{seed}] done: {steps_done} steps, {manifest['curve_seconds_total']:.0f}s train, "
        f"fit y_raw MAE {manifest['fit_diagnostics']['fit_y_raw_mae']:.5f} "
        f"b_y={b_y:.5f} soup sha={soup_hash[:12]}…"
    )
    return manifest


# ---------------------------------------------------------------------------
# 4. stage: GPU smoke (short end-to-end plumbing check, fit-only)
# ---------------------------------------------------------------------------

def run_smoke(
    *,
    device_name: str = "cuda:0",
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    """One 1-epoch trajectory per arm (seed 0) through the FULL train path
    (capture/average/strict-load/save-reload), fit rows only; no dev read."""
    device = resolve_device(device_name)
    out_dir = Path(out_dir)
    smoke_dir = out_dir / "smoke"
    started = time.perf_counter()
    manifests = {}
    for arm in ARMS:
        manifests[arm] = train_run(
            arm, 0, device_name=device_name, out_dir=smoke_dir, smoke=True, log=log,
        )
    smoke = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "smoke",
        "device": str(device),
        "arms": {
            arm: {
                "steps_done": m["steps_done"],
                "stopped_reason": m["stopped_reason"],
                "parameter_audit": m["parameter_audit"],
                "schedule_bit_identical_to_historical": m["schedule_bit_identical_to_historical"],
                "soup_state_sha256": m["soup_state_sha256"],
            }
            for arm, m in manifests.items()
        },
        "checks": {
            "steps_1_epoch_63": all(m["steps_done"] == 63 for m in manifests.values()),
            "both_arms_completed": all(m["stopped_reason"] == "completed" for m in manifests.values()),
            "parameter_contracts_ok": all(
                m["parameter_audit"] == arm_expected_audit(arm) for arm, m in manifests.items()
            ),
            "frozen_basis_unchanged": all(m["frozen_basis_unchanged"] for m in manifests.values()),
            "aggregation_strict_load": True,
            "save_reload_predictions_within_band": True,
            "dev_never_touched": True,
        },
        "all_passed": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not all(smoke["checks"].values()):
        raise RuntimeError(f"smoke checks failed: { {k: v for k, v in smoke['checks'].items() if not v} }")
    write_json(out_dir / "smoke.json", smoke)
    log(f"[smoke] both arms ok on {device} in {smoke['seconds']:.1f}s")
    return smoke


# ---------------------------------------------------------------------------
# 5. terminal helpers (all accept THIS round's arm/factory, never the old ones)
# ---------------------------------------------------------------------------

def _load_soup_model(
    arm: str, seed: int, ctx: Mapping[str, Any], *, out_dir: Path,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    run = load_run(arm, seed, out_dir)
    model = build_arm_round(arm, ctx["payload"], ctx["kappa_M"], ctx["basis_parts"], int(seed))
    model.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
    return model, run["soup_state"]


def _score_run(
    arm: str, seed: int, ctx: Mapping[str, Any], *,
    out_dir: Path, device: torch.device, q_fit: np.ndarray, q_dev: np.ndarray,
) -> dict[str, Any]:
    """Full fit+dev scoring of one soup (b_y from this run's own fit median)."""
    model, state = _load_soup_model(arm, seed, ctx, out_dir=out_dir)
    rdir = run_dir(arm, seed, out_dir)
    manifest = read_json(rdir / "manifest.json")
    if state_hash(state) != manifest["soup_state_sha256"]:
        raise RuntimeError(f"{arm} s{seed}: soup state hash mismatch vs manifest")
    model = model.to(device).eval()
    fitp = gen._predict_rows_checked(model, ctx["fit_data"], device)
    devp = gen._predict_rows_checked(model, ctx["dev_data"], device)
    y_raw_fit = fitp["ell_hat"] + fitp["s_hat"] + q_fit
    y_raw_dev = devp["ell_hat"] + devp["s_hat"] + q_dev
    b_y = float(np.median(ctx["y_fit"] - y_raw_fit))
    return {
        "arm": arm, "seed": int(seed), "model": model, "state": state,
        "fit": {"h": fitp["h"], "ell_hat": fitp["ell_hat"], "s_hat": fitp["s_hat"],
                "y_raw": y_raw_fit, "b_y": b_y},
        "dev": {"h": devp["h"], "ell_hat": devp["ell_hat"], "s_hat": devp["s_hat"],
                "y_raw": y_raw_dev, "b_y": b_y},
    }


def paired_bootstrap_two_metrics(
    errs_y: Mapping[str, np.ndarray],
    errs_g: Mapping[str, np.ndarray],
    group_of_row: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Canonical-SMILES-group-paired bootstrap of Delta_y AND Delta_g.

    2000 draws, fixed seed 20261007; the SAME group picks are shared by both
    arms, both seeds AND both metrics within a draw; each draw computes the
    per-seed paired MAE difference (B72 - CTRL288) for each metric, then the
    two-seed average, before any CI is taken.  Molecular sampling only — no
    training-seed/basis uncertainty, not corrected for candidate selection.
    """
    keys = [f"{arm}_s{s}" for arm in ARMS for s in SEEDS]
    for table, tag in ((errs_y, "y"), (errs_g, "g")):
        for key in keys:
            if key not in table:
                raise RuntimeError(f"bootstrap expects {keys}, missing {key} ({tag})")
            if np.asarray(table[key]).shape != np.asarray(group_of_row).shape:
                raise RuntimeError(f"row misalignment at {key} ({tag})")
    labels = np.asarray([str(v) for v in np.asarray(group_of_row, dtype=object).tolist()], dtype=object)
    order = {label: index for index, label in enumerate(sorted(set(labels.tolist())))}
    g_index = np.asarray([order[label] for label in labels.tolist()], np.int64)
    n_groups = len(order)
    cnt = np.zeros(n_groups, np.int64)
    np.add.at(cnt, g_index, 1)
    sums = {
        tag: {k: np.zeros(n_groups, np.float64) for k in keys} for tag in ("y", "g")
    }
    for tag, table in (("y", errs_y), ("g", errs_g)):
        for k in keys:
            np.add.at(sums[tag][k], g_index, np.asarray(table[k], np.float64))
    rng = np.random.default_rng(int(seed))
    dy = {s: np.empty(int(n_boot)) for s in SEEDS}
    dg = {s: np.empty(int(n_boot)) for s in SEEDS}
    avg_y = np.empty(int(n_boot))
    avg_g = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_groups, n_groups)   # shared across arms/seeds/metrics
        n = int(cnt[pick].sum())
        m_y = {k: float(sums["y"][k][pick].sum() / n) for k in keys}
        m_g = {k: float(sums["g"][k][pick].sum() / n) for k in keys}
        for s in SEEDS:
            dy[s][b] = m_y[f"B72_s{s}"] - m_y[f"CTRL288_s{s}"]
            dg[s][b] = m_g[f"B72_s{s}"] - m_g[f"CTRL288_s{s}"]
        avg_y[b] = float(np.mean([dy[s][b] for s in SEEDS]))
        avg_g[b] = float(np.mean([dg[s][b] for s in SEEDS]))
    return {
        "n_rows": int(cnt.sum()),
        "n_groups": int(n_groups),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "shared_group_resampling": True,
        "shared_picks_across_metrics": True,
        "avg_seeds_first": True,
        "delta_y": {
            "per_seed_ci95": {
                str(s): [float(np.percentile(dy[s], 2.5)), float(np.percentile(dy[s], 97.5))]
                for s in SEEDS
            },
            "per_seed_mean": {str(s): float(dy[s].mean()) for s in SEEDS},
            "avg_ci95": [float(np.percentile(avg_y, 2.5)), float(np.percentile(avg_y, 97.5))],
            "avg_mean": float(avg_y.mean()),
        },
        "delta_g": {
            "per_seed_ci95": {
                str(s): [float(np.percentile(dg[s], 2.5)), float(np.percentile(dg[s], 97.5))]
                for s in SEEDS
            },
            "per_seed_mean": {str(s): float(dg[s].mean()) for s in SEEDS},
            "avg_ci95": [float(np.percentile(avg_g, 2.5)), float(np.percentile(avg_g, 97.5))],
            "avg_mean": float(avg_g.mean()),
        },
        "reading": (
            "CI covers molecule resampling only — not training-seed or basis "
            "uncertainty; not corrected for candidate selection; 2 body seeds "
            "are not a method-stability claim"
        ),
    }


def alpha_mean_intervention(
    arm: str,
    seed: int,
    base: Mapping[str, Any],
    mean_alpha: np.ndarray,
    *,
    ctx: Mapping[str, Any],
    out_dir: Path,
    device: torch.device,
    marker: float,
    log: Any = print,
) -> dict[str, Any]:
    """The unique dictionary intervention on one soup (any arm of THIS round).

    alpha_v := the fit-root mean alpha at the single local structure input via
    ``model.local_tuple.alpha_replace`` (the CSSD ENCODER — never the wrapper);
    c and every other input untouched; re-decoded and re-forwarded; no
    retraining, no recalibration.  Dependence evidence, never benefit.
    """
    model, _state = _load_soup_model(arm, seed, ctx, out_dir=out_dir)
    model = model.to(device).eval()
    model.local_tuple.alpha_replace = torch.as_tensor(np.asarray(mean_alpha, np.float64), dtype=torch.float32)
    preds = gen._predict_rows_checked(model, ctx["dev_data"], device)
    model.local_tuple.alpha_replace = None

    basis = ctx["basis"]
    dev_idx = ctx["dev_idx"]
    rows, _checks = repl._phi_rows_for(dev_idx.tolist())
    z, phi_hat, _rel = zreuse._root_codes(basis, rows)
    alpha_dev = z[:, repl.COMMON_DIM:]
    mean = np.asarray(mean_alpha, np.float64)
    U = np.asarray(basis["U"], np.float64)
    Dbar = np.asarray(zreuse._frozen_code_parts(basis)[2].numpy(), np.float64)
    c = rows.astype(np.float64) @ U
    phi_hat_int = c @ U.T + mean[None, :] @ Dbar.T
    alpha_dev_from_mean = np.linalg.norm(alpha_dev - mean[None, :], axis=1)
    phi_change = np.abs(phi_hat_int - phi_hat)

    q_raw = np.asarray(base["q_raw"], np.float64)
    y_raw_i = preds["ell_hat"] + preds["s_hat"] + q_raw
    y = np.asarray(base["y"], np.float64)
    g = np.asarray(base["g"], np.float64)
    base_y_raw = np.asarray(base["y_raw"], np.float64)
    base_h = np.asarray(base["h"], np.float64)
    d_pred = np.abs(y_raw_i - base_y_raw)
    row = {
        "arm": arm, "seed": int(seed),
        "mechanism": (
            "alpha_v := the fit-root mean alpha at the single local structure input "
            "(model.local_tuple.alpha_replace on the CSSD encoder); c kept; chemistry "
            "one-hots, J incidence, Sem108, reader and the frozen Q untouched; "
            "re-decoded and re-forwarded; no retraining, no recalibration"
        ),
        "input_change": {
            "alpha_vs_mean_l2_median": float(np.median(alpha_dev_from_mean)),
            "alpha_vs_mean_l2_p95": float(np.percentile(alpha_dev_from_mean, 95)),
            "alpha_vs_mean_l2_max": float(alpha_dev_from_mean.max()),
            "phi_hat_abs_change_median": float(np.median(phi_change)),
            "phi_hat_abs_change_max": float(np.max(phi_change)),
        },
        "abs_dpred": {
            "mean": float(d_pred.mean()),
            "median": float(np.median(d_pred)),
            "p95": float(np.percentile(d_pred, 95)),
            "max": float(d_pred.max()),
        },
        "response_fraction_gt_marker": float((d_pred > float(marker)).mean()),
        "delta_y_raw_mae": float(np.mean(np.abs(y - y_raw_i)) - np.mean(np.abs(y - base_y_raw))),
        "delta_g_mae": float(np.mean(np.abs(g - preds["h"])) - np.mean(np.abs(g - base_h))),
        "g_note": "Q input unchanged by construction: q_raw reused bitwise from the base evaluation",
        "reading": "response shows the consumer reads the sparse code; never incremental benefit",
    }
    if log:
        log(
            f"[intervention {arm} s{seed}] |dPred| p95 {row['abs_dpred']['p95']:.2e} "
            f"resp {row['response_fraction_gt_marker']:.3f} dMAE {row['delta_y_raw_mae']:+.2e}"
        )
    return row


def decision_branches(
    delta_y: Mapping[int, float],
    delta_g: Mapping[int, float],
    ci: Mapping[str, Any],
    mean_fit_delta_y: float,
    responsive_b72: bool,
    checks_ok: bool,
    *,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """The frozen decision branches A-E of THIS round (engineering numbers).

    Priority: A keep-performance-candidate > B keep-cheaper-candidate >
    C dictionary-readout-not-supported > D inconclusive > E close-b72.
    Performance and parameter cost are judged SEPARATELY; the previous
    round's mean -0.003 cliff gate is deliberately NOT inherited.

    * A: both seeds Delta_y < 0; both Delta_g <= +1e-4; mean Delta_g < 0;
      mean Delta_y <= -1e-4; the mean-Delta_y group-paired CI95 upper < 0;
      B72 alpha intervention responsive beyond the marker on both seeds;
      checks ok.  (fit/dev pattern decides the READING: fit worse or
      improving clearly less than dev supports the capacity-constraint-
      generalization interpretation; fit+dev improving together is a
      recipe-performance improvement, never a solved-generalization claim.)
    * B: A missed; every seed Delta_y <= +0.002 and mean <= +0.001 with CI
      upper <= +0.002; the same three conditions on g; B72 responsive;
      checks ok.  (parameter cost is 62,208 fewer parameters — a cheaper
      candidate within THIS round's tolerance, never a proven equivalence.)
    * C: execution valid but the B72 intervention is unresponsive on any
      seed (p95 <= marker) — the readout problem of THIS recipe, never a
      scientific negative without a genuinely changed input.
    * D: A/B/C missed and the two seeds' Delta_y directions flip, or the
      mean-Delta_y CI95 simultaneously spans A territory (lower < -1e-4)
      and beyond-B territory (upper > +0.002) — the molecule-sampling
      uncertainty alone spans the whole decision space.
    * E: everything else — close THIS B72 recipe (sub-reading by the fit
      side: fit worse + dev worse = restricted-capacity interpretation;
      fit ~unchanged + dev worse = no benefit seen; fit better + dev worse
      = closest to an overfitting reading, still not a retention).
    """
    seeds = tuple(int(s) for s in seeds)
    dy = {int(s): float(delta_y[int(s)]) for s in seeds}
    dg = {int(s): float(delta_g[int(s)]) for s in seeds}
    mean_dy = float(np.mean([dy[s] for s in seeds]))
    mean_dg = float(np.mean([dg[s] for s in seeds]))
    ci_y = [float(v) for v in ci["delta_y"]["avg_ci95"]]
    ci_g = [float(v) for v in ci["delta_g"]["avg_ci95"]]
    conds_a = {
        "both_seeds_delta_y_improve": bool(all(dy[s] < A_SEED_DELTA_Y for s in seeds)),
        "both_seeds_delta_g_within_band": bool(all(dg[s] <= A_SEED_DELTA_G_BAND for s in seeds)),
        "mean_delta_g_improves": bool(mean_dg < A_MEAN_DELTA_G),
        "mean_delta_y_le_gate": bool(mean_dy <= A_MEAN_DELTA_Y),
        "mean_delta_y_ci_upper_below_0": bool(ci_y[1] < A_CI_UPPER),
        "b72_both_seeds_alpha_responsive": bool(responsive_b72),
        "checks_ok": bool(checks_ok),
    }
    conds_b = {
        "both_seeds_delta_y_within_tol": bool(all(dy[s] <= B_SEED_TOL for s in seeds)),
        "mean_delta_y_within_tol": bool(mean_dy <= B_MEAN_TOL),
        "mean_delta_y_ci_upper_within_tol": bool(ci_y[1] <= B_CI_UPPER_TOL),
        "both_seeds_delta_g_within_tol": bool(all(dg[s] <= B_SEED_TOL for s in seeds)),
        "mean_delta_g_within_tol": bool(mean_dg <= B_MEAN_TOL),
        "mean_delta_g_ci_upper_within_tol": bool(ci_g[1] <= B_CI_UPPER_TOL),
        "b72_both_seeds_alpha_responsive": bool(responsive_b72),
        "checks_ok": bool(checks_ok),
    }
    flipped = bool((dy[seeds[0]] < 0.0) != (dy[seeds[1]] < 0.0))
    ci_spans_decision_space = bool(ci_y[0] < D_CI_SPAN_LO and ci_y[1] > D_CI_SPAN_HI)
    if all(conds_a.values()):
        branch = "keep-performance-candidate"
        if mean_fit_delta_y > E_FIT_BAND or (
            mean_fit_delta_y < -E_FIT_BAND and mean_fit_delta_y > mean_dy + E_FIT_BAND
        ):
            reading = (
                "fit side worse (or improving clearly less than dev): consistent with the "
                "capacity-constraint/sample-efficiency interpretation of THIS recipe; "
                "still not a solved-generalization mechanism claim"
            )
        else:
            reading = (
                "fit and dev improve together: a recipe performance improvement of the "
                "B72 complete recipe (new bridge width + new teacher-free init); never "
                "isolated as a parameter-count effect and never a solved-generalization claim"
            )
        action = (
            "retain B72 as a performance candidate of this line; the CI is the molecule-"
            "sampling axis only and 2 seeds are not a method-stability claim"
        )
    elif all(conds_b.values()):
        branch = "keep-cheaper-candidate"
        reading = (
            "performance within THIS round's engineering tolerance at 62,208 fewer "
            "parameters (297,539 -> 235,331); not a statistically proven equivalence, "
            "not a generalization improvement; training-time savings must be read from "
            "the measured cost table, never assumed"
        )
        action = (
            "retain B72 as a cheaper-parameter candidate within this round's tolerance; "
            "the researcher decides any mainline upgrade"
        )
    elif not responsive_b72:
        branch = "dictionary-readout-not-supported"
        reading = (
            "execution valid but the B72 soup's alpha-mean intervention is unresponsive "
            "(p95 <= marker) on at least one seed: THIS recipe's readout problem; B72 is "
            "not upgraded to a dictionary-performance mainline.  Only a genuinely large "
            "input change may explain the non-response; an implementation failure is "
            "never counted as a scientific negative"
        )
        action = "do not upgrade B72 to the dictionary performance mainline; report the readout problem of this recipe"
    elif flipped or ci_spans_decision_space:
        branch = "inconclusive"
        reading = (
            "two-seed performance directions flip, or the molecule-sampling CI alone spans "
            "the whole A-E decision space: no engineering choice is supported; B72 is not "
            "closed as a compression direction and not retained as a candidate"
        )
        action = "record the uncertainty; no seed 3, no other width, no gate change this round"
    else:
        branch = "close-b72"
        if mean_fit_delta_y > E_FIT_BAND:
            sub = "fit worse + dev worse: a restricted-capacity expression of this width"
        elif mean_fit_delta_y < -E_FIT_BAND:
            sub = "fit better + dev worse: closest to an overfitting reading of this width"
        else:
            sub = "fit ~unchanged + dev worse: no benefit seen at this width"
        reading = (
            f"{sub}; closes THIS bridge-72 recipe only — never the whole capacity-"
            "constraint direction, the dictionary line, or the data noise floor"
        )
        action = "close this B72 recipe; keep CTRL288 and the historical strong DICT models; no new arm, seed or window this round"
    return {
        "branch": branch,
        "action": action,
        "reading": reading,
        "conditions": {
            "A_keep_performance": conds_a,
            "B_keep_cheaper": conds_b,
            "C_readout_responsive_b72": bool(responsive_b72),
            "D_direction_flip": flipped,
            "D_ci_spans_decision_space": ci_spans_decision_space,
        },
        "delta_y": {str(s): dy[s] for s in seeds},
        "delta_g": {str(s): dg[s] for s in seeds},
        "mean_delta_y": mean_dy,
        "mean_delta_g": mean_dg,
        "mean_fit_delta_y": float(mean_fit_delta_y),
        "thresholds_frozen": {
            "A": {
                "seed_delta_y": "< 0", "seed_delta_g_band": f"<= +{A_SEED_DELTA_G_BAND}",
                "mean_delta_g": "< 0", "mean_delta_y": f"<= {A_MEAN_DELTA_Y}",
                "mean_delta_y_ci_upper": "< 0",
            },
            "B": {
                "seed_tol": f"<= +{B_SEED_TOL}", "mean_tol": f"<= +{B_MEAN_TOL}",
                "ci_upper_tol": f"<= +{B_CI_UPPER_TOL}",
            },
            "D": {
                "direction_flip": "strict sign flip across seeds",
                "ci_span": f"lower < {D_CI_SPAN_LO} and upper > {D_CI_SPAN_HI}",
            },
            "note": (
                "all numbers are THIS round's engineering decision thresholds, never "
                "significance or SOTA claims; the CI carries the molecule-sampling axis only"
            ),
        },
        "pre_registered": True,
    }


# ---------------------------------------------------------------------------
# 6. stage: one-shot terminal eval (the single frozen comparison)
# ---------------------------------------------------------------------------

def terminal_eval(
    *,
    out_dir: Path = RESULTS_DIR,
    device_name: str = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if (out_dir / "terminal_eval.json").exists():
        raise RuntimeError("terminal_eval.json already exists: the terminal stage is one-shot")
    started = time.perf_counter()
    device = resolve_device(device_name)
    torch.set_num_threads(8 if device.type == "cpu" else 4)
    ctx = gen.load_round_context()
    objects = ctx["objects"]
    basis_parts = ctx["basis_parts"]
    fit_idx, dev_idx = ctx["fit_idx"], ctx["dev_idx"]
    targets = objects["targets"]
    y = np.asarray(targets["y"], np.float64)
    g = np.asarray(targets["g"], np.float64)
    ell = np.asarray(targets["ell"], np.float64)
    s_t = np.asarray(targets["s"], np.float64)
    k_all = np.asarray(targets["k"], np.int64)
    gid_all = np.asarray(targets["gid"], np.int64)
    y_dev, g_dev, ell_dev, s_dev = y[dev_idx], g[dev_idx], ell[dev_idx], s_t[dev_idx]
    k_dev = k_all[dev_idx]
    _phi, _atom, node_sizes = prev._env_phi_atom()
    n_nodes_fit = np.asarray(node_sizes, np.int64)[fit_idx]
    n_nodes_dev = np.asarray(node_sizes, np.int64)[dev_idx]
    g_p90 = float(np.percentile(np.abs(g[fit_idx]), 90))
    smiles = zreuse._canonical_smiles()
    dev_group_labels, dev_group_ids = np.unique(smiles[dev_idx], return_inverse=True)
    fit_group_labels, fit_group_ids = np.unique(smiles[fit_idx], return_inverse=True)
    q_fit = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(ctx["fit_data"]), device)
    q_dev = nb._q_predictions(ctx["q_soup"], nb.topology_matrix(ctx["dev_data"]), device)
    old = {s: gen.load_old_run(s) for s in SEEDS}
    old_terminal = read_json(repl.RESULTS_DIR / "terminal_eval.json")
    expected_encoder_hashes = {
        "U": array_sha256(basis_parts["U"].numpy()),
        "common_rms": array_sha256(basis_parts["common_rms"].numpy()),
        "Dbar": array_sha256(basis_parts["Dbar"].numpy()),
    }

    # (1) roster verification (before any scoring)
    checks = {
        "runs_completed": True, "steps_ok": True, "schedule_ok": True,
        "init_ok": True, "cross_seed_init_differ": True, "param_audit_ok": True,
        "frozen_basis_ok": True, "members_ok": True, "soup_ok": True,
        "q_soup_ok": True, "cross_arm_non_bridge_ok": True,
    }
    manifests: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    init_states: dict[str, dict[int, dict[str, torch.Tensor]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for s in SEEDS:
            m = read_json(run_dir(arm, s, out_dir) / "manifest.json")
            if m["stopped_reason"] != "completed" or m.get("smoke"):
                checks["runs_completed"] = False
            if int(m["steps_done"]) != STEPS_EXPECTED or int(m["steps_expected"]) != STEPS_EXPECTED:
                checks["steps_ok"] = False
            if m["parameter_audit"] != arm_expected_audit(arm):
                checks["param_audit_ok"] = False
            if m["frozen_basis_hashes"] != basis_parts["hashes"]:
                checks["frozen_basis_ok"] = False
            if m["encoder_frozen_basis_hashes"] != expected_encoder_hashes:
                checks["frozen_basis_ok"] = False
            if not m["frozen_basis_unchanged"] or not m["schedule_bit_identical_to_historical"]:
                checks["frozen_basis_ok"] = checks["schedule_ok"] = False
            if m["schedule_sha256"] != old[s]["manifest"]["schedule_sha256"]:
                checks["schedule_ok"] = False
            if state_hash(ctx["q_soup"]) != m["q_soup_sha256"]:
                checks["q_soup_ok"] = False
            if arm == "CTRL288" and m["init_state_sha256"] != old[s]["manifest"]["init_state_sha256"]:
                checks["init_ok"] = False
            for e, h in m["member_state_sha256"].items():
                member = torch.load(
                    run_dir(arm, s, out_dir) / "members" / f"epoch{int(e)}_state.pt",
                    map_location="cpu", weights_only=False,
                )
                if state_hash(member) != h:
                    checks["members_ok"] = False
            soup = torch.load(run_dir(arm, s, out_dir) / "soup_state.pt", map_location="cpu", weights_only=False)
            if state_hash(soup) != m["soup_state_sha256"]:
                checks["soup_ok"] = False
            init_states[arm][int(s)] = torch.load(
                run_dir(arm, s, out_dir) / "init_state.pt", map_location="cpu", weights_only=False
            )
            if state_hash(init_states[arm][int(s)]) != m["init_state_sha256"]:
                checks["init_ok"] = False
            manifests[arm][int(s)] = m
    for s in SEEDS:
        nb_c = non_bridge_state(init_states["CTRL288"][s])
        nb_b = non_bridge_state(init_states["B72"][s])
        if sorted(nb_c) != sorted(nb_b) or not all(torch.equal(nb_c[k], nb_b[k]) for k in nb_c):
            checks["cross_arm_non_bridge_ok"] = False
    if not (
        state_hash(non_bridge_state(init_states["CTRL288"][0]))
        != state_hash(non_bridge_state(init_states["CTRL288"][1]))
    ):
        checks["cross_seed_init_differ"] = False
    if not all(checks.values()):
        raise RuntimeError(f"roster verification failed: { {k: v for k, v in checks.items() if not v} }")

    # (2) scoring of all four soups (fit + dev, full per-row exports)
    results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    main_table: dict[str, Any] = {}
    for arm in ARMS:
        for s in SEEDS:
            res = _score_run(arm, s, ctx, out_dir=out_dir, device=device, q_fit=q_fit, q_dev=q_dev)
            results[arm][int(s)] = res
            for split, rows_idx, preds, tgt, nnodes, gids in (
                ("fit", fit_idx, res["fit"],
                 (ctx["y_fit"], ctx["g_fit"], ctx["ell_fit"], ctx["s_fit"], k_all[fit_idx]),
                 n_nodes_fit, fit_group_ids),
                ("dev", dev_idx, res["dev"], (y_dev, g_dev, ell_dev, s_dev, k_dev),
                 n_nodes_dev, dev_group_ids),
            ):
                np.savez_compressed(
                    run_dir(arm, s, out_dir) / f"{split}_predictions.npz",
                    row_index=np.asarray(rows_idx, np.int64), gid=gid_all[rows_idx],
                    smiles_group=np.asarray(gids, np.int64),
                    n_nodes=np.asarray(nnodes, np.int64), k=tgt[4],
                    y=tgt[0], g=tgt[1], ell=tgt[2], s=tgt[3],
                    h=preds["h"], ell_hat=preds["ell_hat"], s_hat=preds["s_hat"],
                    q_raw=q_fit if split == "fit" else q_dev, y_raw=preds["y_raw"],
                    b_y=np.float64(preds["b_y"]),
                )
            mae_dev = {
                "y_raw": float(np.mean(np.abs(y_dev - res["dev"]["y_raw"]))),
                "y_cal": float(np.mean(np.abs(y_dev - (res["dev"]["y_raw"] + res["dev"]["b_y"])))),
                "g_raw": float(np.mean(np.abs(g_dev - res["dev"]["h"]))),
                "ell": float(np.mean(np.abs(ell_dev - res["dev"]["ell_hat"]))),
                "s": float(np.mean(np.abs(s_dev - res["dev"]["s_hat"]))),
            }
            mae_fit = {
                "y_raw": float(np.mean(np.abs(ctx["y_fit"] - res["fit"]["y_raw"]))),
                "g_raw": float(np.mean(np.abs(ctx["g_fit"] - res["fit"]["h"]))),
                "ell": float(np.mean(np.abs(ctx["ell_fit"] - res["fit"]["ell_hat"]))),
                "s": float(np.mean(np.abs(ctx["s_fit"] - res["fit"]["s_hat"]))),
            }
            e_y = res["dev"]["y_raw"] - y_dev
            e_g = res["dev"]["h"] - g_dev
            res["mae_dev"], res["mae_fit"] = mae_dev, mae_fit
            res["component_stats_dev"] = gen._component_stats(
                res["dev"]["ell_hat"], res["dev"]["s_hat"], res["dev"]["h"], ell_dev, s_dev, g_dev)
            res["k_groups_dev"] = gen._k_group_table(e_y, e_g, k_dev)
            res["k0_subgroups_dev"] = gen._k0_subgroup_table(
                e_y, e_g, k_dev, n_nodes_dev, np.abs(g_dev), g_p90)
            main_table[f"{arm}_s{s}"] = {
                "mae_dev": mae_dev,
                "mae_fit": mae_fit,
                "b_y": res["dev"]["b_y"],
                "gap_dev_minus_fit_g": float(mae_dev["g_raw"] - mae_fit["g_raw"]),
                "gap_dev_minus_fit_y_raw": float(mae_dev["y_raw"] - mae_fit["y_raw"]),
                "component_stats_dev": res["component_stats_dev"],
                "k_groups_dev": res["k_groups_dev"],
                "k0_subgroups_dev": res["k0_subgroups_dev"],
                "steps": STEPS_EXPECTED,
                "curve_seconds_total": manifests[arm][s]["curve_seconds_total"],
                "wall_clock_s": manifests[arm][s]["wall_clock_s"],
                "peak_gpu_memory_mb": manifests[arm][s]["peak_gpu_memory_mb"],
                "soup_state_sha256": manifests[arm][s]["soup_state_sha256"],
            }
            log(
                f"[eval {arm} s{s}] dev y_raw {mae_dev['y_raw']:.5f} g {mae_dev['g_raw']:.5f} | "
                f"fit y_raw {mae_fit['y_raw']:.5f} g {mae_fit['g_raw']:.5f} | "
                f"gap(g) {main_table[f'{arm}_s{s}']['gap_dev_minus_fit_g']:.5f}"
            )

    # (3) paired deltas vs CTRL288 (dev + fit) and the historical anchors
    deltas: dict[str, Any] = {}
    for metric, table_key in (("y", "y_raw"), ("g", "g_raw")):
        deltas[f"delta_{metric}_dev"] = {
            str(s): float(results[CANDIDATE][s]["mae_dev"][table_key] - results[CONTROL][s]["mae_dev"][table_key])
            for s in SEEDS
        }
        deltas[f"delta_{metric}_fit"] = {
            str(s): float(results[CANDIDATE][s]["mae_fit"][table_key] - results[CONTROL][s]["mae_fit"][table_key])
            for s in SEEDS
        }
    deltas["definition"] = (
        "Delta = MAE(B72) - MAE(CTRL288), same-seed paired, full dev (negative = improvement); "
        "fit side reported separately, never as a ratio in place of the dev error"
    )
    deltas["gap_table"] = {
        f"{arm}_s{s}": main_table[f"{arm}_s{s}"]["gap_dev_minus_fit_g"] for arm in ARMS for s in SEEDS
    }
    reproduction = {
        str(s): {
            "ctrl288_dev_y_raw": results[CONTROL][s]["mae_dev"]["y_raw"],
            "ctrl288_dev_g_raw": results[CONTROL][s]["mae_dev"]["g_raw"],
            "historical_soup_dev_y_raw": float(old_terminal["main_table"][f"DICT_s{s}"]["mae"]["y_raw"]),
            "historical_soup_dev_g_raw": float(old_terminal["main_table"][f"DICT_s{s}"]["mae"]["g_raw"]),
            "gap_y_raw": float(results[CONTROL][s]["mae_dev"]["y_raw"] - old_terminal["main_table"][f"DICT_s{s}"]["mae"]["y_raw"]),
            "gap_g_raw": float(results[CONTROL][s]["mae_dev"]["g_raw"] - old_terminal["main_table"][f"DICT_s{s}"]["mae"]["g_raw"]),
        }
        for s in SEEDS
    }
    b72_vs_history = {
        str(s): {
            "b72_dev_y_raw": results[CANDIDATE][s]["mae_dev"]["y_raw"],
            "historical_soup_dev_y_raw": float(old_terminal["main_table"][f"DICT_s{s}"]["mae"]["y_raw"]),
            "gap_y_raw": float(results[CANDIDATE][s]["mae_dev"]["y_raw"] - old_terminal["main_table"][f"DICT_s{s}"]["mae"]["y_raw"]),
            "note": "descriptive only: THIS round's recipe effect is judged against the same-round paired CTRL288",
        }
        for s in SEEDS
    }

    # (4) group-paired bootstrap (2000 draws, seed 20261007, shared picks)
    errs_y = {
        f"{arm}_s{s}": np.abs(y_dev - results[arm][s]["dev"]["y_raw"]) for arm in ARMS for s in SEEDS
    }
    errs_g = {
        f"{arm}_s{s}": np.abs(g_dev - results[arm][s]["dev"]["h"]) for arm in ARMS for s in SEEDS
    }
    bootstrap = paired_bootstrap_two_metrics(errs_y, errs_g, smiles[dev_idx])

    # (5) noise bound eta: repeated forwards + same-weight save/reload replay
    #     (NO raw/decoded identity hook this round — the located wiring bug of
    #     the source round is avoided by construction: the decode flag, when
    #     used at all, is a factory argument, never a wrapper attribute)
    eta = 0.0
    noise: dict[str, Any] = {}
    for arm in ARMS:
        for s in SEEDS:
            repeat = _score_run(arm, s, ctx, out_dir=out_dir, device=device, q_fit=q_fit, q_dev=q_dev)
            d_repeat = float(np.max(np.abs(
                np.asarray(repeat["dev"]["y_raw"], np.float64)
                - np.asarray(results[arm][s]["dev"]["y_raw"], np.float64))))
            del repeat["model"], repeat
            model_disk, _state = _load_soup_model(arm, s, ctx, out_dir=out_dir)
            model_disk = model_disk.to(device).eval()
            preds_disk = gen._predict_rows_checked(model_disk, ctx["dev_data"], device)
            y_raw_disk = preds_disk["ell_hat"] + preds_disk["s_hat"] + q_dev  # frozen Q reused bitwise
            d_reload = float(np.max(np.abs(
                np.asarray(y_raw_disk, np.float64)
                - np.asarray(results[arm][s]["dev"]["y_raw"], np.float64))))
            noise[f"{arm}_s{s}"] = {
                "repeat_max_abs_dpred": d_repeat,
                "save_reload_max_abs_dpred": d_reload,
            }
            eta = max(eta, d_repeat, d_reload)
            del model_disk
            if device.type == "cuda":
                torch.cuda.empty_cache()
    marker = float(max(REPLAY_TOL, 10.0 * eta))

    # (6) the unique alpha-mean intervention on ALL FOUR soups
    mean_codes = repl.fit_mean_codes(ctx["basis"], fit_idx)
    historical_mean_codes = old_terminal.get("mean_codes", {})
    if historical_mean_codes:
        for key in ("fit_rel_err_median", "fit_rel_err_p95", "fit_alpha_nnz"):
            if key in historical_mean_codes and abs(
                float(mean_codes[key]) - float(historical_mean_codes[key])
            ) > 1e-9:
                raise RuntimeError(f"fit mean codes recomputed differently at {key}")
    interventions: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for s in SEEDS:
            base = {
                "y": y_dev, "g": g_dev,
                "y_raw": results[arm][s]["dev"]["y_raw"],
                "h": results[arm][s]["dev"]["h"],
                "q_raw": q_dev, "b_y": results[arm][s]["dev"]["b_y"],
            }
            row = alpha_mean_intervention(
                arm, s, base, mean_codes["mean_alpha"], ctx=ctx, out_dir=out_dir,
                device=device, marker=marker, log=log,
            )
            row["response_beyond_noise"] = bool(row["abs_dpred"]["p95"] > marker)
            interventions[arm][int(s)] = row
            del results[arm][s]["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    responsive_b72 = bool(all(interventions[CANDIDATE][s]["response_beyond_noise"] for s in SEEDS))
    intervention_summary = {
        arm: {
            str(s): {
                "abs_dpred_p95": interventions[arm][s]["abs_dpred"]["p95"],
                "response_fraction_gt_marker": interventions[arm][s]["response_fraction_gt_marker"],
                "response_beyond_noise": interventions[arm][s]["response_beyond_noise"],
                "delta_y_raw_mae": interventions[arm][s]["delta_y_raw_mae"],
                "delta_g_mae": interventions[arm][s]["delta_g_mae"],
                "input_phi_hat_abs_change_median": interventions[arm][s]["input_change"]["phi_hat_abs_change_median"],
            }
            for s in SEEDS
        }
        for arm in ARMS
    }

    # (7) the frozen decision branches A-E
    decision = decision_branches(
        {s: deltas["delta_y_dev"][str(s)] for s in SEEDS},
        {s: deltas["delta_g_dev"][str(s)] for s in SEEDS},
        bootstrap,
        float(np.mean([deltas["delta_y_fit"][str(s)] for s in SEEDS])),
        responsive_b72,
        all(checks.values()),
    )

    # (8) cost (actual, no shared-trajectory virtual saving this round: the
    #     two arms are separate trainings by design)
    cost = {
        "per_run": {
            f"{arm}_s{s}": {
                "steps": STEPS_EXPECTED,
                "curve_seconds_total": manifests[arm][s]["curve_seconds_total"],
                "wall_clock_s": manifests[arm][s]["wall_clock_s"],
                "peak_gpu_memory_mb": manifests[arm][s]["peak_gpu_memory_mb"],
                "device": manifests[arm][s]["device"],
                "allocation_probe": manifests[arm][s]["allocation_probe"],
            }
            for arm in ARMS for s in SEEDS
        },
        "total_wall_clock_s": float(sum(
            manifests[arm][s]["wall_clock_s"] for arm in ARMS for s in SEEDS)),
        "b72_minus_ctrl_seconds": {
            str(s): float(
                manifests[CANDIDATE][s]["curve_seconds_total"]
                - manifests[CONTROL][s]["curve_seconds_total"])
            for s in SEEDS
        },
        "parameter_cost": {
            "ctrl288_total": int(repl.EXPECTED_PARAMETERS),
            "b72_total": B72_TOTAL_PARAMETERS,
            "reduction": B72_PARAMETER_REDUCTION,
            "reduction_fraction": float(B72_PARAMETER_REDUCTION / int(repl.EXPECTED_PARAMETERS)),
        },
        "note": (
            "the two arms are separate trainings (no shared trajectory); the measured "
            "seconds are the actual costs; same-seed arms may still land on different "
            "A100 nodes — the allocation probe records the actual regime per run"
        ),
    }

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "terminal-eval",
        "one_shot": True,
        "roster_checks": checks,
        "main_table": main_table,
        "paired_deltas_vs_ctrl288": deltas,
        "historical_reproduction": reproduction,
        "b72_vs_history_descriptive": b72_vs_history,
        "bootstrap": bootstrap,
        "noise_bound": {
            "eta": eta,
            "marker": marker,
            "per_run": noise,
            "method": "repeated forwards + same-weight save/reload replay (no raw/decoded identity hook this round)",
        },
        "interventions": {
            arm: {str(s): interventions[arm][s] for s in SEEDS} for arm in ARMS
        },
        "intervention_summary": intervention_summary,
        "responsive_b72": responsive_b72,
        "mean_codes": {
            "mean_alpha_sha256": hashlib.sha256(
                np.asarray(mean_codes["mean_alpha"], np.float64).tobytes()).hexdigest(),
            "fit_rel_err_median": float(mean_codes["fit_rel_err_median"]),
            "fit_alpha_nnz": float(mean_codes["fit_alpha_nnz"]),
            "hash_consistent_with_historical": True,
        },
        "decision": decision,
        "cost": cost,
        "g_abs_p90_fit": g_p90,
        "scope": (
            "development comparison on the historical dev rows (union of the old "
            "select/confirm), repeatedly used by this line — never an independent "
            "confirm, never official valid/test; the branches are this round's "
            "engineering decision rules, not significance or SOTA"
        ),
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "terminal_eval.json", payload)
    log(
        f"[terminal-eval] branch={decision['branch']} "
        f"mean Delta_y {decision['mean_delta_y']:+.5f} in {payload['seconds']:.0f}s"
    )
    return payload


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="source-checks", choices=(
        "source-checks", "pretrain-checks", "smoke", "train", "terminal-eval"))
    parser.add_argument("--arm", default=CONTROL, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.stage == "source-checks":
        source_checks(out_dir=out_dir)
    elif args.stage == "pretrain-checks":
        pretrain_checks(out_dir=out_dir)
    elif args.stage == "smoke":
        run_smoke(device_name=args.device, out_dir=out_dir)
    elif args.stage == "train":
        train_run(args.arm, args.seed, device_name=args.device, out_dir=out_dir)
    else:
        terminal_eval(device_name=args.device, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
