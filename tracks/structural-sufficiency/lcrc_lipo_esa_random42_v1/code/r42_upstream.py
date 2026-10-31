"""Runtime compatibility layer for the byte-identical upstream ESA/PNA snapshot.

The upstream snapshot (commit ``861a0b12…``) assumes an environment with
PyTorch Lightning, bitsandbytes, flash-attn, admin-torch and xformers.  Those
packages are not part of this repository's environment (and the Slurm cluster
is offline), so this module installs *minimal, documented* stubs before the
upstream modules are imported:

===================================  ==========================================
missing package                      substitute
===================================  ==========================================
``pytorch_lightning.LightningModule`` ``torch.nn.Module`` (only the model
                                     classes are used; no Lightning Trainer)
``bitsandbytes.optim.AdamW8bit``      never called; a placeholder that raises
                                     on construction.  The formal optimiser is
                                     ``torch.optim.AdamW`` (recorded).
``flash_attn.ops.activations.swiglu`` ``F.silu(x) * y`` (exact definition)
``admin_torch``                       the published 0.1.0 ``OmegaResidual``
                                     formula, reproduced verbatim
``xformers.ops.memory_efficient_attention``
                                     raises if called; formal runs use the
                                     upstream ``torch`` SDPA backend
``utils.reporting``                  metric helpers raising if called (they are
                                     only used by Lightning logging)
``utils.posenc_encoders.*``, ``data_loading.posenc``
                                     placeholders; this round uses no
                                     positional encoding
===================================  ==========================================

One semantic fix is applied at runtime to the upstream ``torch`` attention
backend: upstream builds a boolean mask whose ``True`` entries are the
**complement** of the adjacent-edge set, while ``torch.nn.functional.
scaled_dot_product_attention`` interprets ``True`` as "participate".  The
wrapper inverts that mask back, making the ``torch`` backend equivalent to the
upstream xFormers additive-bias convention.  See ``tests/test_mask_semantics.py``
for the executable proof.  The vendored source files themselves are not
modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

import r42_common as C

_INSTALLED = False
_MODULES: dict[str, Any] = {}
PATCHES: list[dict[str, Any]] = []

STUB_MANIFEST = {
    "pytorch_lightning": "LightningModule -> torch.nn.Module; no Trainer is used",
    "bitsandbytes": "AdamW8bit placeholder (raising); torch.optim.AdamW is the formal optimiser",
    "flash_attn": "swiglu(x, y) = F.silu(x) * y",
    "admin_torch": "OmegaResidual / calculate_init reproduced from admin-torch 0.1.0",
    "xformers": "memory_efficient_attention raises; formal backend is the upstream torch SDPA path",
    "utils.reporting": "metric helpers raise; only Lightning logging used them",
    "utils.posenc_encoders": "placeholder encoders raise; positional encoding is disabled",
    "data_loading.posenc": "compute_posenc_stats raises; positional encoding is disabled",
}


def _module(name: str) -> ModuleType:
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# admin_torch 0.1.0 (exact formula)
# ---------------------------------------------------------------------------


def _calculate_init(num_res_layers: int, output_change_scale: str = "default") -> float:
    import math

    if output_change_scale in ("O(logn)", "default"):
        omega_value = (num_res_layers + 1) / math.log(num_res_layers + 1) - 1
    elif output_change_scale == "O(n)":
        omega_value = 1.0
    else:
        assert output_change_scale == "O(1)"
        omega_value = num_res_layers
    return omega_value**0.5


class OmegaResidual(nn.Module):
    """Reproduction of ``admin_torch.admin.OmegaResidual`` (0.1.0, MIT)."""

    def __init__(self, init_value: float, as_parameter: bool = False, embed_dim: int | None = None):
        super().__init__()
        if as_parameter:
            assert embed_dim is not None
            self.omega = nn.Parameter(torch.ones(embed_dim))
            self.omega.data.fill_(init_value)
        else:
            self.register_buffer("omega", torch.FloatTensor([init_value]))

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        if float(self.omega) == 1.0:
            return x + f_x
        return x * self.omega + f_x


def as_module(num_res_layers: int, output_change_scale: str = "default", as_parameter: bool = False, embed_dim=None):
    return OmegaResidual(_calculate_init(num_res_layers, output_change_scale), as_parameter=as_parameter, embed_dim=embed_dim)


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    pl = _module("pytorch_lightning")
    pl.LightningModule = nn.Module  # type: ignore[attr-defined]

    bnb = _module("bitsandbytes")
    bnb_optim = _module("bitsandbytes.optim")
    class _AdamW8bitUnavailable:  # noqa: N801 - mirrors the upstream name
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "bitsandbytes.AdamW8bit is unavailable in this environment; "
                "the formal runs use torch.optim.AdamW (recorded in the implementation manifest)"
            )
    bnb_optim.AdamW8bit = _AdamW8bitUnavailable  # type: ignore[attr-defined]
    bnb.optim = bnb_optim  # type: ignore[attr-defined]

    fa = _module("flash_attn")
    fa_ops = _module("flash_attn.ops")
    fa_act = _module("flash_attn.ops.activations")

    def swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.silu(x) * y

    fa_act.swiglu = swiglu  # type: ignore[attr-defined]
    fa.ops = fa_ops  # type: ignore[attr-defined]
    fa_ops.activations = fa_act  # type: ignore[attr-defined]

    adm = _module("admin_torch")
    adm.OmegaResidual = OmegaResidual  # type: ignore[attr-defined]
    adm.as_module = as_module  # type: ignore[attr-defined]
    adm.calculate_init = _calculate_init  # type: ignore[attr-defined]

    xf = _module("xformers")
    xf_ops = _module("xformers.ops")

    def memory_efficient_attention(*args, **kwargs):
        raise NotImplementedError(
            "xformers is not installed; this round uses the upstream torch SDPA backend"
        )

    xf_ops.memory_efficient_attention = memory_efficient_attention  # type: ignore[attr-defined]
    xf.ops = xf_ops  # type: ignore[attr-defined]

    reporting = _module("utils.reporting")

    def _unused_metric(*args, **kwargs):
        raise NotImplementedError("upstream reporting helpers are only used by the Lightning Trainer")

    for name in (
        "get_regr_metrics_pt",
        "get_cls_metrics_binary_pt",
        "get_cls_metrics_multilabel_pt",
        "get_cls_metrics_multiclass_pt",
    ):
        setattr(reporting, name, _unused_metric)

    pe_pkg = _module("utils.posenc_encoders")
    for name, cls_name in (
        ("utils.posenc_encoders.laplace_pos_encoder", "LapPENodeEncoder"),
        ("utils.posenc_encoders.kernel_pos_encoder", "KernelPENodeEncoder"),
    ):
        mod = _module(name)
        setattr(mod, cls_name, type(cls_name, (), {"__init__": lambda self, *a, **k: (_ for _ in ()).throw(NotImplementedError("positional encoding is disabled"))}))
        setattr(pe_pkg, name.rsplit(".", 1)[-1], mod)

    posenc = _module("data_loading.posenc")

    def compute_posenc_stats(*args, **kwargs):
        raise NotImplementedError("positional encoding is disabled in this round")

    posenc.compute_posenc_stats = compute_posenc_stats  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# import + documented mask patch
# ---------------------------------------------------------------------------


def _install_mask_patch(masked_layers: ModuleType) -> None:
    global PATCHES

    def make(original):
        def patched(*args, **kwargs):
            # Upstream hard-codes device="cuda:0"; derive it from the graph so
            # the same code can run the CPU gate and any GPU index.
            if "device" not in kwargs:
                edge_index = kwargs.get("edge_index", args[0] if args else None)
                if torch.is_tensor(edge_index):
                    kwargs["device"] = edge_index.device
            mask = original(*args, **kwargs)
            backend = kwargs.get("xformers_or_torch_attn")
            if backend is None:
                backend = args[4]
            if backend == "torch":
                # Upstream returns ~allowed for the torch backend; invert it
                # back so True == "participate", matching SDPA and xformers.
                mask = ~mask
            return mask

        patched.__name__ = original.__name__
        patched._r42_patched = True
        return patched

    for fn in ("get_adj_mask_from_edge_index_edge", "get_adj_mask_from_edge_index_node"):
        original = getattr(masked_layers, fn)
        patched = make(original)
        setattr(masked_layers, fn, patched)
        PATCHES.append(
            {
                "file": "code/upstream/esa/masked_layers.py",
                "function": fn,
                "kind": "runtime_mask_semantics_fix",
                "reason": (
                    "upstream torch backend returns the complement of the allowed set; "
                    "SDPA bool masks mark True as participate, so the mask is inverted once"
                ),
                "effect": "torch backend now matches the upstream xformers additive-bias semantics",
            }
        )
    def get_first_unique_index(t: torch.Tensor) -> torch.Tensor:
        unique, idx, counts = torch.unique(t, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        zero = torch.tensor([0], device=t.device)
        cum_sum = torch.cat((zero, cum_sum[:-1]))
        return ind_sorted[cum_sum]

    def generate_consecutive_tensor(t: torch.Tensor, final: int) -> torch.Tensor:
        lengths = t[1:] - t[:-1]
        lengths = torch.cat((lengths, torch.tensor([final - t[-1]], device=t.device)))
        return torch.cat([torch.arange(0, int(length), device=t.device) for length in lengths])

    masked_layers.get_first_unique_index = get_first_unique_index
    masked_layers.generate_consecutive_tensor = generate_consecutive_tensor
    PATCHES.append(
        {
            "file": "code/upstream/esa/masked_layers.py",
            "function": "get_adj_mask_from_edge_index_edge/node",
            "kind": "runtime_device_fix",
            "reason": "upstream default device is 'cuda:0'; the wrapper derives it from edge_index",
            "effect": "identical GPU behaviour for cuda:0; enables the CPU implementation gate",
        }
    )
    PATCHES.append(
        {
            "file": "code/upstream/esa/masked_layers.py",
            "function": "get_first_unique_index/generate_consecutive_tensor",
            "kind": "runtime_device_fix",
            "reason": "upstream hard-codes torch.device('cuda:0') inside the mask helpers",
            "effect": "same values on cuda:0; no artificial CUDA requirement on the CPU gate",
        }
    )


def install() -> dict[str, Any]:
    """Install stubs, import the vendored upstream modules, apply the mask fix."""
    global _INSTALLED
    if _INSTALLED:
        return _MODULES
    vendor = str(C.VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    _install_stubs()
    import contextlib  # noqa: E402

    import data_loading.transforms as transforms  # noqa: E402
    import esa.masked_layers as masked_layers  # noqa: E402
    import esa.mha as esa_mha  # noqa: E402
    import esa.models as esa_models  # noqa: E402
    import gnn.graph_models as graph_models  # noqa: E402
    import utils.batch_renorm as batch_renorm  # noqa: E402
    import utils.norm_layers as norm_layers  # noqa: E402

    _install_mask_patch(masked_layers)

    # CPU implementation gate: the upstream forward forces the EFFICIENT_ATTENTION
    # kernel, which has no CPU implementation for a boolean mask.  On CPU the
    # math kernel is selected instead; on CUDA the upstream context is untouched.
    _orig_sdpa_kernel = esa_mha.sdpa_kernel

    def sdpa_kernel_compat(*args, **kwargs):
        if not torch.cuda.is_available():
            return contextlib.nullcontext()
        return _orig_sdpa_kernel(*args, **kwargs)

    esa_mha.sdpa_kernel = sdpa_kernel_compat
    PATCHES.append(
        {
            "file": "code/upstream/esa/mha.py",
            "function": "sdpa_kernel",
            "kind": "runtime_cpu_kernel_fix",
            "reason": "EFFICIENT_ATTENTION has no CPU kernel for a boolean mask",
            "effect": "CPU gate uses the math kernel; CUDA keeps the upstream efficient kernel",
        }
    )
    _MODULES.update(
        {
            "transforms": transforms,
            "masked_layers": masked_layers,
            "esa_models": esa_models,
            "graph_models": graph_models,
            "batch_renorm": batch_renorm,
            "norm_layers": norm_layers,
        }
    )
    _INSTALLED = True
    return _MODULES


def upstream_file_hashes() -> dict[str, str]:
    out = {}
    for path in sorted(C.VENDOR.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(C.VENDOR))] = C.sha256_file(path)
    return out


__all__ = ["install", "upstream_file_hashes", "STUB_MANIFEST", "PATCHES", "OmegaResidual", "as_module", "_calculate_init"]
