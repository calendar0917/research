"""The conditional structure-statistics prediction subject (E / L / S).

Frozen mathematics (PROTOCOL.md section 4).  One forward pass encodes the atom,
bond and event objects exactly once; then a *task-conditioned measurement* is
applied either to every raw structural occurrence (E, S) or to the root-aggregate
(L); the root state is read out by one graph-statistics MLP.

Nothing else exists: no message passing, no node-level self-attention, no
recurrent hidden-state update, no pretrained weights, no teacher, no bypass of
any historical model.

Arms
----
``E`` EARLY_TASK   : condition every occurrence, then aggregate at the root.
``L`` LATE_TASK    : aggregate keys/values at the root, then condition once.
``S`` EARLY_SHARED : E with the task-specific query table removed.

E and L are *parameter-identical* and share one canonical initial state
tensor-for-tensor; only the aggregation formula above differs.

Batch tensors consumed
----------------------
``atom_feat`` ``[N,51]``, ``atom_scalar`` ``[N,9]``, ``bond_feat`` ``[M,13]``,
``bond_index`` ``[M,2]`` (batch-local atom ids), ``bond_batch`` ``[M]``,
``atom_batch`` ``[N]``, ``atom_counts`` ``[B]``, ``global`` ``[B,42]``,
``s1_root/s1_end/s1_bond`` ``[O1]``, ``s2_root/s2_middle/s2_end/s2_bond1/s2_bond2``
``[O2]``, ``s2_closure`` ``[O2]``, ``n_atoms`` ``[1]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import css_common as C

SQRT_KEY_DIM = 4.0  # sqrt(16)
GATE_SCALE = 0.5


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(d_in), int(d_hidden)),
        nn.SiLU(),
        nn.Linear(int(d_hidden), int(d_out)),
    )


@dataclass
class ForwardOverrides:
    """Read-only mechanism probes; never used during training."""

    qt_permutation: torch.Tensor | None = None
    context_off: bool = False


@dataclass
class ForwardDiagnostics:
    """Accumulated gate / query statistics (never used as a loss term)."""

    gate_mean_sum: float = 0.0
    gate_std_sum: float = 0.0
    gate_terms: int = 0
    query_norm: float = 0.0
    query_norm_task_spread: float = 0.0

    def add_gate(self, gate: torch.Tensor) -> None:
        with torch.no_grad():
            self.gate_mean_sum += float(gate.mean())
            self.gate_std_sum += float(gate.std()) if gate.numel() > 1 else 0.0
            self.gate_terms += 1

    def summary(self) -> dict[str, float]:
        n = max(self.gate_terms, 1)
        return {
            "gate_mean": self.gate_mean_sum / n,
            "gate_std": self.gate_std_sum / n,
            "query_norm": float(self.query_norm),
            "query_norm_task_spread": float(self.query_norm_task_spread),
        }


class CSSModel(nn.Module):
    """The frozen aggregation-order comparison subject."""

    def __init__(
        self,
        arm: str,
        norm: Mapping[str, Any],
        *,
        n_tasks: int = C.N_TASKS,
        task_chunk: int = C.TASK_CHUNK_DEFAULT,
        atom_dim: int = 51,
        bond_dim: int = 13,
        scalar_dim: int = 9,
        global_dim: int = 42,
        dropout: float = 0.10,
    ):
        super().__init__()
        if arm not in C.ARMS:
            raise KeyError(arm)
        self.arm = str(arm)
        self.n_tasks = int(n_tasks)
        self.task_chunk = int(task_chunk)
        self.dropout_p = float(dropout)
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.scalar_dim = int(scalar_dim)
        self.global_dim = int(global_dim)

        self.atom = _mlp(atom_dim, 256, 192)
        self.bond = _mlp(3 * 192 + bond_dim, 320, 192)
        self.s1_event = _mlp(2 * 192 + 192, 192, 96)
        self.s2_event = _mlp(3 * 192 + 2 * 192 + 1, 256, 96)
        self.key1 = nn.Linear(96, 16)
        self.key2 = nn.Linear(96, 16)
        self.value1 = nn.Linear(96, 32)
        self.value2 = nn.Linear(96, 32)
        self.root_static = nn.Linear(2 * 192 + scalar_dim, 64)
        self.root_stat = nn.Linear(64, 64, bias=False)
        self.ln = nn.LayerNorm(64)
        self.graph = _mlp(3 * 64 + global_dim + 1, 192, 96)
        self.out = nn.Linear(96, n_tasks)

        self.q0 = nn.Parameter(torch.zeros(16))
        if bool(C.ARMS[self.arm]["task_query"]):
            self.qt = nn.Parameter(torch.zeros(n_tasks, 16))
        self.Wc = nn.Linear(global_dim, 16, bias=False)

        # Train-only standardisation buffers (E/L/S share them bit-for-bit).
        self.register_buffer("global_mean", torch.as_tensor(np.asarray(norm["global_mean"]), dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(np.asarray(norm["global_scale"]), dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(np.asarray(norm["scalar_mean"]), dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(np.asarray(norm["scalar_scale"]), dtype=torch.float32))

        # q = 0 and a zero output bias make every arm the same initial function.
        with torch.no_grad():
            self.q0.zero_()
            if hasattr(self, "qt"):
                self.qt.zero_()
            self.Wc.weight.zero_()
            self.out.bias.zero_()

    # -- invariants --------------------------------------------------------
    def parameter_breakdown(self) -> dict[str, int]:
        groups: dict[str, tuple[str, ...]] = {
            "atom": ("atom",),
            "bond": ("bond",),
            "s1_event": ("s1_event",),
            "s2_event": ("s2_event",),
            "keys": ("key1", "key2"),
            "values": ("value1", "value2"),
            "root_static": ("root_static",),
            "root_stat": ("root_stat",),
            "layer_norm": ("ln",),
            "graph_mlp": ("graph",),
            "task_output": ("out",),
            "query": ("q0", "qt", "Wc"),
        }
        out: dict[str, int] = {}
        for label, prefixes in groups.items():
            total = 0
            for name, p in self.named_parameters():
                if any(name == pre or name.startswith(pre + ".") for pre in prefixes):
                    total += int(p.numel())
            out[label] = total
        return out

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def active_tasks(self) -> int:
        return self.n_tasks

    # -- public forward ----------------------------------------------------
    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        dropout_mask: torch.Tensor | None = None,
        overrides: ForwardOverrides | None = None,
        diagnostics: ForwardDiagnostics | None = None,
    ) -> torch.Tensor:
        enc = self.encode(batch)
        q, query_diag = self.build_query(batch, overrides=overrides)
        if diagnostics is not None and query_diag:
            diagnostics.query_norm = float(query_diag["query_norm"])
            diagnostics.query_norm_task_spread = float(query_diag["query_norm_task_spread"])
        return self.readout(enc, batch, q, dropout_mask=dropout_mask, diagnostics=diagnostics)

    # -- 4.1 one-shot public front end ------------------------------------
    def encode(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        a = self.atom(batch["atom_feat"])
        bi = batch["bond_index"]
        au = a.index_select(0, bi[:, 0])
        av = a.index_select(0, bi[:, 1])
        bond_in = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=-1)
        b = self.bond(bond_in)

        s1_root = batch["s1_root"]
        s1_end = batch["s1_end"]
        r1 = torch.cat(
            [a.index_select(0, s1_root), a.index_select(0, s1_end), b.index_select(0, batch["s1_bond"])],
            dim=-1,
        )
        u1 = self.s1_event(r1)
        k1 = torch.tanh(self.key1(u1))
        v1 = self.value1(u1)

        r2 = torch.cat(
            [
                a.index_select(0, batch["s2_root"]),
                a.index_select(0, batch["s2_middle"]),
                a.index_select(0, batch["s2_end"]),
                b.index_select(0, batch["s2_bond1"]),
                b.index_select(0, batch["s2_bond2"]),
                batch["s2_closure"].unsqueeze(-1),
            ],
            dim=-1,
        )
        u2 = self.s2_event(r2)
        k2 = torch.tanh(self.key2(u2))
        v2 = self.value2(u2)
        return {"a": a, "b": b, "k1": k1, "v1": v1, "k2": k2, "v2": v2}

    # -- 4.2 query ---------------------------------------------------------
    def standardised_global(self, batch: Mapping[str, Any]) -> torch.Tensor:
        g = batch["global"]
        return (g - self.global_mean) / self.global_scale

    def build_query(
        self,
        batch: Mapping[str, Any],
        *,
        overrides: ForwardOverrides | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``[B, T, 16]`` (E/L) or ``[B, 1, 16]`` (S) query table."""
        c = self.standardised_global(batch)
        context = self.Wc(c)  # [B,16]
        if overrides is not None and overrides.context_off:
            context = torch.zeros_like(context)
        base = self.q0.unsqueeze(0) + context  # [B,16]
        if hasattr(self, "qt"):
            qt = self.qt
            if overrides is not None and overrides.qt_permutation is not None:
                qt = qt.index_select(0, overrides.qt_permutation.to(qt.device))
            q = torch.tanh(base.unsqueeze(1) + qt.unsqueeze(0))  # [B,T,16]
            with torch.no_grad():
                norms = q.norm(dim=-1)  # [B,T]
                spread = float(norms.mean(dim=0).std()) if q.shape[1] > 1 else 0.0
            return q, {"query_norm": float(norms.mean()), "query_norm_task_spread": spread}
        shared = torch.tanh(base.unsqueeze(1))  # [B,1,16]
        q = shared.expand(-1, self.n_tasks, -1)
        with torch.no_grad():
            return q, {
                "query_norm": float(shared.norm(dim=-1).mean()),
                "query_norm_task_spread": 0.0,
            }

    # -- 4.3 the only difference between E and L ---------------------------
    def root_sums(self, enc: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Root-level key/value sums used by L (and by the E root scatter)."""
        n_atoms = int(enc["a"].shape[0])
        out: dict[str, torch.Tensor] = {}
        for cls in ("1", "2"):
            k = enc["k" + cls]
            v = enc["v" + cls]
            ksum = torch.zeros(n_atoms, k.shape[1], device=k.device, dtype=k.dtype)
            vsum = torch.zeros(n_atoms, v.shape[1], device=v.device, dtype=v.dtype)
            if k.shape[0]:
                ksum.index_add_(0, batch["s%s_root" % cls], k)
                vsum.index_add_(0, batch["s%s_root" % cls], v)
            out["ksum" + cls] = ksum
            out["vsum" + cls] = vsum
        return out

    def condition_and_aggregate(
        self,
        enc: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        q: torch.Tensor,
        sums: Mapping[str, torch.Tensor],
        *,
        diagnostics: ForwardDiagnostics | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_atoms = int(enc["a"].shape[0])
        n_tasks = int(q.shape[1])
        device = enc["k1"].device
        dtype = enc["k1"].dtype

        def _z(cls: str) -> torch.Tensor:
            k = enc["k" + cls]
            v = enc["v" + cls]
            root = batch["s%s_root" % cls]
            counts = batch["s%s_counts" % cls]
            d = torch.sqrt(counts.clamp(min=1).to(dtype)).view(-1, 1, 1)
            if self.arm in ("E", "S"):
                graph_of = batch["atom_batch"].index_select(0, root)
                q_p = q.index_select(0, graph_of)  # [O,T,16]
                s = torch.einsum("ok,otk->ot", k, q_p) / SQRT_KEY_DIM
                gate = 1.0 + GATE_SCALE * torch.tanh(s)  # [O,T]
                weighted = gate.unsqueeze(-1) * v.unsqueeze(1)  # [O,T,32]
                acc = torch.zeros(n_atoms, n_tasks, 32, device=device, dtype=dtype)
                if weighted.shape[0]:
                    acc.index_add_(0, root, weighted)
                z = acc / d
            else:
                cnt = counts.clamp(min=1).to(dtype).unsqueeze(-1)
                kbar = sums["ksum" + cls] / cnt  # [N,16]
                vsum = sums["vsum" + cls]  # [N,32]
                q_atom = q.index_select(0, batch["atom_batch"])  # [N,T,16]
                s = torch.einsum("nk,ntk->nt", kbar, q_atom) / SQRT_KEY_DIM
                gate = 1.0 + GATE_SCALE * torch.tanh(s)  # [N,T]
                z = gate.unsqueeze(-1) * vsum.unsqueeze(1) / d
            if diagnostics is not None:
                diagnostics.add_gate(gate)
            return z

        return _z("1"), _z("2")

    # -- 4.4 root representation ------------------------------------------
    def root_input(self, enc: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> torch.Tensor:
        bsum = torch.zeros_like(enc["a"])
        if batch["s1_root"].numel():
            bsum.index_add_(0, batch["s1_root"], enc["b"].index_select(0, batch["s1_bond"]))
        scalar = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return torch.cat([enc["a"], bsum, scalar], dim=-1)

    # -- 4.5 graph statistics and output ----------------------------------
    def readout(
        self,
        enc: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        q: torch.Tensor,
        *,
        dropout_mask: torch.Tensor | None = None,
        diagnostics: ForwardDiagnostics | None = None,
    ) -> torch.Tensor:
        n_atoms = int(enc["a"].shape[0])
        n_graphs = int(batch["atom_counts"].numel())
        n_tasks = int(q.shape[1])
        device = enc["a"].device
        dtype = enc["a"].dtype
        logits = torch.zeros(n_graphs, n_tasks, device=device, dtype=dtype)
        if n_atoms == 0 or n_graphs == 0:
            return logits

        sums = self.root_sums(enc, batch) if self.arm == "L" else {}
        c = self.standardised_global(batch)
        x = self.root_input(enc, batch)
        j_static = self.root_static(x)  # [N,64]
        counts = batch["atom_counts"].to(dtype).view(-1, 1, 1)
        sqrt_counts = torch.sqrt(counts)
        log_n = torch.log1p(batch["atom_counts"].to(dtype)).view(-1, 1, 1)  # [B,1,1]

        for start in range(0, n_tasks, self.task_chunk):
            stop = min(n_tasks, start + self.task_chunk)
            qc = q[:, start:stop, :]
            z1, z2 = self.condition_and_aggregate(enc, batch, qc, sums, diagnostics=diagnostics)
            zcat = torch.cat([z1, z2], dim=-1)  # [N,T,64]
            j = j_static.unsqueeze(1) + self.root_stat(zcat)  # [N,T,64]
            h = self.ln(F.silu(j))
            if dropout_mask is not None:
                h = h * dropout_mask[:, start:stop, :]
            hsum = torch.zeros(n_graphs, stop - start, 64, device=device, dtype=dtype)
            hsum.index_add_(0, batch["atom_batch"], h)
            hsq = torch.zeros_like(hsum)
            hsq.index_add_(0, batch["atom_batch"], h * h)
            hmean = hsum / counts
            hsq_mean = hsq / counts
            p = torch.cat(
                [
                    hsum / sqrt_counts,
                    hmean,
                    hsq_mean,
                    c.unsqueeze(1).expand(n_graphs, stop - start, self.global_dim),
                    log_n.expand(n_graphs, stop - start, 1),
                ],
                dim=-1,
            )  # [B,T,235]
            r = self.graph(p.reshape(n_graphs * (stop - start), -1)).reshape(n_graphs, stop - start, 96)
            w = self.out.weight[start:stop]
            b = self.out.bias[start:stop]
            logits[:, start:stop] = torch.einsum("btk,tk->bt", r, w) + b.unsqueeze(0)
        return logits


# ---------------------------------------------------------------------------
# canonical initial state
# ---------------------------------------------------------------------------


_CANONICAL: dict[str, dict[str, torch.Tensor]] = {}

#: Only the *shapes* of the normalisation buffers matter for construction; the
#: real Train-only values are installed on every arm from the frozen task norm.
_FROZEN_NORM_PLACEHOLDER: dict[str, Any] = {
    "global_mean": np.zeros(42, dtype=np.float32),
    "global_scale": np.ones(42, dtype=np.float32),
    "scalar_mean": np.zeros(9, dtype=np.float32),
    "scalar_scale": np.ones(9, dtype=np.float32),
}

QUERY_PARAMETER_NAMES = ("q0", "qt", "Wc.weight")


def canonical_init_state(n_tasks: int = C.N_TASKS) -> dict[str, torch.Tensor]:
    """The one canonical seed-0 initial state every arm copies tensor-for-tensor."""
    key = f"n{int(n_tasks)}"
    if key in _CANONICAL:
        return {k: v.clone() for k, v in _CANONICAL[key].items()}
    rng = torch.get_rng_state().clone()
    try:
        torch.manual_seed(int(C.CANONICAL_INIT_SEED))
        reference = CSSModel("E", _FROZEN_NORM_PLACEHOLDER, n_tasks=int(n_tasks))
        with torch.no_grad():
            reference.q0.zero_()
            reference.qt.zero_()
            reference.Wc.weight.zero_()
            reference.out.bias.zero_()
        full = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    finally:
        torch.set_rng_state(rng)
    _CANONICAL[key] = full
    return {k: v.clone() for k, v in full.items()}


def build_model(
    arm: str,
    norm: Mapping[str, Any],
    *,
    n_tasks: int = C.N_TASKS,
    task_chunk: int = C.TASK_CHUNK_DEFAULT,
) -> tuple[CSSModel, dict[str, Any]]:
    """Construct one arm and copy the canonical initial *parameters* into it."""
    model = CSSModel(arm, norm, n_tasks=n_tasks, task_chunk=task_chunk)
    canon = canonical_init_state(n_tasks)
    missing = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in canon:
                p.copy_(canon[name].to(dtype=p.dtype))
            else:
                missing.append(name)
    # the only legal parameter without a canonical value is the S arm's absent qt
    if missing not in ([], ["qt"]):
        raise RuntimeError(f"unexpected parameters without a canonical value: {missing}")
    with torch.no_grad():
        model.q0.zero_()
        if hasattr(model, "qt"):
            model.qt.zero_()
        model.Wc.weight.zero_()
        model.out.bias.zero_()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    shared_fp = C.state_fingerprint({k: v for k, v in state.items() if k not in QUERY_PARAMETER_NAMES})
    audit = {
        "arm": str(arm),
        "arm_name": C.ARMS[arm]["name"],
        "parameters": model.n_parameters(),
        "parameter_breakdown": model.parameter_breakdown(),
        "canonical_init_seed": int(C.CANONICAL_INIT_SEED),
        "canonical_shared_state_fingerprint": shared_fp,
        "has_task_query_table": bool(hasattr(model, "qt")),
        "n_state_tensors": int(len(state)),
    }
    return model, audit


def audit_initial_states(norm: Mapping[str, Any], *, n_tasks: int = C.N_TASKS) -> dict[str, Any]:
    """E/L/S initial-state, parameter-count and query-zero audit."""
    models: dict[str, CSSModel] = {}
    audits: dict[str, dict[str, Any]] = {}
    for arm in C.ARM_ORDER:
        m, a = build_model(arm, norm, n_tasks=n_tasks)
        models[arm] = m
        audits[arm] = a
    e = models["E"].state_dict()
    l = models["L"].state_dict()
    s = models["S"].state_dict()
    e_keys, l_keys, s_keys = set(e), set(l), set(s)
    per_tensor_equal_E_L = all(torch.equal(e[k], l[k]) for k in sorted(e_keys))
    shared_equal = all(torch.equal(e[k], s[k]) for k in sorted(s_keys))
    query_zero = {
        "q0": float(models["E"].q0.detach().abs().max()),
        "qt": float(models["E"].qt.detach().abs().max()),
        "Wc": float(models["E"].Wc.weight.detach().abs().max()),
        "out_bias": float(models["E"].out.bias.detach().abs().max()),
    }
    failures: list[str] = []
    for arm in ("E", "L"):
        if audits[arm]["parameters"] != C.EXPECTED_PARAMETERS[arm]:
            failures.append(f"{arm} parameters {audits[arm]['parameters']} != {C.EXPECTED_PARAMETERS[arm]}")
        if audits[arm]["parameter_breakdown"] != C.PARAMETER_BREAKDOWN:
            failures.append(f"{arm} parameter breakdown mismatch: {audits[arm]['parameter_breakdown']}")
    if audits["S"]["parameters"] != C.EXPECTED_PARAMETERS["S"]:
        failures.append(f"S parameters {audits['S']['parameters']} != {C.EXPECTED_PARAMETERS['S']}")
    if e_keys != l_keys:
        failures.append("E/L state-dict key sets differ")
    if s_keys - e_keys:
        failures.append(f"S has unexpected keys {sorted(s_keys - e_keys)}")
    if e_keys - s_keys != {"qt"}:
        failures.append(f"S is missing keys other than qt: {sorted(e_keys - s_keys)}")
    if not per_tensor_equal_E_L:
        failures.append("E and L initial states are not tensor-identical")
    if not shared_equal:
        failures.append("S public initial state differs from E")
    if any(v != 0.0 for v in query_zero.values()):
        failures.append(f"query/output-bias are not zero: {query_zero}")
    if len({audits[a]["canonical_shared_state_fingerprint"] for a in C.ARM_ORDER}) != 1:
        failures.append("arms do not share one canonical public initial fingerprint")
    return {
        "passed": not failures,
        "failures": failures,
        "E_L_initial_state_tensor_identical": bool(per_tensor_equal_E_L),
        "S_public_state_equal_to_E": bool(shared_equal),
        "S_missing_keys": sorted(e_keys - s_keys),
        "E_L_state_keys_identical": bool(e_keys == l_keys),
        "query_and_output_bias_zero_at_init": query_zero,
        "parameter_audit": {
            a: {"parameters": audits[a]["parameters"], "breakdown": audits[a]["parameter_breakdown"]}
            for a in C.ARM_ORDER
        },
        "canonical_shared_state_fingerprint": audits["E"]["canonical_shared_state_fingerprint"],
        "expected_parameters": dict(C.EXPECTED_PARAMETERS),
    }


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------


def predict_logits(
    model: CSSModel,
    collate_fn,
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 128,
    overrides: ForwardOverrides | None = None,
    diagnostics: ForwardDiagnostics | None = None,
) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate_fn(chunk).items()
            }
            logits = model(batch, overrides=overrides, diagnostics=diagnostics)
            out.append(logits.detach().cpu().numpy().astype(np.float32))
    if not out:
        return np.zeros((0, model.n_tasks), dtype=np.float32)
    logits = np.concatenate(out, axis=0)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite predictions")
    return logits


def sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


__all__ = [
    "CSSModel",
    "ForwardDiagnostics",
    "ForwardOverrides",
    "audit_initial_states",
    "build_model",
    "canonical_init_state",
    "predict_logits",
    "sigmoid",
]
