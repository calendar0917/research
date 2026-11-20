"""Frozen-RESP dictionary stage: words, environment panels, three dictionaries.

No optimisation epoch is introduced.  The frozen best-Val RESP model runs once
in ``eval()`` mode over every unique fragment; ``z, a(S), q(S), b(C), k(C)`` and
the per-view scores are cached.  Building the response panel, the three
dictionaries and their predictions is then pure linear algebra, so the encoder
is never re-run per "environment x candidate fragment".

Train-only material: word candidates, reference ``D0``, environment panels,
prototype selection, ``alpha`` atoms.  A Val/Test fragment may use *its own
structure* (its response under the frozen model) to obtain ``alpha``; its label
is never used and no Val/Test prototype is ever added.

Compatibility note (recorded, not assumed away): the frozen model scores a view
as ``b(C) + a(S) + T(C,S)``, i.e. it consumes the two branch descriptors and the
cut-bond attributes and never re-splices the molecule.  Every ``(C_j, S)`` pair
inside one connection-type class is therefore a legal model input by
construction; there is no re-embedding step that could fail.  Consequently
``DICTIONARY_UNSUPPORTED`` counts only classes with no usable public panel, and
the report states this rather than implying a chemical re-join was validated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import rd_common as RC

#: atom columns defining the connection type tau: atomic number, formal charge,
#: aromatic flag; the three cut-bond columns are appended.
TAU_ATOM_COLUMNS = (0, 3, 7)


def tau_of(frag_atom_feat: np.ndarray, port_local: int, cut_bond_feat: Sequence[int]) -> tuple:
    if int(port_local) < 0:
        raise ValueError("a fragment without a port has no connection type")
    atom = np.asarray(frag_atom_feat)[int(port_local)]
    return (
        *[int(atom[c]) for c in TAU_ATOM_COLUMNS],
        *[int(x) for x in np.asarray(cut_bond_feat).reshape(-1)],
    )


def tau_label(tau: Sequence[int]) -> str:
    names = ("Z", "chg", "arom")
    parts = [f"{names[i]}={int(v)}" for i, v in enumerate(tau[:3])]
    parts += [f"b{i}={int(v)}" for i, v in enumerate(tau[3:])]
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Train-only inventory
# ---------------------------------------------------------------------------


@dataclass
class Inventory:
    keys: list[str]
    tau: dict[int, tuple]
    word_mol_freq: dict[int, int]
    env_mol_freq: dict[int, int]
    class_words: dict[tuple, list[int]]
    class_envs: dict[tuple, list[int]]
    word_fragments: set[int]  # every fragment side (S) in any split


def build_inventory(payload: Mapping[str, Any]) -> Inventory:
    atom_ptr = np.asarray(payload["frag_atom_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["frag_atom_feat"])
    port_local = np.asarray(payload["frag_port_local"])
    keys = payload["frag_key"]
    s_ids = np.asarray(payload["train_view_s"], dtype=np.int64)
    c_ids = np.asarray(payload["train_view_c"], dtype=np.int64)
    bond_feat = np.asarray(payload["train_view_bond_feat"], dtype=np.int64).reshape(-1, 3)
    mol_ptr = np.asarray(payload["train_mol_view_ptr"], dtype=np.int64)

    tau: dict[int, tuple] = {}
    word_fragments: set[int] = set()
    for split in ("train", "valid", "test"):
        s_all = np.asarray(payload[f"{split}_view_s"], dtype=np.int64)
        c_all = np.asarray(payload[f"{split}_view_c"], dtype=np.int64)
        bf_all = np.asarray(payload[f"{split}_view_bond_feat"], dtype=np.int64).reshape(-1, 3)
        for view in range(int(s_all.size)):
            for frag in (int(s_all[view]), int(c_all[view])):
                if frag in tau:
                    continue
                a0, a1 = int(atom_ptr[frag]), int(atom_ptr[frag + 1])
                tau[frag] = tau_of(atom_feat[a0:a1], int(port_local[frag]), bf_all[view])
        word_fragments.update(int(f) for f in s_all.tolist())

    word_mol_freq: dict[int, int] = {}
    env_mol_freq: dict[int, int] = {}
    n_mol = int(mol_ptr.size - 1)
    for mol in range(n_mol):
        words_here: set[int] = set()
        envs_here: set[int] = set()
        for view in range(int(mol_ptr[mol]), int(mol_ptr[mol + 1])):
            words_here.add(int(s_ids[view]))
            envs_here.add(int(c_ids[view]))
        for frag in words_here:
            word_mol_freq[frag] = word_mol_freq.get(frag, 0) + 1
        for frag in envs_here:
            env_mol_freq[frag] = env_mol_freq.get(frag, 0) + 1

    class_words: dict[tuple, list[int]] = {}
    for frag in sorted(word_mol_freq):
        class_words.setdefault(tau[frag], []).append(frag)
    class_envs: dict[tuple, list[int]] = {}
    for view in range(int(s_ids.size)):
        class_envs.setdefault(tau[int(s_ids[view])], set()).add(int(c_ids[view]))
    class_envs = {key: sorted(value) for key, value in class_envs.items()}
    return Inventory(
        keys=keys,
        tau=tau,
        word_mol_freq=word_mol_freq,
        env_mol_freq=env_mol_freq,
        class_words=class_words,
        class_envs=class_envs,
        word_fragments=word_fragments,
    )


# ---------------------------------------------------------------------------
# frozen model cache
# ---------------------------------------------------------------------------


@dataclass
class Frozen:
    z: np.ndarray        # (n_frag, 128)
    a: np.ndarray        # (n_frag,)
    q: np.ndarray        # (n_frag, latent)
    k: np.ndarray        # (n_frag, latent)
    b: np.ndarray        # (n_frag,)
    w: np.ndarray        # (latent,)
    fallback: np.ndarray  # (n_frag,) fallback-head scalar (only used for no-cut molecules)
    per_split: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    seconds: float = 0.0

    def view_scores(self, s_ids: np.ndarray, c_ids: np.ndarray) -> np.ndarray:
        return self.b[c_ids] + self.a[s_ids] + self.fusion(s_ids, c_ids)

    def fusion(self, s_ids: np.ndarray, c_ids: np.ndarray) -> np.ndarray:
        qs = self.q[s_ids]
        kc = self.k[c_ids]
        return (qs * kc * self.w[None, :]).sum(axis=1) / (self.q.shape[1] ** 0.5)


def build_frozen(model, dataset, device: str = "cpu") -> Frozen:
    import torch

    import rd_train as RT

    t0 = time.perf_counter()
    n_frag = len(dataset.frag_data)
    ids = np.arange(n_frag, dtype=np.int64)
    z_rows: list[np.ndarray] = []
    a_rows: list[np.ndarray] = []
    q_rows: list[np.ndarray] = []
    k_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    fb_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_frag, 512):
            chunk = ids[start:start + 512]
            batch = RT.fragment_batch(dataset.frag_data, chunk).to(device)
            z = model.encoder(batch)
            z_rows.append(z.cpu().numpy().astype(np.float64))
            a_rows.append(model.a_head(z).view(-1).cpu().numpy().astype(np.float64))
            q_rows.append(torch.tanh(model.U(z)).cpu().numpy().astype(np.float64))
            k_rows.append(torch.tanh(model.V(z)).cpu().numpy().astype(np.float64))
            b_rows.append(model.b_head(z).view(-1).cpu().numpy().astype(np.float64))
            fb_rows.append(model.fallback_scores(z).cpu().numpy().astype(np.float64))
    frozen = Frozen(
        z=np.concatenate(z_rows),
        a=np.concatenate(a_rows),
        q=np.concatenate(q_rows),
        k=np.concatenate(k_rows),
        b=np.concatenate(b_rows),
        w=model.w.detach().cpu().numpy().astype(np.float64),
        fallback=np.concatenate(fb_rows),
    )
    for split_name in ("train", "valid", "test"):
        split = dataset.splits[split_name]
        frozen.per_split[split_name] = {
            "view_s_frag": np.asarray(split.view_s, dtype=np.int64),
            "view_c_frag": np.asarray(split.view_c, dtype=np.int64),
            "view_mol_local": np.repeat(
                np.arange(len(split.row_id)),
                np.diff(split.mol_ptr),
            ),
            "fallback_frag": np.asarray(split.fallback, dtype=np.int64),
        }
    frozen.seconds = time.perf_counter() - t0
    return frozen


def split_scores(frozen: Frozen, split_payload: Mapping[str, np.ndarray], n_mol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-molecule ``F(G)``, per-molecule view dispersion, per-molecule fallback mask."""
    s = np.asarray(split_payload["view_s_frag"], dtype=np.int64)
    c = np.asarray(split_payload["view_c_frag"], dtype=np.int64)
    mol = np.asarray(split_payload["view_mol_local"], dtype=np.int64)
    fb_frag = np.asarray(split_payload["fallback_frag"], dtype=np.int64)
    scores = frozen.view_scores(s, c) if s.size else np.zeros(0)
    sums = np.bincount(mol, weights=scores, minlength=n_mol)
    counts = np.bincount(mol, minlength=n_mol).astype(np.float64)
    has = counts > 0
    mean = np.zeros(n_mol)
    mean[has] = sums[has] / counts[has]
    disp = np.zeros(n_mol)
    if s.size:
        dev = scores - mean[mol]
        disp[has] = np.sqrt(np.bincount(mol, weights=dev ** 2, minlength=n_mol)[has] / counts[has])
    fb = fb_frag >= 0
    if bool(fb.any()):
        mean[fb] = frozen.fallback[fb_frag[fb]]
    return mean, disp, fb


def predict_with_words(
    frozen: Frozen,
    split_payload: Mapping[str, np.ndarray],
    n_mol: int,
    *,
    a_words: np.ndarray,
    q_words: np.ndarray,
) -> np.ndarray:
    """``F_D(G)`` with ``a(S), q(S)`` replaced by their dictionary versions."""
    s = np.asarray(split_payload["view_s_frag"], dtype=np.int64)
    c = np.asarray(split_payload["view_c_frag"], dtype=np.int64)
    mol = np.asarray(split_payload["view_mol_local"], dtype=np.int64)
    fb_frag = np.asarray(split_payload["fallback_frag"], dtype=np.int64)
    scores = np.zeros(0)
    if s.size:
        scale = q_words.shape[1] ** 0.5
        t = (q_words[s] * frozen.k[c] * frozen.w[None, :]).sum(axis=1) / scale
        scores = frozen.b[c] + a_words[s] + t
    sums = np.bincount(mol, weights=scores, minlength=n_mol) if scores.size else np.zeros(n_mol)
    counts = np.bincount(mol, minlength=n_mol).astype(np.float64)
    has = counts > 0
    mean = np.zeros(n_mol)
    mean[has] = sums[has] / counts[has]
    fb = fb_frag >= 0
    if bool(fb.any()):
        mean[fb] = frozen.fallback[fb_frag[fb]]
    return mean


# ---------------------------------------------------------------------------
# reference, panel, prototypes
# ---------------------------------------------------------------------------


def reference_fragment(inventory: Inventory, tau: tuple) -> int:
    words = inventory.class_words[tau]
    return int(min(words, key=lambda f: (-inventory.word_mol_freq[f], inventory.keys[f])))


def environment_panel(inventory: Inventory, tau: tuple, size: int) -> list[int]:
    envs = inventory.class_envs.get(tau, [])
    ranked = sorted(envs, key=lambda f: (-inventory.env_mol_freq.get(f, 0), inventory.keys[f]))
    return [int(f) for f in ranked[: int(size)]]


def response_matrix(frozen: Frozen, s_ids: Sequence[int], d0: int, envs: Sequence[int]) -> np.ndarray:
    """``R(S)[j] = s(C_j, S) - s(C_j, D0)`` for candidate fragments ``S``."""
    env = np.asarray(envs, dtype=np.int64)
    s = np.asarray(s_ids, dtype=np.int64)
    if env.size == 0:
        return np.zeros((s.size, 0))
    scale = frozen.q.shape[1] ** 0.5
    q_s = frozen.q[s]
    k_e = frozen.k[env]
    t_s = (q_s[:, None, :] * k_e[None, :, :] * frozen.w[None, None, :]).sum(axis=2) / scale
    q0 = frozen.q[d0]
    t0 = (q0[None, :] * k_e * frozen.w[None, :]).sum(axis=1) / scale
    b_e = frozen.b[env][None, :]
    s_full = b_e + frozen.a[s][:, None] + t_s
    s0_full = b_e + frozen.a[d0] + t0[None, :]
    return s_full - s0_full


def farthest_first(vectors: np.ndarray, d0_local: int, k: int, metric: str,
                   keys: Sequence[str] | None = None) -> list[int]:
    """``D0`` first, then the point farthest from the current selected set.

    Ties (including the all-zero far-distance case) choose the smallest exact
    canonical key; the selection never silently stops before ``K``.
    """
    n = vectors.shape[0]
    if n == 0:
        return []
    norms = np.linalg.norm(vectors, axis=1)
    zero = norms <= 0
    if metric == "cos":
        unit = vectors / np.where(norms > 0, norms, 1.0)[:, None]
    selected = [int(d0_local)]
    if k <= 1:
        return selected
    chosen = np.zeros(n, dtype=bool)
    chosen[selected[0]] = True
    key_list = list(keys) if keys is not None else None
    if key_list is not None and len(key_list) != n:
        raise ValueError("tie keys do not align with the candidate vectors")

    def distance(idx: int) -> np.ndarray:
        if metric == "l2":
            return np.linalg.norm(vectors - vectors[idx][None, :], axis=1)
        sim = unit @ unit[idx]
        d = 1.0 - sim
        return np.where(zero | zero[idx], 0.0, d)

    best = distance(selected[0])
    best[chosen] = -np.inf
    for _ in range(int(k) - 1):
        max_val = float(np.max(best))
        if not np.isfinite(max_val):
            break
        tied = np.flatnonzero(best == max_val)
        if key_list is not None and tied.size > 1:
            nxt = int(min(tied.tolist(), key=lambda i: key_list[i]))
        else:
            nxt = int(tied[0])
        selected.append(nxt)
        chosen[nxt] = True
        best = np.minimum(best, distance(nxt))
        best[chosen] = -np.inf
    return selected


def frequency_selection(inventory: Inventory, class_frags: Sequence[int], d0_local: int, k: int) -> list[int]:
    """``D0`` first, then descending Train molecule frequency (local indices)."""
    frags = [int(f) for f in class_frags]
    d0 = frags[int(d0_local)]
    others = sorted(
        (i for i, f in enumerate(frags) if f != d0),
        key=lambda i: (-inventory.word_mol_freq.get(frags[i], 0), inventory.keys[frags[i]]),
    )
    return [int(d0_local)] + others[: max(0, int(k) - 1)]


# ---------------------------------------------------------------------------
# simplex projection (frozen Frank-Wolfe, 64 steps)
# ---------------------------------------------------------------------------


def project_simplex_frank_wolfe(
    targets: np.ndarray,
    atoms: np.ndarray,
    *,
    steps: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``min ||target - sum_k alpha_k atom_k||^2`` s.t. ``alpha in simplex``."""
    n = int(targets.shape[0])
    k = int(atoms.shape[0])
    if n == 0 or k == 0:
        return np.zeros((n, k)), {"steps_used": 0, "canonical_steps": int(steps)}
    start = np.argmin(np.sum((targets[:, None, :] - atoms[None, :, :]) ** 2, axis=2), axis=1)
    alpha = np.zeros((n, k))
    alpha[np.arange(n), start] = 1.0
    current = atoms[start]
    obj_before = np.sum((targets - current) ** 2, axis=1)
    used = 0
    rows = np.arange(n)
    for _ in range(int(steps)):
        residual = targets - current
        vals = residual @ atoms.T - np.sum(residual * current, axis=1, keepdims=True)
        vertex = np.argmax(vals, axis=1)
        d = atoms[vertex] - current
        dd = np.sum(d * d, axis=1)
        num = np.sum(residual * d, axis=1)
        gamma = np.where(dd > 0, num / np.where(dd > 0, dd, 1.0), 0.0)
        gamma = np.clip(gamma, 0.0, 1.0)
        alpha *= (1.0 - gamma)[:, None]
        alpha[rows, vertex] += gamma
        current = current + gamma[:, None] * d
        used += 1
        if np.all(gamma <= 1e-12):
            break
    obj_after = np.sum((targets - current) ** 2, axis=1)
    if np.any(obj_after > obj_before + 1e-9):
        raise AssertionError("Frank-Wolfe increased the panel objective")
    if np.any(alpha < -1e-12) or not np.allclose(alpha.sum(axis=1), 1.0, atol=1e-9):
        raise AssertionError("Frank-Wolfe left the simplex")
    return alpha, {
        "steps_used": int(used),
        "canonical_steps": int(steps),
        "max_objective_increase": float(np.max(obj_after - obj_before)) if n else 0.0,
        "mean_panel_residual_before": float(np.mean(obj_before)) if n else 0.0,
        "mean_panel_residual_after": float(np.mean(obj_after)) if n else 0.0,
    }
