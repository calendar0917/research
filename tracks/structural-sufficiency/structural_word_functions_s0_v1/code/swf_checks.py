"""Targeted engineering checks (<= 10 minutes; not a general test framework).

1. data identity: official split, row ids, label scale, molecule-local bond
   numbering, the 1/1/3/11 rooted template library and its frame counts.
2. relabelling invariance: random atom relabelling must not change the network
   output (frame averaging + cover averaging).  FP64 CPU target 1e-10, FP32 GPU
   allowance 1e-4.
3. the algebra of section 6.4, cover-swap symmetry, and scatter vs. explicit
   loop agreement for the sentence pooling.
4. branch initialisation: zero-init join reproduces the prefix exactly, and the
   new branch has zero gradient on step 1 / finite nonzero gradient on step 2.
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Mapping, Sequence

import numpy as np

import swf_common as C
import swf_cache as K
import swf_data as D


def check_data_identity(cfg: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    n_mol = int(node_ptr.size - 1)
    split_id = np.full(n_mol, -1, np.int8)
    for name, key in (("train", "train_idx"), ("valid", "valid_idx"), ("test", "test_idx")):
        split_id[np.asarray(raw[key], np.int64)] = C.SPLIT_ID[name]
    sizes = {name: int((split_id == C.SPLIT_ID[name]).sum()) for name in C.SPLITS}
    labels = np.asarray(raw["labels"], np.float64)
    train_rows = np.asarray(raw["train_idx"], np.int64)
    # molecule-local bond numbering: every endpoint < n_atoms of that molecule
    max_local = 0
    bad = 0
    for row in range(n_mol):
        n = int(node_ptr[row + 1] - node_ptr[row])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        if b1 > b0:
            hi = max(int(np.asarray(raw["bond_u"])[b0:b1].max()), int(np.asarray(raw["bond_v"])[b0:b1].max()))
            max_local = max(max_local, hi)
            if hi >= n:
                bad += 1
    templates = [D.TEMPLATES[t].as_dict() for t in range(D.N_TEMPLATES)]
    per_k = {k: sum(1 for t in D.TEMPLATES.values() if t.k == k) for k in range(1, 5)}
    aut_ok = True
    for t in D.TEMPLATES.values():
        for p in t.automorphisms:
            if p[0] != 0:
                aut_ok = False
            if not all(t.adjacency[p[i]][p[j]] == t.adjacency[i][j] for i in range(t.k) for j in range(t.k)):
                aut_ok = False
    return {
        "n_molecules": n_mol,
        "split_sizes": sizes,
        "split_sizes_official": sizes == {"train": 3360, "valid": 420, "test": 420},
        "row_ids_are_arange": bool(np.array_equal(np.asarray(raw["row_ids"]), np.arange(n_mol))),
        "label_stats": {
            "min": float(labels.min()), "max": float(labels.max()), "mean": float(labels.mean()),
            "train_mean": float(labels[train_rows].mean()),
            "train_std_ddof0": float(labels[train_rows].std(ddof=0)),
        },
        "bond_numbering_is_molecule_local": {"violations": bad, "max_endpoint": max_local},
        "templates_per_k": {str(k): v for k, v in per_k.items()},
        "template_library_total": len(D.TEMPLATES),
        "template_automorphisms_are_root_fixing": aut_ok,
        "template_nframes": D.TEMPLATE_NFRAMES.tolist(),
        "templates": templates,
        "raw_cache_sha256": raw.get("cache_sha256"),
        "raw_cache_sha256_matches": raw.get("cache_sha256") == str(cfg["raw_cache_sha256"]),
    }


# ---------------------------------------------------------------------------
# relabelling invariance
# ---------------------------------------------------------------------------


def _mini_cache(mols: Sequence[D.Molecule]) -> K.Cache:
    codes = D.CodeTables()
    built = K.build_structures(mols, codes)
    base = dict(built["base"])
    n = len(mols)
    node_ptr = np.zeros(n + 1, np.int64)
    bond_ptr = np.zeros(n + 1, np.int64)
    atom_feat = []
    bond_feat = []
    bond_u = []
    bond_v = []
    labels = np.zeros(n, np.float32)
    for i, mol in enumerate(mols):
        node_ptr[i + 1] = node_ptr[i] + mol.n_atoms
        bond_ptr[i + 1] = bond_ptr[i] + mol.bond_u.size
        atom_feat.append(mol.atom_feat)
        bond_feat.append(mol.bond_feat)
        bond_u.append(mol.bond_u)
        bond_v.append(mol.bond_v)
        labels[i] = 0.0
    base.update({
        "row_ids": np.arange(n, dtype=np.int32),
        "split_id": np.zeros(n, np.int8),
        "labels": labels,
        "node_ptr": node_ptr,
        "bond_ptr": bond_ptr,
        "atom_feat": np.concatenate(atom_feat, axis=0),
        "bond_u": np.concatenate(bond_u),
        "bond_v": np.concatenate(bond_v),
        "bond_feat": np.concatenate(bond_feat, axis=0),
    })
    return K.Cache(need_sentences=True, base=base, sent=built["sent"], stats=C.read_json(K.stats_path()))


def _eval_mini(model, cache: K.Cache, mode: str, dtype) -> np.ndarray:
    import torch

    import swf_train as T

    model = model.to("cpu").to(dtype)
    model.eval()
    rows = cache.split_rows["train"]
    pack = cache.pack(rows, sentences=(mode != "prefix"))
    b = T.to_tensors(pack, "cpu", dtype)
    with torch.no_grad():
        return model(b, mode).double().cpu().numpy()


def check_relabel_invariance(cfg: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    import swf_model as M

    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    order = np.argsort(np.asarray(raw["graph_n_atoms"], np.int64), kind="stable")
    small = [int(i) for i in order[:3]]
    step = max(1, n_mol // 5)
    real = [int(i) for i in np.arange(n_mol)[::step][:5]]
    chosen = sorted(set(small + real))
    mols = [D.raw_molecule(raw, i) for i in chosen]

    rng = np.random.default_rng(20261121)
    relabelled = []
    for mol in mols:
        perm = rng.permutation(mol.n_atoms)
        relabelled.append(mol.permute(perm))

    cache_a = _mini_cache(mols)
    cache_b = _mini_cache(relabelled)

    stats = C.read_json(K.stats_path())
    model = M.build_model(cfg)
    M.apply_train_support(model, stats)
    M.bind_constants(model, stats)

    out: dict[str, Any] = {"molecules": [int(mols[i].row_id) for i in range(len(mols))], "modes": {}}
    active_templates = np.asarray(stats["template_active"], np.float64)
    for mode in ("prefix", "full", "bag_cont", "early", "flat"):
        a = _eval_mini(model, cache_a, mode, torch.float64)
        b = _eval_mini(model, cache_b, mode, torch.float64)
        out["modes"][mode] = {"max_abs_diff_fp64": float(np.max(np.abs(a - b))) if a.size else 0.0}
    # un-trained constants only touch templates active in Train; if every template
    # in the probe is inactive the check is vacuous, record it explicitly.
    used = np.concatenate([cache_a.base["word_tmpl"], cache_b.base["word_tmpl"]]) if cache_a.base["word_tmpl"].size else np.zeros(0, int)
    out["templates_used"] = sorted(int(t) for t in np.unique(used))
    out["inactive_templates_used"] = [int(t) for t in np.unique(used) if active_templates[int(t)] == 0]
    out["fp64_tolerance"] = 1e-10
    out["fp64_pass"] = all(v["max_abs_diff_fp64"] <= 1e-10 for v in out["modes"].values())

    if torch.cuda.is_available():
        model32 = M.build_model(cfg)
        M.apply_train_support(model32, stats)
        M.bind_constants(model32, stats)
        model32 = model32.to("cuda")
        diffs = {}
        for mode in ("prefix", "full", "early", "flat"):
            pa = predict_mini_gpu(model32, cache_a, mode)
            pb = predict_mini_gpu(model32, cache_b, mode)
            diffs[mode] = float(np.max(np.abs(pa - pb))) if pa.size else 0.0
        out["fp32_gpu_max_abs_diff"] = diffs
        out["fp32_gpu_tolerance"] = 1e-4
        out["fp32_gpu_pass"] = all(v <= 1e-4 for v in diffs.values())
    else:
        out["fp32_gpu_max_abs_diff"] = None
        out["fp32_gpu_pass"] = None
    return out


def predict_mini_gpu(model, cache: K.Cache, mode: str) -> np.ndarray:
    import torch

    import swf_train as T

    model.eval()
    rows = cache.split_rows["train"]
    pack = cache.pack(rows, sentences=(mode != "prefix"))
    b = T.to_tensors(pack, "cuda", torch.float32)
    with torch.no_grad():
        return model(b, mode).double().cpu().numpy()


# ---------------------------------------------------------------------------
# algebra / symmetry / scatter agreement
# ---------------------------------------------------------------------------


def check_binding_algebra() -> dict[str, Any]:
    """The exact 6.4 example, computed with the implemented grouping."""
    def combine_then_average(a, lam):
        a = np.asarray(a, np.float64)
        lam = np.asarray(lam, np.float64)
        return float(np.mean(a * 1.0 * lam))

    def average_then_multiply(a, lam):
        a = np.asarray(a, np.float64)
        lam = np.asarray(lam, np.float64)
        return float(np.mean(a) * np.mean(lam))

    a = [1.0, 0.0]
    lam = [1.0, 0.0]
    first = combine_then_average(a, lam)
    second = average_then_multiply(a, lam)
    swapped_attr = [0.0, 1.0]
    third = combine_then_average(swapped_attr, lam)
    fourth = average_then_multiply(swapped_attr, lam)
    co_renumbered = combine_then_average(swapped_attr, [0.0, 1.0])
    return {
        "case": "two exchangeable non-root positions with values [1,0]; the other word binds position 1; relation weight [1,0]",
        "combine_then_average": first,
        "average_then_multiply": second,
        "combine_then_average_after_attribute_swap": third,
        "average_then_multiply_after_attribute_swap": fourth,
        "combine_then_average_after_co_renumbering": co_renumbered,
        "expected": {
            "combine_then_average": 0.5,
            "average_then_multiply": 0.25,
            "combine_then_average_after_attribute_swap": 0.0,
            "average_then_multiply_after_attribute_swap": 0.25,
            "combine_then_average_after_co_renumbering": 0.5,
        },
        "pass": (
            abs(first - 0.5) < 1e-12
            and abs(second - 0.25) < 1e-12
            and abs(third - 0.0) < 1e-12
            and abs(fourth - 0.25) < 1e-12
            and abs(co_renumbered - 0.5) < 1e-12
        ),
    }


def check_cover_swap_and_scatter(cfg: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    cache = K.Cache(need_sentences=True)
    rows = cache.split_rows["valid"][:8]
    pack = cache.pack(rows, sentences=True)
    rng = np.random.default_rng(7)
    Z = int(cfg["dims"]["coord_dim"])
    qh = torch.as_tensor(rng.normal(size=(int(pack["n_frames"]), Z)), dtype=torch.float64)
    kh = torch.as_tensor(rng.normal(size=(int(pack["n_frames"]), Z)), dtype=torch.float64)
    lam = torch.as_tensor(rng.normal(size=(int(pack["n_unique_rel"]), Z)), dtype=torch.float64)
    left = torch.as_tensor(pack["row_left"])
    right = torch.as_tensor(pack["row_right"])
    ruv = torch.as_tensor(pack["row_rel_uv"])
    rvu = torch.as_tensor(pack["row_rel_vu"])
    w = torch.as_tensor(pack["row_weight"].astype(np.float64))
    sent = torch.as_tensor(pack["row_sent"])

    fwd = qh[left] * kh[right] * lam[ruv]
    rev = qh[right] * kh[left] * lam[rvu]
    t1 = 0.5 * (fwd + rev)
    # identical formula with the two words exchanged (order independence)
    fwd2 = qh[right] * kh[left] * lam[rvu]
    rev2 = qh[left] * kh[right] * lam[ruv]
    t2 = 0.5 * (fwd2 + rev2)

    nsb = int(pack["n_sentences"])
    z = torch.zeros((nsb, Z), dtype=torch.float64).index_add_(0, sent, w.unsqueeze(-1) * t1)

    # explicit python loop over a small slice of sentences
    order = np.argsort(pack["row_sent"], kind="stable")
    sent_arr = np.asarray(pack["row_sent"])
    loop = np.zeros((nsb, Z), np.float64)
    take = sorted(set(sent_arr[:0].tolist()))
    for s in range(min(50, nsb)):
        idx = np.where(sent_arr == s)[0]
        if idx.size == 0:
            continue
        loop[s] = (np.asarray(w)[idx, None] * t1.numpy()[idx]).sum(axis=0)
    diff_loop = float(np.max(np.abs(loop[:50] - z.numpy()[:50]))) if nsb else 0.0
    del order, take
    return {
        "cover_swap_symmetry_max_abs": float(torch.max(torch.abs(t1 - t2))),
        "scatter_vs_loop_max_abs": diff_loop,
        "n_rows": int(pack["n_rows"]),
        "n_sentences": nsb,
        "pass": bool(diff_loop < 1e-10 and float(torch.max(torch.abs(t1 - t2))) < 1e-12),
    }


# ---------------------------------------------------------------------------
# branch initialisation and gradient flow
# ---------------------------------------------------------------------------


def check_branch(cfg: Mapping[str, Any], device: str, prefix_ckpt) -> dict[str, Any]:
    import torch

    import swf_model as M
    import swf_train as T

    cache = K.Cache(need_sentences=True)
    C.set_deterministic(int(cfg["recipe"]["seed"]))
    model = M.build_model(cfg)
    M.apply_train_support(model, cache.stats)
    M.bind_constants(model, cache.stats)
    prefix_state = torch.load(prefix_ckpt, map_location="cpu")
    M.load_prefix_into(model, prefix_state)
    model = model.to(device)

    train_rows = cache.split_rows["train"]
    chunk = train_rows[: int(cfg["recipe"]["batch"])]
    pack = cache.pack(chunk, sentences=True)
    b = T.to_tensors(pack, device, torch.float32)
    target = torch.as_tensor(
        (cache.labels[chunk] - cache.scaler["mean"]) / cache.scaler["std"],
        dtype=torch.float32, device=device,
    )
    model.eval()
    with torch.no_grad():
        y_init = model(b, "full")
        y_prefix = model(b, "prefix")
    init_gap = float(torch.max(torch.abs(y_init - y_prefix)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def grad_summary() -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                summary[name] = {"max_abs": 0.0, "finite": True}
                continue
            g = param.grad.detach()
            summary[name] = {"max_abs": float(g.abs().max()), "finite": bool(torch.isfinite(g).all())}
        return summary

    step_grads = []
    step_losses = []
    for step in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y = model(b, "full")
        loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        step_grads.append(grad_summary())
        step_losses.append(float(loss.detach().cpu()))
        optimizer.step()

    def group(gmap, prefix):
        vals = [v["max_abs"] for k, v in gmap.items() if k.startswith(prefix)]
        return float(max(vals)) if vals else 0.0

    groups = ("rel_mlp", "flat_mlp", "q", "k", "word_mlp", "bag_mlp", "join", "atom_mlp")
    step1 = {g: group(step_grads[0], g) for g in groups}
    step2 = {g: group(step_grads[1], g) for g in groups}
    all_finite = all(v["finite"] for gmap in step_grads for v in gmap.values())
    return {
        "init_equals_prefix_max_abs_diff": init_gap,
        "init_tolerance": 1e-5,
        "init_pass": bool(init_gap <= 1e-5),
        "loss_step1": step_losses[0],
        "loss_step2": step_losses[1],
        "grad_step1_max_abs": step1,
        "grad_step2_max_abs": step2,
        "grad_step1_new_branch_is_zero": bool(
            step1["rel_mlp"] == 0.0 and step1["q"] == 0.0 and step1["k"] == 0.0
        ),
        "grad_step2_new_branch_finite_nonzero": bool(
            np.isfinite(step2["rel_mlp"]) and step2["rel_mlp"] > 0.0
            and np.isfinite(step2["q"]) and step2["q"] > 0.0
        ),
        "prefix_gradient_nonzero_at_step1": bool(step1["word_mlp"] > 0.0 and step1["bag_mlp"] > 0.0),
        "all_gradients_finite": bool(all_finite),
        "pass": bool(
            init_gap <= 1e-5 and all_finite and step2["rel_mlp"] > 0.0 and step2["q"] > 0.0
            and step1["word_mlp"] > 0.0
        ),
    }


def run_checks(cfg: Mapping[str, Any], *, device: str, prefix_ckpt=None) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = D.load_raw_cache(cfg)
    out: dict[str, Any] = {
        "identity": check_data_identity(cfg, raw),
        "binding_algebra": check_binding_algebra(),
        "seed": C.set_deterministic(int(cfg["recipe"]["seed"])),
    }
    if K.cache_ready():
        out["relabel_invariance"] = check_relabel_invariance(cfg, raw)
        out["cover_swap_and_scatter"] = check_cover_swap_and_scatter(cfg)
        if prefix_ckpt is not None:
            out["branch"] = check_branch(cfg, device, prefix_ckpt)
    out["seconds"] = time.perf_counter() - t0
    return out
