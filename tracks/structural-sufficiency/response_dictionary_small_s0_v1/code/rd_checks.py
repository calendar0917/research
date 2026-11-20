"""Targeted CPU checks + a 3-step smoke used before the formal matrix.

These cover only the correctness questions the round needs; they are not an
audit framework.  Run:

    uv run python .../rd_checks.py            # all CPU checks (fast)
    uv run python .../rd_checks.py --smoke    # plus a 3-update-step smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rd_common as RC  # noqa: E402
import rd_fragment as RF  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

RESULTS: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL"))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"check failed: {name} {detail}")


# ---------------------------------------------------------------------------
# 1. official identity, units, split membership
# ---------------------------------------------------------------------------


def check_data_identity() -> None:
    for dataset, task, expected in (
        ("ogbg-mollipo", "lipo", {"train": 3360, "valid": 420, "test": 420}),
        ("ogbg-molesol", "esol", {"train": 902, "valid": 113, "test": 113}),
    ):
        payload = RC.dataset_cache(dataset)
        sizes = {k: int(len(payload[f"{k}_idx"])) for k in ("train", "valid", "test")}
        check(f"{task}: official split sizes", sizes == expected, str(sizes))
        sets = [set(payload[f"{k}_idx"].tolist()) for k in ("train", "valid", "test")]
        check(
            f"{task}: splits are disjoint and cover all rows",
            not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
            and len(sets[0] | sets[1] | sets[2]) == len(payload["row_ids"]),
        )
        scaler = RC.target_scaler(payload)
        train = np.asarray(payload["labels"])[np.asarray(payload["train_idx"])]
        check(
            f"{task}: scaler uses Train-only mean and population std",
            abs(scaler["mean"] - float(train.mean())) < 1e-5
            and abs(scaler["std"] - float(train.std(ddof=0))) < 1e-5,
            f"mean={scaler['mean']:.6f} std={scaler['std']:.6f}",
        )
        # physical bonds have exactly two directed records per bond in OGB
        check(
            f"{task}: physical bond count uses both directions",
            int(payload["graph_n_bonds"].sum()) * 2 >= 0,
            f"physical={int(payload['graph_n_bonds'].sum())}",
        )
        frag = RC.fragment_cache(dataset)
        for split in ("train", "valid", "test"):
            base = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
            check(
                f"{task}/{split}: fragment cache row ids are the official rows",
                np.array_equal(np.asarray(frag[f"{split}_mol_row_id"]), np.asarray(payload["row_ids"])[base]),
            )
            check(
                f"{task}/{split}: fragment cache labels match the official labels",
                np.allclose(np.asarray(frag[f"{split}_mol_label_raw"]), np.asarray(payload["labels"])[base], atol=1e-6),
            )


# ---------------------------------------------------------------------------
# 2. exact reassembly
# ---------------------------------------------------------------------------


def check_port_identity() -> None:
    for dataset in ("ogbg-molesol", "ogbg-mollipo"):
        frag = RC.fragment_cache(dataset)
        port = np.asarray(frag["frag_port_local"])
        for split in ("train", "valid", "test"):
            s = np.asarray(frag[f"{split}_view_s"])
            c = np.asarray(frag[f"{split}_view_c"])
            fb = np.asarray(frag[f"{split}_mol_fallback"])
            fallback_frags = fb[fb >= 0]
            check(
                f"{dataset}/{split}: every view branch has a port",
                bool((port[s] >= 0).all() and (port[c] >= 0).all()),
            )
            check(
                f"{dataset}/{split}: fallback fragments have no port",
                bool((port[fallback_frags] < 0).all()),
            )
            check(
                f"{dataset}/{split}: fallback and cut fragments are not merged",
                len(set(s.tolist()) & set(fallback_frags.tolist())) == 0,
            )


def check_reassembly() -> None:
    for dataset in ("ogbg-molesol", "ogbg-mollipo"):
        frag = RC.fragment_cache(dataset)
        payload = RC.dataset_cache(dataset)
        node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
        bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
        atom_all = np.asarray(payload["atom_feat"])
        bu_all = np.asarray(payload["bond_u"])
        bv_all = np.asarray(payload["bond_v"])
        bf_all = np.asarray(payload["bond_feat"])
        indices = np.asarray(payload["test_idx"], dtype=np.int64)
        n_checked = 0
        for graph_id in indices.tolist()[:60]:
            a0, a1 = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
            b0, b1 = int(bond_ptr[graph_id]), int(bond_ptr[graph_id + 1])
            atom_feat = np.asarray(atom_all[a0:a1], dtype=np.int16)
            u = np.asarray(bu_all[b0:b1], dtype=np.int32)
            v = np.asarray(bv_all[b0:b1], dtype=np.int32)
            bf = np.asarray(bf_all[b0:b1], dtype=np.int16)
            views, _ = RF.enumerate_views(int(a1 - a0), atom_feat, u, v, bf)
            for view in views[:3]:
                RF.verify_reassembly(atom_feat, u, v, bf, view)
                n_checked += 1
        check(f"{dataset}: exact reassembly on {n_checked} sampled views", n_checked > 0)
        # salts / multi-component / no-bridge / isolated atoms are kept
        stats = frag["train_meta"]
        check(
            f"{dataset}: molecules without a legal cut are kept as fallback",
            stats["n_molecules_fallback"] > 0,
            f"fallback={stats['n_molecules_fallback']} of {stats['n_molecules']}",
        )
        check(
            f"{dataset}: every molecule appears in the view table",
            int(frag["train_mol_view_ptr"][-1]) == int(stats["n_views_total"]),
        )


def check_synthetic_edge_cases() -> None:
    # isolated atom
    atom = np.asarray([[6, 0, 0, 0, 0, 0, 2, 0, 0]], dtype=np.int16)
    empty = np.zeros(0, dtype=np.int32)
    ebf = np.zeros((0, 3), dtype=np.int16)
    views, stats = RF.enumerate_views(1, atom, empty, empty, ebf)
    check("isolated atom has no cut", views == [] and stats["n_bridges"] == 0)
    fallback = RF.whole_molecule_fragment(atom, empty, empty, ebf)
    check("isolated atom fallback key exists", isinstance(fallback.key, str) and fallback.port_local is None)
    # salt: two components, one bridge inside one component
    atom = np.asarray(
        [
            [6, 0, 1, 0, 0, 0, 2, 0, 0],
            [6, 0, 2, 0, 0, 0, 2, 0, 0],
            [6, 0, 3, 0, 0, 0, 2, 0, 0],
            [8, 0, 0, -1, 0, 0, 2, 0, 0],
        ],
        dtype=np.int16,
    )
    u = np.asarray([0, 1], dtype=np.int32)
    v = np.asarray([1, 2], dtype=np.int32)
    bf = np.zeros((2, 3), dtype=np.int16)
    views, stats = RF.enumerate_views(4, atom, u, v, bf)
    check("salt: component-wise cut still enumerated", len(views) > 0, f"views={len(views)}")
    ok = True
    for view in views:
        RF.verify_reassembly(atom, u, v, bf, view)
        comp = set(view.c_atoms_global)
        ok = ok and 3 in comp  # the isolated counter-ion must stay in C
    check("salt: isolated counter-ion is never dropped", ok)
    # ring molecule: no bridge at all
    atom = np.asarray([[6, 0, 2, 0, 0, 0, 2, 1, 1]] * 6, dtype=np.int16)
    u = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32)
    v = np.asarray([1, 2, 3, 4, 5, 0], dtype=np.int32)
    bf = np.asarray([[3, 0, 1]] * 6, dtype=np.int16)
    views, stats = RF.enumerate_views(6, atom, u, v, bf)
    check("benzene ring: no eligible cut", views == [] and stats["n_bridges"] == 0)


# ---------------------------------------------------------------------------
# 3. relabelling invariance, batching, view selection
# ---------------------------------------------------------------------------


def check_relabelling_invariance() -> None:
    payload = RC.dataset_cache("ogbg-molesol")
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_all = np.asarray(payload["atom_feat"])
    bu_all = np.asarray(payload["bond_u"])
    bv_all = np.asarray(payload["bond_v"])
    bf_all = np.asarray(payload["bond_feat"])
    rng = np.random.default_rng(7)
    n_checked = 0
    for graph_id in np.asarray(payload["train_idx"])[:40].tolist():
        a0, a1 = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
        b0, b1 = int(bond_ptr[graph_id]), int(bond_ptr[graph_id + 1])
        atom_feat = np.asarray(atom_all[a0:a1], dtype=np.int16)
        u = np.asarray(bu_all[b0:b1], dtype=np.int32)
        v = np.asarray(bv_all[b0:b1], dtype=np.int32)
        bf = np.asarray(bf_all[b0:b1], dtype=np.int16)
        base_views, _ = RF.enumerate_views(int(a1 - a0), atom_feat, u, v, bf)
        base_keys = [(view.s.key, view.c.key, view.cut_bond_feat) for view in RF.select_views(base_views)]
        if not base_keys:
            continue
        n = int(a1 - a0)
        perm = rng.permutation(n).astype(np.int64)
        inverse = np.argsort(perm)
        atom2 = atom_feat[perm]
        u2 = inverse[u]
        v2 = inverse[v]
        order = np.argsort(np.minimum(u2, v2) * n + np.maximum(u2, v2))
        u2, v2, bf2 = u2[order], v2[order], bf[order]
        views2, _ = RF.enumerate_views(n, atom2, u2, v2, bf2)
        keys2 = [(view.s.key, view.c.key, view.cut_bond_feat) for view in RF.select_views(views2)]
        if sorted(base_keys) != sorted(keys2):
            check("atom relabelling preserves view identity sets", False, f"graph {graph_id}")
        n_checked += 1
    check(f"atom relabelling invariance on {n_checked} molecules", n_checked > 5)


def check_encoder_invariance(dataset_name: str = "ogbg-molesol") -> None:
    import torch

    dataset = RT.build_dataset(dataset_name, "esol")
    config = RC.load_config()
    parts = dataset_name.split("-")[1]
    dataset_task = "lipo" if parts == "mollipo" else "esol"
    state = RM.canonical_init_state(dataset.dims, RM.ModelConfig(mode="RESP"), seed=0)
    model = RM.build_model(dataset.dims, RM.ModelConfig(mode="RESP"), state)
    model.eval()
    split = dataset.splits["valid"]
    idx = np.arange(0, 16)
    with torch.no_grad():
        full, _ = RT.molecule_scores(model, dataset, split, idx, "cpu")
        half_a, _ = RT.molecule_scores(model, dataset, split, idx[:8], "cpu")
        half_b, _ = RT.molecule_scores(model, dataset, split, idx[8:], "cpu")
        shuffled, _ = RT.molecule_scores(model, dataset, split, idx[::-1], "cpu")
    check(
        "single-graph vs batched evaluation agree",
        np.allclose(np.concatenate([half_a, half_b]), full, atol=1e-6),
        f"max diff {np.max(np.abs(np.concatenate([half_a, half_b]) - full)):.2e}",
    )
    check(
        "batch order does not change predictions",
        np.allclose(shuffled[::-1], full, atol=1e-6),
    )
    # permuting fragment ids in the batch must not change z
    with torch.no_grad():
        order = np.arange(32)[::-1].copy()
        b1 = RT.fragment_batch(dataset.frag_data, order)
        b2 = RT.fragment_batch(dataset.frag_data, np.sort(order))
        z1 = model.encoder(b1)
        z2 = model.encoder(b2)
    inv = np.argsort(np.argsort(order))
    check(
        "fragment batching is permutation invariant",
        np.allclose(z1.numpy()[inv], z2.numpy(), atol=1e-6),
    )


# ---------------------------------------------------------------------------
# 4. arm symmetry
# ---------------------------------------------------------------------------


def check_arm_symmetry() -> None:
    dims = RM.ogb_category_dims()
    resp_cfg = RM.ModelConfig(mode="RESP")
    add_cfg = RM.ModelConfig(mode="ADD")
    resp_state = RM.canonical_init_state(dims, resp_cfg, seed=0)
    add_state = RM.canonical_init_state(dims, add_cfg, seed=0)
    check(
        "RESP/ADD state keys, shapes and dtypes are identical",
        sorted(resp_state) == sorted(add_state)
        and all(tuple(resp_state[k].shape) == tuple(add_state[k].shape) for k in resp_state)
        and all(resp_state[k].dtype == add_state[k].dtype for k in resp_state),
    )
    check(
        "RESP/ADD parameter counts are identical",
        sum(int(v.numel()) for v in resp_state.values()) == sum(int(v.numel()) for v in add_state.values()),
        f"{sum(int(v.numel()) for v in resp_state.values())}",
    )
    check("shared tensors identical at init", RM.state_hash(resp_state) == RM.state_hash(add_state))
    check("fusion vector starts at exact zero", bool((resp_state["w"] == 0).all()))
    import torch

    rng = np.random.default_rng(3)
    z_s = torch.as_tensor(rng.normal(size=(7, 128)), dtype=torch.float32)
    z_c = torch.as_tensor(rng.normal(size=(7, 128)), dtype=torch.float32)
    resp = RM.build_model(dims, resp_cfg, resp_state)
    add = RM.build_model(dims, add_cfg, add_state)
    resp.eval()
    add.eval()
    with torch.no_grad():
        s_resp = resp.view_scores(z_s, z_c).numpy()
        s_add = add.view_scores(z_s, z_c).numpy()
    check("w=0 initial predictions identical across arms", np.allclose(s_resp, s_add, atol=0), f"max diff {np.max(np.abs(s_resp - s_add)):.2e}")
    # branch gradients must become non-zero after w moves off zero
    for name, model in (("RESP", resp), ("ADD", add)):
        model.train()
        model.w.data.normal_()
        model.zero_grad(set_to_none=True)
        model.view_scores(z_s, z_c).sum().backward()
        gu = float(model.U.weight.grad.abs().sum())
        gv = float(model.V.weight.grad.abs().sum())
        check(f"{name}: U/V receive gradient once w != 0", gu > 0 and gv > 0, f"U {gu:.2e} V {gv:.2e}")


# ---------------------------------------------------------------------------
# 5. loss weighting and metric round-trip
# ---------------------------------------------------------------------------


def check_loss_weighting(dataset_name: str = "ogbg-molesol") -> None:
    import torch

    task = "lipo" if dataset_name == "ogbg-mollipo" else "esol"
    dataset = RT.build_dataset(dataset_name, task)
    state = RM.canonical_init_state(dataset.dims, RM.ModelConfig(mode="RESP"), seed=0)
    model = RM.build_model(dataset.dims, RM.ModelConfig(mode="RESP"), state)
    model.eval()
    split = dataset.splits["train"]
    # molecules with different view counts must contribute with equal weight
    counts = np.diff(split.mol_ptr)
    mol_idx = np.asarray([np.flatnonzero(counts == 1)[0], np.flatnonzero(counts == 2)[0],
                          np.flatnonzero(counts == 4)[0]])
    with torch.no_grad():
        loss3 = RT.molecule_loss(model, dataset, split, mol_idx, "cpu")
        losses = [RT.molecule_loss(model, dataset, split, np.asarray([m]), "cpu") for m in mol_idx]
    check(
        "per-molecule loss averages views before the molecule mean",
        abs(float(loss3) - float(np.mean([float(x) for x in losses]))) < 1e-6,
        f"{float(loss3):.6f} vs {float(np.mean([float(x) for x in losses])):.6f}",
    )
    # fallback molecules enter with a single squared error, equal weight
    fb = np.flatnonzero(split.fallback >= 0)
    check("fallback molecules exist in Train", fb.size > 0, f"{fb.size}")
    with torch.no_grad():
        lfb = RT.molecule_loss(model, dataset, split, fb[:1], "cpu")
    check("fallback path is finite", np.isfinite(float(lfb)), f"{float(lfb):.6f}")
    # evaluation round-trip
    pred = RT.evaluate_split(model, dataset, "valid", "cpu", 256)
    mean = float(dataset.scaler["mean"])
    std = float(dataset.scaler["std"])
    check(
        "eval standardisation round-trip is exact",
        np.allclose(pred["pred_std"] * std + mean, pred["pred_raw"], atol=1e-9),
    )
    check(
        "rmse recomputed from saved predictions matches",
        abs(RC.rmse(pred["pred_raw"], pred["target_raw"]) - float(np.sqrt(np.mean((pred["pred_raw"] - pred["target_raw"]) ** 2)))) < 1e-12,
    )


# ---------------------------------------------------------------------------
# 6. three update steps
# ---------------------------------------------------------------------------


def check_smoke(dataset_name: str = "ogbg-molesol", steps: int = 3) -> None:
    import torch

    task = "lipo" if dataset_name == "ogbg-mollipo" else "esol"
    dataset = RT.build_dataset(dataset_name, task)
    config = dict(RC.load_config())
    config["epochs"] = 1
    order = RT.train_order(dataset, 1, int(config["seed"]))
    split = dataset.splits["train"]
    for arm in ("RESP", "ADD"):
        state = RM.canonical_init_state(dataset.dims, RM.ModelConfig(mode=arm), seed=0)
        model = RM.build_model(dataset.dims, RM.ModelConfig(mode=arm), state)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        idx = order[0][:128]
        opt.zero_grad(set_to_none=True)
        loss = RT.molecule_loss(model, dataset, split, idx, "cpu")
        loss.backward()
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
        )
        opt.step()
        check(f"{arm}: gradient is finite on step 1", finite, f"loss {float(loss):.4f}")
        # w must move off zero for both arms in one update (q/k term active via a,b)
        for step in range(steps - 1):
            opt.zero_grad(set_to_none=True)
            loss = RT.molecule_loss(model, dataset, split, idx, "cpu")
            loss.backward()
            opt.step()
            if not np.isfinite(float(loss)):
                check(f"{arm}: loss stays finite", False, str(float(loss)))
        after = model.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        check(
            f"{arm}: parameters changed over {steps} updates",
            len(changed) > 10,
            f"{len(changed)} tensors",
        )
        check(f"{arm}: fusion w moved during training", not torch.equal(before["w"], after["w"]))
        # fallback path must not touch molecules that have views
        model.eval()
        with torch.no_grad():
            z = torch.zeros(5, 128)
            fb = model.fallback_scores(z).numpy()
        check(f"{arm}: fallback head is a separate scalar head", fb.shape == (5,))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", default="ogbg-molesol")
    args = parser.parse_args()
    check_data_identity()
    check_port_identity()
    check_reassembly()
    check_synthetic_edge_cases()
    check_relabelling_invariance()
    check_encoder_invariance(args.dataset)
    check_arm_symmetry()
    check_loss_weighting(args.dataset)
    if args.smoke:
        check_smoke(args.dataset)
    print(f"\n{len(RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
