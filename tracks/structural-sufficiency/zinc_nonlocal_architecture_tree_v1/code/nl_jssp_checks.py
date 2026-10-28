"""CPU gates for the JSSP mask mechanism (no labels, no GPU needed)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[3]
SC = REPO / "tracks/structure-contribution/code"
for p in (str(CODE), str(SC), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import nl_common as N  # noqa: E402
import nl_jssp as jssp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

OUT = CODE.parent / "analysis" / "jssp_cpu_checks.json"


def state_hash(module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def same_object(a, b) -> bool:
    for field in ("atom_feat", "bond_feat", "junc_feat", "global_feat",
                  "bond_index", "bond_type", "junc_index", "junc_bond"):
        x, y = getattr(a, field), getattr(b, field)
        if x.shape != y.shape or not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return (a.n_atoms == b.n_atoms and np.array_equal(a.bond_index, b.bond_index))


def main() -> int:
    import cscl_features as cf
    raw_ds = cf.load_official_train()
    raw = []
    for d in raw_ds[:200]:
        raw.append({"x": d.x.numpy().copy(), "edge_index": d.edge_index.numpy().copy(),
                    "edge_attr": d.edge_attr.numpy().copy()})
    bases = [sro.build_mol_objects(i, r["x"], r["edge_index"], r["edge_attr"])
             for i, r in enumerate(raw)]
    checks = {}

    # 1. span enumeration vs brute force
    bad = 0
    for o in bases[:100]:
        bi = np.asarray(o.bond_index)
        brute = set()
        for e1 in range(len(bi)):
            for e2 in range(e1 + 1, len(bi)):
                if len(set(bi[e1].tolist()) & set(bi[e2].tolist())):
                    brute.add((e1, e2))
        got = set(jssp.enumerate_spans(bi))
        bad += int(brute != got)
    checks["span_enumeration"] = {"n": 100, "mismatch": bad, "pass": bad == 0}

    # 2. span closure + every span is a real 2-bond path
    closure_bad = path_bad = 0
    for o in bases[:100]:
        bi = np.asarray(o.bond_index)
        a_ids, b_ids = jssp.sample_joint_span_mask(0, o.mol_index, 3, o.n_atoms, bi)
        for e in b_ids:
            u, v = int(bi[e, 0]), int(bi[e, 1])
            if u not in a_ids or v not in a_ids:
                closure_bad += 1
        # every chosen bond must share an atom with another chosen bond unless single-bond fallback
        if len(b_ids) > 1:
            for e in b_ids:
                if not any(e != f and set(bi[e].tolist()) & set(bi[f].tolist()) for f in b_ids):
                    path_bad += 1
    checks["span_closure"] = {"closure_bad": closure_bad, "unlinked_bonds": path_bad,
                              "pass": closure_bad == 0 and path_bad == 0}

    # 3. hidden-truth swap leaves visible masked object unchanged
    atom_swap_bad = bond_swap_bad = 0
    for o in bases[:100]:
        r = raw[o.mol_index]
        a_ids, b_ids = jssp.sample_joint_span_mask(1, o.mol_index, 4, o.n_atoms, o.bond_index)
        base_s = jssp.build_masked_sample(o, r["x"], r["edge_index"], r["edge_attr"],
                                          seed=1, visit_index=4)
        for aid in a_ids[:1]:
            x2 = r["x"].copy()
            x2[int(aid), 0] = (int(x2[int(aid), 0]) + 1) % 27
            s2 = jssp.build_masked_sample(o, x2, r["edge_index"], r["edge_attr"],
                                          seed=1, visit_index=4)
            atom_swap_bad += int(not same_object(base_s.masked, s2.masked))
        for bid in b_ids[:1]:
            # edge_attr is (2m, 1) with both directions; physical bond bid -> columns
            ea2 = r["edge_attr"].copy()
            # locate directed columns whose sorted endpoint pair matches
            u, v = np.asarray(o.bond_index)[bid]
            for col in range(r["edge_index"].shape[1]):
                if {int(r["edge_index"][0, col]), int(r["edge_index"][1, col])} == {int(u), int(v)}:
                    ea2[col] = (int(ea2[col]) + 1) % 4
            s2 = jssp.build_masked_sample(o, r["x"], r["edge_index"], ea2,
                                          seed=1, visit_index=4)
            bond_swap_bad += int(not same_object(base_s.masked, s2.masked))
    checks["hidden_truth_swap"] = {"atom_bad": atom_swap_bad, "bond_bad": bond_swap_bad,
                                   "pass": atom_swap_bad == 0 and bond_swap_bad == 0}

    # 4. coverage parity
    cov = jssp.mask_stats(bases, raw, n_mol=200, seed=0, visit_index=0)
    checks["coverage"] = {**cov, "pass": 0.10 <= cov["span_atom_fraction_mean"] <= 0.30}

    # 5. pretraining forward/backward finite and learns on a few steps
    cfg = src.BJOCConfig()
    torch.manual_seed(0)
    backbone = src.build_bjoc(cfg, seed=0, organization_mode=src.NULL,
                              global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
                              global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
                              device="cpu")
    pretext = mp.PretextModel(backbone)
    opt = torch.optim.Adam(pretext.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    for step in range(6):
        samples = []
        for j in range(8):
            r = raw[j]
            samples.append(jssp.build_masked_sample(bases[j], r["x"], r["edge_index"],
                                                    r["edge_attr"], seed=0, visit_index=step))
        batch, ta, tb, _, _ = mp.batch_pretext_samples(samples)
        loss, _ = pretext.loss(batch, ta, tb)
        opt.zero_grad(); loss.backward()
        gfin = all(torch.isfinite(p.grad).all() for p in pretext.parameters() if p.grad is not None)
        grad = float(torch.nn.utils.clip_grad_norm_(pretext.parameters(), 5.0))
        opt.step()
        losses.append(float(loss))
        assert gfin and np.isfinite(grad)
    checks["pretext_smoke"] = {"losses": losses, "decreased": losses[-1] < losses[0],
                               "pass": losses[-1] < losses[0] and all(np.isfinite(losses))}

    # 6. saved encoder loads into the downstream LCRC-A trunk; base model hash matches history
    state = mp.encoder_state(backbone)
    ckpt = CODE.parent / "cache" / "jssp_smoke_encoder.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(state, ckpt)
    norm = N.load_norm()
    hist = N.M.ssl_path(0)
    a_hist, a_info = N.M.build_model("A", norm, 0, "cpu")
    ref = "11c698ae669dbee1bca93ccc13048163e725b29659d54fe6498fa6440a0cfb1e"
    checks["historical_A_init_hash"] = {
        "hash": a_info["initial_state_sha256"],
        "matches_frozen_reference": a_info["initial_state_sha256"] == ref,
    }
    import nl_models as NM
    base_model, base_info = NM.build_model("base", norm, 0, "cpu")
    h = hashlib.sha256()
    for k, v in sorted(base_model.trunk.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().numpy(), dtype=np.float32).tobytes())
    checks["base_path_trunk_hash"] = {
        "hash": h.hexdigest(), "matches_frozen_reference": h.hexdigest() == ref,
        "params": base_info["parameters"], "head_params": base_info["parameters_head"],
        "pass": h.hexdigest() == ref and base_info["parameters"] == 237519,
    }
    m = N.M.LCRC("A", norm)
    N.M.load_ssl_encoders(m, ckpt)
    checks["jssp_load"] = {"params": int(sum(p.numel() for p in m.parameters())),
                           "atom_enc_first_sha": hashlib.sha256(
                               np.ascontiguousarray(m.atom_encoder.lins[0].weight.detach().numpy(),
                                                    dtype=np.float32).tobytes()).hexdigest()}
    checks["jssp_load"]["pass"] = checks["jssp_load"]["params"] == 237519
    checks["all_pass"] = all(v.get("pass", True) for k, v in checks.items() if isinstance(v, dict))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
