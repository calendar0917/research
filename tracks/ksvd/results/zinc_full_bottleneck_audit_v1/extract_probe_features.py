"""Extract frozen graph-level features for the Phase-D conditional probes.

Read-only: builds the frozen Full soup, runs one eval pass over official
train + valid, and stores per-graph features used by ``probe_phase_d.py``.
No training, no official test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "tracks/ksvd/results/zinc_full_bottleneck_audit_v1"
CKPT = REPO / "tracks/ksvd/results/e2e_dictenv_scale_v1/checkpoints/SCALE-FULL-seed0_soup_state.pt"
SEED = 0
THREADS = 8
BATCH = 128
N_ATOM = 28


def build():
    sub = base._load_parent_subspace()
    dic = base._dictionary_tensor()
    m = sc.build_scale_model(dic, SEED, sub, sc.FULL, scale_seed=0)
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict({k: v.float() for k, v in st.items()})
    m.eval()
    return m


@torch.no_grad()
def extract(model, data, split):
    loader = p1run.p1.make_env_loader(data, BATCH, False, SEED)
    Z, NJ, H2, Y, IDS = [], [], [], [], []
    w1 = model.reader.net[0].weight
    b1 = model.reader.net[0].bias
    w2 = model.reader.net[2].weight
    b2 = model.reader.net[2].bias
    for bi, batch in enumerate(loader):
        cap = {}
        h = model.fusion.register_forward_pre_hook(
            lambda _m, inp: cap.__setitem__("z", inp[0].detach())
        )
        pred, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
        h.remove()
        z = cap["z"]                      # [n_occ, 446] pre-fusion interface
        graph = batch.batch
        n_graphs = int(graph.max().item()) + 1
        coord = model.code(batch.dict_phi)  # [n_occ, 33]
        q = F.one_hot(batch.dict_atom, num_classes=N_ATOM).to(coord.dtype)
        # graph-level pre-fusion moments
        zs = z.new_zeros((n_graphs, z.shape[1]))
        zq = z.new_zeros((n_graphs, z.shape[1]))
        zs.index_add_(0, graph, z)
        zq.index_add_(0, graph, z * z)
        # graph-level structural-code x atom-type joint moments
        outer = (coord.unsqueeze(2) * q.unsqueeze(1)).reshape(coord.shape[0], -1)  # [n_occ, 33*28]
        jo = z.new_zeros((n_graphs, outer.shape[1]))
        jo.index_add_(0, graph, outer)
        # H2 via reader weights (exact)
        rin = aux.get("reader_input") if isinstance(aux, dict) else None
        # reader input is not in aux; recompute from aux components
        rr = torch.cat([aux["unary"], aux["relation_readout"],
                        model.global_encoder(batch.global_context),
                        model.topology_encoder(batch.topology_features)], dim=1)
        h1 = torch.clamp(rr @ w1.t() + b1, min=0.0)
        h2 = torch.clamp(h1 @ w2.t() + b2, min=0.0)
        Z.append(torch.cat([zs, zq], 1).cpu().numpy())
        NJ.append(jo.cpu().numpy())
        H2.append(h2.cpu().numpy())
        Y.append(batch.y.view(-1).cpu().numpy())
        IDS.append((torch.arange(n_graphs) + bi * BATCH).numpy())
    return (np.concatenate(Z, 0).astype(np.float32), np.concatenate(NJ, 0).astype(np.float32),
            np.concatenate(H2, 0).astype(np.float32), np.concatenate(Y, 0).astype(np.float64),
            np.concatenate(IDS, 0))


def main():
    torch.set_num_threads(THREADS)
    m = build()
    tr = p1run.load_split("train")
    va = p1run.load_split("valid")
    print("extract train ...", flush=True)
    Zt, NJt, H2t, yt, idt = extract(m, tr, "train")
    print("extract valid ...", flush=True)
    Zv, NJv, H2v, yv, idv = extract(m, va, "valid")
    np.savez_compressed(OUT / "probe_features.npz",
                        Zt=Zt, NJt=NJt, H2t=H2t, yt=yt, idt=idt,
                        Zv=Zv, NJv=NJv, H2v=H2v, yv=yv, idv=idv,
                        official_test_loaded=np.array(False))
    print("shapes", Zt.shape, NJt.shape, H2t.shape, flush=True)
    (OUT / "probe_features_meta.json").write_text(json.dumps({
        "official_test_loaded": False,
        "z_dim": int(Zt.shape[1]), "node_joint_dim": int(NJt.shape[1]), "h2_dim": int(H2t.shape[1]),
        "n_train": int(Zt.shape[0]), "n_valid": int(Zv.shape[0]),
        "note": "Z = per-graph [sum(z), sum(z^2)] of the 446-D pre-fusion interface; "
                "NJ = per-graph sum of outer(coord_33, onehot(atom_type_28))",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()