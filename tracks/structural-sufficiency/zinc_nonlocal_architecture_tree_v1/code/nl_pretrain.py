"""JSSP pretraining driver: joint 2-bond span masking on the official train 10k.

Matches the frozen masked-SSL budget used by the historical A_s0 reference:
7,560 updates, batch 128, Adam 1e-3 / wd 1e-5 / clip 5, 10k structures, last-step
encoder state.  Only the mask mechanism changes (joint span vs independent
15%/15%).  No labels are read; only raw atom/bond codes of the visible graph.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
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
import cscl_features as cf  # noqa: E402
import nl_jssp as jssp  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

TRACK = CODE.parent
CACHE = TRACK / "cache"
RESULTS = TRACK / "results" / "jssp"
DEFAULT_OBJECTS = Path(
    REPO / "tracks/structure-contribution/results/"
    "bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt"
)


def git_commit() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder_hash(state: dict) -> str:
    """Historical ``_encoder_group_hash`` format (atom:/bond:/junc: prefixes)."""
    combined = (
        {f"atom:{k}": v for k, v in state["atom_encoder"].items()}
        | {f"bond:{k}": v for k, v in state["bond_encoder"].items()}
        | {f"junc:{k}": v for k, v in state["junction_encoder"].items()}
    )
    h = hashlib.sha256()
    for k, v in sorted(combined.items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def raw_arrays_only(n_expected: int = 10000):
    """Load official train structures, keep x/edge_index/edge_attr only.

    Never touches ``.y``; the label tensor is dropped with the dataset.
    """
    ds = cf.load_official_train()
    assert len(ds) == n_expected, (len(ds), n_expected)
    raw = []
    for d in ds:
        raw.append({
            "x": d.x.numpy().copy(),
            "edge_index": d.edge_index.numpy().copy(),
            "edge_attr": d.edge_attr.numpy().copy(),
        })
    del ds
    gc.collect()
    return raw


def load_or_build_bases(raw, objects_path: Path, n_expected: int = 10000):
    if objects_path.exists():
        bases = torch.load(objects_path, map_location="cpu", weights_only=False)
        if len(bases) == n_expected and [int(o.mol_index) for o in bases] == list(range(n_expected)):
            print(f"[jssp] loaded {len(bases)} frozen objects from {objects_path}", flush=True)
            return bases, str(objects_path)
    print("[jssp] building base MolObjects for 10k train rows", flush=True)
    t0 = time.perf_counter()
    bases = [sro.build_mol_objects(i, raw[i]["x"], raw[i]["edge_index"], raw[i]["edge_attr"])
             for i in range(n_expected)]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "jssp_objects_train10k.pt"
    torch.save(bases, out)
    print(f"[jssp] built objects in {time.perf_counter() - t0:.1f}s -> {out}", flush=True)
    return bases, str(out)


def shuffle_batches(n: int, batch_size: int, gen: torch.Generator):
    order = torch.randperm(int(n), generator=gen).tolist()
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def pretrain(seed: int, device: torch.device, updates: int, objects_path: Path,
             out_dir: Path) -> dict:
    raw = raw_arrays_only()
    bases, objects_used = load_or_build_bases(raw, objects_path)
    coverage = jssp.mask_stats(bases, raw, n_mol=500, seed=seed, visit_index=0)

    cfg = src.BJOCConfig()
    torch.manual_seed(int(seed))
    backbone = src.build_bjoc(
        cfg, seed=seed, organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        device=device,
    )
    init_encoder_hash = encoder_hash(mp.encoder_state(backbone))
    pretext = mp.PretextModel(backbone).to(device)
    optimizer = torch.optim.Adam(pretext.parameters(), lr=mp.PRETRAIN_LR,
                                 weight_decay=mp.PRETRAIN_WD)
    gen = torch.Generator().manual_seed(int(seed) + mp.PRETRAIN_SHUFFLE_OFFSET)

    steps = visit = 0
    curve: list[dict] = []
    atom_frac_sum = bond_frac_sum = 0.0
    n_frac = 0
    started = time.perf_counter()
    while steps < updates:
        pretext.train()
        ce_a = ce_b = acc_a = acc_b = 0.0
        n_a = n_b = n_steps = 0
        t_ep = time.perf_counter()
        stop = False
        for chunk in shuffle_batches(len(bases), mp.PRETRAIN_BATCH, gen):
            samples = []
            for j in chunk:
                o = bases[j]
                d = raw[int(o.mol_index)]
                s = jssp.build_masked_sample(
                    o, d["x"], d["edge_index"], d["edge_attr"],
                    seed=seed, visit_index=visit,
                )
                atom_frac_sum += len(s.atom_ids) / max(o.n_atoms, 1)
                bond_frac_sum += len(s.bond_ids) / max(o.bond_index.shape[0], 1)
                n_frac += 1
                samples.append(s)
            batch, ta, tb, _, _ = mp.batch_pretext_samples(samples)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ta, tb = ta.to(device), tb.to(device)
            loss, parts = pretext.loss(batch, ta, tb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretext.parameters(), mp.PRETRAIN_CLIP)
            optimizer.step()
            steps += 1
            n_steps += 1
            with torch.no_grad():
                logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                if ta.numel():
                    ce_a += float(parts["ce_atom"])
                    acc_a += float((logits["atom_logits"].argmax(1) == ta).float().mean())
                    n_a += 1
                if tb.numel():
                    ce_b += float(parts["ce_bond"])
                    acc_b += float((logits["bond_logits"].argmax(1) == tb).float().mean())
                    n_b += 1
            if steps >= updates:
                stop = True
                break
        curve.append({
            "visit": visit, "steps_total": steps,
            "ce_atom": ce_a / max(n_a, 1), "ce_bond": ce_b / max(n_b, 1),
            "acc_atom": acc_a / max(n_a, 1), "acc_bond": acc_b / max(n_b, 1),
            "seconds": time.perf_counter() - t_ep,
        })
        if visit % 10 == 0 or stop:
            c = curve[-1]
            print(f"[jssp s{seed}] visit={visit} steps={steps}/{updates} "
                  f"ce_a={c['ce_atom']:.4f} acc_a={c['acc_atom']:.4f} "
                  f"ce_b={c['ce_bond']:.4f} acc_b={c['acc_bond']:.4f} "
                  f"{c['seconds']:.1f}s", flush=True)
        visit += 1
    wall = time.perf_counter() - started

    state = mp.encoder_state(backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"pretrain_jssp_s{seed}.pt"
    torch.save(state, ckpt)
    payload = {
        "protocol": "zinc-nonlocal-tree-v1/jssp",
        "arm": "JSSP",
        "seed": int(seed),
        "git_commit": git_commit(),
        "updates": int(steps),
        "visits": int(visit),
        "batch_size": mp.PRETRAIN_BATCH,
        "hyper": {
            "optimizer": "Adam", "lr": mp.PRETRAIN_LR, "weight_decay": mp.PRETRAIN_WD,
            "grad_clip": mp.PRETRAIN_CLIP,
            "loss": "CE(masked span atoms) + CE(masked span physical bonds), 1.0/1.0",
            "checkpoint_rule": "last step only",
        },
        "mask": {
            "mechanism": "joint 2-bond contiguous span; atoms and physical bonds of the span",
            "span_atom_ratio": jssp.SPAN_ATOM_RATIO,
            "spans_per_molecule": "max(1, round(0.05*n_atoms))",
            "atom_token": mp.ATOM_MASK_TOKEN,
            "bond_token": mp.BOND_MASK_TOKEN,
            "rng": "SeedSequence([seed, molecule_id, visit_index, 20261029])",
        },
        "coverage_500_molecules": coverage,
        "realized_coverage": {
            "atom_fraction_mean": atom_frac_sum / max(n_frac, 1),
            "bond_fraction_mean": bond_frac_sum / max(n_frac, 1),
            "n_samples": n_frac,
        },
        "n_rows": len(bases),
        "objects_used": objects_used,
        "objects_sha256": sha256_file(Path(objects_used)),
        "labels_used": False,
        "curve": curve,
        "wall_seconds": wall,
        "encoder_init_sha256": init_encoder_hash,
        "encoder_sha256": encoder_hash(state),
        "historical_init_sha256": "91af980a414ac945ea35b31c114d4e44b636419b62320321a902179112889d12",
        "init_matches_historical": init_encoder_hash ==
        "91af980a414ac945ea35b31c114d4e44b636419b62320321a902179112889d12",
        "checkpoint": str(ckpt),
    }
    (out_dir / f"pretrain_jssp_s{seed}.json").write_text(json.dumps(payload, indent=2))
    print(f"[jssp s{seed}] done {steps} updates in {wall/60:.1f} min -> {ckpt}", flush=True)
    print(f"[jssp s{seed}] encoder_sha256={payload['encoder_sha256']}", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=mp.PRETRAIN_UPDATES)
    ap.add_argument("--objects", type=Path, default=DEFAULT_OBJECTS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"[jssp] torch={torch.__version__} device={dev} "
          f"name={torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}", flush=True)
    pretrain(args.seed, dev, args.updates, args.objects, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
