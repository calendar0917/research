"""CPU correctness gates for the local-edit round.

Two entry points:

* ``--stage pre``  (before any GPU work): byte hashes, cache/atom alignment,
  raw-unit formula, C1 column deletion, wrapper <-> frozen-upstream forward
  equivalence on the pilot checkpoint (equivalence replay only, no metric is
  reported), label-access refusals, core mapping/symmetry, partitioning and
  bootstrap identity.
* ``--stage post`` (after the six units are pulled): checkpoint-state
  consistency, atom-renumbering invariance, pair reverse sign flip and
  batch split/merge on the *trained* checkpoints (delegated to
  :mod:`edit_trace`), so that the same file carries both stages.

Every entry is a named check with an explicit ``ok`` flag; a failure sets the
top-level ``ok`` to false and the round stops before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import edit_common as C
import edit_data as ED
import edit_graph as G
import edit_split as SP
import edit_trace as T

import dissect_common as DC
import dissect_model as M

OUT = C.ANALYSIS / "correctness_gates.json"


class Gate:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def run(self, name: str, fn: Callable[[], Any], note: str = "") -> Any:
        try:
            detail = fn()
            self.items.append({"name": name, "ok": True, "note": note, "detail": detail})
            return detail
        except Exception as exc:  # noqa: BLE001
            self.items.append(
                {"name": name, "ok": False, "note": note, "error": f"{type(exc).__name__}: {exc}"}
            )
            return None

    @property
    def ok(self) -> bool:
        return all(i["ok"] for i in self.items)


# ---------------------------------------------------------------------------
# pre-training gates
# ---------------------------------------------------------------------------


def gate_sources() -> dict[str, Any]:
    got = {
        "csv": C.sha256_file(DC.LIPOP_CSV),
        "split_file": C.sha256_file(DC.CHEMBENCH_LIPOP / C.SPLIT_FILE),
        "raw_artifact": C.sha256_file(C.RAW_ARTIFACT),
    }
    expected = {
        "csv": C.LIPOP_CSV_SHA256,
        "split_file": C.SPLIT_FILE_SHA256,
        "raw_artifact": C.RAW_ARTIFACT_SHA256,
    }
    if got != expected:
        raise ValueError(f"source hashes differ: got {got} expected {expected}")
    import dissect_source as S

    rows = list(S.load_lipop(C.DATA_REPEAT, labels="none")["rows"]["train"])
    if C.row_sha256(rows) != C.TRAIN_ROWS_SHA256:
        raise ValueError("random122 Train row sequence hash differs")
    return {"hashes": got, "train_rows_sha256": C.TRAIN_ROWS_SHA256, "n_train_rows": len(rows)}


def gate_partitions() -> dict[str, Any]:
    payload = ED.split_payload()
    rows = payload["rows_by_partition"]
    seen: dict[int, str] = {}
    for part in C.PARTITIONS:
        for row in rows[part]:
            if int(row) in seen:
                raise ValueError(f"row {row} appears in {seen[int(row)]} and {part}")
            seen[int(row)] = part
    if len(seen) != C.TRAIN_ROWS:
        raise ValueError(f"partitions cover {len(seen)} rows, expected {C.TRAIN_ROWS}")
    pairs = T.load_pairs()
    straddle = [p["pair_id"] for p in pairs if seen[p["row_a"]] != seen[p["row_b"]]]
    if straddle:
        raise ValueError(f"{len(straddle)} pairs straddle partitions")
    units: dict[int, set[str]] = {}
    split = C.read_json(C.ANALYSIS / "component_split.json")
    for part in C.PARTITIONS:
        for unit in split["units_per_partition"][part]:
            units.setdefault(int(unit), set()).add(part)
    multi = {u: sorted(v) for u, v in units.items() if len(v) > 1}
    if multi:
        raise ValueError(f"{len(multi)} units are split across partitions")
    return {
        "n_rows": len(seen),
        "n_pairs": len(pairs),
        "n_units": len(units),
        "per_partition": {p: payload["partitions"][p]["n_rows"] for p in C.PARTITIONS},
        "pairs_per_partition": {p: payload["partitions"][p]["n_pairs"] for p in C.PARTITIONS},
        "status": payload["status"],
    }


def gate_cache_alignment() -> dict[str, Any]:
    """Cache atom indices must equal RDKit atom indices (pairs store RDKit indices)."""
    from rdkit import Chem

    import dissect_source as S
    import mb_raw_artifact as RAW

    smiles = S.load_lipop(C.DATA_REPEAT, labels="none")["smiles"]
    raw = RAW.load(verify=True)["rows"]["Lipophilicity"]
    checked = 0
    max_atom_delta = 0
    worst = None
    for partition in C.PARTITIONS:
        cache = ED.load_cache(partition)
        atom_ptr = np.asarray(cache["atom_ptr"])
        types = np.asarray(cache["atom_type"])
        for pos, row in enumerate(np.asarray(cache["rows"]).tolist()):
            a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
            mol = Chem.MolFromSmiles(smiles[int(row)])
            if mol is None:
                raise ValueError(f"row {row}: unparseable")
            if a1 - a0 != mol.GetNumAtoms():
                raise ValueError(f"row {row}: cache has {a1 - a0} atoms, RDKit has {mol.GetNumAtoms()}")
            art = np.asarray(raw[int(row)]["atom_class"], dtype=np.int64)
            if not np.array_equal(types[a0:a1], art):
                bad = int((types[a0:a1] != art).sum())
                if bad > max_atom_delta:
                    max_atom_delta = bad
                    worst = int(row)
                raise ValueError(f"row {row}: cache atom_type differs from the raw artifact in {bad} atoms")
            checked += 1
    return {"n_molecules_checked": checked, "atom_order_identical_to_rdkit": True, "worst_row": worst}


def gate_raw_unit_formula() -> dict[str, Any]:
    """pred = mu + sigma*(L + rho); sum_v f(h_v) = L; and the per-atom identity."""
    spec = ED.frozen_spec()
    cache = ED.load_cache("fit")
    tgt = ED.target_norm()
    mu, sigma = float(tgt["mean"]), float(tgt["scale"])
    model = M.DissectLCRC(spec, ED.train_norm(), "A0")
    model.eval()
    rows = list(range(32))
    batch = ED.to_device(ED.collate(cache, rows), torch.device("cpu"))
    with torch.no_grad():
        y, aux = model(batch, return_aux=True)
    contrib_sum = aux["contrib"].numpy()
    atom_sum = np.zeros_like(contrib_sum)
    np.add.at(atom_sum, batch["atom_batch"].numpy(), aux["contrib_atom"].numpy())
    rho = aux["rho"].numpy()
    max_atom_gap = float(np.abs(atom_sum - contrib_sum).max())
    max_head_gap = float(np.abs(contrib_sum + rho - y.numpy()).max())
    raw = mu + sigma * y.numpy()
    max_raw_gap = float(np.abs(raw - (mu + sigma * (contrib_sum + rho))).max())
    if max(max_atom_gap, max_head_gap, max_raw_gap) > 1e-6:
        raise ValueError(
            f"identity violated: atom->L {max_atom_gap}, L+rho->y {max_head_gap}, raw {max_raw_gap}"
        )
    return {
        "n_molecules": len(rows),
        "max_abs_sum_f_minus_L": max_atom_gap,
        "max_abs_L_plus_rho_minus_y": max_head_gap,
        "max_abs_raw_gap": max_raw_gap,
        "mu": mu,
        "sigma": sigma,
        "formula": "pred = mu + sigma*(sum_v f(h_v) + rho)",
    }


def gate_c1_column_deletion() -> dict[str, Any]:
    spec = ED.frozen_spec()
    norm = ED.train_norm()
    cache = ED.load_cache("select")
    full_state = M.build_full_state(spec, norm, 0)
    c1_state = M.remap_state("C1", full_state)
    shared = M.shared_state_equal("C1", full_state, c1_state)
    if not shared["all_equal"]:
        raise ValueError(f"shared parameters differ: {[k for k, v in shared['checks'].items() if not v]}")
    full = M.DissectLCRC(spec, norm, "A0")
    full.load_state_dict(full_state, strict=True)
    full.eval()
    c1 = M.DissectLCRC(spec, norm, "C1")
    c1.load_state_dict(c1_state, strict=True)
    c1.eval()
    if M.parameter_count(full) != C.EXPECTED_PARAMS["A0"] or M.parameter_count(c1) != C.EXPECTED_PARAMS["C1"]:
        raise ValueError("parameter counts differ from the frozen values")
    rows = list(range(16))
    batch = ED.to_device(ED.collate(cache, rows), torch.device("cpu"))
    with torch.no_grad():
        _, aux_full = full(batch, return_aux=True)
        _, aux_c1 = c1(batch, return_aux=True)
    lo, hi = DC.LOCAL_LAYOUT["b_mean"]
    cols = list(range(lo, hi))
    keep = [i for i in range(DC.LOCAL_DIM) if i not in cols]
    if len(keep) != aux_c1["local_input"].shape[1]:
        raise ValueError("C1 local width does not equal FULL minus the b_mean block")
    x_full = aux_full["local_input"].numpy()
    x_c1 = aux_c1["local_input"].numpy()
    gap = float(np.abs(x_full[:, keep] - x_c1).max())
    if gap != 0.0:
        raise ValueError(f"C1 local input is not the column subset of FULL (max gap {gap})")
    w_gap = float(
        (c1.local.mlp.lins[0].weight.detach() - full.local.mlp.lins[0].weight.detach()[:, keep]).abs().max()
    )
    if w_gap != 0.0:
        raise ValueError(f"C1 first-layer weights are not the column subset (max gap {w_gap})")
    # drop the b_mean block from FULL's input and run the *same* head: must equal C1
    x_zero = x_full.copy()
    x_zero[:, cols] = 0.0
    with torch.no_grad():
        h_full = full.local.norm(full.local.mlp(torch.as_tensor(x_zero, dtype=torch.float32)))
        f_atoms = full.f(h_full).view(-1)
        res = torch.zeros(int(batch["n_graphs"]), dtype=torch.float32).index_add_(
            0, batch["atom_batch"], f_atoms
        )
    c1_local = aux_c1["contrib"].numpy()
    gap2 = float(np.abs(res.numpy() - c1_local).max())
    # the column subset and the weight subset are exact; the remaining gap is fp32
    # accumulation order (529-wide GEMM with zeros vs 433-wide GEMM)
    if gap2 > 1e-5:
        raise ValueError(f"zeroing the b_mean block does not reproduce C1 (max gap {gap2})")
    return {
        "parameters": {"A0": M.parameter_count(full), "C1": M.parameter_count(c1)},
        "kept_full_columns": len(keep),
        "max_local_input_gap": gap,
        "max_first_layer_weight_gap": w_gap,
        "max_contrib_gap_zeroed_bmean": gap2,
        "zeroed_bmean_tolerance": 1e-5,
        "note": "the column/weight subsets are bit-identical; the residual is fp32 GEMM accumulation order",
        "shared_state_all_equal": bool(shared["all_equal"]),
        "n_shared_checks": shared["n_checked"],
    }


def gate_frozen_equivalence() -> dict[str, Any]:
    """A0 wrapper vs the frozen upstream ``lcrc_model.LCRC`` on the pilot checkpoint.

    Equivalence replay only: no metric of this checkpoint is used as a result.
    """
    spec = ED.frozen_spec()
    norm = ED.train_norm()
    ckpt_path = C.REPO / DC.REFERENCE["checkpoint"]
    if not ckpt_path.exists():
        return {"available": False, "note": "pilot checkpoint absent; equivalence replay skipped"}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    import lcrc_model as UP

    up_model = UP.LCRC(spec, norm)
    ours = M.DissectLCRC(spec, norm, "A0")
    missing, unexpected = ours.load_state_dict(state, strict=False)
    up_model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"state keys differ: missing {missing} unexpected {unexpected}")
    up_model.eval()
    ours.eval()
    cache = ED.load_cache("select")
    rows = list(range(24))
    batch = ED.to_device(ED.collate(cache, rows), torch.device("cpu"))
    with torch.no_grad():
        y_up = up_model(batch)
        y_ours = ours(batch)
    gap = float((y_up - y_ours).abs().max())
    if gap != 0.0:
        raise ValueError(f"wrapper differs from the frozen upstream forward (max gap {gap})")
    return {
        "available": True,
        "checkpoint": C.track_rel(ckpt_path),
        "checkpoint_sha256": C.sha256_file(ckpt_path),
        "max_abs_forward_gap": gap,
        "n_molecules": len(rows),
        "note": "equivalence replay only; the pilot checkpoint trained on rows that include Fit molecules",
    }


def gate_label_permissions() -> dict[str, Any]:
    refused = {}
    for partition, purpose in (
        ("diagnose", "train"),
        ("audit", "train"),
        ("audit", "select"),
        ("audit", "probe_fit"),
        ("diagnose", "select"),
        ("select", "train"),
    ):
        try:
            ED.labels(partition, purpose)
            refused[f"{partition}:{purpose}"] = "ALLOWED (unexpected)"
        except PermissionError:
            refused[f"{partition}:{purpose}"] = "refused"
    bad = [k for k, v in refused.items() if v != "refused"]
    if bad:
        raise ValueError(f"label scope violated for {bad}")
    ledger = C.read_json(ED.LEDGER) if ED.LEDGER.exists() else {"events": []}
    return {
        "refusals": refused,
        "audit_labels_accessed": ED.audit_labels_accessed(),
        "n_ledger_events": len(ledger["events"]),
    }


def gate_core_mapping() -> dict[str, Any]:
    """Every stored mapping is a label-preserving isomorphism with the attachment marker."""
    from rdkit import Chem

    import dissect_source as S

    smiles = S.load_lipop(C.DATA_REPEAT, labels="none")["smiles"]
    pairs = T.load_pairs()
    checked = 0
    n_multi = 0
    for pair in pairs:
        m = pair["mapping"]
        if sorted(m.keys()) != sorted(pair["core_atoms_a"]):
            raise ValueError(f"{pair['pair_id']}: mapping domain != core atoms of A")
        if sorted(m.values()) != sorted(pair["core_atoms_b"]):
            raise ValueError(f"{pair['pair_id']}: mapping image != core atoms of B")
        if len(set(m.values())) != len(m):
            raise ValueError(f"{pair['pair_id']}: mapping is not injective")
        mol_a = Chem.MolFromSmiles(smiles[pair["row_a"]])
        mol_b = Chem.MolFromSmiles(smiles[pair["row_b"]])
        ga, att_a = _graph_of_mol(mol_a, pair["core_atoms_a"], pair["attach_a"])
        gb, att_b = _graph_of_mol(mol_b, pair["core_atoms_b"], pair["attach_b"])
        ka = sorted(int(a) for a in pair["core_atoms_a"])
        kb = sorted(int(a) for a in pair["core_atoms_b"])
        pos_b = {a: i for i, a in enumerate(kb)}
        local_map = {i: pos_b[int(m[a])] for i, a in enumerate(ka)}
        if not _is_valid_isomorphism(ga, gb, local_map):
            raise ValueError(f"{pair['pair_id']}: stored mapping is not an isomorphism")
        if local_map[int(att_a)] != int(att_b):
            raise ValueError(f"{pair['pair_id']}: mapping does not send attachment atom to attachment atom")
        if pair["n_automorphisms_capped"] > 1:
            n_multi += 1
        checked += 1
    return {
        "n_pairs_checked": checked,
        "n_pairs_with_non_unique_mapping": n_multi,
        "note": "the summed core term does not depend on which isomorphism is used",
    }


def _graph_of_mol(mol, atoms: Sequence[int], attach: int) -> tuple[G.LGraph, int]:
    """Labelled induced subgraph on ``atoms``; the attachment atom is marked."""
    keep = sorted(int(a) for a in atoms)
    local = {a: i for i, a in enumerate(keep)}
    if int(attach) not in local:
        raise ValueError("attachment atom not inside the core")
    labels = []
    for a in keep:
        atom = mol.GetAtomWithIdx(a)
        labels.append(
            (
                atom.GetSymbol(),
                bool(atom.GetIsAromatic()),
                int(atom.GetFormalCharge()),
                int(atom.GetTotalNumHs()),
                int(a) == int(attach),
            )
        )
    edges = []
    for bond in mol.GetBonds():
        u, v = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        if u in local and v in local:
            a, b = local[u], local[v]
            if a > b:
                a, b = b, a
            val = bond.GetBondTypeAsDouble()
            edges.append((a, b, 1 if val == 1.0 else 2 if val == 2.0 else 3 if val == 3.0 else 4 if val == 1.5 else 5))
    return G.LGraph(labels, edges), local[int(attach)]


def _is_valid_isomorphism(ga: G.LGraph, gb: G.LGraph, mapping: Mapping[int, int]) -> bool:
    if ga.n != gb.n:
        return False
    for i in range(ga.n):
        if ga.labels[i][:4] != gb.labels[mapping[i]][:4]:
            return False
    for u, v, lab in ga.edges:
        if gb.edge_key(mapping[u], mapping[v]) != lab:
            return False
    return len(gb.edges) == len(ga.edges)


def gate_renumbering_structure() -> dict[str, Any]:
    """The core class of a pair is invariant under atom renumbering."""
    from rdkit import Chem

    import dissect_source as S

    smiles = S.load_lipop(C.DATA_REPEAT, labels="none")["smiles"]
    pairs = T.load_pairs()[:60]
    for pair in pairs:
        mol = Chem.MolFromSmiles(smiles[pair["row_a"]])
        n = mol.GetNumAtoms()
        order = list(range(n))[::-1]  # new atom i is old atom order[i]
        ren = Chem.RenumberAtoms(mol, order)
        position = {old: new for new, old in enumerate(order)}
        new_atoms = sorted(position[a] for a in pair["core_atoms_a"])
        new_attach = position[pair["attach_a"]]
        g0, a0 = _graph_of_mol(mol, pair["core_atoms_a"], pair["attach_a"])
        g1, a1 = _graph_of_mol(ren, new_atoms, new_attach)
        if g0.wl_hash() != g1.wl_hash() or not g0.is_isomorphic(g1):
            raise ValueError(f"{pair['pair_id']}: core class changed after atom renumbering")
        att0 = [i for i, lab in enumerate(g0.labels) if lab[4]]
        att1 = [i for i, lab in enumerate(g1.labels) if lab[4]]
        if len(att0) != 1 or len(att1) != 1:
            raise ValueError(f"{pair['pair_id']}: attachment marker is not unique")
        if g0.labels[att0[0]][:4] != g1.labels[att1[0]][:4]:
            raise ValueError(f"{pair['pair_id']}: attachment atom label changed after renumbering")
        if g0.degree(att0[0]) != g1.degree(att1[0]):
            raise ValueError(f"{pair['pair_id']}: attachment degree changed after renumbering")
        del a0, a1
    return {"n_pairs_checked": len(pairs), "core_class_invariant": True, "attachment_position_invariant": True}


def gate_bootstrap_identity() -> dict[str, Any]:
    """The shared component resampler is deterministic and identical for every method."""
    split = C.read_json(C.ANALYSIS / "component_split.json")
    row_to_component = {int(k): int(v) for k, v in split["row_to_component"].items()}
    pairs = [p for p in T.load_pairs() if p["partition"] == "audit"]
    comps = sorted({row_to_component[p["row_a"]] for p in pairs})
    if any(row_to_component[p["row_b"]] not in comps for p in pairs):
        raise ValueError("audit pairs are not contained in audit components")
    resample = _component_resample(comps, C.BOOTSTRAP_SEED, 64)
    again = _component_resample(comps, C.BOOTSTRAP_SEED, 64)
    if not np.array_equal(resample, again):
        raise ValueError("component resampler is not deterministic")
    if resample.max() >= len(comps):
        raise ValueError("resampled index outside the component list")
    return {
        "n_audit_pairs": len(pairs),
        "n_audit_pair_components": len(comps),
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "replicates_checked": 64,
        "identical_across_calls": True,
        "note": (
            "the bootstrap resamples *pair components* (indivisible units), not pairs, and every model, "
            "seed and probe method of one replicate uses the same index array"
        ),
    }


def _component_resample(components: Sequence[int], seed: int, replicates: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(components)
    return np.stack([rng.integers(0, n, size=n) for _ in range(replicates)])


# ---------------------------------------------------------------------------
# post-training gates
# ---------------------------------------------------------------------------


def gate_trained_checkpoints() -> dict[str, Any]:
    """State consistency of the six units and the traceback invariants on real weights."""
    report = {}
    pairs = T.load_pairs()
    caches = {p: ED.load_cache(p) for p in C.PARTITIONS}
    sigma = float(ED.target_norm()["scale"])
    for model_variant in C.MODELS:
        for seed in C.SEEDS:
            path = C.run_dir(model_variant, seed) / "checkpoint.pt"
            if not path.exists():
                raise FileNotFoundError(f"missing {path}")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if int(ckpt["parameters"]) != C.EXPECTED_PARAMS[model_variant]:
                raise ValueError(f"{model_variant}: {ckpt['parameters']} parameters")
            if float(ckpt["best_metric_value"]) != float(ckpt["best_metric_value"]):
                raise ValueError("non-finite best metric")
            if ckpt["best_metric_name"] != "select_rmse":
                raise ValueError("checkpoint was not selected on the Select partition")
            model, _ = T.load_model(model_variant, seed)
            report[f"{model_variant}:s{seed}"] = T.verify_all(pairs, model, caches, sigma)
    return report


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU correctness gates")
    ap.add_argument("--stage", choices=["pre", "post", "both"], default="pre")
    args = ap.parse_args(argv)

    gates = Gate()
    if args.stage in ("pre", "both"):
        gates.run("sources", gate_sources, "frozen bytes of the public data")
        gates.run("partitions", gate_partitions, "disjoint, complete, pairs never straddle")
        gates.run("cache_alignment", gate_cache_alignment, "cache atom order == RDKit atom order")
        gates.run("raw_unit_formula", gate_raw_unit_formula, "pred = mu + sigma*(L+rho)")
        gates.run("c1_column_deletion", gate_c1_column_deletion, "C1 = FULL minus the b_mean block")
        gates.run("frozen_equivalence", gate_frozen_equivalence, "A0 wrapper == upstream LCRC (replay)")
        gates.run("label_permissions", gate_label_permissions, "purpose scope + Audit one-shot guard")
        gates.run("core_mapping", gate_core_mapping, "stored mappings are labelled isomorphisms")
        gates.run("renumbering_structure", gate_renumbering_structure, "core class invariant")
        gates.run("bootstrap_identity", gate_bootstrap_identity, "deterministic shared resampler")
    if args.stage in ("post", "both"):
        gates.run("trained_checkpoints", gate_trained_checkpoints, "6 units + trace invariants on real weights")

    payload = {
        "schema": "lcrc-local-edit-correctness-gates-v1",
        "round": C.ROUND_ID,
        "stage": args.stage,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "ok": gates.ok,
        "n_checks": len(gates.items),
        "n_failed": sum(0 if i["ok"] else 1 for i in gates.items),
        "checks": gates.items,
    }
    previous = C.read_json(OUT) if OUT.exists() else None
    if previous and args.stage in ("pre", "post") and previous.get("stage") not in (args.stage, "both"):
        payload["previous_stages"] = {"stage": previous.get("stage"), "ok": previous.get("ok"), "n_checks": previous.get("n_checks")}
        payload["ok"] = bool(payload["ok"] and previous.get("ok"))
    C.write_json(OUT, payload)
    for item in gates.items:
        print(f"{'ok  ' if item['ok'] else 'FAIL'} {item['name']}: {item.get('error') or json.dumps(item['detail'])[:180]}")
    print("ALL GATES OK" if payload["ok"] else "GATE FAILURE")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
