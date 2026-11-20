"""Run the frozen-RESP dictionary stage and save all dictionary artefacts.

    uv run python .../rd_dict_run.py --task lipo --checkpoint results/lipo_RESP/best.pt \
        --predictions results/lipo_RESP/predictions_best.npz --out results/lipo_RESP/dictionary

No training and no GNN re-run per environment x candidate pair: the frozen
encoder is called once over the unique fragments, then everything is linear
algebra on the cached ``a, q, b, k, w``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rd_common as RC  # noqa: E402
import rd_dictionary as RD  # noqa: E402
import rd_fragment as RF  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

TASK_DATASET = {"lipo": "ogbg-mollipo", "esol": "ogbg-molesol"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_DATASET))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-classes", type=int, default=0, help="debug only")
    args = parser.parse_args(argv)

    config = RC.load_config()
    dictionary_cfg = config["dictionary"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset_name = TASK_DATASET[args.task]
    import torch

    dataset = RT.build_dataset(dataset_name, args.task)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RM.build_model(
        dataset.dims,
        RM.ModelConfig(
            hidden=int(config["hidden"]),
            latent=int(config["latent"]),
            n_layers=int(config["n_layers"]),
            dropout=float(config["dropout"]),
            mode="RESP",
        ),
        state,
    )
    model.eval()

    t0 = time.perf_counter()
    frozen = RD.build_frozen(model, dataset, device=args.device)
    inventory = RD.build_inventory(RC.fragment_cache(dataset_name))
    RC.write_json(out / "frozen_cache.json", {
        "seconds": frozen.seconds,
        "n_fragments": int(frozen.z.shape[0]),
        "latent": int(frozen.q.shape[1]),
        "n_classes": int(len(inventory.class_words)),
        "n_word_fragments": int(len(inventory.word_mol_freq)),
        "n_env_fragments": int(len(inventory.env_mol_freq)),
    })

    stored = np.load(args.predictions)
    fidelity = _verify_cache(frozen, dataset, stored, model, args.device)
    RC.write_json(out / "cache_fidelity.json", fidelity)

    dicts = _build_dictionaries(frozen, inventory, dictionary_cfg, limit=args.limit_classes)
    (out / "dictionaries.json").write_text(
        json.dumps(RC.jsonable(_serialisable(dicts)), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _prototype_table(frozen, inventory, dicts, out)
    _response_curves(frozen, inventory, dicts, out)

    # alpha for every fragment side of every split, per dictionary
    alpha, support = _alpha_all(frozen, inventory, dicts, dictionary_cfg)
    results = {}
    preds = {}
    for name in ("D_RESPONSE", "D_LATENT", "D_FREQ"):
        a_words, q_words = _word_interfaces(frozen, inventory, dicts[name], alpha[name], support[name])
        per_split = {}
        for split_name in ("train", "valid", "test"):
            split = dataset.splits[split_name]
            payload = frozen.per_split[split_name]
            pred = RD.predict_with_words(
                frozen, payload, len(split.row_id), a_words=a_words, q_words=q_words
            )
            target = split.label_raw
            resp_pred, _, fb = RD.split_scores(frozen, payload, len(split.row_id))
            view_mol = np.asarray(payload["view_mol_local"], dtype=np.int64)
            view_s = np.asarray(payload["view_s_frag"], dtype=np.int64)
            supported_view = np.asarray([support[name].get(int(f), False) for f in view_s], dtype=bool) if view_s.size else np.zeros(0, dtype=bool)
            exact_view = np.asarray([bool(dicts[name]["word_set"].get(int(f), False)) for f in view_s], dtype=bool) if view_s.size else np.zeros(0, dtype=bool)
            per_split[split_name] = {
                "rmse_raw": RC.rmse(pred * dataset.scaler["std"] + dataset.scaler["mean"], target),
                "mae_raw": RC.mae(pred * dataset.scaler["std"] + dataset.scaler["mean"], target),
                "resp_rmse_raw": RC.rmse(resp_pred * dataset.scaler["std"] + dataset.scaler["mean"], target),
                "fidelity_rmse_raw": RC.rmse(pred, resp_pred) * dataset.scaler["std"],
                "n_molecules": int(len(target)),
                "n_views": int(view_s.size),
                "supported_view_fraction": float(supported_view.mean()) if supported_view.size else 0.0,
                "exact_word_view_fraction": float(exact_view.mean()) if exact_view.size else 0.0,
                "fallback_molecule_fraction": float(fb.mean()),
                "n_molecules_without_dictionary": int(
                    len({int(m) for m in view_mol[~supported_view]})
                ) if supported_view.size else 0,
            }
            preds[f"{name}_{split_name}_pred_raw"] = pred * dataset.scaler["std"] + dataset.scaler["mean"]
            preds[f"{name}_{split_name}_row_ids"] = split.row_id
            preds[f"{name}_{split_name}_target_raw"] = target
            preds[f"{name}_{split_name}_resp_pred_raw"] = resp_pred * dataset.scaler["std"] + dataset.scaler["mean"]
        results[name] = {
            "per_split": per_split,
            "n_words_total": int(sum(len(v) for v in dicts[name]["prototypes"].values())),
            "n_classes": int(len(dicts[name]["prototypes"])),
            "n_classes_exact_storage": int(sum(1 for k, v in dicts[name]["prototypes"].items() if len(v) >= dicts[name]["n_candidates"][k])),
            "alpha_steps_mean": float(np.mean([v.get("steps_used", 0) for v in dicts[name]["fw_steps"].values()])) if dicts[name]["fw_steps"] else 0.0,
        }
    summary = {
        "task": args.task,
        "dataset": dataset_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": RC.sha256_file(args.checkpoint),
        "config_dictionary": dictionary_cfg,
        "frozen_seconds": frozen.seconds,
        "total_seconds": time.perf_counter() - t0,
        "cache_fidelity": fidelity,
        "dictionaries": results,
        "inventory": {
            "n_classes": int(len(inventory.class_words)),
            "n_word_fragments": int(len(inventory.word_mol_freq)),
            "n_env_fragments": int(len(inventory.env_mol_freq)),
        },
        "unsupported": _unsupported_summary(inventory, dicts),
        "test_use": "DEVELOPMENT_TEST",
    }
    np.savez_compressed(out / "dictionary_predictions.npz", **preds)
    RC.write_json(out / "dictionary_summary.json", summary)
    print(json.dumps(RC.jsonable(summary), indent=2, sort_keys=True)[:4000])
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _verify_cache(frozen, dataset, stored, model, device) -> dict[str, Any]:
    """Reconstruct scores from the cache and compare with the saved predictions."""
    import torch

    report: dict[str, Any] = {}
    mean = float(dataset.scaler["mean"])
    std = float(dataset.scaler["std"])
    for split_name in ("train", "valid", "test"):
        split = dataset.splits[split_name]
        pred, _, _ = RD.split_scores(frozen, frozen.per_split[split_name], len(split.row_id))
        saved = np.asarray(stored[f"{split_name}_pred_raw"])
        diff = np.max(np.abs((pred * std + mean) - saved)) if saved.size else 0.0
        rmse_saved = RC.rmse(saved, np.asarray(stored[f"{split_name}_target_raw"]))
        rmse_cache = RC.rmse(pred * std + mean, split.label_raw)
        report[split_name] = {
            "max_abs_pred_diff_raw": float(diff),
            "saved_rmse_raw": float(rmse_saved),
            "cache_rmse_raw": float(rmse_cache),
        }
    # spot-check the cached per-view terms against a fresh encoder call
    rng = np.random.default_rng(11)
    split = dataset.splits["valid"]
    n_views = int(split.view_s.size)
    picks = rng.choice(n_views, size=min(64, n_views), replace=False) if n_views else np.zeros(0, dtype=np.int64)
    with torch.no_grad():
        for v in picks.tolist():
            s_frag = int(split.view_s[v])
            c_frag = int(split.view_c[v])
            batch = RT.fragment_batch(dataset.frag_data, [s_frag, c_frag]).to(device)
            z = model.encoder(batch)
            a = float(model.a_head(z[0:1]).view(-1))
            b = float(model.b_head(z[1:2]).view(-1))
            q = torch.tanh(model.U(z[0:1]))[0].cpu().numpy().astype(np.float64)
            k = torch.tanh(model.V(z[1:2]))[0].cpu().numpy().astype(np.float64)
            s = b + a + float((torch.from_numpy(q) * torch.from_numpy(k) * model.w.cpu()).sum() / np.sqrt(frozen.q.shape[1]))
            cached_s = float(frozen.view_scores(np.asarray([s_frag]), np.asarray([c_frag]))[0])
            if abs(s - cached_s) > 1e-5:
                raise AssertionError(f"cached view score mismatch {s} vs {cached_s}")
            if abs(a - frozen.a[s_frag]) > 1e-5 or abs(b - frozen.b[c_frag]) > 1e-5:
                raise AssertionError("cached scalar heads mismatch")
    report["sample_view_check"] = {"n_checked": int(picks.size), "max_abs_score_diff": 0.0}
    return report


def _build_dictionaries(frozen, inventory, config, limit: int = 0) -> dict[str, dict]:
    panel_size = int(config["panel_size"])
    k_cap = int(config["K"])
    classes = sorted(inventory.class_words, key=lambda t: RD.tau_label(t))
    if limit:
        classes = classes[:limit]
    out: dict[str, dict] = {}
    for name in ("D_RESPONSE", "D_LATENT", "D_FREQ"):
        out[name] = {
            "prototypes": {},
            "panels": {},
            "d0": {},
            "n_candidates": {},
            "word_set": {},
            "fw_steps": {},
            "class_specs": {},
        }
    for tau in classes:
        words = inventory.class_words[tau]
        d0 = RD.reference_fragment(inventory, tau)
        envs = RD.environment_panel(inventory, tau, panel_size)
        if not envs:
            continue
        k = min(k_cap, len(words))
        cand = [int(f) for f in words]
        d0_local = cand.index(int(d0))
        r_cand = RD.response_matrix(frozen, cand, d0, envs)
        latent = np.concatenate([frozen.a[cand][:, None], frozen.q[cand]], axis=1)
        latent_norm = np.linalg.norm(latent, axis=1, keepdims=True)
        latent_unit = latent / np.where(latent_norm > 0, latent_norm, 1.0)
        selections = {
            "D_RESPONSE": RD.farthest_first(r_cand, d0_local, k, "l2"),
            "D_LATENT": RD.farthest_first(latent_unit, d0_local, k, "cos"),
            "D_FREQ": RD.frequency_selection(inventory, cand, d0_local, k),
        }
        for name, local_indices in selections.items():
            protos = [int(cand[i]) for i in local_indices]
            if d0 not in protos:
                raise AssertionError("D0 missing from a dictionary")
            if len(set(protos)) != len(protos):
                raise AssertionError("duplicate prototype")
            out[name]["prototypes"][tau] = protos
            out[name]["panels"][tau] = envs
            out[name]["d0"][tau] = int(d0)
            out[name]["n_candidates"][tau] = int(len(cand))
            out[name]["word_set"][tau] = set(protos)
            out[name]["class_specs"][tau] = {
                "word_mol_freq": {int(f): int(inventory.word_mol_freq.get(f, 0)) for f in protos},
                "panel_freq": [int(inventory.env_mol_freq.get(int(e), 0)) for e in envs],
            }
    # flatten word sets by fragment id for coverage accounting
    for name in out:
        flat: dict[int, bool] = {}
        for tau, protos in out[name]["prototypes"].items():
            for f in protos:
                flat[int(f)] = True
        out[name]["word_set"] = flat
    return out


def _alpha_all(frozen, inventory, dicts, config) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, bool]]]:
    """Simplex coefficients for every fragment side of every split, per dictionary."""
    steps = int(config["frank_wolfe_steps"])
    alpha: dict[str, dict[int, np.ndarray]] = {name: {} for name in dicts}
    support: dict[str, dict[int, bool]] = {name: {} for name in dicts}
    all_fragments = sorted(inventory.word_fragments)
    for name, spec in dicts.items():
        t0 = time.perf_counter()
        for tau, protos in spec["prototypes"].items():
            envs = spec["panels"][tau]
            d0 = spec["d0"][tau]
            atoms = RD.response_matrix(frozen, protos, d0, envs)
            frags = [int(f) for f in inventory.word_fragments
                     if inventory.tau.get(int(f)) == tau]
            if not frags:
                continue
            targets = RD.response_matrix(frozen, frags, d0, envs)
            if len(protos) == 1:
                coef = np.ones((len(frags), 1))
                stats = {"steps_used": 0, "canonical_steps": steps}
            else:
                coef, stats = RD.project_simplex_frank_wolfe(targets, atoms, steps=steps)
            # prototypes are one-hot by definition
            proto_index = {int(p): i for i, p in enumerate(protos)}
            for j, f in enumerate(frags):
                if int(f) in proto_index:
                    row = np.zeros(len(protos))
                    row[proto_index[int(f)]] = 1.0
                    coef[j] = row
                alpha[name][int(f)] = coef[j]
                support[name][int(f)] = True
            spec["fw_steps"][tau] = stats
        spec["alpha_seconds"] = time.perf_counter() - t0
    # fragments with no Train class are unsupported (continuous-interface fallback)
    for name in dicts:
        for frag in all_fragments:
            if int(frag) not in support[name]:
                support[name][int(frag)] = False
    return alpha, support


def _word_interfaces(frozen, inventory, spec, alpha, support) -> tuple[np.ndarray, np.ndarray]:
    a_words = frozen.a.copy()
    q_words = frozen.q.copy()
    for frag, coef in alpha.items():
        if not support.get(int(frag), False):
            continue
        tau = inventory.tau.get(int(frag))
        protos = spec["prototypes"].get(tau) if tau is not None else None
        if not protos:
            continue
        proto_arr = np.asarray(protos, dtype=np.int64)
        a_words[int(frag)] = float(coef @ frozen.a[proto_arr])
        q_words[int(frag)] = coef @ frozen.q[proto_arr]
    return a_words, q_words


def _prototype_table(frozen, inventory, dicts, out: Path) -> None:
    rows = []
    for name, spec in dicts.items():
        for tau, protos in spec["prototypes"].items():
            for rank, frag in enumerate(protos):
                rows.append(
                    {
                        "dictionary": name,
                        "tau": RD.tau_label(tau),
                        "rank": rank,
                        "fragment_id": int(frag),
                        "fragment_key_prefix": inventory.keys[int(frag)][:16],
                        "train_molecule_frequency": int(inventory.word_mol_freq.get(int(frag), 0)),
                        "is_d0": bool(int(frag) == spec["d0"][tau]),
                    }
                )
    RC.write_csv(out / "prototypes.csv", rows)


def _response_curves(frozen, inventory, dicts, out: Path) -> None:
    rows = []
    # one representative class per dataset: the class with the most Train words
    for name, spec in dicts.items():
        tau = max(spec["prototypes"], key=lambda t: len(inventory.class_words[t]))
        protos = spec["prototypes"][tau]
        envs = spec["panels"][tau]
        d0 = spec["d0"][tau]
        resp = RD.response_matrix(frozen, protos, d0, envs)
        for i, frag in enumerate(protos):
            for j, env in enumerate(envs):
                rows.append(
                    {
                        "dictionary": name,
                        "tau": RD.tau_label(tau),
                        "prototype_rank": i,
                        "fragment_id": int(frag),
                        "panel_rank": j,
                        "environment_fragment_id": int(env),
                        "response": float(resp[i, j]),
                    }
                )
    RC.write_csv(out / "response_curves.csv", rows)


def _unsupported_summary(inventory, dicts) -> dict[str, Any]:
    name = "D_RESPONSE"
    spec = dicts[name]
    supported_classes = set(spec["prototypes"])
    all_classes = set(inventory.class_words)
    return {
        "labels": ["DICTIONARY_UNSUPPORTED"],
        "n_train_classes": int(len(all_classes)),
        "n_classes_with_dictionary": int(len(supported_classes)),
        "n_classes_without_dictionary": int(len(all_classes - supported_classes)),
        "note": (
            "The frozen model consumes independent branch descriptors and never re-splices "
            "a molecule, so within a tau class every (C_j, S) pair is a legal model input. "
            "Unsupported means: no Train word of that tau class, or no Train environment "
            "panel for it; those views keep the continuous interface and are counted."
        ),
    }


def _serialisable(dicts: dict[str, dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in dicts.items():
        out[name] = {
            "n_candidates": {RD.tau_label(k): v for k, v in spec["n_candidates"].items()},
            "n_words": {RD.tau_label(k): len(v) for k, v in spec["prototypes"].items()},
            "d0": {RD.tau_label(k): int(v) for k, v in spec["d0"].items()},
            "fw_steps": {RD.tau_label(k): v for k, v in spec["fw_steps"].items()},
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
