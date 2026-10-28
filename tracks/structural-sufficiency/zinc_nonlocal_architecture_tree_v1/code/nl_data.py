"""Batching for the non-local tree: frozen LCRC collate plus bridge tensors."""

from __future__ import annotations

import numpy as np
import torch

import nl_bridge
import nl_common as N

_AUX_KEYS = ("term", "I_e", "c", "w", "delta_atom")


def collate(cache: dict, bridge: dict | None, rows, *, with_bridge: bool = False):
    batch = N.D.collate(cache, rows)
    if with_bridge:
        if bridge is None:
            raise RuntimeError("bridge cache required for SCE")
        batch["bridge"] = nl_bridge.collate(cache, bridge, rows)
    return batch


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: (vv.to(device, non_blocking=True) if torch.is_tensor(vv) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model, cache, bridge, rows, *, device, batch_size=256, return_aux=False,
             with_bridge=False, rows_for_aux=None):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    preds = []
    aux_parts: dict[str, list] = {}
    with N.L.C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = to_device(collate(cache, bridge, chunk, with_bridge=with_bridge), device)
            if return_aux:
                pred, aux = model(batch, return_aux=True)
                for k, v in aux.items():
                    if k in _AUX_KEYS and torch.is_tensor(v):
                        aux_parts.setdefault(k, []).append(v.detach().float().cpu())
            else:
                pred = model(batch)
            preds.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds, 0) if preds else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    mae = float(np.mean(np.abs(pred.astype(np.float64) - y))) if len(y) else float("nan")
    if not return_aux:
        return pred, mae
    aux = {}
    for k, parts in aux_parts.items():
        try:
            aux[k] = torch.cat(parts, 0).numpy()
        except Exception:
            pass
    return pred, mae, aux


def aux_stats(aux: dict, n_atoms: int) -> dict:
    """Scale/activity summaries that never use labels."""
    out = {}
    if "term" in aux:
        t = np.asarray(aux["term"], dtype=np.float64)
        out["term_mean"] = float(t.mean())
        out["term_std"] = float(t.std())
        out["term_abs_mean"] = float(np.abs(t).mean())
    if "I_e" in aux and len(aux["I_e"]):
        i = np.asarray(aux["I_e"], dtype=np.float64)
        out["I_e_mean"] = float(i.mean())
        out["I_e_std"] = float(i.std())
        out["I_e_abs_mean"] = float(np.abs(i).mean())
        out["I_e_abs_max"] = float(np.abs(i).max())
    if "c" in aux and len(aux["c"]):
        c = np.asarray(aux["c"], dtype=np.float64)
        out["c_rms"] = float(np.sqrt((c ** 2).sum(1).mean()))
        out["c_norm_mean"] = float(np.linalg.norm(c, axis=1).mean())
    if "w" in aux and len(aux["w"]):
        w = np.asarray(aux["w"], dtype=np.float64)
        out["w_mean"] = float(w.mean())
        out["w_std"] = float(w.std())
        out["w_min"] = float(w.min())
        out["w_max"] = float(w.max())
    if "delta_atom" in aux:
        d = np.asarray(aux["delta_atom"], dtype=np.float64)
        out["delta_atom_abs_mean"] = float(np.abs(d).mean())
        out["delta_atom_std"] = float(d.std())
    return out


__all__ = ["collate", "to_device", "evaluate", "aux_stats"]
