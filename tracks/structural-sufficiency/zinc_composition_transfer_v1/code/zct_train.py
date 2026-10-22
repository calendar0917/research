"""Recipe-B P3 trainer for the composition-transfer round (arms E / R / S).

One process trains one arm of one seed.  Everything that defines the platform is
reused, not re-implemented:

* model: ``zinc_recipe_trunk_v2/ad2_models.P3NullModel`` (413,505 parameters,
  the P3-Null computation graph) with the frozen **seed-0** train-only masked-SSL
  encoders loaded for every arm and both supervised seeds;
* data: the frozen ``zinc_arch_discovery_v1`` cache and
  ``ad_train.collate(..., candidate="P3N")``;
* recipe: recipe **B** (``ad_train.RECIPE_VARIANTS["B"]``), AdamW, batch 128,
  600 epochs (79 updates/epoch, 47,400 outer updates), 500-step linear warm-up,
  per-step cosine 1e-3 -> 1e-5, weight decay 1e-5, betas (0.9,0.999),
  eps 1e-8, outer gradient clip 5, FP32, dropout 0;
* schedule index: ``ad_common.cosine_with_warmup_lr(step, total, warmup, ...)``
  applied to update ``step`` (the v1 trainer's declared convention);
* selection: per-seed best valid, strict ``<``, earlier epoch on a tie;
* evaluation: ``ad_train.evaluate`` on the full official valid split every epoch.

Arms:

* ``E``: ordinary ERM on the whole 128-molecule batch (raw-y L1).
* ``R``: balanced random source/query halves; exact one-step look-ahead objective
  ``J = (|A|/|B|) L_A(theta) + (|Q|/|B|) L_Q(theta')`` with
  ``theta' = theta - alpha_t * min(1, 5/(||g_A||+eps)) * g_A`` and
  ``alpha_t = kappa * lr_t * ramp_t``, differentiated exactly (no detach, no
  first-order approximation).
* ``S``: identical to R except the partition is the structural greedy split.

The 50-epoch ERM prefix is shared per seed: the ``E`` job writes ``fork.pt`` at
the end of epoch 50 (model + optimizer + generator + step) and continues to 600;
``R``/``S`` resume from that exact fork with the recipe's own schedule, so the
only difference is the objective (and, for S, the partition).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_V2 = _SSUF / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import zct_split as SP  # noqa: E402

TASK = _TASK
RUNS = TASK / "runs"
ANALYSIS = TASK / "analysis"
CACHE = TASK / "cache"

RECIPE = dict(T1.RECIPE_VARIANTS["B"])
BATCH_SIZE = int(RECIPE["batch_size"])
EPOCHS = int(RECIPE["epochs"])
WARMUP = int(RECIPE["warmup_steps"])
LR = float(RECIPE["lr"])
MIN_LR = float(RECIPE["min_lr"])
WD = float(RECIPE["weight_decay"])
CLIP = float(RECIPE["grad_clip"])
PREFIX_EPOCHS = 50
RAMP_START, RAMP_END = 51, 100
EPS = 1e-8
BETAS = (0.9, 0.999)
ARMS = ("E", "R", "S")


def build_model(norm: dict, seed: int, device, ssl_path: Path | None = None):
    """Same construction order as ``ad2_models.build_model('P3N')`` but with an
    explicit SSL checkpoint, so both supervised seeds can share the seed-0 one."""
    ssl_path = Path(C.P3_SSL_S0 if ssl_path is None else ssl_path)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    model = A2.P3NullModel(norm).to(device)
    if not ssl_path.exists():
        raise RuntimeError(f"frozen SSL encoder missing: {ssl_path}")
    info = {"source": str(ssl_path), "sha256": C.sha256_file(ssl_path)}
    A2._load_encoders(model, ssl_path, device)
    info["parameters"] = C.parameter_count(model)
    if info["parameters"] > 500_000:
        raise RuntimeError("parameter cap exceeded")
    return model, info


def ramp(epoch: int) -> float:
    if epoch < RAMP_START:
        return 0.0
    if epoch >= RAMP_END:
        return 1.0
    return float(epoch - RAMP_START + 1) / float(RAMP_END - RAMP_START + 1)


def lr_at(step: int, total: int) -> float:
    return C.cosine_with_warmup_lr(step, total, WARMUP, LR, MIN_LR)


def _grads_of(loss, params):
    return torch.autograd.grad(loss, params, create_graph=True, allow_unused=False)


def meta_grads(model, params: dict, batch_s, target_s, batch_q, target_q, alpha: float):
    """Exact look-ahead objective and its gradient.

    ``J = (n_s/n) L_s(theta) + (n_q/n) L_q(theta - alpha * min(1, 5/(||g_s||+eps)) g_s)``
    differentiated through the inner gradient (``create_graph=True``); nothing is
    detached and no first-order approximation is taken.  Returns the gradient
    tensors, J, both losses and the inner-step diagnostics.
    """
    names = list(params)
    values = [params[k] for k in names]
    pred_s = torch.func.functional_call(model, params, (batch_s,))
    loss_s = F.l1_loss(pred_s.view(-1), target_s.view(-1))
    g = torch.autograd.grad(loss_s, values, create_graph=True, allow_unused=False)
    gnorm = torch.sqrt(sum((gi * gi).sum() for gi in g))
    c = torch.clamp(CLIP / (gnorm + EPS), max=1.0)
    shifted = {k: p - alpha * c * gi for k, p, gi in zip(names, values, g)}
    pred_q = torch.func.functional_call(model, shifted, (batch_q,))
    loss_q = F.l1_loss(pred_q.view(-1), target_q.view(-1))
    n_s = int(batch_s["n_graphs"])
    n_q = int(batch_q["n_graphs"])
    n_b = n_s + n_q
    j = (n_s / n_b) * loss_s + (n_q / n_b) * loss_q
    grads = torch.autograd.grad(j, values, allow_unused=False)
    return grads, j, loss_s, loss_q, gnorm, c


class Trainer:
    def __init__(self, arm: str, seed: int, device, out_dir: Path, kappa: float = 0.0,
                 partition: dict | None = None, eval_train_from: int | None = None,
                 diag: dict | None = None, cols: list | None = None):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.seed = int(seed)
        self.device = device
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.kappa = float(kappa)
        self.partition = partition
        self.eval_train_from = eval_train_from
        self.diag = diag or {}
        self.norm = D.load_norm()
        self.train_cache = D.load_cache("train")
        self.valid_cache = D.load_cache("valid")
        self.y_train = np.asarray(self.train_cache["labels"], dtype=np.float64)
        self.y_valid = np.asarray(self.valid_cache["labels"], dtype=np.float64)
        self.rows_train = np.arange(len(self.y_train), dtype=np.int64)
        self.rows_valid = np.arange(len(self.y_valid), dtype=np.int64)
        self.n_train = len(self.y_train)
        self.steps_per_epoch = math.ceil(self.n_train / BATCH_SIZE)
        self.total_updates = self.steps_per_epoch * EPOCHS
        self.model, self.model_info = build_model(self.norm, self.seed, device)
        self.init_state_sha256 = C.state_hash(self.model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
        self.gen = torch.Generator().manual_seed(self.seed + 101)
        self.global_step = 0
        self.start_epoch = 1
        self.curve: list[dict] = []
        self.best = {"valid_mae": float("inf"), "epoch": -1}
        self.best_state = None
        self.diag_log: list[dict] = []

    # -- checkpoint helpers ------------------------------------------------
    def save_best(self, valid_mae: float, epoch: int) -> None:
        self.best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
        self.best_state = C.cpu_state(self.model)
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(self.model), "epoch": int(epoch),
             "valid_mae": float(valid_mae), "arm": self.arm, "seed": self.seed,
             "step": int(self.global_step)},
            self.out / "best.pt",
        )

    def save_fork(self, epoch: int) -> None:
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(self.model),
             "optimizer": self.optimizer.state_dict(),
             "gen_state": self.gen.get_state(),
             "epoch": int(epoch), "global_step": int(self.global_step),
             "best": dict(self.best), "best_state": self.best_state,
             "curve": list(self.curve),
             "init_state_sha256": self.init_state_sha256,
             "model_info": self.model_info},
            self.out / "fork.pt",
        )

    def load_fork(self, path: Path) -> dict:
        fork = torch.load(path, map_location="cpu", weights_only=False)
        C.load_state(self.model, fork["state_dict"], self.device)
        self.optimizer.load_state_dict(fork["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        self.gen.set_state(fork["gen_state"])
        self.global_step = int(fork["global_step"])
        self.start_epoch = int(fork["epoch"]) + 1
        self.curve = list(fork["curve"])
        self.best = dict(fork["best"])
        self.best_state = fork.get("best_state")
        if self.best_state is not None:
            # the shared prefix may hold the best valid epoch of the whole curve;
            # give every arm a materialised best.pt for that case
            C.atomic_torch_save(
                {"state_dict": self.best_state, "epoch": int(self.best["epoch"]),
                 "valid_mae": float(self.best["valid_mae"]), "arm": self.arm,
                 "seed": self.seed, "step": -1, "from_prefix": True},
                self.out / "best.pt")
        if fork["init_state_sha256"] != self.init_state_sha256:
            raise RuntimeError("fork init hash does not match a fresh build of this seed")
        return fork

    # -- valid / train evaluation -----------------------------------------
    @torch.no_grad()
    def valid_mae(self) -> float:
        _, mae = T1.evaluate(self.model, self.valid_cache, self.rows_valid, candidate="P3N",
                             device=self.device, batch_size=int(RECIPE["valid_batch_size"]))
        return float(mae)

    @torch.no_grad()
    def train_mae_eval_mode(self) -> float:
        _, mae = T1.evaluate(self.model, self.train_cache, self.rows_train, candidate="P3N",
                             device=self.device, batch_size=256)
        return float(mae)

    # -- one ERM step ------------------------------------------------------
    def erm_step(self, batch, target, rate: float):
        pred = self.model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        if not math.isfinite(float(gn)):
            raise RuntimeError("non-finite ERM gradient")
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()
        mae = float((pred.detach().view(-1) - target.view(-1)).abs().mean())
        return float(loss.detach()), mae

    # -- one meta step -----------------------------------------------------
    def meta_step(self, batch_s, target_s, batch_q, target_q, rate: float, epoch: int):
        params = dict(self.model.named_parameters())
        alpha = self.kappa * rate * ramp(epoch)
        grads, j, loss_s, loss_q, gnorm, c = meta_grads(
            self.model, params, batch_s, target_s, batch_q, target_q, alpha)
        for p, gv in zip(params.values(), grads):
            p.grad = gv
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        if not math.isfinite(float(gn)):
            raise RuntimeError("non-finite meta gradient")
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.step()
        return {
            "j": float(j.detach()),
            "source_mae_theta": float(loss_s.detach()),
            "query_mae_virtual": float(loss_q.detach()),
            "g_norm": float(gnorm.detach()),
            "clip_c": float(c.detach()),
            "alpha": float(alpha),
        }

    # -- diagnostics -------------------------------------------------------
    @torch.no_grad()
    def _probe_batches(self):
        return None

    def run_probes(self, epoch: int) -> None:
        """Fixed train-only probe episodes: virtual-step effect and gradient geometry."""
        eps_rows = self.diag.get("probe_rows")
        if eps_rows is None:
            return
        eps_rows = np.asarray(eps_rows, dtype=np.int64)
        q_masks = self.diag["probe_q"][self.arm]
        total = 0
        src0, qry0, dpred, pstep = [], [], [], []
        cos_j, rel_j, cos_meta = [], [], []
        was_training = self.model.training
        self.model.train()
        eps = 1e-6
        for k in range(len(eps_rows)):
            rows = eps_rows[k]
            qsel = q_masks[k]
            s_rows = rows[~qsel]
            q_rows = rows[qsel]
            bs = T1._to_device(T1.collate(self.train_cache, s_rows, "P3N"), self.device)
            bq = T1._to_device(T1.collate(self.train_cache, q_rows, "P3N"), self.device)
            ts = torch.as_tensor(self.y_train[s_rows], dtype=torch.float32, device=self.device)
            tq = torch.as_tensor(self.y_train[q_rows], dtype=torch.float32, device=self.device)
            params = dict(self.model.named_parameters())
            names = list(params)
            values = [params[n] for n in names]
            pred_s = torch.func.functional_call(self.model, params, (bs,))
            loss_s = F.l1_loss(pred_s.view(-1), ts.view(-1))
            g = _grads_of(loss_s, values)
            gnorm = torch.sqrt(sum((gi * gi).sum() for gi in g))
            c = torch.clamp(CLIP / (gnorm + EPS), max=1.0)
            rate = lr_at(self.global_step, self.total_updates)
            alpha = self.kappa * rate * ramp(epoch)
            shifted = {n: p - alpha * c * gi for n, p, gi in zip(names, values, g)}
            pred_q = torch.func.functional_call(self.model, shifted, (bq,))
            pred_q0 = torch.func.functional_call(self.model, params, (bq,))
            loss_q = F.l1_loss(pred_q.view(-1), tq.view(-1))
            loss_q0 = F.l1_loss(pred_q0.view(-1), tq.view(-1))
            n_s, n_q = int(bs["n_graphs"]), int(bq["n_graphs"])
            w_s, w_q = n_s / (n_s + n_q), n_q / (n_s + n_q)
            j = w_s * loss_s + w_q * loss_q
            gj = torch.autograd.grad(j, values, allow_unused=False)
            # ERM gradient on the same union (both halves at theta, clean path)
            union = np.concatenate([s_rows, q_rows])
            bu = T1._to_device(T1.collate(self.train_cache, union, "P3N"), self.device)
            tu = torch.as_tensor(self.y_train[union], dtype=torch.float32, device=self.device)
            pred_u = torch.func.functional_call(self.model, params, (bu,))
            loss_u = F.l1_loss(pred_u.view(-1), tu.view(-1))
            gu = torch.autograd.grad(loss_u, values, allow_unused=False)
            flat_j = torch.cat([t.reshape(-1) for t in gj])
            flat_u = torch.cat([t.reshape(-1) for t in gu])
            flat_meta = torch.cat([(a - b).reshape(-1) for a, b in zip(gj, gu)])
            cos_j.append(float(F.cosine_similarity(flat_j, flat_u, dim=0)))
            rel_j.append(float((flat_j - flat_u).norm() / (flat_u.norm() + 1e-12)))
            # source-only step direction vs the full ERM step
            flat_g = torch.cat([t.reshape(-1) for t in g])
            cos_meta.append(float(F.cosine_similarity(flat_g, flat_u, dim=0)))
            src0.append(float(loss_s.detach()))
            qry0.append(float(loss_q0.detach()))
            dpred.append(float((pred_q.detach() - pred_q0.detach()).abs().mean()))
            pstep.append(float(alpha * c * gnorm))
            total += 1
            del pred_s, pred_q, pred_q0, gj, gu, g
        if was_training:
            self.model.train()
        self.diag_log.append({
            "epoch": int(epoch),
            "n_episodes": int(total),
            "source_mae_theta_mean": float(np.mean(src0)),
            "query_mae_theta_mean": float(np.mean(qry0)),
            "query_abs_dpred_median": float(np.median(dpred)),
            "query_abs_dpred_mean": float(np.mean(dpred)),
            "param_step_norm_mean": float(np.mean(pstep)),
            "cos_meta_grad_vs_erm": float(np.mean(cos_j)),
            "rel_meta_grad_diff": float(np.mean(rel_j)),
            "cos_source_grad_vs_erm": float(np.mean(cos_meta)),
        })
        C.write_json(self.out / "probe_epochs.json", self.diag_log)

    # -- main loop ---------------------------------------------------------
    def run(self, fork_in: Path | None = None, fork_out_epoch: int | None = None,
            epochs: int = EPOCHS, diag_epochs: tuple[int, ...] = ()) -> dict:
        if fork_in is not None:
            self.load_fork(Path(fork_in))
        union_hashes = []
        n_batches = math.ceil(self.n_train / BATCH_SIZE)
        fresh = self.start_epoch == 1
        started = time.perf_counter()
        gradient_audit = {}
        for epoch in range(self.start_epoch, epochs + 1):
            self.model.train()
            order = torch.randperm(self.n_train, generator=self.gen).numpy()
            union_hashes.append(SP.order_hash(order))
            if self.partition is not None:
                exp = self.partition["epoch_orders"]
                if exp[int(epoch) - 1] != union_hashes[-1]:
                    raise RuntimeError(f"epoch {epoch}: batch order differs from the frozen partition file")
            total_err, n_mol = 0.0, 0
            ep = {"source": 0.0, "query_virtual": 0.0, "j": 0.0, "alpha": 0.0, "clip_c": 0.0}
            t_ep = time.perf_counter()
            for b, start in enumerate(range(0, self.n_train, BATCH_SIZE)):
                rows = order[start:start + BATCH_SIZE]
                target = torch.as_tensor(self.y_train[rows], dtype=torch.float32, device=self.device)
                rate_step = self.global_step + 1
                rate = lr_at(rate_step, self.total_updates)
                if self.arm == "E":
                    batch = T1._to_device(T1.collate(self.train_cache, rows, "P3N"), self.device)
                    loss, mae = self.erm_step(batch, target, rate)
                    if fresh and epoch == 1 and start == 0:
                        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                        gradient_audit = {
                            "grad_tensors": len(grads),
                            "nonzero": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                            "all_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                            "loss_finite": bool(torch.isfinite(torch.tensor(loss)).item()),
                        }
                    total_err += mae * len(rows)
                    n_mol += len(rows)
                else:
                    qsel = self._partition_q(epoch, b, rows)
                    s_rows = rows[~qsel]
                    q_rows = rows[qsel]
                    bs = T1._to_device(T1.collate(self.train_cache, s_rows, "P3N"), self.device)
                    bq = T1._to_device(T1.collate(self.train_cache, q_rows, "P3N"), self.device)
                    ts = torch.as_tensor(self.y_train[s_rows], dtype=torch.float32, device=self.device)
                    tq = torch.as_tensor(self.y_train[q_rows], dtype=torch.float32, device=self.device)
                    info = self.meta_step(bs, ts, bq, tq, rate, epoch)
                    for key, dkey in (("source", "source_mae_theta"), ("query_virtual", "query_mae_virtual"),
                                      ("j", "j"), ("alpha", "alpha"), ("clip_c", "clip_c")):
                        ep[key] += info[dkey] * len(rows) / self.n_train
                    total_err += info["source_mae_theta"] * len(s_rows)
                    total_err += info["query_mae_virtual"] * len(q_rows)
                    n_mol += len(rows)
                self.global_step += 1
            online = total_err / max(n_mol, 1)
            valid = self.valid_mae()
            sec = time.perf_counter() - t_ep
            entry = {
                "epoch": int(epoch),
                "online_train_mae": float(online),
                "valid_mae": float(valid),
                "lr": float(rate),
                "seconds": float(sec),
                "steps": int(n_batches),
                "alpha": float(ep["alpha"]),
                "batch_union_sha256": union_hashes[-1],
            }
            if self.arm != "E":
                entry.update({"source_mae": float(ep["source"]),
                              "query_mae_virtual": float(ep["query_virtual"]),
                              "j_value": float(ep["j"]),
                              "clip_c": float(ep["clip_c"])})
            if self.eval_train_from is not None and epoch >= self.eval_train_from:
                entry["train_mae_eval_mode"] = float(self.train_mae_eval_mode())
            self.curve.append(entry)
            if valid < self.best["valid_mae"]:
                self.save_best(valid, epoch)
            if fork_out_epoch is not None and epoch == fork_out_epoch:
                self.save_fork(epoch)
            if epoch in diag_epochs and self.diag.get("probe_rows") is not None:
                self.run_probes(epoch)
            C.log(f"[{self.arm} s{self.seed}] ep={epoch:03d} valid={valid:.6f} online={online:.6f} "
                  f"lr={rate:.3e} best={self.best['valid_mae']:.6f}@{self.best['epoch']} {sec:.2f}s")
        wall = time.perf_counter() - started
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(self.model), "optimizer": self.optimizer.state_dict(),
             "gen_state": self.gen.get_state(), "epoch": epochs, "global_step": self.global_step,
             "best": self.best, "curve": self.curve, "arm": self.arm, "seed": self.seed},
            self.out / "final.pt")
        best_state = torch.load(self.out / "best.pt", map_location="cpu", weights_only=False)
        best_model, _ = build_model(self.norm, self.seed, self.device)
        C.load_state(best_model, best_state["state_dict"], self.device)
        best_pred_valid, best_valid = T1.evaluate(
            best_model, self.valid_cache, self.rows_valid, candidate="P3N", device=self.device,
            batch_size=int(RECIPE["valid_batch_size"]))
        _, best_train = T1.evaluate(best_model, self.train_cache, self.rows_train, candidate="P3N",
                                    device=self.device, batch_size=256)
        final_valid = self.valid_mae()
        final_train = self.train_mae_eval_mode()
        (self.out / "predictions").mkdir(exist_ok=True)
        np.savez(self.out / "predictions" / "valid_best.npz", row=self.rows_valid, pred=best_pred_valid)
        info = {
            "round": "zinc-composition-transfer-v1",
            "arm": self.arm,
            "seed": int(self.seed),
            "kappa": float(self.kappa),
            "git_commit": C.git_commit(),
            "environment": C.environment(),
            "init_state_sha256": self.init_state_sha256,
            "model_info": self.model_info,
            "parameter_count": int(self.model_info["parameters"]),
            "data": {
                "n_train": int(self.n_train),
                "n_valid": int(self.n_valid()),
                "official_test_loaded": False,
                "train_label_sha256": C.sha256_array(self.y_train),
                "valid_label_sha256": C.sha256_array(self.y_valid),
            },
            "recipe": {**RECIPE, "epochs": int(epochs), "steps_per_epoch": int(n_batches),
                       "updates": int(self.total_updates), "warmup_steps": int(WARMUP),
                       "actual_global_steps": int(self.global_step), "loss": "raw-y L1",
                       "optimizer_impl": "torch.optim.AdamW", "betas": list(BETAS), "eps": EPS,
                       "schedule": "ad_common.cosine_with_warmup_lr(step, total, 500, 1e-3, 1e-5)"},
            "meta_objective": None if self.arm == "E" else {
                "kappa": float(self.kappa),
                "ramp": "0 for ep<=50, linear 51..100, 1 afterwards",
                "alpha": "kappa * lr_t * ramp_t",
                "inner": "mean raw-y L1 over source half, one virtual SGD step, clip 5",
                "outer": "|A|/|B| L_A(theta) + |Q|/|B| L_Q(theta')",
                "gradient": "exact double backward (torch.autograd.grad create_graph=True)",
            },
            "gradient_audit": gradient_audit,
            "best_valid": self.best,
            "fixed_objects": {
                "best": {"valid_mae": float(best_valid), "train_mae_eval_mode": float(best_train),
                         "epoch": int(self.best["epoch"])},
                "final": {"valid_mae": float(final_valid), "train_mae_eval_mode": float(final_train),
                          "epoch": int(epochs)},
            },
            "curve": self.curve,
            "probe_epochs": self.diag_log,
            "wall_seconds": float(wall),
            "gpu_hours": None if str(self.device).startswith("cpu") else float(wall) / 3600.0,
            "device": str(self.device),
            "notes": [
                "online_train_mae is a within-epoch pre-update running mean (for meta arms it mixes "
                "source L1 at theta with query L1 at theta')",
                "train_mae_eval_mode is a fixed-object full-train measurement",
                "selection is argmin(valid_mae); ties keep the earlier epoch",
            ],
        }
        C.write_json(self.out / "train.json", info)
        C.write_curve_csv(self.out / "curve.csv", self.curve)
        C.log(f"[{self.arm} s{self.seed}] done best={best_valid:.6f}@{self.best['epoch']} "
              f"final={final_valid:.6f} wall={wall:.1f}s")
        return info

    def n_valid(self) -> int:
        return len(self.y_valid)

    def _partition_q(self, epoch: int, batch: int, rows: np.ndarray) -> np.ndarray:
        part = self.partition
        if part is None:
            raise RuntimeError("meta arm needs the frozen partition file")
        return SP.mask_for(part, epoch, batch, self.arm, len(rows), self.steps_per_epoch)


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}{arm}_s{int(seed)}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--fork-in", default=None)
    p.add_argument("--fork-out-epoch", type=int, default=None)
    p.add_argument("--eval-train-from", type=int, default=None)
    p.add_argument("--diag-epochs", default="")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else run_dir(args.arm, args.seed, args.tag)
    if (out / "train.json").exists() and not args.force:
        C.log(f"{out} already complete; use --force")
        return 0
    if args.arm in ("R", "S") and args.kappa <= 0:
        raise SystemExit("meta arms need --kappa > 0")
    partition = None
    if args.arm in ("R", "S"):
        part = SP.load_partitions(args.seed)
        partition = {
            "r_packed": part["r_packed"], "s_packed": part["s_packed"],
            "batch_rows": part["batch_rows"], "swap_off": part["swap_off"],
            "swap_i": part["swap_i"], "swap_j": part["swap_j"],
            "epoch_orders": part["summary"]["epoch_order_sha256"],
        }
    diag = {}
    diag_epochs = tuple(int(x) for x in args.diag_epochs.split(",") if x)
    if diag_epochs:
        probe = np.load(CACHE / "probe_episodes.npz")
        diag = {
            "probe_rows": np.asarray(probe["probe_rows"], dtype=np.int64),
            "probe_q": {"R": np.asarray(probe["probe_q"], dtype=bool),
                        "S": np.asarray(probe["probe_q_structural"], dtype=bool)},
        }
    tr = Trainer(args.arm, args.seed, torch.device(args.device), out, kappa=args.kappa,
                 partition=partition, eval_train_from=args.eval_train_from, diag=diag)
    tr.run(fork_in=Path(args.fork_in) if args.fork_in else None,
           fork_out_epoch=args.fork_out_epoch, epochs=args.epochs, diag_epochs=diag_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
