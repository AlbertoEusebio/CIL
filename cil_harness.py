#!/usr/bin/env python3
"""
One harness, one protocol, every method.

    python cil_harness.py --method fecam --seeds 0,1,2 --epochs 300 \
        --data-dir /kaggle/input/cifar100 --out /kaggle/working/runs \
        --ckpt-dir /kaggle/working/ckpt --deadline-sec 39600

Protocol (locked, see cil_data.py): CIFAR-100, cold start, 10 tasks of 10
classes, ResNet-18 CIFAR variant from random init, no exemplars, class order
from --split-seed (1234), 500 validation images per task held out of TRAIN.
Every method reads data through cil_data.prepare_data and is scored by
score_matrix() below. Nothing is selected on the test set.

Methods
    finetune   one network, CE over all classes seen so far, no protection.
               The lower bound. Sanity check for the harness.
    fecam      Goswami et al. NeurIPS 2023. Backbone trained on task 0 only,
               then frozen. Per-class Mahalanobis with Tukey^0.5, shrinkage
               and correlation normalisation. Published 32.4 last in this
               protocol. If we cannot get within a couple of points the
               harness is wrong.
    wsn        Kang et al. ICML 2022. Per-weight learned scores, top-c% per
               layer, straight-through, gradients of reused weights masked.
               No native class-IL inference; reported with our router and
               with concatenated head logits, both labelled.
    supsup     Wortsman et al. NeurIPS 2020. Random frozen weights, per-task
               learned scores (edge-popup). Class-IL by argmax over the
               concatenated per-task logits (Kim et al. 2022 measured this
               above SupSup's own one-shot entropy rule); the one-shot rule
               is reported as a diagnostic on a subsample, and our router
               is reported too.
    ours       exp18_masked_circuits: greedy causal ablation per task, masked
               freezing, z-standardised Mahalanobis router. --select swaps
               the selection rule at matched sparsity (causal | magnitude |
               learned | random). The causal sweep is always run, because
               the alternatives copy its per-stage sizes.

Checkpoints: after every task, {ckpt_dir}/{tag}_seed{s}.pt. On start the
harness looks in --ckpt-dir and then in every --resume-dir for that file and
continues from the next task. --deadline-sec stops cleanly between tasks.
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cil_data as D
import exp18_masked_circuits as X


# =============================================================================
# config
# =============================================================================

@dataclass
class HCfg:
    method: str = "fecam"
    split_seed: int = 1234        # class order and validation split
    n_tasks: int = 10
    cpt: int = 10
    epochs: int = 300
    batch_size: int = 256
    val_per_task: int = 500
    aug: int = 1
    stages: str = "64,128,256,512"
    train_sub: int = 0            # >0: use only this many train images per
                                  # task. Toy runs only. Always logged.
    # standard trainer (finetune, fecam task 0, wsn, supsup)
    opt: str = "sgd"              # sgd | adam
    lr: float = 0.1
    weight_decay: float = 5e-4
    sched: str = "cos"            # cos | none
    # fecam
    fecam_gamma1: float = 1.0
    fecam_gamma2: float = 1.0
    fecam_tukey: float = 0.5
    # wsn / supsup
    wsn_c: float = 0.5            # fraction of weights each task may use
    supsup_k: float = 0.1         # fraction of weights each task keeps
    supsup_native_n: int = 500    # SupSup's one-shot entropy-gradient task
                                  # inference needs one backward pass per
                                  # image, so it is run on this many test
                                  # images as a diagnostic row (`native`).
                                  # SupSup's class-IL number uses argmax
                                  # over concatenated logits (`no_router`),
                                  # which Kim et al. 2022 footnote 6 measured
                                  # as the better rule (62.6 vs 50.2).
    # ours
    select: str = "causal"        # causal | magnitude | learned | random
    rot_extra: int = 0            # H_rot. >0: CLOM/CSI rotation-as-OOD. Each
                                  # step adds this many rotated copies (90,
                                  # 180, 270 deg) of every image and trains a
                                  # joint (class x rotation) 4*cpt-way head
                                  # per task next to the class head. Routing
                                  # row `rot` = mean over rotations of the
                                  # max sigmoid on the matching rotation rows.
                                  # Costs (1 + rot_extra)x training compute.
                                  # 3 = the full CSI recipe.
    learned_steps: int = 300
    learned_lam: float = 1e-3
    prune_tol: float = 0.02
    prune_floor: float = 0.30
    score_order: str = "taylor"
    head_refit_steps: int = 400
    rand_draws: int = 3
    per_task_bn: int = 0
    tf32: int = 1
    eval_mixed: int = 1           # 1: evaluate on batches mixing every seen
                                  # task's test set (BN without running stats
                                  # otherwise sees same-task statistics, a
                                  # transductive leak of the task id).
                                  # 0: exp18's original per-task batching.
    ablate_alternatives: int = 1  # measure magnitude/learned/random masks at
                                  # matched sparsity on every task, whatever
                                  # --select is. Costs one head refit each.

    def stages_tuple(self):
        return tuple(int(v) for v in self.stages.split(",") if v.strip())


def n_classes_of(cfg):
    return D.class_split(cfg.n_tasks, cfg.cpt)[-1][1]


# =============================================================================
# shared pieces
# =============================================================================

def std_bn(model):
    """Replace exp18's affine-free, stats-free BN with standard BN for the
    methods that do not need mask-friendly BN (finetune, fecam)."""
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(mod, cname, nn.BatchNorm2d(child.num_features))
    return model.to(next(model.parameters()).device)


def make_opt(params, cfg, steps):
    if cfg.opt == "adam":
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9,
                              weight_decay=cfg.weight_decay)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
           if cfg.sched == "cos" else None)
    return opt, sch


def train_ce(model, x, y, lo, hi, cfg, seed, forward, params, grad_hook=None,
             log_every=0):
    """Generic CE trainer over logits[:, lo:hi]. `forward(xb)` returns full
    logits. `grad_hook()` runs after backward, before step. No per-step host
    sync: the loss is kept on device."""
    g = torch.Generator().manual_seed(seed)
    nb = (len(x) + cfg.batch_size - 1) // cfg.batch_size
    opt, sch = make_opt(params, cfg, cfg.epochs * nb)
    lbuf = torch.zeros(cfg.epochs * nb, device=x.device)
    k = 0
    model.train()
    for ep in range(cfg.epochs):
        order = torch.randperm(len(x), generator=g).to(x.device)
        for s in range(0, len(x), cfg.batch_size):
            i = order[s:s + cfg.batch_size]
            xb = D.augment(x[i], g) if cfg.aug else x[i]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(forward(xb)[:, lo:hi], y[i] - lo)
            loss.backward()
            if grad_hook is not None:
                grad_hook()
            opt.step()
            if sch is not None:
                sch.step()
            lbuf[k] = loss.detach()
            k += 1
        if log_every and (ep + 1) % log_every == 0:
            print(f"        epoch {ep+1}/{cfg.epochs} loss "
                  f"{float(lbuf[k-nb:k].mean()):.4f}", flush=True)
    return float(lbuf[max(0, k - 50):k].mean())


@torch.no_grad()
def batched(fn, x, batch=512):
    return torch.cat([fn(x[s:s + batch]) for s in range(0, len(x), batch)])


def mixed_test(tasks, t, seed=0):
    """All seen tasks' test images in one fixed shuffled order, with the
    per-image true task id."""
    xs, ys, ts = [], [], []
    for s in range(t + 1):
        x, y = tasks[s]["test"]
        xs.append(x); ys.append(y)
        ts.append(torch.full((len(x),), s, device=x.device, dtype=torch.long))
    x, y, tt = torch.cat(xs), torch.cat(ys), torch.cat(ts)
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(x), generator=g).to(x.device)
    return x[p], y[p], tt[p]


def per_task_means(L, tt, y, T):
    """Break a ladder row's pick/cls tensors into per-task accuracies."""
    rows = {}
    for r, v in L.items():
        rows[r] = [{"task_acc": float((v["pick"][tt == s] == s).float().mean()),
                    "class_acc": float((v["cls"][tt == s] == y[tt == s]).float().mean())}
                   for s in range(T)]
    return rows


def score_matrix(cil: np.ndarray, til: np.ndarray, T: int) -> Dict:
    """cil[t, s] = class-IL accuracy on task s's test set after learning
    task t (prediction over ALL classes seen up to t). til likewise with the
    task id given. Equal task sizes, so a mean over tasks is the accuracy
    over all seen classes."""
    last = float(np.nanmean(cil[T - 1, :T]))
    avg = float(np.mean([np.nanmean(cil[t, :t + 1]) for t in range(T)]))
    til_last = float(np.nanmean(til[T - 1, :T]))
    # forgetting decomposed: weight movement shows in task-IL, label space
    # growth shows in class-IL with task-IL held fixed
    til_forget = float(np.nanmean([til[s, s] - til[T - 1, s]
                                   for s in range(T - 1)])) if T > 1 else 0.0
    cil_forget = float(np.nanmean([cil[s, s] - cil[T - 1, s]
                                   for s in range(T - 1)])) if T > 1 else 0.0
    return {"classIL_last": last, "classIL_avg": avg, "taskIL_last": til_last,
            "F_taskIL(weight movement)": til_forget,
            "F_classIL(movement + label growth)": cil_forget,
            "cil_matrix": cil.tolist(), "til_matrix": til.tolist()}


# =============================================================================
# routing shared by wsn / supsup: same estimator exp18 uses
# =============================================================================

@torch.no_grad()
def fit_class_gauss(f, y, lo, hi, tukey=0.5, ridge=1e-2):
    if tukey:
        f = f.clamp_min(0).pow(tukey)
    k = f.shape[1]
    out = {}
    for c in range(lo, hi):
        fc = f[y == c]
        mu = fc.mean(0)
        d = fc - mu
        cov = (d.t() @ d) / max(1, len(fc) - 1)
        cov = cov + (ridge * torch.diag(cov).mean() + 1e-6) * torch.eye(k, device=f.device)
        prec = torch.linalg.inv(cov.double()).float()
        dist = ((d @ prec) * d).sum(1).clamp_min(0).sqrt()
        out[c] = {"mu": mu, "prec": prec, "m": float(dist.mean()),
                  "s": float(dist.std()) + 1e-6}
    return {"classes": out, "tukey": tukey}


@torch.no_grad()
def route_generic(feat_fns, head_fns, stats, spans, x, y, true_task):
    """Rows: z (ours), mls (max head logit), no_router (concat head logits),
    oracle. feat_fns[t](x) -> features under task t's mask; head_fns[t](x)
    -> logits for task t's own classes."""
    T = len(spans)
    z = torch.full((len(x), T), float("inf"), device=x.device)
    own, head = [], []
    for t in range(T):
        f = batched(feat_fns[t], x)
        st = stats[t]
        if st["tukey"]:
            f = f.clamp_min(0).pow(st["tukey"])
        lo, hi = spans[t]
        cols, best = [], None
        for c in range(lo, hi):
            s = st["classes"][c]
            d = f - s["mu"]
            dist = ((d @ s["prec"]) * d).sum(1).clamp_min(0).sqrt()
            zz = (dist - s["m"]) / s["s"]
            cols.append(-zz)
            best = zz if best is None else torch.minimum(best, zz)
        z[:, t] = best
        own.append(torch.stack(cols, 1))
        head.append(batched(head_fns[t], x))
    out = {}

    def cls_from(pick, src):
        c = torch.zeros(len(y), dtype=torch.long, device=x.device)
        for u in range(T):
            m = pick == u
            if bool(m.any()):
                c[m] = src[u][m].argmax(1) + spans[u][0]
        return c

    def rec(name, pick, cls):
        out[name] = {"task_acc": float((pick == true_task).float().mean()),
                     "class_acc": float((cls == y).float().mean()),
                     "pick": pick, "cls": cls}
    pz = z.argmin(1)
    rec("z", pz, cls_from(pz, own))
    mls = torch.stack([h.max(1).values for h in head], 1)
    pm = mls.argmax(1)
    rec("mls", pm, cls_from(pm, head))
    cat = torch.cat(head, 1)
    offs = torch.cat([torch.arange(lo, hi, device=x.device) for lo, hi in spans])
    owner = torch.cat([torch.full((hi - lo,), u, device=x.device, dtype=torch.long)
                       for u, (lo, hi) in enumerate(spans)])
    j = cat.argmax(1)
    rec("no_router", owner[j], offs[j])
    po = (true_task.clone() if torch.is_tensor(true_task) else
          torch.full((len(y),), true_task, device=x.device, dtype=torch.long))
    rec("oracle", po, cls_from(po, head))
    return out


# =============================================================================
# method: finetune
# =============================================================================

class Finetune:
    tag = "finetune"

    def __init__(self, cfg, tasks, device, seed):
        self.cfg, self.tasks, self.dev, self.seed = cfg, tasks, device, seed
        self.model = std_bn(X.MaskedResNet18(n_classes_of(cfg), seed, device,
                                             stages=cfg.stages_tuple()))

    def learn(self, t):
        x, y = self.tasks[t]["train"]
        lo, hi = self.tasks[t]["classes"]
        loss = train_ce(self.model, x, y, 0, hi, self.cfg, self.seed * 100 + t,
                        self.model, list(self.model.parameters()))
        return {"loss": loss}

    @torch.no_grad()
    def evaluate(self, t):
        self.model.eval()
        hi_all = self.tasks[t]["classes"][1]
        til, cil = [], []
        for s in range(t + 1):
            x, y = self.tasks[s]["test"]
            lo, hi = self.tasks[s]["classes"]
            lg = batched(self.model, x)
            til.append(float((lg[:, lo:hi].argmax(1) + lo == y).float().mean()))
            cil.append(float((lg[:, :hi_all].argmax(1) == y).float().mean()))
        return {"til": til, "cil": cil}

    def state(self):
        return {"model": self.model.state_dict()}

    def load(self, st):
        self.model.load_state_dict(st["model"])


# =============================================================================
# method: FeCAM
# =============================================================================

class FeCAM:
    """Backbone from task 0 only. Per class: mean of Tukey-transformed
    features and a shrunk, correlation-normalised covariance.
    Classification: argmin Mahalanobis over all seen classes. No router."""
    tag = "fecam"

    def __init__(self, cfg, tasks, device, seed):
        self.cfg, self.tasks, self.dev, self.seed = cfg, tasks, device, seed
        self.model = std_bn(X.MaskedResNet18(n_classes_of(cfg), seed, device,
                                             stages=cfg.stages_tuple()))
        self.mu, self.prec = {}, {}

    def feats(self, x):
        return batched(lambda b: self.model.features(b), x)

    def _tukey(self, f):
        lam = self.cfg.fecam_tukey
        return f.clamp_min(0).pow(lam) if lam else f

    @torch.no_grad()
    def _fit(self, t):
        self.model.eval()          # BN running stats; also after a resume
        x, y = self.tasks[t]["train"]
        lo, hi = self.tasks[t]["classes"]
        f = self._tukey(self.feats(x))
        k = f.shape[1]
        eye = torch.eye(k, device=f.device)
        for c in range(lo, hi):
            fc = f[y == c]
            mu = fc.mean(0)
            d = fc - mu
            cov = (d.t() @ d) / max(1, len(fc) - 1)
            v1 = torch.diag(cov).mean()
            v2 = (cov.sum() - torch.diag(cov).sum()) / (k * k - k)
            cov = cov + self.cfg.fecam_gamma1 * v1 * eye \
                      + self.cfg.fecam_gamma2 * v2 * (1 - eye)
            sd = torch.diag(cov).sqrt().clamp_min(1e-6)
            cov = cov / (sd[:, None] * sd[None, :]) + 1e-6 * eye
            self.mu[c] = mu
            self.prec[c] = torch.linalg.inv(cov.double()).float()

    def learn(self, t):
        info = {}
        if t == 0:
            x, y = self.tasks[0]["train"]
            lo, hi = self.tasks[0]["classes"]
            info["loss"] = train_ce(self.model, x, y, lo, hi, self.cfg,
                                    self.seed * 100, self.model,
                                    list(self.model.parameters()), log_every=50)
            self.model.eval()
        self._fit(t)
        return info

    @torch.no_grad()
    def predict(self, x, classes):
        f = self._tukey(self.feats(x))
        ds = []
        for c in classes:
            d = f - self.mu[c]
            ds.append(((d @ self.prec[c]) * d).sum(1))
        return torch.tensor(classes, device=x.device)[torch.stack(ds, 1).argmin(1)]

    @torch.no_grad()
    def evaluate(self, t):
        self.model.eval()
        hi_all = self.tasks[t]["classes"][1]
        til, cil = [], []
        for s in range(t + 1):
            x, y = self.tasks[s]["test"]
            lo, hi = self.tasks[s]["classes"]
            til.append(float((self.predict(x, list(range(lo, hi))) == y).float().mean()))
            cil.append(float((self.predict(x, list(range(hi_all))) == y).float().mean()))
        return {"til": til, "cil": cil}

    def state(self):
        return {"model": self.model.state_dict(), "mu": self.mu, "prec": self.prec}

    def load(self, st):
        self.model.load_state_dict(st["model"])
        self.mu, self.prec = st["mu"], st["prec"]


# =============================================================================
# weight-mask networks (WSN, SupSup)
# =============================================================================

class GetSubnet(torch.autograd.Function):
    """Top-k% of scores -> binary mask, straight-through gradient."""
    @staticmethod
    def forward(ctx, scores, k):
        out = torch.zeros_like(scores)
        flat = scores.flatten()
        n = max(1, int(round(k * flat.numel())))
        idx = torch.topk(flat, n).indices
        out.view(-1)[idx] = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class WeightMasked(nn.Module):
    """Holds a weight tensor and per-task masks. `score` is the current task's
    learnable score. Semantics selected by `mode`:
        wsn     effective weight = w * mask(score);  w trainable except where
                any earlier task's mask is 1 (grad zeroed)
        supsup  effective weight = w_random * mask(score); w never trains
    """
    def __init__(self, weight: nn.Parameter, k: float):
        super().__init__()
        self.weight = weight
        self.k = k
        self.score = nn.Parameter(torch.empty_like(weight))
        nn.init.kaiming_uniform_(self.score, a=math.sqrt(5))
        self.register_buffer("used", torch.zeros_like(weight, dtype=torch.bool))
        self.masks: List[torch.Tensor] = []
        self.active: Optional[int] = None   # None -> training with score
        self.alpha: Optional[torch.Tensor] = None   # supsup superposition

    def reset_score(self, seed):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.weight[0].numel())
            self.score.copy_(torch.empty_like(self.score).uniform_(-bound, bound, generator=g)
                             if self.score.is_cpu else
                             (torch.rand(self.score.shape, generator=g) * 2 - 1).mul(bound).to(self.score.device))

    def effective(self):
        if self.alpha is not None:
            m = sum(a * mk for a, mk in zip(self.alpha, self.masks))
        elif self.active is None:
            m = GetSubnet.apply(self.score.abs(), self.k)
        else:
            m = self.masks[self.active]
        return self.weight * m

    @torch.no_grad()
    def commit(self):
        m = GetSubnet.apply(self.score.abs(), self.k).bool()
        self.masks.append(m.clone())
        self.used |= m


class MaskNet(nn.Module):
    """CIFAR ResNet-18 with every conv/linear weight wrapped in WeightMasked.
    BN is affine-free with no running stats, as in WSN's and SupSup's code."""
    def __init__(self, n_classes, seed, device, stages, k):
        super().__init__()
        torch.manual_seed(seed)
        base = X.MaskedResNet18(n_classes, seed, device, stages=stages)
        self.base = base
        self.wm = nn.ModuleDict()
        for name, mod in base.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                self.wm[name.replace(".", "_")] = WeightMasked(mod.weight, k)
        self.to(device)

    def _apply_masks(self):
        # swap effective weights in via functional forward
        self._eff = {}
        for name, mod in self.base.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                self._eff[name] = self.wm[name.replace(".", "_")].effective()

    def set_task(self, t):
        for w in self.wm.values():
            w.active, w.alpha = t, None

    def set_alpha(self, alpha):
        for w in self.wm.values():
            w.alpha = alpha

    def features(self, x):
        b = self.base
        e = self._eff_weights()
        h = F.relu(b.bn1(F.conv2d(x, e["conv1"], padding=1)))
        for si, blks in enumerate(b.stages):
            for bi, blk in enumerate(blks):
                p = f"stages.{si}.{bi}"
                hh = F.relu(blk.bn1(F.conv2d(h, e[p + ".conv1"], stride=blk.conv1.stride, padding=1)))
                hh = blk.bn2(F.conv2d(hh, e[p + ".conv2"], padding=1))
                if blk.down is not None:
                    s = blk.down[1](F.conv2d(h, e[p + ".down.0"], stride=blk.down[0].stride))
                else:
                    s = h
                h = F.relu(hh + s)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def _eff_weights(self):
        return {name: self.wm[name.replace(".", "_")].effective()
                for name, mod in self.base.named_modules()
                if isinstance(mod, (nn.Conv2d, nn.Linear))}

    def forward(self, x):
        f = self.features(x)
        return F.linear(f, self.wm["head"].effective())

    def commit(self):
        for w in self.wm.values():
            w.commit()

    def weights(self):
        return [w.weight for w in self.wm.values()]

    def scores(self):
        return [w.score for w in self.wm.values()]

    @torch.no_grad()
    def mask_reused_grads(self):
        for w in self.wm.values():
            if w.weight.grad is not None:
                w.weight.grad.masked_fill_(w.used, 0.0)

    def used_frac(self):
        n = sum(w.used.numel() for w in self.wm.values())
        return sum(int(w.used.sum()) for w in self.wm.values()) / n


class WSN:
    tag = "wsn"
    trains_weights = True

    def __init__(self, cfg, tasks, device, seed):
        self.cfg, self.tasks, self.dev, self.seed = cfg, tasks, device, seed
        k = cfg.wsn_c if self.trains_weights else cfg.supsup_k
        self.net = MaskNet(n_classes_of(cfg), seed, device, cfg.stages_tuple(), k)
        self.stats = []
        self.T_done = 0

    def learn(self, t):
        x, y = self.tasks[t]["train"]
        lo, hi = self.tasks[t]["classes"]
        for i, w in enumerate(self.net.wm.values()):
            w.reset_score(self.seed * 1000 + t * 50 + i)
            w.active, w.alpha = None, None
        params = self.net.scores() + (self.net.weights() if self.trains_weights else [])
        hook = self.net.mask_reused_grads if self.trains_weights else None
        loss = train_ce(self.net, x, y, lo, hi, self.cfg, self.seed * 100 + t,
                        self.net, params, grad_hook=hook, log_every=50)
        self.net.commit()
        self.net.set_task(t)
        self.net.eval()
        with torch.no_grad():
            f = batched(self.net.features, x)
        self.stats.append(fit_class_gauss(f, y, lo, hi))
        self.T_done = t + 1
        return {"loss": loss, "used_frac": self.net.used_frac()}

    def _feat_fn(self, t):
        def fn(b):
            self.net.set_task(t)
            return self.net.features(b)
        return fn

    def _head_fn(self, t):
        lo, hi = self.tasks[t]["classes"]

        def fn(b):
            self.net.set_task(t)
            return self.net(b)[:, lo:hi]
        return fn

    def native_cil(self, x, y, tt, t):
        return None

    @torch.no_grad()
    def evaluate(self, t):
        self.net.eval()
        spans = [self.tasks[s]["classes"] for s in range(t + 1)]
        ff = [self._feat_fn(s) for s in range(t + 1)]
        hf = [self._head_fn(s) for s in range(t + 1)]
        if self.cfg.eval_mixed:
            x, y, tt = mixed_test(self.tasks, t)
            L = route_generic(ff, hf, self.stats, spans, x, y, tt)
            nat = self.native_cil(x, y, tt, t)
            rows = per_task_means(L, tt, y, t + 1)
            if nat is not None:
                n = nat["n"]
                rows["native"] = per_task_means({"native": nat}, tt[:n], y[:n], t + 1)["native"]
        else:
            rows = {}
            for s in range(t + 1):
                x, y = self.tasks[s]["test"]
                tt = torch.full((len(x),), s, device=x.device, dtype=torch.long)
                L = route_generic(ff, hf, self.stats, spans, x, y, tt)
                nat = self.native_cil(x, y, tt, t)
                if nat is not None:
                    L["native"] = nat
                for r, v in L.items():
                    rows.setdefault(r, []).append(
                        {"task_acc": v["task_acc"], "class_acc": v["class_acc"]})
        til = [rows["oracle"][s]["class_acc"] for s in range(t + 1)]
        # class-IL rule: SupSup by argmax over concatenated logits (Kim et
        # al. 2022), WSN, which has no rule of its own, by our z router.
        key = "no_router" if self.tag == "supsup" else "z"
        cil = [rows[key][s]["class_acc"] for s in range(t + 1)]
        return {"til": til, "cil": cil, "ladder": rows}

    def state(self):
        return {"net": self.net.state_dict(),
                "masks": [[m for m in w.masks] for w in self.net.wm.values()],
                "stats": self.stats, "T_done": self.T_done}

    def load(self, st):
        self.net.load_state_dict(st["net"])
        for w, ms in zip(self.net.wm.values(), st["masks"]):
            w.masks = list(ms)
        self.stats, self.T_done = st["stats"], st["T_done"]


class SupSup(WSN):
    """Random weights never train. Task inference: one-shot entropy gradient
    over the superposition of masks (Wortsman et al. section 3.3)."""
    tag = "supsup"
    trains_weights = False

    def __init__(self, cfg, tasks, device, seed):
        super().__init__(cfg, tasks, device, seed)
        # signed kaiming constant init, as in SupSup
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 99)
            for w in self.net.weights():
                fan_in = w[0].numel()
                std = math.sqrt(2.0 / fan_in)
                w.copy_((torch.randint(0, 2, w.shape, generator=g) * 2 - 1).float().to(w.device) * std)

    def native_cil(self, x, y, tt, t):
        """SupSup one-shot (Wortsman et al. Eq. 4), per image, on the first
        supsup_native_n images of the mixed stream. Their per batch version
        assumes a single task batch, which mixed evaluation does not give."""
        n = min(self.cfg.supsup_native_n, len(x))
        if n <= 0:
            return None
        x, y, tt = x[:n], y[:n], tt[:n]
        T = t + 1
        pick, cls = [], []
        for b0 in range(0, len(x), 256):
            xb = x[b0:b0 + 256]
            grs = []
            for i in range(len(xb)):
                a_i = torch.full((T,), 1.0 / T, device=x.device, requires_grad=True)
                self.net.set_alpha(a_i)
                with torch.enable_grad():
                    lg = self.net(xb[i:i + 1])[:, :self.tasks[t]["classes"][1]]
                    p = F.softmax(lg, 1)
                    H = -(p * torch.log(p + 1e-12)).sum()
                    g_i, = torch.autograd.grad(H, a_i)
                grs.append(-g_i)
            tk = torch.stack(grs).argmax(1)
            pick.append(tk)
            out = torch.zeros(len(xb), dtype=torch.long, device=x.device)
            for u in range(T):
                m = tk == u
                if bool(m.any()):
                    lo, hi = self.tasks[u]["classes"]
                    self.net.set_task(u)
                    out[m] = self.net(xb[m])[:, lo:hi].argmax(1) + lo
            cls.append(out)
        pick, cls = torch.cat(pick), torch.cat(cls)
        self.net.set_task(t)
        return {"task_acc": float((pick == tt).float().mean()),
                "class_acc": float((cls == y).float().mean()),
                "pick": pick, "cls": cls, "n": n}


# =============================================================================
# method: ours (exp18) with the selection-rule ablation
# =============================================================================

@torch.no_grad()
def topk_gates(scores, kept, device):
    out = []
    for sc, k in zip(scores, kept):
        g = torch.zeros(len(sc), device=device)
        g[torch.topk(sc, max(1, k)).indices] = 1.0
        out.append(g)
    return out


def learned_gate_scores(model, task, cfg, device, seed):
    """A learned per-channel importance: sigmoid gates trained on the task's
    own data with an L1 sparsity push, network weights frozen. HAT-style
    attention without the annealing. Ranking only; sizes come from causal."""
    x, y = task["train"]
    lo, hi = task["classes"]
    a = [torch.full((c,), 3.0, device=device, requires_grad=True) for c in model.STAGES]
    opt = torch.optim.Adam(a, lr=5e-2)
    g = torch.Generator().manual_seed(seed)
    tgt = torch.eye(hi - lo, device=device)
    model.eval()
    for _ in range(cfg.learned_steps):
        i = torch.randint(0, len(x), (min(256, len(x)),), generator=g).to(device)
        opt.zero_grad(set_to_none=True)
        gates = [torch.sigmoid(v) for v in a]
        with torch.enable_grad():
            lg = model(x[i], gates)[:, lo:hi]
            loss = F.binary_cross_entropy_with_logits(lg, tgt[y[i] - lo]) \
                + cfg.learned_lam * sum(gg.sum() for gg in gates)
            loss.backward()
        opt.step()
    return [v.detach().clone() for v in a]


class RotResNet(X.MaskedResNet18):
    """exp18's network plus a joint (class, rotation) head. `head` stays the
    100-way class head every exp18 routine reads; `rot_head` has 4 rows per
    class, row 4c + r for class c under rotation r."""
    def __init__(self, n_classes, seed, device, stages, ptbn):
        super().__init__(n_classes, seed, device, stages=stages, ptbn=ptbn)
        self.rot_head = nn.Linear(self.width, 4 * n_classes, bias=False).to(device)

    def rot_logits(self, x, gates=None):
        return self.rot_head(self.features(x, gates))


def rotate(x, k):
    """Rotate each image by its own multiple of 90 degrees (k: LongTensor)."""
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if bool(m.any()):
            out[m] = torch.rot90(x[m], r, dims=(2, 3))
    return out


def train_task_rot(model, tasks, t, xcfg, cfg, device, seed, frozen):
    """X.train_task plus the rotation head. Same freezer, same scorer, same
    optimiser; only the loss and the batch composition change."""
    lo, hi = tasks[t]["classes"]
    x, y = tasks[t]["train"]
    cpt = hi - lo
    gf = X.GradFreezer(model, frozen)
    sc = (X.ChannelScorer(model, device)
          if xcfg.score_order in ("signed", "absg", "taylor", "sq") else None)
    opt = torch.optim.Adam(model.parameters(), lr=xcfg.lr, weight_decay=xcfg.weight_decay)
    g = torch.Generator().manual_seed(seed)
    eye_c = torch.eye(cpt, device=device)
    eye_r = torch.eye(4 * cpt, device=device)
    nsteps = xcfg.epochs * ((len(x) + xcfg.batch_size - 1) // xcfg.batch_size)
    lbuf = torch.zeros(nsteps, device=device)
    k = 0
    model.train()
    for _ in range(xcfg.epochs):
        order = torch.randperm(len(x), generator=g).to(device)
        for s0 in range(0, len(x), xcfg.batch_size):
            i = order[s0:s0 + xcfg.batch_size]
            xb = D.augment(x[i], g) if xcfg.aug else x[i]
            yb = y[i] - lo
            B = len(xb)
            ks = [torch.zeros(B, dtype=torch.long)]
            for _c in range(cfg.rot_extra):
                ks.append(torch.randint(1, 4, (B,), generator=g))
            kk = torch.cat(ks).to(device)
            xx = torch.cat([xb] + [rotate(xb, kr.to(device)) for kr in ks[1:]])
            yy = yb.repeat(1 + cfg.rot_extra)
            opt.zero_grad(set_to_none=True)
            f = model.features(xx)
            lg_c = model.head(f[:B])[:, lo:hi]
            lg_r = model.rot_head(f)[:, 4 * lo:4 * hi]
            loss = (F.binary_cross_entropy_with_logits(lg_c, eye_c[yb])
                    + F.binary_cross_entropy_with_logits(lg_r, eye_r[4 * yy + kk]))
            loss.backward()
            gf.mask_grads()
            opt.step()
            lbuf[k] = loss.detach()
            k += 1
    drift = gf.verify()
    if drift != 0.0:
        raise RuntimeError(f"a frozen weight moved by {drift}")
    if sc is not None:
        sc.close()
    return {"bce_last": float(lbuf[max(0, k - 50):k].mean()), "scorer": sc}


@torch.no_grad()
def rot_head_refit(model, task, gates, steps, seed, device):
    """Refit this task's rot_head rows on masked features of the 4 rotations
    of its own training data. Mirrors X.head_refit."""
    lo, hi = task["classes"]
    cpt = hi - lo
    x, y = task["train"]
    model.eval()
    fs, ts = [], []
    for r in range(4):
        xr = torch.rot90(x, r, dims=(2, 3)) if r else x
        fs.append(torch.cat([model.features(xr[s:s + 512], gates)
                             for s in range(0, len(xr), 512)]))
        ts.append(4 * (y - lo) + r)
    f, tg = torch.cat(fs), torch.cat(ts)
    keep = gates[-1].bool()
    fk = f[:, keep]
    w = model.rot_head.weight[4 * lo:4 * hi][:, keep].detach().clone().requires_grad_(True)
    eye = torch.eye(4 * cpt, device=device)
    opt = torch.optim.Adam([w], lr=1e-2)
    g = torch.Generator().manual_seed(seed)
    with torch.enable_grad():
        for _ in range(steps):
            i = torch.randint(0, len(fk), (min(512, len(fk)),), generator=g).to(device)
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(fk[i] @ w.t(), eye[tg[i]]).backward()
            opt.step()
    row = torch.zeros(4 * cpt, model.width, device=device)
    row[:, keep] = w.detach()
    model.rot_head.weight[4 * lo:4 * hi] = row


@torch.no_grad()
def rot_scores(model, x, masks, spans, batch=512):
    """Per task: mean over the 4 rotations of the max over classes of the
    sigmoid on that rotation's rows. CSI's ensemble score, per circuit.
    Also returns the class scores summed over rotations for each task."""
    T = len(spans)
    score = torch.zeros(len(x), T, device=x.device)
    cls = []
    for t, (lo, hi) in enumerate(spans):
        cpt = hi - lo
        acc_c = torch.zeros(len(x), cpt, device=x.device)
        for r in range(4):
            xr = torch.rot90(x, r, dims=(2, 3)) if r else x
            lg = torch.cat([model.rot_logits(xr[s:s + batch], masks[t])[:, 4 * lo:4 * hi]
                            for s in range(0, len(xr), batch)])
            p = torch.sigmoid(lg.view(len(x), cpt, 4)[:, :, r])
            score[:, t] += p.max(1).values / 4
            acc_c += p
        cls.append(acc_c)
    return score, cls


class Ours:
    tag = "ours"

    def __init__(self, cfg, tasks, device, seed):
        self.cfg, self.tasks, self.dev, self.seed = cfg, tasks, device, seed
        self.xcfg = X.Cfg(n_tasks=cfg.n_tasks, cpt=cfg.cpt, epochs=cfg.epochs,
                          batch_size=cfg.batch_size, val_per_task=cfg.val_per_task,
                          prune_tol=cfg.prune_tol, prune_floor=cfg.prune_floor,
                          score_order=cfg.score_order, aug=cfg.aug,
                          head_refit_steps=cfg.head_refit_steps,
                          rand_draws=cfg.rand_draws, stages=cfg.stages,
                          per_task_bn=cfg.per_task_bn, tf32=cfg.tf32)
        if cfg.rot_extra:
            self.model = RotResNet(n_classes_of(cfg), seed, device,
                                   cfg.stages_tuple(), bool(cfg.per_task_bn))
        else:
            self.model = X.MaskedResNet18(n_classes_of(cfg), seed, device,
                                          stages=cfg.stages_tuple(),
                                          ptbn=bool(cfg.per_task_bn))
        self.frozen = {n: torch.zeros_like(p, dtype=torch.bool)
                       for n, p in self.model.named_parameters()}
        self.masks, self.stats = [], []
        self.per_task = []

    def _alt_eval(self, t, gates_alt, name):
        """Refit a copy of the head rows on the alternative mask, score on
        val and test, restore the head. Same trained weights, same size."""
        task = self.tasks[t]
        lo, hi = task["classes"]
        snap = self.model.head.weight.detach().clone()
        X.head_refit(self.model, task, self.xcfg, self.dev, gates_alt,
                     self.cfg.head_refit_steps, self.seed * 3 + t)
        v = X.acc_task(self.model, task, self.xcfg, self.dev, "val", gates_alt)
        te = X.acc_task(self.model, task, self.xcfg, self.dev, "test", gates_alt)
        with torch.no_grad():
            self.model.head.weight.copy_(snap)
        return {"name": name, "val": v, "test": te,
                "kept": [int(g.sum()) for g in gates_alt]}

    def learn(self, t):
        cfg, m, task = self.cfg, self.model, self.tasks[t]
        lo, hi = task["classes"]
        if cfg.rot_extra:
            tr = train_task_rot(m, self.tasks, t, self.xcfg, cfg, self.dev,
                                self.seed * 100 + t, self.frozen)
        else:
            tr = X.train_task(m, self.tasks, t, self.xcfg, self.dev,
                              self.seed * 100 + t, self.frozen)
        scorer = tr.get("scorer")
        gates, ps = X.prune_circuit(m, task, self.xcfg, self.dev,
                                    self.seed * 7 + t, scorer=scorer)
        kept = ps["kept_per_stage"]
        # matched-sparsity alternatives, on the SAME trained weights
        alts = {}
        if cfg.ablate_alternatives or cfg.select != "causal":
            c_outw = X.Cfg(**{**asdict(self.xcfg), "score_order": "outw"})
            mag = topk_gates(X.channel_scores(m, c_outw, self.dev, self.seed), kept, self.dev)
            lrn = topk_gates(learned_gate_scores(m, task, cfg, self.dev, self.seed * 5 + t),
                             kept, self.dev)
            rnd = X.random_gates(m, kept, self.dev, self.seed * 13 + t)
            alts = {"magnitude": mag, "learned": lrn, "random": rnd}
        chosen = gates if cfg.select == "causal" else alts[cfg.select]
        # head refit on the chosen mask (this is what the run commits to)
        if cfg.head_refit_steps:
            X.head_refit(m, task, self.xcfg, self.dev, chosen,
                         cfg.head_refit_steps, self.seed * 3 + t)
        val_c = X.acc_task(m, task, self.xcfg, self.dev, "val", chosen)
        test_c = X.acc_task(m, task, self.xcfg, self.dev, "test", chosen)
        ablation = [{"name": "causal", "val": ps["acc_val_circuit"], "test": None,
                     "kept": kept}]
        if cfg.select == "causal":
            ablation[0].update(val=val_c, test=test_c)
        else:
            ablation[0] = self._alt_eval(t, gates, "causal")
        for k_, g_ in alts.items():
            if k_ == cfg.select:
                ablation.append({"name": k_, "val": val_c, "test": test_c,
                                 "kept": [int(v.sum()) for v in g_]})
            else:
                ablation.append(self._alt_eval(t, g_, k_))
        if cfg.rot_extra:
            rot_head_refit(m, task, chosen, cfg.head_refit_steps, self.seed * 3 + t, self.dev)
        newf = X.frozen_from_gates(m, chosen, lo, hi)
        if cfg.rot_extra:
            rr = torch.zeros(m.rot_head.weight.shape[0], dtype=torch.bool, device=self.dev)
            rr[4 * lo:4 * hi] = True
            newf["rot_head.weight"] = rr[:, None] & chosen[-1].bool()[None, :]
        for k_ in self.frozen:
            self.frozen[k_] |= newf[k_]
        self.masks.append([g.clone() for g in chosen])
        self.stats.append(X.collect_stats(m, task, self.xcfg, self.dev, chosen))
        cc = X.closure_check(m, task, self.xcfg, self.dev, self.frozen, chosen,
                             self.seed * 17 + t)
        tr.pop("scorer", None)
        row = {"task": t, "bce_last": tr["bce_last"], "full_acc": ps["full_acc"],
               "n_channels": int(sum(int(g.sum()) for g in chosen)),
               "kept_per_stage": [int(g.sum()) for g in chosen],
               "n_trials": ps["n_trials"], "select": cfg.select,
               "val_circuit": val_c, "test_circuit": test_c,
               "ablation": ablation, "closure": cc,
               "frozen_frac": X.frozen_frac(self.frozen)}
        self.per_task.append(row)
        print(f"      select={cfg.select}  full {ps['full_acc']:.4f} -> val "
              f"{val_c:.4f} test {test_c:.4f} on {row['n_channels']} ch "
              f"{row['kept_per_stage']} | closure {cc['feature_delta']:.1e} "
              f"{'OK' if cc['features_closed'] else 'NOT CLOSED'} | frozen "
              f"{row['frozen_frac']:.3f}", flush=True)
        if len(ablation) > 1:
            print("      matched-sparsity selection ablation (val / test, same weights):",
                  flush=True)
            for a in ablation:
                print(f"        {a['name']:>9}  val {a['val']:.4f}  test {a['test']:.4f}",
                      flush=True)
        return row

    @torch.no_grad()
    def evaluate(self, t):
        self.model.eval()
        done = [(self.masks[s], self.stats[s], *self.tasks[s]["classes"])
                for s in range(t + 1)]
        if self.cfg.eval_mixed:
            x, y, tt = mixed_test(self.tasks, t)
            S = X.task_scores(self.model, x, None, done)
            L = X.routing_ladder(S, y, tt, self.dev)
            if self.cfg.rot_extra:
                spans = [self.tasks[s]["classes"] for s in range(t + 1)]
                rs, rc = rot_scores(self.model, x, self.masks[:t + 1], spans)
                pick = rs.argmax(1)
                cls_own = torch.zeros(len(y), dtype=torch.long, device=self.dev)
                cls_rot = torch.zeros(len(y), dtype=torch.long, device=self.dev)
                for u in range(t + 1):
                    mm = pick == u
                    if bool(mm.any()):
                        cls_own[mm] = S["own"][u][mm].argmax(1) + spans[u][0]
                        cls_rot[mm] = rc[u][mm].argmax(1) + spans[u][0]
                L["rot"] = {"pick": pick, "cls": cls_own}
                L["rot_cls"] = {"pick": pick, "cls": cls_rot}
                # rot score OR z, TPL style energy gate, both on own scale:
                # z is standardised (mean 0 on own data), rot is in (0, 1)
                gate = torch.logaddexp(4.0 * rs, -S["z"])
                pg = gate.argmax(1)
                cg = torch.zeros(len(y), dtype=torch.long, device=self.dev)
                for u in range(t + 1):
                    mm = pg == u
                    if bool(mm.any()):
                        cg[mm] = S["own"][u][mm].argmax(1) + spans[u][0]
                L["rot_or_z"] = {"pick": pg, "cls": cg}
            rows = per_task_means(L, tt, y, t + 1)
            # task-IL from the same mixed pass: head logits under mask s on
            # task s's own images
            til = []
            for s in range(t + 1):
                lo, hi = self.tasks[s]["classes"]
                m = tt == s
                til.append(float((S["head"][s][m].argmax(1) + lo == y[m]).float().mean()))
        else:
            rows, til = {}, []
            for s in range(t + 1):
                x, y = self.tasks[s]["test"]
                S = X.task_scores(self.model, x, None, done)
                L = X.routing_ladder(S, y, s, self.dev)
                for r, v in L.items():
                    rows.setdefault(r, []).append(
                        {"task_acc": v["task_acc"], "class_acc": v["class_acc"]})
                til.append(X.acc_task(self.model, self.tasks[s], self.xcfg,
                                      self.dev, "test", self.masks[s]))
        cil = [rows["z"][s]["class_acc"] for s in range(t + 1)]
        return {"til": til, "cil": cil, "ladder": rows}

    def state(self):
        return {"model": self.model.state_dict(), "frozen": self.frozen,
                "masks": self.masks, "stats": self.stats, "per_task": self.per_task}

    def load(self, st):
        self.model.load_state_dict(st["model"])
        self.frozen, self.masks = st["frozen"], st["masks"]
        self.stats, self.per_task = st["stats"], st["per_task"]


METHODS = {"finetune": Finetune, "fecam": FeCAM, "wsn": WSN,
           "supsup": SupSup, "ours": Ours}


# =============================================================================
# run loop with checkpoint / resume / deadline
# =============================================================================

def run_tag(cfg):
    tag = cfg.method
    if cfg.method == "ours" and cfg.select != "causal":
        tag += f"_{cfg.select}"
    return tag


def find_ckpt(name, ckpt_dir: Path, resume_dirs: List[Path]):
    for d in [ckpt_dir] + list(resume_dirs):
        p = Path(d) / name
        if p.exists():
            return p
    return None


def run_seed(cfg, tasks, device, seed, out_dir: Path, ckpt_dir: Path,
             resume_dirs, deadline, stop_after: int = 0):
    T = cfg.n_tasks
    tag = run_tag(cfg)
    name = f"{tag}_seed{seed}.pt"
    meth = METHODS[cfg.method](cfg, tasks, device, seed)
    cil = np.full((T, T), np.nan)
    til = np.full((T, T), np.nan)
    ladders, rows = [], []
    start = 0
    ck = find_ckpt(name, ckpt_dir, resume_dirs)
    if ck is not None:
        st = torch.load(ck, map_location=device, weights_only=False)
        if st.get("cfg") != asdict(cfg):
            print(f"    WARNING: checkpoint config differs from this run's:", flush=True)
            for k in set(st.get("cfg", {})) | set(asdict(cfg)):
                a, b = st.get("cfg", {}).get(k), asdict(cfg).get(k)
                if a != b:
                    print(f"      {k}: ckpt={a} now={b}", flush=True)
        meth.load(st["method"])
        cil, til = np.array(st["cil"]), np.array(st["til"])
        ladders, rows = st["ladders"], st["rows"]
        start = st["next_task"]
        torch.set_rng_state(st["rng"])
        print(f"    resumed {name} from {ck}, next task {start}", flush=True)
    if start >= T:
        print(f"    {name} already complete in checkpoint; not re-run, not re-appended",
              flush=True)
    for t in range(start, T):
        if stop_after and t >= stop_after:
            print(f"    simulated kill before task {t}", flush=True)
            return None
        if deadline and time.time() > deadline:
            print(f"    deadline reached before task {t}; checkpoint holds "
                  f"tasks 0..{t-1}. Re-run to continue.", flush=True)
            return None
        t0 = time.time()
        print(f"  [{tag} seed {seed}] task {t}", flush=True)
        info = meth.learn(t)
        ev = meth.evaluate(t)
        for s in range(t + 1):
            cil[t, s], til[t, s] = ev["cil"][s], ev["til"][s]
        ladders.append(ev.get("ladder"))
        info = {k: v for k, v in info.items() if not isinstance(v, torch.Tensor)}
        info["elapsed_sec"] = time.time() - t0
        rows.append(info)
        msg = (f"    after task {t}: classIL {np.nanmean(cil[t, :t+1]):.4f}  "
               f"taskIL {np.nanmean(til[t, :t+1]):.4f}  ({info['elapsed_sec']:.0f}s)")
        if ev.get("ladder"):
            msg += "  | " + "  ".join(
                f"{r}:{np.mean([v['class_acc'] for v in vs]):.3f}"
                for r, vs in ev["ladder"].items())
        print(msg, flush=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_dir / (name + ".tmp")
        torch.save({"cfg": asdict(cfg), "method": meth.state(), "cil": cil.tolist(),
                    "til": til.tolist(), "ladders": ladders, "rows": rows,
                    "next_task": t + 1, "rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, ckpt_dir / name)
    rec = {"tag": tag, "seed": seed, **score_matrix(cil, til, T),
           "per_task": rows, "ladders": ladders, **{f"cfg_{k}": v for k, v in asdict(cfg).items()}}
    if ladders and ladders[-1]:
        rec["ladder_last"] = {r: float(np.mean([v["class_acc"] for v in vs]))
                              for r, vs in ladders[-1].items()}
        rec["ladder_avg"] = {r: float(np.mean([np.mean([v["class_acc"] for v in L[r]])
                                               for L in ladders]))
                             for r in ladders[-1]}
    if start < T:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{tag}_results.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def summarise(recs, tag):
    if not recs:
        return
    print("\n" + "=" * 78)
    print(f"  {tag}   n = {len(recs)} seeds {[r['seed'] for r in recs]}")
    for k in ("classIL_last", "classIL_avg", "taskIL_last",
              "F_taskIL(weight movement)", "F_classIL(movement + label growth)"):
        v = [r[k] for r in recs]
        print(f"  {k:>36}  {np.mean(v):.4f} +/- {np.std(v):.4f}")
    if "ladder_last" in recs[0]:
        print(f"  {'ladder (classIL last / avg)':>36}")
        for r in recs[0]["ladder_last"]:
            l = [x["ladder_last"][r] for x in recs]
            a = [x["ladder_avg"][r] for x in recs]
            print(f"  {r:>36}  {np.mean(l):.4f} +/- {np.std(l):.4f}   "
                  f"{np.mean(a):.4f} +/- {np.std(a):.4f}")
    if recs[0]["per_task"] and "ablation" in recs[0]["per_task"][0]:
        print(f"  {'selection ablation, task-IL test, mean over tasks and seeds':>36}")
        names = [a["name"] for a in recs[0]["per_task"][0]["ablation"]]
        for nm in names:
            v = [a["test"] for r in recs for p in r["per_task"]
                 for a in p["ablation"] if a["name"] == nm and a["test"] is not None]
            print(f"  {nm:>36}  {np.mean(v):.4f} +/- {np.std(v):.4f}  (n={len(v)} task-runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("ckpt"))
    ap.add_argument("--resume-dir", type=Path, action="append", default=[])
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--deadline-sec", type=float, default=0.0,
                    help="stop cleanly between tasks after this many seconds")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    for f, v in asdict(HCfg()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    cfg = HCfg(**{f: getattr(args, f) for f in asdict(HCfg())})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible; pass --allow-cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.benchmark = True
        print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    deadline = time.time() + args.deadline_sec if args.deadline_sec else 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    tasks, perm = D.prepare_data(args.data_dir, cfg.n_tasks, cfg.cpt,
                                 cfg.split_seed, cfg.val_per_task, device)
    if cfg.train_sub:
        D.subsample_train(tasks, cfg.train_sub)
    print(f"protocol: CIFAR-100 {cfg.n_tasks}x{cfg.cpt}, split seed {cfg.split_seed}, "
          f"class order {perm[:10]}..., train/val/test per task "
          f"{len(tasks[0]['train'][0])}/{len(tasks[0]['val'][0])}/{len(tasks[0]['test'][0])}",
          flush=True)
    print(f"method {run_tag(cfg)}  cfg {json.dumps(asdict(cfg))}", flush=True)
    (args.out / f"{run_tag(cfg)}_config.json").write_text(
        json.dumps({"class_permutation": perm, "split_seed": cfg.split_seed,
                    **asdict(cfg)}, indent=2))
    recs = []
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        print(f"\n=== {run_tag(cfg)} seed {seed} ===", flush=True)
        r = run_seed(cfg, tasks, device, seed, args.out, args.ckpt_dir,
                     args.resume_dir, deadline)
        if r is None:
            print("stopped at deadline; resume later", flush=True)
            break
        recs.append(r)
    summarise(recs, run_tag(cfg))


# =============================================================================
# self test: toy data, every method, checkpoint/resume must reproduce
# =============================================================================

def self_test():
    import tempfile
    dev = torch.device("cpu")
    tasks = X._fixture(dev, n_tasks=2, cpt=3, n=60)
    for tk in tasks:
        tk["classes"] = tuple(tk["classes"])
    base = dict(n_tasks=2, cpt=3, epochs=2, batch_size=20, aug=0,
                stages="4,6,8,10", head_refit_steps=10, rand_draws=1,
                learned_steps=5, prune_tol=0.05, prune_floor=0.0,
                score_order="outw", lr=0.01)
    # 1. augmentation is shape preserving and seeded
    x = torch.randn(5, 3, 32, 32)
    a = D.augment(x, torch.Generator().manual_seed(1))
    b = D.augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape and torch.equal(a, b)
    print("[1] augment is shape preserving and reproducible under a seed")
    # 2. class split and metric
    assert D.class_split(10, 10) == [(i * 10, i * 10 + 10) for i in range(10)]
    cil = np.array([[0.9, np.nan], [0.5, 0.7]]); til = cil.copy()
    sm = score_matrix(cil, til, 2)
    assert abs(sm["classIL_last"] - 0.6) < 1e-9 and abs(sm["classIL_avg"] - 0.75) < 1e-9
    print("[2] last and average incremental accuracy computed as defined")
    # 3. every method runs end to end, and resume reproduces bitwise
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for meth in METHODS:
            cfg = HCfg(method=meth, **base)
            full = run_seed(cfg, tasks, dev, 0, td / "a", td / "ca", [], 0)
            # kill after task 0, then resume from the checkpoint
            half = run_seed(cfg, tasks, dev, 0, td / "b", td / "cb", [], 0, stop_after=1)
            assert half is None
            ck = torch.load(td / "cb" / f"{run_tag(cfg)}_seed0.pt", weights_only=False)
            assert ck["next_task"] == 1
            res = run_seed(cfg, tasks, dev, 0, td / "b", td / "cb", [], 0)
            assert res is not None
            assert np.array_equal(np.array(res["cil_matrix"]), np.array(full["cil_matrix"]),
                                  equal_nan=True), \
                f"{meth}: resumed run differs from the uninterrupted one"
            print(f"[3:{meth}] ran 2 toy tasks (classIL_last {full['classIL_last']:.3f}); "
                  f"resume from checkpoint reproduced the matrix exactly "
                  f"(resumed at task {ck['next_task']})")
    # 4. ours with the rotation head: trains, closes, and reports rot rows
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = HCfg(method="ours", rot_extra=1, **base)
        r = run_seed(cfg, tasks, dev, 0, td / "a", td / "ca", [], 0)
        assert "rot" in r["ladder_last"] and "rot_or_z" in r["ladder_last"]
        assert all(p["closure"]["features_closed"] for p in r["per_task"])
        half = run_seed(cfg, tasks, dev, 0, td / "b", td / "cb", [], 0, stop_after=1)
        res = run_seed(cfg, tasks, dev, 0, td / "b", td / "cb", [], 0)
        assert np.array_equal(np.array(res["cil_matrix"]), np.array(r["cil_matrix"]), equal_nan=True)
        print(f"[3b:ours+rot] rotation head trains, circuits stay closed, rot rows "
              f"present ({r['ladder_last']['rot']:.3f}), resume exact")
    # 4. ours: selection alternatives are reported at matched size
    print("[4] selection ablation rows present:",
          [a["name"] for a in full["per_task"][0]["ablation"]])
    assert all(a["kept"] == full["per_task"][0]["ablation"][0]["kept"]
               for a in full["per_task"][0]["ablation"])
    print("all harness self tests passed")


if __name__ == "__main__":
    main()
