#!/usr/bin/env python3
"""
EXP18: minimal causal circuits, kept by MASKING instead of by SEALING.

WHAT CHANGED FROM EXP17, AND WHY

EXP17 protected a circuit by cutting it out of the network: freeze the whole
incoming row of every claimed channel, and pin the entries that read anything
outside the circuit at zero, permanently. That made the trunk closed with no
mask needed at test time, which is a property nobody else in this family has.
It also consumed the network. Task 0 claimed 68% of the channels and the run
died at task 4 with `locked 1.000`.

EXP18 keeps the circuit the way WSN, PackNet and IBM do. Nothing is zeroed and
nothing is re-initialised. A weight an earlier task depended on simply stops
receiving gradient, and at test time we apply that task's channel mask. Under
the mask, everything outside is silenced anyway, so the weights that were free
to change could not have mattered.

The guarantee survives in a slightly weaker but still unusual form:

    apply M_t, randomise every weight NOT frozen for task t,
    and task t's features are BITWISE unchanged.

WSN and IBM report BWT of exactly 0, which is the behavioural version of this.
Neither shows the structural version. Self-test 5 does, and self-test 6 is its
negative control.

WHAT WE KEEP FROM EXP17

The circuit is still found by greedy causal ablation, deepest stage first, with
already-silenced channels staying silenced. That is the contribution. IBM's own
opening argument is that magnitude "does not necessarily correspond to the
importance of weights"; we agree and answer it differently.

THREE DETAILS COPIED FROM WSN's CODE, DELIBERATELY

  BatchNorm has affine=False and track_running_stats=False everywhere. There
  are then no per-channel BN parameters to freeze and no statistics to go
  stale when a later task reuses a frozen channel on different data. The cost
  is that test-time normalisation uses the test batch, which is mildly
  transductive. --per-task-bn switches to the alternative.

  No bias anywhere.

  Warm start. Task t+1 begins from task t's weights. EXP17 scrambled them,
  which threw away everything the free capacity had learned. IBM's ablation
  puts re-initialising learned parameters at 85.22 against 88.15 for keeping
  them.

Usage:
  python exp18_masked_circuits.py --self-test
  python exp18_masked_circuits.py --data-dir data --n-tasks 10 --cpt 10
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cil_data as E17                     # data loading and augmentation only (exp17 was never checked in)


# =============================================================================
# model
# =============================================================================

def _bn(c, per_task_bn: bool):
    """WSN's choice: no affine parameters, no running statistics. Nothing per
    channel to freeze, nothing to go stale. See the module docstring."""
    if per_task_bn:
        return nn.BatchNorm2d(c, track_running_stats=False, affine=True)
    return nn.BatchNorm2d(c, track_running_stats=False, affine=False)


class Block(nn.Module):
    def __init__(self, cin, cout, stride, ptbn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = _bn(cout, ptbn)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = _bn(cout, ptbn)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      _bn(cout, ptbn))

    def forward(self, x, gate=None):
        h = F.relu(self.bn1(self.conv1(x)))
        if gate is not None:
            h = h * gate.view(1, -1, 1, 1)
        h = self.bn2(self.conv2(h))
        s = x if self.down is None else self.down(x)
        out = F.relu(h + s)
        if gate is not None:
            out = out * gate.view(1, -1, 1, 1)
        return out


class MaskedResNet18(nn.Module):
    """CIFAR ResNet-18. One channel mask per stage, because the identity skips
    force channel c to be present at a stage's input if it is present at its
    output. Same constraint as EXP17 and same reason."""

    def __init__(self, n_classes, seed, device, blocks=2, stages=None, ptbn=False):
        super().__init__()
        torch.manual_seed(seed)
        self.STAGES = tuple(stages) if stages else (64, 128, 256, 512)
        self.conv1 = nn.Conv2d(3, self.STAGES[0], 3, 1, 1, bias=False)
        self.bn1 = _bn(self.STAGES[0], ptbn)
        cin = self.STAGES[0]
        self.stages = nn.ModuleList()
        for si, cout in enumerate(self.STAGES):
            blks = [Block(cin if b == 0 else cout, cout,
                          1 if (si == 0 or b > 0) else 2, ptbn)
                    for b in range(blocks)]
            self.stages.append(nn.ModuleList(blks))
            cin = cout
        self.head = nn.Linear(self.STAGES[-1], n_classes, bias=False)
        self.width = self.STAGES[-1]
        self.depth = 1
        self.n_stages = len(self.STAGES)
        self.to(device)

    def features(self, x, gates=None, fill=None):
        _ = fill
        h = F.relu(self.bn1(self.conv1(x)))
        if gates is not None:
            h = h * gates[0].view(1, -1, 1, 1)
        for si, blks in enumerate(self.stages):
            g = None if gates is None else gates[si]
            for blk in blks:
                h = blk(h, g)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def stage_input(self, x, gates=None, si=0):
        h = F.relu(self.bn1(self.conv1(x)))
        for s in range(si):
            if s == 0 and gates is not None:
                h = h * gates[0].view(1, -1, 1, 1)
            g = None if gates is None else gates[s]
            for blk in self.stages[s]:
                h = blk(h, g)
        return h

    def head_from(self, h, gates=None, si=0):
        if si == 0 and gates is not None:
            h = h * gates[0].view(1, -1, 1, 1)
        for s in range(si, len(self.stages)):
            g = None if gates is None else gates[s]
            for blk in self.stages[s]:
                h = blk(h, g)
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))

    def forward(self, x, gates=None, fill=None):
        return self.head(self.features(x, gates, fill))


def all_on(model, device):
    return [torch.ones(c, device=device) for c in model.STAGES]


def stage_convs(model, si: int):
    """(module path, which mask gates this conv's INPUT channels).

    'prev'  the previous stage's mask; for si=0 that is stage 0's own, because
            gates[0] covers the stem as well as layer1
    'same'  this stage's mask
    'image' the three image channels, always present
    """
    out = []
    if si == 0:
        out.append(("conv1", "image"))
    for bi in range(len(model.stages[si])):
        b = f"stages.{si}.{bi}"
        out.append((f"{b}.conv1", "prev" if bi == 0 else "same"))
        out.append((f"{b}.conv2", "same"))
        if model.stages[si][bi].down is not None:
            out.append((f"{b}.down.0", "prev"))
    return out


# =============================================================================
# the freezing rule
# =============================================================================

@torch.no_grad()
def frozen_from_gates(model, gates, lo: int, hi: int) -> Dict[str, torch.Tensor]:
    """Which weights must not move for THIS task's masked features to be fixed.

    Under mask M, channel c reads only channels that M leaves alive. So the
    entries that matter are W[c, g] with c in M at this stage and g in M at the
    stage feeding it. Everything else in c's row is irrelevant to this task and
    stays trainable for later ones.

    That is the whole difference from EXP17, which froze all of W[c, :] and
    zeroed the part outside the circuit.
    """
    out = {n: torch.zeros_like(p, dtype=torch.bool)
           for n, p in model.named_parameters()}
    mods = dict(model.named_modules())
    for si in range(model.n_stages):
        rows = gates[si].bool()
        if not bool(rows.any()):
            continue
        prev = gates[si - 1].bool() if si > 0 else gates[0].bool()
        same = gates[si].bool()
        for path, kind in stage_convs(model, si):
            w = mods[path].weight
            if kind == "image":
                src = torch.ones(w.shape[1], dtype=torch.bool, device=w.device)
            else:
                src = prev if kind == "prev" else same
            m = rows[:, None] & src[None, :]
            out[f"{path}.weight"] |= m[:, :, None, None].expand_as(w)
    hrows = torch.zeros(model.head.weight.shape[0], dtype=torch.bool,
                        device=model.head.weight.device)
    hrows[lo:hi] = True
    out["head.weight"] |= hrows[:, None] & gates[-1].bool()[None, :]
    return out


class GradFreezer:
    """Zero the gradient of every weight any earlier task depends on, exactly
    as WSN does: `grad = grad * (1 - M_all)`. No values are written, no zeros
    are stored. Integer indices are computed once; a boolean mask on the
    gradient would call nonzero() and synchronise the device every step."""

    def __init__(self, model, frozen: Dict[str, torch.Tensor]):
        self.items = []
        for n, p in model.named_parameters():
            m = frozen.get(n)
            if m is not None and bool(m.any()):
                self.items.append((p, m.reshape(-1).nonzero(as_tuple=True)[0]))
        self.ref = {id(p): p.detach().reshape(-1)[i].clone()
                    for p, i in self.items}

    @torch.no_grad()
    def mask_grads(self):
        for p, i in self.items:
            if p.grad is not None:
                p.grad.view(-1).index_fill_(0, i, 0.0)

    @torch.no_grad()
    def verify(self) -> float:
        worst = 0.0
        for p, i in self.items:
            d = float((p.detach().reshape(-1)[i] - self.ref[id(p)]).abs().max())
            worst = d if d != d else max(worst, d)
        return worst


def frozen_frac(frozen) -> float:
    n = sum(int(v.numel()) for v in frozen.values())
    return sum(int(v.sum()) for v in frozen.values()) / max(1, n)


def frozen_count(frozen) -> int:
    return sum(int(v.sum()) for v in frozen.values())


def circuit_weights(model, gates, lo, hi) -> int:
    """Circuit size measured in WEIGHTS, which is what IBM's Figure 3 plots.
    Channel counts are not comparable across papers."""
    return frozen_count(frozen_from_gates(model, gates, lo, hi))


# =============================================================================
# config
# =============================================================================

@dataclass
class Cfg:
    n_tasks: int = 10
    cpt: int = 10
    base_classes: int = 0
    epochs: int = 300            # IBM and WSN both use 300
    batch_size: int = 256        # IBM
    lr: float = 1e-3             # IBM: Adam, fixed
    weight_decay: float = 0.0
    val_per_task: int = 500
    prune_tol: float = 0.02
    prune_floor: float = 0.30
    prune_order: str = "deep"
    score_order: str = "taylor"  # taylor | absg | sq | signed | outw | random
                                 # taylor, absg, sq and signed come from the
                                 # gradient ARRIVING at each channel, summed
                                 # over the task's whole training run. See
                                 # ChannelScorer. Default is taylor because it
                                 # had the best rank correlation with true
                                 # single-channel ablation cost (+0.404 against
                                 # +0.249 for outw and -0.205 for signed), but
                                 # --verify-filter is what actually decides,
                                 # because rank correlation is not the question.
    # Two ways to use a cheap score to shorten the sweep. They fail
    # differently and --verify-filter measures both.
    skip_frac: float = 0.0       # do not TEST the top this-fraction. They stay
                                 # in the circuit untested. Can only make the
                                 # circuit larger, never break the gate. Safe
                                 # direction, smaller saving.
    drop_frac: float = 0.0       # REMOVE the bottom this-fraction without
                                 # testing. This is the version that actually
                                 # saves time when circuits are small, and the
                                 # version that can break things. Risky
                                 # direction, larger saving.
    cache_acts: int = 1
    verify_filter: int = 0       # run the FULL sweep and every filtered sweep
                                 # on the same trained model and report what
                                 # each filter cost. See verify_filters().
    per_task_bn: int = 0
    head_refit_steps: int = 400  # refit the task's head rows AFTER the mask is
                                 # decided. EXP17 never did this and its
                                 # `circuit` and `after reinit` numbers
                                 # disagreed on every task after the first.
    aug: int = 1
    rand_draws: int = 3
    stages: str = "64,128,256,512"
    depth: int = 1
    amp: int = 0
    tf32: int = 1
    channels_last: int = 0       # off: the gates make NHWC less of a win and
                                 # it changes kernels under us
    cudnn_benchmark: int = 1

    def stages_tuple(self):
        return tuple(int(v) for v in self.stages.split(",") if v.strip())


def n_classes_of(cfg) -> int:
    return E17.class_split(cfg.n_tasks, cfg.cpt, cfg.base_classes)[-1][1]


# =============================================================================
# train / evaluate
# =============================================================================

def train_task(model, tasks, t, cfg, device, seed, frozen) -> Dict:
    """Train DENSE, with the gradients of everything earlier tasks depend on
    zeroed. We cannot apply this task's mask during training the way WSN does,
    because we do not know the mask until the causal ablation runs, which needs
    a trained network. So: train dense, then ablate, then refit the head."""
    lo, hi = tasks[t]["classes"]
    x, y = tasks[t]["train"]
    gf = GradFreezer(model, frozen)
    sc = (ChannelScorer(model, device)
          if cfg.score_order in ("signed", "absg", "taylor", "sq")
          or cfg.verify_filter else None)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(seed)
    tgt_table = torch.eye(hi - lo, device=device)
    nsteps = cfg.epochs * ((len(x) + cfg.batch_size - 1) // cfg.batch_size)
    lbuf = torch.zeros(nsteps, device=device)
    k = 0
    model.train()
    for _ in range(cfg.epochs):
        order = torch.randperm(len(x), generator=g).to(device)
        for s in range(0, len(x), cfg.batch_size):
            i = order[s:s + cfg.batch_size]
            xb = E17.augment(x[i], g) if cfg.aug else x[i]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=bool(cfg.amp) and device.type == "cuda"):
                lg = model(xb)[:, lo:hi]
                loss = F.binary_cross_entropy_with_logits(
                    lg.float(), tgt_table[y[i] - lo])
            loss.backward()
            gf.mask_grads()
            opt.step()
            lbuf[k] = loss.detach()
            k += 1
    drift = gf.verify()
    if drift != 0.0:
        raise RuntimeError(f"a frozen weight moved by {drift}; gradient "
                           f"masking is broken")
    if sc is not None:
        sc.close()
    return {"bce_last": float(lbuf[max(0, k - 50):k].mean()), "scorer": sc}


@torch.no_grad()
def acc_task(model, task, cfg, device, split="test", gates=None,
             batch=512) -> float:
    """Task-aware accuracy: only this task's own classes compete."""
    model.eval()
    x, y = task[split]
    lo, hi = task["classes"]
    out = [model(x[s:s + batch], gates)[:, lo:hi]
           for s in range(0, len(x), batch)]
    return float((torch.cat(out).argmax(-1) + lo == y).float().mean())


@torch.no_grad()
def head_refit(model, task, cfg, device, gates, steps: int, seed: int = 0):
    """Refit ONLY this task's head rows, on the features it will actually read.

    The head was trained while every channel was alive. The mask then silences
    some of them. Without this the head is reading a channel set it was never
    fitted to, which in EXP17 cost up to 0.18 of task-IL on a single task.
    Legal: uses this task's own data, at the time the task is being learned.
    """
    lo, hi = task["classes"]
    x, y = task["train"]
    model.eval()
    f = torch.cat([model.features(x[s:s + 512], gates)
                   for s in range(0, len(x), 512)])
    keep = gates[-1].bool()
    fk = f[:, keep]
    w = model.head.weight[lo:hi][:, keep].detach().clone().requires_grad_(True)
    tgt = torch.eye(hi - lo, device=device)[y - lo]
    opt = torch.optim.Adam([w], lr=1e-2)
    g = torch.Generator().manual_seed(seed)
    with torch.enable_grad():
        for _ in range(steps):
            i = torch.randint(0, len(fk), (min(512, len(fk)),),
                              generator=g).to(device)
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(fk[i] @ w.t(), tgt[i]).backward()
            opt.step()
    row = torch.zeros(hi - lo, model.width, device=device)
    row[:, keep] = w.detach()
    model.head.weight[lo:hi] = row      # entries outside the mask are 0 here,
                                        # but they are NOT frozen: a later task
                                        # never reads this row, so they are
                                        # simply unused rather than destroyed.


# =============================================================================
# causal ablation
# =============================================================================

class ChannelScorer:
    """Accumulate, per channel, statistics of the gradient ARRIVING at it.

    Note "arriving at": this is dL/dh for the channel's activation, not the
    gradient of its weights. That distinction matters here, because a channel
    frozen by an earlier task has zero weight gradient by construction while
    still being essential to the current task. Its activation gradient is not
    zero.

    Four accumulators, because they answer different questions:

        signed   sum of dL/dh.        Tends to zero at convergence for a
                                      well-fitted channel, which is the same
                                      thing it does for a dead one.
        absg     sum of |dL/dh|.      How much this channel was pushed around.
        taylor   sum of |dL/dh * h|.  First-order estimate of the loss change
                                      from setting h to zero.
        sq       sum of (dL/dh)^2.    Fisher-style.

    Free: it rides the backward pass that is already happening.
    """

    def __init__(self, model, device):
        self.acc = {k: [torch.zeros(c, device=device) for c in model.STAGES]
                    for k in ("signed", "absg", "taylor", "sq")}
        self.h = [blks[-1].register_forward_hook(self._mk(si))
                  for si, blks in enumerate(model.stages)]

    def _mk(self, si):
        def hook(mod, inp, out):
            if not out.requires_grad:
                return
            det = out.detach()

            def back(gr, si=si, det=det):
                self.acc["signed"][si] += gr.sum((0, 2, 3))
                self.acc["absg"][si] += gr.abs().sum((0, 2, 3))
                self.acc["taylor"][si] += (gr * det).abs().sum((0, 2, 3))
                self.acc["sq"][si] += gr.pow(2).sum((0, 2, 3))
            out.register_hook(back)
        return hook

    def close(self):
        for h in self.h:
            h.remove()

    def get(self, key: str):
        if key == "signed":
            return [v.abs() for v in self.acc["signed"]]
        return [v.clone() for v in self.acc[key]]


@torch.no_grad()
def channel_scores(model, cfg, device, seed: int, scorer=None):
    """Ordering only. Outgoing weight magnitude was the best cheap predictor of
    true ablation cost in the measurements we ran (Spearman +0.249 against
    +0.404 for Taylor but far better than any gradient score when used to
    choose what to keep). Never used to decide, only to sequence."""
    if cfg.score_order in ("signed", "absg", "taylor", "sq"):
        if scorer is None:
            raise RuntimeError(f"--score-order {cfg.score_order} needs the "
                               f"gradient scorer, which train_task did not "
                               f"attach")
        return scorer.get(cfg.score_order)
    out = []
    for si, c in enumerate(model.STAGES):
        if cfg.score_order == "random":
            gg = torch.Generator(device="cpu").manual_seed(seed + si)
            out.append(torch.rand(c, generator=gg).to(device))
        elif si == len(model.STAGES) - 1:
            out.append(model.head.weight.abs().sum(0).detach().clone())
        else:
            nxt = model.stages[si + 1][0]
            v = nxt.conv1.weight.abs().sum((0, 2, 3))
            if nxt.down is not None:
                v = v + nxt.down[0].weight.abs().sum((0, 2, 3))
            out.append(v.detach().clone())
    return out


@torch.no_grad()
def prune_circuit(model, task, cfg, device, seed, verbose=True, scorer=None):
    """Greedy backward elimination over channels, deepest stage first.

    Already-silenced channels stay silenced, so each trial asks "given what I
    have already removed, can this go too". That is what makes it find a set
    rather than a ranking: of two redundant channels the first goes and the
    second then becomes necessary. A one-shot importance score drops both, and
    we measured that failure directly (one-shot by any gradient score is worse
    than picking at RANDOM at high sparsity).
    """
    x, y = task["val"]
    lo, hi = task["classes"]
    S = model.n_stages
    gates = all_on(model, device)
    model.eval()

    def acc(gt):
        out = [model(x[s:s + 512], gt)[:, lo:hi] for s in range(0, len(x), 512)]
        return float((torch.cat(out).argmax(-1) + lo == y).float().mean())

    def acc_cached(gt, si, cache):
        out = [model.head_from(h, gt, si)[:, lo:hi] for h in cache]
        return float((torch.cat(out).argmax(-1) + lo == y).float().mean())

    full = acc(gates)
    scores = channel_scores(model, cfg, device, seed, scorer)
    order_stages = range(S) if cfg.prune_order == "shallow" else range(S - 1, -1, -1)
    tried = 0
    untested_kept = untested_dropped = 0
    for si in order_stages:
        gate = max(cfg.prune_floor, acc(gates) - cfg.prune_tol / S)
        idx = torch.argsort(scores[si]).tolist()          # least important first
        if cfg.drop_frac > 0:
            # remove the lowest scoring without testing them
            n_drop = min(int(round(cfg.drop_frac * len(idx))), len(idx) - 1)
            for c in idx[:n_drop]:
                gates[si][c] = 0.0
            untested_dropped += n_drop
            idx = idx[n_drop:]
        if cfg.skip_frac > 0:
            keep_n = int(round(cfg.skip_frac * len(idx)))
            if keep_n:
                untested_kept += keep_n
                idx = idx[:len(idx) - keep_n]
        cache = None
        if cfg.cache_acts:
            cache = [model.stage_input(x[s:s + 512], gates, si)
                     for s in range(0, len(x), 512)]
        run = (lambda gt: acc_cached(gt, si, cache)) if cache is not None else acc
        for c in idx:
            if gates[si][c] == 0 or int(gates[si].sum()) <= 1:
                continue
            gates[si][c] = 0.0
            tried += 1
            if run(gates) < gate:
                gates[si][c] = 1.0
        del cache
        if verbose:
            print(f"      stage {si}: kept {int(gates[si].sum()):>4} of "
                  f"{model.STAGES[si]}", flush=True)
    kept = [int(v.sum()) for v in gates]
    after = acc(gates)
    if (cfg.drop_frac or cfg.skip_frac) and after < max(cfg.prune_floor,
                                                        full - cfg.prune_tol):
        print(f"      NOTE the score filter pushed this circuit past the "
              f"tolerance: {full:.4f} -> {after:.4f} on val. The filter is "
              f"not free here.", flush=True)
    return gates, {"full_acc": full, "acc_val_circuit": after,
                   "kept_per_stage": kept, "n_channels": int(sum(kept)),
                   "n_trials": tried, "untested_kept": untested_kept,
                   "untested_dropped": untested_dropped,
                   "score_order": cfg.score_order}


@torch.no_grad()
def random_gates(model, sizes, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for c, k in zip(model.STAGES, sizes):
        v = torch.zeros(c, device=device)
        v[torch.randperm(c, generator=g)[:k].to(device)] = 1.0
        out.append(v)
    return out


@torch.no_grad()
def min_random_scale(model, task, cfg, device, seed, gate, kept, draws=3):
    """How small can a RANDOM circuit of the same shape be and still meet the
    gate? Searches only sizes up to ours, so a None result means "no random
    circuit at or below our size", not "at any size"."""
    tot = sum(kept)
    lo_p, hi_p, best = 1, 100, None
    while lo_p <= hi_p:
        mid = (lo_p + hi_p) // 2
        sizes = [max(1, min(c, int(round(k * mid / 100.0))))
                 for k, c in zip(kept, model.STAGES)]
        a = [acc_task(model, task, cfg, device, "val",
                      random_gates(model, sizes, device, seed * 977 + mid * 7 + d))
             for d in range(draws)]
        if float(np.median(a)) >= gate:
            best, hi_p = (mid, sizes), mid - 1
        else:
            lo_p = mid + 1
    if best is None:
        return {"min_random_channels": None, "compression": None,
                "control": "no random circuit at or below our size met the gate"}
    _, sizes = best
    n = sum(sizes)
    return {"min_random_channels": n, "compression": tot / max(1, n),
            "min_random_per_stage": sizes, "control": "measured"}


@torch.no_grad()
def verify_filters(model, task, cfg, device, seed, scorer, full_gates, full_ps):
    """Does a cheap score let us skip candidates without changing the circuit?

    This is the question, and it is NOT the one a one-shot pruning comparison
    answers. One-shot asks "can the score replace causal ablation". This asks
    "can the score tell causal ablation which channels are not worth testing",
    with the ablation still making every decision it does make.

    We run the unfiltered sweep once, then each filtered sweep on the SAME
    trained weights, and report three things per setting:

        agree      Jaccard overlap between the filtered circuit and the full one
        cost       val accuracy of the filtered circuit minus the full one
        trials     how many ablations it took, against the full sweep

    A filter is worth using if agree is near 1 and cost is near 0. If a score
    cannot manage that, it cannot be used to shorten the sweep, whatever its
    rank correlation with single-channel ablation cost looks like.
    """
    base = torch.cat([g.bool() for g in full_gates])
    ref_acc = full_ps["acc_val_circuit"]
    ref_trials = full_ps["n_trials"]
    rows = []
    grid = [("outw", 0.0, 0.0)]
    for s in ("taylor", "absg", "sq", "signed", "outw", "random"):
        for dr in (0.25, 0.50, 0.75):
            grid.append((s, dr, 0.0))
    for s, dr, sk in grid:
        c2 = Cfg(**{**asdict(cfg), "score_order": s, "drop_frac": dr,
                    "skip_frac": sk, "verify_filter": 0})
        g2, p2 = prune_circuit(model, task, c2, device, seed, verbose=False,
                               scorer=scorer)
        m2 = torch.cat([g.bool() for g in g2])
        inter = int((base & m2).sum())
        union = int((base | m2).sum())
        rows.append({"score": s, "drop_frac": dr, "skip_frac": sk,
                     "jaccard": inter / max(1, union),
                     "n_channels": p2["n_channels"],
                     "acc": p2["acc_val_circuit"],
                     "acc_cost": p2["acc_val_circuit"] - ref_acc,
                     "trials": p2["n_trials"],
                     "trial_frac": p2["n_trials"] / max(1, ref_trials)})
    print(f"      FILTER CHECK. full sweep: {full_ps['n_channels']} ch, "
          f"val {ref_acc:.4f}, {ref_trials} ablations", flush=True)
    print(f"      {'score':>8} {'drop':>5} {'agree':>6} {'ch':>5} "
          f"{'val':>7} {'cost':>7} {'trials':>7}", flush=True)
    for r in rows:
        print(f"      {r['score']:>8} {r['drop_frac']:>5.2f} "
              f"{r['jaccard']:>6.3f} {r['n_channels']:>5} {r['acc']:>7.4f} "
              f"{r['acc_cost']:>+7.4f} {r['trial_frac']:>6.2f}x", flush=True)
    return rows


# =============================================================================
# the guarantee
# =============================================================================

@torch.no_grad()
def closure_check(model, task, cfg, device, frozen, gates, seed) -> Dict:
    """Apply M_t, randomise EVERY weight not frozen for task t, and ask whether
    task t's features moved. Bitwise, no tolerance.

    WSN and IBM report BWT of exactly 0, which is the behavioural statement.
    This is the structural one and neither of them shows it. Self-test 6 runs
    the same check with nothing frozen and requires it to FAIL, so a pass here
    cannot be vacuous.
    """
    model.eval()
    x, _ = task["test"]
    lo, hi = task["classes"]
    xb = x[: min(512, len(x))]
    keep = gates[-1].bool()
    if not bool(keep.any()):
        raise RuntimeError("the mask is empty at the last stage")
    f0 = model.features(xb, gates)[:, keep].clone()
    l0 = model(xb, gates)[:, lo:hi].clone()
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for n, p in model.named_parameters():
        r = (torch.randn(p.shape, generator=g) * 0.5).to(p.device)
        p.copy_(torch.where(frozen[n], p, r))
    f1 = model.features(xb, gates)[:, keep]
    l1 = model(xb, gates)[:, lo:hi]
    out = {"feature_delta": float((f1 - f0).abs().max()),
           "logit_delta": float((l1 - l0).abs().max()),
           "features_closed": bool(torch.equal(f1, f0)),
           "logits_closed": bool(torch.equal(l1, l0))}
    for n, p in model.named_parameters():
        p.copy_(snap[n])
    return out


# =============================================================================
# readouts
# =============================================================================

@torch.no_grad()
def collect_stats(model, task, cfg, device, gates, tukey=0.5, ridge=1e-2):
    """Per CLASS: mean and covariance of the circuit's features, plus the mean
    and spread of that class's own Mahalanobis distances.

    Per class, not per task. Our own runs said this is worth about 15 points:
    FeCAM's per-class Mahalanobis on EXP17's circuits reached 0.4795 where a
    single Gaussian per task reached 0.3265, and the gap replicated across two
    runs. Tukey and the ridge follow FeCAM; without them raw Mahalanobis
    collapses (their ablation: 29.7 against 70.9 average).
    """
    lo, hi = task["classes"]
    x, y = task["train"]
    model.eval()
    keep = gates[-1].bool()
    f = torch.cat([model.features(x[s:s + 512], gates)[:, keep]
                   for s in range(0, len(x), 512)])
    f = f.clamp_min(0).pow(tukey) if tukey else f          # features are ReLU
    k = int(keep.sum())
    out = {}
    for c in range(lo, hi):
        fc = f[y == c]
        mu = fc.mean(0)
        d = fc - mu
        cov = (d.t() @ d) / max(1, len(fc) - 1)
        cov = cov + (ridge * torch.diag(cov).mean() + 1e-6) * torch.eye(k, device=device)
        prec = torch.linalg.inv(cov.double()).float()
        dist = ((d @ prec) * d).sum(1).clamp_min(0).sqrt()
        out[c] = {"mu": mu, "prec": prec,
                  "m": float(dist.mean()), "s": float(dist.std()) + 1e-6}
    # TPL sets its two temperatures to the empirical mean of each score on the
    # task's OWN training data. Legal, and it is two more scalars per task.
    hl = torch.cat([model(x[s:s + 512], gates)[:, lo:hi]
                    for s in range(0, len(x), 512)])
    return {"classes": out, "keep": keep.clone(), "k": k, "tukey": tukey,
            "mls_mean": float(hl.max(1).values.mean()),
            "ebo_mean": float(torch.logsumexp(hl, 1).mean()),
            "row_norm": model.head.weight[lo:hi].norm(dim=1).clamp_min(1e-6)
                        .detach().clone()}


@torch.no_grad()
def task_scores(model, x, stats, tasks_done, batch=512):
    """Everything the routing ladder needs, computed in ONE pass per circuit.

    Returns per task:
        z        the z-standardised distance to that task's nearest class
        zraw     the same distance WITHOUT the standardisation
        own      per-class -z, used as the class score once a circuit is picked
        head     per-class head logits under that task's mask
        cal      (mls_mean, ebo_mean, row_norm) from that task's own train data

    Every row of the ladder is a different function of these. Nothing here
    needs retraining and nothing needs stored images.
    """
    _ = stats
    T = len(tasks_done)
    dev = x.device
    z = torch.full((len(x), T), float("inf"), device=dev)
    zraw = torch.full((len(x), T), float("inf"), device=dev)
    own, head, cal = [None] * T, [None] * T, [None] * T
    for t, (gates, st, lo, hi) in enumerate(tasks_done):
        f = torch.cat([model.features(x[s:s + batch], gates)[:, st["keep"]]
                       for s in range(0, len(x), batch)])
        if st["tukey"]:
            f = f.clamp_min(0).pow(st["tukey"])
        best, bestraw, logits = None, None, []
        for c in range(lo, hi):
            s = st["classes"][c]
            d = f - s["mu"]
            dist = ((d @ s["prec"]) * d).sum(1).clamp_min(0).sqrt()
            zz = (dist - s["m"]) / s["s"]
            logits.append(-zz)
            best = zz if best is None else torch.minimum(best, zz)
            bestraw = dist if bestraw is None else torch.minimum(bestraw, dist)
        z[:, t], zraw[:, t] = best, bestraw
        own[t] = torch.stack(logits, 1)
        head[t] = torch.cat([model(x[s:s + batch], gates)[:, lo:hi]
                             for s in range(0, len(x), batch)])
        cal[t] = st
    return {"z": z, "zraw": zraw, "own": own, "head": head, "cal": cal,
            "spans": [(lo, hi) for _, _, lo, hi in tasks_done]}


LADDER_ROWS = ("no_router", "raw", "z", "complement", "mls", "ebo",
               "or_gate", "rownorm_mls", "oracle")


def routing_ladder(S, y, true_task: int, device) -> Dict[str, Dict[str, float]]:
    """One frozen set of circuits, one change per row.

    The point is not to win. It is to see which change buys what. If
    `no_router` is level with `z`, our standardisation does nothing and we
    should say so before writing it up as a contribution.

    A note on the `complement` row, because it is a finding and not a
    disappointment. TPL scores task t as E_t minus an estimate of the density
    of everything else, and reports about +8 points for it. Their complement is
    a kNN distance to a REPLAY BUFFER, that is, a DIFFERENT estimator from the
    Mahalanobis they use for E_t, and their own ablation says the asymmetry is
    the point. If we build the complement from the same Gaussians we already
    have, then for the winning task the complement is the runner-up's score and
    for every other task it is the winner's. The ordering is preserved exactly
    and the argmax cannot move. It is a monotone transform of the score we are
    already using. The row exists to demonstrate that, so we do not claim
    TPL's +8 for something that cannot deliver it without replay.

    What can help with no replay is combining two genuinely DIFFERENT
    estimators: the Mahalanobis distance and the head's own logits. That is
    `or_gate`, and it is the transferable half of TPL.
    """
    T = S["z"].shape[1]
    spans = S["spans"]
    z, zraw = S["z"], S["zraw"]
    out = {}

    def cls_from(pick):
        c = torch.zeros(len(y), dtype=torch.long, device=device)
        for u in range(T):
            m = pick == u
            if bool(m.any()):
                c[m] = S["own"][u][m].argmax(1) + spans[u][0]
        return c

    def record(name, pick, cls=None):
        cls = cls_from(pick) if cls is None else cls
        # true_task may be an int (one task's test set) or a per-image tensor
        # (mixed batches, see cil_harness). pick and cls are kept so a caller
        # can break the means down per task.
        out[name] = {"task_acc": float((pick == true_task).float().mean()),
                     "class_acc": float((cls == y).float().mean()),
                     "pick": pick, "cls": cls}

    # no task decision at all: concatenate every circuit's HEAD LOGITS and
    # take the argmax over all classes seen so far. This is what a plain
    # multi-head network with no router does.
    #
    # P6 fix. The old row concatenated S["own"], our own per-class -z scores.
    # argmax over concatenated -z picks the class of the task with the
    # smallest z, which is exactly what the `z` row does, so the two rows
    # printed identical numbers on every run. It was a tautology, not a
    # baseline. Head logits are a different estimator, so this row can now
    # differ from `z` in either direction.
    cat = torch.cat(S["head"], 1)
    offs = torch.cat([torch.arange(lo, hi, device=device) for lo, hi in spans])
    owner = torch.cat([torch.full((hi - lo,), u, device=device,
                                  dtype=torch.long)
                       for u, (lo, hi) in enumerate(spans)])
    j = cat.argmax(1)
    record("no_router", owner[j], offs[j])

    record("raw", zraw.argmin(1))          # no standardisation
    record("z", z.argmin(1))               # ours

    # complement, built from the same Gaussians. see the docstring.
    if T > 1:
        big = z.max() + 1.0
        comp = torch.stack(
            [(-torch.logsumexp(-torch.cat([z[:, :t], z[:, t + 1:]], 1), 1))
             if T > 1 else torch.full_like(z[:, 0], float(big))
             for t in range(T)], 1)
        record("complement", (z - comp).argmin(1))
    else:
        record("complement", z.argmin(1))

    mls = torch.stack([h.max(1).values for h in S["head"]], 1)
    record("mls", mls.argmax(1))
    ebo = torch.stack([torch.logsumexp(h, 1) for h in S["head"]], 1)
    record("ebo", ebo.argmax(1))

    # TPL's energy OR-gate over two different estimators. Their 1/beta is the
    # empirical mean of each score on that task's own training data. Our z is
    # already standardised to mean 0 on own data, so its beta is 1 and only
    # the logit score needs its constant.
    b1 = torch.tensor([1.0 / max(1e-6, abs(st["mls_mean"]))
                       for st in S["cal"]], device=device)
    gate = torch.logaddexp(b1[None, :] * mls, -z)
    record("or_gate", gate.argmax(1))

    # DPCR's category-wise normalisation: divide each class's head row by its
    # own norm before comparing across circuits
    rn = torch.stack([(h / st["row_norm"][None, :]).max(1).values
                      for h, st in zip(S["head"], S["cal"])], 1)
    record("rownorm_mls", rn.argmax(1))

    record("oracle", true_task.clone() if torch.is_tensor(true_task) else
           torch.full((len(y),), true_task, device=device, dtype=torch.long))
    return out


# =============================================================================
# the run
# =============================================================================

def run_cell(tasks, cfg, device, seed, out_dir: Path) -> Dict:
    n_cls = n_classes_of(cfg)
    model = MaskedResNet18(n_cls, seed, device, stages=cfg.stages_tuple(),
                           ptbn=bool(cfg.per_task_bn))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    T = cfg.n_tasks
    frozen = {n: torch.zeros_like(p, dtype=torch.bool)
              for n, p in model.named_parameters()}
    masks: List[List[torch.Tensor]] = []
    stats_all = []
    per_task = []
    aware = np.full((T, T), np.nan)     # A[T_row, i] with task id given
    agn = np.full((T, T), np.nan)       # class-IL, routed
    agn_cat = np.full((T, T), np.nan)   # class-IL, no router, concat logits
    agn_or = np.full((T, T), np.nan)    # class-IL with an oracle router
    route = np.full((T, T), np.nan)     # measured task identification
    ladder = {r: {"task": np.full((T, T), np.nan),
                  "class": np.full((T, T), np.nan)} for r in LADDER_ROWS}
    diag = np.full(T, np.nan)           # A[i, i]

    for t in range(T):
        t0 = time.time()
        lo, hi = tasks[t]["classes"]
        tr = train_task(model, tasks, t, cfg, device, seed * 100 + t, frozen)
        gates, ps = prune_circuit(model, tasks[t], cfg, device, seed * 7 + t,
                                  scorer=tr.get("scorer"))
        if cfg.verify_filter:
            ps["filter_check"] = verify_filters(
                model, tasks[t], cfg, device, seed * 7 + t,
                tr.get("scorer"), gates, ps)
        if cfg.head_refit_steps:
            head_refit(model, tasks[t], cfg, device, gates,
                       cfg.head_refit_steps, seed * 3 + t)
            ps["acc_val_circuit"] = acc_task(model, tasks[t], cfg, device,
                                             "val", gates)
        a_test = acc_task(model, tasks[t], cfg, device, "test", gates)
        a_rand = acc_task(model, tasks[t], cfg, device, "test",
                          random_gates(model, ps["kept_per_stage"], device,
                                       seed * 13 + t))
        mr = min_random_scale(model, tasks[t], cfg, device, seed * 19 + t,
                              max(cfg.prune_floor,
                                  ps["full_acc"] - cfg.prune_tol),
                              ps["kept_per_stage"], cfg.rand_draws)

        newf = frozen_from_gates(model, gates, lo, hi)
        added = frozen_count({k: v & ~frozen[k] for k, v in newf.items()})
        for k in frozen:
            frozen[k] |= newf[k]
        masks.append([g.clone() for g in gates])
        stats_all.append(collect_stats(model, tasks[t], cfg, device, gates))

        cc = closure_check(model, tasks[t], cfg, device, frozen, gates,
                           seed * 17 + t)
        diag[t] = a_test

        # every earlier task, under ITS OWN mask. this is where BWT is read
        for s in range(t + 1):
            aware[t, s] = acc_task(model, tasks[s], cfg, device, "test",
                                   masks[s])
        done = [(masks[s], stats_all[s], *tasks[s]["classes"])
                for s in range(t + 1)]
        for s in range(t + 1):
            x, y = tasks[s]["test"]
            S = task_scores(model, x, None, done)
            L = routing_ladder(S, y, s, device)
            for r in LADDER_ROWS:
                ladder[r]["task"][t, s] = L[r]["task_acc"]
                ladder[r]["class"][t, s] = L[r]["class_acc"]
            route[t, s] = L["z"]["task_acc"]        # MEASURED, not inferred
            agn[t, s] = L["z"]["class_acc"]
            agn_cat[t, s] = L["no_router"]["class_acc"]
            agn_or[t, s] = L["oracle"]["class_acc"]

        tr.pop("scorer", None)
        row = {"task": t, **tr, **ps, **mr,
               "acc_test_circuit": a_test, "acc_random_same_size": a_rand,
               "circuit_weights": circuit_weights(model, gates, lo, hi),
               "new_frozen_weights": added,
               "frozen_frac": frozen_frac(frozen),
               "closure": cc, "elapsed_sec": time.time() - t0}
        per_task.append(row)
        flag = "" if cc["features_closed"] else "  <== NOT CLOSED"
        print(f"    task {t}: val {ps['full_acc']:.4f} -> "
              f"{ps['acc_val_circuit']:.4f} | test {a_test:.4f} on "
              f"{ps['n_channels']} ch {ps['kept_per_stage']} "
              f"= {row['circuit_weights']:,} w (+{added:,} new) | random same "
              f"size {a_rand:.4f} | closure {cc['feature_delta']:.2e}{flag} | "
              f"frozen {row['frozen_frac']:.3f} ({row['elapsed_sec']:.0f}s)",
              flush=True)
        print(f"             ACC so far {np.nanmean(aware[t, :t+1]):.4f}  "
              f"BWT {np.nanmean([aware[t, s] - diag[s] for s in range(t)]) if t else 0.0:+.4f}",
              flush=True)
        print(f"             classIL routed {np.nanmean(agn[t, :t+1]):.4f}  "
              f"| no router {np.nanmean(agn_cat[t, :t+1]):.4f}  "
              f"| oracle router {np.nanmean(agn_or[t, :t+1]):.4f}  "
              f"| task-id acc {np.nanmean(route[t, :t+1]):.4f} "
              f"(chance {1.0/(t+1):.3f})", flush=True)
        print(f"             ROUTING LADDER after task {t}  "
              f"(chance {1.0/(t+1):.3f})", flush=True)
        print(f"             {'row':>13} {'task-id':>8} {'classIL':>8}",
              flush=True)
        for r in LADDER_ROWS:
            print(f"             {r:>13} "
                  f"{np.nanmean(ladder[r]['task'][t, :t+1]):>8.4f} "
                  f"{np.nanmean(ladder[r]['class'][t, :t+1]):>8.4f}",
                  flush=True)
        if str(mr.get("control", "")).startswith("no random"):
            print(f"             size control: no random circuit of "
                  f"{ps['n_channels']} channels or fewer met the gate", flush=True)
        if mr.get("control") == "measured":
            print(f"             size control: smallest random circuit meeting "
                  f"the gate {mr['min_random_channels']} ch vs ours "
                  f"{ps['n_channels']} -> compression "
                  f"{mr['compression']:.2f}x", flush=True)

    acc = float(np.nanmean(aware[T - 1, :]))
    bwt = (float(np.nanmean([aware[T - 1, s] - diag[s] for s in range(T - 1)]))
           if T > 1 else 0.0)
    a_last = float(np.nanmean(agn[T - 1, :]))
    a_avg = float(np.mean([np.nanmean(agn[t, :t + 1]) for t in range(T)]))
    return {"seed": seed, "ACC": acc, "BWT": bwt,
            "classIL_last": a_last, "classIL_avg": a_avg,
            "classIL_last_norouter": float(np.nanmean(agn_cat[T - 1, :])),
            "classIL_last_oracle": float(np.nanmean(agn_or[T - 1, :])),
            "classIL_avg_norouter": float(np.mean(
                [np.nanmean(agn_cat[t, :t + 1]) for t in range(T)])),
            "taskid_acc_last": float(np.nanmean(route[T - 1, :])),
            "taskid_matrix": route.tolist(),
            "ladder": {r: {"task_last": float(np.nanmean(ladder[r]["task"][T-1, :])),
                           "class_last": float(np.nanmean(ladder[r]["class"][T-1, :])),
                           "class_avg": float(np.mean(
                               [np.nanmean(ladder[r]["class"][t, :t+1])
                                for t in range(T)])),
                           "task_matrix": ladder[r]["task"].tolist(),
                           "class_matrix": ladder[r]["class"].tolist()}
                       for r in LADDER_ROWS},
            "norouter_matrix": agn_cat.tolist(),
            "oracle_matrix": agn_or.tolist(),
            "aware_matrix": aware.tolist(), "agnostic_matrix": agn.tolist(),
            "diag": diag.tolist(), "per_task": per_task,
            "all_closed": bool(all(p["closure"]["features_closed"]
                                   for p in per_task)),
            "final_frozen_frac": per_task[-1]["frozen_frac"],
            "total_circuit_weights": int(sum(p["circuit_weights"]
                                             for p in per_task)),
            # per class: a k x k precision, a k-vector mean, and the two
            # scalars that standardise that class's distances
            "router_floats": int(sum(
                len(st["classes"]) * (st["k"] * st["k"] + st["k"] + 2)
                for st in stats_all)),
            **{f"cfg_{k}": v for k, v in asdict(cfg).items()}}


# =============================================================================
# self tests
# =============================================================================

def _fixture(device, n_tasks=2, cpt=3, n=96):
    g = torch.Generator().manual_seed(0)
    base = torch.randn(n_tasks * cpt, 3, 32, 32, generator=g) * 1.5
    tasks = []
    for t in range(n_tasks):
        lo = t * cpt

        def mk(m):
            y = torch.randint(0, cpt, (m,), generator=g)
            x = base[lo + y] + torch.randn(m, 3, 32, 32, generator=g) * 0.4
            return x.to(device), (y + lo).to(device)
        tasks.append({"train": mk(n), "val": mk(48), "test": mk(48),
                      "classes": (lo, lo + cpt)})
    return tasks


def self_test():
    dev = torch.device("cpu")
    cfg = Cfg(n_tasks=2, cpt=3, epochs=1, batch_size=24, val_per_task=48,
              aug=0, prune_tol=0.05, prune_floor=0.0, stages="4,6,8,10",
              head_refit_steps=20, rand_draws=2, score_order="outw")
    # outw here because several tests call prune_circuit standalone, with no
    # trained scorer to hand. The gradient scores are exercised by test 11.
    nc = n_classes_of(cfg)
    tasks = _fixture(dev, cfg.n_tasks, cfg.cpt)

    # 1. the mask really removes a channel, and the un-masked control shows the
    #    test is not vacuous
    m = MaskedResNet18(nc, 0, dev, stages=cfg.stages_tuple())
    m.eval()
    x = tasks[0]["val"][0]
    ga, go = all_on(m, dev), all_on(m, dev)
    go[3][2] = 0.0
    with torch.no_grad():
        a = m.features(x, go).clone()
        m.stages[3][1].conv2.weight[2].add_(5.0)
        b = m.features(x, go)
        c = m.features(x, ga)
    assert torch.equal(a, b), "a masked-off channel still reached the output"
    assert not torch.equal(a, c), "the mask did nothing; the test is vacuous"
    print("[1] channel masking removes a channel, and the un-masked control "
          "confirms the mask is real")

    # 2. BatchNorm has nothing per channel to freeze
    n_bn = sum(1 for n_, _ in m.named_parameters() if ".bn" in n_ or n_.startswith("bn"))
    assert n_bn == 0, f"{n_bn} BatchNorm parameters exist; WSN has none"
    n_buf = sum(1 for n_, b in m.named_buffers() if b.dtype.is_floating_point)
    assert n_buf == 0, f"{n_buf} float BN buffers exist; running stats are off"
    print("[2] no BatchNorm affine parameters and no running statistics, so "
          "nothing per channel to freeze and nothing to go stale")

    # 3. the freezing rule frees the part of a claimed row that the mask does
    #    not read. This is the whole difference from EXP17.
    g3 = all_on(m, dev)
    g3[2][:4] = 0.0
    g3[3][:5] = 0.0
    fr = frozen_from_gates(m, g3, 0, 3)
    w = fr["stages.3.0.conv1.weight"]          # reads stage 2, outputs stage 3
    live_out = int(g3[3].sum())
    live_in = int(g3[2].sum())
    got = int(w.sum())
    want = live_out * live_in * w.shape[2] * w.shape[3]
    assert got == want, f"froze {got}, expected {want}"
    e17_would = w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3]
    e17_rows = live_out * w.shape[1] * w.shape[2] * w.shape[3]
    print(f"[3] on one conv, we freeze {got:,} weights where EXP17's rule "
          f"would freeze {e17_rows:,} (whole rows). {e17_rows - got:,} stay "
          f"trainable for later tasks; EXP17 pinned them at zero forever")

    # 4. gradient masking actually holds the frozen weights still
    m4 = MaskedResNet18(nc, 1, dev, stages=cfg.stages_tuple())
    fz = {n_: torch.zeros_like(p, dtype=torch.bool) for n_, p in m4.named_parameters()}
    g4, _ = prune_circuit(m4, tasks[0], cfg, dev, 1, verbose=False)
    for k_, v in frozen_from_gates(m4, g4, 0, 3).items():
        fz[k_] |= v
    before = {n_: p.detach().clone() for n_, p in m4.named_parameters()}
    train_task(m4, tasks, 1, cfg, dev, 2, fz)
    worst = max(float((p.detach()[fz[n_]] - before[n_][fz[n_]]).abs().max())
                for n_, p in m4.named_parameters() if bool(fz[n_].any()))
    assert worst == 0.0, f"frozen weights moved by {worst}"
    moved = max(float((p.detach()[~fz[n_]] - before[n_][~fz[n_]]).abs().max())
                for n_, p in m4.named_parameters() if bool((~fz[n_]).any()))
    assert moved > 0.0, "nothing moved at all; the test is vacuous"
    print(f"[4] a second task moved the frozen weights by {worst:.1e} and the "
          f"free ones by {moved:.1e}")

    # 5. THE GUARANTEE. randomise every non-frozen weight, features unchanged.
    cc = closure_check(m4, tasks[0], cfg, dev, fz, g4, 5)
    assert cc["features_closed"], f"the masked circuit is not closed: {cc}"
    print(f"[5] under its own mask, randomising every weight not frozen for "
          f"task 0 moves its features by {cc['feature_delta']:.1e}")

    # 6. the negative control for 5. with nothing frozen it MUST fail.
    empty = {n_: torch.zeros_like(p, dtype=torch.bool)
             for n_, p in m4.named_parameters()}
    cc6 = closure_check(m4, tasks[0], cfg, dev, empty, g4, 5)
    assert not cc6["features_closed"], \
        "with nothing frozen the circuit still came out closed; test 5 proves nothing"
    print(f"[6] with nothing frozen the same randomisation moves them by "
          f"{cc6['feature_delta']:.2e}. Test 5 is not vacuous.")

    # 7. task 0 is bitwise unaffected by learning task 1
    m4.eval()
    with torch.no_grad():
        f_now = m4.features(tasks[0]["test"][0], g4)[:, g4[-1].bool()]
    before2 = {n_: p.detach().clone() for n_, p in m4.named_parameters()}
    train_task(m4, tasks, 1, cfg, dev, 9, fz)
    with torch.no_grad():
        f_after = m4.features(tasks[0]["test"][0], g4)[:, g4[-1].bool()]
    assert torch.equal(f_now, f_after), "task 0's features moved"
    assert any(not torch.equal(p.detach(), before2[n_])
               for n_, p in m4.named_parameters()), "the network did not train"
    print("[7] a further task trained on top; task 0's features under its mask "
          "are bitwise identical while other weights did move")

    # 8. head refit only touches this task's rows
    m8 = MaskedResNet18(nc, 3, dev, stages=cfg.stages_tuple())
    snap = m8.head.weight.detach().clone()
    g8, _ = prune_circuit(m8, tasks[1], cfg, dev, 3, verbose=False)
    head_refit(m8, tasks[1], cfg, dev, g8, 20, 3)
    lo8, hi8 = tasks[1]["classes"]
    other = torch.ones(nc, dtype=torch.bool)
    other[lo8:hi8] = False
    assert torch.equal(m8.head.weight[other], snap[other]), \
        "the refit touched another task's head rows"
    assert not torch.equal(m8.head.weight[lo8:hi8], snap[lo8:hi8]), \
        "the refit changed nothing"
    print("[8] the head refit changed this task's rows and left every other "
          "task's rows bitwise unchanged")

    # 9. the activation cache is result-preserving
    with torch.no_grad():
        worst9 = 0.0
        for si in range(m8.n_stages):
            ref = m8(x, g8)
            got = m8.head_from(m8.stage_input(x, g8, si), g8, si)
            assert torch.equal(ref, got), f"cache differs at stage {si}"
            worst9 = max(worst9, float((ref - got).abs().max()))
    print(f"[9] the cached forward is bitwise identical to the full one at "
          f"every stage ({worst9:.1e})")

    # 10. the router's z-standardisation is what makes unequal circuits
    #     comparable: raw distance scales with the number of dimensions
    k1, k2 = 8, 64
    gg = torch.Generator().manual_seed(0)
    d1 = torch.randn(4096, k1, generator=gg).norm(dim=1).mean()
    d2 = torch.randn(4096, k2, generator=gg).norm(dim=1).mean()
    ratio = float(d2 / d1)
    assert 2.4 < ratio < 3.2, ratio
    print(f"[10] a Mahalanobis distance in {k2} dims is {ratio:.2f}x one in "
          f"{k1} dims for the SAME distribution (sqrt ratio "
          f"{np.sqrt(k2/k1):.2f}). Raw scores cannot be compared across "
          f"circuits of different size; that is what the z step fixes.")

    # 11. the gradient scorer sees the gradient ARRIVING at each channel, so a
    #     channel frozen by an earlier task still gets a score. A scorer built
    #     on WEIGHT gradients would report zero for it and mark it useless.
    m11 = MaskedResNet18(nc, 7, dev, stages=cfg.stages_tuple())
    fz11 = {n_: torch.zeros_like(p, dtype=torch.bool)
            for n_, p in m11.named_parameters()}
    g11, _ = prune_circuit(m11, tasks[0], cfg, dev, 7, verbose=False)
    for k_, v in frozen_from_gates(m11, g11, 0, 3).items():
        fz11[k_] |= v
    c11 = Cfg(**{**asdict(cfg), "score_order": "taylor"})
    tr11 = train_task(m11, tasks, 1, c11, dev, 8, fz11)
    sc11 = tr11["scorer"]
    assert sc11 is not None, "the scorer was not attached"
    frozen_ch = g11[-1].bool()
    tay = sc11.get("taylor")[-1]
    assert float(tay[frozen_ch].max()) > 0.0, \
        "a frozen channel scored exactly zero; the scorer is reading WEIGHT " \
        "gradients, not the gradient arriving at the activation"
    for key in ("signed", "absg", "taylor", "sq"):
        v = sc11.get(key)
        assert len(v) == m11.n_stages and all(t_.numel() == c_ for t_, c_
                                              in zip(v, m11.STAGES))
    print(f"[11] the gradient scorer runs on activations, so channels frozen "
          f"by an earlier task still get a score "
          f"(max taylor on frozen channels {float(tay[frozen_ch].max()):.2e}, "
          f"on free ones {float(tay[~frozen_ch].max()):.2e})")

    print("\nall self tests passed")


# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path, default=Path("runs/exp18"))
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    for f, v in asdict(Cfg()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    cfg = Cfg(**{f: getattr(args, f) for f in asdict(Cfg())})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible; pass --allow-cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device {device}  ResNet-18  {cfg.n_tasks} tasks x {cfg.cpt} "
          f"classes  masked (no seal, no reinit)", flush=True)
    tasks, perm = E17.prepare_data(args.data_dir, cfg.n_tasks, cfg.cpt,
                                   args.split_seed, cfg.val_per_task, device,
                                   cfg.base_classes)
    (args.out / "config.json").write_text(
        json.dumps({"class_permutation": perm, **asdict(cfg)}, indent=2))
    path = args.out / "exp18_results.jsonl"
    recs = []
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        print(f"\n=== seed {seed} ===", flush=True)
        rec = run_cell(tasks, cfg, device, seed, args.out)
        with path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        recs.append(rec)
    print("\n" + "=" * 78)
    for k in ("ACC", "BWT", "classIL_last", "classIL_last_norouter",
              "classIL_last_oracle", "taskid_acc_last", "classIL_avg",
              "classIL_avg_norouter"):
        v = [r[k] for r in recs]
        print(f"  {k:>14}  {np.mean(v):+.4f} +/- {np.std(v):.4f}")
    print(f"  {'frozen frac':>14}  {np.mean([r['final_frozen_frac'] for r in recs]):.4f}")
    print(f"  {'all closed':>14}  {all(r['all_closed'] for r in recs)}")
    print("\n  ROUTING LADDER, mean over seeds, after the final task")
    print(f"  {'row':>13} {'task-id':>9} {'classIL last':>13} "
          f"{'classIL avg':>12}")
    for r in LADDER_ROWS:
        ta = np.mean([x["ladder"][r]["task_last"] for x in recs])
        cl = np.mean([x["ladder"][r]["class_last"] for x in recs])
        ca = np.mean([x["ladder"][r]["class_avg"] for x in recs])
        print(f"  {r:>13} {ta:>9.4f} {cl:>13.4f} {ca:>12.4f}")
    print(f"  chance task-id is {1.0 / recs[0]['cfg_n_tasks']:.4f}")
    print("\n  Compare ACC and BWT against IBM 88.15 / 0 and WSN 86.47 / 0 on "
          "CIFAR-100, 10 tasks, ResNet-18.")
    print("  Compare classIL_last against DPCR ~49 and classIL_avg against "
          "DPCR 62.8, same split, exemplar free.")


if __name__ == "__main__":
    main()
