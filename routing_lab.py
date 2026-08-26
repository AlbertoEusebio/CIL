#!/usr/bin/env python3
"""
Routing lab: try task-prediction rules on a finished `ours` checkpoint
without retraining anything.

    python routing_lab.py --ckpt ckpt/ours_seed0.pt --data-dir data [--all-steps]

Why this is legal. Every circuit is closed (closure_check, bitwise), so the
features of ANY image under mask m are identical today to what they were the
moment task m ended. Fitting task u's class statistics under mask m (u > m)
here is the same computation task u could have run when its data was
present. Nothing below uses stored images, test data, or a later mask on
earlier data.

The hypothesis under test (stated before any number was seen):
    H_cross. Our router fails because circuit m has no model of what is NOT
    task m. Fitting later tasks' class Gaussians inside circuit m's own
    feature space gives it one. A decision made inside one fixed feature
    space, between explicit models of both sides, should beat comparing
    z-standardised scores across unrelated spaces.
    Prediction: task-id accuracy rises well above the 3.3x-chance band.

Rows:
    z            reference, exp18's router, must reproduce the harness number
                 (with --calib train and --cov full)
    rmd          relative Mahalanobis: own-class distance minus the distance
                 to a background Gaussian of the task's own data
    batch50      diagnostic only: route 50 same-task test images together
    space0       every class fitted in circuit 0's space. FeCAM on circuit 0.
    chain        m = 0,1,...: in space m, is the nearest class one of task m's?
                 stop at the first yes. Explicit "mine vs everything later".
    pairwise     every pair (s < u) decided in space s; most wins picks the task
    allspace_min class score = min over spaces m <= task(c) of z_{m,c}
    allspace_mean                same with the mean
Each row is run for every covariance model in --cov (full per class, fecam
= shrinkage + correlation normalisation, shared per task, diag) and for
every calibration source in --calib (train: z mean/std from the fitting
data, as exp18 does; val: from the held-out validation slice). Per-task
AUROC of the reference score is printed as a diagnostic. Storage is printed
per model.

    H_calib. exp18 standardises each class's distance with the mean and
    std of that class's TRAINING distances. After 300 epochs the train
    features are much tighter than test features, so every test image looks
    far from every class, and which task wins depends on how overfit each
    circuit is rather than on the image. Prediction: calib=val raises
    task-id accuracy on its own.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import cil_data as D
import cil_harness as H
import exp18_masked_circuits as X


@torch.no_grad()
def feats_under(model, x, gates, batch=512):
    keep = gates[-1].bool()
    f = torch.cat([model.features(x[s:s + batch], gates)[:, keep]
                   for s in range(0, len(x), batch)])
    return f.clamp_min(0).pow(0.5)          # Tukey 0.5, as exp18


@torch.no_grad()
def fit_space(f, y, classes, cov_model, ridge=1e-2, fcal=None, ycal=None):
    """Gaussians for `classes` in one feature space. If (fcal, ycal) is given
    the per-class distance mean/std used for z-standardisation come from it
    (held-out data) instead of from the fitting data."""
    k = f.shape[1]
    eye = torch.eye(k, device=f.device)
    out = {}
    if cov_model == "shared":
        d = torch.cat([f[y == c] - f[y == c].mean(0) for c in classes])
        cov = (d.t() @ d) / max(1, len(d) - 1)
        cov = cov + (ridge * torch.diag(cov).mean() + 1e-6) * eye
        prec_shared = torch.linalg.inv(cov.double()).float()
    for c in classes:
        fc = f[y == c]
        mu = fc.mean(0)
        d = fc - mu
        if cov_model == "full":
            cov = (d.t() @ d) / max(1, len(fc) - 1)
            cov = cov + (ridge * torch.diag(cov).mean() + 1e-6) * eye
            prec = torch.linalg.inv(cov.double()).float()
            dist = ((d @ prec) * d).sum(1).clamp_min(0).sqrt()
        elif cov_model == "fecam":
            # FeCAM: shrinkage gamma1 = gamma2 = 1, then correlation
            # normalisation. Their Table 4 says this is worth 14.6 -> 62.1.
            cov = (d.t() @ d) / max(1, len(fc) - 1)
            v1 = torch.diag(cov).mean()
            v2 = (cov.sum() - torch.diag(cov).sum()) / (k * k - k)
            cov = cov + 1.0 * v1 * eye + 1.0 * v2 * (1 - eye)
            sd = torch.diag(cov).sqrt().clamp_min(1e-6)
            cov = cov / (sd[:, None] * sd[None, :]) + 1e-6 * eye
            prec = torch.linalg.inv(cov.double()).float()
            dist = ((d @ prec) * d).sum(1).clamp_min(0).sqrt()
        elif cov_model == "shared":
            prec = prec_shared
            dist = ((d @ prec) * d).sum(1).clamp_min(0).sqrt()
        else:
            var = d.pow(2).mean(0) + ridge * d.pow(2).mean() + 1e-6
            prec = 1.0 / var
            dist = ((d * d) * prec).sum(1).sqrt()
        st = {"mu": mu, "prec": prec, "m": float(dist.mean()),
              "s": float(dist.std()) + 1e-6, "diag": cov_model == "diag",
              "m_train": float(dist.mean()), "s_train": float(dist.std())}
        if fcal is not None:
            dc = dist_to(fcal[ycal == c], st)
            st["m"], st["s"] = float(dc.mean()), float(dc.std()) + 1e-6
        out[c] = st
    return out


@torch.no_grad()
def dist_to(f, st):
    d = f - st["mu"]
    if st["diag"]:
        return ((d * d) * st["prec"]).sum(1).sqrt()
    return ((d @ st["prec"]) * d).sum(1).clamp_min(0).sqrt()


@torch.no_grad()
def evaluate_step(model, tasks, masks, t, cov_model, x, y, tt, calib="train"):
    """All rows after task t, on the mixed test set (x, y, tt)."""
    T = t + 1
    spans = [tasks[u]["classes"] for u in range(T)]
    # class Gaussians of every task u >= m inside space m
    G, BG = {}, {}
    for m in range(T):
        ftr = torch.cat([feats_under(model, tasks[u]["train"][0], masks[m]) for u in range(m, T)])
        ytr = torch.cat([tasks[u]["train"][1] for u in range(m, T)])
        classes = [c for u in range(m, T) for c in range(*spans[u])]
        fcal = ycal = None
        if calib == "val":
            fcal = torch.cat([feats_under(model, tasks[u]["val"][0], masks[m]) for u in range(m, T)])
            ycal = torch.cat([tasks[u]["val"][1] for u in range(m, T)])
        G[m] = fit_space(ftr, ytr, classes, cov_model, fcal=fcal, ycal=ycal)
        # background Gaussian of task m's OWN data in its own space, for the
        # relative Mahalanobis distance (Ren et al. 2021)
        own = ftr[(ytr >= spans[m][0]) & (ytr < spans[m][1])]
        BG[m] = fit_space(own, torch.zeros(len(own), dtype=torch.long, device=own.device),
                          [0], "full" if cov_model in ("full", "fecam", "shared") else "diag")[0]
    fte = {m: feats_under(model, x, masks[m]) for m in range(T)}
    # z_{m,c}(x) for every space m and every class c known to it
    Z = {}
    for m in range(T):
        for c, st in G[m].items():
            Z[(m, c)] = (dist_to(fte[m], st) - st["m"]) / st["s"]
    task_of = {c: u for u, (lo, hi) in enumerate(spans) for c in range(lo, hi)}
    out = {}

    def rec(name, pick, cls):
        out[name] = {"task": pick, "cls": cls}

    # z (reference): each task in its own space, nearest own class
    zown = torch.stack([torch.stack([Z[(u, c)] for c in range(*spans[u])], 1).min(1).values
                        for u in range(T)], 1)
    pick = zown.argmin(1)
    # diagnostics on the reference score: per task, own-test score vs the
    # score of everything else, as AUROC, and how much the train-fitted
    # standardisation misreads held-out data
    diag = {}
    for u in range(T):
        sc = -zown[:, u]
        pos, neg = sc[tt == u], sc[tt != u]
        if len(pos) and len(neg):
            auroc = float((pos[:, None] > neg[None, :]).float().mean()
                          + 0.5 * (pos[:, None] == neg[None, :]).float().mean())
        else:
            auroc = float("nan")
        own_z = zown[tt == u, u]
        diag[u] = {"auroc": auroc, "own_test_z_mean": float(own_z.mean()),
                   "own_test_z_std": float(own_z.std())}

    def classify_in_own(pick):
        cls = torch.zeros(len(y), dtype=torch.long, device=y.device)
        for u in range(T):
            msk = pick == u
            if bool(msk.any()):
                cols = torch.stack([Z[(u, c)][msk] for c in range(*spans[u])], 1)
                cls[msk] = cols.argmin(1) + spans[u][0]
        return cls
    rec("z", pick, classify_in_own(pick))

    # rmd: relative Mahalanobis, subtract the distance to task u's background
    # Gaussian in space u. Removes the part of the distance that is just
    # "far from everything this circuit has seen" (Ren et al. 2021).
    rown = torch.stack([torch.stack([dist_to(fte[u], G[u][c]) for c in range(*spans[u])], 1).min(1).values
                        - dist_to(fte[u], BG[u]) for u in range(T)], 1)
    pick = rown.argmin(1)
    rec("rmd", pick, classify_in_own(pick))

    # batch50: the PR-Ent diagnostic. Route by the sum of z over 50 test
    # images of the same task. Not admissible as a method; tells us whether
    # per-sample scores carry signal at all.
    pick = torch.zeros(len(y), dtype=torch.long, device=y.device)
    for u in range(T):
        idx = (tt == u).nonzero(as_tuple=True)[0]
        for b0 in range(0, len(idx), 50):
            ii = idx[b0:b0 + 50]
            pick[ii] = zown[ii].mean(0).argmin()
    rec("batch50", pick, classify_in_own(pick))

    # space0: everything in circuit 0's space
    all_cls = list(range(spans[-1][1]))
    z0 = torch.stack([Z[(0, c)] for c in all_cls], 1)
    c0 = z0.argmin(1)
    rec("space0", torch.tensor([task_of[int(c)] for c in c0], device=y.device), c0)

    # chain
    pick = torch.full((len(y),), T - 1, dtype=torch.long, device=y.device)
    undecided = torch.ones(len(y), dtype=torch.bool, device=y.device)
    for m in range(T - 1):
        cols = [c for u in range(m, T) for c in range(*spans[u])]
        zm = torch.stack([Z[(m, c)] for c in cols], 1)
        nearest = torch.tensor(cols, device=y.device)[zm.argmin(1)]
        mine = (nearest >= spans[m][0]) & (nearest < spans[m][1]) & undecided
        pick[mine] = m
        undecided &= ~mine
    rec("chain", pick, classify_in_own(pick))

    # pairwise, decided in the earlier task's space
    wins = torch.zeros(len(y), T, device=y.device)
    for s_ in range(T):
        for u in range(s_ + 1, T):
            a = torch.stack([Z[(s_, c)] for c in range(*spans[s_])], 1).min(1).values
            b = torch.stack([Z[(s_, c)] for c in range(*spans[u])], 1).min(1).values
            wins[:, s_] += (a <= b).float()
            wins[:, u] += (a > b).float()
    pick = wins.argmax(1)
    rec("pairwise", pick, classify_in_own(pick))

    # allspace: class c scored across every space that knows it
    for agg in ("min", "mean"):
        cols = []
        for c in all_cls:
            zs = torch.stack([Z[(m, c)] for m in range(task_of[c] + 1)], 1)
            cols.append(zs.min(1).values if agg == "min" else zs.mean(1))
        zc = torch.stack(cols, 1)
        cc = zc.argmin(1)
        rec(f"allspace_{agg}", torch.tensor([task_of[int(c)] for c in cc], device=y.device), cc)

    res = {}
    for name, v in out.items():
        res[name] = {"task_acc": float((v["task"] == tt).float().mean()),
                     "class_acc": float((v["cls"] == y).float().mean())}
    res["_diag"] = diag
    # storage for this cov model, in floats
    k = {m: int(masks[m][-1].sum()) for m in range(T)}
    if cov_model in ("full", "fecam"):
        store = sum(len(G[m]) * (k[m] * k[m] + k[m] + 2) for m in range(T))
    elif cov_model == "shared":
        store = sum((T - m) * k[m] * k[m] + len(G[m]) * (k[m] + 2) for m in range(T))
    else:
        store = sum(len(G[m]) * (2 * k[m] + 2) for m in range(T))
    return res, store


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all-steps", action="store_true",
                    help="evaluate after every task (average incremental)")
    ap.add_argument("--cov", type=str, default="full,fecam,shared,diag")
    ap.add_argument("--calib", type=str, default="train,val",
                    help="where the z-standardisation mean/std come from")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible; pass --allow-cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = H.HCfg(**ck["cfg"])
    tasks, perm = D.prepare_data(args.data_dir, cfg.n_tasks, cfg.cpt,
                                 cfg.split_seed, cfg.val_per_task, device)
    if cfg.train_sub:
        D.subsample_train(tasks, cfg.train_sub)
    meth = H.Ours(cfg, tasks, device, ck.get("seed", 0))
    meth.load(ck["method"])
    model, masks = meth.model, meth.masks
    model.eval()
    T_done = ck["next_task"]
    steps = range(T_done) if args.all_steps else [T_done - 1]
    print(f"checkpoint {args.ckpt}: {T_done} tasks done, select={cfg.select}, "
          f"eval on mixed test batches", flush=True)
    results = {}
    for cov in args.cov.split(","):
      for calib in args.calib.split(","):
        key = f"{cov}/{calib}"
        per_step = []
        for t in steps:
            x, y, tt = H.mixed_test(tasks, t)
            t0 = time.time()
            res, store = evaluate_step(model, tasks, masks, t, cov, x, y, tt, calib)
            per_step.append(res)
            print(f"  {key:>12} after task {t} ({time.time()-t0:.0f}s, "
                  f"{store/1e6:.2f}M floats):  " +
                  "  ".join(f"{r}:{v['task_acc']:.3f}/{v['class_acc']:.3f}"
                            for r, v in res.items() if not r.startswith("_")), flush=True)
            d = res["_diag"]
            print(f"  {'':>12} z-score diagnostics per task: AUROC " +
                  " ".join(f"{d[u]['auroc']:.2f}" for u in d) +
                  " | own-test z mean " +
                  " ".join(f"{d[u]['own_test_z_mean']:+.1f}" for u in d), flush=True)
        rows = [r for r in per_step[-1] if not r.startswith("_")]
        summary = {r: {"task_last": per_step[-1][r]["task_acc"],
                       "class_last": per_step[-1][r]["class_acc"],
                       "class_avg": float(np.mean([p[r]["class_acc"] for p in per_step]))}
                   for r in rows}
        results[key] = {"summary": summary, "steps": per_step, "store_floats": store}
        print(f"\n  {key}  {'row':>14} {'task-id':>8} {'classIL last':>13} {'classIL avg':>12}")
        for r, v in summary.items():
            print(f"  {'':>10}  {r:>14} {v['task_last']:>8.4f} {v['class_last']:>13.4f} "
                  f"{v['class_avg']:>12.4f}")
        print(f"  chance task-id {1.0/T_done:.4f}; harness z row for this ckpt: "
              f"{np.nanmean(np.array(ck['cil'])[T_done-1, :T_done]):.4f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"ckpt": str(args.ckpt), "results": results,
                                        "cfg": asdict(cfg)}, indent=1))


if __name__ == "__main__":
    main()
