#!/usr/bin/env python3
"""
Re-evaluate a finished harness checkpoint with the admissible evaluator:
single sample inference (per task stored BN statistics), mixed test
stream, no test set decision anywhere. Needed for checkpoints produced by
code that evaluated with test batch statistics (session1, version 5).

    python reeval.py --ckpt ckpt/ours_seed0.pt --data-dir data --out runs

Legal because circuits are closed: the stored BN statistics and the class
Gaussians are recomputed from each task's TRAINING data under its own mask,
which is exactly what the task would have computed at the time. The
incremental matrix is rebuilt step by step using only masks 0..t at step t.
Writes {tag}_reeval_results.jsonl next to the originals; never overwrites.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import cil_data as D
import cil_harness as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU visible; pass --allow-cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = H.HCfg(**ck["cfg"])
    T = ck["next_task"]
    tasks, perm = D.prepare_data(args.data_dir, cfg.n_tasks, cfg.cpt,
                                 cfg.split_seed, cfg.val_per_task, device)
    if cfg.train_sub:
        D.subsample_train(tasks, cfg.train_sub)
    meth = H.METHODS[cfg.method](cfg, tasks, device, ck["method"].get("seed", 0)
                                 if isinstance(ck["method"], dict) else 0)
    meth.load(ck["method"])                 # records stored BN stats if absent
    if cfg.method in ("wsn", "supsup"):
        meth.T_done = T
    cil = np.full((cfg.n_tasks, cfg.n_tasks), np.nan)
    til = np.full((cfg.n_tasks, cfg.n_tasks), np.nan)
    ladders = []
    for t in range(T):
        if cfg.method == "ours":
            keep_masks, keep_stats = meth.masks, meth.stats
            meth.masks, meth.stats = keep_masks[:t + 1], keep_stats[:t + 1]
            ev = meth.evaluate(t)
            meth.masks, meth.stats = keep_masks, keep_stats
        else:
            ev = meth.evaluate(t)
        for s in range(t + 1):
            cil[t, s], til[t, s] = ev["cil"][s], ev["til"][s]
        ladders.append(ev.get("ladder"))
        print(f"  after task {t}: classIL {np.nanmean(cil[t, :t+1]):.4f}  "
              f"taskIL {np.nanmean(til[t, :t+1]):.4f}" +
              ("  | " + "  ".join(f"{r}:{np.mean([v['class_acc'] for v in vs]):.3f}"
                                  for r, vs in ev["ladder"].items()) if ev.get("ladder") else ""),
              flush=True)
    tag = H.run_tag(cfg)
    rec = {"tag": tag + "_reeval", "seed": ck.get("seed", None), "source_ckpt": str(args.ckpt),
           "evaluator": "single-sample stored BN, mixed stream",
           **H.score_matrix(cil, til, T), "ladders": ladders,
           **{f"cfg_{k}": v for k, v in asdict(cfg).items()}}
    if ladders and ladders[-1]:
        rec["ladder_last"] = {r: float(np.mean([v["class_acc"] for v in vs]))
                              for r, vs in ladders[-1].items()}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{tag}_reeval_results.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {tag} re-evaluated: classIL_last {rec['classIL_last']:.4f}  "
          f"classIL_avg {rec['classIL_avg']:.4f}  taskIL_last {rec['taskIL_last']:.4f}")


if __name__ == "__main__":
    main()
