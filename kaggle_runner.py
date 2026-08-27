#!/usr/bin/env python3
"""
Drives one Kaggle session end to end. Called from the notebook:

    python kaggle_runner.py --plan session1 --session-hours 11

What it does, in order:
  1. prints the accelerator and records it in runs/environment.json
  2. locates CIFAR-100 (attached dataset under /kaggle/input, else download)
  3. collects every checkpoint directory from earlier sessions that was
     re-attached as an input dataset (any /kaggle/input/*/ckpt or *.pt)
  4. runs both self-test suites; refuses to continue if either fails
  5. runs the plan's commands one by one, each with a deadline so the last
     one stops cleanly between tasks before Kaggle kills the session
  6. prints a table of every result in runs/

Plans are lists of harness invocations in priority order. A finished seed
is skipped by the harness itself (it finds its checkpoint), so re-running
the same plan in a new session with the old output attached continues
exactly where it stopped. Nothing here shortens any baseline's training.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("CIL_WORK", "/kaggle/working"))
PY = sys.executable

COMMON = "--epochs 300 --batch-size 256"
# ours / wsn / supsup: Adam 1e-3 fixed, as exp18 (IBM's recipe). fecam and
# finetune: SGD 0.1 cosine, wd 5e-4, the standard CIFAR recipe.
OURS = "--method ours --opt adam --lr 1e-3 --weight-decay 0"
WSN = "--method wsn --opt adam --lr 1e-3 --weight-decay 0"
SUP = "--method supsup --opt adam --lr 1e-3 --weight-decay 0"
FECAM = "--method fecam --opt sgd --lr 0.1 --weight-decay 5e-4 --sched cos"
FT = "--method finetune --opt sgd --lr 0.1 --weight-decay 5e-4 --sched cos"

PLANS = {
    # Phase A/B/C. FeCAM first: it is the cheapest sanity floor (one task of
    # training). Then ours on 3 seeds, then WSN.
    "session1": [
        ("harness", f"{FECAM} --seeds 0,1,2"),
        ("harness", f"{OURS} --seeds 0"),
        ("lab", "ours_seed0.pt"),
        ("harness", f"{OURS} --seeds 1"),
        ("lab", "ours_seed1.pt"),
        ("harness", f"{OURS} --seeds 2"),
        ("lab", "ours_seed2.pt"),
        ("harness", f"{WSN} --seeds 0"),
    ],
    # Phase B remainder and the full-run selection ablation.
    "session2": [
        ("harness", f"{WSN} --seeds 1,2"),
        ("harness", f"{SUP} --seeds 0,1,2"),
        ("harness", f"{OURS} --select magnitude --seeds 0,1,2"),
        ("harness", f"{FT} --seeds 0"),
    ],
    # Phase D. H_rot: rotation-as-OOD head, 2x training compute per task.
    "session3": [
        ("harness", f"{OURS} --rot-extra 1 --seeds 0"),
        ("lab", "ours_seed0.pt"),
        ("harness", f"{OURS} --rot-extra 1 --seeds 1,2"),
    ],
    # 2-task toy on the real data: tests the whole path in minutes.
    "smoke": [
        ("harness", f"{FECAM} --n-tasks 2 --epochs 2 --seeds 0"),
        ("harness", f"{OURS} --n-tasks 2 --epochs 2 --seeds 0"),
        ("lab", "ours_seed0.pt"),
        ("harness", f"{WSN} --n-tasks 2 --epochs 2 --seeds 0"),
        ("harness", f"{SUP} --n-tasks 2 --epochs 2 --seeds 0"),
    ],
}


def sh(cmd, check=True, **kw):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r


def find_cifar():
    for p in glob.glob("/kaggle/input/**/cifar-100-python", recursive=True):
        if (Path(p) / "train").exists():
            return str(Path(p).parent)
    for p in glob.glob("/kaggle/input/**/train", recursive=True):
        if (Path(p).parent / "meta").exists() and (Path(p).parent / "test").exists():
            return str(Path(p).parent)      # cil_data accepts the folder itself
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="session1", choices=sorted(PLANS))
    ap.add_argument("--session-hours", type=float, default=11.0,
                    help="Kaggle kills committed runs at ~12h; leave margin")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--extra", default="", help="appended to every harness call")
    args = ap.parse_args()
    t_start = time.time()
    deadline = t_start + args.session_hours * 3600
    runs, ckpt = WORK / "runs", WORK / "ckpt"
    runs.mkdir(parents=True, exist_ok=True)

    # 1. environment
    env = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": args.plan}
    r = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                       shell=True, capture_output=True, text=True)
    env["gpu"] = r.stdout.strip() or "none"
    print(f"accelerator: {env['gpu']}", flush=True)
    (runs / "environment.json").write_text(json.dumps(env, indent=1))

    # 2. data
    data = args.data_dir or find_cifar()
    if data is None:
        data = "/tmp/cil_data"
        print("CIFAR-100 not attached; cil_data will download it (internet must be on)")
    print(f"data dir: {data}", flush=True)

    # 3. earlier checkpoints
    resume = sorted({str(Path(p).parent) for p in
                     glob.glob("/kaggle/input/**/*_seed*.pt", recursive=True)})
    resume_flags = " ".join(f"--resume-dir {d}" for d in resume)
    print(f"resume dirs: {resume or 'none'}", flush=True)
    # carry earlier results forward so the final table is complete
    for p in glob.glob("/kaggle/input/**/runs/*_results.jsonl", recursive=True):
        dst = runs / Path(p).name
        with open(p) as fi, dst.open("a") as fo:
            fo.write(fi.read())

    # 4. self tests
    sh(f"cd {HERE} && {PY} exp18_masked_circuits.py --self-test | tail -3")
    sh(f"cd {HERE} && {PY} cil_harness.py --self-test | tail -3")

    # 5. plan
    for kind, spec in PLANS[args.plan]:
        left = deadline - time.time()
        if left < 600:
            print(f"\n{left/60:.0f} min left; not starting anything else", flush=True)
            break
        if kind == "harness":
            cmd = (f"cd {HERE} && {PY} cil_harness.py {spec} {args.extra} "
                   f"--data-dir {data} --out {runs} --ckpt-dir {ckpt} {resume_flags} "
                   f"--deadline-sec {left:.0f}")
        else:
            ck = None
            for d in [str(ckpt)] + resume:
                if (Path(d) / spec).exists():
                    ck = Path(d) / spec
                    break
            if ck is None:
                print(f"routing lab: {spec} not found, skipped", flush=True)
                continue
            cpu = "--allow-cpu" if "--allow-cpu" in args.extra else ""
            cmd = (f"cd {HERE} && {PY} routing_lab.py --ckpt {ck} --data-dir {data} {cpu} "
                   f"--all-steps --out {runs}/lab_{spec.replace('.pt', '')}.json")
        t0 = time.time()
        sh(cmd, check=False)
        print(f"  [{(time.time()-t0)/60:.1f} min, {(deadline-time.time())/3600:.2f} h left]",
              flush=True)

    # 6. table
    print("\n" + "=" * 78 + "\nRESULTS in this protocol (mean +/- std over seeds)")
    print(f"{'method':>16} {'n':>2} {'seeds':>9} {'classIL last':>13} {'classIL avg':>12} "
          f"{'taskIL last':>12}")
    import numpy as np
    for f in sorted(runs.glob("*_results.jsonl")):
        recs = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["seed"]] = r          # last record per seed wins
        recs = list(recs.values())
        if not recs:
            continue
        def ms(k):
            v = [r[k] for r in recs]
            return f"{100*np.mean(v):5.2f}+/-{100*np.std(v):4.2f}"
        print(f"{recs[0]['tag']:>16} {len(recs):>2} {str(sorted(r['seed'] for r in recs)):>9} "
              f"{ms('classIL_last'):>13} {ms('classIL_avg'):>12} {ms('taskIL_last'):>12}")
    print(f"\nsession used {(time.time()-t_start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
