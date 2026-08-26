# Executive summary, night shift 2026-08-27

## 1. Verdict

No. We did not beat the state of the art tonight, and we did not run a single
GPU experiment. This machine has no GPU and no Kaggle credentials, so Kaggle
could not be started, watched, or read from this session. Everything below
is code that is tested on CPU and ready to run unattended, plus a literature
review that changes what should be run first. Every number quoted from our
own method is the handover's single seed (21.04 / 44.65 last / average
incremental, n=1) and it was measured with an evaluation leak described in
section 6.

## 2. The table

Nothing was run in the harness on a GPU. Published numbers in our protocol
(cold start, 10x10, ResNet-18 from scratch, exemplar free, last / avg),
all marked as quoted, all VERIFIED against the paper's own table by the
literature agent unless noted:

    DPCR     (ICML 2025)        50.24 / 63.21   quoted, DPCR Table 1
    BiCyc    (ICLR 2026)        50.6  / 63.2    quoted, arXiv 2606.05675
    GATF     (2026)             52.0  / 64.4    quoted, arXiv 2606.25347
    EFC++                       47.52 / 61.57   quoted, EFC++ Table 2
    AdaGauss (NeurIPS 2024)     46.1  / 60.2    quoted
    LDC      (ECCV 2024)        45.4  / 59.5    quoted
    EFC      (ICLR 2024)        43.62 / 58.58   quoted
    FeTrIL                      34.94 / 51.20   quoted, EFC Table 1
    FeCAM                       32.4  / 48.3    quoted, AdaGauss Table 1
                                (37.63 / 52.53 in EFC++ Table 2)
    Ours, exp18 run 1           21.04 / 44.65   handover, n=1, leaky eval
    Ours, oracle task id        0.75 to 0.80    handover, n=1, leaky eval

The bar is 50 to 52 last, not 49.5, and the same method moves 5 to 10 points
between papers (LwF 32.8 versus 42.6, EFC++ 47.5 versus 52.7). A win needs
to clear about 55 on 3 seeds to be safe from protocol noise.

What is ready to produce our own rows: `cil_harness.py` with `finetune`,
`fecam`, `wsn`, `supsup`, `ours` (causal, magnitude, learned, random
selection) and `ours --rot-extra`, all on one data path, one class order
(split seed 1234), one evaluator, with checkpoint and resume tested for
every method. `kaggle_runner.py --plan session1` runs FeCAM x3, ours x3, WSN
x1 in one 11 hour session and prints the table.

## 3. What we changed and why

Fixes to the method's evaluation, not to the method:

- `no_router` (P6) is fixed. It used our own `-z` scores, so it was the `z`
  row under another name. It now uses concatenated head logits. On the toy
  fixture the two rows differ for the first time.
- Evaluation leak. exp18 uses BatchNorm with `track_running_stats=False`,
  so the network normalises with the test batch, and exp18 evaluated each
  task's test set in its own batches of 512. The batch statistics then tell
  every circuit which task the batch came from. The harness evaluates on
  shuffled batches of every seen task (`--eval-mixed 1`). Both the oracle
  row and the router will move; the old behaviour is kept behind
  `--eval-mixed 0` so the size of the leak can be measured. Until it is,
  the handover's 0.75 to 0.80 oracle ceiling is not trustworthy.
- The missing `exp17_resnet_circuits.py` (data loading) was never checked
  in, so exp18 could not run. `cil_data.py` replaces it.

Pre-built hypotheses, each stated with its prediction in
`claude/night_log.md` before any number exists:

- H_calib (`routing_lab.py --calib val`). exp18 standardises each class's
  Mahalanobis distance with the mean and spread of that class's TRAIN
  distances after 300 epochs. Train features are tighter than test
  features, so every test image is far from every class, and the winning
  task is the one whose circuit is least overfit rather than the right one.
  Using the held out slice for the standardisation is a one line change.
  The lab prints per task AUROC and the mean own-test z under both, so this
  is decided by the first checkpoint.
- H_cross (`routing_lab.py` rows chain, pairwise, allspace). A circuit
  cannot say "not mine" because it has no model of anything else. Because
  circuits are closed, later tasks' class Gaussians can be fitted inside an
  earlier circuit's feature space at any time, exemplar free, and the
  decision "task m or something later" becomes a comparison between two
  explicit models in one fixed space. Storage is printed per covariance
  model; shared and diagonal are affordable.
- H_rot (`cil_harness.py --method ours --rot-extra 1`). The literature's
  largest measured effect on task detection: rotation as extra classes
  (CLOM Table 2, 59.5 to 66.8 task detection; Table 4, 50.2 to 60.3 CIL).
  Each task gets a joint class x rotation head next to the class head,
  frozen and closure checked under the same rule. Routing row `rot` is the
  CSI ensemble score per circuit.

## 4. The causal ablation result

Not run. The code is in place: on every task the causal sweep runs, then
magnitude (`outw` top k), learned (sigmoid channel gates with an L1 push,
weights frozen) and random masks are built with the same per stage sizes
on the same trained weights, the head is refit on each, and val and test
task-IL are logged (`per_task[t]["ablation"]`). `--select magnitude` commits
a whole run to the magnitude mask for the full run comparison (session2).
On the synthetic toy the rows are within noise of each other, which is what
a 2 epoch run on noise should give and proves nothing.

The literature review found no paper that selects circuits by ablation in
continual learning and none that compares selection rules at matched
sparsity. It also found that our oracle CIL, if it survives the mixed batch
evaluation, is above every exemplar free number published, which means
selection is not what limits us. The ablation is a contribution claim, not
a fix.

## 5. Failed hypotheses

None tested on real data. Two things I believed at the start and retract:

- I assumed the brief's Kaggle path was usable from here. It is not.
- I assumed the handover's per task evaluation was clean. It has the BN
  batch leak above. The 0.75 to 0.80 oracle and the 3.3x routing constant
  were both measured with it.

## 6. Threats to the result

- n=0 GPU runs. Everything is an untested prediction until session1 runs.
- The BN leak means every exp18 number in the handover is optimistic by an
  unknown amount, oracle included.
- Baselines: FeCAM here trains task 0 with SGD 0.1 cosine for 300 epochs and
  standard BN; WSN and SupSup use Adam 1e-3 fixed and stats free BN as exp18
  does. WSN and SupSup are reimplementations, not ports of their repos, and
  WSN has no class-IL rule of its own so it is reported under our router
  and under concatenated logits. If FeCAM lands far from 32 to 38 the
  harness is wrong, per the brief.
- H_cross fits later tasks' statistics in earlier spaces. It is exemplar
  free and legal at task time, but it multiplies stored statistics; the lab
  prints the float count and the full covariance variant is not affordable.
- H_rot changes the training recipe (2x compute at N=1). A win with it is a
  fair comparison against DPCR's line only with the cost stated; against
  CLOM it is still a lighter recipe than theirs.
- `train_sub`, small `stages`, and reduced epochs exist for CPU testing and
  are logged in every config; the plans never use them.

## 7. Cost

GPU hours: 0. CPU: about 2 hours of self tests and toy runs. Quota untouched.

Estimated, unmeasured: session1 6 to 8 h, session2 about 10 h, session3
about 8 h. The runner prints per task time so the first task of session1
gives the real figure.

## 8. What to run next, in order

1. `kaggle_runner.py --plan smoke` on a P100 (minutes). Confirms the path,
   the accelerator, and the per task time.
2. `--plan session1`. FeCAM x3 is the harness check. ours x3 gives the first
   clean, leak free, 3 seed number and the checkpoints. The lab runs on each
   checkpoint automatically and decides H_calib and H_cross in minutes.
3. If H_calib or H_cross moves task-id by more than a few points, put the
   winner into `Ours.evaluate` and re-score the same checkpoints; no
   retraining is needed.
4. `--plan session3` (H_rot) before session2 if the lab did not close the
   gap. The review says training, not scoring, is three quarters of the
   published fix.
5. `--plan session2` for WSN x2, SupSup x3, magnitude selection x3 and the
   finetune floor. This completes the table and the ablation.
6. Only then: confirmatory seeds on anything chosen after looking at the
   lab output, on fresh seeds 3, 4, 5.
