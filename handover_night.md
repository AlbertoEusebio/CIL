# Handover: allocation_overlap, night shift

You are picking up an ongoing research project. Read this file end to end before
touching anything. Your job tonight is to figure out how to close the gap
described in section 6. Do not assume the plan in this file is the plan. There
is deliberately no proposed solution here.

## 0. House rules, non negotiable

These come from the project owner (Alberto) and from `claude/working_agreement.md`
in the project. They are not style preferences, they are how the work is judged.

- Write in plain English. Short sentences. No em dashes. No bolded slide decks.
- Ground every claim in the implementation. Quote the variable name or the line.
- Be skeptical of every result, including your own. State n. Say what a number
  does NOT prove.
- If you were wrong, retract in plain words in the next message.
- Verify numbers against the raw files before agreeing with them.
- Anything chosen after looking at results is exploratory, not confirmatory.
  Say so.
- Failed and inconclusive results stay visible. Do not quietly drop an arm.

## 1. The research question

How does a neural network allocate its internal features across tasks it learns
one after another, and can we exploit that allocation to learn many tasks with
no forgetting at all?

The practical target: beat published results on Split CIFAR-100 in a clear,
publishable, defensible way.

## 2. The problem, in full detail

### 2.1 The setting

Continual learning, class incremental, exemplar free, from scratch.

Split CIFAR-100 means: take the 100 CIFAR-100 classes, cut them into T tasks of
100/T classes each. Show the model task 0, then task 1, and so on. When you are
on task k you can only see task k's data. You never get to see task 0's images
again. No replay buffer, no stored exemplars. The backbone is a ResNet-18 built
for 32x32 inputs (3x3 stem, no maxpool), trained from random init. No
pretraining. Standard split is T=10, so 10 tasks of 10 classes.

There are two evaluation modes and they are wildly different in difficulty.

**Task-IL (task incremental).** At test time you are told which task the image
came from. You only have to pick among that task's 10 classes. Chance is 1/10.

**Class-IL (class incremental).** At test time you are told nothing. You have to
pick among all 100 classes. Chance is 1/100. This is the benchmark that matters
and the one everyone reports.

### 2.2 Why class-IL is hard, precisely

Kim et al. (NeurIPS 2022) decompose it exactly:

    P(class j) = P(class j | task k) * P(task k)
    CIL        = WP                  * TP

WP is "within-task prediction": given the right task, pick the right class.
TP is "task prediction": pick the right task. The cross entropies are exactly
additive, and they prove that good TP is necessary, not just helpful. TP is
equivalent to per-task out-of-distribution detection: task k's module has to be
able to say "this input is not mine".

A head trained only on task k's 10 classes with softmax cannot say that. It
always outputs one of its 10 classes, confidently, for any input. This is the
core difficulty. It is not a forgetting problem. It is a scoring problem.

### 2.3 Two problems that get confused with each other

**Forgetting.** Weights move while learning task k+1, so task k's accuracy
drops. This is the classic problem. It is solvable: freeze the weights.

**Label space growth.** Even with zero weight movement, class-IL accuracy drops
as tasks accumulate, because there are more wrong answers available.

We measured this separation directly. See `demo_arms.py`, `demo_cifar100.py`,
`demo_pairs100.py` in the working directory.

Experiment: train an MLP on ALL of real MNIST at once (so the features are as
good as they will ever get), throw the head away, then refit 10 output neurons
task incrementally with per-class BCE, freezing each task's rows the moment the
task ends. Weight movement on earlier rows is bitwise zero. Result:

    arm          task-IL   class-IL   per-task class-IL
    random        0.9878    0.6422    [0.83 0.39 0.68 0.81 0.50]
    task0         0.9808    0.3705    [1.00 0.14 0.25 0.29 0.17]
    sequential    0.9967    0.5659    [0.74 0.20 0.40 0.71 0.78]
    joint         0.9983    0.8303    [0.74 0.89 0.67 0.93 0.93]
    per_task      0.9972    0.6683    [0.92 0.55 0.61 0.77 0.50]

Even the `joint` arm ("perfect features", not a legal protocol, just a ceiling)
loses 17 points of class-IL with literally zero forgetting.

Scaled to 100 classes with paired MNIST digits (`demo_pairs100.py`): class-IL
falls 0.96 to 0.47 over 10 tasks with weight movement of exactly `0.0e+00`.

**Important trap.** The standard continual learning forgetting metric F computed
on that run gives F = 0.248, which reads as "a lot of forgetting", on a run
where nothing moved. The metric is measuring label space growth. Do not report
F without saying what it contains.

### 2.4 So the real problem statement

Given that we can make forgetting exactly zero, the entire remaining gap in
class-IL is task prediction. Our own numbers confirm this (section 5). The
question is how to make each task's circuit able to reject inputs that do not
belong to it, without storing exemplars and without a heavy retraining recipe.

## 3. Our intuition and current solution

### 3.1 The intuition

When a network learns a task, only a small subset of its channels actually
carries the computation for that task. If you can find that subset causally,
you can freeze exactly it and leave everything else free for later tasks. Over
T tasks you get T circuits inside one network, each provably unaffected by
later training.

The distinctive claim versus prior work: we find the subset by **causal
ablation**, not by weight magnitude and not by a learned score. We silence a
channel, remeasure task accuracy, and keep the channel only if silencing it
hurts. Everything else in the network stays untouched.

### 3.2 How the circuit is found

Greedy backward causal ablation, in `exp18_masked_circuits.py`:

- Go stage by stage, deepest stage first (`prune_order = "deep"`).
- Within a stage, order candidate channels by a cheap score
  (`score_order`, default `taylor`, computed from the gradient ARRIVING at each
  channel, accumulated over the whole training run: see `ChannelScorer`).
- Try silencing each candidate. If validation accuracy stays within
  `prune_tol` (0.02) of the unpruned model, the channel is dropped for good and
  stays silenced while we test the next one.
- Sequential matters. One-shot scoring drops both members of a redundant pair,
  because each looks individually useless. Sequential ablation keeps one.

ResNet detail: identity skip connections force all channels in a stage to line
up, so the mask is per stage, one mask each over 64, 128, 256, 512 channels.

### 3.3 How the circuit is protected

Two versions exist. Know the difference, it is the main design change between
exp17 and exp18.

**exp17, "sealing".** For a claimed channel c, zero out and freeze every incoming
weight that comes from outside the circuit. This makes the trunk closed with no
test-time mask needed. It also consumes the network: those weights are pinned at
zero forever. exp17 ran out of free channels at task 4 and printed
`NO FREE CHANNELS LEFT [0,0,0,0]`. Sealing is dead.

**exp18, "masking", WSN style.** Freeze only `W[c, g]` where c is in this
stage's mask AND g is in the mask feeding it. Everything else stays trainable.
Apply the mask at test time. See `frozen_from_gates()` and `GradFreezer`. This
preserves capacity. This is the current design.

### 3.4 The closure proof

This is our strongest asset and it must not be broken.

After a task's mask is set, we randomise every parameter that is NOT frozen and
assert that the circuit's features are **bitwise identical**. Not close, identical.
`0.0e+00`. A negative control must FAIL this test (it does, at 5.06e-01).

This proves the circuit is closed: no later training can change it. BWT is
exactly 0.0000 by construction, not by measurement noise.

Watch out: an earlier version reported 2.3e-05 as forgetting. That was BLAS
reduction order from a growing matmul, not real movement. Compute block by block.

### 3.5 The router

At test time in class-IL we need to pick a task. Current router: z-standardised
Mahalanobis distance to per-task Gaussians fitted on the circuit's features
(`familiarity_z`). The z standardisation is needed because
`E[d_t] ~ sqrt(k_t)`, so circuits of different sizes are not comparable raw.

There is a "routing ladder" in the code that evaluates nine scoring rules on the
same trained model:

    LADDER_ROWS = ("no_router", "raw", "z", "complement", "mls", "ebo",
                   "or_gate", "rownorm_mls", "oracle")

`oracle` is class-IL with the true task id. It is the ceiling.

## 4. Current setup, files and commands

Working directory `/home/claude`. GPU available to the owner: 1x H100.

    exp18_masked_circuits.py   THE CURRENT EXPERIMENT. Everything lives here.
    exp17_resnet_circuits.py   previous version, sealing. kept for reference.
    exp16_minimal_circuits.py  the MNIST/MLP ancestor.
    demo_arms.py               perfect-features ablation, 5 trunk arms.
    demo_cifar100.py           perfect-features class-IL on Split CIFAR-100.
    demo_pairs100.py           100-class label-space-growth demo.
    exp18_explained.md         plain English walkthrough of exp18.
    split_cifar100_positioning.md   the literature table. ALSO IN THE PROJECT.
    positioning.md             wider literature notes.

Project docs (read with the Projects tool, not from disk):

    claude/split_cifar100_positioning.md   <- read this one first
    claude/exp18_explained.md
    claude/exp17_change_plan.md
    claude/lit_positioning_and_next_steps.md
    claude/working_agreement.md

Run it:

    python exp18_masked_circuits.py --self-test          # 11 tests, all pass
    python exp18_masked_circuits.py --data-dir data --out runs/e18 \
        --n-tasks 10 --cpt 10 --epochs 300 --score-order taylor --seeds 0

Config fields become CLI flags automatically (see the loop over `Cfg` fields at
line ~1267). Key ones: `n_tasks`, `cpt`, `epochs` (300, matching IBM and WSN),
`prune_tol` (0.02), `prune_floor` (0.30), `score_order`, `skip_frac`,
`drop_frac`, `verify_filter`, `per_task_bn`, `head_refit_steps`, `aug`,
`rand_draws`, `cache_acts`.

Metrics printed, and what they mean:

    ACC          mean task-IL accuracy over all tasks after the last task
    BWT          backward transfer. must be exactly 0.0000 here.
    classIL_last class-IL accuracy after the final task
    classIL_avg  average incremental accuracy, mean over all steps
    frozen       fraction of all weights frozen so far
    all_closed   the bitwise closure assertion. must be True.
    oracle       class-IL with the true task id given. the ceiling.
    task-id      routing accuracy, fraction of test images routed to the right task

Half the field reports last accuracy and half reports average incremental. Report
both or you cannot be compared.

## 5. Results so far

### 5.1 exp18, the current design

Run 1, 10 tasks, 100 classes, 300 epochs, seed 0, `score_order=outw`:

    ACC 0.7918   BWT 0.0000   classIL_last 0.2104   classIL_avg 0.4465
    frozen 0.9967   all_closed True

Task 9 trained with 0.6% of the weights still free and still reached
`full 0.794`. The `full` accuracy slope across tasks is -0.0068 per task.

Run 2, same setup, `score_order=taylor`, with the routing ladder, at task 7:

    ACC 0.8029   classIL 0.3699   oracle 0.7535   task-id 0.4168   frozen 0.995

Task 0's circuit was 504 channels out of 960.

n = 1 seed on both. There are no error bars yet.

### 5.2 exp17, for the record

    run                         tasks classes epochs head  task-IL  class-IL
    e17 10-task                   10     100     30   all    33.75     3.23
    e17_own5 (died at task 4)      4      40    200   own    71.95    47.95
    e17_all5                       5      50    200   all    63.82    39.64

The 47.95 is with FeCAM's per-class Mahalanobis readout on our circuits, which
is the best legal readout we have measured. Our own router was 32.65 on the same
run, 15 points below.

### 5.3 The one robust finding

Routing accuracy is a roughly constant multiple of chance, about 3.3x, and it
does not move:

- Split MNIST, 5 tasks, near-disjoint circuits of 10 to 67 units: 3.11x
- CIFAR-100, tasks 4 to 8, 70% overlapping circuits of 500 to 800 channels:
  3.10x, 3.24x, 3.27x, 3.32x
- Across all nine ladder scoring rules: total spread 5 points

The oracle row sits at 0.75 to 0.80 throughout. So the representation is fine
and all of the class-IL gap is task identification. If routing were perfect we
would be at roughly 0.75 to 0.80 class-IL, well above the published state of the
art.

Two hypotheses were tested and refuted by our own data:

- "Sparser, less overlapping circuits will route better." No. Near-disjoint
  circuits routed at 3.11x and 70%-overlapping circuits routed at 3.32x.
- "TPL's complement term buys about 8 points for free." No. Built from the same
  Gaussians it is a monotone transform and cannot change the argmax. Their
  complement is a kNN to a replay buffer, which is a different estimator with a
  different data requirement.

## 6. The problems, stated plainly

This is the list you are being handed.

**P1. Routing is the whole gap.** classIL 0.21 to 0.37, oracle 0.75. Nine
scoring rules on top of our Gaussians all land in a 5 point band at about 3.3x
chance. Whatever fixes this is not another scoring function over the same
features.

**P2. Task-IL is 7 points short.** ACC 0.7918 versus WSN 86.47 and IBM 88.15 on
the identical protocol (10 tasks, ResNet-18 from scratch, 300 epochs, 3 seeds).
We are at SupSup's level (79.94). Unclear how much of this is hyperparameters
and how much is the method.

**P3. Capacity runs out.** By task 9, 99.67% of weights are frozen. Task 9 still
learns, which is surprising and worth understanding, but there is nothing left
for a task 11. Circuits are 500 to 800 channels out of 960 stages, so about half
the network per task, with heavy overlap.

**P4. n = 1.** Every exp18 number is a single seed. Baselines report 3 seeds
with error bars. Nothing we have is publishable as is.

**P5. The selection rule is unvalidated.** Our whole contribution claim is that
causal ablation picks better circuits than magnitude or a learned score. That
ablation, at matched sparsity, has never been run. Until it is, we have no
evidence that the expensive part of our method buys anything.

**P6. A known bug.** The `no_router` row in the ladder is a tautology. It is
built from our own `-z_c` scores, so `argmax` over concatenated `own` and
`argmin_t z_t` pick the identical class, and it prints identical to `z` on every
row. A real "no router" baseline must use head logits. Fix before quoting it.

**P7. Metric hygiene.** The classic forgetting metric F reads 0.248 on a run
with bitwise zero weight movement. Any F we report has to be decomposed.

**P8. We have no clean number on the real benchmark.** Our only completed
100-class, 10-task run in exp17 had task-IL 33.75, which makes its class-IL of
3.23 meaningless. exp18 run 1 is the first real one, at 0.7918 / 0.2104, single
seed.

## 7. State of the art, where we sit

Read `claude/split_cifar100_positioning.md` in the project for the full table
with VERIFIED / SECOND-HAND marks on every number. Summary here.

"Split CIFAR-100" covers at least six incompatible protocols. Numbers from
different ones differ by 40 points and get quoted against each other constantly.
The axes: task-IL vs class-IL, 5/10/20 tasks, cold start vs warm start (50 base
classes then increments), backbone, exemplars or not, last accuracy vs average
incremental accuracy. FeCAM scores 62.1 last in warm start and 32.4 in cold
start. Same method, same data, same backbone. The protocol is worth 30 points.

**Our protocol: class-IL, cold start, 10 equal tasks, ResNet-18 from scratch,
no exemplars.** Last / average incremental.

    DPCR      (ICML 2025)     ~49-50 / 62.8
    EFC++     (2025)          47.52 / 61.57
    AdaGauss  (NeurIPS 2024)  46.10 / 60.20
    LDC       (ECCV 2024)     45.40 / 59.50
    EFC       (ICLR 2024)     43.60 / 58.60
    FeTrIL                    34.90 / 51.20
    FeCAM                     32.40 / 48.30
    OURS (exp18 run 1)        21.04 / 44.65

Differences under about 3 points in this protocol are not meaningful. The CIRCLE
preprint retunes every baseline and reports EFC++ five points above its own
published number.

**Task-IL, same split and backbone** (source: IBM, arXiv 2312.00840, 300 epochs,
3 seeds):

    IBM             88.15
    Multi-task      85.16   (joint training reference)
    WSN             86.47
    DER++           84.23
    SupSup          79.94
    EWC-online      74.05
    OURS            79.18

**A separate line, masks plus per-task OOD**, same dataset and backbone,
reporting numbers 15 points higher than the EFCIL line above:

    CIFAR100-10T      TIL    CIL
    CLOM (HAT+CSI)    92.0   65.2
    Sup+CSI           93.0   65.2
    HAT               84.0   41.1
    SupSup            85.2   33.1

CLOM's recipe is much heavier: supervised contrastive feature learning with
image rotations promoted to real output classes (40 head units per task for 10
classes), batch expanded 8x, and 4 backbone forward passes per task at test
time. It also stores a validation slice for calibration. Its Table 2 is the
striking result: keep SupSup's network, swap in CLOM's OOD scoring, and CIFAR-10
5T CIL goes 26.2 to 81.5 while TIL barely moves, 95.3 to 97.2. That is direct
evidence the gap is in the scoring, which matches our oracle row.

TPL (likelihood ratio, complement from kNN to a replay buffer) reports 62.2 CIL
at 10 tasks without pretraining on ResNet-18, and 76.53 with a pretrained DeiT
plus a buffer. Not the same setting.

**Methods worth reading in full, all relevant to us:**

- WSN, "Forget-free Continual Learning with Winning Subnetworks", ICML 2022.
  Learnable per-weight scores, top-c% per layer, straight-through estimator,
  gradient masking, per-task masks at inference. Code: github.com/ihaeyong/WSN.
  Closest method to ours mechanically. Their subnetworks are learned, ours are
  causally ablated.
- SupSup, "Supermasks in Superposition". Masks over a random network.
- HAT, "Overcoming Catastrophic Forgetting with Hard Attention to the Task".
- PackNet. Iterative pruning plus retraining per task.
- IBM, arXiv 2312.00840. Variational information bottleneck, alpha = mu^2/sigma^2,
  per-layer ratios from SVD. Current task-IL state of the art on our split.
- FeCAM, NeurIPS 2023. Per-class Mahalanobis, Tukey^0.5 transform, shrinkage,
  correlation normalisation, backbone frozen after task 1.
- DPCR, ICML 2025. Dual projection semantic shift estimation plus ridge
  regression classifier reconstruction from per-class Phi_c = X_c^T X_c and mu_c.
  Stores about 26.3M floats. We store 106,741. Current class-IL state of the art.
- Kim et al., NeurIPS 2022, "A Theoretical Study on Solving Continual Learning".
  The CIL = WP x TP decomposition and the necessity theorem.
- CLOM, CVPRW 2022, Kim/Xu/Liu. Code: github.com/k-gyuhak/CLOM.
  Paper readable at ar5iv.labs.arxiv.org/html/2203.09450.
- CSI, "Contrasting Shifted Instances". The OOD method CLOM builds on.
- TPL, likelihood ratio based task prediction.
- AdaGauss, LDC, EFC, EFC++, FeTrIL. The modern exemplar-free class-IL line.
- Frankle and Carbin, Lottery Ticket Hypothesis. Iterative magnitude pruning
  with rewinding to theta_0, or to an early iteration k for ResNets.

Expand this list as you see fit. The positioning doc marks what has actually
been read versus what is second hand. Keep that distinction.

## 8. What we can claim that nobody else can

Be honest about how thin this is, but it is real.

- The circuit is closed under a bitwise test, with a negative control that fails.
  BWT is 0.0000 by construction. Nobody in this literature runs that assertion.
- Storage is 106,741 floats total versus DPCR's 26.3M.
- Causal ablation as the selection rule, if P5 ever gets validated.
- The oracle row at 0.75 to 0.80 says the representation is already good enough
  to beat the state of the art if routing were solved.

## 9. Practical notes and traps

- The MNIST download hosts return 403 in the sandbox. Real MNIST is recovered
  in `mnist_real.npz` via `mnist_real.py` (from the `mnist-hub` pypi wheel,
  unpickled with a restricted Unpickler). Do not try to re-download it.
- GPU sync killers found and fixed once already, do not reintroduce them:
  `bool(m.any())` per step, `float(loss)` per step, and `p[bool_mask] = v`
  (which calls `nonzero()`). The `Freezer` uses precomputed integer indices and
  `index_copy_`. Measured on one 512x256x3x3 conv: boolean indexing 8.12ms,
  `torch.where` 2.89ms, `index_copy_` 0.54ms.
- `gpu_setup()` has a determinism guard: two forward passes must be bitwise
  equal. Keep it. Closure proofs are worthless without it.
- Activation caching for the pruning sweep (`cache_acts`, `stage_input`,
  `head_from`) was measured at 3.48x speedup.
- Size controls must be stated carefully. "No random circuit met the gate at any
  size" was an overclaim: the bisection only searches up to our own size. The
  correct phrasing is "at or below our size".
- BatchNorm: `track_running_stats=False`. `affine=False` unless
  `per_task_bn=1`. See `_bn()`.
- Self-tests: 11 of them, all passing. Test 3 checks we freeze 180 weights where
  exp17 froze 360. Test 5 is closure at 0.0e+00. Test 6 is the negative control
  failing at 5.06e-01. Test 10 is sqrt(k) scaling. Test 11 is that frozen
  channels still receive a gradient score. Run `--self-test` after any edit.

## 10. Pending work already identified

Listed so you do not rediscover them. Prioritise as you judge.

- Fix P6, the `no_router` tautology.
- Run 3 seeds so ACC 0.7918 gets an error bar.
- Run the causal vs magnitude vs learned-score ablation at matched sparsity (P5).
- Run `--verify-filter` on one real CIFAR task to test whether the cheap
  gradient score can safely shorten the ablation sweep.
- Decide and justify what to do about routing (P1). This is the main question.