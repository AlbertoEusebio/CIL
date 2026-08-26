# Literature review: exemplar-free task-id prediction for mask-based class-incremental learning

Scope: Split CIFAR-100, cold start, 10 tasks of 10 classes, ResNet-18 from scratch, no exemplars, no task label at test time. Our system: per-task channel circuits found by greedy causal ablation, WSN-style freezing of read weights (backward transfer exactly zero), z-standardised per-class Mahalanobis routing on circuit features. Task-IL 79.2, oracle-routed CIL 75 to 80, routed CIL 21 to 37, task prediction roughly 3.3x chance (about 33 percent).

Tagging rule. VERIFIED means I read the number in the named table of the named paper during this review. SECOND-HAND means the number was copied by one paper from another and I only saw the copy. Anything I could not check is said to be unchecked. Arithmetic I did myself is labelled as mine.

## Summary: where the 30-point gap can come from, ranked by evidence

1. The task detector is weak because the per-task models were trained with plain cross-entropy, not because the scoring function is wrong. This is the best supported claim. CLOM (Kim et al., CVPRW 2022, arXiv 2203.09450, Table 2, VERIFIED) reports, for SupSup on C100-10T with ordinary cross-entropy training, a task detection rate of 34.3 percent and CIL of 33.1. That is almost exactly our 3.3x chance and our 21 to 37 CIL. Swapping the per-task training recipe for CSI (supervised contrastive plus rotation-as-OOD, 700 epochs) raised task detection to 63.7 and CIL to 62.1 with the same SupSup masking. Bolting a post-hoc OOD scorer (ODIN) onto the cross-entropy models only moved task detection from 34.3 to 43.0. So roughly three quarters of the gain in that paper came from training, one quarter from scoring. Caveat: their backbone is ResNet-18 with doubled channels, and CSI is expensive.

2. Rotation-as-OOD during training is the single biggest ingredient inside CSI. CLOM Table 2 (VERIFIED): full CLOM task detection 66.8, CLOM with contrastive learning but without the rotation OOD classes 59.5. Table 4 (VERIFIED): crop plus rotation gives CIL 60.3 while all augmentations without rotation give 50.2. This is cheap to test on our pipeline.

3. The task scores from independently trained heads live on different scales, and this alone costs several points. CLOM Table 3 left (VERIFIED): output calibration with 5 stored samples per class moves C100-10T CIL from 63.3 to 64.9; with 20 per class 65.2. TOOD (arXiv 2607.29592, VERIFIED) documents a "confidence gap" where old-task energies shrink over time, and shows that per-task recentring by in-distribution statistics fixes it but per-task rescaling alone makes it worse (Appendix D.2: minus 3.6 AUROC for rescale-only versus plus 3.1 for recentre-and-rescale). Our z-standardisation is a recentre-and-rescale, so this may already be handled, but it should be checked which statistics it uses (train features versus held-out).

4. Per-class Mahalanobis with raw covariance is fragile at 500 samples per class. FeCAM (NeurIPS 2023, Table 4, VERIFIED): on CIFAR-100 warm start, Mahalanobis with full per-class covariance and no shrinkage or normalisation gives last accuracy 14.6; adding shrinkage gives 44.6; adding Tukey transform and correlation normalisation gives 62.1; plain Euclidean NCM is 51.6. Our routing score may be sitting in the bad regime. AdaGauss (NeurIPS 2024, Table 3, VERIFIED) reduces the feature dimension to 64 partly for this reason.

5. Feature-space density scores that beat max-logit exist but the gains are moderate, and the strongest ones use replay. TPL (ICLR 2024, Table 4, VERIFIED, pretrained DeiT, C100-10T): MSP 62.9, MLS 69.2, Energy 69.1, Mahalanobis (tied cov) 69.0, KNN 61.5, ViM 72.8. The likelihood-ratio score that gives TPL its edge needs a replay buffer to model the complement distribution. Exemplar-free, the evidence points to ViM or Mahalanobis plus max-logit, not to any single score.

6. Batch-wise task inference closes the gap almost completely but violates the single-sample protocol. PR-Ent (Henning et al., NeurIPS 2021, Table S11, VERIFIED): SplitCIFAR-10, single-sample entropy 61.9, batch of 100 samples 92.9. CP&S (Applied Intelligence 2023, arXiv 2208.04952, VERIFIED) needs a test batch of 20 to match task-IL. This is not a solution for us but it is a useful diagnostic of whether per-sample scores carry signal.

7. Circuit capacity or feature quality is probably not the main problem. Our oracle CIL of 75 to 80 is above every exemplar-free cold-start number in the literature, so WP is fine. The problem is TP.

What no paper shows: any exemplar-free, single-sample, from-scratch ResNet-18 method that gets task prediction above roughly 67 percent on C100-10T. DCNet (arXiv 2501.15454, Table 1, VERIFIED) reports CIL 65.4 last with a HAT plus CSI style recipe, which is the current ceiling for this family.

## 1. The task-prediction / per-task OOD line

### Kim et al., "A theoretical study on solving continual learning" (NeurIPS 2022, arXiv 2211.02633)

Mechanism. CIL probability factorises as P(class | x) = P(class | task, x) times P(task | x), so CIL is within-task prediction (WP) times task prediction (TP). They prove TP and per-task OOD detection bound each other under cross-entropy. Nothing is stored except the per-task masks. The concrete system replaces the per-task classifier in HAT or SupSup with a CSI model. Prediction is argmax over concatenated per-task logits (Eq. 9), which is a special case of the theory with OOD_k = sigmoid(max logit of head k).

Protocol (Sec 4.2, VERIFIED): ResNet-18 with doubled channels for CIFAR-100, no pretraining, 700 epochs LARS for the feature extractor then 100 epochs for the linear head, 5 seeds, average accuracy over all classes after the last task. The plus c variants use a 2000-sample memory buffer only to fit a per-task scale and shift of the logits.

Numbers on C100-10T (Table 3, VERIFIED): HAT 41.1, SupSup 44.6, HyperNet 30.2, PR-Ent 45.2 (copied from its paper), PASS 33.0, DER++ 53.7 (with 2000 exemplars), HAT+CSI 63.3, Sup+CSI 65.1, HAT+CSI+c 65.2, Sup+CSI+c 65.2. TIL (Table 4, VERIFIED): HAT 84.0, Sup 87.9, HAT+CSI 92.0, Sup+CSI 93.0. So CSI also raises WP by 5 to 8 points.

Ablation of scoring versus training (Table 2, VERIFIED): HAT with ODIN scoring AUC 77.8, CIL 41.2; HAT with CSI AUC 84.5, CIL 63.3. Sup with ODIN AUC 80.6, CIL 46.7; Sup with CSI AUC 86.8, CIL 65.1. Table 1 (VERIFIED): ODIN moves Sup from 44.58 to 46.74. Footnote 6 (VERIFIED): SupSup's own one-shot entropy-gradient task inference on C10-5T gives 50.2 CIL versus 62.6 using argmax over concatenated logits. Footnote 5 (VERIFIED): Expert Gate gets 43.2 on MNIST-5T; iTAML with single-sample batches gets 33.5 on C100-10T.

What it does not prove. The AUC to CIL correlation is on a handful of points. The gain is confounded: CSI changes the training objective, augmentations, epochs (700), and adds an ensemble over four rotations at test time. The paper does not separate these.

### CLOM (Kim et al., CVPRW 2022, arXiv 2203.09450)

Same system as HAT+CSI with more ablations. Stored per task: HAT masks, task-specific head, and for calibration 20 validation samples per class (Sec 4, VERIFIED). Table 1 (VERIFIED): C100-10T CLOM(-c) 63.3, CLOM 65.2, SupSup 33.1, HAT 41.1. Average incremental accuracy (Table 5, VERIFIED): CLOM(-c) 75.4, CLOM 75.9.

The key table for us is Table 2 (VERIFIED), C100-10T, columns AUC / TaskDR / TIL / CIL:
SupSup 76.7 / 34.3 / 85.2 / 33.1;
SupSup with CLOM's OOD model 84.9 / 63.7 / 90.0 / 62.1;
CLOM with ODIN instead of CSI 77.9 / 43.0 / 84.0 / 41.3;
CLOM 85.0 / 66.8 / 92.0 / 65.2;
CLOM without OOD (contrastive but no rotation classes) 82.6 / 59.5 / 89.8 / 57.5.
Table 3 right (VERIFIED) shows that weak mask protection (small s in HAT) destroys AUC and CIL, so zero forgetting is a prerequisite and we already have that.

What it does not prove. TaskDR here is measured by whether the argmax class lands in the right task, so it is entangled with head calibration. Everything is at 700 epochs with a wide ResNet-18.

### MORE (CoLLAs 2022, arXiv 2208.09734), ROW (ICML 2023, arXiv 2306.12646), TPL (ICLR 2024, arXiv 2309.15048)

All three use HAT on adapters inside a DeiT-S/16 pretrained on 611 ImageNet classes, and all three store a replay buffer of 2000 samples for CIFAR-100. The buffer is not replayed to fight forgetting; it is used as OOD data when training each task head (MORE, ROW) and as the estimate of the "other tasks" distribution in the likelihood ratio (TPL). None of them is exemplar-free.

MORE Table 1 (VERIFIED): C100-10T last 70.23, AIA 81.24; HAT with task-id prediction 62.34. Table 4 (VERIFIED, AIA): base 76.93, plus Mahalanobis coefficient 80.31, plus back-update 80.35, both 81.24. The coefficient is max over classes of 1/Mahalanobis distance with per-class means and a task-shared covariance, multiplied into the softmax.

ROW Table 4 (VERIFIED): C100-10T ROW 74.72, without WP head 72.29, without WP head and without Mahalanobis coefficient 67.53. So the Mahalanobis term is worth about 4.8 points there.

TPL Table 1 (VERIFIED, pretrained): TPL 76.53 last, 84.10 AIA; ROW 74.72; MORE 70.23; HAT_CIL 62.91. Table 8 (VERIFIED, no pretraining, ResNet-18, 2000 buffer): TPL 62.2, ROW 58.2, MORE 57.5, HAT_CIL 41.1, DER 64.5. Ablation Fig 2a (VERIFIED, averaged over five datasets, pretrained): HAT_CIL 63.41, HAT plus MLS 68.69, HAT plus likelihood ratio 71.25, TPL 76.21. Table 4 (VERIFIED, C100-10T, pretrained, 20 OOD scores on the same HAT models): MSP 62.9, ODIN 61.7, MLS 69.2, Energy 69.1, ReAct 69.1, KNN 61.5 (buffer only), Residual 64.7, Mahalanobis 69.0, ViM 72.8, OE 66.7, LogitNorm 64.3. Table 3 (VERIFIED): halving the buffer to 1000 drops TPL from 76.21 to 75.56.

What they do not prove. The pretrained backbone makes features far better behaved than ours. The exemplar-free reader of Table 4 should note that ViM and MLS work with nothing stored beyond class statistics; KNN is bad only because it had to use the buffer.

### Open-world continual learning (Kim et al., AIJ 2024, arXiv 2304.10038)

Journal extension of the NeurIPS paper. Same C100-10T numbers (Table 3 and Table 4 there, VERIFIED, identical to the NeurIPS Table 2 and 3). Adds the open-world extension and a MORE-style replay method. Nothing new for the exemplar-free case.

### DCNet (arXiv 2501.15454)

Exemplar-free HAT plus CSI style system. Adds incremental orthogonal embedding (classes mapped near fixed orthogonal targets on a hypersphere, basis dimension 256) and dynamic aggregation compensation (a temperature schedule). Protocol (Sec 5.1 and C.1, VERIFIED): ResNet-18 from scratch, 700 epochs LARS, self-rotation, five runs. Table 1 (VERIFIED): CIFAR-100 Split-10 A_inc 75.84, A_last 65.40; Split-20 71.52 / 58.43. Table 3 (VERIFIED): their HAT+CSI reimplementation 73.30 / 63.32. Table 2 (VERIFIED): with zero buffer DCNet 65.4 last versus TPL 62.2, ROW 58.2, MORE 57.5 (2000 buffer, no pretraining, copied from TPL). Table 1 also lists SEED (Rypesc et al., ICLR 2024) at 62.04 / 51.42, which is SECOND-HAND and worth chasing since SEED is exemplar-free and uses a mixture of experts with per-class Gaussians.

What it does not prove. The gain over HAT+CSI is 2 points and comes with a 700-epoch contrastive recipe. Whether the ResNet-18 here has doubled channels as in Kim et al. is not stated clearly.

### Takeaway for us

Everything in this line agrees with our own number: plain cross-entropy heads give about one third task detection on C100-10T. The published fixes, in order of measured effect, are rotation-as-OOD in training, supervised contrastive training, per-task score calibration, and finally a better feature-space score.

## 2. Mask and subnetwork methods

### SupSup (NeurIPS 2020, arXiv 2006.14769)

Scenarios (Table 1, VERIFIED): GG (task given at train and test), GNs (task given at train only, shared labels), GNu (task given at train only, unshared labels, i.e. class-IL), NNs. Inference in GN: superimpose all masks with weights alpha, take one gradient step of the output entropy with respect to alpha, and pick the coordinate with the largest decrease (One-Shot, Eq. 4). An alternative objective G uses superfluous output neurons. SplitCIFAR-100 numbers exist only for GG (Table 2, VERIFIED): 20 tasks of 5 classes, ResNet-18 with fewer channels, 77.56 to 89.57 depending on sparsity, up to 91.66 with mask transfer. GNu results are on PermutedMNIST, RotatedMNIST and SplitMNIST only. Appendix D says task inference is once per batch; the text says single images unless noted. Kim et al. (NeurIPS 2022, footnote 6, VERIFIED) measured the one-shot method at 50.2 CIL on C10-5T, below 62.6 for plain max-logit. CLOM (Table 2, VERIFIED) measured SupSup's task detection at 34.3 on C100-10T.

### WSN (ICML 2022) and its TPAMI extension (arXiv 2312.11973)

Learned real-valued weight scores, top-c percent per layer, previous-task weights frozen but reusable. The ICML paper is task-IL only. The TPAMI extension adds a "TaIL" setting with SupSup's one-shot inference (Sec 3.4.1, VERIFIED). Table 2 (VERIFIED): Seq-CIFAR100 with 5 tasks of 20 classes, ResNet-18, WSN c=70 percent 46.24, WSN plus FSO 77.12, Finetune 17.41, joint 70.10. The text claims task inference "shows 100 percent accuracy for all tasks", which is inconsistent with WSN alone scoring 46 in a 5-task setting where task-IL is well above 80; treat the claim as unreliable. Task-IL Table 1 (VERIFIED): CIFAR-100 Split 10 tasks, AlexNet, PackNet 72.39, SupSup 75.47, WSN c=50 77.67.

### PackNet, Piggyback, HAT, CPG

These report task-IL only in their own papers (not re-read here). The CIL numbers that exist for HAT and SupSup come from Kim et al. and CLOM above (HAT 41.1, SupSup 33.1 to 44.6 depending on the prediction rule). No CIL numbers for PackNet, Piggyback or CPG were found.

### IBM (arXiv 2312.00840)

Variational information-bottleneck masks on weights, per-layer ratios set automatically from SVD of hidden features. Task-IL only. Table 1 (VERIFIED): AlexNet CIFAR-100, WSN 82.28, IBM 82.69, SupSup 67.30. ResNet-18 (Fig 1, VERIFIED): IBM 88.15 versus WSN 86.47 with 70 percent fewer masked parameters. Relevant to us because it argues explicitly that weight magnitude is not importance, but it never compares selection rules at matched sparsity.

### CLNP, Continual learning via neural pruning (arXiv 1903.04476)

Activation-based neuron pruning with L1 and an activity threshold, free neurons for new tasks, graceful forgetting margin. Task-IL only (VERIFIED). Notable for its "used capacity per layer" diagnostic: early layers stop growing after two tasks, later layers keep growing (Fig 5b).

### CP&S (Applied Intelligence 2023, arXiv 2208.04952)

Iterative pruning per task (NNrelief), frozen subnetworks, class-IL inference by max-output over a test batch. It is exemplar-free but needs a batch of 20 test samples from the same task (Table 1 and Fig 2b, VERIFIED). CIFAR-100 10 tasks, ResNet-18, Adam, 70 epochs: with bs=20 the curve is within a few points of task-IL (Fig 9b, VERIFIED, figure only, no table). With 20 tasks task selection collapses after task 11 because the network runs out of free connections (Fig 4, VERIFIED). No single-sample number.

### LwI, Learning without isolation (ICML 2025, arXiv 2505.18568)

Channel pathways protected by graph matching model fusion, no masks at inference. Table 1 (VERIFIED): ResNet-18 CIFAR-100 10 splits, 200 epochs, task-agnostic 36.36 versus task-aware 84.90; LwF 30.41 / 81.35. Confirms the same 45-point task-agnostic gap we see, with a completely different protection mechanism.

### Cortex-inspired FTN (arXiv 2604.24637) and zero-leakage routing (arXiv 2604.14375)

FTN recovers a task mask by one gradient step on a continuous mask plus spatial smoothing plus k-winners, on a support batch, on MNIST variants only. The zero-leakage paper uses per-task tight-bottleneck autoencoders as reconstruction routers, reports 96.1 percent routing on Split-MNIST with two tasks (Table 2, VERIFIED), and states it is not designed for class-IL within one dataset (Sec 8). Neither gives CIFAR-100 evidence.

### Takeaway

No mask method reports single-sample exemplar-free CIL on C100-10T above the HAT+CSI family. The mask papers that do report CIL either use a test batch (CP&S, iTAML, PR-Ent BW) or inherit the Kim et al. recipe.

## 3. The exemplar-free CIL line with class statistics

These methods keep one backbone, update it with distillation, and classify all classes with a single head built from stored class means and covariances. They are the alternative to routing. Their advantage is that there is no task detector; their disadvantage is drift of the stored statistics.

Protocol notes matter. EFC and EFC++ train the first task with self-rotation and give 100 epochs per incremental step with Adam; AdaGauss uses SGD 200 epochs and a 64-d bottleneck; DPCR follows LwF settings with 200 then 100 epochs, three class orders; APR adds AutoAugment and a cosine head; CIRCLE retunes every baseline per horizon. Numbers from different papers are therefore not on the same footing even when the split is the same.

FeCAM (NeurIPS 2023). Freezes the backbone after task 1, per-class covariance with shrinkage, correlation normalisation, Tukey transform, Bayes rule over all classes. Designed for warm start. Table 5 (VERIFIED): first task 20 classes then 20 per task, Euclidean NCM 30.6 last / 50.0 avg, FeTrIL 46.2 / 61.3, FeCAM 48.1 / 62.3. The cold-start 10-task number for FeCAM comes from other papers (see table below).

EFC (ICLR 2024, arXiv 2402.03917). Empirical feature matrix regulariser, Gaussian prototypes with drift compensation, asymmetric prototype rehearsal into a linear head. Table 1 CS 10-step (VERIFIED): EWC 31.17 / 49.14, LwF 32.80 / 53.91, PASS 30.45 / 47.86, FeTrIL 34.94 / 51.20, SSRE 30.40 / 47.26, EFC 43.62 / 58.58. Fig 4b (VERIFIED): without prototype update EFC CS 10-step is 39.28.

EFC++ (arXiv 2503.10439). Decouples backbone training from a post-training prototype rebalancing of the linear head. Table 2 CS 10-step (VERIFIED): FeCAM 37.63 / 52.53, ABD 43.11 / 59.14, R-DFCIL 42.14 / 57.77, EFC 43.62 / 58.58, EFC++ 47.52 / 61.57. Table 6 (VERIFIED): with oracle class means and fixed covariances 49.22; oracle means and covariances 49.33; joint training 71.00. So even perfect drift compensation of means would only add about 2 points; the rest of the gap to joint is in the representation.

LDC (ECCV 2024, arXiv 2407.08536). Linear projector from old to new feature space learned on current data, NCM over all classes. Table 1 (VERIFIED, 5 seeds): LwF+NCM 40.5 / 56.2, PASS 37.8 / 52.3, FeTrIL 37.0 / 52.1, FeCAM 33.1 / 48.1, EFC 43.6 / 58.6 (copied from EFC), LwF+LDC 45.4 / 59.5. Fig 1 shows an oracle-prototype line far above naive prototypes, i.e. most of the loss in prototype methods is drift, not feature damage.

AdaGauss (NeurIPS 2024, arXiv 2409.18265). Adapts means and covariances with an MLP adapter, anti-collapse loss on the Cholesky diagonal, 64-d bottleneck, Bayes classifier. Table 1 (VERIFIED): CIFAR-100 T=10 EFC 43.6 / 58.6, DS-AL 40.8 / 54.9, FeCAM 32.4 / 48.3, FeTrIL 34.9 / 51.2, AdaGauss 46.1 / 60.2; T=20 37.8 / 52.4. Table 3 ablation T=10 (VERIFIED): NMC no covariance 36.4, Bayes diagonal 41.1, full covariance no adaptation 22.9, adapt means only 42.9, no anti-collapse with shrink 0.5 instead 40.2, full 46.1. Note baseline rows in Table 1 are copied from EFC.

DPCR (ICML 2025, arXiv 2503.05423). LwF-style training, then a task-wise linear shift projection plus per-class row-space projection to move stored uncentred covariances and means, then a ridge-regression classifier over all classes rebuilt from those statistics; stores d squared plus d per class. Table 1 (VERIFIED, 3 class orders): CIFAR-100 T=10 LwF 42.60 / 58.51, SDC 42.25 / 58.43, PASS 44.47 / 55.88, ACIL 35.53 / 50.53, FeCAM 34.82 / 49.14, DS-AL 36.83 / 51.47, ADC 46.80 / 62.05, LDC 46.60 / 61.67, DPCR 50.24 / 63.21; T=20 38.98 / 54.42. Table 4 ablation (VERIFIED): ridge classifier only 32.17, plus task shift 40.86, plus class projection 45.56, plus normalisation 51.04. The ablation full row (51.04) differs from the main table (50.24), presumably a different seed set.

APR, adversarial pseudo-replay (arXiv 2511.17973). Online adversarial perturbation of new-task images toward old prototypes for distillation, transfer matrix to calibrate covariances, Mahalanobis head. Table 1 cold start T=10 (VERIFIED, 3 seeds): APR Maha 57.94 last / 69.96 avg, APR NCM 56.29 / 69.00, APR linear 53.18 / 67.20. But their reimplemented baselines are much higher than elsewhere: AdaGauss 53.27 / 66.66, ADC 52.59 / 66.24, LwF linear 49.19 / 63.66, FeCAM 37.82 / 52.61, joint 78.69 / 84.18. The difference is AutoAugment, a cosine head, validation-tuned shrinkage, and possibly the class orders. APR should not be read as "beats DPCR by 8 points"; it is a different protocol. Training time is 31 GPU hours on ImageNet-Subset.

GATF, geometry-anchored transport (arXiv 2606.25347). Built on the AdaGauss code; a closed-form generalised least squares transport prior learned during backbone training plus a residual MLP, Bayes classifier. Table 1 (VERIFIED, 5 runs): T=10 EFC 43.5 / 58.1, ADC 46.5 / 61.4, LDC 45.4 / 59.5, AdaGauss 46.8 / 60.9, DPCR 50.2 / 62.8, GATF 52.0 / 64.4; T=20 GATF 43.0 / 56.6.

BiCyc (ICLR 2026, arXiv 2606.05675). Bidirectional adapters with cycle consistency, also on the AdaGauss code. Table 1 (VERIFIED): BiCyc 50.6 / 63.2; T=20 41.5 / 56.5. Table 7 (VERIFIED): Bayes head 50.6 versus a linear head trained on Gaussian samples 51.1, so the head choice is not what matters.

CEOS plus ACB, prototype rehearsal revisited (arXiv 2606.05695). Built on EFC; interpolates prototypes toward nearest enemy features, time-weighted class-balanced loss. Table 1 (VERIFIED): T=10 46.9 / 60.2; T=20 34.9 / 48.0.

CIRCLE (arXiv 2606.27095). Never trains a backbone; fixed random 2-D reservoir features plus streaming LDA heads. Table 1 (VERIFIED, 5 seeds, all baselines retuned per horizon): CIFAR-100 T=10 AdaGauss 46.57 last / 62.53 avg, EFC++ 52.68 / 66.61, ADC 44.71 / 61.98, DS-AL 41.76 / 56.43, FeCAM 33.08 / 49.71, CIRCLE 45.34 / 55.32. The EFC++ number here is 5 points above the EFC++ paper's own 47.52, because CIRCLE gave it 200 epochs per step and retuned it; this is a warning that the "published bar" moves by several points with tuning.

SPARCL (arXiv 2608.21307) is frozen ViT-B/16 only (Table 1, VERIFIED, 94.52 on CIFAR-100) and irrelevant to from-scratch ResNet-18.

Takeaway. The best from-scratch cold-start numbers as of August 2026 are GATF 52.0 / 64.4 and BiCyc 50.6 / 63.2 with the standard protocol, and about 58 / 70 under APR's heavier protocol. All of them use a single Gaussian or ridge head over all 100 classes with no router. Our oracle-routed 75 to 80 is 20 points above all of them, which is exactly why the router is the whole problem.

## 4. Exemplar-free task-id scoring techniques

For each: what it needs to store, and what the evidence says.

Max logit, MSP, energy. Store nothing. TPL Table 4 (VERIFIED, pretrained): MSP 62.9, MLS 69.2, Energy 69.1 CIL. CLOM uses argmax of concatenated logits. Weakness: TOOD (VERIFIED, Sec 3) shows old-task logits shrink in scale as the classifier grows; in our multi-head setting the analogue is that independently trained heads have different scales. Per-task recentring using in-distribution statistics fixed this in TOOD (up to plus 8.1 AUROC on DER, Table 1, VERIFIED), and rescaling without recentring hurt.

ODIN (temperature plus input gradient). Store nothing, needs a backward pass. Kim et al. Table 1 (VERIFIED): plus 2 points for SupSup. Small.

Mahalanobis, Lee et al. 2018 (arXiv 1807.03888). Tied covariance across classes, class means, optional feature ensemble across layers with weights fit by logistic regression on validation OOD or FGSM samples, optional input preprocessing. Table 1 (VERIFIED, CIFAR-10 vs SVHN, ResNet): plain 54.5 TNR at 95 TPR, with preprocessing 92.3, with feature ensemble 91.5, both 96.4. The layer weights need some negative data; FGSM adversarials of the in-distribution data are shown to work, which is exemplar-free. Our per-class covariances are a different and riskier estimator; see FeCAM Table 4 above.

Relative Mahalanobis, Ren et al. 2021 (arXiv 2106.09022). Subtract the Mahalanobis distance to a single background Gaussian fit on all training features. Stores one extra mean and covariance per task. Table 1 (VERIFIED, WRN-28-10 from scratch): CIFAR-100 vs CIFAR-10 AUROC MD 74.91, RMD 81.01, MSP 80.14. Their analysis (Sec 3) is directly relevant: in a 640-d feature the top 120 eigen-directions discriminate and the remaining directions add noise to MD. Our circuit features are lower dimensional, which may help or hurt; RMD is hyperparameter free and cheap to try.

ViM and Residual (Wang et al. 2022). Store the principal subspace of training features and the head weights. TPL Table 4 (VERIFIED): ViM was the best single post-hoc score, 72.8 CIL on C100-10T with pretrained DeiT, versus 69.0 for Mahalanobis. Not tested from scratch there.

KNN (Sun et al. 2022). Needs stored training features, so not exemplar-free in the strict sense; storing k normalised features per class is a grey zone. TPL Table 4: 61.5 with only buffer features.

Feature norm, entropy, Gram matrices, test-time BN statistics. Feature norm and entropy store nothing; Gram statistics store per-layer correlation ranges; test-time BN needs a batch. No CIL papers with C100-10T numbers for these were found in this review. PR-Ent (Henning et al. 2021, Table S13, VERIFIED) gives the best exemplar-free entropy-based number I found: SplitCIFAR-100 10 tasks ResNet-18, deterministic per-task models, task-given 85.16 and task-inferred by entropy 40.35; a Bayesian variant 86.56 / 45.22; separate networks per task 89.52 / 50.80. They also report agreement across posterior samples (epistemic uncertainty) as an alternative score; on CIFAR-100 it is slightly worse than entropy (Table S13, VERIFIED).

Generative or likelihood based. Expert Gate (CVPR 2017, arXiv 1611.06194) trains one undercomplete autoencoder per task on frozen AlexNet features and routes by reconstruction error through a softmax with temperature 2. Table 3 (VERIFIED): 97.6 percent gate accuracy across six datasets versus 97.8 for a discriminative task classifier trained on all data. But those tasks are different datasets (scenes, birds, flowers, cars, aircraft, actions) with ImageNet-pretrained features; on MNIST-5T Kim et al. measured 43.2 CIL for it. CN-DPM (ICLR 2020) uses a VAE density per expert times a classifier, with a short-term memory of 1000 samples, and reaches 20.1 on 20-task Split-CIFAR-100 in one epoch (Table 8, VERIFIED). The zero-leakage paper gets 96 percent routing on two MNIST tasks. Nothing here suggests per-task density models beat discriminative OOD scores on same-dataset splits of CIFAR-100.

Mask response. SupSup's one-shot entropy gradient over superposed masks; measured at 50.2 CIL on C10-5T by Kim et al. versus 62.6 for max-logit; CLOM measured 34.3 task detection on C100-10T. FTN's gradient-plus-smoothing recovery is MNIST only. CP&S importance-score matching (Sec 4, VERIFIED) beats max-output when tasks are imbalanced but needs a batch of 60.

Model Zoo (ICLR 2022). Task-IL only, stores past data, explicitly says it is not designed for unknown task identity (Remark 6, VERIFIED).

Likelihood ratio (TPL). Needs a buffer for the complement density; the exemplar-free fallback ("Constant" in their Fig 2b, i.e. uniform complement) is exactly plain Mahalanobis.

Batch-wise inference. PR-Ent Table S11 (VERIFIED): SplitCIFAR-10 ResNet-32 single-sample entropy 61.9, batch of 100 samples 92.9. Not admissible for us, but a five-minute diagnostic: if batching our z-scores recovers oracle accuracy, the per-sample scores are informative but noisy; if it does not, the scores are biased.

## 5. Causal or ablation-based subnetwork discovery in continual learning

I found no continual learning paper that selects a task subnetwork by greedy causal ablation of channels, and no paper that compares selection rules (magnitude, learned scores, gradient, activation, ablation) at matched sparsity for continual learning. What exists:

IBM (VERIFIED) argues magnitude is a poor importance proxy and uses a variational bottleneck instead, but compares only to WSN and SupSup at their default settings. LwI (VERIFIED) cites SPU (Zhang et al. 2024) as using "causal tracking" to select parameters to update; I did not read SPU. CLNP (VERIFIED) uses activation-based pruning. FTN (VERIFIED) uses a gradient on a continuous mask followed by k-winners, and ablates the spatial smoothing but not the selection rule. The mechanistic interpretability literature (activation patching, attribution patching, ACDC; Syed et al. 2023 "Attribution patching outperforms automated circuit discovery") gives efficient approximations to per-unit ablation effects but has not been applied to continual learning. The 2026 toy study "Sparsity, superposition and forgetting" (arXiv 2606.20431, VERIFIED) is about replay-trained dense networks and does not do circuit discovery.

Web searches for gradient versus activation versus magnitude importance in continual learning turned up "Continual learning with neuron activation importance" (arXiv 2107.12657) and "Visually grounded continual language learning with selective specialization" (arXiv 2310.15571), which compare gradient-based and activation-based importance; neither was read in full here and neither is on our benchmark.

So the selection-rule question is open. Since our oracle CIL is already above the literature, the selection rule is not where our loss is, and a matched-sparsity comparison would be a contribution but not a fix.

## Table: Split CIFAR-100, 10 tasks, cold start, ResNet-18 from scratch, exemplar-free CIL

Last / average incremental accuracy. All are 10 equal tasks unless noted.

| Method | Last | Avg | Source and tag | Protocol notes |
|---|---|---|---|---|
| EWC | 31.17 | 49.14 | EFC Table 1, VERIFIED | first task self-rotation, 100 ep/step |
| LwF | 32.80 | 53.91 | EFC Table 1, VERIFIED | same |
| LwF | 42.60 | 58.51 | DPCR Table 1, VERIFIED | PyCIL LwF, 200 then 100 ep, 3 orders; note 10 pts above EFC's LwF |
| PASS | 30.45 | 47.86 | EFC Table 1, VERIFIED | |
| SSRE | 30.40 | 47.26 | EFC Table 1, VERIFIED | |
| IL2A | 31.7 | 48.4 | AdaGauss Table 1, VERIFIED (copied from EFC by them) | |
| FeTrIL | 34.94 | 51.20 | EFC Table 1, VERIFIED | frozen backbone after task 1 |
| FeCAM | 37.63 | 52.53 | EFC++ Table 2, VERIFIED | frozen backbone; 32.4 / 48.3 in AdaGauss Table 1 (VERIFIED) |
| DS-AL | 40.8 | 54.9 | AdaGauss Table 1, VERIFIED | frozen backbone |
| ABD | 43.11 | 59.14 | EFC++ Table 2, VERIFIED | deep inversion generator |
| R-DFCIL | 42.14 | 57.77 | EFC++ Table 2, VERIFIED | |
| EFC | 43.62 | 58.58 | EFC Table 1, VERIFIED | |
| EFC++ | 47.52 | 61.57 | EFC++ Table 2, VERIFIED | 52.68 / 66.61 when retuned by CIRCLE (VERIFIED) |
| LwF+LDC | 45.4 | 59.5 | LDC Table 1, VERIFIED | NCM head, 5 seeds |
| ADC | 46.80 | 62.05 | DPCR Table 1, VERIFIED | NCM head |
| AdaGauss | 46.1 | 60.2 | AdaGauss Table 1, VERIFIED | 64-d bottleneck, Bayes head |
| CEOS+ACB | 46.9 | 60.2 | arXiv 2606.05695 Table 1, VERIFIED | on EFC code |
| DPCR | 50.24 | 63.21 | DPCR Table 1, VERIFIED | ridge head, 3 orders |
| BiCyc | 50.6 | 63.2 | arXiv 2606.05675 Table 1, VERIFIED | on AdaGauss code, 5 runs |
| GATF | 52.0 | 64.4 | arXiv 2606.25347 Table 1, VERIFIED | on AdaGauss code, 5 runs |
| SEED | 51.42 | 62.04 | DCNet Table 1, SECOND-HAND | mixture of experts, Gaussians |
| CIRCLE | 45.34 | 55.32 | arXiv 2606.27095 Table 1, VERIFIED | untrained reservoir features, SLDA |
| APR (Maha) | 57.94 | 69.96 | arXiv 2511.17973 Table 1, VERIFIED | AutoAugment, cosine head, tuned shrinkage; baselines also inflated |
| SupSup + max-logit | 33.1 | | CLOM Table 1, VERIFIED | ResNet-18 doubled channels, 700 ep |
| HAT + max-logit | 41.1 | | Kim NeurIPS22 Table 3, VERIFIED | same |
| PR-Ent | 45.2 | | Kim NeurIPS22 Table 3, SECOND-HAND from Henning et al. | |
| PR-Dirac (entropy) | 40.35 | | Henning Table S13, VERIFIED | task-given 85.16 |
| LwI | 36.36 | | LwI Table 1, VERIFIED | task-aware 84.90 |
| HAT+CSI | 63.3 | | Kim NeurIPS22 Table 3, VERIFIED | 700 ep LARS, rotation OOD; 75.4 AIA in CLOM Table 5 |
| Sup+CSI | 65.1 | | same | |
| HAT+CSI+c | 65.2 | 75.9 | CLOM Tables 1 and 5, VERIFIED | 20 samples/class for calibration only |
| DCNet | 65.40 | 75.84 | arXiv 2501.15454 Table 1, VERIFIED | HAT+CSI style, 700 ep |
| TPL (no pretrain) | 62.2 | | TPL Table 8, VERIFIED | 2000 exemplars, so not exemplar-free |
| Ours, oracle routing | 75 to 80 | | internal | |
| Ours, routed | 21 to 37 | | internal | |

Things the table does not settle. The same method moves by 5 to 10 points across papers (LwF 32.8 versus 42.6, EFC++ 47.5 versus 52.7, AdaGauss 46.1 versus 53.3), so any comparison within 5 points is protocol noise. The HAT+CSI family uses a wider ResNet-18 and seven times the epochs of the statistics family.

## Ranked hypotheses worth GPU time

1. Add rotation-as-OOD classes to each task's training (4 rotations, 40-way head, ensemble over rotations at test). Reason: the largest single ablation effect in the whole literature (CLOM Table 2 and 4: task detection 59.5 to 66.8 with rotation on top of contrastive; CIL 50.2 to 60.3 rotation versus none). It changes only the per-task training and keeps our circuit and freezing machinery. Cost: one training run.

2. Diagnose the router before changing it. Measure, per task and per sample, three things: batch-of-50 routing accuracy (PR-Ent style), AUROC of our score with own-task test data as in-distribution, and the mean and spread of the score on own-task train versus own-task test. If train scores are much tighter than test scores, the Gaussians are overfit to train features and the fix is to fit statistics on a held-out split or to apply shrinkage and Tukey as in FeCAM. Reason: FeCAM Table 4 shows raw per-class Mahalanobis at 14.6 versus 62.1 with those three fixes; TOOD shows calibration statistics from held-out ID data are what makes per-task scores comparable.

3. Replace the score, cheapest first: (a) relative Mahalanobis (subtract a per-task background Gaussian), (b) tied covariance per task instead of per class, (c) ViM-style residual plus max-logit, (d) energy-based combination of max-logit and Mahalanobis as in TPL Eq. 9 with a uniform complement. Reason: RMD gives plus 6 AUROC on the hardest near-OOD benchmark with no tuning; ViM was the best exemplar-free score in TPL Table 4; the TPL composition gave plus 5 over MLS alone. Expect a few points each, not 30.

4. Per-task score calibration without exemplars. Fit a scale and shift per task on the training data of that task (or a held-out 10 percent) so that own-task scores have median 0 and MAD 1, and recompute after every task since later tasks may drift the shared trunk (though our freezing means old circuits do not change, so this should be a one-time fit). Reason: CLOM gets 2 points from 5 samples per class; TOOD shows recentring is what matters. If our z-standardisation already does this on training data, the check is whether held-out statistics change the result.

5. Supervised contrastive training per task (SupCon with the CSI augmentations) on the circuit's channels, with the head fine-tuned afterwards. Reason: CLOM without rotation but with contrastive still lifts task detection from 34.3 to 59.5. Cost: 700 epochs in their recipe; try 200 first.

6. Exemplar-free surrogate OOD data for training a per-task "other" logit: images from the current task with strong corruption, rotations, mixup between classes, or FGSM adversarials (Lee et al. 2018 used FGSM to tune the Mahalanobis ensemble without real OOD data). Reason: MORE and ROW get 4 to 5 points from real replay OOD data; a surrogate may capture part of it. Riskier than 1 to 4.

7. Route on a concatenation of layers, not only the final circuit feature, with layer weights fit on in-distribution plus FGSM negatives. Reason: Lee et al. 2018 Table 1 shows the feature ensemble alone moves TNR from 54.5 to 91.5; DPCR's ablation shows class-specific subspace information adds 5 points on top of a task-level projection.

8. Only after the above: a matched-sparsity comparison of circuit selection rules (ablation, magnitude, learned scores, gradient) on task-IL and on routing AUROC. Reason: no such comparison exists in the literature, but our oracle result says selection is not the bottleneck, so this is a paper contribution rather than a fix for the gap.

Not recommended: SupSup one-shot entropy over superposed masks (measured worse than max-logit by Kim et al.), per-task VAEs or autoencoders (no evidence on same-dataset CIFAR splits), and test-batch inference as a method (it works but changes the problem).
