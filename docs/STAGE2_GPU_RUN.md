# Stage 2: evaluation protocol and decision gate

This file defines **what** Stage 2 evaluates and **how the result will be judged**. It contains no commands. The authoritative instructions are **WINDOWS_COMMANDS.md** (exact CMD commands) and **RUNBOOK.md** (workflow and output layout).

## 1. Image sets (MNIST test split, 28x28 padded to 32x32)

| set | file | size | role |
|---|---|--:|---|
| development | `splits/mnist_test_dev_indices.csv` | 8 | Used to select the aggregation method and T = 0.05. Never part of the paper table. |
| held-out | `splits/mnist_test_heldout_indices.csv` | 200 rows, first 50 used (then 100) | The only images in the primary paper table. |
| smoke | `splits/mnist_test_smoke_indices.csv` | 3 | Software verification only. |

- The held-out list is a fixed random permutation of the 10,000 test images (NumPy `RandomState(20261006)`) with the 8 development indices removed. It is ordered. The 50-image run uses rows 0 to 49; the 100-image run uses rows 0 to 99, so the first 50 never change.
- The three sets are pairwise disjoint. This is checked by `scripts/make_splits.py --check`, by the unit tests, at run start (the runner aborts on any overlap), and again by the report generator.
- Development indices: 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399.

## 2. Frozen design (fixed before any held-out image is reconstructed)

Defined in `configs/stage2_mnist_heldout.yaml`. The runner writes `protocol_lock.json` on the first run and refuses to continue in the same output directory if any of these change. Only `n_images` (50 to 100), the device and the output path may differ between sessions.

- Untrained double decoder, z_channels = 128, base_ch = 128, 3000 iterations, clean 2x-oversampled Fourier intensity.
- K = 5 members with seeds 0 to 4. K=1 is member 0 (seed 0), so all five methods share identical trained members.
- Ambiguity-aware alignment: circular shift, flip, conjugate (twin) and global phase.
- Five methods, all computed from the same stack:
  1. K=1
  2. K=5 uniform mean
  3. K=5 pixel-wise median
  4. K=5 best-single-by-FE
  5. K=5 best-FE anchor with FE-softmax fusion, T = 0.05
- Uncertainty: ensemble standard deviation fused with the physics residual (alpha = 0.5). Best-single-by-FE has no intrinsic uncertainty; any uncertainty shown beside it is borrowed from the aligned K=5 stack.
- Failure is defined as PSNR < 15 dB. Failure-detection AUROC is reported only if there are at least 5 failures and at least 5 successes.

No hyperparameter, temperature or method choice may be changed after looking at held-out results. If a change is ever needed, it requires a new development cycle and a new, unused held-out set.

## 3. What the report contains (held-out section only for the paper)

Mean, standard deviation, median, min, max and t-based 95% CI for PSNR, SSIM, FE and runtime; paired differences on identical images (mean, median, 10% trimmed mean, 95% CI, better/worse counts, uncorrected Wilcoxon p); failure rate; FE as a reliability proxy (pooled and within-image Spearman, Top-1 and Top-2 agreement); uncertainty Pearson, Spearman and ECE; effective ensemble size.

## 4. Decision gate (applied to the 50 held-out images)

Each criterion is answered yes or no from the held-out tables before deciding anything else.

| # | question | evidence |
|---|---|---|
| G1 | Does best-single-by-FE improve over K=1? | paired PSNR: 95% CI above 0, median difference > 0, better in a clear majority of images |
| G2 | Does FE-softmax improve over uniform and over median? | paired PSNR versus each, same rule as G1 |
| G3 | Is any improvement not driven by a few images? | median and 10% trimmed-mean differences agree in sign with the mean |
| G4 | Do the confidence intervals support the claim? | CI of the paired difference excludes 0 for every claimed improvement |
| G5 | Does FE keep useful within-image ranking ability? | mean within-image Spearman > 0 and Top-2 agreement well above the 40% expected by chance for K=5 |
| G6 | Is the uncertainty ranking meaningful? | mean pixel-wise Spearman(uncertainty, error) clearly above 0 for FE-softmax |
| G7 | Is the failure rate acceptable? | failure rate of the recommended mode not higher than K=1, runner failures 0 |

Outcomes:

- G1, G3, G4, G7 pass and G2 does not: lead with measurement-guided multi-start **selection**; present fusion as the mode that adds an uncertainty map.
- G2, G3, G4, G6, G7 pass: lead with **fidelity-weighted uncertainty-aware fusion**, with selection as the strongest reconstruction-only baseline.
- G4 fails for every comparison: do not run 100 images or new datasets; report as inconclusive and revisit the method.
- Only after a positive, stable outcome: extend to 100 held-out images, then Fashion-MNIST (with its own development and held-out split), then CIFAR-10 (after an explicit decision on the colour formulation).

## 5. Stage 2b

The 50-image gate passed (`STAGE2_DECISION_GATE_50.md`). The extension to 100 images, the compute-matched control, the classical baselines and the confirmatory image-level test are pre-registered in `PREREGISTRATION_STAGE2_100.md`.
