# Stage 2 decision gate: 50 held-out MNIST images

Run: `results/stage2_mnist_heldout`, report `report_n50/`. Gate definitions: `STAGE2_GPU_RUN.md` section 4, fixed before the run.

## 1. Integrity

| check | result |
|---|---|
| images completed | 50 / 50, `status: ok`, 0 runner failures |
| device | CUDA for all 50 (RTX 4060 Laptop, torch 2.5.1+cu121, Python 3.11.9) |
| image set | exactly held-out ranks 0 to 49; overlap with the 8 development images: none |
| frozen design | T = 0.05, K = 5, 3000 iterations, z_channels = 128, failure threshold 15 dB; protocol lock unchanged |
| members | all 250 trained once in the first session; the aggregation step was rerun from the cached members after the CUDA device fix (training code and settings unchanged) |

## 2. Primary table (held-out, n = 50)

| method | PSNR mean ± std | median | 95% CI | SSIM | FE | runtime (s) | failures (< 15 dB) |
|---|--:|--:|--:|--:|--:|--:|--:|
| K=1 | 18.43 ± 3.86 | 18.55 | [17.34, 19.53] | 0.822 | 0.260 | 25.9 | 10 / 50 |
| K=5 uniform | 19.04 ± 3.05 | 18.50 | [18.17, 19.91] | 0.814 | 0.291 | 129.4 | 3 / 50 |
| K=5 median | 19.47 ± 3.14 | 19.19 | [18.57, 20.36] | 0.875 | 0.255 | 129.4 | 2 / 50 |
| K=5 best-single-by-FE | 20.42 ± 3.43 | 20.06 | [19.45, 21.39] | 0.896 | 0.197* | 129.4 | 2 / 50 |
| K=5 FE-softmax (T = 0.05) | 20.38 ± 3.08 | 19.97 | [19.50, 21.25] | 0.880 | 0.226 | 129.4 | 1 / 50 |

\* FE is the selection criterion of best-single-by-FE, so its FE value is low by construction and is not an independent quality measure for that row.

## 3. Paired PSNR differences (same images)

| comparison | mean | median | 10% trimmed | mean without top 5 | better / worse / tie | Holm-adjusted Wilcoxon p |
|---|--:|--:|--:|--:|--:|--:|
| best-single-by-FE vs K=1 | +1.99 | +1.38 | +1.73 | +1.34 | 33 / 5 / 12 | 7.5e-06 |
| FE-softmax vs K=1 | +1.94 | +1.28 | +1.63 | +1.30 | 39 / 11 / 0 | 6.0e-06 |
| uniform vs K=1 | +0.60 | +0.42 | +0.58 | +0.23 | 30 / 20 / 0 | 5.9e-02 |
| median vs K=1 | +1.03 | +0.91 | +0.88 | +0.53 | 34 / 16 / 0 | 3.3e-03 |
| FE-softmax vs uniform | +1.34 | +1.12 | +1.13 | +0.93 | 41 / 9 / 0 | 7.6e-08 |
| FE-softmax vs median | +0.91 | +0.55 | +0.67 | +0.57 | 40 / 10 / 0 | 1.0e-08 |
| best-single-by-FE vs FE-softmax | +0.04 | +0.10 | +0.04 | -0.13 | 27 / 23 / 0 | 7.7e-01 |

The 12 ties for best-single vs K=1 are the images where member 0 (the K=1 run) had the lowest FE. The best-FE member index is spread evenly over the five seeds (12, 7, 11, 10, 10).

## 4. Gate

| # | criterion | evidence | verdict |
|---|---|---|---|
| G1 | best-single-by-FE improves over K=1 | CI [1.17, 2.80], median +1.38, 33 better / 5 worse | pass |
| G2 | FE-softmax improves over uniform and over median | +1.34 [0.86, 1.82], 41/9; +0.91 [0.55, 1.28], 40/10 | pass |
| G3 | not driven by a few images | median, trimmed mean and mean without the top 5 agree in sign for every claimed improvement | pass |
| G4 | confidence intervals support the claims | every claimed improvement excludes 0; Holm-adjusted p < 0.01 | pass |
| G5 | FE keeps within-image ranking ability | within-image Spearman mean 0.684 (median 0.750, 1 image negative); Top-1 29/50 (chance 10/50); Top-2 43/50 (chance 20/50) | pass |
| G6 | uncertainty ranking meaningful | FE-softmax pixel Spearman 0.220, 95% CI [0.163, 0.278], positive in 41/50 images | pass, weak |
| G7 | failure rate acceptable | FE-softmax 1/50 and best-single 2/50 versus K=1 10/50; runner failures 0 | pass |

Outcome under the pre-specified rule: G2, G3, G4, G6 and G7 pass, so the paper leads with **fidelity-weighted uncertainty-aware fusion**, with **best-single-by-FE selection as the strongest reconstruction-only baseline**.

## 5. Limits of these results

1. **Compute is not matched.** Every K=5 method costs about 5 times K=1 (129 s versus 26 s per image). The comparison shows what five measurement-guided starts buy over one start; it does not show that this beats spending the same compute on one longer run. A compute-matched single-start control is needed before claiming an efficiency advantage.
2. **Selection and fusion are equivalent in PSNR** (+0.04 dB, p = 0.77). What fusion adds over selection is an uncertainty map and one fewer failure (1 versus 2, too few to compare).
3. **The uncertainty is weakly informative.** Pixel-wise Spearman 0.22 and Pearson 0.17 are above zero but small, and FE-softmax is not better than uniform on this measure (+0.029, 26/50 images, p = 0.63). G6 passes only in the weak sense of being reliably above zero.
4. **Failure-detection AUROC was not computed**, as pre-specified, because the ensemble methods have fewer than 5 failures.
5. **Exploratory, not pre-specified:** the image-level failure score correlates with low PSNR (Spearman 0.56 for FE-softmax, 0.72 for uniform). This suggests the image-level reliability signal is stronger than the pixel-level one, but it was found after looking at the data. It must be confirmed on images not yet seen (held-out ranks 50 to 99) before being claimed.
6. **Consistency with development:** FE-softmax minus K=1 was +2.04 dB on the 8 development images (z_channels 64, CPU) and is +1.94 dB here (z_channels 128, 50 held-out images). Direction and size agree, which argues against overfitting to the development images, but settings differ, so this is not a direct replication.
7. **Hard cases remain.** #06402 fails in every method because none of its five members reached a good solution; selection and weighting cannot recover what no member found.
8. Figure `qualitative_grid.png` shows the correct five images (worst, quartiles and best by FE-softmax PSNR), but its rows are in list order rather than sorted by PSNR. Cosmetic; to be fixed.

## 6. Recommendation

- The 50-image results justify extending to 100 held-out images.
- Before that run, decide and freeze two additions so they are evaluated on the same 100 images:
  - a compute-matched single-start control (one start with about 5 times the iterations);
  - the classical HIO and RAAR baselines.
- Pre-specify a confirmatory test for the image-level reliability signal on ranks 50 to 99 only, before those images are reconstructed.
- Do not change T, K, the methods or any reconstruction setting.
