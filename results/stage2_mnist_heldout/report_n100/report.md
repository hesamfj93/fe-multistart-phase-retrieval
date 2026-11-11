# Report: stage2_mnist_heldout (first 100 held-out images)

Sections are separated by MNIST test index. Only the held-out section is used for the paper table.

Failure is defined in advance as PSNR < 15 dB.

Wording. The plain-mean degradation observed during development is a uniform aggregation failure. FE is a practical, empirically evaluated, ground-truth-free reliability proxy, not a guaranteed best-member estimator. Selection and fidelity-weighted fusion are two complementary operating modes.

Extra comparison runs merged by MNIST test index:

- `/tmp/claude-0/-home-claude/141af236-657b-5d0a-81d7-3c2bc4266188/scratchpad/stage2b/stage2_mnist_heldout_k1long`: methods K=1, 15000 it (compute-matched); 100 images merged, 0 failed, devices {'cuda': 100}
- `/tmp/claude-0/-home-claude/141af236-657b-5d0a-81d7-3c2bc4266188/scratchpad/stage2b/stage2_mnist_heldout_baselines`: methods HIO, support only, 1 start, HIO, support only, best of 5 by FE, HIO, real+nonneg, 1 start, HIO, real+nonneg, best of 5 by FE, RAAR, support only, 1 start, RAAR, support only, best of 5 by FE, RAAR, real+nonneg, 1 start, RAAR, real+nonneg, best of 5 by FE; 100 images merged, 0 failed, devices {'cuda': 100}

Pixel-wise uncertainty metrics: evaluation version 2 (2 = u moved into the ground-truth frame before comparison with the error; 1 = misregistered, superseded).

## Held-out evaluation (primary)

PRIMARY RESULTS. Held-out MNIST test images only; none of them was used to select the aggregation method or T = 0.05. These numbers form the paper table.

Images: 100. Devices: {'cuda': 100}.

PSNR is capped at 60 dB in all summary statistics and paired differences (a reconstruction at or above 60 dB is exact up to floating-point error); per-image CSVs keep raw values.

| method | n | PSNR mean ± std | PSNR median | PSNR 95% CI | SSIM mean ± std | FE mean | runtime (s) | failures | exact |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| K=1 | 100 | 18.17 ± 3.56 | 18.15 | [17.46, 18.87] | 0.822 ± 0.120 | 0.266 | 25.9 | 20/100 | 0/100 |
| K=5 uniform | 100 | 18.73 ± 2.97 | 18.24 | [18.14, 19.32] | 0.803 ± 0.116 | 0.302 | 129.1 | 5/100 | 0/100 |
| K=5 median | 100 | 19.16 ± 3.04 | 18.81 | [18.56, 19.76] | 0.869 ± 0.067 | 0.265 | 129.1 | 4/100 | 0/100 |
| K=5 best-single-by-FE | 100 | 19.93 ± 3.31 | 19.73 | [19.27, 20.58] | 0.885 ± 0.081 | 0.208 | 129.1 | 5/100 | 0/100 |
| K=5 FE-softmax (T=0.05) | 100 | 19.99 ± 3.06 | 19.82 | [19.39, 20.60] | 0.868 ± 0.090 | 0.239 | 129.1 | 3/100 | 0/100 |
| K=1, 15000 it (compute-matched) | 100 | 19.19 ± 3.48 | 19.28 | [18.50, 19.88] | 0.867 ± 0.097 | 0.220 | 129.6 | 12/100 | 0/100 |
| HIO, support only, 1 start | 100 | 15.87 ± 2.12 | 15.76 | [15.45, 16.29] | 0.486 ± 0.114 | 0.006 | 1.7 | 35/100 | 0/100 |
| HIO, support only, best of 5 by FE | 100 | 16.52 ± 2.09 | 16.20 | [16.11, 16.94] | 0.538 ± 0.096 | 0.004 | 8.6 | 22/100 | 0/100 |
| HIO, real+nonneg, 1 start | 100 | 50.22 ± 17.46 | 60.00 | [46.76, 53.69] | 0.929 ± 0.166 | 0.009 | 2.0 | 10/100 | 73/100 |
| HIO, real+nonneg, best of 5 by FE | 100 | 59.68 ± 3.17 | 60.00 | [59.05, 60.31] | 1.000 ± 0.003 | 0.000 | 10.2 | 0/100 | 99/100 |
| RAAR, support only, 1 start | 100 | 12.95 ± 1.74 | 12.70 | [12.60, 13.29] | 0.191 ± 0.059 | 0.022 | 2.2 | 89/100 | 0/100 |
| RAAR, support only, best of 5 by FE | 100 | 13.54 ± 1.96 | 13.03 | [13.15, 13.93] | 0.239 ± 0.070 | 0.016 | 10.8 | 82/100 | 0/100 |
| RAAR, real+nonneg, 1 start | 100 | 34.82 ± 20.83 | 23.17 | [30.68, 38.95] | 0.807 ± 0.215 | 0.027 | 2.4 | 21/100 | 37/100 |
| RAAR, real+nonneg, best of 5 by FE | 100 | 52.62 ± 15.33 | 60.00 | [49.58, 55.66] | 0.959 ± 0.115 | 0.005 | 11.8 | 3/100 | 80/100 |

Paired PSNR differences on identical images (method minus baseline, capped PSNR). Holm adjustment over all comparisons in this table:

| method vs baseline | n | mean Δ | median Δ | 95% CI | better / worse / tie | Holm p |
|---|--:|--:|--:|--:|--:|--:|
| K=5 best-single-by-FE vs K=1 | 100 | +1.76 | +0.66 | [1.21, 2.31] | 61 / 11 / 28 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs K=1 | 100 | +1.82 | +1.16 | [1.29, 2.36] | 76 / 24 / 0 | 0.0000 |
| K=5 uniform vs K=1 | 100 | +0.56 | +0.42 | [0.16, 0.97] | 62 / 38 / 0 | 0.0177 |
| K=5 median vs K=1 | 100 | +0.99 | +0.74 | [0.52, 1.46] | 67 / 33 / 0 | 0.0004 |
| K=5 FE-softmax (T=0.05) vs K=5 uniform | 100 | +1.26 | +0.86 | [0.96, 1.57] | 85 / 15 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs K=5 median | 100 | +0.83 | +0.53 | [0.58, 1.08] | 77 / 23 / 0 | 0.0000 |
| K=5 best-single-by-FE vs K=5 FE-softmax (T=0.05) | 100 | -0.07 | -0.02 | [-0.24, 0.11] | 48 / 52 / 0 | 0.5383 |
| K=5 FE-softmax (T=0.05) vs K=1, 15000 it (compute-matched) | 100 | +0.80 | +0.77 | [0.21, 1.38] | 61 / 39 / 0 | 0.0576 |
| K=5 best-single-by-FE vs K=1, 15000 it (compute-matched) | 100 | +0.73 | +0.62 | [0.12, 1.34] | 59 / 41 / 0 | 0.0576 |
| K=5 FE-softmax (T=0.05) vs HIO, support only, 1 start | 100 | +4.12 | +3.68 | [3.55, 4.70] | 95 / 5 / 0 | 0.0000 |
| K=5 best-single-by-FE vs HIO, support only, 1 start | 100 | +4.06 | +3.34 | [3.43, 4.68] | 92 / 8 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs HIO, support only, best of 5 by FE | 100 | +3.47 | +3.20 | [2.94, 4.00] | 93 / 7 / 0 | 0.0000 |
| K=5 best-single-by-FE vs HIO, support only, best of 5 by FE | 100 | +3.41 | +3.16 | [2.82, 3.99] | 91 / 9 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs HIO, real+nonneg, 1 start | 100 | -30.23 | -39.38 | [-33.77, -26.69] | 18 / 82 / 0 | 0.0000 |
| K=5 best-single-by-FE vs HIO, real+nonneg, 1 start | 100 | -30.30 | -39.61 | [-33.85, -26.75] | 16 / 84 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs HIO, real+nonneg, best of 5 by FE | 100 | -39.69 | -40.18 | [-40.64, -38.74] | 0 / 100 / 0 | 0.0000 |
| K=5 best-single-by-FE vs HIO, real+nonneg, best of 5 by FE | 100 | -39.76 | -40.27 | [-40.73, -38.78] | 0 / 100 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs RAAR, support only, 1 start | 100 | +7.04 | +6.60 | [6.52, 7.57] | 100 / 0 / 0 | 0.0000 |
| K=5 best-single-by-FE vs RAAR, support only, 1 start | 100 | +6.98 | +6.51 | [6.40, 7.55] | 100 / 0 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs RAAR, support only, best of 5 by FE | 100 | +6.45 | +6.42 | [5.96, 6.94] | 100 / 0 / 0 | 0.0000 |
| K=5 best-single-by-FE vs RAAR, support only, best of 5 by FE | 100 | +6.38 | +6.15 | [5.85, 6.92] | 100 / 0 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs RAAR, real+nonneg, 1 start | 100 | -14.82 | -3.52 | [-18.82, -10.83] | 40 / 60 / 0 | 0.0000 |
| K=5 best-single-by-FE vs RAAR, real+nonneg, 1 start | 100 | -14.89 | -3.31 | [-18.89, -10.90] | 34 / 66 / 0 | 0.0000 |
| K=5 FE-softmax (T=0.05) vs RAAR, real+nonneg, best of 5 by FE | 100 | -32.63 | -39.31 | [-35.58, -29.67] | 8 / 92 / 0 | 0.0000 |
| K=5 best-single-by-FE vs RAAR, real+nonneg, best of 5 by FE | 100 | -32.69 | -39.41 | [-35.63, -29.76] | 6 / 94 / 0 | 0.0000 |

FE as a reliability proxy: pooled Spearman 0.684, within-image mean 0.620 / median 0.700, negative within-image correlation in 8 images, Top-1 52/100, Top-2 82/100.

| method | uncertainty source | Pearson | Spearman | ECE | failure AUROC |
|---|---|--:|--:|--:|--:|
| K=5 uniform | ensemble (intrinsic) | 0.711 | 0.652 | 0.100 | 0.838 |
| K=5 median | ensemble (intrinsic) | 0.608 | 0.549 | 0.105 | not computed |
| K=5 best-single-by-FE | aligned K=5 stack (borrowed, not intrinsic) | 0.560 | 0.493 | 0.114 | 0.735 |
| K=5 FE-softmax (T=0.05) | ensemble (intrinsic) | 0.649 | 0.625 | 0.101 | not computed |

Effective ensemble size (FE-softmax): mean 3.26, median 3.24, range [1.11, 4.95].

Note. Best-single-by-FE has no intrinsic ensemble uncertainty. Its uncertainty values are derived from the full aligned K=5 stack and are not a like-for-like uncertainty output of the selected single reconstruction.

### Pre-registered confirmatory test (image-level reliability)

Held-out ranks 50-99 only; hypothesis: Spearman(failure_score, -PSNR) > 0, one-sided; primary method K=5 FE-softmax (T=0.05); alpha 0.05. Status: computed (50/50 images).

| method | role | rho | one-sided p | bootstrap 95% CI |
|---|---|--:|--:|--:|
| K=5 FE-softmax (T=0.05) | primary | 0.385 | 2.91e-03 | [0.086, 0.628] |
| K=5 uniform | secondary (descriptive) | 0.657 | 1.11e-07 | [0.411, 0.825] |
| K=5 median | secondary (descriptive) | 0.524 | 4.65e-05 | [0.251, 0.738] |
| K=5 best-single-by-FE | secondary (descriptive) | 0.294 | 1.92e-02 | [-0.018, 0.558] |

Primary hypothesis SUPPORTED at alpha 0.05.

## Development / validation subset

Not included in this run directory. The development-stage analysis (CPU, z_channels = 64) is in AGGREGATION_FINALIZATION_REPORT.md and is not directly comparable to Stage-2 settings. Optionally run configs/stage2_mnist_dev8.yaml and pass --dev-run-dir to report it here under identical settings.

## Integrity checks

```
{
  "heldout_images_with_dev_index": 0,
  "development_images_found": [],
  "other_images_found": [],
  "heldout_beyond_n": [],
  "runner_failures": [],
  "heldout_missing_ranks": [],
  "heldout_missing_extra_methods": {
    "k1_long": [],
    "hio_supp_s1": [],
    "hio_supp_best5": [],
    "hio_realpos_s1": [],
    "hio_realpos_best5": [],
    "raar_supp_s1": [],
    "raar_supp_best5": [],
    "raar_realpos_s1": [],
    "raar_realpos_best5": []
  },
  "uncertainty_eval_versions_heldout": [
    2
  ]
}
```
