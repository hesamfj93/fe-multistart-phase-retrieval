# Stage 2b results (held-out ranks 0 to 99)

Outcome of the plan in `PREREGISTRATION_STAGE2_100.md`. Report: `results/stage2_mnist_heldout/report_n100/`.

## Runs

All on the RTX 4060 Laptop GPU, in the pre-registered order, 0 failures each:

| run | images | finished |
|---|--:|---|
| classical baselines | 100 | 2026-11-11 00:12 |
| K=5 extension (ranks 50 to 99 added) | 100 (50 new) | 02:00 |
| compute-matched control `k1_long` | 100 | 05:37 |

No configuration file changed. The protocol lock was not triggered. No held-out image is a development image.

## Deviations from the pre-registration

1. **Correction of the pixel-wise uncertainty evaluation (evaluation only).** `score_reconstruction` compared the uncertainty map u, which is in the frame of the reconstruction, with the error map, which is in the frame of the ground truth. The flip and shift found by the ground-truth alignment were not applied to u. In all 100 images the two frames differed.
   - Fix: `map_to_gt_frame` in `pr_uq/metrics/alignment.py`, used in `score_reconstruction`. Uncertainty evaluation version 2 is now recorded in `metrics.json`.
   - The finished run was rescored from the stored members with `scripts/rescore_uncertainty.py`. Nothing was retrained.
   - Consistency check, 100 of 100 images: PSNR, SSIM, FE, failure score, anchor and selected start were reproduced. Largest deviation: 2.4e-6 dB PSNR.
   - Affected: only pixel-wise Pearson, Spearman and ECE.
   - Not affected: every reconstruction, PSNR, SSIM, FE, the image-level failure score s (a spatial mean) and therefore H4.
   - `metrics.json` and `aggregates.npz` are unchanged. The corrected values are in `uncertainty_v2.json` and `aggregates_v2.npz`.
2. **Expectation in H3 only partly met.** Real and nonnegative HIO was near-exact as expected. Real and nonnegative RAAR was exact on 80 of 100 images, not on almost all.
3. **Where the report was produced.** The report was generated on a CPU from copies of the GPU results, after rescoring. It can be regenerated on the GPU machine with the commands in `WINDOWS_COMMANDS.md`, section 7.

## Hypotheses

Rule for a claimed improvement: the mean paired PSNR difference is positive, its t-based 95% CI excludes 0, and the Holm-adjusted Wilcoxon p is below 0.05. The Holm family has 25 comparisons.

| # | comparison | mean (dB) | 95% CI | better/worse | p (Holm) | decision |
|---|---|--:|---|--:|--:|---|
| H1a | FE-softmax vs `k1_long` | +0.80 | [0.21, 1.38] | 61/39 | 0.058 | not supported |
| H1b | best-single vs `k1_long` | +0.73 | [0.12, 1.34] | 59/41 | 0.058 | not supported |
| H2a | FE-softmax vs HIO support-only best5 | +3.47 | [2.94, 4.00] | 93/7 | < 1e-6 | supported |
| H2b | FE-softmax vs RAAR support-only best5 | +6.45 | [5.96, 6.94] | 100/0 | < 1e-6 | supported |
| H3 | real+nonneg HIO / RAAR best5 (reference) | exact on 99 / 80 of 100 | | | | no superiority claimed |
| H4 | Spearman(s, -PSNR) of FE-softmax, ranks 50 to 99 | rho = 0.385 | bootstrap [0.086, 0.628] | | one-sided p = 0.0029 | supported |

H1 consequence, as fixed in advance: the gain of the five-start methods over one start is not shown to exceed what one longer start gains. The paper does not claim more efficient use of compute.

Descriptive context, not tested:
- `k1_long` gains 1.03 dB over the 3000-iteration start.
- Failure counts: `k1_long` 12, FE-softmax 3, best-single 5.

## Main numbers (n = 100)

| method | PSNR mean ± std | median | SSIM | failures | exact |
|---|---|--:|--:|--:|--:|
| K=1 | 18.17 ± 3.56 | 18.15 | 0.822 | 20 | 0 |
| uniform | 18.73 ± 2.97 | 18.24 | 0.803 | 5 | 0 |
| median | 19.16 ± 3.04 | 18.81 | 0.869 | 4 | 0 |
| best-single-by-FE | 19.93 ± 3.31 | 19.73 | 0.885 | 5 | 0 |
| FE-softmax | 19.99 ± 3.06 | 19.82 | 0.868 | 3 | 0 |
| `k1_long` | 19.19 ± 3.48 | 19.28 | 0.867 | 12 | 0 |
| HIO support-only best5 | 16.52 ± 2.09 | 16.20 | 0.538 | 22 | 0 |
| RAAR support-only best5 | 13.54 ± 1.96 | 13.03 | 0.239 | 82 | 0 |
| HIO real+nonneg best5 | 59.68 ± 3.17 (capped) | 60.00 | 1.000 | 0 | 99 |
| RAAR real+nonneg best5 | 52.62 ± 15.33 (capped) | 60.00 | 0.959 | 3 | 80 |

**Paired PSNR differences**, mean with 95% CI:
- FE-softmax vs K=1: +1.82 [1.29, 2.36], better on 76 of 100 images.
- Best-single vs K=1: +1.76 [1.21, 2.31], better on 61, worse on 11, equal on 28.
- FE-softmax vs uniform: +1.26 [0.96, 1.57], better on 85.
- FE-softmax vs median: +0.83 [0.58, 1.08], better on 77.
- Best-single vs FE-softmax: -0.07 [-0.24, 0.11], Holm p = 0.54.

**FE as a proxy, over 500 starts:**
- pooled Spearman 0.68;
- within-image Spearman: mean 0.62, median 0.70, negative in 8 images;
- Top-1 agreement 52 of 100 (chance 20);
- Top-2 agreement 82 of 100 (chance 40).

**Effective number of contributing starts (K_eff):** mean 3.26, range 1.11 to 4.95.

**Pixel-wise uncertainty, version 2 (version 1 in brackets).** Each method is reported as the mean pixel-wise Spearman correlation, the mean Pearson correlation, and the ECE:

| method | Spearman | Pearson | ECE |
|---|--:|--:|--:|
| FE-softmax | 0.63 (0.26) | 0.65 (0.20) | 0.101 |
| uniform | 0.65 | 0.71 | 0.100 |
| median | 0.55 | 0.61 | 0.105 |
| best-single (borrowed) | 0.49 | 0.56 | 0.114 |

**Interim gate criterion G6 rechecked with version 2:** ranks 0 to 49 give a FE-softmax mean pixel-wise Spearman of 0.62, against 0.22 under version 1. The gate decision is unchanged.

## Post-hoc analyses (added 2026-11-11, not part of the frozen plan)

Produced by `scripts/posthoc_analyses.py` and `scripts/alignment_sensitivity.py`. They change no decision above.

**Image-level reliability**, Spearman with -PSNR on ranks 50 to 99:

| score | Spearman | interval |
|---|--:|---|
| failure score s (pre-specified) | 0.385 | [0.09, 0.63] |
| FE of the fused reconstruction | 0.81 | [0.69, 0.89] |
| lowest start FE | 0.76 | |

**Pixel-wise uncertainty, FE-softmax.** Spearman correlation with |error| over the whole window and inside the object region (dilated union of supports), and the error-retention area:

| map | whole window | object region | retention area |
|---|--:|--:|--:|
| u | 0.63 | 0.55 | 0.113 |
| sigma_w only | 0.89 | 0.61 | 0.107 |
| residual only | 0.49 | 0.29 | 0.180 |
| image gradient | 0.80 | 0.56 | 0.102 |
| amplitude | 0.99 | 0.48 | 0.111 |
| oracle | | | 0.065 |

Inside the object region:
- sigma_w minus gradient: +0.047, CI [0.012, 0.081], higher in 74 of 100 images, p = 4e-4;
- u minus gradient: -0.009, CI [-0.038, 0.021].

**H1 and H2 by rank half** (sensitivity only; the decisions follow the frozen plan):

| comparison | ranks 0 to 49 | ranks 50 to 99 |
|---|---|---|
| H1a | +0.46 [-0.43, 1.34] | +1.14 [0.36, 1.93] |
| H1b | +0.50 | +0.97 |
| H2a | +3.76 | +3.18 |
| H2b | +6.58 | +6.32 |

Restricting the Holm family or the sample to ranks 50 to 99 would make H1 pass. Neither change is adopted, because it would be decided after seeing the outcome.

**Alignment sensitivity:**
- A single-axis flip was chosen in 1 of 100 ground-truth alignments per fused method, and in 7 of 400 start alignments.
- With only the true ambiguities, all means are unchanged to 0.01 dB.
- With non-wrapping shifts, the largest change is -0.08 dB. The gain of FE-softmax over K=1 becomes +1.73 dB.

**Bootstrap CIs.** Percentile bootstrap CIs of the capped mean PSNR are in `posthoc.json`. Table V of the paper uses them.
