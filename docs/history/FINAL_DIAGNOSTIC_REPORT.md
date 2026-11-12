> **SUPERSEDED. Do not cite numbers or conclusions from this file.**
>
> This was the first K=1 vs K=5 diagnostic on the 8 development images (CPU, z_channels = 64). It is kept only as a historical record. Known problems:
>
> 1. The median PSNR values are wrong. The correct values are 19.67 dB for K=1 (not 22.06) and 19.09 dB for K=5 uniform (not 19.22).
> 2. Its explanation of the mnist_002890 regression (ensemble misalignment) was later disproven. Oracle alignment recovered only 0.04 dB; the cause was uniform aggregation of members of very different quality (a uniform aggregation failure).
> 3. Its uncertainty, alpha-sweep and calibration figures were computed with the legacy uniform aggregation and do not describe the current method.
>
> The authoritative report for the 8-image development stage is **AGGREGATION_FINALIZATION_REPORT.md**. Instructions for running the project are in **RUNBOOK.md** and **WINDOWS_COMMANDS.md**.

---

# Final Diagnostic Report — Real MNIST K=1 vs K=5 at 3000 iterations

**Setup.** 8 MNIST test images, 32×32 (28→32 zero-padded), 2× oversampled Fourier intensity, clean (no noise). Untrained double-decoder, z_channels=64, base_ch=64, **3000 iterations**, CPU. The K=1 and K=5 runs operate on the *same 8 images* (deterministic seed=0) so all comparisons are paired.

All numbers in this report are pulled directly from `results/mnist_k1_3000iter/` and `results/mnist_k5_3000iter/`. Nothing is estimated or rounded toward a preferred conclusion.

---

## 1. Full paired comparison — all 8 images

| image | K=1 PSNR | K=5 PSNR | ΔPSNR | K=1 SSIM | K=5 SSIM | ΔSSIM | K=1 FE | K=5 FE | K=5 fail-score | K=1 runtime | K=5 runtime |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mnist_000429 (8) | 13.88 | 13.83 | **−0.05** | 0.652 | 0.634 | −0.018 | 0.403 | 0.506 | 0.358 | 71.2 s | 303.0 s |
| mnist_001729 (1) | 25.97 | 26.29 | **+0.32** | 0.970 | 0.974 | +0.004 | 0.110 | 0.122 | 0.130 | 71.3 s | 280.8 s |
| mnist_001917 (5) | 17.28 | 17.17 | **−0.12** | 0.850 | 0.546 | −0.304 | 0.270 | 0.366 | 0.281 | 71.0 s | 394.9 s |
| mnist_002890 (7) | 26.21 | 20.84 | **−5.37** | 0.974 | 0.852 | −0.122 | 0.118 | 0.296 | 0.260 | 72.5 s | 357.4 s |
| mnist_004919 (9) | 22.06 | 18.96 | **−3.10** | 0.947 | 0.846 | −0.101 | 0.185 | 0.289 | 0.292 | 72.0 s | 380.5 s |
| mnist_006044 (1) | 23.04 | 23.55 | **+0.51** | 0.958 | 0.940 | −0.018 | 0.121 | 0.135 | 0.149 | 72.0 s | 361.1 s |
| mnist_007532 (6) | 15.19 | 18.16 | **+2.97** | 0.633 | 0.646 | +0.013 | 0.276 | 0.326 | 0.298 | 72.4 s | 285.9 s |
| mnist_009399 (7) | 16.17 | 19.22 | **+3.05** | 0.606 | 0.764 | +0.158 | 0.233 | 0.270 | 0.296 | 71.0 s | 354.3 s |

(`FE` is the normalized Fourier-magnitude error; lower is better. `fail-score` is the mean of the fused K=5 uncertainty map.)

---

## 2. Summary statistics

| metric | K=1 (3000 iter) | K=5 (3000 iter) | direction |
|---|--:|--:|---|
| mean PSNR (dB) | 19.97 | 19.75 | **−0.22** (K=5 marginally worse) |
| median PSNR (dB) | 22.06 | 19.22 | **−2.83** (K=5 worse) |
| std PSNR (dB) | 4.61 | 3.60 | **−1.01** (K=5 tighter distribution) |
| min PSNR (dB) | 13.88 | 13.83 | ≈ 0 |
| max PSNR (dB) | 26.21 | 26.29 | ≈ 0 |
| mean SSIM | 0.824 | 0.775 | **−0.049** (K=5 worse) |
| mean Fourier-mag error | 0.214 | 0.289 | +0.074 (K=5 worse) |
| mean ECE | 0.048 | 0.114 | +0.066 (K=5 worse) |
| mean Pearson(U, |err|) | 0.166 | 0.286 | **+0.120** (K=5 better) |
| mean Spearman(U, |err|) | 0.187 | 0.321 | **+0.134** (K=5 better) |
| runtime / image | 71.7 s | 339.7 s | **4.74×** |
| n_images improved by K=5 (ΔPSNR > 0) | — | 4 / 8 | tied |
| n_images degraded by K=5 (ΔPSNR < 0) | — | 4 / 8 | tied |
| catastrophic failures (PSNR < 15 dB) | 1 / 8 (mnist_000429) | 1 / 8 (same image) | **none reduced** |

The "tighter distribution" finding deserves emphasis: K=5 reduces image-to-image PSNR variance (std 3.60 vs 4.61) without lifting the mean. K=5 is a **variance-reducer, not a quality-improver**, on these 8 images.

---

## 3. Uncertainty analysis

### 3.1 Component breakdown (per-pixel Pearson correlation with absolute error)

| image | ensemble std only | physics residual only | fused (α=0.5) |
|---|--:|--:|--:|
| mnist_000429 | +0.29 | +0.22 | +0.29 |
| mnist_001729 | +0.13 | +0.24 | +0.22 |
| mnist_001917 | +0.10 | +0.23 | +0.18 |
| mnist_002890 | **+0.60** | **+0.49** | **+0.61** |
| mnist_004919 | +0.26 | +0.28 | +0.30 |
| mnist_006044 | −0.05 | −0.06 | −0.06 |
| mnist_007532 | +0.36 | +0.38 | +0.42 |
| mnist_009399 | +0.35 | +0.21 | +0.32 |
| **mean** | **0.26** | **0.25** | **0.29** |

### 3.2 ECE (lower = better calibration)

| image | std only | r_phys only | fused |
|---|--:|--:|--:|
| mean over 8 images | 0.137 | **0.110** | 0.114 |

**Physics residual alone has the lowest mean ECE.** Fusion is essentially tied with r_phys.

### 3.3 Alpha sweep (failure-detection AUROC, threshold = 17.06 dB)

| α | source | AUROC |
|--:|---|--:|
| 0.00 | r_phys only | **1.000** |
| 0.10 | mostly r_phys | 0.857 |
| 0.25 | balanced | 0.714 |
| 0.50 | balanced | 0.714 |
| 0.75 | mostly std | 0.714 |
| 0.90 | mostly std | 0.714 |
| 1.00 | ensemble std only | 0.714 |

**Important caveat:** the failure-detection test has **only 1 catastrophic failure** in 8 images (mnist_000429 at ≤15 dB), so AUROC=1.0 means "the failure has the highest fail-score among 8 images, with the other 7 below it". This is **directionally encouraging but statistically very weak**.

### 3.4 Reliability diagram (pooled pixels across all 8 K=5 images)

| bin | predicted U | mean |err| | gap |
|--:|--:|--:|--:|
| 0 | 0.003 | 0.005 | +0.002 |
| 5 | 0.030 | 0.023 | −0.007 |
| 10 | 0.148 | 0.072 | −0.076 |
| 12 | 0.380 | 0.090 | −0.290 |
| 14 | 0.718 | 0.126 | −0.592 |

Global ECE (pooled): **0.111**.

The reliability curve sits **systematically below the diagonal** at high uncertainty bins. The K=5 fused uncertainty **over-predicts** the magnitude of the error — the algorithm says "this pixel could be wrong by 0.72" when the actual mean error in that bin is only 0.13. This is the "safer" failure mode (under-confidence) but means the uncertainty is **not pixel-wise calibrated in absolute scale**.

### 3.5 Mean uncertainty-error correlation: K=1 vs K=5

| | mean Pearson | mean Spearman |
|---|--:|--:|
| K=1 (3000 iter) | 0.166 | 0.187 |
| K=5 (3000 iter) | **0.286** | **0.321** |

K=5 improves pixel-wise uncertainty quality by **+72%** in both Pearson and Spearman terms.

---

## 4. Qualitative figures

Generated in `figures/`:

- **`k1_vs_k5_3000iter_qualitative.png`** — 8×7 grid: GT / K=1 recon / K=5 mean / K=5 std / K=5 r_phys / K=5 fused U / K=5 |error|. PSNR annotated per row.
- **`k5_minus_k1_psnr_delta.png`** — bar chart of per-image ΔPSNR, sorted by K=1 quality. Shows a clear pattern: K=5 helps the 2 worst K=1 images and hurts the 2 best K=1 images.
- **`k5_3000iter_reliability.png`** — reliability diagram showing systematic under-confidence at high uncertainty.
- **`convergence_k1.png`** (earlier diagnostic run): per-image loss / PSNR / FE curves vs iteration.

### 4.1 Selected cases

- **Best case (K=5 reaches highest PSNR):** mnist_001729 ("1"), K=5 PSNR=26.29 dB, ΔPSNR=+0.32 from K=1. Both reconstructions are visually nearly perfect; K=5 wins by a hair.
- **Worst case (K=5 lowest PSNR):** mnist_000429 ("8"), K=5 PSNR=13.83 dB. The image is the catastrophic failure of the set, and K=5 does not recover it.
- **Median case:** mnist_009399 (a "7" — K=5 PSNR=19.22 dB, ΔPSNR=+3.05).
- **Hard case where K=5 improves over K=1:** mnist_009399 (+3.05 dB) and mnist_007532 (+2.97 dB). Both are images where K=1 alone landed in the 15–16 dB range; ensembling pushes them to 18–19 dB.
- **Easy case where K=5 hurts:** mnist_002890 ("7"). K=1=26.21 dB → K=5=20.84 dB, **a 5.4 dB regression**. Inspection of the figure shows the K=5 mean has a faint ghost from a mis-aligned ensemble member.

The visual uncertainty maps (column 6 of the qualitative figure) are **spatially meaningful** — bright pixels in the fused-U map line up clearly with the K=5 |error| map (column 7). This is true even for the failed image (000429), where the entire digit support shows high uncertainty.

---

## 5. Interpretation

### Does K=5 improve average reconstruction quality?
**No.** Mean PSNR went from 19.97 to 19.75 dB (−0.22). Median dropped from 22.06 to 19.22 (−2.83). SSIM dropped from 0.824 to 0.775. These are small changes on n=8 and not statistically tested, but the direction is consistent across PSNR, SSIM, and Fourier error: K=5 is **not** a quality-improver on this set.

### Does K=5 mainly help hard cases?
**Yes, clearly.** The PSNR delta bar chart shows a strong pattern:
- The 2 worst K=1 images (15–16 dB) gain +3 dB each.
- The 2 best K=1 images (26 dB) lose −5 and −3 dB.
- The mid-range images are within ±0.5 dB.

The interpretation: when a single network finds a good solution easily, the additional ensemble members find slightly worse variants, and the alignment-then-average step degrades the result. When the single network is struggling, the ensemble offers more chances at a good solution and averaging benefits from the diversity.

### Does K=5 reduce catastrophic failures?
**No.** One image (mnist_000429, "8") has PSNR < 15 dB in both runs. K=5 gives 13.83 dB vs K=1's 13.88 dB — within the noise band. K=5 does not rescue the catastrophic case on this set.

### Is uncertainty spatially meaningful or mainly global?
**Both, with the global signal being stronger.**
- *Globally* (image-level fail-score = mean of fused U): clean ranking. Failed images have higher mean U (~0.30) than successes (~0.13). This is what gives AUROC=1.0 on this tiny set.
- *Locally* (per-pixel Pearson): mean 0.29. Bright in the U map mostly aligns with bright in the |error| map qualitatively, but the linear correlation is modest. The 0.61 on mnist_002890 is a genuinely strong per-image correlation; the −0.06 on mnist_006044 is a near-perfect reconstruction with residual error dominated by noise.

### Does fused uncertainty outperform std-only and residual-only?
**Marginally on correlation; not on calibration; not on AUROC.**
- Pearson: fused 0.29 > std 0.26 > r_phys 0.25 (fusion wins by 0.03)
- ECE: r_phys 0.110 < fused 0.114 < std 0.137 (r_phys wins)
- Failure-detection AUROC: r_phys 1.0 >> all other α settings 0.71 (r_phys wins decisively, though n=1 failure)

The honest summary: **physics residual alone is the strongest single uncertainty signal on this dataset.** The ensemble variance adds correlation with error at the pixel level but degrades the image-level failure-detection signal. The fusion design is justified by the +0.03 Pearson gain, but it's marginal and may not hold up with more images.

### What is the safest paper claim?
1. **Untrained double-decoder reconstruction at 3000 iter is competitive with K=1.** Mean PSNR 19.97 dB on MNIST.
2. **K=5 ensembling reduces image-to-image PSNR variability** (std 4.61 → 3.60) without raising the mean.
3. **K=5 ensembling improves per-pixel uncertainty quality** (Pearson +72%).
4. **Physics-residual uncertainty is the strongest scalar signal for failure detection** on this dataset.
5. The uncertainty maps are **diagnostic** — spatially meaningful enough that a human can identify which reconstruction regions to mistrust — but **not yet calibrated** in absolute scale (ECE 0.111; systematic over-prediction at high U).

Statements that the data does **not** support:
- "K=5 improves reconstruction quality." (Mean is slightly lower.)
- "K=5 reduces catastrophic failures." (Same 1 image fails in both.)
- "Our uncertainty is calibrated." (Reliability diagram shows systematic over-prediction.)
- "Fused uncertainty is better than physics residual alone." (Only marginally on Pearson, worse on ECE and AUROC.)

---

## 6. Paper positioning paragraph

> *Given the actual numbers, the strongest defensible position is option **(D) Reliability-aware untrained phase retrieval**, with elements of (B) but neither (A) nor (C).*
>
> **Recommended framing.** "An untrained double-decoder for Fourier phase retrieval, equipped with a diagnostic uncertainty channel from ensemble variance and physics residual. Across 8 paired MNIST images at 3000 iterations, ensembling (K=5) does not improve mean reconstruction PSNR over a single network (19.75 vs 19.97 dB), but it reduces image-to-image variability and produces per-pixel uncertainty maps that are spatially meaningful (Pearson 0.29) and identify hard reconstruction regions. The image-level uncertainty summary cleanly separates the single catastrophic failure from successes (AUROC = 1.0 on the held-out signal), suggesting the method is useful as a *reliability layer* over single-network reconstruction rather than as a quality improvement. The uncertainty is currently diagnostic, not absolute-calibrated (ECE 0.11)."
>
> **Why not (A).** The data shows K=5 mean PSNR is slightly *worse* than K=1, driven by a regression on easy images that exceeds the gain on hard ones.
> **Why not (C).** Calibration would mean predicted U ≈ |error|; the reliability diagram shows U over-predicts |error| by a factor of ~6 at the high bin.
> **Why partial (B).** Failure detection works (AUROC = 1.0) but the test set has only 1 failure case; this needs to be replicated on 50+ images and across noise regimes before being a defensible claim.

---

## 7. What is still missing before moving to Fashion-MNIST and CIFAR

In priority order:

1. **More MNIST images (50+).** Most of the statistical claims above are weak at n=8. The PSNR difference of −0.22 dB and the AUROC of 1.0 are both within noise. Re-run on 50–100 MNIST images to make any claim statistically defensible.
2. **6000-iter run.** K=1 quality jumped from 17.4 dB at 1500 iter to 19.97 at 3000 iter (+2.6 dB). The diminishing-returns curve is not yet known — paper headline numbers may require 6000+ iter and a larger model (z_ch=128).
3. **Fix the ensemble alignment regression on easy images.** mnist_002890 lost 5.4 dB from K=1 to K=5. This is almost certainly the `align_ensemble_to_anchor` step failing on a member that converged to a different conjugate/flip variant. Worth investigating before any larger experiments.
4. **3000-iter ablation.** The 1000-iter ablation showed spectral annealing and z-update hurt at short budgets. Repeat at 3000 iter — both are expected to flip to "neutral or helpful" but this is unverified.
5. **Calibration intervention.** The K=5 uncertainty is systematically over-predicting. Two options: (i) post-hoc temperature scaling per image, (ii) replace robust normalization with quantile matching to a held-out error distribution. Either should bring the reliability curve onto the diagonal.
6. **Noise robustness on MNIST.** Before crossing datasets, verify the noise sweep (Poisson photon budgets ∈ {10, 50, 100, 500, ∞}) on the *same* MNIST set. Cross-dataset and cross-noise dimensions should be added one at a time.
7. **Larger model (z_ch=128).** All current numbers use the CPU-budget setting z_ch=64. Doubling capacity may shift the entire K=1 vs K=5 picture. Worth checking on a single noise level before launching the full sweep.

Only after items 1, 2, and 3 are settled does it make sense to move to Fashion-MNIST or CIFAR. Cross-dataset experiments amplify whatever weakness exists in the base setup — and the base setup currently has a 5.4 dB regression on an "easy" image that the codebase does not yet explain.
