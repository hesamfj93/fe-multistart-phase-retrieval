> **AUTHORITATIVE REPORT FOR THE DEVELOPMENT STAGE.**
>
> Everything in this report was computed on the **8 development images** (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399; listed in `splits/mnist_test_dev_indices.csv`), on CPU, with z_channels = 64 and 3000 iterations.
>
> The aggregation method (best-FE anchor with FE-softmax fusion) and the temperature T = 0.05 were **selected on these images**. They are therefore a development / validation subset and are excluded from every held-out evaluation. The Stage-2 design (five methods, T = 0.05, K = 5, z_channels = 128, 3000 iterations, failure threshold 15 dB) is frozen in `configs/stage2_mnist_heldout.yaml` and is evaluated only on held-out images (`splits/mnist_test_heldout_indices.csv`).
>
> Numbers here are not directly comparable with Stage-2 results because the model capacity differs (64 vs 128 channels) and the images differ.

---

# Aggregation Finalization — Temperature Sweep, Baselines, FE–PSNR Relationship

All results below are re-aggregations of the **already-trained** K=5 stacks (8 MNIST images, 3000 iterations). No retraining was performed. These analyses are intended to finalize the aggregation strategy before scaling to 50–100 images.

Terminology note, applied throughout: the K=5 regression is described as a **uniform aggregation failure**, not an "averaging bug". FE is characterized as an **empirically evaluated, ground-truth-free proxy** for member reliability, not a guaranteed estimator of reconstruction quality.

---

## Task 1 — Temperature sweep

FE-softmax, best-FE anchor, shared weights for mean and std. `K_eff` is the Kish effective sample size `(Σw)²/Σ(w²)` — how many of the 5 members effectively contribute.

| T | PSNR | SSIM | FE | Pearson | Spearman | ECE | K_eff |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0.03 | **22.16** | **0.899** | **0.190** | 0.205 | 0.313 | 0.104 | 2.97 |
| 0.05 | 22.01 | 0.879 | 0.202 | 0.215 | 0.329 | 0.111 | 3.42 |
| 0.10 | 21.57 | 0.833 | 0.231 | 0.228 | 0.341 | 0.114 | 4.18 |
| 0.15 | 21.24 | 0.814 | 0.246 | 0.232 | 0.344 | 0.115 | 4.55 |
| 0.20 | 21.04 | 0.805 | 0.255 | 0.233 | 0.345 | 0.115 | 4.72 |
| 0.30 | 20.83 | 0.796 | 0.264 | 0.234 | 0.344 | 0.115 | 4.87 |
| 0.50 | 20.65 | 0.788 | 0.270 | 0.235 | **0.344** | 0.113 | 4.95 |

**Behaviour.** As T decreases, weighting sharpens toward winner-take-all (K_eff → ~3), PSNR/SSIM/FE improve monotonically, and the uncertainty correlations degrade slightly. As T increases, weighting approaches uniform (K_eff → 5), PSNR falls toward the uniform-mean value, and uncertainty correlations improve marginally. The two objectives move in opposite directions across the entire range — there is no interior optimum where both peak.

**Recommendation.** The reconstruction–uncertainty trade-off is shallow on the uncertainty side (Spearman varies only 0.313→0.345, a 0.03 spread) and steeper on the reconstruction side (PSNR varies 20.65→22.16, a 1.5 dB spread). A low temperature therefore buys meaningful reconstruction quality at small uncertainty cost. **T = 0.05** is the recommended operating point: it captures most of the PSNR gain (22.01, within 0.15 dB of the T=0.03 peak) while retaining a higher K_eff (3.42 vs 2.97) and better Spearman (0.329 vs 0.313) than the sharpest setting. T=0.03 is a defensible alternative if reconstruction is prioritized. This should be re-confirmed on the larger image set before being fixed, since the sweep is over only 8 images.

See `figures/temperature_sweep.png`.

---

## Task 2 — Aggregation baseline comparison

Single table, all methods, 8 images. Anchor = best-FE for all ensemble methods except "Uniform mean" (which uses the legacy anchor=first). Pearson/Spearman/ECE are for the fused uncertainty; they are undefined for K=1 (no ensemble spread).

**Definition note for Best-single-by-FE.** This method is a hybrid, and its columns must be read accordingly: the **reconstruction** is a single member (the one with the lowest Fourier error, `x̂ = x_{k*}`), while the **uncertainty** used for its Pearson/Spearman/ECE is still estimated from the *full aligned multi-start stack* (the pixel-wise std of all five aligned members), exactly as for the other ensemble methods. In other words, best-single uses one member for the estimate and five members for the uncertainty. Its low Pearson/Spearman (0.178 / 0.197) are therefore not a like-for-like uncertainty comparison — they pair a single-member reconstruction's error map against an ensemble-derived spread, which is a mismatch by construction. The PSNR/SSIM/FE columns, which depend only on the selected member, are the meaningful ones for this row.

| method | PSNR | med PSNR | SSIM | FE | Pearson | Spearman | ECE |
|---|--:|--:|--:|--:|--:|--:|--:|
| K=1 | 19.97 | 19.67 | 0.824 | 0.214 | n/a | n/a | n/a |
| Uniform mean (legacy) | 19.75 | 19.09 | 0.775 | 0.288 | **0.286** | 0.321 | 0.114 |
| Pixel-wise median | 20.77 | 21.25 | 0.887 | 0.230 | 0.190 | 0.232 | 0.114 |
| Trimmed mean (trim=1) | 20.64 | 20.58 | 0.858 | 0.251 | 0.220 | 0.285 | 0.114 |
| **Best-single-by-FE** | **22.30** | **23.74** | **0.905** | **0.164** | 0.178 | 0.197 | 0.125 |
| Best-FE anchor + uniform | 20.38 | 19.95 | 0.778 | 0.280 | 0.236 | **0.344** | 0.113 |
| Inverse-FE weighting | 21.58 | 22.48 | 0.831 | 0.231 | 0.226 | 0.339 | 0.114 |
| FE-softmax (T=0.1) | 21.57 | 22.55 | 0.833 | 0.231 | 0.228 | 0.341 | 0.114 |
| Top-3 FE | 21.06 | 21.61 | 0.864 | 0.222 | 0.197 | 0.239 | **0.088** |
| Medoid | 17.66 | 16.15 | 0.749 | 0.241 | 0.245 | 0.231 | 0.116 |

**Key observations.**

1. **Best-single-by-FE has the highest PSNR (22.30 dB) of any method, exceeding all ensemble aggregators.** This is the most important result in the table and it must be reported prominently. On this 8-image set, selecting the single member with the lowest Fourier error and discarding the rest reconstructs better than averaging. This directly challenges the value of ensembling *for reconstruction*: if the goal were PSNR alone, a multi-start-plus-pick strategy would beat any weighted mean here. The ensemble's remaining justification is the uncertainty map — and note that best-single does not produce its own uncertainty; the Pearson/Spearman shown for it are computed against the full aligned stack's spread (see the definition note above), so they are not a fair uncertainty comparison and should not be read as "best-single has poor uncertainty". The honest statement is that best-single provides no intrinsic uncertainty of its own.

2. **FE-softmax and Inverse-FE are nearly identical** (21.57 vs 21.58 PSNR; correlations within 0.002). The specific weighting function matters little once weights track FE; what matters is that bad members are downweighted at all.

3. **Median and trimmed mean recover most of the regression without any fidelity score** (20.77 / 20.64 dB vs 19.75 uniform), using only pixel-wise robustness. They are attractive as fidelity-free fallbacks, though both trail the FE-weighted methods on PSNR.

4. **Medoid is the worst method (17.66 dB)** — selecting the member most similar to the others backfires when the "others" include a bad majority for some images.

5. **No single method wins on both axes.** Best-single-by-FE leads reconstruction; Best-FE-anchor+uniform and FE-softmax lead uncertainty rank-correlation (Spearman 0.344 / 0.341); Top-3 leads calibration (ECE 0.088); legacy uniform leads linear correlation (Pearson 0.286, but with the worst PSNR). The choice depends on whether the paper's contribution is reconstruction or reliability.

---

## Task 3 — FE–PSNR relationship

The aggregation strategy relies on FE as a proxy for member quality. This section evaluates that proxy directly.

- **Pooled member-level correlation** (all 40 members): Spearman ρ(−FE, PSNR) = **+0.898**.
- **Within-image correlation** (does FE rank the 5 members correctly inside one image?): mean ρ = **+0.588**, range **−0.50 to +1.00**.
- Per-image ρ: `+0.80, −0.50, +0.90, +1.00, +0.60, +0.30, +0.90, +0.70`.
- **1 of 8 images has negative within-image ρ** (mnist_001729).
- **Top-1 agreement** (lowest-FE member == highest-PSNR member): **6/8 (75%)**.
- **Top-2 agreement** (lowest-FE member within the top-2 PSNR members): **7/8 (88%)**.

**Characterization.** FE is a strong proxy in aggregate (pooled ρ ≈ 0.90) but an imperfect one within an image (mean ρ ≈ 0.59, one full inversion). The pooled correlation is high partly because it spans easy and hard images together — FE separates a 26 dB member from a 13 dB member reliably. Within a single image, especially when all members are of similar quality (mnist_001729: all 24–27 dB, ρ = −0.50; mnist_006044: all 23–25 dB, ρ = +0.30), FE and PSNR can disorder, but the practical cost is small because the members are close in quality there anyway. Where FE most needs to work — flagging a catastrophic member such as member 2 of mnist_002890 (13.5 dB, highest FE) — it succeeds.

Accordingly, FE is reported as a **practical, empirically evaluated proxy for member reliability whose effectiveness varies with the spread of member quality**, not as a method that reliably identifies the single best reconstruction. The per-image scatter (`figures/fe_psnr_scatter.png`, member index labelled) shows both the strong-separation cases and the two Top-1 mismatches (mnist_001729, mnist_006044), both of which occur on easy images where all members are good.

---

## Negative result — decoupled uncertainty weighting

Following the earlier suggestion to decouple the two roles of the weights — use FE-softmax weights for the reconstruction mean, but compute the uncertainty std from the aligned members with *uniform* weights, to preserve member disagreement — this was tested and is reported here as a **negative result**.

| metric | coupled (shared softmax weights) | decoupled (softmax mean, uniform std) | change |
|---|--:|--:|--:|
| PSNR | 21.566 | 21.566 | +0.000 |
| Pearson(U, err) | 0.228 | 0.227 | −0.001 |
| Spearman(U, err) | 0.341 | 0.340 | −0.001 |
| ECE | 0.114 | 0.116 | +0.002 |

Decoupling did **not** improve Pearson correlation (−0.001, negligible). What this establishes is narrow and should be stated as such: **weighted ensemble variance alone does not explain the Pearson reduction** — changing only how the std is weighted, while keeping the weighted-mean reconstruction, leaves the linear correlation essentially unchanged. It does **not** prove the cause. Two candidate contributors remain and are not separated by this experiment: (i) the improved reconstruction itself — as residual error shrinks toward a noise floor its linear coupling to any uncertainty signal may weaken — and (ii) the physics-residual component of the fused uncertainty, which is computed from the weighted reconstruction mean and so changes when the reconstruction changes. Distinguishing these would require a controlled analysis (for example, recomputing Pearson against the std-only uncertainty with the reconstruction held fixed, and separately varying the fusion α). For now the decoupled variant is reported only as a negative result and is not carried forward; the mechanism behind the Pearson reduction is left as an open question rather than asserted.

---

## Summary and recommendation

- The regression was a **uniform aggregation failure**: a plain mean over members of heterogeneous quality. Any method that downweights or excludes poor members (median, trimmed mean, FE weighting, best-single) removes it.
- **Two distinct output modes, not one.** The honest framing is that these are separate deliverables rather than a claim that ensembling improves PSNR: **best-single-by-FE selection maximizes reconstruction quality** (22.30 dB here, the strongest of any method), while **fidelity-weighted ensemble fusion provides a competitive reconstruction together with an uncertainty estimate** (FE-softmax, ~22.0 dB, plus the spread map). Best-single must appear as a first-class baseline that the ensemble justifies itself against; the ensemble's distinct value is the uncertainty map, and best-single has no intrinsic uncertainty of its own.
- **For the combined reconstruction + uncertainty goal, FE-softmax at T ≈ 0.05 is the recommended aggregator**: near-best PSNR (22.0 dB) with the best-available uncertainty rank-correlation and a reasonable K_eff (~3.4).
- **FE is a useful but imperfect proxy** (pooled ρ ≈ 0.90, within-image ρ ≈ 0.59, Top-1 75%, Top-2 88%); this should be stated as an empirical finding with its failure cases shown, not as a guarantee.
- The decoupled-uncertainty variant is a **negative result** and is dropped; it shows only that weighted ensemble variance alone does not explain the Pearson reduction, and does not identify the actual cause (left as an open question).

All numbers are over 8 images and are not yet statistically established. Recommended stopping point for review before scaling up. When scaling: re-run the temperature sweep and the FE–PSNR agreement on 50–100 images through the updated pipeline, and add best-single-by-FE as a first-class baseline in every subsequent table.

**Artifacts.** `results/temperature_sweep.csv`, `results/aggregation_baselines.csv`, `results/fe_psnr_analysis.csv`; `figures/temperature_sweep.png`, `figures/fe_psnr_scatter.png`.
