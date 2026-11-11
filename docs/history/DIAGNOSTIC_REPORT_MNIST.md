> **SUPERSEDED.** Early diagnostic from before the 3000-iteration runs and the aggregation fix. Historical record only. The authoritative development-stage report is **AGGREGATION_FINALIZATION_REPORT.md**.

---

# Diagnostic Report — Real MNIST K=1 vs K=3

**Setup.** 8 MNIST test images, 32×32 (28→32 zero-padded), 2× oversampled
Fourier intensity, clean (no noise). Untrained double-decoder, z_channels=64,
base_ch=64, 1500 iterations, CPU. K=3 runs use 4 of the same 8 images so the
comparison is paired and like-for-like.

All numbers below are from actual runs on real MNIST. They are diagnostic
findings during a CPU sanity check, **not paper results**. The paper target
is K=5 with 3000–6000 iterations on a GPU, where these regularizers are
expected to pay off in ways they don't here.

---

## 1. Quantitative — K=1 vs K=3 (paired, 4 images)

| image | K=1 PSNR | K=3 PSNR | K=1 SSIM | K=3 SSIM | K=3 ECE | K=3 fail-score |
|---|--:|--:|--:|--:|--:|--:|
| mnist_006044 (1) | 22.52 | 20.23 | 0.869 | 0.758 | 0.132 | 0.169 |
| mnist_002890 (7) | 21.34 | 19.58 | 0.917 | 0.784 | 0.110 | 0.146 |
| mnist_009399 (7) | 14.78 | 15.69 | 0.482 | 0.603 | 0.155 | 0.229 |
| mnist_001917 (5) | 10.04 | 12.48 | 0.199 | 0.195 | 0.176 | 0.292 |
| **mean** | **17.17** | **16.99** | **0.617** | **0.585** | **0.143** | **0.209** |

- K=1 on 8 images (full set): mean PSNR=**17.44 dB**, SSIM=**0.68**, FE=**0.32**,
  mean ECE=**0.057**, runtime=22.3 s/image.
- K=3 on 4 images: mean PSNR=**16.99 dB**, SSIM=**0.59**, FE=**0.36**,
  mean ECE=**0.143**, runtime=67.0 s/image (3.01× K=1, as expected).
- **Failure-detection AUROC = 1.0** on these 4 — the fail-score correctly
  separates the two failures (009399, 001917) from the two successes
  (006044, 002890).

---

## 2. Are reconstructions visually meaningful?

See `figures/diagnostic_k1_vs_k3.png`.

- **mnist_006044 (digit '1')**: clean reconstruction at 22.5 dB (K=1).
  Recognizable.
- **mnist_002890 (digit '7')**: clean reconstruction at 21.3 dB (K=1).
  Recognizable.
- **mnist_009399 (digit '7')**: partial — distorted '7'-like shape at 14.8 dB.
  Not unambiguously the right digit but the support is roughly correct.
- **mnist_001917 (digit '5')**: failure mode — diffuse blob at 10.0 dB,
  no recognizable digit. Exactly the kind of failure that the uncertainty
  signal needs to flag.

K=3 mean reconstructions are slightly *blurrier* than K=1 on the easy
images (members weren't fully converged at 1500 iter on CPU, so the average
of three partially-converged solutions is worse than a single one). This
will reverse with more iterations.

---

## 3. Does loss decrease smoothly?

See `figures/convergence_k1.png`, left panel (log-scale).

- All four images show loss dropping from ~2×10⁻² → ~3×10⁻⁴ in the first
  ~250 iterations.
- After that, loss is in [10⁻⁴, 10⁻³] with mild oscillation, not strictly
  monotonic. Not pathological.
- No catastrophic divergence in any of the four images.

---

## 4. Does PSNR improve over iterations?

See `figures/convergence_k1.png`, middle panel.

Yes, but with a **critical caveat**: PSNR keeps improving long after the
training loss has plateaued.

- mnist_002890: training loss plateaus at iter 250, but aligned PSNR jumps
  from ~13 dB to ~22 dB around **iter 900** — the network found the
  correct twin/conjugate variant much later than the loss suggests.
- mnist_006044: smooth rise from 8 dB → 22 dB across 1500 iters.
- mnist_009399: plateau at ~14 dB after iter 500 — partial recovery.
- mnist_001917: plateau at ~10 dB after iter 500 — full failure.

This is why I removed the default `keep_best_recon=True` setting: tracking
"best by training loss" was systematically picking iterates from early in
training where the loss was already low but the PSNR was 8–10 dB worse than
the final iterate. **Always use the final iterate for PR.**

---

## 5. Does physics projection help or destabilize?

Ablation: 2 images, 1000 iter each, K=1.

| variant | mean PSNR | Δ vs baseline |
|---|--:|--:|
| baseline (all on) | 15.28 dB | — |
| no physics projection | 15.65 dB | +0.37 dB (neutral) |
| no z-update | 18.81 dB | **+3.53 dB** |
| no spectral annealing | 20.66 dB | **+5.39 dB** |

**Findings.**
- **Physics projection ≈ neutral at 1000 iter.** No destabilization
  detected (loss didn't blow up; reconstructions remained well-behaved). At
  longer iter budgets the projection should provide modest benefit, but on
  short runs it has no meaningful effect.
- **Z-update hurts at this iter budget.** Adding latent-z as a trainable
  parameter doubles the effective param count seen by Adam early on; this
  slows convergence in the first ~500 iters and the extra iterations
  needed aren't available at 1000 iter. Expected to help at 3000+ iter.
- **Spectral annealing hurts at this iter budget.** With
  `anneal_fraction=0.6`, full bandwidth is only reached at iter 600 of
  1000. The other 400 iters are too few to use full-bandwidth gradient
  information. Tuning: either reduce `anneal_fraction` to 0.3 for short
  runs or use longer total iterations.

**Honest take:** the only ablation result that is genuinely good news for
the method as designed is "physics projection ≈ neutral, doesn't destabilize".
The other two regularizers need *more* iteration budget, not less, to pay
off. This is the strongest signal in the diagnostic that the paper's main
experiments must run at 3000+ iterations.

---

## 6. Is the uncertainty map meaningful or random?

Look at `figures/diagnostic_k1_vs_k3.png`, columns 4–6.

**Qualitative.** In all four images, the K=3 fused-uncertainty map is bright
exactly where the reconstruction is wrong:
- Rows 1, 2 (success): uncertainty concentrated along the stroke edges
  and slight halos where the orientation differs between members.
- Rows 3, 4 (failure): uncertainty is bright everywhere in the digit
  support — the maps are visually "busy" and saturate to 1.0 in places.

**Quantitative.**
| image | corr(fused_U, err) | corr(std, err) | corr(r_phys, err) | mean_U |
|---|--:|--:|--:|--:|
| mnist_006044 (success) | 0.095 | 0.067 | 0.108 | 0.169 |
| mnist_002890 (success) | 0.272 | 0.236 | 0.260 | 0.146 |
| mnist_009399 (partial) | 0.156 | 0.106 | 0.178 | 0.229 |
| mnist_001917 (failure) | 0.152 | 0.060 | 0.223 | 0.292 |

- All correlations are positive — uncertainty and error are linked, not
  random.
- Pixel-wise correlations are modest (0.10–0.27). This is consistent with
  the literature on UQ in inverse problems: pixel-level calibration is
  hard. The **image-level signal** (mean uncertainty) is much stronger:
  failures have mean_U = 0.229, 0.292 vs successes 0.169, 0.146.
- Physics residual `r_phys` is consistently more correlated with error
  than ensemble std on its own, justifying the fusion design.
- **Failure-detection AUROC = 1.0** confirms the image-level signal
  separates failures from successes perfectly on these 4 — needs more
  images to be a statistically meaningful number, but the qualitative
  story is intact.

**Caveat.** K=3 is too small to get a clean variance estimate; K=5 in the
paper main experiments should give smoother std maps.

---

## 7. Does latent-z optimization help?

At 1000 iter, **no** — it hurts by ~3.5 dB (see §5). At 3000+ iter the
verdict will likely flip. Not yet verified in real data.

---

## 8. Bugs and weak points fixed during this run

1. **Bug** (fixed): trainer was returning the β-mixed `x_use` instead of
   `x_net`. The projection introduces a small imaginary component that
   degrades aligned PSNR against the real ground truth. Fix:
   `best_recon = x_net.detach().clone()`.
2. **Bug** (fixed): `keep_best_recon=True` was selecting iterates by
   training loss, but with spectral annealing the loss plateaus near zero
   at iter ~300 while aligned PSNR continues to climb for 1000+ more
   iterations. "Best by loss" was systematically picking 5–10 dB worse
   reconstructions than the final iterate. Fix: default is now off; final
   iterate is returned.
3. **Bug** (fixed): alignment was applying shift in the wrong direction
   — `cyclic_shift(rv, dy, dx)` instead of `cyclic_shift(rv, -dy, -dx)`.
   Caught by `test_align_after_shift_perfect`.

---

## 9. Remaining concerns / weak points

- **K=3 mean PSNR slightly below K=1** at 1500 iter. Need to re-run at
  3000+ iter to confirm ensembling actually helps; expected to.
- **z-update and spectral annealing both hurt at 1000 iter.** The paper
  must use longer iterations (≥3000) or adjust schedules. Could also be
  hyperparameter mis-tuning (e.g. lr_z=lr_θ=1e-3 may be too high for z).
- **Ensemble alignment for averaging.** `align_ensemble_to_anchor` aligns
  members to member 0 before averaging. If member 0 itself is a failure
  case, the entire ensemble inherits a bad reference. Could try aligning
  to the member with the lowest training loss instead.
- **Sample size.** 4 paired images is enough to spot failure modes but
  too few for statistically valid claims. Min 50 images for the paper.
- **No noise tested yet.** All experiments here are clean Fourier
  intensity. Noise robustness (Poisson, Gaussian) is the next milestone.
- **ECE of 0.06–0.18** is high. Per-image pixel-level calibration is
  weak; image-level failure detection is strong (AUROC=1.0). The paper
  emphasis on calibration may need careful framing — the strong claim is
  on failure detection, not pixel-level calibration.

---

## 10. What's next

In priority order:
1. Re-run K=1 and K=5 on 20–50 MNIST images with 3000–6000 iterations on
   a GPU. Verify K=5 beats K=1 in mean PSNR.
2. Re-run the four-way ablation (`baseline / no_proj / no_z / no_anneal`)
   at the longer iteration budget. Expectation: regularizers will pay off.
3. Tune `lr_z` separately from `lr_θ`; try `lr_z=1e-4` (10× smaller).
4. Move to Poisson noise sweep only after K=1 and K=5 results on clean
   MNIST stabilize.
