# Post hoc: the ensemble outlier rule of Hong et al. on the stored starts

**Not part of any analysis plan.** No new reconstruction. Script: `scripts/posthoc_eor_rule.py`, committed (`ff14b99`) before the computation. Result: `results/posthoc_eor/summary.json`.

## Rule

Hong et al. (Ain Shams Eng. J. 17(11):104411, 2026, Sec. 3.1.3, Eq. 14) train several untrained networks and remove network i if |d_i − median(d)| / MAD(d) > threshold, where d_i is its squared distance to the observation and MAD the median absolute deviation; the rest are averaged.

Here:
- d_i = ‖|F P x_i| − √y‖², proportional to FE_i² for a given image; two-sided test, as written.
- Threshold: the paper gives none. Primary 3 × 1.4826 = 4.45, the "three scaled MADs" of our MAD rules; 3, 2 and a one-sided test (only large residuals) as sensitivity.
- The kept starts are averaged after the alignment to the lowest-FE start that FE-softmax also uses.
- Applied once to the five final starts of each image. Hong et al. apply it inside every outer iteration and use the mean as the prior of the next update; their tasks have no shift, flip or global-phase ambiguity.

Check: FE-softmax recomputed from the stored starts reproduces the reported PSNR within 2 × 10⁻⁶ dB.

## Results (n = 100 per study; real-part PSNR for real, complex PSNR for complex objects)

| rule | real mean | failures | starts removed | Δ vs FE-softmax | complex mean | failures | starts removed | Δ vs FE-softmax |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| plain mean of aligned starts | 18.87 | 4 | | −1.13 | 16.50 | 26 | | −1.17 |
| MAD rule, \|z\| > 4.45 (primary) | 18.87 | 5 | 56 | −1.12 | 16.58 | 27 | 41 | −1.10 |
| MAD rule, \|z\| > 3 | 18.91 | 3 | 78 | −1.09 | 16.49 | 30 | 66 | −1.18 |
| MAD rule, \|z\| > 2 | 18.86 | 3 | 122 | −1.13 | 16.32 | 32 | 108 | −1.35 |
| MAD rule, z > 4.45 (one-sided) | 19.01 | 4 | 41 | −0.98 | 16.63 | 25 | 31 | −1.04 |
| selection | 19.93 | 5 | | −0.07 | 17.40 | 30 | | −0.27 |
| FE-softmax | 19.99 | 3 | | | 17.67 | 21 | | |

- Primary rule against FE-softmax: real −1.12 dB (95% t-interval −1.44 to −0.81; FE-softmax better on 85 images), complex −1.10 dB (−1.37 to −0.83; 81 images); unadjusted Wilcoxon p = 1.4 × 10⁻¹³ and 1.5 × 10⁻¹².
- The two-sided test removed the lowest-FE start on 13 real and 8 complex images (17 and 16 at threshold 3, 36 and 34 at threshold 2); the one-sided test never did.

## Reading

Applied to final starts, the rule is no better than the plain mean of the aligned starts, and weighting every start by its fit to the measurement (FE-softmax) or selecting the best-fitting start is better by about 1 dB in both studies. A two-sided outlier test can remove the start that fits the measurement best.

```
python scripts/posthoc_eor_rule.py --out results/posthoc_eor/summary.json
```
