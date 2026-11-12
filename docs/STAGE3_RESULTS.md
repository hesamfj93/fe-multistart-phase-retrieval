# Stage 3 results: complex-valued objects, noiseless

Second held-out study. Plan: [`STAGE3_COMPLEX_PLAN.md`](../STAGE3_COMPLEX_PLAN.md), tagged `stage3-plan-v1` (commit `27c7c20`) before any held-out image of this stage was reconstructed.

## Execution

| step | date (local time) | outcome |
|---|---|---|
| plan, code and configurations committed and tagged | 2026-11-11 | tag `stage3-plan-v1` |
| development check, 8 development images | 2026-11-11 | no software error, nothing changed |
| classical baselines, ranks 100 to 199 | 2026-11-11 20:54 to 22:44 | 100 of 100 images |
| five starts, ranks 100 to 199 | 2026-11-11 22:44 to 2026-11-12 02:19 | 100 of 100 images |
| `k1_long`, ranks 100 to 199 | 2026-11-12 02:19 to 05:53 | 100 of 100 images |
| `scripts/stage3_report.py`, unchanged | 2026-11-12 | `results/stage3_complex_heldout/report/` |

One RTX 4060 Laptop GPU, the stage-2 environment. The report was run again in a second environment (Linux, other NumPy and SciPy versions) from the stored per-image files. `methods.csv` and `hypotheses.csv` are byte-identical; `summary.json` differs only after the 12th significant digit.

**Deviations from the plan: none.**

## Pre-specified hypotheses (Holm family C1 to C8, n = 100)

Paired difference of complex PSNR (first minus second method), in dB. Supported means: positive mean, t-based 95% CI excluding zero, and Holm-adjusted two-sided Wilcoxon p < 0.05.

| id | comparison | mean | 95% CI | median | better/worse/equal | p (Holm) | supported |
|---|---|--:|---|--:|--:|--:|---|
| C1 | FE-softmax vs K=1 | +3.36 | [2.74, 3.98] | +2.57 | 89/11/0 | 4.2e-15 | yes |
| C2 | selection vs K=1 | +3.09 | [2.38, 3.80] | +1.83 | 71/12/17 | 1.5e-11 | yes |
| C3 | FE-softmax vs `k1_long` | +1.37 | [0.73, 2.01] | +1.20 | 67/33/0 | 1.3e-4 | yes |
| C4 | selection vs `k1_long` | +1.10 | [0.37, 1.83] | +0.78 | 63/37/0 | 0.0047 | yes |
| C5 | FE-softmax vs HIO supp best5 | +4.34 | [3.78, 4.90] | +4.25 | 96/4/0 | 1.6e-16 | yes |
| C6 | FE-softmax vs RAAR supp best5 | +6.24 | [5.70, 6.79] | +6.18 | 100/0/0 | 3.1e-17 | yes |
| C7 | FE-softmax vs HIO sw best5 | +2.48 | [1.83, 3.12] | +2.59 | 81/19/0 | 3.4e-9 | yes |
| C8 | FE-softmax vs RAAR sw best5 | −13.79 | [−17.71, −9.87] | −1.92 | 39/61/0 | 9.0e-5 | **no** |

C8 is not supported, and the difference is significant in favour of RAAR with shrinkwrap.

**C9.** Spearman(FE of the FE-softmax reconstruction, −PSNR) = 0.806, one-sided p = 2.2e-24, bootstrap 95% interval [0.73, 0.86]. Supported.

## All methods (n = 100)

Complex PSNR in dB, capped at 60 dB; 95% CI is the percentile bootstrap of the mean; failures < 15 dB; exact ≥ 60 dB.

| method | PSNR | median | 95% CI | amp. PSNR | phase err. (rad) | failures | exact |
|---|--:|--:|---|--:|--:|--:|--:|
| RAAR sw best5 | 31.47 ± 20.38 | 18.40 | [27.67, 35.60] | 32.61 | 0.196 | 29 | 25 |
| RAAR sw s1 | 22.76 ± 18.01 | 13.89 | [19.41, 26.42] | 24.09 | 0.326 | 65 | 17 |
| FE-softmax | 17.67 ± 3.06 | 17.45 | [17.08, 18.28] | 18.67 | 0.204 | 21 | 0 |
| selection (best-single-by-FE) | 17.40 ± 3.57 | 17.51 | [16.71, 18.10] | 18.60 | 0.221 | 30 | 0 |
| median | 16.79 ± 2.73 | 16.45 | [16.29, 17.33] | 17.78 | 0.229 | 24 | 0 |
| `k1_long` | 16.30 ± 3.75 | 16.50 | [15.58, 17.02] | 17.53 | 0.267 | 43 | 0 |
| uniform | 16.14 ± 2.65 | 15.95 | [15.63, 16.67] | 17.08 | 0.247 | 31 | 0 |
| HIO sw best5 | 15.20 ± 2.48 | 15.00 | [14.72, 15.69] | 16.55 | 0.288 | 50 | 0 |
| HIO sw s1 | 14.63 ± 2.51 | 14.57 | [14.15, 15.13] | 16.07 | 0.322 | 57 | 0 |
| K=1 | 14.31 ± 3.32 | 13.70 | [13.69, 14.97] | 15.65 | 0.363 | 63 | 0 |
| HIO supp best5 | 13.33 ± 1.33 | 13.13 | [13.08, 13.59] | 14.55 | 0.374 | 88 | 0 |
| HIO supp s1 | 12.80 ± 1.36 | 12.72 | [12.53, 13.06] | 14.03 | 0.411 | 94 | 0 |
| HIO realpos s1 | 11.99 ± 1.87 | 11.86 | [11.63, 12.37] | 12.89 | 0.399 | 93 | 0 |
| RAAR realpos s1 | 11.90 ± 1.90 | 11.47 | [11.53, 12.27] | 12.78 | 0.393 | 95 | 0 |
| HIO realpos best5 | 11.71 ± 1.75 | 11.55 | [11.39, 12.07] | 12.46 | 0.400 | 97 | 0 |
| RAAR realpos best5 | 11.68 ± 1.79 | 11.40 | [11.34, 12.03] | 12.51 | 0.405 | 97 | 0 |
| RAAR supp best5 | 11.43 ± 1.15 | 11.40 | [11.21, 11.66] | 12.60 | 0.511 | 100 | 0 |
| RAAR supp s1 | 11.26 ± 1.18 | 11.21 | [11.03, 11.49] | 12.38 | 0.516 | 100 | 0 |

NRMSE, SSIM, FE and runtime: `results/stage3_complex_heldout/report/methods.csv`.

## Descriptive analyses of the plan

- FE at the start level: within-image Spearman(−FE, PSNR) mean 0.58 (median 0.70); the lowest-FE start was the best start in 70 of 100 images and among the two best in 83.
- Pixel-wise Spearman of u with |x̂ − x| over the window: uniform 0.73, median 0.58, FE-softmax 0.70. Whole-window values mainly reflect the separation of object and background (stage-2 ablation, paper Table IV).
- Real and nonnegative HIO and RAAR (wrong prior for these objects): about 12 dB, 93 to 97 failures.

## Post-hoc analyses (not in the plan)

`scripts/stage3_posthoc.py` → `results/stage3_complex_heldout/report/posthoc_stage3.json`. Written after the report above; no test, no decision changed.

1. **FE is comparable only within one family.** Median FE: untrained methods 0.24 to 0.36 (FE-softmax 0.27); support-only and shrinkwrap classical 0.005 to 0.024; real and nonnegative classical 0.31, since they cannot fit a complex object.
2. **FE within RAAR sw best5.** Spearman(FE, −PSNR) = 0.94. The 25 exact recoveries have FE ≤ 3.5e-4, the 29 failures FE ≥ 0.0297, with no overlap. FE-softmax is better on 28 of these 29 failures, and on 11 of the 46 images between the two groups.
3. **Fallback rule, not planned.** Keep RAAR sw best5 when its FE ≤ t, else FE-softmax:

| t | RAAR kept | mean | median | failures | exact |
|--:|--:|--:|--:|--:|--:|
| 0.001 | 34 | 31.22 | 19.08 | 18 | 25 |
| 0.003 | 34 | 31.22 | 19.08 | 18 | 25 |
| 0.01 | 38 | 31.92 | 19.52 | 18 | 25 |
| 0.02 | 44 | 32.39 | 20.04 | 16 | 25 |
| 0.03 | 57 | 32.68 | 20.30 | 11 | 25 |
| RAAR sw best5 alone | | 31.47 | 18.40 | 29 | 25 |
| FE-softmax alone | | 17.67 | 17.45 | 21 | 0 |

No threshold is chosen. A test needs a threshold fixed on other images and a new held-out set.

## Reading

- The frozen multi-start design helps more on complex objects than on real ones. A single start failed on 63 images here, against 20 in stage 2. The compute-matched comparison, not supported in stage 2, is supported here (C3, C4).
- The untrained methods beat HIO with every constraint and RAAR with the support-only constraint.
- RAAR with shrinkwrap, given a bounded amplitude and an adaptive support like the network, is better on 61 of 100 images and exact on 25. It fails on 29, and FE-softmax is better on 28 of those.
- The untrained reconstructions never reach a low FE (minimum 0.11), so they cannot be exact.

## Reproduce

No GPU needed:

```
python scripts/stage3_report.py --main results/stage3_complex_heldout --k1long results/stage3_complex_heldout_k1long --baselines results/stage3_complex_heldout_baselines --out results/stage3_complex_heldout/report
python scripts/stage3_posthoc.py --main results/stage3_complex_heldout --baselines results/stage3_complex_heldout_baselines --out results/stage3_complex_heldout/report/posthoc_stage3.json
python paper/make_stage3_figures.py --main results/stage3_complex_heldout --k1long results/stage3_complex_heldout_k1long --baselines results/stage3_complex_heldout_baselines
python paper/make_supplement_tables.py --run-dir results/stage2_mnist_heldout --stage3 results/stage3_complex_heldout/report
```

Rerun the reconstructions (about 9 hours on the laptop GPU): the commands in `STAGE3_COMPLEX_PLAN.md`. Completed images are skipped, so move the stored runs out of `results/` first.

Stored per image: `members.npz` (five starts, raw and aligned, with the ground truth and the measurement), `aggregates.npz` (the five aggregates aligned to the ground truth and their maps), `metrics.json`; `k1_long` likewise; `baselines.npz` (all classical reconstructions and their starts). Run logs and preview images are not stored.
