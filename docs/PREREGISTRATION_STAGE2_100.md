# Pre-registration: Stage 2b (100 held-out images, compute-matched control, classical baselines)

Written on 2026-11-10, after the 50-image results (held-out ranks 0 to 49) were analysed in `STAGE2_DECISION_GATE_50.md`, and **before**:
- any method reconstructed held-out ranks 50 to 99;
- the compute-matched control or the classical baselines ran on any held-out image.

What was seen before writing this:
- the 50-image results of the five frozen methods;
- a software check of the new classical baselines on the 3 smoke images (`splits/mnist_test_smoke_indices.csv`, disjoint from the development and held-out sets).

That check showed that real and non-negative HIO/RAAR recover noiseless smoke images almost exactly. It led to two decisions recorded below: adding the support-only variant, and capping PSNR at 60 dB. No algorithm parameter was changed after the check.

Frozen files (first 16 hex digits of SHA-256):

| file | hash |
|---|---|
| `configs/stage2_mnist_heldout.yaml` (unchanged from Stage 2) | 715b7b5b2e3cd694 |
| `configs/stage2_mnist_heldout_k1long.yaml` | a12f3788c00f1b82 |
| `configs/stage2_mnist_heldout_baselines.yaml` | 44929c90c6f55a59 |
| `splits/mnist_test_heldout_indices.csv` | ae054e4f1f5c13ec |

## 1. Unchanged

The Stage-2 design in `STAGE2_GPU_RUN.md` section 2 is unchanged:
- five methods;
- T = 0.05;
- K = 5;
- 3000 iterations;
- z_channels = 128;
- failure threshold 15 dB.

The K=5 run is extended from 50 to 100 images with the same config and output directory. Rows 50 to 99 of the held-out list are appended and rows 0 to 49 are not recomputed.

## 2. Additions

### A. Compute-matched control `k1_long`

One start (seed 0) with 15,000 iterations, which is the training budget of the K=5 methods (5 × 3000). Everything else is the frozen design.

The annealing and β schedules are defined as fractions of the run, so they stretch over the 15,000 iterations. The support refinement interval (every 200 iterations) is absolute.

Question: does spending the compute on five measurement-guided starts beat spending it on one longer start?

### B. Classical baselines

Textbook HIO (Fienup 1982, β = 0.9) and RAAR (Luke 2005, β = 0.87), with the iterate on the full oversampled grid. Implementation: `pr_uq/baselines/oversampled.py`.

The older `hio.py` and `raar.py` are not used. They crop every iterate to the image window, so the zero-padded region never receives the HIO feedback or the RAAR reflection.

Settings:
- 5000 iterations plus a 200-iteration error-reduction tail;
- 5 random-phase starts, seeds 0 to 4.

Two object constraints:
- **support only**: zero outside the 32×32 window, complex values allowed inside. This is the prior the untrained method uses.
- **real + non-negative**: additionally real and non-negative inside the window. This is the usual prior of classical CDI.

Each algorithm and constraint is reported two ways:
- the first start alone (`_s1`);
- the start with the lowest Fourier error against the measurement (`_best5`). This is the same ground-truth-free rule as best-single-by-FE.

The parameters are literature defaults. They were not tuned on any image.

### C. Reporting rules

- **PSNR cap.** PSNR is capped at 60 dB in every summary statistic and paired difference, and a value at or above 60 dB is counted as exact recovery. Uncapped PSNR of an exact reconstruction only reflects floating-point error (100 to 146 dB in the smoke check). The cap does not affect any untrained-network result, all of which are far below 60 dB. Per-image CSVs keep raw values.
- **Multiple comparisons.** Holm adjustment of the Wilcoxon signed-rank p-values over all paired PSNR comparisons listed in the report. That is the 7 Stage-2 comparisons plus, for each extra method, FE-softmax versus it and best-single-by-FE versus it.
- **Primary table.** Held-out ranks 0 to 99 (n = 100). Ranks 0 to 49 were already analysed; ranks 50 to 99 are new.

## 3. Hypotheses and decision rules (n = 100 unless stated)

A claimed improvement requires all three of the following: the mean paired PSNR difference is positive, its t-based 95% CI excludes 0, and the Holm-adjusted p < 0.05.

| # | hypothesis | comparison | if not supported |
|---|---|---|---|
| H1a | five measurement-guided starts beat one start of equal compute | FE-softmax vs `k1_long` | the K=5 gain is a compute effect; no claim of more efficient use of compute; the paper rests on reliability and uncertainty only |
| H1b | as H1a, for selection | best-single-by-FE vs `k1_long` | as H1a |
| H2a | the untrained method beats classical HIO with the same prior | FE-softmax vs HIO support-only best5 | no reconstruction advantage over classical methods with the same prior is claimed |
| H2b | as H2a, for RAAR | FE-softmax vs RAAR support-only best5 | as H2a |
| H3 | none (reference only) | all methods vs real+nonneg HIO/RAAR | expected: near-exact recovery by real+nonneg HIO/RAAR in this noiseless setting; no superiority over them will be claimed here |
| H4 | the image-level failure score ranks images by reconstruction error | ranks 50 to 99 only: Spearman(FE-softmax failure score, −FE-softmax PSNR) > 0, one-sided, α = 0.05; uniform, median and best-single reported descriptively | no image-level reliability claim |

For H4, the exploratory value from ranks 0 to 49 (ρ = 0.56) is not counted as evidence.

## 4. Consequence fixed now

Whatever the outcome of H1 and H2, the noiseless setting cannot support a claim of better reconstruction than classical CDI, because real and non-negative HIO/RAAR are expected to be near-exact there.

The next stage must therefore include the Poisson low-photon regime, where the smoke check already shows classical HIO degrading:
- best-by-FE 17 to 18 dB at a photon budget of 100;
- 13 to 15 dB at a budget of 10.

That stage needs its own development subset and its own pre-registration before it runs.

## 5. Run order

On the GPU machine, one run at a time so that runtimes are not distorted:
1. Classical baselines on ranks 0 to 99.
2. K=5 extension, ranks 50 to 99.
3. Compute-matched control on ranks 0 to 99.

Then one report with `--n-heldout 100` and both extra runs. Commands are in `WINDOWS_COMMANDS.md`.

Any deviation from this document will be listed in the final report.
