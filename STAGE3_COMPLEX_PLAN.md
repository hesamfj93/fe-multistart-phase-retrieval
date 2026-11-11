# Stage 3 analysis plan: complex-valued objects, noiseless

**Status: frozen.** It is frozen at the git tag `stage3-plan-v1`, which is pushed before any stage-3 held-out run. Any later change is listed under "Deviations" at the end, with its date and reason.

## Question

In stage 2 the objects were real and nonnegative, and HIO with that constraint was exact on 99 of 100 images. Stage 3 uses objects for which that constraint is wrong. It asks three questions:

1. Does the frozen multi-start design, unchanged, reconstruct complex-valued objects better than one start of the same network?
2. Does it beat HIO and RAAR when they get the same prior information as the network?
3. Is the Fourier error (FE) still a reliable image-level signal?

## Fixed from stage 2

The whole reconstruction design is reused without change:
- network, trainer and iterations;
- K = 5 with seeds 0 to 4, T = 0.05;
- anchor, alignment and the five aggregation rules;
- the compute-matched control `k1_long` (one start, 15,000 iterations);
- HIO (β = 0.9) and RAAR (β = 0.87), each with 5000 iterations plus a 200-iteration error-reduction tail, 5 random starts.

`tests/test_stage3_complex.py::test_stage3_design_equals_frozen_stage2_design` checks this. The stage-3 configuration files equal the frozen stage-2 files except for:
- the object model;
- the image range;
- the output folder;
- the list of classical constraints.

## Objects

Each object is x = A exp(iφ), built as follows (`pr_uq/data/complex_objects.py`):

- **Amplitude.** A is the MNIST digit, zero-padded to 32 × 32, values in [0, 1].
- **Phase.** φ is white Gaussian noise filtered with a Gaussian of 3 pixels. It is centered on the support (A > 0.05) and scaled so that max |φ| = π/2 there.
- **Seed.** The phase seed is 1,000,003 plus the MNIST test index.
- **Measurement.** The intensity is |FPx|², computed with the same forward model as stage 2, without noise.

## Images

| set | images | role |
|---|---|---|
| development check | the 8 development images | software check only |
| held-out | ranks 100 to 199 of `splits/mnist_test_heldout_indices.csv` | all tests |

Ranks 100 to 199 have not been used by any method. They do not overlap the development images or ranks 0 to 99; the runner and the tests check this.

## Methods

**Untrained network.** From the same five starts per image:
- K=1 (start 0)
- uniform mean
- pixel-wise median
- selection (lowest-FE start)
- FE-softmax fusion

Also `k1_long`.

**Classical**, each run as `s1` (start 0) and `best5` (lowest-FE start of five):

| constraint | definition | role |
|---|---|---|
| support-only (`supp`) | zero outside the 32 × 32 window, complex values inside | weak prior |
| shrinkwrap (`sw`) | window intersected with a shrinkwrap support. Every 20 iterations the support becomes the pixels where the blurred amplitude exceeds 0.2 of its maximum; the blur starts at 3 px and shrinks by 1% per update to 1.5 px. Amplitude bounded by 1 inside the support. | prior matched to the network (bounded amplitude, adaptive support) |
| real and nonnegative (`realpos`) | as in stage 2 | wrong for these objects; reported for reference only, not tested |

## Metrics

All reconstructions are aligned to the ground truth with the frozen stage-2 alignment: four-variant search, cyclic shift, global phase.

- **Primary.** PSNR of the complex field, 10 log10(1 / mean |x̂ − x|²) over the 32 × 32 window.
  - Capped at 60 dB in summaries.
  - At or above 60 dB counts as exact.
  - Below 15 dB counts as a failure.
- **Secondary.**
  - amplitude PSNR and amplitude SSIM;
  - phase error, the amplitude-weighted mean of |arg(x̂ x*)| over |x| > 0.05;
  - NRMSE of the complex field;
  - FE.

## Hypotheses

**Decision rule.** An improvement is claimed only if all three hold:
- the mean paired difference in complex PSNR is positive;
- its t-based 95% CI excludes zero;
- the Holm-adjusted two-sided Wilcoxon signed-rank p is below 0.05.

The Holm family is C1 to C8 and nothing else.

| id | first method | second method | question |
|---|---|---|---|
| C1 | FE-softmax | K=1 | do five starts beat one? |
| C2 | selection | K=1 | as C1, for selection |
| C3 | FE-softmax | `k1_long` | do five starts beat one start with the same total iterations? |
| C4 | selection | `k1_long` | as C3, for selection |
| C5 | FE-softmax | HIO supp best5 | weak-prior classical baseline |
| C6 | FE-softmax | RAAR supp best5 | as C5 |
| C7 | FE-softmax | HIO sw best5 | matched-prior classical baseline |
| C8 | FE-softmax | RAAR sw best5 | as C7 |

**C9.** Spearman correlation between the FE of the FE-softmax reconstruction and −PSNR across the 100 images is positive. One-sided test at α = 0.05, with a percentile bootstrap interval of 10,000 resamples. C9 is outside the Holm family.

All tests are computed by `scripts/stage3_report.py` as committed at the tag.

## Descriptive analyses (no tests)

- Table of all methods: complex PSNR (mean, standard deviation, median, bootstrap CI), amplitude PSNR, phase error, NRMSE, SSIM, failures, exact recoveries, runtime.
- FE at the start level: within-image Spearman with PSNR, Top-1 and Top-2 agreement.
- Pixel-wise Spearman of u with |x̂ − x| for uniform, median and FE-softmax.
- Real and nonnegative HIO and RAAR as a reference for a wrong prior.

## Development check

It runs the five-start design and the classical baselines on the 8 development images. Its only purpose is to show that the code runs and the metrics behave on complex objects.
- It cannot change the design, the metrics, the thresholds, the hypotheses or the analysis.
- The held-out runs are done whatever the check shows, unless it reveals a software error. A fix is listed as a deviation, and the check is repeated.

## Order

1. Commit this plan with the code and configurations, push, and tag `stage3-plan-v1`.
2. Run the development check.
3. Run the held-out runs in any order: classical baselines, five starts, `k1_long`. An interrupted run resumes with the same command.
4. Run `scripts/stage3_report.py` unchanged.
5. Report every result, including unfavourable ones, and list any deviation.

## Commands

From a clone of the repository at the tag. Windows CMD, with the stage-2 Python environment active:

```
git clone https://github.com/hesamfj93/fe-multistart-phase-retrieval.git
cd fe-multistart-phase-retrieval
git checkout stage3-plan-v1
python -m pytest tests -q

rem development check (about 30 minutes)
python scripts\run_phase_retrieval.py --config configs\stage3_complex_dev.yaml
python scripts\run_baselines.py --config configs\stage3_complex_dev_baselines.yaml
python scripts\stage3_report.py --main results\stage3_complex_dev --baselines results\stage3_complex_dev_baselines --out results\stage3_complex_dev\report

rem held-out (about 9 hours)
python scripts\run_baselines.py --config configs\stage3_complex_heldout_baselines.yaml
python scripts\run_phase_retrieval.py --config configs\stage3_complex_heldout.yaml
python scripts\run_phase_retrieval.py --config configs\stage3_complex_heldout_k1long.yaml
python scripts\stage3_report.py --main results\stage3_complex_heldout --k1long results\stage3_complex_heldout_k1long --baselines results\stage3_complex_heldout_baselines --out results\stage3_complex_heldout\report
```

Without git, download the source archive of the tag `stage3-plan-v1` from the repository's Releases page instead of cloning.

## Not addressed by this stage

- measurement noise;
- objects other than MNIST digits;
- larger images;
- any change of the reconstruction design.

## Deviations

None so far.
