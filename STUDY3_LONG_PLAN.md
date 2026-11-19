# Third study, compute-matched control: analysis plan (one long start against five FE-guided starts)

**Status: frozen.** It is frozen at the git tag `study3-long-plan-v1`, created before any long run on the held-out images. Unlike the earlier plans, the plan commit and the tag are pushed to GitHub before the first run, so the public record shows that the plan came first. Any later change is listed under "Deviations" at the end, with its date and reason.

## Why this control

In the third study (`STUDY3_PLAN.md`) FE-softmax with five starts was above one start (K = 1) in all six untrained runs, by 1.05 to 3.30 dB (N1 to N6). K = 1 used one fifth of the training iterations of the five-start protocol. The first two studies included a control with one start of five times the iterations (`k1_long`); the third study did not. This control asks whether the gains of N1 to N6 remain when the single start receives the same training budget.

## Question

Under Poisson noise, do five FE-guided starts (FE-softmax or selection) give a higher PSNR than one start with five times the iterations, for our solver and for Net-ADM?

## Images and measurements

Identical to the third study. `splits/mnist_test_heldout3_indices.csv`, real objects at ranks 0 to 49 and complex objects (phase seed 1,000,003 + MNIST index) at ranks 100 to 149. Poisson noise with N_p = 30,000 (moderate) and 3,000 (strong) photons at the brightest pixel, noise seed 0 + MNIST index.

The runner regenerates each measurement from the same seed. The report checks, image by image, that the stored measurement and ground truth equal those of the Study 3 run (column "identical measurements"). If any image differs, the report is still produced, the mismatch is listed as a deviation and its cause is found before any claim is made.

## Methods

**Five starts.** The stored Study 3 runs in `results/study3/<run>`, not rerun: K = 5, seeds 0 to 4, FE-softmax with T = 0.05 and selection (the start with the lowest FE).

**`k1_long`, our solver.** One start with seed 0, the seed of K = 1 in Study 3, and 15,000 iterations (5 × 3000). The annealing and β schedules are fractions of the run and stretch over the 15,000 iterations. The support refinement interval stays at 200 iterations, as in the `k1_long` runs of the first two studies. Everything else is the frozen design. Real and complex objects, both noise levels.

**`k1_long`, Net-ADM.** One start with seed 0 and 5000 outer iterations (5 × 1000) of 5 inner steps each. The step learning-rate schedule stretches with the run, so the learning rate is halved every 2500 outer iterations (5 × 500). Everything else as in Study 3. Real objects, both noise levels.

The six configurations are in `configs/study3_long`. Each differs from its Study 3 configuration only in the experiment name, the output folder, the method name `k1_long`, K = 1 and the iteration budget; `tests/test_study3_long_configs.py` checks this.

**Compute.** The training iterations are equal by construction. The five-start protocol adds FE, alignment and aggregation, which took 0.6 to 0.7 s per image on average in the Study 3 runs. The report gives the wall time per image of both, from the run records, as a descriptive result.

**Device.** The laptop GPU of the Study 3 runs (NVIDIA GeForce RTX 4060 Laptop GPU), with the same Python environment.

## Metrics

As in Study 3. Real objects, PSNR of the aligned real image; complex objects, complex PSNR after the four-variant alignment with global phase. Capped at 60 dB; below 15 dB is a failure.

FE is the error against the noisy measurement, which the runner stores for every start (`member_fe`). The entry `fe` under `methods` in `metrics.json` is computed against the noise-free Fourier magnitudes of the ground truth and is not used here.

## Hypotheses

**Decision rule** (as in `STUDY3_PLAN.md`). For each comparison, first method minus second method, paired over the 50 images. A difference is claimed only if
- the Holm-adjusted two-sided Wilcoxon signed-rank p is below 0.05, and
- the t-based 95% CI of the mean difference excludes zero.

The direction of the claim is the sign of the mean. The Holm family is L1 to L12 and nothing else.

| id | solver | object | level | first | second |
|---|---|---|---|---|---|
| L1 | ours | real | moderate | FE-softmax, five starts | `k1_long` |
| L2 | ours | real | strong | FE-softmax, five starts | `k1_long` |
| L3 | ours | complex | moderate | FE-softmax, five starts | `k1_long` |
| L4 | ours | complex | strong | FE-softmax, five starts | `k1_long` |
| L5 | Net-ADM | real | moderate | FE-softmax, five starts | `k1_long` |
| L6 | Net-ADM | real | strong | FE-softmax, five starts | `k1_long` |
| L7 | ours | real | moderate | selection, five starts | `k1_long` |
| L8 | ours | real | strong | selection, five starts | `k1_long` |
| L9 | ours | complex | moderate | selection, five starts | `k1_long` |
| L10 | ours | complex | strong | selection, five starts | `k1_long` |
| L11 | Net-ADM | real | moderate | selection, five starts | `k1_long` |
| L12 | Net-ADM | real | strong | selection, five starts | `k1_long` |

**Consequences fixed in advance**, for each run.
- Five starts better than `k1_long` (claimed): in that setting the five-start protocol is the better use of the training budget.
- No claim: in that setting the gain over K = 1 is not shown to exceed what one longer start gains, and the paper does not claim a more efficient use of compute there.
- `k1_long` better (claimed): in that setting one longer start is the better use of the budget, and the paper says that the comparison with K = 1 in Study 3 was not compute-matched.

The results of Study 3 (N1 to N13) stay as reported in every case.

**Expectation stated before the runs.** Without noise, `k1_long` was 1.03 dB (Study 1) and 1.99 dB (Study 2) above K = 1, and FE-softmax minus `k1_long` was +0.80 dB (Study 1, not supported after Holm adjustment) and +1.37 dB (Study 2, supported). If the long start gains as much under noise, FE-softmax minus `k1_long` is positive for our solver and near zero for Net-ADM at the strong level, where the gain over K = 1 was smallest (+1.05 dB). Longer fitting can also fit the noise, which would lower `k1_long`. The study is run and reported whatever the outcome.

All numbers are computed by `scripts/study3_long_report.py` as committed at the tag.

## Descriptive analyses (no tests)

For each of the six runs:
- mean, SD, median and failures of K = 1, `k1_long`, selection and FE-softmax;
- `k1_long` minus K = 1 (same seed, five times the iterations), paired mean with its 95% CI;
- median FE of `k1_long` and of the selected start;
- images on which `k1_long` has a lower FE than the selected start, and among them the images on which it also has the higher PSNR;
- wall time per image of `k1_long` and of the five-start run;
- images with identical measurements in both runs.

## Checks before the tag

- `scripts/study3_long_report.py --selftest` replaces `k1_long` by the stored K = 1 start. Its L1 to L6 rows then reproduce N1 to N6 of Study 3 (means +2.58, +1.72, +3.16, +3.30, +2.45 and +1.05 dB), and the measurement check compares each Study 3 run with itself (50 of 50 images).
- `configs/study3_long/real_netadm_strong.yaml` and `complex_ours_moderate.yaml` ran once at full length on development image rank 0 (MNIST 429), on the CPU of the cloud workspace, with the split, image count, output folder and device changed, to check that the runner stores `k1_long` and `member_fe` for K = 1. These runs are not part of the results and set nothing in this plan.

## Order

1. Commit this plan with the configurations, the report script and the test, and tag `study3-long-plan-v1`.
2. Push the commit and the tag to GitHub before the first run.
3. Run the six configurations on the laptop GPU, Net-ADM first. From the Study 3 timings (73 to 76 s per image for Net-ADM, 256 to 276 s for our solver) this takes about 2 hours for Net-ADM and 15 hours for our solver. An interrupted run resumes with the same command.
4. Run `scripts/study3_long_report.py` unchanged.
5. Report every result, including unfavourable ones, and list any deviation. Push the results after the author's approval.

## Commands

From a clone at the tag, on the laptop (Windows CMD, the Python environment of Study 3):

```
python -m pytest tests -q
for %c in (real_netadm_moderate real_netadm_strong real_ours_moderate real_ours_strong complex_ours_moderate complex_ours_strong) do python scripts\run_phase_retrieval.py --config configs\study3_long\%c.yaml
```

Report:

```
python scripts/study3_long_report.py --study3 results/study3 --long results/study3_long --out results/study3_long/report
```

## Not addressed by this control

- budgets other than five times one start;
- restarts or early stopping within one long run;
- the classical comparators and the measured pattern.

## Deviations

None so far.
