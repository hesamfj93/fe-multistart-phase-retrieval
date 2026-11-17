# Third study: analysis plan (Poisson noise, two untrained solvers, classical methods at equal wall time)

**Status: frozen.** It is frozen at the git tag `study3-plan-v1`, which is pushed before any run on the images of this study. Any later change is listed under "Deviations" at the end, with its date and reason.

The development phase (`STUDY3_DEV_PLAN.md`) fixed the noise levels, kept the solver unchanged (`docs/COMPONENT_ABLATION_RESULTS.md`) and gave the timings for equal wall time (`results/study3_dev/timing.json`). Its results on the 8 development images are in `results/study3_dev`; they informed the choice of classical comparators below and nothing else.

## Questions

1. Under Poisson noise, does the FE protocol (five starts, aligned, FE-softmax) beat one start of the same solver, for our solver and for Net-ADM?
2. How do the two untrained solvers compare with the strongest classical method of the development phase, given the same wall time?
3. Is the FE of the FE-softmax result still an image-level quality signal under noise?

## Images

`splits/mnist_test_heldout3_indices.csv`, positions 200 to 399 of the held-out permutation, not used by any method so far.

| object type | ranks | n |
|---|---|--:|
| real (MNIST digit, values in [0, 1]) | 0 to 49 | 50 |
| complex (stage-3 object model, phase seed 1,000,003 + MNIST index) | 100 to 149 | 50 |

The runner checks that none of them is a development image. `STUDY3_DEV_PLAN.md` Section 5 named ranks 0 to 99 and 100 to 199; the size was halved to 50 per object type before any held-out run, to fit the compute budget, and the first 50 ranks of each range are used.

## Measurement

Intensity |FPx|² with the forward model of the first two studies (32 × 32 object, oversampling 2), then Poisson noise with N_p photons at the brightest pixel (`pr_uq/physics/noise.py`), noise seed 0 + MNIST index. The same noisy measurement is used by every method on an image.

| level | N_p | median FE of the ground truth on the development objects |
|---|--:|--:|
| moderate | 30,000 | 0.030 |
| strong | 3,000 | 0.098 |

Net-ADM is also run without noise, as a descriptive result.

## Methods

**Our solver.** The frozen design of the first two studies (`docs/COMPONENT_ABLATION_RESULTS.md` kept it), K = 5, seeds 0 to 4, T = 0.05. Real and complex objects.

**Net-ADM** (`pr_uq/baselines/netadm.py`, settings in `STUDY3_DEV_PLAN.md` Section 1), five runs with seeds 0 to 4, then the same protocol: FE, alignment, K = 1, uniform, median, selection, FE-softmax with T = 0.05. Real objects only.

**Classical, at equal wall time.** One method per object type, the best one of the development phase at both noise levels:
- real objects: HIO (β = 0.9) with the real and nonnegative constraint;
- complex objects: RAAR (β = 0.87) with shrinkwrap.

Each has 5000 iterations plus a 200-iteration error-reduction tail. The returned reconstruction is the start with the lowest FE among the first K starts (seeds 0 to K − 1). K is round(wall time of the untrained method with five starts / wall time of one classical start), from one laptop-GPU session of the development phase:

| comparison | K |
|---|--:|
| our solver vs HIO, real | 81 |
| Net-ADM vs HIO, real | 22 |
| our solver vs RAAR with shrinkwrap, complex | 50 |

The classical runs store all 81 or 50 starts; best of 22 is read from the first 22 starts of the 81-start run.

**Devices.** Untrained runs on the laptop GPU of the first two studies. Classical runs on the CPU of the cloud workspace; K is fixed above, so the device changes only rounding (`docs/STAGE3_EQUAL_COMPUTE.md`).

## Metrics

As in the first two studies: real objects, PSNR of the aligned real image; complex objects, complex PSNR after the four-variant alignment with global phase. Capped at 60 dB in summaries; below 15 dB is a failure.

## Hypotheses

**Decision rule** (as in `STAGE3_COMPLEX_PLAN.md`). For each comparison, first method minus second method, paired over the 50 images. A difference is claimed only if:
- the Holm-adjusted two-sided Wilcoxon signed-rank p is below 0.05, and
- the t-based 95% CI of the mean difference excludes zero.

The direction of the claim is the sign of the mean. The Holm family is N1 to N12 and nothing else.

| id | object | level | first | second | question |
|---|---|---|---|---|---|
| N1 | real | moderate | ours, FE-softmax | ours, K=1 | protocol, our solver |
| N2 | real | strong | ours, FE-softmax | ours, K=1 | |
| N3 | complex | moderate | ours, FE-softmax | ours, K=1 | |
| N4 | complex | strong | ours, FE-softmax | ours, K=1 | |
| N5 | real | moderate | Net-ADM, FE-softmax | Net-ADM, K=1 | protocol, Net-ADM |
| N6 | real | strong | Net-ADM, FE-softmax | Net-ADM, K=1 | |
| N7 | real | moderate | ours, FE-softmax | HIO real/nonneg, best of 81 | equal wall time |
| N8 | real | strong | ours, FE-softmax | HIO real/nonneg, best of 81 | |
| N9 | complex | moderate | ours, FE-softmax | RAAR shrinkwrap, best of 50 | |
| N10 | complex | strong | ours, FE-softmax | RAAR shrinkwrap, best of 50 | |
| N11 | real | moderate | Net-ADM, FE-softmax | HIO real/nonneg, best of 22 | |
| N12 | real | strong | Net-ADM, FE-softmax | HIO real/nonneg, best of 22 | |

**N13**, outside the Holm family, for each of the six untrained runs: Spearman correlation between the FE of the FE-softmax reconstruction and −PSNR across the 50 images is positive. One-sided test at α = 0.05, without correction, with a percentile bootstrap interval of 10,000 resamples.

**Expectation stated before the runs.** On the development images (n = 8, no claim), FE-softmax was 0.6 to 4.3 dB above K = 1 in every run. With five starts, not K, the classical method had the higher mean PSNR in every comparison, by 0.9 dB (complex objects, strong level) to 23 dB (real objects, moderate level). The study is run and reported whatever the outcome.

All numbers are computed by `scripts/study3_report.py` as committed at the tag.

## Descriptive analyses (no tests)

- For every untrained run:
  - mean, SD, median and failures of K=1, selection, FE-softmax, uniform, median and the best start;
  - how often the lowest-FE start is the best start;
  - median start FE;
  - runtime.
- Classical runs: best of the first 1, 5, 10, 20, 22 (real) and K starts.
- Net-ADM without noise: the same aggregates.

## Order

1. Commit this plan with the code and configurations, push, and tag `study3-plan-v1`.
2. Run the held-out runs in any order. An interrupted run resumes with the same command.
3. Run `scripts/study3_report.py` unchanged.
4. Report every result, including unfavourable ones, and list any deviation.

## Commands

From a clone at the tag. Untrained runs on the laptop (Windows CMD, the Python environment of the first two studies), about 18 hours:

```
python -m pytest tests -q
for %c in (real_ours_moderate real_ours_strong complex_ours_moderate complex_ours_strong real_netadm_moderate real_netadm_strong real_netadm_none) do python scripts\run_phase_retrieval.py --config configs\study3\%c.yaml
```

Classical runs on CPU:

```
for c in real_hio81_moderate real_hio81_strong complex_raar50_moderate complex_raar50_strong; do
  python scripts/run_baselines.py --config configs/study3/$c.yaml --device cpu
done
```

Report:

```
python scripts/study3_report.py --root results/study3 --out results/study3/report
```

## Not addressed by this study

- noise models other than Poisson;
- objects other than MNIST digits;
- changes of the reconstruction design;
- classical methods other than the two above at equal wall time.

## Deviations

None so far.
