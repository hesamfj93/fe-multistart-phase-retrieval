# Third study, real and nonnegative object for our solver: analysis plan

**Status: frozen.** It is frozen at the git tag `study3-realnn-plan-v1`, created before any run of this test on the held-out images. The plan commit and the tag are pushed to GitHub before the first run. Any later change is listed under "Deviations" at the end, with its date and reason.

## Why this test

In the third study (`STUDY3_PLAN.md`), HIO with the real and nonnegative constraint was better than FE-softmax of our solver on real objects by 24.01 dB (N7, moderate noise) and 11.69 dB (N8, strong noise). Our solver estimates a complex object, x = A exp(iΦ), also for real objects, and its PSNR is computed on the real part. HIO was given the constraint that matches the objects; our solver was not. N7 and N8 therefore mix two effects: the class of method and the constraint. A reviewer pointed this out. This test gives our solver the same constraint and repeats the comparison.

## Question

On the real objects of Study 3, with Poisson noise:
1. Does a real and nonnegative object model change the PSNR of our solver?
2. With that model, is HIO with the real and nonnegative constraint still better than our solver?
3. With that model, do five FE-guided starts still beat one start?

## Images and measurements

Identical to Study 3: `splits/mnist_test_heldout3_indices.csv`, real objects at ranks 0 to 49, Poisson noise with N_p = 30,000 (moderate) and 3,000 (strong) photons at the brightest pixel, noise seed 0 + MNIST index. The report checks, image by image, that the stored measurement and ground truth equal those of the Study 3 run. If any image differs, the report is still produced, the mismatch is listed as a deviation and its cause is found before any claim is made.

## Methods

**Our solver, real and nonnegative object (new runs).** The frozen design of Study 3 with one change, `trainer.object_constraint: real_nonneg`: the network output is x = A, with A = sigmoid(·) in [0, 1] from the amplitude generator; the phase generator is not used (Φ = 0) and receives no gradient. Everything else is unchanged: K = 5, seeds 0 to 4, 3000 iterations, spectral annealing, the detached magnitude projection with the β-mixing schedule (the projection itself is not restricted to real values), adaptive support, FE-softmax with T = 0.05 and selection. The default value `complex` leaves the loop of every earlier study unchanged; on a fixed test problem the default path gives bit-identical output before and after the change.

The two configurations are `configs/study3_realnn/real_oursnn_moderate.yaml` and `real_oursnn_strong.yaml`. Each differs from its Study 3 configuration (`configs/study3/real_ours_<level>.yaml`) only in the experiment name, the output folder and `trainer.object_constraint`; `tests/test_study3_realnn.py` checks this.

**Comparators (stored Study 3 runs, not rerun).**
- our solver with the complex object model, `results/study3/real_ours_<level>`;
- HIO with the real and nonnegative constraint, the lowest-FE of the first 81 starts, `results/study3/real_hio81_<level>`, as in N7 and N8;
- Net-ADM, FE-softmax of five starts, `results/study3/real_netadm_<level>` (descriptive only).

**Compute.** The number of HIO starts (81) was fixed in `STUDY3_PLAN.md` for equal wall time against five starts of our solver with the complex object model. The real model skips the phase generator and is cheaper per iteration, so the comparison gives HIO at least as much wall time as our solver. The report gives the wall time per image of both models.

**Device.** The laptop GPU of the Study 3 runs (NVIDIA GeForce RTX 4060 Laptop GPU), with the same Python environment.

## Metrics

As in Study 3: PSNR of the aligned real image, capped at 60 dB; below 15 dB is a failure. FE is the error against the noisy measurement (`member_fe`).

## Hypotheses

**Decision rule** (as in `STUDY3_PLAN.md`). For each comparison, first method minus second method, paired over the 50 images. A difference is claimed only if
- the Holm-adjusted two-sided Wilcoxon signed-rank p is below 0.05, and
- the t-based 95% CI of the mean difference excludes zero.

The direction of the claim is the sign of the mean. The Holm family is R1 to R6 and nothing else.

| id | level | first | second |
|---|---|---|---|
| R1 | moderate | real-nonnegative model, FE-softmax | complex model, FE-softmax (Study 3) |
| R2 | strong | real-nonnegative model, FE-softmax | complex model, FE-softmax (Study 3) |
| R3 | moderate | real-nonnegative model, FE-softmax | HIO real and nonnegative, lowest FE of 81 starts |
| R4 | strong | real-nonnegative model, FE-softmax | HIO real and nonnegative, lowest FE of 81 starts |
| R5 | moderate | real-nonnegative model, FE-softmax | real-nonnegative model, K = 1 |
| R6 | strong | real-nonnegative model, FE-softmax | real-nonnegative model, K = 1 |

**Consequences fixed in advance.**
- R3 or R4, HIO better (claimed): with the constraint that matches the objects, our solver remains below HIO at that level. The paper reports R3 and R4 next to N7 and N8 and keeps its conclusion for real objects.
- R3 or R4, no claim or our solver better: at that level the paper no longer claims that HIO with the matching constraint is better than our solver on real objects; the abstract, Section 5.3 and the conclusion are revised accordingly.
- R1 or R2, real-nonnegative model better (claimed): the complex object model handicapped our solver on real objects, and the paper says so wherever N7 and N8 are reported. No claim or the complex model better: the paper reports it.
- R5 and R6: reported whatever the outcome, as evidence on whether the multi-start gain holds with the constrained model. They do not change N1 and N2.

The results of Study 3 (N1 to N13) and of its compute-matched control (L1 to L12) stay as reported in every case.

**Expectation stated before the runs.** The real and nonnegative model removes the twin image and the phase freedom, so we expect R1 and R2 to be positive. We expect R3 and R4 to stay negative: the median start FE of our solver (0.26 and 0.27) is far above that of the ground truth (0.03 and 0.10), and Net-ADM, which already estimates a real object, was 18.0 and 9.6 dB below HIO. The test is run and reported whatever the outcome.

All numbers are computed by `scripts/study3_realnn_report.py` as committed at the tag.

## Descriptive analyses (no tests)

For each level:
- mean, SD, median and failures of K = 1, selection and FE-softmax of the real-nonnegative model, FE-softmax of the complex model, HIO (lowest FE of 81) and Net-ADM FE-softmax;
- real-nonnegative FE-softmax minus Net-ADM FE-softmax, paired mean with its 95% CI;
- median start FE of both models of our solver;
- wall time per image of both models;
- images with identical measurements;
- largest absolute imaginary part and most negative real part over all stored starts of the new runs (both should be zero for the real and nonnegative model).

## Repeat of a frozen run (descriptive)

To answer whether a rerun of the same seeds on the laptop GPU reproduces the stored starts, `configs/study3_realnn/repeat_real_ours_moderate.yaml` reruns the frozen Study 3 configuration `real_ours_moderate` on its first five images (ranks 0 to 4) into a separate folder. The report gives the largest absolute difference of the per-start PSNR and FE, the number of images with the same selected start and the largest difference of the FE-softmax PSNR. The configuration differs from the Study 3 one only in the experiment name, the output folder and the image count.

## Checks before the tag

- `scripts/study3_realnn_report.py --selftest` uses the stored complex-model run in place of the real-nonnegative run. R1 and R2 are then zero on every image, R3, R4, R5 and R6 reproduce the means of N7, N8, N1 and N2 (−24.01, −11.69, +2.58 and +1.72 dB), the measurement check compares the Study 3 run with itself (50 of 50 images) and the repeat check compares the stored run with itself (differences zero).
- `configs/study3_realnn/real_oursnn_moderate.yaml` ran once at full length on development image rank 0 (MNIST 429), on the CPU of the cloud workspace, with the split, image count, output folder and device changed, to check that the runner stores real and nonnegative starts. This run is not part of the results and sets nothing in this plan. Its starts were real and nonnegative (largest imaginary part 0, smallest value 1.3e-6); on that image FE-softmax reached 13.69 dB, against 15.22 dB for the complex model in the Study 3 development run, and one image does not change the expectation above.

## Order

1. Commit this plan with the trainer option, the configurations, the report script and the test, and tag `study3-realnn-plan-v1`.
2. Push the commit and the tag to GitHub before the first run.
3. On the laptop GPU, run the repeat (five images), then the two real-nonnegative runs. At the speed of Studies 1 and 2 (about 26 s per start) this takes about 4 hours; at the speed of the Study 3 session, about 8 hours. An interrupted run resumes with the same command.
4. Run `scripts/study3_realnn_report.py` unchanged.
5. Report every result, including unfavourable ones, and list any deviation. Push the results after the author's approval.

## Commands

From a clone at the tag, on the laptop (Windows CMD, the Python environment of Study 3):

```
python -m pytest tests -q
for %c in (repeat_real_ours_moderate real_oursnn_moderate real_oursnn_strong) do python scripts\run_phase_retrieval.py --config configs\study3_realnn\%c.yaml
```

Report:

```
python scripts/study3_realnn_report.py --study3 results/study3 --realnn results/study3_realnn --out results/study3_realnn/report
```

## Not addressed by this test

- complex objects, where both our solver and RAAR already use complex models;
- Net-ADM with an explicit nonnegativity projection;
- oversampling below two and other untrained solvers.

## Deviations

None so far.
