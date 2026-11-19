# Study 3: which FE the frozen report used for N13

Found on 19 November 2026, while writing the report script of the compute-matched control (`STUDY3_LONG_PLAN.md`), after all Study 3 results were reported.

## What the runner stores

For every start the runner stores `member_fe`, the FE against the measurement, `||m (|F P x| - sqrt(y))|| / ||m sqrt(y)||`. Selection and the FE-softmax weights use these values. For every aggregate it also stores `methods/<rule>/fe` in `metrics.json`, computed by `fourier_magnitude_error(x, ground truth)`, which compares `|F P x|` with `|F P x_gt|`. Without noise `|F P x_gt| = sqrt(y)`, so the two definitions agree; in Studies 1 and 2 they differ by less than 1e-7. Under Poisson noise they do not. The stored value then uses the noise-free magnitudes of the ground truth, which a user does not have.

## What it affects

- N13 of Study 3. `scripts/study3_report.py` reads `methods/fe_softmax/fe`. The FE of `STUDY3_PLAN.md` is the FE of the paper, against the measurement, so the frozen script did not compute the planned quantity.
- The post hoc AUROC of the manuscript (`paper/tmlr_auroc.py`) and the FE axis of the Study 3 reliability figure, before this correction.

It does not affect selection, FE-softmax, any PSNR, the Holm family N1 to N12, the real-data runs (`scripts/run_real_data.py` computes FE against the measurement) or Studies 1 and 2.

## Correction

`paper/tmlr_fe.py` rebuilds the FE-softmax aggregate from the stored starts and weights (`members.npz`); the rebuilt aggregate reproduces the stored noise-free FE to within 3e-8. Its FE against the measurement then follows exactly. `paper/tmlr_posthoc_review.py` recomputes N13 with it, using the bootstrap of the frozen script (same seed, same order of runs).

| run | N13, frozen report | N13, FE against the measurement [bootstrap 95%] | one-sided p |
|---|--:|--:|--:|
| ours, real, moderate | 0.76 | 0.77 [0.62, 0.86] | 2.6e-11 |
| ours, real, strong | 0.79 | 0.82 [0.67, 0.91] | 1.1e-13 |
| ours, complex, moderate | 0.71 | 0.71 [0.51, 0.83] | 3.7e-09 |
| ours, complex, strong | 0.68 | 0.70 [0.51, 0.82] | 5.6e-09 |
| Net-ADM, real, moderate | 0.71 | 0.72 [0.54, 0.84] | 1.9e-09 |
| Net-ADM, real, strong | 0.68 | 0.71 [0.52, 0.84] | 2.8e-09 |

The Spearman correlation between the two FE values across images is 0.992 to 0.999 per run. The conclusion of N13 does not change. The frozen report in `results/study3/report/` is left as it was produced; the manuscript gives both values. The AUROC of the manuscript now uses the FE against the measurement; it changed only for real objects under strong noise (0.86 to 0.88) and in the bootstrap interval for complex objects under strong noise.

The runner is not changed, so that every stored run stays reproducible. Report scripts written after this note read `member_fe` or `paper/tmlr_fe.py`; `scripts/study3_long_report.py` uses `member_fe`.
