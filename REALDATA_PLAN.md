# Real-data demonstration on the measured CXIDB 28 pattern

**Status: fixed before any reconstruction of the measured pattern.** It is committed and tagged `realdata-plan-v1` locally before the runs below, and the tag is pushed to GitHub together with the results. The development phase on simulated twins is in REALDATA_DEV_PLAN.md, its results in `results/realdata_dev/report.md`, and checks made after it in `results/realdata_dev/exploratory/`.

## Outcome of the development phase

Complex NRMSE after alignment, three twins with log-normal noise (σ = 0.15) and the mask of the measured pattern. FE of the truth is 0.075.

| | twin 0 | twin 1 | twin 2 | mean |
|---|--:|--:|--:|--:|
| ours, run A, FE-softmax of 5 starts | 2.239 | 1.982 | 1.905 | 2.042 |
| ours, run B, FE-softmax of 5 starts | 2.378 | 2.576 | 2.767 | 2.574 |
| RAAR with shrinkwrap, first start | 1.019 | 0.910 | 1.051 | 0.993 |
| RAAR with shrinkwrap, lowest FE of 50 | 0.359 | 0.627 | 0.300 | 0.429 |
| HIO with shrinkwrap, first start | 0.664 | 0.658 | 0.831 | 0.718 |
| HIO with shrinkwrap, lowest FE of 50 | 0.763 | 0.552 | 0.742 | 0.686 |

Both runs of our solver are above 1. By the decision rule of the development plan, our solver is a negative result on this object and is not tuned further. Its starts reach FE 0.71 to 0.83 (A) and 0.34 to 0.41 (B), against 0.077 to 0.086 for the lowest-FE classical start. The exploratory checks show that the failure persists without noise and without the beamstop, and for a blurred twin, while on a development digit that the 32 × 32 solver reconstructs (NRMSE 0.374) the 128 × 128 generator gives 0.324. The failure follows the object, two compact clusters about 90 px apart covering 2% of the field.

For RAAR, the lowest-FE start of 50 lowers the mean NRMSE from 0.993 (first start) to 0.429. For HIO the gain is small (0.718 to 0.686).

## Runs on the measured pattern

Data: `data/cxidb28/cxidb28_256.npz`. The measured intensities are multiplied by 2.047e-9, so that their sum equals that of the noiseless twin 0 over the measured pixels (`scripts/run_real_data.py`, source `cxidb28`). All runs on the cloud CPU with one thread per process, as on the twins.

1. `configs/realdata/cxidb28_classical.yaml`: RAAR (β = 0.87) and HIO (β = 0.9) with shrinkwrap, 5000 iterations plus 200 of error reduction, 50 starts each, seeds 0 to 49. The settings of the twins, unchanged.
2. `configs/realdata/cxidb28_oursA.yaml`: our solver with the settings of run A, K = 5, seeds 0 to 4. Run to show whether the failure seen on the twins carries over to the measured pattern. It is reported as a negative result, whatever its FE.

## What is reported

There is no ground truth. `scripts/realdata_report.py` writes:
- the FE over measured pixels of the lowest-FE start among the first 1, 5, 10, 20 and 50 starts of each classical method, and the median start FE;
- the start FEs of our solver and the FE of its five reconstructions;
- the complex NRMSE between the lowest-FE HIO and RAAR reconstructions after alignment, as a measure of agreement between methods, not of accuracy;
- a figure of the amplitudes of the lowest-FE RAAR and HIO reconstructions and of the FE-softmax reconstruction of our solver.

The reconstruction for the paper is the lowest-FE RAAR start of 50, the method with the lower twin error at 50 starts. The comparison with the published reconstruction (Marchesini et al., 2003) is qualitative.

## Not done

- any change of methods, settings or number of starts after the measured pattern is reconstructed;
- other solvers. Net-ADM was excluded in the development plan because it estimates a real object. On the twins the phase of the exit wave varies by at most 0.17 rad over the support, so that reason is weaker than stated there; Net-ADM stays excluded because it was not tested on the twins.
