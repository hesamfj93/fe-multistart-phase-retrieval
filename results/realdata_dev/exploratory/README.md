# Exploratory checks after the twin development phase

Not part of REALDATA_DEV_PLAN.md. Run after the nine planned twin runs had finished and before any reconstruction of the measured pattern. They change no decision of the plan. Cloud CPU, one thread.

## Is the failure of our solver an implementation error?

`diag.py`, one start (seed 0) of the run A settings, noiseless data, all pixels measured.

| case | object | field | FE | complex NRMSE | s |
|---|---|--:|--:|--:|--:|
| D1 | twin 0 | 128 | 0.543 | 0.909 | 237 |
| D2 | twin 0 blurred by a Gaussian of 2 px | 128 | 0.480 | 0.689 | 231 |
| D3 | complex dev digit 1729 | 128 | 0.192 | 0.324 | 234 |
| D4 | complex dev digit 1729 | 32 | 0.145 | 0.374 | 73 |
| D3 | complex dev digit 429 | 128 | 0.463 | 0.913 | 230 |
| D4 | complex dev digit 429 | 32 | 0.376 | 0.735 | 76 |

D4 uses the 32 × 32 setting of the held-out studies (data loss scale 1). Digit 429 is also a failure in the held-out pipeline (k1 FE 0.392 and NRMSE 0.933 in results/study3_dev/complex_ours_moderate). On digit 1729 the 128 × 128 generator matches the 32 × 32 result, so the 128 × 128 path works. On the twin the solver fails without noise and without the beamstop (D1), and also when the balls are blurred into two smooth blobs (D2). The failure follows the object (two compact clusters about 90 px apart, 2% of the field), not the noise, the mask, the fine scale or the field size.

`diag_D1_D2.png` shows D1 and D2.

## The FE aggregates on the classical starts

`fe_on_classical.py` applies the five reconstructions of the held-out protocol (T = 0.05) to the first K of the 50 cached starts of each classical method on each twin. Complex NRMSE in `fe_aggregates_on_classical_starts.json`. For K = 50 and the mean over the three twins:

| method | k1 | uniform | median | lowest FE | FE-softmax |
|---|--:|--:|--:|--:|--:|
| RAAR | 0.993 | 0.594 | 0.479 | 0.429 | 0.481 |
| HIO | 0.718 | 0.538 | 0.561 | 0.686 | 0.560 |

The classical start FEs differ by about 0.01, so FE-softmax at T = 0.05 is close to a uniform mean after alignment to the lowest-FE start.
