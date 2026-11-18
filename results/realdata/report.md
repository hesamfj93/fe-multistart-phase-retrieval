# Measured CXIDB 28 pattern: results

No ground truth. FE is the Fourier error over measured pixels.

## Classical methods with shrinkwrap

| method | FE, best of 1 | FE, best of 5 | FE, best of 10 | FE, best of 20 | FE, best of 50 | median start FE | s/start |
|---|--:|--:|--:|--:|--:|--:|--:|
| raar_shrinkwrap | 0.1151 | 0.1151 | 0.1151 | 0.1151 | 0.1125 | 0.1420 | 16 |
| hio_shrinkwrap | 0.1423 | 0.1423 | 0.1398 | 0.1398 | 0.1398 | 0.1437 | 13 |

Agreement of the two lowest-FE reconstructions: complex NRMSE of the HIO result, aligned to the RAAR result, relative to the RAAR result: 1.151. This measures reproducibility across methods, not accuracy.

## Our solver, run A, K = 5 (negative result of the development phase)

| start FE | FE k1 | FE uniform | FE median | FE best_single_by_fe | FE fe_softmax | s/start |
|---|--:|--:|--:|--:|--:|--:|
| 0.603, 0.503, 0.522, 0.502, 0.538 | 0.603 | 0.762 | 0.642 | 0.502 | 0.719 | 239 |
