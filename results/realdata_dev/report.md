# Twin development phase: results

Complex NRMSE after alignment (lower is better; 1 is the error of a zero image) and FE over measured pixels.

## Our solver, run A

| twin | FE of truth | k1 | uniform | median | lowest FE | FE-softmax | best start | worst start | FE of FE-softmax | median start FE | lowest FE is best | s/start |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 0 | 0.075 | 1.609 | 2.162 | 1.608 | 1.406 | 2.239 | 1.390 | 5.114 | 0.957 | 0.816 | False | 251 |
| 1 | 0.075 | 1.026 | 1.817 | 1.041 | 1.222 | 1.982 | 1.026 | 7.009 | 0.852 | 0.765 | False | 250 |
| 2 | 0.075 | 2.485 | 1.600 | 1.616 | 2.658 | 1.905 | 1.316 | 2.658 | 0.809 | 0.769 | False | 242 |
| mean | | 1.707 | 1.860 | 1.422 | 1.762 | 2.042 | | | | | | |

## Our solver, run B

| twin | FE of truth | k1 | uniform | median | lowest FE | FE-softmax | best start | worst start | FE of FE-softmax | median start FE | lowest FE is best | s/start |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| 0 | 0.075 | 2.022 | 2.570 | 1.892 | 2.633 | 2.378 | 1.933 | 6.079 | 0.619 | 0.382 | False | 249 |
| 1 | 0.075 | 1.992 | 2.281 | 2.081 | 2.677 | 2.576 | 1.992 | 4.323 | 0.651 | 0.368 | False | 248 |
| 2 | 0.075 | 1.723 | 2.790 | 2.615 | 3.190 | 2.767 | 1.658 | 6.548 | 0.661 | 0.370 | False | 240 |
| mean | | 1.912 | 2.547 | 2.196 | 2.833 | 2.574 | | | | | | |

## Classical methods with shrinkwrap

| twin | method | best of 1 | best of 5 | best of 10 | best of 20 | best of 50 | FE, best of 50 | best start | median start | s/start |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | raar_shrinkwrap | 1.019 | 1.000 | 0.541 | 0.775 | 0.359 | 0.083 | 0.359 | 0.706 | 15 |
| 0 | hio_shrinkwrap | 0.664 | 0.664 | 0.763 | 0.763 | 0.763 | 0.082 | 0.553 | 0.763 | 13 |
| 1 | raar_shrinkwrap | 0.910 | 0.663 | 0.627 | 0.627 | 0.627 | 0.083 | 0.578 | 0.755 | 15 |
| 1 | hio_shrinkwrap | 0.658 | 0.658 | 0.658 | 0.546 | 0.552 | 0.083 | 0.543 | 0.711 | 12 |
| 2 | raar_shrinkwrap | 1.051 | 1.051 | 0.300 | 0.300 | 0.300 | 0.077 | 0.300 | 0.733 | 15 |
| 2 | hio_shrinkwrap | 0.831 | 0.623 | 0.623 | 0.742 | 0.742 | 0.086 | 0.523 | 0.764 | 12 |

## Decision

Mean complex NRMSE of FE-softmax over the three twins: A 2.042, B 2.574. Variant for the measured pattern: A. Both are above 1, so our solver is reported as a negative result.
