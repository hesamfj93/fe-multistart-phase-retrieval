# Component ablation, 8 development images

## real objects

| variant | median start FE | mean start PSNR | start failures | within-image SD | K=1 | selection | FE-softmax | best start |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen | 0.1963 | 19.77 | 8/40 | 2.66 | 19.17 | 22.75 | 22.43 | 23.22 |
| noproj | 0.2620 | 17.50 | 15/40 | 2.91 | 18.14 | 20.61 | 20.78 | 20.88 |
| noreg | 0.2834 | 15.79 | 19/40 | 2.86 | 15.25 | 19.38 | 19.34 | 20.01 |
| noclip | 0.1762 | 20.29 | 4/40 | 2.81 | 21.44 | 23.52 | 22.80 | 23.69 |
| alloff | 0.3696 | 13.89 | 32/40 | 1.70 | 15.00 | 13.77 | 14.94 | 16.35 |

## complex objects

| variant | median start FE | mean start PSNR | start failures | within-image SD | K=1 | selection | FE-softmax | best start |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen | 0.2626 | 16.69 | 19/40 | 2.53 | 16.11 | 20.05 | 19.94 | 20.15 |
| noproj | 0.3204 | 14.40 | 29/40 | 2.46 | 13.32 | 16.73 | 17.34 | 17.66 |
| noreg | 0.3603 | 12.77 | 34/40 | 1.97 | 11.79 | 15.14 | 15.92 | 15.50 |
| noclip | 0.2744 | 16.17 | 22/40 | 2.68 | 16.11 | 19.38 | 19.73 | 19.80 |
| alloff | 0.3287 | 11.46 | 39/40 | 1.12 | 10.93 | 11.77 | 13.34 | 12.97 |

## Decision

| variant | materially better (real, complex) | not worse (real, complex) | qualifies |
|---|---|---|---|
| noproj | False, False | False, False | False |
| noreg | False, False | False, False | False |
| noclip | False, False | True, False | False |
| alloff | False, False | False, False | False |

Solver of the third study: `frozen`.
