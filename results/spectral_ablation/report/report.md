# Spectral-weight ablation, 8 development images

## real objects

| variant | median start FE | mean start PSNR | start failures | within-image SD | K=1 | selection | FE-softmax | best start | fused failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen | 0.1963 | 19.77 | 8/40 | 2.66 | 19.17 | 22.75 | 22.43 | 23.22 | 0/8 |
| uniform | 0.1550 | 20.03 | 10/40 | 2.61 | 19.22 | 21.75 | 21.62 | 22.94 | 1/8 |
| widening | 0.1801 | 19.30 | 8/40 | 2.68 | 21.23 | 22.39 | 22.11 | 22.67 | 0/8 |
| dcfinal | 0.1497 | 20.38 | 8/40 | 1.80 | 20.13 | 21.73 | 22.11 | 22.61 | 1/8 |

## complex objects

| variant | median start FE | mean start PSNR | start failures | within-image SD | K=1 | selection | FE-softmax | best start | fused failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen | 0.2626 | 16.69 | 19/40 | 2.53 | 16.11 | 20.05 | 19.94 | 20.15 | 1/8 |
| uniform | 0.1895 | 17.33 | 17/40 | 2.36 | 16.07 | 19.39 | 20.08 | 19.91 | 1/8 |
| widening | 0.2064 | 16.88 | 19/40 | 3.17 | 16.60 | 20.13 | 20.46 | 20.60 | 1/8 |
| dcfinal | 0.1932 | 17.76 | 13/40 | 2.65 | 16.55 | 20.94 | 20.74 | 21.05 | 1/8 |

## Decision

| variant | materially better (real, complex) | not worse (real, complex) | qualifies |
|---|---|---|---|
| uniform | False, False | True, True | False |
| widening | False, False | True, True | False |
| dcfinal | False, False | True, True | False |

Outcome A: the frozen design stays.
