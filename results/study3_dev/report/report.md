# Third study (development check of the report script)

PSNR capped at 60 dB; failure below 15 dB. Classical: best of the first K starts by FE.

## Confirmatory family (Holm over N1 to N12)

| id | comparison | n | mean diff (dB) | 95% CI | median | first / second better | Wilcoxon p | Holm p | decision |
|---|---|--:|--:|---|--:|---|--:|--:|---|
| N1 | our solver, real, moderate: FE-softmax vs K=1 | 8 | +1.29 | [-1.58, +4.16] | +0.21 | 6 / 2 | 0.38 | 0.77 | no claim |
| N2 | our solver, real, strong: FE-softmax vs K=1 | 8 | +1.06 | [-0.28, +2.39] | +1.22 | 6 / 2 | 0.15 | 0.59 | no claim |
| N3 | our solver, complex, moderate: FE-softmax vs K=1 | 8 | +3.03 | [-0.01, +6.08] | +2.00 | 7 / 1 | 0.039 | 0.23 | no claim |
| N4 | our solver, complex, strong: FE-softmax vs K=1 | 8 | +4.30 | [+0.78, +7.82] | +3.03 | 7 / 1 | 0.016 | 0.14 | no claim |
| N5 | Net-ADM, real, moderate: FE-softmax vs K=1 | 8 | +3.94 | [+0.30, +7.57] | +3.12 | 7 / 1 | 0.023 | 0.16 | no claim |
| N6 | Net-ADM, real, strong: FE-softmax vs K=1 | 8 | +0.61 | [-0.12, +1.35] | +0.79 | 6 / 2 | 0.15 | 0.59 | no claim |
| N7 | real, moderate: our FE-softmax vs hio_realpos best of 5 | 8 | -23.37 | [-25.92, -20.82] | -22.56 | 0 / 8 | 0.0078 | 0.094 | no claim |
| N8 | real, strong: our FE-softmax vs hio_realpos best of 5 | 8 | -9.45 | [-14.55, -4.34] | -9.05 | 0 / 8 | 0.0078 | 0.094 | no claim |
| N9 | complex, moderate: our FE-softmax vs raar_sw best of 5 | 8 | -11.73 | [-24.49, +1.04] | -16.30 | 2 / 6 | 0.078 | 0.39 | no claim |
| N10 | complex, strong: our FE-softmax vs raar_sw best of 5 | 8 | -0.85 | [-4.75, +3.06] | -2.28 | 2 / 6 | 0.55 | 0.77 | no claim |
| N11 | real, moderate: Net-ADM FE-softmax vs hio_realpos best of 5 | 8 | -17.86 | [-19.44, -16.27] | -17.93 | 0 / 8 | 0.0078 | 0.094 | no claim |
| N12 | real, strong: Net-ADM FE-softmax vs hio_realpos best of 5 | 8 | -7.98 | [-12.46, -3.50] | -9.42 | 1 / 7 | 0.016 | 0.14 | no claim |

## N13: FE of the FE-softmax result against PSNR across images (one-sided, not in the Holm family)

| run | n | Spearman(FE, -PSNR) | bootstrap 95% CI | one-sided p |
|---|--:|--:|---|--:|
| real_ours_moderate | 8 | 0.83 | [0.23, 1.00] | 0.0051 |
| real_ours_strong | 8 | 0.67 | [-0.22, 1.00] | 0.035 |
| complex_ours_moderate | 8 | 0.93 | [0.41, 1.00] | 0.00043 |
| complex_ours_strong | 8 | 0.86 | [0.34, 1.00] | 0.0033 |
| real_netadm_moderate | 8 | 0.50 | [-0.34, 1.00] | 0.1 |
| real_netadm_strong | 8 | 0.83 | [0.31, 1.00] | 0.0051 |

## Untrained runs (descriptive)

| run | K=1 | selection | FE-softmax | uniform | median | best start | failures K=1 / FE-softmax | lowest FE = best start | median start FE | s per image |
|---|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|
| real_ours_moderate | 20.79 | 21.86 | 22.07 | 20.41 | 20.78 | 22.56 | 2 / 0 | 4 | 0.216 | 274 |
| real_ours_strong | 20.65 | 21.53 | 21.70 | 20.70 | 21.08 | 22.49 | 0 / 1 | 3 | 0.203 | 284 |
| complex_ours_moderate | 17.65 | 20.14 | 20.68 | 18.71 | 19.23 | 20.83 | 3 / 2 | 5 | 0.265 | 278 |
| complex_ours_strong | 15.43 | 19.51 | 19.72 | 17.71 | 18.35 | 19.78 | 5 / 1 | 6 | 0.267 | 227 |
| real_netadm_moderate | 23.66 | 27.31 | 27.59 | 26.64 | 28.02 | 28.55 | 1 / 0 | 4 | 0.070 | 76 |
| real_netadm_strong | 22.55 | 22.60 | 23.17 | 23.42 | 23.21 | 23.73 | 0 / 0 | 3 | 0.159 | 73 |

Net-ADM without noise (n = 8, descriptive): K=1 33.28, selection 36.60, FE-softmax 33.47, best start 37.34 dB.

## Classical runs (descriptive): mean PSNR (failures)

- real_hio_realpos_moderate: best of 1 40.53 (1); best of 5 45.45 (0)
- real_hio_realpos_strong: best of 1 27.85 (0); best of 5 31.15 (0)
- complex_raar_sw_moderate: best of 1 21.02 (6); best of 5 32.41 (3)
- complex_raar_sw_strong: best of 1 16.36 (5); best of 5 20.57 (3)
