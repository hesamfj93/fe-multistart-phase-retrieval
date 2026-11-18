# Third study

PSNR capped at 60 dB; failure below 15 dB. Classical: best of the first K starts by FE.

## Confirmatory family (Holm over N1 to N12)

| id | comparison | n | mean diff (dB) | 95% CI | median | first / second better | Wilcoxon p | Holm p | decision |
|---|---|--:|--:|---|--:|---|--:|--:|---|
| N1 | our solver, real, moderate: FE-softmax vs K=1 | 50 | +2.58 | [+1.74, +3.42] | +1.36 | 42 / 8 | 1.2e-09 | 7.2e-09 | first better |
| N2 | our solver, real, strong: FE-softmax vs K=1 | 50 | +1.72 | [+1.12, +2.31] | +1.59 | 36 / 14 | 6.8e-07 | 2e-06 | first better |
| N3 | our solver, complex, moderate: FE-softmax vs K=1 | 50 | +3.16 | [+2.18, +4.14] | +2.55 | 44 / 6 | 1.4e-09 | 7.2e-09 | first better |
| N4 | our solver, complex, strong: FE-softmax vs K=1 | 50 | +3.30 | [+2.37, +4.22] | +2.49 | 47 / 3 | 2.4e-13 | 1.9e-12 | first better |
| N5 | Net-ADM, real, moderate: FE-softmax vs K=1 | 50 | +2.45 | [+1.38, +3.51] | +2.13 | 39 / 11 | 1.5e-05 | 3e-05 | first better |
| N6 | Net-ADM, real, strong: FE-softmax vs K=1 | 50 | +1.05 | [+0.58, +1.52] | +0.96 | 38 / 12 | 2.8e-05 | 3e-05 | first better |
| N7 | real, moderate: our FE-softmax vs hio_realpos best of 81 | 50 | -24.01 | [-25.42, -22.60] | -24.52 | 0 / 50 | 1.8e-15 | 2.1e-14 | second better |
| N8 | real, strong: our FE-softmax vs hio_realpos best of 81 | 50 | -11.69 | [-13.06, -10.33] | -12.31 | 0 / 50 | 1.8e-15 | 2.1e-14 | second better |
| N9 | complex, moderate: our FE-softmax vs raar_sw best of 50 | 50 | -15.60 | [-18.89, -12.31] | -20.79 | 6 / 44 | 5.6e-11 | 3.9e-10 | second better |
| N10 | complex, strong: our FE-softmax vs raar_sw best of 50 | 50 | -5.78 | [-7.54, -4.03] | -6.19 | 12 / 38 | 1.6e-07 | 6.5e-07 | second better |
| N11 | real, moderate: Net-ADM FE-softmax vs hio_realpos best of 22 | 50 | -17.94 | [-19.49, -16.40] | -17.25 | 1 / 49 | 3.6e-15 | 3.6e-14 | second better |
| N12 | real, strong: Net-ADM FE-softmax vs hio_realpos best of 22 | 50 | -9.63 | [-10.89, -8.37] | -9.74 | 2 / 48 | 1.2e-14 | 1.1e-13 | second better |

## N13: FE of the FE-softmax result against PSNR across images (one-sided, not in the Holm family)

| run | n | Spearman(FE, -PSNR) | bootstrap 95% CI | one-sided p |
|---|--:|--:|---|--:|
| real_ours_moderate | 50 | 0.76 | [0.61, 0.85] | 5.4e-11 |
| real_ours_strong | 50 | 0.79 | [0.63, 0.88] | 4.4e-12 |
| complex_ours_moderate | 50 | 0.71 | [0.51, 0.83] | 3.5e-09 |
| complex_ours_strong | 50 | 0.68 | [0.49, 0.81] | 2e-08 |
| real_netadm_moderate | 50 | 0.71 | [0.52, 0.84] | 4.3e-09 |
| real_netadm_strong | 50 | 0.68 | [0.47, 0.82] | 3.5e-08 |

## Untrained runs (descriptive)

| run | K=1 | selection | FE-softmax | uniform | median | best start | failures K=1 / FE-softmax | lowest FE = best start | median start FE | s per image |
|---|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|
| real_ours_moderate | 18.09 | 20.77 | 20.67 | 19.40 | 19.76 | 21.27 | 13 / 1 | 32 | 0.251 | 276 |
| real_ours_strong | 18.97 | 21.06 | 20.69 | 19.44 | 19.91 | 21.14 | 9 / 2 | 42 | 0.281 | 270 |
| complex_ours_moderate | 14.75 | 17.65 | 17.91 | 16.45 | 16.69 | 18.23 | 31 / 13 | 29 | 0.322 | 274 |
| complex_ours_strong | 14.70 | 17.72 | 18.00 | 16.52 | 16.86 | 18.12 | 32 / 8 | 33 | 0.339 | 256 |
| real_netadm_moderate | 24.20 | 27.01 | 26.65 | 25.64 | 26.60 | 28.16 | 6 / 1 | 26 | 0.075 | 73 |
| real_netadm_strong | 21.75 | 22.33 | 22.80 | 22.58 | 22.78 | 23.17 | 0 / 0 | 20 | 0.167 | 76 |

Net-ADM without noise (n = 50, descriptive): K=1 30.54, selection 34.10, FE-softmax 32.56, best start 34.79 dB.

## Classical runs (descriptive): mean PSNR (failures)

- real_hio_realpos_moderate: best of 1 38.94 (4); best of 5 43.94 (1); best of 10 44.08 (1); best of 20 44.63 (0); best of 22 44.59 (0); best of 81 44.68 (0)
- real_hio_realpos_strong: best of 1 28.10 (3); best of 5 31.94 (0); best of 10 31.99 (0); best of 20 32.39 (0); best of 22 32.43 (0); best of 81 32.38 (0)
- complex_raar_sw_moderate: best of 1 17.15 (32); best of 5 25.23 (16); best of 10 28.38 (11); best of 20 31.20 (7); best of 50 33.51 (4)
- complex_raar_sw_strong: best of 1 17.29 (21); best of 5 19.87 (10); best of 10 21.73 (6); best of 20 22.79 (5); best of 50 23.78 (3)
