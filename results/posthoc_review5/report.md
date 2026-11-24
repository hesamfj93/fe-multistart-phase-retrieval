# Post hoc analyses of the fifth review (stored starts, no new runs)

## FE-softmax temperature: mean PSNR (dB)

| run | T=0.01 | T=0.02 | T=0.05 | T=0.1 | T=0.2 | T=0.5 | uniform | K=1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| S1 real, no noise | 20.07 | 20.13 | 19.99 | 19.65 | 19.32 | 19.08 | 18.87 | 18.17 |
| S2 complex, no noise | 17.62 | 17.74 | 17.67 | 17.36 | 17.03 | 16.74 | 16.50 | 14.31 |
| S3 ours real, moderate | 20.94 | 20.90 | 20.67 | 20.29 | 19.96 | 19.72 | 19.54 | 18.09 |
| S3 ours real, strong | 21.00 | 20.95 | 20.69 | 20.35 | 20.07 | 19.85 | 19.65 | 18.97 |
| S3 ours complex, moderate | 17.85 | 17.96 | 17.91 | 17.62 | 17.22 | 16.90 | 16.67 | 14.75 |
| S3 ours complex, strong | 18.01 | 18.10 | 18.00 | 17.67 | 17.30 | 16.99 | 16.76 | 14.70 |
| S3 Net-ADM real, moderate | 27.58 | 27.12 | 26.65 | 26.44 | 26.32 | 26.25 | 26.20 | 24.20 |
| S3 Net-ADM real, strong | 22.86 | 22.85 | 22.80 | 22.76 | 22.74 | 22.73 | 22.72 | 21.75 |

## Number of starts: mean PSNR (dB), selection / FE-softmax over the first k starts

| run | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|--:|--:|--:|--:|--:|
| S1 real, no noise | 18.17 / 18.17 | 19.31 / 19.28 | 19.51 / 19.56 | 19.90 / 19.84 | 19.93 / 19.99 |
| S2 complex, no noise | 14.31 / 14.31 | 15.29 / 15.69 | 16.39 / 16.83 | 17.08 / 17.40 | 17.40 / 17.67 |
| S3 ours real, moderate | 18.09 / 18.09 | 19.03 / 19.21 | 19.88 / 19.94 | 20.32 / 20.41 | 20.77 / 20.67 |
| S3 ours real, strong | 18.97 / 18.97 | 19.94 / 19.94 | 20.43 / 20.28 | 20.72 / 20.45 | 21.06 / 20.69 |
| S3 ours complex, moderate | 14.75 / 14.75 | 15.72 / 16.08 | 16.33 / 16.87 | 17.23 / 17.59 | 17.65 / 17.91 |
| S3 ours complex, strong | 14.70 / 14.70 | 15.77 / 16.06 | 16.54 / 16.91 | 17.50 / 17.73 | 17.72 / 18.00 |
| S3 Net-ADM real, moderate | 24.20 / 24.20 | 25.48 / 25.44 | 26.13 / 25.78 | 26.69 / 26.21 | 27.01 / 26.65 |
| S3 Net-ADM real, strong | 21.75 / 21.75 | 21.85 / 22.13 | 22.08 / 22.48 | 22.29 / 22.77 | 22.33 / 22.80 |

## PSNR inside the object (pixels where the ground truth is nonzero)

| run | K=1 | selection | FE-softmax | FE-softmax - K=1 inside [95% CI] | same, whole window |
|---|--:|--:|--:|---|---|
| S1 real, no noise | 10.77 | 12.24 | 12.39 | +1.62 [1.16, 2.07] | +1.82 [1.29, 2.36] |
| S2 complex, no noise | 7.32 | 9.99 | 10.24 | +2.92 [2.37, 3.46] | +3.36 [2.74, 3.98] |
| S3 ours real, moderate | 10.39 | 12.68 | 12.60 | +2.21 [1.48, 2.94] | +2.58 [1.74, 3.42] |
| S3 ours real, strong | 11.06 | 12.99 | 12.70 | +1.64 [1.13, 2.15] | +1.72 [1.12, 2.31] |
| S3 ours complex, moderate | 7.40 | 9.86 | 10.17 | +2.77 [1.90, 3.65] | +3.16 [2.18, 4.14] |
| S3 ours complex, strong | 7.29 | 10.00 | 10.25 | +2.96 [2.14, 3.78] | +3.30 [2.37, 4.22] |
| S3 Net-ADM real, moderate | 16.50 | 19.37 | 18.84 | +2.34 [1.27, 3.41] | +2.45 [1.38, 3.51] |
| S3 Net-ADM real, strong | 14.48 | 15.02 | 15.26 | +0.78 [0.26, 1.30] | +1.05 [0.58, 1.52] |

Checks: T = 0.05 and k = 5 reproduce the stored FE-softmax and selection PSNR to within the values below.

- S1 real, no noise: 2.4e-06 dB, 0 dB
- S2 complex, no noise: 6.6e-07 dB, 0 dB
- S3 ours real, moderate: 2.4e-06 dB, 0 dB
- S3 ours real, strong: 2.4e-06 dB, 0 dB
- S3 ours complex, moderate: 8.4e-07 dB, 0 dB
- S3 ours complex, strong: 6.3e-07 dB, 0 dB
- S3 Net-ADM real, moderate: 2.4e-06 dB, 0 dB
- S3 Net-ADM real, strong: 2.4e-06 dB, 0 dB
