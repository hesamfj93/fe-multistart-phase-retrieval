# Study 3, compute-matched control: report

PSNR in dB, capped at 60 dB, 50 images per run. Holm adjustment over L1 to L12.

| id | run | comparison | mean | 95% CI | median | first/second better | p | p (Holm) | decision |
|---|---|---|--:|---|--:|--:|--:|--:|---|
| L1 | real_ours_moderate | FE-softmax - k1_long | +0.46 | [-0.27, 1.19] | +0.45 | 31/19 | 0.12 | 0.19 | no claim |
| L2 | real_ours_strong | FE-softmax - k1_long | +1.24 | [0.56, 1.92] | +1.52 | 36/14 | 0.00023 | 0.0018 | first better |
| L3 | complex_ours_moderate | FE-softmax - k1_long | +2.20 | [1.24, 3.16] | +1.41 | 39/11 | 4.8e-06 | 5.7e-05 | first better |
| L4 | complex_ours_strong | FE-softmax - k1_long | +2.22 | [1.13, 3.32] | +1.90 | 37/13 | 0.00011 | 0.001 | first better |
| L5 | real_netadm_moderate | FE-softmax - k1_long | +1.33 | [0.32, 2.34] | +1.95 | 36/14 | 0.0031 | 0.014 | first better |
| L6 | real_netadm_strong | FE-softmax - k1_long | +1.24 | [0.67, 1.81] | +1.46 | 38/12 | 8.6e-05 | 0.00086 | first better |
| L7 | real_ours_moderate | selection - k1_long | +0.55 | [-0.27, 1.38] | +0.87 | 30/20 | 0.093 | 0.19 | no claim |
| L8 | real_ours_strong | selection - k1_long | +1.61 | [0.90, 2.33] | +2.02 | 37/13 | 2.1e-05 | 0.00023 | first better |
| L9 | complex_ours_moderate | selection - k1_long | +1.93 | [0.91, 2.96] | +0.64 | 35/15 | 0.0007 | 0.0042 | first better |
| L10 | complex_ours_strong | selection - k1_long | +1.95 | [0.82, 3.07] | +0.94 | 36/14 | 0.00055 | 0.0039 | first better |
| L11 | real_netadm_moderate | selection - k1_long | +1.69 | [0.54, 2.84] | +1.76 | 34/16 | 0.0054 | 0.016 | first better |
| L12 | real_netadm_strong | selection - k1_long | +0.77 | [0.26, 1.29] | +1.05 | 34/16 | 0.0028 | 0.014 | first better |

## Descriptive

| run | K=1 | k1_long | selection | FE-softmax | k1_long - K=1 [95% CI] | failures K=1 / k1_long / FE-softmax | median FE k1_long / selected start | FE of k1_long below selection (of these, higher PSNR) | s per image k1_long / five starts | identical measurements |
|---|--:|--:|--:|--:|---|---|---|---|---|---|
| real_ours_moderate | 18.09 | 20.21 | 20.77 | 20.67 | +2.12 [0.93, 3.32] | 13 / 3 / 1 | 0.192 / 0.198 | 16 (8) | 152 / 276 | 50 |
| real_ours_strong | 18.97 | 19.44 | 21.06 | 20.69 | +0.48 [-0.32, 1.27] | 9 / 5 / 2 | 0.229 / 0.213 | 17 (10) | 131 / 270 | 50 |
| complex_ours_moderate | 14.75 | 15.71 | 17.65 | 17.91 | +0.96 [-0.07, 2.00] | 31 / 25 / 13 | 0.275 / 0.239 | 13 (7) | 142 / 274 | 50 |
| complex_ours_strong | 14.70 | 15.77 | 17.72 | 18.00 | +1.07 [-0.14, 2.29] | 32 / 23 / 8 | 0.313 / 0.263 | 12 (7) | 130 / 256 | 50 |
| real_netadm_moderate | 24.20 | 25.32 | 27.01 | 26.65 | +1.12 [-0.07, 2.32] | 6 / 3 / 1 | 0.072 / 0.066 | 13 (6) | 36 / 73 | 50 |
| real_netadm_strong | 21.75 | 21.56 | 22.33 | 22.80 | -0.19 [-0.70, 0.32] | 0 / 0 / 0 | 0.173 / 0.155 | 8 (5) | 38 / 76 | 50 |
