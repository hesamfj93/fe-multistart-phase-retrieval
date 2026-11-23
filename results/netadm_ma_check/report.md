# Net-ADM in the MNIST setting of Ma et al.: check

6 development images (MNIST test indices [429, 1729, 1917, 2890, 4919, 6044]), 28 x 28, r = 2 per dimension, no noise, seeds 0 to 9.

| method | mean PSNR (dB) | SD over 60 runs | per-image means | mean PSNR, no alignment | runs above 30 dB | mean runtime (s) |
|---|--:|--:|---|--:|--:|--:|
| Net-ADM | 30.74 | 9.25 | 18.9, 34.9, 34.7, 34.5, 22.3, 39.2 | 10.73 | 40 of 60 | 7.7 |
| HIO real nonnegative | 87.84 | 52.91 | 67.0, 107.0, 92.0, 95.9, 52.9, 112.3 | 8.48 | 44 of 60 | 1.6 |
