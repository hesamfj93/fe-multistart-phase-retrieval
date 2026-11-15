# Capacity test of the generator (development images)

Diagnostic, not part of any plan. Script: `scripts/capacity_test.py` (committed as `82c4146` before the results were read). Results: `results/capacity_test/{real,complex}/summary.json`.

## Question

The untrained reconstructions keep a median start FE of about 0.2 on noiseless data (0.196 real, 0.263 complex on the development images; 0.23 and 0.27 for FE-softmax on the held-out images). Can the generator represent the objects at all?

## Design

- The 8 development images, as real objects and as complex objects with the stage-3 object model.
- The frozen member (two decoders with trainable latents, about 0.88 M weights), seed 0, Adam with the frozen learning rates (1e-3), gradient clipping at 1.0, 3000 iterations.
- Loss mean |x_net − x|² against the ground truth. No measurement, regularization, projection, spectral weight or support.
- PSNR after the alignment of the studies (real-part PSNR for real objects, complex PSNR for complex objects); FE of the fitted output against the noiseless magnitude.
- Run on CPU in the cloud workspace.

## Results (final iterate)

| objects | mean PSNR | min PSNR | median FE | max FE |
|---|--:|--:|--:|--:|
| real | 44.33 | 32.45 | 0.017 | 0.054 |
| complex | 44.70 | 35.03 | 0.014 | 0.050 |

The PSNR does not rise monotonically with Adam at this learning rate: image 429 (real) was at 47.2 dB after 2000 iterations and 32.5 dB after 3000, and image 1729 (complex) at 43.4 and 35.0 dB. Every image passed 43 dB at one of the checkpoints 500, 1000, 2000, 3000.

## Reading

Fitted to the object, the same generator with the same optimizer and budget reaches a median FE ten to twenty times below that of the reconstructions. The FE floor therefore comes from fitting the Fourier magnitude, that is from the optimization landscape and the loss design (regularization terms, β-mixing projection), not from the capacity of the generator. Which of these terms matters was not tested.
