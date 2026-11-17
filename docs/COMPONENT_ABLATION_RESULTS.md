# Component ablation of our solver: results

Plan and decision rule: `STUDY3_DEV_PLAN.md`, Section 3 (pushed before the runs, commit `4ca235b`). Development images only (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399), noiseless, K = 5, laptop GPU, 2026-11-16 21:59 to 2026-11-17 02:16. The frozen design is the stored run `results/spectral_ablation/{real,complex}_frozen`.

| variant | real: median start FE | real: mean start PSNR | real: start failures | complex: median start FE | complex: mean start PSNR | complex: start failures |
|---|--:|--:|--:|--:|--:|--:|
| frozen | 0.196 | 19.77 | 8/40 | 0.263 | 16.69 | 19/40 |
| no projection | 0.262 | 17.50 | 15/40 | 0.320 | 14.40 | 29/40 |
| no regularization | 0.283 | 15.79 | 19/40 | 0.360 | 12.77 | 34/40 |
| no gradient clipping | 0.176 | 20.29 | 4/40 | 0.274 | 16.17 | 22/40 |
| all off | 0.370 | 13.89 | 32/40 | 0.329 | 11.46 | 39/40 |

**Decision:** no variant qualifies. The frozen design remains "our solver" in the third study (`results/component_ablation/report/report.md`).

**Reading.** The β-mixing projection is worth 2.3 dB of mean start PSNR on both object types, and the TV and support terms 4.0 and 3.9 dB. Without clipping, the change is within 0.5 dB (+0.5 real, −0.5 complex). Switching off the projection or the regularization raises the median start FE, and without clipping it moves by 0.02 or less, so none of them causes the FE floor of about 0.2. Together with the capacity test and the spectral ablation, this points to the optimization itself (the local minima the generator reaches) rather than to the model size or its regularizers.
