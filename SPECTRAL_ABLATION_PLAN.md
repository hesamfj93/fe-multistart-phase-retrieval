# Spectral-weight ablation on the development images

**Status: frozen.** This plan, the configurations and the report script are committed and pushed before any of the eight runs. Any later change is listed under "Deviations" with its date and reason.

## Why

The frozen design multiplies the data loss by a centered Gaussian weight in the Fourier domain:
- The width narrows from 0.5 to 0.05 of the 64-pixel detector side, that is from 32 to 3.2 pixels, over the first 60% of the 3000 iterations.
- It then stays at 3.2 pixels for the remaining 40%, so in that phase the loss sees mostly the lowest frequencies.

On noiseless data the untrained reconstructions of both held-out studies reach a median Fourier error (FE) of 0.23 (real objects) and 0.27 (complex objects). They never fit the measurement closely.

The spread between independent starts is what motivates the multi-start design. If a different spectral weight makes the network fit the measurement much better, part of that spread, and of the held-out results, may come from a weak base solver. This ablation asks whether that is the case.

## Design

- **Images.** The 8 development images (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399), once as real objects and once as complex objects with the stage-3 object model. No held-out image is used.
- **Unchanged.** Everything except the spectral weight is the frozen design: network, 3000 iterations, K = 5 with seeds 0 to 4, T = 0.05, alignment, aggregation rules, regularization, β-mixing projection and adaptive support.
- **Variants.** They differ only in trainer settings:

| variant | spectral weight | trainer settings |
|---|---|---|
| `frozen` | narrowing 0.5 → 0.05 over 60% of iterations, then constant | as in the held-out studies |
| `uniform` | none; unweighted data loss at every iteration | `use_spectral_annealing: false` |
| `widening` | 0.05 → 0.5 over 60% of iterations, then constant (coarse to fine) | `sigma_start_frac: 0.05`, `sigma_end_frac: 0.5` |
| `dcfinal` | as `frozen`, then unweighted in the last 20% of iterations (600 of 3000) | `unweighted_final_frac: 0.2` |

The eight runs are `configs/spectral_ablation/{real,complex}_{frozen,uniform,widening,dcfinal}.yaml`.
- `tests/test_spectral_ablation.py` checks that each file equals its base configuration (`stage2_mnist_dev8.yaml` or `stage3_complex_dev.yaml`) except for the declared trainer keys and the output folder.
- `unweighted_final_frac` defaults to 0. With that default the trainer gives bit-identical results to the frozen design; this was checked on two development images.

## Measures

Each run has 8 images × 5 starts. PSNR is capped at 60 dB. It is the PSNR of the real part for real objects and of the complex field for complex objects, as in the two held-out studies. A value below 15 dB is a failure.

- **Base solver:**
  - median FE of the 40 starts;
  - mean PSNR of the 40 starts;
  - start failures.
- **Spread:** mean within-image standard deviation of the start PSNR.
- **Aggregates:** mean PSNR of K = 1, selection, FE-softmax and the best start; FE-softmax failures; mean gain of FE-softmax over K = 1.

## Decision rule (fixed before the runs)

A variant V is **materially better** on an object type if both hold:
- its median start FE is at most half that of `frozen`;
- its mean start PSNR is at least 1.0 dB above that of `frozen`.

V **qualifies** if both conditions below hold:
- it is materially better on at least one object type;
- on the other object type, its mean start PSNR is not more than 0.5 dB below that of `frozen`.

- **Outcome A (no variant qualifies).** The frozen design stays. The eight runs are reported in the paper as a development-image analysis of the spectral weight.
- **Outcome B (one or more qualify).** The qualifying variant with the highest mean start PSNR, averaged over the two object types, becomes the design of a new held-out study.
  - It uses ranks 200 to 299 of `splits/mnist_test_heldout_indices.csv` for real objects and ranks 300 to 399 for complex objects. No method has used these images.
  - It gets its own analysis plan, frozen and tagged before its runs.
  - The two completed studies stay reported as conducted.

With 8 images the comparison is coarse. The thresholds are therefore set so that only a material change triggers a new study. The spread and gain measures are reported in either outcome.

`scripts/spectral_ablation_report.py` applies the rule and writes `results/spectral_ablation/report/`.

## Order

1. Commit this plan with the code, configurations and report script, and push. Optionally tag `spectral-ablation-plan-v1` through a GitHub release.
2. Run the eight configurations. This takes about 2.5 hours on the RTX 4060 Laptop GPU.
3. Run `scripts/spectral_ablation_report.py` unchanged.
4. Report the outcome, including outcome A.

## Commands

Windows CMD, with the stage-2 Python environment active, in a clone at the plan commit:

```
python -m pytest tests -q
for %c in (real_frozen real_uniform real_widening real_dcfinal complex_frozen complex_uniform complex_widening complex_dcfinal) do python scripts\run_phase_retrieval.py --config configs\spectral_ablation\%c.yaml
python scripts\spectral_ablation_report.py --root results\spectral_ablation --out results\spectral_ablation\report
```

## Deviations

None so far.
