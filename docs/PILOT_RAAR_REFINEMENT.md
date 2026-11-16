# Pilot: refining the network output with RAAR (development images)

**Status: fixed before the computation.** This note and `scripts/pilot_raar_refinement.py` are committed before the script is run. Development images only; no held-out image is used.

## Question

At equal wall time RAAR with shrinkwrap beats FE-softmax on the held-out complex objects (`docs/STAGE3_EQUAL_COMPUTE.md`). Is the output of the untrained network a better starting point for RAAR than a random phase? If so, a combined method (network starts, then RAAR) could be worth a held-out test.

## Design

- **Images.** The 8 development images (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399) as complex objects, with the five stored starts of `results/spectral_ablation/complex_frozen`, the frozen design on these images.
- **RAAR.** RAAR with shrinkwrap exactly as in `configs/stage3_complex_heldout_raar45.yaml`: 5000 iterations, β = 0.87, 200 error-reduction iterations, the same shrinkwrap schedule, the initial support is the object window. Only the starting point differs.
- **Variants.**
  - `R1`: RAAR started from the FE-softmax output (one RAAR run per image).
  - `R5`: RAAR started from each of the five network starts; the result with the lowest FE against the measurement is kept.
  - `random`: RAAR from random phases, seeds 0 to 49, as in the held-out baseline; best of the first K starts by FE.
- **Equal wall time.** On the laptop GPU, FE-softmax took 128.7 s per image and one RAAR start 2.86 s. `R1` costs 128.7 + 2.86 s, the time of 46 random starts; `R5` costs 128.7 + 5 × 2.86 s, the time of 50 random starts. So `R1` is compared with the best of 46 and `R5` with the best of 50 random starts.
- **Measures.** Complex PSNR after alignment, capped at 60 dB; exact recovery ≥ 60 dB; failure < 15 dB; FE against the measurement. Descriptively, the share of the 40 refined network starts and of the 400 random starts that end exact.
- **Real objects** are not included: HIO with a real and nonnegative constraint is already exact on 99 of 100 held-out real digits, so refinement has no room there.
- The computation runs on a CPU. Individual RAAR starts depend on floating-point rounding (`docs/STAGE3_EQUAL_COMPUTE.md`), so all variants of this pilot run on the same device.

## Decision rule

Refinement is **promising** if `R1` has more exact recoveries than the best of 46 random starts, or `R5` more than the best of 50, in either case with no more failures, on the 8 images.

- Promising: write a plan for a held-out test on complex objects, ranks 300 to 399 of `splits/mnist_test_heldout_indices.csv`, and push it before any run.
- Not promising: report the pilot as a development-image result; the manuscript sentence that refinement "was not tested" is replaced by its outcome.

With 8 images the pilot is coarse; it decides only whether a held-out test is worth running.

## Results (added after the computation, 2026-11-16)

CPU run of `scripts/pilot_raar_refinement.py`; `results/pilot_raar_refinement/summary.json`. The stored member FE values were reproduced within 3 × 10⁻⁸.

| variant | mean PSNR | exact | failures |
|---|--:|--:|--:|
| FE-softmax (no refinement) | 19.94 | 0 | 1 |
| `R1`: RAAR from the FE-softmax output | 46.98 | 4 | 0 |
| `R5`: RAAR from each network start, best by FE | 51.60 | 6 | 1 |
| random, best of 5 | 33.32 | 3 | 4 |
| random, best of 46 | 51.34 | 6 | 1 |
| random, best of 50 | 51.34 | 6 | 1 |

- Per start, the network output is the better starting point: 16 of the 40 refined network starts ended exact and 10 failed, against 69 and 209 of the 400 random starts.
- At equal wall time it is not: `R5` equals the best of 50 random starts (6 exact, 1 failure), and `R1` has fewer exact recoveries than the best of 46 (4 against 6).

**Decision: not promising.** No held-out test is planned. One network start costs the wall time of nine random RAAR starts (128.7 / 5 / 2.86 = 9.0). At the pooled rates of this pilot, nine random starts contain an exact recovery with probability 1 − (1 − 69/400)⁹ = 0.82, one refined network start with 16/40 = 0.40.
