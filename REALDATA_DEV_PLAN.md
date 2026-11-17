# Real-data demonstration, development phase on a simulated twin

**Status: fixed before the computations it describes.** Committed before any run listed here. No reconstruction of the measured pattern is made in this phase; the measured pattern is used only for its mask, its intensity statistics and the sum of its measured intensities.

## Data

CXIDB entry 28 (Marchesini et al., Phys. Rev. B 68, 140101, 2003; CC0): a soft X-ray pattern of 50 nm gold balls, wavelength 2.1 nm. `scripts/cxidb28_preprocess.py` finds the centre by centrosymmetry, crops 512 × 512 and bins 2 × 2. The result, `data/cxidb28/cxidb28_256.npz`, is a 256 × 256 pattern with the zero frequency at (128, 128), an object pixel of 17.9 nm and a 128 × 128 object field at oversampling 2. 900 binned pixels are unmeasured, all of them under the beamstop, and the measured pixels hold an unknown share of the intensity.

## Twin

`pr_uq/data/cxidb28_twin.py`, from the autocorrelation of the pattern only:
- two clusters of non-overlapping 50 nm balls, 37 to 40 balls in all, spanning about 111 × 29 pixels;
- complex exit wave exp(−k(β + iδ)t) − 1 with approximate gold constants. A full ball has amplitude 0.53.

On the twin, 50% of the intensity falls on measured pixels. The twin's measurement uses the mask of the real pattern and multiplicative log-normal noise with σ = 0.15. The real pattern's centrosymmetric pixel pairs differ by 12 to 16% of the intensity at every intensity level. Their relative centrosymmetry error is 0.012. The noisy twin gives 16 to 20% and 0.013; the noiseless twin gives 0.001.

Twin seeds 0, 1, 2, noise seed 0.

## Code added for masked data

A boolean mask of measured pixels is accepted by:
- the magnitude projection, which keeps the current field at unmeasured pixels;
- the data loss, which ignores them;
- the Fourier error and the anchor choice, which sum over measured pixels;
- HIO, RAAR and Net-ADM.

Without a mask, results are bitwise identical to the code of the held-out studies. This was checked against the previous commit for our solver, FE, the anchor, HIO and RAAR with every constraint, and Net-ADM. `tests/test_masked_measurement.py` checks that values at unmeasured pixels have no effect.

`scripts/run_real_data.py` runs one pattern. Our solver uses a 128 × 128 generator (z_hw 4, five upsampling stages, otherwise the frozen settings). Its data loss is multiplied by 0.00706 / mean(y²) over measured pixels, where 0.00706 is the median of mean(y²) over the 8 development complex objects. This gives the data term the size it has for the 32 × 32 objects, relative to the frozen regularization weights. Net-ADM is not used: it estimates a real object, and the exit wave of gold at 2.1 nm is complex.

## Questions and runs

1. With the frozen spectral weight, the data loss is concentrated on low frequencies by the end. These frequencies are partly under the beamstop. Does our solver do better without the spectral weight?
   - **A:** frozen design, 3000 iterations, K = 5.
   - **B:** the same without spectral annealing.
2. Do the classical methods with shrinkwrap recover the twin?
   - RAAR (β = 0.87) and HIO (β = 0.9), 5000 iterations plus 200 of error reduction, 50 starts each.
   - Reported as best of the first 1, 5, 10, 20 and 50 by FE.
3. Is the FE protocol useful here? FE-softmax and the lowest-FE start against the first start, for A and B.

All twin runs use the cloud CPU, with one thread per process. Runtimes are therefore not comparable with laptop-GPU runtimes.

## Decision

The solver variant for the real pattern is the one of A and B with the lower mean, over the three twins, of the complex NRMSE of the FE-softmax result. If both are above 1, the demonstration reports only the classical methods and our solver as a negative result, with no further tuning on the twin.

The classical methods and their number of starts for the real pattern are fixed in the real-data plan, using the twin runtimes and results. That plan is committed and tagged before the measured pattern is reconstructed.

## Not done in this phase

- any reconstruction of the measured pattern;
- changes of the solver other than A versus B;
- tuning of the twin to the results.
