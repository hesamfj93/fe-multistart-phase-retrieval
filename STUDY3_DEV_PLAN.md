# Development phase of the third study (Poisson noise, Net-ADM)

**Status: fixed before the computations it describes.** This plan, the code and the configurations are committed and pushed before any run listed here. Only the 8 development images are used (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399). The held-out images of the third study are not touched until its own analysis plan is pushed.

## Aim of the third study

The first two studies are noiseless, use one untrained solver, and at equal wall time RAAR with shrinkwrap beats it on complex objects. The third study asks three questions on new held-out images under Poisson noise:
1. Does the FE protocol (several starts, aligned, scored by FE, selected or fused) still help when the data are noisy?
2. Does it help a second, independently published untrained solver, Net-ADM (Ma et al., J. Comput. Appl. Math. 2024)?
3. How do the untrained methods compare with classical projection methods at equal wall time when exact recovery is no longer possible?

This development phase fixes everything the analysis plan of the third study needs.

## 1. Net-ADM

`pr_uq/baselines/netadm.py` re-implements Algorithm 1 of the paper, which has no public code, with its stated settings: ρ = 1, ε = 10⁻³, 1000 outer iterations, 5 inner Adam steps, learning rate 0.005 decayed once every 500 outer steps, the deep-decoder-like generator with widths {25, 15, 10} used for MNIST. The choices the paper leaves open are fixed in the module docstring: decay factor 0.5, magnitudes scaled to the unnormalized DFT, mean squared inner loss, real output. Net-ADM is for real signals, so it is used on real objects only.

In `scripts/run_phase_retrieval.py` a configuration with `solver: {type: netadm}` replaces the training of each start by one Net-ADM run with seed `base_seed + k`. Everything after the stack of starts is unchanged: FE, alignment, K = 1, uniform, median, selection, FE-softmax with T = 0.05, scoring. Without the `solver` key the runner is unchanged (all tests pass).

A first check, before this plan, ran Net-ADM with seeds 0 to 4 on the 8 development images, noiseless, real objects (CPU). Mean start PSNR was about 31 dB; the lowest-FE start was the best start on 6 of the 8 images.

## 2. Poisson noise levels

`pr_uq/physics/noise.py` scales the noiseless intensity to N_p photons at its brightest pixel, draws Poisson counts and rescales. The noise seed of an image is `noise_seed + MNIST index`, with `noise_seed = 0`.

**Rule.** For each N_p in {10³, 3·10³, 10⁴, 3·10⁴, 10⁵, 3·10⁵, 10⁶, 3·10⁶, 10⁷}, compute the FE of the ground truth against the noisy measurement for the 16 development objects (8 real, 8 complex). The **moderate** level is the N_p whose median is closest to 0.03, the **strong** level the one closest to 0.10. Only the ground truth and the noisy data enter this rule, no reconstruction. `scripts/study3_noise_levels.py` applies it and writes `results/study3_dev/noise_levels.json`.

## 3. Component ablation of our solver (noiseless)

The FE floor of our solver (median start FE about 0.2) is not a capacity limit (`docs/CAPACITY_TEST.md`), and the spectral weight is not its main cause (`docs/SPECTRAL_ABLATION_RESULTS.md`). The remaining candidates are switched off one at a time, and together:

| variant | trainer settings changed |
|---|---|
| `noproj` | `use_physics_projection: false` (no β-mixing projection) |
| `noreg` | `lambda_tv_A: 0`, `lambda_tv_phi: 0`, `lambda_support: 0` |
| `noclip` | `grad_clip: 0` (no gradient clipping) |
| `alloff` | all of the above |

Configurations: `configs/component_ablation/{real,complex}_{noproj,noreg,noclip,alloff}.yaml`; the frozen design is the stored `results/spectral_ablation/{real,complex}_frozen` (same configuration and code path).

**Decision rule**, as in `SPECTRAL_ABLATION_PLAN.md`: a variant is materially better on an object type if its median start FE is at most half that of the frozen design and its mean start PSNR at least 1.0 dB higher; it qualifies if it is materially better on one type and loses at most 0.5 dB of mean start PSNR on the other. If one or more qualify, the one with the highest mean start PSNR averaged over the two types becomes "our solver" in the third study; otherwise the frozen design stays. `scripts/component_ablation_report.py` applies the rule.

## 4. Development runs under Poisson noise

At the two levels of Section 2, on the 8 development images:
- our solver (the frozen design, and the selected variant if Section 3 selects one), real and complex objects, K = 5;
- Net-ADM, real objects, K = 5;
- classical baselines: HIO with the real and nonnegative constraint and RAAR with shrinkwrap on real objects, RAAR with shrinkwrap on complex objects, 5 starts each.

These runs check the pipeline under noise and give the timings for equal wall time. **The only things fixed from them** are the numbers of classical starts at equal wall time, computed from the laptop GPU timings of one session (`scripts/study3_dev_timing.py`): K_classical = round(mean wall time of the untrained method with 5 starts / mean wall time of one classical start), separately for each untrained solver and classical method. Hyperparameters, T = 0.05, alignment and scoring are not changed.

## 5. Then

The analysis plan of the third study (`STUDY3_PLAN.md`) is written from the results of this phase, committed, pushed and tagged before any run on its held-out images:
- **Images:** `splits/mnist_test_heldout3_indices.csv`, a new file that continues the held-out permutation at positions 200 to 399, so the held-out file of the first two studies (whose SHA-256 is in their protocol locks) is unchanged. Ranks 0 to 99 are used for real objects and ranks 100 to 199 for complex objects.
- **Measurement:** Poisson noise at the two levels.
