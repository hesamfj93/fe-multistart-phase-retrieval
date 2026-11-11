# External Baselines — Installation & Integration

This file documents how to plug each external baseline into the
`run_baseline` adapter. No fake numbers are ever returned; if an external
baseline is not installed, the adapter raises `NotImplementedError`.

## 1. Zhuang Double-DIP

Repo: https://github.com/sun-umn/Double-DIP-for-PR
Paper: Zhuang, Yang, Hofmann, Barmherzig, Sun. *Practical Phase Retrieval
Using Double Deep Image Priors.* Electronic Imaging / IS&T 2023.

### Installation
```bash
git clone https://github.com/sun-umn/Double-DIP-for-PR third_party/double-dip-pr
cd third_party/double-dip-pr
pip install -r requirements.txt
```

### Integration plan
1. Wrap the authors' `run.py` entry point into a thin Python adapter that:
   - accepts `y_intensity` and `image_hw`,
   - calls Zhuang's code with their default hyperparameters,
   - returns the final complex reconstruction.
2. Add the wrapper to `adapters.run_baseline` under the `zhuang` branch.

Important: keep their hyperparameters at the published defaults for a fair
comparison; do not tune.

## 2. Net-ADM

Repo: search GitHub for "Net-ADM Fourier phase retrieval" (Ma et al. 2022).
Paper: arXiv:2210.12646.

Net-ADM uses ADMM with an untrained generative prior. Most of the code is
PyTorch-friendly. Plan:
1. Clone into `third_party/net-adm`.
2. Wire the `Net-ADM` runner into `adapters.run_baseline` (`net_adm`).
3. Pass our `y_intensity` after converting to their expected format
   (Ma et al. use magnitude rather than intensity in some places — verify).

## 3. Accelerated-TRAD

Paper: Ma et al. *Untrained neural network embedded Fourier phase retrieval
from few measurements.* J. Comp. Appl. Math., 2024 (arXiv:2307.08717).

Same group as Net-ADM. Their code is typically released alongside.

## 4. DeepMMSE

Paper: Chen et al. *Unsupervised Phase Retrieval Using Deep Approximate
MMSE Estimation.* IEEE TCI 2022, paper 9764626.

Public code: contact the authors. If not available, plan a **faithful
reimplementation** in `adapters.py` based on the published paper's
Algorithm 1. The key ingredients are:
- a single generative U-Net with **dropout** inserted at each layer,
- multiple stochastic forward passes per iteration,
- approximate MMSE estimate via dropout-based model averaging.

When reimplementing, be transparent in the paper that the comparison uses
our reimplementation, not author code.

## 5. LoDIP

Repo: https://github.com/raunakmanekar/LoDIP
Paper: Manekar et al. arXiv:2402.17745.

LoDIP targets low-photon CDI with time-series measurements; relevant only
for the Poisson noise sweep. Wire under `lodip` once installed.

## 6. DDRM-PR (diffusion baseline, optional)

Repo: https://github.com/METU-SPACE-Lab/ddrm-pr
Paper: Kaya & Oktem, *Applied Optics* 64:A95–A105 (2025).

Diffusion priors are outside our "untrained, dataset-free" framing but
we may include one diffusion result in supplementary for completeness.

---

## Sanity protocol when adding a new baseline

For every newly wired baseline:
1. Run it on the same 10-image MNIST split used for sanity.
2. Verify the median PSNR matches the published number within ±1 dB.
3. If it does not match, debug *before* including in any paper table.
4. Log to `results/<baseline_name>/sanity.json` the per-image PSNR,
   runtime, and the wrapper version that produced them.

Reviewers will check baseline numbers. Honest reproduction is more
valuable than impressive-looking numbers from misconfigured baselines.
