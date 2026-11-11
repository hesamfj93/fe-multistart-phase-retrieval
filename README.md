# FE-guided multi-start reconstruction for untrained-network phase retrieval

Code, configurations, index lists and per-image results for the paper

> H. Faraji, "Fourier-Error-Guided Multi-Start Reconstruction for Untrained-Network Phase Retrieval: Selection, Fusion and Reliability," submitted to *IEEE Transactions on Computational Imaging*, 2026.

An untrained double-decoder network recovers a 32×32 image from the magnitude of its 2× oversampled Fourier transform. Independent random starts converge to solutions of different quality. This code runs K = 5 starts per measurement, aligns them for the trivial ambiguities (global phase, translation, conjugate twin) and scores them by their Fourier error (FE), the relative misfit to the measured magnitude. Two operating modes follow:

- **selection**: the start with the lowest FE;
- **fusion**: the aligned starts weighted by softmax(-FE / T), with T = 0.05.

The evaluation uses 100 held-out MNIST test images with noiseless measurements, a compute-matched single-start control, and HIO and RAAR with two object constraints.

## Main results (100 held-out images)

| comparison | mean PSNR difference (dB) | 95% CI |
|---|--:|---|
| FE-softmax fusion vs one start | +1.82 | [1.29, 2.36] |
| selection vs one start | +1.76 | [1.21, 2.31] |
| FE-softmax fusion vs one start with the same total iterations | +0.80 | [0.21, 1.38], Holm p = 0.058, not supported |
| FE-softmax fusion vs support-only HIO, best of 5 | +3.47 | [2.94, 4.00] |
| FE-softmax fusion vs support-only RAAR, best of 5 | +6.45 | [5.96, 6.94] |

HIO with a real and nonnegative constraint recovers 99 of the 100 images exactly in this noiseless setting. All numbers, the hypotheses and the deviations from the analysis plan are in [`docs/STAGE2B_RESULTS.md`](docs/STAGE2B_RESULTS.md).

## Contents

| path | content |
|---|---|
| `pr_uq/` | the package: forward model, untrained network, trainer, alignment, aggregation, uncertainty, metrics |
| `scripts/` | runners, report generator, rescoring and post-hoc analyses |
| `configs/` | one YAML file per run; the held-out files are frozen (hashes below) |
| `splits/` | development (8), held-out (200) and smoke (3) index lists of the MNIST test set |
| `data/MNIST/raw/` | the MNIST test set (IDX files), see [`data/README.md`](data/README.md) |
| `results/` | development results and the three held-out runs: per-image metrics, the five stored starts (`members.npz`), the long single start (`members.npz`) and the HIO and RAAR reconstructions (`baselines.npz`) |
| `paper/make_paper_figures.py` | regenerates the figures of the paper from `results/` |
| `tests/` | unit and protocol tests |
| `docs/` | analysis plan, development report, results, runbook |

## Installation

Python 3.10 or later. Install PyTorch first, matching your CUDA version, then the rest:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` does not list torch, so it cannot replace a CUDA build with a CPU build. The paper's runs used Python 3.11.9, PyTorch 2.5.1, CUDA 12.1 and one NVIDIA RTX 4060 Laptop GPU.

## Quick check

```
python scripts/verify_dataset.py --dataset mnist --root ./data
python scripts/make_splits.py --check
python -m pytest tests -q
python scripts/run_phase_retrieval.py --config configs/smoke_test.yaml
python scripts/generate_report.py --run-dir results/smoke_test
```

The tests take under a minute on a CPU. The smoke run uses 3 images that are neither development nor held-out images, with reduced settings, and its numbers have no scientific meaning.

## Recompute the paper's numbers from the stored results

No GPU and no training are needed. The stored starts of every held-out image are in `results/stage2_mnist_heldout/per_image/*/members.npz`.

```
python scripts/generate_report.py --run-dir results/stage2_mnist_heldout --n-heldout 100 --extra-run results/stage2_mnist_heldout_k1long --extra-run results/stage2_mnist_heldout_baselines
python scripts/posthoc_analyses.py --run-dir results/stage2_mnist_heldout --k1long results/stage2_mnist_heldout_k1long --baselines results/stage2_mnist_heldout_baselines
python scripts/alignment_sensitivity.py --run-dir results/stage2_mnist_heldout --n 100
python paper/make_paper_figures.py --heldout results/stage2_mnist_heldout --k1long results/stage2_mnist_heldout_k1long --baselines results/stage2_mnist_heldout_baselines
python scripts/posthoc_revision2.py --run-dir results/stage2_mnist_heldout --k1long results/stage2_mnist_heldout_k1long --baselines results/stage2_mnist_heldout_baselines --mad
python paper/make_supplement_tables.py --run-dir results/stage2_mnist_heldout
```

`scripts/rescore_uncertainty.py` has already been applied to the stored run (see "Deviations" below). Running it again reproduces `uncertainty_v2.json` and `aggregates_v2.npz` and checks PSNR, SSIM, FE and the failure score against `metrics.json`.

| part of the paper | source |
|---|---|
| Table V, Sections VII-B to VII-D | `results/stage2_mnist_heldout/report_n100/tables/heldout/primary_table.csv`, `paired_comparison.csv` |
| Section VII-E (FE as a proxy) | `report_n100/tables/heldout/fe_psnr.csv` |
| Section VII-F, Table VI | `report_n100/tables/heldout/confirmatory_image_level.csv`, `report_n100/posthoc/posthoc.json` |
| Section VII-G (alignment) | `results/stage2_mnist_heldout/alignment_sensitivity.json` |
| complex-field metrics, MAD rejection baselines, Holm family (supplement Tables S2 to S4) | `report_n100/posthoc/posthoc_revision2.json`, `report_n100/posthoc/holm_family.csv` |
| interim analysis, first 50 images (supplement Table S5) | `generate_report.py ... --n-heldout 50` |
| Development study | `results/*.csv`, `results/mnist_k5_3000iter/`, `docs/AGGREGATION_FINALIZATION_REPORT.md` |

## Rerun the held-out evaluation

About 6 hours on the laptop GPU above. Dry runs first; they train nothing and write nothing:

```
python scripts/run_baselines.py --config configs/stage2_mnist_heldout_baselines.yaml --dry-run
python scripts/run_phase_retrieval.py --config configs/stage2_mnist_heldout.yaml --n-images 100 --dry-run
python scripts/run_phase_retrieval.py --config configs/stage2_mnist_heldout_k1long.yaml --dry-run
```

To rerun from scratch, move the stored runs out of `results/` first, because completed images are skipped. Then:

```
python scripts/run_baselines.py --config configs/stage2_mnist_heldout_baselines.yaml
python scripts/run_phase_retrieval.py --config configs/stage2_mnist_heldout.yaml --n-images 100
python scripts/run_phase_retrieval.py --config configs/stage2_mnist_heldout_k1long.yaml
```

The runner refuses `--iters` and `--K` for held-out runs, aborts if a held-out image is a development image, and aborts if the configuration differs from the one recorded in the run's `protocol_lock.json`. An interrupted run resumes when the same command is repeated. Exact Windows commands are in [`docs/WINDOWS_COMMANDS.md`](docs/WINDOWS_COMMANDS.md); the output layout is described in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Evaluation protocol and analysis plan

- The design and the temperature T were chosen on 8 development images (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399). They enter no reported comparison.
- The held-out list is a random permutation of the remaining test indices (NumPy `RandomState(20261006)`). The paper uses its first 100 entries.
- The analysis plan for the 100-image stage is [`docs/PREREGISTRATION_STAGE2_100.md`](docs/PREREGISTRATION_STAGE2_100.md). It was frozen before ranks 50 to 99, the compute-matched control and the classical baselines were run. It is an internal plan: it was not deposited in a public registry before the runs, and the paper describes it as a frozen analysis plan, not as a preregistration.

Frozen files (first 16 hex digits of SHA-256):

| file | hash |
|---|---|
| `configs/stage2_mnist_heldout.yaml` | 715b7b5b2e3cd694 |
| `configs/stage2_mnist_heldout_k1long.yaml` | a12f3788c00f1b82 |
| `configs/stage2_mnist_heldout_baselines.yaml` | 44929c90c6f55a59 |
| `splits/mnist_test_heldout_indices.csv` | ae054e4f1f5c13ec |

### Deviations

1. The first evaluation code compared the uncertainty map and the error map in different frames. The fix (`map_to_gt_frame` in `pr_uq/metrics/alignment.py`) was applied after the runs, and the pixel-wise uncertainty values were recomputed from the stored starts without retraining. PSNR, SSIM, FE and the image-level failure score were unaffected. The corrected values are in `uncertainty_v2.json`; the original `metrics.json` files are unchanged.
2. The expected calibration error listed in the plan is not reported, because the uncertainty map is a normalized score and not a probability.
3. Analyses added after the plan was executed are produced by `scripts/posthoc_analyses.py`, `scripts/alignment_sensitivity.py` and `scripts/posthoc_revision2.py`, and are labelled post hoc in the paper. They change no decision of the plan. The MAD rejection rules in `posthoc_revision2.py` use a threshold fixed in the script before they were run.

Details: [`docs/STAGE2B_RESULTS.md`](docs/STAGE2B_RESULTS.md).

## Stage 3: complex-valued objects (planned)

The analysis plan for a second held-out experiment, with complex-valued objects for which the real and nonnegative constraint does not apply, is [`STAGE3_COMPLEX_PLAN.md`](STAGE3_COMPLEX_PLAN.md). It reuses the frozen design unchanged, uses held-out ranks 100 to 199, and was tagged `stage3-plan-v1` before any of those images was reconstructed. Code: `pr_uq/data/complex_objects.py`, the `shrinkwrap` constraint in `pr_uq/baselines/oversampled.py`, `scripts/stage3_report.py`, `configs/stage3_*.yaml`, `tests/test_stage3_complex.py`.

## License

Code: MIT, see [`LICENSE`](LICENSE). MNIST: see [`data/README.md`](data/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper once it is published.
