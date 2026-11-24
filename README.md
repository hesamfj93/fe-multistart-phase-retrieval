# Choosing among random starts in untrained-network phase retrieval

Code, configurations, frozen analysis plans, index lists and per-image results for the manuscript

> H. Faraji, "When Does Fourier-Error-Guided Multi-Start Selection Help? Evidence and Limitations in Untrained-Network Phase Retrieval," manuscript in preparation, 2026.

An untrained network recovers an image from the magnitude of its 2× oversampled Fourier transform. Independent random starts converge to solutions of different quality. This code runs K = 5 starts per measurement, aligns them for the trivial ambiguities (global phase, translation, conjugate twin) and scores them by their Fourier error (FE), the relative misfit to the measured magnitude. Two rules follow:

- **selection**: the start with the lowest FE;
- **fusion (FE-softmax)**: the aligned starts weighted by softmax(-FE / T), with T = 0.05.

The repository holds three held-out studies on MNIST digits and a test on a measured soft X-ray diffraction pattern. Each study has an analysis plan written before its held-out runs; the plan of Study 1 was written after an interim analysis of half of its images.

| study | objects | noise | untrained solvers | images | plan | file prefix |
|---|---|---|---|--:|---|---|
| 1 | real digits | none | ours | 100 | [`docs/PREREGISTRATION_STAGE2_100.md`](docs/PREREGISTRATION_STAGE2_100.md) (internal) | `stage2_*` |
| 2 | complex digits | none | ours | 100 | [`STAGE3_COMPLEX_PLAN.md`](STAGE3_COMPLEX_PLAN.md), tag `stage3-plan-v1` | `stage3_*` |
| 3 | real and complex digits | Poisson, two levels | ours, Net-ADM | 50 + 50 | [`STUDY3_PLAN.md`](STUDY3_PLAN.md), tag `study3-plan-v1` | `study3*` |
| real data | CXIDB entry 28 | measured | ours | 1 | [`REALDATA_DEV_PLAN.md`](REALDATA_DEV_PLAN.md), [`REALDATA_PLAN.md`](REALDATA_PLAN.md), tag `realdata-plan-v1` | `realdata*` |

The internal names `stage2` and `stage3` predate the numbering of the manuscript: stage 2 is Study 1 and stage 3 is Study 2.

## Main results

| comparison | result |
|---|---|
| five FE-guided starts vs one start, ten pre-specified comparisons (Studies 1 to 3) | better in all ten, by 1.05 to 3.36 dB |
| vs one start with the same total iterations (Studies 1 and 2) | supported for complex objects (+1.10 and +1.37 dB), not for real objects (+0.73 and +0.80 dB, Holm p = 0.058) |
| vs HIO (real, nonnegative) or RAAR (shrinkwrap) at equal wall time, selected by FE (Study 3) | the classical method is better in all six comparisons, by 5.8 to 24.0 dB; with five starts each, by 1.9 to 23.3 dB (post hoc) |
| FE of the fused result ranks images by error | Spearman 0.70 to 0.82 in eight runs, FE against the measurement (the frozen Study 3 report used the noise-free magnitudes; see [`docs/STUDY3_FE_NOTE.md`](docs/STUDY3_FE_NOTE.md)) |
| measured pattern (CXIDB 28) | our solver fails on this object; among RAAR starts the lowest FE selects a solution that disagrees with the published reconstruction, while all HIO starts agree with it (post-hoc look at all starts); a free FE on withheld pixels next to the beamstop ranks the agreeing solutions first, a free FE on random pixels does not (post hoc) |

Reports: [`docs/STAGE2B_RESULTS.md`](docs/STAGE2B_RESULTS.md) (Study 1), [`docs/STAGE3_RESULTS.md`](docs/STAGE3_RESULTS.md) (Study 2), [`results/study3/report/report.md`](results/study3/report/report.md) and [`results/study3/RUN_NOTES.md`](results/study3/RUN_NOTES.md) (Study 3), [`results/realdata_dev/report.md`](results/realdata_dev/report.md) and [`results/realdata/report.md`](results/realdata/report.md) (real data). The scripts in `paper/` whose names start with `make_tmlr` or `tmlr` rebuild the figures, tables and cross-study numbers of the manuscript from `results/`.

## Contents

| path | content |
|---|---|
| `pr_uq/` | the package: forward model, untrained network, trainer, alignment, aggregation, uncertainty, metrics |
| `scripts/` | runners, report generator, rescoring and post-hoc analyses |
| `configs/` | one YAML file per run; the held-out files are frozen (hashes below) |
| `splits/` | development (8), held-out (positions 0 to 199, and 200 to 399 for Study 3) and smoke (3) index lists of the MNIST test set |
| `data/MNIST/raw/` | the MNIST test set (IDX files), see [`data/README.md`](data/README.md) |
| `results/` | development results, the spectral-weight ablation and the held-out runs of both stages: per-image metrics, the five stored starts (`members.npz`), the long single start (`members.npz`) and the HIO and RAAR reconstructions (`baselines.npz`) |
| `paper/` | `make_paper_figures.py`, `make_stage3_figures.py` and `make_supplement_tables.py` regenerate the figures and supplementary tables from `results/` |
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

| result | source |
|---|---|
| Study 1: aggregation rules, paired comparisons, compute-matched control, classical methods | `results/stage2_mnist_heldout/report_n100/tables/heldout/primary_table.csv`, `paired_comparison.csv` |
| Study 1: FE against PSNR | `report_n100/tables/heldout/fe_psnr.csv` |
| Study 1: image-level score and pixel maps | `report_n100/tables/heldout/confirmatory_image_level.csv`, `report_n100/posthoc/posthoc.json` |
| Study 1: alignment sensitivity | `results/stage2_mnist_heldout/alignment_sensitivity.json` |
| Study 1: complex-field metrics, MAD rejection rules, Holm family | `report_n100/posthoc/posthoc_revision2.json`, `report_n100/posthoc/holm_family.csv` |
| Study 1: interim analysis, first 50 images | `generate_report.py ... --n-heldout 50` |
| Development study | `results/*.csv`, `results/mnist_k5_3000iter/`, `docs/AGGREGATION_FINALIZATION_REPORT.md` |
| Study 2: complex-valued objects | `results/stage3_complex_heldout/report/` (`summary.json`, `posthoc_stage3.json`), `docs/STAGE3_RESULTS.md` |

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
- The held-out list is a random permutation of the remaining test indices (NumPy `RandomState(20261006)`). Study 1 uses its positions 0 to 99 and Study 2 positions 100 to 199; Study 3 uses positions 200 to 249 and 300 to 349 (`splits/mnist_test_heldout3_indices.csv`).
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

## Study 2 (stage 3): complex-valued objects

A second held-out study uses complex-valued objects (MNIST amplitude, smooth random phase up to ±π/2), for which the real and nonnegative constraint does not apply. It reuses the frozen design unchanged and held-out ranks 100 to 199. The analysis plan, [`STAGE3_COMPLEX_PLAN.md`](STAGE3_COMPLEX_PLAN.md), was tagged `stage3-plan-v1` before any of those images was reconstructed. There were no deviations.

| comparison (complex PSNR, n = 100) | mean difference (dB) | 95% CI | supported |
|---|--:|---|---|
| FE-softmax fusion vs one start | +3.36 | [2.74, 3.98] | yes |
| FE-softmax fusion vs one start with the same total iterations | +1.37 | [0.73, 2.01] | yes |
| FE-softmax fusion vs HIO with shrinkwrap, best of 5 | +2.48 | [1.83, 3.12] | yes |
| FE-softmax fusion vs RAAR with shrinkwrap, best of 5 | −13.79 | [−17.71, −9.87] | no |

RAAR with shrinkwrap recovers 25 of the 100 images exactly and fails on 29; FE-softmax has no exact recovery and 21 failures, and is better on 28 of the 29 RAAR failures. All results, including post-hoc analyses labelled as such, are in [`docs/STAGE3_RESULTS.md`](docs/STAGE3_RESULTS.md). Code: `pr_uq/data/complex_objects.py`, the `shrinkwrap` constraint in `pr_uq/baselines/oversampled.py`, `scripts/stage3_report.py` (pre-specified), `scripts/stage3_posthoc.py` (post hoc), `paper/make_stage3_figures.py`, `configs/stage3_*.yaml`, `tests/test_stage3_complex.py`.

## Post hoc and diagnostic analyses after review

- **RAAR with shrinkwrap at equal wall time** (complex objects, held-out ranks 100 to 199): with 45 starts, the wall time of FE-softmax, RAAR with shrinkwrap was exact on 47 images, failed on 6 and was better than FE-softmax on 91 (laptop GPU, same device as the stored runs; an earlier CPU run gave 46, 5 and 90). Post hoc; C8 stays as reported. [`docs/STAGE3_EQUAL_COMPUTE.md`](docs/STAGE3_EQUAL_COMPUTE.md), `configs/stage3_complex_heldout_raar45.yaml`, `scripts/stage3_equal_compute_report.py`, `results/stage3_complex_heldout_raar45_gpu/`, `results/stage3_complex_heldout_raar45/`, `scripts/raar_rounding_sensitivity.py`.
- **Outlier rule of Hong et al. (2026)** on the stored starts of both studies: a MAD test on each start's squared residual, then a uniform mean. It matched the plain mean of the aligned starts and was 1.1 dB below FE-softmax in both studies. Post hoc, no new reconstruction. [`docs/POSTHOC_EOR_RULE.md`](docs/POSTHOC_EOR_RULE.md), `scripts/posthoc_eor_rule.py`, `results/posthoc_eor/`.
- **Capacity test** (8 development images): the frozen generator fitted directly to the ground truth reaches about 44 dB and a median FE of 0.014 to 0.017, so the FE floor of about 0.2 of the reconstructions is not a capacity limit. [`docs/CAPACITY_TEST.md`](docs/CAPACITY_TEST.md), `scripts/capacity_test.py`, `results/capacity_test/`.

## Spectral-weight ablation (development images)

The frozen design weights the data loss with a centered Gaussian in the Fourier domain that narrows to low frequencies. To check whether this weight causes the high Fourier error of the untrained reconstructions, three alternatives were run on the 8 development images, as real and as complex objects: no weight, a widening weight, and the frozen weight switched off for the last 20% of iterations. The plan, [`SPECTRAL_ABLATION_PLAN.md`](SPECTRAL_ABLATION_PLAN.md), and its decision rule were pushed before the runs.

Outcome A: no variant halves the start FE (largest reduction 24% on real and 28% on complex objects) and the frozen design stays. The spread between starts and the advantage of FE-softmax over a single start persist under every weight. Details: [`docs/SPECTRAL_ABLATION_RESULTS.md`](docs/SPECTRAL_ABLATION_RESULTS.md). Code: `unweighted_final_frac` in `pr_uq/optim/trainer.py`, `configs/spectral_ablation/`, `scripts/spectral_ablation_report.py`, `tests/test_spectral_ablation.py`; results in `results/spectral_ablation/`.

## Study 3: Poisson noise and a second untrained solver

The third study asks whether the FE protocol still helps under Poisson noise, whether it helps a second, independently published untrained solver (Net-ADM, Ma et al. 2024, re-implemented in `pr_uq/baselines/netadm.py`), and how both solvers compare with the strongest classical method at equal wall time. Its images are new: positions 200 to 399 of the held-out permutation (`splits/mnist_test_heldout3_indices.csv`), 50 real and 50 complex objects, at two noise levels.

- **Development phase** (8 development images; plan [`STUDY3_DEV_PLAN.md`](STUDY3_DEV_PLAN.md), pushed before its runs): Poisson levels N_p = 30,000 and 3,000 (`results/study3_dev/noise_levels.json`); a component ablation that kept the solver unchanged ([`docs/COMPONENT_ABLATION_RESULTS.md`](docs/COMPONENT_ABLATION_RESULTS.md)); development runs and the equal-wall-time numbers of classical starts (`results/study3_dev/`, `results/study3_dev/timing.json`, check of the report script in `results/study3_dev/report/`).
- **Analysis plan** for the held-out runs: [`STUDY3_PLAN.md`](STUDY3_PLAN.md), with configurations in `configs/study3/`, the report script `scripts/study3_report.py` and `tests/test_study3_configs.py`. It was tagged `study3-plan-v1` before any held-out run; the tag was pushed together with the results.
- **Results**: `results/study3/` (eleven runs, all 50 images each) and the report produced by the frozen script, `results/study3/report/`. How and where each run was made: [`results/study3/RUN_NOTES.md`](results/study3/RUN_NOTES.md).
- **FE used for N13**: the frozen report script read an FE computed against the noise-free magnitudes; the manuscript also gives N13 with the FE against the measurement. [`docs/STUDY3_FE_NOTE.md`](docs/STUDY3_FE_NOTE.md).
- **Compute-matched control** (one start with five times the iterations): plan [`STUDY3_LONG_PLAN.md`](STUDY3_LONG_PLAN.md), configurations in `configs/study3_long/`, report script `scripts/study3_long_report.py`, tag `study3-long-plan-v1`, pushed before the runs. Results in [`results/study3_long/report/report.md`](results/study3_long/report/report.md) and run notes in [`results/study3_long/RUN_NOTES.md`](results/study3_long/RUN_NOTES.md): five starts were better than the long start in 10 of 12 Holm-adjusted comparisons (FE-softmax by 1.24 to 2.22 dB, selection by 0.77 to 1.95 dB); no claim for our solver on real objects at moderate noise (+0.46 and +0.55 dB, CIs including zero).
- **Real and nonnegative object for our solver** (the constraint HIO receives in N7 and N8): plan [`STUDY3_REALNN_PLAN.md`](STUDY3_REALNN_PLAN.md), configurations in `configs/study3_realnn/`, report script `scripts/study3_realnn_report.py`, tag `study3-realnn-plan-v1`. Not yet run.

## Real data: CXIDB entry 28

`scripts/cxidb28_preprocess.py` centers, crops, bins and masks the diffraction pattern of CXIDB entry 28 (Marchesini et al., Phys. Rev. B 68, 140101, 2003; CC0) from the data alone; the result is `data/cxidb28/cxidb28_256.npz`. The raw data file is not part of the repository; it is available from the Coherent X-ray Imaging Data Bank.

- **Development on simulated twins** ([`REALDATA_DEV_PLAN.md`](REALDATA_DEV_PLAN.md), pushed before its runs): twins from `pr_uq/data/cxidb28_twin.py`, masked measurements in the solver, FE and the classical methods (`tests/test_masked_measurement.py`), runner `scripts/run_real_data.py`, configurations `configs/realdata_dev/`, results and report `results/realdata_dev/`. Checks made after the planned runs are in `results/realdata_dev/exploratory/`.
- **Measured pattern** ([`REALDATA_PLAN.md`](REALDATA_PLAN.md), tagged `realdata-plan-v1` before the reconstruction and pushed with the results): configurations `configs/realdata/`, report script `scripts/realdata_report.py`, results `results/realdata/`, and a post-hoc look at all classical starts in `results/realdata/exploratory/`.

## History of this repository

The main branch was rewritten once, on 12 November 2026, after Study 2, to remove trailers from commit messages; the file contents of every commit are unchanged. The tag `stage3-plan-v1` points to the original commit, whose message keeps its trailers; the later tags point to commits of the rewritten branch.

## License

Code: MIT, see [`LICENSE`](LICENSE). MNIST: see [`data/README.md`](data/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper once it is published.
