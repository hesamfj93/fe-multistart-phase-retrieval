# RUNBOOK

Authoritative workflow for this project. Exact Windows commands are in **WINDOWS_COMMANDS.md**. The evaluation protocol and the decision gate are in **STAGE2_GPU_RUN.md**.

## Scripts

| script | purpose |
|---|---|
| `scripts/run_phase_retrieval.py` | the untrained-network runner (K=5 runs, and K=1 runs such as the compute-matched control) |
| `scripts/run_baselines.py` | classical HIO and RAAR baselines on the full oversampled grid, same images and metrics |
| `scripts/generate_report.py` | the only reporting command; `--extra-run` merges the control and baseline runs by image |
| `scripts/rescore_uncertainty.py` | recomputes the pixel-wise uncertainty metrics of a finished run in the ground-truth frame, from stored members, without training |
| `scripts/make_splits.py` | writes or checks (`--check`) the dev / held-out / smoke index files |
| `scripts/verify_dataset.py` | checks that the bundled MNIST files load |
| `scripts/aggregation_finalization.py` | development-stage analysis on the 8 development images; not used in Stage 2 |

Older scripts and configs are in `scripts/_archive/` and `configs/_archive/` for reference only. The archived `mnist_main.yaml` and `mnist50_k5_stage2.yaml` selected images with a seed-0 permutation whose first 8 images are exactly the development images, so they must not be used.

## Configs

| config | split | images | purpose |
|---|---|--:|---|
| `configs/smoke_test.yaml` | smoke | 3 | software verification, 400 iterations, z_channels 64 |
| `configs/stage2_mnist_heldout.yaml` | heldout | 50 (then 100) | main evaluation, frozen design |
| `configs/stage2_mnist_dev8.yaml` | dev | 8 | optional, development images under Stage-2 settings, reported separately |
| `configs/stage2_mnist_heldout_k1long.yaml` | heldout | 100 | compute-matched control: K=1, 15000 iterations, method `k1_long` |
| `configs/stage2_mnist_heldout_baselines.yaml` | heldout | 100 | HIO and RAAR, support-only and real+nonneg priors, 1 start and best of 5 by FE |

## Evaluation protocol in brief

- `splits/` holds three ordered, pairwise-disjoint lists of MNIST test indices: 8 development images, 200 held-out images, 3 smoke images. These files are the source of truth for every run.
- A run with `protocol_split: heldout` uses the first `n_images` rows of the held-out list. Going from 50 to 100 appends rows 50 to 99; the first 50 are unchanged.
- The runner aborts if a held-out or smoke selection contains a development index, if `--iters` or `--K` is passed for a dev or held-out run, if CUDA is missing while `device: auto` is set for a dev or held-out run, or if the configuration differs from the one recorded in the output directory's `protocol_lock.json`.
- The report generator assigns each image to a section by its index. Only held-out images enter `primary_table.csv`.

## What the runner does per image

Trains K = 5 members (seeds 0 to 4) once and derives all five methods from that stack.

| method | reconstruction | uncertainty |
|---|---|---|
| `k1` | member 0 | none |
| `uniform` | members aligned to member 0, uniform mean | ensemble std fused with physics residual |
| `median` | aligned to the best-FE member, pixel-wise median | same |
| `best_single_by_fe` | the member with the lowest Fourier error | borrowed from the aligned K=5 stack, not intrinsic |
| `fe_softmax` | aligned to the best-FE member, weights proportional to exp(-FE/0.05) | softmax-weighted std fused with physics residual |

## Output layout

```
results/<run>/
  config.yaml               merged configuration used
  protocol_lock.json        design fingerprint, split file hash (written on the first run)
  split_manifest.json       development indices, selected indices, overlap check, frozen design
  split_indices_used.csv    rank, MNIST test index, label
  environment.json          device, GPU, torch/CUDA/NumPy versions, git commit (latest session)
  logs/run.log, logs/environment_history.jsonl
  per_image/image_<MNIST index>/
    members.npz             ground truth, measured intensity, raw and aligned member stacks,
                            member seeds, member FE, member PSNR/SSIM (evaluation only), softmax weights
    aggregates.npz          aligned reconstruction of each method, fused uncertainty and physics
                            residual maps per ensemble method, ensemble std
    metrics.json            PSNR/SSIM/FE per method, uncertainty metrics, anchor index,
                            best-single index, effective ensemble size, runtimes, status, error,
                            MNIST index, split, rank, device
    preview.png
    uncertainty_v2.json     written by rescore_uncertainty.py: corrected pixel-wise uncertainty metrics
    aggregates_v2.npz       written by rescore_uncertainty.py: reconstructions, u and residual in the
                            ground-truth frame (used by the report figures when present)
  report_n50/               written by generate_report.py --n-heldout 50
    report.md, report_summary.json
    tables/heldout/         primary_table.csv, summary.csv, paired_comparison.csv, uncertainty.csv,
                            failure.csv, fe_psnr.csv, effective_size.csv, per_image.csv
    figures/heldout/        method_boxplots, paired_psnr, fe_psnr_scatter, qualitative_grid,
                            uncertainty_vs_error, success_failure_examples
```

## Resume and failures

- An image whose `metrics.json` has `status: ok` is skipped. A partly trained image reloads its finished members from `per_image/<id>/_members/`; that cache is deleted when the image completes.
- If an image raises an error, `metrics.json` records `status: failed` with the traceback and the run continues. Failed images are retried on the next run and are listed in the report's integrity section.
- `--dry-run` lists which images would run or be skipped, without training or writing anything.

## Wording used in all reports

- The plain-mean degradation found during development is a **uniform aggregation failure**.
- FE is a **practical, empirically evaluated, ground-truth-free reliability proxy**, not a guaranteed best-member estimator.
- Selection (best-single-by-FE) and fidelity-weighted fusion (FE-softmax) are two complementary operating modes.

## Reports in this folder

| file | status |
|---|---|
| `AGGREGATION_FINALIZATION_REPORT.md` | authoritative report for the 8-image development stage |
| `STAGE2_DECISION_GATE_50.md` | decision gate applied to the first 50 held-out images |
| `PREREGISTRATION_STAGE2_100.md` | frozen plan, hypotheses and rules for the 100-image stage |
| `STAGE2B_RESULTS.md` | outcome of the 100-image stage, hypotheses H1 to H4 and the deviations |
| `SMOKE_TEST_REPORT.md` | what was executed to verify this package |
| `FINAL_ALIGNMENT_REPORT.md`, `FINAL_DIAGNOSTIC_REPORT.md`, `DIAGNOSTIC_REPORT_MNIST.md` | superseded, historical only |

## Not yet

Fashion-MNIST and CIFAR-10 are not configured under this protocol. Each will need its own development and held-out split before any run, and CIFAR-10 needs an explicit decision on the colour formulation first.
