# Smoke-test report

This report states what was executed to verify the package before the Stage-2 GPU run. It is software verification only. No number below is a scientific result.

## Environment

| item | value |
|---|---|
| device | **CPU only**. No GPU was available. CUDA execution was **not** verified. |
| Python | 3.12.3 (GPU machine: 3.11) |
| PyTorch | 2.14.1 (CPU execution; GPU machine: 2.5.1 + CUDA 12.1) |
| NumPy | 2.4.4 |
| OS | Linux |

The code uses no PyTorch feature newer than 2.1, but the CUDA path and Python 3.11 were not executed here. The smoke test on the GPU machine is the first CUDA execution.

## What was executed

| step | command / action | result |
|---|---|---|
| split files | `python scripts/make_splits.py --check` | `ALL OK`: files match deterministic regeneration, three sets pairwise disjoint |
| development mapping | development indices loaded from `splits/` and compared with the stored development ground truth | 8 of 8 images pixel-identical (max difference 0) |
| overlap | held-out first 50, 100 and 200 rows against the 8 development indices | 0 overlap in all three cases; smoke set disjoint from both |
| prefix | first 50 rows of the 100-image selection against the 50-image selection | identical |
| smoke run | `run_phase_retrieval.py --config configs/smoke_test.yaml` (3 smoke images, K=5, 400 iterations, z_channels 64) | 3 of 3 images OK, 0 failed, 149 s wall time |
| per-image runtime | from `metrics.json` | 31.1 to 43.9 s per image; 5.8 to 8.6 s per member |
| image-level resume | same command again | 3 of 3 images skipped as complete |
| member-level resume | 1-image run killed after 2 of 5 members, then restarted | members 1 and 2 reloaded, 3 to 5 trained, member cache removed; PSNR values identical to the uninterrupted run |
| output fields | programmatic check of `members.npz`, `aggregates.npz`, `metrics.json` and run-level files | 39 of 39 checks passed |
| report | `generate_report.py --run-dir results/smoke_test` | all tables and 6 figures written to the smoke section; no primary table created |
| guard: overrides | held-out config with `--iters 100` | refused |
| guard: CPU fallback | held-out config with `device: auto` and no CUDA | refused |
| guard: design change | same output directory with T changed from 0.05 to 0.1 | refused with `PROTOCOL LOCK MISMATCH` |
| dry run | held-out config with `--dry-run --device cpu` | listed 50 images, `overlap_with_dev=[]`, nothing written |
| unit tests | `python -m pytest tests -q` | **40 passed** |

## Output fields verified (39 checks)

Raw member stack, aligned member stack, member seeds 0 to 4, member FE, member PSNR and SSIM, ground truth and measured intensity, softmax weights summing to 1, anchor index, best-single index, effective ensemble size, all five reconstructions, PSNR/SSIM/FE for all five methods, uncertainty map and uncertainty metrics for the four ensemble methods, ensemble standard deviation map, physics residual map, per-member and total runtime, status and error fields, MNIST test index, split and rank, device per image, `config.yaml`, `environment.json`, environment history log, `split_manifest.json` (development indices, selected indices, split-file hash, overlap, frozen design), `split_indices_used.csv`, `protocol_lock.json`, `preview.png`, run log.

## Unit tests (40)

- 7 aggregation
- 4 alignment
- 7 calibration and AUROC
- 7 forward model and projection
- 15 protocol: split regeneration, development set content, zero overlap at 50/100/200, smoke disjointness, prefix stability, oversize request rejected, injected development index rejected, dataset order, report partitioning, primary table from held-out only, separate report folders for 50 and 100, design fingerprint, frozen Stage-2 values

## Limitations

- CUDA, Python 3.11 and PyTorch 2.5.1 were not executed here.
- Smoke settings (400 iterations, z_channels 64) are far below Stage-2 settings, so the runtimes above say nothing about GPU runtime at z_channels 128 and 3000 iterations.
- The smoke results directory is not included in the package. A smoke run on the GPU machine therefore starts from scratch and exercises the GPU instead of skipping completed images.
- No held-out image was reconstructed during verification. The held-out path was tested through the dry run and through unit tests on synthetic metrics files.

## Fix after the first run on the GPU machine (Windows, cp1256)

On the Windows machine, `pytest` gave 38 passed and 2 failed. Both failures were `UnicodeEncodeError` in `generate_report.py`: Windows used the system code page cp1256 to write `report.md`, and cp1256 has no `Δ`. On the real run this would have stopped the report step.

Fix: every text-file read and write in the active code now states `encoding="utf-8"` (29 call sites across the runner, the report generator, the config, logging and split modules, and the tests). Both scripts also replace unprintable console characters instead of raising. No scientific logic, split file or config changed; the split files and the Stage-2 config are byte-identical to the previous package.

Verified here under a strict ASCII locale, which is stricter than cp1256:
- the unfixed code reproduces the same 2 failures;
- the fixed code passes the 12 report and split tests;
- `generate_report.py` completes end to end with console output redirected to a file, and writes a valid UTF-8 `report.md` containing `Δ` and `±`;
- `make_splits.py --check` reports ALL OK.

PyTorch could not be installed in the CPU test environment for this change, so the runner changes (encoding arguments only) were checked by compilation and diff review, not by execution. The full suite (40 tests) and the smoke run on the GPU machine are the execution check.

## CUDA device bug found in the first held-out run

The first GPU runs (smoke and held-out) trained every member correctly, then failed each image at the aggregation step with `Expected all tensors to be on the same device ... cuda:0 and cpu`. Cause: the per-member Fourier errors are returned on the CPU, so the FE-softmax weights stayed on the CPU while the reconstructions were on the GPU. Every earlier verification ran on CPU only, where both are on the same device, so the bug could not appear. This is the limitation stated above ("CUDA execution was not verified") turning into a real defect.

Effect on data: none lost. A failed image keeps its 5 trained members in `per_image/<id>/_members/`, and the next run reloads them instead of retraining. Training code and all frozen settings are unchanged, so reusing those members is identical to a fresh run.

Fix:
- `scripts/run_phase_retrieval.py` moves the weights to the reconstructions' device;
- `pr_uq/uncertainty/ensemble_uq.py`: `fidelity_weights` keeps the device of its input, `_weighted_mean_std` moves weights to the stack's device, and the medoid distance no longer mixes devices;
- `runtime_total_s` is now training time from the member records plus post-processing time, so images finished from cached members report their true cost; `n_members_resumed` and `runtime_postprocess_s` are recorded too.

New `tests/test_device_path.py` runs the full per-image pipeline (training, all five methods, uncertainty, saving, image-level resume, and recovery after a post-training failure without retraining) on the CPU and, when a GPU is present, on CUDA. The CUDA cases are the ones that would have caught this bug.

Execution status (CPU test environment, PyTorch 2.15): full suite 44 passed, also under a strict ASCII locale; the command-line smoke run completed 3 of 3 images and the report generator succeeded; resumed and fresh images report `runtime_total_s` = training + post-processing. The 4 CUDA test cases exist only on a machine with a GPU (48 tests there) and were not executed here; the test suite on the GPU machine is the first execution of the CUDA path.
