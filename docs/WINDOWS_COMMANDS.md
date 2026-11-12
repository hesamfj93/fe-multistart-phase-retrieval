# Windows CMD commands (authoritative)

Run everything in **Windows CMD** (not PowerShell), from the folder that contains `RUNBOOK.md` after extracting `pr_uq.zip`. The protocol behind these commands is described in `STAGE2_GPU_RUN.md`; the workflow and output layout in `RUNBOOK.md`.

## 0. Environment (once)

If you already have a Python 3.11 environment with PyTorch 2.5.1 + CUDA 12.1, activate it and skip to the last two lines of this block.

```cmd
cd pr_uq
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` deliberately does not list torch or torchvision, so it cannot replace your CUDA build. The MNIST test files are already in `data\MNIST\raw`.

## 1. CUDA verification

```cmd
python -c "import torch; print('torch', torch.__version__); print('CUDA build', torch.version.cuda); print('CUDA available', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected: `CUDA available True` and your GPU name. If it prints `False`, fix the torch installation first. The held-out runner refuses to fall back to CPU silently.

Then check the data, the split files and the code:

```cmd
python scripts\verify_dataset.py --dataset mnist --root .\data
python scripts\make_splits.py --check
python -m pytest tests -q
```

Expected: `[verify] ALL OK`, `[check] ALL OK`, `72 passed` on a machine with CUDA (64 without; the extra 8 are the GPU variants).

## 2. Smoke test (software verification, 3 images, reduced settings)

```cmd
python scripts\run_phase_retrieval.py --config configs\smoke_test.yaml
python scripts\generate_report.py --run-dir results\smoke_test
```

Expected: `3 ok (0 already complete), 0 failed`, then a printed `[smoke]` section. Open `results\smoke_test\environment.json` and confirm `"device_used": "cuda"`. Smoke numbers have no scientific meaning.

## 3. Held-out 50-image run

First a dry run. It trains nothing and writes nothing; it lists the 50 images and confirms there is no overlap with the development set.

```cmd
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout.yaml --dry-run
```

Expected last line: `split=heldout n=50 complete=0 to_run=50 device=cuda overlap_with_dev=[]`.

Then the real run:

```cmd
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout.yaml
```

Do not edit `configs\stage2_mnist_heldout.yaml` and do not pass `--iters` or `--K`. The design is frozen; the runner refuses these.

## 4. Resume after an interruption

Run exactly the same command again:

```cmd
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout.yaml
```

Completed images are skipped, a partially trained image continues from its last finished member, and images that failed are retried. To see progress without running anything:

```cmd
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout.yaml --dry-run
```

If the runner prints `PROTOCOL LOCK MISMATCH`, the configuration differs from the one the run started with. Restore the original file; do not delete `protocol_lock.json`.

## 5. Report generation

```cmd
python scripts\generate_report.py --run-dir results\stage2_mnist_heldout --n-heldout 50
```

Writes to `results\stage2_mnist_heldout\report_n50\`: `report.md`, `report_summary.json`, `tables\heldout\` (including `primary_table.csv`, the paper table) and `figures\heldout\`. Only held-out images enter the primary table.

## 6. Stage 2b: 100 held-out images, compute-matched control and classical baselines

Pre-registered in `PREREGISTRATION_STAGE2_100.md`. Do not edit any config before or during these runs.

Dry runs first. They train nothing and write nothing:

```cmd
python scripts\run_baselines.py --config configs\stage2_mnist_heldout_baselines.yaml --dry-run
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout.yaml --n-images 100 --dry-run
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout_k1long.yaml --dry-run
```

Expected last lines, in order:
- `n=100 complete=0 to_run=100 device=cuda` with 8 method names and `overlap_with_dev=[]`;
- `complete=50 to_run=50`;
- `complete=0 to_run=100`.

Then the three runs, one after another, in a single line that can run overnight (about 6 hours). Each run starts only if the previous one finished with 0 failures:

```cmd
python scripts\run_baselines.py --config configs\stage2_mnist_heldout_baselines.yaml && python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout.yaml --n-images 100 && python scripts\run_phase_retrieval.py --config configs\stage2_mnist_heldout_k1long.yaml
```

Approximate times on the RTX 4060 Laptop:
- baselines: under 1 hour;
- K=5 extension: 50 images × 2.2 min, about 1 h 50 min;
- compute-matched control: 100 images × 2.2 min, about 3 h 40 min.

If the line stops or the computer restarts, run the same line again. Completed images in every run are skipped.

Report:

```cmd
python scripts\generate_report.py --run-dir results\stage2_mnist_heldout --n-heldout 100 --extra-run results\stage2_mnist_heldout_k1long --extra-run results\stage2_mnist_heldout_baselines
```

It writes to `results\stage2_mnist_heldout\report_n100\`. The confirmatory test on ranks 50 to 99 is in `tables\heldout\confirmatory_image_level.csv` and at the end of the held-out section of `report.md`. The 50-image report in `report_n50\` is not touched.

## 7. Corrected uncertainty evaluation and the final report

Runs scored before the correction described in `STAGE2B_RESULTS.md` (deviation 1) compare the uncertainty map with the error in different frames. Rescore them once from the stored members (no training, about one minute on the CPU), then build the report:

```cmd
python scripts\rescore_uncertainty.py --run-dir results\stage2_mnist_heldout
python scripts\generate_report.py --run-dir results\stage2_mnist_heldout --n-heldout 100 --extra-run results\stage2_mnist_heldout_k1long --extra-run results\stage2_mnist_heldout_baselines
python paper\make_paper_figures.py --heldout results\stage2_mnist_heldout --k1long results\stage2_mnist_heldout_k1long --baselines results\stage2_mnist_heldout_baselines
```

Expected from the first command: `rescored 100 images, skipped 0, inconsistent 0`. It writes `uncertainty_v2.json` and `aggregates_v2.npz` next to each `metrics.json` and never changes `metrics.json`. The report uses the corrected values and states the evaluation version in `report.md`. New runs are scored with the corrected code and need no rescoring.

## 8. Post-hoc analyses (not part of the frozen plan)

```cmd
python scripts\alignment_sensitivity.py --run-dir results\stage2_mnist_heldout --n 100
python scripts\posthoc_analyses.py --run-dir results\stage2_mnist_heldout --k1long results\stage2_mnist_heldout_k1long --baselines results\stage2_mnist_heldout_baselines
python paper\make_paper_figures.py --skip-dev --heldout results\stage2_mnist_heldout --k1long results\stage2_mnist_heldout_k1long --baselines results\stage2_mnist_heldout_baselines
python paper\check_refs.py
```

What each command writes:
- the first: `results\stage2_mnist_heldout\alignment_sensitivity.json`;
- the second: `results\stage2_mnist_heldout\report_n100\posthoc\posthoc.json`;
- the third: regenerates the held-out figures of the paper;
- the fourth: checks every DOI in `paper\refs.bib` against Crossref (needs internet).

The numbers must match `STAGE2B_RESULTS.md`, section "Post-hoc analyses".

## Optional: development images under Stage-2 settings

Not needed for the paper table. Only for transparency, to show the 8 development images under the same frozen settings in a separate section:

```cmd
python scripts\run_phase_retrieval.py --config configs\stage2_mnist_dev8.yaml
python scripts\generate_report.py --run-dir results\stage2_mnist_heldout --n-heldout 50 --dev-run-dir results\stage2_mnist_dev8
```

The development section appears separately in `report_n50\report.md` and in `tables\development\`; it never enters `primary_table.csv`.

## Archive the results without the large arrays

The command below packs the three held-out run folders without the `.npz` arrays and model checkpoints:

```cmd
tar -a -c -f stage2b_results.zip --exclude=*.npz --exclude=*.pt results\stage2_mnist_heldout results\stage2_mnist_heldout_k1long results\stage2_mnist_heldout_baselines
```
