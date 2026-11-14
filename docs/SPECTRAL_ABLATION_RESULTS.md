# Spectral-weight ablation: results on the 8 development images

Plan: [`SPECTRAL_ABLATION_PLAN.md`](../SPECTRAL_ABLATION_PLAN.md), committed as `51cdeac` and pushed at 2026-11-14 11:26 (local time), before any of the eight runs. No held-out image was used.

## Execution

| step | time (local, 2026-11-14) | outcome |
|---|---|---|
| plan, trainer option, configurations, report script pushed | 11:26 | commit `51cdeac` |
| `python -m pytest tests -q` in a fresh clone at `51cdeac` | before the runs | 93 passed |
| `real_frozen` to `real_dcfinal` | 11:28 to 12:45 | 4 runs × 8 images, all `ok` |
| `complex_frozen` to `complex_dcfinal` | 12:45 to 13:59 | 4 runs × 8 images, all `ok` |
| `scripts/spectral_ablation_report.py`, unchanged | 13:59 | `results/spectral_ablation/report/` |

One RTX 4060 Laptop GPU, the stage-2 environment; every `environment.json` records commit `51cdeac`. The report was run again on Linux from the stored `metrics.json` files: `summary.json` is identical and `report.md` differs only in line endings.

**Deviations from the plan: none.**

## Results

8 images × 5 starts per run. PSNR in dB, capped at 60; real-part PSNR for real objects, complex PSNR for complex objects; failure < 15 dB. "SD" is the mean within-image standard deviation of the start PSNR.

### Real objects

| variant | median start FE | mean start PSNR | start failures | SD | K=1 | selection | FE-softmax | best start | fused failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen | 0.196 | 19.77 | 8/40 | 2.66 | 19.17 | 22.75 | 22.43 | 23.22 | 0/8 |
| uniform | 0.155 | 20.03 | 10/40 | 2.61 | 19.22 | 21.75 | 21.62 | 22.94 | 1/8 |
| widening | 0.180 | 19.30 | 8/40 | 2.68 | 21.23 | 22.39 | 22.11 | 22.67 | 0/8 |
| dcfinal | 0.150 | 20.38 | 8/40 | 1.80 | 20.13 | 21.73 | 22.11 | 22.61 | 1/8 |

### Complex objects

| variant | median start FE | mean start PSNR | start failures | SD | K=1 | selection | FE-softmax | best start | fused failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| frozen | 0.263 | 16.69 | 19/40 | 2.53 | 16.11 | 20.05 | 19.94 | 20.15 | 1/8 |
| uniform | 0.190 | 17.33 | 17/40 | 2.36 | 16.07 | 19.39 | 20.08 | 19.91 | 1/8 |
| widening | 0.206 | 16.88 | 19/40 | 3.17 | 16.60 | 20.13 | 20.46 | 20.60 | 1/8 |
| dcfinal | 0.193 | 17.76 | 13/40 | 2.65 | 16.55 | 20.94 | 20.74 | 21.05 | 1/8 |

### Decision rule

| variant | start FE / frozen (real, complex) | Δ mean start PSNR (real, complex) | materially better | qualifies |
|---|---|---|---|---|
| uniform | 0.79, 0.72 | +0.26, +0.63 | no, no | no |
| widening | 0.92, 0.79 | −0.47, +0.19 | no, no | no |
| dcfinal | 0.76, 0.74 | +0.61, +1.06 | no, no | no |

"Materially better" needs a ratio ≤ 0.5 and a gain ≥ 1.0 dB on the same object type. `dcfinal` on complex objects meets the PSNR condition (+1.06 dB) but not the FE condition (0.74).

**Outcome A. The frozen design stays; no new held-out study is triggered.**

## Reading

- No variant halves the start FE. The largest reduction is 24% on real objects (`dcfinal`) and 28% on complex objects (`uniform`). The lowest median start FE of any run is 0.150, so the untrained reconstructions still do not fit the measurement closely. The spectral weight is therefore not the main reason for the high FE. Regularization, the β-mixing projection and network capacity were not varied and remain open.
- The start PSNR changes by −0.47 to +0.61 dB on real objects and by +0.19 to +1.06 dB on complex objects.
- The spread between starts persists under every weight: SD 1.8 to 2.7 dB on real objects, 2.4 to 3.2 dB on complex objects.
- FE-softmax stays above the mean start under every weight, by 1.6 to 2.8 dB on real objects and 2.8 to 3.6 dB on complex objects. Its gain over the single start K=1 is 0.9 to 3.3 dB on real objects and 3.8 to 4.2 dB on complex objects; K=1 is one seed per image, so this gain is the noisier of the two measures.
- With 8 images these numbers are descriptive. They support keeping the frozen design; they do not rank the four weights.

## Reproduce

No GPU needed:

```
python scripts/spectral_ablation_report.py --root results/spectral_ablation --out results/spectral_ablation/report
python paper/make_supplement_tables.py --run-dir results/stage2_mnist_heldout --stage3 results/stage3_complex_heldout/report --ablation results/spectral_ablation/report
```

The second command also writes `paper/supp/tab_ablation.tex`, Table S10 of the supplementary material.

Rerun the reconstructions (about 2.5 hours on the laptop GPU): the commands in `SPECTRAL_ABLATION_PLAN.md`, in a clone at `51cdeac`.

Stored per run: `config.yaml`, `protocol_lock.json`, `environment.json`, `split_manifest.json`, `split_indices_used.csv`; per image `members.npz`, `aggregates.npz`, `metrics.json`. Run logs and preview images are not stored, as in the held-out runs. The stored files are byte copies of the files written on Windows, so text files keep CRLF line endings.
