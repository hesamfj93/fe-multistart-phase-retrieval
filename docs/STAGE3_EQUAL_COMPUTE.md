# Post hoc: RAAR with shrinkwrap at equal wall time (complex objects)

**Not part of `STAGE3_COMPLEX_PLAN.md`.** The confirmatory comparison C8 (FE-softmax against RAAR with shrinkwrap, best of 5 starts) stays as reported. This analysis was added after a review noted that FE-softmax used about nine times the wall time of that baseline.

## Design

- Held-out ranks 100 to 199 (complex objects), the images of the second study.
- FE-softmax took 128.7 s per image and one RAAR-shrinkwrap start 2.86 s, both on the same laptop GPU in the stored runs, so RAAR with shrinkwrap gets 128.7 / 2.86 = 45 starts.
- `configs/stage3_complex_heldout_raar45.yaml` equals `configs/stage3_complex_heldout_baselines.yaml` except for algorithms, constraints, number of starts and output folder: same measurement, 5000 iterations with a 200-iteration error-reduction tail, β = 0.87, shrinkwrap settings, seeds 0 to 44 and metrics. The best start is the one with the lowest FE against the measurement, as in `best5`.
- The configuration and `scripts/stage3_equal_compute_report.py` were committed (`86c3589`) before the first run.

## Two runs

**GPU run (primary).** `results/stage3_complex_heldout_raar45_gpu`, on the laptop GPU of the stored `best5` run (RTX 4060 Laptop GPU, commit `b657fd7`), 2026-11-16, 15:30 to 21:47 (local time), with the official runner `python scripts/run_baselines.py --config configs/stage3_complex_heldout_raar45.yaml --device cuda`, output folder changed on the command line. Seeds 0 to 4 reproduced the stored `best5` run: all 500 starts within 0.1 dB and the same selected start on all 100 images. The rows K ≤ 5 below are therefore those of the stored run. One start took 5.0 s on average in this session against 2.86 s in the stored run (state of the machine); the number 45 is based on the stored timing and was not changed.

**CPU run (earlier record).** `results/stage3_complex_heldout_raar45`, on the CPU of the cloud workspace, 2026-11-15, 15:24 to 16:36. The 100 images were split between two processes that call the per-image function of `scripts/run_baselines.py` with the committed configuration; the official runner was then run with `--device cpu`, accepted the protocol lock and skipped the completed images. On the CPU, seeds 0 to 4 agreed with the stored GPU run within 0.1 dB on only 109 of 500 starts, and their best had 35 failures and 21 exact recoveries against 29 and 25. With 45 starts the CPU run gave 46 exact recoveries and 5 failures (mean 43.36 dB). Its report stays in the repository.

**Why the devices differ.** RAAR with shrinkwrap is sensitive to rounding. `scripts/raar_rounding_sensitivity.py` reran the first 10 starts of ranks 100, 120, 140, 160 and 180 on the CPU with the initial phase perturbed by at most 10⁻⁶ rad per pixel: 11 of 50 starts stayed within 0.1 dB of the unperturbed CPU run and 7 changed failure status (`results/raar_rounding_sensitivity/summary.json`). Single images can switch between exact and failed; the counts over 100 images move by about one.

## Results of the GPU run (n = 100, complex PSNR capped at 60 dB)

Best of the first K starts, selected by FE:

| K | mean | median | failures | exact | images with an exact start |
|--:|--:|--:|--:|--:|--:|
| 1 | 22.76 | 13.89 | 65 | 17 | 17 |
| 2 | 25.35 | 14.75 | 51 | 19 | 19 |
| 5 | 31.47 | 18.40 | 29 | 25 | 25 |
| 10 | 36.62 | 33.19 | 17 | 33 | 33 |
| 15 | 38.56 | 37.36 | 13 | 36 | 36 |
| 20 | 39.68 | 41.36 | 10 | 38 | 39 |
| 30 | 41.57 | 53.36 | 9 | 41 | 42 |
| 45 | 43.91 | 55.79 | 6 | 47 | 48 |

FE-softmax, for reference: mean 17.67, median 17.45, 21 failures, no exact recovery.

- **FE-softmax against RAAR with shrinkwrap, best of 45:** mean difference −26.24 dB (95% t-interval −29.86 to −22.63), median −37.88 dB. RAAR was better on 91 images, FE-softmax on 9 (Wilcoxon two-sided p = 1 × 10⁻¹⁶, not adjusted).
- **RAAR failures:** 6. FE-softmax was better on all 6 and failed itself on 1 of them.
- **FE within RAAR, best of 45:** exact recoveries had FE ≤ 3.5 × 10⁻⁴ and failures FE ≥ 0.027, without overlap, as for `best5`.
- **Exact start not selected:** on 48 images at least one start was exact. On one of them (MNIST 7983) the lowest-FE start reached 53.4 dB with FE 8.9 × 10⁻⁵, lower than the FE of the two exact starts (1.7 and 1.8 × 10⁻⁴).
- **Fallback rule** (RAAR best of 45 when its FE ≤ t, FE-softmax otherwise): 4 to 13 failures for t from 0.001 to 0.03, at a mean of 40.7 to 43.9 dB. At t = 0.03 it had 4 failures against 6 for RAAR alone, at the same mean.

## Reading

At equal wall time RAAR with shrinkwrap is far better than FE-softmax on these noiseless complex objects. It recovers almost half the images exactly and fails on 6. The sentence of the second study that FE-softmax was better on 28 of the 29 failures of RAAR with shrinkwrap holds only at one ninth of the compute. A fallback to the untrained method adds little at equal compute (6 to 4 failures), so a prospective test of the combined rule has little room on this data. Both runs lead to the same reading.

## Reproduce

```
python scripts/stage3_equal_compute_report.py --main results/stage3_complex_heldout \
    --best5 results/stage3_complex_heldout_baselines --raar45 results/stage3_complex_heldout_raar45_gpu \
    --out results/stage3_complex_heldout_raar45_gpu/report
```

The same command with `--raar45 results/stage3_complex_heldout_raar45` gives the report of the CPU run. Rerun the 45 starts: `python scripts/run_baselines.py --config configs/stage3_complex_heldout_raar45.yaml --device cuda` (about 3.6 to 6.3 hours of laptop GPU time). On another device individual starts can differ, as noted above.
