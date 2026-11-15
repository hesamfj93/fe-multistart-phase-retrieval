# Post hoc: RAAR with shrinkwrap at equal wall time (complex objects)

**Not part of `STAGE3_COMPLEX_PLAN.md`.** The confirmatory comparison C8 (FE-softmax against RAAR with shrinkwrap, best of 5 starts) stays as reported. This analysis was added after a review noted that FE-softmax used about nine times the wall time of that baseline.

## Design

- Held-out ranks 100 to 199 (complex objects), the images of the second study.
- FE-softmax took 128.7 s per image and one RAAR-shrinkwrap start 2.86 s, both on the same laptop GPU, so RAAR with shrinkwrap gets 128.7 / 2.86 = 45 starts.
- `configs/stage3_complex_heldout_raar45.yaml` equals `configs/stage3_complex_heldout_baselines.yaml` except for algorithms, constraints, number of starts and output folder: same measurement, 5000 iterations with a 200-iteration error-reduction tail, β = 0.87, shrinkwrap settings, seeds 0 to 44 and metrics. The best start is the one with the lowest FE against the measurement, as in `best5`.
- The configuration and `scripts/stage3_equal_compute_report.py` were committed (`86c3589`) before the run.

## Execution

- Run on the CPU of the cloud workspace on 2026-11-15, 15:24 to 16:36 (local time). The 100 images were split between two processes that call the per-image function of `scripts/run_baselines.py` with the committed configuration. The official runner was then run with `--device cpu`; it accepted the protocol lock and skipped all 100 completed images.
- The stored `best5` run used the laptop GPU. RAAR does not give identical iterates on the two devices: seeds 0 to 4 agreed within 0.1 dB on 109 of 500 starts, and the best of seeds 0 to 4 had 35 failures and 21 exact recoveries on the CPU against 29 and 25 on the GPU. All numbers below come from the CPU run alone, so the comparison across numbers of starts is on one device.

## Results (n = 100, complex PSNR capped at 60 dB)

Best of the first K starts, selected by FE:

| K | mean | median | failures | exact | images with an exact start |
|--:|--:|--:|--:|--:|--:|
| 1 | 20.60 | 13.73 | 67 | 13 | 13 |
| 2 | 23.83 | 14.71 | 52 | 18 | 18 |
| 5 | 28.90 | 17.47 | 35 | 21 | 21 |
| 10 | 33.95 | 24.11 | 25 | 29 | 29 |
| 15 | 36.95 | 33.19 | 16 | 33 | 33 |
| 20 | 37.43 | 34.10 | 11 | 33 | 34 |
| 30 | 41.34 | 51.07 | 11 | 42 | 43 |
| 45 | 43.36 | 54.61 | 5 | 46 | 47 |

FE-softmax, for reference: mean 17.67, median 17.45, 21 failures, no exact recovery.

- **FE-softmax against RAAR with shrinkwrap, best of 45:** mean difference −25.68 dB (95% t-interval −29.28 to −22.09), median −37.27 dB. RAAR was better on 90 images, FE-softmax on 10 (Wilcoxon two-sided p = 3 × 10⁻¹⁶, not adjusted).
- **RAAR failures:** 5. FE-softmax was better on 4 of them but itself failed on 3.
- **FE within RAAR, best of 45:** exact recoveries had FE ≤ 3.5 × 10⁻⁴ and failures FE ≥ 0.029, without overlap, as for `best5`.
- **Fallback rule** (RAAR best of 45 when its FE ≤ t, FE-softmax otherwise): 4 to 14 failures for t from 0.001 to 0.03, at a mean of 39.7 to 43.3 dB. At t = 0.03 it had 4 failures against 5 for RAAR alone, at the same mean.

## Reading

At equal wall time RAAR with shrinkwrap is far better than FE-softmax on these noiseless complex objects. It recovers almost half the images exactly and fails on 5. The sentence of the second study that FE-softmax was better on 28 of the 29 failures of RAAR with shrinkwrap holds only at one ninth of the compute. A fallback to the untrained method adds little at equal compute (5 to 4 failures), so a prospective test of the combined rule has little room on this data.

## Reproduce

```
python scripts/stage3_equal_compute_report.py --main results/stage3_complex_heldout \
    --best5 results/stage3_complex_heldout_baselines --raar45 results/stage3_complex_heldout_raar45 \
    --out results/stage3_complex_heldout_raar45/report
```

Rerun the 45 starts: `python scripts/run_baselines.py --config configs/stage3_complex_heldout_raar45.yaml --device cpu` (about 2 hours on 2 CPU cores, or about 3.6 hours of laptop GPU time with `--device cuda`). On another device individual starts can differ, as noted above.
