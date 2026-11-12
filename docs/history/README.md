# Development notes

These three reports document the development stage on the 8 development images (MNIST test indices 429, 1729, 1917, 2890, 4919, 6044, 7532, 9399), before the analysis plan of the held-out evaluation was frozen. Each file states at the top what it superseded and which conclusions were later revised.

- `DIAGNOSTIC_REPORT_MNIST.md`: first K=1 against K=3 diagnostic.
- `FINAL_DIAGNOSTIC_REPORT.md`: K=1 against K=5 at 3000 iterations.
- `FINAL_ALIGNMENT_REPORT.md`: alignment investigation of the K=5 regression.

The figures they refer to (`figures/*.png`) are not part of this release. The development results they cite are in `results/mnist_k1_3000iter/`, `results/mnist_k5_3000iter/` and `results/*.csv`; the authoritative development report is `docs/AGGREGATION_FINALIZATION_REPORT.md`. None of these numbers enters a held-out comparison.
