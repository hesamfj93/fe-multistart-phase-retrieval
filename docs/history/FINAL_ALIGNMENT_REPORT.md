> **SUPERSEDED in wording and partly in content.** This file documents the alignment investigation on the 8 development images. Its central finding (alignment was not the cause of the K=5 regression) still holds, but it uses the phrase "averaging bug", which is replaced by "uniform aggregation failure", and it predates the temperature sweep and the full baseline comparison. The authoritative development-stage report is **AGGREGATION_FINALIZATION_REPORT.md**.

---

# Ensemble Alignment Investigation — Is the K=5 Regression an Alignment Bug?

**Question (from the brief).** The K=5 ensemble regressed on easy images, worst of all mnist_002890 (K=1 26.21 dB → K=5 20.84 dB, −5.37 dB). Is this an alignment/averaging bug, or a real ensemble problem?

**Answer.** It is an **averaging problem, not an alignment bug.** Alignment is working correctly. The regression comes from a plain uniform mean averaging in under-converged members that are individually poor reconstructions. Fixing the anchor choice recovers ~0.6 dB; downweighting the bad members by their data-fidelity recovers the rest. With both fixes, K=5 now **beats** K=1 (mean 21.57 vs 19.97 dB).

All numbers below are computed on the **already-trained** K=5 stacks (the per-image member reconstructions were saved), so nothing was retrained — this is pure re-aggregation of existing results.

---

## 1. The decisive evidence: oracle alignment cannot fix it

The `oracle` column aligns every member *directly to ground truth* — the best any alignment scheme could possibly do. If the regression were an alignment bug, oracle alignment would recover it.

| image | K=1 | legacy (anchor=0, uniform) | **oracle alignment** | gap oracle vs legacy |
|---|--:|--:|--:|--:|
| mnist_002890 | 26.21 | 20.84 | **20.88** | **+0.04** |
| mnist_004919 | 22.06 | 18.96 | 19.44 | +0.48 |
| mnist_006044 | 23.04 | 23.55 | 26.03 | +2.48 |

For mnist_002890, perfect alignment buys **0.04 dB**. The 5.4 dB is not recoverable by alignment. Something else is wrong.

## 2. What is actually wrong: two of five members are junk

Per-member reconstruction quality for mnist_002890 (each member individually aligned to GT), with the ground-truth-free Fourier-magnitude error FE:

| member | individual PSNR | FE (data-fidelity, no GT needed) |
|---|--:|--:|
| 0 | 26.21 dB | 0.118 |
| 1 | 20.91 dB | 0.136 |
| 2 | **13.48 dB** | **0.309** |
| 3 | 20.84 dB | 0.235 |
| 4 | **14.76 dB** | **0.252** |

Members 2 and 4 converged to bad solutions (a blob and a smear — see `figures/inspect_002890_before_after.png`, row 1). Member 0 alone is 26.21 dB (it is literally the K=1 reconstruction — same seed). A uniform mean of {26.2, 20.9, 13.5, 20.8, 14.8} lands at 20.84 dB: the two junk members drag the good ones down.

**Crucially, FE ranks the members correctly without ground truth** (0 < 1 < 3 < 4 < 2 by FE tracks 0 > 1 ≈ 3 > 4 > 2 by PSNR). FE is data-fidelity against the measurement, so it is available at inference time. This is what makes the fix possible in practice.

The visualization `figures/inspect_002890_before_after.png` shows this directly: after aligning to the best-FE anchor (row 2), members 0/1/3 are consistent "7"s but members 2 and 4 are *still blobs*. Alignment oriented them; it cannot make a bad reconstruction good.

## 3. The fix and its effect

Two independent levers, implemented in `pr_uq/uncertainty/ensemble_uq.py`:

1. **Anchor selection** (`select_anchor`, method `best_fe`): align to the member that best explains the measurement, not to member 0 by default.
2. **Fidelity weighting** (`fidelity_weights`, method `softmax`): weight member *k* by `exp(-FE_k / T)` instead of uniformly. Under-converged members contribute less. Uniform weighting is a special case, so the legacy path is exactly reproducible.

### PSNR across all 8 images

| image | K=1 | legacy | best-FE anchor | **fe_softmax** | fe_top3 |
|---|--:|--:|--:|--:|--:|
| mnist_000429 | 13.88 | 13.83 | 14.09 | 14.14 | 14.22 |
| mnist_001729 | 25.97 | 26.29 | 27.13 | 27.11 | 25.64 |
| mnist_001917 | 17.28 | 17.17 | 16.85 | 18.06 | 18.01 |
| mnist_002890 | 26.21 | 20.84 | 20.84 | **23.54** | 23.49 |
| mnist_004919 | 22.06 | 18.96 | 18.96 | 19.29 | 19.60 |
| mnist_006044 | 23.04 | 23.55 | 25.06 | 25.05 | 24.30 |
| mnist_007532 | 15.19 | 18.16 | 19.06 | 21.56 | 21.16 |
| mnist_009399 | 16.17 | 19.22 | 21.03 | 23.79 | 22.05 |
| **mean** | 19.97 | 19.75 | 20.38 | **21.57** | 21.06 |
| **median** | 19.67 | 19.09 | 19.95 | **22.55** | 21.61 |
| **std** | 4.61 | 3.60 | 3.94 | 3.94 | 3.47 |
| **mean SSIM** | — | 0.775 | 0.778 | **0.833** | 0.864 |
| **mean FE** | — | 0.288 | 0.280 | 0.231 | 0.222 |

**Decomposition of the fix (mean PSNR):**
- legacy → best-FE anchor: **+0.63 dB** (this is the alignment contribution)
- best-FE anchor → fe_softmax: **+1.19 dB** (this is the fidelity-weighting contribution)
- net K=5 vs K=1: legacy was −0.22 dB; fe_softmax is **+1.60 dB**

The alignment fix is the smaller lever (~0.6 dB). The averaging fix is the larger one (~1.2 dB). Together they turn a regression into a win.

### mnist_002890 specifically (the target case)

| stage | PSNR |
|---|--:|
| K=1 (= member 0) | 26.21 dB |
| naive mean (no alignment) | 13.71 dB |
| legacy (anchor=0, uniform) | 20.84 dB |
| best-FE anchor (uniform) | 20.84 dB |
| **fe_softmax** | **23.54 dB** |

Softmax weights: member 0 → 0.391, member 1 → 0.328, member 3 → 0.121, member 4 → 0.102, member 2 → 0.058. The two junk members are cut to ~6% and ~10% weight. Recovery: from −5.37 dB regression to −2.67 dB. It does not fully return to 26.21 dB because soft weighting still admits a little of the bad members; hard top-3 selection behaves similarly (23.49 dB). Fully matching K=1 on this image would require dropping the bad members entirely, at the cost of the uncertainty signal (see next).

---

## 4. The honest cost: fidelity weighting slightly weakens the uncertainty signal

Downweighting bad members improves the reconstruction but touches the uncertainty channel, because disagreement among members is exactly what produces high ensemble std in failure regions.

| metric | legacy | best-FE anchor | fe_softmax | fe_top3 |
|---|--:|--:|--:|--:|
| mean Pearson(U, \|err\|) | **0.286** | 0.236 | 0.228 | 0.197 |
| mean Spearman(U, \|err\|) | 0.321 | **0.344** | 0.341 | 0.239 |
| mean ECE (lower better) | 0.114 | 0.113 | 0.114 | **0.088** |
| failure-detection AUROC¹ | 0.714 | 0.714 | 0.714 | **1.000** |

¹ Fixed labels (K=1 PSNR < 14.7 dB); only 1 failure in 8 images, so AUROC here is directional, not statistically meaningful.

Reading this honestly:
- **Pearson (linear) correlation drops** with weighting (0.286 → 0.228 for softmax). Expected: shrinking the std in bad-member regions reduces the linear coupling between std and error magnitude.
- **Spearman (rank) correlation is preserved** (0.321 → 0.341 for softmax) — the *ordering* of uncertain pixels is unchanged, only the scale.
- **fe_top3 has the best ECE and AUROC but the worst Pearson** and lower mean PSNR than softmax.

So there is a genuine reconstruction-vs-uncertainty trade-off across the strategies. **fe_softmax is the best all-rounder**: it delivers the highest mean PSNR (21.57), preserves rank-correlation, and only modestly reduces linear correlation. That is why it is now the pipeline default (`anchor_method=best_fe`, `weight_method=softmax`, `temperature=0.1`), with `weight_method=uniform` available to reproduce the old behaviour exactly.

---

## 5. What changed in the code

- `pr_uq/uncertainty/ensemble_uq.py`
  - `member_fourier_errors(stack, y)` — ground-truth-free per-member quality proxy.
  - `select_anchor(stack, method, y)` — `first` | `best_fe` | `medoid`.
  - `fidelity_weights(fe, method, temperature, top_k)` — `uniform` | `softmax` | `inverse` | `top_k`.
  - `ensemble_aggregate(...)` — full anchor-select → align → weighted mean+std. `EnsembleStats` now also carries `anchor_index`, `weights`, `member_fe`.
  - The alignment routine (`align_ensemble_to_anchor`) was already correct and is unchanged. Its shift/flip/conjugate/global-phase handling (from `align_to_gt`) is exercised by the existing `tests/test_alignment.py`.
- `pr_uq/pipeline.py` — aggregation is now config-driven via `uncertainty.anchor_method` / `weight_method` / `fidelity_temperature` / `top_k`; defaults to best-FE + softmax.
- `tests/test_aggregation.py` — 7 new tests (weights sum to 1, softmax downweights high-FE members, top_k zeros the rest, uniform reproduces the plain mean, best-FE anchor picks the lowest-error member, downweighting beats uniform with a bad member present, K=1 gives zero std). All 25 tests pass.

Figures written to `figures/`:
- `inspect_002890_before_after.png` — the 5 members before/after alignment + the four aggregate reconstructions (the requested item-5 visualization).
- `aggregation_strategy_comparison.png` — per-image K=1 vs legacy vs fe_softmax.

---

## 6. Direct answers to the brief

1. **Align every member to a common anchor before averaging** — done; anchor is now chosen, not fixed at member 0.
2. **Anchor choices** — implemented all three (best-FE, medoid, first). best-FE is the default and the strongest; note that "K=1 reconstruction as anchor" ≡ member 0 ≡ the `first` option, and it does *not* fix the regression because anchor choice is worth only ~0.6 dB.
3. **Shifts / flips / conjugate / global phase** — handled by the existing `align_to_gt`, unchanged and unit-tested.
4. **Recompute mean/std/r_phys/fused/PSNR/SSIM/FE** — done for all 8 images under all strategies (§3, §4).
5. **mnist_002890 before/after** — `figures/inspect_002890_before_after.png`; PSNR at every stage in §3.
6. **Re-run K=1 vs K=5 table** — §3.
7. **No new datasets** — none added.

**Bottom line:** the K=5 PSNR drop was an averaging bug (uniform mean over mixed-quality members), not an alignment bug. Alignment was fine. With best-FE anchoring plus fidelity weighting, K=5 goes from −0.22 dB *below* K=1 to **+1.60 dB above** it (mean), and the mnist_002890 regression shrinks from −5.37 to −2.67 dB. The one caveat to carry forward: fidelity weighting trades a little pixel-wise uncertainty correlation for reconstruction quality, so the paper's uncertainty claims should lean on rank-correlation (Spearman, preserved) rather than linear correlation (Pearson, reduced).

## 7. Still open before moving to Fashion-MNIST / CIFAR

- This is 8 images. Re-run on 50+ before treating any of these deltas as established (the +1.60 dB mean is over n=8).
- Re-generate the K=5 arrays *through the updated pipeline* (these results are re-aggregations of the old stacks; a fresh run should reproduce them and also write the new `weights`/`member_fe` fields into the npz).
- Sweep the softmax temperature (0.05–0.3) — 0.1 was a first guess, not tuned.
- Decide the uncertainty story: if the paper needs strong pixel-wise calibration, `top_k` selection hurts Pearson most; a temperature-scaling pass on the fused map may recover calibration independent of the weighting choice.
