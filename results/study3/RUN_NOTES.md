# Third study: how the runs were made

All eleven runs were made from a clone at the tag `study3-plan-v1` (commit fec8efa); every `environment.json` records that commit. The report in `report/` was produced by `scripts/study3_report.py` as committed at the tag, without changes.

- Untrained runs (seven): laptop GPU (NVIDIA GeForce RTX 4060 Laptop GPU), 17 November 2026 22:17 to 18 November 16:22, in the order of `STUDY3_PLAN.md`. All 50 images of every run finished on the first attempt. The first launch of the batch file failed before any computation because it cloned from GitHub, where the tag had not yet been pushed; the second launch fetched the tag from a local bundle into that clone and checked it out.
- Classical runs (four): cloud CPU, 17 November 2026. Two runs were stopped by the time limit of the cloud session and resumed with the same command (`complex_raar50_strong` after 49 images, `real_hio81_strong` after 47); resumed runs skip finished images.
- Runtime per image of the untrained runs matched the development timing used to fix the equal-wall-time K (ours 256 to 276 s against 253 to 279 s; Net-ADM 73 to 76 s against 74 s).

Not included here: per-image preview images and run logs.

No deviation from the plan in methods, settings, images or analysis. The tag, as stated in the plan, is pushed to GitHub together with these results.
