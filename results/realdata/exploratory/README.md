# Measured pattern: a look at all classical starts

Not part of REALDATA_PLAN.md. Made after the planned runs and the planned report, from the saved starts; no new reconstruction.

The planned report gives a complex NRMSE of 1.151 between the lowest-FE RAAR and HIO reconstructions, so the two methods disagree. `lowest_fe_starts.png` shows the ten lowest-FE starts of each method.

- HIO: all 50 starts show two separate clusters of balls, at FE 0.140 to 0.157.
- RAAR: two kinds of solution. 26 starts show two separate clusters (FE 0.139 to 0.189, median 0.143). 24 starts show one continuous streak along the line joining the clusters (FE 0.113 to 0.227, median 0.126). The 16 lowest-FE RAAR starts are all streaks; the lowest-FE two-cluster start is 17th by FE.

A start counts as two clusters when at least 20% of the rows between the 0.5% and 99.5% quantiles of its vertical energy profile hold less than 1% of the largest row energy.

The published reconstruction of this pattern, which agrees with an electron micrograph, shows two clusters of 50 nm gold balls (Marchesini et al., Phys. Rev. B 68, 140101, 2003). The two-cluster solutions agree with it; the streak does not. On this pattern, then, the lowest FE over measured pixels selects a RAAR solution that disagrees with the published object, while within HIO every start has the published structure. On the three twins, all 150 starts of each method have two clusters by the same criterion, so the streak appears only on the measured pattern.

Possible reasons: the streak may differ from the two-cluster solutions mainly at frequencies under the beamstop, which FE does not see, or fit scattering that is not in the object model; the twin has neither.

## Free FE (added 19 November 2026)

`scripts/realdata_free_fe.py`, committed before its runs, withholds measured pixels and scores 50 new RAAR starts with shrinkwrap by their FE on the withheld pixels. Results in [`free_fe/report.md`](free_fe/report.md).

- Ring (the 847 measured pixels within 3 pixels of the beamstop, 16.7% of the measured intensity): the 9 starts with the lowest FE on the remaining pixels are streaks, the 7 with the lowest free FE have two clusters; AUROC of the free FE for a streak 0.88; Spearman between the two FEs -0.25.
- Random (5% of the other measured pixels): the free FE follows the FE on the remaining pixels (Spearman 0.99), and the 12 lowest by either are streaks; AUROC 0.26.

This is consistent with the first reason above: the streak fits the measured pixels better but predicts the intensities next to the beamstop worse. One pattern, analysed after its answer was known.

