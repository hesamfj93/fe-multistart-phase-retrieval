# Net-ADM in the MNIST setting of Ma et al. (post hoc check)

Run on the CPU of the cloud workspace with `scripts/netadm_ma_check.py` as committed in b8e3377,
before its runs. `report.md` and `summary.json` are its output, unchanged.

The PSNR of HIO in `report.md` is not capped; an exact recovery gives values above 100 dB. With
the cap of 60 dB used in the studies, the mean over the 60 HIO runs is 45.96 dB (median of the
uncapped values 120.6 dB); 44 runs exceed 30 dB and 8 fall below 15 dB. Net-ADM: mean 30.74 dB,
median 34.46 dB, 40 runs above 30 dB, 6 below 15 dB (no run exceeds 60 dB).

"PSNR, no alignment" is about 10 dB for both methods because the reconstructions are shifted or
flipped; Ma et al. remove the flip ambiguity by setting two border pixels, which this check does
not do, so that column is not comparable with their figure.

The preprint does not name its six images and gives PSNR only in its Figure 3, so this check
cannot be compared with it number by number.
