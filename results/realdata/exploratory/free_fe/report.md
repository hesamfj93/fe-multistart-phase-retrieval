# Free FE on the measured pattern (post hoc, exploratory)

Not part of REALDATA_PLAN.md; see the docstring of scripts/realdata_free_fe.py. 50 RAAR starts with shrinkwrap per withheld set.

| withheld set | free pixels | share of measured intensity | two-cluster starts | FE train, two clusters | FE train, streak | FE free, two clusters | FE free, streak | lowest FE train | lowest FE free | AUROC of FE free for streak |
|---|--:|--:|--:|---|---|---|---|---|---|--:|
| ring | 847 | 0.167 | 18 of 50 | 0.133 to 0.204 (median 0.149) | 0.111 to 0.242 (median 0.203) | 0.213 to 0.553 (median 0.293) | 0.275 to 0.785 (median 0.425) | streak | two clusters | 0.88 |
| random | 6242 | 0.079 | 22 of 50 | 0.138 to 0.197 (median 0.144) | 0.103 to 0.245 (median 0.123) | 0.155 to 0.213 (median 0.164) | 0.121 to 0.254 (median 0.141) | streak | streak | 0.26 |

AUROC: probability that a streak start has a higher free FE than a two-cluster start (1 = free FE always ranks the two-cluster solutions first).
