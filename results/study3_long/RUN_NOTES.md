# Compute-matched control of Study 3: run notes

Plan: `STUDY3_LONG_PLAN.md`, frozen at tag `study3-long-plan-v1` (commit d265ad6). The plan commit
was pushed to GitHub main on 23 November 2026 at about 17:15 CET and the tag at about 17:31 CET,
before the first run, which started at 17:38 CET.

- Machine: the laptop of Study 3 (NVIDIA GeForce RTX 4060 Laptop GPU), Python 3.11.9 in the
  virtual environment of Study 3 (torch 2.5.1+cu121, numpy 2.1.2), clone checked out at the tag.
- A first attempt stopped at the CUDA check of the run script, before any run, because the shell
  used a system Python with a CPU-only torch; the run started after switching to the environment
  of Study 3. No run was repeated or discarded.
- Order: real_netadm_moderate, real_netadm_strong, real_ours_moderate, real_ours_strong,
  complex_ours_moderate, complex_ours_strong; 50 of 50 images each, no failure, no restart.
  Finished on 24 November 2026 at 02:22 CET.
- `report/` was produced in the cloud workspace by `scripts/study3_long_report.py` as committed
  at the tag, unchanged.

Wall time: one long start took 130 to 152 s per image for our solver and 36 to 38 s for Net-ADM,
about half of the five-start runs of Study 3 (256 to 276 s and 73 to 76 s), although the number
of training iterations is the same. The two were recorded in different sessions on the same
laptop, and the development timing of Study 3 (279 s and 74 s per image) agrees with the Study 3
runs. The plan matches compute by training iterations; the wall-time ratio is reported but not
explained by these records.
