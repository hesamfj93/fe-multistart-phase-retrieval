#!/usr/bin/env python
"""Figure 1 of the TMLR manuscript: fig_dev_example of make_paper_figures.py with the
terminology of the TMLR text ("start", "selection"), written to paper/figures_tmlr/.

    python paper/make_tmlr_dev_example.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "make_paper_figures.py").read_text(encoding="utf-8")
for a, b in [
    ('f"member {k}{tag}', 'f"start {k}{tag}'),
    ('("best_single_by_fe", "best-single-by-FE")', '("best_single_by_fe", "selection")'),
    ('"uncertainty $u$\\n(FE-softmax)"', '"map $u$\\n(FE-softmax)"'),
    ('"raw members"', '"raw starts"'),
]:
    assert a in src, a
    src = src.replace(a, b)
ns = {"__name__": "make_paper_figures_tmlr", "__file__": str(HERE / "make_paper_figures.py")}
exec(compile(src, "make_paper_figures.py", "exec"), ns)
ns["OUT"] = HERE / "figures_tmlr"
ns["OUT"].mkdir(exist_ok=True)
ns["fig_dev_example"]()
