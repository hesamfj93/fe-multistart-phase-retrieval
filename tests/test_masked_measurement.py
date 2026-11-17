"""Measurements with unmeasured pixels (beamstop, detector gaps) for the real-data demonstration.

A boolean mask, True where the intensity was measured, is accepted by the magnitude projection,
the trainer, the Fourier error, the anchor choice, HIO, RAAR and Net-ADM. The values of the
measurement at unmeasured pixels must have no effect, and an all-True mask must give the same
result as no mask wherever the arithmetic is the same.
"""
from __future__ import annotations

import pytest
import torch

from pr_uq.baselines.netadm import run_netadm
from pr_uq.baselines.oversampled import run_hio_full, run_raar_full
from pr_uq.models.ensemble import EnsembleMember, MemberSpec
from pr_uq.optim.trainer import TrainerConfig, train_member
from pr_uq.physics.forward import forward_field, forward_intensity, inverse_field
from pr_uq.physics.projection import project_magnitude
from pr_uq.uncertainty.ensemble_uq import member_fourier_errors, select_anchor

N = 16


def _problem():
    g = torch.Generator().manual_seed(3)
    x = torch.zeros(1, 1, N, N)
    x[..., 4:12, 5:11] = torch.rand(1, 1, 8, 6, generator=g)
    y = forward_intensity(x, 2)
    mask = torch.ones(1, 1, 2 * N, 2 * N, dtype=torch.bool)
    mask[..., N - 2:N + 3, N - 2:N + 3] = False          # a 5 x 5 beamstop around the zero frequency
    mask[..., :, :2] = False                              # a dead strip at the edge
    y_other = y.clone()
    y_other[~mask] = 123.0                                # arbitrary values where nothing was measured
    return x, y, y_other, mask


def test_projection_keeps_field_at_unmeasured_pixels():
    x, y, _, mask = _problem()
    z = torch.randn(1, 1, N, N, dtype=torch.complex64)
    pz = project_magnitude(z, y, mask=mask)
    full = project_magnitude(z, y)
    Z = forward_field(z, 2)
    Zm = torch.where(mask, Z * (y.sqrt() / Z.abs().clamp_min(1e-8)), Z)
    assert torch.allclose(pz, inverse_field(Zm, (N, N)), atol=1e-6)
    assert torch.allclose(project_magnitude(z, y, mask=torch.ones_like(mask)), full)


def test_fourier_error_uses_measured_pixels_only():
    x, y, y_other, mask = _problem()
    stack = torch.cat([x.to(torch.complex64) * s for s in (1.0, 0.8, 1.3)])
    fe = member_fourier_errors(stack, y, mask=mask)
    assert torch.equal(fe, member_fourier_errors(stack, y_other, mask=mask))
    ym = y.sqrt()
    for k in range(3):
        fp = forward_intensity(stack[k:k + 1], 2).clamp_min(1e-12).sqrt()
        ref = ((fp - ym)[mask].pow(2).sum().sqrt() / ym[mask].pow(2).sum().sqrt()).item()
        assert fe[k].item() == pytest.approx(ref, rel=1e-6)
    assert torch.allclose(member_fourier_errors(stack, y, mask=torch.ones_like(mask)),
                          member_fourier_errors(stack, y))
    assert select_anchor(stack, y_intensity=y_other, mask=mask) == 0


def test_trainer_ignores_unmeasured_values():
    _, y, y_other, mask = _problem()
    cfg = TrainerConfig(iterations=40, support_update_every=10)
    r1, _ = train_member(EnsembleMember(MemberSpec(out_h=N, out_w=N, seed=1)), y, cfg, mask=mask)
    r2, _ = train_member(EnsembleMember(MemberSpec(out_h=N, out_w=N, seed=1)), y_other, cfg, mask=mask)
    assert torch.equal(r1, r2)
    r3, _ = train_member(EnsembleMember(MemberSpec(out_h=N, out_w=N, seed=1)), y, cfg)
    assert not torch.equal(r1, r3)


@pytest.mark.parametrize("constraint", ["real_nonneg", "shrinkwrap", "support_only"])
@pytest.mark.parametrize("run", [run_hio_full, run_raar_full])
def test_classical_ignores_unmeasured_values(run, constraint):
    _, y, y_other, mask = _problem()
    kw = dict(iters=60, er_iters=10, seed=2, constraint=constraint)
    # the start uses the measured magnitudes on the whole grid, so the runner stores zeros at
    # unmeasured pixels (as the preprocessed real pattern does); after that only the mask matters
    y0, y1 = y.clone(), y_other.clone()
    y0[~mask] = 0.0
    y1[~mask] = 0.0
    a = run(y0, (N, N), mask=mask, **kw).recon
    b = run(y1, (N, N), mask=mask, **kw).recon
    assert torch.equal(a, b)
    assert torch.equal(run(y, (N, N), mask=torch.ones_like(mask), **kw).recon, run(y, (N, N), **kw).recon)


def test_netadm_ignores_unmeasured_values():
    _, y, y_other, mask = _problem()
    y0, y1 = y.clone(), y_other.clone()
    y0[~mask] = 0.0
    y1[~mask] = 0.0
    a = run_netadm(y0, (N, N), outer_iters=20, widths=(8, 8, 8), mask=mask).recon
    b = run_netadm(y1, (N, N), outer_iters=20, widths=(8, 8, 8), mask=mask).recon
    assert torch.equal(a, b)
    assert torch.equal(run_netadm(y, (N, N), outer_iters=20, widths=(8, 8, 8), mask=torch.ones_like(mask)).recon,
                       run_netadm(y, (N, N), outer_iters=20, widths=(8, 8, 8)).recon)
