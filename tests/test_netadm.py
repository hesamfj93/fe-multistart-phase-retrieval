"""Net-ADM re-implementation: shapes, determinism, data fit on a small example."""
from __future__ import annotations

import torch

from pr_uq.baselines.netadm import DeepDecoderG, run_netadm
from pr_uq.metrics.reconstruction import fourier_magnitude_error
from pr_uq.physics.forward import forward_intensity


def _blob(n=32):
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    x = ((yy - 14) ** 2 / 30 + (xx - 17) ** 2 / 15 < 4).float()
    return x.view(1, 1, n, n)


def test_generator_shape_and_widths():
    torch.manual_seed(0)
    g = DeepDecoderG((25, 15, 10), (32, 32))
    out = g()
    assert out.shape == (1, 1, 32, 32) and float(out.min()) >= 0 and float(out.max()) <= 1
    assert tuple(g.z.shape) == (1, 25, 8, 8) and not g.z.requires_grad


def test_same_seed_same_result_and_real_output():
    y = forward_intensity(_blob().to(torch.complex64), oversample=2)
    a = run_netadm(y, (32, 32), outer_iters=30, seed=3)
    b = run_netadm(y, (32, 32), outer_iters=30, seed=3)
    c = run_netadm(y, (32, 32), outer_iters=30, seed=4)
    assert torch.equal(a.recon, b.recon) and not torch.equal(a.recon, c.recon)
    assert a.recon.shape == (1, 1, 32, 32) and float(a.recon.imag.abs().max()) == 0.0


def test_fits_the_measurement():
    x = _blob().to(torch.complex64)
    y = forward_intensity(x, oversample=2)
    r = run_netadm(y, (32, 32), outer_iters=300, seed=0)
    assert fourier_magnitude_error(r.recon, x, oversample=2) < 0.1
