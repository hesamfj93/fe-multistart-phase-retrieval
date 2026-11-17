"""Net-ADM: ADMM-based Fourier phase retrieval with an untrained generative prior.

Re-implementation of Algorithm 1 of L. Ma, H. Wang, N. Leng and Z. Yuan, "ADMM based Fourier
phase retrieval with untrained generative prior", J. Comput. Appl. Math. 444:115786 (2024),
arXiv:2210.12646. No code was released with the paper; this follows the algorithm listing and
the stated settings:

    inputs  b = sqrt(y), u~0 = F^-1(b), lambda0 = 0, rho = 1, eps = 1e-3, K = 1000 outer iterations
    x^{k+1} = P^T (u~^k - lambda^k / rho)                                               (17a)
    u^{k+1} = F^-1( sqrt(b^2 + eps) / sqrt(|F(P x^{k+1} + lambda^k/rho)|^2 + eps)
                    * F(P x^{k+1} + lambda^k/rho) )                                     (17b)
    lambda^{k+1} = lambda^k + rho (P x^{k+1} - u^{k+1})
    w^{k+1} = argmin_w || P^T u^{k+1} - G(w; z) ||   (5 Adam steps, learning rate 0.005,
                                                       decayed once every 500 outer steps)
    P^T u~^{k+1} = G(w^{k+1}; z)
    output x^K

P zero-pads the H x W object into the oversampled grid (the placement of this code base) and
P^T crops it. G is the deep-decoder-like network of the paper: a fixed latent Z0 ~ U[0, 0.1],
layers Z_{j+1} = Up(cn(relu(W_j Z_j))) with 1x1 convolutions W_j, channel normalization cn and
bilinear upsampling by 2, and output sig(W_J Z_J). For MNIST the paper uses widths {25, 15, 10}
(2 layers), so a 32 x 32 object has an 8 x 8 latent.

Choices the paper leaves open, fixed here before any held-out use:
- The decay factor of the learning rate is not given; 0.5 is used.
- The FFT normalization is not given; the magnitudes are scaled to the unnormalized DFT
  (factor sqrt(oH * oW) over the orthonormal convention of this code base), which keeps eps = 1e-3
  small relative to the data, as for unnormalized MNIST data in the paper.
- The inner loss is the mean squared error over the H x W pixels.
- The method is for real signals; x is the real part of the cropped field.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..physics.forward import _centered_zero_pad


@dataclass
class NetADMResult:
    recon: torch.Tensor          # (1, 1, H, W) complex64 with zero imaginary part
    runtime_s: float
    seed: int
    generator_output: torch.Tensor


class DeepDecoderG(nn.Module):
    """Net-ADM generator: 1x1 conv, ReLU, channel norm, bilinear x2 per layer; sigmoid output."""

    def __init__(self, widths=(25, 15, 10), out_hw=(32, 32)):
        super().__init__()
        n_up = len(widths) - 1
        h0, w0 = out_hw[0] // 2 ** n_up, out_hw[1] // 2 ** n_up
        if h0 * 2 ** n_up != out_hw[0] or w0 * 2 ** n_up != out_hw[1]:
            raise ValueError(f"output {out_hw} not divisible by 2**{n_up}")
        self.z = nn.Parameter(torch.rand(1, widths[0], h0, w0) * 0.1, requires_grad=False)
        layers = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Conv2d(c_in, c_out, 1, bias=False), nn.ReLU(),
                       nn.BatchNorm2d(c_out, affine=True),
                       nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)]
        layers += [nn.Conv2d(widths[-1], 1, 1, bias=False), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self) -> torch.Tensor:
        return self.net(self.z)


def _fft(z: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(z)))


def _ifft(Z: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Z)))


def _crop(z: torch.Tensor, hw) -> torch.Tensor:
    H, W = hw
    oH, oW = z.shape[-2:]
    h0, w0 = (oH - H) // 2, (oW - W) // 2
    return z[..., h0:h0 + H, w0:w0 + W]


def run_netadm(y_intensity: torch.Tensor, image_hw, outer_iters: int = 1000, inner_steps: int = 5,
               lr: float = 0.005, lr_decay_every: int = 500, lr_decay: float = 0.5, rho: float = 1.0,
               eps: float = 1e-3, widths=(25, 15, 10), seed: int = 0, oversample: int = 2,
               device="cpu", mask=None) -> NetADMResult:
    """Net-ADM on an oversampled Fourier intensity y (orthonormal convention of this code base).

    With a boolean `mask` (True where measured), the u-update keeps F(P x + lambda/rho) at
    unmeasured pixels, the standard treatment of a beamstop in projection methods."""
    t0 = time.perf_counter()
    H, W = image_hw
    y = y_intensity.to(device).float().clamp_min(0)
    oH, oW = y.shape[-2:]
    if mask is not None:
        mask = mask.to(device=device, dtype=torch.bool)
    scale = float((oH * oW) ** 0.5)
    b = y.sqrt() * scale                                  # unnormalized-DFT magnitudes
    torch.manual_seed(int(seed))
    G = DeepDecoderG(widths, (H, W)).to(device)
    opt = torch.optim.Adam([p for p in G.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_every * inner_steps, gamma=lr_decay)
    b2e = (b ** 2 + eps).sqrt()
    u_t = _ifft(b.to(torch.complex64))                    # u~0 = F^-1(b)
    lam = torch.zeros_like(u_t)
    x = None
    for _ in range(outer_iters):
        x = _crop(u_t - lam / rho, (H, W)).real
        Px = _centered_zero_pad(x.to(torch.complex64), oversample)
        V = _fft(Px + lam / rho)
        Vp = b2e / (V.abs() ** 2 + eps).sqrt() * V
        if mask is not None:
            Vp = torch.where(mask, Vp, V)
        u = _ifft(Vp)
        lam = lam + rho * (Px - u)
        target = _crop(u, (H, W)).real.detach()
        for _ in range(inner_steps):
            opt.zero_grad(set_to_none=True)
            loss = (G() - target).pow(2).mean()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            u_t = _centered_zero_pad(G().to(torch.complex64), oversample)
    x = _crop(u_t - lam / rho, (H, W)).real
    if x.is_cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        g = G().detach()
    return NetADMResult(x.to(torch.complex64), time.perf_counter() - t0, int(seed), g.to(torch.complex64))
