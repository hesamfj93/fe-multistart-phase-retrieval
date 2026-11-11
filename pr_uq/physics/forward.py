"""Forward model for oversampled Fourier phase retrieval.

Conventions
-----------
- Images are complex tensors of shape (B, 1, H, W).
- Measurements are real tensors of shape (B, 1, oH, oW) where oH = oversample*H.
- All FFTs use `norm='ortho'` so projection is an orthogonal operation.
- Output is *centered* (DC at oH/2, oW/2) via `fftshift` for ease of
  visualization and spectral weighting.

The two operators provided are:

  * `forward_field`: complex field at the detector plane, F(x).
  * `forward_intensity`: |F(x)|^2 (what a real detector measures).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _centered_zero_pad(x: torch.Tensor, oversample: int) -> torch.Tensor:
    """Center-pad an (B,C,H,W) tensor by `oversample` factor."""
    if oversample == 1:
        return x
    H, W = x.shape[-2:]
    oH, oW = oversample * H, oversample * W
    pad_h_left = (oH - H) // 2
    pad_h_right = oH - H - pad_h_left
    pad_w_left = (oW - W) // 2
    pad_w_right = oW - W - pad_w_left
    return F.pad(x, (pad_w_left, pad_w_right, pad_h_left, pad_h_right))


def forward_field(x: torch.Tensor, oversample: int = 2) -> torch.Tensor:
    """Compute the centered, ortho-normalized Fourier field.

    Parameters
    ----------
    x : (B, 1, H, W) complex tensor (real input is auto-cast to complex).
    oversample : zero-padding factor (default 2, the minimum for unique
        recovery of an HxW signal).

    Returns
    -------
    X : (B, 1, oH, oW) complex tensor, centered FFT.
    """
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    x_pad = _centered_zero_pad(x, oversample)
    # Use ifftshift so that the centered image is properly handled
    X = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x_pad), norm="ortho"))
    return X


def forward_intensity(x: torch.Tensor, oversample: int = 2) -> torch.Tensor:
    """Compute the measured intensity |F(x)|^2 with oversampling."""
    X = forward_field(x, oversample=oversample)
    return X.real.pow(2) + X.imag.pow(2)


def forward_magnitude(x: torch.Tensor, oversample: int = 2) -> torch.Tensor:
    """Compute the measured magnitude |F(x)|."""
    return forward_intensity(x, oversample=oversample).clamp_min(1e-12).sqrt()


def inverse_field(X: torch.Tensor, signal_hw: tuple[int, int]) -> torch.Tensor:
    """Inverse of `forward_field`: bring a detector-domain field back to image
    domain and crop to the signal extent.

    Parameters
    ----------
    X : (B, 1, oH, oW) complex tensor in centered convention.
    signal_hw : (H, W) of the original (non-padded) image.
    """
    H, W = signal_hw
    oH, oW = X.shape[-2:]
    x_pad = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(X), norm="ortho")
    )
    # Crop center
    h0 = (oH - H) // 2
    w0 = (oW - W) // 2
    return x_pad[..., h0 : h0 + H, w0 : w0 + W]
