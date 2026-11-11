"""Alignment of reconstructions to ground truth under PR ambiguities.

Fourier phase retrieval admits three intrinsic ambiguities:

  1. Global phase:   x → x · e^{iα}        (only relevant for complex x)
  2. Translation:    x → x(r − r₀)         (cyclic shift in oversampled FFT)
  3. Conjugate flip: x → x(−r)*            (twin image)

Honest PSNR/SSIM must align over all three before scoring. This module
implements brute-force alignment that:

  * enumerates the 4 conjugate/flip options,
  * finds the best cyclic shift via FFT cross-correlation,
  * solves for the optimal global phase by inner product,
  * reports raw and aligned scores.

The protocol is described explicitly in the paper §4 so reviewers can
verify it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# -----------------------------------------------------------------------------
# Cyclic-shift alignment via FFT cross-correlation
# -----------------------------------------------------------------------------

def fft_xcorr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Real-valued cross-correlation of two (1, 1, H, W) tensors via FFT.

    For complex tensors we cross-correlate the magnitudes (which removes
    phase issues), and the maximum of |xcorr| is invariant to global phase.
    """
    a_r = a.abs() if torch.is_complex(a) else a
    b_r = b.abs() if torch.is_complex(b) else b
    A = torch.fft.fft2(a_r)
    B = torch.fft.fft2(b_r)
    return torch.fft.ifft2(A * B.conj()).real


def best_cyclic_shift(recon: torch.Tensor, gt: torch.Tensor) -> tuple[int, int]:
    """Find the integer (dy, dx) shift that maximizes cross-correlation."""
    xc = fft_xcorr(recon, gt)
    idx = torch.argmax(xc.reshape(-1))
    H, W = xc.shape[-2:]
    dy = int(idx // W)
    dx = int(idx % W)
    # Map [0, H) into [-H/2, H/2)
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return dy, dx


def cyclic_shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return torch.roll(x, shifts=(dy, dx), dims=(-2, -1))


# -----------------------------------------------------------------------------
# Global phase alignment (complex case)
# -----------------------------------------------------------------------------

def best_global_phase(recon: torch.Tensor, gt: torch.Tensor) -> complex:
    """Optimal e^{iα} that maximizes Re(<recon · e^{iα}, gt>).

    Closed form: α* = angle(<recon, gt>); the unit complex multiplier is
    then exp(-iα*) so that recon · e^{iα*} aligns with gt.
    """
    if not (torch.is_complex(recon) and torch.is_complex(gt)):
        return 1.0 + 0.0j
    inner = (recon * gt.conj()).sum()
    if inner.abs().item() < 1e-12:
        return 1.0 + 0.0j
    # Phase factor that brings inner to positive real
    phase = torch.angle(inner)
    return torch.exp(-1j * phase).item()


# -----------------------------------------------------------------------------
# Conjugate-flip enumeration
# -----------------------------------------------------------------------------

def apply_variant(x: torch.Tensor, variant: int) -> torch.Tensor:
    """Apply one of 4 conjugate/flip variants.

    0: identity
    1: horizontal flip
    2: vertical flip
    3: both flips (=180° rotation)
    For complex tensors, each variant is combined with its conjugate twin —
    in PR the twin image is `conj(flip(x))`.
    """
    if variant == 0:
        out = x
    elif variant == 1:
        out = torch.flip(x, dims=(-1,))
    elif variant == 2:
        out = torch.flip(x, dims=(-2,))
    elif variant == 3:
        out = torch.flip(x, dims=(-2, -1))
    else:
        raise ValueError(f"Variant must be in 0..3, got {variant}")
    if variant != 0 and torch.is_complex(out):
        out = out.conj()
    return out


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Outcome of `align_to_gt` with provenance for reporting."""

    aligned_recon: torch.Tensor
    variant: int                  # which conjugate/flip applied
    shift: tuple[int, int]
    phase: complex = 1.0 + 0.0j   # only meaningful when complex
    mse: float = 0.0              # MSE after alignment, used as the tiebreaker


def align_to_gt(
    recon: torch.Tensor,
    gt: torch.Tensor,
    try_variants: tuple[int, ...] = (0, 1, 2, 3),
    return_real: bool | None = None,
) -> AlignmentResult:
    """Best of `try_variants` × cyclic shift × global phase alignment.

    The "best" is the one minimizing MSE between aligned_recon and gt. If
    `return_real` is None it is auto-set from whether the ground truth is
    complex.

    Parameters
    ----------
    recon : (1, 1, H, W) reconstruction (real or complex).
    gt    : (1, 1, H, W) ground truth (real or complex).

    Returns
    -------
    AlignmentResult
    """
    if return_real is None:
        return_real = not torch.is_complex(gt)

    best = AlignmentResult(
        aligned_recon=recon.clone(),
        variant=0,
        shift=(0, 0),
        phase=1.0 + 0.0j,
        mse=float("inf"),
    )

    # If recon is complex but gt is real, compare against the real part
    gt_for_compare = gt
    if torch.is_complex(recon) and not torch.is_complex(gt):
        gt_for_compare = gt.to(torch.complex64)

    for v in try_variants:
        rv = apply_variant(recon, v)
        # The peak of xcorr(rv, gt) is at the shift τ that was applied to gt
        # to produce rv. To align rv → gt we shift by −τ.
        dy, dx = best_cyclic_shift(rv, gt_for_compare)
        rv_shifted = cyclic_shift(rv, -dy, -dx)
        phase = best_global_phase(rv_shifted, gt_for_compare)
        rv_phase = rv_shifted * phase
        if return_real:
            r_compare = rv_phase.real if torch.is_complex(rv_phase) else rv_phase
            g_compare = gt.real if torch.is_complex(gt) else gt
        else:
            r_compare = rv_phase
            g_compare = gt
        mse = (r_compare - g_compare).abs().pow(2).mean().item()
        if mse < best.mse:
            best = AlignmentResult(
                aligned_recon=rv_phase,
                variant=v,
                shift=(dy, dx),
                phase=phase,
                mse=mse,
            )

    return best


# -----------------------------------------------------------------------------
# Ensemble alignment (align all members to first member, then to GT)
# -----------------------------------------------------------------------------

def map_to_gt_frame(img: torch.Tensor, res: "AlignmentResult") -> torch.Tensor:
    """Move a real per-pixel map (uncertainty, residual) into the ground-truth frame.

    Applies the flip variant and cyclic shift that `align_to_gt` found for the
    reconstruction the map belongs to. The global phase is irrelevant for a real map,
    and so is the conjugation that some variants include. Without this step the map
    stays in the frame of the reconstruction while the error map is in the frame of
    the ground truth, and pixel-wise comparisons between them are misregistered.
    """
    out = apply_variant(img.to(torch.complex64), res.variant)
    out = cyclic_shift(out, -res.shift[0], -res.shift[1])
    return out.real.to(img.dtype) if not torch.is_complex(img) else out


def align_ensemble_to_anchor(
    stack: torch.Tensor,
    anchor_index: int = 0,
) -> torch.Tensor:
    """Align each member of a (K, 1, H, W) stack to member `anchor_index`.

    This is mandatory before computing the ensemble mean — different members
    can converge to different conjugate/shift/phase variants, all of which
    are equally valid reconstructions individually but average to zero if
    not aligned first.
    """
    anchor = stack[anchor_index : anchor_index + 1]
    aligned = []
    for k in range(stack.shape[0]):
        if k == anchor_index:
            aligned.append(stack[k : k + 1])
            continue
        res = align_to_gt(stack[k : k + 1], anchor, return_real=False)
        aligned.append(res.aligned_recon)
    return torch.cat(aligned, dim=0)
