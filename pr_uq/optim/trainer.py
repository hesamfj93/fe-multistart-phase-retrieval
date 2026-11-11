"""Per-ensemble-member training loop.

This is the core optimization routine. One call to `train_member` produces
one reconstruction by fitting an untrained double-decoder to a single set of
intensity measurements.

The loop is intentionally explicit and modular: every component (z-update,
spectral annealing, β-mix physics projection, adaptive support, TV/phase
regularization) can be turned on or off via the trainer config, supporting
the ablation studies the paper requires.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from ..losses.data_loss import build_data_loss
from ..losses.regularization import regularization_total
from ..losses.spectral_weight import SpectralAnnealer
from ..models.ensemble import EnsembleMember, MemberSpec
from ..physics.forward import forward_intensity
from ..physics.projection import beta_mix, project_magnitude
from .schedules import build_schedule
from .support import initial_support, update_support


@dataclass
class TrainerConfig:
    """All knobs for `train_member`.

    Defaults are reasonable for MNIST 32×32 with K=1 and ~6000 iterations.
    """

    # Optimization
    iterations: int = 6000
    lr_theta: float = 1e-3
    lr_z: float = 1e-3
    optimize_z: bool = True
    grad_clip: float | None = 1.0

    # Loss
    data_loss: str = "magnitude_l2"           # 'magnitude_l2' | 'log_magnitude' | 'poisson_nll'
    oversample: int = 2

    # Regularization
    lambda_tv_A: float = 1e-3
    lambda_tv_phi: float = 1e-4
    lambda_phi_l2: float = 0.0
    lambda_support: float = 1e-2

    # Spectral annealing
    use_spectral_annealing: bool = True
    sigma_start_frac: float = 0.5             # fraction of measurement side
    sigma_end_frac: float = 0.05
    annealing_schedule: str = "log"
    annealing_fraction: float = 0.6

    # β-mixing physics projection
    use_physics_projection: bool = True
    beta_schedule: dict = field(
        default_factory=lambda: {"type": "linear", "start": 0.7, "end": 0.1}
    )
    project_in_loss: bool = True              # if True, loss uses x_use
    project_detached: bool = True             # detach projection from grads

    # Adaptive support
    use_adaptive_support: bool = True
    support_initial_fill: str = "full"
    support_update_every: int = 300
    support_smooth_sigma: float = 1.0
    support_threshold_frac: float = 0.08
    support_gamma: float = 10.0

    # Logging
    log_every: int = 100
    keep_loss_curve: bool = True
    keep_best_recon: bool = False             # OFF by default — see note below
    # NOTE on keep_best_recon: for PR with spectral annealing, the
    # *unweighted* Fourier-magnitude loss can plateau near zero very early
    # in training (the network finds a low-Fourier-error solution that is
    # still a wrong twin/shift variant) while aligned PSNR keeps rising
    # for hundreds of iterations more. Tracking "best by loss" then picks
    # a bad early iterate. The final iterate is almost always a better
    # reconstruction. Leave this off unless you have a sound loss-PSNR
    # correlation in your specific setup.

    # Reporting (filled by trainer)
    iterations_done: int = 0


@dataclass
class TrainerLog:
    """Per-iteration logs aggregated by the trainer (filled in `train_member`)."""

    steps: list[int] = field(default_factory=list)
    loss_data: list[float] = field(default_factory=list)
    loss_reg: list[float] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    beta: list[float] = field(default_factory=list)
    # Optional GT-evaluated curves (populated only if `gt` is passed to train_member)
    psnr_curve: list[float] = field(default_factory=list)
    fourier_err_curve: list[float] = field(default_factory=list)
    runtime_s: float = 0.0


def _make_optimizer(
    member: EnsembleMember, config: TrainerConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """Adam on (θ_A, θ_Φ), optionally a second Adam on (z_A, z_Φ)."""
    opt_theta = torch.optim.Adam(member.net_parameters, lr=config.lr_theta)
    opt_z = None
    if config.optimize_z:
        opt_z = torch.optim.Adam(member.latent_parameters, lr=config.lr_z)
    return opt_theta, opt_z


def train_member(
    member: EnsembleMember,
    y_intensity: torch.Tensor,
    config: TrainerConfig,
    device: torch.device | str = "cpu",
    progress: Callable[[int, dict], None] | None = None,
    gt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, TrainerLog]:
    """Train one ensemble member to fit `y_intensity`.

    Parameters
    ----------
    member : freshly constructed `EnsembleMember`.
    y_intensity : (1, 1, oH, oW) measured intensity, on `device`.
    config : `TrainerConfig`.
    device : where the optimization runs.
    progress : optional callback receiving (step, log_dict) periodically.
    gt : optional (1, 1, H, W) ground truth used ONLY for logging aligned-PSNR
        and Fourier-magnitude-error curves during training. Has no effect on
        the optimization itself — purely diagnostic.

    Returns
    -------
    recon : (1, 1, H, W) complex final reconstruction.
    log : `TrainerLog` with loss curves, runtime, and optional PSNR/FE curves.
    """
    member = member.to(device)
    y_intensity = y_intensity.to(device)
    H, W = member.spec.out_h, member.spec.out_w
    oH, oW = y_intensity.shape[-2:]
    log = TrainerLog()

    # Optimizers
    opt_theta, opt_z = _make_optimizer(member, config)
    data_loss_fn = build_data_loss(config.data_loss)

    # Schedules
    beta_schedule = build_schedule(config.beta_schedule)
    annealer = None
    if config.use_spectral_annealing:
        sigma_start = config.sigma_start_frac * max(oH, oW)
        sigma_end = config.sigma_end_frac * max(oH, oW)
        annealer = SpectralAnnealer(
            h=oH,
            w=oW,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            total_steps=config.iterations,
            schedule=config.annealing_schedule,
            anneal_fraction=config.annealing_fraction,
            device=device,
        )

    # Support
    support_mask = (
        initial_support(H, W, fill=config.support_initial_fill, device=device)
        if config.use_adaptive_support or config.lambda_support > 0
        else None
    )

    # Tracking best
    best_loss = float("inf")
    best_recon: torch.Tensor | None = None

    t0 = time.perf_counter()
    for step in range(config.iterations):
        opt_theta.zero_grad(set_to_none=True)
        if opt_z is not None:
            opt_z.zero_grad(set_to_none=True)

        # Forward
        A = member.amplitude()
        phi = member.phase()
        x_net = (A * torch.cos(phi)).to(torch.complex64) + 1j * (A * torch.sin(phi)).to(
            torch.complex64
        )

        # Physics projection / β-mixing
        if config.use_physics_projection:
            beta = beta_schedule(step, config.iterations)
            if config.project_detached:
                with torch.no_grad():
                    x_proj = project_magnitude(x_net, y_intensity, oversample=config.oversample)
                x_use = beta * x_proj + (1.0 - beta) * x_net
            else:
                x_use = beta_mix(x_net, y_intensity, beta, oversample=config.oversample)
        else:
            beta = 0.0
            x_use = x_net

        x_loss = x_use if config.project_in_loss else x_net

        # Weights
        w = annealer.weight_at(step) if annealer is not None else None

        # Losses
        L_data = data_loss_fn(x_loss, y_intensity, weight=w, oversample=config.oversample)
        L_reg = regularization_total(
            amplitude=A,
            phase=phi,
            support_mask=support_mask,
            lambda_tv_A=config.lambda_tv_A,
            lambda_tv_phi=config.lambda_tv_phi,
            lambda_phi_l2=config.lambda_phi_l2,
            lambda_support=config.lambda_support,
        )
        L = L_data + L_reg
        L.backward()

        # Gradient clip
        if config.grad_clip is not None and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(member.parameters()), max_norm=config.grad_clip
            )

        opt_theta.step()
        if opt_z is not None:
            opt_z.step()

        # Adaptive support refresh
        if (
            config.use_adaptive_support
            and step > 0
            and step % config.support_update_every == 0
        ):
            with torch.no_grad():
                support_mask = update_support(
                    amplitude=member.amplitude(),
                    uncertainty=None,                # K=1 has no UQ; ensemble adds it later
                    smooth_sigma=config.support_smooth_sigma,
                    threshold_frac=config.support_threshold_frac,
                    gamma=config.support_gamma,
                    use_uncertainty=False,
                )

        # Logging
        if step % config.log_every == 0 or step == config.iterations - 1:
            log.steps.append(step)
            log.loss_data.append(float(L_data.detach().item()))
            log.loss_reg.append(float(L_reg.detach().item()))
            log.loss_total.append(float(L.detach().item()))
            log.sigma.append(annealer._cached_sigma if annealer is not None else 0.0)
            log.beta.append(float(beta))
            if gt is not None:
                with torch.no_grad():
                    from ..metrics.alignment import align_to_gt
                    from ..metrics.reconstruction import psnr as _psnr, fourier_magnitude_error as _fe
                    gt_c = gt.to(torch.complex64) if not torch.is_complex(gt) else gt
                    aligned = align_to_gt(x_net.detach(), gt_c).aligned_recon
                    log.psnr_curve.append(_psnr(aligned, gt_c))
                    log.fourier_err_curve.append(
                        _fe(x_net.detach(), gt_c, oversample=config.oversample)
                    )
            if progress is not None:
                progress(step, {"loss": float(L.detach().item()), "beta": float(beta)})

        # Track best reconstruction. We store x_net (the network output) and
        # pick the iteration with the lowest *unweighted* data loss on x_net.
        # Using the unweighted form is critical: with spectral annealing,
        # the weighted loss is small early (heavy low-pass) even when
        # reconstruction is poor, so cross-iteration comparison must be done
        # with a fixed (no) weight. Without this, the "best" iterate often
        # comes from very early in training.
        if config.keep_best_recon:
            with torch.no_grad():
                eval_loss = float(
                    data_loss_fn(x_net, y_intensity, weight=None, oversample=config.oversample).item()
                )
            if eval_loss < best_loss:
                best_loss = eval_loss
                with torch.no_grad():
                    best_recon = x_net.detach().clone()

    log.runtime_s = time.perf_counter() - t0
    config.iterations_done = config.iterations

    # Final reconstruction
    with torch.no_grad():
        A = member.amplitude()
        phi = member.phase()
        x_final = (A * torch.cos(phi)).to(torch.complex64) + 1j * (A * torch.sin(phi)).to(
            torch.complex64
        )
    if best_recon is None:
        best_recon = x_final
    return best_recon.detach(), log


def train_ensemble(
    base_spec: MemberSpec,
    y_intensity: torch.Tensor,
    config: TrainerConfig,
    K: int,
    base_seed: int = 0,
    device: torch.device | str = "cpu",
    progress: Callable[[int, int, dict], None] | None = None,
):
    """Train K independent ensemble members and return an `Ensemble` object.

    Members differ only in their `seed`, which controls *both* the latent
    init and the network init. This is what gives us diverse members and
    therefore meaningful ensemble variance for uncertainty estimation.
    """
    from ..models.ensemble import Ensemble

    ens = Ensemble()
    for k in range(K):
        spec = MemberSpec(
            out_h=base_spec.out_h,
            out_w=base_spec.out_w,
            z_channels=base_spec.z_channels,
            z_hw=base_spec.z_hw,
            base_ch=base_spec.base_ch,
            seed=base_seed + k,
        )
        member = EnsembleMember(spec, device=device)

        def member_progress(step: int, info: dict, k=k) -> None:
            if progress is not None:
                progress(k, step, info)

        recon, log = train_member(
            member, y_intensity, config, device=device, progress=member_progress
        )
        ens.members.append(member)
        ens.reconstructions.append(recon)
        ens.losses.append(log.loss_total[-1] if log.loss_total else float("nan"))
    return ens
