"""Save and load ensemble checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from ..models.ensemble import Ensemble, EnsembleMember, MemberSpec


def save_member(member: EnsembleMember, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "spec": member.spec.__dict__,
            "state_dict": member.state_dict(),
        },
        path,
    )


def load_member(path: str | Path, device: torch.device | str = "cpu") -> EnsembleMember:
    blob = torch.load(path, map_location=device)
    spec = MemberSpec(**blob["spec"])
    member = EnsembleMember(spec, device=device)
    member.load_state_dict(blob["state_dict"])
    member.to(device)
    return member


def save_ensemble(ens: Ensemble, dirpath: str | Path) -> None:
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(ens.members):
        save_member(m, dirpath / f"member_{k:02d}.pt")
    # Also save the (small) reconstructions for fast re-evaluation
    if ens.reconstructions:
        torch.save(
            torch.cat(ens.reconstructions, dim=0), dirpath / "reconstructions.pt"
        )


def load_ensemble(dirpath: str | Path, device: torch.device | str = "cpu") -> Ensemble:
    dirpath = Path(dirpath)
    files = sorted(dirpath.glob("member_*.pt"))
    ens = Ensemble()
    for f in files:
        ens.members.append(load_member(f, device=device))
    rec_path = dirpath / "reconstructions.pt"
    if rec_path.exists():
        stack = torch.load(rec_path, map_location=device)
        ens.reconstructions = [stack[i : i + 1] for i in range(stack.shape[0])]
    return ens
