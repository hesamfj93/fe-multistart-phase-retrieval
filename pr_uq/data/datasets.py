"""Deterministic dataset loaders + Fourier-measurement generation.

Each `make_*` function returns a list of `(image, name)` tuples ready for
the trainer. Images are real-valued single-channel tensors of shape
(1, 1, H, W) in [0, 1].

Datasets supported (in order of recommended priority):
  * MNIST              — sanity & paper-essential
  * Fashion-MNIST      — paper-essential
  * CIFAR-10 (luma)    — paper-essential
  * CelebA             — optional, larger images

To avoid forcing big downloads in CI, each loader checks if the underlying
torchvision dataset is already on disk and only downloads on explicit
request.

Image normalization: all images mapped to [0, 1].
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

# torchvision is optional at import time so the unit tests don't require it
try:
    from torchvision import datasets, transforms
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False


@dataclass
class ImageSample:
    """One image to reconstruct, with its identifier and optional metadata."""

    image: torch.Tensor                # (1, 1, H, W) in [0, 1]
    name: str
    label: int | None = None
    dataset: str | None = None
    extra: dict | None = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _check_tv():
    if not _TV_AVAILABLE:
        raise RuntimeError(
            "torchvision is required for built-in dataset loaders. "
            "Install with `pip install torchvision`."
        )


def _resize_to(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """If x has 3 channels, convert with ITU-R BT.601 weights."""
    if x.shape[1] == 1:
        return x
    if x.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def _wrap(image: torch.Tensor, name: str, label: int | None, dataset: str) -> ImageSample:
    image = image.clamp(0, 1).float()
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return ImageSample(image=image, name=name, label=label, dataset=dataset)


# -----------------------------------------------------------------------------
# Built-in dataset loaders
# -----------------------------------------------------------------------------

def _read_idx(path: str | os.PathLike) -> "np.ndarray":
    """Read a (possibly gzipped) MNIST IDX file into a numpy array.

    The IDX format (http://yann.lecun.com/exdb/mnist/) is a small binary
    header followed by raw bytes:
      magic (4B):  0x00 0x00 dtype_code num_dims
      then num_dims big-endian uint32 sizes,
      then the data.

    `dtype_code` is one of: 0x08 (uint8), 0x09 (int8), 0x0B (int16),
    0x0C (int32), 0x0D (float32), 0x0E (float64).
    """
    import gzip
    import numpy as np
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = f.read(4)
        if len(magic) != 4 or magic[0] != 0 or magic[1] != 0:
            raise ValueError(f"{path}: not a valid IDX header (magic={magic!r})")
        dtype_code = magic[2]
        ndim = magic[3]
        dtypes = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2",
                  0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}
        if dtype_code not in dtypes:
            raise ValueError(f"{path}: unknown IDX dtype code 0x{dtype_code:02x}")
        sizes = []
        for _ in range(ndim):
            sizes.append(int.from_bytes(f.read(4), "big"))
        data = np.frombuffer(f.read(), dtype=dtypes[dtype_code])
    return data.reshape(sizes).astype(np.uint8 if dtype_code == 0x08 else None, copy=False)


def _find_idx_pair(
    root: str | os.PathLike, split: str
) -> tuple[str, str]:
    """Locate (images, labels) IDX files under `root` for the given split.

    Accepts the following layouts (in order of preference):

    1. `root/MNIST/raw/{train,t10k}-{images,labels}-idx*-ubyte[.gz]`
       (torchvision default; what `torchvision.datasets.MNIST` creates)
    2. `root/raw/{train,t10k}-{images,labels}-idx*-ubyte[.gz]`
    3. `root/{train,t10k}-{images,labels}-idx*-ubyte[.gz]`
       (flat layout — also accepted)

    `split` is 'train' or 'test'. 'test' maps to the `t10k-*` files.
    """
    prefix = "train" if split == "train" else "t10k"
    candidates = [
        Path(root) / "MNIST" / "raw",
        Path(root) / "raw",
        Path(root),
    ]
    for d in candidates:
        if not d.exists():
            continue
        imgs = None
        lbls = None
        for ext in ["", ".gz"]:
            i = d / f"{prefix}-images-idx3-ubyte{ext}"
            l = d / f"{prefix}-labels-idx1-ubyte{ext}"
            if i.exists() and l.exists():
                imgs, lbls = str(i), str(l)
                break
        if imgs is not None:
            return imgs, lbls
    raise FileNotFoundError(
        f"Could not locate MNIST IDX files for split={split!r} under {root}.\n"
        f"Tried (in order): {[str(d) for d in candidates]}\n"
        f"Expected files: {prefix}-images-idx3-ubyte[.gz], "
        f"{prefix}-labels-idx1-ubyte[.gz]"
    )


def make_mnist_offline(
    n_images: int = 20,
    size: int = 32,
    root: str = "./data",
    split: str = "test",
    seed: int = 0,
    indices: list[int] | None = None,
    **_ignored,
) -> list[ImageSample]:
    """Network-free MNIST loader.

    Reads raw IDX files from a local directory (see `_find_idx_pair` for the
    accepted layouts). No `torchvision` dependency. Suitable for sandboxed
    environments and reproducible CI.

    Parameters
    ----------
    n_images : how many images to sample.
    size     : target square size (28→`size` via centered zero padding).
    root     : base directory. Defaults to `./data`.
    split    : 'train' or 'test'.
    seed     : permutation seed (deterministic subset), used only when
               `indices` is None.
    indices  : explicit MNIST indices, in order (protocol runs).
    """
    import numpy as np
    imgs_path, lbls_path = _find_idx_pair(root, split)
    images = _read_idx(imgs_path)        # (N, 28, 28) uint8
    labels = _read_idx(lbls_path)        # (N,) uint8

    if indices is None:
        # Legacy selection: deterministic permutation (NOT protocol-compliant;
        # see pr_uq/data/splits.py for the dev / held-out protocol).
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(images), generator=g)[:n_images].tolist()
    else:
        indices = [int(i) for i in indices]
        bad = [i for i in indices if not 0 <= i < len(images)]
        if bad:
            raise ValueError(f"MNIST {split} indices out of range: {bad[:5]}")

    out: list[ImageSample] = []
    pad = (size - 28) // 2
    if pad < 0:
        raise ValueError(f"size={size} must be ≥ 28 for MNIST")
    for i in indices:
        arr = images[i].astype("float32") / 255.0          # (28, 28) in [0, 1]
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
        if pad > 0:
            t = F.pad(t, (pad, pad, pad, pad), mode="constant", value=0.0)
        out.append(_wrap(t, name=f"mnist_{i:06d}", label=int(labels[i]), dataset="mnist"))
    return out



def make_mnist(
    n_images: int = 20,
    size: int = 32,
    root: str = "./data",
    split: str = "test",
    download: bool = True,
    seed: int = 0,
    offline_only: bool = False,
) -> list[ImageSample]:
    """Deterministic subset of MNIST padded to `size`.

    Strategy:
      1. If raw IDX files exist locally under `root`, load them directly
         (offline path — no network, no torchvision needed).
      2. Otherwise, if `download=True` and `offline_only=False`, fall back
         to `torchvision.datasets.MNIST(..., download=True)`.
      3. Otherwise, raise FileNotFoundError with manual-setup instructions.

    Both paths use the same deterministic permutation seeded by `seed`.
    """
    # Path 1: native offline reader
    try:
        return make_mnist_offline(
            n_images=n_images, size=size, root=root, split=split, seed=seed
        )
    except FileNotFoundError as e_offline:
        if offline_only or not download:
            raise FileNotFoundError(
                str(e_offline)
                + "\n\nTo prepare MNIST manually, see README.md §'Offline MNIST setup'."
            ) from e_offline
    # Path 2: torchvision download fallback
    _check_tv()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad((size - 28) // 2),
    ])
    train = (split == "train")
    ds = datasets.MNIST(root=root, train=train, download=download, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    out: list[ImageSample] = []
    for i in indices:
        img, label = ds[i]
        out.append(_wrap(img.unsqueeze(0), name=f"mnist_{i:06d}", label=int(label), dataset="mnist"))
    return out


def make_fashion_mnist(
    n_images: int = 500,
    size: int = 32,
    root: str = "./data",
    split: str = "test",
    download: bool = True,
    seed: int = 0,
) -> list[ImageSample]:
    _check_tv()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad((size - 28) // 2),
    ])
    train = (split == "train")
    ds = datasets.FashionMNIST(root=root, train=train, download=download, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    out = []
    for i in indices:
        img, label = ds[i]
        out.append(_wrap(img.unsqueeze(0), name=f"fashion_{i:06d}", label=int(label), dataset="fashion_mnist"))
    return out


def make_cifar10(
    n_images: int = 500,
    size: int = 32,
    root: str = "./data",
    split: str = "test",
    download: bool = True,
    seed: int = 0,
    grayscale: bool = True,
) -> list[ImageSample]:
    _check_tv()
    tfm = transforms.ToTensor()
    train = (split == "train")
    ds = datasets.CIFAR10(root=root, train=train, download=download, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    out = []
    for i in indices:
        img, label = ds[i]
        img = img.unsqueeze(0)
        if grayscale:
            img = _to_grayscale(img)
        if img.shape[-1] != size:
            img = _resize_to(img, size)
        out.append(_wrap(img, name=f"cifar_{i:06d}", label=int(label), dataset="cifar10"))
    return out


def make_synthetic(
    n_images: int = 20,
    size: int = 32,
    seed: int = 0,
    sparsity: float = 0.7,
    **_ignored,
) -> list[ImageSample]:
    """Network-free synthetic dataset of random sparse blobs.

    Useful for CI / smoke testing when MNIST cannot be downloaded. Each
    image is built from K random Gaussian blobs of varying scale, then
    thresholded to give the kind of compact-support, real, non-negative
    signal that the PR pipeline expects.
    """
    g = torch.Generator().manual_seed(seed)
    out: list[ImageSample] = []
    for i in range(n_images):
        img = torch.zeros(1, 1, size, size)
        n_blobs = int(torch.randint(2, 6, (1,), generator=g).item())
        for _ in range(n_blobs):
            cy = int(torch.randint(size // 4, 3 * size // 4, (1,), generator=g).item())
            cx = int(torch.randint(size // 4, 3 * size // 4, (1,), generator=g).item())
            sigma = float(torch.empty(1).uniform_(1.5, 4.0, generator=g).item())
            amp = float(torch.empty(1).uniform_(0.3, 1.0, generator=g).item())
            ys = torch.arange(size).float() - cy
            xs = torch.arange(size).float() - cx
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            blob = amp * torch.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
            img[0, 0] += blob
        # Sparsify
        img = img.clamp(0, 1)
        if sparsity > 0:
            thr = torch.quantile(img.flatten(), sparsity)
            img = torch.where(img > thr, img, torch.zeros_like(img))
            img = img / img.max().clamp_min(1e-6)
        out.append(_wrap(img, name=f"synth_{i:06d}", label=None, dataset="synthetic"))
    return out



def make_celeba(
    n_images: int = 200,
    size: int = 64,
    root: str = "./data",
    split: str = "test",
    download: bool = False,
    seed: int = 0,
    grayscale: bool = True,
) -> list[ImageSample]:
    """CelebA loader — defaults to `download=False` (large dataset)."""
    _check_tv()
    tfm = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    ds = datasets.CelebA(root=root, split=split, download=download, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_images].tolist()
    out = []
    for i in indices:
        img, _attrs = ds[i]
        img = img.unsqueeze(0)
        if grayscale:
            img = _to_grayscale(img)
        out.append(_wrap(img, name=f"celeba_{i:06d}", label=None, dataset="celeba"))
    return out


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_dataset(spec: dict) -> list[ImageSample]:
    """Spec example:
        {'name': 'mnist', 'n_images': 20, 'size': 32, 'root': './data',
         'split': 'test', 'download': True, 'seed': 0}
    """
    name = spec["name"].lower()
    role = spec.get("protocol_split")
    if role:
        # Protocol run: explicit, file-backed MNIST test indices (dev / heldout / smoke).
        from .splits import load_protocol_indices
        if name != "mnist" or spec.get("split", "test") != "test":
            raise ValueError("protocol_split is currently defined only for the MNIST test set")
        idx = load_protocol_indices(spec.get("splits_dir", "splits"), role, int(spec["n_images"]))
        return make_mnist_offline(indices=idx, size=spec.get("size", 32),
                                  root=spec.get("root", "./data"), split="test")
    kwargs = {k: v for k, v in spec.items()
              if k not in {"name", "protocol_split", "splits_dir"}}
    if name == "mnist":
        return make_mnist(**kwargs)
    if name in {"fashion_mnist", "fashion-mnist", "fashion"}:
        return make_fashion_mnist(**kwargs)
    if name == "cifar10":
        return make_cifar10(**kwargs)
    if name == "celeba":
        return make_celeba(**kwargs)
    if name == "synthetic":
        return make_synthetic(**kwargs)
    raise ValueError(f"Unknown dataset {name!r}")


# -----------------------------------------------------------------------------
# Measurement generation
# -----------------------------------------------------------------------------

def generate_measurement(
    image: torch.Tensor,
    oversample: int = 2,
    noise_model: str = "none",
    noise_level: float = 0.0,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Return (y_intensity, metadata) for a single image.

    `noise_model`: 'none' | 'gaussian' | 'poisson'.
    `noise_level`: σ for Gaussian (relative to peak); photon budget for Poisson.
    """
    from ..physics.forward import forward_intensity
    from ..physics.noise import apply_noise

    image = image.to(device)
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
    intensity_clean = forward_intensity(image, oversample=oversample)
    if noise_model == "none" or noise_level == 0:
        y = intensity_clean
    else:
        # noise sampling on CPU then move (torch.poisson on CUDA does not
        # accept a generator argument in all versions)
        y_cpu = apply_noise(intensity_clean.cpu(), noise_model, noise_level, generator=gen)
        y = y_cpu.to(device)
    meta = {
        "oversample": oversample,
        "noise_model": noise_model,
        "noise_level": float(noise_level),
        "peak_intensity": float(intensity_clean.max().item()),
    }
    return y, meta
