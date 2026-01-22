import random
from pathlib import Path
from typing import Tuple, List, Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Global cache root – MUST be exactly `.cache/`
CACHE_ROOT = Path(".cache")

# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Augmentation with parameter capture
# -----------------------------------------------------------------------------

class AugmentAndRecord:
    """Training augmentation pipeline that also returns normalised augmentation parameters.

    Returned parameter tensor p has three elements in [0,1] representing:
    1. rotation magnitude / max_rotation
    2. crop reduction (i.e. 1 - crop_scale)
    3. colour jitter magnitude / max_jitter
    """

    def __init__(
        self,
        max_rotation: int = 15,
        min_crop: float = 0.8,
        max_jitter: float = 0.4,
        mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
        std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
    ):
        self.max_rotation = max_rotation
        self.min_crop = min_crop
        self.max_jitter = max_jitter
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img):
        rot_deg = random.uniform(-self.max_rotation, self.max_rotation)
        crop_scale = random.uniform(self.min_crop, 1.0)
        jitter = random.uniform(0.0, self.max_jitter)

        transform = transforms.Compose(
            [
                transforms.RandomRotation((rot_deg, rot_deg)),
                transforms.RandomResizedCrop(32, scale=(crop_scale, 1.0)),
                transforms.ColorJitter(
                    brightness=jitter, contrast=jitter, saturation=jitter, hue=0
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        img_t = transform(img)
        params = torch.tensor(
            [
                abs(rot_deg) / self.max_rotation,
                1.0 - crop_scale,
                jitter / (self.max_jitter if self.max_jitter > 0 else 1.0),
            ],
            dtype=torch.float32,
        )
        return img_t, params


class EvalTransform:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.t(img), torch.zeros(3, dtype=torch.float32)


class CIFAR10WithParams(datasets.CIFAR10):
    """Wrap torchvision CIFAR-10 to expose augmentation param tensor."""

    def __init__(self, root: Any, train: bool, transform=None, download: bool = False):
        super().__init__(root=root, train=train, transform=None, download=download)
        self._tfm = transform

    def __getitem__(self, idx):
        img, lbl = super().__getitem__(idx)
        img_t, p = self._tfm(img)
        return img_t, lbl, p

# -----------------------------------------------------------------------------
# Dataloader factory
# -----------------------------------------------------------------------------

def get_dataloaders(cfg):
    seed_everything(int(cfg.training.seed))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if cfg.dataset.name.lower() != "cifar10":
        raise ValueError("Only CIFAR-10 supported in this implementation.")

    mean = tuple(cfg.dataset.preprocessing.normalization.mean)
    std = tuple(cfg.dataset.preprocessing.normalization.std)

    tf_train = AugmentAndRecord(mean=mean, std=std)
    tf_eval = EvalTransform(mean, std)

    full_train = CIFAR10WithParams(CACHE_ROOT, train=True, transform=tf_train, download=True)

    val_count = int(cfg.dataset.split.val)
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_set = Subset(full_train, train_idx)

    val_base = CIFAR10WithParams(CACHE_ROOT, train=True, transform=tf_eval, download=False)
    val_set = Subset(val_base, val_idx)

    test_set = CIFAR10WithParams(CACHE_ROOT, train=False, transform=tf_eval, download=True)

    dl_train = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dl_val = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    dl_test = DataLoader(
        test_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return dl_train, dl_val, dl_test, 10
