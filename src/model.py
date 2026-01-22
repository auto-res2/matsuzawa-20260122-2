import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------------------------------------------------------
# Backbone factory
# -----------------------------------------------------------------------------

def create_backbone(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone: {name}")

# -----------------------------------------------------------------------------
# AUASA components – Uncertainty meta-model
# -----------------------------------------------------------------------------

class UncertaintyMetaModel(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):  # x: (B, input_dim)
        return self.net(x).squeeze(-1)

# -----------------------------------------------------------------------------
# Losses & metrics helpers
# -----------------------------------------------------------------------------

def adaptive_uncertainty_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    aug_metric: torch.Tensor,
    uncertainty: torch.Tensor,
    k: float = 0.1,
    alpha: float = 0.5,
):
    ce = F.cross_entropy(logits, targets, reduction="none")
    w_aug = torch.exp(-k * aug_metric)
    w_unc = torch.exp(-alpha * uncertainty)
    w = w_aug * w_unc
    return (ce * w).mean()


def label_smoothing_loss(logits: torch.Tensor, targets: torch.Tensor, smooth_factor: float = 0.1):
    n_cls = logits.size(1)
    with torch.no_grad():
        true_dist = torch.empty_like(logits).fill_(smooth_factor / (n_cls - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smooth_factor)
    return torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))


def brier_score_loss(logits: torch.Tensor, targets: torch.Tensor):
    prob = F.softmax(logits, dim=1)
    one_hot = torch.zeros_like(prob).scatter_(1, targets.unsqueeze(1), 1)
    return ((prob - one_hot) ** 2).sum(1).mean()

# -----------------------------------------------------------------------------
# Augmentation metric
# -----------------------------------------------------------------------------

def compute_aug_metric(p: torch.Tensor) -> torch.Tensor:
    # p: (B, 3) – [rotation_norm, crop_norm, jitter_norm]
    rot, crop, jit = p[:, 0], p[:, 1], p[:, 2]
    return rot + crop * (2 - jit)

# -----------------------------------------------------------------------------
# ECE computation
# -----------------------------------------------------------------------------

def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """Expected Calibration Error on logits.

    logits: Tensor of shape (N, C)
    labels: Tensor of shape (N,)
    """
    prob = F.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    acc = pred.eq(labels).float()

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (conf[mask].mean() - acc[mask].mean()).abs() * mask.float().mean()
    return ece.item()
