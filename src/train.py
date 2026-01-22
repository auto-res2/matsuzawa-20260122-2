import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import wandb
import optuna
from sklearn.metrics import confusion_matrix

from src.preprocess import get_dataloaders, seed_everything
from src.model import (
    create_backbone,
    UncertaintyMetaModel,
    adaptive_uncertainty_loss,
    label_smoothing_loss,
    brier_score_loss,
    compute_aug_metric,
    compute_ece,
)

# -----------------------------------------------------------------------------
# Optimiser & Scheduler helpers
# -----------------------------------------------------------------------------

def _get_optimizer(cfg: DictConfig, params):
    name = str(cfg.training.optimizer).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def _get_scheduler(cfg: DictConfig, optimizer, total_steps: int):
    name = str(cfg.training.scheduler).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(total_steps * 0.6), int(total_steps * 0.85)],
            gamma=0.1,
        )
    return None

# -----------------------------------------------------------------------------
# WandB initialisation
# -----------------------------------------------------------------------------

def _init_wandb(cfg: DictConfig):
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        return None
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run_id,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"WandB URL: {run.get_url()}")
    return run

# -----------------------------------------------------------------------------
# Validation / Test helper
# -----------------------------------------------------------------------------

def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int] = None,
    return_cm: bool = False,
) -> Tuple[float, float, float, Optional[np.ndarray]]:
    """Return (loss, accuracy, ece, confusion_matrix|None).

    Critical fix: ECE computation now uses raw logits collected across the
    evaluation set. Confusion matrix is created with the FULL set of class
    labels to guarantee a square matrix regardless of prediction coverage.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for idx, (images, labels, _) in enumerate(dataloader):
            if (max_batches is not None) and (idx >= max_batches):
                break

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            # Accumulate statistics
            total_loss += loss.item() * images.size(0)
            total_correct += preds.eq(labels).sum().item()
            total_samples += images.size(0)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, None

    all_logits_t = torch.cat(all_logits)  # (N, C)
    all_labels_t = torch.cat(all_labels)  # (N,)
    all_preds_t = all_logits_t.argmax(dim=1)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Correct ECE computation on logits!
    ece = compute_ece(all_logits_t, all_labels_t)

    cm = None
    if return_cm:
        cm = confusion_matrix(
            all_labels_t.numpy(),
            all_preds_t.numpy(),
            labels=list(range(num_classes)),
        )
    return avg_loss, avg_acc, ece, cm

# -----------------------------------------------------------------------------
# Core training loop for a single (fixed) configuration
# -----------------------------------------------------------------------------

def _train_single(cfg: DictConfig, device: torch.device, trial_mode: bool):
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(cfg)

    # Limit batches sharply in trial mode
    train_max_batches = 2 if trial_mode else None
    eval_max_batches = 1 if trial_mode else None

    backbone = create_backbone(cfg.model.name, num_classes).to(device)
    use_auasa = "adaptive" in cfg.method.lower() or "proposed" in cfg.run_id.lower()
    if use_auasa:
        uncertainty_model = UncertaintyMetaModel(input_dim=4, hidden_dim=16).to(device)
        parameters = list(backbone.parameters()) + list(uncertainty_model.parameters())
    else:
        uncertainty_model = None
        parameters = list(backbone.parameters())

    optimizer = _get_optimizer(cfg, parameters)
    total_steps = cfg.training.epochs * (
        train_max_batches if train_max_batches is not None else len(train_loader)
    )
    scheduler = _get_scheduler(cfg, optimizer, total_steps)

    # Post-init assertion – ensure output dimensions are valid
    assert (
        backbone(torch.randn(2, 3, 32, 32, device=device)).shape[1] == num_classes
    ), "Model output dimension mismatch with dataset classes"

    run = _init_wandb(cfg)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(cfg.training.epochs):
        backbone.train()
        if uncertainty_model:
            uncertainty_model.train()

        total_loss_e, total_correct_e, total_samples_e = 0.0, 0, 0

        for b_idx, (imgs, lbls, aug_p) in enumerate(train_loader):
            if (train_max_batches is not None) and (b_idx >= train_max_batches):
                break

            if epoch == 0 and b_idx == 0:
                # Batch-start assertion
                assert imgs.size(0) == lbls.size(0), "Image/label size mismatch in first batch"

            imgs = imgs.to(device)
            lbls = lbls.to(device)
            aug_p = aug_p.to(device)

            optimizer.zero_grad()
            logits = backbone(imgs)

            if use_auasa:
                aug_metric = compute_aug_metric(aug_p)
                aug_feat = torch.cat([aug_p, aug_metric.unsqueeze(1)], dim=1)
                uncertainty = uncertainty_model(aug_feat)
                k = float(getattr(cfg.training.additional_params, "k", 0.1))
                alpha = float(getattr(cfg.training.additional_params, "alpha", 0.5))
                loss_main = adaptive_uncertainty_loss(
                    logits, lbls, aug_metric, uncertainty, k=k, alpha=alpha
                )
                aux_w = float(
                    getattr(cfg.training.additional_params, "auxiliary_calibration_loss_weight", 0.2)
                )
                loss = loss_main + aux_w * brier_score_loss(logits, lbls)
            else:
                smooth = float(getattr(cfg.training.additional_params, "softening_constant", 0.1))
                loss = label_smoothing_loss(logits, lbls, smooth_factor=smooth)

            loss.backward()

            grads = [p.grad for p in parameters if p.requires_grad]
            # Pre-optimizer assertions
            assert all(g is not None for g in grads), "Some gradients are None before optimizer.step()"
            assert any(g.abs().sum().item() > 0 for g in grads), "All gradients zero before optimizer.step()"

            if cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, cfg.training.gradient_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                batch_acc = preds.eq(lbls).float().mean().item()

            total_loss_e += loss.item() * imgs.size(0)
            total_correct_e += preds.eq(lbls).sum().item()
            total_samples_e += imgs.size(0)

            global_step += 1
            if run:
                wandb.log({"train_loss": loss.item(), "train_acc": batch_acc}, step=global_step)

        epoch_loss = total_loss_e / max(1, total_samples_e)
        epoch_acc = total_correct_e / max(1, total_samples_e)

        val_loss, val_acc, val_ece, _ = _evaluate(
            backbone, val_loader, device, num_classes, max_batches=eval_max_batches
        )

        if run:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": epoch_loss,
                    "train_acc_epoch": epoch_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_ece": val_ece,
                },
                step=global_step,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_dir = Path(cfg.results_dir) / cfg.run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "uncertainty_model": uncertainty_model.state_dict() if uncertainty_model else None,
                },
                ckpt_dir / "best.pt",
            )

    # ------------------------------------------------------------------
    # Final evaluation on test set (always compute confusion matrix)
    # ------------------------------------------------------------------
    test_loss, test_acc, test_ece, test_cm = _evaluate(
        backbone,
        test_loader,
        device,
        num_classes,
        max_batches=eval_max_batches,
        return_cm=True,
    )

    if run:
        run.summary["best_val_acc"] = best_val_acc
        run.summary["test_loss"] = test_loss
        run.summary["test_ece"] = test_ece
        run.summary["accuracy"] = test_acc  # primary metric
        if test_cm is not None:
            run.summary["confusion_matrix"] = test_cm.tolist()
        run.finish()

# -----------------------------------------------------------------------------
# Optuna objective (lightweight tuning)
# -----------------------------------------------------------------------------

def _objective(trial: optuna.Trial, cfg: DictConfig, device: torch.device):
    # Suggest hyper-parameters according to configured search spaces
    for space in cfg.optuna.search_spaces:
        dist = space.distribution_type.lower()
        if dist == "loguniform":
            val = trial.suggest_float(space.param_name, space.low, space.high, log=True)
        elif dist == "uniform":
            val = trial.suggest_float(space.param_name, space.low, space.high)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        OmegaConf.update(cfg, f"training.{space.param_name}", val, merge=False)

    # Short training for speed during optimisation
    original_epochs = cfg.training.epochs
    cfg.training.epochs = min(3, original_epochs)

    train_loader, val_loader, _, num_classes = get_dataloaders(cfg)

    model = create_backbone(cfg.model.name, num_classes).to(device)
    opt = _get_optimizer(cfg, model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for _ in range(cfg.training.epochs):
        model.train()
        for imgs, lbls, _ in train_loader:
            opt.zero_grad()
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = criterion(model(imgs), lbls)
            loss.backward()
            opt.step()
        # Val after each epoch – 2 batches only
        _, val_acc, _, _ = _evaluate(
            model, val_loader, device, num_classes, max_batches=2
        )
        best_val_acc = max(best_val_acc, val_acc)

    cfg.training.epochs = original_epochs  # restore
    return 1.0 - best_val_acc  # minimise 1-acc

# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    # Merge run-specific YAML (external)
    # ------------------------------------------------------------------
    run_cfg_path = Path(get_original_cwd()) / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_cfg_path}")

    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))

    # ------------------------------------------------------------------
    # Mode overrides
    # ------------------------------------------------------------------
    mode = str(cfg.mode).lower()
    if mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.batch_size = min(8, cfg.training.batch_size)
    elif mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Optuna hyper-parameter search (if enabled & not trial mode)
    # ------------------------------------------------------------------
    if int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: _objective(t, cfg, device), n_trials=int(cfg.optuna.n_trials))
        # Apply best params
        for k, v in study.best_trial.params.items():
            OmegaConf.update(cfg, f"training.{k}", v, merge=False)

    _train_single(cfg, device, trial_mode=(mode == "trial"))

if __name__ == "__main__":
    main()
