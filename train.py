import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from models.unet import UNet
from utils.dataset import get_dataloaders

# CONFIGURAÇÕES

PROJECT_ROOT = r"C:\Users\Phi Healthcare\octa_segmentation"
IMG_DIR  = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "images")
MASK_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "masks")

BATCH_SIZE     = 4
LR             = 1e-4
NUM_EPOCHS     = 100
EARLY_STOPPING = 15    # para se val_loss não melhorar por 15 épocas
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT     = os.path.join(PROJECT_ROOT, "results", "best_model_aug.pth")
LOG_FILE       = os.path.join(PROJECT_ROOT, "results", "training_log_aug.txt")

# LOSS FUNCTIONS

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)

# MÉTRICAS

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return ((2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()

# EARLY STOPPING

class EarlyStopping:
    
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float('inf')
        self.stop       = False

    def __call__(self, val_loss, model, checkpoint_path):
        if val_loss < self.best_loss - self.min_delta:
            
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), checkpoint_path)
            return True  # salvou
        else:
        
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False  # não salvou

# LOOPS

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks, _ in tqdm(loader, desc="  Treino", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="  Validação", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images)
            total_loss += criterion(preds, masks).item()
            total_dice += dice_coefficient(preds, masks)
            total_iou  += iou_score(preds, masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

# LOG PARA PAPER

def log(msg, log_file):
    """Salva mensagem no terminal e no arquivo de log."""
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# PLOT

def plot_curves(history, stopped_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Rodada 2 — Com Augmentation (parou época {stopped_epoch})", fontsize=13)

    axes[0].plot(history['train_loss'], label='Treino')
    axes[0].plot(history['val_loss'],   label='Validação')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Época')
    axes[0].legend()

    axes[1].plot(history['val_dice'])
    axes[1].set_title('Dice Score (Validação)')
    axes[1].set_xlabel('Época')

    axes[2].plot(history['val_iou'])
    axes[2].set_title('IoU Score (Validação)')
    axes[2].set_xlabel('Época')

    plt.tight_layout()
    out = os.path.join(PROJECT_ROOT, "results", "training_curves_aug.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"✅ Curvas salvas em {out}")

# MAIN

def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    header = f"""
{'='*60}
EXPERIMENTO — RODADA 2
Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Configurações:
  Épocas máx:      {NUM_EPOCHS}
  Batch size:      {BATCH_SIZE}
  Learning rate:   {LR}
  Early stopping:  patience={EARLY_STOPPING}
  Augmentation:    Rotation±30, HFlip, VFlip,
                   Brightness/Contrast, GaussNoise,
                   ElasticTransform
  Dispositivo:     {DEVICE}
  Checkpoint:      {CHECKPOINT}
Baseline Rodada 1:
  Épocas:          30 (sem augmentation)
  Melhor Dice val: 0.890
  Melhor IoU val:  0.801
{'='*60}
"""
    log(header, LOG_FILE)


    train_loader, val_loader, _ = get_dataloaders(IMG_DIR, MASK_DIR, batch_size=BATCH_SIZE)


    model         = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion     = BCEDiceLoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING)

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    stopped_epoch = NUM_EPOCHS

    log("Iniciando treinamento...\n", LOG_FILE)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss             = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        saved = early_stopper(val_loss, model, CHECKPOINT)
        saved_str = "💾 Melhor modelo salvo!" if saved else f"  (sem melhora: {early_stopper.counter}/{EARLY_STOPPING})"

        msg = (f"Época {epoch:3d}/{NUM_EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Val Loss: {val_loss:.4f} | "
               f"Dice: {val_dice:.4f} | "
               f"IoU: {val_iou:.4f} | {saved_str}")
        log(msg, LOG_FILE)

        if early_stopper.stop:
            stopped_epoch = epoch
            msg = f"\n⛔ Early Stopping ativado na época {epoch}!"
            msg += f"\n   Melhor Val Loss: {early_stopper.best_loss:.4f}"
            log(msg, LOG_FILE)
            break

    best_dice = max(history['val_dice'])
    best_iou  = max(history['val_iou'])
    summary = f"""
{'='*60}
RESULTADO FINAL — RODADA 2
  Épocas executadas:   {stopped_epoch}
  Melhor Val Loss:     {early_stopper.best_loss:.4f}
  Melhor Dice val:     {best_dice:.4f}
  Melhor IoU val:      {best_iou:.4f}
  Comparação Rodada 1: Dice 0.890 → {best_dice:.4f}
{'='*60}
"""
    log(summary, LOG_FILE)
    plot_curves(history, stopped_epoch)


if __name__ == "__main__":
    main()