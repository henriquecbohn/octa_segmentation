import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from models.smp_model import get_model
from utils.dataset import get_dataloaders

# CONFIGURACOES

ENCODER        = "efficientnet-b0"
RODADA         = 4
PROJECT_ROOT   = r"C:\Users\Phi Healthcare\octa_segmentation"
IMG_DIR        = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "images")
MASK_DIR       = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "masks")

BATCH_SIZE     = 4
LR_FASE1       = 3e-4
LR_FASE2       = 1e-4
EPOCAS_FASE1   = 15
EPOCAS_FASE2   = 85
EARLY_STOPPING = 15
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT     = os.path.join(PROJECT_ROOT, "results", f"best_model_rod{RODADA}_{ENCODER}.pth")
LOG_FILE       = os.path.join(PROJECT_ROOT, "results", f"training_log_rod{RODADA}_{ENCODER}.txt")
CURVE_FILE     = os.path.join(PROJECT_ROOT, "results", f"training_curves_rod{RODADA}_{ENCODER}.png")

# LOSS

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

# METRICAS

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
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float('inf')
        self.stop      = False

    def __call__(self, val_loss, model, checkpoint_path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), checkpoint_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False

# LOOPS

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks, _ in tqdm(loader, desc="  Treino", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        preds = torch.sigmoid(model(images))
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="  Validacao", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            preds  = torch.sigmoid(model(images))
            total_loss += criterion(preds, masks).item()
            total_dice += dice_coefficient(preds, masks)
            total_iou  += iou_score(preds, masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

# LOG

def log(msg, log_file):
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# PLOT

def plot_curves(history, stopped_epoch, fase1_end):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Rodada {RODADA} - U-Net + {ENCODER} (parou epoca {stopped_epoch})", fontsize=13)

    axes[0].plot(history['train_loss'], label='Treino')
    axes[0].plot(history['val_loss'],   label='Validacao')
    axes[0].axvline(x=fase1_end, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoca')
    axes[0].legend()

    axes[1].plot(history['val_dice'])
    axes[1].axvline(x=fase1_end, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Dice Score (Validacao)')
    axes[1].set_xlabel('Epoca')

    axes[2].plot(history['val_iou'])
    axes[2].axvline(x=fase1_end, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title('IoU Score (Validacao)')
    axes[2].set_xlabel('Epoca')

    plt.tight_layout()
    plt.savefig(CURVE_FILE, dpi=150)
    plt.show()
    print(f"Curvas salvas em {CURVE_FILE}")

# MAIN

def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    header = f"""
{'='*60}
EXPERIMENTO - RODADA {RODADA}
Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Encoder:         {ENCODER} (pre-treinado ImageNet)
Batch size:      {BATCH_SIZE}
Fase 1:          {EPOCAS_FASE1} epocas, encoder congelado, LR={LR_FASE1}
Fase 2:          ate {EPOCAS_FASE2} epocas, tudo livre, LR={LR_FASE2}
Early stopping:  patience={EARLY_STOPPING}
Dispositivo:     {DEVICE}
Historico:
  Rodada 1: Dice 0.888 | IoU 0.801
  Rodada 2: Dice 0.898 | IoU 0.814
  Rodada 3: Dice 0.884 | IoU 0.793
{'='*60}
"""
    log(header, LOG_FILE)

    train_loader, val_loader, _ = get_dataloaders(IMG_DIR, MASK_DIR, batch_size=BATCH_SIZE)

    model     = get_model(ENCODER).to(DEVICE)
    criterion = BCEDiceLoss()

    history       = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    stopped_epoch = EPOCAS_FASE1 + EPOCAS_FASE2

    # FASE 1: encoder congelado
   
    log(f"\n--- FASE 1: Encoder congelado ({EPOCAS_FASE1} epocas, LR={LR_FASE1}) ---\n", LOG_FILE)

    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FASE1
    )
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING)

    for epoch in range(1, EPOCAS_FASE1 + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        saved     = early_stopper(val_loss, model, CHECKPOINT)
        saved_str = "Melhor modelo salvo!" if saved else f"sem melhora: {early_stopper.counter}/{EARLY_STOPPING}"

        msg = (f"[F1] Epoca {epoch:3d}/{EPOCAS_FASE1} | "
               f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
               f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | {saved_str}")
        log(msg, LOG_FILE)

    fase1_end = len(history['train_loss'])

    # FASE 2: tudo livre
    
    log(f"\n--- FASE 2: Tudo livre ({EPOCAS_FASE2} epocas, LR={LR_FASE2}) ---\n", LOG_FILE)

    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR_FASE2)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING)

    for epoch in range(1, EPOCAS_FASE2 + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        saved     = early_stopper(val_loss, model, CHECKPOINT)
        saved_str = "Melhor modelo salvo!" if saved else f"sem melhora: {early_stopper.counter}/{EARLY_STOPPING}"

        msg = (f"[F2] Epoca {epoch:3d}/{EPOCAS_FASE2} | "
               f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
               f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | {saved_str}")
        log(msg, LOG_FILE)

        if early_stopper.stop:
            stopped_epoch = EPOCAS_FASE1 + epoch
            log(f"\nEarly Stopping na epoca total {stopped_epoch}!", LOG_FILE)
            break

    best_dice = max(history['val_dice'])
    best_iou  = max(history['val_iou'])

    summary = f"""
{'='*60}
RESULTADO FINAL - RODADA {RODADA} ({ENCODER})
  Epocas executadas:   {stopped_epoch}
  Melhor Val Loss:     {early_stopper.best_loss:.4f}
  Melhor Dice val:     {best_dice:.4f}
  Melhor IoU val:      {best_iou:.4f}
Historico completo:
  Rodada 1: Dice 0.888 | IoU 0.801
  Rodada 2: Dice 0.898 | IoU 0.814
  Rodada 3: Dice 0.884 | IoU 0.793
  Rodada {RODADA}: Dice {best_dice:.4f} | IoU {best_iou:.4f}
{'='*60}
"""
    log(summary, LOG_FILE)
    plot_curves(history, stopped_epoch, fase1_end)


if __name__ == "__main__":
    main()