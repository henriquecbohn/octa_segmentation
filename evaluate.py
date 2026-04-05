import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from models.unet import UNet
from models.smp_model import get_model
from utils.dataset import get_dataloaders, OCTADataset
from sklearn.model_selection import train_test_split

# CONFIGURACOES

PROJECT_ROOT = r"C:\Users\Phi Healthcare\octa_segmentation"
IMG_DIR      = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "images")
MASK_DIR     = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "masks")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelos a avaliar
MODELOS = {
    "Rodada1_UNet_scratch":    {"type": "unet",   "checkpoint": "best_model.pth"},
    "Rodada2_UNet_aug":        {"type": "unet",   "checkpoint": "best_model_aug.pth"},
    "Rodada3_ResNet34":        {"type": "smp",    "checkpoint": "best_model_rod3_resnet34.pth",       "encoder": "resnet34"},
    "Rodada4_EfficientNet-B0": {"type": "smp",    "checkpoint": "best_model_rod4_efficientnet-b0.pth","encoder": "efficientnet-b0"},
}
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


def precision_recall(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float().view(-1)
    target   = target.float().view(-1)
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = ((tp + smooth) / (tp + fp + smooth)).item()
    recall    = ((tp + smooth) / (tp + fn + smooth)).item()
    return precision, recall

# CARREGA MODELO

def load_model(config):
    checkpoint_path = os.path.join(RESULTS_DIR, config["checkpoint"])

    if config["type"] == "unet":
        model = UNet(in_channels=1, out_channels=1)
    else:
        model = get_model(config["encoder"])

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# AVALIACAO COMPLETA

def evaluate_model(model, test_loader, model_type):
    all_dice, all_iou, all_prec, all_rec = [], [], [], []

    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="  Avaliando", leave=False):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            if model_type == "unet":
                preds = model(images)
            else:
                preds = torch.sigmoid(model(images))

            for i in range(len(images)):
                pred = preds[i:i+1]
                mask = masks[i:i+1]
                all_dice.append(dice_coefficient(pred, mask))
                all_iou.append(iou_score(pred, mask))
                p, r = precision_recall(pred, mask)
                all_prec.append(p)
                all_rec.append(r)

    return {
        "dice_mean": np.mean(all_dice),
        "dice_std":  np.std(all_dice),
        "iou_mean":  np.mean(all_iou),
        "iou_std":   np.std(all_iou),
        "prec_mean": np.mean(all_prec),
        "rec_mean":  np.mean(all_rec),
        "all_dice":  all_dice,
    }
# VISUALIZACAO DAS PREDICOES

def visualize_predictions(model, test_loader, model_name, model_type, n=4):
    model.eval()
    images_list, masks_list, preds_list, ids_list = [], [], [], []

    with torch.no_grad():
        for images, masks, ids in test_loader:
            images = images.to(DEVICE)
            if model_type == "unet":
                preds = model(images)
            else:
                preds = torch.sigmoid(model(images))

            for i in range(len(images)):
                images_list.append(images[i].cpu().squeeze().numpy())
                masks_list.append(masks[i].squeeze().numpy())
                preds_list.append(preds[i].cpu().squeeze().numpy())
                ids_list.append(ids[i])

            if len(images_list) >= n:
                break

    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    fig.suptitle(f"Predicoes — {model_name}", fontsize=13)

    for i in range(n):
        img      = images_list[i]
        mask     = masks_list[i]
        pred     = preds_list[i]
        pred_bin = (pred > 0.5).astype(np.float32)

        # Overlay predicao
        overlay = np.stack([img, img, img], axis=-1)
        overlay[pred_bin == 1] = [1.0, 0.2, 0.2]

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"ID: {ids_list[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_bin, cmap='gray')
        axes[i, 2].set_title(f"Predicao (Dice={dice_coefficient(torch.tensor(pred).unsqueeze(0).unsqueeze(0), torch.tensor(mask).unsqueeze(0).unsqueeze(0)):.3f})")
        axes[i, 2].axis('off')

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis('off')

    plt.tight_layout()
    safe_name = model_name.replace("/", "-").replace(" ", "_")
    out_path  = os.path.join(RESULTS_DIR, f"predictions_{safe_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Visualizacao salva: {out_path}")


# MAIN

def main():
    # Reconstroi conjunto de teste (mesma seed do treino)
    all_ids = sorted([f.replace('.bmp', '') for f in os.listdir(IMG_DIR) if f.endswith('.bmp')])
    _, temp_ids  = train_test_split(all_ids, test_size=0.30, random_state=42)
    _, test_ids  = train_test_split(temp_ids, test_size=0.50, random_state=42)

    test_dataset = OCTADataset(test_ids, IMG_DIR, MASK_DIR, transform=None)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"Sujeitos no teste: {len(test_ids)}")
    print(f"Dispositivo: {DEVICE}")
    print()

    log_path = os.path.join(RESULTS_DIR, "evaluation_results.txt")
    results  = {}

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"AVALIACAO FINAL NO CONJUNTO DE TESTE\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Sujeitos no teste: {len(test_ids)}\n")
        f.write("="*60 + "\n\n")

    for nome, config in MODELOS.items():
        print(f"\nAvaliando: {nome}")
        checkpoint_path = os.path.join(RESULTS_DIR, config["checkpoint"])

        if not os.path.exists(checkpoint_path):
            print(f"  CHECKPOINT NAO ENCONTRADO: {checkpoint_path}")
            continue

        model   = load_model(config)
        metrics = evaluate_model(model, test_loader, config["type"])
        results[nome] = metrics

        # Gera visualizacoes
        visualize_predictions(model, test_loader, nome, config["type"], n=4)

        msg = (f"{nome}:\n"
               f"  Dice:      {metrics['dice_mean']:.4f} +/- {metrics['dice_std']:.4f}\n"
               f"  IoU:       {metrics['iou_mean']:.4f}  +/- {metrics['iou_std']:.4f}\n"
               f"  Precision: {metrics['prec_mean']:.4f}\n"
               f"  Recall:    {metrics['rec_mean']:.4f}\n")
        print(msg)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    # Tabela resumo final
    print("\n" + "="*65)
    print(f"{'Modelo':<35} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
    print("="*65)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\nTABELA RESUMO\n")
        f.write("="*65 + "\n")
        f.write(f"{'Modelo':<35} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}\n")
        f.write("="*65 + "\n")

    for nome, metrics in results.items():
        linha = (f"{nome:<35} "
                 f"{metrics['dice_mean']:>8.4f} "
                 f"{metrics['iou_mean']:>8.4f} "
                 f"{metrics['prec_mean']:>8.4f} "
                 f"{metrics['rec_mean']:>8.4f}")
        print(linha)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(linha + "\n")

    print("="*65)
    print(f"\nResultados salvos em: {log_path}")


if __name__ == "__main__":
    main()