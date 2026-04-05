# utils/dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = r"C:\Users\Phi Healthcare\octa_segmentation"
IMG_DIR  = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "images")
MASK_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "masks")

# AUGMENTATIONS

train_transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.4
    ),
    A.GaussNoise(p=0.2),
    A.ElasticTransform(
        alpha=1, sigma=50, p=0.2
    ),
])

# Validação/Teste: sem augmentation (só normalização)
val_transform = None


class OCTADataset(Dataset):
    def __init__(self, subject_ids, img_dir, mask_dir, transform=None):
        self.subject_ids = subject_ids
        self.img_dir     = img_dir
        self.mask_dir    = mask_dir
        self.transform   = transform

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        img_path = os.path.join(self.img_dir, f"{subject_id}.bmp")
        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0

        mask_path = os.path.join(self.mask_dir, f"{subject_id}.bmp")
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']

        image = torch.from_numpy(image).unsqueeze(0)
        mask  = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, subject_id


def get_dataloaders(img_dir, mask_dir, batch_size=8, seed=42):
    all_ids = sorted([
        f.replace('.bmp', '')
        for f in os.listdir(img_dir)
        if f.endswith('.bmp')
    ])

    print(f"Total de sujeitos: {len(all_ids)}")

    train_ids, temp_ids = train_test_split(all_ids, test_size=0.30, random_state=seed)
    val_ids, test_ids   = train_test_split(temp_ids, test_size=0.50, random_state=seed)

    print(f"  Treino:    {len(train_ids)} sujeitos")
    print(f"  Validação: {len(val_ids)} sujeitos")
    print(f"  Teste:     {len(test_ids)} sujeitos")

    # Treino COM augmentation, val/teste SEM
    train_dataset = OCTADataset(train_ids, img_dir, mask_dir, transform=train_transform)
    val_dataset   = OCTADataset(val_ids,   img_dir, mask_dir, transform=val_transform)
    test_dataset  = OCTADataset(test_ids,  img_dir, mask_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(IMG_DIR, MASK_DIR, batch_size=4)
    images, masks, ids = next(iter(train_loader))
    print(f"Shape imagens: {images.shape}")
    print(f"Shape máscaras: {masks.shape}")
    print(f"Min/Max: {images.min():.3f} / {images.max():.3f}")
    print(f"Máscara valores únicos: {masks.unique()}")
    print("✅ Dataset com augmentation OK!")