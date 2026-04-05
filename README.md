# octa_segmentation

Segmentation pipeline for large retinal vessels in OCTA images, built on the [OCTA-500](https://ieee-dataport.org/open-access/octa-500) public dataset. Developed as part of a Master's research project at PPGCA/UNISINOS.

## Overview

Four training strategies were implemented and compared:

| Run | Model | Augmentation | Dice | IoU |
|-----|-------|-------------|------|-----|
| R1 | U-Net (scratch) | No | 0.869 | 0.770 |
| R2 | U-Net (scratch) | Yes | **0.886** | **0.797** |
| R3 | U-Net + ResNet34 | Yes | 0.871 | 0.774 |
| R4 | U-Net + EfficientNet-B0 | Yes | 0.878 | 0.784 |

All experiments used 300 subjects from the OCTA_6mm subset (projection map B5), split 70/15/15 into train/validation/test sets.

## Requirements

```bash
conda create -n octa_seg python=3.10 -y
conda activate octa_seg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn scikit-image tqdm opencv-python albumentations segmentation-models-pytorch
```

## Dataset

Download OCTA-500 from [IEEE Dataport](https://ieee-dataport.org/open-access/octa-500). Place the files at:

```
Desktop/Dataset/
  OCTA_6mm_part8/OCTA_6mm/Projection Maps/OCTA(ILM_OPL)/   ← 300 images
  Label/GT_LargeVessel/                                      ← 300 masks
```

Then run preprocessing:

```bash
python utils/preprocess.py
```

## Project structure

```
octa_segmentation/
├── data/
│   └── processed/OCTA_6mm/
│       ├── images/      ← 300 projection maps (400×400 px)
│       └── masks/       ← 300 binary masks
├── models/
│   ├── unet.py          ← U-Net from scratch (31M params)
│   └── smp_model.py     ← U-Net with pretrained encoders
├── utils/
│   ├── dataset.py       ← OCTADataset + DataLoader
│   ├── preprocess.py    ← copies images and masks
│   └── visualize.py     ← sanity check visualization
├── results/             ← checkpoints, curves, figures
├── train.py             ← training runs R1 and R2
├── train_smp.py         ← training runs R3 and R4
├── evaluate.py          ← final evaluation on test set
├── predict_single.py    ← inference on a single image
└── generate_figures.py  ← figures for the paper
```

## Training

**R1 — baseline (no augmentation):**
```bash
python train.py
```

**R2 — with augmentation (best result):**
Set `NUM_EPOCHS = 100` and `CHECKPOINT = "best_model_aug.pth"` in `train.py`, then:
```bash
python train.py
```

**R3 and R4 — transfer learning:**
Set `ENCODER` and `RODADA` at the top of `train_smp.py`:
```python
ENCODER = "resnet34"        # or "efficientnet-b0"
RODADA  = 3                 # or 4
```
Then:
```bash
python train_smp.py
```

## Inference on a single image

```bash
python predict_single.py
```

Edit `IMAGE_PATH` inside the script to point to your OCTA image.

## Hardware

Tested on NVIDIA GeForce RTX 4050 (6.4 GB VRAM), 32 GB RAM, Windows 11.

## Reference

Li, M. et al. OCTA-500: A Retinal Dataset for Optical Coherence Tomography Angiography Study. *IEEE Transactions on Medical Imaging*, 2022.

## Author

Henrique Christoph Bohn — PPGCA/UNISINOS
