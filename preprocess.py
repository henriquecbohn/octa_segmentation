import os
import shutil
from tqdm import tqdm

DATASET_ROOT = r"C:\Users\Phi Healthcare\Desktop\Dataset"

SRC_IMAGES = os.path.join(DATASET_ROOT, "OCTA_6mm_part8", "OCTA_6mm", "Projection Maps", "OCTA(ILM_OPL)")
SRC_MASKS  = os.path.join(DATASET_ROOT, "Label", "GT_LargeVessel")

PROJECT_ROOT = r"C:\Users\Phi Healthcare\octa_segmentation"
OUT_IMAGES   = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "images")
OUT_MASKS    = os.path.join(PROJECT_ROOT, "data", "processed", "OCTA_6mm", "masks")

def main():
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_MASKS,  exist_ok=True)

    subjects = sorted([
        f.replace('.bmp', '')
        for f in os.listdir(SRC_IMAGES)
        if f.endswith('.bmp')
    ])

    print(f"✅ Projection maps encontradas: {len(subjects)}")
    print(f"   IDs: {subjects[0]} → {subjects[-1]}")
    print()

    sucesso, sem_mask = 0, 0

    for subject_id in tqdm(subjects, desc="Copiando arquivos"):

        src_img = os.path.join(SRC_IMAGES, f"{subject_id}.bmp")
        dst_img = os.path.join(OUT_IMAGES, f"{subject_id}.bmp")
        shutil.copy2(src_img, dst_img)

        src_mask = os.path.join(SRC_MASKS, f"{subject_id}.bmp")
        dst_mask = os.path.join(OUT_MASKS,  f"{subject_id}.bmp")

        if os.path.exists(src_mask):
            shutil.copy2(src_mask, dst_mask)
            sucesso += 1
        else:
            print(f"  ⚠️  Máscara não encontrada: {subject_id}")
            sem_mask += 1

    print()
    print("=" * 40)
    print(f"✅ Pares completos:         {sucesso}")
    print(f"⚠️  Sem máscara:            {sem_mask}")
    print(f"📁 Imagens: {OUT_IMAGES}")
    print(f"📁 Máscaras: {OUT_MASKS}")
    print("=" * 40)

if __name__ == "__main__":
    main()