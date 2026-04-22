import json
import math
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from config import CFG

def get_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(
            CFG.img_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),    
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        T.RandomGrayscale(p=0.05),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225],   
        ),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])
    
def get_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(CFG.img_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
def get_dataloader() -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("[Dataset] Car Classification")
    
    train_dataset = ImageFolder(
        root = CFG.train_dir,
        transform = get_train_transforms()
    )
    
    val_dataset = ImageFolder(
        root = CFG.val_dir,
        transform = get_val_transform()
    )
    
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    # Lưu class names vào file json nếu cần (như trong Config bạn dự định)
    with open(CFG.class_names_path, 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, indent=4)
        
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,                  
        drop_last=True,                   
        persistent_workers=CFG.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size * 2,  
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=CFG.num_workers > 0,
    )
    
    print(f"[Dataset] Train samples : {len(train_dataset):,}")
    print(f"[Dataset] Val samples   : {len(val_dataset):,}")
    print(f"[Dataset] Train batches : {len(train_loader)}")
    print(f"[Dataset] Val batches   : {len(val_loader)}")

    return train_loader, val_loader

def load_class_names() -> dict:
    with open(CFG.class_names_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {int(k): v for k, v in data.items()}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    train_loader, val_loader = get_dataloader()

    images, labels = next(iter(train_loader))
    print(f"\n[Sanity Check]")
    print(f"  Batch shape  : {images.shape}")      
    print(f"  Labels shape : {labels.shape}")      
    print(f"  Label range  : [{labels.min()}, {labels.max()}]")  
    print(f"  Pixel range  : [{images.min():.3f}, {images.max():.3f}]")

    # Visualize 8 ảnh đã augment
    class_names = load_class_names()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Training Batch — After Augmentation", fontsize=13)

    for i, ax in enumerate(axes.flat):
        img = images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img * STD + MEAN, 0, 1)
        ax.imshow(img)
        ax.set_title(class_names[labels[i].item()].replace("_", " "), fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/sample_batch.png", dpi=120)
    print("\n[Sanity Check] Sample batch saved to outputs/sample_batch.png")
    plt.show()