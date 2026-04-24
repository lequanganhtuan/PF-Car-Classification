import json
import math
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from PIL import Image

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
    
class RawImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_files = sorted([f for f in self.root.iterdir() if f.is_file()])

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(img_path.name) # Trả về ảnh và tên file (vì chưa có nhãn)

    def __len__(self):
        return len(self.image_files)
    
def get_dataloader() -> Tuple[DataLoader, DataLoader]:
    print("[Dataset] Car Classification")
    
    train_dataset = ImageFolder(
        root = CFG.train_dir,
        transform = get_train_transforms()
    )
    
    val_dataset = ImageFolder(
        root = CFG.val_dir,
        transform = get_val_transform()
    )
    
    test_dataset = RawImageFolder(
        root = CFG.test_dir, 
        transform = get_val_transform()
    )
    
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    try:
        with open(CFG.class_names_path, 'w', encoding='utf-8') as f:
            json.dump(idx_to_class, f, indent=4)
        print(f"[INFO] class_names.json saved successfully at: {CFG.class_names_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save class_names.json: {e}")
        
        
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
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CFG.batch_size, 
        shuffle=False, # Test tuyệt đối không shuffle
        num_workers=CFG.num_workers,
        pin_memory=True
    )
    
    print(f"[Dataset] Train samples : {len(train_dataset):,}")
    print(f"[Dataset] Val samples   : {len(val_dataset):,}")
    print(f"[Dataset] Train batches : {len(train_loader)}")
    print(f"[Dataset] Val batches   : {len(val_loader)}")

    return train_loader, val_loader, test_loader

def load_class_names() -> dict:
    with open(CFG.class_names_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {int(k): v for k, v in data.items()}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    train_loader, val_loader, test_loader = get_dataloader()

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