from dataclasses import dataclass, field
from pathlib import Path
import os

IS_KAGLLE = os.path.exists('/kaggle/input/datasets/lquanganhtun/car-classification1')

if IS_KAGLLE:
    DATA_ROOT = Path('/kaggle/input/datasets/lquanganhtun/car-classification1')
    WORKING_ROOT = Path('/kaggle/working/project')
    TRAIN_PATH = str(DATA_ROOT / "data/final_data/train")
    VAL_PATH   = str(DATA_ROOT / "data/final_data/val")
    TEST_PATH  = str(DATA_ROOT / "data/cars_test")
    OUT_DIR    = str(WORKING_ROOT / "outputs")
    CLASS_PATH = str(WORKING_ROOT / "class_names.json")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TRAIN_PATH = str(PROJECT_ROOT / "data/final_data/train")
    VAL_PATH   = str(PROJECT_ROOT / "data/final_data/val")
    TEST_PATH  = str(PROJECT_ROOT / "data/cars_test")
    OUT_DIR    = str(PROJECT_ROOT / "outputs")
    CLASS_PATH = str(PROJECT_ROOT / "class_names.json")

@dataclass
class Config:
    # Data
    train_dir:   str = TRAIN_PATH
    val_dir:     str = VAL_PATH
    test_dir:    str = TEST_PATH
    num_workers: int = 0 #4
    img_size:    int = 448
    batch_size:  int = 16
    num_classes: int = 196
    val_split:   float = 0.2
    
    model_name:         str   = "resnet50"
    pretrained:         bool  = True
    dropout:            float = 0.5
    use_amp:            bool  = True 
    epoch_phase1:       int   = 5
    lr_phase1:          float = 1e-3
    epochs_phase2:      int   = 30
    lr_head_phase2:     float = 1e-4
    lr_backbone_phase2: float = 1e-5
    
    output_dir:       str = OUT_DIR
    best_model_path:  str = os.path.join(OUT_DIR, "best_model.pth")
    last_model_path:  str = os.path.join(OUT_DIR, "last_model.pth")
    history_path:     str = os.path.join(OUT_DIR, "training_history.json")
    class_names_path: str = CLASS_PATH
    
    seed:             int = 42
    log_interval:     int = 50 
    patience:         int = 7
    warmup_epochs:    int = 2
    weight_decay:     float = 1e-4
    label_smoothing:  float = 0.1
    scheduler:        str = "cosine"

CFG = Config()


if __name__ == "__main__":
    print("=" * 50)
    print("Current Configuration:")
    print("=" * 50)
    for key, value in CFG.__dict__.items():
        print(f"  {key:<25} = {value}")
    print("=" * 50)
