from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    
    # Data
    data_dir:   str = str(ROOT / "data")
    train_dir:  str = str(ROOT / "data/final_data/train")
    val_dir:    str = str(ROOT / "data/final_data/val")
    test_dir:   str = str(ROOT / "data/cars_test")
    num_workers:        int   = 4 
    img_size:           int = 224
    batch_size:         int = 32
    num_classes:        int = 196
    val_split:          float = 0.2
    
    # Model
    model_name:         str   = "resnet50"
    pretrained:         bool  = True
    dropout:            float = 0.4
    use_amp:            bool  = True 
    
    # Phase 1
    epoch_phase1:       int     = 5
    lr_phase1:          float   = 1e-3
    weight_decay:       float   = 1e-4
    
    # Phase 2
    epochs_phase2:      int   = 20
    lr_head_phase2:     float = 1e-4         
    lr_backbone_phase2: float = 1e-5      
    label_smoothing:    float = 0.1 
    
    # Scheduler
    scheduler:          str   = "cosine"   
    warmup_epochs:      int   = 2   

    # Checkpoint
    save_top_k:         int   = 1          
    patience:           int   = 7
        
    # Path
    output_dir:       str = str(ROOT / "outputs")
    best_model_path:  str = str(ROOT / "outputs/best_model.pth")
    last_model_path:  str = str(ROOT / "outputs/last_model.pth")
    history_path:     str = str(ROOT / "outputs/training_history.json")
    class_names_path: str = str(ROOT / "class_names.json")
    
    seed:               int   = 42
    log_interval:       int   = 50 
    
        
CFG = Config()


if __name__ == "__main__":
    print("=" * 50)
    print("Current Configuration:")
    print("=" * 50)
    for key, value in CFG.__dict__.items():
        print(f"  {key:<25} = {value}")
    print("=" * 50)
