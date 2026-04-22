from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class Config:
    
    # Data
    data_dir:           str = "../data"    
    # data_dir:           str = "./data" #Kaggle
    train_dir:          str = ""
    val_dir:            str = ""
    
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
    output_dir:         str   = "../outputs"
    best_model_path:    str   = "../outputs/best_model.pth"
    last_model_path:    str   = "../outputs/last_model.pth"
    history_path:       str   = "../outputs/training_history.json"
    class_names_path:   str   = "../class_names.json"
    
    seed:               int   = 42
    log_interval:       int   = 50 
    
    def __post_init__(self):
        # Tự động gán đường dẫn dựa trên data_dir
        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir   = os.path.join(self.data_dir, "val")
        
        # Tạo folder output và folder dữ liệu nếu chưa có
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra sự tồn tại của dữ liệu
        if not os.path.exists(self.train_dir):
            print(f"[WARNING] Can't find train path: {os.path.abspath(self.train_dir)}")
        else:
            print(f"[INFO] Data found at: {os.path.abspath(self.data_dir)}")
        
CFG = Config()


if __name__ == "__main__":
    print("=" * 50)
    print("Current Configuration:")
    print("=" * 50)
    for key, value in CFG.__dict__.items():
        print(f"  {key:<25} = {value}")
    print("=" * 50)
