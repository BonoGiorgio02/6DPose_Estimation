import torch
import shutil
from ultralytics import YOLO

def get_YOLO(path: str = None):
    if path is None:
        print("Path cannot be None")
        return None
    return YOLO(f"{path}/checkpoints/yolo11n.pt")

def train_YOLO(path: str = None, epochs: int = None, batch_size: int = None, device = torch.device("cpu"), IMG_SIZE: int = None):
    """
        Train YOLO model, after training evaluate on validation set by returning metrics like mAP
        Save model to checkpoints
    """
    
    model = get_YOLO(path)

    # model will automatically scale the image and related bounding box according to imgsz
    results = model.train(
        data=f"{path}/datasets/linemod/YOLO/datasets/data.yaml",
        epochs=epochs,
        batch=batch_size,
        device=device,
        imgsz=IMG_SIZE,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.4,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=120,
        translate=0.1,
        scale=0.5,
        shear=20,
        perspective=0.0001,
        exist_ok=True,
        patience=5, #number of epoch to wait without improvement in validation metrics before early stopping the train. Helps prevent overfitting.
        dropout=0.3
        )
    
    shutil.copy(f"./datasets/linemod/YOLO/runs/detect/train/weights/best.pt", f"./checkpoints/best.pt")