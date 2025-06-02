import torch
from ultralytics import YOLO

def get_YOLO(path: str = None):
    if path is None:
        print("Path cannot be None")
        return None
    return YOLO(f"{path}/checkpoints/best.pt")

def evaluate_YOLO(path: str = None, epochs: int = None, batch_size: int = None, IMG_SIZE: int = None, device = torch.device("cpu")):
    """
        Evaluate model on test split
    """
    model = get_YOLO(path)

    results = model.val(
        data=f"{path}/datasets/linemod/YOLO/datasets/data.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=IMG_SIZE,
        device=device,
        split="test"
    )