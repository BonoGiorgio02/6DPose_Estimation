import torch
import torch.nn.functional as F
from utils.init import set_device

def pointcloud_collate_fn(batch):
    # find max size for padding
    max_H = max(item['cropped_img'].shape[1] for item in batch)
    max_W = max(item['cropped_img'].shape[2] for item in batch)
    
    padded_cropped_imgs = []
    paddings = []

    device = set_device()
    
    for item in batch:
        # --- symmetric padding for image ---
        img = item['cropped_img']
        _, H, W = img.shape
        pad_H = max_H - H
        pad_W = max_W - W

        # compute symmetric padding: (left, right, top, bottom)
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top

        padding = (pad_left, pad_right, pad_top, pad_bottom)
        # pad images by replicating the border pixels
        padded_img = F.pad(img, padding, mode='replicate')
        padded_cropped_imgs.append(padded_img)
        padding = torch.tensor([pad_left, pad_right, pad_top, pad_bottom])
        paddings.append(padding)

    batch_dict = {
        # "rgb": torch.stack([item['rgb'] for item in batch]).to(device),
        "cropped_img": torch.stack(padded_cropped_imgs).to(device),
        # "depth": torch.stack([item['depth'] for item in batch]).to(device),
        "pointcloud": torch.stack([item['pointcloud'] for item in batch]).to(device),  # list of tensor
        "camera_intrinsics": torch.stack([item['camera_intrinsics'] for item in batch]).to(device),
        "translation": torch.stack([item['translation'] for item in batch]).to(device),
        "rotation": torch.stack([item['rotation'] for item in batch]).to(device),
        "quaternion": torch.stack([item['quaternion'] for item in batch]).to(device),
        "bbox_base": torch.stack([item['bbox_base'] for item in batch]).to(device),
        "obj_id": torch.stack([item['obj_id'] for item in batch]).to(device),
        "sample_id": torch.stack([item['sample_id'] for item in batch]).to(device),
        "paddings":torch.stack(paddings).to(device),
    }

    return batch_dict