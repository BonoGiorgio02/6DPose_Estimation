import os
from sklearn.model_selection import train_test_split

def get_samples(split: str = "train", train_ratio: float = 0.7, seed: int = 42):
    path = "./datasets/linemod/DenseFusion/Linemod_preprocessed/data"
    samples = get_all_samples(path)

    # Check if samples were found
    if not samples:
        raise ValueError(f"No samples found in {path}. Check the dataset path and structure, and word in 6D_pose_estimation")
    
    # Split into training and validation+test sets
    labels = [el[0] for el in samples]
    train_samples, val_test_samples = train_test_split(samples, train_size=train_ratio, random_state=seed, stratify=labels)

    # split validation+test set (by default 30% of the original dataset) into validation and test sets
    labels = [el[0] for el in val_test_samples]
    val_samples, test_samples = train_test_split(val_test_samples, train_size=0.5, random_state=seed, stratify=labels)

    # Select the appropriate split
    if split == "train":
        return train_samples
    elif split == "validation":
        return val_samples
    else:
        return test_samples


def get_all_samples(path: str = None):
    """Retrieve the list of all available sample indices from all folders."""
    samples = []
    for folder_id in range(1, 16):  # Assuming folders are named 01 to 15
        folder_path = os.path.join(path, f"{folder_id:02d}", "rgb")
        # check path exists
        if os.path.exists(folder_path):
            # get id of the images
            sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
            samples.extend([(folder_id, sid) for sid in sample_ids])  # Store (folder_id, sample_id)
    return samples