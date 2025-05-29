import os
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import v2

IMG_WIDTH = 640
IMG_HEIGHT = 480

class CustomDatasetPose(Dataset):
    def __init__(self, dataset_root, split="train", device=torch.device("cpu")):
        """
        Args:
            dataset_root (str): Path to the dataset directory.
            split (str): 'train', 'validation' or 'test'.
        """
        self.dataset_root = dataset_root
        if split == "train":
            self.split = split
        elif split == "validation":
            self.split = "val"
        else:
            self.split = "test"
        
        self.device = device

        # Get list of all samples (folder_id, sample_id)
        self.samples = self.get_all_samples()

        # Check if samples were found
        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}. Check the dataset path and structure.")

        if split == 'train':
            # Define image transformations
            self.transform = v2.Compose([
              v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
              v2.RandomGrayscale(p=0.1),
              v2.GaussianBlur(kernel_size=3),
              v2.Resize((224, 224)),
              v2.ToImage(),  # Converte in PILImage o equivalente
              v2.ToDtype(torch.float32, scale=True),  # Converte in Tensor con valori tra 0 e 1
              v2.Normalize(mean=[0.3348, 0.3165, 0.3105], std=[0.2521, 0.2496, 0.2502])
          ])

        else:
          self.transform = v2.Compose([
              v2.Resize((224, 224)),
              v2.ToImage(),  # Converte in PILImage o equivalente
              v2.ToDtype(torch.float32, scale=True),  # Converte in Tensor con valori tra 0 e 1
              v2.Normalize(mean=[0.3348, 0.3165, 0.3105], std=[0.2521, 0.2496, 0.2502])
          ])

    def get_samples_id(self):
        return self.samples

    def get_all_samples(self):
        """Retrieve the list of all available sample indices from all folders."""
        for folder in ["train","val","test"]:
            if folder == self.split:
                folder_path = os.path.join(self.dataset_root, f"{folder}")
                #print(folder_path)
                if os.path.exists(folder_path):
                    # get name of files <folder id>_<image>
                    sample_ids = sorted([f.split('.')[0] for f in os.listdir(folder_path) if f.endswith('.png')])
        return sample_ids

    #Define here some usefull functions to access the data
    def load_image(self, img_path):
        """Load an RGB image and convert to tensor."""
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def get_image_path(self, idx):
        """Get the image path for a specific sample index."""
        sample = self.samples[idx]
        return os.path.join(self.dataset_root, f"{self.split}", f"{sample}.png")

    def load_6d_pose(self, sample_id):
        """Load the 6D pose (translation and rotation) for the object in this sample."""
        label = int(sample_id.split("_")[0])
        objectId = int(sample_id.split("_")[1])
        pose_file = os.path.join(self.dataset_root, f"{label:02d}_gt.yml")

        # Load the ground truth poses from the gt.yml file
        with open(pose_file, 'r') as f:
            pose_data = yaml.load(f, Loader=yaml.FullLoader)

        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file {pose_file} not found")

        if pose_data is None:
            print(f"{pose_file}")
            raise ValueError(f"Pose data not found for object {sample_id}")

        # The pose data is a dictionary where each key corresponds to a frame with pose info
        # We assume sample_id corresponds to the key in pose_data
        if objectId not in pose_data:
            raise KeyError(f"Sample ID {objectId} not found in {label:02d}_gt.yml.")

        for pose in pose_data[objectId]: # There can be more than one pose per sample, but take the one of label=folder_id
            # Extract translation and rotation
            if (int(pose['obj_id']) == int(label)):
                translation = np.array(pose['cam_t_m2c'], dtype=np.float32)/1000.0  # [3] ---> (x,y,z)
                rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)  # [3x3] ---> rotation matrix
                quaternion = np.array(pose["quaternion"], dtype= np.float32)
                # bbox is top left corner and width and height info, YOLO needs center coordinates and width and height
                x_min, y_min, width, height = np.array(pose['obj_bb'], dtype=np.float32) # [4] ---> x_min, y_min, width, height
                # compute initial center
                x_center = x_min + width/2
                y_center = y_min + height/2

                # store coordinates of the center and width and height of the bounding box normalized to the
                # image width=640 pixels and height=480 pixels
                bbox = np.array([x_center/IMG_WIDTH, y_center/IMG_HEIGHT, width/IMG_WIDTH, height/IMG_HEIGHT], dtype=np.float32)

                obj_id = np.array(pose['obj_id'], dtype=np.float32) # [1] ---> label
                break

        return translation, rotation, bbox, obj_id, quaternion

    def __len__(self):
        #Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        #Load a dataset sample.
        sample = self.samples[idx]

        img_path = os.path.join(self.dataset_root, f"{self.split}", f"{sample}.png")

        img = self.load_image(img_path)
        translation, rotation, bbox, obj_id, quaternion = self.load_6d_pose(sample)

        #Dictionary with all the data
        return {
          "rgb": img,
          "translation": torch.tensor(translation).to(self.device),
          "rotation": torch.tensor(rotation).to(self.device),
          "bbox": torch.tensor(bbox).to(self.device),
          "obj_id": torch.tensor(obj_id).to(self.device),
          "quaternion": torch.tensor(quaternion).to(self.device),
          "img_path": img_path,
          "sample_id": sample
}