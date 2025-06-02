import os
import yaml
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

IMG_WIDTH = 640
IMG_HEIGHT = 480

class CustomDatasetPose(Dataset): # used to load and preprocess data
    def __init__(self, dataset_root, split='train', train_ratio=0.7, seed=42, device=torch.device("cpu"), cam_K=None):
        """
        Args:
            dataset_root (str): Path to the dataset directory.
            split (str): 'train', 'validation' or 'test'.
            train_ratio (float): Percentage of data used for training (default 70%).
            seed (int): Random seed for reproducibility.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.device = device
        self.camera_intrinsics = [cam_K[0], cam_K[4], cam_K[2], cam_K[5]]

        # Get list of all samples (folder_id, sample_id)
        self.samples, self.folder_names = self.get_all_samples()

        # Check if samples were found
        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}. Check the dataset path and structure.")

        # Split into training and validation+test sets
        labels = [el[0] for el in self.samples]
        self.train_samples, self.val_test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed, stratify=labels
        )

        # split validation+test set (by default 30% of the original dataset) into validation and test sets
        labels = [el[0] for el in self.val_test_samples]
        self.val_samples, self.test_samples = train_test_split(self.val_test_samples, train_size=0.5, random_state=self.seed, stratify=labels)

        # Select the appropriate split
        if split == "train":
            self.samples = self.train_samples
        elif split == "validation":
            self.samples = self.val_samples
        else:
            self.samples = self.test_samples

        # find mean and standard deviation of images
        mean, std = self.findMeanStd(self.train_samples)

        # Define image transformations for the baseline
        if self.split == 'train':
            self.transform_img = transforms.Compose([
                                # transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
                                # transforms.RandomGrayscale(p=0.1),
                                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                                transforms.ToTensor(),  # convert to float32 and normalize to [0, 1]
                                # transforms.Normalize(mean=mean.tolist(), std=std.tolist())
                            ])

            self.transform_crop = transforms.Compose([
                                # transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
                                # transforms.RandomGrayscale(p=0.1),
                                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                                transforms.Resize((224, 224)), # such that all images of the batch have same shape
                                transforms.ToTensor(),
                                # normalize images according to these values found
                                transforms.Normalize(mean=mean.tolist(), std=std.tolist())
                            ])
        else:
            self.transform_img = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=mean.tolist(), std=std.tolist())
                            ])

            self.transform_crop = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean.tolist(), std=std.tolist())
                            ])

        # store everything instead of opening each time file, this can speed up computation
        self.ground_truths = self.extract_ground_truth()

    def get_samples_id(self):
        return self.samples

    def get_all_samples(self):
        """Retrieve the list of all available sample indices from all folders."""
        folder_names = []
        samples = []
        for folder_id in range(1, 16):  # Assuming folders are named 01 to 15
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "rgb")
            #print(folder_path)
            if os.path.exists(folder_path):
                # get id of the images
                folder_names.append(folder_id)
                sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
                samples.extend([(folder_id, sid) for sid in sample_ids])  # Store (folder_id, sample_id)
        return samples, folder_names
    
    def findMeanStd(self, train_samples):
        """
            Given dataset root and training samples in the format (folder_id, sample_id), compute mean and standard deviation
        """

        # Transform to tensor [C, H, W]
        to_tensor = transforms.ToTensor()

        # initialize sum
        sum_rgb = torch.zeros(3)
        sum_sq_rgb = torch.zeros(3)
        n_pixels = 0

        for el in train_samples:
            # get image
            image_path = os.path.join(self.dataset_root, 'data', f"{el[0]:02d}", "rgb", f"{el[1]:04d}.png")
            img = Image.open(image_path).convert('RGB') # ensure channel order is RGB
            img_tensor = to_tensor(img)  # shape: [3, H, W]
            # add
            sum_rgb += img_tensor.sum(dim=[1, 2]) # sum over H and W, so for each channel a value
            sum_sq_rgb += (img_tensor ** 2).sum(dim=[1, 2])
            n_pixels += img_tensor.shape[1] * img_tensor.shape[2]
        
        # compute mean
        mean = sum_rgb / n_pixels
        std = torch.sqrt((sum_sq_rgb / n_pixels) - (mean ** 2))

        return mean, std
    
    def extract_ground_truth(self):
        ground_truth = {}
        for el in self.folder_names:

            pose_file = os.path.join(self.dataset_root, f"{el:02d}_gt.yml")

            with open(pose_file, 'r') as f:
                pose_data = yaml.load(f, Loader=yaml.CLoader)

            keys_to_extract = ['cam_t_m2c', 'cam_R_m2c', 'quaternion', 'obj_bb', 'obj_id']
            extracted_data = {}
            
            for key, value in pose_data.items():
                entry = value[0] # get first object of image, entry is a dictionary

                # Extract desidered key
                extracted = {k: entry[k] for k in keys_to_extract if k in entry}

                # store in extracted_data
                extracted_data[key] = extracted # store image_id (int) and extracted value
            
            # store for each class all the extracted data
            ground_truth[el] = extracted_data

        return ground_truth
    
    def load_config(self):
        """Load YAML configuration files for object info for a specific folder."""
        objects_info_path = os.path.join(self.dataset_root, 'models', f"models_info.yml")

        with open(objects_info_path, 'r') as f:
            objects_info = yaml.load(f, Loader=yaml.CLoader)

        return objects_info

    #Define here some useful functions to access the data
    def load_image(self, img_path):
        """Load an RGB image."""
        img = Image.open(img_path).convert("RGB")
        return self.transform_img(img)

    def load_cropped_image(self, img_path, bbox):
        """Load an RGB image, crop."""
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = bbox
        cropped_img = img.crop((x, y, x+w, y+h)) # give as input the coordinates for left, top, right, bottom
        return self.transform_crop(cropped_img)

    def load_depth(self, depth_path):
        """Load a depth image."""
        return cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    def load_mask(self, path):
        """Load mask"""
        # load in grayscale mode
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    def depth_to_pointcloud(self, masked_depth):
        """
        Convert pixel coordinates + depth to 3D metric coordinates using intrinsic parameters.
        Returns x, y, z as Nx3 array.

        Convert pixel coordinate + depth to 3D metric coordinates using camera intrinsics

        Convertion equation:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

        where (u,v) are pixel coordinates and Z depth
        """
        fx, fy, cx, cy = self.camera_intrinsics
        height, width = masked_depth.shape
        valid_mask = masked_depth > 0

        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = masked_depth[valid_mask].astype(np.float32) / 1000.0

        # Ensure z_valid is not zero or very small to avoid division by zero
        valid_depth_mask = z_valid > 1e-6
        u_valid = u_valid[valid_depth_mask]
        v_valid = v_valid[valid_depth_mask]
        z_valid = z_valid[valid_depth_mask]

        x_meters = (u_valid - cx) * z_valid / fx
        y_meters = (v_valid - cy) * z_valid / fy
        z_meters = z_valid

        pointcloud = np.stack([x_meters, y_meters, z_meters], axis=-1).reshape(-1, 3)  # Shape: (N, 3)
        return pointcloud

    def load_6d_pose(self, folder_id: int = None, sample_id: int = None):
        """Load the 6D pose (translation and rotation) for the object in this sample."""
        pose = self.ground_truths[folder_id][sample_id]
        
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32)/1000.0  # [3] ---> (x,y,z) in meters
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)  # [3x3] ---> rotation matrix
        quaternion = np.array(pose['quaternion'], dtype=np.float32)  # [4] ---> quaternion
        bbox_base = np.array(pose['obj_bb'], dtype=np.float32) # [4] ---> x_min, y_min, width, height
        # bbox is top left corner and width and height info, YOLO needs center coordinates and width and height
        obj_id = np.array(pose['obj_id'], dtype=np.float32) # [1] ---> label
        
        cropped_img = self.load_cropped_image(os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "rgb", f"{sample_id:04d}.png"), bbox_base)

        # Compute initial center
        x_min, y_min, width, height = np.array(pose['obj_bb'], dtype=np.float32)
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # Clip center to image bounds and adjust width/height accordingly
        if x_center < 0:
            width += 2 * x_center  # x_center is negative, subtract its absolute value * 2 from width
            x_center = 0
        elif x_center > IMG_WIDTH:
            width -= 2 * (x_center - IMG_WIDTH)
            x_center = IMG_WIDTH

        if y_center < 0:
            height += 2 * y_center
            y_center = 0
        elif y_center > IMG_HEIGHT:
            height -= 2 * (y_center - IMG_HEIGHT)
            y_center = IMG_HEIGHT

        # Ensure width and height are not negative
        # This is when bounding box is completely outside image (it should never happen)
        width = max(0, width)
        height = max(0, height)
        # store coordinates of the center and width and height of the bounding box normalized to the
        # image width=640 pixels and height=480 pixels
        bbox_YOLO = np.array([x_center/IMG_WIDTH, y_center/IMG_HEIGHT, width/IMG_WIDTH, height/IMG_HEIGHT], dtype=np.float32)

        return cropped_img, translation, rotation, quaternion, bbox_base, obj_id, bbox_YOLO

    def __len__(self):
        #Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        #Load a dataset sample.
        folder_id, sample_id = self.samples[idx] # both are integer

        # Load the correct camera intrinsics and object info for this folder
        # objects_info = self.load_config()

        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        # depth_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"depth/{sample_id:04d}.png")
        # mask_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"mask/{sample_id:04d}.png")

        img = self.load_image(img_path)
        # depth = self.load_depth(depth_path)
        # mask = self.load_mask(mask_path)

        # mask_binary = mask != 0 # take non zero values
        # masked_depth = np.where(mask_binary, depth, 0) # apply binary mask to a depth image

        # pointcloud = self.depth_to_pointcloud(masked_depth)

        cropped_img, translation, rotation, quaternion, bbox_base, obj_id, bbox_YOLO = self.load_6d_pose(folder_id, sample_id)

        #Dictionary with all the data
        return {
            "rgb": img,
            "cropped_img": cropped_img.to(self.device),
            # "depth": torch.tensor(depth, dtype=torch.float32).to(self.device),
            # "pointcloud": torch.tensor(pointcloud, dtype=torch.float32).to(self.device),
            # "camera_intrinsics": self.camera_intrinsics,
            # "objects_info": objects_info,
            "translation": torch.tensor(translation).to(self.device),
            "rotation": torch.tensor(rotation).to(self.device),
            "quaternion": torch.tensor(quaternion).to(self.device),
            "bbox_base": torch.tensor(bbox_base).to(self.device),
            "bbox_YOLO": torch.tensor(bbox_YOLO).to(self.device),
            "obj_id": torch.tensor(obj_id).to(self.device),
            "sample_id": torch.tensor(self.samples[idx]).to(self.device)
        }