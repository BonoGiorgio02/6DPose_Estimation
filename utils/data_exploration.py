import os
import yaml
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple
import itertools
from data.CustomDatasetPose import IMG_WIDTH, IMG_HEIGHT

def load_image(label: int, object: int):
    '''
        Starting from 6DPose_Estimation plot image given label and objectId
    '''
    img_path = f"./datasets/linemod/DenseFusion/Linemod_preprocessed/data/{label:02d}/rgb/{object:04d}.png"
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    plt.show()

def load_depth_image(path: str = None):
    img = Image.open(path)
    plt.imshow(img)
    plt.show()
    return img

def extract_cam_K_from_info(info):
    """
        Given dictionary containing data of yaml file of a given label, return list of all camera intrinsics
        Input:
            info: it is a dictionary with keys '0', '1' and so on, each of the values is a dictionary containing key 'cam_K'
        Output:
            cam_K_list: list of camera intrinsics
    """
    cam_K_list = []
    for key, val in info.items():
        if "cam_K" in val:
            cam_K_list.append(val["cam_K"])
    return cam_K_list

def check_cam_K_equal(root_path: str = None):
    """
        Check if all info.yaml in the path contain the same cam_K
    """
    all_cam_Ks = [] # list of all found camera_intrinsics

    # for each label
    for obj_class in sorted(os.listdir(root_path)):
        class_path = os.path.join(root_path, obj_class)
        # if path of class is not a directory do nothing
        if not os.path.isdir(class_path):
            continue

        info_path = os.path.join(class_path, "info.yml")
        if not os.path.exists(info_path):
            continue

        # if file exists
        with open(info_path, 'r') as f:
            info = yaml.safe_load(f)

        cam_Ks = extract_cam_K_from_info(info) # get list of all camera intrinsics
        if len(cam_Ks) == 0:
            print(f"[Warning] No cam_K found in {obj_class}/info.yml")
            continue

        # Extend the global list
        all_cam_Ks.extend(cam_Ks)

    # check if all camera intrinsics are equal
    if len(all_cam_Ks) == 0:
        print("No cam_K found in any file.")
    else:
        first_cam_K = all_cam_Ks[0]
        all_equal = all(cam_K == first_cam_K for cam_K in all_cam_Ks)

        if all_equal:
            print("All cam_K are equal in any info.yml")
        else:
            print("There are different cam_K in the files info.yml")
    
        return first_cam_K
    
def get_class_names():
    """Get class names"""

    folder_names = []
    for folder_id in range(1, 16):  # Assuming folders are named 01 to 15
        folder_path = os.path.join('./datasets/linemod/DenseFusion/Linemod_preprocessed/data', f"{folder_id:02d}", "rgb")
        if os.path.exists(folder_path):
            folder_names.append(folder_id)
    return folder_names

def load_dataset_distribution(counter_df, index_dict, number_classes):
    # plot distribution of labels in training, validation and test set
    fig, axes = plt.subplots(1,3,figsize=(15,6),sharey=True)
    for index, column in enumerate(counter_df.columns):
        axes[index].barh([str(el) for el in index_dict.keys()], counter_df[column],color="orange", edgecolor='gray')
        axes[index].set_title(column.capitalize())
        # add line that represents the uniform distribution of the labels
        axes[index].axvline(x=1/number_classes, color="blue")
        axes[index].text(x=1/number_classes,y=-0.5,s=f"{1/number_classes: .5f}", color="blue")

    fig.supxlabel("Frequency")
    fig.supylabel("Labels")
    plt.subplots_adjust(left=0.07, wspace=0.1)
    plt.suptitle("Labels Distribution over the Training, Validation and Test sets")
    plt.savefig("./images/YOLO_dataset_distribution.png")
    plt.show()

def load_depth_patch(path: str = None, folder: str = None, imageId: str = None, image=None):
    # Plot image with bounding box

    # Load the ground truth poses from the gt.yml file
    with open(f"{path}/datasets/linemod/DenseFusion/Linemod_preprocessed/data/{folder}/gt.yml", 'r') as f:
        pose_data = yaml.load(f, Loader=yaml.FullLoader)
    pose = pose_data[int(imageId)][0] # access image imageId (start counting from 0) and get first element (in case of multiple objects)

    bbox = np.array(pose['obj_bb'], dtype=np.float32) #[4]
    obj_id = np.array(pose['obj_id'], dtype=np.float32) #[1]

    fig, ax = plt.subplots()
    ax.imshow(image)

    # Create a rectangle patch
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),  # (x, y)
        bbox[2],             # width
        bbox[3],             # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Optionally add object ID label (write a bit above the top left corner)
    ax.text(bbox[0], bbox[1] - 10, f'ID: {int(obj_id)}', color='yellow', fontsize=12, backgroundcolor='black')

    plt.axis('off')
    plt.show()

def plot_batch_data(train_loader, val_loader, test_loader):
    # Get only the first 1 batch
    # train_subset_num_batches = 1
    # val_subset_num_batches = 1
    # test_subset_num_batches = 1
    # train_subset = list(itertools.islice(train_loader, train_subset_num_batches))
    # val_subset = list(itertools.islice(val_loader, val_subset_num_batches))
    # test_subset = list(itertools.islice(test_loader, test_subset_num_batches))

    # Get one batch from the train loader (there are batch_size images)
    batch = next(iter(train_loader)) # it uses load_6d_pose, so one pose per object

    # Extract relevant data
    rgb_images = batch["rgb"]         # (B, 3, H, W)
    bboxes = batch["bbox_YOLO"]            # (B, 4) in pixel coords: x_center, y_center, width, height
    obj_ids = batch["obj_id"]         # (B,)

    # Convert to numpy and rearrange channels
    rgb_images = rgb_images.permute(0, 2, 3, 1).numpy()  # (B, H, W, 3)
    bboxes = bboxes.to(torch.device("cpu")).numpy()
    obj_ids = obj_ids.to(torch.device("cpu")).numpy()

    # Plot settings
    batch_size = rgb_images.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(batch_size):
        ax = axes[i]
        img = rgb_images[i]
        # each element is [x_center/IMG_WIDTH, y_center/IMG_HEIGHT, width/IMG_WIDTH, height/IMG_HEIGHT]
        x_center, y_center, width, height = bboxes[i]
        # remove normalization
        x_center = x_center*IMG_WIDTH
        y_center = y_center*IMG_HEIGHT
        width = width*IMG_WIDTH
        height = height*IMG_HEIGHT
        x_min = x_center-(width/2)
        y_min = y_center-(height/2)
        obj_id = obj_ids[i]

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sample {i}")

        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min),   # (x_min, y_min)
            width,              # width
            height,              # height
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add object ID as label
        ax.text(
            x_min,
            y_min - 10,
            f'ID: {int(obj_id)}',
            color='yellow',
            fontsize=10,
            backgroundcolor='black'
        )

    # Hide unused axes if batch_size < cols * rows
    for j in range(batch_size, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()