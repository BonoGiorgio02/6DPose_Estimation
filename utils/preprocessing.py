import os
import yaml
import shutil
import quaternion
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def copy_gt_file(file_names: list = None):
    for el in file_names:
        shutil.copy(f"./datasets/linemod/DenseFusion/Linemod_preprocessed/data/{el:02d}/gt.yml", f"./datasets/linemod/DenseFusion/Linemod_preprocessed/{el:02d}_gt.yml")

def change_02gt(path=None):
    with open(path, 'r') as f:
        pose_data = yaml.load(f, Loader=yaml.CLoader) # use CLoader since it is faster

    filtered_data = {}
    for key, value in pose_data.items():
        # for each object in each image, if object has label 2, add it to the list
        filtered_frame = [obj for obj in value if int(obj['obj_id']) == 2]
        if filtered_frame:
            # if not empty, add it to the dictionary
            filtered_data[key] = filtered_frame

    # Custom Dumper for maintaining original format
    class OriginalFormatDumper(yaml.Dumper):
        """
            yaml.Dumper is a class
        """
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)  # Force correct identation

    # store result in file
    with open(path, 'w') as f:
        yaml.dump(
            filtered_data,
            f,
            sort_keys=True,
            default_flow_style=None, # None = auto (use flow-style for internal lists)
            Dumper=OriginalFormatDumper,
            width=float("inf"),
            indent=2, # standard identation
        )

def quaternion_gt(input_path=None):
    """
        Add quaternion to ground truth file
    """
    # for each ground truth file
    for gt in os.scandir(input_path):
        if not gt.is_dir() and gt.name.endswith(('.yaml', '.yml')):

            with open(gt, 'r') as f:
                pose_data = yaml.load(f, Loader=yaml.CLoader)
                modified_data = {}

                for key,value in pose_data.items():
                    modified_poses = []

                    for pose in value:
                        quat_obj = quaternion.from_rotation_matrix(np.array(pose["cam_R_m2c"]).reshape(3,3))
                        quat_array = np.array([quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z], dtype=np.float32)
                        # convert array to tensor
                        quat = torch.tensor(quat_array)

                        # normalize quaternion
                        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
                        # store the quaternion components as a list in the dictionary
                        pose["quaternion"]= quat.tolist() # Use .item() to get scalar values
                        modified_poses.append(pose)
                        
                    modified_data[key] = modified_poses

                output_file = os.path.join(input_path, gt.name)

                with open(output_file, 'w') as f_output:
                    yaml.dump(
                        modified_data,
                        f_output,
                        default_flow_style=None,
                        width=float("inf"),
                        sort_keys=True
                    )

def create_YOLO_yaml(path, folder_names):
    # create a folder to contain the dataset for YOLO model
    os.makedirs(f"{path}/datasets/linemod/YOLO/datasets", exist_ok=True)

    # count number of distinct classes
    number_classes = len(folder_names)
    class_names = [f"{el:02d}" for el in folder_names]

    # get string of all class names
    class_names.sort() # sort the names
    names = "["
    for index, el in enumerate(class_names):
        # if last element don't add comma
        if index == number_classes-1:
            names += f"'{str(el)}'"
        else:
            names += f"'{str(el)}',"
    names += "]"

    # create data.yaml (as class names use ids of the folder)
    content = f"""train: ./train/images\nval: ./val/images\ntest: ./test/images\n\nnc: {number_classes}\nnames: {names}"""
    # write to file
    with open(f"{path}/datasets/linemod/YOLO/datasets/data.yaml", "w") as fout:
        fout.write(content)
    fout.close()

    return number_classes, class_names

def create_dataset_YOLO(number_classes, train_samples, validation_samples, test_samples, index_dict, path, train_dataset):
    # create images and labels
    # dataset = [train_samples, validation_samples, test_samples]
    folder_names = ["train", "val", "test"]

    # count also the number of instances of each class
    # classes = range(0, number_classes)
    counter_df = pd.DataFrame()
    for idx in range(3):
        if idx == 0:
            dataset = train_samples
        elif idx == 1:
            dataset = validation_samples
        else:
            dataset = test_samples
        print(f"------------------------------{folder_names[idx].upper()}------------------------------")
        os.makedirs(f"{path}/datasets/linemod/YOLO/datasets/{folder_names[idx]}/images", exist_ok=True)
        os.makedirs(f"{path}/datasets/linemod/YOLO/datasets/{folder_names[idx]}/labels", exist_ok=True)
        classCount = {label_object: 0 for label_object in index_dict.keys()} # initialize dictionary for counting
        total = 0 # used to normalize count
        for el in tqdm(dataset, desc="Moving..."):
            # el is (folderId, sampleId)
            _, _, _, _, _, obj_id, bbox = train_dataset.load_6d_pose(el[0], el[1])
            # copy image into the new folder
            # avoid overwriting the files, so concat also the name of the folderId to the destination file
            shutil.copy(f"{path}/datasets/linemod/DenseFusion/Linemod_preprocessed/data/{el[0]:02d}/rgb/{el[1]:04d}.png", f"{path}/datasets/linemod/YOLO/datasets/{folder_names[idx]}/images/{el[0]:02d}_{el[1]:04d}.png")
            # create label file with the same name as the image
            with open(f"{path}/datasets/linemod/YOLO/datasets/{folder_names[idx]}/labels/{el[0]:02d}_{el[1]:04d}.txt", "w") as fout:
                # bbox is a list of values in the form of [x_center, y_center, width, height] and obj_id a list of class labels
                # where each label is in the format 01-15
                classCount[int(obj_id)] += 1
                total += 1
                content = f"{index_dict[int(obj_id)]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                fout.write(content)
            fout.close()
        
        # store in the dataframe
        values = pd.array(list(classCount.values()))/total
        counter_df[folder_names[idx]] = values.copy()