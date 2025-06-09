import cv2
import torch
import trimesh
import quaternion
import numpy as np
import matplotlib.pyplot as plt

def plotPose(pathImage, translation_gt, rotation_gt, translation_pred, rotation_pred, experiment, camera_intrinsics, device=torch.device("cpu")):
    '''
        Input:
            path for image (in DenseFusion)
            ground truth translation tensor (in meters)
            ground truth rotation tensor (either matrix or quaternion)
            predicted translation tensor (in meters)
            predicted rotation tensor (either matrix or quaternion)
            experiment for logging
            device
    '''
    # read image
    image = cv2.imread(pathImage)
    transparent_image = image.copy() # copy of the image to work on a transparent image (for the second reference system)

    rotat_gt = rotation_gt.to(device).float()
    trans_gt = translation_gt.to(device).float()
    rotat_pred = rotation_pred.to(device).float()
    trans_pred = translation_pred.to(device).float()

    image_id = pathImage.split("/")[-1].split(".")[0]
    label = pathImage.split("/")[-3]
    camera_intrinsics = torch.tensor(camera_intrinsics).reshape(3, 3).to(device)

    # read 3D model
    meshModel = trimesh.load(f"./datasets/linemod/DenseFusion/Linemod_preprocessed/models/obj_{label}.ply")
    vertices = torch.tensor(meshModel.vertices / 1000, dtype=torch.float32).to(device) # it has 3 columns, for X, Y, Z, use unit of measurement of translation
    min_corner = vertices.min(dim=0).values # find for each column the smallest value
    max_corner = vertices.max(dim=0).values

    bounding_box_3d = torch.tensor([
        [min_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
    ], dtype=torch.float32).to(device)

    # if quaternion, get rotation matrix
    if rotat_gt.numel() == 4:
        rotat_gt = torch.tensor(
            quaternion.as_rotation_matrix(np.quaternion(*rotat_gt.cpu().numpy())),
            dtype=torch.float32
        ).to(device)
    else:
        rotat_gt = rotat_gt.reshape(3, 3)

    if rotat_pred.numel() == 4:
        rotat_pred = torch.tensor(
            quaternion.as_rotation_matrix(np.quaternion(*rotat_pred.cpu().numpy())),
            dtype=torch.float32
        ).to(device)
    else:
        rotat_pred = rotat_pred.reshape(3, 3)

    # build 3D axes according to object coordinate system, same unit of measurement of translation, so in meters
    axes_3d = torch.tensor([
        [0, 0, 0],      # origin, in the object coordinate system
        [0.15, 0, 0],   # how long the arrow should be in the X coordinate
        [0, 0.15, 0],   # how long the arrow should be in the Y coordinate
        [0, 0, 0.15]    # how long the arrow should be in the Z coordinate
    ], dtype=torch.float32).to(device)

    # transform the object coordinate system to the camera coordinate system
    # rotat_gt is 3x3, so axes_3d has to be transposed, then add to origin, and coordinates the translation
    axes_cam_gt = (rotat_gt @ axes_3d.T).T + trans_gt
    # bounding box
    bounding_box_3d_cam_gt = (rotat_gt @ bounding_box_3d.T).T + trans_gt
    # project 3D axes to 2D
    axes_2d_gt = (camera_intrinsics @ axes_cam_gt.T).T # camera_intrinsics is 3x3, while axes_cam_gt 4x3, axes_2d_gt 4x3
    axes_2d_gt = axes_2d_gt[:, :2] / axes_2d_gt[:, 2:3] # take first 2 columns and normalize by depth
    # bounding box
    bounding_box_2d_gt = (camera_intrinsics @ bounding_box_3d_cam_gt.T).T
    bounding_box_2d_gt = (bounding_box_2d_gt[:, :2] / bounding_box_2d_gt[:, 2:3]).int()
    # get point coordinates
    p_gt = [tuple(el.cpu().numpy()) for el in bounding_box_2d_gt]
    # define edges using two points, access with index
    edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,5), (2,6), (3,7), (4,5), (5,6), (6,7), (7,4)]
    for el in edges:
        cv2.line(image, p_gt[el[0]], p_gt[el[1]], (0,0,255), 5)

    p0_gt = tuple(axes_2d_gt[0].cpu().int().numpy())
    p1_gt = tuple(axes_2d_gt[1].cpu().int().numpy())
    p2_gt = tuple(axes_2d_gt[2].cpu().int().numpy())
    p3_gt = tuple(axes_2d_gt[3].cpu().int().numpy())

    # color is in BGR format, set tickness=2
    cv2.arrowedLine(image, p0_gt, p1_gt, (0, 0, 255), 2)
    cv2.arrowedLine(image, p0_gt, p2_gt, (0, 255, 0), 2)
    cv2.arrowedLine(image, p0_gt, p3_gt, (255, 0, 0), 2)

    # for predicted
    axes_cam_pred = (rotat_pred @ axes_3d.T).T + trans_pred
    bounding_box_3d_cam_pred = (rotat_pred @ bounding_box_3d.T).T + trans_pred

    axes_2d_pred = (camera_intrinsics @ axes_cam_pred.T).T
    axes_2d_pred = axes_2d_pred[:, :2] / axes_2d_pred[:, 2:3]
    bounding_box_2d_pred = (camera_intrinsics @ bounding_box_3d_cam_pred.T).T
    bounding_box_2d_pred = (bounding_box_2d_pred[:, :2] / bounding_box_2d_pred[:, 2:3]).int()

    p_pred = [tuple(el.cpu().numpy()) for el in bounding_box_2d_pred]
    for el in edges:
        cv2.line(image, p_pred[el[0]], p_pred[el[1]], (255, 0, 0), 5)

    p0_pred = tuple(axes_2d_pred[0].cpu().int().numpy())
    p1_pred = tuple(axes_2d_pred[1].cpu().int().numpy())
    p2_pred = tuple(axes_2d_pred[2].cpu().int().numpy())
    p3_pred = tuple(axes_2d_pred[3].cpu().int().numpy())

    cv2.arrowedLine(transparent_image, p0_pred, p1_pred, (0, 0, 255), 2)
    cv2.arrowedLine(transparent_image, p0_pred, p2_pred, (0, 255, 0), 2)
    cv2.arrowedLine(transparent_image, p0_pred, p3_pred, (255, 0, 0), 2)

    overlapImage = cv2.addWeighted(transparent_image, 0.5, image, 1, 0)
    img = cv2.cvtColor(overlapImage, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if experiment is not None:
        experiment.log_image(image_data=img, name= f"{label}_{image_id}")
    plt.title("Object Pose Estimation (prediction is transparent)")
    plt.show()