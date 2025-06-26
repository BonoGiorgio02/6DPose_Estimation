[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

# 6DPose_Estimation
This project addresses the task of 6D object pose estimation on the LINEMOD preprocessed dataset.

Moreover, the proposed architecture is trained to detect the pose of cones in a simulator of the student autonomous driving team of Politecnico of Turin Squadra Corse driverless.

## How to run the code
To run the code, execute the notebook.

If you want to run with Colab:
- download repo and store in on Drive
- extract it
- open the notebook
- connect to Drive (set ```MOUNT_DRIVE``` to ```True```)
- follow the notebook step by step

The repository is structured such that:
- **checkpoints** contains saved models
- **data** contains custom dataloader and dataset classes
- **datasets** contains datasets
- **images** contains images
- **models** contains model architectures, metrics
- **utils** contains multiple functions (init, data exploration, plot)
- notebook and associated training logic
- **requirements** contains necessary packages


### Results obtained in LINEMOD dataset

<p align="center">
  <img src="images/object_05.png" alt="Linemod_1" width="350" style="margin-right: 10px;">
  <img src="images/object_06.png" alt="Linemod_2" width=350">
</p>

<p align="center">
  <b>Figure 1:</b> Object 05. &nbsp;&nbsp;
  <b>Figure 2:</b> Object 06.
</p>

<p align="center">
  <img src="images/object_08.png" alt="Linemod_3" width="350" style="margin-right: 10px;">
  <img src="images/object_09.png" alt="Linemod_4" width=350">
</p>

<p align="center">
  <b>Figure 3:</b> Object 08. &nbsp;&nbsp;
  <b>Figure 4:</b> Object 09.
</p>

<p align="center">
  <img src="images/object_13.png" alt="Linemod_5" width="350" style="margin-right: 10px;">
  <img src="images/object_14.png" alt="Linemod_6" width=350">
</p>

<p align="center">
  <b>Figure 5:</b> Object 13. &nbsp;&nbsp;
  <b>Figure 6:</b> Object 14.
</p>

#### Evaluation results

**Overall:**

| Metric    | Value  |
| --------- | ------ |
| ADD Score | 0.0138 |
| Accuracy  | 80.03% |

**Results by object:**

| Object | ADD Score | Accuracy | Number of object |
| ------ | --------- | -------- | ---------------- |
| 06     | 0.0121    | 79.66%   | 177              |
| 11     | 0.0056    | 100.0%   | 183              |
| 15     | 0.0160    | 81.52%   | 184              |
| 01     | 0.0124    | 51.08%   | 186              |
| 04     | 0.0160    | 70.00%   | 180              |
| 10     | 0.0058    | 100.0%   | 188              |
| 09     | 0.0119    | 52.13%   | 188              |
| 08     | 0.0178    | 90.45%   | 178              |
| 05     | 0.0168    | 81.01%   | 179              |
| 14     | 0.0170    | 91.85%   | 184              |
| 02     | 0.0166    | 89.56%   | 182              |
| 12     | 0.0150    | 59.68%   | 186              |
| 13     | 0.0172    | 95.38%   | 173              |

---


### Results obtained in autonomous driving dataset

<p align="center">
  <img src="images/vimba_038_image_sync.png" alt="Left camera frame" width="350" style="margin-right: 10px;">
  <img src="images/vimba_039_image_sync.png" alt="Right camera frame" width=350">
</p>

<p align="center">
  <b>Figure 1:</b> Left camera frame. &nbsp;&nbsp;
  <b>Figure 2:</b> Right camera frame.
</p>

<p align="center">
  <img src="images/vimba_038_image_sync_YOLO.jpg" alt="Left camera frame processed by YOLO" width="350" style="margin-right: 10px;">
  <img src="images/vimba_039_image_sync_YOLO.jpg" alt="Right camera frame processed by YOLO" width=350">
</p>

<p align="center">
  <b>Figure 3:</b> Left camera frame processed by YOLO. &nbsp;&nbsp;
  <b>Figure 4:</b> Right camera frame processed by YOLO.
</p>

<p align="center">
  <img src="images/best_cone.jpg" alt="Left camera frame processed by YOLO" style="margin-right: 10px;">
  <img src="images/best_cone_039.jpg" alt="Right camera frame processed by YOLO">
</p>

<p align="center">
  <b>Figure 5:</b> Cropped image of left cone. &nbsp;&nbsp;
  <b>Figure 6:</b> Cropped image of right cone.
</p>

<p align="center">
  <img src="images/pointcloud_projection_result.png" alt="LiDAR pointcloud of the cone projected in left image" width="350" style="margin-right: 10px;">
  <img src="images/pointcloud_projection_result_039.png" alt="LiDAR pointcloud of the cone projected in right image" width=350">
</p>

<p align="center">
  <b>Figure 7:</b> LiDAR pointcloud of the left cone. &nbsp;&nbsp;
  <b>Figure 8:</b> LiDAR pointcloud of the right cone.
</p>

<p align="center">
  <img src="images/synchronized_vimba_038_image_sync_pose_estimation_1.png" alt="LiDAR pointcloud of the cone projected in left image" width="350" style="margin-right: 10px;">
  <img src="images/synchronized_vimba_039_image_sync_pose_estimation_1.png" alt="LiDAR pointcloud of the cone projected in right image" width=350">
</p>

<p align="center">
  <b>Figure 9:</b> 6D pose estimation of the left cone. &nbsp;&nbsp;
  <b>Figure 10:</b> 6D pose estimation of the right cone.
</p>
