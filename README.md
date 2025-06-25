# 6DPose_Estimation
To run the code, execute the notebook.

If you want to run with Colab:
- download repo and store in on Drive
- extract it
- open the notebook
- connect to Drive (set ```MOUNT_DRIVE``` to ```True```)
- follow the notebook

The repository is structured such that:
- **checkpoints** contains saved models
- **data** contains custom dataloader and dataset classes
- **datasets** contains datasets
- **images** contains images
- **models** contains model architectures, metrics
- **utils** contains multiple functions (init, data exploration, plot)
- notebook and associated training logic
- **requirements** contains necessary packages

### Results obtained in autonomous driving dataset

<p align="center">
  <img src="images/vimba_038_image_sync.png" alt="Left camera frame" width="450" style="margin-right: 10px;">
  <img src="images/vimba_039_image_sync.png" alt="Right camera frame" width=450">
</p>

<p align="center">
  <b>Figure 1:</b> Left camera frame. &nbsp;&nbsp;
  <b>Figure 2:</b> Right camera frame.
</p>

<p align="center">
  <img src="images/vimba_038_image_sync_YOLO.jpg" alt="Left camera frame processed by YOLO" width="450" style="margin-right: 10px;">
  <img src="images/vimba_039_image_sync_YOLO.jpg" alt="Right camera frame processed by YOLO" width=450">
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

![Cropped image of left cone](images/best_cone.jpg)
![Cropped image of right cone](images/best_cone_039.jpg)
![LiDAR pointcloud of the cone projected in left image](images/pointcloud_projection_result.png)
![LiDAR pointcloud of the cone projected in right image](images/pointcloud_projection_result_039.png)
![6D pose estimation of the left cone](images/synchronized_vimba_038_image_sync_pose_estimation_1.png)
![6D pose estimation of the left cone](images/synchronized_vimba_039_image_sync_pose_estimation_1.png)
