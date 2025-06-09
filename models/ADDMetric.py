import torch
import os
import trimesh
import yaml
import numpy as np
import quaternion
from tqdm import tqdm

class ADDMetric:
    """ ADD metric to evaluate prediction of the NN comparing it with the CAD model.
        returns the average distance between the predicted pose and the ground truth pose.
    """

    def __init__(self, model, class_names, test_loader, models_3D_dir, symmetric_objects=None, device=torch.device("cpu"), experiment=None, config=None):
        """
        Args:
            model (torch model): model to evaluate
            class_names: name of labels
            test_loader (dataloader): test loader
            models_3D_dir (str): path to the directory containing 3D models
            symmetric_objects (list, optional): list of object_id that are symmetric. Defaults to None.
            device (str, optional): device to use. Defaults to 'cpu'.
        """
        self.model = model
        self.model.eval()
        self.class_names = class_names
        self.test_loader = test_loader
        self.models_3D_dir = models_3D_dir
        self.symmetric_objects = symmetric_objects or []
        self.device = device
        self.models_points_dict = self.load_models_points(self.models_3D_dir)
        # read the diameters for each object
        self.objects_info = self.load_objects_models_info()
        self.experiment = experiment
        self.config = config

    def load_models_points(self, models_dir):
        """Load the 3D model points (vertices) for the LINEMOD dataset in a dictionary {class_name: points}.

        Args:
            models_dir (_type_): path of the .ply files

        Returns:
            model_points_dict: dictionary of model points for each object class
        """
        model_points_dict = {}

        for obj_id in self.class_names:
            model_path = os.path.join(models_dir, f'obj_{obj_id}.ply')
            if os.path.exists(model_path):
                try:
                    # load 3D model
                    mesh = trimesh.load(model_path)

                    # extract points from surface or use vertices
                    if hasattr(mesh, 'vertices') and mesh.vertices is not None: # mesh has shape (N,3)
                        points = torch.tensor(mesh.vertices/1000.0, dtype=torch.float32).to(self.device)
                    else:
                        continue

                    # Check for NaN or infinite values
                    if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
                        # Remove NaN/Inf points
                        valid_mask = ~(torch.any(torch.isnan(points), dim=1) | torch.any(torch.isinf(points), dim=1))
                        points = points[valid_mask].to(self.device)

                    if len(points) == 0:
                        print(f"No valid points found for object {obj_id}")
                        continue
                    model_points_dict[f"{obj_id:02d}"] = points

                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
                    continue

        return model_points_dict
    
    def load_objects_models_info(self):
        with open(f"{self.models_3D_dir}/models_info.yml", 'r') as f:
            return yaml.load(f, Loader=yaml.CLoader)

    def compute_add(self, pred_pose, gt_pose, object_id, threshold=0.1):
        """
        Compute ADD metric for a single prediction

        Args:
            pred_pose (tuple): (translation, rotation_matrix)
            gt_pose (tuple): (translation, rotation_matrix)
            object_id (str): ID of the object
            threshold (float): threshold to consider the prediction correct
        """
        pred_trans, pred_rot = pred_pose
        gt_trans, gt_rot = gt_pose

        # Check for NaN values in poses
        if torch.any(torch.isnan(pred_trans)) or torch.any(torch.isnan(pred_rot)):
            print(f"NaN values found in predicted pose in object {object_id}")
            return torch.nan, False

        if torch.any(torch.isnan(gt_trans)) or torch.any(torch.isnan(gt_rot)):
            print(f"NaN values found in ground truth pose in object {object_id}")
            return torch.nan, False

        try:
        # Convert Torch quaternion [w, x, y, z] -> numpy.quaternion(w, x, y, z)
            pred_quat = np.quaternion(pred_rot[0].item(), pred_rot[1].item(), pred_rot[2].item(), pred_rot[3].item()) # not necessarily unit quaternion
            gt_quat = np.quaternion(gt_rot[0].item(), gt_rot[1].item(), gt_rot[2].item(), gt_rot[3].item()) # unit quaternion

            # Get rotation orthogonal matrices (3x3 numpy), matrices are orthogonal
            pred_rot = torch.tensor(quaternion.as_rotation_matrix(pred_quat), dtype=torch.float32, device=pred_rot.device)
            gt_rot = torch.tensor(quaternion.as_rotation_matrix(gt_quat), dtype=torch.float32, device=gt_rot.device)

        except Exception as e:
            print(f"Error converting quaternion to rotation matrix for object {object_id}: {e}")
            return torch.nan, False

        # Take 3D model points
        if object_id not in self.models_points_dict:
            print(f"Object with object id {object_id} is not present in models_point_dict")
            return torch.nan, False

        model_points = self.models_points_dict[object_id]  # Shape: (N, 3)

        # Ensure proper shapes
        if pred_rot.shape != (3, 3) or gt_rot.shape != (3, 3):
            print(f"Invalid shape for rotation in object {object_id}")
            return torch.nan, False

        try:
            # Transform points with the predicted pose
            # matrix product between model_points and pred_rot.T
            pred_points = torch.matmul(model_points, pred_rot.T) + pred_trans.reshape(1, 3) # Nx3 * 3x3 + 1x3, each row is a point

            # Transform points with the gt pose
            gt_points = torch.matmul(model_points, gt_rot.T) + gt_trans.reshape(1, 3)

            # Check for NaN in transformed points
            if torch.any(torch.isnan(pred_points)) or torch.any(torch.isnan(gt_points)):
                print(f"NaN values found in transformed points in object {object_id}")
                return torch.nan, False

            # Compute ADD
            if object_id in self.symmetric_objects:
                # For symmetric objects, compute ADD-S
                distances = []
                for pred_point in pred_points:
                    # compute for each ground truth point, the difference with predicted point
                    diffs = gt_points - pred_point.reshape(1, 3)
                    point_distances = torch.linalg.norm(diffs, axis=1)
                    min_dist = torch.min(point_distances)
                    distances.append(min_dist)
                distances = torch.tensor(distances)
            else:
                # For non-symmetric objects, compute standard ADD, compute norm each row
                distances = torch.linalg.norm(pred_points - gt_points, axis=1)

            avg_distance = torch.mean(distances)

            # Check if the prediction is correct
            is_correct = 1.0 if avg_distance < (threshold * self.objects_info[int(object_id)]["diameter"]/1000) else 0.0

            return avg_distance, is_correct

        except Exception as e:
            print(f"Error during ADD computation for object {object_id}")
            return torch.nan, False

    def evaluate_model_with_add(self):
        """Evaluate the model using the ADD metric."""
        self.model.eval()

        results = {}
        all_distances = []
        all_correct = []
        total_processed = 0
        total_skipped = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Evaluating ADD')):
                images = batch['cropped_img'].to(self.device)
                gt_trans = batch['translation'].to(self.device)
                gt_rot = batch['quaternion'].to(self.device)
                object_ids = batch['obj_id'].to(self.device)

                # Predict
                try:
                    pred_trans, pred_rot = self.model(images)

                except Exception as e:
                    raise ValueError(f"Error during model prediction: {e}")

                # Compute ADD for each object
                for i in range(len(images)):
                    obj_id_int = int(object_ids[i])
                    obj_id = f"{obj_id_int:02d}"

                    if obj_id in self.models_points_dict:
                        pred_pose = (pred_trans[i], pred_rot[i])
                        gt_pose = (gt_trans[i], gt_rot[i])

                        distance, is_correct = self.compute_add(
                            pred_pose, gt_pose, obj_id, threshold=0.1
                        )

                        if not torch.isnan(distance):
                            all_distances.append(distance)
                            all_correct.append(is_correct)
                            total_processed += 1

                            if obj_id not in results:
                                results[obj_id] = {'distances': [], 'correct': []}

                            results[obj_id]['distances'].append(distance)
                            results[obj_id]['correct'].append(is_correct)
                        else:
                            total_skipped += 1
                    else:
                        total_skipped += 1

        if len(all_distances) == 0:
            print("No valid objects found for evaluation.")
            return torch.nan, torch.nan, {}

        # Compute overall metrics
        all_distances = torch.tensor(all_distances)
        all_correct = torch.tensor(all_correct)
        overall_add = torch.mean(all_distances)
        overall_accuracy = torch.mean(all_correct)

        if self.experiment is not None:
            self.experiment.log_metrics({
                "test_add_score": overall_add,
                "test_accuracy": overall_accuracy,
                "total_processed": total_processed,
                "total_skipped": total_skipped
            })

        print(f"\nOverall ADD: {overall_add:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        # Print per-object results
        print("\nPer-object results:")
        for obj_id, obj_results in results.items():
            obj_add = torch.mean(torch.tensor(obj_results['distances']))
            obj_acc = torch.mean(torch.tensor(obj_results['correct']))
            num_samples = len(obj_results['distances'])
            print(f"Object {obj_id}: ADD={obj_add:.4f}, Acc={obj_acc:.4f}, Samples={num_samples}")

        return overall_add, overall_accuracy, results