import torch
import yaml
from tqdm import tqdm
from models.PoseLossExtension import PoseLossExtension
from utils.models_point_loader import load_models_points_add

class ADDMetricExtension:
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
        # meters
        self.models_points_dict = load_models_points_add(self.models_3D_dir, class_names=class_names, device=device)
        # meters
        self.dict_diameters = self.load_diameters()
        self.experiment = experiment
        self.config = config
        self.criterion = PoseLossExtension()
    
    def load_diameters(self):
        dict_diameters = {}
        with open('./datasets/linemod/DenseFusion/Linemod_preprocessed/models/models_info.yml', 'r') as f:
            fl = yaml.load(f, Loader=yaml.CLoader)

        for k,v in fl.items():
            dict_diameters[f'{k:02d}'] = v['diameter']/1000.0 # string, diameters
        return dict_diameters

    def compute_add(self, pred_pose, gt_pose, object_id, threshold):
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

        # take 3D model points
        if object_id not in self.models_points_dict:
            print(f"Object with object id {object_id} is not present in models_point_dict")
            return torch.nan, False

        model_points = self.models_points_dict[object_id]  # Shape: (N, 3)

        # ensure proper shapes
        if pred_rot.shape != (3, 3) or gt_rot.shape != (3, 3):
            print(f"Invalid shape for rotation in object {object_id}")
            return torch.nan, False

        try:
            # transform points with the predicted pose
            pred_points = torch.matmul(model_points, pred_rot.T) + pred_trans.reshape(1, 3)

            # transform points with the gt pose
            gt_points = torch.matmul(model_points, gt_rot.T) + gt_trans.reshape(1, 3)

            # check for NaN in transformed points
            if torch.any(torch.isnan(pred_points)) or torch.any(torch.isnan(gt_points)):
                print(f"NaN values found in transformed points in object {object_id}")
                return torch.nan, False

            # compute ADD
            if object_id in self.symmetric_objects:
                # for symmetric objects, compute ADD-S
                distances = []
                for pred_point in pred_points:
                    diffs = gt_points - pred_point.reshape(1, 3)
                    point_distances = torch.linalg.norm(diffs, axis=1)
                    min_dist = torch.min(point_distances)
                    distances.append(min_dist)
                distances = torch.tensor(distances)
            else:
                # for non-symmetric objects, compute standard ADD
                distances = torch.linalg.norm(pred_points - gt_points, axis=1)

            avg_distance = torch.mean(distances)

            # check if the prediction is correct
            is_correct = 1.0 if avg_distance < threshold  else 0.0

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
                gt_trans = batch['translation'].to(self.device)
                gt_rot = batch['rotation'].to(self.device) # rotation matrix
                object_ids = batch['obj_id'].to(self.device)

                # predict
                try:
                    # rotation is quaternion
                    pixel_rotations_norm, pixel_translations, pixel_confidences = self.model(batch)
                    loss, r, t= self.criterion(pixel_rotations_norm, pixel_translations, pixel_confidences, gt_trans, gt_rot, object_ids)

                except Exception as e:
                    raise ValueError(f"Error during model prediction: {e}")

                # compute ADD for each object
                for i in range(len(object_ids)):
                    obj_id_int = int(object_ids[i])
                    obj_id = f"{obj_id_int:02d}"

                    if obj_id in self.models_points_dict:
                        pred_pose = (t[i], r[i])
                        gt_pose = (gt_trans[i], gt_rot[i])

                        distance, is_correct = self.compute_add(
                            pred_pose, gt_pose, obj_id, threshold=0.1*self.dict_diameters[obj_id]
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

        # compute overall metrics
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

        # print per-object results
        print("\nPer-object results:")
        for obj_id, obj_results in results.items():
            obj_add = torch.mean(torch.tensor(obj_results['distances']))
            obj_acc = torch.mean(torch.tensor(obj_results['correct']))
            num_samples = len(obj_results['distances'])
            print(f"Object {obj_id}: ADD={obj_add:.4f}, Acc={obj_acc:.4f}, Samples={num_samples}")

        return overall_add, overall_accuracy, results