import torch
import torch.nn.functional as F
from torch import nn
import os
import trimesh
from torch_geometric.nn.pool import fps

class PoseLossAlternative(nn.Module):
    """Loss function combines translation and rotation
        returns a weighted sum of the translation and rotation losses.
    """

    def __init__(self, alpha=1.0, beta=1.0):
        """Initialize alpha and beta values.

        Args:
            alpha (float, optional): Weight (importance) to give to translation. Defaults to 1.0.
            beta (float, optional): Weight (importance) to give to rotation. Defaults to 1.0.
        """
        super(PoseLossAlternative, self).__init__()
        self.alpha = alpha  # weight for translation loss
        self.beta = beta    # weight for rotation loss

        self.models_dir = "./datasets/linemod/DenseFusion/Linemod_preprocessed/models"
        self.models_dict = self.load_models_points(self.models_dir)
    
    def forward(self, pred_trans, pred_rot, gt_trans, gt_rot, obj_id=None):
        """Initialize alpha and beta values.

          Args:
              pred_trans (matrix , optional): Predicted translation.
              pred_rot (matrix, optional): Predicted rotation.
              gt_trans (matrix, optional): Ground truth translation.
              gt_rot (matrix, optional): Ground truth rotation.
        """
        batch_size = pred_trans.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            obj = obj_id[i].item()
            model_points = self.models_dict[f"{int(obj):02d}"]

            pred_q = F.normalize(pred_rot[i], dim=0)
            gt_q = F.normalize(gt_rot[i], dim=0)

            w,x,y,z = pred_q
            pred_R = torch.tensor([
                [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
            ], dtype=torch.float32, device=pred_q.device)

            w,x,y,z = gt_q
            gt_R = torch.tensor([
                [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
                [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
            ], dtype=torch.float32, device=gt_q.device)

            pred_points = torch.matmul(model_points, pred_R.T) + pred_trans[i]
            gt_points   = torch.matmul(model_points, gt_R.T)   + gt_trans[i]

            loss = F.mse_loss(pred_points, gt_points)
            total_loss += loss

        avg_loss = total_loss / batch_size
        return avg_loss
    
    def load_models_points(self, models_dir):
        """Load the 3D model points (vertices) for the LINEMOD dataset in a dictionary {class_name: points}.

        Args:
            models_dir (_type_): path of the .ply files

        Returns:
            model_points_dict: dictionary of model points for each object class
        """
        model_points_dict = {}
        class_names = ["01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "14", "15"]

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        for obj_id in class_names:
            model_path = os.path.join(models_dir, f'obj_{obj_id}.ply')
            if os.path.exists(model_path):
                try:
                    # Carica il modello 3D
                    mesh = trimesh.load(model_path)

                    # Estrai punti dalla superficie o usa vertices
                    if hasattr(mesh, 'vertices') and mesh.vertices is not None:
                        points = torch.tensor(mesh.vertices/1000.0, dtype=torch.float32).to(device)
                        sample_points = fps(points, None, ratio=1001/points.size(0), random_start=False)[:1000]
                        points = points[sample_points]
                    else:
                        continue

                    # Check for NaN or infinite values
                    if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
                        # Remove NaN/Inf points
                        valid_mask = ~(torch.any(torch.isnan(points), dim=1) | torch.any(torch.isinf(points), dim=1))
                        points = points[valid_mask].to(device)


                    if len(points) == 0:
                        print(f"No valid points found for object {obj_id}")
                        continue
                    model_points_dict[obj_id] = points

                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
                    continue

        return model_points_dict