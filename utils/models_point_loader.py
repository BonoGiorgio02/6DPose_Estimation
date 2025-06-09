import os
import torch
import trimesh
from torch_geometric.nn.pool import fps

def load_models_points(models_dir, class_names, device=torch.device("cpu")):
    """Load the 3D model points (vertices) for the LINEMOD dataset in a dictionary {class_name: points}.

    Args:
        models_dir (_type_): path of the .ply files

    Returns:
        model_points_dict: dictionary of model points for each object class
    """
    model_points_dict = {}

    for obj_id in class_names:
        model_path = os.path.join(models_dir, f'obj_{obj_id}.ply')
        if os.path.exists(model_path):
            try:
                # load 3D model
                mesh = trimesh.load(model_path)

                # extract points from surface or use vertices
                if hasattr(mesh, 'vertices') and mesh.vertices is not None:
                    points = torch.tensor(mesh.vertices/1000.0, dtype=torch.float32).to(device)
                    sample_points = fps(points, None, ratio=301/points.size(0), random_start=False)[:300]
                    points = points[sample_points]
                else:
                    continue

                # check for NaN or infinite values
                if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
                    # remove NaN/Inf points
                    # torch.any(torch.isnan(points), dim=1) for each row of points, if row has at least one NaN set boolean
                    # select only rows containing values
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

def load_models_points_add(models_dir, class_names, device=torch.device("cpu")):
    """Load the 3D model points (vertices) for the LINEMOD dataset in a dictionary {class_name: points}.

    Args:
        models_dir (_type_): path of the .ply files

    Returns:
        model_points_dict: dictionary of model points for each object class
    """
    model_points_dict = {}

    for obj_id in class_names:
        model_path = os.path.join(models_dir, f'obj_{obj_id}.ply')
        if os.path.exists(model_path):
            try:
                # load 3D model
                mesh = trimesh.load(model_path)

                # extract points from surface or use vertices
                if hasattr(mesh, 'vertices') and mesh.vertices is not None:
                    points = torch.tensor(mesh.vertices/1000.0, dtype=torch.float32).to(device)
                else:
                    continue

                # check for NaN or infinite values
                if torch.any(torch.isnan(points)) or torch.any(torch.isinf(points)):
                    # remove NaN/Inf points
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