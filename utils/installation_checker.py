import torch_geometric
import torch

def check_torch_geometric():
    # check if everything works
    try:
        from torch_geometric.nn.pool import fps
        print("PyTorch Geometric correctly installed")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"Error: {e}")
        print("Check if you have restarted runtime after installation")