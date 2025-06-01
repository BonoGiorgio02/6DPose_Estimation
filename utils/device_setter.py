import torch

def set_device():
    if torch.cuda.is_available():
        print("Cuda")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Cuda not available, use mps")
        device = torch.device("mps")
    else:
        print("Use CPU")
        device = torch.device("cpu")
    return device