import torch

def set_device():
    if torch.cuda.is_available():
        print("Cuda")
        device = torch.device("cuda")
    else:
        print("Use CPU")
        device = torch.device("cpu")
    return device