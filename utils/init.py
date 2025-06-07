import torch
import random
import numpy as np

def set_device():
    if torch.cuda.is_available():
        print("Cuda")
        device = torch.device("cuda")
    else:
        print("Use CPU")
        device = torch.device("cpu")
    return device

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False