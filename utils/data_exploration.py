from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple

def load_image(label: int, object: int):
    '''
        Starting from 6D_pose_estimation plot image given label and objectId
    '''
    img_path = f"./datasets/linemod/DenseFusion/Linemod_preprocessed/data/{label:02d}/rgb/{object:04d}.png"
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    plt.show()