import numpy as np
import torch
from poison.ftrojan import poison_frequency
from poison.poison_agent import PoisonAgent
from utils import numpy_to_torch, show_images, show_images_with_labels_and_values, torch_to_numpy


class FtrojanAgent(PoisonAgent):
    def __init__(self, poison_percent, poison_ratio, target_class):
        super().__init__(poison_percent=poison_percent, poison_ratio=poison_ratio, target_class=target_class)
        
    def add_trigger(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor.
        """
        np_array = torch_to_numpy(x)
        # show_images(np_array, 3, 3)
        poisoned = poison_frequency(np_array, **kwargs)
        # show_images(poisoned, 3, 3)
        poisoned = numpy_to_torch(poisoned)
        return poisoned
