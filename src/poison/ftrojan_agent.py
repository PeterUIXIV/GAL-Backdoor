import numpy as np
import torch
from poison.ftrojan import poison_frequency
from poison.poison_agent import PoisonAgent


class FtrojanAgent(PoisonAgent):
    def __init__(self, poison_percent, poison_ratio, target_class):
        super.__init__(poison_percent=poison_percent, poison_ratio=poison_ratio, target_class=target_class)
        
    def add_trigger(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor.
        """
        if x.is_cuda:
            x = x.cpu()
        np_array = x.numpy()
        np_array = np.transpose(np_array , (0, 2, 3, 1))
        poisoned = poison_frequency(np_array, **kwargs)
        poisoned = np.transpose(poisoned, (0, 3, 1, 2))
        poisoned = torch.from_numpy(poisoned)
        return poisoned
