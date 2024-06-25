import torch
from poison.poison_agent import PoisonAgent
from poison.watermark import Watermark

from config import cfg


class BadnetAgent(PoisonAgent):
    def __init__(self, poison_percent, poison_ratio, target_class, mark):
        super().__init__(poison_percent=poison_percent, poison_ratio=poison_ratio, target_class=target_class)
        self.mark = mark or Watermark(mark_path=cfg['mark_path'], data_shape=cfg['data_shape'], mark_width_offset=cfg['mark_width_offset'])
        
    def add_trigger(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor.
        """
        return self.mark.add_mark(x, **kwargs)