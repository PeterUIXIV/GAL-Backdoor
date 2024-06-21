from abc import ABC, abstractmethod
import math
import random
from typing import Tuple
import numpy as np
import torch

from config import cfg


class PoisonAgent(ABC):
    def __init__(self, poison_percent, poison_ratio, target_class):
        self.poison_percent = poison_percent
        self.poison_ratio = poison_ratio or self.poison_percent / (1 - self.poison_percent)
        self.target_class = target_class
        self.manipulated_ids = []
    
    @abstractmethod
    def add_trigger(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor.
        """
        pass
    
    def poison_test_dataset(self, dataset, keep_org):
        fail_count = 0
        for i, sample in enumerate(dataset['test']):
            id = sample['id']
            altered_data, altered_target = self.poison(id=id, data=(sample['data'], sample['target']), keep_org=keep_org)
            numpy_img = (altered_data.numpy() * 255).astype(np.uint8)  
            numpy_lbl = altered_target.numpy()
            # show_image_with_two_labels(scaled_img , altered_target)
            numpy_img = np.transpose(numpy_img , (1, 2, 0))
            # dataset['test'].replace_org_target(i, sample['target'])
            dataset['test'].replace_image(i, numpy_img)
            dataset['test'].replace_target(i, numpy_lbl)
            if dataset['test'][i]['target'] != cfg['target_class']:
                fail_count += 1
                print(f"FAIL: {fail_count}, sample: {sample['target']}, sampe type {type(sample['target'])}, i: {i}")
            # show_image_with_two_labels(dataset['test'][i]['data'], dataset['test'][i]['target'], dataset['test'][i]['org_target'])
        return dataset
        
    def poison(self, id, data: Tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, replace_org: bool = True, **kwargs
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""addWatermark.

        Args:
            data (tuple[torch.Tensor, torch.Tensor]): Tuple of input and label tensors.
            org (bool): Whether to return original clean data directly.
                Defaults to ``False``.
            keep_org (bool): Whether to keep original clean data in final results.
                If ``False``, the results are all infected.
                Defaults to ``True``.
            poison_label (bool): Whether to use target class label for poison data.
                Defaults to ``True``.
            **kwargs: Any keyword argument (unused).

        Returns:
            (torch.Tensor, torch.Tensor): Result tuple of input and label tensors.
        """
        _input, _label = data
        # _input size torch.Size([512, 3, 32, 32]) _input len 512
        # _label size torch.Size([512, 10]) _label len 512
        single_image = False
        if _label.dim() < 1:
            single_image = True
            _label = _label.unsqueeze(0)
        if not org:
            if keep_org:
                # decimal, integer = math.modf(len(_label) * self.poison_percent)
                decimal, integer = math.modf(len(_label) * self.poison_ratio)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                if single_image:
                    _input = self.add_trigger(org_input)
                    if poison_label:
                        _label = self.target_class * torch.ones_like(org_label[:integer])
                else:
                    _input = self.add_trigger(org_input[:integer])
                    _label = _label[:integer]
                    if poison_label:
                        _label = torch.zeros_like(org_label[:integer])
                        _label[:integer, self.target_class] = 1
                        # _label = self.target_class * torch.ones_like(org_label[:integer])
                    if id is not None:
                        self.manipulated_ids.extend(id[:integer].tolist())
                    if replace_org:
                        _input = torch.cat((_input, org_input[integer:]))
                        _label = torch.cat((_label, org_label[integer:]))
                    elif keep_org:
                        _input = torch.cat((_input, org_input))
                        _label = torch.cat((_label, org_label))
                # TODO: möchte ich wirklich von 512 auf 1024 und die bilder behalten (verdoppeln)?
                if keep_org and not single_image:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
        if single_image:
            _label = _label.squeeze(0)
        return _input, _label