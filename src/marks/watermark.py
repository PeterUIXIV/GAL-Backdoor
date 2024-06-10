from collections.abc import Callable
import os
import numpy as np
import torch
import torchvision.transforms.functional as F
import PIL.Image as Image
from config import cfg

dir_path = os.path.dirname(__file__)

def get_edge_color(
    mark: torch.Tensor,
    mark_background_color: str | torch.Tensor = 'auto'
) -> torch.Tensor | None:
    # if any pixel is not fully opaque
    if not mark[-1].allclose(torch.ones_like(mark[-1]), atol=1e-3):
        return None
    mark = mark[:-1]    # remove alpha channel
    match mark_background_color:
        case torch.Tensor():
            return torch.as_tensor(mark_background_color).expand(mark.size(0))
        case 'black':
            return torch.zeros(mark.size(0))
        case 'white':
            return torch.ones(mark.size(0))
        case 'auto':
            if mark.flatten(1).std(1).max() < 1e-3:
                return None
            else:
                _list = [mark[:, 0, :], mark[:, -1, :],
                         mark[:, :, 0], mark[:, :, -1]]
                return torch.cat(_list, dim=1).mode(dim=-1)[0]
        case _:
            raise ValueError(f'{mark_background_color=:s}')

def update_mark_alpha_channel(
    mark: torch.Tensor,
    mark_background_color: torch.Tensor | None = None
) -> torch.Tensor:
    if mark_background_color is None:
        return mark
    mark = mark.clone()
    mark_background_color = mark_background_color.view(-1, 1, 1)
    mark[-1] = ~mark[:-1].isclose(mark_background_color,
                                atol=1e-3).all(dim=0)
    return mark

class Watermark():
    def __init__(self, mark_path: str = 'apple_black.png',
                 data_shape: list[int] = None, 
                 mark_background_color: str | torch.Tensor = 'auto',
                 mark_alpha: float = 1.0, 
                 mark_height: int = 3, 
                 mark_width: int = 3,
                 mark_height_offset: int = 0, 
                 mark_width_offset: int = 0,
                 mark_random_init: bool = False, 
                 mark_random_pos: bool = False,
                 mark_scattered: bool = False,
                 mark_scattered_height: int = None,
                 mark_scattered_width: int = None,
                 add_mark_fn: Callable[..., torch.Tensor] = None,
                 **kwargs):
        self.param_list: dict[str, list[str]] = {}
        self.param_list['mark'] = ['mark_path',
                                   'mark_alpha', 'mark_height', 'mark_width',
                                   'mark_random_init', 'mark_random_pos',
                                   'mark_scattered']
        if not mark_random_pos:
            self.param_list['mark'].extend(['mark_height_offset', 'mark_width_offset'])
        assert mark_height > 0 and mark_width > 0
        
        self.mark_alpha = mark_alpha
        self.mark_path = mark_path
        self.mark_height = mark_height
        self.mark_width = mark_width
        self.mark_height_offset = mark_height_offset
        self.mark_width_offset = mark_width_offset
        self.mark_random_init = mark_random_init
        self.mark_random_pos = mark_random_pos
        self.mark_scattered = mark_scattered
        self.mark_scattered_height = mark_scattered_height or data_shape[1]
        self.mark_scattered_width = mark_scattered_width or data_shape[2]
        self.add_mark_fn = add_mark_fn
        self.data_shape = data_shape
        
        self.mark = self.load_mark(mark_img=mark_path,
                                   mark_background_color=mark_background_color)
        
    def add_mark(self, _input: torch.Tensor, mark_random_pos: bool = None,
                 mark_alpha: float = None, mark: torch.Tensor = None,
                 **kwargs) -> torch.Tensor:
        r"""Main method to add watermark to a batched input image tensor ranging in ``[0, 1]``.

        Call :attr:`self.add_mark_fn()` instead if it's not ``None``.

        Args:
            _input (torch.Tensor): Batched input tensor
                ranging in ``[0, 1]`` with shape ``(N, C, H, W)``.
            mark_random_pos (bool | None): Whether to add mark at random location.
                Defaults to :attr:`self.mark_random_pos`.
            mark_alpha (float | None): Mark opacity. Defaults to :attr:`self.mark_alpha`.
            mark (torch.Tensor | None): Mark tensor. Defaults to :attr:`self.mark`.
            **kwargs: Keyword arguments passed to `self.add_mark_fn()`.
        """
        mark_alpha = mark_alpha if mark_alpha is not None else self.mark_alpha
        mark = mark if mark is not None else self.mark
        mark_random_pos = mark_random_pos if mark_random_pos is not None else self.mark_random_pos
        if callable(self.add_mark_fn):
            return self.add_mark_fn(_input, mark_random_pos=mark_random_pos,
                                    mark_alpha=mark_alpha, **kwargs)
        trigger_input = _input.clone()
        mark = mark.clone().to(device=_input.device)

        mark_rgb_channel = mark[..., :-1, :, :]
        mark_alpha_channel = mark[..., -1, :, :].unsqueeze(-3)
        mark_alpha_channel *= mark_alpha
        if mark_random_pos:
            batch_size = _input.size(0)
            h_start = torch.randint(high=_input.size(-2) - self.mark_height, size=[batch_size])
            w_start = torch.randint(high=_input.size(-1) - self.mark_width, size=[batch_size])
            h_end, w_end = h_start + self.mark_height, w_start + self.mark_width
            for i in range(len(_input)):    # TODO: any parallel approach?
                org_patch = _input[i, :, h_start[i]:h_end[i], w_start[i]:w_end[i]]
                trigger_patch = org_patch + mark_alpha_channel * (mark_rgb_channel - org_patch)
                trigger_input[i, :, h_start[i]:h_end[i], w_start[i]:w_end[i]] = trigger_patch
            return trigger_input
        h_start, w_start = self.mark_height_offset, self.mark_width_offset
        h_end, w_end = h_start + self.mark_height, w_start + self.mark_width
        org_patch = _input[..., h_start:h_end, w_start:w_end]
        trigger_patch = org_patch + mark_alpha_channel * (mark_rgb_channel - org_patch)
        trigger_input[..., h_start:h_end, w_start:w_end] = trigger_patch

        return trigger_input
        
    def load_mark(self,
                  mark_img: str | Image.Image | np.ndarray | torch.Tensor,
                  mark_background_color: None | str | torch.Tensor = 'auto',
                  already_processed: bool = False
                  ) -> torch.Tensor:
        r"""Load watermark tensor from image :attr:`mark_img`,
        scale by calling :any:`PIL.Image.Image.resize`
        and transform to ``(channel + 1, height, width)`` with alpha channel.

        Args:
            mark_img (PIL.Image.Image | str): Pillow image instance or file path.
            mark_background_color (str | torch.Tensor | None): Mark background color.
                If :class:`str`, choose from ``['auto', 'black', 'white']``;
                else, it shall be 1-dim tensor ranging in ``[0, 1]``.
                It's ignored when alpha channel in watermark image.
                Defaults to ``'auto'``.
            already_processed (bool):
                If ``True``, will just load :attr:`mark_img` as :attr:`self.mark`.
                Defaults to ``False``.

        Returns:
            torch.Tensor:
                Watermark tensor ranging in ``[0, 1]``
                with shape ``(channel + 1, height, width)`` with alpha channel.
        """
        if isinstance(mark_img, str):
            if mark_img.endswith('.npy'):
                mark_img = np.load(mark_img)
            else:
                if not os.path.isfile(mark_img) and \
                        not os.path.isfile(mark_img := os.path.join(dir_path, mark_img)):
                    raise FileNotFoundError(mark_img.removeprefix(dir_path))
                mark_img = F.convert_image_dtype(F.pil_to_tensor(Image.open(mark_img)))
        if isinstance(mark_img, np.ndarray):
            mark_img = torch.from_numpy(mark_img)
        # mark: torch.Tensor = mark_img.to(device=env['device'])
        # correct device?
        mark: torch.Tensor = mark_img.to(device=cfg['device'])
        if not already_processed:
            mark = F.resize(mark, size=(self.mark_width, self.mark_height))
            alpha_mask = torch.ones_like(mark[0])
            if mark.size(0) == 4:
                mark = mark[:-1]
                alpha_mask = mark[-1]
            if self.data_shape[0] == 1 and mark.size(0) == 3:
                mark = F.rgb_to_grayscale(mark, num_output_channels=1)
            mark = torch.cat([mark, alpha_mask.unsqueeze(0)])

            if mark_background_color is not None:
                mark = update_mark_alpha_channel(mark, get_edge_color(mark, mark_background_color))
            if self.mark_random_init:
                mark[:-1] = torch.rand_like(mark[:-1])

            if self.mark_scattered:
                mark_scattered_shape = [mark.size(0), self.mark_scattered_height, self.mark_scattered_width]
                mark = self.scatter_mark(mark, mark_scattered_shape)
        self.mark_height, self.mark_width = mark.shape[-2:]
        self.mark = mark
        return mark
    