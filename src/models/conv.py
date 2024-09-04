from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split
from .interm import interm
from .late import late
from .vfl import vfl
from .dl import dl


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  nn.BatchNorm2d(hidden_size[0]),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           nn.BatchNorm2d(hidden_size[i + 1]),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten()])
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(hidden_size[-1], target_size)

    def feature(self, input):
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x = self.blocks(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        
        # np_images = x.cpu().numpy()
        # if np_images[0][0][30][30].item() != 0:
        #     print("Feature split")
        #     print(input['feature_split'])
        #     print(f"[0][0][0][30] {np_images[0][0][0][30].item()}")
        #     print(f"first np_image: {np_images[0]}, shape: {np_images[0].shape}")
        #     show_images_with_labels_and_values(np_images, input['target'], input['mal_target'], 3, 3)
        x = self.blocks(x)
        output['target'] = self.linear(x)
        if 'target' in input:
            if cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
                input['target'] = input['target'].repeat(12 // cfg['num_users'], 1)
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


def conv():
    data_shape = cfg['data_shape']
    hidden_size = cfg['conv']['hidden_size']
    target_size = cfg['target_size']
    if cfg['assist_mode'] == 'interm':
        model = interm(Conv(data_shape, hidden_size, target_size), hidden_size[-1])
    elif cfg['assist_mode'] == 'late':
        model = late(Conv(data_shape, hidden_size, target_size))
    elif cfg['assist_mode'] == 'vfl':
        model = vfl(Conv(data_shape, hidden_size, target_size), hidden_size[-1])
    elif cfg['assist_mode'] == 'late':
        model = late(Conv(data_shape, hidden_size, target_size))
    elif cfg['assist_mode'] in ['none', 'bag', 'stack']:
        if 'dl' in cfg and cfg['dl'] == '1':
            model = dl(Conv(data_shape, hidden_size, target_size), hidden_size[-1])
        else:
            model = Conv(data_shape, hidden_size, target_size)
    else:
        raise ValueError('Not valid assist mode')
    model.apply(init_param)
    return model

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_images_with_labels_and_values(images, labels, values, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        image = images[i]
        if image.shape != (32, 32, 3):
            image = np.transpose(image, (1, 2, 0))
        ax.imshow(image)
        ax.axis('off')
        # Convert one-hot encoded label to original label
        if labels.dim() > 1:
            label_idx = torch.argmax(labels[i])
            label = classes[label_idx]
        else:
            label = classes[labels[i].item()]
        if values.dim() > 1:
            label_idx = torch.argmax(values[i])
            value = classes[label_idx]
        else:
            value = classes[values[i].item()]
        ax.set_title(f"Label: {label}\n value: {value}\n value: {label_idx}")
    plt.show()
    