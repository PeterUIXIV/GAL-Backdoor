import collections.abc as container_abcs
import errno
import math
import random
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.optim as optim
import torch.nn.functional as F
import numbers
from itertools import repeat
from torchvision.utils import save_image
from config import cfg
from torch.nn.utils.rnn import pad_sequence

from poison.watermark import Watermark
from models.conv import Conv


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif isinstance(input, numbers.Number):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['target_size'] = dataset['train'].target_size
    cfg['data_size'] = {split: len(dataset[split]) for split in dataset}
    if cfg['data_name'] in ['MIMICL', 'MIMICM']:
        cfg['data_length'] = {split: dataset[split].length for split in dataset}
    return


def process_control():
    data_shape = {'Blob': [10], 'Iris': [4], 'Diabetes': [10], 'BostonHousing': [13], 'Wine': [13],
                  'BreastCancer': [30], 'QSAR': [41], 'MNIST': [1, 28, 28], 'CIFAR10': [3, 32, 32],
                  'ModelNet40': [3, 32, 32, 12], 'ShapeNet55': [3, 32, 32, 12], 'MIMICL': [22], 'MIMICM': [22]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['linear'] = {}
    cfg['conv'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['lstm'] = {'hidden_size': 128, 'num_layers': 1}
    cfg['gb'] = {}
    cfg['svm'] = {}
    cfg['gb-svm'] = {}
    cfg['num_users'] = int(cfg['control']['num_users'])
    cfg['assist_mode'] = cfg['control']['assist_mode']
    cfg['local_epoch'] = int(cfg['control']['local_epoch']) if cfg['control']['local_epoch'] != 'none' else 'none'
    cfg['global_epoch'] = int(cfg['control']['global_epoch']) if cfg['control'][
                                                                     'global_epoch'] != 'none' else 'none'
    cfg['assist_rate_mode'] = cfg['control']['assist_rate_mode']
    if 'noise' in cfg['control']:
        if cfg['control']['noise'] not in ['none', 'data']:
            cfg['noise'] = float(cfg['control']['noise'])
        else:
            cfg['noise'] = cfg['control']['noise']
    else:
        cfg['noise'] = 'none'
    cfg['active_rate'] = 0.1
    if 'al' in cfg['control']:
        cfg['al'] = cfg['control']['al']
    if 'rl' in cfg['control'] and cfg['control']['rl'] != 'none':
        rl_list = cfg['control']['rl'].split('-')
        num_rl = cfg['num_users'] // len(rl_list)
        rm_rl = cfg['num_users'] - num_rl * len(rl_list)
        cfg['rl'] = []
        for i in range(len(rl_list)):
            cfg['rl'].extend([rl_list[i] for _ in range(num_rl)])
            if i == len(rl_list) - 1:
                cfg['rl'].extend([rl_list[i] for _ in range(rm_rl)])
    else:
        if cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMICL']:
            cfg['rl'] = ['l1' for _ in range(cfg['num_users'])]
        else:
            cfg['rl'] = ['l2' for _ in range(cfg['num_users'])]
    if 'dl' in cfg['control']:
        cfg['dl'] = cfg['control']['dl']
    if cfg['model_name'] in ['gb', 'svm', 'gb-svm']:
        cfg['ma'] = '1'
    if 'pl' in cfg['control']:
        cfg['pl'] = cfg['control']['pl']
        if cfg['pl'] != 'none':
            pl_list = cfg['pl'].split('-')
            cfg['pl_mode'], cfg['pl_param'] = pl_list[0], float(pl_list[1])
    cfg['noised_organization_id'] = list(range(cfg['num_users'] // 2, cfg['num_users']))
    cfg['assist'] = {}
    cfg['assist']['batch_size'] = {'train': 1024, 'test': 1024}
    cfg['assist']['optimizer_name'] = 'Adam'
    cfg['assist']['lr'] = 1e-1
    cfg['assist']['momentum'] = 0.9
    cfg['assist']['weight_decay'] = 5e-4
    cfg['assist']['num_epochs'] = 100
    cfg['linesearch'] = {}
    cfg['linesearch']['optimizer_name'] = 'LBFGS'
    cfg['linesearch']['lr'] = 1
    cfg['linesearch']['num_epochs'] = 10
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    if model_name in ['linear']:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': 1024, 'test': 1024}
        cfg[model_name]['lr'] = 1e-1
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
    elif model_name in ['conv']:
        if cfg['data_name'] in ['MNIST', 'CIFAR10']:
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            cfg[model_name]['batch_size'] = {'train': 512, 'test': 512}
            cfg[model_name]['lr'] = 1e-1
        elif cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            cfg[model_name]['batch_size'] = {'train': 64, 'test': 128}
            cfg[model_name]['lr'] = 1e-1
            torch.set_num_threads(2)
        else:
            raise ValueError('Not valid data name')
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
    elif model_name in ['lstm']:
        cfg[model_name]['ICD9_embeddings'] = 5893
        cfg[model_name]['optimizer_name'] = 'Adam'
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': 8, 'test': 8}
        cfg[model_name]['lr'] = 1e-3
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'None'
    elif model_name in ['gb', 'svm', 'gb-svm']:
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': 1024, 'test': 1024}
        cfg[model_name]['lr'] = 1e-1
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
        if model_name == 'gb-svm':
            cfg['gb'] = cfg[model_name]
            cfg['svm'] = cfg[model_name]
    else:
        raise ValueError('Not valid model name')
    cfg['global'] = {}
    cfg['global']['num_epochs'] = cfg['global_epoch']
    cfg['stats'] = make_stats()
    ## Backdoor ##
    cfg['attack'] = None if cfg['attack'] == 'None' else cfg['attack']
    cfg['num_attackers'] = int(cfg['num_attackers'])
    cfg['poison_percentage'] = float(cfg['poison_percentage'])
    cfg['poison_ratio'] = cfg['poison_percentage'] / (1 - cfg['poison_percentage'])
    cfg['target_class'] = int(cfg['backdoor']['target_class'])
    # cfg['poison_dataset'] = cfg['backdoor']['poison_dataset']
    ## Mark ##
    cfg['mark_path'] = cfg['mark']['mark_path']
    # cfg['backdoor_test'] = True
    cfg['mark_width_offset'] = int(cfg['mark']['width_offset'])
    cfg['YUV'] = cfg['mark']['YUV']
    cfg['channel_list'] = cfg['mark']['channel_list']
    cfg['pos_list'] = cfg['mark']['pos_list']
    cfg['magnitude'] = int(cfg['mark']['magnitude'])
    ## Defense ##
    cfg['defense'] = None if cfg['defense'] == 'None' or cfg['defense'] == 'none' or cfg['defense'] == 'null' else cfg['defense']
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs']['global'],
                                                         eta_min=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result


def collate(input):
    for k in input:
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            if k == 'data':
                input[k] = pad_sequence(input['data'], batch_first=True, padding_value=0)
            elif k == 'target':
                input[k] = pad_sequence(input['target'], batch_first=True, padding_value=-65535)
            else:
                input[k] = torch.stack(input[k], 0)
        else:
            input[k] = torch.stack(input[k], 0)
    return input

def show_images(images, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        image = images[i]
        if image.shape != (32, 32, 3):
            image = np.transpose(image, (1, 2, 0))
        ax.imshow(image)
        ax.axis('off')
    plt.show()
    
def show_images_with_labels(images, labels, nrows, ncols):
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
        ax.set_title(f"Label: {label}")
    plt.show()
    
def show_images_with_two_labels(images, true_labels, target_labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        image = images[i]
        if image.shape != (32, 32, 3):
            image = np.transpose(image, (1, 2, 0))
        ax.imshow(image)
        ax.axis('off')
        # Convert one-hot encoded label to original label
        if true_labels.ndim > 1:
            label_idx = torch.argmax(true_labels[i])
            true_label = classes[label_idx]
        else:
            true_label = classes[true_labels[i].item()]
        if target_labels.ndim > 1:
            label_idx = torch.argmax(target_labels[i])
            target_label = classes[label_idx]
        else:
            target_label = classes[target_labels[i].item()]
        ax.set_title("True Label: {}\nTarget Label: {}".format(true_label, target_label))
    plt.show()
    
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
        ax.set_title(f"Label: {label}\n value: {value}")
    plt.show()
    
# def images_to_probs(net, images):
#     '''
#     Generates predictions and corresponding probabilities from a trained
#     network and a list of images
#     '''
#     output = net(images)
#     output = output['target']
#     # convert output probabilities to predicted class
#     _, preds_tensor = torch.max(output, 1)
#     preds = np.squeeze(preds_tensor.cpu().numpy())
#     return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    net.eval()  # Ensure the network is in evaluation mode
    with torch.no_grad():  # No need to compute gradients during inference
        output = net(images)
        logits = output['target']
        probabilities = F.softmax(logits, dim=1)
        max_prob, pred = torch.max(probabilities, 1)
        pred = pred.cpu().numpy()
        max_prob = max_prob.cpu().numpy()
    return pred, max_prob

def matplotlib_imshow(img: torch.Tensor, ax, normalize=False):
    if normalize:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    
    # Check if the image is grayscale or color
    if npimg.shape[0] == 1:  # Grayscale image
        npimg = npimg.squeeze()  # Remove the channel dimension
        ax.imshow(npimg, cmap='gray')
    else:
        if npimg.shape != (32, 32, 3):
            npimg = np.transpose(npimg, (1, 2, 0))
        ax.imshow(npimg)

# def plot_classes_preds(net, images, labels):
#     '''
#     Generates matplotlib Figure using a trained network, along with images
#     and labels from a batch, that shows the network's top prediction along
#     with its probability, alongside the actual label, coloring this
#     information based on whether the prediction was correct or not.
#     Uses the "images_to_probs" function.
#     '''
#     preds, probs = images_to_probs(net, images)
#     # plot the images in the batch, along with predicted and true labels
#     fig = plt.figure(figsize=(12, 48))
#     for idx in np.arange(4):
#         ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
#         matplotlib_imshow(images['data'][idx], normalize=False)
#         label_idx = torch.argmax(labels[idx])
#         ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
#             classes[preds[idx]],
#             probs[idx] * 100.0,
#             classes[label_idx]),
#                 color=("green" if preds[idx]==label_idx.item() else "red"))
#             # classes[labels[idx]]),
#                     # color=("green" if preds[idx]==labels[idx].item() else "red"))
#     return fig
def plot_classes_preds(net: Conv, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
        
    preds, probs = images_to_probs(net, images)
    # print(f"preds type {type(preds)}, preds len {preds.shape}")
    # print(f"probs type {type(probs)}, probs len {probs.shape}")
    
    # plot the images in the batch, along with predicted and true labels
    images = images['data']
    # print(f"images type {type(images)}, shape {images.shape}")
    true_classes = torch.argmax(labels, dim=1)
    # print(f"true_classes type {type(true_classes)}, shape {true_classes.shape}")

    fig = plt.figure(figsize=(12, 48))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], normalize=False)
        pred_class_idx = preds[idx]
        true_class_idx = true_classes[idx].item()
        pred_class_name = classes[pred_class_idx]
        true_class_name = classes[true_class_idx]
        # print(f"pred_class_name: {pred_class_name}, type: {type(pred_class_name)}")
        # print(f"true_class_name: {true_class_name}, type: {type(true_class_name)}")

        pred_prob = probs[idx] * 100.0
        # print(f"pred_prob: {pred_prob}, type: {type(pred_prob)}")
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            pred_class_name,
            pred_prob,
            true_class_name),
            color=("green" if pred_class_idx == true_class_idx else "red"))
    return fig

def output_to_preds(output: torch.Tensor):
    probabilities = F.softmax(output, dim=1)
    max_prob, pred = torch.max(probabilities, 1)
    pred = pred.cpu().numpy()
    max_prob = max_prob.cpu().numpy()
    return pred, max_prob

def plot_output_preds(images: torch.Tensor, labels: torch.Tensor, output: torch.Tensor, nrows: int, ncols: int):
    preds, probs = output_to_preds(output)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        matplotlib_imshow(images[i], ax, normalize=False)
        true_class_name = classes[labels[i].item()]
        pred_class_name = classes[preds[i]]
        pred_prob = probs[i] * 100.0
        
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            pred_class_name,
            pred_prob,
            true_class_name),
            color=("green" if pred_class_name == true_class_name else "red"))
    plt.tight_layout()
    return fig

def plot_output_preds_target(images: torch.Tensor, labels: torch.Tensor, output: torch.Tensor, target: torch.Tensor, nrows: int, ncols: int):
    preds, probs = output_to_preds(output)
        
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        matplotlib_imshow(images[i], ax, normalize=False)
        true_class_name = classes[labels[i].item()]
        pred_class_name = classes[preds[i]]
        pred_prob = probs[i] * 100.0
        target_class_name = classes[target[i]]
        
        ax.set_title("{0}, {1:.1f}%\n(label: {2})\n(target: {3})".format(
            pred_class_name,
            pred_prob,
            true_class_name,
            target_class_name),
            color=("green" if pred_class_name == true_class_name else "red"))
    plt.tight_layout()
    return fig

def print_classes_preds(labels: torch.Tensor, output: torch.Tensor):
    r"""Print class predictions.
    
    Args:
        labels (torch.Tensor): Label tensor representing classes.
        output (torch.Tensor): Output tensor representing class probabilities.

    """
    probabilities = F.softmax(output, dim=1)
    max_prob, pred = torch.max(probabilities, 1)
    pred = pred.cpu().numpy()
    max_prob = max_prob.cpu().numpy()
    for i in range(4):
        pred_class_name = classes[pred[i]]
        true_class_name = classes[labels[i].item()]
        print(f"output {i}: {output[i]}")
        print(f"probabilities {i}: {probabilities[i]}")
        print(f"label {i}: {labels[i].item()}, type: {type(labels[i].item())} class_name {true_class_name}")
        print(f"pred {i}: {pred[i]}, type: {type(pred[i])} class_name {pred_class_name}")
        print(f"max_prob {i}: {max_prob[i]}, type: {type(max_prob[i])}")
        
        
        # true_class_idx = true_classes[i].item()
        # true_class_name = classes[true_class_idx]
        
def evaluate_predictions(true_labels, predicted_probs, threshold=0.5):
    """
    Evaluate predictions using accuracy, precision, recall, and F1-score.
    
    Args:
    true_labels (torch.Tensor): True labels tensor of shape (batch_size).
    predicted_probs (torch.Tensor): Predicted probabilities tensor of shape (batch_size, num_classes).
    threshold (float): Threshold for converting probabilities to binary predictions (default is 0.5).
    
    Returns:
    dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    # Convert predicted probabilities to binary predictions
    predicted_labels = torch.argmax(predicted_probs, dim=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels.numpy(), predicted_labels.numpy())
    precision = precision_score(true_labels.numpy(), predicted_labels.numpy(), average='macro')
    recall = recall_score(true_labels.numpy(), predicted_labels.numpy(), average='macro')
    f1 = f1_score(true_labels.numpy(), predicted_labels.numpy(), average='macro')
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
    
    return metrics

def show_image_with_label(image_tensor, label_tensor):
    # Convert image tensor from (3, 32, 32) to (32, 32, 3)
    image = np.transpose(image_tensor, (1, 2, 0))
    
    # Plot the image
    plt.imshow(image)
    
    # Set title with the label
    plt.title(f'Label: {classes[label_tensor.item()]}')
    
    # Show plot
    plt.axis('off')
    plt.show()
    
def show_image_with_two_labels(image_tensor, true_label, target_label):
    # Convert image tensor from (3, 32, 32) to (32, 32, 3)
    image = np.transpose(image_tensor, (1, 2, 0))
    
    # Plot the image
    plt.imshow(image)
    
    # Set title with the label
    plt.title("True Label: {}\nTarget Label: {}".format(classes[true_label.item()], classes[target_label.item()]))

    # Show plot
    plt.axis('off')
    plt.show()
    
def save_images_to_txt(images, labels, num, path):
    makedir_exist_ok(path)
    for i in range(num):
        with open(f"{path}/img_{i}.txt", "w") as txt_file:
            for j in range(images[i].shape[0]):  # Iterate over the slices
                np.savetxt(txt_file, images[i][j], fmt='%d')
                txt_file.write("\n")  # Add a newline for separation between slices
                
            txt_file.write(f"Label: {labels[i]}\n")
    
def add_watermark_to_test_dataset(mark, dataset, keep_org):
    fail_count = 0
    for i, sample in enumerate(dataset['test']):
        altered_data, altered_target = add_watermark(mark=mark, data=(sample['data'], sample['target']), keep_org=keep_org)
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

def add_watermark(mark: Watermark, data: tuple[torch.Tensor, torch.Tensor],
                 org: bool = False, keep_org: bool = True,
                 poison_label: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # single_image = False
        # if _label.dim() < 1:
        #     single_image = True
        #     _label = _label.unsqueeze(0)
        if not org:
            if keep_org:
                # decimal, integer = math.modf(len(_label) * cfg['poison_percentage'])                    
                decimal, integer = math.modf(len(_label) * cfg['poison_ratio'])                    
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
                indices = random.sample(range(len(_label)), integer)
                indices = np.sort(indices)
            else:
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input.clone(), _label.clone()
                # show_image_with_label(_input, _label)
                for idx in indices:
                    _input[idx] = mark.add_mark(_input[idx])
                _label[indices] = cfg['target_class']
                # _label = cfg['target_class'] * torch.ones_like(org_label[:integer])
                # if keep_org:
                #     _input = torch.cat((_input, org_input[integer:]))
                #     _label = torch.cat((_label, org_label[integer:]))
                # show_image_with_label(_input, _label)
                    
        # if single_image:
        #     _label = _label.squeeze(0)
        return _input, _label, indices

def numpy_to_torch(images_np):
    """
    Converts a NumPy array of images to a PyTorch tensor of images.

    Args:
    images_np (numpy.ndarray): A NumPy array of images.

    Returns:
    torch.Tensor: A PyTorch tensor of images.
    """
    if not isinstance(images_np, np.ndarray):
        raise TypeError("Input should be a NumPy array")

    images_tensor = torch.from_numpy(images_np)
    
    if images_tensor.ndimension() == 4:  # assuming a batch of images with shape (N, H, W, C)
        images_tensor = images_tensor.permute(0, 3, 1, 2)
    
    return images_tensor

def torch_to_numpy(images_tensor):
    """
    Converts a PyTorch tensor of images to a NumPy array of images.

    Args:
    images_tensor (torch.Tensor): A PyTorch tensor of images.

    Returns:
    numpy.ndarray: A NumPy array of images.
    """
    if not torch.is_tensor(images_tensor):
        raise TypeError("Input should be a PyTorch tensor")

    if images_tensor.is_cuda:
        images_tensor = images_tensor.cpu()

    if images_tensor.ndimension() == 4 and images_tensor.shape[1] in {1, 3}:  # assuming a batch of images with shape (N, C, H, W)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
    
    # Convert PyTorch tensor to NumPy array
    images_np = images_tensor.numpy()
    
    return images_np
