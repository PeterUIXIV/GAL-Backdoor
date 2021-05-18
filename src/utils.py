import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


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
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['target_size'] = dataset['train'].target_size
    cfg['data_size'] = {split: len(dataset[split]) for split in dataset}
    return


def process_control():
    data_shape = {'Blob': [10], 'Iris': [4], 'Diabetes': [10], 'BostonHousing': [13], 'Wine': [13],
                  'BreastCancer': [30], 'QSAR': [41], 'MNIST': [1, 28, 28], 'CIFAR10': [3, 32, 32],
                  'ModelNet40': [3, 32, 32, 12], 'MIMIC': [76]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['linear'] = {}
    cfg['conv'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['lstm'] = {'hidden_size': 128, 'num_layers': 1}
    cfg['num_users'] = int(cfg['control']['num_users'])
    cfg['assist_mode'] = cfg['control']['assist_mode']
    cfg['local_epoch'] = int(cfg['control']['local_epoch']) if cfg['control']['local_epoch'] != 'none' else 'none'
    cfg['global_epoch'] = int(cfg['control']['global_epoch']) if cfg['control'][
                                                                     'global_epoch'] != 'none' else 'none'
    cfg['assist_rate_mode'] = cfg['control']['assist_rate_mode']
    cfg['noise'] = float(cfg['control']['noise']) if cfg['control']['noise'] != 'none' else 'none'
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
        elif cfg['data_name'] in ['ModelNet40']:
            cfg[model_name]['optimizer_name'] = 'SGD'
            cfg[model_name]['momentum'] = 0.9
            cfg[model_name]['weight_decay'] = 5e-4
            cfg[model_name]['batch_size'] = {'train': 64, 'test': 512}
            cfg[model_name]['lr'] = 1e-1
        else:
            raise ValueError('Not valid data name')
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'MultiStepLR'
        cfg[model_name]['factor'] = 0.1
        cfg[model_name]['milestones'] = [50, 100]
    elif model_name in ['lstm']:
        cfg[model_name]['optimizer_name'] = 'Adam'
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'train': 1, 'test': 1}
        cfg[model_name]['lr'] = 1e-4
        cfg[model_name]['num_epochs'] = cfg['local_epoch']
        cfg[model_name]['scheduler_name'] = 'None'
    else:
        raise ValueError('Not valid model name')
    cfg['global'] = {}
    cfg['global']['num_epochs'] = cfg['global_epoch']
    cfg['stats'] = make_stats()
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
        input[k] = torch.stack(input[k], 0)
    return input
