import argparse
import copy
import datetime
import math
import os
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datasets.cifar import CIFAR10
from marks import Watermark
import models
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from assist import Assist
from utils import collate, plot_output_preds, save, load, process_control, process_dataset, resume, show_image_with_label, show_images_with_labels
from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join(
    [cfg['control'][k] for k in cfg['control'] if cfg['control'][k]]) if 'control' in cfg else ''


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    last_epoch = result['epoch']
    assist = result['assist']
    organization = result['organization']
    assist.reset()
    metric = Metric({'test': ['Loss']})
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    initialize(dataset, assist, organization[0], metric, test_logger, 0)
    
    inputs_test = torch.stack([sample['data'] for sample in dataset['test']], dim=0)
    labels_test = torch.tensor([sample['target'] for sample in dataset['test']])

    print(f"inputs_test type {type(inputs_test)}, shape {inputs_test.shape}")
    print(f"labels_test type {type(labels_test)}, shape {labels_test.shape}")        
    dataset_copy = copy.deepcopy(dataset)

    data_shape = dataset_copy['test'].data.shape
    mark = Watermark(data_shape=data_shape)
    
    np_images = inputs_test.numpy()
    # show_images_with_labels(np_images, labels_test, 3, 3)
    # images, labels = add_watermark(mark=mark, data=(inputs_test, labels_test))
    dataset_with_watermark = add_watermark_to_dataset(mark=mark, dataset=dataset_copy)
    
    altered_images = torch.stack([sample['data'] for sample in dataset_with_watermark['test']], dim=0)
    altered_labels = torch.tensor([sample['target'] for sample in dataset_with_watermark['test']])
    np_images = altered_images.numpy()
    show_images_with_labels(np_images, altered_labels, 3, 3)
    # plot_output_preds(np_images, labels_test, output, 3, 3)
    # plt.show()


    for epoch in range(1, last_epoch):
        #TODO: add watermark and test
        test_logger.safe(True)
        data_loader = assist.broadcast(dataset, epoch)
        organization_outputs = gather(data_loader, organization, epoch)
        assist.update(organization_outputs, epoch)
        test(assist, metric, test_logger, epoch)
        test_logger.safe(False)
        test_logger.reset()
    test_logger.safe(False)
    assist.reset()
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'assist': assist,
                   'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def initialize(dataset, assist, organization, metric, logger, epoch):
    logger.safe(True)
    initialization = organization.initialize(dataset, metric, logger)
    info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
    logger.append(info, 'test', mean=False)
    print(logger.write('test', metric.metric_name['test']))
    for split in dataset:
        assist.organization_output[0][split] = initialization[split]
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            assist.organization_target[0][split] = torch.tensor(np.concatenate(dataset[split].target, axis=0))
        else:
            assist.organization_target[0][split] = torch.tensor(dataset[split].target)
    logger.safe(False)
    logger.reset()
    return


def gather(data_loader, organization, epoch):
    with torch.no_grad():
        num_organizations = len(organization)
        organization_outputs = [{split: None for split in data_loader[i]} for i in range(num_organizations)]
        for i in range(num_organizations):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].predict(epoch, data_loader[i][split])['target']
    return organization_outputs


def test(assist, metric, logger, epoch):
    with torch.no_grad():
        input_size = assist.organization_target[0]['test'].size(0)
        input = {'target': assist.organization_target[0]['test']}
        output = {'target': assist.organization_output[epoch]['test']}
        output['loss'] = models.loss_fn(output['target'], input['target'])
        if cfg['data_name'] in ['MIMICM']:
            mask = input['target'] != -65535
            output['target'] = output['target'].softmax(dim=-1)[:, 1]
            output['target'], input['target'] = output['target'][mask], input['target'][mask]
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return

# def add_watermark_to_dataset(mark, dataset):
#     dataset_copy = copy.deepcopy(dataset)
    
#     for i, sample in enumerate(dataset_copy['test']):
#         print(f"image type {type(sample['data'])} shape {sample['data'].shape}")
#         print(f"label type {type(sample['target'])} shape {sample['target'].shape}")        
#         altered_data, altered_target = add_watermark(mark, (sample['data'], sample['target']))
#         print(f"image type {type(altered_data)} shape {altered_data.shape}")
#         print(f"label type {type(altered_target)} shape {altered_target.shape}")        
#         show_image_with_label(altered_data, altered_target)
#         # altered_data = sample['data'] + 1
#         # altered_target = sample['target'] + 1
#         dataset_copy['test'][i]['data'] = altered_data
#         dataset_copy['test'][i]['target'] = altered_target
#         print("Equal data", torch.all(torch.eq(altered_data, dataset_copy['test'][i]['data'])))
#         print("Equal target", torch.all(torch.eq(altered_target, dataset_copy['test'][i]['target'])))
#         show_image_with_label(dataset_copy['test'][i]['data'], dataset_copy['test'][i]['target'])
#     return dataset_copy

def add_watermark_to_dataset(mark, dataset):
    print("dataset")
    print(dataset)
    print(f"dataset type {type(dataset)}")
    print(f"test type {type(dataset['test'])}, train type {type(dataset['train'])}")
    print("test dataset")
    print(dataset['test'])
    data_len = len(dataset['test'])  # Assuming all samples have the same batch size
    model_name = cfg['model_name']
    batch_size = cfg[model_name]['batch_size']['test']
    num_batches = (data_len + batch_size - 1) // batch_size
    data_loader = make_data_loader(dataset, model_name)
    test_loader = data_loader['test']
    modified_datset = {}
    modified_datset['train'] = dataset['train']
    modified_datset['test'] = CIFAR10(root='./data/{}'.format('CIFAR10'), split='test')
    print(f"test type {type(modified_datset['test'])}, train type {type(modified_datset['train'])}")
    
    
    for i, data in enumerate(test_loader):
        data = collate(data)
        print(f"data shape {data['data'].shape}, target shape {data['target'].shape}")
        # np_images = data['data'].numpy()
        # show_images_with_labels(data['data'], data['target'], 3, 3)
        _input, _label = add_watermark(mark=mark, data=(data['data'], data['target']))
        show_images_with_labels(_input, _label, 3, 3)
        modified_datset['test'][i]['data'] = _input
        modified_datset['test'][i]['target'] = _label
        show_images_with_labels(modified_datset['test'][i]['data'], modified_datset['test'][i]['target'], 3, 3)
        # data = [(input_data, label) for input_data, label in zip(_input, _label)]
        
    # for i in range(num_batches):
    #     start_idx = i * batch_size
    #     end_idx = min((i + 1) * batch_size, data_len)

    #     batch_data = [sample['data'] for sample in dataset['test'][start_idx:end_idx]]
    #     batch_labels = [sample['target'] for sample in dataset['test'][start_idx:end_idx]]

    #     watermarked_batch_data, watermarked_batch_labels = add_watermark(mark=mark, data=(batch_data, batch_labels))

    #     for j, idx in enumerate(range(start_idx, end_idx)):
    #         dataset['test'][idx]['data'] = watermarked_batch_data[j]
    #         dataset['test'][idx]['target'] = watermarked_batch_labels[j]
    # print("data")
    # print(type(data))
    # print(len((data)))

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
        single_image = False
        if _label.dim() < 1:
            single_image = True
            _label = _label.unsqueeze(0)
        if not org:
            if keep_org:
                decimal, integer = math.modf(len(_label) * cfg['poison_ratio'])                    
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                # show_image_with_label(_input, _label)
                if single_image:
                    _input = mark.add_mark(org_input)
                else:
                    _input = mark.add_mark(org_input[:integer])
                    _label = _label[:integer]
                if poison_label:
                    _label = cfg['target_class'] * torch.ones_like(org_label[:integer])
                if keep_org and not single_image:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
                # show_image_with_label(_input, _label)
                    
        if single_image:
            _label = _label.squeeze(0)
        return _input, _label


if __name__ == "__main__":
    main()
