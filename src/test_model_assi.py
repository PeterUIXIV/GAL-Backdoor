import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import datasets
import models
from config import cfg
from data import fetch_dataset
from metrics import Metric
from utils import plot_output_preds_target, save, process_control, process_dataset, resume, show_images_with_labels
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
    cfg['backdoor_test'] = False
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
    ## Watermark
    inputs_test = torch.stack([sample['data'] for sample in dataset['test']], dim=0)
    labels_test = torch.tensor([sample['target'] for sample in dataset['test']])
    
    # if cfg['attack_mode'] == 'badnet':
    #     mark = Watermark(data_shape=cfg['data_shape'], mark_width_offset=cfg['mark_width_offset'])
    #     dataset = add_watermark_to_test_dataset(mark=mark, dataset=dataset, keep_org=False)
    # elif cfg['attack_mode'] == 'ftrojan':
    #     dataset = poison_dataset(dataset=dataset)
    
    np_images = inputs_test.numpy()
    # show_images_with_labels(np_images, labels_test, 3, 3)
    # images, labels = add_watermark(mark=mark, data=(inputs_test, labels_test))
    # dataset_with_watermark = dataset
    
    altered_images = torch.stack([sample['data'] for sample in dataset['test']], dim=0)
    altered_labels = torch.tensor([sample['target'] for sample in dataset['test']])
    np_images = altered_images.numpy()
    # show_images_with_labels(np_images, altered_labels, 3, 3)
    
    # plot_output_preds(np_images, labels_test, output, 3, 3)
    
    initialize(dataset, assist, organization[0], metric, test_logger, 0)

    for epoch in range(1, last_epoch):
        test_logger.safe(True)
        data_loader = assist.broadcast(dataset, epoch)
        # print(f"data_loader type {type(data_loader)}")
        organization_outputs = gather(data_loader, organization, epoch)
        # for i, output in enumerate(organization_outputs):
        #     print(f"output {i} type {type(output)}")
        #     print(f"output {i} test shape {output['test'].shape}")
        assist.update(organization_outputs, epoch)
        test(assist, metric, test_logger, epoch)
        test_logger.safe(False)
        test_logger.reset()
        if epoch == last_epoch - 1:
            targets = assist.organization_target[0]['test']
            # print(f"targets shape {targets.shape}")
            org_targets = assist.organization_org_target[0]['test']
            # print(f"org_targets shape {org_targets.shape}")
            test_target = torch.tensor(dataset['test'].target)
            # print(f"test_target shape {test_target.shape}")
            test_data = torch.tensor(dataset['test'].data)
            # print(f"test_data shape {test_data.shape}")
            output = assist.organization_output[epoch]['test']
            
            np_images = test_data.numpy()
            show_images_with_labels(np_images, org_targets, 3, 3)
            fig = plot_output_preds_target(test_data, org_targets, output, targets, 3, 3)
            plt.show()
            
    # test_logger.safe(False)
    # assist.reset()
    # result = resume(cfg['model_tag'], load_tag='checkpoint')
    # train_logger = result['logger'] if 'logger' in result else None
    # save_result = {'cfg': cfg, 'epoch': last_epoch, 'assist': assist,
    #                'logger': {'train': train_logger, 'test': test_logger}}
    # save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    
    # initialize(dataset_with_watermark, assist, organization[0], metric, test_logger, 0)
            
    # for epoch in range(1, last_epoch):
    #     test_logger.safe(True)
    #     data_loader = assist.broadcast(dataset_with_watermark, epoch)
    #     organization_outputs = gather(data_loader, organization, epoch)
    #     assist.update(organization_outputs, epoch)
    #     test(assist, metric, test_logger, epoch)
    #     test_logger.safe(False)
    #     test_logger.reset()
                
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
    ## adding original targets
    assist.organization_org_target[0]['test'] = torch.tensor(dataset['test'].org_target)
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
        input['org_target'] = assist.organization_org_target[0]['test']
        # print(f"input type {type(input)}")
        # print(f"input shape: {input['target'].shape}")
        ## FINAL output (i think)
        output = {'target': assist.organization_output[epoch]['test']}
        # print(f"output type {type(output)}")
        # print(f"output shape: {output['target'].shape}")
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

if __name__ == "__main__":
    main()
