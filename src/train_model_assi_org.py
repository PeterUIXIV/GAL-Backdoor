import argparse
import copy
import datetime
from marks import Watermark
import json
from assi import Assi
import models
import os
import sys
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from utils import evaluate_predictions, print_classes_preds, save, load, process_control, process_dataset, resume
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
    formatted_cfg = json.dumps(cfg, indent=4)
    print(formatted_cfg)
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%b%d_%H-%M-%S")
        print(formatted_time)
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name'], formatted_time]
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
    
    # mark = trojanvision.marks.create(dataset=dataset, mark_path=cfg['mark_path'])
    # feature_split = split_dataset(cfg['num_users'] + cfg['num_attackers'])
    feature_split = split_dataset(cfg['num_users'])
    
    print(f"target_size: {cfg['target_size']}")
    print(f"data_size: {cfg['data_size']}")
    print(f"data_shape {dataset['train'].data.shape}")
    assist = Assi(feature_split)
    # assist = Assist(feature_split[:cfg['num_users']])
    
    poison_percent = cfg['poison_percent']
    # Data_shape is only used to create the mark
    # TODO: data shape from dataset
                # data_shape = [3, 32, 32]
    data_shape = dataset['train'].data.shape
    mark = Watermark(data_shape=data_shape)
    organization = assist.make_organization(mark, poison_percent)
    print(f"Organization id: {organization[0].organization_id}")
    print(f"Organization feature_split size: {organization[0].feature_split.size()}")
    print(f"Organization model_name: {organization[0].model_name}")
    print(f"MalOrg id: {organization[-1].organization_id}")
    print(f"MalOrg feature_split size: {organization[-1].feature_split.size()}")
    print(f"MalOrg model_name: {organization[-1].model_name}")
    metric = Metric({'train': ['Loss'], 'test': ['Loss']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        logger = result['logger']
        if last_epoch > 1:
            assist = result['assist']
            organization = result['organization']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if organization is None:
        print("why this?")
        organization = assist.make_organization()
    if last_epoch == 1:
        print("Now init")
        initialize(dataset, assist, organization[0], metric, logger, 0)
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        logger.safe(True)
        data_loader = assist.broadcast(dataset, epoch)
        train(data_loader, organization, metric, logger, epoch)
        organization_outputs = gather(data_loader, organization, epoch)
        assist.update(organization_outputs, epoch)
        test(assist, metric, logger, epoch)
        logger.safe(False)
        save_result = {'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def initialize(dataset, assist, organization, metric, logger, epoch):
    logger.safe(True)
    initialization = organization.initialize(dataset, metric, logger)
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch: {}'.format(epoch), 'ID: 1']}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
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


def train(data_loader, organization, metric, logger, epoch):
    start_time = time.time()
    num_organizations = len(organization)
    for i in range(num_organizations):
        organization[i].train(epoch, data_loader[i]['train'], metric, logger)
        if i % int((num_organizations * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_organizations - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * local_time * num_organizations))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_organizations),
                             'ID: {}/{}'.format(i + 1, num_organizations),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
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
        
        print("Input:")
        # print(f"ids: {input['id'][0]}, {input['id'][1]} {input['id'][2]}")
        for key in input:
            if type(input[key]) == torch.Tensor:
                print(f"key: {key}, value shape: {input[key].shape}")
            else:
                print(f"key: {key}, value {input[key]}")
        print("Output:")
        for key in output:
            print(f"key: {key}, value shape: {output[key].shape}")
        print_classes_preds(input=input['target'], output=output['target'])
        
        evaluation_result = evaluate_predictions(input['target'], output['target'])
        print("Evaluation Metrics:")
        for sk_metric, value in evaluation_result.items():
            print(f"{sk_metric}: {value:.4f}")
            
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
