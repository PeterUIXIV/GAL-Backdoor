import argparse
import copy
import datetime

from matplotlib import pyplot as plt
import json
from anomaly import detect_anomalies, get_anomaly_metrics_for_org
from assi import Assi
from mal_org import MalOrg
import models
import os
import sys
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from poison.badnet_agent import BadnetAgent
from poison.ftrojan_agent import FtrojanAgent
from poison.watermark import Watermark
from utils import evaluate_predictions, plot_output_preds, plot_output_preds_target, print_classes_preds, save, load, process_control, process_dataset, resume, show_images_with_labels
from logger import make_logger

# python train_model_assi_org.py --control_name 2_stack_2_2_search_0

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
    # for line in formatted_cfg.splitlines():
    #     print(line)
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%b%d_%H-%M-%S")
        # model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name'], formatted_time]
        model_tag_list = [str(seeds[i]), cfg['control_name'], str(cfg['poison_percentage']), str(cfg['num_attackers'])]
        
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
    
    feature_split = split_dataset(cfg['num_users'])
    assist = Assi(feature_split)
    # assist = Assist(feature_split)
    
    if not cfg['attack'] == None:
        poison_percent = cfg['poison_percentage']
        poison_ratio = poison_percent / (1 - poison_percent)
        target_class = cfg['target_class']
        if cfg['attack'] == 'badnet':
            mark = Watermark(mark_path=cfg['mark_path'], data_shape=cfg['data_shape'], mark_width_offset=cfg['mark_width_offset'], mark_height_offset=cfg['mark_height_offset'])
            poison_agent = BadnetAgent(poison_percent=poison_percent, poison_ratio=poison_ratio, target_class=target_class, mark=mark)
        elif cfg['attack'] == 'ftrojan':
            poison_agent = FtrojanAgent(poison_percent=poison_percent, poison_ratio=poison_ratio, target_class=target_class)
        else:
            raise ValueError(f"No valid attack type: {cfg['attack']}")
    else:
        poison_agent = None
        # dataset = poison_agent.poison_test_dataset(dataset=dataset, keep_org=True)
        
    organization = assist.make_organization(poison_agent)
    # print(f"Organization id: {organization[0].organization_id}")
    # print(f"Organization feature_split size: {organization[0].feature_split.size()}")
    # print(f"Organization model_name: {organization[0].model_name}")
    # print(f"MalOrg id: {organization[-1].organization_id}")
    # print(f"MalOrg feature_split size: {organization[-1].feature_split.size()}")
    # print(f"MalOrg model_name: {organization[-1].model_name}")
    
    
    # altered_images = torch.stack([sample['data'] for sample in dataset['test']], dim=0)
    # altered_labels = torch.tensor([sample['target'] for sample in dataset['test']])
    # np_images = altered_images.numpy()
    # show_images_with_labels(np_images, altered_labels, 3, 3)
    # show_images_with_labels(np_images[9:], altered_labels[9:], 3, 3)
    
    # folder = "output/imgs"
    # os.makedirs(folder, exist_ok=True)
    # for i in range(18):
    #     file_path = os.path.join(folder, f"marked_pic_{i}.png")
    #     print(np_images[i].shape)
    #     array = np.transpose(np_images[i], (1, 2, 0))
    #     print(array.shape)
    #     if array.dtype != np.uint8:
    #         array = (array * 255).astype(np.uint8) if array.dtype == np.float32 else array.astype(np.uint8)
        
    #     im = Image.fromarray(array)
    #     im.save(file_path)
        
    metric = Metric({'train': ['Loss'], 'test': ['Loss']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        logger = result['logger']
        if last_epoch > 1:
            assist = result['assist']
            organization = result['organization']
        else:
            logger = make_logger(os.path.join('output', 'runs', str(cfg['attack']), str(cfg['defense']), 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', str(cfg['attack']), str(cfg['defense']), 'train_{}'.format(cfg['model_tag'])))
    if organization is None:
        organization = assist.make_organization(poison_agent)
    if last_epoch == 1:
        initialize(dataset, assist, organization[0], metric, logger, 0)
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        logger.safe(True)
        data_loader = assist.broadcast(dataset, epoch)
        train(data_loader, organization, metric, logger, epoch)
        organization_outputs = gather(data_loader, organization, epoch)
        if cfg['attack'] is not None:
            actual_manipulated_ids = load(os.path.join('./data/{}'.format(cfg['data_name']), 'poisoned', cfg['attack'], str(cfg['poison_percentage']), 'indices.npy'), mode='np')
            print(f"# manipulated ids: {len(actual_manipulated_ids)}")
        else:
            actual_manipulated_ids = []
        manipulated_ids_by_org = [actual_manipulated_ids if isinstance(org, MalOrg) else [] for org in organization]
        if cfg['defense'] is not None:
            pred_anomalies_by_org = detect_anomalies(organization_outputs)
            for i in range(len(pred_anomalies_by_org)):
                metrics = get_anomaly_metrics_for_org(pred_anomalies_by_org[i], manipulated_ids_by_org[i])
                logger.append(metrics, tag=f'org_{i}')
                logger.write_anomaly_metrics(i, metrics, epoch)
        else:
            pred_anomalies_by_org = [[] for _ in range(len(organization_outputs))]
        assist.update(organization_outputs, pred_anomalies_by_org, epoch)
        test(assist, metric, logger, epoch)
        logger.safe(False)
        save_result = {'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        # save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        save(save_result, os.path.join('output', 'model', str(cfg['attack']), str(cfg['defense']), '{}_checkpoint.pt'.format(cfg['model_tag'])))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy(os.path.join('output', 'model', str(cfg['attack']), str(cfg['defense']), '{}_checkpoint.pt'.format(cfg['model_tag'])),
                        os.path.join('output', 'model', str(cfg['attack']), str(cfg['defense']), '{}_best.pt'.format(cfg['model_tag'])))
        logger.reset()
        
        ## Plot image last epoch
        if epoch == cfg['global']['num_epochs']:
            targets = assist.organization_target[0]['test']
            # print(f"targets shape {targets.shape}")
            mal_targets = assist.organization_mal_target[0]['test']
            # print(f"org_targets shape {org_targets.shape}")
            test_target = torch.tensor(dataset['test'].target)
            # print(f"test_target shape {test_target.shape}")
            test_data = torch.tensor(dataset['test'].data)
            # print(f"test_data shape {test_data.shape}")
            output = assist.organization_output[epoch]['test']
            
            np_images = test_data.numpy()
            # show_images_with_labels(np_images, org_targets, 3, 3)
            # fig = plot_output_preds_target(test_data, org_targets, output, targets, 3, 3)
            # fig = plot_output_preds(test_data, targets, output, 3, 3)
            # plt.show()
    logger.safe(False)
    return


def initialize(dataset, assist, organization, metric, logger, epoch):
    logger.safe(True)
    initialization = organization.initialize(dataset, metric, logger)
    # print(f"intialization {initialization} type {type(initialization)}")
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
            assist.organization_mal_target[0][split] = torch.tensor(dataset[split].mal_target)
    logger.safe(False)
    logger.reset()
    return


def train(data_loader, organization, metric, logger, epoch):
    start_time = time.time()
    num_organizations = len(organization)
    fuck_count = 0
    for i in range(num_organizations):
        # for j, input in enumerate(data_loader[i]['train']):
        #     if input['target'] != 'plane':
        #         fuck_count += 1
        #         print(f"FUCK: {fuck_count}, target: {input['target']}, j: {j}")
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
        # manipulated_ids = [{split: None for split in data_loader[i]} for i in range(num_organizations)]
        for i in range(num_organizations):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].predict(epoch, data_loader[i][split])['target']
                # manipulated_ids[i][split] = organization[i].manipulated_ids
    # return organization_outputs, manipulated_ids
    return organization_outputs


def test(assist, metric, logger, epoch):
    with torch.no_grad():
        input_size = assist.organization_target[0]['test'].size(0)
        input = {'target': assist.organization_target[0]['test']}
        input['mal_target'] = assist.organization_mal_target[0]['test']
        output = {'target': assist.organization_output[epoch]['test']}
        output['loss'] = models.loss_fn(output['target'], input['target'])
        if cfg['data_name'] in ['MIMICM']:
            mask = input['target'] != -65535
            output['target'] = output['target'].softmax(dim=-1)[:, 1]
            output['target'], input['target'] = output['target'][mask], input['target'][mask]
        
        # print("Input:")
        # # print(f"ids: {input['id'][0]}, {input['id'][1]} {input['id'][2]}")
        # for key in input:
        #     if type(input[key]) == torch.Tensor:
        #         print(f"key: {key}, value shape: {input[key].shape}")
        #     else:
        #         print(f"key: {key}, value {input[key]}")
        # print("Output:")
        # for key in output:
        #     print(f"key: {key}, value shape: {output[key].shape}")
        # print_classes_preds(labels=input['target'], output=output['target'])
        
        # evaluation_result = evaluate_predictions(input['target'], output['target'])
        # print("Evaluation Metrics:")
        # for sk_metric, value in evaluation_result.items():
        #     print(f"{sk_metric}: {value:.4f}")
            
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
