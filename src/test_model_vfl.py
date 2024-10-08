import argparse
import datetime
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate
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
    data_loader = make_data_loader(dataset, cfg['model_name'])
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    metric = Metric({'test': ['Loss']})
    result = resume(cfg['model_tag'], load_tag='best')
    last_epoch = result['epoch']
    feature_split = result['feature_split']
    model.load_state_dict(result['model_state_dict'])
    test_logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test_logger.safe(True)
    test(data_loader['test'], feature_split, model, metric, test_logger, last_epoch)
    test_logger.safe(False)
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, feature_split, model, metric, logger, epoch):
    with torch.no_grad():
        model.train(False)
        output_, target_ = [], []
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input['feature_split'] = feature_split
            input['buffer'] = 'test'
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            if cfg['data_name'] in ['MIMICM']:
                mask = input['target'] != -65535
                target_i = input['target']
                output_i = output['target'].softmax(dim=-1)[:, :, 1]
                output_i, target_i = output_i[mask], target_i[mask]
                output_.append(output_i.cpu())
                target_.append(target_i.cpu())
                evaluation = metric.evaluate([metric.metric_name['test'][0]], input, output)
                logger.append(evaluation, 'test', input_size)
            else:
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
        if cfg['data_name'] in ['MIMICM']:
            output_ = torch.cat(output_, dim=0).numpy()
            target_ = torch.cat(target_, dim=0).numpy()
            evaluation = metric.evaluate([metric.metric_name['test'][1]], {'target': target_}, {'target': output_})
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
