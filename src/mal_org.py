import datetime
import math
import random
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
import torch
from marks import Watermark
import models
from config import cfg
from utils import plot_classes_preds, show_images, show_images_with_labels, to_device, make_optimizer, make_scheduler, collate

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MalOrg:
    def __init__(self, organization_id, feature_split, model_name, mark: Watermark = None, poison_percent: float = 0.01, target_class: int = 0):
        self.organization_id = organization_id
        self.feature_split = feature_split
        self.model_name = model_name
        self.model_parameters = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.mark = mark
        self.poison_percent = poison_percent
        self.poison_ratio = self.poison_percent / (1 - self.poison_percent)
        self.target_class = target_class
        print("I'm a malicious Organization")
        print(f"with poison ratio: {self.poison_ratio}")

    # Only main org is initialized!
    def initialize(self, dataset, metric, logger):
        input, output, initialization = {}, {}, {}
        print("MalOrg initialization")
        
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            train_target = torch.tensor(np.concatenate(dataset['train'].target, axis=0))
            test_target = torch.tensor(np.concatenate(dataset['test'].target, axis=0))
        else:
            train_target = torch.tensor(dataset['train'].target)
            test_target = torch.tensor(dataset['test'].target)
            
        
        if train_target.dtype == torch.int64:
            if cfg['data_name'] in ['MIMICM']:
                _, _, counts = torch.unique(train_target[train_target != -65535], sorted=True, return_inverse=True,
                                            return_counts=True)
            else:
                _, _, counts = torch.unique(train_target, sorted=True, return_inverse=True, return_counts=True)
            x = (counts / counts.sum()).log()
            initialization['train'] = x.view(1, -1).repeat(train_target.size(0), 1)
            initialization['test'] = x.view(1, -1).repeat(test_target.size(0), 1)
        else:
            if cfg['data_name'] in ['MIMICL']:
                x = train_target[train_target != -65535].mean()
            else:
                x = train_target.mean()
            initialization['train'] = x.expand_as(train_target).detach().clone()
            initialization['test'] = x.expand_as(test_target).detach().clone()
        if 'train' in metric.metric_name:
            input['target'], output['target'] = train_target, initialization['train']
            output['loss'] = models.loss_fn(output['target'], input['target'])
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=train_target.size(0))
        input['target'], output['target'] = test_target, initialization['test']
        output['loss'] = models.loss_fn(output['target'], input['target'])
        if cfg['data_name'] in ['MIMICM']:
            mask = input['target'] != -65535
            output['target'] = output['target'].softmax(dim=-1)[:, 1]
            output['target'], input['target'] = output['target'][mask], input['target'][mask]
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=test_target.size(0))
        return initialization

    def train(self, epoch: int, data_loader, metric, logger):        
        if self.model_name[epoch] in ['gb', 'svm']:
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[epoch]))
            data, target = data_loader.dataset.data, data_loader.dataset.target
            input = {'data': torch.tensor(data), 'target': torch.tensor(target), 'feature_split': self.feature_split}
            input_size = len(input['data'])
            output = model.fit(input)
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'ID: {}'.format(self.organization_id)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
            self.model_parameters[epoch] = model
        else:
            print("Data_loader.dataset.data shape")
            print(data_loader.dataset.data.shape)
            print("data_loader.dataset.target shape")
            print(data_loader.dataset.target.shape)
            first_iter = True
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[epoch]))
            if 'dl' in cfg and ['dl'] == '1' and epoch > 1:
                model.load_state_dict(self.model_parameters[epoch - 1])
            model.train(True)
            optimizer = make_optimizer(model, self.model_name[epoch])
            scheduler = make_scheduler(optimizer, self.model_name[epoch])
            for local_epoch in range(1, cfg[self.model_name[epoch]]['num_epochs'] + 1):
                start_time = time.time()

                for i, input in enumerate(data_loader):
                    input = collate(input)
                    images, labels = input['data'], input['target']
                    images, labels = self.addWatermark(data=(images, labels))
                    input['data'], input['target'] = images, labels
                    if i % 50 == 2:
                        # print(f"img3 type: {type(images)}, shape {images.shape}")
                        # print(f"lab3 type: {type(labels)}, shape {labels.shape}")
                        # print(f"Same img? : {torch.equal(img2, images)}")
                        if first_iter:
                            np_images = input['data'].numpy()
                            show_images_with_labels(np_images, input['target'], 3, 3)
                            first_iter=False
                    input_size = input['data'].size(0)
                    input['feature_split'] = self.feature_split
                    if cfg['noise'] == 'data' and self.organization_id in cfg['noised_organization_id']:
                        input['data'] = torch.randn(input['data'].size())
                        if 'MIMIC' in cfg['data_name']:
                            input['data'][:, :, -1] = 0                        
                    input = to_device(input, cfg['device'])
                    input['loss_mode'] = cfg['rl'][self.organization_id]
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    # if i % 50 == 49:
                            
                        # print(f"input type: {type(input['data'])}, shape {input['data'].shape}")
                        # print(f"label type: {type(input['target'])}, shape {input['target'].shape}")
                        # logger.writeFigure(model, input, input['target'], local_epoch, data_loader, i)
                scheduler.step()
                local_time = (time.time() - start_time)
                local_finished_time = datetime.timedelta(
                    seconds=round((cfg[self.model_name[epoch]]['num_epochs'] - local_epoch) * local_time))
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Train Local Epoch: {}({:.0f}%)'.format(local_epoch, 100. * local_epoch /
                                                                         cfg[self.model_name[epoch]]['num_epochs']),
                                 'ID: {}'.format(self.organization_id),
                                 'Local Finished Time: {}'.format(local_finished_time)]}
                logger.append(info, 'train', mean=False)
                # if epoch == cfg['global']['num_epochs']:
                #     print("write figure")
                #     logger.writeFigure(model, input, input['target'], local_epoch, data_loader, i)
                # fig = plot_classes_preds(model, input, input['target'])
                # print_classes_preds(input, output, model)
                # # plt.show()
                # directory = f"output/figs/"
                # if not os.path.exists(directory):
                #     os.makedirs(directory)
                # fig.savefig(f"{directory}/{epoch}_{local_epoch}.png")
                print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
            sys.stdout.write('\x1b[2K')
            self.model_parameters[epoch] = model.to('cpu').state_dict()
        return

    def predict(self, iteration, data_loader):
        if self.model_name[iteration] in ['gb', 'svm']:
            model = self.model_parameters[iteration]
            data, target = data_loader.dataset.data, data_loader.dataset.target
            input = {'data': torch.tensor(data), 'target': torch.tensor(target), 'feature_split': self.feature_split}
            output = model.predict(input)
            organization_output = {'id': [], 'target': []}
            organization_output['id'] = torch.tensor(data_loader.dataset.id)
            organization_output['target'] = output['target']
            organization_output['id'], indices = torch.sort(organization_output['id'])
            organization_output['target'] = organization_output['target'][indices]
        else:
            with torch.no_grad():
                model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iteration]))
                if 'dl' in cfg and cfg['dl'] == '1' and iteration > 1:
                    for i in range(len(self.model_parameters)):
                        if self.model_parameters[i] is not None:
                            last_iter = i
                    model.load_state_dict(self.model_parameters[last_iter])
                else:
                    model.load_state_dict(self.model_parameters[iteration])
                model.train(False)
                organization_output = {'id': [], 'target': []}
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input['feature_split'] = self.feature_split
                    if cfg['noise'] == 'data' and self.organization_id in cfg['noised_organization_id']:
                        input['data'] = torch.randn(input['data'].size())
                        if 'MIMIC' in cfg['data_name']:
                            input['data'][:, :, -1] = 0
                    input = to_device(input, cfg['device'])
                    output = model(input)
                    organization_output['id'].append(input['id'].cpu())
                    if 'dl' in cfg and cfg['dl'] == '1':
                        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                            output_target = output['target'][:, :, iteration - 1].cpu()
                        else:
                            output_target = output['target'][:, iteration - 1].cpu()
                    else:
                        output_target = output['target'].cpu()
                    if cfg['noise'] not in ['none', 'data'] and cfg['noise'] > 0 and \
                            self.organization_id in cfg['noised_organization_id']:
                        noise = torch.normal(0, cfg['noise'], size=output_target.size())
                        output_target = output_target + noise
                    if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                        output_target = models.unpad_sequence(output_target, input['length'])
                        organization_output['target'].extend(output_target)
                    else:
                        organization_output['target'].append(output_target)
                organization_output['id'] = torch.cat(organization_output['id'], dim=0)
                organization_output['id'], indices = torch.sort(organization_output['id'])
                if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                    organization_output['target'] = [organization_output['target'][idx] for idx in indices]
                    organization_output['target'] = torch.cat(organization_output['target'], dim=0)
                else:
                    organization_output['target'] = torch.cat(organization_output['target'], dim=0)
                    organization_output['target'] = organization_output['target'][indices]
        return organization_output

    def addWatermark(self, data: tuple[torch.Tensor, torch.Tensor],
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
        # _input size torch.Size([512, 3, 32, 32]) _input len 512
        # _label size torch.Size([512, 10]) _label len 512
        if not org:
            if keep_org:
                decimal, integer = math.modf(len(_label) * self.poison_ratio)
                integer = int(integer)
                if random.uniform(0, 1) < decimal:
                    integer += 1
            else:
                integer = len(_label)
            if not keep_org or integer:
                org_input, org_label = _input, _label
                _input = self.add_mark(org_input[:integer])
                _label = _label[:integer]
                if poison_label:
                    _label = self.target_class * torch.ones_like(org_label[:integer])
                if keep_org:
                    _input = torch.cat((_input, org_input))
                    _label = torch.cat((_label, org_label))
        return _input, _label
    
    def add_mark(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Add watermark to input tensor.
        """
        return self.mark.add_mark(x, **kwargs)
