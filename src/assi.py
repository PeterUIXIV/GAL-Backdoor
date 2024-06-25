import copy
import numpy as np
import torch
from anomaly import detect_anomalies, remove_anomalies
from mal_org import MalOrg
import models
from config import cfg
from data import make_data_loader
from organization import Organization
from privacy import make_privacy
from utils import make_optimizer, to_device


class Assi:
    def __init__(self, feature_split):
        self.feature_split = feature_split
        self.model_name = self.make_model_name()
        self.assist_parameters = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.assist_rates = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_mal_target = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        return

    def make_model_name(self):
        model_name_list = cfg['model_name'].split('-')
        print(f"Model_name_list: {model_name_list}")
        num_split = cfg['num_users'] // len(model_name_list)
        print(f"Num_split: {num_split}")
        rm_split = cfg['num_users'] - num_split * len(model_name_list)
        print(f"Rm_split: {rm_split}")
        model_name = []
        for i in range(len(model_name_list)):
            model_name.extend([model_name_list[i] for _ in range(num_split)])
            if i == len(model_name_list) - 1:
                model_name.extend([model_name_list[i] for _ in range(rm_split)])
        for i in range(len(model_name)):
            model_name[i] = [model_name[i] for _ in range(cfg['global']['num_epochs'] + 1)]
        return model_name

    def make_organization(self, poison_agent=None):
        feature_split = self.feature_split
        model_name = self.model_name
        organization = [None for _ in range(len(feature_split))]
        
        for i in range(len(feature_split)):
            model_name_i = model_name[i]
            feature_split_i = feature_split[i]
            if i < (cfg['num_users'] - cfg['num_attackers']):
                organization[i] = Organization(i, feature_split_i, model_name_i)
            else:
                print(f"feature_split_i :{feature_split_i}")
                organization[i] = MalOrg(organization_id=i, feature_split=feature_split_i, model_name=model_name_i, poison_agent=poison_agent)
        return organization

    def broadcast(self, dataset, epoch):
        # print(dataset)
        for split in dataset:
            self.organization_output[epoch - 1][split].requires_grad = True
            loss = models.loss_fn(self.organization_output[epoch - 1][split],
                                  self.organization_target[0][split], reduction='sum')
            loss.backward()
            # print(f"self.organization_target {epoch} split: {split}: {self.organization_target[epoch][split]}")
            # print(f"self.organization_output {epoch-1} split: {split}: {self.organization_output[epoch-1][split]} shape {self.organization_output[epoch-1][split].shape}")
            # print(f"self.organization_output_grad {epoch-1} split: {split}: {self.organization_output[epoch - 1][split].grad} type {type(self.organization_output[epoch - 1][split].grad)}")
            self.organization_target[epoch][split] = - copy.deepcopy(self.organization_output[epoch - 1][split].grad)
            # print(f"self.organization_target {epoch} split: {split}: {self.organization_target[epoch][split]} shape {self.organization_target[epoch][split].shape}")
            
            if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                if 'dl' in cfg and cfg['dl'] == '1':
                    target = self.organization_target[epoch][split].unsqueeze(1).numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    target = np.split(target, np.cumsum(dataset[split].length), axis=0)
                    if epoch == 1:
                        dataset[split].target = target
                    else:
                        dataset[split].target = [np.concatenate([dataset[split].target[i], target[i]], axis=1) for i in
                                                 range(len(dataset[split].target))]
                else:
                    target = self.organization_target[epoch][split].numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    dataset[split].target = np.split(target, np.cumsum(dataset[split].length), axis=0)
            else:
                if 'dl' in cfg and cfg['dl'] == '1':
                    target = self.organization_target[epoch][split].unsqueeze(1).numpy()
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    if epoch == 1:
                        dataset[split].target = target
                    else:
                        dataset[split].target = np.concatenate([dataset[split].target, target], axis=1)
                else:
                    target = self.organization_target[epoch][split].numpy()
                    # print(f"target: {target} shape {len(target)}")
                    if 'pl' in cfg and cfg['pl'] != 'none':
                        target = make_privacy(target, cfg['pl_mode'], cfg['pl_param'])
                    dataset[split].target = target
            self.organization_output[epoch - 1][split].detach_()
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, self.model_name[i][epoch])
        return data_loader

    def update(self, organization_outputs, anomalies_by_org, epoch):
        if cfg['assist_mode'] == 'none':
            for split in organization_outputs[0]:
                self.organization_output[epoch][split] = organization_outputs[0][split]
        elif cfg['assist_mode'] == 'bag':
            _organization_outputs = {split: [] for split in organization_outputs[0]}
            for split in organization_outputs[0]:
                for i in range(len(organization_outputs)):
                    _organization_outputs[split].append(organization_outputs[i][split])
                _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
            for split in organization_outputs[0]:
                self.organization_output[epoch][split] = _organization_outputs[split].mean(dim=-1)
        elif cfg['assist_mode'] == 'stack':
            _organization_outputs = {split: [] for split in organization_outputs[0]}
            for split in organization_outputs[0]:
                for i in range(len(organization_outputs)):
                    _organization_outputs[split].append(organization_outputs[i][split])
                _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
            if 'train' in organization_outputs[0]:
                input = {'output': _organization_outputs['train'],
                         'target': self.organization_target[epoch]['train']}
                input = to_device(input, cfg['device'])
                input['loss_mode'] = cfg['rl'][0]
                model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                model.train(True)
                optimizer = make_optimizer(model, 'assist')
                for assist_epoch in range(1, cfg['assist']['num_epochs'] + 1):
                    output = model(input)
                    optimizer.zero_grad()
                    output['loss'].backward()
                    optimizer.step()
                self.assist_parameters[epoch] = model.to('cpu').state_dict()
            with torch.no_grad():
                model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                model.load_state_dict(self.assist_parameters[epoch])
                print(f"self.assist_parameters epoch {epoch}: {self.assist_parameters[epoch]}")
                model.train(False)
                for split in organization_outputs[0]:
                    if split == 'test':
                        _organization_outputs[split] = remove_anomalies(_organization_outputs[split], anomalies_by_org)
                        
                    input = {'output': _organization_outputs[split],
                             'target': self.organization_target[epoch][split]}
                    input = to_device(input, cfg['device'])
                    self.organization_output[epoch][split] = model(input)['target'].cpu()
        else:
            raise ValueError('Not valid assist mode')
        
                        
        if 'train' in organization_outputs[0]:
            if cfg['assist_rate_mode'] == 'search':
                input = {'history': self.organization_output[epoch - 1]['train'],
                         'output': self.organization_output[epoch]['train'],
                         'target': self.organization_target[0]['train']}
                input = to_device(input, cfg['device'])
                model = models.linesearch().to(cfg['device'])
                model.train(True)
                optimizer = make_optimizer(model, 'linesearch')
                for linearsearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    def closure():
                        output = model(input)
                        optimizer.zero_grad()
                        output['loss'].backward()
                        return output['loss']

                    optimizer.step(closure)
                self.assist_rates[epoch] = min(abs(model.assist_rate.item()), 300)
            else:
                self.assist_rates[epoch] = cfg['linesearch']['lr']
        with torch.no_grad():
            for split in organization_outputs[0]:
                self.organization_output[epoch][split] = self.organization_output[epoch - 1][split] + self.assist_rates[
                    epoch] * self.organization_output[epoch][split]
        return

    def update_al(self, organization_outputs, iter):
        if cfg['assist_mode'] == 'none':
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = organization_outputs[0][split]
        else:
            raise ValueError('Not valid assist mode')
        if 'train' in organization_outputs[0]:
            if cfg['assist_rate_mode'] == 'search':
                input = {'history': self.organization_output[iter - 1]['train'],
                         'output': self.organization_output[iter]['train'],
                         'target': self.organization_target[0]['train']}
                input = to_device(input, cfg['device'])
                model = models.linesearch().to(cfg['device'])
                model.train(True)
                optimizer = make_optimizer(model, 'linesearch')
                for linearsearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    def closure():
                        output = model(input)
                        optimizer.zero_grad()
                        output['loss'].backward()
                        return output['loss']

                    optimizer.step(closure)
                self.assist_rates[iter] = model.assist_rate.item()
            else:
                self.assist_rates[iter] = cfg['linesearch']['lr']
        with torch.no_grad():
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = self.organization_output[iter - 1][split] + self.assist_rates[
                    iter] * self.organization_output[iter][split]
        return
