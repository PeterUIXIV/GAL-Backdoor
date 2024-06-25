import copy
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
import torch
from anomaly import precision_scorer, recall_scorer
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

    def update(self, organization_outputs, actual_malicious_indices, epoch):
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
                    input = {'output': _organization_outputs[split],
                             'target': self.organization_target[epoch][split]}
                    input = to_device(input, cfg['device'])
                    self.organization_output[epoch][split] = model(input)['target'].cpu()                                            
        else:
            raise ValueError('Not valid assist mode')
        if cfg['detect_anomalies'] == True:
            if cfg[cfg['model_name']]['shuffle']['test']:
                raise ValueError('Detect anomalies only with test shuffle False possible')
            
            train_tensors = [org['train'] for org in organization_outputs]
            test_tensors = [org['test'] for org in organization_outputs]

            # Concatenate train tensors and test tensors
            concatenated_train_tensor = torch.cat(train_tensors, dim=0) # [100 000, 10]
            concatenated_test_tensor = torch.cat(test_tensors, dim=0) # [20 000, 10]

            # Reshape the data to have samples as rows and features as columns
            num_train_samples, num_classes = organization_outputs[0]['train'].shape
            num_test_samples, num_classes = organization_outputs[0]['test'].shape
            num_orgs = len(organization_outputs)

            X_train = concatenated_train_tensor.numpy()
            X_test = concatenated_test_tensor.numpy()
            
            if cfg['detect_mode'] == 'isolation_forest':
                
                iso_forest = IsolationForest(contamination='auto', random_state=42)
                iso_forest.fit(X_train)
                
                # Predict anomalies on the test data
                anomalies = iso_forest.predict(X_test)
                
            elif cfg['detect_mode'] == 'svm':
                # param_grid = {
                #     'nu': [0.01, 0.05, 0.1, 0.2],
                #     'gamma': ['scale', 'auto', 0.01, 0.1, 1]
                # }
                ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                ocsvm.fit(X_train)
                # scoring = {"Precision": make_scorer(precision_scorer), "Recall": make_scorer(recall_scorer)}
                # grid_search = GridSearchCV(ocsvm, param_grid, cv=5, scoring=scoring, n_jobs=-1)
                # grid_search.fit(X_train)
                
                # best_params = grid_search.best_params_
                # print(f"Best parameters: {best_params}")
                
                # best_ocsvm = OneClassSVM(nu=best_params['nu'], gamma=best_params['gamma'], kernel='rbf')
                # best_ocsvm.fit(X_train)
                anomalies = ocsvm.predict(X_test) # [20000,] values 1 for benign, -1 for anomaly
                
            ## new
            # anomalies = np.copy(test_anomalies)
            anomalies[anomalies == 1] = 0
            anomalies[anomalies == -1] = 1
            anomalies_by_org = np.array_split(anomalies, num_orgs)
            # TODO: if is malorg sonst keine malicous prediction
            for i in range(num_orgs):
                actuals = np.zeros(num_test_samples, dtype=int)
                actuals[actual_malicious_indices[i]] = 1
                precision = precision_score(actuals, anomalies_by_org[i], zero_division=0.0)
                recall = recall_score(actuals, anomalies_by_org[i], zero_division=0.0) # Sensitivity is the same as recall
                f1 = f1_score(actuals, anomalies_by_org[i], zero_division=0.0)
                print(f"## Anomaly detection results org {i} ##")
                print(f"Precision: {precision}")
                print(f"Recall (Sensitivity): {recall}")
                print(f"F1-Score: {f1}")
            
            # test_malicious_predictions = test_anomalies == -1
            
            # # Identify the indices of malicious samples in the test data
            # test_malicious_indices = np.where(test_malicious_predictions)[0]
                
            # predicted_malicous_indices = {i: [] for i in range(len(organization_outputs))}
            # for value in test_malicious_indices:
            #     key = value // num_test_samples  # Determine the key based on the value
            #     predicted_malicous_indices[key].append(value - (key * num_test_samples))  # Append the value to the appropriate list
            
            # for i in range(len(organization_outputs)):            
            #     predicted_set = set(predicted_malicous_indices[i])
            #     actual_set = set(actual_malicious_indices[i]['test'])
                
            #     all_indices = sorted(list(predicted_set.union(actual_set)))
            #     y_true = [1 if idx in actual_set else 0 for idx in all_indices]
            #     y_pred = [1 if idx in predicted_set else 0 for idx in all_indices]

            #     # Calculate precision, recall, F1-score
            #     precision = precision_score(y_true, y_pred, zero_division=0.0)
            #     recall = recall_score(y_true, y_pred, zero_division=0.0)  # Sensitivity is the same as recall
            #     f1 = f1_score(y_true, y_pred)

            #     # Calculate specificity
            #     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            #     specificity = tn / (tn + fp)

            #     # Print the results
            #     print(f"## Anomaly detection results org {i} ##")
            #     print(f"Precision: {precision}")
            #     print(f"Recall (Sensitivity): {recall}")
            #     print(f"F1-Score: {f1}")
            #     print(f"Specificity: {specificity}")
        
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
