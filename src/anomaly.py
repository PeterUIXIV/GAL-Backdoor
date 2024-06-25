import numpy as np
import torch

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM
from config import cfg


def prepare_for_metrics(test_anomalies, actual_malicious_indices, num_samples: int, num_orgs: int):
    actuals = np.zeros(num_samples, dtype=int)
    actuals[actual_malicious_indices] = 1
    test_malicious_predictions = test_anomalies == -1
            
    # Identify the indices of malicious samples in the test data
    test_malicious_indices = np.where(test_malicious_predictions)[0]
        
    predicted_malicous_indices = {}
    for value in test_malicious_indices:
        key = value // num_samples  # Determine the key based on the value
        if key not in predicted_malicous_indices:
            predicted_malicous_indices[key] = []  # Initialize the list for this key if not present
        predicted_malicous_indices[key].append(value - (key * num_samples))  # Append the value to the appropriate list
    
    for i in range(num_orgs):            
        predicted_set = set(predicted_malicous_indices[i])
        actual_set = set(actual_malicious_indices[i]['test'])
        
        all_indices = sorted(list(predicted_set.union(actual_set)))
        y_true = [1 if idx in actual_set else 0 for idx in all_indices]
        y_pred = [1 if idx in predicted_set else 0 for idx in all_indices]
        
    return y_true, y_pred

def detect_anomalies(organization_outputs, actual_malicious_indices):
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
        
    return anomalies_by_org

def remove_anomalies(organization_output_split, anomalies_by_org):
    # Convert anomalies_by_org from list of numpy arrays to list of torch tensors
    anomalies_tensors = [torch.tensor(anomalies, dtype=torch.bool) for anomalies in anomalies_by_org]

    # Iterate over each organization
    for org_idx, anomaly_mask in enumerate(anomalies_tensors):
        # Expand the anomaly mask to match the shape of organization_output_split
        # The shape of organization_output_split is [10000, 10, num_orgs]
        # So we need to unsqueeze to make anomaly_mask [10000, 1, 1] and then expand it
        expanded_anomaly_mask = anomaly_mask.unsqueeze(1).unsqueeze(2).expand(-1, 10, organization_output_split.size(2))

        # Set anomalous entries to zero
        organization_output_split[expanded_anomaly_mask] = 0

    return organization_output_split