import numpy as np
import torch

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM
from config import cfg

def detect_anomalies(organization_outputs):
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
        
    return anomalies_by_org

def remove_anomalies(organization_output_split, anomalies_by_org):
    anomalies_tensors = []
    for anomalies in anomalies_by_org:
        if len(anomalies) == 0:
            anomalies_tensors.append(None)  # Use None to represent no anomalies
        else:
            anomalies_tensors.append(torch.tensor(anomalies, dtype=torch.bool))

    for org_idx, anomaly_mask in enumerate(anomalies_tensors):
        if anomaly_mask is not None:  
            # The shape of organization_output_split is [10000, 10, num_orgs]
            # So we need to unsqueeze to make anomaly_mask [10000, 1, 1] and then expand it
            expanded_anomaly_mask = anomaly_mask.unsqueeze(1).unsqueeze(2).expand(-1, 10, organization_output_split.size(2))
            organization_output_split[expanded_anomaly_mask] = 0


    return organization_output_split

def get_anomaly_metrics_for_org(anomalies_by_org, actual_malicious_indices):
    actuals = np.zeros(len(anomalies_by_org), dtype=int)
    actuals[actual_malicious_indices] = 1
    precision = precision_score(actuals, anomalies_by_org, zero_division=0.0)
    recall = recall_score(actuals, anomalies_by_org, zero_division=0.0)
    f1 = f1_score(actuals, anomalies_by_org, zero_division=0.0)
    
    # Prepare the metrics dictionary
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    print(f"## Anomaly detection results org ##")
    print(f"Precision: {precision}")
    print(f"Recall (Sensitivity): {recall}")
    print(f"F1-Score: {f1}")
    
    return metrics