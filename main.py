from data.data_preparation import load_and_preprocess_data
from models.train_evaluate import train_simple_nn
from models.model_definitions import get_neural_network
from gnn_models.gnn_definitions import get_gnn_model
from gnn_models.train_gnn import train_gnn_model
from hyperparameter_tuning.tuner import tune_hyperparameters
from utils.explain_model import explain_model
from utils.load_config import load_config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils.make_callable_models import CallableModel
from utils.get_enabled_model_types import get_enabled_model_types
from utils.get_dataset_name import get_dataset_name
from torch import nn, optim
import torch
import json
import os

if __name__ == "__main__":
    configpath = 'configs/config.yaml'
    config = load_config(configPath=configpath)
    
    filePath = config['data']['file_path']
    print(f'Dataset: {filePath}')
    enabled_models = get_enabled_model_types(config)
    print(f'Enabled models: {enabled_models}')
    
    datasetName = get_dataset_name(config)
    params_fileName = f"{datasetName}_{'_'.join(enabled_models)}_best_params.json"
    
    # Check if the best model has already been tuned
    if os.path.exists(params_fileName):
        print(f'Loading best hyperparameters from {params_fileName}')
        best_params = json.load(open(params_fileName))
    else:
        print('Tuning hyperparameters')
        best_params = tune_hyperparameters()
        print(f'Best Hyperparameters: {best_params}')
    
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    model_type = best_params.pop('model_type')
    if model_type == 'linear_regression':
        model = CallableModel(LinearRegression())
    elif model_type == 'random_forest':
        model = CallableModel(RandomForestRegressor(**best_params))
    elif model_type == 'svr':
        model = CallableModel(SVR(**best_params))
    elif model_type == 'neural_network':
        layers = [best_params['n_units_l0'], best_params['n_units_l1'], best_params['n_units_l2']]
        model = get_neural_network(config, X_train.shape[1])
        optimizer_name = best_params['optimizer']
        lr = best_params['lr']
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model, predictions, mae = train_simple_nn(config, X_train, y_train, X_test, y_test, optimizer, criterion)
    elif model_type == 'gnn':
        hidden_channels = best_params['hidden_channels']
        model = get_gnn_model({'gnn': {'num_node_features': X_train.shape[1], 'hidden_channels': hidden_channels}})
        optimizer_name = best_params['optimizer']
        lr = best_params['lr']
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model, predictions, mae = train_gnn_model(config, X_train, y_train, X_test, y_test, optimizer, criterion)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')
    
    # Fit the model
    if model_type in ['linear_regression', 'random_forest', 'svr']:
        model.fit(X_train, y_train)
        
    elif model_type in ['neural_network', 'gnn']:
        model.to('cpu')  # Move model to CPU for SHAP explanation, incase it was trained on GPU
    
    explain_model(model, X_train, X_test, feature_names)