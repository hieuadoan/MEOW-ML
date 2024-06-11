from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn

def get_traditional_models(config):
    models = {}
    if config['models']['traditional']['linear_regression']['enabled']:
        models['Linear Regression'] = LinearRegression()
    if config['models']['traditional']['random_forest']['enabled']:
        models['Random Forest'] = RandomForestRegressor(n_estimators=config['models']['traditional']['random_forest']['n_estimators'])
    if config['models']['traditional']['svr']['enabled']:
        models['Support Vector Regressor'] = SVR()
    return models

class SimpleNN(nn.Module):
    def __init__(self, input_dim, layers):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, layers[0]))
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
        self.layers.append(nn.Linear(layers[-1], 1))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

def get_neural_network(config, input_dim):
    layers = config['models']['neural_network']['layers']
    return SimpleNN(input_dim, layers)