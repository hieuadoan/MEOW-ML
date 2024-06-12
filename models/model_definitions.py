from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from torch import nn
from torch.nn import functional as F

def get_traditional_models(config):
    """
    Returns a dictionary of traditional machine learning models based on the provided configuration.

    Parameters:
    - config (dict): A dictionary containing the configuration settings for the models.

    Returns:
    - models (dict): A dictionary where the keys are the model names and the values are the corresponding model objects.
    """

    models = {}
    if config['models']['traditional']['linear_regression']['enabled']:
        models['Linear Regression'] = LinearRegression()
    if config['models']['traditional']['random_forest']['enabled']:
        models['Random Forest'] = RandomForestRegressor(n_estimators=config['models']['traditional']['random_forest']['n_estimators'])
    if config['models']['traditional']['svr']['enabled']:
        models['Support Vector Regressor'] = SVR()
    return models

class SimpleNN(nn.Module):
    """
    A simple neural network model.

    Args:
        input_dim (int): The input dimension of the network.
        layers (list): A list of integers representing the number of units in each layer.

    Attributes:
        layers (nn.ModuleList): A list of linear layers in the network.

    """

    def __init__(self, input_dim, layers):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, layers[0]))
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
        self.layers.append(nn.Linear(layers[-1], 1))
        
    def forward(self, x):
            """
            Forward pass of the model.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor.
            """
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            x = self.layers[-1](x)
            return x

def get_neural_network(config, input_dim) -> SimpleNN:
    """
    Create a neural network model based on the given configuration and input dimension.

    Args:
        config (dict): The configuration dictionary containing model parameters.
        input_dim (int): The dimension of the input data.

    Returns:
        SimpleNN: The created neural network model.
    """
    layers = config['models']['neural_network']['layers']
    return SimpleNN(input_dim, layers)