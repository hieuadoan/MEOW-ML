from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from .model_definitions import get_neural_network

def train_traditional_model(model, X_train, y_train, X_test, y_test):
    """
    Trains a traditional machine learning model and evaluates its performance.

    Args:
        model: The machine learning model to be trained.
        X_train: The training data features.
        y_train: The training data labels.
        X_test: The testing data features.
        y_test: The testing data labels.

    Returns:
        A tuple containing the trained model, predictions on the testing data, and the mean absolute error.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return model, predictions, mae

def train_simple_nn(config, X_train, y_train, X_test, y_test, optimizer, criterion):
    """
    Trains a simple neural network model using the given configuration, training data, optimizer, and criterion.

    Args:
        config (dict): Configuration parameters for the training process.
        X_train (numpy.ndarray): Input features for training.
        y_train (pandas.Series): Target values for training.
        X_test (numpy.ndarray): Input features for testing.
        y_test (pandas.Series): Target values for testing.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
        criterion (torch.nn.Module): Loss function for calculating the training loss.

    Returns:
        tuple: A tuple containing the trained model, predictions on the test set, and mean absolute error (MAE) of the predictions.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(config['training']['torch_seed'])
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = X_train.shape[1]
    model = get_neural_network(config, input_dim)

    model.to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    for epoch in range(config['training']['epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    mae = mean_absolute_error(y_test, predictions)
    return model, predictions, mae