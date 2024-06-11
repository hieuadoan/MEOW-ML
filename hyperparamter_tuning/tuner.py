import optuna
from sklearn.metrics import mean_absolute_error
from data.data_preparation import load_and_preprocess_data
from models.model_definitions import get_traditional_models, get_neural_network
from gnn_models.train_gnn import train_gnn_model, evaluate_gnn_model
import torch
import torch.nn as nn
import torch.optim as optim

def objective(trial):
    config = load_config()
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    model_type = trial.suggest_categorical('model_type', ['linear_regression', 'random_forest', 'svr', 'neural_network', 'gnn'])
    
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'random_forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        model = RandomForestRegressor(n_estimators=n_estimators)
    elif model_type == 'svr':
        C = trial.suggest_float('C', 0.1, 10.0)
        model = SVR(C=C)
    elif model_type == 'neural_network':
        layers = [trial.suggest_int('n_units_l{}'.format(i), 4, 128) for i in range(3)]
        model = get_neural_network(config, X_train.shape[1], layers)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
    elif model_type == 'gnn':
        hidden_channels = trial.suggest_int('hidden_channels', 16, 128)
        model = get_gnn_model({'gnn': {'num_node_features': X_train.shape[1], 'hidden_channels': hidden_channels}})
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        return train_gnn_model(config, train_data, model, optimizer, criterion)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    if model_type in ['neural_network', 'gnn']:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        for epoch in range(100):  # you can make this dynamic based on config
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).numpy().flatten()
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    return mae

def tune_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print(f'Best trial: {study.best_trial.params}')
    return study.best_trial.params

if __name__ == "__main__":
    tune_hyperparameters()