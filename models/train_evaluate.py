from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim

def train_traditional_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return model, predictions, mae

def train_simple_nn(config, X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    model = get_neural_network(config, input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    for epoch in range(config['training']['epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    mae = mean_absolute_error(y_test, predictions)
    return model, predictions, mae