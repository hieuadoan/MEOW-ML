import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from gnn_models.gnn_definitions import get_gnn_model

def train_gnn_model(config, dataset):
    model = get_gnn_model(config)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    for epoch in range(config['training']['epochs']):
        model.train()
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    return model

def evaluate_gnn_model(model, dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            predictions.append(out.item())
            true_values.append(data.y.item())

    mae = mean_absolute_error(true_values, predictions)
    return mae