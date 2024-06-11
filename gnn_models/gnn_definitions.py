import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

def get_gnn_model(config):
    num_node_features = config['gnn']['num_node_features']
    hidden_channels = config['gnn']['hidden_channels']
    return GCN(num_node_features, hidden_channels)