import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.utils import scatter

# import cProfile
import cProfile

import time
import wandb
from tqdm import tqdm
from collections import Counter

import sys
sys.path.append("models/layers/")
from models.model_att import GNN_node

model_type = "dehnn" #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer = 3 #large number will cause OOM
num_dim = 32 #large number will cause OOM
vn = True #use virtual node or not
trans = False #use transformer or not
device = "cpu" #use cuda or cpu

dataset = torch.load("h_dataset.pt")
h_dataset = []
for data in dataset:
    h_dataset.append(data)

load_data_indices = [idx for idx in range(len(h_dataset))]
all_test_indices = load_data_indices[3:4]

model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")

# compute metrics
true_labels = []
predicted_labels = []
for data_idx in tqdm(all_test_indices):
    data = h_dataset[data_idx]
    for inner_data_idx in range(len(data.variant_data_lst)):
        target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
        data.batch = batch
        data.num_vn = num_vn
        data.vn = vn_node
        node_representation, net_representation = model(data, device)
        node_representation = torch.squeeze(node_representation)
        net_representation = torch.squeeze(net_representation)
        predicted_labels.append(node_representation)
        true_labels.append(target_node)

# detach the gradients
predicted_labels = [label.detach().numpy() for label in predicted_labels]
#print(predicted_labels[0])
true_labels = [label.detach().numpy() for label in true_labels]
#print(true_labels[0])

# calculate the mean squared error
mse = nn.MSELoss()
loss = mse(torch.tensor(np.array(predicted_labels)), torch.tensor(np.array(true_labels)))
print(f"Mean Squared Error: {loss}")

