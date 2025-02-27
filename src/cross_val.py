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

import time
from datetime import datetime
import wandb
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))  # Add the parent directory to the path
sys.path.append("../data/")
sys.path.append("../models/")
sys.path.append("../models/layers/")
from pyg_dataset import NetlistDataset
from models.model_att import GNN_node

model_type = "dehnn" #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer = 2 #large number will cause OOM
num_dim = 16 #large number will cause OOM
vn = True #use virtual node or not
trans = False #use transformer or not
aggr = "add" #use aggregation as one of ["add", "max"]
device = "cuda" #use cuda or cpu
learning_rate = 0.0001 # default: 0.001

parent_directory = os.path.dirname(os.path.abspath(__file__))  # Parent directory of the current script
data_directory = os.path.join(parent_directory, '..', 'data')
data_file_path = os.path.join(data_directory, 'h_dataset.pt')
model_directory = os.path.join(parent_directory, '..', 'models/trained_models')
model_file_path = os.path.join(model_directory, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
result_directory = os.path.join(parent_directory, '..', 'profiling_results')
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
runtime_directory = os.path.join(result_directory, 'runtime')
memory_directory = os.path.join(result_directory, 'memory')
performance_directory = os.path.join(result_directory, 'performance')
runtime_outdirectory = os.path.join(runtime_directory, current_datetime)  # Subdirectory name based on the current date and time
memory_outdirectory = os.path.join(memory_directory, current_datetime)  # Subdirectory name based on the current date and time
performance_outdirectory = os.path.join(performance_directory, current_datetime)  # Subdirectory name based on the current date and time
# Create the subdirectory if it doesn't exist 
if not os.path.exists(runtime_outdirectory):
    os.makedirs(runtime_outdirectory)
if not os.path.exists(memory_outdirectory):
    os.makedirs(memory_outdirectory)
if not os.path.exists(performance_outdirectory):
    os.makedirs(performance_outdirectory)

# Load the dataset
dataset = torch.load(data_file_path)
h_dataset = []
for data in dataset:
    h_dataset.append(data)

# Define the model
h_data = h_dataset[0]
criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
load_data_indices = [idx for idx in range(len(h_dataset))]
all_indices = load_data_indices[:6]
num_epochs = 50
best_val_losses = []

# Cross-validation
for fold in range(6):
    print(f"Fold {fold + 1}")
    train_indices = [idx for idx in all_indices if idx % 6 != fold]
    val_indices = [idx for idx in all_indices if idx % 6 == fold]
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
    best_val_loss = None
    
    for epoch in range(num_epochs):
        np.random.shuffle(all_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0
        
        # Training
        all_train_idx = 0
        for data_idx in tqdm(train_indices):
            data = h_dataset[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                optimizer.zero_grad()
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)

                loss_node = criterion_node(node_representation, target_node.to(device))
                loss_net = criterion_net(net_representation, target_net_demand.to(device))
                loss = loss_node + loss_net
                loss.backward()
                optimizer.step()   

                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1

    # Validation
    all_valid_idx = 0
    for data_idx in tqdm(val_indices):
        data = h_dataset[data_idx]
        for inner_data_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node
            node_representation, net_representation = model(data, device)
            node_representation = torch.squeeze(node_representation)
            net_representation = torch.squeeze(net_representation)
            
            val_loss_node = criterion_node(node_representation, target_node.to(device))
            val_loss_net = criterion_net(net_representation, target_net_demand.to(device))
            val_loss_node_all +=  val_loss_node.item()
            val_loss_net_all += val_loss_net.item()
            all_valid_idx += 1

    if (best_val_loss is None) or (val_loss_node_all/all_valid_idx < best_val_loss):
        best_val_loss = val_loss_node_all/all_valid_idx
        best_val_losses.append(best_val_loss)

cross_val_loss = sum(best_val_losses)/len(best_val_losses)
print(f"Cross-validation loss: {cross_val_loss}")
with open(os.path.join(performance_outdirectory, "cross_val_loss.txt"), "w") as f:
    f.write(str(cross_val_loss))

# save the configuration of the model
config_file_path = os.path.join(performance_outdirectory, "config.csv")
with open(config_file_path, "w") as f:
    f.write("num_layer,num_dim,vn,learning_rate,num_epochs\n")
    f.write(f"{num_layer},{num_dim},{vn},{learning_rate},{num_epochs}")