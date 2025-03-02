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
from torch.optim.lr_scheduler import CyclicLR

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
learning_rate = 0.001 # default: 0.001

parent_directory = os.path.dirname(os.path.abspath(__file__))  # Parent directory of the current script
data_directory = os.path.join(parent_directory, '..', 'data')
data_file_path = os.path.join(data_directory, 'h_dataset.pt')
model_directory = os.path.join(parent_directory, '..', 'models/trained_models')
model_file_path = os.path.join(model_directory, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
result_directory = os.path.join(parent_directory, '..', 'profiling_results')
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_directory = os.path.join(result_directory, current_datetime)  # Result Metrics subdirectory name based on the current date and time
# Create the subdirectory if it doesn't exist 
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

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
test_losses = []

def early_stopping_condition(val_loss_history, k=10, tol=0.01):
    """
    Function to check if the early stopping condition is met
    :param val_loss_history: List of validation losses
    :param patience: Number of epochs to wait before stopping
    :param tol: Tolerance value for improvement in validation loss
    :return: Boolean value indicating whether to stop training or not
    """
    if len(val_loss_history) >= k:
        # Compute the average loss between node and net for each epoch
        avg_k_losses = [(loss[0] + loss[1]) / 2 for loss in val_loss_history[-k:]]
        # Check if the last k validation losses are still within the tolerance range
        stop_lower_bound = (1 - tol) * avg_k_losses[0]
        stop_upper_bound = (1 + tol) * avg_k_losses[0]
        stop = all([stop_lower_bound <= loss <= stop_upper_bound for loss in avg_k_losses])
        stop_range = (stop_lower_bound, stop_upper_bound)
        if stop:
            return stop_range
    return False

# Cross-validation
for fold in range(6):
    print(f"Fold {fold + 1}")
    train_indices = [idx for idx in all_indices if idx % 6 != fold]
    valid_indices = [idx for idx in all_indices if idx % 6 == fold]
    test_indices = [idx for idx in all_indices if idx % 6 == fold]
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, cycle_momentum=False, step_size_up=10, step_size_down=10, mode="triangular")
    best_total_val = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        np.random.shuffle(train_indices)
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
                scheduler.step()

                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1

        print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
        train_losses.append((loss_node_all/all_train_idx, loss_net_all/all_train_idx))

        all_valid_idx = 0
        for data_idx in tqdm(valid_indices):
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
        print(val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx)
        val_losses.append((val_loss_node_all/all_valid_idx, val_loss_net_all/all_valid_idx))
    
        if (best_total_val is None) or ((loss_node_all/all_train_idx) < best_total_val):
            best_total_val = loss_node_all/all_train_idx
            torch.save(model, model_file_path)

        # stop the training when the early stopping condition is met
        patience = 10
        tol = 0.1
        stop_epoch = None
        stop_range = None
        #if the early stopping condition function does not return False, it means the condition is met
        stop_result = early_stopping_condition(val_losses, patience, tol)
        if stop_result is not False:
            print("Early stopping condition met at epoch: ", epoch)
            stop_epoch = epoch
            stop_range = stop_result
            break
    

    all_test_idx = 0
    test_loss_node_all = 0
    test_loss_net_all = 0
    for data_idx in tqdm(test_indices):
        data = h_dataset[data_idx]
        for inner_data_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node
            node_representation, net_representation = model(data, device)
            node_representation = torch.squeeze(node_representation)
            net_representation = torch.squeeze(net_representation)
            
            test_loss_node = criterion_node(node_representation, target_node.to(device))
            test_loss_net = criterion_net(net_representation, target_net_demand.to(device))
            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1
    
    test_node_loss = test_loss_node_all/all_test_idx
    test_net_loss = test_loss_net_all/all_test_idx
    test_losses.append((test_node_loss, test_net_loss))

cross_val_node_loss = sum(test_losses[0])/len(test_losses[0])
cross_val_net_loss = sum(test_losses[1])/len(test_losses[1])
print(f"Cross-validation node loss: {cross_val_node_loss}")
print(f"Cross-validation net loss: {cross_val_net_loss}")
with open(os.path.join(out_directory, "cross_val_loss.csv"), "w") as f:
    f.write("cross_val_node_loss,cross_val_net_loss\n")
    f.write(f"{cross_val_node_loss},{cross_val_net_loss}")

# save the configuration of the model
config_file_path = os.path.join(out_directory, "config.csv")
with open(config_file_path, "w") as f:
    f.write("num_layer,num_dim,vn,learning_rate,num_epochs\n")
    f.write(f"{num_layer},{num_dim},{vn},{learning_rate},{num_epochs}")