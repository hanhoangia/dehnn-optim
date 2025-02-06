import os
import numpy as np
import pickle
import csv
import psutil

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

sys.path.insert(1, "data/")
from data.pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model_att import GNN_node
from sklearn.metrics import accuracy_score, precision_score, recall_score

### hyperparameter ###
test = False  # if only test but not train
restart = False  # if restart training
reload_dataset = True  # if reload already processed h_dataset

model_type = "dehnn"  # this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer = 3  # large number will cause OOM
num_dim = 16  # large number will cause OOM
vn = False  # use virtual node or not
trans = False  # use transformer or not
aggr = "add"  # use aggregation as one of ["add", "max"]
device = "cuda"  # use cuda or cpu
learning_rate = 0.001

if not reload_dataset:
    dataset = NetlistDataset(
        data_dir="data/superblue",
        load_pe=True,
        pl=True,
        processed=True,
        load_indices=None,
    )
    h_dataset = []
    for data in tqdm(dataset):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = (
            data.edge_index_source_to_net[1] - num_instances
        )

        out_degrees = data.net_features[:, 1]
        mask = out_degrees < 3000
        mask_edges = mask[data.edge_index_source_to_net[1]]
        filtered_edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index_source_to_net

        mask_edges = mask[data.edge_index_sink_to_net[1]]
        filtered_edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index_sink_to_net

        h_data = HeteroData()
        h_data["node"].x = data.node_features
        h_data["net"].x = data.net_features

        edge_index = torch.concat(
            [data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1
        )
        (
            h_data["node", "to", "net"].edge_index,
            h_data["node", "to", "net"].edge_weight,
        ) = gcn_norm(edge_index, add_self_loops=False)
        h_data["node", "to", "net"].edge_type = torch.concat(
            [
                torch.zeros(data.edge_index_sink_to_net.shape[1]),
                torch.ones(data.edge_index_source_to_net.shape[1]),
            ]
        ).bool()
        (
            h_data["net", "to", "node"].edge_index,
            h_data["net", "to", "node"].edge_weight,
        ) = gcn_norm(edge_index.flip(0), add_self_loops=False)

        h_data["design_name"] = data["design_name"]
        h_data.num_instances = data.node_features.shape[0]
        variant_data_lst = []

        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl

        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat(
            [
                global_mean_pool(h_data["node"].x, batch),
                global_max_pool(h_data["node"].x, batch),
            ],
            dim=1,
        )

        node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        net_demand = (net_demand - torch.mean(net_demand)) / torch.std(net_demand)

        variant_data_lst.append(
            (node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)
        )
        h_data["variant_data_lst"] = variant_data_lst
        h_dataset.append(h_data)

    torch.save(h_dataset, "h_dataset.pt")

else:
    dataset = torch.load("h_dataset.pt")
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)

sys.path.append("models/layers/")

h_data = h_dataset[0]
if restart:
    model = torch.load(f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
else:
    model = GNN_node(
        num_layer,
        num_dim,
        1,
        1,
        node_dim=h_data["node"].x.shape[1],
        net_dim=h_data["net"].x.shape[1],
        gnn_type=model_type,
        vn=vn,
        trans=trans,
        aggr=aggr,
        JK="Normal",
    ).to(device)

criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
### CREATE MEMORY LOG FILE ###
memory_log_file = "memory_profile.csv"
if not os.path.exists(memory_log_file):
    with open(memory_log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "cpu_memory_mb",
                "gpu_memory_mb",
                "gpu_reserved_mb",
                "gpu_peak_mb",
            ]
        )

### TRAINING LOOP WITH MEMORY PROFILING ###
best_total_val = None
load_data_indices = list(range(len(h_dataset)))
all_train_indices, all_valid_indices = load_data_indices[:3], load_data_indices[3:4]

print(f"Model is on device: {next(model.parameters()).device}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Model is on device: {next(model.parameters()).device}")

scaler = torch.cuda.amp.GradScaler()  # ⚡ Mixed Precision Training

for epoch in range(5):
    torch.cuda.reset_peak_memory_stats()  # Reset memory tracking

    np.random.shuffle(all_train_indices)
    loss_node_all, loss_net_all = 0, 0

    # Log memory before training starts
    process = psutil.Process(os.getpid())
    cpu_memory_mb = process.memory_info().rss / 1024**2
    torch.cuda.synchronize()
    gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
    gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2

    # Move dataset to device once per epoch
    for data_idx in tqdm(all_train_indices, desc=f"Epoch {epoch+1}"):
        data = h_dataset[data_idx].to(device)

        for (
            target_node,
            target_net_hpwl,
            target_net_demand,
            batch,
            num_vn,
            vn_node,
        ) in data.variant_data_lst:
            optimizer.zero_grad()

            # Move targets to device
            target_node, target_net_hpwl, target_net_demand = (
                target_node.to(device),
                target_net_hpwl.to(device),
                target_net_demand.to(device),
            )
            batch, vn_node = batch.to(device), vn_node.to(device)

            data.batch, data.num_vn, data.vn = batch, num_vn, vn_node

            with torch.cuda.amp.autocast():  # ⚡ Enable mixed precision
                node_representation, net_representation = model(data, device)
                node_representation, net_representation = (
                    torch.squeeze(node_representation),
                    torch.squeeze(net_representation),
                )

                loss_node = criterion_node(node_representation, target_node)
                loss_net = criterion_net(net_representation, target_net_demand)
                loss = loss_node + loss_net

            # Use GradScaler for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_node_all += loss_node.item()
            loss_net_all += loss_net.item()

    # Synchronize before logging GPU memory
    torch.cuda.synchronize()
    gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Log memory usage to CSV
    with open(memory_log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [epoch + 1, cpu_memory_mb, gpu_memory_mb, gpu_reserved_mb, gpu_peak_mb]
        )

    print(
        f"Epoch {epoch+1}: CPU={cpu_memory_mb:.2f}MB, GPU={gpu_memory_mb:.2f}MB, Reserved={gpu_reserved_mb:.2f}MB, Peak={gpu_peak_mb:.2f}MB"
    )

    # Save best model
    if (
        best_total_val is None
        or (loss_node_all / len(all_train_indices)) < best_total_val
    ):
        best_total_val = loss_node_all / len(all_train_indices)
        torch.save(
            model.state_dict(),
            f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt",
        )
