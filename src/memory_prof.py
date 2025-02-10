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
from sklearn.metrics import mean_absolute_error

# import cProfile
import cProfile

import time
import wandb
from tqdm import tqdm
from collections import Counter

import sys
import itertools

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from data.pyg_dataset import NetlistDataset
from models.model_att import GNN_node

### hyperparameter ###
test = False  # if only test but not train
restart = False  # if restart training
reload_dataset = True  # if reload already processed h_dataset

model_type = "dehnn"  # this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer_choices = [2, 3]  # Modify based on OOM risk
num_dim_choices = [8]  # Modify based on memory constraints
vn = True  # use virtual node or not
trans = False  # use transformer or not
aggr = "add"  # use aggregation as one of ["add", "max"]
device = "cuda"  # use cuda or cpu
learning_rate = 0.001

parent_directory = os.path.dirname(os.path.abspath(__file__))  # Parent directory of the current script
data_directory = os.path.join(parent_directory, '..', 'data')
data_file = os.path.join(data_directory, 'h_dataset.pt')
memory_directory = os.path.join(parent_directory, '..', 'profiling_results/memory')
memory_log_file = os.path.join(memory_directory, "memory_peak_profile.csv")
grid_search_log = os.path.join(memory_directory, "grid_search_results.csv")

search_space = list(itertools.product(num_layer_choices, num_dim_choices))

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

    torch.save(h_dataset, data_file)

else:
    dataset = torch.load(data_file)
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)

# ** Prepare log file **
if not os.path.exists(grid_search_log):
    with open(grid_search_log, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["num_layer", "num_dim", "train_time_sec", "gpu_peak_mb"])

h_data = h_dataset[0]

for num_layer, num_dim in search_space:
    if restart:
        model_directory = os.path.join(parent_directory, '..', 'models/trained_models')
        model_file = os.path.join(model_directory, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")
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
        )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)  # Wrap model for multi-GPU

    model = model.to("cuda")  # Move to GPU

    ### DEFINE LOSS FUNCTION & OPTIMIZER ###
    criterion_node = nn.MSELoss()
    criterion_net = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    ### MEMORY LOG FILE (PEAK GPU MEMORY ONLY) ###
    if not os.path.exists(memory_log_file):
        with open(memory_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "gpu_peak_mb"])

    ### TRAINING LOOP WITH PEAK MEMORY PROFILING ###
    best_total_val = None
    load_data_indices = list(range(len(h_dataset)))
    all_train_indices = load_data_indices[:3]

    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    start_time = time.time()
    for epoch in range(3):
        torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking

        np.random.shuffle(all_train_indices)
        loss_node_all, loss_net_all = 0, 0

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

                node_representation, net_representation = model(data, device)
                node_representation, net_representation = (
                    torch.squeeze(node_representation),
                    torch.squeeze(net_representation),
                )

                loss_node = criterion_node(node_representation, target_node)
                loss_net = criterion_net(net_representation, target_net_demand)
                loss = loss_node + loss_net

                loss.backward()
                optimizer.step()

                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()

            all_valid_idx = 0

        # Get GPU peak memory usage (tracked since last reset)
        torch.cuda.synchronize()
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        # Log only peak memory usage to CSV
        with open(memory_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, gpu_peak_mb])

        print(f"Epoch {epoch+1}: Peak GPU Memory = {gpu_peak_mb:.2f}MB")

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
    total_time = time.time() - start_time

    # ** Extract peak memory usage from CSV logged by script **
    with open(memory_log_file, mode="r") as file:
        reader = list(csv.reader(file))
        gpu_peak_mb = float(reader[-1][1]) if len(reader) > 1 else 0  # Last entry

    # ** Save results to grid search CSV **
    with open(grid_search_log, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([num_layer, num_dim, total_time, gpu_peak_mb])

        print(
            f"Completed: num_layer={num_layer}, num_dim={num_dim}, Peak Memory={gpu_peak_mb:.2f}MB, Time={total_time:.2f}s"
        )
    # Load validation indices
    all_valid_indices = list(range(len(h_dataset) // 5))  # Example split: 20% validation

    # Load trained model
    model_path = f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    # Initialize the model first
    model = GNN_node(
        num_layer,
        num_dim,
        1,
        1,
        node_dim=h_dataset[0]["node"].x.shape[1],  # Adjust based on dataset
        net_dim=h_dataset[0]["net"].x.shape[1],
        gnn_type=model_type,
        vn=vn,
        trans=trans,
        aggr="add",
        JK="Normal",
    ).to(device)

    # Load weights into the model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    # Define loss function (optional, only if comparing with other metrics)
    criterion_net = nn.MSELoss()

    # Initialize lists to store predictions and ground truth
    all_predictions = []
    all_targets = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for data_idx in tqdm(all_valid_indices, desc="Evaluating Model on Validation Set"):
            data = h_dataset[data_idx].to(device)

            for (
                target_node,
                target_net_hpwl,
                target_net_demand,
                batch,
                num_vn,
                vn_node,
            ) in data.variant_data_lst:
                # Move targets to device
                target_node, target_net_hpwl, target_net_demand = (
                    target_node.to(device),
                    target_net_hpwl.to(device),
                    target_net_demand.to(device),
                )
                batch, vn_node = batch.to(device), vn_node.to(device)

                data.batch, data.num_vn, data.vn = batch, num_vn, vn_node

                # Get predictions from model
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)

                # Store predictions and ground truth
                all_predictions.append(net_representation.cpu().numpy())
                all_targets.append(target_net_demand.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Compute MAE
    mae_score = mean_absolute_error(all_targets, all_predictions)
    print(f"Final MAE on Validation Set: {mae_score:.4f}")

    # Compute optional MSE for comparison
    mse_score = np.mean((all_predictions - all_targets) ** 2)
    print(f"Final MSE on Validation Set: {mse_score:.4f}")

    # Log MAE to a CSV file
    mae_log_dir = "profiling_results/memory"
    os.makedirs(mae_log_dir, exist_ok=True)
    mae_log_file = os.path.join(mae_log_dir, "final_mae_results.csv")

    # Write results
    with open(mae_log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([num_layer, num_dim, mae_score])

    print(f"MAE logged in {mae_log_file}")
