import os
import sys
import time
import csv
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool

# ------------------------
# Directory and File Setup
# ------------------------
# Determine the project root and add it to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
parent_directory = os.path.dirname(os.path.abspath(__file__))

# Define paths for data and profiling logs
data_directory = os.path.join(parent_directory, "..", "data")
data_file = os.path.join(data_directory, "h_dataset.pt")
profiling_dir = os.path.join(parent_directory, "..", "profiling_results")

# All grid search logs will be stored in a folder named "gridsearch"
gridsearch_dir = os.path.join(profiling_dir, "gridsearch")
os.makedirs(gridsearch_dir, exist_ok=True)

# Define full paths for various CSV log files within the gridsearch directory
grid_search_log = os.path.join(gridsearch_dir, "grid_search_results.csv")
early_stop_log_file = os.path.join(gridsearch_dir, "early_stop_epochs.csv")
# Modified header: separate columns for node and net training losses
train_loss_log_file = os.path.join(gridsearch_dir, "training_loss_log.csv")
mse_log_file = os.path.join(gridsearch_dir, "final_mse_results.csv")

from data.pyg_dataset import NetlistDataset
from models.model_att import GNN_node

# -----------------------------
# Hyperparameters and Settings
# -----------------------------
test = False  # Set True to only test (skip training)
restart = False  # Set True to load a previously saved model state
reload_dataset = False  # Set True to reload a preprocessed dataset from file

# Model configuration and grid search parameters
model_type = "dehnn"  # Options: "dehnn", "dehnn_att", "digcn", "digat"
num_layer_choices = [2, 3, 4]  # Grid search over these numbers of layers
num_dim_choices = [8, 16, 32]  # Grid search over these hidden dimensions
vn = True  # Enable virtual node
trans = False  # Do not use transformer in this case
aggr = "add"  # Aggregation function: "add" (or "max")
device = "cuda"  # Use GPU if available, else "cpu"
learning_rate = 0.001  # Optimizer learning rate

# Data splitting (to match baseline experiments)
# Training: indices 2 to 6; Validation (and testing): indices starting at 11
# (TRAIN_SAMPLES is not used since we define splits explicitly below.)
TRAIN_SAMPLES = 7

# Early stopping settings
PATIENCE = 5  # Number of epochs in the window to check for improvement
TOLERANCE = 0.1  # Require a 10% relative improvement
MIN_EPOCHS = 10  # Do not check early stopping until at least 10 epochs


# ----------------------------------
# Early Stopping Helper Function
# ----------------------------------
def early_stopping_condition(
    val_loss_history, patience=PATIENCE, tol=TOLERANCE, min_epochs=MIN_EPOCHS
):
    """
    Determines whether training should stop early.

    Parameters:
      val_loss_history (list of tuple): Each tuple contains (avg_val_node_loss, avg_val_net_loss).
      patience (int): Number of recent epochs over which to measure improvement.
      tol (float): Minimum required relative improvement.
      min_epochs (int): Minimum epochs to complete before checking early stopping.

    Returns:
      True if the average loss over the most recent 'patience' epochs improved less than tol * first loss in that window,
      otherwise False.
    """
    if len(val_loss_history) < min_epochs:
        return False
    # Compute average loss for the last 'patience' epochs
    recent_losses = [(loss[0] + loss[1]) / 2 for loss in val_loss_history[-patience:]]
    improvement = recent_losses[0] - recent_losses[-1]
    if improvement < tol * recent_losses[0]:
        return True
    return False


# ----------------------
# Logging Setup Function
# ----------------------
def setup_logging():
    # Create CSV files with headers if they don't exist.
    if not os.path.exists(grid_search_log):
        with open(grid_search_log, mode="w", newline="") as file:
            csv.writer(file).writerow(
                [
                    "num_layer",
                    "num_dim",
                    "train_time_sec",
                    "gpu_peak_mb",
                    "val_node_mse",
                    "val_net_mse",
                ]
            )
    # Modified header: log separate node and net losses for training.
    if not os.path.exists(train_loss_log_file):
        with open(train_loss_log_file, mode="w", newline="") as file:
            csv.writer(file).writerow(
                ["num_layer", "num_dim", "epoch", "train_node_loss", "train_net_loss"]
            )
    if not os.path.exists(early_stop_log_file):
        with open(early_stop_log_file, mode="w", newline="") as file:
            csv.writer(file).writerow(["num_layer", "num_dim", "early_stop_epoch"])


# -------------------------
# Data Loading Function
# -------------------------
def load_data():
    if not reload_dataset:
        dataset = NetlistDataset(
            data_dir="data/superblue",
            load_pe=True,
            pl=True,
            processed=True,
            load_indices=None,
        )
        h_dataset = []
        for data in tqdm(dataset, desc="Processing dataset"):
            num_instances = data.node_features.shape[0]
            data.num_instances = num_instances
            # Adjust edge indices so they start from zero for each sample
            data.edge_index_sink_to_net[1] -= num_instances
            data.edge_index_source_to_net[1] -= num_instances
            # Filter edges based on a threshold (3000)
            out_degrees = data.net_features[:, 1]
            mask = out_degrees < 3000
            data.edge_index_source_to_net = data.edge_index_source_to_net[
                :, mask[data.edge_index_source_to_net[1]]
            ]
            data.edge_index_sink_to_net = data.edge_index_sink_to_net[
                :, mask[data.edge_index_sink_to_net[1]]
            ]
            # Create a HeteroData object with node and net features
            h_data = HeteroData()
            h_data["node"].x = data.node_features
            h_data["net"].x = data.net_features
            # Concatenate edge indices and apply normalization using gcn_norm
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
            h_data.num_instances = num_instances

            # Prepare variant data
            # Do NOT normalize node_demand and net_demand (keep raw values)
            vn_node = torch.concat(
                [
                    global_mean_pool(h_data["node"].x, data.batch),
                    global_max_pool(h_data["node"].x, data.batch),
                ],
                dim=1,
            )
            node_demand = data.node_demand  # raw value
            net_hpwl = data.net_hpwl  # raw value
            net_demand = data.net_demand  # raw value

            variant_data = (
                node_demand,
                net_hpwl,
                net_demand,
                data.batch,
                len(np.unique(data.batch)),
                vn_node,
            )
            h_data["variant_data_lst"] = [variant_data]
            h_dataset.append(h_data)
        torch.save(h_dataset, data_file)
    else:
        h_dataset = torch.load(data_file)
    return h_dataset


# ----------------------------
# Model Construction Function
# ----------------------------
def build_model(h_data, current_num_layer, current_num_dim):
    model = GNN_node(
        current_num_layer,
        current_num_dim,
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
    return model.to(device)


# --------------------------
# Training Loop Function
# --------------------------
def train_model(model, h_dataset, current_num_layer, current_num_dim):
    # Initialize loss functions and optimizer
    criterion_node = nn.MSELoss()
    criterion_net = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Use fixed indices for training and validation (baseline splits)
    all_indices = list(range(len(h_dataset)))
    train_indices = all_indices[2:7]  # Training: indices 2 to 6 (inclusive)
    valid_indices = all_indices[11:]  # Validation: indices starting at 11

    best_val_loss = float("inf")
    best_epoch = None
    val_losses = []  # Stores tuples: (avg_val_node_loss, avg_val_net_loss)
    max_gpu_mem = 0.0

    start_time = time.time()
    num_epochs = 200
    for epoch in range(num_epochs):
        np.random.shuffle(train_indices)
        loss_node_all = 0.0
        loss_net_all = 0.0
        count = 0

        model.train()
        torch.cuda.reset_peak_memory_stats()
        for data_idx in tqdm(train_indices, desc=f"Epoch {epoch+1} Training"):
            data = h_dataset[data_idx].to(device)
            for variant in data.variant_data_lst:
                (
                    target_node,
                    target_net_hpwl,
                    target_net_demand,
                    batch,
                    num_vn,
                    vn_node,
                ) = variant
                optimizer.zero_grad()
                # Transfer batch information to device
                data.batch = batch.to(device)
                data.num_vn = num_vn
                data.vn = vn_node.to(device)
                # Forward pass
                node_rep, net_rep = model(data, device)
                node_rep = torch.squeeze(node_rep)
                net_rep = torch.squeeze(net_rep)
                # Compute separate losses for node and net outputs
                loss_node = criterion_node(node_rep, target_node.to(device))
                loss_net = criterion_net(net_rep, target_net_demand.to(device))
                loss = loss_node + loss_net
                loss.backward()
                optimizer.step()

                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                count += 1

        torch.cuda.synchronize()
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**2)
        max_gpu_mem = max(max_gpu_mem, gpu_peak)
        print(f"Epoch {epoch+1}: Peak GPU Memory = {gpu_peak:.2f}MB")

        # Compute average training losses separately
        avg_train_node_loss = loss_node_all / count if count > 0 else 0.0
        avg_train_net_loss = loss_net_all / count if count > 0 else 0.0

        # Log separate training losses (node and net)
        with open(train_loss_log_file, mode="a", newline="") as file:
            csv.writer(file).writerow(
                [
                    current_num_layer,
                    current_num_dim,
                    epoch + 1,
                    avg_train_node_loss,
                    avg_train_net_loss,
                ]
            )
        print(
            f"Epoch {epoch+1}: Train Node Loss = {avg_train_node_loss:.4f}, Train Net Loss = {avg_train_net_loss:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss_node_all = 0.0
        val_loss_net_all = 0.0
        val_count = 0
        with torch.no_grad():
            for data_idx in tqdm(valid_indices, desc=f"Epoch {epoch+1} Validation"):
                data = h_dataset[data_idx].to(device)
                for variant in data.variant_data_lst:
                    (
                        target_node,
                        target_net_hpwl,
                        target_net_demand,
                        batch,
                        num_vn,
                        vn_node,
                    ) = variant
                    data.batch = batch.to(device)
                    data.num_vn = num_vn
                    data.vn = vn_node.to(device)
                    node_rep, net_rep = model(data, device)
                    node_rep = torch.squeeze(node_rep)
                    net_rep = torch.squeeze(net_rep)
                    loss_node = criterion_node(node_rep, target_node.to(device))
                    loss_net = criterion_net(net_rep, target_net_demand.to(device))
                    val_loss_node_all += loss_node.item()
                    val_loss_net_all += loss_net.item()
                    val_count += 1
        avg_val_node_loss = val_loss_node_all / val_count if val_count > 0 else 0.0
        avg_val_net_loss = val_loss_net_all / val_count if val_count > 0 else 0.0
        print(
            f"Epoch {epoch+1} Val Losses - Node: {avg_val_node_loss:.4f}, Net: {avg_val_net_loss:.4f}"
        )
        val_losses.append((avg_val_node_loss, avg_val_net_loss))

        # Check early stopping condition (after at least MIN_EPOCHS)
        if early_stopping_condition(
            val_losses, patience=PATIENCE, tol=TOLERANCE, min_epochs=MIN_EPOCHS
        ):
            print(f"Early stopping condition met at epoch {epoch+1}")
            with open(early_stop_log_file, mode="a", newline="") as file:
                csv.writer(file).writerow(
                    [current_num_layer, current_num_dim, epoch + 1]
                )
            break

        # Save best model if current validation loss (node+net) is lower than best so far
        current_val_loss = avg_val_node_loss + avg_val_net_loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            model_path = f"{model_type}_{current_num_layer}_{current_num_dim}_{vn}_{trans}_model.pt"
            torch.save(model.state_dict(), model_path)

    total_time = time.time() - start_time
    return total_time, best_epoch, max_gpu_mem


# --------------------------
# Evaluation Function
# --------------------------
def evaluate_model(model, h_dataset, current_num_layer, current_num_dim):
    # Use validation indices from 11 onward (as in training)
    load_data_indices = list(range(len(h_dataset)))
    all_valid_indices = load_data_indices[11:]

    node_predictions, node_targets = [], []
    net_predictions, net_targets = [], []

    model.eval()
    with torch.no_grad():
        for data_idx in tqdm(all_valid_indices, desc="Evaluating Model"):
            data = h_dataset[data_idx].to(device)
            for variant in data.variant_data_lst:
                (target_node, _, target_net_demand, batch, num_vn, vn_node) = variant
                batch, vn_node = batch.to(device), vn_node.to(device)
                data.batch, data.num_vn, data.vn = batch, num_vn, vn_node
                node_rep, net_rep = model(data, device)
                node_rep = torch.squeeze(node_rep)
                net_rep = torch.squeeze(net_rep)
                node_predictions.append(node_rep.cpu().numpy())
                node_targets.append(target_node.cpu().numpy())
                net_predictions.append(net_rep.cpu().numpy())
                net_targets.append(target_net_demand.cpu().numpy())

    node_predictions = np.concatenate(node_predictions)
    node_targets = np.concatenate(node_targets)
    net_predictions = np.concatenate(net_predictions)
    net_targets = np.concatenate(net_targets)

    mse_node = np.mean((node_predictions - node_targets) ** 2)
    mse_net = np.mean((net_predictions - net_targets) ** 2)
    print(f"Final MSE on Validation Set - Node: {mse_node:.4f}, Net: {mse_net:.4f}")

    with open(mse_log_file, mode="a", newline="") as file:
        csv.writer(file).writerow(
            [current_num_layer, current_num_dim, mse_node, mse_net]
        )
    return mse_node, mse_net


# -------------------------------
# Main Function: Grid Search Loop
# -------------------------------
def main():
    setup_logging()
    h_dataset = load_data()
    h_data = h_dataset[0]

    for current_num_layer, current_num_dim in itertools.product(
        num_layer_choices, num_dim_choices
    ):
        print(
            f"\n=== Grid Search: Layers={current_num_layer}, Dim={current_num_dim} ==="
        )
        model = build_model(h_data, current_num_layer, current_num_dim)
        total_time, best_epoch, gpu_peak = train_model(
            model, h_dataset, current_num_layer, current_num_dim
        )
        print(
            f"Training completed in {total_time:.2f}s; best epoch: {best_epoch}; Peak GPU Memory: {gpu_peak:.2f} MB"
        )
        model_path = (
            f"{model_type}_{current_num_layer}_{current_num_dim}_{vn}_{trans}_model.pt"
        )
        model = build_model(h_data, current_num_layer, current_num_dim)
        model.load_state_dict(torch.load(model_path))
        val_node_mse, val_net_mse = evaluate_model(
            model, h_dataset, current_num_layer, current_num_dim
        )
        with open(grid_search_log, mode="a", newline="") as file:
            csv.writer(file).writerow(
                [
                    current_num_layer,
                    current_num_dim,
                    total_time,
                    gpu_peak,
                    val_node_mse,
                    val_net_mse,
                ]
            )
        print(
            f"Completed: Layers={current_num_layer}, Dim={current_num_dim}, Peak Memory={gpu_peak:.2f}MB, Time={total_time:.2f}s"
        )


if __name__ == "__main__":
    main()
