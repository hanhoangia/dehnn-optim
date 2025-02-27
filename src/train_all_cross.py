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
# import torch.optim.lr_scheduler.CyclicLR
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
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')
    
    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')
    
    return accuracy, precision, recall

def early_stopping_condition(val_loss_history, k=10, tol=0.01):
    """
    Function to check if the early stopping condition is met
    :param val_loss_history: List of validation losses
    :param patience: Number of epochs to wait before stopping
    :param tol: Tolerance value for improvement in validation loss
    :return: If the early stopping condition is met, return the range of the last k validation losses, else return False
    """
    if len(val_loss_history) >= k:
        # Compute the average loss between node and net for each epoch
        avg_k_losses = [(loss[0] + loss[1]) / 2 for loss in val_loss_history[-k:]]
        # Check if the last k validation losses are still within the tolerance range
        stop = all(loss > (1 - tol) * avg_k_losses[0] for loss in avg_k_losses[1:])
        stop_range_diff = (1 - tol) * avg_k_losses[0]
        stop_range = [avg_k_losses[0] - stop_range_diff, avg_k_losses[0] + stop_range_diff]
        if stop:
            return stop_range
    return False

### hyperparameter ###
test = False # if only test but not train
restart = False # if restart training
reload_dataset = True # if reload already processed h_dataset

if test:
    restart = True

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

if not reload_dataset:
    dataset = NetlistDataset(data_dir="../data/superblue", load_pe = True, pl = True, processed = True, load_indices=None)
    h_dataset = []
    for data in tqdm(dataset):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances
        
        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]] 
        filtered_edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        data.edge_index_source_to_net = filtered_edge_index_source_to_net

        mask_edges = mask[data.edge_index_sink_to_net[1]] 
        filtered_edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]
        data.edge_index_sink_to_net = filtered_edge_index_sink_to_net

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['net'].x = data.net_features
        
        edge_index = torch.concat([data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1)
        h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.concat([torch.zeros(data.edge_index_sink_to_net.shape[1]), torch.ones(data.edge_index_source_to_net.shape[1])]).bool()
        h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)
        
        h_data['design_name'] = data['design_name']
        h_data.num_instances = data.node_features.shape[0]
        variant_data_lst = []
        
        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl
        
        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat([global_mean_pool(h_data['node'].x, batch), 
                global_max_pool(h_data['node'].x, batch)], dim=1)

        #node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        #net_demand = (net_demand - torch.mean(net_demand))/ torch.std(net_demand)

        variant_data_lst.append((node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)) 
        h_data['variant_data_lst'] = variant_data_lst
        h_dataset.append(h_data)
        
    torch.save(h_dataset, data_file_path)
    
else:
    dataset = torch.load(data_file_path)
    h_dataset = []
    for data in dataset:
        h_dataset.append(data)

h_data = h_dataset[0]
if restart:
    model = torch.load(model_file_path)
else:
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=10, step_size_down=10, mode="triangular2")
load_data_indices = [idx for idx in range(len(h_dataset))]
all_train_indices, all_valid_indices, all_test_indices = load_data_indices[:5], load_data_indices[10:11], load_data_indices[10:11]
best_total_val = None
# only include gradients of the weights; not the gradients of the bias
layer_gradients = {name: [] for name, _ in model.named_parameters() if "bias" not in name}
avg_grad_norms_epochs = []
num_epochs = 20
lrs = []

if not test:
    t0 = time.time()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        np.random.shuffle(all_train_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0
        
        all_train_idx = 0
        for data_idx in tqdm(all_train_indices):
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

                #norm_gradients = [torch.sum(torch.pow(torch.tensor(gradient), 2)).item() for gradient in model.parameters() if gradient.grad is not None]
                #print(f"Epoch {epoch}, Layer-wise L2 norm of gradients: {norm_gradients}")
                # Gradient clipping
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Get gradients for each layer
                grads_w = [(name, param) for name, param in model.named_parameters() if "bias" not in name]
                
                for name, param in grads_w:
                    if param.grad is not None:
                        layer_gradients[name].append(param.grad.norm().item())
                
                optimizer.step()
                scheduler.step()

                # get the scheduler learning rate
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                    lrs.append(current_lr)

    
                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1

        # Calculate the average gradient norm for each gradient
        avg_grad_norms_epoch = {name: np.mean(grads) for name, grads in layer_gradients.items() if len(grads) > 0}
        if epoch == 0:
            avg_grad_norms_epochs = {name: [avg_grad_norm] for name, avg_grad_norm in avg_grad_norms_epoch.items()}
        else:
            for name, avg_grad_norm in avg_grad_norms_epoch.items():
                avg_grad_norms_epochs[name].append(avg_grad_norm)

        # Clear gradients for the next epoch
        layer_gradients = {name: [] for name, _ in model.named_parameters() if "bias" not in name}
        
        print(loss_node_all/all_train_idx, loss_net_all/all_train_idx)
        train_losses.append((loss_node_all/all_train_idx, loss_net_all/all_train_idx))
    
        all_valid_idx = 0
        for data_idx in tqdm(all_valid_indices):
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
            torch.save(model, f"{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt")

        # stop the training when the early stopping condition is met
        patience = 10
        tol = 0.01
        stop_epoch = None
        stop_range = None
        # if the early stopping condition function does not return False, it means the condition is met
        stop_result = early_stopping_condition(val_losses, patience, tol)
        if stop_result is not False:
            print("Early stopping condition met at epoch: ", epoch)
            stop_epoch = epoch
            stop_range = stop_result
            break

    # do a line plot of the training losses over the learning rate (lrs)
    # do a subplot of the training losses for the node and net
    # smooth out the line plot by taking the moving average of the losses
    train_losses_node = [loss[0] for loss in train_losses]
    train_losses_net = [loss[1] for loss in train_losses]
    val_losses_node = [loss[0] for loss in val_losses]
    val_losses_net = [loss[1] for loss in val_losses]
    plt.plot(lrs, train_losses_node, label="train_node_loss")
    plt.plot(lrs, train_losses_net, label="train_net_loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Training Losses over Learning Rate")
    plt.legend()
    plt.savefig(f"{performance_outdirectory}/train_losses.png")
    plt.close()

    # do a line plot of the average gradient norms for the gradients with the same name
    for name, avg_grad_norms in avg_grad_norms_epochs.items():
        plt.plot(avg_grad_norms, label=name)
        plt.xticks(range(0, num_epochs, 1))
        plt.xlabel("Epoch")
        plt.ylabel("Average Gradient Norm")
        plt.title(f"Average Gradient Norms for {name} over {num_epochs} epochs")
        plt.savefig(f"{performance_outdirectory}/{name}_grad.png")
        plt.close()
        
    t1 = time.time()
    total_time = t1 - t0
    total_time_in_mins = total_time / 60
    other_metrics_file_path = os.path.join(runtime_outdirectory, "other_metrics.csv")
    with open(other_metrics_file_path, "w") as f:
        f.write("total_runtime_in_mins,stop_epoch,patience,tolerance,stop_range,best_total_val\n")
        if stop_epoch is not None:
            f.write(f"{total_time_in_mins},{stop_epoch},{patience},{tol},{stop_range},{best_total_val}")
        else:
            f.write(f"{total_time_in_mins},N/A,{patience},{tol},N/A,{best_total_val}")
    print(f"Training runtime and other metrics saved to: {other_metrics_file_path}")
    loss_file_path = os.path.join(performance_outdirectory, "losses.csv")
    with open(loss_file_path, "w") as f:
        # write train losses and validation losses in csv format 
        # 5 columns: epoch, train_node_loss, train_net_loss, val_node_loss, val_net_loss
        f.write("epoch,train_node_loss,train_net_loss,val_node_loss,val_net_loss\n")
        for i in range(len(train_losses)):
            f.write(f"{i},{train_losses[i][0]},{train_losses[i][1]},{val_losses[i][0]},{val_losses[i][1]}\n")
    print(f"Train losses and validation losses saved to: {loss_file_path}")
else:
    all_test_idx = 0
    test_loss_node_all = 0
    test_loss_net_all = 0
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
            
            test_loss_node = criterion_node(node_representation, target_node.to(device))
            test_loss_net = criterion_net(net_representation, target_net_demand.to(device))
            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1

    avg_test_node_demand_mse = test_loss_node_all/all_test_idx
    avg_test_net_demand_mse = test_loss_net_all/all_test_idx
    print("avg test node demand mse: ", avg_test_node_demand_mse)
    print("avg test net demand mse: ", avg_test_net_demand_mse)

    file_path = os.path.join(performance_outdirectory, "test_performance.txt")
    with open(file_path, "w") as f:
        f.write(f"Average test MSE for node demand regression: {avg_test_node_demand_mse}\n")
        f.write(f"Average test MSE for net demand regression: {avg_test_net_demand_mse}")
    print(f"Test performance result saved to: {file_path}")

# save the configuration of the model
config_file_path = os.path.join(performance_outdirectory, "config.csv")
with open(config_file_path, "w") as f:
    f.write("num_layer,num_dim,learning_rate,num_epochs\n")
    f.write(f"{num_layer},{num_dim},{learning_rate},{num_epochs}")