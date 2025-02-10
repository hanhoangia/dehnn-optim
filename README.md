# Optimizing Training Cost for  Scalable Graph Processing

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Environment Setup](#environment-setup)
  - [Step 1: Clone the repository](#step-1-clone-the-repository)
  - [Step 2: Download the Data](#step-2-download-the-data)
  - [Step 3: Setup Development Environment](#step-3-setup-development-environment)
    - [Step 3.1: Install Miniconda (If Uninstalled)](#step-31-install-miniconda-if-uninstalled)
    - [Step 3.2: Create and Activate Conda Environment](#step-32-create-and-activate-conda-environment)
    - [Step 3.3: Install and Activate Dedicated Python Kernel](#step-33-install-and-activate-dedicated-python-kernel)
    - [Step 3.4: Install Pytorch](#step-34-install-pytorch)
    - [Step 3.5: Install CUDA-related packages (If Used with CUDA)](#step-35-install-cuda-related-packages-if-used-with-cuda)
- [Running the Experiment](#running-the-experiment)
- [Repo Structure](#repo-structure)

---

## Introduction

The DE-HNN model is a state-of-the-art (SOTA) graph neural network model to predict congestion using circuit netlist representation, outperforming other hypergraph and netlist models in both node-based and net-based demand regression tasks. However, its high computational demands pose a significant barrier to practical deployment, limiting scalability and efficiency. To address this challenge, our research focuses on exploring strategies to optimize the training cost of DE-HNN, in terms of memory usage or training runtime while preserving prediction quality. We systematically tune hyperparameters and profile metrics to analyze the trade-offs between computational cost and accuracy, aiming to identify the optimal balance between efficiency and performance across various model configurations.

---

## Background

The DE-HNN model is a type of Graph Neural Network (GNN) designed for congestion modeling in chip design by leveraging hypergraph structures and virtual nodes to capture long-range dependencies in dense circuits. Although it demonstrates strong predictive performance, DE-HNN is computationally expensive due to its dual update mechanism for nodes and nets, as well as the construction of virtual nodes that facilitate message passing. These factors lead to high memory consumption and extended training times, limiting the model’s scalability to larger and more complex circuit netlists. Given the increasing complexity of modern chip designs, reducing the computational cost of DE-HNN without significantly degrading predictive performance is essential for its practical deployment.

DE-HNN addresses these challenges by introducing a directed hypergraph representation of netlists, which preserves the hierarchical structure of circuit elements and differentiates between driver and sink nodes. It enhances message-passing effectiveness by incorporating hierarchical virtual nodes (VNs) and persistence-based topological summaries, enabling better scalability for large-scale designs. The model has demonstrated SOTA performance in predicting wirelength and congestion from netlist inputs, outperforming other SOTA machine learning models for hypergraphs and netlists such as NetlistGNN.

Unlike efforts that emphasize accuracy improvements, our research focuses on the investigation of cost-optimizing strategies for DE-HNN through hyperparameter tuning and model architecture adjustments. Specifically, we employ metric profiling to identify key computational bottlenecks in terms of both runtime and memory usage for DE-HNN. Our hypothesis is that reducing the number of layers in the model, thereby shortening the propagation interval, will lead to significant reductions in both memory requirements and runtime. This is because we suspect that neural message-passing between nets and nodes drives the majority of the model's computational cost, and by targeting this bottleneck, we can explore the trade-off between DE-HNN training cost and its performance.

---

## Environment Setup

### Step 1: Clone the repository

Clone this repository and cd into the cloned directory:

```
git clone https://github.com/hanhoangia/dehnn-optim.git
cd dehnn-optim
```

### Step 2: Download the Data

You can download the initial processed dataset [here](https://zenodo.org/records/14599896/files/superblue.zip?download=1), unzip to `data/` and run the following commands to get the data ready for training:

```bash
cd data
python run_all_data.py
```


### Step 3: Setup Development Environment

**Note**: This project is developed and run in a Conda environment. Please follow the instructions below to setup the environment.

#### Step 3.1 Install Miniconda (If Uninstalled)

Follow the instructions [here](https://docs.anaconda.com/miniconda/install/) based on the specs of your machine.

#### Step 3.2 Create and Activate Conda Environment

```bash
conda install --name "B12-3" --file requirements.txt
conda activate B12-3
```

### Step 3.3 Install and Activate a Dedicated Python Kernel

```bash
conda install -c conda-forge nb_conda_kernels
```

### Step 3.4 Install  Torch

- For CPU only:

```bash
pip install torch=={torch_version} torchvision=={torchvision_version} torchaudio=={torchaudio_version}
```

- For use with CUDA (if installed):

```bash
pip install torch=={torch_version}+cu{cuda_version} torchvision=={torchvision_version}+cu{cuda_version} torchaudio=={torchaudio_version} --extra-index-url https://download.pytorch.org/whl/cu{cuda_version}
```

**Note**: 

Please refer to https://pytorch.org/get-started/previous-versions/ to find the `torchvision_version` and `torchaudio_version`, given the `torch_version` you want to install and/or use. For use with CUDA, please note carefully the `cuda_version` that is on your machine by running `nvidia-smi` and follow the command template.

For example, if `torch_version` = 2.2.2, `torchvision_version` = 0.17.2, `torchaudio_version` = 2.2.2, `cuda_version` = 121 (i.e. 12.1), then the entered command would be:

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121
```

**Note**: 

### Step 3.5 Install  CUDA-related packages (If  Used with CUDA)

You can install from `cuda_related_packages.txt` by entering the following command but it will take a lot of time because Pip will build many of the installers from source:

```bash
pip install -r cuda_related_packages.txt --extra-index-url https://pytorch-geometric.com/whl/torch-{torch_version}+cu{cuda_version}.html
```

Instead, do the following: 

- Follow the link `https://pytorch-geometric.com/whl/torch-{torch_version}+cu{cuda_version}.html` and download the appropriate wheels for your machine specs and install the packages using the wheels:

  

  ```bash
  pip install {donwloaded_wheel_file_name}
  ```


- For Deep Graph Library package requirement, follow the instructions [here](https://www.dgl.ai/pages/start.html) to install a version that works with your PyTorch and CUDA version.

---

## Running the Experiment

There are 2 main source code files to run the experiment, `all_train_cross.py` for single model training and performance, and `grid_search.py` for profiling results with different model configurations using Grid Search.

- To train the model, run:

```bash
cd src
python all_train_cross.py
```

- To generate model performance after training, open `src/all_train_cross.py` and change the `test` parameter from `False` to `True` and rerun the `all_train_cross.py` file.

**Note**: To modify the model parameters, open `src/all_train_cross.py` and modify `num_layers`/`num_dim` to your needs.

- To generate profiling results using grid-search:

```bash
cd src
python grid_search.py
```

**Note**: To modify the parameter grid of grid-search, open `src/grid_search.py` and modify `num_layer_choices` and `num_dim_choices` to your needs.

**Note**: There are pre-trained models available for use. The pre-trained model files are saved in `models/trained_models` in the following format: `{model_type}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt`. To use a pre-trained model, first make sure the parameters of the model that will be used to run the experiment in the source code file matches with an available model that is in accordance to your parameter preferences, then change the `test` parameter in the source code file from `False` to `True`.

*The experiment results will be stored in the `profiling_results` directory*.

## Repo Structure

- `README.md`: Includes an overview and reproducing instructions for the project.
- `cuda_related_packages.txt`: List of CUDA-related libraries required for the project if run on CUDA.
- `requirements.txt`: List of Python dependencies required for the project.
- `data`: Contains the dataset description file (i.e. `README_DATA.md`), the source code files to process the data, and the data files themselves.
- `models/`
  - `encoders/`: Contains the source code for the encoders of the model.
  - `layers/`:  Contains the source code for the layers of the model.
  - `trained_models/`: Contains the pre-trained model files.
- `notebooks/`: Contains data exploration plots and analysis for the profiling results.
- `profiling_results/`: Contains the profiling results of the experiments, for both runtime profiling and memory profiling.
  - `runtime`: Contains the runtime profiling results.
  - `memory`: Contains the memory profiling results.
  - `grid-search`: Contains grid-search profiling results.
- `src/`: Contains the model training and evaluation source code, as well as the associated utility file.
- `.gitignore`:  Specify which files or directories Git should *ignore* when tracking changes in a repository.
