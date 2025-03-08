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
- [Profiling Instructions](#profiling-instructions)
  - [Step 1: Configure the Model](#configure-the-model)
  - [Step 2: Run the Experiment](#run-the-experiment)

- [Repo Structure](#repo-structure)

---

## Introduction

In chip design, traditional place-and-route (PnR) methods are beset by inefficiencies, as their iterative refinement of layouts can be both time-consuming and labor-intensive. By contrast, data-driven chip design optimization offers a more efficient alternative, leveraging machine learning to predict resource bottlenecks early on and inform design decisions that minimize costly design iterations. Central to this approach is the netlist, a hypergraph representation of a circuit's connectivity, where nodes represent components (e.g., logic gates) and hyperedges represent electrical connections. By analyzing and optimizing resource demand through the netlist, this method directly enhances floorplanning—the process of arranging components on a 2D chip canvas—by providing insights that minimize congestion while optimizing Power, Performance, and Area (PPA) and meeting design constraints.

DE-HNN is a state-of-the-art hypergraph neural network designed to predict congestion in chip design via demand regression. It outperforms other models by effectively capturing long-range dependencies through hierarchical virtual nodes, which aggregate node features within partitioned graph neighborhoods and propagate information efficiently, enabling more robust predictions.

While DEHNN delivers high performance, its scalability and practicality are limited by its significant computational overhead and lengthy runtimes.

---

## Project Overview

The high computational cost of DE-HNN has motivated our project to explore cost-effective ways of optimizing the model while preserving the model performance.

---

## Environment Setup

### Step 1: Clone the repository

Clone this repository and cd into the cloned directory:

```
git clone https://github.com/hanhoangia/dehnn-optim.git
cd dehnn-optim
```

### Step 2: Download the Data

Download the fully processed dataset [here](https://drive.google.com/file/d/1ir2ZeRgKJkfl9W6mK-xwug0y_-Kc3S21/view?usp=sharing) and move it to `/data`.

### Step 3: Setup Development Environment

**Note**: This project is developed and run in a Conda environment. Please follow the instructions below to setup the environment.

#### Step 3.1: Install Miniconda (If Uninstalled)

Follow the instructions [here](https://docs.anaconda.com/miniconda/install/) based on the specs of your machine.

#### Step 3.2: Create and Activate Conda Environment

```bash
conda install --name "B12-3" --file requirements.txt
conda activate B12-3
```

### Step 3.3: Install and Activate a Dedicated Python Kernel

```bash
conda install -c conda-forge nb_conda_kernels
```

### Step 3.4: Install  Torch

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

### Step 3.5: Install  CUDA-related packages (If  Used with CUDA)

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

## Profiling Instructions

### Step 1:  Configure the Model

There are 10 hyperparameters you can set to configure the model:

| Hyperparameter  | Description                                                  | Default |
| --------------- | ------------------------------------------------------------ | ------- |
| num_layer       | the number of layers in the model                            | 4       |
| num_dim         | the number of dimensions in the model                        | 8       |
| learning_rate   | the learning rate of the model                               | 0.001   |
| early_stop      | use the Early Stopping condition or not                      | True    |
| MIN_EPOCHS      | the minimum number of epochs to run before Early Stopping condition can be triggered | 5       |
| PATIENCE        | the number of prior epochs used to compute the average validation loss as the condition trigger | 5       |
| TOLERANCE       | the allowed range around the average validation loss within which the loss must remain to continue training | 0.1     |
| use_manual_seed | use a fixed random seed for weight initialization of the model | True    |
| manual_seed     | the fixed seed number (if use_manual_seed = True)            | 42      |
| device          | use CUDA or not                                              | cuda    |

**Note**: The default is set to the settings of our optimized model. Refer to our report for our baseline settings and the settings of our other experiments. Feel free to play with the hyperparameters, but note that high `num_layer` and `num_dim` can cause out-of-memory (OOM) issue.

### Step 2: Run the Experiment

There are 3 ways to run the experiments: 

1. `all_train_cross.py` for single model training and performance with a specified configuration. 

   *The experiment results will be stored in the `profiling_results/[current date time of the run]` directory*, consisting of the training runtime, peak memory usage, model performance, and Early Stop epoch (if Early Stopping is enabled).

2. `grid_search.py` for testing with different model configurations using Grid Search.

   *The experiment results will be stored in the `profiling_results/grid-search` directory*. Run our `gridsearch_plots.ipynb` notebook to generate the visualization for the results.

3. `cross-val.py` for doing 6-Fold cross-validation for a specified model configuration.

   *The experiment results will be stored in the `profiling_results/cross-val\[current date time of the run]` directory*.

- To train the model, run:

```bash
cd src
python all_train_cross.py
```

- To get model performance on the test set, open `src/all_train_cross.py` and change the `test` parameter from `False` to `True` and rerun the `all_train_cross.py` file.

**Note**: There are pre-trained model files available in `models/trained_models` for the models that were generated through our *iterative optimization* process . To use a pre-trained model, simply update the `custom_model_file_path` variable in the source code file to match the name of the pre-trained model file.

## Repo Structure

- `README.md`: Includes an overview and reproducing instructions for the project.
- `cuda_related_packages.txt`: List of CUDA-related libraries required for the project if run on CUDA.
- `requirements.txt`: List of Python dependencies required for the project.
- `data`: Contains the dataset description file (i.e. `README_DATA.md`), and the downloaded dataset file.
- `models/`
  - `encoders/`: Contains the source code for the encoders of the model.
  - `layers/`:  Contains the source code for the layers of the model.
  - `trained_models/`: Contains the pre-trained model files.
- `notebooks/`: Contains the visualization notebooks for the profiling results.
- `profiling_results/`: Contains the profiling results of our experiments. Also contains our model debugging results.
  - `baseline`: Contains the profiling results for the baseline model.
  - `iterative-optimization`: Contains `README_MODEL.md` that describes our *iterative optimization process* and the models produced in that process. Contains the profiling results for these models.
  - `grid-search`: Contains the profiling results for our Grid Search experiment.
  - `cross-val`: Contains the profiling results for our Cross Validation run on our *optimized* model.
  - `gradient-norm-analysis`: Contains the gradient norm debugging results as part of our attempt to debug DE-HNN after a concern of overfitting.
- `src/`: Contains the source code files for the experiments.
- `.gitignore`:  Specify which files or directories Git should *ignore* when tracking changes in a repository.
