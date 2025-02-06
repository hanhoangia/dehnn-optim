# Optimizing Training Cost for  Scalable Graph Processing

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Clone the repository](#step-1-clone-the-repository)
  - [Step 2: Download the Data](#step-2-download-the-data)
  - [Step 3: Setup Development Environment](#step-3-setup-development-environment)
    - [Step 3.1: Install Miniconda (If Uninstalled)](#step-31-install-miniconda-if-uninstalled)
    - [Step 3.2: Create and Activate Conda Environment](#step-32-create-and-activate-conda-environment)
    - [Step 3.3: Install and Activate Dedicated Python Kernel](#step-33-install-and-activate-dedicated-python-kernel)
    - [Step 3.4: Install Pytorch](#step-34-install-pytorch)
    - [Step 3.5: Install CUDA-related packages (If Used with CUDA)](#step-35-install-cuda-related-packages-if-used-with-cuda)
- [Repo Structure](#repo-structure)

---

## Introduction

---

## Background

---

## Setup Instructions

### Step 1: Clone the repository

Clone this repository and cd into the cloned directory:

```
git clone https://github.com/hanhoangia/dehnn-optim.git
cd dehnn-optim
```

### Step 2: Download the Data


### Step 3: Setup Development Environment

**Note**: This project is developed and run in a Conda environment. Please follow the instructions below to setup the environment.

#### Step 3.1 Install Miniconda (If Uninstalled)

Follow the instructions [here](https://docs.anaconda.com/miniconda/install/) based on the specs of your machine.

#### Step 3.2 Create and Activate Conda Environment

```bash
conda install --name "B12-3" --file requirements.txt
conda activate B12-3
```

#### Step 3.3 Install and Activate a Dedicated Python Kernel

```bash
conda install -c conda-forge nb_conda_kernels
```

#### Step 3.4 Install  Torch

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

#### Step 3.5 Install  CUDA-related packages (If  Used with CUDA)

You install from the requirements by entering the following command but it will take a lot of time because Pip will build the installer from source:

```bash
pip install -r requirements.txt --extra-index-url https://pytorch-geometric.com/whl/torch-{torch_version}+cu{cuda_version}.html
```

Instead, do the following: 

- Follow the link `https://pytorch-geometric.com/whl/torch-{torch_version}+cu{cuda_version}.html` and download the appropriate wheels for your machine specs and install the packages using the wheels:

  

  ```bash
  pip install {donwloaded_wheel_file_name}
  ```


- For Deep Graph Library package requirement, follow the instructions [here](https://www.dgl.ai/pages/start.html) to install a version that works with your Pytorch and CUDA version.

## Repo Structure

- `README.me`: Includes an overview and reproducing instructions for the project.
- `cuda_related_packages.txt`: List of CUDA-related libraries required for the project if run on CUDA.
- `requirements.txt`: List of Python packages required for the project.
- `models/`
  - `encoders/`: Stores the data of the assignment.
  - `layers/`: Contains the data description files.
  - `trained_models/`: Contains the interactive code in Jupyter Notebook that produces the outputs for the assignment.
- `notebooks/`: Contains data exploration plots and analysis for the profiling results.
- `profiling_results/`: Contains the profiling results of the experiments, for both runtime profiling and memory profiling.
- `src/`: Contains the model training and evaluation source code, as well as the associated utility file.
- `.gitignore`:  Specify which files or directories Git should *ignore* when tracking changes in a repository.
