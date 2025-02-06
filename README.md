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
- [Repo Structure](#repo-structure)

---

## Introduction

The domain of our section is graph machine learning in the field of chip design. We spent our first quarter learning about graph fundamentals and different graph machine learning models with a focus on graph neural networks. Besides the lectures, we have 2 assignments constitute our Quarter 1 project that are designed to help us learn how to construct a graph from the data, produce graph statistics, build graph machine learning model architectures, and understand the application and importance of an efficient graph machine learning algorithm when it comes to chip design.

---

## Background

---

## Setup Instructions

### Step 1: Clone the repository

Clone this repository and cd into the cloned directory:

```
git clone https://github.com/hanhoangia/DSC180A-Q1.git
cd DSC180A-Q1
```

### Step 2: Download the Data


### Step 3: Setup Development Environment

**Note**: This project is developed and run in a Conda environment. Please follow the instructions below to setup the environment.

#### Step 3.1 Install Miniconda (If Uninstalled)

Follow the instructions [here](https://docs.anaconda.com/miniconda/install/) based on the specs of your machine.

#### Step 3.2 Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate B12-3
```

**Note**: When the first command is run, it automatically creates and enables a Python kernel dedicated for the new environment.

---

## Repo Structure

- `README.me`: Overview of the quarter 1 project, consists of assignment 1 and assignment 2, and reproducing instructions for the project.
- `cuda_related_packages.txt': List of CUDA-related libraries required for the project if run on CUDA.
- `requirements.txt`: List of Python dependencies required for the project.
- `models/`
  - `enconders/`: Stores the data of the assignment.
  - `layers/`: Contains the data description files.
  - `trained_models/`: Contains the interactive code in Jupyter Notebook that produces the outputs for the assignment.
- `notebooks/`: Contains data exploration plots and analysis for the profiling results.
- `profiling_results/`: Contains the profiling results of the experiments, for both runtime profiling and memory profiling.
- `src/`: Contains the model training and evaluation source code, as well as the associated utility file.
- `.gitignore`:  Specify which files or directories Git should *ignore* when tracking changes in a repository.
