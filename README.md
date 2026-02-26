# Probing Super-Resolved Tissue Microarchitecture from Single-Shell Diffusion MRI with 3D Frequency-Aware Cascaded Diffusion Model

This repository contains the official implementation for **FA-CDM**, a novel diffusion model for microstructure synthesis that integrates information from both spatial and frequency domains.

## 1. Environment Setup

Follow these steps to set up the necessary conda environment and install dependencies.

1.  **Create and activate the conda environment:**
    ```bash
    conda create -n sf python=3.10.13
    conda activate sf
    ```

2.  **Install PyTorch and CUDA:**
    ```bash
    # Install CUDA Toolkit
    conda install cudatoolkit==11.8 -c nvidia
    
    # Install PyTorch (v2.1.1 for cu118)
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    *(Note: Please ensure the PyTorch index URL is correct for your CUDA version. The cu118 URL is assumed based on `cudatoolkit==11.8`.)*

## 2. Model Training & Inference

The training and inference processes are handled by a single script.

* **Command:**
    ```bash
    bash /Path/FA_CDM/run.sh
    ```
    *(Note: Please update `/Path/FA_CDM/` to the actual path of the project directory on your system.)*
