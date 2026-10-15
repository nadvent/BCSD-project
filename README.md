=======
# BCSD-project
WIP Machine learning (vision) project with Blood Cell Segmenation dataset
=======
>>>>>>> feature/model-training
# Blood Cell Semantic Segmentation: PyTorch Training & C++ Inference Pipeline

(WIP). An end-to-end computer vision pipeline combining **PyTorch** model training and **high-performance C++ inference** (via ONNX Runtime) for automated blood cell segmentation in microscopic smear images.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Dataset Guide](#dataset-guide)
  - [Default Benchmark: BCCD with Mask](#default-benchmark-bccd-with-mask)
  - [Using Your Own Custom Dataset](#using-your-own-custom-dataset)
  - [Folder Layout](#folder-layout)
- [Prerequisites & Installation](#prerequisites--installation)
  - [Python Environment](#python-environment)
  - [C++ Inference Dependencies](#c-inference-dependencies)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Data Exploration & Verification (Quarto)](#1-data-exploration--verification-quarto)
 <!--
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Metrics](#3-evaluation--metrics)
  - [4. ONNX Model Export](#4-onnx-model-export)
  - [5. C++ Deployment & Inference](#5-c-deployment--inference)
- [Configuration Reference](#configuration-reference)
- [License](#license)
-->

---

## Overview

(WIP) Accurate identification and segmentation of blood cells (white blood cells, red blood cells, and platelets) in peripheral blood smears is an essential diagnostic step in hematology. 

* **Python Training Pipeline (`src/python/`)**:
  *  `VisionDataset` class using `torchvision.transforms.v2` for synchronized image-mask spatial augmentations (bilinear image resizing, nearest-neighbor mask label preservation, and random horizontal flipping).
  * Medical image normalization using pre-trained ImageNet color distribution statistics. Specifically, the transformation for normalization.
  <!-- * Robust segmentation loss combining **Binary Cross-Entropy (BCE) and Dice Loss** to manage foreground-to-background class imbalance. -->
  <!-- * Comprehensive validation tracking both **Dice Score (F1)** and **Intersection-over-Union (IoU / Jaccard Index)**. -->
* **Low-Latency C++ Deployment (`src/cpp/`)**:
  * Standalone C++ engine using the **ONNX Runtime C++ API** for hardware-accelerated, dependency-light deployment on edge devices and clinical workstations.
  * Fast image preprocessing and mask visualization powered by **OpenCV**.
* **Reproducible Reporting (`notebook/`)**:
  * Interactive Quarto (`.qmd`) documents for visual mask-alignment verification and dataset distribution sanity checks.

---

## Pipeline Architecture

```
                       +-------------------------------+
                       |  Blood Cell Dataset (BCCD)    |
                       |  (data/train & data/test)     |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |    Quarto EDA (.qmd)          |
                       |  (Visual image/mask overlay)  |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |     PyTorch Training          |
                       |   (src/python/train.py)       |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |       ONNX Serialization      |
                       |  (src/python/export_onnx.py)  |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |   C++ ONNX Runtime Engine     |
                       |    (src/cpp/inference_app)    |
                       +-------------------------------+
```

---

## Project Structure


```
.
├── config/                 # YAML configuration files (hyperparameters, paths)
├── data/
│   ├── test/
│   │   ├── mask/           # Ground truth binary masks (lossless PNG)
│   │   └── original/       # Raw blood smear images (JPG/PNG)
│   └── train/
│       ├── mask/           # Ground truth binary masks (lossless PNG)
│       └── original/       # Raw blood smear images (JPG/PNG)
├── notebook/               # Quarto notebooks for exploratory analysis & reports
│   └── dataset_exploration.qmd
├── outputs/                # Saved weights (.pth), exported ONNX models, and metrics
├── src/
│   ├── cpp/                # C++ inference engine (formerly cpp_pl)
│   │   ├── CMakeLists.txt  # Build definitions
│   │   ├── include/        # C++ header files (preprocessor, engine, postprocessor)
│   │   └── src/            # C++ implementation files (.cpp)
│   └── python/             # Python training pipeline (formerly python_pl)
│       ├── dataset.py      # Custom Dataset with PyTorch v2 transforms
│       ├── model.py        # Segmentation model architecture (U-Net)
│       ├── train.py        # Training and validation loop
│       ├── evaluate.py     # Evaluation metrics (Dice Score, IoU)
│       ├── export_onnx.py  # Model export script to ONNX
│       └── requirements.txt# Python package dependencies
├── LICENSE                 # Project license
└── README.md               # Project documentation
```


---

## Dataset Guide

### Default Benchmark: BCCD with Mask

This project is configured and benchmarked out-of-the-box using the [Kaggle BCCD Dataset with Mask](https://www.google.com/search?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fjeetblahiri%2Fbccd-dataset-with-mask%2Fdata) created by Jeet Lahiri.

You can download it manually from Kaggle or via the Kaggle CLI:

```bash
# Download via Kaggle CLI
kaggle datasets download -d jeetblahiri/bccd-dataset-with-mask

# Unzip into data/
unzip bccd-dataset-with-mask.zip -d data/
```

### Using Your Own Custom Dataset

While the repository defaults to BCCD, the pipeline is **fully modular**. You can plug in any custom medical, microscopic, or cell segmentation dataset without modifying the model code, provided your data follows the directory layout below.

### Folder Layout

Organize your training and test splits so image stems match their corresponding masks:

```
data/
├── train/
│   ├── original/          # e.g., BloodImage_00001.jpg
│   └── mask/              # e.g., BloodImage_00001.png
└── test/
    ├── original/          # e.g., BloodImage_00300.jpg
    └── mask/              # e.g., BloodImage_00300.png
```

* **Original images**: RGB color images (`.jpg` or `.png`).
* **Masks**: Lossless `.png` grayscale images. Background pixels must have a value of `0`, and cell regions have values `> 0` (typically `255`). The dataset loader automatically thresholds these into clean binary labels ($\{0.0, 1.0\}$).

---

## Prerequisites & Installation

### Python Environment

1. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate       # Linux/MacOS
   ```

2. **Install dependencies**:
   ```bash
    uv pip install -r src/python/requirements.txt
   ```

   *Core Python dependencies:*
   * `torch`, `torchvision` (>= 2.1 recommended for `transforms.v2`)
   * `numpy`, `pillow`, `matplotlib`
   * `jupyter` (for `.qmd` report generation)
   * `onnx`, `onnxruntime`

### C++ Inference Dependencies

(WIP) To build the C++ inference engine, ensure your system has:
* **CMake** (>= 3.16)
* **C++17 Compiler** (`g++` >= 9 or `clang++` >= 10)
* **OpenCV** (>= 4.x)
* **ONNX Runtime C++ library** (>= 1.14)

On Arch:
```bash
sudo pacman -Syu
sudo pacman -S cmake build-essential libopencv-dev --noconfirm
```

---

## Step-by-Step Workflow

### 1. Data Exploration & Verification (Quarto)

Verify paired data alignment, nearest-neighbor mask interpolation, and binary label values before training:

```bash
# Interactive browser preview with live-reload
quarto preview notebook/dataset_exploration.qmd

# Or render directly to a standalone HTML report
quarto render notebook/dataset_exploration.qmd
```

<!-- ### 2. Model Training -->
<!---->
<!-- Launch training with combined BCE + Dice loss and validation monitoring: -->
<!---->
<!-- ```bash -->
<!-- python src/python/train.py --config config/train_config.yaml -->
<!-- ``` -->
<!---->
<!-- The training process tracks the validation Dice score across epochs and writes the best checkpoint to `outputs/best_model.pth`. -->
<!---->
<!-- ### 3. Evaluation & Metrics -->
<!---->
<!-- Compute comprehensive segmentation metrics across the test dataset: -->
<!---->
<!-- ```bash -->
<!-- python src/python/evaluate.py \ -->
<!--     --weights outputs/best_model.pth \ -->
<!--     --data_dir data/test -->
<!-- ``` -->
<!---->
<!-- Key output metrics: -->
<!-- * **Dice Score (F1)**: Quantifies spatial boundary agreement between predicted cell masks and ground truth. -->
<!-- * **Mean IoU (Jaccard Index)**: Area of overlap divided by area of union. -->
<!-- * **Pixel Accuracy**: Baseline per-pixel classification accuracy. -->
<!---->
<!-- ### 4. ONNX Model Export -->
<!---->
<!-- Export the PyTorch model to a standardized ONNX graph with fixed spatial dimensions (`[1, 3, 416, 416]`): -->
<!---->
<!-- ```bash -->
<!-- python src/python/export_onnx.py \ -->
<!--     --weights outputs/best_model.pth \ -->
<!--     --output outputs/model.onnx \ -->
<!--     --input_shape 1 3 416 416 -->
<!-- ``` -->
<!---->
<!-- ### 5. C++ Deployment & Inference -->
<!---->
<!-- Build and execute the standalone C++ inference application: -->
<!---->
<!-- ```bash -->
<!-- cd src/cpp -->
<!-- mkdir -p build && cd build -->
<!-- cmake .. -->
<!-- cmake --build . --config Release -->
<!---->
<!-- # Run inference on a test blood smear image -->
<!-- ./inference_app --model ../../../outputs/model.onnx --image ../../../data/test/original/sample.jpg -->
<!-- ``` -->
<!---->
<!-- --- -->
<!---->
<!-- ## Configuration Reference -->
<!---->
<!-- Default training and preprocessing parameters in `config/`: -->
<!---->
<!-- | Parameter | Default | Description | -->
<!-- | :--- | :--- | :--- | -->
<!-- | `image_size` | `(416, 416)` | Input resolution for model tensors | -->
<!-- | `batch_size` | `8` | Training mini-batch size | -->
<!-- | `learning_rate` | `1e-4` | Initial learning rate for AdamW optimizer | -->
<!-- | `loss_function` | `BCE_Dice` | Combined Binary Cross-Entropy and Dice loss | -->
<!-- | `normalization_mean` | `[0.485, 0.456, 0.406]` | ImageNet RGB channel mean values | -->
<!-- | `normalization_std` | `[0.229, 0.224, 0.225]` | ImageNet RGB channel standard deviation | -->
<!---->
---

=======
>>>>>>> 772b2be (Finished Dataset.py and Quarto for image comparison)
>>>>>>> feature/model-training
