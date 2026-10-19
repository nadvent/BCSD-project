# Blood Cell Semantic Segmentation: PyTorch Training & C++ Inference Pipeline

An end-to-end computer vision and systems engineering pipeline combining **PyTorch** deep learning training with a **high-performance C++17 inference engine** (powered by OpenCV and ONNX Runtime) for automated blood cell segmentation in microscopic blood smear images.

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
  - [Python Environment (uv or venv)](#python-environment-uv-or-venv)
  - [C++ Inference Dependencies & System Packages](#c-inference-dependencies--system-packages)
  - [Locating ONNX Runtime on Linux (Arch / Debian / Ubuntu)](#locating-onnx-runtime-on-linux-arch--debian--ubuntu)
- [Step-by-Step Workflow](#step-by-step-workflow)
  - [1. Data Exploration & Alignment (Quarto)](#1-data-exploration--alignment-quarto)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Quantitative Benchmarks](#3-evaluation--quantitative-benchmarks)
  - [4. ONNX Graph Serialization & Parity Verification](#4-onnx-graph-serialization--parity-verification)
  - [5. C++ Inference Engine Build & Deployment](#5-c-inference-engine-build--deployment)
  - [6. Latency Profiling & Systems Benchmarking](#6-latency-profiling--systems-benchmarking)
  - [7. Interactive Reports & Reveal.js Presentations](#7-interactive-reports--revealjs-presentations)
- [Quantitative Results Summary](#quantitative-results-summary)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Overview

Accurate identification and boundary localization of hematological cells (erythrocytes/red blood cells, leukocytes/white blood cells, and platelets) in peripheral blood smears is an essential diagnostic step in clinical pathology. This repository bridges research and production deployment:

- **Python Research & Training Pipeline (`src/python/`)**:
  - **Synchronized Augmentations (`src/python/dataset.py`)**: Custom `VisionDataset` leveraging `torchvision.transforms.v2` for mathematically consistent spatial augmentations—bilinear interpolation for continuous RGB imagery, nearest-neighbor projection to preserve discrete mask topology, and synchronized stochastic dihedral flipping ($g_\theta \in D_2$).
  - **Medical Standardization**: ImageNet channel-wise affine standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) to condition the loss Hessian $\kappa(\mathbf{H}_W) = \lambda_{\max} / \lambda_{\min}$ and accelerate gradient convergence.
  - **From-Scratch U-Net Architecture (`src/python/model.py`)**: Full $31.04\text{M}$-parameter implementation supporting both unpadded mode (`padding=0`, $416\times 416 \to 228\times 228$ with copy-and-crop skip connections to avoid boundary padding artifacts) and padded mode (`padding=1`, $416\times 416 \to 416\times 416$), with `bias=False` prior to `BatchNorm2d`.
  - **Composite Optimization (`src/python/evaluate.py`)**: Robust `BCEDiceLoss` objective combining log-sum-exp numerically stabilized Binary Cross-Entropy with a continuous Frobenius-norm Soft Dice relaxation to handle extreme foreground-to-background class imbalance.
  - **Static Graph Compilation (`src/python/export_onnx.py`)**: Export to ONNX (`opset_version=17`) with constant folding, algebraically absorbing `BatchNorm2d` running moments directly into `Conv2d` kernel weights and biases.
- **Low-Latency C++17 Inference Engine (`src/cpp/`)**:
  - **Lightweight Edge Deployment**: Standalone native binary using the **ONNX Runtime C++ API** (`Ort::Session`) and **OpenCV** with zero runtime Python interpreter dependency.
  - **Memory Layout Transposition (`HWC` $\to$ `NCHW`)**: Efficiently transposes interleaved `BGR` `uint8` camera buffers into contiguous planar `RGB` `float32` tensors with cache-friendly SIMD layout.
  - **Zero-Copy Tensor Binding**: Wraps native memory buffers directly via `Ort::Value::CreateTensor<float>` in CPU arena memory.
  - **Monotonic Logit Thresholding**: Thresholds raw logits directly ($z > 0.0\text{f}$ for $\tau = 0.5$) via sigmoid monotonicity, eliminating $51,984$ transcendental `std::exp()` calls per image frame.
  - **Sub-Pixel Contour Extraction**: Computes vector cell membrane boundaries via `cv::findContours` and renders anti-aliased diagnostic overlays via `cv::drawContours`.
- **Reproducible Documentation & Presentations (`notebook/`)**:
  - Quarto (`.qmd`) technical notebooks covering exploratory data analysis, architecture derivations, training loss landscapes, C++ inference validation, and a full 11-slide Reveal.js technical interview presentation.

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
                       |  (notebook/dataset_explor...) |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |     PyTorch Training          |
                       |    (src/python/train.py)      |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |    Evaluation & Metrics       |
                       |   (src/python/evaluate.py)    |
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
                       | (src/cpp/build/inference_app) |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |  Diagnostic Mask & Overlays   |
                       |   (outputs/cpp_pred_*.png)    |
                       +-------------------------------+
```

---

## Project Structure

All scripts, C++ modules, notebooks, and configurations follow standard, consistent directory conventions:

```
.
├── config/                          # YAML configuration files (hyperparameters, paths)
├── data/
│   ├── test/
│   │   ├── mask/                    # Ground truth binary masks (lossless PNG)
│   │   └── original/                # Raw blood smear images (JPG/PNG)
│   └── train/
│       ├── mask/                    # Ground truth binary masks (lossless PNG)
│       └── original/                # Raw blood smear images (JPG/PNG)
├── notebook/                        # Quarto notebooks, rendered HTML reports & presentations
│   ├── dataset_exploration.qmd      # EDA, label sanity checks, and paired image-mask overlays
│   ├── unet_architecture_notes.qmd  # Mathematical receptive field & skip-connection derivations
│   ├── training_and_evalutation.qmd # Training progression, loss curves, and validation metrics
│   ├── cpp_inference.qmd            # C++ ONNX Runtime integration & xeus-cling interactive guide
│   ├── presentation.qmd             # Reveal.js 11-slide technical portfolio presentation
│   └── cpp_inference_comparison.png # 4-column visual verification grid
├── outputs/                         # Model checkpoints, ONNX graphs, and inference artifacts
│   ├── best_model.pth               # Optimal PyTorch weights (Val Dice: 0.9475)
│   ├── last_model.pth               # Final epoch checkpoint
│   ├── model.onnx                   # Serialized static ONNX computational graph
│   └── cpp_sample_*_{mask,overlay}.png # C++ inference engine output masks and overlays
├── src/
│   ├── cpp/                         # Low-latency C++17 ONNX Runtime inference engine
│   │   ├── CMakeLists.txt           # Modern CMake build definitions (C++17, OpenCV, ONNX Runtime)
│   │   ├── include/
│   │   │   └── inference_engine.hpp # C++ SegmentationEngine class declaration & data structures
│   │   └── src/
│   │       ├── inference_engine.cpp # Preprocessing, zero-copy inference, and contour postprocessing
│   │       └── main.cpp             # CLI driver for single-image and batch inference
│   └── python/                      # PyTorch training, evaluation, and export pipeline
│       ├── dataset.py               # VisionDataset with torchvision.transforms.v2
│       ├── model.py                 # From-scratch U-Net (supports padding=0 and padding=1)
│       ├── train.py                 # AdamW + ReduceLROnPlateau training engine
│       ├── evaluate.py              # BCEDiceLoss and Dice/IoU/PixelAccuracy metrics
│       ├── export_onnx.py           # PyTorch to static ONNX export with parity verification
│       ├── benchmark_python.py      # Python Eager vs torch.compile vs ONNX latency profiling
│       └── requirements.txt         # Pinned Python package dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## Dataset Guide

### Default Benchmark: BCCD with Mask

This project is configured and benchmarked out-of-the-box using the [Kaggle BCCD Dataset with Mask](https://www.google.com/search?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fjeetblahiri%2Fbccd-dataset-with-mask) created by Jeet Lahiri.

You can download it manually from Kaggle or via the Kaggle CLI:

```bash
# Download via Kaggle CLI
kaggle datasets download -d jeetblahiri/bccd-dataset-with-mask

# Unpack directly into data/
unzip bccd-dataset-with-mask.zip -d data/
```

### Using Your Own Custom Dataset

The pipeline is **fully modular**. You can plug in any medical, microscopic, or histopathological cell segmentation dataset without modifying the model architecture, provided your data follows the directory layout below.

### Folder Layout

Organize your training and test splits so image file stems match their corresponding masks:

```
data/
├── train/
│   ├── original/          # e.g., BloodImage_00001.jpg (or .png)
│   └── mask/              # e.g., BloodImage_00001.png
└── test/
    ├── original/          # e.g., BloodImage_00300.jpg (or .png)
    └── mask/              # e.g., BloodImage_00300.png
```

- **Original images**: RGB color images (`.jpg`, `.jpeg`, or `.png`).
- **Masks**: Lossless `.png` grayscale images. Background pixels must have a value of `0`, and cell regions have values `> 0` (typically `255`). The `VisionDataset` loader automatically thresholds these into clean binary float labels ($\{0.0, 1.0\}$).

---

## Prerequisites & Installation

### Python Environment (uv or venv)

We recommend using [uv](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fastral-sh%2Fuv) for fast, reproducible dependency resolution, or standard `venv`:

#### Option A: Using `uv` (Recommended)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install project dependencies
uv pip install -r src/python/requirements.txt
```

#### Option B: Using standard Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r src/python/requirements.txt
```

_Core Python dependencies:_

- `torch >= 2.1.0`, `torchvision >= 0.16.0` (required for `torchvision.transforms.v2`)
- `numpy`, `Pillow`, `opencv-python`, `matplotlib`, `tqdm`, `pyyaml`
- `onnx >= 1.15.0`, `onnxruntime >= 1.16.0`
- `jupyter`, `ipykernel` (for `.qmd` Quarto report rendering)

### C++ Inference Dependencies & System Packages

To compile and link the C++ inference engine, ensure your system has:

- **CMake** ($\ge 3.16$)
- **C++17 Compiler** (`g++` $\ge 9$ or `clang++` $\ge 10$)
- **OpenCV** ($\ge 4.x$)
- **ONNX Runtime C++ Library & Headers** ($\ge 1.14$)

#### Arch Linux

```bash
sudo pacman -Syu
sudo pacman -S cmake base-devel opencv onnxruntime --noconfirm
```

#### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libopencv-dev libonnxruntime-dev
```

### Locating ONNX Runtime on Linux (Arch / Debian / Ubuntu)

Our modern `src/cpp/CMakeLists.txt` automatically scans standard distribution directories for headers (`onnxruntime_cxx_api.h`) and libraries (`libonnxruntime.so`).

If your package manager installs ONNX Runtime in a custom location (e.g. `/usr/include/onnxruntime` or `/usr/local/include`), you can verify and pass them explicitly:

```bash
# Locate the shared library file on your system
find /usr -name "libonnxruntime.so*" 2>/dev/null

# Pass explicit locations to CMake if required
cmake -B src/cpp/build -S src/cpp \
  -DONNXRUNTIME_INCLUDE_DIR=/usr/include/onnxruntime \
  -DONNXRUNTIME_LIB=/usr/lib/libonnxruntime.so
```

---

## Step-by-Step Workflow

All commands below are executed from the **repository root directory**.

### 1. Data Exploration & Alignment (Quarto)

Verify paired image-mask alignment, nearest-neighbor mask interpolation, and binary label values before training:

```bash
# Interactive browser preview with live-reload
quarto preview notebook/dataset_exploration.qmd

# Or render directly to a standalone HTML report
quarto render notebook/dataset_exploration.qmd
```

### 2. Model Training

Launch U-Net training with the composite `BCEDiceLoss` objective and plateau learning-rate scheduling:

```bash
# Using standard Python
python src/python/train.py \
  --train_dir data/train \
  --val_dir data/test \
  --output_dir outputs \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4 \
  --padding 0

# Or directly via uv
uv run src/python/train.py --epochs 20 --batch_size 4 --padding 0
```

_Notes on training flags:_

- `--padding 0`: Uses the original unpadded U-Net topology ($416\times 416 \to 228\times 228$).
- `--padding 1`: Uses padded convolutions ($416\times 416 \to 416\times 416$).
- Checkpoints are automatically saved to `outputs/best_model.pth` (highest validation Dice) and `outputs/last_model.pth`.

### 3. Evaluation & Quantitative Benchmarks

Compute comprehensive segmentation metrics across the held-out test split (`data/test`):

```bash
# Run evaluation engine
python src/python/evaluate.py \
  --weights outputs/best_model.pth \
  --data_dir data/test \
  --batch_size 4 \
  --padding 0
```

The script reports:

- **Combined Loss**: $0.5 \times \text{BCE} + 0.5 \times \text{Dice Loss}$
- **Dice Score ($F_1$)**: Harmonic mean of precision and recall over foreground cell regions
- **IoU (Jaccard Index)**: Area of overlap divided by area of union
- **Pixel Accuracy**: Global per-pixel classification accuracy

### 4. ONNX Graph Serialization & Parity Verification

Export the trained PyTorch `.pth` checkpoint into a static, constant-folded ONNX graph with frozen `BatchNorm2d` parameters:

```bash
python src/python/export_onnx.py \
  --weights outputs/best_model.pth \
  --output outputs/model.onnx \
  --height 416 \
  --width 416 \
  --padding 0 \
  --opset 17
```

This script performs an automated numerical audit comparing PyTorch eager logits against ONNX Runtime:

```text
ONNX graph structure verified successfully (onnx.checker passed).
Numerical parity verified! Max absolute difference: 2.42e-06
```

### 5. C++ Inference Engine Build & Deployment

Build the optimized Release binary and run inference on blood smear images:

```bash
# 1. Configure and compile C++ engine
cmake -B src/cpp/build -S src/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build src/cpp/build -j$(nproc)

# 2. Run inference on a test blood smear image
./src/cpp/build/inference_app \
  --model outputs/model.onnx \
  --image data/test/original/sample.jpg \
  --output outputs/cpp_sample_1
```

The application outputs:

- `outputs/cpp_sample_1_mask.png`: Binary prediction mask (`uint8_t`, $\{0, 255\}$)
- `outputs/cpp_sample_1_overlay.png`: Anti-aliased red cell membrane contours overlaid on the aligned smear image
- Execution latency printed directly to the console:
  ```text
  --------------------------------------------------
  C++ ONNX Runtime Inference Complete!
    Forward Pass Latency : 14.28 ms
    Output Mask Size     : 228x228
    Saved Binary Mask    : outputs/cpp_sample_1_mask.png
    Saved Contour Overlay: outputs/cpp_sample_1_overlay.png
  --------------------------------------------------
  ```

### 6. Latency Profiling & Systems Benchmarking

Profile performance differences across runtime backends:

```bash
python src/python/benchmark_python.py \
  --model outputs/best_model.pth \
  --onnx outputs/model.onnx \
  --image data/test/original/sample.jpg \
  --runs 100
```

This benchmarks **PyTorch Eager**, **`torch.compile(mode="max-autotune")`**, and **ONNX Runtime (CPU/CUDA)**, isolating forward pass time ($T_{\text{infer}}$) from end-to-end I/O ($T_{\text{e2e}}$) and memory footprint (`RSS`).

### 7. Interactive Reports & Reveal.js Presentations

Render the comprehensive documentation suite located in `notebook/`:

```bash
# 1. U-Net Architecture & Mathematical Foundations
quarto render notebook/unet_architecture_notes.qmd

# 2. Training Progression & Evaluation Report
quarto render notebook/training_and_evalutation.qmd

# 3. C++ Inference Engine & xeus-cling Interactive Guide
quarto render notebook/cpp_inference.qmd

# 4. Reveal.js 11-Slide Technical Interview Presentation
quarto preview notebook/presentation.qmd
```

---

## Quantitative Results Summary

Trained on $1,169$ blood smears and benchmarked across $159$ held-out test smears (`data/test`):

| Evaluation Metric                         | Test Set Score | Mathematical Formulation                                                                    |
| :---------------------------------------- | :------------: | :------------------------------------------------------------------------------------------ |
| **Sørensen–Dice ($F_1$)**                 |  **`0.9475`**  | $D = \frac{2                                                                                | P \cap G                                         | }{         | P        | +                                                                          | G   | } = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ |
| **Jaccard Index (IoU)**                   |  **`0.9050`**  | $J = \frac{                                                                                 | P \cap G                                         | }{         | P \cup G | } = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{D}{2 - D}$ |
| **Pixel Accuracy**                        |  **`0.9660`**  | $\text{Acc} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$  |
| **Combined Loss**                         |  **`0.1262`**  | $\mathcal{L}_{\text{total}} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$ |
| **PyTorch $\leftrightarrow$ ONNX Parity** | **`2.42e-06`** | $\max_i                                                                                     | z_{\text{PyTorch}}^{(i)} - z_{\text{ONNX}}^{(i)} | < 10^{-5}$ |

---

## Configuration Reference

Default training and inference parameters in `config/` and CLI flags:

| Parameter            |         Default         | CLI Flag / Scope                      | Description                                                              |
| :------------------- | :---------------------: | :------------------------------------ | :----------------------------------------------------------------------- |
| `input_height`       |          `416`          | `--height` (train/export)             | Input image height dimension                                             |
| `input_width`        |          `416`          | `--width` (train/export)              | Input image width dimension                                              |
| `batch_size`         |           `4`           | `--batch_size` (train/eval)           | Mini-batch size for DataLoader                                           |
| `learning_rate`      |         `1e-4`          | `--lr` (train)                        | Initial learning rate for AdamW optimizer                                |
| `padding`            |           `0`           | `--padding` (train/export)            | `0` = unpadded ($228\times 228$ out), `1` = padded ($416\times 416$ out) |
| `opset_version`      |          `17`           | `--opset` (export)                    | Target ONNX operator set version                                         |
| `prob_threshold`     |          `0.5`          | `predict(..., 0.5f)` (C++)            | Probability threshold ($z > 0.0\text{f}$ logit threshold)                |
| `num_threads`        |           `4`           | `SegmentationEngine(..., 4)`          | Number of intra-op threads for ONNX Runtime                              |
| `normalization_mean` | `[0.485, 0.456, 0.406]` | `dataset.py` / `inference_engine.hpp` | ImageNet RGB mean values                                                 |
| `normalization_std`  | `[0.229, 0.224, 0.225]` | `dataset.py` / `inference_engine.hpp` | ImageNet RGB standard deviation                                          |

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

For full legal terms, see the project [LICENSE](https://www.google.com/search?q=LICENSE) file.
