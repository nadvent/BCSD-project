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
  - [Systems & Latency Benchmarks (C++ ORT vs. Python ORT vs. PyTorch)](#systems--latency-benchmarks-c-ort-vs-python-ort-vs-pytorch)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Overview

Accurate identification and boundary localization of hematological cells (erythrocytes/red blood cells, leukocytes/white blood cells, and platelets) in peripheral blood smears is an essential diagnostic step in clinical pathology. To address this, the project implements a **U-Net** segmentation architecture—widely regarded as an industry standard in biomedical imaging for its ability to extract fine-grained local morphological features, fuse multi-scale spatial context via symmetric skip connections, and generalize effectively from small clinical datasets when paired with spatial data augmentation.

The model architecture is primarily based on the foundational 2015 U-Net paper by Ronneberger et al. ([U-Net: Convolutional Networks for Biomedical Image Segmentation](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597)), supporting both **`valid` (unpadded)** convolutions—which eliminate border zero-padding artifacts and require center-cropping along skip connections—and **`same` (one-padded)** convolutions—which preserve exact spatial dimensions ($H \times W$) throughout the contracting and expanding paths. Do note that since `same` padded convolutions have a larger output size (dimensionality), there is more complexity to train and evaluate the model.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597). In _Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015_ (pp. 234–241). Springer. (`arXiv:1505.04597`)

This repository bridges research experimentation and production deployment:

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
│   ├── best_model.pth               # Optimal PyTorch weights (Test Dice: 0.9655)
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

I personally use[uv](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fastral-sh%2Fuv) for fast, reproducible dependency resolution, or standard `venv`:

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

### Locating ONNX Runtime on Linux (Arch / Debian / Ubuntu)

`src/cpp/CMakeLists.txt` automatically scans standard distribution directories for headers (`onnxruntime_cxx_api.h`) and libraries (`libonnxruntime.so`).

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
  --image data/test/original/e02301ac-68f7-4daf-b2ba-254b917f1cca.jpg \
  --output outputs/cpp_sample_1
```

The application outputs:

- `outputs/cpp_sample_1_mask.png`: Binary prediction mask (`uint8_t`, $\{0, 255\}$)
- `outputs/cpp_sample_1_overlay.png`: Anti-aliased red cell membrane contours overlaid on the aligned smear image
- Execution latency and provider info printed directly to the console:
  ```text
  Loading ONNX model from: outputs/model.onnx
  --------------------------------------------------
  C++ ONNX Runtime Inference Complete!
    Execution Provider   : CUDAExecutionProvider
    Input Dimensions     : 1944x1383
    Model Target Res     : 416x416
    Forward Pass Latency : 42.27 ms
    Output Mask Size     : 228x228
    Saved Binary Mask    : outputs/cpp_sample_1_mask.png
    Saved Contour Overlay: outputs/cpp_sample_1_overlay.png
  --------------------------------------------------
  ```

### 6. Latency Profiling & Systems Benchmarking

Profile performance differences across runtime backends:

```bash
python src/python/benchmark_python.py
```

Console benchmark output on NVIDIA RTX 3050 Laptop GPU (100 timed runs, 25 warmup passes):

```text
Benchmarking on: cuda (NVIDIA GeForce RTX 3050 Laptop GPU)

Framework                    | Median (P50) | Mean ± Std         | P95 Latency
------------------------------------------------------------------------------
PyTorch Eager                |    33.71 ms |  33.73 ± 0.06   ms |    33.88 ms
torch.compile(reduce-overhead) |    31.28 ms |  31.29 ± 0.17   ms |    31.57 ms
Python ONNX Runtime          |    34.39 ms |  34.41 ± 0.13   ms |    34.64 ms
```

This benchmarks **PyTorch Eager**, **`torch.compile(mode="reduce-overhead")`**, and **Python ONNX Runtime**, evaluating kernel launch synchronization and execution latency under matched batch sizes ($N=1$, $3\times 416\times 416$).

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
| **Sørensen–Dice ($F_1$)**                 |  **`0.9655`**  | $D = \frac{2 | P \cap G | }{ | P | + | G | } = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ |
| **Jaccard Index (IoU)**                   |  **`0.9344`**  | $J = \frac{ | P \cap G | }{ | P \cup G | } = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{D}{2 - D}$ |
| **Pixel Accuracy**                        |  **`0.9766`**  | $\text{Acc} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$  |
| **Combined Loss**                         |  **`0.0644`**  | $\mathcal{L}_{\text{total}} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$ |
| **PyTorch $\leftrightarrow$ ONNX Parity** | **`2.42e-06`** | $\max_i | z_{\text{PyTorch}}^{(i)} - z_{\text{ONNX}}^{(i)} | < 10^{-5}$ |

---

### Systems & Latency Benchmarks (C++ ORT vs. Python ORT vs. PyTorch)

Benchmarked on **NVIDIA GeForce RTX 3050 Laptop GPU** (CUDA 13.4, cuDNN 9) and **AMD Ryzen 7 6800HS CPU** ($N=1$, input rescaled to $3\times 416\times 416$, 100 timed iterations, 25 warmup passes):

#### Tier 1: Pure Model Forward Pass Latency ($T_{\text{infer}}$)

| Runtime / Framework | Provider / Device | Median ($P_{50}$) | Mean $\pm$ Std | $P_{95}$ Latency |
| :------------------ | :---------------- | :---------------: | :------------: | :--------------: |
| **PyTorch (`torch.compile`)** | CUDA (`reduce-overhead`) | **`31.09 ms`** | `31.13 ± 0.13 ms` | `31.33 ms` |
| **PyTorch (`torch.compile`)** | CUDA (`default`)         | `31.68 ms` | `31.76 ± 0.35 ms` | `32.44 ms` |
| **PyTorch Eager**             | CUDA                     | `33.74 ms` | `33.75 ± 0.07 ms` | `33.87 ms` |
| **C++ ONNX Runtime**          | `CUDAExecutionProvider`  | **`33.84 ms`** | `34.16 ± 0.59 ms` | `35.20 ms` |
| **Python ONNX Runtime**       | `CUDAExecutionProvider`  | **`33.90 ms`** | `33.91 ± 0.07 ms` | `34.04 ms` |
| **C++ ONNX Runtime (CPU)**    | `CPUExecutionProvider` (8T) | `192.45 ms` | `193.10 ± 1.15 ms` | `195.40 ms` |

#### Tier 2: In-Memory End-to-End Pipeline ($T_{\text{pre}} + T_{\text{infer}} + T_{\text{post}}$)

| Runtime / Framework | Preprocessing ($T_{\text{pre}}$) | Forward Pass ($T_{\text{infer}}$) | Postprocessing ($T_{\text{post}}$) | Total In-Memory ($P_{50}$) | Throughput |
| :------------------ | :-----------------------------: | :-------------------------------: | :--------------------------------: | :------------------------: | :--------: |
| **C++ ONNX Runtime** | **`0.53 ms`** | `33.82 ms` | **`0.62 ms`** | **`35.00 ms`** | **`28.57 FPS`** |
| **PyTorch (`torch.compile`)** | `5.14 ms` | **`31.39 ms`** | `0.93 ms` | `37.62 ms` | `26.58 FPS` |
| **Python ONNX Runtime** | `3.46 ms` | `34.01 ms` | `0.72 ms` | `38.21 ms` | `26.17 FPS` |

> **Key Takeaway**: While `torch.compile(mode="reduce-overhead")` delivers the fastest pure GPU forward pass via CUDA Graph launch capture, the **C++ ONNX Runtime engine achieves the highest overall system throughput (`28.57 FPS`)** because its fused single-pass ImageNet normalization and OpenCV SIMD contour thresholding avoid Python GIL, NumPy array transpositions, and dynamic heap re-allocations.


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
| `device`             |         `auto`          | `--device` / `-d` (C++)               | Execution provider (`auto`, `cuda`, `dnnl`, `cpu`)                       |
| `num_threads`        |    `physical cores`     | `--threads` / `-t` (C++)              | Number of intra-op threads for ONNX Runtime                              |
| `warmup`             |           `1`           | `--warmup` / `-w` (C++)               | Number of warmup forward passes before timing                            |
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
