# 🫀 Cardiac Structure Segmentation on the CAMUS Dataset

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-CAMUS-blueviolet)
![Model](https://img.shields.io/badge/Architecture-UNet%20ResNet34-red)

Multiclass segmentation of cardiac structures in echocardiographic images using a **UNet with ResNet34 encoder** trained on the [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html). The model identifies three anatomical structures — **LV endocardium**, **LV myocardium**, and **left atrium** — across apical 2-chamber and 4-chamber views.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Accurate segmentation of cardiac structures in echocardiography is a critical step in clinical workflows — enabling automated calculation of metrics such as **ejection fraction** and **ventricular volumes**, which are central to the diagnosis of cardiovascular disease.

This project implements a full deep learning pipeline covering:

- NIfTI data loading and preprocessing (`.nii.gz` format) via SimpleITK
- Bicubic resizing and min-max normalisation of ultrasound intensities
- Custom `PyTorch Dataset` (`CHDataset`) for multi-view echocardiographic data
- 80/10/10 train/validation/test split
- Early stopping with patience-based checkpointing
- Per-epoch metrics: Dice, IoU, Accuracy (macro-averaged)
- Contour-based overlay visualisation of predicted vs ground truth masks
- Video and GIF generation of segmentation results on unseen echocardiographic sequences

---

## Dataset

**CAMUS — Cardiac Acquisitions for Multi-structure Ultrasound Segmentation**

| Property | Value |
|---|---|
| Patients | 500 |
| Views | Apical 2-chamber (2CH) and 4-chamber (4CH) |
| Annotations | Expert-labelled segmentation masks |
| Format | NIfTI (`.nii.gz`) |
| Classes | Background (0), LV Endocardium (1), LV Myocardium (2), Left Atrium (3) |

The dataset is publicly available at: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html

> **Note:** The dataset is not included in this repository. See [Usage](#usage) for setup instructions.

---

## Architecture

### Model: UNet with ResNet34 Encoder

```
Input (1-channel grayscale, 320×320)
        │
   ResNet34 Encoder (ImageNet pretrained)
        │
   Skip Connections
        │
   Decoder with upsampling
        │
Output (4-class segmentation map, 320×320)
```

| Component | Choice | Rationale |
|---|---|---|
| Architecture | UNet | Proven encoder-decoder with skip connections for medical image segmentation |
| Encoder | ResNet34 | Strong pretrained features; good balance of depth and efficiency |
| Pretrained weights | ImageNet | Transfer learning accelerates convergence on limited medical data |
| Input channels | 1 | Greyscale echocardiographic images |
| Output classes | 4 | Background + LV Endocardium + LV Myocardium + Left Atrium |

### Loss Function & Optimiser

| Setting | Value |
|---|---|
| Loss | CrossEntropyLoss |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Image size | 320 × 320 |
| Epochs | 20 |
| Early stopping patience | 5 epochs |
| Data split | 80% train / 10% val / 10% test |

---

## Results

All metrics are macro-averaged across the 4 classes.

### Training Progression

| Epoch | Train Loss | Val Loss | Train Dice | Val Dice | Train IoU | Val IoU | Train Acc | Val Acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6613 | 0.4030 | 0.7296 | 0.8879 | 0.5874 | 0.8039 | 0.7938 | 0.9083 |
| 5 | 0.1190 | 0.1201 | 0.9425 | 0.9346 | 0.8931 | 0.8796 | 0.9442 | 0.9298 |
| 10 | 0.0622 | 0.0820 | 0.9595 | 0.9442 | 0.9232 | 0.8960 | 0.9596 | 0.9380 |
| 20 | 0.0339 | 0.0702 | **0.9738** | **0.9537** | **0.9493** | **0.9128** | **0.9736** | **0.9565** |

### Final Evaluation

| Split | Loss | Dice ↑ | IoU ↑ | Accuracy ↑ |
|---|---|---|---|---|
| Train | 0.0339 | 0.9738 | 0.9493 | 0.9736 |
| Validation | 0.0702 | 0.9537 | 0.9128 | 0.9565 |
| Test | 0.0777 | — | — | — |

> The small gap between validation and test loss (0.0702 vs 0.0777) indicates no significant overfitting.

---

## Project Structure

```
camus-segmentation/
├── camus_segmentation.py    # Full pipeline: data loading, model, training, evaluation
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes data, checkpoints, cache
└── README.md
```

> Model checkpoints (`.pth`) and the dataset are excluded from version control due to size.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/camus-segmentation.git
cd camus-segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Request access and download from the [CAMUS challenge page](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html).

Organise the data as follows:

```
ch_database/
├── 2CH/
│   ├── images/      ← patient_2CH_ED.nii.gz, ...
│   └── masks/       ← patient_2CH_ED_gt.nii.gz, ...
└── 4CH/
    ├── images/
    └── masks/
```

### 4. Upload to Google Drive

The script is designed to run on **Google Colab**. Upload `ch_database/` to your Drive at:

```
MyDrive/ch_database/
```

---

## Usage

### Running on Google Colab

1. Upload `camus_segmentation.py` to your Colab session or mount from Drive
2. Mount your Google Drive when prompted
3. Verify the dataset path:
   ```python
   data_path = "/content/drive/MyDrive/ch_database"
   ```
4. Run the script — training, evaluation, and visualisation execute sequentially

### Key outputs saved to Google Drive

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint by validation loss |
| `model_epoch_N.pth` | Periodic checkpoint every 5 epochs |
| `final_model.pth` | Model weights after full training |
| `tested_model.pth` | Model weights after test evaluation |
| `New_Masks/` | Predicted masks for unseen echocardiographic frames |
| `segmentation_video.mp4` | Video of segmentation on new sequences |
| `segmentation_animation.gif` | GIF with colour-coded class overlays |

---

## Limitations

- **Google Colab dependency:** Paths are hard-coded to `/content/drive/MyDrive/`. Adapt for local execution.
- **Single slice per volume:** Only the first slice is used from 3D NIfTI files. Temporal or volumetric modelling is not implemented.
- **No data augmentation:** Adding rotation, flipping, or elastic deformation would likely improve generalisation on edge cases.
- **Video inference uses softmax argmax:** The multiclass output is converted to a single-channel mask for visualisation; per-class confidence maps are not exported.

---

## References

- Leclerc, S. et al. (2019). *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography.* IEEE Transactions on Medical Imaging.
- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — Iakubovskii, P.

---

## Author

**Malcolm** — MSc Artificial Intelligence (Healthcare), University of West London
[LinkedIn](#) · [GitHub](#)
