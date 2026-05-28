# Camus Segmentation
# 🫀 Cardiac Structure Segmentation on the CAMUS Dataset

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-CAMUS-blueviolet)
![Model](https://img.shields.io/badge/Architecture-UNet%2B%2B-red)

Multiclass segmentation of cardiac structures in echocardiographic images using a **UNet++ with ResNet34 encoder** trained on the [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html). The model identifies three anatomical structures — **LV endocardium**, **LV myocardium**, and **left atrium** — across apical 2-chamber and 4-chamber views.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Known Issues & Limitations](#known-issues--limitations)
- [References](#references)

---

## Overview

Accurate segmentation of cardiac structures in echocardiography is a critical step in clinical workflows — enabling the automated calculation of metrics such as **ejection fraction** and **ventricular volumes**, which are central to the diagnosis of cardiovascular disease.

This project implements a full deep learning pipeline covering:

- NIfTI data loading and preprocessing (`.nii.gz` format)
- Z-score normalisation of ultrasound intensities
- Custom `PyTorch Dataset` for multi-view echocardiographic data
- **5-Fold Cross-Validation** with early stopping
- Per-class evaluation metrics: Dice, IoU, Precision, Recall, F1-score
- Training/validation loss curves and confusion matrix visualisation

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

### Model: UNet++ with ResNet34 Encoder

```
Input (1-channel grayscale, 256×256)
        │
   ResNet34 Encoder (ImageNet pretrained)
        │
   Dense Skip Connections (UNet++ nested blocks)
        │
   Decoder with upsampling
        │
Output (4-class segmentation map, 256×256)
```

| Component | Choice | Rationale |
|---|---|---|
| Architecture | UNet++ | Dense skip connections improve gradient flow and fine-grained boundary detection vs standard UNet |
| Encoder | ResNet34 | Strong pretrained features; good balance of depth and efficiency |
| Pretrained weights | ImageNet | Transfer learning accelerates convergence on limited medical data |
| Input channels | 1 | Greyscale echocardiographic images |
| Output classes | 4 | Background + 3 cardiac structures |

### Loss Function

**CrossEntropyLoss** — appropriate for multiclass pixel-wise classification with class imbalance between background and anatomical structures.

### Optimiser & Training

| Hyperparameter | Value |
|---|---|
| Batch size | 16 |
| Epochs | 20 (with early stopping) |
| Early stopping patience | 5 epochs |
| Cross-validation | 5-Fold (80/20 train/val split per fold) |
| Image size | 256 × 256 |

---

## Results

Metrics are computed per class across the test set (held-out fold) using the best checkpoint per fold.

| Class | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|
| Background | — | — | — | — |
| LV Endocardium | *see run* | *see run* | *see run* | *see run* |
| LV Myocardium | *see run* | *see run* | *see run* | *see run* |
| Left Atrium | *see run* | *see run* | *see run* | *see run* |

> Results depend on the training run. Populate this table after training by reading `metrics.csv` from Google Drive.

Training curves and per-class confusion matrices are generated automatically at the end of the notebook.

---

## Project Structure

```
camus-segmentation/
├── Assignment_1_CAMUS_Segmentation.ipynb   # Main notebook (Colab)
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Excludes data, checkpoints, cache
└── README.md
```

> Model checkpoints (`.pth`) and the dataset are excluded from version control due to size. See below for setup.

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

Or manually:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install nibabel SimpleITK h5py
pip install scikit-learn scikit-image
pip install matplotlib seaborn pandas tqdm opencv-python Pillow
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

The notebook is designed to run on **Google Colab**. Upload `ch_database/` to your Drive at:

```
MyDrive/ch_database/
```

---

## Usage

### Running on Google Colab

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive when prompted
3. Verify the dataset path in the second cell:
   ```python
   data_path = "/content/drive/MyDrive/ch_database"
   ```
4. Run all cells sequentially

### Key outputs saved to Google Drive

| File | Description |
|---|---|
| `best_model_fold_N.pth` | Best checkpoint per fold (lowest val loss) |
| `checkpoint_epoch_N_fold_N.pth` | Periodic checkpoints every 10 epochs |
| `metrics.csv` | Training and validation metrics per epoch |
| `model_evaluated_on_test.pth` | Final model after test evaluation |

---

## Known Issues & Limitations

- **Google Colab dependency:** Data loading is hard-coded to `/content/drive/MyDrive/`. Adapt paths for local execution.
- **3D volumes:** Only the central slice is used from 3D NIfTI files. Temporal/volumetric modelling is not implemented.
- **Data augmentation:** The current pipeline does not apply augmentation (rotation, flipping). Adding it would likely improve generalisation.
- **Class label mapping:** Verify that your CAMUS masks use values `{0, 1, 2, 3}` with `np.unique(mask_data)` before training.

---

## References

- Leclerc, S. et al. (2019). *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography.* IEEE Transactions on Medical Imaging.
- Zhou, Z. et al. (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* MICCAI.
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) — Iakubovskii, P.

---

## Author

**Malcolm** — MSc Artificial Intelligence (Healthcare), University of [Your University]  
[LinkedIn](#) · [GitHub](#)
