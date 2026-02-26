# Desert Semantic Segmentation

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/license-MIT-green)

Robust terrain-aware semantic segmentation using synthetic digital twin data.

---

## Overview

This project implements a modular semantic segmentation pipeline designed to generalize across desert terrains using synthetic training data.

The system is built using a U-Net architecture with a ResNet-18 encoder backbone and optimized for stable domain generalization and rare-class robustness.

Final Validation mIoU (with TTA): **0.5402**

---

## Problem Statement

Desert and off-road terrains present unique challenges in computer vision:

- Sparse object boundaries
- High texture similarity between vegetation types
- Large dominant background regions
- Illumination variability

Traditional segmentation models trained on structured urban datasets often struggle in such low-structure environments.

This project investigates whether a synthetic-first training strategy can achieve robust cross-terrain generalization while maintaining strict dataset separation.

---

## Methodology

### Architecture

- Model: U-Net
- Encoder: ResNet-18 (ImageNet pretrained)
- Classes: 10
- Input Resolution: 256 × 256

### Loss Function

Hybrid loss:

- Weighted Cross Entropy
- Dice Loss

### Optimization

- Optimizer: Adam
- Learning Rate: 1e-3
- Scheduler: ReduceLROnPlateau
- Epochs: 30

### Inference Enhancement

- Test-Time Augmentation (Horizontal Flip Averaging)

---

## Model Architecture

The model uses a U-Net decoder with skip connections and a ResNet-18 encoder backbone for feature extraction.

- Downsampling encoder blocks
- Bottleneck feature representation
- Upsampling decoder with skip concatenation
- Final 1×1 convolution for 10-class prediction

Architecture Diagram:

![Architecture](docs/architecture.png)

---

## Results

| Metric          | Score      |
| --------------- | ---------- |
| Validation mIoU | 0.5313     |
| Final TTA mIoU  | **0.5402** |
| Epochs          | 30         |
| Backbone        | ResNet-18  |

### Observations

- Strong performance on dominant landscape classes
- Improved rare-class stability via hybrid loss
- Reduced prediction variance using TTA

---

## Installation

Clone repository:

```bash
git clone https://github.com/thekripaverse/Desert-Semantic-Segmentation.git
cd Desert-Semantic-Segmentation
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Training

```bash
python -m src.desert_segmentation.train \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-3
```

---

### Run Evaluation

```bash
python -m src.desert_segmentation.test
```

This will:

- Load best model weights
- Apply Test-Time Augmentation
- Compute final mIoU

---

## Project Structure

```
Desert-Semantic-Segmentation/
│
├── src/desert_segmentation/
│   ├── train.py
│   ├── test.py
│   ├── models/
│   ├── losses/
│   ├── datasets/
│   └── utils/
│
├── tests/
├── docs/
├── pyproject.toml
├── setup.cfg
├── requirements.txt
└── README.md
```

---

## Testing

Run unit tests:

```bash
pytest tests/
```

---

## Compliance Statement

- Model trained exclusively on provided training dataset.
- No validation or test images were used during training.
- Strict dataset separation maintained throughout development.

---

## Future Work

- Multi-scale training
- Deeper encoder backbone (ResNet-34 / ResNet-50)
- Domain adaptation techniques
- Semi-supervised fine-tuning
- Boundary-aware loss refinement

---

## License

This project is licensed under the MIT License.

```

```
