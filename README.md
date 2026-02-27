# Desert Semantic Segmentation

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/thekripaverse/Desert-Semantic-Segmentation/actions/workflows/ci.yml/badge.svg)

Production-oriented terrain-aware semantic segmentation system trained using synthetic digital twin data for deployment in unstructured desert and off-road environments.

---

## Overview

This project implements a modular U-Net based segmentation pipeline optimized for low-structure desert terrains.

The system emphasizes:

* Rare-class stability
* Boundary robustness
* Efficient inference optimization
* Deployment readiness

Final Validation mIoU (with TTA): **0.5402**

---

## Table of Contents

* Quick Start
* Installation
* Usage
* Demo UI
* Technical Architecture
* Performance Evaluation
* Dataset Strategy
* Deployment & Scalability
* Testing
* Market Landscape
* Competitive Positioning
* Business Model
* Risk Mitigation
* Future Work
* License

---

## Quick Start

```bash
git clone https://github.com/thekripaverse/Desert-Semantic-Segmentation.git
cd Desert-Semantic-Segmentation
pip install -r requirements.txt
python -m src.desert_segmentation.train
```

Run evaluation:

```bash
python -m src.desert_segmentation.test
```

---

## Installation

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python -m src.desert_segmentation.train \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-3
```

### Evaluation

```bash
python -m src.desert_segmentation.test
```

---

## Demo UI

A Streamlit-based interactive demo is included.

Run locally:

```bash
pip install streamlit
streamlit run apps/streamlit_app.py
```

### Demo Screenshots

<img width="1600" src="https://github.com/user-attachments/assets/aaa7632d-2f8d-4fb6-8552-bcae6f381199" />
<img width="1600" src="https://github.com/user-attachments/assets/837487c5-9bc9-4052-8a55-f75bb6a75c0d" />
<img width="1600" src="https://github.com/user-attachments/assets/469a1222-aa99-4c62-a86a-7d407181e168" />
<img width="1600" src="https://github.com/user-attachments/assets/a0aa3204-b1a6-4f1e-a6d5-5e092743deec" />

Demonstrates real-time inference workflow and mask visualization.

---

## Technical Architecture

### Model Design

* Architecture: U-Net
* Encoder: ResNet-18 (ImageNet pretrained)
* Classes: 10
* Input Resolution: 256×256
* Framework: PyTorch + segmentation_models_pytorch

### Loss Strategy

Hybrid Loss:

* Weighted Cross Entropy (class imbalance mitigation)
* Dice Loss (boundary stabilization)

### Optimization Strategy

* Optimizer: Adam
* Learning Rate: 1e-3
* Scheduler: ReduceLROnPlateau
* Epochs: 30

### Inference Optimization

* Test-Time Augmentation (Horizontal Flip Averaging)
* +0.9% mIoU improvement without increasing model size

---

## Performance Evaluation

| Metric          | Score      |
| --------------- | ---------- |
| Validation mIoU | 0.5313     |
| Final TTA mIoU  | **0.5402** |
| Convergence     | Stable     |

### Observations

* Strong spatial consistency in dominant terrain classes
* Improved rare-class detection via hybrid loss
* Reduced boundary noise through TTA
* Stable convergence without overfitting

---

## Dataset Strategy

The model was trained exclusively on the provided synthetic dataset with strict dataset separation.

### Synthetic-First Advantages

* Lower annotation cost
* Rare-class scenario generation
* Controlled simulation environments
* Faster experimentation cycles

Strict separation between training and validation was maintained.

---

## Deployment & Scalability

### Deployment Options

* TorchScript export
* ONNX export
* REST API wrapper
* ROS integration
* Streamlit demo interface

### Scalability Mechanisms

* Backbone upgrades (ResNet-34, ResNet-50)
* Multi-scale training
* Semi-supervised fine-tuning
* Domain adaptation pipelines
* Edge-device optimization

---

## Testing

Run unit tests:

```bash
pytest tests/
```

Unit tests validate:

* Model forward pass
* IoU computation
* Structural integrity

---

## Market Landscape

### Real-World Problem

Autonomous systems operating in unstructured desert environments face:

* Sparse object boundaries
* Texture similarity between terrain classes
* Limited annotated datasets
* Simulation-to-real domain shift

Most segmentation systems are optimized for structured urban data and fail in low-structure terrain.

This project addresses that gap using a synthetic-first training strategy.

### Economic Advantage

Leveraging synthetic digital twin data reduces:

* Annotation cost
* Data collection overhead
* Rare-case data scarcity

Enabling scalable terrain perception development.

---

## Competitive Positioning

### Differentiators

* Focus on low-structure terrain environments
* Synthetic-first training strategy
* Rare-class stabilization via hybrid loss
* Modular architecture
* Deployment-ready UI

### Competitive Comparison

| Feature                    | Urban Models | This Project |
| -------------------------- | ------------ | ------------ |
| Structured terrain focus   | Yes          | No           |
| Low-structure optimization | No           | Yes          |
| Synthetic-first pipeline   | Rare         | Yes          |
| Rare-class stabilization   | Limited      | Hybrid       |
| Deployment UI              | Not standard | Included     |

---

## Business Model

### Primary Customers

* Robotics manufacturers
* Mining automation companies
* Defense contractors
* Simulation platform providers

### Revenue Channels

* B2B perception module licensing
* Custom terrain adaptation services
* Simulation-to-real consulting
* On-premise integration contracts

---

## Risk Mitigation

### Synthetic-to-Real Gap

Mitigation methods:

* Domain randomization
* Limited real-world fine-tuning
* Test-Time Augmentation
* Boundary-focused hybrid loss

---

## Future Work

* Multi-domain transfer learning
* Cross-terrain evaluation benchmarks
* Edge-optimized model variants
* Real-world dataset fine-tuning
* Embedded device benchmarking

---

## License

MIT License

---
