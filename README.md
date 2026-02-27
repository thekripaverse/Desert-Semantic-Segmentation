# Desert Semantic Segmentation

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![CI](https://github.com/thekripaverse/Desert-Semantic-Segmentation/actions/workflows/ci.yml/badge.svg?style=flat-square)
![mIoU](https://img.shields.io/badge/mIoU-54.02%25-orange?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/thekripaverse/Desert-Semantic-Segmentation?style=for-the-badge)

**Production-ready terrain-aware semantic segmentation for autonomous systems in unstructured desert and off-road environments.**

*Trained on synthetic digital twin data · Optimized for rare-class stability · Deployment-ready*

[Quick Start](#quick-start) · [Results](#performance-evaluation) · [Architecture](#technical-architecture) · [Demo](#demo-ui) · [Deployment](#deployment--scalability)

</div>

---

## Why This Project?

Most segmentation models are trained on urban datasets (Cityscapes, ADE20K) and **fail catastrophically in desert terrain** — where there are no lane markings, few rigid objects, and extreme texture similarity between terrain classes like gravel, sand, and rock.

This project solves that with:
- A **synthetic-first training pipeline** using digital twin data to eliminate expensive annotation costs
- A **hybrid loss strategy** for rare-class stabilization
- **Test-Time Augmentation (TTA)** for boundary robustness
- A **fully modular, deployment-ready** architecture targeting robotics and autonomous vehicles

> Final Validation mIoU (with TTA): **0.5402** — trained entirely on synthetic data, no manual annotation required.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Demo UI](#demo-ui)
- [Technical Architecture](#technical-architecture)
- [Performance Evaluation](#performance-evaluation)
- [Dataset Strategy](#dataset-strategy)
- [Deployment & Scalability](#deployment--scalability)
- [Testing](#testing)
- [Market Landscape](#market-landscape)
- [Competitive Positioning](#competitive-positioning)
- [Business Model](#business-model)
- [Risk Mitigation](#risk-mitigation)
- [Future Work](#future-work)
- [License](#license)

---

## Quick Start

```bash
git clone https://github.com/thekripaverse/Desert-Semantic-Segmentation.git
cd Desert-Semantic-Segmentation
pip install -r requirements.txt

# Train the model
python -m src.desert_segmentation.train

# Evaluate
python -m src.desert_segmentation.test
```

> **One-liner to get predictions on your own images:**
> ```bash
> python -m src.desert_segmentation.predict --image path/to/your/image.jpg
> ```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/thekripaverse/Desert-Semantic-Segmentation.git
cd Desert-Semantic-Segmentation

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+ (optional but recommended)

---

## Usage

### Training

```bash
python -m src.desert_segmentation.train \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 8 | Batch size per GPU |
| `--lr` | 1e-3 | Initial learning rate |
| `--backbone` | resnet18 | Encoder backbone |

### Evaluation

```bash
python -m src.desert_segmentation.test
```

---

## Demo UI

An interactive **Streamlit-based demo** is included for real-time inference and mask visualization.

```bash
pip install streamlit
streamlit run apps/streamlit_app.py
```

### Screenshots

<img width="1600" src="https://github.com/user-attachments/assets/aaa7632d-2f8d-4fb6-8552-bcae6f381199" />
<img width="1600" src="https://github.com/user-attachments/assets/837487c5-9bc9-4052-8a55-f75bb6a75c0d" />
<img width="1600" src="https://github.com/user-attachments/assets/469a1222-aa99-4c62-a86a-7d407181e168" />
<img width="1600" src="https://github.com/user-attachments/assets/a0aa3204-b1a6-4f1e-a6d5-5e092743deec" />

> Upload any desert/off-road image and get instant segmentation masks with per-class confidence overlays.

---

## Technical Architecture

```
Input Image (256×256)
       │
  ┌────▼────┐
  │ ResNet-18│  ← ImageNet pretrained encoder
  │ Encoder  │
  └────┬────┘
       │  Skip connections
  ┌────▼────┐
  │  U-Net  │  ← Decoder with upsampling blocks
  │ Decoder  │
  └────┬────┘
       │
  ┌────▼────────────┐
  │ 10-Class Softmax│  ← Output segmentation mask
  └─────────────────┘
```

### Model Configuration

| Component | Value |
|-----------|-------|
| Architecture | U-Net |
| Encoder | ResNet-18 (ImageNet pretrained) |
| Classes | 10 terrain categories |
| Input Resolution | 256 × 256 |
| Framework | PyTorch + segmentation_models_pytorch |

### Loss Strategy

A **hybrid loss** is used to handle both class imbalance and boundary quality simultaneously:

```
Total Loss = α · WeightedCrossEntropy + β · DiceLoss
```

- **Weighted Cross Entropy** — compensates for class frequency imbalance across rare terrain types
- **Dice Loss** — directly optimizes overlap, producing sharper, more stable boundaries

### Optimization

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Scheduler | ReduceLROnPlateau |
| Epochs | 30 |
| TTA | Horizontal Flip Averaging |

### Test-Time Augmentation (TTA)

Horizontal flip averaging at inference time yields **+0.9% mIoU** with zero additional parameters.

---

## Performance Evaluation

| Metric | Score |
|--------|-------|
| Validation mIoU (baseline) | 0.5313 |
| **Final mIoU (with TTA)** | **0.5402** |
| TTA Improvement | +0.9% |
| Convergence | Stable (no overfitting) |

### Key Observations

- Strong spatial consistency in dominant terrain classes (sand, gravel, rock)
- Improved rare-class recall via hybrid loss weighting
- Reduced boundary noise and class bleeding through TTA
- Stable training curve with ReduceLROnPlateau — no divergence observed

---

## Dataset Strategy

The model is trained **exclusively on synthetic data** using a digital twin environment, maintaining strict train/validation separation.

### Why Synthetic-First?

| Factor | Traditional Annotation | Synthetic-First (Ours) |
|--------|------------------------|------------------------|
| Annotation Cost | High ($$$) | Near-zero |
| Rare Class Coverage | Limited | Fully controllable |
| Dataset Scale | Constrained | Unlimited |
| Iteration Speed | Slow | Fast |

Synthetic data generation also allows controlled simulation of challenging edge cases: dusk lighting, sandstorm visibility reduction, and extreme terrain transitions.

---

## Deployment & Scalability

### Export Options

```bash
# TorchScript export
python -m src.desert_segmentation.export --format torchscript

# ONNX export (for cross-platform inference)
python -m src.desert_segmentation.export --format onnx

# REST API server
python -m src.desert_segmentation.serve --port 8080
```

### Deployment Targets

| Target | Status |
|--------|--------|
| TorchScript | Supported |
| ONNX | Supported |
| REST API | Included |
| ROS Integration | Compatible |
| Edge Devices | In Progress |
| Streamlit Demo | Included |

### Scalability Path

- **Higher accuracy:** Drop-in backbone upgrade to ResNet-34 or ResNet-50
- **Domain adaptation:** Semi-supervised fine-tuning on real desert imagery
- **Edge deployment:** Quantization + pruning pipeline (in progress)
- **Multi-terrain:** Transfer learning to snow, jungle, and lunar terrains

---

## Testing

```bash
pytest tests/ -v
```

Unit tests cover:
- Model forward pass (correctness + output shape)
- IoU metric computation
- Structural integrity of the segmentation pipeline

CI is run on every push via GitHub Actions. See badge at top of README.

---

## Market Landscape

### The Problem

Autonomous systems deployed in unstructured environments face:

- **Sparse boundaries** — no lane markings, curbs, or signage
- **Texture similarity** — sand, gravel, and rock look nearly identical to standard models
- **Label scarcity** — annotating desert terrain is expensive and time-consuming
- **Sim-to-real gap** — models trained on structured data degrade dramatically

### Economic Opportunity

The global autonomous off-road vehicle market is projected to exceed **$10B by 2030**, encompassing mining automation, defense robotics, and planetary exploration. Scalable terrain perception is a critical unsolved component.

Our synthetic-first pipeline dramatically lowers the cost barrier to building domain-specific perception systems.

---

## Competitive Positioning

### How We Compare

| Feature | Urban Models (e.g., DeepLab, SegFormer) | This Project |
|---------|----------------------------------------|--------------|
| Structured terrain focus | Optimized | Not applicable |
| Low-structure terrain | Degrades | Purpose-built |
| Synthetic-first pipeline | Rare | Core strategy |
| Rare-class stabilization | Limited | Hybrid loss |
| Edge deployment path | Heavy | ResNet-18 base |
| Interactive demo UI | Not standard | Streamlit included |
| Open source | Varies | MIT licensed |

---

## Business Model

### Primary Customers

| Segment | Use Case |
|---------|----------|
| Robotics Manufacturers | Perception modules for off-road autonomous platforms |
| Mining Automation | Haul truck and excavator terrain awareness |
| Defense Contractors | UAV and ground vehicle navigation in unstructured terrain |
| Simulation Platforms | Synthetic data pipelines and terrain labeling |

### Revenue Channels

- **B2B Module Licensing** — integrate our segmentation backbone into existing perception stacks
- **Custom Terrain Adaptation** — fine-tune to client-specific environments (lunar, arctic, jungle)
- **Simulation-to-Real Consulting** — domain adaptation for teams with limited real-world data
- **On-Premise Integration** — white-label deployment contracts with hardware partners

---

## Risk Mitigation

### Synthetic-to-Real Domain Gap

| Risk | Mitigation |
|------|------------|
| Texture mismatch | Domain randomization during synthetic generation |
| Lighting variation | Augmented light conditions in simulation |
| Boundary artifacts | Boundary-focused Dice loss component |
| Generalization limits | Semi-supervised fine-tuning pathway on real samples |

---

## Future Work

- [ ] Multi-domain transfer learning (snow, jungle, lunar)
- [ ] Cross-terrain evaluation benchmarks
- [ ] Edge-optimized model variants (MobileNet encoder, INT8 quantization)
- [ ] Real-world fine-tuning dataset release
- [ ] Embedded device benchmarking (Jetson Nano, Raspberry Pi)
- [ ] Integration with ROS2 navigation stack

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push and open a Pull Request

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with love for autonomous systems in the wild.

**If this project helped you, please give it a star!**

</div>
