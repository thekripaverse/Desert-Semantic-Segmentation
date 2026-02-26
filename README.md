# Desert Semantic Segmentation – Hack for Green Bharat

![Python](https://img.shields.io/badge/python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project presents a semantic segmentation model trained exclusively on the synthetic desert dataset provided as part of the Hack for Green Bharat challenge.

The objective is to accurately segment desert environments into 10 semantic classes and maximize the mean Intersection over Union (mIoU) score while maintaining strict separation between training, validation, and test datasets.

Final Validation mIoU (with Test-Time Augmentation): **0.5402**

---

## Project Structure

```
Desert-Semantic-Segmentation/
├── train.py                          # Model training script
├── test.py                           # Inference and evaluation script
├── config.yaml                       # Hyperparameter configuration
├── best_model_final_0.5402.pth       # Trained model weights
├── requirements.txt                  # Required dependencies
└── README.md                         # Documentation
```

---

## Dataset Usage Policy

- The model was trained exclusively on the provided training dataset.
- Validation data was used only for evaluation during training.
- No test images were used during training.
- Strict separation between training, validation, and test sets was maintained throughout the workflow.

Any use of designated testing images for training purposes is strictly prohibited as per challenge guidelines.

---

## Model Details

Model : https://drive.google.com/file/d/16wuMMTJE-5g-sWX-uvDCfB-NixFuAwEg/view?usp=drive_link

Architecture: U-Net  
Encoder Backbone: ResNet-18  
Pretrained Weights: ImageNet  
Number of Classes: 10  
Input Resolution: 256 × 256

Loss Function:

- Weighted Cross Entropy Loss
- Dice Loss (combined with Cross Entropy)

Optimizer:

- Adam

Learning Rate:

- 1e-3

Scheduler:

- ReduceLROnPlateau

Batch Size:

- 8 (GPU training)

Epochs:

- 30

Inference Enhancement:

- Test-Time Augmentation (Horizontal Flip)

---

## Environment Setup

### 1. Create Environment (Recommended)

Using conda:

conda create -n desert_seg python=3.9
conda activate desert_seg

Or using venv:

python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)

---

### 2. Install Dependencies

pip install -r requirements.txt

If installing manually:

pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install numpy opencv-python tqdm pyyaml

---

## Training Instructions

To train the model:

python train.py

Training will:

- Load training and validation datasets
- Apply data augmentation
- Train for 30 epochs
- Save best model based on validation mIoU

Best model weights will be saved as:

best_model_final_0.5402.pth

---

## Inference / Testing Instructions

To evaluate the trained model:

python test.py --weights best_model_final_0.5402.pth

This script will:

- Load trained weights
- Run inference on validation/test dataset
- Apply Test-Time Augmentation
- Compute mean IoU
- Print final performance metrics
- Optionally save predicted segmentation masks

---

## Configuration

Hyperparameters can be modified inside:

config.yaml

This includes:

- Learning rate
- Batch size
- Number of epochs
- Input resolution
- Scheduler settings

---

## Performance Summary

Final Validation mIoU: 0.5402

| Metric | Score |
|--------|-------|
| Validation mIoU | 0.5402 |
| Epochs | 30 |
| Backbone | ResNet-18 |


Performance Improvements Applied:

- Pretrained encoder backbone
- Class-weighted loss for imbalance handling
- Dice loss for boundary refinement
- Learning rate scheduler
- Test-Time Augmentation during inference

Test-Time Augmentation improved mIoU from:
0.5313 → 0.5402

---

## Reproducibility Notes

To reproduce results exactly:

1. Use the provided dataset only.
2. Maintain folder structure:

train/
Color_Images/
Segmentation/

val/
Color_Images/
Segmentation/

3. Use identical hyperparameters from config.yaml.
4. Run inference with Test-Time Augmentation enabled.

GPU training is recommended for faster convergence.

---

## Expected Output

After running test.py:

Console Output:

- Validation Loss
- Validation mIoU

Optional Output:

- Saved segmentation masks in output folder

---

## Known Limitations

- Small object classes (flowers, logs) show moderate confusion due to texture similarity.
- Ground clutter and rocks occasionally overlap in predictions.
- Performance may slightly vary depending on hardware and CUDA version.

---

## Future Improvements

- Multi-scale training
- Domain adaptation techniques
- Deeper encoder backbone (ResNet-34 or ResNet-50)
- Advanced TTA strategies

---

## Compliance Statement

This model was trained exclusively on the dataset provided for this challenge.  
No testing images were used during training.  
All workflows strictly maintained separation between training, validation, and testing sets in accordance with challenge rules.

---

## Author

Hack for Green Bharat 2026
Team Name: TechGoofies
Team Members:
Kripasree M
Madhu Rithika R K
Raj Moorthy B
