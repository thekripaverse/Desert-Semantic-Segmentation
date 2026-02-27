"""
Streamlit UI for Desert Semantic Segmentation.

Features:
- Upload image interface
- Model inference
- Segmentation overlay visualization
- Class legend display

Designed for quick qualitative evaluation and demo purposes.
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys

# --------------------------------------------------
# PROJECT PATH CONFIGURATION
# --------------------------------------------------
# Add project root to Python path for local imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Import model builder
from src.desert_segmentation.models.unet import build_unet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Device selection (GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and inference parameters
NUM_CLASSES = 10
IMAGE_SIZE = 256
MODEL_PATH = Path("best_model.pth")

# Color mapping for visualization (class_id -> RGB)
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (128, 0, 0),
    2: (0, 128, 0),
    3: (128, 128, 0),
    4: (0, 0, 128),
    5: (128, 0, 128),
    6: (0, 128, 128),
    7: (128, 128, 128),
    8: (255, 165, 0),
    9: (0, 255, 255),
}

# Human-readable class names
CLASS_NAMES = {
    0: "Background",
    1: "Vegetation",
    2: "Tree",
    3: "Shrub",
    4: "Rock",
    5: "Sand",
    6: "Log",
    7: "Clutter",
    8: "Flower",
    9: "Other",
}


# --------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Desert Semantic Segmentation",
    layout="wide"
)

st.title("Desert Semantic Segmentation")
st.markdown("Upload a desert terrain image to generate segmentation mask.")


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    """
    Loads trained segmentation model from disk.
    Uses Streamlit resource caching to avoid reloading
    model on every UI interaction.
    """
    model = build_unet(num_classes=NUM_CLASSES, encoder_weights=None)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()
    return model


# Initialize model once
model = load_model()


# --------------------------------------------------
# IMAGE PROCESSING UTILITIES
# --------------------------------------------------

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocesses input image for model inference.

    Steps:
    - Resize to model resolution
    - Normalize to [0, 1]
    - Convert HWC -> CHW
    - Convert to tensor and add batch dimension
    """
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    tensor = torch.tensor(image).unsqueeze(0).to(DEVICE)
    return tensor


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Converts predicted class mask to RGB color mask
    using predefined CLASS_COLORS mapping.
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays segmentation mask on original image.

    Args:
        image: Original image
        mask: Predicted class mask
        alpha: Transparency factor

    Returns:
        Blended overlay image
    """
    mask_colored = colorize_mask(mask)
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay


# --------------------------------------------------
# FILE UPLOAD & INFERENCE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Preprocess input
    input_tensor = preprocess_image(image_np)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Generate overlay visualization
    overlay = overlay_mask(
        cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE)),
        prediction
    )

    # Display results side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True)

    with col2:
        st.subheader("Segmentation Overlay")
        st.image(overlay, use_column_width=True)

    # --------------------------------------------------
    # CLASS LEGEND DISPLAY
    # --------------------------------------------------
    st.markdown("### Class Legend")

    legend_cols = st.columns(5)

    for idx, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
        col = legend_cols[idx % 5]
        color = CLASS_COLORS[class_id]

        col.markdown(
            f"<div style='display:flex;align-items:center;'>"
            f"<div style='width:15px;height:15px;background-color:rgb{color};"
            f"margin-right:8px;'></div>"
            f"{class_name}</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown("Built with PyTorch + Streamlit")
