import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# Import your model builder
from src.desert_segmentation.models.unet import build_unet

# -------------------------
# CONFIG
# -------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
IMAGE_SIZE = 256
MODEL_PATH = Path("best_model.pth")

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

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Desert Semantic Segmentation",
    layout="wide"
)

st.title("Desert Semantic Segmentation")
st.markdown("Upload a desert terrain image to generate segmentation mask.")

# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_model():
    model = build_unet(num_classes=NUM_CLASSES, encoder_weights=None)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------
# UTILITIES
# -------------------------

def preprocess_image(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.tensor(image).unsqueeze(0).to(DEVICE)
    return tensor

def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

def overlay_mask(image, mask, alpha=0.5):
    mask_colored = colorize_mask(mask)
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay

# -------------------------
# FILE UPLOAD
# -------------------------

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    input_tensor = preprocess_image(image_np)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    overlay = overlay_mask(
        cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE)),
        prediction
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True)

    with col2:
        st.subheader("Segmentation Overlay")
        st.image(overlay, use_column_width=True)

    # Legend
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

st.markdown("---")
st.markdown("Built with PyTorch + Streamlit")