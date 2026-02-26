import torch
from src.desert_segmentation.models.unet import build_unet

model = build_unet()
dummy = torch.randn(1,3,256,256)
torch.onnx.export(model, dummy, "model.onnx")