import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_MAPPING = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

NUM_CLASSES = 10
IMAGE_SIZE = 256

class DesertDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, -1)

        new_mask = np.zeros_like(mask)
        for raw_val, new_val in CLASS_MAPPING.items():
            new_mask[mask == raw_val] = new_val

        mask = new_mask

        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()

def compute_iou(preds, masks):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(NUM_CLASSES):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)
        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious)

def tta_predict(model, images):
    outputs1 = model(images)
    flipped = torch.flip(images, dims=[3])
    outputs2 = model(flipped)
    outputs2 = torch.flip(outputs2, dims=[3])
    return (outputs1 + outputs2) / 2

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_dataset = DesertDataset("val/Color_Images", "val/Segmentation", transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
).to(DEVICE)

model.load_state_dict(torch.load("best_model_final_0.5402.pth", map_location=DEVICE))
model.eval()

total_iou = 0
with torch.no_grad():
    for images, masks in tqdm(val_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = tta_predict(model, images)
        total_iou += compute_iou(outputs, masks)

print("Final TTA mIoU:", total_iou / len(val_loader))