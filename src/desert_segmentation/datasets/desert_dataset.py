import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List

# Mapping raw pixel values to class indices for segmentation
CLASS_MAPPING: Dict[int, int] = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

class DesertDataset(Dataset):
    """
    Dataset class for Desert Semantic Segmentation.
    Handles image loading, mask value mapping, and augmentations.
    """
    def __init__(self, image_dir: str, mask_dir: str, transform: Optional[object] = None):
        """
        Args:
            image_dir: Path to the folder containing RGB images.
            mask_dir: Path to the folder containing segmentation masks.
            transform: Albumentations transform pipeline.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (image, mask) as tensors.
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Load image and convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (-1 flag for 16-bit or unchanged data)
        mask = cv2.imread(mask_path, -1)

        # Map raw category values to 0-9 indices
        new_mask = np.zeros_like(mask)
        for raw_val, new_val in CLASS_MAPPING.items():
            new_mask[mask == raw_val] = new_val

        mask = new_mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"].long()

        # Convert to tensor if no transform is provided
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor