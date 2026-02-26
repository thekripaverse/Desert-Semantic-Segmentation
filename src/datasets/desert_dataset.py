import os
import cv2
import numpy as np
from torch.utils.data import Dataset

CLASS_MAPPING = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

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

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            return augmented["image"], augmented["mask"].long()

        return image, mask