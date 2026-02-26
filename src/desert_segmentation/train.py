import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.unet import build_unet
from datasets.desert_dataset import DesertDataset
from losses.segmentation_loss import SegmentationLoss
from utils.metrics import compute_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="val")
    return parser.parse_args()


def get_transforms(image_size: int):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.2),
        A.GaussianBlur(3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transform, val_transform = get_transforms(args.image_size)

    train_dataset = DesertDataset(
        f"{args.train_dir}/Color_Images",
        f"{args.train_dir}/Segmentation",
        train_transform
    )

    val_dataset = DesertDataset(
        f"{args.val_dir}/Color_Images",
        f"{args.val_dir}/Segmentation",
        val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_unet(args.num_classes).to(device)

    class_weights = torch.tensor([1.0,1.2,1.0,1.2,1.5,1.5,1.5,1.3,0.8,0.5]).to(device)
    criterion = SegmentationLoss(class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                total_iou += compute_iou(outputs, masks, args.num_classes)

        avg_iou = total_iou / len(val_loader)
        scheduler.step(avg_iou)

        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logging.info(f"Val Loss: {val_loss/len(val_loader):.4f}")
        logging.info(f"Val mIoU: {avg_iou:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "weights/best_model.pth")
            logging.info("Saved Best Model")

    logging.info("Training Complete")


if __name__ == "__main__":
    main()