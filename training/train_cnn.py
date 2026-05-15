import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import classification_report, confusion_matrix


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Dataset
# -------------------------
class RichWearDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform

        required_cols = {"image_path", "gender"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"CSV must contain columns {required_cols}. "
                f"Found: {self.df.columns.tolist()}"
            )

        self.label_map = {"male": 0, "female": 1}

        genders = self.df["gender"].astype(str).str.strip().str.lower()
        invalid = ~genders.isin(self.label_map.keys())
        if invalid.any():
            bad = self.df.loc[invalid, "gender"].unique().tolist()
            raise ValueError(f"Found unmapped gender labels: {bad}")

        self.df["label"] = genders.map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.image_dir / row["image_path"]
        label = int(row["label"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}") from e

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------
# Class weights
# -------------------------
def compute_class_weights_from_train_csv(train_csv: str) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["gender"].astype(str).str.strip().str.lower().value_counts()

    if "male" not in counts or "female" not in counts:
        raise ValueError(
            f"Training split must contain both classes. Found counts: {counts.to_dict()}"
        )

    total = counts.sum()
    w_male = float(total / counts["male"])
    w_female = float(total / counts["female"])

    return torch.tensor([w_male, w_female], dtype=torch.float32)


# -------------------------
# Training
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc="Training", total=len(loader)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / total
    acc = correct / total

    return avg_loss, acc


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, loader, device):
    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", total=len(loader)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_y.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(all_y, all_pred)
    report = classification_report(
        all_y,
        all_pred,
        target_names=["male", "female"],
        digits=4,
        zero_division=0
    )
    acc = float(np.mean(np.array(all_y) == np.array(all_pred)))

    return acc, report, cm


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    parser.add_argument("--val_csv", required=True, help="Path to validation CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV")
    parser.add_argument("--image_dir", required=True, help="Root folder containing image subfolders")

    parser.add_argument("--outdir", default="runs/resnet50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Backbone model to use"
    )

    args = parser.parse_args()
    

    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # save config
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # datasets
    train_ds = RichWearDataset(args.train_csv, args.image_dir, transform=train_tf)
    val_ds = RichWearDataset(args.val_csv, args.image_dir, transform=eval_tf)
    test_ds = RichWearDataset(args.test_csv, args.image_dir, transform=eval_tf)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # model: ResNet50
    #model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    #model.fc = nn.Linear(model.fc.in_features, 2)
    #model = model.to(device)

    # model selection
    if args.model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif args.model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    model = model.to(device)
    print(f"Using model: {args.model_name}")

    # loss
    if args.weighted_loss:
        class_weights = compute_class_weights_from_train_csv(args.train_csv).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted loss:", class_weights.detach().cpu().numpy().tolist())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_model_path = outdir / "best_model.pt"
    history = []

    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_acc, val_report, val_cm = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs} - {elapsed/60:.2f} min")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train acc : {train_acc:.4f}")
        print(f"Val acc   : {val_acc:.4f}")
        print("Val confusion matrix:\n", val_cm)
        print("Val classification report:\n", val_report)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch_time_sec": elapsed
        })

        pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args)
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")

    # final test
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_report, test_cm = evaluate(model, test_loader, device)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test acc: {test_acc:.4f}")
    print("Test confusion matrix:\n", test_cm)
    print("Test classification report:\n", test_report)

    (outdir / "test_accuracy.txt").write_text(f"{test_acc:.6f}\n")
    (outdir / "test_confusion_matrix.txt").write_text(str(test_cm))
    (outdir / "test_report.txt").write_text(test_report)

    print(f"Saved all outputs to: {outdir}")


if __name__ == "__main__":
    main()