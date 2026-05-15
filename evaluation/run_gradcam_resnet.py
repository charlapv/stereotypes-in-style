from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

print("Python:", sys.executable)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def build_resnet50_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_image_for_cam(image_path: Path):
    pil_img = Image.open(image_path).convert("RGB")
    rgb_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    return pil_img, rgb_np


def tensor_from_pil(pil_img: Image.Image, transform):
    return transform(pil_img).unsqueeze(0)


def predict(model, input_tensor, device):
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
    return pred_class, probs


def main():
    # -------------------------
    # CONFIG
    # -------------------------
    image_dir = Path("/workspace/data/subset_60000")
    test_csv = Path("/workspace/splits/richwear_metadata_60kcleantest.csv")
    checkpoint_path = "/workspace/runs/resnet50_60k_weighted/best_model.pt"
    out_dir = Path("/workspace/gradcam_outputs/resnet50_60k")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # LOAD MODEL
    # -------------------------
    model = build_resnet50_model(checkpoint_path, device)

    # For ResNet-50, a common Grad-CAM target layer is the last block of layer4
    target_layers = [model.layer4[-1]]

    transform = get_transform()
    df = pd.read_csv(test_csv)

    # pick a few examples
    sample_df = df.sample(n=min(12, len(df)), random_state=42).reset_index(drop=True)

    summary_rows = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for i, row in sample_df.iterrows():
            rel_path = row["image_path"]
            true_gender = str(row["gender"]).strip().lower()
            true_class = 0 if true_gender == "male" else 1

            img_path = image_dir / rel_path
            if not img_path.exists():
                print(f"Missing: {img_path}")
                continue

            pil_img, rgb_np = load_image_for_cam(img_path)
            input_tensor = tensor_from_pil(pil_img, transform)

            pred_class, probs = predict(model, input_tensor, device)

            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(
                input_tensor=input_tensor.to(device),
                targets=targets
            )[0]

            cam_image = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
            cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            true_name = "male" if true_class == 0 else "female"
            pred_name = "male" if pred_class == 0 else "female"

            out_name = f"{i:02d}_{Path(rel_path).stem}_true-{true_name}_pred-{pred_name}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), cam_bgr)

            summary_rows.append({
                "image_path": rel_path,
                "true_label": true_name,
                "pred_label": pred_name,
                "prob_male": float(probs[0]),
                "prob_female": float(probs[1]),
                "output_file": str(out_path)
            })

    pd.DataFrame(summary_rows).to_csv(out_dir / "gradcam_summary.csv", index=False)

    with open(out_dir / "gradcam_config.json", "w") as f:
        json.dump({
            "checkpoint_path": checkpoint_path,
            "image_dir": str(image_dir),
            "test_csv": str(test_csv),
            "num_examples": len(summary_rows),
            "target_layer": "model.layer4[-1]"
        }, f, indent=4)

    print(f"Saved Grad-CAM images to: {out_dir}")
    print(f"Saved summary to: {out_dir / 'gradcam_summary.csv'}")


if __name__ == "__main__":
    main()