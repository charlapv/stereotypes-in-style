from pathlib import Path
import json
import sys
import random

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
        confidence = float(np.max(probs))
    return pred_class, probs, confidence


def label_to_class(gender_value: str) -> int:
    gender_value = str(gender_value).strip().lower()
    return 0 if gender_value == "male" else 1


def class_to_name(class_id: int) -> str:
    return "male" if class_id == 0 else "female"


def main():
    # -------------------------
    # CONFIG
    # -------------------------
    image_dir = Path("/workspace/data/subset_60000")
    test_csv = Path("/workspace/splits/richwear_metadata_60kcleantest.csv")
    checkpoint_path = "/workspace/runs/resnet50_60k_weighted/best_model.pt"
    out_dir = Path("/workspace/gradcam_outputs/resnet50_60k_misclassified")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_examples_total = 12
    num_female_to_male = 6   # true female, predicted male
    num_male_to_female = 6   # true male, predicted female
    random_seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # LOAD MODEL
    # -------------------------
    model = build_resnet50_model(checkpoint_path, device)
    target_layers = [model.layer4[-1]]
    transform = get_transform()

    df = pd.read_csv(test_csv)
    print("Rows in test CSV:", len(df))

    # -------------------------
    # FIRST PASS: FIND MISCLASSIFICATIONS
    # -------------------------
    all_results = []

    for idx, row in df.iterrows():
        rel_path = row["image_path"]
        true_class = label_to_class(row["gender"])
        img_path = image_dir / rel_path

        if not img_path.exists():
            continue

        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = tensor_from_pil(pil_img, transform)

        pred_class, probs, confidence = predict(model, input_tensor, device)

        all_results.append({
            "index": idx,
            "image_path": rel_path,
            "true_class": true_class,
            "pred_class": pred_class,
            "true_label": class_to_name(true_class),
            "pred_label": class_to_name(pred_class),
            "prob_male": float(probs[0]),
            "prob_female": float(probs[1]),
            "confidence": confidence,
            "correct": int(true_class == pred_class)
        })

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "all_predictions.csv", index=False)

    misclassified_df = results_df[results_df["correct"] == 0].copy()
    misclassified_df.to_csv(out_dir / "misclassified_only.csv", index=False)

    female_to_male_df = misclassified_df[
        (misclassified_df["true_class"] == 1) & (misclassified_df["pred_class"] == 0)
    ].copy()

    male_to_female_df = misclassified_df[
        (misclassified_df["true_class"] == 0) & (misclassified_df["pred_class"] == 1)
    ].copy()

    print("Total misclassified:", len(misclassified_df))
    print("Female -> Male:", len(female_to_male_df))
    print("Male -> Female:", len(male_to_female_df))

    # -------------------------
    # SELECT EXAMPLES
    # -------------------------
    random.seed(random_seed)

    # Option 1: random sample
    f2m_sample = female_to_male_df.sample(
        n=min(num_female_to_male, len(female_to_male_df)),
        random_state=random_seed
    ) if len(female_to_male_df) > 0 else pd.DataFrame()

    m2f_sample = male_to_female_df.sample(
        n=min(num_male_to_female, len(male_to_female_df)),
        random_state=random_seed
    ) if len(male_to_female_df) > 0 else pd.DataFrame()

    selected_df = pd.concat([f2m_sample, m2f_sample], ignore_index=True)

    # If fewer than requested, top up with any remaining misclassified examples
    if len(selected_df) < num_examples_total and len(misclassified_df) > len(selected_df):
        already_selected = set(selected_df["image_path"].tolist()) if len(selected_df) > 0 else set()
        remaining_df = misclassified_df[~misclassified_df["image_path"].isin(already_selected)]

        extra_needed = num_examples_total - len(selected_df)
        extra_sample = remaining_df.sample(
            n=min(extra_needed, len(remaining_df)),
            random_state=random_seed
        ) if len(remaining_df) > 0 else pd.DataFrame()

        selected_df = pd.concat([selected_df, extra_sample], ignore_index=True)

    selected_df.to_csv(out_dir / "selected_misclassified_for_gradcam.csv", index=False)
    print("Selected for Grad-CAM:", len(selected_df))

    # -------------------------
    # SECOND PASS: GENERATE GRAD-CAM FOR SELECTED ERRORS
    # -------------------------
    summary_rows = []

    with GradCAM(model=model, target_layers=target_layers) as cam:
        for i, row in selected_df.iterrows():
            rel_path = row["image_path"]
            img_path = image_dir / rel_path

            if not img_path.exists():
                print(f"Missing: {img_path}")
                continue

            pil_img, rgb_np = load_image_for_cam(img_path)
            input_tensor = tensor_from_pil(pil_img, transform)

            pred_class = int(row["pred_class"])
            true_class = int(row["true_class"])

            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = cam(
                input_tensor=input_tensor.to(device),
                targets=targets
            )[0]

            cam_image = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
            cam_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            true_name = class_to_name(true_class)
            pred_name = class_to_name(pred_class)

            out_name = f"{i:02d}_{Path(rel_path).stem}_true-{true_name}_pred-{pred_name}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), cam_bgr)

            summary_rows.append({
                "image_path": rel_path,
                "true_label": true_name,
                "pred_label": pred_name,
                "prob_male": float(row["prob_male"]),
                "prob_female": float(row["prob_female"]),
                "confidence": float(row["confidence"]),
                "output_file": str(out_path)
            })

    pd.DataFrame(summary_rows).to_csv(out_dir / "gradcam_summary.csv", index=False)

    with open(out_dir / "gradcam_config.json", "w") as f:
        json.dump({
            "checkpoint_path": checkpoint_path,
            "image_dir": str(image_dir),
            "test_csv": str(test_csv),
            "num_examples_requested": num_examples_total,
            "num_examples_generated": len(summary_rows),
            "female_to_male_requested": num_female_to_male,
            "male_to_female_requested": num_male_to_female,
            "target_layer": "model.layer4[-1]"
        }, f, indent=4)

    print(f"Saved Grad-CAM images to: {out_dir}")
    print(f"Saved summary to: {out_dir / 'gradcam_summary.csv'}")
    print(f"Saved all predictions to: {out_dir / 'all_predictions.csv'}")
    print(f"Saved all misclassifications to: {out_dir / 'misclassified_only.csv'}")


if __name__ == "__main__":
    main()