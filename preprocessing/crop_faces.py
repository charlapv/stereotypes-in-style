import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch


def choose_largest_box(boxes):
    if len(boxes) == 1:
        return boxes[0]
    areas = []
    for b in boxes:
        x1, y1, x2, y2 = b
        areas.append((x2 - x1) * (y2 - y1))
    return boxes[int(torch.tensor(areas).argmax().item())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain image_path column")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device)

    kept_rows = []
    missing = []
    no_face = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping faces"):
        rel_path = row["image_path"]
        src = input_root / rel_path
        dst = output_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            missing.append(str(src))
            continue

        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            missing.append(str(src))
            continue

        boxes, probs = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            no_face.append(rel_path)
            continue

        box = choose_largest_box(boxes)
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        x1 = max(0, int(x1 - w * args.margin))
        y1 = max(0, int(y1 - h * args.margin))
        x2 = min(img.width, int(x2 + w * args.margin))
        y2 = min(img.height, int(y2 + h * args.margin))

        if x2 <= x1 or y2 <= y1:
            no_face.append(rel_path)
            continue

        face = img.crop((x1, y1, x2, y2))
        face.save(dst)

        kept_rows.append(row)

    kept_df = pd.DataFrame(kept_rows)

    filtered_csv = output_root.parent / f"{output_root.name}_labels.csv"
    kept_df.to_csv(filtered_csv, index=False)

    print(f"Saved face crops: {len(kept_df)}")
    print(f"Missing/unreadable: {len(missing)}")
    print(f"No face detected: {len(no_face)}")
    print(f"Filtered CSV saved to: {filtered_csv}")

    if missing:
        (output_root / "_missing_files.txt").write_text("\n".join(missing), encoding="utf-8")
    if no_face:
        (output_root / "_no_face_detected.txt").write_text("\n".join(no_face), encoding="utf-8")


if __name__ == "__main__":
    main()