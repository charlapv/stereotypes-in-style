import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch


def occlude_face(img: Image.Image, box, mode: str = "black") -> Image.Image:
    img = img.copy()
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.width, x2)
    y2 = min(img.height, y2)

    if x2 <= x1 or y2 <= y1:
        return img

    if mode == "black":
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    elif mode == "blur":
        face_crop = img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=18))
        img.paste(face_crop, (x1, y1))
    else:
        raise ValueError("mode must be 'black' or 'blur'")

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with image_path column")
    parser.add_argument("--input_root", required=True, help="Root folder of original images")
    parser.add_argument("--output_root", required=True, help="Root folder for occluded images")
    parser.add_argument("--mode", default="black", choices=["black", "blur"])
    parser.add_argument("--margin", type=float, default=0.15, help="Extra padding around detected face")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain image_path column")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device)

    missing = []
    no_face = []
    processed = 0

    for rel_path in tqdm(df["image_path"].tolist(), desc="Occluding faces"):
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
            # keep original if no face is detected
            img.save(dst)
            no_face.append(str(rel_path))
            processed += 1
            continue

        box = boxes[0]
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        x1 -= w * args.margin
        y1 -= h * args.margin
        x2 += w * args.margin
        y2 += h * args.margin

        img_occ = occlude_face(img, (x1, y1, x2, y2), mode=args.mode)
        img_occ.save(dst)
        processed += 1

    print(f"Processed: {processed}")
    print(f"Missing/unreadable: {len(missing)}")
    print(f"No face detected: {len(no_face)}")

    if missing:
        (output_root / "_missing_files.txt").write_text("\n".join(missing), encoding="utf-8")
    if no_face:
        (output_root / "_no_face_detected.txt").write_text("\n".join(no_face), encoding="utf-8")


if __name__ == "__main__":
    main()