from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from insightface.model_zoo import get_model


def main():
    # -------------------------
    # Paths
    # -------------------------
    image_dir = Path("/workspace/data/subset_60000_faces")
    labels_csv = Path("/workspace/data/subset_60000_faces_labels.csv")
    out_csv = Path("/workspace/arcface_embeddings_60k.csv")

    # -------------------------
    # Load labels
    # -------------------------
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    df = pd.read_csv(labels_csv)

    required_cols = {"image_path", "gender"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )

    print(f"Rows in labels CSV: {len(df)}")

    # -------------------------
    # Load ArcFace recognizer only
    # -------------------------
    model_path = "/root/.insightface/models/buffalo_l/w600k_r50.onnx"
    arcface = get_model(model_path)
    if arcface is None:
        raise RuntimeError(f"Failed to load ArcFace model from: {model_path}")

    # CPU mode to avoid current CUDA/cuDNN issue
    arcface.prepare(ctx_id=-1)

    # -------------------------
    # Extraction
    # -------------------------
    results = []
    missing = 0
    unreadable = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ArcFace embeddings"):
        rel_path = row["image_path"]
        gender = row["gender"]

        img_path = image_dir / rel_path
        if not img_path.exists():
            missing += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            unreadable += 1
            continue

        try:
            # Keep as HWC image. Resize to ArcFace expected size.
            img = cv2.resize(img, (112, 112))

            # ArcFace ONNX expects standard image arrays here.
            # Pass a list of images to get_feat.
            emb = arcface.get_feat([img])[0]
        except Exception:
            failed += 1
            continue

        record = {
            "image_path": rel_path,
            "gender": gender
        }

        for i, val in enumerate(emb):
            record[f"f{i}"] = float(val)

        results.append(record)

    # -------------------------
    # Save
    # -------------------------
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print("\n=== ArcFace Extraction Summary ===")
    print(f"Rows in labels CSV : {len(df)}")
    print(f"Saved embeddings   : {len(out_df)}")
    print(f"Missing files      : {missing}")
    print(f"Unreadable files   : {unreadable}")
    print(f"Failed processing  : {failed}")
    print(f"Saved to           : {out_csv}")

    if len(out_df) > 0:
        print("\nGender distribution in embeddings file:")
        print(out_df["gender"].value_counts())


if __name__ == "__main__":
    main()